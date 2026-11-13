# -*- coding: utf-8 -*-
"""
manifest — what was drawn, where. The picture explains itself in plain text.

Every top-level draw call of the kits (a tree, a campfire, a 3D mug, a diagram node, a chart, a drafting view,
a texture fill, a text label) adds one entry to ``cv.manifest``: the kind, the name a person would use, where it
is, how much of the canvas it covers, the paint order and a small coverage mask (at most 256 px on the long side).
From that the library can say, without looking at pixels:

>>> from pixelcanvas import Canvas, manifest
>>> cv = Canvas(640, 360)
>>> X, Y = cv.xy()
>>> cv.paint("#c84", ((X - 200) ** 2 + (Y - 200) ** 2 < 60 ** 2).astype("float32"))
>>> e = manifest.note(cv, "object", "ball", bbox=(140, 140, 260, 260))
>>> manifest.summary(cv)[0]["name"], manifest.relations(cv)           # positions, sizes, visibility
>>> manifest.check(cv, prompt='a red ball with the word "Hello"')      # structured issues

* ``note(cv, kind, name, mask=None, bbox=None, **meta)`` — record a hand-drawn object (the kits do it for you).
* ``summary(cv)`` — one dict per entry: name, kind, centre, size, area %, visible %, clipped %, hidden_by, step.
* ``relations(cv)`` — short statements: "cat is left of dog", "mug overlaps bottle (12%)".
* ``check(cv, prompt=None)`` — issues in the critique format ``{rule, severity, message, where, value, threshold,
  fix}``: off-canvas, hidden, tiny, stacked duplicates, touching the edge, and — with a prompt — quoted text
  that was not drawn, a named style that was not applied, and counts that do not match.
* ``infer_kind(cv)`` / ``relax(warnings, kind)`` — painting / design / diagram / drawing, and the critique rules
  that do not apply to it.

Text drawn with ``cv.text`` is listed from ``cv.text_log`` (the manifest only keeps a reference), and
``stylize.apply`` records the style in ``cv.styles_applied``.
"""
from __future__ import annotations

import difflib
import functools
import inspect
import math
import os
import re

import numpy as np

F32 = np.float32
GRID = 256                         # stored masks: at most this many cells on the long side
_OWN = 26                          # uint8 coverage that counts as part of an object (alpha >= 0.1)
_OCCLUDE = 128                     # coverage that hides what is below (alpha >= 0.5)
_CONTAINER = {"view", "group", "diagram", "drawing", "style", "background"}   # never hide other entries
_SEE_THROUGH = {"effect"}                                   # light, haze, rain: never counted as hiding things
CUE = "cue"       # what a renderer records so COVERAGE can answer lighting / shadows / reflections words:
#                   not an object, so summary(), counts, relations and the object rules never see it
_SUBJECT = {"plant", "creature", "person", "object", "node", "chart", "shape"}
_DETAIL = {"grass", "moss", "pebbles", "reeds", "leaf", "falling leaves", "leaf pile", "ivy", "wheat", "cattails",
           "stone", "stones", "lily pads", "meadow"}                            # scattered detail, not subjects
# things drawn as a scatter or a layer, often in several calls (far and near): never "stacked duplicates"
_SCATTER = {"forest", "trees", "grass", "meadow", "rain", "snow", "stars", "clouds", "cloud", "fog", "mist", "haze",
            "flowers", "bushes", "shrubs", "rocks", "stones", "pebbles", "leaves", "falling leaves", "reeds", "waves",
            "hills", "mountains", "mountain", "sparks", "embers", "bubbles", "crowd", "city", "skyline", "buildings",
            "lights", "fireflies", "confetti", "particles", "dust", "petals", "foliage", "hedge", "fence"}
_STATE = {"depth": 0, "step": None, "on": os.environ.get("PIXELCANVAS_MANIFEST", "1") not in ("0", "off", "no")}

__all__ = ["note", "entries", "summary", "relations", "check", "object_checks", "prompt_checks", "infer_kind",
           "relax", "drawn_text", "step", "reset", "drawn", "register", "table", "PAINTERLY_RULES", "META_KEYS",
           "coverage", "count_phrases", "prompt_status", "quoted_status", "reconcile_cropped", "prompt_relax",
           "pasted", "is_backdrop", "is_see_through", "note_raw", "extent_from_args", "CUE", "colour_checks",
           "colour_requests", "colour_name", "place_checks", "place_requests", "request_summary", "COLOUR_NAMES"]


# --------------------------------------------------------------------------------------------- geometry helpers

def _factor(H, W):
    return max(1, int(math.ceil(max(H, W) / float(GRID))))


def _small_shape(H, W, f):
    return (int(math.ceil(H / f)), int(math.ceil(W / f)))


def _bbox(m):
    rows = np.flatnonzero(m.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(m.any(0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _shrink(crop, x0, y0, H, W, f):
    """Coverage crop (h, w) at canvas offset (x0, y0) -> uint8 grid of the whole canvas (block mean)."""
    sh, sw = _small_shape(H, W, f)
    out = np.zeros((sh, sw), np.uint8)
    h, w = crop.shape
    if h == 0 or w == 0:
        return out
    ax0, ay0 = (x0 // f) * f, (y0 // f) * f
    ax1, ay1 = int(math.ceil((x0 + w) / f)) * f, int(math.ceil((y0 + h) / f)) * f
    pad = np.zeros((ay1 - ay0, ax1 - ax0), F32)
    pad[y0 - ay0:y0 - ay0 + h, x0 - ax0:x0 - ax0 + w] = crop
    blk = pad.reshape((ay1 - ay0) // f, f, (ax1 - ax0) // f, f).mean((1, 3))
    gy0, gx0 = ay0 // f, ax0 // f
    gy1, gx1 = min(sh, gy0 + blk.shape[0]), min(sw, gx0 + blk.shape[1])
    out[gy0:gy1, gx0:gx1] = np.round(np.clip(blk[:gy1 - gy0, :gx1 - gx0], 0, 1) * 255).astype(np.uint8)
    return out


def _thumb(cv):
    """Small point-sampled copy of the canvas on the manifest grid (for 'what changed' when nothing was painted)."""
    H, W = cv.h, cv.w
    f = _factor(H, W)
    sh, sw = _small_shape(H, W, f)
    t = np.clip(np.asarray(cv.buf)[f // 2::f, f // 2::f], 0, 1).astype(F32)
    if t.shape[:2] != (sh, sw):
        t = np.pad(t, ((0, max(0, sh - t.shape[0])), (0, max(0, sw - t.shape[1])), (0, 0)), mode="edge")[:sh, :sw]
    return t


def _frac_box(b, W, H):
    """A box in pixels or fractions -> pixel floats (x0, y0, x1, y1). Fractions when every value is within 1.5."""
    b = [float(v) for v in b]
    if max(abs(v) for v in b) <= 1.5:
        return b[0] * W, b[1] * H, b[2] * W, b[3] * H
    return tuple(b)


def _r(v, n=4):
    return round(float(v), n)


# --------------------------------------------------------------------------------------------- capture

class _Capture:
    """Coverage of every paint / paste into one canvas while a kit call runs (fed by Canvas.paint)."""

    def __init__(self, cv):
        self.cv = cv
        self.cov = None
        self.tiles = []                      # (name, x0, y0, alpha) from vector.Drawing.draw
        self.children = []                   # entries noted by inner calls (e.g. one per 3D object)
        self.before = _thumb(cv) if cv is not None else None

    def add(self, cv, a, region):
        if self.cv is None:
            self.cv = cv
        if cv is not self.cv:
            return
        H, W = cv.h, cv.w
        if self.cov is None or self.cov.shape != (H, W):
            self.cov = np.zeros((H, W), F32)
        x0, y0, x1, y1 = region if region is not None else (0, 0, W, H)
        tgt = self.cov[y0:y1, x0:x1]
        try:
            a = np.asarray(a, F32)
            if a.ndim == 3:
                a = a.max(-1)
            if a.ndim == 0:
                v = float(a)
                if v > 0:
                    np.maximum(tgt, min(v, 1.0), out=tgt)
                return
            np.maximum(tgt, np.clip(np.broadcast_to(a, tgt.shape), 0, 1), out=tgt)
        except (ValueError, TypeError):
            pass

    def mask(self):
        """(full coverage or None, crop tuple or None)."""
        if self.cov is not None and float(self.cov.max()) > 0.02:
            return self.cov
        if self.cv is None or self.before is None:
            return None
        now = _thumb(self.cv)
        if now.shape != self.before.shape:
            return None
        d = np.abs(now - self.before).max(-1) > 2.0 / 255
        if not d.any():
            return None
        return ("grid", d)


def _active():
    from . import canvas as _canvas
    return _canvas._CAPTURE[0] if _canvas._CAPTURE else None


def tile(cv, name, x0, y0, alpha):
    """Called by ``vector.Drawing.draw`` for each pasted group tile (named coverage for diagrams and charts)."""
    cap = _active()
    if cap is None:
        return
    if cap.cv is None:
        cap.cv = cv
    if cap.cv is cv:
        cap.tiles.append((str(name), int(x0), int(y0), np.asarray(alpha, F32)))


def drawing_drawn(cv, dwg, box=None):
    """Called by ``vector.Drawing.draw``: remember (by reference) which drawings were painted on this canvas."""
    try:
        lst = cv.__dict__.setdefault("_drawings", [])
        if dwg not in lst:
            lst.append(dwg)
        dwg._pc_drawn = (None if box is None else tuple(box), (cv.w, cv.h))
    except Exception:                                    # noqa: BLE001
        pass


def dominant_share(img, tol=0.045):
    """Share of the pixels within ``tol`` (per channel, 0..1) of the most common colour: 1.0 = a blank page."""
    a = np.asarray(img, F32)
    if a.ndim != 3 or not a.size:
        return 0.0
    step = max(1, int(math.ceil(max(a.shape[:2]) / 320.0)))
    a = np.clip(a[::step, ::step, :3], 0, 1).reshape(-1, 3)
    q = np.minimum((a * 16).astype(np.int32), 15)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    top = int(np.bincount(key, minlength=4096).argmax())
    sel = a[key == top]
    mode = np.median(sel, axis=0)
    return float((np.abs(a - mode).max(-1) <= tol).mean())


def style_applied(cv, name, before=None, params=None, notes=None):
    """Called by ``stylize.apply``: record the style on the canvas (``cv.styles_applied``) and, in
    ``cv.style_log``, its options, clamping notes and how uniform the picture was before and after."""
    if hasattr(cv, "buf"):
        try:
            cv.__dict__.setdefault("styles_applied", []).append(str(name))
            rec = {"style": str(name), "params": {k: v for k, v in (params or {}).items()
                                                   if isinstance(v, (str, int, float, bool, type(None)))},
                   "notes": list(notes or [])}
            if before is not None:
                rec["uniform_before"] = round(dominant_share(before), 4)
                rec["uniform_after"] = round(dominant_share(cv.buf), 4)
                b = np.clip(np.asarray(before, F32)[..., :3], 0, 1)
                k = max(1, int(math.ceil(b.shape[1] / 960.0)))
                rec["_before"] = (b[::k, ::k] * 255 + 0.5).astype(np.uint8)   # composition is judged on this
            cv.__dict__.setdefault("style_log", []).append(rec)
        except Exception:                                # noqa: BLE001
            pass


def pasted(dst, src, x=0, y=0, mask=None):
    """Called by ``Canvas.paste`` when a Canvas is pasted: its recorded entries (a creature drawn on a sticker
    sub-canvas, a 3D object rendered apart) move to ``dst`` at (x, y), cut to the paste ``mask``; its applied
    styles count for ``dst`` too. Returns the number of entries carried over."""
    if not _STATE["on"] or not hasattr(dst, "buf") or not hasattr(src, "buf") or dst is src:
        return 0
    n = 0
    try:
        for s in getattr(src, "styles_applied", []) or []:
            lst = dst.__dict__.setdefault("styles_applied", [])
            if s not in lst:
                lst.append(s)
        E = [e for e in entries(src) if e["kind"] not in ("text", "style")]
        if not E:
            return 0
        sh_, sw_ = src.h, src.w
        pm = None if mask is None else np.clip(np.asarray(mask, F32), 0, 1)
        if pm is not None and pm.ndim == 3:
            pm = pm.max(-1)
        for e in E:
            f = e["_f"]
            full = np.kron(e["_mask"].astype(F32) / 255.0, np.ones((f, f), F32))[:sh_, :sw_]
            if pm is not None and pm.shape == full.shape:
                full = full * pm
            m = np.zeros((dst.h, dst.w), F32)
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + sw_, dst.w), min(y + sh_, dst.h)
            if x1 <= x0 or y1 <= y0:
                continue
            m[y0:y1, x0:x1] = full[y0 - y:y1 - y, x0 - x:x1 - x]
            if _note_raw(dst, e["kind"], e["name"], mask=m, alias=e.get("aliases"), **(e.get("meta") or {})):
                n += 1
    except Exception:                                    # noqa: BLE001 - recording never breaks a paste
        if os.environ.get("PIXELCANVAS_DEBUG"):
            raise
    return n


def drawn_text(cv, index):
    """Called by ``pixelcanvas.text`` after logging a text draw: add a reference entry (no copy of the data)."""
    if not _STATE["on"] or _STATE["depth"] or not hasattr(cv, "buf"):
        return
    try:
        cv.manifest.append({"kind": "text", "ref": int(index), "order": _seq(cv), "step": _STATE["step"]})
    except Exception:                                    # noqa: BLE001
        pass


def step(name):
    """The scene's current ``log("step")`` name (render.py calls this); stored on every entry."""
    _STATE["step"] = None if name is None else str(name)


def reset():
    """Forget the current step (render.py calls this before a scene runs)."""
    _STATE["step"] = None


def _seq(cv):
    n = cv.__dict__.get("_manifest_seq", 0)
    cv.__dict__["_manifest_seq"] = n + 1
    return n


# --------------------------------------------------------------------------------------------- note

def _make(cv, kind, name, mask=None, bbox=None, extent=None, alias=None, crop=None, meta=None):
    H, W = cv.h, cv.w
    f = _factor(H, W)
    sh, sw = _small_shape(H, W, f)
    small, area, box = None, 0.0, None
    if isinstance(mask, tuple) and len(mask) == 2 and mask[0] == "grid":        # changed cells of the grid
        g = mask[1]
        small = np.where(g, 255, 0).astype(np.uint8)
        bb = _bbox(g)
        box = (bb[2] * f, bb[0] * f, min(W, bb[3] * f), min(H, bb[1] * f))
        area = float(g.mean())
    elif crop is not None:
        arr, x0, y0 = crop
        arr = np.clip(np.asarray(arr, F32), 0, 1)
        bb = _bbox(arr > 0.1)
        if bb is None and bbox is None:
            return None
        if bb is not None:
            arr = arr[bb[0]:bb[1], bb[2]:bb[3]]
            x0, y0 = x0 + bb[2], y0 + bb[0]
            small = _shrink(arr, x0, y0, H, W, f)
            area = float(arr.sum()) / (H * W)
            box = (x0, y0, x0 + arr.shape[1], y0 + arr.shape[0])
    elif mask is not None:
        m = np.asarray(mask, F32)
        if m.ndim == 3:
            m = m.max(-1)
        if m.shape != (H, W):
            raise ValueError(f"mask shape {m.shape} does not match the canvas ({H}, {W})")
        bb = _bbox(m > 0.1)
        if bb is None and bbox is None:
            return None
        if bb is not None:
            y0, y1, x0, x1 = bb
            sub = np.clip(m[y0:y1, x0:x1], 0, 1)
            small = _shrink(sub, x0, y0, H, W, f)
            area = float(sub.sum()) / (H * W)
            box = (x0, y0, x1, y1)
    rect = False
    if small is None:
        if bbox is None:
            return None
        rect = True                                      # a box only: no mask to read colours from
        bx0, by0, bx1, by1 = _frac_box(bbox, W, H)
        cx0, cy0, cx1, cy1 = max(0.0, bx0), max(0.0, by0), min(float(W), bx1), min(float(H), by1)
        small = np.zeros((sh, sw), np.uint8)
        if cx1 > cx0 and cy1 > cy0:
            small[int(cy0 // f):int(math.ceil(cy1 / f)), int(cx0 // f):int(math.ceil(cx1 / f))] = 255
            area = (cx1 - cx0) * (cy1 - cy0) / (W * H)
            box = (cx0, cy0, cx1, cy1)
        else:
            box = (min(max(bx0, 0), W), min(max(by0, 0), H), min(max(bx0, 0), W), min(max(by0, 0), H))
        if extent is None:
            extent = (bx0, by0, bx1, by1)
    ext = list(box)
    for e in (extent, bbox):
        if e is not None:
            ex = _frac_box(e, W, H)
            ext = [min(ext[0], ex[0]), min(ext[1], ex[1]), max(ext[2], ex[2]), max(ext[3], ex[3])]
            break
    aliases = [alias] if isinstance(alias, str) else list(alias or [])
    return {"kind": str(kind), "name": str(name), "order": _seq(cv), "step": _STATE["step"],
            "bbox": [_r(box[0] / W), _r(box[1] / H), _r(box[2] / W), _r(box[3] / H)],
            "extent": [_r(ext[0] / W), _r(ext[1] / H), _r(ext[2] / W), _r(ext[3] / H)],
            "area": _r(area, 6), "aliases": aliases, "meta": dict(meta or {}), "_mask": small, "_f": f,
            "_rect": rect}


META_KEYS = ("count", "backdrop", "see_through")          # meta keys the report gives meaning to


def _check_meta(meta):
    """Validate the meta keys the report reads: a near-miss spelling of one of them raises a did-you-mean
    TypeError (other keys are free-form notes); ``count`` is a positive int, ``backdrop`` / ``see_through`` bool."""
    from ._args import check_choice, reject_unknown
    for k in list(meta):
        if k in META_KEYS:
            continue
        flat = k.lower().replace("_", "").replace("-", "")
        near = difflib.get_close_matches(k, META_KEYS, n=1, cutoff=0.75) or \
            [m for m in META_KEYS if m.replace("_", "") == flat]
        if near:
            reject_unknown("manifest.note", {k: meta[k]}, META_KEYS)
    if "count" in meta and meta["count"] is not None:
        c = meta["count"]
        if isinstance(c, bool) or not isinstance(c, (int, np.integer)) or int(c) < 0:
            raise TypeError(f"manifest.note: count={c!r} must be a whole number >= 0 (how many of the named thing "
                            "this entry stands for, e.g. count=12 for 12 monthly points)")
        meta["count"] = int(c)
    for k in ("backdrop", "see_through"):
        if k in meta:
            check_choice("manifest.note", k, meta[k], (True, False))
    return meta


def _same_thing(a, b):
    """Two names for the same thing: equal, one an alias of the other, or species and family word."""
    def forms(e):
        out = set()
        for w in [e["name"].split(" #")[0]] + list(e.get("aliases") or []):
            w = str(w).lower().strip()
            if not w:
                continue
            out.add(w)
            out.update(_singular(w.split()[-1]))
            for c in _singular(w.split()[-1]):
                if c in _FAMILY:
                    out.add(_FAMILY[c])
        return out
    return bool(forms(a) & forms(b))


def _duplicate_of(cv, e):
    """An earlier entry that ``e`` (a hand-written note) re-records: same place and the same or an alias name."""
    m = e["_mask"] >= _OWN
    n = int(m.sum())
    if not n:
        return None
    for old in reversed(list(getattr(cv, "manifest", []) or [])):
        if old.get("kind") == "text" or old.get("_mask") is None or old["_mask"].shape != e["_mask"].shape:
            continue
        if not _same_thing(old, e):
            continue
        o = old["_mask"] >= _OWN
        no = int(o.sum())
        inter = int((m & o).sum())
        if no and inter >= 0.6 * min(n, no) and inter >= 0.35 * max(n, no):
            return old
    return None


def note(cv, kind, name, mask=None, bbox=None, extent=None, alias=None, **meta):
    """Record that ``name`` (a ``kind``: plant, creature, person, effect, object, node, chart, view, text,
    texture ...) was drawn on ``cv``. Give its coverage ``mask`` (H, W) or a ``bbox`` (x0, y0, x1, y1 in pixels
    or fractions). ``extent``: the intended box when part of it lies off the canvas; ``alias``: other words for
    it ("tree" for an oak); ``meta``: anything else worth reporting. Returns the entry (None if nothing drawn).

    Meta keys the report reads: ``count=12`` (this entry stands for 12 of the named thing: a line of 12 monthly
    points is ``note(cv, "chart", "month", count=12)``), ``backdrop=True`` (a background area: never the subject,
    never cropped, hides nothing), ``see_through=True`` (glass, a water surface, a region outline: does not hide
    what lies under it). ``kind="style"`` records a look drawn by hand (``note(cv, "style", "sticker")``) so the
    request's named style counts as applied. Noting something a kit already recorded (same place, same or alias
    name) does not add a second entry: the kit's entry takes your name and keeps its own as an alias."""
    if not _STATE["on"]:
        return None
    meta = _check_meta(dict(meta))
    if str(kind) == "style":
        try:
            lst = cv.__dict__.setdefault("styles_applied", [])
            if str(name) not in lst:
                lst.append(str(name))
        except Exception:                                # noqa: BLE001
            pass
        if mask is None and bbox is None:
            bbox = (0, 0, 1, 1)
    return _note_raw(cv, kind, name, mask=mask, bbox=bbox, extent=extent, alias=alias, dedupe=True, **meta)


def _note_raw(cv, kind, name, mask=None, bbox=None, extent=None, alias=None, dedupe=False, **meta):
    e = _make(cv, kind, name, mask=mask, bbox=bbox, extent=extent, alias=alias, meta=meta)
    if e is None:
        return None
    cap = _active() if _STATE["depth"] else None
    if cap is not None and (cap.cv is None or cap.cv is cv):
        cap.children.append(e)
        return e
    if dedupe and e["kind"] not in ("style", "text"):
        old = _duplicate_of(cv, e)
        if old is not None:                              # the kit already recorded it: keep one entry
            if old["name"].lower() != e["name"].lower():
                al = [a for a in [old["name"]] + list(old.get("aliases") or []) + list(e.get("aliases") or [])
                      if str(a).lower() != e["name"].lower()]
                old["aliases"] = list(dict.fromkeys(al))
                old["name"] = e["name"]
            for k, v in e["meta"].items():
                old.setdefault("meta", {})[k] = v
            return old
    cv.manifest.append(e)
    return e


note_raw = _note_raw     # public name for kit after-hooks: records without the same-place dedupe of ``note``


def _note_crop(cv, kind, name, crop, alias=None, **meta):
    e = _make(cv, kind, name, crop=crop, alias=alias, meta=meta)
    if e is not None:
        cv.manifest.append(e)
    return e


def _union_tiles(tiles):
    """[(name, x0, y0, alpha)] -> (alpha crop, x0, y0) covering all of them, or None."""
    tiles = [t for t in tiles if t[3].size]
    if not tiles:
        return None
    x0 = min(t[1] for t in tiles)
    y0 = min(t[2] for t in tiles)
    x1 = max(t[1] + t[3].shape[1] for t in tiles)
    y1 = max(t[2] + t[3].shape[0] for t in tiles)
    if len(tiles) == 1:
        return tiles[0][3], x0, y0
    out = np.zeros((y1 - y0, x1 - x0), F32)
    for _, tx, ty, a in tiles:
        sub = out[ty - y0:ty - y0 + a.shape[0], tx - x0:tx - x0 + a.shape[1]]
        np.maximum(sub, np.clip(a, 0, 1), out=sub)
    return out, x0, y0


def _clip_crop(cv, crop):
    """Clip a crop (alpha, x0, y0) to the canvas."""
    a, x0, y0 = crop
    H, W = cv.h, cv.w
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x0 + a.shape[1]), min(H, y0 + a.shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0], cx0, cy0


# --------------------------------------------------------------------------------------------- decorator

def _extent_from_args(args, W, H):
    """Intended box from ``at`` + ``height`` (fractions), when a kit call has them."""
    at, h = args.get("at"), args.get("height")
    try:
        u, v = float(at[0]), float(at[1])
    except (TypeError, ValueError, IndexError):
        return None
    if max(abs(u), abs(v)) > 3:
        return None
    hh = float(h) if isinstance(h, (int, float)) and 0 < float(h) <= 2 else 0.0
    return (u * W, (v - hh) * H, u * W, v * H)


extent_from_args = _extent_from_args     # public name for kit after-hooks (the intended box of a call)


def drawn(kind, name=None, arg=None, alias=None, after=None, cv_of=None):
    """Decorator for a kit's top-level draw call: the outermost call records one manifest entry from the
    coverage of everything it painted. ``name``: fixed name (default: the function name); ``arg``: the
    parameter that holds the name ("species", "kind"); ``alias``: other words for it; ``after(cap, args,
    result)``: return a list of entries to record instead (dicts with kind, name, crop / mask, meta);
    ``cv_of(args)``: where to find the canvas (default the ``cv`` argument)."""
    def deco(fn):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn
        label = name or fn.__name__.strip("_").replace("_", " ")

        @functools.wraps(fn)
        def wrapper(*a, **k):
            if _STATE["depth"] or not _STATE["on"]:
                return fn(*a, **k)
            try:
                b = sig.bind(*a, **k)
                b.apply_defaults()
                args = b.arguments
                cv = cv_of(args) if cv_of else args.get("cv")
            except Exception:                            # noqa: BLE001 - let fn raise its own error
                return fn(*a, **k)
            if cv is not None and not hasattr(cv, "buf"):
                return fn(*a, **k)
            from . import canvas as _canvas
            cap = _Capture(cv)
            _STATE["depth"] += 1
            _canvas._CAPTURE.append(cap)
            try:
                res = fn(*a, **k)
            finally:
                _STATE["depth"] -= 1
                if cap in _canvas._CAPTURE:
                    _canvas._CAPTURE.remove(cap)
            try:
                _finish(cap, res, args, kind, label, arg, alias, after)
            except Exception:                            # noqa: BLE001 - recording never breaks a drawing
                if os.environ.get("PIXELCANVAS_DEBUG"):
                    raise
            return res
        wrapper.__pc_manifest__ = kind
        return wrapper
    return deco


def _finish(cap, res, args, kind, label, arg, alias, after):
    cv = cap.cv if cap.cv is not None else (res if hasattr(res, "buf") else None)
    if cv is None:
        return
    if cap.cv is None:
        cap.cv = cv
    items = after(cap, args, res) if after else None
    if items is None and cap.children:
        cv.manifest.extend(cap.children)
        return
    if items:
        for it in items:
            it = dict(it)
            crop = it.pop("crop", None)
            if crop is not None:
                crop = _clip_crop(cv, crop)
                if crop is None:
                    continue
                _note_crop(cv, it.pop("kind", kind), it.pop("name"), crop, alias=it.pop("alias", None),
                           **it.pop("meta", {}))
            else:
                _note_raw(cv, it.pop("kind", kind), it.pop("name"), mask=it.pop("mask", None), bbox=it.pop("bbox", None),
                     alias=it.pop("alias", None), **it.pop("meta", {}))
        return
    if items is not None:                                # after() said: nothing to record
        return
    nm = label
    v = None
    if callable(arg):
        try:
            v = arg(args)
        except Exception:                                # noqa: BLE001
            v = None
    elif arg:
        v = args.get(arg)
    if isinstance(v, str) and v.strip():
        nm = v.strip()
    m = cap.mask()
    if m is None:
        return
    ext = _extent_from_args(args, cv.w, cv.h)
    meta = {}
    if nm != label:
        meta["call"] = label
    al = alias(args) if callable(alias) else alias
    if al is None and nm != label:
        al = [label]
    _note_raw(cv, kind, nm, mask=m, extent=ext, alias=al, **meta)


def register(namespace, kind, names, arg=None, alias=None, labels=None):
    """Wrap the functions ``names`` of a module namespace (``globals()``) with ``drawn(kind, ...)``.
    ``arg`` / ``alias`` may be dicts per function name; ``labels`` maps a function name to its entry name."""
    for n in names:
        fn = namespace.get(n)
        if fn is None or not callable(fn) or getattr(fn, "__pc_manifest__", None):
            continue
        a = arg.get(n) if isinstance(arg, dict) else arg
        al = alias.get(n) if isinstance(alias, dict) else alias
        lb = (labels or {}).get(n)
        namespace[n] = drawn(kind, name=lb, arg=a, alias=al)(fn)


def wrap_method(cls, method, kind, after=None, cv_of=None, name=None, arg=None, alias=None):
    """Wrap ``cls.method`` with ``drawn`` (for draw methods of kit classes). ``arg`` / ``alias`` may be callables
    of the bound arguments, e.g. ``arg=lambda a: a["self"].name``."""
    fn = cls.__dict__.get(method)
    if fn is None or getattr(fn, "__pc_manifest__", None):
        return
    setattr(cls, method, drawn(kind, name=name, arg=arg, alias=alias, after=after, cv_of=cv_of)(fn))


def count_calls(cls, methods, attr):
    """Count top-level calls of ``methods`` on each instance in ``instance.<attr>`` (nested calls once)."""
    for m in methods:
        fn = cls.__dict__.get(m)
        if fn is None or getattr(fn, "__pc_count__", None):
            continue

        def make(fn):
            @functools.wraps(fn)
            def w(self, *a, **k):
                d = self.__dict__
                if d.get("_pc_in"):
                    return fn(self, *a, **k)
                d["_pc_in"] = True
                try:
                    return fn(self, *a, **k)
                finally:
                    d["_pc_in"] = False
                    d[attr] = d.get(attr, 0) + 1
            w.__pc_count__ = True
            return w
        setattr(cls, m, make(fn))


# kit-specific "after" hooks ------------------------------------------------------------------------------------

_NODE_PREFIX = ("node ", "participant ", "topic ", "phase ", "milestone ", "marker ")


def _fig_attr(fig, name):
    """A figure's ``nodes`` / ``edges`` (public or underscored attribute, never a method)."""
    for n in (name, "_" + name):
        v = getattr(fig, n, None)
        if v is not None and not callable(v):
            return v
    return None


def after_figure(cap, args, res):
    """diagram figures: one entry per node / participant / phase (from the named vector groups) + containers."""
    fig = args.get("self")
    cv = cap.cv
    try:                                                 # keep the figure for CONNECTIONS (by reference)
        dw = cv.__dict__.get("_drawings") or []
        figs = cv.__dict__.setdefault("_figures", [])
        if not any(f is fig for f, _ in figs):
            figs.append((fig, dw[-1] if dw else None))
    except Exception:                                    # noqa: BLE001
        pass
    out, node_boxes = [], {}
    nodes = _fig_attr(fig, "nodes") or {}
    for nm, x0, y0, a in cap.tiles:
        for p in _NODE_PREFIX:
            if nm.startswith(p):
                key = nm[len(p):]
                n = nodes.get(key) if isinstance(nodes, dict) else None
                label = str(getattr(n, "label", key) if n is not None else key)
                label = " ".join(label.split())
                meta = {"id": key} if n is not None else {}
                if n is not None and getattr(n, "shape", None):
                    meta["shape"] = n.shape
                out.append(dict(kind="node", name=label, crop=(a, x0, y0), alias="node", meta=meta))
                node_boxes[key] = (x0, y0, x0 + a.shape[1], y0 + a.shape[0])
                break
    groups = getattr(fig, "groups", None) or {}
    if isinstance(groups, dict) and groups and nodes:
        pre = []
        for gid, g in groups.items():
            bs = [node_boxes[k] for k, n in nodes.items() if getattr(n, "group", None) == gid and k in node_boxes]
            if not bs:
                continue
            pad = 0.02 * min(cv.w, cv.h)
            box = (min(b[0] for b in bs) - pad, min(b[1] for b in bs) - pad,
                   max(b[2] for b in bs) + pad, max(b[3] for b in bs) + pad)
            pre.append(dict(kind="group", name=" ".join(str(getattr(g, "label", gid)).split()), bbox=box,
                            meta={"id": gid, "nodes": len(bs)}))
        out = pre + out
    if not out:
        u = _union_tiles(cap.tiles)
        if u is None:
            return None
        title = getattr(fig, "title", None)
        return [dict(kind="diagram", name=str(title or type(fig).__name__.lower()), crop=u)]
    return out


def after_chart_part(cap, args, res):
    """chart: one entry per Axes / KPI / header with its title and series names."""
    part = args.get("self")
    parts = [part] if not hasattr(part, "parts") else list(part.parts)
    out = []
    pnames = sorted({str(getattr(p, "name", "")) for p in parts if getattr(p, "name", None)}, key=len, reverse=True)
    owner = {}
    for t in cap.tiles:                                  # longest matching part name wins ("Orders" vs "Orders by ...")
        for pn in pnames:
            if t[0] == pn or t[0].startswith(pn + " "):
                owner.setdefault(pn, []).append(t)
                break
    for p in parts:
        pn = getattr(p, "name", None)
        if pn is None:
            continue
        u = _union_tiles(owner.get(str(pn), []))
        if u is None:
            continue
        title = getattr(p, "title", None) or getattr(p, "label", None)
        series = [s.get("label") for s in getattr(p, "series", []) if isinstance(s, dict) and s.get("label")]
        meta = {"type": type(p).__name__.lower()}
        if title:
            meta["title"] = str(title)
        if series:
            meta["series"] = [str(s) for s in series]
        out.append(dict(kind="chart", name=str(title or pn), crop=u, alias="chart", meta=meta))
    return out or None


def after_sheet(cap, args, res):
    """drafting: one entry per view (with the number of dimensions) + the title block."""
    sheet = args.get("self")
    views = {v.name: v for v in getattr(sheet, "views", [])}
    out = []
    fit = str(getattr(sheet, "fit_report", "") or "")
    if fit and cap.cv is not None:                                # a view was auto-scaled: --check prints the line
        reports = cap.cv.__dict__.setdefault("fit_reports", [])
        if fit not in reports:
            reports.append(fit)
    for nm, x0, y0, a in cap.tiles:
        if nm in views:
            v = views[nm]
            meta = {"dimensions": int(v.__dict__.get("_n_dims", 0)), "scale": getattr(v, "s", None)}
            out.append(dict(kind="view", name=str(v.label_text or v.name), crop=(a, x0, y0), alias="view", meta=meta))
        elif nm == "title block":
            info = getattr(sheet, "info", {}) or {}
            out.append(dict(kind="object", name="title block", crop=(a, x0, y0),
                            meta={k: info[k] for k in ("title", "drawing_no") if info.get(k)}))
    return out or None


def after_drawing(cap, args, res):
    """vector.Drawing.draw: one entry per top-level group (loose items under the drawing's name)."""
    dwg = args.get("self")
    out = []
    for nm, x0, y0, a in cap.tiles:
        if nm in ("background",):
            continue
        out.append(dict(kind="shape", name=nm, crop=(a, x0, y0), meta={"drawing": getattr(dwg, "name", "")}))
    return out or None


def note_render3d(cv, result):
    """solid.Scene.render: one entry per named 3D object, from the id buffer coverage."""
    try:
        used = cv.__dict__.setdefault("kits_used", set())
        used.add("solid")
        models = getattr(getattr(result, "scene", None), "models", []) or []
        if any(type(m).__module__.startswith("pixelcanvas.clay") for m in models):
            used.add("clay")
    except Exception:                                    # noqa: BLE001
        pass
    for nm in list(getattr(result, "layers", []))[2:]:
        try:
            m = result.mask(nm)
        except Exception:                                # noqa: BLE001
            continue
        _note_raw(cv, "object", nm, mask=m)


# --------------------------------------------------------------------------------------------- reading

def _text_entry(cv, r):
    log = getattr(cv, "text_log", None) or []
    i = r.get("ref", -1)
    if not (0 <= i < len(log)):
        return None
    t = log[i]
    txt = " ".join(str(t.get("text", "")).split())
    if not txt:
        return None
    H, W = cv.h, cv.w
    f = _factor(H, W)
    x0, y0, x1, y1 = (int(v) for v in t["bbox"])
    m = t.get("_mask")
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if m is not None and m.shape == (y1 - y0, x1 - x0) and cx1 > cx0 and cy1 > cy0:
        crop = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0].astype(F32)
        small = _shrink(crop, cx0, cy0, H, W, f)
        area = float(crop.sum()) / (H * W)
    else:
        small = np.zeros(_small_shape(H, W, f), np.uint8)
        if cx1 > cx0 and cy1 > cy0:
            small[cy0 // f:int(math.ceil(cy1 / f)), cx0 // f:int(math.ceil(cx1 / f))] = 255
        area = max(0, cx1 - cx0) * max(0, cy1 - cy0) / (W * H) * 0.3
    lb = t.get("layout_bbox") or t["bbox"]
    ext = [min(x0, lb[0]), min(y0, lb[1]), max(x1, lb[2]), max(y1, lb[3])]
    return {"kind": "text", "name": txt, "order": r["order"], "step": r.get("step"),
            "bbox": [_r(cx0 / W), _r(cy0 / H), _r(cx1 / W), _r(cy1 / H)],
            "extent": [_r(ext[0] / W), _r(ext[1] / H), _r(ext[2] / W), _r(ext[3] / H)],
            "area": _r(area, 6), "aliases": ["text", "label"], "meta": {"size": t.get("size")},
            "_mask": small, "_f": f}


def entries(cv):
    """All manifest entries in paint order, text references resolved (each with a small ``_mask``)."""
    out = []
    for e in list(getattr(cv, "manifest", []) or []):
        if e.get("kind") == "text" and "ref" in e:
            e = _text_entry(cv, e)
        if e is not None and e.get("_mask") is not None:
            out.append(e)
    out.sort(key=lambda e: e["order"])
    return _merge_pane_views(out)


def _merge_pane_views(E):
    """A window pane that shows rain (``pane_view=True``) and ``elements.rain`` over the picture are one
    "rain": the pane's entry folds into the same-named effect, so the report lists it once."""
    drop = set()
    out = list(E)
    for i, e in enumerate(E):
        if not (e.get("meta") or {}).get("pane_view"):
            continue
        low = e["name"].split(" #")[0].lower()
        for j, o in enumerate(out):
            if j == i or j in drop or o["kind"] != e["kind"] or (o.get("meta") or {}).get("pane_view"):
                continue
            if o["name"].split(" #")[0].lower() != low or o["_mask"].shape != e["_mask"].shape:
                continue
            o = dict(o)
            o["_mask"] = np.maximum(o["_mask"], e["_mask"])
            b, c = o["bbox"], e["bbox"]
            o["bbox"] = [min(b[0], c[0]), min(b[1], c[1]), max(b[2], c[2]), max(b[3], c[3])]
            o["area"] = max(o["area"], e["area"])
            o["aliases"] = list(dict.fromkeys(list(o.get("aliases") or []) + list(e.get("aliases") or [])))
            out[j] = o
            drop.add(i)
            break
    return [e for i, e in enumerate(out) if i not in drop]


def _display(e):
    return f"label '{e['name'][:40]}'" if e["kind"] == "text" else e["name"]


def _crop_pair(a, b):
    """Intersection of two entries' grid boxes -> slices or None."""
    sh, sw = a["_mask"].shape
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    gx0, gy0 = int(x0 * sw), int(y0 * sh)
    gx1, gy1 = int(math.ceil(x1 * sw)) + 1, int(math.ceil(y1 * sh)) + 1
    return slice(max(0, gy0 - 1), min(sh, gy1)), slice(max(0, gx0 - 1), min(sw, gx1))


def is_backdrop(e):
    """A background area: a container / background kind, ``backdrop=True``, or an entry that fills the canvas."""
    if e.get("kind") in _CONTAINER or (e.get("meta") or {}).get("backdrop"):
        return True
    b = e.get("bbox") or [0, 0, 0, 0]
    return e.get("kind") != "text" and float(e.get("area", 0)) >= 0.85 and b[2] - b[0] >= 0.97 and b[3] - b[1] >= 0.97


def is_see_through(e):
    """Glass, haze, a water surface, a region outline: never counted as hiding what lies under it."""
    return e.get("kind") in _SEE_THROUGH or bool((e.get("meta") or {}).get("see_through"))


def _small_image(cv, shape):
    """The canvas block-averaged onto the manifest grid (what the final picture shows in each cell)."""
    a = np.clip(np.asarray(cv.buf, F32)[..., :3], 0, 1)
    sh, sw = shape
    H, W = a.shape[:2]
    f = _factor(H, W)
    pad = np.pad(a, ((0, sh * f - H), (0, sw * f - W), (0, 0)), mode="edge")
    return pad.reshape(sh, f, sw, f, 3).mean((1, 3))


def _visible_in_picture(cv, E, own, occ, vis):
    """Visible share of each entry judged from the final picture, not from note order alone: a later entry
    hides a cell of an earlier one only where the picture shows the later entry's own colour there (a pond
    noted after its lily pads, a glass panel noted after the people behind it hide nothing). Returns
    (visible shares, {index: index of the main hider})."""
    try:
        img = _small_image(cv, own[0].shape)
    except Exception:                                    # noqa: BLE001 - fall back to paint order
        return vis, {}
    refs = {}

    def ref(j):
        if j not in refs:
            cells = img[occ[j]]
            if len(cells) < 4:
                refs[j] = None
            else:
                c = np.median(cells, axis=0)
                spread = float(np.percentile(np.abs(cells - c).max(-1), 75))
                refs[j] = (c, max(0.06, 1.25 * spread))
        return refs[j]
    out, hider = list(vis), {}
    later_union = [None] * len(E)
    u = np.zeros_like(own[0])
    for i in range(len(E) - 1, -1, -1):
        later_union[i] = u.copy()
        if occ[i] is not None:
            u |= occ[i]
    for i, e in enumerate(E):
        if vis[i] >= 0.9 or e["kind"] in _CONTAINER or is_backdrop(e):
            continue
        n = int(own[i].sum())
        if not n:
            continue
        mine = own[i] & ~later_union[i]
        ref_i = np.median(img[mine], axis=0) if int(mine.sum()) >= 4 else None
        covered = np.zeros_like(own[i])
        best, bj = 0, None
        for j in range(i + 1, len(E)):
            if occ[j] is None:
                continue
            sl = _crop_pair(e, E[j])
            if sl is None:
                continue
            O = own[i][sl] & occ[j][sl]
            if not O.any():
                continue
            r = ref(j)
            if r is None:
                top = O
            else:
                px = img[sl][O]
                dj = np.abs(px - r[0]).max(-1)
                hit = dj <= r[1]
                if ref_i is not None:
                    hit &= dj <= np.abs(px - ref_i).max(-1)
                top = np.zeros_like(O)
                top[O] = hit
            k = int(top.sum())
            if k:
                covered[sl] |= top
                if k > best:
                    best, bj = k, j
        out[i] = 1.0 - float(covered.sum()) / n
        if bj is not None:
            hider[i] = bj
    return out, hider


def summary(cv, entries_=None, cues=False):
    """One dict per drawn entry (paint order): name, kind, centre (u, v), size (w, h), area %, visible % (share
    not covered by later opaque entries), clipped % (share of its intended box off the canvas), hidden_by, step.
    Renderer cues (kind ``"cue"``: lighting, shadows, reflections) are left out unless ``cues=True``."""
    E = entries(cv) if entries_ is None else entries_
    if not cues:
        E = [e for e in E if e.get("kind") != CUE]
    if not E:
        return []
    own = [e["_mask"] >= _OWN for e in E]
    occ = [None if is_backdrop(e) or is_see_through(e) else e["_mask"] >= _OCCLUDE for e in E]
    vis = [1.0] * len(E)
    union = np.zeros_like(own[0])
    for i in range(len(E) - 1, -1, -1):
        n = int(own[i].sum())
        if n and E[i]["kind"] not in _CONTAINER and not is_backdrop(E[i]):
            vis[i] = 1.0 - float((own[i] & union).sum()) / n
        if occ[i] is not None:
            union |= occ[i]
    hider = {}
    if any(v < 0.9 for v in vis):
        vis, hider = _visible_in_picture(cv, E, own, occ, vis)
    names, seen = [], {}
    for e in E:
        d = _display(e)
        seen[d] = seen.get(d, 0) + 1
        names.append(d if seen[d] == 1 else f"{d} #{seen[d]}")
    out = []
    for i, e in enumerate(E):
        hid = None
        if vis[i] < 0.9 and i in hider:
            hid = names[hider[i]]
        elif vis[i] < 0.9:
            best, bn = 0, None
            for j in range(i + 1, len(E)):
                if occ[j] is None:
                    continue
                sl = _crop_pair(e, E[j])
                if sl is None:
                    continue
                k = int((own[i][sl] & occ[j][sl]).sum())
                if k > best:
                    best, bn = k, names[j]
            hid = bn
        m = own[i]
        ys, xs = np.nonzero(m)
        sh, sw = m.shape
        if len(xs):
            cu, cvv = (xs.mean() + 0.5) / sw, (ys.mean() + 0.5) / sh
        else:
            cu, cvv = (e["bbox"][0] + e["bbox"][2]) / 2, (e["bbox"][1] + e["bbox"][3]) / 2
        ex = e["extent"]
        ea = max((ex[2] - ex[0]) * (ex[3] - ex[1]), 1e-9)
        ia = max(0.0, min(ex[2], 1) - max(ex[0], 0)) * max(0.0, min(ex[3], 1) - max(ex[1], 0))
        clipped = 0.0 if ea <= 1e-8 else max(0.0, 1.0 - ia / ea)
        if ex[2] - ex[0] <= 1e-6 or ex[3] - ex[1] <= 1e-6:          # a point / line extent: inside or not
            clipped = 0.0 if (0 <= ex[0] <= 1 and 0 <= ex[1] <= 1) else 1.0
        d = {"name": names[i], "kind": e["kind"], "centre": [_r(cu, 3), _r(cvv, 3)],
             "size": [_r(e["bbox"][2] - e["bbox"][0], 3), _r(e["bbox"][3] - e["bbox"][1], 3)],
             "bbox": e["bbox"], "area_pct": _r(e["area"] * 100, 2), "visible_pct": _r(vis[i] * 100, 1),
             "clipped_pct": _r(clipped * 100, 1), "hidden_by": hid, "step": e.get("step"), "order": e["order"]}
        sb = _bbox(e["_mask"] >= _OCCLUDE)                       # the solid part (no soft cast shadow / glow)
        if sb is not None:
            d["solid_bbox"] = [_r(sb[2] / sw, 3), _r(sb[0] / sh, 3), _r(sb[3] / sw, 3), _r(sb[1] / sh, 3)]
        if is_backdrop(e):
            d["backdrop"] = True
        if is_see_through(e):
            d["see_through"] = True
        if e.get("aliases"):
            d["aliases"] = list(e["aliases"])
        if e.get("meta"):
            d["meta"] = {k: v for k, v in e["meta"].items() if isinstance(v, (str, int, float, bool, list, type(None)))}
        out.append(d)
    return out


def _where(u, v):
    h = "left" if u < 1 / 3 else "right" if u > 2 / 3 else "centre"
    vv = "top" if v < 1 / 3 else "bottom" if v > 2 / 3 else "middle"
    return f"{vv} {h}" if (h, vv) != ("centre", "middle") else "centre"


def relations(cv, limit=12, rows=None):
    """Plain statements about the largest entries: 'cat is left of dog', 'bird is above tree',
    'mug overlaps bottle (12%)', "label 'Blue Hour' is inside poster top third"."""
    E = entries(cv)
    S = summary(cv, E) if rows is None else rows
    if not S or is_text_only(S):
        return []                                        # a pure typography picture: see reading_order()
    by_name = {s["name"]: (s, e) for s, e in zip(S, E)}
    cand = [s for s in S if s["kind"] not in ("style", "background", "text") and s["area_pct"] < 85
            and s["visible_pct"] > 5 and not (s["kind"] in _SEE_THROUGH and s["area_pct"] > 15)]
    cand.sort(key=lambda s: -s["area_pct"])
    top = cand[:8]
    texts = [s for s in S if s["kind"] == "text"][:4]
    pairs = []
    for i, a in enumerate(top):                          # nearest pairs first, each entry in at most 3 statements
        for b in top[i + 1:]:
            d = math.hypot(a["centre"][0] - b["centre"][0], a["centre"][1] - b["centre"][1])
            pairs.append((d - 0.002 * (a["area_pct"] + b["area_pct"]), a, b))
    pairs.sort(key=lambda t: t[0])
    uses = {}
    out = []
    for _, a, b in pairs:
        if len(out) >= limit - min(len(texts), 3):
            break
        if uses.get(a["name"], 0) >= 3 or uses.get(b["name"], 0) >= 3:
            continue
        uses[a["name"]] = uses.get(a["name"], 0) + 1
        uses[b["name"]] = uses.get(b["name"], 0) + 1
        ea, eb = by_name[a["name"]][1], by_name[b["name"]][1]
        sl = _crop_pair(ea, eb)
        inter = 0
        if sl is not None:
            inter = int(((ea["_mask"][sl] >= _OWN) & (eb["_mask"][sl] >= _OWN)).sum())
        na, nb = int((ea["_mask"] >= _OWN).sum()), int((eb["_mask"] >= _OWN).sum())
        small, big = (a, b) if na <= nb else (b, a)
        nsmall = min(na, nb)
        if nsmall and inter / nsmall > 0.9 and max(na, nb) > 2 * nsmall:
            bb = big["bbox"]
            rv = (small["centre"][1] - bb[1]) / max(bb[3] - bb[1], 1e-6)
            part = "top third" if rv < 1 / 3 else "bottom third" if rv > 2 / 3 else "middle"
            out.append(f"{small['name']} is inside {big['name']} {part}")
        elif nsmall and inter / nsmall > 0.03:
            out.append(f"{a['name']} overlaps {b['name']} ({inter / nsmall * 100:.0f}%)")
        else:
            du = b["centre"][0] - a["centre"][0]
            dv = b["centre"][1] - a["centre"][1]
            if abs(du) >= abs(dv):
                out.append(f"{a['name']} is {'left' if du > 0 else 'right'} of {b['name']}")
            else:
                out.append(f"{a['name']} is {'above' if dv > 0 else 'below'} {b['name']}")
    for t in texts:
        if len(out) >= limit:
            break
        if any(t["name"] in s for s in out):
            continue
        host = None
        for s in sorted(S, key=lambda s: s["area_pct"]):
            if s is t or s["kind"] in ("text", "effect") or s["area_pct"] <= t["area_pct"] * 2:
                continue
            b = s["bbox"]
            if b[0] <= t["centre"][0] <= b[2] and b[1] <= t["centre"][1] <= b[3]:
                host = s
                break
        if host is not None:
            bb = host["bbox"]
            rv = (t["centre"][1] - bb[1]) / max(bb[3] - bb[1], 1e-6)
            part = "top third" if rv < 1 / 3 else "bottom third" if rv > 2 / 3 else "middle"
            out.append(f"{t['name']} is inside {host['name']} {part}")
        else:
            out.append(f"{t['name']} is at the {_where(*t['centre'])} of the canvas")
    return out[:limit]


def display_width(t):
    """Terminal cells a string takes: CJK / full-width characters are two cells wide."""
    import unicodedata
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in t)


def _fit(t, n):
    while display_width(t) > n:
        t = t[:-2] + "~"
    return t + " " * (n - display_width(t))


def is_text_only(rows):
    """True when every entry that is not a container / style is a text entry (and there is at least one)."""
    body = [r for r in rows if r["kind"] not in _CONTAINER]
    return bool(body) and all(r["kind"] == "text" for r in body)


def reading_order(rows, size=None, limit=20):
    """Text entries top to bottom (left to right within a line): ``"1. label 'Blue Hour'  top centre  64 px high"``.
    ``size`` = (width, height) of the canvas in px; without it heights are fractions."""
    T = [r for r in rows if r["kind"] == "text"]
    T.sort(key=lambda r: (round(r["bbox"][1] * 40), r["bbox"][0]))
    out = []
    for i, r in enumerate(T[:limit], 1):
        h = f"{r['size'][1] * size[1]:.0f} px high" if size else f"{r['size'][1]:.3f} high"
        out.append(f"{i}. {r['name']}  {_where(*r['centre'])}  {h}")
    if len(T) > limit:
        out.append(f"... {len(T) - limit} more text entries (all of them are in --json)")
    return out


def compact(rows, limit=20):
    """Rows to print, at most about ``limit``: repeated names (pine, pine #2 ...) merge into one entry
    ``{"merged": True, name, kind, count, u: (lo, hi), v: (lo, hi), area_pct}``; when still too many the largest stay."""
    if len(rows) <= limit:
        return list(rows)
    groups, order = {}, []
    for r in rows:
        key = (r["name"].split(" #")[0], r["kind"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    items = []
    for key in order:
        g = groups[key]
        if len(g) == 1:
            items.append(g[0])
            continue
        n = sum(int((r.get("meta") or {}).get("count", 1)) for r in g)
        items.append({"merged": True, "name": key[0], "kind": key[1], "count": n, "n_entries": len(g),
                      "u": (min(r["bbox"][0] for r in g), max(r["bbox"][2] for r in g)),
                      "v": (min(r["bbox"][1] for r in g), max(r["bbox"][3] for r in g)),
                      "area_pct": sum(r["area_pct"] for r in g), "order": min(r["order"] for r in g),
                      "visible_pct": sum(r["visible_pct"] for r in g) / len(g)})
    if len(items) > limit:
        items = sorted(sorted(items, key=lambda s: -s["area_pct"])[:limit], key=lambda s: s["order"])
    return items


def table(rows, limit=20):
    """Compact aligned text table of ``summary`` rows (CJK-aware column widths). With more than ``limit`` rows,
    repeated names merge into one line ("pine x14  plant  spread 0.10-0.92 wide, 0.55-0.80 high")."""
    if not rows:
        return ["(nothing recorded: draw with the kits or call manifest.note(cv, kind, name, mask=...))"]
    show = compact(rows, limit)
    shown = sum(int(s["n_entries"]) if s.get("merged") else 1 for s in show)
    label = lambda s: f"{s['name']} x{s['count']}" if s.get("merged") else s["name"]          # noqa: E731
    nw = min(30, max(4, max(display_width(label(s)) for s in show)))
    kw = max(4, max(len(s["kind"]) for s in show))
    lines = [f"{'name':<{nw}}  {'kind':<{kw}}  centre(u,v)  size(w,h)    area%  visible%  clipped%  hidden_by"]
    for s in show:
        nm = _fit(s["name"], nw)
        if s.get("merged"):
            lines.append(f"{_fit(label(s), nw)}  {s['kind']:<{kw}}  spread "
                         f"{s['u'][0]:.2f}-{s['u'][1]:.2f} wide, {s['v'][0]:.2f}-{s['v'][1]:.2f} high   "
                         f"area {s['area_pct']:.2f}%  (merged {s['n_entries']} entries)")
            continue
        m = s.get("meta") or {}
        extra = []
        if m.get("series"):
            extra.append(f"series: {', '.join(m['series'][:4])}")
        if "dimensions" in m:
            extra.append(f"{m['dimensions']} dims")
        if m.get("count"):
            extra.append(f"x{m['count']}")
        lines.append(f"{nm}  {s['kind']:<{kw}}  {s['centre'][0]:.2f},{s['centre'][1]:.2f}    "
                     f"{s['size'][0]:.2f}x{s['size'][1]:.2f}  {s['area_pct']:6.2f}  {s['visible_pct']:7.0f}  "
                     f"{s['clipped_pct']:7.0f}   {s['hidden_by'] or '-'}" + (f"   ({'; '.join(extra)})" if extra else ""))
    if shown < len(rows):
        lines.append(f"... {len(rows) - shown} smaller entries not shown (all of them are in --json)")
    return lines


# --------------------------------------------------------------------------------------------- checks

def _issue(rule, severity, message, where, value, threshold, fix):
    from .vision import _warn
    return _warn(rule, severity, message, where, value, threshold, fix)


def _short(line, w):
    """``w`` with ``short``: its one line for the FIX block ("cup: painted green, the request says blue")."""
    w["short"] = line
    return w


_MARKERS = {"arrow", "arrows", "leader", "leaders", "pointer", "callout", "line", "lines", "label", "labels",
            "arrowhead", "connector", "tick", "ticks"}


def _is_marker(name):
    """An arrow, leader line or label drawn to point at things: never a subject that must read at size."""
    ws = str(name).split(" #")[0].lower().split()
    return bool(ws) and (ws[-1] in _MARKERS or ws[0] in ("arrow", "label", "leader"))


def _is_band(s):
    """A row spanning the canvas in one direction (a paper sheet, sky, ground, hedge, footer band)."""
    return s["size"][0] >= 0.9 or s["size"][1] >= 0.9


def _part_of_assembly(s, rows):
    """True for a small named part of a larger thing ("lamp stand" next to "lamp shade", "round rug centre"
    inside "round rug"): its first word names a bigger entry that touches it."""
    words = s["name"].split(" #")[0].lower().split()
    if len(words) < 2:
        return False
    b = s["bbox"]
    for r in rows:
        if r is s or r["kind"] == "text":
            continue
        rw = r["name"].split(" #")[0].lower().split()
        if not rw or rw[0] != words[0] or r["area_pct"] <= s["area_pct"]:
            continue
        o, pad = r["bbox"], 0.02
        if b[0] <= o[2] + pad and o[0] <= b[2] + pad and b[1] <= o[3] + pad and o[1] <= b[3] + pad:
            return True
    return False


def object_checks(cv, rows=None):
    """Issues about the recorded objects (off-canvas, hidden, tiny, stacked duplicates, touching the edge)."""
    S = summary(cv) if rows is None else rows
    out = []
    S_all = S
    S = [s for s in S if not s.get("backdrop")]          # backgrounds: never the subject, never cropped or hidden
    subj = [s for s in S if s["kind"] in _SUBJECT]
    big = sorted([s for s in subj if s["area_pct"] >= 1.0 and not _is_band(s)],
                 key=lambda s: -s["area_pct"])[:5]
    same = {}
    for s in S_all:
        b = s["name"].split(" #")[0]
        same[b] = same.get(b, 0) + 1
    for s in S:
        nm, k = s["name"], s["kind"]
        if k in _CONTAINER or k == "text":
            continue
        if s["clipped_pct"] > 50:
            out.append(_issue("object_off_canvas", "fail" if s["clipped_pct"] > 80 else "warn",
                              f"{nm} ({k}) is {s['clipped_pct']:.0f}% outside the canvas.", s["bbox"],
                              s["clipped_pct"] / 100, 0.5, f"move {nm} inside the frame (its at= / box=) or make it smaller"))
        hb = next((r for r in S_all if r["name"] == s["hidden_by"]), None) if s["hidden_by"] else None
        layered = _is_band(s) and (hb is None or _is_band(hb))          # stacked sheets / tiers show a band each
        if s["visible_pct"] < 30 and k not in ("effect", "texture") and s["area_pct"] > 0.05 and not layered \
                and not s.get("see_through"):
            by = f" behind {s['hidden_by']}" if s["hidden_by"] else ""
            out.append(_issue("object_hidden", "warn", f"Only {s['visible_pct']:.0f}% of {nm} is visible{by}.",
                              s["bbox"], s["visible_pct"] / 100, 0.3,
                              f"draw {nm} after {s['hidden_by']} or move one of them" if s["hidden_by"] else
                              f"draw {nm} later or move it"))
        base = nm.split(" #")[0]
        if k in _SUBJECT and s["area_pct"] < 0.15 and base.lower() not in _DETAIL and same.get(base, 0) < 3 \
                and not _is_marker(base) \
                and not _part_of_assembly(s, S) and not (s.get("meta") or {}).get("count", 1) > 2:
            out.append(_issue("object_tiny", "warn", f"{nm} ({k}) covers only {s['area_pct']:.2f}% of the canvas: "
                              "it will not read as a subject.", s["bbox"], s["area_pct"] / 100, 0.0015,
                              f"make {nm} larger (height= / size=) or drop it"))
    for i, a in enumerate(subj):
        base_a = a["name"].split(" #")[0]
        for b in subj[i + 1:]:
            if b["name"].split(" #")[0] != base_a:
                continue
            if base_a.lower() in _SCATTER or _is_band(a) or _is_band(b) or \
                    min(a["size"][0], b["size"][0]) > 0.45:  # two calls of one scatter (far and near forest)
                continue
            A, B = a["bbox"], b["bbox"]
            iw = max(0.0, min(A[2], B[2]) - max(A[0], B[0]))
            ih = max(0.0, min(A[3], B[3]) - max(A[1], B[1]))
            inter = iw * ih
            uni = (A[2] - A[0]) * (A[3] - A[1]) + (B[2] - B[0]) * (B[3] - B[1]) - inter
            if uni > 0 and inter / uni > 0.6:
                out.append(_issue("objects_stacked", "warn", f"{a['name']} and {b['name']} sit on top of each other "
                                  f"({inter / uni * 100:.0f}% of their boxes overlap).", A, inter / uni, 0.6,
                                  "give them different at= positions, or draw only one"))
    for s in big:
        b = s.get("solid_bbox") or s["bbox"]                     # a soft cast shadow reaching the edge is fine
        e = 0.004
        sides = [n for n, hit in (("left", b[0] <= e), ("right", b[2] >= 1 - e), ("top", b[1] <= e)) if hit]
        if b[3] >= 1 - e and s["clipped_pct"] > 0:
            sides.append("bottom")
        if s["size"][0] > 0.45:                          # bands (hedges, ground, sky) run edge to edge on purpose
            sides = [x for x in sides if x not in ("left", "right")]
        if s["size"][1] > 0.8:
            sides = [x for x in sides if x not in ("top", "bottom")]
        if sides and s["clipped_pct"] <= 50 and (s["kind"] != "chart" or s["clipped_pct"] > 0):
            out.append(_issue("object_at_edge", "warn",
                              f"{s['name']} touches the {'/'.join(sides)} edge of the canvas"
                              + (f" and is cut by the frame ({s['clipped_pct']:.0f}%)." if s["clipped_pct"] > 0 else "."),
                              b, sides, 0.004, f"move {s['name']} inward so a margin of at least 4% remains, "
                              "or let it overlap the edge clearly on purpose"))
    return out


def reconcile_cropped(warnings, rows, object_issues=()):
    """Check the pixel critique's ``subject_cropped`` against what was recorded: keep it only when a recorded
    subject (not a backdrop, band, effect or text) really reaches the cut edge, and name that subject. A pixel
    shape that no recorded subject explains at the edge (a footer band, a gradient backdrop, a ground shadow) is
    dropped. Without recorded subjects the warning stays as it is."""
    subj = [s for s in rows if s["kind"] not in ("text", "effect", "texture", "style") and s["kind"] not in _CONTAINER
            and not s.get("backdrop") and not _is_band(s)]
    if not subj:
        if rows and all(s.get("backdrop") or s["kind"] in _CONTAINER for s in rows):
            return [w for w in warnings if w.get("rule") != "subject_cropped"]       # a wallpaper has no subject
        return list(warnings)
    named = {w["message"].split(" touches")[0] for w in object_issues if w.get("rule") == "object_at_edge"}
    e = 0.004
    out = []
    for w in warnings:
        if w.get("rule") != "subject_cropped":
            out.append(w)
            continue
        box = w.get("where") or [0, 0, 1, 1]
        sides = w.get("value") if isinstance(w.get("value"), list) else ["left", "right", "top", "bottom"]
        hits = []
        for s in subj:
            b = s.get("solid_bbox") or s["bbox"]
            if b[0] > box[2] or b[2] < box[0] or b[1] > box[3] or b[3] < box[1]:
                continue
            reach = [d for d, ok in (("left", b[0] <= e), ("right", b[2] >= 1 - e), ("top", b[1] <= e),
                                     ("bottom", b[3] >= 1 - e)) if ok and d in sides]
            if reach and s["name"] not in named:
                hits.append((s, reach))
        if not hits:
            continue                                     # no recorded subject is cut there
        s, reach = max(hits, key=lambda t: t[0]["area_pct"])
        out.append(dict(w, where=s["bbox"], value=reach,
                        message=f"{s['name']} ({s['kind']}) touches the {'/'.join(reach)} edge and is cut off by "
                                "the frame.",
                        fix=f"move or shrink {s['name']} so a margin of at least 4% remains, or let it overlap the "
                            "edge clearly on purpose"))
    return out


_QUOTES = [r'"([^"\n]{1,200})"', "\u201c([^\u201d\n]{1,200})\u201d", "\u300c([^\u300d\n]{1,200})\u300d",
           "\u300e([^\u300f\n]{1,200})\u300f", "\u2018([^\u2019\n]{1,200})\u2019",
           r"(?:(?<=^)|(?<=[\s(\[:,]))'([^'\n]{1,120})'(?=$|[\s.,!?;:)\]])"]


_NOT_TEXT_INTRO = re.compile(r"(?:contains?|containing|sequence|letters?|pattern|substring|string|ends? with|"
                             r"starts? with|spelled|spelt|syllable|sound)\s*(?:of\s+)?(?:the\s+)?$", re.I)


_CLAUSE_END = r"(?=[.,;:!?()\[\]]|\s(?:and|but|while|so)\s|$)"
_UNASKED = [
    r"\b(?:nothing|none|nobody|neither|nor)\b[^.,;:!?()]*?" + _CLAUSE_END,    # "none of them red"
    r"(?:\b(?:not|never)\b|n't\b)[^.,;:!?()]*?" + _CLAUSE_END,             # "not on the rock", "do not draw a person"
    r"\b(?:seen|viewed|shown|drawn|painted|rendered|photographed|lit)\s+(?:simply\s+)?(?:from|in|as|with|by|for|"
    r"at)\b[^.,;:!?()]*" + _CLAUSE_END,                             # "seen from the doorway"
    r"\bthe\s+same\s+colou?r\s+as\s+(?:the\s+|a\s+|an\s+)?",         # "the same colour as the cloth": no colour
    r"\bfor\s+(?:a|an|my|our|his|her|their|your)\b(?![^.,;:!?()]*\d)[^.,;:!?()]*" + _CLAUSE_END,   # "for a kids' book"
    r"\bin\s+the\s+(?:style|manner|spirit)\s+of\b[^.,;:!?()]*" + _CLAUSE_END,   # "in the style of a woodblock print"
    r"^\s*(?:(?:please\s+)?(?:make|draw|create|render|paint|generate|show|give\s+me)\s+(?:me\s+)?)?(?:an?|the)\s+"
    r"(?:[\w'-]+\s+){0,3}?(?:drawing|print|painting|illustration|sketch|engraving|etching|woodcut|rendering|"
    r"render|photo|photograph|picture|image)\s+of\b",                # "An engineering drawing of", "a woodblock print of"
]
_UNASKED_RE = [re.compile(p, re.I) for p in _UNASKED]


_NEG_NP = re.compile(r"\b(?:no|without|except|excluding|free of|other than|instead of)\s+", re.I)
_NP_END = {"on", "in", "at", "of", "by", "to", "from", "for", "near", "under", "over", "into", "onto", "beside",
           "behind", "next", "above", "below", "beneath", "with", "and", "or", "but", "while", "inside", "outside",
           "around", "across", "along", "toward", "towards", "against", "between", "than", "anywhere", "at"}


def _blank_negated_nps(t):
    """Blank "no X", "without X", "except X" up to the end of X's noun phrase; "no X is red" to the clause end."""
    out = t
    for m in _NEG_NP.finditer(t):
        k = m.end()
        for w in re.finditer(r"[A-Za-z0-9'’-]+|[^\sA-Za-z0-9]", t[m.end():]):
            tok = w.group(0).lower()
            if tok in _NP_END or not tok[0].isalnum() or (len(tok) > 4 and tok.endswith("ing")):
                break
            if tok in ("is", "are", "was", "were", "has", "have"):
                e = re.search(r"[.,;:!?()]|\s(?:and|but)\s|$", t[m.end() + w.start():])
                k = m.end() + w.start() + e.start()
                break
            k = m.end() + w.end()
        out = out[:m.start()] + " " * (k - m.start()) + out[k:]
    return out


def _clean_request(text):
    """``text`` with the parts that ask for nothing blanked (same length): quoted strings, negations ("no
    handle", "without a lid", "none of them red", "not on the rock"), viewpoints ("seen from the front"), purposes
    ("for a children's book") and "the same colour as"; "black-and-white" reads as one colour pair."""
    t = text or ""
    for q in quoted(t):
        t = t.replace(q, " " * len(q))
    t = re.sub(r'(?<=\d)(?:"|\'\'|″)', " ", t)
    t = re.sub(r"\b([A-Za-z]+)-and-([A-Za-z]+)\b",
               lambda m: (f"{m.group(1)}-{m.group(2)}" + "    ") if _colour_of(m.group(1).lower()) and
               _colour_of(m.group(2).lower()) else m.group(0), t)
    t = _blank_negated_nps(t)
    for rx in _UNASKED_RE:
        t = rx.sub(lambda m: " " * len(m.group(0)), t)
    return t


def _inch_marks(text):
    """``text`` with the inch / foot marks after a number blanked (6" ruler), never a closing quote ("2 + 2 = 4")."""
    out, open_ = list(text), False
    for i, ch in enumerate(text):
        if ch == '"':
            if i and text[i - 1].isdigit() and not open_:
                out[i] = " "
                continue
            open_ = not open_
        elif ch in "″′" and i and text[i - 1].isdigit():
            out[i] = " "
    return "".join(out)


def quoted(prompt):
    """Quoted strings of a request to draw, in the request's order: straight or curly double / single quotes and
    CJK corner brackets. Inch and foot marks (6" ruler), a quote inside a quote, and a string the request talks
    about rather than shows ("words that contain the sequence 'oo'") are left out."""
    text = _inch_marks(prompt or "")                                # 6" ruler, 5' fence: measurement marks
    found = []
    for p in _QUOTES:
        for m in re.finditer(p, text):
            s = m.group(1).strip()
            if not s or _NOT_TEXT_INTRO.search(text[:m.start()][-40:]):
                continue
            found.append((m.start(), m.end(), s))
    found.sort()
    out, spans = [], []
    for a, b, s in found:
        if any(a0 < a and b <= b0 for a0, b0 in spans):
            continue                                     # 'Moby Dick' inside "Read 'Moby Dick' today"
        spans.append((a, b))
        if s not in out:
            out.append(s)
    return out


def _norm(s):
    return " ".join(str(s).casefold().split())


def drawn_texts(cv):
    """Every string drawn on the canvas: ``cv.text`` calls plus text inside vector drawings (diagrams, charts,
    drafting sheets)."""
    out = [str(t.get("text", "")) for t in (getattr(cv, "text_log", None) or [])]

    def walk(items):
        for it in items:
            if hasattr(it, "text") and isinstance(getattr(it, "text"), str) and not hasattr(it, "children"):
                out.append(it.text)
            ch = getattr(it, "children", None)
            if ch:
                walk(ch)
    for d in cv.__dict__.get("_drawings", []):
        root = getattr(d, "root", None)
        if root is not None:
            walk(root.children)
    return [t for t in out if t.strip()]


def _dwg_matrix(dwg, W, H):
    """The drawing -> canvas matrix ``Drawing.draw`` used (fit into its box, centred)."""
    info = getattr(dwg, "_pc_drawn", None)
    box = info[0] if info else None
    bx0, by0, bx1, by1 = box if box is not None else (0, 0, W, H)
    k = min((bx1 - bx0) / max(dwg.width, 1e-9), (by1 - by0) / max(dwg.height, 1e-9))
    return (k, 0.0, 0.0, k, bx0 + ((bx1 - bx0) - dwg.width * k) / 2, by0 + ((by1 - by0) - dwg.height * k) / 2)


def vector_texts(cv):
    """Every text label inside the vector drawings painted on ``cv`` (diagrams, charts, UI, drafting):
    ``{"text", "bbox" (canvas px), "size" (px), "drawing", "group", "owner_box" (the node's body, px, or None),
    "color" (rgb or None), "opacity"}`` in paint order."""
    import sys as _sys
    vec = _sys.modules.get("pixelcanvas.vector")
    drawings = cv.__dict__.get("_drawings") or []
    if vec is None or not drawings:
        return []
    out = []
    W, H = cv.w, cv.h

    def colour(fill, dwg):
        try:
            if fill is None:
                th = dwg.theme
                fill = th["ink"] if th is not None else "#000000"
            c = vec._color(fill)
            return None if c is None else [float(v) for v in np.asarray(c[0], F32).reshape(-1)[:3]]
        except Exception:                                # noqa: BLE001 - gradients and other paints
            return None

    def walk(item, m, dwg, group, owner, opacity):
        for ch in getattr(item, "children", []):
            if isinstance(ch, vec.Group):
                mm = vec._mul(m, ch.matrix)
                nm = ch.name or group
                own = owner
                if ch.name and str(ch.name).startswith("node "):         # a diagram node: its card
                    body = next((c for c in ch.children if type(c).__name__ == "Shape"), None)
                    try:
                        own = body._bbox(mm) if body is not None else None
                    except Exception:                    # noqa: BLE001
                        own = None
                walk(ch, mm, dwg, nm, own, opacity * float(getattr(ch, "opacity", 1.0)))
            elif isinstance(ch, vec.Text):
                txt = " ".join(str(ch.text).split())
                if not txt:
                    continue
                try:
                    b = ch._bbox(m)
                except Exception:                        # noqa: BLE001
                    continue
                k = math.sqrt(abs(m[0] * m[3] - m[1] * m[2]))
                out.append({"text": txt, "bbox": [float(v) for v in b], "size": float(ch.size) * k,
                            "drawing": getattr(dwg, "name", "drawing"), "group": group, "owner_box": owner,
                            "color": colour(ch.fill, dwg), "halo": bool(getattr(ch, "halo", None)),
                            "opacity": opacity * float(getattr(ch, "opacity", 1.0))})

    for d in drawings:
        root = getattr(d, "root", None)
        if root is None:
            continue
        try:
            walk(root, _dwg_matrix(d, W, H), d, None, None, 1.0)
        except Exception:                                # noqa: BLE001 - reporting never breaks a render
            if os.environ.get("PIXELCANVAS_DEBUG"):
                raise
    return out


_ARROWS = {"end": "→", "start": "←", "both": "↔", "none": "—"}
_DASHED = {"end": "⇢", "start": "⇠", "both": "⇄", "none": "┄"}


def connections(cv):
    """The edges of every diagram drawn on ``cv``: ``{"from", "to", "label", "style", "arrow", "text"}``
    (ids as in the scene code; ``text`` like ``web → lb (HTTPS / WSS)``)."""
    out = []
    for fig, _ in cv.__dict__.get("_figures", []) or []:
        edges, nodes = _fig_attr(fig, "edges"), _fig_attr(fig, "nodes")
        if not isinstance(edges, list) or not isinstance(nodes, dict):
            continue
        for e in edges:
            a, b = getattr(e, "a", None), getattr(e, "b", None)
            if a is None or b is None:
                continue
            style = str(getattr(e, "style", "solid") or "solid")
            arrow = str(getattr(e, "arrow", "end") or "none")
            lab = getattr(e, "label", None)
            sym = (_DASHED if style in ("dashed", "dotted") else _ARROWS).get(arrow, "→")
            extra = [str(lab)] if lab else []
            if style != "solid":
                extra.append(style)
            out.append({"from": str(a), "to": str(b), "label": None if lab is None else str(lab), "style": style,
                        "arrow": arrow, "text": f"{a} {sym} {b}" + (f" ({', '.join(extra)})" if extra else "")})
    return out


def connection_checks(cv):
    """Issues about diagram edges: a node with no edge (when others have edges), duplicate edges, an edge
    label plate that covers a node."""
    out = []
    for fig, dwg in cv.__dict__.get("_figures", []) or []:
        edges, nodes = _fig_attr(fig, "edges"), _fig_attr(fig, "nodes")
        if not isinstance(edges, list) or not isinstance(nodes, dict) or not edges:
            continue
        linked = {getattr(e, "a", None) for e in edges} | {getattr(e, "b", None) for e in edges}
        lonely = [k for k, n in nodes.items() if k not in linked and not getattr(n, "dummy", False)]
        for k in lonely[:6]:
            lab = " ".join(str(getattr(nodes[k], "label", k)).split())
            out.append(_issue("node_unconnected", "warn", f"node {k!r} ({lab}) has no edge while the diagram has "
                              f"{len(edges)}.", None, k, None,
                              f'connect it: d.edge("{k}", "<other id>"), or drop it if it is not part of the flow'))
        seen = {}
        for e in edges:
            key = (e.a, e.b)
            seen[key] = seen.get(key, 0) + 1
        for (a, b), n in seen.items():
            if n > 1:
                out.append(_issue("edge_duplicate", "warn", f"edge {a} → {b} is drawn {n} times.", None, n, 1,
                                  "keep one d.edge(...) per connection (a list fans out: d.edge(a, [b, c]))"))
        res = getattr(fig, "_res", None) or {}
        plates = getattr(fig, "label_boxes", None) or []
        labelled = [e for e, _ in res.get("paths", []) if getattr(e, "label", None)]
        boxes = getattr(fig, "boxes", None) or {}
        if dwg is not None and plates and len(plates) == len(labelled):
            M = _dwg_matrix(dwg, cv.w, cv.h)
            for e, pb in zip(labelled, plates):
                for k, nb in boxes.items():
                    ix = min(pb[2], nb[2]) - max(pb[0], nb[0])
                    iy = min(pb[3], nb[3]) - max(pb[1], nb[1])
                    if ix > 1 and iy > 1:
                        u = [round((pb[0] * M[0] + M[4]) / cv.w, 3), round((pb[1] * M[3] + M[5]) / cv.h, 3),
                             round((pb[2] * M[0] + M[4]) / cv.w, 3), round((pb[3] * M[3] + M[5]) / cv.h, 3)]
                        out.append(_issue("edge_label_on_node", "warn",
                                          f"the label {e.label!r} of edge {e.a} → {e.b} covers node {k!r}.", u,
                                          k, None, "shorten the label, move it with label_pos=0.3 / 0.7, or "
                                          "give the diagram more room (spacing=1.3)"))
                        break
    return out


_ABSTRACT = {"light", "lighting", "afternoon", "morning", "evening", "night", "noon", "dawn", "dusk", "sunset",
             "sunrise", "day", "mood", "atmosphere", "feel", "vibe", "style", "theme", "palette", "colour", "color",
             "colours", "colors", "tone", "tones", "look", "medium", "format", "background", "composition",
             "picture", "image", "illustration", "scene", "poster", "diagram", "chart", "drawing", "painting",
             "sketch", "render", "design", "layout", "detail", "details", "outside", "inside", "view", "shot",
             "silhouette", "product", "real-time", "realtime", "architecture", "grouped", "tier", "tiers",
             "dark", "bright", "soft", "warm", "cool", "deep", "gold", "blue", "red", "green", "night", "flyer",
             "card", "cover", "banner", "thumbnail", "wallpaper", "icon", "logo", "infographic", "dashboard",
             "flowchart", "mockup", "wireframe", "plan", "photo", "photograph", "art", "version"}
_ING_NOUNS = {"building", "ceiling", "painting", "string", "wedding", "evening", "morning", "clothing", "railing",
              "awning", "landing", "pudding", "dumpling", "earring", "ring", "wing", "king", "swing", "spring",
              "sibling", "duckling", "seedling", "sapling", "frosting", "icing", "lightning", "lighting",
              "messaging", "streaming", "billing", "shipping", "logging", "caching", "routing", "booking",
              "parking", "sterling", "herring", "viking", "pudding", "stuffing", "bedding", "sling", "thing"}
_LEAD = {"make", "draw", "create", "render", "show", "paint", "design", "generate", "give", "illustrate", "build",
         "keep", "picture", "imagine", "add", "include", "leave",
         "mock", "sketch", "want", "need", "please", "do"}
_TRAIL = {"called", "titled", "named", "entitled", "saying", "says", "reading"}
_ABSTRACT |= {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tonight",
              "weekend", "event", "jazz night", "pain", "posture", "emotion", "emotions", "feeling", "feelings",
              "energy", "motion", "movement", "speed", "silence", "peace", "love", "hope", "freedom", "joy",
              "sadness", "fear", "anger", "sense", "idea", "concept", "beauty", "contrast", "detail"}
_ARTICLES = {"a", "an", "the", "some", "his", "her", "its", "their", "my", "our", "your", "one", "two", "three",
             "four", "five", "six", "seven", "eight", "nine", "ten", "several", "many", "few"}
# words about the picture itself: never a subject of the request
_PICTURE = {"caption", "captions", "title", "titles", "subtitle", "label", "labels", "banner", "text", "texts",
            "words", "word", "background", "backdrop", "foreground", "midground", "scene", "picture", "image",
            "view", "style", "palette", "theme", "layout", "canvas", "sheet", "page", "heading", "headline",
            "slogan", "tagline", "wording", "typography", "font", "composition"}
# positions and collectives of a request, adjectives and adverbs a splitter leaves behind: never subjects
_PICTURE |= {"lines", "line", "section", "sections", "item", "items", "part", "parts", "list", "price", "prices",
             "edge", "side", "top", "bottom", "middle", "centre", "center", "corner", "left", "right", "surface",
             "distance", "end", "front", "back", "rim", "school", "flock", "herd", "pack", "group", "bunch", "pile",
             "stack", "row", "array", "collection", "lot", "bit", "couple", "pair", "dozen", "crowd", "number"}
_FORMAT_HEAD = {"chart", "graph", "plot", "diagram", "table", "infographic", "dashboard", "poster", "banner",
                "wallpaper", "thumbnail", "card", "cover", "flyer", "sheet", "menu", "screen", "mockup", "wireframe",
                "render", "illustration", "drawing", "painting", "sketch", "photo", "view", "shot", "workflow",
                "pipeline", "cutaway", "silhouette"}
_FILLER = {"labelled", "labeled", "exploded", "numbered", "seen", "shown", "open", "marked", "dashed", "called",
           "different", "three-quarter", "technical", "modern", "flat", "clean", "rainy", "elegant", "gentle",
           "seamless", "pale", "simple", "friendly", "minimalist", "stylised", "stylized", "subtle", "calm","it", "them", "they", "he", "she", "this", "that", "these", "those", "there", "here", "overhead", "asleep",
           "awake", "huge", "tiny", "big", "large", "little", "cosy", "cozy", "full", "up", "down", "out", "off",
           "outside", "inside", "away", "together", "apart", "autumn", "winter", "cute", "lovely", "nice", "very",
           "quite", "also", "again", "around", "nearby", "alone", "everywhere", "ahead", "away", "both", "all",
           "same", "other", "another", "such", "only", "just", "more", "most", "less", "like", "unlike",
           "about", "approximately", "roughly", "nearly", "almost", "exactly", "fully", "overall"}
_BOUNDARY = {"where", "when", "then", "which", "whose", "how", "every", "each", "than", "while", "as", "about",
             "but",
             "someone", "anywhere", "somewhere", "everywhere"}
_ABSTRACT |= {"jazz", "textbook", "storybook", "editorial", "studio", "kpi", "donut", "pie", "serif", "workflow", "pipeline", "3d", "isometric", "low-poly", "lowpoly", "cutaway", "poly", "low"}
# verbs (and their -s forms) that a noun-phrase splitter cannot tell from nouns
_VERBS = {"run", "runs", "ran", "build", "builds", "deploy", "deploys", "merge", "merges",
          "tied", "laid", "hung", "drawn", "written", "shown", "seen", "held", "kept", "given", "taken", "made",
          "covers", "lines", "fills", "frames", "crowns", "tops", "names", "lists", "marks",
          "fail", "fails", "trigger", "triggers", "send", "sends", "receive",
          "receives", "sit", "sits", "stand", "stands", "hold", "holds",
          "go", "goes", "flow", "flows", "connect", "connects", "wear",
          "wears", "carry", "carries", "read", "reads", "write", "writes", "make", "makes", "need", "needs",
          "want", "wants", "have", "has", "get", "gets", "use", "uses", "take", "takes", "come", "comes",
          "sleep", "sleeps", "walk", "walks", "eat", "eats", "drink", "drinks", "wait", "waits",
          "jump", "jumps", "swim", "swims", "smile", "smiles", "laugh", "laughs", "glow", "glows", "grow", "grows",
          "fall", "falls", "rise", "rises", "shine", "shines", "burn", "burns", "hang", "hangs",
          "lie", "lies", "lean", "leans", "approve", "approves",
          "ask", "asks", "tell", "tells", "see", "sees",
          "is", "are", "was", "were", "be", "been", "do", "does", "did", "will", "can", "should", "must"}
# a verb that is also a noun: kept as a subject when it stands first in its phrase or follows an article
_NOUN_VERBS = {"review", "reviews", "watch", "watches", "start", "starts", "check", "checks", "show", "shows",
               "point", "points", "push", "pushes", "run", "runs", "build", "builds", "stop", "stops", "pass",
               "passes", "look", "looks", "rest", "rests", "cross", "crosses", "float", "floats", "ship", "ships",
               "drink", "drinks", "walk", "walks", "fall", "falls", "stand", "stands", "swing", "hold", "can",
               "cans"}
# verbs a scene description uses after its subject ("a red toolbox sits on the floor", "five fish swim above"):
# a verb when the word before it is a noun; a noun after an article, a number or an adjective ("ten white pins")
_SCENE_VERBS = {"add", "bark", "bend", "bloom", "blow", "burn", "carry", "chase", "circle", "climb", "cluster",
                "contain", "crash", "crouch", "crowd", "cry", "curl", "cut", "dance", "dangle", "depict", "draw",
                "drift", "drink", "drive", "eat", "feature", "fill", "flank", "float", "fly", "follow", "gather",
                "gaze", "give", "glide", "glow", "graze", "greet", "grow", "guard", "hang", "hold", "hover",
                "huddle", "hug", "include", "jump", "keep", "kiss", "kneel", "laugh", "lean", "leap", "leave",
                "let", "lie", "look", "loom", "lurk", "make", "march", "meet", "occupy", "orbit", "pack", "paint",
                "parade", "pass", "peck", "peek", "peer", "perch", "play", "pour", "prop", "protect", "pull",
                "purr", "push", "put", "reach", "read", "reflect", "rest", "rise", "roam", "roll", "run", "say",
                "shine", "show", "sing", "sit", "sleep", "smile", "spill", "spread", "sprout", "squat", "stand",
                "stare", "stretch", "surround", "sweep", "swim", "tilt", "use", "wait", "walk", "wander", "watch",
                "wear", "wrap", "write"}


def _verb_forms(base):
    if base.endswith("y") and base[-2:-1] not in "aeiou":
        s = base[:-1] + "ies"
    elif base.endswith(("s", "sh", "ch", "x", "z", "o")):
        s = base + "es"
    else:
        s = base + "s"
    return {base, s}


_SCENE_VERB_FORMS = set().union(*(_verb_forms(b) for b in _SCENE_VERBS)) | {"line", "name", "stuck", "lies",
                                                                           "flies", "cut", "says",
                                                                           "holds", "sits", "stands", "hung"}


def _noun_like(w):
    """A word that can end a noun phrase (not an article, number, colour, size word, state or preposition)."""
    lw = w.lower()
    return bool(lw) and lw not in _ARTICLES and lw not in _STOP and not lw.isdigit() and lw not in _NUM \
        and _modifier(lw) is None and lw not in _PUNCT


def _verb_after(prev, w, nxt=None):
    """True when ``w`` is the verb of a clause: a scene verb right after a noun ("the toolbox sits", "hens peck"),
    or one followed by an article or a number ("above it swim five fish")."""
    lw = w.lower()
    if lw not in _SCENE_VERB_FORMS and lw not in _VERBS:
        return False
    if prev is not None and _noun_like(prev):
        return True
    return nxt is not None and (nxt.lower() in _ARTICLES or nxt.lower().isdigit() or nxt.lower() in _NUM)


# young animals and diminutives -> the species they are drawn as; species -> family word
_YOUNG = {"kitten": ("cat",), "kitty": ("cat",), "puppy": ("dog",), "pup": ("dog", "seal", "wolf"),
          "doggy": ("dog",), "foal": ("horse",), "colt": ("horse",), "filly": ("horse",), "pony": ("horse",),
          "calf": ("cow",), "chick": ("chicken",), "cub": ("bear", "lion", "tiger", "wolf", "fox", "panda"),
          "lamb": ("sheep",), "piglet": ("pig",), "duckling": ("duck",), "gosling": ("goose",),
          "fawn": ("deer",), "bunny": ("rabbit",), "owlet": ("owl",), "cygnet": ("swan",), "joey": ("kangaroo",),
          "tadpole": ("frog",), "hatchling": ("turtle", "bird"), "fledgling": ("bird",), "leveret": ("rabbit",),
          "eaglet": ("eagle",)}
_FAMILY = {"labrador": "dog", "corgi": "dog", "poodle": "dog", "husky": "dog", "dachshund": "dog", "shiba": "dog",
           "beagle": "dog", "bulldog": "dog", "terrier": "dog", "retriever": "dog", "spaniel": "dog",
           "puppy": "dog", "tabby": "cat", "siamese": "cat", "persian": "cat", "kitten": "cat",
           "pony": "horse", "stallion": "horse", "mare": "horse", "unicorn": "horse",
           "panda": "bear", "grizzly": "bear", "chick": "bird", "chicken": "bird", "duck": "bird", "penguin": "bird",
           "owl": "bird", "parrot": "bird", "flamingo": "bird", "eagle": "bird", "sparrow": "bird", "robin": "bird",
           "swan": "bird", "goose": "bird", "pigeon": "bird", "crow": "bird", "shark": "fish", "whale": "fish",
           "dolphin": "fish", "goldfish": "fish", "salmon": "fish", "triceratops": "dinosaur",
           "stegosaurus": "dinosaur", "ladybug": "insect", "ladybird": "insect", "bee": "insect",
           "butterfly": "insect", "beetle": "insect", "ant": "insect", "bunny": "rabbit", "hare": "rabbit",
           "lamb": "sheep", "ram": "sheep"}


def _unaccent(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", str(text)) if not unicodedata.combining(c))


def _tokens(text, split=False):
    """Lower-case words of ``text``; a hyphenated compound stays one word ("sign-ups") unless ``split``."""
    t = _unaccent(text).lower()
    if split:
        return re.findall(r"[a-z0-9]+", t)
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", t)


def _is_abs(t):
    """A word about the picture itself, a colour or a format: never a subject."""
    return any(c in _ABSTRACT or c in _PICTURE or c in _COLOURS for c in _singular(t))


def _forms(word):
    """Whole words a request word may stand for: itself, its singular forms and, for a young animal, its species."""
    out = []
    for c in _singular(word):
        out.append(c)
        out += list(_YOUNG.get(c, ()))
    return list(dict.fromkeys(out))


def _split_phrase(words):
    """Split one noun-phrase chunk at verbs; drop articles, numbers, picture words and request verbs."""
    groups, cur = [], []
    seq = list(words) + [None]
    for k, w in enumerate(seq):
        lw = None if w is None else w.lower().strip("'’-")
        nx = seq[k + 1].lower() if k + 1 < len(seq) and seq[k + 1] else None
        ing = lw is not None and len(lw) > 4 and lw.endswith("ing") and lw not in _ING_NOUNS and (
            (cur and _noun_like(cur[-1])) or nx is None or nx in _ARTICLES or nx in _DETS or nx in _STOP
            or nx in _PRONOUNS or nx.isdigit() or nx in _NUM)
        verb = lw is not None and (lw in _BOUNDARY or lw in _FILLER and lw in ("up", "down", "out", "off") or (
            ing
            or (lw in _VERBS and not (lw in _NOUN_VERBS and (not cur or cur[-1].lower() in _ARTICLES or
                                                             cur[-1].lower().endswith("ing"))))
            or _verb_after(cur[-1] if cur else None, lw, seq[k + 1]) or (lw.endswith("ed") and cur and
                                                                         not lw.endswith("eed") and
                                                                         _noun_like(cur[-1]) and len(lw) > 4)))
        if verb and lw in _STATE_WORDS and all(_modifier(c) is not None or c.lower() in _ARTICLES for c in cur):
            verb = False                                 # "a melting ice cream cone": a state in front of a thing
        if lw in _ARTICLES and lw not in _NUM and cur and _noun_like(cur[-1]):
            groups.append(cur)                           # "on the horizon a large ship"
            cur = [w]
            continue
        if lw is None or verb:
            if cur:
                groups.append(cur)
            cur = []
            continue
        cur.append(w)
    out = []
    for g in groups:
        if g and (any(c in _PICTURE or c in _ABSTRACT for c in _singular(g[-1])) or g[-1].lower() in _SPATIAL):
            continue                                     # "the far end", "a sunny afternoon"
        ws = [x for x in g if x.lower() not in _ARTICLES and not x.isdigit()]
        while ws and (ws[0].lower() in _STOP or ws[0].lower() in _LEAD):
            ws = ws[1:]
        while ws and ws[-1].lower() in _TRAIL:
            ws = ws[:-1]
        ws = [x for x in ws if not any(c in _PICTURE for c in _singular(x)) and x.lower() not in _FILLER
              and not (x.lower().endswith("ly") and len(x) > 4)]
        while len(ws) > 1 and ws[-1].lower().endswith("ed") and len(ws[-1]) > 4:      # "fox curled"
            ws = ws[:-1]
        if len(ws) == 1 and ws[0].lower().endswith("ed") and len(ws[0]) > 4:          # "deployed"
            ws = []
        if ws:
            out.append(ws)
    return out


def _chunks_of(plain):
    """Noun-phrase chunks (lists of words) from catalog's phrase tools, with a local fallback splitter."""
    try:
        from . import catalog as _cat
        if hasattr(_cat, "_chunks2"):
            chunks = [list(c[0]) for c in _cat._chunks2(plain)]
        else:
            chunks = [list(c) for c in _cat._chunks(plain)]
        if chunks:
            return chunks
    except Exception:                                    # noqa: BLE001 - catalog is advisory
        pass
    split = {"with", "and", "under", "over", "on", "in", "at", "beside", "behind", "near", "by", "of", "from",
             "into", "onto", "above", "below", "between", "for", "or", "next", "to", "while", "who", "that"}
    chunks, cur = [], []
    for w in re.findall(r"[A-Za-z][A-Za-z\-']*|[,.;:!?]", plain):
        if w in ",.;:!?" or w.lower() in split:
            if cur:
                chunks.append(cur)
            cur = []
        else:
            cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


# words in front of a noun that say how big it is or praise it: dropped from subjects
_SIZE_ADJ = {"small", "tiny", "little", "large", "big", "huge", "giant", "tall", "short", "long", "wide", "narrow",
             "thin", "thick", "enormous", "massive", "miniature", "mini", "medium", "oversized", "high", "low", "round",
             "square", "flat", "shiny", "calm", "soft", "cute", "beautiful", "lovely", "pretty", "nice", "simple",
             "cozy", "cosy", "gentle", "quiet", "peaceful", "elegant", "fancy", "plain", "single", "lone", "lonely",
             "whole", "entire", "real", "typical", "classic", "traditional", "famous", "various", "assorted",
             "distant", "nearby", "warm", "cool", "cold", "hot", "glossy", "matte", "smooth", "rough", "sharp",
             "heavy", "light", "dark", "bright", "pale", "deep", "vivid", "colourful", "colorful", "detailed",
             "realistic", "stylized", "stylised", "magnificent", "majestic", "graceful", "delicate", "sturdy",
             "busy", "crowded", "fresh", "new", "modern", "main", "central", "small-scale", "same", "half",
             "illustrated", "animated", "cartoon", "cartoonish", "stylish", "cheerful", "friendly", "funny",
             "clear", "plenty", "everything", "something", "anything", "nothing", "underwater", "fantasy",
             "cluttered", "tidy", "sunny", "rainy", "cloudy", "snowy", "foggy", "windy", "stormy", "far", "near",
             "summer", "spring", "winter", "autumn", "someone", "anywhere", "somewhere", "anyone", "everyone"}
# states and materials a judge checks and the report cannot see: printed with their thing as a reminder
_STATE_WORDS = {"cracked", "broken", "rusty", "rusted", "torn", "ripped", "wet", "soaked", "dry", "frozen", "melting",
                "melted", "burning", "burnt", "burned", "steaming", "smoking", "dripping", "glowing", "flickering",
                "dusty", "dirty", "muddy", "messy", "weathered", "worn", "crumpled", "wrinkled", "fallen", "spilled",
                "spilt", "overturned", "upturned", "tipped", "shattered", "smashed", "dented", "bent", "faded",
                "peeling", "crumbling", "rotten", "rotting", "ripe", "unripe", "half-eaten", "bitten", "sliced",
                "peeled", "empty", "full", "half-full", "open", "opened", "closed", "shut", "lit", "unlit", "dead",
                "wilted", "wilting", "blooming", "withered", "bare", "leafless", "snowy", "snow-covered", "icy",
                "frosty", "mossy", "moss-covered", "overgrown", "abandoned", "ruined", "old", "ancient", "vintage",
                "antique", "young", "elderly", "sleeping", "asleep", "smiling", "crying", "laughing", "angry", "sad",
                "happy", "tired", "injured", "wounded", "floating", "flying", "hanging", "coiled", "tangled",
                "folded", "stacked", "scattered", "woven", "knitted", "embroidered", "carved", "polished",
                "tarnished", "stained", "chipped", "patched", "boiled", "fried", "cooked", "raw", "glazed",
                "sparkling", "shining", "falling", "setting", "rising", "arriving", "departing", "approaching", "passing", "waiting", "rocky", "sandy", "leafy", "furry", "fluffy", "shaggy", "hairy", "wooly",
                "woolly", "striped", "spotted", "checkered", "polka-dot", "curly", "flooded", "burning",
                "lighted", "illuminated", "blurred", "dented", "cast-iron", "wrought-iron", "fresh-cut"}
_MATERIALS = {"wooden", "wood", "stone", "glass", "brick", "metal", "metallic", "iron", "steel", "velvet", "silk",
              "silken", "leather", "plastic", "paper", "cardboard", "ceramic", "porcelain", "clay", "marble",
              "wicker", "wool", "woolen", "woollen", "cotton", "denim", "bamboo", "copper", "brass", "bronze",
              "concrete", "crystal", "rubber", "straw", "lace", "felt", "tin", "chrome", "granite", "terracotta",
              "linen", "suede", "fur", "glass-topped", "stained-glass", "origami"}
_QUANTITY = {"bar", "bars", "coil", "coils", "piece", "pieces", "slice", "slices", "chunk", "chunks", "lump", "lumps", "scrap", "scraps", "strip",
             "strips", "handful", "pinch", "splash", "sliver", "wedge", "hunk", "morsel", "kind", "sort", "type",
             "variety", "amount", "plenty", "lots", "set", "pile", "heap", "bunch", "stack", "glimpse", "hint",
             "touch", "trace"}
# a state -> what to show (the line reads "show the crack"); others read "show it rusty"
_STATE_CUE = {"cracked": "the crack", "broken": "the break", "rusty": "the rust", "rusted": "the rust",
              "torn": "the tear", "ripped": "the tear", "wet": "the wet sheen and drips", "soaked": "the wet",
              "frozen": "the ice", "melting": "the melt and drips", "melted": "the melt", "burning": "the flame",
              "burnt": "the char", "burned": "the char", "steaming": "the steam", "smoking": "the smoke",
              "dripping": "the drips", "dusty": "the dust", "muddy": "the mud", "spilled": "the spill",
              "spilt": "the spill", "shattered": "the shards", "crumpled": "the creases", "weathered": "the wear",
              "mossy": "the moss", "snowy": "the snow", "snow-covered": "the snow", "icy": "the ice",
              "peeling": "the peeling", "overgrown": "the growth", "half-eaten": "the bite", "bitten": "the bite",
              "sliced": "the slices", "lit": "the flame", "glowing": "the glow", "wooden": "the wood grain",
              "wood": "the wood grain", "stone": "the stone", "brick": "the bricks", "glass": "the glass",
              "metal": "the metal", "velvet": "the velvet", "leather": "the leather", "paper": "the paper",
              "ceramic": "the glaze", "marble": "the marble veins", "wicker": "the weave", "woven": "the weave",
              "knitted": "the knit", "cast-iron": "the iron", "wrought-iron": "the iron", "striped": "the stripes",
              "spotted": "the spots", "coiled": "the coils", "chipped": "the chips", "dented": "the dents",
              "stained": "the stains", "faded": "the fading", "wilted": "the droop", "leafless": "the bare branches"}
_POS_MOD = {"left", "right", "top", "bottom", "front", "back", "middle", "upper", "lower", "far", "near", "rear"}
# relations and positions: never subjects ("below: not recorded" would be a false line)
_SPATIAL = {"below", "above", "left", "right", "beside", "behind", "front", "top", "bottom", "middle", "centre",
            "center", "near", "next", "under", "over", "between", "inside", "outside", "across", "along", "beneath",
            "underneath", "atop", "around", "back", "side", "corner", "upper", "lower", "foreground", "midground"}


def _modifier(word):
    """What a word in front of a noun is: "colour", "size" (size, praise, plain adjective), "state",
    "material", or None (a noun that is part of the name: "park" bench, "sports" car)."""
    w = word.lower().strip("'’-")
    if w in _NOUN_MODS:
        return None
    if _colour_of(w) is not None or w in _COLOUR_MODS or w in _COLOURS:
        return "colour"
    if w in _MATERIALS:
        return "material"
    if w in _STATE_WORDS:
        return "state"
    if w in _SIZE_ADJ or w in _FILLER or w in _POS_MOD or w.endswith("ly") and len(w) > 4:
        return "size"
    if w.endswith("ed") and len(w) > 4:
        return "state"                                   # "arched", "carved", "frosted"
    if w.endswith(("ful", "ous", "ish", "less", "able", "ible", "ive")) and len(w) > 5:
        return "size"
    return None


# words in front of a noun that belong to its name ("light bulb", "potted plant", "stuffed animal")
_NOUN_MODS = {"light", "potted", "stuffed", "canned", "red-haired",
              "animal", "musical", "electrical", "chemical", "medical", "tropical", "coral", "royal", "crystal-ball"}


def _subject_parts(words):
    """(thing words, states) of one noun phrase: "weathered wooden bench" -> (["bench"], ["weathered", "wooden"]);
    "shiny red sports car" -> (["sports", "car"], [])."""
    ws = list(words)
    while len(ws) > 1 and ws[-1].lower() in ("made",):
        ws = ws[:-1]
    head, mods = ws[-1], ws[:-1]
    keep, states = [], []
    for j, m in enumerate(mods):
        k = _modifier(m)
        if k == "colour" and j and mods[j - 1].lower() in _NO_COLOUR_BEFORE and m.lower() == "cream":
            k = None                                     # "ice cream cone"
        if m.lower() in _UNITS or m.isdigit():
            keep = []                                    # "a 12 inch ruler"
            continue
        if k in ("state", "material"):
            states.append(m.lower())
        elif k is None:
            keep.append(m)
        else:
            keep = []                                    # "sports car" stays whole, "red sports car" loses "red"
    return keep + [head], states


def _subject_list(prompt):
    """The cached ``_subject_list_of`` (fresh lists every call)."""
    return [(p, list(st)) for p, st in _subject_list_cached(prompt)]


@functools.lru_cache(maxsize=64)
def _subject_list_cached(prompt):
    return tuple((p, tuple(st)) for p, st in _subject_list_of(prompt))


# style keywords that also name things: a style only in a style construction ("in chalkboard style")
_STYLE_THINGS = {"chalkboard", "blackboard", "greenboard", "sticker", "decal", "mosaic", "clay", "collage", "comic",
                 "cartoon", "photo", "photograph", "sketch", "drawing", "painting", "embroidery", "stitch",
                 "blueprint", "schematic", "sprite", "zine", "polaroid", "diorama", "miniature", "chalk", "crayon",
                 "crayons", "pencil", "charcoal", "pastel", "stained glass", "church window", "scrapbook",
                 "banknote", "heatmap", "heat map"}


def _subject_list_of(prompt):
    """[(thing phrase, states)] of a request in its order: things only (no adjectives, verbs, size words,
    quantities); a state or material in front of a thing ("cracked teapot", "made of chocolate") is kept apart."""
    plain = prompt
    for q in quoted(prompt):
        plain = plain.replace(q, " ")
    plain = re.sub(r"[\"'“”‘’「」『』]\s*[\"'“”‘’「」『』]", " ", plain)
    plain = _unaccent(re.sub(r"(?<=\w)/(?=\w)", " ", _clean_request(plain)))
    made = {}                                            # "cake made of chocolate": chocolate is the cake's material
    for m in re.finditer(r"\b([A-Za-z][\w-]*)\s+(?:is\s+|are\s+)?made\s+(?:out\s+)?(?:of|from)\s+(?:an?\s+|the\s+)?"
                         r"([a-z][\w-]*)(?:\s+([a-z][\w-]*))?", plain, re.I):
        what = m.group(2).lower()
        if _modifier(what) is not None and m.group(3) and m.group(3).lower() not in _STOP:
            what = m.group(3).lower()                    # "made of dark chocolate"
        made[what] = m.group(1).lower()
    plain = re.sub(r"\bmade\s+(?:out\s+)?(?:of|from)\b", " with ", plain, flags=re.I)
    try:
        from . import stylize as _st
        style_words = {k.lower() for s in _st.STYLES.values() for k in s["keywords"] if k.isascii()}
    except Exception:                                    # noqa: BLE001
        style_words = set()
    out = []
    for ch in _chunks_of(plain):
        for ws in _split_phrase(ch):
            out.append(_subject_parts(ws))
    keep = []
    for thing, states in out:
        low = " ".join(thing).lower()
        toks = [t for t in low.split() if t not in _SPATIAL]
        if not toks or set(_singular(toks[-1])) & _STYLE_WORDS:
            continue                                     # "below", "in large letters": no thing to draw
        thing = [t for t in thing if t.lower() not in _SPATIAL]
        low = " ".join(toks)
        if toks[-1] in _QUANTITY or toks[-1] in _SIZE_ADJ or toks[-1] in _STATE_WORDS or \
                toks[-1] in _MATERIALS and len(toks) == 1 and toks[-1] in made:
            if toks[-1] in made and keep:                # "made of glass": a material of the thing before it
                for k in keep:
                    if made[toks[-1]] in k[0].lower().split() or _singular(made[toks[-1]])[-1] in k[2]:
                        k[1].append(toks[-1])
            continue
        if (low in style_words or toks[-1] in style_words) and (low not in _STYLE_THINGS or re.search(
                r"\b(?:in|as)\s+(?:an?\s+)?" + re.escape(low) + r"\b|\b" + re.escape(low) +
                r"[\s-](?:style|look|art|aesthetic|effect|painting|drawing|illustration|print|render)", plain, re.I)):
            continue                                     # "watercolour"; "a chalkboard" stays a thing
        if all(_is_abs(t) or t in _NOT_OBJECTS or t in _STOP for t in toks):
            continue
        if any(c in _FORMAT_HEAD for c in _singular(toks[-1])) and len(toks) > 1:
            continue                                     # "bar chart", "labelled diagram": a kind of picture
        heads = [_singular(t)[-1] for t in toks if t not in _STOP and not _is_abs(t)]
        if toks[-1] in made and keep:                    # "a cake made of chocolate": no separate chocolate
            owner = made[toks[-1]]
            k = next((k for k in keep if owner in k[0].lower().split() or _singular(owner)[-1] == k[2]), None)
            if k is not None:
                k[1].append(toks[-1])
                continue
        if heads and heads[-1] in [k[2] for k in keep]:
            k = next(k for k in keep if k[2] == heads[-1])          # "fresh snow" then "snow"
            k[1].extend(s for s in states if s not in k[1])
            continue
        if low not in [k[0].lower() for k in keep]:
            keep.append([" ".join(thing), list(dict.fromkeys(states)), heads[-1] if heads else low])
    # a state after its thing: "one cup broken into pieces", "one of the cups is cracked", "the gate is rusty"
    nums = "|".join(list(_NUM) + [r"\d+"])
    low = plain.lower()
    found = []
    for m in re.finditer(r"\b(?:(" + nums + r")\s+of\s+(?:them|the\s+([a-z]+))\s+(?:is|are|was|were)\s+|(" + nums +
                         r")\s+([a-z]+)\s+(?:is\s+|are\s+)?|the\s+([a-z]+)\s+(?:is|are|was|were|looks?)\s+)"
                         r"(?:(?:very|badly|slightly|half)\s+)?([a-z-]+)\b", low):
        state = m.group(6)
        if state not in _STATE_WORDS and state not in _MATERIALS:
            continue
        num = m.group(1) or m.group(3)
        noun = m.group(2) or m.group(4) or m.group(5)
        if not noun:                                     # "one of them is broken": the last plural thing named
            plural = [k for k in keep if k[0].lower().split()[-1] != _one(k[0].lower().split()[-1])]
            noun = (plural or keep)[-1][2] if keep else None
        found.append((num, noun, state))
    for num, noun, state in found:
        k = next((k for k in keep if noun and k[2] == _one(noun)), None)
        if k is None:
            continue
        tag = state
        if num and _NUM.get(num, int(num) if num.isdigit() else 0) and any(
                _one(w) == k[2] and n > (_NUM.get(num) or int(num) if not num.isdigit() else int(num))
                for _, n, ws, _ in count_phrases(prompt) for w in ws[:3]):
            tag = f"{state} ({num} of them)"
        if tag not in k[1] and state not in k[1]:
            k[1].append(tag)
    return [(k[0], k[1]) for k in keep]


def _phrases(prompt):
    """Subject phrases of a request (things only: quoted text, style, picture, verb, size and state words
    removed): ["cat", "park bench", ...]."""
    return [p for p, _ in _subject_list(prompt)]


def subjects(prompt):
    """The things a request names, in its order, each with the states and materials attached to it:
    ``[{"thing": "teapot", "states": ["cracked"]}, ...]``. Adjectives, verbs, size words and quantities
    ("small", "piece", "made", "wooden" on its own) are never things."""
    return [{"thing": p, "states": s} for p, s in _subject_list(prompt or "")]


_CAPTION_GROUPS = ("legend", "caption", "subtitle", "footer", "footnote", "note", "source", "credit", "annotation")


def _label_like(text, group=None):
    """True for a drawn string that labels a thing (node label, chart title, sign); false for legend / caption
    sentences: a group named legend / caption, an '=' key, a sentence."""
    g = str(group or "").lower()
    if any(k in g for k in _CAPTION_GROUPS):
        return False
    words = str(text).split()
    if "=" in text or len(words) > 7:
        return False
    if len(words) > 3 and (text.rstrip().endswith((".", "!", "?")) or text.count(",") >= 2):
        return False
    return True


def _label_texts(cv):
    """Drawn strings that are labels of things (see ``_label_like``)."""
    groups = {}
    try:
        for t in vector_texts(cv):
            groups.setdefault(_norm(t["text"]), t["group"])
    except Exception:                                    # noqa: BLE001
        pass
    return [t for t in drawn_texts(cv) if _label_like(t, groups.get(_norm(t)))]


_LABEL_KINDS = {"node", "group", "view", "screen", "chart", "annotation", "shape"}


# words a request uses about how the picture looks, what its text says, what the renderer adds, or the setting
# as a whole: never listed as "not recorded" (they are answered when something matches, skipped otherwise)
_STYLE_WORDS = {"bold", "type", "typeface", "typography", "font", "fonts", "serif", "sans", "sans-serif", "letter",
                "letters", "lettering", "outlined", "outline", "contrast", "accent", "accents", "circular", "round",
                "consistent", "gradient", "gradients", "shading", "finish", "hue", "hues", "saturation", "glossy",
                "matte", "minimal", "retro", "modern", "flat", "theme", "weight", "spacing", "margin", "margins",
                "alignment", "emphasis", "border", "borders", "corner", "corners", "height", "heights", "size",
                "sizes", "width", "length", "shape", "shapes", "proportion", "proportions", "scale", "angle",
                "perspective", "consistency", "harmony", "balance", "hierarchy", "whitespace", "space",
                "capital", "capitals", "caps", "uppercase", "lowercase"}
_CONTENT_WORDS = {"name", "names", "date", "dates", "time", "times", "location", "address", "english", "chinese",
                  "japanese", "korean", "french", "german", "spanish", "readme", "tagline", "greeting", "title",
                  "titles", "heading", "headings", "subtitle", "caption", "line-up", "lineup", "price", "prices",
                  "command", "commands", "text", "word", "words", "slogan", "number", "numbers", "url", "email",
                  "area", "areas", "value", "values", "label", "labels", "tag", "tags", "quote", "message", "copy",
                  "headline", "description", "translation", "language", "initials", "signature", "dimension",
                  "dimensions", "measurement", "measurements", "figure", "figures", "percentage", "total", "totals"}
_RENDER_WORDS = {"shadow", "shadows", "reflection", "reflections", "light", "lighting", "lights", "highlight",
                 "highlights", "glow", "shine", "sheen", "gloss", "bokeh", "depth", "ambient", "rim", "backlight",
                 "sunlight", "moonlight", "daylight", "glare", "exposure", "atmosphere", "mist", "haze"}
_SOFTWARE = {"app", "apps", "suite", "product", "tool", "tools", "software", "platform", "website", "site", "system",
             "project", "brand", "company", "startup", "business", "open-source", "dotfile", "dotfiles", "repo",
             "repository", "library", "package", "sdk", "cli", "command-line", "subscription"}
_SETTING = {"garden", "kitchen", "street", "village", "nook", "apartment", "town", "interior", "studio", "office",
            "landscape", "scene", "setting", "place", "cafe", "restaurant", "shop", "store", "market", "corner",
            "neighbourhood", "neighborhood", "countryside", "outdoors", "indoors", "home", "world", "environment",
            "cutaway", "spot", "area"}
# places a scene's setting block draws (new_scene.py SETTINGS): "park" is covered when a step named "setting" painted
_PLACES = {"park", "meadow", "field", "lawn", "yard", "backyard", "picnic", "hill", "hills", "countryside",
           "playground", "orchard", "beach", "sea", "ocean", "shore", "coast", "seaside", "pier", "harbour",
           "harbor", "island", "road", "city", "sidewalk", "pavement", "alley", "crosswalk", "downtown", "station",
           "plaza", "room", "bedroom", "bathroom", "classroom", "library", "bakery", "lounge", "night", "midnight"}
_GENRES = {"still life", "life drawing", "flat lay", "pin up", "pin-up"}          # a kind of picture, not a thing
_EVENT_TAIL = {"night", "evening", "morning", "afternoon", "weekend", "festival", "fest", "party", "concert",
               "session", "gala", "show", "jam"}          # "summer jazz night": the name of an event
_MATERIAL_BG = {"paper", "paper-like", "backdrop", "background", "seamless", "canvas", "sheet", "surface"}
# worn things and body parts: answered by the person / animal wearing them
_WORN = {"jacket", "coat", "hat", "cap", "apron", "shirt", "t-shirt", "dress", "skirt", "trousers", "pants", "jeans",
         "shoe", "shoes", "boot", "boots", "scarf", "glasses", "uniform", "hoodie", "sweater", "jumper", "glove",
         "gloves", "tie", "helmet", "hair", "beard", "moustache", "face", "smile", "eye", "eyes", "hand", "hands",
         "arm", "arms", "leg", "legs", "expression", "pose", "outfit", "clothes", "clothing", "costume", "backpack",
         "bag", "necklace", "earring", "earrings", "collar", "stripes", "stripe", "coat", "fur", "tail", "paw",
         "paws", "whiskers", "ears", "ear", "wing", "wings", "beak", "feather", "feathers"}
# more worn things: never a missing thing, never a place to put someone "in"
_WORN_MORE = {"ponytail", "braid", "braids", "hairstyle", "haircut", "suit", "robe", "gown", "kimono", "armour",
              "armor", "attire", "sleeve", "sleeves", "zipper", "zippers", "shoes", "sneakers", "jersey"}
# a request word -> words of things that answer to it (a part answers to its whole, near-synonyms)
_ANSWERED_BY = {"cup": ("mug", "teacup", "cup"), "mug": ("cup", "mug"), "sea": ("ocean", "sea", "water"),
                "ocean": ("sea", "ocean", "water"), "water": ("lake", "sea", "ocean", "pond", "river", "water",
                                                            "stream", "pool", "wave", "waves"),
                "lake": ("lake", "pond", "water"), "pond": ("pond", "lake", "water"), "boat": ("boat", "ship"),
                "ship": ("ship", "boat"), "rug": ("rug", "carpet", "mat"), "carpet": ("carpet", "rug"),
                "sofa": ("sofa", "couch"), "couch": ("couch", "sofa"), "book": ("book", "bookshelf", "bookcase"),
                "armrest": ("armchair", "chair", "sofa", "couch"), "sill": ("windowsill", "window"),
                "windowsill": ("windowsill", "sill", "window"), "swing": ("door",), "leader": ("annotation",
                                                                                            "callout", "leader"),
                "callout": ("annotation", "callout"), "steam": ("steam", "pot", "kettle", "cup", "mug"),
                "tile": ("tile", "kpi", "card"), "card": ("card", "tile"), "timer": ("timer", "ring", "clock"),
                "wave": ("wave", "waves", "breaking wave", "ocean"), "path": ("path", "road", "street"),
                "stars": ("star", "stars"), "star": ("star", "stars"), "tree": ("tree", "forest", "maple", "pine",
                                                                            "oak", "birch", "palm"),
                "flower": ("flower", "flowers", "bougainvillea", "rose", "tulip", "daisy")}
_KIND_WORDS = {"creature": ("animal", "creature", "pet"), "person": ("person", "people", "figure", "character",
                                                                      "friend"),
               "plant": ("plant",), "annotation": ("annotation", "callout", "leader", "label"),
               "chart": ("chart", "graph", "plot"), "view": ("view",), "screen": ("screen",)}
_TYPE_WORDS = {"kpi": ("kpi", "tile", "card", "metric"), "axes": ("chart", "graph", "plot")}
for _w in ("carrot", "tomato", "corn", "pepper", "mushroom", "pumpkin", "eggplant", "aubergine", "broccoli", "onion",
           "lettuce", "cabbage", "potato", "radish", "turnip", "garlic", "cucumber", "pea", "bean", "zucchini",
           "leek", "celery", "spinach", "kale", "beet", "chili", "squash"):
    _FAMILY.setdefault(_w, "vegetable")
for _w in ("apple", "banana", "orange", "pear", "grape", "grapes", "cherry", "lemon", "lime", "peach", "plum",
           "strawberry", "watermelon", "melon", "pineapple", "mango", "kiwi", "berry", "blueberry", "fig"):
    _FAMILY.setdefault(_w, "fruit")
# first words that make a compound a different thing: a "title block" never answers to "block"
_QUALIFIERS = {"title", "label", "text", "name", "caption", "tag", "price", "logo", "heading", "legend", "note",
               "scale", "dimension", "info", "status"}
# the part itself and its geometry: in a drafting drawing the views draw them
_GEOMETRY = {"block", "plate", "part", "body", "housing", "bracket", "shaft", "flange", "component", "piece",
             "assembly", "face", "edge", "side", "end", "surface", "centre", "center", "axis", "centreline",
             "centerline", "profile", "outline", "section", "web", "rib", "boss", "slot", "step", "shoulder",
             "groove", "keyway", "fillet", "radius", "thread", "counterbore", "countersink"}
# the last word of a recorded name that only names a part of a thing: "lamp stand" answers to "lamp",
# "cell wall" to "cell"; any other last word must itself match ("bus stop" never answers to "bus")
_PART_HEADS = {"stand", "shade", "base", "body", "top", "leg", "legs", "door", "doors", "handle", "frame", "wall",
               "walls", "centre", "center", "inner", "outer", "rim", "lid", "cap", "head", "trunk", "roof", "seat",
               "back", "arm", "arms", "window", "windows", "front", "side", "tip", "edge", "ring", "core", "case",
               "shell", "cover", "panel", "outline", "membrane", "pane", "glass", "light", "glow", "flame",
               "fire", "cushion", "post", "pole", "foot", "feet", "screen", "dial", "face"}


def _content_phrase(nouns):
    """True when a phrase names something that is not a separately drawable thing (see the word sets above)."""
    head = nouns[-1]
    hs = set(_singular(head)) | {head}
    if hs & (_STYLE_WORDS | _CONTENT_WORDS | _RENDER_WORDS | _SOFTWARE | _SETTING | _MATERIAL_BG):
        return True
    parts = [p for p in head.split("-") if p]
    if len(parts) > 1 and all(p in _COLOURS or p in ("to", "and", "or", "like") for p in parts):
        return True                                       # teal-to-navy
    if head.endswith("-like"):
        return True
    return False


def _proper_noun(word, prompt):
    """A capitalised word in the middle of the request (Taipei, README, Velvet): a name, not a thing to draw."""
    for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", _unaccent(prompt), re.I):
        w = m.group(0)
        if not (w[:1].isupper() and any(c.islower() for c in w[1:])):
            continue                                      # lower case, or an acronym (CI, API): a thing
        before = _unaccent(prompt)[:m.start()].rstrip()
        if before and before[-1] not in ".!?:;—(\"'":
            return True
    return False


def coverage(cv, prompt, rows=None):
    """For each subject phrase of the request, what answers to it: ``{"phrase", "status" ("ok" | "missing"),
    "by" (kind and name), "line"}`` in request order. An object answers by the last word of its name (or a part
    name such as "lamp stand" for "lamp"), an alias, a singular, young-animal or family word, its kind ("animal"
    for a creature) or a count entry; a worn thing by the person or animal wearing it; text answers when every
    word of the phrase is in one label of a thing. Words about the look, the text content, the renderer's light
    and shadow, the software or the setting as a whole, colour ranges and proper names are never listed as
    missing."""
    if not prompt:
        return []
    S = summary(cv) if rows is None else rows
    if not any(r.get("kind") == CUE for r in S):               # renderer cues answer light / shadow words
        S = list(S) + [r for r in summary(cv, cues=True) if r.get("kind") == CUE]

    def toks_of(text):
        return {c for x in _tokens(text, split=True) for c in _singular(x)}
    cands = []                                           # (display, all words, head words, aliases, kind)
    for s in S:
        if s["kind"] in ("style", "text"):
            continue
        nm = s["name"].split(" #")[0]
        words, heads = set(), set()
        for w in _entry_words(s) | {nm.lower(), _bare(nm)}:
            words |= toks_of(w)
            last = _tokens(_bare(w), split=True)
            if last:
                heads |= set(_singular(last[-1]))
                if last[-1] in _PART_HEADS and len(last) > 1:      # "lamp stand": a part of a lamp
                    heads |= set(_singular(last[-2]))
        meta = s.get("meta") or {}
        if meta.get("id"):
            words |= toks_of(str(meta["id"]))
        for t in _TYPE_WORDS.get(str(meta.get("type", "")).lower(), ()):
            words.add(t)
            heads.add(t)
        for t in _KIND_WORDS.get(s["kind"], ()):
            words.add(t)
            heads.add(t)
        for w in list(words):                            # species -> family word (corgi -> dog)
            if w in _FAMILY:
                words.add(_FAMILY[w])
                if w in heads:
                    heads.add(_FAMILY[w])
        nw = _tokens(_bare(nm), split=True)
        need = {w for w in nw[:-1] if w in _QUALIFIERS}           # a "title block" is no "block"
        cands.append((f'{s["kind"]} "{nm}"', words, heads, [str(a).lower() for a in s.get("aliases", []) or []],
                      s["kind"], need))
    wearers = [c for c in cands if c[4] in ("person", "creature")]
    views = [c for c in cands if c[4] == "view"]
    setting_rows = [(f'{s["kind"]} "{s["name"].split(" #")[0]}"',) for s in S
                    if str(s.get("step") or "").lower().startswith("setting") and s["kind"] not in ("text", "style")]
    texts = []
    for t in _label_texts(cv):
        d = f'text "{" ".join(t.split())[:40]}"'
        if d not in [x[0] for x in texts]:
            texts.append((d, toks_of(t)))
    out = []
    for p in _phrases(prompt):
        toks = [t for t in _tokens(p) if t not in _STOP]
        nouns = [t for t in toks if not _is_abs(t) and t not in _FILLER and not t.isdigit() and t not in _UNITS]
        if not nouns:
            continue
        if " ".join(toks) in _GENRES or (len(toks) > 1 and toks[-1] in _EVENT_TAIL):
            continue                                     # "still life", "summer jazz night": no object to draw
        head = _forms(nouns[-1])
        if "-" in nouns[-1]:                             # "sign-ups": the whole word, then its last part
            head += _forms(nouns[-1].split("-")[-1])
        wide = list(head)
        for h in head:
            wide += list(_ANSWERED_BY.get(h, ()))
        mods = [_singular(t) for n in nouns[:-1] for t in n.split("-") if len(t) > 2]
        best, score, via = None, 0, []
        ptoks = {x for t in _tokens(p, split=True) for x in _singular(t)}
        for c in cands:
            hit = [h for h in wide if h in c[2]]
            if c[5] and not (c[5] & ptoks):
                hit = []
            if c[4] in _LABEL_KINDS and c[4] not in ("annotation",):    # a node / label answers to all of the phrase
                every = all(set(_singular(w)) & c[1] for n in nouns for w in n.split("-") if w not in _CONTENT_WORDS)
                hit = [h for h in head if h in c[1]] or hit if every else []
            sc = (2 if hit else 0) + sum(1 for m in mods if any(x in c[1] for x in m))
            sc += (0.5 + (0.25 if c[4] in _SUBJECT else 0)) if sc else 0
            if hit and h_exact(head, c[2]):
                sc += 0.1
            if sc > score and hit:
                best, score, via = c, sc, hit
        if best is None and views and (ptoks & _UNITS or set(head) & _GEOMETRY):
            out.append({"phrase": p, "status": "ok", "by": views[0][0],
                        "line": f"{p} ✓ {views[0][0]} (drawn in the views)"})
            continue
        if best is None and wearers and set(head) & _WORN:
            best, via = wearers[0], [nouns[-1]]
            out.append({"phrase": p, "status": "ok", "by": best[0], "line": f"{p} ✓ {best[0]} (worn)"})
            continue
        if best is not None:
            lit = _singular(nouns[-1])
            if via[0] not in lit:
                al = f" ({nouns[-1]} = {via[0]})"
            else:
                al = next((f" (alias {a})" for a in best[3] if a in head and a not in best[0].lower()), "")
            out.append({"phrase": p, "status": "ok", "by": best[0], "line": f"{p} ✓ {best[0]}{al}"})
            continue
        words = [w for t in nouns for w in t.split("-")]
        hit = next((d for d, w in texts if all(set(_singular(t)) & w for t in words)), None)
        if hit is None and len(nouns) > 1:               # "launch date": the drawable words in one label
            core = [w for t in nouns for w in t.split("-") if not _content_phrase([w])]
            if core and core != words:
                hit = next((d for d, w in texts if all(set(_singular(t)) & w for t in core)), None)
        if hit:
            out.append({"phrase": p, "status": "ok", "by": hit, "line": f"{p} ✓ {hit}"})
            continue
        if set(_singular(nouns[-1])) & _PLACES and setting_rows:
            out.append({"phrase": p, "status": "ok", "by": setting_rows[0][0],
                        "line": f"{p} ✓ {setting_rows[0][0]} (drawn by the setting step)"})
            continue
        if _content_phrase(nouns) or _proper_noun(nouns[-1], prompt) or _is_marker(nouns[-1]):
            continue                                     # not a separately drawable thing: nothing to record
        name = _singular(nouns[-1])[-1]
        out.append({"phrase": p, "status": "missing", "by": None,
                    "line": f'{p} – not recorded: after painting it add manifest.note(cv, "object", "{name}", '
                            "mask=...)"})
    states = dict(_subject_list(prompt))
    for c in out:
        c["states"] = list(states.get(c["phrase"], []))
    return out


def state_lines(cov, limit=6):
    """One info line per drawn thing the request gives a state or material: "teapot (cracked): drawn as
    "teapot" - show the crack; the report cannot see states"."""
    out = []
    for c in cov:
        if c["status"] != "ok" or not c.get("states") or str(c.get("by", "")).startswith("text ") or \
                c["phrase"][:1].isupper():
            continue                                     # "The Velvet Room": a name, drawn as text
        plain_states = [st.split(" (")[0] for st in c["states"]]
        show = " and ".join(_STATE_CUE.get(st) or f"that it is {st}" for st in plain_states)
        by = c["by"].split(" ", 1)[-1] if c.get("by") else c["phrase"]
        some = [re.match(r"(\S+) \((\w+) of them\)", st) for st in c["states"]]
        if any(some):                                    # "cups, one of them broken"
            parts = [f"{m.group(2)} of them {m.group(1)}" if m else st for m, st in zip(some, c["states"])]
            out.append(f"{c['phrase']}, {', '.join(parts)}: drawn as {by} - show {show}; the report cannot see "
                       "states")
            continue
        out.append(f"{c['phrase']} ({', '.join(c['states'])}): drawn as {by} - show {show}; "
                   "the report cannot see states")
    return out[:limit]


def h_exact(head, words):
    """The phrase's own head word (not a near-synonym) is among the entry's head words."""
    return any(h in words for h in head)


def coverage_lines(cov, limit=10):
    """At most ``limit`` printable lines of ``coverage()``, the missing subjects first; one line when nothing is
    missing."""
    miss = [c for c in cov if c["status"] == "missing"]
    if not miss:
        return [f"all {len(cov)} subjects of the request are accounted for"] if cov else []
    ok = [c for c in cov if c["status"] != "missing"]
    if len(cov) <= limit:
        return [c["line"] for c in miss + ok]
    miss = miss[:limit - 1]
    ok = ok[:max(0, limit - 1 - len(miss))]
    return [c["line"] for c in miss + ok] + [f"… +{len(cov) - len(miss) - len(ok)} more (all of them are in --json)"]


_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "dozen": 12, "pair": 2}
_IRREG = {"people": "person", "children": "child", "mice": "mouse", "geese": "goose", "leaves": "leaf",
          "wolves": "wolf", "knives": "knife", "men": "man", "women": "woman", "teeth": "tooth", "feet": "foot",
          "shelves": "shelf", "loaves": "loaf", "cacti": "cactus", "fungi": "fungus"}


_UNITS = {"mm", "cm", "m", "km", "in", "inch", "inches", "ft", "feet", "foot", "metre", "metres", "meter", "meters",
          "square", "sq", "kg", "g", "lb", "lbs", "px", "pt", "pts", "percent", "degree", "degrees", "deg", "hour",
          "hours", "hrs", "hr", "h", "minute", "minutes", "min", "mins", "second", "seconds", "sec", "secs", "s",
          "ms", "fps", "bit", "bits", "kb", "mb", "gb", "tb", "x", "by", "am", "pm", "o'clock", "dpi", "ppi", "mph",
          "kph", "km/h", "kw", "w", "v", "ml", "l", "litre", "litres", "liter", "liters", "oz", "mile", "miles",
          "yard", "yards", "acre", "acres", "hectare", "hectares", "k", "%", "year-old", "years-old"}
_NUMBER_RE = re.compile(r"(?<![\w.:/\-Øø∅⌀×#$€£])(\d{1,3}|" + "|".join(_NUM) +
                        r")(?![\w.:/%\-×])\s+(?=[A-Za-z])", re.I)


def count_phrases(prompt):
    """(number text, n, following words, start) for each number in the request that counts things: "three
    apples", "12 months", "two bedrooms". Sizes, times, ratios and versions never count: "120 x 80 x 40 mm",
    "40 mm", "Ø52", "1:75", "21:00", "24 hours", "16-bit", "83%", "75 square metres"."""
    plain = _clean_request(prompt or "")
    out = []
    for m in _NUMBER_RE.finditer(plain):
        tok = m.group(1).lower()
        if re.search(r"\d\s*:\s*$", plain[:m.start()]):
            continue                                      # "1 : 75 scale bar": a ratio, never a count
        if re.search(r"\d\s*:\s*$", plain[:m.start()]):
            continue                                      # "1 : 75 scale bar": a ratio, never a count
        before = plain[:m.start()].split()
        raw = prompt or ""
        if raw[m.start() + len(m.group(1)):m.start() + len(m.group(1)) + 1] in ('"', "″", "'", "′") or \
                re.search(r"['’]s\s*$", plain[:m.start()]):
            continue                                      # "the baker's dozen": a name, not a count
        if before and before[-1].lower() in ("x", "×", "by", "of", "at", "to", "from", "-", "–"):
            if not (before[-1].lower() == "of" and tok.isalpha()):
                continue                                  # "80 x 40", "1 of 3", "at 9", "from 2 to 5"
        rest = plain[m.end():]
        words = re.findall(r"[A-Za-z][A-Za-z\-']*", re.split(r"[^A-Za-z\-'\s]", rest, maxsplit=1)[0])[:5]
        if not words or words[0].lower() in _UNITS:
            continue
        n = int(tok) if tok.isdigit() else _NUM[tok]
        out.append((m.group(1), n, words, m.start()))
    return out


def _bare(name):
    """A recorded name without its '#n' and trailing numbers / letters: 'BEDROOM 2' -> 'bedroom'."""
    ws = name.split(" #")[0].lower().split()
    while len(ws) > 1 and re.fullmatch(r"[0-9ivx]+|[a-z]", ws[-1]):
        ws = ws[:-1]
    return " ".join(ws)


def _count_words(s):
    """Words a recorded entry answers to when the request counts: its bare name and the name's last word, its
    aliases, and the words of its chart type (a KPI tile answers to "tile")."""
    nm = _bare(s["name"])
    out = {nm, nm.split()[-1]} if nm else set()
    ws = s["name"].split(" #")[0].lower().split()
    out.update(ws[i - 1] for i in range(1, len(ws)) if re.fullmatch(r"[0-9]+", ws[i]))   # "layer 1 mist"
    for a in s.get("aliases", []) or []:
        a = _bare(str(a))
        if a:
            out.update({a, a.split()[-1]})
    t = (s.get("meta") or {}).get("type")
    out.update(_TYPE_WORDS.get(str(t).lower(), ()))
    for w in list(out):
        if w in _FAMILY:
            out.add(_FAMILY[w])
    return out


def _is_title_of(s, num, word):
    """A recorded name that quotes the count ("Revenue, last 12 months"): it is about the things, not one."""
    low = s["name"].split(" #")[0].lower()
    return bool(re.search(r"(?<!\d)" + re.escape(str(num).lower()) + r"(?!\d)", low)) and \
        any(c in low for c in _singular(word))



def _singular(w):
    w = w.lower().strip("-'")
    if w in _IRREG:
        return [_IRREG[w]]
    c = [w]
    if w.endswith("ies") and len(w) > 4:
        c.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        c.append(w[:-2])
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        c.append(w[:-1])
    return c


def _entry_words(s):
    """Names an entry answers to: its name, the name's last word, its aliases."""
    nm = s["name"].split(" #")[0].lower()
    words = {nm, nm.split()[-1]} if nm else set()
    for a in s.get("aliases", []) or []:
        words.add(str(a).lower())
    return words


_WEAK_STYLE = {"vintage", "retro", "antique", "clay", "charcoal", "chalk", "ink", "pencil", "crayon", "pastel", "marker", "paper", "glass",
               "wood", "metal", "stone", "wax", "sand", "neon", "lego", "felt", "wool", "knit", "knitted", "gold"}
_STYLE_NOUN = r"(?:style|look|painting|illustration|drawing|print|render|rendering|art|artwork|medium|aesthetic|"               r"technique|filter|effect|sketch|photo|photograph|shot|animation|figurines?|figures?|sculpture)"
_COLOURS = {"red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "black", "white",
            "grey", "gray", "teal", "navy", "cyan", "magenta", "gold", "golden", "silver", "beige", "cream", "ivory",
            "crimson", "scarlet", "maroon", "turquoise", "indigo", "lilac", "amber", "olive", "charcoal", "sepia"}


def style_requested(prompt, name):
    """True when the request uses a keyword of style ``name`` as a style: a strong keyword ("watercolor") always;
    an ambiguous one ("clay", "charcoal") only in a construction: "in clay", "clay style", "as a charcoal sketch",
    never "a clay pot" or the colour pair "in orange and charcoal"."""
    try:
        from . import stylize
        kws = [k.lower() for k in stylize.STYLES[name]["keywords"] if k.isascii()]
        generic = {k.lower() for k in getattr(stylize, "_GENERIC", ())}
    except Exception:                                    # noqa: BLE001
        return True
    t = " " + " ".join(re.findall(r"[a-z0-9&'\-]+", str(prompt).lower())) + " "
    hits = [k for k in kws if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", t)]
    if not hits:
        return True                                      # a non-English keyword: pick() already decided
    for k in hits:
        if k not in generic and k not in _WEAK_STYLE:
            return True
        e = re.escape(k)
        for m in re.finditer(r"(?<![a-z0-9])" + e + r"(?![a-z0-9])", t):
            before, after = t[:m.start()].split(), t[m.end():].split()
            if after and re.fullmatch(_STYLE_NOUN, after[0]):
                return True
            if before and before[-1] in ("and", "or") or after and after[0] in ("and", "or") and                     len(after) > 1 and after[1] in _COLOURS:
                continue                                  # a colour pair
            if before and before[-1] in ("in", "as", "using", "with", "of") or \
                    len(before) > 1 and before[-2] in ("in", "as", "using", "with", "of") and before[-1] in ("a", "an", "the"):
                if after and after[0] in _COLOURS:
                    continue
                return True
    return False


def kits_used(cv):
    """Kit words the picture was drawn with ("clay", "solid"): a request word naming one is satisfied by it."""
    used = set(getattr(cv, "kits_used", ()) or ())
    for a in getattr(cv, "styles_applied", []) or []:
        used.add(str(a))
    return used


def _count_matches(rows, num, word):
    """The recorded (non-text) entries a counted request word answers to; for words such as months, layers or
    points only entries that carry an explicit ``count=``."""
    cands = _singular(word)
    match = [s for s in rows if s["kind"] != "text" and any(c in _count_words(s) for c in cands)
             and not _is_title_of(s, num, word)]
    if any(c in _NOT_OBJECTS for c in cands):
        match = [s for s in match if (s.get("meta") or {}).get("count")]
    return match


def requested_style(cv, prompt):
    """The style the request names as a style (stylize.pick + style_requested), or None; None too when the
    request names the kit that drew the picture ("clay figures" drawn with clay)."""
    try:
        from . import stylize
        want = stylize.pick(prompt)
    except Exception:                                    # noqa: BLE001
        return None
    if want and (not style_named(prompt, want) or not style_requested(prompt, want)):
        return None
    if want and want in set(getattr(cv, "kits_used", ()) or ()):
        return None
    return want


_STYLE_NAMES = {"watercolor": ("watercolor", "watercolour", "aquarelle"), "ukiyoe": ("ukiyo-e", "ukiyoe", "ukiyo e"),
                "pen_ink": ("pen and ink", "pen-and-ink"), "cel": ("cel shaded", "cel-shaded", "cel shading", "anime"),
                "oil": ("oil painting", "oil paint", "oil on canvas"), "paper_cut": ("paper cut", "papercut",
                                                                                     "paper-cut", "cut paper"),
                "colored_pencil": ("colored pencil", "coloured pencil"), "tilt_shift": ("tilt-shift", "tilt shift"),
                "low_poly": ("low poly", "low-poly", "lowpoly"), "pixel_art": ("pixel art", "pixel-art", "pixelart"),
                "comic": ("comic", "comic book", "comic-book"), "pencil": ("pencil drawing", "pencil sketch",
                                                                          "in pencil", "graphite"),
                "impressionist": ("impressionist", "impressionism"), "pointillism": ("pointillism", "pointillist"),
                "cross_stitch": ("cross-stitch", "cross stitch"), "noir": ("film noir", "noir"),
                "stained_glass": ("stained glass", "stained-glass"), "photo_finish": ("photo finish",)}


def style_named(prompt, name):
    """True when the request names the style ``name`` itself ("watercolour", "a comic-book page", "ukiyo-e"), not
    only a word stylize.pick links to it ("cyberpunk" is no request for synthwave, "scrapbook" none for collage)."""
    low = " " + " ".join(re.findall(r"[a-z0-9&'\-]+", _unaccent(str(prompt)).lower())) + " "
    forms = _STYLE_NAMES.get(name, (name.replace("_", " "), name.replace("_", "-"), name.replace("_", "")))
    return any(re.search(r"(?<![a-z0-9])" + re.escape(f) + r"(?![a-z0-9])", low) for f in forms)


def quoted_status(cv, prompt):
    """[(quoted string, drawn?, closest drawn text or None)] for every quoted string of the request."""
    texts = drawn_texts(cv)
    normed = [_norm(t) for t in texts]
    joined = " ".join(normed)
    out = []
    for q in quoted(prompt or ""):
        nq = _norm(q)
        if any(nq in t for t in normed) or nq in joined:
            out.append((q, True, None))
            continue
        close = difflib.get_close_matches(nq, normed, n=1, cutoff=0.0) if normed else []
        out.append((q, False, texts[normed.index(close[0])] if close else None))
    return out


def prompt_status(cv, prompt, rows=None):
    """What the PROMPT checks compared and found in order, as short phrases ("every quoted text is drawn",
    "the named style is applied", "counts match"): only checks the request gives material for and that passed."""
    if not prompt:
        return []
    S = summary(cv) if rows is None else rows
    done = []
    qs = quoted_status(cv, prompt)
    if qs and all(ok for _, ok, _ in qs):
        done.append("every quoted text is drawn")
    want = requested_style(cv, prompt)
    if want and want in (getattr(cv, "styles_applied", []) or []):
        done.append("the named style is applied")
    issues = {w["rule"] for w in prompt_checks(cv, prompt, S)}
    counted = any(_count_matches(S, num, w) for num, n, words, _ in count_phrases(prompt) for w in words[:3]
                  if w.lower() not in _STOP)
    if counted and "prompt_count" not in issues:
        done.append("counts match")
    return done


def prompt_checks(cv, prompt, rows=None):
    """Issues comparing the drawing with the request: quoted text, named style, counted objects."""
    if not prompt:
        return []
    out = []
    texts = drawn_texts(cv)
    normed = [_norm(t) for t in texts]
    joined = " ".join(normed)
    for q in quoted(prompt):
        nq = _norm(q)
        if any(nq in t for t in normed) or nq in joined:
            continue
        close = difflib.get_close_matches(nq, normed, n=1, cutoff=0.0) if normed else []
        near = texts[normed.index(close[0])] if close else None
        msg = f"The request quotes {q!r} but no drawn text contains it."
        if near is not None:
            msg += f" Closest drawn text: {near[:60]!r}."
        w = _issue("prompt_text_missing", "fail", msg, None, q, near,
                   f"draw it exactly: cv.text({q!r}, ...) — spelling, case and spaces as quoted")
        w["short"] = f"{q!r}: not drawn" + (f" (closest: {near[:30]!r})" if near is not None else "") +             " - draw it exactly as quoted"
        out.append(w)
    want = requested_style(cv, prompt)
    applied = list(getattr(cv, "styles_applied", []) or [])
    if want and not applied:
        look = want.replace("_", " ")
        out.append(_short(f"the request asks for a {want.replace('_', ' ')} look and no style was applied - apply "
                          f"{want!r} as the last step", _issue("prompt_style_missing", "warn",
                          f"The request asks for a {look} look (stylize.pick) but no style was applied.",
                          None, want, None,
                          f"preview it first: render.py SCENE --style {want} (it restyles the whole picture); keep it "
                          f'with stylize.apply(cv, "{want}") as the last step of render(), or, when you drew the '
                          f'{look} look by hand, record it: manifest.note(cv, "style", "{want}")')))
    elif want and want not in applied:
        out.append(_issue("prompt_style_other", "info",
                          f"The request reads as {want!r} but {', '.join(applied)} was applied.", None, applied, want,
                          f'use stylize.apply(cv, "{want}") if that was the intent'))
    # counts: "three apples", "2 dogs" (sizes, times and ratios never count)
    S = summary(cv) if rows is None else rows
    done = set()
    for num, n, words, _ in count_phrases(prompt):
        words = words[:3]
        if words and words[0].lower() == "of":
            words = words[1:]
        for w in words:
            if w.lower() in _STOP:
                break
            cands = _singular(w)
            match = _count_matches(S, num, w)
            if not match:
                continue
            key = cands[-1]
            if key in done:
                break
            done.add(key)
            have = sum(int((s.get("meta") or {}).get("count", 1) or 1) for s in match)
            if have != n:
                rec = "recorded entry answers" if have == 1 else "recorded entries answer"
                out.append(_short(f"{_one(w)}: {have} drawn, the request says {n}"
                                  + (" - remove the extra ones" if have > n else f" - draw all {n}"),
                                  _issue("prompt_count", "warn",
                                  f"The request says {num} {w} but {have} {rec} to it "
                                  f"({', '.join(s['name'] for s in match[:6])}).", None, have, n,
                                  f"draw exactly {n} {w}" + (" (remove the extra ones)" if have > n else
                                  f'; when all {n} are drawn, record the unnamed ones under that name '
                                  f'(manifest.note(cv, "object", "{_one(w)}", mask=...), or count={n} on '
                                  "one entry for a series)"))))
            break
    try:
        out += colour_checks(cv, prompt, S)[1] + place_checks(cv, prompt, S)[1] + single_count_issues(cv, prompt, S)
        out += text_checks(cv, prompt)[1] + size_checks(cv, prompt, S)
    except Exception:                                    # noqa: BLE001 - a parsing slip never breaks the report
        if os.environ.get("PIXELCANVAS_DEBUG"):
            raise
    unc = uncounted(prompt, S)
    if unc:
        ex = unc[0][1]
        out.append(_issue("prompt_uncounted", "info",
                          "The request counts things that no recorded object answers to, so the count cannot be "
                          f"checked: {', '.join(repr(p) for p, _ in unc[:6])}.", None, [p for p, _ in unc], None,
                          f'after painting each hand-drawn one call manifest.note(cv, "object", "{ex}", mask) '
                          "(kit calls record themselves)"))
    return out


_STOP = {"and", "a", "an", "the", "on", "in", "at", "of", "with", "by", "to", "from", "for", "or", "near", "under",
         "over", "into", "that", "which", "who", "is", "are", "as", "each", "every", "next", "beside", "behind"}
_NOT_OBJECTS = {"color", "colour", "colors", "colours", "px", "pixel", "pixels", "step", "steps", "point", "points",
                "layer", "layers", "time", "times", "hour", "hours", "minute", "minutes", "second", "seconds", "day",
                "days", "week", "weeks", "month", "months", "year", "years", "percent", "side", "sides", "way", "ways",
                "kind", "kinds", "type", "types", "style", "styles", "tone", "tones", "shade", "shades", "part",
                "parts", "column", "columns", "row", "rows", "word", "words", "line", "lines", "dimension", "frame",
                "frames", "fps", "mm", "cm", "km", "kg", "pt", "am", "pm", "d", "k", "x", "of"}


def uncounted(prompt, rows):
    """Number-plus-noun phrases of the request that no recorded (non-text) object answers to, as
    (phrase, singular noun) pairs: "three violins" -> ("three violins", "violin"). English nouns only."""
    if not prompt:
        return []
    known = set()
    for s in rows:
        if s["kind"] == "text":
            continue
        known |= _count_words(s)
    out, seen = [], set()
    skip = _NOT_OBJECTS | _PICTURE | _CONTENT_WORDS | _STYLE_WORDS | _RENDER_WORDS
    for num, _, words, _ in count_phrases(prompt):
        of = ""
        if words and (words[0].lower() in _VERBS or words[0].lower().endswith("ing") and
                      words[0].lower() not in _ING_NOUNS):
            continue                                     # "one holding an umbrella": a person, not a count
        if num.lower() in ("pair", "dozen") and words and words[0].lower() == "of":
            words, of = words[1:], " of"
        cut = []
        for w in words:
            if w.lower() in _STOP:
                break
            cut.append(w)
        cut = cut[:3]
        if not cut or cut[0].lower() in _NOT_OBJECTS:
            continue
        cands = [c for w in cut for c in _singular(w)]
        head = _singular(cut[-1])[-1]
        if head in skip or any(c in skip for c in _singular(cut[0])) or any(c in known for c in cands):
            continue
        if head not in seen:
            seen.add(head)
            out.append((f"{num}{of} {' '.join(cut)}", head))
    return out


def check(cv, prompt=None, rows=None):
    """Object checks plus, with a ``prompt``, the prompt checks — structured issues, most severe first."""
    S = summary(cv) if rows is None else rows
    out = object_checks(cv, S) + prompt_checks(cv, prompt, S)
    order = {"fail": 0, "warn": 1, "info": 2}
    out.sort(key=lambda w: order.get(w["severity"], 3))
    return out


# --------------------------------------------------------------------------------------------- literal request
# The request spells out colours, counts and places ("three red apples to the left of a blue cup"): the report
# compares each with what the final picture shows inside the recorded object's mask.

COLOUR_NAMES = ("red", "orange", "yellow", "green", "teal", "blue", "purple", "pink", "brown", "black", "white",
                "grey")                                  # what a painted colour is called in the report
_CI = {n: i for i, n in enumerate(COLOUR_NAMES)}
_SOFT_HUE = 4.0                                          # degrees: a hue this close to a border also counts as its neighbour


def _oklch(rgb):
    """sRGB (..., 3) in 0~1 -> (L, C, hue in degrees) in OKLab polar form: distances follow how colours look."""
    c = np.clip(np.asarray(rgb, np.float64), 0, 1)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return L, np.hypot(a, bb), np.degrees(np.arctan2(bb, a)) % 360.0


def _classify(L, C, h):
    """Index into COLOUR_NAMES for each (lightness, chroma, hue): the name a person gives the colour. Dark
    reds and oranges of low chroma are brown, light reds are pink, dark low-chroma tints are black."""
    L, C, h = np.asarray(L, np.float64), np.asarray(C, np.float64), np.asarray(h, np.float64) % 360.0
    pinkish = ((h >= 335) | (h < 8)) & (L >= 0.6)
    pinkish = pinkish | ((h >= 8) & (h < 28) & (L >= 0.78)) | ((h >= 320) & (h < 335) & (L >= 0.72))
    brown = ((h >= 28) & (h < 100) & (L < 0.62) & (C < 0.135)) | ((h >= 12) & (h < 28) & (L < 0.5) & (C < 0.1))
    brown = brown | (((h >= 335) | (h < 12)) & (L < 0.45) & (C < 0.08))
    brown = brown | ((h >= 28) & (h < 100) & (L >= 0.62) & (C >= 0.04) & (C < 0.075))
    ci = _CI
    cond = [(C < 0.045) & (L < 0.3), (C < 0.045) & (L >= 0.88), C < 0.045, (L < 0.2) & (C < 0.07),
            (L >= 0.95) & (C < 0.05), (L >= 0.9) & (C < 0.07) & (h >= 60) & (h < 115),
            pinkish, brown, (h >= 335) | (h < 45), h < 90, (h < 115) & (L >= 0.65), h < 100, h < 175, h < 215,
            h < 282]
    pick = [ci["black"], ci["white"], ci["grey"], ci["black"], ci["white"], ci["white"], ci["pink"], ci["brown"],
            ci["red"], ci["orange"], ci["yellow"], ci["orange"], ci["green"], ci["teal"], ci["blue"]]
    # rows 100~115 with lightness under 0.65 are olive: green (dark yellow below 100 is ochre: orange)
    cond = cond[:11] + [h < 100, h < 115, h < 175, h < 215, h < 282]
    pick = pick[:11] + [ci["orange"], ci["green"], ci["green"], ci["teal"], ci["blue"]]
    return np.select(cond, pick, default=ci["purple"]).astype(np.int8)


def colour_name(color):
    """The name (``COLOUR_NAMES``) a person gives a colour: '#c0392b' -> 'red', (0.2, 0.3, 0.9) -> 'blue'."""
    from .color import rgb as _rgb
    rgb = np.asarray(_rgb(color), np.float64)[:3]
    L, C, h = _oklch(rgb)
    return COLOUR_NAMES[int(_classify(L, C, h))]


# the colour words of a request -> the painted names that satisfy them; (names, max chroma or None)
_WARM_LIGHT = ("white", "yellow", "orange", "brown", "grey", "pink")
_COLOUR_WORDS = {
    "red": (("red",), None), "orange": (("orange",), None), "yellow": (("yellow",), None),
    "green": (("green",), None), "blue": (("blue",), None), "purple": (("purple",), None),
    "pink": (("pink",), None), "brown": (("brown",), None), "black": (("black",), None),
    "white": (("white",), None), "grey": (("grey",), None), "gray": (("grey",), None),
    "violet": (("purple",), None), "lilac": (("purple", "pink"), None), "lavender": (("purple", "pink", "blue"), None),
    "magenta": (("purple", "pink"), None), "indigo": (("purple", "blue"), None), "mauve": (("purple", "pink"), None),
    "crimson": (("red",), None), "scarlet": (("red",), None), "maroon": (("red", "brown"), None),
    "burgundy": (("red", "purple", "brown"), None), "navy": (("blue",), None), "teal": (("teal", "green", "blue"), None),
    "cyan": (("teal", "blue", "green"), None), "turquoise": (("teal", "blue", "green"), None),
    "aqua": (("teal", "blue", "green"), None), "gold": (("yellow", "orange", "brown"), None),
    "golden": (("yellow", "orange", "brown"), None), "silver": (("grey", "white"), None),
    "beige": (_WARM_LIGHT, 0.11), "ivory": (_WARM_LIGHT, 0.11), "khaki": (_WARM_LIGHT, 0.11),
    "cream": (_WARM_LIGHT, 0.11), "emerald": (("green", "teal"), None), "fuchsia": (("pink", "purple"), None),
}
_COLOUR_MODS = {"dark", "light", "pale", "bright", "deep", "vivid", "pastel", "soft", "rich", "dull", "muted",
                "neon", "hot", "sky", "baby", "royal", "midnight", "forest", "lime", "burnt", "warm", "cool",
                "faded", "light-", "dark-"}
_NO_COLOUR_BEFORE = {"ice", "whipped", "sour", "heavy", "double", "single"}       # "ice cream"
_NON_THING = {"hour", "hours", "ratio", "balance", "noise", "tone", "tones", "palette", "scheme", "theme", "accent",
              "accents", "gradient", "gradients", "photo", "photograph", "picture", "image", "illustration",
              "drawing", "painting", "sketch", "film", "movie", "style", "look", "colour", "color", "colours",
              "colors", "shade", "shades", "light", "lighting", "mood", "feel", "vibe", "version", "theme", "ink",
              "text", "lettering", "font", "typography", "title", "caption", "label", "logo", "border", "outline",
              "outlines", "contrast", "hues", "hue", "tint", "tints", "wash", "glow", "highlights", "highlight",
              "shadows", "shadow", "pattern", "dots", "spots"}
_FILLER_ADJ = {"large", "big", "small", "tiny", "little", "huge", "tall", "short", "round", "shiny", "glossy",
               "old", "new", "wooden", "ripe", "fresh", "cute", "fluffy", "striped", "spotted", "long"}
_AUX = {"is", "are", "was", "were", "sits", "sit", "sitting", "placed", "positioned", "located", "standing",
        "stands", "lying", "lies", "resting", "rests", "shown", "drawn", "appears", "seen", "stays", "stay", "be",
        "being", "situated", "put", "set", "hangs", "hanging", "floats", "floating", "painted", "should", "must",
        "will", "can", "to", "the", "placed", "lies", "perched", "positioned"}
_DETS = {"a", "an", "the", "some", "any", "several", "many", "few", "both", "each", "every", "this", "these",
         "those", "another", "other", "my", "his", "her", "its", "their", "our", "your", "all"}
_NP_BREAK = {"and", "or", "but", "with", "without", "on", "in", "at", "of", "by", "to", "from", "for", "near",
             "under", "over", "into", "onto", "beside", "behind", "next", "as", "while", "where", "when", "which",
             "that", "who", "whose", "is", "are", "was", "were", "has", "have", "than", "between", "above", "below",
             "beneath", "underneath", "atop", "against", "across", "along", "around", "inside", "outside", "toward",
             "towards", "facing", "there", "then", "also", "too", "left", "right", "top", "bottom", "front", "back",
             "side", "centre", "center", "middle", "upper", "lower", "while", "so", "if", "because", "like"}
_PUNCT = set(",.;:!?()")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[,.;:!?()]")


def _words(text):
    """Lower-case word and punctuation tokens of a request; quoted text is already blanked."""
    return [w.lower() for w in _WORD_RE.findall(_unaccent(text))]


def _colour_of(tok):
    """(the colour words, painted names, max chroma) a token stands for; None when it is no colour. ``dark-green``
    and ``blue-green`` are read part by part."""
    parts = tok.split("-")
    names, cmax, shown, mods = [], None, [], []
    for p in parts:
        if p in _COLOUR_MODS:
            mods.append(p)
            continue
        if p in _COLOUR_WORDS:
            ns, cm = _COLOUR_WORDS[p]
            names += list(ns)
            cmax = cm if cmax is None else cmax
            shown.append(p)
            continue
        return None
    if not shown:
        return None
    return {"word": "-".join(mods + shown) if mods else "-".join(shown), "names": list(dict.fromkeys(names)),
            "cmax": cmax, "mods": mods}


_COLOUR_NAMED = {"juice", "wine", "tea", "pepper", "sugar", "rice", "bread", "chocolate", "pudding", "dwarf",
                 "giant", "hole", "list", "market", "sheep", "widow", "hour", "smith", "frost", "belt"}


_PRONOUNS = {"it", "its", "them", "they", "him", "her", "his", "this", "that", "these", "those", "one", "ones",
             "itself", "themselves", "each", "both", "other", "others", "someone", "something", "everything",
             "anything", "nothing", "everyone", "anyone"}


def _np_forward(tokens, i):
    """Read a noun phrase from tokens[i:]: count, colour words, head noun (the last word before a break)."""
    n, colours, run, j = None, [], [], i
    mods = []
    while j < len(tokens):
        t = tokens[j]
        if t in _PUNCT:
            break
        if t == "and" and colours and j + 1 < len(tokens) and (
                _colour_of(tokens[j + 1]) or tokens[j + 1] in _COLOUR_MODS):
            j += 1
            continue
        if t in _NP_BREAK:
            break
        if t in _DETS:
            if run and t in _ARTICLES:
                break                                    # "above the door a wooden sign": a new phrase
            if t in ("a", "an") and n is None:
                n = 1
            j += 1
            continue
        if t.isdigit() and len(t) <= 3 or t in _NUM and t != "pair":
            n = int(t) if t.isdigit() else _NUM[t]
            j += 1
            continue
        if t in _COLOUR_MODS and j + 1 < len(tokens) and (
                _colour_of(tokens[j + 1]) or tokens[j + 1] in _COLOUR_MODS):
            mods.append(t)
            j += 1
            continue
        nxt = tokens[j + 1] if j + 1 < len(tokens) else None
        if run and (_verb_after(run[-1], t, nxt) or (t in _VERBS and t not in _NOUN_VERBS) or (t.endswith("ed") and len(t) > 4 and
                                                                     not t.endswith("eed") and
                                                                     t not in _STATE_WORDS)):
            break                                        # "a red toolbox sits on ...": the verb ends the phrase
        c = _colour_of(t)
        if c is not None and nxt in _COLOUR_NAMED and not colours:
            c = None                                     # "orange juice", "green tea": a name, not a colour
        if c is not None and not run[-1:] == ["ice"] and not (j > 0 and tokens[j - 1] in _NO_COLOUR_BEFORE):
            if mods:
                c = dict(c, word=" ".join(mods + [c["word"]]), mods=mods + c["mods"])
                mods = []
            colours.append(c)
            j += 1
            continue
        if run and len(t) > 4 and t.endswith("ing") and t not in _ING_NOUNS:
            break
        if t.endswith("ly") and len(t) > 4:
            j += 1
            continue
        run.append(t)
        j += 1
    nouns = [w for w in run if _modifier(w) is None]
    head = (nouns[-1] if nouns else run[-1]) if run else None
    if head is not None and head != run[-1]:
        run = run[:run.index(head) + 1] if head in run else run
    return {"n": n, "colours": colours, "run": run, "head": head, "end": j}


def _verb_like(w):
    """A verb that ends the phrase in front of it: "watching" in "a cat watching a bird on a branch"."""
    return (len(w) > 4 and w.endswith("ing") and w not in _ING_NOUNS) or (w in _VERBS and w not in _NOUN_VERBS)


def _np_before(tokens, k, owner=False):
    """The noun phrase that ends just before tokens[k] (a relation word): aux verbs are skipped first."""
    t = list(tokens[:k])
    while t and (t[-1] in _AUX or t[-1] in ("on", "at", "in", "that", "which", "who", "with", "says", "reads",
                                             "shows", "saying", "reading", "showing", "writes", "spells") or
                 (len(t) > 1 and _verb_after(t[-2], t[-1]) and t[-1] not in _ING_NOUNS)):
        t.pop()
    j = len(t)
    while j > 0 and t[j - 1] not in _PUNCT and not _verb_like(t[j - 1]) and not (
            j < len(t) and t[j - 1] in _ARTICLES and j - 1 > 0 and _noun_like(t[j - 2])) and (t[j - 1] not in _NP_BREAK or (
            t[j - 1] == "and" and j + 0 < len(t) and j - 2 >= 0 and _colour_of(t[j - 2]) and _colour_of(t[j]))):
        j -= 1
    np_ = _np_forward(t, j)
    st = j - 1
    if owner and np_["head"] and st >= 1 and t[st] in ("with", "of"):   # "a sailboat with a sail on the sea"
        o = _np_before(t, st)
        if o["head"] and _is_thing(o["head"]) and o["head"] not in _QUANTITY and o["head"] not in _NUM:
            return o
    return np_


def _head_keys(run):
    """Words a head noun may be recorded under: its singular forms, the two-word compound and request synonyms."""
    if not run:
        return []
    out = list(_singular(run[-1]))
    if len(run) > 1:
        out += [run[-2] + run[-1]] + [run[-2] + c for c in _singular(run[-1])]
    for c in list(out):
        out += list(_ANSWERED_BY.get(c, ()))
    return list(dict.fromkeys(out))


_NOT_THINGS = None


def _not_things():
    global _NOT_THINGS
    if _NOT_THINGS is None:
        _NOT_THINGS = (_NOT_OBJECTS | _PICTURE | _CONTENT_WORDS | _STYLE_WORDS | _RENDER_WORDS | _NON_THING |
                       {"colour", "color", "image", "picture", "scene"})
    return _NOT_THINGS


def _is_thing(head):
    if not head or head in _STOP or head in _DETS or len(head) < 2 or head.isdigit() or head in _PRONOUNS:
        return False
    if head.endswith(("'s", "’s", "'", "’")) or any(c in _ABSTRACT and c not in _COLOURS for c in _singular(head)):
        return False
    return not any(c in _not_things() for c in _singular(head))


def _one(word):
    """The singular of one English noun: "sandwiches" -> "sandwich", "strawberries" -> "strawberry",
    "tomatoes" -> "tomato", "children" -> "child", "glass" -> "glass"."""
    w = str(word).lower()
    if w in _IRREG:
        return _IRREG[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses", "zes")) or (w.endswith("oes") and len(w) > 4):
        return w[:-2]
    if w.endswith("ves") and w[:-3] + "f" in ("shelf", "loaf", "leaf", "wolf", "calf", "half", "scarf", "elf"):
        return w[:-3] + "f"
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


def _one_phrase(p):
    ws = str(p).lower().split()
    return " ".join(ws[:-1] + [_one(ws[-1])]) if ws else ""


_GREY = {"gray": "grey", "golden": "gold"}


def request_items(prompt):
    """What a request asks for, as a careful reader writes it down: ``{"things": [...], "counts": {head: n},
    "colours": {head: [colour words]}, "places": [(a, relation, b)], "text": [...]}``. Things keep the request's
    order and their compound names ("park bench"); counts, colours and places use the head noun; relations are
    "left of", "right of", "above", "below", "on", "in", "next to", or "at" with a part of the frame. Negated
    things ("no handle", "none of them red") ask for nothing."""
    p = prompt or ""
    things = []
    for d in subjects(p):
        t = _one_phrase(d["thing"])
        if t and t not in things:
            things.append(t)
    counts = {}
    toks = _words(_clean_request(p))
    for num, n, words, _ in count_phrases(p):
        ws = [w.lower() for w in words]
        if not ws or (ws[0] == "of" and num.lower() not in ("pair", "dozen")):
            continue                                     # "one of its arms"
        if ws[0] == "of":
            ws = ws[1:]
        np_ = _np_forward(ws, 0)
        if np_["head"] and _is_thing(np_["head"]) and np_["head"] not in _UNITS:
            counts.setdefault(_one(np_["head"]), n)
    colours = {}
    for r in colour_requests(p):
        k = _one(r["head"])
        colours.setdefault(k, [])
        for c in r["colours"]:
            for w in c["word"].split("-") if all(_colour_of(x) for x in c["word"].split("-")) else [c["word"]]:
                w = _GREY.get(w, w)
                if w not in colours[k]:
                    colours[k].append(w)
    # "two swans, one black and the other white", "four pawns, one red, one blue ...": colours of the members
    last_plural = last_noun = None
    for i, t in enumerate(toks):
        if t in _PUNCT:
            continue
        if _noun_like(t) and _one(t) != t and _is_thing(t):
            last_plural = _one(t)
        if t == "one" and i >= 2 and _colour_of(toks[i - 1]) and toks[i - 2] in ("a", "an", "the") and last_noun:
            w = _GREY.get(_colour_of(toks[i - 1])["word"], _colour_of(toks[i - 1])["word"])
            colours.setdefault(last_noun, [])                # "a second ball, a green one"
            if w not in colours[last_noun]:
                colours[last_noun].append(w)
        if _noun_like(t) and _is_thing(t) and t != "one":
            last_noun = _one(t)
        if t in ("one", "other", "another", "first", "second", "third", "fourth") and i + 1 < len(toks):
            c = _colour_of(toks[i + 1])
            after = toks[i + 2] if i + 2 < len(toks) else "."
            if c and last_plural and (after in _PUNCT or after in ("and", "or")):
                w = _GREY.get(c["word"], c["word"])
                colours.setdefault(last_plural, [])
                if w not in colours[last_plural]:
                    colours[last_plural].append(w)
    places = []
    for r in place_requests(p):
        a = _one(r["a"]["head"])
        if r["kind"] == "rel":
            pl = (a, _REL_WORD[r["rel"]], _one(r["b"]["head"]))
        else:
            pl = (a, "at", " ".join(x for x in r["where"] if x))
        if pl not in places:
            places.append(pl)
    return {"things": things, "counts": counts, "colours": {k: v for k, v in colours.items() if v},
            "places": places, "text": quoted(p)}


def colour_requests(prompt):
    """Colours the request attaches to things: ``[{"colours": [...], "phrase", "head", "run", "n", "pos"}]`` for
    "a red apple", "two green bottles", "a yellow bus with a black roof", "the apples are red", "an apple that
    is red". Colours of quoted text and of picture words ("blue hour", "black and white photo") are skipped."""
    plain = _clean_request(prompt or "")
    toks = _words(plain)
    out, used = [], set()
    i = 0
    while i < len(toks):
        t = toks[i]
        starts = _colour_of(t) is not None or (t in _COLOUR_MODS and i + 1 < len(toks) and _colour_of(toks[i + 1]))
        if not starts or i in used or (i > 0 and toks[i - 1] in _NO_COLOUR_BEFORE):
            i += 1
            continue
        # back up over a determiner / number / plain adjectives that open the phrase: "three large red apples"
        s = i
        while s > 0 and (toks[s - 1] in _DETS or toks[s - 1] in _NUM or toks[s - 1].isdigit()
                         or toks[s - 1] in _FILLER_ADJ) and toks[s - 1] not in _PUNCT:
            s -= 1
        np_ = _np_forward(toks, s)
        first = i
        if np_["colours"] and np_["head"] and _is_thing(np_["head"]) and np_["end"] > i:
            used.update(range(i, np_["end"]))
            out.append({"colours": np_["colours"], "head": np_["head"], "run": np_["run"], "n": np_["n"],
                        "pos": first, "form": "before"})
            i = np_["end"]
            continue
        # "the apples are red", "an apple that is red", "a cup, which is blue"
        j = i
        while j > 0 and toks[j - 1] in ("very", "quite", "bright", "deep", "dark", "light"):
            j -= 1
        if j > 0 and toks[j - 1] in ("is", "are", "was", "were", "looks", "look", "painted", "coloured", "colored",
                                     "turned", "appears"):
            k = j - 1
            while k > 0 and toks[k - 1] in ("that", "which", "who", ",", "now", "all", "both"):
                k -= 1
            subj = _np_before(toks, k)
            st = subj["end"] - len(subj["run"]) - 1
            while st >= 0 and toks[st] in _DETS:
                st -= 1
            if st >= 1 and toks[st] == "of":                # "the roof of the house is red": the roof
                owner = _np_before(toks, st)
                if owner["head"] and _is_thing(owner["head"]) and owner["head"] not in _QUANTITY:
                    subj = owner
            if subj["head"] and _is_thing(subj["head"]) and subj["end"] > 0:
                cs = _np_forward(toks, i)["colours"] or ([_colour_of(t)] if _colour_of(t) else [])
                out.append({"colours": cs, "head": subj["head"], "run": subj["run"], "n": subj["n"],
                            "pos": first, "form": "after"})
                used.add(i)
        i += 1
    for o in out:
        o["phrase"] = " ".join([" and ".join(c["word"] for c in o["colours"])] + o["run"])
    return out


def _aligned(cv, rows=None):
    """(entries, rows) of the same objects in the same order, renderer cues left out."""
    E = [e for e in entries(cv) if e.get("kind") != CUE]
    S = summary(cv, E) if rows is None else rows
    by = {e["order"]: e for e in E}
    pairs = [(by[r["order"]], r) for r in S if r.get("order") in by]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _rows_for(rows, run):
    """The recorded objects (not text, style or cues) a head noun answers to."""
    keys = set(_head_keys(run))
    if not keys:
        return []
    out = []
    for i, s in enumerate(rows):
        if s["kind"] in ("text", "style", CUE):
            continue
        if keys & _count_words(s):
            out.append(i)
    return out


def _accepted(c):
    """Painted names a requested colour accepts: its own, plus near-white for 'light', near-black for 'dark'."""
    names = list(c["names"])
    if {"light", "pale", "pastel"} & set(c["mods"]):
        names.append("white")
    if {"dark", "deep", "midnight"} & set(c["mods"]) or c["word"] == "navy":
        names.append("black")
    return list(dict.fromkeys(names))


class _Paint:
    """The colours inside one recorded object's solid mask, named. ``share(word)`` is the share of its pixels a
    colour word accepts; ``top`` the largest named cluster."""

    def __init__(self, L, C, h, rgb):
        self.n = int(L.size)
        self.L, self.C, self.h, self.rgb = L, C, np.asarray(h) % 360.0, rgb
        prim = _classify(L, C, h)
        lo, hi = _classify(L, C, h - _SOFT_HUE), _classify(L, C, h + _SOFT_HUE)
        self.acc = np.zeros((self.n, len(COLOUR_NAMES)), bool)
        idx = np.arange(self.n)
        for a in (prim, lo, hi):
            self.acc[idx, a] = True
        self.prim = prim
        self.hist = np.bincount(prim, minlength=len(COLOUR_NAMES)) / max(self.n, 1)
        self.top = COLOUR_NAMES[int(np.argmax(self.hist))] if self.n else None
        if self.n:
            sel = prim == int(np.argmax(self.hist))
            self.hex = "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.median(rgb[sel], axis=0))
        else:
            self.hex = None

    def share(self, c):
        """Share of the pixels the requested colour ``c`` (from ``_colour_of``) accepts, lightness variants
        ("light blue" also accepts a near-white blue, "dark green" a near-black green) included."""
        cols = [_CI[n] for n in _accepted(c)]
        hit = self.acc[:, cols].any(1)
        hit = hit | self._shade(set(_accepted(c)))
        if c["cmax"] is not None:
            hit = hit & (self.C < c["cmax"])
        return float(hit.mean()) if self.n else 0.0

    def _shade(self, names):
        """Pixels a person still calls the requested colour when the light has dulled it: the shaded side of a red
        or orange thing, a white thing in shadow or under warm light, a black thing with a tint."""
        P, L, C, h = self.prim, self.L, self.C, self.h
        ci = _CI
        out = np.zeros(self.n, bool)
        if "red" in names:
            out |= (P == ci["brown"]) & ((h < 32) | (h > 335))
        if "orange" in names:
            out |= (P == ci["brown"]) & (h >= 32) & (h < 80) & (C >= 0.08)
        if "white" in names:
            out |= ((P == ci["grey"]) & (L >= 0.5)) | (np.isin(P, [ci["brown"], ci["yellow"], ci["orange"], ci["pink"]])
                                                        & (L >= 0.62) & (C < 0.075))
        if "black" in names:
            out |= ((P == ci["grey"]) & (L < 0.4)) | ((L < 0.28) & (C < 0.08))
        return out

    def mostly(self):
        """'green' or 'green and brown': the one or two largest clusters (at least a fifth each)."""
        order = np.argsort(-self.hist)
        out = [COLOUR_NAMES[int(order[0])]]
        if self.hist[order[1]] >= 0.2:
            out.append(COLOUR_NAMES[int(order[1])])
        return " and ".join(out)


def _paint_of(cv, E, i, ctx):
    """_Paint of entry ``E[i]`` read from the final picture, or (None, reason). Pixels hidden by later objects,
    thin edges and soft shadow are left out."""
    e = E[i]
    if e.get("_rect"):
        return None, "no mask"
    m = e["_mask"]
    if "img" not in ctx:
        ctx["img"] = _small_image(cv, m.shape)
        ctx["lch"] = _oklch(ctx["img"])
    solid = m >= _OCCLUDE
    if not solid.any():
        solid = m >= _OWN
    hide = np.zeros_like(solid)
    for j in range(i + 1, len(E)):
        o = E[j]
        if o["_mask"].shape != m.shape or o["kind"] in _CONTAINER or is_backdrop(o) or is_see_through(o):
            continue
        hide |= o["_mask"] >= _OCCLUDE
    free = solid & ~hide
    if int(free.sum()) >= max(6, 0.25 * int(solid.sum())):
        solid = free
    for _ in range(2):                                   # leave the outline band out (two cells deep when it fits)
        core = solid.copy()
        core[1:, :] &= solid[:-1, :]
        core[:-1, :] &= solid[1:, :]
        core[:, 1:] &= solid[:, :-1]
        core[:, :-1] &= solid[:, 1:]
        if int(core.sum()) < 12:
            break
        solid = core
    if int(solid.sum()) < 3:
        return None, "too small"
    L, C, h = ctx["lch"]
    sel = solid
    if int(solid.sum()) >= 30:                           # and the darkest and lightest tenth: shade and highlight
        lo, hi = np.percentile(L[solid], [10, 90])
        tail = solid & (L >= lo) & (L <= hi)
        if int(tail.sum()) >= 12:
            sel = tail
    return _Paint(L[sel], C[sel], h[sel], ctx["img"][sel]), None


def _a(word):
    return ("an " if word[:1] in "aeiou" else "a ") + word


def colour_checks(cv, prompt, rows=None):
    """Compare each colour the request attaches to a thing with the colour painted inside the recorded object:
    ``(results, issues)``. A result: ``{"phrase", "thing", "colours", "requested", "recorded", "status" ("ok" |
    "wrong" | "not recorded"), "instances": [{"name", "painted", "share", "hex", "ok"}], "line"}``. The request
    is met when the colour covers a third of the object's solid pixels or is its largest cluster, so an object
    with two legitimate colours is never flagged for the smaller one."""
    reqs = colour_requests(prompt) if prompt else []
    if not reqs:
        return [], []
    E, S = _aligned(cv, rows)
    ctx = {}
    paints = {}

    def paint(i):
        if i not in paints:
            paints[i] = _paint_of(cv, E, i, ctx) if E else (None, "no mask")
        return paints[i]
    by_head = {}
    for r in reqs:
        by_head.setdefault(_one(r["head"]), []).append(r)
    results, issues = [], []
    for key, rs in by_head.items():
        idx = _rows_for(S, rs[0]["run"])
        wanted = {tuple(c["word"] for c in r["colours"]) for r in rs}
        assign = {}
        if len(wanted) > 1 and idx:                       # "a red apple and a green apple": best fit per colour
            left = list(idx)
            need = {id(r): (r["n"] or 1) for r in rs}
            for r in sorted(rs, key=lambda r: -(r["n"] or 1)):
                sc = []
                for i in left:
                    p, _ = paint(i)
                    sc.append((min(p.share(c) for c in r["colours"]) if p else -1, i))
                sc.sort(reverse=True)
                take = [i for _, i in sc[:need[id(r)]]]
                assign[id(r)] = take
                left = [i for i in left if i not in take]
        seen = set()
        for r in rs:
            sig = tuple(c["word"] for c in r["colours"])
            if sig in seen and not assign:
                continue
            seen.add(sig)
            mine = assign.get(id(r), idx)
            thing = key
            cw = " and ".join(c["word"] for c in r["colours"])
            recorded = sum(int((S[i].get("meta") or {}).get("count", 1) or 1) for i in mine)
            res = {"phrase": r["phrase"], "thing": thing, "colours": [c["word"] for c in r["colours"]],
                   "requested": r["n"], "recorded": recorded, "status": "ok", "instances": []}
            if not mine:
                res["status"] = "not recorded"
                res["line"] = f"{thing}: {cw} requested, {thing} not recorded, colour not checked"
                results.append(res)
                continue
            bad, unchecked = [], 0
            for i in mine:
                p, why = paint(i)
                inst = {"name": S[i]["name"]}
                if p is None:
                    unchecked += 1
                    inst.update(painted=None, share=None, ok=None, note=why)
                else:
                    shares = [p.share(c) for c in r["colours"]]
                    if len(shares) > 1:                   # "red and white": each is part of it
                        ok = all(s >= 0.08 for s in shares)
                    else:
                        ok = shares[0] >= 0.20 or p.top in _accepted(r["colours"][0])   # warn under a fifth only
                    desc = p.top
                    if ok and len(shares) == 1 and p.top not in _accepted(r["colours"][0]):
                        desc = f"partly {cw} ({shares[0] * 100:.0f}%, mostly {p.top})"
                    inst.update(painted=p.mostly(), top=p.top, desc=desc, hex=p.hex, share=round(min(shares), 3),
                                ok=bool(ok))
                    if not ok:
                        bad.append((S[i]["name"], p, min(shares)))
                res["instances"].append(inst)
            if unchecked == len(mine):
                res["status"] = "not recorded"
                res["line"] = f"{thing}: {cw} requested, {thing} recorded by box only (no mask), colour not checked"
                results.append(res)
                continue
            k = len(mine)
            tag = f"{thing} x{recorded}" if recorded > 1 else thing
            if bad:
                res["status"] = "wrong"
                names = [b[0] for b in bad]
                p0 = bad[0][1]
                who = names[0] if len(names) == 1 else f"{', '.join(names[:3])}{' ...' if len(names) > 3 else ''}"
                res["line"] = f"{tag}: {cw} requested, {who} painted mostly {p0.mostly()} ✗"
                msg = (f"the request says {_a(r['phrase']) if (r['n'] or 1) == 1 and not r['phrase'].endswith('s') else r['phrase']}"
                       f", the {who} {'is' if len(names) == 1 else 'are'} painted mostly {p0.mostly()} "
                       f"({cw} covers {bad[0][2] * 100:.0f}% of it)")
                issues.append(_short(
                    f"{who}: painted {p0.mostly()}, the request says {cw} - recolour its solid areas {cw}", _issue(
                        "prompt_colour", "warn", msg[0].upper() + msg[1:] + ".", None, p0.top, cw,
                        f"recolour the {thing} {cw} (paint its solid areas {cw}, keep highlights and shadow thin), "
                        f'or record the right object under that name: manifest.note(cv, "object", "{thing}", '
                        "mask=...)")))
            else:
                lead = next((x.get("top") for x in res["instances"] if x.get("top")), None)
                shares = [x["share"] for x in res["instances"] if x.get("share") is not None]
                extra = ""
                if lead and len(r["colours"]) == 1 and lead not in _accepted(r["colours"][0]):
                    extra = f" ({cw} covers {min(shares) * 100:.0f}% of it, mostly {lead})"
                res["line"] = f"{tag}: {cw} ✓{extra}"
                if unchecked:
                    res["line"] += f" ({unchecked} of {k} recorded by box only, not checked)"
            results.append(res)
    return results, issues


# ---- places: left / right / above / below and corners
_REL_RE = re.compile(
    r"\b(?:(?:to|on|at|in) the )?(left|right)(?:[- ]?hand)?(?: side)? of\b"
    r"|\b(above|over|higher than)\b|\b(below|underneath|beneath|under|lower than)\b"
    r"|\b(on top of|atop|on)\b|\b(inside|within|in)\b(?! front\b)|\b(next to|beside|alongside)\b")
_ABS_RE = re.compile(
    r"\b(?:on|at|in|toward|towards|to)\s+the\s+(?:(top|bottom|upper|lower)(?:[\s-]+(left|right|centre|center|middle))?"
    r"|(left|right|centre|center|middle))(?:[- ]hand)?(?:\s+(?:side|corner|part|half|portion|edge))?"
    r"(?:\s+of\s+the\s+(?:image|picture|frame|canvas|scene|photo|photograph|page|poster|screen))?(?!\s+of\b)")
_REL_WORD = {"left": "left of", "right": "right of", "above": "above", "below": "below", "on": "on", "in": "in",
             "next": "next to"}
_CONTACT_RELS = ("on", "in", "next")                     # checked only when both things are recorded
_NOT_CONTAINER = {"suit", "armor", "armour", "robe", "gown", "kimono", "uniform", "costume", "dress", "coat",
                  "jacket", "outfit", "clothes", "style", "rain", "snow", "fog", "mist", "light", "sun", "shade",
                  "shadow", "sky", "air", "wind", "dark", "darkness", "distance"}


def place_requests(prompt):
    """Places the request states: ``{"kind": "rel", "a", "b", "rel"}`` ("a cup to the right of the apples") and
    ``{"kind": "abs", "a", "where"}`` ("a bird in the top left corner", "a cup on the right")."""
    plain = _clean_request(prompt or "")
    low = _unaccent(plain).lower()
    out = []
    spans = []
    for m in _REL_RE.finditer(low):
        rel = m.group(1) or ("above" if m.group(2) else "below" if m.group(3) else "on" if m.group(4) else
                             "in" if m.group(5) else "next")
        if rel in ("on", "in") and re.match(r"\s*(?:the\s+)?(?:left|right|top|bottom|middle|centre|center|corner|"
                                            r"foreground|background|distance|front|back|side|sky)\b", low[m.end():]):
            continue                                     # "on the left", "in the top corner": a place in the frame
        if m.group(3) == "under" and re.match(r"\s*(?:the\s+)?(?:light|moon|sun|stars?|sky|rain|snow|water|cover)", low[m.end():]):
            continue
        before = _words(low[:m.start()])
        a = _np_before(before, len(before), owner=True)
        after = _words(low[m.end():])
        b = _np_forward(after, 0)
        if not a["head"] and before and before[-1] == "and" and out and out[-1]["kind"] == "rel" and \
                len(before) > 1 and before[-2] not in _PUNCT:
            a = out[-1]["a"]                             # "a turtle below a jellyfish and above a starfish"
        elif not a["head"] and b["head"]:                # "Above the bench hangs a clock": the thing comes after
            k = b["end"]
            while k < len(after) and (after[k] in (",", "there", "is", "are", "was", "were", "sits", "stands")
                                      or after[k] in _SCENE_VERB_FORMS or after[k] in _VERBS):
                k += 1
            if k > b["end"] or (k < len(after) and (after[k] in _ARTICLES or after[k].isdigit() or
                                                     after[k] in _NUM)):
                a = _np_forward(after, k)
        if not (a["head"] and b["head"]) or not _is_thing(a["head"]) or not _is_thing(b["head"]) or \
                a["head"] in _PRONOUNS or b["head"] in _PRONOUNS:
            continue
        if rel in _CONTACT_RELS and set(_singular(b["head"])) & (_NOT_CONTAINER | _WORN | _WORN_MORE):
            continue                                     # "a knight in armour", "a cat in the rain"
        spans.append(m.span())
        out.append({"kind": "rel", "a": a, "b": b, "rel": rel})
    for m in _ABS_RE.finditer(low):
        if any(s[0] <= m.start() < s[1] or s[0] < m.end() <= s[1] for s in spans):
            continue
        v = m.group(1) or ""
        hz = m.group(2) or m.group(3) or ""
        v = {"upper": "top", "lower": "bottom"}.get(v, v)
        hz = {"center": "centre", "middle": "centre"}.get(hz, hz)
        if not v and hz == "":
            continue
        if v in ("top", "bottom") and hz == "centre":
            hz = ""
        if v in ("", "top", "bottom") and hz == "" and not v:
            continue
        a = _np_before(_words(low[:m.start()]), len(_words(low[:m.start()])))
        if not a["head"] or not _is_thing(a["head"]):
            continue
        out.append({"kind": "abs", "a": a, "where": (v, hz)})
    return out


def _box_px(s, W, H):
    b = s.get("solid_bbox") or s["bbox"]
    return b[0] * W, b[1] * H, b[2] * W, b[3] * H


def _direction(ba, bb, rel_threshold=0.1):
    """Which way box ``ba`` lies from ``bb``: 'left' | 'right' | 'above' | 'below' | None (too close to tell).
    The offset of the centres is shrunk by a tenth of the mean sizes, then the larger axis decides."""
    ca = ((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2)
    cb = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
    da = (ba[2] - ba[0], ba[3] - ba[1])
    db = (bb[2] - bb[0], bb[3] - bb[1])
    rev = []
    for k in (0, 1):
        o = ca[k] - cb[k]
        rev.append(math.copysign(max(abs(o) - rel_threshold * (da[k] + db[k]) / 2, 0.0), o))
    if abs(rev[0]) == 0 and abs(rev[1]) == 0:
        return None
    if abs(rev[0]) > abs(rev[1]):
        return "right" if rev[0] > 0 else "left"
    return "below" if rev[1] > 0 else "above"


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _contact_check(S, ia, ib, an, bn, rel, W, H):
    """"on" (above with contact), "in" (inside the other's box) and "next to" (beside, close) between recorded
    things: (result, issues). Holds when every instance of ``an`` stands that way to one instance of ``bn``."""
    text = f"{an} {_REL_WORD[rel]} {bn}"
    tol_y, tol_x = 0.015 * H, 0.01 * W
    bad = []
    for i in ia:
        a = _box_px(S[i], W, H)
        best = None
        for j in ib:
            b = _box_px(S[j], W, H)
            ox = min(a[2], b[2]) - max(a[0], b[0])
            if rel == "on":
                gap = b[1] - a[3]                        # a's bottom above b's top: floating
                if a[1] > b[3] - tol_y or (a[1] + a[3]) / 2 > b[3]:
                    why = f"the {S[i]['name']} is below the {bn}"
                elif ox < -tol_x:
                    why = f"the {S[i]['name']} is beside the {bn}, not on it"
                elif gap > tol_y:
                    why = f"the {S[i]['name']} floats {gap / H * 100:.0f} % of the height above the {bn}"
                else:
                    why = None
            elif rel == "in":
                cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
                inter = max(0.0, ox) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                area = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
                inside = (b[0] - tol_x <= cx <= b[2] + tol_x and b[1] - tol_y <= cy <= b[3] + tol_y) or inter > 0.5 * area
                why = None if inside else f"the {S[i]['name']} is outside the {bn}"
            else:
                gx = max(0.0, a[0] - b[2], b[0] - a[2])
                gy = max(0.0, a[1] - b[3], b[1] - a[3])
                near = gx <= 0.6 * max(a[2] - a[0], b[2] - b[0]) + tol_x and \
                    gy <= 0.6 * max(a[3] - a[1], b[3] - b[1]) + tol_y
                why = None if near else f"the {S[i]['name']} is {max(gx / W, gy / H) * 100:.0f} % of the canvas " \
                                        f"away from the {bn}"
            if why is None:
                best = None
                break
            best = best or why
        if best:
            bad.append(best)
    if not bad:
        return {"text": text, "status": "holds", "detail": "", "line": f"{text}: holds ✓"}, []
    why = bad[0] + (f" (+{len(bad) - 1} more)" if len(bad) > 1 else "")
    fix = {"on": f"set the {an} on the {bn}: its bottom edge on the {bn}'s top, overlapping it sideways",
           "in": f"move the {an} inside the {bn} (its centre within the {bn}'s outline)",
           "next": f"move the {an} beside the {bn}, less than its own width away"}[rel]
    w = _issue("prompt_relation", "warn", f"The request puts the {an} {_REL_WORD[rel]} the {bn}, but {why}.", None,
               rel, rel, fix)
    w["short"] = f"{why[4:]}, the request says {_REL_WORD[rel]} - {fix}"
    return {"text": text, "status": "fails", "detail": why, "line": f"{text}: fails ✗ ({why})"}, [w]


def text_place_requests(prompt):
    """Quoted text the request places in the frame: [(text, (vertical, horizontal))] for "'ORCHARD' in large
    letters at the bottom", "the title 'SALE' at the top left"."""
    out = []
    p = prompt or ""
    for q in quoted(p):
        k = p.find(q)
        if k < 0:
            continue
        rest = p[k + len(q) + 1:]
        rest = re.split(r"[\"“”「」『』;.!?]|,\s|\band\b", rest, maxsplit=1)[0][:80]
        m = _ABS_RE.search(_unaccent(rest).lower())
        if not m or len(rest[:m.start()].split()) > 5:
            continue
        v = {"upper": "top", "lower": "bottom"}.get(m.group(1) or "", m.group(1) or "")
        hz = {"center": "centre", "middle": "centre"}.get(m.group(2) or m.group(3) or "", m.group(2) or m.group(3) or "")
        if v and hz == "centre":
            hz = ""
        if v or hz:
            out.append((q, (v, hz)))
    return out


def _text_boxes(cv, q):
    """Boxes (px) and letter sizes (px) of the drawn text that contains the quoted string ``q``."""
    nq = _norm(q)
    out = []
    for t in getattr(cv, "text_log", None) or []:
        if nq and nq in _norm(str(t.get("text", ""))):
            out.append((tuple(t["bbox"]), float(t.get("size") or 0)))
    try:
        for t in vector_texts(cv):
            if nq and nq in _norm(t["text"]):
                out.append((tuple(t["bbox"]), float(t["size"])))
    except Exception:                                    # noqa: BLE001
        pass
    return out


def text_checks(cv, prompt):
    """Quoted text drawn too small to read (letters under 3 % of the canvas height) or away from the place the
    request gives it: (results, issues)."""
    results, issues = [], []
    H, W = cv.h, cv.w
    places = dict(text_place_requests(prompt))
    for q in dict.fromkeys(quoted(prompt or "")):
        boxes = _text_boxes(cv, q)
        if not boxes:
            continue
        pct = max(s for _, s in boxes) / H * 100
        if pct < 3.0:
            w = _issue("prompt_text_small", "warn",
                       f"The quoted text {q!r} is drawn with letters {pct:.1f} % of the canvas height: too small to "
                       "read at a glance.", list(boxes[0][0]), round(pct, 2), 3.0,
                       "draw it larger: letters at least 4 % of the canvas height (size= or a larger box=)")
            w["short"] = f"{q!r}: letters are {pct:.1f} % of the height - make them at least 4 %"
            issues.append(w)
        if q in places:
            v, hz = places[q]
            label = " ".join(x for x in (v, hz) if x)
            b = max(boxes, key=lambda t: t[1])[0]
            u, y = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
            ok = (v != "bottom" or y > 2 / 3) and (v != "top" or y < 1 / 3) and \
                 (hz != "left" or u < 0.5) and (hz != "right" or u > 0.5) and (hz != "centre" or abs(u - 0.5) < 0.2)
            line = f"{q!r} at the {label}: " + ("holds ✓" if ok else f"fails ✗ (its centre is at ({u:.2f}, {y:.2f}))")
            results.append({"text": f"{q!r} at the {label}", "status": "holds" if ok else "fails", "line": line,
                            "detail": "" if ok else f"centre at ({u:.2f}, {y:.2f})"})
            if not ok:
                region = {"bottom": "the bottom third (centre y > 0.67)", "top": "the top third (centre y < 0.33)"}
                want = " and ".join(([region[v]] if v else []) + ([f"the {hz} half"] if hz else []))
                w = _issue("prompt_text_place", "warn",
                           f"The request puts {q!r} at the {label}, but its centre is at ({u:.2f}, {y:.2f}).",
                           [round(b[0] / W, 3), round(b[1] / H, 3), round(b[2] / W, 3), round(b[3] / H, 3)],
                           [round(u, 2), round(y, 2)], label, f"move it into {want or 'that part of the canvas'}")
                w["short"] = f"{q!r} is centred at ({u:.2f}, {y:.2f}), the request says {label} - move it there"
                issues.append(w)
    return results, issues


def size_checks(cv, prompt, rows=None, cov=None, checked=None):
    """Things the request names that are drawn too small to be seen (under 0.5 % of the canvas) or mostly hidden
    (under half visible): structured issues with the size printed. For a checked request (``checked_request``)
    also a named thing cut by the frame (``prompt_cut``), one more than a tenth covered by another named thing the
    request does not relate it to (``prompt_covered``), and one object over a third of the frame while three or
    more named things are under 3 % (``prompt_crowded``)."""
    S = summary(cv) if rows is None else rows
    cov = coverage(cv, prompt, S) if cov is None else cov
    checked = checked_request(prompt) if checked is None else checked
    out, seen, named = [], set(), []
    for c in cov:
        tail = c["line"].split("✓", 1)[-1]
        if c["status"] != "ok" or not c.get("by") or any(k in tail for k in ("(worn)", "(drawn ", " = ")):
            continue                                     # worn things, views, setting steps, aliases of a part
        kind, _, name = c["by"].partition(" ")
        name = name.strip('"')
        if kind not in ("object", "plant", "creature", "person") or name in seen:
            continue
        seen.add(name)
        if name.lower() in _SCATTER or name.lower() in _DETAIL or _is_marker(name):
            continue
        inst = [s for s in S if s["name"].split(" #")[0] == name and not s.get("backdrop")]
        if not inst:
            continue
        first = name.lower().split()[0]
        if name.lower().split()[-1] in _PART_HEADS and len(name.split()) > 1:     # "lamp stand": the whole lamp
            inst = [s for s in S if s["name"].split(" #")[0].lower().split()[:1] == [first]
                    and not s.get("backdrop")] or inst
            parts = inst
            area = sum(s["area_pct"] for s in parts)
            bb = _union([s["bbox"] for s in parts])
            side = max((bb[2] - bb[0]) * cv.w, (bb[3] - bb[1]) * cv.h) / min(cv.w, cv.h)
        else:
            area = max(s["area_pct"] / max(1, int((s.get("meta") or {}).get("count", 1) or 1)) for s in inst)
            side = max(max(s["size"][0] * cv.w, s["size"][1] * cv.h) for s in inst) / min(cv.w, cv.h)
        thing = c["phrase"]
        if set(_singular(thing.split()[-1])) & _RENDER_WORDS:
            continue                                     # a shadow or a glow is light, not a thing to enlarge
        named.append((thing, name, inst, area))
        hw = _one(thing.lower().split()[-1])
        if area < 0.5 and side < 0.1 and not _part_here(hw, prompt, S) and hw not in _SCATTER and \
                hw not in _DETAIL and confident_thing(thing, prompt, S) and \
                hw not in {"drop", "droplet", "crumb", "bead", "seed", "spark", "candle", "dot", "speck", "bubble"}:                    # small and short: a thin tall lamp still reads
            w = _issue("prompt_small", "warn",
                       f"The request names the {thing}, drawn at {area:.2f} % of the canvas: too small to be seen.",
                       inst[0]["bbox"], round(area / 100, 5), 0.005,
                       f"draw the {thing} large and recognisable: at least 3 % of the canvas")
            w["short"] = f"{thing}: {area:.2f} % of the canvas - draw it at least 3 % or it will not be seen"
            w["names"] = [s["name"] for s in inst]
            out.append(w)
            continue
        vis = max(s["visible_pct"] for s in inst)
        hb = next((s["hidden_by"] for s in inst if s["hidden_by"]), None)
        if vis < 50 and not any(s.get("see_through") for s in inst) and confident_thing(thing, prompt, S) and \
                not (hb and _staged(max(inst, key=lambda s: s["visible_pct"]), hb, S, thing, narrow=True, prompt=prompt)):
            w = _issue("prompt_hidden", "warn",
                       f"The request names the {thing}, but only {vis:.0f} % of it is visible"
                       + (f" (behind {hb})." if hb else "."), inst[0]["bbox"], round(vis / 100, 3), 0.5,
                       f"draw the {thing} after {hb} or move it" if hb else f"draw the {thing} later or move it")
            w["short"] = f"{thing}: only {vis:.0f} % visible" + (f" (behind {hb})" if hb else "") + \
                f" - draw it after {hb or 'what covers it'} or move it"
            w["names"] = [s["name"] for s in inst]
            out.append(w)
    if not checked:
        return out
    # a checked request: every named thing whole, inside the frame, with room
    names = {n for _, n, _, _ in named} | {c["by"].partition(" ")[2].strip('"') for c in cov if c.get("by")}
    for thing, name, inst, area in named:
        cut = max(inst, key=lambda s: s["clipped_pct"])
        b = cut["bbox"]
        e = 0.002
        sides = [n for n, hit, run in (("left", b[0] <= e, b[3] - b[1]), ("right", b[2] >= 1 - e, b[3] - b[1]),
                                       ("top", b[1] <= e, b[2] - b[0]), ("bottom", b[3] >= 1 - e, b[2] - b[0]))
                 if hit and run >= 0.05]                 # touching an edge along more than a sliver
        whole = cut["kind"] != "plant" and max(cut["size"]) < 0.5      # one object, not a group, a plant or ground
        hw = thing.lower().split()[-1]
        group = _one(hw) != hw or len(inst) > 1 or hw in _CUT_OK or _part_here(hw, prompt, S)
        portrait = (cut["kind"] in ("person", "creature") or _one(hw) in _FIGURE_WORDS) and set(sides) <= {"bottom"}
        if (cut["clipped_pct"] > 3 or sides and whole) and not _is_band(cut) and not group and not portrait and \
                confident_thing(thing, prompt, S):
            sides = sides or ["edge"]
            w = _issue("prompt_cut", "warn", f"The request names the {thing}, and {cut['clipped_pct']:.0f} % of it is "
                       f"outside the canvas ({'/'.join(sides)} edge).", b, cut["clipped_pct"] / 100, 0.03,
                       f"move the {thing} inside the frame")
            w["short"] = f"{thing}: cut off at the {'/'.join(sides)} edge - move it inside the frame"
            w["names"] = [s["name"] for s in inst]
            out.append(w)
            continue
        for s in inst:
            hb = s.get("hidden_by")
            cover = 100 - s["visible_pct"]
            if not hb or cover <= 10 or s.get("see_through") or hb.split(" #")[0] not in names:
                continue
            if _related(prompt, name, hb.split(" #")[0]):
                continue                                 # "a cat on the table", "a girl holding a sign": meant
            if _staged(s, hb, S, thing, prompt=prompt) or not confident_thing(thing, prompt, S):
                continue                                 # a chef in front of the shelves: staging, not a clash
            w = _issue("prompt_covered", "warn", f"The request names the {thing}, and {cover:.0f} % of it is covered "
                       f"by {hb}.", s["bbox"], round(cover / 100, 3), 0.1, f"move the {thing} or {hb} so both are whole")
            w["short"] = f"{thing}: {cover:.0f} % covered by {hb.split(' #')[0]} - move one so both are whole"
            w["names"] = [s["name"]]
            out.append(w)
            break
    small = [t for t, _, _, a in named if a < 3.0]
    big = [s for s in S if s["area_pct"] > 33.3 and not s.get("backdrop")
           and not (_is_band(s) and (min(s["size"]) < 0.5 or max(s["size"]) >= 0.98))   # a strip or a backdrop
           and s["kind"] in ("object", "plant", "creature", "person")]
    if big and len(small) >= 3:
        b = max(big, key=lambda s: s["area_pct"])
        nm = b["name"].split(" #")[0]
        w = _issue("prompt_crowded", "warn", f"The {nm} takes {b['area_pct']:.0f} % of the frame while "
                   f"{len(small)} requested things are under 3 %: {', '.join(small[:6])}.", b["bbox"],
                   b["area_pct"] / 100, 0.333, f"shrink the {nm} and give every requested item room")
        w["short"] = (f"the {nm} takes {b['area_pct']:.0f} % of the frame; {len(small)} requested things are under "
                      "3 % - give every item room")
        out.append(w)
    return out


_STEPS = re.compile(r"\b(?:steps?|rules?|stages?|first\b.*\bthen|label(?:led|ed)?\s+with|show\s+how|explain|"
                    r"diagram\s+of)\b|\?", re.I)


# confident things: what a missing-thing line may name (precision first)
_NOT_CONFIDENT = set("""way side moment sense kind idea scene view style mood light air time part piece lot rest image
picture photo photograph illustration drawing painting render rendering composition background foreground setting
atmosphere vibe feel look design layout format thing object item creature pet character animal stuff element beast
critter action order history story rule rules principle analogy physics warmth expression terror resemblance form
situation peculiarity complement imagination worship prayer feat focus confidence stance pose stride motif motifs
pattern patterns effect effects property properties content contents aftermath celebration investigation cleanup
service hint hints era evolution interface sentence link phrase symbol symbols function convention conventions
position arrangement component components system circuit region elevation elevations difference diameter measure
year country vehicle transportation tip tips mechanics bundle bundles curve attribute attributes detail details
texture colour color colours colors tone tones contrast emphasis balance harmony energy movement motion speed
direction distance height width size length number amount area space expanse surroundings environment nature world
life death beauty spirit memory dream emotion feeling day night morning evening afternoon season step steps process
stage phase sequence version type variety series concept theme message quality condition state states mark marks
note notes beat beats viewer focus manner spirit word words name names label labels text title caption heading
version copy example examples sample samples case cases fact facts reason result results answer question problem
solution method methods way ways use purpose goal value values level levels degree degrees unit units point points
total sum average shape shapes line lines edge edges corner corners centre center middle top bottom front back left
right inside outside surface everything something anything nothing everyone someone anyone people folk folks
workday temperature ambiance ambience setup release grip visual visuals boundary screenshot broadcast stretch
precision tableau casual longer closer taller shorter bigger smaller wider higher lower larger farther further
nearer adjacent sideways overview depiction rendition close-up closeup portrait being beings attire outfit outfits clothing clothes costume garment garments""".split())
# parts of a recorded whole: the whole answers for them
_PART_WORDS = set("""wheel wheels handle handles roof trunk branch branches leaf leaves seat seats slat slats base lid rim
spout petal petals stem stems fin fins headlight headlights thruster thrusters exhaust bumper blade blades knob hinge
strap straps sole laces button buttons pocket pockets cuff buckle chimney engine grille tire tires tyre tyres pedal
pedals spoke spokes mane horn horns antler antlers hoof hooves claw claws scale scales shell finger fingers neck chest
elbow knee knees foot feet head mouth nose teeth lips body""".split())
# a place the picture is set in: drawn as the setting, never a missing thing
_SETTING_WORDS = set("""room hall hallway kitchen bathroom bedroom office studio street alley park garden yard field
landscape city town village sky ground floor ceiling wall walls interior exterior station platform beach shore desert
jungle countryside lane corridor courtyard plaza square market shop store cafe restaurant classroom library lobby
backyard meadow valley hill hills mountains horizon""".split())
# words a parser leaves in a phrase that make it no noun phrase
_VERBISH = set("""displays display creates create evoke evokes captures capture boasts provides provide appears appear
become becomes could would should might swung used frame frames leads lead jut extends extend states lay having begin
introduce organize indicate passes looking standing follows following after before since rather within upwards
downwards ajar visible apparent radiate maintain went came forgive travels makes make shows show gives give takes take
holds hold uses features feature includes include contains contain depicts depict reads read says flew flies fell
falls adjacent sideways found containing holding showing""".split())
_DET_RE = (r"(?:a|an|the|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+|several|some|many|few|"
           r"both|each|every|another|his|her|their|its|my|our|your)")
_FOLLOW_OK = {"is", "are", "was", "were", "has", "have", "had", "that", "which", "who", "whose", "with", "and", "or",
              "but", "while", "as", "than", "so", "then", "each", "all", "both", "one"}


_WEARERS = _FIGURE_WORDS_BASE = {"woman", "man", "girl", "boy", "person", "child", "lady", "gentleman", "figure",
                                  "kid", "baby", "people", "dancer", "player", "runner", "rider", "knight", "king",
                                  "queen", "chef", "cat", "dog", "bird", "horse", "bear", "fox", "rabbit", "owl",
                                  "robot", "doll", "statue", "monkey", "lion", "tiger", "cow", "pig", "sheep", "duck"}


def _part_here(word, prompt, rows=None):
    """True when ``word`` names a part or a worn thing whose whole is also there: named in the request or recorded
    in the scene ("a cat with green eyes", a recorded "bicycle" for "wheels"). A part the request asks for on its
    own ("a black wheel", "a red hat") is the requested thing itself."""
    w = str(word or "").lower()
    one = _one(w)
    worn = bool({w, one} & (_WORN | _WORN_MORE))
    part = bool({w, one} & _PART_WORDS)
    if not (worn or part):
        return False
    skip = _PART_WORDS | _WORN | _WORN_MORE | _SETTING_WORDS
    kinds, heads = set(), set()
    for r in rows or []:
        if r.get("kind") in ("object", "plant", "creature", "person") and not r.get("backdrop"):
            h = r["name"].split(" #")[0].lower().split()
            if h and _one(h[-1]) != one and h[-1] not in skip and _one(h[-1]) not in skip:
                kinds.add(r["kind"])
                heads.add(_one(h[-1]))
    try:
        subj = {_one(x["thing"].lower().split()[-1]) for x in subjects(prompt) if x["thing"].split()}
    except Exception:
        subj = set()
    subj = {x for x in subj if x != one and x not in skip}
    if worn:
        return bool(kinds & {"person", "creature"}) or bool(subj & (_WEARERS | _FIGURE_WORDS))
    return bool(heads or subj)


def confident_thing(phrase, prompt, rows=None):
    """True when ``phrase`` (a thing COVERAGE lists) is surely a concrete thing the request asks to draw: written in
    the request after a determiner or a count, its head a noun (no function word, verb, adjective, abstract or
    framing word, body part, worn thing, part of a whole or setting), ending its noun phrase. A missing-thing line
    names only these."""
    ws = str(phrase or "").lower().split()
    if not ws or len(ws) > 3 or not prompt:
        return False
    if any(not re.fullmatch(r"[a-z][a-z'-]*", w) for w in ws):
        return False
    head = ws[-1]
    one = _one(head)
    if len(one) < 3 or {head, one} & (_NOT_CONFIDENT | _SETTING_WORDS | _PRONOUNS) or _part_here(head, prompt, rows):
        return False
    if any(w in _VERBISH or w in _PRONOUNS or w in _NOT_CONFIDENT and w != head for w in ws):
        return False
    if head.endswith("s") and (head in _SCENE_VERB_FORMS or head in _VERBS) or \
            head in {"stands", "sits", "hangs", "lies", "rests", "leans", "floats", "flies", "sleeps", "walks"}:
        return False                                     # "a heron stands": a verb, not a thing
    if head.endswith(("ly", "ous", "ful", "ive", "ish", "less", "est")) or head.endswith("ed") and len(head) > 4 or \
            head.endswith("ing") and head not in _ING_NOUNS:
        return False
    low = " ".join(re.findall(r"[a-z0-9'-]+|[^\sa-z0-9'-]", _clean_request(prompt).lower()))
    for m in re.finditer(r"\b" + _DET_RE + r"\s+(?:[a-z'-]+\s+){0,2}?" + re.escape(" ".join(ws)) + r"\b", low):
        nxt = low[m.end():].split()[:1]
        if not nxt or not re.match(r"[a-z]", nxt[0]) or nxt[0] in _NP_END or nxt[0] in _FOLLOW_OK or \
                nxt[0] in _SCENE_VERB_FORMS or nxt[0] in _VERBS or nxt[0] in _VERBISH or \
                        re.fullmatch(r"[a-z]{2,}(?:ing|ed)", nxt[0]) and nxt[0] not in _ING_NOUNS:
            return True
    return False


def scene_words(cv=None, rows=None, source=None):
    """Every word the scene uses for what it drew, singular: recorded names and aliases, label and text strings,
    and the scene's own source (identifiers, strings and comments, without the PROMPT statement)."""
    words = set()

    def add(text):
        for t in re.findall(r"[A-Za-z]+", str(text or "").replace("_", " ")):
            t = t.lower()
            words.add(t)
            words.add(_one(t))
    for r in rows or []:
        add(r.get("name", ""))
        for a in r.get("aliases", []) or []:
            add(a)
    if cv is not None:
        try:
            for t in _label_texts(cv):
                add(t)
        except Exception:
            pass
        for t in getattr(cv, "text_log", []) or []:
            add(t.get("text", "") if isinstance(t, dict) else t)
    if source:
        lines = source.splitlines()
        try:
            import ast
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PROMPT" for t in node.targets):
                    for k in range(node.lineno - 1, node.end_lineno):
                        lines[k] = ""
        except (SyntaxError, ValueError):
            pass
        add(" ".join(lines))
    return words


def answered_by_words(phrase, words):
    """True when the scene's words answer ``phrase``: its head (singular, plural stem or an alias of it), or the
    whole phrase written as one word ("tool box" -> "toolbox")."""
    ws = str(phrase or "").lower().split()
    if not ws:
        return False
    head = ws[-1]
    cand = {head, _one(head), "".join(ws), "".join(_one(w) for w in ws)} | set(_ANSWERED_BY.get(_one(head), ()))
    if cand & words:
        return True
    stem = _one(head)
    if any(len(w) >= 4 and stem.endswith(w) for w in words):
        return True                                      # "workbench" answered by "bench"
    return len(stem) >= 6 and any(w.startswith(stem[:5]) and abs(len(w) - len(stem)) <= 3 for w in words)


def _inside(a, b, frac=0.8):
    """True when at least ``frac`` of box ``a`` lies inside box ``b``."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-9, (a[2] - a[0]) * (a[3] - a[1])) >= frac


def _staged(s, hb, S, thing, narrow=False, prompt=None):
    """True when ``hb`` in front of ``s`` is staging, not a clash: worn things and parts, a figure in front of
    things, a thing in front of a wider one, one inside the other, or two entries of one thing."""
    fn = hb.split(" #")[0]
    head = _one(thing.lower().split()[-1])
    if head in _SETTING_WORDS or _part_here(head, prompt, S) or prompt is None and head in _PART_WORDS | _WORN:
        return True
    a = {_one(w) for w in s["name"].split(" #")[0].lower().split()}
    if a & {_one(w) for w in fn.lower().split()}:
        return True
    front = next((r for r in S if r["name"] == hb), None) or \
        next((r for r in S if r["name"].split(" #")[0] == fn), None)
    if front is None:
        return False
    if front["kind"] in ("person", "creature") or _inside(front["bbox"], s["bbox"]):
        return True                                      # a figure in front; a part or a thing set on it
    return not narrow and (s["size"][0] > front["size"][0] or _inside(s["bbox"], front["bbox"]))


_FIGURE_WORDS = {"woman", "man", "girl", "boy", "person", "child", "lady", "gentleman", "figure", "kid", "baby",
                 "teenager", "student", "worker", "chef", "dancer", "player", "musician", "farmer", "doctor"}
_CUT_OK = set("""steam smoke mist fog haze light glow spray splash cloud clouds rain snow water sky ground floor wall
road river sea ocean field grass forest beam beams ray rays shadow wave waves trail fire flame flames""".split())


def checked_request(prompt):
    """True for a request a checker can hold the picture to: three or more things, a count, a colour on a thing, a
    place, quoted text, or steps / rules / a question. One subject, a mood or a style is not checked."""
    if not prompt:
        return False
    it = request_items(prompt)
    return len(it["things"]) >= 3 or bool(it["counts"] or it["colours"] or it["places"] or it["text"]) or \
            bool(_STEPS.search(prompt))


_RELATING = re.compile(r"\b(?:on|onto|in|inside|into|holding|holds|hold|wearing|wears|carrying|carries|riding|rides|"
                       r"behind|in front of|under|beneath|over|covering|covers|hugging|atop|on top of|against|"
                       r"leaning|perched|sitting|standing|lying)\b", re.I)


def _related(prompt, a, b):
    """True when the request puts ``a`` and ``b`` together in one clause with a relation that lets one cover the
    other ("a cat on the table", "a boy holding a kite", "a hat behind the vase")."""
    ka, kb = _one(a.split()[-1]), _one(b.split()[-1])
    for clause in re.split(r"[.,;:!?]|\s(?:and|while|but)\s", _clean_request(prompt or "")):
        words = {_one(w) for w in _tokens(clause, split=True)}
        if ka in words and kb in words and _RELATING.search(clause):
            return True
    return False


def place_checks(cv, prompt, rows=None):
    """Compare each place the request states with where the recorded objects are: ``(results, issues)``; a
    result is ``{"text", "status" ("holds" | "fails" | "not checked"), "detail", "line"}``."""
    reqs = place_requests(prompt) if prompt else []
    if not reqs:
        return [], []
    S = summary(cv) if rows is None else rows
    W, H = cv.w, cv.h
    results, issues = [], []

    def find(np_):
        idx = [i for i in _rows_for(S, np_["run"])
               if not S[i].get("backdrop") and S[i]["area_pct"] < 50 and S[i]["kind"] not in _CONTAINER]
        return idx
    for r in reqs:
        a = r["a"]
        ia = find(a)
        an = _one(a["head"])
        if r["kind"] == "rel":
            b = r["b"]
            bn = _one(b["head"])
            ib = find(b)
            text = f"{an} {_REL_WORD[r['rel']]} {bn}"
            if an == bn:
                continue
            if r["rel"] in _CONTACT_RELS:
                if not ia or not ib:
                    continue                             # "on" / "in" / "next to" of unrecorded things: no line
                res, iss = _contact_check(S, ia, ib, an, bn, r["rel"], W, H)
                results.append(res)
                issues += iss
                continue
            if not ia or not ib:
                miss = an if not ia else bn
                results.append({"text": text, "status": "not checked",
                                "detail": f"{miss} is not recorded", "line": f"{text}: not checked ({miss} not recorded)"})
                continue
            ub = _union([_box_px(S[i], W, H) for i in ib])
            wrong = []
            for i in ia:
                d = _direction(_box_px(S[i], W, H), ub)
                if d != r["rel"]:
                    wrong.append((S[i]["name"], d))
            if not wrong:
                results.append({"text": text, "status": "holds", "detail": "", "line": f"{text}: holds ✓"})
                continue
            nm, d = wrong[0]
            why = (f"the {nm} is {_REL_WORD.get(d, d) if d else 'overlapping or too close to'} the {bn}" if d else
                   f"the {nm} overlaps the {bn} or sits too close to tell")
            if d in ("left", "right"):
                why = f"the {nm} is {d} of the {bn}"
            elif d:
                why = f"the {nm} is {d} the {bn}"
            more = f" (+{len(wrong) - 1} more)" if len(wrong) > 1 else ""
            results.append({"text": text, "status": "fails", "detail": why + more,
                            "line": f"{text}: fails ✗ ({why}{more})"})
            word = _REL_WORD[r["rel"]]
            axis = "x" if r["rel"] in ("left", "right") else "y"
            w_ = _issue(
                "prompt_relation", "warn", f"The request puts the {an} {word} the {bn}, but {why}{more}.", None,
                d, r["rel"],
                f"move the {an} so its centre lies clearly {word} the {bn} (a different {axis} position by more "
                f"than half the two sizes), or swap the two positions")
            said = r["rel"] if r["rel"] in ("left", "right") else word
            w_["short"] = (f"{why[4:]}{more}, the request says {said} - move the {an} or swap the two"
                           if why.startswith("the ") else f"{an}: {why}, the request says {said}")
            issues.append(w_)
        else:
            v, hz = r["where"]
            label = " ".join(x for x in (v, hz) if x) or "centre"
            text = f"{an} at the {label}"
            if not ia:
                results.append({"text": text, "status": "not checked", "detail": f"{an} is not recorded",
                                "line": f"{text}: not checked ({an} not recorded)"})
                continue
            bad = []
            for i in ia:
                u, v_ = S[i]["centre"]
                ok = True
                if hz == "left":
                    ok &= u < 0.5
                elif hz == "right":
                    ok &= u > 0.5
                if v == "top":
                    ok &= v_ < 0.5
                elif v == "bottom":
                    ok &= v_ > 0.5
                if hz == "centre" or (not v and not hz):
                    ok &= abs(u - 0.5) < 0.2 and abs(v_ - 0.5) < 0.2
                if not ok:
                    bad.append((S[i]["name"], u, v_))
            if not bad:
                results.append({"text": text, "status": "holds", "detail": "", "line": f"{text}: holds ✓"})
                continue
            nm, u, v_ = bad[0]
            why = f"the {nm} is centred at ({u:.2f}, {v_:.2f}) of the canvas"
            results.append({"text": text, "status": "fails", "detail": why, "line": f"{text}: fails ✗ ({why})"})
            issues.append(_short(f"{nm} is centred at ({u:.2f}, {v_:.2f}), the request says {label} - move it there",
                          _issue("prompt_relation", "warn", f"The request puts the {an} at the {label}, but {why}.",
                                 None, [round(u, 2), round(v_, 2)], label,
                                 f"move the {an} into the {label} of the canvas (x < 0.5 for left, > 0.5 for right, "
                                 "y < 0.5 for top, > 0.5 for bottom)")))
    return results, issues


def single_count_issues(cv, prompt, rows=None):
    """"a green bench": exactly one is drawn. A thing the request gives a colour or a place and names once (an
    article, no number) with several recorded objects answering to it."""
    if not prompt:
        return []
    S = summary(cv) if rows is None else rows
    seen, out = set(), []
    nums = {_one(w) for _, _, ws, _ in count_phrases(prompt) for w in ws[:3]}
    cands = [(r["head"], r["run"], r["n"]) for r in colour_requests(prompt)]
    for p in [p for p in place_requests(prompt) if p.get("rel") not in _CONTACT_RELS]:
        cands.append((p["a"]["head"], p["a"]["run"], p["a"]["n"]))
        if p["kind"] == "rel":
            cands.append((p["b"]["head"], p["b"]["run"], p["b"]["n"]))
    for head, run, n in cands:
        key = _one(head)
        if n != 1 or key in seen or key in nums or head.endswith("s") and not head.endswith("ss"):
            continue
        seen.add(key)
        if key in _SCATTER or key in _DETAIL:
            continue
        idx = [i for i in _rows_for(S, run) if S[i]["kind"] not in _CONTAINER and not S[i].get("backdrop")]
        have = sum(int((S[i].get("meta") or {}).get("count", 1) or 1) for i in idx)
        if have > 1:
            out.append(_short(f"{key}: {have} drawn, the request says one - remove the extra ones",
                              _issue("prompt_count", "warn",
                              f"The request says {_a(key)} but {have} recorded entries answer to it "
                              f"({', '.join(S[i]['name'] for i in idx[:6])}).", None, have, 1,
                              f"draw exactly one {key} (remove the extra ones), or, when they are parts of one "
                              f'thing, record only the whole: manifest.note(cv, "object", "{key}", mask=...)')))
    return out


def request_summary(cv, prompt, rows=None):
    """The literal parts of a request against the picture, for an agent to read at a glance: ``{"things": [...],
    "places": [...], "text": [...], "lines": [...], "colours": [...]}`` or None when the request has no count,
    colour or place. ``lines`` are printable; a ✗ names what to change."""
    if not prompt:
        return None
    S = summary(cv) if rows is None else rows
    cres, _ = colour_checks(cv, prompt, S)
    pres, _ = place_checks(cv, prompt, S)
    heads = {}                                           # singular head -> {"run", "n"}

    def add(head, run, n):
        if not _is_thing(head):
            return
        k = _one(head)
        d = heads.setdefault(k, {"run": run, "n": None, "plural": False})
        if n is not None and d["n"] is None:
            d["n"] = n
        if head != k and head.endswith("s"):
            d["plural"] = True
    for r in colour_requests(prompt):
        add(r["head"], r["run"], r["n"])
    for num, n, words, _ in count_phrases(prompt):
        ws = [w.lower() for w in words[:3]]
        cut = []
        for w in ws:
            if w in _STOP or w in _DETS:
                break
            cut.append(w)
        w0 = words[0].lower() if words else ""
        if w0 in _VERBS or (w0.endswith("ing") and w0 not in _ING_NOUNS):
            continue                                     # "one holding an umbrella": a person, not a count
        if ws and ws[0] == "of":
            continue                                     # "one of the cups"
        np_ = _np_forward([w.lower() for w in words], 0)   # "one cup broken into pieces": the cup, not "broken"
        if np_["head"] and _is_thing(np_["head"]) and _modifier(np_["head"]) is None:
            if n == 1 and _one(np_["head"]) in heads and heads[_one(np_["head"])]["n"]:
                continue                                 # "three cups ... one cup broken": one of them
            add(np_["head"], np_["run"], n)
    for p in [p for p in place_requests(prompt) if p.get("rel") not in _CONTACT_RELS]:
        add(p["a"]["head"], p["a"]["run"], p["a"]["n"])
        if p["kind"] == "rel":
            add(p["b"]["head"], p["b"]["run"], p["b"]["n"])
    if not heads and not pres and not text_place_requests(prompt):
        return None
    things, lines = [], []
    for k, d in heads.items():
        idx = [i for i in _rows_for(S, d["run"]) if S[i]["kind"] not in _CONTAINER or S[i]["kind"] == "background"]
        drawn = sum(int((S[i].get("meta") or {}).get("count", 1) or 1) for i in idx)
        row = {"thing": k, "requested": d["n"], "drawn": drawn, "colour_requested": [], "colour_painted": [],
               "ok": True}
        parts = []
        if not idx:
            row["ok"] = None
            parts.append("not recorded" + (f" ({d['n']} requested)" if d["n"] else ""))
        elif d["n"] is not None:
            good = drawn == d["n"]
            row["ok"] = row["ok"] and good
            parts.append(f"{drawn} drawn, {d['n']} requested " + ("✓" if good else "✗"))
        else:
            parts.append(f"{drawn} drawn")
        for c in [c for c in cres if c["thing"] == k]:
            cw = " and ".join(c["colours"])
            row["colour_requested"].append(cw)
            if c["status"] == "not recorded":
                parts.append(f"{cw} requested, colour not checked")
                continue
            seen = [x["desc"] for x in c["instances"] if x.get("desc")]
            if c["status"] == "wrong":
                bad = [x for x in c["instances"] if x.get("ok") is False]
                painted = bad[0]["top"] if bad else (seen[0] if seen else "?")
                row["ok"] = False
                row["colour_painted"].append(painted)
                parts.append(f"{cw} requested, {painted} painted ✗")
            else:
                painted = ", ".join(f"{s_} x{seen.count(s_)}" if len(set(seen)) > 1 else s_
                                    for s_ in dict.fromkeys(seen)) or cw
                row["colour_painted"].append(painted)
                parts.append(f"{cw} requested, {painted} painted ✓")
        things.append(row)
        lines.append(f"{k}: " + "; ".join(parts))
    for p in pres:
        lines.append(p["line"])
    texts = []
    for q, ok, near in quoted_status(cv, prompt):
        texts.append({"text": q, "drawn": ok, "closest": near})
        lines.append(f"{q!r}: " + ("drawn ✓" if ok else "missing ✗" + (f" (closest drawn: {near[:40]!r})" if near else "")))
    tres, tiss = text_checks(cv, prompt)
    for w in tiss:
        if w["rule"] == "prompt_text_small":
            lines.append(w["short"].replace(" - make", " ✗ - make"))
    lines += [r["line"] for r in tres]
    return {"things": things, "places": pres, "text": texts, "colours": cres, "lines": lines}


# --------------------------------------------------------------------------------------------- picture kind

PAINTERLY_RULES = ("blown_highlights", "narrow_tonal_range", "sparse_composition", "low_focal_contrast", "banding",
                   "flat_cells", "monochrome", "scattered_highlights", "focal_centered", "focal_at_edge",
                   "horizon_centered", "subjects_in_a_row", "crushed_blacks", "unbalanced_lr", "high_contrast")
_CORE_PAINTERLY = PAINTERLY_RULES[:5]
KINDS = ("painting", "design", "diagram", "drawing")


def infer_kind(cv):
    """'painting' | 'design' | 'diagram' | 'drawing' from what was drawn and how the canvas looks."""
    kinds = [e["kind"] for e in (getattr(cv, "manifest", []) or [])]
    if "view" in kinds:
        return "drawing"
    if any(k in ("node", "group", "diagram") for k in kinds):
        return "diagram"
    if "chart" in kinds:
        return "design"
    H, W = cv.h, cv.w
    f = max(1, max(H, W) // 256)
    im = np.clip(np.asarray(cv.buf)[::f, ::f], 0, 1)
    lum = im @ np.array([0.299, 0.587, 0.114], F32)
    mx, mn = im.max(-1), im.min(-1)
    sat = float(((mx - mn) / np.maximum(mx, 1e-6)).mean())
    g = np.abs(np.diff(lum, axis=0))[:, :-1] + np.abs(np.diff(lum, axis=1))[:-1]
    flat = float((g < 0.004).mean())
    tl = getattr(cv, "text_log", None) or []
    has_text = bool(tl)
    vec = bool(cv.__dict__.get("_drawings"))
    if len(tl) >= 3:
        ink = sum(max(0, t["bbox"][2] - t["bbox"][0]) * max(0, t["bbox"][3] - t["bbox"][1]) for t in tl) / (H * W)
        if ink > 0.06:
            return "design"                               # a poster / card / slide: type carries the picture
    if float(np.median(lum)) > 0.85 and sat < 0.12 and flat > 0.6:
        return "diagram" if vec else "drawing"
    if vec and flat > 0.5:
        return "diagram"
    if has_text and flat > 0.45:
        return "design"
    return "painting"


# --------------------------------------------------------------------------------------------- styled pictures

_PAPER = ("blown_highlights", "scattered_highlights", "low_focal_contrast", "flat_cells", "sparse_composition")
_FLAT = ("flat_cells", "sparse_composition", "blends_into_background", "banding")
_POSTER = ("banding", "blends_into_background", "high_contrast")
_GRAIN = ("jpeg_noise",)
_MONO = ("monochrome",)
_DARK = ("crushed_blacks", "narrow_tonal_range", "low_focal_contrast")
STYLE_EXPECTS = {
    # style: critique rules its own look triggers on purpose (reported as info, "expected for <style>")
    "oil": ("jpeg_noise",),
    "watercolor": _PAPER + ("narrow_tonal_range", "jpeg_noise"),
    "pencil": _PAPER + _MONO + ("narrow_tonal_range", "jpeg_noise"),
    "colored_pencil": _PAPER + ("jpeg_noise", "narrow_tonal_range"),
    "charcoal": _MONO + _GRAIN + ("crushed_blacks", "high_contrast", "narrow_tonal_range"),
    "ink_wash": _PAPER + _MONO + ("narrow_tonal_range", "banding"),
    "pen_ink": _PAPER + _MONO + _GRAIN + ("high_contrast",),
    "engraving": _PAPER + _MONO + _GRAIN + ("high_contrast", "crushed_blacks"),
    "comic": _POSTER + _GRAIN + ("oversaturated", "flat_cells", "blown_highlights", "crushed_blacks",
                                 "scattered_highlights"),
    "manga": _PAPER + _MONO + _POSTER + _GRAIN + ("crushed_blacks",),
    "cel": _FLAT + _POSTER + ("oversaturated",),
    "gouache": _FLAT + ("blown_highlights", "scattered_highlights"),
    "pastel": _GRAIN + ("narrow_tonal_range", "low_focal_contrast"),
    "crayon": _PAPER + _GRAIN,
    "impressionist": _GRAIN,
    "pointillism": _PAPER + _GRAIN,
    "ukiyoe": _FLAT + _POSTER + ("scattered_highlights",),
    "risograph": _POSTER + _GRAIN + ("scattered_highlights", "flat_cells", "oversaturated", "narrow_tonal_range", "monochrome"),
    "linocut": _POSTER + _MONO + _GRAIN + ("scattered_highlights", "crushed_blacks", "flat_cells", "sparse_composition"),
    "stained_glass": _POSTER + ("crushed_blacks", "oversaturated", "jpeg_noise", "scattered_highlights"),
    "mosaic": _GRAIN + ("high_contrast", "scattered_highlights"),
    "low_poly": _FLAT + _POSTER + ("scattered_highlights",),
    "pixel_art": _FLAT + _POSTER + _GRAIN + ("oversaturated",),
    "paper_cut": _FLAT + _POSTER + ("scattered_highlights",),
    "collage": _GRAIN + ("blown_highlights", "scattered_highlights", "high_contrast"),
    "cross_stitch": _GRAIN + ("scattered_highlights", "high_contrast"),
    "blueprint": _MONO + _DARK + _FLAT + _GRAIN + ("scattered_highlights", "high_contrast"),
    "cyanotype": _MONO + ("narrow_tonal_range", "scattered_highlights", "jpeg_noise"),
    "chalkboard": _DARK + _MONO + _GRAIN + _FLAT + ("scattered_highlights",),
    "synthwave": _DARK + ("blown_highlights", "scattered_highlights", "oversaturated", "banding", "high_contrast"),
    "vaporwave": _GRAIN + ("oversaturated", "banding", "narrow_tonal_range"),
    "duotone": _POSTER + ("monochrome", "narrow_tonal_range", "flat_cells"),
    "vintage": _GRAIN + ("narrow_tonal_range", "low_focal_contrast", "blown_highlights"),
    "sepia": _GRAIN + _MONO + ("narrow_tonal_range", "low_focal_contrast"),
    "noir": _GRAIN + _MONO + ("blown_highlights", "crushed_blacks", "high_contrast", "narrow_tonal_range"),
    "tilt_shift": ("oversaturated", "banding"),
    "photo_finish": _GRAIN,
    "glitch": _GRAIN + ("banding", "oversaturated", "subject_cropped"),
    "thermal": _POSTER + ("oversaturated", "crushed_blacks", "monochrome"),
    "infrared": ("oversaturated", "blown_highlights"),
    "sticker": ("blown_highlights", "scattered_highlights", "sparse_composition", "flat_cells"),
    "storybook": _GRAIN + ("narrow_tonal_range", "low_focal_contrast", "flat_cells"),
    "anime": _FLAT + ("oversaturated", "scattered_highlights"),
    "clay": _FLAT + ("jpeg_noise",),
}
# a style does not move things: these are judged on the picture before the style
COMPOSITION_RULES = ("subject_cropped", "subjects_in_a_row", "horizon_centered", "focal_centered", "focal_at_edge",
                     "unbalanced_lr")
# measured inside the picture area when a style frames the picture with paper / a mat
_AREA_RULES = PAINTERLY_RULES + ("oversaturated", "blends_into_background")


def style_expected(styles):
    """{rule: style} for the critique rules the applied ``styles`` trigger on purpose."""
    out = {}
    for s in styles or []:
        for r in STYLE_EXPECTS.get(str(s), ()):
            out.setdefault(r, str(s))
    return out


def style_relax(warnings, styles):
    """Downgrade the rules a style triggers by being itself to info with "expected for <style>"."""
    exp = style_expected(styles)
    if not exp:
        return list(warnings)
    out = []
    for w in warnings:
        s = exp.get(w.get("rule"))
        if s and w.get("severity") != "info":
            w = dict(w, severity="info", message=f"{w['message']} (expected for {s})")
        elif s:
            w = dict(w, message=f"{w['message']} (expected for {s})")
        out.append(w)
    order = {"fail": 0, "warn": 1, "info": 2}
    out.sort(key=lambda w: order.get(w["severity"], 3))
    return out


def picture_area(img, tol=0.085):
    """Box (x0, y0, x1, y1) in px of the picture inside a uniform paper border / mat (a frame of near-constant
    colour on at least three sides), or None when there is none."""
    a = np.clip(np.asarray(img, F32)[..., :3], 0, 1)
    H, W = a.shape[:2]
    if min(H, W) < 48:
        return None
    ring = np.concatenate([a[:2].reshape(-1, 3), a[-2:].reshape(-1, 3), a[:, :2].reshape(-1, 3),
                           a[:, -2:].reshape(-1, 3)])
    paper = np.median(ring, axis=0)
    if float((np.abs(ring - paper).max(-1) <= tol).mean()) < 0.85:
        return None
    near = np.abs(a - paper).max(-1) <= tol                   # (H, W) paper-coloured pixels
    short = min(H, W)
    need, extra = max(2, int(round(0.006 * short))), int(round(0.02 * short))

    def depth(rows):                                          # rows: fraction of paper per line, outside in
        n = len(rows)
        d = 0
        while d < n and rows[d] >= 0.5:
            d += 1
        if d < need:
            return 0
        e = d
        while e < min(n, d + extra) and rows[e] >= 0.08:      # past the ragged (deckled) edge
            e += 1
        return e
    lim_y, lim_x = int(H * 0.3), int(W * 0.3)
    top = depth(near[:lim_y].mean(1))
    bottom = depth(near[::-1][:lim_y].mean(1))
    left = depth(near[:, :lim_x].mean(0))
    right = depth(near[:, ::-1][:, :lim_x].mean(0))
    fr = [top / H, bottom / H, left / W, right / W]
    if sum(1 for d in fr if d) < 3 or max(fr) > 0.15 or max(fr) > 2.2 * min(d for d in fr if d):
        return None                                           # a paper border is narrow and even all round
    x0, y0, x1, y1 = left, top, W - right, H - bottom
    if (x1 - x0) * (y1 - y0) < 0.4 * W * H:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def mostly_blank(img, style_log):
    """A ``mostly_blank`` fail when a style left a picture that is almost one colour although the picture
    before the style was not; else None."""
    log = [r for r in (style_log or []) if "uniform_before" in r]
    if not log:
        return None
    before = float(log[0]["uniform_before"])
    after = dominant_share(img)
    if after <= 0.85 or after - before < 0.15:
        return None
    names = ", ".join(r["style"] for r in log)
    opts = "; ".join(f"{r['style']}: " + ", ".join(f"{k}={v!r}" for k, v in r["params"].items())
                     for r in log if r.get("params"))
    return _issue("mostly_blank", "fail",
                  f"{after * 100:.0f}% of the picture is one colour after {names} (it was {before * 100:.0f}% "
                  "before the style): the content is lost." + (f" Options given: {opts}." if opts else ""),
                  None, after, 0.85,
                  "go back to the style's defaults; stylize.options(NAME) lists each option's useful range")


# photo-style rules: about light, lenses and film, never about icon sheets, UI screens, charts or flat posters
PHOTO_RULES = ("scattered_highlights", "jpeg_noise", "narrow_tonal_range", "blown_highlights", "crushed_blacks",
               "focal_centered", "focal_at_edge", "horizon_centered", "subjects_in_a_row", "unbalanced_lr",
               "monochrome", "high_contrast", "low_focal_contrast", "flat_cells")


_PLAIN_BG = re.compile(r"\b(seamless|plain|solid|empty|minimal|minimalist|clean|blank|flat|studio|white|"
                       r"single[- ]colou?r)\b[^.;:]{0,30}?\b(backdrop|background|backgrounds|sweep)\b|"
                       r"\bnegative space\b|\bproduct shot\b", re.I)


_GROUND_WORDS = re.compile(r"\b(floors?|tiles?|tiled|walls?|table ?tops?|tabletop|desk|counter(?:top)?|"
                           r"parquet|carpet|rug|board)\b", re.I)


def horizon_relax(warnings, img, kind=None, prompt=None):
    """Keep ``horizon_centered`` only for a real horizon: one dominant, full-width edge that separates two large
    regions of different tone. A tiled floor, a grid or any repeated line pattern is no horizon; neither is the
    centre line of a design, diagram or drawing, or of a request whose ground is a floor, wall, tiles or table."""
    if not any(w.get("rule") == "horizon_centered" for w in warnings):
        return list(warnings)
    drop = kind in ("design", "diagram", "drawing") or bool(prompt and _GROUND_WORDS.search(prompt))
    if not drop:
        a = np.clip(np.asarray(img, np.float32), 0, 1)
        if a.ndim == 3:
            a = a[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        h, w = a.shape
        rows = np.convolve(a.mean(1), np.ones(5) / 5, mode="same")
        dif = np.abs(np.diff(rows[3:-3]))
        if not dif.size:
            return list(warnings)
        k = int(np.argmax(dif)) + 3
        peak = float(dif.max())
        rest = dif.copy()
        lo, hi = max(0, k - 3 - int(0.04 * h)), min(dif.size, k - 3 + int(0.04 * h) + 1)
        rest[lo:hi] = 0
        dominant = peak >= 2.0 * float(rest.max() if rest.size else 0)
        band = max(2, int(0.03 * h))
        above, below = a[max(0, k - band):k], a[k + 1:k + 1 + band]
        if above.size and below.size:
            step = below.mean(0) - above.mean(0)
            sign = np.sign(step.mean()) or 1.0
            wide = float(np.mean(step * sign > 0.03)) >= 0.8
        else:
            wide = False
        regions = abs(float(a[:k].mean()) - float(a[k + 1:].mean())) >= 0.08 if 0 < k < h - 1 else False
        drop = not (dominant and wide and regions)
    if not drop:
        return list(warnings)
    return [w for w in warnings if w.get("rule") != "horizon_centered"]


def prompt_relax(warnings, prompt):
    """A request that asks for a plain backdrop (a product shot on a seamless backdrop, negative space) gets
    ``sparse_composition`` as info: the empty area is what was asked for."""
    if not prompt or not _PLAIN_BG.search(prompt):
        return list(warnings)
    out = []
    for w in warnings:
        if w.get("rule") == "sparse_composition" and w.get("severity") != "info":
            w = dict(w, severity="info", message=w["message"] + " (the request asks for a plain backdrop)")
        out.append(w)
    order = {"fail": 0, "warn": 1, "info": 2}
    out.sort(key=lambda w: order.get(w["severity"], 3))
    return out


def relax(warnings, kind):
    """For design work (posters, UI, icon sheets, charts) drop the photo-style rules and soften the rest of the
    painterly rules to info; for diagrams and drawings drop every painterly rule."""
    if kind in (None, "painting"):
        return list(warnings)
    out = []
    for w in warnings:
        if kind in ("diagram", "drawing") and w.get("rule") in PAINTERLY_RULES + PHOTO_RULES:
            continue
        if kind == "design" and w.get("rule") in PHOTO_RULES:
            continue
        soften = _CORE_PAINTERLY + ("blends_into_background", "oversaturated") if kind == "design" else \
            ("blends_into_background",)
        if w.get("rule") in soften and w.get("severity") != "info":
            w = dict(w, severity="info")               # tinted cards and hairlines are deliberate here
        out.append(w)
    order = {"fail": 0, "warn": 1, "info": 2}
    out.sort(key=lambda w: order.get(w["severity"], 3))
    return out
