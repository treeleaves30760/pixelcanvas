# -*- coding: utf-8 -*-
"""studio compositor: a picture dict + cell sources -> the composite, with cells from the cache, ``box()`` /
``anchor()`` between layers, passes and the scene finish; ``annotate`` records one manifest entry per layer (and
moves the cells' text records with their layer) so the render.py report can check the picture."""
from __future__ import annotations

import hashlib
import os
import re
import time

import numpy as np

from ..canvas import Canvas
from .. import manifest as _manifest
from .engine import (F32, Cache, CellError, run_cell, run_pass, place, effects, body, over, paint_piece,
                     cache_key, render_k, _alpha_box, _clip, providers, binder_after)

LAYER_DEFAULTS = {"visible": True, "opacity": 1.0, "dx": 0.0, "dy": 0.0, "scale": 1.0, "rotate": 0.0, "flip": False,
                  "outline": None, "outline_width": 0.004, "shadow": None, "shadow_strength": 1.0,
                  "shadow_size": 1.0, "glow": None, "glow_size": 0.03, "glow_strength": 1.0, "tint": None,
                  "tint_amount": 1.0, "blur": 0.0, "role": None, "label": None, "alias": None}
SCENE_DEFAULTS = {"light": [-0.6, -0.8], "medium": None, "medium_strength": None, "medium_mark": None,
                  "medium_palette": None, "medium_paper": None, "grade": None, "grade_amount": 0.7, "paper": False,
                  "background": "#000000", "shadow_color": "#1d1830", "reviews": 3, "cell_timeout": 60}

_PASS_CALLS = (r"^\s*(?:cv\s*=\s*)?(?:grade|paper|limit_palette)\s*\(", r"\bcv\.(?:vignette|bloom|grain|soft_clip|"
               r"blur|contrast|saturation|split_tone|levels|filmic|apply|displace)\s*\(", r"stylize\.apply\s*\(\s*cv\b")
_DRAW_CALLS = (r"\bsk\.|\bcv\.(?:paint|text|draw|cairo|paste|gradient|glow|radial|linear|fill)\s*\(|\bcreature\s*\(|"
               r"\b(?:ball|cylinder|cuboid|lathe|tabletop|Ground)\s*\(|\b(?:flora|elements|solid|form|figure|clay|"
               r"pictograms|backdrop|vector|diagram|chart|drafting|ui|scenery)\.")


def guess_role(src):
    """'pass' for a cell that only changes the whole picture (grade, paper, vignette ...); a ``# role: X`` line
    names it; else None (an object, or a backdrop when it fills the frame)."""
    m = re.search(r"^\s*#\s*role\s*[:=]\s*(\w+)", src or "", re.M)
    if m and m.group(1) in ("pass", "backdrop", "object"):
        return m.group(1)
    code = "\n".join(ln for ln in (src or "").splitlines() if not ln.strip().startswith("#"))
    if any(re.search(p, code, re.M) for p in _PASS_CALLS) and not re.search(_DRAW_CALLS, code):
        return "pass"
    return None


def layer_params(L):
    p = dict(LAYER_DEFAULTS)
    p.update({k: v for k, v in L.items() if k in LAYER_DEFAULTS})
    return p


_TEXT_NAMES = re.compile(r"^(title|caption|text|heading|headline|label|subtitle)\d*$")


def text_like(L, src):
    """A title / caption layer: named so, or a cell that only writes text."""
    if _TEXT_NAMES.match(L["name"]) or L.get("role") == "text":
        return True
    code = [ln.strip() for ln in (src or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return bool(code) and all(ln.startswith("cv.text(") or ln.startswith(")") for ln in code)


def crop_of(pic, L, src):
    """(x0, y0, f) when the picture is cropped (``crop``) and this layer is in the crop, else None."""
    c = (pic.get("scene") or {}).get("crop")
    if not c or not c.get("box"):
        return None
    if text_like(L, src) and L["name"] not in (c.get("with") or []):
        return None
    x0, y0, x1, y1 = (float(v) for v in c["box"])
    return (x0, y0, 1.0 / max(1e-3, x1 - x0))


def base_name(name):
    """'fox2' / 'tree_3' -> 'fox' / 'tree': the noun the report checks a layer by."""
    b = re.sub(r"[\s_\-]*\d+$", "", str(name)).replace("_", " ").strip()
    return b or str(name)


def _err(short, kind="Error"):
    return {"short": short, "type": kind, "line": None, "source": "", "causes": []}


class Compositor:
    """Render a picture dict: cells from the cache or run, placed, with effects, passes and the scene finish."""

    def __init__(self, pic, cells, size=None, cache=None, paths=None, log=None, folder=None):
        self.pic = pic
        self.cells = cells
        self.W, self.H = (int(v) for v in (size or pic["size"]))
        self.base = tuple(int(v) for v in pic["size"])
        self.cache = cache or Cache()
        self.paths = paths or {}
        self.log = log
        sc = dict(SCENE_DEFAULTS)
        sc.update(pic.get("scene") or {})
        self.scene = sc
        self.look = {"light": [float(v) for v in sc["light"]]}   # the medium is applied after the layers: no section re-runs
        self.seed = int(pic.get("seed", 1))
        self.by_name = {L["name"]: L for L in pic["layers"]}
        self.results, self.placed, self.info = {}, {}, {}
        self.deps_stack, self._busy = [], []
        self.canvas = None
        self.folder = folder
        self.limit = float(sc.get("cell_timeout") or 0)
        order = [c for c in (pic.get("program") or []) if c in cells]       # program order: the draft's sections
        for L in pic["layers"]:                                              # (z-order is the layer stack)
            c = L.get("cell") or L["name"]
            if c in cells and c not in order:
                order.append(c)
        self.order = order
        self._pre = {}

    # ---- names shared between sections
    def pre_of(self, cell):
        """[(provider cell, names, source, path)]: the earlier sections this cell takes names from."""
        if cell not in self._pre:
            self._pre[cell] = [(p, names, self.cells[p], self.paths.get(p) or f"layers/{p}.py")
                               for p, names in providers(cell, self.order, self.cells)]
        return self._pre[cell]

    def defined_after(self, layer, name):
        """The section after this layer's cell that binds ``name`` (a NameError explained), else None."""
        L = self.by_name.get(layer) or {}
        return binder_after(L.get("cell") or layer, self.order, self.cells, name)

    # ---- roles
    def role(self, L):
        r = L.get("role")
        if r:
            return r
        g = guess_role(self.cells.get(L.get("cell") or L["name"]))
        if g:
            return g
        cr = self.results.get(L["name"])
        if cr is not None and self._fills_frame(cr):
            return "backdrop"
        return "object"

    @staticmethod
    def _fills_frame(cr):
        if cr.rgba.size == 0:
            return False
        h, w = cr.rgba.shape[:2]
        if w < 0.97 * cr.size[0] or h < 0.97 * cr.size[1]:
            return False
        return float((cr.rgba[..., 3] > 240).mean()) > 0.9

    def info_of(self, name):
        return self.info.setdefault(name, {"ran": False, "error": None, "stale": False, "seconds": 0.0,
                                           "cached": False})

    # ---- one layer's cell result
    def result(self, name):
        if name in self.results:
            return self.results[name]
        L = self.by_name[name]
        if name in self._busy:
            loop = " -> ".join(self._busy[self._busy.index(name):] + [name])
            raise ValueError(f"box() / anchor() loop: {loop}; one of these layers must stop reading the other")
        self._busy.append(name)
        try:
            cr = self._result(L)
        finally:
            self._busy.pop()
        self.results[name] = cr
        return cr

    def _result(self, L):
        name = L["name"]
        info = self.info_of(name)
        cell = L.get("cell") or name
        src = self.cells.get(cell)
        if src is None:
            info["error"] = _err(f"layers/{cell}.py is missing", "MissingCell")
            return None
        c = crop_of(self.pic, L, src)
        k = render_k(float(L.get("scale", 1.0) or 1.0) * (c[2] if c else 1.0), self.W, self.H)
        cr = self._at(L, src, k, info)
        if cr is not None and abs(k - 1.0) > 1e-6 and not cr.extra:
            # drawn larger for sharpness: what the cell draws must grow with the canvas, not faster (a kit size
            # already in 1080p px multiplied by S again grows with k twice); else the 1x drawing is used
            base = self._at(L, src, 1.0, None)
            r = _growth(cr, base, k)
            if r is not None and abs(r - 1.0) > 0.08:
                info["grows"] = round(r, 2)
                return base
        if cr is not None:
            return cr
        good = self.cache.get(L.get("good"))
        if good is not None:
            info["stale"] = True
        return good

    def _at(self, L, src, k, info):
        """The cell's result drawn at k times the picture size: from the cache or run (None on an error)."""
        name, cell = L["name"], L.get("cell") or L["name"]
        pre = self.pre_of(cell)
        key = cache_key(_keyed(src, pre), (self.W, self.H), k, self.seed, self.look)
        cr = self.cache.get(key)
        if cr is not None and not self._deps_ok(cr, name):
            cr = None
        if cr is not None:
            if info is not None:
                info["cached"] = True
                if (self.W, self.H) == self.base:
                    L["good"] = key
            return cr
        path = self.paths.get(cell) or f"layers/{cell}.py"
        try:
            cr = run_cell(src, path, (self.W, self.H), k, self.seed, self.look, name, self, None,
                          self.limit, pre)
        except CellError as e:
            if info is not None:
                info["error"] = e.info
            return None
        self.cache.put(key, cr)
        if info is not None:
            info["ran"], info["seconds"] = True, info.get("seconds", 0.0) + cr.seconds
            if (self.W, self.H) == self.base:
                L["good"] = key
        return cr

    def _deps_ok(self, cr, owner=None):
        for fn, args, val in cr.deps:
            if fn == "file":
                if _file_hash(args[0]) != (val or [None])[0]:
                    return False
                continue
            try:
                now = self._box(args[0]) if fn == "box" else self._anchor(args[0], args[1])
                now = self._for_reader(owner, now) if owner else now
            except Exception:                            # noqa: BLE001
                return False
            if now is None or val is None or len(now) != len(val) or \
                    any(abs(float(a) - float(b)) > 1e-4 for a, b in zip(now, val)):
                return False
        return True

    # ---- box() / anchor() for cells
    def _check_layer(self, owner, name, fn):
        from .._args import unknown_name
        if name not in self.by_name:
            raise unknown_name("layer", name, sorted(self.by_name),
                               hint=f"({fn}() in layer {owner!r}; layers: {', '.join(sorted(self.by_name))})")
        if name == owner:
            raise ValueError(f"{fn}({name!r}) inside layer {name!r}: a layer cannot read its own place")
        if self.role(self.by_name[name]) == "pass":
            raise ValueError(f"{fn}({name!r}): {name} is a pass (it changes the whole picture) and has no place")

    def place_layer(self, name):
        if name in self.placed:
            return self.placed[name]
        cr = self.result(name)
        pl = None if cr is None else place(cr, self.params(self.by_name[name]), self.W, self.H)
        self.placed[name] = pl
        return pl

    def params(self, L):
        """The layer's place and effects, with the picture's crop when it applies to this layer."""
        p = layer_params(L)
        c = crop_of(self.pic, L, self.cells.get(L.get("cell") or L["name"]))
        if c:
            p["crop"] = c
        return p

    def _box(self, name):
        pl = self.place_layer(name)
        if pl is None or pl.box is None:
            raise ValueError(f"box({name!r}): layer {name} draws nothing yet")
        x0, y0, x1, y1 = pl.box
        return [round(x0 / self.W, 4), round(y0 / self.H, 4), round(x1 / self.W, 4), round(y1 / self.H, 4)]

    def anchors_of(self, name):
        """{anchor: (x, y) fractions} of a placed layer: the cell's own landmarks plus the points of its box."""
        pl = self.place_layer(name)
        cr = self.results.get(name)
        out = {}
        if pl is None or pl.box is None:
            return out
        x0, y0, x1, y1 = pl.box
        W, H = self.W, self.H
        out.update(center=((x0 + x1) / 2 / W, (y0 + y1) / 2 / H), top=((x0 + x1) / 2 / W, y0 / H),
                   bottom=((x0 + x1) / 2 / W, y1 / H), left=(x0 / W, (y0 + y1) / 2 / H),
                   right=(x1 / W, (y0 + y1) / 2 / H), top_left=(x0 / W, y0 / H), top_right=(x1 / W, y0 / H),
                   bottom_left=(x0 / W, y1 / H), bottom_right=(x1 / W, y1 / H))
        for a, (fx, fy) in (cr.anchors if cr is not None else {}).items():
            tx, ty = pl.point(fx * cr.size[0], fy * cr.size[1])
            out[a] = (tx / W, ty / H)
        return {a: (round(v[0], 4), round(v[1], 4)) for a, v in out.items()}

    def _anchor(self, name, which):
        an = self.anchors_of(name)
        if not an:
            raise ValueError(f"anchor({name!r}, {which!r}): layer {name} draws nothing yet")
        if which not in an:
            from .._args import unknown_name
            raise unknown_name(f"anchor of {name}", which, sorted(an))
        return list(an[which])

    def _for_reader(self, owner, v):
        """A place read from another layer, in the reading layer's own drawing frame: when the picture is cropped
        and the reader is enlarged with it, the point it draws at is the uncropped one (its own crop moves it
        back), so a lantern placed at the fox's mouth stays at the mouth."""
        L = self.by_name.get(owner)
        if L is None:
            return v
        c = crop_of(self.pic, L, self.cells.get(L.get("cell") or owner))
        if not c:
            return v
        x0, y0, f = c
        return [round(float(a) / f + (x0 if i % 2 == 0 else y0), 4) for i, a in enumerate(v)]

    def box_fn(self, owner):
        def box(name):
            """box("fox") -> (x0, y0, x1, y1): where layer fox is drawn now, in fractions of the picture."""
            self._check_layer(owner, name, "box")
            v = self._for_reader(owner, self._box(name))
            if self.deps_stack:
                self.deps_stack[-1].append(["box", [name], v])
            return tuple(v)
        return box

    def anchor_fn(self, owner):
        def anchor(name, which="center"):
            """anchor("fox", "mouth") -> (x, y) in fractions: a landmark of another layer, where it is now."""
            self._check_layer(owner, name, "anchor")
            v = self._for_reader(owner, self._anchor(name, which))
            if self.deps_stack:
                self.deps_stack[-1].append(["anchor", [name, which], v])
            return tuple(v)
        return anchor

    def scene_fn(self, owner, cell_cv):
        def scene(path):
            """scene("chart.py") -> the whole picture that scene file's render() draws, at this layer's size: a
            chart, diagram, drafting sheet or UI screen from tools/new_scene.py. Use it as ``cv = scene(...)``."""
            import inspect
            from .. import cli
            p = path if os.path.isabs(path) else os.path.join(self.folder or os.getcwd(), path)
            if not os.path.isfile(p) and os.path.isfile(path):
                p = path
            if not os.path.isfile(p):
                raise FileNotFoundError(f"scene({path!r}): no file {os.path.abspath(p)}")
            if self.deps_stack:
                self.deps_stack[-1].append(["file", [os.path.abspath(p)], [_file_hash(p)]])
            mod = cli.import_scene(os.path.abspath(p))
            params = inspect.signature(mod.render).parameters
            args = {"width": cell_cv.w, "height": cell_cv.h}
            if "seed" in params:
                args["seed"] = self.seed
            if "log" in params:
                args["log"] = lambda step: None
            out = mod.render(**args)
            if not isinstance(out, Canvas):
                raise TypeError(f"scene({path!r}): render() returned {type(out).__name__}, not a Canvas")
            if out.shape != cell_cv.shape:
                out.resize(cell_cv.w, cell_cv.h)
            out.__dict__["_studio_whole"] = True
            return out
        return scene

    # ---- the whole picture
    def run(self):
        """Composite every visible layer bottom to top on ``self.canvas``; returns its (H, W, 3) buffer."""
        W, H = self.W, self.H
        cv = Canvas(W, H, self.scene.get("background") or "#000000")
        self.canvas = cv
        styles = []
        light = self.look["light"]
        shc = self.scene.get("shadow_color") or "#1d1830"
        for L in self.pic["layers"]:
            name = L["name"]
            if not L.get("visible", True):
                continue
            if self.log is not None:
                self.log(name)
            if self.role(L) == "pass":
                styles += self._run_pass(L, cv)
                continue
            try:
                pl = self.place_layer(name)
            except ValueError as e:
                self.info_of(name)["error"] = _err(str(e), "ValueError")
                continue
            if pl is None:
                continue
            p = self.params(L)
            op = F32(float(p.get("opacity", 1.0)))
            for mode, col, m, x0, y0 in effects(pl, p, W, H, light, shc):
                paint_piece(cv.buf, mode, col, m * op, x0, y0, cv)
            P, A, x0, y0 = body(pl, p, H)
            over(cv.buf, P, A, x0, y0, cv)
            cr = self.results.get(name)
            styles += list(cr.styles if cr is not None else [])
        if self.log is not None and (self.scene.get("grade") or self.scene.get("paper")):
            self.log("finish")
        if self.scene.get("paper"):
            from ..illustrate import paper
            col = self.scene["paper"] if isinstance(self.scene["paper"], str) else None
            paper(cv, col) if col else paper(cv)
        if self.scene.get("grade"):
            from ..illustrate import grade
            grade(cv, self.scene["grade"], float(self.scene.get("grade_amount", 0.7)))
        cv.__dict__["styles_applied"] = list(dict.fromkeys(styles))
        return cv.buf

    def _run_pass(self, L, cv):
        name = L["name"]
        info = self.info_of(name)
        cell = L.get("cell") or name
        src = self.cells.get(cell)
        if src is None:
            info["error"] = _err(f"layers/{cell}.py is missing", "MissingCell")
            return []
        t0 = time.time()
        try:
            new, st = run_pass(src, self.paths.get(cell) or f"layers/{cell}.py", cv.buf, self.seed, self.look,
                               name, self, None, self.pre_of(cell))
        except CellError as e:
            info["error"] = e.info
            return []
        op = float(L.get("opacity", 1.0))
        cv.buf[...] = cv.buf + (new - cv.buf) * F32(op) if op < 0.999 else new
        info["ran"], info["seconds"] = True, time.time() - t0
        return st

    # ---- what each layer shows of itself
    def visibility(self):
        """{layer: (visible share 0..1, the layer above that covers it most or None)} from the real stack
        order (a layer under a backdrop is hidden), on a coarse grid."""
        if getattr(self, "_vis", None) is not None:
            return self._vis
        W, H = self.W, self.H
        f = max(1, int(np.ceil(max(W, H) / 320.0)))
        sh, sw = int(np.ceil(H / f)), int(np.ceil(W / f))
        stack = []
        for L in self.pic["layers"]:
            pl = self.placed.get(L["name"])
            if not L.get("visible", True) or pl is None or pl.box is None or self.role(L) == "pass":
                continue
            a = pl.full_alpha(W, H) * F32(float(L.get("opacity", 1.0)))
            a = np.pad(a, ((0, sh * f - H), (0, sw * f - W))).reshape(sh, f, sw, f).mean((1, 3))
            stack.append((L["name"], a))
        out = {}
        T = np.ones((sh, sw), F32)
        for i in range(len(stack) - 1, -1, -1):
            name, a = stack[i]
            tot = float(a.sum())
            vis = float((a * T).sum()) / tot if tot > 0 else 1.0
            cover, best = None, 0.0
            if vis < 0.97:
                for j in range(i + 1, len(stack)):
                    ov = float((a * stack[j][1]).sum())
                    if ov > best:
                        cover, best = stack[j][0], ov
            out[name] = (vis, cover)
            T = T * (1.0 - a)
        self._vis = out
        return out

    # ---- the check canvas: one manifest entry per layer, text records moved with their layer
    def annotate(self, cv):
        """Record every visible object layer on ``cv`` (the composite): one manifest entry named by the layer
        (aliases: the names its kit calls recorded) and its text records, moved with it. -> {layer: [orders]}."""
        from .. import text as _text
        W, H = cv.w, cv.h
        owner = {}
        for L in self.pic["layers"]:
            name = L["name"]
            if not L.get("visible", True) or self.role(L) == "pass":
                continue
            pl, cr = self.placed.get(name), self.results.get(name)
            if pl is None or cr is None or pl.box is None:
                continue
            p = self.params(L)
            orders = []
            if cr.extra is not None and self._identity(p, cr):
                owner[name] = self._carry(cv, cr.extra)
                continue
            text_only = not cr.entries and cr.texts and self._text_only(cr)
            if not text_only:
                kinds = [e["kind"] for e in cr.entries]
                kind = max(set(kinds), key=kinds.count) if kinds else "object"
                if kind in ("text", "style", _manifest.CUE):
                    kind = "object"
                label = p.get("label") or base_name(name)
                aliases = []
                for e in cr.entries:
                    aliases += [e["name"]] + list(e.get("aliases") or [])
                if name != label:
                    aliases.append(name)
                extra = p.get("alias")
                if extra:
                    aliases += [extra] if isinstance(extra, str) else list(extra)
                aliases = [a for a in dict.fromkeys(str(a) for a in aliases) if a.lower() != label.lower()]
                meta = {}
                if self.role(L) == "backdrop":
                    meta["backdrop"] = True
                names = {e["name"].split(" #")[0].lower() for e in cr.entries}
                if len(cr.entries) > 1 and len(names) == 1:
                    meta["count"] = int(sum(int((e.get("meta") or {}).get("count", 1) or 1) for e in cr.entries))
                if any((e.get("meta") or {}).get("see_through") for e in cr.entries):
                    meta["see_through"] = True
                m = pl.full_alpha(W, H) * F32(float(p.get("opacity", 1.0)))
                e = _manifest._note_raw(cv, kind, label, mask=m, alias=aliases, **meta)
                if e is not None:
                    orders.append(e["order"])
            for t in cr.texts:
                rec = self._move_text(t, pl, cv, _text)
                if rec is None:
                    continue
                log = cv.__dict__.setdefault("text_log", [])
                log.append(rec)
                _manifest.drawn_text(cv, len(log) - 1)
                orders.append(cv.manifest[-1]["order"])
            owner[name] = orders
        return owner

    def _identity(self, p, cr):
        return (not p.get("crop") and abs(float(p.get("scale", 1)) - 1) < 1e-6 and not float(p.get("rotate", 0) or 0) and not p.get("flip")
                and not float(p.get("dx", 0) or 0) and not float(p.get("dy", 0) or 0)
                and tuple(cr.size) == (self.W, self.H))

    @staticmethod
    def _carry(cv, ex):
        """A whole picture's own records on the check canvas: its entries, text, vector drawings."""
        log = cv.__dict__.setdefault("text_log", [])
        base = len(log)
        log.extend(dict(t) for t in ex.get("text_log") or [])
        orders = []
        for e in ex.get("manifest") or []:
            e = dict(e)
            if e.get("kind") == "text" and "ref" in e:
                e["ref"] = int(e["ref"]) + base
            e["order"] = _manifest._seq(cv)
            cv.manifest.append(e)
            orders.append(e["order"])
        cv.__dict__.setdefault("_drawings", []).extend(ex.get("drawings") or [])
        if ex.get("fit_reports"):
            cv.__dict__.setdefault("fit_reports", []).extend(ex["fit_reports"])
        return orders

    @staticmethod
    def _text_only(cr):
        """The cell drew text and nothing outside its text boxes (a title, a caption)."""
        bb = _alpha_box(cr.rgba[..., 3].astype(F32) / 255.0, 0.5)
        if bb is None:
            return True
        ox, oy = cr.off
        x0, y0, x1, y1 = bb[0] + ox, bb[1] + oy, bb[2] + ox, bb[3] + oy
        tb = [t.get("layout_bbox") or t.get("bbox") for t in cr.texts if t.get("bbox")]
        if not tb:
            return False
        pad = 0.15 * max(1.0, max(b[3] - b[1] for b in tb))
        ux0, uy0 = min(b[0] for b in tb) - pad, min(b[1] for b in tb) - pad
        ux1, uy1 = max(b[2] for b in tb) + pad, max(b[3] for b in tb) + pad
        return x0 >= ux0 and y0 >= uy0 and x1 <= ux1 and y1 <= uy1

    @staticmethod
    def _move_text(t, pl, cv, _text):
        """A text record of a cell, moved / scaled with its layer, its contrast measured on the composite."""
        W, H = cv.w, cv.h
        if not t.get("bbox"):
            return None

        def box_of(b):
            pts = [pl.point(b[0], b[1]), pl.point(b[2], b[1]), pl.point(b[0], b[3]), pl.point(b[2], b[3])]
            return (min(q[0] for q in pts), min(q[1] for q in pts), max(q[0] for q in pts), max(q[1] for q in pts))
        nb = box_of(t["bbox"])
        ink = tuple(int(round(v)) for v in nb)
        if ink[2] <= ink[0] or ink[3] <= ink[1]:
            return None
        rec = dict(t)
        rec["bbox"] = ink
        rec["layout_bbox"] = tuple(round(v, 1) for v in box_of(t.get("layout_bbox") or t["bbox"]))
        rec["size"] = round(float(t.get("size", 0) or 0) * pl.scale_px, 1)
        rec["canvas"] = (W, H)
        rec["overlaps"] = []
        ow, oh = ink[2] - ink[0], ink[3] - ink[1]
        m = t.get("_mask")
        mask = None
        if m is not None and np.asarray(m).size:
            mm = np.asarray(m, bool)
            ratio_src = mm.shape[1] / max(mm.shape[0], 1)
            if abs(ow / max(oh, 1) - ratio_src) <= 0.08 * max(ratio_src, 1.0):   # not rotated: resize the glyphs
                from PIL import Image
                im = Image.fromarray(mm.astype(np.uint8) * 255).resize((max(ow, 1), max(oh, 1)), Image.NEAREST)
                mask = np.asarray(im) > 127
                if pl.point(0, 0)[0] > pl.point(1, 0)[0]:          # flipped layer: mirrored glyphs
                    mask = mask[:, ::-1]
        rec["_mask"] = mask if mask is not None else np.ones((oh, ow), bool)
        x0, y0, x1, y1 = ink
        rec["clipped"] = bool(x0 < -1 or y0 < -1 or x1 > W + 1 or y1 > H + 1)
        sl = _clip(x0, y0, ow, oh, W, H)
        if sl:
            fill = np.zeros((H, W), F32)
            (dy, dx), (sy, sx) = sl
            fill[dy, dx] = rec["_mask"][sy, sx].astype(F32)
            try:
                c, worst, wbox, busy = _text._measure_contrast(cv, fill, ink, max(rec["size"], 1.0))
                rec.update(contrast=None if c is None else round(c, 2),
                           min_contrast=None if worst is None else round(worst, 2), worst_box=wbox, busy=busy)
            except Exception:                            # noqa: BLE001 - keep the cell's own measurement
                pass
        return rec


def _keyed(src, pre):
    """The text a cell's cache key covers: its source and the sources of the sections it takes names from."""
    if not pre:
        return src
    return src + "".join(f"\n# <<provider {p}>>\n{s}" for p, _, s, _ in pre)


def _growth(cr, base, k):
    """How much more than k times the 1x drawing a k-times drawing is (1.0 = as it should be)."""
    import numpy as np
    def ext(c):
        if c is None or not c.rgba.size:
            return None
        a = c.rgba[..., 3] > 128
        if not a.any():
            return None
        ys, xs = np.nonzero(a)
        return max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) / float(c.k)
    e1, ek = ext(base), ext(cr)
    if not e1 or not ek or e1 < 6:
        return None
    return ek / e1


def _file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:16]
    except OSError:
        return None


def render_picture(pic, cells, width=None, height=None, log=None, annotate=True, folder=None):
    """A picture dict + {cell: source} -> Canvas (composite with one manifest entry per layer). Exported scenes
    call this from their ``render()``; layer errors raise (a scene must draw every layer)."""
    size = (int(width or pic["size"][0]), int(height or pic["size"][1]))
    comp = Compositor(pic, cells, size=size, log=log, folder=folder)
    comp.run()
    for name, info in comp.info.items():
        if info.get("error"):
            raise RuntimeError(f"layer {name}: {info['error'].get('short')}")
    cv = comp.canvas
    if annotate:
        comp.annotate(cv)
    return cv
