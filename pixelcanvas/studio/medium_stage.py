"""The medium stage of a studio picture: layers -> buffers (recorded where the sections recorded, estimated
elsewhere) -> one key light -> the medium. It runs after the drawing is clean; the measured checks keep running on
the drawn layers. Buffers are cached per picture state, so changing the medium re-runs no section and no buffer."""
import hashlib
import json
import os
import time

import numpy as np

from .engine import body, place_patch, F32

MEDIA = ("render", "flat", "oil", "watercolour", "ink", "pixel")
DIALS = ("medium_strength", "medium_mark", "medium_palette", "medium_paper")
_MEM = {}                                                # buffer cache in memory: key -> buffers


def setting(scene):
    """(medium or None, dials) from the scene."""
    from ..medium import canonical
    m = canonical((scene or {}).get("medium") or "off")
    if m not in MEDIA:
        return None, {}
    d = {}
    if scene.get("medium_strength") is not None:
        d["strength"] = float(scene["medium_strength"])
    if scene.get("medium_mark") is not None:
        d["mark"] = float(scene["medium_mark"])
    if scene.get("medium_palette") is not None:
        d["palette"] = int(scene["medium_palette"])
    if scene.get("medium_paper"):
        d["paper"] = scene["medium_paper"]
    return m, d


def _hint_horizon(comp, pic):
    """The floor line the drawing knows: a tabletop / Ground record (its top), as a fraction of the height."""
    best = None
    for L in pic.data["layers"]:
        cr = comp.results.get(L["name"])
        for e in (getattr(cr, "entries", None) or []):
            nm = str(e.get("name") or "").lower()
            b = e.get("bbox")
            if b and any(w == nm or nm.startswith(w) for w in ("table", "tabletop", "ground", "floor", "counter")):
                pl = comp.placed.get(L["name"])
                if pl is None:
                    continue
                W, H = cr.size
                y = pl.point(b[0] * W, b[1] * H)[1] / comp.H
                best = y if best is None else min(best, y)
    return best


def layers_for(comp, pic):
    """The visible, non-pass layers as medium inputs (full frame), with their recorded paints placed exactly like
    the layer (move / scale / rotate / crop) and their tint applied."""
    W, H = comp.W, comp.H
    out = []
    from .compositor import text_like
    for L in pic.data["layers"]:
        name = L["name"]
        role = comp.role(L)
        if not L.get("visible", True) or role == "pass":
            continue
        pl = comp.placed.get(name)
        if pl is None or pl.A.size == 0:
            continue
        p = comp.params(L)
        P, A, x0, y0 = body(pl, p, H)
        Pf = np.zeros((H, W, 3), F32)
        Af = np.zeros((H, W), F32)
        ys0, xs0 = max(y0, 0), max(x0, 0)
        ys1, xs1 = min(y0 + A.shape[0], H), min(x0 + A.shape[1], W)
        if ys1 <= ys0 or xs1 <= xs0:
            continue
        Pf[ys0:ys1, xs0:xs1] = P[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        Af[ys0:ys1, xs0:xs1] = A[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        cr = comp.results.get(name)
        src = comp.cells.get(L.get("cell") or name)
        if p.get("outline") and text_like(L, src):       # an outlined text keeps its outline over the medium
            from ..medium import nd
            from ..color import rgb as _rgb
            r = max(1, int(round(float(p.get("outline_width") or 0.004) * H)))
            ring = nd.blur(nd.dilate(Af > 0.3, r).astype(F32), 0.8)
            ring = np.maximum(ring - Af, 0)
            Pf = Pf + ring[..., None] * np.asarray(_rgb(p["outline"])[:3], F32)
            Af = np.clip(Af + ring, 0, 1)
        recs = None
        standing = bool(p.get("shadow") == "ground")
        if cr is not None and getattr(cr, "records", None) and role != "backdrop":
            recs = []
            for r in cr.records:
                m = np.asarray(r["mask"], np.uint8)
                c = np.asarray(r["color"], np.float32)
                h, w = m.shape
                rgba = np.zeros((h, w, 4), np.uint8)
                a = m.astype(np.float32) / 255.0
                col = c if c.ndim == 3 else np.broadcast_to(c.reshape(1, 1, 3), (h, w, 3))
                rgba[..., :3] = np.round(np.clip(col, 0, 1) * a[..., None] * 255).astype(np.uint8)
                rgba[..., 3] = m
                Pr, Ar, rx, ry = place_patch(cr, p, W, H, rgba, r["off"])
                if p.get("tint") and r["kind"] in ("body", "line"):
                    from .engine import tint
                    Pr = tint(Pr, Ar, p["tint"], float(p.get("tint_amount", 1.0)))
                Nr = None
                if r.get("normals") is not None:             # true normals, moved like the layer
                    nrgba = np.zeros((h, w, 4), np.uint8)
                    nrgba[..., :3] = np.round(np.asarray(r["normals"], np.float32) * a[..., None]).astype(np.uint8)
                    nrgba[..., 3] = m
                    Pn, An, _, _ = place_patch(cr, p, W, H, nrgba, r["off"])
                    Nr = np.clip(Pn / np.maximum(An[..., None], 1e-3), 0, 1) * 2 - 1
                    if p.get("flip"):
                        Nr[..., 0] = -Nr[..., 0]
                PPr = None
                if r.get("painted") is not None:             # the colour the call left, moved like the layer
                    prgba = np.zeros((h, w, 4), np.uint8)
                    prgba[..., :3] = np.round(np.asarray(r["painted"], np.float32) * a[..., None]).astype(np.uint8)
                    prgba[..., 3] = m
                    PPr, _, _, _ = place_patch(cr, p, W, H, prgba, r["off"])
                    if p.get("tint"):
                        from .engine import tint
                        PPr = tint(PPr, Ar, p["tint"], float(p.get("tint_amount", 1.0)))
                Hr = None
                if r.get("hollow") is not None:              # the opening of a hollow solid, moved like the layer
                    hrgba = np.zeros((h, w, 4), np.uint8)
                    hrgba[..., :3] = np.asarray(r["hollow"], np.uint8)[..., None]
                    hrgba[..., 3] = 255
                    Ph, Ah, _, _ = place_patch(cr, p, W, H, hrgba, r["off"])
                    Hr = Ph[..., 0] / np.maximum(Ah, 1e-3) * (Ah > 0.5)
                recs.append(dict(kind=r["kind"], A=Ar, P=Pr, x0=rx, y0=ry, volume=r["volume"], gloss=r["gloss"],
                                 strength=r["strength"], order=r["order"], color=r["color"], N=Nr, hollow=Hr, PP=PPr))
                standing = standing or r["kind"] == "shadow"
        entries = []
        for e in (getattr(cr, "entries", None) or []):
            b = e.get("bbox")
            if b and cr is not None:
                Wc, Hc = cr.size
                q0, q1 = pl.point(b[0] * Wc, b[1] * Hc), pl.point(b[2] * Wc, b[3] * Hc)
                entries.append(dict(name=e.get("name"), box=(min(q0[0], q1[0]) / W, min(q0[1], q1[1]) / H,
                                                             max(q0[0], q1[0]) / W, max(q0[1], q1[1]) / H)))
        out.append(dict(name=name, role="backdrop" if role == "backdrop" else "object", P=Pf, A=Af,
                        text=bool(text_like(L, src)), records=recs, entries=entries, standing=standing,
                        desc=(src or "").split("\n", 1)[0]))
    return out


def _key(comp, pic, max_side):
    from ..medium.buffers import VERSION
    from .engine import ENGINE
    parts = [f"{ENGINE}.{VERSION}", str(max_side), json.dumps(pic.data["scene"].get("light")), str(comp.W), str(comp.H)]
    for L in pic.data["layers"]:
        cr = comp.results.get(L["name"])
        parts.append(f"{L['name']}:{getattr(cr, 'key', None)}:{json.dumps(comp.params(L), sort_keys=True, default=str)}"
                     f":{L.get('visible', True)}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def buffers(comp, pic, max_side=1024):
    """The picture's buffers (cached by the picture state) -> (buffers, seconds spent now)."""
    from ..medium.buffers import build
    t0 = time.time()
    key = _key(comp, pic, max_side)
    if key in _MEM:
        return _MEM[key], 0.0
    import pickle
    path = os.path.join(pic.cache_dir, f"buffers_{key}.pkl")
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                b = pickle.load(f)
            _MEM.clear()
            _MEM[key] = b
            return b, time.time() - t0
        except Exception:                                # noqa: BLE001 - rebuilt below
            pass
    sc = pic.data["scene"]
    b = build(layers_for(comp, pic), light=tuple(sc.get("light") or (-0.6, -0.8)), max_side=max_side,
              horizon=_hint_horizon(comp, pic))
    b["key"] = key
    _MEM.clear()                                         # one picture state at a time
    _MEM[key] = b
    try:
        for f in os.listdir(pic.cache_dir):
            if f.startswith("buffers_"):
                os.remove(os.path.join(pic.cache_dir, f))
        with open(path + ".tmp", "wb") as f:
            pickle.dump(b, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".tmp", path)
    except OSError:
        pass
    return b, time.time() - t0


def apply(comp, pic, medium, dials, max_side=1024):
    """The medium picture at the composite's size -> (image float HxWx3, buffers, seconds of buffers, seconds of
    the medium). The painted result is cached in the picture's cache folder by buffers + medium + dials."""
    from ..medium import paint
    from ..medium.nd import resize
    b, tb = buffers(comp, pic, max_side)
    mkey = hashlib.sha1(f"{b['key']}|{medium}|{json.dumps(dials, sort_keys=True)}".encode()).hexdigest()[:16]
    path = os.path.join(pic.cache_dir, f"medium_{mkey}.png")
    t0 = time.time()
    img = None
    if os.path.isfile(path):
        try:
            from PIL import Image
            with Image.open(path) as im:
                img = np.asarray(im.convert("RGB"), np.float32) / 255.0
        except OSError:
            img = None
    if img is None:
        img = paint(b, medium, **dials)
        try:
            from PIL import Image
            for f in os.listdir(pic.cache_dir):           # one painted result per medium is enough
                if f.startswith("medium_") and f != os.path.basename(path):
                    os.remove(os.path.join(pic.cache_dir, f))
            Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
        except OSError:
            pass
    tm = time.time() - t0
    if img.shape[0] != comp.H or img.shape[1] != comp.W:
        img = np.clip(resize(img, comp.W, comp.H, "bicubic"), 0, 1)
    return img.astype(np.float32), b, tb, tm


# ---- gates: the medium picture still shows what the layers show

def _lum(img):
    from ..medium.core import luma
    return luma(np.clip(np.asarray(img, F32), 0, 1))


def _cells(x, n=48):
    """Block means of a 2D map on a grid of about ``n`` cells along the long side."""
    H, W = x.shape
    s = max(1, int(round(max(H, W) / float(n))))
    h, w = H // s, W // s
    return x[:h * s, :w * s].reshape(h, s, w, s).mean((1, 3))


def _ring(mask, r):
    from ..medium import nd
    return nd.dilate(mask, r) & ~mask


def _contrast(img, inside, ring, light_only=False):
    """Oklab distance between the median colours inside and around a shape (lightness only for ink)."""
    from .. import manifest as M
    if int(inside.sum()) < 4 or int(ring.sum()) < 4:
        return None
    a = np.median(img[inside], 0)
    b = np.median(img[ring], 0)
    La, Ca, ha = M._oklch(a[None])
    Lb, Cb, hb = M._oklch(b[None])
    ka = np.array([La[0], Ca[0] * np.cos(np.radians(ha[0])), Ca[0] * np.sin(np.radians(ha[0]))])
    kb = np.array([Lb[0], Cb[0] * np.cos(np.radians(hb[0])), Cb[0] * np.sin(np.radians(hb[0]))])
    if light_only:
        return float(abs(ka[0] - kb[0]))
    return float(np.linalg.norm(ka - kb))


def _strength_fix(dials, medium, why):
    s = float(dials.get("strength", 1.0))
    if s > 0.35:
        return f"set scene medium_strength={max(0.3, round(s - 0.4, 1)):g} ({why})"
    if medium == "render":
        return f"set scene medium=off ({why})"
    return f"set scene medium=render ({why}; {medium} at this strength still loses it)"


def gates(comp, pic, out, img, medium, dials):
    """[(name, ok, measured, fix)] for the medium picture ``img`` (H x W x 3, the composite's size):

    value pattern  the light-dark pattern of the medium picture follows the drawn picture's (correlation of block
                   means of lightness 0.4 or more, or every block within 0.05 of the drawing's)
    colours        every requested colour that reads on the layers still reads on the medium picture
    contrast       every requested thing keeps at least 40 % of its colour contrast against what surrounds it
    text           text keeps at least 60 % of its contrast against its background"""
    from . import check as C
    from ..medium import nd
    from .compositor import text_like
    from .. import manifest as M
    lit = np.clip(np.asarray(comp.canvas.buf, F32)[..., :3], 0, 1)
    img = np.clip(np.asarray(img, F32)[..., :3], 0, 1)
    H, W = lit.shape[:2]
    res = []
    tm = np.zeros((H, W), bool)                          # text is laid over the medium as drawn: not measured here
    for L in pic.data["layers"]:
        pl = comp.placed.get(L["name"])
        if pl is not None and L.get("visible", True) and text_like(L, comp.cells.get(L.get("cell") or L["name"])):
            tm |= nd.dilate(pl.full_alpha(W, H) > 0.2, max(2, int(min(W, H) * 0.01)))
    a0, b0 = _cells(_lum(lit)), _cells(_lum(np.where(tm[..., None], lit, img)))
    from ..medium import nd as _nd

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / den) if den > 1e-9 else 1.0
    # the pattern as a whole, or the things against their surroundings (a vignette or an exposure ramp is not a
    # changed pattern): the better of the two
    r = max(corr(a0, b0), corr(a0 - _nd.blur(a0, 6.0), b0 - _nd.blur(b0, 6.0)))
    dv = float(np.abs((b0 - b0.mean()) - (a0 - a0.mean())).mean())      # a nearly even picture stays even
    ok = r >= 0.4 or dv < 0.05
    res.append(("value pattern", ok, f"follows the drawing {r:.2f} (needs 0.40)" + ("" if r >= 0.4 or not ok else
                f"; values within {dv:.2f}"),
                None if ok else _strength_fix(dials, medium, "its lights and darks drifted from the drawing")))
    rpx = max(3, int(round(min(W, H) * 0.02)))
    worst_c, worst_t, colours_bad = None, None, None
    seen = set()
    for n in out.needs:
        ls = C.layers_for(pic, comp, n.get("key") or "") if n.get("key") else []
        if n.get("cell"):
            ls = list(dict.fromkeys(ls + [x for x in C.family(pic, n["cell"]) if x in pic.names()]))
        for x in ls:
            L = comp.by_name.get(x)
            pl = comp.placed.get(x)
            if L is None or pl is None or not L.get("visible", True) or comp.role(L) in ("backdrop", "pass"):
                continue
            m = pl.full_alpha(W, H) > 0.5
            if m.sum() < 16 or m.sum() > 0.25 * W * H:
                continue
            core = nd.erode(m, 1) if m.sum() > 64 else m
            if n.get("colours") and colours_bad is None and medium != "ink":     # ink is drawn in black
                p0 = C._painted(comp, x)
                if p0 is not None and C.colour_ok(p0, n["colours"])[0]:
                    rgb = img[core]
                    Lk, Ck, hk = M._oklch(rgb)
                    sel = np.ones(len(Lk), bool)
                    if len(Lk) >= 30:
                        lo, hi = np.percentile(Lk, [10, 90])
                        sel = (Lk >= lo) & (Lk <= hi)
                    ok, mostly = C.colour_ok(M._Paint(Lk[sel], Ck[sel], hk[sel], rgb[sel]), n["colours"])
                    if not ok:
                        colours_bad = (x, " and ".join(n["colours"]), mostly)
            if x in seen:
                continue
            seen.add(x)
            ring = _ring(m, rpx)
            mono = medium == "ink"
            cl, cm = _contrast(lit, core, ring, mono), _contrast(img, core, ring, mono)
            if cl is None or cm is None or cl < 0.06:
                continue
            q = cm / cl
            if worst_c is None or q < worst_c[1]:
                worst_c = (x, q)
    if medium == "ink":
        res.append(("colours", True, "ink is drawn in black on paper: the colours are the drawing's", None))
    elif colours_bad:
        x, want, mostly = colours_bad
        if medium in ("pixel", "flat"):
            fix = f"set scene medium_palette={max(int(dials.get('palette') or 20) * 2, 24)} (the {want} of {x} " \
                  f"was merged into another colour)"
        else:
            fix = _strength_fix(dials, medium, f"the {want} of {x} reads as {mostly}")
        res.append(("colours", False, f"{x} should read {want}; the medium makes it {mostly}", fix))
    else:
        res.append(("colours", True, "every requested colour still reads", None))
    if worst_c is not None and worst_c[1] < 0.4:
        x, q = worst_c
        res.append(("contrast", False, f"{x} keeps {q * 100:.0f}% of its contrast against its ground (needs 40%)",
                    _strength_fix(dials, medium, f"{x} melts into its ground")))
    else:
        res.append(("contrast", True, "every requested thing stands out from its ground" +
                    (f" (least: {worst_c[0]} {worst_c[1] * 100:.0f}%)" if worst_c else ""), None))
    for L in pic.data["layers"]:
        src = comp.cells.get(L.get("cell") or L["name"])
        pl = comp.placed.get(L["name"])
        if pl is None or not L.get("visible", True) or not text_like(L, src):
            continue
        m = pl.full_alpha(W, H) > 0.5
        if m.sum() < 8:
            continue
        ring = _ring(m, max(2, rpx // 2))
        cl, cm = _contrast(lit, m, ring), _contrast(img, m, ring)
        if cl is None or cm is None or cl < 0.06:
            continue
        q = cm / cl
        if worst_t is None or q < worst_t[1]:
            worst_t = (L["name"], q)
    if worst_t is not None and worst_t[1] < 0.6:
        x, q = worst_t
        res.append(("text", False, f"{x} keeps {q * 100:.0f}% of its contrast (needs 60%)",
                    f"set {x} outline=on (the text sits on a busy {medium} ground)"))
    elif worst_t is not None:
        res.append(("text", True, f"text stays legible ({worst_t[0]} {worst_t[1] * 100:.0f}%)", None))
    return res
