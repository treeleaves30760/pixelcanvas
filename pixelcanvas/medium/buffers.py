"""Scene buffers from the studio's layers: albedo without the painted shading, a relief height per recorded shape
(per layer where nothing was recorded), normals, depth from draw order, ids, emission, a floor / wall split, one key
light + ambient, cast shadows on the floor, relief shadows between layers, contact occlusion, specular.

Recorded (``Sketch.paint`` in a studio section, see ``medium.record``): each paint's local colour (the albedo),
its volume hint (relief per shape; a later shape sits on the earlier one; negative = hollow), its gloss, and its
contact shadow and glow as shadow / light (removed from the thing; the light becomes emission). Estimated, for layers
drawn by kits through ``cv.paint`` with nothing recorded: the painted shading is fitted and divided out, painted
contact shadows are guessed and removed, the relief is the layer inflated (plus colour parts that stick out).
"""
import math
import re
import time

import numpy as np
from PIL import Image

from . import nd
from .core import F, srgb_to_lin, luma, blur

EST_RELIEF = 0.35       # relief of a layer with no volume information, as a share of a recorded volume= shape
FLAT_WORDS = ('puddle', 'rug', 'carpet', 'mat', 'petal', 'shard', 'debris', 'spill', 'water', 'decal', 'pattern',
              'picture', 'painting', 'poster', 'sign', 'label', 'stain', 'paper', 'map', 'crumb', 'confetti', 'tile',
              'reflection', 'leaves', 'litter', 'splash', 'glare', 'road', 'path', 'floor', 'ground', 'shadow',
              'stripe', 'marking', 'lines', 'grass', 'soil', 'dirt', 'sand', 'snow', 'pond', 'lake', 'river', 'sea')
DESHADE = 0.65          # share of the painted low-frequency shading divided out (estimated layers)
GLOSSY = {'glass': 0.9, 'metal': 0.8, 'steel': 0.8, 'chrome': 0.9, 'ceramic': 0.55, 'porcelain': 0.6,
          'glossy': 0.7, 'plastic': 0.45, 'shiny': 0.6, 'wet': 0.5, 'mug': 0.55, 'bottle': 0.8,
          'sink': 0.55, 'knife': 0.8, 'blade': 0.8, 'bowl': 0.5, 'moto': 0.6, 'motorcycle': 0.6}
VERSION = 12                                              # bump when the buffers change (studio cache key)
LIGHTS = ('fire', 'flame', 'campfire', 'candle', 'lamp', 'lantern', 'torch', 'fireworks', 'neon', 'bulb', 'ember',
          'sparkler', 'lightning', 'glow')     # a moon or sun disc is far away: painted as paint, not a light here


# ---------------------------------------------------------------- helpers
def resize(a, W, H, mode=Image.BOX):
    if a.shape[1] == W and a.shape[0] == H:
        return np.asarray(a, F)
    if a.ndim == 3:
        return np.stack([resize(a[..., c], W, H, mode) for c in range(a.shape[2])], -1)
    return np.asarray(Image.fromarray(a.astype(np.float32), mode='F').resize((W, H), mode), np.float32)


membrane = nd.membrane


def robust_shading(logY, mask, wts, extra_feats=()):
    """Fit log-luminance with a smooth low-order field (quadratic in x, y + rim terms), robust to local detail.
    Returns the fitted field over the mask (0 elsewhere)."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return np.zeros(mask.shape, F)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    X = (xs - (x0 + x1) / 2) / max((x1 - x0) / 2, 1)
    Yc = (ys - (y0 + y1) / 2) / max((y1 - y0) / 2, 1)
    feats = [np.ones_like(X), X, Yc, X * X, Yc * Yc, X * Yc] + [e[ys, xs] for e in extra_feats]
    M = np.stack(feats, 1)
    v = logY[ys, xs]
    w = wts[ys, xs]
    step = max(1, len(v) // 40000)
    Ms, vs, ws = M[::step], v[::step], w[::step]
    coef = np.zeros(M.shape[1])
    rw = np.ones(len(vs))
    for _ in range(5):
        ww = ws * rw
        A = Ms * ww[:, None]
        coef = np.linalg.lstsq(A.T @ Ms + 1e-4 * np.eye(M.shape[1]), A.T @ vs, rcond=None)[0]
        r = vs - Ms @ coef
        s = 1.4826 * np.median(np.abs(r)) + 1e-3
        rw = 1 / (1 + (r / (1.5 * s)) ** 2)
    out = np.zeros(mask.shape, F)
    out[ys, xs] = M @ coef
    return out


def colour_parts(albedo, mask, k=5, seed=0):
    """Colour regions of one layer: [(part mask, is_pattern)]. A colour class split into many pieces is a pattern
    (stripes, spots); a class in one or two pieces that reaches the silhouette is a part (handle, leg, wheel)."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return []
    X = albedo[ys, xs]
    lum = luma(X)[:, None]
    feat = np.concatenate([X / (lum + 0.05) * 0.5, np.log(lum + 0.02) * 0.35], 1)
    rng = np.random.default_rng(seed)
    S = feat[rng.choice(len(feat), min(len(feat), 6000), replace=False)]
    C = S[rng.choice(len(S), k, replace=False)]
    for _ in range(10):
        lab = np.argmin(((S[:, None] - C[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                C[j] = S[lab == j].mean(0)
    lab = np.argmin(((feat[:, None] - C[None]) ** 2).sum(-1), 1)
    L = -np.ones(mask.shape, np.int32)
    L[ys, xs] = lab
    edge = mask & ~nd.erode(mask, 2)
    area = mask.sum()
    parts = []
    for j in range(k):
        cm = nd.opening(L == j, 1)
        cc, n = nd.label(cm)
        if n == 0:
            continue
        sizes = np.bincount(cc.ravel(), minlength=n + 1)[1:]
        big = [i + 1 for i, s in enumerate(sizes) if s > 0.03 * area]
        small = sum(1 for s in sizes if s > 0.002 * area)
        pattern = small >= 4
        for i in big:
            pm = cc == i
            if pm.sum() > 0.5 * area or pm.sum() < 0.05 * area or pattern:
                continue
            per = pm & ~nd.erode(pm)
            touch = (per & nd.dilate(edge, 1)).sum() / max(per.sum(), 1)
            rest = mask & ~pm
            a, r = albedo[pm].mean(0), albedo[rest].mean(0)
            la, lr = luma(a[None])[0] + 1e-3, luma(r[None])[0] + 1e-3
            chroma = np.abs(a / la - r / lr).max() / 3
            ratio = max(la / lr, lr / la)
            hull = nd.fill_holes(pm)
            solid = pm.sum() / max(hull.sum(), 1)
            if touch > 0.4 and solid > 0.75 and (chroma > 0.12 or ratio > 3.0):
                parts.append(pm)
    return parts


# ---------------------------------------------------------------- recorded shapes
def _window(crop, x0, y0, W, H, f, pad=2):
    """A crop at frame pixel (x0, y0) scaled by f -> (the crop clipped to the frame with ``pad`` px around it, the
    frame slices), or None when it is off the frame. Every per-shape step works inside this window."""
    crop = np.asarray(crop, F)
    h, w = crop.shape[:2]
    X0, Y0 = int(round(x0 * f)), int(round(y0 * f))
    nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
    if (nw, nh) != (w, h):
        crop = resize(crop, nw, nh, Image.BILINEAR)
    ys0, xs0 = max(Y0 - pad, 0), max(X0 - pad, 0)
    ys1, xs1 = min(Y0 + nh + pad, H), min(X0 + nw + pad, W)
    if ys1 <= ys0 or xs1 <= xs0:
        return None
    out = np.zeros((ys1 - ys0, xs1 - xs0) + crop.shape[2:], F)
    cy0, cx0 = max(Y0, ys0), max(X0, xs0)
    cy1, cx1 = min(Y0 + nh, ys1), min(X0 + nw, xs1)
    if cy1 > cy0 and cx1 > cx0:
        out[cy0 - ys0:cy1 - ys0, cx0 - xs0:cx1 - xs0] = crop[cy0 - Y0:cy1 - Y0, cx0 - X0:cx1 - X0]
    return out, (slice(ys0, ys1), slice(xs0, xs1))


def _recorded(L, W, H, f, puff):
    """Albedo, relief, cover, shadow and light of a layer from its recorded paints (None when nothing usable)."""
    recs = sorted(L.get("records") or [], key=lambda r: r["order"])
    body = [r for r in recs if r["kind"] in ("body", "line", "solid")]
    if not body:
        return None
    alb = np.zeros((H, W, 3), F)
    cov = np.zeros((H, W), F)
    Nrec = np.zeros((H, W, 3), F)
    ncov = np.zeros((H, W), F)
    Hl = np.zeros((H, W), F)
    shad = np.zeros((H, W), F)
    emit = np.zeros((H, W, 3), F)
    pp = np.zeros((H, W, 3), F)                          # what the recorded calls left (sRGB), -1 where unknown
    known = np.zeros((H, W), F)
    gl_w, gl_s = 0.0, 0.0
    step = 0.004 * H
    for r in recs:
        wa = _window(r["A"], r["x0"], r["y0"], W, H, f)
        if wa is None:
            continue
        a, sl = wa
        a = np.clip(a, 0, 1)
        if not a.any():
            continue
        if r["kind"] == "shadow":
            shad[sl] = np.maximum(shad[sl], a)
            continue
        if r["kind"] == "light":
            c = srgb_to_lin(np.clip(np.asarray(r["color"], F).reshape(-1)[:3], 0, 1))
            emit[sl] += a[..., None] * c * F(0.6 * max(r["strength"], 0.2))
            continue
        Pp = _window(r["P"], r["x0"], r["y0"], W, H, f)[0]
        c = np.clip(Pp / np.maximum(a[..., None], 1e-4), 0, 1)
        cl = srgb_to_lin(c)
        a3 = a[..., None]
        alb[sl] = alb[sl] * (1 - a3) + cl * a3
        cov[sl] = cov[sl] * (1 - a) + a
        if r.get("PP") is not None or r["kind"] == "line":
            q = c if r.get("PP") is None else np.clip(_window(r["PP"], r["x0"], r["y0"], W, H, f)[0] /
                                                       np.maximum(a3, 1e-4), 0, 1)
            pp[sl] = pp[sl] * (1 - a3) + q * a3
            known[sl] = known[sl] * (1 - a) + a
        else:
            known[sl] = known[sl] * (1 - a)
        if r["kind"] != "line":                          # a glow is light around a thing, not on it: the thing keeps
            e = emit[sl]                                 # its local colour (a bright flame body stays a light)
            if e.max() > 1e-3:
                hot = (luma(cl) > 0.45)[..., None] & (e.max(-1, keepdims=True) > 1e-3)
                emit[sl] = e * (1 - a3) + np.where(hot, e + cl * 0.8, 0) * a3
        ncov[sl] = ncov[sl] * (1 - a)
        if r.get("N") is not None:                       # a form solid: its true normals (y up in the buffers)
            n = _window(r["N"], r["x0"], r["y0"], W, H, f)[0]
            n = np.stack([n[..., 0], -n[..., 1], n[..., 2]], -1)
            Nrec[sl] = Nrec[sl] * (1 - a3) + n * a3
            ncov[sl] = ncov[sl] + a
        if r["kind"] == "line":
            continue
        m = a > 0.5
        if m.sum() < 12:
            continue
        Hw = Hl[sl]
        under = Hw[m]
        z0 = float(np.percentile(under, 60)) + step if under.max() > 0 else 0.0
        v = float(r["volume"])
        if abs(v) >= 0.05:
            h = membrane(m) * puff * min(abs(v), 1.0) ** 0.5
            zl = z0 - h * 0.8 if v < 0 else z0 + h       # a hollow dips into what it is drawn on
            zl = np.maximum(zl, z0 - h * 0.8)
            if r.get("hollow") is not None:              # a solid's opening (a bowl's inside), as recorded: it dips
                hollow = m & (_window(r["hollow"], r["x0"], r["y0"], W, H, f)[0] > 0.5)
                if hollow.sum() > 0.01 * m.sum():
                    zl = np.where(hollow, z0 + h * 0.15 - membrane(hollow) * puff * 0.35, zl)
        else:
            zl = np.full(m.shape, z0, F)                 # no volume hint: a flat shape on top
        Hl[sl] = np.where(m, zl, Hw).astype(F)
        if r["gloss"] > 0:
            n_ = float(m.sum())
            gl_w += n_
            gl_s += n_ * min(0.9, 0.35 + 0.6 * r["gloss"])
    gloss = gl_s / gl_w if gl_w else 0.0
    nn = np.linalg.norm(Nrec, axis=-1, keepdims=True)
    Nrec = np.where(nn > 1e-3, Nrec / np.maximum(nn, 1e-3), np.array([0, 0, 1], F))
    return dict(alb=alb, cov=cov, H=Hl, shad=shad, emit=emit, gloss=gloss, N=Nrec.astype(F), ncov=ncov,
                painted=pp / np.maximum(known, 1e-3)[..., None], known=known)


def _kit_lights(L, C, A, W, H, f):
    """A kit's light sources (a campfire's flames, a lamp) painted as paint: their bright pixels are
    emission, recognised from the kit's own records (manifest names) inside their boxes."""
    out = np.zeros((H, W, 3), F)
    for e in L.get("entries") or []:
        nm = str(e.get("name") or "").lower()
        if not any(w in nm for w in LIGHTS) or not e.get("box"):
            continue
        x0, y0, x1, y1 = e["box"]
        X0, Y0, X1, Y1 = int(x0 * W), int(y0 * H), int(np.ceil(x1 * W)), int(np.ceil(y1 * H))
        if X1 <= X0 or Y1 <= Y0:
            continue
        sub = C[Y0:Y1, X0:X1]
        lum = luma(sub)
        warm = sub[..., 0] - sub[..., 2]
        cool = "bulb" in nm or "lamp" in nm
        hot = (lum > (0.35 if cool else 0.3)) & ((warm > 0.15) | cool) & (A[Y0:Y1, X0:X1] > 0.3)
        k = 0.6 if cool else 1.6
        out[Y0:Y1, X0:X1] = np.maximum(out[Y0:Y1, X0:X1], srgb_to_lin(sub) * hot[..., None] * F(k))
    return out


def _edge_fill(v, mask, A, r=2, unit=False):
    """The anti-aliased edge band of a layer takes the values of its solid interior, extended outward: normals and
    colours at a silhouette never turn into a dark or bright outline."""
    core = nd.erode(mask, r)
    if core.sum() < 20:
        return v
    w = core.astype(F)
    sg = r + 1.0
    ext = blur(v * w[..., None], sg) / np.maximum(blur(w, sg), 1e-3)[..., None]
    if unit:                                             # normals stay unit length
        ext = ext / np.maximum(np.linalg.norm(ext, axis=-1, keepdims=True), 1e-4)
    band = (A > 0.01) & ~core
    return np.where(band[..., None], ext, v).astype(F)


def _lies_flat(L, mask, ncomp, horizon, H, W):
    """A layer with no volume information that is flat: it lies on the floor or hangs on the wall (puddles, rugs,
    petals, shards, pictures, decals, patterns: judged by its name, its footing and its shape), or it is many small
    fragments that are never inflated one by one."""
    name = re.sub(r"\d+$", "", str(L.get("name") or "").lower())
    if any(w in re.split(r"[^a-z]+", name) or name.endswith(w) for w in FLAT_WORDS):
        return True
    n = float(mask.sum())
    if n < 1:
        return True
    if ncomp > 6:
        lab, k = nd.label(mask)
        big = np.bincount(lab.reshape(-1))[1:].max() if k else 0
        if big < 0.5 * n:
            return True
    ys, xs = np.nonzero(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    on_floor = horizon is None or ys.max() > float(horizon) * H
    return bool(h < 0.22 * w and on_floor and h < 0.12 * H)


# ---------------------------------------------------------------- the build
def build(layers, light=(-0.6, -0.8), max_side=1200, horizon=None, puff=0.55, part_puff=0.5, verbose=False):
    """Buffers from studio layers (back to front): ``dict(name, role, P, A, text, records, entries, standing)``,
    full-frame premultiplied sRGB colour and alpha. ``horizon``: the floor line as a fraction of the height when
    the picture knows it (a tabletop, a Ground), else found from the backdrop. Returns the buffer dict the media
    read; ``b['estimated']`` names the layers whose volume was estimated."""
    t0 = time.time()
    H0, W0 = layers[0]["A"].shape if layers else (2, 2)
    f = min(1.0, max_side / max(W0, H0))
    W, H = int(round(W0 * f)), int(round(H0 * f))
    lx, ly = light
    l2 = np.array([lx, ly], F) / (math.hypot(lx, ly) + 1e-9)        # screen direction toward the light (y down)
    elev = math.radians(42)
    Lv = np.array([l2[0] * math.cos(elev), -l2[1] * math.cos(elev), math.sin(elev)], F)
    Lv /= np.linalg.norm(Lv)
    work = []
    for L in layers:
        P, A = resize(L["P"], W, H), np.clip(resize(L["A"], W, H), 0, 1)
        work.append(dict(L, P=P, A=A))
    layers = work
    backdrop = [L for L in layers if L["role"] == "backdrop"]
    objs = [L for L in layers if L["role"] != "backdrop" and not L.get("text")]
    texts = [L for L in layers if L.get("text")]
    yy, xx = np.mgrid[0:H, 0:W].astype(F)
    estimated, n_baked = [], 0
    drawn = np.zeros((H, W, 3), F)                       # the drawing itself (sRGB): the colours the media keep
    for L in layers:
        if L["role"] == "backdrop":
            drawn = np.clip(L["P"] / np.maximum(L["A"][..., None], 1e-3), 0, 1) * L["A"][..., None] +                 drawn * (1 - L["A"][..., None])
        else:
            drawn = drawn * (1 - L["A"][..., None]) + np.clip(L["P"], 0, None)
    drawn = np.clip(drawn, 0, 1)

    # ---- per layer: colour, emission, albedo, relief (recorded per shape, else estimated)
    for k, L in enumerate(layers):
        P, A = L["P"], L["A"]
        e = np.clip(P - A[..., None], 0, None)                       # light beyond paint: a glow (emission)
        C = np.clip(np.where(A[..., None] > 1e-3, (P - e) / np.maximum(A[..., None], 1e-3), 0), 0, 1)
        el = srgb_to_lin(np.clip(e, 0, 1))
        el = np.where((luma(el) > 0.02)[..., None], el, 0)           # faint glow haze is not a light source
        el = el + _kit_lights(L, C, A, W, H, f)
        rec = None if L.get("text") or L["role"] == "backdrop" else _recorded(L, W, H, f, puff)
        L["recorded"] = rec is not None
        if rec is not None:
            # painted contact shadows are not the thing: where only a shadow record covers, the layer is empty
            only_shadow = (rec["cov"] < 0.05) & (rec["shad"] > 0.02)
            A = np.where(only_shadow, 0, A)
            P = np.where(only_shadow[..., None], 0, P)
            el = el + rec["emit"]
            L.update(P=P, A=A)
        Cl = srgb_to_lin(C)
        mask = A > 0.6                       # see-through paint (glows, smoke, light pools) gets no relief
        bright = luma(el) > 0.05
        mask = mask & ~(bright & (luma(Cl) > 0.2))      # flames, lamp glass: emission, no body
        L.update(C=Cl, emit=el.astype(F), mask=mask)
        if L.get("text") or mask.sum() < 30:
            L.update(albedo=Cl, H=np.zeros((H, W), F), gloss=0.0, parts=[], frag=False)
            continue
        cc, ncomp = nd.label(mask)
        frag = ncomp > 25 and mask.sum() / ncomp < 0.0004 * W * H
        if rec is not None:
            got = rec["cov"] > 0.5
            ra = rec["alb"] / np.maximum(rec["cov"], 1e-3)[..., None]
            # paint the records do not know (a kit or cv.paint inside the section, drawn over a recorded shape):
            # its hue differs from the recorded colour where shading alone cannot explain it -> keep what is drawn
            # differs from what the recorded calls left -> keep what is drawn there
            drift = np.abs(C - rec["painted"]).max(-1) * (rec["known"] > 0.5)
            alb = np.where((got & (drift < 0.16))[..., None], ra, Cl)
            Hh = rec["H"]
            rest = mask & ~got                               # a kit drew here inside the section: inflate it
            if rest.sum() > 0.02 * mask.sum() and rest.sum() > 200 and not frag:
                Hh = np.where(rest, membrane(rest) * puff * EST_RELIEF + float(np.percentile(Hh[mask], 50)), Hh)
            Hh = (blur(Hh, 0.8) * mask).astype(F)
            gl = rec["gloss"]
            L.update(albedo=np.clip(alb, 0, 1).astype(F), H=Hh, gloss=gl, parts=[], frag=frag,
                     shade_fit=np.zeros((H, W), F), Nrec=rec["N"], ncov=rec["ncov"])
            continue
        # ---- estimated: no records (kits that draw through cv.paint, layers from scenes)
        if L["role"] != "backdrop":
            estimated.append(L["name"])
            ys = np.nonzero(A > 0.5)[0]
            if len(ys):                                  # contact shadows painted into the layer: ours replace them
                y0, y1 = ys.min(), ys.max()
                low = (np.arange(H)[:, None] > y1 - 0.2 * (y1 - y0 + 1)) & (A > 0.02)
                baked = low & (A < 0.9) & (luma(np.clip(C, 0, 1)) < 0.45)
                baked = nd.dilate(baked, 1) & (A < 0.9)
                if baked.sum() > 30:
                    n_baked += 1
                    A = np.where(baked, 0, A)
                    L["P"] = np.where(baked[..., None], 0, L["P"])
                    L["A"] = A
                    mask = mask & ~baked
                    L["mask"] = mask
        Y = np.clip(luma(Cl), 1e-4, None)
        logY = np.log(Y)
        rim = nd.edt(mask)
        rim = rim / (rim.max() + 1e-6)
        if L["role"] == "backdrop":
            Hh = np.zeros((H, W), F)
            shade = np.zeros((H, W), F)
            parts = []
        else:
            shade = robust_shading(logY, mask, A, (np.sqrt(rim), rim))
            flat = frag or _lies_flat(L, mask, ncomp, horizon, H, W)
            parts = [] if flat else colour_parts(Cl / np.exp(shade - np.percentile(shade[mask], 90))[..., None]
                                                 .clip(0.01, 50), mask)
            # no volume information: a gentle relief (a fraction of a recorded volume= shape), flat things flat
            Hh = np.zeros((H, W), F) if flat else membrane(mask) * puff * EST_RELIEF
            for pm in parts:
                Hh = Hh + membrane(pm) * part_puff * EST_RELIEF * 0.5
            Hh = blur(Hh, 1.0) * mask
            L["flat"] = flat
        if L["role"] != "backdrop":
            ref = np.percentile(shade[mask], 75)
            # when in doubt keep the drawn colour: the de-shading may lift or lower a pixel by 25 % at most, so a
            # grey animal stays grey and a brown one brown
            sfield = np.clip(np.exp(np.clip(shade - ref, -2.5, 0.6) * DESHADE), 0.8, 1.25)
            alb = Cl / sfield[..., None]
            alb = _edge_fill(alb, mask, A)
            alb = alb / np.maximum(alb.max(-1, keepdims=True), 1.0)      # keep the hue (no per-channel clip)
            resid = logY - shade
            sat = Cl.max(-1) - Cl.min(-1)
            hl = mask & (resid > np.percentile(resid[mask], 97) + 0.05) & (resid > 0.35) & \
                (sat < 0.35 * Cl.max(-1) + 0.05)
            gl = min(0.5, hl.sum() / max(mask.sum(), 1) * 6)
            desc = (L.get("desc") or "") + " " + L["name"]
            for kw, g in GLOSSY.items():
                if kw in desc.lower():
                    gl = max(gl, g)
            if frag:
                gl = 0.0
        else:
            alb, gl = Cl, 0.0
        L.update(albedo=alb.astype(F), H=Hh.astype(F), gloss=gl, parts=parts, frag=frag, shade_fit=shade)

    # ---- the backdrop: floor / wall split
    bases = []
    for L in objs:
        m = L["mask"]
        if m.sum() < 0.002 * W * H or L.get("frag"):
            continue
        bases.append(np.percentile(np.nonzero(m)[0], 99.5))
    if horizon is not None:
        horizon = int(np.clip(float(horizon) * H, 0, H - 1))
    elif backdrop:
        B = backdrop[0]
        bl = luma(B["C"])
        prof = np.abs(np.diff(blur(bl, 2).mean(1)))
        lo = int(0.15 * H)
        standing = [y for y in bases if y > 0.45 * H]
        hi = int(np.percentile(standing, 75) if standing else 0.85 * H)
        hi = max(hi, lo + 2)
        cand = prof[lo:hi]
        if len(cand) and cand.max() > 0.004:
            horizon = lo + int(np.argmax(cand))
        else:
            horizon = int(min(bases) - 0.12 * H) if bases else int(0.6 * H)
        tops = [np.nonzero(L["mask"])[0].min() for L in objs if L["mask"].sum() > 0.002 * W * H
                and not L.get("frag")]
        if tops and horizon < min(tops) - 0.01 * H:
            horizon = 0
        horizon = int(np.clip(horizon, 0, H - 1))
    else:
        horizon = int(0.6 * H)
    floor = (yy >= horizon)
    if horizon == 0:                   # seen from above (everything is floor): a high sun, short shadows
        elev = math.radians(66)
        Lv = np.array([l2[0] * math.cos(elev), -l2[1] * math.cos(elev), math.sin(elev)], F)
        Lv /= np.linalg.norm(Lv)

    # ---- global relief and order
    order_step = 0.02 * H
    Z = np.zeros((H, W), F)
    idmap = -np.ones((H, W), np.int32)
    gidmap = -np.ones((H, W), np.int32)
    topA = np.zeros((H, W), F)
    alb_c = np.zeros((H, W, 3), F)
    names = []
    base_z = {}
    for k, L in enumerate(layers):
        if L.get("text"):
            continue
        m = L["mask"]
        if L["role"] == "backdrop":
            z0 = 0.0
        else:                          # drawn later = in front: sits on whatever relief it covers
            under = Z[m] if m.any() else np.zeros(1)
            z0 = float(np.percentile(under, 60)) + (order_step if not L.get("flat") else 0.15 * order_step)
        base_z[k] = z0
        zl = z0 + L["H"]
        cover = m if L["role"] != "backdrop" else np.ones((H, W), bool)
        if L.get("frag"):
            cover = np.zeros((H, W), bool)          # particles (snow, sparks, shards): no body of their own
        Z = np.where(cover, zl, Z)
        idmap = np.where(cover, len(names), idmap)
        gidmap = np.where(cover, k, gidmap)
        names.append(L["name"])
        for pm in L["parts"]:
            idmap = np.where(pm & cover, len(names), idmap)
            names.append(L["name"] + ".part")
        topA = np.where(cover, L["A"], topA)

    # ---- normals from the relief (per layer, so silhouettes do not smear across layers)
    N = np.zeros((H, W, 3), F)
    N[..., 2] = 1
    if backdrop:
        N = np.where(floor[..., None], np.array([0, 0.82, 0.57], F), N)
    for k, L in enumerate(layers):
        if L["role"] == "backdrop" or L.get("text"):
            continue
        Hh = blur(L["H"], 1.2)
        gy, gx = np.gradient(Hh)
        n = np.stack([-gx, gy, np.ones_like(gx)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        if L.get("Nrec") is not None:                    # recorded normals win where a solid recorded them
            n = np.where((L["ncov"] > 0.5)[..., None], L["Nrec"], n)
        n = _edge_fill(n, L["mask"], L["A"], unit=True)
        N = np.where((gidmap == k)[..., None], n, N)

    # ---- light: diffuse, relief shadows, floor shadows, occlusion, specular
    bidx = layers.index(backdrop[0]) if backdrop else -9
    diff = np.clip((N * Lv).sum(-1), 0, 1)
    slope = math.tan(elev)
    R = int(0.12 * max(W, H))
    occl = np.zeros((H, W), F)
    for t in np.unique(np.geomspace(1, max(R, 2), 28).astype(int)):
        Zs = nd.sample(Z, yy + l2[1] * t, xx + l2[0] * t)
        occl = np.maximum(occl, (Zs - Z - t * slope) / (0.25 * t + 2))
    shadow = np.clip(1 - occl, 0, 1)
    fshadow = np.zeros((H, W), F)
    if backdrop and horizon > 0:
        ax, ay = -l2[0] * 0.9, -0.28                 # away from the light along the floor, receding (up)
        for k, L in enumerate(layers):
            if L["role"] == "backdrop" or L.get("text") or L.get("frag"):
                continue
            m = L["mask"]
            if m.sum() < 0.001 * W * H:
                continue
            ys, xs = np.nonzero(m)
            yb = np.percentile(ys, 99.5)
            if yb < horizon + 0.01 * H and not L.get("standing"):
                continue                              # hanging / flying: no floor shadow
            h = np.clip(yb - ys, 0, None)
            sx = (xs + h * ax).astype(int)
            sy = (yb + h * ay).astype(int)
            ok_ = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)     # clipped only by the frame
            fade = np.exp(-h / (0.9 * (yb - ys.min() + 1)))        # fades with the distance from the foot
            acc = np.zeros((H, W), F)
            np.maximum.at(acc, (sy[ok_], sx[ok_]), fade[ok_].astype(F))
            hit = nd.closing(acc > 0, 3)
            val = blur(acc, 1.5) / np.maximum(blur((acc > 0).astype(F), 1.5), 1e-3)
            acc = (hit * np.clip(val, 0, 1)).astype(F)
            fshadow = np.maximum(fshadow, blur(acc, 2.0) * 0.6 + blur(acc, 8.0) * 0.4)
        band = 0.06 * H                               # onto the wall it fades out softly, never a straight cut
        fshadow = fshadow * np.clip((yy - (horizon - band)) / band, 0, 1)
    if horizon > 0:                    # standing scene: no drop shadows on the wall / floor from the relief
        shadow = np.where(gidmap == bidx, 1.0, shadow)
    shadow = np.where(gidmap == bidx, np.minimum(shadow, 1 - 0.85 * fshadow), shadow)
    ao = np.ones((H, W), F)
    for r in (3, 8, 20):                 # creases and contacts; a big step to the layer behind is not a crease
        ao -= np.clip(blur(Z, r) - Z, 0, 0.5 * r) / (r * 1.5) * 0.35
    ao = np.where(gidmap == bidx, 1.0, ao)   # no contact occlusion against the backdrop behind an object
    if backdrop:
        below = np.zeros((H, W), F)
        for L in objs:
            m = L["mask"]
            if m.sum() < 0.001 * W * H or L.get("frag"):
                continue
            ys = np.nonzero(m)[0]
            yb = np.percentile(ys, 99.5)
            if yb < horizon and not L.get("standing"):
                continue
            foot = m & (yy > yb - 0.03 * H)
            below = np.maximum(below, blur(foot.astype(F), 0.012 * H) * 2.0)
        ao = np.where(gidmap == bidx, ao * (1 - 0.55 * np.clip(below, 0, 1)), ao)
    ao = np.clip(ao, 0.2, 1)

    # ---- shade each layer and composite with its own alpha (soft edges, translucency survive)
    key_c = np.array([1.0, 0.96, 0.9], F) * 2.4
    if backdrop:
        Bc = backdrop[0]["albedo"]
        top = Bc[: max(horizon, int(0.3 * H))].reshape(-1, 3).mean(0) if horizon > 0 else Bc.reshape(-1, 3).mean(0)
        bot = Bc[horizon:].reshape(-1, 3).mean(0) if horizon < H - 2 else top
        sky_c = 0.5 * top / (luma(top[None])[0] + 1e-3) * 0.6 + 0.4 * np.array([0.8, 0.85, 1.0], F)
        gnd_c = bot / (luma(bot[None])[0] + 1e-3) * 0.5
    else:
        sky_c = np.array([0.8, 0.85, 1.0], F) * 0.8
        gnd_c = np.array([0.5, 0.45, 0.4], F)
    amb_w = (0.5 + 0.5 * N[..., 1])[..., None]
    amb = (gnd_c * (1 - amb_w) + sky_c * amb_w) * 0.55 * ao[..., None]
    V = np.array([0, 0, 1], F)
    Hv = (Lv + V) / np.linalg.norm(Lv + V)
    nh = np.clip((N * Hv).sum(-1), 0, 1)
    if backdrop:                       # the drawn backdrop is not relit: it only receives shadows and occlusion
        B = backdrop[0]
        amb_unocc = amb / np.maximum(ao, 0.2)[..., None]
        B["albedo"] = np.clip(B["C"] / np.maximum(key_c * diff[..., None] + amb_unocc, 1e-3), 0, 4).astype(F)
    # light sources (fire, lamps, glows) light the reliefs around them
    pts = np.zeros((H, W, 3), F)
    n_pts = 0
    for k, L in enumerate(layers):
        el = luma(L["emit"])
        tot = float(el.sum())
        if tot < 2e-4 * W * H:
            continue
        cy, cx = float((yy * el).sum() / tot), float((xx * el).sum() / tot)
        col = L["emit"].reshape(-1, 3).sum(0) / tot
        col = col / (luma(col[None])[0] + 1e-6)
        r0 = 0.22 * max(W, H)
        v = np.stack([cx - xx, -(cy - yy), (base_z.get(k, 0) + 0.05 * H) - Z], -1)
        d = np.linalg.norm(v, axis=-1) + 1e-3
        dp = np.clip((N * v).sum(-1) / d, 0, 1)
        power = min(3.0, tot / (W * H) * 400)
        pts += (dp * power / (1 + (d / r0) ** 2))[..., None] * col * ao[..., None]
        n_pts += 1
    if backdrop:
        pts = np.where((gidmap == bidx)[..., None], pts * 0.35, pts)   # the drawn backdrop shows most of its light
    beauty = np.zeros((H, W, 3), F)
    spec_c = np.zeros((H, W, 3), F)
    gloss_m = np.zeros((H, W), F)
    emit_c = np.zeros((H, W, 3), F)
    for k, L in enumerate(layers):
        if L.get("text"):
            continue
        own = (gidmap == k)
        gl = L["gloss"]
        spec = (gl * nh ** (30 + 150 * gl) * shadow * (diff > 0) * (1 + 3 * gl))[..., None] * key_c * 0.18
        fres = (1 - np.clip(N[..., 2], 0, 1)) ** 4 * gl * 0.12
        spec = spec + fres[..., None] * sky_c * ao[..., None]
        if L["role"] == "backdrop":
            spec = spec * 0
        col = L["albedo"] * (key_c * (diff * shadow)[..., None] + amb + pts) + spec + L["emit"]
        if L.get("frag"):
            col = L["C"] * 0.9 + L["emit"]              # particles keep their painted colour
        a = L["A"][..., None] if L["role"] != "backdrop" else np.ones((H, W, 1), F)
        beauty = beauty * (1 - a) + col * a + L["emit"] * (1 - a)
        alb_c = alb_c * (1 - a) + L["albedo"] * a
        spec_c = np.where(own[..., None], spec, spec_c)
        gloss_m = np.where(own, gl, gloss_m)
        emit_c = emit_c * (1 - a) + L["emit"]
    text_rgba = None
    if texts:
        Pt = np.zeros((H, W, 3), F)
        At = np.zeros((H, W), F)
        for L in texts:
            Pt = Pt * (1 - L["A"][..., None]) + L["P"]
            At = At * (1 - L["A"]) + L["A"]
        text_rgba = (Pt, At)

    # ---- the buffer format of the media (camera = identity: x right, y up, z into the screen)
    pxw = 10.0 * 0.6 / H
    pos = np.stack([(xx - W / 2) * pxw, -(yy - H / 2) * pxw, -Z * pxw], -1).astype(F)
    depth = (10.0 - Z * pxw * 0.5).astype(F)
    normal = np.stack([N[..., 0], N[..., 1], -N[..., 2]], -1).astype(F)
    buf = dict(albedo=alb_c, normal=normal, pos=pos, depth=depth, id=idmap, gid=gidmap,
               gloss=gloss_m, transp=np.where(topA < 0.97, 1 - topA, 0).astype(F), emit=emit_c.astype(F),
               key_diff=diff.astype(F), key_shadow=shadow.astype(F), ao=ao.astype(F), amb=amb.astype(F),
               pts=pts.astype(F), spec=spec_c.astype(F), beauty=beauty.astype(F),
               flow=np.zeros((H, W, 2), F), cam=np.array([[0, 0, -10], [0, 0, 1], [1, 0, 0], [0, 1, 0]], F),
               key_color=key_c.astype(F), light2d=l2.astype(F), names=list(names), fog=np.zeros((H, W), F),
               fog_col=np.zeros(3, F), Z=Z.astype(F), horizon=horizon, floor_shadow=fshadow.astype(F),
               estimated=estimated, recorded=[L["name"] for L in layers if L.get("recorded")],
               text_P=None, text_A=None, drawn=drawn.astype(F), backdrop_id=bidx)
    if text_rgba is not None:
        buf["text_P"], buf["text_A"] = text_rgba
    buf["time"] = time.time() - t0
    if verbose:
        print(f"  buffers: {W}x{H}, horizon {horizon}, estimated {estimated}, baked shadows removed {n_baked}, "
              f"point lights {n_pts}, {time.time() - t0:.1f}s")
    return buf
