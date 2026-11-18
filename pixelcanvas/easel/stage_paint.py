"""Painterly strokes: coarse-to-fine brush passes that follow the forms (Hertzmann 1998).

One call is one pass with one brush size: the reference is blurred in proportion to the brush, a grid of candidate
points is laid over the area, and wherever the canvas still differs from that blurred reference a stroke starts at
the worst point of its cell, runs along the isophotes (the direction in which the reference colour does not change)
and stops where the reference no longer matches the stroke's colour or where going on would not improve the canvas.

Every stroke is a mark a painter could make (DESIGN.md 10.1): one colour, picked with the eyedropper as the median of
the reference under the stroke's footprint; width >= 1 px; length >= 4 x width, or a dab at least 3 px wide.  Small
passes narrow the brush next to reference edges so a stroke never reaches across an edge.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .ops import Fill, Stroke, mask_to_rle
from .trace import as_mask, edge_map, orientation_field, to_lab

__all__ = ["paint_ops", "paint_error", "ground_ops", "texture_ops", "highlight_ops", "thin_line_ops"]


def _blur(img, sigma):
    if sigma < 0.3:
        return img.astype(np.float32)
    return cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)


def paint_error(ref, canvas, size=0.0):
    """Per-pixel Delta E between the reference blurred for brush ``size`` and the canvas."""
    refb = _blur(ref, 0.25 * size if size > 3 else 0.0)
    return np.sqrt(((to_lab(np.clip(refb, 0, 255) / 255.0) - to_lab(canvas)) ** 2).sum(-1))


def _bilinear(img, x, y):
    H, W = img.shape[:2]
    x = np.clip(x - 0.5, 0, W - 1.001)
    y = np.clip(y - 0.5, 0, H - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = (x - x0)[..., None] if img.ndim == 3 else (x - x0)
    fy = (y - y0)[..., None] if img.ndim == 3 else (y - y0)
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def ground_ops(ref, mask=None, *, shift=0.85):
    """A toned ground before the block-in: the area filled with the reference's mean colour, pulled slightly
    toward mid value, so no bare paper shows between the first strokes."""
    ref = np.asarray(ref)
    m = as_mask(mask, ref.shape)
    if not m.any():
        return []
    c = ref[m].reshape(-1, 3).astype(np.float32).mean(0)
    c = c * shift + 128.0 * (1 - shift)
    return [Fill(polygons=None, mask_rle=mask_to_rle(m), color=tuple(int(round(v)) for v in c), feather=1.0)]


def _footprint_px(pts, widths, H, W):
    """Pixel indices (xi, yi) covered by a stroke's footprint: its centre line and lines at +-w/3 and +-w/2 - 0.5."""
    P = np.asarray(pts, np.float64).reshape(-1, 2)
    w = np.asarray(widths, np.float64).reshape(-1)
    if len(P) == 1:
        r = max(1, int(round(float(np.max(w)) / 2)))
        x, y = int(P[0, 0]), int(P[0, 1])
        yy, xx = np.mgrid[max(0, y - r + 1):min(H, y + r), max(0, x - r + 1):min(W, x + r)]
        if not xx.size:
            return np.array([min(x, W - 1)]), np.array([min(y, H - 1)])
        return xx.ravel(), yy.ravel()
    if w.size == 1:
        w = np.full(len(P), float(w[0]))
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    offs = [0.0, -1 / 3, 1 / 3]
    S = np.concatenate([P + n * (k * w[:, None]) for k in offs]
                       + [P + n * (s_ * np.maximum(w / 2 - 0.5, 0))[:, None] for s_ in (-1, 1)], 0)
    xi = np.clip(S[:, 0].astype(np.int64), 0, W - 1)
    yi = np.clip(S[:, 1].astype(np.int64), 0, H - 1)
    return xi, yi


def _footprint_median(ref, pts, widths, return_samples=False):
    """Eyedropper: median colour of the reference under a stroke (centre line and two offset lines)."""
    H, W = ref.shape[:2]
    P = pts.astype(np.float64)
    if len(P) == 1:
        r = max(1, int(round(float(np.max(widths)) / 2)))
        x, y = int(P[0, 0]), int(P[0, 1])
        patch = ref[max(0, y - r + 1):y + r, max(0, x - r + 1):x + r].reshape(-1, 3)
        if not len(patch):
            patch = ref[min(y, H - 1), min(x, W - 1)].reshape(1, 3)
        patch = patch.astype(np.float64)
        return (np.median(patch, 0), patch) if return_samples else np.median(patch, 0)
    t = np.gradient(P, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    w = np.asarray(widths, np.float64).reshape(-1)
    if w.size == 1:
        w = np.full(len(P), float(w[0]))
    samples = [P] + [P + n * (k * w[:, None] / 3.0) for k in (-1, 1)]
    S = np.concatenate(samples, 0)
    xi = np.clip(S[:, 0].astype(np.int64), 0, W - 1)
    yi = np.clip(S[:, 1].astype(np.int64), 0, H - 1)
    px = ref[yi, xi].astype(np.float64)
    return (np.median(px, 0), px) if return_samples else np.median(px, 0)


TIP_COHERENT = 0.5      # tips="auto": structure coherence from which a stroke takes the bristle tip
TIP_TEXTURE = 3.0       # tips="auto": fine texture (Lab L std) from which an incoherent place takes the sponge tip


PAINT_BUDGET = 9.0     # seconds per paint_ops call (generation; the strokes it returns take about as long to paint)
CALL_TARGET = 20.0     # seconds a tips="local" detail call aims at (strokes, lines, then the shape pass)
SHAPE_BUDGET = 6.0     # seconds for the shape pass that closes a tips="local" call


def paint_ops(ref, mask=None, canvas=None, *, size=24.0, brush="round", density=1.0, max_strokes=None, seed=0,
              threshold=None, max_len=None, opacity=1.0, edge_aware=None, structure=False, tips=None,
              budget="auto", stats=None, avoid=None):
    """Strokes of about ``size`` px that move ``canvas`` (uint8 RGB composite) toward ``ref`` inside ``mask``.

    density: grid density multiplier (more, overlapping strokes above 1).  threshold: Delta E below which an area
    counts as done (default from size).  max_strokes: keep the strokes with the largest error.  edge_aware: narrow
    the brush next to reference edges (default: size <= 8).
    structure: strokes follow the reference's structure direction measured over a wider window (structure tensor
    over ~2 x the brush), stiffly, and run as long as the structure is coherent (sand ripples, grass and grain become
    long directed marks; incoherent places get short ones).
    tips: None = ``brush`` for every stroke; "auto" = per stroke, the bristle tip where the structure is coherent,
    the sponge tip where the reference is finely textured without a direction (stone, foliage), else ``brush``;
    textured tips only inside the areas ``stage_local.char_map`` calls textured; "local" = as "auto", and each area
    gets its own treatment (``_local_pass``): a forms-sized pass airbrushes the soft ramps, a detail-sized pass closes
    with the residual shape pass (``shapes.shape_ops``) so the hard-edged content is drawn last, crisp.
    avoid: bool [H,W] where no stroke starts (another treatment owns those pixels).
    budget: seconds for one call ("auto" = PAINT_BUDGET, None = no limit): strokes are made worst error first and
    the call stops making new ones when the time is up (``stats["out_of_time"]``); calling again on the updated
    canvas carries on where the error is still high.
    brush="shapes": the residual shape pass (``shapes.shape_ops``): the reference's small hard-edged shapes that the
    canvas gets wrong, as outlined fills with sub-pixel edges (``size``, ``threshold`` and the stroke options are
    not used)."""
    import time as _time
    t_start = _time.perf_counter()
    stats = {} if stats is None else stats
    stats["out_of_time"] = False
    if budget == "auto":
        budget = PAINT_BUDGET
    if brush == "shapes":
        from .shapes import shape_ops
        kw = {} if budget is None else {"budget": float(budget)}
        ops = shape_ops(ref, mask, canvas, seed=seed, stats=stats, **kw)
        if max_strokes:
            ops = ops[:int(max_strokes)]
        return ops
    if tips == "local" and float(size) > 3 and brush not in ("soft",):
        return _local_pass(ref, mask, canvas, size=float(size), brush=brush, density=density, max_strokes=max_strokes,
                           seed=seed, threshold=threshold, max_len=max_len, opacity=opacity, edge_aware=edge_aware,
                           structure=structure, budget=budget, stats=stats)
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    # on a region: strokes start inside it and run on as they would in a whole-picture pass (each stroke whole,
    # across the region's pixel outline), so region calls over a partition paint like one call on the whole
    own_edge = not bool(m.all())
    m_start = m
    if canvas is None:
        canvas = np.full_like(ref, 255)
    canvas = np.asarray(canvas)[..., :3]
    size = float(max(size, 1.0))
    if brush == "pixel":
        raise ValueError("the pixel brush is not a stage-tool brush (DESIGN.md 10.1)")
    rng = np.random.default_rng(seed)
    if edge_aware is None:
        edge_aware = size <= 8
    if threshold is None:
        threshold = float(np.clip(1.5 + 0.35 * size, 2.5, 14.0))
    sigma = 0.25 * size if size > 3 else 0.0
    refb = _blur(ref, sigma)
    lab_refb = to_lab(np.clip(refb, 0, 255) / 255.0)
    lab_can = to_lab(canvas)
    lab_ref0 = to_lab(ref) if sigma > 0 else lab_refb
    if max_len is None:
        max_len = 16 if size >= 4 else 24
    if structure:
        max_len = int(max_len * 2)
    if own_edge:
        reach = int(np.ceil(max_len * max(1.0, 0.5 * size) + size))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
        m = cv2.dilate(m.astype(np.uint8), k) > 0
    D = np.sqrt(((lab_refb - lab_can) ** 2).sum(-1))
    D[~m] = 0
    # candidate grid: one start per cell, at the cell's worst pixel, where the cell's mean error is above threshold
    g = max(1, int(round(size * 0.55 / max(density, 0.1) ** 0.5)))
    ny, nx = -(-H // g), -(-W // g)
    Dp = np.zeros((ny * g, nx * g), np.float32)
    Dp[:H, :W] = D
    cells = Dp.reshape(ny, g, nx, g).transpose(0, 2, 1, 3).reshape(ny, nx, g * g)
    cmean = cells.mean(-1)
    carg = cells.argmax(-1)
    thr_cell = threshold
    if tips in ("local", "local-strokes"):
        # textured areas are worked to a finer tolerance: their low-contrast structure (stone, bark, foliage in the
        # dark) is what makes them read as textured; the clean areas keep the stage's threshold
        from .stage_local import TEXTURE, char_map
        tp = np.zeros((ny * g, nx * g), np.float32)
        tp[:H, :W] = char_map(ref).cls == TEXTURE
        tshare = tp.reshape(ny, g, nx, g).mean((1, 3))
        thr_cell = np.where(tshare >= 0.5, min(float(threshold), LOCAL_TEX_THRESHOLD), float(threshold))
    cy, cx = np.nonzero(cmean > thr_cell)
    if len(cy) == 0:
        return []
    a = carg[cy, cx]
    sx = cx * g + (a % g) + 0.5 + rng.uniform(-0.25, 0.25, len(a)) * (g > 1)
    sy = cy * g + (a // g) + 0.5 + rng.uniform(-0.25, 0.25, len(a)) * (g > 1)
    keep = (sx < W) & (sy < H)
    keep[keep] &= m_start[np.clip(sy[keep].astype(np.int64), 0, H - 1), np.clip(sx[keep].astype(np.int64), 0, W - 1)]
    if avoid is not None:
        keep[keep] &= ~np.asarray(avoid, bool)[np.clip(sy[keep].astype(np.int64), 0, H - 1),
                                               np.clip(sx[keep].astype(np.int64), 0, W - 1)]
    sx, sy, score = sx[keep], sy[keep], cmean[cy, cx][keep]
    if len(sx) == 0:
        return []
    if structure:
        tx, ty, coh = orientation_field(ref, sigma=max(2.5, 2.0 * size))
    else:
        tx, ty, coh = orientation_field(np.clip(refb, 0, 255).astype(np.uint8), sigma=max(1.0, 0.5 * size))
    if tips in ("auto", "local", "local-strokes"):
        Lr = lab_ref0[..., 0]
        hp = Lr - cv2.GaussianBlur(Lr, (0, 0), 1.0)
        tex = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 3.0))
        # textured tips only where the local character map calls the reference textured: a clean flat, a thin
        # line or a soft ramp gets the plain brush (no bristle streaks or sponge speckle where the original has none)
        from .stage_local import TEXTURE, char_map
        textured = char_map(ref).cls == TEXTURE
    step = max(1.0, 0.5 * size)
    min_len = 2
    col_tol = max(threshold * 2.0, 8.0)
    if edge_aware:
        edist = ndi.distance_transform_edt(~edge_map(ref, sigma=0.7))
    else:
        edist = None

    def trace_one(bx, by, sign0, max_len):
        K = len(bx)
        P = np.zeros((max_len + 1, K, 2), np.float64)
        P[0, :, 0], P[0, :, 1] = bx, by
        alive = np.ones(K, bool)
        n_pts = np.ones(K, np.int64)
        c0 = _bilinear(lab_refb, bx, by)
        prev = np.zeros((K, 2))
        fc = 0.4 if structure else 0.75
        for s in range(1, max_len + 1):
            idx = np.flatnonzero(alive)
            if len(idx) == 0:
                break
            cur = P[s - 1, idx]
            xi = np.clip(cur[:, 0].astype(np.int64), 0, W - 1)
            yi = np.clip(cur[:, 1].astype(np.int64), 0, H - 1)
            d = np.stack([tx[yi, xi], ty[yi, xi]], 1).astype(np.float64)
            if s == 1:
                d *= sign0[idx, None]
            else:
                pv = prev[idx]
                flip = (d * pv).sum(1) < 0
                d[flip] *= -1
                d = fc * d + (1 - fc) * pv
                d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            nxt = cur + step * d
            inside = (nxt[:, 0] >= 0) & (nxt[:, 0] < W) & (nxt[:, 1] >= 0) & (nxt[:, 1] < H)
            nxi = np.clip(nxt[:, 0].astype(np.int64), 0, W - 1)
            nyi = np.clip(nxt[:, 1].astype(np.int64), 0, H - 1)
            inside &= m[nyi, nxi]
            r = _bilinear(lab_refb, nxt[:, 0], nxt[:, 1])
            cv = lab_can[nyi, nxi]
            d_stroke = np.sqrt(((r - c0[idx]) ** 2).sum(1))
            d_canvas = np.sqrt(((r - cv) ** 2).sum(1))
            ok = inside & (d_stroke < col_tol)
            if s > min_len:
                # Hertzmann: stop where the stroke colour is worse than what the canvas already has
                ok &= ~(d_stroke > d_canvas)
            P[s] = P[s - 1]
            go = idx[ok]
            P[s, go] = nxt[ok]
            prev[go] = d[ok]
            n_pts[go] += 1
            alive[idx[~ok]] = False
        return P, n_pts

    def trace(bx, by):
        """Trace each stroke both ways from its start (the start is the worst pixel, often mid-feature)."""
        K = len(bx)
        sign0 = np.where(rng.random(K) < 0.5, -1.0, 1.0)
        back_len = max_len // 3
        Pf, nf = trace_one(bx, by, sign0, max_len - back_len)
        Pb, nb = trace_one(bx, by, -sign0, back_len)
        n = nf + nb - 1
        P = np.zeros((max_len + 1, K, 2), np.float64)
        # backward part reversed, then the forward part (sharing the start point)
        for k in range(K):
            b = nb[k]
            P[:b, k] = Pb[:b, k][::-1]
            P[b:b + nf[k] - 1, k] = Pf[1:nf[k], k]
        return P, n

    # greedy, strongest error first and in batches: a start already covered by an accepted stroke of this pass is
    # skipped before it is traced, and an accepted stroke is cut where it runs on over covered paint
    order = np.argsort(-score, kind="stable")
    sx, sy, score = sx[order], sy[order], score[order]
    cover = np.zeros((H, W), np.uint8)
    cth = max(1, int(round(size * 0.6)))
    strokes = []
    B = 3000
    pos = 0
    limit = int(max_strokes) if max_strokes is not None else None
    while pos < len(sx):
        if limit is not None and len(strokes) >= limit:
            break
        if budget is not None and _time.perf_counter() - t_start > 0.6 * float(budget):
            stats["out_of_time"] = True
            break
        bx, by = sx[pos:pos + B], sy[pos:pos + B]
        base = pos
        pos += B
        xi = np.clip(bx.astype(np.int64), 0, W - 1)
        yi = np.clip(by.astype(np.int64), 0, H - 1)
        free = np.flatnonzero(cover[yi, xi] == 0)
        if len(free) == 0:
            continue
        P, n_pts = trace(bx[free], by[free])
        PI = np.rint(P - 0.5).astype(np.int32)
        PX = np.clip(PI[..., 0], 0, W - 1)
        PY = np.clip(PI[..., 1], 0, H - 1)
        cov = cover[PY, PX].astype(np.int32)
        valid = np.arange(P.shape[0])[:, None] < n_pts[None, :]
        cov = cov * valid
        run = ndi.convolve1d(cov, np.ones(3, np.int32), axis=0, mode="constant", origin=-1)
        hit = (run >= 3) & valid
        hit[0] = False
        first = np.where(hit.any(0), hit.argmax(0), n_pts)
        n_eff = np.maximum(1, np.minimum(n_pts, first))
        sx0 = PX[0].tolist()
        sy0 = PY[0].tolist()
        ne = n_eff.tolist()
        for j in range(len(free)):
            if cover[sy0[j], sx0[j]]:
                continue
            n = ne[j]
            if structure:
                # as long as the structure is coherent: a ripple or a blade is a long mark, a pebble a short one
                cj = float(coh[sy0[j], sx0[j]])
                n = max(1, min(n, int(round(max_len * float(np.clip(cj * 1.2, 0.2, 1.0)))) + 1))
            pts = P[:n, j, :]
            # widths: the brush, narrowed beside reference edges
            if edist is not None:
                dd = edist[PY[:n, j], PX[:n, j]]
                wid = np.clip(2.0 * dd - 0.5, 1.0, size)
            else:
                wid = np.full(n, size)
            L = float(np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)).sum()) if n > 1 else 0.0
            wm = float(wid.mean())
            if L < 4.0 * wm + 0.05:
                if size >= 3.0:
                    wid = np.full(n, size)        # a dab with the full brush
                    wm = size
                else:
                    # too short for a line at this size: a 3 px dab, if it brings its footprint closer
                    cxp, cyp = float(pts[:, 0].mean()), float(pts[:, 1].mean())
                    xa, ya = int(cxp), int(cyp)
                    fy0, fy1, fx0, fx1 = max(ya - 1, 0), min(ya + 2, H), max(xa - 1, 0), min(xa + 2, W)
                    fr = lab_ref0[fy0:fy1, fx0:fx1].reshape(-1, 3)
                    fc = lab_can[fy0:fy1, fx0:fx1].reshape(-1, 3)
                    dc = np.median(fr, 0)
                    spread = np.sqrt(((fr - dc) ** 2).sum(1)).mean()
                    # a dab is for a small feature (a dot, a mark, a highlight) on a clean area, not for texture
                    cspread = np.sqrt(((fc - np.median(fc, 0)) ** 2).sum(1)).mean()
                    before_e = np.sqrt(((fr - fc) ** 2).sum(1)).mean()
                    after_e = np.sqrt(((fr - dc) ** 2).sum(1)).mean()
                    if after_e > 0.75 * before_e or (spread > 10.0 and cspread > 4.0):
                        continue
                    pts = np.array([[xa + 0.5, ya + 0.5]])
                    n = 1
                    wid = np.array([3.0])
            if n == 1:
                cv2.circle(cover, (sx0[j], sy0[j]), cth // 2, 1, -1)
            else:
                cv2.polylines(cover, [np.ascontiguousarray(PI[:n, j, :]).reshape(-1, 1, 2)], False, 1, cth)
            strokes.append((pts.astype(np.float32), wid.astype(np.float32), base + int(free[j])))
            if limit is not None and len(strokes) >= limit:
                break
    if not strokes:
        return []
    perm = rng.permutation(len(strokes))
    ref_c = _blur(ref, 0.15 * size if size > 4 else 0.0)
    protect = None
    if tips in ("local", "local-strokes"):
        from .stage_local import LINE, char_map
        cmp_ = char_map(ref)
        protect = cv2.dilate((cmp_.edge | (cmp_.cls == LINE)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ops = []
    for k in perm:
        pts, wid, i = strokes[k]
        med, samples = _footprint_median(ref_c, pts, wid, return_samples=True)
        col = tuple(int(v) for v in np.clip(np.rint(med), 0, 255))
        # a painter only keeps a stroke that helps: over the stroke's footprint its one colour must be closer to
        # the reference than the canvas already is (a stroke across an edge that would smear it is not made)
        xi, yi = _footprint_px(pts, wid, H, W)
        lr = lab_ref0[yi, xi]
        before_e = float(np.sqrt(((lr - lab_can[yi, xi]) ** 2).sum(1)).mean())
        lc = to_lab(np.asarray(col, np.uint8)[None, None])[0, 0]
        after_e = float(np.sqrt(((lr - lc) ** 2).sum(1)).mean())
        if after_e >= 0.9 * before_e:
            continue
        if protect is not None:
            # crisp work is protected: a stroke that makes the reference's edges and thin lines under it worse
            # (paint pulled across an outline) is not made, however much it helps the area beside them
            pe = protect[yi, xi]
            if pe.sum() >= 2:
                lp = lr[pe]
                if float(np.sqrt(((lp - lc) ** 2).sum(1)).mean()) >                         float(np.sqrt(((lp - lab_can[yi, xi][pe]) ** 2).sum(1)).mean()) + 1.0:
                    continue
        w = float(wid[0]) if (len(pts) == 1 or np.allclose(wid, wid[0])) else wid
        br = brush
        if tips in ("auto", "local", "local-strokes") and len(pts) > 1:
            xa, ya = int(np.clip(pts[0, 0], 0, W - 1)), int(np.clip(pts[0, 1], 0, H - 1))
            if not textured[ya, xa]:
                pass
            elif coh[ya, xa] >= TIP_COHERENT and float(np.mean(wid)) >= 2.0:
                br = "bristle"
            elif tex[ya, xa] >= TIP_TEXTURE:
                br = "sponge"
        ops.append(Stroke(pts, w, col, opacity=opacity, brush=br, seed=seed * 7919 + int(i),
                          own=own_edge))
    return ops


LOCAL_TEX_THRESHOLD = 3.0   # tips="local": Delta E per cell above which a textured area still gets strokes
LOCAL_SMALL = 600.0    # tips="local": on a picture whose long side is below this, brush sizes shrink in proportion
LOCAL_SHAPES_MAX = 5.0  # tips="local": a pass up to this size closes with the shape pass; above it, ramps get airbrush
RAMP_EDGE_PX = 3.0     # tips="local" detail: in a soft ramp strokes start only this close to an edge
AIR_MIN_AREA = 400     # px: smallest soft-ramp area the local forms pass airbrushes


def _paint_onto(canvas, ops):
    from .brush import apply_ops as _apply, new_layer as _new_layer
    base = np.asarray(canvas)[..., :3]
    if not ops:
        return base
    lay = _new_layer(*base.shape[:2])
    lay[..., :3] = base.astype(np.float32) / 255.0
    lay[..., 3] = 1.0
    _apply(lay, ops)
    return np.clip(np.rint(lay[..., :3] * 255), 0, 255).astype(np.uint8)


def _local_pass(ref, mask, canvas, *, size, brush, density, max_strokes, seed, threshold, max_len, opacity,
                edge_aware, structure, budget, stats):
    """tips="local": one pass in which each area of the reference gets its own treatment (stage_local.char_map).

    A forms-sized pass (above LOCAL_SHAPES_MAX): the soft ramps are airbrushed inside their own friskets (soft dabs
    coarse to fine, clear of the hard steps) and the strokes start only outside them.  A detail-sized pass: the
    strokes (textured tips only in textured areas), then the residual shape pass over the canvas with the strokes on
    it, so hard-edged flats, lettering, architecture and thin lines are drawn last, crisp.  On a small picture the
    brush shrinks in proportion (a 12 px stroke on a 200 px picture is a smear)."""
    import time as _time
    from .stage_local import RAMP, char_map
    t0 = _time.perf_counter()
    ref = np.asarray(ref)[..., :3]
    H, W = ref.shape[:2]
    canvas = np.full_like(ref, 255) if canvas is None else np.asarray(canvas)[..., :3]
    nominal = float(size)
    L = float(max(H, W))
    eff = nominal * min(1.0, L / LOCAL_SMALL)
    eff = max(eff, 1.0)
    br = brush if eff > 3 else "liner"
    cm = char_map(ref)
    m = as_mask(mask, ref.shape)
    kw = dict(size=eff, brush=br, density=density, max_strokes=max_strokes, seed=seed, threshold=threshold,
              max_len=max_len, opacity=opacity, edge_aware=edge_aware, structure=structure, tips="local-strokes",
              budget=budget, stats=stats)
    if nominal > LOCAL_SHAPES_MAX:
        from scipy import ndimage as _ndi
        from .airbrush import airbrush_ops, step_edges
        ramp = (cm.cls == RAMP) & m
        air = []
        if ramp.sum() >= AIR_MIN_AREA:
            lab_, n = _ndi.label(ramp)
            if n:
                cnt = np.bincount(lab_.ravel(), minlength=n + 1)
                small = cnt < AIR_MIN_AREA
                small[0] = True
                lab_[small[lab_]] = 0
            sizes = tuple(s_ for s_ in (64, 32, 16, 8) if s_ <= max(8.0, 5.0 * eff) and s_ >= min(8.0, max(3.0, 0.6 * eff)))
            if lab_.max() > 0 and sizes:
                ast = {}
                air = airbrush_ops(ref, canvas, lab_.astype(np.int32), sizes=sizes, seed=seed, stats=ast,
                                   edges=step_edges(ref, sigma=2.0, thr=4.0))
                stats["air"] = ast
        can2 = _paint_onto(canvas, air)
        ops = paint_ops(ref, mask, can2, avoid=(cm.cls == RAMP), **kw)
        return list(air) + list(ops)
    # the soft ramps were airbrushed: strokes there only close to an edge (where the dabs kept clear), so the
    # stroke-by-stroke colour steps do not mottle a smooth passage
    far = ndi.distance_transform_edt(~cm.edge) > RAMP_EDGE_PX
    ops = paint_ops(ref, mask, canvas, avoid=(cm.cls == RAMP) & far, **kw)
    if stats.get("out_of_time") or (max_strokes and len(ops) >= int(max_strokes)):
        return ops       # the caller calls again on the updated canvas; the shapes come with the last call
    from .shapes import shape_ops
    # crisp work last: the thin dark lines still missing as pen strokes at their width, then the shape pass
    can2 = _paint_onto(canvas, ops)
    lops = thin_line_ops(ref, mask, can2, seed=seed + 7)
    if lops:
        ops = list(ops) + lops
        can2 = _paint_onto(can2, lops)
    stats["lines"] = len(lops)
    sst = {}
    # (the shape pass gets what is left of a ~20 s call, at least SHAPE_BUDGET)
    left = CALL_TARGET - (_time.perf_counter() - t0)
    sops = shape_ops(ref, mask, can2, seed=seed, stats=sst,
                     **({} if budget is None else {"budget": max(SHAPE_BUDGET, left)}))
    stats["shapes"] = {k: sst.get(k) for k in ("pieces", "tried", "fills", "small", "out_of_time")}
    return ops + list(sops)


def texture_ops(ref, mask=None, canvas=None, *, width=1.3, max_len=28, min_coh=0.3, contrast=3.0, brush="pen",
                seed=0, max_strokes=None, local=True):
    """Fine texture as directed marks: thin pen strokes laid along the light and dark streaks of the reference
    (sand ripples, grass, grain, bark) where the canvas is smoother than the reference.

    Each streak is found as a ridge of the reference's fine detail (lightness minus its 2 px blur), followed along
    the structure direction (structure tensor over 3 px) for as long as the streak lasts and the structure stays
    coherent, and drawn with one colour: the median of the reference along the stroke (eyedropper over its own
    footprint).  A stroke is kept only when that colour brings its footprint closer to the reference than the canvas
    already is.  On a region: the whole picture's streaks anchored in the region, each one whole.
    local: only where ``stage_local.char_map`` calls the reference textured."""
    from .ops import is_partial, owned_ops
    if is_partial(mask):
        return owned_ops(texture_ops(ref, None, canvas, width=width, max_len=max_len, min_coh=min_coh,
                                     contrast=contrast, brush=brush, seed=seed, max_strokes=max_strokes,
                                     local=local), mask)
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    if canvas is None:
        canvas = np.full_like(ref, 255)
    canvas = np.asarray(canvas)[..., :3]
    rng = np.random.default_rng(seed)
    lab_r = to_lab(ref)
    lab_c = to_lab(canvas)
    Lr, Lc = lab_r[..., 0], lab_c[..., 0]
    tx, ty, coh = orientation_field(ref, sigma=3.0)
    # the fine detail, smoothed along the streaks only (a short line integral along the structure direction)
    hp0 = Lr - cv2.GaussianBlur(Lr, (0, 0), 2.0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hp = hp0.copy()
    for k in (-2, -1, 1, 2):
        hp += cv2.remap(hp0, xx + k * tx, yy + k * ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    hp /= 5.0
    # ridge centres: |hp| at a maximum across the streak (along the normal)
    nx, ny = -ty, tx
    a = np.abs(hp)
    a1 = cv2.remap(a, xx + nx, yy + ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a2 = cv2.remap(a, xx - nx, yy - ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    cand = m & (a >= contrast) & (a >= a1) & (a >= a2) & (coh >= min_coh)
    if local:
        # only where the local character map calls the reference textured (streaks laid on a clean flat or a soft
        # ramp would be noise the original does not have)
        from .stage_local import TEXTURE, char_map
        texm = char_map(ref).cls == TEXTURE
        cand &= texm
        m = m & (cv2.dilate(texm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    cy, cx = np.nonzero(cand)
    if len(cy) == 0:
        return []
    order = np.argsort(-a[cy, cx], kind="stable")
    cy, cx = cy[order], cx[order]
    sgn = np.sign(hp)
    half = max_len // 2
    cover = np.zeros((H, W), np.uint8)
    ops = []
    B = 4000
    for b0 in range(0, len(cy), B):
        by, bx = cy[b0:b0 + B], cx[b0:b0 + B]
        free = cover[by, bx] == 0
        by, bx = by[free], bx[free]
        if not len(by):
            continue
        K = len(by)
        s0 = sgn[by, bx]
        a0 = a[by, bx]
        paths = []
        for direction in (1.0, -1.0):
            P = np.zeros((half + 1, K, 2), np.float32)
            P[0, :, 0], P[0, :, 1] = bx + 0.5, by + 0.5
            n_ok = np.ones(K, np.int64)
            alive = np.ones(K, bool)
            prev = np.stack([tx[by, bx], ty[by, bx]], 1) * direction
            for s in range(1, half + 1):
                cur = P[s - 1]
                xi = np.clip(cur[:, 0].astype(np.int64), 0, W - 1)
                yi = np.clip(cur[:, 1].astype(np.int64), 0, H - 1)
                d = np.stack([tx[yi, xi], ty[yi, xi]], 1)
                flip = (d * prev).sum(1) < 0
                d[flip] *= -1
                nxt = cur + d
                xn = np.clip(nxt[:, 0].astype(np.int64), 0, W - 1)
                yn = np.clip(nxt[:, 1].astype(np.int64), 0, H - 1)
                # the streak goes on while it keeps its sign and half its strength, and the structure stays coherent
                ok = alive & (sgn[yn, xn] == s0) & (a[yn, xn] >= 0.3 * a0) & (coh[yn, xn] >= min_coh) & m[yn, xn]
                ok &= (nxt[:, 0] >= 0) & (nxt[:, 0] < W) & (nxt[:, 1] >= 0) & (nxt[:, 1] < H)
                P[s] = np.where(ok[:, None], nxt, cur)
                prev = np.where(ok[:, None], d, prev)
                n_ok += ok
                alive &= ok
            paths.append((P, n_ok))
        (Pf, nf), (Pb, nb) = paths
        for k in range(K):
            if cover[by[k], bx[k]]:
                continue
            pts = np.vstack([Pb[:nb[k], k][::-1], Pf[1:nf[k], k]])
            if len(pts) < 2:
                continue
            L = float(np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)).sum())
            if L < 4.0 * width:
                continue
            xi = np.clip(pts[:, 0].astype(np.int64), 0, W - 1)
            yi = np.clip(pts[:, 1].astype(np.int64), 0, H - 1)
            col = np.median(ref[yi, xi].astype(np.float64), 0)
            lc = to_lab(np.clip(np.rint(col), 0, 255).astype(np.uint8)[None, None])[0, 0]
            before = float(np.sqrt(((lab_r[yi, xi] - lab_c[yi, xi]) ** 2).sum(1)).mean())
            after = float(np.sqrt(((lab_r[yi, xi] - lc) ** 2).sum(1)).mean())
            if after >= 0.8 * before:
                continue
            cv2.polylines(cover, [np.rint(pts - 0.5).astype(np.int32).reshape(-1, 1, 2)], False, 1, 2)
            q = pts[::2] if len(pts) > 6 else pts
            if not np.allclose(q[-1], pts[-1]):
                q = np.vstack([q, pts[-1:]])
            ops.append(Stroke(q.astype(np.float32), float(width), tuple(int(v) for v in np.clip(np.rint(col), 0, 255)),
                              brush=brush, seed=seed * 7919 + len(ops)))
            if max_strokes is not None and len(ops) >= max_strokes:
                return ops
    perm = rng.permutation(len(ops))
    return [ops[i] for i in perm]


def thin_line_ops(ref, mask=None, canvas=None, *, detail=0.5, min_gain=0.8, seed=0):
    """Thin dark lines the canvas is missing (a cable, a window bar, a crease, a stem), drawn as pen strokes at their
    measured width in their own colour (the median along the line), where the canvas is lighter than the reference
    there; only lines the local character map calls lines (thin ridges on a clean or smooth ground; a texture's
    streaks are the strokes').  A stroke is kept only when it brings its footprint closer to the reference by the
    factor ``min_gain``.  On a region: the whole picture's lines anchored in the region, each one whole."""
    from .lines import extract_lines
    from .ops import is_partial, owned_ops
    from .stage_local import TEXTURE, char_map
    if is_partial(mask):
        return owned_ops(thin_line_ops(ref, None, canvas, detail=detail, min_gain=min_gain, seed=seed), mask)
    ref = np.asarray(ref)[..., :3]
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    canvas = np.full_like(ref, 255) if canvas is None else np.asarray(canvas)[..., :3]
    lab_r, lab_c = to_lab(ref), to_lab(canvas)
    miss = cv2.dilate(((lab_c[..., 0] - lab_r[..., 0]) > 6.0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    clean = char_map(ref).cls != TEXTURE
    pieces, _ = extract_lines(ref, m & miss & clean, detail=detail)
    ops = []
    for i, pc in enumerate(pieces):
        xi = np.clip(pc.pts[:, 0].astype(np.int64), 0, W - 1)
        yi = np.clip(pc.pts[:, 1].astype(np.int64), 0, H - 1)
        col = np.median(ref[yi, xi].astype(np.float64), 0)
        lc = to_lab(np.clip(np.rint(col), 0, 255).astype(np.uint8)[None, None])[0, 0]
        before = float(np.sqrt(((lab_r[yi, xi] - lab_c[yi, xi]) ** 2).sum(1)).mean())
        after = float(np.sqrt(((lab_r[yi, xi] - lc) ** 2).sum(1)).mean())
        if after >= min_gain * before:
            continue
        w = np.maximum(pc.width.astype(np.float32), 1.0)
        L = float(np.sqrt((np.diff(pc.pts, axis=0) ** 2).sum(1)).sum())
        if L < 4.0 * float(w.mean()):
            w = np.maximum(w * (L / (4.0 * float(w.mean()))) * 0.98, 1.0)
            if L < 4.0 * float(w.mean()):
                continue
        ops.append(Stroke(pc.pts, w, tuple(int(v) for v in np.clip(np.rint(col), 0, 255)), brush="liner",
                          seed=seed * 7919 + i))
    return ops


def highlight_ops(ref, mask=None, canvas=None, *, detail=0.5, min_gain=0.8, seed=0):
    """Thin light marks: the reference's light ridges (a strap's edge light, a buckle's glint, a lash highlight)
    traced like line art on the inverted picture and drawn as pen strokes in their own light colour (the median
    along the ridge), where the canvas is darker than the reference there.  A stroke is kept only when its colour
    brings its footprint closer to the reference than the canvas is (by the factor ``min_gain``).
    On a region: the whole picture's strokes anchored in the region, each one whole."""
    from .lines import extract_lines
    from .ops import is_partial, owned_ops
    if is_partial(mask):
        return owned_ops(highlight_ops(ref, None, canvas, detail=detail, min_gain=min_gain, seed=seed), mask)
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    canvas = np.full_like(ref, 0) if canvas is None else np.asarray(canvas)[..., :3]
    lab_r, lab_c = to_lab(ref), to_lab(canvas)
    # only where the canvas is darker than the reference (a missing light)
    miss = cv2.dilate(((lab_r[..., 0] - lab_c[..., 0]) > 6.0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pieces, _ = extract_lines(255 - ref, m & miss, detail=detail)
    ops = []
    for i, pc in enumerate(pieces):
        xi = np.clip(pc.pts[:, 0].astype(np.int64), 0, W - 1)
        yi = np.clip(pc.pts[:, 1].astype(np.int64), 0, H - 1)
        col = np.median(ref[yi, xi].astype(np.float64), 0)
        lc = to_lab(np.clip(np.rint(col), 0, 255).astype(np.uint8)[None, None])[0, 0]
        before = float(np.sqrt(((lab_r[yi, xi] - lab_c[yi, xi]) ** 2).sum(1)).mean())
        after = float(np.sqrt(((lab_r[yi, xi] - lc) ** 2).sum(1)).mean())
        if after >= min_gain * before:
            continue
        w = np.maximum(pc.width.astype(np.float32), 1.0)
        L = float(np.sqrt((np.diff(pc.pts, axis=0) ** 2).sum(1)).sum())
        if L < 4.0 * float(w.mean()):
            w = np.maximum(w * (L / (4.0 * float(w.mean()))) * 0.98, 1.0)
            if L < 4.0 * float(w.mean()):
                continue
        ops.append(Stroke(pc.pts, w, tuple(int(v) for v in np.clip(np.rint(col), 0, 255)), brush="liner",
                          seed=seed * 7919 + i))
    return ops
