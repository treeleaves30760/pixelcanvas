"""Residual shape pass: the small hard-edged detail of a picture as outlined fills, fitted where the canvas is wrong.

A copyist with a loupe paints a buckle, an eye, a lettering stroke as small flat shapes with crisp edges, each in
one colour, laid where the picture still differs from the model.  This pass does the same, greedily, large to small:

1. mixed pixels: a pixel of the reference whose colour lies between the colours found on either side of it (along
   the local gradient, 1-3 px away) is an antialiased edge pixel; its position between the two colours (``t``) says
   how much of it belongs to each side.  Every other pixel is pure.
2. pieces: pure pixels chained to their neighbours while the colour changes by less than ``tol`` Delta E (a flat or
   a smooth ramp; thin dark lines also chain diagonally);
3. each piece's coverage field: 1 on its pure pixels, ``t`` (its share) on the mixed pixels beside it; upsampled
   smoothly (cubic, ``ss`` x) and cut at one half by marching squares, which places the outline to a fraction of a
   pixel (locating an edge, DESIGN.md 10.1); the rings are smoothed along the outline with corners kept;
4. one colour per piece (the median of the reference over its pure pixels) or a two-colour gradient on a ramp;
5. a piece is kept only when it lowers the Delta E between canvas and reference under its footprint; pieces under
   12 px^2 become a short stroke or a 3 px dab under the same test.

Rendered antialiased (fills supersample 4x), a correct sub-pixel outline reproduces the reference's edge pixels
without copying them.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy import ndimage as ndi

from .ops import Fill, Stroke, mark_area
from .trace import to_lab

__all__ = ["shape_ops", "mixed_pixels", "pieces"]

# a thin mark's stretches by local peak coverage: (from, to, the level it is drawn at)
THIN_LEVELS = ((0.78, 9.0, 1.0), (0.55, 0.78, 0.66), (0.35, 0.55, 0.45), (0.12, 0.35, 0.25))


# ------------------------------------------------------------------------------------------------ mixed pixels
def _gradient_dir(lab, sigma=1.0):
    """Unit direction of the strongest colour change (structure tensor over the Lab channels)."""
    jxx = jxy = jyy = 0.0
    for c in range(3):
        gx = cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        jxx = jxx + gx * gx
        jxy = jxy + gx * gy
        jyy = jyy + gy * gy
    jxx = cv2.GaussianBlur(jxx, (0, 0), sigma)
    jxy = cv2.GaussianBlur(jxy, (0, 0), sigma)
    jyy = cv2.GaussianBlur(jyy, (0, 0), sigma)
    th = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)


def _unmix(x, a, b, min_step, rel_res):
    """t (share of ``a``), residual and validity of x = t a + (1 - t) b (RGB 0..255, [..., 3])."""
    ab = a - b
    den = (ab * ab).sum(-1)
    t = np.clip(((x - b) * ab).sum(-1) / np.maximum(den, 1e-6), 0.0, 1.0)
    res = np.sqrt(((x - b - t[..., None] * ab) ** 2).sum(-1))
    step = np.sqrt(den)
    ok = (step >= min_step) & (res <= np.maximum(6.0, rel_res * step))
    return t, res, ok


def mixed_pixels(ref, *, min_step=16.0, rel_res=0.14, t_lo=0.06, sigma=1.0, lab=None):
    """Classify the reference's pixels as pure or as antialiased mixes of the colours on either side.

    Returns dict: ``pure`` bool [H,W]; for mixed pixels ``ia`` / ``ib`` (flat indices of the pure pixels on the two
    sides, -1 where none), ``t`` (share of the ``ia`` side's colour, 0..1)."""
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    rgb = ref.astype(np.float32)
    if lab is None:
        lab = to_lab(ref)
    nx, ny = _gradient_dir(lab, sigma)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    flat = rgb.reshape(-1, 3)

    def side(sign, d):
        qx = np.clip(np.rint(xx + sign * d * nx), 0, W - 1).astype(np.int32)
        qy = np.clip(np.rint(yy + sign * d * ny), 0, H - 1).astype(np.int32)
        return qy * W + qx

    # 0. thin marks: the same ground 2 (or 3) px either side of the pixel and a clearly different colour on it: the
    # core of a line narrower than ~2.5-4.5 px (its colour is the line's ink, partly mixed with the ground).  Such a
    # pixel is never an edge mix, even where it lies between the line's darker core and the ground.
    thin = np.zeros((H, W), bool)
    ground = rgb.copy()
    for d in (3, 2):
        ga = flat[side(1, d)]
        gb = flat[side(-1, d)]
        g = 0.5 * (ga + gb)
        off = np.sqrt(((rgb - g) ** 2).sum(-1))
        gd = np.sqrt(((ga - gb) ** 2).sum(-1))
        ok = (off >= min_step) & (gd < np.maximum(14.0, 0.25 * off))
        thin |= ok
        ground = np.where(ok[..., None], g, ground)
    # 1. is the pixel a mix of what lies 1-2 px either side of it?
    best_res = np.full((H, W), np.inf, np.float32)
    mixed = np.zeros((H, W), bool)
    for da, db in ((1, 1), (1, 2), (2, 1), (2, 2)):
        a = flat[side(1, da)]
        b = flat[side(-1, db)]
        t, res, ok = _unmix(rgb, a, b, min_step, rel_res)
        ok &= (t > t_lo) & (t < 1.0 - t_lo)
        upd = ok & (res < best_res)
        best_res = np.where(upd, res, best_res)
        mixed |= ok
    mixed &= ~thin
    pure = ~mixed
    # 2. the mixed pixel's two sides: the nearest pure pixel each way (1-3 px), and its share between them
    pflat = pure.reshape(-1)
    ia = np.full(H * W, -1, np.int64)
    ib = np.full(H * W, -1, np.int64)
    for arr, sign in ((ia, 1), (ib, -1)):
        for d in (3, 2, 1):
            q = side(sign, d).reshape(-1)
            okq = pflat[q]
            arr[okq] = q[okq]
    ia = ia.reshape(H, W)
    ib = ib.reshape(H, W)
    has = mixed & (ia >= 0) & (ib >= 0)
    t = np.zeros((H, W), np.float32)
    if has.any():
        x = rgb[has]
        a = flat[ia[has]]
        b = flat[ib[has]]
        tt, res, ok = _unmix(x, a, b, 0.6 * min_step, rel_res * 1.5)
        t[has] = tt
        bad = np.zeros((H, W), bool)
        bad[has] = ~ok
        has &= ~bad
    # a mixed pixel with no clean pair of sides (junctions, soft blends) counts as pure: its own colour
    orphan = mixed & ~has
    pure = pure | orphan
    ia = np.where(has, ia, -1)
    ib = np.where(has, ib, -1)
    thin &= pure
    return {"pure": pure, "ia": ia, "ib": ib, "t": t, "orphan": int(orphan.sum()), "mixed": int(has.sum()),
            "thin": thin, "ground": ground.astype(np.float32)}


# ------------------------------------------------------------------------------------------------ pieces
def pieces(ref, pure, *, tol=4.5, lab=None, thin=None, ground=None):
    """Label the pure pixels into colour-coherent pieces: 4-neighbours chained while their colour differs by less
    than ``tol`` Delta E; diagonal neighbours only when both are darker than the two pixels between them (a thin
    dark line holds together, the areas on either side of it stay apart).  ``thin`` / ``ground`` (from
    ``mixed_pixels``): the cores of thin marks chain (8-connected) while they lie on the same ground and their colour
    departs from it in the same direction (a 1 px line whose pixels are more or less ink).  Returns int32 [H,W]
    (0 = mixed)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    if lab is None:
        lab = to_lab(ref)
    # a light denoise for the chaining only (JPEG speckle breaks flats into crumbs)
    labs = cv2.bilateralFilter(lab, 5, 6.0, 2.0)
    L = labs[..., 0]
    idx = np.arange(H * W, dtype=np.int64).reshape(H, W)
    rows, cols = [], []

    def de(a, b):
        return np.sqrt(((a - b) ** 2).sum(-1))

    if thin is not None and thin.any():
        x = np.asarray(ref, np.float32)
        u = x - ground
        u /= np.maximum(np.sqrt((u * u).sum(-1, keepdims=True)), 1e-6)
        for a, b in (((slice(0, H), slice(0, W - 1)), (slice(0, H), slice(1, W))),
                     ((slice(0, H - 1), slice(0, W)), (slice(1, H), slice(0, W))),
                     ((slice(0, H - 1), slice(0, W - 1)), (slice(1, H), slice(1, W))),
                     ((slice(0, H - 1), slice(1, W)), (slice(1, H), slice(0, W - 1)))):
            ok = thin[a] & thin[b] & ((u[a] * u[b]).sum(-1) > 0.95) & \
                (np.sqrt(((ground[a] - ground[b]) ** 2).sum(-1)) < 18.0)
            rows.append(idx[a][ok])
            cols.append(idx[b][ok])

    # a thin core joins an ordinary pixel only where that pixel is the line's own ink (the thick part of the same
    # line: close to the ink of the thin mark it belongs to), never the ground or a shadow beside it (a faint fringe
    # would chain the line to them)
    if thin is not None and thin.any():
        xr = np.asarray(ref, np.float32)
        rt = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        ct = np.concatenate(cols) if cols else np.zeros(0, np.int64)
        gt = coo_matrix((np.ones(len(rt), np.uint8), (rt, ct)), shape=(H * W, H * W))
        _, tcomp = connected_components(gt, directed=False)
        tcomp = tcomp.reshape(H, W)
        offt = np.where(thin, np.sqrt(((xr - ground) ** 2).sum(-1)), 0.0)
        tid = np.where(thin, tcomp, 0).reshape(-1)
        nt = int(tcomp.max()) + 1
        mxo = np.zeros(nt)
        np.maximum.at(mxo, tid[thin.reshape(-1)], offt.reshape(-1)[thin.reshape(-1)])
        topm = thin & (offt >= 0.85 * mxo[tcomp])
        tt = np.where(topm, tcomp, 0).reshape(-1)
        cnt = np.bincount(tt, weights=topm.reshape(-1).astype(np.float64), minlength=nt)
        inkl = np.stack([np.bincount(tt, weights=(lab[..., k] * topm).reshape(-1), minlength=nt) for k in range(3)], 1)
        inkl /= np.maximum(cnt, 1)[:, None]
        ink_at = inkl[tcomp]            # [H,W,3] Lab ink of the thin mark through each thin pixel

        def ok_mixed(a, b):
            ta, tb = thin[a], thin[b]
            near_b = de(lab[b], ink_at[a]) < 2.5 * tol
            near_a = de(lab[a], ink_at[b]) < 2.5 * tol
            return (~ta & ~tb) | (ta & tb) | (ta & ~tb & near_b) | (tb & ~ta & near_a)
    else:
        def ok_mixed(a, b):
            return True
    # right, down
    for (sy, sx) in ((0, 1), (1, 0)):
        a = (slice(0, H - sy), slice(0, W - sx))
        b = (slice(sy, H), slice(sx, W))
        ok = pure[a] & pure[b] & (de(labs[a], labs[b]) < tol) & ok_mixed(a, b)
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
    # diagonals (dark thin lines)
    for (sx,) in ((1,), (-1,)):
        if sx == 1:
            a = (slice(0, H - 1), slice(0, W - 1))
            b = (slice(1, H), slice(1, W))
            m1 = (slice(0, H - 1), slice(1, W))
            m2 = (slice(1, H), slice(0, W - 1))
        else:
            a = (slice(0, H - 1), slice(1, W))
            b = (slice(1, H), slice(0, W - 1))
            m1 = (slice(0, H - 1), slice(0, W - 1))
            m2 = (slice(1, H), slice(1, W))
        mx = np.maximum(L[a], L[b])
        ok = pure[a] & pure[b] & (de(labs[a], labs[b]) < tol) & (mx + 3.0 < L[m1]) & (mx + 3.0 < L[m2]) & \
            ok_mixed(a, b)
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(H * W, H * W))
    _, comp = connected_components(g, directed=False)
    comp = comp.reshape(H, W)
    lab_img = np.zeros((H, W), np.int32)
    u, inv = np.unique(comp[pure], return_inverse=True)
    lab_img[pure] = inv.astype(np.int32) + 1
    return lab_img


def merge_through_mixed(lab_img, mp, lab, *, tol=4.5):
    """Join pieces of one colour that touch only through mixed pixels (a 1-1.5 px line has few or no pure pixels
    along it: its pure dots hold together through the edge pixels that are mostly line).  A mixed pixel belongs to
    the side it mostly shows; neighbouring pixels (8-connected) of two pieces whose colours differ by less than
    ``tol`` Delta E join them."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = int(lab_img.max())
    if n < 2:
        return lab_img
    H, W = lab_img.shape
    lf = lab_img.reshape(-1)
    has = mp["ia"] >= 0
    M = lab_img.copy()
    side_a = np.zeros((H, W), np.int32)
    side_b = np.zeros((H, W), np.int32)
    side_a[has] = lf[mp["ia"][has]]
    side_b[has] = lf[mp["ib"][has]]
    M = np.where(has, np.where(mp["t"] >= 0.5, side_a, side_b), M)
    cnt = np.bincount(lf, minlength=n + 1).astype(np.float64)
    mean = np.stack([np.bincount(lf, weights=lab[..., c].reshape(-1), minlength=n + 1) for c in range(3)], 1)
    mean /= np.maximum(cnt, 1)[:, None]
    rows, cols = [], []
    for a, b in (((slice(0, H), slice(0, W - 1)), (slice(0, H), slice(1, W))),
                 ((slice(0, H - 1), slice(0, W)), (slice(1, H), slice(0, W))),
                 ((slice(0, H - 1), slice(0, W - 1)), (slice(1, H), slice(1, W))),
                 ((slice(0, H - 1), slice(1, W)), (slice(1, H), slice(0, W - 1)))):
        p, q = M[a], M[b]
        ok = (p > 0) & (q > 0) & (p != q) & (has[a] | has[b])
        p, q = p[ok], q[ok]
        de = np.sqrt(((mean[p] - mean[q]) ** 2).sum(1))
        sel = de < tol
        rows.append(p[sel])
        cols.append(q[sel])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    if not len(r):
        return lab_img
    g = coo_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(n + 1, n + 1))
    _, comp = connected_components(g, directed=False)
    out = comp[lab_img].astype(np.int64) + 1
    out[lab_img == 0] = 0
    _, inv2 = np.unique(np.concatenate([[0], out.reshape(-1)]), return_inverse=True)
    return inv2[1:].reshape(H, W).astype(np.int32)


# ------------------------------------------------------------------------------------------------ outlines
def _edge_field(cov):
    """Signed distance (px) to the shape's edge, estimated from coverage: inside an edge pixel the edge lies
    ``cov - 0.5`` px from the pixel's centre (for an edge across the pixel); a pixel fully in or out takes its
    neighbour's distance plus or minus one pixel.  Interpolated linearly, its zero level places the edge where the
    coverage says, also for a line one pixel wide (whose interpolated coverage would put the edge too far out)."""
    c = np.asarray(cov, np.float32)
    edge = (c > 0.02) & (c < 0.98)
    f = np.where(edge, c - 0.5, np.where(c >= 0.98, 2.0, -2.0)).astype(np.float32)
    # one step out from the edge pixels (4-neighbours, distance 1), then clamp
    pad = np.pad(f, 1, constant_values=-2.0)
    epad = np.pad(edge, 1)
    out_v = np.full(c.shape, -2.0, np.float32)
    in_v = np.full(c.shape, 2.0, np.float32)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = pad[1 + dy:1 + dy + c.shape[0], 1 + dx:1 + dx + c.shape[1]]
        ne = epad[1 + dy:1 + dy + c.shape[0], 1 + dx:1 + dx + c.shape[1]]
        out_v = np.where(ne, np.maximum(out_v, nb - 1.0), out_v)
        in_v = np.where(ne, np.minimum(in_v, nb + 1.0), in_v)
    f = np.where(edge, f, np.where(c >= 0.98, np.maximum(in_v, 0.5), np.minimum(out_v, -0.5)))
    return np.clip(f, -1.5, 1.5)


def _area(q):
    x, y = q[:, 0].astype(np.float64), q[:, 1].astype(np.float64)
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]) + x[-1] * y[0] - y[-1] * x[0])


STRAIGHT_TOL = 0.3     # px: a run of outline within this of its chord is a straight edge, drawn as one segment
STRAIGHT_MIN = 6.0     # px: shortest run that is snapped straight


def straighten_ring(p, tol=STRAIGHT_TOL, min_len=STRAIGHT_MIN):
    """A closed outline [N,2] with its straight stretches made exactly straight: Douglas-Peucker at ``tol`` finds
    the stretches; every one at least ``min_len`` long whose points all lie within ``tol`` of its chord is replaced
    by points on the chord (the edge of a window frame, a letter's stem, a roof line stays straight instead of
    following the sub-pixel wobble of the coverage contour)."""
    n = len(p)
    if n < 8:
        return p
    q = cv2.approxPolyDP(p.astype(np.float32).reshape(-1, 1, 2), tol, True).reshape(-1, 2).astype(np.float64)
    if len(q) < 3:
        return p
    # indices of the simplified vertices on the original ring
    idx = []
    for v in q:
        idx.append(int(np.argmin(((p - v) ** 2).sum(1))))
    out = p.copy()
    m = len(idx)
    for k in range(m):
        i, j = idx[k], idx[(k + 1) % m]
        a, b = p[i], p[j]
        L = float(np.hypot(*(b - a)))
        if L < min_len:
            continue
        seg = np.arange(i, j + 1) if j >= i else np.concatenate([np.arange(i, n), np.arange(0, j + 1)])
        if len(seg) < 3 or len(seg) > n // 2 + 2 and m > 2:
            continue
        d = b - a
        t = ((p[seg] - a) @ d) / (L * L)
        proj = a + np.clip(t, 0, 1)[:, None] * d
        v = p[seg] - a
        if float(np.abs(d[0] * v[..., 1] - d[1] * v[..., 0]).max()) / L <= tol:     # 2-D cross product
            out[seg] = proj
    return out


def _fit_ring(cc, ss, sigma_px, eps_px):
    """Smooth closed outline from a marching-squares contour (ss-pixel units): Gaussian smoothing along the ring,
    corners kept (as ``masses.fit_ring``), simplified with Douglas-Peucker to ``eps_px``.  Canvas coordinates."""
    from .masses import _turn
    from .ops import _stair_share
    p = (cc.astype(np.float64) + 0.5) / ss
    n = len(p)
    if n < 3:
        return None
    if n >= 8:
        s = sigma_px * ss
        sm = np.stack([ndi.gaussian_filter1d(p[:, k], s, mode="wrap") for k in range(2)], 1)
        k = max(2, int(round(ss)))
        if n > 2 * k + 2:
            corner = _turn(p, k) > np.deg2rad(60.0)
            if corner.any():
                wgt = ndi.maximum_filter1d(corner.astype(np.float64), 2 * k + 1, mode="wrap")
                wgt = ndi.uniform_filter1d(wgt, k + 1, mode="wrap")
                sm = sm * (1 - wgt[:, None]) + p * wgt[:, None]
        p = sm
    p = straighten_ring(p)
    q = cv2.approxPolyDP(p.astype(np.float32).reshape(-1, 1, 2), eps_px, True).reshape(-1, 2)
    if len(q) > 12 and _stair_share(q) > 0.4:
        p0 = (cc.astype(np.float64) + 0.5) / ss
        p0 = np.stack([ndi.gaussian_filter1d(p0[:, k], 0.45 * ss, mode="wrap") for k in range(2)], 1)
        q = cv2.approxPolyDP(p0.astype(np.float32).reshape(-1, 1, 2), eps_px, True).reshape(-1, 2)
    if len(q) < 3:
        return None
    return np.ascontiguousarray(q, dtype=np.float32)


def _rings_from_cov(cov, x0, y0, ss, smooth_px=0.25):
    """Rings (canvas coordinates) of the 0.5 iso-contour of a coverage crop, upsampled ``ss`` x (cubic).
    Returns list of (ring [N,2] float32, signed area > 0 for outer rings)."""
    from skimage.measure import find_contours
    h, w = cov.shape
    up = cv2.resize(_edge_field(cov), (w * ss, h * ss), interpolation=cv2.INTER_LINEAR)
    up = np.pad(up, 1, constant_values=-1.0)
    out = []
    for c in find_contours(up, 0.0, positive_orientation="high"):
        if len(c) < 4:
            continue
        # contour in ss-grid sample indices (row, col) of the padded array -> (x, y) in ss pixels
        cc = np.stack([c[:, 1] - 1 + x0 * ss, c[:, 0] - 1 + y0 * ss], 1)
        if np.allclose(cc[0], cc[-1]):
            cc = cc[:-1]
        r = _fit_ring(cc, ss, smooth_px, 0.06)
        if r is None:
            continue
        a = -_area(r)               # (x, y) with y down: an outer ring (high values on its left) is negative
        if abs(a) < 0.3:
            continue
        out.append((r, a))
    return out


FILL_GROW_PX = 0.125   # brush.render_fill's rasteriser (cv2.fillPoly, inclusive of the boundary, 4x supersampled)
                       # paints half a subpixel beyond the outline; the outlines are moved in by as much


def _compensate(polys):
    """Outer ring moved in, holes moved out, by FILL_GROW_PX (so the painted edge lies on the fitted edge)."""
    from .masses import _offset_ring
    out = []
    for k, r in enumerate(polys):
        q = _offset_ring(r, FILL_GROW_PX, k > 0)
        if q is None:
            if k == 0:
                return polys
            continue
        out.append(q)
    return out


UNDERLAY_PX = 1.0


def _distinct(col, other, min_de):
    """True when ``col`` differs from ``other`` (None = unknown: True) by at least ``min_de`` Delta E."""
    if other is None:
        return True
    lab = to_lab(np.array([[col, other]], np.uint8))[0]
    return float(np.sqrt(((lab[0] - lab[1]) ** 2).sum())) >= min_de


def _grow(polys, d):
    """Outer ring moved out, holes moved in, by ``d`` px (None when the outer ring fails)."""
    from .masses import _offset_ring
    out = []
    for k, r in enumerate(polys):
        q = _offset_ring(r, d, k == 0)
        if q is None:
            if k == 0:
                return None
            continue
        out.append(q)
    return out


def _evaluate(alt, C, deC, labR, bx0, by0, H, W):
    """Paint the ops ``alt`` in order on a copy of the work canvas ``C`` (box coordinates) and measure: (gain,
    better, worse, painted px, old error, box, new crop, new error crop) or None."""
    from .brush import _render
    h, w = C.shape[:2]
    rs = []
    for op in alt:
        r = _render(op, (H, W))
        if r is None:
            return None
        rs.append(r)
    sx0 = max(min(r[0] for r in rs) - bx0, 0)
    sy0 = max(min(r[1] for r in rs) - by0, 0)
    sx1 = min(max(r[0] + r[2].shape[1] for r in rs) - bx0, w)
    sy1 = min(max(r[1] + r[2].shape[0] for r in rs) - by0, h)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    new = C[sy0:sy1, sx0:sx1].copy()
    cover = np.zeros(new.shape[:2], np.float32)
    for ox, oy, a, rgb in rs:
        ox -= bx0 + sx0
        oy -= by0 + sy0
        hh, ww = a.shape
        tx0, ty0 = max(ox, 0), max(oy, 0)
        tx1, ty1 = min(ox + ww, sx1 - sx0), min(oy + hh, sy1 - sy0)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        aa = a[ty0 - oy:ty1 - oy, tx0 - ox:tx1 - ox]
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            rgb = rgb[ty0 - oy:ty1 - oy, tx0 - ox:tx1 - ox]
        rgb = np.asarray(rgb, np.float32) * 255.0
        sub = new[ty0:ty1, tx0:tx1]
        new[ty0:ty1, tx0:tx1] = sub * (1.0 - aa[..., None]) + rgb * aa[..., None]
        cover[ty0:ty1, tx0:tx1] = np.maximum(cover[ty0:ty1, tx0:tx1], aa)
    fp = cover > 0.004
    if not fp.any():
        return None
    de_new = np.sqrt(((to_lab(np.clip(np.rint(new), 0, 255).astype(np.uint8)) - labR[sy0:sy1, sx0:sx1]) ** 2)
                     .sum(-1))
    de_old = deC[sy0:sy1, sx0:sx1]
    dd = (de_old - de_new)[fp]
    return (float(dd.sum()), float(dd[dd > 0].sum()), float(-dd[dd < 0].sum()), float(cover[fp].sum()),
            float(de_old[fp].sum()), (sx0, sy0, sx1, sy1), new, de_new)


def _group_rings(rings):
    """Group rings into (outer, holes) by orientation and containment."""
    outers = [(r, a) for r, a in rings if a > 0]
    holes = [(r, a) for r, a in rings if a <= 0]
    if not outers:
        # orientation flipped (should not happen): treat the largest as outer
        rings = sorted(rings, key=lambda t: -abs(t[1]))
        outers = [(rings[0][0], abs(rings[0][1]))]
        holes = [(r, -abs(a)) for r, a in rings[1:]]
    groups = [[o, []] for o in outers]
    for hr, ha in holes:
        p = (float(hr[0, 0]), float(hr[0, 1]))
        best = None
        for gi, (o, _) in enumerate(groups):
            if o[1] <= abs(ha):
                continue
            if cv2.pointPolygonTest(o[0].reshape(-1, 1, 2).astype(np.float32), p, False) >= 0:
                if best is None or o[1] < groups[best][0][1]:
                    best = gi
        if best is not None:
            groups[best][1].append((hr, ha))
    return groups


# ------------------------------------------------------------------------------------------------ the pass
def shape_ops(ref, mask=None, canvas=None, *, tol=4.5, min_step=16.0, ss=4, min_gain=0.6, abs_gain=8.0,
              cand_de=4.0, max_area=None, gradients=True, small=True, rounds=2, budget=12.0, seed=0, stats=None,
              exclude=None, min_hard=0.5, bad_px=3, repair_share=0.3, clean_ratio=0.25, extend_under=False,
              underlay=True, band_above=3000.0, band_px=3, band_tile=32, small_de=20.0, debug=None):
    """Fills (and small marks) that move ``canvas`` toward ``ref`` where the reference has small hard-edged shapes
    the canvas gets wrong (see the module doc).  ``mask``: only pieces anchored in it (each one whole, ``own``).

    tol: Delta E chaining tolerance of a piece; min_step: RGB step that makes an edge (pixels between are mixes);
    min_gain: required mean Delta E improvement per footprint px; abs_gain: and in total; cand_de: a piece is tried
    when its footprint is on average this far off, or has ``bad_px`` pixels 3 x that far off; max_area: largest piece
    fitted whole (px, default 3 % of the picture); band_above: pieces larger than this are redrawn only along their
    hard edge (a ``band_px`` strip in ``band_tile`` px stretches); repair_share / clean_ratio: further acceptance
    (a share of the error under the mark closed; improvement with at most this ratio of spoiling); underlay: also
    try each fill over its outline grown 1 px in the colour across its edge; small_de: a mark under 12 px^2 must
    differ this much from its surroundings; min_hard: a piece must be bounded at least this much by sharp edges;
    exclude: bool [H,W] pixels whose pieces are left alone (e.g. grey hatching); budget: seconds (stats
    "out_of_time" when it ran out); debug: a dict that collects the internals (diagnostics)."""
    from .masses import _small_mark
    from .ops import is_partial, owned_ops
    t_start = time.perf_counter()
    ref = np.asarray(ref)[..., :3]
    H, W = ref.shape[:2]
    stats = {} if stats is None else stats
    if canvas is None:
        canvas = np.full_like(ref, 255)
    canvas = np.asarray(canvas)[..., :3]
    full = mask is None or not is_partial(mask)
    m = np.ones((H, W), bool) if full else (np.asarray(mask) > 0.5 if np.asarray(mask).dtype != bool
                                            else np.asarray(mask))
    if not m.any():
        return []
    # work on the mask's box with a margin (pieces anchored in the mask, reaching out of the box, are cut by it:
    # those touching the box's inner edge are dropped)
    MARG = 32
    ys, xs = np.nonzero(m)
    bx0, by0 = max(int(xs.min()) - MARG, 0), max(int(ys.min()) - MARG, 0)
    bx1, by1 = min(int(xs.max()) + 1 + MARG, W), min(int(ys.max()) + 1 + MARG, H)
    R = np.ascontiguousarray(ref[by0:by1, bx0:bx1])
    C = canvas[by0:by1, bx0:bx1].astype(np.float32)
    h, w = R.shape[:2]
    labR = to_lab(R)
    mp = mixed_pixels(R, min_step=min_step, lab=labR)
    thin, ground = mp["thin"], mp["ground"]
    lab_img = pieces(R, mp["pure"], tol=tol, lab=labR, thin=thin, ground=ground)
    lab_img = merge_through_mixed(lab_img, mp, labR, tol=tol)
    n = int(lab_img.max())
    if n == 0:
        return []
    lf = lab_img.reshape(-1)
    # a piece made mostly of thin cores is a thin mark: its colour is its ink (the colour of its most saturated
    # cores, as far from the ground as the mark gets) and each core pixel is that ink to the extent it departs
    # from its ground (the coverage ``cpx``); any other pure pixel is fully its piece's colour
    Rf = R.astype(np.float32)
    npx = np.bincount(lf, minlength=n + 1).astype(np.float64)
    nthin = np.bincount(lf, weights=thin.reshape(-1).astype(np.float64), minlength=n + 1)
    is_thin = (nthin >= 0.3 * np.maximum(npx, 1)) & (npx > 0)
    is_thin[0] = False
    offv = np.sqrt(((Rf - ground) ** 2).sum(-1))
    offv = np.where(lab_img > 0, offv, 0.0)
    mx = np.zeros(n + 1)
    if is_thin.any():
        mx[1:] = ndi.maximum(offv, lab_img, np.arange(1, n + 1))
    top = (lab_img > 0) & (offv >= 0.85 * mx[lab_img]) & is_thin[lab_img]
    ltop = np.where(top, lab_img, 0).reshape(-1)
    ntop = np.bincount(ltop, minlength=n + 1).astype(np.float64)
    ink = np.stack([np.bincount(ltop, weights=Rf[..., k].reshape(-1), minlength=n + 1) for k in range(3)], 1)
    ink /= np.maximum(ntop, 1)[:, None]
    cpx = np.ones((h, w), np.float32)
    tm = thin & is_thin[lab_img]
    if tm.any():
        F = ink[lab_img[tm]]
        G = ground[tm]
        v = F - G
        cpx[tm] = np.clip(((Rf[tm] - G) * v).sum(-1) / np.maximum((v * v).sum(-1), 1e-6), 0.0, 1.0)
    cflat = cpx.reshape(-1)
    ia, ib, tmix = mp["ia"], mp["ib"], mp["t"]
    has = ia >= 0
    la = np.zeros((h, w), np.int32)
    lb = np.zeros((h, w), np.int32)
    la[has] = lf[ia[has]]
    lb[has] = lf[ib[has]]
    # a mixed pixel's share of each side, times that side pixel's own coverage
    ta = np.zeros((h, w), np.float32)
    tb = np.zeros((h, w), np.float32)
    ta[has] = tmix[has] * cflat[ia[has]]
    tb[has] = (1.0 - tmix[has]) * cflat[ib[has]]
    # the canvas's error
    deC = np.sqrt(((to_lab(C.astype(np.uint8)) - labR) ** 2).sum(-1))
    # piece footprints: pure pixels + mixed pixels weighted by their share
    wa = ta
    wb = tb
    area = (np.bincount(lab_img.ravel(), weights=cpx.ravel(), minlength=n + 1)
            + np.bincount(la.ravel(), weights=wa.ravel(), minlength=n + 1)
            + np.bincount(lb.ravel(), weights=wb.ravel(), minlength=n + 1))
    area[0] = 0
    max_area = float(max_area) if max_area is not None else max(0.03 * H * W, 4000.0)
    # pieces that touch the work box's inner edge (not the picture's edge) are cut by it: left out
    edge = np.zeros((h, w), bool)
    if bx0 > 0:
        edge[:, 0] = True
    if by0 > 0:
        edge[0, :] = True
    if bx1 < W:
        edge[:, -1] = True
    if by1 < H:
        edge[-1, :] = True
    cut = np.zeros(n + 1, bool)
    cut[np.unique(lab_img[edge])] = True
    if exclude is not None:
        ex = np.asarray(exclude, bool)[by0:by1, bx0:bx1]
        exs = np.bincount(lab_img.ravel(), weights=ex.ravel().astype(np.float64), minlength=n + 1)
        cut |= exs > 0.3 * np.bincount(lab_img.ravel(), minlength=n + 1)
    cut[0] = True
    # hard pieces only: a piece bounded mostly by soft steps (pure pixels of another piece right beside its own:
    # a smooth ramp cut into bands by the chaining tolerance) is shading, not a shape; it stays with the strokes
    if min_hard > 0:
        soft = np.zeros(n + 1)
        for a_, b_ in (((slice(0, h), slice(0, w - 1)), (slice(0, h), slice(1, w))),
                       ((slice(0, h - 1), slice(0, w)), (slice(1, h), slice(0, w)))):
            p, q = lab_img[a_], lab_img[b_]
            sel = (p > 0) & (q > 0) & (p != q)
            soft += np.bincount(p[sel], minlength=n + 1) + np.bincount(q[sel], minlength=n + 1)
        hard = np.bincount(la.ravel(), minlength=n + 1) + np.bincount(lb.ravel(), minlength=n + 1)
        hard = hard.astype(np.float64)
        hard[0] = 0
        hardness = hard / np.maximum(hard + soft, 1.0)
        cut |= (hardness < min_hard) & ~is_thin
    # anchored in the mask: the piece's deepest pure pixel is not known yet; use "has a pure pixel in the mask"
    # (the final ownership is decided by ops.owned_ops on the finished marks)
    slices = ndi.find_objects(lab_img)
    sl_a = ndi.find_objects(la)
    sl_b = ndi.find_objects(lb)

    def box_of(i):
        bxs = [s for s in (slices[i - 1] if i - 1 < len(slices) else None,
                           sl_a[i - 1] if i - 1 < len(sl_a) else None,
                           sl_b[i - 1] if i - 1 < len(sl_b) else None) if s is not None]
        y0 = min(s[0].start for s in bxs)
        y1 = max(s[0].stop for s in bxs)
        x0 = min(s[1].start for s in bxs)
        x1 = max(s[1].stop for s in bxs)
        return x0, y0, x1, y1

    ops = []
    tried = accepted = smalls = 0
    tl = time.perf_counter
    order_pos = np.zeros(n + 1, np.int64)
    done = np.zeros(n + 1, bool)
    out_of_time = False
    for rnd in range(max(1, rounds)):
        # candidates: pieces whose footprint the canvas gets wrong
        e_sum = (np.bincount(lab_img.ravel(), weights=deC.ravel(), minlength=n + 1)
                 + np.bincount(la.ravel(), weights=(deC * wa).ravel(), minlength=n + 1)
                 + np.bincount(lb.ravel(), weights=(deC * wb).ravel(), minlength=n + 1))
        mean_e = e_sum / np.maximum(area, 1e-6)
        # ... or that have a few clearly wrong pixels (a speck of paper inside a line, a nick in an edge)
        bad = (deC > 3.0 * cand_de).astype(np.float64)
        n_bad = np.bincount(lab_img.ravel(), weights=bad.ravel(), minlength=n + 1) + \
            np.bincount(la.ravel(), weights=(bad * (wa >= 0.5)).ravel(), minlength=n + 1) + \
            np.bincount(lb.ravel(), weights=(bad * (wb >= 0.5)).ravel(), minlength=n + 1)
        cand = (~cut) & (~done) & (area >= 3.0) & ((area <= max_area) | (area > band_above)) & \
            (((mean_e >= cand_de) & (e_sum >= abs_gain * 2)) | (n_bad >= bad_px))
        if isinstance(debug, dict) and rnd == 0:
            debug.update(lab=lab_img, cut=cut.copy(), area=area, mean_e=mean_e.copy(), cand=cand.copy(),
                         box=(bx0, by0, bx1, by1), mp=mp, is_thin=is_thin, deC=deC.copy(), tried=[],
                         n_bad=n_bad.copy())
        ids = np.flatnonzero(cand)
        if not len(ids):
            break
        ids = ids[np.argsort(-area[ids], kind="stable")]
        order_pos[:] = -1
        order_pos[ids] = np.arange(len(ids))
        for i in ids:
            if tl() - t_start > budget:
                out_of_time = True
                break
            tried += 1
            x0, y0, x1, y1 = box_of(int(i))
            x0, y0 = max(x0 - 2, 0), max(y0 - 2, 0)
            x1, y1 = min(x1 + 2, w), min(y1 + 2, h)
            Lc = lab_img[y0:y1, x0:x1]
            core = Lc == i
            A = la[y0:y1, x0:x1]
            B = lb[y0:y1, x0:x1]
            cov = np.where(core, cpx[y0:y1, x0:x1], 0.0) + np.where(A == i, ta[y0:y1, x0:x1], 0.0) + \
                np.where(B == i, tb[y0:y1, x0:x1], 0.0)
            # under a later (smaller) piece the edge pixels are covered fully: that piece is painted on top with its
            # own antialiased edge, so no canvas shows through the seam
            later_b = (A == i) & (order_pos[B] > order_pos[i]) & (B > 0)
            later_a = (B == i) & (order_pos[A] > order_pos[i]) & (A > 0)
            if extend_under:
                cov = np.where(later_a | later_b, 1.0, cov)
            cov = np.clip(cov, 0.0, 1.0).astype(np.float32)
            Rc = R[y0:y1, x0:x1]
            layers_ = []                  # (coverage, colour, gradient or None)
            if is_thin[i]:
                # a thin mark is as dark as it gets where it is solid; where the reference shows it fainter (a
                # tapered end, a line blurred over two pixels) a crisp full-ink shape would be too dark, so each
                # stretch is drawn at its own level: the ink mixed with the ground by the line's local peak
                # coverage, as wide as that level holds the same ink
                F = ink[i]
                gm = ground[y0:y1, x0:x1][core].mean(0) if core.any() else F
                peak = cv2.dilate(cov, np.ones((3, 3), np.uint8))
                for lo, hi, lev in THIN_LEVELS:
                    part = (peak >= lo) & (peak < hi) & (cov > 0)
                    if not part.any():
                        continue
                    # the part and a pixel around it (its edge pixels belong to it)
                    near = cv2.dilate(part.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    near &= (peak >= lo - 0.08) & (peak < hi + 0.08)
                    cl = np.where(near, np.clip(cov / lev, 0.0, 1.0), 0.0).astype(np.float32)
                    colv = gm + (F - gm) * lev
                    layers_.append((cl, 0, 0, tuple(int(round(v)) for v in np.clip(colv, 0, 255)), None))
            elif area[i] > band_above:
                # a large piece (a flat, a hair mass) is not repainted whole (its inside holds shading the earlier
                # stages put there): only a strip along its hard edge, ``band_px`` wide, cut into stretches of about
                # ``band_tile`` px with their own eyedropped colour: the edge comes back crisp where it went soft
                edge_px = ((A == i) | (B == i)).astype(np.uint8)
                if not edge_px.any():
                    done[i] = True
                    continue
                kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
                band = cv2.dilate(edge_px, kb) > 0
                hh_, ww_ = cov.shape
                ys_, xs_ = np.nonzero(band)
                for ty in range(int(ys_.min()) // band_tile * band_tile, hh_, band_tile):
                    for tx in range(int(xs_.min()) // band_tile * band_tile, ww_, band_tile):
                        t_sl = (slice(ty, ty + band_tile), slice(tx, tx + band_tile))
                        bt = band[t_sl]
                        if not bt.any():
                            continue
                        sel = bt & core[t_sl]
                        if sel.sum() < 6:
                            continue
                        cl = np.where(bt, cov[t_sl], 0.0).astype(np.float32)
                        if cl.max() < 0.5:
                            continue
                        col = tuple(int(round(v)) for v in np.median(Rc[t_sl][sel].astype(np.float32), 0))
                        layers_.append((cl, tx, ty, col, None))
                if not layers_:
                    done[i] = True
                    continue
            else:
                if cov.max() < 0.5:
                    done[i] = True
                    continue
                # colour: the median over the piece's pure pixels (else the pixels it mostly covers)
                sel = core if core.sum() >= 1 else cov >= 0.7
                if not sel.any():
                    done[i] = True
                    continue
                px = Rc[sel].astype(np.float32)
                col = tuple(int(round(v)) for v in np.median(px, 0))
                grad = None
                if gradients and len(px) >= 40:
                    from .trace import fit_gradient, lab_to_rgb
                    yy, xx = np.nonzero(sel)
                    g = fit_gradient(labR[y0:y1, x0:x1][sel].astype(np.float64),
                                     np.column_stack([xx + bx0 + x0 + 0.5, yy + by0 + y0 + 0.5]))
                    if g is not None:
                        c0, c1, p0, p1 = g
                        rgb01 = lab_to_rgb(np.array([c0, c1], np.float32)[None])[0]
                        grad = (tuple(int(round(v)) for v in rgb01[0]), tuple(int(round(v)) for v in rgb01[1]),
                                (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
                layers_.append((cov, 0, 0, col, grad))
            if isinstance(debug, dict):
                debug.setdefault("covs", {})[int(i)] = (cov.copy(), bx0 + x0, by0 + y0)
            # the colour on the other side of the piece's edge (where the reference mixes the two): an underlay in
            # that colour, a pixel wider than the piece, puts the ground its antialiased edge expects under it (the
            # canvas there may already hold the mixed colour, and the edge laid over it would mix it again)
            nb_col = None
            sa = (A == i)
            sb = (B == i)
            if sa.any() or sb.any():
                oth = np.concatenate([ib[y0:y1, x0:x1][sa], ia[y0:y1, x0:x1][sb]])
                oth = oth[oth >= 0]
                if len(oth) >= 4:
                    nb_col = tuple(int(round(v)) for v in np.median(Rf.reshape(-1, 3)[oth], 0))
            groups = []
            for cl, lx0, ly0, col, grad in layers_:
                if cl.max() < 0.5:
                    continue
                # only the part of the crop the layer covers (a band stretch is small inside a large piece's box)
                nzy = np.flatnonzero(cl.max(1) > 0)
                nzx = np.flatnonzero(cl.max(0) > 0)
                cy0_, cy1_ = max(int(nzy[0]) - 2, 0), min(int(nzy[-1]) + 3, cl.shape[0])
                cx0_, cx1_ = max(int(nzx[0]) - 2, 0), min(int(nzx[-1]) + 3, cl.shape[1])
                rings = _rings_from_cov(np.ascontiguousarray(cl[cy0_:cy1_, cx0_:cx1_]), bx0 + x0 + lx0 + cx0_,
                                        by0 + y0 + ly0 + cy0_, ss)
                for (o, oa), hs in _group_rings(rings) if rings else []:
                    a_net = float(oa) - sum(abs(float(x)) for _, x in hs)
                    polys = _compensate([o] + [r for r, _ in hs])
                    if a_net >= 12.0 and mark_area(Fill(polygons=polys)) < 12.0:
                        polys = [o] + [r for r, _ in hs]
                    if a_net >= 12.0 and mark_area(Fill(polygons=polys)) >= 12.0:
                        if grad is not None:
                            op = Fill(polygons=polys, color=grad[0], color2=grad[1], p0=grad[2], p1=grad[3])
                        else:
                            op = Fill(polygons=polys, color=col)
                        alts = [[op]]
                        if nb_col is not None and underlay:
                            ul = _grow(polys, UNDERLAY_PX)
                            if ul is not None:
                                alts.append([Fill(polygons=ul, color=nb_col), op])
                        groups.append(alts)
                    elif small and a_net >= 2.0 and _distinct(col, nb_col, small_de):
                        # a speck or glint that stands out from what surrounds it (texture grain does not)
                        op = _small_mark(o, a_net, col, seed + len(ops), ground=nb_col)
                        if op is not None:
                            groups.append([[op]])
            if not groups:
                done[i] = True
                continue
            any_acc = False
            if debug is not None:
                (debug["tried"] if isinstance(debug, dict) else debug).append(
                    (int(i), float(area[i]), [type(g[0][0]).__name__ for g in groups], len(groups)))
            for alts in groups:
                best = None
                for alt in alts:
                    ev = _evaluate(alt, C, deC, labR, bx0, by0, H, W)
                    if ev is None:
                        continue
                    gain, better, worse, wsum, old_sum, box, new, de_new = ev
                    need = abs_gain if isinstance(alt[-1], Fill) else 0.5 * abs_gain
                    # kept when it closes enough of the gap per px it paints, or (a repair inside a mostly right
                    # shape) a clear share of the error under it, or (a cleaner edge, a sharper corner) it improves
                    # places and spoils hardly any
                    ok_gain = gain >= need and (gain >= min_gain * wsum or gain >= repair_share * old_sum
                                                or worse <= clean_ratio * better)
                    if isinstance(debug, dict):
                        debug.setdefault("evals", []).append((int(i), alt[-1], round(gain, 1), round(wsum, 1),
                                                              round(old_sum, 1), ok_gain, len(alt)))
                    if ok_gain and (best is None or gain > best[0]):
                        best = (gain, alt, box, new, de_new)
                if best is not None:
                    _, alt, (sx0, sy0, sx1, sy1), new, de_new = best
                    C[sy0:sy1, sx0:sx1] = new
                    deC[sy0:sy1, sx0:sx1] = de_new
                    ops.extend(alt)
                    any_acc = True
                    for op in alt:
                        if isinstance(op, Fill):
                            accepted += 1
                        else:
                            smalls += 1
            done[i] = done[i] or any_acc
        if out_of_time:
            break
    if not full:
        ops = owned_ops(ops, m)
    stats.update({"pieces": n, "tried": tried, "fills": accepted, "small": smalls, "mixed": mp["mixed"],
                  "orphan": mp["orphan"], "seconds": round(time.perf_counter() - t_start, 2),
                  "out_of_time": out_of_time})
    return ops
