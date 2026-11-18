"""Shared image analysis for the stage tools: Lab conversion, dark-line maps, skeleton -> ordered polylines with
per-point width, edge maps, colour clustering in Lab and small-region merging.

All functions take ``uint8 [H,W,3]`` sRGB images (or masks) and return numpy arrays; nothing touches the disk.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

__all__ = ["to_lab", "lab_to_rgb", "luminance", "delta_e", "line_map", "LineMap", "skeleton_polylines",
           "edge_map", "cluster_colors", "segments", "merge_small_regions", "region_colors", "orientation_field",
           "smooth_polyline", "as_mask"]


# ------------------------------------------------------------------------------------------------ colour
def to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 or float 0..255 RGB -> float32 Lab (L 0..100)."""
    a = np.asarray(rgb, np.float32)
    if a.max() > 1.0 + 1e-6 or np.asarray(rgb).dtype == np.uint8:
        a = a / 255.0
    return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2Lab)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float32 Lab -> float32 RGB 0..255."""
    lab = np.asarray(lab, np.float32)
    shp = lab.shape
    out = cv2.cvtColor(np.ascontiguousarray(lab.reshape(-1, 1, 3)), cv2.COLOR_Lab2RGB).reshape(shp)
    return np.clip(out * 255.0, 0, 255)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return to_lab(rgb)[..., 0]


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel CIE76 Delta E between two RGB images."""
    return np.sqrt(((to_lab(a) - to_lab(b)) ** 2).sum(-1))


def as_mask(mask, shape) -> np.ndarray:
    if mask is None:
        return np.ones(shape[:2], bool)
    m = np.asarray(mask)
    if m.dtype != bool:
        m = m > 0.5
    return m


# ------------------------------------------------------------------------------------------------ lines
@dataclass
class LineMap:
    strength: np.ndarray     # float32 [H,W] 0..1: how much each pixel is line ink (antialiased)
    mask: np.ndarray         # bool [H,W]: line pixels (strength >= 0.5), at the picture's own line width
    width: float             # typical line width in px (median over the skeleton)
    color: np.ndarray        # float32 RGB of the typical line ink


def line_map(ref: np.ndarray, *, max_width: float = 9.0, contrast: float = 0.0, mask=None) -> LineMap:
    """Thin dark ridges of the picture at its own line width.

    Multi-scale black-hat (closing minus image) on Lab lightness: a pixel is line ink when it is darker than its
    surroundings within a window of up to ``max_width`` px.  ``contrast`` (Lab L units) is the darkness at which a
    pixel counts as full ink; 0 = choose from the picture."""
    L = to_lab(ref)[..., 0]
    bh = np.zeros_like(L)
    radii = sorted({1, 2, 3, int(round(max_width / 2 + 0.5))})
    for r in radii:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        clo = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k)
        bh = np.maximum(bh, clo - L)
    m = as_mask(mask, L.shape)
    if contrast <= 0:
        v = bh[m & (bh > 6)]
        contrast = float(np.clip(np.percentile(v, 75) if v.size > 50 else 30.0, 14.0, 45.0))
    strength = np.clip((bh - 0.25 * contrast) / (0.75 * contrast), 0, 1).astype(np.float32)
    strength[~m] = 0
    lmask = strength >= 0.5
    # remove isolated specks (noise, texture) that are not line-like
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lmask.astype(np.uint8), connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < 4
    small[0] = False
    lmask[small[lab]] = False
    strength[~ndi.binary_dilation(lmask, iterations=2)] *= 0.0
    if lmask.any():
        dt = ndi.distance_transform_edt(lmask)
        sk = skeletonize(lmask)
        w = float(np.median(2 * dt[sk] - 1)) if sk.any() else 1.0
        col = ref[lmask].astype(np.float32)
        lum = L[lmask]
        col = col[lum <= np.percentile(lum, 40)].mean(0) if len(col) else np.zeros(3, np.float32)
    else:
        w, col = 1.0, np.zeros(3, np.float32)
    return LineMap(strength=strength, mask=lmask, width=max(w, 1.0), color=np.asarray(col, np.float32))


_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_graph(sk: np.ndarray):
    ys, xs = np.nonzero(sk)
    H, W = sk.shape
    idx = -np.ones((H + 2, W + 2), np.int64)
    idx[ys + 1, xs + 1] = np.arange(len(ys))
    nb = np.stack([idx[ys + 1 + dy, xs + 1 + dx] for dy, dx in _OFFS], 1)  # [n, 8]
    return ys, xs, nb


def skeleton_polylines(mask: np.ndarray, *, min_len: float = 3.0, spur: float = 0.0, dist=None):
    """Skeleton of a line mask as ordered polylines.

    Returns a list of (pts [N,2] float32 (x,y) pixel centres, widths [N] float32 from the distance transform,
    end_free (bool, bool): whether each end is a free line end (not a junction)).  ``spur`` removes branches shorter
    than that many px that end freely (skeleton noise at blobby lines)."""
    mask = mask.astype(bool)
    if not mask.any():
        return []
    sk = skeletonize(mask)
    if dist is None:
        dist = ndi.distance_transform_edt(mask)
    ys, xs, nb = _skeleton_graph(sk)
    n = len(ys)
    valid = nb >= 0
    deg = valid.sum(1)
    # cardinal-first neighbour preference keeps paths from cutting junction corners
    nbl = [[int(v) for v in row if v >= 0] for row in nb]
    is_node = deg != 2
    visited_edge = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        prev, cur = a, b
        while not is_node[cur]:
            nxt = [v for v in nbl[cur] if v != prev]
            if not nxt:
                break
            # prefer the unvisited continuation
            v = nxt[0]
            if len(nxt) > 1:
                for c in nxt:
                    if (min(cur, c), max(cur, c)) not in visited_edge:
                        v = c
                        break
            e = (min(cur, v), max(cur, v))
            if e in visited_edge:
                break
            visited_edge.add(e)
            path.append(v)
            prev, cur = cur, v
            if cur == path[0]:
                break
        return path

    for a in np.flatnonzero(is_node):
        a = int(a)
        if deg[a] == 0:
            paths.append([a])
            continue
        for b in nbl[a]:
            e = (min(a, b), max(a, b))
            if e in visited_edge:
                continue
            visited_edge.add(e)
            paths.append(walk(a, b))
    # loops without nodes
    seen = np.zeros(n, bool)
    for p in paths:
        seen[p] = True
    for a in np.flatnonzero(~seen):
        a = int(a)
        if seen[a]:
            continue
        b = nbl[a][0]
        visited_edge.add((min(a, b), max(a, b)))
        p = walk(a, b)
        seen[p] = True
        paths.append(p)

    out = []
    for p in paths:
        p = np.asarray(p)
        pts = np.stack([xs[p] + 0.5, ys[p] + 0.5], 1).astype(np.float64)
        seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)).sum() if len(p) > 1 else 0.0
        free0 = deg[p[0]] <= 1
        free1 = deg[p[-1]] <= 1
        w = 2.0 * dist[ys[p], xs[p]] - 1.0
        if seg < min_len and (free0 or free1) and len(paths) > 1:
            if seg < max(min_len, float(np.max(w))):
                continue
        if spur > 0 and (free0 != free1) and seg < spur:
            continue
        out.append((pts.astype(np.float32), np.maximum(w, 0.8).astype(np.float32), (bool(free0), bool(free1))))
    return out


def smooth_polyline(pts: np.ndarray, values=None, window: int = 5, keep_ends: bool = True):
    """Moving-average smoothing of a polyline (and per-point values) with fixed end points."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n < 3 or window < 2:
        return pts.astype(np.float32), values
    w = min(window, n if n % 2 else n - 1)
    if w < 3:
        return pts.astype(np.float32), values
    k = np.ones(w) / w
    h = w // 2
    closed = n > 4 and np.allclose(pts[0], pts[-1])
    mode = "wrap" if closed else "edge"
    sm = np.stack([np.convolve(np.pad(pts[:, i], h, mode=mode), k, "valid") for i in range(2)], 1)
    if keep_ends and not closed:
        sm[0], sm[-1] = pts[0], pts[-1]
    if values is not None:
        v = np.asarray(values, np.float64)
        if v.ndim == 1:
            v = np.convolve(np.pad(v, h, mode=mode), k, "valid")
        else:
            v = np.stack([np.convolve(np.pad(v[:, i], h, mode=mode), k, "valid") for i in range(v.shape[1])], 1)
        values = v
    return sm.astype(np.float32), values


# ------------------------------------------------------------------------------------------------ edges / flow
def edge_map(ref: np.ndarray, sigma: float = 1.0, low: float = 0.0, high: float = 0.0) -> np.ndarray:
    """Canny edges on Lab lightness; thresholds from the gradient distribution when not given."""
    L = to_lab(ref)[..., 0]
    if sigma > 0:
        L = cv2.GaussianBlur(L, (0, 0), sigma)
    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    if high <= 0:
        gx = cv2.Sobel(L8, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(L8, cv2.CV_32F, 0, 1)
        g = np.sqrt(gx * gx + gy * gy)
        nz = g[g > 4]
        high = float(np.percentile(nz, 80)) if nz.size else 100.0
        high = float(np.clip(high, 40, 300))
        low = 0.4 * high
    return cv2.Canny(L8, low, high, L2gradient=True) > 0


def orientation_field(img: np.ndarray, sigma: float = 2.0):
    """Isophote direction field of an RGB or grey image from the smoothed structure tensor.

    Returns (tx, ty, coherence) where (tx, ty) is the unit direction along the isophotes (perpendicular to the
    dominant gradient)."""
    if img.ndim == 3:
        chans = [to_lab(img)[..., k] * (1.0 if k == 0 else 0.6) for k in range(3)]
    else:
        chans = [img.astype(np.float32)]
    jxx = jyy = jxy = 0
    for c in chans:
        gx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
        jxx = jxx + gx * gx
        jyy = jyy + gy * gy
        jxy = jxy + gx * gy
    jxx = cv2.GaussianBlur(jxx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(jyy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(jxy, (0, 0), sigma)
    # dominant gradient angle
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    tx = -np.sin(theta)
    ty = np.cos(theta)
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-3)
    return tx.astype(np.float32), ty.astype(np.float32), coh.astype(np.float32)


# ------------------------------------------------------------------------------------------------ colour regions
def cluster_colors(ref: np.ndarray, k: int, mask=None, *, l_weight: float = 1.0, samples: int = 60000,
                   seed: int = 0):
    """k-means in Lab (lightness weighted by ``l_weight``). Returns (labels int32 [H,W] (-1 outside mask),
    centres uint8 RGB [k,3])."""
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    lab = to_lab(ref)
    feat = lab.copy()
    feat[..., 0] *= l_weight
    X = feat[m]
    labels = -np.ones((H, W), np.int32)
    if len(X) == 0:
        return labels, np.zeros((0, 3), np.uint8)
    k = int(max(1, min(k, len(X))))
    rng = np.random.default_rng(seed)
    S = X if len(X) <= samples else X[rng.choice(len(X), samples, replace=False)]
    cv2.setRNGSeed(int(seed))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, _, centres = cv2.kmeans(np.ascontiguousarray(S, np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1) if len(X) * k < 4e7 else None
    if d is None:
        lab_idx = np.empty(len(X), np.int32)
        for i in range(0, len(X), 200000):
            lab_idx[i:i + 200000] = ((X[i:i + 200000, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
    else:
        lab_idx = d.argmin(1).astype(np.int32)
    labels[m] = lab_idx
    c = centres.copy()
    c[:, 0] /= l_weight
    rgb = np.clip(np.rint(lab_to_rgb(c[None])[0]), 0, 255).astype(np.uint8)
    return labels, rgb


def segments(labels: np.ndarray) -> np.ndarray:
    """Connected components of equal labels (4-connectivity); -1 stays -1. Returns int32 segment ids."""
    out = -np.ones(labels.shape, np.int32)
    nxt = 0
    for v in np.unique(labels):
        if v < 0:
            continue
        n, cc = cv2.connectedComponents((labels == v).astype(np.uint8), connectivity=4)
        sel = cc > 0
        out[sel] = cc[sel] - 1 + nxt
        nxt += n - 1
    return out


def merge_small_regions(seg: np.ndarray, min_area: int, ref: np.ndarray = None) -> np.ndarray:
    """Merge segments smaller than ``min_area`` px into their neighbours (nearest-pixel assignment), so every pixel
    keeps a label and no speckle remains. Returns relabelled consecutive ids."""
    seg = seg.copy()
    valid = seg >= 0
    if not valid.any():
        return seg
    ids, counts = np.unique(seg[valid], return_counts=True)
    small_ids = ids[counts < min_area]
    if len(small_ids):
        small = np.isin(seg, small_ids)
        keep = valid & ~small
        if keep.any():
            _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
            seg[small] = seg[iy[small], ix[small]]
    _, inv = np.unique(seg[seg >= 0], return_inverse=True)
    out = -np.ones_like(seg)
    out[seg >= 0] = inv.astype(np.int32)
    return out


def region_colors(ref: np.ndarray, seg: np.ndarray, how: str = "median") -> np.ndarray:
    """RGB colour per segment id (median, mean or mode of the reference inside it). Returns uint8 [n,3]."""
    n = int(seg.max()) + 1 if seg.size and seg.max() >= 0 else 0
    out = np.zeros((n, 3), np.uint8)
    if n == 0:
        return out
    flat = seg.reshape(-1)
    ok = flat >= 0
    px = ref.reshape(-1, 3)[ok].astype(np.float32)
    lb = flat[ok]
    if how == "mean":
        for c in range(3):
            s = np.bincount(lb, px[:, c], minlength=n)
            cnt = np.bincount(lb, minlength=n)
            out[:, c] = np.clip(np.rint(s / np.maximum(cnt, 1)), 0, 255)
        return out
    order = np.argsort(lb, kind="stable")
    lb_s = lb[order]
    px_s = px[order]
    starts = np.searchsorted(lb_s, np.arange(n))
    ends = np.searchsorted(lb_s, np.arange(n), side="right")
    for i in range(n):
        p = px_s[starts[i]:ends[i]]
        if len(p) == 0:
            continue
        if how == "mode":
            q = (p // 8).astype(np.int32)
            key = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
            vals, cnts = np.unique(key, return_counts=True)
            best = vals[np.argmax(cnts)]
            out[i] = np.clip(np.rint(np.median(p[key == best], 0)), 0, 255)
        else:
            out[i] = np.clip(np.rint(np.median(p, 0)), 0, 255)
    return out


# ------------------------------------------------------------------------------------------------ vector shapes
def _rdp(pts, eps):
    if len(pts) < 4:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        n = float(np.hypot(*ab))
        seg = pts[i + 1:j]
        if n < 1e-9:
            d = np.hypot(*(seg - a).T)
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / n
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def _poly_area_signed(q):
    return 0.5 * float(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))


def _smooth_ring(c, sigma, shift, eps):
    """Smooth a closed pixel-traced ring; shift > 0 shrinks it by that many px, < 0 grows it."""
    if len(c) >= 8:
        s = min(sigma * (1.0 + 0.002 * len(c)), 3.0)
        c = np.stack([ndi.gaussian_filter1d(c[:, k], s, mode="wrap") for k in range(2)], 1)
    if abs(shift) > 1e-3 and len(c) >= 4:
        t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        a = c + nrm * abs(shift)
        b = c - nrm * abs(shift)
        smaller, larger = (a, b) if abs(_poly_area_signed(a)) < abs(_poly_area_signed(b)) else (b, a)
        c = smaller if shift > 0 else larger
    c = _rdp(np.vstack([c, c[:1]]), eps)[:-1]
    return c if len(c) >= 3 else None


def region_polygon(mask_crop: np.ndarray, x0: int, y0: int, *, grow: int = 0, sigma: float = 1.2,
                   eps: float = 0.3, offset: float = 0.5, holes: bool = False, min_hole: float = 12.0,
                   hole_offset=None):
    """Smooth outline of a region (crop with offset) as float32 polygons in canvas coordinates.

    Returns the outer ring, or with ``holes`` a list [outer, hole, ...] for even-odd filling.  The outer outline
    lies ``offset`` px outside the region's pixel edge (0 = exact edge; 0.5 lets shapes painted next to each other
    overlap instead of leaving seams); holes shrink by the same amount.  Outlines are Gaussian-smoothed along their
    length and simplified."""
    pad = grow + 2
    m0 = np.pad(mask_crop.astype(np.uint8), pad)
    m = cv2.dilate(m0, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) if grow > 0 else m0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=len)[:, 0, :].astype(np.float64)
    if len(c) < 3:
        return None
    off = np.array([x0 - pad + 0.5, y0 - pad + 0.5])
    # the traced outline runs through the centres of the grown ring: (grow - 0.5) px outside the pixel edge
    outer = _smooth_ring(c + off, sigma, (grow - 0.5) - offset, eps)
    if outer is None:
        return None
    if not holes:
        return outer.astype(np.float32)
    rings = [outer.astype(np.float32)]
    # holes of the raw region: their traced rings run through the region's own pixel centres, 0.5 px from the edge
    cs, hier = cv2.findContours(m0, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is not None:
        for ci, h in zip(cs, hier[0]):
            if h[3] < 0 or len(ci) < 4:
                continue  # an outer boundary
            hc = ci[:, 0, :].astype(np.float64)
            if abs(_poly_area_signed(hc)) < min_hole:
                continue
            r = _smooth_ring(hc + off, sigma * 0.8, 0.5 + (offset if hole_offset is None else hole_offset), eps)
            if r is not None and abs(_poly_area_signed(r)) >= min_hole:
                rings.append(r.astype(np.float32))
    return rings


def fit_gradient(lab_px: np.ndarray, xy: np.ndarray, *, min_range: float = 4.0, gain: float = 0.55):
    """Fit Lab = a + b.x + c.y over a region. Returns (lab0, lab1, p0, p1) when a linear gradient explains the
    region clearly better than one colour (residual RMS < gain x constant RMS and end-to-end change >= min_range
    Delta E), else None."""
    if len(lab_px) < 30:
        return None
    X = np.column_stack([np.ones(len(xy)), xy - xy.mean(0)])
    coef, *_ = np.linalg.lstsq(X, lab_px, rcond=None)
    pred = X @ coef
    rms_fit = float(np.sqrt(((lab_px - pred) ** 2).sum(1).mean()))
    rms_c = float(np.sqrt(((lab_px - lab_px.mean(0)) ** 2).sum(1).mean()))
    G = coef[1:]                     # [2,3]: d Lab / d (x, y)
    gv = np.sqrt((G ** 2).sum(1))    # strength per axis
    # direction of fastest Lab change
    M = G @ G.T
    w, v = np.linalg.eigh(M)
    d = v[:, -1]
    t = (xy - xy.mean(0)) @ d
    t0, t1 = np.percentile(t, 2), np.percentile(t, 98)
    if t1 - t0 < 2:
        return None
    c0 = coef[0] + (G.T @ d) * t0
    c1 = coef[0] + (G.T @ d) * t1
    if np.sqrt(((c1 - c0) ** 2).sum()) < min_range or rms_fit > gain * rms_c:
        return None
    p0 = xy.mean(0) + d * t0
    p1 = xy.mean(0) + d * t1
    return c0, c1, p0, p1


def segment_shapes(seg: np.ndarray, ref: np.ndarray, *, lab=None, gradients: bool = True, min_area: int = 12,
                   sigma: float = 1.2, sample=None, exact_below: float = 40.0, holes: bool = True,
                   grow_px: float = 0.5, hole_offset=None):
    """Vector shapes for every segment id: list of dicts with keys id, poly, area (filled outline), npx,
    lab (median), rgb, grad (None or (rgb0, rgb1, p0, p1)). ``sample``: bool mask of pixels used for colours
    (e.g. without line pixels)."""
    H, W = seg.shape
    if lab is None:
        lab = to_lab(ref)
    out = []
    slices = ndi.find_objects(seg + 1)
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        mk = seg[y0:y1, x0:x1] == i
        npx = int(mk.sum())
        if npx < 1:
            continue
        # large shapes reach 0.5 px under their neighbours (no seams); small shapes, painted last, keep their edge
        rings = region_polygon(mk, x0, y0, sigma=sigma if npx > 40 else 0.8,
                               offset=0.0 if npx < exact_below else grow_px, holes=holes, hole_offset=hole_offset)
        if rings is None:
            continue
        if not holes:
            rings = [rings]
        poly = rings[0]
        area = abs(_poly_area_signed(poly.astype(np.float64)))
        if area < min_area:
            continue
        use = mk if sample is None else (mk & sample[y0:y1, x0:x1])
        if use.sum() < max(3, 0.2 * npx):
            use = mk
        yy, xx = np.nonzero(use)
        lp = lab[y0:y1, x0:x1][use]
        med = np.median(lp, 0)
        grad = None
        if gradients and npx >= 60:
            g = fit_gradient(lp.astype(np.float64), np.column_stack([xx + x0 + 0.5, yy + y0 + 0.5]))
            if g is not None:
                c0, c1, p0, p1 = g
                rgb01 = lab_to_rgb(np.array([c0, c1], np.float32)[None])[0]
                grad = (tuple(int(round(v)) for v in rgb01[0]), tuple(int(round(v)) for v in rgb01[1]),
                        (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))
        rgb = tuple(int(round(v)) for v in lab_to_rgb(med.astype(np.float32)[None, None])[0, 0])
        out.append({"id": i, "poly": poly, "rings": rings, "area": area, "npx": npx, "lab": med, "rgb": rgb, "grad": grad,
                    "box": (x0, y0, x1, y1)})
    return out
