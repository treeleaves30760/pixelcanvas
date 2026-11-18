"""Region masks: box (+ GrabCut refine), numbered marks, polygon (live-wire snap), wand, plus / minus / rest.

All functions are pure (arrays in, arrays out); the CLI stores the results in the project.
"""
from __future__ import annotations

import heapq
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import coords
from .coords import UsageError

RESERVED = {"all", "canvas", "whole", "everything", "rest", "none"}


# ----------------------------------------------------------------------------------------------- helpers

def to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 sRGB -> float32 CIE Lab (L 0..100)."""
    return cv2.cvtColor(rgb_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def clean_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9_\-]+", "_", str(name).strip().lower()).strip("_")
    if not n:
        raise UsageError(f"region name {name!r} has no letters; use a word like hair or sky",
                         "easel region add hair --box 200,50,700,450")
    if n in RESERVED or re.fullmatch(r"m?\d+", n):
        raise UsageError(f"{name!r} is reserved (all, numbers and mark ids); choose a word like face or sky",
                         "easel region add face --marks 12,13")
    return n


def box_mask(H: int, W: int, box_px) -> np.ndarray:
    m = np.zeros((H, W), bool)
    x0, y0, x1, y1 = box_px
    m[y0:y1, x0:x1] = True
    return m


def _resize_mask(m: np.ndarray, W: int, H: int) -> np.ndarray:
    return cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5


def _downscale(img: np.ndarray, long_side: int):
    H, W = img.shape[:2]
    s = min(1.0, long_side / float(max(H, W)))
    if s >= 1.0:
        return img, 1.0
    w, h = max(1, int(round(W * s))), max(1, int(round(H * s)))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), s


# ----------------------------------------------------------------------------------------------- box + GrabCut

def refine_box(ref: np.ndarray, box_px, notes: Optional[list] = None) -> np.ndarray:
    """Grow / shrink a rough box to the object's silhouette with GrabCut seeded by the box."""
    H, W = ref.shape[:2]
    x0, y0, x1, y1 = box_px
    plain = box_mask(H, W, box_px)
    if (x1 - x0) * (y1 - y0) > 0.92 * W * H or (x1 - x0) < 8 or (y1 - y0) < 8:
        if notes is not None:
            notes.append("box too large or too small to refine; kept the plain box")
        return plain
    small, s = _downscale(ref, 520)
    h, w = small.shape[:2]
    rx0, ry0 = int(x0 * s), int(y0 * s)
    rx1, ry1 = max(rx0 + 2, int(round(x1 * s))), max(ry0 + 2, int(round(y1 * s)))
    rect = (rx0, ry0, max(2, min(w - rx0, rx1 - rx0)), max(2, min(h - ry0, ry1 - ry0)))
    gmask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), gmask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        if notes is not None:
            notes.append("refine failed; kept the plain box")
        return plain
    fg = (gmask == cv2.GC_FGD) | (gmask == cv2.GC_PR_FGD)
    fg = ndi.binary_fill_holes(fg)
    lab, n = ndi.label(fg)
    if n > 1:
        sizes = ndi.sum(fg, lab, range(1, n + 1))
        keep = np.flatnonzero(sizes >= 0.04 * sizes.max()) + 1
        fg = np.isin(lab, keep)
    full = _resize_mask(fg, W, H) & plain
    share = full.sum() / max(1, plain.sum())
    if share < 0.08 or share > 0.995:
        if notes is not None:
            notes.append(f"refine found {share:.0%} of the box as object; kept the plain box "
                         "(use --no-refine to skip, or --marks / --poly for a tighter shape)")
        return plain
    if notes is not None:
        msg = f"refined to the object's silhouette: {share:.0%} of the box"
        if share < 0.35:
            msg += ("; if the part fills most of the box this is wrong: use  --wand x,y  on the part (best for a "
                    "light part on a light ground) or  --no-refine")
        notes.append(msg)
    return full


# ----------------------------------------------------------------------------------------------- marks

def compute_marks(ref: np.ndarray, n: int = 40) -> Tuple[np.ndarray, List[dict]]:
    """Segment the reference into about ``n`` colour-coherent connected areas.

    Returns (labels int32 [H,W] numbered 1..k in reading order, table rows {n, box (units), colour, hex, share}).
    Deterministic: SLIC superpixels, then greedy merging of the most similar neighbours (small ones first).
    """
    from skimage.segmentation import slic
    H, W = ref.shape[:2]
    n = int(max(4, min(80, n)))
    small, s = _downscale(ref, 480)
    h, w = small.shape[:2]
    # Dark thin lines are boundaries, not areas: segment the colour areas between them (lines closed away for the
    # colours), then give each line pixel to the nearest area. Areas that touch only across a line are "weak"
    # neighbours and merge last, so same-coloured areas separated by line art (face / background) stay apart.
    from .metrics import line_map
    L = to_lab(small)[..., 0]
    raw = line_map(L, max(h, w))
    if raw.mean() > 0.08:            # dense "lines" are texture (foliage, stone), not line art
        lines = np.zeros_like(raw)
    else:
        lines = cv2.dilate(raw.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ker)
    lab = to_lab(closed)
    seg0 = slic(closed, n_segments=n * 9, compactness=12, start_label=1, channel_axis=-1,
                enforce_connectivity=True, mask=~lines)
    if lines.any():
        _, idx = ndi.distance_transform_edt(seg0 == 0, return_indices=True)
        seg = seg0[idx[0], idx[1]]
    else:
        seg = seg0
    k = int(seg.max()) + 1
    flat = seg.ravel()
    cnt = np.bincount(flat, minlength=k).astype(np.float64)
    sums = np.stack([np.bincount(flat, weights=lab[..., c].ravel(), minlength=k) for c in range(3)], 1)

    def pair_keys(lbl, skip_zero):
        keys = []
        for a, b in ((lbl[:, :-1], lbl[:, 1:]), (lbl[:-1, :], lbl[1:, :])):
            d = a != b
            if skip_zero:
                d &= (a > 0) & (b > 0)
            aa, bb = a[d].astype(np.int64), b[d].astype(np.int64)
            keys.append(np.minimum(aa, bb) * k + np.maximum(aa, bb))
        return set(np.unique(np.concatenate(keys)).tolist()) if keys else set()

    strong_keys = pair_keys(seg0, True)
    all_keys = pair_keys(seg, False) | strong_keys
    nbr: Dict[int, Dict[int, bool]] = {i: {} for i in range(k)}
    for key in all_keys:
        a, b = int(key // k), int(key % k)
        if a == b or a == 0 or b == 0:
            continue
        st = key in strong_keys
        nbr[a][b] = nbr[a].get(b, False) or st
        nbr[b][a] = nbr[b].get(a, False) or st
    parent = list(range(k))
    alive = set(i for i in range(1, k) if cnt[i] > 0)
    version = [0] * k
    total = float(h * w)
    min_area = total * 0.002

    def de_of(a, b):
        return float(np.sqrt((((sums[a] / cnt[a]) - (sums[b] / cnt[b])) ** 2).sum()))

    def cost(a, b):
        small_sz = min(cnt[a], cnt[b])
        de = de_of(a, b)
        if not nbr[a].get(b, False):
            de += 25.0                      # only touching across a line
        c = de * np.sqrt(small_sz / total) * 10.0
        if small_sz < min_area:
            c *= 0.05
        return c

    heap = []
    for a in range(k):
        for b in nbr[a]:
            if a < b:
                heapq.heappush(heap, (cost(a, b), a, b, 0, 0))
    count = len(alive)

    def must_merge(a, b):
        """Below the target count, still merge near-identical touching colours and specks."""
        if min(cnt[a], cnt[b]) < 2 * min_area:
            return True
        return nbr[a].get(b, False) and de_of(a, b) < 6.0

    while heap:
        c, a, b, va, vb = heapq.heappop(heap)
        if a not in alive or b not in alive or version[a] != va or version[b] != vb:
            continue
        if count <= n and not must_merge(a, b):
            continue
        # merge b into a
        parent[b] = a
        cnt[a] += cnt[b]
        sums[a] += sums[b]
        alive.discard(b)
        for x, st in nbr[b].items():
            if x in (a, b):
                continue
            nbr[a][x] = nbr[a].get(x, False) or st
            nbr[x].pop(b, None)
            nbr[x][a] = nbr[a][x]
        nbr[a].pop(b, None)
        nbr[b] = {}
        version[a] += 1
        count -= 1
        for x in nbr[a]:
            if x in alive:
                lo, hi = (a, x) if a < x else (x, a)
                heapq.heappush(heap, (cost(lo, hi), lo, hi, version[lo], version[hi]))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    root = np.array([find(i) for i in range(k)])
    seg2 = root[seg]
    # reading order: bands of 1/6 of the height by the centroid, then x
    uniq = np.unique(seg2)
    ys, xs = np.mgrid[0:h, 0:w]
    f2 = seg2.ravel()
    m = len(uniq)
    remap = np.zeros(k, np.int64)
    remap[uniq] = np.arange(m)
    f3 = remap[f2]
    area = np.bincount(f3, minlength=m)
    cy = np.bincount(f3, weights=ys.ravel(), minlength=m) / np.maximum(area, 1)
    cx = np.bincount(f3, weights=xs.ravel(), minlength=m) / np.maximum(area, 1)
    band = np.floor(cy / (h / 6.0))
    order = sorted(range(m), key=lambda i: (band[i], cx[i], cy[i]))
    number = np.zeros(m, np.int32)
    for rank, i in enumerate(order):
        number[i] = rank + 1
    lab_small = number[f3].reshape(h, w)
    labels = cv2.resize(lab_small.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    table = mark_table(ref, labels)
    return labels, table


def mark_table(ref: np.ndarray, labels: np.ndarray) -> List[dict]:
    H, W = labels.shape
    k = int(labels.max())
    flat = labels.ravel()
    area = np.bincount(flat, minlength=k + 1)
    rows = []
    objs = ndi.find_objects(labels)
    for i in range(1, k + 1):
        sl = objs[i - 1] if i - 1 < len(objs) else None
        if sl is None or area[i] == 0:
            continue
        sub = labels[sl] == i
        col = np.median(ref[sl][sub], axis=0)
        box = coords.px_box_to_units((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop), W, H)
        rows.append({"n": i, "box": list(box), "hex": coords.hex_colour(col), "colour": coords.colour_name(col),
                     "share": round(float(area[i]) / (H * W), 4)})
    return rows


def label_anchor(mask: np.ndarray) -> Tuple[int, int]:
    """The point deepest inside a mask (pole of inaccessibility approx.) for placing a label."""
    sl = coords.mask_bbox_px(mask)
    if sl is None:
        return 0, 0
    x0, y0, x1, y1 = sl
    sub = np.pad(mask[y0:y1, x0:x1].astype(np.uint8), 1)
    d = cv2.distanceTransform(sub, cv2.DIST_L2, 3)
    yy, xx = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(x0 + xx - 1), int(y0 + yy - 1)


# ----------------------------------------------------------------------------------------------- live-wire / polygon

def edge_cost(ref: np.ndarray) -> np.ndarray:
    """Cost map for live-wire: cheap on strong reference edges."""
    gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = to_lab(ref)
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt((gx ** 2 + gy ** 2).sum(axis=2))
    del gray
    g = cv2.GaussianBlur(g, (0, 0), 0.8)
    q = float(np.percentile(g, 97)) or 1.0
    gn = np.clip(g / q, 0.0, 1.0)
    # ridge term: the crest of an edge is cheaper than its flanks, so paths ride the exact edge
    crest = g / (cv2.dilate(g, np.ones((5, 5), np.uint8)) + 1e-3)
    return (0.03 + (1.0 - gn) ** 2 + 0.25 * (1.0 - crest)).astype(np.float32)


def snap_points(cost: np.ndarray, pts: np.ndarray, radius: Optional[float] = None) -> np.ndarray:
    """Move each waypoint to the cheapest (strongest-edge) pixel nearby; stays put where there is no edge."""
    H, W = cost.shape
    r = radius if radius is not None else max(4.0, 0.012 * max(H, W))
    ri = int(np.ceil(r))
    out = pts.copy()
    for i, (x, y) in enumerate(pts):
        xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
        x0, x1 = max(0, xi - ri), min(W, xi + ri + 1)
        y0, y1 = max(0, yi - ri), min(H, yi + ri + 1)
        sub = cost[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - x, yy - y)
        score = sub + 0.15 * (d / r) ** 2
        score[d > r] = np.inf
        k = np.unravel_index(int(np.argmin(score)), score.shape)
        if sub[k] < 0.5:            # a real edge is near
            out[i] = (xx[k] + 0.5, yy[k] + 0.5)
    return out


def livewire(cost: np.ndarray, pts_px: np.ndarray, corridor: Optional[float] = None) -> np.ndarray:
    """Snap a polyline (float px [N,2]) to cheap paths between consecutive waypoints. Returns [M,2] float px."""
    from skimage.graph import route_through_array
    H, W = cost.shape
    pts = snap_points(cost, np.asarray(pts_px, np.float64))
    if len(pts) < 2:
        return pts.astype(np.float32)
    out = [pts[0]]
    long_side = max(H, W)
    for a, b in zip(pts[:-1], pts[1:]):
        seg_len = float(np.hypot(*(b - a)))
        r = corridor if corridor is not None else max(6.0, 0.15 * seg_len + 0.01 * long_side)
        x0 = int(max(0, min(a[0], b[0]) - r))
        y0 = int(max(0, min(a[1], b[1]) - r))
        x1 = int(min(W, max(a[0], b[0]) + r + 1))
        y1 = int(min(H, max(a[1], b[1]) + r + 1))
        sub = cost[y0:y1, x0:x1].astype(np.float64)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        # distance to the straight segment keeps the path in a corridor
        d = b - a
        L2 = max(1e-6, float(d @ d))
        t = np.clip(((xx - a[0]) * d[0] + (yy - a[1]) * d[1]) / L2, 0, 1)
        dist = np.hypot(xx - (a[0] + t * d[0]), yy - (a[1] + t * d[1]))
        sub = sub + 0.6 * (dist / r) ** 2
        sub[dist > r] = 50.0
        sa = (int(np.clip(a[1], y0, y1 - 1)) - y0, int(np.clip(a[0], x0, x1 - 1)) - x0)
        sb = (int(np.clip(b[1], y0, y1 - 1)) - y0, int(np.clip(b[0], x0, x1 - 1)) - x0)
        path, _ = route_through_array(sub, sa, sb, fully_connected=True, geometric=True)
        p = np.array([(c + x0 + 0.5, rr + y0 + 0.5) for rr, c in path], np.float64)
        out.extend(p[1:])
    res = np.array(out, np.float32)
    return res


def polygon_mask(H: int, W: int, pts_px: np.ndarray) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts_px) * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
    return m > 0


def poly_region(ref: Optional[np.ndarray], H: int, W: int, pts_px: np.ndarray, snap: bool = True) -> np.ndarray:
    pts = np.asarray(pts_px, np.float32)
    if snap and ref is not None and len(pts) >= 3:
        closed = np.vstack([pts, pts[:1]])
        pts = livewire(edge_cost(ref), closed)
    return polygon_mask(H, W, pts)


# ----------------------------------------------------------------------------------------------- wand

def wand(img: np.ndarray, pt_px, tol: float = 12.0, contiguous: bool = True) -> np.ndarray:
    """Connected area of colour within ``tol`` (CIE Lab Delta E) of the colour at ``pt_px``."""
    H, W = img.shape[:2]
    x = int(np.clip(pt_px[0], 0, W - 1))
    y = int(np.clip(pt_px[1], 0, H - 1))
    lab = to_lab(cv2.GaussianBlur(img, (0, 0), 1.0))
    y0, y1, x0, x1 = max(0, y - 1), min(H, y + 2), max(0, x - 1), min(W, x + 2)
    seed = np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    de = np.sqrt(((lab - seed) ** 2).sum(axis=2))
    sim = de <= float(tol)
    if not contiguous:
        return sim
    lab_cc, _ = ndi.label(sim, structure=np.ones((3, 3), bool))
    k = lab_cc[y, x]
    if k == 0:
        m = np.zeros((H, W), bool)
        m[y, x] = True
        return m
    m = lab_cc == k
    # close pin-holes (anti-aliasing specks), not real holes
    holes = ndi.binary_fill_holes(m) & ~m
    hl, hn = ndi.label(holes)
    if hn:
        sizes = ndi.sum(holes, hl, range(1, hn + 1))
        small = np.flatnonzero(sizes < max(4, 0.0003 * H * W)) + 1
        m |= np.isin(hl, small)
    return m


def grow_into_lines(ref: np.ndarray, mask: np.ndarray, frac: float = 0.012) -> np.ndarray:
    """Extend a colour area over the dark line art that bounds it (a face takes its outline), up to ~1% of the
    canvas side, so the line stage can draw the outline inside the region."""
    from .metrics import line_map
    H, W = mask.shape
    r = max(2, int(round(frac * max(H, W))))
    lines = line_map(to_lab(ref)[..., 0], max(H, W))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near = cv2.dilate(mask.astype(np.uint8), ker) > 0
    return mask | (near & lines)


# ----------------------------------------------------------------------------------------------- name lookup

def norm_key(name: str) -> str:
    return re.sub(r"[\s_\-]+", "_", str(name).strip().lower())


def match_name(spec: str, names: Sequence[str]) -> Optional[str]:
    k = norm_key(spec)
    for n in names:
        if norm_key(n) == k:
            return n
    # plural / singular
    for n in names:
        nk = norm_key(n)
        if nk.rstrip("s") == k.rstrip("s"):
            return n
    return None


def suggest(spec: str, names: Sequence[str]) -> Optional[str]:
    import difflib
    c = difflib.get_close_matches(norm_key(spec), [norm_key(n) for n in names], n=1, cutoff=0.5)
    if not c:
        return None
    for n in names:
        if norm_key(n) == c[0]:
            return n
    return None


MARK_RE = re.compile(r"^\s*(?:m|mark|#)\s*(\d+)\s*$", re.I)
