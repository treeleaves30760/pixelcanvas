"""Flats: one clean base colour per area, filled under the line art (no gaps or halos at the lines)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .ops import Fill, is_partial, mask_to_rle, owned_stack
from .trace import (as_mask, cluster_colors, line_map, merge_small_regions, region_colors, segment_shapes,
                    segments, to_lab)

__all__ = ["flat_ops", "flat_segments"]


def _auto_k(ref, m, seed, l_weight, target=8.0, kmax=24):
    lab = to_lab(ref)[m]
    rng = np.random.default_rng(seed)
    X = lab if len(lab) <= 20000 else lab[rng.choice(len(lab), 20000, replace=False)]
    X = X.copy()
    X[:, 0] *= l_weight
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    for k in [v for v in (3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64) if v < kmax] + [kmax]:
        if k >= len(X):
            return max(1, len(X))
        cv2.setRNGSeed(int(seed))
        _, lab_i, c = cv2.kmeans(np.ascontiguousarray(X, np.float32), k, None, crit, 1, cv2.KMEANS_PP_CENTERS)
        d = np.sqrt(((X - c[lab_i[:, 0]]) ** 2).sum(1))
        li = lab_i[:, 0]
        cnt = np.bincount(li, minlength=k)
        mean_d = np.bincount(li, d, minlength=k) / np.maximum(cnt, 1)
        # every cluster that matters must be tight (a dominant background must not hide loose ones)
        if np.all(mean_d[cnt >= 0.004 * len(X)] < target):
            return k
    return kmax


def flat_segments(ref, mask=None, *, colors="auto", line_mask=None, seed=0, min_area=None, merge_unlined=True,
                  merge_de=22.0):
    """Partition the masked picture into flat areas. Returns (seg int32 [H,W] (-1 outside), colours uint8 [n,3])."""
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    if line_mask is None:
        line_mask = line_map(ref, mask=m).mask
    line_mask = np.asarray(line_mask) > 0.5 if line_mask.dtype != bool else line_mask
    inner = m & ~line_mask
    if inner.sum() < 16:
        inner = m.copy()
    # smooth away texture and antialiasing before clustering, keep edges
    sm = cv2.bilateralFilter(ref, 7, 30, 5)
    lw = 0.6
    k = _auto_k(sm, inner, seed, lw) if colors in (None, "auto") else int(colors)
    labels, _ = cluster_colors(sm, k, inner, l_weight=lw, seed=seed)
    # mode filter on labels removes antialias fringes between areas
    seg = segments(labels)
    if min_area is None:
        min_area = max(24, int(H * W * 0.00012))
    seg = merge_small_regions(seg, min_area)
    # extend every area under the line art (and into tiny gaps) by nearest-area assignment
    have = seg >= 0
    if (m & ~have).any() and have.any():
        _, (iy, ix) = ndi.distance_transform_edt(~have, return_indices=True)
        fillin = m & ~have
        seg[fillin] = seg[iy[fillin], ix[fillin]]
    seg[~m] = -1
    cols = region_colors(np.where(inner[..., None], ref, 0).astype(np.uint8), np.where(inner, seg, -1), "mode")
    # areas that lie entirely under lines (no inner pixel) take their reference median
    n = int(seg.max()) + 1
    cnt_inner = np.bincount(seg[inner & (seg >= 0)], minlength=n)
    if (cnt_inner == 0).any():
        alt = region_colors(ref, seg, "median")
        cols[cnt_inner == 0] = alt[cnt_inner == 0]
    if line_mask.any() and merge_unlined:
        seg, cols = _merge_unlined(seg, cols, line_mask, max_de=merge_de)
        # the merged area's base colour is its most common colour
        cols2 = region_colors(np.where(inner[..., None], ref, 0).astype(np.uint8), np.where(inner, seg, -1), "mode")
        n2 = int(seg.max()) + 1
        has = np.bincount(seg[inner & (seg >= 0)], minlength=n2) > 0
        cols[has] = cols2[has]
    return seg, cols


def _merge_unlined(seg, cols, line_mask, max_de=22.0, line_share=0.35):
    """Merge neighbouring areas that no line separates and whose colours are related: in cel work those are
    shading inside one flat (blush, soft shadow), which the shade stage paints, not separate flats."""
    n = int(seg.max()) + 1
    if n <= 1:
        return seg, cols
    near_line = cv2.dilate(line_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pa, pb, on = [], [], []
    for a, b, ln in ((seg[:, :-1], seg[:, 1:], near_line[:, :-1] | near_line[:, 1:]),
                     (seg[:-1, :], seg[1:, :], near_line[:-1, :] | near_line[1:, :])):
        d = (a != b) & (a >= 0) & (b >= 0)
        pa.append(np.minimum(a[d], b[d]))
        pb.append(np.maximum(a[d], b[d]))
        on.append(ln[d])
    pa, pb, on = np.concatenate(pa), np.concatenate(pb), np.concatenate(on)
    if len(pa) == 0:
        return seg, cols
    key = pa.astype(np.int64) * n + pb
    uk, inv, tot = np.unique(key, return_inverse=True, return_counts=True)
    online = np.bincount(inv, on.astype(np.float64))
    share = online / tot
    lab = to_lab(cols[None].astype(np.uint8))[0]
    areas = np.bincount(seg[seg >= 0], minlength=n)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    a_ids, b_ids = uk // n, uk % n
    de = np.sqrt(((lab[a_ids] - lab[b_ids]) ** 2).sum(1))
    cand = np.flatnonzero((share < line_share) & (tot >= 6) & (de < max_de))
    rep_lab = lab.copy()
    for e in cand[np.argsort(de[cand])]:
        ra, rb = find(int(a_ids[e])), find(int(b_ids[e]))
        if ra == rb:
            continue
        if np.sqrt(((rep_lab[ra] - rep_lab[rb]) ** 2).sum()) >= max_de:
            continue
        big, small = (ra, rb) if areas[ra] >= areas[rb] else (rb, ra)
        parent[small] = big
        areas[big] += areas[small]
    roots = np.array([find(i) for i in range(n)])
    uniq, newid = np.unique(roots, return_inverse=True)
    out = np.where(seg >= 0, newid[np.maximum(seg, 0)], -1).astype(np.int32)
    return out, cols[uniq]


def flat_ops(ref, mask=None, *, colors="auto", line_mask=None, seed=0, holes=False, skip_dark=None):
    """Flat base colours inside ``mask``: one smooth-outlined Fill per area, largest first; each large area reaches
    half a pixel under its neighbours so no seams show.  ``colors``: "auto" or the number of colours.
    ``line_mask``: the line art (bool or float [H,W]); areas extend under it.  ``skip_dark``: Lab lightness below
    which an area is left to the ink (an ink drawing's solid blacks are drawn by the line stage; a flat under them
    would only cover the white details they hold).
    On a region: the whole picture's areas anchored in the region, each one complete, plus the smaller areas that
    sit on them (so the order of region calls does not matter)."""
    if is_partial(mask):
        allops = flat_ops(ref, None, colors=colors, line_mask=line_mask, seed=seed, holes=holes, skip_dark=skip_dark)
        return owned_stack(allops, mask, np.asarray(ref).shape)
    ref = np.asarray(ref)
    seg, cols = flat_segments(ref, mask, colors=colors, line_mask=line_mask, seed=seed)
    if seg.size == 0 or seg.max() < 0:
        return []
    # stacked largest first, each filled across the areas inside it (no seams); ``holes`` leaves holes instead
    shapes = segment_shapes(seg, ref, gradients=False, holes=holes)
    shapes.sort(key=lambda d: -d["area"])
    if skip_dark is not None:
        lab = to_lab(cols[None].astype(np.uint8))[0]
        shapes = [d for d in shapes if lab[d["id"], 0] >= skip_dark]
    return [Fill(polygons=d["rings"], mask_rle=None, color=tuple(int(v) for v in cols[d["id"]])) for d in shapes]
