"""Loose pencil construction: silhouettes and the main inner divisions, drawn as light, slightly wandering pencil
lines with overshoot and the occasional second pass, the way a sketch is laid in before inking."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .ops import is_partial, owned_ops
from .ops import Stroke, stroke_length
from .trace import as_mask, cluster_colors, merge_small_regions, segments, skeleton_polylines, smooth_polyline

__all__ = ["sketch_ops"]

GRAPHITE = (70, 70, 82)


def _rdp(pts, eps):
    """Ramer-Douglas-Peucker simplification."""
    if len(pts) < 3:
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
        n = np.hypot(*ab)
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


def _catmull(pts, n_per=6):
    if len(pts) < 3:
        return pts
    P = np.vstack([pts[:1], pts, pts[-1:]])
    out = []
    t = np.linspace(0, 1, n_per, endpoint=False)[:, None]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2:-1])
    return np.vstack(out)


def sketch_ops(ref, mask=None, *, looseness=0.5, seed=0):
    """Pencil construction lines for the area: the borders of its few big colour / value masses, simplified and
    drawn loosely (``looseness`` 0..1: 0 = tight, 1 = gestural with more wobble and overshoot).
    On a region: the whole picture's sketch lines anchored in the region (each line complete)."""
    if is_partial(mask):
        return owned_ops(sketch_ops(ref, None, looseness=looseness, seed=seed), mask)
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    rng = np.random.default_rng(seed)
    looseness = float(np.clip(looseness, 0, 1))
    # work at a reduced size: a sketch only sees the big shapes
    sc = min(1.0, 420.0 / max(H, W))
    small = cv2.resize(ref, (max(1, int(W * sc)), max(1, int(H * sc))), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(m.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    sm = cv2.bilateralFilter(cv2.GaussianBlur(small, (0, 0), 1.2), 9, 40, 7)
    lab, _ = cluster_colors(sm, 7, msmall, l_weight=1.4, seed=seed)
    seg = merge_small_regions(segments(lab), max(40, int(small.shape[0] * small.shape[1] * 0.002)))
    # region borders, plus the outline of the masked area itself
    seg_m = np.where(msmall, seg, -2)
    b = np.zeros(seg.shape, bool)
    b[:, 1:] |= seg_m[:, 1:] != seg_m[:, :-1]
    b[1:, :] |= seg_m[1:, :] != seg_m[:-1, :]
    b &= ndi.binary_dilation(msmall, iterations=1)
    if (~msmall).sum() == 0:
        b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = False
    polys = skeleton_polylines(b, min_len=6.0)
    # construction first: the axis of each big mass, then an ellipse (round masses) or a proportion box (angular
    # ones) around it, light and solid, before the loose outlines
    ops = _construction_ops(seg_m, sc, rng, seed, looseness)
    eps = 1.0 + 2.0 * looseness
    for i, (pts, _, _) in enumerate(polys):
        p = pts.astype(np.float64) / sc
        L = np.sqrt((np.diff(p, axis=0) ** 2).sum(1)).sum() if len(p) > 1 else 0
        if L < 18:
            continue
        p, _ = smooth_polyline(p, None, window=7)
        p = _rdp(p.astype(np.float64), eps / sc ** 0.5)
        passes = 1 + int(rng.random() < 0.25 + 0.35 * looseness)
        for k in range(passes):
            q = p.copy()
            amp = (0.6 + 2.6 * looseness) * (1.0 + 0.6 * k)
            q += rng.normal(0, amp, q.shape) * 0.6
            # overshoot past both ends
            if len(q) >= 2:
                over = (2 + 10 * looseness) * rng.random(2)
                d0 = q[0] - q[1]
                d1 = q[-1] - q[-2]
                q[0] += d0 / (np.hypot(*d0) + 1e-9) * over[0]
                q[-1] += d1 / (np.hypot(*d1) + 1e-9) * over[1]
            q = _catmull(q, 5)
            # long contours are drawn as several overlapping strokes, as by hand
            seg_len = 120 + 140 * looseness
            s = np.concatenate([[0], np.cumsum(np.sqrt((np.diff(q, axis=0) ** 2).sum(1)))])
            start = 0.0
            while start < s[-1] - 4:
                end = min(s[-1], start + seg_len * (0.7 + 0.6 * rng.random()))
                sel = (s >= start) & (s <= end)
                if sel.sum() >= 2:
                    w = 1.1 + 0.7 * rng.random() + 0.4 * looseness
                    ops.append(Stroke(q[sel].astype(np.float32), float(w), GRAPHITE,
                                      opacity=float(0.6 + 0.3 * rng.random()) / (1 + 0.3 * k),
                                      brush="pencil", seed=seed * 104729 + len(ops)))
                start = end - 6.0 * rng.random()
    # every stroke is a real pencil line (length >= 4 x width); construction first, then big shapes first
    ops = [o for o in ops if stroke_length(o) >= 4.0 * float(np.mean(o.width))]
    n_con = sum(1 for o in ops if o.seed < 0)
    con, rest = ops[:n_con], ops[n_con:]
    rest.sort(key=lambda o: -len(o.pts))
    for o in con:
        o.seed = -o.seed
    return con + rest


def _construction_ops(seg, sc, rng, seed, looseness, max_masses=8):
    """Axis lines, ellipses and proportion boxes of the biggest masses (strokes flagged by a negative seed until
    the caller orders them first)."""
    ids, counts = np.unique(seg[seg >= 0], return_counts=True)
    if not len(ids):
        return []
    total = seg.size
    order = np.argsort(-counts)
    ops = []
    k = 0
    for j in order[:max_masses]:
        if counts[j] < 0.02 * total:
            break
        mk = (seg == ids[j]).astype(np.uint8)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        if len(c) < 6 or cv2.contourArea(c) < 0.01 * total:
            continue
        ys, xs = np.nonzero(mk)
        if (xs.min() == 0) + (ys.min() == 0) + (xs.max() == seg.shape[1] - 1) + (ys.max() == seg.shape[0] - 1) >= 2:
            continue  # background / ground running off the canvas: nothing to construct
        P = np.stack([xs, ys], 1).astype(np.float64) + 0.5
        cen = P.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((P - cen).T))
        d = evecs[:, -1]
        t = (P - cen) @ d
        t0, t1 = np.percentile(t, 1), np.percentile(t, 99)
        lines = []
        # the axis (gesture) line, drawn a little past the mass
        over = 0.08 * (t1 - t0)
        lines.append(np.array([cen + d * (t0 - over), cen + d * (t1 + over)]))
        (ex, ey), (ea, eb), ang = cv2.fitEllipse(c)
        fill = cv2.contourArea(c) / max(np.pi * ea * eb / 4.0, 1e-6)
        if 0.82 <= fill <= 1.12:
            tt = np.linspace(0, 2 * np.pi, 48)
            ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            ex_ = ea / 2 * np.cos(tt)
            ey_ = eb / 2 * np.sin(tt)
            lines.append(np.stack([ex + ex_ * ca - ey_ * sa, ey + ex_ * sa + ey_ * ca], 1))
        else:
            box = cv2.boxPoints(cv2.minAreaRect(c))
            for i in range(4):
                lines.append(np.array([box[i], box[(i + 1) % 4]], np.float64))
        for q in lines:
            q = np.asarray(q, np.float64) / sc
            if len(q) == 2:
                q = np.linspace(q[0], q[1], max(2, int(np.hypot(*(q[1] - q[0])) / 6)))
            q = q + rng.normal(0, 0.4 + 1.2 * looseness, q.shape) * 0.5
            k += 1
            ops.append(Stroke(q.astype(np.float32), float(1.0 + 0.3 * rng.random()), GRAPHITE,
                              opacity=float(0.3 + 0.15 * rng.random()), brush="pencil",
                              seed=-(seed * 7919 + k + 1)))
    return ops
