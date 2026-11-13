# -*- coding: utf-8 -*-
"""Brush: stamp soft round dabs on a mask and stroke along paths. (Paint to a mask first, then colour it with cv.paint(color, mask).)"""
from __future__ import annotations

import numpy as np

from .color import F32, smoothstep

__all__ = ["stamp", "stroke"]


def stamp(mask, cx, cy, r, hardness=0.5, opacity=1.0):
    """Stamp one soft round dab on the mask (in place; composited like alpha-over, never exceeds 1).

    ``hardness`` 0~1: 1 = hard edge; ``opacity`` is the opacity of a single dab.
    """
    h, w = mask.shape
    x0, x1 = max(int(cx - r - 1), 0), min(int(cx + r + 2), w)
    y0, y1 = max(int(cy - r - 1), 0), min(int(cy + r + 2), h)
    if x0 >= x1 or y0 >= y1:
        return mask
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(float(r), 1e-6)
    if hardness >= 1.0:
        dot = (d <= 1.0).astype(F32)
    else:
        dot = 1.0 - smoothstep(hardness, 1.0, d)
    sub = mask[y0:y1, x0:x1]
    sub += opacity * dot * (1.0 - sub)
    return mask


def stroke(mask, points, r, hardness=0.5, opacity=1.0, spacing=0.3, r_end=None, jitter=0.0, seed=0):
    """Stroke along the polyline ``points`` with dabs (modifies mask in place).

    ``spacing`` is in radius multiples; ``r_end`` tapers the stroke from r to r_end; ``jitter`` is position jitter (radius multiples).
    """
    pts = np.asarray(points, F32)
    if len(pts) == 1:
        return stamp(mask, pts[0, 0], pts[0, 1], r, hardness, opacity)
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    L = float(seglen.sum())
    if L <= 1e-6:
        return stamp(mask, pts[0, 0], pts[0, 1], r, hardness, opacity)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    step = max(spacing * r, 0.5)
    n = int(L / step) + 1
    rng = np.random.default_rng(seed)
    for k in range(n + 1):
        d = min(k * step, L)
        i = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
        f = (d - cum[i]) / max(seglen[i], 1e-6)
        x, y = pts[i] + seg[i] * f
        tt = d / L
        rr = r if r_end is None else r + (r_end - r) * tt
        if jitter:
            x += rng.normal(0, jitter * rr)
            y += rng.normal(0, jitter * rr)
        stamp(mask, x, y, rr, hardness, opacity)
    return mask


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "brush": "use brush.stamp(mask, cx, cy, r, hardness=0.5, opacity=1) for one dab, brush.stroke(mask, points, r) along a path",
    "dab": "use brush.stamp(mask, cx, cy, r, hardness=0.5, opacity=1)",
    "paint": "use brush.stamp(mask, cx, cy, r) for a dab or brush.stroke(mask, points, r) for a path; then cv.paint(color, mask)",
    "draw": "use brush.stroke(mask, points, r, hardness=0.5, spacing=0.3)",
    "line": "use brush.stroke(mask, points, r, hardness=0.5, spacing=0.3)",
    "path": "use brush.stroke(mask, points, r, hardness=0.5, spacing=0.3)",
    "spray": "use brush.stroke(mask, points, r, jitter=0.5) for a scattered stroke",
    "pen": "use brush.stroke(mask, points, r, hardness=1.0)",
    "dot": "use brush.stamp(mask, cx, cy, r, hardness=1.0)",
}
__getattr__ = _module_getattr(__name__, _HINTS)
