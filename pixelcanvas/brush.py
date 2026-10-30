# -*- coding: utf-8 -*-
"""筆刷：在遮罩上噴柔邊圓點、沿路徑畫筆觸。（先畫到遮罩，再用 cv.paint(color, mask) 上色）"""
from __future__ import annotations

import numpy as np

from .color import F32, smoothstep

__all__ = ["stamp", "stroke"]


def stamp(mask, cx, cy, r, hardness=0.5, opacity=1.0):
    """在遮罩上蓋一個柔邊圓點（就地修改；像 alpha over 疊加，不會超過 1）。

    ``hardness`` 0~1：1 = 硬邊；``opacity`` 單次不透明度。
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
    """沿折線 ``points`` 噴點成筆觸（就地修改 mask）。

    ``spacing`` 以半徑倍數計；``r_end`` 讓筆畫由 r 漸變到 r_end；``jitter`` 位置抖動（半徑倍數）。
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
