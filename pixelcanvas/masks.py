# -*- coding: utf-8 -*-
"""遮罩：用 Pillow 畫任意形狀、幾何遮罩、文字、柔邊。遮罩 = float32 (H, W)，0~1。"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .color import F32, grid, smoothstep
from .filters import gaussian_blur

_RES = getattr(Image, "Resampling", Image)

__all__ = ["shape_mask", "polygon_mask", "circle_mask", "rect_mask", "band_mask", "line_mask", "bezier_points", "feather"]


def shape_mask(h, w, draw_fn, aa=3):
    """用 Pillow ImageDraw 畫任意形狀成抗鋸齒遮罩。

    ``draw_fn(draw, s)``：在 draw 上畫圖，所有座標請乘上 s（超取樣倍率）。
    例：``shape_mask(H, W, lambda d, s: d.ellipse([(10*s, 10*s), (50*s, 40*s)], fill=255))``
    """
    img = Image.new("L", (w * aa, h * aa), 0)
    draw_fn(ImageDraw.Draw(img), float(aa))
    if aa > 1:
        img = img.resize((w, h), _RES.BOX)
    return np.asarray(img, F32) / 255.0


def polygon_mask(h, w, points, aa=3):
    """多邊形遮罩。``points = [(x, y), ...]``。"""
    return shape_mask(h, w, lambda d, s: d.polygon([(x * s, y * s) for x, y in points], fill=255), aa)


def circle_mask(h, w, cx, cy, r, feather=1.0):
    """圓形遮罩，``feather`` 為邊緣柔化寬度（像素）。"""
    X, Y = grid(h, w)
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    f = max(float(feather), 0.5)
    return (1.0 - smoothstep(r - f, r + f, d)).astype(F32)


def rect_mask(h, w, x0, y0, x1, y1, feather=0.0):
    """矩形遮罩（可柔邊）。"""
    X, Y = grid(h, w)
    f = max(float(feather), 0.5)
    m = smoothstep(x0 - f, x0 + f, X) * (1 - smoothstep(x1 - f, x1 + f, X))
    m *= smoothstep(y0 - f, y0 + f, Y) * (1 - smoothstep(y1 - f, y1 + f, Y))
    return m.astype(F32)


def band_mask(h, w, y0, y1, soft=0.0):
    """水平帶遮罩：y0~y1 之間為 1，兩端以 soft 像素漸變。"""
    _, Y = grid(h, w)
    s = max(float(soft), 0.5)
    return (smoothstep(y0 - s, y0 + s, Y) * (1 - smoothstep(y1 - s, y1 + s, Y))).astype(F32)


def line_mask(h, w, points, width=2.0, aa=3, closed=False):
    """Anti-aliased polyline / curve stroke of constant ``width`` px with round joints and caps.
    ``points = [(x, y), ...]`` (e.g. from ``bezier_points``)."""
    pts = [(float(x), float(y)) for x, y in points]
    if closed:
        pts = pts + pts[:1]

    def _draw(d, s):
        sp = [(x * s, y * s) for x, y in pts]
        wd = max(1, int(round(width * s)))
        d.line(sp, fill=255, width=wd, joint="curve")
        r = width * s / 2
        for x, y in (sp if len(sp) < 3 else (sp[0], sp[-1])):      # round caps
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return shape_mask(h, w, _draw, aa)


def bezier_points(*ctrl, n=64):
    """Sample a quadratic (3 control points) or cubic (4) Bézier curve → list of n (x, y).
    Pass more points for a higher-order curve. Use with ``line_mask`` / ``polygon_mask`` / ``stroke``."""
    P = np.asarray(ctrl, np.float64)                          # (k, 2)
    t = np.linspace(0.0, 1.0, int(n))[:, None, None]          # (n, 1, 1)
    P = np.broadcast_to(P, (len(t),) + P.shape)
    while P.shape[1] > 1:                                     # de Casteljau
        P = P[:, :-1] * (1 - t) + P[:, 1:] * t
    return [(float(x), float(y)) for x, y in P[:, 0]]


def feather(mask, sigma):
    """柔化遮罩邊緣（高斯模糊）。"""
    return gaussian_blur(mask, sigma)
