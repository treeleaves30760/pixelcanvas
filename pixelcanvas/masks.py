# -*- coding: utf-8 -*-
"""Masks: arbitrary shapes via Pillow, geometric masks, text, soft edges. A mask = float32 (H, W), 0~1."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .color import F32, grid, smoothstep
from .filters import gaussian_blur

_RES = getattr(Image, "Resampling", Image)

__all__ = ["shape_mask", "polygon_mask", "circle_mask", "rect_mask", "band_mask", "line_mask", "bezier_points", "feather",
           "ellipse_mask", "shift", "drop_shadow"]


def shape_mask(h, w, draw_fn, aa=3):
    """Draw an arbitrary shape with Pillow ImageDraw into an anti-aliased mask.

    ``draw_fn(draw, s)``: draw on ``draw``; multiply all coordinates by s (the supersampling factor).
    Example: ``shape_mask(H, W, lambda d, s: d.ellipse([(10*s, 10*s), (50*s, 40*s)], fill=255))``
    """
    img = Image.new("L", (w * aa, h * aa), 0)
    draw_fn(ImageDraw.Draw(img), float(aa))
    if aa > 1:
        img = img.resize((w, h), _RES.BOX)
    return np.asarray(img, F32) / 255.0


def polygon_mask(h, w, points, aa=3):
    """Polygon mask. ``points = [(x, y), ...]``."""
    return shape_mask(h, w, lambda d, s: d.polygon([(x * s, y * s) for x, y in points], fill=255), aa)


def circle_mask(h, w, cx, cy, r, feather=1.0):
    """Circle mask; ``feather`` is the edge softening width in pixels."""
    X, Y = grid(h, w)
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    f = max(float(feather), 0.5)
    return (1.0 - smoothstep(r - f, r + f, d)).astype(F32)


def ellipse_mask(h, w, cx, cy, rx, ry, angle=0.0):
    """Anti-aliased ellipse mask: centre (cx, cy), half-axes rx (horizontal) and ry (vertical) in px, ``angle`` in
    degrees (clockwise on screen)."""
    from .sdf import sd_ellipse, fill
    X, Y = grid(h, w)
    return fill(sd_ellipse(X, Y, cx, cy, rx, ry, angle))


def rect_mask(h, w, x0, y0, x1, y1, feather=0.0):
    """Rectangle mask (optionally soft-edged)."""
    X, Y = grid(h, w)
    f = max(float(feather), 0.5)
    m = smoothstep(x0 - f, x0 + f, X) * (1 - smoothstep(x1 - f, x1 + f, X))
    m *= smoothstep(y0 - f, y0 + f, Y) * (1 - smoothstep(y1 - f, y1 + f, Y))
    return m.astype(F32)


def band_mask(h, w, y0, y1, soft=0.0):
    """Horizontal band mask: 1 between y0 and y1, fading over ``soft`` pixels at both ends."""
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
    """Soften mask edges (Gaussian blur)."""
    return gaussian_blur(mask, sigma)


def shift(mask, dx, dy):
    """Move a mask by (dx, dy) px (fractions allowed) and fill the uncovered border with zero (no wrap-around).
    Positive dx moves right, positive dy moves down."""
    m = np.asarray(mask, F32)
    h, w = m.shape[:2]
    out = np.zeros_like(m)
    ix, iy = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = float(dx) - ix, float(dy) - iy
    for oy, wy in ((iy, 1 - fy), (iy + 1, fy)):
        for ox, wx in ((ix, 1 - fx), (ix + 1, fx)):
            wt = wx * wy
            if wt <= 0 or abs(ox) >= w or abs(oy) >= h:
                continue
            out[max(oy, 0):h + min(oy, 0), max(ox, 0):w + min(ox, 0)] += wt * m[max(-oy, 0):h - max(oy, 0),
                                                                                  max(-ox, 0):w - max(ox, 0)]
    return out.astype(F32)


def drop_shadow(mask, offset=(6, 10), blur=8.0):
    """Soft shadow mask of ``mask``: moved by ``offset`` = (dx, dy) px and blurred by ``blur`` px (sigma).
    Paint it under the subject: ``cv.paint('#000000', drop_shadow(m, (6, 10), 8), opacity=0.4)``."""
    dx, dy = offset
    sh = shift(mask, dx, dy)
    return gaussian_blur(sh, blur) if blur and blur > 0 else sh


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "rect": "use masks.rect_mask(h, w, x0, y0, x1, y1, feather=0)",
    "rectangle": "use masks.rect_mask(h, w, x0, y0, x1, y1, feather=0)",
    "box": "use masks.rect_mask(h, w, x0, y0, x1, y1, feather=0)",
    "circle": "use masks.circle_mask(h, w, cx, cy, r, feather=1)",
    "disc": "use masks.circle_mask(h, w, cx, cy, r, feather=1)",
    "ellipse": "use masks.ellipse_mask(h, w, cx, cy, rx, ry, angle=0)",
    "polygon": "use masks.polygon_mask(h, w, points)",
    "triangle": "use masks.polygon_mask(h, w, [(x1, y1), (x2, y2), (x3, y3)])",
    "line": "use masks.line_mask(h, w, points, width=2)",
    "polyline": "use masks.line_mask(h, w, points, width=2)",
    "path": "use masks.line_mask(h, w, points, width=2); curve points come from masks.bezier_points(p0, p1, p2, ...)",
    "bezier": "use masks.bezier_points(p0, p1, p2, ..., n=64) for the points, then masks.line_mask(h, w, points)",
    "curve": "use masks.bezier_points(p0, p1, p2, ..., n=64) for the points, then masks.line_mask(h, w, points)",
    "band": "use masks.band_mask(h, w, y0, y1, soft=0)",
    "stripe": "use masks.band_mask(h, w, y0, y1, soft=0)",
    "roll": "use masks.shift(mask, dx, dy) (zero fill, no wrap-around)",
    "move": "use masks.shift(mask, dx, dy)",
    "translate": "use masks.shift(mask, dx, dy)",
    "shadow": "use masks.drop_shadow(mask, (dx, dy), blur)",
    "blur": "use masks.feather(mask, sigma)",
    "soften": "use masks.feather(mask, sigma)",
    "union": "masks are float arrays: np.maximum(a, b); intersection np.minimum(a, b); inverse 1 - a",
    "invert": "masks are float arrays 0..1: 1 - mask",
    "draw": "use masks.shape_mask(h, w, draw_fn, aa=3) where draw_fn receives a Pillow ImageDraw",
}
__getattr__ = _module_getattr(__name__, _HINTS)
