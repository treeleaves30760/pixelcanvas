# -*- coding: utf-8 -*-
"""Signed distance fields (SDF): crisp vector shapes, booleans, outlines, glows and bevels.

An SDF gives, for every pixel, the signed distance (in px) to a shape's edge: < 0 inside, 0 on the edge, > 0 outside.
That one array gives you everything a designer needs:

* ``fill(d)``           → anti-aliased coverage mask (exactly 1 px soft edge at any resolution)
* ``outline(d, w)``     → a stroke of width ``w`` centred on the edge
* ``glow(d, r)``        → soft halo outside the shape
* ``offset(d, k)``      → k > 0 GROWS the shape by k px, k < 0 SHRINKS it (a wall ring = ``subtract(offset(d, +wall), d)``)
* ``union / intersect / subtract / smooth_union`` → combine shapes like a vector editor
* ``bevel_light(d, ...)`` → shade as if the shape were embossed (buttons, badges, coins, logos)

All functions take pixel coordinate grids ``X, Y`` (``cv.xy()``) and return float32 (H, W).
Good for icons, logos, UI, badges, geometric posters, flat illustration, and any shape you can describe precisely.
"""
from __future__ import annotations

import numpy as np

from .color import F32

__all__ = ["sd_circle", "sd_ellipse", "sd_box", "sd_segment", "sd_polygon", "sd_star", "sd_ring", "sd_arc",
           "union", "intersect", "subtract", "smooth_union", "offset", "annular",
           "fill", "outline", "glow", "bevel_light"]


def _rot(X, Y, cx, cy, angle):
    """Translate to (cx, cy) and rotate by -angle (degrees) → local coordinates."""
    x, y = X - cx, Y - cy
    if angle:
        a = np.deg2rad(angle)
        c, s = np.cos(a), np.sin(a)
        x, y = c * x + s * y, -s * x + c * y
    return x, y


# ------------------------------------------------------------------ primitives

def sd_circle(X, Y, cx, cy, r):
    return (np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r).astype(F32)


def sd_ellipse(X, Y, cx, cy, rx, ry, angle=0.0):
    """Approximate ellipse distance (accurate near the edge, which is what matters for fill/outline)."""
    x, y = _rot(X, Y, cx, cy, angle)
    k0 = np.sqrt((x / rx) ** 2 + (y / ry) ** 2)
    k1 = np.sqrt((x / rx ** 2) ** 2 + (y / ry ** 2) ** 2)
    d = k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)
    return np.where(k1 < 1e-12, -min(rx, ry), d).astype(F32)               # exactly at the centre: 0/0


def sd_box(X, Y, cx, cy, half_w, half_h, radius=0.0, angle=0.0):
    """Rectangle centred at (cx, cy) with half sizes; ``radius`` rounds the corners; ``angle`` in degrees."""
    x, y = _rot(X, Y, cx, cy, angle)
    qx = np.abs(x) - half_w + radius
    qy = np.abs(y) - half_h + radius
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2)
    return (out + np.minimum(np.maximum(qx, qy), 0) - radius).astype(F32)


def sd_segment(X, Y, ax, ay, bx, by, r=0.0):
    """Capsule: segment A→B thickened by radius r (r = 0 → distance to the line segment)."""
    px, py, dx, dy = X - ax, Y - ay, bx - ax, by - ay
    h = np.clip((px * dx + py * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
    return (np.sqrt((px - dx * h) ** 2 + (py - dy * h) ** 2) - r).astype(F32)


def sd_polygon(X, Y, points):
    """Exact signed distance to a simple polygon ``[(x, y), ...]`` (any winding). Cost ∝ number of vertices."""
    P = np.asarray(points, np.float64)
    d = np.full(X.shape, np.inf, np.float64)
    inside = np.zeros(X.shape, bool)
    n = len(P)
    for i in range(n):
        ax, ay = P[i]
        bx, by = P[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        h = np.clip((wx * ex + wy * ey) / max(ex * ex + ey * ey, 1e-12), 0, 1)
        d = np.minimum(d, (wx - ex * h) ** 2 + (wy - ey * h) ** 2)
        cond = (ay <= Y) != (by <= Y)                       # even-odd crossing test
        xint = ax + (Y - ay) * ex / np.where(ey == 0, 1e-12, ey)
        inside ^= cond & (X < xint)
    d = np.sqrt(d)
    return np.where(inside, -d, d).astype(F32)


def sd_star(X, Y, cx, cy, r_outer, r_inner=None, n=5, angle=-90.0):
    """n-pointed star (``angle`` = direction of the first tip, degrees; -90 = pointing up)."""
    r_inner = r_outer * 0.45 if r_inner is None else r_inner
    pts = []
    for k in range(2 * n):
        a = np.deg2rad(angle + k * 180.0 / n)
        r = r_outer if k % 2 == 0 else r_inner
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return sd_polygon(X, Y, pts)


def sd_ring(X, Y, cx, cy, r, thickness):
    return annular(sd_circle(X, Y, cx, cy, r), thickness / 2)


def sd_arc(X, Y, cx, cy, r, start, end, thickness):
    """Arc of radius r from ``start`` to ``end`` degrees (clockwise on screen, 0 = +x), round caps."""
    a0, a1 = np.deg2rad(start), np.deg2rad(end)
    ang = np.arctan2(Y - cy, X - cx)
    rel = np.mod(ang - a0, 2 * np.pi)
    span = np.mod(a1 - a0, 2 * np.pi) or 2 * np.pi
    on_arc = rel <= span
    ring = np.abs(np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r)
    e0 = np.sqrt((X - cx - r * np.cos(a0)) ** 2 + (Y - cy - r * np.sin(a0)) ** 2)
    e1 = np.sqrt((X - cx - r * np.cos(a1)) ** 2 + (Y - cy - r * np.sin(a1)) ** 2)
    return (np.where(on_arc, ring, np.minimum(e0, e1)) - thickness / 2).astype(F32)


# ------------------------------------------------------------------ operators

def union(*ds):
    """Shapes combined: inside any of them (minimum of the distances)."""
    return np.minimum.reduce([np.asarray(d, F32) for d in ds])


def intersect(*ds):
    """Only the part inside every shape (maximum of the distances)."""
    return np.maximum.reduce([np.asarray(d, F32) for d in ds])


def subtract(a, b):
    """Shape ``a`` with shape ``b`` cut out of it: ``subtract(outer, hole)``; argument order matters."""
    return np.maximum(a, -b).astype(F32)


def smooth_union(a, b, k=8.0):
    """Union of two shapes with a smooth fillet of size ``k`` px (bigger k = blobbier join); order does not matter."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return (b * (1 - h) + a * h - k * h * (1 - h)).astype(F32)


def offset(d, k):
    """Grow or shrink the shape by ``k`` px: **k > 0 grows it outward, k < 0 shrinks it inward** (also rounds convex
    corners when growing). A ring of thickness t around a shape: ``subtract(offset(d, t), d)``."""
    return (d - k).astype(F32)


def annular(d, half_thickness):
    """Turn a filled shape into a hollow shell of thickness ``2 * half_thickness`` centred on the original edge
    (half inside, half outside). Fill the result with ``fill``."""
    return (np.abs(d) - half_thickness).astype(F32)


# ------------------------------------------------------------------ rendering

def fill(d, aa=1.0):
    """Coverage mask with an ``aa``-px anti-aliased edge (use larger aa for a soft edge)."""
    return np.clip(0.5 - d / max(float(aa), 1e-6), 0, 1).astype(F32)


def outline(d, width=2.0, aa=1.0):
    """Stroke mask of ``width`` px centred on the shape edge. Stroke fully outside: ``outline(offset(d, width/2), width)``;
    fully inside: ``outline(offset(d, -width/2), width)``."""
    return fill(np.abs(d) - width / 2.0, aa)


def glow(d, radius=20.0, falloff=1.5):
    """Soft halo around the shape (distance ``d`` > 0 is outside): 1 inside / at the edge → 0 far away (``radius`` px ≈ where it drops to ~13%)."""
    return np.exp(-2.0 * (np.maximum(d, 0) / max(radius, 1e-6)) ** falloff).astype(F32)


def bevel_light(d, width=8.0, light=(-0.6, -0.7), strength=0.5):
    """Emboss shading for a shape → (H, W) brightness multiplier around 1.

    The outer ``width`` px of the shape slope like a rounded bevel; ``light`` is the 2D direction *towards* the
    light (screen space, y down: (-0.6, -0.7) = upper-left). Edges facing the light get up to ``1 + strength``,
    edges facing away down to ``1 - strength``; the flat interior stays 1.
    Use: ``cv.paint(lerp(dark, light, 0.5) * shade[..., None], fill(d))`` or multiply a colour map.
    """
    w = max(float(width), 1e-6)
    t = np.clip(-d / w, 0, 1)                                    # 0 at the edge → 1 where the bevel ends
    h = np.sin(t * np.pi / 2)                                    # rounded profile (height)
    gy, gx = np.gradient(h)
    lx, ly = light
    nrm = np.hypot(lx, ly) or 1.0
    facing = -(gx * lx + gy * ly) / nrm * (w / (np.pi / 2))      # outward-facing slope · light, ≈ -1..1
    return (1.0 + strength * np.clip(facing, -1, 1)).astype(F32)


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "rectangle": "use sdf.sd_box(X, Y, cx, cy, half_w, half_h, radius=0); shapes: sd_circle, sd_ellipse, sd_box, sd_segment, sd_polygon, sd_star, sd_ring, sd_arc",
    "rect": "use sdf.sd_box(X, Y, cx, cy, half_w, half_h, radius=0); shapes: sd_circle, sd_ellipse, sd_box, sd_segment, sd_polygon, sd_star, sd_ring, sd_arc",
    "box": "use sdf.sd_box(X, Y, cx, cy, half_w, half_h, radius=0)",
    "square": "use sdf.sd_box(X, Y, cx, cy, half, half)",
    "roundrect": "use sdf.sd_box(X, Y, cx, cy, half_w, half_h, radius=r) (radius rounds the corners)",
    "circle": "use sdf.sd_circle(X, Y, cx, cy, r); (X, Y) come from pixelcanvas.color.grid(h, w) or cv.xy(); shapes: sd_circle, sd_ellipse, sd_box, sd_segment, sd_polygon, sd_star, sd_ring, sd_arc",
    "ellipse": "use sdf.sd_ellipse(X, Y, cx, cy, rx, ry, angle=0)",
    "line": "use sdf.sd_segment(X, Y, ax, ay, bx, by, r=0) for a rounded line",
    "segment": "use sdf.sd_segment(X, Y, ax, ay, bx, by, r=0)",
    "polygon": "use sdf.sd_polygon(X, Y, points)",
    "triangle": "use sdf.sd_polygon(X, Y, [(x1, y1), (x2, y2), (x3, y3)])",
    "star": "use sdf.sd_star(X, Y, cx, cy, r_outer, r_inner=None, n=5)",
    "ring": "use sdf.sd_ring(X, Y, cx, cy, r, thickness)",
    "arc": "use sdf.sd_arc(X, Y, cx, cy, r, start, end, thickness)",
    "abs": "distances are numpy arrays: use np.abs(d); for a hollow outline use sdf.annular(d, half_thickness)",
    "min": "to combine shapes use sdf.union(a, b, ...) (smooth: sdf.smooth_union(a, b, k)); np.minimum also works",
    "max": "to cut shapes use sdf.intersect(a, b, ...) or sdf.subtract(a, b); np.maximum also works",
    "merge": "use sdf.union(a, b, ...) or sdf.smooth_union(a, b, k=8)",
    "combine": "use sdf.union(a, b, ...) or sdf.smooth_union(a, b, k=8)",
    "difference": "use sdf.subtract(a, b)",
    "mask": "turn a distance into a coverage mask with sdf.fill(d, aa=1.0)",
    "render": "turn a distance into a coverage mask with sdf.fill(d, aa=1.0), then cv.paint(color, mask)",
    "stroke": "use sdf.outline(d, width=2) for a line along the edge",
    "border": "use sdf.outline(d, width=2) for a line along the edge",
    "shadow": "use sdf.glow(d, radius=20) for a soft halo, or offset the shape and sdf.fill it",
    "grow": "use sdf.offset(d, k) (positive k grows the shape)",
    "round": "use sdf.sd_box(..., radius=r) for rounded corners, or sdf.offset(d, k) to round any shape",
}
__getattr__ = _module_getattr(__name__, _HINTS)
