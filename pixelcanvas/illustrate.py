# -*- coding: utf-8 -*-
"""Illustrate: draw a picture freehand from shapes, the way an illustrator does.

``Sketch(cv)`` gives shapes in canvas fractions (x of the width, y of the height; lengths are fractions of the
height), local frames for characters (``with sk.at(x, y, size, rotate, mirror):``), smooth blobs and tapered
strokes through points, one-call painting with gradient, form shading, rim light, outline and drop shadow, soft
spots (blush, highlights, contact shadows), sticker die-cuts, watercolour washes and paper. Every shape is an
anti-aliased float32 mask (H, W), so numpy combines them: ``np.maximum(a, b)``, ``np.minimum(a, b)``, ``a * (1 - b)``.

    sk = Sketch(cv)
    head = sk.ellipse(0.5, 0.4, 0.12, 0.11)
    sk.paint(head, "#f6c39a", shade=0.35, highlight=0.3, line=True, name="head")
"""
from __future__ import annotations

import contextlib
import math

import numpy as np
from PIL import Image, ImageDraw

from . import filters as _f
from .color import F32, as_color, clamp01, mix, smoothstep, lerp
from . import manifest as _manifest
from ._args import strict as _strict, strict_methods as _strict_methods, check_choice as _choice

try:                                                    # optional identical speed-up
    from scipy import ndimage as _ndi
except Exception:                                       # noqa: BLE001
    _ndi = None

_RES = getattr(Image, "Resampling", Image)
_AA = 4                                                 # supersampling of Pillow-rasterised shapes

__all__ = ["Sketch", "cel_tone", "signed_distance", "grow", "edge", "paper", "limit_palette", "grade", "MOODS"]


# ----------------------------------------------------------------------------------------------- mask helpers

def _bbox(m, pad=0):
    ys = np.flatnonzero(m.max(1) > 1e-4)
    if ys.size == 0:
        return None
    xs = np.flatnonzero(m.max(0) > 1e-4)
    h, w = m.shape
    return (max(int(ys[0]) - pad, 0), min(int(ys[-1]) + 1 + pad, h), max(int(xs[0]) - pad, 0), min(int(xs[-1]) + 1 + pad, w))


_INTERSECT = {np.multiply, np.minimum, np.fmin, np.logical_and}
_UNION = {np.maximum, np.fmax, np.add, np.logical_or}


class Shape(np.ndarray):
    """A float mask (H, W) that remembers its intended box in canvas pixels (``.extent`` = x0, y0, x1, y1), parts
    off the canvas included, so ``--check`` can tell how much of a named thing the frame cuts off. ``a * b`` keeps
    the overlap of the boxes, ``np.maximum(a, b)`` / ``a + b`` their union, ``a - b`` the box of ``a``."""

    def __new__(cls, arr, extent=None):
        obj = np.asarray(arr).view(cls)
        obj.extent = None if extent is None else tuple(float(v) for v in extent)
        return obj

    def __array_finalize__(self, obj):
        self.extent = getattr(obj, "extent", None)

    def __reduce__(self):                                 # pickles as a plain array
        return np.asarray(self).__reduce__()

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kw):
        args = [x.view(np.ndarray) if isinstance(x, Shape) else x for x in inputs]
        if out is not None:
            kw["out"] = tuple(o.view(np.ndarray) if isinstance(o, Shape) else o for o in out)
        r = getattr(ufunc, method)(*args, **kw)
        if out is not None:
            return out[0] if len(out) == 1 else out
        if method != "__call__" or not isinstance(r, np.ndarray) or r.ndim != 2:
            return r
        return Shape(r, _combine(ufunc, inputs))


def _combine(ufunc, inputs):
    """The intended box of a ufunc's result from its inputs' boxes (None = unknown: the mask's own box is used)."""
    arrays = [x for x in inputs if isinstance(x, np.ndarray) and x.ndim == 2]
    exts = [getattr(x, "extent", None) for x in arrays]
    if ufunc in _INTERSECT:
        ks = [e for e in exts if e is not None]
        if not ks:
            return None
        b = (max(e[0] for e in ks), max(e[1] for e in ks), min(e[2] for e in ks), min(e[3] for e in ks))
        return b if b[2] > b[0] and b[3] > b[1] else None
    if ufunc in _UNION:
        ks = []
        for x, e in zip(arrays, exts):
            if e is None:
                bb = _bbox(np.asarray(x, F32))
                if bb is None:
                    continue
                e = (bb[2], bb[0], bb[3], bb[1])
            ks.append(e)
        if not ks:
            return None
        return (min(e[0] for e in ks), min(e[1] for e in ks), max(e[2] for e in ks), max(e[3] for e in ks))
    if ufunc is np.subtract:
        return getattr(inputs[0], "extent", None)
    return next((e for e in exts if e is not None), None)


def extent_of(mask):
    """The intended box of a mask in canvas pixels (x0, y0, x1, y1), or None when only the visible part is known."""
    return getattr(mask, "extent", None)


def _pad_box(e, by):
    if e is None:
        return None
    b = (e[0] - by, e[1] - by, e[2] + by, e[3] + by)
    return b if b[2] > b[0] and b[3] > b[1] else None


def _union_box(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _edt_sq(feature, limit):
    """Squared distance (px²) from every pixel to the nearest True pixel of ``feature``, exact up to ``limit``."""
    h, w = feature.shape
    big = float(limit + 2)
    if _ndi is not None:
        if not feature.any():
            return np.full((h, w), big * big, F32)
        d = _ndi.distance_transform_edt(~feature)
        return np.minimum(d, big).astype(F32) ** 2
    idx = np.arange(h, dtype=F32)[:, None]
    last = np.where(feature, idx, -np.inf)
    last = np.maximum.accumulate(last, axis=0)
    nxt = np.where(feature, idx, np.inf)
    nxt = np.minimum.accumulate(nxt[::-1], axis=0)[::-1]
    g = np.minimum(np.minimum(idx - last, nxt - idx), big).astype(F32)
    g2 = g * g
    r = int(math.ceil(limit)) + 1
    out = g2.copy()
    for dx in range(1, r + 1):
        dd = F32(dx * dx)
        out[:, dx:] = np.minimum(out[:, dx:], g2[:, :-dx] + dd)
        out[:, :-dx] = np.minimum(out[:, :-dx], g2[:, dx:] + dd)
    return np.minimum(out, big * big)


def signed_distance(mask, limit=64):
    """Signed distance in px of an anti-aliased mask (negative inside, positive outside), exact up to ``limit`` px.
    Feed it to ``sdf.fill`` / ``sdf.outline``; ``grow`` and ``edge`` use it."""
    m = np.asarray(mask, F32)
    inside = m >= 0.5
    d_out = np.sqrt(_edt_sq(inside, limit)) - 0.5
    d_in = np.sqrt(_edt_sq(~inside, limit)) - 0.5
    d = np.where(inside, -d_in, d_out)
    near = (m > 0.0) & (m < 1.0)
    d = np.where(near & (np.abs(d) <= 1.0), 0.5 - m, d)
    return np.clip(d, -limit, limit).astype(F32)


def _windowed(mask, reach, fn):
    """Run ``fn(sub)`` on the bbox of mask padded by reach px and paste the result into a full-size array."""
    m = np.asarray(mask, F32)
    out = np.zeros_like(m)
    bb = _bbox(m, int(reach) + 2)
    if bb is None:
        return out
    y0, y1, x0, x1 = bb
    out[y0:y1, x0:x1] = fn(m[y0:y1, x0:x1])
    return out


def grow(mask, px):
    """The mask grown outward by ``px`` pixels (negative shrinks), with an anti-aliased edge."""
    r = abs(float(px)) + 2
    return _windowed(mask, r, lambda s: clamp01(0.5 - (_f.gaussian_blur(signed_distance(s, r + 2), 0.8) - float(px))))


def edge(mask, px, where="center"):
    """A line of ``px`` pixels along the mask's edge: ``where`` = center / outside / inside."""
    _choice("edge", "where", where, ("center", "outside", "inside"))
    off = {"center": 0.0, "outside": px / 2.0, "inside": -px / 2.0}[where]
    r = abs(px) + 3

    def f(s):
        d = _f.gaussian_blur(signed_distance(s, r + 2), 0.6) - off
        return clamp01(0.5 - (np.abs(d) - px / 2.0))
    return _windowed(mask, r, f)


def _catmull(P, closed, tension, per):
    """Catmull-Rom curve through the points P (n, 2) → dense (m, 2)."""
    P = np.asarray(P, np.float64)
    n = len(P)
    if n < 3:
        return P
    if closed:
        Q = np.concatenate([P[-1:], P, P[:2]])
        segs = n
    else:
        Q = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
        segs = n - 1
    k = 0.5 * (1.0 - float(tension or 0.0))
    t = np.linspace(0, 1, per, endpoint=False)[:, None]
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        m1, m2 = k * (p2 - p0), k * (p3 - p1)
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        out.append(h00 * p1 + h10 * m1 + h01 * p2 + h11 * m2)
    out = np.concatenate(out)
    if not closed:
        out = np.concatenate([out, P[-1:]])
    return out


def _shadow_of(base):
    """Shadow colour for a fill: light fills get a light, cool tint (they stay white), others a deeper cool tone."""
    base = np.asarray(base, F32)
    if float(np.dot(base, [0.3, 0.59, 0.11])) > 0.8:
        return (base * np.array([0.76, 0.8, 0.94], F32)).astype(F32)
    return (base * np.array([0.58, 0.56, 0.68], F32)).astype(F32)


_PATTERNS = ("seigaiha", "stripes", "dots", "speckle", "checker", "tiles", "scales")


def _pattern_field(kind, X, Y, cell, width, rings):
    """Coverage of a repeated motif at pixel coordinates X, Y (already rotated); ``cell`` px, ``width`` px."""
    hw = max(width / 2.0, 0.35)
    aa = lambda d: clamp01(0.5 - d)                                              # noqa: E731
    if kind == "stripes":                                 # stripes run along the pattern angle: 0 = horizontal
        return aa(np.abs(np.mod(Y, cell) - cell / 2.0) - hw)
    if kind == "tiles":
        dx = np.abs(np.mod(X + cell / 2.0, cell) - cell / 2.0)
        dy = np.abs(np.mod(Y + cell / 2.0, cell) - cell / 2.0)
        return aa(np.minimum(dx, dy) - hw)
    if kind == "speckle":                                 # irregular grains: soil, sand, stone, granite, crumbs
        ix, iy = np.floor(X / cell), np.floor(Y / cell)
        h1 = np.mod(np.sin(ix * 127.1 + iy * 311.7) * 43758.5453, 1.0)
        h2 = np.mod(np.sin(ix * 269.5 + iy * 183.3) * 43758.5453, 1.0)
        h3 = np.mod(np.sin(ix * 419.2 + iy * 371.9) * 43758.5453, 1.0)
        r = hw * (0.4 + 1.2 * h3)
        cx, cy = (ix + 0.2 + 0.6 * h1) * cell, (iy + 0.2 + 0.6 * h2) * cell
        return aa(np.hypot(X - cx, Y - cy) - r) * (h3 > 0.25)
    if kind == "dots":
        row = np.floor(Y / cell)
        xs = X + (np.mod(row, 2) * cell / 2.0)
        d = np.hypot(np.mod(xs, cell) - cell / 2.0, np.mod(Y, cell) - cell / 2.0)
        return aa(d - hw)
    if kind == "checker":
        s = np.sin(np.pi * X / cell) * np.sin(np.pi * Y / cell)
        return clamp01(0.5 + s * cell / np.pi * 2.0)
    # seigaiha / scales: overlapping circles in rows, each lower row covering the one above
    R = cell / 2.0
    step = R / 2.0 if kind == "seigaiha" else R
    j0 = np.floor(Y / step)
    best_d = np.full(X.shape, np.inf, F32)
    for k in (2, 1, 0):                                   # nearest lower row first: it is on top
        j = j0 + k
        cx = np.round((X - np.mod(j, 2) * R) / (2 * R)) * 2 * R + np.mod(j, 2) * R
        d = np.hypot(X - cx, Y - j * step).astype(F32)
        take = (d < R) & ~np.isfinite(best_d)
        best_d = np.where(take, d, best_d)
    d = np.where(np.isfinite(best_d), best_d, R)
    if kind == "scales":
        return aa(np.abs(d - R + hw) - hw)
    t = d / (R / rings)
    return aa(np.abs(t - np.round(t)) * (R / rings) - hw) * (d > hw)


# ----------------------------------------------------------------------------------------------- frames

class _Frame:
    __slots__ = ("ox", "oy", "unit", "theta", "flip", "root")

    def __init__(self, ox, oy, unit, theta, flip, root):
        self.ox, self.oy, self.unit, self.theta, self.flip, self.root = ox, oy, unit, theta, flip, root


class _Layer:
    """Premultiplied colour + coverage that shapes are painted into (``Sketch.layer`` / ``Sketch.sticker``)."""

    def __init__(self, h, w):
        self.P = np.zeros((h, w, 3), F32)
        self.A = np.zeros((h, w), F32)
        self.ext = None                                   # intended box of what was painted in (canvas px)

    def over(self, color, a):
        bb = _bbox(a)
        if bb is None:
            return
        y0, y1, x0, x1 = bb
        sub = a[y0:y1, x0:x1, None]
        if isinstance(color, np.ndarray) and color.ndim == 3:
            col = color[y0:y1, x0:x1]
        else:
            col = as_color(color)
        self.P[y0:y1, x0:x1] = self.P[y0:y1, x0:x1] * (1 - sub) + col * sub
        self.A[y0:y1, x0:x1] = self.A[y0:y1, x0:x1] * (1 - sub[..., 0]) + sub[..., 0]

    @property
    def mask(self):
        return self.A

    @property
    def rgb(self):
        return self.P / np.maximum(self.A, 1e-6)[..., None]


# ----------------------------------------------------------------------------------------------- Sketch

class Sketch:
    """Freehand drawing on a Canvas in fractions. Positions: x as a fraction of the width, y of the height;
    lengths (radii, widths) as fractions of the height. Inside ``with sk.at(x, y, size):`` coordinates are local
    and isotropic: 1 unit = ``size`` × the parent unit, origin at (x, y) — design a character once, place it many
    times. ``light`` is the screen direction towards the light, used by shading, rim light and shadows.
    ``finish``: "auto" (default) scales shade / highlight / volume / shadow with the shape's share of the canvas, so
    small things stay flat and crisp (they read better); "full" always applies them; "flat" never does."""

    def __init__(self, cv, light=(-0.6, -0.8), finish="auto"):
        _choice("Sketch", "finish", finish, ("auto", "full", "flat"))
        self.finish = finish
        self.cv = cv
        self.H, self.W = cv.h, cv.w
        n = math.hypot(*light) or 1.0
        self.light = (light[0] / n, light[1] / n)
        self._frames = [_Frame(0.0, 0.0, float(cv.h), 0.0, 1, True)]
        self._dst = []
        self._X = self._Y = None

    # ---- coordinates
    @property
    def _f(self):
        return self._frames[-1]

    def px(self, x, y):
        """Pixel position of a point given in the current frame."""
        f = self._f
        if f.root:
            return f.ox + x * self.W, f.oy + y * self.H
        a = math.radians(f.theta)
        c, s = math.cos(a), math.sin(a)
        x = x * f.flip
        return f.ox + f.unit * (c * x - s * y), f.oy + f.unit * (s * x + c * y)

    def size(self, v):
        """Pixel length of a length given in the current frame."""
        return float(v) * self._f.unit

    def _pts(self, points):
        return np.array([self.px(float(x), float(y)) for x, y in points], np.float64)

    def _angle(self, a):
        return self._f.theta + self._f.flip * float(a)

    @contextlib.contextmanager
    def at(self, x, y, size=1.0, rotate=0.0, mirror=False):
        """A local frame: origin at (x, y), 1 unit = ``size`` × the current unit, rotated ``rotate`` degrees
        clockwise, mirrored left-right with ``mirror=True``. Nest freely; ``with sk.at(...):`` draws a character
        in its own coordinates (head at (0, -1), feet at (0, 1) ...)."""
        ox, oy = self.px(x, y)
        f = self._f
        self._frames.append(_Frame(ox, oy, f.unit * float(size), f.theta + f.flip * float(rotate),
                                   f.flip * (-1 if mirror else 1), False))
        try:
            yield self
        finally:
            self._frames.pop()

    # ---- raster helpers
    def _grid(self, x0, y0, x1, y1):
        x0, y0 = max(int(math.floor(x0)) - 2, 0), max(int(math.floor(y0)) - 2, 0)
        x1, y1 = min(int(math.ceil(x1)) + 3, self.W), min(int(math.ceil(y1)) + 3, self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(F32)
        return (y0, y1, x0, x1), xx + 0.5, yy + 0.5

    def _place(self, win, sub):
        out = np.zeros((self.H, self.W), F32)
        if win is not None:
            y0, y1, x0, x1 = win
            out[y0:y1, x0:x1] = sub
        return out

    def _poly_raster(self, polys, circles=()):
        """Rasterise polygons (list of (n, 2) px arrays) and discs ((x, y, r) px) into one mask, supersampled."""
        allp = [p for p in polys if len(p)] + [np.array([[x - r, y - r], [x + r, y + r]]) for x, y, r in circles]
        if not allp:
            return np.zeros((self.H, self.W), F32)
        P = np.concatenate(allp)
        ext = (float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max()))
        x0, y0 = max(int(P[:, 0].min()) - 2, 0), max(int(P[:, 1].min()) - 2, 0)
        x1, y1 = min(int(P[:, 0].max()) + 3, self.W), min(int(P[:, 1].max()) + 3, self.H)
        if x1 <= x0 or y1 <= y0:
            return Shape(np.zeros((self.H, self.W), F32), ext)
        img = Image.new("L", ((x1 - x0) * _AA, (y1 - y0) * _AA), 0)
        d = ImageDraw.Draw(img)
        for p in polys:
            if len(p) >= 3:
                d.polygon([((x - x0) * _AA, (y - y0) * _AA) for x, y in p], fill=255)
        for x, y, r in circles:
            r = r * _AA
            cx, cy = (x - x0) * _AA, (y - y0) * _AA
            if r > 0.3:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        img = img.resize((x1 - x0, y1 - y0), _RES.BOX)
        return Shape(self._place((y0, y1, x0, x1), np.asarray(img, F32) / 255.0), ext)

    # ---- shapes → masks
    def circle(self, x, y, r):
        """Disc mask: centre (x, y), radius r."""
        return self.ellipse(x, y, r, r)

    def ellipse(self, x, y, rx, ry, angle=0.0):
        """Ellipse mask: centre (x, y), half-axes rx, ry, ``angle`` degrees clockwise."""
        from .sdf import sd_ellipse
        cx, cy = self.px(x, y)
        a, b = self.size(rx), self.size(ry)
        t = math.radians(self._angle(angle))
        hw, hh = math.hypot(a * math.cos(t), b * math.sin(t)), math.hypot(a * math.sin(t), b * math.cos(t))
        ext = (cx - hw, cy - hh, cx + hw, cy + hh)
        R = max(a, b) + 2
        g = self._grid(cx - R, cy - R, cx + R, cy + R)
        if g is None:
            return Shape(np.zeros((self.H, self.W), F32), ext)
        win, X, Y = g
        d = sd_ellipse(X, Y, cx, cy, max(a, 0.3), max(b, 0.3), self._angle(angle))
        return Shape(self._place(win, clamp01(0.5 - d)), ext)

    def box(self, x0, y0, x1, y1, radius=0.0):
        """Rectangle mask between two corners, ``radius`` rounds the corners (rotates with the frame)."""
        from .sdf import sd_box
        cx, cy = self.px((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        f = self._f
        if f.root:
            hw, hh = abs(x1 - x0) * self.W / 2.0, abs(y1 - y0) * self.H / 2.0
        else:
            hw, hh = self.size(abs(x1 - x0)) / 2.0, self.size(abs(y1 - y0)) / 2.0
        r = min(self.size(radius), hw, hh)
        t = math.radians(self._f.theta)
        ew, eh = abs(math.cos(t)) * hw + abs(math.sin(t)) * hh, abs(math.sin(t)) * hw + abs(math.cos(t)) * hh
        ext = (cx - ew, cy - eh, cx + ew, cy + eh)
        R = math.hypot(hw, hh) + 2
        g = self._grid(cx - R, cy - R, cx + R, cy + R)
        if g is None:
            return Shape(np.zeros((self.H, self.W), F32), ext)
        win, X, Y = g
        d = sd_box(X, Y, cx, cy, hw, hh, r, self._f.theta)
        return Shape(self._place(win, clamp01(0.5 - d)), ext)

    def polygon(self, points):
        """Polygon mask with straight sides through ``points`` [(x, y), ...]."""
        s = self._poly_raster([self._pts(points)])
        s.planar = True                                   # paint(volume=) shades it as a flat face
        return s

    def blob(self, points, tension=0.0):
        """Closed smooth shape through ``points`` (a Catmull-Rom curve): leaves, hair, clouds, bodies.
        ``tension`` 0 = round, 0.5 = tighter, 1 = the polygon."""
        P = _catmull(self._pts(points), True, tension, 24)
        return self._poly_raster([P])

    def curve(self, points, n=16):
        """Dense points of a smooth curve through ``points`` (same coordinates): join several into one
        ``polygon``, offset copies for hatching, or feed ``stroke`` with ``smooth=False``."""
        P = np.asarray(points, np.float64)
        return [tuple(p) for p in (_catmull(P, False, 0.0, int(n)) if len(P) >= 3 else P)]

    def stroke(self, points, width=0.01, taper=None, smooth=True, cap="round"):
        """A line through ``points``, smooth by default. ``width`` is one value or one per point (variable width);
        ``taper`` = "end" / "start" / "both" narrows it to a point: brush lines, hair, whiskers, tails, the
        saxophone's body. ``cap`` = round / flat."""
        if taper is not None:
            _choice("stroke", "taper", taper, ("end", "start", "both"))
        _choice("stroke", "cap", cap, ("round", "flat"))
        pts = [(float(x), float(y)) for x, y in points]
        P0 = self._pts(pts)
        if np.ndim(width) == 0:
            w0 = np.full(len(pts), self.size(width))
        else:
            if len(width) != len(pts):
                raise ValueError("stroke: give one width or one width per point (%d points, %d widths)" % (len(pts), len(width)))
            w0 = np.array([self.size(v) for v in width])
        return self._stroke_px(P0, w0, taper, smooth, cap)

    def _stroke_px(self, P0, w0, taper=None, smooth=True, cap="round"):
        """Stroke through pixel points P0 (n, 2) with pixel widths w0 (n,)."""
        P0 = np.asarray(P0, np.float64)
        if smooth and len(P0) >= 3:
            P = _catmull(P0, False, 0.0, 24)
        else:
            seg = [np.linspace(P0[i], P0[i + 1], 16, endpoint=False) for i in range(len(P0) - 1)]
            P = np.concatenate(seg + [P0[-1:]]) if seg else P0
        idx = np.linspace(0, len(P0) - 1, len(P))
        Wd = np.interp(idx, np.arange(len(P0)), w0)
        t = np.linspace(0, 1, len(P))
        if taper == "end":
            Wd = Wd * (1 - t) ** 0.8
        elif taper == "start":
            Wd = Wd * t ** 0.8
        elif taper == "both":
            Wd = Wd * np.sin(np.pi * t) ** 0.7
        r = Wd / 2.0
        if len(P) < 2:
            return self._poly_raster([], [(P[0, 0], P[0, 1], r[0])])
        tg = np.gradient(P, axis=0)
        tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-9)[:, None]
        nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
        L, R = P + nrm * r[:, None], P - nrm * r[:, None]
        quads = [np.array([L[i], L[i + 1], R[i + 1], R[i]]) for i in range(len(P) - 1)]
        discs = [(P[0, 0], P[0, 1], r[0]), (P[-1, 0], P[-1, 1], r[-1])] if cap == "round" else []
        if not smooth:                                    # round joins at the corners of a polyline
            discs += [(x, y, w / 2.0) for (x, y), w in zip(P0[1:-1], w0[1:-1])]
        return self._poly_raster(quads, discs)

    def arrow(self, p0, p1, via=None, width=0.006, head=None, style="filled", color="#222222", outline=None,
              both=False, opacity=1.0, name=None):
        """Draw an arrow from point ``p0`` to ``p1`` (straight, or curved so it passes through ``via``); returns its
        mask. ``head`` = head length (default 5 x width); ``style`` = filled / open / none; ``outline`` = a colour
        around it so it reads on any ground; ``both=True`` puts a head at each end."""
        _choice("arrow", "style", style, ("filled", "open", "none"))
        A, B = np.array(self.px(*p0), np.float64), np.array(self.px(*p1), np.float64)
        if via is not None:
            V = np.array(self.px(*via), np.float64)
            C = 2 * V - (A + B) / 2.0                      # quadratic control point: the curve passes through via
            t = np.linspace(0, 1, 48)[:, None]
            P = (1 - t) ** 2 * A + 2 * (1 - t) * t * C + t ** 2 * B
        else:
            P = np.linspace(A, B, 48)
        w = self.size(width)
        hl = self.size(head) if head is not None else 5.0 * w
        hl = min(hl, 0.45 * float(np.hypot(*(B - A))) + 1e-6)
        ends = (-1, 0) if both else (-1,)
        heads, shaft = [], P.copy()
        for k in ends:
            tip = P[k]
            dist = np.hypot(*(P - tip).T)
            j = int(np.argmin(np.where(dist > 1e-6, np.abs(dist - hl), np.inf)))   # one head length back
            d = tip - P[j]
            d = d / max(float(np.hypot(*d)), 1e-9)
            n = np.array([-d[1], d[0]])
            base = tip - d * hl
            heads.append(np.array([base + n * hl * 0.6, tip, base - n * hl * 0.6]))
            if style == "filled":                          # the shaft stops inside the head: the tip stays sharp
                shaft[k] = base + d * hl * 0.3
        m = self._stroke_px(shaft, np.full(len(shaft), w), None, via is not None, "round")
        for h in heads:
            if style == "filled":
                m = np.maximum(m, self._poly_raster([h]))
            elif style == "open":
                m = np.maximum(m, self._stroke_px(h, np.full(3, w), None, False, "round"))
        if outline:
            self._blend(outline, grow(m, max(1.5, 0.7 * w)), opacity)
        self._blend(color, m, opacity)
        if name:
            _manifest.note(self.cv, "object", name, mask=m)
        return m

    def label(self, text, at, to=None, size=0.035, color="auto", box="pill", fill=None, arrow=False,
              line_color=None, font=None, bold=True, anchor="mm", pad=0.35):
        """A caption or callout in one call: ``text`` at the point ``at`` (``anchor`` as in cv.text) on a ``box`` =
        pill / box / None plate, with a leader line (``arrow=True``: an arrow) to the point ``to``. ``color="auto"``
        and ``fill=None`` contrast with what is underneath; the label is kept inside the canvas. Recorded as text
        for the report, never as an object. Returns the plate's bbox in px."""
        if box is not None:
            _choice("label", "box", box, ("pill", "box"))
        from .text import text_bbox
        px, py = self.px(*at)
        sz = self.size(size)
        tb = text_bbox(text, (px, py), sz, font=font, bold=bold, anchor=anchor)
        p = pad * sz if box else 0.15 * sz
        x0, y0, x1, y1 = tb[0] - p * 1.4, tb[1] - p, tb[2] + p * 1.4, tb[3] + p
        mg = 0.012 * self.H
        dx = max(mg - x0, 0) - max(x1 - (self.W - mg), 0)
        dy = max(mg - y0, 0) - max(y1 - (self.H - mg), 0)
        x0, x1, y0, y1, px, py = x0 + dx, x1 + dx, y0 + dy, y1 + dy, px + dx, py + dy
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        under = self._luma_in(x0, y0, x1, y1)
        if box:
            plate = fill if fill is not None else ("#1f2330" if under > 0.6 else "#fbfaf6")   # stands out
            plate_l = float(np.dot(as_color(plate), [0.2126, 0.7152, 0.0722]))
        else:
            plate_l = under
        ink = color if color != "auto" else ("#1f2330" if plate_l > 0.5 else "#fbfaf6")
        if to is not None:
            tx, ty = self.px(*to)
            lc = line_color or (plate if box else ink)               # the label's own colour ...
            halo = "#fbfaf6" if float(np.dot(as_color(lc), [0.2126, 0.7152, 0.0722])) < 0.5 else "#1f2330"
            sx, sy = min(max(tx, x0), x1), min(max(ty, y0), y1)
            w = max(0.004 * self.H, 1.0)
            if arrow:                                                # ... with a thin halo: reads on any ground
                with self._root():
                    self.arrow((sx / self.W, sy / self.H), (tx / self.W, ty / self.H), width=w / self.H, color=lc,
                               outline=halo)
            else:
                lm = np.maximum(self._stroke_px(np.array([[sx, sy], [tx, ty]]), np.array([w, w]), None, False),
                                self._poly_raster([], [(tx, ty, 2.2 * w)]))
                self._blend(halo, grow(lm, max(1.5, 0.6 * w)))
                self._blend(lc, lm)
        if box:
            from .sdf import sd_box
            from .masks import shift
            r = (y1 - y0) / 2.0 if box == "pill" else 0.2 * (y1 - y0)
            g = self._grid(x0, y0, x1, y1)
            if g is not None:
                win, X, Y = g
                plate_m = self._place(win, clamp01(0.5 - sd_box(X, Y, cx, cy, (x1 - x0) / 2.0, (y1 - y0) / 2.0, r)))
                soft = 0.006 * self.H
                self._blend("#000000", _windowed(plate_m, 4 * soft, lambda s_: _f.gaussian_blur(
                    shift(s_, 0, soft), soft)), 0.18)
                self._blend(plate, plate_m, 0.95)
        with self._root():
            self.text(text, px / self.W, py / self.H, sz / self.H, ink, anchor=anchor, bold=bold,
                      **({"font": font} if font else {}))
        return (x0, y0, x1, y1)

    @contextlib.contextmanager
    def _root(self):
        saved = self._frames
        self._frames = [saved[0]]
        try:
            yield
        finally:
            self._frames = saved

    def _luma_in(self, x0, y0, x1, y1):
        xi0, yi0 = min(max(int(x0), 0), self.W - 1), min(max(int(y0), 0), self.H - 1)
        xi1, yi1 = max(min(int(x1) + 1, self.W), xi0 + 1), max(min(int(y1) + 1, self.H), yi0 + 1)
        sub = np.clip(self.cv.buf[yi0:yi1, xi0:xi1], 0, 1)
        return float((sub * np.array([0.2126, 0.7152, 0.0722], F32)).sum(-1).mean())

    def arc(self, x, y, r, start, end, width=0.01, taper=None):
        """An arc stroke of radius r from ``start`` to ``end`` degrees (clockwise from +x): smiles, brows, handles."""
        n = max(int(abs(end - start) / 6) + 3, 4)
        a = np.radians(np.linspace(start, end, n))
        return self.stroke([(x + r * math.cos(t), y + r * math.sin(t)) for t in a], width, taper=taper)

    # ---- mask operations in frame units
    def grow(self, mask, by):
        """The mask grown by ``by`` (frame units; negative shrinks)."""
        return Shape(grow(mask, self.size(by)), _pad_box(extent_of(mask), self.size(by)))

    def outline(self, mask, width=0.006, where="center"):
        """A line of ``width`` along the mask's edge: ``where`` = center / outside / inside."""
        return Shape(edge(mask, self.size(width), where), _pad_box(extent_of(mask), self.size(width) / 2.0))

    # ---- painting
    def _blend(self, color, a, opacity=1.0):
        if opacity != 1.0:
            a = a * F32(opacity)
        if self._dst:
            self._dst[-1].over(color, a)
        else:
            self.cv.paint(color, a)

    def _fill_color(self, mask, color, angle, win):
        """Colour (3,) or a full (H, W, 3) gradient map over the mask's bbox."""
        if not (isinstance(color, (list, tuple)) and color and isinstance(color[0], (list, tuple))
                and len(color[0]) == 2 and not isinstance(color[0][0], (list, tuple))):
            c = np.asarray(color, F32) if isinstance(color, np.ndarray) else as_color(color)
            if c.ndim == 3:
                y0, y1, x0, x1 = win
                wm = mask[y0:y1, x0:x1, None]
                return c, (c[y0:y1, x0:x1] * wm).sum((0, 1)) / max(float(wm.sum()), 1e-6)
            return c, c
        stops = [(float(t), c) for t, c in color]
        y0, y1, x0, x1 = win
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(F32)
        cx, cy, hw, hh = (x0 + x1) / 2.0, (y0 + y1) / 2.0, max((x1 - x0) / 2.0, 1), max((y1 - y0) / 2.0, 1)
        if angle == "radial":
            lx, ly = self.light
            t = np.hypot((xx - cx - lx * hw * 0.35) / hw, (yy - cy - ly * hh * 0.35) / hh) / 1.3
        else:
            a = math.radians(float(angle))
            dx, dy = math.cos(a), math.sin(a)
            ext = abs(dx) * hw + abs(dy) * hh
            t = ((xx - cx) * dx + (yy - cy) * dy) / max(ext, 1e-6) * 0.5 + 0.5
        full = np.zeros((self.H, self.W, 3), F32)
        full[y0:y1, x0:x1] = mix(stops, clamp01(t))
        return full, np.mean([as_color(c) for _, c in stops], 0)

    def paint(self, mask, color, *, line=None, line_width=0.005, shade=0.0, highlight=0.0, volume=0.0, shadow=None,
              spec=0.0, glow=None, angle=90, shade_color=None, opacity=1.0, name=None, kind="object"):
        """Paint a shape in one call. ``color``: a colour, or stops ``[(0, c0), (1, c1)]`` for a gradient across
        the shape (``angle`` degrees, 90 = top to bottom, "radial" = round from the lit side).
        ``shade`` 0–1: soft form shadow on the side away from the light; ``highlight`` 0–1: rim light on the lit
        edge; ``volume`` 0–1: a rounded, puffy body (darker towards every edge, most on the far side) in a tinted
        shadow colour, so whites and pastels stay light (cloth, clouds, snow, a sail); on a ``sk.polygon`` face it
        shades the face as a plane, darker away from the light (box sides, roofs); ``spec`` 0–1: a specular hot spot
        on the lit side (glossy fruit, plastic, glass, metal); ``line``: outline colour (True = a darker tone of the fill) of ``line_width``; ``shadow``: True or
        dict(offset=(dx, dy), blur, opacity, color) for a soft drop shadow under it, "ground" for a contact shadow on
        the floor under its bottom edge; ``glow``: a colour or
        dict(color, radius, strength) for light spilling around it (neon, lamps, the moon); ``name``: records the shape
        for ``--check`` (``kind`` defaults to "object"). Returns the mask."""
        ext = extent_of(mask)
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        for L in self._dst:                               # the group's intended box, parts off the canvas included
            L.ext = _union_box(L.ext, ext if ext is not None or bb is None else (bb[2], bb[0], bb[3], bb[1]))
        if bb is None:
            return Shape(m, ext)
        area = float(m[bb[0]:bb[1], bb[2]:bb[3]].sum())
        R = math.sqrt(max(area, 1.0) / math.pi)
        k = self._detail(area)
        shade, highlight, volume = float(shade) * k, float(highlight) * k, float(volume) * k
        spec = float(spec)                              # cues a small thing needs too: not scaled by detail
        H_, W_ = m.shape

        def oval(cx, cy, a, b, edge=0.0):              # soft oval alpha in canvas pixels, full frame
            a, b = max(a, 0.5), max(b, 0.5)
            y0, y1 = max(int(cy - b) - 2, 0), min(int(cy + b) + 3, H_)
            x0, x1 = max(int(cx - a) - 2, 0), min(int(cx + a) + 3, W_)
            out = np.zeros((H_, W_), F32)
            if y1 > y0 and x1 > x0:
                Y, X = np.mgrid[y0:y1, x0:x1].astype(F32)
                r = np.sqrt(((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2)
                out[y0:y1, x0:x1] = 1.0 - smoothstep(edge, 1.0, r)
            return out
        if isinstance(shadow, str) and shadow == "ground":   # contact shadow on the floor under the bottom edge
            w = bb[3] - bb[2]
            cx = (bb[2] + bb[3]) / 2 - self.light[0] * 0.12 * w
            self._blend("#1d1830", oval(cx, bb[1] - 0.015 * w, 0.62 * w, max(0.1 * w, 2.0)), 0.5 * opacity)
            shadow = None
        if shadow:
            sh = dict(offset=None, blur=0.012, opacity=0.3, color="#1d1830")
            if isinstance(shadow, dict):
                from ._args import reject_unknown
                sh.update(reject_unknown("paint shadow", shadow, list(sh)))
            off = sh["offset"]
            if off is None:
                dx, dy = -self.light[0] * self.size(0.012), -self.light[1] * self.size(0.012)
            else:
                dx, dy = self.size(off[0]), self.size(off[1])
            from .masks import shift
            blur = self.size(sh["blur"])
            s = _windowed(m, abs(dx) + abs(dy) + 3 * blur + 2,
                          lambda sub: _f.gaussian_blur(shift(sub, dx, dy), blur) if blur > 0 else shift(sub, dx, dy))
            self._blend(sh["color"], s, sh["opacity"] * opacity * k)
        if glow is not None and glow is not False:
            gl = dict(color=glow, radius=0.03, strength=0.8) if not isinstance(glow, dict) else                 dict(dict(color="#ffd27a", radius=0.03, strength=0.8), **glow)
            r = self.size(gl["radius"])
            g = _windowed(m, 3 * r, lambda sub: _f.gaussian_blur(sub, r)) * F32(gl["strength"] * opacity)
            if self._dst:
                self._blend(gl["color"], clamp01(g))
            else:
                self.cv.paint(gl["color"], g, "add")
        col, base = self._fill_color(m, color, angle, bb)
        self._blend(col, m, opacity)
        lx, ly = self.light
        from .masks import shift
        dark = as_color(shade_color) if shade_color is not None else _shadow_of(base)
        if (volume or shade) and getattr(mask, "planar", False):   # a flat face: a plane, darker away from the light
            y0, y1, x0, x1 = bb
            Y, X = np.mgrid[y0:y1, x0:x1].astype(F32)
            t = ((X - (x0 + x1) / 2) * lx + (Y - (y0 + y1) / 2) * ly) / max(R, 1.0)
            g = np.zeros_like(m)
            g[y0:y1, x0:x1] = m[y0:y1, x0:x1] * (0.35 * clamp01(0.5 - 0.5 * t))
            self._blend(dark, g * F32(min(max(float(volume), float(shade)), 1.0)), opacity)
            volume = shade = 0.0
        if volume:
            reach = 0.9 * R

            def fv(sub):
                inside = clamp01(-signed_distance(sub, reach + 2) / reach)       # 0 at the edge -> 1 deep inside
                away = _f.gaussian_blur(shift(sub, lx * 0.5 * R, ly * 0.5 * R), 0.4 * R)
                return sub * (1.0 - inside) ** 1.6 * (0.45 + 0.55 * clamp01(1.0 - away))
            self._blend(dark, _windowed(m, R, fv) * F32(min(float(volume), 1.0)), opacity)
        if shade:
            k = 0.75 * R

            def fs(sub):
                away = _f.gaussian_blur(shift(sub, lx * k, ly * k), 0.4 * R)
                return sub * smoothstep(0.0, 0.75, 1.0 - away)
            self._blend(dark, _windowed(m, 2 * R, fs) * F32(min(float(shade), 1.0)), opacity)
        if highlight:
            light = lerp(base, (1.0, 0.98, 0.94), 0.65)
            k = 0.14 * R

            def fh(sub):
                to = _f.gaussian_blur(shift(sub, -lx * k, -ly * k), 0.12 * R)
                return sub * clamp01(1.0 - to)
            self._blend(light, _windowed(m, R, fh) * F32(min(float(highlight), 1.0)), opacity)
        if spec:
            y0, y1, x0, x1 = bb
            hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
            a = 0.16 * min(hw, hh)
            sp = oval((x0 + x1) / 2 + lx * 0.5 * hw, (y0 + y1) / 2 + ly * 0.5 * hh, a, 0.75 * a, 0.25) * m
            self._blend((1.0, 0.99, 0.96), sp * F32(min(float(spec), 1.0) * 0.9), opacity)
        if line is not None and line is not False:
            lw = line_width
            if isinstance(line, str) and line == "ink":                 # comic / ink styles: bold near-black line
                lc, lw = "#15131a", line_width * 2.4
            else:
                lc = base * np.array([0.42, 0.38, 0.45], F32) if line is True else line
            self._blend(lc, self.outline(m, lw), opacity)
        if name:
            _manifest.note(self.cv, kind, name, mask=m, extent=ext)
        return Shape(m, ext)

    def _detail(self, area_px):
        """How much modelling a shape of this area gets: 1 = full, 0 = flat (finish="auto": flat below 3 % of the
        canvas, full from 10 %)."""
        if self.finish == "full":
            return 1.0
        if self.finish == "flat":
            return 0.0
        share = area_px / float(self.W * self.H)
        t = min(max((share - 0.03) / 0.07, 0.0), 1.0)
        return float(t * t * (3 - 2 * t))

    def pattern(self, mask, kind, color, size=0.05, width=None, angle=0.0, rings=4, opacity=1.0):
        """A repeated motif clipped to ``mask``: ``kind`` = seigaiha (overlapping wave rings), scales, stripes
        (pinstripes), dots, speckle (irregular grains: soil, sand, stone), checker, tiles (grid lines). ``size`` is the cell (frame units); stripes run along ``angle`` (0 = horizontal, 90 = vertical); ``width`` the line
        width or dot diameter (default: a fine line, dots a third of the cell), ``angle`` turns the pattern. Paint the base colour first, the pattern over it.
        Returns its alpha."""
        _choice("pattern", "kind", kind, _PATTERNS)
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        if bb is None:
            return m
        y0, y1, x0, x1 = bb
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(F32)
        t = math.radians(self._angle(angle))
        c, s_ = math.cos(t), math.sin(t)
        U, V = xx * c + yy * s_, -xx * s_ + yy * c
        cell = max(self.size(size), 2.0)
        w = self.size(width) if width is not None else (cell * 0.35 if kind in ("dots", "speckle") else self.size(0.004))
        sub = _pattern_field(kind, U, V, cell, w, int(rings)) * m[y0:y1, x0:x1]
        al = self._place((y0, y1, x0, x1), sub.astype(F32))
        self._blend(color, al, opacity)
        return al

    def spot(self, x, y, rx, ry=None, color="#000000", opacity=1.0, angle=0.0, clip=None):
        """A soft oval that fades to nothing at its edge: blush, a highlight, a glow, a contact shadow under feet.
        ``clip`` keeps it inside a mask. Returns its alpha."""
        ry = rx if ry is None else ry
        cx, cy = self.px(x, y)
        a, b = max(self.size(rx), 0.5), max(self.size(ry), 0.5)
        R = max(a, b) + 2
        g = self._grid(cx - R, cy - R, cx + R, cy + R)
        if g is None:
            return np.zeros((self.H, self.W), F32)
        win, X, Y = g
        t = math.radians(self._angle(angle))
        c, s = math.cos(t), math.sin(t)
        u, v = (X - cx) * c + (Y - cy) * s, -(X - cx) * s + (Y - cy) * c
        r = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        al = self._place(win, 1.0 - smoothstep(0.0, 1.0, r))
        if clip is not None:
            al = al * np.asarray(clip, F32)
        self._blend(color, al, opacity)
        return al

    def wash(self, mask, color, edge=0.6, grain=0.35, bleed=0.006, opacity=0.85, seed=0, name=None):
        """Watercolour wash: a transparent layer with a wobbly edge, pigment pooled at the wet edge and granulated
        in the paper. Paint light washes first, darker ones over them."""
        from .noise import fbm2d
        m = np.asarray(mask, F32)
        bb = _bbox(m, int(self.size(bleed) * 3) + 4)
        if bb is None:
            return m
        y0, y1, x0, x1 = bb
        sub = m[y0:y1, x0:x1]
        h, w = sub.shape
        sc = max(self.H * 0.02, 4.0)
        n1 = fbm2d(h, w, sc * 3, 4, seed=seed, offset=(x0, y0))
        n2 = fbm2d(h, w, sc * 0.35, 3, seed=seed + 7, offset=(x0, y0))
        soft = _f.gaussian_blur(sub, max(self.size(bleed), 0.5))
        body = clamp01((soft - 0.5 + 0.18 * n1) * 4.0 + 0.5)
        rim = clamp01(body - _f.gaussian_blur(body, max(self.size(0.008), 1.0))) * 2.2
        a = body * (0.55 + 0.25 * n1 + float(grain) * 0.35 * n2 + float(edge) * rim)
        full = self._place((y0, y1, x0, x1), clamp01(a))
        base = as_color(color)
        self._blend(base, full, opacity)
        if name:
            _manifest.note(self.cv, "object", name, mask=m)
        return full

    def text(self, text, x, y, size=0.05, color="#ffffff", **style):
        """Exact text at (x, y) of height ``size`` (frame units); every ``cv.text`` style works: ``anchor``,
        ``font``, ``bold``, ``tracking`` (letter spacing), ``outline``, ``shadow``, ``glow``, ``effect`` ...
        Returns the bbox in px."""
        px, py = self.px(x, y)
        style = dict(style)
        if self._f.theta:
            style["rotate"] = style.get("rotate", 0.0) - self._f.theta
        if self._dst:
            from .text import text_mask
            m = text_mask(self.H, self.W, text, (px, py), self.size(size),
                          **{k: v for k, v in style.items() if k in ("font", "bold", "anchor", "align", "tracking",
                                                                    "max_width", "line_spacing", "rotate")})
            self._blend(color, m)
            b = _bbox(m)
            return (b[2], b[0], b[3], b[1]) if b else None
        self.cv.text(text, (px, py), self.size(size), color, **style)
        return self.cv.last_bbox

    # ---- materials and states: one call each
    def glass(self, mask, tint="#cfe8ff", alpha=0.22, line="#5b7a99", rim="#ffffff", streak=True, name=None,
              kind="object"):
        """A see-through object that still reads: the full shape in a faint ``tint``, a thin ``line`` edge, a bright
        inner ``rim`` and one highlight streak; whatever was drawn before shows through (draw the contents first,
        e.g. ``sk.level``). Recorded as see-through. Returns the mask."""
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        if bb is None:
            return m
        w = max(self.size(0.004), 1.0)
        self._blend(tint, m, alpha)
        self._blend(line, edge(m, w, "inside"), 0.9)
        self._blend(rim, edge(grow(m, -w), 0.8 * w, "inside"), 0.75)
        if streak:
            y0, y1, x0, x1 = bb
            bw, bh = x1 - x0, y1 - y0
            sx = x0 + 0.24 * bw
            st = self._stroke_px(np.array([[sx, y0 + 0.16 * bh], [sx + 0.02 * bw, y1 - 0.24 * bh]]),
                                 np.full(2, max(0.07 * bw, 1.5)), "both", False, "round")
            self._blend("#ffffff", st * grow(m, -2 * w), 0.7)
        if name:
            _manifest.note(self.cv, kind, name, mask=m, see_through=True)
        return m

    def level(self, mask, frac=0.5, color="#6aa8d8", opacity=0.85, name=None):
        """A liquid filling ``mask`` (a container) up to ``frac`` of its height, with a lighter surface line:
        liquid in a flask, a fishbowl, a teapot three-quarters full. Returns the liquid mask."""
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        if bb is None:
            return m
        y0, y1, x0, x1 = bb
        yl = y1 - float(frac) * (y1 - y0)
        Y = np.arange(self.H, dtype=F32)[:, None] + 0.5
        liquid = m * clamp01(Y - yl + 0.5)
        self._blend(color, liquid, opacity)
        w = max(self.size(0.004), 1.0)
        surf = m * clamp01(1.0 - np.abs(Y - yl) / w)
        self._blend(lerp(as_color(color), (1, 1, 1), 0.55), surf, 0.9)
        if name:
            _manifest.note(self.cv, "object", name, mask=liquid)
        return liquid

    def crack(self, mask, start=None, color="#2a1c16", width=0.015, branches=2, seed=0, name=None):
        """A jagged crack across ``mask`` (an egg, a flask, a teacup): a zig-zag line that narrows, with short
        branches and a lit lip beside it, kept inside the shape. ``start`` = a point (frame coordinates) or the
        upper part of the shape. Returns the crack's mask."""
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        if bb is None:
            return m
        rng = np.random.default_rng(seed)
        y0, y1, x0, x1 = bb
        bw, bh = x1 - x0, y1 - y0
        A = np.array(self.px(*start), np.float64) if start is not None else np.array([x0 + 0.35 * bw, y0 + 0.12 * bh])
        B = np.array([x0 + (0.55 + 0.15 * rng.random()) * bw, y0 + 0.88 * bh])
        n = 7
        P = np.linspace(A, B, n)
        d = (B - A) / max(float(np.hypot(*(B - A))), 1e-9)
        nrm = np.array([-d[1], d[0]])
        L = float(np.hypot(*(B - A)))
        P[1:-1] += nrm * (0.07 * L * np.array([(-1) ** i for i in range(1, n - 1)]) * (0.6 + 0.8 * rng.random(n - 2)))[:, None]
        w = max(self.size(width), 1.0)
        main = self._stroke_px(P, np.full(n, w), "end", False, "round")
        parts = [main]
        for b in range(int(branches)):
            i = 2 + int(rng.integers(0, n - 3))
            ang = math.atan2(d[1], d[0]) + (1 if b % 2 else -1) * (0.7 + 0.4 * rng.random())
            Q = np.array([P[i], P[i] + np.array([math.cos(ang), math.sin(ang)]) * L * (0.18 + 0.1 * rng.random())])
            parts.append(self._stroke_px(Q, np.full(2, 0.7 * w), "end", False, "round"))
        cr = np.maximum.reduce(parts) * m
        lip = np.maximum.reduce([self._stroke_px(P + np.array(self.light) * 1.2 * w, np.full(n, 0.6 * w), "end",
                                                 False, "round")]) * m
        self._blend("#ffffff", lip * (1 - cr), 0.45)
        self._blend(color, cr)
        if name:
            _manifest.note(self.cv, "object", name, mask=cr)
        return cr

    def shards(self, at, size=0.1, color="#d8ecf8", n=6, glass=True, seed=0, name=None):
        """Broken pieces scattered around the point ``at`` within ``size``: jagged triangles and quads with gaps
        between them, drawn as glass (``glass=True``) or as solid pieces of ``color`` with a darker edge.
        Returns their joint mask."""
        rng = np.random.default_rng(seed)
        cx, cy = self.px(*at)
        R = self.size(size)
        allm = []
        for i in range(int(n)):
            a = 2 * math.pi * (i + rng.random() * 0.6) / n
            dist = R * (0.25 + 0.75 * rng.random())
            px, py = cx + math.cos(a) * dist, cy + math.sin(a) * dist * 0.55
            r = R * (0.28 + 0.22 * rng.random())
            k = 3 + int(rng.integers(0, 2))
            angs = np.sort((np.arange(k) + 0.6 * rng.random(k)) * 2 * math.pi / k + rng.random() * 6)
            pts = np.stack([px + np.cos(angs) * r * (0.6 + 0.4 * rng.random(k)),
                            py + np.sin(angs) * r * (0.55 + 0.35 * rng.random(k))], 1)
            pm = self._poly_raster([pts])
            if glass:
                self.glass(pm, tint=color, alpha=0.55, streak=False)
            else:
                self._blend(color, pm)
                self._blend(_shadow_of(as_color(color)) * 0.8, edge(pm, max(self.size(0.003), 1.0), "inside"))
            allm.append(pm)
        out = np.maximum.reduce(allm)
        if name:
            _manifest.note(self.cv, "object", name, mask=out, see_through=bool(glass))
        return out

    def spill(self, at, color="#6aa8d8", size=0.12, on=None, opacity=0.85, seed=0, name=None):
        """A spilled pool spreading over the ground from the point ``at`` (a rim, a tipped teapot's spout): a flat,
        lobed puddle of width ``size`` on the ground line ``on`` (default: the point's own height), with a stream
        down to it when the point is above the ground, and a light sheen. Returns the pool's mask."""
        rng = np.random.default_rng(seed)
        ax, ay = self.px(*at)
        gy = self.px(at[0], on)[1] if on is not None else ay
        R = self.size(size)
        cx = ax + 0.35 * R * (1 if rng.random() > 0.5 else -1)
        t = np.linspace(0, 2 * math.pi, 14, endpoint=False)
        rad = R * (0.75 + 0.35 * rng.random(14))
        pts = np.stack([cx + np.cos(t) * rad, gy + np.sin(t) * rad * 0.26], 1)
        pool = self._poly_raster([_catmull(pts, True, 0.0, 10)])
        if gy - ay > 2:
            stream = self._stroke_px(np.array([[ax, ay], [ax + 0.05 * R, (ay + gy) / 2], [cx * 0.3 + ax * 0.7, gy]]),
                                     np.array([0.12 * R, 0.09 * R, 0.14 * R]), None, True, "round")
            pool = np.maximum(pool, stream)
        self._blend(color, pool, opacity)
        sheen = self._poly_raster([], [(cx - 0.3 * R, gy - 0.05 * R, 0.09 * R)])
        self._blend(lerp(as_color(color), (1, 1, 1), 0.7), _f.gaussian_blur(sheen, 0.05 * R) * pool, 0.8)
        if name:
            _manifest.note(self.cv, "object", name, mask=pool)
        return pool

    def steam(self, at, size=0.15, n=3, color="#ffffff", opacity=0.8, seed=0, name=None):
        """Rising steam above the point ``at``: ``n`` tapered, wavy wisps ``size`` tall with a soft glow, readable on
        dark and light grounds (a faint cool edge). Returns their mask."""
        rng = np.random.default_rng(seed)
        ax, ay = self.px(*at)
        Hs = self.size(size)
        parts = []
        for i in range(int(n)):
            x0 = ax + (i - (n - 1) / 2) * 0.3 * Hs
            ph = rng.random() * 6
            ts = np.linspace(0, 1, 7)
            P = np.stack([x0 + 0.12 * Hs * np.sin(ts * 5 + ph) * (0.4 + ts), ay - Hs * ts], 1)
            parts.append(self._stroke_px(P, np.full(7, 0.11 * Hs), "both", True, "round"))
        m = np.maximum.reduce(parts)
        self._blend("#8da2b8", edge(m, max(self.size(0.003), 1.0), "outside"), 0.5 * opacity)
        self._blend(color, _windowed(m, 0.1 * Hs, lambda s_: _f.gaussian_blur(s_, 0.04 * Hs)), 0.35 * opacity)
        self._blend(color, m, opacity)
        if name:
            _manifest.note(self.cv, "effect", name, mask=m, see_through=True)
        return m

    def drops(self, at, n=5, size=0.025, spread=0.1, color="#4a9be0", seed=0, name=None):
        """``n`` teardrops of ``size`` scattered within ``spread`` of the point ``at`` (rain on glass, sweat, a
        dripping tap), each with a highlight. Returns their mask."""
        rng = np.random.default_rng(seed)
        cx, cy = self.px(*at)
        S, sp = self.size(size), self.size(spread)
        t = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        parts = []
        for _ in range(int(n)):
            x, y = cx + (rng.random() * 2 - 1) * sp, cy + (rng.random() * 2 - 1) * sp
            r = S * (0.7 + 0.5 * rng.random())
            pts = np.stack([x + r * np.sin(t) * np.abs(np.sin(t / 2)) ** 1.0, y - r * np.cos(t) * 1.3], 1)
            parts.append(self._poly_raster([pts]))
        m = np.maximum.reduce(parts)
        self._blend(color, m)
        self._blend("#ffffff", m * clamp01(1 - _windowed(m, S, lambda s_: _f.gaussian_blur(
            __import__("pixelcanvas.masks", fromlist=["shift"]).shift(s_, -0.25 * S, -0.25 * S), 0.15 * S))), 0.6)
        if name:
            _manifest.note(self.cv, "object", name, mask=m)
        return m

    def stain(self, mask, color="#9a4a1e", amount=0.5, scale=0.06, seed=0):
        """Irregular patches over ``mask``: rust (orange-brown), dust (pale grey), mud, mould, wear. ``amount``
        0-1 = how much of the shape they cover. Returns their alpha."""
        from .noise import fbm2d
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        if bb is None:
            return m
        y0, y1, x0, x1 = bb
        n = fbm2d(y1 - y0, x1 - x0, max(self.size(scale), 3.0), 4, seed=seed, offset=(x0, y0))
        thr = 0.25 - 0.5 * float(amount)
        a = smoothstep(thr, thr + 0.12, n) * m[y0:y1, x0:x1] * 0.85
        out = self._place((y0, y1, x0, x1), a.astype(F32))
        self._blend(color, out)
        return out

    # ---- layout
    def row(self, n, box=(0.05, 0.15, 0.95, 0.85), gap=0.15):
        """``n`` frames side by side inside ``box`` (x0, y0, x1, y1): use each as ``with sk.at(*f):`` and draw the
        thing within -0.5..0.5; every thing gets the same room and none overlap. Returns [(x, y, size), ...]."""
        return self.grid(n, box, cols=n, gap=gap)

    def grid(self, n, box=(0.05, 0.1, 0.95, 0.9), cols=None, gap=0.15):
        """``n`` evenly spaced frames inside ``box`` in rows of ``cols`` (default: cells as square as possible); an
        incomplete last row is centred. Use each as ``with sk.at(*f):`` and draw within -0.5..0.5."""
        x0, y0, x1, y1 = box
        f = self._f
        if f.root:
            bw, bh = (x1 - x0) * self.W, (y1 - y0) * self.H
        else:
            bw, bh = self.size(x1 - x0), self.size(y1 - y0)
        n = int(n)
        if cols is None:
            cols = max(1, min(n, int(round(math.sqrt(n * bw / max(bh, 1e-6))))))
        rows = int(math.ceil(n / float(cols)))
        cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
        size = min(bw / cols, bh / rows) * (1.0 - float(gap)) / f.unit
        out = []
        for i in range(n):
            r, c = divmod(i, cols)
            in_row = min(cols, n - r * cols)
            shift = (cols - in_row) / 2.0
            out.append((x0 + (c + shift + 0.5) * cw, y0 + (r + 0.5) * ch, size))
        return out

    # ---- soft cel (anime, cel-shaded games): defined tone bands, sheen, eyes, coloured line art
    def cel(self, mask, color, *, bands=1, shift=(-0.12, None), depth=0.4, softness=0.003, highlight=0.0,
            rim=None, bounce=None, line=None, line_width=0.005, weight=0.5, shade_color=None, angle=90,
            shadow=None, glow=None, opacity=1.0, name=None, kind="object"):
        """Soft-cel paint: the base colour, then ``bands`` (1 or 2) hard-edged shadow tones on the side away from
        the light. Each band edge follows the shape's form with a small ``softness`` (frame units); ``depth`` is
        the first band's reach as a share of the shape's radius (the second band reaches half as far).
        ``shift`` = (lightness, hue degrees) of the shadow tone in OKLab: (-0.12, None) = darker and turned
        towards violet (tools/match.py prints the reference's own pair); ``shade_color`` sets it directly.
        ``highlight`` 0–1: a hard light band inside the lit edge; ``rim``: True or a colour, a thin back-light
        on the shadow-side edge; ``bounce``: True or a colour, soft reflected light inside the shadow;
        ``line``: True (a dark tone of the base, never black) or a colour, ``line_width`` thick and heavier on
        the shadow side by ``weight``. ``color`` may be stops ``[(0, c0), (1, c1)]`` (``angle``, ``shadow``,
        ``glow``, ``opacity``, ``name`` as in paint). Returns the mask."""
        _choice("cel", "bands", int(bands), (0, 1, 2))
        ext = extent_of(mask)
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        if bb is None:
            return Shape(m, ext)
        self.paint(mask, color, angle=angle, shadow=shadow, glow=glow, opacity=opacity)
        base = self._fill_color(m, color, angle, bb)[1]
        area = float(m[bb[0]:bb[1], bb[2]:bb[3]].sum())
        R = math.sqrt(max(area, 1.0) / math.pi)
        lx, ly = self.light
        soft = max(self.size(softness), 0.6)
        from .masks import shift as _shift
        dl, dh = (tuple(shift) + (None,))[:2]
        dark = as_color(shade_color) if shade_color is not None else as_color(cel_tone(base, dl, dh, 1.12))

        def band(d, lit=False):
            sgn = -1.0 if lit else 1.0

            def fn(sub):
                moved = _f.gaussian_blur(_shift(sub, sgn * lx * d, sgn * ly * d), soft)
                return sub * (1.0 - smoothstep(0.35, 0.65, moved))
            return _windowed(m, d + 3 * soft + 3, fn)

        if bands >= 1:
            b1 = band(depth * R)
            self._blend(dark, b1, opacity)
            if bounce:
                bc = as_color(cel_tone(base, -0.02, dh, 1.25)) if bounce is True else as_color(bounce)
                w = 0.22 * R
                sub = _windowed(m, w + 6, lambda s: _f.gaussian_blur(
                    s * (1.0 - _shift(s, lx * w, ly * w)), max(0.12 * R, 1.0)) * s)
                self._blend(bc, sub * b1, 0.55 * opacity)
            if bands >= 2:
                dark2 = as_color(cel_tone(dark, (dl if dl is not None else -0.12) * 0.8, dh, 1.08))
                self._blend(dark2, band(depth * R * 0.45), opacity)
        if highlight:
            lite = as_color(cel_tone(base, 0.09, 0.0, 0.92))
            inset = 0.05 * R

            def fh(sub):
                inner = grow(sub, -inset)
                moved = _f.gaussian_blur(_shift(inner, -lx * 0.12 * R, -ly * 0.12 * R), soft)
                return inner * (1.0 - smoothstep(0.35, 0.65, moved))
            self._blend(lite, _windowed(m, 0.2 * R + 4, fh), min(float(highlight), 1.0) * opacity)
        if rim:
            rc = as_color(cel_tone(base, 0.16, 0.0, 0.7)) if rim is True else as_color(rim)
            self._blend(rc, band(max(0.06 * R, 1.2)), 0.9 * opacity)
        if line is not None and line is not False:
            lc = as_color(cel_tone(base, -0.4, dh, 0.8)) if line is True else as_color(line)
            self._blend(lc, self._weighted_line(m, self.size(line_width), weight), opacity)
        if name:
            _manifest.note(self.cv, kind, name, mask=m, extent=ext)
        return Shape(m, ext)

    def _weighted_line(self, m, w, weight=0.5):
        """An outline of mean width ``w`` px, thinner on the lit side and heavier on the shadow side."""
        from .masks import shift as _shift
        lx, ly = self.light
        s = float(weight) * w

        def fn(sub):
            outer = grow(sub, w / 2.0)
            inner = _shift(grow(sub, -w / 2.0), lx * s, ly * s)
            return clamp01(outer - np.minimum(inner, sub))
        return _windowed(m, w + s + 4, fn)

    def ink(self, points, width=0.006, color="#3b2633", taper="both", shadow=None, glow=None, opacity=1.0,
            name=None, kind="object"):
        """Coloured line art: a smooth stroke through ``points`` that swells in the middle and tapers to points
        (``taper`` = both / start / end / None), in a dark coloured ink rather than black; ``shadow``, ``glow``,
        ``opacity``, ``name`` as in paint. Returns the mask."""
        return self.paint(self.stroke(points, width, taper=taper), color, shadow=shadow, glow=glow,
                          opacity=opacity, name=name, kind=kind)

    def ramp(self, mask, colors, angle=90, steps=0, line=None, line_width=0.005, shadow=None, glow=None,
             opacity=1.0, name=None, kind="object"):
        """A gradient fill mixed in OKLab through ``colors`` (evenly spaced, or stops [(t, c), ...]) across the
        shape: skies, hair, cloth. ``angle`` 90 = top to bottom, 0 = left to right. ``steps`` > 1 cuts it into
        that many hard bands (anime skies, cel hair). ``line``, ``shadow``, ``glow``, ``opacity``, ``name`` as in
        paint. Returns the mask."""
        from .palette import srgb_to_oklab, oklab_to_srgb
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        if bb is None:
            return Shape(m, extent_of(mask))
        if colors and isinstance(colors[0], (list, tuple)) and len(colors[0]) == 2 and \
                not isinstance(colors[0][0], (list, tuple)) and isinstance(colors[0][1], (str, tuple, list)):
            ts = [float(t) for t, _ in colors]
            cs = [c for _, c in colors]
        else:
            cs = list(colors)
            ts = list(np.linspace(0.0, 1.0, len(cs))) if len(cs) > 1 else [0.0]
        labs = srgb_to_oklab(np.array([as_color(c)[:3] for c in cs], F32))
        y0, y1, x0, x1 = bb
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(F32)
        a = math.radians(self._angle(angle))
        dx, dy = math.cos(a), math.sin(a)
        cx, cy, hw, hh = (x0 + x1) / 2.0, (y0 + y1) / 2.0, max((x1 - x0) / 2.0, 1), max((y1 - y0) / 2.0, 1)
        ext = max(abs(dx) * hw + abs(dy) * hh, 1e-6)
        t = clamp01(((xx - cx) * dx + (yy - cy) * dy) / ext * 0.5 + 0.5)
        n = int(steps)
        if n > 1:
            f = t * n
            i = np.minimum(np.floor(f), n - 1)
            w = n / (2.0 * ext)                                    # one pixel in band units: anti-aliased steps
            s = clamp01((f - i - 1.0 + w) / w) * (i < n - 1)
            t = (i + s) / (n - 1)
        lab = np.stack([np.interp(t, ts, labs[:, k]) for k in range(3)], -1)
        full = np.zeros((self.H, self.W, 3), F32)
        full[y0:y1, x0:x1] = oklab_to_srgb(lab)
        return self.paint(mask, full, line=line, line_width=line_width, shadow=shadow, glow=glow, opacity=opacity,
                          name=name, kind=kind)

    def sheen(self, mask, color=None, y=0.3, width=0.035, teeth=9, bend=0.22, opacity=0.85):
        """Hair sheen (the angel ring): a light band with jagged teeth across ``mask`` at ``y`` (0 = top of the
        mask's box, 1 = bottom) that dips at the sides like a ring round the head. ``width`` = band height
        (frame units), ``teeth`` = spikes across, ``bend`` = how far the ends dip (share of the box height).
        ``color`` defaults to a light, less saturated tone of what is painted under the mask. Returns its alpha."""
        m = np.asarray(mask, F32)
        bb = _bbox(m)
        if bb is None:
            return m
        y0, y1, x0, x1 = bb
        if color is None:
            wm = m[y0:y1, x0:x1, None]
            under = (np.clip(self.cv.buf[y0:y1, x0:x1, :3], 0, 1) * wm).sum((0, 1)) / max(float(wm.sum()), 1e-6)
            color = cel_tone(under, 0.2, 0.0, 0.75)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(F32) + 0.5
        h = float(y1 - y0)
        u = (xx - (x0 + x1) / 2.0) / max((x1 - x0) / 2.0, 1.0)
        vc = y0 + h * (float(y) + float(bend) * u * u)
        hw = max(self.size(width) / 2.0, 1.0)
        tri = np.abs(2.0 * ((u + 1.0) * float(teeth) / 2.0 % 1.0) - 1.0)        # 0..1 triangle wave across
        lower = vc + hw * (0.2 + 2.6 * (1.0 - tri) ** 1.5)                 # long teeth hanging down
        upper = vc - hw * (0.8 + 0.3 * tri)
        al = clamp01(yy - upper + 0.5) * clamp01(lower - yy + 0.5) * (1.0 - smoothstep(0.7, 1.0, np.abs(u)))
        al = self._place((y0, y1, x0, x1), al.astype(F32)) * grow(m, -1.5)
        self._blend(color, al, opacity)
        return al

    def blush(self, x, y, rx=0.035, ry=None, color="#ff7d8c", opacity=0.45, lines=3, clip=None):
        """Anime blush: a soft gradient oval plus ``lines`` short diagonal hatch strokes. Returns its alpha."""
        ry = rx * 0.55 if ry is None else ry
        al = self.spot(x, y, rx, ry, color, opacity, clip=clip)
        n = int(lines)
        for i in range(n):
            ox = x + (i - (n - 1) / 2.0) * rx * 0.55
            m = self.stroke([(ox + rx * 0.12, y - ry * 0.35), (ox - rx * 0.12, y + ry * 0.35)],
                            max(rx * 0.09, 0.0015), taper="both")
            self._blend(cel_tone(color, -0.08, 0.0, 1.1), m * (np.asarray(clip, F32) if clip is not None else 1.0),
                        min(1.0, opacity * 1.5))
        return al

    def eye(self, x, y, size=0.06, iris="#3f7fd0", look=(0.0, 0.0), lash=None, white="#fbf8ff", mirror=False,
            lid=0.0, name=None):
        """A multi-tone anime eye in one call, ``size`` = its height: sclera with the upper lid's shadow, a tall
        iris graded dark (top) to light (bottom) with a darker rim, pupil, lid shadow across the iris, a light
        crescent low in the iris, two highlights on the lit side, a heavy tapered upper lash line and a light lower
        one. The outer corner is on the right; draw the left-hand eye with ``mirror=True``. ``look`` = (dx, dy)
        moves the iris (fractions of the eye, about ±0.15); ``lid`` 0–0.6 lowers the upper lid (sleepy, smiling);
        ``lash`` defaults to a very dark tone of the iris. Returns the eye's mask."""
        lash = cel_tone(iris, -0.62, 0.0, 0.45) if lash is None else lash
        lx, ly = float(look[0]), float(look[1])
        lit_x = self.light[0] * (-1.0 if mirror else 1.0)
        top = -0.42 + 0.7 * float(lid)
        with self.at(x, y, size, mirror=mirror):
            outline = [(-0.36, 0.12), (-0.3, top + 0.12), (0.0, top), (0.32, top + 0.08), (0.44, 0.0),
                       (0.3, 0.38), (-0.18, 0.4)]
            white_m = self.blob(outline)
            self._blend(white, white_m)
            from .masks import shift as _shift
            d = self.size(0.2)
            lid_sh = white_m * (1.0 - _shift(white_m, 0.0, d))
            self._blend(cel_tone(white, -0.16, 0.0, 1.0) if as_color(white).mean() < 0.9 else "#c9c6e6", lid_sh)
            ix, iy = 0.02 + lx, 0.02 + ly
            iris_e = self.ellipse(ix, iy, 0.25, 0.37)
            ir = iris_e * white_m
            self.ramp(ir, [cel_tone(iris, -0.24, 0.0, 1.0), iris, cel_tone(iris, 0.14, 0.0, 1.15)])
            self._blend(cel_tone(iris, -0.3, 0.0, 0.9), self.outline(iris_e, 0.045, "inside") * white_m)
            self._blend(cel_tone(iris, -0.42, 0.0, 0.8), self.ellipse(ix, iy + 0.04, 0.1, 0.16) * white_m)
            self._blend(cel_tone(iris, -0.3, 0.0, 0.9), ir * (1.0 - _shift(white_m, 0.0, self.size(0.24))), 0.75)
            self.spot(ix, iy + 0.24, 0.17, 0.09, cel_tone(iris, 0.3, 0.0, 1.1), 0.85, clip=ir)
            self._blend("#ffffff", self.ellipse(ix + 0.1 * lit_x, iy - 0.14, 0.085, 0.1, angle=-20) * white_m)
            self._blend("#ffffff", self.circle(ix - 0.1 * lit_x, iy + 0.16, 0.04) * white_m, 0.9)
            upper = self.stroke([(-0.38, top + 0.2), (-0.28, top + 0.1), (0.0, top - 0.02), (0.3, top + 0.06),
                                 (0.48, top + 0.2), (0.56, top + 0.3)], [0.035, 0.08, 0.11, 0.1, 0.07, 0.02])
            flick = self.stroke([(0.42, top + 0.12), (0.56, top + 0.04)], [0.05, 0.01])
            self._blend(lash, np.maximum(upper, flick))
            self._blend(lash, self.stroke([(0.06, 0.41), (0.26, 0.38), (0.4, 0.22)], [0.012, 0.03, 0.012]), 0.8)
        if name:
            _manifest.note(self.cv, "object", name, mask=white_m, extent=extent_of(white_m))
        return white_m

    # ---- groups
    @contextlib.contextmanager
    def layer(self, name=None, kind="object", opacity=1.0, paste=True):
        """Collect the shapes painted inside into one layer; ``g.mask`` is their joint coverage afterwards (for a
        die-cut, a group shadow, a clip). Composited on exit (``paste=False`` keeps it for you: ``g.rgb``)."""
        L = _Layer(self.H, self.W)
        self._dst.append(L)
        try:
            yield L
        finally:
            self._dst.pop()
        if paste:
            self._blend(L.rgb, L.A, opacity)
        if self._dst:                                     # nested group: the outer one spans this one too
            for O in self._dst:
                O.ext = _union_box(O.ext, L.ext)
        if name:
            _manifest.note(self.cv, kind, name, mask=L.A, extent=L.ext)

    @contextlib.contextmanager
    def sticker(self, border=0.03, color="#ffffff", line=None, line_width=0.004, ink=None, ink_width=0.006,
                shadow=True, name=None, kind="object"):
        """Draw a sticker: the shapes painted inside get an optional ``ink`` outline hugging the art, a die-cut
        ``border`` (smoothed around gaps), an optional ``line`` around the border and a soft drop shadow."""
        with self.layer(paste=False) as L:
            yield L
        art = L.A
        if ink:
            art = grow(L.A, self.size(ink_width))
        b = self.size(border)
        close = b * 0.8
        outer = grow(grow(art, b + close), -close)
        if shadow:
            from .masks import shift
            blur = self.size(0.012)
            off = self.size(0.012)
            s = _windowed(outer, off + 3 * blur + 2,
                          lambda sub: _f.gaussian_blur(shift(sub, -self.light[0] * off, -self.light[1] * off), blur))
            self._blend("#2a2440", s, 0.28 if shadow is True else float(shadow))
        self._blend(color, outer)
        if line:
            self._blend(line, edge(outer, self.size(line_width), "inside"))
        if ink:
            self._blend(ink, art)
        self._blend(L.rgb, L.A)
        if name:
            _manifest.note(self.cv, kind, name, mask=outer,
                           extent=_pad_box(L.ext, b + (self.size(ink_width) if ink else 0.0)))


# ----------------------------------------------------------------------------------------------- finishing

def cel_tone(color, light=-0.12, hue=None, chroma=1.0):
    """A related tone of ``color`` in OKLab -> '#rrggbb': ``light`` added to the lightness (-0.12 = a cel shadow,
    +0.1 = a highlight), ``hue`` degrees turned (+ = counter-clockwise: red -> yellow -> green -> blue; None =
    up to 15 degrees towards violet, the usual anime shadow), ``chroma`` multiplied. Near-greys get a cool tint
    when darkened, so white cloth shades lavender."""
    from .palette import srgb_to_oklab, oklab_to_srgb, to_hex
    lab = srgb_to_oklab(np.clip(np.asarray(as_color(color), F32)[:3], 0, 1)).astype(np.float64)
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    C = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360
    dl = -0.12 if light is None else float(light)
    if hue is None:
        turn = ((300.0 - h + 180.0) % 360.0) - 180.0
        dh = max(-15.0, min(15.0, turn)) if dl < 0 else 0.0
    else:
        dh = float(hue)
    C2 = C * float(chroma)
    if C < 0.03 and dl < 0:                                 # greys and whites: a cool lavender shadow
        C2, h, dh = max(C2, min(0.045, -dl * 0.35)), 290.0, 0.0
    h2 = math.radians(h + dh)
    out = oklab_to_srgb(np.array([min(max(L + dl, 0.0), 1.0), C2 * math.cos(h2), C2 * math.sin(h2)]))
    return to_hex(out)


def paper(cv, color=None, grain=0.03, fibres=0.02, blotch=0.02, seed=0):
    """Paper under the picture: tints towards ``color`` (optional), then multiplies fine grain, horizontal
    fibres and large soft blotches. Call it near the end (before ``grade``)."""
    H, W = cv.h, cv.w
    rng = np.random.default_rng(seed)
    S = H / 1080.0
    g = _f.gaussian_blur(rng.standard_normal((H, W)).astype(F32), 0.7 * S)
    small = rng.standard_normal((H, max(W // 12, 2))).astype(F32)
    fib = _f.resize(_f.gaussian_blur(small, 0.6), W, H, "bilinear")
    bl = _f.resize(rng.standard_normal((max(H // 60, 2), max(W // 60, 2))).astype(F32), W, H, "bicubic")
    k = 1.0 + float(grain) * g + float(fibres) * fib + float(blotch) * bl
    if color is not None:
        cv.paint(color, None, "multiply", 0.12)
    cv.buf = (cv.buf * k[..., None]).astype(F32)
    return cv


def limit_palette(cv, colors, amount=1.0, mask=None):
    """Pull every pixel towards its nearest colour of ``colors`` (``amount`` 1 = exactly the palette): the
    limited inks of a woodblock print, a screen print or pixel art."""
    pal = np.stack([as_color(c) for c in colors])
    buf = cv.buf
    out = np.empty_like(buf)
    step = max(1, 400000 // max(buf.shape[1], 1))
    for y in range(0, buf.shape[0], step):
        blk = buf[y:y + step]
        d = ((blk[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
        out[y:y + step] = pal[np.argmin(d, -1)]
    if mask is None:
        cv.paint(out, None, opacity=float(amount))
    else:
        cv.paint(out, np.asarray(mask, F32) * F32(amount))
    return cv


MOODS = {
    "warm night": dict(shadows="#1b2350", highlights="#ffb766", tone=0.16, vignette=0.45, contrast=0.12, sat=1.08),
    "golden": dict(shadows="#5a3a2a", highlights="#ffd28a", tone=0.12, vignette=0.3, contrast=0.08, sat=1.05),
    "cosy": dict(shadows="#4a2a20", highlights="#ffe0b0", tone=0.1, vignette=0.38, contrast=0.06, sat=1.04),
    "cool day": dict(shadows="#2a4060", highlights="#f2f6ff", tone=0.08, vignette=0.18, contrast=0.06, sat=1.0),
    "moody": dict(shadows="#10182a", highlights="#c0d0e0", tone=0.14, vignette=0.55, contrast=0.16, sat=0.85),
    "print": dict(shadows="#20283a", highlights="#f4e6c8", tone=0.1, vignette=0.22, contrast=0.04, sat=0.92),
}


def grade(cv, mood="warm night", amount=1.0):
    """One-call art direction at the end of a picture: split tone, contrast, saturation and vignette for a
    ``mood`` (``MOODS``: warm night, golden, cosy, cool day, moody, print), scaled by ``amount``."""
    from ._args import unknown_name
    if mood not in MOODS:
        raise unknown_name("mood", mood, MOODS)
    p = MOODS[mood]
    k = float(amount)
    cv.split_tone(p["shadows"], p["highlights"], p["tone"] * k)
    cv.contrast(p["contrast"] * k)
    cv.saturation(1.0 + (p["sat"] - 1.0) * k)
    cv.vignette(p["vignette"] * k)
    return cv


_strict_methods(Sketch, skip=("px", "size"))
for _n in ("grow", "edge", "paper", "limit_palette", "grade", "signed_distance", "cel_tone"):
    globals()[_n] = _strict(globals()[_n])

from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "rect": "use sk.box(x0, y0, x1, y1, radius=0)",
    "rectangle": "use sk.box(x0, y0, x1, y1, radius=0)",
    "line": "use sk.stroke(points, width) (smooth, variable width, taper=)",
    "path": "use sk.stroke(points, width) for a line, sk.blob(points) for a closed smooth shape",
    "bezier": "use sk.blob(points) / sk.stroke(points): smooth curves through the points",
    "shadow": "use sk.paint(mask, color, shadow=True) or sk.spot(x, y, rx, ry, '#000', 0.3) for a contact shadow",
    "gradient": "use sk.paint(mask, [(0, c0), (1, c1)], angle=90)",
    "outline": "use sk.paint(mask, color, line=True) or sk.outline(mask, width)",
}
__getattr__ = _module_getattr(__name__, _HINTS)
