# -*- coding: utf-8 -*-
"""Object form kit: make flat shapes look like solid objects standing in a lit space.

Five building blocks, each a one-liner:

* **Local frames** — ``Frame(cv, center=(u, v), size=0.2, rotate=30)`` gives local grids ``fr.X, fr.Y`` (unit = size)
  for the ``pixelcanvas.sdf`` primitives; write a shape once, then place / rotate / scale / mirror it anywhere.
  Only the frame's bounding box is evaluated, so hundreds of small shapes stay fast.
* **Volume shading** — ``paint_form(cv, d, base, material="ceramic")`` turns any signed distance into a lit volume
  (diffuse, wrap, specular, window reflections, rim, bounce, subsurface) with 15 material presets.
* **Shadows** — ``cast_shadow`` (projected, penumbra widening with distance), ``contact_shadow``, ``drop_shadow``,
  ``occlusion_between``. Shadows multiply the colours underneath.
* **Perspective** — ``Ground`` (floor plane: placement, checker / tiles / planks in perspective), ``Room`` (one-point
  interior with window light), ``tabletop`` / ``shelf`` / ``counter`` surfaces for still lifes.
* **2.5D solids** — ``ball``, ``cuboid``, ``cylinder``, ``cone``, ``capsule``, ``torus``, ``lathe`` (bottles, vases,
  mugs, lamps): shaded volumes with correct faces, contact shadows and anchor points.
* **Arrangement** — ``row``, ``grid_positions``, ``scatter_positions``, ``ring_positions`` with depth-sorted order.

Units: positions are fractions of the canvas ``(u, v)`` (u of the width, v of the height); sizes are fractions of the
canvas height. A ``Spot`` returned by ``ground.place`` / ``surface.spot`` carries a perspective scale ``k`` and view
tilt, so sizes given at a Spot are scaled automatically (deeper = smaller). Masks and anchors come back in pixels.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .color import F32, rgb, as_color, clamp01, smoothstep
from . import sdf as _sdf
from . import filters as _f
from .noise import fbm2d, fbm2d_at
from ._args import (reject_unknown as _reject_unknown, unknown_name as _unknown_name,
                    signature_options as _sig_options, strict as _strict, strict_methods as _strict_methods)

__all__ = [
    "P", "px", "Spot", "Frame", "d_circle", "d_ellipse", "d_box", "d_segment", "d_polygon",
    "Light", "set_light", "get_light", "MATERIALS", "shade", "paint_form", "normals",
    "cast_shadow", "contact_shadow", "drop_shadow", "occlusion_between", "reflection",
    "Ground", "Room", "Surface", "tabletop", "shelf", "counter",
    "Solid", "ball", "cuboid", "cylinder", "cone", "capsule", "torus", "lathe", "disc", "book",
    "row", "grid_positions", "scatter_positions", "ring_positions",
    "resolve_light", "shadow_color", "multiply_shadow", "contact_shadow_px", "clip_box", "full_mask", "bbox_of",
    "full_frame", "grid_box", "sd_ellipse", "contact_px", "multiply", "check_frac", "guard_fractions",
]

VIEW_TILT = 0.28          # default view tilt for solids (sin of the angle we look down at them)

# ===================================================================== small helpers


def _grid_box(x0, y0, x1, y1):
    X, Y = np.meshgrid(np.arange(x0, x1, dtype=F32), np.arange(y0, y1, dtype=F32))
    return X, Y


def _clip_box(cv, x0, y0, x1, y1):
    H, W = cv.h, cv.w
    a, b = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
    c, d = min(int(math.ceil(x1)), W), min(int(math.ceil(y1)), H)
    if a >= c or b >= d:
        return None
    return a, b, c, d


def _full(cv, a, box, fill=0.0):
    out = np.full((cv.h, cv.w), fill, F32)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = a
    return out


def _crop_color(c, box, full_shape):
    c = as_color(c)
    if c.ndim == 3:
        x0, y0, x1, y1 = box
        if c.shape[:2] == tuple(full_shape):
            return c[y0:y1, x0:x1]
        if c.shape[:2] == (y1 - y0, x1 - x0):
            return c
        raise ValueError("colour map must match the canvas or the crop size")
    return c


def _bilinear(img, X, Y):
    """Bilinear sample of a 2D or 3D array at float pixel coordinates (edges clamp); numpy only."""
    h, w = img.shape[:2]
    X = np.clip(np.asarray(X, F32), 0, w - 1)
    Y = np.clip(np.asarray(Y, F32), 0, h - 1)
    x0 = np.minimum(np.floor(X).astype(np.int32), w - 2) if w > 1 else np.zeros(X.shape, np.int32)
    y0 = np.minimum(np.floor(Y).astype(np.int32), h - 2) if h > 1 else np.zeros(Y.shape, np.int32)
    fx, fy = X - x0, Y - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(F32)


def _bbox_of(a, thresh):
    rows = np.flatnonzero((a > thresh).any(1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero((a[rows[0]:rows[-1] + 1] > thresh).any(0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def clip_box(cv, x0, y0, x1, y1):
    """Pixel box ``(x0, y0, x1, y1)`` clipped to the canvas as ints, or ``None`` when it lies outside."""
    return _clip_box(cv, x0, y0, x1, y1)


def full_mask(cv, a, box, fill=0.0):
    """A crop-sized array placed at ``box`` in a full-frame (H, W) array."""
    return _full(cv, a, box, fill)


def bbox_of(mask, thresh=0.004):
    """Tight pixel box ``(x0, y0, x1, y1)`` of ``mask > thresh``, or ``None`` for an empty mask."""
    return _bbox_of(np.asarray(mask), thresh)


full_frame = full_mask


def grid_box(x0, y0, x1, y1):
    """Pixel coordinate grids ``(X, Y)`` (float32) of the box ``[x0, x1) x [y0, y1)``."""
    return _grid_box(x0, y0, x1, y1)


def _texture_or_color(option, v, names):
    """Raise a ValueError naming the closest texture when a string is neither one of ``names`` nor a colour."""
    if isinstance(v, str) and v not in names:
        try:
            as_color(v)
        except ValueError:
            raise _unknown_name(option, v, list(names)) from None
    return v


def _hash01(*ints, seed=0):
    """Deterministic integer hash → float in [0, 1)."""
    with np.errstate(over="ignore"):
        h = np.uint32((int(seed) * 2654435761 + 0x9E3779B9) & 0xFFFFFFFF)
        mul = (np.uint32(0x85EBCA6B), np.uint32(0xC2B2AE35), np.uint32(0x27D4EB2F))
        for k, v in enumerate(ints):
            v = np.asarray(v).astype(np.int64).astype(np.uint32)
            h = (h ^ (v * mul[k % 3])).astype(np.uint32)
            h = h ^ (h >> np.uint32(15))
            h = (h * np.uint32(0x2C1B3C6D)).astype(np.uint32)
            h = h ^ (h >> np.uint32(12))
    return (np.asarray(h, np.uint32) & np.uint32(0xFFFFFF)).astype(F32) / F32(16777216.0)


class Spot(tuple):
    """A placement ``(u, v)`` that also carries a perspective scale ``k`` and the view ``tilt`` at that point.

    Solids placed at a Spot multiply their sizes by ``k`` (deeper → smaller) and take its tilt. A Spot from a
    table, shelf, room or ground also remembers that surface, so cast shadows stay on it (and climb the wall
    behind a shelf or the back wall of a room).
    """

    def __new__(cls, u, v, k=1.0, tilt=None, surface=None, lift=0.0, hw=None):
        obj = tuple.__new__(cls, (float(u), float(v)))
        obj.k = float(k)
        obj.tilt = tilt
        obj.surface = surface
        obj.lift = float(lift)
        obj.hw = hw                  # canvas height / width (for moves measured in size units)
        return obj

    def up(self, h):
        """The same spot raised by ``h`` (size units) — to stack objects or stand them on something."""
        ce = math.sqrt(1 - min(self.tilt if self.tilt is not None else VIEW_TILT, 0.95) ** 2)
        return Spot(self[0], self[1] - h * self.k * ce, self.k, self.tilt, self.surface, self.lift + h * self.k * ce,
                    self.hw)

    def move(self, dx, dz=0.0, yaw=0.0):
        """The spot moved on the ground by ``dx`` (right) and ``dz`` (towards the viewer) size units, turned by
        ``yaw`` degrees like a ``cuboid``: ``body.move(0.3, -0.12, yaw=30)`` is a wheel position of a car."""
        a = math.radians(yaw)
        se = min(self.tilt if self.tilt is not None else VIEW_TILT, 0.95)
        x = dx * math.cos(a) - dz * math.sin(a)
        z = dx * math.sin(a) + dz * math.cos(a)
        hw = self.hw if self.hw is not None else 9 / 16
        return Spot(self[0] + x * self.k * hw, self[1] + z * self.k * se, self.k, self.tilt, self.surface, self.lift,
                    self.hw)

    @property
    def u(self):
        return self[0]

    @property
    def v(self):
        return self[1]


def P(cv, u, v):
    """Fractions → pixels: ``P(cv, 0.5, 0.5)`` is the canvas centre ``(x, y)``."""
    return u * cv.w, v * cv.h


def px(cv, f):
    """A size given as a fraction of the canvas height → pixels."""
    return f * cv.h


def check_frac(owner, option, value, limit=3.0):
    """Return ``value``; raise ValueError when a canvas-fraction position is clearly pixels (any number beyond
    ``limit``). Positions are fractions of the canvas: ``(0.4, 0.9)`` is 40% across and 90% down."""
    try:
        nums = [float(v) for v in np.asarray(value, dtype=float).ravel()]
    except (TypeError, ValueError):
        return value
    if any(abs(v) > limit for v in nums):
        raise ValueError(
            "%s: %s=%s looks like pixels. Positions are fractions of the canvas, (0..1, 0..1): (0.4, 0.9) is 40%% "
            "across and 90%% down. Divide pixels by the canvas size: (x / cv.w, y / cv.h)."
            % (owner, option, tuple(round(v, 1) for v in nums)))
    return value


_FRAC_NAMES = ("at", "region", "box", "start", "end", "center", "path", "points", "horizon", "v_range", "fit", "target",
               "a", "b")


def guard_fractions(fn, owner=None, names=_FRAC_NAMES):
    """Wrap ``fn`` so that a position argument given in pixels (``at=(640, 360)``) raises the ValueError of
    ``check_frac`` instead of drawing nothing. Only arguments named in ``names`` are checked."""
    import functools
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn
    present = [n for n in names if n in sig.parameters]
    if not present:
        return fn
    label = owner or getattr(fn, "__qualname__", getattr(fn, "__name__", "call"))

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            bound = sig.bind_partial(*a, **k).arguments
        except TypeError:
            return fn(*a, **k)
        for n in present:
            if n in bound and bound[n] is not None and not hasattr(bound[n], "shape"):
                check_frac(label, n, bound[n])
        return fn(*a, **k)
    return wrapper


def _pos(cv, at):
    """``at`` (u, v) or Spot → (x px, y px, k, tilt)."""
    check_frac("position", "at", (at[0], at[1]))
    k = getattr(at, "k", 1.0)
    tilt = getattr(at, "tilt", None)
    return at[0] * cv.w, at[1] * cv.h, k, tilt


# ---- distance helpers in fractions (full-frame px distances)

def sd_ellipse(X, Y, cx, cy, rx, ry, angle=0.0):
    """Approximate signed pixel distance to an ellipse on the grids ``X, Y`` (negative inside)."""
    return _sd_ellipse(X, Y, cx, cy, rx, ry, angle)


def _sd_ellipse(X, Y, cx, cy, rx, ry, angle=0.0):
    """sdf.sd_ellipse that is also correct at the exact centre."""
    d = _sdf.sd_ellipse(X, Y, cx, cy, rx, ry, angle)
    x, y = X - cx, Y - cy
    near = (x * x + y * y) < 1e-6
    return np.where(near, -float(min(rx, ry)), d).astype(F32)


def _XY(cv):
    return cv.xy()


def d_circle(cv, center, r):
    """Signed px distance to a circle; centre (u, v), radius as a fraction of the canvas height."""
    X, Y = _XY(cv)
    return _sdf.sd_circle(X, Y, center[0] * cv.w, center[1] * cv.h, r * cv.h)


def d_ellipse(cv, center, rx, ry, angle=0.0):
    """Signed px distance to an ellipse; radii as fractions of the canvas height."""
    X, Y = _XY(cv)
    return _sd_ellipse(X, Y, center[0] * cv.w, center[1] * cv.h, rx * cv.h, ry * cv.h, angle)


def d_box(cv, center, half, radius=0.0, angle=0.0):
    """Signed px distance to a (rounded) box; ``half=(hw, hh)`` and ``radius`` as fractions of the height."""
    X, Y = _XY(cv)
    return _sdf.sd_box(X, Y, center[0] * cv.w, center[1] * cv.h, half[0] * cv.h, half[1] * cv.h, radius * cv.h, angle)


def d_segment(cv, a, b, r=0.0):
    """Signed px distance to a capsule from point a to b (fractions), radius as a fraction of the height."""
    X, Y = _XY(cv)
    return _sdf.sd_segment(X, Y, a[0] * cv.w, a[1] * cv.h, b[0] * cv.w, b[1] * cv.h, r * cv.h)


def d_polygon(cv, points):
    """Signed px distance to a polygon given as fraction points ``[(u, v), ...]``."""
    X, Y = _XY(cv)
    return _sdf.sd_polygon(X, Y, [(u * cv.w, v * cv.h) for u, v in points])


# ===================================================================== local frames


class Frame:
    """A local coordinate frame: write a shape once in local units, place it anywhere.

    ``Frame(cv, center=(u, v), size=0.2, rotate=0, flip=False, squash=(1, 1), extent=1.5)``

    * one local unit = ``size`` × canvas height; local y points down like the screen; ``rotate`` is clockwise degrees;
      ``flip`` mirrors the drawing left-right (after rotation); ``squash`` scales the local x / y axes;
      ``flat=tilt`` (or ``True`` at a Spot) lays the frame flat on the ground seen at that tilt (spoons, papers, rugs on a table).
    * ``fr.X, fr.Y`` are local coordinate grids covering only the frame's box (local ``[-extent, extent]²``, or
      ``extent=(x0, y0, x1, y1)``), so evaluation cost depends on the object size, not the canvas.
    * ``fr.px(d)`` converts a local distance to pixels; ``fr.fill(d)`` / ``fr.paint(color, d)`` / ``fr.form(d, base)``
      give a full-frame mask, a flat fill or a shaded volume.
    """

    def __init__(self, cv, center=(0.5, 0.5), size=0.2, rotate=0.0, flip=False, squash=(1.0, 1.0), extent=1.5,
                 flat=None):
        self.cv = cv
        self.center = (float(center[0]), float(center[1]))
        self.cx, self.cy, k, tilt = _pos(cv, center)
        if flat is True:
            flat = tilt if tilt is not None else VIEW_TILT
        self.flat = float(flat) if flat else 1.0
        self.size = float(size) * k
        self.s = self.size * cv.h
        self.rotate = float(rotate)
        self.flip = bool(flip)
        self.squash = (float(squash[0]), float(squash[1])) if np.ndim(squash) else (float(squash),) * 2
        self.extent = extent
        if np.ndim(extent) == 0:
            e = float(extent)
            ext = (-e, -e, e, e)
        else:
            ext = tuple(float(v) for v in extent)
        xs, ys = zip(*[self.to_canvas(x, y) for x in (ext[0], ext[2]) for y in (ext[1], ext[3])])
        self.box = _clip_box(cv, min(xs) - 2, min(ys) - 2, max(xs) + 3, max(ys) + 3) or (0, 0, 1, 1)
        self._XY = None

    # ---- transforms
    def _cs(self):
        a = math.radians(self.rotate)
        return math.cos(a), math.sin(a)

    def to_canvas(self, x, y):
        """Local → canvas pixels (works on scalars or arrays)."""
        c, s = self._cs()
        lx = np.asarray(x, np.float64) * self.squash[0]
        ly = np.asarray(y, np.float64) * self.squash[1]
        rx = (c * lx - s * ly) * (-1.0 if self.flip else 1.0)
        ry = (s * lx + c * ly) * self.flat
        px_, py_ = self.cx + self.s * rx, self.cy + self.s * ry
        if np.ndim(px_) == 0:
            return float(px_), float(py_)
        return px_, py_

    def from_canvas(self, px_, py_):
        """Canvas pixels → local coordinates."""
        c, s = self._cs()
        dx = (np.asarray(px_, F32) - self.cx) / self.s * (-1.0 if self.flip else 1.0)
        dy = (np.asarray(py_, F32) - self.cy) / (self.s * self.flat)
        x = (c * dx + s * dy) / self.squash[0]
        y = (-s * dx + c * dy) / self.squash[1]
        if np.ndim(x) == 0:
            return float(x), float(y)
        return x.astype(F32), y.astype(F32)

    def uv(self, x, y):
        """Local → canvas fractions (u, v)."""
        a, b = self.to_canvas(x, y)
        return a / self.cv.w, b / self.cv.h

    def child(self, at=(0.0, 0.0), size=1.0, rotate=0.0, flip=False, squash=None, extent=1.5):
        """A frame attached to this one: ``at`` in local units, ``size`` relative, ``rotate`` relative degrees.

        The child moves, turns and mirrors with its parent. ``flip=True`` mirrors the child's own shape (for the
        left/right twin of a limb pass the mirrored attach point too: ``at=(-x, y)``)."""
        px_, py_ = self.to_canvas(*at)
        rot = rotate - self.rotate if flip else self.rotate + rotate
        return Frame(self.cv, (px_ / self.cv.w, py_ / self.cv.h), self.size * size, rot,
                     self.flip ^ bool(flip), squash or self.squash, extent, self.flat if self.flat != 1 else None)

    # ---- grids & conversion
    @property
    def X(self):
        return self._grids()[0]

    @property
    def Y(self):
        return self._grids()[1]

    def _grids(self):
        if self._XY is None:
            Xc, Yc = _grid_box(*self.box)
            x, y = self.from_canvas(Xc + F32(1e-3), Yc + F32(1e-3))   # never exactly on a shape centre
            self._XY = (x, y)
        return self._XY

    def px(self, d):
        """Local distance → pixels."""
        return np.asarray(d, F32) * F32(self.s * min(self.squash) * math.sqrt(self.flat))

    def full(self, a, fill=0.0):
        """Crop-sized array → full-frame (H, W) array."""
        return _full(self.cv, a, self.box, fill)

    def fill(self, d, aa=1.0):
        """Anti-aliased full-frame mask of a local distance field."""
        return self.full(_sdf.fill(self.px(d), aa))

    mask = fill

    def paint(self, color, d, opacity=1.0, mode="normal", aa=1.0):
        """Flat fill of a local shape (colour or full-frame / crop colour map). Returns the full-frame mask."""
        a = _sdf.fill(self.px(d), aa)
        x0, y0, x1, y1 = self.box
        col = _crop_color(color, self.box, (self.cv.h, self.cv.w))
        if col.ndim == 1:
            col = np.broadcast_to(col, a.shape + (3,))
        self.cv.paste(col, x0, y0, mask=a * float(opacity), mode=mode)
        return self.full(a)

    def shade(self, d, base, **kw):
        """Shaded colour map (crop-sized) of a local shape; keywords as ``shade``."""
        return shade(self.px(d), _crop_color(base, self.box, (self.cv.h, self.cv.w)), **kw)

    def form(self, d, base, aa=1.0, opacity=1.0, **kw):
        """Paint a local shape as a lit volume (keywords as ``paint_form``). Returns the full-frame mask."""
        return _paint_form_box(self.cv, self.px(d), self.box, base, aa=aa, opacity=opacity, **kw)


# ===================================================================== light & materials


class Light:
    """One light for the whole scene.

    ``direction``: 2D screen direction *towards* the light ((-0.6, -0.7) = upper left); ``elevation``: degrees out of
    the picture plane towards the viewer (90 = light from the camera); ``color`` / ``intensity`` of the key light;
    ``sky`` / ``ambient``: fill light from above; ``bounce``: colour reflected from the ground (lights faces looking
    down); ``shadow``: shadow tint; ``front`` (0~1): how much shadows on the ground fall backwards (up the screen).
    """

    def __init__(self, direction=(-0.6, -0.7), elevation=38.0, color="#fff3e2", intensity=1.05, sky="#a9bede",
                 ambient=0.36, bounce="#b9906c", shadow="#3a2c40", front=0.5):
        self.direction = (float(direction[0]), float(direction[1]))
        self.elevation = float(elevation)
        self.color = color
        self.intensity = float(intensity)
        self.sky = sky
        self.ambient = float(ambient)
        self.bounce = bounce
        self.shadow = shadow
        self.front = float(front)

    def replace(self, **kw):
        kw = _reject_unknown("Light", kw, _sig_options(Light.__init__), {"dir": "direction", "colour": "color"})
        d = dict(self.__dict__)
        d.update(kw)
        return Light(**d)

    @property
    def dir2(self):
        dx, dy = self.direction
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n

    @property
    def vec(self):
        """Unit vector towards the light (x right, y down, z towards the viewer)."""
        dx, dy = self.dir2
        e = math.radians(self.elevation)
        return np.array([dx * math.cos(e), dy * math.cos(e), math.sin(e)], F32)

    def shear(self, length=1.0):
        """Ground-shadow offset per pixel of object height: (dx, dy) — dy < 0 means the shadow runs up the screen."""
        e = math.radians(min(max(self.elevation, 8.0), 85.0))
        cot = 1.0 / math.tan(e)
        dx, _ = self.dir2
        return -dx * cot * 0.8 * length, -self.front * cot * 0.45 * length


_LIGHT = Light()


def set_light(light=None, **kw):
    """Set the scene light used by every call that gets ``light=None``. ``set_light(direction=(0.7, -0.5))``."""
    global _LIGHT
    if light is None:
        light = _LIGHT.replace(**kw) if kw else Light()
    elif kw:
        light = _resolve_light(light).replace(**kw)
    _LIGHT = _resolve_light(light)
    return _LIGHT


def get_light():
    """The current scene light."""
    return _LIGHT


def resolve_light(light=None):
    """``None`` → the scene light; a ``Light`` as is; a dict → scene light with those fields; (dx, dy) → direction."""
    if light is None:
        return _LIGHT
    if isinstance(light, Light):
        return light
    if isinstance(light, dict):
        return _LIGHT.replace(**light)
    return _LIGHT.replace(direction=tuple(light))


_resolve_light = resolve_light


_MAT_DEFAULT = dict(kd=1.0, wrap=0.15, ks=0.0, shin=20.0, spec_tint=0.0, env=0.0, env_blur=0.5, env_tint=0.0,
                    window=0.0, fresnel=0.04, rim=0.1, rim_pow=3.0, sss=0.0, sss_col=(1.0, 0.36, 0.2), sheen=0.0,
                    bounce=0.28, ambient=1.0, alpha=1.0, refract=0.0, emit=0.0, edge=0.0, bump=0.0, bump_size=0.02,
                    mottle=0.0, seed=0)

MATERIALS = {
    "matte": dict(wrap=0.2, ks=0.05, shin=10, rim=0.1),
    "plastic": dict(wrap=0.1, ks=0.5, shin=70, window=0.35, env=0.06),
    "glossy": dict(wrap=0.1, ks=0.9, shin=200, window=0.85, env=0.2, fresnel=0.06),
    "ceramic": dict(wrap=0.25, ks=0.75, shin=140, window=0.65, env=0.14, fresnel=0.05),
    "metal": dict(kd=0.35, wrap=0.1, ks=0.8, shin=40, spec_tint=0.8, env=0.75, env_blur=0.75, env_tint=0.85,
                  window=0.45, rim=0.0, fresnel=1.0),
    "chrome": dict(kd=0.05, ks=1.6, shin=500, spec_tint=0.1, env=1.0, env_blur=0.1, env_tint=0.12, window=1.1,
                   rim=0.0, fresnel=1.0),
    "gold": dict(kd=0.3, ks=1.1, shin=140, spec_tint=0.95, env=0.9, env_blur=0.35, env_tint=1.0, window=0.75,
                 rim=0.0, fresnel=1.0),
    "glass": dict(kd=0.04, ks=1.5, shin=400, env=0.55, env_blur=0.2, window=1.0, alpha=0.3, refract=0.3,
                  fresnel=0.08, rim=0.0, edge=0.55, bounce=0.0),
    "rubber": dict(wrap=0.1, ks=0.16, shin=14, rim=0.05),
    "skin": dict(wrap=0.35, ks=0.13, shin=22, sss=0.4, sss_col=(1.0, 0.3, 0.18), rim=0.1, window=0.08),
    "fabric": dict(wrap=0.45, ks=0.0, rim=0.0, sheen=0.2),
    "wood": dict(wrap=0.2, ks=0.22, shin=28, window=0.12),
    "clay": dict(wrap=0.3, ks=0.03, shin=6, sss=0.08),
    "fruit": dict(wrap=0.22, ks=0.45, shin=45, sss=0.25, window=0.3, env=0.04, mottle=0.08),
    "citrus": dict(wrap=0.25, ks=0.3, shin=30, sss=0.3, window=0.18, env=0.03, bump=0.35, bump_size=0.008,
                   mottle=0.07),
    "velvet": dict(kd=0.7, wrap=0.3, ks=0.0, sheen=0.8, rim=0.0),
    "paper": dict(wrap=0.35, ks=0.02, shin=8, rim=0.05, sss=0.05, sss_col=(1.0, 0.9, 0.75), mottle=0.03),
    "leather": dict(wrap=0.2, ks=0.22, shin=18, window=0.08, bump=0.2, bump_size=0.01, mottle=0.06),
}


def _material(material, overrides=None):
    m = dict(_MAT_DEFAULT)
    if isinstance(material, dict):
        m.update(MATERIALS.get(material.get("preset", "matte"), {}))
        m.update({k: v for k, v in material.items() if k != "preset"})
    else:
        if material not in MATERIALS:
            raise _unknown_name("material", material, MATERIALS, "Try form.MATERIALS for the list")
        m.update(MATERIALS[material])
    if overrides:
        m.update(_reject_unknown("material", overrides, list(_MAT_DEFAULT)))
    return m


def _env_color(rx, ry, blur, light):
    """Studio environment seen in a reflection: bright sky above, dark horizon line, warm floor below, a tall bright
    window strip on the light's side and a darker side opposite."""
    t = -ry
    sky = rgb(light.sky) * 1.15
    hz = (sky + 1.0) * 0.5
    gnd = rgb(light.bounce) * 0.4
    k = smoothstep(-0.04 - blur * 0.3, 0.04 + blur * 0.3, t)[..., None]
    up = np.clip(t, 0, 1)[..., None] ** 0.6
    skyc = hz + (sky - hz) * up
    out = gnd + (skyc - gnd) * k
    if blur < 0.45:
        line = np.exp(-(t / 0.035) ** 2) * (1 - blur / 0.45) * 0.6
        out = out * (1 - line[..., None])
    lx = light.vec[0]
    w = 0.16 + 0.3 * blur
    strip = np.exp(-((rx - lx * 1.1) / w) ** 2)
    shade_ = smoothstep(0.0, 0.9, rx * np.sign(-lx or 1.0)) * 0.45
    out = out * (1 - shade_[..., None]) + rgb(light.color) * (0.9 * strip * (0.6 + 0.4 * k[..., 0]))[..., None]
    return out


def _window_mask(a, b, blur):
    soft = 0.03 + 0.25 * blur
    hw = 0.2
    box = smoothstep(hw + soft, hw - soft, np.abs(a)) * np.exp(-(np.maximum(np.abs(b) - 0.4, 0) / (0.45 + blur)) ** 2)
    if blur < 0.6:
        mw = 0.022 + 0.02 * blur
        mull = np.maximum(smoothstep(mw, mw * 0.4, np.abs(a)), smoothstep(mw, mw * 0.4, np.abs(b + 0.04)))
        box = box * (1 - 0.8 * (1 - blur / 0.6) * mull)
    return box


def _illuminate(nx, ny, nz, base, m, light, occl=None, behind=None):
    """Light a normal field → (colour (h, w, 3), alpha (h, w) or None)."""
    L = light.vec
    lc = rgb(light.color) * F32(light.intensity)
    sky = rgb(light.sky)
    bnc = rgb(light.bounce)
    base = as_color(base)
    if base.ndim == 1:
        base = base[None, None, :]
    if (m["bump"] or m["mottle"]) and nx.ndim == 2 and min(nx.shape) > 2:
        h, w = nx.shape
        size = float(max(h, w))
        if m["bump"]:
            s = max(m["bump_size"] * size, 0.7)
            n = fbm2d(h, w, s, 3, seed=int(m["seed"]) + 17)
            gy, gx = np.gradient(n)
            amp = F32(m["bump"] * s) * np.clip(nz, 0, 1) ** 0.5
            nx, ny = nx - gx * amp, ny - gy * amp
            ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
            nx, ny, nz = nx / ln, ny / ln, nz / ln
        if m["mottle"]:
            mt = fbm2d(h, w, max(0.22 * size, 2.0), 3, seed=int(m["seed"]) + 29)
            base = base * (1 + F32(m["mottle"]) * mt)[..., None]
    occ = np.ones_like(nx) if occl is None else np.asarray(occl, F32)
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    wrap = m["wrap"]
    diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
    diff = diff * diff * (3 - 2 * diff) * 0.4 + diff * 0.6
    up = (0.5 - 0.5 * ny)[..., None]
    amb = (bnc * (1 - up) + sky * up) * F32(light.ambient * m["ambient"])
    E = (m["kd"] * diff * occ)[..., None] * lc + amb * (0.35 + 0.65 * occ)[..., None]
    col = base * E
    if m["bounce"]:
        col = col + base * bnc * (m["bounce"] * np.clip(ny, 0, 1) ** 1.5 * occ)[..., None]
    if m["sss"]:
        band = np.exp(-((ndl - 0.05) / 0.3) ** 2)
        col = col + rgb(m["sss_col"]) * np.sqrt(np.maximum(base, 0)) * (m["sss"] * band * occ)[..., None] * 0.6
    if m["sheen"]:
        sheen = m["sheen"] * (1 - np.clip(nz, 0, 1)) ** 2.5 * (0.35 + 0.65 * diff)
        col = col + (base * 0.65 + lc * 0.35) * sheen[..., None]
    f0 = m["fresnel"]
    F = f0 + (1 - f0) * (1 - np.clip(nz, 0, 1)) ** 5 if f0 < 0.5 else np.ones_like(nz)
    tint = 1.0 + (base - 1.0) * F32(m["spec_tint"])
    if m["ks"]:
        Hv = L + np.array([0, 0, 1], F32)
        Hv = Hv / np.linalg.norm(Hv)
        ndh = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1)
        lobe = ndh ** m["shin"] + 0.12 * ndh ** max(m["shin"] / 10.0, 2.0)
        col = col + lc * tint * (m["ks"] * lobe * occ)[..., None]
    if m["env"] or m["window"]:
        rx, ry, rz = 2 * nz * nx, 2 * nz * ny, 2 * nz * nz - 1
        rt = 1.0 + (base - 1.0) * F32(m["env_tint"])
        if m["env"]:
            col = col + _env_color(rx, ry, m["env_blur"], light) * rt * (m["env"] * F * (0.4 + 0.6 * occ))[..., None]
        if m["window"]:
            win = _window_mask(rx - L[0], ry - L[1], m["env_blur"]) * smoothstep(-0.3, 0.1, rz)
            col = col + lc * rt * (m["window"] * win * ((0.45 + 0.55 * F) if f0 < 0.5 else 1.0) * occ)[..., None]
    if m["rim"]:
        col = col + sky * (m["rim"] * (1 - np.clip(nz, 0, 1)) ** m["rim_pow"] * occ)[..., None]
    if m["emit"]:
        col = col + base * F32(m["emit"])
    alpha = None
    if m["alpha"] < 1:
        edge = (1 - np.clip(nz, 0, 1)) ** 2
        if behind is not None:
            absorb = np.power(np.clip(base, 1e-3, None), (0.75 + 2.6 * m["edge"] * edge)[..., None])
            trans = behind * absorb * (0.92 - 0.35 * m["edge"] * edge)[..., None] * (0.8 + 0.2 * diff)[..., None]
            col = trans + col
        else:
            alpha = np.clip(m["alpha"] + (1 - m["alpha"]) * edge * 1.2 + 0.5 * (col.max(-1) - 1).clip(0), 0, 1)
    return col.astype(F32), alpha


# ===================================================================== normals from a distance field


def _run_pos(M):
    """Signed position across the shape along axis 0: -1 at the top edge … +1 at the bottom edge."""
    h = M.shape[0]
    idx = np.arange(h, dtype=np.int32)[:, None]
    last = np.maximum.accumulate(np.where(~M, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(~M, idx, h)[::-1], axis=0)[::-1]
    up = (idx - last) - 0.5
    dn = (nxt - idx) - 0.5
    s = (up - dn) / np.maximum(up + dn, 1e-3)
    return np.clip(s, -1, 1).astype(F32)


def _rotate_f(a, angle, fill):
    im = Image.fromarray(np.ascontiguousarray(a, np.float32), "F")
    return np.asarray(im.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=float(fill)), F32)


def _cylinder_s(d, axis):
    """Across-axis position s (-1~1) for a shape whose long axis points at ``axis`` degrees; plus the unit
    perpendicular (px, py) that s is measured along."""
    a = float(axis) % 180.0
    if abs(a - 90.0) < 1e-3:
        return _run_pos((d < 0.75).T).T, (1.0, 0.0)
    if a < 1e-3 or abs(a - 180.0) < 1e-3:
        return _run_pos(d < 0.75), (0.0, 1.0)
    h, w = d.shape
    big = float(np.abs(d).max() + 10)
    dr = _rotate_f(d, a, big)
    sr = _run_pos(dr < 0.75)
    back = _rotate_f(sr, -a, 0.0)
    oy, ox = (back.shape[0] - h) // 2, (back.shape[1] - w) // 2
    s = back[oy:oy + h, ox:ox + w]
    r = math.radians(a)
    return s, (-math.sin(r), math.cos(r))


def normals(d, profile="round", width=None, axis=90.0, bevel=None, height=1.0, smooth=1.0):
    """Normal field (nx, ny, nz) of a signed distance treated as a raised volume.

    profile: ``round`` (sphere-like, exact for circles / capsules), ``dome`` (lower), ``flat`` (flat top, bevelled
    rim of ``bevel`` px), ``cushion`` (pillow), ``ridge`` (constant slope, crease along the middle), ``cylinder``
    (curved across ``axis`` degrees only: tubes, bottles, limbs). ``width`` = px over which the edge rounds
    (default: the shape's inscribed radius); ``height`` scales the bulge; ``smooth`` softens the crease along the
    middle of non-round shapes (0 = sharp). A negative ``height`` makes a dent (spoon bowls, dimples, wells).
    """
    d = np.asarray(d, F32)
    if profile == "cylinder":
        s, (pxx, pyy) = _cylinder_s(d, axis)
        s = np.clip(s * height, -0.995, 0.995)
        return (s * pxx).astype(F32), (s * pyy).astype(F32), np.sqrt(1 - s * s).astype(F32)
    t = np.maximum(-d, 0)
    if width is None:
        width = max(float(t.max()), 1.0)
    width = max(float(width), 1e-3)
    gy, gx = np.gradient(d)
    g = np.sqrt(gx * gx + gy * gy) + 1e-6
    gx, gy = gx / g, gy / g
    if smooth and width > 3:
        # round the crease along the shape's medial axis: blur the direction field, keep its shrinking length
        sig = max(0.12 * width * smooth, 0.8)
        bx, by_ = _f.gaussian_blur(gx, sig), _f.gaussian_blur(gy, sig)
        edge = np.clip(t / (sig * 2.5), 0, 1)
        gx = gx * (1 - edge) + bx * edge
        gy = gy * (1 - edge) + by_ * edge
    u = np.clip(1 - t / width, 0, 1)
    if profile in ("round", "sphere"):
        s = u
    elif profile == "dome":
        s = u * 0.62
    elif profile in ("flat", "bevel"):
        b = max(float(bevel) if bevel else width * 0.2, 1.0)
        s = np.clip(1 - t / b, 0, 1) * 0.95
    elif profile == "cushion":
        s = u ** 2.2 * 0.9 + 0.08 * u
    elif profile == "ridge":
        s = (0.55 * np.clip((width - t) / (0.25 * width), 0, 1)).astype(F32)
    else:
        raise _unknown_name("profile", profile, ("round", "sphere", "dome", "flat", "bevel", "cushion", "ridge", "cylinder"))
    s = np.clip(s * height, -0.995, 0.995)
    nx, ny = gx * s, gy * s
    return nx.astype(F32), ny.astype(F32), np.sqrt(np.clip(1 - nx * nx - ny * ny, 1e-4, 1)).astype(F32)


def shade(d, base, light=None, material="matte", profile="round", width=None, axis=90.0, bevel=None, height=1.0,
          contact=0.0, behind=None, **mat):
    """Colour map (same size as ``d``) of a shape shaded as a volume.

    d: signed px distance (negative inside); base: colour or colour map / texture of the same size;
    material: a name from ``MATERIALS`` or a dict of overrides; profile / width / axis / bevel / height: see
    ``normals``; contact: 0~1 darkening along the bottom edge where the object meets the ground.
    Extra keywords override material parameters (``ks=0.3``, ``sss=0.5``, ``emit=0.4`` ...).
    """
    light = _resolve_light(light)
    m = _material(material, mat)
    nx, ny, nz = normals(d, profile, width, axis, bevel, height)
    occ = None
    if contact:
        occ = 1 - contact * np.clip(ny, 0, 1) ** 2 * (1 - nz) ** 0.5
    col, _ = _illuminate(nx, ny, nz, base, m, light, occ, behind)
    return col


def _behind(cv, box, nx, ny, amount):
    x0, y0, x1, y1 = box
    X, Y = _grid_box(x0, y0, x1, y1)
    return _bilinear(cv.buf, X - nx * amount, Y - ny * amount)


def _paint_form_box(cv, dpx, box, base, light=None, material="matte", profile="round", width=None, axis=90.0,
                    bevel=None, height=1.0, contact=0.0, aa=1.0, opacity=1.0, mode="normal", **mat):
    light = _resolve_light(light)
    m = _material(material, mat)
    a = _sdf.fill(dpx, aa)
    nx, ny, nz = normals(dpx, profile, width, axis, bevel, height)
    occ = None
    if contact:
        occ = 1 - contact * np.clip(ny, 0, 1) ** 2 * (1 - nz) ** 0.5
    behind = None
    if m["refract"]:
        r = max(float(np.maximum(-dpx, 0).max()), 2.0)
        behind = _behind(cv, box, nx, ny, m["refract"] * r)
    col, alpha = _illuminate(nx, ny, nz, _crop_color(base, box, (cv.h, cv.w)), m, light, occ, behind)
    if alpha is not None:
        a = a * alpha
    cv.paste(col, box[0], box[1], mask=a * float(opacity), mode=mode)
    return _full(cv, a, box)


def paint_form(cv, d, base, light=None, material="matte", profile="round", aa=1.0, opacity=1.0, **kw):
    """Paint a shape as a lit volume and return its full-frame mask.

    ``d``: full-frame signed px distance (any ``sdf`` / ``d_*`` result); other keywords as ``shade``.
    Only the shape's bounding box is shaded, so many small shapes stay fast.
    ``paint_form(cv, d_circle(cv, (0.5, 0.5), 0.2), '#d94030', material='glossy')``
    """
    d = np.asarray(d, F32)
    bb = _bbox_of(-d, -aa)
    if bb is None:
        return np.zeros(d.shape, F32)
    x0, y0, x1, y1 = bb
    box = _clip_box(cv, x0 - 2, y0 - 2, x1 + 2, y1 + 2)
    return _paint_form_box(cv, d[box[1]:box[3], box[0]:box[2]], box, base, light, material, profile,
                           aa=aa, opacity=opacity, **kw)


# ===================================================================== shadows


def shadow_color(light=None, color=None):
    """The shadow tint: ``color`` if given, else the light's ``shadow`` colour (as an rgb array)."""
    return rgb(color) if color is not None else rgb(resolve_light(light).shadow)


def _shadow_color(light, color):
    return shadow_color(light, color)


def multiply_shadow(cv, a, box, color=None):
    """Multiply a crop-sized shadow strength ``a`` (0~1) placed at pixel ``box`` into the canvas with ``color``
    (default: the scene light's shadow tint). Never brightens."""
    x0, y0, x1, y1 = box
    col = np.broadcast_to(as_color(color if color is not None else shadow_color()), a.shape + (3,))
    cv.paste(np.ascontiguousarray(col), x0, y0, mask=np.clip(a, 0, 1), mode="multiply")


_multiply = multiply_shadow
multiply = multiply_shadow


def _shape_alpha(cv, shape):
    """Solid / mask / distance → (alpha crop, box, exclude alpha or None)."""
    if isinstance(shape, Solid):
        return shape.alpha, shape.box, shape.alpha
    a = np.asarray(shape, F32)
    if a.min() < -0.01 or a.max() > 1.01:
        a = _sdf.fill(a)
    bb = _bbox_of(a, 0.004)
    if bb is None:
        return None, None, None
    return a[bb[1]:bb[3], bb[0]:bb[2]], bb, None


def cast_shadow(cv, shape, light=None, ground_y=None, length=1.0, softness=1.0, opacity=0.7, color=None,
                fade=1.0, surface=None):
    """Project a silhouette onto the ground away from the light and multiply it in.

    shape: a ``Solid``, a full-frame mask or a signed distance; ground_y: the ground line (fraction of the height;
    default = the bottom of the shape, i.e. where it stands); length / softness / opacity / fade scale the shadow.
    The penumbra widens with distance from the contact line and the contact core is darker. On a table, shelf or
    room (a Solid placed at their Spot, or ``surface=``) the shadow stays on the top and continues up the wall
    behind. Returns the full-frame shadow strength mask.
    """
    light = _resolve_light(light)
    m, box, excl = _shape_alpha(cv, shape)
    if m is None:
        return np.zeros((cv.h, cv.w), F32)
    if surface is None:
        surface = getattr(shape, "surface", None)
    x0, y0, x1, y1 = box
    if ground_y is not None:
        gy = ground_y * cv.h
    elif isinstance(shape, Solid) and "base" in shape.anchors:
        gy = shape.anchors["base"][1]
    else:
        gy = y1 - 1.0
    hgt = max(gy - y0, 2.0)
    sx, sy = light.shear(length)
    if abs(sy) < 0.04:
        sy = -0.04 if sy <= 0 else 0.04
    S = cv.h / 1080.0
    sig0 = softness * 1.2 * S + 0.3
    sig1 = softness * (1.2 * S + 0.06 * hgt) + 0.3
    pad = int(3 * sig1) + 3
    src = np.pad(m, pad)
    levels = [sig0, (sig0 + sig1) / 2, sig1]
    blurred = [_f.gaussian_blur(src, s) for s in levels]
    reach_x = sx * hgt
    reach_y = sy * hgt
    dbox = _clip_box(cv, min(x0, x0 + reach_x) - pad, min(gy, gy + reach_y) - pad,
                     max(x1, x1 + reach_x) + pad, max(gy, gy + reach_y) + pad)
    if dbox is None:
        return np.zeros((cv.h, cv.w), F32)
    def strength(Xd, h, slide=None):
        """Shadow strength at pixels whose shadow-casting point is at height h above the contact line (slide:
        sideways offset in px, default h * sx)."""
        valid = smoothstep(-1.5, 0.5, h) * smoothstep(hgt + 2, hgt - 2, h)
        hc = np.clip(h, 0, hgt)
        sxp = Xd - (hc * sx if slide is None else slide) - (x0 - pad)
        syp = gy - hc - (y0 - pad)
        samp = [_bilinear(b, sxp, syp) for b in blurred]
        sig = np.clip(softness * (1.2 * S + 0.06 * hc) + 0.3, sig0, sig1)
        w = np.clip((sig - sig0) / max(sig1 - sig0, 1e-6) * 2, 0, 2)
        sh = np.where(w < 1, samp[0] * (1 - w) + samp[1] * w, samp[1] * (2 - w) + samp[2] * (w - 1))
        dark = opacity * (0.8 + 0.45 * np.exp(-hc / (0.05 * hgt + 1.0))) * np.exp(-0.7 * fade * hc / hgt)
        return np.clip(sh * dark * valid, 0, 0.95)

    def exclude(a, bx):
        if excl is None:
            return a
        ex = np.zeros_like(a)
        ix0, iy0 = max(box[0], bx[0]), max(box[1], bx[1])
        ix1, iy1 = min(box[2], bx[2]), min(box[3], bx[3])
        if ix0 < ix1 and iy0 < iy1:
            ex[iy0 - bx[1]:iy1 - bx[1], ix0 - bx[0]:ix1 - bx[0]] =                 excl[iy0 - box[1]:iy1 - box[1], ix0 - box[0]:ix1 - box[0]]
        return a * (1 - ex)

    Xd, Yd = _grid_box(*dbox)
    a = exclude(strength(Xd, (Yd - gy) / sy), dbox)
    regions = getattr(surface, "_shadow_regions", None)
    if regions is None:
        _multiply(cv, a, dbox, _shadow_color(light, color))
        return _full(cv, a, dbox)
    floor_w, _ = regions(dbox)
    if floor_w is not None:
        a = a * floor_w
    _multiply(cv, a, dbox, _shadow_color(light, color))
    out = _full(cv, a, dbox)
    # past the back edge the shadow climbs the wall: the point at height h lands (h - h_edge) above the edge with
    # the sideways offset it had at the edge (upright edges stay upright on a wall), joining the floor part
    yb = surface._back_y() if hasattr(surface, "_back_y") else None
    if yb is not None and sy < 0 and yb < gy:
        h_edge = (yb - gy) / sy
        if h_edge < hgt:
            rise = hgt - h_edge
            wbox = _clip_box(cv, min(x0, x0 + sx * h_edge) - pad, yb - rise - pad,
                             max(x1, x1 + sx * h_edge) + pad, yb + 2)
            if wbox is not None:
                _, wall_w = regions(wbox)
                if wall_w is not None:
                    Xw, Yw = _grid_box(*wbox)
                    w = exclude(strength(Xw, h_edge + (yb - Yw), h_edge * sx), wbox) * wall_w
                    _multiply(cv, w, wbox, _shadow_color(light, color))
                    o = out[wbox[1]:wbox[3], wbox[0]:wbox[2]]
                    out[wbox[1]:wbox[3], wbox[0]:wbox[2]] = np.maximum(o, w)
    return out


def contact_shadow_px(cv, x, y, rx, ry, strength=0.75, softness=1.0, light=None, color=None, clip=None):
    """Soft dark ellipse at pixel position (x, y) with pixel radii (rx, ry): the contact shadow under a foot, a
    wheel or a box. It fades smoothly to zero at any ``softness``; ``clip`` = optional full-frame mask it stays in.
    Returns the full-frame strength mask."""
    a, box = _contact_core(cv, x, y, rx, ry, strength, resolve_light(light), color, softness, clip)
    if box is None:
        return np.zeros((cv.h, cv.w), F32)
    return _full(cv, a, box)


def _contact_core(cv, x, y, rx, ry, strength, light, color, softness=1.0, clip=None):
    soft = max(float(softness), 0.05)
    rx, ry = max(rx, 1.0), max(ry, 0.8)
    shift = -light.vec[0] * rx * 0.15
    reach = 2.6 * math.sqrt(max(soft, 0.5))                 # exp(-0.9 q / soft) is < 0.3% beyond this radius
    box = _clip_box(cv, x - reach * rx + shift, y - reach * ry, x + reach * rx + shift, y + reach * ry)
    if box is None:
        return None, None
    X, Y = _grid_box(*box)
    q = ((X - x - shift) / rx) ** 2 + ((Y - y) / ry) ** 2
    a = strength * (0.55 * np.exp(-q * 2.6 / soft) + 0.45 * np.exp(-q * 0.9 / soft))
    a = a * smoothstep(reach * reach, reach * reach * 0.6, q)      # reaches exactly zero at the box edge
    if clip is not None:
        a = a * _crop_mask(clip, box)
    multiply_shadow(cv, a, box, shadow_color(light, color) * 0.7)
    return a, box


def _crop_mask(m, box):
    x0, y0, x1, y1 = box
    return np.asarray(m, F32)[y0:y1, x0:x1]


def _contact_px(cv, cx, cy, rx, ry, strength, light, color, softness=1.0, clip=None):
    _contact_core(cv, cx, cy, rx, ry, strength, light, color, softness, clip)


def contact_px(cv, cx, cy, rx, ry, strength=0.75, light=None, color=None, softness=1.0, clip=None):
    """Multiply a soft contact-shadow ellipse (pixel centre and radii) into the canvas; returns nothing
    (``contact_shadow_px`` returns the mask)."""
    _contact_core(cv, cx, cy, rx, ry, strength, resolve_light(light), color, softness, clip)


def contact_shadow(cv, at, width, height=None, strength=0.75, softness=1.0, light=None, color=None):
    """Tight dark ellipse where an object meets a surface. ``at`` = contact point (u, v) or Spot, ``width`` =
    full width as a fraction of the height (scaled at a Spot); ``height`` defaults to a tilt-based flat ellipse."""
    light = _resolve_light(light)
    cx, cy, k, tilt = _pos(cv, at)
    rx = width * k * cv.h / 2
    ry = height * k * cv.h / 2 if height else rx * max(0.16, 0.55 * (tilt if tilt is not None else VIEW_TILT))
    _contact_px(cv, cx, cy, rx, ry, strength, light, color, softness)


def drop_shadow(cv, shape, offset=(0.006, 0.01), blur=0.008, opacity=0.4, light=None, color=None):
    """Flat-design drop shadow: the shape's silhouette offset (fractions of the height) and blurred, multiplied in."""
    light = _resolve_light(light)
    m, box, excl = _shape_alpha(cv, shape)
    if m is None:
        return
    ox, oy = offset[0] * cv.h, offset[1] * cv.h
    sig = max(blur * cv.h, 0.3)
    pad = int(3 * sig) + 2
    src = _f.gaussian_blur(np.pad(m, pad), sig)
    x0, y0 = int(round(box[0] - pad + ox)), int(round(box[1] - pad + oy))
    dbox = _clip_box(cv, x0, y0, x0 + src.shape[1], y0 + src.shape[0])
    if dbox is None:
        return
    sub = src[dbox[1] - y0:dbox[3] - y0, dbox[0] - x0:dbox[2] - x0] * opacity
    _multiply(cv, sub, dbox, _shadow_color(light, color))


def reflection(cv, solid, strength=0.25, fade=0.6, blur=0.0, ground_y=None):
    """Mirror a just-drawn ``Solid`` in a glossy surface below it (varnished table, polished floor, water).

    strength: reflection opacity at the contact line; fade: fraction of the object height over which it fades;
    blur: px softening (rough surfaces). Call right after drawing the object. Returns the full-frame mask.
    """
    x0, y0, x1, y1 = solid.box
    gy = ground_y * cv.h if ground_y is not None else solid.anchors.get("base", (0, y1))[1]
    hgt = max(gy - y0, 1.0)
    src = cv.buf[y0:y1, x0:x1] * solid.alpha[..., None]
    a = solid.alpha
    dst_top, dst_bot = int(math.floor(gy)), int(math.ceil(2 * gy - y0))
    box = _clip_box(cv, x0, dst_top, x1, dst_bot)
    if box is None:
        return np.zeros((cv.h, cv.w), F32)
    yy = np.arange(box[1], box[3], dtype=np.float64)
    sy = 2 * gy - yy                                     # source row for each destination row
    idx = np.clip(np.round(sy - y0).astype(int), 0, y1 - y0 - 1)
    valid = ((sy >= y0) & (sy <= gy)).astype(F32)
    col = src[idx][:, box[0] - x0:box[2] - x0]
    al = a[idx][:, box[0] - x0:box[2] - x0] * valid[:, None]
    if blur:
        col = _f.gaussian_blur(col, blur)
        al = _f.gaussian_blur(al, blur)
    k = strength * np.exp(-(yy - gy) / (fade * hgt + 1e-6))[:, None]
    col = col / np.maximum(al[..., None], 1e-4)
    m = np.clip(al * k, 0, 1)
    n = min(box[3], y1) - box[1]                         # rows shared with the object itself
    if n > 0:
        m[:n] *= 1 - solid.alpha[box[1] - y0:box[1] - y0 + n, box[0] - x0:box[2] - x0]
    cv.paste(col.astype(F32), box[0], box[1], mask=m)
    return _full(cv, m, box)


def occlusion_between(d_a, d_b, radius=10.0, strength=0.6):
    """Mask that darkens shape b where it comes close to shape a (px distances). Paint it with a dark colour in
    ``multiply`` mode, or ``cv.paint('#000', mask * 0.5)``."""
    d_a = np.asarray(d_a, F32)
    d_b = np.asarray(d_b, F32)
    return (strength * _sdf.fill(d_b) * (1 - _sdf.fill(d_a)) * np.exp(-np.maximum(d_a, 0) / float(radius))).astype(F32)


# ===================================================================== solids


class Solid:
    """A drawn object: ``mask`` (full frame), ``alpha`` + ``box`` (its crop), ``anchors`` (pixel points such as
    ``base``, ``top``, ``center``, ``left``, ``right``), and the data to re-shade parts of it (``recolor`` /
    ``band``)."""

    def __init__(self, cv, box, alpha, anchors=None, normals=None, material=None, light=None, hmap=None,
                 occl=None):
        self.cv = cv
        self.box = box
        self.alpha = alpha
        self.anchors = dict(anchors or {})
        self.normals = normals
        self.material = material
        self.light = light
        self.hmap = hmap
        self.occl = occl
        self.surface = None          # the table / shelf / room it stands on (keeps cast shadows on it)
        self.lift_px = 0.0           # how far its base is raised above that surface (px, screen)
        self._point = None

    @property
    def mask(self):
        return _full(self.cv, self.alpha, self.box)

    def __getitem__(self, key):
        return self.anchors[key]

    def uv(self, key):
        """Anchor as canvas fractions."""
        x, y = self.anchors[key]
        return x / self.cv.w, y / self.cv.h

    def point(self, t, side=1):
        """Pixel point on the outline at height fraction ``t`` (0 = base, 1 = top); ``side`` = 1 right, -1 left."""
        if self._point is None:
            raise AttributeError("point() is available on lathe-based solids")
        return self._point(t, side)

    def recolor(self, base, where=None, material=None, opacity=1.0, **mat):
        """Re-shade part of the object with a new base colour / texture (labels, stripes, patterns).
        ``where``: full-frame or crop-sized mask (default: the whole object)."""
        if self.normals is None:
            raise AttributeError("this solid has no stored normals")
        x0, y0, x1, y1 = self.box
        m = _material(material or self.material, mat)
        nx, ny, nz = self.normals
        col, _ = _illuminate(nx, ny, nz, _crop_color(base, self.box, (self.cv.h, self.cv.w)), m, self.light,
                             self.occl)
        a = self.alpha
        if where is not None:
            w_ = np.asarray(where, F32)
            if w_.shape == (self.cv.h, self.cv.w):
                w_ = w_[y0:y1, x0:x1]
            a = a * w_
        self.cv.paste(col, x0, y0, mask=a * float(opacity))
        return _full(self.cv, a, self.box)

    def band(self, t0, t1, base, material=None, soft=1.0, **mat):
        """Re-shade the part between height fractions t0 and t1 (bottle labels, mug stripes, lamp rings)."""
        if self.hmap is None:
            raise AttributeError("band() is available on lathe-based solids")
        hpx = self.anchors["height_px"]
        e = soft / max(hpx, 1.0)
        w_ = smoothstep(t0 - e, t0 + e, self.hmap) * smoothstep(t1 + e, t1 - e, self.hmap)
        return self.recolor(base, w_, material, **mat)


def _finish_solid(cv, col, alpha, box, opacity=1.0):
    cv.paste(col, box[0], box[1], mask=alpha * float(opacity))


def _attach(sol, at, cv):
    """Remember the surface a solid was placed on (from its Spot)."""
    sol.surface = getattr(at, "surface", None)
    sol.lift_px = getattr(at, "lift", 0.0) * cv.h
    return sol


def _ground_shadows(cv, shadow, cast, light, base_xy, rx, tilt, alpha, box, anchors, strength=0.7, at=None):
    if cast:
        kw = cast if isinstance(cast, dict) else {}
        cast_shadow(cv, _attach(Solid(cv, box, alpha, anchors), at, cv), light, **kw)
    if shadow:
        s = shadow if isinstance(shadow, (int, float)) and not isinstance(shadow, bool) else strength
        _contact_px(cv, base_xy[0], base_xy[1], rx * 1.05, rx * max(0.16, 0.6 * tilt), s, light, None)


def ball(cv, at, radius, base="#c8402f", material="plastic", light=None, shadow=True, cast=False, tilt=None,
         opacity=1.0, name=None, alias=None, **mat):
    """A sphere (or ellipsoid with ``radius=(rx, ry)``) resting on the ground at ``at`` (its contact point).

    Sizes are fractions of the canvas height (scaled by ``k`` at a Spot). ``shadow`` draws a contact shadow,
    ``cast=True`` (or a dict of ``cast_shadow`` options) a projected shadow. Returns a ``Solid``.
    """
    light = _resolve_light(light)
    bx, by, k, t0 = _pos(cv, at)
    tilt = tilt if tilt is not None else (t0 if t0 is not None else VIEW_TILT)
    rx, ry = (radius, radius) if np.ndim(radius) == 0 else radius
    rx, ry = rx * k * cv.h, ry * k * cv.h
    cx, cy = bx, by - ry
    box = _clip_box(cv, cx - rx - 2, cy - ry - 2, cx + rx + 3, cy + ry + 3)
    anchors = dict(base=(bx, by), center=(cx, cy), top=(cx, cy - ry), left=(cx - rx, cy), right=(cx + rx, cy),
                   height_px=2 * ry)
    if box is None:
        return Solid(cv, (0, 0, 1, 1), np.zeros((1, 1), F32), anchors)
    X, Y = _grid_box(*box)
    ex, ey = (X - cx) / rx, (Y - cy) / ry
    q = ex * ex + ey * ey
    rq = np.sqrt(q)
    dpx = (rq - 1) * min(rx, ry)
    alpha = _sdf.fill(dpx)
    ez = np.sqrt(np.clip(1 - q, 0, 1))
    rz = min(rx, ry)
    nx, ny, nz = ex / rx, ey / ry, ez / rz
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx, ny, nz = nx / n, ny / n, nz / n
    m = _material(material, mat)
    occ = 1 - 0.45 * smoothstep(0.55, 1.0, ey) * (1 - nz) ** 0.5
    behind = _behind(cv, box, nx, ny, m["refract"] * rz) if m["refract"] else None
    col, a2 = _illuminate(nx, ny, nz, _crop_color(base, box, (cv.h, cv.w)), m, light, occ, behind)
    if a2 is not None:
        alpha = alpha * a2
    sol = _attach(Solid(cv, box, alpha, anchors, (nx, ny, nz), material, light, None, occ), at, cv)
    _ground_shadows(cv, shadow, cast, light, (bx, by), rx * 0.85, tilt, alpha, box, anchors, at=at)
    _finish_solid(cv, col, alpha, box, opacity)
    return sol


def capsule(cv, a, b, radius, base="#4a7bd0", material="plastic", light=None, shadow=False, opacity=1.0, name=None, alias=None, **mat):
    """A rounded rod from point a to b (fractions), radius as a fraction of the height. Exact tube shading."""
    light = _resolve_light(light)
    ax, ay = a[0] * cv.w, a[1] * cv.h
    bx_, by_ = b[0] * cv.w, b[1] * cv.h
    r = radius * cv.h
    box = _clip_box(cv, min(ax, bx_) - r - 2, min(ay, by_) - r - 2, max(ax, bx_) + r + 3, max(ay, by_) + r + 3)
    anchors = dict(a=(ax, ay), b=(bx_, by_), center=((ax + bx_) / 2, (ay + by_) / 2),
                   base=(ax, ay) if ay >= by_ else (bx_, by_))
    if box is None:
        return Solid(cv, (0, 0, 1, 1), np.zeros((1, 1), F32), anchors)
    X, Y = _grid_box(*box)
    d = _sdf.sd_segment(X, Y, ax, ay, bx_, by_, r)
    if shadow:
        _contact_px(cv, anchors["base"][0], anchors["base"][1] + r * 0.9, r * 1.2, r * 0.35, 0.6, light, None)
    m = _material(material, mat)
    nx, ny, nz = normals(d, "round", r)
    col, a2 = _illuminate(nx, ny, nz, _crop_color(base, box, (cv.h, cv.w)), m, light)
    alpha = _sdf.fill(d) * (a2 if a2 is not None else 1)
    _finish_solid(cv, col, alpha, box, opacity)
    return Solid(cv, box, alpha, anchors, (nx, ny, nz), material, light)


# ---- lathe family


def _profile(points, n=480, smooth=True):
    P_ = np.asarray(points, np.float64)
    if smooth and len(P_) >= 3:
        Q = np.vstack([P_[0], P_, P_[-1]])
        segs = []
        t = np.linspace(0, 1, 24, endpoint=False)[:, None]
        for i in range(len(P_) - 1):
            p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
            segs.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                               + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
        segs.append(P_[-1:])
        C = np.vstack(segs)
    else:
        C = P_
    hh = np.maximum.accumulate(C[:, 1])
    hh = hh + np.arange(len(hh)) * 1e-9
    rr = np.maximum(C[:, 0], 0)
    hs = np.linspace(hh[0], hh[-1], n)
    rs = np.interp(hs, hh, rr)
    return hs - hs[0], rs


def lathe(cv, at, profile, base="#d8d2c8", material="ceramic", light=None, tilt=None, open=False, inside=None,
          rim=0.07, fill=None, fill_color="#3a2216", top=None, handle=None, shadow=True, cast=False, smooth=True,
          opacity=1.0, name=None, alias=None, **mat):
    """A vertical profile revolved into a round object: bottles, vases, mugs, bowls, lamps, pins.

    at: contact point (u, v) or Spot; profile: ``[(radius, height), ...]`` from the bottom up, fractions of the
    canvas height (scaled by ``k`` at a Spot); tilt: how much we look down (0 = side view, ~0.3 = table view);
    open: hollow top (``inside`` colour, ``rim`` wall thickness as a fraction of the top radius, ``fill`` 0~1
    liquid level with ``fill_color``); top: colour of a closed top cap; handle: ``True`` or
    ``dict(at=0.55, side=1, size=0.5, thickness=0.3)`` adds a mug handle. Returns a ``Solid`` whose
    ``point(t, side)`` gives outline points and ``band(t0, t1, colour)`` re-shades a horizontal band.
    """
    light = _resolve_light(light)
    bx, by, k, t0 = _pos(cv, at)
    se = float(tilt if tilt is not None else (t0 if t0 is not None else VIEW_TILT))
    se = min(max(se, 0.0), 0.95)
    ce = math.sqrt(1 - se * se)
    sc = k * cv.h
    hs, rs = _profile(profile, smooth=smooth)
    hs, rs = hs * sc, rs * sc
    top_h = float(hs[-1])
    rt, r0 = float(rs[-1]), float(rs[0])
    rmax = float(rs.max())
    drdh = np.gradient(rs, hs)
    yt = by - top_h * ce
    ymin = float(np.min(by - hs * ce - rs * se))
    box = _clip_box(cv, bx - rmax - 3, ymin - 3, bx + rmax + 4, by + rmax * se + 4)
    imax = int(np.argmax(rs))

    def _point(t, side=1):
        hh_ = float(np.clip(t, 0, 1)) * top_h
        return bx + side * float(np.interp(hh_, hs, rs)), by - hh_ * ce

    anchors = dict(base=(bx, by), top=(bx, yt), center=(bx, by - top_h * ce / 2),
                   left=(bx - rmax, by - hs[imax] * ce), right=(bx + rmax, by - hs[imax] * ce),
                   height_px=top_h, top_radius=rt, tilt=se)
    if box is None:
        s = Solid(cv, (0, 0, 1, 1), np.zeros((1, 1), F32), anchors)
        s._point = _point
        return s
    X, Y = _grid_box(*box)
    x = X - bx
    hc = (by - Y) / ce
    r_c = np.interp(np.clip(hc, 0, top_h), hs, rs)
    d_side = np.abs(x) - r_c
    d_bot = Y - (by + se * np.sqrt(np.maximum(r0 * r0 - x * x, 0)))
    d_top = (by - top_h * ce) - Y
    d_body = np.maximum(np.maximum(d_side, d_bot), d_top)
    has_cap = rt > 0.6 and se > 0.01
    if has_cap:
        d_cap = _sd_ellipse(X, Y, bx, yt, rt, rt * se)
        d_all = np.minimum(d_body, d_cap)
    else:
        d_cap = None
        d_all = d_body
    alpha = _sdf.fill(d_all)
    # side: height of the front surface per pixel (fixed-point iteration)
    h = hc + se * np.sqrt(np.maximum(r_c * r_c - x * x, 0)) / ce
    for i in range(7):
        rr = np.interp(np.clip(h, 0, top_h), hs, rs)
        hn = hc + se * np.sqrt(np.maximum(rr * rr - x * x, 0)) / ce
        h = hn if i < 2 else 0.5 * (h + hn)
    hcl = np.clip(h, 0, top_h)
    rr = np.maximum(np.interp(hcl, hs, rs), 1e-3)
    dr = np.interp(hcl, hs, drdh)
    cphi = np.clip(x / rr, -1, 1)
    sphi = np.sqrt(1 - cphi * cphi)
    n0, n1, n2 = cphi, -dr, sphi
    nn = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2) + 1e-9
    n0, n1, n2 = n0 / nn, n1 / nn, n2 / nn
    side_n = (n0.astype(F32), (-n1 * ce + n2 * se).astype(F32), (n1 * se + n2 * ce).astype(F32))
    hmap = (hcl / max(top_h, 1e-6)).astype(F32)
    occ = (1 - 0.35 * np.exp(-hcl / (0.06 * top_h + 1)) * (0.4 + 0.6 * sphi)).astype(F32)
    m = _material(material, mat)
    full_shape = (cv.h, cv.w)
    base_c = _crop_color(base, box, full_shape)
    if base_c.ndim == 1:
        base_c = np.broadcast_to(base_c, X.shape + (3,))
    if fill is not None and m["alpha"] < 1:                  # liquid seen through glass
        wet = smoothstep(fill * top_h + 1, fill * top_h - 1, hcl)[..., None]
        base_c = base_c * (1 - wet) + rgb(fill_color) * wet
    behind = None
    if m["refract"]:
        behind = _behind(cv, box, side_n[0], side_n[1], m["refract"] * rmax)
    col, a2 = _illuminate(*side_n, base_c, m, light, occ, behind)
    if a2 is not None:
        alpha = alpha * a2
    nx, ny, nz = side_n
    if has_cap:
        wcap = _sdf.fill(d_cap)[..., None]
        ex = x / max(rt, 1e-3)
        ey = (Y - yt) / max(rt * se, 0.3)
        one = np.ones_like(X)
        up_n = ((ex * 0.12).astype(F32), (-ce + ey * 0.06 * se).astype(F32), (se * one).astype(F32))
        if open:
            ri = rt * (1 - rim)
            w_in = _sdf.fill(_sd_ellipse(X, Y, bx, yt, ri, ri * se))[..., None]
            lip_c, _ = _illuminate(*up_n, base_c, m, light)
            # inner back wall
            bk = np.sqrt(np.clip(1 - (x / ri) ** 2, 0, 1))
            hh = (by - Y - se * ri * bk) / ce
            wn = (-np.clip(x / ri, -1, 1), bk * se, bk * ce)
            l_ = np.sqrt(wn[0] ** 2 + wn[1] ** 2 + wn[2] ** 2) + 1e-6
            wn = tuple((c / l_).astype(F32) for c in wn)
            depth = np.clip(top_h - hh, 0, None)
            occ_in = np.clip(np.exp(-depth / (ri * 1.3 + 1)), 0.15, 1)
            occ_in = occ_in * (1 - 0.5 * smoothstep(-0.2, 1.0, ey) * (1 - occ_in))
            in_c = rgb(inside) if inside is not None else base_c * 0.85
            wall_c, _ = _illuminate(*wn, in_c, m, light, occ_in.astype(F32))
            capc = wall_c
            if fill is not None:
                hf = float(fill) * top_h
                rf = float(np.interp(hf, hs, rs)) * (1 - rim)
                yf = by - hf * ce
                qf = np.sqrt((x / max(rf, 1)) ** 2 + ((Y - yf) / max(rf * se, 0.3)) ** 2)
                men = smoothstep(0.8, 1.0, qf)
                ln = ((x / max(rf, 1) * 0.2 * men).astype(F32), (-ce + 0.3 * men * se).astype(F32),
                      (se * one).astype(F32))
                l_ = np.sqrt(ln[0] ** 2 + ln[1] ** 2 + ln[2] ** 2)
                ln = tuple((c / l_).astype(F32) for c in ln)
                occ_l = (0.6 + 0.4 * (1 - smoothstep(0.3, 1.0, ey * 0.5 + 0.5))).astype(F32)
                liq_c, _ = _illuminate(*ln, rgb(fill_color), _material(dict(preset="glossy", env=0.8, fresnel=0.12)),
                                       light, occ_l)
                w_l = _sdf.fill((hh - hf) * ce)[..., None]
                capc = wall_c * (1 - w_l) + liq_c * w_l
            cap_col = lip_c * (1 - w_in) + capc * w_in
        else:
            cap_base = rgb(top) if top is not None else base_c
            cap_col, _ = _illuminate(*up_n, cap_base, m, light)
        col = col * (1 - wcap) + cap_col * wcap
        w2 = wcap[..., 0]
        nx = nx * (1 - w2) + up_n[0] * w2
        ny = ny * (1 - w2) + up_n[1] * w2
        nz = nz * (1 - w2) + up_n[2] * w2
    sol = _attach(Solid(cv, box, alpha, anchors, (nx, ny, nz), material, light, hmap, occ), at, cv)
    sol._point = _point
    _ground_shadows(cv, shadow, cast, light, (bx, by), max(r0, rmax * 0.8), se, alpha, box, anchors, at=at)
    if handle:
        hd = handle if isinstance(handle, dict) else {}
        side = 1 if hd.get("side", 1) >= 0 else -1
        hx, hy = _point(hd.get("at", 0.55), side)
        R = hd.get("size", 0.5) * top_h / 2
        th = hd.get("thickness", 0.3)
        fr = Frame(cv, ((hx + side * R * 0.38) / cv.w, hy / cv.h), R / cv.h, squash=(0.78, 1.0), flip=side < 0,
                   extent=1.4)
        hcol = base_c.reshape(-1, 3).mean(0) if np.ndim(base_c) == 3 else base_c
        fr.form(_sdf.sd_arc(fr.X, fr.Y, 0, 0, 1.0, -105, 105, th), hcol, light=light, material=material,
                width=fr.px(th / 2), **mat)
        sol.anchors["handle"] = (hx + side * R * 1.3, hy)
    _finish_solid(cv, col, alpha, box, opacity)
    return sol


def cylinder(cv, at, radius, height, base="#5d8fc9", material="ceramic", light=None, tilt=None, round=0.06,
             open=False, name=None, alias=None, **kw):
    """Upright cylinder standing at ``at`` (sizes: fractions of the height). ``round`` softens the edges;
    ``open=True`` makes a cup / can (see ``lathe`` for inside / fill / rim). Returns a ``Solid``."""
    ro = max(min(round, 0.45), 0.0) * min(radius, height)
    prof = [(radius - ro, 0.0), (radius, ro), (radius, height - (0 if open else ro)), ]
    prof.append((radius, height) if open else (radius - ro, height))
    return lathe(cv, at, prof, base, material, light, tilt, open=open, smooth=bool(ro) and not open, name=name, alias=alias, **kw)


def cone(cv, at, radius, height, base="#e0a040", material="matte", light=None, tilt=None, top_radius=0.0, name=None, alias=None, **kw):
    """Cone (or frustum with ``top_radius``) standing at ``at``. Returns a ``Solid``."""
    return lathe(cv, at, [(radius, 0.0), (top_radius, height)], base, material, light, tilt, smooth=False, name=name, alias=alias, **kw)


def torus(cv, at, radius, thickness, base="#d9a441", material="glossy", light=None, tilt=None, shadow=True,
          cast=False, opacity=1.0, name=None, alias=None, **mat):
    """A ring lying on the ground at ``at``: ``radius`` = ring radius, ``thickness`` = tube radius (fractions of
    the height). Ray-marched for exact silhouettes at any tilt. Returns a ``Solid``."""
    light = _resolve_light(light)
    bx, by, k, t0 = _pos(cv, at)
    se = float(tilt if tilt is not None else (t0 if t0 is not None else VIEW_TILT))
    se = min(max(se, 0.02), 0.98)
    ce = math.sqrt(1 - se * se)
    R, r = radius * k * cv.h, thickness * k * cv.h
    cx, cy = bx, by - r * ce
    ext_y = (R + r) * se + r * ce
    box = _clip_box(cv, cx - R - r - 2, cy - ext_y - 2, cx + R + r + 3, cy + ext_y + 3)
    anchors = dict(base=(bx, by), center=(cx, cy), top=(cx, cy - ext_y), left=(cx - R - r, cy), right=(cx + R + r, cy))
    if box is None:
        return Solid(cv, (0, 0, 1, 1), np.zeros((1, 1), F32), anchors)
    X, Y = _grid_box(*box)
    sx_, su = X - cx, cy - Y
    T = R + r + 4
    ox, oy, oz = sx_, su * ce + T * se, -su * se + T * ce
    dx, dy, dz = 0.0, -se, -ce
    t = np.zeros_like(X)
    dmin = np.full_like(X, 1e9)
    for _ in range(64):
        px_, py_, pz_ = ox + dx * t, oy + dy * t, oz + dz * t
        q = np.sqrt(px_ * px_ + pz_ * pz_) - R
        d = np.sqrt(q * q + py_ * py_) - r
        dmin = np.minimum(dmin, d)
        t = t + np.where(d > 0.05, d * 0.9, 0)
        t = np.minimum(t, 2 * T + 2)
    alpha = _sdf.fill(dmin - 0.5)
    px_, py_, pz_ = ox + dx * t, oy + dy * t, oz + dz * t
    lxz = np.sqrt(px_ * px_ + pz_ * pz_) + 1e-6
    q = lxz - R
    n0, n1, n2 = px_ / lxz * q, py_, pz_ / lxz * q
    nn = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2) + 1e-9
    n0, n1, n2 = n0 / nn, n1 / nn, n2 / nn
    nx, ny, nz = n0, -n1 * ce + n2 * se, n1 * se + n2 * ce
    occ = 1 - 0.4 * np.exp(-np.maximum(py_ + r, 0) / (0.35 * r + 1))
    m = _material(material, mat)
    behind = _behind(cv, box, nx, ny, m["refract"] * r) if m["refract"] else None
    col, a2 = _illuminate(nx, ny, nz, _crop_color(base, box, (cv.h, cv.w)), m, light, occ, behind)
    if a2 is not None:
        alpha = alpha * a2
    sol = _attach(Solid(cv, box, alpha, anchors, (nx, ny, nz), material, light, None, occ), at, cv)
    if shadow or cast:
        _ground_shadows(cv, False, cast, light, (bx, by), R + r, se, alpha, box, anchors, at=at)
        if shadow:
            ring = (R + r) * 0.98
            _contact_px(cv, cx, cy + 0.0, ring, ring * se * 0.95, 0.35, light, None)
    _finish_solid(cv, col, alpha, box, opacity)
    return sol


def disc(cv, at, radius, width, yaw=0.0, base="#26262a", hub="#d5d8dc", hub_radius=0.55, material="rubber",
         hub_material="metal", light=None, tilt=None, shadow=True, cast=False, opacity=1.0, name=None, alias=None, **mat):
    """A disc standing on its edge at ``at`` (its contact point): wheels, tyres, coins on edge, rolls of tape.

    radius / width: fractions of the canvas height (scaled at a Spot); yaw: degrees the axle turns (0 = round face
    towards the viewer, 90 = seen edge-on; wheels of a ``cuboid`` body take the body's own yaw);
    hub: colour of the centre (None for a plain disc) covering ``hub_radius`` of the face. Returns a ``Solid``.
    """
    light = _resolve_light(light)
    bx, by, k, t0 = _pos(cv, at)
    se = float(tilt if tilt is not None else (t0 if t0 is not None else VIEW_TILT))
    se = min(max(se, 0.0), 0.95)
    ce = math.sqrt(1 - se * se)
    R, w = radius * k * cv.h, width * k * cv.h
    a = math.radians(yaw)
    A = (-math.sin(a), 0.0, math.cos(a))
    rr = max(min(0.3 * w, 0.18 * R), 0.5)
    ext = R + w / 2 + 3
    box = _clip_box(cv, bx - ext, by - 2 * R * ce - ext * se - 3, bx + ext, by + ext * se + 3)
    cy = by - R * ce
    anchors = dict(base=(bx, by), center=(bx, cy), top=(bx, by - 2 * R * ce), left=(bx - ext, cy),
                   right=(bx + ext, cy), height_px=2 * R)
    if box is None:
        return Solid(cv, (0, 0, 1, 1), np.zeros((1, 1), F32), anchors)

    def sdf3(px_, py_, pz_):
        qy = py_ - R
        ax_ = px_ * A[0] + pz_ * A[2]
        rx_, ry_, rz_ = px_ - ax_ * A[0], qy, pz_ - ax_ * A[2]
        rad = np.sqrt(rx_ * rx_ + ry_ * ry_ + rz_ * rz_)
        dx, dy = rad - (R - rr), np.abs(ax_) - (w / 2 - rr)
        return np.minimum(np.maximum(dx, dy), 0) + np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) - rr, ax_, rad

    X, Y = _grid_box(*box)
    su = by - Y
    T = 2 * ext + 4
    ox, oy, oz = X - bx, su * ce + T * se, -su * se + T * ce
    dx_, dy_, dz_ = 0.0, -se, -ce
    t = np.zeros_like(X)
    dmin = np.full_like(X, 1e9)
    for _ in range(56):
        d, _, _ = sdf3(ox + dx_ * t, oy + dy_ * t, oz + dz_ * t)
        dmin = np.minimum(dmin, d)
        t = np.minimum(t + np.where(d > 0.05, d * 0.9, 0), 2 * T + 2)
    alpha = _sdf.fill(dmin - 0.5)
    px_, py_, pz_ = ox + dx_ * t, oy + dy_ * t, oz + dz_ * t
    e = 0.35
    g0 = sdf3(px_ + e, py_, pz_)[0] - sdf3(px_ - e, py_, pz_)[0]
    g1 = sdf3(px_, py_ + e, pz_)[0] - sdf3(px_, py_ - e, pz_)[0]
    g2 = sdf3(px_, py_, pz_ + e)[0] - sdf3(px_, py_, pz_ - e)[0]
    gn = np.sqrt(g0 * g0 + g1 * g1 + g2 * g2) + 1e-9
    n0, n1, n2 = g0 / gn, g1 / gn, g2 / gn
    nx, ny, nz = n0, -n1 * ce + n2 * se, n1 * se + n2 * ce
    _, axv, rad = sdf3(px_, py_, pz_)
    occ = (1 - 0.35 * np.exp(-np.maximum(py_, 0) / (0.12 * R + 1))).astype(F32)
    m = _material(material, mat)
    col, _ = _illuminate(nx.astype(F32), ny.astype(F32), nz.astype(F32), _crop_color(base, box, (cv.h, cv.w)), m,
                         light, occ)
    if hub is not None:
        face = smoothstep(w / 2 - rr * 0.6, w / 2 - rr * 0.2, np.abs(axv))
        hr = hub_radius * R
        hw_ = face * clamp01((hr - rad) + 0.5)
        hcol, _ = _illuminate(nx.astype(F32), ny.astype(F32), nz.astype(F32), rgb(hub), _material(hub_material),
                              light, occ)
        ring = np.exp(-((rad - hr) / max(0.04 * R, 0.6)) ** 2) * face
        col = col * (1 - hw_[..., None]) + hcol * hw_[..., None]
        col = col * (1 - 0.45 * ring)[..., None]
        nut = face * clamp01((0.18 * R - rad) + 0.5)
        col = col * (1 - 0.25 * nut)[..., None]
    sol = _attach(Solid(cv, box, alpha, anchors, (nx, ny, nz), material, light, None, occ), at, cv)
    _ground_shadows(cv, shadow, cast, light, (bx, by), R * abs(math.sin(a)) * 0.7 + w * 0.55, se, alpha, box,
                    anchors, at=at)
    _finish_solid(cv, col, alpha, box, opacity)
    return sol


def book(cv, at, size=(0.22, 0.15, 0.035), cover="#8a2f2a", pages="#efe6d2", spine="front", yaw=8.0,
         accent="#d8b765", material="matte", light=None, tilt=None, shadow=True, cast=False, opacity=1.0, name=None, alias=None):
    """A closed book lying flat at ``at``: hard covers, a cream page block on three sides, a spine with two
    accent bands. size = (width, depth, thickness) as fractions of the height; spine: ``front`` / ``back`` /
    ``left`` / ``right``. Stack books with ``spot.up(thickness)``. Returns a ``Solid`` (a cuboid)."""
    k = getattr(at, "k", 1.0)
    w, d, t = [s_ * k * cv.h for s_ in size]
    hx, hy, hz = w / 2, t / 2, d / 2
    c = max(min(0.14 * t, 0.012 * min(w, d)), 0.6)
    cv_, pg, ac = rgb(cover), rgb(pages), rgb(accent) if accent else None

    def face_col(face, lx, ly, lz):
        n = lx.shape[0]
        if face in ("top", "bottom"):
            return np.broadcast_to(cv_, (n, 3))
        if face == spine:
            out = np.broadcast_to(cv_, (n, 3)).copy()
            # rounded spine: a little lighter in the middle; accent bands near both ends
            along = (lx / hx) if face in ("front", "back") else (lz / hz)
            out *= (0.92 + 0.12 * (1 - (ly / hy) ** 2))[:, None]
            if ac is not None:
                band = np.zeros(n, F32)
                for p in (-0.78, 0.78):
                    band = np.maximum(band, clamp01((0.035 - np.abs(along - p)) * hx + 0.5))
                out = out + (ac - out) * band[:, None]
            return out
        # page block, set back between the covers
        cov = clamp01((np.abs(ly) - (hy - c)) * 1.5 + 0.5)
        if spine in ("front", "back"):
            sp = (lz if spine == "front" else -lz) / hz
            wrap = (sp > 0.995).astype(F32) if face in ("left", "right") else np.zeros(n, F32)
        else:
            sp = (lx if spine == "right" else -lx) / hx
            wrap = (sp > 0.995).astype(F32) if face in ("front", "back") else np.zeros(n, F32)
        lines = 0.05 * np.sin(ly / max(c * 0.35, 0.5) * math.pi) ** 2
        p = pg[None, :] * (0.9 - lines - 0.12 * clamp01(1 - (hy - c - np.abs(ly)) / max(1.5 * c, 1)))[:, None]
        m_ = np.maximum(cov, wrap)[:, None]
        return p * (1 - m_) + cv_[None, :] * m_

    cols = {k_: face_col for k_ in ("front", "back", "left", "right", "top")}
    return cuboid(cv, at, size, base=cover, material=material, light=light, yaw=yaw, tilt=tilt, round=0.08,
                  shadow=shadow, cast=cast, opacity=opacity, colors=cols)


# ---- boxes (orthographic cuboid and perspective ground.box share this)


def _hull(pts):
    pts = sorted(set(map(tuple, np.round(np.asarray(pts, np.float64), 4))))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _box_faces(center, axes, half):
    """6 faces: (name, centre, normal, t1, e1, t2, e2)."""
    names = {(0, 1): "right", (0, -1): "left", (1, 1): "top", (1, -1): "bottom", (2, 1): "front", (2, -1): "back"}
    faces = []
    for i in range(3):
        j, k_ = [a for a in range(3) if a != i]
        for sgn in (1, -1):
            n = axes[i] * sgn
            faces.append((names[(i, sgn)], center + n * half[i], n, axes[j], half[j], axes[k_], half[k_]))
    return faces


def _render_box(cv, corners_px, faces, ray, to_screen, visible, colors, material, light, round_w, y_floor_occ,
                opacity, shadow_fn=None, anchors=None, y0=0.0, frame=None):
    hull = _hull(corners_px)
    xs = [p[0] for p in hull]
    ys = [p[1] for p in hull]
    box = _clip_box(cv, min(xs) - 2, min(ys) - 2, max(xs) + 3, max(ys) + 3)
    if box is None:
        return Solid(cv, (0, 0, 1, 1), np.zeros((1, 1), F32), anchors)
    X, Y = _grid_box(*box)
    dpoly = _sdf.sd_polygon(X, Y, hull)
    alpha = _sdf.fill(dpoly)
    (ox, oy, oz), (dx, dy, dz) = ray(X, Y)
    best_t = np.full(X.shape, np.inf, F32)
    best_v = np.full(X.shape, np.inf, F32)
    fid = np.full(X.shape, -1, np.int32)
    UV = []
    vis = [f for f in faces if visible(f)]
    for i, (name, c, n, t1, e1, t2, e2) in enumerate(vis):
        den = n[0] * dx + n[1] * dy + n[2] * dz
        den = np.where(np.abs(den) < 1e-6, 1e-6, den)
        t = ((c[0] - ox) * n[0] + (c[1] - oy) * n[1] + (c[2] - oz) * n[2]) / den
        px_, py_, pz_ = ox + dx * t, oy + dy * t, oz + dz * t
        rx_, ry_, rz_ = px_ - c[0], py_ - c[1], pz_ - c[2]
        u = rx_ * t1[0] + ry_ * t1[1] + rz_ * t1[2]
        v = rx_ * t2[0] + ry_ * t2[1] + rz_ * t2[2]
        viol = np.maximum(np.abs(u) - e1, np.abs(v) - e2)
        inside = viol <= 1e-4
        better = np.where(inside, (best_v > 1e-4) | (t < best_t), (best_v > 1e-4) & (viol < best_v))
        fid = np.where(better, i, fid)
        best_t = np.where(better, t, best_t)
        best_v = np.where(better, np.maximum(viol, 0), best_v)
        UV.append((u, v, py_))
    nx = np.zeros(X.shape, F32)
    ny = np.zeros(X.shape, F32)
    nz = np.ones(X.shape, F32)
    occ = np.ones(X.shape, F32)
    base_c = np.zeros(X.shape + (3,), F32)
    for i, (name, c, n, t1, e1, t2, e2) in enumerate(vis):
        sel = fid == i
        if not sel.any():
            continue
        u, v, wy = UV[i]
        u, v, wy = u[sel], v[sel], wy[sel]
        wn = np.array(n, np.float64)[:, None] * np.ones_like(u)[None]
        if round_w > 0:
            # quarter-round bevel: the normal turns 45 degrees by each edge, so neighbouring faces meet with the
            # same normal and the edge shows no seam
            au = np.clip(1 - (e1 - np.abs(u)) / round_w, 0, 1)
            av = np.clip(1 - (e2 - np.abs(v)) / round_w, 0, 1)
            su = np.sin(au * au * (3 - 2 * au) * (math.pi / 4))
            sv = np.sin(av * av * (3 - 2 * av) * (math.pi / 4))
            wn = wn * np.sqrt(np.clip(1 - su * su - sv * sv, 0.0, 1))[None] \
                + np.array(t1)[:, None] * (np.sign(u) * su)[None] + np.array(t2)[:, None] * (np.sign(v) * sv)[None]
        sn = to_screen(wn)
        l_ = np.sqrt((sn ** 2).sum(0)) + 1e-9
        nx[sel], ny[sel], nz[sel] = sn[0] / l_, sn[1] / l_, sn[2] / l_
        if y_floor_occ and abs(n[1]) < 0.5:
            occ[sel] = 1 - 0.4 * np.exp(-np.maximum(wy - y0, 0) / y_floor_occ)
        col = colors.get(name, colors["side"])
        if callable(col) and frame is not None:
            # face colour from local box coordinates: col(face, lx, ly, lz) → colour or (n, 3) colours
            fc, fax = frame
            P_ = [c[q] + u * t1[q] + v * t2[q] - fc[q] for q in range(3)]
            lx, ly, lz = (P_[0] * a_[0] + P_[1] * a_[1] + P_[2] * a_[2] for a_ in fax)
            cc = as_color(col(name, lx, ly, lz))
            base_c[sel] = cc if cc.ndim == 1 else cc.reshape(-1, 3)
        else:
            cc = _crop_color(col, box, (cv.h, cv.w))
            base_c[sel] = cc if cc.ndim == 1 else cc[sel]
    m = _material(material)
    col, a2 = _illuminate(nx, ny, nz, base_c, m, light, occ)
    if a2 is not None:
        alpha = alpha * a2
    sol = Solid(cv, box, alpha, anchors, (nx, ny, nz), material, light, None, occ)
    sol.face_id = fid
    sol.faces = [f[0] for f in vis]
    if shadow_fn:
        shadow_fn(sol)
    _finish_solid(cv, col, alpha, box, opacity)
    return sol


def cuboid(cv, at, size=(0.3, 0.2, 0.2), base="#c58b4c", top=None, material="matte", light=None, yaw=30.0,
           tilt=None, round=0.0, shadow=True, cast=False, opacity=1.0, colors=None, name=None, alias=None):
    """A three-quarter-view box standing at ``at`` (centre of its footprint).

    size: (width, depth, height) as fractions of the canvas height (scaled at a Spot); yaw: turn in degrees
    (0 = front face square to the viewer); tilt: view from above; top: colour of the top face; round: 0~1 edge
    softening (cushions, soap bars); colors: per-face colours ``front`` / ``back`` / ``left`` / ``right`` / ``top``,
    each a colour or a function ``f(face, lx, ly, lz)`` of local box coordinates (centred, same units as size × px)
    returning colours. Anchors: ``base``, ``top``, ``corners`` (8 px points), ``face_<name>`` (4 px points of each
    visible face, for labels). Returns a ``Solid``.
    """
    colors_ = colors
    light = _resolve_light(light)
    bx, by, k, t0 = _pos(cv, at)
    se = float(tilt if tilt is not None else (t0 if t0 is not None else VIEW_TILT))
    se = min(max(se, 0.0), 0.95)
    ce = math.sqrt(1 - se * se)
    w, d, h = [s * k * cv.h for s in size]
    a = math.radians(yaw)
    ax = np.array([math.cos(a), 0, math.sin(a)])
    az = np.array([-math.sin(a), 0, math.cos(a)])
    ay = np.array([0.0, 1.0, 0.0])
    half = (w / 2, h / 2, d / 2)
    center = np.array([0.0, h / 2, 0.0])
    axes = (ax, ay, az)

    def proj(p):
        return bx + p[0], by - p[1] * ce + p[2] * se
    corners = [center + sx * ax * half[0] + sy * ay * half[1] + sz * az * half[2]
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    cpx = [proj(p) for p in corners]
    faces = _box_faces(center, axes, half)
    view = np.array([0, se, ce])

    def ray(X, Y):
        su = by - Y
        T = (w + h + d) * 2
        return (X - bx, su * ce + T * se, -su * se + T * ce), (0.0, -se, -ce)

    def to_screen(n):
        return np.stack([n[0], -n[1] * ce + n[2] * se, n[1] * se + n[2] * ce])

    def visible(f):
        return float(np.dot(f[2], view)) > 1e-3
    anchors = dict(base=(bx, by), top=proj(center + ay * half[1]), center=proj(center), corners=cpx)
    for f in faces:
        if visible(f):
            c, t1, e1, t2, e2 = f[1], f[3], f[4], f[5], f[6]
            anchors["face_" + f[0]] = [proj(c + s1 * t1 * e1 + s2 * t2 * e2)
                                       for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    colors = {"side": base, "top": top if top is not None else base}
    colors.update(colors_ or {})

    def shadow_fn(sol):
        _attach(sol, at, cv)
        if cast:
            cast_shadow(cv, sol, light, **(cast if isinstance(cast, dict) else {}))
        if shadow:
            foot = _hull([proj(p) for p in corners if p[1] < 1e-6])
            s = shadow if isinstance(shadow, (int, float)) and not isinstance(shadow, bool) else 0.6
            _footprint_shadow(cv, foot, max(0.08 * min(w, d), 1.0), s, light)
    rw = round * min(half) if round else 0.0
    return _render_box(cv, cpx, faces, ray, to_screen, visible, colors, material, light, rw, h * 0.18, opacity,
                       shadow_fn, anchors, frame=(center, axes))


def _footprint_shadow(cv, pts, soft, strength, light, color=None):
    """Contact shadow hugging a footprint polygon (pixel points): dark at the edge, fading outwards over ``soft``."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    r = 6 * soft
    box = _clip_box(cv, min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r)
    if box is None:
        return
    X, Y = _grid_box(*box)
    d = np.maximum(_sdf.sd_polygon(X, Y, pts), 0)
    a = strength * (0.6 * np.exp(-d / soft) + 0.4 * np.exp(-d / (2.5 * soft))) * smoothstep(r, 0.7 * r, d)
    _multiply(cv, a, box, _shadow_color(light, color) * 0.7)


# ===================================================================== perspective ground


class Ground:
    """A floor plane in perspective.

    ``Ground(cv, horizon=0.45, vanishing=0.5, camera_height=1.6, focal=1.0)`` — world units are metres-like:
    ``x`` across (0 = straight ahead), ``depth`` away from the camera, ``y`` up.

    * ``at(x, depth)`` → (px, py, scale px per unit); ``place(x, depth)`` → ``Spot`` (pass to any solid; sizes in
      world units); ``spot(u, v)`` → ``Spot`` for a screen point on the floor; ``project(x, y, z)`` → pixels.
    * ``uv()`` → per-pixel world (x, depth) for sampling any texture in perspective; ``mask()`` = the floor area;
      ``area(x0, x1, z0, z1)`` → (mask, inset) of a rectangle on the floor (rugs, mats).
    * ``checker`` / ``tiles`` / ``planks`` → anti-aliased colour maps; ``grid_mask`` → line mask.
    * ``box(x, depth, size=(w, d, h), yaw)`` → a perspective-correct box standing on the floor.
    """

    def __init__(self, cv, horizon=0.45, vanishing=0.5, camera_height=1.6, focal=1.0, camera_x=0.0):
        self.cv = cv
        self.f = float(focal) * cv.h
        self.hor = float(horizon) * cv.h
        self.vp = float(vanishing) * cv.w
        self.h = float(camera_height)
        self.cam_x = float(camera_x)
        self._uv = None
        self._owner = None           # the Surface / Room that Spots from this ground belong to

    # ---- placement
    def project(self, x, y, z):
        """World point → canvas pixels."""
        z = np.maximum(np.asarray(z, np.float64), 1e-6)
        px_ = self.vp + self.f * (np.asarray(x, np.float64) - self.cam_x) / z
        py_ = self.hor + self.f * (self.h - np.asarray(y, np.float64)) / z
        if np.ndim(px_) == 0:
            return float(px_), float(py_)
        return px_, py_

    def at(self, x, depth):
        """World floor point → (px, py, scale in px per world unit)."""
        px_, py_ = self.project(x, 0.0, depth)
        return px_, py_, self.f / max(float(depth), 1e-6)

    def tilt(self, depth):
        return self.h / math.hypot(self.h, float(depth))

    def place(self, x, depth):
        """World floor point → ``Spot`` (u, v, k, tilt); solids at a Spot take sizes in world units."""
        px_, py_, s = self.at(x, depth)
        return Spot(px_ / self.cv.w, py_ / self.cv.h, s / self.cv.h, self.tilt(depth), self._owner, 0.0,
                    self.cv.h / self.cv.w)

    def depth(self, v):
        """Depth of the floor at screen row v (fraction)."""
        return self.f * self.h / max(v * self.cv.h - self.hor, 1e-3)

    def spot(self, u, v):
        """Screen point (u, v) on the floor → ``Spot`` with the perspective scale there."""
        z = self.depth(v)
        return Spot(u, v, self.f / z / self.cv.h, self.tilt(z), self._owner, 0.0, self.cv.h / self.cv.w)

    # ---- textures
    def _rows(self):
        return max(int(math.floor(self.hor + 1)), 0)

    def uv(self):
        """Per-pixel world coordinates (x, depth) of the floor (full-frame; above the horizon depth is huge)."""
        if self._uv is None:
            H, W = self.cv.h, self.cv.w
            X, Y = self.cv.xy()
            dy = np.maximum(Y - self.hor, 0.35)
            Z = (self.f * self.h / dy).astype(F32)
            Xw = (self.cam_x + (X - self.vp) * Z / self.f).astype(F32)
            self._uv = (Xw, Z)
        return self._uv

    def _tex(self, box):
        Xw, Z = self.uv()
        fw = self.footprint()
        if box is not None:
            x0, y0, x1, y1 = box
            return Xw[y0:y1, x0:x1], Z[y0:y1, x0:x1], fw[y0:y1, x0:x1]
        return Xw, Z, fw

    def footprint(self):
        """World size of one pixel on the floor (for anti-aliasing textures)."""
        Xw, Z = self.uv()
        return np.maximum(Z / self.f, Z * Z / (self.f * self.h)).astype(F32)

    def mask(self):
        """Anti-aliased mask of the floor (everything below the horizon)."""
        _, Y = self.cv.xy()
        return clamp01(Y - self.hor + 0.5)

    def _lines(self, coord, period, width, fw):
        dist = np.abs((coord / period + 0.5) % 1.0 - 0.5) * period
        cov = np.clip((width / 2 - dist) / np.maximum(fw, 1e-6) + 0.5, 0, 1)
        far = smoothstep(period * 0.15, period * 0.6, fw)
        return cov * (1 - far) + min(width / period, 1.0) * far

    def grid_mask(self, size=1.0, width=0.03, box=None):
        """Anti-aliased grid lines every ``size`` world units (line ``width`` in world units)."""
        Xw, Z, fw = self._tex(box)
        m = np.maximum(self._lines(Xw, size, width, fw), self._lines(Z, size, width, fw))
        gm = self.mask()
        if box is not None:
            gm = gm[box[1]:box[3], box[0]:box[2]]
        return (m * gm).astype(F32)

    def checker(self, size=1.0, colors=("#e8e2d6", "#2d2a2a"), box=None):
        """Checkerboard colour map in perspective (filtered so it never moirés in the distance)."""
        Xw, Z, fw = self._tex(box)
        a, b = rgb(colors[0]), rgb(colors[1])
        par = ((np.floor(Xw / size) + np.floor(Z / size)) % 2).astype(F32)
        de = np.minimum(np.abs((Xw / size + 0.5) % 1 - 0.5), np.abs((Z / size + 0.5) % 1 - 0.5)) * size
        t = np.clip(de / np.maximum(fw, 1e-6) * 1.5, 0, 1) * (1 - smoothstep(size * 0.25, size * 0.8, fw))
        avg = (a + b) / 2
        sel = (a * (1 - par[..., None]) + b * par[..., None])
        return (avg + (sel - avg) * t[..., None]).astype(F32)

    def tiles(self, size=0.5, grout=0.02, colors=("#d9d2c5", "#cfc6b6", "#e2dbcf"), grout_color="#9c9283",
              jitter=0.06, seed=0, aspect=1.0, box=None):
        """Floor tiles: per-tile colour variation and grout lines, in perspective."""
        Xw, Z, fw = self._tex(box)
        sx, sz = size * aspect, size
        i, j = np.floor(Xw / sx), np.floor(Z / sz)
        pal = np.stack([rgb(c) for c in colors])
        idx = (_hash01(i, j, seed=seed) * len(pal)).astype(int) % len(pal)
        col = pal[idx] * (1 + jitter * (2 * _hash01(i, j, seed=seed + 7) - 1))[..., None]
        g = np.maximum(self._lines(Xw, sx, grout, fw), self._lines(Z, sz, grout, fw))
        return (col + (rgb(grout_color) - col) * g[..., None]).astype(F32)

    def planks(self, width=0.2, length=2.0, colors=("#a8744a", "#8f5e3a", "#b98656", "#9a6844"), seam="#4a2e1c",
               gap=0.006, grain=1.0, seed=0, direction="depth", variation=0.06, box=None):
        """Wooden planks in perspective: staggered ends, per-plank tone, grain along the boards.

        direction: ``"depth"`` (boards run away from the viewer, lines converge) or ``"across"``.
        """
        Xw, Z, fw = self._tex(box)
        a, b = (Xw, Z) if direction == "depth" else (Z, Xw)
        i = np.floor(a / width)
        off = _hash01(i, seed=seed) * length
        bb = b + off
        j = np.floor(bb / length)
        pal = np.stack([rgb(c) for c in colors])
        hsh = _hash01(i, j, seed=seed)
        col = pal[(hsh * len(pal)).astype(int) % len(pal)]
        col = col * (1 + variation * (2 * _hash01(i, j, seed=seed + 3) - 1))[..., None]
        far = smoothstep(0.06 * width, 0.4 * width, fw)[..., None]
        col = col + (pal.mean(0) - col) * far
        if grain:
            fa = (a / width - i)
            lb = bb / width                                  # along the board, in board widths
            warp = fbm2d_at(fa * 1.2 + hsh * 37.0, lb * 0.35 + i * 3.1, 3, seed=seed + 11)
            fine = fbm2d_at(fa * 6.0 + i * 1.7, lb * 0.08, 2, seed=seed + 12)
            ring = fa * (3.0 + 3.0 * hsh) + 1.6 * warp + 0.25 * fine
            lines = np.abs(((ring % 1.0) - 0.5) * 2) ** 3          # thin dark late-wood lines
            figure = fbm2d_at(fa * 0.8 + hsh * 11, lb * 0.12, 2, seed=seed + 13)
            amp = grain * (1 - smoothstep(0.03 * width, 0.25 * width, fw))
            col = col * (1 - 0.2 * amp * lines + 0.07 * grain * figure)[..., None]
        sa = self._lines(a, width, gap, fw)
        sb = self._lines(bb, length, gap * 1.2, fw)
        s = np.maximum(sa, sb)
        da = np.abs((a / width + 0.5) % 1.0 - 0.5) * width
        bev = 1 - 0.12 * np.exp(-da / (gap * 2.5)) * (1 - smoothstep(0.05 * width, 0.3 * width, fw))
        col = col * bev[..., None]
        return (col + (rgb(seam) - col) * s[..., None]).astype(F32)

    def area(self, x0, x1, z0, z1, box=None):
        """A rectangle lying on the floor (rugs, mats, tiles, shadows): ``(mask, inset)`` where ``inset`` is the
        world distance to its border (for borders and patterns). x across, z depth, world units."""
        Xw, Z, fw = self._tex(box)
        fw = np.maximum(fw, 1e-6)
        inset = np.minimum(np.minimum(Xw - x0, x1 - Xw), np.minimum(Z - z0, z1 - Z))
        gm = self.mask()
        if box is not None:
            gm = gm[box[1]:box[3], box[0]:box[2]]
        return (clamp01(inset / fw + 0.5) * gm).astype(F32), inset.astype(F32)

    def fade(self, color_map, to="#000000", near=None, far=None, amount=0.5):
        """Blend a floor colour map towards ``to`` with distance (depth from ``near`` to ``far``)."""
        _, Z = self.uv()
        near = near if near is not None else self.depth(1.0)
        far = far if far is not None else near * 4
        t = smoothstep(near, far, Z) * amount
        c = as_color(color_map)
        return (c + (rgb(to) - c) * t[..., None]).astype(F32)

    def floor(self, texture="checker", colors=None, size=None, haze=0.5, haze_color=None, **kw):
        """Paint the whole floor in one call and return its mask.

        texture: ``checker`` / ``tiles`` / ``planks`` / ``grid`` / a colour / a colour map; colors / size: passed to
        the texture; haze: 0~1 width of the band near the horizon that melts into ``haze_color`` (default: the
        colour already on the canvas just above the horizon, so the floor meets the sky without a seam).
        """
        cv = self.cv
        _texture_or_color("floor texture", texture, ("checker", "tiles", "planks", "grid"))
        if isinstance(texture, str) and texture in ("checker", "tiles", "planks", "grid"):
            if colors is not None:
                kw["colors"] = colors
            if size is not None:
                kw["width" if texture == "planks" else "size"] = size
            if texture == "grid":
                gc = kw.pop("colors", ("#e9e6df", "#9c9890"))
                g = self.grid_mask(**kw)
                a, b = rgb(gc[0]), rgb(gc[1])
                col = a + (b - a) * g[..., None]
            else:
                col = getattr(self, texture)(**kw)
        else:
            col = np.broadcast_to(as_color(texture), (cv.h, cv.w, 3)) if as_color(texture).ndim == 1 \
                else as_color(texture)
        col = np.array(col, F32)
        if haze:
            row = int(min(max(self.hor - 2, 0), cv.h - 1))
            hc = rgb(haze_color)[None, None, :] if haze_color is not None else cv.buf[row][None, :, :].copy()
            _, Y = cv.xy()
            t = np.exp(-np.maximum(Y - self.hor, 0) / max(0.12 * haze * cv.h, 1.0))[..., None]
            col = col + (hc - col) * t
        m = self.mask()
        cv.paint(col, m)
        return m

    def rug(self, x0, x1, z0, z1, color="#b5523b", border="#efe2c8", border_width=0.1, pattern="diamonds",
            pattern_color="#2f4d7a", shadow=0.35, lighting=None):
        """A rug lying on the floor: world rectangle x0..x1 (across) × z0..z1 (depth), with a border band
        (``border_width`` world units, None for none) and a ``pattern`` (``diamonds`` / ``stripes`` / ``None``).
        ``lighting``: optional function (x, z) → brightness factor (``Room.rug`` passes its window light).
        Returns the full-frame mask."""
        cv = self.cv
        corners = [self.project(x, 0.0, z) for x in (x0, x1) for z in (z0, z1)]
        bx0, by0 = min(p[0] for p in corners), min(p[1] for p in corners)
        bx1, by1 = max(p[0] for p in corners), max(p[1] for p in corners)
        pad = 0.02 * cv.h
        box = _clip_box(cv, bx0 - pad, by0 - pad, bx1 + pad, by1 + pad)
        if box is None:
            return np.zeros((cv.h, cv.w), F32)
        m, inset = self.area(x0, x1, z0, z1, box=box)
        Xw, Z, fw = self._tex(box)
        col = np.broadcast_to(rgb(color), m.shape + (3,)).astype(F32)
        fw = np.maximum(fw, 1e-6)
        bw = float(border_width or 0.0)
        if pattern:
            if pattern == "diamonds":
                cell = 0.5 * min(x1 - x0, z1 - z0) / 3
                a = (np.abs(((Xw - x0) / cell) % 2 - 1) + np.abs(((Z - z0) / cell) % 2 - 1))
                pm = clamp01((0.18 - np.abs(a - 1)) * cell / fw + 0.5)
            else:
                period = max(x1 - x0, z1 - z0) / 9
                pm = clamp01((0.25 * period - np.abs(((Z - z0) % period) - period / 2)) / fw + 0.5)
            inner = clamp01((inset - bw * 1.6) / fw + 0.5)
            col = col + (rgb(pattern_color) - col) * (pm * inner)[..., None]
        if bw:
            band = clamp01((bw - inset) / fw + 0.5) * clamp01((inset - bw * 0.25) / fw + 0.5)
            col = col + (rgb(border) - col) * band[..., None]
        n = fbm2d_at(Xw * 40, Z * 40, 2, seed=5)
        col = col * (1 + 0.05 * n)[..., None] * (1 - 0.12 * np.exp(-np.maximum(inset, 0) / 0.02))[..., None]
        if lighting is not None:
            col = col * np.asarray(lighting(Xw, Z), F32)[..., None]
        if shadow:
            sh = clamp01(1 + inset / 0.03) * (1 - m) * shadow * 0.5         # thin soft edge around the rug
            _multiply(cv, sh, box, rgb("#2a2018"))
        cv.paste(col, box[0], box[1], mask=m)
        return _full(cv, m, box)

    # ---- objects
    def box(self, x, depth, size=(1.0, 0.6, 0.8), base="#8a6a52", top=None, material="matte", light=None,
            yaw=0.0, round=0.0, lift=0.0, shadow=True, cast=False, colors=None, opacity=1.0):
        """A box standing on the floor at world (x, depth) — centre of its footprint — drawn in true perspective.

        size: (width, depth, height) in world units; yaw: degrees; round: 0~1 edge softening; lift: height of its
        bottom above the floor (stack cushions on a sofa base, books on a table); colors: optional dict of
        per-face colours (``front``, ``left``, ``right``, ``top``, ``back``). Returns a ``Solid``.
        """
        light = _resolve_light(light)
        w, d, h = size
        a = math.radians(yaw)
        ax = np.array([math.cos(a), 0, -math.sin(a)])
        az = np.array([math.sin(a), 0, math.cos(a)])
        ay = np.array([0.0, 1.0, 0.0])
        half = (w / 2, h / 2, d / 2)
        center = np.array([x, lift + h / 2, depth])
        axes = (ax, ay, -az)               # third axis points towards the camera → "front"
        corners = [center + sx * ax * half[0] + sy * ay * half[1] + sz * az * half[2]
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        cpx = [self.project(*p) for p in corners]
        faces = _box_faces(center, axes, half)
        cam = np.array([self.cam_x, self.h, 0.0])

        def ray(X, Y):
            return (self.cam_x, self.h, 0.0), ((X - self.vp) / self.f, (self.hor - Y) / self.f, 1.0)

        def to_screen(n):
            return np.stack([n[0], -n[1], -n[2]])

        def visible(f):
            return float(np.dot(f[2], cam - f[1])) > 1e-6
        anchors = dict(base=self.project(x, lift, depth), top=self.project(x, lift + h, depth),
                       center=self.project(x, lift + h / 2, depth), corners=cpx)
        for f in faces:
            if visible(f):
                c, t1, e1, t2, e2 = f[1], f[3], f[4], f[5], f[6]
                anchors["face_" + f[0]] = [self.project(*(c + s1 * t1 * e1 + s2 * t2 * e2))
                                           for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cols = {"side": base, "top": top if top is not None else base}
        cols.update(colors or {})

        def shadow_fn(sol):
            sol.surface = self._owner
            sol.lift_px = self.project(x, 0.0, depth)[1] - self.project(x, lift, depth)[1]
            if cast:
                cast_shadow(cv=self.cv, shape=sol, light=light, **(cast if isinstance(cast, dict) else {}))
            if shadow:
                foot = _hull([self.project(*p) for p in corners if p[1] < lift + 1e-9])
                s = shadow if isinstance(shadow, (int, float)) and not isinstance(shadow, bool) else 0.6
                soft = max(0.06 * min(w, d) * self.f / max(depth, 1e-3), 1.0)
                _footprint_shadow(self.cv, foot, soft, s, light)
        rw = round * min(half) if round else 0.0
        return _render_box(self.cv, cpx, faces, ray, to_screen, visible, cols, material, light, rw, h * 0.2,
                           opacity, shadow_fn, anchors, lift, frame=(center, axes))


# ===================================================================== surfaces for still lifes


class Surface(Ground):
    """A tabletop / shelf / counter: a Ground whose unit is the canvas height at the middle of the surface.

    ``place(u, depth)`` → ``Spot``: u = screen x fraction, depth 0 = front edge … 1 = back edge (measured on screen,
    0.5 = halfway between the two edges); sizes at that Spot
    are fractions of the canvas height at the middle of the surface (larger in front, smaller further back).
    ``spot(u, v)`` takes a screen point.
    """

    def __init__(self, cv, top, front, horizon, vanishing=0.5, x0=0.0, x1=1.0, wall=False):
        if top * cv.h <= horizon * cv.h + 1:
            raise ValueError("the surface top must be below the horizon")
        cam_h = (top + front) / 2 - horizon          # world unit: canvas height at the middle of the surface
        super().__init__(cv, horizon, vanishing, camera_height=cam_h, focal=1.0)
        self.top = top
        self.front = front
        self.x0, self.x1 = float(x0), float(x1)
        self.wall = bool(wall)
        self.z_front = self.depth(front)
        self.z_back = self.depth(top)
        self._owner = self

    def place(self, u, depth=0.5):
        v = self.front + (self.top - self.front) * float(depth)
        return self.spot(u, v)

    def _shadow_regions(self, box):
        """(top-surface weight, wall weight or None) crops for a pixel box: where cast shadows may fall."""
        X, Y = _grid_box(*box)
        W_, H_ = self.cv.w, self.cv.h
        xm = clamp01(X - self.x0 * W_ + 0.5) * clamp01(self.x1 * W_ - X + 0.5)
        top = clamp01(Y - self.top * H_ + 0.5) * clamp01(self.front * H_ - Y + 0.5) * xm
        wall = clamp01(self.top * H_ - Y + 0.5) if self.wall else None
        return top, wall

    def _back_y(self):
        """Pixel row where the top meets the wall behind it (None without a wall)."""
        return self.top * self.cv.h if self.wall else None


def _stone(ground, colors, seed, box=None):
    Xw, Z, _ = ground._tex(box)
    n = fbm2d_at(Xw * 2.0, Z * 2.0, 5, seed=seed)
    veins = np.exp(-np.abs(fbm2d_at(Xw * 1.2 + 3 * n, Z * 1.2, 4, seed=seed + 5)) * 18)
    a, b = rgb(colors[0]), rgb(colors[1])
    col = a * (1 + 0.04 * n)[..., None]
    return (col + (b - col) * (veins * 0.6)[..., None]).astype(F32)


def _cloth(ground, color, seed, box=None):
    Xw, Z, fw = ground._tex(box)
    w = 0.012
    amp = 0.06 * (1 - smoothstep(w * 0.3, w * 1.2, fw))
    weave = np.sin(Xw / w * np.pi) * np.sin(Z / w * np.pi)
    n = fbm2d_at(Xw * 3, Z * 3, 3, seed=seed)
    return (rgb(color) * (1 + amp * weave + 0.04 * n)[..., None]).astype(F32)


def tabletop(cv, top=0.55, front=0.9, horizon=0.3, vanishing=0.5, kind="wood", colors=None, edge=0.04,
             edge_color=None, x0=0.0, x1=1.0, plank=0.16, plank_length=3.0, seed=0, light=None, sheen=0.25,
             under=0.08, wall=False, name=None, alias=None):
    """Draw a table top seen in perspective (back edge at row ``top``, front edge at ``front``) with a front edge
    of thickness ``edge``; ``kind``: ``wood`` (planks running across), ``stone`` (marble), ``cloth`` or a colour;
    ``under``: depth (fraction of the height) of the shadow below the front edge; ``wall=True`` when it stands
    against a wall (cast shadows then climb the wall behind it).
    ``name``: what the scene manifest calls it (default ``tabletop``).
    Returns a ``Surface`` for placing objects (``surf.place(u, depth)``)."""
    _texture_or_color("tabletop kind", kind, ("wood", "stone", "cloth"))
    light = _resolve_light(light)
    s = Surface(cv, top, front, horizon, vanishing, x0, x1, wall)
    W_, H_ = cv.w, cv.h
    box = _clip_box(cv, x0 * W_ - 1, top * H_ - 1, x1 * W_ + 1, front * H_ + 1)
    if box is None:
        return s
    X, Y = _grid_box(*box)
    if kind == "wood":
        cols = colors or ("#8a5a36", "#7a4e2e", "#94643c", "#83552f")
        tex = s.planks(width=plank, length=plank_length, colors=cols, seam="#3a2414", direction="across", seed=seed,
                       gap=0.004, box=box)
        ec = rgb(edge_color) if edge_color is not None else rgb(cols[1]) * 0.62
    elif kind == "stone":
        cols = colors or ("#e9e4dc", "#9a948c")
        tex = _stone(s, cols, seed, box)
        ec = rgb(edge_color) if edge_color is not None else rgb(cols[0]) * 0.8
    elif kind == "cloth":
        c = colors[0] if colors else "#e8e0d0"
        tex = _cloth(s, c, seed, box)
        ec = rgb(edge_color) if edge_color is not None else rgb(c) * 0.8
    else:
        tex = np.broadcast_to(_crop_color(kind, box, (H_, W_)), X.shape + (3,))
        ec = rgb(edge_color) if edge_color is not None else (rgb(kind) * 0.7 if isinstance(kind, str) else
                                                              rgb("#806040"))
    # light falloff: brighter on the light's side, a little darker at the back
    L = light.dir2
    U = X / W_
    side = 1 + 0.12 * (0.5 - U) * np.sign(-L[0] or 1) * 2
    depth_t = clamp01((Y - top * H_) / max((front - top) * H_, 1))
    lit = (0.82 + 0.18 * depth_t) * side
    tex = tex * lit[..., None]
    if sheen:
        refl = np.exp(-((U - (0.5 + 0.35 * L[0])) / 0.25) ** 2) * (1 - depth_t) ** 1.5
        tex = tex + rgb(light.sky) * (sheen * 0.25 * refl)[..., None]
    xm = clamp01(X - x0 * W_ + 0.5) * clamp01(x1 * W_ - X + 0.5)
    topm = clamp01(Y - top * H_ + 0.5) * clamp01(front * H_ - Y + 0.5) * xm
    cv.paste(tex.astype(F32), box[0], box[1], mask=topm)
    if edge:
        e0, e1 = front * H_, (front + edge) * H_
        ebox = _clip_box(cv, x0 * W_ - 1, e0 - 1, x1 * W_ + 1, e1 + 1)
        if ebox is not None:
            X, Y = _grid_box(*ebox)
            xm = clamp01(X - x0 * W_ + 0.5) * clamp01(x1 * W_ - X + 0.5)
            em = clamp01(Y - e0 + 0.5) * clamp01(e1 - Y + 0.5) * xm
            t = clamp01((Y - e0) / max(e1 - e0, 1))
            ecol = ec[None, None, :] * (1.0 - 0.35 * t)[..., None]
            if kind == "wood":
                n = fbm2d_at(X / H_ * 40, Y / H_ * 3, 3, seed=seed + 9)
                ecol = ecol * (1 + 0.06 * n)[..., None]
            cv.paste(ecol.astype(F32), ebox[0], ebox[1], mask=em)
            bevel = clamp01(1.6 * H_ / 1080 - np.abs(Y - e0 - 0.8 * H_ / 1080)) * xm
            cv.paste(np.broadcast_to(rgb(light.color) * 0.9, X.shape + (3,)).astype(F32), ebox[0], ebox[1],
                     mask=bevel * 0.35, mode="screen")
        if front + edge < 1 and under:
            bbox = _clip_box(cv, x0 * W_ - 0.05 * H_, e1 - 1, x1 * W_ + 0.05 * H_, e1 + 4 * under * H_)
            if bbox is not None:
                X, Y = _grid_box(*bbox)
                soft = max(under * H_ * 0.4, 1.0)
                xm = smoothstep(x0 * W_ - soft, x0 * W_ + soft, X) * smoothstep(x1 * W_ + soft, x1 * W_ - soft, X)
                below = clamp01(Y - e1 + 0.5) * xm * np.exp(-(Y - e1).clip(0) / (under * H_))
                _multiply(cv, below * 0.6, bbox, rgb(light.shadow) * 0.5)
    return s


def shelf(cv, y=0.6, x0=0.1, x1=0.9, depth=0.05, thickness=0.025, color="#b98a5c", horizon=0.35, name=None,
          alias=None, **kw):
    """A wall shelf board: top surface of ``depth`` (screen fraction) ending at row ``y``, front face
    ``thickness``; objects on it cast shadows onto the board and the wall behind. Returns a ``Surface``."""
    return tabletop(cv, top=y - depth, front=y, horizon=min(horizon, y - depth - 0.02), kind=kw.pop("kind", "wood"),
                    colors=kw.pop("colors", (color, color, color)), edge=thickness, x0=x0, x1=x1, sheen=0.1,
                    plank=10.0, plank_length=1e4, under=kw.pop("under", 0.025), wall=kw.pop("wall", True), name=name,
                    alias=alias, **kw)


def counter(cv, top=0.58, front=0.82, horizon=0.35, kind="stone", edge=0.05, name=None, alias=None, **kw):
    """A kitchen / shop counter (stone top, thick front edge). Returns a ``Surface``."""
    return tabletop(cv, top=top, front=front, horizon=horizon, kind=kind, edge=edge, name=name, alias=alias, **kw)


# ===================================================================== room


class Room:
    """A one-point-perspective interior: back wall, two side walls, floor and ceiling.

    ``Room(cv, horizon=0.42, vanishing=0.5, back=0.5, width=4.4, height=2.8, eye=1.5)`` — ``back`` = back-wall
    width as a fraction of the canvas width; world units are metres (x across, y up, z depth). Then
    ``room.draw(floor="planks", window="left")`` paints walls, baseboards, window, sunlight patch and corner
    occlusion. ``room.place(x, z)`` → ``Spot`` on the floor, ``room.box(...)`` → furniture blocks,
    ``room.picture(...)`` → framed picture on any wall, ``room.ground`` → the matching ``Ground``.
    """

    def __init__(self, cv, horizon=0.42, vanishing=0.5, back=0.5, width=4.4, height=2.8, eye=1.5, focal=1.0):
        self.cv = cv
        self.width, self.height, self.eye = float(width), float(height), float(eye)
        self.ground = Ground(cv, horizon, vanishing, eye, focal)
        g = self.ground
        self.zb = g.f * self.width / (back * cv.w)
        self.window = None
        self.sun_dir = None
        self._world = None
        g._owner = self

    # ---- placement
    def place(self, x, z):
        """Floor point → ``Spot`` (solids take sizes in metres there)."""
        return self.ground.place(x, z)

    def project(self, x, y, z):
        return self.ground.project(x, y, z)

    @property
    def z_near(self):
        return self.ground.depth(1.0)

    def wall_point(self, wall, a, y):
        """Pixel position of a point on a wall: back (a = x), left / right (a = depth z)."""
        w2 = self.width / 2
        if wall == "back":
            return self.project(a, y, self.zb)
        return self.project(-w2 if wall == "left" else w2, y, a)

    def box(self, x, z, size, **kw):
        """Furniture block on the floor (see ``Ground.box``)."""
        return self.ground.box(x, z, size, **kw)

    def rug(self, x0, x1, z0, z1, **kw):
        """A rug on the floor, world rectangle x0..x1 × z0..z1 in metres, lit by the room's window (see
        ``Ground.rug`` for colours and patterns; call after ``draw``). Returns the mask."""
        if self.window and "lighting" not in kw:
            kw["lighting"] = lambda x, z: 0.66 + 0.7 * self.sunlight(x, np.zeros_like(x), z)
        return self.ground.rug(x0, x1, z0, z1, **kw)

    def _shadow_regions(self, box):
        X, Y = _grid_box(*box)
        P_ = self._polys()
        floor = _sdf.fill(_sdf.sd_polygon(X, Y, P_["floor"]))
        back = _sdf.fill(_sdf.sd_polygon(X, Y, P_["back"]))
        return floor, back

    def _back_y(self):
        return self.ground.project(0.0, 0.0, self.zb)[1]

    def wall_quad(self, wall, center, size):
        """Four pixel corners (top-left, top-right, bottom-right, bottom-left) of a rectangle on a wall:
        ``center=(a, y)`` (a = x on the back wall, depth on a side wall), ``size=(w, h)`` in metres."""
        ca, cb = center
        hw, hh = size[0] / 2, size[1] / 2
        sgn = -1 if wall == "right" else 1
        return [self.wall_point(wall, ca + sgn * sa * hw, cb + sb * hh) for sa, sb in ((-1, 1), (1, 1), (1, -1), (-1, -1))]

    def panel(self, wall="back", center=(0.0, 1.6), size=(1.2, 0.8), color="#2b2f2c", frame="#6b4a2e",
              frame_width=0.04, image=None, shadow=0.3):
        """A flat board on a wall — menu board, sign, blackboard, poster, screen: a plain ``color`` (or an
        ``image``) with an optional ``frame`` colour. Returns the 4 pixel corners of the board (for ``cv.text`` or
        pasting), as ``wall_quad``."""
        self.picture(wall, center, size, image=image if image is not None else np.broadcast_to(rgb(color), (4, 4, 3)),
                     frame=frame or color, frame_width=frame_width if frame else 1e-4, mat=color, mat_width=0.0,
                     shadow=shadow)
        fw = frame_width if frame else 0.0
        return self.wall_quad(wall, center, (size[0] - 2 * fw, size[1] - 2 * fw))

    # ---- geometry
    def _polys(self):
        g, w2, H_ = self.ground, self.width / 2, self.height
        zb = self.zb
        bl, br = g.project(-w2, 0, zb), g.project(w2, 0, zb)
        tl, tr = g.project(-w2, H_, zb), g.project(w2, H_, zb)
        far = 0.02

        def ext(x, y):
            return g.project(x, y, far)
        return dict(back=[tl, tr, br, bl],
                    floor=[bl, br, ext(w2, 0), ext(-w2, 0)],
                    ceiling=[tl, ext(-w2, H_), ext(w2, H_), tr],
                    left=[tl, bl, ext(-w2, 0), ext(-w2, H_)],
                    right=[tr, ext(w2, H_), ext(w2, 0), br])

    def world(self):
        """Per-surface world coordinates: dict name → (x, y, z) full-frame arrays."""
        if self._world is None:
            g, cv = self.ground, self.cv
            X, Y = cv.xy()
            f, hor, vp = g.f, g.hor, g.vp
            w2 = self.width / 2
            dxr = (X - vp) / f
            dyr = (hor - Y) / f
            safe = lambda a: np.where(np.abs(a) < 1e-5, 1e-5, a)  # noqa: E731
            out = {}
            t = self.zb
            out["back"] = (g.cam_x + dxr * t, self.eye + dyr * t, np.full_like(X, t))
            t = self.eye / safe(-dyr)
            out["floor"] = (g.cam_x + dxr * t, np.zeros_like(X), t)
            t = (self.height - self.eye) / safe(dyr)
            out["ceiling"] = (g.cam_x + dxr * t, np.full_like(X, self.height), t)
            t = (-w2 - g.cam_x) / safe(dxr)
            out["left"] = (np.full_like(X, -w2), self.eye + dyr * t, t)
            t = (w2 - g.cam_x) / safe(dxr)
            out["right"] = (np.full_like(X, w2), self.eye + dyr * t, t)
            self._world = {k: tuple(np.asarray(a, F32) for a in v) for k, v in out.items()}
        return self._world

    def _wall_ab(self, wall, P_):
        x, y, z = P_
        return (x, y) if wall in ("back", "front") else (z, y)

    def sunlight(self, x, y, z):
        """Sunlight from the window reaching world points (arrays) → 0~1 (0 everywhere without a window)."""
        if not self.window:
            return np.zeros(np.shape(x), F32)
        return self._window_hit((x, y, z), self.window)

    def _window_hit(self, P_, win):
        """Sunlight reaching world points P through the window → 0~1 (soft penumbra, mullion shadows)."""
        D = self.sun_dir
        x, y, z = P_
        w2 = self.width / 2
        wall = win["wall"]
        if wall == "left":
            t = (x - (-w2)) / max(D[0], 1e-4)
        elif wall == "right":
            t = (w2 - x) / max(-D[0], 1e-4)
        else:
            t = (self.zb - z) / max(-D[2], 1e-4)
        t = np.maximum(t, 0)
        qx, qy, qz = x - D[0] * t, y - D[1] * t, z - D[2] * t
        a, b = (qz, qy) if wall in ("left", "right") else (qx, qy)
        a0, a1 = win["a"]
        b0, b1 = win["y"]
        soft = 0.015 + 0.025 * t
        m = smoothstep(a0 - soft, a0 + soft, a) * smoothstep(a1 + soft, a1 - soft, a) \
            * smoothstep(b0 - soft, b0 + soft, b) * smoothstep(b1 + soft, b1 - soft, b)
        mw = win["mullion"] / 2 + soft * 0.5
        mid_a, mid_b = (a0 + a1) / 2, b0 + (b1 - b0) * 0.62
        mull = np.maximum(smoothstep(mw, mw * 0.3, np.abs(a - mid_a)), smoothstep(mw, mw * 0.3, np.abs(b - mid_b)))
        return (m * (1 - 0.85 * mull) * (t > 0.05)).astype(F32)

    # ---- drawing
    def draw(self, wall="#e2d4bf", side=None, floor="planks", floor_colors=None, ceiling="#efe8dc",
             baseboard=0.1, baseboard_color="#f4efe7", window="left", window_rect=None, sky=("#9cc3ea", "#e8f3fb"),
             sun=1.0, sun_dir=None, light_color="#ffe2b4", ambient=0.78, occlusion=0.5, gloss=0.18, seed=0):
        """Paint the room shell.

        wall / side / ceiling: colours (``side`` defaults to ``wall``); floor: ``planks`` / ``tiles`` /
        ``checker`` / a colour / a colour map; baseboard: height in metres (0 = none); window: ``left`` /
        ``right`` / ``back`` / None, ``window_rect=((a0, a1), (y0, y1))`` (a = depth on a side wall, x on the back
        wall); sun: strength of the sunlight patch; occlusion: soft darkening in the corners; gloss: window
        reflection on the floor.
        """
        cv, g = self.cv, self.ground
        w2, Hh, zb = self.width / 2, self.height, self.zb
        zn = max(self.z_near, 0.5)
        _texture_or_color("floor texture", floor, ("planks", "tiles", "checker"))
        if window:
            if window not in ("left", "right", "back"):
                raise _unknown_name("window wall", window, ("left", "right", "back"))
            if window_rect is None:
                if window in ("left", "right"):
                    span = zb - zn
                    window_rect = ((zn + span * 0.35, zn + span * 0.82), (0.85, 2.25))
                else:
                    window_rect = ((-self.width * 0.22, self.width * 0.22), (0.9, 2.25))
            self.window = dict(wall=window, a=window_rect[0], y=window_rect[1], mullion=0.05)
            if sun_dir is None:
                sun_dir = {"left": (1.0, -0.62, 0.3), "right": (-1.0, -0.62, 0.3), "back": (0.3, -0.55, -1.0)}[window]
            D = np.asarray(sun_dir, np.float64)
            self.sun_dir = D / np.linalg.norm(D)
        Wd = self.world()
        polys = self._polys()
        X, Y = cv.xy()
        lc = rgb(light_color)
        side = side or wall
        base = dict(back=rgb(wall), left=rgb(side), right=rgb(side), ceiling=rgb(ceiling))
        orient = dict(back=1.0, left=0.93, right=0.97, ceiling=0.92, floor=1.0)
        if window == "left":
            orient.update(left=0.84, right=1.04)
        elif window == "right":
            orient.update(right=0.84, left=1.04)
        win_c = None
        if self.window:
            a0, a1 = self.window["a"]
            win_c = np.array([-w2 + 0.01, sum(self.window["y"]) / 2, (a0 + a1) / 2]) if window == "left" else \
                np.array([w2 - 0.01, sum(self.window["y"]) / 2, (a0 + a1) / 2]) if window == "right" else \
                np.array([(a0 + a1) / 2, sum(self.window["y"]) / 2, zb - 0.01])
        for name in ("ceiling", "left", "right", "back", "floor"):
            pts = polys[name]
            box = _clip_box(cv, min(p[0] for p in pts) - 2, min(p[1] for p in pts) - 2,
                            max(p[0] for p in pts) + 2, max(p[1] for p in pts) + 2)
            if box is None:
                continue
            bx0, by0, bx1, by1 = box
            Xc, Yc = X[by0:by1, bx0:bx1], Y[by0:by1, bx0:bx1]
            mk = _sdf.fill(_sdf.sd_polygon(Xc, Yc, pts))
            if not mk.any():
                continue
            x, y, z = (a[by0:by1, bx0:bx1] for a in Wd[name])
            if name == "floor":
                if isinstance(floor, str) and floor == "planks":
                    col = g.planks(colors=floor_colors or ("#a8744a", "#8f5e3a", "#b98656", "#9a6844"), seed=seed,
                                   box=box)
                elif isinstance(floor, str) and floor == "tiles":
                    col = g.tiles(colors=floor_colors or ("#d9d2c5", "#cfc6b6", "#e2dbcf"), seed=seed, box=box)
                elif isinstance(floor, str) and floor == "checker":
                    col = g.checker(0.5, floor_colors or ("#e8e2d6", "#2d2a2a"), box=box)
                else:
                    col = np.broadcast_to(_crop_color(floor, box, (cv.h, cv.w)), Xc.shape + (3,)).astype(F32)
            else:
                col = np.broadcast_to(base[name], Xc.shape + (3,)).astype(F32)
                if name != "ceiling":
                    n = fbm2d_at(x * 1.5 + z * 1.5, y * 1.5, 3, seed=seed + 3)
                    col = col * (1 + 0.015 * n)[..., None]
            # baseboards
            if baseboard and name in ("back", "left", "right"):
                bb = clamp01((baseboard - y) / 0.004 + 0.5)
                lip = np.exp(-((y - baseboard + 0.008) / 0.006) ** 2)
                bcol = rgb(baseboard_color) * (1 - 0.18 * smoothstep(baseboard, 0, y))[..., None]
                col = col * (1 - bb[..., None]) + bcol * bb[..., None]
                col = col * (1 - 0.25 * lip)[..., None]
                col = col * (1 - 0.18 * np.exp(-np.maximum(y - baseboard, 0) / 0.03) * (1 - bb))[..., None]
            # corner occlusion (world-space distance to neighbouring planes)
            if name == "floor":
                dist = np.minimum(np.minimum(x + w2, w2 - x), zb - z)
            elif name == "ceiling":
                dist = np.minimum(np.minimum(x + w2, w2 - x), zb - z)
            elif name == "back":
                dist = np.minimum(np.minimum(np.minimum(x + w2, w2 - x), y), Hh - y)
            else:
                dist = np.minimum(np.minimum(zb - z, y), Hh - y)
            occ = 1 - occlusion * np.exp(-np.maximum(dist, 0) / 0.32)
            # ambient falloff from the window
            amb = ambient * orient[name]
            if win_c is not None:
                dw = np.sqrt((x - win_c[0]) ** 2 + (y - win_c[1]) ** 2 + (z - win_c[2]) ** 2)
                amb = amb * (0.82 + 0.4 * np.exp(-dw / 2.2))
            E = (amb * occ)[..., None] * (0.85 + 0.15 * lc)
            if self.window and sun and name != self.window["wall"]:
                hit = self._window_hit((x, y, z), self.window)
                E = E + (sun * 0.85 * hit)[..., None] * lc
            col = col * E
            if self.window and name == self.window["wall"]:
                col = self._draw_window(col, (x, y, z), sky, lc)
            if name == "floor" and gloss and self.window:
                col = col + self._floor_reflection((x, y, z), sky) * gloss
            cv.paste(col, bx0, by0, mask=mk)
        return self

    def sun_beam(self, strength=0.1, color="#ffe2b4", softness=0.03):
        """Faint volumetric beam from the window to its sunlight patch (call after ``draw``). Returns the mask."""
        if not self.window:
            return np.zeros((self.cv.h, self.cv.w), F32)
        win, D, cv = self.window, self.sun_dir, self.cv
        w2 = self.width / 2
        (a0, a1), (b0, b1) = win["a"], win["y"]
        pts = []
        for a in (a0, a1):
            for b in (b0, b1):
                if win["wall"] == "left":
                    P_ = np.array([-w2, b, a])
                elif win["wall"] == "right":
                    P_ = np.array([w2, b, a])
                else:
                    P_ = np.array([a, b, self.zb])
                Q = P_ + D * (P_[1] / max(-D[1], 1e-3))
                pts += [self.project(*P_), self.project(*Q)]
        hull = _hull(pts)
        box = _clip_box(cv, min(p[0] for p in hull) - 4, min(p[1] for p in hull) - 4,
                        max(p[0] for p in hull) + 4, max(p[1] for p in hull) + 4)
        if box is None:
            return np.zeros((cv.h, cv.w), F32)
        X, Y = _grid_box(*box)
        d = _sdf.sd_polygon(X, Y, hull)
        m = _sdf.fill(d + softness * cv.h, softness * cv.h * 2)
        wx, wy = np.mean([p[0] for p in pts[0::2]]), np.mean([p[1] for p in pts[0::2]])
        fx, fy = np.mean([p[0] for p in pts[1::2]]), np.mean([p[1] for p in pts[1::2]])
        L2 = max((fx - wx) ** 2 + (fy - wy) ** 2, 1.0)
        t = np.clip(((X - wx) * (fx - wx) + (Y - wy) * (fy - wy)) / L2, 0, 1)
        m = m * (0.55 + 0.45 * (1 - t)) * strength
        cv.paste(np.broadcast_to(rgb(color), m.shape + (3,)).astype(F32), box[0], box[1], mask=m, mode="add")
        return _full(cv, m, box)

    def _draw_window(self, col, P_, sky, lc):
        win = self.window
        a, b = self._wall_ab(win["wall"], P_)
        (a0, a1), (b0, b1) = win["a"], win["y"]
        fr = 0.06
        mw = win["mullion"] / 2
        fp = 0.004
        inside = smoothstep(a0 - fp, a0 + fp, a) * smoothstep(a1 + fp, a1 - fp, a) \
            * smoothstep(b0 - fp, b0 + fp, b) * smoothstep(b1 + fp, b1 - fp, b)
        outer = smoothstep(a0 - fr - fp, a0 - fr + fp, a) * smoothstep(a1 + fr + fp, a1 + fr - fp, a) \
            * smoothstep(b0 - fr - fp, b0 - fr + fp, b) * smoothstep(b1 + fr + fp, b1 + fr - fp, b)
        tb = clamp01((b - b0) / max(b1 - b0, 1e-3))
        s0, s1 = rgb(sky[1]), rgb(sky[0])
        skyc = (s0 + (s1 - s0) * tb[..., None]) * 1.35
        hz = np.exp(-((b - (b0 + (b1 - b0) * 0.18)) / 0.12) ** 2)
        skyc = skyc + np.array([0.25, 0.22, 0.12], F32) * hz[..., None]
        ua = (a - a0) / max(a1 - a0, 1e-3)
        far = 0.22 + 0.05 * np.sin(ua * 5.1 + 1.3) + 0.03 * np.sin(ua * 13.7)
        near = 0.12 + 0.06 * np.sin(ua * 3.3 + 0.4) + 0.02 * np.sin(ua * 21.0 + 2)
        hill_far = smoothstep(far + 0.004, far - 0.004, tb)[..., None]
        hill_near = smoothstep(near + 0.004, near - 0.004, tb)[..., None]
        skyc = skyc * (1 - hill_far) + np.array([0.62, 0.74, 0.78], F32) * 1.1 * hill_far
        skyc = skyc * (1 - hill_near) + np.array([0.45, 0.62, 0.42], F32) * 1.05 * hill_near
        mid_a, mid_b = (a0 + a1) / 2, b0 + (b1 - b0) * 0.62
        mull = np.maximum(smoothstep(mw + fp, mw - fp, np.abs(a - mid_a)),
                          smoothstep(mw + fp, mw - fp, np.abs(b - mid_b)))
        frame_c = np.array([0.95, 0.93, 0.9], F32) * 0.9
        glass = skyc * (1 - mull[..., None]) + frame_c * mull[..., None]
        framing = outer * (1 - inside)
        sill = smoothstep(b0 - fr - 0.05, b0 - fr - 0.04, b) * smoothstep(b0 - fr + 0.002, b0 - fr - 0.002, b) \
            * smoothstep(a0 - fr - 0.06, a0 - fr - 0.05, a) * smoothstep(a1 + fr + 0.06, a1 + fr + 0.05, a)
        col = col * (1 - framing[..., None]) + frame_c * (0.95 + 0.05 * tb)[..., None] * framing[..., None]
        col = col * (1 - inside[..., None]) + glass * inside[..., None]
        col = col * (1 - sill[..., None]) + frame_c * 1.02 * sill[..., None]
        return col

    def _floor_reflection(self, P_, sky):
        win = self.window
        g = self.ground
        x, _, z = P_
        cam = np.array([g.cam_x, self.eye, 0.0])
        dx, dy, dz = x - cam[0], -self.eye, z - cam[2]
        ry = -dy
        w2 = self.width / 2
        if win["wall"] in ("left", "right"):
            wx = -w2 if win["wall"] == "left" else w2
            t = (wx - x) / np.where(np.abs(dx) < 1e-4, 1e-4, dx)
            a, b = z + dz * t, ry * t
        else:
            t = (self.zb - z) / np.maximum(dz, 1e-4)
            a, b = x + dx * t, ry * t
        (a0, a1), (b0, b1) = win["a"], win["y"]
        soft = 0.05 + 0.08 * np.abs(t)
        m = smoothstep(a0 - soft, a0 + soft, a) * smoothstep(a1 + soft, a1 - soft, a) \
            * smoothstep(b0 - soft, b0 + soft, b) * smoothstep(b1 + soft, b1 - soft, b) * (t > 0)
        return (rgb(sky[1]) * m[..., None] * 0.9).astype(F32)

    def picture(self, wall="back", center=(0.0, 1.55), size=(1.0, 0.7), image=None, frame="#2e241c", frame_width=0.045,
                mat="#f3eee4", mat_width=0.06, seed=0, shadow=0.35):
        """A framed picture hung on a wall, in perspective. ``center=(a, y)`` (a = x on the back wall, depth on a
        side wall), ``size`` in metres; ``image``: Canvas / (h, w, 3) array, or None for a painted landscape print.
        Returns the full-frame mask."""
        cv = self.cv
        ca, cb = center
        hw, hh = size[0] / 2, size[1] / 2
        m_ = 0.08
        corners = [self.wall_point(wall, ca + sa * (hw + m_), cb + sb * (hh + m_)) for sa in (-1, 1) for sb in (-1, 1)]
        box = _clip_box(cv, min(p[0] for p in corners) - 2, min(p[1] for p in corners) - 2,
                        max(p[0] for p in corners) + 2, max(p[1] for p in corners) + 2)
        if box is None:
            return np.zeros((cv.h, cv.w), F32)
        bx0, by0, bx1, by1 = box
        x, y, z = (q[by0:by1, bx0:bx1] for q in self.world()[wall])
        a, b = (x, y) if wall == "back" else (z, y)
        Xc, Yc = _grid_box(*box)
        wall_m = _sdf.fill(_sdf.sd_polygon(Xc, Yc, self._polys()[wall]))
        if wall == "right":
            ua = ((ca + hw) - a) / (2 * hw)
        else:
            ua = (a - (ca - hw)) / (2 * hw)
        vb = ((cb + hh) - b) / (2 * hh)
        fp = 0.003

        def rect(da, db, pad):
            return smoothstep(-pad - fp, -pad + fp, da) * smoothstep(-pad - fp, -pad + fp, db)

        def put(col, m, mode="normal"):
            col = np.broadcast_to(as_color(col), m.shape + (3,)).astype(F32)
            cv.paste(col, bx0, by0, mask=m, mode=mode)
        du = np.minimum(ua, 1 - ua) * 2 * hw
        dv = np.minimum(vb, 1 - vb) * 2 * hh
        outer = rect(du, dv, 0) * wall_m
        if shadow:
            sh = smoothstep(-0.05, 0.03, np.minimum(du + 0.01, dv - 0.03)) * wall_m * (1 - outer) * (vb > 0.3)
            put(rgb("#2a2018"), sh * shadow, "multiply")
        inner = rect(du - frame_width, dv - frame_width, 0) * outer
        art = rect(du - frame_width - mat_width, dv - frame_width - mat_width, 0) * outer
        put(rgb(frame) * (1 + 0.25 * smoothstep(frame_width, 0, np.minimum(du, dv)))[..., None], outer)
        put(rgb(mat), inner)
        ia = mat_width + frame_width
        u_ = clamp01((ua * 2 * hw - ia) / max(2 * hw - 2 * ia, 1e-3))
        v_ = clamp01((vb * 2 * hh - ia) / max(2 * hh - 2 * ia, 1e-3))
        if image is None:
            img = _landscape_print(u_, v_, seed)
        else:
            src = image.buf if hasattr(image, "buf") else np.asarray(image, F32)
            img = _bilinear(src, u_ * (src.shape[1] - 1), v_ * (src.shape[0] - 1))
        put(img, art)
        put("#000000", smoothstep(0.012, 0.0, np.abs(np.minimum(du, dv) - ia)) * inner * 0.25)
        return _full(cv, outer, box)


def _landscape_print(u, v, seed=0):
    sky = np.stack([0.95 - 0.25 * v, 0.78 - 0.1 * v, 0.6 + 0.1 * v], -1)
    sun = np.exp(-(((u - 0.68) / 0.09) ** 2 + ((v - 0.38) / 0.09 / 1.3) ** 2) * 2)
    col = sky + np.array([0.3, 0.2, 0.05]) * sun[..., None]
    h1 = 0.55 + 0.07 * np.sin(u * 7 + seed) + 0.03 * np.sin(u * 19 + 2 * seed)
    h2 = 0.7 + 0.06 * np.sin(u * 5 + 1.7 + seed)
    col = np.where((v > h1)[..., None], np.array([0.42, 0.45, 0.55]), col)
    col = np.where((v > h2)[..., None], np.array([0.22, 0.3, 0.3]), col)
    return col.astype(F32)


# ===================================================================== arrangement


def _order(pts):
    return sorted(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))


def row(n, start=(0.3, 0.6), end=(0.7, 0.6), jitter=0.0, seed=0):
    """n evenly spaced points from start to end (fractions). Returns ``(points, order)``; draw in ``order``
    (back to front)."""
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n):
        t = i / max(n - 1, 1)
        u = start[0] + (end[0] - start[0]) * t + jitter * rng.uniform(-1, 1) * 0.02
        v = start[1] + (end[1] - start[1]) * t + jitter * rng.uniform(-1, 1) * 0.01
        pts.append((u, v))
    return pts, _order(pts)


def grid_positions(rows, cols, region=(0.2, 0.4, 0.8, 0.8), jitter=0.0, seed=0):
    """rows × cols points filling ``region=(u0, v0, u1, v1)``. Returns ``(points, order)``."""
    rng = np.random.default_rng(seed)
    u0, v0, u1, v1 = region
    pts = []
    for r in range(rows):
        for c in range(cols):
            u = u0 + (u1 - u0) * ((c + 0.5) / cols) + jitter * rng.uniform(-0.5, 0.5) * (u1 - u0) / cols
            v = v0 + (v1 - v0) * ((r + 0.5) / rows) + jitter * rng.uniform(-0.5, 0.5) * (v1 - v0) / rows
            pts.append((u, v))
    return pts, _order(pts)


def scatter_positions(n, region=(0.1, 0.5, 0.9, 0.9), min_gap=0.08, seed=0, aspect=16 / 9, tries=4000):
    """Up to n random points in ``region`` at least ``min_gap`` apart (fraction of the canvas height; no overlaps).
    Returns ``(points, order)``."""
    rng = np.random.default_rng(seed)
    u0, v0, u1, v1 = region
    pts = []
    for _ in range(tries):
        if len(pts) >= n:
            break
        u, v = rng.uniform(u0, u1), rng.uniform(v0, v1)
        if all(math.hypot((u - a) * aspect, v - b) >= min_gap for a, b in pts):
            pts.append((float(u), float(v)))
    return pts, _order(pts)


def ring_positions(n, center=(0.5, 0.6), radius=0.15, tilt=0.35, start=90.0, aspect=16 / 9):
    """n points on a circle lying on the ground (an ellipse squashed by ``tilt``); radius as a fraction of the
    height; start = angle of the first point (90 = front). Returns ``(points, order)``."""
    pts = []
    for i in range(n):
        a = math.radians(start + 360.0 * i / n)
        pts.append((center[0] + radius * math.cos(a) / aspect, center[1] + radius * tilt * math.sin(a)))
    return pts, _order(pts)


# ===================================================================== checked options

def _strict_kw(fn, owner, extra=()):
    """Like ``_args.strict`` for a function whose ``**kw`` is forwarded: the allowed names are the signature plus
    ``extra`` (the callee's options). Unknown names raise a TypeError naming the closest option."""
    import functools
    names = _sig_options(fn)
    allowed = names + [e for e in extra if e not in names]
    known = frozenset(allowed)

    @functools.wraps(fn)
    def wrapper(*a, **k):
        if k and not known.issuperset(k):
            k = _reject_unknown(owner, k, allowed)
        return fn(*a, **k)
    return wrapper


def _kw_free(fn, skip=("cv", "self")):
    return [n for n in _sig_options(fn) if n not in skip]


_MAT = list(_MAT_DEFAULT)
for _n in ("shade", "paint_form", "capsule", "lathe", "ball", "torus", "disc"):
    globals()[_n] = _strict_kw(globals()[_n], _n, _MAT)
for _n in ("cylinder", "cone"):
    globals()[_n] = _strict_kw(globals()[_n], _n, _kw_free(lathe) + _MAT)
for _n in ("shelf", "counter"):
    globals()[_n] = _strict_kw(globals()[_n], _n, _kw_free(tabletop))
set_light = _strict_kw(set_light, "set_light", _sig_options(Light.__init__))
Frame.shade = _strict_kw(Frame.shade, "Frame.shade", _kw_free(shade) + _MAT)
Frame.form = _strict_kw(Frame.form, "Frame.form", _kw_free(_paint_form_box, ("cv", "dpx", "box", "base")) + _MAT)
Solid.recolor = _strict_kw(Solid.recolor, "Solid.recolor", _MAT)
Solid.band = _strict_kw(Solid.band, "Solid.band", _MAT)
Room.box = _strict_kw(Room.box, "Room.box", _kw_free(Ground.box))
Room.rug = _strict_kw(Room.rug, "Room.rug", _kw_free(Ground.rug))
_HELPERS = {"check_frac", "guard_fractions", "resolve_light", "clip_box", "full_mask", "full_frame", "bbox_of", "multiply_shadow", "multiply",
            "shadow_color", "contact_px", "grid_box", "sd_ellipse"}      # building blocks keep their private aliases
for _n in __all__:
    _f_ = globals().get(_n)
    if isinstance(_f_, type(_kw_free)) and _n not in _HELPERS:
        globals()[_n] = _strict(_f_, _n)
for _c in (Frame, Light, Solid, Ground, Surface, Room):
    _strict_methods(_c)
del _n, _c, _f_

from . import manifest as _manifest  # noqa: E402

def _alias_arg(label):
    """alias= for a form call: the request's own alias words plus the call name ("ball" for a lemon)."""
    def f(a):
        al = a.get("alias")
        al = [al] if isinstance(al, str) else list(al or [])
        nm = a.get("name")
        return al + [label] if isinstance(nm, str) and nm.strip() and label not in al else (al or None)
    return f


_FORM_OBJECTS = ["ball", "capsule", "lathe", "cylinder", "cone", "torus", "cuboid", "tabletop", "shelf", "counter",
                 "disc", "book"]
_manifest.register(globals(), "object", _FORM_OBJECTS, arg="name", alias={n: _alias_arg(n) for n in _FORM_OBJECTS})
_manifest.wrap_method(Room, "draw", "background", name="room", cv_of=lambda a: a["self"].cv)
_manifest.wrap_method(Room, "rug", "object", name="rug")
_manifest.wrap_method(Room, "panel", "object", name="panel")
