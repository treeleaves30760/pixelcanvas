# -*- coding: utf-8 -*-
"""solid: a 3D renderer in numpy (signed-distance-field ray marching).

Describe objects in 3D and get perspective, shading, soft shadows, ambient occlusion, reflections, glass and depth
for free. Every named object is painted as its own layer, and the render returns depth / normal / id buffers.

>>> from pixelcanvas import Canvas
>>> from pixelcanvas.solid import Scene, Camera, sphere, box, car
>>> sc = Scene(lighting="studio", ground="checker")
>>> sc.add(car(color="#e8452c").rotate_y(-30), "car")
>>> sc.add(sphere(0.6).place(x=2.6), "ball", "chrome")
>>> sc.camera = Camera.orbit(target=(0.5, 0.7, 0), distance=9, azimuth=35, elevation=18)
>>> cv = Canvas(960, 540)
>>> res = sc.render(cv)                     # paints into cv, one layer per object; res.depth / res.mask("car")

World units are free (metres work well); y is up, the ground is y = 0, and the default camera looks from +z.
Shapes are built centred on the origin and moved with chainable transforms:
``box(1, 0.5, 2).rotate_y(30).at(1, 0.25, 0)`` or ``cylinder(0.3, 1).place(x=2, z=-1)`` (stand on the floor).
"""
from __future__ import annotations

import math

import numpy as np

from .color import rgb, F32
from . import manifest as _manifest
from ._args import (reject_unknown as _reject_unknown, unknown_name as _unknown_name,
                    check_choice as _check_choice, signature_options as _sig_options, strict as _strict,
                    strict_methods as _strict_methods)

__all__ = [
    # primitives
    "Shape", "sphere", "ellipsoid", "box", "rounded_box", "cylinder", "cone", "capsule", "torus", "plane",
    "prism", "wedge", "lathe", "extrude", "extrude_text", "union", "smooth_union", "subtract", "intersect",
    "smooth_subtract",
    # materials / patterns
    "Material", "material", "PRESETS", "wood", "marble", "checker", "stripes", "weave", "planks", "stone", "moss",
    "noise3", "fbm3",
    # scene
    "Model", "Part", "model", "Camera", "Lighting", "LIGHTING", "Scene", "RenderResult",
    # prefabs
    "mug", "cup", "bottle", "vase", "table", "chair", "sofa", "bed", "desk_lamp", "pendant_lamp", "bookshelf",
    "house", "tree", "car", "truck", "rocket", "robot", "character", "quadruped", "laptop", "potted_plant", "rug",
    "toy_block", "window_frame", "book", "wall", "room", "counter", "coffee_machine", "speaker",
    "armchair", "floor_lamp", "side_table", "earbuds", "charging_case", "smartphone", "smartwatch", "tent",
    "stone_lantern", "arched_bridge", "rock",
    # registry
    "PREFABS", "WINDOW_VIEWS", "names", "find", "describe", "NameNotFound",
]

_BIG = 1e9
_REF = False                     # reference mode: every part and child evaluated in full (see _reference)


class _reference:
    """``with solid._reference(): sc.render(cv)`` renders by brute force: no distance caps, no culling of
    union / subtract children, no part tree, plain soft-shadow marching over every part. The accelerated
    renderer must agree with it; the tests compare the two."""

    def __enter__(self):
        global _REF
        self._old, _REF = _REF, True
        return self

    def __exit__(self, *exc):
        global _REF
        _REF = self._old
        return False


class NameNotFound(KeyError, ValueError):
    """An unknown part / object / prefab name, with the closest valid names in the message."""

    def __str__(self):
        return str(self.args[0]) if self.args else ""


def _bad_name(kind, name, valid):
    return NameNotFound(str(_unknown_name(kind, name, list(dict.fromkeys(valid)))))


def _f(a):
    return np.asarray(a, F32)


def _vec(v):
    return np.asarray(v, np.float64).reshape(3)


def _norm(v):
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


# ===================================================================================================== 2D helpers

def _poly2d(x, y, P, skip_axis=False):
    """Signed distance to a closed polygon P [(x, y), ...] for 1D float32 arrays x, y."""
    P = np.asarray(P, np.float64)
    n = len(P)
    d = np.full(x.shape, np.inf, F32)
    inside = np.zeros(x.shape, bool)
    for i in range(n):
        ax, ay = P[i]
        bx, by = P[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        wx, wy = x - F32(ax), y - F32(ay)
        if not (skip_axis and abs(ax) < 1e-9 and abs(bx) < 1e-9):
            ee = ex * ex + ey * ey
            if ee > 1e-18:
                h = np.clip((wx * F32(ex) + wy * F32(ey)) * F32(1.0 / ee), 0, 1)
                qx, qy = wx - F32(ex) * h, wy - F32(ey) * h
            else:
                qx, qy = wx, wy
            np.minimum(d, qx * qx + qy * qy, out=d)
        if ay != by:
            cond = (y >= F32(min(ay, by))) & (y < F32(max(ay, by)))
            xint = F32(ax) + (y - F32(ay)) * F32(ex / ey)
            inside ^= cond & (x < xint)
    d = np.sqrt(d)
    return np.where(inside, -d, d)


def _catmull(points, samples=8, closed=False):
    """Resample a polyline through ``points`` with a Catmull-Rom spline (smooth curves from a few control points)."""
    P = np.asarray(points, np.float64)
    if len(P) < 3:
        return P
    ext = np.vstack([P[-1:], P, P[:2]]) if closed else np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    nseg = len(P) if closed else len(P) - 1
    for i in range(nseg):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-1])
    return np.array(out)


# ===================================================================================================== shapes

def _bdist(lo, hi, x, y, z):
    """Distance from points to an axis-aligned box (0 inside): a lower bound for anything inside the box."""
    dx = np.maximum(np.maximum(F32(lo[0]) - x, x - F32(hi[0])), 0)
    dy = np.maximum(np.maximum(F32(lo[1]) - y, y - F32(hi[1])), 0)
    dz = np.maximum(np.maximum(F32(lo[2]) - z, z - F32(hi[2])), 0)
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def _lowb(shape):
    """(lam, mu) such that ``shape._d >= lam * box distance - mu`` outside its bounding box.

    Box distance is a lower bound for exact distance fields only. Ellipsoids, scaled shapes and twisted shapes
    report smaller distances than that (soft shadows and AO read those values), so culling compares
    ``lam * lb - mu`` with the cap. The factors are measured once per shape on points around its box."""
    v = getattr(shape, "_lbmc", None)
    if v is not None:
        return v
    lo, hi = shape.bounds()
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or np.any(np.abs(lo) > 1e8) \
            or np.any(np.abs(hi) > 1e8):
        v = (1.0, 0.0)
    else:
        c, h = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-9)
        hd = float(np.linalg.norm(h))
        rng = np.random.default_rng(7)
        n = 320
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        far = c + dirs * (hd + 2 * hd * np.geomspace(0.01, 4.0, n))[:, None]
        near = c + (rng.random((n, 3)) * 2 - 1) * h * 1.25
        p = np.concatenate([far, near]).astype(F32)
        x, y, z = p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy()
        lb = _bdist(lo, hi, x, y, z)
        d = np.asarray(shape._d(x.copy(), y.copy(), z.copy()), np.float64)
        ok = lb > 0.02 * hd
        r = float(np.min(d[ok] / lb[ok])) if ok.any() else 0.0
        lam = 1.0 if r > 0.995 else float(np.clip(r * 0.9, 0.02, 1.0))
        out = lb > 0
        mu = float(max(0.0, np.max(lam * lb[out] - d[out])) * 1.25) if out.any() else 0.0
        v = (lam, mu)
    shape._lbmc = v
    return v


def _corners(lo, hi):
    return np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])


def _rotm(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class _Xf:
    """Chainable transforms shared by Shape and Model (subclasses implement ``_affine(F, g)``)."""

    def at(self, x=0.0, y=0.0, z=0.0):
        """Move by (x, y, z). Shapes are built around the origin, so this is also 'put it at (x, y, z)'."""
        if isinstance(x, (tuple, list, np.ndarray)):
            x, y, z = x
        return self._affine(np.eye(3), np.array([x, y, z], np.float64))

    translate = move = at

    def rotate_x(self, deg):
        """Rotate about the x axis (degrees; positive tips +y toward +z)."""
        return self._affine(_rotm("x", deg), np.zeros(3))

    def rotate_y(self, deg):
        """Rotate about the vertical axis (degrees; positive turns +x toward -z, i.e. counter-clockwise from above)."""
        return self._affine(_rotm("y", deg), np.zeros(3))

    def rotate_z(self, deg):
        """Rotate about the z axis (degrees; positive turns +x toward +y)."""
        return self._affine(_rotm("z", deg), np.zeros(3))

    def rotate(self, x=0.0, y=0.0, z=0.0):
        """Rotate about x, then y, then z (degrees)."""
        F = _rotm("z", z) @ _rotm("y", y) @ _rotm("x", x)
        return self._affine(F, np.zeros(3))

    def scale(self, s):
        """Scale uniformly (``s``) or per axis (``(sx, sy, sz)``) about the origin."""
        s = np.broadcast_to(np.asarray(s, np.float64), (3,))
        return self._affine(np.diag(s), np.zeros(3))

    def place(self, x=0.0, z=0.0, y=0.0):
        """Stand the object on the floor: move it so its bounding-box bottom is at height ``y`` and its
        footprint is centred on (x, z). Hanging prefabs (``pendant_lamp``) keep their ceiling height and
        move up by ``y``."""
        lo, hi = self.bounds()
        dy = y if getattr(self, "hangs", False) else y - lo[1]
        return self.at(x - (lo[0] + hi[0]) / 2, dy, z - (lo[2] + hi[2]) / 2)

    def on(self, target, part=None, at=(0.5, 0.5), level=None, xz=None):
        """Rest this object on top of ``target`` (a placed Model or Shape) or on one of its parts, touching the
        surface without sinking into it: ``cat.on(armchair, "armrest_r")``, ``cup.on(table, "top", at=(0.3, 0.6))``,
        ``vase.on(shelf, "shelf", level=2)``.

        ``part``: a part name (``seat``, ``armrest_l``, ``armrest_r``, ``top``, ``deck``, ``shelf`` ...; None = the
        whole target). ``at``: where across that surface, as fractions (0~1) of its width along x and depth along z
        (0.5, 0.5 = the middle); ``xz``: a world (x, z) point instead. ``level``: which surface down from the top
        (0 = the top; ``shelf`` starts at 1, the highest shelf under the top board). Turn the object first
        (``.face()`` / ``.rotate_y()``), then call ``on``."""
        surf, level = _surface_of(target, part, level)
        lo, hi = surf.bounds()
        if xz is None:
            u, v = at
            px, pz = lo[0] + (hi[0] - lo[0]) * float(u), lo[2] + (hi[2] - lo[2]) * float(v)
        else:
            px, pz = float(xz[0]), float(xz[1])
        mlo, mhi = self.bounds()
        moved = self.at(px - (mlo[0] + mhi[0]) / 2, 0.0, pz - (mlo[2] + mhi[2]) / 2)
        top = float(hi[1]) + 1e-3
        y0 = _surface_y(surf, np.array([px]), np.array([pz]), top, float(lo[1]) - 1e-3, level)[0]
        if not np.isfinite(y0):
            raise ValueError(f"on(): nothing of {getattr(target, 'name', 'the target')!r}"
                             f"{' part ' + repr(part) if part else ''} lies under ({px:.3f}, {pz:.3f}); "
                             "pick another at= / xz= inside its footprint")
        mlo, mhi = moved.bounds()
        moved = moved.at(0.0, y0 - mlo[1], 0.0)
        # contact: the lowest point of the object over each column against the surface under it
        own = _shape_of(moved)
        mlo, mhi = moved.bounds()
        g = np.linspace(0.02, 0.98, 21)
        gx, gz = np.meshgrid(mlo[0] + (mhi[0] - mlo[0]) * g, mlo[2] + (mhi[2] - mlo[2]) * g)
        gx, gz = gx.ravel(), gz.ravel()
        bottom = _surface_y(own, gx, gz, float(mlo[1]) - 1e-3, float(mhi[1]) + 1e-3, 0, up=True)
        under = _surface_y(surf, gx, gz, top, float(lo[1]) - 1e-3, level)
        ok = np.isfinite(bottom) & np.isfinite(under)
        dy = float(np.max(under[ok] - bottom[ok])) if ok.any() else 0.0
        return moved.at(0.0, dy, 0.0)


class Shape(_Xf):
    """A signed distance field in 3D (negative inside). Build with the primitive functions, combine with
    ``|`` (union), ``-`` (subtract), ``&`` (intersect), ``a.blend(b, k)`` (smooth union)."""

    cost = 1

    def _d(self, x, y, z):                       # pragma: no cover - abstract
        raise NotImplementedError

    def _dc(self, x, y, z, cap):
        """Distance with a cap: exact where it is below ``cap``, otherwise any value between ``cap`` and the
        distance. Groups use the cap to skip children that cannot come closer (shadow, AO and hit queries)."""
        return self._d(x, y, z)

    def _bb(self):                               # pragma: no cover - abstract
        raise NotImplementedError

    def bounds(self):
        """Axis-aligned bounding box ``(lo, hi)`` as two (3,) arrays."""
        if not hasattr(self, "_bbc"):
            lo, hi = self._bb()
            self._bbc = (np.asarray(lo, np.float64), np.asarray(hi, np.float64))
        return self._bbc

    @property
    def size(self):
        lo, hi = self.bounds()
        return hi - lo

    def __call__(self, p):
        """Distance at points ``p`` of shape (N, 3) (or one point)."""
        p = _f(p)
        one = p.ndim == 1
        p = p.reshape(-1, 3)
        d = self._d(p[:, 0].copy(), p[:, 1].copy(), p[:, 2].copy())
        return float(d[0]) if one else d

    # -- transforms
    def _affine(self, F, g):
        return _Affine(self, F, g)

    # -- operators
    def __or__(self, other):
        return union(self, other)

    def __sub__(self, other):
        return subtract(self, other)

    def __and__(self, other):
        return intersect(self, other)

    def blend(self, other, k=0.1):
        """Smooth union with a fillet of size ``k`` (organic joins: clay, animals, fruit)."""
        return smooth_union(self, other, k=k)

    def carve(self, other, k=0.05):
        """Smooth subtraction of ``other`` with a rounded edge of size ``k``."""
        return smooth_subtract(self, other, k=k)

    def round(self, r):
        """Inflate by ``r`` (rounds every edge; the shape grows by r)."""
        return _Round(self, r)

    def shell(self, thickness):
        """Hollow the shape: keep a skin ``thickness`` thick around the surface."""
        return _Shell(self, thickness)

    def mirror(self, axis="x"):
        """Mirror across the plane through the origin (``'x'`` copies the +x half to -x; ``'xz'`` both)."""
        return _Mirror(self, axis)

    def repeat(self, spacing, count):
        """Finite array of copies: ``spacing`` = (sx, sy, sz) step, ``count`` = (nx, ny, nz), centred on the origin."""
        return _Repeat(self, spacing, count)

    def twist(self, deg_per_unit):
        """Twist around the y axis by ``deg_per_unit`` degrees per unit of height."""
        return _Twist(self, deg_per_unit)

    def bend(self, deg_per_unit):
        """Bend in the xy plane: the shape curls up by ``deg_per_unit`` degrees per unit along x."""
        return _Bend(self, deg_per_unit)

    def cut(self, normal=(0, 1, 0), offset=0.0):
        """Half-space cut: remove everything on the side ``normal`` points to, beyond ``dot(p, n) = offset``."""
        return intersect(self, plane(normal, offset))


# -------------------------------------------------------------------------------------------- transform node

class _Affine(Shape):
    def __init__(self, child, F, g):
        if isinstance(child, _Affine):               # collapse chains into one matrix
            F, g = F @ child.F, F @ child.g + g
            child = child.child
        self.child, self.F, self.g = child, np.asarray(F, np.float64), np.asarray(g, np.float64)
        self.Minv = np.linalg.inv(self.F)
        self.c = -self.Minv @ self.g
        sv = np.linalg.svd(self.F, compute_uv=False)
        self.k = float(sv.min())
        self.pure = np.allclose(self.F, np.eye(3))
        self.cost = child.cost

    def _local(self, x, y, z):
        if self.pure:
            return x - F32(self.g[0]), y - F32(self.g[1]), z - F32(self.g[2])
        if len(x) <= 40000:                               # one small matrix product: fewer numpy calls
            m32 = getattr(self, "_m32", None)
            if m32 is None:
                m32 = self._m32 = (self.Minv.astype(F32), self.c.astype(F32)[:, None])
            Q = m32[0] @ np.stack([x, y, z])
            Q += m32[1]
            return Q[0], Q[1], Q[2]
        M, c = self.Minv, self.c
        out = []
        for i in range(3):
            v = F32(c[i])
            acc = None
            for j, a in enumerate((x, y, z)):
                m = M[i, j]
                if abs(m) > 1e-12:
                    term = a * F32(m) if abs(m - 1.0) > 1e-12 else a
                    acc = term if acc is None else acc + term
            acc = np.zeros_like(x) if acc is None else acc
            out.append(acc + v if v != 0 else acc)
        return out

    def _d(self, x, y, z):
        lx, ly, lz = self._local(x, y, z)
        d = self.child._d(lx, ly, lz)
        return d if abs(self.k - 1.0) < 1e-9 else d * F32(self.k)

    def _dc(self, x, y, z, cap):
        if _REF or type(self.child)._dc is Shape._dc:
            return self._d(x, y, z)
        lx, ly, lz = self._local(x, y, z)
        if abs(self.k - 1.0) < 1e-9:
            return self.child._dc(lx, ly, lz, cap)
        return self.child._dc(lx, ly, lz, cap * F32(1.0 / self.k)) * F32(self.k)

    def _bb(self):
        lo, hi = self.child.bounds()
        if np.any(np.abs(lo) > 1e8) or np.any(np.abs(hi) > 1e8):
            return np.full(3, -_BIG), np.full(3, _BIG)
        C = _corners(lo, hi) @ self.F.T + self.g
        return C.min(0), C.max(0)


# -------------------------------------------------------------------------------------------- primitives

class _Prim(Shape):
    def __init__(self, fn, lo, hi):
        self.fn, self.lo, self.hi = fn, lo, hi

    def _d(self, x, y, z):
        return self.fn(x, y, z)

    def _bb(self):
        return np.asarray(self.lo, np.float64), np.asarray(self.hi, np.float64)


def sphere(r=0.5):
    """Sphere of radius ``r`` centred on the origin."""
    r = F32(r)
    return _Prim(lambda x, y, z: np.sqrt(x * x + y * y + z * z) - r, [-r] * 3, [r] * 3)


def ellipsoid(rx=0.6, ry=0.4, rz=0.5):
    """Ellipsoid with radii (rx, ry, rz)."""
    a, b, c = F32(rx), F32(ry), F32(rz)

    def fn(x, y, z):
        k0 = np.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2)
        k1 = np.sqrt((x / (a * a)) ** 2 + (y / (b * b)) ** 2 + (z / (c * c)) ** 2)
        return k0 * (k0 - 1) / np.maximum(k1, F32(1e-9))
    return _Prim(fn, [-rx, -ry, -rz], [rx, ry, rz])


def rounded_box(w=1.0, h=1.0, d=1.0, r=0.1):
    """Box of full size w x h x d (x, y, z) with edges rounded by ``r``."""
    r = float(min(r, w / 2, h / 2, d / 2))
    bx, by, bz = F32(w / 2 - r), F32(h / 2 - r), F32(d / 2 - r)
    rr = F32(r)

    def fn(x, y, z):
        qx, qy, qz = np.abs(x) - bx, np.abs(y) - by, np.abs(z) - bz
        mx, my, mz = np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0)
        out = np.sqrt(mx * mx + my * my + mz * mz)
        return out + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - rr
    return _Prim(fn, [-w / 2, -h / 2, -d / 2], [w / 2, h / 2, d / 2])


def box(w=1.0, h=1.0, d=1.0, r=0.0):
    """Box of full size w x h x d along x, y, z (``r`` rounds the edges)."""
    return rounded_box(w, h, d, r)


def cylinder(r=0.5, h=1.0, round=0.0):
    """Upright cylinder (axis = y) of radius ``r`` and full height ``h``; ``round`` softens the rims."""
    rd = float(min(round, r, h / 2))
    R, Hh, rr = F32(r - rd), F32(h / 2 - rd), F32(rd)

    def fn(x, y, z):
        dx = np.sqrt(x * x + z * z) - R
        dy = np.abs(y) - Hh
        mx, my = np.maximum(dx, 0), np.maximum(dy, 0)
        return np.minimum(np.maximum(dx, dy), 0) + np.sqrt(mx * mx + my * my) - rr
    return _Prim(fn, [-r, -h / 2, -r], [r, h / 2, r])


def cone(r1=0.5, r2=0.0, h=1.0):
    """Capped cone along y: bottom radius ``r1`` at y = -h/2, top radius ``r2`` at y = +h/2."""
    hh = h / 2.0
    k2x, k2y = r2 - r1, 2 * hh
    k2l = k2x * k2x + k2y * k2y

    def fn(x, y, z):
        qx = np.sqrt(x * x + z * z)
        rsel = np.where(y < 0, F32(r1), F32(r2))
        cax = qx - np.minimum(qx, rsel)
        cay = np.abs(y) - F32(hh)
        t = np.clip(((F32(r2) - qx) * F32(k2x) + (F32(hh) - y) * F32(k2y)) / F32(k2l), 0, 1)
        cbx = qx - F32(r2) + F32(k2x) * t
        cby = y - F32(hh) + F32(k2y) * t
        s = np.where((cbx < 0) & (cay < 0), F32(-1), F32(1))
        return s * np.sqrt(np.minimum(cax * cax + cay * cay, cbx * cbx + cby * cby))
    rm = max(r1, r2)
    return _Prim(fn, [-rm, -hh, -rm], [rm, hh, rm])


def capsule(a=(0, 0, 0), b=(0, 1, 0), r=0.2):
    """Capsule (rounded rod) from point ``a`` to point ``b`` with radius ``r``."""
    A, B = _vec(a), _vec(b)
    e = B - A
    ee = max(float(e @ e), 1e-12)

    def fn(x, y, z):
        wx, wy, wz = x - F32(A[0]), y - F32(A[1]), z - F32(A[2])
        h = np.clip((wx * F32(e[0]) + wy * F32(e[1]) + wz * F32(e[2])) * F32(1 / ee), 0, 1)
        qx, qy, qz = wx - F32(e[0]) * h, wy - F32(e[1]) * h, wz - F32(e[2]) * h
        return np.sqrt(qx * qx + qy * qy + qz * qz) - F32(r)
    return _Prim(fn, np.minimum(A, B) - r, np.maximum(A, B) + r)


def torus(R=0.5, r=0.15):
    """Torus lying flat (ring in the xz plane): ring radius ``R``, tube radius ``r``."""
    def fn(x, y, z):
        q = np.sqrt(x * x + z * z) - F32(R)
        return np.sqrt(q * q + y * y) - F32(r)
    return _Prim(fn, [-R - r, -r, -R - r], [R + r, r, R + r])


def plane(normal=(0, 1, 0), offset=0.0):
    """Half-space below the plane ``dot(p, n) = offset`` (use for cuts; the floor is ``Scene(ground=...)``)."""
    n = _norm(normal)

    def fn(x, y, z):
        return x * F32(n[0]) + y * F32(n[1]) + z * F32(n[2]) - F32(offset)
    return _Prim(fn, [-_BIG] * 3, [_BIG] * 3)


def _extrude_fn(d2, depth):
    hd = F32(depth / 2)

    def fn(x, y, z):
        a = d2(x, y)
        b = np.abs(z) - hd
        ma, mb = np.maximum(a, 0), np.maximum(b, 0)
        return np.minimum(np.maximum(a, b), 0) + np.sqrt(ma * ma + mb * mb)
    return fn


def prism(points, depth=0.2, smooth=False):
    """Extruded 2D polygon: ``points`` [(x, y), ...] in the xy plane (y up), extruded ``depth`` along z.
    ``smooth=True`` turns the points into a closed spline (smooth, facet-free sides)."""
    if not smooth:
        P = np.asarray(points, np.float64)
        lo, hi = P.min(0), P.max(0)
        sh = _Prim(_extrude_fn(lambda x, y: _poly2d(x, y, P), depth), [lo[0], lo[1], -depth / 2],
                   [hi[0], hi[1], depth / 2])
        sh.cost = 2 + len(P) // 4
        return sh
    P0 = np.asarray(points, np.float64)
    P = _catmull(P0, 6, closed=True)
    Pf = _catmull(P0, 32, closed=True)
    lo, hi = P.min(0), P.max(0)
    ext = float(max(hi - lo))
    cell = ext / 400.0
    key = ("prism", P0.tobytes())
    coarse = (lambda x, y: _poly2d(x, y, P))
    far = ext * 0.6
    outer = _grid2d(coarse, lo[0] - far, hi[0] + far, lo[1] - far, hi[1] + far, cell * 8, key + ("outer",),
                    coarse, False)
    pad = max(cell * 4, ext * 0.03)
    g = _grid2d(lambda x, y: _poly2d(x, y, Pf), lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, cell, key,
                outer, _LATHE_CUBIC)
    sh = _Prim(_extrude_fn(g, depth), [lo[0], lo[1], -depth / 2], [hi[0], hi[1], depth / 2])
    sh.cost = 2
    return sh


def wedge(w=1.0, h=0.5, d=1.0):
    """Ramp / doorstop: full size w x h x d, high side at -z, sloping down to +z."""
    return prism([(-d / 2, -h / 2), (d / 2, -h / 2), (-d / 2, h / 2)], w).rotate_y(-90)


def _polyline2d(x, y, P):
    """Unsigned distance to an open polyline P [(x, y), ...] for 1D float32 arrays x, y."""
    d = np.full(x.shape, np.inf, F32)
    for i in range(len(P) - 1):
        ax, ay = P[i]
        ex, ey = P[i + 1] - P[i]
        ee = max(ex * ex + ey * ey, 1e-18)
        wx, wy = x - F32(ax), y - F32(ay)
        h = np.clip((wx * F32(ex) + wy * F32(ey)) * F32(1 / ee), 0, 1)
        qx, qy = wx - F32(ex) * h, wy - F32(ey) * h
        np.minimum(d, qx * qx + qy * qy, out=d)
    return np.sqrt(d)


_GRID_CACHE = {}
_LATHE_CUBIC = False                              # bilinear is smooth enough at 400 cells and twice as fast


def _grid2d(exact, a0, a1, b0, b1, cell, key=None, fallback=None, cubic=True):
    """Sample a 2D distance function ``exact(a, b)`` on a grid once; return a fast interpolating evaluator.

    The caller pads the grid so the surface lies a few cells inside it; points off the grid use ``fallback``
    (a cheaper stand-in for ``exact``). ``cubic`` = Catmull-Rom interpolation (C1: smooth normals) instead of
    bilinear. Smooth curves sampled this way shade without the facets of a polyline, and each lookup costs the
    same few operations however detailed the curve is."""
    nx = int(math.ceil((a1 - a0) / cell)) + 2
    ny = int(math.ceil((b1 - b0) / cell)) + 2
    g = _GRID_CACHE.get(key) if key is not None else None
    if g is None:
        A = (a0 + np.arange(nx) * cell).astype(F32)
        B = (b0 + np.arange(ny) * cell).astype(F32)
        AA, BB = np.meshgrid(A, B)
        g = np.asarray(exact(AA.ravel(), BB.ravel()), F32).reshape(ny, nx)
        if key is not None:
            if len(_GRID_CACHE) > 96:
                _GRID_CACHE.clear()
            _GRID_CACHE[key] = g
    inv, fa0, fb0 = F32(1.0 / cell), F32(a0), F32(b0)
    if cubic:
        mx, my = F32(nx - 2.001), F32(ny - 2.001)
        gp = np.pad(g, 1, mode="edge")            # one extra ring of cells for the 4x4 stencil
    else:
        mx, my = F32(nx - 1.001), F32(ny - 1.001)
        gp = g
    gw, gf = gp.shape[1], gp.ravel()
    fb = fallback or exact

    def ev(a, b):
        ga = (a - fa0) * inv
        gb = (b - fb0) * inv
        ia = np.clip(ga, 0, mx)
        ib = np.clip(gb, 0, my)
        i0, j0 = ib.astype(np.int32), ia.astype(np.int32)
        fy, fx = ib - i0, ia - j0
        base = i0 * gw + j0
        if cubic:
            x2, y2 = fx * fx, fy * fy
            x3, y3 = x2 * fx, y2 * fy
            wx = ((2 * x2 - x3 - fx), (3 * x3 - 5 * x2 + 2), (4 * x2 - 3 * x3 + fx), (x3 - x2))
            wy = ((2 * y2 - y3 - fy), (3 * y3 - 5 * y2 + 2), (4 * y2 - 3 * y3 + fy), (y3 - y2))
            v = 0
            for r in range(4):
                br = base + r * gw
                v = v + wy[r] * (wx[0] * gf[br] + wx[1] * gf[br + 1] + wx[2] * gf[br + 2] + wx[3] * gf[br + 3])
            v = v * F32(0.25)
        else:
            v00, v01, v10, v11 = gf[base], gf[base + 1], gf[base + gw], gf[base + gw + 1]
            v = (v00 + (v01 - v00) * fx) * (1 - fy) + (v10 + (v11 - v10) * fx) * fy
        out = np.flatnonzero((ga != ia) | (gb != ib))
        if len(out):                              # off the grid (shadow / AO probes)
            v[out] = fb(a[out], b[out])
        return v
    return ev


def lathe(profile, smooth=True, thickness=None):
    """Revolve a 2D profile around the y axis (bottles, vases, cups, lamps, chess pieces).

    ``profile`` = [(radius, y), ...] from the bottom to the top; the axis closes the shape. ``smooth=True``
    runs a spline through the points (glossy surfaces reflect it cleanly, with no facets). ``thickness`` makes a
    hollow skin (a vase that is open at the top: end the profile at the rim instead of the axis)."""
    P0 = np.asarray(profile, np.float64)
    smooth = bool(smooth and len(P0) >= 3)
    t = 0.0 if thickness is None else float(thickness)

    def dist_fn(P):
        P = P.copy()
        P[:, 0] = np.maximum(P[:, 0], 0)
        if thickness is not None:
            return (lambda q, y: _polyline2d(q, y, P) - F32(t / 2)), P
        Q = P
        if Q[0, 0] > 1e-9:
            Q = np.vstack([[0.0, Q[0, 1]], Q])
        if Q[-1, 0] > 1e-9:
            Q = np.vstack([Q, [0.0, Q[-1, 1]]])
        return (lambda q, y: _poly2d(q, y, Q, skip_axis=True)), Q

    coarse, Q = dist_fn(_catmull(P0, 8) if smooth else P0)
    rm, ylo, yhi = Q[:, 0].max() + t / 2, Q[:, 1].min() - t / 2, Q[:, 1].max() + t / 2
    if smooth:
        fine, _ = dist_fn(_catmull(P0, 40))
        ext = max(rm, yhi - ylo)
        cell = ext / 400.0
        if t:
            cell = min(cell, t / 4)
        cell = max(cell, ext / 1400.0)
        key = ("lathe", P0.tobytes(), t, cell)
        # a coarse wide grid answers the shadow / AO probes around the object, the fine one the surface
        far = ext * 0.6
        outer = _grid2d(coarse, 0.0, rm + far, ylo - far, yhi + far, cell * 8, key + ("outer",), coarse, False)
        pad = max(cell * 4, ext * 0.03)
        g = _grid2d(fine, 0.0, rm + pad, ylo - pad, yhi + pad, cell, key, outer, _LATHE_CUBIC)

        def fn(x, y, z):
            return g(np.sqrt(x * x + z * z), y)
        sh = _Prim(fn, [-rm, ylo, -rm], [rm, yhi, rm])
        sh.cost = 2
        return sh

    def fn(x, y, z):
        return coarse(np.sqrt(x * x + z * z), y)
    sh = _Prim(fn, [-rm, ylo, -rm], [rm, yhi, rm])
    sh.cost = 2 + len(Q) // 4
    return sh


def extrude(shape2d, depth=0.2, size=(2.0, 2.0)):
    """Extrude any 2D distance function along z (logos, signs, letters with depth).

    ``shape2d(X, Y)`` returns distances in world units; it is called with y pointing *down*, so the shapes in
    ``pixelcanvas.sdf`` work as drawn: ``extrude(lambda X, Y: sdf.sd_star(X, Y, 0, 0, 0.5), 0.1)``.
    ``size`` = (w, h) bounding box of the 2D shape around the origin."""
    w, h = size

    def d2(x, y):
        return np.asarray(shape2d(x, -y), F32)
    sh = _Prim(_extrude_fn(d2, depth), [-w / 2, -h / 2, -depth / 2], [w / 2, h / 2, depth / 2])
    sh.cost = 4
    return sh


def extrude_text(text, height=0.5, depth=0.1, font=None, bold=True, res=160):
    """3D letters: ``text`` with cap height about ``height``, extruded ``depth`` along z, centred on the origin."""
    from .text import text_mask, text_bbox
    size = res
    x0, y0, x1, y1 = text_bbox(text, (0, 0), size, font=font, bold=bold)
    pad = int(size * 0.25)
    W, H = int(x1 - x0) + 2 * pad, int(y1 - y0) + 2 * pad
    cov = text_mask(H, W, text, (pad - x0, pad - y0), size, font=font, bold=bold, aa=4)
    m = cov > 0.5
    dist = _edt_signed(m)                                # px, negative inside
    edge = (cov > 0.02) & (cov < 0.98)
    dist = np.where(edge, 0.5 - cov, dist - np.sign(dist) * 0.5)   # sub-pixel edge from the coverage
    from .filters import gaussian_blur
    dist = gaussian_blur(dist.astype(F32), 1.5)                     # smooth side walls
    unit = height / float(size * 0.72)
    sdf_grid = (dist * unit).astype(F32)
    ww, hh = W * unit, H * unit

    def d2(x, y):
        gx = (x + ww / 2) / unit - 0.5
        gy = (y + hh / 2) / unit - 0.5
        ix = np.clip(gx, 0, W - 1.001)
        iy = np.clip(gy, 0, H - 1.001)
        i0, j0 = iy.astype(np.int32), ix.astype(np.int32)
        fy, fx = iy - i0, ix - j0
        g = sdf_grid
        v = (g[i0, j0] * (1 - fx) + g[i0, j0 + 1] * fx) * (1 - fy) + (g[i0 + 1, j0] * (1 - fx) + g[i0 + 1, j0 + 1] * fx) * fy
        out = np.maximum(np.abs(gx - ix), np.abs(gy - iy)) * F32(unit)
        return v + out
    sh = _Prim(_extrude_fn(lambda x, y: d2(x, -y), depth), [-ww / 2, -hh / 2, -depth / 2], [ww / 2, hh / 2, depth / 2])
    sh.cost = 4
    return sh


def _edt_signed(m):
    """Signed Euclidean distance (px) for a boolean mask (negative inside)."""
    try:
        from scipy import ndimage
        return ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)
    except Exception:                                     # pragma: no cover - numpy fallback
        H, W = m.shape
        edge = m ^ np.roll(m, 1, 0) | m ^ np.roll(m, 1, 1)
        ey, ex = np.nonzero(edge)
        Y, X = np.mgrid[0:H, 0:W]
        d = np.full((H, W), np.inf)
        for i in range(0, len(ey), 512):
            dy = Y[..., None] - ey[None, None, i:i + 512]
            dx = X[..., None] - ex[None, None, i:i + 512]
            d = np.minimum(d, np.sqrt(dx * dx + dy * dy).min(-1))
        return np.where(m, -d, d)


# -------------------------------------------------------------------------------------------- operator nodes

class _Union(Shape):
    def __init__(self, children, k=0.0):
        flat = []
        for c in children:
            if isinstance(c, _Union) and c.k == 0 and k == 0 and not getattr(c, "keep", False):
                flat.extend(c.children)
            else:
                flat.append(c)
        self.children, self.k = flat, float(k)
        self.cost = sum(c.cost for c in flat)

    def _d(self, x, y, z):
        if self.k > 0:
            return self._smooth(x, y, z)
        d = None
        for c in self.children:
            if d is None:
                d = c._d(x, y, z)
                continue
            if c.cost >= 2 and len(x) > 64 and not _REF:
                lo, hi = c.bounds()
                lam, mu = _lowb(c)
                lb = _bdist(lo, hi, x, y, z)
                if lam < 1 or mu > 0:                    # a child that under-reports its distance
                    lb = lb * F32(lam) - F32(mu)
                need = np.flatnonzero((lb < d) | (lb <= 0))
                if len(need) == 0:
                    continue
                if len(need) < len(x):
                    sub = c._d(x[need], y[need], z[need])
                    d[need] = np.minimum(d[need], sub)
                    continue
            d = np.minimum(d, c._d(x, y, z))
        return d

    def _smooth(self, x, y, z):
        """Smooth union. A child only changes the blend where it comes within ``k`` of the running distance, so
        children whose bounding box is farther than that are skipped: the field is the same as blending all."""
        k = F32(self.k)
        d = self.children[0]._d(x, y, z)
        n = len(x)
        for c in self.children[1:]:
            if n > 48 and not _REF:
                lo, hi = c.bounds()
                lam, mu = _lowb(c)
                lb = _bdist(lo, hi, x, y, z)
                need = np.flatnonzero((lb * F32(lam) - F32(mu) < d + k) if lam < 1 or mu > 0 else (lb < d + k))
                if len(need) == 0:
                    continue
                if len(need) < n * 0.8:
                    dn = d[need]
                    b = c._d(x[need], y[need], z[need])
                    h = np.clip(F32(0.5) + F32(0.5) * (b - dn) / k, 0, 1)
                    d[need] = b * (1 - h) + dn * h - k * h * (1 - h)
                    continue
            b = c._d(x, y, z)
            h = np.clip(F32(0.5) + F32(0.5) * (b - d) / k, 0, 1)
            d = b * (1 - h) + d * h - k * h * (1 - h)
        return d

    def _dc(self, x, y, z, cap):
        if self.k > 0 or _REF:
            return self._d(x, y, z)
        d = np.array(cap, F32, copy=True)
        n = len(x)
        for c in self.children:
            if c.cost >= 2 and n > 16:
                lo, hi = c.bounds()
                lam, mu = _lowb(c)
                lb = _bdist(lo, hi, x, y, z)
                need = np.flatnonzero((lb * F32(lam) - F32(mu) < d) if lam < 1 or mu > 0 else (lb < d))
                if len(need) == 0:
                    continue
                if len(need) < n:
                    sub = c._dc(x[need], y[need], z[need], d[need])
                    d[need] = np.minimum(d[need], sub)
                    continue
                np.minimum(d, c._dc(x, y, z, d), out=d)
            else:
                np.minimum(d, c._d(x, y, z), out=d)
        return d

    def _bb(self):
        los, his = zip(*(c.bounds() for c in self.children))
        m = self.k
        return np.min(los, 0) - m * 0.25, np.max(his, 0) + m * 0.25


class _Sub(Shape):
    def __init__(self, a, b, k=0.0):
        self.a, self.b, self.k = a, b, float(k)
        self.cost = a.cost + b.cost

    def _blb(self, x, y, z):
        """A lower bound of b's distance from its box (scaled for fields that under-report)."""
        lo, hi = self.b.bounds()
        lb = _bdist(lo, hi, x, y, z)
        if len(x) > 64:
            lam, mu = _lowb(self.b)
            if lam < 1 or mu > 0:
                lb = lb * F32(lam) - F32(mu)
        return lb

    def _d(self, x, y, z):
        da = self.a._d(x, y, z)
        k = self.k
        lb = self._blb(x, y, z)
        need = np.flatnonzero((lb < k - da) | (lb <= k)) if len(x) > 64 and not _REF else np.arange(len(x))
        if len(need) == 0:
            return da
        db = self.b._d(x[need], y[need], z[need])
        a = da[need]
        if k > 0:
            kk = F32(k)
            h = np.clip(F32(0.5) - F32(0.5) * (a + db) / kk, 0, 1)
            r = a * (1 - h) - db * h + kk * h * (1 - h)
        else:
            r = np.maximum(a, -db)
        out = da.copy()
        out[need] = r
        return out

    def _dc(self, x, y, z, cap):
        # a - b is never closer than a, so the cap goes to a; b is only needed where a is near
        if _REF or type(self.a)._dc is Shape._dc:
            return self._d(x, y, z)
        da = self.a._dc(x, y, z, cap)
        k = self.k
        lb = self._blb(x, y, z)
        need = np.flatnonzero(((lb < k - da) | (lb <= k)) & (da < cap)) if len(x) > 64 else \
            np.flatnonzero(da < cap)
        if len(need) == 0:
            return da
        db = self.b._d(x[need], y[need], z[need])
        a = da[need]
        if k > 0:
            kk = F32(k)
            h = np.clip(F32(0.5) - F32(0.5) * (a + db) / kk, 0, 1)
            r = a * (1 - h) - db * h + kk * h * (1 - h)
        else:
            r = np.maximum(a, -db)
        out = np.array(da, F32, copy=True)
        out[need] = r
        return out

    def _bb(self):
        return self.a.bounds()


class _Inter(Shape):
    def __init__(self, a, b, k=0.0):
        self.a, self.b, self.k = a, b, float(k)
        self.cost = a.cost + b.cost

    def _d(self, x, y, z):
        da, db = self.a._d(x, y, z), self.b._d(x, y, z)
        if self.k > 0:
            kk = F32(self.k)
            h = np.clip(F32(0.5) - F32(0.5) * (db - da) / kk, 0, 1)
            return db * (1 - h) + da * h + kk * h * (1 - h)
        return np.maximum(da, db)

    def _dc(self, x, y, z, cap):
        # max(a, b) is never closer than a: b is only needed where a comes within the cap
        if self.k > 0 or _REF:
            return self._d(x, y, z)
        da = self.a._dc(x, y, z, cap)
        need = np.flatnonzero(da < cap)
        if len(need) == len(x):
            return np.maximum(da, self.b._dc(x, y, z, cap))
        if len(need) == 0:
            return da
        out = np.array(da, F32, copy=True)
        out[need] = np.maximum(da[need], self.b._dc(x[need], y[need], z[need], cap[need]))
        return out

    def _bb(self):
        (l1, h1), (l2, h2) = self.a.bounds(), self.b.bounds()
        return np.maximum(l1, l2), np.minimum(h1, h2)


class _Round(Shape):
    def __init__(self, child, r):
        self.child, self.r = child, float(r)
        self.cost = child.cost

    def _d(self, x, y, z):
        return self.child._d(x, y, z) - F32(self.r)

    def _dc(self, x, y, z, cap):
        if _REF:
            return self._d(x, y, z)
        return self.child._dc(x, y, z, cap + F32(self.r)) - F32(self.r)

    def _bb(self):
        lo, hi = self.child.bounds()
        return lo - self.r, hi + self.r


class _Shell(Shape):
    def __init__(self, child, t):
        self.child, self.t = child, float(t)
        self.cost = child.cost

    def _d(self, x, y, z):
        return np.abs(self.child._d(x, y, z)) - F32(self.t / 2)

    def _bb(self):
        lo, hi = self.child.bounds()
        return lo - self.t, hi + self.t


class _Mirror(Shape):
    def __init__(self, child, axis):
        self.child, self.axis = child, axis.lower()
        self.cost = child.cost

    def _d(self, x, y, z):
        if "x" in self.axis:
            x = np.abs(x)
        if "y" in self.axis:
            y = np.abs(y)
        if "z" in self.axis:
            z = np.abs(z)
        return self.child._d(x, y, z)

    def _dc(self, x, y, z, cap):
        if _REF:
            return self._d(x, y, z)
        if "x" in self.axis:
            x = np.abs(x)
        if "y" in self.axis:
            y = np.abs(y)
        if "z" in self.axis:
            z = np.abs(z)
        return self.child._dc(x, y, z, cap)

    def _bb(self):
        lo, hi = self.child.bounds()
        lo, hi = lo.copy(), hi.copy()
        for i, a in enumerate("xyz"):
            if a in self.axis:
                m = max(abs(lo[i]), abs(hi[i]))
                lo[i], hi[i] = -m, m
        return lo, hi


class _Repeat(Shape):
    def __init__(self, child, spacing, count):
        self.child = child
        self.s = np.broadcast_to(np.asarray(spacing, np.float64), (3,)).copy()
        self.n = np.broadcast_to(np.asarray(count, np.int64), (3,)).copy()
        self.cost = child.cost

    def _d(self, x, y, z):
        out = []
        for a, s, n in zip((x, y, z), self.s, self.n):
            if n <= 1 or s <= 0:
                out.append(a)
                continue
            off = F32((n - 1) * s / 2)
            k = np.clip(np.round((a + off) / F32(s)), 0, n - 1)
            out.append(a + off - k * F32(s))
        return self.child._d(*out)

    def _bb(self):
        lo, hi = self.child.bounds()
        ext = np.where(self.n > 1, (self.n - 1) * self.s / 2, 0)
        return lo - ext, hi + ext


class _Twist(Shape):
    def __init__(self, child, k):
        self.child, self.k = child, math.radians(k)
        self.cost = child.cost

    def _d(self, x, y, z):
        a = y * F32(self.k)
        c, s = np.cos(a), np.sin(a)
        return self.child._d(c * x + s * z, y, -s * x + c * z) * F32(0.7)

    def _bb(self):
        lo, hi = self.child.bounds()
        r = math.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[2]), abs(hi[2])))
        return np.array([-r, lo[1], -r]), np.array([r, hi[1], r])


class _Bend(Shape):
    def __init__(self, child, k):
        self.child, self.k = child, math.radians(k)
        self.cost = child.cost

    def _d(self, x, y, z):
        a = x * F32(self.k)
        c, s = np.cos(a), np.sin(a)
        return self.child._d(c * x + s * y, -s * x + c * y, z) * F32(0.7)

    def _bb(self):
        lo, hi = self.child.bounds()
        r = math.hypot(max(abs(lo[0]), abs(hi[0])), max(abs(lo[1]), abs(hi[1])))
        return np.array([-r, -r, lo[2]]), np.array([r, r, hi[2]])


def _bvh(shapes):
    """Group many shapes into a bounding-volume tree so far-away groups are skipped during evaluation."""
    if len(shapes) <= 3:
        u = _Union(shapes)
        u.cost = max(u.cost, 2)
        u.keep = True
        return u
    cs = np.array([(s.bounds()[0] + s.bounds()[1]) / 2 for s in shapes])
    ax = int(np.argmax(cs.max(0) - cs.min(0)))
    order = np.argsort(cs[:, ax], kind="stable")
    half = len(shapes) // 2
    a = _bvh([shapes[i] for i in order[:half]])
    b = _bvh([shapes[i] for i in order[half:]])
    u = _Union([a, b])
    u.cost = max(u.cost, 2)
    u.keep = True
    return u


def union(*shapes):
    """Union of any number of shapes (also ``a | b``)."""
    shapes = [s for s in shapes if s is not None]
    if len(shapes) == 1:
        return shapes[0]
    flat = []
    for s in shapes:
        if isinstance(s, _Union) and s.k == 0 and not getattr(s, "keep", False):
            flat.extend(s.children)
        else:
            flat.append(s)
    return _bvh(flat) if len(flat) > 4 else _Union(flat)


def smooth_union(*shapes, k=0.1):
    """Blobby union with a fillet of size ``k`` (clay figures, animals, fruit, soft furniture)."""
    return shapes[0] if len(shapes) == 1 else _Union(list(shapes), k=k)


def subtract(a, *b, k=0.0):
    """``a`` minus every shape in ``b`` (also ``a - b``)."""
    for s in b:
        a = _Sub(a, s, k)
    return a


def smooth_subtract(a, b, k=0.05):
    """``a`` minus ``b`` with a rounded cut edge of size ``k``."""
    return _Sub(a, b, k)


def intersect(a, b, k=0.0):
    """Keep only where both shapes overlap (also ``a & b``)."""
    return _Inter(a, b, k)


# ===================================================================================================== 3D noise

def _hash3(ix, iy, iz, seed=0):
    h = (ix.astype(np.uint32) * np.uint32(374761393) + iy.astype(np.uint32) * np.uint32(668265263)
         + iz.astype(np.uint32) * np.uint32(2147483647) + np.uint32(seed * 1013 + 7))
    h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
    h = h ^ (h >> np.uint32(16))
    return (h & np.uint32(0xFFFF)).astype(F32) * F32(1 / 65535.0)


def noise3(x, y, z, seed=0):
    """Smooth 3D value noise in 0~1 for arrays x, y, z."""
    with np.errstate(over="ignore"):
        x, y, z = _f(x), _f(y), _f(z)
        fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
        ix, iy, iz = fx.astype(np.int64), fy.astype(np.int64), fz.astype(np.int64)
        tx, ty, tz = x - fx, y - fy, z - fz
        tx, ty, tz = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty), tz * tz * (3 - 2 * tz)
        acc = 0
        for dx in (0, 1):
            wx = tx if dx else 1 - tx
            for dy in (0, 1):
                wy = ty if dy else 1 - ty
                for dz in (0, 1):
                    wz = tz if dz else 1 - tz
                    acc = acc + _hash3(ix + dx, iy + dy, iz + dz, seed) * (wx * wy * wz)
        return acc


def fbm3(x, y, z, octaves=4, seed=0):
    """Fractal 3D noise in about 0~1 (sum of ``octaves`` layers of ``noise3``)."""
    tot, amp, norm = 0, 1.0, 0.0
    for o in range(octaves):
        f = F32(2.0 ** o)
        tot = tot + noise3(x * f, y * f, z * f, seed + o * 17) * F32(amp)
        norm += amp
        amp *= 0.5
    return tot / F32(norm)


# ===================================================================================================== materials

def _lin(c):
    """sRGB display colour -> linear light."""
    return np.power(np.clip(rgb(c), 0, None), 2.2).astype(F32)


class _Wood:
    """Wood colour: fine late-wood grain lines that meander and form cathedral arches, pores along the grain
    and a slow tone drift between boards. Detail smaller than a pixel fades to its average (no moire)."""

    _kind = "wood"

    def __init__(self, light="#d6a46a", dark="#8f5a30", scale=1.0, axis="x", seed=0):
        self.L, self.D, self.size, self.ax, self.seed = _lin(light), _lin(dark), float(scale), "xyz".index(axis), seed

    def __call__(self, P, N=None, foot=None):
        sd = self.seed
        p = P / F32(self.size)
        a, b = [p[:, i] for i in range(3) if i != self.ax]
        along = p[:, self.ax]
        fpx = (np.full(len(P), F32(1e-4)) if foot is None else foot / F32(self.size)).astype(F32)
        # growth rings of a log whose axis runs along the grain; slow meanders bend them into arches
        bend = (fbm3(along * 0.8, a * 2.0, b * 2.0, 3, sd) - 0.5) * F32(0.09)
        wob = (fbm3(along * 3.5, a * 9, b * 9, 2, sd + 1) - 0.5) * F32(0.014)
        r = np.sqrt((a + F32(0.37)) ** 2 + (b + F32(0.21)) ** 2) + bend + wob
        period = F32(0.0055)
        ph = r / period
        ph = ph + (fbm3(r * 30, along * 0.3, r * 7, 2, sd + 3) - 0.5) * F32(5.0)   # uneven ring widths
        f = ph - np.floor(ph)
        line = np.clip((f - 0.62) / 0.2, 0, 1) ** 2 * np.clip((1.0 - f) / 0.18, 0, 1)   # sharp late-wood edge
        lk = np.clip(1.6 - fpx / (period * 0.35), 0, 1)                                    # fade when < 1 px
        line = line * lk + F32(0.16) * (1 - lk)
        pores = noise3(along * 9.0, a * 700, b * 700, sd + 5)
        pk = np.clip(1.6 - fpx / F32(0.0007), 0, 1)
        pores = (pores - 0.5) * pk
        tone = fbm3(along * 0.5, a * 5, b * 5, 2, sd + 7) - 0.5
        t = np.clip(line * 0.62 + pores * 0.22 + tone * 0.55 + 0.18, 0, 1)
        return self.L[None] * (1 - t[:, None]) + self.D[None] * t[:, None]


def wood(light="#d6a46a", dark="#8f5a30", scale=1.0, axis="x", seed=0):
    """Procedural wood colour (``'x'`` = grain runs along x): meandering grain lines, pores and board-to-board
    tone. ``scale`` > 1 makes the grain coarser."""
    return _Wood(light, dark, scale, axis, seed)


def marble(base="#eeeae4", vein="#7a7a80", scale=1.0, seed=0):
    """Procedural marble: soft veins through a 3D noise field."""
    B, V = _lin(base), _lin(vein)

    def fn(P, N=None):
        p = P / F32(scale)
        n = fbm3(p[:, 0] * 2, p[:, 1] * 2, p[:, 2] * 2, 5, seed)
        n2 = fbm3(p[:, 0] * 5 + 3, p[:, 1] * 5, p[:, 2] * 5, 3, seed + 9)
        v = np.abs(np.sin((p[:, 0] + p[:, 1] * 0.5 + p[:, 2] * 0.3) * 2.5 + n * 7))
        t = np.clip(1 - v, 0, 1) ** 4 * 0.55 + np.clip(1 - np.abs(np.sin(n2 * 11)), 0, 1) ** 10 * 0.25
        t = np.clip(t + (n - 0.5) * 0.12, 0, 1)
        return B[None] * (1 - t[:, None]) + V[None] * t[:, None]
    fn._kind = "marble"
    return fn


class _Checker:
    """Box-filtered checker in the (x, z) plane (no moire at a distance)."""

    def __init__(self, size=1.0, a="#e9e4da", b="#c9c1b3"):
        self.size, self.a, self.b = float(size), _lin(a), _lin(b)

    def __call__(self, P, N=None, foot=None):
        u, v = P[:, 0] / F32(self.size), P[:, 2] / F32(self.size)
        w = np.full(u.shape, F32(0.01)) if foot is None else np.maximum(foot / F32(self.size) * 1.5, F32(0.01))

        def tri(x):
            h = x * 0.5
            return np.abs(h - np.floor(h + 0.5)) * 2
        # integral of the square wave, box-filtered over width w
        def filt(x):
            return (tri(x + w * 0.5) - tri(x - w * 0.5)) / w

        sx, sz = filt(u), filt(v)
        t = np.clip(0.5 - 0.5 * sx * sz, 0, 1)
        return self.a[None] * (1 - t[:, None]) + self.b[None] * t[:, None]


def checker(size=1.0, a="#e9e4da", b="#c9c1b3"):
    """Checkerboard colour function for a floor (squares of ``size`` world units)."""
    return _Checker(size, a, b)


def _tri(x):
    h = x * 0.5
    return np.abs(h - np.floor(h + 0.5)) * 2


class _Stripes(_Checker):
    """Box-filtered stripes across ``axis`` (crisp up close, no moire at a distance)."""

    def __init__(self, width=0.2, a="#f2efe8", b="#d24a3a", axis="x"):
        self.size, self.a, self.b, self.ax = float(width), _lin(a), _lin(b), "xyz".index(axis)

    def __call__(self, P, N=None, foot=None):
        u = P[:, self.ax] / F32(self.size)
        w = np.full(u.shape, F32(0.01)) if foot is None else np.maximum(foot / F32(self.size) * 1.5, F32(0.01))
        t = np.clip(0.5 - 0.5 * (_tri(u + w * 0.5) - _tri(u - w * 0.5)) / w, 0, 1)
        return self.a[None] * (1 - t[:, None]) + self.b[None] * t[:, None]


def stripes(width=0.2, a="#f2efe8", b="#d24a3a", axis="x"):
    """Stripes colour function (bands across ``axis``) -> awnings, rugs, toys, prints."""
    return _Stripes(width, a, b, axis)


class _Weave(_Checker):
    """Woven-fabric colour: a plain weave of round threads (about 1.2 mm apart at ``scale`` 1) over heathered
    yarn. The weave fades to its average once a thread is smaller than a pixel."""

    def __init__(self, color="#8a94a8", scale=1.0, seed=0):
        self.c, self.size, self.seed = _lin(color), float(scale), seed

    def __call__(self, P, N=None, foot=None):
        p = P / F32(self.size)
        heather = fbm3(p[:, 0] * 45, p[:, 1] * 45, p[:, 2] * 45, 2, self.seed) - 0.5
        blotch = fbm3(p[:, 0] * 6, p[:, 1] * 6, p[:, 2] * 6, 2, self.seed + 3) - 0.5
        th = F32(0.0012 * self.size)
        f = np.ones(len(P), F32) if foot is None else np.clip(1.6 - foot / (th * F32(0.45)), 0, 1)
        def cloth(u, v):
            u, v = u / th, v / th
            fu, fv = u - np.floor(u), v - np.floor(v)
            over = ((np.floor(u) + np.floor(v)) % 2) < 1
            prof = np.where(over, np.sin(fu * F32(math.pi)), np.sin(fv * F32(math.pi)))
            gap = np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv))
            return np.sqrt(np.clip(prof, 0, 1)) * F32(0.55) + np.clip(gap / F32(0.12), 0, 1) * F32(0.15)
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        if N is None:
            tw = cloth(y, x + z)
        else:                                              # triplanar: threads follow every side of a form
            w = np.abs(np.asarray(N, F32)) ** 4
            w = w / np.maximum(w.sum(1, keepdims=True), F32(1e-6))
            tw = w[:, 0] * cloth(y, z) + w[:, 2] * cloth(y, x) + w[:, 1] * cloth(x, z)
        tw = tw - F32(0.48)
        k = 1 + heather * F32(0.22) + blotch * F32(0.12) + tw * f * F32(0.9)
        return self.c[None] * np.clip(k, 0.05, None)[:, None].astype(F32)


def weave(color="#8a94a8", scale=1.0, seed=0):
    """Fabric colour function: heathered woven texture in ``color`` (the 'fabric' preset uses it)."""
    return _Weave(color, scale, seed)


def planks(light="#c99a68", dark="#a8774a", width=0.18, length=1.6, seed=4):
    """Floorboards running along x: each board its own tone, wood grain streaks and dark seams."""
    A, B = _lin(light), _lin(dark)

    def fn(P, N=None):
        x, z = P[:, 0], P[:, 2]
        row = np.floor(z / width)
        off = (row * 0.37) % 1.0 * length
        col = np.floor((x + off) / length)
        tone = noise3(row * 3.1, col * 5.7, 0.5, seed=seed)
        grain = noise3(x * 1.5, z * 40, row, seed=seed + 3)
        t = np.clip(tone * 0.7 + grain * 0.5 - 0.1, 0, 1)
        c = A[None] * (1 - t[:, None]) + B[None] * t[:, None]
        fz = (z / width) % 1.0
        fx = ((x + off) / length) % 1.0
        seam = np.minimum(np.minimum(fz, 1 - fz) * width, np.minimum(fx, 1 - fx) * length)
        dark_ = np.clip(seam / 0.006, 0, 1) * 0.35 + 0.65
        return c * dark_[:, None]
    return fn


def stone(color="#8e8a82", scale=1.0, seed=0):
    """Stone colour function: mottled tone, fine grain speckle and faint darker veins in ``color``."""
    A = _lin(color)
    B, L = A * F32(0.6 ** 2.2), np.minimum(A * F32(1.18 ** 2.2), 1)

    def fn(P, N=None):
        p = P / F32(scale)
        mot = fbm3(p[:, 0] * 3, p[:, 1] * 3, p[:, 2] * 3, 3, seed)
        grain = noise3(p[:, 0] * 60, p[:, 1] * 60, p[:, 2] * 60, seed + 3)
        vein = np.clip(1 - np.abs(fbm3(p[:, 0] * 1.6, p[:, 1] * 1.6, p[:, 2] * 1.6, 3, seed + 7) - 0.5) * 18, 0, 1)
        t = np.clip((mot - 0.5) * 1.8 + 0.5, 0, 1)[:, None]
        c = B[None] * (1 - t) * 0.35 + A[None] * (1 - t) * 0.65 + L[None] * t
        c = c * (0.86 + 0.28 * grain)[:, None] * (1 - 0.35 * vein)[:, None]
        return c.astype(F32)
    fn._kind = "stone"
    return fn


def moss(base="#8e8a82", amount=0.5, color="#4e6a2a", scale=1.0, seed=0):
    """Moss over any colour or colour function ``base``: soft green patches that grow on the upward-facing
    sides first. ``amount`` 0 (bare) ~ 1 (covered)."""
    M1 = _lin(color)
    M2 = np.minimum(M1 * F32(1.3 ** 2.2) * np.array([1.08, 1.0, 0.8], F32), 1)
    amt = float(amount)

    def fn(P, N=None):
        if callable(base):
            try:
                c = np.asarray(base(P, N), F32).reshape(len(P), 3)
            except TypeError:
                c = np.asarray(base(P), F32).reshape(len(P), 3)
        else:
            c = np.broadcast_to(_lin(base), (len(P), 3))
        p = P / F32(scale)
        up = np.full(len(P), F32(0.5)) if N is None else np.asarray(N, F32)[:, 1]
        patch = fbm3(p[:, 0] * 4, p[:, 1] * 4, p[:, 2] * 4, 4, seed + 21)
        rag = noise3(p[:, 0] * 22, p[:, 1] * 22, p[:, 2] * 22, seed + 23)
        t = up * F32(0.9) + (patch - 0.5) * F32(2.2) + (rag - 0.5) * F32(0.5) + F32(amt * 1.9 - 1.15)
        t = np.clip(t / F32(0.3), 0, 1)
        fuzz = noise3(p[:, 0] * 45, p[:, 1] * 45, p[:, 2] * 45, seed + 22)
        mc = M1[None] * (1 - fuzz[:, None]) + M2[None] * fuzz[:, None]
        mc = mc * (0.6 + 0.6 * patch * rag)[:, None]                      # darker, thinner moss at the edges
        mc = mc * (0.65 + 0.35 * t)[:, None]
        return (c * (1 - t[:, None]) + mc * t[:, None]).astype(F32)
    fn._kind = "moss"
    return fn


def _tinted(preset, c):
    """A preset's procedural look in the user's colour: wood grain, marble veins or woven fabric."""
    if preset == "wood":
        return wood(c, _shade_hex(c, 0.68))
    if preset == "marble":
        return marble(c, _shade_hex(c, 0.5))
    if preset == "fabric":
        return weave(c)
    if preset == "stone":
        return stone(c)
    return c


class Material:
    """Surface description. ``color`` is a colour or a function ``f(P, N) -> (N, 3) linear colours`` of the
    object's local coordinates (see ``wood``, ``marble``, ``stripes``).

    roughness 0 (mirror) ~ 1 (chalk); specular = highlight strength; metallic 0/1; reflect = extra mirror
    reflection on top of Fresnel; emissive = glow strength; subsurface = soft light wrap (skin, wax, clay);
    glass = transmission 0~1 (tinted by colour); ior; clearcoat = glossy varnish layer; sheen = soft
    velvet glow at grazing angles (cloth)."""

    def __init__(self, color="#c8c8c8", roughness=0.5, specular=0.5, metallic=0.0, reflect=0.0, emissive=0.0,
                 subsurface=0.0, sss_color=None, glass=0.0, ior=1.45, clearcoat=0.0, shadow=True, sheen=0.0):
        self.color = color
        self.roughness, self.specular, self.metallic = float(roughness), float(specular), float(metallic)
        self.reflect, self.emissive, self.subsurface = float(reflect), float(emissive), float(subsurface)
        self.sss_color = sss_color
        self.glass, self.ior, self.clearcoat, self.shadow = float(glass), float(ior), float(clearcoat), shadow
        self.sheen = float(sheen)

    def copy(self, **kw):
        m = Material(**{k: getattr(self, k) for k in ("color", "roughness", "specular", "metallic", "reflect",
                                                         "emissive", "subsurface", "sss_color", "glass", "ior",
                                                         "clearcoat", "shadow", "sheen")})
        for k, v in self.__dict__.items():              # tags such as a window's view name travel along
            if k not in m.__dict__:
                setattr(m, k, v)
        for k, v in _reject_unknown("Material", kw, _sig_options(Material.__init__)).items():
            setattr(m, k, v)
        return m

    def albedo(self, P, N, foot=None):
        c = self.color
        if callable(c):
            try:
                out = c(P, N, foot) if isinstance(c, (_Checker, _Wood)) else c(P, N)
            except TypeError:
                out = c(P)
            return np.asarray(out, F32).reshape(len(P), 3)
        return np.broadcast_to(_lin(c), (len(P), 3))


PRESETS = {
    "matte": dict(color="#bdb8b0", roughness=0.9, specular=0.15),
    "plastic": dict(color="#d84a3a", roughness=0.32, specular=0.6),
    "glossy": dict(color="#2a6fd6", roughness=0.08, specular=1.0, reflect=0.06, clearcoat=0.6),
    "ceramic": dict(color="#f2efe9", roughness=0.12, specular=0.9, reflect=0.03, clearcoat=0.7),
    "metal": dict(color="#b8bcc4", roughness=0.3, metallic=1.0),
    "chrome": dict(color="#f4f6f8", roughness=0.02, metallic=1.0),
    "gold": dict(color="#ffcf6b", roughness=0.15, metallic=1.0),
    "copper": dict(color="#f0a080", roughness=0.2, metallic=1.0),
    "glass": dict(color="#eef8f6", roughness=0.02, specular=1.0, glass=0.92, ior=1.45, shadow=0.35),
    "rubber": dict(color="#26272b", roughness=0.85, specular=0.25),
    "clay": dict(color="#e0a084", roughness=0.75, specular=0.2, subsurface=0.45),
    "wood": dict(color=wood(), roughness=0.5, specular=0.35),
    "marble": dict(color=marble(), roughness=0.12, specular=0.8, reflect=0.04),
    "skin": dict(color="#f0c0a0", roughness=0.55, specular=0.3, subsurface=0.6, sss_color="#ff6a50"),
    "emissive": dict(color="#ffd27a", roughness=0.5, specular=0.0, emissive=3.0, shadow=False),
    "fabric": dict(color="#8a94a8", roughness=0.95, specular=0.08, subsurface=0.15, sheen=0.6),
    "paint": dict(color="#e8452c", roughness=0.25, specular=0.8, clearcoat=0.8),
    "leaf": dict(color="#4f9a3c", roughness=0.6, specular=0.3, subsurface=0.5, sss_color="#b8e060"),
    "stone": dict(color=stone(), roughness=0.85, specular=0.15),
}


def material(spec=None, color=None, **kw):
    """Make a Material from a preset name, a colour, or an existing Material, then apply overrides.

    ``material("chrome")``, ``material("plastic", "#2a9df4")``, ``material("#e33")`` (plastic in that colour),
    ``material("clay", "#9ad", roughness=0.6)``. A tuple ``("gold", "#fc0")`` works too. ``moss=0.5`` grows
    moss on the upward-facing sides of any material (``moss_color=`` to change its green)."""
    mo, mo_c = kw.pop("moss", None), kw.pop("moss_color", None)
    if mo or mo_c:
        m = material(spec, color, **kw)
        m.color = moss(m.color, 0.5 if mo is None else mo, mo_c or "#4e6a2a")
        m.roughness = max(m.roughness, 0.6)
        return m
    if isinstance(spec, Material):
        m = spec.copy()
    elif isinstance(spec, (tuple, list)) and len(spec) and isinstance(spec[0], str) and spec[0] in PRESETS:
        return material(spec[0], *spec[1:], color=color, **kw) if color else material(*spec, **kw)
    elif isinstance(spec, str) and spec in PRESETS:
        m = Material(**PRESETS[spec])
        if color is not None and not callable(color):
            color = _tinted(spec, color)        # material("wood", "#7a4a2c") keeps the grain, in that colour
        elif color is None and spec == "fabric":
            color = weave(m.color)
    elif spec is None:
        m = Material(**PRESETS["plastic"])
        m.color = "#c8c8c8"
    else:                                                   # a colour / colour function
        if isinstance(spec, str):
            try:
                _lin(spec)
            except (ValueError, KeyError, TypeError):
                raise _unknown_name("material", spec, PRESETS, "Try solid.PRESETS for the list") from None
        m = Material(**PRESETS["plastic"])
        m.color = spec
    if color is not None:
        m.color = color
    return m.copy(**kw) if kw else m


# ===================================================================================================== models

class Part:
    """One named, single-material piece of a Model."""

    def __init__(self, name, shape, mat):
        self.name, self.shape, self.mat = name, shape, material(mat)

    def local(self, P):
        """World points -> this part's own coordinates (procedural colours stick to the object)."""
        s = self.shape
        if isinstance(s, _Affine):
            M, c = s.Minv.astype(F32), s.c.astype(F32)
            return P @ M.T + c
        return P


class Model(_Xf):
    """A named group of parts (each with its own material). Transforms apply to every part.

    ``m["wheels"]`` -> Part; ``m.recolor(body="#3a7")``; ``m.material(body="chrome")``; ``m + other`` merges."""

    def __init__(self, name, parts, hangs=False):
        self.name, self.parts, self.hangs = name, list(parts), hangs

    def _affine(self, F, g):
        return Model(self.name, [Part(p.name, _Affine(p.shape, F, g), p.mat) for p in self.parts], self.hangs)

    def bounds(self):
        los, his = zip(*(p.shape.bounds() for p in self.parts))
        return np.min(los, 0), np.max(his, 0)

    @property
    def size(self):
        lo, hi = self.bounds()
        return hi - lo

    def __getitem__(self, name):
        for p in self.parts:
            if p.name == name:
                return p
        raise _bad_name(f"part of {self.name!r}", name, [p.name for p in self.parts])

    def __add__(self, other):
        o = _as_model(other, "extra")
        return Model(self.name, self.parts + [Part(p.name if p.name not in self._names() else f"{o.name}.{p.name}",
                                                   p.shape, p.mat) for p in o.parts])

    def _names(self):
        return {p.name for p in self.parts}

    def recolor(self, **colors):
        """Change part colours: ``car().recolor(body="#2a6", cabin="#fff")``."""
        colors = _reject_unknown("Model.recolor", colors, [p.name for p in self.parts])

        def new(p):
            c = colors[p.name]
            old = p.mat.color
            if not callable(c):                        # keep wood grain / marble veins / weave, in the new colour
                kind = "fabric" if isinstance(old, _Weave) else getattr(old, "_kind", None)
                if kind:
                    c = _tinted(kind, c)
            return p.mat.copy(color=c)
        return Model(self.name, [Part(p.name, p.shape, new(p) if p.name in colors else p.mat) for p in self.parts],
                     self.hangs)

    def material(self, **mats):
        """Swap part materials: ``mug().material(body="gold")``. A preset name keeps the part's colour unless the
        preset is a metal (chrome, gold, copper, metal); pass a Material or ``(preset, colour)`` to set both."""
        mats = _reject_unknown("Model.material", mats, [p.name for p in self.parts])
        out = []
        for p in self.parts:
            if p.name not in mats:
                out.append(p)
                continue
            spec = mats[p.name]
            if isinstance(spec, str) and spec in PRESETS and PRESETS[spec].get("metallic", 0) == 0 \
                    and not callable(PRESETS[spec]["color"]):
                m = material(spec, color=p.mat.color)
            else:
                m = material(spec)
            out.append(Part(p.name, p.shape, m))
        return Model(self.name, out, self.hangs)

    def named(self, name):
        """Rename the model (its layer name)."""
        return Model(name, self.parts, self.hangs)

    def shape(self):
        """All parts as one Shape (for booleans)."""
        return union(*(p.shape for p in self.parts))


def _as_model(obj, name=None, mat=None):
    if isinstance(obj, Model):
        return obj.named(name) if name else obj
    if isinstance(obj, Shape):
        nm = name or "object"
        return Model(nm, [Part(nm, obj, mat)])
    raise TypeError("expected a Shape or a Model")


def model(name, **parts):
    """Build a Model from named parts: ``model("lamp", base=(cylinder(...), "metal"), shade=(cone(...), "#fe9"))``.
    Each value is a Shape or (Shape, material spec). The keyword names are the part names (free-form)."""
    out = []
    for k, v in parts.items():
        if isinstance(v, Shape):
            out.append(Part(k, v, None))
        else:
            out.append(Part(k, v[0], v[1] if len(v) > 1 else None))
    return Model(name, out)


# ----------------------------------------------------------------------------------------------- resting on things

# words for a surface -> part names that provide it (first match wins); "top" falls back to the whole model
_SURFACE_PARTS = {"seat": ("seat", "cushions", "cushion", "top"), "top": ("top",), "tabletop": ("top",),
                  "armrest_l": ("armrest_l", "arm_l"), "armrest_r": ("armrest_r", "arm_r"),
                  "armrest": ("armrest_r", "armrest_l"), "arm": ("armrest_r", "armrest_l"),
                  "shelf": ("shelves", "frame"), "deck": ("deck",), "lid": ("lid",), "roof": ("roof",),
                  "bed": ("duvet", "mattress"), "mattress": ("mattress",), "sill": ("sill", "frames")}


def _shape_of(obj):
    if isinstance(obj, Shape):
        return obj
    if isinstance(obj, Part):
        return obj.shape
    if isinstance(obj, Model):
        return union(*(p.shape for p in obj.parts))
    raise TypeError(f"on(): expected a Model, Part or Shape to rest on, got {type(obj).__name__}")


def _surface_of(target, part, level):
    """(shape, level) of the surface named ``part`` on ``target``."""
    if part is None or isinstance(target, Shape) and not isinstance(target, Model):
        return _shape_of(target), int(level or 0)
    if isinstance(part, Part):
        return part.shape, int(level or 0)
    names = [p.name for p in target.parts]
    if part in names:
        return target[part].shape, int(level or 0)
    for cand in _SURFACE_PARTS.get(part, ()):
        if cand in names:
            lv = level if level is not None else (1 if part == "shelf" else 0)
            return target[cand].shape, int(lv)
    if part == "top":
        return _shape_of(target), int(level or 0)
    raise _bad_name(f"part of {target.name!r}", part, names + [k for k, v in _SURFACE_PARTS.items()
                                                               if any(c in names for c in v)])


def _surface_y(shape, x, z, y0, y1, level=0, up=False):
    """Heights where vertical lines through (x, z) meet ``shape``: marching down from y0 (or up, ``up=True``)
    toward y1, the ``level``-th surface entered (0 = the first). NaN where there is none."""
    n = len(x)
    x = np.asarray(x, F32)
    z = np.asarray(z, F32)
    s = F32(1.0 if up else -1.0)
    y = np.full(n, F32(y0), F32)
    out = np.full(n, np.nan, np.float64)
    count = np.zeros(n, np.int32)
    inside = np.zeros(n, bool)
    eps = F32(max(abs(y1 - y0) * 2e-4, 2e-5))
    act = np.arange(n)
    for _ in range(800):
        if not len(act):
            break
        d = np.asarray(shape._d(x[act].copy(), y[act].copy(), z[act].copy()), F32)
        ins = inside[act]
        hit = ~ins & (d < eps)
        if hit.any():
            h = act[hit]
            done = count[h] >= level
            out[h[done]] = y[h[done]]
            count[h[~done]] += 1
            inside[h[~done]] = True
        left = ins & (d > eps)
        inside[act[left]] = False
        step = np.where(inside[act], np.maximum(-d, eps), np.maximum(d, eps))
        y[act] = y[act] + s * step
        keep = np.isnan(out[act]) & ((y[act] <= y1) if up else (y[act] >= y1))
        act = act[keep]
    return out


# ===================================================================================================== camera

class Camera:
    """Perspective camera (``fov`` = vertical field of view in degrees) or orthographic (``ortho`` = visible
    height in world units). ``Camera.orbit`` and ``Camera.isometric`` are the easy ways to aim it."""

    def __init__(self, position=(0, 1.6, 6), look_at=(0, 0.6, 0), fov=35.0, up=(0, 1, 0), ortho=None):
        self.position, self.look_at = _vec(position), _vec(look_at)
        self.fov, self.up, self.ortho = float(fov), _vec(up), ortho

    @classmethod
    def orbit(cls, target=(0, 0.5, 0), distance=6.0, azimuth=30.0, elevation=20.0, fov=35.0):
        """Camera on a sphere around ``target``: azimuth 0 = from +z (front), 90 = from +x (right side);
        elevation in degrees above the horizon."""
        a, e = math.radians(azimuth), math.radians(elevation)
        t = _vec(target)
        pos = t + distance * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        return cls(pos, t, fov)

    @classmethod
    def frame(cls, obj, azimuth=30.0, elevation=20.0, fov=35.0, fill=0.75, aspect=None):
        """Aim an orbit camera so ``obj`` (a Shape, Model, Scene or a (lo, hi) box) fills ``fill`` of the
        picture. ``aspect`` (w / h) defaults to the canvas it renders into."""
        args = dict(obj=obj, azimuth=azimuth, elevation=elevation, fov=fov, fill=fill)
        auto = aspect is None
        aspect = 16 / 9 if auto else float(aspect)
        lo, hi = obj if isinstance(obj, tuple) else obj.bounds()
        P = _corners(np.asarray(lo, np.float64), np.asarray(hi, np.float64))
        th = math.tan(math.radians(fov) / 2) * max(fill, 1e-3)
        tw = th * aspect

        def fit(P, c, d):
            for _ in range(4):           # fit the projected points, then re-centre them in the frame
                cam = cls.orbit(c, d, azimuth, elevation, fov)
                f, r, u = cam.basis()
                q = P - cam.position
                z = np.maximum(q @ f, 1e-6)
                x, y = (q @ r) / z, (q @ u) / z
                c = c + r * (x.max() + x.min()) / 2 * d + u * (y.max() + y.min()) / 2 * d
                rel = P - c
                zc = rel @ f
                d = float(max(np.max(np.abs(rel @ r) / tw - zc), np.max(np.abs(rel @ u) / th - zc), 1e-3))
            return c, d

        c, d = fit(P, P.mean(0), float(np.linalg.norm(P.max(0) - P.min(0))))
        models = obj.models if isinstance(obj, Scene) else [obj] if isinstance(obj, Model) else \
            [_as_model(obj)] if isinstance(obj, Shape) else None
        if models:                       # refine on the real silhouette (a tiny ray-marched id pass)
            for _ in range(2):
                S = _silhouette(models, cls.orbit(c, d, azimuth, elevation, fov), aspect)
                if len(S) < 8:
                    break
                c, d = fit(S, c, d)
        cam = cls.orbit(c, d, azimuth, elevation, fov)
        cam._refit = (cls.frame, args, aspect) if auto else None
        return cam

    @classmethod
    def isometric(cls, target=(0, 0.5, 0), size=None, azimuth=45.0, elevation=35.264, fill=0.9, aspect=None):
        """True isometric view (orthographic). ``size`` = visible height in world units. ``target`` may be a
        Scene / Model / Shape: then the view is centred on it and ``size`` fits it (``fill``; ``aspect`` w / h
        defaults to the canvas it renders into)."""
        args = dict(target=target, size=size, azimuth=azimuth, elevation=elevation, fill=fill)
        auto = aspect is None and hasattr(target, "bounds") and size is None
        aspect = 16 / 9 if aspect is None else float(aspect)
        if hasattr(target, "bounds"):
            P = _corners(*[np.asarray(v, np.float64) for v in target.bounds()])
            c = cls.orbit(P.mean(0), 100.0, azimuth, elevation)
            f, r, u = c.basis()
            x, y = P @ r, P @ u
            ctr = P.mean(0) + r * ((x.max() + x.min()) / 2 - P.mean(0) @ r) + u * ((y.max() + y.min()) / 2
                                                                                    - P.mean(0) @ u)
            fit = max(y.max() - y.min(), (x.max() - x.min()) / aspect) / max(fill, 1e-3)
            target, size = ctr, (size or fit)
        c = cls.orbit(target, 100.0, azimuth, elevation)
        c.ortho = float(size or 6.0)
        c._refit = (cls.isometric, args, aspect) if auto else None
        return c

    def _for_aspect(self, aspect):
        """This camera re-fitted to a canvas of ``aspect`` (w / h) when it was made by frame / isometric."""
        rf = getattr(self, "_refit", None)
        if not rf or abs(rf[2] - aspect) < 1e-3:
            return self
        return rf[0](**rf[1], aspect=aspect)._with_refit((rf[0], rf[1], aspect))

    def _with_refit(self, rf):
        self._refit = rf
        return self

    def basis(self):
        f = _norm(self.look_at - self.position)
        r = np.cross(f, self.up)
        if np.linalg.norm(r) < 1e-9:
            r = np.cross(f, [0, 0, 1])
        r = _norm(r)
        u = np.cross(r, f)
        return f, r, u

    def _scale(self, h):
        """World units per pixel at unit distance (perspective) or per pixel (ortho)."""
        if self.ortho:
            return self.ortho / h
        return 2 * math.tan(math.radians(self.fov) / 2) / h

    def rays(self, px, py, frame):
        """Ray origins / directions for float pixel coordinates (canvas space) inside ``frame``."""
        x0, y0, w, h = frame
        f, r, u = self.basis()
        s = self._scale(h)
        sx = ((px - (x0 + w / 2)) * s).astype(np.float64)
        sy = (-(py - (y0 + h / 2)) * s).astype(np.float64)
        if self.ortho:
            O = self.position[None] + sx[:, None] * r[None] + sy[:, None] * u[None]
            D = np.broadcast_to(f, O.shape)
        else:
            D = f[None] + sx[:, None] * r[None] + sy[:, None] * u[None]
            D = D / np.linalg.norm(D, axis=1, keepdims=True)
            O = np.broadcast_to(self.position, D.shape)
        return np.ascontiguousarray(O, F32), np.ascontiguousarray(D, F32)

    def project(self, p, frame):
        """World point(s) -> canvas pixel (x, y)."""
        P = np.asarray(p, np.float64)
        one = P.ndim == 1
        P = P.reshape(-1, 3)
        x0, y0, w, h = frame
        f, r, u = self.basis()
        v = P - self.position
        zc, xc, yc = v @ f, v @ r, v @ u
        s = self._scale(h)
        if self.ortho:
            sx, sy = xc / s, yc / s
        else:
            sx, sy = xc / np.maximum(zc, 1e-9) / s, yc / np.maximum(zc, 1e-9) / s
        out = np.stack([x0 + w / 2 + sx, y0 + h / 2 - sy], -1)
        return tuple(out[0]) if one else out


# ===================================================================================================== lighting

def _dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])


LIGHTING = {
    "studio": dict(sun=(-40, 48), sun_color="#fff3e6", sun_power=1.25, softness=0.55,
                   zenith="#d8dce4", horizon="#a9adb6", ground_color="#6d6a66", ambient=0.34,
                   fill=(65, 15), fill_color="#dfe8ff", fill_power=0.16, rim=(160, 35), rim_color="#ffffff",
                   rim_power=0.45, softboxes=[(-40, 48, 34, 22, 6.0), (60, 18, 18, 30, 2.0), (175, 62, 60, 8, 2.5)],
                   env_floor="#4a4846", sun_disk=False, exposure=1.0),
    "soft_box": dict(sun=(-30, 55), sun_color="#ffffff", sun_power=1.0, softness=0.85,
                     zenith="#e6e8ee", horizon="#c4c6cc", ground_color="#8a8884", ambient=0.45,
                     fill=(60, 20), fill_color="#ffffff", fill_power=0.16, rim=(170, 40), rim_power=0.25,
                     softboxes=[(-30, 55, 50, 40, 4.0), (60, 20, 40, 40, 1.5)], env_floor="#6a6866",
                     sun_disk=False, exposure=1.0),
    "sunny": dict(sun=(-35, 52), sun_color="#fff1dc", sun_power=1.55, softness=0.12,
                  zenith="#5f97e8", horizon="#cfe2f5", ground_color="#8c8270", ambient=0.36,
                  fill=(150, 25), fill_color="#bcd4ff", fill_power=0.06, rim=(170, 30), rim_power=0.12,
                  softboxes=[], env_floor="#77705f", sun_disk=True, exposure=1.0),
    "sunset": dict(sun=(-62, 11), sun_color="#ffb070", sun_power=1.6, softness=0.25,
                   zenith="#3a4a86", horizon="#f7a86a", ground_color="#5a4048", ambient=0.3,
                   fill=(120, 20), fill_color="#7a8cff", fill_power=0.14, rim=(150, 15), rim_color="#ffb080",
                   rim_power=0.4, softboxes=[], env_floor="#4a3a3e", sun_disk=True, exposure=1.0),
    "overcast": dict(sun=(-20, 70), sun_color="#f2f4f8", sun_power=0.4, softness=1.0,
                     zenith="#d6dbe2", horizon="#c2c8d0", ground_color="#7c7c78", ambient=0.62,
                     fill=(90, 30), fill_power=0.0, rim=(180, 30), rim_power=0.0, softboxes=[],
                     env_floor="#6c6c68", sun_disk=False, exposure=1.0),
    "night": dict(sun=(-50, 40), sun_color="#9fb4ff", sun_power=0.4, softness=0.35,
                  zenith="#070b1c", horizon="#1b2446", ground_color="#0c0e16", ambient=0.3,
                  fill=(80, 15), fill_color="#ffb070", fill_power=0.06, rim=(165, 30), rim_color="#8fb0ff",
                  rim_power=0.5, softboxes=[], env_floor="#0a0c14", sun_disk=False, exposure=1.2),
    "dramatic": dict(sun=(-85, 28), sun_color="#ffe2c0", sun_power=1.8, softness=0.35,
                     zenith="#30343c", horizon="#1c1e22", ground_color="#141414", ambient=0.08,
                     fill=(70, 10), fill_color="#6a80c0", fill_power=0.06, rim=(150, 30), rim_color="#cfe0ff",
                     rim_power=0.9, softboxes=[(-85, 28, 24, 30, 5.0)], env_floor="#141414", sun_disk=False,
                     exposure=1.0),
    # warm afternoon sun through a window on the left (-x) wall of a room(); angles are world angles
    "interior": dict(sun=(-100, 33), sun_color="#ffd6a8", sun_power=2.4, softness=0.3,
                     zenith="#e8e2e2", horizon="#f4d8bc", ground_color="#9a7a60", ambient=0.78,
                     fill=(45, 35), fill_color="#fff2e4", fill_power=0.12, rim=(170, 30), rim_power=0.0,
                     softboxes=[], env_floor="#6a5444", sun_disk=False, exposure=1.0, world=True),
}


class Lighting:
    """Light rig: a key 'sun' with soft shadows, a sky/environment that lights everything (with ambient
    occlusion), fill and rim lights, optional point lights. Angles are relative to the camera: azimuth 0 = light
    from behind the camera, -90 = from the left, 180 = backlight; elevation in degrees.

    ``Lighting("sunset")``, ``Lighting("studio", sun=(-60, 30), softness=0.8)``,
    ``lit.point((1, 2, 1), color="#ffb060", power=4)``. ``relative=False`` -> angles in world space
    (azimuth 0 = light from +z, 90 = from +x). The 'interior' preset uses world angles: sun through the
    left (-x) wall of a ``room()``."""

    def __init__(self, preset="studio", relative=None, **kw):
        if preset not in LIGHTING:
            raise _unknown_name("lighting preset", preset, LIGHTING, "Try solid.LIGHTING for the list")
        cfg = dict(LIGHTING[preset])
        world = cfg.pop("world", False)
        if relative is None:
            relative = not world
        cfg.setdefault("rim_color", "#ffffff")
        cfg.setdefault("fill_color", "#ffffff")
        for k, v in _reject_unknown("Lighting", kw, sorted(cfg)).items():
            cfg[k] = v
        self.preset, self.cfg, self.points = preset, cfg, []
        self.camera_relative = bool(relative)

    def point(self, position, color="#ffd9a0", power=3.0, radius=1.0, shadow=True):
        """Add a point light (world position). ``radius`` = distance where it falls to half strength."""
        self.points.append(dict(pos=_vec(position), color=_lin(color), power=float(power), radius=float(radius),
                                shadow=bool(shadow)))
        return self

    def set(self, **kw):
        for k, v in _reject_unknown("Lighting.set", kw, sorted(self.cfg)).items():
            self.cfg[k] = v
        return self


# ===================================================================================================== scene

_QUALITY = {
    "draft": dict(steps=110, shadow_steps=26, ao=3, aa=4, refl_steps=56, hit=0.6),
    "final": dict(steps=180, shadow_steps=44, ao=5, aa=8, refl_steps=90, hit=0.4),
}

_AA_PATTERNS = {
    4: [(-0.375, -0.125), (0.125, -0.375), (0.375, 0.125), (-0.125, 0.375)],
    8: [(-0.4375, -0.0625), (-0.3125, 0.3125), (-0.1875, -0.3125), (-0.0625, 0.0625), (0.0625, -0.4375),
        (0.1875, 0.4375), (0.3125, -0.1875), (0.4375, 0.1875)],
}


def _aa_offsets(n):
    if n in _AA_PATTERNS:
        return _AA_PATTERNS[n]
    k = int(math.ceil(math.sqrt(n)))
    g = (np.arange(k) + 0.5) / k - 0.5
    return [(a, b) for a in g for b in g][:max(n, 1)]


class _Ground:
    def __init__(self, kind="plane", mat=None, y=0.0, distance=4.0, radius=1.5, yaw=0.0):
        self.kind, self.mat, self.y = kind, mat, float(y)
        self.distance, self.radius, self.yaw = float(distance), float(radius), float(yaw)


class Scene:
    """A 3D scene: objects, a ground, a camera and a light rig. ``render(cv)`` paints it into a Canvas.

    ``ground``: a colour, ``"checker"``, a function ``f(u, v) -> colour`` (texture over x / z), ``"backdrop"``
    (seamless studio sweep) or None. ``lighting``: a preset name or a Lighting. ``camera``: a Camera."""

    def __init__(self, lighting="studio", ground="#d9d4cc", camera=None, background=None):
        self.models = []
        self.lighting = lighting if isinstance(lighting, Lighting) else Lighting(lighting)
        self.camera = camera or Camera()
        self.background = background
        self._ground = None
        self._frame = None
        if isinstance(ground, str) and ground == "backdrop":
            self.backdrop()
        elif isinstance(ground, str) and ground == "checker":
            self.floor(checker=True)
        elif ground is not None and ground is not False:
            self.floor(ground)

    # ---- building
    def add(self, obj, name=None, material=None):
        """Add a Shape or Model. Returns the Model. Each added object becomes its own layer ``name``."""
        m = _as_model(obj, name, material)
        if material is not None and isinstance(obj, Model):
            m = Model(m.name, [Part(p.name, p.shape, material) for p in m.parts])
        used = {mm.name for mm in self.models}
        base, i = m.name, 2
        while m.name in used:
            m = m.named(f"{base} {i}")
            i += 1
        self.models.append(m)
        return m

    def floor(self, color="#d9d4cc", checker=False, size=1.0, texture=None, roughness=0.7, reflect=0.0, y=0.0,
              material=None):
        """Infinite floor at height ``y``. ``color`` = a colour, a preset (``"wood"``, ``("wood", "#d6b08a")``,
        ``"marble"``) or a colour function; ``checker=True`` (or two colours) -> checkerboard of ``size``;
        ``texture=f(u, v)`` -> colour from world x / z; ``reflect`` = glossy floor reflection (0~1)."""
        if material is None and _is_preset(color):
            material = color
        if material is not None:
            mat = globals()["material"](material)
        else:
            if checker:
                a, b = checker if isinstance(checker, (tuple, list)) else (color if color != "#d9d4cc" else
                                                                         "#ece6da", "#c4bba9")
                col = _Checker(size, a, b)
            elif texture is not None or callable(color):
                tex = texture or color

                def col(P, N=None, _t=tex):
                    out = np.asarray(_t(P[:, 0], P[:, 2]), F32)
                    if out.ndim == 1:
                        out = np.broadcast_to(out, (len(P), 3))
                    return np.power(np.clip(out, 0, None), 2.2)
            else:
                col = color
            mat = Material(color=col, roughness=roughness, specular=0.3, reflect=reflect)
        self._ground = _Ground("plane", mat, y)
        return self

    def backdrop(self, color="#e9e5df", distance=3.0, radius=1.6, y=0.0, yaw=0.0, roughness=0.8, reflect=0.0):
        """Seamless studio backdrop: a floor that curves (``radius``) up into a wall ``distance`` behind the
        origin (wall faces +z; ``yaw`` turns it). The classic product-shot sweep. ``color`` may also be a
        preset such as ``("wood", "#d6b08a")``."""
        if _is_preset(color):
            mat = material(color)
        else:
            mat = Material(color=color, roughness=roughness, specular=0.2, reflect=reflect)
        self._ground = _Ground("backdrop", mat, y, distance, radius, yaw)
        return self

    def light(self, preset=None, **kw):
        """Switch the light preset and/or change options: ``sc.light("sunset", softness=0.5)``."""
        if preset is not None:
            pts = self.lighting.points
            self.lighting = Lighting(preset, **kw)
            self.lighting.points = pts
        else:
            self.lighting.set(**kw)
        return self.lighting

    def light_from(self, obj, color="#ffd9a0", power=1.5, radius=None, shadow=True):
        """Put a point light at the centre of a Part / Model / Shape (a lamp bulb, a window, a candle).
        ``radius`` (default: 6x the object's size) = distance where the light falls to half strength. A Model
        with a 'bulb', 'lamp', 'light' or 'flame' part (``floor_lamp()``, ``tent(lit=True)``) lights from it."""
        if isinstance(obj, Model) and not isinstance(obj, Part):
            names_ = [p.name for p in obj.parts]
            for k in ("bulb", "lamp", "light", "flame"):
                if k in names_:
                    obj = obj[k]
                    break
        sh = obj.shape if isinstance(obj, Part) else obj
        lo, hi = sh.bounds()
        c = (lo + hi) / 2
        r = radius if radius is not None else float(np.linalg.norm(hi - lo)) * 6
        self.lighting.point(c, color, power, r, shadow)
        return self

    def look(self, position, target=(0, 0.5, 0), fov=35.0):
        """Aim a perspective camera from ``position`` at ``target``."""
        self.camera = Camera(position, target, fov)
        return self.camera

    def project(self, p, cv=None, box=None):
        """World point(s) -> canvas pixel (x, y) for labels and 2D detail on top of the render."""
        frame = self._frame_for(cv, box) if (cv is not None or box is not None) else self._frame
        if frame is None:
            raise ValueError("render first or pass cv=")
        self.camera = self.camera._for_aspect(frame[2] / max(frame[3], 1))
        return self.camera.project(p, frame)

    @staticmethod
    def _frame_for(cv, box):
        if box is not None:
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            return (x0, y0, x1 - x0, y1 - y0)
        return (0, 0, cv.w, cv.h)

    def bounds(self):
        if not self.models:
            return np.array([-1.0, 0, -1]), np.array([1.0, 1, 1])
        los, his = zip(*(m.bounds() for m in self.models))
        return np.min(los, 0), np.max(his, 0)

    # ---- render
    def render(self, cv, log=None, aa=None, box=None, quality="auto", background=None):
        """Render into ``cv`` (one layer step per object) and return a RenderResult (depth, normal, ids, masks).

        ``aa`` = samples per silhouette pixel (default 4 draft / 8 final; 1 = off); ``box`` = (x0, y0, x1, y1)
        canvas region to render into; ``quality`` = 'draft' | 'final' | 'auto' (draft below 700 px tall);
        ``background`` = None (sky / environment), a colour, [(pos, colour), ...] vertical gradient, or False
        (keep what is already on the canvas -> composite 3D objects over a 2D painting)."""
        log = log or (lambda m: None)
        frame = self._frame_for(cv, box)
        self._frame = frame
        fx0, fy0, fw, fh = frame
        self.camera = self.camera._for_aspect(fw / max(fh, 1))
        _check_choice("Scene.render", "quality", quality, ("auto", "draft", "final"))
        if quality == "auto":
            quality = "draft" if fh < 700 else "final"
        q = dict(_QUALITY[quality])
        if aa is not None:
            q["aa"] = int(aa)
        bg = self.background if background is None else background
        ctx = _Ctx(self, q, fh)
        ctx.horizon_fade = bg is None
        # the part of the frame that lies on the canvas (a box may reach past the edges)
        x0, y0 = max(fx0, 0), max(fy0, 0)
        w, h = max(min(fx0 + fw, cv.w) - x0, 0), max(min(fy0 + fh, cv.h) - y0, 0)
        vis = (x0, y0, w, h)
        if w == 0 or h == 0:                               # box entirely off the canvas: nothing to draw
            x0, y0 = min(x0, cv.w), min(y0, cv.h)
            e3 = np.zeros((0, 0, 3), F32)
            return RenderResult(self, frame, (cv.h, cv.w), np.zeros((0, 0), F32), e3,
                                np.zeros((0, 0), np.int32), ctx, {}, e3, e3, vis=(x0, y0, 0, 0))

        # primary rays at pixel centres
        jj, ii = np.mgrid[0:h, 0:w]
        px = (ii.ravel() + x0 + 0.5).astype(np.float64)
        py = (jj.ravel() + y0 + 0.5).astype(np.float64)
        O, D = self.camera.rays(px, py, frame)
        res = ctx.render_rays(O, D)
        col = _tonemap(res["color"], ctx.exposure)
        pid = res["pid"].reshape(h, w)
        T = res["t"].reshape(h, w)
        layer_of_part = ctx.layer_of_part                  # part id -> layer index (0 bg, 1 ground, 2.. models)
        lay = layer_of_part[pid + 1]

        # edge pixels: different part than a neighbour, a depth jump, or a sharp crease (box edges, rims)
        edge = np.zeros((h, w), bool)
        Tf = np.where(np.isfinite(T), T, 1e6)
        Nm = res["normal"].reshape(h, w, 3)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if dx >= 0:
                sa = (slice(dy, h), slice(dx, w))
                sb = (slice(0, h - dy), slice(0, w - dx))
            else:
                sa = (slice(dy, h), slice(0, w + dx))
                sb = (slice(0, h - dy), slice(-dx, w))
            a, b, ta, tb = pid[sa], pid[sb], Tf[sa], Tf[sb]
            crease = np.einsum("ijk,ijk->ij", Nm[sa], Nm[sb]) < 0.8
            e = (a != b) | (np.abs(ta - tb) > 0.04 * np.minimum(ta, tb)) | (crease & (a > 0))
            if dx >= 0:
                edge[dy:h, dx:w] |= e
                edge[0:h - dy, 0:w - dx] |= e
            else:
                edge[dy:h, 0:w + dx] |= e
                edge[0:h - dy, -dx:w] |= e

        nlay = len(ctx.layer_names)
        offs = _aa_offsets(q["aa"]) if q["aa"] > 1 else []
        ey, ex = np.nonzero(edge) if offs else (np.zeros(0, int), np.zeros(0, int))
        E = len(ey)
        cov = None
        if E:
            K = len(offs)
            ox = np.array([o[0] for o in offs])
            oy = np.array([o[1] for o in offs])
            spx = (ex[:, None] + x0 + 0.5 + ox[None]).ravel()
            spy = (ey[:, None] + y0 + 0.5 + oy[None]).ravel()
            O2, D2 = self.camera.rays(spx, spy, frame)
            hit0 = ctx.q["hit"]
            ctx.q["hit"] = hit0 * q.get("aa_hit", 0.25)   # edge samples: tight hits, so grazing rays miss corners
            # a sample on the same surface as its pixel centre (same part, normal and depth) reuses its colour
            ci = np.repeat(ey * w + ex, K)
            ref = None if q.get("aa_shade_all") else \
                (res["pid"][ci], res["normal"][ci], res["t"][ci], res["color"][ci])
            r2 = ctx.render_rays(O2, D2, ref=ref)
            ctx.q["hit"] = hit0
            c2 = _tonemap(r2["color"], ctx.exposure)
            l2 = layer_of_part[r2["pid"] + 1]
            if bg is not None and bg is not False:
                skyc = self._bg_colors(bg, spy, frame)
                sky = r2["pid"] < 0
                c2[sky] = skyc[sky]
            key = np.repeat(np.arange(E), K) * nlay + l2
            cnt = np.bincount(key, minlength=E * nlay).reshape(E, nlay).astype(F32)
            sums = np.stack([np.bincount(key, weights=c2[:, c], minlength=E * nlay) for c in range(3)], -1)
            sums = sums.reshape(E, nlay, 3).astype(F32)
            cov = cnt / F32(K)
            ecol = sums / np.maximum(cnt, 1)[..., None]

        # background colours for the whole frame
        col = col.reshape(h, w, 3)
        if bg is not None and bg is not False:
            bgc = self._bg_colors(bg, py, frame).reshape(h, w, 3)
        else:
            Db = D.reshape(h, w, 3)
            bgc = _tonemap(ctx.backdrop_env(Db.reshape(-1, 3)), ctx.exposure).reshape(h, w, 3)
        sky = pid < 0
        col[sky] = bgc[sky]

        # ---- paint layers: background, ground, then objects far -> near (exact coverage compositing)
        H, W = cv.h, cv.w
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        order = [0, 1]
        depth_of = []
        for li in range(2, nlay):
            sel = lay == li
            depth_of.append((-(float(np.median(T[sel])) if sel.any() else 0.0), li))
        order += [li for _, li in sorted(depth_of)]
        covs = {}
        for li in order:
            c = (lay == li).astype(F32)
            if E:
                c[ey, ex] = cov[:, li]
            covs[li] = c
        cum = np.zeros((h, w), F32)
        if bg is not False:
            cum += covs[0]
        full_col = np.zeros((H, W, 3), F32)
        full_a = np.zeros((H, W), F32)
        for li in order:
            c = covs[li]
            if li == 0:
                if bg is False:
                    continue
                log("background")
                full_col[sl] = bgc
                full_a[:] = 0
                full_a[sl] = 1.0
                cv.paint(full_col, full_a)
                continue
            if not c.any():
                continue
            cum += c
            a = np.where(cum > 1e-6, c / np.maximum(cum, 1e-6), 0).astype(F32)
            lc = col.copy()
            if E:
                has = cov[:, li] > 0
                lc[ey[has], ex[has]] = ecol[has, li]
            log(ctx.layer_names[li])
            full_col[sl] = lc
            full_a[:] = 0
            full_a[sl] = a
            cv.paint(full_col, full_a)

        # ---- final composite colour (for the result) and buffers
        out_col = col.copy()
        if E:
            out_col[ey, ex] = (cov[..., None] * ecol).sum(1) + (bgc[ey, ex] * cov[:, 0:1] if bg is False else 0)
        result = RenderResult(self, frame, (H, W), T, res["normal"].reshape(h, w, 3), pid, ctx, covs, out_col,
                              res["pos"].reshape(h, w, 3), vis=vis)
        _manifest.note_render3d(cv, result)               # one manifest entry per named object
        self._note_lighting(cv, ctx, vis, (H, W), result)
        return result

    def _note_lighting(self, cv, ctx, frame, hw, result=None):
        """Record what the renderer itself drew for lighting words in a request (lighting, shadows,
        reflections), so the report does not ask for them as missing objects."""
        info = getattr(ctx, "lit_info", None)
        x0, y0, w, h = frame
        H, W = hw
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        L = self.lighting
        preset = getattr(L, "preset", None) or "studio"
        words = ["light", "lights", "lighting", "shading", f"{preset} lighting", f"{preset} light",
                 "soft lighting", "studio lighting", "soft studio lighting", "illumination"]
        try:
            full = np.zeros((H, W), F32)
            full[sl] = 1.0
            _manifest.note(cv, "cue", "lighting", mask=full, alias=words, backdrop=True, see_through=True)
            for m in self.models:                          # room windows and what they show
                if result is None or not any(p.name == "panes" for p in m.parts):
                    continue
                view = getattr(m["panes"].mat, "view_name", None)
                wm = result.mask(f"{m.name}.panes")
                if "frames" in [p.name for p in m.parts]:
                    wm = np.maximum(wm, result.mask(f"{m.name}.frames"))
                if float(wm.max()) > 0:
                    _manifest.note(cv, "object", "window", mask=wm, see_through=True,
                                   alias=["windows", "window pane", "window frame", "glass"])
                if isinstance(view, str) and view not in ("day",):
                    _manifest.note(cv, "effect", view, mask=result.mask(f"{m.name}.panes"), see_through=True,
                                   backdrop=True, pane_view=True, alias=[w for w in view.split() if w not in ("city",)] +
                                   ["view", "outside", "window view"])
            if info is None:
                return
            n, hit = info["n"], info["hit"]
            sun = np.ones(n, F32)
            sun[hit] = info["sun"]
            shade_m = np.zeros((H, W), F32)
            shade_m[sl] = np.clip((0.75 - sun.reshape(h, w)) / 0.5, 0, 1)
            if float(shade_m.mean()) > 2e-4:
                _manifest.note(cv, "cue", "shadows", mask=shade_m, backdrop=True, see_through=True,
                               alias=["shadow", "soft shadows", "cast shadows", "contact shadows", "gentle shadows",
                                      "drop shadow", "shade"])
            shiny = np.zeros(n, F32)
            shiny[hit] = info["shiny"]
            refl = np.zeros((H, W), F32)
            refl[sl] = shiny.reshape(h, w)
            if float(refl.mean()) > 2e-4:
                _manifest.note(cv, "cue", "reflections", mask=refl, backdrop=True, see_through=True,
                               alias=["reflection", "reflective", "gloss", "glossy", "shine", "highlights",
                                      "specular"])
        except Exception:                                   # noqa: BLE001 - the report must never break a render
            pass

    def _bg_colors(self, bg, py, frame):
        x0, y0, w, h = frame
        v = (np.asarray(py, np.float64) - y0) / max(h, 1)
        if isinstance(bg, (list, tuple)) and len(bg) and isinstance(bg[0], (list, tuple)) and len(bg[0]) == 2 \
                and not isinstance(bg[0][0], str):
            from .color import mix
            return np.asarray(mix(bg, v.astype(F32)), F32).reshape(-1, 3)
        return np.broadcast_to(rgb(bg), (len(v), 3)).astype(F32)


def _silhouette(models, cam, aspect, h=72):
    """World points where a small id pass through ``cam`` hits the models (for auto-framing)."""
    sc = Scene(ground=None, camera=cam)
    sc.models = list(models)
    w = max(8, int(round(h * aspect)))
    ctx = _Ctx(sc, _QUALITY["draft"], h)
    jj, ii = np.mgrid[0:h, 0:w]
    O, D = cam.rays(ii.ravel() + 0.5, jj.ravel() + 0.5, (0, 0, w, h))
    n = len(O)
    T, pid = ctx.march(O, D, np.zeros(n, F32), np.full(n, F32(1e9)), np.zeros(n, F32), 90)
    k = pid > 0
    return (O[k] + D[k] * T[k, None]).astype(np.float64)


def _tonemap(lin, exposure=1.0):
    """Linear light -> display: gamma 2.2 with a soft highlight shoulder (keeps colours as specified up to ~0.8)."""
    x = np.power(np.clip(lin * F32(exposure), 0, None), F32(1 / 2.2))
    k = F32(0.8)
    over = x > k
    if over.any():
        xs = x[over]
        x[over] = k + (1 - k) * (1 - np.exp(-(xs - k) / (1 - k)))
    return x.astype(F32)


# ===================================================================================================== render context

class _Tree:
    """Two-level bounding-volume tree over parts: one subtree per model, then a tree over the models. A query
    walks it with a whole batch of points at once, so each point only evaluates the parts its cap can reach."""

    def __init__(self, shapes, groups):
        self.shapes = shapes
        self.lo, self.hi, self.kids, self.leaf, self.lam, self.mu = [], [], [], [], [], []
        roots = [self._build([self._node(i) for i in g]) for g in groups if len(g)]
        self.model_roots = set(roots)
        self.root = self._build(roots) if roots else None

    def _node(self, pid=-1, kids=()):
        if pid >= 0:
            lo, hi = self.shapes[pid].bounds()
            lam, mu = _lowb(self.shapes[pid])
        else:
            lo = np.min([self.lo[k] for k in kids], 0)
            hi = np.max([self.hi[k] for k in kids], 0)
            lam = min(self.lam[k] for k in kids)
            mu = max(self.mu[k] for k in kids)
        self.lam.append(F32(lam))
        self.mu.append(F32(mu))
        self.lo.append(np.asarray(lo, np.float64))
        self.hi.append(np.asarray(hi, np.float64))
        self.kids.append(tuple(kids))
        self.leaf.append(pid)
        return len(self.leaf) - 1

    def _build(self, nodes):
        if len(nodes) == 1:
            return nodes[0]
        c = np.array([np.clip((self.lo[i] + self.hi[i]) / 2, -1e6, 1e6) for i in nodes])
        ax = int(np.argmax(c.max(0) - c.min(0)))
        order = np.argsort(c[:, ax], kind="stable")
        half = len(nodes) // 2
        a = self._build([nodes[i] for i in order[:half]])
        b = self._build([nodes[i] for i in order[half:]])
        return self._node(kids=(a, b))

    def span(self, O, iD, t0, t1, margin):
        """Per ray, the first entry into and the last exit from the leaf boxes (grown by ``margin``) that the
        segment t0..t1 crosses: (start, end), start = inf where it crosses none."""
        n = len(O)
        start = np.full(n, np.inf, F32)
        end = np.zeros(n, F32)
        if self.root is None:
            return start, end
        stack = [(self.root, np.arange(n))]
        while stack:
            nd, sel = stack.pop()
            Os, iDs = O[sel], iD[sel]
            lo, hi = self.lo[nd] - margin, self.hi[nd] + margin
            tn = None
            for a in range(3):
                u1 = (F32(lo[a]) - Os[:, a]) * iDs[:, a]
                u2 = (F32(hi[a]) - Os[:, a]) * iDs[:, a]
                if tn is None:
                    tn, tf = np.minimum(u1, u2), np.maximum(u1, u2)
                else:
                    np.maximum(tn, np.minimum(u1, u2), out=tn)
                    np.minimum(tf, np.maximum(u1, u2), out=tf)
            a0, a1 = t0[sel], t1[sel]
            tn = np.maximum(tn, a0)
            tf = np.minimum(tf, a1)
            ok = tf >= tn
            if not ok.any():
                continue
            s = sel[ok]
            if self.leaf[nd] >= 0 or nd in self.model_roots:      # model boxes are tight enough here
                start[s] = np.minimum(start[s], tn[ok])
                end[s] = np.maximum(end[s], tf[ok])
                continue
            for c in self.kids[nd]:
                stack.append((c, s))
        return start, end

    def walk(self, x, y, z, cap, leaf_fn, lb_out=None):
        """Visit, for each point, every leaf whose box is closer than ``cap`` (which ``leaf_fn(pid, sel)`` may
        lower as it goes). ``lb_out`` collects the box distance of the nodes skipped (a safe step)."""
        if self.root is None:
            return
        if not hasattr(self, "_b32"):
            self._b32 = [tuple(F32(v) for v in (*l, *h)) for l, h in zip(self.lo, self.hi)]
        stack = [(self.root, None)]
        lo, hi, kids, leaf, lam, mu, b32 = self.lo, self.hi, self.kids, self.leaf, self.lam, self.mu, self._b32
        while stack:
            nd, sel = stack.pop()
            if sel is None:
                xs, ys, zs, cs = x, y, z, cap
            else:
                xs, ys, zs, cs = x[sel], y[sel], z[sel], cap[sel]
            l0, l1, l2, h0, h1, h2 = b32[nd]
            dx = np.maximum(l0 - xs, xs - h0)
            np.maximum(dx, 0, out=dx)
            dy = np.maximum(l1 - ys, ys - h1)
            np.maximum(dy, 0, out=dy)
            dz = np.maximum(l2 - zs, zs - h2)
            np.maximum(dz, 0, out=dz)
            dx *= dx
            dy *= dy
            dz *= dz
            dx += dy
            dx += dz                                     # squared box distance
            lm, mm = lam[nd], mu[nd]
            if lm < 1 or mm > 0:                         # lam * lb - mu < cap  <=>  lb < (cap + mu) / lam
                thr = (cs + mm) * (1 / lm)
                near = (dx < thr * thr) & (thr > 0)
            else:
                near = (dx < cs * cs) & (cs > 0)
            if lb_out is not None:
                far = ~near
                if far.any():
                    lbf = np.sqrt(dx[far])
                    if lm < 1 or mm > 0:
                        lbf = lbf * lm - mm
                    fi = np.flatnonzero(far) if sel is None else sel[far]
                    lb_out[fi] = np.minimum(lb_out[fi], lbf)
            s = np.flatnonzero(near) if sel is None else sel[near]
            if not len(s):
                continue
            if leaf[nd] >= 0:
                leaf_fn(leaf[nd], s)
                continue
            a, b = kids[nd]
            # nearer child last, so it is evaluated first and tightens the caps for the other one
            i0, i1 = s[0], s[-1]
            m = np.array([x[i0] + x[i1], y[i0] + y[i1], z[i0] + z[i1]], np.float64) * 0.5
            da = np.linalg.norm(np.maximum(np.maximum(lo[a] - m, m - hi[a]), 0))
            db = np.linalg.norm(np.maximum(np.maximum(lo[b] - m, m - hi[b]), 0))
            if da <= db:
                stack.append((b, s))
                stack.append((a, s))
            else:
                stack.append((a, s))
                stack.append((b, s))


class _Ctx:
    """Everything the ray marcher needs for one render: flattened parts, boxes, lights, quality."""

    def __init__(self, scene, q, h):
        self.scene, self.q = scene, q
        self.horizon_fade = False
        self.parts, self.mats, self.names, self.part_layer = [], [], [], []
        self.layer_names = ["background", "ground"]
        self.model_parts = []
        for mi, m in enumerate(scene.models):
            idx = []
            for p in m.parts:
                self.parts.append(p)
                self.names.append(m.name if (len(m.parts) == 1 and p.name == m.name) else f"{m.name}.{p.name}")
                self.part_layer.append(2 + mi)
                idx.append(len(self.parts))          # part ids start at 1 (0 = ground)
            self.model_parts.append(idx)
            self.layer_names.append(m.name)
        n = len(self.parts)
        self.shapes = [None] + [p.shape for p in self.parts]
        self.lo = np.zeros((n + 1, 3))
        self.hi = np.zeros((n + 1, 3))
        for i, p in enumerate(self.parts, 1):
            self.lo[i], self.hi[i] = p.shape.bounds()
        self.layer_of_part = np.array([0, 1] + self.part_layer, np.int32)   # index = pid + 1
        slo, shi = scene.bounds()
        self.extent = float(np.linalg.norm(shi - slo)) if n else 4.0
        self.scale = max(self.extent, 1e-3)
        self.emin = 2e-5 * self.scale
        self.margin = 1e-3 * self.scale
        self.opacity = np.array([1.0] + [1.0 - p.mat.glass * 0.85 if p.mat.shadow is True else
                                         (0.0 if p.mat.shadow is False else float(p.mat.shadow))
                                         for p in self.parts], F32)
        cam = scene.camera
        self.cam = cam
        self.pix = cam._scale(h)
        self.ortho = bool(cam.ortho)
        self.ground = scene._ground
        # lights (camera-relative angles -> world directions)
        L = scene.lighting
        c = L.cfg
        f, r, u = cam.basis()
        yaw = math.degrees(math.atan2(-f[0], -f[2]))       # camera azimuth (from +z)

        def wdir(az, el):
            return _norm(_dir(az + yaw, el)) if L.camera_relative else _norm(_dir(az, el))
        self.sun = wdir(*c["sun"]).astype(F32)
        self.sun_col = _lin(c["sun_color"]) * F32(c["sun_power"])
        self.k_soft = float(np.interp(c["softness"], [0, 0.5, 1], [40, 9, 3.2]))
        self.fill = wdir(*c["fill"]).astype(F32)
        self.fill_col = _lin(c["fill_color"]) * F32(c["fill_power"])
        self.rim = wdir(*c["rim"]).astype(F32)
        self.rim_col = _lin(c["rim_color"]) * F32(c["rim_power"])
        self.zen, self.hor, self.gnd = _lin(c["zenith"]), _lin(c["horizon"]), _lin(c["ground_color"])
        self.env_floor = _lin(c["env_floor"])
        self.amb = F32(c["ambient"])
        self.sun_disk = c["sun_disk"]
        self.softboxes = []
        for (az, el, wd, ht, pw) in c["softboxes"]:
            d = wdir(az, el)
            rr = _norm(np.cross([0, 1, 0], d)) if abs(d[1]) < 0.99 else np.array([1.0, 0, 0])
            uu = np.cross(d, rr)
            self.softboxes.append((d.astype(F32), rr.astype(F32), uu.astype(F32),
                                   math.tan(math.radians(wd / 2)), math.tan(math.radians(ht / 2)), F32(pw)))
        self.points = L.points
        self.exposure = float(c["exposure"])
        self.shadow_margin = self.extent / self.k_soft * 0.6
        self._tree = None

    @property
    def tree(self):
        """Bounding-volume tree over the parts that cast shadows (built on first use)."""
        if self._tree is None:
            self._tree = _Tree(self.shapes, [[i for i in idx if self.opacity[i] > 0] for idx in self.model_parts])
        return self._tree

    # ---------------------------------------------------------------- environment
    def env(self, D):
        y = D[:, 1:2]
        t = np.clip(y, 0, 1) ** F32(0.55)
        sky = self.hor[None] * (1 - t) + self.zen[None] * t
        below = np.clip(-y * 4, 0, 1)
        out = sky * (1 - below) + self.env_floor[None] * below
        if self.sun_disk:
            cs = np.clip(D @ self.sun, 0, 1)
            out = out + self.sun_col[None] * (cs ** 900 * 6 + cs ** 24 * 0.08)[:, None]
        for d, rr, uu, tw, th, pw in self.softboxes:
            wv = D @ d
            u = np.abs(D @ rr) / np.maximum(wv, 1e-3)
            v = np.abs(D @ uu) / np.maximum(wv, 1e-3)
            m = (np.clip((tw - u) / (tw * 0.6), 0, 1) * np.clip((th - v) / (th * 0.6), 0, 1) * (wv > 0))
            m = m * m * (3 - 2 * m)
            out = out + (m * pw)[:, None] * F32(1.0)
        return out.astype(F32)

    def backdrop_env(self, D):
        """What the camera sees where it misses everything. With no ground, looking down shows the horizon
        colour (a clean backdrop for isometric views) instead of the dark world below."""
        if self.ground is not None:
            return self.env(D)
        D = D.copy()
        np.maximum(D[:, 1], 0, out=D[:, 1])
        return self.env(D)

    def irradiance(self, N):
        t = N[:, 1:2] * 0.5 + 0.5
        side = 1 - np.abs(N[:, 1:2])
        base = self.gnd[None] * (1 - t) + self.zen[None] * t
        return (base * (1 - 0.4 * side) + self.hor[None] * (0.4 * side)) * self.amb

    # ---------------------------------------------------------------- ground
    def ground_hit(self, O, D):
        n = len(O)
        t = np.full(n, np.inf, F32)
        N = np.zeros((n, 3), F32)
        g = self.ground
        if g is None:
            return t, N
        if g.kind == "plane":
            dy = D[:, 1]
            ok = dy < -1e-6
            tt = (F32(g.y) - O[:, 1]) / np.where(ok, dy, -1)
            ok &= tt > 0
            t[ok] = tt[ok]
            N[:, 1] = 1
            return t, N
        # backdrop in its local frame (wall faces +z)
        a = math.radians(g.yaw)
        c, s = F32(math.cos(a)), F32(math.sin(a))
        oz = -s * O[:, 0] + c * O[:, 2]
        dz = -s * D[:, 0] + c * D[:, 2]
        oy, dy = O[:, 1] - F32(g.y), D[:, 1]
        R, zw = F32(g.radius), F32(-g.distance)
        zc, yc = zw + R, R
        # floor
        okf = dy < -1e-6
        tf = np.where(okf, -oy / np.where(okf, dy, -1), np.inf)
        okf &= (tf > 0) & (oz + dz * tf >= zc)
        tf = np.where(okf, tf, np.inf)
        # wall
        okw = dz < -1e-6
        tw = np.where(okw, (zw - oz) / np.where(okw, dz, -1), np.inf)
        okw &= (tw > 0) & (oy + dy * tw >= yc)
        tw = np.where(okw, tw, np.inf)
        # cylinder (far root)
        py, pz = oy - yc, oz - zc
        A = dy * dy + dz * dz
        B = 2 * (py * dy + pz * dz)
        C = py * py + pz * pz - R * R
        disc = B * B - 4 * A * C
        okc = (disc >= 0) & (A > 1e-9)
        sq = np.sqrt(np.maximum(disc, 0))
        tc = (-B + sq) / np.maximum(2 * A, 1e-9)
        hy, hz = py + dy * tc, pz + dz * tc
        okc &= (tc > 0) & (hy <= 0) & (hz <= 0)
        tc = np.where(okc, tc, np.inf)
        tt = np.minimum(np.minimum(tf, tw), tc)
        t[:] = tt
        which = np.where(tt == tf, 0, np.where(tt == tw, 1, 2))
        ny = np.where(which == 0, 1, np.where(which == 1, 0, -hy / R))
        nz = np.where(which == 0, 0, np.where(which == 1, 1, -hz / R))
        N[:, 0] = -s * nz
        N[:, 1] = ny
        N[:, 2] = c * nz
        return t, N

    def ground_sdf(self, x, y, z):
        g = self.ground
        if g is None:
            return np.full(x.shape, np.inf, F32)
        if g.kind == "plane":
            return y - F32(g.y)
        a = math.radians(g.yaw)
        c, s = F32(math.cos(a)), F32(math.sin(a))
        lz = -s * x + c * z
        ly = y - F32(g.y)
        R, zw = F32(g.radius), F32(-g.distance)
        zc = zw + R
        d_floor = ly
        d_wall = lz - zw
        d_curve = R - np.sqrt((lz - zc) ** 2 + (ly - R) ** 2)
        return np.where(lz >= zc, d_floor, np.where(ly >= R, d_wall, d_curve))

    # ---------------------------------------------------------------- marching
    def _slab(self, O, iD, lo, hi):
        t1 = (F32(lo[0]) - O[:, 0]) * iD[:, 0]
        t2 = (F32(hi[0]) - O[:, 0]) * iD[:, 0]
        tn, tf = np.minimum(t1, t2), np.maximum(t1, t2)
        for a in (1, 2):
            t1 = (F32(lo[a]) - O[:, a]) * iD[:, a]
            t2 = (F32(hi[a]) - O[:, a]) * iD[:, a]
            np.maximum(tn, np.minimum(t1, t2), out=tn)
            np.minimum(tf, np.maximum(t1, t2), out=tf)
        return tn, tf

    def _lists(self, O, D, t0, t1, margin, exclude=None, skip_unshadowed=False):
        Ds = np.where(np.abs(D) < 1e-9, F32(1e-9), D)
        iD = (1.0 / Ds).astype(F32)
        n = len(O)
        lists = []
        Tstart = np.full(n, np.inf, F32)
        Tend = np.zeros(n, F32)
        for idx in self.model_parts:
            if skip_unshadowed:
                idx = [i for i in idx if self.opacity[i] > 0]
                if not idx:
                    continue
            lo = self.lo[idx].min(0) - margin
            hi = self.hi[idx].max(0) + margin
            tn, tf = self._slab(O, iD, lo, hi)
            sel = np.flatnonzero((tf >= np.maximum(tn, t0)) & (tn <= t1))
            if not len(sel):
                continue
            Os, iDs, a0, a1 = O[sel], iD[sel], t0[sel], t1[sel]
            for pi in idx:
                if len(idx) > 1:
                    tn2, tf2 = self._slab(Os, iDs, self.lo[pi] - margin, self.hi[pi] + margin)
                else:
                    tn2, tf2 = tn[sel], tf[sel]
                tn2 = np.maximum(tn2, a0)
                tf2 = np.minimum(tf2, a1)
                ok = tf2 >= tn2
                if exclude is not None:
                    ok &= ~(exclude[sel] == pi).any(1)
                k = np.flatnonzero(ok)
                if not len(k):
                    continue
                r = sel[k]
                lists.append([pi, r, tn2[k], tf2[k]])
                Tstart[r] = np.minimum(Tstart[r], tn2[k])
                Tend[r] = np.maximum(Tend[r], tf2[k])
        return lists, Tstart, Tend

    def march(self, O, D, t0, t1, foot0, steps, exclude=None):
        """Sphere-trace rays against all parts. Returns (t, part id or -1)."""
        n = len(O)
        foot1 = F32(0 if self.ortho else self.pix)
        hitk = F32(self.q["hit"])
        lists, Tstart, Tend = self._lists(O, D, t0, t1, self.margin, exclude)
        T = np.where(np.isfinite(Tstart), Tstart, t1).astype(F32)
        alive = np.isfinite(Tstart)
        pid = np.full(n, -1, np.int32)
        Dist = np.zeros(n, F32)
        Who = np.full(n, -1, np.int32)
        emin = F32(self.emin)
        Ox, Oy, Oz = O[:, 0], O[:, 1], O[:, 2]
        Dx, Dy, Dz = D[:, 0], D[:, 1], D[:, 2]
        act = np.flatnonzero(alive)
        # UB: an upper bound on the distance to the nearest surface (the one found last step is at most
        # d + step away). Parts are evaluated with cap min(Dist, UB): anything farther cannot be the nearest.
        UB = np.full(n, np.inf, F32)
        Wcap = np.full(n, np.inf, F32)
        for it in range(steps):
            if not len(act):
                break
            Dist[act] = Tend[act] - T[act] + emin
            Who[act] = -1
            for L in lists:
                pi, r, tn, tf = L
                if not len(r):
                    continue
                t = T[r]
                before = t < tn
                if before.any():
                    rb = r[before]
                    v = tn[before] - t[before] + emin
                    b = v < Dist[rb]
                    if b.any():
                        rb = rb[b]
                        Dist[rb] = v[b]
                        Who[rb] = -1
                    ins = ~before & (t <= tf)
                else:
                    ins = t <= tf
                ri = r[ins]
                if not len(ri):
                    continue
                tt = t[ins]
                cur = Dist[ri]
                cap = np.minimum(cur, UB[ri])
                d = self.shapes[pi]._dc(Ox[ri] + Dx[ri] * tt, Oy[ri] + Dy[ri] * tt, Oz[ri] + Dz[ri] * tt, cap)
                b = d < cur
                if b.any():
                    rr = ri[b]
                    db, cb = d[b], cap[b]
                    Dist[rr] = db
                    # a value at the cap is only a lower bound (a part farther than the surface found last step,
                    # possibly rounded a hair below the cap by a scale): it may shorten the step, never name the hit
                    Who[rr] = np.where(db < cb * F32(0.999), pi, -1)
                    Wcap[rr] = cb
            d = Dist[act]
            t = T[act]
            eps = (foot0[act] + foot1 * t) * hitk + emin
            w = Who[act]
            hit = (w >= 0) & (d < eps)
            if hit.any():
                pid[act[hit]] = w[hit]
            step = np.maximum(d, emin)
            UB[act] = np.where((w >= 0) & (d < Wcap[act]), d + step + emin, np.inf)
            T[act] = t + step
            done = hit | (T[act] > Tend[act])
            if done.any():
                alive[act[done]] = False
                act = act[~done]
            if it % 4 == 3 and len(act):
                for L in lists:
                    if len(L[1]):
                        keep = alive[L[1]] & (T[L[1]] <= L[3])
                        if not keep.all():
                            L[1], L[2], L[3] = L[1][keep], L[2][keep], L[3][keep]
        if len(act):                                     # out of steps: accept near misses
            d = Dist[act]
            w = Who[act]
            ok = (w >= 0) & (d < (foot0[act] + foot1 * T[act]) * 4 + emin * 50)
            pid[act[ok]] = w[ok]
        return T, pid

    def shadow(self, P, L, tmax, steps, k=None):
        """Soft shadow factor (1 lit, 0 shadowed) for points P toward direction(s) L."""
        if _REF:
            return self._shadow_ref(P, L, tmax, steps, k)
        # the part tree with penumbra caps gives the same shadows as marching every part (tests compare it with
        # _reference); per-part ray lists are kept as an option for very sparse scenes
        mode = self.q.get("shadow_mode", "walk")
        if mode == "auto":
            mode = "walk" if self._crowded(P, L, tmax, k) else "lists"
        if mode == "walk":
            return self._shadow_walk(P, L, tmax, steps, k)
        return self._shadow_lists(P, L, tmax, steps, k)

    def _pen_margin(self, tmax, k=None):
        """How far from a part a shadow ray can still be darkened by it: the penumbra test ``k * d / t < 1``
        reaches ``t / k`` at distance t, so parts are looked at within ``tmax / k`` of the ray."""
        tm = float(np.max(tmax)) if np.size(tmax) else 0.0
        return max(self.shadow_margin, tm / float(k or self.k_soft))

    def _crowded(self, P, L, tmax, k=None):
        """True when shadow rays pass within the penumbra margin of many parts each (characters in a crowd,
        a wide scene with a large margin): then the part tree, which only visits parts near each ray, wins over
        per-part ray lists. Measured on a sample of the rays."""
        n = len(P)
        if n < 4096:
            return True
        s = slice(0, n, max(1, n // 2048))
        Ps = np.ascontiguousarray(P[s], F32)
        D = np.ascontiguousarray(np.broadcast_to(np.asarray(L, F32), P.shape)[s], F32)
        m = len(Ps)
        t0 = np.full(m, F32(self.scale * 2e-3), F32)
        t1 = np.broadcast_to(np.asarray(tmax, F32), (n,))[s].astype(F32)
        lists, _, _ = self._lists(Ps, D, t0, t1, self._pen_margin(tmax, k), skip_unshadowed=True)
        work = sum(len(e[1]) * self.shapes[e[0]].cost for e in lists)     # part evaluations per step
        return work > 40 * m

    def _shadow_lists(self, P, L, tmax, steps, k=None):
        """Soft shadows with per-part ray lists. Each part is evaluated with a cap: below it the distance is
        exact, above it the part can neither darken the penumbra (``res * t / k``) nor shorten the step, so the
        result is the same as evaluating every part in full."""
        n = len(P)
        k = F32(k or self.k_soft)
        if n == 0:
            return np.ones(0, F32)
        D = np.broadcast_to(np.asarray(L, F32), P.shape).astype(F32)
        t0 = np.full(n, F32(self.scale * 2e-3), F32)
        t1 = np.broadcast_to(np.asarray(tmax, F32), (n,)).astype(F32)
        lists, Tstart, Tend = self._lists(P, D, t0, t1, self._pen_margin(tmax, k), skip_unshadowed=True)
        res = np.ones(n, F32)
        T = np.where(np.isfinite(Tstart), Tstart, t1).astype(F32)
        alive = np.isfinite(Tstart)
        Dist = np.zeros(n, F32)
        smin, smax = F32(self.scale * 1.5e-3), F32(self.scale * 0.06)
        emin = F32(self.emin)
        Px, Py, Pz = P[:, 0], P[:, 1], P[:, 2]
        Dx, Dy, Dz = D[:, 0], D[:, 1], D[:, 2]
        act = np.flatnonzero(alive)
        smin0 = smin
        shapes, opacity = self.shapes, self.opacity
        for it in range(steps * 4):
            if not len(act):
                break
            if it >= steps:                 # the few rays still going: take bigger strides so they finish
                smin = F32(smin0 * (1 + (it - steps) * 0.5))
            Dist[act] = Tend[act] - T[act] + emin
            for L_ in lists:
                pi, r, tn, tf = L_
                if not len(r):
                    continue
                t = T[r]
                before = t < tn
                rb = r[before]
                if len(rb):
                    Dist[rb] = np.minimum(Dist[rb], tn[before] - t[before] + emin)
                ins = ~before & (t <= tf)
                ri = r[ins]
                if not len(ri):
                    continue
                tt = t[ins]
                tk = np.maximum(tt, emin) / k
                cap = np.maximum(res[ri] * tk, np.minimum(Dist[ri], smax))
                d = shapes[pi]._dc(Px[ri] + Dx[ri] * tt, Py[ri] + Dy[ri] * tt, Pz[ri] + Dz[ri] * tt, cap)
                op = opacity[pi]
                v = np.clip(k * d / np.maximum(tt, emin), 0, 1)
                if op < 1:
                    v = 1 - (1 - v) * op
                    res[ri] = np.minimum(res[ri], v)
                    Dist[ri] = np.minimum(Dist[ri], np.clip(np.abs(d), smin, smax))
                else:
                    res[ri] = np.minimum(res[ri], v)
                    Dist[ri] = np.minimum(Dist[ri], np.clip(d, smin, smax))
            T[act] = T[act] + Dist[act]
            done = (res[act] < 0.004) | (T[act] > Tend[act])
            if done.any():
                alive[act[done]] = False
                act = act[~done]
            if it % 4 == 3 and len(act):
                for L_ in lists:
                    if len(L_[1]):
                        keep = alive[L_[1]] & (T[L_[1]] <= L_[3])
                        if not keep.all():
                            L_[1], L_[2], L_[3] = L_[1][keep], L_[2][keep], L_[3][keep]
        return res * res * (3 - 2 * res)

    def _shadow_walk(self, P, L, tmax, steps, k=None):
        """Soft shadows that walk the part tree every step (best for few rays: no lists to build)."""
        n = len(P)
        k = F32(k or self.k_soft)
        tree = self.tree
        if n == 0 or tree.root is None:
            return np.ones(n, F32)
        P = np.ascontiguousarray(P, F32)
        D = np.ascontiguousarray(np.broadcast_to(np.asarray(L, F32), P.shape), F32)
        t0 = np.full(n, F32(self.scale * 2e-3), F32)
        t1 = np.broadcast_to(np.asarray(tmax, F32), (n,)).astype(F32)
        Ds = np.where(np.abs(D) < 1e-9, F32(1e-9), D)
        # rays that never come near a part's penumbra are lit; the others march from their start, so they take the
        # same samples as marching every part (a part is skipped only while it can neither darken the penumbra
        # nor shorten the step, see the cap below)
        T, Tend = tree.span(P, (1.0 / Ds).astype(F32), t0, t1, self._pen_margin(tmax, k))
        T = np.where(np.isfinite(T), t0, T).astype(F32)
        res = np.ones(n, F32)
        smin, smax = F32(self.scale * 1.5e-3), F32(self.scale * 0.06)
        emin = F32(self.emin)
        # parts farther than the stride (and than the penumbra reach) are skipped: the same penumbra as marching
        # every part, sampled at different steps (shadow_stride=1e6 reproduces the reference steps exactly)
        stride = F32(self.q.get("shadow_stride", 0.3))
        dark = F32(self.q.get("shadow_dark", 0.004))
        Px, Py, Pz = P[:, 0], P[:, 1], P[:, 2]
        Dx, Dy, Dz = D[:, 0], D[:, 1], D[:, 2]
        act = np.flatnonzero(Tend >= T)
        shapes, opacity = self.shapes, self.opacity
        smin0 = smin
        for it in range(steps * 4):
            if not len(act):
                break
            if it >= steps:
                smin = F32(smin0 * (1 + (it - steps) * 0.5))
            t = T[act]
            x, y, z = Px[act] + Dx[act] * t, Py[act] + Dy[act] * t, Pz[act] + Dz[act] * t
            r = res[act]
            tk = np.maximum(t, emin) / k
            want = np.clip(t * stride, smin, smax)
            dist = np.full(len(act), np.inf, F32)
            cap = np.maximum(r * tk, want)
            lbs = np.full(len(act), np.inf, F32)

            def leaf(pi, s):
                d = shapes[pi]._dc(x[s], y[s], z[s], cap[s])
                op = opacity[pi]
                v = np.clip(d / tk[s], 0, 1)
                if op < 1:
                    v = 1 - (1 - v) * op
                    d = np.abs(d)
                rs = np.minimum(r[s], v)
                r[s] = rs
                ds = np.minimum(dist[s], np.clip(d, smin, smax))
                dist[s] = ds
                cap[s] = np.maximum(rs * tk[s], np.minimum(want[s], ds))
            tree.walk(x, y, z, cap, leaf, lbs)
            res[act] = r
            T[act] = t + np.maximum(np.minimum(np.minimum(dist, lbs), smax), smin)
            done = (r < dark) | (T[act] > Tend[act])
            if done.any():
                act = act[~done]
        return res * res * (3 - 2 * res)

    def _shadow_ref(self, P, L, tmax, steps, k=None):
        """Reference soft shadows: march every shadow ray from its start, evaluating every part in full."""
        n = len(P)
        k = F32(k or self.k_soft)
        if n == 0:
            return np.ones(0, F32)
        P = np.ascontiguousarray(P, F32)
        D = np.ascontiguousarray(np.broadcast_to(np.asarray(L, F32), P.shape), F32)
        T = np.full(n, F32(self.scale * 2e-3), F32)
        t1 = np.broadcast_to(np.asarray(tmax, F32), (n,)).astype(F32)
        res = np.ones(n, F32)
        smin0, smax = F32(self.scale * 1.5e-3), F32(self.scale * 0.06)
        emin = F32(self.emin)
        casters = [i for i in range(1, len(self.shapes)) if self.opacity[i] > 0]
        act = np.arange(n)
        for it in range(steps * 4):
            if not len(act):
                break
            smin = F32(smin0 * (1 + max(it - steps, 0) * 0.5))
            t = T[act]
            x, y, z = P[act, 0] + D[act, 0] * t, P[act, 1] + D[act, 1] * t, P[act, 2] + D[act, 2] * t
            r = res[act]
            dist = np.full(len(act), np.inf, F32)
            for pi in casters:
                d = self.shapes[pi]._d(x.copy(), y.copy(), z.copy())
                v = np.clip(k * d / np.maximum(t, emin), 0, 1)
                op = self.opacity[pi]
                if op < 1:
                    v = 1 - (1 - v) * op
                    d = np.abs(d)
                r = np.minimum(r, v)
                dist = np.minimum(dist, np.clip(d, smin, smax))
            res[act] = r
            T[act] = t + dist
            done = (r < 0.004) | (T[act] > t1[act])
            act = act[~done]
        return res * res * (3 - 2 * res)

    def scene_dist(self, x, y, z, cap):
        """Distance to the nearest surface (objects + ground), clamped at ``cap`` (for ambient occlusion)."""
        d = np.minimum(self.ground_sdf(x, y, z), F32(cap)).astype(F32)
        shapes = self.shapes
        if _REF:
            for pi in range(1, len(shapes)):
                if self.opacity[pi] > 0:                 # the same parts the tree holds
                    d = np.minimum(d, shapes[pi]._d(x.copy(), y.copy(), z.copy()))
            return d

        def leaf(pi, s):
            d[s] = np.minimum(d[s], shapes[pi]._dc(x[s], y[s], z[s], d[s]))
        self.tree.walk(x, y, z, d, leaf)
        return d

    def normals(self, P, pid, foot):
        N = np.zeros_like(P)
        for pi in np.unique(pid):
            if pi <= 0:
                continue
            idx = np.flatnonzero(pid == pi)
            p = P[idx]
            e = np.maximum(foot[idx] * 0.5, F32(self.scale * 2e-5))
            sh = self.shapes[pi]
            acc = np.zeros((len(idx), 3), F32)
            for kx, ky, kz in ((1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, 1, 1)):
                d = sh._d(p[:, 0] + kx * e, p[:, 1] + ky * e, p[:, 2] + kz * e)
                acc[:, 0] += kx * d
                acc[:, 1] += ky * d
                acc[:, 2] += kz * d
            N[idx] = acc / np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), F32(1e-12))
        return N

    def ao(self, P, N, n):
        if n <= 0 or not len(P):
            return np.ones(len(P), F32)
        # contact-sized occlusion: proportional for small scenes, capped for rooms so thin chair legs do not
        # cast dark halos on the floor
        rad = min(self.extent * 0.05, 0.08 + self.extent * 0.01)
        occ = np.zeros(len(P), F32)
        wsum = 0.0
        for i in range(n):
            hh = rad * (0.08 + 0.92 * ((i + 1) / n) ** 1.6)
            q = P + N * F32(hh)
            d = self.scene_dist(q[:, 0], q[:, 1], q[:, 2], hh)
            w = 1.0 / (1 + i)
            occ += F32(w) * np.clip((F32(hh) - d) / F32(hh), 0, 1)
            wsum += w
        return np.clip(1 - 1.05 * occ / F32(wsum), 0, 1)

    # ---------------------------------------------------------------- full ray pipeline
    def render_rays(self, O, D, foot0=None, level=0, exclude=None, ref=None):
        """Trace and shade rays. ``ref`` = (pid, normal, t, colour) of a nearby already-shaded ray per ray
        (the pixel centre for anti-aliasing samples): a ray that lands on the same part with the same normal and
        depth takes that colour instead of being shaded again."""
        n = len(O)
        if foot0 is None:
            foot0 = np.full(n, F32(self.pix if self.ortho else 0), F32)
        tg, Ng = self.ground_hit(O, D)
        far = F32(self.extent * 60 + float(np.abs(self.cam.position).max()) * 4 + 200)
        # march slightly past the ground so a surface lying on it (a tabletop, a rug) wins the tie
        grow = F32(0 if self.ortho else self.pix)
        tol = (foot0 + grow * np.where(np.isfinite(tg), tg, 0)) * F32(2) + F32(self.margin)
        t1 = np.minimum(tg + tol, far)
        steps = self.q["steps"] if level == 0 else self.q["refl_steps"]
        T, pid = self.march(O, D, np.zeros(n, F32), t1, foot0, steps, exclude)
        gnd = (pid < 0) & np.isfinite(tg)
        pid[gnd] = 0
        T = np.where(pid > 0, T, np.where(gnd, tg, np.inf)).astype(F32)
        hit = np.flatnonzero(pid >= 0)
        P = np.zeros((n, 3), F32)
        Nn = np.zeros((n, 3), F32)
        color = np.zeros((n, 3), F32)
        spread = F32((0 if self.ortho and level == 0 else self.pix) * (1 if level == 0 else 6))
        foot = foot0 + spread * np.where(np.isfinite(T), T, 0)
        if len(hit):
            P[hit] = O[hit] + D[hit] * T[hit, None]
            Nn[hit] = Ng[hit]
            ob = hit[pid[hit] > 0]
            if len(ob):
                Nn[ob] = self.normals(P[ob], pid[ob], foot[ob])
                # face normals toward the ray (thin shells / inside views / grazing hits on a far edge):
                # mirror across the view plane, which keeps the normal's sideways and up direction
                nd = np.einsum("ij,ij->i", Nn[ob], D[ob])
                flip = nd > 0.2
                if flip.any():
                    fo = ob[flip]
                    Nf = Nn[fo] - 2 * nd[flip, None] * D[fo]
                    Nn[fo] = Nf / np.maximum(np.linalg.norm(Nf, axis=1, keepdims=True), F32(1e-9))
            if ref is not None:
                rp, rn, rt, rc = ref
                h = hit
                same = (pid[h] == rp[h]) & (pid[h] > 0) & (np.einsum("ij,ij->i", Nn[h], rn[h]) > F32(0.985)) \
                    & (np.abs(T[h] - rt[h]) < F32(0.01) * T[h])
                color[h[same]] = rc[h[same]]
                hit = h[~same]
            if len(hit):
                color[hit] = self.shade(P[hit], Nn[hit], D[hit], pid[hit], foot[hit], level,
                                        None if exclude is None else exclude[hit])
                if level == 0 and ref is None:              # what the lighting did, for the manifest
                    self.lit_info = dict(n=n, hit=hit, sun=self._last_sh, shiny=self._last_shiny)
        miss = np.flatnonzero(pid < 0)
        if len(miss):
            color[miss] = self.backdrop_env(D[miss]) if level == 0 else self.env(D[miss])
        if self.horizon_fade and self.ground is not None and self.ground.kind == "plane":
            g = np.flatnonzero(pid == 0)                 # the endless floor melts into the sky at the horizon
            if len(g):
                f = np.clip((T[g] - F32(self.extent * 6)) / F32(self.extent * 30), 0, 1) ** F32(0.6)
                color[g] = color[g] * (1 - f[:, None]) + self.env(D[g] * F32([1, 0, 1])
                                                                 + F32([0, 0.002, 0])) * f[:, None]
        return dict(color=color, t=T, pid=pid, normal=Nn, pos=P)

    def shade(self, P, N, Dv, pid, foot, level, exclude=None):
        n = len(P)
        V = -Dv
        q = self.q
        # material arrays
        alb = np.zeros((n, 3), F32)
        rough = np.zeros(n, F32)
        spec = np.zeros(n, F32)
        metal = np.zeros(n, F32)
        refl = np.zeros(n, F32)
        emis = np.zeros(n, F32)
        sss = np.zeros(n, F32)
        sssc = np.ones((n, 3), F32)
        glass = np.zeros(n, F32)
        ior = np.ones(n, F32)
        coat = np.zeros(n, F32)
        sheen = np.zeros(n, F32)
        for pi in np.unique(pid):
            idx = np.flatnonzero(pid == pi)
            if pi == 0:
                mat = self.ground.mat if self.ground is not None else Material()
                loc = P[idx]
            else:
                part = self.parts[pi - 1]
                mat = part.mat
                loc = part.local(P[idx])
            if isinstance(mat.color, (_Checker, _Wood)):
                graze = np.maximum(np.abs(np.einsum("ij,ij->i", N[idx], Dv[idx])), F32(0.08))
                alb[idx] = mat.color(loc, N[idx], foot[idx] / graze)
            else:
                alb[idx] = mat.albedo(loc, N[idx])
            rough[idx], spec[idx], metal[idx] = mat.roughness, mat.specular, mat.metallic
            refl[idx], emis[idx], sss[idx] = mat.reflect, mat.emissive, mat.subsurface
            glass[idx], ior[idx], coat[idx] = mat.glass, mat.ior, mat.clearcoat
            sheen[idx] = getattr(mat, "sheen", 0.0)
            if mat.sss_color is not None:
                sssc[idx] = _lin(mat.sss_color)
            elif mat.subsurface > 0:
                sssc[idx] = np.clip(alb[idx] * 1.6 + 0.05, 0, 1) ** 0.7
        ndv = np.clip(np.einsum("ij,ij->i", N, V), 0, 1)
        bias = N * F32(self.scale * 1.5e-3) + F32(0)
        Pb = P + bias
        # --- key light with soft shadow
        L = self.sun
        ndl = N @ L
        sh = np.ones(n, F32)
        need = np.flatnonzero((ndl > -0.4 * sss) & (self.sun_col.max() > 0))
        if len(need):
            steps = q["shadow_steps"] if level == 0 else q.get("refl_shadow_steps", max(q["shadow_steps"] // 2, 12))
            sh[need] = self.shadow(Pb[need], L, F32(self.extent * 4), steps)
        if level == 0:
            self._last_sh = np.where(ndl > 0, sh, 1.0).astype(F32)       # cast shadows (not the dark side)
            self._last_shiny = ((rough < 0.3) | (refl > 0.02) | (metal > 0) | (glass > 0)) & (emis <= 0)
        w = sss
        wrap = np.clip((ndl + w) / (1 + w), 0, 1)
        lam = np.clip(ndl, 0, 1)
        sh_soft = sh + (1 - sh) * F32(0.35) * w
        diff_k = lam[:, None] * sh[:, None] + (wrap - lam)[:, None] * sh_soft[:, None] * sssc
        light = diff_k * self.sun_col[None]
        # --- ambient (sky) with occlusion
        aon = q["ao"] if level == 0 else 0
        occl = self.ao(Pb, N, aon) if aon else np.full(n, F32(0.85))
        amb = self.irradiance(N) * occl[:, None]
        amb = amb * (1 + (sssc - 1) * (0.3 * w)[:, None])
        light = light + amb
        # --- fill & rim
        if self.fill_col.max() > 0:
            light = light + np.clip(N @ self.fill, 0, 1)[:, None] * self.fill_col[None] * (0.4 + 0.6 * occl)[:, None]
        if self.rim_col.max() > 0:
            rimt = np.clip(N @ self.rim + 0.25, 0, 1) * (1 - ndv) ** 2.5
            light = light + rimt[:, None] * self.rim_col[None] * occl[:, None]
        # --- point lights
        spec_acc = np.zeros((n, 3), F32)
        shin = np.clip(2.0 / np.maximum(rough, 0.02) ** 4 - 2, 2, 6000).astype(F32)
        F0 = (F32(0.04) * (spec * 2))[:, None] * (1 - metal[:, None]) + alb * metal[:, None]
        norm = np.minimum((shin + 2) / F32(8), F32(60))
        for pl in self.points:
            Lp = pl["pos"].astype(F32)[None] - P
            dist = np.linalg.norm(Lp, axis=1)
            Ld = Lp / np.maximum(dist, 1e-6)[:, None]
            att = pl["power"] / (1 + (dist / pl["radius"]) ** 2)
            nl = np.einsum("ij,ij->i", N, Ld)
            psh = np.ones(n, F32)
            if pl["shadow"] and level == 0:             # shadow rays only where this light still matters
                nd = np.flatnonzero((nl > 0) & (att * float(np.max(pl["color"])) > 0.006))
                psh[nd] = self.shadow(Pb[nd], Ld[nd], dist[nd], q["shadow_steps"], k=10)
            light = light + (np.clip(nl, 0, 1) * att * psh)[:, None] * pl["color"][None]
            Hh = Ld + V
            Hh /= np.maximum(np.linalg.norm(Hh, axis=1, keepdims=True), 1e-6)
            nh = np.clip(np.einsum("ij,ij->i", N, Hh), 0, 1)
            spec_acc += ((nh ** shin) * norm * np.clip(nl, 0, 1) * att * psh)[:, None] * pl["color"][None]
        # --- direct specular from the key light
        Hs = L[None] + V
        Hs /= np.maximum(np.linalg.norm(Hs, axis=1, keepdims=True), 1e-6)
        nh = np.clip(np.einsum("ij,ij->i", N, Hs), 0, 1)
        spec_acc += ((nh ** shin) * norm * lam * sh)[:, None] * self.sun_col[None]
        Fr = F0 + (np.maximum(1 - rough, F0.max(1))[:, None] - F0) * ((1 - ndv) ** 5)[:, None]
        Fr = np.clip(Fr + refl[:, None] * (1 - Fr) * (1 - rough)[:, None], 0, 1)
        specular = spec_acc * Fr * (spec * 2)[:, None]
        # --- reflections
        R = Dv - 2 * np.einsum("ij,ij->i", Dv, N)[:, None] * N
        rcol = self.env(R)
        rough_env = self.irradiance(R) / max(float(self.amb), 1e-3) * 0.9
        cw = coat * (0.04 + 0.96 * (1 - ndv) ** 5)
        trace = np.flatnonzero(((rough < 0.45) & ((Fr.max(1) > 0.035) | (cw > 0.02)) | (glass > 0)) & (level == 0))
        if len(trace) and level == 0:
            Ro = P[trace] + N[trace] * F32(self.scale * 2e-3)
            sub = self.render_rays(Ro, R[trace], foot[trace], level=1)
            rcol[trace] = sub["color"]
        blur = np.clip(rough * 2.2, 0, 1)[:, None]
        envr = rcol * (1 - blur) + rough_env * blur
        envr = envr * (0.35 + 0.65 * occl[:, None])
        refl_term = envr * Fr
        coat_term = rcol * cw[:, None]
        # --- glass transmission
        trans = np.zeros((n, 3), F32)
        gi = np.flatnonzero(glass > 0)
        if len(gi):
            if level <= 1:
                eta = 1.0 / ior[gi]
                d_ = Dv[gi]
                n_ = N[gi]
                cosi = -np.einsum("ij,ij->i", d_, n_)
                k_ = 1 - eta * eta * (1 - cosi * cosi)
                tdir = eta[:, None] * d_ + (eta * cosi - np.sqrt(np.maximum(k_, 0)))[:, None] * n_
                tdir = d_ * 0.55 + tdir * 0.45
                tdir /= np.linalg.norm(tdir, axis=1, keepdims=True)
                excl = pid[gi][:, None].astype(np.int32)
                if exclude is not None:
                    excl = np.concatenate([excl, exclude[gi]], 1)
                sub = self.render_rays(P[gi] - n_ * F32(self.scale * 2e-3), tdir, foot[gi], level=level + 1,
                                       exclude=excl)
                behind = sub["color"]
            else:
                behind = self.env(Dv[gi])
            tint = np.clip(alb[gi], 1e-3, 1) ** (1.0 + 2.5 * (1 - ndv[gi]))[:, None]
            trans[gi] = behind * tint * (1 - Fr[gi])
        # --- assemble
        kd = (1 - metal) * (1 - glass) * (1 - cw)
        diffuse = alb * light * kd[:, None] * (1 - Fr.mean(1, keepdims=True) * 0.5)
        col = diffuse + specular + refl_term + coat_term + trans * glass[:, None]
        if sheen.any():                                 # cloth: fibres catch light at grazing angles
            sh_t = (sheen * (1 - ndv) ** 3)[:, None]
            col = col + sh_t * np.sqrt(np.clip(alb, 0, 1)) * (amb + light * 0.25) * F32(0.6)
        col = col + alb * emis[:, None]
        return col.astype(F32)


# ===================================================================================================== result

class RenderResult:
    """Buffers from ``Scene.render``: ``depth`` (H, W; inf = sky), ``normal`` (H, W, 3), ``ids`` (part index;
    -1 sky, 0 ground), ``names`` (id -> name), ``color`` (H, W, 3) and ``mask(name)``. All are canvas-sized."""

    def __init__(self, scene, frame, canvas_hw, T, Nrm, pid, ctx, covs, col, pos, vis=None):
        H, W = canvas_hw
        x0, y0, w, h = frame if vis is None else vis      # buffers cover the visible part of the frame
        self.scene, self.frame = scene, frame
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        self.depth = np.full((H, W), np.inf, F32)
        self.depth[sl] = T
        self.normal = np.zeros((H, W, 3), F32)
        self.normal[sl] = Nrm
        self.position = np.zeros((H, W, 3), F32)
        self.position[sl] = pos
        self.ids = np.full((H, W), -1, np.int32)
        self.ids[sl] = pid
        self.color = np.zeros((H, W, 3), F32)
        self.color[sl] = col
        self.names = ["ground"] + list(ctx.names)
        self.layers = list(ctx.layer_names)
        self._covs, self._sl, self._hw = covs, sl, (H, W)
        self._part_layer = ctx.layer_of_part
        self._layer_names = ctx.layer_names

    def mask(self, name):
        """Anti-aliased coverage (H, W) of an object (layer name), a part ('car.wheels'), 'ground' or 'sky'."""
        H, W = self._hw
        out = np.zeros((H, W), F32)
        if name in ("sky", "background"):
            out[self._sl] = self._covs.get(0, np.zeros(1))
            return out
        if name in self._layer_names:
            li = self._layer_names.index(name)
            if li in self._covs:
                out[self._sl] = self._covs[li]
            return out
        if name in self.names:
            pid = self.names.index(name)
            m = (self.ids == pid).astype(F32)
            # soften with the owning layer's coverage at the silhouette
            li = self._part_layer[pid + 1]
            lay = np.zeros((H, W), F32)
            lay[self._sl] = self._covs.get(li, 0)
            return np.minimum(m, lay) if pid > 0 else m
        win = self._window_part(name)
        if win is not None:
            return self.mask(win)
        raise _bad_name("object or part", name, list(self._layer_names[2:]) + list(self.names) + ["sky", "ground"])

    def _window_part(self, name):
        """'room.window' / 'room.view' / 'window' -> the panes part of a room."""
        base, _, leaf = str(name).rpartition(".")
        if (leaf if base else name) not in ("window", "windows", "view", "pane", "panes", "window view"):
            return None
        for n in self.names:
            if n.endswith(".panes") and (not base or n == base + ".panes"):
                return n
        return None

    def window_mask(self, name=None):
        """Screen mask (H, W) of the window panes of a ``room()`` (the first room, or the object ``name``): only
        the glass that is visible, so rain or a view painted with it stays inside the frames."""
        return self.mask((name + ".window") if name else "window")

    def project(self, p):
        """World point(s) -> canvas pixel (x, y)."""
        return self.scene.camera.project(p, self.frame)

    def depth_of_field(self, cv, focus=None, blur=8.0, log=None):
        """Lens blur from the depth buffer. ``focus`` = an object name (``"speaker"``), a distance or a world
        point (default: what is in the middle of the frame); ``blur`` = maximum blur radius in px at 1080p."""
        if log:
            log("depth of field")
        H, W = self._hw
        S = H / 1080.0
        dep = self.depth
        if focus is None:
            x0, y0, w, h = self.frame
            c = dep[max(y0 + h // 3, 0):max(y0 + 2 * h // 3, 0), max(x0 + w // 3, 0):max(x0 + 2 * w // 3, 0)]
            c = c[np.isfinite(c)]
            fd = float(np.median(c)) if len(c) else 5.0
        elif isinstance(focus, str):
            m = self.mask(focus) > 0.5
            c = dep[m & np.isfinite(dep)]
            if not len(c):
                raise ValueError(f"{focus!r} is not visible in the frame")
            fd = float(np.percentile(c, 30))          # its front part: what the eye reads as 'the object'
        elif np.ndim(focus) == 0:
            fd = float(focus)
        else:
            cam = self.scene.camera
            fd = float(np.dot(_vec(focus) - cam.position, cam.basis()[0]))
        dep_f = np.where(np.isfinite(dep), dep, fd * 1000)
        coc = np.clip(np.abs(1 - fd / np.maximum(dep_f, 1e-6)), 0, 1.5) * blur * S       # px
        from .filters import gaussian_blur
        img = cv.buf
        levels = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        levels = [lv for lv in levels if lv <= max(coc.max(), 1.0) * 1.01 + 1e-6] or [0.0]
        if levels[-1] < coc.max():
            levels.append(float(coc.max()))
        behind = (dep_f >= fd).astype(F32)
        stack = []
        for lv in levels:
            if lv == 0:
                stack.append(img)
                continue
            sg = lv * 0.5
            wb = gaussian_blur(behind, sg)[..., None]
            bb = gaussian_blur(img * behind[..., None], sg) / np.maximum(wb, 1e-4)
            fg = gaussian_blur(img, sg)
            stack.append(np.where(wb > 1e-3, bb, fg) * behind[..., None] + fg * (1 - behind[..., None]))
        lv = np.array(levels, F32)
        ci = np.clip(np.interp(coc, lv, np.arange(len(lv))), 0, len(lv) - 1)
        i0 = np.floor(ci).astype(int)
        i1 = np.minimum(i0 + 1, len(lv) - 1)
        fr = (ci - i0)[..., None].astype(F32)
        S_ = np.stack(stack, 0)
        yy, xx = np.mgrid[0:H, 0:W]
        out = S_[i0, yy, xx] * (1 - fr) + S_[i1, yy, xx] * fr
        cv.paint(out.astype(F32))
        return self

    def fog(self, cv, color="#c8d0dc", density=0.05, start=0.0, sky=False, log=None):
        """Distance fog from the depth buffer: amount = 1 - exp(-density * (depth - start))."""
        if log:
            log("fog")
        dep = self.depth
        d = np.where(np.isfinite(dep), np.maximum(dep - start, 0), 1e9 if sky else 0)
        a = (1 - np.exp(-density * d)).astype(F32)
        cv.paint(color, a)
        return self


# ===================================================================================================== prefabs
#
# Every prefab stands on y = 0, is centred on x = z = 0, faces +z (towards the default camera) and uses metres.
# Vehicles point along +x, so the default camera sees their side; turn them with .rotate_y().

def _noisy(c1, c2, scale=1.0, octaves=3, seed=0, contrast=1.0):
    """Colour function: two colours mixed by 3D noise (foliage, stone, fur variation)."""
    A, B = _lin(c1), _lin(c2)

    def fn(P, N=None):
        p = P / F32(scale)
        t = fbm3(p[:, 0], p[:, 1], p[:, 2], octaves, seed)
        t = np.clip((t - 0.5) * F32(contrast * 2.2) + 0.5, 0, 1)
        return A[None] * (1 - t[:, None]) + B[None] * t[:, None]
    return fn


def mug(color="#f2efe9", coffee=True, size=1.0):
    """Ceramic mug (9.5 cm tall) with a handle; ``coffee`` adds a dark drink inside."""
    r, h = 0.045, 0.095
    outer = cylinder(r, h, round=0.004).at(0, h / 2, 0)
    inner = cylinder(r - 0.0055, h, round=0.004).at(0, h / 2 + 0.008, 0)
    handle = torus(0.026, 0.0062).rotate_x(90).scale((1.0, 1.15, 1.0)).at(r + 0.006, h * 0.52, 0)
    handle = handle.cut(normal=(-1, 0, 0), offset=-(r - 0.002))
    body = (outer - inner) | handle
    parts = [Part("body", body, material("ceramic", color))]
    if coffee:
        drink = cylinder(r - 0.0048, 0.02).at(0, h - 0.024, 0)
        parts.append(Part("coffee", drink, Material(color="#3a1d0e", roughness=0.18, specular=0.6)))
    return Model("mug", parts).scale(size)


def bottle(color="#d8efe6", liquid="#7a3a1a", level=0.55, label="#f4ecd8", cap="#20242a", height=0.3):
    """Glass bottle with an optional liquid (fill ``level`` 0~1), paper label and cap; ``height`` in metres."""
    s = height / 0.3
    prof = [(0.0, 0.003), (0.034, 0.003), (0.039, 0.012), (0.04, 0.05), (0.04, 0.165), (0.036, 0.195),
            (0.02, 0.228), (0.0135, 0.25), (0.0135, 0.283)]
    glass_ = lathe(prof, thickness=0.0028)
    parts = [Part("glass", glass_, material("glass", color))]
    if liquid and level > 0:
        top = 0.012 + level * 0.175
        lp = [(0.0, 0.006), (0.034, 0.006), (0.037, 0.014), (0.0372, 0.05), (0.0372, min(top, 0.165))]
        if top > 0.165:
            lp += [(0.033, min(top, 0.193))]
        lp += [(0.0, lp[-1][1])]
        parts.append(Part("liquid", lathe(lp, smooth=False),
                          Material(color=liquid, roughness=0.05, specular=1.0, glass=0.75, ior=1.33, shadow=0.5)))
    if label:
        band = (cylinder(0.0428, 0.075) - cylinder(0.0418, 0.09)).at(0, 0.1, 0)
        parts.append(Part("label", band, Material(color=label, roughness=0.8, specular=0.2)))
    if cap:
        parts.append(Part("cap", cylinder(0.0155, 0.022, round=0.003).at(0, 0.29, 0),
                          Material(color=cap, roughness=0.3, specular=0.7)))
    return Model("bottle", parts).scale(s)


def vase(color="#3f6f8f", height=0.32, shape="round"):
    """Ceramic vase with an open mouth; ``shape`` = 'round' | 'tall' | 'bowl'."""
    _check_choice("vase", "shape", shape, ("round", "tall", "bowl"))
    profiles = {
        "round": [(0.0, 0.0), (0.06, 0.0), (0.105, 0.07), (0.11, 0.13), (0.07, 0.23), (0.045, 0.28), (0.055, 0.32)],
        "tall": [(0.0, 0.0), (0.05, 0.0), (0.07, 0.08), (0.065, 0.2), (0.045, 0.28), (0.06, 0.32)],
        "bowl": [(0.0, 0.0), (0.05, 0.0), (0.12, 0.06), (0.16, 0.13), (0.17, 0.16)],
    }
    p = profiles[shape]
    top = p[-1][1]
    body = lathe(p, thickness=0.008)
    return Model("vase", [Part("body", body, material("ceramic", color))]).scale(height / (top if shape != "bowl"
                                                                                             else 0.32))


def table(width=1.2, depth=0.7, height=0.75, color=None, legs=None, round_top=False):
    """Table with four legs (wood by default). ``round_top=True`` -> round top of diameter ``width``."""
    top_m = material("wood") if color is None else material("wood", color)
    leg_m = top_m if legs is None else material(legs)
    t = 0.04
    if round_top:
        top = cylinder(width / 2, t, round=0.008).at(0, height - t / 2, 0)
        leg = cylinder(0.035, height - t).at(0, (height - t) / 2, 0)
        foot = cylinder(0.22, 0.03, round=0.01).at(0, 0.015, 0)
        legs_ = leg | foot
    else:
        top = rounded_box(width, t, depth, 0.01).at(0, height - t / 2, 0)
        lh = height - t
        legs_ = rounded_box(0.05, lh, 0.05, 0.01).at(width / 2 - 0.07, lh / 2, depth / 2 - 0.07).mirror("xz")
        apron = (rounded_box(width - 0.14, 0.08, 0.025, 0.005).at(0, lh - 0.04, depth / 2 - 0.07).mirror("z")
                 | rounded_box(0.025, 0.08, depth - 0.14, 0.005).at(width / 2 - 0.07, lh - 0.04, 0).mirror("x"))
        legs_ = legs_ | apron
    return Model("table", [Part("top", top, top_m), Part("legs", legs_, leg_m)])


def chair(color="#c9a27a", seat="#c85a3a", height=0.9):
    """Dining chair: wooden frame, cushioned seat, slatted back."""
    fm = material("wood", wood("#d8ae7e", "#9a6a40") if color is None else color)
    sh = 0.45
    legs = rounded_box(0.04, sh, 0.04, 0.008).at(0.19, sh / 2, 0.19).mirror("xz")
    posts = rounded_box(0.04, height - sh, 0.04, 0.008).at(0.19, sh + (height - sh) / 2, -0.19).mirror("x")
    rail = rounded_box(0.42, 0.07, 0.03, 0.01).at(0, height - 0.06, -0.19)
    rail2 = rounded_box(0.42, 0.05, 0.025, 0.01).at(0, sh + 0.17, -0.19)
    frame = legs | posts | rail | rail2 | rounded_box(0.44, 0.03, 0.44, 0.008).at(0, sh - 0.015, 0)
    cushion = rounded_box(0.42, 0.05, 0.42, 0.022).at(0, sh + 0.02, 0.01)
    return Model("chair", [Part("frame", frame, fm), Part("seat", cushion, material("fabric", seat))])


def sofa(color="#5b7fa6", width=2.0, cushions=3, legs="#3a2a1e"):
    """Three-seat sofa with soft cushions. Parts: body, armrest_l (+x, the sitter's left), armrest_r, cushions,
    legs."""
    d, base_h = 0.9, 0.42
    fab = material("fabric", color)
    frame = rounded_box(width, 0.26, d, 0.06).at(0, 0.08 + 0.13, 0)
    back = rounded_box(width, 0.5, 0.22, 0.09).at(0, 0.34 + 0.25, -d / 2 + 0.11)
    arm = rounded_box(0.2, 0.4, d, 0.09)
    body = smooth_union(frame, back, k=0.03)
    cw = (width - 0.42) / cushions
    cs = [rounded_box(cw - 0.015, 0.16, d - 0.28, 0.06).at(-width / 2 + 0.21 + cw * (i + 0.5), base_h, 0.08)
          for i in range(cushions)]
    bc = [rounded_box(cw - 0.03, 0.4, 0.16, 0.08).rotate_x(-12).at(-width / 2 + 0.21 + cw * (i + 0.5), 0.66,
                                                                    -d / 2 + 0.28)
          for i in range(cushions)]
    lg = cone(0.025, 0.018, 0.09).at(width / 2 - 0.1, 0.045, d / 2 - 0.1).mirror("xz")
    cush = material("fabric", _shade_hex(color, 1.12))
    return Model("sofa", [Part("body", body, fab),
                          Part("armrest_l", arm.at(width / 2 - 0.1, 0.08 + 0.2 + 0.08, 0), fab),
                          Part("armrest_r", arm.at(-width / 2 + 0.1, 0.08 + 0.2 + 0.08, 0), fab),
                          Part("cushions", union(*cs, *bc), cush),
                          Part("legs", lg, material("wood", legs))])


def _shade_hex(c, k):
    v = np.clip(rgb(c) * k, 0, 1)
    return "#%02x%02x%02x" % tuple(int(round(x * 255)) for x in v)


def bed(width=1.6, length=2.1, duvet="#e9eef4", frame="#b48a62", pillows="#ffffff", throw="#d9785a"):
    """Double bed: wooden frame and headboard, mattress, duvet, two pillows and a throw at the foot."""
    fm = material("wood", wood("#c49a6c", "#8a5e3a"))
    if frame is not None and not str(frame).startswith("#b48a62"):
        fm = material("wood", frame)
    base = rounded_box(width + 0.08, 0.28, length + 0.04, 0.02).at(0, 0.18, 0)
    legs = rounded_box(0.07, 0.06, 0.07, 0.01).at(width / 2 - 0.02, 0.03, length / 2 - 0.05).mirror("xz")
    head = rounded_box(width + 0.12, 0.95, 0.07, 0.03).at(0, 0.48, -length / 2 - 0.03)
    mattress = rounded_box(width, 0.2, length, 0.06).at(0, 0.42, 0)
    duv = rounded_box(width + 0.04, 0.09, length * 0.72, 0.045).at(0, 0.53, length * 0.14)
    duv = duv | rounded_box(0.03, 0.3, length * 0.72, 0.015).at(width / 2 + 0.02, 0.4, length * 0.14).mirror("x")
    pil = rounded_box(0.6, 0.13, 0.38, 0.065).rotate_x(-8).at(width / 4 + 0.01, 0.61, -length / 2 + 0.3).mirror("x")
    thr = rounded_box(width + 0.08, 0.04, 0.45, 0.02).at(0, 0.58, length / 2 - 0.3)
    thr = thr | rounded_box(0.03, 0.26, 0.45, 0.012).at(width / 2 + 0.05, 0.46, length / 2 - 0.3).mirror("x")
    return Model("bed", [Part("frame", base | legs | head, fm),
                         Part("mattress", mattress, material("fabric", "#f4f2ee")),
                         Part("duvet", duv, material("fabric", duvet, subsurface=0.25)),
                         Part("pillows", pil, material("fabric", pillows, subsurface=0.3)),
                         Part("throw", thr, material("fabric", throw))])


def desk_lamp(color="#e8c547", glow="#ffe2a8", angle=30):
    """Desk lamp: round base, two-segment arm, cone shade and a glowing bulb (pair with ``Scene.light_from``)."""
    m = material("plastic", color, roughness=0.3)
    base = cylinder(0.075, 0.022, round=0.008).at(0, 0.011, 0)
    j1 = np.array([0.0, 0.03, 0.0])
    j2 = j1 + np.array([0.06, 0.3, 0.0])
    j3 = j2 + np.array([0.24, 0.03, 0.0])
    arm = capsule(j1, j2, 0.009) | capsule(j2, j3, 0.009) | sphere(0.016).at(*j2)
    a = angle
    shade = (cone(0.075, 0.03, 0.12) - cone(0.068, 0.024, 0.124).at(0, -0.004, 0)).rotate_z(a).at(*(j3 + [0.02,
                                                                                                     -0.07, 0]))
    bulb_c = j3 + np.array([0.02, -0.07, 0]) + np.array([math.sin(math.radians(a)) * 0.01,
                                                        -math.cos(math.radians(a)) * 0.01, 0])
    bulb = sphere(0.022).at(*bulb_c)
    return Model("desk lamp", [Part("base", base, m), Part("arm", arm, material("metal", "#cfd2d6")),
                               Part("shade", shade, m), Part("bulb", bulb, material("emissive", glow))])


def book(w=0.03, h=0.24, d=0.17, color="#a33a2a"):
    """A single closed book standing upright (spine faces +z)."""
    cover = rounded_box(w, h, d, 0.004).at(0, h / 2, 0)
    pages = box(w - 0.006, h - 0.01, d - 0.004).at(0, h / 2, -0.004)
    return Model("book", [Part("cover", cover - box(w - 0.006, h - 0.01, d).at(0, h / 2, -0.006),
                               material("plastic", color, roughness=0.6)),
                          Part("pages", pages, material("matte", "#f2ead8"))])


_BOOK_COLORS = ["#a33a2a", "#2f5d8a", "#e2b33c", "#3f7f5a", "#d9d2c3", "#6b4a8a", "#c8643a", "#1f2a3a"]


def _book_rows(width, shelf_ys, depth, seed=0, fill=0.85):
    rng = np.random.default_rng(seed)
    groups = {c: [] for c in _BOOK_COLORS}
    for y in shelf_ys:
        x = -width / 2 + 0.02
        while x < width / 2 - 0.05:
            if rng.random() > fill:
                x += rng.uniform(0.04, 0.12)
                continue
            bw = rng.uniform(0.02, 0.045)
            bh = rng.uniform(0.17, 0.27)
            bd = depth * rng.uniform(0.7, 0.9)
            lean = 0.0
            if x + bw > width / 2 - 0.02:
                break
            c = _BOOK_COLORS[rng.integers(len(_BOOK_COLORS))]
            b = rounded_box(bw, bh, bd, 0.004)
            if rng.random() < 0.08:
                lean = rng.uniform(-14, 14)
            groups[c].append(b.rotate_z(lean).at(x + bw / 2, y + bh / 2 + 0.002, -depth / 2 + bd / 2 + 0.01))
            x += bw + rng.uniform(0.0, 0.006)
    return groups


def bookshelf(width=0.9, height=1.8, depth=0.3, color=None, books=True, shelves=5, seed=1):
    """Bookshelf with ``shelves`` shelves, filled with books of mixed colours and heights."""
    fm = material("wood", wood("#c79c6e", "#94683f") if color is None else color)
    t = 0.025
    sides = box(t, height, depth).at(width / 2 - t / 2, height / 2, 0).mirror("x")
    back = box(width, height, 0.01).at(0, height / 2, -depth / 2 + 0.005)
    ys = [0.06 + i * (height - 0.08) / shelves for i in range(shelves + 1)]
    boards = union(*[box(width - 2 * t, t, depth - 0.01).at(0, y, 0.005) for y in ys])
    frame = sides | back | boards | box(width, 0.06, 0.02).at(0, 0.03, depth / 2 - 0.01)
    parts = [Part("frame", frame, fm)]
    if books:
        groups = _book_rows(width - 2 * t, [y + t / 2 for y in ys[:-1]], depth - 0.02, seed)
        for i, (c, shapes) in enumerate(groups.items()):
            if shapes:
                parts.append(Part(f"books {i + 1}", union(*shapes), material("plastic", c, roughness=0.55)))
    return Model("bookshelf", parts)


def house(width=6.0, depth=5.0, height=3.0, walls="#f1e6d2", roof="#b5523b", door="#6b4630", windows="#9cc4e4"):
    """Cottage: walls, pitched roof with overhang, door, windows with frames, chimney."""
    wall = box(width, height, depth).at(0, height / 2, 0)
    rh = depth * 0.42
    ov = 0.35
    roof_s = prism([(-depth / 2 - ov, 0), (depth / 2 + ov, 0), (0, rh + ov * rh / (depth / 2))], width + 2 * ov)
    roof_s = (roof_s - prism([(-depth / 2 - ov + 0.25, -0.01), (depth / 2 + ov - 0.25, -0.01),
                              (0, rh + ov * rh / (depth / 2) - 0.18)], width + 2 * ov + 1)).rotate_y(90)
    roof_s = roof_s.at(0, height - 0.05, 0)
    gable = prism([(-depth / 2, 0), (depth / 2, 0), (0, rh)], width - 0.02).rotate_y(90).at(0, height - 0.01, 0)
    chim = box(0.5, 1.6, 0.5).at(width * 0.25, height + rh * 0.55, -depth * 0.18)
    dw, dh = 0.95, 2.05
    door_s = rounded_box(dw, dh, 0.08, 0.02).at(-width * 0.18, dh / 2, depth / 2)
    win = []
    frames = []
    for wx, wy in ((width * 0.22, 1.55), (-width * 0.38, 1.55)):
        win.append(box(1.0, 1.0, 0.06).at(wx, wy, depth / 2 + 0.005))
        fr = (box(1.16, 1.16, 0.1) - box(1.0, 1.0, 0.3)) | box(1.0, 0.06, 0.08) | box(0.06, 1.0, 0.08)
        frames.append(fr.at(wx, wy, depth / 2 + 0.02))
    side_win = box(0.06, 1.0, 1.0).at(width / 2 + 0.005, 1.55, 0)
    side_fr = ((box(0.1, 1.16, 1.16) - box(0.3, 1.0, 1.0)) | box(0.08, 0.06, 1.0) | box(0.08, 1.0, 0.06)).at(
        width / 2 + 0.02, 1.55, 0)
    step = box(1.4, 0.15, 0.5).at(-width * 0.18, 0.075, depth / 2 + 0.25)
    return Model("house", [
        Part("walls", wall | gable | step, material("matte", walls)),
        Part("roof", roof_s | chim, material("clay", roof, subsurface=0.0, roughness=0.7)),
        Part("door", door_s, material("plastic", door, roughness=0.5)),
        Part("windows", union(*win, side_win), Material(color=windows, roughness=0.05, specular=1.0, reflect=0.35)),
        Part("frames", union(*frames, side_fr), material("matte", "#ffffff")),
    ])


def _rot3(axis, ang):
    a = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return (np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K).astype(F32)


_LEAF_ROT = _rot3((0.3, 0.8, 0.5), 0.9)


def _leafy(x, y, z, k, ids=False):
    """Cheap leaf-cluster field in 0~1 (world space, leaf period 2 pi / k): two rotated lattices of round lobes,
    high on the leaves, low in the gaps between them. ``ids=True`` also returns a random 0~1 per leaf."""
    x, y, z = x * k, y * k, z * k
    e1 = np.cos(x) + np.cos(y) + np.cos(z)
    m = _LEAF_ROT
    u = x * m[0, 0] + y * m[0, 1] + z * m[0, 2] + F32(1.3)
    v = x * m[1, 0] + y * m[1, 1] + z * m[1, 2] + F32(0.4)
    w = x * m[2, 0] + y * m[2, 1] + z * m[2, 2] + F32(2.1)
    e2 = np.cos(u) + np.cos(v) + np.cos(w)
    f = np.clip((np.maximum(e1, e2) + F32(0.6)) * F32(0.28), 0, 1)
    if not ids:
        return f
    two_pi = F32(1 / (2 * math.pi))
    first = e1 >= e2
    a = np.where(first, x, u + F32(311.0)) * two_pi
    b = np.where(first, y, v) * two_pi
    c = np.where(first, z, w) * two_pi
    return f, _hash3(np.round(a).astype(np.int64), np.round(b).astype(np.int64), np.round(c).astype(np.int64), 3)


class _Canopy(Shape):
    """A crown of leaf clumps: slightly flattened spheres (centres ``C``, radii ``R``) whose surface breaks up
    into leaf bumps of size ``leaf``. One shape for all clumps; a coarse grid keeps, per cell, only the clumps
    that can be nearest, so the cost stays low with many clumps."""

    cost = 4

    def __init__(self, C, R, leaf, flat=1.25, cell=0.12):
        self.C, self.R = np.asarray(C, F32), np.asarray(R, F32)
        self.k, self.flat = F32(2 * math.pi / leaf), F32(flat)
        self.amp = F32(leaf * 0.4)
        fl = np.array([1, flat, 1], np.float64)
        Sc = self.C.astype(np.float64) * fl                                 # centres in the flattened space
        m = self.margin = 0.3
        lo = (Sc - self.R[:, None]).min(0) - m
        hi = (Sc + self.R[:, None]).max(0) + m
        dims = np.maximum(np.ceil((hi - lo) / cell).astype(int), 1)
        self.lo, self.dims, self.cell = lo.astype(F32), dims, F32(cell)
        g = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), -1).reshape(-1, 3)
        c0 = lo + g * cell
        # box-to-sphere-surface distance per (cell, clump): keep the clumps closer than the margin
        q = np.maximum(np.maximum(c0[:, None] - Sc[None], Sc[None] - (c0[:, None] + cell)), 0)
        lb = np.sqrt((q ** 2).sum(-1)) - self.R[None]
        far = np.maximum(np.abs(c0[:, None] - Sc[None]), np.abs(c0[:, None] + cell - Sc[None]))
        ub = (np.sqrt((far ** 2).sum(-1)) - self.R[None]).min(1, keepdims=True)   # some clump is this close
        near = lb < np.minimum(ub, m)
        M = max(int(near.sum(1).max()), 1)
        lists = np.full((len(g), M), len(self.C), np.int64)                 # padding -> a far dummy clump
        for ci in np.flatnonzero(near.any(1)):
            ks = np.flatnonzero(near[ci])
            lists[ci, :len(ks)] = ks
        self.lists, self.count = lists, near.sum(1)
        self.Sd = np.concatenate([Sc, [[1e4, 1e4, 1e4]]]).astype(F32)
        self.Rd = np.concatenate([self.R, [0]]).astype(F32)
        self.blo, self.bhi = lo.astype(F32), hi.astype(F32)

    def _d(self, x, y, z):
        fy = self.flat
        P = np.stack([x, y * fy, z], 1)
        # outside the grid: distance to the grid box plus the margin is a safe lower bound
        q = np.maximum(np.maximum(self.blo - P, P - self.bhi), 0)
        outd = np.sqrt((q * q).sum(1))
        out = (outd + F32(self.margin)) * F32(1.0 / fy)
        inside = np.flatnonzero(outd <= 0)
        if len(inside):
            p = P[inside]
            ci = np.clip(((p - self.lo) / self.cell).astype(np.int64), 0, self.dims - 1)
            cid = (ci[:, 0] * self.dims[1] + ci[:, 1]) * self.dims[2] + ci[:, 2]
            cnt = self.count[cid]
            d = np.full(len(p), F32(self.margin))
            sel = np.arange(len(p))
            for j in range(self.lists.shape[1]):         # clump j of each point's cell, while the cell has one
                sel = sel[cnt[sel] > j]
                if not len(sel):
                    break
                k = self.lists[cid[sel], j]
                v = p[sel] - self.Sd[k]
                dj = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2]) - self.Rd[k]
                d[sel] = np.minimum(d[sel], dj)
            d = d * F32(1.0 / fy)
            o = d - self.amp
            near = np.flatnonzero(d < self.amp * 1.5)
            if len(near):
                j = inside[near]
                n = _leafy(x[j], y[j], z[j], self.k)
                o[near] = (d[near] - self.amp * n) * F32(0.62)
            out[inside] = o
        return out

    def _bb(self):
        e = (self.R + self.amp)[:, None]
        return (self.C - e).min(0).astype(np.float64), (self.C + e).max(0).astype(np.float64)


class _Foliage:
    """Canopy colour: every clump its own blend of ``palette`` tones, lit leaf bumps and dark gaps between them,
    shade inside the crown and under it."""

    _kind = "foliage"

    def __init__(self, centres, radii, palette, crown_c, crown_r, leaf, seed=0):
        self.C = np.asarray(centres, F32)
        self.R = np.asarray(radii, F32)
        rng = np.random.default_rng(seed + 11)
        pal = np.stack([_lin(c) for c in palette])
        n = len(self.C)
        i, j = rng.integers(0, len(pal), n), rng.integers(0, len(pal), n)
        t = rng.uniform(0, 1, n)[:, None]
        bright = rng.uniform(0.85, 1.12, n)[:, None]
        self.tone = ((pal[i] * (1 - t) + pal[j] * t) * bright).astype(F32)
        self.cc, self.cr = np.asarray(crown_c, F32), np.asarray(crown_r, F32)
        self.k, self.seed = F32(2 * math.pi / leaf), seed

    def __call__(self, P, N=None):
        P = np.asarray(P, F32)
        out = np.empty((len(P), 3), F32)
        r_ = np.arange(min(len(P), 20000))
        for a in range(0, len(P), 20000):
            p = P[a:a + 20000]
            r = r_[:len(p)]
            d = np.sqrt(((p[:, None, :] - self.C[None]) ** 2).sum(-1)) - self.R[None]
            o = np.argsort(d, 1)[:, :2]
            i, j = o[:, 0], o[:, 1]
            gap = d[r, j] - d[r, i]
            wb = (np.clip(gap / F32(0.25), 0, 1) * 0.5 + 0.5)[:, None]   # soft seam where two clumps meet
            col = self.tone[i] * wb + self.tone[j] * (1 - wb)
            leaf, lid = _leafy(p[:, 0], p[:, 1], p[:, 2], self.k, ids=True)
            fine = _leafy(p[:, 0] + 0.37, p[:, 1], p[:, 2], self.k * F32(2.6))
            sp = noise3(p[:, 0] * 9, p[:, 1] * 9, p[:, 2] * 9, self.seed + 5)
            # each leaf bump its own shade, leaning to the tone of the neighbouring clump; dark gaps between
            col = col * (1 - lid[:, None] * 0.35) + self.tone[j] * (lid[:, None] * 0.35)
            k = 0.32 + 0.6 * leaf + 0.4 * fine ** 2 + 0.3 * (sp - 0.5) + 0.35 * (lid - 0.5)
            q = (p - self.cc[None]) / self.cr[None]
            rad = np.sqrt((q ** 2).sum(1))
            k = k * np.clip(0.3 + 0.75 * rad, 0.35, 1.05)                # inside the crown: in shade
            k = k * np.clip(0.8 + 0.35 * q[:, 1], 0.55, 1.0)             # underside darker
            out[a:a + 20000] = col * k[:, None].astype(F32)
        return out


_CANOPY = {
    # trunk (height, radius); limbs: count, reach range, mass height range; leaf masses:
    # radius range, vertical squash; clumps per mass and clump radius range; leader = the top mass (height, radius);
    # sub = where a second mass sits on every limb (outward, up) from the first; fork = where limbs leave the trunk
    "maple": dict(trunk=(1.05, 0.12), limbs=5, reach=(0.42, 0.68), mh=(1.55, 2.0), mass=(0.42, 0.56),
                  squash=0.95, per=12, r=(0.15, 0.26), leader=(2.48, 0.5), sub=(-0.3, 0.5), fork=(0.85, 0.97)),
    "oak": dict(trunk=(0.95, 0.17), limbs=5, reach=(0.75, 1.1), mh=(1.55, 1.95), mass=(0.46, 0.62),
                squash=0.75, per=12, r=(0.16, 0.27), leader=(2.28, 0.58), sub=(0.45, 0.05), fork=(0.6, 0.97)),
}
_SEASONS = {
    "maple": {"summer": ["#3f7a26", "#5a952f", "#76ad3c", "#4c8a2a", "#8dbb4a"],
              "autumn": ["#b8321c", "#d8492a", "#e6702a", "#f09a2e", "#f2bf3a", "#9e2618"]},
    "oak": {"summer": ["#3d6a24", "#4f7e2b", "#648f34", "#466f22", "#7a9c3c"],
            "autumn": ["#8a5a22", "#b07a2a", "#c79a36", "#9c4a22", "#7a7230", "#d0a83e"]},
}


def _palette_from(c):
    """A few tones around one leaf colour (lighter, darker, slightly warmer and cooler)."""
    base = rgb(c)
    out = []
    for k, warm in ((1.0, 0), (0.8, 0.03), (1.15, -0.02), (0.9, -0.03), (1.08, 0.05)):
        v = np.clip(base * k + np.array([warm, warm * 0.4, -warm]), 0, 1)
        out.append("#%02x%02x%02x" % tuple(int(round(t * 255)) for t in v))
    return out


def _cluster_tree(kind, leaves, trunk, season, seed):
    cfg = _CANOPY[kind]
    rng = np.random.default_rng(seed)
    if leaves is None:
        palette = _SEASONS[kind][season]
    elif isinstance(leaves, (list, tuple)) and len(leaves) and not isinstance(leaves[0], (int, float)):
        palette = list(leaves)
    else:
        palette = _palette_from(leaves)
    th, tr = cfg["trunk"]
    bark = material("matte", _noisy(trunk, _shade_hex(trunk, 0.62), 0.04, 4, seed=seed, contrast=1.3))
    # trunk with a flared foot; it forks into limbs that each carry a mass of leaf clumps
    wood_ = [cone(tr, tr * 0.7, th).at(0, th / 2, 0), cone(tr * 1.45, tr * 0.9, 0.16).at(0, 0.08, 0)]
    nl = cfg["limbs"]
    masses = []
    a0 = rng.uniform(0, 2 * math.pi)
    for i in range(nl):
        a = a0 + 2 * math.pi * (i + rng.uniform(-0.25, 0.25)) / nl
        reach = rng.uniform(*cfg["reach"])
        top = rng.uniform(*cfg["mh"])
        base = np.array([0, th * rng.uniform(*cfg["fork"]), 0])          # limbs leave at different heights
        end = np.array([reach * math.cos(a), top, reach * math.sin(a)])
        mid = base + (end - base) * 0.45 + np.array([0, -0.08, 0])
        mid[0] += 0.08 * math.cos(a + 1.2)
        mid[2] += 0.08 * math.sin(a + 1.2)
        wood_.append(capsule(tuple(base), tuple(mid), tr * 0.52))
        wood_.append(capsule(tuple(mid), tuple(end), tr * 0.34))
        masses.append((end + np.array([0, 0.1, 0]), rng.uniform(*cfg["mass"])))
        # a branch from the limb to a second, smaller mass
        out_, up = cfg["sub"]
        sub = end + np.array([out_ * math.cos(a), up, out_ * math.sin(a)]) * rng.uniform(0.8, 1.15)
        wood_.append(capsule(tuple(mid + (end - mid) * 0.5), tuple(sub), tr * 0.25))
        masses.append((sub + np.array([0, 0.08, 0]), rng.uniform(*cfg["mass"]) * 0.8))
    ly, lr = cfg["leader"]
    wood_.append(capsule((0, th - 0.1, 0), (0, ly - lr * 0.3, 0), tr * 0.45))
    masses.append((np.array([rng.uniform(-0.1, 0.1), ly, rng.uniform(-0.1, 0.1)]), lr))
    # leaf clumps on the shell of every mass, open underneath so limbs and the shade under the crown show
    C, R = [], []
    sq = cfg["squash"]
    for mc, mr in masses:
        got, tries = 0, 0
        while got < cfg["per"] and tries < 400:
            tries += 1
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            if v[1] < -0.6:
                continue
            f = rng.uniform(0.55, 1.0)
            p = mc + np.array([v[0], v[1] * sq, v[2]]) * mr * f
            r = rng.uniform(*cfg["r"]) * (0.8 if v[1] < -0.2 else 1.0)
            if any(np.linalg.norm(p - q) < 0.6 * (r + rq) for q, rq in zip(C, R)):
                continue
            C.append(p)
            R.append(r)
            got += 1
        C.append(mc + np.array([0, 0.05, 0]))              # the core of the mass
        R.append(mr * 0.6)
    C, R = np.array(C), np.array(R)
    lo, hi = (C - R[:, None]).min(0), (C + R[:, None]).max(0)
    leaf_size = 0.09
    fol_mat = material("leaf", _Foliage(C, R, palette, (lo + hi) / 2, (hi - lo) / 2, leaf_size, seed))
    fol_mat.roughness, fol_mat.specular, fol_mat.subsurface = 0.8, 0.1, 0.3
    wood_ = smooth_union(*wood_, k=tr * 0.22)
    return Model("tree", [Part("trunk", wood_, bark), Part("foliage", _Canopy(C, R, leaf_size), fol_mat)])


def tree(height=3.0, kind="round", leaves=None, trunk="#7a5236", seed=0, season="summer"):
    """Tree: trunk + foliage. ``kind`` = 'round' (blobby crown) | 'pine' (stacked cones) | 'bush' | 'maple' |
    'oak' (limbs under a canopy of many leaf clumps, each its own tone; oak is wider and flatter).

    ``leaves`` = one colour, or a list of colours for the clumps of a maple / oak (default: the ``season``
    palette, 'summer' greens or 'autumn' reds, oranges and golds)."""
    _check_choice("tree", "kind", kind, ("round", "pine", "bush", "maple", "oak"))
    _check_choice("tree", "season", season, ("summer", "autumn"))
    s = height / 3.0
    if kind in _CANOPY:
        return _cluster_tree(kind, leaves, trunk, season, seed).scale(s)
    if leaves is None:
        leaves = "#5f9e3e" if season == "summer" else "#d2702c"
    rng = np.random.default_rng(seed)
    bark = material("matte", _noisy(trunk, _shade_hex(trunk, 0.7), 0.05, seed=seed))
    leaf = material("leaf", _noisy(leaves, _shade_hex(leaves, 0.72), 0.18 * s, 4, seed=seed + 3, contrast=1.4))
    if kind == "pine":
        tr = cone(0.12, 0.07, 0.8).at(0, 0.4, 0)
        tiers = [cone(0.95 - 0.22 * i, 0.08, 1.0).at(0, 0.75 + i * 0.6, 0) for i in range(4)]
        fol = smooth_union(*tiers, k=0.05)
        return Model("tree", [Part("trunk", tr, bark), Part("foliage", fol, leaf)]).scale(s)
    if kind == "bush":
        blobs = [sphere(rng.uniform(0.35, 0.5)).at(rng.uniform(-0.45, 0.45), rng.uniform(0.35, 0.6),
                                                    rng.uniform(-0.35, 0.35)) for _ in range(6)]
        return Model("bush", [Part("foliage", smooth_union(*blobs, k=0.18), leaf)]).scale(s)
    tr = cone(0.16, 0.09, 1.7).at(0, 0.85, 0)
    br = capsule((0, 1.3, 0), (0.45, 1.9, 0.1), 0.06) | capsule((0, 1.4, 0), (-0.4, 2.0, -0.15), 0.055)
    blobs = [sphere(0.78).at(0, 2.15, 0)]
    for i in range(7):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.45, 0.7)
        blobs.append(sphere(rng.uniform(0.38, 0.58)).at(r * math.cos(a), 2.0 + rng.uniform(-0.3, 0.55),
                                                        r * math.sin(a)))
    fol = smooth_union(*blobs, k=0.25)
    return Model("tree", [Part("trunk", tr | br, bark), Part("foliage", fol, leaf)]).scale(s)


def _wheel(r=0.36, w=0.24, rim="#d9dce0", tire="#1d1e22", hub=None):
    tyre = cylinder(r, w, round=r * 0.28).rotate_x(90)
    groove = torus(r - 0.002, 0.012).rotate_x(90)
    rimc = (cylinder(r * 0.6, w + 0.012, round=0.01) - cylinder(r * 0.44, w + 0.1).at(0, 0, 0)).rotate_x(90)
    spokes = union(*[box(r * 1.0, 0.045, w * 0.86).rotate_z(a) for a in (0, 60, 120)])
    hubc = cylinder(r * 0.16, w + 0.02).rotate_x(90)
    return tyre - groove, rimc | spokes | hubc


_CAR_STYLES = {
    # greenhouse side profile (x, y) above the belt line; lower body (bottom, top); glass band; pillar x positions
    "hatch": dict(cabin=[(1.0, 0.8), (0.02, 1.42), (-1.45, 1.43), (-1.97, 0.98), (-1.97, 0.8)],
                  lower=(0.3, 0.9), glass=(0.94, 1.33), pillars=(-0.42,), nose=0.16),
    "sport": dict(cabin=[(1.2, 0.74), (-0.08, 1.16), (-0.85, 1.17), (-1.95, 0.84), (-1.95, 0.74)],
                  lower=(0.26, 0.8), glass=(0.85, 1.11), pillars=(-0.5,), nose=0.13),
    "van": dict(cabin=[(1.55, 0.82), (0.95, 1.6), (-1.93, 1.64), (-2.0, 1.5), (-2.0, 0.82)],
                lower=(0.32, 0.92), glass=(0.98, 1.5), pillars=(0.3, -0.75), nose=0.18),
}


def car(color="#e8452c", length=4.2, style="hatch"):
    """Car pointing along +x: painted body with a sloped windshield, wheel arches, tinted glass with pillars,
    four wheels, grille, head- and tail-lights. ``style`` = 'hatch' | 'sport' | 'van'."""
    _check_choice("car", "style", style, tuple(_CAR_STYLES))
    st = _CAR_STYLES[style]
    s = length / 4.2
    W = 1.78
    y0, y1 = st["lower"]
    lower = rounded_box(4.1, y1 - y0, W, st["nose"]).at(0, (y0 + y1) / 2, 0)
    rr = 0.1
    cabin = prism(st["cabin"], W - 0.22 - 2 * rr).round(rr)
    body = smooth_union(lower, cabin, k=0.12)
    arch = cylinder(0.47, W + 0.6).rotate_x(90).at(1.3, 0.37, 0).mirror("x")
    body = smooth_subtract(body, arch, k=0.05)
    g0, g1 = st["glass"]
    band = box(6, g1 - g0, W + 1).at(0, (g0 + g1) / 2, 0)
    for px in st["pillars"]:
        band = band - box(0.1, 3, W + 2).at(px, 1.2, 0)
    glass_ = body & band                              # glass and paint meet flush: no ledge to catch light
    paint = body - band
    tire, rim = _wheel(0.37, 0.26)
    wz = W / 2 - 0.16
    tires = tire.at(1.3, 0.37, wz).mirror("xz")
    rims = rim.at(1.3, 0.37, wz + 0.02).mirror("xz")
    ly = y1 - 0.17
    head = rounded_box(0.06, 0.1, 0.34, 0.04).at(2.04, ly, 0.6).mirror("z")
    tail = rounded_box(0.06, 0.12, 0.3, 0.04).at(-2.04, ly + 0.02, 0.62).mirror("z")
    grille = rounded_box(0.06, 0.14, 0.72, 0.04).at(2.04, ly - 0.03, 0)
    bumper = rounded_box(0.12, 0.1, W - 0.1, 0.05).at(2.02, y0 + 0.08, 0).mirror("x")
    plate = rounded_box(0.04, 0.12, 0.42, 0.02).at(-2.06, y0 + 0.28, 0)
    sill = rounded_box(1.8, 0.06, 0.04, 0.02).at(0.0, y0 + 0.08, W / 2 - 0.01).mirror("z")
    grille = grille | bumper | plate | sill
    return Model("car", [
        Part("body", paint, material("paint", color)),
        Part("windows", glass_, Material(color="#121a22", roughness=0.04, specular=1.0, reflect=0.1)),
        Part("tires", tires, material("rubber")),
        Part("rims", rims, material("metal", "#d4d8de", roughness=0.22)),
        Part("trim", grille | plate, material("plastic", "#2a2c30", roughness=0.45)),
        Part("headlights", head, Material(color="#fff6d8", emissive=1.4, roughness=0.1, specular=1.0)),
        Part("taillights", tail, Material(color="#ff3020", emissive=1.0, roughness=0.1, specular=1.0)),
    ]).scale(s)


def truck(color="#2f6fd0", cargo="#f2f0ea", length=6.5):
    """Box truck pointing along +x: cab, cargo box, six wheels."""
    s = length / 6.5
    W = 2.2
    gc = 0.3
    cab = rounded_box(1.8, 1.9, W, 0.22).at(2.25, gc + 0.95 + 0.25, 0)
    cab = cab.cut(normal=(0.8, 0.6, 0), offset=0.8 * 2.9 + 0.6 * 2.85)
    hood = rounded_box(0.8, 0.9, W - 0.05, 0.16).at(2.95, gc + 0.45 + 0.25, 0)
    cabin = smooth_union(cab, hood, k=0.1)
    win = (cab.round(0.01) & box(6, 0.75, 6).at(0, gc + 1.85, 0)) - box(0.12, 3, 3).at(2.7, 1, 0)
    box_ = rounded_box(4.2, 2.5, W + 0.1, 0.08).at(-0.85, gc + 0.25 + 1.25 + 0.15, 0)
    chassis = box(6.2, 0.25, W - 0.3).at(0, gc + 0.3, 0)
    tire, rim = _wheel(0.48, 0.32)
    tires = tire.at(2.4, 0.48, W / 2 - 0.15).mirror("z") | tire.at(-1.2, 0.48, W / 2 - 0.15).mirror("z") | \
        tire.at(-2.3, 0.48, W / 2 - 0.15).mirror("z")
    rims = rim.at(2.4, 0.48, W / 2 - 0.13).mirror("z") | rim.at(-1.2, 0.48, W / 2 - 0.13).mirror("z") | \
        rim.at(-2.3, 0.48, W / 2 - 0.13).mirror("z")
    head = rounded_box(0.06, 0.16, 0.3, 0.03).at(3.35, gc + 0.6, 0.75).mirror("z")
    return Model("truck", [
        Part("cab", cabin, material("paint", color)),
        Part("cargo", box_, material("plastic", cargo, roughness=0.45)),
        Part("windows", win, Material(color="#1e2a36", roughness=0.03, specular=1.0, reflect=0.25)),
        Part("chassis", chassis, material("plastic", "#2a2c30", roughness=0.6)),
        Part("tires", tires, material("rubber")),
        Part("rims", rims, material("metal", "#cfd3d8")),
        Part("headlights", head, Material(color="#fff6d8", emissive=1.5, roughness=0.1)),
    ]).scale(s)


def rocket(body="#f2f0ec", accent="#e0402a", window="#7fc8f0", height=2.0, flame=False):
    """Retro rocket standing on three fins: ogive body, red nose, porthole, nozzle (``flame=True`` adds fire)."""
    s = height / 2.0
    prof = [(0.0, 0.25), (0.22, 0.25), (0.3, 0.5), (0.33, 0.9), (0.29, 1.3), (0.18, 1.7), (0.0, 2.0)]
    hull = lathe(prof)
    nose = hull & box(1, 0.5, 1).at(0, 1.75 + 0.25, 0)
    band = hull.round(0.006) & box(1, 0.06, 1).at(0, 0.55, 0)
    shell = hull - box(1, 0.5, 1).at(0, 1.75 + 0.25, 0)
    fin2d = prism([(0.2, 0.0), (0.62, 0.0), (0.62, 0.12), (0.28, 0.75), (0.2, 0.8)], 0.05, smooth=False)
    fins = union(*[fin2d.rotate_y(a) for a in (0, 120, 240)])
    port = torus(0.085, 0.022).rotate_x(90).at(0, 1.2, 0.315)
    glass_ = ellipsoid(0.085, 0.085, 0.03).at(0, 1.2, 0.31)
    nozzle = (cone(0.16, 0.12, 0.2) - cone(0.13, 0.09, 0.24).at(0, -0.03, 0)).at(0, 0.17, 0)
    parts = [Part("hull", shell, material("glossy", body)),
             Part("nose", nose | fins, material("glossy", accent)),
             Part("band", band | port, material("metal", "#c9ced6", roughness=0.25)),
             Part("window", glass_, Material(color=window, roughness=0.03, specular=1.0, reflect=0.3, emissive=0.15)),
             Part("nozzle", nozzle, material("metal", "#4a4e56", roughness=0.4))]
    if flame:
        fl = smooth_union(ellipsoid(0.13, 0.32, 0.13).at(0, -0.15, 0), ellipsoid(0.08, 0.2, 0.08).at(0, -0.4, 0),
                          k=0.08)
        parts.append(Part("flame", fl, Material(color="#ffb040", emissive=3.5, shadow=False)))
    m = Model("rocket", parts)
    return (m.at(0, 0.5, 0) if flame else m).scale(s)


def robot(color="#d8dde4", accent="#f0a030", eyes="#6ff0ff", height=1.2):
    """Friendly robot: rounded head with glowing eyes and antenna, boxy torso, capsule arms and legs."""
    s = height / 1.2
    head = rounded_box(0.42, 0.32, 0.34, 0.1).at(0, 1.02, 0)
    torso = rounded_box(0.46, 0.44, 0.3, 0.08).at(0, 0.66, 0)
    neck = cylinder(0.05, 0.08).at(0, 0.86, 0)
    arms = (capsule((0.27, 0.8, 0), (0.33, 0.55, 0.04), 0.045)).mirror("x")
    hands = sphere(0.06).at(0.34, 0.5, 0.05).mirror("x")
    legs = capsule((0.1, 0.42, 0), (0.11, 0.12, 0), 0.055).mirror("x")
    feet = rounded_box(0.14, 0.08, 0.2, 0.035).at(0.11, 0.04, 0.03).mirror("x")
    ant = capsule((0, 1.18, 0), (0, 1.3, 0), 0.01)
    ant_ball = sphere(0.03).at(0, 1.32, 0)
    eye = sphere(0.045).at(0.09, 1.04, 0.16).mirror("x")
    visor = rounded_box(0.34, 0.14, 0.04, 0.05).at(0, 1.04, 0.15)
    chest = cylinder(0.07, 0.03).rotate_x(90).at(0, 0.7, 0.15)
    return Model("robot", [
        Part("body", head | torso | feet, material("plastic", color, roughness=0.28)),
        Part("joints", neck | arms | legs | ant, material("metal", "#8a9099", roughness=0.35)),
        Part("accents", hands | ant_ball | chest, material("glossy", accent)),
        Part("visor", visor, Material(color="#1a1f2a", roughness=0.05, specular=1.0, reflect=0.2)),
        Part("eyes", eye, Material(color=eyes, emissive=2.2)),
    ]).scale(s)


def character(skin="#f2c4a2", shirt="#3f86d8", pants="#384a6a", shoes="#f4f1ea", hair="#4a2f22", height=1.0,
              pose="stand"):
    """Clay-style character facing +z (big head, soft limbs, simple face). ``pose`` = 'stand' | 'wave'."""
    _check_choice("character", "pose", pose, ("stand", "wave"))
    s = height / 1.0
    hy = 0.76
    head = sphere(0.2).at(0, hy, 0)
    ear = ellipsoid(0.03, 0.045, 0.025).at(0.198, hy - 0.01, 0).mirror("x")
    nose = sphere(0.021).at(0, hy - 0.02, 0.196)
    torso = smooth_union(ellipsoid(0.155, 0.19, 0.12).at(0, 0.42, 0), capsule((0, 0.5, 0), (0, 0.58, 0), 0.065),
                         k=0.06)
    if pose == "wave":
        arm_r = capsule((0.13, 0.52, 0), (0.27, 0.68, 0.02), 0.045)
        hand_r = sphere(0.052).at(0.29, 0.72, 0.03)
    else:
        arm_r = capsule((0.13, 0.52, 0), (0.2, 0.31, 0.02), 0.045)
        hand_r = sphere(0.052).at(0.21, 0.28, 0.03)
    arm_l = capsule((-0.13, 0.52, 0), (-0.2, 0.31, 0.02), 0.045)
    hand_l = sphere(0.052).at(-0.21, 0.28, 0.03)
    shirt_s = smooth_union(torso, arm_l, arm_r, k=0.05)
    legs = capsule((0.07, 0.28, 0), (0.075, 0.08, 0.0), 0.056).mirror("x")
    hips = ellipsoid(0.145, 0.08, 0.11).at(0, 0.28, 0)
    pants_s = smooth_union(hips, legs, k=0.05)
    shoe = smooth_union(ellipsoid(0.066, 0.05, 0.1).at(0.08, 0.045, 0.03),
                        sphere(0.048).at(0.08, 0.055, -0.02), k=0.03).cut((0, -1, 0), 0.0).mirror("x")
    cap = sphere(0.213).cut((0.0, -0.62, 0.78), -0.035).at(0, hy + 0.01, -0.012)
    hair_s = smooth_union(cap, sphere(0.07).at(0.07, hy + 0.17, 0.09), sphere(0.065).at(-0.06, hy + 0.18, 0.1),
                          sphere(0.06).at(0.0, hy + 0.2, 0.04), k=0.05)
    eye = ellipsoid(0.026, 0.036, 0.016)
    eyes = eye.at(0.072, hy + 0.035, 0.183).mirror("x")
    shine = sphere(0.008).at(0.079, hy + 0.047, 0.196).mirror("x")
    smile = torus(0.04, 0.0075).rotate_x(90).cut((0, 1, 0), -0.018).at(0, hy - 0.035, 0.187)
    cheeks = ellipsoid(0.03, 0.02, 0.012).at(0.118, hy - 0.04, 0.155).mirror("x")
    clay = dict(roughness=0.7, specular=0.25, subsurface=0.4)
    return Model("character", [
        Part("skin", head | ear | nose | hand_l | hand_r, material("skin", skin, **dict(clay, subsurface=0.55))),
        Part("hair", hair_s, material("clay", hair, **clay)),
        Part("shirt", shirt_s, material("clay", shirt, **clay)),
        Part("pants", pants_s, material("clay", pants, **clay)),
        Part("shoes", shoe, material("clay", shoes, **clay)),
        Part("eyes", eyes, Material(color="#1a1210", roughness=0.25, specular=0.8)),
        Part("eye shine", shine, Material(color="#ffffff", emissive=1.2, shadow=False)),
        Part("mouth", smile, material("clay", "#6a2a26", roughness=0.6, subsurface=0.0)),
        Part("cheeks", cheeks, material("clay", "#f39a8c", roughness=0.8, subsurface=0.5)),
    ]).scale(s)


def quadruped(color="#d9a066", belly="#f4e3c8", kind="dog", height=0.6):
    """Clay-style four-legged animal facing +x. ``kind`` = 'dog' | 'cat' | 'bear' | 'sheep'."""
    _check_choice("quadruped", "kind", kind, ("dog", "cat", "bear", "sheep"))
    s = height / 0.6
    cfg = {"dog": dict(body=(0.32, 0.17, 0.16), leg=0.05, snout=0.09, ear="flop", tail=0.22),
           "cat": dict(body=(0.3, 0.14, 0.13), leg=0.04, snout=0.05, ear="point", tail=0.32),
           "bear": dict(body=(0.36, 0.22, 0.2), leg=0.075, snout=0.08, ear="round", tail=0.05),
           "sheep": dict(body=(0.33, 0.21, 0.19), leg=0.04, snout=0.07, ear="flop", tail=0.06)}[kind]
    bx, by, bz = cfg["body"]
    lr = cfg["leg"]
    top = 0.24 + by
    body = ellipsoid(bx, by, bz).at(0, 0.26 + by * 0.6, 0)
    neck = capsule((bx * 0.6, 0.3 + by, 0), (bx * 0.95, 0.42 + by, 0), 0.08)
    hx, hy = bx * 1.05, 0.48 + by
    head = sphere(0.13 if kind != "bear" else 0.15).at(hx, hy, 0)
    snout = ellipsoid(cfg["snout"] + 0.03, 0.06, 0.07).at(hx + 0.11, hy - 0.04, 0)
    legs = capsule((bx * 0.62, 0.3, bz * 0.55), (bx * 0.64, lr * 0.6, bz * 0.6), lr).mirror("z") | \
        capsule((-bx * 0.62, 0.3, bz * 0.55), (-bx * 0.66, lr * 0.6, bz * 0.6), lr).mirror("z")
    paws = ellipsoid(lr * 1.35, lr * 0.75, lr * 1.15).at(bx * 0.68, lr * 0.62, bz * 0.6).mirror("z") | \
        ellipsoid(lr * 1.35, lr * 0.75, lr * 1.15).at(-bx * 0.62, lr * 0.62, bz * 0.6).mirror("z")
    tail = capsule((-bx * 0.9, 0.26 + by * 1.0, 0), (-bx * 0.9 - cfg["tail"] * 0.6, 0.3 + by + cfg["tail"] * 0.7,
                                                     0), 0.035 if kind != "sheep" else 0.06)
    if cfg["ear"] == "flop":
        ears = ellipsoid(0.04, 0.1, 0.025).rotate_x(-25).at(hx - 0.03, hy + 0.02, 0.12).mirror("z")
    elif cfg["ear"] == "point":
        ears = cone(0.05, 0.005, 0.11).rotate_x(18).at(hx - 0.02, hy + 0.13, 0.065).mirror("z")
    else:
        ears = sphere(0.05).at(hx - 0.03, hy + 0.11, 0.09).mirror("z")
    fur = smooth_union(body, neck, head, legs, tail, k=0.06)
    fur = smooth_union(fur, snout, k=0.03)
    fur = fur | ears | paws
    mat = dict(roughness=0.75, specular=0.2, subsurface=0.35)
    if kind == "sheep":
        wool = smooth_union(*[sphere(0.11).at(x, 0.27 + by * 0.6 + y, z)
                              for x in (-0.2, 0, 0.2) for y in (-0.06, 0.1) for z in (-0.09, 0.09)], k=0.06)
        parts_extra = [Part("wool", wool | sphere(0.09).at(hx - 0.03, hy + 0.07, 0),
                            material("clay", "#f6f2ea", **dict(mat, subsurface=0.5)))]
    else:
        parts_extra = []
    eyes = sphere(0.022).at(hx + 0.08, hy + 0.04, 0.07).mirror("z")
    nose = sphere(0.03).scale((0.8, 0.7, 1.0)).at(hx + 0.11 + cfg["snout"] + 0.02, hy - 0.02, 0)
    belly_s = ellipsoid(bx * 0.8, by * 0.6, bz * 0.9).at(0.02, 0.26 + by * 0.35, 0) & plane((0, 1, 0), 0.26 + by * 0.2)
    return Model(kind, [
        Part("fur", fur, material("clay", color, **mat)),
        Part("belly", belly_s.round(0.003), material("clay", belly, **mat)),
        Part("eyes", eyes, Material(color="#120e0c", roughness=0.08, specular=1.2, reflect=0.1)),
        Part("nose", nose, Material(color="#2a1a18", roughness=0.3, specular=0.6)),
    ] + parts_extra).scale(s)


def laptop(color="#c8ccd2", screen="#3a6ab0", open_deg=105):
    """Open laptop (34 cm wide) facing +z: aluminium base with keyboard, glowing screen."""
    base = rounded_box(0.34, 0.016, 0.235, 0.006).at(0, 0.008, 0)
    keys = box(0.29, 0.004, 0.11).at(0, 0.017, -0.025)
    pad = box(0.1, 0.002, 0.065).at(0, 0.0165, 0.07)
    lid = rounded_box(0.34, 0.01, 0.235, 0.004).at(0, 0.005, 0.1175)
    disp = box(0.31, 0.003, 0.19).at(0, -0.0005, 0.115)
    a = -float(open_deg)
    lid = lid.rotate_x(a).at(0, 0.016, -0.1175)
    disp = disp.rotate_x(a).at(0, 0.016, -0.1175)
    return Model("laptop", [Part("body", base | lid, material("metal", color, roughness=0.32)),
                            Part("keys", keys | pad, material("plastic", "#2a2c30", roughness=0.6)),
                            Part("screen", disp, Material(color=screen, emissive=1.1, roughness=0.05, specular=1.0,
                                                          shadow=True))])


def potted_plant(height=0.6, pot="#c8643a", leaves="#4f9a3c", seed=2):
    """Pot with a leafy plant (leaf blobs on stems)."""
    rng = np.random.default_rng(seed)
    pot_s = lathe([(0.0, 0.0), (0.09, 0.0), (0.12, 0.2), (0.13, 0.22)], thickness=0.012, smooth=False)
    soil = cylinder(0.115, 0.02).at(0, 0.185, 0)
    stems, lv = [], []
    for i in range(9):
        a = i * 2.4 + rng.uniform(-0.3, 0.3)
        r = rng.uniform(0.12, 0.25)
        hgt = rng.uniform(0.32, 0.55)
        tip = np.array([r * math.cos(a), hgt, r * math.sin(a)])
        stems.append(capsule((0, 0.19, 0), tip, 0.006))
        leaf = ellipsoid(0.1, 0.012, 0.05).rotate_z(rng.uniform(15, 35)).rotate_y(-math.degrees(a))
        lv.append(leaf.at(*(tip + np.array([0.05 * math.cos(a), 0.01, 0.05 * math.sin(a)]))))
    s = height / 0.6
    return Model("plant", [Part("pot", pot_s, material("clay", pot, subsurface=0.0, roughness=0.8)),
                           Part("soil", soil, material("matte", "#3a2a20")),
                           Part("stems", union(*stems), material("leaf", "#5a8a3a")),
                           Part("leaves", union(*lv), material("leaf", _noisy(leaves, _shade_hex(leaves, 0.7), 0.1,
                                                                              seed=seed)))]).scale(s)


def rug(width=2.0, depth=1.4, color="#c0563c", border="#f0e2c8", pattern="stripes", shape="rect"):
    """Flat rug with a border. ``shape`` = 'rect' | 'round' (diameter ``width``) | 'oval' (``width`` x
    ``depth``); ``pattern`` = 'stripes' | 'rings' | 'plain' (round rugs default to rings)."""
    _check_choice("rug", "shape", shape, ("rect", "round", "oval"))
    _check_choice("rug", "pattern", pattern, ("stripes", "rings", "plain"))
    if shape != "rect":
        r = float(width) / 2
        sz = 1.0 if shape == "round" else float(depth) / float(width)
        base = cylinder(r, 0.012, round=0.004).at(0, 0.006, 0)
        inner = cylinder(r - 0.1, 0.0125, round=0.003).at(0, 0.0065, 0)
        hole = cylinder(r - 0.1, 0.1)
        if pattern == "plain":
            col = color
        else:
            c1, c2 = _lin(color), _lin(_shade_hex(color, 0.78))
            c3 = _lin(border)

            def col(P, N=None):                     # concentric bands with a thin light ring
                rr = np.hypot(P[:, 0], P[:, 2])
                f = (rr / F32(0.16)) % 1.0
                k = (f > 0.5).astype(F32)[:, None]
                thin = (np.abs(f - 0.5) < 0.05).astype(F32)[:, None]
                return (c1[None] * (1 - k) + c2[None] * k) * (1 - thin) + c3[None] * thin
        m = Model("rug", [Part("border", base - hole, material("fabric", border)),
                          Part("field", inner, material("fabric", col))])
        return m.scale((1.0, 1.0, sz)) if sz != 1.0 else m
    base = rounded_box(width, 0.012, depth, 0.005).at(0, 0.006, 0)
    inner = rounded_box(width - 0.2, 0.0125, depth - 0.2, 0.004).at(0, 0.0065, 0)
    col = stripes(0.12, color, _shade_hex(color, 0.8), "x") if pattern in ("stripes", "rings") else color
    return Model("rug", [Part("border", base - box(width - 0.2, 0.1, depth - 0.2), material("fabric", border)),
                         Part("field", inner, material("fabric", col))])


def toy_block(size=0.3, color="#e8452c", letter=None, letter_color="#ffffff"):
    """Rounded toy cube; ``letter`` adds a raised letter on the front face."""
    b = rounded_box(size, size, size, size * 0.1).at(0, size / 2, 0)
    parts = [Part("block", b, material("plastic", color, roughness=0.35))]
    if letter:
        t = extrude_text(letter, size * 0.55, size * 0.06).at(0, size / 2, size / 2)
        parts.append(Part("letter", t, material("plastic", letter_color, roughness=0.35)))
    return Model("block", parts)


def window_frame(width=1.2, height=1.4, color="#ffffff", glow="#fff4dc", panes=(2, 2), depth=0.12):
    """Window in a wall plane facing +z: frame + mullions + glowing panes (daylight)."""
    outer = box(width, height, depth)
    hole = box(width - 0.12, height - 0.12, depth * 3)
    fr = outer - hole
    nx, ny = panes
    bars = []
    for i in range(1, nx):
        bars.append(box(0.04, height - 0.1, depth * 0.8).at(-width / 2 + 0.06 + (width - 0.12) * i / nx, 0, 0))
    for j in range(1, ny):
        bars.append(box(width - 0.1, 0.04, depth * 0.8).at(0, -height / 2 + 0.06 + (height - 0.12) * j / ny, 0))
    sill = box(width + 0.12, 0.04, depth + 0.1).at(0, -height / 2 - 0.02, 0.05)
    pane = box(width - 0.12, height - 0.12, 0.01).at(0, 0, -depth * 0.2)
    G = _lin(glow)
    hh = max(height - 0.12, 1e-3)

    def daylight(P, N=None):                     # brighter towards the top, with a soft diagonal sheen
        v = np.clip(P[:, 1] / F32(hh) + 0.5, 0, 1)
        s = np.exp(-((P[:, 0] * 0.8 + P[:, 1] - hh * 0.1) / F32(hh * 0.18)) ** 2) * F32(0.12)
        k = (F32(0.72) + F32(0.28) * v + s).astype(F32)
        return G[None] * k[:, None]
    return Model("window", [Part("frame", union(fr, sill, *bars), material("matte", color)),
                            Part("panes", pane, Material(color=daylight, emissive=1.35, shadow=False))])


# ---------------------------------------------------------------------------------------------- interiors, products

def wall(width=4.0, height=2.7, thickness=0.15, openings=()):
    """Wall slab standing on y = 0, centred on x = z = 0, face toward +z, with rectangular holes.

    ``openings`` = [(x, bottom, w, h), ...]: centre x of the hole, its bottom edge and its size
    (a door: ``(1.0, 0, 0.9, 2.1)``; a window: ``(-0.5, 0.9, 1.2, 1.2)``)."""
    s = box(width, height, thickness).at(0, height / 2, 0)
    holes = [box(w, h, thickness * 3).at(x, b + h / 2, 0) for (x, b, w, h) in openings]
    return subtract(s, *holes) if holes else s


def _is_preset(spec):
    """True for a material preset name, a (preset, colour) pair or a Material."""
    if isinstance(spec, Material):
        return True
    if isinstance(spec, str):
        return spec in PRESETS
    return isinstance(spec, (tuple, list)) and len(spec) > 0 and isinstance(spec[0], str) and spec[0] in PRESETS


def _openings(spec, side, default_size, default_bottom):
    """Normalise a window / door spec into [(offset, bottom, w, h)] for one wall side."""
    if spec is None or spec is False:
        return []
    if isinstance(spec, str):
        spec = [(spec, 0.0)]
    elif isinstance(spec, tuple) and spec and isinstance(spec[0], str):
        spec = [spec]
    out = []
    for it in spec:
        it = (it,) if isinstance(it, str) else tuple(it)
        if it[0] != side:
            continue
        off = float(it[1]) if len(it) > 1 else 0.0
        w, h = (float(it[2]), float(it[3])) if len(it) > 3 else default_size
        out.append((off, default_bottom(h), w, h))
    return out


def room(width=4.4, depth=4.4, height=2.7, walls="#efe2cf", walls2=None, floor=None, window="left", door="back",
         window_size=(1.3, 1.3), door_size=(0.9, 2.1), frames="#ffffff", door_color="#8a5a3a", glow="#fff1d8",
         thickness=0.15, skirting=True, view=None, seed=0):
    """Cut-away room for isometric dioramas: floor slab, back and left walls with window and door openings.

    The floor's top is y = 0 and the open interior spans x in [-width/2, width/2], z in [-depth/2, depth/2], so
    furniture stands in it with ``.place(x, z)``. ``window`` / ``door`` = 'left' | 'back' | None, or a list of
    (side, offset) or (side, offset, w, h); offset runs along the wall (world z on the left wall, x on the back).
    ``walls`` colours the left wall, ``walls2`` the back wall; ``floor`` = colour or colour function
    (default: oak ``planks()``). ``view`` = what the windows show: 'day' (default glow), 'rain', 'night city',
    'city', 'garden', 'snow', 'sunset', a colour, or ``f(u, v) -> (N, 3)`` colours 0~1 painted across each pane
    (u left -> right, v bottom -> top, as seen from inside the room). Parts: floor, left wall, back wall, frames,
    panes, doors, knobs, skirting; ``res.window_mask()`` is the visible glass on screen."""
    if view is not None and isinstance(view, str) and not callable(view) and view not in WINDOW_VIEWS:
        try:
            _lin(view)
        except (ValueError, KeyError, TypeError):
            raise _bad_name("window view", view, list(WINDOW_VIEWS)) from None
    W, D, H, t = float(width), float(depth), float(height), float(thickness)
    walls2 = walls2 or _shade_hex(walls, 0.94)
    floor_c = planks() if floor is None else floor
    parts = [Part("floor", box(W + t, 0.3, D + t).at(-t / 2, -0.15, -t / 2),
                  material("wood", floor_c, roughness=0.45) if callable(floor_c) else material("matte", floor_c))]
    win_b = lambda h: max(0.85, (H - h) * 0.55)
    wins = {s: _openings(window, s, window_size, win_b) for s in ("left", "back")}
    drs = {s: _openings(door, s, door_size, lambda h: 0.0) for s in ("left", "back")}
    left_len, back_len = D + t, W + t
    # left wall: built along local x, turned to face +x; local x = -world z
    lw = wall(left_len, H, t, [(-(o + t / 2), b, w, h) for (o, b, w, h) in wins["left"] + drs["left"]])
    lw = lw.rotate_y(90).at(-W / 2 - t / 2, 0, -t / 2)
    bw = wall(back_len, H, t, [(o + t / 2, b, w, h) for (o, b, w, h) in wins["back"] + drs["back"]])
    bw = bw.at(-t / 2, 0, -D / 2 - t / 2)
    parts += [Part("left wall", lw, material("matte", walls)), Part("back wall", bw, material("matte", walls2))]

    def on_wall(shape, side, off, y):
        if side == "left":
            return shape.rotate_y(90).at(-W / 2 - t / 2, y, off)
        return shape.at(off, y, -D / 2 - t / 2)

    frames_, panes, doors, knobs, rects = [], [], [], [], []
    for side in ("left", "back"):
        for (o, b, w, h) in wins[side]:
            wf = window_frame(w + 0.1, h + 0.1, color=frames, glow=glow, depth=t + 0.04)
            frames_.append(on_wall(wf["frame"].shape, side, o, b + h / 2))
            panes.append(on_wall(wf["panes"].shape, side, o, b + h / 2))
            pw, ph = w + 0.1 - 0.12, h + 0.1 - 0.12
            rects.append((side, o, b + h / 2, pw, ph))
        for (o, b, w, h) in drs[side]:
            trim = box(w + 0.16, h + 0.08, t + 0.04).at(0, (h + 0.08) / 2, 0) - box(w, h, t * 3).at(0, h / 2, 0)
            frames_.append(on_wall(trim, side, o, 0))
            leaf = rounded_box(w, h, 0.05, 0.008).at(0, h / 2, t / 2 - 0.04)
            leaf = leaf - box(w * 0.6, h * 0.3, 0.02).at(0, h * 0.72, t / 2 - 0.01).round(0.004) \
                - box(w * 0.6, h * 0.3, 0.02).at(0, h * 0.3, t / 2 - 0.01).round(0.004)
            doors.append(on_wall(leaf, side, o, 0))
            knobs.append(on_wall(sphere(0.035).at(w / 2 - 0.1, h * 0.48, t / 2), side, o, 0))
    if frames_:
        parts.append(Part("frames", union(*frames_), material("matte", frames)))
    if panes:
        pm = wf["panes"].mat
        if view is not None and view != "day":
            pm = Material(color=_pane_view(view, rects, W, D, t, seed), emissive=1.0, roughness=0.1, specular=0.6,
                          shadow=False)
            pm.view_name = view if isinstance(view, str) else "view"
        parts.append(Part("panes", union(*panes), pm))
    if doors:
        parts.append(Part("doors", union(*doors), material("wood", door_color)))
        parts.append(Part("knobs", union(*knobs), material("gold", roughness=0.25)))
    if skirting:
        sk = box(0.03, 0.1, D).at(-W / 2 + 0.015, 0.05, 0) | box(W, 0.1, 0.03).at(0, 0.05, -D / 2 + 0.015)
        cut = [box(w, 0.4, 0.4).at(o, 0.1, -D / 2) for (o, b, w, h) in drs["back"]] + \
              [box(0.4, 0.4, w).at(-W / 2, 0.1, o) for (o, b, w, h) in drs["left"]]
        parts.append(Part("skirting", subtract(sk, *cut) if cut else sk, material("matte", "#ffffff")))
    return Model("room", parts)


def counter(width=2.4, depth=0.65, height=1.0, color="#6e4a32", top="#ece6dc"):
    """Shop / kitchen counter: wooden panelled front, toe kick and a marble top (front faces +z)."""
    t, k = 0.05, 0.1                                          # top thickness, toe-kick height
    bh = height - t - k
    body = box(width, bh, depth - 0.02).at(0, k + bh / 2, -0.01)
    kick = box(width - 0.04, k, depth - 0.1).at(0, k / 2, -0.05)
    n = max(1, int(round(width / 0.6)))
    pw = width / n
    xs = [-width / 2 + pw * (i + 0.5) for i in range(n)]
    zf = depth / 2 - 0.02
    grooves = [box(pw - 0.1, bh - 0.12, 0.03).at(x, k + bh / 2, zf) for x in xs]
    panels = union(*[rounded_box(pw - 0.15, bh - 0.17, 0.03, 0.006).at(x, k + bh / 2, zf - 0.008) for x in xs])
    slab = rounded_box(width + 0.06, t, depth + 0.04, 0.008).at(0, height - t / 2, 0.0)
    return Model("counter", [Part("body", subtract(body, *grooves) | kick | panels, material("wood", color)),
                             Part("top", slab, material("marble", top))])


def coffee_machine(color="#b8322a", steel="#cfd1d4"):
    """Espresso machine: glossy body, steel top with cups warming, two group heads with portafilters, a gauge
    and a drip tray (front faces +z; 55 cm wide)."""
    st = material("metal", steel, roughness=0.22)
    body = rounded_box(0.55, 0.36, 0.42, 0.035).at(0, 0.24, 0)
    plinth = rounded_box(0.57, 0.06, 0.45, 0.01).at(0, 0.03, 0.01)
    top = box(0.5, 0.025, 0.38).at(0, 0.435, 0) | (box(0.5, 0.05, 0.38) - box(0.46, 0.1, 0.34)).at(0, 0.46, 0)
    heads = union(*[cylinder(0.04, 0.05).at(x, 0.23, 0.23) for x in (-0.13, 0.13)])
    pf = union(*[(cylinder(0.038, 0.03).at(x, 0.19, 0.24) | capsule((x, 0.19, 0.26), (x, 0.17, 0.38), 0.012))
                 for x in (-0.13, 0.13)])
    tray = box(0.46, 0.02, 0.14).at(0, 0.07, 0.25) - box(0.42, 0.02, 0.1).at(0, 0.08, 0.25)
    wand = capsule((0.24, 0.3, 0.2), (0.25, 0.12, 0.27), 0.006)
    gauge = cylinder(0.045, 0.015).rotate_x(90).at(0, 0.34, 0.215)
    cups = union(*[cylinder(0.03, 0.05, round=0.006).at(x, 0.47, z) for x, z in ((-0.1, 0.05), (0.05, -0.05),
                                                                                  (0.14, 0.07))])
    return Model("coffee machine", [Part("body", body, material("paint", color)),
                                    Part("steel", plinth | top | heads | pf | tray | wand, st),
                                    Part("gauge", gauge, material("ceramic", "#f4f1ea")),
                                    Part("cups", cups, material("ceramic", "#ffffff"))])


def pendant_lamp(height=2.7, drop=0.9, color="#2b2b2e", glow="#ffd9a0", shade="dome"):
    """Hanging lamp: a cord from the ceiling at ``height`` down ``drop`` metres to a shade with a glowing bulb.
    ``shade`` = 'dome' | 'cone' | 'globe'. Pair with ``sc.light_from(lamp["bulb"])``."""
    _check_choice("pendant_lamp", "shade", shade, ("dome", "cone", "globe"))
    y = height - drop
    cord = cylinder(0.006, drop).at(0, height - drop / 2, 0)
    if shade == "globe":
        sh = sphere(0.14).shell(0.004).at(0, y - 0.12, 0)
        bulb = sphere(0.05).at(0, y - 0.12, 0)
        sm = material("glass", "#fff6e6", glass=0.5, emissive=0.6)
        return Model("pendant lamp", [Part("cord", cord, material("rubber", "#1a1a1a")), Part("shade", sh, sm),
                                      Part("bulb", bulb, material("emissive", glow))], hangs=True)
    if shade == "cone":
        sh = (cone(0.16, 0.03, 0.22) - cone(0.15, 0.022, 0.23).at(0, -0.006, 0)).at(0, y - 0.11, 0)
    else:
        sh = lathe([(0.02, 0.0), (0.06, -0.01), (0.15, -0.09), (0.2, -0.2)], thickness=0.012).at(0, y, 0)
    bulb = sphere(0.04).at(0, y - 0.13, 0)
    return Model("pendant lamp", [Part("cord", cord, material("rubber", "#1a1a1a")),
                                  Part("shade", sh, material("paint", color, roughness=0.35)),
                                  Part("bulb", bulb, material("emissive", glow))], hangs=True)


def cup(color="#f4f1ea", coffee=True, saucer=True, size=1.0):
    """Coffee cup on a saucer (7 cm tall cup, 15 cm saucer); ``coffee`` adds the drink."""
    prof = [(0.0, 0.0), (0.024, 0.0), (0.03, 0.008), (0.04, 0.035), (0.044, 0.065), (0.045, 0.07)]
    y0 = 0.012 if saucer else 0.0
    body = lathe(prof, thickness=0.003).at(0, y0, 0)
    handle = torus(0.018, 0.005).rotate_x(90).scale((1.0, 1.2, 1.0)).at(0.05, y0 + 0.042, 0)
    handle = handle.cut(normal=(-1, 0, 0), offset=-0.04)
    parts = [Part("cup", body | handle, material("ceramic", color))]
    if saucer:
        parts.append(Part("saucer", lathe([(0.0, 0.0), (0.05, 0.0), (0.072, 0.008), (0.076, 0.013)],
                                          thickness=0.004), material("ceramic", color)))
    if coffee:
        parts.append(Part("coffee", cylinder(0.04, 0.004).at(0, y0 + 0.056, 0),
                          Material(color="#3a1d0e", roughness=0.15, specular=0.7)))
    return Model("cup", parts).scale(size)


def speaker(color="#1d1d20", fabric=None, height=0.19, kind="round"):
    """Smart speaker: rounded body with a woven fabric band and a top control ring. ``kind`` = 'round'
    (cylinder) | 'pill' (capsule) | 'box' (bookshelf speaker with two drivers)."""
    _check_choice("speaker", "kind", kind, ("round", "pill", "box"))
    fab = fabric or _shade_hex(color, 1.25)
    if kind == "box":
        w, h, d = height * 0.62, height, height * 0.8
        cab = rounded_box(w, h, d, 0.012).at(0, h / 2, 0)
        grille = rounded_box(w - 0.016, h - 0.02, 0.01, 0.008).at(0, h / 2, d / 2 + 0.002)
        drv = (cylinder(w * 0.36, 0.01).rotate_x(90).at(0, h * 0.36, d / 2 + 0.006)
               | cylinder(w * 0.14, 0.01).rotate_x(90).at(0, h * 0.78, d / 2 + 0.006))
        return Model("speaker", [Part("body", cab, material("wood", "#6b4a32")),
                                 Part("grille", grille - drv.scale((1, 1, 1)), material("fabric", fab)),
                                 Part("drivers", drv, material("rubber", color, roughness=0.5))])
    r = height * 0.29
    if kind == "pill":
        body = capsule((0, r, 0), (0, height - r, 0), r)
    else:
        body = cylinder(r, height, round=r * 0.45).at(0, height / 2, 0)
    band = (cylinder(r + 0.0015, height * 0.62).at(0, height * 0.45, 0) & body.round(0.0015))
    ring = torus(r * 0.4, 0.0022).at(0, height - 0.0004, 0)
    return Model("speaker", [Part("body", body, material("rubber", color, roughness=0.6, specular=0.35)),
                             Part("fabric", band, material("fabric", fab)),
                             Part("ring", ring, material("emissive", "#9fd8ff", emissive=1.2))])


# ===================================================================================================== window views

WINDOW_VIEWS = ("day", "rain", "night city", "city", "garden", "snow", "sunset")


def _h1(i, seed):
    i = np.asarray(i, np.int64)
    z = np.zeros_like(i)
    return _hash3(i, z, z, seed)


def _vgrad(v, stops):
    """Vertical gradient: stops = [(v, '#hex'), ...] -> (N, 3) display colours."""
    vs = np.array([s[0] for s in stops], F32)
    cs = np.array([rgb(s[1]) for s in stops], F32)
    return np.stack([np.interp(v, vs, cs[:, k]) for k in range(3)], -1).astype(F32)


def _skyline(xm, ym, seed, base=0.0, hmin=0.15, hmax=0.55, wmin=0.12, wmax=0.3):
    """Building silhouettes along a pane (metres): (inside mask, building index, local x, local y)."""
    cell = (wmin + wmax) / 2
    ci = np.floor(xm / cell).astype(np.int64)
    hgt = base + hmin + (hmax - hmin) * _h1(ci, seed)
    gap = 0.01 + 0.03 * _h1(ci, seed + 1)
    lx = xm - ci * cell
    inside = (ym < hgt) & (lx > gap) & (lx < cell - gap)
    return inside, ci, lx, ym


def _view_colors(view, u, v, pw, ph, seed):
    """Display colours (N, 3) of a window view at pane coordinates u, v in 0~1 (pane size pw x ph metres)."""
    xm, ym = u * pw, v * ph
    n = len(u)
    if callable(view):
        out = np.asarray(view(u, v), F32)
        return np.broadcast_to(out, (n, 3)) if out.ndim == 1 else out.reshape(n, 3)
    if view not in WINDOW_VIEWS:
        c = rgb(view)
        return (c[None] * (F32(0.92) + F32(0.12) * v[:, None])).astype(F32)
    if view == "rain":
        col = _vgrad(v, [(0, "#9aa7b4"), (0.55, "#aab6c2"), (1, "#8794a3")])
        inside, ci, lx, _ = _skyline(xm, ym, seed, base=0.0, hmin=0.12, hmax=0.42)
        col = np.where(inside[:, None], col * F32(0.82) + F32(0.02), col)
        a = xm * 38 + ym * 8                                   # slanted streaks
        k = np.floor(a)
        on = _h1(k, seed + 3) > 0.62
        ph_ = _h1(k, seed + 4)
        fy = (ym * 2.2 + ph_) % 1.0
        streak = on & (np.abs(a - k - 0.5) < 0.16) & (fy < 0.35)
        col = np.where(streak[:, None], col + F32(0.13), col)
        gx, gy = np.floor(xm * 22), np.floor(ym * 22)       # drops on the glass
        r = _hash3(gx.astype(np.int64), gy.astype(np.int64), np.zeros(n, np.int64), seed + 5)
        cx, cy = (gx + 0.3 + 0.4 * r) / 22, (gy + 0.3 + 0.4 * _h1(gx.astype(np.int64) * 7 + 1, seed)) / 22
        dd = np.hypot(xm - cx, ym - cy) * 22
        drop = (r > 0.7) & (dd < 0.22)
        col = np.where(drop[:, None], col * F32(0.86) + F32(0.12) * (dd < 0.1)[:, None], col)
        return col
    if view in ("night city", "city"):
        night = view == "night city"
        col = _vgrad(v, [(0, "#2c3566"), (0.5, "#18204a"), (1, "#0b1230")] if night else
                     [(0, "#dfeaf3"), (0.6, "#b8d4ec"), (1, "#8fbbe3")])
        if night:
            st = (_hash3(np.floor(xm * 90).astype(np.int64), np.floor(ym * 90).astype(np.int64),
                         np.zeros(n, np.int64), seed) > 0.995) & (v > 0.45)
            col = np.where(st[:, None], np.array([0.95, 0.95, 1.0], F32), col)
        far, fi, _, _ = _skyline(xm + 0.07, ym, seed + 9, hmin=0.25, hmax=0.6, wmin=0.08, wmax=0.18)
        col = np.where(far[:, None], np.array(rgb("#232a52" if night else "#a9b9c8"), F32), col)
        near, ci, lx, ly = _skyline(xm, ym, seed, hmin=0.12, hmax=0.48)
        bcol = np.array(rgb("#121831" if night else "#7d8c9c"), F32) * \
            (F32(0.85) + F32(0.3) * _h1(ci, seed + 2))[:, None]
        wx, wy = np.floor(lx / 0.035), np.floor(ly / 0.05)
        win = ((lx % 0.035) > 0.012) & ((ly % 0.05) > 0.02) & (ly > 0.03)
        lit = _hash3(wx.astype(np.int64) + ci * 31, wy.astype(np.int64), np.zeros(n, np.int64), seed + 7)
        wcol = np.where((lit > (0.55 if night else 0.0))[:, None],
                        np.array(rgb("#ffd27a" if night else "#d6e4f0"), F32), bcol * F32(0.8))
        col = np.where(near[:, None], np.where(win[:, None], wcol, bcol), col)
        return col
    if view in ("garden", "snow"):
        snow = view == "snow"
        col = _vgrad(v, [(0, "#eef2f5"), (1, "#c9d4de")] if snow else [(0, "#e6f2f8"), (1, "#9fcdee")])
        hill = 0.22 + 0.05 * np.sin(xm * 4.1 + seed) + 0.03 * np.sin(xm * 9.7 + 1.3 * seed)
        trees = 0.42 + 0.09 * np.abs(np.sin(xm * 7.3 + seed * 0.7)) + 0.05 * np.sin(xm * 17 + 2.1)
        tn = noise3(xm * 9, ym * 9, np.full(n, F32(seed)), seed) - 0.5
        tcol = np.array(rgb("#3f7044"), F32) * (F32(0.85) + F32(0.4) * tn)[:, None]
        if snow:
            tcol = tcol * F32(0.6) + F32(0.35)
        col = np.where((ym < trees * min(ph, 1.2) * 0.9)[:, None], tcol, col)
        gcol = np.array(rgb("#f4f6f8" if snow else "#86b85a"), F32) * \
            (F32(0.92) + F32(0.16) * noise3(xm * 4, ym * 30, np.zeros(n, F32), seed + 1))[:, None]
        col = np.where((ym < hill * min(ph, 1.2) * 0.6)[:, None], gcol, col)
        if snow:
            fl = _hash3(np.floor(xm * 60).astype(np.int64), np.floor(ym * 60).astype(np.int64),
                        np.zeros(n, np.int64), seed + 4) > 0.985
            col = np.where(fl[:, None], np.array([1.0, 1.0, 1.0], F32), col)
        return col
    if view == "sunset":
        col = _vgrad(v, [(0, "#ffcf8a"), (0.35, "#f79b5f"), (0.7, "#c8607a"), (1, "#5b4a7c")])
        sd = np.hypot(xm - pw * 0.6, ym - ph * 0.32)
        col = col + (np.clip(1 - sd / (0.09 * ph), 0, 1) ** 0.5)[:, None] * F32(0.5)
        inside, _, _, _ = _skyline(xm, ym, seed, hmin=0.08, hmax=0.3)
        return np.where(inside[:, None], np.array(rgb("#3b2a45"), F32), col)
    # day: the soft daylight of window_frame
    return _vgrad(v, [(0, "#f6f1e6"), (1, "#fffaf0")])


def _pane_view(view, rects, W, D, t, seed=0):
    """Colour function for room() panes: finds the pane each point lies on and paints the view across it."""
    rects = list(rects)

    def fn(P, N=None):
        n = len(P)
        best = np.full(n, np.inf, F32)
        u = np.zeros(n, F32)
        v = np.zeros(n, F32)
        pwa = np.ones(n, F32)
        pha = np.ones(n, F32)
        for side, o, cy, pw, ph in rects:
            if side == "left":
                dist = np.abs(P[:, 0] - F32(-W / 2 - t / 2))
                uu = (F32(o) - P[:, 2] + F32(pw / 2)) / F32(pw)
            else:
                dist = np.abs(P[:, 2] - F32(-D / 2 - t / 2))
                uu = (P[:, 0] - F32(o) + F32(pw / 2)) / F32(pw)
            vv = (P[:, 1] - F32(cy - ph / 2)) / F32(ph)
            dist = dist + (np.abs(uu - 0.5) > 0.6) * F32(10)
            b = dist < best
            best = np.where(b, dist, best)
            u = np.where(b, uu, u)
            v = np.where(b, vv, v)
            pwa = np.where(b, F32(pw), pwa)
            pha = np.where(b, F32(ph), pha)
        u, v = np.clip(u, 0, 1), np.clip(v, 0, 1)
        if len(rects) == 1 or np.all(pwa == pwa[0]):
            col = _view_colors(view, u, v, float(pwa[0]) if n else 1.0, float(pha[0]) if n else 1.0, seed)
        else:
            col = np.zeros((n, 3), F32)
            for pw_, ph_ in {(float(a), float(b)) for a, b in zip(pwa, pha)}:
                k = (pwa == pw_) & (pha == ph_)
                col[k] = _view_colors(view, u[k], v[k], pw_, ph_, seed)
        return np.power(np.clip(col, 0, None), F32(2.2)).astype(F32)
    return fn


# ===================================================================================================== more prefabs

def armchair(color="#c4623f", width=0.9, legs="#5a3b26", cushions=None):
    """Upholstered armchair: frame, seat cushion, back cushion, two padded armrests and short wooden legs.
    Parts: frame, seat, back, armrest_l, armrest_r, legs (armrest_l is the sitter's left = +x, on the viewer's
    right in a front view). Seat top at 0.47 m, armrest tops at 0.64 m for the default width."""
    s = float(width) / 0.9
    W, D = 0.9, 0.86
    fab = material("fabric", color)
    cush = material("fabric", cushions or _shade_hex(color, 1.1))
    base = rounded_box(W - 0.04, 0.28, D - 0.04, 0.07).at(0, 0.08 + 0.14, 0)
    back = rounded_box(W - 0.04, 0.66, 0.2, 0.09).rotate_x(-7).at(0, 0.62, -D / 2 + 0.11)
    arm = rounded_box(0.19, 0.38, D - 0.02, 0.085)
    roll = capsule((0, 0, -(D - 0.02) / 2 + 0.06), (0, 0, (D - 0.02) / 2 - 0.05), 0.105)
    arm_l = smooth_union(arm.at(0, 0.45, 0), roll.at(0, 0.54, 0.0), k=0.04).at(W / 2 - 0.1, 0, 0)
    arm_r = smooth_union(arm.at(0, 0.45, 0), roll.at(0, 0.54, 0.0), k=0.04).at(-W / 2 + 0.1, 0, 0)
    seat = rounded_box(W - 0.4, 0.13, D - 0.24, 0.055).at(0, 0.405, 0.07)
    backc = rounded_box(W - 0.42, 0.44, 0.15, 0.07).rotate_x(-10).at(0, 0.68, -D / 2 + 0.27)
    leg = cone(0.026, 0.018, 0.08).at(W / 2 - 0.09, 0.04, D / 2 - 0.09).mirror("xz")
    m = Model("armchair", [Part("frame", smooth_union(base, back, k=0.03), fab), Part("seat", seat, cush),
                           Part("back", backc, cush), Part("armrest_l", arm_l, fab), Part("armrest_r", arm_r, fab),
                           Part("legs", leg, material("wood", legs))])
    return m.scale(s) if abs(s - 1) > 1e-9 else m


def floor_lamp(color="#2e2b29", shade="#f1e2c6", glow="#ffd9a0", height=1.6, style="drum", lit=True):
    """Floor lamp: weighted base, thin pole, fabric shade (``style`` = 'drum' | 'cone' | 'globe') and a bulb.
    ``lit=True`` makes the shade glow; light the room with ``sc.light_from(lamp)`` (uses the bulb).
    Parts: base, pole, shade, bulb."""
    _check_choice("floor_lamp", "style", style, ("drum", "cone", "globe"))
    H = float(height)
    metal = material("metal", color, roughness=0.35)
    base = cylinder(0.16, 0.03, round=0.012).at(0, 0.015, 0)
    pole = cylinder(0.012, H - 0.2).at(0, 0.03 + (H - 0.2) / 2, 0)
    by = H - 0.2
    if style == "globe":
        sh = sphere(0.17).at(0, by + 0.02, 0) - cylinder(0.05, 0.1).at(0, by - 0.16, 0)
        sh = sh.shell(0.006)
    elif style == "cone":
        sh = lathe([(0.21, by - 0.14), (0.09, by + 0.16)], smooth=False, thickness=0.006)
    else:
        sh = lathe([(0.2, by - 0.15), (0.18, by + 0.15)], smooth=False, thickness=0.006)
    gl = _lin(glow)
    sc_ = _lin(shade)

    def shade_col(P, N=None):                 # brighter low on the shade, where the bulb shines through
        k = np.clip(1.0 - (P[:, 1] - F32(by - 0.15)) / F32(0.4), 0.55, 1.0)
        return sc_[None] * k[:, None] + gl[None] * (F32(0.15) * k)[:, None]
    smat = Material(color=shade_col if lit else shade, roughness=0.9, specular=0.1, subsurface=0.6,
                    sss_color=glow, emissive=0.55 if lit else 0.0, shadow=0.45, sheen=0.4)
    bulb = sphere(0.05).at(0, by - 0.02, 0)
    return Model("floor lamp", [Part("base", base, metal), Part("pole", pole, metal), Part("shade", sh, smat),
                                Part("bulb", bulb, material("emissive", glow, emissive=2.5 if lit else 0.3))])


def side_table(color=None, height=0.55, width=0.45, round_top=True, shelf=True):
    """Small side table: top, three splayed legs (round) or four (square) and an optional lower shelf.
    Parts: top, legs, shelf."""
    wm = material("wood") if color is None else material("wood", color)
    H, r = float(height), float(width) / 2
    t = 0.028
    if round_top:
        top = cylinder(r, t, round=0.008).at(0, H - t / 2, 0)
        legs = [capsule((math.cos(a) * r * 0.62, 0.012, math.sin(a) * r * 0.62),
                        (math.cos(a) * r * 0.42, H - t, math.sin(a) * r * 0.42), 0.016)
                for a in np.radians([90, 210, 330])]
        low = cylinder(r * 0.6, 0.02, round=0.006).at(0, H * 0.3, 0)
    else:
        top = rounded_box(2 * r, t, 2 * r, 0.008).at(0, H - t / 2, 0)
        legs = [rounded_box(0.035, H - t, 0.035, 0.006).at(r - 0.04, (H - t) / 2, r - 0.04).mirror("xz")]
        low = rounded_box(2 * r - 0.06, 0.02, 2 * r - 0.06, 0.006).at(0, H * 0.3, 0)
    parts = [Part("top", top, wm), Part("legs", union(*legs), wm)]
    if shelf:
        parts.append(Part("shelf", low, wm))
    return Model("side table", parts)


def _bud(color, tip, mesh, flip=False):
    """One earbud standing on its stem tip: head, stem, silicone tip, speaker mesh (stem bottom at y = 0)."""
    sx = -1.0 if flip else 1.0                  # the tip points inward (toward the other bud) and forward
    head = ellipsoid(0.0098, 0.0088, 0.0094).at(0, 0.034, 0)
    stem = capsule((0, 0.0032, 0.0015), (0, 0.03, 0.0), 0.0035)
    body = smooth_union(head, stem, k=0.004)
    tp = ellipsoid(0.0046, 0.0042, 0.0046).at(-sx * 0.0068, 0.0352, 0.0055)
    msh = ellipsoid(0.0012, 0.0034, 0.0026).at(sx * 0.0094, 0.0352, 0.001)
    cap = cylinder(0.0033, 0.0012, round=0.0005).at(0, 0.0004, 0.0015)
    return body, tp, msh | cap


def earbuds(color="#f6f6f4", tips="#e3e5e8", mesh="#3a3d42", spacing=0.03):
    """A pair of wireless earbuds standing side by side (stems down, tips facing out).
    Parts: earbud_l, earbud_r, tips, mesh."""
    lb, lt, lm = _bud(color, tips, mesh, flip=False)
    rb, rt, rm = _bud(color, tips, mesh, flip=True)
    o = float(spacing) / 2
    gl = material("glossy", color, roughness=0.2)
    return Model("earbuds", [Part("earbud_l", lb.at(o, 0, 0), gl), Part("earbud_r", rb.at(-o, 0, 0), gl),
                             Part("tips", lt.at(o, 0, 0) | rt.at(-o, 0, 0), material("rubber", tips, roughness=0.6)),
                             Part("mesh", lm.at(o, 0, 0) | rm.at(-o, 0, 0), material("rubber", mesh))])


def charging_case(color="#f6f6f4", open=100, buds=True, light="#5be37a", bud_color=None):
    """Earbud charging case: rounded base with two wells, hinged lid opened ``open`` degrees (0 = shut), a
    status light and the earbuds sitting in their wells (``buds=False`` for an empty case).
    Parts: case, lid, hinge, light, wells, earbud_l, earbud_r, tips, mesh."""
    W, H, D = 0.062, 0.03, 0.024
    mat = material("glossy", color, roughness=0.24)
    base = rounded_box(W, H, D, 0.0105).at(0, H / 2, 0)
    wells = [capsule((sx * 0.0145, H + 0.004, 0), (sx * 0.0145, H - 0.022, 0), 0.0072) for sx in (-1, 1)]
    case = subtract(base, *wells)
    floor_ = union(*[cylinder(0.0068, 0.0012).at(sx * 0.0145, H - 0.0225, 0) for sx in (-1, 1)])
    lid = rounded_box(W, 0.012, D, 0.0105).at(0, 0.006, 0) - rounded_box(W - 0.003, 0.012, D - 0.003, 0.009).at(
        0, 0.0005, 0)
    lid = lid.at(0, 0, D / 2).rotate_x(-float(open)).at(0, H + 0.0002, -D / 2)
    hinge = cylinder(0.0021, 0.012).rotate_z(90).at(0, H, -D / 2 - 0.0004)
    led = ellipsoid(0.0016, 0.0016, 0.0007).at(0, H * 0.58, D / 2 + 0.0002)
    parts = [Part("case", case, mat), Part("lid", lid, mat), Part("hinge", hinge, material("chrome")),
             Part("light", led, material("emissive", light, emissive=2.0)),
             Part("wells", floor_, material("matte", _shade_hex(color, 0.82)))]
    if buds:
        e = earbuds(bud_color or color, spacing=0.029)
        for p in e.parts:                                   # stems down into the wells
            parts.append(Part(p.name, p.shape.at(0, H - 0.02, 0), p.mat))
    return Model("charging case", parts)


def _screen_fn(screen, w, h):
    if callable(screen):
        def fn(P, N=None):
            u = np.clip(P[:, 0] / F32(w) + 0.5, 0, 1)
            v = np.clip(P[:, 1] / F32(h) + 0.5, 0, 1)
            out = np.asarray(screen(u, v), F32)
            out = np.broadcast_to(out, (len(P), 3)) if out.ndim == 1 else out.reshape(len(P), 3)
            return np.power(np.clip(out, 0, None), F32(2.2))
        return fn
    c0 = rgb(screen)
    c1 = np.clip(c0 * 0.45 + np.array([0.35, 0.12, 0.38]), 0, 1)

    def wall(P, N=None):                       # wallpaper: diagonal gradient with a soft glow
        u = np.clip(P[:, 0] / F32(w) + 0.5, 0, 1)
        v = np.clip(P[:, 1] / F32(h) + 0.5, 0, 1)
        k = np.clip(0.6 * v + 0.4 * (1 - u), 0, 1)
        col = c1[None] * (1 - k[:, None]) + c0[None] * k[:, None]
        g = np.exp(-((u - 0.3) ** 2 + (v - 0.7) ** 2) / 0.08)
        return np.power(np.clip(col + g[:, None] * F32(0.15), 0, 1), F32(2.2)).astype(F32)
    return wall


def smartphone(color="#1d1d20", screen="#3a6ab0", lying=False, frame="#9a9da3"):
    """Smartphone standing upright, screen toward +z (``lying=True``: flat on its back, screen up).
    ``screen`` = a colour (wallpaper gradient) or ``f(u, v) -> (N, 3)`` colours 0~1 across the display.
    Parts: body, screen, camera, buttons."""
    w, h, d = 0.0715, 0.147, 0.0079
    body = rounded_box(w, h, d, 0.0038)
    body = body & rounded_box(w, h, d * 3, 0.0095)
    scr_w, scr_h = w - 0.005, h - 0.005
    scr = (rounded_box(scr_w, scr_h, 0.0006, 0.0003) & rounded_box(scr_w, scr_h, 0.01, 0.008)).at(0, 0, d / 2)
    smat = Material(color=_screen_fn(screen, scr_w, scr_h), roughness=0.05, specular=1.0, reflect=0.05,
                    emissive=1.0)
    cam = (rounded_box(0.026, 0.026, 0.0016, 0.006).at(-w / 2 + 0.019, h / 2 - 0.019, -d / 2 - 0.0007)
           | cylinder(0.0055, 0.002).rotate_x(90).at(-w / 2 + 0.013, h / 2 - 0.013, -d / 2 - 0.0016)
           | cylinder(0.0055, 0.002).rotate_x(90).at(-w / 2 + 0.013, h / 2 - 0.025, -d / 2 - 0.0016))
    btn = (rounded_box(0.0012, 0.02, 0.0028, 0.0005).at(w / 2 + 0.0004, h * 0.18, 0)
           | rounded_box(0.0012, 0.012, 0.0028, 0.0005).at(-w / 2 - 0.0004, h * 0.22, 0)
           | rounded_box(0.0012, 0.012, 0.0028, 0.0005).at(-w / 2 - 0.0004, h * 0.12, 0))
    m = Model("smartphone", [Part("body", body, material("glossy", color, roughness=0.18)),
                             Part("screen", scr, smat), Part("camera", cam, material("glossy", "#141416")),
                             Part("buttons", btn, material("metal", frame))])
    m = m.rotate_x(-90).at(0, 0, 0) if lying else m
    return m.place()


def _watch_face(face, w, h):
    acc = rgb(face)

    def fn(P, N=None):
        u = P[:, 0] / F32(w)
        v = P[:, 1] / F32(h)
        r = np.hypot(u, v)
        a = np.arctan2(u, v)
        col = np.zeros((len(P), 3), F32) + F32(0.02)
        ring = np.abs(r - 0.36) < 0.035
        prog = (a % (2 * np.pi)) < 2 * np.pi * 0.72
        col = np.where((ring & prog)[:, None], acc[None], col)
        col = np.where((ring & ~prog)[:, None], acc[None] * F32(0.25), col)
        tick = (np.abs(r - 0.43) < 0.02) & (np.abs(((a / (2 * np.pi) * 12) % 1.0) - 0.5) > 0.42)
        col = np.where(tick[:, None], F32(0.85), col)
        hand = ((np.abs(u) < 0.012) & (v > 0) & (v < 0.24)) | ((np.abs(v + u * 0.0) < 0.012) & (u > 0) & (u < 0.17))
        col = np.where(hand[:, None], F32(0.95), col)
        return np.power(np.clip(col, 0, 1), F32(2.2)).astype(F32)
    return fn


def smartwatch(case="#cfd2d6", band="#2b2d31", face="#5ad1ff", screen=None):
    """Smartwatch standing on its looped band, face toward +z: rounded case, glass display with a watch face
    (``face`` = accent colour, or ``screen=f(u, v)``), side crown and a soft sport band. Parts: case, screen,
    crown, band."""
    w, h, d = 0.041, 0.048, 0.0105
    R = 0.03
    cy = 2 * R + 0.0035
    body = rounded_box(w, h, d, 0.0045) & rounded_box(w, h, d * 4, 0.011)
    scr = (rounded_box(w - 0.004, h - 0.004, 0.0008, 0.0003) & rounded_box(w - 0.004, h - 0.004, 0.02, 0.009))
    smat = Material(color=_screen_fn(screen, w - 0.004, h - 0.004) if screen is not None else
                    _watch_face(face, w - 0.004, h - 0.004), roughness=0.04, specular=1.0, reflect=0.06, emissive=1.0)
    crown = cylinder(0.0035, 0.004, round=0.001).rotate_z(90).at(w / 2 + 0.0015, 0.006, 0)
    loop = torus(R, 0.0028).rotate_z(90).scale((3.6, 1.0, 1.0)).at(0, R + 0.0028, 0)
    loop = loop - box(0.04, 0.03, 0.02).at(0, 2 * R, R)          # opening where the case sits
    off = np.array([0, cy - h / 2 + 0.002, R - 0.002])
    return Model("smartwatch", [Part("case", body.at(*off), material("metal", case, roughness=0.18)),
                                Part("screen", scr.at(*(off + [0, 0, d / 2]))   , smat),
                                Part("crown", crown.at(*off), material("metal", case, roughness=0.25)),
                                Part("band", loop, material("rubber", band, roughness=0.55))])


def tent(color="#e0803a", width=2.0, depth=2.3, height=1.35, lit=False, glow="#ffb257", floor="#5a4632"):
    """Ridge tent with an open door, rolled flaps, poles and a ground sheet. ``lit=True`` glows from inside
    (the canvas lights up warm, brightest low and near the door); add the light with
    ``sc.light_from(t["lamp"], color=glow)``. Parts: canvas, floor, flaps, poles, lamp (when lit)."""
    w, d, h = float(width), float(depth), float(height)
    tri = [(-w / 2, 0.0), (w / 2, 0.0), (0.0, h)]
    outer = prism(tri, depth=d).at(0, 0, 0)
    inner = prism([(-w / 2 + 0.04, 0.0), (w / 2 - 0.04, 0.0), (0.0, h - 0.06)], depth=d - 0.04).at(0, -0.02, 0)
    door = prism([(-w * 0.28, -0.05), (w * 0.28, -0.05), (0.0, h * 0.82)], depth=0.3).at(0, 0, d / 2)
    canvas = (outer - inner) - door
    flaps = (capsule((-w * 0.29, 0.05, d / 2 + 0.02), (-0.03, h * 0.8, d / 2 + 0.02), 0.035)
             | capsule((w * 0.29, 0.05, d / 2 + 0.02), (0.03, h * 0.8, d / 2 + 0.02), 0.035))
    poles = (cylinder(0.015, h + 0.12).at(0, (h + 0.12) / 2, d / 2 + 0.03)
             | cylinder(0.015, h + 0.12).at(0, (h + 0.12) / 2, -d / 2 - 0.03))
    sheet = box(w - 0.06, 0.01, d - 0.06).at(0, 0.005, 0)
    base = _lin(color)
    gl = _lin(glow)

    def glow_col(P, N=None):                   # lantern light shining through the fabric
        k = np.exp(-(P[:, 1] / F32(h * 0.7)) ** 2) * np.clip(0.6 + 0.4 * P[:, 2] / F32(d / 2), 0.2, 1)
        return base[None] * (1 - k[:, None] * F32(0.5)) + gl[None] * (k[:, None] * F32(0.9))
    cmat = Material(color=glow_col if lit else color, roughness=0.85, specular=0.1, subsurface=0.5,
                    sss_color=glow if lit else None, emissive=0.9 if lit else 0.0, sheen=0.3)
    parts = [Part("canvas", canvas, cmat), Part("floor", sheet, material("fabric", floor)),
             Part("flaps", flaps, material("fabric", _shade_hex(color, 0.85))),
             Part("poles", poles, material("metal", "#6a6e74"))]
    if lit:
        parts.append(Part("lamp", sphere(0.06).at(0, 0.45, 0.1), material("emissive", glow, emissive=3.0)))
    return Model("tent", parts)


def rock(size=1.0, color="#8e8a82", moss=0.0, seed=0):
    """Boulder about ``size`` wide: a lumpy, faceted stone cut flat where it meets the ground, in stone colours;
    ``moss`` 0 ~ 1 grows moss over its top."""
    rng = np.random.default_rng(seed)
    rx, ry, rz = 0.5 * rng.uniform(0.9, 1.1), 0.34 * rng.uniform(0.85, 1.15), 0.42 * rng.uniform(0.9, 1.1)
    body = ellipsoid(rx, ry, rz)
    planes = []
    for _ in range(9):                                       # flat facets, as on a split stone
        v = rng.normal(size=3)
        v[1] = abs(v[1]) * 0.6
        v /= np.linalg.norm(v)
        sup = math.sqrt(float(((v * np.array([rx, ry, rz])) ** 2).sum()))   # how far the ellipsoid reaches along v
        planes.append((v.astype(F32), F32(rng.uniform(0.8, 0.92) * sup)))
    amp, sd = F32(0.06), seed

    def fn(x, y, z):
        d = body._d(x, y, z)
        for v, o in planes:
            d = np.maximum(d, x * v[0] + y * v[1] + z * v[2] - o)
        lump = fbm3(x * 3.2 + 7, y * 3.2, z * 3.2, 3, sd + 40) - F32(0.5)
        d = (d + lump * amp * 2) * F32(0.8)
        return np.maximum(d, -(y + F32(sink)))              # cut flat where it meets the ground
    e, sink = 0.08, ry * 0.45
    shape = _Prim(fn, [-rx - e, -sink, -rz - e], [rx + e, ry + e, rz + e]).at(0, sink, 0)
    mat = material("stone", color, moss=moss) if moss else material("stone", color)
    return Model("rock", [Part("stone", shape, mat)]).scale(size)


def stone_lantern(height=1.0, color="#9c9890", moss="#6f8a4a", lit=False, glow="#ffcf7a"):
    """Japanese stone lantern (tōrō): base, pillar, platform, fire box with open windows, curved roof and
    jewel finial. ``lit=True`` puts a flame in the fire box (``sc.light_from(l["light"])``).
    Parts: base, pillar, platform, firebox, roof, finial, light (when lit)."""
    s = float(height) / 1.0
    stone = Material(color=_noisy(color, _shade_hex(color, 0.72), scale=0.05, octaves=4, seed=3, contrast=1.1),
                     roughness=0.92, specular=0.08)
    mossy = Material(color=_noisy(_shade_hex(color, 0.9), moss, scale=0.09, octaves=4, seed=5, contrast=0.9),
                     roughness=0.95, specular=0.05)

    def hexa(r, hgt):
        pts = [(r * math.cos(a), r * math.sin(a)) for a in np.radians(np.arange(6) * 60 + 30)]
        return prism(pts, depth=hgt).rotate_x(-90)
    base = smooth_union(hexa(0.24, 0.1).at(0, 0.05, 0), cylinder(0.16, 0.08).at(0, 0.13, 0), k=0.02)
    pillar = cylinder(0.075, 0.36, round=0.01).at(0, 0.17 + 0.18, 0)
    plat = smooth_union(hexa(0.2, 0.07).at(0, 0.565, 0), cylinder(0.11, 0.05).at(0, 0.52, 0), k=0.02)
    fb = hexa(0.15, 0.2).at(0, 0.7, 0)
    win = box(0.12, 0.11, 0.5).at(0, 0.71, 0) | box(0.5, 0.11, 0.12).at(0, 0.71, 0)
    firebox = fb - win
    roof = cone(0.32, 0.05, 0.13).at(0, 0.865, 0)
    roof = smooth_union(roof, hexa(0.3, 0.025).at(0, 0.81, 0), k=0.02)
    tips = union(*[cone(0.03, 0.0, 0.07).rotate_z(-60).rotate_y(a).at(0.27 * math.sin(math.radians(a + 90)), 0.84,
                                                                       0.27 * math.cos(math.radians(a + 90)))
                   for a in range(30, 360, 60)])
    fin = smooth_union(sphere(0.05).at(0, 0.97, 0), cone(0.035, 0.0, 0.07).at(0, 1.02, 0),
                       cylinder(0.035, 0.03).at(0, 0.935, 0), k=0.01)
    parts = [Part("base", base, mossy), Part("pillar", pillar, stone), Part("platform", plat, stone),
             Part("firebox", firebox, stone), Part("roof", roof | tips, mossy), Part("finial", fin, stone)]
    if lit:
        parts.append(Part("light", ellipsoid(0.035, 0.05, 0.035).at(0, 0.7, 0), material("emissive", glow,
                                                                                         emissive=3.0)))
    m = Model("stone lantern", parts)
    return m.scale(s) if abs(s - 1) > 1e-9 else m


def arched_bridge(length=4.0, width=1.2, height=0.75, color="#b8432e", deck="#8a6a4a", posts=7):
    """Japanese arched (drum) bridge spanning x: curved plank deck, lacquered railings with posts and caps,
    and arched side beams. Ends rest on y = 0 at x = ±length/2. Parts: deck, beams, railings, posts, caps."""
    L, Wd, hgt = float(length), float(width), float(height)
    R = (L * L / 4 + hgt * hgt) / (2 * hgt)
    cy = hgt - R

    def ring(r_out, r_in, depth_):
        # the hole reaches well past both faces, so the field inside the ring stays far from zero
        return (cylinder(r_out, depth_) - cylinder(r_in, depth_ + 2.0)).rotate_x(90).at(0, cy, 0)
    clip = box(L, hgt + 1.0, Wd + 0.4).at(0, (hgt + 1.0) / 2, 0)
    deck_s = ring(R + 0.06, R - 0.02, Wd) & clip
    beams = (ring(R - 0.02, R - 0.16, 0.1) & clip)
    beams = beams.at(0, 0, Wd / 2 - 0.08) | beams.at(0, 0, -Wd / 2 + 0.08)
    rh = 0.55
    rail = ring(R + 0.06 + rh, R + 0.06 + rh - 0.05, 0.06) & box(L * 0.955, hgt + 2, Wd + 0.4).at(0, (hgt + 2) / 2, 0)
    rails = rail.at(0, 0, Wd / 2 - 0.03) | rail.at(0, 0, -Wd / 2 + 0.03)
    mid = ring(R + 0.06 + rh * 0.5, R + 0.06 + rh * 0.5 - 0.035, 0.04) & box(L * 0.955, hgt + 2, Wd + 0.4).at(
        0, (hgt + 2) / 2, 0)
    rails = rails | mid.at(0, 0, Wd / 2 - 0.03) | mid.at(0, 0, -Wd / 2 + 0.03)
    pst, caps = [], []
    for i in range(int(posts)):
        x = -L * 0.47 + L * 0.94 * i / max(int(posts) - 1, 1)
        y = cy + math.sqrt(max(R * R - x * x, 0)) + 0.06
        for z in (Wd / 2 - 0.03, -Wd / 2 + 0.03):
            pst.append(box(0.07, rh + 0.08, 0.07).at(x, y + (rh + 0.08) / 2 - 0.02, z))
            caps.append(smooth_union(sphere(0.045).at(x, y + rh + 0.1, z),
                                     cone(0.03, 0.0, 0.07).at(x, y + rh + 0.16, z), k=0.01))
    return Model("arched bridge", [Part("deck", deck_s, material("wood", deck, roughness=0.6)),
                                   Part("beams", beams, material("paint", color, roughness=0.4)),
                                   Part("railings", rails, material("paint", color, roughness=0.35)),
                                   Part("posts", union(*pst), material("paint", color, roughness=0.35)),
                                   Part("caps", union(*caps), material("gold", "#c9a24a", roughness=0.3))])


# ===================================================================================================== registry

PREFABS = {
    "mug": ("tableware", "cup coffee tea"), "cup": ("tableware", "teacup saucer coffee"),
    "bottle": ("tableware", "wine beer glass drink"), "vase": ("tableware", "flowers pot ceramic"),
    "table": ("furniture", "dining desk"), "chair": ("furniture", "dining seat"),
    "armchair": ("furniture", "easy chair lounge chair reading chair recliner upholstered"),
    "sofa": ("furniture", "couch settee"), "bed": ("furniture", "double bed bedroom"),
    "side_table": ("furniture", "end table nightstand coffee table bedside"),
    "bookshelf": ("furniture", "bookcase shelves books library"), "counter": ("furniture", "bar kitchen shop"),
    "desk_lamp": ("light", "lamp reading lamp"), "pendant_lamp": ("light", "ceiling lamp hanging lamp"),
    "floor_lamp": ("light", "standing lamp standard lamp reading lamp"),
    "rug": ("furniture", "carpet mat round rug"), "potted_plant": ("nature", "plant houseplant pot"),
    "book": ("tableware", "novel"), "window_frame": ("architecture", "window"),
    "wall": ("architecture", "wall opening"), "room": ("architecture", "interior cutaway diorama window view"),
    "house": ("architecture", "home cottage building"), "tree": ("nature", "pine bush oak maple autumn"),
    "tent": ("outdoor", "camping campsite glowing tent"),
    "rock": ("outdoor", "boulder stone rocks mossy rock garden"),
    "stone_lantern": ("outdoor", "japanese lantern toro ishidoro garden lantern"),
    "arched_bridge": ("outdoor", "japanese bridge drum bridge garden bridge footbridge"),
    "car": ("vehicle", "automobile hatchback sports car van"), "truck": ("vehicle", "lorry"),
    "rocket": ("vehicle", "spaceship"), "robot": ("character", "bot android"),
    "character": ("character", "person figure"), "quadruped": ("character", "animal dog cat bear sheep"),
    "laptop": ("product", "computer notebook"), "speaker": ("product", "smart speaker audio"),
    "coffee_machine": ("product", "espresso machine"),
    "earbuds": ("product", "wireless earbuds earphones headphones airpods"),
    "charging_case": ("product", "earbud case earbuds case airpods case"),
    "smartphone": ("product", "phone mobile iphone cellphone"),
    "smartwatch": ("product", "watch wristwatch apple watch fitness tracker"),
    "toy_block": ("toy", "block letter cube"),
}


def names(kind=None):
    """Prefab names (``kind`` = 'furniture', 'product', 'outdoor', 'light', 'vehicle', 'tableware' ...)."""
    kinds = sorted({k for k, _ in PREFABS.values()})
    if kind is not None:
        _check_choice("solid.names", "kind", kind, kinds)
        return [n for n, (k, _) in PREFABS.items() if k == kind]
    return list(PREFABS)


def find(query):
    """Prefab names that match a word or phrase: ``find("reading lamp")`` -> ['floor_lamp', 'desk_lamp']."""
    q = str(query).lower().replace("-", " ").strip()
    qn = q.replace(" ", "_")
    words = [w for w in q.split() if w]
    scored = []
    for n, (k, alias) in PREFABS.items():
        text = f"{n.replace('_', ' ')} {alias} {k}"
        s = 0.0
        if qn == n:
            s += 10
        if q and q in text:
            s += 4
        s += sum(1.0 for w in words if w in text.split()) + sum(0.3 for w in words if w in text)
        if s > 0:
            scored.append((-s, n))
    return [n for _, n in sorted(scored)]


def describe(name):
    """First docstring paragraph and signature of a prefab."""
    if name not in PREFABS:
        raise _bad_name("solid prefab", name, list(PREFABS))
    import inspect
    fn = globals()[name]
    fn = getattr(fn, "__wrapped__", fn)
    doc = (fn.__doc__ or "").strip()
    return f"{name}{inspect.signature(fn)}\n{doc}"


# ===================================================================================================== checked options

for _n in __all__:
    _o = globals().get(_n)
    if isinstance(_o, type(_f)):
        globals()[_n] = _strict(_o, _n)
for _c in (_Xf, Shape, Material, Part, Model, Camera, Lighting, Scene, RenderResult):
    _strict_methods(_c)
for _n in ("orbit", "frame", "isometric"):
    setattr(Camera, _n, classmethod(_strict(Camera.__dict__[_n].__func__, "Camera." + _n)))
del _n, _o, _c

from ._args import module_getattr as _module_getattr                       # noqa: E402

__getattr__ = _module_getattr(__name__, {
    "lamp": "floor_lamp(), desk_lamp() or pendant_lamp()", "couch": "sofa()", "carpet": "rug(shape='round')",
    "phone": "smartphone()", "watch": "smartwatch()", "bridge": "arched_bridge()", "lantern": "stone_lantern()",
    "bookcase": "bookshelf()", "nightstand": "side_table()", "airpods": "earbuds() / charging_case()",
    "cat": "pixelcanvas.clay.animal('cat')", "dog": "pixelcanvas.clay.animal('dog')"})
