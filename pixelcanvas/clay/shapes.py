# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from .. import solid as _S
from ..color import F32
from ..color import rgb
from ..solid import (Shape, ellipsoid, sphere)


_Prim, _Affine = _S._Prim, _S._Affine


class _SoftUnion(_S._Union):
    """Smooth union that skips a child wherever its bounding box is farther than the fillet size: there the blend
    leaves the distance unchanged, so the result is identical and much cheaper for 20-30 part figures."""

    def _d(self, x, y, z):
        k = F32(self.k)
        d = self.children[0]._d(x, y, z)
        n = len(x)
        for c in self.children[1:]:
            if n > 48:
                lo, hi = c.bounds()
                need = np.flatnonzero(_S._bdist(lo, hi, x, y, z) < d + k)
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


def smooth_union(*shapes, k=0.1):
    """Blobby union with a fillet of size ``k`` (same field as ``solid.smooth_union``, evaluated with culling)."""
    shapes = [s for s in shapes if s is not None]
    if len(shapes) == 1:
        return shapes[0]
    if k <= 0:
        return _S.union(*shapes)
    return _SoftUnion(list(shapes), k=k)
_lin = _S._lin
_Checker = _S._Checker


# ===================================================================================================== math helpers

def _v(*a):
    return np.array(a if len(a) > 1 else a[0], np.float64).reshape(3)


def _Rx(d):
    return _S._rotm("x", d)


def _Ry(d):
    return _S._rotm("y", d)


def _Rz(d):
    return _S._rotm("z", d)


def _exp(rx, ry, rz):
    """Rotation matrix from a rotation vector given in degrees (axis * angle)."""
    v = np.radians(np.array([rx, ry, rz], np.float64))
    a = float(np.linalg.norm(v))
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def _xf(shape, R, t):
    """Place a shape built in a local frame: world = R @ local + t."""
    return shape._affine(np.asarray(R, np.float64), np.asarray(t, np.float64).reshape(3))


def _grp(shapes):
    """Group shapes (BVH) as one Shape that keeps its own coordinate frame for colour functions."""
    shapes = [s for s in shapes if s is not None]
    if not shapes:
        return None
    u = _S._bvh(list(shapes))
    return u


def _look_rot(fwd, up=(0, 1, 0)):
    """Rotation whose +z axis points along ``fwd`` (and +y as close to ``up`` as possible)."""
    f = np.asarray(fwd, np.float64)
    f = f / max(np.linalg.norm(f), 1e-12)
    u = np.asarray(up, np.float64)
    r = np.cross(u, f)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross([0, 0, 1.0] if abs(f[2]) < 0.9 else [1.0, 0, 0], f)
    r /= np.linalg.norm(r)
    u2 = np.cross(f, r)
    return np.stack([r, u2, f], 1)


def _axis_rot(axis_from, axis_to):
    """Smallest rotation taking unit vector ``axis_from`` onto ``axis_to``."""
    a = np.asarray(axis_from, np.float64)
    b = np.asarray(axis_to, np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(a @ b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        return _exp(*(p / np.linalg.norm(p) * 180.0))
    return _exp(*(v / s * math.degrees(math.atan2(s, c))))


def _hex(c):
    r, g, b = [int(round(max(0.0, min(1.0, float(x))) * 255)) for x in rgb(c)]
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix(a, b, t):
    A, B = np.asarray(rgb(a), np.float64), np.asarray(rgb(b), np.float64)
    return _hex(A * (1 - t) + B * t)


def _shade(c, k):
    return _hex(np.clip(np.asarray(rgb(c), np.float64) * k, 0, 1))


# ===================================================================================================== primitives

def round_cone(a, b, ra, rb):
    """Tapered capsule from point ``a`` (radius ``ra``) to point ``b`` (radius ``rb``): limbs, tails, horns."""
    A, B = _v(a), _v(b)
    ba = B - A
    l2 = float(ba @ ba)
    if l2 < 1e-14 or abs(ra - rb) >= math.sqrt(l2) * 0.995:
        return sphere(max(ra, rb)).at(*(A if ra >= rb else B))
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    bx, by, bz = F32(ba[0]), F32(ba[1]), F32(ba[2])
    Ax, Ay, Az = F32(A[0]), F32(A[1]), F32(A[2])
    L2, A2, IL2, RR, RA, RB = F32(l2), F32(a2), F32(il2), F32(rr), F32(ra), F32(rb)
    K = F32(np.sign(rr) * rr * rr)
    A2IL2 = F32(a2 * il2)

    def fn(x, y, z):
        px, py, pz = x - Ax, y - Ay, z - Az
        yv = px * bx + py * by + pz * bz
        zv = yv - L2
        qx, qy, qz = px * L2 - bx * yv, py * L2 - by * yv, pz * L2 - bz * yv
        x2 = qx * qx + qy * qy + qz * qz
        y2 = yv * yv * L2
        z2 = zv * zv * L2
        k = K * x2
        d_end = np.sqrt(x2 + z2) * IL2 - RB
        d_start = np.sqrt(x2 + y2) * IL2 - RA
        d_side = (np.sqrt(x2 * A2IL2) + yv * RR) * IL2 - RA
        return np.where(np.sign(zv) * A2 * z2 > k, d_end, np.where(np.sign(yv) * A2 * y2 < k, d_start, d_side))
    lo = np.minimum(A - ra, B - rb)
    hi = np.maximum(A + ra, B + rb)
    return _Prim(fn, lo, hi)


def _chain(points, radii):
    """Union of tapered capsules through ``points`` (joints share their sphere, so the chain is seamless)."""
    out = []
    for i in range(len(points) - 1):
        out.append(round_cone(points[i], points[i + 1], radii[i], radii[i + 1]))
    return out


def _sub_chain(points, radii, t0, t1):
    """The piece of a polyline limb between arc-length fractions t0 and t1 (0 = first joint, 1 = last)."""
    P = [np.asarray(p, np.float64) for p in points]
    seg = [float(np.linalg.norm(P[i + 1] - P[i])) for i in range(len(P) - 1)]
    tot = max(sum(seg), 1e-9)
    cum = np.concatenate([[0], np.cumsum(seg)]) / tot

    def at(t):
        i = int(np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1))
        f = (t - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
        return P[i] + (P[i + 1] - P[i]) * f, radii[i] + (radii[i + 1] - radii[i]) * f
    t0, t1 = max(0.0, t0), min(1.0, t1)
    if t1 <= t0 + 1e-6:
        return [], [], []
    pts, rs = [], []
    p, r = at(t0)
    pts.append(p)
    rs.append(r)
    for i in range(1, len(P) - 1):
        if t0 < cum[i] < t1:
            pts.append(P[i])
            rs.append(radii[i])
    p, r = at(t1)
    pts.append(p)
    rs.append(r)
    return _chain(pts, rs), pts, rs


def _ell(x, y, z, a, b, c):
    k0 = np.sqrt((x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2)
    k1 = np.sqrt((x / (a * a)) ** 2 + (y / (b * b)) ** 2 + (z / (c * c)) ** 2)
    return k0 * (k0 - 1) / np.maximum(k1, F32(1e-9))


def _smin(a, b, k):
    h = np.clip(F32(0.5) + F32(0.5) * (b - a) / F32(k), 0, 1)
    return b * (1 - h) + a * h - F32(k) * h * (1 - h)


def _rbox(x, y, z, bx, by, bz, r):
    qx, qy, qz = np.abs(x) - F32(bx - r), np.abs(y) - F32(by - r), np.abs(z) - F32(bz - r)
    mx, my, mz = np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0)
    return np.sqrt(mx * mx + my * my + mz * mz) + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - F32(r)


def _flat_ell(rx, ry, rz, R=None, c=(0, 0, 0)):
    s = ellipsoid(rx, ry, rz)
    return _xf(s, np.eye(3) if R is None else R, c)


# ===================================================================================================== 2D shapes (face)
# All 2D functions take float32 arrays x, y (head-local, y up) and return signed distances (negative inside).

def _e2(x, y, cx, cy, rx, ry, rot=0.0):
    px, py = x - F32(cx), y - F32(cy)
    if rot:
        c, s = F32(math.cos(math.radians(rot))), F32(math.sin(math.radians(rot)))
        px, py = c * px + s * py, -s * px + c * py
    k0 = np.sqrt((px / F32(rx)) ** 2 + (py / F32(ry)) ** 2)
    k1 = np.sqrt((px / F32(rx * rx)) ** 2 + (py / F32(ry * ry)) ** 2)
    return k0 * (k0 - 1) / np.maximum(k1, F32(1e-9))


def _seg2(x, y, a, b, w):
    ax, ay = a
    bx, by = b
    ex, ey = bx - ax, by - ay
    ee = max(ex * ex + ey * ey, 1e-12)
    px, py = x - F32(ax), y - F32(ay)
    h = np.clip((px * F32(ex) + py * F32(ey)) / F32(ee), 0, 1)
    qx, qy = px - F32(ex) * h, py - F32(ey) * h
    return np.sqrt(qx * qx + qy * qy) - F32(w)


def _stroke2(x, y, pts, w):
    """Round-capped stroke through 2D points (a polyline), half-width ``w`` (float or per-point list)."""
    ws = w if isinstance(w, (list, tuple, np.ndarray)) else [w] * len(pts)
    d = None
    for i in range(len(pts) - 1):
        ww = 0.5 * (ws[i] + ws[i + 1])
        s = _seg2(x, y, pts[i], pts[i + 1], ww)
        d = s if d is None else np.minimum(d, s)
    return d


def _bez(p0, p1, p2, n=7):
    t = np.linspace(0, 1, n)[:, None]
    P0, P1, P2 = (np.asarray(p, np.float64) for p in (p0, p1, p2))
    return [tuple(v) for v in (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t * t * P2]


def _arc_pts(cx, cy, r, a0, a1, n=9, ry=None):
    ry = r if ry is None else ry
    a = np.radians(np.linspace(a0, a1, n))
    return [(cx + r * math.cos(t), cy + ry * math.sin(t)) for t in a]


def _hp(x, y, p, n):
    """Half-plane: negative on the side opposite to normal ``n`` through point ``p``."""
    nx, ny = n
    ln = math.hypot(nx, ny)
    return ((x - F32(p[0])) * F32(nx / ln) + (y - F32(p[1])) * F32(ny / ln))


# ===================================================================================================== head core

HEAD_SHAPES = {
    # radii (x, y, z) in units of R; taper of the width toward the chin (x, z); chin pushed forward; box = how
    # square the jaw is (0 = egg, 1 = rounded box)
    "round": dict(r=(1.0, 0.98, 0.96), tx=0.16, tz=0.1, chin=0.06, box=0.0),
    "oval": dict(r=(0.88, 1.0, 0.94), tx=0.24, tz=0.12, chin=0.06, box=0.0),
    "square": dict(r=(0.94, 0.97, 0.94), tx=0.06, tz=0.06, chin=0.04, box=0.45),
    "heart": dict(r=(1.04, 0.98, 0.95), tx=0.42, tz=0.16, chin=0.08, box=0.0),
    "wide": dict(r=(1.12, 0.92, 0.98), tx=0.12, tz=0.08, chin=0.05, box=0.0),
    "long": dict(r=(0.84, 1.05, 0.92), tx=0.22, tz=0.1, chin=0.06, box=0.0),
}


class _HeadCore:
    """Analytic head SDF in head-local units (origin = head centre, facing +z) plus surface queries.

    An egg-like ellipsoid whose width tapers toward the chin (round / oval / heart / wide / long), optionally
    blended with a rounded box (square): one smooth surface, no seams."""

    def __init__(self, shape="round", R=0.1, width=1.0):
        cfg = HEAD_SHAPES.get(shape, HEAD_SHAPES["round"])
        self.R, self.shape = float(R), shape
        w = float(width)
        r = cfg["r"]
        self.ca = (r[0] * R * w, r[1] * R, r[2] * R)
        self.cc = 0.0
        a, b, c = (F32(v) for v in self.ca)
        tx, tz, chin, bx = cfg["tx"], cfg["tz"], cfg["chin"] * R, cfg["box"]
        lip = F32(1.0 - max(tx, tz))
        TX, TZ, CH, BX = F32(tx), F32(tz), F32(chin), F32(bx)
        rb = 0.42 * R

        def fn(x, y, z):
            u = np.clip(-y / b, 0, 1)
            u2 = u * u
            X = x / (1 - TX * u2)
            Z = (z - CH * u2) / (1 - TZ * u2)
            d = _ell(X, y, Z, a, b, c)
            if bx > 0:
                d = d * (1 - BX) + _rbox(X, y, Z, float(a) * 0.97, float(b) * 0.97, float(c) * 0.97, rb) * BX
            return d * lip
        self.fn = fn
        m = 0.05 * R
        self.lo = np.array([-self.ca[0] - m, -self.ca[1] - m, -self.ca[2] - m])
        self.hi = np.array([self.ca[0] + m, self.ca[1] + m, self.ca[2] + chin + m])

    def shape_(self, inflate=0.0):
        p = _Prim(self.fn, self.lo - inflate, self.hi + inflate)
        return p.round(inflate) if inflate else p

    def d(self, P):
        P = np.asarray(P, F32).reshape(-1, 3)
        return self.fn(P[:, 0].copy(), P[:, 1].copy(), P[:, 2].copy())

    def front(self, x, y):
        """Surface point(s) seen from the front (+z) at head-local (x, y), and the normal there."""
        x = np.atleast_1d(np.asarray(x, np.float64))
        y = np.atleast_1d(np.asarray(y, np.float64))
        lo = np.full(x.shape, -0.2 * self.R)
        hi = np.full(x.shape, 2.0 * self.R)
        for _ in range(28):
            m = (lo + hi) / 2
            inside = self.d(np.stack([x, y, m], -1)) < 0
            lo = np.where(inside, m, lo)
            hi = np.where(inside, hi, m)
        P = np.stack([x, y, (lo + hi) / 2], -1)
        return P, self.normal(P)

    def ray(self, dirs, origin=(0, 0, 0)):
        """Surface points hit by rays from ``origin`` (inside the head) along ``dirs``."""
        D = np.asarray(dirs, np.float64).reshape(-1, 3)
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        O = np.asarray(origin, np.float64)
        lo = np.zeros(len(D))
        hi = np.full(len(D), 3.0 * self.R)
        for _ in range(28):
            m = (lo + hi) / 2
            inside = self.d(O + D * m[:, None]) < 0
            lo = np.where(inside, m, lo)
            hi = np.where(inside, hi, m)
        P = O + D * ((lo + hi) / 2)[:, None]
        return P, self.normal(P)

    def normal(self, P):
        P = np.asarray(P, np.float64).reshape(-1, 3)
        e = 1e-3 * self.R
        g = np.zeros_like(P)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = e
            g[:, i] = self.d(P + dp) - self.d(P - dp)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


class _Decal(Shape):
    """A thin solid that sits on the head surface inside a 2D footprint (front projection in head-local x, y).

    ``relief`` lifts it above the skin (constant or a function of x, y). Painted faces stay crisp: the renderer
    anti-aliases decal outlines like any other object edge."""

    cost = 2

    def __init__(self, head, d2, box2, relief, zmin):
        self.head, self.d2, self.relief, self.zmin = head, d2, relief, float(zmin)
        (x0, y0), (x1, y1) = box2
        self.lo = np.array([x0, y0, zmin])
        self.hi = np.array([x1, y1, head.hi[2] + 0.2 * head.R])

    def _d(self, x, y, z):
        r = self.relief(x, y) if callable(self.relief) else F32(self.relief)
        d = self.head.fn(x, y, z) - r
        return np.maximum(np.maximum(d, self.d2(x, y)), F32(self.zmin) - z)

    def _bb(self):
        return self.lo, self.hi
