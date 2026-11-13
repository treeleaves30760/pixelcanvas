# -*- coding: utf-8 -*-
"""
flora — plants with real structure, one call each: trees, shrubs, grass, flowers, potted plants, leaves, rocks.

Every plant is built from the structure that makes it recognisable: trees grow a tapering trunk with root flare and
a recursive, seeded branch skeleton; canopies are many overlapping leaf clusters shaded as volumes (lit side, shadow
side, cast shadows between clusters, sky holes, an edge made of individual leaves); flowers are petals in polar
layers with veins; rocks are faceted Voronoi solids.

Units: ``at=(u, v)`` is the base point in canvas fractions; ``height`` / ``size`` are fractions of the canvas
height; texture sizes are 1080p pixels scaled by ``S = H / 1080``. Everything is deterministic by ``seed``.
Every plant paints through ``cv.paint`` and returns a :class:`Plant` (mask, base, top, canopy centre, bbox).

    from pixelcanvas import Canvas
    from pixelcanvas.flora import plant, tree, flower
    cv = Canvas(960, 540, "#cfe0ea")
    tree(cv, "oak", at=(0.35, 0.92), height=0.8, seed=3, shadow=True)
    plant(cv, "sunflower", at=(0.75, 0.95), height=0.5)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .color import F32, rgb, clamp01, smoothstep
from .filters import gaussian_blur, sample
from .noise import fbm2d_at
from ._args import reject_unknown as _reject, signature_options as _sigopts, check_choice as _choice, unknown_name as _unknown_name
from .form import check_frac as _check_frac

_RES = getattr(Image, "Resampling", Image)

LIGHT = (-0.62, -0.78)          # default direction towards the light (screen coords, y down): upper left

__all__ = [
    "LIGHT", "Plant", "PALETTES", "BARKS", "CLIMBERS", "TREES", "FLOWERS", "POTTED", "LEAVES", "ROCKS", "POTS", "AUTUMN",
    "plant", "names", "find",
    "tree", "forest", "palm", "bamboo", "saguaro", "prickly_pear", "banana",
    "bush", "hedge", "topiary", "grass", "reeds", "cattails", "fern", "ivy", "moss", "lily_pads", "mushroom",
    "pumpkin", "wheat",
    "flower", "flower_bed", "meadow", "bouquet", "vase", "potted",
    "leaf", "falling_leaves", "leaf_pile", "branch_frame", "perch", "climber",
    "rock", "rock_pile", "pebbles", "cliff", "stepping_stones",
    "cellular", "strokes", "ground_shadow",
]


# ============================================================================
# small helpers
# ============================================================================
def _S(cv):
    return cv.h / 1080.0


def _rng(seed, salt=0):
    return np.random.default_rng((int(seed) * 7919 + int(salt) * 104729) & 0x7FFFFFFF)


def _l3(light, lz=0.72):
    lx, ly = float(light[0]), float(light[1])
    n = math.hypot(lx, ly) or 1.0
    v = np.array([lx / n, ly / n, lz], np.float64)
    return v / np.linalg.norm(v)


def _l2(light):
    lx, ly = float(light[0]), float(light[1])
    n = math.hypot(lx, ly) or 1.0
    return lx / n, ly / n


def _ramp(colors, t):
    """Multi-stop colour ramp: ``colors`` = list of colours evenly spaced over 0~1 (or [(pos, colour), ...])."""
    t = np.asarray(t, F32)
    c0 = colors[0]
    if isinstance(c0, (tuple, list)) and len(c0) == 2 and isinstance(c0[0], (int, float)) and not isinstance(c0[1], (int, float)):
        pos = np.array([c[0] for c in colors], F32)
        cols = np.stack([rgb(c[1]) for c in colors])
    else:
        cols = np.stack([rgb(c) for c in colors])
        pos = np.linspace(0, 1, len(cols)).astype(F32)
    out = np.empty(t.shape + (3,), F32)
    tc = np.clip(t, pos[0], pos[-1])
    for k in range(3):
        out[..., k] = np.interp(tc, pos, cols[:, k])
    return out


def _hash(ix, iy, seed):
    """Integer lattice hash → float 0~1 (deterministic, vectorised)."""
    with np.errstate(over="ignore"):
        h = (np.asarray(ix, np.int64).astype(np.uint64) * np.uint64(374761393)
             + np.asarray(iy, np.int64).astype(np.uint64) * np.uint64(668265263)
             + np.uint64((int(seed) * 2654435761) & 0xFFFFFFFF))
        h = h & np.uint64(0xFFFFFFFF)
        h = ((h ^ (h >> np.uint64(13))) * np.uint64(1274126177)) & np.uint64(0xFFFFFFFF)
        h = h ^ (h >> np.uint64(16))
    return (h.astype(np.float64) / 4294967295.0).astype(F32)


def _rot(dx, dy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return dx * c + dy * s, -dx * s + dy * c


def _aa(d, w=1.0):
    """Signed distance (px, negative inside) → anti-aliased coverage."""
    return np.clip(0.5 - d / w, 0.0, 1.0).astype(F32)


def _blend_rgb(a, b, t):
    t = np.asarray(t, F32)
    if t.ndim == 2 or (t.ndim == 1 and np.ndim(a) > 1):
        t = t[..., None]
    return a + (b - a) * t


# ============================================================================
# texture primitives: cellular (Worley / Voronoi) and gradient-aligned strokes
# ============================================================================
def _worley(X, Y, size, seed=0, jitter=0.85, aspect=1.0):
    """Voronoi at arbitrary pixel coordinates. Returns F1, F2 (px), nearest feature point (fx, fy), cell id 0~1."""
    gx = np.asarray(X, F32) / size
    gy = np.asarray(Y, F32) / (size * aspect)
    ix = np.floor(gx).astype(np.int64)
    iy = np.floor(gy).astype(np.int64)
    F1 = np.full(gx.shape, 1e9, F32)
    F2 = np.full(gx.shape, 1e9, F32)
    FX = np.zeros(gx.shape, F32)
    FY = np.zeros(gx.shape, F32)
    ID = np.zeros(gx.shape, F32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = ix + ox, iy + oy
            h1 = _hash(cx, cy, seed)
            h2 = _hash(cx, cy, seed + 101)
            px = cx + 0.5 + (h1 - 0.5) * jitter
            py = cy + 0.5 + (h2 - 0.5) * jitter
            d = np.hypot(gx - px, (gy - py) * aspect)
            closer = d < F1
            F2 = np.where(closer, F1, np.minimum(F2, d))
            F1 = np.where(closer, d, F1)
            FX = np.where(closer, px, FX)
            FY = np.where(closer, py, FY)
            ID = np.where(closer, _hash(cx, cy, seed + 202), ID)
    return F1 * size, F2 * size, FX * size, FY * size * aspect, ID


def cellular(h, w, size, seed=0, jitter=0.85, X=None, Y=None):
    """Cellular (Worley / Voronoi) noise: cells of ``size`` px. Returns ``(f1, f2, cell_id)``.

    ``f1`` = distance to the nearest cell point, ``f2 - f1`` ≈ distance to the cell edge (0 on borders → mortar,
    cracks, leaf edges), ``cell_id`` 0~1 is constant per cell (colour each stone / leaf / scale). Pass ``X, Y`` grids
    to evaluate at arbitrary coordinates (perspective, warps)."""
    if X is None:
        Y, X = np.mgrid[0:h, 0:w].astype(F32)
    f1, f2, _, _, cid = _worley(X, Y, float(size), seed, jitter)
    return f1, f2, cid


def strokes(X, Y, angle, length=12.0, width=2.0, seed=0):
    """Gradient-aligned dashes: short strokes ``length`` x ``width`` px oriented by ``angle`` (radians, scalar or map).

    Returns ``(coverage, stroke_id)``: coverage 0~1 of the dash under each pixel and a per-dash id 0~1. Use for
    needles, grass, fur, bark fibres, rain — anything whose texture has a direction. ``length`` / ``width`` are
    scalars; scale the coordinates for perspective."""
    X = np.asarray(X, F32)
    Y = np.asarray(Y, F32)
    ca, sa = np.cos(angle), np.sin(angle)
    u = X * ca + Y * sa
    v = -X * sa + Y * ca
    asp = length / max(width, 1e-3)
    cell = max(width * 1.6, 1.0)
    f1, f2, fx, fy, cid = _worley(u / asp, v, cell, seed)
    du = (u / asp - fx)
    dv = (v - fy)
    r = np.hypot(du / (cell * 0.55), dv / (width * 0.5))
    cov = clamp01((1.0 - r) * width * 0.9 + 0.5)
    return cov.astype(F32), cid


# ============================================================================
# patches: local RGBA buffers in canvas pixel coordinates
# ============================================================================
class _Patch:
    """Local straight-alpha RGBA buffer covering canvas pixels [x0, x1) x [y0, y1)."""

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = int(x0), int(y0)
        self.x1, self.y1 = max(int(x1), self.x0 + 1), max(int(y1), self.y0 + 1)
        h, w = self.y1 - self.y0, self.x1 - self.x0
        self.rgb = np.zeros((h, w, 3), F32)
        self.a = np.zeros((h, w), F32)

    @property
    def shape(self):
        return self.a.shape

    def grid(self):
        Y, X = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(F32)
        return X + 0.5 - 0.5, Y

    def over(self, rgb_, a, x0=None, y0=None):
        """Composite (rgb_, a) — arrays at canvas offset (x0, y0), default this patch's origin — over this patch."""
        x0 = self.x0 if x0 is None else int(x0)
        y0 = self.y0 if y0 is None else int(y0)
        h, w = a.shape
        X0, Y0 = max(x0, self.x0), max(y0, self.y0)
        X1, Y1 = min(x0 + w, self.x1), min(y0 + h, self.y1)
        if X0 >= X1 or Y0 >= Y1:
            return self
        sa = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        sc = rgb_[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] if np.ndim(rgb_) == 3 else np.broadcast_to(rgb(rgb_), sa.shape + (3,))
        da = self.a[Y0 - self.y0:Y1 - self.y0, X0 - self.x0:X1 - self.x0]
        dc = self.rgb[Y0 - self.y0:Y1 - self.y0, X0 - self.x0:X1 - self.x0]
        oa = sa + da * (1 - sa)
        safe = np.maximum(oa, 1e-6)[..., None]
        dc[:] = (sc * sa[..., None] + dc * (da * (1 - sa))[..., None]) / safe
        da[:] = oa
        return self

    def merge(self, other):
        return self.over(other.rgb, other.a, other.x0, other.y0)


def _clip_box(cv, x0, y0, x1, y1):
    return max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0), min(int(math.ceil(x1)), cv.w), \
        min(int(math.ceil(y1)), cv.h)


def _full(cv, p, clip=None):
    """Patch → full-canvas (colour map, mask)."""
    m = np.zeros((cv.h, cv.w), F32)
    c = np.zeros((cv.h, cv.w, 3), F32)
    m[p.y0:p.y1, p.x0:p.x1] = p.a
    c[p.y0:p.y1, p.x0:p.x1] = p.rgb
    if clip is not None:
        m *= np.asarray(clip, F32)
    return c, m


def _commit(cv, p, clip=None, opacity=1.0):
    c, m = _full(cv, p, clip)
    cv.paint(c, m, opacity=opacity)
    return m


class _Vec:
    """Supersampled Pillow drawing on a local box: shapes carry an RGB attribute (colour or shading data)."""

    def __init__(self, x0, y0, x1, y1, aa=3):
        self.x0, self.y0 = int(x0), int(y0)
        self.w, self.h = max(int(x1) - self.x0, 1), max(int(y1) - self.y0, 1)
        self.s = aa
        self.A = Image.new("L", (self.w * aa, self.h * aa), 0)
        self.C = Image.new("RGB", (self.w * aa, self.h * aa), (0, 0, 0))
        self.dA = ImageDraw.Draw(self.A)
        self.dC = ImageDraw.Draw(self.C)

    def _p(self, pts):
        s = self.s
        return [((x - self.x0) * s, (y - self.y0) * s) for x, y in pts]

    @staticmethod
    def _c(col):
        c = np.clip(rgb(col), 0, 1)
        return tuple(int(v * 255 + 0.5) for v in c)

    def poly(self, pts, col, erase=False):
        if len(pts) < 3:
            return
        p = self._p(pts)
        self.dA.polygon(p, fill=0 if erase else 255)
        self.dC.polygon(p, fill=(0, 0, 0) if erase else self._c(col))

    def line(self, pts, width, col, erase=False):
        if len(pts) < 2:
            return
        p = self._p(pts)
        wd = max(1, int(round(width * self.s)))
        a = 0 if erase else 255
        cc = (0, 0, 0) if erase else self._c(col)
        self.dA.line(p, fill=a, width=wd, joint="curve")
        self.dC.line(p, fill=cc, width=wd, joint="curve")
        r = width * self.s / 2
        if r >= 1.0:
            for x, y in (p[0], p[-1]):
                self.dA.ellipse([x - r, y - r, x + r, y + r], fill=a)
                self.dC.ellipse([x - r, y - r, x + r, y + r], fill=cc)

    def ellipse(self, cx, cy, rx, ry, col, erase=False):
        s = self.s
        box = [(cx - rx - self.x0) * s, (cy - ry - self.y0) * s, (cx + rx - self.x0) * s, (cy + ry - self.y0) * s]
        self.dA.ellipse(box, fill=0 if erase else 255)
        self.dC.ellipse(box, fill=(0, 0, 0) if erase else self._c(col))

    def result(self):
        A, C = self.A, self.C
        if self.s > 1:
            A = A.resize((self.w, self.h), _RES.BOX)
            C = C.resize((self.w, self.h), _RES.BOX)
        a = np.asarray(A, F32) / 255.0
        c = np.asarray(C, F32) / 255.0
        c = c / np.maximum(a, 1e-3)[..., None]
        return np.clip(c, 0, 1), a

    def patch(self):
        c, a = self.result()
        p = _Patch(self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)
        p.rgb, p.a = c.astype(F32), a
        return p


def _ribbon(spine, widths):
    """Polygon around a spine polyline with half-widths per point."""
    P = np.asarray(spine, np.float64)
    wv = np.asarray(widths, np.float64)
    d = np.gradient(P, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    L = P + n * wv[:, None]
    R = P - n * wv[:, None]
    return [tuple(p) for p in L] + [tuple(p) for p in R[::-1]]


def _bez(p0, p1, p2, n=16):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(p, np.float64) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def _shade_from_alpha(a, sigma, light, strength=0.6):
    """Pillowy volume shading from a mask: normal from the gradient of the blurred mask → multiplier ~ (1-s)..(1+s)."""
    b = gaussian_blur(a, max(sigma, 0.6))
    gy, gx = np.gradient(b)
    k = 1.0 / max(sigma, 0.6) * 1.2
    nx, ny = -gx / k, -gy / k
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0.05, 1))
    L = _l3(light)
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    d = (nx * L[0] + ny * L[1] + nz * L[2]) / n
    return (1 + strength * (d - 0.55) * 1.6).astype(F32)


# ============================================================================
# Plant result
# ============================================================================
@dataclass
class Plant:
    """What every plant call returns. ``mask`` is full-canvas (H, W); anchors are canvas pixels (x, y)."""
    name: str
    mask: np.ndarray = None
    base: tuple = (0.0, 0.0)
    top: tuple = (0.0, 0.0)
    canopy: tuple = (0.0, 0.0)
    bbox: tuple = (0, 0, 0, 0)
    extra: dict = field(default_factory=dict)
    patch: object = None

    def __repr__(self):
        return f"Plant({self.name!r}, base={tuple(round(v, 1) for v in self.base)}, " \
               f"top={tuple(round(v, 1) for v in self.top)}, bbox={self.bbox})"


def _finish(cv, name, patch, base, top, canopy, *, paint=True, clip=None, shadow=False, light=LIGHT,
            shadow_len=None, haze=None, extra=None, opacity=1.0):
    """Common tail: optional haze, ground shadow, paint, build the Plant."""
    if patch is None:
        return Plant(name, np.zeros((cv.h, cv.w), F32), base, top, canopy, (0, 0, 0, 0), extra or {})
    if haze is not None:
        hc, ha = haze
        patch.rgb = _blend_rgb(patch.rgb, rgb(hc), float(ha))
    if paint:
        if shadow:
            ground_shadow(cv, patch, base, light=light, length=shadow_len if shadow_len else 1.0,
                          strength=0.45 if shadow is True else float(shadow))
        m = _commit(cv, patch, clip, opacity)
    else:
        m = None
    return Plant(name, m, tuple(map(float, base)), tuple(map(float, top)), tuple(map(float, canopy)),
                 (patch.x0, patch.y0, patch.x1, patch.y1), extra or {}, patch)


def ground_shadow(cv, patch, base, light=LIGHT, length=1.0, strength=0.45, color=(0.42, 0.46, 0.58), blur=None):
    """Cast the patch's silhouette onto flat ground at ``base`` (canvas px) away from ``light``.

    The silhouette is flattened (``length`` = shadow length / object height) and sheared; blur grows with distance."""
    bx, by = float(base[0]), float(base[1])
    lx, ly = _l2(light)
    sx = -lx * length * 1.1                   # ground shift per unit height (x)
    sy = -0.3 * length if ly < 0 else 0.3 * length     # light from the front-upper side → shadow recedes
    h = max(by - patch.y0, 2.0)
    reach_x = abs(sx) * h + 4
    reach_y = abs(sy) * h + 4
    x0 = min(patch.x0, patch.x0 + sx * h) - 6
    x1 = max(patch.x1, patch.x1 + sx * h) + 6
    y0, y1 = by - (reach_y if sy < 0 else 6), by + (reach_y if sy > 0 else 6) + 2
    X0, Y0, X1, Y1 = _clip_box(cv, x0, y0, x1, y1)
    if X0 >= X1 or Y0 >= Y1:
        return None
    Y, X = np.mgrid[Y0:Y1, X0:X1].astype(F32)
    t = (Y - by) / sy if abs(sy) > 1e-4 else np.zeros_like(Y)       # height above base of the source point
    xs = X - sx * t
    ys = by - t
    valid = (t >= -2) & (t <= h + 2)
    src = sample(patch.a, xs - patch.x0, ys - patch.y0) * valid
    # contact darkening near base
    sig = 1.0 + np.clip(t, 0, None) * 0.02
    m = gaussian_blur(src.astype(F32), max(1.0, 0.004 * h + 1.0))
    fade = 1 - 0.55 * np.clip(t / max(h, 1), 0, 1)
    m = m * fade
    full = np.zeros((cv.h, cv.w), F32)
    full[Y0:Y1, X0:X1] = clamp01(m) * strength
    cv.paint(rgb(color), full, mode="multiply")
    return full


# ============================================================================
# branch skeletons (recursive, seeded) and bark
# ============================================================================
def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class _Skel:
    def __init__(self):
        self.segs = []          # (x0, y0, x1, y1, r0, r1, s0, s1, depth)
        self.tips = []          # (x, y, angle, depth, length)
        self.nodes = []         # points along thin branches (for foliage)


def _grow(sk, rng, sp, x, y, ang, length, radius, depth, along=0.0, leader=False):
    """Grow one branch as a crooked polyline, then its children (recursion)."""
    nseg = sp.get("nseg", 4)
    wig = math.radians(sp.get("wiggle", 12))
    up = sp.get("up", 0.1) + sp.get("up_d", 0.0) * depth
    wind = sp.get("wind", 0.0)
    taper = sp.get("taper", 0.72)
    pts = [(x, y)]
    rads = [radius]
    a = ang
    for i in range(nseg):
        a += rng.normal(0, wig) + up * _angdiff(-math.pi / 2, a) / nseg * 1.5 + wind * 0.08
        if leader:
            a += 0.5 * _angdiff(-math.pi / 2, a)
        elif depth >= 1 and y > -sp.get("floor", 0.22) and math.sin(a) > -0.3:
            a += 0.6 * _angdiff(-math.pi / 2, a)            # keep branches off the ground
        x += length / nseg * math.cos(a)
        y += length / nseg * math.sin(a)
        pts.append((x, y))
        rads.append(radius * (1 - (1 - taper) * (i + 1) / nseg))
    seglen = length / nseg
    for i in range(nseg):
        sk.segs.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], rads[i], rads[i + 1],
                        along + i * seglen, along + (i + 1) * seglen, depth))
    maxd = sp["depth"]
    if depth >= maxd - 1:
        for i in range(1, nseg + 1):
            sk.nodes.append((pts[i][0], pts[i][1], a, depth))
    if depth >= maxd or (sp.get("broken") and depth >= 2 and rng.random() < sp["broken"]):
        sk.tips.append((x, y, a, depth, length))
        return
    lr = sp.get("len_ratio", 0.7)
    rr = sp.get("rad_ratio", 0.68)
    spread = math.radians(sp.get("spread", 35))
    sj = math.radians(sp.get("spread_jitter", 10))
    if leader:
        nl = sp.get("laterals", 8)
        t0, t1 = sp.get("lat_range", (0.3, 0.95))
        side = rng.choice([-1, 1])
        lat_ang = math.radians(sp.get("lat_angle", 60))
        prof = sp.get("lat_profile", "cone")
        for k in range(nl):
            t = t0 + (t1 - t0) * (k + rng.uniform(0.1, 0.9)) / nl
            i = min(int(t * nseg), nseg - 1)
            f = t * nseg - i
            px = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f
            py = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f
            if prof == "cone":
                lf = (1 - t) ** 0.9 + 0.08
            elif prof == "column":
                lf = 0.35 + 0.25 * math.sin(math.pi * t)
            else:                                         # ellipse
                lf = math.sin(math.pi * min(1.0, (t - t0) / max(1 - t0, 1e-3) * 0.85 + 0.12)) ** 0.8
            side = -side if rng.random() < 0.85 else side
            aa = -math.pi / 2 + side * (lat_ang + rng.normal(0, sj))
            sub = dict(sp)
            _grow(sk, rng, sp, px, py, aa, length * sp.get("lat_len", 0.5) * lf * rng.uniform(0.8, 1.15),
                  rads[i] * sp.get("lat_rad", 0.45), depth + 1, along + t * length)
        # leader continues a little above
        return
    n = int(rng.integers(sp.get("kids", (2, 3))[0], sp.get("kids", (2, 3))[1] + 1))
    rad_child = radius * taper * (1.0 / n) ** (1 / 2.4) / (rr / 0.68) ** 0 if n > 0 else 0
    for k in range(n):
        off = (k / (n - 1) - 0.5) * 2 if n > 1 else 0.0
        aa = a + off * spread + rng.normal(0, sj)
        ll = length * lr * rng.uniform(0.82, 1.15)
        _grow(sk, rng, sp, x, y, aa, ll, max(rad_child * rng.uniform(0.85, 1.1), radius * 0.25), depth + 1,
              along + length)
    # a lateral twig from the middle of thick branches
    if sp.get("side_twigs", 0.5) > rng.random() and depth < maxd - 1:
        i = int(rng.integers(1, nseg))
        side = rng.choice([-1, 1])
        _grow(sk, rng, sp, pts[i][0], pts[i][1], a + side * spread * 1.3, length * lr * 0.6, rads[i] * 0.45,
              depth + 2, along + i * seglen)


def _render_bark(P, segs, light, bark, kind="furrow", S=1.0, seed=0, flare=None, ground=None):
    """Tapered cylinders along segments → (rgb, alpha) on patch P. ``bark`` = (shadow, mid, lit) colours."""
    X, Y = P.grid()
    h, w = P.shape
    D = np.full((h, w), 1e9, F32)
    SN = np.zeros((h, w), F32)       # across coordinate -1..1
    AL = np.zeros((h, w), F32)       # along (px)
    RR = np.ones((h, w), F32)        # radius
    AX = np.zeros((h, w), F32)
    AY = np.full((h, w), -1.0, F32)
    DEP = np.zeros((h, w), F32)
    for (x0, y0, x1, y1, r0, r1, s0, s1, dep) in segs:
        rm = max(r0, r1)
        bx0 = int(math.floor(min(x0, x1) - rm - 2)) - P.x0
        bx1 = int(math.ceil(max(x0, x1) + rm + 2)) - P.x0
        by0 = int(math.floor(min(y0, y1) - rm - 2)) - P.y0
        by1 = int(math.ceil(max(y0, y1) + rm + 2)) - P.y0
        bx0, by0 = max(bx0, 0), max(by0, 0)
        bx1, by1 = min(bx1, w), min(by1, h)
        if bx0 >= bx1 or by0 >= by1:
            continue
        sx = X[by0:by1, bx0:bx1]
        sy = Y[by0:by1, bx0:bx1]
        vx, vy = x1 - x0, y1 - y0
        L2 = vx * vx + vy * vy + 1e-9
        L = math.sqrt(L2)
        t = np.clip(((sx - x0) * vx + (sy - y0) * vy) / L2, 0, 1)
        r = r0 + (r1 - r0) * t
        dx = sx - (x0 + vx * t)
        dy = sy - (y0 + vy * t)
        dd = np.sqrt(dx * dx + dy * dy)
        d = dd - np.maximum(r, 0.35)
        sub = D[by0:by1, bx0:bx1]
        upd = d < sub
        if not upd.any():
            continue
        sub[upd] = d[upd]
        cr = (dx * (-vy) + dy * vx) / L
        SN[by0:by1, bx0:bx1][upd] = np.clip(cr / np.maximum(r, 0.35), -1, 1)[upd]
        AL[by0:by1, bx0:bx1][upd] = (s0 + (s1 - s0) * t)[upd]
        RR[by0:by1, bx0:bx1][upd] = r[upd]
        AX[by0:by1, bx0:bx1][upd] = vx / L
        AY[by0:by1, bx0:bx1][upd] = vy / L
        DEP[by0:by1, bx0:bx1][upd] = dep
    a = clamp01(0.5 - D)
    # cylinder normal
    s = SN
    nz = np.sqrt(np.clip(1 - s * s, 0, 1))
    px, py = -AY, AX
    L3 = _l3(light, 0.55)
    diff = np.clip((px * s) * L3[0] + (py * s) * L3[1] + nz * L3[2], 0, 1)
    arc = s * RR * 1.4
    shadow_c, mid_c, lit_c = (rgb(c) for c in bark[:3])
    tone = np.clip(diff, 0, 1)
    if kind == "furrow":            # oak / generic: vertical ridges and furrows
        fw = 5.0 * S
        n = fbm2d_at(arc / fw + DEP * 3.1, AL / (fw * 5.0), 3, seed=seed, mode="ridged")
        tex = 1 - 0.55 * (1 - n) ** 2.2
        tone = tone * (0.55 + 0.45 * tex) + 0.12 * (tex - 0.6)
    elif kind == "smooth":          # beech / maple / cherry: smooth with fine horizontal lenticels
        n = fbm2d_at(arc / (4 * S), AL / (1.0 * S), 2, seed=seed)
        tone = tone * (0.92 + 0.12 * n)
    elif kind == "birch":
        n = fbm2d_at(arc / (14 * S), AL / (3.2 * S), 3, seed=seed)
        n2 = fbm2d_at(arc / (5 * S), AL / (1.2 * S), 2, seed=seed + 9)
        mark = smoothstep(0.28, 0.42, n + 0.25 * n2) * (RR > 1.6 * S)
        tone = tone * (0.85 + 0.15 * n2)
    elif kind == "plates":          # pine: orange plates
        f1, f2, _, _, cid = _worley(arc * 1.6, AL, 9 * S, seed)
        edge = smoothstep(0.0, 2.2 * S, f2 - f1)
        tone = tone * (0.6 + 0.4 * edge) * (0.85 + 0.3 * cid)
    elif kind == "rings":           # palm: rings and diamond scars
        ring = (AL / (9.0 * S)) % 1.0
        rr_ = smoothstep(0.0, 0.25, ring) * smoothstep(1.0, 0.7, ring)
        tone = tone * (0.55 + 0.45 * rr_)
    elif kind == "papery":          # baobab / smooth grey with wrinkles
        n = fbm2d_at(arc / (18 * S), AL / (40 * S), 4, seed=seed)
        tone = tone * (0.85 + 0.25 * n)
    col = np.where((tone < 0.5)[..., None], _blend_rgb(shadow_c, mid_c, tone / 0.5),
                   _blend_rgb(mid_c, lit_c, (tone - 0.5) / 0.5))
    if kind == "birch":
        col = _blend_rgb(col, rgb(bark[3] if len(bark) > 3 else "#221c18"), (mark * (0.85 + 0.1 * tone) * 0.92))
    # root / joint darkening at thin AO
    col = col * (0.75 + 0.25 * nz[..., None])
    return col.astype(F32), a.astype(F32), dict(D=D, AL=AL, SN=SN, RR=RR)


# ============================================================================
# canopy: union of leaf-cluster ellipsoids, leaf-quantised, shaded as volume
# ============================================================================
def _render_canopy(P, items, light, palette, leaf_px, *, seed=0, style="painted", leaf_shape=(1.0, 0.55),
                   leaf_align="random", holes=0.35, global_c=None, gloss=0.0, tints=None, edge_soft=1.0,
                   speckle=0.18, needle=False):
    """items: array (N, 7) = cx, cy, rx, ry, angle, zc, tint (0~1, chooses palette variant).
    ``palette`` = list of colour ramps (each 5 colours dark → light); tint picks one."""
    X, Y = P.grid()
    h, w = P.shape
    NEG = -1e9
    ZL = np.full((h, w), NEG, F32)
    ZP = np.full((h, w), NEG, F32)
    OWN = np.full((h, w), -1, np.int32)
    NX = np.zeros((h, w), F32)
    NY = np.zeros((h, w), F32)
    NZ = np.ones((h, w), F32)
    PD = np.full((h, w), -1e4, F32)  # pixel depth inside the owner cluster, px (negative outside)
    flat = style in ("flat", "cute")
    lp = max(float(leaf_px), 0.9)
    if not flat:
        F1, F2, FX, FY, CID = _worley(X, Y, lp, seed)
    else:
        FX, FY, F1, F2 = X, Y, np.zeros_like(X), np.ones_like(X)
        CID = np.zeros_like(X)
    if global_c is None:
        gcx, gcy = items[:, 0].mean(), items[:, 1].mean()
        grx = max((items[:, 0] + items[:, 2]).max() - gcx, (gcx - (items[:, 0] - items[:, 2])).min() * -1, 1)
        gry = max((items[:, 1] + items[:, 3]).max() - gcy, gcy - (items[:, 1] - items[:, 3]).min(), 1)
        grx = max((items[:, 0] + items[:, 2]).max() - gcx, gcx - (items[:, 0] - items[:, 2]).min(), 1)
    else:
        gcx, gcy, grx, gry = global_c
    margin = lp * 1.4 + 2
    for i, (cx, cy, rx, ry, ang, zc, tint) in enumerate(items):
        rm = max(rx, ry) + margin
        bx0, bx1 = max(int(cx - rm) - P.x0, 0), min(int(cx + rm) + 2 - P.x0, w)
        by0, by1 = max(int(cy - rm) - P.y0, 0), min(int(cy + rm) + 2 - P.y0, h)
        if bx0 >= bx1 or by0 >= by1:
            continue
        sl = (slice(by0, by1), slice(bx0, bx1))
        rz = (rx + ry) * 0.5
        # leaf point
        u, v = _rot(FX[sl] - cx, FY[sl] - cy, ang)
        q = (u / rx) ** 2 + (v / ry) ** 2
        zl = zc + rz * np.sqrt(np.clip(1 - q, 0, 1))
        zl = np.where(q < 1, zl, NEG)
        upd = zl > ZL[sl]
        if upd.any():
            ZL[sl][upd] = zl[upd]
            OWN[sl][upd] = i
            nu, nv = u / rx, v / ry
            nz = np.sqrt(np.clip(1 - nu * nu - nv * nv, 0.0, 1))
            c, s = math.cos(ang), math.sin(ang)
            NX[sl][upd] = (nu * c - nv * s)[upd]
            NY[sl][upd] = (nu * s + nv * c)[upd]
            NZ[sl][upd] = nz[upd]
        # pixel
        u2, v2 = _rot(X[sl] - cx, Y[sl] - cy, ang)
        q2 = np.sqrt((u2 / rx) ** 2 + (v2 / ry) ** 2)
        zp = np.where(q2 < 1, zc + rz * np.sqrt(np.clip(1 - q2 * q2, 0, 1)), NEG)
        upd2 = zp > ZP[sl]
        ZP[sl][upd2] = zp[upd2]
        pd = (1 - q2) * min(rx, ry)
        PD[sl] = np.maximum(PD[sl], pd)
    inside_leaf = OWN >= 0
    if flat:
        alpha = _aa(-PD, edge_soft)
    else:
        # leaf shape around each feature point
        if leaf_align == "random":
            la = CID * math.pi * 2
        elif leaf_align == "down":
            la = np.full_like(CID, math.pi / 2) + (CID - 0.5) * 0.6
        else:                                   # align with the owning cluster's angle
            angs = items[:, 4]
            la = np.where(inside_leaf, angs[np.clip(OWN, 0, None)], 0) + (CID - 0.5) * 0.9
        du, dv = _rot(X - FX, Y - FY, la)
        e1, e2 = leaf_shape
        rr = np.sqrt((du / (lp * 0.78 * e1)) ** 2 + (dv / (lp * 0.78 * e2)) ** 2)
        cov = clamp01((1 - rr) * lp * 0.5 + 0.5) * inside_leaf
        deep = clamp01((PD - lp * 0.9) / max(lp * 0.4, 0.8)) * (ZP > NEG / 2)
        rmean0 = float(np.mean((items[:, 2] + items[:, 3]) * 0.5))
        if holes > 0:
            hs = lp * 4.5
            h1, _, _, _, hid = _worley(FX, FY, hs, seed + 77)          # evaluated per leaf → leafy hole edges
            pdl = sample(PD, FX - P.x0, FY - P.y0)
            thin = clamp01(1.3 - pdl / (rmean0 * 0.5))
            hole = ((hid < holes * 0.6) & (h1 < hs * 0.5) & (thin > 0.5)).astype(F32)
            deep = deep * (1 - hole)
            cov = cov * (1 - hole * (CID < 0.75))
        alpha = np.maximum(deep, cov)
    # ---- shading ----
    L3 = _l3(light, 0.62)
    gnx = (FX - gcx) / grx
    gny = (FY - gcy) / gry
    gnz = np.sqrt(np.clip(1 - gnx ** 2 - gny ** 2, 0.0, 1))
    wg = 0.42
    nx = NX * (1 - wg) + gnx * wg
    ny = NY * (1 - wg) + gny * wg
    nz = NZ * (1 - wg) + gnz * wg
    if not flat:
        jit = 0.55
        nx = nx + (CID - 0.5) * jit
        ny = ny + (_hash((CID * 9973).astype(np.int64), 7, seed) - 0.5) * jit
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    diff = np.clip((nx * L3[0] + ny * L3[1] + nz * L3[2]) / nn, 0, 1)
    # cluster crevices (ambient occlusion): lower z relative to the owner's centre
    zc_own = np.where(inside_leaf, items[np.clip(OWN, 0, None), 5], 0)
    rz_own = np.where(inside_leaf, (items[np.clip(OWN, 0, None), 2] + items[np.clip(OWN, 0, None), 3]) * 0.5, 1)
    zrel = np.clip((ZL - zc_own) / rz_own, 0, 1)
    valid = ZP > NEG / 2
    zmin, zmax = (float(ZP[valid].min()), float(ZP[valid].max())) if valid.any() else (0.0, 1.0)
    zglob = np.clip((np.where(inside_leaf, ZL, zmin) - zmin) / max(zmax - zmin, 1e-3), 0, 1)
    ao = 0.3 + 0.5 * zrel ** 0.8 + 0.2 * zglob
    # cast shadows between clusters: march towards the light over the pixel heightfield
    Zs = np.where(ZP > NEG / 2, ZP, zmin - 1).astype(F32)
    rmean = float(np.mean((items[:, 2] + items[:, 3]) * 0.5))
    lx, ly = _l2(light)
    sh = np.zeros((h, w), F32)
    for k in (0.12, 0.25, 0.45, 0.7):
        dist = rmean * k
        zz = sample(Zs, X - P.x0 + lx * dist, Y - P.y0 + ly * dist)
        sh = np.maximum(sh, clamp01((zz - np.where(inside_leaf, ZL, Zs) - dist * 0.55) / (rmean * 0.25)))
    light_v = diff * (1 - 0.65 * sh)
    tone = np.clip(light_v * 0.95 * (0.45 + 0.55 * ao) + 0.12 * ao - 0.04, 0, 1) ** 1.1
    if style == "flat":
        tone = np.floor(tone * 4.0 + 0.5) / 4.0
    elif style == "cute":
        tone = smoothstep(0.32, 0.4, tone) * 0.45 + smoothstep(0.62, 0.7, tone) * 0.35 + 0.18
    ramps = palette if isinstance(palette[0], (list, tuple)) else [palette]
    tsel = np.where(inside_leaf, items[np.clip(OWN, 0, None), 6], 0)
    col = np.zeros((h, w, 3), F32)
    nr = len(ramps)
    if nr == 1:
        col = _ramp(ramps[0], tone)
    else:
        idx = np.clip((tsel * nr).astype(np.int32), 0, nr - 1)
        for k in range(nr):
            mk = idx == k
            if mk.any():
                col[mk] = _ramp(ramps[k], tone[mk])
    if not flat:
        # per-leaf brightness / hue speckle; the gaps between leaves fall into shadow (depth, not tiles)
        sp_ = (CID - 0.5) * speckle
        col = col * (1 + sp_)[..., None]
        under = _ramp(ramps[0], tone * 0.42)
        col = _blend_rgb(under, col, clamp01(cov * 1.15))
    if gloss > 0:
        spec = np.clip(2 * diff * nz / nn - L3[2], 0, 1) ** 12
        col = col + gloss * spec[..., None]
    if style == "cute":
        edge = clamp01(1 - PD / max(2.2 * leaf_px, 1.5)) * (alpha > 0.02)
        col = col * (1 - 0.45 * smoothstep(0.55, 0.9, edge))[..., None]
    return col.astype(F32), alpha.astype(F32), dict(OWN=OWN, ZL=ZL, PD=PD, tone=tone)


# ============================================================================
# palettes
# ============================================================================
PALETTES = {
    # foliage ramps: deep shadow → shadow → mid → lit → highlight
    "oak": ["#132416", "#223d1d", "#3d6128", "#6f8f38", "#adc160"],
    "maple": ["#14271a", "#244422", "#3f6c2c", "#74a03e", "#b8d270"],
    "birch": ["#1d3219", "#365a25", "#5f8c34", "#98bf52", "#d8e88c"],
    "spring": ["#2a4a1f", "#4c7a2c", "#7aab3e", "#addc62", "#e2f5a0"],
    "willow": ["#1f2f17", "#3b5626", "#668a35", "#9cbe55", "#d6e891"],
    "conifer": ["#08150f", "#11281c", "#1f3f2c", "#365e42", "#5f8a66"],
    "pine": ["#0f1d14", "#1b3122", "#2c4a33", "#4a6e4a", "#7d9e74"],
    "cypress": ["#0a170f", "#14291a", "#22402a", "#3c5f3c", "#6b8a5e"],
    "palm": ["#1b3014", "#2f5220", "#4e7c2c", "#7fac44", "#bfd878"],
    "tropical": ["#0e2a16", "#1a4524", "#2c6b34", "#4f9a48", "#93c86c"],
    "olive": ["#1e2a20", "#344535", "#566c52", "#879c78", "#bccaa8"],
    "autumn_red": ["#360d0b", "#6a1a12", "#a52e1b", "#d75a2a", "#f39a52"],
    "autumn_orange": ["#3a1c0a", "#74360f", "#b5601b", "#e1922f", "#f7c65c"],
    "autumn_gold": ["#352709", "#6b5011", "#a8851e", "#d9b73a", "#f6e27c"],
    "autumn_brown": ["#2a1a0e", "#4f3317", "#7a5224", "#a6783a", "#cfa45e"],
    "cherry": ["#7c3d5a", "#b5668a", "#e297b4", "#f7c4d6", "#fff1f6"],
    "cherry_white": ["#7d6f78", "#b8a9b4", "#e0d4dc", "#f6eef3", "#ffffff"],
    "wisteria": ["#3e2a5a", "#61458a", "#8c6fbd", "#b9a2e0", "#e6dbfa"],
    "jacaranda": ["#2e2c6a", "#4a4596", "#7068c6", "#9d97e4", "#d4d0fa"],
    "hedge": ["#0f2214", "#1c3a1f", "#2f5a2c", "#4f823c", "#8ab562"],
    "bush": ["#13261a", "#22401f", "#3a632b", "#64903f", "#a4c46a"],
    "cute": ["#2c5a3a", "#3f7f48", "#5ea65a", "#8fcf72", "#c8ef9c"],
}

BARKS = {
    "oak": ("#1f1813", "#4a3a2c", "#7d6650"),
    "maple": ("#2a2420", "#5a4e44", "#8c7f72"),
    "birch": ("#8a857c", "#d6d1c6", "#f6f3ec", "#1e1a17"),
    "pine": ("#3e2016", "#8a4a2a", "#cf8656"),
    "conifer": ("#1c1410", "#3f2e22", "#6a5040"),
    "cherry": ("#24130f", "#4f2b22", "#80503f"),
    "palm": ("#4a3c2c", "#8a7458", "#c4ab86"),
    "baobab": ("#4f4136", "#8e7a68", "#c4b09a"),
    "dead": ("#3a3532", "#6e6660", "#a69c92"),
    "willow": ("#241c16", "#4e4033", "#7f6c58"),
    "olive": ("#3a3630", "#6e665a", "#a49a88"),
    "cute": ("#4a2e1e", "#7a4e30", "#a8744a"),
}


def _palette(name):
    """A palette by name (ramp of colours dark to light); unknown names get a did-you-mean."""
    if name not in PALETTES:
        raise _unknown_name("palette", name, PALETTES, "Valid palettes: %s." % ", ".join(sorted(PALETTES)))
    return PALETTES[name]


def _season_palette(base, season, species_autumn=("autumn_orange", "autumn_red", "autumn_gold")):
    if season in (None, "summer"):
        return [_palette(base) if isinstance(base, str) else base]
    if season == "spring":
        b = _palette(base) if isinstance(base, str) else base
        return [[_hex_mix(c, "#c8f070", 0.25 + 0.05 * i) for i, c in enumerate(b)]]
    if season == "autumn":
        return [PALETTES[k] for k in species_autumn]
    return [_palette(base) if isinstance(base, str) else base]


def _hex_mix(a, b, t):
    c = rgb(a) * (1 - t) + rgb(b) * t
    return tuple(float(v) for v in c)


def _as_ramps(palette):
    """Accept a palette name, a single ramp (list of colours), or a list of ramps."""
    if isinstance(palette, str):
        return [_palette(palette)]
    if isinstance(palette[0], (list, tuple)) and len(palette[0]) and not isinstance(palette[0][0], (int, float)):
        return [list(p) for p in palette]
    if isinstance(palette[0], (list, tuple)) and len(palette[0]) == 5:
        return [list(p) for p in palette]
    return [list(palette)]


# ============================================================================
# trees
# ============================================================================
# Each preset: builder + skeleton parameters (unit space: height ≈ 1) + looks.
TREES = {
    "oak": dict(builder="broad", trunk_h=0.3, trunk_r=0.052, flare=0.9, depth=5, kids=(2, 3), spread=40,
                spread_jitter=12, len_ratio=0.74, first_len=0.28, up=0.06, floor=0.3, min_foliage=0.3, wiggle=17, cluster=0.08, aspect=0.8,
                node_p=0.35, palette="oak", autumn=("autumn_brown", "autumn_orange", "autumn_gold"), bark="oak",
                bark_kind="furrow", leaf=0.010, lean0=5),
    "maple": dict(builder="broad", trunk_h=0.33, trunk_r=0.040, flare=0.6, depth=5, kids=(2, 3), spread=30,
                  spread_jitter=9, len_ratio=0.76, first_len=0.26, up=0.18, wiggle=10, cluster=0.085, aspect=0.9,
                  node_p=0.6, palette="maple", autumn=("autumn_red", "autumn_orange", "autumn_red"), bark="maple",
                  bark_kind="smooth", leaf=0.010),
    "birch": dict(builder="leader", trunk_h=1.0, trunk_r=0.024, flare=0.35, depth=3, laterals=12,
                  lat_range=(0.3, 0.97), lat_angle=34, lat_len=0.3, lat_rad=0.4, lat_profile="ellipse",
                  kids=(2, 2), spread=24, len_ratio=0.6, up=0.18, up_d=-0.06, wiggle=9, cluster=0.065, aspect=1.2,
                  node_p=0.45, fill=0.5, palette="birch", autumn=("autumn_gold", "autumn_gold", "autumn_orange"), bark="birch",
                  bark_kind="birch", leaf=0.008, holes=0.75, deciduous=True),
    "poplar": dict(builder="column", width=0.20, cluster=0.05, palette="birch", autumn=("autumn_gold",),
                   bark="maple", bark_kind="smooth", trunk_r=0.022, leaf=0.008, shape_pow=0.6, holes=0.3,
                   deciduous=True),
    "cypress": dict(builder="column", width=0.13, cluster=0.04, palette="cypress", bark="conifer",
                    bark_kind="furrow", trunk_r=0.02, leaf=0.007, shape_pow=0.75, holes=0.08, needle=True,
                    leaf_shape=(1.0, 0.5)),
    "spruce": dict(builder="tiered", width=0.42, tiers=24, palette="conifer", bark="conifer", bark_kind="furrow",
                   trunk_r=0.018, leaf=0.0065, droop=0.45, holes=0.15),
    "fir": dict(builder="tiered", width=0.4, tiers=22, palette="pine", bark="conifer", bark_kind="furrow",
                trunk_r=0.018, leaf=0.0065, droop=0.3, holes=0.1, body=0.0),
    "pine": dict(builder="leader", trunk_h=1.0, trunk_r=0.028, flare=0.3, depth=3, laterals=7,
                 lat_range=(0.55, 0.96), lat_angle=72, lat_len=0.38, lat_rad=0.4, lat_profile="ellipse",
                 kids=(2, 3), spread=34, len_ratio=0.62, up=0.35, up_d=0.05, wiggle=14, cluster=0.085, aspect=0.45,
                 node_p=0.35, palette="pine", bark="pine", bark_kind="plates", leaf=0.009, needle=True,
                 leaf_shape=(1.0, 0.32), lean0=6, holes=0.4),
    "willow": dict(builder="willow", trunk_h=0.3, trunk_r=0.034, flare=0.7, depth=4, kids=(2, 3), spread=30,
                   spread_jitter=10, len_ratio=0.72, first_len=0.27, up=0.3, up_d=-0.16, wiggle=12, cluster=0.085,
                   aspect=0.8, node_p=0.4, palette="willow", autumn=("autumn_gold",), bark="willow",
                   bark_kind="furrow", leaf=0.0075, holes=0.3),
    "cherry": dict(builder="broad", trunk_h=0.26, trunk_r=0.045, flare=0.6, depth=5, kids=(2, 3), spread=44,
                   spread_jitter=14, len_ratio=0.74, first_len=0.32, up=0.02, wiggle=15, cluster=0.055, aspect=0.7,
                   node_p=0.3, palette="cherry", bark="cherry", bark_kind="smooth", leaf=0.0065, holes=0.95,
                   leaf_shape=(0.8, 0.7), blossom=True, summer="oak"),
    "apple": dict(builder="broad", trunk_h=0.30, trunk_r=0.045, flare=0.5, depth=5, kids=(2, 3), spread=44,
                  spread_jitter=14, len_ratio=0.72, first_len=0.28, up=0.08, wiggle=16, cluster=0.09, aspect=0.85,
                  node_p=0.55, palette="maple", autumn=("autumn_gold", "autumn_orange"), bark="oak",
                  bark_kind="furrow", leaf=0.010, fruit="#c8231e", spring="cherry_white"),
    "baobab": dict(builder="baobab", depth=3, kids=(2, 3), spread=42, spread_jitter=16, len_ratio=0.7, up=0.35,
                   wiggle=18, cluster=0.055, aspect=0.75, node_p=0.2, palette="olive", bark="baobab",
                   bark_kind="papery", leaf=0.009, holes=0.5),
    "olive": dict(builder="broad", trunk_h=0.3, trunk_r=0.055, flare=0.7, depth=4, kids=(2, 3), spread=46,
                  spread_jitter=18, len_ratio=0.72, first_len=0.32, up=0.1, wiggle=24, cluster=0.09, aspect=0.75,
                  node_p=0.6, palette="olive", bark="olive", bark_kind="furrow", leaf=0.008, holes=0.45, evergreen=True),
    "bonsai": dict(builder="bonsai", depth=3, kids=(2, 3), spread=55, spread_jitter=14, len_ratio=0.66,
                   wiggle=12, cluster=0.1, aspect=0.45, node_p=0.0, palette="pine", up=0.15, bark="oak", bark_kind="furrow",
                   leaf=0.009, holes=0.12, trunk_r=0.05, evergreen=True),
    "winter": dict(builder="broad", trunk_h=0.30, trunk_r=0.045, flare=0.7, depth=6, kids=(2, 3), spread=36,
                   spread_jitter=12, len_ratio=0.74, first_len=0.27, up=0.14, wiggle=14, cluster=0.0, bark="maple",
                   bark_kind="smooth", bare=True),
    "dead": dict(builder="broad", trunk_h=0.34, trunk_r=0.05, flare=1.0, depth=5, kids=(2, 3), spread=44,
                 spread_jitter=20, len_ratio=0.72, first_len=0.3, up=0.06, wiggle=22, cluster=0.0, bark="dead",
                 bark_kind="furrow", bare=True, broken=0.3),
    "palm": dict(builder="palm", palette="palm", bark="palm"),
    "coconut palm": dict(builder="palm", palette="palm", bark="palm", coconuts=True, curve=0.35),
    "cute": dict(builder="broad", trunk_h=0.42, trunk_r=0.06, flare=0.4, depth=3, kids=(2, 3), spread=40,
                 spread_jitter=8, len_ratio=0.75, first_len=0.25, up=0.3, wiggle=6, cluster=0.2, aspect=0.95,
                 node_p=0.3, palette="cute", bark="cute", bark_kind="smooth", leaf=0.01, style="cute"),
}
_TREE_ALIAS = {"autumn": "maple", "bare": "winter", "fall": "maple", "conifer": "spruce", "christmas": "fir",
               "sakura": "cherry", "column": "poplar", "evergreen": "spruce"}


def _items_from_skel(sk, rng, sp, fullness, H=1.0):
    """Foliage clusters at branch tips and along the last branches."""
    cr = sp["cluster"] * H * (0.75 + 0.35 * fullness)
    asp = sp.get("aspect", 0.85)
    out = []
    for (x, y, a, dep, ln) in sk.tips:
        r = cr * rng.uniform(0.7, 1.25)
        out.append([x, y, r, r * asp * rng.uniform(0.85, 1.15), rng.uniform(-0.3, 0.3), 0.0, rng.random()])
    for (x, y, a, dep) in sk.nodes:
        if rng.random() < sp.get("node_p", 0.5) * fullness:
            r = cr * rng.uniform(0.55, 0.95)
            out.append([x, y, r, r * asp, rng.uniform(-0.3, 0.3), 0.0, rng.random()])
    return np.array(out, np.float64).reshape(-1, 7)


def _fill_crown(items, rng, frac):
    """Add deeper clusters inside the crown envelope so the canopy reads as one volume, not a ring of clumps."""
    if len(items) < 4 or frac <= 0:
        return items
    w = items[:, 2] ** 2
    cx, cy = np.average(items[:, 0], weights=w), np.average(items[:, 1], weights=w)
    rx = np.percentile(np.abs(items[:, 0] - cx), 80)
    ry = np.percentile(np.abs(items[:, 1] - cy), 80)
    r = float(np.median(items[:, 2]))
    k = int(len(items) * frac)
    a = rng.uniform(0, 2 * math.pi, k)
    rr = np.sqrt(rng.uniform(0, 1, k)) * 0.85
    fill = np.zeros((k, 7))
    fill[:, 0] = cx + np.cos(a) * rr * rx
    fill[:, 1] = cy + np.sin(a) * rr * ry
    fill[:, 2] = r * rng.uniform(0.9, 1.3, k)
    fill[:, 3] = fill[:, 2] * float(np.median(items[:, 3] / items[:, 2]))
    fill[:, 4] = rng.uniform(-0.3, 0.3, k)
    fill[:, 5] = -1.0                       # marker: pushed back by _bulge
    fill[:, 6] = rng.random(k)
    return np.concatenate([fill, items])


def _bulge(items, rng, depth_jitter=0.3):
    """Give every cluster a z on a big ellipsoid so the whole canopy reads as one volume."""
    if len(items) == 0:
        return items
    cx = np.average(items[:, 0], weights=items[:, 2])
    cy = np.average(items[:, 1], weights=items[:, 2])
    Rx = max(np.max(np.abs(items[:, 0] - cx) + items[:, 2]), 1e-6)
    Ry = max(np.max(np.abs(items[:, 1] - cy) + items[:, 3]), 1e-6)
    q = ((items[:, 0] - cx) / Rx) ** 2 + ((items[:, 1] - cy) / Ry) ** 2
    rz = (Rx + Ry) * 0.5
    back = items[:, 5] < -0.5
    items[:, 5] = rz * np.sqrt(np.clip(1 - q, 0, 1)) * 0.8 + rng.normal(0, depth_jitter, len(items)) * items[:, 2]
    items[back, 5] -= items[back, 2] * 0.9
    return items


def _skel_broad(rng, sp, lean, age):
    sk = _Skel()
    depth = sp["depth"]
    a0 = -math.pi / 2 + math.radians(lean * 25 + rng.normal(0, sp.get("lean0", 4)))
    tr = sp.get("trunk_r", 0.05)
    th = sp.get("trunk_h", 0.3)
    spc = dict(sp, depth=depth, nseg=4)
    _grow(sk, rng, dict(spc, depth=0, nseg=5, wiggle=sp.get("wiggle", 12) * 0.5, up=0.4), 0.0, 0.0, a0, th, tr, 0)
    sk.tips.clear()
    sk.nodes.clear()
    x, y = sk.segs[-1][2], sk.segs[-1][3]
    rtop = sk.segs[-1][5]
    n = int(rng.integers(sp.get("kids", (2, 3))[0], sp.get("kids", (2, 3))[1] + 2))
    spread = math.radians(sp.get("spread", 40)) * 1.15
    for k in range(n):
        off = (k / (n - 1) - 0.5) * 2 if n > 1 else 0
        aa = a0 + off * spread + rng.normal(0, math.radians(sp.get("spread_jitter", 10)))
        ll = sp.get("first_len", 0.3) * rng.uniform(0.85, 1.15) * (1.0 - 0.25 * abs(off))
        _grow(sk, rng, spc, x, y, aa, ll, rtop * (1.0 / n) ** (1 / 2.3) * rng.uniform(0.9, 1.1), 1, th)
    return sk


def _add_roots(sk, rng, tr, flare, n=4):
    """Buttress roots: thick where they leave the trunk, spreading low along the ground."""
    for k in range(n):
        side = -1 if k % 2 == 0 else 1
        L = tr * rng.uniform(1.8, 2.6) * flare
        sk.segs.insert(0, (side * tr * 0.3, -tr * 2.0, side * L * 0.55, -tr * 0.3, tr * 0.5, tr * 0.26, 0, L, 0))
        sk.segs.insert(0, (side * L * 0.55, -tr * 0.3, side * L, tr * 0.15, tr * 0.26, tr * 0.06, 0, L, 0))


def _trunk_profile(sk, tr, flare):
    """Flare the bottom of the trunk."""
    out = []
    for s in sk.segs:
        x0, y0, x1, y1, r0, r1, s0, s1, dep = s
        if dep == 0:
            f0 = 1 + 0.5 * flare * math.exp(-abs(y0) / (tr * 3.0))
            f1 = 1 + 0.5 * flare * math.exp(-abs(y1) / (tr * 3.0))
            s = (x0, y0, x1, y1, r0 * f0, r1 * f1, s0, s1, dep)
        out.append(s)
    sk.segs = out


def _build_tree(species, sp, rng, lean, fullness, age, season):
    """Return (skeleton, items (unit space), extras dict)."""
    b = sp["builder"]
    extras = {}
    if b in ("broad", "willow", "bonsai", "baobab"):
        if b == "baobab":
            sk = _Skel()
            prof = [0.11, 0.135, 0.14, 0.13, 0.11, 0.085, 0.06]
            ys = np.linspace(0, -0.52, len(prof))
            for i in range(len(prof) - 1):
                sk.segs.append((rng.normal(0, 0.004), ys[i], rng.normal(0, 0.004), ys[i + 1], prof[i], prof[i + 1],
                                -ys[i], -ys[i + 1], 0))
            spc = dict(sp, nseg=3)
            n = 5
            for k in range(n):
                off = (k / (n - 1) - 0.5) * 2
                aa = -math.pi / 2 + off * math.radians(62) + rng.normal(0, 0.15)
                _grow(sk, rng, spc, off * 0.04, ys[-1] + 0.01, aa, 0.2 * rng.uniform(0.8, 1.1), 0.035, 1, 0.52)
        elif b == "bonsai":
            sk = _Skel()
            pts = [(0, 0), (0.08, -0.1), (-0.06, -0.22), (0.05, -0.36), (-0.02, -0.48)]
            r = sp.get("trunk_r", 0.05)
            for i in range(len(pts) - 1):
                sk.segs.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], r * (1 - 0.18 * i),
                                r * (1 - 0.18 * (i + 1)), i * 0.12, (i + 1) * 0.12, 0))
            spc = dict(sp, nseg=3)
            for (px, py), ang in zip(pts[1:], (-0.2, math.pi + 0.25, -0.15, -math.pi / 2)):
                _grow(sk, rng, spc, px, py, ang, 0.13 * rng.uniform(0.85, 1.1), r * 0.45, 1, 0.3)
        else:
            sk = _skel_broad(rng, sp, lean, age)
        tr = sp.get("trunk_r", 0.05)
        if b != "baobab":
            _trunk_profile(sk, tr, sp.get("flare", 0.6))
            if b != "bonsai":
                _add_roots(sk, rng, tr, 1.0 + sp.get("flare", 0.6) * 0.5)
        items = np.zeros((0, 7))
        if not sp.get("bare") and not (season == "winter" and not sp.get("evergreen")):
            items = _items_from_skel(sk, rng, sp, fullness)
            mf = sp.get("min_foliage", 0.2)
            items = items[items[:, 1] < -mf]
            if b != "willow":
                # twigs that ended below the foliage line lost their leaves: drop them instead of leaving bare sticks
                sk.segs = [s_ for s_ in sk.segs if not (s_[8] >= 2 and s_[1] > -mf * 0.9 and s_[3] > -mf * 0.9)]
            items = _fill_crown(items, rng, sp.get("fill", 0.35) * fullness)
            if b == "willow":
                strands = []
                for it in items:
                    for _ in range(int(rng.integers(3, 6))):
                        sx = it[0] + rng.uniform(-1, 1) * it[2] * 0.9
                        sy = it[1] + rng.uniform(-0.6, 0.3) * it[3]
                        strands.append((sx, sy, max(-sy - 0.06, 0.05) * rng.uniform(0.6, 1.0), rng.random()))
                extras["strands"] = strands
                items[:, 3] *= 0.8
            items = _bulge(items, rng)
    elif b == "leader":
        sk = _Skel()
        tr = sp.get("trunk_r", 0.03)
        a0 = -math.pi / 2 + math.radians(lean * 25 + rng.normal(0, sp.get("lean0", 3)))
        _grow(sk, rng, dict(sp, nseg=8), 0.0, 0.0, a0, 1.0, tr, 0, leader=True)
        _trunk_profile(sk, tr, sp.get("flare", 0.4))
        _add_roots(sk, rng, tr, 1.0, n=2)
        if season == "winter" and sp.get("deciduous"):          # birch in winter: the bare branches only
            items = np.zeros((0, 7))
        else:
            items = _items_from_skel(sk, rng, sp, fullness)
            items = _fill_crown(items, rng, sp.get("fill", 0.25) * fullness)
            tx, ty = [s for s in sk.segs if s[8] == 0][-1][2:4]
            cr = sp["cluster"]
            items = np.concatenate([items, np.array([[tx, ty + cr * 0.3, cr, cr * sp.get("aspect", 1), 0, 0, 0.5]])])
            if sp.get("needle"):
                items[:, 4] = rng.uniform(-0.25, 0.25, len(items))
            items = _bulge(items, rng)
    elif b == "column" and season == "winter" and sp.get("deciduous"):
        sk = _Skel()                                            # poplar in winter: a narrow bare crown of branches
        tr = sp.get("trunk_r", 0.02)
        bare = dict(sp, nseg=8, depth=3, laterals=26, lat_range=(0.15, 0.99), lat_angle=20, lat_len=0.3, lat_rad=0.4,
                    lat_profile="column", kids=(2, 2), spread=18, len_ratio=0.6, up=0.35, wiggle=7)
        _grow(sk, rng, bare, 0.0, 0.0, -math.pi / 2 + math.radians(lean * 12), 1.0, tr * 1.1, 0, leader=True)
        _trunk_profile(sk, tr, 0.3)
        _add_roots(sk, rng, tr, 1.0, n=2)
        items = np.zeros((0, 7))
    elif b == "column":
        sk = _Skel()
        tr = sp.get("trunk_r", 0.02)
        sk.segs.append((0, 0, 0, -0.35, tr * 1.4, tr, 0, 0.35, 0))
        sk.segs.append((0, -0.35, 0, -0.8, tr, tr * 0.5, 0.35, 0.8, 0))
        n = int(140 * (0.6 + 0.4 * fullness))
        wid = sp["width"]
        pw = sp.get("shape_pow", 0.7)
        its = []
        for _ in range(n):
            t = rng.uniform(0.05, 0.98)
            prof = math.sin(math.pi * t ** pw) ** 0.9 * (1 - t) ** 0.12
            half = wid * 0.5 * prof
            x = rng.uniform(-1, 1) * half * 0.85 + lean * t * 0.15
            r = sp["cluster"] * rng.uniform(0.7, 1.3) * (0.5 + 0.6 * prof)
            its.append([x, -t, r * 0.8, r * 1.25, rng.uniform(-0.3, 0.3), 0, rng.random()])
        items = _bulge(np.array(its), rng, 0.4)
        extras["leaf_align"] = "item" if sp.get("needle") else "random"
    elif b == "tiered":
        sk = _Skel()
        tr = sp.get("trunk_r", 0.02)
        sk.segs.append((0, 0, lean * 0.05, -0.98, tr * 1.6, tr * 0.2, 0, 1.0, 0))
        its = []
        nt = sp["tiers"]
        wid = sp["width"]
        droop = sp.get("droop", 0.25)
        for i in range(nt):
            t = 0.06 + 0.92 * (i + rng.uniform(-0.2, 0.2)) / nt
            xc = lean * 0.05 * t
            half = wid * 0.5 * (1 - t) ** 0.85 * rng.uniform(0.85, 1.08) + 0.018
            nb = 7 if half > 0.05 else 4
            for j in range(nb):
                side = -1 if j % 2 == 0 else 1
                fore = rng.uniform(0.35, 1.0) if j > 1 else 1.0         # branches pointing at / away from us look shorter
                L = half * fore * rng.uniform(0.9, 1.05)
                th = droop * rng.uniform(0.6, 1.3) + (1 - fore) * 0.4
                dx, dy = math.cos(th) * side, math.sin(th)
                rx = L * 0.52
                ry = (0.010 + 0.022 * (1 - t)) * rng.uniform(0.8, 1.2) * (1.0 + 0.6 * (1 - fore))
                its.append([xc + dx * rx, -t + dy * rx, rx, ry, math.atan2(dy, dx), 0, rng.random()])
            # tier body around the trunk
            if sp.get("body", 1.0):
                its.append([xc, -t + 0.012, half * 0.22, 0.015 + 0.015 * (1 - t), 0, 0, rng.random()])
        its.append([lean * 0.05, -0.965, 0.008, 0.03, -math.pi / 2, 0, 0.5])
        items = np.array(its)
        items = _bulge(items, rng, 0.35)
        extras["leaf_align"] = "item"
        extras["leaf_shape"] = (1.0, 0.28)
    else:
        raise ValueError(b)
    return sk, items, extras


def _balance(sk, items, lean, extras, keep=0.3):
    """Shear the crown so its centre of mass sits over the trunk (plus the wanted wind lean)."""
    if len(items):
        w = items[:, 2] ** 2
        cx, cy = float(np.average(items[:, 0], weights=w)), float(np.average(items[:, 1], weights=w))
    elif sk.tips:
        cx, cy = float(np.mean([t[0] for t in sk.tips])), float(np.mean([t[1] for t in sk.tips]))
    else:
        return
    k = (cx - lean * 0.25 * -cy) / max(-cy, 1e-3) * (1 - keep)
    sk.segs = [(x0 - k * max(-y0, 0), y0, x1 - k * max(-y1, 0), y1, r0, r1, s0, s1, d)
               for (x0, y0, x1, y1, r0, r1, s0, s1, d) in sk.segs]
    sk.tips = [(x - k * max(-y, 0), y, a, d, l) for (x, y, a, d, l) in sk.tips]
    if len(items):
        items[:, 0] -= k * np.maximum(-items[:, 1], 0)
    if extras.get("strands"):
        extras["strands"] = [(x - k * max(-y, 0), y, L, d) for (x, y, L, d) in extras["strands"]]


def _xform(sk, items, bx, by, scale):
    segs = [(bx + a * scale, by + b * scale, bx + c * scale, by + d * scale, r0 * scale, r1 * scale, s0 * scale,
             s1 * scale, dep) for (a, b, c, d, r0, r1, s0, s1, dep) in sk.segs]
    it = items.copy()
    if len(it):
        it[:, 0] = bx + it[:, 0] * scale
        it[:, 1] = by + it[:, 1] * scale
        it[:, 2:4] *= scale
        it[:, 5] *= scale
    return segs, it


def _unit_top(sk, items):
    tops = [min(s[1], s[3]) - max(s[4], s[5]) for s in sk.segs] or [0]
    t = min(tops)
    if len(items):
        t = min(t, float(np.min(items[:, 1] - np.maximum(items[:, 2], items[:, 3]))))
    return t


def _tree_ramps(sp, season, palette):
    if palette is not None:
        return _as_ramps(palette)
    if season == "autumn":
        return [PALETTES[k] for k in sp.get("autumn", ("autumn_orange", "autumn_red", "autumn_gold"))]
    if season == "spring" and sp.get("blossom") and not sp.get("spring"):
        return [PALETTES[sp.get("palette", "cherry")]]
    if season in ("spring", "summer") and sp.get(season):        # blossom in spring (cherry, apple), green leaves in summer
        return [PALETTES[sp[season]]]
    return _season_palette(sp.get("palette", "oak"), season)


def tree(cv, species="oak", at=(0.5, 0.9), height=0.6, seed=0, *, season=None, fullness=1.0, age=1.0, lean=0.0,
         palette=None, bark=None, light=LIGHT, style=None, shadow=False, clip=None, log=None, haze=None,
         fruit=None, snow=None, paint=True, detail=1.0, params=None):
    """Paint a tree with real structure. Returns a :class:`Plant`.

    species: oak, maple, birch, poplar, cypress, spruce, fir, pine, willow, cherry, apple, baobab, olive, bonsai,
    winter (bare), dead, palm, coconut palm, cute.
    at: base (u, v) fractions; height: fraction of canvas height (base → crown top).
    season: None/"summer", "spring", "autumn", "winter" (bare branches); fullness 0.3~1.3 (sparse → dense);
    age 0.3~1 (young trees are slimmer with fewer branches); lean: wind lean −1~1 (+ = to the right);
    palette: name in PALETTES, a 5-colour ramp, or several ramps; bark: (shadow, mid, lit) colours;
    style: "painted" (default), "flat", "cute"; shadow: True / strength → ground shadow away from the light;
    haze: (colour, amount) atmospheric fade; log: callable → trunk and canopy become separate log steps;
    fruit: fruit colour (apple has red apples); snow: snow caps on bare branches (default: on for season="winter"); detail: LOD multiplier;
    params: dict overriding preset entries of TREES (e.g. dict(trunk_h=0.4, spread=30, cluster=0.1))."""
    _choice("tree", "season", season, (None, "summer", "spring", "autumn", "winter"))
    _choice("tree", "style", style, (None, "painted", "flat", "cute"))
    if isinstance(palette, str) and palette not in PALETTES:
        raise _unknown("palette", palette, PALETTES)
    key = _TREE_ALIAS.get(species.lower().strip(), species.lower().strip())
    if key not in TREES:
        _check_one_word(species, "tree species", list(TREES) + list(_TREE_ALIAS))
        alt = find(species)
        if alt and alt not in TREES:
            return plant(cv, alt, at=at, height=height, seed=seed, light=light, shadow=shadow)
        if alt not in TREES:
            raise _unknown(f"tree species", species, list(TREES) + list(_TREE_ALIAS),
                           "Try flora.find('...') or flora.names('tree')")
        key = alt
    sp = dict(TREES[key])
    if params:
        sp.update(params)
    if sp["builder"] == "palm":
        return palm(cv, at=at, height=height, seed=seed, light=light, shadow=shadow, lean=lean, paint=paint,
                    coconuts=sp.get("coconuts", False), clip=clip, haze=haze)
    S = _S(cv)
    Hpx = float(height) * cv.h
    pot_lift = 0.0
    if key == "bonsai":
        pot_lift = Hpx * 0.13
        Hpx = Hpx - pot_lift
    style = style or sp.get("style", "painted")
    lod = Hpx * detail / max(S, 1e-6) * S        # px on this canvas
    if "depth" in sp:
        d = sp["depth"]
        if lod < 70:
            d -= 2
        elif lod < 160:
            d -= 1
        d = max(2, int(round(d * (0.6 + 0.4 * min(age, 1.0)))))
        sp["depth"] = d
        if lod < 160 and sp.get("cluster"):
            sp["cluster"] = sp["cluster"] * (1.25 if lod >= 70 else 1.5)
    if age < 1:
        sp["trunk_r"] = sp.get("trunk_r", 0.04) * (0.55 + 0.45 * age)
    sp["wind"] = float(lean) * 2.0
    rng = _rng(seed, 11)
    sk, items, extras = _build_tree(key, sp, rng, lean, fullness, age, season)
    if sp["builder"] in ("broad", "leader", "willow", "baobab"):
        _balance(sk, items, lean, extras)
    top = _unit_top(sk, items)
    scale = Hpx / max(-top, 1e-6)
    bx, by = at[0] * cv.w, at[1] * cv.h - pot_lift
    segs, its = _xform(sk, items, bx, by, scale)
    xs = [min(s[0], s[2]) - max(s[4], s[5]) for s in segs] + [max(s[0], s[2]) + max(s[4], s[5]) for s in segs]
    ys = [min(s[1], s[3]) - max(s[4], s[5]) for s in segs] + [max(s[1], s[3]) + max(s[4], s[5]) for s in segs]
    if len(its):
        xs += list(its[:, 0] - its[:, 2] * 1.3) + list(its[:, 0] + its[:, 2] * 1.3)
        ys += list(its[:, 1] - its[:, 3] * 1.3) + list(its[:, 1] + its[:, 3] * 1.3)
    if key == "bonsai":
        xs += [bx - 0.4 * scale, bx + 0.4 * scale]
        ys += [by + 0.15 * scale]
    pad = 4 + 3 * S
    x0, y0, x1, y1 = _clip_box(cv, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    base, topp = (bx, by), (bx, by - Hpx)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, key, None, base, topp, topp)
    P = _Patch(x0, y0, x1, y1)
    if isinstance(bark, str):
        if bark not in BARKS:
            raise _unknown_name("bark", bark, BARKS, "Valid barks: %s, or a tuple of colours dark to light." %
                                ", ".join(sorted(BARKS)))
        bark = BARKS[bark]
    barkc = bark or BARKS.get(sp.get("bark", "oak"), BARKS["oak"])
    tex_s = S * float(np.clip(Hpx / (700 * S), 0.35, 1.6))
    bk = sp.get("bark_kind", "furrow") if style == "painted" else "smooth"
    bcol, ba, info = _render_bark(P, segs, light, barkc, kind=bk, S=tex_s, seed=seed)
    if key != "bonsai":
        Yg = np.arange(P.y0, P.y1, dtype=F32)[:, None]
        ba = ba * clamp01(by + 0.006 * scale - Yg + 0.5)
    if style in ("flat", "cute"):
        lit = (info["SN"] * (-_l2(light)[0] * -1)) > 0.1
        bcol = np.where(lit[..., None], rgb(barkc[2]) * 0.6 + rgb(barkc[1]) * 0.4, rgb(barkc[1]) * 0.85).astype(F32)
    if snow is None:
        snow = season == "winter"
    if snow:
        above = np.zeros_like(ba)
        k = max(1, int(round(2.0 * S)))
        above[k:] = ba[:-k]
        cap = clamp01((ba - above) * 1.4)
        bcol = _blend_rgb(bcol, rgb("#f2f6fb"), clamp01(gaussian_blur(cap, 0.5) * 1.5))
    trunkP = _Patch(x0, y0, x1, y1)
    trunkP.rgb, trunkP.a = bcol, ba
    canopy_c = topp
    canP = None
    if len(its):
        lp = float(np.clip(Hpx * sp.get("leaf", 0.009), 0.9, 9.5 * S))
        if style == "cute":
            lp = max(lp, 2.0)
        ramps = _tree_ramps(sp, season, palette)
        ccol, ca, cinfo = _render_canopy(P, its, light, ramps, lp, seed=seed + 5, style=style,
                                         leaf_shape=extras.get("leaf_shape", sp.get("leaf_shape", (1.0, 0.55))),
                                         leaf_align=extras.get("leaf_align", "item" if sp.get("needle") else "random"),
                                         holes=sp.get("holes", 0.32) if lod > 120 else 0.0)
        canP = _Patch(x0, y0, x1, y1)
        canP.rgb, canP.a = ccol, ca
        canopy_c = (float(np.average(its[:, 0], weights=its[:, 2] ** 2)),
                    float(np.average(its[:, 1], weights=its[:, 2] ** 2)))
        fr = fruit or sp.get("fruit")
        if fr and season not in ("winter", "spring"):
            _add_fruit(canP, its, fr, rng, light, scale)
        if extras.get("strands"):
            _willow_strands(canP, extras["strands"], bx, by, scale, light, ramps[0], lp, rng, canopy_c, lean)
    if key == "bonsai":
        _bonsai_pot(trunkP, bx, by, scale, light)
    out = _Patch(x0, y0, x1, y1)
    out.merge(trunkP)
    if canP is not None:
        out.merge(canP)
    if log is not None and paint:
        for p_ in (trunkP, canP):
            if p_ is not None and haze is not None:
                p_.rgb = _blend_rgb(p_.rgb, rgb(haze[0]), float(haze[1]))
        if shadow:
            ground_shadow(cv, out, base, light=light, strength=0.45 if shadow is True else float(shadow))
        log(f"{key} trunk")
        m = _commit(cv, trunkP, clip)
        if canP is not None:
            log(f"{key} canopy")
            m = np.maximum(m, _commit(cv, canP, clip))
        return Plant(key, m, base, topp, canopy_c, (x0, y0, x1, y1), dict(scale=scale), out)
    return _finish(cv, key, out, base, topp, canopy_c, paint=paint, clip=clip, shadow=shadow, light=light,
                   haze=haze, extra=dict(scale=scale))


def _willow_strands(P, strands, bx, by, scale, light, ramp, lp, rng, cc, lean):
    """Cascading strands of narrow leaves hanging from the crown (drawn leaf by leaf)."""
    v = _Vec(P.x0, P.y0, P.x1, P.y1, aa=2)
    lx, ly = _l2(light)
    cx = cc[0]
    span = max(abs(s[0]) for s in strands) * scale + 1
    order = sorted(strands, key=lambda s: s[3])
    leaf_l = max(lp * 1.5, 1.2)
    for (ux, uy, L, dep) in order:
        x0, y0 = bx + ux * scale, by + uy * scale
        Lp = L * scale
        n = max(4, int(Lp / (leaf_l * 0.55)))
        sway = (lean * 0.25 + rng.normal(0, 0.05)) * Lp
        for k in range(n):
            t = k / n
            x = x0 + sway * t * t + math.sin(t * 5 + dep * 9) * lp * 0.6
            y = y0 + Lp * t
            side = (x - cx) / span
            tone = 0.5 + 0.38 * (-side * lx) * 1.2 - 0.25 * t + 0.32 * (dep - 0.5) + rng.normal(0, 0.08)
            tone = min(max(tone, 0.05), 0.98)
            ang = math.pi / 2 + rng.normal(0, 0.5) + (0.5 if k % 2 else -0.5)
            ll = leaf_l * rng.uniform(0.8, 1.2) * (1 - 0.4 * t)
            ww = ll * 0.28
            ca, sa = math.cos(ang), math.sin(ang)
            pts = [(x, y), (x + ca * ll * 0.5 - sa * ww, y + sa * ll * 0.5 + ca * ww), (x + ca * ll, y + sa * ll),
                   (x + ca * ll * 0.5 + sa * ww, y + sa * ll * 0.5 - ca * ww)]
            v.poly(pts, _ramp(ramp, np.float32(tone)))
    c, a = v.result()
    P.over(c.astype(F32), a)


def _sphere_dot(P, fx, fy, r, color, light, gloss=0.6, gate=None):
    """A small shaded sphere (fruit, berry, bead) composited onto patch P."""
    X0, X1 = int(fx - r - 2), int(fx + r + 3)
    Y0, Y1 = int(fy - r - 2), int(fy + r + 3)
    if X0 < P.x0 or Y0 < P.y0 or X1 > P.x1 or Y1 > P.y1:
        return
    sy, sx = np.mgrid[Y0:Y1, X0:X1].astype(F32)
    dx, dy = (sx - fx) / r, (sy - fy) / r
    q = dx * dx + dy * dy
    a = clamp01((1 - np.sqrt(q)) * r + 0.5)
    if gate is not None:
        a = a * gate[Y0 - P.y0:Y1 - P.y0, X0 - P.x0:X1 - P.x0]
    L3 = _l3(light)
    nz = np.sqrt(np.clip(1 - q, 0, 1))
    d = np.clip(dx * L3[0] + dy * L3[1] + nz * L3[2], 0, 1)
    spec = np.clip(2 * d * nz - L3[2], 0, 1) ** 24 * gloss
    col = rgb(color)[None, None] * (0.22 + 0.9 * d[..., None]) + spec[..., None]
    P.over(col.astype(F32), a.astype(F32), X0, Y0)


def _add_fruit(P, its, color, rng, light, scale):
    """Round fruit on the front of the clusters."""
    fr = scale * 0.011
    gate = (P.a > 0.9).astype(F32)
    n = max(4, int(len(its) * 0.2))
    for i in rng.choice(len(its), size=min(n, len(its)), replace=False):
        cx, cy, rx, ry = its[i, :4]
        for _ in range(1):
            ang = rng.uniform(0, 2 * math.pi)
            rr = rng.uniform(0.2, 0.7)
            _sphere_dot(P, cx + math.cos(ang) * rx * rr, cy + math.sin(ang) * ry * rr, fr * rng.uniform(0.8, 1.15),
                        color, light, gate=gate)


def _bonsai_pot(P, bx, by, scale, light):
    """A shallow glazed bonsai tray under the tree base."""
    w = 0.62 * scale
    h = 0.1 * scale
    v = _Vec(P.x0, P.y0, P.x1, P.y1)
    for f in (-0.32, 0.32):
        v.poly([(bx + w * f - w * 0.06, by + h), (bx + w * f + w * 0.06, by + h),
                (bx + w * f + w * 0.05, by + h * 1.3), (bx + w * f - w * 0.05, by + h * 1.3)], "#22343e")
    v.poly([(bx - w * 0.5, by - h * 0.1), (bx + w * 0.5, by - h * 0.1), (bx + w * 0.45, by + h * 1.05),
            (bx - w * 0.45, by + h * 1.05)], "#3c5a6e")
    v.poly([(bx - w * 0.53, by - h * 0.3), (bx + w * 0.53, by - h * 0.3), (bx + w * 0.51, by),
            (bx - w * 0.51, by)], "#6a8ea4")
    c, a = v.result()
    sh = _shade_from_alpha(a, max(h * 0.5, 1), light, 0.35)
    P.over((c * sh[..., None]).astype(F32), a)


# ============================================================================
# palms, bamboo, cacti, banana — structured plants that are not branch + canopy
# ============================================================================
def _half_blades(spine, widths):
    """Left and right half polygons of a blade along a spine (half-widths per point)."""
    P = np.asarray(spine, np.float64)
    wv = np.asarray(widths, np.float64)
    d = np.gradient(P, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    Lp = P + n * wv[:, None]
    Rp = P - n * wv[:, None]
    left = [tuple(p) for p in P] + [tuple(p) for p in Lp[::-1]]
    right = [tuple(p) for p in P] + [tuple(p) for p in Rp[::-1]]
    return left, right, n


def _tone_col(ramp, t):
    return _ramp(ramp, np.float32(min(max(t, 0.0), 1.0)))


def _box_of(points, pad):
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    return pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad


def _frond(v, C, ang, L, droop, ramp, tone, rng, S, leaflet=0.32, density=1.0, width=None, light=LIGHT):
    """A pinnate frond (palm, fern): curved rachis with leaflets on both sides that hang under gravity."""
    dx, dy = math.cos(ang), math.sin(ang)
    end = (C[0] + dx * L, C[1] + dy * L + droop * L)
    ctrl = (C[0] + dx * L * 0.55, C[1] + dy * L * 0.55 - abs(dx) * L * 0.18 * (1 - droop))
    spine = _bez(C, ctrl, end, 40)
    n = int(max(8, 34 * density * min(L / (60 * S), 1.6)))
    lx, ly = _l2(light)
    lw = width if width is not None else max(L * 0.012, 0.6 * S)
    for k in range(n, 0, -1):
        t = 0.06 + 0.94 * k / n
        i = min(int(t * (len(spine) - 1)), len(spine) - 2)
        p = spine[i]
        tg = spine[i + 1] - spine[i]
        tg = tg / (np.linalg.norm(tg) + 1e-9)
        ll = L * leaflet * math.sin(math.pi * (0.12 + 0.82 * t)) ** 0.7 * (1.05 - 0.45 * t) * rng.uniform(0.85, 1.1)
        for side in (-1, 1):
            a = math.atan2(tg[1], tg[0]) + side * math.radians(rng.uniform(48, 68))
            ex, ey = math.cos(a), math.sin(a) + 0.55          # gravity bends the leaflet down
            nrm = math.hypot(ex, ey)
            ex, ey = ex / nrm, ey / nrm
            q = (p[0] + ex * ll, p[1] + ey * ll)
            m = (p[0] + ex * ll * 0.5, p[1] + ey * ll * 0.5 + ll * 0.06)
            pts = _bez(p, m, q, 6)
            wd = [lw * 1.4 * (1 - j / 5) ** 0.7 + 0.2 for j in range(6)]
            poly = _ribbon(pts, wd)
            lit = 0.5 + 0.5 * (-(ex * lx + ey * ly))
            v.poly(poly, _tone_col(ramp, tone + 0.18 * (lit - 0.5) + (0.07 if side < 0 else -0.04) + rng.normal(0, 0.04)))
    v.line([tuple(p) for p in spine[3:]], max(lw * 1.1, 0.8), _tone_col(ramp, tone * 0.8 + 0.15))
    return spine


def palm(cv, at=(0.5, 0.95), height=0.7, seed=0, *, fronds=13, curve=0.22, lean=0.0, coconuts=False,
         palette="palm", bark=None, light=LIGHT, shadow=False, clip=None, haze=None, paint=True):
    """Palm tree: curved ringed trunk and a crown of drooping pinnate fronds (``coconuts=True`` adds nuts)."""
    rng = _rng(seed, 21)
    S = _S(cv)
    Hpx = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    side = 1 if (lean > 0 or (lean == 0 and rng.random() < 0.5)) else -1
    bend = (curve + abs(lean) * 0.3) * side
    top = (bx + bend * Hpx * 0.55, by - Hpx * 0.78)
    ctrl = (bx + bend * Hpx * 0.05, by - Hpx * 0.45)
    pts = _bez((bx, by), ctrl, top, 14)
    r0, r1 = 0.026 * Hpx, 0.017 * Hpx
    segs = []
    acc = 0.0
    for i in range(len(pts) - 1):
        t0, t1 = i / (len(pts) - 1), (i + 1) / (len(pts) - 1)
        L = float(np.hypot(*(pts[i + 1] - pts[i])))
        fl0 = 1 + 0.5 * math.exp(-t0 * 18)
        fl1 = 1 + 0.5 * math.exp(-t1 * 18)
        segs.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], (r0 + (r1 - r0) * t0) * fl0,
                     (r0 + (r1 - r0) * t1) * fl1, acc, acc + L, 0))
        acc += L
    L = Hpx * 0.42
    x0, y0, x1, y1 = _clip_box(cv, min(bx, top[0]) - L * 1.2, top[1] - L * 0.75, max(bx, top[0]) + L * 1.2,
                               by + 4 * S)
    base = (bx, by)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "palm", None, base, top, top)
    P = _Patch(x0, y0, x1, y1)
    bcol, ba, _ = _render_bark(P, segs, light, bark or BARKS["palm"], kind="rings",
                               S=S * max(min(Hpx / (700 * S), 1.6), 0.35), seed=seed)
    Yg = np.arange(P.y0, P.y1, dtype=F32)[:, None]
    ba = ba * clamp01(by + 2 - Yg)
    P.rgb, P.a = bcol, ba
    ramp = _palette(palette) if isinstance(palette, str) else palette
    C = (top[0], top[1] + 0.01 * Hpx)
    angs = np.linspace(0, 2 * math.pi, fronds, endpoint=False) + rng.uniform(0, 1)
    order = sorted(angs, key=lambda a: math.sin(a))           # fronds pointing up / away first
    lx, ly = _l2(light)
    back = _Vec(x0, y0, x1, y1, aa=2)
    front = _Vec(x0, y0, x1, y1, aa=2)
    for a in order:
        dx, dy = math.cos(a), math.sin(a) * 0.55 - 0.15
        ang = math.atan2(dy, dx)
        Lf = L * rng.uniform(0.8, 1.1) * (0.75 + 0.25 * abs(math.cos(a)))
        droop = 0.35 + 0.35 * max(math.sin(a), 0) + rng.uniform(-0.05, 0.1)
        lit = 0.5 - 0.5 * (dx * lx + dy * ly)
        tone = 0.28 + 0.4 * lit + (0.15 if math.sin(a) > 0 else -0.08)
        _frond(back if math.sin(a) < 0.1 else front, C, ang, Lf, droop, ramp, tone, rng, S, light=light)
    bp = back.patch()
    P.merge(bp)
    if coconuts:
        for k in range(5):
            a = rng.uniform(0, 2 * math.pi)
            _sphere_dot(P, C[0] + math.cos(a) * 0.022 * Hpx, C[1] + 0.02 * Hpx + abs(math.sin(a)) * 0.015 * Hpx,
                        0.017 * Hpx, "#6b4a22", light, gloss=0.2)
    P.merge(front.patch())
    return _finish(cv, "palm", P, base, (top[0], top[1] - L * 0.3), C, paint=paint, clip=clip, shadow=shadow,
                   light=light, haze=haze)


def bamboo(cv, at=(0.5, 0.95), height=0.8, seed=0, *, width=0.25, stalks=9, light=LIGHT, shadow=False,
           clip=None, haze=None, paint=True, palette=("#1f3a12", "#3e6a1c", "#6f9a2e", "#a8c550", "#dbe88a")):
    """Bamboo grove: segmented stalks with node rings and sprays of lance leaves; back stalks darker."""
    rng = _rng(seed, 31)
    S = _S(cv)
    Hpx = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    Wpx = width * cv.h
    x0, y0, x1, y1 = _clip_box(cv, bx - Wpx * 0.5 - Hpx * 0.25, by - Hpx * 1.05, bx + Wpx * 0.5 + Hpx * 0.25,
                               by + 3)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "bamboo", None, (bx, by), (bx, by - Hpx), (bx, by - Hpx * 0.6))
    P = _Patch(x0, y0, x1, y1)
    ramp = list(palette)
    order = sorted(range(stalks), key=lambda k: rng.random())
    for rank, k in enumerate(order):
        depth = rank / max(stalks - 1, 1)                     # 0 back → 1 front
        sx = bx + (rng.random() - 0.5) * Wpx
        h = Hpx * rng.uniform(0.75, 1.0) * (0.85 + 0.15 * depth)
        r = Hpx * 0.011 * (0.7 + 0.5 * depth)
        lean_ = rng.normal(0, 0.06)
        tip = (sx + lean_ * h, by - h)
        pts = _bez((sx, by + 2), (sx + lean_ * h * 0.3, by - h * 0.5), tip, 10)
        segs = []
        acc = 0
        for i in range(len(pts) - 1):
            Ls = float(np.hypot(*(pts[i + 1] - pts[i])))
            segs.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], r, r * 0.92 if i < 8 else r * 0.6,
                         acc, acc + Ls, 0))
            acc += Ls
        q = _Patch(*_clip_box(cv, min(sx, tip[0]) - r * 3, tip[1] - r, max(sx, tip[0]) + r * 3, by + 3))
        dark = 0.55 + 0.45 * depth
        cols = (_hex_mix(ramp[0], "#000000", 0.2), ramp[2], ramp[4])
        c, a, info = _render_bark(q, segs, light, cols, kind="smooth", S=S, seed=seed + k)
        node = Hpx * rng.uniform(0.07, 0.09)
        ph = (info["AL"] / node) % 1.0
        ring = 1 - 0.55 * np.exp(-((ph - 0.5) * node / (1.6 * S + r * 0.08)) ** 2 * 0.5)
        bump = np.exp(-((ph - 0.53) * node / (2.5 * S + r * 0.1)) ** 2 * 0.5) * 0.25
        c = c * (ring + bump)[..., None] * dark
        c = _blend_rgb(c, rgb("#c6c08a"), (clamp01(0.4 - ph) * 0.4 * (info["AL"] < node * 1.2))[..., None])
        q.rgb, q.a = c.astype(F32), a
        P.merge(q)
        # leaf sprays from the upper nodes
        v = _Vec(P.x0, P.y0, P.x1, P.y1, aa=2)
        nn = int(h / node)
        for j in range(max(nn // 3, 1), nn):
            if rng.random() > 0.75:
                continue
            t = j * node / h
            i = min(int(t * (len(pts) - 1)), len(pts) - 1)
            p = pts[i]
            sd = rng.choice([-1, 1])
            for m in range(int(rng.integers(3, 6))):
                ang = math.radians(rng.uniform(10, 60)) * 1 + (0 if sd > 0 else math.pi - math.radians(rng.uniform(20, 120)))
                ang = (math.radians(rng.uniform(5, 55)) if sd > 0 else math.pi - math.radians(rng.uniform(5, 55)))
                ll = Hpx * rng.uniform(0.06, 0.1) * (0.8 + 0.4 * depth)
                ex, ey = math.cos(ang), math.sin(ang) + 0.25
                base_p = (p[0] + sd * r * 2, p[1])
                spine = _bez(base_p, (base_p[0] + ex * ll * 0.5, base_p[1] + ey * ll * 0.4),
                             (base_p[0] + ex * ll, base_p[1] + ey * ll + ll * 0.25), 10)
                wv = [ll * 0.085 * math.sin(math.pi * min(0.97, (u + 0.08) / 1.08)) ** 0.8 for u in np.linspace(0, 1, 10)]
                left, right, _ = _half_blades(spine, wv)
                tone = (0.3 + 0.45 * depth + rng.normal(0, 0.06))
                v.poly(left, _tone_col(ramp, tone + 0.08))
                v.poly(right, _tone_col(ramp, tone - 0.06))
        P.merge(v.patch())
    return _finish(cv, "bamboo", P, (bx, by), (bx, by - Hpx), (bx, by - Hpx * 0.6), paint=paint, clip=clip,
                   shadow=shadow, light=light, haze=haze)


def _ribbed(P, segs, light, cols, ribs=9, S=1.0, spines=True, seed=0):
    """Cactus columns: cylinders with vertical ribs and spine dots along the crests."""
    c, a, info = _render_bark(P, segs, light, cols, kind="smooth", S=S, seed=seed)
    sn = info["SN"]
    ang = np.arcsin(np.clip(sn, -1, 1))
    rib = np.cos(ang * ribs)
    shade = 0.78 + 0.28 * rib
    c = c * shade[..., None]
    if spines:
        crest = smoothstep(0.75, 0.97, rib) * (np.abs(sn) < 0.92)
        dots = ((info["AL"] / (7.0 * S)) % 1.0)
        dot = crest * smoothstep(0.75, 1.0, 1 - np.abs(dots - 0.5) * 2) ** 2
        c = _blend_rgb(c, rgb("#efe6c2"), (dot * 0.9)[..., None])
    return c.astype(F32), a


def saguaro(cv, at=(0.5, 0.95), height=0.6, seed=0, *, arms=None, light=LIGHT, shadow=False, clip=None,
            haze=None, paint=True, colors=("#1d3a22", "#3f6c3a", "#86ad6a")):
    """Saguaro cactus: ribbed column with upturned J-shaped arms and spine rows."""
    rng = _rng(seed, 41)
    S = _S(cv)
    Hpx = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    r = Hpx * 0.075
    segs = [(bx, by + r, bx, by - Hpx + r, r, r * 0.95, 0, Hpx, 0)]
    na = arms if arms is not None else int(rng.integers(2, 4))
    for k in range(na):
        sd = -1 if k % 2 == 0 else 1
        hy = by - Hpx * rng.uniform(0.32, 0.6)
        ra = r * rng.uniform(0.6, 0.72)
        out = r + ra * rng.uniform(1.0, 1.8)
        up = Hpx * rng.uniform(0.18, 0.32)
        path = np.concatenate([_bez((bx, hy), (bx + sd * out, hy + ra * 0.6), (bx + sd * out, hy - ra * 1.2), 8),
                               _bez((bx + sd * out, hy - ra * 1.2), (bx + sd * out, hy - up * 0.6),
                                    (bx + sd * out * 1.04, hy - up), 6)[1:]])
        acc = 0.0
        for i in range(len(path) - 1):
            Ls = float(np.hypot(*(path[i + 1] - path[i])))
            segs.append((path[i][0], path[i][1], path[i + 1][0], path[i + 1][1], ra, ra * (0.99 if i < len(path) - 2 else 0.95),
                         acc, acc + Ls, 1))
            acc += Ls
    xs = [s[0] for s in segs] + [s[2] for s in segs]
    x0, y0, x1, y1 = _clip_box(cv, min(xs) - r * 1.5, by - Hpx - 2, max(xs) + r * 1.5, by + 2)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "saguaro", None, (bx, by), (bx, by - Hpx), (bx, by - Hpx / 2))
    P = _Patch(x0, y0, x1, y1)
    c, a = _ribbed(P, segs, light, colors, ribs=7, S=S * max(Hpx / (600 * S), 0.4), seed=seed)
    Yg = np.arange(P.y0, P.y1, dtype=F32)[:, None]
    P.rgb, P.a = c, a * clamp01(by + 1 - Yg)
    return _finish(cv, "saguaro", P, (bx, by), (bx, by - Hpx), (bx, by - Hpx / 2), paint=paint, clip=clip,
                   shadow=shadow, light=light, haze=haze)


def _pad(P, cx, cy, rx, ry, ang, light, ramp, S, areoles=True, seed=0):
    """Flattened ellipsoid pad (prickly pear, succulent leaf) with areole dots."""
    rm = max(rx, ry) + 2
    X0, Y0, X1, Y1 = int(cx - rm), int(cy - rm), int(cx + rm) + 2, int(cy + rm) + 2
    X0, Y0 = max(X0, P.x0), max(Y0, P.y0)
    X1, Y1 = min(X1, P.x1), min(Y1, P.y1)
    if X0 >= X1 or Y0 >= Y1:
        return
    sy, sx = np.mgrid[Y0:Y1, X0:X1].astype(F32)
    u, v = _rot(sx - cx, sy - cy, ang)
    q = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
    a = clamp01((1 - q) * min(rx, ry) + 0.5)
    nu, nv = u / rx, v / ry
    nz = np.sqrt(np.clip(1 - nu * nu - nv * nv, 0, 1)) * 0.8
    c_, s_ = math.cos(ang), math.sin(ang)
    nx, ny = nu * c_ - nv * s_, nu * s_ + nv * c_
    L3 = _l3(light)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    d = np.clip((nx * L3[0] + ny * L3[1] + nz * L3[2]) / nn, 0, 1)
    col = _ramp(ramp, 0.12 + 0.8 * d)
    col = col * (0.7 + 0.3 * smoothstep(1.0, 0.82, q))[..., None]
    if areoles:
        g = 0.22 * min(rx, ry)
        gu = u / g
        gv = v / g + 0.5 * (np.floor(gu) % 2)
        du = gu - np.round(gu)
        dv = gv - np.round(gv)
        dot = clamp01((0.16 - np.hypot(du, dv)) * g * 0.8 + 0.5) * (q < 0.88)
        col = _blend_rgb(col, rgb("#5a4a2a"), dot * 0.8)
        sp_ = clamp01((0.07 - np.hypot(du + 0.12, dv - 0.12)) * g + 0.5) * (q < 0.88)
        col = _blend_rgb(col, rgb("#f3ecc8"), sp_)
    P.over(col.astype(F32), a.astype(F32), X0, Y0)


def prickly_pear(cv, at=(0.5, 0.95), height=0.4, seed=0, *, fruit=True, light=LIGHT, shadow=False, clip=None,
                 haze=None, paint=True, palette=("#1e3a1e", "#3b6630", "#6a9a48", "#a6c66e", "#d8e6a0")):
    """Prickly pear (Opuntia): oval pads growing from pad edges, areole dots, magenta fruit on top."""
    rng = _rng(seed, 43)
    S = _S(cv)
    Hpx = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    pads = []

    def grow(x, y, ang, size, level):
        rx, ry = size * 0.36, size * 0.5
        cx, cy = x + math.cos(ang) * ry, y + math.sin(ang) * ry
        pads.append((cx, cy, rx, ry, ang + math.pi / 2, level))
        if level >= 3:
            return
        for _ in range(int(rng.integers(1, 3)) if level else 2):
            a2 = ang + rng.uniform(-0.9, 0.9)
            tx, ty = cx + math.cos(a2) * ry * 0.85, cy + math.sin(a2) * ry * 0.85
            grow(tx, ty, a2, size * rng.uniform(0.75, 0.92), level + 1)

    grow(bx, by + Hpx * 0.02, -math.pi / 2 + rng.normal(0, 0.2), Hpx * 0.42, 0)
    xs = [p[0] for p in pads]
    ys = [p[1] for p in pads]
    pad = Hpx * 0.3
    x0, y0, x1, y1 = _clip_box(cv, min(xs) - pad, min(ys) - pad, max(xs) + pad, by + 3)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "prickly pear", None, (bx, by), (bx, by - Hpx), (bx, by - Hpx / 2))
    P = _Patch(x0, y0, x1, y1)
    for (cx, cy, rx, ry, ang, lvl) in sorted(pads, key=lambda p: p[5]):
        _pad(P, cx, cy, rx, ry, ang, light, list(palette), S)
        if fruit and lvl >= 2 and rng.random() < 0.7:
            for k in range(int(rng.integers(1, 4))):
                t = ang - math.pi / 2 + rng.uniform(-0.6, 0.6)
                fx, fy = cx + math.cos(t) * ry * 0.95, cy + math.sin(t) * ry * 0.95
                _pad(P, fx, fy, rx * 0.22, rx * 0.3, t + math.pi / 2, light,
                     ["#4a0f2a", "#8a1c4a", "#c4336a", "#e86a92", "#f6b0c6"], S, areoles=False)
    Yg = np.arange(P.y0, P.y1, dtype=F32)[:, None]
    P.a = P.a * clamp01(by + 1 - Yg)
    top_y = min(p[1] - p[3] for p in pads)
    return _finish(cv, "prickly pear", P, (bx, by), (bx, top_y), (bx, (by + top_y) / 2), paint=paint, clip=clip,
                   shadow=shadow, light=light, haze=haze)


def banana(cv, at=(0.5, 0.95), height=0.6, seed=0, *, leaves=7, fruit=True, light=LIGHT, shadow=False, clip=None,
           haze=None, paint=True, palette=("#1d3a14", "#2f5e1e", "#4f8a2c", "#86b84a", "#c4e07c")):
    """Banana plant: pseudostem and huge arching paddle leaves with a pale midrib and wind tears."""
    rng = _rng(seed, 47)
    S = _S(cv)
    Hpx = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    stem_h = Hpx * 0.5
    r = Hpx * 0.032
    segs = [(bx, by + 2, bx + rng.normal(0, 0.01) * Hpx, by - stem_h, r * 1.15, r * 0.8, 0, stem_h, 0)]
    C = (segs[0][2], by - stem_h)
    x0, y0, x1, y1 = _clip_box(cv, bx - Hpx * 0.75, by - Hpx * 1.05, bx + Hpx * 0.75, by + 3)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "banana", None, (bx, by), (bx, by - Hpx), C)
    P = _Patch(x0, y0, x1, y1)
    c, a, info = _render_bark(P, segs, light, ("#3a4a1c", "#6f8a38", "#a8b868"), kind="smooth", S=S, seed=seed)
    stripes = fbm2d_at(info["SN"] * 6, info["AL"] / (30 * S), 2, seed=seed)
    c = _blend_rgb(c, rgb("#5a3a1e"), clamp01(stripes * 1.5)[..., None] * 0.5)
    P.rgb, P.a = c.astype(F32), a
    ramp = list(palette)
    lx, ly = _l2(light)
    angs = sorted(rng.uniform(-math.pi, 0, leaves), key=lambda t: -abs(math.cos(t)))
    vb = _Vec(x0, y0, x1, y1, aa=2)
    for k, ang in enumerate(angs):
        L = Hpx * rng.uniform(0.42, 0.55)
        dx, dy = math.cos(ang), math.sin(ang)
        upr = max(-dy, 0)
        p0 = C
        p1 = (C[0] + dx * L * 0.12, C[1] - L * 0.18)
        petiole = _bez(p0, (p0[0] + dx * L * 0.03, p0[1] - L * 0.1), p1, 6)
        p2 = (C[0] + dx * L * 1.0, C[1] - L * 0.05 + L * 0.38 * (1 - upr))
        ctrl = (C[0] + dx * L * 0.6, C[1] - L * (0.35 + 0.35 * upr))
        spine = _bez(p1, ctrl, p2, 30)
        wmax = L * 0.115
        uu = np.linspace(0, 1, len(spine))
        wv = wmax * np.sin(np.pi * np.clip(uu * 0.92 + 0.06, 0, 1)) ** 0.45
        left, right, nrm = _half_blades(spine, wv)
        lit = 0.5 - 0.5 * (dx * lx + dy * ly)
        tone = 0.32 + 0.35 * lit + rng.normal(0, 0.05)
        vb.line([tuple(p) for p in petiole], wmax * 0.12, _tone_col(ramp, tone * 0.7))
        up_side = 0.12 if nrm[len(nrm) // 2][1] < 0 else -0.1
        vb.poly(left, _tone_col(ramp, tone + up_side))
        vb.poly(right, _tone_col(ramp, tone - up_side))
        for j in range(2, len(spine) - 2, 2):                  # parallel veins
            for sd in (1, -1):
                q = spine[j] + nrm[j] * wv[j] * sd * 0.98
                vb.line([tuple(spine[j]), tuple(q)], max(0.6 * S, 0.5), _tone_col(ramp, tone + up_side * sd * 0.5 - 0.08))
        for j in range(3, len(spine) - 3):                    # wind tears
            if rng.random() < 0.28:
                sd = rng.choice([1, -1])
                q = spine[j] + nrm[j] * wv[j] * sd * 1.1
                s0 = spine[j] + nrm[j] * wv[j] * sd * rng.uniform(0.15, 0.6)
                vb.line([tuple(s0), tuple(q)], max(1.0 * S, 0.7), (0, 0, 0), erase=True)
        vb.line([tuple(p) for p in spine], max(wmax * 0.06, 0.8), _tone_col(ramp, 0.85))
    if fruit:
        bunch = (C[0] + Hpx * 0.04, C[1] + Hpx * 0.06)
        for k in range(14):
            row = k // 4
            ang = math.radians(-60 + (k % 4) * 30)
            px = bunch[0] + (k % 4 - 1.5) * Hpx * 0.018
            py = bunch[1] + row * Hpx * 0.03
            sp_ = _bez((px, py), (px + Hpx * 0.02, py - Hpx * 0.01), (px + Hpx * 0.026, py - Hpx * 0.04), 8)
            vb.line([tuple(p) for p in sp_], Hpx * 0.013, _tone_col(["#3a4a10", "#6a8a1e", "#a6c03a", "#d6e070", "#f0f0a0"], 0.45 + 0.1 * (k % 2)))
        vb.ellipse(bunch[0] + Hpx * 0.01, bunch[1] + Hpx * 0.16, Hpx * 0.018, Hpx * 0.035, "#6a1e3a")
    P.merge(vb.patch())
    sh = _shade_from_alpha(P.a, Hpx * 0.01, light, 0.25)
    P.rgb = (P.rgb * sh[..., None]).astype(F32)
    return _finish(cv, "banana", P, (bx, by), (bx, by - Hpx), C, paint=paint, clip=clip, shadow=shadow,
                   light=light, haze=haze)


# ============================================================================
# forest: many trees, depth sorted, smaller and hazier with distance
# ============================================================================
def forest(cv, region=(0.0, 0.55, 1.0, 0.75), n=30, species=("oak", "maple", "birch"), height=(0.12, 0.35),
           seed=0, *, haze=("#b9c9d6", 0.55), light=LIGHT, season=None, style="painted", shadow=False, clip=None,
           avoid=None, paint=True):
    """Scatter ``n`` trees with bases inside ``region`` = (u0, v0, u1, v1). Trees near v0 (far) use height[0] and
    the most haze; trees near v1 (near) use height[1] and no haze. All trees are painted in one step.
    ``species`` may be one name or a list (picked at random); ``avoid`` = mask where no tree is planted."""
    rng = _rng(seed, 61)
    if isinstance(species, str):
        species = [species]
    u0, v0, u1, v1 = region
    pts = []
    for i in range(int(n)):
        u = rng.uniform(u0, u1)
        v = v0 + (v1 - v0) * rng.random() ** 0.8
        if avoid is not None:
            yy, xx = min(int(v * cv.h), cv.h - 1), min(int(u * cv.w), cv.w - 1)
            if avoid[yy, xx] > 0.3:
                continue
        pts.append((v, u, species[int(rng.integers(len(species)))], int(rng.integers(1 << 30))))
    pts.sort()
    full = _Patch(0, 0, cv.w, cv.h)
    trees = []
    for (v, u, spn, sd) in pts:
        t = (v - v0) / max(v1 - v0, 1e-6)
        h = (height[0] + (height[1] - height[0]) * t) * rng.uniform(0.8, 1.15)
        hz = None if haze is None else (haze[0], haze[1] * (1 - t) ** 1.2)
        p = tree(cv, spn, at=(u, v), height=h, seed=sd, season=season, light=light, style=style, haze=hz,
                 paint=False, fullness=rng.uniform(0.85, 1.15))
        if p.patch is not None:
            if hz is not None:
                p.patch.rgb = _blend_rgb(p.patch.rgb, rgb(hz[0]), float(hz[1]))
            if shadow:
                pass
            full.merge(p.patch)
        trees.append(p)
    ys, xs = np.nonzero(full.a > 0.004)
    if len(ys) == 0:
        return Plant("forest", np.zeros((cv.h, cv.w), F32), extra=dict(trees=trees))
    m = _commit(cv, full, clip) if paint else full.a
    return Plant("forest", m, (cv.w * (u0 + u1) / 2, cv.h * v1), (cv.w * (u0 + u1) / 2, float(ys.min())),
                 (cv.w * (u0 + u1) / 2, cv.h * v0), (int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                                                     int(ys.max()) + 1), dict(trees=trees), full)


# ============================================================================
# shrubs and ground plants
# ============================================================================
def _leafy(cv, items, light, palette, leaf_px, seed, style="painted", holes=0.15, leaf_shape=(1.0, 0.6),
           leaf_align="random", pad=None):
    """Patch with a leafy canopy for items given in canvas px."""
    S = _S(cv)
    pad = pad if pad is not None else leaf_px * 2 + 3
    x0, y0, x1, y1 = _clip_box(cv, np.min(items[:, 0] - items[:, 2]) - pad, np.min(items[:, 1] - items[:, 3]) - pad,
                               np.max(items[:, 0] + items[:, 2]) + pad, np.max(items[:, 1] + items[:, 3]) + pad)
    if x0 >= x1 or y0 >= y1:
        return None, None
    P = _Patch(x0, y0, x1, y1)
    c, a, info = _render_canopy(P, items, light, _as_ramps(palette), leaf_px, seed=seed, style=style, holes=holes,
                                leaf_shape=leaf_shape, leaf_align=leaf_align)
    P.rgb, P.a = c, a
    return P, info


def bush(cv, at=(0.5, 0.9), width=0.25, height=None, seed=0, *, palette="bush", flowers=None, berries=None,
         light=LIGHT, style="painted", shadow=False, clip=None, haze=None, paint=True, fullness=1.0):
    """Rounded shrub: a dome of leaf clusters. ``flowers`` = colour → flowering shrub (azalea, hydrangea);
    ``berries`` = colour. width / height are fractions of the canvas height."""
    rng = _rng(seed, 71)
    S = _S(cv)
    Wp = width * cv.h
    Hp = (height if height is not None else width * 0.62) * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    n = int(26 * fullness + 8)
    its = []
    for i in range(n):
        a = rng.uniform(math.pi * 1.02, math.pi * 1.98)
        rr = rng.random() ** 0.5
        cx = bx + math.cos(a) * rr * Wp * 0.42
        cy = by - Hp * 0.35 + math.sin(a) * rr * Hp * 0.55 + Hp * 0.12
        r = Wp * rng.uniform(0.12, 0.2)
        its.append([cx, min(cy, by - r * 0.55), r, r * 0.85, rng.uniform(-0.4, 0.4), 0, rng.random()])
    items = _bulge(np.array(its), rng, 0.3)
    lp = float(np.clip(Hp * 0.035, 0.9, 7 * S))
    P, info = _leafy(cv, items, light, palette, lp, seed, style=style, holes=0.12)
    if P is None:
        return _finish(cv, "bush", None, (bx, by), (bx, by - Hp), (bx, by - Hp / 2))
    if flowers is not None or berries is not None:
        gate = (P.a > 0.8).astype(F32)
        L3 = _l3(light)
        for k in range(int(n * 3)):
            i = int(rng.integers(len(items)))
            cx, cy, rx, ry = items[i, :4]
            a = rng.uniform(0, 2 * math.pi)
            rr = rng.uniform(0.0, 0.8)
            fx, fy = cx + math.cos(a) * rx * rr, cy + math.sin(a) * ry * rr
            if berries is not None:
                _sphere_dot(P, fx, fy, max(lp * 0.55, 1.0), berries, light, gate=gate)
            else:
                _blossom(P, fx, fy, max(lp * 1.1, 1.3), flowers, light, rng)
    Yg = np.arange(P.y0, P.y1, dtype=F32)[:, None]
    P.a = P.a * clamp01(by + 1 - Yg)
    return _finish(cv, "bush", P, (bx, by), (bx, by - Hp), (bx, by - Hp * 0.5), paint=paint, clip=clip,
                   shadow=shadow, light=light, haze=haze, shadow_len=0.9)


def _blossom(P, fx, fy, r, color, light, rng, petals=5):
    """Tiny 5-petal blossom (for shrubs, branches, meadows)."""
    X0, Y0, X1, Y1 = int(fx - r - 2), int(fy - r - 2), int(fx + r + 3), int(fy + r + 3)
    if X0 < P.x0 or Y0 < P.y0 or X1 > P.x1 or Y1 > P.y1:
        return
    sy, sx = np.mgrid[Y0:Y1, X0:X1].astype(F32)
    dx, dy = sx - fx, sy - fy
    rr = np.hypot(dx, dy) / r
    th = np.arctan2(dy, dx) + rng.uniform(0, 6.3)
    edge = 0.62 + 0.38 * np.abs(np.cos(th * petals / 2))
    a = clamp01((edge - rr) * r + 0.5)
    c = rgb(color)
    L = _l2(light)
    lit = 0.85 + 0.25 * clamp01(-(dx * L[0] + dy * L[1]) / r)
    col = c[None, None] * (lit * (0.75 + 0.25 * rr))[..., None]
    col = _blend_rgb(col, rgb("#f6d24a"), clamp01((0.25 - rr) * r + 0.5) * 0.9)
    P.over(col.astype(F32), a.astype(F32), X0, Y0)


def hedge(cv, box=(0.2, 0.62, 0.8, 0.8), top=0.05, seed=0, *, palette="hedge", light=LIGHT, side=0.0,
          shadow=False, clip=None, haze=None, paint=True, leaf=5.0):
    """Clipped box hedge. ``box`` = front face (u0, v0, u1, v1); ``top`` = visible depth of the flat top
    (fraction of canvas height); ``side`` = visible side face width (+ right, − left). Leaves follow the faces."""
    rng = _rng(seed, 73)
    S = _S(cv)
    u0, v0, u1, v1 = box
    X0, Y0, X1, Y1 = u0 * cv.w, v0 * cv.h, u1 * cv.w, v1 * cv.h
    T = top * cv.h
    sd = side * cv.h
    lp = max(min(leaf * S, (Y1 - Y0) * 0.045), 0.9)
    x0, y0, x1, y1 = _clip_box(cv, min(X0, X0 + sd) - lp * 2, Y0 - T - lp * 2, max(X1, X1 + sd) + lp * 2,
                               Y1 + lp)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "hedge", None, ((X0 + X1) / 2, Y1), ((X0 + X1) / 2, Y0 - T), ((X0 + X1) / 2, Y0))
    P = _Patch(x0, y0, x1, y1)
    X, Y = P.grid()
    F1, F2, FX, FY, CID = _worley(X, Y, lp, seed)
    sh = T * 0.6                                  # top face shifts back (up) and towards the side

    def region(x, y):
        """0 = outside, 1 = front, 2 = top, 3 = side (evaluated at leaf points)."""
        r = np.zeros(x.shape, np.int8)
        wob = (fbm2d_at(x / (40 * S), y / (40 * S), 2, seed=seed) * lp * 0.9)
        front = (x > X0 + wob) & (x < X1 - wob) & (y > Y0 + wob * 0.5) & (y < Y1)
        tt = clamp01((Y0 - y) / max(T, 1e-3))
        topm = (y <= Y0 + wob * 0.5) & (y > Y0 - T + np.abs(wob)) & (x > X0 + sd * tt * (sd < 0) + wob) & \
               (x < X1 + sd * tt * (sd > 0) - wob)
        r[topm] = 2
        r[front] = 1
        if sd != 0:
            xs = (x - (X1 if sd > 0 else X0)) / sd
            sidem = (xs > 0) & (xs < 1) & (y > Y0 - T * xs + wob) & (y < Y1 - T * 0.4 * xs)
            r[sidem & (r == 0)] = 3
        return r
    rl = region(FX, FY)
    rp = region(X, Y)
    du, dv = _rot(X - FX, Y - FY, CID * 6.28)
    cov = clamp01((1 - np.hypot(du / (lp * 0.85), dv / (lp * 0.5))) * lp * 0.5 + 0.5) * (rl > 0)
    # inner pixels: solid where both the pixel and its neighbours are inside
    inner = (rp > 0).astype(F32)
    deep = clamp01(gaussian_blur(inner, lp * 0.6) * 1.6 - 0.75) * inner
    alpha = np.maximum(cov, deep)
    L3 = _l3(light, 0.6)
    normals = {1: (0.0, 0.1, 1.0), 2: (0.0, -1.0, 0.45), 3: (math.copysign(1, sd or 1), 0.05, 0.55)}
    reg = np.where(rl > 0, rl, rp)
    tone = np.zeros(X.shape, F32)
    for k, (nx, ny, nz) in normals.items():
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        d = max((nx * L3[0] + ny * L3[1] + nz * L3[2]) / nn, 0)
        tone = np.where(reg == k, d, tone)
    jit = (CID - 0.5) * 0.35
    ao = 0.55 + 0.45 * clamp01((Y1 - Y) / (max(Y1 - Y0, 1) * 0.5))
    tone = np.clip((tone * 0.85 + 0.12 + jit) * ao, 0, 1)
    ramp = _as_ramps(palette)[0]
    col = _ramp(ramp, tone)
    under = _ramp(ramp, tone * 0.45)
    col = _blend_rgb(under, col, clamp01(cov * 1.2))
    P.rgb, P.a = col.astype(F32), alpha.astype(F32)
    return _finish(cv, "hedge", P, ((X0 + X1) / 2, Y1), ((X0 + X1) / 2, Y0 - T), ((X0 + X1) / 2, (Y0 + Y1) / 2),
                   paint=paint, clip=clip, shadow=shadow, light=light, haze=haze, shadow_len=0.4)


def topiary(cv, at=(0.5, 0.9), height=0.4, seed=0, *, shape="ball", pot=True, palette="hedge", light=LIGHT,
            shadow=True, clip=None, haze=None, paint=True, pot_color="#b5653a"):
    """Clipped topiary in a pot: shape = "ball", "cone", "spiral", "tiers" (stacked balls) or "poodle"."""
    rng = _rng(seed, 75)
    S = _S(cv)
    Hp = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    pot_h = Hp * 0.2 if pot else 0
    gy = by - pot_h
    _choice("topiary", "shape", shape, ("ball", "cone", "spiral", "tiers", "poodle"))
    its = []
    stem_top = gy
    if shape == "ball":
        r = Hp * 0.3
        its.append([bx, gy - Hp * 0.8 + r, r, r, 0, 0, 0.5])
        stem_top = gy - Hp * 0.8 + r
    elif shape == "cone":
        for k in range(14):
            t = k / 13
            r = Hp * 0.25 * (1 - t) + Hp * 0.02
            its.append([bx, gy - Hp * 0.12 - t * Hp * 0.62, r, r * 0.6, 0, -t * Hp * 0.1, 0.5])
        stem_top = gy - Hp * 0.12
    elif shape == "spiral":
        for k in range(60):
            t = k / 59
            ang = t * 2 * math.pi * 3.2
            r = Hp * 0.2 * (1 - t) + Hp * 0.035
            its.append([bx + math.cos(ang) * r * 0.75, gy - Hp * 0.12 - t * Hp * 0.62, Hp * 0.05 * (1.2 - t * 0.6),
                        Hp * 0.04 * (1.2 - t * 0.6), 0.3, math.sin(ang) * r, 0.5])
        stem_top = gy - Hp * 0.1
    else:                                              # tiers / poodle
        ys = [0.28, 0.55, 0.75] if shape == "tiers" else [0.35, 0.72]
        rs = [0.17, 0.13, 0.09] if shape == "tiers" else [0.14, 0.12]
        for yy, rr in zip(ys, rs):
            its.append([bx, gy - Hp * yy, Hp * rr, Hp * rr * 0.92, 0, 0, 0.5])
        stem_top = gy - Hp * ys[-1]
    items = np.array(its)
    lp = float(np.clip(Hp * 0.016, 0.9, 5 * S))
    P, _ = _leafy(cv, items, light, palette, lp, seed, holes=0.0, pad=Hp * 0.3)
    if P is None:
        return _finish(cv, "topiary", None, (bx, by), (bx, by - Hp), (bx, by - Hp / 2))
    x0, y0, x1, y1 = _clip_box(cv, bx - Hp * 0.35, by - Hp * 1.05, bx + Hp * 0.35, by + 3)
    out = _Patch(min(x0, P.x0), min(y0, P.y0), max(x1, P.x1), max(y1, P.y1))
    sr = max(Hp * 0.012, 1.0)
    st = _Patch(out.x0, out.y0, out.x1, out.y1)
    c, a, _ = _render_bark(st, [(bx, gy + 2, bx, stem_top, sr, sr * 0.8, 0, gy - stem_top, 0)], light,
                           BARKS["oak"], kind="smooth", S=S, seed=seed)
    st.rgb, st.a = c, a
    out.merge(st)
    out.merge(P)
    if pot:
        _pot(out, bx, by, Hp * 0.3, pot_h, "classic", pot_color, light, S)
    return _finish(cv, "topiary", out, (bx, by), (bx, by - Hp), (bx, by - Hp * 0.55), paint=paint, clip=clip,
                   shadow=shadow, light=light, haze=haze)


def _blades(v, rng, base_x, base_y, n, height, width, ramp, *, spread=0.35, wind=0.0, curl=0.25, tone=(0.25, 0.85),
            S=1.0, segs=3, stiff=False, thick=1.0):
    """Grass blades as tapered curved polygons, darker at the base, lighter at the tip."""
    for k in range(n):
        h = height * rng.uniform(0.55, 1.0)
        x = base_x + rng.normal(0, 1) * width * 0.35
        lean_ = rng.normal(0, spread) + wind
        bend = rng.normal(0, curl) + wind * 0.8
        tip = (x + lean_ * h, base_y - h * (1 - 0.3 * abs(bend)))
        ctrl = (x + lean_ * h * 0.3, base_y - h * 0.6)
        tip = (tip[0] + bend * h * 0.4, tip[1] + abs(bend) * h * 0.25)
        by_ = base_y + 1 - h * rng.uniform(0.0, 0.14)               # ragged base: blades do not start on one line
        pts = _bez((x, by_), ctrl, tip, 9)
        w0 = max(h * rng.uniform(0.025, 0.045) * thick, 0.6 * S)
        t0 = rng.uniform(*tone)
        for j in range(segs):
            a, b = int(j * 8 / segs), int((j + 1) * 8 / segs) + 1
            seg = pts[a:b]
            ws = [w0 * (1 - (a + i) / 8.6) + 0.15 for i in range(len(seg))]
            v.poly(_ribbon(seg, ws), _tone_col(ramp, t0 * (0.68 + 0.32 * (j + 1) / segs)))


GRASS = ["#1e3214", "#34521e", "#557a2a", "#86a63e", "#c2d474"]


def grass(cv, at=(0.5, 0.95), width=0.08, height=0.07, seed=0, *, blades=40, region=None, tufts=60,
          palette=GRASS, wind=0.0, light=LIGHT, clip=None, haze=None, paint=True, dry=False, shadow=False):
    """Grass: one tuft at ``at`` (width / height in canvas-height fractions) or, with ``region=(u0, v0, u1, v1)``,
    ``tufts`` tufts scattered with depth (smaller and paler at the top of the region). Blades have direction:
    tapered, curved, dark at the base, light at the tip. ``wind`` bends them (+ right). ``dry`` = straw colours."""
    rng = _rng(seed, 81)
    S = _S(cv)
    ramp = ["#3a2e14", "#6a5424", "#a08840", "#cdb466", "#efe0a0"] if dry else list(palette)
    if region is None:
        spots = [(at[0] * cv.w, at[1] * cv.h, 1.0)]
    else:
        u0, v0, u1, v1 = region
        spots = sorted([(rng.uniform(u0, u1) * cv.w, (v0 + (v1 - v0) * rng.random() ** 0.7) * cv.h, 0)
                        for _ in range(int(tufts))], key=lambda s: s[1])
        spots = [(x, y, 0.45 + 0.55 * (y / cv.h - v0) / max(v1 - v0, 1e-6)) for x, y, _ in spots]
    Hp, Wp = height * cv.h, width * cv.h
    xs = [s[0] for s in spots]
    ys = [s[1] for s in spots]
    x0, y0, x1, y1 = _clip_box(cv, min(xs) - Wp - Hp, min(ys) - Hp * 1.1, max(xs) + Wp + Hp, max(ys) + 3)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "grass", None, (xs[0], ys[0]), (xs[0], ys[0] - Hp), (xs[0], ys[0]))
    v = _Vec(x0, y0, x1, y1, aa=2)
    for (x, y, d) in spots:
        _blades(v, rng, x, y, max(int(blades * (0.6 + 0.4 * d)), 3), Hp * d, Wp * d, ramp, wind=wind * 0.6, S=S,
                tone=(0.25 + 0.2 * (1 - d), 0.95))
    P = v.patch()
    return _finish(cv, "grass", P, (float(np.mean(xs)), max(ys)), (float(np.mean(xs)), min(ys) - Hp),
                   (float(np.mean(xs)), float(np.mean(ys))), paint=paint, clip=clip, light=light, haze=haze)


def reeds(cv, at=(0.5, 0.95), width=0.12, height=0.35, seed=0, *, stems=24, cattails=False, light=LIGHT,
          shadow=False, clip=None, haze=None, paint=True, palette=("#24301a", "#3e5426", "#6a7e3a", "#9cab5e", "#d2d69a")):
    """Reeds: tall, nearly vertical blades with a few bent leaves; ``cattails=True`` adds brown seed heads."""
    rng = _rng(seed, 83)
    S = _S(cv)
    Hp, Wp = height * cv.h, width * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    x0, y0, x1, y1 = _clip_box(cv, bx - Wp - Hp * 0.3, by - Hp * 1.1, bx + Wp + Hp * 0.3, by + 3)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "reeds", None, (bx, by), (bx, by - Hp), (bx, by - Hp / 2))
    v = _Vec(x0, y0, x1, y1, aa=2)
    ramp = list(palette)
    _blades(v, rng, bx, by, stems, Hp, Wp, ramp, spread=0.08, curl=0.18, S=S, tone=(0.3, 0.9), thick=0.4)
    heads = []
    if cattails:
        for k in range(max(stems // 3, 3)):
            x = bx + rng.normal(0, 0.3) * Wp
            h = Hp * rng.uniform(0.75, 1.05)
            lean_ = rng.normal(0, 0.05)
            top = (x + lean_ * h, by - h)
            v.line([(x, by), (x + lean_ * h * 0.5, by - h * 0.5), top], max(Hp * 0.006, 0.8), _tone_col(ramp, 0.55))
            heads.append((top, lean_, h))
    P = v.patch()
    for (top, lean_, h) in heads:
        hl = h * 0.16
        cx, cy = top[0] - lean_ * hl * 0.8, top[1] + hl * 0.75
        _pad(P, cx, cy, hl * 0.13, hl * 0.5, math.atan(lean_), light,
             ["#2a170c", "#4a2a14", "#6e4220", "#94643a", "#bf9a6e"], S, areoles=False)
        vv = _Vec(P.x0, P.y0, P.x1, P.y1)
        vv.line([(top[0] - lean_ * hl * 0.1, top[1] + hl * 0.2), (top[0] + lean_ * hl * 0.4, top[1] - hl * 0.45)],
                max(h * 0.004, 0.7), "#8a7a4a")
        P.merge(vv.patch())
    return _finish(cv, "cattails" if cattails else "reeds", P, (bx, by), (bx, by - Hp), (bx, by - Hp * 0.6),
                   paint=paint, clip=clip, shadow=shadow, light=light, haze=haze)


def cattails(cv, at=(0.5, 0.95), width=0.12, height=0.35, seed=0, **kw):
    """Reeds with brown cattail seed heads (the options of ``reeds``)."""
    kw = _reject("cattails", kw, ["at", "width", "height", "seed"] + [
        n for n in _sigopts(reeds) if n not in ("cv", "at", "width", "height", "seed", "cattails")])
    return reeds(cv, at=at, width=width, height=height, seed=seed, cattails=True, **kw)


def fern(cv, at=(0.5, 0.95), height=0.25, seed=0, *, fronds=9, palette=("#14280f", "#24461a", "#3e6e28", "#6f9e3e", "#aacc6c"),
         light=LIGHT, shadow=False, clip=None, haze=None, paint=True):
    """Fern: arching pinnate fronds from a central crown, finely divided leaflets that droop under gravity."""
    rng = _rng(seed, 85)
    S = _S(cv)
    Hp = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    L = Hp * 1.15
    x0, y0, x1, y1 = _clip_box(cv, bx - L * 1.15, by - Hp * 1.2, bx + L * 1.15, by + Hp * 0.1)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "fern", None, (bx, by), (bx, by - Hp), (bx, by - Hp / 2))
    v = _Vec(x0, y0, x1, y1, aa=2)
    ramp = list(palette)
    angs = sorted(np.linspace(-math.pi * 0.92, -math.pi * 0.08, fronds) + rng.normal(0, 0.12, fronds),
                  key=lambda a: -abs(math.cos(a)))
    for a in angs:
        Lf = L * rng.uniform(0.75, 1.0) * (0.75 + 0.25 * abs(math.sin(a)))
        droop = 0.25 + 0.55 * abs(math.cos(a))
        tone = 0.35 + 0.35 * (0.5 - 0.5 * math.cos(a) * _l2(light)[0]) + rng.normal(0, 0.05)
        _frond(v, (bx, by), a, Lf, droop, ramp, tone, rng, S, leaflet=0.16, density=1.3, light=light)
    P = v.patch()
    return _finish(cv, "fern", P, (bx, by), (bx, by - Hp), (bx, by - Hp * 0.5), paint=paint, clip=clip,
                   shadow=shadow, light=light, haze=haze, shadow_len=0.6)


def _ivy_leaf(v, x, y, size, ang, col, vein):
    """Three-to-five lobed ivy leaf polygon with light veins."""
    th = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    r = size * (0.62 + 0.38 * np.abs(np.cos(th * 2.5)) ** 1.5) * (1 - 0.25 * np.sin(th) ** 2)
    r = r * (0.7 + 0.3 * (np.cos(th) > -0.2))
    pts = [(x + math.cos(ang) * size * 0.6 + rr * math.cos(t + ang), y + math.sin(ang) * size * 0.6 + rr * math.sin(t + ang))
           for t, rr in zip(th, r)]
    v.poly(pts, col)
    cx, cy = x + math.cos(ang) * size * 0.6, y + math.sin(ang) * size * 0.6
    for k in (-1.2, -0.6, 0.0, 0.6, 1.2):
        v.line([(x, y), (cx + math.cos(ang + k) * size * 0.75, cy + math.sin(ang + k) * size * 0.75)],
               max(size * 0.06, 0.5), vein)


def ivy(cv, path=None, seed=0, *, along=None, leaf=0.018, density=1.0, hang=0.12,
        palette=("#0f240f", "#1c3c19", "#2f5e28", "#4e843a", "#86b064"), light=LIGHT, clip=None, paint=True):
    """Ivy / vine. ``path`` = [(u, v), ...] the vine follows; or ``along=mask`` grows along the mask's top edge
    with tendrils hanging down its face (``hang`` = max tendril length, canvas-height fraction). ``leaf`` = leaf
    size (canvas-height fraction)."""
    rng = _rng(seed, 87)
    S = _S(cv)
    lpx = leaf * cv.h
    paths = []
    if along is not None:
        m = np.asarray(along) > 0.5
        cols = np.nonzero(m.any(0))[0]
        if len(cols):
            topy = np.argmax(m, 0).astype(F32)
            xs = np.arange(cols.min(), cols.max() + 1, max(int(lpx * 0.6), 1))
            main = [(float(x), float(topy[x]) + lpx * 0.2) for x in xs if m[:, x].any()]
            paths.append(main)
            for (x, y) in main[::max(int(4 / density), 1)]:
                if rng.random() < 0.55:
                    L = hang * cv.h * rng.uniform(0.3, 1.0)
                    n = max(int(L / (lpx * 0.6)), 2)
                    tp = [(x + math.sin(k * 0.5 + rng.uniform(0, 6)) * lpx * 0.3, y + L * k / n) for k in range(n + 1)]
                    tp = [p for p in tp if m[min(int(p[1]), m.shape[0] - 1), min(max(int(p[0]), 0), m.shape[1] - 1)]]
                    if len(tp) > 1:
                        paths.append(tp)
    if path is not None:
        pp = [(u * cv.w, v_ * cv.h) for u, v_ in path]
        dense = []
        for (a, b) in zip(pp[:-1], pp[1:]):
            n = max(int(math.hypot(b[0] - a[0], b[1] - a[1]) / (lpx * 0.5)), 1)
            for k in range(n):
                t = k / n
                dense.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        dense.append(pp[-1])
        paths.append(dense)
    if not paths:
        return Plant("ivy", np.zeros((cv.h, cv.w), F32))
    allp = np.array([p for ph in paths for p in ph])
    x0, y0, x1, y1 = _clip_box(cv, allp[:, 0].min() - lpx * 2, allp[:, 1].min() - lpx * 2, allp[:, 0].max() + lpx * 2,
                               allp[:, 1].max() + lpx * 2)
    if x0 >= x1 or y0 >= y1:
        return Plant("ivy", np.zeros((cv.h, cv.w), F32))
    v = _Vec(x0, y0, x1, y1, aa=2)
    ramp = list(palette)
    lx, ly = _l2(light)
    for ph in paths:
        wob = [(x + math.sin(i * 0.7) * lpx * 0.15, y + math.cos(i * 0.9) * lpx * 0.1) for i, (x, y) in enumerate(ph)]
        v.line(wob, max(lpx * 0.08, 0.7), "#4a3a22")
        for i, (x, y) in enumerate(wob):
            if rng.random() > 0.85 * density:
                continue
            sd = 1 if i % 2 else -1
            ang = -math.pi / 2 + sd * rng.uniform(0.6, 1.4) + rng.normal(0, 0.3)
            size = lpx * rng.uniform(0.6, 1.05)
            lit = 0.5 - 0.5 * (math.cos(ang) * lx + math.sin(ang) * ly)
            tone = 0.3 + 0.4 * lit + rng.normal(0, 0.08)
            _ivy_leaf(v, x, y, size * 0.55, ang, _tone_col(ramp, tone), _tone_col(ramp, tone + 0.25))
    P = v.patch()
    return _finish(cv, "ivy", P, tuple(allp[0]), tuple(allp[np.argmin(allp[:, 1])]), tuple(allp.mean(0)),
                   paint=paint, clip=clip, light=light)


def moss(cv, mask=None, region=None, seed=0, *, coverage=0.55, scale=40, light=LIGHT,
         palette=("#1a2a10", "#2f4818", "#4f6e22", "#7f9a34", "#b6c45c"), paint=True, clip=None):
    """Moss patches: clumped cushions (cellular, not noise blotches) with a fine fuzzy surface and lit tops.
    Give ``mask`` (where moss may grow, e.g. a rock's top) or ``region=(u0, v0, u1, v1)``."""
    S = _S(cv)
    if mask is None:
        u0, v0, u1, v1 = region or (0, 0.8, 1, 1)
        mask = np.zeros((cv.h, cv.w), F32)
        mask[int(v0 * cv.h):int(v1 * cv.h), int(u0 * cv.w):int(u1 * cv.w)] = 1
    ys, xs = np.nonzero(np.asarray(mask) > 0.02)
    if len(ys) == 0:
        return Plant("moss", np.zeros((cv.h, cv.w), F32))
    P = _Patch(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    X, Y = P.grid()
    m = np.asarray(mask, F32)[P.y0:P.y1, P.x0:P.x1]
    cs = scale * S
    f1, f2, fx, fy, cid = _worley(X, Y, cs, seed + 3)
    cush = clamp01(1 - f1 / (cs * (0.45 + 0.5 * cid))) ** 0.6           # each cell = one cushion
    on = (cid < coverage) * cush
    fine1, fine2, _, _, fid = _worley(X, Y, 1.6 * S + 0.6, seed + 4)
    fuzz = (fine2 - fine1) / (1.6 * S + 0.6)
    edge = smoothstep(0.05, 0.35, on + (fuzz - 0.3) * 0.25)
    a = edge * m
    L = _l2(light)
    lit = clamp01(0.5 + (-(X - fx) * L[0] - (Y - fy) * L[1]) / cs)
    tone = np.clip(0.2 + 0.55 * cush * lit + 0.25 * (fid - 0.5), 0, 1)
    P.rgb = _ramp(list(palette), tone)
    P.a = a.astype(F32)
    return _finish(cv, "moss", P, (float(xs.mean()), float(ys.max())), (float(xs.mean()), float(ys.min())),
                   (float(xs.mean()), float(ys.mean())), paint=paint, clip=clip)


def lily_pads(cv, at=(0.5, 0.8), n=7, size=0.07, seed=0, *, spread=0.18, flowers=2, light=LIGHT,
              palette=("#163018", "#2a5226", "#467a34", "#74a24e", "#b2d07e"), flower_color="#f2a6c4", paint=True,
              clip=None, shadow=False):
    """Lily pads floating on water (seen in perspective, each with its notch and radial veins) and a few water
    lilies. ``size`` = pad width (canvas-height fraction); ``spread`` = cluster radius."""
    rng = _rng(seed, 89)
    S = _S(cv)
    cx, cy = at[0] * cv.w, at[1] * cv.h
    R = size * cv.h * 0.5
    sp = spread * cv.h
    x0, y0, x1, y1 = _clip_box(cv, cx - sp - R * 2, cy - sp * 0.4 - R * 2, cx + sp + R * 2, cy + sp * 0.4 + R)
    if x0 >= x1 or y0 >= y1:
        return Plant("lily pads", np.zeros((cv.h, cv.w), F32))
    P = _Patch(x0, y0, x1, y1)
    X, Y = P.grid()
    L3 = _l3(light)
    pads = sorted([(cx + rng.normal(0, 0.5) * sp, cy + rng.normal(0, 0.5) * sp * 0.35, R * rng.uniform(0.6, 1.2),
                    rng.uniform(0, 2 * math.pi)) for _ in range(n)], key=lambda p: p[1])
    ramp = list(palette)
    for (px, py, r, rot) in pads:
        sl = (slice(max(int(py - r) - P.y0, 0), min(int(py + r) + 2 - P.y0, P.shape[0])),
              slice(max(int(px - r) - P.x0, 0), min(int(px + r) + 2 - P.x0, P.shape[1])))
        dx, dy = X[sl] - px, (Y[sl] - py) / 0.38
        rr = np.hypot(dx, dy)
        th = np.arctan2(dy, dx) - rot
        notch = np.abs(((th + math.pi) % (2 * math.pi)) - math.pi) < 0.18 * (rr / r) ** 0.2 + 0.05
        a = clamp01((r - rr) * 0.38 + 0.5) * (1 - notch)
        vein = smoothstep(0.92, 1.0, np.cos(th * 11)) * smoothstep(0.1 * r, 0.3 * r, rr)
        rim = smoothstep(r * 0.82, r, rr)
        tone = 0.45 + 0.2 * (-dx / r * L3[0]) + 0.15 * rim - 0.1 * vein * 0 + rng.normal(0, 0.04)
        col = _ramp(ramp, np.clip(tone + 0.18 * vein, 0, 1))
        col = _blend_rgb(col, rgb("#8a4a3a"), rim * 0.25)
        P.over(col.astype(F32), a.astype(F32), sl[1].start + P.x0, sl[0].start + P.y0)
    picks = rng.choice(len(pads), size=min(flowers, len(pads)), replace=False) if flowers else []
    for i in picks:
        px, py, r, _ = pads[i]
        fl = _Patch(P.x0, P.y0, P.x1, P.y1)
        _flower_head(fl, px, py - r * 0.15, r * 0.45, dict(petals=12, shape="pointed", layers=2, color=flower_color,
                                                        center="#f2cf4a", cup=0.55, tilt=0.55), light, rng)
        P.merge(fl)
    return _finish(cv, "lily pads", P, (cx, cy), (cx, cy - sp * 0.4), (cx, cy), paint=paint, clip=clip)


def mushroom(cv, at=(0.5, 0.95), height=0.15, seed=0, *, kind="amanita", cluster=1, light=LIGHT, shadow=True,
             clip=None, paint=True):
    """Mushroom(s): kind = "amanita" (red, white spots), "porcini" (brown cap, fat stem), "chanterelle"
    (golden funnel) or "button"; ``cluster`` = how many (sizes vary, sorted by depth)."""
    rng = _rng(seed, 91)
    S = _S(cv)
    Hp = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    looks = {
        "amanita": dict(cap=["#4a0a08", "#8a140e", "#c8261a", "#e8503a", "#ff9a7a"], stem="#efe8d8", spots=True,
                        cap_w=0.62, cap_h=0.32, stem_w=0.11, gill="#e8dcc4"),
        "porcini": dict(cap=["#2a160a", "#4e2c14", "#7a4a24", "#a8743e", "#d0a46a"], stem="#e6d6b0", spots=False,
                        cap_w=0.66, cap_h=0.36, stem_w=0.2, gill="#d8c48a"),
        "chanterelle": dict(cap=["#5a3206", "#94580e", "#d08a1e", "#eeb442", "#fbe08a"], stem="#e8a83a",
                            spots=False, cap_w=0.5, cap_h=0.14, stem_w=0.14, gill="#d89a2e", funnel=True),
        "button": dict(cap=["#5a4c40", "#8e7e6e", "#c4b6a4", "#e6dccc", "#fbf6ee"], stem="#efe8dc", spots=False,
                       cap_w=0.55, cap_h=0.35, stem_w=0.16, gill="#9a7a6a"),
    }
    if kind not in looks:
        raise _unknown("mushroom kind", kind, looks)
    lk = looks[kind]
    shrooms = [(bx, by, Hp)]
    for k in range(int(cluster) - 1):
        shrooms.append((bx + rng.normal(0, 0.5) * Hp * 0.6, by + rng.uniform(-0.06, 0.04) * Hp,
                        Hp * rng.uniform(0.45, 0.85)))
    shrooms.sort(key=lambda s: s[1])
    x0, y0, x1, y1 = _clip_box(cv, min(s[0] for s in shrooms) - Hp, by - Hp * 1.1, max(s[0] for s in shrooms) + Hp,
                               by + Hp * 0.1)
    if x0 >= x1 or y0 >= y1:
        return Plant("mushroom", np.zeros((cv.h, cv.w), F32))
    P = _Patch(x0, y0, x1, y1)
    X, Y = P.grid()
    L3 = _l3(light)
    for (sx, sy, h) in shrooms:
        cw, ch, swr = lk["cap_w"] * h * 0.5, lk["cap_h"] * h, lk["stem_w"] * h * 0.5
        capy = sy - h * 0.7
        lean_ = rng.normal(0, 0.05) * h
        # stem
        t = clamp01((sy - Y) / (sy - capy))
        sw = swr * (1 + 0.35 * (1 - t) ** 3 if kind != "porcini" else 1 + 0.6 * np.sin(np.pi * np.clip(t * 0.9, 0, 1)))
        xc = sx + lean_ * t
        d = np.abs(X - xc) - sw
        a = clamp01(0.5 - d) * (Y <= sy) * (Y >= capy)
        nx = np.clip((X - xc) / np.maximum(sw, 1e-3), -1, 1)
        lit = np.clip(nx * L3[0] + np.sqrt(1 - nx * nx) * L3[2], 0, 1)
        col = rgb(lk["stem"])[None, None] * (0.45 + 0.6 * lit)[..., None]
        col = col * (0.75 + 0.25 * smoothstep(0, 0.25, t))[..., None]
        P.over(col.astype(F32), a.astype(F32))
        # gills: a darker ellipse under the cap rim
        cx = sx + lean_
        gd = ((X - cx) / cw) ** 2 + ((Y - capy) / (ch * 0.28)) ** 2
        ga = clamp01((1 - gd) * min(cw, ch) * 0.3 + 0.5) * (Y >= capy)
        gl = np.cos(np.arctan2((Y - capy) / (ch * 0.28), (X - cx) / cw) * 40) * 0.5 + 0.5
        gcol = rgb(lk["gill"])[None, None] * (0.55 + 0.25 * gl)[..., None]
        P.over(gcol.astype(F32), ga.astype(F32))
        # cap
        if lk.get("funnel"):
            cd = ((X - cx) / cw) ** 2 + ((Y - capy) / ch) ** 2
            ca = clamp01((1 - cd) * ch * 0.5 + 0.5) * (Y <= capy + ch * 0.25)
        else:
            cd = ((X - cx) / cw) ** 2 + ((Y - capy) / ch) ** 2
            ca = clamp01((1 - cd) * ch * 0.5 + 0.5) * clamp01(capy + 1 - Y + ch * 0.02)
        nxx = (X - cx) / cw
        nyy = (Y - capy) / ch
        nz = np.sqrt(np.clip(1 - nxx ** 2 - nyy ** 2, 0, 1))
        d = np.clip(nxx * L3[0] + nyy * L3[1] + nz * L3[2], 0, 1)
        spec = np.clip(2 * d * nz - L3[2], 0, 1) ** 16 * 0.35
        ccol = _ramp(lk["cap"], 0.15 + 0.75 * d) + spec[..., None]
        if lk["spots"]:
            f1, f2, _, _, cid = _worley((X - cx) / (1 + 0.6 * np.abs(nxx)), Y, cw * 0.32, seed + int(sx))
            spot = clamp01((cw * 0.075 * (0.6 + 0.6 * cid) - f1) * 0.8 + 0.5) * (nz > 0.25)
            ccol = _blend_rgb(ccol, rgb("#f6efe2") * (0.7 + 0.35 * d[..., None]), spot)
        P.over(ccol.astype(F32), ca.astype(F32))
    return _finish(cv, "mushroom", P, (bx, by), (bx, by - Hp), (bx, by - Hp * 0.7), paint=paint, clip=clip,
                   shadow=shadow, light=light, shadow_len=0.6)


def pumpkin(cv, at=(0.5, 0.95), size=0.15, seed=0, *, color="#e07a1e", ribs=10, light=LIGHT, shadow=True,
            clip=None, paint=True):
    """Pumpkin: ribbed squashed sphere with grooves, a glossy highlight and a curly stem."""
    rng = _rng(seed, 93)
    S = _S(cv)
    R = size * cv.h * 0.5
    bx, by = at[0] * cv.w, at[1] * cv.h
    cx, cy = bx, by - R * 0.72
    x0, y0, x1, y1 = _clip_box(cv, cx - R * 1.4, cy - R * 1.3, cx + R * 1.4, by + 2)
    if x0 >= x1 or y0 >= y1:
        return Plant("pumpkin", np.zeros((cv.h, cv.w), F32))
    P = _Patch(x0, y0, x1, y1)
    X, Y = P.grid()
    dx, dy = (X - cx) / (R * 1.18), (Y - cy) / (R * 0.74)
    q = dx * dx + dy * dy
    a = clamp01((1 - np.sqrt(q)) * R * 0.74 + 0.5)
    lon = np.arcsin(np.clip(dx / np.sqrt(np.clip(1 - dy * dy, 1e-3, 1)), -1, 1))
    rib = np.abs(np.cos(lon * ribs / 2))
    groove = 1 - smoothstep(0.0, 0.35, rib)
    nz = np.sqrt(np.clip(1 - q, 0, 1))
    nx = dx + np.sin(lon * ribs / 2) * 0.18 * np.sign(np.cos(lon * ribs / 2)) * nz
    L3 = _l3(light)
    nn = np.sqrt(nx * nx + dy * dy + nz * nz) + 1e-6
    d = np.clip((nx * L3[0] + dy * L3[1] + nz * L3[2]) / nn, 0, 1)
    base = rgb(color)
    col = base[None, None] * (0.18 + 0.95 * d)[..., None] * (1 - 0.4 * groove)[..., None]
    spec = np.clip(2 * d * nz / nn - L3[2], 0, 1) ** 14 * 0.35 * rib
    col = col + spec[..., None]
    P.over(col.astype(F32), a.astype(F32))
    v = _Vec(P.x0, P.y0, P.x1, P.y1)
    top = (cx, cy - R * 0.68)
    st = _bez(top, (cx + R * 0.05, cy - R * 0.95), (cx + R * 0.22, cy - R * 1.05), 8)
    v.line([tuple(p) for p in st], R * 0.13, "#5a4a22")
    v.line([tuple(p) for p in st[-3:]], R * 0.08, "#7a6a3a")
    P.merge(v.patch())
    return _finish(cv, "pumpkin", P, (bx, by), (cx, cy - R), (cx, cy), paint=paint, clip=clip, shadow=shadow,
                   light=light, shadow_len=0.7)


def wheat(cv, region=(0.0, 0.65, 1.0, 1.0), seed=0, *, density=1.0, height=0.16, light=LIGHT, wind=0.15,
          palette=("#5a3e12", "#94701e", "#c9a23a", "#e8cb68", "#f8e8a8"), paint=True, clip=None):
    """Wheat field: rows of stalks with grain ears (chevrons of kernels and awns), smaller towards the horizon."""
    rng = _rng(seed, 95)
    S = _S(cv)
    u0, v0, u1, v1 = region
    x0, y0, x1, y1 = _clip_box(cv, u0 * cv.w, v0 * cv.h - height * cv.h, u1 * cv.w, v1 * cv.h)
    if x0 >= x1 or y0 >= y1:
        return Plant("wheat", np.zeros((cv.h, cv.w), F32))
    ramp = list(palette)
    base = _Patch(x0, y0, x1, y1)
    BX, BY = base.grid()
    top = v0 * cv.h - height * cv.h * 0.25 + fbm2d_at(BX / (90 * S), np.zeros_like(BX), 2, seed=seed) * 6 * S
    dpt = np.clip((BY / cv.h - v0) / max(v1 - v0, 1e-6), 0, 1)
    fx = max((x1 - x0) * 0.04, 6 * S)
    ragged = fbm2d_at(BX / (30 * S + 1), BY / (30 * S + 1), 2, seed=seed + 3) * 0.5 + 0.5
    base.a = (clamp01(BY - top + 0.5) * clamp01((v1 * cv.h - BY) / (height * cv.h * 0.12) + 0.2)
              * smoothstep(0, fx, (BX - u0 * cv.w) + ragged * fx) * smoothstep(0, fx, (u1 * cv.w - BX) + ragged * fx))
    streak = fbm2d_at(BX / (3 * S + 1), BY / (25 * S + 1), 3, seed=seed + 1)
    base.rgb = _ramp(ramp, np.clip(0.62 - 0.3 * dpt + 0.12 * streak, 0, 1)).astype(F32)
    v = _Vec(x0, y0, x1, y1, aa=2)
    n = int(1600 * density * (u1 - u0) * (v1 - v0) / 0.35)
    stalks = sorted([(rng.uniform(u0, u1) * cv.w, (v0 + (v1 - v0) * rng.random() ** 0.6) * cv.h) for _ in range(n)],
                    key=lambda s: s[1])
    lx = _l2(light)[0]
    for (x, y) in stalks:
        d = 0.25 + 0.75 * (y / cv.h - v0) / max(v1 - v0, 1e-6)
        h = height * cv.h * d * rng.uniform(0.8, 1.1)
        lean_ = wind + rng.normal(0, 0.06)
        top = (x + lean_ * h, y - h)
        tone = 0.35 + 0.45 * d + rng.normal(0, 0.06)
        v.line([(x, y), (x + lean_ * h * 0.4, y - h * 0.5), top], max(h * 0.012, 0.5), _tone_col(ramp, tone * 0.8))
        ear = h * 0.28
        ang = math.atan2(-1, lean_ * 1.5)
        ca, sa = math.cos(ang), math.sin(ang)
        for k in range(7):
            t = k / 7
            px, py = top[0] + ca * ear * t, top[1] + sa * ear * t
            for sd in (-1, 1):
                kx = px + (-sa) * sd * ear * 0.1
                ky = py + ca * sd * ear * 0.1
                v.poly([(kx - ca * ear * 0.1, ky - sa * ear * 0.1), (kx + (-sa) * sd * ear * 0.08, ky + ca * sd * ear * 0.08),
                        (kx + ca * ear * 0.11, ky + sa * ear * 0.11), (kx - (-sa) * sd * ear * 0.05, ky - ca * sd * ear * 0.05)],
                       _tone_col(ramp, tone + 0.1 * sd * -lx + 0.05))
        v.line([top, (top[0] + ca * ear * 1.6, top[1] + sa * ear * 1.6)], max(h * 0.004, 0.4), _tone_col(ramp, tone + 0.2))
    base.merge(v.patch())
    P = base
    return _finish(cv, "wheat", P, (cv.w * (u0 + u1) / 2, v1 * cv.h), (cv.w * (u0 + u1) / 2, y0),
                   (cv.w * (u0 + u1) / 2, (y0 + y1) / 2), paint=paint, clip=clip)


# ============================================================================
# flowers: polar petal layers + centre + stem + leaves
# ============================================================================
def _petal_w(shape, u, W):
    """Half-width of a petal at position u (0 base → 1 tip), normalised by the layer radius."""
    u = np.clip(u, 0, 1)
    if shape == "pointed":
        return W * np.clip(np.sin(np.pi * u ** 0.75), 0, 1) ** 0.85 * (1 - u) ** 0.15
    if shape == "spatula":            # daisy ray: narrow strap with a rounded end
        return W * (0.45 + 0.55 * smoothstep(0, 0.35, u)) * np.sqrt(np.clip(1 - ((u - 0.55) / 0.45) ** 8, 0, 1))
    if shape == "notched":            # cherry / sakura: round with a notch at the tip
        base = W * np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1)) ** 0.8
        return base
    if shape == "frilly":
        return W * np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1)) ** 0.6
    if shape == "heart":
        return W * np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1)) ** 0.5
    if shape in ("obovate", "toothed"):
        return W * 1.42 * np.sqrt(u) * np.sqrt(np.clip(1 - u ** 3.2, 0, 1))
    return W * np.sqrt(np.clip(1 - (2 * u - 1) ** 2, 0, 1)) ** 0.7     # round


def _flower_head(P, cx, cy, R, spec, light, rng):
    """Paint a flower head (front / tilted view) of radius R px centred at (cx, cy) into patch P."""
    n = int(spec.get("petals", 5))
    layers = int(spec.get("layers", 1))
    shape = spec.get("shape", "round")
    tilt = float(spec.get("tilt", 0.0))
    squash = 1 - 0.62 * tilt
    rot0 = spec.get("rot", rng.uniform(0, 2 * math.pi))
    cols = spec.get("color", "#e8443a")
    cols = cols if isinstance(cols, (list, tuple)) and not isinstance(cols[0], (int, float)) else [cols]
    tip = spec.get("tip")
    base_c = spec.get("base")
    W = spec.get("width", min(0.95, 2.6 / n + 0.12))
    cup = spec.get("cup", 0.3)
    veins = spec.get("veins", 0.25)
    pad = R * 1.45 + 3
    X0, Y0, X1, Y1 = int(cx - pad), int(cy - pad * squash - 2), int(cx + pad) + 2, int(cy + pad * squash) + 3
    X0, Y0, X1, Y1 = max(X0, P.x0), max(Y0, P.y0), min(X1, P.x1), min(Y1, P.y1)
    if X0 >= X1 or Y0 >= Y1:
        return
    sy, sx = np.mgrid[Y0:Y1, X0:X1].astype(F32)
    dx = sx - cx
    dy = (sy - cy) / squash
    rho = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    lx, ly = _l2(light)
    L3 = _l3(light)
    ruffle = float(spec.get("ruffle", 0.0))
    for li in range(layers):
        Rl = R * (1 - cup * li / max(layers, 1) * 0.9) if layers > 1 else R
        rot = rot0 + li * (math.pi / n) * (1.0 if spec.get("stagger", True) else 0) + li * 0.37
        c0 = rgb(cols[min(li, len(cols) - 1)])
        shs = max(R * 0.05, 0.8)                           # contact-shadow offset / softness
        for j in range(n):                                # painter's order: every petal over the previous ones
            jit = float(_hash(j, li, 17))
            Lk = Rl * (0.88 + 0.2 * jit)
            pa = rot + j * 2 * math.pi / n
            ca, sa_ = math.cos(pa), math.sin(pa)
            ext = Lk * (W + 0.12) + shs * 2 + 2
            px0 = int(max(X0, math.floor(cx + min(0, ca * Lk) - ext)))
            px1 = int(min(X1, math.ceil(cx + max(0, ca * Lk) + ext)))
            py0 = int(max(Y0, math.floor(cy + min(0, sa_ * Lk * squash) - ext * squash - 1)))
            py1 = int(min(Y1, math.ceil(cy + max(0, sa_ * Lk * squash) + ext * squash + 1)))
            if px0 >= px1 or py0 >= py1:
                continue
            gy_, gx_ = np.mgrid[py0:py1, px0:px1].astype(F32)
            ddx = gx_ - cx
            ddy = (gy_ - cy) / squash

            def field(ddx, ddy):
                u_ = (ddx * ca + ddy * sa_) / Lk
                v_ = (-ddx * sa_ + ddy * ca) / Lk
                w_ = _petal_w(shape, u_, W)
                if shape == "frilly":
                    w_ = w_ * (1 + 0.12 * np.sin(u_ * 30 + jit * 6))
                elif shape == "toothed":
                    w_ = w_ * 1.0
                d_ = (np.abs(v_) - w_) * Lk
                dt_ = (u_ - 1) * Lk
                if shape == "notched":                  # V notch at the petal tip
                    dt_ = dt_ + Lk * 0.2 * np.exp(-(v_ / max(0.28 * W, 0.05)) ** 2) * smoothstep(0.5, 1.0, u_)
                if shape == "toothed":
                    dt_ = dt_ + Lk * 0.09 * (0.5 + 0.5 * np.cos(v_ / max(W, 0.1) * 7.5)) * smoothstep(0.55, 1, u_)
                if ruffle > 0:
                    e_ = ruffle * 0.05 * Lk * np.sin(v_ * 11 + u_ * 5 + jit * 9) * smoothstep(0.35, 0.9, u_)
                    d_ = d_ + e_
                    dt_ = dt_ + e_ * 1.4
                dd_ = np.maximum(np.maximum(d_, dt_), np.maximum(-u_ * Lk, (u_ - 1.0) * Lk))
                return dd_, u_, v_, w_
            dd, u, v_, w = field(ddx, ddy)
            a = clamp01(0.5 - dd)
            if not a.any():
                continue
            if j or li:                                    # contact shadow on what is already painted below
                dd2 = field(ddx + lx * shs, ddy + ly * shs)[0]
                sh = clamp01(0.5 - dd2 / shs) * (1 - a)
                reg = P.rgb[py0 - P.y0:py1 - P.y0, px0 - P.x0:px1 - P.x0]
                reg *= (1 - 0.28 * sh)[..., None]
            vv = v_ / np.maximum(w, 1e-3)
            facing = 0.5 - 0.5 * (ca * lx + sa_ * ly * squash)
            tone = 0.9 + 0.22 * (facing - 0.5) * (0.4 + tilt)
            tone = tone - 0.4 * (1 - smoothstep(0.0, 0.6, u)) * (0.35 + cup)
            tone = tone - 0.2 * smoothstep(0.6, 1.0, np.abs(vv))
            tone = tone + 0.06 * (li / max(layers, 1)) + 0.05 * (jit - 0.5)
            # petal volume: crease down the middle, lit side towards the light
            side = (-sa_ * lx + ca * ly)
            tone = tone + 0.14 * vv * side * smoothstep(0.1, 0.5, u) - 0.05 * np.cos(np.clip(vv, -1, 1) * 1.4)
            if ruffle > 0:
                tone = tone + 0.045 * ruffle * np.sin(v_ * 13 + u * 7 + jit * 7)
            vein = (0.5 + 0.5 * np.cos(vv * np.pi * 4.5)) ** 6 * smoothstep(0.08, 0.3, u) * (1 - u)
            col = c0[None, None] * np.clip(tone, 0.05, 1.3)[..., None]
            if tip is not None:
                col = _blend_rgb(col, rgb(tip) * np.clip(tone + 0.1, 0, 1.3)[..., None], smoothstep(0.45, 1.0, u))
            if base_c is not None:
                col = _blend_rgb(col, rgb(base_c) * np.clip(tone + 0.15, 0, 1.3)[..., None], 1 - smoothstep(0.05, 0.4, u))
            col = col * (1 - veins * 0.35 * vein)[..., None]
            P.over(col.astype(F32), a.astype(F32), px0, py0)
    # centre
    ctr = spec.get("center")
    if ctr:
        cr = R * spec.get("center_r", 0.22)
        q = rho / cr
        a = clamp01((1 - q) * cr + 0.5)
        kind = spec.get("center_kind", "dome")
        nz = np.sqrt(np.clip(1 - q * q, 0, 1))
        d = np.clip((dx / cr) * L3[0] + (dy / cr) * L3[1] + nz * L3[2], 0, 1)
        col = rgb(ctr)[None, None] * (0.35 + 0.75 * d)[..., None]
        if kind == "seeds":                                      # sunflower: phyllotaxis spirals
            g = np.log(np.maximum(rho, 0.5)) * 9.0
            sp1 = np.cos(th * 21 + g * 2.0)
            sp2 = np.cos(th * 34 - g * 3.2)
            pat = clamp01(sp1 * sp2 * 1.5 + 0.3)
            col = col * (0.6 + 0.45 * pat)[..., None]
            col = _blend_rgb(col, rgb("#c88a1a") * (0.6 + 0.4 * d[..., None]), smoothstep(0.75, 0.98, q) * 0.8)
        elif kind == "stamens":                                  # poppy / lily: dark eye and a ring of stamens
            ring = smoothstep(0.55, 0.75, q) * (0.5 + 0.5 * np.cos(th * 28)) ** 4
            col = _blend_rgb(col, rgb(spec.get("stamen", "#2a2030")), ring)
        elif kind == "spiral":                                   # rose bud centre: tight spiral of petal edges
            sp_ = 0.5 + 0.5 * np.cos(th + q * 14)
            col = col * (1 - 0.45 * smoothstep(0.7, 1.0, sp_))[..., None]
        elif kind == "dots":                                     # daisy: tiny florets
            f1, f2, _, _, _ = _worley(sx, sy, max(cr * 0.16, 0.8), 5)
            col = col * (0.8 + 0.25 * smoothstep(0, cr * 0.06, f2 - f1))[..., None]
        P.over(col.astype(F32), a.astype(F32), X0, Y0)


FLOWERS = {
    "rose": dict(petals=5, layers=6, shape="obovate", color=["#8e0f1e", "#a8152a", "#b81c30", "#c62236", "#d02a3c", "#c01a2e"],
                 cup=0.85, width=0.75, tilt=0.25, veins=0.05, center="#9a1424", center_r=0.2, center_kind="spiral",
                 head=0.11, stem="#3e5a24", leaf="rose",
                 leaves=4),
    "tulip": dict(side="tulip", color="#e2283a", tip="#f25a5a", head=0.13, stem="#4a7a2e", leaf="strap", leaves=2),
    "sunflower": dict(petals=26, layers=2, shape="pointed", color=["#e8a813", "#f2c21e"], width=0.16, cup=0.15,
                      center="#4a2a12", center_r=0.42, center_kind="seeds", tilt=0.1, head=0.16, stem="#4f7a28",
                      leaf="heart", leaves=4, veins=0.35),
    "daisy": dict(petals=22, layers=1, shape="spatula", color="#f8f6f0", width=0.1, center="#f2b81e", center_r=0.26,
                  center_kind="dots", tilt=0.2, head=0.1, stem="#5a8a32", leaf="narrow", leaves=3, veins=0.4),
    "poppy": dict(petals=4, layers=2, shape="obovate", ruffle=0.9, color=["#d8221a", "#e8301e"], width=0.95, cup=0.3, center="#2a1c22", center_r=0.2,
                  center_kind="stamens", stamen="#140e14", base="#401016", tilt=0.25, head=0.11, stem="#6a8a3e",
                  leaf="narrow", leaves=2, veins=0.5),
    "cosmos": dict(petals=8, layers=1, shape="toothed", color="#e66aa8", width=0.4, center="#f2c234", center_r=0.18,
                   center_kind="dots", tilt=0.15, head=0.09, stem="#5a8a32", leaf="narrow", leaves=2, veins=0.45),
    "lily": dict(petals=6, layers=1, shape="pointed", color="#f4f0ea", tip="#fbe8ee", base="#f6d6a0", width=0.4,
                 center="#c8a23a", center_r=0.08, center_kind="dome", tilt=0.35, head=0.14, stem="#4a7a2e",
                 leaf="strap", leaves=4, veins=0.5, spots="#a2365a", stamens=True),
    "orchid": dict(orchid=True, color="#d77fc4", lip="#8a1a6a", head=0.11, stem="#3e5a2a", leaf="orchid", leaves=2),
    "lavender": dict(spike="lavender", color="#8a6ccc", head=0.2, stem="#7a9060", leaf="narrow", leaves=2),
    "dandelion": dict(puff=True, head=0.09, stem="#7a9a4a", leaf="dandelion", leaves=3),
    "dandelion flower": dict(petals=40, layers=3, shape="spatula", color=["#f0b81a", "#f6c822", "#fad640"],
                             width=0.06, cup=0.6, tilt=0.15, head=0.07, stem="#6a9a3a", leaf="dandelion", leaves=3,
                             veins=0.2),
    "lotus": dict(side="lotus", color="#f4a6c0", base="#fff0f4", head=0.16, stem=None, leaf=None, leaves=0),
    "cherry blossom": dict(petals=5, layers=1, shape="notched", color="#f7c4d6", base="#fde9f0", width=0.46,
                           center="#e8608a", center_r=0.16, center_kind="stamens", stamen="#b0284c", tilt=0.1,
                           head=0.05, stem="#4a2a22", leaf=None, leaves=0, veins=0.3),
    "cornflower": dict(petals=13, layers=2, shape="toothed", color=["#2f56c8", "#4a6ee0"], width=0.2, center="#26246a",
                       center_r=0.2, center_kind="dots", cup=0.4, tilt=0.15, head=0.07, stem="#6a8a5a", leaf="narrow",
                       leaves=2, veins=0.4),
    "marigold": dict(petals=14, layers=5, shape="obovate", ruffle=0.7, color=["#d8580e", "#e66a12", "#f08018", "#f4921e", "#f8a82a"],
                     width=0.42, cup=0.85, tilt=0.2, head=0.08, stem="#4a7a2e", leaf="narrow", leaves=3, veins=0.25),
    "hibiscus": dict(petals=5, layers=1, shape="obovate", ruffle=0.55, color="#ee3a50", base="#6a0a1c", width=0.66, center="#f6d23a",
                     center_r=0.06, center_kind="dome", tilt=0.2, head=0.15, stem="#3e6a2a", leaf="rose", leaves=3,
                     veins=0.6, stamens=True),
    "peony": dict(petals=7, layers=7, shape="obovate", ruffle=0.45, color=["#e88aa8", "#ec96b2", "#f0a2bc", "#f2aec4", "#f6bccd",
                                                             "#f8c8d6", "#fad4de"], width=0.6, cup=0.85, tilt=0.25,
                  head=0.13, stem="#4a6a2a", leaf="rose", leaves=3, veins=0.15),
}
_FLOWER_ALIAS = {"sakura": "cherry blossom", "water lily": "lotus", "gerbera": "daisy", "aster": "cosmos",
                 "chrysanthemum": "marigold", "carnation": "marigold"}


def _flower_leaf(v, base, ang, L, kind, ramp, tone, S):
    """Leaves on a flower stem: ovate (rose), strap (tulip, lily), narrow, heart, dandelion (toothed), orchid."""
    dx, dy = math.cos(ang), math.sin(ang)
    if kind in ("strap", "orchid"):
        end = (base[0] + dx * L, base[1] + dy * L + L * 0.25)
        spine = _bez(base, (base[0] + dx * L * 0.4, base[1] + dy * L * 0.6), end, 14)
        wmax = L * (0.09 if kind == "strap" else 0.16)
        prof = lambda u: wmax * np.sin(np.pi * np.clip(u * 0.9 + 0.1, 0, 1)) ** 0.6 * (1 - 0.4 * u)
    elif kind == "narrow":
        end = (base[0] + dx * L, base[1] + dy * L)
        spine = _bez(base, (base[0] + dx * L * 0.5, base[1] + dy * L * 0.5 - L * 0.1), end, 12)
        prof = lambda u: L * 0.07 * np.sin(np.pi * np.clip(u, 0, 1)) ** 0.7
    elif kind == "heart":
        end = (base[0] + dx * L, base[1] + dy * L + L * 0.15)
        spine = _bez(base, (base[0] + dx * L * 0.5, base[1] + dy * L * 0.5 - L * 0.05), end, 16)
        prof = lambda u: L * 0.36 * np.sin(np.pi * np.clip(u * 0.85 + 0.15, 0, 1)) ** 0.8 * (1 - 0.6 * u ** 2)
    elif kind == "dandelion":
        end = (base[0] + dx * L, base[1] + dy * L)
        spine = _bez(base, (base[0] + dx * L * 0.5, base[1] + dy * L * 0.5), end, 24)
        prof = lambda u: L * 0.1 * (0.45 + 0.55 * np.abs(np.sin(u * 14))) * np.sin(np.pi * np.clip(u, 0, 1)) ** 0.5
    else:                                                       # rose: serrated ovate
        end = (base[0] + dx * L, base[1] + dy * L)
        spine = _bez(base, (base[0] + dx * L * 0.5, base[1] + dy * L * 0.5 - L * 0.06), end, 18)
        prof = lambda u: L * 0.26 * np.sin(np.pi * np.clip(u, 0, 1) ** 0.8) * (1 + 0.08 * np.sin(u * 40))
    uu = np.linspace(0, 1, len(spine))
    wv = prof(uu)
    left, right, _ = _half_blades(spine, wv)
    v.poly(left, _tone_col(ramp, tone + 0.1))
    v.poly(right, _tone_col(ramp, tone - 0.08))
    v.line([tuple(p) for p in spine[:-2]], max(L * 0.012, 0.5), _tone_col(ramp, tone + 0.28))


LEAF_GREEN = ["#14280e", "#24461a", "#3c6a26", "#66963a", "#a4c868"]


def _side_flower(P, kind, cx, cy, R, spec, light, rng):
    """Side-view cup flowers: tulip (three overlapping pointed petals) and lotus (layered fan of petals)."""
    v = _Vec(P.x0, P.y0, P.x1, P.y1, aa=3)
    lx, _ = _l2(light)
    c = rgb(spec.get("color", "#e2283a"))
    if kind == "tulip":
        # egg-shaped cup of three pointed petals: two outer ones lean outwards, the centre one stands in front
        for off, hgt, wid, tn in ((-1.0, 0.8, 0.5, 0.66), (1.0, 0.8, 0.5, 0.58), (0.0, 1.0, 0.56, 0.98)):
            t = np.linspace(0.0, 1.0, 28)
            w = wid * R * 0.95 * (t ** 0.5) * ((1 - t) ** 1.1) * 2.6
            xc = cx + off * R * (0.42 * t ** 1.6 + 0.04)
            yy = cy + R * 0.95 - t * R * 1.9 * hgt
            poly = [(xc[i] - w[i], yy[i]) for i in range(len(t))] + [(xc[i] + w[i], yy[i]) for i in range(len(t) - 1, -1, -1)]
            shade = tn * (1 - 0.18 * off * -lx)
            v.poly(poly, tuple(np.clip(c * shade, 0, 1)))
            if off == 0.0:                                   # lit left half of the front petal and a soft crease
                half = [(xc[i] - w[i], yy[i]) for i in range(len(t))] + [(xc[i] - w[i] * 0.05, yy[i]) for i in range(len(t) - 1, -1, -1)]
                v.poly(half, tuple(np.clip(c * 1.12 * (1 - 0.0 * lx), 0, 1)))
        pp = v.patch()
        sh = _shade_from_alpha(pp.a, R * 0.25, light, 0.35)
        Y = np.arange(pp.y0, pp.y1, dtype=F32)[:, None]
        tipm = clamp01((cy - R * 0.2 - Y) / R)
        pp.rgb = (pp.rgb * sh[..., None]).astype(F32)
        if spec.get("tip"):
            pp.rgb = _blend_rgb(pp.rgb, rgb(spec["tip"]) * sh[..., None], (tipm * 0.6))
        P.merge(pp)
        return
    # lotus
    base = rgb(spec.get("base", "#fff0f4"))
    for ring, (n, spread, L, tn) in enumerate(((7, 1.25, 1.0, 0.75), (5, 0.9, 0.95, 0.9), (3, 0.45, 0.85, 1.0))):
        for j in range(n):
            a = -math.pi / 2 + (j / max(n - 1, 1) - 0.5) * 2 * spread
            dx, dy = math.cos(a), math.sin(a)
            ln = R * L * (1 - 0.25 * abs(j / max(n - 1, 1) - 0.5))
            tipp = (cx + dx * ln, cy + R * 0.4 + dy * ln * 0.95)
            b0 = (cx + dx * R * 0.05, cy + R * 0.45)
            perp = (-dy, dx)
            wd = ln * 0.3
            mid = ((b0[0] + tipp[0]) / 2, (b0[1] + tipp[1]) / 2)
            poly = [b0, (mid[0] + perp[0] * wd, mid[1] + perp[1] * wd), tipp, (mid[0] - perp[0] * wd, mid[1] - perp[1] * wd)]
            poly = [tuple(p) for p in _bez(b0, (mid[0] + perp[0] * wd * 1.6, mid[1] + perp[1] * wd * 1.6), tipp, 10)] + \
                   [tuple(p) for p in _bez(tipp, (mid[0] - perp[0] * wd * 1.6, mid[1] - perp[1] * wd * 1.6), b0, 10)]
            lit = 0.85 + 0.15 * (-dx * lx)
            col = (c * 0.8 + base * 0.2) * tn * lit
            v.poly(poly, tuple(np.clip(col, 0, 1)))
    pp = v.patch()
    Y = np.arange(pp.y0, pp.y1, dtype=F32)[:, None]
    g = clamp01((Y - (cy - R * 0.5)) / (R * 1.0))
    pp.rgb = _blend_rgb(pp.rgb, base[None, None] * np.ones_like(pp.rgb), g * 0.65)
    P.merge(pp)


def _spike(P, kind, x, y, L, ang, spec, light, rng, S):
    """Flower spike (lavender): whorls of tiny florets along the top of the stem."""
    c = rgb(spec.get("color", "#8a6ccc"))
    n = max(int(L / max(1.6 * S, 1.0)), 10)
    dx, dy = math.cos(ang), math.sin(ang)
    px, py = -dy, dx
    dots = []
    for k in range(n):
        t = k / n
        w = L * 0.075 * (1 - 0.55 * t ** 1.3) * (0.35 + 0.65 * min(1.0, t * 5))   # slim spike, widest in the lower third
        for j in range(5):
            side = rng.uniform(-1, 1)
            ox = side * w
            oy = rng.normal(0, L / n * 0.9)
            r = max(L * 0.03 * (1 - 0.4 * t) * rng.uniform(0.85, 1.2), 0.9)
            dots.append((abs(side) * -1 + rng.normal(0, 0.1), t, ox, oy, r))
    dots.sort(key=lambda d: d[0])                                 # centre florets in front of the outer ones
    for _, t, ox, oy, r in dots:
        cxp = x + dx * L * t + px * ox + dx * oy
        cyp = y + dy * L * t + py * ox + dy * oy
        tone = c * rng.uniform(0.78, 1.12) * (0.92 + 0.16 * t) + (0.08 * t)
        _sphere_dot(P, cxp, cyp, r, tuple(np.clip(tone, 0, 1)), light, gloss=0.04)
    for k in range(3):                                            # unopened buds at the tip
        t = 0.97 + k * 0.02
        _sphere_dot(P, x + dx * L * t, y + dy * L * t, max(L * 0.016, 0.8), tuple(np.clip(c * 1.05 + 0.05, 0, 1)), light, gloss=0.04)


def _puff(P, cx, cy, R, light, rng, S):
    """Dandelion seed head: radial pappus threads with tufts, on a small receptacle."""
    v = _Vec(P.x0, P.y0, P.x1, P.y1, aa=3)
    n = 90
    for k in range(n):
        u = rng.uniform(-1, 1)
        a = rng.uniform(0, 2 * math.pi)
        r = math.sqrt(1 - u * u)
        dx, dy = r * math.cos(a), u
        tip = (cx + dx * R, cy + dy * R)
        v.line([(cx, cy), tip], max(0.35 * S, 0.35), (0.86, 0.86, 0.84))
        for j in range(7):
            b = a + (j - 3) * 0.35
            v.line([tip, (tip[0] + math.cos(math.atan2(dy, dx) + (j - 3) * 0.3) * R * 0.12,
                          tip[1] + math.sin(math.atan2(dy, dx) + (j - 3) * 0.3) * R * 0.12)],
                   max(0.3 * S, 0.3), (0.97, 0.97, 0.96))
    pp = v.patch()
    pp.a = pp.a * 0.92
    P.merge(pp)
    _sphere_dot(P, cx, cy, R * 0.1, "#8a7a4a", light, gloss=0.1)


def _orchid(P, cx, cy, R, spec, light, rng):
    """Phalaenopsis orchid: three narrow sepals, two broad petals, a dark lip."""
    c = spec.get("color", "#d77fc4")
    _flower_head(P, cx, cy, R * 0.95, dict(petals=3, shape="pointed", color=c, width=0.32, rot=-math.pi / 2,
                                           tilt=0.0, veins=0.3, cup=0.0), light, rng)
    for sd in (-1, 1):
        _flower_head(P, cx + sd * R * 0.05, cy, R * 0.95, dict(petals=1, shape="round", color=c, width=0.62,
                                                              rot=math.pi / 2 - sd * math.pi / 2 + sd * 0.12,
                                                              tilt=0.0, veins=0.45, cup=0.0), light, rng)
    _flower_head(P, cx, cy + R * 0.28, R * 0.38, dict(petals=1, shape="heart", color=spec.get("lip", "#8a1a6a"),
                                                    width=0.9, rot=math.pi / 2, veins=0.2, tip="#c03a8a"), light, rng)
    _sphere_dot(P, cx, cy - R * 0.02, R * 0.09, "#f2e4c0", light)


def flower(cv, kind="daisy", at=(0.5, 0.95), height=0.3, seed=0, *, color=None, tilt=None, stem=True, leaves=None,
           light=LIGHT, shadow=False, clip=None, haze=None, paint=True, head=None, lean=0.0, spec=None):
    """One flower with stem and leaves. kind: rose, tulip, sunflower, daisy, poppy, cosmos, lily, orchid, lavender,
    dandelion, dandelion flower, lotus, cherry blossom, cornflower, marigold, hibiscus, peony — or a custom
    ``spec=dict(petals=8, layers=2, shape="pointed"|"round"|"spatula"|"notched"|"frilly"|"heart", color=..., tip=...,
    base=..., width=..., cup=..., center=..., center_r=..., center_kind="dome"|"seeds"|"stamens"|"dots")``.
    ``height`` = base → head (canvas fraction); ``head`` = head radius override (fraction); ``stem=False`` = head
    only (``at`` is then the head centre)."""
    rng = _rng(seed, 101)
    S = _S(cv)
    key = _FLOWER_ALIAS.get(kind.lower().strip(), kind.lower().strip()) if kind else "daisy"
    if key not in FLOWERS:
        raise _unknown("flower", kind, list(FLOWERS) + list(_FLOWER_ALIAS), "Try flora.names('flower')")
    sp = dict(FLOWERS[key])
    if spec:
        sp.update(spec)
    if color is not None:
        sp["color"] = color
    if tilt is not None:
        sp["tilt"] = tilt
    Hp = height * cv.h
    R = (head if head is not None else sp.get("head", 0.1)) * (Hp * 1.6 if stem else cv.h)
    if not stem and head is None:
        R = Hp * 0.5
    if stem and sp.get("stem") is None and sp.get("side") == "lotus":     # floats on the water: no stem
        stem = False
        R = Hp * 0.42
        at = (at[0], at[1] - R * 0.45 / cv.h)
    if key == "cherry blossom" and stem:
        return _blossom_branch(cv, at, Hp, seed, sp, light, shadow, clip, haze, paint)
    bx, by = at[0] * cv.w, at[1] * cv.h
    if stem:
        lean_ = lean + rng.normal(0, 0.08)
        hx, hy = bx + lean_ * Hp, by - Hp
    else:
        hx, hy = bx, by
    pad = max(R * 1.5, Hp * 0.45 if stem else R * 1.5)
    x0, y0, x1, y1 = _clip_box(cv, min(bx, hx) - pad, hy - R * 1.6, max(bx, hx) + pad, (by if stem else hy + R) + 3)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, key, None, (bx, by), (hx, hy), (hx, hy))
    P = _Patch(x0, y0, x1, y1)
    ramp = LEAF_GREEN
    if stem and sp.get("stem"):
        v = _Vec(x0, y0, x1, y1, aa=3)
        sw = max(Hp * 0.012, 0.8 * S) * (1.6 if key == "sunflower" else 1.0)
        ctrl = (bx + (hx - bx) * 0.2 + rng.normal(0, 0.03) * Hp, by - Hp * 0.55)
        spine = _bez((bx, by), ctrl, (hx, hy + R * 0.2), 20)
        nl = sp.get("leaves", 2) if leaves is None else leaves
        lk = sp.get("leaf")
        if lk and nl:
            for j in range(nl):
                if lk in ("strap", "dandelion", "orchid"):
                    t = 0.02
                    ang = -math.pi / 2 + (j / max(nl - 1, 1) - 0.5) * 2.4 + rng.normal(0, 0.15)
                    L = Hp * (0.55 if lk == "strap" else 0.35) * rng.uniform(0.8, 1.1)
                else:
                    t = 0.25 + 0.5 * j / max(nl, 1)
                    sd = -1 if j % 2 == 0 else 1
                    ang = -math.pi / 2 + sd * rng.uniform(0.7, 1.2)
                    L = Hp * (0.28 if lk == "heart" else 0.2) * rng.uniform(0.8, 1.15) * (1 - 0.4 * t)
                i = min(int(t * (len(spine) - 1)), len(spine) - 1)
                _flower_leaf(v, tuple(spine[i]), ang, L, lk, ramp, 0.45 + rng.normal(0, 0.06), S)
        v.line([tuple(p) for p in spine], sw, _tone_col(ramp, 0.5) if sp["stem"] is None else sp["stem"])
        sp_ = v.patch()
        sh = _shade_from_alpha(sp_.a, max(sw * 0.6, 0.8), light, 0.4)
        sp_.rgb = (sp_.rgb * sh[..., None]).astype(F32)
        P.merge(sp_)
    if sp.get("side"):
        _side_flower(P, sp["side"], hx, hy - R * 0.3, R, sp, light, rng)
    elif sp.get("orchid"):
        _orchid(P, hx, hy, R, sp, light, rng)
    elif sp.get("spike"):
        _spike(P, "lavender", hx, hy + R * 0.9, R * 1.6, -math.pi / 2 + lean * 0.3, sp, light, rng, S)
    elif sp.get("puff"):
        _puff(P, hx, hy, R, light, rng, S)
    else:
        _flower_head(P, hx, hy, R, sp, light, rng)
        if sp.get("spots"):
            _spots(P, hx, hy, R, sp, rng)
        if sp.get("stamens"):
            vv = _Vec(P.x0, P.y0, P.x1, P.y1, aa=3)
            if key == "hibiscus":                         # staminal column: pollen-dusted tube, anthers, red stigma
                a = -math.pi / 3 + rng.normal(0, 0.15)
                L = R * 0.8
                sq = 1 - 0.6 * sp.get("tilt", 0)
                ex, ey = hx + math.cos(a) * L, hy + math.sin(a) * L * sq
                vv.line([(hx, hy), (ex, ey)], max(R * 0.05, 1.0), "#f0d070")
                for t in np.linspace(0.45, 0.95, 9):
                    ox, oy = hx + math.cos(a) * L * t, hy + math.sin(a) * L * t * sq
                    for sd in (-1, 1):
                        vv.ellipse(ox - sd * math.sin(a) * R * 0.045, oy + sd * math.cos(a) * R * 0.045 * sq,
                                   R * 0.03, R * 0.022, "#f6b630")
                for k in range(5):
                    ang2 = a + (k - 2) * 0.5
                    vv.ellipse(ex + math.cos(ang2) * R * 0.04, ey + math.sin(ang2) * R * 0.03, R * 0.028, R * 0.028,
                               "#b01020")
                P.merge(vv.patch())
                vv = _Vec(P.x0, P.y0, P.x1, P.y1, aa=3)
            for j in range(0 if key == "hibiscus" else 6 if key == "lily" else 1):
                a = -math.pi / 2 + (j - 2.5) * 0.3 if key == "lily" else -math.pi / 3
                L = R * (0.6 if key == "lily" else 0.75)
                e = (hx + math.cos(a) * L, hy + math.sin(a) * L * (1 - 0.6 * sp.get("tilt", 0)))
                vv.line([(hx, hy), e], max(R * 0.025, 0.6), "#d8c890")
                vv.ellipse(e[0], e[1], R * 0.05, R * 0.03, "#a0461e" if key == "lily" else "#f2c63a")
            P.merge(vv.patch())
    Yg = np.arange(P.y0, P.y1, dtype=F32)[:, None]
    if stem:
        P.a = P.a * clamp01(by + 1 - Yg)
    return _finish(cv, key, P, (bx, by), (hx, hy - R), (hx, hy), paint=paint, clip=clip, shadow=shadow, light=light,
                   haze=haze, shadow_len=0.4, extra=dict(head_radius=R))


def _blossom_branch(cv, at, Hp, seed, sp, light, shadow, clip, haze, paint):
    """Cherry blossom sprig: a dark twig with clusters of notched five-petal flowers and buds."""
    rng = _rng(seed, 103)
    S = _S(cv)
    bx, by = at[0] * cv.w, at[1] * cv.h
    sk = _Skel()
    _grow(sk, rng, dict(nseg=5, wiggle=14, up=0.25, taper=0.55, depth=2, kids=(2, 2), spread=38, spread_jitter=10,
                        len_ratio=0.55, side_twigs=0.8), bx, by, -math.pi / 2 + rng.normal(0, 0.25), Hp * 0.6,
          Hp * 0.022, 0)
    xs = [s[0] for s in sk.segs] + [s[2] for s in sk.segs]
    ys = [s[1] for s in sk.segs] + [s[3] for s in sk.segs]
    x0, y0, x1, y1 = _clip_box(cv, min(xs) - Hp * 0.2, min(ys) - Hp * 0.2, max(xs) + Hp * 0.2, by + 3)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, "cherry blossom", None, (bx, by), (bx, by - Hp), (bx, by - Hp / 2))
    P = _Patch(x0, y0, x1, y1)
    c, a, _ = _render_bark(P, sk.segs, light, BARKS["cherry"], kind="smooth", S=S, seed=seed)
    P.rgb, P.a = c, a
    spots = [(s[2], s[3]) for s in sk.segs if s[8] >= 1] + [(t[0], t[1]) for t in sk.tips]
    for (x, y) in spots:
        for j in range(int(rng.integers(1, 4))):
            fx, fy = x + rng.normal(0, Hp * 0.035), y + rng.normal(0, Hp * 0.03)
            if rng.random() < 0.2:
                _sphere_dot(P, fx, fy, Hp * 0.018, "#e86a92", light, gloss=0.2)
            else:
                _flower_head(P, fx, fy, Hp * rng.uniform(0.045, 0.06), dict(sp, tilt=rng.uniform(0, 0.55)), light, rng)
    return _finish(cv, "cherry blossom", P, (bx, by), (bx, min(ys)), (bx, (by + min(ys)) / 2), paint=paint,
                   clip=clip, shadow=shadow, light=light, haze=haze, shadow_len=0.4)


def _spots(P, cx, cy, R, sp, rng):
    v = _Vec(P.x0, P.y0, P.x1, P.y1, aa=3)
    sq = 1 - 0.62 * sp.get("tilt", 0)
    for k in range(40):
        a = rng.uniform(0, 2 * math.pi)
        r = R * rng.uniform(0.12, 0.45)
        v.ellipse(cx + math.cos(a) * r, cy + math.sin(a) * r * sq, R * 0.018, R * 0.014, sp["spots"])
    pp = v.patch()
    pp.a = pp.a * (P.a[pp.y0 - P.y0:pp.y1 - P.y0, pp.x0 - P.x0:pp.x1 - P.x0] > 0.5)
    P.merge(pp)


def flower_bed(cv, region=(0.1, 0.7, 0.9, 0.95), kinds=("tulip", "daisy", "poppy"), n=40, seed=0, *, colors=None,
               height=(0.08, 0.18), foliage=True, light=LIGHT, clip=None, paint=True, haze=None):
    """Flower bed: flowers scattered in ``region`` (u0, v0, u1, v1), depth sorted, smaller towards the back,
    over a leafy base. ``kinds`` = names (cycled at random); ``colors`` = optional colour list per flower."""
    rng = _rng(seed, 111)
    S = _S(cv)
    u0, v0, u1, v1 = region
    x0, y0, x1, y1 = _clip_box(cv, u0 * cv.w - 0.1 * cv.h, v0 * cv.h - height[1] * cv.h * 1.4, u1 * cv.w + 0.1 * cv.h,
                               v1 * cv.h + 3)
    if x0 >= x1 or y0 >= y1:
        return Plant("flower bed", np.zeros((cv.h, cv.w), F32))
    full = _Patch(x0, y0, x1, y1)
    spots = sorted([(v0 + (v1 - v0) * rng.random() ** 0.8, rng.uniform(u0, u1)) for _ in range(int(n))])
    if foliage:
        vv = _Vec(x0, y0, x1, y1, aa=2)
        for k in range(int(n * 2)):
            vy = v0 + (v1 - v0) * rng.random()
            d = (vy - v0) / max(v1 - v0, 1e-6)
            _blades(vv, rng, rng.uniform(u0, u1) * cv.w, vy * cv.h, 6, (height[0] + (height[1] - height[0]) * d) * cv.h * 0.42,
                    0.02 * cv.h, LEAF_GREEN, spread=0.5, curl=0.4, S=S, tone=(0.25, 0.7))
        full.merge(vv.patch())
    for i, (vv_, uu) in enumerate(spots):
        d = (vv_ - v0) / max(v1 - v0, 1e-6)
        h = (height[0] + (height[1] - height[0]) * d) * rng.uniform(0.85, 1.15)
        k = kinds[int(rng.integers(len(kinds)))] if not isinstance(kinds, str) else kinds
        col = None if colors is None else colors[int(rng.integers(len(colors)))]
        p = flower(cv, k, at=(uu, vv_), height=h, seed=int(rng.integers(1 << 30)), color=col, light=light,
                   paint=False, leaves=1 if k not in ("tulip",) else 2)
        if p.patch is not None:
            dark = 0.82 + 0.18 * d
            p.patch.rgb = p.patch.rgb * dark
            full.merge(p.patch)
    m = _commit(cv, full, clip) if paint else full.a
    return Plant("flower bed", m, (cv.w * (u0 + u1) / 2, v1 * cv.h), (cv.w * (u0 + u1) / 2, y0),
                 (cv.w * (u0 + u1) / 2, (v0 + v1) / 2 * cv.h), (x0, y0, x1, y1), {}, full)


def meadow(cv, region=(0.0, 0.6, 1.0, 1.0), seed=0, *, flowers=("daisy", "poppy", "cornflower", "dandelion flower"),
           density=1.0, grass_height=0.07, light=LIGHT, wind=0.1, clip=None, paint=True):
    """Wild meadow: grass tufts with small flowers scattered through it, both scaled by depth."""
    rng = _rng(seed, 113)
    u0, v0, u1, v1 = region
    nb = 6                                                  # depth bands: far grass, its flowers, nearer grass ...
    full = _Patch(int(u0 * cv.w), int((v0 - grass_height * 2.6) * cv.h), int(u1 * cv.w), int(v1 * cv.h))
    spots = sorted([(v0 + (v1 - v0) * rng.random() ** 0.7, rng.uniform(u0, u1)) for _ in range(int(70 * density))])
    for bi in range(nb):
        d0, d1 = bi / nb, (bi + 1) / nb
        dm = (d0 + d1) / 2
        gh = grass_height * (0.55 + 0.45 * dm)
        bv0, bv1 = v0 + (v1 - v0) * d0, v0 + (v1 - v0) * d1
        g = grass(cv, region=(u0, bv0, u1, bv1 + 0.01), height=gh, width=0.05 * (0.6 + 0.4 * dm),
                  tufts=max(int(28 * density), 4), blades=18, seed=seed * 7 + bi, wind=wind, light=light, paint=False)
        if g.patch is not None:
            full.merge(g.patch)
        for (vv_, uu) in spots:
            if not (bv0 <= vv_ < bv1 or (bi == nb - 1 and vv_ >= bv1)):
                continue
            d = (vv_ - v0) / max(v1 - v0, 1e-6)
            h = grass_height * (0.6 + 0.8 * d) * rng.uniform(0.9, 1.5)
            kind = flowers[int(rng.integers(len(flowers)))]
            hd = FLOWERS.get(kind, {}).get("head")
            p = flower(cv, kind, at=(uu, vv_), height=h, seed=int(rng.integers(1 << 30)), light=light, paint=False,
                       leaves=0, head=None if hd is None else hd * (0.62 if kind == "poppy" else 1.0))
            if p.patch is not None:
                full.merge(p.patch)
    m = _commit(cv, full, clip) if paint else full.a
    return Plant("meadow", m, (cv.w / 2, v1 * cv.h), (cv.w / 2, v0 * cv.h), (cv.w / 2, (v0 + v1) / 2 * cv.h),
                 (full.x0, full.y0, full.x1, full.y1), {}, full)


def bouquet(cv, at=(0.5, 0.75), size=0.35, seed=0, *, kinds=("rose", "daisy", "tulip", "lavender"), n=11,
            colors=None, wrap="#e8dcc8", ribbon="#c8324a", light=LIGHT, clip=None, paint=True, shadow=False,
            _stems_only=False):
    """Hand-tied bouquet: heads arranged in a dome, stems converging at the tie point ``at``; paper wrap and
    ribbon. ``size`` = bouquet height (canvas fraction). Set ``wrap=None`` for bare stems."""
    rng = _rng(seed, 117)
    S = _S(cv)
    Hp = size * cv.h
    tx, ty = at[0] * cv.w, at[1] * cv.h
    x0, y0, x1, y1 = _clip_box(cv, tx - Hp * 0.75, ty - Hp * 1.1, tx + Hp * 0.75, ty + Hp * 0.55)
    if x0 >= x1 or y0 >= y1:
        return Plant("bouquet", np.zeros((cv.h, cv.w), F32))
    P = _Patch(x0, y0, x1, y1)
    heads = []
    for i in range(int(n)):
        a = -math.pi / 2 + (rng.random() - 0.5) * 2.2
        rr = Hp * (0.55 + 0.35 * rng.random() ** 0.5)
        heads.append((tx + math.cos(a) * rr * 0.75, ty + math.sin(a) * rr, kinds[i % len(kinds)]))
    heads.sort(key=lambda h: -h[1] + rng.normal(0, Hp * 0.05))
    v = _Vec(x0, y0, x1, y1, aa=2)
    for (hx, hy, k) in heads:                                  # stems and greenery
        v.line([(tx + rng.normal(0, Hp * 0.01), ty + Hp * 0.45), (tx, ty), (hx, hy)], max(Hp * 0.008, 0.7),
               _tone_col(LEAF_GREEN, 0.45 + rng.normal(0, 0.05)))
    for k in range(6):
        a = -math.pi / 2 + (rng.random() - 0.5) * 2.6
        _flower_leaf(v, (tx, ty), a, Hp * rng.uniform(0.4, 0.6), "rose", LEAF_GREEN, 0.4 + rng.normal(0, 0.05), S)
    st = v.patch()
    P.merge(st)
    for (hx, hy, k) in heads:
        col = None if colors is None else colors[int(rng.integers(len(colors)))]
        p = flower(cv, k, at=(hx / cv.w, hy / cv.h), height=Hp * 0.6 / cv.h, stem=False,
                   head=FLOWERS.get(k, FLOWERS["daisy"]).get("head", 0.1) * Hp * 0.75 / cv.h, color=col,
                   seed=int(rng.integers(1 << 30)), light=light, paint=False)
        if p.patch is not None:
            P.merge(p.patch)
    if wrap and not _stems_only:
        vw = _Vec(x0, y0, x1, y1, aa=3)
        w0, w1 = Hp * 0.36, Hp * 0.07
        cone = [(tx - w0, ty - Hp * 0.18), (tx - w1 * 0.5, ty + Hp * 0.5), (tx + w1 * 0.5, ty + Hp * 0.5),
                (tx + w0, ty - Hp * 0.18), (tx + w0 * 0.3, ty - Hp * 0.05), (tx - w0 * 0.3, ty - Hp * 0.08)]
        vw.poly(cone, wrap)
        vw.poly([(tx - w0 * 0.55, ty + Hp * 0.03), (tx + w0 * 0.55, ty + Hp * 0.02), (tx + w0 * 0.5, ty + Hp * 0.08),
                 (tx - w0 * 0.5, ty + Hp * 0.09)], ribbon)
        wp = vw.patch()
        sh = _shade_from_alpha(wp.a, Hp * 0.06, light, 0.4)
        wp.rgb = (wp.rgb * sh[..., None]).astype(F32)
        P.merge(wp)
    return _finish(cv, "bouquet", P, (tx, ty + Hp * 0.5), (tx, ty - Hp), (tx, ty - Hp * 0.6), paint=paint,
                   clip=clip, light=light, shadow=shadow, shadow_len=0.3)


# ============================================================================
# pots and potted plants
# ============================================================================
POTS = {
    "classic": lambda t: 1.0 - 0.24 * t,                     # terracotta flowerpot (with a rim)
    "cylinder": lambda t: 0.86 + 0.0 * t,
    "bowl": lambda t: np.sqrt(np.clip(1 - t ** 2, 0, 1)) * 0.98 + 0.02,
    "urn": lambda t: 0.6 + 0.4 * np.sin(np.pi * (0.25 + 0.65 * t)) - 0.2 * t,
    "square": lambda t: 0.9 - 0.06 * t,
    "hanging": lambda t: np.sqrt(np.clip(1 - t ** 2, 0, 1)) * 0.98 + 0.02,
    "vase": lambda t: 0.45 + 0.4 * np.clip(np.sin(np.pi * np.clip(t * 0.95 + 0.1, 0, 1)), 0, 1) ** 1.2 - 0.1 * (1 - t) ** 8,
}


def _pot(P, cx, by, width, height, shape="classic", color="#b5653a", light=LIGHT, S=1.0, soil=True, glaze=None,
         band=None):
    """Lathe-shaded pot into patch P: mouth at by - height, bottom at by. ``width`` = mouth width px."""
    if shape not in POTS:
        raise _unknown_name("pot shape", shape, POTS)
    prof = POTS[shape]
    top = by - height
    X0, Y0, X1, Y1 = int(cx - width * 0.62), int(top - width * 0.1), int(cx + width * 0.62) + 2, int(by + 2)
    X0, Y0, X1, Y1 = max(X0, P.x0), max(Y0, P.y0), min(X1, P.x1), min(Y1, P.y1)
    if X0 >= X1 or Y0 >= Y1:
        return
    sy, sx = np.mgrid[Y0:Y1, X0:X1].astype(F32)
    t = np.clip((sy - top) / height, 0, 1)
    hw = width * 0.5 * prof(t)
    e = 0.1 * width * 0.5                         # ellipse depth of the rim
    nx = (sx - cx) / np.maximum(hw, 1e-3)
    inside = clamp01((hw - np.abs(sx - cx)) + 0.5)
    bottom = clamp01((by - e * np.sqrt(np.clip(1 - nx * nx, 0, 1)) * 0.6 - sy) + 0.5) if shape != "hanging" else \
        clamp01((by - sy) + 0.5)
    topcut = clamp01((sy - (top - e * np.sqrt(np.clip(1 - ((sx - cx) / (width * 0.5)) ** 2, 0, 1)))) + 0.5)
    a = inside * bottom * topcut
    L3 = _l3(light, 0.5)
    nxc = np.clip(nx, -1, 1)
    nz = np.sqrt(np.clip(1 - nxc * nxc, 0, 1))
    d = np.clip(nxc * L3[0] + nz * L3[2], 0, 1)
    if shape == "square":
        d = np.where(nx < -0.2, 0.9, np.where(nx < 0.6, 0.62, 0.32)) * 1.0
    c = rgb(color)
    col = c[None, None] * (0.3 + 0.85 * d)[..., None]
    if glaze is not None or shape in ("vase", "urn", "cylinder"):
        spec = np.clip(2 * d * nz - L3[2], 0, 1) ** 30 * 0.5
        col = col + spec[..., None]
    if shape == "classic":                         # thick rim band
        rim = (t < 0.2)
        col = np.where(rim[..., None], col * 1.08, col * (0.92 + 0.08 * t[..., None]))
        col = col * (1 - 0.35 * np.exp(-((t - 0.2) * height / (1.2 * S + 0.5)) ** 2))[..., None]
    if band is not None:
        bm = (np.abs(t - 0.55) < 0.07)
        col = np.where(bm[..., None], rgb(band)[None, None] * (0.3 + 0.85 * d)[..., None], col)
    if shape == "hanging":
        pass
    P.over(col.astype(F32), a.astype(F32), X0, Y0)
    if soil:
        sd = ((sx - cx) / (width * 0.46)) ** 2 + ((sy - top) / (e * 0.9)) ** 2
        sa = clamp01((1 - sd) * e * 0.9 + 0.5) * (sy > top - e)
        f1, f2, _, _, cid = _worley(sx, sy, max(1.5 * S, 0.8), 3)
        scol = rgb("#3a2a1e")[None, None] * (0.6 + 0.5 * cid[..., None]) * (0.6 + 0.4 * clamp01((sy - top + e) / (2 * e)))[..., None]
        P.over(scol.astype(F32), sa.astype(F32), X0, Y0)


def _heart_leaf(v, base, ang, L, ramp, tone, holes=False, rng=None, slits=0):
    """Heart-shaped leaf (pothos, monstera with slits / holes) along direction ang."""
    dx, dy = math.cos(ang), math.sin(ang)
    px, py = -dy, dx
    pts_l, pts_r = [], []
    for u in np.linspace(0, 1, 30):
        w = L * 0.44 * math.sin(math.pi * min(1.0, u * 0.86 + 0.14)) ** 0.75 * (1 - 0.15 * u)
        if u < 0.14:
            w = L * 0.44 * (0.65 + u * 2.4)
        cxp, cyp = base[0] + dx * L * (u - 0.1), base[1] + dy * L * (u - 0.1) + L * 0.12 * u * u
        pts_l.append((cxp + px * w, cyp + py * w))
        pts_r.append((cxp - px * w, cyp - py * w))
    v.poly([tuple(base)] + pts_l + [pts_l[-1]], _tone_col(ramp, tone + 0.08))
    v.poly([tuple(base)] + pts_r + [pts_r[-1]], _tone_col(ramp, tone - 0.06))
    mid = [(base[0] + dx * L * (u - 0.1), base[1] + dy * L * (u - 0.1) + L * 0.12 * u * u) for u in np.linspace(0.1, 0.95, 12)]
    v.line(mid, max(L * 0.018, 0.6), _tone_col(ramp, tone + 0.22))
    if slits and rng is not None:
        for k in range(slits):
            u = 0.25 + 0.65 * k / max(slits - 1, 1)
            for sd in (-1, 1):
                cxp, cyp = base[0] + dx * L * (u - 0.1), base[1] + dy * L * (u - 0.1) + L * 0.12 * u * u
                w = L * 0.46 * math.sin(math.pi * min(1.0, u * 0.86 + 0.14)) ** 0.75
                s0 = (cxp + px * sd * w * 0.32, cyp + py * sd * w * 0.32)
                s1 = (cxp + px * sd * w * 1.1 + dx * L * 0.1, cyp + py * sd * w * 1.1 + dy * L * 0.1)
                ex, ey = s1[0] - s0[0], s1[1] - s0[1]               # wedge: hairline at the rib, wide at the margin
                el = math.hypot(ex, ey) or 1.0
                nx_, ny_ = -ey / el, ex / el
                ww = max(L * 0.05, 1.2)
                v.poly([s0, (s1[0] + nx_ * ww, s1[1] + ny_ * ww), (s1[0] + ex / el * ww * 1.5, s1[1] + ey / el * ww * 1.5),
                        (s1[0] - nx_ * ww, s1[1] - ny_ * ww)], (0, 0, 0), erase=True)
                if holes and rng.random() < 0.6:
                    hx, hy = cxp + px * sd * w * 0.2 - dx * L * 0.05, cyp + py * sd * w * 0.2 - dy * L * 0.05
                    v.ellipse(hx, hy, L * 0.032, L * 0.05, (0, 0, 0), erase=True)


def potted(cv, kind="monstera", at=(0.5, 0.95), height=0.4, seed=0, *, pot="classic", pot_color=None, light=LIGHT,
           shadow=True, clip=None, paint=True, haze=None):
    """Potted plant: kind = monstera, snake plant, succulent, ficus, pothos (hanging), fern, cactus, aloe,
    flowering (geranium-like). ``pot`` = classic, cylinder, bowl, urn, square, hanging; ``height`` includes the pot."""
    rng = _rng(seed, 121)
    S = _S(cv)
    Hp = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    kind = kind.lower().replace("_", " ")
    if kind not in POTTED:
        raise _unknown("potted plant", kind, POTTED, "Try flora.names('potted')")
    defaults = {"monstera": ("classic", "#c9c2b6"), "snake plant": ("cylinder", "#2e3036"),
                "succulent": ("bowl", "#b5653a"), "ficus": ("cylinder", "#e8e2d6"), "pothos": ("hanging", "#d8cfc0"),
                "fern": ("urn", "#8a8f96"), "cactus": ("classic", "#b5653a"), "aloe": ("square", "#4a6a8a"),
                "flowering": ("classic", "#b5653a")}
    dpot, dcol = defaults.get(kind, ("classic", "#b5653a"))
    pot = pot if pot != "classic" or kind not in defaults else dpot
    pot_color = pot_color or dcol
    ph = Hp * (0.3 if kind != "pothos" else 0.28)
    pw = Hp * (0.36 if kind not in ("succulent",) else 0.6)
    if kind == "succulent":
        ph = Hp * 0.35
    pot_by = by
    if kind == "pothos":                                       # hanging planter: hanger on top, vines trail below
        ph = Hp * 0.2
        pw = Hp * 0.34
        pot_by = by - Hp * 0.42
    mouth = pot_by - ph
    x0, y0, x1, y1 = _clip_box(cv, bx - Hp * 0.8, by - Hp * 1.1, bx + Hp * 0.8, by + Hp * 0.05)
    if x0 >= x1 or y0 >= y1:
        return _finish(cv, kind, None, (bx, by), (bx, by - Hp), (bx, by - Hp / 2))
    P = _Patch(x0, y0, x1, y1)
    ramp = LEAF_GREEN
    v = _Vec(x0, y0, x1, y1, aa=3)
    ph_plant = Hp - ph
    if kind == "monstera":
        for k in range(7):
            a = -math.pi / 2 + (k / 6 - 0.5) * 2.2 + rng.normal(0, 0.12)
            L = ph_plant * rng.uniform(0.55, 0.8)
            tip = (bx + math.cos(a) * L * 0.6, mouth + math.sin(a) * L * 0.8)
            v.line([(bx, mouth), (bx + math.cos(a) * L * 0.2, mouth - L * 0.35), tip], max(Hp * 0.008, 0.7),
                   _tone_col(ramp, 0.45))
            leaf_ang = a + rng.normal(0, 0.2) + (0.6 if math.cos(a) > 0 else -0.6) * 0.5
            _heart_leaf(v, tip, leaf_ang + math.pi * 0.0, ph_plant * rng.uniform(0.38, 0.5),
                        ["#0c2410", "#14401c", "#22622a", "#3e8a40", "#7ab86e"], 0.45 + rng.normal(0, 0.06),
                        holes=True, rng=rng, slits=5)
    elif kind == "snake plant":
        for k in range(9):
            a = -math.pi / 2 + (k / 8 - 0.5) * 0.7 + rng.normal(0, 0.05)
            L = ph_plant * rng.uniform(0.6, 1.0)
            b0 = (bx + (k / 8 - 0.5) * pw * 0.6, mouth + 2)
            spine = _bez(b0, (b0[0] + math.cos(a) * L * 0.5, b0[1] + math.sin(a) * L * 0.5),
                         (b0[0] + math.cos(a) * L, b0[1] + math.sin(a) * L), 14)
            uu = np.linspace(0, 1, 14)
            wv = Hp * 0.035 * np.sin(np.pi * np.clip(uu * 0.9 + 0.1, 0, 1)) ** 0.35 * (1 - uu * 0.4)
            left, right, nrm = _half_blades(spine, wv)
            v.poly(_ribbon(spine, wv * 1.12), "#c8c060")
            v.poly(left, _tone_col(["#0e2412", "#1c4022", "#2e5e30", "#4a7e44", "#7aa66a"], 0.55))
            v.poly(right, _tone_col(["#0e2412", "#1c4022", "#2e5e30", "#4a7e44", "#7aa66a"], 0.35))
            for j in range(1, 13):                           # horizontal bands
                if j % 2:
                    q0 = spine[j] + nrm[j] * wv[j] * 0.9
                    q1 = spine[j] - nrm[j] * wv[j] * 0.9
                    v.line([tuple(q0), tuple((spine[j] + spine[j + 1]) / 2), tuple(q1)], max(Hp * 0.006, 0.6), "#a6be8a")
    elif kind in ("succulent", "aloe"):
        pp = P
        if kind == "succulent":
            for k in range(3):
                cxp = bx + (k - 1) * pw * 0.28
                r = ph_plant * (0.42 if k == 1 else 0.3)
                _flower_head(P, cxp, mouth - r * 0.25, r, dict(petals=11, layers=4, shape="pointed",
                                                                color=["#6f9a8e", "#7ea898", "#8eb6a2", "#9ec2aa"],
                                                                tip="#d08a9a", width=0.32, cup=0.75, tilt=0.55,
                                                                veins=0.0), light, rng)
        else:
            for k in range(12):
                a = -math.pi / 2 + (k / 11 - 0.5) * 2.3
                L = ph_plant * rng.uniform(0.6, 0.95)
                b0 = (bx, mouth)
                spine = _bez(b0, (b0[0] + math.cos(a) * L * 0.5, b0[1] + math.sin(a) * L * 0.6),
                             (b0[0] + math.cos(a) * L, b0[1] + math.sin(a) * L + L * 0.1 * abs(math.cos(a))), 12)
                wv = Hp * 0.035 * (1 - np.linspace(0, 1, 12)) ** 0.8
                left, right, nrm = _half_blades(spine, wv)
                v.poly(left, _tone_col(["#2a3e22", "#3e5a34", "#5e7e4e", "#86a272", "#b6c8a0"], 0.6))
                v.poly(right, _tone_col(["#2a3e22", "#3e5a34", "#5e7e4e", "#86a272", "#b6c8a0"], 0.4))
                for j in range(1, 11):
                    for sd in (1, -1):
                        q = spine[j] + nrm[j] * wv[j] * sd
                        v.poly([tuple(q), tuple(q + nrm[j] * sd * Hp * 0.008 + (spine[j + 1] - spine[j]) * 0.5),
                                tuple(spine[j] + nrm[j] * wv[j] * sd * 0.8 + (spine[j + 1] - spine[j]) * 0.6)], "#c8c49a")
    elif kind == "ficus":
        p = tree(cv, "maple", at=(bx / cv.w, mouth / cv.h), height=ph_plant / cv.h, seed=seed, light=light,
                 paint=False, fullness=1.1, palette=["#0e2612", "#1a4420", "#2c6a30", "#4e9046", "#8ec070"])
        if p.patch is not None:
            P.merge(p.patch)
    elif kind == "pothos":
        for k in range(9):
            a = rng.uniform(-math.pi, 0)
            sx0 = bx + rng.uniform(-0.45, 0.45) * pw
            L = (by - mouth) * rng.uniform(0.45, 1.0)
            sd = 1 if sx0 > bx else -1
            pts = [(sx0, mouth)]
            for j in range(1, 16):
                t = j / 15
                pts.append((sx0 + sd * pw * 0.35 * min(t * 3, 1) + math.sin(t * 6 + k) * Hp * 0.02, mouth - Hp * 0.05 + t * L))
            v.line(pts, max(Hp * 0.005, 0.6), _tone_col(ramp, 0.4))
            for j in range(1, 16, 2):
                ang = math.pi / 2 + rng.normal(0, 0.5) + (0.9 if j % 4 == 1 else -0.9)
                _heart_leaf(v, pts[j], ang, Hp * rng.uniform(0.08, 0.12), ["#163010", "#2a5a1c", "#4a8a2a", "#7ab64a",
                                                                         "#c2e07a"], 0.5 + rng.normal(0, 0.08))
        for k in range(8):
            a = -math.pi / 2 + (k / 7 - 0.5) * 2.0
            _heart_leaf(v, (bx + math.cos(a) * pw * 0.2, mouth), a, Hp * 0.12,
                        ["#163010", "#2a5a1c", "#4a8a2a", "#7ab64a", "#c2e07a"], 0.55 + rng.normal(0, 0.06))
    elif kind == "fern":
        p = fern(cv, at=(bx / cv.w, (mouth + 2) / cv.h), height=ph_plant * 0.7 / cv.h, seed=seed, light=light,
                 paint=False, fronds=12)
        if p.patch is not None:
            P.merge(p.patch)
    elif kind == "cactus":
        p = saguaro(cv, at=(bx / cv.w, (mouth + ph_plant * 0.05) / cv.h), height=ph_plant * 0.9 / cv.h, seed=seed,
                    light=light, paint=False, arms=2)
        if p.patch is not None:
            P.merge(p.patch)
    else:                                                      # flowering: leafy mound with blossoms
        its = []
        for k in range(16):
            a = rng.uniform(math.pi, 2 * math.pi)
            rr = rng.random() ** 0.5
            r = ph_plant * rng.uniform(0.14, 0.2)
            its.append([bx + math.cos(a) * rr * pw * 0.6, mouth - ph_plant * 0.3 + math.sin(a) * rr * ph_plant * 0.4,
                        r, r * 0.85, 0, 0, rng.random()])
        bp, _ = _leafy(cv, _bulge(np.array(its), rng), light, "bush", max(ph_plant * 0.03, 0.9), seed, holes=0.0)
        if bp is not None:
            for k in range(30):
                it = its[int(rng.integers(len(its)))]
                _blossom(bp, it[0] + rng.normal(0, it[2] * 0.5), it[1] + rng.normal(0, it[3] * 0.5),
                         max(ph_plant * 0.04, 1.2), "#e8344a", light, rng)
            P.merge(bp)
    vp = v.patch()
    if vp.a.any():
        sh = _shade_from_alpha(vp.a, Hp * 0.012, light, 0.3)
        vp.rgb = (vp.rgb * sh[..., None]).astype(F32)
        if kind == "pothos":
            back = _Patch(vp.x0, vp.y0, vp.x1, vp.y1)
            P.merge(vp)
        else:
            P.merge(vp)
    potP = _Patch(x0, y0, x1, y1)
    if kind == "pothos":
        hv = _Vec(x0, y0, x1, y1, aa=3)
        for sd in (-1, -0.3, 0.3, 1):
            hv.line([(bx, by - Hp), (bx + sd * pw * 0.45, mouth - ph * 0.05)], max(Hp * 0.004, 0.5), "#8a7a5e")
        hv.ellipse(bx, by - Hp, Hp * 0.012, Hp * 0.012, "#6a5a40")
        potP.merge(hv.patch())
    _pot(potP, bx, pot_by, pw, ph, pot, pot_color, light, S, soil=kind not in ("pothos",))
    if kind == "pothos":
        out = _Patch(x0, y0, x1, y1)
        out.merge(potP)
        # trailing vines hang in front of the pot
        out.merge(P)
        P = out
    else:
        potP.merge(P) if False else None
        out = _Patch(x0, y0, x1, y1)
        out.merge(potP)
        Y = np.arange(P.y0, P.y1, dtype=F32)[:, None]
        X = np.arange(P.x0, P.x1, dtype=F32)[None, :]
        e = pw * 0.05
        front = (Y > mouth + e * 0.0) & (np.abs(X - bx) < pw * 0.5)
        P.a = np.where(front, 0, P.a).astype(F32)            # stems vanish into the soil
        out.merge(P)
        P = out
    return _finish(cv, kind, P, (bx, by), (bx, by - Hp), (bx, mouth - ph_plant * 0.5) if kind != "pothos" else (bx, mouth), paint=paint, clip=clip,
                   shadow=shadow, light=light, haze=haze, extra=dict(pot_top=mouth))


POTTED = ("monstera", "snake plant", "succulent", "ficus", "pothos", "fern", "cactus", "aloe", "flowering")


def vase(cv, at=(0.5, 0.92), height=0.6, seed=0, *, kinds=("rose", "peony", "lavender", "daisy"), n=11, shape="vase",
         color="#3e6a8a", colors=None, light=LIGHT, shadow=False, clip=None, paint=True):
    """Vase arrangement: a glazed vase (``shape`` from POTS: vase, urn, cylinder, bowl) with an arrangement of
    flowers rising from its mouth. ``height`` = whole arrangement (canvas fraction)."""
    if shape not in POTS:
        raise _unknown_name("vase shape", shape, POTS)
    S = _S(cv)
    Hp = height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    vh = Hp * 0.42
    vw = vh * 0.55
    mouth = by - vh
    b = bouquet(cv, at=(at[0], (mouth + vh * 0.05) / cv.h), size=Hp * 0.6 / cv.h, seed=seed, kinds=kinds, n=n,
                colors=colors, wrap=None, light=light, paint=False)
    x0, y0, x1, y1 = _clip_box(cv, bx - Hp * 0.6, by - Hp * 1.05, bx + Hp * 0.6, by + 3)
    P = _Patch(x0, y0, x1, y1)
    if b.patch is not None:
        P.merge(b.patch)
    vp = _Patch(x0, y0, x1, y1)
    _pot(vp, bx, by, vw, vh, shape, color, light, S, soil=False, glaze=True)
    P.merge(vp)
    return _finish(cv, "vase", P, (bx, by), (bx, by - Hp), (bx, mouth - Hp * 0.3), paint=paint, clip=clip,
                   shadow=shadow, light=light, shadow_len=0.4)


# ============================================================================
# leaves: single leaf shapes with veins, scatters, piles, a framing branch
# ============================================================================
LEAVES = ("ovate", "oak", "maple", "ginkgo", "monstera", "palm", "birch", "willow", "heart")
AUTUMN = ["#8a1e12", "#b8341a", "#d8601e", "#e8902a", "#f0b83a", "#c8a03a", "#9a6a2a"]


def _leaf_field(kind, u, v):
    """Leaf outline in local coords: u along the midrib (0 stalk → 1 tip), v across, both in leaf lengths.
    Returns signed distance-ish (negative inside, in leaf lengths) and the vein strength 0~1."""
    au = np.clip(u, 0, 1)
    av = np.abs(v)
    if kind in ("maple", "ginkgo"):
        r = np.hypot(u, v)
        th = np.arctan2(v, u)                      # 0 = towards the tip
        if kind == "maple":
            c0 = 0.5
            r = np.hypot(u - c0, v)
            th = np.arctan2(v, u - c0)
            at_ = np.abs(th)
            lobe = lambda c, w_: np.maximum(1 - np.abs(at_ - c) / w_, 0) ** 1.6
            R = 0.2 + 0.3 * lobe(0.0, 0.5) + 0.26 * lobe(1.1, 0.5) + 0.12 * lobe(2.0, 0.45)
            R = R - 0.018 * np.abs(np.sin(th * 15)) * (R > 0.22)
            d = r - R
            vein = sum(np.exp(-((th - a0) * np.maximum(r, 0.02) / 0.006) ** 2) * (r < R)
                       for a0 in (0.0, 1.1, -1.1, 2.0, -2.0))
            d = np.minimum(d, np.maximum(np.maximum(av - 0.012, u - c0), -u))
            return d, np.clip(vein, 0, 1)
        else:
            R = 0.95 - 0.25 * np.exp(-(th / 0.07) ** 2) + 0.03 * np.sin(th * 14)
            inside_ang = np.abs(th) < 1.0
            d = np.where(inside_ang, r - R * 0.8, 1.0)
            d = np.maximum(d, (np.abs(th) - 1.0) * r)
            stalk = np.maximum(np.abs(v) - 0.012, -u) if True else 0
            d = np.minimum(d, np.maximum(np.maximum(av - 0.012, u - 0.05), -u - 0.25))
            vein = (0.5 + 0.5 * np.cos(th * 46)) ** 8 * (r > 0.05)
        return d, vein
    if kind == "oak":
        w = 0.24 * np.sin(np.pi * au ** 0.8) ** 0.8 * (0.55 + 0.45 * np.abs(np.sin(au * math.pi * 4.5)) ** 0.7)
    elif kind == "monstera":
        w = 0.48 * np.sin(np.pi * np.clip(au, 0, 1) ** 0.62) ** 0.7
    elif kind == "heart":
        w = 0.5 * np.sin(np.pi * np.clip(au, 0, 1) ** 0.6) ** 0.8
    elif kind == "birch":
        w = 0.33 * np.sin(np.pi * au ** 0.7) ** 0.9 * (1 + 0.05 * np.sin(au * 60))
    elif kind == "willow":
        w = 0.09 * np.sin(np.pi * au) ** 0.8
    elif kind == "palm":
        w = 0.06 * np.sin(np.pi * au) ** 0.5
    else:                                        # ovate
        w = 0.28 * np.sin(np.pi * au ** 0.75) ** 0.85
    d = np.maximum(av - w, np.maximum(-u, u - 1))
    # petiole
    d = np.minimum(d, np.maximum(np.maximum(av - 0.012, u - 0.02), -u - 0.18))
    sec = (u - 0.04 - av * 1.1) * (9 if kind != "monstera" else 7)
    vein = (0.5 + 0.5 * np.cos(sec * 2 * math.pi)) ** 10 * smoothstep(0.02, 0.06, av) * (au < 0.97)
    if kind in ("heart", "monstera"):                      # cordate base: a notch where the stalk joins
        d = np.maximum(d, 0.1 * (1 - np.clip(av / 0.28, 0, 1)) - u)
    if kind == "monstera":                                 # slits cut in from the margin, holes beside the midrib
        ph = 0.5 + 0.5 * np.cos((u - av * 0.8) * 5 * 2 * math.pi)
        thr = 1 - (0.02 + 0.075 * np.clip((av - w * 0.3) / (w * 0.7 + 1e-3), 0, 1))
        gate = np.minimum((av - w * 0.32) * 9, np.minimum((au - 0.12) * 9, (0.93 - au) * 9))
        c1 = np.minimum((ph - thr) * 12, gate)
        ph2 = 0.5 + 0.5 * np.cos((u - av * 0.8) * 5 * 2 * math.pi + math.pi)
        c2 = np.minimum((ph2 - 0.93) * 12, np.minimum((0.045 - np.abs(av - w * 0.2)) * 12, (au - 0.2) * 9))
        d = np.maximum(d, np.maximum(c1, c2) * 0.15)
    mid = np.exp(-(av / 0.008) ** 2) * (au < 1)
    return d, np.maximum(vein * 0.8, mid)


def _leaf_patch(cx, cy, L, angle, kind, color, light, flip=1.0, tip=None, S=1.0, vein_color=None):
    """Numpy leaf of length L px at canvas (cx, cy) = stalk end, pointing at ``angle``. Returns a _Patch."""
    pad = L * 0.6 + 3
    p = _Patch(int(cx - pad - L * 0.5), int(cy - pad - L * 0.5), int(cx + pad + L * 0.5) + 2,
               int(cy + pad + L * 0.5) + 2)
    X, Y = p.grid()
    u, v = _rot(X - cx, Y - cy, angle)
    u, v = u / L, v / (L * max(abs(flip), 0.15))
    d, vein = _leaf_field(kind, u, v)
    a = clamp01(0.5 - d * L * max(abs(flip), 0.15))
    if not a.any():
        return None
    lx, ly = _l2(light)
    side = np.sign(v) * np.sign(flip)
    fold = 0.06 * side * (-lx * math.sin(angle) + ly * math.cos(angle)) * 4
    base = rgb(color)
    col = base[None, None] * np.clip(0.82 + fold - 0.18 * np.abs(v) * 2 + 0.12 * u, 0.3, 1.3)[..., None]
    if tip is not None:
        col = _blend_rgb(col, rgb(tip) * np.clip(0.9 + fold, 0.3, 1.3)[..., None], smoothstep(0.3, 1.0, u + 0.3 * np.abs(v)))
    vc = rgb(vein_color) if vein_color is not None else np.clip(base * 1.35 + 0.08, 0, 1)
    col = _blend_rgb(col, vc[None, None] * np.ones_like(col), vein * 0.55 * smoothstep(0.0, 6.0 * S, L))
    if flip < 0:
        col = col * 0.82
    p.rgb, p.a = col.astype(F32), a.astype(F32)
    return p


def leaf(cv, kind="maple", at=(0.5, 0.5), size=0.2, angle=-90, seed=0, *, color=None, tip=None, flip=1.0,
         light=LIGHT, shadow=False, clip=None, paint=True):
    """Single leaf with veins: kind = ovate, oak (lobed), maple (palmate, 5 lobes), ginkgo (fan), monstera (split),
    palm (pinnate frond), birch, willow, heart. ``at`` = leaf centre, ``size`` = length (canvas fraction),
    ``angle`` = direction of the tip in degrees (−90 = up), ``flip`` = −1~1 (foreshortening; < 0 shows the back)."""
    if kind not in LEAVES:
        raise _unknown("leaf kind", kind, LEAVES)
    rng = _rng(seed, 131)
    S = _S(cv)
    L = size * cv.h
    ang = math.radians(angle)
    cx, cy = at[0] * cv.w - math.cos(ang) * L * 0.5, at[1] * cv.h - math.sin(ang) * L * 0.5
    col = color or {"maple": "#d2401e", "ginkgo": "#e8b82a", "oak": "#8a6a2a", "monstera": "#1e6a34",
                    "palm": "#3e7a2c"}.get(kind, "#4f8a34")
    if kind == "palm":
        v = _Vec(*_clip_box(cv, cx - L * 1.2, cy - L * 1.2, cx + L * 1.2, cy + L * 1.2), aa=3)
        _frond(v, (cx, cy), ang, L, 0.15, [_hex_mix(col, "#000", 0.6), _hex_mix(col, "#000", 0.35), col,
                                            _hex_mix(col, "#fff", 0.25), _hex_mix(col, "#fff", 0.5)], 0.55, rng, S,
               leaflet=0.3, light=light)
        P = v.patch()
    else:
        P = _leaf_patch(cx, cy, L, ang, kind, col, light, flip, tip, S)
    if P is None:
        return Plant(kind, np.zeros((cv.h, cv.w), F32))
    Pc = _Patch(*_clip_box(cv, P.x0, P.y0, P.x1, P.y1))
    Pc.merge(P)
    return _finish(cv, kind + " leaf", Pc, (cx, cy), (cx + math.cos(ang) * L, cy + math.sin(ang) * L),
                   (at[0] * cv.w, at[1] * cv.h), paint=paint, clip=clip, shadow=shadow, light=light)


def falling_leaves(cv, n=40, region=(0, 0, 1, 1), seed=0, *, kinds=("maple", "oak", "birch"), colors=AUTUMN,
                   size=(0.025, 0.05), light=LIGHT, clip=None, paint=True):
    """Leaves tumbling through the air: random kinds, colours, spins and foreshortening (some show their back)."""
    rng = _rng(seed, 133)
    S = _S(cv)
    u0, v0, u1, v1 = region
    full = _Patch(*_clip_box(cv, u0 * cv.w, v0 * cv.h, u1 * cv.w, v1 * cv.h))
    for i in range(int(n)):
        L = rng.uniform(*size) * cv.h
        cx, cy = rng.uniform(u0, u1) * cv.w, rng.uniform(v0, v1) * cv.h
        c = colors[int(rng.integers(len(colors)))]
        p = _leaf_patch(cx, cy, L, rng.uniform(0, 2 * math.pi), kinds[int(rng.integers(len(kinds)))], c, light,
                        flip=rng.uniform(-1, 1), tip=colors[int(rng.integers(len(colors)))], S=S)
        if p is not None:
            full.merge(p)
    m = _commit(cv, full, clip) if paint else full.a
    return Plant("falling leaves", m, (cv.w / 2, v1 * cv.h), (cv.w / 2, v0 * cv.h), (cv.w / 2, cv.h * (v0 + v1) / 2),
                 (full.x0, full.y0, full.x1, full.y1), {}, full)


def leaf_pile(cv, at=(0.5, 0.95), width=0.3, height=0.1, seed=0, *, n=None, kinds=("maple", "oak", "birch"),
              colors=AUTUMN, size=0.035, light=LIGHT, shadow=False, clip=None, paint=True):
    """Pile of fallen leaves: a mound of overlapping leaves, darker deep inside and at the bottom."""
    rng = _rng(seed, 135)
    S = _S(cv)
    Wp, Hp = width * cv.h, height * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    n = n or int(60 + 900 * width * height / 0.03)
    P = _Patch(*_clip_box(cv, bx - Wp * 0.6 - size * cv.h, by - Hp * 1.1 - size * cv.h, bx + Wp * 0.6 + size * cv.h,
                          by + size * cv.h * 0.3))
    pts = []
    for i in range(n):
        x = rng.uniform(-1, 1)
        top = (1 - x * x) ** 0.8
        y = rng.random() ** 0.7 * top
        pts.append((x, y))
    pts.sort(key=lambda p: p[1] - abs(p[0]) * 0.1)            # inner / low leaves first
    for (x, y) in pts:
        L = size * cv.h * rng.uniform(0.7, 1.25)
        cx, cy = bx + x * Wp * 0.5, by - y * Hp
        c = rgb(colors[int(rng.integers(len(colors)))]) * (0.55 + 0.45 * y)
        p = _leaf_patch(cx, cy, L, rng.uniform(0, 2 * math.pi), kinds[int(rng.integers(len(kinds)))], c, light,
                        flip=rng.uniform(0.25, 1.0) * rng.choice([1, 1, 1, -1]), S=S)
        if p is not None:
            P.merge(p)
    return _finish(cv, "leaf pile", P, (bx, by), (bx, by - Hp), (bx, by - Hp / 2), paint=paint, clip=clip,
                   shadow=shadow, light=light, shadow_len=0.7)


def branch_frame(cv, corner="tl", size=0.45, seed=0, *, kind="leafy", light=LIGHT, dark=0.0, blur=0.0, clip=None,
                 paint=True, color=None):
    """Foreground framing branch entering from a corner ("tl", "tr", "bl", "br", "t"). kind = "leafy",
    "autumn", "blossom" (cherry), "pine", "bare". ``dark`` 0~1 pushes it towards a silhouette, ``blur`` px
    (1080p) softens it like an out-of-focus foreground."""
    corner = _choice("branch_frame", "corner", corner, ("tl", "tr", "bl", "br", "t"))
    kind = _choice("branch_frame", "kind", kind, ("leafy", "autumn", "blossom", "pine", "bare"))
    rng = _rng(seed, 137)
    S = _S(cv)
    W, H = cv.w, cv.h
    Lb = size * max(W, H)
    sx = {"tl": -0.02, "bl": -0.02, "tr": 1.02, "br": 1.02, "t": 0.3}[corner] * W
    sy = {"tl": 0.08, "tr": 0.08, "bl": 0.75, "br": 0.75, "t": -0.02}[corner] * H
    dirx = 1 if corner in ("tl", "bl", "t") else -1
    diry = 0.18 if corner in ("tl", "tr", "t") else -0.25
    sk = _Skel()
    ang = math.atan2(diry, dirx)
    sp = dict(nseg=6, wiggle=10, up=-0.05 if corner in ("tl", "tr") else 0.1, taper=0.55, depth=3, kids=(2, 3),
              spread=40, spread_jitter=12, len_ratio=0.55, side_twigs=0.9, floor=-1e9)
    _grow(sk, rng, sp, sx, sy, ang, Lb, 0.024 * Lb, 1, 0)
    P = _Patch(0, 0, W, H)
    barkc = BARKS["cherry"] if kind == "blossom" else BARKS["oak"]
    c, a, _ = _render_bark(P, sk.segs, light, barkc, kind="smooth" if kind == "blossom" else "furrow", S=S, seed=seed)
    P.rgb, P.a = c, a
    pts = [(s[2], s[3], s[8]) for s in sk.segs if s[8] >= 1] + [(t[0], t[1], 3) for t in sk.tips]
    if kind in ("leafy", "autumn"):
        cols = AUTUMN if kind == "autumn" else ["#2a5a22", "#3a7a2c", "#4e8a32", "#2e6a2a", "#5a9a3a"]
        for (x, y, dep) in pts:
            for j in range(int(rng.integers(2, 5))):
                L = Lb * rng.uniform(0.06, 0.1)
                p = _leaf_patch(x, y, L, rng.uniform(0, 2 * math.pi) if kind == "autumn" else rng.uniform(0.3, 2.8) * (1 if diry < 0 else 1),
                                "ovate" if kind == "leafy" else rng.choice(["maple", "oak"]),
                                color or cols[int(rng.integers(len(cols)))], light, flip=rng.uniform(0.4, 1.0), S=S)
                if p is not None:
                    P.merge(p)
    elif kind == "blossom":
        for (x, y, dep) in pts:
            for j in range(int(rng.integers(3, 7))):
                fx, fy = x + rng.normal(0, Lb * 0.025), y + rng.normal(0, Lb * 0.02)
                _flower_head(P, fx, fy, Lb * rng.uniform(0.024, 0.036), dict(FLOWERS["cherry blossom"],
                                                                         color=color or "#f6bcd0", tilt=rng.uniform(0, 0.5)),
                             light, rng)
    elif kind == "pine":
        v = _Vec(0, 0, W, H, aa=2)
        for s in sk.segs:
            if s[8] < 1:
                continue
            for t in np.linspace(0, 1, 5):
                x, y = s[0] + (s[2] - s[0]) * t, s[1] + (s[3] - s[1]) * t
                for k in range(10):
                    a2 = rng.uniform(0, 2 * math.pi)
                    L = Lb * rng.uniform(0.025, 0.04)
                    v.line([(x, y), (x + math.cos(a2) * L, y + math.sin(a2) * L + L * 0.2)], max(1.0 * S, 0.6),
                           _tone_col(PALETTES["pine"], rng.uniform(0.3, 0.8)))
        P.merge(v.patch())
    if dark > 0:
        P.rgb = _blend_rgb(P.rgb, rgb("#0c120e"), dark)
    if blur > 0:
        b = blur * S
        pre = gaussian_blur(P.rgb * P.a[..., None], b)
        P.a = gaussian_blur(P.a, b)
        P.rgb = (pre / np.maximum(P.a, 1e-3)[..., None]).astype(F32)
    return _finish(cv, "branch frame", P, (sx, sy), (sx, sy), (sx + dirx * Lb * 0.5, sy), paint=paint, clip=clip)


def perch(cv, a=(0.2, 0.5), b=(0.8, 0.45), seed=0, *, thickness=0.016, kind="leafy", leaves=True, twigs=3,
          droop=0.015, bark="oak", color=None, light=LIGHT, clip=None, paint=True):
    """A plain branch from point ``a`` to point ``b`` (canvas fractions) that a bird can sit on: a gently bowed,
    tapering limb with side twigs and leaves hanging from the ends and the underside, so the top of the limb stays
    clear. ``thickness`` = limb thickness at ``a`` (canvas-height fraction); ``kind`` = "leafy", "autumn",
    "blossom" or "bare"; ``droop`` = how far the middle sags. The returned Plant has ``extra["seat"](t)``, the
    canvas pixel (x, y) on top of the limb at ``t`` (0 at ``a``, 1 at ``b``), and ``extra["spine"]``."""
    kind = _choice("perch", "kind", kind, ("leafy", "autumn", "blossom", "bare"))
    _check_frac("perch", "a", a)
    _check_frac("perch", "b", b)
    if bark not in BARKS:
        raise _unknown_name("bark", bark, BARKS)
    rng = _rng(seed, 151)
    S = _S(cv)
    W, H = cv.w, cv.h
    A = np.array([a[0] * W, a[1] * H], np.float64)
    B = np.array([b[0] * W, b[1] * H], np.float64)
    d = B - A
    L = float(np.hypot(*d)) or 1.0
    nrm = np.array([-d[1], d[0]]) / L
    if nrm[1] < 0:
        nrm = -nrm                                        # normal pointing down the screen
    n = 14
    ts = np.linspace(0, 1, n + 1)
    r0 = thickness * H / 2
    wob = np.cumsum(rng.normal(0, 1, n + 1)) * 0.0025 * H
    wob -= np.linspace(wob[0], wob[-1], n + 1)
    spine = [tuple(A + d * t + nrm * (math.sin(math.pi * t) * droop * H + w_)) for t, w_ in zip(ts, wob)]
    rad = [r0 * (1 - 0.45 * t) for t in ts]
    segs = [(spine[i][0], spine[i][1], spine[i + 1][0], spine[i + 1][1], rad[i], rad[i + 1], ts[i] * L, ts[i + 1] * L, 0)
            for i in range(n)]
    down = math.atan2(nrm[1], nrm[0])
    along_a = math.atan2(d[1], d[0])
    nodes = []                                            # (x, y, angle) where leaves may grow
    for k in range(int(twigs)):
        t = float(np.clip((k + rng.uniform(0.15, 0.85)) / max(twigs, 1), 0.08, 0.95))
        i = min(int(t * n), n - 1)
        f = t * n - i
        px_ = spine[i][0] + (spine[i + 1][0] - spine[i][0]) * f
        py_ = spine[i][1] + (spine[i + 1][1] - spine[i][1]) * f
        if 0.3 < t < 0.7:
            ang = down + rng.normal(0, 0.45)             # the middle stays clear on top: twigs hang below
        else:
            ang = (along_a if t > 0.5 else along_a + math.pi) + (-0.5 if rng.random() < 0.6 else 0.5) + rng.normal(0, 0.4)
        ln = H * rng.uniform(0.04, 0.085)
        r_ = rad[i] * 0.38
        x_, y_ = px_, py_
        for j in range(3):
            ang += rng.normal(0, 0.25)
            x2, y2 = x_ + math.cos(ang) * ln / 3, y_ + math.sin(ang) * ln / 3
            segs.append((x_, y_, x2, y2, r_ * (1 - j * 0.25), r_ * (1 - (j + 1) * 0.25), 0, ln / 3, 1))
            nodes.append((x2, y2, ang))
            x_, y_ = x2, y2
    nodes.append((spine[-1][0], spine[-1][1], along_a))
    for i in range(2, n, 2):                              # leaves under the limb
        nodes.append((spine[i][0], spine[i][1] + rad[i] * 0.6, down))
    allx = [q[0] for q in spine] + [q[0] for q in nodes]
    ally = [q[1] for q in spine] + [q[1] for q in nodes]
    pad = H * 0.14
    x0, y0, x1, y1 = _clip_box(cv, min(allx) - pad, min(ally) - pad, max(allx) + pad, max(ally) + pad)
    if x0 >= x1 or y0 >= y1:
        return Plant("branch", np.zeros((cv.h, cv.w), F32))
    P = _Patch(x0, y0, x1, y1)
    barkc = BARKS["cherry"] if kind == "blossom" else BARKS[bark]
    c, al, _ = _render_bark(P, segs, light, barkc, kind="smooth" if kind == "blossom" else "furrow", S=S, seed=seed)
    P.rgb, P.a = c, al
    if leaves and kind != "bare":
        cols = AUTUMN if kind == "autumn" else ["#2a5a22", "#3a7a2c", "#4e8a32", "#2e6a2a", "#5a9a3a"]
        lead = H * 0.03
        for (x, y, an) in nodes:
            for j in range(int(rng.integers(2, 4))):
                if kind == "blossom":
                    _flower_head(P, x + rng.normal(0, lead * 0.6), y + rng.normal(0, lead * 0.5),
                                 H * rng.uniform(0.012, 0.018), dict(FLOWERS["cherry blossom"], color=color or "#f6bcd0",
                                                                    tilt=rng.uniform(0, 0.5)), light, rng)
                    continue
                pl = _leaf_patch(x, y, H * rng.uniform(0.035, 0.06), an + rng.uniform(-1.1, 1.1),
                                 "ovate" if kind == "leafy" else rng.choice(["maple", "oak"]),
                                 color or cols[int(rng.integers(len(cols)))], light, flip=rng.uniform(0.45, 1.0), S=S)
                if pl is not None:
                    P.merge(pl)

    def seat(t):
        t = float(np.clip(t, 0.0, 1.0))
        i = min(int(t * n), n - 1)
        f = t * n - i
        return (spine[i][0] + (spine[i + 1][0] - spine[i][0]) * f,
                spine[i][1] + (spine[i + 1][1] - spine[i][1]) * f - (rad[i] + (rad[i + 1] - rad[i]) * f) * 0.85)
    top = min(spine, key=lambda q: q[1])
    return _finish(cv, "branch", P, tuple(A), top, tuple((A + B) / 2), paint=paint, clip=clip,
                   extra={"seat": seat, "spine": spine})


CLIMBERS = {
    "bougainvillea": dict(bract=["#a3124d", "#c2185b", "#e0307c", "#f0529a", "#ff7ab8"],
                          leaf=["#1f4a1c", "#2f6a26", "#478a34"], flower_p=0.55, hang=False),
    "wisteria": dict(bract=["#5a3f93", "#7656b4", "#9877d4", "#b99aea", "#dccbf8"],
                     leaf=["#26531f", "#3b7a2b", "#5a9a3a"], flower_p=0.0, hang=True),
    "ivy": dict(bract=None, leaf=["#0f240f", "#1c3c19", "#2f5e28", "#4e843a", "#86b064"], flower_p=0.0, hang=False),
    "jasmine": dict(bract=["#f7f2e6", "#fffdf4", "#efe8d2"], leaf=["#1c4a24", "#2f6a34", "#4a8a48"], flower_p=0.5,
                    hang=False),
}


def climber(cv, kind="bougainvillea", box=(0.1, 0.05, 0.9, 0.95), seed=0, *, drape="frame", along=None, density=1.0,
            size=0.022, color=None, light=LIGHT, clip=None, paint=True):
    """A flowering climber draped over a wall or doorway: ``kind`` = "bougainvillea", "wisteria" (hanging
    racemes), "ivy" or "jasmine". ``box`` = (u0, v0, u1, v1) of the wall / doorway; ``drape`` = "frame" (up one side,
    over the top and down the other: a doorway or window), "top" (a curtain hanging from the top edge: a pergola or
    wall head) or "wall" (vines wandering over the whole box); ``along=mask`` hangs the vines from the top edge of
    a mask instead. ``size`` = leaf size (canvas-height fraction), ``color`` = flower colour, ``density`` thins or
    fills the growth."""
    kind = _choice("climber", "kind", kind, tuple(CLIMBERS))
    drape = _choice("climber", "drape", drape, ("frame", "top", "wall"))
    spec = CLIMBERS[kind]
    rng = _rng(seed, 163)
    S = _S(cv)
    W, H = cv.w, cv.h
    lpx = max(size * H, 3.0)
    keep = None
    if along is not None:
        keep = np.asarray(along) > 0.5
        ys_, xs_ = np.nonzero(keep)
        if len(xs_) == 0:
            return Plant("climber", np.zeros((H, W), F32))
        bx0, by0, bx1, by1 = float(xs_.min()), float(ys_.min()), float(xs_.max()), float(ys_.max())
        drape = "top"
    else:
        _check_frac("climber", "box", box)
        bx0, by0, bx1, by1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    bw, bh = bx1 - bx0, by1 - by0
    step = lpx * 0.55
    paths, hangers = [], []
    amp = lpx * 0.5

    def wiggle(p, ph):
        return [(x + math.sin(i * 0.55 + ph) * amp * 0.6, y + math.cos(i * 0.43 + ph) * amp * 0.4)
                for i, (x, y) in enumerate(p)]

    def seg(p0, p1):
        m = max(int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step), 1)
        return [(p0[0] + (p1[0] - p0[0]) * k / m, p0[1] + (p1[1] - p0[1]) * k / m) for k in range(m + 1)]
    hang_from = []
    if drape == "frame":
        inset = lpx * 0.8
        left = seg((bx0 + inset, by1), (bx0 + inset, by0 + inset))
        top = seg((bx0 + inset, by0 + inset), (bx1 - inset, by0 + inset))
        right = seg((bx1 - inset, by0 + inset), (bx1 - inset, by1))
        paths.append(wiggle(left + top[1:] + right[1:], rng.uniform(0, 6)))
        for k in range(int(5 * density) + 1):          # a second, shorter vine up the doorway sides
            sx = bx0 + inset + rng.uniform(0.2, 1.8) * lpx if rng.random() < 0.5 else bx1 - inset - rng.uniform(0.2, 1.8) * lpx
            y_a = by1 - rng.uniform(0.0, 0.5) * bh
            paths.append(wiggle(seg((sx, y_a), (sx + rng.normal(0, lpx), y_a - rng.uniform(0.2, 0.5) * bh)),
                                rng.uniform(0, 6)))
        hang_from = paths[0][len(left) - 1:]
    elif drape == "top":
        if keep is not None:
            cols = np.nonzero(keep.any(0))[0]
            topy = np.argmax(keep, 0).astype(np.float64)
            line = [(float(x), float(topy[x]) + lpx * 0.2) for x in range(int(cols.min()), int(cols.max()) + 1, max(int(step), 1))
                    if keep[:, x].any()]
        else:
            line = seg((bx0, by0 + lpx * 0.3), (bx1, by0 + lpx * 0.3))
        paths.append(wiggle(line, rng.uniform(0, 6)))
        hang_from = paths[0]
    else:
        for k in range(int(4 * density) + 2):
            x, y = rng.uniform(bx0, bx1), by1
            pts = [(x, y)]
            ang = -math.pi / 2 + rng.normal(0, 0.5)
            while y > by0 and len(pts) < 400:
                ang += rng.normal(0, 0.35) + 0.06 * (-math.pi / 2 - ang)
                x = float(np.clip(x + math.cos(ang) * step, bx0, bx1))
                y += math.sin(ang) * step
                pts.append((x, y))
            paths.append(pts)
    # hanging strands from the upper vine: racemes for wisteria, trailing shoots for the others
    hang_len = bh * (0.28 if spec["hang"] else 0.16)
    every = 2 if spec["hang"] else max(int(3 / max(density, 0.2)), 1)
    for (x, y) in hang_from[::every]:
        if rng.random() < (0.75 if spec["hang"] else 0.5) * min(density, 1.5):
            hangers.append((x + rng.normal(0, lpx * 0.3), y, hang_len * rng.uniform(0.35, 1.0)))
    xs = [q[0] for p in paths for q in p] + [h[0] for h in hangers]
    ys = [q[1] for p in paths for q in p] + [h[1] + h[2] for h in hangers]
    x0, y0, x1, y1 = _clip_box(cv, min(xs) - lpx * 3, min(ys) - lpx * 3, max(xs) + lpx * 3, max(ys) + lpx * 3)
    if x0 >= x1 or y0 >= y1:
        return Plant("climber", np.zeros((H, W), F32))
    v = _Vec(x0, y0, x1, y1, aa=2)
    lx, ly = _l2(light)
    leaf_ramp = spec["leaf"]
    bract = list(spec["bract"]) if spec["bract"] else None
    if bract is not None and color is not None:
        base_c = rgb(color)
        bract = [tuple(np.clip(base_c * f, 0, 1)) for f in (0.62, 0.8, 1.0, 1.15, 1.3)]
    stem = "#4a3a22"

    def leaf_at(x, y, ang, sz):
        tone = 0.2 + 0.55 * (0.5 - 0.5 * (math.cos(ang) * lx + math.sin(ang) * ly)) + rng.normal(0, 0.1)
        col = _tone_col(leaf_ramp, tone)
        if kind == "ivy":
            _ivy_leaf(v, x, y, sz * 0.62, ang, col, _tone_col(leaf_ramp, tone + 0.25))
            return
        th = np.linspace(0, 2 * math.pi, 18, endpoint=False)
        ca, sa = math.cos(ang), math.sin(ang)
        pts = []
        for t in th:
            u_, w_ = math.cos(t) * sz * 0.5 + sz * 0.5, math.sin(t) * sz * 0.28 * (1 - 0.3 * math.cos(t))
            pts.append((x + u_ * ca - w_ * sa, y + u_ * sa + w_ * ca))
        v.poly(pts, col)

    def bract_cluster(x, y, r):
        for j in range(int(rng.integers(3, 7))):
            bxx, byy = x + rng.normal(0, r * 0.7), y + rng.normal(0, r * 0.6)
            tone = float(np.clip(rng.uniform(0.0, 1.0) ** 1.3, 0, 1))
            col = _tone_col(bract, tone)
            ang = rng.uniform(0, 2 * math.pi)
            for q in range(3):
                a2 = ang + q * 2.1
                v.ellipse(bxx + math.cos(a2) * r * 0.35, byy + math.sin(a2) * r * 0.35, r * 0.42, r * 0.34, col)
            if kind == "bougainvillea":
                v.ellipse(bxx, byy, r * 0.1, r * 0.1, "#f6e9c8")

    for pth in paths:
        v.line(pth, max(lpx * 0.1, 0.8 * S), stem)
        for i, (x, y) in enumerate(pth):
            if rng.random() > 0.8 * min(density, 1.6):
                continue
            sd = 1 if i % 2 else -1
            ang = math.atan2(pth[min(i + 1, len(pth) - 1)][1] - pth[max(i - 1, 0)][1],
                             pth[min(i + 1, len(pth) - 1)][0] - pth[max(i - 1, 0)][0]) + sd * rng.uniform(0.7, 1.5)
            leaf_at(x, y, ang, lpx * rng.uniform(0.7, 1.15))
            if spec["flower_p"] and rng.random() < spec["flower_p"] * min(density, 1.6):
                bract_cluster(x + rng.normal(0, lpx * 0.5), y + rng.normal(0, lpx * 0.5), lpx * rng.uniform(0.55, 0.85))
    for (x, y, Lh) in hangers:
        sway = rng.uniform(0, 6)
        m = max(int(Lh / (lpx * 0.5)), 2)
        strand = [(x + math.sin(k * 0.4 + sway) * lpx * 0.2, y + Lh * k / m) for k in range(m + 1)]
        if keep is not None:
            strand = [q for q in strand if keep[min(max(int(q[1]), 0), H - 1), min(max(int(q[0]), 0), W - 1)]]
            if len(strand) < 2:
                continue
        v.line(strand, max(lpx * 0.07, 0.7), stem)
        if kind == "wisteria":
            for k, (qx, qy) in enumerate(strand):
                t = k / max(len(strand) - 1, 1)
                wd = lpx * (0.7 * (1 - t) ** 0.7 + 0.12)
                for side in (-1, 1):
                    fx = qx + side * wd * rng.uniform(0.2, 1.0)
                    tone = float(np.clip(0.25 + 0.7 * t + rng.normal(0, 0.12), 0, 1))
                    v.ellipse(fx, qy + rng.normal(0, lpx * 0.1), lpx * 0.3, lpx * 0.26, _tone_col(bract, tone))
        else:
            for k, (qx, qy) in enumerate(strand):
                if rng.random() < 0.7:
                    leaf_at(qx, qy, math.pi / 2 + (1 if k % 2 else -1) * rng.uniform(0.5, 1.2), lpx * rng.uniform(0.6, 1.0))
                if bract is not None and spec["flower_p"] and rng.random() < spec["flower_p"] * 0.4:
                    bract_cluster(qx, qy, lpx * rng.uniform(0.5, 0.7))
    Pt = v.patch()
    return _finish(cv, "climber", Pt, (bx0 + bw / 2, by1), (bx0 + bw / 2, by0), (bx0 + bw / 2, by0 + bh / 2),
                   paint=paint, clip=clip, light=light)


# ============================================================================
# rocks
# ============================================================================
ROCKS = {
    "granite": ["#3a3a3e", "#5e5e62", "#86868a", "#adacaa", "#d2d0cc"],
    "sandstone": ["#5a3a22", "#8a5a34", "#b8804a", "#d8a66a", "#f0cc94"],
    "slate": ["#22262c", "#363c44", "#4e5660", "#6e7884", "#9aa4ae"],
    "basalt": ["#141414", "#26262a", "#3a3a40", "#56565c", "#7a7a80"],
    "limestone": ["#6a665c", "#948e80", "#bab4a4", "#d6d0c0", "#eeeadc"],
    "mossy": ["#2e3230", "#4a4e4a", "#6a6e68", "#8e928a", "#b8bab2"],
}


def _hull(points):
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


# per kind: planes (facet count), grain, strata, cracks, rough normals, plane cut range, speckle
_ROCK_STYLE = {
    "granite": dict(planes=16, grain=0.10, strata=0.0, cracks=1.0, rough=0.35, cut=(0.84, 1.0), speckle=0.5),
    "sandstone": dict(planes=14, grain=0.05, strata=1.0, cracks=0.6, rough=0.25, cut=(0.82, 1.0), speckle=0.15),
    "slate": dict(planes=12, grain=0.04, strata=0.55, cracks=0.5, rough=0.18, cut=(0.70, 1.0), speckle=0.1),
    "basalt": dict(planes=14, grain=0.05, strata=0.0, cracks=0.8, rough=0.2, cut=(0.74, 1.0), speckle=0.2),
    "limestone": dict(planes=15, grain=0.07, strata=0.45, cracks=0.9, rough=0.3, cut=(0.84, 1.0), speckle=0.3),
    "mossy": dict(planes=16, grain=0.10, strata=0.0, cracks=1.0, rough=0.35, cut=(0.84, 1.0), speckle=0.5),
}


def _rock_patch(cv, cx, by, Wp, Hp, rng, palette, light, facets=7, moss_amt=0.0, seed=0, S=1.0, flat_base=True,
                kind="granite"):
    """Boulder carved from a solid: an ellipsoid cut by random planes (an angular polyhedron with big flat faces),
    softly rounded along the edges, with bumpy surface normals, grain, strata, cracks, ground contact shade and
    optional moss on the up-facing surfaces."""
    st = _ROCK_STYLE.get(kind, _ROCK_STYLE["granite"])
    x0, y0, x1, y1 = _clip_box(cv, cx - Wp * 0.62, by - Hp * 1.2, cx + Wp * 0.62, by + 2)
    if x0 >= x1 or y0 >= y1:
        return None
    P = _Patch(x0, y0, x1, y1)
    hw, hh = Wp * 0.5, Hp * 0.5
    K = int(st["planes"])
    v = rng.normal(size=(K, 3))
    v[:, 1] = np.abs(v[:, 1]) * 0.8 + v[:, 1] * 0.25 + 0.1       # favour faces looking up and sideways
    v[0] = (rng.normal(0, 0.25), 1.0, rng.uniform(0.25, 0.6))       # one top face
    v[1] = (rng.uniform(0.7, 1.0) * rng.choice([-1, 1]), 0.25, 0.8)   # two shoulder faces
    v[2] = (rng.uniform(0.7, 1.0) * rng.choice([-1, 1]), 0.25, 0.8)
    v[3:6, 2] = -np.abs(v[3:6, 2]) - 0.3                           # a few back planes close the solid
    v[6] = (-0.8, -0.5, 0.35)                                        # chamfers: round off the lower corners
    v[7] = (0.8, -0.5, 0.35)
    v[8] = (rng.normal(0, 0.3), -0.45, 0.9)
    v[9:, 1] = np.where(rng.random(len(v) - 9) < 0.3, -np.abs(v[9:, 1]), v[9:, 1])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cutlo, cuthi = st["cut"]
    planes = []
    for nx_, ny_, nz_ in v:
        sup = math.sqrt(nx_ * nx_ + ny_ * ny_ + (0.8 * nz_) ** 2)
        d_ = ny_ * 1.0 + sup * rng.uniform(cutlo, cuthi)
        planes.append((float(nx_), float(ny_), float(nz_), float(d_)))
    if flat_base:
        planes.append((0.0, -1.0, 0.0, 0.0))
    ss = 3
    gy, gx = np.mgrid[0:(y1 - y0) * ss, 0:(x1 - x0) * ss].astype(F32)
    gx = (gx + 0.5) / ss + x0
    gy = (gy + 0.5) / ss + y0
    ph = float(rng.uniform(0, 100))

    def warp(X, Y):
        wx = fbm2d_at(X / (Wp * 0.35) + ph, Y / (Wp * 0.35), 2, seed=seed + 5) * 0.07
        wy = fbm2d_at(X / (Wp * 0.35) + ph + 40, Y / (Wp * 0.35) + 9, 2, seed=seed + 6) * 0.07
        return (X - cx) / hw + wx, (by - Y) / hh + wy

    def solid(xn, yn):
        zhi = np.full(xn.shape, 1e3, F32)
        zlo = np.full(xn.shape, -1e3, F32)
        ok = np.ones(xn.shape, bool)
        zs, ns = [], []
        for nx_, ny_, nz_, d_ in planes:
            t = d_ - nx_ * xn - ny_ * yn
            if nz_ > 1e-3:
                z = t / nz_
                zhi = np.minimum(zhi, z)
                zs.append(z)
                ns.append((nx_, ny_, nz_))
            elif nz_ < -1e-3:
                zlo = np.maximum(zlo, t / nz_)
            else:
                ok &= t >= 0
        return ok & (zlo < zhi), zs, ns

    xn, yn = warp(gx, gy)
    ins, _, _ = solid(xn, yn)
    m = ins.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3)).astype(F32)
    if not m.any():
        return None
    X, Y = P.grid()
    xn, yn = warp(X, Y)
    _, zs, ns = solid(xn, yn)
    zst = np.stack(zs)
    wgt = np.exp(-(zst - zst.min(axis=0)) / 0.035)
    wgt /= wgt.sum(axis=0)
    nxs = sum(wgt[i] * ns[i][0] for i in range(len(ns)))
    nys = sum(wgt[i] * ns[i][1] for i in range(len(ns)))
    nzs = sum(wgt[i] * ns[i][2] for i in range(len(ns)))
    bsz = max(Wp * 0.05, 3.0)
    r_ = st["rough"]
    nxs = nxs + fbm2d_at(X / bsz, Y / bsz, 3, seed=seed + 11) * r_
    nys = nys + fbm2d_at(X / bsz + 17, Y / bsz + 5, 3, seed=seed + 12) * r_
    bl = gaussian_blur(m, max(Wp * 0.03, 1))
    gy_, gx_ = np.gradient(bl)
    nxs = nxs - gx_ * Wp * 0.35
    nys = nys + gy_ * Wp * 0.35
    nn = np.sqrt(nxs * nxs + nys * nys + nzs * nzs) + 1e-6
    nxs, nys, nzs = nxs / nn, nys / nn, nzs / nn
    L3 = _l3(light, 0.55)
    d = np.clip(nxs * L3[0] - nys * L3[1] + nzs * L3[2], 0, 1)
    grain = fbm2d_at(X / (2.2 * S + 0.5), Y / (2.2 * S + 0.5), 2, seed=seed + 1) * st["grain"]
    tone = 0.08 + 0.95 * d ** 1.15 + grain
    if st["strata"] > 0:
        yy = (by - Y) / hh
        band = np.sin((yy + fbm2d_at(X / (Wp * 0.4), Y / (Wp * 0.4), 2, seed=seed + 3) * 0.5) * 9.0
                      + float(rng.uniform(0, 6)))
        tone = tone + st["strata"] * 0.07 * band
    if st["cracks"] > 0:
        cn = fbm2d_at(X / (Wp * 0.3) + 3, Y / (Wp * 0.3), 3, seed=seed + 21)
        crack = np.exp(-(cn / 0.03) ** 2)
        cn2 = fbm2d_at(X / (Wp * 0.18) + 11, Y / (Wp * 0.18), 2, seed=seed + 22)
        crack = np.maximum(crack, 0.7 * np.exp(-(cn2 / 0.035) ** 2))
        crack = crack * smoothstep(0.02, 0.2, fbm2d_at(X / (Wp * 0.5) + 5, Y / (Wp * 0.5) + 2, 2, seed=seed + 23))
        tone = tone - 0.24 * st["cracks"] * crack
    ao = clamp01((by - Y) / (Hp * 0.22))
    tone = tone * (0.6 + 0.4 * ao)
    col = _ramp(palette, np.clip(tone, 0, 1))
    spk = _hash(np.floor(X).astype(np.int64), np.floor(Y).astype(np.int64), seed)
    flecks = _hash(np.floor(X / 1.6).astype(np.int64), np.floor(Y / 1.6).astype(np.int64), seed + 7)
    col = col * (1 + (spk - 0.5) * 0.10)[..., None]
    if st["speckle"] > 0:
        dark = (flecks > 0.9).astype(F32) * st["speckle"]
        pale = (flecks < 0.06).astype(F32) * st["speckle"]
        col = col * (1 - 0.28 * dark + 0.22 * pale)[..., None]
    if moss_amt > 0:
        upf = clamp01((nys - 0.2) * 2.2)
        cs = max(Wp * 0.08, 2)
        mf1, _, _, _, mid = _worley(X, Y, cs, seed + 9)
        clump = clamp01(1 - mf1 / (cs * 0.8)) * (mid < moss_amt + 0.2)
        fine = fbm2d_at(X / (1.5 * S + 0.5), Y / (1.5 * S + 0.5), 2, seed=seed + 2)
        mm = smoothstep(0.25, 0.5, upf * (0.6 + 0.6 * clump) * moss_amt * 1.6 + fine * 0.15)
        mcol = _ramp(["#16240c", "#2a4216", "#46681e", "#6f9230", "#9ab84c"], np.clip(d * 0.85 + fine * 0.3, 0, 1))
        col = _blend_rgb(col, mcol, mm)
    P.rgb, P.a = col.astype(F32), m
    return P


def _shape_mask_local(P, poly, aa=3):
    """Anti-aliased polygon mask on a patch (points in canvas px)."""
    v = _Vec(P.x0, P.y0, P.x1, P.y1, aa=aa)
    v.poly([tuple(p) for p in poly], (1, 1, 1))
    _, a = v.result()
    return a


def rock(cv, at=(0.5, 0.9), size=0.2, seed=0, *, kind="granite", height=None, moss=0.0, palette=None, light=LIGHT,
         shadow=True, clip=None, haze=None, paint=True):
    """Boulder with faceted shading (flat planes meeting at edges, cracks, grain) and optional ``moss`` 0~1 on
    top. ``size`` = width (canvas-height fraction); kind: granite, sandstone, slate, basalt, limestone, mossy."""
    rng = _rng(seed, 141)
    S = _S(cv)
    Wp = size * cv.h
    Hp = (height if height is not None else size * rng.uniform(0.62, 0.88)) * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    if kind not in ROCKS:
        raise _unknown("rock kind", kind, ROCKS, "Try flora.names('rock')")
    pal = list(palette or ROCKS[kind])
    if kind == "mossy" and moss == 0:
        moss = 0.7
    P = _rock_patch(cv, bx, by, Wp, Hp, rng, pal, light, moss_amt=moss, seed=seed, S=S, kind=kind)
    return _finish(cv, "rock", P, (bx, by), (bx, by - Hp), (bx, by - Hp / 2), paint=paint, clip=clip, shadow=shadow,
                   light=light, haze=haze, shadow_len=0.6)


def rock_pile(cv, at=(0.5, 0.92), width=0.35, seed=0, *, n=7, kind="granite", moss=0.0, light=LIGHT, shadow=True,
              clip=None, paint=True):
    """Pile of boulders: big ones at the bottom, smaller ones resting on top, painted back to front."""
    rng = _rng(seed, 143)
    S = _S(cv)
    Wp = width * cv.h
    bx, by = at[0] * cv.w, at[1] * cv.h
    if kind not in ROCKS:
        raise _unknown("rock kind", kind, ROCKS, "Try flora.names('rock')")
    pal = list(ROCKS[kind])
    specs = []
    for i in range(int(n)):
        t = i / max(n - 1, 1)
        w = Wp * rng.uniform(0.3, 0.5) * (1 - 0.45 * t)
        x = bx + rng.uniform(-0.5, 0.5) * Wp * (1 - 0.6 * t)
        y = by - t * Wp * 0.35 + rng.uniform(-0.03, 0.03) * Wp
        specs.append((y - w * 0.1 * (t > 0.3), x, w))
    specs.sort(key=lambda s: s[0])
    full = _Patch(*_clip_box(cv, bx - Wp, by - Wp, bx + Wp, by + 4))
    for (y, x, w) in specs:
        p = _rock_patch(cv, x, y, w, w * rng.uniform(0.55, 0.8), rng, pal, light, moss_amt=moss,
                        seed=int(rng.integers(1 << 20)), S=S, kind=kind)
        if p is not None:
            full.merge(p)
    return _finish(cv, "rock pile", full, (bx, by), (bx, by - Wp * 0.6), (bx, by - Wp * 0.3), paint=paint,
                   clip=clip, shadow=shadow, light=light, shadow_len=0.5)


def pebbles(cv, region=(0.2, 0.85, 0.8, 0.98), n=120, seed=0, *, size=(0.008, 0.025), palette=None, light=LIGHT,
            wet=0.0, clip=None, paint=True):
    """Pebbles: smooth rounded stones with soft highlights and contact shadows, smaller towards the top of the
    region. ``wet`` 0~1 darkens and adds gloss."""
    rng = _rng(seed, 145)
    S = _S(cv)
    u0, v0, u1, v1 = region
    pal = palette or ["#6e6a62", "#8a847a", "#a49c90", "#5e5a56", "#b8aea0", "#7a6e62", "#c4bcb0", "#4e4c4a"]
    full = _Patch(*_clip_box(cv, u0 * cv.w - 20 * S, v0 * cv.h - 20 * S, u1 * cv.w + 20 * S, v1 * cv.h + 10 * S))
    L3 = _l3(light)
    stones = sorted([(v0 + (v1 - v0) * rng.random(), rng.uniform(u0, u1)) for _ in range(int(n))])
    for (vv, uu) in stones:
        d = (vv - v0) / max(v1 - v0, 1e-6)
        rx = rng.uniform(*size) * cv.h * (0.5 + 0.5 * d)
        ry = rx * rng.uniform(0.45, 0.7)
        cx, cy = uu * cv.w, vv * cv.h
        X0, Y0, X1, Y1 = int(cx - rx - 3), int(cy - ry - 3), int(cx + rx + 4), int(cy + ry * 1.6 + 4)
        if X0 < full.x0 or Y0 < full.y0 or X1 > full.x1 or Y1 > full.y1:
            continue
        sy, sx = np.mgrid[Y0:Y1, X0:X1].astype(F32)
        a_rot = rng.uniform(-0.4, 0.4)
        u, v = _rot(sx - cx, sy - cy, a_rot)
        q = (u / rx) ** 2 + (v / ry) ** 2
        a = clamp01((1 - np.sqrt(q)) * ry + 0.5)
        sh = clamp01(1 - ((sx - cx - rx * 0.15) / (rx * 1.1)) ** 2 - ((sy - cy - ry * 0.75) / (ry * 0.55)) ** 2) * 0.5
        full.over(np.zeros(sx.shape + (3,), F32), (sh * (1 - a)).astype(F32), X0, Y0)
        nz = np.sqrt(np.clip(1 - q, 0, 1))
        nx, ny = u / rx, v / ry
        dd = np.clip(nx * L3[0] + ny * L3[1] + nz * L3[2], 0, 1)
        base = rgb(pal[int(rng.integers(len(pal)))]) * (1 - 0.35 * wet)
        spec = np.clip(2 * dd * nz - L3[2], 0, 1) ** (8 + 30 * wet) * (0.12 + 0.6 * wet)
        col = base[None, None] * (0.35 + 0.75 * dd)[..., None] + spec[..., None]
        full.over(col.astype(F32), a.astype(F32), X0, Y0)
    m = _commit(cv, full, clip) if paint else full.a
    return Plant("pebbles", m, (cv.w * (u0 + u1) / 2, v1 * cv.h), (cv.w * (u0 + u1) / 2, v0 * cv.h),
                 (cv.w * (u0 + u1) / 2, cv.h * (v0 + v1) / 2), (full.x0, full.y0, full.x1, full.y1), {}, full)


def cliff(cv, region=(0.0, 0.2, 0.45, 1.0), seed=0, *, kind="sandstone", strata=12, palette=None, light=LIGHT,
          edge="right", clip=None, paint=True, moss=0.0):
    """Cliff / rock face with strata: stacked layers of varying thickness, lit ledges with shadowed undersides,
    vertical joints and a ragged outline. ``edge`` = which side of the region is the open (ragged) side:
    "right", "left" or "top". ``palette`` = a list of colours from dark to light (``["#3a2a1c", "#7a5a3a", "#b8905e",
    "#e6c898"]``); without it ``kind`` picks one of ``ROCKS``."""
    from .noise import fbm1d
    if palette is None and kind not in ROCKS:
        raise _unknown("rock kind", kind, ROCKS, "Try flora.names('rock')")
    edge = _choice("cliff", "edge", edge, ("right", "left", "top"))
    rng = _rng(seed, 147)
    S = _S(cv)
    u0, v0, u1, v1 = region
    x0, y0, x1, y1 = _clip_box(cv, u0 * cv.w, v0 * cv.h, u1 * cv.w, v1 * cv.h)
    if x0 >= x1 or y0 >= y1:
        return Plant("cliff", np.zeros((cv.h, cv.w), F32))
    P = _Patch(x0, y0, x1, y1)
    X, Y = P.grid()
    h, w = P.shape
    pal = list(palette or ROCKS.get(kind, ROCKS["sandstone"]))
    # strata: layer index from a warped vertical coordinate
    warp = fbm2d_at(X / (300 * S), Y / (300 * S), 3, seed=seed) * 40 * S
    tilt = rng.uniform(-0.08, 0.08)
    yy = Y + warp + (X - x0) * tilt
    bounds = np.cumsum(rng.uniform(0.4, 1.6, strata * 3))
    bounds = (bounds / bounds[strata - 1]) * (y1 - y0) + y0 - 40 * S
    idx = np.searchsorted(bounds, yy).astype(np.int32)
    lo = np.concatenate([[y0 - 400 * S], bounds])[np.clip(idx, 0, len(bounds))]
    hi = bounds[np.clip(idx, 0, len(bounds) - 1)]
    t = np.clip((yy - lo) / np.maximum(hi - lo, 1), 0, 1)
    # each layer juts out by a different amount → ledge on top (lit), overhang shadow under the layer above
    jut = _hash(idx, 1, seed) * 30 * S
    # outline
    n_rows = h
    if edge in ("right", "left"):
        prof = fbm1d(h, 120 * S, 4, seed=seed)[:, None] * 40 * S
        lay = _hash(idx, 5, seed) * 50 * S
        edge_x = (x1 - 60 * S if edge == "right" else x0 + 60 * S)
        if edge == "right":
            a = clamp01(edge_x + prof + lay * 0.6 - X + 0.5)
        else:
            a = clamp01(X - (edge_x - prof - lay * 0.6) + 0.5)
        tprof = fbm1d(w, 110 * S, 4, seed=seed + 2)[None, :] * 30 * S
        a = a * clamp01(Y - (y0 + 30 * S + tprof) + 0.5)
    else:
        prof = fbm1d(w, 150 * S, 4, seed=seed)[None, :] * 50 * S
        a = clamp01(Y - (y0 + 60 * S + prof) + 0.5)
    ledge = smoothstep(0.12, 0.0, t)                         # top of each layer
    under = smoothstep(0.75, 1.0, t)                          # underside, in shadow of the layer below's jut
    lx, ly = _l2(light)
    face = 0.55 + 0.15 * (_hash(idx, 2, seed) - 0.5)
    f1, f2, fx, fy, cid = _worley(X / 2.5, Y, 70 * S, seed + 3)          # tall blocks → vertical joints
    joint = 1 - smoothstep(0, 2.2 * S, (f2 - f1))
    blockv = (cid - 0.5) * 0.12
    grain = fbm2d_at(X / (5 * S), Y / (2 * S), 3, seed=seed + 4) * 0.06
    streaks = fbm2d_at(X / (45 * S), Y / (220 * S), 3, seed=seed + 6) * 0.07
    tone = face - 0.05 + blockv + grain + streaks + 0.38 * ledge * clamp01(-ly + 0.3) - 0.42 * under - 0.3 * joint
    tone = tone + 0.1 * (-lx) * (fbm2d_at(X / (60 * S), Y / (60 * S), 2, seed=seed + 5))
    colsel = _hash(idx, 3, seed)
    col = _ramp(pal, np.clip(tone, 0, 1))
    col = col * (0.88 + 0.22 * colsel)[..., None]
    if moss > 0:
        mm = ledge * clamp01((_hash(idx, 7, seed) - (1 - moss)) * 4) * (0.6 + 0.6 * grain * 8)
        col = _blend_rgb(col, _ramp(["#16240c", "#2a4216", "#46681e", "#6f9230", "#9ab84c"], np.clip(tone, 0, 1)),
                         clamp01(mm))
    P.rgb, P.a = col.astype(F32), a.astype(F32)
    m = _commit(cv, P, clip) if paint else P.a
    return Plant("cliff", m, (cv.w * (u0 + u1) / 2, y1), (cv.w * (u0 + u1) / 2, y0),
                 (cv.w * (u0 + u1) / 2, (y0 + y1) / 2), (x0, y0, x1, y1), {}, P)


def stepping_stones(cv, path=((0.5, 0.98), (0.45, 0.8), (0.55, 0.65)), n=7, size=0.12, seed=0, *, kind="slate",
                    palette=None, light=LIGHT, clip=None, paint=True):
    """Stepping stones along ``path`` (list of (u, v)): flat slabs in perspective (lit top, darker rim, contact
    shadow), smaller towards the top of the path."""
    rng = _rng(seed, 149)
    S = _S(cv)
    if palette is None and kind not in ROCKS:
        raise _unknown("rock kind", kind, ROCKS, "Try flora.names('rock')")
    pal = list(palette or ROCKS[kind])
    pp = np.array([(u * cv.w, v * cv.h) for u, v in path], np.float64)
    seg = np.hypot(*np.diff(pp, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    ys = pp[:, 1]
    vmin, vmax = ys.min(), ys.max()
    full = _Patch(*_clip_box(cv, pp[:, 0].min() - size * cv.h, vmin - size * cv.h * 0.5, pp[:, 0].max() + size * cv.h,
                             vmax + size * cv.h * 0.4))
    L3 = _l3(light)
    stones = []
    for i in range(int(n)):
        d = cum[-1] * (i + 0.5) / n
        j = min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1)
        f = (d - cum[j]) / max(seg[j], 1e-6)
        x, y = pp[j] + (pp[j + 1] - pp[j]) * f
        depth = (y - vmin) / max(vmax - vmin, 1)
        stones.append((y, x, depth))
    stones.sort()
    for (y, x, depth) in stones:
        rx = size * cv.h * 0.5 * (0.45 + 0.55 * depth) * rng.uniform(0.85, 1.1)
        ry = rx * (0.28 + 0.15 * depth)
        th = rx * 0.12
        X0, Y0, X1, Y1 = int(x - rx * 1.3), int(y - ry * 1.3), int(x + rx * 1.3) + 2, int(y + ry * 1.5 + th) + 3
        X0, Y0, X1, Y1 = max(X0, full.x0), max(Y0, full.y0), min(X1, full.x1), min(Y1, full.y1)
        if X0 >= X1 or Y0 >= Y1:
            continue
        sy, sx = np.mgrid[Y0:Y1, X0:X1].astype(F32)
        th_ = np.arctan2(sy - y, sx - x)
        wob = 1 + 0.07 * np.sin(th_ * 3 + rng.uniform(0, 6)) + 0.04 * np.sin(th_ * 7 + rng.uniform(0, 6))
        q = np.sqrt(((sx - x) / rx) ** 2 + ((sy - y) / ry) ** 2) / wob
        qs = np.sqrt(((sx - x) / rx) ** 2 + ((sy - y - th) / ry) ** 2) / wob
        sha = clamp01(1.2 - np.sqrt(((sx - x - rx * 0.1) / (rx * 1.15)) ** 2 + ((sy - y - th * 1.5) / (ry * 1.15)) ** 2)) * 0.55
        full.over(np.zeros(sx.shape + (3,), F32), sha.astype(F32), X0, Y0)
        side = clamp01((1 - qs) * ry + 0.5)
        full.over(_ramp(pal, np.full(sx.shape, 0.22, F32)), side.astype(F32), X0, Y0)
        top = clamp01((1 - q) * ry + 0.5)
        f1, f2, _, _, cid = _worley(sx, sy * 2.5, rx * 0.5, int(x))
        tone = 0.55 + 0.25 * (-(sx - x) / rx * L3[0] * 0.5) + (cid - 0.5) * 0.08 - 0.15 * (1 - smoothstep(0, 1.5 * S, f2 - f1)) * 0.5
        tone = tone + fbm2d_at(sx / (4 * S), sy / (4 * S), 2, seed=int(y)) * 0.06 - 0.1 * smoothstep(0.75, 1.0, q)
        full.over(_ramp(pal, np.clip(tone, 0, 1)).astype(F32), top.astype(F32), X0, Y0)
    m = _commit(cv, full, clip) if paint else full.a
    return Plant("stepping stones", m, tuple(pp[0]), tuple(pp[-1]), tuple(pp.mean(0)),
                 (full.x0, full.y0, full.x1, full.y1), {}, full)


# ============================================================================
# registry: one keyword → one call (for small models)
# ============================================================================
class _UnknownName(ValueError, KeyError):
    """Unknown registry name (a ValueError that is also a KeyError)."""
    __str__ = ValueError.__str__


def _unknown(kind, name, valid, hint=None):
    return _UnknownName(*_unknown_name(kind, name, list(valid), hint).args)


def _tree_entry(sp, **fixed):
    return lambda cv, at, height, seed, **kw: tree(cv, sp, at=at, height=height, seed=seed, **{**fixed, **kw})


def _flower_entry(k):
    return lambda cv, at, height, seed, **kw: flower(cv, k, at=at, height=height, seed=seed, **kw)


def _potted_entry(k):
    return lambda cv, at, height, seed, **kw: potted(cv, k, at=at, height=height, seed=seed, **kw)


_REG = {}
_ALIASES = {}


def _reg(name, fn, kind, aliases=()):
    _REG[name] = (fn, kind)
    for a in aliases:
        _ALIASES[a] = name


for _sp in TREES:
    _reg(_sp, _tree_entry(_sp), "tree")
for _sp, _al in (("oak", ("oak tree", "deciduous", "shade tree", "broadleaf")),
                 ("maple", ("maple tree",)), ("birch", ("silver birch", "white birch", "aspen")),
                 ("poplar", ("lombardy poplar", "columnar")), ("cypress", ("italian cypress", "juniper")),
                 ("spruce", ("conifer", "evergreen", "spruce tree")), ("fir", ("christmas tree", "fir tree")),
                 ("pine", ("pine tree", "scots pine", "umbrella pine", "stone pine")),
                 ("willow", ("weeping willow",)), ("cherry", ("sakura", "cherry tree", "cherry blossom tree")),
                 ("apple", ("apple tree", "fruit tree", "orchard")), ("baobab", ("bottle tree",)),
                 ("olive", ("olive tree",)), ("bonsai", ("bonsai tree", "miniature tree")),
                 ("winter", ("bare tree", "leafless tree", "winter tree")), ("dead", ("dead tree", "snag", "spooky tree")),
                 ("palm", ("palm tree", "date palm")), ("coconut palm", ("coconut", "coconut tree", "tropical palm")),
                 ("cute", ("cartoon tree", "storybook tree", "round tree"))):
    for _a in _al:
        _ALIASES[_a] = _sp
_reg("old oak", _tree_entry("oak", age=1.4, params=dict(trunk_h=0.22, trunk_r=0.06, flare=0.5, spread=55,
                                                         first_len=0.38, wiggle=14, floor=0.1, aspect=0.65,
                                                         up=0.0)), "tree",
     ("ancient oak", "gnarled oak", "veteran oak", "old tree", "old oak tree"))
_reg("autumn maple", _tree_entry("maple", season="autumn"), "tree", ("fall maple", "red maple", "autumn tree", "fall tree"))
_reg("autumn oak", _tree_entry("oak", season="autumn"), "tree", ("fall oak",))
_reg("autumn birch", _tree_entry("birch", season="autumn"), "tree", ("golden birch", "fall birch"))
_reg("snowy tree", _tree_entry("winter", snow=True), "tree", ("snow tree", "snowy", "tree in snow"))
_reg("bamboo", lambda cv, at, height, seed, **kw: bamboo(cv, at=at, height=height, seed=seed,
                                                       **{"width": height * 0.35, **kw}), "tree",
     ("bamboo grove", "bamboo forest"))
_reg("saguaro", lambda cv, at, height, seed, **kw: saguaro(cv, at=at, height=height, seed=seed, **kw), "tree",
     ("cactus", "desert cactus", "saguaro cactus"))
_reg("prickly pear", lambda cv, at, height, seed, **kw: prickly_pear(cv, at=at, height=height, seed=seed, **kw),
     "tree", ("opuntia", "paddle cactus", "nopal"))
_reg("banana", lambda cv, at, height, seed, **kw: banana(cv, at=at, height=height, seed=seed, **kw), "tree",
     ("banana plant", "banana tree", "plantain"))
_reg("bush", lambda cv, at, height, seed, **kw: bush(cv, at=at, width=height * 1.6, height=height, seed=seed, **kw),
     "shrub", ("shrub", "boxwood", "green bush"))
_reg("flowering bush", lambda cv, at, height, seed, **kw: bush(cv, at=at, width=height * 1.6, height=height,
                                                               seed=seed, **{"flowers": "#e85a8a", **kw}),
     "shrub", ("azalea", "rhododendron", "flowering shrub", "hydrangea"))
_reg("berry bush", lambda cv, at, height, seed, **kw: bush(cv, at=at, width=height * 1.6, height=height, seed=seed,
                                                           **{"berries": "#b81a2a", **kw}), "shrub",
     ("holly", "berries"))
_reg("hedge", lambda cv, at, height, seed, **kw: hedge(cv, box=(at[0] - height * 0.9 * cv.h / cv.w, at[1] - height * 0.8,
                                                                at[0] + height * 0.9 * cv.h / cv.w, at[1]),
                                                       top=height * 0.2, side=height * 0.25, seed=seed, **kw), "shrub",
     ("box hedge", "clipped hedge", "hedgerow"))
for _shp in ("ball", "cone", "spiral", "tiers", "poodle"):
    _reg(f"topiary {_shp}", (lambda s: lambda cv, at, height, seed, **kw: topiary(cv, at=at, height=height,
                                                                                  seed=seed, shape=s, **kw))(_shp),
         "shrub", ((f"{_shp} topiary",) + (("topiary",) if _shp == "ball" else ())))
_reg("grass", lambda cv, at, height, seed, **kw: grass(cv, at=at, height=height, width=height * 1.2, seed=seed, **kw),
     "ground", ("grass tuft", "tuft", "tall grass", "lawn"))
_reg("dry grass", lambda cv, at, height, seed, **kw: grass(cv, at=at, height=height, width=height * 1.2, seed=seed,
                                                           dry=True, **kw), "ground", ("straw", "hay", "savanna grass"))
_reg("reeds", lambda cv, at, height, seed, **kw: reeds(cv, at=at, height=height, width=height * 0.35, seed=seed, **kw),
     "ground", ("reed", "rushes", "sedge"))
_reg("cattails", lambda cv, at, height, seed, **kw: cattails(cv, at=at, height=height, width=height * 0.35,
                                                             seed=seed, **kw), "ground", ("cattail", "bulrush"))
_reg("fern", lambda cv, at, height, seed, **kw: fern(cv, at=at, height=height, seed=seed, **kw), "ground",
     ("ferns", "bracken"))
_reg("mushroom", lambda cv, at, height, seed, **kw: mushroom(cv, at=at, height=height, seed=seed, **kw), "ground",
     ("amanita", "toadstool", "fly agaric"))
_reg("mushrooms", lambda cv, at, height, seed, **kw: mushroom(cv, at=at, height=height, seed=seed,
                                                              **{"cluster": 4, "kind": "porcini", **kw}), "ground",
     ("porcini", "mushroom cluster"))
_reg("pumpkin", lambda cv, at, height, seed, **kw: pumpkin(cv, at=at, size=height, seed=seed, **kw), "ground",
     ("squash", "gourd", "jack o lantern"))
_reg("lily pads", lambda cv, at, height, seed, **kw: lily_pads(cv, at=at, size=height * 0.55, seed=seed,
                                                             **{"spread": height * 0.55, **kw}),
     "ground", ("lily pad", "water lilies", "pond lilies"))
_reg("rock", lambda cv, at, height, seed, **kw: rock(cv, at=at, size=height * 1.2, height=height, seed=seed, **kw),
     "rock", ("boulder", "stone", "big rock"))
_reg("mossy rock", lambda cv, at, height, seed, **kw: rock(cv, at=at, size=height * 1.2, height=height, seed=seed,
                                                           **{"moss": 0.8, **kw}), "rock",
     ("moss rock", "mossy boulder", "moss"))
_reg("sandstone", lambda cv, at, height, seed, **kw: rock(cv, at=at, size=height * 1.2, height=height, seed=seed,
                                                          **{"kind": "sandstone", **kw}), "rock", ("red rock", "desert rock"))
_reg("rock pile", lambda cv, at, height, seed, **kw: rock_pile(cv, at=at, width=height * 1.8, seed=seed, **kw), "rock",
     ("rocks", "boulders", "cairn", "rock heap"))
for _k in FLOWERS:
    _reg(_k, _flower_entry(_k), "flower", ())
for _a, _k in (("roses", "rose"), ("red rose", "rose"), ("tulips", "tulip"), ("sunflowers", "sunflower"),
               ("daisies", "daisy"), ("poppies", "poppy"), ("red poppy", "poppy"), ("lilies", "lily"),
               ("orchids", "orchid"), ("dandelion puff", "dandelion"), ("seed head", "dandelion"),
               ("water lily", "lotus"), ("lotus flower", "lotus"), ("sakura flower", "cherry blossom"),
               ("blue flower", "cornflower"), ("chrysanthemum", "marigold"), ("gerbera", "daisy")):
    _ALIASES[_a] = _k
for _k in POTTED:
    _reg(f"potted {_k}", _potted_entry(_k), "potted", (_k + " pot", "pot " + _k, _k + " in a pot"))
for _a, _k in (("monstera", "monstera"), ("swiss cheese plant", "monstera"), ("snake plant", "snake plant"),
               ("sansevieria", "snake plant"), ("succulent", "succulent"), ("echeveria", "succulent"),
               ("succulents", "succulent"), ("ficus", "ficus"), ("fiddle leaf fig", "ficus"), ("rubber plant", "ficus"),
               ("pothos", "pothos"), ("hanging plant", "pothos"), ("houseplant", "monstera"),
               ("potted plant", "monstera"), ("aloe vera", "aloe"), ("aloe", "aloe"), ("potted cactus", "cactus"),
               ("geranium", "flowering"), ("potted flowers", "flowering")):
    _ALIASES[_a] = "potted " + _k
_reg("vase", lambda cv, at, height, seed, **kw: vase(cv, at=at, height=height, seed=seed, **kw), "flower",
     ("vase of flowers", "flower vase", "flower arrangement"))
_reg("bouquet", lambda cv, at, height, seed, **kw: bouquet(cv, at=(at[0], at[1] - height * 0.35), size=height,
                                                           seed=seed, **kw), "flower", ("bunch of flowers",))
for _k in LEAVES:
    _reg(f"{_k} leaf", (lambda k: lambda cv, at, height, seed, **kw: leaf(cv, k, at=(at[0], at[1] - height / 2),
                                                                         size=height, seed=seed, **kw))(_k),
         "leaf", ())
_ALIASES["leaf"] = "maple leaf"
_ALIASES["autumn leaf"] = "maple leaf"


_STOP = {"a", "an", "the", "of", "with", "in", "on", "tree", "plant", "some", "small", "big", "large", "tall",
         "little", "green", "beautiful", "pretty", "flower", "flowers"}


def names(kind=None):
    """Registered plant names (optionally only one kind: tree, shrub, ground, flower, potted, leaf, rock)."""
    return [k for k, (_, kd) in _REG.items() if kind is None or kd == kind]


def find(query, kind=None):
    """Best registered name for a free-text query ("a big oak tree" → "oak", "palm tree" → "palm"), or None."""
    q = " ".join(str(query).lower().replace("_", " ").replace("-", " ").split())
    pool = {k: k for k in names(kind)}
    pool.update({a: k for a, k in _ALIASES.items() if kind is None or _REG[k][1] == kind})
    if q in pool:
        return pool[q]
    if q.endswith("s") and q[:-1] in pool:
        return pool[q[:-1]]
    words = [w.rstrip("s") if len(w) > 3 else w for w in q.split()]
    best, score = None, 0.0
    for key, target in pool.items():
        kw = [w.rstrip("s") if len(w) > 3 else w for w in key.split()]
        if key in q:
            s = 10 + len(key)
        else:
            common = [w for w in kw if w in words]
            if not common:
                continue
            strong = [w for w in common if w not in _STOP]
            if not strong:
                continue
            s = 2 * len(strong) + len(common) - 0.5 * (len(kw) - len(common))
        if s > score:
            best, score = target, s
    return best


def _check_one_word(query, kind, valid):
    """A single word that is not a name ("oakk") raises the did-you-mean error; free text ("a big oak tree")
    still goes through :func:`find`."""
    q = " ".join(str(query).lower().replace("_", " ").replace("-", " ").split())
    if " " in q or not q:
        return
    pool = set(_REG) | set(_ALIASES) | set(valid)
    stems = {q, q[:-1] if q.endswith("s") else q, q[:-2] if q.endswith("es") else q,
             q[:-3] + "y" if q.endswith("ies") else q}
    if not stems & pool:
        raise _unknown(kind, query, list(_REG) + list(_ALIASES), "Try flora.find('...') or flora.names()")


def plant(cv, name, at=(0.5, 0.9), height=0.4, seed=0, **kw):
    """Paint any registered plant by keyword: ``plant(cv, "oak", at=(0.3, 0.9), height=0.6, seed=2)``.
    ``at`` = base point (u, v), ``height`` = canvas-height fraction. Free text works: "weeping willow",
    "potted monstera", "mossy boulder". Extra keyword options go to the underlying function, which checks them (unknown names raise
    TypeError with the closest valid option; both colour / color spellings are accepted)."""
    if name not in _REG:
        _check_one_word(name, "plant", ())
    key = name if name in _REG else find(name)
    if key is None:
        raise _unknown("plant", name, list(_REG) + list(_ALIASES), "Try flora.find('...') or flora.names()")
    fn, _ = _REG[key]
    return fn(cv, at=at, height=height, seed=seed, **kw)


# ---- scene manifest: each top-level call records what it drew (pixelcanvas.manifest) ----
from . import manifest as _manifest  # noqa: E402

_manifest.register(globals(), "plant", [
    "tree", "palm", "bamboo", "saguaro", "prickly_pear", "banana", "forest", "bush", "hedge", "topiary", "grass",
    "reeds", "cattails", "fern", "ivy", "moss", "lily_pads", "mushroom", "wheat", "flower", "flower_bed", "meadow",
    "bouquet", "potted", "vase", "leaf", "falling_leaves", "leaf_pile", "branch_frame", "perch", "climber"],
    arg={"tree": "species", "flower": "kind", "potted": "kind", "mushroom": "kind", "climber": "kind"},
    labels={"perch": "branch"},
    alias={"palm": "tree", "bush": "shrub", "potted": ["potted plant", "plant"], "vase": "flowers",
           "bouquet": "flowers", "perch": ["branch", "perch", "twig"],
           "climber": ["vine", "climber", "climbing plant"], "ivy": ["vine", "climber"]})
_manifest.register(globals(), "object", ["pumpkin", "rock", "rock_pile", "pebbles", "cliff", "stepping_stones"],
                   alias={"rock_pile": "rocks", "pebbles": "stones"})
_manifest.register(globals(), "plant", ["plant"], arg="name",
                   alias=lambda a: [(_REG.get(a.get("name")) or _REG.get(find(str(a.get("name"))) or "") or
                                     (None, "plant"))[1], "plant"])


# ---- strict options: unknown keyword names raise TypeError naming the closest valid option; colour / color,
# ---- tags / tag, w / width ... are accepted (pixelcanvas._args)
from ._args import strict as _strict, strict_methods as _strict_methods  # noqa: E402
from .form import guard_fractions as _guard_fractions  # noqa: E402

for _n, _f in list(globals().items()):
    if (not _n.startswith("_") and callable(_f) and getattr(_f, "__module__", None) == __name__
            and not isinstance(_f, type)):
        globals()[_n] = _guard_fractions(_strict(_f, _n), _n)
