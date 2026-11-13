# -*- coding: utf-8 -*-
"""Surface textures and materials: wood, stone, brick, fabric, metal, tiles, grass, fur, water, lava and more.

Every texture is built from the structure that makes the real thing recognisable (cells with mortar, blades with a
direction, threads that go over and under, cracks that glow) and lit through a height map, so bumps answer the
scene light.

    from pixelcanvas import Canvas
    from pixelcanvas.texture import texture, fill, find
    cv = Canvas(960, 540)
    cv.paint(texture("brick", 540, 960, seed=3))            # a colour map, lit from the upper left
    fill(cv, mask, "cobblestone", light=(-0.5, -0.8))         # paint a texture into a mask

Conventions
* Sizes are in *texture units* = 1080p pixels at ``scale=1``; ``scale=2`` doubles every feature. The result looks the
  same at any canvas size.
* ``uv=(U, V)``: coordinate arrays (H, W) in texture units instead of the pixel grid -> lay a texture on a floor in
  perspective (``plane_uv``), along an arch, on a cylinder...
* ``mask``: only the bounding box of the mask is computed (fast), the texture stays continuous across masks.
* ``light``: direction towards the light, (x, y) on screen (y down) or (x, y, z). ``bump`` scales the relief.
* ``palette``: a list of colours (dark -> light, or per-piece choices) or gradient stops ``[(0, c), (1, c)]``.
* ``maps=True`` returns ``Maps(color, height, alpha, shine, emit, lit)`` instead of the lit colour map.
"""
from __future__ import annotations

import math
import inspect
import re

import numpy as np

from .color import F32, rgb, mix, clamp01, smoothstep
from .noise import fbm2d, fbm2d_at
from . import filters as _f
from ._args import (reject_unknown as _reject, unknown_name as _unknown_name, check_choice as _choice,
                    strict as _strict)

__all__ = [
    "LIGHT", "TEXTURES", "FAMILIES", "Maps", "Cells", "Strokes",
    "texture", "fill", "names", "find", "describe",
    "cellular", "voronoi_shade", "strokes", "relief", "plane_uv", "polar_uv", "patches", "weather",
]

LIGHT = (-0.6, -0.7)
TEXTURES = {}
FAMILIES = {}
_META = {}
_SQ3 = math.sqrt(3.0)


# ======================================================================== hashing

def _hash(ix, iy, seed=0, salt=0):
    """uint32 hash of integer lattice coordinates."""
    with np.errstate(over="ignore"):
        a = np.asarray(ix).astype(np.int64).astype(np.uint32)
        b = np.asarray(iy).astype(np.int64).astype(np.uint32)
        k = np.uint32((int(seed) * 0x9E3779B1 + int(salt) * 0x85EBCA77 + 0x27D4EB2F) & 0xFFFFFFFF)
        h = (a * np.uint32(0x8DA6B343)) ^ (b * np.uint32(0xD8163841)) ^ k
        h = h ^ (h >> np.uint32(16))
        h = h * np.uint32(0x7FEB352D)
        h = h ^ (h >> np.uint32(15))
        h = h * np.uint32(0x846CA68B)
        h = h ^ (h >> np.uint32(16))
    return h


def _rnd(ix, iy, seed=0, salt=0):
    """Deterministic 0~1 float per lattice point."""
    return ((_hash(ix, iy, seed, salt) >> np.uint32(8)).astype(F32) * F32(1.0 / 16777216.0))


# ======================================================================== colour helpers

def _is_stops(pal):
    p0 = pal[0]
    return (isinstance(p0, (tuple, list)) and len(p0) == 2 and not isinstance(p0[0], str)
            and np.ndim(p0[0]) == 0)


def _ramp(pal, t):
    """Palette lookup: a list of colours spread evenly over 0~1, or gradient stops."""
    if isinstance(pal, str):
        pal = [pal]
    pal = list(pal)
    if _is_stops(pal):
        return mix(pal, t)
    if len(pal) == 1:
        return np.broadcast_to(rgb(pal[0]), np.shape(t) + (3,)).astype(F32)
    n = len(pal) - 1
    return mix([(i / n, c) for i, c in enumerate(pal)], t)


def _pick(pal, r):
    """Discrete choice from a palette with 0~1 randoms r -> (..., 3)."""
    if isinstance(pal, str):
        pal = [pal]
    pal = list(pal)
    if _is_stops(pal):
        return mix(pal, r)
    cols = np.stack([rgb(c) for c in pal]).astype(F32)
    idx = np.clip((np.asarray(r) * len(cols)).astype(int), 0, len(cols) - 1)
    return cols[idx]


def _c(x):
    return rgb(x).astype(F32)


def _m(a):
    """(H, W) -> (H, W, 1) for colour arithmetic."""
    return np.asarray(a, F32)[..., None]


def _lerp(a, b, t):
    t = _m(t)
    return (a + (b - a) * t).astype(F32)


# ======================================================================== building blocks

class Cells:
    """Result of ``cellular``: ``f1``/``f2`` distances to the nearest / second feature point, ``edge`` = distance to
    the cell border, ``id`` (uint32) and ``rand`` (0~1) per cell, ``cx, cy`` the feature point, ``dx, dy`` the
    offset from it, all in the input units. ``cells.r(k)`` = another 0~1 random per cell. Free-form record: every
    keyword becomes an attribute."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def r(self, salt):
        return _rnd(self.gu, self.gv, self.seed, 50 + int(salt))


def cellular(U, V, jitter=1.0, seed=0, round=0.0):
    """Worley / Voronoi cells at coordinates U, V measured in cells (one feature point per unit square).

    jitter: 0 = square grid, 1 = fully random points. round: > 0 rounds the cell corners of ``edge``
    (smooth-min of the border distances, in cell units, try 0.05~0.3) -> pebbles, cushions.
    """
    U = np.asarray(U, F32)
    V = np.asarray(V, F32)
    fu0 = np.floor(U)
    fv0 = np.floor(V)
    iu = fu0.astype(np.int64)
    iv = fv0.astype(np.int64)
    fu = U - fu0
    fv = V - fv0
    j = F32(jitter)
    f1 = np.full(U.shape, np.inf, F32)
    f2 = np.full(U.shape, np.inf, F32)
    mx = np.zeros(U.shape, F32)
    my = np.zeros(U.shape, F32)
    bx = np.zeros(U.shape, np.int64)
    by = np.zeros(U.shape, np.int64)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            gx, gy = iu + ox, iv + oy
            px = ox + 0.5 + j * (_rnd(gx, gy, seed, 1) - 0.5) - fu
            py = oy + 0.5 + j * (_rnd(gx, gy, seed, 2) - 0.5) - fv
            d2 = px * px + py * py
            c = d2 < f1
            f2 = np.where(c, f1, np.minimum(f2, d2))
            f1 = np.where(c, d2, f1)
            mx = np.where(c, px, mx)
            my = np.where(c, py, my)
            bx = np.where(c, ox, bx)
            by = np.where(c, oy, by)
    gu, gv = iu + bx, iv + by
    edge = np.full(U.shape, np.inf, F32)
    e2 = np.full(U.shape, np.inf, F32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            if ox == 0 and oy == 0:
                continue
            gx, gy = gu + ox, gv + oy
            px = (bx + ox).astype(F32) + 0.5 + j * (_rnd(gx, gy, seed, 1) - 0.5) - fu
            py = (by + oy).astype(F32) + 0.5 + j * (_rnd(gx, gy, seed, 2) - 0.5) - fv
            dx, dy = px - mx, py - my
            ln = np.sqrt(np.maximum(dx * dx + dy * dy, 1e-12))
            e = ((mx + px) * 0.5 * dx + (my + py) * 0.5 * dy) / ln
            if round > 0:
                e2 = np.where(e < edge, edge, np.minimum(e2, e))
            edge = np.minimum(edge, e)
    if round > 0:                                   # polynomial smooth-min of the two nearest borders
        k = F32(round)
        hh = np.maximum(k - np.abs(e2 - edge), 0) / k
        edge = np.maximum(edge - hh * hh * k * 0.25, 0)
    return Cells(f1=np.sqrt(f1).astype(F32), f2=np.sqrt(f2).astype(F32), edge=edge.astype(F32),
                 id=_hash(gu, gv, seed, 3), rand=_rnd(gu, gv, seed, 4), cx=(U + mx).astype(F32),
                 cy=(V + my).astype(F32), dx=(-mx).astype(F32), dy=(-my).astype(F32), gu=gu, gv=gv, seed=seed)


def voronoi_shade(cells, palette=("#6b6b6b", "#9a958c", "#c2bbb0"), vary=0.15, gap=0.06, gap_color="#2a2622",
                  pick=True):
    """Voronoi region shading: one palette colour per cell (``pick``) or a ramp by cell random, darker borders.

    Returns an (H, W, 3) colour map; ``gap`` = border width in cell units painted with ``gap_color``.
    """
    col = _pick(palette, cells.rand) if pick else _ramp(palette, cells.rand)
    col = col * (1 + vary * (cells.r(1) - 0.5) * 2)[..., None]
    edge = cells.edge
    k = smoothstep(gap * 0.5, gap * 0.5 + 0.05, edge)
    col = col * (0.75 + 0.25 * smoothstep(0, 0.25, edge))[..., None]
    return _lerp(np.broadcast_to(_c(gap_color), col.shape), col, k)


class Strokes:
    """Result of ``strokes``: ``cov`` coverage 0~1, ``tone`` per-stroke random, ``along`` 0 (root) -> 1 (tip),
    ``across`` -1~1, ``height`` (rounded strands, later layers higher). Free-form record: every keyword becomes an
    attribute."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _line_cov(d, half, fw):
    """Anti-aliased coverage of a line of half width ``half`` at distance ``d`` (keeps thin lines visible)."""
    hw = np.maximum(half, 0.5 * fw)
    return clamp01((hw - d) / fw + 0.5) * clamp01(2.0 * half / fw)


def _reach(extent, angle, spread, kmax, cw, ch, jitter):
    """Neighbour-cell offsets that can hold an instance covering a pixel, from the shape's local bbox."""
    x0, x1, y0, y1 = [v * kmax for v in extent]
    a = np.asarray(angle, F32)
    lo, hi = (float(a.min()), float(a.max())) if a.ndim else (float(a), float(a))
    lo, hi = lo - spread, hi + spread
    xs, ys = [], []
    for t in np.linspace(lo, hi, 13):
        r = math.radians(t)
        ca, sa = math.cos(r), math.sin(r)
        for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            xs.append(px * ca - py * sa)
            ys.append(px * sa + py * ca)
    j = 0.5 * jitter + 0.5
    ox = range(int(math.floor(-max(xs) / cw - j)), int(math.ceil(-min(xs) / cw + j)) + 1)
    oy = range(int(math.floor(-max(ys) / ch - j)), int(math.ceil(-min(ys) / ch + j)) + 1)
    return list(ox), list(oy)


def _scatter(c, cell, shape, extent, layers=1, jitter=1.0, angle=0.0, spread=0.0, size=(1.0, 1.0),
             seed=0, keep=1.0, aspect=1.0, stagger=0.0, order=1, U=None, V=None):
    """Scatter instances of ``shape`` on a jittered grid (vectorised over neighbour cells) and composite them.

    shape(lx, ly, k, r) -> (cov, height, along, across) in local coordinates (texture units, x along ``angle``);
    extent = its local bbox (x0, x1, y0, y1) at size 1. angle: degrees, scalar or per-pixel array; spread: random
    +/- degrees per instance; size: (min, max) factor. order: +1 paints lower rows on top, -1 upper rows on top.
    Returns dict cov / tone / height / along / across / r2.
    """
    U = c.U if U is None else U
    V = c.V if V is None else V
    cw, ch = float(cell), float(cell) * aspect
    shp = U.shape
    out = {k: np.zeros(shp, F32) for k in ("cov", "tone", "height", "along", "across", "r2")}
    base = np.deg2rad(np.asarray(angle, F32))
    oxs, oys = _reach(extent, angle, spread, max(size), cw, ch, jitter)
    if stagger:
        oxs = list(range(oxs[0] - 1, oxs[-1] + 2))
    if order < 0:
        oys = oys[::-1]
    cb, sb = np.cos(base).astype(F32), np.sin(base).astype(F32)
    # cull: only pixels within the shape's bounding radius run the (expensive) shape function
    cull = bool(getattr(shape, "cullable", False)) and not np.ndim(c.fw)
    if cull:
        ex0, ex1, ey0, ey1 = [v * max(size) for v in extent]
        rmax2 = F32((math.hypot(max(abs(ex0), abs(ex1)), max(abs(ey0), abs(ey1))) + 2 * float(c.fw)) ** 2)
        flat = {k: v.reshape(-1) for k, v in out.items()}
        cbf = cb.reshape(-1) if cb.ndim else None
        sbf = sb.reshape(-1) if sb.ndim else None
    spr = F32(np.deg2rad(spread) * 2)
    jit = F32(jitter)
    for L in range(layers):
        sd = seed * 131 + L * 7919 + 5
        lu, lv = F32(0.37 * L), F32(0.61 * L)
        gu = (U / F32(cw) + lu).astype(F32)
        gv = (V / F32(ch) + lv).astype(F32)
        fu0 = np.floor(gu)
        fv0 = np.floor(gv)
        for oy in oys:
            fgy = fv0 + F32(oy)
            gy = fgy.astype(np.int64)
            if stagger:
                sh = (gy & 1).astype(F32) * F32(stagger)
                fgx0 = np.floor(gu - sh)
            else:
                sh = F32(0)
                fgx0 = fu0
            for ox in oxs:
                fgx = fgx0 + F32(ox)
                hsh = _hash(fgx.astype(np.int64), gy, sd, 0)
                cx = (fgx + sh + F32(0.5) + jit * (_sub(hsh, 1) - F32(0.5)) - lu) * F32(cw)
                cy = (fgy + F32(0.5) + jit * (_sub(hsh, 2) - F32(0.5)) - lv) * F32(ch)
                dx, dy = U - cx, V - cy
                if cull:
                    idx = np.flatnonzero((dx * dx + dy * dy).reshape(-1) < rmax2)
                    if idx.size == 0:
                        continue
                    hs = hsh.reshape(-1)[idx]
                    dxs, dys = dx.reshape(-1)[idx], dy.reshape(-1)[idx]
                    cbs = cbf[idx] if cbf is not None else cb
                    sbs = sbf[idx] if sbf is not None else sb
                    if spread:
                        j = spr * (_sub(hs, 3) - F32(0.5))
                        cj, sj = np.cos(j), np.sin(j)
                        ca, sa = cbs * cj - sbs * sj, sbs * cj + cbs * sj
                    else:
                        ca, sa = cbs, sbs
                    lx = dxs * ca + dys * sa
                    ly = dys * ca - dxs * sa
                    r1 = _sub(hs, 4)
                    k = F32(size[0]) + F32(size[1] - size[0]) * _sub(hs, 5) if size[1] != size[0] else F32(size[0])
                    cov, hh, al, ac = shape(lx, ly, k, r1)
                    if keep < 1:
                        cov = cov * (_sub(hs, 6) < keep)
                    cov = np.asarray(cov, F32)
                    for key, val in (("tone", r1), ("r2", _sub(hs, 7)), ("height", hh + F32(0.15 * L)),
                                     ("along", al), ("across", ac)):
                        t = flat[key][idx]
                        flat[key][idx] = t + (val - t) * cov
                    t = flat["cov"][idx]
                    flat["cov"][idx] = t + (1 - t) * cov
                    continue
                if spread:
                    j = spr * (_sub(hsh, 3) - F32(0.5))
                    cj, sj = np.cos(j), np.sin(j)
                    ca, sa = cb * cj - sb * sj, sb * cj + cb * sj
                else:
                    ca, sa = cb, sb
                lx = dx * ca + dy * sa
                ly = dy * ca - dx * sa
                r1 = _sub(hsh, 4)
                k = F32(size[0]) + F32(size[1] - size[0]) * _sub(hsh, 5) if size[1] != size[0] else F32(size[0])
                cov, hh, al, ac = shape(lx, ly, k, r1)
                if keep < 1:
                    cov = cov * (_sub(hsh, 6) < keep)
                cov = np.asarray(cov, F32)
                out["tone"] += (r1 - out["tone"]) * cov
                out["r2"] += (_sub(hsh, 7) - out["r2"]) * cov
                out["height"] += (hh + F32(0.15 * L) - out["height"]) * cov
                out["along"] += (al - out["along"]) * cov
                out["across"] += (ac - out["across"]) * cov
                out["cov"] += (1 - out["cov"]) * cov
    return out


_SUBK = [np.uint32(v) for v in (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F, 0x165667B1, 0xD3A2646C,
                                0xFD7046C5, 0xB55A4F09, 0x2C1B3C6D)]


def _sub(h, k):
    """Extra 0~1 random number k derived from an existing hash."""
    with np.errstate(over="ignore"):
        x = h * _SUBK[k]
        x = x ^ (x >> np.uint32(15))
    return (x >> np.uint32(8)).astype(F32) * F32(1.0 / 16777216.0)


def _blade(c, length, width, taper=0.9, bend=0.0, tip=1.0):
    """Shape function for tapered strokes: root at the instance point, growing along +x."""
    fw = c.fw if np.ndim(c.fw) else F32(c.fw)
    ifw = F32(1) / fw

    def shape(lx, ly, k, r):
        Ln = F32(length) * k
        s = lx / Ln
        sc = np.clip(s, F32(0), F32(1))
        if bend:
            ly2 = ly - (F32(2 * bend) * Ln * (r - F32(0.5))) * sc * sc
        else:
            ly2 = ly
        half = F32(0.5 * width) * k * (F32(1) - F32(taper) * (sc if tip == 1 else sc ** (1.0 / tip)))
        a = np.abs(ly2)
        hw = np.maximum(half, F32(0.5) * fw)
        cov = np.clip((hw - a) * ifw + F32(0.5), 0, 1) * np.clip(F32(2) * half * ifw, 0, 1)
        cov = cov * np.clip(lx * ifw + F32(0.5), 0, 1) * np.clip((Ln - lx) * ifw + F32(0.5), 0, 1)
        across = np.clip(ly2 / np.maximum(half, F32(1e-3)), F32(-1), F32(1))
        hgt = np.sqrt(np.maximum(F32(1) - across * across, 0)) * F32(0.6) + sc * F32(0.4)
        return cov, hgt, sc, across
    shape.cullable = True
    return shape


def strokes(h, w, angle=-90.0, length=30.0, width=2.0, density=1.0, spread=20.0, bend=0.3, taper=0.8,
            layers=3, scale=1.0, seed=0, uv=None, mask=None):
    """Many short oriented strokes following a direction field (fur, grass, hair, brush marks, straw).

    angle: degrees (0 = right, -90 = up), a number or an (H, W) direction map in degrees; length / width in
    texture units; density > 1 packs strokes tighter; spread = random +/- degrees per stroke; bend = curvature.
    Returns ``Strokes(cov, tone, along, across, height)`` -> colour them with a palette, e.g.
    ``col = mix([(0, '#2a1a0e'), (1, '#c79a62')], s.along)``.
    """
    def gen(c):
        ang = angle
        if np.ndim(ang) == 2:
            ang = c.crop(np.asarray(ang, F32))
        cell = max(width * 1.6, length * 0.35) / max(density, 1e-3) ** 0.5
        o = _scatter(c, cell, _blade(c, length, width, taper, bend), _blade_extent(length, width, bend),
                     layers=layers, angle=ang, spread=spread, size=(0.75, 1.15), seed=c.s(1))
        return Maps(color=np.zeros(c.U.shape + (3,), F32), height=o["height"], extra=o)
    m = _run(gen, h, w, scale, seed, uv, mask, LIGHT, 1.0, True, 1, {})
    e = m.extra
    return Strokes(cov=e["cov"], tone=e["tone"], along=e["along"], across=e["across"], height=e["height"])


# ======================================================================== coordinates & lighting

def plane_uv(h, w, horizon=0.4, near=1.0, center=0.5, shift=(0.0, 0.0)):
    """(U, V) texture coordinates of a flat ground plane seen in perspective (pass as ``uv=``).

    horizon: eye level as a fraction of the height (the plane is below it); near: texture size at the bottom edge
    relative to a flat swatch (2 = bigger); center: vanishing point x fraction; shift: (du, dv) texture offset.
    V grows towards the viewer. Rows at or above the horizon get far-away coordinates -> mask them out.
    """
    S = h / 1080.0
    hy = horizon * h
    x = np.arange(w, dtype=F32) - center * w
    y = np.maximum(np.arange(h, dtype=F32) + 0.5 - hy, 0.5)
    d = max(h - hy, 1.0)
    k = d / (S * near)
    U = (x[None, :] / y[:, None]) * k + shift[0]
    V = k - k * d / y + shift[1]                       # 0 at the bottom edge, -> -inf at the horizon
    V = np.broadcast_to(V[:, None], (h, w))
    return np.ascontiguousarray(U, F32), np.ascontiguousarray(V, F32)


def polar_uv(h, w, cx, cy, radius=None, angle0=-180.0, ring_scale=1.0):
    """(U, V) around a centre (pixels): U = arc length along circles (texture units), V = distance from the
    ``radius`` circle outwards. Bricks laid with it make an arch with radial joints."""
    S = h / 1080.0
    Y, X = np.mgrid[0:h, 0:w].astype(F32)
    dx, dy = X - cx, Y - cy
    r = np.sqrt(dx * dx + dy * dy)
    a = np.arctan2(dy, dx) - np.deg2rad(angle0)
    a = np.mod(a, 2 * np.pi)
    r0 = r.mean() if radius is None else radius
    U = a * r0 / S * ring_scale
    V = (r - r0) / S
    return U.astype(F32), V.astype(F32)


def _light_vec(light):
    l = [float(v) for v in light]
    if len(l) == 2:
        d = math.hypot(l[0], l[1]) or 1.0
        l = [l[0] / d * 0.77, l[1] / d * 0.77, 0.64]
    v = np.array(l, F32)
    return v / max(float(np.linalg.norm(v)), 1e-6)


def _grad(hu, fwx, fwy):
    if hu.shape[0] < 2 or hu.shape[1] < 2:
        return np.zeros_like(hu), np.zeros_like(hu)
    gy, gx = np.gradient(hu)
    return (gx / fwx).astype(F32), (gy / fwy).astype(F32)


def _shade(hu, L, fwx, fwy, ambient=0.35, rough=24.0):
    """Height (texture units) -> (diffuse multiplier with flat = 1, specular, normal z)."""
    gx, gy = _grad(hu, fwx, fwy)
    nz = 1.0 / np.sqrt(1 + gx * gx + gy * gy)
    nx, ny = -gx * nz, -gy * nz
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    diff = (ambient + (1 - ambient) * np.maximum(ndl, 0) / L[2]).astype(F32)
    hv = L + np.array([0, 0, 1], F32)
    hv = hv / np.linalg.norm(hv)
    ndh = np.maximum(nx * hv[0] + ny * hv[1] + nz * hv[2], 0)
    spec = (ndh ** rough).astype(F32)
    return diff, spec, ny


def relief(height, light=LIGHT, depth=4.0, color=None, shine=0.0, rough=24.0, ambient=0.35):
    """Light a height map: returns a brightness multiplier (flat = 1), or the lit colour map when ``color`` is given.

    height: (H, W) any range (0~1 typical); depth: how many 1080p pixels a height step of 1 rises;
    light: direction towards the light (x, y) or (x, y, z); shine: specular strength; rough: specular exponent.
    """
    hgt = np.asarray(height, F32)
    S = hgt.shape[0] / 1080.0
    fw = 1.0 / S
    diff, spec, _ = _shade(hgt * depth, _light_vec(light), fw, fw, ambient, rough)
    if color is None:
        return diff + spec * shine
    col = np.asarray(rgb(color) if not isinstance(color, np.ndarray) else color, F32)
    return (col * diff[..., None] + spec[..., None] * shine).astype(F32)


# ======================================================================== context & runner

class Maps:
    """Texture maps: ``color`` (albedo), ``height`` (0~1), ``alpha``, ``shine``, ``emit`` (added light), ``lit``."""

    def __init__(self, color, height=None, alpha=None, shine=0.0, rough=24.0, emit=None, depth=4.0,
                 metal=False, ambient=0.35, lit=None, extra=None):
        self.color, self.height, self.alpha, self.shine, self.rough = color, height, alpha, shine, rough
        self.emit, self.depth, self.metal, self.ambient, self.lit, self.extra = emit, depth, metal, ambient, lit, extra

    def __iter__(self):
        return iter((self.lit if self.lit is not None else self.color, self.height))


class _Ctx:
    """Coordinates and helpers handed to every texture generator."""

    def __init__(self, U, V, fwx, fwy, seed, sep, box, full, light, bump, ru, rv):
        self.U, self.V = U, V
        self.fwx, self.fwy = fwx, fwy
        self.fw = np.maximum(fwx, fwy) if np.ndim(fwx) else max(fwx, fwy)
        self.seed = int(seed)
        self.sep = sep
        self.box = box
        self.full = full
        self.L = _light_vec(light)
        self.bump = bump
        self.ru, self.rv = ru, rv
        self.h, self.w = U.shape

    def s(self, k):
        return (self.seed * 7919 + int(k) * 1009 + 13) & 0x7FFFFFFF

    def crop(self, a):
        """Crop a full-canvas (H, W[, C]) array to this region (and supersampling)."""
        a = np.asarray(a, F32)
        y0, y1, x0, x1, ss = self.box
        if a.shape[:2] != self.full:
            return a
        a = a[y0:y1, x0:x1]
        if ss > 1:
            a = np.repeat(np.repeat(a, ss, 0), ss, 1)
        return a

    def n(self, size, octaves=4, seed=0, aspect=1.0, mode="plain", angle=0.0, gain=0.5, U=None, V=None):
        """fBm noise with feature size ``size`` texture units (aspect > 1 stretches along V)."""
        sd = self.s(seed + 500)
        if U is None and V is None and not angle and self.sep is not None:
            ox, oy, ppu = self.sep
            return fbm2d(self.h, self.w, size * ppu, octaves, 2.0, gain, seed=sd, offset=(ox, oy),
                         aspect=aspect, mode=mode)
        U = self.U if U is None else U
        V = self.V if V is None else V
        if angle:
            U, V = _rot(U, V, angle)
        return fbm2d_at(U / size, V / (size * aspect), octaves, 2.0, gain, seed=sd, mode=mode)

    def cells(self, size, jitter=1.0, seed=0, aspect=1.0, round=0.0, warp=0.0, U=None, V=None):
        """Voronoi cells of ``size`` texture units (aspect > 1 = taller cells); warp roughens the borders."""
        U = self.U if U is None else U
        V = self.V if V is None else V
        if warp:
            U = U + warp * size * self.n(size * 0.6, 3, seed + 71, U=U, V=V)
            V = V + warp * size * self.n(size * 0.6, 3, seed + 73, U=U, V=V)
        return cellular(U / size, V / (size * aspect), jitter, self.s(seed + 900), round)

    def rot(self, angle, U=None, V=None):
        return _rot(self.U if U is None else U, self.V if V is None else V, angle)

    def inside(self, d, soft=0.0):
        """Anti-aliased 0 to 1 step where the signed distance ``d`` (texture units, > 0 inside) crosses 0."""
        return clamp01(d / (self.fw + soft) + 0.5)

    def fade(self, feature):
        """1 while ``feature`` (texture units) is well resolved, -> 0 once it shrinks below ~2 pixels."""
        return clamp01(feature / (2.5 * self.fw) - 0.6)

    def shade(self, height, depth, ambient=0.35, rough=24.0):
        return _shade(np.asarray(height, F32) * depth * self.bump, self.L, self.fwx, self.fwy, ambient, rough)

    def grad(self, height):
        return _grad(np.asarray(height, F32), self.fwx, self.fwy)

    def center(self):
        return 0.5 * (float(self.U.min()) + float(self.U.max())), 0.5 * (float(self.V.min()) + float(self.V.max()))


def _rot(U, V, angle):
    if not angle:
        return U, V
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    return (U * ca + V * sa).astype(F32), (-U * sa + V * ca).astype(F32)


def _down(a, ss):
    if ss <= 1 or a is None:
        return a
    h, w = a.shape[0] // ss, a.shape[1] // ss
    a = a[:h * ss, :w * ss]
    sh = (h, ss, w, ss) + a.shape[2:]
    return a.reshape(sh).mean(axis=(1, 3)).astype(F32)


def _compose(c, m):
    """Albedo + height -> lit colour (relief lighting, specular, emission)."""
    col = np.asarray(m.color, F32)
    if col.ndim == 1:
        col = np.broadcast_to(col, c.U.shape + (3,)).astype(F32)
    if m.lit is not None:
        out = m.lit
    elif m.height is not None and c.bump:
        diff, spec, ny = c.shade(m.height, m.depth, m.ambient, m.rough)
        if m.metal:
            env = (0.55 + 0.6 * smoothstep(0.5, -0.5, ny))[..., None]
            out = col * (0.35 * diff[..., None] + 0.65 * env) + col * spec[..., None] * (1.6 * m.shine)
        else:
            out = col * diff[..., None] + spec[..., None] * m.shine
    else:
        out = col
    if m.emit is not None:
        out = out + m.emit
    return np.asarray(out, F32)


def _run(gen, h, w, scale, seed, uv, mask, light, bump, maps, ss, kw):
    h, w = int(h), int(w)
    ss = max(int(ss), 1)
    S = h / 1080.0
    y0, y1, x0, x1 = 0, h, 0, w
    mk = None
    if mask is not None:
        mk = np.asarray(mask, F32)
        ys = np.nonzero(mk.max(1) > 1e-4)[0]
        xs = np.nonzero(mk.max(0) > 1e-4)[0]
        if len(ys) == 0:
            z = np.zeros((h, w, 3), F32)
            return Maps(z, np.zeros((h, w), F32), np.zeros((h, w), F32), lit=z) if maps else z
        y0, y1 = max(int(ys[0]) - 2, 0), min(int(ys[-1]) + 3, h)
        x0, x1 = max(int(xs[0]) - 2, 0), min(int(xs[-1]) + 3, w)
    rh, rw = (y1 - y0) * ss, (x1 - x0) * ss
    if uv is not None:
        Uf = np.asarray(uv[0], F32)[y0:y1, x0:x1] / scale
        Vf = np.asarray(uv[1], F32)[y0:y1, x0:x1] / scale
        if ss > 1:
            Uf = _f.resize(Uf, rw, rh, "bilinear")
            Vf = _f.resize(Vf, rw, rh, "bilinear")
        if Uf.shape[0] >= 2 and Uf.shape[1] >= 2:
            Uy, Ux = np.gradient(Uf)
            Vy, Vx = np.gradient(Vf)
            fwx = np.maximum(np.hypot(Ux, Vx), 1e-5).astype(F32)
            fwy = np.maximum(np.hypot(Uy, Vy), 1e-5).astype(F32)
        else:
            fwx = fwy = 1.0 / (S * ss)
        U, V, sep = Uf, Vf, None
    else:
        ppu = S * scale * ss
        ox = x0 * ss + 0.5 - 0.5 * ss
        oy = y0 * ss + 0.5 - 0.5 * ss
        xs_ = (np.arange(rw, dtype=F32) + ox) / ppu
        ys_ = (np.arange(rh, dtype=F32) + oy) / ppu
        U = np.broadcast_to(xs_[None, :], (rh, rw)).astype(F32)
        V = np.broadcast_to(ys_[:, None], (rh, rw)).astype(F32)
        fwx = fwy = 1.0 / ppu
        sep = (ox, oy, ppu)
    ru = np.broadcast_to(((np.arange(rw, dtype=F32) + 0.5) / rw)[None, :], (rh, rw))
    rv = np.broadcast_to(((np.arange(rh, dtype=F32) + 0.5) / rh)[:, None], (rh, rw))
    c = _Ctx(U, V, fwx, fwy, seed, sep, (y0, y1, x0, x1, ss), (h, w), light, bump, ru, rv)
    kw = dict(kw)
    if "tint" not in kw and "tints" in kw:
        kw["tint"] = kw.pop("tints")
    tint = kw.pop("tint", None)
    kw = _check_options(gen, kw)
    m = gen(c, **kw)
    if tint is not None:
        _apply_tint(m, tint)
    if m.height is not None:
        m.height = np.asarray(m.height, F32)
    lit = _compose(c, m)
    col = np.broadcast_to(np.asarray(m.color, F32), lit.shape)
    parts = dict(lit=lit, color=col, height=m.height, alpha=m.alpha, emit=m.emit)
    if ss > 1:
        parts = {k: _down(np.asarray(v, F32), ss) if v is not None else None for k, v in parts.items()}
    if (y0, y1, x0, x1) == (0, h, 0, w):
        full = parts
    else:
        full = {}
        for k, v in parts.items():
            if v is None:
                full[k] = None
                continue
            v = np.asarray(v, F32)
            z = np.zeros((h, w) + v.shape[2:], F32)
            z[y0:y1, x0:x1] = v
            full[k] = z
    if not maps:
        return np.ascontiguousarray(full["lit"], F32)
    extra = m.extra
    if isinstance(extra, dict):
        ex = {}
        for k, v in extra.items():
            v = _down(np.asarray(v, F32), ss)
            if (y0, y1, x0, x1) != (0, h, 0, w):
                z = np.zeros((h, w) + v.shape[2:], F32)
                z[y0:y1, x0:x1] = v
                v = z
            ex[k] = v
        extra = ex
    return Maps(full["color"], full["height"], full["alpha"], m.shine, m.rough, full["emit"], m.depth, m.metal,
                m.ambient, lit=full["lit"], extra=extra)


_DOC_TAIL = ("\n\n    Common: h, w, scale=1 (feature size factor), seed=0, uv=(U, V) texture coordinates, mask (only its box is "
             "computed),\n    light=(-0.6, -0.7), bump=1 (relief strength), maps=False (True -> Maps), ss=1 (supersampling),\n"
             "    tint=colour (recolour any material, keeping its light and dark structure)."
             "\n    Returns the lit (H, W, 3) colour map.")


def _apply_tint(m, tint):
    """Recolour a material: keep its light / dark structure, take hue and saturation from ``tint``.
    ``tint`` is a colour, or ``(colour, amount)`` with amount 0~1."""
    amount = 1.0
    if isinstance(tint, tuple) and len(tint) == 2 and not isinstance(tint[0], (int, float)):
        tint, amount = tint
    t = _c(tint)
    col = np.asarray(m.color, F32)
    lum = (col * np.array([0.3, 0.59, 0.11], F32)).sum(-1, keepdims=True)
    tl = float((t * np.array([0.3, 0.59, 0.11], F32)).sum())
    new = t * (lum / max(tl, 1e-3))
    m.color = col + (new - col) * float(amount)


def _check_options(gen, kw):
    """Check the keywords against the options of this texture: unknown names raise a TypeError naming the closest
    option, aliased spellings (colour, w ...) are accepted. Returns the keywords with canonical names."""
    if not kw:
        return kw
    gen = getattr(gen, "_base", gen)               # a preset (``_alias``) takes the options of its base texture
    try:
        sig = inspect.signature(gen)
    except (TypeError, ValueError):
        return kw
    ps = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in ps.values()):
        return kw
    ok = [n for n, p in ps.items() if n != "c" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    return _reject(getattr(gen, "__name__", "texture"), kw, ok + ["tint"])


def _options_doc(gen):
    """Own options of a generator as a doc line: ``Options: size=46.0, palette=None, ...``."""
    try:
        ps = [p for p in inspect.signature(gen).parameters.values() if p.name != "c" and p.kind != p.VAR_KEYWORD]
    except (TypeError, ValueError):
        return ""
    if not ps:
        return ""
    return "\n\n    Options: " + ", ".join(f"{p.name}={p.default!r}" for p in ps)


def _tex(name, family, keywords=""):
    """Register a texture generator ``gen(c, **opts) -> Maps`` under ``name``."""
    def deco(gen):
        def fn(h, w, scale=1.0, seed=0, uv=None, mask=None, light=LIGHT, bump=1.0, maps=False, ss=1, **kw):
            return _run(gen, h, w, scale, seed, uv, mask, light, bump, maps, ss, kw)
        fn.__name__ = name
        fn.__qualname__ = name
        fn.__doc__ = (gen.__doc__ or name).strip() + _options_doc(gen) + _DOC_TAIL
        fn.family = family
        TEXTURES[name] = fn
        FAMILIES.setdefault(family, []).append(name)
        _META[name] = (family, keywords)
        return fn
    return deco


def _alias(name, base, family, keywords="", **preset):
    """Register a preset of another texture under a new name."""
    def gen(c, **kw):
        kk = dict(preset)
        kk.update(kw)
        return _GENS[base](c, **kk)
    gen.__doc__ = f"{name.replace('_', ' ').capitalize()}: ``{base}`` with {preset}."
    gen._base = _GENS[base]
    _tex(name, family, keywords)(gen)


_GENS = {}


def _gen(name, family, keywords=""):
    """Decorator: register and keep the raw generator for presets."""
    def deco(g):
        _GENS[name] = g
        return _tex(name, family, keywords)(g)
    return deco


# ======================================================================== shared geometry

PAL = dict(
    wood=["#4a2a12", "#7a4a24", "#a8713f", "#cf9a62", "#e6bd88"],
    oak=["#5b3a1c", "#8a6236", "#b48a56", "#d3ad7a"],
    cobble=["#6d6a66", "#827c73", "#5b5f63", "#8f8577", "#746454", "#9a948a"],
    pebble=["#8b8378", "#a59a88", "#6e6a66", "#b8ad9a", "#8a6f5a", "#5e5a56", "#c9bfae"],
    slate=["#6f6a61", "#857a68", "#5d6168", "#93836a", "#7a6c5e", "#8a8a84"],
    brick=["#5e2418", "#7e3320", "#9a4428", "#ad5a36", "#bf7048"],
    stone=["#7b766c", "#8f887b", "#6b6760", "#9d9585", "#857b69", "#a8a191"],
    soil=["#24180f", "#3b2a1b", "#55402b", "#705539"],
    copper=["#4a2010", "#8c4826", "#c27a4c", "#e8b48a"],
    gold=["#5c3d0c", "#a2761f", "#d8ad41", "#f4dc8a"],
    autumn=["#b8401c", "#d9772a", "#e8a838", "#9a2c1a", "#c95a24", "#7d5a26", "#d9b44a"],
    grass=["#16260d", "#2f5218", "#4f7d26", "#86ad44", "#b9cf6e"],
)


def _blade_extent(length, width, bend=0.0):
    b = abs(bend) * length + width
    return (-width, length + width, -b, b)


def _courses(U, V, pitch, height, offset=0.5, jitter=0.0, seed=0, random_offset=False):
    """Rows of pieces (bricks, planks, ashlar). Returns (col, row, fx, fy, length) -> fx, fy from the piece corner."""
    row = np.floor(V / height)
    fy = V - row * height
    ri = row.astype(np.int64)
    if random_offset:
        off = _rnd(ri, 0, seed, 11) * pitch
    else:
        off = np.mod(row * offset, 1.0) * pitch
    u = U + off
    k0 = np.floor(u / pitch).astype(np.int64)
    if jitter:
        def jit(k):
            return (_rnd(k, ri, seed, 12) - 0.5) * jitter * pitch
        s0 = k0 * pitch + jit(k0)
        s1 = (k0 + 1) * pitch + jit(k0 + 1)
        k = np.where(u < s0, k0 - 1, np.where(u >= s1, k0 + 1, k0))
        start = k * pitch + jit(k)
        end = (k + 1) * pitch + jit(k + 1)
    else:
        k = k0
        start = k * pitch
        end = start + pitch
    return k, ri, (u - start).astype(F32), fy.astype(F32), (end - start).astype(F32)


def _hexgrid(U, V, R):
    """Pointy-top hexagons of circumradius R -> (q, r, dx, dy, edge distance)."""
    q = (_SQ3 / 3 * U - V / 3) / R
    r = (2.0 / 3 * V) / R
    x, z = q, r
    y = -x - z
    rx, ry, rz = np.rint(x), np.rint(y), np.rint(z)
    ddx, ddy, ddz = np.abs(rx - x), np.abs(ry - y), np.abs(rz - z)
    c1 = (ddx > ddy) & (ddx > ddz)
    c2 = ~c1 & (ddy > ddz)
    rx = np.where(c1, -ry - rz, rx)
    rz = np.where(~c1 & ~c2, -rx - ry, rz)
    cx = R * (_SQ3 * rx + _SQ3 / 2 * rz)
    cy = R * 1.5 * rz
    dx, dy = U - cx, V - cy
    ri = R * _SQ3 / 2
    ax = np.abs(dx)
    e = ri - np.maximum(ax, np.maximum(np.abs(0.5 * dx + 0.866 * dy), np.abs(-0.5 * dx + 0.866 * dy)))
    return rx.astype(np.int64), rz.astype(np.int64), dx.astype(F32), dy.astype(F32), e.astype(F32)


def _speck(c, size=1.6, seed=0):
    """Fine grain (about -1~1) that fades out when it gets smaller than a pixel."""
    return c.n(size, 2, seed) * c.fade(size)


def _fibres(c, size=8.0, seed=0, n=4, aspect=0.12):
    """Criss-cross fibre texture (paper, felt): ridged streaks at several angles."""
    acc = 0.0
    for k in range(n):
        a = 180.0 * (k + 0.37 * (c.seed % 7)) / n
        acc = acc + c.n(size, 2, seed + 11 * k, aspect=aspect, angle=a, mode="ridged")
    return (acc / n) * c.fade(size * aspect * 2)


def _tile_grid(c, sx, sy, grout, offset=0.0, U=None, V=None):
    """Rectangular tiles with joints: (col, row, inside distance d (units), fx, fy)."""
    U = c.U if U is None else U
    V = c.V if V is None else V
    k, row, fx, fy, L = _courses(U, V, sx, sy, offset)
    d = np.minimum(np.minimum(fx, L - fx), np.minimum(fy, sy - fy)) - grout * 0.5
    return k, row, d.astype(F32), fx, fy


# ======================================================================== wood

def _grain(c, U, V, seed, spacing=14.0, knots=0.3, knot_size=16.0, warp=1.0):
    """Flat-sawn grain along U -> (dark 0~1, height, knot mask)."""
    n1 = fbm2d_at(U / 520.0, V / 150.0, 3, seed=seed)
    n2 = fbm2d_at(U / 160.0, V / 34.0, 3, seed=seed + 1)
    r = V + warp * (80.0 * n1 + 9.0 * n2)
    core = np.zeros(U.shape, F32)
    if knots > 0:
        ks = 300.0
        cel = cellular(U / ks, V / (ks * 0.5), 0.7, seed + 2)
        has = (cel.rand < knots).astype(F32)
        kr = knot_size * (0.6 + 0.8 * cel.r(1))
        dxk = cel.dx * ks / 1.6
        dyk = cel.dy * ks * 0.5
        d = np.sqrt(dxk * dxk + dyk * dyk)
        infl = np.exp(-(d / (kr * 2.8)) ** 2) * has
        r = r + infl * kr * 2.4 * np.tanh(dyk / (kr * 0.6))
        core = (1 - smoothstep(kr * 0.8, kr * 1.05, d)) * has
    ring = r / spacing
    fr = ring - np.floor(ring)
    fwr = c.fw / spacing * 1.5
    t = fr ** 2.2 * (1 - smoothstep(1 - 0.08 - fwr, 1.0, fr))
    pores = fbm2d_at(U / 70.0, V / 2.4, 2, seed=seed + 3)
    pore = smoothstep(0.2, 0.7, pores) * c.fade(2.4)
    t = t * 0.85 + pore * 0.3
    if knots > 0:
        kt = d / (kr * 0.3)
        kt = kt - np.floor(kt)
        kdark = 0.75 + 0.25 * kt ** 2 + 0.25 * (1 - smoothstep(0, kr * 0.5, d))
        rim = np.exp(-((d - kr * 0.9) / (kr * 0.15)) ** 2) * has
        t = t + (kdark - t) * core + rim * 0.5
    h = 0.6 * t - 0.4 * pore
    return np.clip(t, 0, 1.4).astype(F32), h.astype(F32), core


@_gen("planks", "wood", "wooden floor floorboards boards plank deck timber hardwood parquet laminate")
def _planks(c, width=110.0, length=760.0, gap=1.8, angle=0.0, palette=None, vary=0.45, grain=1.0, knots=0.25,
            nails=True, bevel=3.0, spacing=13.0):
    """Wooden plank floor: boards with dark joints, staggered ends, per-plank tone, grain with rings and knots.
    width / length: board size; angle: 90 = vertical boards (doors, fences); nails: nail heads at board ends."""
    pal = palette or PAL["wood"]
    U, V = c.rot(angle)
    k, row, fx, fy, Lp = _courses(U, V, length, width, jitter=0.7, seed=c.s(1), random_offset=True)
    d = np.minimum(np.minimum(fx, Lp - fx), np.minimum(fy, width - fy)) - gap * 0.5
    board = c.inside(d)
    pr = _rnd(k, row, c.s(2), 0)
    gu = fx + _rnd(k, row, c.s(2), 1) * 4000.0
    gv = fy + _rnd(k, row, c.s(2), 2) * 600.0
    sp = spacing * (0.7 + 0.6 * _rnd(k, row, c.s(2), 3))
    t, gh, core = _grain(c, gu, gv, c.s(3), sp, knots)
    light = 0.62 + vary * (pr - 0.5) + 0.06 * c.n(300, 3, 4) - 0.5 * grain * t
    col = _ramp(pal, clamp01(light))
    col = col * (0.8 + 0.2 * smoothstep(0, 14, d))[..., None]
    if nails:
        e = np.minimum(fx, Lp - fx) - 22.0
        nd = np.sqrt(e * e + np.minimum((fy - width * 0.28) ** 2, (fy - width * 0.72) ** 2))
        nail = 1 - c.inside(nd - 2.6)
        col = _lerp(col, np.broadcast_to(_c("#2a221c"), col.shape), nail * 0.85)
    col = _lerp(np.broadcast_to(_c("#140c07"), col.shape), col, board)
    hgt = board * (0.55 + 0.45 * smoothstep(0, bevel, d)) + 0.06 * grain * gh * board
    return Maps(col, hgt, depth=3.0, shine=0.12, rough=18)


@_gen("wood_grain", "wood", "wood grain timber board tabletop oak walnut knots rings plank surface")
def _wood_grain(c, angle=0.0, spacing=16.0, knots=0.4, palette=None, knot_size=18.0):
    """Fine wood grain on one wide board: growth rings, pores along the grain and knots the rings flow around."""
    pal = palette or PAL["wood"]
    U, V = c.rot(angle)
    t, gh, core = _grain(c, U, V, c.s(1), spacing, knots, knot_size)
    light = 0.6 + 0.08 * c.n(400, 3, 2) - 0.48 * t
    return Maps(_ramp(pal, clamp01(light)), gh, depth=1.2, shine=0.15, rough=20)


@_gen("end_grain", "wood", "end grain log slice tree rings stump cross section cut log firewood")
def _end_grain(c, spacing=9.0, palette=None, cracks=4, center=None, radius=None, bark=True):
    """End grain / log slice: concentric growth rings, rays, radial cracks, dark heartwood, bark ring.
    center: (u, v) in texture units (default: centre of the region); radius: log radius (default fills it)."""
    pal = palette or ["#5a3216", "#9a6434", "#c99560", "#e8c79a"]
    cu, cv = center if center is not None else c.center()
    dx, dy = c.U - cu, c.V - cv
    d = np.sqrt(dx * dx + dy * dy) + 1e-4
    th = np.arctan2(dy, dx)
    R = radius or 0.5 * min(float(c.U.max() - c.U.min()), float(c.V.max() - c.V.min())) * 0.98
    wob = fbm2d_at(np.cos(th) * 1.3 + 5, np.sin(th) * 1.3 + 5, 3, seed=c.s(1))
    rr = d * (1 + 0.06 * wob) + 4.0 * c.n(70, 3, 2)
    ring = rr / spacing
    fr = ring - np.floor(ring)
    t = fr ** 2.0 * (1 - smoothstep(0.9 - c.fw / spacing * 1.5, 1.0, fr))
    ray = np.abs(np.mod(th / (2 * np.pi) * 140 + 0.4 * c.n(30, 2, 3), 1.0) - 0.5) * 2
    rays = smoothstep(0.85, 1.0, ray) * 0.25 * c.fade(d * 2 * np.pi / 140)
    heart = 1 - smoothstep(R * 0.5, R * 0.7, rr)
    light = 0.72 - 0.4 * t - 0.2 * heart + 0.1 * rays + 0.05 * c.n(120, 3, 4)
    col = _ramp(pal, clamp01(light))
    hgt = 0.3 * t
    crack = np.zeros_like(d)
    for i in range(int(cracks)):
        a0 = (_rnd(i, 1, c.s(5), 0) * 2 - 1) * np.pi
        L0 = 0.35 + 0.6 * float(_rnd(i, 2, c.s(5), 0))
        da = np.angle(np.exp(1j * (th - a0)))
        perp = np.abs(da) * d
        wdt = 0.4 + 3.5 * clamp01((d / R - (1 - L0)) / L0)
        crack = np.maximum(crack, (1 - c.inside(perp - wdt)) * smoothstep(R * (1 - L0), R * (1 - L0) + 6, d))
    pith = 1 - c.inside(d - spacing * 0.35)
    col = _lerp(col, np.broadcast_to(_c("#24140a"), col.shape), np.maximum(crack, pith * 0.8))
    hgt = hgt - crack * 0.8
    alpha = None
    if bark:
        bw = R * 0.07
        bz = smoothstep(R - bw - 1, R - bw + 1, d)
        bn = c.n(10, 4, 6, mode="ridged")
        bcol = _ramp(["#1e140d", "#3d2b1e", "#5e4733"], clamp01(0.3 + 0.7 * bn))
        col = _lerp(col, bcol, bz)
        hgt = hgt + bz * (0.6 + 0.6 * bn)
        col = col * (1 - 0.3 * np.exp(-((d - (R - bw)) / 2.5) ** 2))[..., None]
        alpha = c.inside(R - d)
    return Maps(col, hgt, alpha=alpha, depth=2.0, shine=0.05)


@_gen("bark", "wood", "tree bark trunk oak bark pine bark rough bark")
def _bark(c, size=38.0, palette=None, lichen=0.35):
    """Tree bark: vertical plates split by deep furrows, horizontal cracks, lichen patches."""
    pal = palette or ["#1b130d", "#3a2c21", "#5d4c3d", "#7f705f", "#9a8d7c"]
    wv = c.n(300, 3, 1, aspect=4)
    cel = c.cells(size, 0.9, 2, aspect=5.5, round=0.12, U=c.U + 30 * wv)
    e = cel.edge * size
    plate = smoothstep(0.2, 3.5, e)
    dome = np.sqrt(clamp01(e / (size * 0.45)))
    fib = 0.6 * c.n(7, 3, 3, aspect=14, mode="ridged") + 0.4 * c.n(18, 3, 8, aspect=10)
    hcr = smoothstep(0.86, 0.97, c.n(34, 3, 4, aspect=0.18, mode="ridged")) * smoothstep(2, 8, e)
    hgt = plate * (0.5 + 0.35 * dome + 0.3 * fib) - 0.45 * hcr
    light = 0.1 + 0.5 * plate * (0.75 + 0.5 * (cel.rand - 0.5)) + 0.12 * dome * plate + 0.3 * fib * plate - 0.3 * hcr
    col = _ramp(pal, clamp01(light))
    if lichen:
        lm = smoothstep(0.25, 0.5, c.n(110, 4, 5) + 0.3 * (lichen - 0.35)) * plate * dome
        lcol = _ramp(["#7d8270", "#a3a892", "#c2c5b2"], clamp01(0.5 + 0.6 * c.n(5, 2, 6)))
        col = _lerp(col, lcol, lm * 0.7)
    return Maps(col, hgt, depth=9.0, shine=0.0)


# ======================================================================== stone

def _stones(c, size, gap, palette, mortar, round=0.15, jitter=0.85, warp=0.05, dome=0.4, tilt=0.6, rough=0.08,
            aspect=1.0, seed=1, speck=0.12, mortar_noise=0.12):
    """Shared engine for cobbles, pebbles, gravel, flagstones, rubble: Voronoi stones with rounded shading."""
    cel = c.cells(size, jitter, seed, aspect=aspect, round=round, warp=warp)
    unit = size * min(aspect, 1.0)
    eu = cel.edge * unit - gap * unit * 0.5
    stone = c.inside(eu)
    dm = np.sqrt(clamp01(eu / (dome * unit)))
    tl = ((cel.r(1) - 0.5) * cel.dx + (cel.r(2) - 0.5) * cel.dy * aspect) * tilt
    rn = c.n(max(size * 0.12, 3), 3, seed + 3)
    hgt = stone * (0.3 + 0.7 * dm + tl + rough * rn)
    col = _pick(palette, cel.rand) * (0.82 + 0.36 * cel.r(3))[..., None]
    col = col * (1 + speck * _speck(c, 2.0, seed + 4) + 0.08 * rn)[..., None]
    mcol = _c(mortar) * (1 + mortar_noise * c.n(5, 3, seed + 5))[..., None]
    col = col * (0.72 + 0.28 * smoothstep(0, unit * 0.18, eu))[..., None]
    return _lerp(mcol, col, stone), hgt, cel, eu


@_gen("cobblestone", "stone", "cobblestone cobbles cobbled street old town paving stones rounded setts")
def _cobblestone(c, size=72.0, gap=0.1, palette=None, mortar="#3a332b", round=0.18, wet=0.0):
    """Cobblestones: rounded stones with per-stone colour and shading, set in dark joints. wet: 0~1 sheen."""
    col, hgt, cel, eu = _stones(c, size, gap, palette or PAL["cobble"], mortar, round=round)
    return Maps(col, hgt, depth=size * 0.2, shine=0.08 + 0.6 * wet, rough=30 + 40 * wet)


@_gen("pebbles", "stone", "pebbles beach pebbles river stones gravel path smooth stones shingle")
def _pebbles(c, size=30.0, gap=0.2, palette=None, mortar="#b5a385", wet=0.0):
    """Pebbles: smooth, very rounded stones of mixed colour on sand."""
    col, hgt, cel, eu = _stones(c, size, gap, palette or PAL["pebble"], mortar, round=0.32, jitter=1.0,
                                warp=0.08, dome=0.55, tilt=0.2, rough=0.03, speck=0.08)
    sand = _speck(c, 1.5, 9)
    col = col * (1 + 0.1 * sand * (1 - c.inside(eu)))[..., None]
    return Maps(col, hgt, depth=size * 0.3, shine=0.25 + 0.6 * wet, rough=26)


@_gen("flagstone", "stone", "flagstone slate paving patio irregular stone slabs crazy paving")
def _flagstone(c, size=170.0, gap=0.035, palette=None, mortar="#6f675c"):
    """Flagstones: large irregular flat slabs (slate, sandstone) with thin mortar joints and layered surfaces."""
    col, hgt, cel, eu = _stones(c, size, gap, palette or PAL["slate"], mortar, round=0.03, jitter=1.0, warp=0.14,
                                dome=0.05, tilt=0.08, rough=0.0, speck=0.05)
    lay = c.n(60, 4, 6, aspect=0.4)
    stone = c.inside(eu)
    col = col * (1 + 0.12 * lay * stone)[..., None]
    hgt = hgt + 0.15 * lay * stone
    return Maps(col, hgt, depth=size * 0.04, shine=0.05)


@_gen("brick", "stone", "brick wall bricks brickwork red brick masonry facade")
def _brick(c, size=(215.0, 65.0), mortar=10.0, bond="running", palette=None, mortar_color="#b8b0a2",
           weathering=0.0, vary=1.0):
    """Brick wall: bond pattern (running / stack / english / flemish), mortar joints, per-brick tone, burnt
    headers, irregular edges; weathering 0~1 adds chipped edges, soot, stains and efflorescence."""
    _choice("brick", "bond", bond, ("running", "stack", "english", "flemish"))
    pal = palette or PAL["brick"]
    bw, bh = size
    pw, ph = bw + mortar, bh + mortar
    U, V = c.U, c.V
    row = np.floor(V / ph)
    ri = row.astype(np.int64)
    fy = V - row * ph
    odd = (ri & 1) == 1
    header = np.zeros(U.shape, bool)
    if bond == "flemish":
        u = U + np.where(odd, 0.75 * pw, 0.0)
        per = 1.5 * pw
        kk = np.floor(u / per)
        r = u - kk * per
        st = r < pw
        col_i = (2 * kk + (~st)).astype(np.int64)
        fx = np.where(st, r, r - pw)
        Lk = np.where(st, pw, pw * 0.5)
        header = ~st
    else:
        if bond == "english":
            pitch = np.where(odd, pw * 0.5, pw)
            off = np.where(odd, pw * 0.25, 0.0)
            header = odd
        elif bond == "stack":
            pitch, off = np.full(U.shape, pw, F32), 0.0
        else:
            pitch, off = np.full(U.shape, pw, F32), np.where(odd, pw * 0.5, 0.0)
        u = U + off
        col_i = np.floor(u / pitch).astype(np.int64)
        fx = u - col_i * pitch
        Lk = pitch
    d = np.minimum(np.minimum(fx, Lk - mortar - fx), np.minimum(fy, bh - fy))
    d = d + 1.0 * c.n(7, 2, 1) * c.fade(4)
    if weathering:
        d = d - weathering * 7.0 * clamp01(c.n(16, 3, 2) * 1.8 + 0.1)
    b = c.inside(d)
    r = _rnd(col_i, ri, c.s(3), 0)
    tone = 0.2 + 0.65 * r * vary + (1 - vary) * 0.35 - 0.25 * header * (_rnd(col_i, ri, c.s(3), 1) > 0.4)
    col = _ramp(pal, clamp01(tone + 0.1 * c.n(40, 3, 4)))
    face = c.n(3.0, 2, 5) * c.fade(3.0)
    spots = c.cells(7, 1, 6)
    dark = (spots.rand < 0.22) * (1 - smoothstep(0.15, 0.35, spots.f1))
    col = col * (1 + 0.07 * face - 0.25 * dark)[..., None]
    col = col * (0.85 + 0.15 * smoothstep(0, 6, d))[..., None]
    mcol = _c(mortar_color) * (1 + 0.12 * c.n(4, 3, 7) - 0.15 * smoothstep(0, 4, -d))[..., None]
    col = _lerp(mcol, col, b)
    hgt = b * (0.7 + 0.3 * smoothstep(0, 3, d)) + 0.04 * face + 0.05 * c.n(5, 3, 8)
    if weathering:
        w = weathering
        soot = smoothstep(0.05, 0.6, c.n(220, 4, 9)) * w
        streak = smoothstep(0.1, 0.5, c.n(28, 3, 10, aspect=7)) * w * 0.6
        col = col * (1 - 0.45 * soot - 0.25 * streak)[..., None]
        eff = smoothstep(0.25, 0.65, c.n(70, 4, 11)) * w * 0.55 * (1 - smoothstep(0, 10, d))
        col = _lerp(col, np.broadcast_to(_c("#e6e2d8"), col.shape), eff)
    return Maps(col.astype(F32), hgt, depth=5.0, shine=0.03)


@_gen("stone_wall", "stone", "stone wall rubble wall dry stone ashlar castle wall masonry blocks old wall")
def _stone_wall(c, style="rubble", size=95.0, gap=0.06, palette=None, mortar="#8a8276", lichen=0.3):
    """Stone wall: ``rubble`` (irregular flat stones, the old field / castle wall) or ``ashlar`` (cut blocks in
    courses), rough faces, per-stone colour, recessed mortar, lichen."""
    _choice("stone_wall", "style", style, ("rubble", "ashlar"))
    pal = palette or PAL["stone"]
    if style == "ashlar":
        bh = size * 1.15
        k, row, fx, fy, L = _courses(c.U, c.V, size * 2.4, bh, jitter=0.55, seed=c.s(1), random_offset=True)
        eu = np.minimum(np.minimum(fx, L - fx), np.minimum(fy, bh - fy)) - gap * size * 0.5
        eu = eu + 1.5 * c.n(9, 3, 2)
        rand = _rnd(k, row, c.s(1), 3)
        stone = c.inside(eu)
        rough = c.n(30, 5, 3, mode="ridged")
        hgt = stone * (0.5 + 0.3 * smoothstep(0, 10, eu) + 0.3 * rough)
        col = _pick(pal, rand) * (0.85 + 0.3 * _rnd(k, row, c.s(1), 4))[..., None]
        col = col * (1 + 0.1 * c.n(20, 3, 4) + 0.08 * rough)[..., None]
        col = _lerp(_c(mortar) * (1 + 0.1 * c.n(5, 3, 5))[..., None], col, stone)
    else:
        col, hgt, cel, eu = _stones(c, size, gap, pal, mortar, round=0.1, jitter=0.9, warp=0.14, dome=0.3,
                                    tilt=0.3, rough=0.0, aspect=0.55, speck=0.1)
        stone = c.inside(eu)
        rough = c.n(26, 5, 3, mode="ridged")
        hgt = hgt + 0.3 * rough * stone
        col = col * (1 + 0.1 * rough * stone)[..., None]
    if lichen:
        lc = c.cells(4.0, 1, 7)
        spot = (lc.rand < 0.5) * (1 - smoothstep(0.25, 0.45, lc.f1))
        lm = spot * smoothstep(0.3, 0.6, c.n(80, 4, 8) + lichen - 0.3) * stone
        col = _lerp(col, np.broadcast_to(_c("#b5b08a"), col.shape), lm * 0.7)
    return Maps(col, hgt, depth=12.0, shine=0.02)


@_gen("marble", "stone", "marble carrara polished stone veins countertop luxury white marble")
def _marble(c, palette=None, vein="#5b6270", veins=1.0, size=420.0):
    """Polished marble: soft cloudy body with branching dark veins and fine secondary veins."""
    pal = palette or ["#aeaba6", "#d2cfca", "#ecebe7", "#f7f6f3"]
    q1, q2 = c.n(size * 0.8, 4, 1), c.n(size * 0.8, 4, 2)
    Uw, Vw = c.U + size * 0.5 * q1, c.V + size * 0.5 * q2
    n1 = fbm2d_at(Uw / size, Vw / size, 6, seed=c.s(3))
    n2 = fbm2d_at(Uw / (size * 0.35) + 7, Vw / (size * 0.35), 5, seed=c.s(4))
    v1 = np.exp(-(n1 / 0.03) ** 2)
    halo = np.exp(-(n1 / 0.11) ** 2)
    v2 = np.exp(-(n2 / 0.014) ** 2) * 0.6
    cloud = c.n(size * 0.5, 5, 5)
    col = _ramp(pal, clamp01(0.6 + 0.4 * cloud - 0.3 * halo))
    col = _lerp(col, np.broadcast_to(_c(vein), col.shape), clamp01(veins * (0.92 * v1 + v2)))
    return Maps(col, 0.2 * v1, depth=0.3, shine=0.55, rough=70)


@_gen("granite", "stone", "granite speckled stone countertop polished granite igneous")
def _granite(c, palette=None, grain=3.4):
    """Polished granite: interlocking mineral grains (black mica, grey quartz, white / pink feldspar) speckle."""
    pal = palette or ["#141213", "#4d4a49", "#8f8b88", "#c6c1bb", "#dcd7d1", "#b8958a"]
    cel = c.cells(grain, 1.0, 1)
    col = _pick(pal, cel.rand)
    big = c.cells(grain * 2.8, 1.0, 2, warp=0.2)
    feld = (big.rand < 0.22) * c.inside(big.edge * grain * 2.8 - 1.0)
    col = _lerp(col, np.broadcast_to(_c(pal[-1]), col.shape) * (0.9 + 0.2 * big.r(1))[..., None], feld * 0.9)
    col = col * (1 + 0.15 * _speck(c, 1.4, 3) + 0.05 * c.n(120, 3, 4))[..., None]
    return Maps(col, 0.1 * cel.rand, depth=0.3, shine=0.45, rough=60)


@_gen("concrete", "stone", "concrete cement wall brutalist raw concrete beton formwork")
def _concrete(c, palette=None, formwork=True, panel=(480.0, 240.0), stains=0.5):
    """Concrete: mottled grey with air pores, rain stains and formwork panel lines with tie holes."""
    pal = palette or ["#77756f", "#8f8c85", "#a6a39b", "#bab7ae"]
    tone = 0.55 + 0.25 * c.n(220, 5, 1) + 0.1 * c.n(30, 3, 2)
    hgt = 0.1 * c.n(12, 3, 3)
    pw, ph = panel
    if formwork:
        k, row, fx, fy, L = _courses(c.U, c.V, pw, ph, 0.0)
        tone = tone + 0.12 * (_rnd(k, row, c.s(4), 0) - 0.5)
        dj = np.minimum(np.minimum(fx, L - fx), np.minimum(fy, ph - fy))
        line = 1 - c.inside(dj - 0.8)
        tone = tone - 0.18 * line
        hgt = hgt + 0.3 * line
        hx = np.mod(fx, pw / 3.0) - pw / 6.0
        hy = fy - ph * 0.5
        hole = 1 - c.inside(np.sqrt(hx * hx + hy * hy) - 7.0)
        tone = tone - 0.4 * hole
        hgt = hgt - 0.8 * hole
    pc = c.cells(9.0, 1.0, 5)
    r = 0.06 + 0.14 * pc.r(1)
    pore = (pc.rand < 0.25) * (1 - c.inside(pc.f1 * 9.0 - r * 9.0))
    tone = tone - 0.35 * pore
    hgt = hgt - 0.6 * pore
    if stains:
        st = smoothstep(0.0, 0.5, c.n(35, 4, 6, aspect=6)) * smoothstep(-0.2, 0.4, c.n(300, 3, 7)) * stains
        tone = tone - 0.22 * st
    col = _ramp(pal, clamp01(tone)) * (1 + 0.06 * _speck(c, 1.5, 8))[..., None]
    return Maps(col, hgt, depth=3.0, shine=0.02)


@_gen("gravel", "stone", "gravel crushed stone aggregate driveway path chippings ballast")
def _gravel(c, size=11.0, palette=None, mortar="#3a352d"):
    """Gravel: dense small angular stones of mixed tones with deep dark gaps."""
    pal = palette or ["#8a857c", "#a39d92", "#6f6b66", "#b4ab9a", "#7d7266", "#5d5a57", "#c2bbb0"]
    col, hgt, cel, eu = _stones(c, size, 0.1, pal, mortar, round=0.05, jitter=1.0, warp=0.2, dome=0.3,
                                tilt=1.8, rough=0.12, speck=0.1)
    col = col * (0.7 + 0.6 * cel.r(5))[..., None]
    return Maps(col, hgt, depth=size * 0.4, shine=0.05)


@_gen("sand", "stone", "sand beach sand dunes desert ripples sandy")
def _sand(c, wavelength=55.0, angle=-8.0, palette=None, ripples=1.0):
    """Sand with wind ripples (asymmetric crests, darker heavy grains in the troughs) and fine grain."""
    pal = palette or ["#9c7f55", "#c2a274", "#dcc196", "#ecd9b4"]
    U, V = c.rot(angle)
    ph = V / wavelength + 1.1 * c.n(380, 3, 1) + 0.3 * c.n(110, 3, 2)
    fr = ph - np.floor(ph)
    h = np.where(fr < 0.72, smoothstep(0.0, 0.72, fr), 1 - smoothstep(0.72, 1.0, fr)) * ripples
    g = _speck(c, 1.2, 3)
    tone = 0.55 + 0.12 * c.n(260, 3, 4) + 0.1 * g - 0.12 * (1 - h) * ripples
    col = _ramp(pal, clamp01(tone))
    return Maps(col, h + 0.05 * g, depth=6.0, shine=0.03)


@_gen("soil", "stone", "soil dirt earth garden soil ground loam field tilled")
def _soil(c, palette=None, moisture=0.4):
    """Soil: crumbly clods, small pale stones, damp darker patches."""
    pal = palette or PAL["soil"]
    cel = c.cells(18.0, 1.0, 1, round=0.25, warp=0.25)
    eu = cel.edge * 18.0 - 0.8
    clod = c.inside(eu)
    dm = np.sqrt(clamp01(eu / 6.0))
    lumps = c.n(70, 4, 2, mode="billow")
    damp = smoothstep(0.0, 0.5, c.n(200, 4, 3)) * moisture
    tone = 0.25 + 0.45 * cel.rand * clod + 0.25 * lumps * clod - 0.25 * damp
    col = _ramp(pal, clamp01(tone)) * (1 + 0.12 * _speck(c, 1.5, 4))[..., None]
    pb = c.cells(11.0, 1.0, 5)
    peb = (pb.rand < 0.06) * c.inside(0.28 * 11.0 - pb.f1 * 11.0)
    col = _lerp(col, np.broadcast_to(_c("#8a8173"), col.shape) * (0.8 + 0.4 * pb.r(1))[..., None], peb)
    hgt = clod * (0.4 + 0.6 * dm) + 0.4 * lumps + 0.6 * peb
    return Maps(col, hgt, depth=5.0, shine=0.04 + 0.2 * moisture)


@_gen("dry_earth", "stone", "cracked earth dry mud drought cracked mud parched desert floor clay cracks")
def _dry_earth(c, size=90.0, palette=None, crack="#3d2a1a"):
    """Cracked dry earth: polygon plates split by deep cracks, finer secondary cracks, edges curling up."""
    pal = palette or ["#8f6c48", "#b08a5e", "#c9a676", "#d9bd90"]
    cel = c.cells(size, 0.9, 1, warp=0.12)
    w = 0.025 + 0.025 * clamp01(c.n(120, 3, 2) + 0.5)
    e1 = cel.edge - w
    cr1 = 1 - c.inside(e1 * size)
    sub = c.cells(size * 0.38, 1.0, 3, warp=0.2)
    part = smoothstep(-0.05, 0.25, c.n(80, 3, 4))
    cr2 = (1 - c.inside((sub.edge - 0.02) * size * 0.38)) * part
    plate = clamp01(1 - cr1 - cr2 * 0.85)
    curl = 1 - smoothstep(0, 0.2, e1)
    tone = 0.55 + 0.2 * (cel.rand - 0.5) + 0.15 * c.n(40, 3, 5) + 0.15 * curl
    col = _ramp(pal, clamp01(tone)) * (1 + 0.08 * _speck(c, 1.5, 6))[..., None]
    col = _lerp(np.broadcast_to(_c(crack), col.shape), col, plate)
    hgt = plate * (0.6 + 0.3 * curl + 0.1 * c.n(10, 3, 7))
    return Maps(col, hgt, depth=7.0, shine=0.0)


@_gen("asphalt", "stone", "asphalt tarmac road blacktop street pavement bitumen")
def _asphalt(c, palette=None, wear=0.4):
    """Asphalt: black binder packed with small light aggregate stones, worn paler patches."""
    pal = palette or ["#5b5a58", "#76726c", "#8d8880", "#a39e94"]
    cel = c.cells(4.2, 1.0, 1, round=0.15)
    eu = cel.edge * 4.2 - 0.25
    st = (cel.rand < 0.6) * c.inside(eu)
    col = _pick(pal, cel.r(1)) * (0.7 + 0.5 * cel.r(2))[..., None]
    bind = _c("#232325") * (1 + 0.15 * _speck(c, 1.2, 2))[..., None]
    col = _lerp(bind, col, st * (0.55 + 0.45 * c.fade(4.2)))
    worn = smoothstep(0.0, 0.6, c.n(260, 4, 3)) * wear
    col = col * (1 + 0.35 * worn)[..., None]
    return Maps(col, st * (0.5 + 0.5 * np.sqrt(clamp01(eu))), depth=1.5, shine=0.04)


# ======================================================================== fabric

def _pattern(p, i, j):
    if p == "plain":
        return ((i + j) & 1) == 0
    if p == "twill":                       # 3/1 warp-faced (denim)
        return np.mod(i + j, 4) < 3
    if p == "twill2":                      # 2/2 (tartan, houndstooth, carbon)
        return np.mod(i + j, 4) < 2
    if p == "basket":
        return (((i >> 1) + (j >> 1)) & 1) == 0
    if p == "satin":
        return np.mod(i * 2 + j, 5) != 0
    raise ValueError(f"unknown weave {p!r}: plain / twill / twill2 / basket / satin")


def _weave(c, pitch, pattern="plain", gap=0.14, wobble=0.12, angle=0.0, U=None, V=None):
    """Woven threads: warp (vertical) and weft (horizontal) going over and under. Returns a dict with ``t`` (1 where
    warp shows), ``h`` height, thread indices ``i, j``, fibre and slub noise per thread."""
    U, V = c.rot(angle, U, V)
    x = U / pitch
    y = V / pitch
    if wobble:
        x = x + wobble * c.n(pitch * 14, 3, 31, U=U, V=V)
        y = y + wobble * c.n(pitch * 14, 3, 37, U=U, V=V)
    i = np.floor(x).astype(np.int64)
    j = np.floor(y).astype(np.int64)
    fx = x - i - 0.5
    fy = y - j - 0.5
    on = _pattern(pattern, i, j)
    g = 0.5 - gap * 0.5
    px = np.sqrt(clamp01(1 - (fx / g) ** 2))
    py = np.sqrt(clamp01(1 - (fy / g) ** 2))
    up = np.where(fy < 0, _pattern(pattern, i, j - 1), _pattern(pattern, i, j + 1))     # float continues?
    lf = np.where(fx < 0, ~_pattern(pattern, i - 1, j), ~_pattern(pattern, i + 1, j))
    bw = np.where(up, 1.0, np.cos(fy * np.pi * 0.85))
    bf = np.where(lf, 1.0, np.cos(fx * np.pi * 0.85))
    hw = px * np.where(on, 0.55 + 0.45 * bw, 0.3)
    hf = py * np.where(on, 0.3, 0.55 + 0.45 * bf)
    t = smoothstep(-0.05, 0.05, hw - hf)
    fade = c.fade(pitch * 0.5)
    fd = c.fade(pitch * 0.8)                       # threads below ~2 px blend to their average (no moire)
    tm = {"plain": 0.5, "twill": 0.75, "twill2": 0.5, "basket": 0.5, "satin": 0.8}[pattern]
    t = tm + (t - tm) * fd
    hw = 0.5 + (hw - 0.5) * fd
    hf = 0.5 + (hf - 0.5) * fd
    fib_w = fbm2d_at(x * 3.0, y * 0.6, 2, seed=c.s(41)) * fade
    fib_f = fbm2d_at(x * 0.6, y * 3.0, 2, seed=c.s(43)) * fade
    slub_w = fbm2d_at(i * 3.17 + 0.5, y * 0.12, 2, seed=c.s(47))
    slub_f = fbm2d_at(x * 0.12, j * 3.71 + 0.5, 2, seed=c.s(53))
    return dict(t=t, h=np.maximum(hw, hf), i=i, j=j, fx=fx, fy=fy, fib=fib_f + (fib_w - fib_f) * t,
                slub=slub_f + (slub_w - slub_f) * t, x=x, y=y, fade=fade)


def _cloth(c, w, warp, weft, slub=0.12, fibre=0.1, ao=0.45):
    """Colour woven threads: warp / weft colours (single colours or (H, W, 3) maps per thread)."""
    warp = np.broadcast_to(np.asarray(warp, F32), w["t"].shape + (3,))
    weft = np.broadcast_to(np.asarray(weft, F32), w["t"].shape + (3,))
    col = _lerp(weft, warp, w["t"])
    k = (1 + slub * w["slub"] + fibre * w["fib"]) * ((1 - ao) + ao * np.sqrt(w["h"]) * w["fade"] + ao * (1 - w["fade"]))
    return (col * k[..., None]).astype(F32)


def _sett_table(sett, mirror=True):
    seq = []
    for col, n in sett:
        seq += [rgb(col)] * int(n)
    if mirror and len(seq) > 2:
        seq = seq + seq[-2:0:-1]
    return np.stack(seq).astype(F32)


@_gen("linen", "fabric", "linen plain weave cloth fabric canvas cotton burlap tablecloth natural fabric")
def _linen(c, pitch=6.0, palette=None, slub=0.5):
    """Linen / plain weave: threads over-under one by one, uneven slubby threads, natural colour."""
    pal = palette or ["#cbbd9f", "#c2b293"]
    w = _weave(c, pitch, "plain", gap=0.16, wobble=0.22)
    warp = _ramp([pal[0]], w["t"]) * (1 + 0.06 * (_rnd(w["i"], 0, c.s(1)) - 0.5))[..., None]
    weft = _ramp([pal[-1]], w["t"]) * (1 + 0.06 * (_rnd(0, w["j"], c.s(2)) - 0.5))[..., None]
    col = _cloth(c, w, warp, weft, slub=0.18 * slub / 0.5)
    col = col * (1 + 0.05 * c.n(200, 3, 3))[..., None]
    return Maps(col, w["h"], depth=pitch * 0.3, shine=0.02)


@_gen("denim", "fabric", "denim jeans twill blue jeans indigo cotton twill")
def _denim(c, pitch=4.2, warp="#22406e", weft="#b4b8c2", fade=0.5):
    """Denim: 3/1 twill (diagonal ribs), indigo warp over white weft, faded areas and slub streaks."""
    w = _weave(c, pitch, "twill", gap=0.1, wobble=0.1)
    ring = clamp01(0.5 + 0.9 * w["slub"])
    fz = smoothstep(-0.3, 0.6, c.n(260, 4, 1)) * fade
    wc = _lerp(np.broadcast_to(_c(warp), w["t"].shape + (3,)), np.broadcast_to(_c("#7f9cc4"), w["t"].shape + (3,)),
               clamp01(0.55 * fz + 0.25 * ring))
    col = _cloth(c, w, wc, _c(weft), slub=0.1)
    return Maps(col, w["h"], depth=pitch * 0.3, shine=0.02)


@_gen("tartan", "fabric", "tartan plaid scottish kilt check flannel woven check")
def _tartan(c, pitch=3.0, sett=None):
    """Tartan / plaid woven in 2/2 twill: the same colour sett on warp and weft makes the crossing checks.
    sett: [(colour, thread count), ...] (mirrored)."""
    sett = sett or [("#a3172a", 22), ("#0f2347", 9), ("#a3172a", 3), ("#101012", 2), ("#a3172a", 3),
                    ("#1e5236", 12), ("#e9e2c6", 2), ("#1e5236", 12), ("#d8ae2e", 2)]
    tbl = _sett_table(sett)
    w = _weave(c, pitch, "twill2", gap=0.08, wobble=0.04)
    n = len(tbl)
    col = _cloth(c, w, tbl[np.mod(w["i"], n)], tbl[np.mod(w["j"], n)], slub=0.06, fibre=0.12, ao=0.35)
    return Maps(col, w["h"], depth=pitch * 0.25, shine=0.0)


@_gen("stripes", "fabric", "stripes striped fabric ticking awning stripe shirt deck chair")
def _stripes(c, widths=(14, 14), palette=None, pitch=3.0, angle=0.0, weft=None):
    """Woven stripes: coloured warp threads in bands (widths in threads, cycling through the palette)."""
    pal = palette or ["#1d3557", "#f1ede2"]
    sett = [(pal[k % len(pal)], widths[k % len(widths)]) for k in range(max(len(pal), len(widths)))]
    tbl = _sett_table(sett, mirror=False)
    w = _weave(c, pitch, "plain", gap=0.1, wobble=0.04, angle=angle)
    wf = _c(weft or pal[-1])
    col = _cloth(c, w, tbl[np.mod(w["i"], len(tbl))], wf * 0.95, slub=0.06, ao=0.35)
    return Maps(col, w["h"], depth=pitch * 0.25)


@_gen("polka", "fabric", "polka dots dotted fabric spots print dots pattern cloth")
def _polka(c, spacing=60.0, radius=13.0, color="#c8283c", dot="#f6f0e4", pitch=3.0):
    """Polka dots printed on a plain-weave cotton (staggered rows of round dots)."""
    w = _weave(c, pitch, "plain", gap=0.12, wobble=0.05)
    base = _cloth(c, w, _c(color), _c(color) * 0.92, slub=0.05, ao=0.3)
    _, _, dx, dy, _ = _hexgrid(c.U, c.V, spacing / _SQ3)
    dd = np.sqrt(dx * dx + dy * dy) - radius
    m = c.inside(-dd - 0.3 * w["h"])
    dcol = _cloth(c, w, _c(dot), _c(dot) * 0.95, slub=0.04, ao=0.25)
    return Maps(_lerp(base, dcol, m), w["h"], depth=pitch * 0.25)


@_gen("houndstooth", "fabric", "houndstooth dogstooth tweed check jacket wool pattern pied de poule")
def _houndstooth(c, pitch=3.4, dark="#1a1a1c", light="#ece6d6", block=4):
    """Houndstooth: 2/2 twill with 4 dark / 4 light threads in warp and weft -> the broken check appears by itself."""
    tbl = _sett_table([(dark, block), (light, block)], mirror=False)
    w = _weave(c, pitch, "twill2", gap=0.08, wobble=0.03)
    n = len(tbl)
    col = _cloth(c, w, tbl[np.mod(w["i"], n)], tbl[np.mod(w["j"] + 2, n)], slub=0.05, fibre=0.15, ao=0.3)
    return Maps(col, w["h"], depth=pitch * 0.25)


@_gen("chevron", "fabric", "chevron zigzag fabric print stripes zig zag pattern")
def _chevron(c, period=70.0, band=24.0, palette=None, pitch=3.0):
    """Chevron / zigzag stripes printed on plain-weave cloth."""
    pal = palette or ["#2b3a55", "#f0e9dc", "#d9a441", "#f0e9dc"]
    w = _weave(c, pitch, "plain", gap=0.12, wobble=0.05)
    tri = np.abs(np.mod(c.U / period, 1.0) - 0.5) * period
    z = c.V + tri
    k = np.floor(z / band)
    fz = z - k * band
    n = len(pal)
    ca = _pick(pal, np.mod(k, n) / n + 0.5 / n)
    cb = _pick(pal, np.mod(k + 1, n) / n + 0.5 / n)
    sw = c.inside(fz - band + 0.5) if True else 0
    col = _lerp(ca, cb, sw) * 1.0
    col = _cloth(c, w, col, col * 0.94, slub=0.05, ao=0.3)
    return Maps(col, w["h"], depth=pitch * 0.25)


@_gen("knit", "fabric", "knit knitted sweater jumper wool stitches jersey knitwear scarf")
def _knit(c, size=(16.0, 13.0), color="#9e2f3c", angle=0.0):
    """Knitted wool: interlocking V-shaped stitches in columns, twisted plies, fuzzy fibres."""
    sw, sh = size
    U, V = c.rot(angle)
    x = U / sw
    y = V / sh
    i = np.floor(x)
    fx = x - i - 0.5
    best = np.zeros(U.shape, F32)
    ply = np.zeros(U.shape, F32)
    tone = np.zeros(U.shape, F32)
    j0 = np.floor(y)
    for dj in (-1, 0, 1):
        j = j0 + dj
        ly = (y - (j + 0.5)) * (sh / sw)
        for side in (-1.0, 1.0):
            ux = fx - side * 0.24
            dxv, dyv = side * 0.45, -0.89
            al = ux * dxv + ly * dyv
            ac = ux * 0.89 + ly * side * 0.45
            r = (al / 0.62) ** 2 + (ac / 0.25) ** 2
            leg = np.sqrt(clamp01(1 - r))
            better = leg > best
            best = np.where(better, leg, best)
            ply = np.where(better, al * 6.0 + ac * side * 7.0, ply)
            tone = np.where(better, _rnd(i.astype(np.int64), j.astype(np.int64), c.s(1)), tone)
    tw = 0.5 + 0.5 * np.sin(ply * 2 * np.pi) * c.fade(sw * 0.12)
    fuzz = _speck(c, 1.4, 2)
    k = (0.3 + 0.7 * best ** 0.7) * (0.88 + 0.16 * tw + 0.05 * (tone - 0.5) + 0.06 * fuzz)
    col = _c(color) * k[..., None]
    return Maps(col, best, depth=sh * 0.3, shine=0.0)


@_gen("felt", "fabric", "felt wool felt matted fibres billiard cloth baize")
def _felt(c, color="#3d6b4a", fibre=1.0):
    """Felt: dense matted fibres with no weave, soft low-contrast mottling and fine pills."""
    f = _fibres(c, 7.0, 1, n=5, aspect=0.12)
    m = c.n(30, 4, 2)
    pill = c.cells(9.0, 1.0, 4)
    pl = (pill.rand < 0.3) * (1 - smoothstep(0.15, 0.4, pill.f1))
    k = 1 + 0.34 * fibre * (f - 0.5) + 0.14 * m + 0.07 * _speck(c, 1.2, 3) + 0.06 * pl
    return Maps(_c(color) * k[..., None], 0.4 * f + 0.25 * m + 0.12 * pl, depth=1.5)


@_gen("velvet", "fabric", "velvet velour plush crushed velvet curtain sheen")
def _velvet(c, palette=None, crush=1.0):
    """Velvet: deep saturated pile with soft bright sheen where the pile is crushed."""
    pal = palette or ["#1e040b", "#4a0c1c", "#86223b", "#d0687f"]
    wv = c.n(300, 2, 1)
    f = np.sin(2 * np.pi * (c.U + 70 * wv) / 230.0) + 0.6 * c.n(200, 3, 2) * crush
    gx, gy = c.grad(f * 30.0)
    slope = np.sqrt(gx * gx + gy * gy)
    face = -(gx * c.L[0] + gy * c.L[1])
    L = clamp01(0.3 + 0.25 * f * 0.5 + 0.35 * face)
    sheen = clamp01(slope * 0.7) ** 1.5
    nap = c.n(40, 3, 3, aspect=4) * 0.06
    col = _ramp(pal, clamp01(0.15 + 0.55 * L + 0.45 * sheen + nap))
    col = col * (1 + 0.05 * _speck(c, 1.1, 4))[..., None]
    return Maps(col, None)


@_gen("leather", "fabric", "leather hide pebble grain leather bag sofa saddle brown leather")
def _leather(c, palette=None, grain=9.0, creases=0.6):
    """Leather: pebbled grain, creases and mottled tone with a soft sheen."""
    pal = palette or ["#2e1a0f", "#5a3520", "#82522f", "#a06d44"]
    cel = c.cells(grain, 1.0, 1, round=0.2)
    bump = smoothstep(0.0, 0.3, cel.edge)
    cr = c.n(150, 4, 2, mode="ridged")
    crease = smoothstep(0.82, 0.96, cr) * creases
    tone = 0.55 + 0.25 * c.n(120, 4, 3) + 0.12 * bump - 0.25 * crease + 0.05 * (cel.rand - 0.5)
    col = _ramp(pal, clamp01(tone))
    return Maps(col, 0.5 * bump - 0.8 * crease, depth=1.8, shine=0.18, rough=10)


@_gen("carpet", "fabric", "carpet pile wall to wall carpet tufted floor covering shag")
def _carpet(c, color="#9a8570", tuft=4.2, palette=None):
    """Carpet pile: dense tufts of yarn with fuzzy tips and soft worn tracks."""
    cel = c.cells(tuft, 0.8, 1, round=0.35)
    dm = smoothstep(0.0, 0.45, cel.edge)
    base = _c(color) if palette is None else _ramp(palette, clamp01(0.5 + 0.6 * c.n(200, 3, 2)))
    fibre = _speck(c, 1.3, 5) + 0.6 * c.n(2.2, 2, 6)           # loose yarn fibres on top of the tufts
    k = (0.72 + 0.28 * dm) * (1 + 0.1 * (cel.rand - 0.5) + 0.07 * c.n(220, 4, 3) + 0.14 * fibre)
    k = k * c.fade(tuft) + (1 - c.fade(tuft)) * 0.85
    return Maps(base * k[..., None], 0.7 * dm + 0.5 * fibre, depth=tuft * 0.35)


@_gen("rug", "fabric", "rug oriental rug persian carpet kilim pattern rug ornamental carpet")
def _rug(c, size=150.0, palette=None, tuft=3.6):
    """Patterned rug: diamond medallions in rich colours knotted into a tufted pile (pattern edges follow tufts)."""
    pal = palette or ["#7a1c1f", "#1d2b4e", "#e3d2a8", "#c48d2e", "#2f5a49", "#7a1c1f"]
    cel = c.cells(tuft, 0.7, 1, round=0.3)
    U = c.U - cel.dx * tuft
    V = c.V - cel.dy * tuft
    fx = np.mod(U / size, 1.0) - 0.5
    fy = np.mod(V / size, 1.0) - 0.5
    dmd = np.abs(fx) + np.abs(fy)
    band = np.floor(dmd * 9.0)
    idx = np.mod(band, len(pal) - 1)
    col = _pick(pal[:-1], idx / (len(pal) - 1) + 0.01)
    cross = (np.minimum(np.abs(fx), np.abs(fy)) < 0.03) & (dmd < 0.2)
    col = np.where(cross[..., None], _c(pal[2]), col)
    ring = (np.abs(dmd - 0.5) < 0.035)
    col = np.where(ring[..., None], _c(pal[3]), col)
    dm = smoothstep(0.0, 0.45, cel.edge)
    k = (0.55 + 0.45 * dm) * (1 + 0.1 * (cel.rand - 0.5))
    return Maps(col * k[..., None], dm, depth=tuft * 0.35)


@_gen("quilted", "fabric", "quilted quilt padded puffer jacket diamond quilting upholstery mattress")
def _quilted(c, size=60.0, color="#2f4b6e", angle=45.0):
    """Quilted padding: puffy diamond cushions between stitched seams, satin sheen."""
    U, V = c.rot(angle)
    fa = np.mod(U / size, 1.0) - 0.5
    fb = np.mod(V / size, 1.0) - 0.5
    pa = np.sqrt(clamp01(1 - (2 * fa) ** 2))
    pb = np.sqrt(clamp01(1 - (2 * fb) ** 2))
    puff = (pa * pb) ** 0.6
    ea = (0.5 - np.abs(fa)) * size
    eb = (0.5 - np.abs(fb)) * size
    seam = np.minimum(ea, eb)
    along = np.where(ea < eb, V, U)
    dash = (np.mod(along / 7.0, 1.0) < 0.6).astype(F32)
    stitch = (1 - c.inside(seam - 0.9)) * dash
    w = _weave(c, 2.2, "plain", gap=0.1, wobble=0.0)
    col = _c(color) * (1 + 0.05 * (w["h"] - 0.5))[..., None]
    col = _lerp(col, np.broadcast_to(_c(color) * 1.35, col.shape), stitch * 0.8)
    return Maps(col, puff, depth=size * 0.22, shine=0.35, rough=10)


# ======================================================================== metal and man-made

@_gen("brushed_metal", "metal", "brushed metal brushed steel stainless aluminium satin metal steel plate")
def _brushed(c, angle=0.0, palette=None):
    """Brushed metal: fine anisotropic streaks along one direction and a broad soft sheen."""
    pal = palette or ["#4c5259", "#8f959c", "#c9cdd2", "#eef0f2"]
    U, V = c.rot(angle)
    s1 = c.n(600, 4, 1, aspect=1.6 / 600, U=U, V=V)
    s2 = c.n(160, 3, 2, aspect=5.0 / 160, U=U, V=V)
    _, rv = _rot(c.ru - 0.5, c.rv - 0.5, angle)
    sheen = np.exp(-((rv + 0.12) / 0.2) ** 2)
    t = 0.3 + 0.3 * s1 + 0.2 * s2 + 0.3 * sheen - 0.1 * (rv + 0.5)
    return Maps(_ramp(pal, clamp01(t)), 0.3 * s1, depth=0.3, metal=True, shine=0.25, rough=8)


@_gen("hammered_metal", "metal", "hammered metal hammered copper beaten metal dimpled copper pot planished")
def _hammered(c, size=30.0, palette=None):
    """Hammered metal (copper by default): overlapping round dents, each catching its own highlight."""
    pal = palette or PAL["copper"]
    cel = c.cells(size, 0.85, 1)
    dent = cel.f1 ** 1.6
    col = _ramp(pal, clamp01(0.55 + 0.15 * c.n(200, 3, 2) + 0.06 * (cel.rand - 0.5)))
    return Maps(col, dent * 1.6, depth=size * 0.3, metal=True, shine=1.0, rough=30)


@_gen("rust", "metal", "rust rusty metal corroded steel oxidised iron weathered metal corrosion")
def _rust(c, amount=0.6, palette=None):
    """Rust: flaking orange-brown corrosion spreading over grey steel, pits and drips."""
    pal = palette or ["#2b1207", "#56240f", "#874014", "#b4622a", "#cf8a42"]
    steel = _ramp(["#3a3e44", "#6a7078", "#8a9098"], clamp01(0.5 + 0.5 * c.n(80, 3, 1)))
    cov = smoothstep(-0.12, 0.12, c.n(150, 5, 2) + (amount - 0.5) * 0.8)
    rt = clamp01(0.5 + 0.6 * c.n(30, 5, 3))
    rcol = _ramp(pal, rt)
    fl = c.cells(16, 1.0, 4, warp=0.25)
    crack = 1 - smoothstep(0.015, 0.06, fl.edge)
    rcol = rcol * (0.85 + 0.3 * fl.rand - 0.45 * crack)[..., None]
    pits = c.cells(7, 1.0, 5)
    pit = (pits.rand < 0.18) * (1 - smoothstep(0.1, 0.3, pits.f1))
    rcol = rcol * (1 - 0.5 * pit)[..., None]
    drip = smoothstep(0.1, 0.5, c.n(22, 3, 6, aspect=9)) * (1 - cov) * amount
    steel = _lerp(steel, np.broadcast_to(_c("#7a3a17"), steel.shape), drip * 0.6)
    col = _lerp(steel, rcol, cov)
    hgt = cov * (0.4 + 0.3 * fl.rand * (1 - crack) + 0.3 * rt) - 0.3 * pit
    m = Maps(col, hgt, depth=3.0, shine=0.0)
    return m


@_gen("galvanized", "metal", "galvanized steel galvanised zinc spangle sheet metal hot dip")
def _galvanized(c, size=46.0, palette=None):
    """Galvanised steel: zinc spangle crystals, each with its own grey and feathery dendrite direction."""
    pal = palette or ["#5a6168", "#8e959c", "#bcc2c7", "#e1e4e6"]
    cel = c.cells(size, 1.0, 1, warp=0.15)
    a = cel.r(1) * np.pi
    ca, sa = np.cos(a), np.sin(a)
    ur = c.U * ca + c.V * sa
    vr = -c.U * sa + c.V * ca
    den = fbm2d_at(ur / 14.0, vr / 1.6, 3, seed=c.s(2)) * c.fade(2.0)
    # feathery dendrite: fine stripes across each crystal's own direction, bending with the noise
    fern = np.sin(vr * 1.5 + 5.0 * den + 9.0 * cel.r(3)) * (0.5 + 0.5 * np.sin(ur * 0.12 + 6.0 * cel.r(4)))
    t = 0.18 + 0.62 * cel.rand + 0.35 * den + 0.13 * fern - 0.2 * (1 - smoothstep(0, 0.04, cel.edge))
    t = t + 0.04 * c.n(3, 2, 7)
    return Maps(_ramp(pal, clamp01(t)), 0.15 * den - 0.2 * (1 - smoothstep(0, 0.04, cel.edge)), depth=0.8,
                metal=True, shine=0.35, rough=12)


@_gen("chrome", "metal", "chrome mirror polished metal reflective chrome strip shiny metal")
def _chrome(c, angle=0.0, stops=None):
    """Chrome: a sharp mirror reflection strip (dark sky / bright horizon / dark ground) along the surface."""
    stops = stops or [(0.0, "#161a21"), (0.16, "#55606f"), (0.3, "#f2f6fc"), (0.36, "#b9c3d1"), (0.5, "#e8edf4"),
                      (0.545, "#1d2128"), (0.66, "#3b414b"), (0.86, "#aab3bf"), (1.0, "#eef1f5")]
    ru, rv = _rot(c.ru - 0.5, c.rv - 0.5, angle)
    t = rv + 0.5 + 0.012 * c.n(300, 3, 1)
    return Maps(mix(stops, clamp01(t)), None)


@_gen("gold_leaf", "metal", "gold leaf gilded gilt gold foil gilding")
def _gold_leaf(c, size=80.0, palette=None):
    """Gold leaf: overlapping square leaves, each a slightly different gold, crinkles and red bole cracks."""
    pal = palette or PAL["gold"]
    k, row, fx, fy, L = _courses(c.U, c.V, size, size, jitter=0.0, seed=c.s(1), random_offset=True)
    tone = _rnd(k, row, c.s(1), 5)
    dj = np.minimum(np.minimum(fx, L - fx), np.minimum(fy, size - fy)) + 1.5 * c.n(6, 2, 2)
    seam = 1 - smoothstep(0.0, 2.0, dj)
    cr = c.n(26, 5, 3, mode="ridged")
    crink = smoothstep(0.7, 0.95, cr)
    bole = smoothstep(0.93, 0.98, c.n(18, 4, 4, mode="ridged"))
    col = _ramp(pal, clamp01(0.6 + 0.22 * (tone - 0.5) + 0.1 * c.n(60, 3, 5) - 0.12 * seam))
    col = _lerp(col, np.broadcast_to(_c("#7a2a17"), col.shape), bole * 0.9)
    return Maps(col, 0.5 * crink + 0.4 * seam, depth=1.5, metal=True, shine=1.1, rough=22)


@_gen("carbon_fiber", "metal", "carbon fiber carbon fibre composite weave racing kevlar")
def _carbon(c, pitch=10.0):
    """Carbon fibre: 2/2 twill of flat tows; tows facing different ways reflect differently, under clear coat."""
    w = _weave(c, pitch, "twill2", gap=0.04, wobble=0.0)
    fl_w = 0.5 + 0.5 * np.sin(w["x"] * 2 * np.pi * 7)
    fl_f = 0.5 + 0.5 * np.sin(w["y"] * 2 * np.pi * 7)
    fl = (fl_f + (fl_w - fl_f) * w["t"]) * c.fade(pitch / 7)
    g = 0.06 + 0.2 * w["t"] + 0.12 * w["h"] + 0.03 * fl
    col = np.stack([g, g * 1.03, g * 1.08], -1).astype(F32)
    return Maps(col, w["h"] * 0.15, depth=1.0, shine=0.8, rough=70)


@_gen("perforated", "metal", "perforated metal sheet holes mesh grille speaker grill punched metal")
def _perforated(c, spacing=26.0, radius=8.0, palette=None):
    """Perforated sheet: staggered round holes punched in brushed metal (alpha = 0 in the holes)."""
    pal = palette or ["#6a7179", "#a4abb2", "#d3d7db"]
    _, _, dx, dy, _ = _hexgrid(c.U, c.V, spacing / _SQ3)
    d = np.sqrt(dx * dx + dy * dy) - radius
    hole = 1 - c.inside(d)
    s = c.n(400, 3, 1, aspect=2.0 / 400)
    sheet = _ramp(pal, clamp01(0.5 + 0.25 * s + 0.2 * (0.5 - c.rv)))
    col = _lerp(sheet, np.broadcast_to(_c("#0b0c0e"), sheet.shape), hole)
    return Maps(col, (1 - hole) * smoothstep(0, 2.5, d), alpha=1 - hole, depth=2.5, metal=True, shine=0.5,
                rough=16)


@_gen("corrugated", "metal", "corrugated metal sheet tin roof iron sheet shed corrugated iron wavy metal")
def _corrugated(c, period=34.0, angle=0.0, color=None, rust=0.0):
    """Corrugated sheet: sinusoidal ribs (galvanised, or painted with ``color``), optional rust streaks."""
    U, V = c.rot(angle)
    ph = 2 * np.pi * U / period
    h = 0.5 + 0.5 * np.cos(ph)
    lx = 1.0 if c.L[0] >= 0 else -1.0
    if color is None:
        # each rib is a curved mirror: dark flank, bright crest, soft zinc mottling along the ribs
        refl = 0.5 + 0.5 * np.cos(ph + 1.0)
        tone = 0.45 + 0.4 * refl - 0.25 * lx * np.sin(ph) + 0.12 * c.n(90, 4, 1, aspect=0.05)
        col = _ramp(["#5d646c", "#8d949b", "#bcc2c7", "#e4e7ea"], clamp01(tone))
        metal = True
    else:
        col = _c(color) * (1 + 0.05 * c.n(80, 3, 1))[..., None]
        metal = False
    if rust:
        st = smoothstep(0.0, 0.5, c.n(24, 3, 2, aspect=10) + rust - 0.5) * (1 - h * 0.5)
        col = _lerp(col, np.broadcast_to(_c("#8a4318"), col.shape), clamp01(st * rust * 1.4))
    return Maps(col, h, depth=period * 0.45, metal=metal, shine=0.6 if metal else 0.15, rough=14)


@_gen("paper", "man-made", "paper sheet watercolor paper parchment stationery handmade paper cream")
def _paper(c, color="#f7f0de", fibre=1.0):
    """Paper: a light warm sheet with fine grain and faint fibres; ``color=`` sets the sheet colour. Painted straight
    onto a canvas it makes a usable background."""
    f = _fibres(c, 7.0, 1, n=4, aspect=0.1)
    m = c.n(90, 3, 2)
    tooth = c.n(1.8, 2, 6)
    k = 1 + 0.03 * m + 0.07 * fibre * (f - 0.5) + 0.04 * tooth + 0.02 * _speck(c, 1.2, 3)
    return Maps(np.clip(_c(color) * k[..., None], 0, 1.2), 0.25 * f + 0.1 * m + 0.35 * tooth, depth=0.4, ambient=0.8)


@_gen("cardboard", "man-made", "cardboard kraft paper box carton brown paper corrugated board")
def _cardboard(c, color="#b88a58", flute=12.0):
    """Cardboard / kraft: brown fibres, dark specks and the faint ribs of the flutes underneath."""
    f = _fibres(c, 8.0, 1, n=4, aspect=0.12)
    fl = 0.5 + 0.5 * np.cos(2 * np.pi * c.V / flute)
    sp = c.cells(10.0, 1.0, 2)
    speck = (sp.rand < 0.12) * (1 - smoothstep(0.05, 0.16, sp.f1))
    k = 1 + 0.06 * c.n(80, 4, 3) + 0.09 * (f - 0.5) + 0.04 * fl - 0.35 * speck
    return Maps(_c(color) * k[..., None], 0.3 * f + 0.4 * fl, depth=0.8)


@_gen("cork", "man-made", "cork board pinboard cork tiles bottle cork corkboard")
def _cork(c, grain=5.5, palette=None):
    """Cork: compressed granules of mixed tan and brown with dark pores."""
    pal = palette or ["#6b4425", "#9a6a3e", "#b98a57", "#cfa572", "#83552f"]
    cel = c.cells(grain, 1.0, 1, round=0.15)
    gran = c.inside((cel.edge - 0.06) * grain)
    col = _pick(pal, cel.rand) * (0.85 + 0.3 * cel.r(1))[..., None]
    pores = c.cells(grain * 4, 1.0, 2)
    pore = (pores.rand < 0.3) * (1 - smoothstep(0.08, 0.2, pores.f1))
    col = _lerp(np.broadcast_to(_c("#3b2412"), col.shape), col, gran * (1 - pore))
    return Maps(col, gran * (0.5 + 0.5 * smoothstep(0, 0.3, cel.edge)), depth=1.5)


@_gen("crackle_glaze", "man-made", "crackle glaze craquelure ceramic glaze celadon pottery raku crazing")
def _crackle(c, color="#a9c3b0", size=70.0, crack="#5a5446"):
    """Ceramic crackle glaze: glossy glaze crazed by a network of fine stained cracks at two scales."""
    base = _c(color) * (1 + 0.06 * c.n(120, 4, 1) + 0.04 * c.n(25, 3, 2))[..., None]
    big = c.cells(size, 1.0, 3, warp=0.25)
    l1 = 1 - c.inside((big.edge - 0.008) * size)
    sm = c.cells(size * 0.35, 1.0, 4, warp=0.3)
    l2 = (1 - c.inside((sm.edge - 0.012) * size * 0.35)) * smoothstep(-0.1, 0.3, c.n(90, 3, 5))
    m = clamp01(l1 + 0.6 * l2)
    col = _lerp(base, np.broadcast_to(_c(crack), base.shape), m * 0.8)
    return Maps(col, -0.3 * m, depth=0.5, shine=0.7, rough=90)


@_gen("glass_tiles", "man-made", "glass mosaic tiles pool mosaic glass tile backsplash mosaic")
def _glass_tiles(c, size=42.0, grout=4.0, palette=None, grout_color="#d8d6cf"):
    """Glass mosaic: small square glass tiles in mixed blues, glossy bevels and a glow inside each tile."""
    pal = palette or ["#1f6f8b", "#2d8fa8", "#4cb5c4", "#16506e", "#6fcad0", "#2a7f9e"]
    k, row, d, fx, fy = _tile_grid(c, size, size, grout)
    r = _rnd(k, row, c.s(1), 0)
    col = _pick(pal, r)
    inner = (1 - np.abs(fx / size - 0.45) * 1.5) * (1 - np.abs(fy / size - 0.45) * 1.5)
    col = col * (0.75 + 0.5 * clamp01(inner) + 0.1 * (_rnd(k, row, c.s(1), 1) - 0.5))[..., None]
    t = c.inside(d)
    col = _lerp(_c(grout_color) * (1 + 0.08 * c.n(4, 2, 2))[..., None], col, t)
    return Maps(col, t * smoothstep(0, 4, d), depth=3.0, shine=1.0, rough=80)


@_gen("terrazzo", "man-made", "terrazzo chips speckled floor composite stone confetti")
def _terrazzo(c, palette=None, base="#ece6da"):
    """Terrazzo: angular chips of coloured stone set in a pale polished matrix, at two sizes."""
    pal = palette or ["#c8664a", "#5f7f6e", "#2b2b2e", "#9ca3a8", "#d9a441", "#e0b9a5"]
    col = np.broadcast_to(_c(base), c.U.shape + (3,)) * (1 + 0.04 * _speck(c, 1.5, 1))[..., None]
    for sz, keep, inset, sd in ((40.0, 0.45, 0.22, 2), (13.0, 0.5, 0.25, 3), (5.0, 0.4, 0.25, 4)):
        cel = c.cells(sz, 1.0, sd, warp=0.1)
        chip = (cel.rand < keep) * c.inside((cel.edge - inset) * sz)
        cc = _pick(pal, cel.r(1)) * (0.85 + 0.3 * cel.r(2))[..., None]
        col = _lerp(col, cc, chip)
    return Maps(col, None, shine=0.4, rough=60)


@_gen("damask", "man-made", "damask wallpaper ornamental baroque victorian wallpaper floral pattern brocade")
def _damask(c, size=(130.0, 170.0), palette=None):
    """Damask wallpaper: mirrored ornamental motif in a half-drop repeat, tone on tone with a satin sheen."""
    pal = palette or ["#4a1820", "#7b2c37"]
    W, H = size
    col_i = np.floor(c.U / W)
    vv = c.V + np.mod(col_i, 2) * H * 0.5
    row = np.floor(vv / H)
    x = np.abs(c.U - (col_i + 0.5) * W) / W
    y = (vv - (row + 0.5) * H) / H

    def ell(cx, cy, rx, ry, ang=0.0):
        a = math.radians(ang)
        px, py = x - cx, y - cy
        qx = px * math.cos(a) + py * math.sin(a)
        qy = -px * math.sin(a) + py * math.cos(a)
        return np.sqrt((qx / rx) ** 2 + (qy / ry) ** 2) - 1.0

    d = np.minimum.reduce([
        ell(0.0, 0.08, 0.09, 0.2), ell(0.0, -0.22, 0.05, 0.07), ell(0.0, -0.33, 0.025, 0.04),
        ell(0.15, 0.06, 0.05, 0.14, -35), ell(0.2, -0.12, 0.035, 0.09, 30), ell(0.13, 0.3, 0.045, 0.08, 50),
        ell(0.3, 0.32, 0.03, 0.05), ell(0.33, -0.3, 0.06, 0.035, 20)])
    ring = np.abs(np.sqrt((x - 0.3) ** 2 + (y + 0.1) ** 2) - 0.09) / 0.012 - 1.0
    frame = np.abs(x - 0.5 * np.abs(np.sin(np.pi * (y + 0.5)))) / 0.01 - 1.0
    d = np.minimum(np.minimum(d, ring), frame)
    m = c.inside(-d * W * 0.08)
    base = _c(pal[0]) * (1 + 0.04 * c.n(4, 2, 1))[..., None]
    mot = _c(pal[1]) * (1 + 0.06 * c.n(60, 3, 2))[..., None]
    return Maps(_lerp(base, mot, m), m * 0.3, depth=0.5, shine=0.2, rough=6)


@_gen("wallpaper", "man-made", "wallpaper geometric art deco fan pattern scallop retro wallpaper")
def _wallpaper(c, size=56.0, palette=None):
    """Geometric wallpaper: art-deco fans in overlapping rows with gold line work."""
    pal = palette or ["#123c3c", "#d4af5a", "#1c5050"]
    R = size
    best = np.full(c.U.shape, -1.0, F32)
    out_d = np.zeros(c.U.shape, F32)
    out_a = np.zeros(c.U.shape, F32)
    j0 = np.floor(c.V / (R * 0.5))
    for dj in (0, 1, 2):
        j = j0 + dj
        sh = np.mod(j, 2) * R
        i = np.floor((c.U - sh) / (2 * R) + 0.5)
        for di in (-1, 0, 1):
            cx = (i + di) * 2 * R + sh
            cy = j * R * 0.5
            dx, dy = c.U - cx, c.V - cy
            d = np.sqrt(dx * dx + dy * dy)
            inside = (d < R) & (dy <= 0)
            pri = np.where(inside, j, -1e9)
            take = pri > best
            best = np.where(take, pri, best)
            out_d = np.where(take, d, out_d)
            out_a = np.where(take, np.arctan2(dx, -dy), out_a)
    rings = np.abs(np.mod(out_d / (R / 5.0), 1.0) - 0.5) * (R / 5.0)
    rays = np.abs(np.mod(out_a / (np.pi / 8) + 0.5, 1.0) - 0.5) * out_d * (np.pi / 8)
    edge = R - out_d
    line = np.minimum(np.minimum(rings, np.where(out_d > R * 0.4, rays, 99.0)), edge)
    m = 1 - c.inside(line - 1.1)
    col = _lerp(np.broadcast_to(_c(pal[0]), c.U.shape + (3,)) * (1 + 0.05 * c.n(4, 2, 1))[..., None],
                np.broadcast_to(_c(pal[1]), c.U.shape + (3,)), m)
    return Maps(col, m * 0.3, depth=0.4, shine=0.25, rough=12)


def _tiles_common(c, k, row, d, palette, grout_color, bevel, vary, checker=False):
    pal = palette
    r = _rnd(k, row, c.s(1), 0)
    if checker:
        col = _pick(pal[:2], np.mod(k + row, 2) * 0.5 + 0.25)
    else:
        col = _pick(pal, r)
    col = col * (1 + vary * (_rnd(k, row, c.s(1), 1) - 0.5) + 0.03 * c.n(30, 3, 2))[..., None]
    t = c.inside(d)
    g = _c(grout_color) * (1 + 0.1 * c.n(3, 2, 3) * c.fade(3))[..., None]
    col = _lerp(g, col, t)
    return col, t * (0.6 + 0.4 * smoothstep(0, bevel, d))


@_gen("tiles", "man-made", "tiles square tiles ceramic tiles bathroom floor tiles kitchen tile grout checkerboard")
def _tiles(c, size=100.0, grout=5.0, palette=None, grout_color="#a29d93", checker=False):
    """Square ceramic tiles with grout lines, glazed bevels and slight tile-to-tile variation.
    checker=True alternates the first two palette colours."""
    pal = palette or (["#1c1c1e", "#ece8e0"] if checker else ["#e9e4da", "#efebe3", "#e2dccf"])
    k, row, d, fx, fy = _tile_grid(c, size, size, grout)
    col, hgt = _tiles_common(c, k, row, d, pal, grout_color, 4.0, 0.06, checker)
    return Maps(col, hgt, depth=3.0, shine=0.6, rough=50)


@_gen("subway_tiles", "man-made", "subway tiles metro tiles white brick tiles kitchen backsplash glazed")
def _subway(c, size=(150.0, 75.0), grout=4.0, palette=None, grout_color="#8f8c86"):
    """Subway tiles: glossy white 2:1 tiles in running bond with rounded bevels."""
    pal = palette or ["#eef0ee", "#e8ebe8", "#f2f3f0"]
    k, row, fx, fy, L = _courses(c.U, c.V, size[0], size[1], 0.5)
    d = np.minimum(np.minimum(fx, L - fx), np.minimum(fy, size[1] - fy)) - grout * 0.5
    col, hgt = _tiles_common(c, k, row, d, pal, grout_color, 7.0, 0.04)
    return Maps(col, np.sqrt(clamp01(hgt)), depth=3.5, shine=0.8, rough=60)


@_gen("hex_tiles", "man-made", "hexagon tiles hex tiles honeycomb tiles hexagonal floor bathroom mosaic")
def _hex_tiles(c, size=34.0, grout=3.0, palette=None, grout_color="#9b978f"):
    """Hexagon tiles with grout, mixed tones."""
    pal = palette or ["#f0ede6", "#f0ede6", "#e6e2d8", "#5e6b64", "#f0ede6", "#c9c4b8"]
    q, r, dx, dy, e = _hexgrid(c.U, c.V, size)
    d = e - grout * 0.5
    col, hgt = _tiles_common(c, q, r, d, pal, grout_color, 3.0, 0.05)
    return Maps(col, hgt, depth=2.5, shine=0.6, rough=50)


def _herring_index(U, V, n):
    """Herringbone of n:1 blocks (unit = block width): (block id a, b, is_vertical, local x, local y, length)."""
    jb = np.floor((U - V) / (2.0 * n))
    best_in = np.zeros(U.shape, bool)
    ka = np.zeros(U.shape, np.int64)
    kb = np.zeros(U.shape, np.int64)
    vert = np.zeros(U.shape, bool)
    lx = np.zeros(U.shape, F32)
    ly = np.zeros(U.shape, F32)
    for dj in (-1, 0, 1):
        j = jb + dj
        qx = U - j * n
        qy = V + j * n
        k = np.floor(qy)
        inh = (qx >= k) & (qx < k + n) & ~best_in
        ka = np.where(inh, k.astype(np.int64), ka)
        kb = np.where(inh, j.astype(np.int64), kb)
        lx = np.where(inh, qx - k, lx)
        ly = np.where(inh, qy - k, ly)
        best_in |= inh
        kv = np.floor(qx - n)
        inv = (qy >= kv + 1 - n) & (qy < kv + 1) & ~best_in
        ka = np.where(inv, kv.astype(np.int64), ka)
        kb = np.where(inv, j.astype(np.int64), kb)
        vert |= inv
        lx = np.where(inv, qy - (kv + 1 - n), lx)
        ly = np.where(inv, qx - kv - n, ly)
        best_in |= inv
    return ka, kb, vert, lx.astype(F32), ly.astype(F32)


@_gen("herringbone", "man-made", "herringbone parquet floor zigzag wood floor chevron parquet herringbone tiles")
def _herringbone(c, size=(240.0, 60.0), gap=1.6, angle=45.0, palette=None, wood=True, grout_color="#2a1a0e"):
    """Herringbone parquet: n:1 blocks alternating direction in a zigzag, each with its own grain and tone."""
    pal = palette or PAL["oak"]
    L, Wb = size
    n = max(int(round(L / Wb)), 2)
    U, V = c.rot(angle)
    ka, kb, vert, lx, ly = _herring_index(U / Wb, V / Wb, n)
    x, y = lx * Wb, ly * Wb
    d = np.minimum(np.minimum(x, n * Wb - x), np.minimum(y, Wb - y)) - gap * 0.5
    b = c.inside(d)
    r = _rnd(ka, kb * 2 + vert, c.s(1), 0)
    if wood:
        t, gh, _ = _grain(c, x + r * 3000, y + _rnd(ka, kb, c.s(1), 1) * 500, c.s(2), 9.0, 0.0)
        col = _ramp(pal, clamp01(0.62 + 0.35 * (r - 0.5) - 0.45 * t))
    else:
        col = _pick(pal, r)
        gh = 0
    col = col * (0.85 + 0.15 * smoothstep(0, 8, d))[..., None]
    col = _lerp(np.broadcast_to(_c(grout_color), col.shape), col, b)
    return Maps(col, b * (0.6 + 0.4 * smoothstep(0, 2.5, d)) + 0.05 * gh, depth=2.5, shine=0.15, rough=20)


# ======================================================================== organic

@_gen("lawn", "organic", "lawn grass mown grass turf garden lawn green grass football pitch golf")
def _lawn(c, length=28.0, width=3.2, palette=None, stripes=0.0, stripe_width=170.0, density=1.0, angle=-90.0,
          spread=12.0, layers=3):
    """Lawn: dense tapered grass blades (dark roots, light tips, mixed greens). stripes: 0~1 mown stripes."""
    pal = palette or PAL["grass"]
    ang = angle
    st = 0.0
    if stripes:
        st = smoothstep(-0.25, 0.25, np.sin(np.pi * c.U / stripe_width)) * 2 - 1
        ang = angle + 14.0 * stripes * st
    cw = width * 1.5 / max(density, 1e-3) ** 0.5
    o = _scatter(c, cw, _blade(c, length, width, 0.85, 0.25), _blade_extent(length, width, 0.25), layers=layers,
                 angle=ang, spread=spread, size=(0.7, 1.2), seed=c.s(1), aspect=length * 0.32 / cw)
    t = 0.12 + 0.72 * o["along"] ** 0.8 + 0.25 * (o["tone"] - 0.5)
    col = _ramp(pal, clamp01(t))
    warm = (o["r2"] > 0.72).astype(F32) * 0.6
    col = col * (1 + warm[..., None] * (np.array([1.12, 1.03, 0.7], F32) - 1))
    under = _ramp(pal, clamp01(0.06 + 0.06 * c.n(40, 3, 2)))
    col = _lerp(under, col, o["cov"])
    col = col * (1 + 0.12 * c.n(240, 3, 3) + 0.26 * stripes * st)[..., None]
    return Maps(col, o["height"], depth=2.0, shine=0.04)


@_gen("meadow", "organic", "meadow tall grass wild grass prairie field windswept grass savanna wheat field")
def _meadow(c, length=110.0, width=5.0, wind=0.6, palette=None, flowers=0.12, layers=3):
    """Meadow: tall curved grass leaning with the wind (direction varies softly), dry tips, a few flowers."""
    pal = palette or ["#1a2a10", "#3c5a1e", "#6e8a32", "#a8b15a", "#d8cf8e"]
    ang = -90.0 + 25.0 * wind + 12.0 * c.n(300, 3, 1)
    cw = width * 2.2
    o = _scatter(c, cw, _blade(c, length, width, 0.9, 0.3), _blade_extent(length, width, 0.3), layers=layers,
                 angle=ang, spread=8.0, size=(0.6, 1.15), seed=c.s(2), aspect=length * 0.3 / cw)
    t = 0.1 + 0.75 * o["along"] + 0.25 * (o["tone"] - 0.5)
    col = _ramp(pal, clamp01(t))
    under = _ramp(pal, clamp01(0.05 + 0.08 * c.n(60, 3, 3)))
    col = _lerp(under, col, o["cov"])
    if flowers:
        fl = c.cells(46.0, 1.0, 4)
        fm = (fl.rand < flowers) * c.inside(3.6 - fl.f1 * 46.0)
        fc = _pick(["#f4f1ea", "#f1d14a", "#b46fc9", "#e8e4f4"], fl.r(1))
        col = _lerp(col, fc, fm)
    col = col * (1 + 0.1 * c.n(260, 3, 5))[..., None]
    return Maps(col, o["height"], depth=2.5)


@_gen("moss", "organic", "moss mossy green moss forest floor moss cushion lichen")
def _moss(c, palette=None, size=60.0):
    """Moss: soft cushions of tiny leafy shoots, bright tops and dark crevices."""
    pal = palette or ["#141c07", "#2f4210", "#557318", "#8aa83a", "#c4d46c"]
    clump = c.n(size, 5, 1, mode="billow")
    cel = c.cells(3.0, 1.0, 2, round=0.3)
    dome = smoothstep(0.0, 0.4, cel.edge) * c.fade(3.0) + 0.5 * (1 - c.fade(3.0))
    t = 0.12 + 0.6 * clump + 0.25 * dome + 0.1 * (cel.rand - 0.5)
    col = _ramp(pal, clamp01(t)) * (1 + 0.1 * c.n(200, 3, 3))[..., None]
    return Maps(col, 1.2 * clump + 0.25 * dome, depth=10.0, shine=0.03)


@_gen("fur", "organic", "fur animal fur pelt hair coat fox fur cat fur dog fur fluffy")
def _fur(c, kind="short", flow=80.0, palette=None, length=None, width=None, density=1.0):
    """Fur: dense hair strands following a flow direction (degrees, or an (H, W) map in degrees),
    dark roots, light tips. kind: short / long / curly."""
    pal = palette or ["#1a0f08", "#4a2e19", "#8a5c35", "#c49565", "#e8c99a"]
    _choice("fur", "kind", kind, ("short", "long", "curly"))
    P = dict(short=(18.0, 1.7, 10.0, 0.15, 3), long=(64.0, 2.3, 6.0, 0.35, 3), curly=(24.0, 2.2, 15.0, 0.9, 3))[kind]
    L = length or P[0]
    Wd = width or P[1]
    ang = flow if np.ndim(flow) == 0 else c.crop(flow)
    ang = ang + 22.0 * c.n(160, 3, 1)
    if kind == "long":
        ang = ang + 10.0 * c.n(26, 2, 2)
    if kind == "curly":
        ang = ang + 160.0 * c.n(46, 2, 3)
    if kind == "curly":
        R = (length or 7.0)
        Wc = width or 2.4
        fw = c.fw

        def arc(lx, ly, k, r):
            rr = R * k
            d = np.sqrt(lx * lx + ly * ly)
            t = (np.arctan2(ly, lx) + np.pi) * F32(1 / (2 * np.pi))
            half = 0.5 * Wc * k * (1.15 - 0.6 * t)
            cov = _line_cov(np.abs(d - rr), half, fw) * clamp01((0.85 - t) / 0.04)
            ac = np.clip((d - rr) / np.maximum(half, 1e-3), -1, 1)
            return cov, np.sqrt(clamp01(1 - ac * ac)) * 0.6 + 0.4 * t, t, ac

        e = R * 1.25 + Wc
        o = _scatter(c, R * 1.15 / max(density, 1e-3) ** 0.5, arc, (-e, e, -e, e), layers=3, spread=180.0,
                     size=(0.75, 1.25), seed=c.s(4))
    else:
        cw = max(Wd * 1.6, L * 0.3) / max(density, 1e-3) ** 0.5
        o = _scatter(c, cw, _blade(c, L, Wd, 0.75, P[3]), _blade_extent(L, Wd, P[3]), layers=P[4], angle=ang,
                     spread=P[2], size=(0.7, 1.2), seed=c.s(4))
    t = 0.08 + 0.75 * o["along"] ** 1.3 + 0.2 * (o["tone"] - 0.5) + 0.1 * (1 - np.abs(o["across"]))
    col = _ramp(pal, clamp01(t))
    col = _lerp(_ramp(pal, np.full(c.U.shape, 0.05, F32)), col, o["cov"])
    col = col * (0.75 + 0.4 * clamp01(0.5 + c.n(110, 3, 5)))[..., None]
    return Maps(col, o["height"], depth=0.5, shine=0.1, rough=6)


_alias("fur_long", "fur", "organic", "long fur long hair shaggy coat goat yak mane flowing hair", kind="long")
_alias("fur_curly", "fur", "organic", "curly fur sheep wool fleece poodle astrakhan curls lamb", kind="curly")


@_gen("feathers", "organic", "feathers plumage bird feathers wing down owl pheasant")
def _feathers(c, size=60.0, palette=None, angle=90.0, bars=False):
    """Feathers: overlapping rows of pointed feathers with a central shaft, slanted barbs, downy bases.
    bars=True adds dark tip bands (pheasant / owl)."""
    pal = palette or ["#1d140d", "#4e3a26", "#8a6c48", "#c2a57c", "#ead8b8"]
    Lf, Wf = size, size * 0.42

    def shape(lx, ly, k, r):
        Ln, Wn = Lf * k, Wf * k
        s = lx / Ln
        sc = np.clip(s, 0, 1)
        half = 0.5 * Wn * np.maximum(np.sin(np.pi * sc ** 0.75), 0) ** 0.55 + 1e-3
        cov = c.inside(half - np.abs(ly)) * c.inside(lx) * c.inside(Ln - lx)
        ac = np.clip(ly / half, -1, 1)
        return cov, 0.1 + 0.8 * sc + 0.1 * (1 - np.abs(ac)), sc, ac

    cell = Wf * 0.8
    o = _scatter(c, cell, shape, (0, Lf, -Wf / 2, Wf / 2), jitter=0.25, angle=angle, spread=8.0,
                 size=(0.9, 1.1), seed=c.s(1), aspect=Lf * 0.42 / cell, stagger=0.5, order=-1)
    al, ac = o["along"], o["across"]
    barb = 0.5 + 0.5 * np.cos(2 * np.pi * (al * 16 + np.abs(ac) * 6)) * c.fade(size / 16)
    t = 0.3 + 0.45 * al + 0.15 * (o["tone"] - 0.5) - 0.1 * barb - 0.25 * smoothstep(0.7, 1.0, np.abs(ac))
    if bars:
        t = t - 0.3 * np.exp(-((al - 0.8) / 0.05) ** 2) + 0.2 * np.exp(-((al - 0.92) / 0.035) ** 2)
    col = _ramp(pal, clamp01(t))
    rach = np.exp(-(ac / 0.05) ** 2) * smoothstep(0.02, 0.1, al) * (1 - smoothstep(0.75, 0.9, al))
    col = _lerp(col, np.broadcast_to(_c(pal[-2]), col.shape), rach * 0.5)
    col = _lerp(np.broadcast_to(_c(pal[0]), col.shape), col, o["cov"])
    return Maps(col, o["height"], depth=9.0, shine=0.08, rough=8)


@_gen("fish_scales", "organic", "fish scales fish skin mermaid scales carp koi salmon scales")
def _fish_scales(c, size=34.0, palette=None):
    """Fish scales: overlapping round scales in staggered rows, silver iridescence, darker back."""
    pal = palette or ["#1b2f40", "#3f6a86", "#86b2c6", "#dcecf0"]
    R = size * 0.62

    def shape(lx, ly, k, r):
        d = np.sqrt(lx * lx + ly * ly)
        cov = c.inside(R * k - d)
        return cov, 0.25 + 0.75 * clamp01(ly / (R * k) * 0.5 + 0.5), d / (R * k), ly / (R * k)

    o = _scatter(c, size, shape, (-R * 1.05, R * 1.05, -R * 1.05, R * 1.05), jitter=0.0, size=(1.0, 1.0),
                 seed=c.s(1), aspect=0.5, stagger=0.5, order=-1)
    al = o["along"]
    t = 0.25 + 0.5 * clamp01(o["across"] * 0.5 + 0.5) + 0.15 * (o["tone"] - 0.5) + 0.2 * (0.5 - c.rv)
    col = _ramp(pal, clamp01(t))
    ph = al * 5 + o["tone"] * 6
    iri = np.stack([1 + 0.1 * np.sin(ph), 1 + 0.05 * np.sin(ph + 2), 1 + 0.1 * np.sin(ph + 4)], -1)
    col = col * iri * (1 - 0.35 * smoothstep(0.86, 1.0, al))[..., None]
    return Maps(col.astype(F32), o["height"], depth=3.0, shine=0.6, rough=30)


@_gen("reptile_scales", "organic", "reptile scales snake skin lizard skin dragon scales crocodile scales")
def _reptile(c, size=20.0, palette=None, pattern="#3a2a14"):
    """Reptile scales: domed, nearly regular scales with dark seams and a blotched pattern across them."""
    pal = palette or ["#2f3a18", "#55652a", "#7f8f3e", "#b3b766"]
    cel = c.cells(size, 0.45, 1, round=0.25)
    eu = cel.edge * size
    sc = c.inside(eu - 0.8)
    dome = np.sqrt(clamp01(eu / (size * 0.4)))
    bl = smoothstep(-0.05, 0.1, c.n(110, 4, 2))
    col = _ramp(pal, clamp01(0.35 + 0.4 * dome + 0.15 * (cel.rand - 0.5)))
    col = _lerp(col, np.broadcast_to(_c(pattern), col.shape) * (0.8 + 0.4 * dome)[..., None], bl * 0.85)
    col = _lerp(np.broadcast_to(_c("#10120a"), col.shape), col, sc)
    return Maps(col, sc * (0.3 + 0.7 * dome), depth=size * 0.25, shine=0.3, rough=24)


@_gen("leaves", "organic", "leaves fallen leaves autumn leaves leaf litter forest floor foliage carpet of leaves")
def _leaves(c, size=56.0, palette=None, layers=3):
    """Carpet of fallen leaves: overlapping pointed leaves in autumn colours with midribs and veins."""
    pal = palette or PAL["autumn"]
    Ll, Wl = size, size * 0.5

    def shape(lx, ly, k, r):
        Ln, Wn = Ll * k, Wl * k
        s = lx / (Ln * 0.5)
        half = 0.5 * Wn * np.sqrt(clamp01(1 - s * s)) ** 0.9 * (1 - 0.2 * s) + 1e-3
        cov = c.inside(half - np.abs(ly))
        ac = np.clip(ly / half, -1, 1)
        return cov, 0.35 + 0.45 * (1 - ac * ac) + 0.2 * r, clamp01(s * 0.5 + 0.5), ac

    o = _scatter(c, size * 0.55, shape, (-Ll / 2, Ll / 2, -Wl / 2, Wl / 2), layers=layers, spread=180.0,
                 size=(0.7, 1.2), seed=c.s(1))
    al, ac = o["along"], o["across"]
    col = _pick(pal, o["tone"]) * (0.8 + 0.35 * o["r2"])[..., None]
    vein = (0.5 + 0.5 * np.cos(2 * np.pi * (al * 7 - np.abs(ac) * 2.2))) * c.fade(size / 7)
    mid = np.exp(-(ac / 0.06) ** 2)
    col = col * (1 - 0.12 * vein + 0.2 * mid)[..., None]
    spots = smoothstep(0.35, 0.7, c.n(14, 3, 2)) * 0.5
    col = _lerp(col, np.broadcast_to(_c("#4a2c14"), col.shape), spots * o["cov"])
    col = _lerp(np.broadcast_to(_c("#20140b"), col.shape), col, o["cov"])
    return Maps(col, o["height"], depth=4.0, shine=0.1, rough=12)


@_gen("hay", "organic", "hay straw haystack bale thatch dry grass")
def _hay(c, length=90.0, width=3.6, angle=-10.0, palette=None, layers=3):
    """Hay / straw: long hollow golden stalks lying mostly one way, tubular highlights, dark gaps."""
    pal = palette or ["#4a3512", "#8f6f2a", "#c9a64e", "#ecd58a"]
    cell = length * 0.3
    o = _scatter(c, cell, _blade(c, length, width, 0.15, 0.08), _blade_extent(length, width, 0.08),
                 layers=layers, angle=angle, spread=18.0, size=(0.6, 1.2), seed=c.s(1), aspect=width * 2.0 / cell)
    ac = o["across"]
    t = 0.3 + 0.45 * (1 - ac * ac) + 0.25 * (o["tone"] - 0.5)
    col = _ramp(pal, clamp01(t))
    col = _lerp(np.broadcast_to(_c("#2a1d0a"), col.shape), col, o["cov"])
    return Maps(col, o["height"], depth=3.0, shine=0.25, rough=14)


@_gen("wicker", "organic", "wicker rattan basket weave cane woven basket chair")
def _wicker(c, pitch=26.0, palette=None, pattern="plain"):
    """Wicker / rattan: wide rounded canes woven over and under, fibre lines, dark gaps between."""
    pal = palette or ["#4e3418", "#9c7442", "#d2ac72", "#ecd3a0"]
    w = _weave(c, pitch, pattern, gap=0.2, wobble=0.04)
    hole = 1 - smoothstep(0.02, 0.12, w["h"])
    fl = 0.5 + 0.5 * np.sin(2 * np.pi * np.where(w["t"] > 0.5, w["x"], w["y"]) * 5) * c.fade(pitch / 5)
    t = 0.25 + 0.6 * w["h"] + 0.08 * fl + 0.1 * w["slub"]
    col = _ramp(pal, clamp01(t))
    col = _lerp(col, np.broadcast_to(_c("#140c05"), col.shape), hole)
    return Maps(col, w["h"], depth=pitch * 0.3, shine=0.2, rough=12)


@_gen("rattan", "organic", "rattan cane webbing caning chair seat cane mesh bentwood thonet")
def _rattan(c, pitch=30.0, palette=None):
    """Rattan cane webbing: pairs of horizontal and vertical canes crossed by diagonals, octagonal open holes."""
    pal = palette or ["#5c3d1a", "#a77b42", "#d8b276", "#f0d7a4"]
    p = pitch
    hgt = np.zeros(c.U.shape, F32)
    cov = np.zeros(c.U.shape, F32)
    for k, (ang, spacing, gap, wd) in enumerate(((90, p, 0.13 * p, 0.075 * p), (0, p, 0.13 * p, 0.075 * p),
                                                  (45, p / _SQ3 * 1.2247, 0.0, 0.07 * p),
                                                  (135, p / _SQ3 * 1.2247, 0.0, 0.07 * p))):
        u, _ = c.rot(ang)
        d = np.abs(np.mod(u + spacing * 0.5, spacing) - spacing * 0.5)
        if gap:
            d = np.abs(d - gap)
        m = c.inside(wd - d)
        prof = np.sqrt(clamp01(1 - (d / wd) ** 2))
        h = 0.3 * k + 0.6 * prof
        hgt = hgt + (h - hgt) * m
        cov = cov + (1 - cov) * m
    fib = c.n(p * 2, 3, 1, aspect=0.1) * 0.08
    col = _ramp(pal, clamp01(0.35 + 0.45 * (hgt - np.floor(hgt / 0.3) * 0.3) / 0.6 + fib + 0.1 * hgt))
    col = _lerp(np.broadcast_to(_c("#120c06"), col.shape), col, cov)
    return Maps(col, hgt, alpha=cov, depth=4.0, shine=0.25, rough=14)


@_gen("honeycomb", "organic", "honeycomb beehive honey wax comb bees hexagon")
def _honeycomb(c, size=30.0, capped=0.45):
    """Honeycomb: hexagonal wax cells, some filled with glossy amber honey, some capped with pale wax."""
    q, r, dx, dy, e = _hexgrid(c.U, c.V, size)
    wall = 1 - c.inside(e - 2.2)
    d = np.sqrt(dx * dx + dy * dy) / (size * 0.85)
    cap = (_rnd(q, r, c.s(1), 0) < capped).astype(F32)
    honey = _ramp(["#4a1f02", "#9a4e06", "#dc8f18", "#f7c14e"], clamp01(0.3 + 0.6 * d ** 1.5 + 0.1 * c.n(20, 2, 2)))
    capc = _ramp(["#c9a24e", "#e9cf86", "#f6e4a8"], clamp01(0.7 - 0.4 * d + 0.1 * c.n(6, 2, 3)))
    col = _lerp(honey, capc, cap)
    col = _lerp(col, np.broadcast_to(_c("#f0d48a"), col.shape), wall)
    hgt = wall * 1.0 + (1 - wall) * (cap * (0.75 - 0.25 * d * d) + (1 - cap) * (0.15 + 0.25 * d * d))
    return Maps(col, hgt, depth=6.0, shine=0.8, rough=40)


# ======================================================================== liquids and misc

@_gen("water", "liquid", "water still water lake pond calm water surface ripples reflection")
def _water(c, sky=("#b9d3e6", "#5f8db0"), deep="#123446", ripple=1.0):
    """Still water: gentle wind ripples that reflect a sky tint (more towards the top), dark depth below,
    small glints."""
    h = c.n(150, 4, 1, aspect=0.22) + 0.5 * c.n(40, 3, 2, aspect=0.3)
    gx, gy = c.grad(h * 8.0 * ripple)
    sk = mix([(0, sky[1]), (1, sky[0])], clamp01(0.55 - 1.6 * gy + 0.3 * (0.5 - c.rv)))
    fres = clamp01(0.85 - 0.45 * c.rv + 0.5 * gy)
    col = _lerp(np.broadcast_to(_c(deep), sk.shape), sk, fres)
    gl = smoothstep(0.16, 0.26, -gy) * smoothstep(0.4, 0.7, c.n(30, 2, 3)) * 0.6
    return Maps(col, None, emit=gl[..., None] * _c("#ffffff"))


@_gen("caustics", "liquid", "pool caustics swimming pool water caustic light underwater sunlight pool floor")
def _caustics(c, size=60.0, floor=None, tiles=True):
    """Pool caustics: a bright net of refracted sunlight dancing on a turquoise pool floor."""
    pal = floor or ["#0b5a73", "#1d8eaa", "#4cc0d2"]
    base = _ramp(pal, clamp01(0.55 + 0.25 * c.n(220, 3, 1)))
    if tiles:
        k, row, d, fx, fy = _tile_grid(c, 48, 48, 2.0)
        base = base * (0.88 + 0.12 * c.inside(d))[..., None]
    net = 0.0
    for sz, sd, a in ((size, 2, 1.0), (size * 0.62, 5, 0.7)):
        w1, w2 = c.n(sz * 2, 3, sd), c.n(sz * 2, 3, sd + 1)
        cel = c.cells(sz, 1.0, sd + 2, U=c.U + sz * 0.45 * w1, V=c.V + sz * 0.45 * w2)
        net = net + a * np.exp(-(cel.edge / 0.06) ** 2)
    light = clamp01(net) ** 1.4
    emit = light[..., None] * _c((0.75, 1.0, 0.98)) * 0.9
    return Maps(base * (0.85 + 0.25 * light)[..., None], None, emit=emit)


@_gen("ice", "liquid", "ice frozen lake ice sheet cracked ice frost glacier frozen")
def _ice(c, palette=None, frost=0.5):
    """Ice: clear blue depth with bright internal cracks, trapped bubbles and feathery frost."""
    pal = palette or ["#3f6f96", "#86b4d3", "#cfe7f4"]
    col = _ramp(pal, clamp01(0.5 + 0.35 * c.n(220, 4, 1) + 0.15 * c.n(40, 3, 2)))
    big = c.cells(120, 1.0, 3, warp=0.3)
    cr = np.exp(-(big.edge * 120 / 1.3) ** 2)
    sm = c.cells(42, 1.0, 4, warp=0.3)
    cr = cr + 0.6 * np.exp(-(sm.edge * 42 / 0.9) ** 2) * smoothstep(0.0, 0.4, c.n(100, 3, 5))
    glow = np.exp(-(big.edge * 120 / 6.0) ** 2) * 0.25
    bb = c.cells(14, 1.0, 6)
    bub = (bb.rand < 0.08) * np.exp(-((bb.f1 * 14 - 1.6) / 0.5) ** 2)
    fr = c.n(9, 4, 7, mode="ridged") * smoothstep(0.05, 0.5, c.n(160, 3, 8) + frost - 0.5) * frost
    m = clamp01(0.9 * cr + glow + 0.8 * bub + 0.7 * smoothstep(0.55, 0.9, fr))
    col = _lerp(col, np.broadcast_to(_c("#f4fbff"), col.shape), m)
    return Maps(col, 0.3 * fr - 0.2 * cr, depth=1.0, shine=0.6, rough=60)


@_gen("snow", "liquid", "snow snowfield fresh snow snowdrift winter ground powder snow")
def _snow(c, drifts=1.0, sparkle=1.0, shadow="#93a9d1"):
    """Snow: soft rolling drifts with blue shadows and tiny sparkles."""
    h = c.n(260, 5, 1) * drifts + 0.15 * c.n(30, 3, 2)
    diff, spec, _ = c.shade(h, 30.0, ambient=0.3)
    t = clamp01((diff - 0.45) / 0.8)
    col = mix([(0, shadow), (0.6, "#e4ecf8"), (1, "#ffffff")], t)
    sp = c.cells(7.0, 1.0, 3)
    s = (sp.rand < 0.05) * np.exp(-(sp.f1 * 7.0 / 0.5) ** 2) * sparkle * smoothstep(0.6, 1.1, diff)
    return Maps(col, h, lit=col.astype(F32), emit=(s * 1.6)[..., None] * _c("#ffffff"))


@_gen("lava", "liquid", "lava magma molten rock volcanic lava flow glowing cracks")
def _lava(c, size=80.0, heat=1.0):
    """Lava: dark rough crust plates split by glowing cracks with a white-hot core (HDR emission)."""
    cel = c.cells(size, 1.0, 1, warp=0.25, round=0.05)
    w = 0.04 + 0.06 * clamp01(c.n(160, 3, 2) + 0.5)
    e = cel.edge
    glow = np.exp(-(e / w) ** 2)
    rough = c.n(14, 4, 3, mode="ridged")
    crust = smoothstep(w * 0.6, w * 1.4, e)
    col = _ramp(["#0b0706", "#1e1512", "#3b2b24"], clamp01(0.3 + 0.6 * rough * crust))
    col = _lerp(np.broadcast_to(_c("#3a0700"), col.shape), col, crust)
    hot = glow ** 1.3 * heat
    emit = _ramp(["#3a0500", "#c41f00", "#ff6a00", "#ffd76a"], clamp01(hot)) * (hot * 1.8)[..., None]
    emit = emit + _c("#ff3a00") * (np.exp(-(e / (w * 3.0)) ** 2) * 0.22 * heat)[..., None]
    hgt = crust * (0.5 + 0.3 * rough + 0.4 * smoothstep(0, 0.3, e))
    return Maps(col, hgt, emit=emit, depth=8.0, shine=0.1)


@_gen("fire", "liquid", "fire flames campfire fireplace blaze burning inferno flame bonfire")
def _fire(c, size=90.0, height=1.0, palette=None):
    """Fire: tongues of flame rising from the bottom of the region, white-yellow core, red tips, black smoke
    gaps (HDR emission; alpha follows the flames)."""
    pal = palette or ["#000000", "#4a0800", "#b82400", "#ff6a0c", "#ffc24a", "#fff4d6"]
    up = 1.0 - c.rv
    q = c.n(size * 0.7, 3, 1)
    n = fbm2d_at((c.U + size * 0.6 * q) / size, c.V / (size * 2.4), 5, seed=c.s(2))
    n2 = fbm2d_at(c.U / (size * 0.35), c.V / (size * 0.9), 3, seed=c.s(3))
    flame = clamp01(0.62 + 0.75 * n + 0.25 * n2 - up * 1.35 / height)
    heat = clamp01(flame * 1.25) ** 1.3
    col = _ramp(pal, heat)
    alpha = smoothstep(0.0, 0.12, flame)
    return Maps(col * 0.3, None, alpha=alpha, emit=col * (0.4 + 1.4 * heat)[..., None])


@_gen("mud", "liquid", "mud muddy ground wet mud puddle swamp mire")
def _mud(c, palette=None, wet=0.6):
    """Mud: lumpy wet earth with glossy puddles reflecting the sky."""
    pal = palette or ["#1f150c", "#3a2817", "#5b4127", "#76583a"]
    lump = c.n(70, 5, 1, mode="billow")
    pool = smoothstep(0.05, -0.15, c.n(240, 4, 2) + 0.3 * (0.5 - wet))
    col = _ramp(pal, clamp01(0.3 + 0.5 * lump - 0.15 * pool))
    sky = mix([(0, "#8c8172"), (0.5, "#5a4a38"), (1, "#2e2318")], clamp01(c.rv + 0.2 * c.n(60, 2, 4)))
    col = _lerp(col, sky, pool * 0.8)
    rim = np.exp(-((c.n(240, 4, 2) + 0.3 * (0.5 - wet) + 0.05) / 0.05) ** 2) * 0.35
    col = col * (1 - rim)[..., None]
    hgt = lump * (1 - pool) * 0.8 + 0.1 * c.n(12, 3, 5) * (1 - pool)
    return Maps(col, hgt, depth=7.0, shine=0.2 + 0.4 * wet, rough=18)


@_gen("clouds", "liquid", "clouds cloudy sky cumulus sky texture fluffy clouds overcast")
def _clouds(c, coverage=0.5, sky=("#3f7cc4", "#9cc4ea"), size=260.0):
    """Clouds seen as a flat texture: billowing white clouds with grey shaded undersides on blue sky."""
    q = c.n(size, 3, 1)
    den = fbm2d_at((c.U + size * 0.4 * q) / size, (c.V + size * 0.4 * c.n(size, 3, 2)) / size, 6, seed=c.s(3),
                   mode="billow")
    d = smoothstep(0.5 - coverage * 0.35, 0.64 - coverage * 0.3, den)
    gx, gy = c.grad(d * 20.0)
    lit = clamp01(0.75 - (gx * c.L[0] + gy * c.L[1]) * 0.8)
    sk = mix([(0, sky[0]), (1, sky[1])], c.rv)
    cl = mix([(0, "#8e9db2"), (0.6, "#e4e9f0"), (1, "#ffffff")], lit)
    return Maps(_lerp(sk, cl, d), None)


@_gen("starfield", "liquid", "stars starfield night sky space galaxy cosmos milky way nebula")
def _starfield(c, density=1.0, nebula=0.5):
    """Starfield: thousands of stars of different brightness and colour over a faint nebula (HDR)."""
    base = mix([(0, "#02040b"), (1, "#0a0f26")], clamp01(0.5 + 0.5 * c.n(400, 3, 1)))
    neb = smoothstep(0.0, 0.7, c.n(260, 5, 2)) * nebula
    nc = mix([(0, "#3b1a5e"), (1, "#1a4c6e")], clamp01(0.5 + c.n(180, 3, 3)))
    col = base + nc * (neb * 0.35)[..., None]
    emit = np.zeros(col.shape, F32)
    temps = ["#9fb6ff", "#ffffff", "#fff1d6", "#ffd2a8"]
    for sz, keep, r, b, sd in ((6.0, 0.22, 0.55, 1.2, 4), (16.0, 0.25, 0.85, 2.2, 5), (50.0, 0.3, 1.3, 4.0, 6)):
        cel = c.cells(sz, 1.0, sd)
        rr = np.maximum(r, 0.7 * c.fw)
        g = np.exp(-(cel.f1 * sz / rr) ** 2) * (cel.rand < keep * density) * (r / rr) ** 2
        emit = emit + _pick(temps, cel.r(2)) * (g * b * (0.25 + 0.75 * cel.r(1) ** 2))[..., None]
    return Maps(col, None, emit=emit)


@_gen("circuit", "man-made", "circuit board pcb electronics traces motherboard printed circuit tech")
def _circuit(c, size=24.0, board="#0e4a2a", trace="#2f8a4e", pad="#d6b24c"):
    """Circuit board: right-angle copper traces on green solder mask, gold pads at trace ends, vias."""
    x, y = c.U / size, c.V / size
    j = np.floor(y)
    fy = y - j - 0.5
    a = np.floor(x - 0.5)
    i = np.floor(x)
    fx = x - i - 0.5
    b = np.floor(y - 0.5)
    sd = c.s(1)

    def H(ii, jj):
        return _rnd(ii.astype(np.int64), jj.astype(np.int64), sd, 1) < 0.42

    def Vv(ii, jj):
        return _rnd(ii.astype(np.int64), jj.astype(np.int64), sd, 2) < 0.34

    dh = np.where(H(a, j), np.abs(fy), 9.0)
    dv = np.where(Vv(i, b), np.abs(fx), 9.0)
    tr = 1 - c.inside(np.minimum(dh, dv) * size - 1.6)
    deg = H(i, j).astype(int) + H(i - 1, j) + Vv(i, j) + Vv(i, j - 1)
    nd = np.sqrt(fx * fx + fy * fy) * size
    via = (deg == 0) & (_rnd(i.astype(np.int64), j.astype(np.int64), sd, 3) < 0.2)
    padm = ((deg == 1) | via) * (1 - c.inside(nd - np.where(via, 3.0, 4.6)))
    holem = ((deg == 1) | via) * (1 - c.inside(nd - 1.4))
    weave = _weave(c, 3.0, "plain", gap=0.1, wobble=0.0)
    col = _c(board) * (1 + 0.06 * (weave["h"] - 0.5) + 0.05 * c.n(120, 3, 2))[..., None]
    col = _lerp(col, np.broadcast_to(_c(trace), col.shape), tr)
    col = _lerp(col, np.broadcast_to(_c(pad), col.shape), padm)
    col = _lerp(col, np.broadcast_to(_c("#0a0b08"), col.shape), holem)
    hgt = 0.5 * tr + 0.8 * padm - 0.8 * holem
    return Maps(col, hgt, depth=1.2, shine=0.3, rough=30)


@_gen("lined_paper", "man-made", "lined paper notebook ruled paper school paper exercise book writing paper")
def _lined_paper(c, spacing=34.0, margin=110.0, line="#7fa6d6", margin_color="#e06a6a", color="#f7f0de"):
    """Ruled notebook paper: a light warm sheet (``color=``), blue lines, red margin line."""
    base = _GENS["paper"](c, color=color).color
    fy = np.mod(c.V, spacing)
    ln = 1 - c.inside(np.abs(fy - spacing * 0.5) - 0.7)
    x0 = float(c.U.min()) + margin
    mg = 1 - c.inside(np.abs(c.U - x0) - 0.9)
    col = _lerp(base, np.broadcast_to(_c(line), base.shape), ln * 0.85)
    col = _lerp(col, np.broadcast_to(_c(margin_color), base.shape), mg * 0.9)
    return Maps(col, None)


@_gen("graph_paper", "man-made", "graph paper grid paper squared paper engineering paper millimetre paper")
def _graph_paper(c, minor=16.0, major=80.0, color="#5ba08a", paper="#f7f0de"):
    """Graph paper: fine grid every ``minor`` units (``color=`` the grid colour, ``paper=`` the sheet), bolder lines
    every ``major``."""
    base = _GENS["paper"](c, color=paper).color

    def gl(p, w):
        dx = np.abs(np.mod(c.U + p * 0.5, p) - p * 0.5)
        dy = np.abs(np.mod(c.V + p * 0.5, p) - p * 0.5)
        return 1 - c.inside(np.minimum(dx, dy) - w)

    m = np.maximum(gl(minor, 0.35) * 0.45, gl(major, 0.9) * 0.85)
    return Maps(_lerp(base, np.broadcast_to(_c(color), base.shape), m), None)


@_gen("film_grain", "man-made", "film grain noise grain photographic grain analog film texture overlay")
def _film_grain(c, amount=0.12, clump=0.5, color=False):
    """Film grain: mid-grey (0.5) photographic grain to lay over a picture with ``mode="overlay"``."""
    y0, y1, x0, x1, ss = c.box
    yy, xx = np.mgrid[0:c.h, 0:c.w]
    yy = yy + y0 * ss
    xx = xx + x0 * ss
    sd = c.s(1)
    g = (_rnd(xx, yy, sd, 1) + _rnd(xx, yy, sd, 2) + _rnd(xx, yy, sd, 3) - 1.5) * 1.15
    g = _f.gaussian_blur(g.astype(F32), 0.55) * 1.6
    g = g * (1 - clump) + clump * (g * (0.6 + 0.8 * clamp01(0.5 + c.n(4, 2, 2))))
    if color:
        g2 = (_rnd(xx, yy, sd, 4) - 0.5) * 0.5
        col = 0.5 + amount * np.stack([g + g2, g, g - g2], -1)
    else:
        col = 0.5 + amount * np.repeat(g[..., None], 3, -1)
    return Maps(col.astype(F32), None)


@_gen("brushstrokes", "man-made", "brush strokes paint strokes oil paint impasto painterly canvas paint")
def _brushstrokes(c, length=70.0, width=16.0, palette=None, flow=-30.0):
    """Painterly brush strokes: thick dabs of paint following a flowing direction, bristle grooves, impasto."""
    pal = palette or ["#1f3b5c", "#2f6f8f", "#e0b04a", "#d9643a", "#efe4cf", "#3f7a52"]
    ang = flow + 35.0 * c.n(260, 3, 1)
    o = _scatter(c, width * 1.4, _blade(c, length, width, 0.25, 0.3), _blade_extent(length, width, 0.3),
                 layers=2, angle=ang, spread=15.0, size=(0.6, 1.2), seed=c.s(2), aspect=length * 0.4 / (width * 1.4))
    col = _pick(pal, o["tone"]) * (0.85 + 0.25 * o["r2"])[..., None]
    br = (0.5 + 0.5 * np.sin(o["across"] * 9 + o["r2"] * 20)) * c.fade(width / 9)
    col = col * (0.9 + 0.15 * br)[..., None]
    col = _lerp(np.broadcast_to(_c("#efe8da"), col.shape), col, o["cov"])
    return Maps(col, o["height"] + 0.2 * br * o["cov"], depth=4.0, shine=0.3, rough=16)


# ======================================================================== registry & one-call API

_FAMILY_TITLES = {"wood": "Wood", "stone": "Stone & ground", "fabric": "Fabric", "metal": "Metal",
                  "man-made": "Man-made", "organic": "Organic", "liquid": "Liquids & sky"}

_STEM = re.compile(r"(?:en|s|es|ed|ing|y)$")


def _tok(s):
    out = []
    for t in re.findall(r"[a-z0-9]+", s.lower()):
        out.append(t)
        st = _STEM.sub("", t) if len(t) > 4 else t
        if st != t:
            out.append(st)
    return out


def names(family=None):
    """All texture names, or those of one family (wood / stone / fabric / metal / man-made / organic / liquid)."""
    return list(TEXTURES) if family is None else list(FAMILIES.get(family, []))


def find(query, n=1):
    """Keyword search: ``find("wooden floor")`` -> ``"planks"``. n > 1 returns the n best names."""
    if query in TEXTURES and n == 1:
        return query
    q = _tok(query.replace("_", " "))
    scores = []
    for i, (name, (fam, kw)) in enumerate(_META.items()):
        nt = set(_tok(name.replace("_", " ")))
        kt = set(_tok(kw))
        s = 0.0
        for t in q:
            if t in nt:
                s += 3
            elif t in kt:
                s += 1.5
            elif any(k.startswith(t) or t.startswith(k) for k in kt | nt if len(k) > 3 and len(t) > 3):
                s += 0.5
        phrase = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
        if phrase and phrase in kw:
            s += 2
        scores.append((-s, i, name))
    scores.sort()
    best = [nm for s, _, nm in scores if s < 0]
    if query in TEXTURES:
        best = [query] + [b for b in best if b != query]
    if n == 1:
        return best[0] if best else None
    return best[:n]


def describe(name=None):
    """One line per texture: ``name -> first docstring sentence`` (or the full docstring of one texture)."""
    if name:
        return TEXTURES[_resolve(name)].__doc__
    return "\n".join(f"{k}: {(TEXTURES[k].__doc__ or '').split(chr(10))[0]}" for k in TEXTURES)


class _UnknownName(ValueError, KeyError):
    """Unknown registry name (a ValueError that is also a KeyError)."""
    __str__ = ValueError.__str__


def _resolve(name):
    if name in TEXTURES:
        return name
    k = find(name)
    if k is None:
        e = _unknown_name("texture", name, list(TEXTURES), "Try textures.find('...') or textures.names()")
        raise _UnknownName(*e.args)
    return k


def texture(name, h, w, scale=1.0, seed=0, **kw):
    """Make a texture by name (or by description: "red brick wall" -> brick) -> lit (H, W, 3) colour map.
    Keywords: uv, mask, light, bump, maps, ss and the texture's own options (palette, size, ...); options are
    checked (unknown names raise TypeError with the closest valid one) and both spellings are accepted."""
    return TEXTURES[_resolve(name)](h, w, scale=scale, seed=seed, **kw)


def fill(cv, mask, name, light=LIGHT, opacity=1.0, mode="normal", shadow=0.0, **kw):
    """Paint a texture into ``mask`` on canvas ``cv`` with relief lighting (only the mask's box is computed).
    shadow > 0 adds a soft contact shadow around the mask first. Returns the canvas."""
    mask = np.asarray(mask, F32)
    if shadow:
        S = cv.h / 1080.0
        sh = _f.gaussian_blur(mask, 10 * S)
        sh = np.roll(np.roll(sh, int(round(-light[1] * 8 * S)), 0), int(round(-light[0] * 8 * S)), 1)
        cv.paint("#000000", clamp01(sh * (1 - mask)) * shadow)
    m = TEXTURES[_resolve(name)](cv.h, cv.w, mask=mask, light=light, maps=True, **kw)
    a = mask if m.alpha is None else mask * m.alpha
    cv.paint(m.lit, a, mode=mode, opacity=opacity)
    return cv


# ======================================================================== effects

def patches(h, w, coverage=0.4, size=120.0, seed=0, softness=0.12, bias=None, mask=None, scale=1.0):
    """Organic patch mask (moss, snow, rust, dirt placement): noise thresholded to ``coverage`` 0~1.
    bias: (H, W) array added before thresholding (e.g. ``1 - V`` to favour the bottom); mask multiplies."""
    S = h / 1080.0
    n = fbm2d(h, w, size * S * scale, 5, seed=seed * 7919 + 3) + 0.3 * fbm2d(h, w, size * 0.3 * S * scale, 3,
                                                                             seed=seed * 7919 + 5)
    if bias is not None:
        n = n + np.asarray(bias, F32)
    from statistics import NormalDist
    t = 0.2 * NormalDist().inv_cdf(min(max(1.0 - float(coverage), 0.005), 0.995))   # fBm is ~normal, sigma ~0.2
    m = smoothstep(t - softness, t + softness, n)
    return m if mask is None else m * np.asarray(mask, F32)


def weather(cv, mask=None, amount=0.5, seed=0, streaks=True, grime="#2a2118", scale=1.0):
    """Age a surface already painted on ``cv``: grime blotches, rain streaks running down, darker edges.
    Paints through ``cv.paint`` (multiply) inside ``mask``. Returns the canvas."""
    h, w = cv.h, cv.w
    S = h / 1080.0
    base = smoothstep(-0.1, 0.6, fbm2d(h, w, 260 * S * scale, 5, seed=seed * 31 + 1))
    k = base * 0.6
    if streaks:
        st = smoothstep(0.05, 0.5, fbm2d(h, w, 34 * S * scale, 4, seed=seed * 31 + 2, aspect=9))
        k = k + st * 0.5 * smoothstep(-0.3, 0.4, fbm2d(h, w, 300 * S * scale, 3, seed=seed * 31 + 3))
    a = clamp01(k * amount)
    if mask is not None:
        a = a * np.asarray(mask, F32)
    col = _c(grime)
    cv.paint(1 - (1 - col) * 0.9, a, mode="multiply")
    return cv


# ---- scene manifest: each top-level call records what it drew (pixelcanvas.manifest) ----
from . import manifest as _manifest  # noqa: E402

_manifest.register(globals(), "texture", ["fill"], arg="name", alias="texture")
_manifest.register(globals(), "effect", ["weather"])


# ---- strict options: unknown keyword names raise TypeError naming the closest valid option; colour / color,
# ---- w / width ... are accepted (pixelcanvas._args). Texture options are checked in ``_run``.
for _n, _f_ in list(globals().items()):
    if (not _n.startswith("_") and callable(_f_) and getattr(_f_, "__module__", None) == __name__
            and not isinstance(_f_, type)):
        globals()[_n] = _strict(_f_, _n)
