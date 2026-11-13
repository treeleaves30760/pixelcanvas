# -*- coding: utf-8 -*-
"""Noise: Perlin / fBm / ridged / billow / domain warp (2D and 1D).

This is the source of terrain, clouds, water ripples and materials. Every function is deterministic (same seed, same result)
and a continuous field in "cell" units: changing resolution only resamples it, the pattern stays the same.
Scales are in pixels: write ``scale = 120 * S`` with ``S = H / 1080`` so a half-size draft keeps the same pattern.
``mode`` is "plain", "ridged" or "billow".
"""
from __future__ import annotations

import numpy as np

from .color import F32, grid



def _scipy_ndimage():
    """scipy.ndimage when installed (imported on first use: it costs ~0.3 s at import time), else None."""
    try:
        from scipy import ndimage
        return ndimage
    except Exception:  # pragma: no cover
        return None

__all__ = ["perlin2d_at", "perlin2d", "fbm2d_at", "fbm2d", "warp2d", "perlin1d_at", "fbm1d", "smooth1d"]

_U32 = np.uint32
_KX, _KY, _KS, _KM = _U32(374761393), _U32(668265263), _U32(1442695041), _U32(1274126177)


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _mix32(h):
    """uint32 hash scramble -> 0~255 index (numpy uint32 arithmetic wraps automatically)."""
    with np.errstate(over="ignore"):
        h = h ^ (h >> _U32(13))
        h = h * _KM
        h = h ^ (h >> _U32(16))
    return (h & _U32(255)).astype(np.intp)


def _hash2(ix, iy, seed):
    with np.errstate(over="ignore"):
        h = ix.astype(_U32) * _KX + iy.astype(_U32) * _KY + _U32(int(seed) & 0xFFFFFFFF) * _KS
    return _mix32(h)


def _hash1(ix, seed):
    with np.errstate(over="ignore"):
        h = ix.astype(_U32) * _KX + _U32(int(seed) & 0xFFFFFFFF) * _KS
    return _mix32(h)


_GRAD2 = {}
_GRAD1 = {}


def _grad2(seed):
    if seed not in _GRAD2:
        ang = np.random.default_rng(seed).uniform(0, 2 * np.pi, 256)
        _GRAD2[seed] = np.stack([np.cos(ang), np.sin(ang)], -1).astype(F32)
    return _GRAD2[seed]


def _grad1(seed):
    if seed not in _GRAD1:
        _GRAD1[seed] = np.random.default_rng(seed).uniform(-1, 1, 256).astype(F32)
    return _GRAD1[seed]


def perlin2d_at(X, Y, seed=0):
    """Sample 2D Perlin gradient noise at arbitrary coordinates (in cells); returns roughly -1~1, same shape as X."""
    X = np.asarray(X, F32)
    Y = np.asarray(Y, F32)
    xi = np.floor(X).astype(np.int64)
    yi = np.floor(Y).astype(np.int64)
    xf = X - xi
    yf = Y - yi
    tbl = _grad2(seed)

    def dot(dx, dy, ox, oy):
        g = tbl[_hash2(xi + ox, yi + oy, seed)]
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = dot(xf, yf, 0, 0)
    n10 = dot(xf - 1, yf, 1, 0)
    n01 = dot(xf, yf - 1, 0, 1)
    n11 = dot(xf - 1, yf - 1, 1, 1)
    u = _fade(xf)
    v = _fade(yf)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return ((nx0 + v * (nx1 - nx0)) * 1.4142).astype(F32)


def _perlin_grid(x, y, seed=0):
    """Separable fast path: x (w,) and y (h,) are 1D coordinates in cells; returns (h, w)."""
    x = np.asarray(x, F32)
    y = np.asarray(y, F32)
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    xf = (x - xi).astype(F32)
    yf = (y - yi).astype(F32)
    u = _fade(xf)[None, :]
    v = _fade(yf)[:, None]
    tbl = _grad2(seed)
    gx, gy = np.ascontiguousarray(tbl[:, 0]), np.ascontiguousarray(tbl[:, 1])
    with np.errstate(over="ignore"):
        hx0 = xi.astype(_U32) * _KX
        hx1 = (xi + 1).astype(_U32) * _KX
        hy0 = yi.astype(_U32) * _KY + _U32(int(seed) & 0xFFFFFFFF) * _KS
        hy1 = (yi + 1).astype(_U32) * _KY + _U32(int(seed) & 0xFFFFFFFF) * _KS
        i00 = _mix32(hy0[:, None] + hx0[None, :])
        i10 = _mix32(hy0[:, None] + hx1[None, :])
        i01 = _mix32(hy1[:, None] + hx0[None, :])
        i11 = _mix32(hy1[:, None] + hx1[None, :])
    X0, X1 = xf[None, :], (xf - 1)[None, :]
    Y0, Y1 = yf[:, None], (yf - 1)[:, None]
    n00 = gx[i00] * X0 + gy[i00] * Y0
    n10 = gx[i10] * X1 + gy[i10] * Y0
    n01 = gx[i01] * X0 + gy[i01] * Y1
    n11 = gx[i11] * X1 + gy[i11] * Y1
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return ((nx0 + v * (nx1 - nx0)) * 1.4142).astype(F32)


def perlin2d(h, w, scale, seed=0, offset=(0.0, 0.0), aspect=1.0):
    """Full (h, w) Perlin noise. ``scale`` = pixel width of one cell; ``aspect`` > 1 stretches features along y, < 1 stretches them horizontally."""
    x = (np.arange(w, dtype=F32) + offset[0]) / scale
    y = (np.arange(h, dtype=F32) + offset[1]) / (scale * aspect)
    return _perlin_grid(x, y, seed)


MODES = ("plain", "ridged", "billow")


def _accumulate(sampler, octaves, lacunarity, gain, seed, mode):
    if mode not in MODES:
        from ._args import check_choice
        check_choice("fbm mode", "mode", mode, MODES)
    total = None
    amp, norm, freq, weight = 1.0, 0.0, 1.0, 1.0
    for i in range(octaves):
        n = sampler(freq, seed + 101 * i)
        if mode == "ridged":
            n = 1.0 - np.abs(n)
            n = n * n * weight
            weight = np.clip(n * 2.0, 0.0, 1.0)
        elif mode == "billow":
            n = np.abs(n)
        total = amp * n if total is None else total + amp * n
        norm += amp
        amp *= gain
        freq *= lacunarity
    return (total / norm).astype(F32)


def fbm2d_at(X, Y, octaves=6, lacunarity=2.0, gain=0.5, seed=0, mode="plain"):
    """Fractal noise (octaves summed), coordinates in cells.

    mode:
      'plain'  -> about -1~1, general terrain / clouds
      'ridged' -> 0~1, sharp ridges (ridged multifractal)
      'billow' -> 0~1, puffy cloud-like
    """
    X = np.asarray(X, F32)
    Y = np.asarray(Y, F32)
    return _accumulate(lambda f, s: perlin2d_at(X * f, Y * f, s), octaves, lacunarity, gain, seed, mode)


def fbm2d(h, w, scale, octaves=6, lacunarity=2.0, gain=0.5, seed=0, offset=(0.0, 0.0), aspect=1.0, mode="plain"):
    """Full (h, w) fBm noise (fast separable path), same parameters as fbm2d_at; ``scale`` is the lowest-frequency cell width in pixels."""
    x = (np.arange(w, dtype=F32) + offset[0]) / scale
    y = (np.arange(h, dtype=F32) + offset[1]) / (scale * aspect)
    return _accumulate(lambda f, s: _perlin_grid(x * f, y * f, s), octaves, lacunarity, gain, seed, mode)


def warp2d(h, w, scale, strength=1.0, octaves=5, seed=0, aspect=1.0, mode="plain", offset=(0.0, 0.0)):
    """Domain warping: distort the coordinates with noise, then sample noise -> flowing / mist / marble-like patterns."""
    qx = fbm2d(h, w, scale, 4, seed=seed + 7, offset=offset, aspect=aspect)
    qy = fbm2d(h, w, scale, 4, seed=seed + 13, offset=(offset[0] + 5.2 * scale, offset[1] + 1.3 * scale), aspect=aspect)
    X, Y = grid(h, w)
    X = (X + offset[0]) / scale
    Y = (Y + offset[1]) / (scale * aspect)
    return fbm2d_at(X + strength * qx, Y + strength * qy, octaves, seed=seed, mode=mode)


def perlin1d_at(X, seed=0):
    """1D Perlin, coordinates in cells; returns roughly -1~1."""
    X = np.asarray(X, F32)
    xi = np.floor(X).astype(np.int64)
    xf = X - xi
    tbl = _grad1(seed)
    g0 = tbl[_hash1(xi, seed)]
    g1 = tbl[_hash1(xi + 1, seed)]
    n0 = g0 * xf
    n1 = g1 * (xf - 1)
    u = _fade(xf)
    return ((n0 + u * (n1 - n0)) * 2.0).astype(F32)


def fbm1d(n, scale, octaves=5, lacunarity=2.0, gain=0.5, seed=0, offset=0.0, mode="plain"):
    """1D fBm of length n (e.g. mountain ridge, river bank). mode as in fbm2d_at."""
    X = (np.arange(n, dtype=F32) + offset) / scale
    return _accumulate(lambda f, s: perlin1d_at(X * f, s), octaves, lacunarity, gain, seed, mode)


def smooth1d(a, sigma):
    """1D Gaussian smoothing (edges replicated)."""
    a = np.asarray(a, F32)
    if sigma <= 0:
        return a.copy()
    ndi = _scipy_ndimage()
    if ndi is not None:
        return ndi.gaussian_filter1d(a, sigma, mode="nearest").astype(F32)
    r = max(int(4.0 * sigma + 0.5), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(a, r, mode="edge"), k, mode="valid").astype(F32)


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "perlin": "use noise.perlin2d(h, w, scale, seed=0) for a field, or noise.perlin2d_at(X, Y, seed) at coordinates",
    "perlin3d": "use noise.perlin2d(h, w, scale, seed=0); there is no 3D noise, change seed or offset for variants",
    "simplex": "use noise.perlin2d(h, w, scale, seed=0) or noise.fbm2d(h, w, scale, octaves=6, seed=0)",
    "fbm": "use noise.fbm2d(h, w, scale, octaves=6, seed=0) for a field, or noise.fbm1d(n, scale) for a line",
    "noise": "use noise.fbm2d(h, w, scale, octaves=6, seed=0) (0..1 field), or noise.perlin2d(h, w, scale)",
    "noise2d": "use noise.fbm2d(h, w, scale, octaves=6, seed=0)",
    "value": "use noise.perlin2d(h, w, scale, seed=0)",
    "turbulence": "use noise.fbm2d(h, w, scale, mode='billow') (modes: plain, ridged, billow)",
    "ridged": "use noise.fbm2d(h, w, scale, mode='ridged')",
    "billow": "use noise.fbm2d(h, w, scale, mode='billow')",
    "warp": "use noise.warp2d(h, w, scale, strength=1.0) for a displacement field",
    "domainwarp": "use noise.warp2d(h, w, scale, strength=1.0)",
    "worley": "use pixelcanvas.elements.worley / voronoi for cellular patterns",
    "cellular": "use pixelcanvas.elements.worley / voronoi for cellular patterns",
    "smooth": "use noise.smooth1d(a, sigma) for a 1D array; for images use pixelcanvas.filters.gaussian_blur(a, sigma)",
    "blur": "use pixelcanvas.filters.gaussian_blur(a, sigma)",
    "random": "use np.random.default_rng(seed); noise.fbm2d(h, w, scale, seed=seed) for smooth noise",
}
__getattr__ = _module_getattr(__name__, _HINTS)
