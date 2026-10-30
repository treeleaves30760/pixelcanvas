# -*- coding: utf-8 -*-
"""噪聲：Perlin / fBm / ridged / billow / domain warp（2D 與 1D）。

這是地形、雲、水紋、材質的來源。所有函數都是確定性的（同 seed 同結果），
而且是「以格為單位」的連續場：換解析度只是重新取樣，圖案不變。
"""
from __future__ import annotations

import numpy as np

from .color import F32, grid

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

__all__ = ["perlin2d_at", "perlin2d", "fbm2d_at", "fbm2d", "warp2d", "perlin1d_at", "fbm1d", "smooth1d"]

_U32 = np.uint32
_KX, _KY, _KS, _KM = _U32(374761393), _U32(668265263), _U32(1442695041), _U32(1274126177)


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _mix32(h):
    """uint32 雜湊攪拌 → 0~255 索引（numpy uint32 運算自動 wrap）。"""
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
    """在任意座標（以「格」為單位）取樣 2D Perlin 梯度噪聲，回傳約 -1~1，形狀同 X。"""
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
    """可分離的快速路徑：x (w,)、y (h,) 皆為「格」單位的 1D 座標，回傳 (h, w)。"""
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
    """整張 (h, w) 的 Perlin 噪聲。``scale`` = 一格的像素寬；``aspect`` > 1 讓特徵在 y 方向拉長、< 1 則橫向拉長。"""
    x = (np.arange(w, dtype=F32) + offset[0]) / scale
    y = (np.arange(h, dtype=F32) + offset[1]) / (scale * aspect)
    return _perlin_grid(x, y, seed)


def _accumulate(sampler, octaves, lacunarity, gain, seed, mode):
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
    """分形噪聲（多八度疊加），座標以「格」為單位。

    mode:
      'plain'  → 約 -1~1，一般地形 / 雲
      'ridged' → 0~1，尖銳山脊（ridged multifractal）
      'billow' → 0~1，鼓起的雲朵狀
    """
    X = np.asarray(X, F32)
    Y = np.asarray(Y, F32)
    return _accumulate(lambda f, s: perlin2d_at(X * f, Y * f, s), octaves, lacunarity, gain, seed, mode)


def fbm2d(h, w, scale, octaves=6, lacunarity=2.0, gain=0.5, seed=0, offset=(0.0, 0.0), aspect=1.0, mode="plain"):
    """整張 (h, w) 的 fBm 噪聲（快速可分離路徑），參數同 fbm2d_at；``scale`` 為最低頻的格寬（像素）。"""
    x = (np.arange(w, dtype=F32) + offset[0]) / scale
    y = (np.arange(h, dtype=F32) + offset[1]) / (scale * aspect)
    return _accumulate(lambda f, s: _perlin_grid(x * f, y * f, s), octaves, lacunarity, gain, seed, mode)


def warp2d(h, w, scale, strength=1.0, octaves=5, seed=0, aspect=1.0, mode="plain", offset=(0.0, 0.0)):
    """Domain warping：先用噪聲扭曲座標再取噪聲 → 流動 / 雲霧 / 大理石狀圖案。"""
    qx = fbm2d(h, w, scale, 4, seed=seed + 7, offset=offset, aspect=aspect)
    qy = fbm2d(h, w, scale, 4, seed=seed + 13, offset=(offset[0] + 5.2 * scale, offset[1] + 1.3 * scale), aspect=aspect)
    X, Y = grid(h, w)
    X = (X + offset[0]) / scale
    Y = (Y + offset[1]) / (scale * aspect)
    return fbm2d_at(X + strength * qx, Y + strength * qy, octaves, seed=seed, mode=mode)


def perlin1d_at(X, seed=0):
    """1D Perlin，座標以「格」為單位，回傳約 -1~1。"""
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
    """長度 n 的 1D fBm（例如山稜線、河岸線）。mode 同 fbm2d_at。"""
    X = (np.arange(n, dtype=F32) + offset) / scale
    return _accumulate(lambda f, s: perlin1d_at(X * f, s), octaves, lacunarity, gain, seed, mode)


def smooth1d(a, sigma):
    """1D 高斯平滑（邊界複製）。"""
    a = np.asarray(a, F32)
    if sigma <= 0:
        return a.copy()
    if _ndi is not None:
        return _ndi.gaussian_filter1d(a, sigma, mode="nearest").astype(F32)
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(a, r, mode="edge"), k, mode="valid").astype(F32)
