# -*- coding: utf-8 -*-
"""混合模式與濾鏡 / 後製（全部是「陣列進、陣列出」的純函數，Canvas 上有同名包裝）。"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .color import F32, as_color, rgb, luma, clamp01, smoothstep, grid

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

_RES = getattr(Image, "Resampling", Image)

__all__ = [
    "blend", "gaussian_blur", "sample", "resize", "radial_blur", "light_shafts", "bloom", "vignette",
    "grain", "levels", "contrast", "saturation", "soft_clip", "filmic", "split_tone",
]


def blend(base, top, alpha=1.0, mode="normal"):
    """把 top 以 alpha（標量或 (H, W) 遮罩）用指定模式混到 base 上，回傳新陣列（不裁切到 0~1）。

    mode: normal / add / multiply / screen / overlay / softlight / lighten / darken
    """
    base = np.asarray(base, F32)
    top = as_color(top)
    if top.ndim == 1:
        top = np.broadcast_to(top, base.shape)
    a = np.asarray(alpha, F32)
    if a.ndim == 2:
        a = a[..., None]
    if mode == "normal":
        res = top
    elif mode == "add":
        res = base + top
    elif mode == "multiply":
        res = base * top
    elif mode == "screen":
        res = 1.0 - (1.0 - base) * (1.0 - top)
    elif mode == "overlay":
        res = np.where(base < 0.5, 2.0 * base * top, 1.0 - 2.0 * (1.0 - base) * (1.0 - top))
    elif mode == "softlight":
        res = (1.0 - 2.0 * top) * base * base + 2.0 * top * base
    elif mode == "lighten":
        res = np.maximum(base, top)
    elif mode == "darken":
        res = np.minimum(base, top)
    else:
        raise ValueError(f"未知的混合模式: {mode}")
    return (base + (res - base) * a).astype(F32)


def _box1d(a, r, axis):
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(F32)


def gaussian_blur(a, sigma):
    """高斯模糊，支援 (H, W) 或 (H, W, C)。沒有 scipy 時用三次 box blur 近似。"""
    a = np.asarray(a, F32)
    if sigma <= 0:
        return a.copy()
    if _ndi is not None:
        if a.ndim == 3:
            return np.stack([_ndi.gaussian_filter(a[..., c], sigma, mode="nearest") for c in range(a.shape[2])], -1).astype(F32)
        return _ndi.gaussian_filter(a, sigma, mode="nearest").astype(F32)
    r = max(int(round((-1 + math.sqrt(1 + 4 * sigma * sigma)) / 2)), 1)
    out = a
    for _ in range(3):
        out = _box1d(_box1d(out, r, 0), r, 1)
    return out


def sample(img, X, Y, order=1):
    """雙線性取樣：在浮點像素座標 (X, Y) 取 img 的值（邊界取最近）。用於倒影、扭曲、位移、折射。"""
    img = np.asarray(img, F32)
    X = np.asarray(X, F32)
    Y = np.asarray(Y, F32)
    if _ndi is not None:
        coords = [Y, X]
        if img.ndim == 2:
            return _ndi.map_coordinates(img, coords, order=order, mode="nearest").astype(F32)
        return np.stack([_ndi.map_coordinates(img[..., c], coords, order=order, mode="nearest") for c in range(img.shape[2])], -1).astype(F32)
    h, w = img.shape[:2]
    xi = np.clip(np.rint(X).astype(int), 0, w - 1)
    yi = np.clip(np.rint(Y).astype(int), 0, h - 1)
    return img[yi, xi]


def resize(arr, w, h, method="bilinear"):
    """縮放 (H, W) 或 (H, W, C) 的 float 陣列。method: bilinear / bicubic / lanczos / box / nearest"""
    arr = np.asarray(arr, F32)
    m = {"bilinear": _RES.BILINEAR, "bicubic": _RES.BICUBIC, "lanczos": _RES.LANCZOS, "box": _RES.BOX, "nearest": _RES.NEAREST}[method]
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(np.ascontiguousarray(arr), "F").resize((w, h), m), F32)
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(arr[..., c]), "F").resize((w, h), m), F32) for c in range(arr.shape[2])], -1)


def radial_blur(img, center, strength=0.3, samples=24, decay=1.0, downscale=1):
    """放射狀（zoom）模糊：把影像朝 center 縮放多次取加權平均。``strength`` = 最大縮放量 (0~1)。

    ``downscale`` > 1 時先縮小計算再放大（快很多，適合柔光）。
    """
    img = np.asarray(img, F32)
    h, w = img.shape[:2]
    cx, cy = center
    if downscale > 1 and w // downscale >= 4 and h // downscale >= 4:
        small = resize(img, w // downscale, h // downscale, "box")
        out = radial_blur(small, (cx / downscale, cy / downscale), strength, samples, decay, 1)
        return resize(out, w, h, "bilinear")
    X, Y = grid(h, w)
    acc = np.zeros_like(img)
    wsum, wgt = 0.0, 1.0
    for i in range(samples):
        s = 1.0 - strength * i / max(samples - 1, 1)
        acc += wgt * sample(img, cx + (X - cx) * s, cy + (Y - cy) * s)
        wsum += wgt
        wgt *= decay
    return (acc / wsum).astype(F32)


def light_shafts(source_mask, center, samples=32, decay=0.95, length=0.9, downscale=2):
    """體積光 / 耶穌光：把亮源遮罩（例如太陽周圍「沒被山擋住」的天空）朝光源做放射模糊。

    回傳 (H, W) 的光線強度，之後可用 ``cv.paint(color, rays * strength, 'add')`` 疊上。
    """
    return radial_blur(np.asarray(source_mask, F32), center, strength=length, samples=samples, decay=decay, downscale=downscale)


def bloom(img, threshold=0.75, sigma=30, strength=0.4):
    """泛光：把高於 threshold 的亮部模糊後加回去。"""
    img = np.asarray(img, F32)
    l = luma(img)
    k = clamp01((l - threshold) / max(1.0 - threshold, 1e-6))[..., None]
    return (img + gaussian_blur(img * k, sigma) * strength).astype(F32)


def vignette(img, strength=0.35, radius=0.55, softness=0.6):
    """暗角：``radius`` 開始變暗的半徑（0 中心 ~ 1 角落），``softness`` 過渡寬度。"""
    img = np.asarray(img, F32)
    h, w = img.shape[:2]
    X, Y = grid(h, w)
    d = np.sqrt(((X / w - 0.5) * 2) ** 2 + ((Y / h - 0.5) * 2) ** 2) / 1.4142
    v = 1.0 - strength * smoothstep(radius - softness * 0.5, radius + softness * 0.5, d)
    return (img * v[..., None]).astype(F32)


def grain(img, amount=0.02, seed=0, mono=True):
    """底片顆粒。"""
    img = np.asarray(img, F32)
    rng = np.random.default_rng(seed)
    if mono:
        n = rng.normal(0, amount, img.shape[:2]).astype(F32)[..., None]
    else:
        n = rng.normal(0, amount, img.shape).astype(F32)
    return (img + n).astype(F32)


def levels(img, black=0.0, white=1.0, gamma=1.0):
    """色階：black→0、white→1，再套 gamma。"""
    x = clamp01((np.asarray(img, F32) - black) / max(white - black, 1e-6))
    return (x ** (1.0 / gamma)).astype(F32)


def contrast(img, amount=0.2):
    """S 曲線對比（amount 可為負 → 降低對比）。"""
    x = clamp01(img)
    return (x + amount * (x * x * (3.0 - 2.0 * x) - x)).astype(F32)


def saturation(img, s=1.0):
    """飽和度倍率。"""
    img = np.asarray(img, F32)
    l = luma(img)[..., None]
    return (l + (img - l) * s).astype(F32)


def soft_clip(img, knee=0.8):
    """柔性裁切：knee 以下不變，以上平滑壓縮到 1（處理 HDR > 1 的亮部）。"""
    x = np.asarray(img, F32)
    y = x.copy()
    over = x > knee
    y[over] = knee + (1.0 - knee) * (1.0 - np.exp(-(x[over] - knee) / (1.0 - knee)))
    return np.clip(y, 0.0, 1.0).astype(F32)


def filmic(img):
    """ACES 近似的 filmic 色調映射（會略微提升中間調、壓縮亮部）。"""
    x = np.maximum(np.asarray(img, F32), 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return clamp01((x * (a * x + b)) / (x * (c * x + d) + e))


def split_tone(img, shadows="#405070", highlights="#f0c080", amount=0.1):
    """分離色調：暗部往 shadows、亮部往 highlights 偏移（以 0.5 灰為中性）。"""
    img = np.asarray(img, F32)
    l = luma(clamp01(img))[..., None]
    off = (rgb(shadows) - 0.5) * (1 - l) + (rgb(highlights) - 0.5) * l
    return (img + off * amount * 2).astype(F32)
