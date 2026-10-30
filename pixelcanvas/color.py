# -*- coding: utf-8 -*-
"""顏色與基本數學工具（所有模組共用）。

慣例：顏色是 ``'#rrggbb'`` / ``(r, g, b)``（0~1，或整數 0~255）/ 整張 ``(H, W, 3)`` 顏色圖；
值允許 > 1（HDR），最後靠 ``soft_clip`` / ``filmic`` 壓回 0~1。
"""
from __future__ import annotations

import numpy as np

F32 = np.float32

__all__ = ["F32", "rgb", "as_color", "lerp", "mix", "hsv_to_rgb", "luma", "clamp01", "smoothstep", "normalize", "grid"]


def rgb(c):
    """把顏色轉成 float32 ndarray(3)。

    接受 ``'#rrggbb'`` / ``'#rgb'`` / ``(r, g, b)``。
    若三個分量都是整數且有值 > 1（例如 ``(255, 128, 0)``），視為 0~255；
    否則視為 0~1（允許 > 1 的 HDR 值，例如 ``(1.6, 1.4, 1.2)``）。
    已是 ndarray 的輸入原樣回傳（方便傳整張顏色圖）。
    """
    if isinstance(c, np.ndarray):
        return c.astype(F32, copy=False)
    if isinstance(c, str):
        s = c.lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], F32)
    a = np.asarray(c, dtype=np.float64)
    if a.ndim == 1 and a.max() > 1.0 and np.all(a == np.round(a)) and a.max() <= 255:
        a = a / 255.0
    return a.astype(F32)


def as_color(c):
    """接受單色 / 整張 (H, W, 3) 顏色圖 / Canvas，統一成 ndarray。"""
    if hasattr(c, "buf"):
        return np.asarray(c.buf, F32)
    if isinstance(c, np.ndarray):
        return c.astype(F32, copy=False)
    return rgb(c)


def _operand(x):
    if isinstance(x, (str, tuple, list)) or hasattr(x, "buf"):
        return as_color(x)
    return np.asarray(x, F32)


def lerp(a, b, t):
    """線性內插 a + (b - a) * t。a, b 可為單色 / 顏色圖 / 任意陣列；t 可為標量或 (H, W) 遮罩。"""
    a = _operand(a)
    b = _operand(b)
    t = np.asarray(t, F32)
    if t.ndim == 2 and (a.ndim != 2 or b.ndim != 2):
        t = t[..., None]
    return (a + (b - a) * t).astype(F32)


def mix(stops, t):
    """多段漸層。``stops = [(位置, 顏色), ...]`` 位置遞增（通常 0~1）；``t`` 可為標量或陣列。

    回傳形狀 ``t.shape + (3,)`` 的顏色。例：``mix([(0, '#000'), (1, '#fff')], V)`` 得到整張垂直漸層。
    """
    pos = np.array([p for p, _ in stops], F32)
    cols = np.stack([rgb(c) for _, c in stops]).astype(F32)
    t = np.clip(np.asarray(t, F32), pos[0], pos[-1])
    out = np.empty(np.shape(t) + (3,), F32)
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, cols[:, ch])
    return out


def hsv_to_rgb(h, s, v):
    """HSV → RGB（全部 0~1，可為陣列）。"""
    h = np.asarray(h, F32) % 1.0
    s = np.asarray(s, F32)
    v = np.asarray(v, F32)
    h, s, v = np.broadcast_arrays(h, s, v)
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1).astype(F32)


def luma(img):
    """亮度 (H, W)。"""
    img = np.asarray(img, F32)
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def clamp01(a):
    return np.clip(np.asarray(a, F32), 0.0, 1.0)


def smoothstep(e0, e1, x):
    """平滑階梯：x<=e0 → 0，x>=e1 → 1，中間為 S 曲線。e0/e1 可為陣列；e0 > e1 也可以（反向）。"""
    x = np.asarray(x, F32)
    e0 = np.asarray(e0, F32)
    e1 = np.asarray(e1, F32)
    d = e1 - e0
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    t = np.clip((x - e0) / d, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(F32)


def normalize(a, lo=0.0, hi=1.0):
    """把陣列線性映射到 lo~hi。"""
    a = np.asarray(a, F32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12:
        return np.full_like(a, lo)
    return ((a - mn) / (mx - mn) * (hi - lo) + lo).astype(F32)


def grid(h, w):
    """回傳像素座標網格 (X, Y)，皆為 float32 (h, w)。"""
    Y, X = np.mgrid[0:h, 0:w]
    return X.astype(F32), Y.astype(F32)
