# -*- coding: utf-8 -*-
"""Canvas：float32 (H, W, 3) 畫布，所有方法回傳 self（可串接），直接改 ``cv.buf`` 也可以。"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .color import F32, rgb, as_color, mix, clamp01, grid
from . import filters as _f
from . import vision as _vision

_RES = getattr(Image, "Resampling", Image)

__all__ = ["Canvas"]


class Canvas:
    """float32 (H, W, 3) 畫布。座標 x 向右、y 向下；``cv.buf[y, x]`` 就是像素。"""

    def __init__(self, width, height, color=(0.0, 0.0, 0.0)):
        self.buf = np.empty((int(height), int(width), 3), F32)
        self.buf[:] = rgb(color)
        _vision._track(self)

    # ---- 建立 / 輸出 ----
    @classmethod
    def load(cls, path):
        """從檔案載入成畫布（轉為 RGB、0~1）。"""
        im = Image.open(path).convert("RGB")
        cv = cls(1, 1)
        cv.buf = np.asarray(im, F32) / 255.0
        return cv

    @classmethod
    def from_array(cls, arr):
        arr = np.asarray(arr, F32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, -1)
        cv = cls(1, 1)
        cv.buf = arr.copy()
        return cv

    @property
    def w(self):
        return self.buf.shape[1]

    @property
    def h(self):
        return self.buf.shape[0]

    @property
    def shape(self):
        return self.buf.shape

    def copy(self):
        return Canvas.from_array(self.buf)

    def to_pil(self):
        return Image.fromarray((clamp01(self.buf) * 255 + 0.5).astype(np.uint8), "RGB")

    def save(self, path, preview=None, preview_width=960, quality=92):
        """存檔（副檔名決定格式）。``preview`` 另存一張縮小的預覽圖，方便快速檢視。"""
        im = self.to_pil()
        if str(path).lower().endswith((".jpg", ".jpeg")):
            im.save(path, quality=quality)
        else:
            im.save(path)
        if preview:
            pw = int(preview_width)
            ph = max(1, round(self.h * pw / self.w))
            im.resize((pw, ph), _RES.LANCZOS).save(preview, quality=88)
        return self

    def stats(self):
        """印出尺寸與亮度統計（>1 比例高 = 可能過曝）。"""
        b = self.buf
        print(f"{self.w}x{self.h}  min={b.min():.3f} max={b.max():.3f} mean={b.mean():.3f}  >1 的像素比例={np.mean(b.max(-1) > 1):.4f}")
        return self

    # ---- 座標 ----
    def xy(self):
        """像素座標網格 (X, Y)。"""
        return grid(self.h, self.w)

    def uv(self):
        """正規化座標 (U, V)，0~1。"""
        X, Y = self.xy()
        return X / self.w, Y / self.h

    # ---- 像素級讀寫 ----
    def get(self, x, y):
        """讀像素（x, y 可為整數或整數陣列）。"""
        return self.buf[y, x].copy()

    def put(self, x, y, color):
        """寫像素。x, y 可為整數陣列（一次寫很多點），color 可為單色或每點一色 (N, 3)。"""
        self.buf[y, x] = as_color(color)
        return self

    def region(self, x0, y0, x1, y1):
        """回傳子區域的 numpy view（直接修改即生效）。"""
        return self.buf[y0:y1, x0:x1]

    # ---- 上色 ----
    def paint(self, color, mask=None, mode="normal", opacity=1.0):
        """用 color（單色或整張 (H, W, 3) 顏色圖）依 mask 透明度、以 mode 混合塗上畫布。"""
        a = float(opacity) if mask is None else np.asarray(mask, F32) * float(opacity)
        self.buf = _f.blend(self.buf, color, a, mode)
        return self

    def fill(self, color, mask=None, mode="normal", opacity=1.0):
        return self.paint(color, mask, mode, opacity)

    def gradient(self, stops, axis="y", mask=None, mode="normal", opacity=1.0):
        """線性漸層：``stops = [(0~1 位置, 顏色), ...]``，axis = 'y'（上→下）或 'x'（左→右）。"""
        U, V = self.uv()
        return self.paint(mix(stops, V if axis == "y" else U), mask, mode, opacity)

    def linear(self, p0, p1, stops, mask=None, mode="normal", opacity=1.0):
        """Linear gradient in any direction: position 0 at point ``p0`` (x, y) → 1 at ``p1``."""
        X, Y = self.xy()
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        t = ((X - p0[0]) * dx + (Y - p0[1]) * dy) / max(dx * dx + dy * dy, 1e-9)
        return self.paint(mix(stops, t), mask, mode, opacity)

    def text(self, text, xy=None, size=None, color="#ffffff", **kw):
        """Draw styled text (see ``pixelcanvas.text.draw_text``). Returns self; the bbox is in ``cv.last_bbox``."""
        from .text import draw_text
        self.last_bbox = draw_text(self, text, xy, size, color, **kw)
        return self

    def radial(self, cx, cy, r, stops, mask=None, mode="normal", opacity=1.0):
        """放射狀漸層：stops 位置 0（中心）~1（半徑 r 處）。"""
        X, Y = self.xy()
        t = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / float(r)
        return self.paint(mix(stops, t), mask, mode, opacity)

    def glow(self, cx, cy, sigma, color, strength=1.0, mode="add", mask=None):
        """高斯光暈（預設相加 → 可產生 HDR 亮度）。"""
        X, Y = self.xy()
        g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma * sigma)).astype(F32) * strength
        if mask is not None:
            g = g * np.asarray(mask, F32)
        return self.paint(color, g, mode)

    def paste(self, other, x=0, y=0, mask=None, mode="normal", opacity=1.0):
        """把另一張畫布 / 陣列貼到 (x, y)（會自動裁切），mask 為與 other 同大小的遮罩。"""
        src = other.buf if isinstance(other, Canvas) else np.asarray(other, F32)
        sh, sw = src.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + sw, self.w), min(y + sh, self.h)
        if x0 >= x1 or y0 >= y1:
            return self
        sub = src[y0 - y:y1 - y, x0 - x:x1 - x]
        a = float(opacity) if mask is None else np.asarray(mask, F32)[y0 - y:y1 - y, x0 - x:x1 - x] * float(opacity)
        self.buf[y0:y1, x0:x1] = _f.blend(self.buf[y0:y1, x0:x1], sub, a, mode)
        return self

    # ---- 取樣 / 幾何 ----
    def sample(self, X, Y):
        """在浮點座標取樣目前畫面（倒影、扭曲用）。"""
        return _f.sample(self.buf, X, Y)

    def displace(self, dx, dy):
        """依位移場扭曲畫面。"""
        X, Y = self.xy()
        self.buf = _f.sample(self.buf, X + dx, Y + dy)
        return self

    def resize(self, w, h, method="lanczos"):
        self.buf = _f.resize(self.buf, w, h, method)
        return self

    def crop(self, x0, y0, x1, y1):
        self.buf = self.buf[y0:y1, x0:x1].copy()
        return self

    def flip(self, axis="y"):
        self.buf = self.buf[::-1].copy() if axis == "y" else self.buf[:, ::-1].copy()
        return self

    # ---- 後製（都是 apply 的包裝）----
    def apply(self, fn, *args, **kw):
        """``cv.apply(fn, ...)`` → ``cv.buf = fn(cv.buf, ...)``。"""
        self.buf = np.asarray(fn(self.buf, *args, **kw), F32)
        return self

    def blur(self, sigma):
        return self.apply(_f.gaussian_blur, sigma)

    def bloom(self, threshold=0.75, sigma=30, strength=0.4):
        return self.apply(_f.bloom, threshold, sigma, strength)

    def vignette(self, strength=0.35, radius=0.55, softness=0.6):
        return self.apply(_f.vignette, strength, radius, softness)

    def grain(self, amount=0.02, seed=0, mono=True):
        return self.apply(_f.grain, amount, seed, mono)

    def levels(self, black=0.0, white=1.0, gamma=1.0):
        return self.apply(_f.levels, black, white, gamma)

    def contrast(self, amount=0.2):
        return self.apply(_f.contrast, amount)

    def saturation(self, s=1.0):
        return self.apply(_f.saturation, s)

    def soft_clip(self, knee=0.8):
        return self.apply(_f.soft_clip, knee)

    def filmic(self):
        return self.apply(_f.filmic)

    def split_tone(self, shadows="#405070", highlights="#f0c080", amount=0.1):
        return self.apply(_f.split_tone, shadows, highlights, amount)

    def clip(self):
        self.buf = clamp01(self.buf)
        return self
