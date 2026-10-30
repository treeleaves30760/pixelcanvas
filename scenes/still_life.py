# -*- coding: utf-8 -*-
"""
靜物：桌上的紅球（still_life）— 由「法線」算光照的最小範例；任何曲面物件都可以照這個寫法。

構圖：桌緣在 V=0.62；球在畫面中央偏上，下緣壓在桌面上。
光源：左上前方（光向量 L 指向光源）→ 高光在左上、影子拉向右後方。
技巧：
* 球的法線：dx, dy = (像素 - 圓心) / 半徑，dz = sqrt(1 - dx² - dy²)。
* 光照 = 環境光 + 漫射 max(n·L, 0) + 鏡面 (r·v)^k + 桌面反光（朝下的面微亮）。
* 落影是貼著接觸點、往背光側拉長的扁橢圓；接觸處再疊一道窄的環境遮蔽（AO）才不會像浮在空中。
"""
import numpy as np

from pixelcanvas import *

NAME = "still_life"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 2


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()
    L = np.array([-0.55, -0.6, 0.58])
    L /= np.linalg.norm(L)

    log("wall & table")
    cv.paint(mix([(0, "#3b3a44"), (1, "#5a5560")], U * 0.3 + V * 0.7))
    grain_wood = 1 + 0.06 * fbm2d(H, W, 0.4 * W, 4, seed=seed, aspect=0.05)[..., None]   # 拉長的木紋
    cv.paint(mix([(0.62, "#6e5a48"), (1, "#3a2c22")], V) * grain_wood, smoothstep(0.618 * H - 1, 0.618 * H + 1, Y))

    log("sphere")
    cx, cy, r = 0.5 * W, 0.47 * H, 0.2 * H
    dx, dy = (X - cx) / r, (Y - cy) / r
    rr = dx * dx + dy * dy
    dz = np.sqrt(np.clip(1 - rr, 0, 1))
    ndl = np.clip(dx * L[0] + dy * L[1] + dz * L[2], 0, 1)
    spec = np.clip(2 * ndl * dz - L[2], 0, 1) ** 40                # 視線 v = (0, 0, 1) 的反射項
    bounce = np.clip(dy, 0, 1) * 0.25
    col = rgb("#c0392b") * (0.12 + 0.95 * ndl[..., None]) + rgb("#d9a07a") * bounce[..., None] + spec[..., None] * 1.4
    sphere = 1 - smoothstep(1 - 2 / r, 1 + 1 / r, np.sqrt(rr))     # 約 1px 抗鋸齒邊

    sx, sy = cx - L[0] / L[2] * r * 0.45, cy + r * 0.92             # 落影：從接觸點往背光側拉長
    e = ((X - sx) / (r * 1.3)) ** 2 + ((Y - sy) / (r * 0.22)) ** 2
    cv.paint("#120c0a", feather(clamp01(1.2 - e), r * 0.08) * 0.75)
    ao = np.exp(-(((X - cx) / (r * 0.9)) ** 2 + ((Y - (cy + r)) / (r * 0.12)) ** 2))
    cv.paint("#000000", ao * 0.6)
    cv.paint(col, sphere)

    log("post")
    cv.vignette(0.35).soft_clip(0.85).grain(0.01, seed=seed)
    return cv
