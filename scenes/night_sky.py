# -*- coding: utf-8 -*-
"""
星空夜景（night_sky）— 只用低階 API：漸層、domain warp、散點、光暈、1D 稜線。

構圖：眼平線很低（山丘在 V≈0.76~0.9），天空佔 3/4；月亮在右上三分點，銀河由左上斜向右下。
光源：月光很弱 → 剪影幾乎全黑，只靠天光把遠山和近山分開。
技巧：
* 銀河 = 斜帶高斯（到中心線的距離）× warp 噪聲；再用另一個噪聲疊暗塵帶。
* 星星 = 隨機座標 + 冪次分布亮度（大多數暗、少數很亮）→ np.add.at 撒到遮罩 → 微模糊後相加。
* 月面不要 > 1，否則斑紋會在 soft_clip 後消失。
"""
import numpy as np

from pixelcanvas import *

NAME = "night_sky"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080                                                   # 像素常數以 1080p 為基準
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()

    log("sky & milky way")
    cv.gradient([(0, "#02040c"), (0.55, "#0b1838"), (0.78, "#23385f"), (1, "#3a4a6a")])
    d = (V - 0.15 - 0.55 * U) / 0.16                               # 到銀河中心線的距離
    band = np.exp(-d * d)
    dust = warp2d(H, W, 0.12 * H, 0.8, seed=seed + 2) * 0.5 + 0.5
    cv.paint("#8ea6d8", band * smoothstep(0.35, 0.8, dust) * 0.45, "add")
    cv.paint("#05070f", band * smoothstep(0.55, 0.75, fbm2d(H, W, 0.05 * H, 5, seed=seed + 3)) * 0.6)

    log("stars")
    rng = np.random.default_rng(seed)
    n = int(1620 * W / H)                                          # 星星數量只看畫面比例：草稿與全尺寸同一片星空
    sx, sy = rng.uniform(0, W, n), rng.uniform(0, H * 0.8, n)
    b = rng.random(n) ** 6 * 1.6 + 0.15                            # 冪次分布：大多數暗、少數亮
    b *= (1 - sy / (H * 0.8)) ** 0.7 * (1 + 1.5 * np.exp(-((sy / H - 0.15 - 0.55 * sx / W) / 0.16) ** 2))
    stars = np.zeros((H, W), F32)
    np.add.at(stars, (sy.astype(int), sx.astype(int)), b)
    cv.paint("#ffffff", gaussian_blur(stars, 1.2 * S) * 12 * S * S, "add")   # 模糊半徑 ∝ S → 亮度補 S²
    for i in rng.choice(n, 14, replace=False):                     # 亮星隨機挑，避免全擠在銀河裡
        cv.glow(sx[i], sy[i], 6 * S, "#cfe0ff", strength=0.5)

    log("moon")
    mx, my, mr = 0.78 * W, 0.22 * H, 0.045 * H
    moon = circle_mask(H, W, mx, my, mr, feather=S)
    maria = fbm2d(H, W, mr * 0.9, 4, seed=seed + 6)
    cv.glow(mx, my, mr * 4, "#5f78b0", strength=0.35)
    cv.paint(lerp("#f2efe4", "#a29f95", smoothstep(-0.1, 0.35, maria)), moon)

    log("hills")
    for base, amp, col, s in [(0.76, 0.10, "#141d33", 10), (0.87, 0.07, "#04060c", 11)]:
        ridge = (base - amp * fbm1d(W, 0.35 * W, seed=seed + s)) * H
        cv.paint(col, clamp01(Y - ridge[None, :] + 0.5))          # 1px 抗鋸齒剪影

    log("post")
    cv.bloom(0.8, 0.02 * H, 0.3).vignette(0.3).soft_clip(0.85).grain(0.012, seed=seed)
    return cv
