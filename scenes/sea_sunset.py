# -*- coding: utf-8 -*-
"""
海上日落（sea_sunset）— 透視水面：世界座標反推、鏡射倒影、菲涅耳、波面閃光。

構圖：海平線 V=0.55（略低於中線），太陽在正中、剛好在海平線上方；光帶把視線拉向前景。
光源：低角度太陽 → 雲底暖色、海面只有朝向太陽的波面會亮。
技巧：
* 透視：每個海面像素反推深度 z ∝ 1 / (Y - 海平線)、橫向 wx = (X - 中心) · z，
  在 (wx, z) 上取噪聲 → 遠處波浪自然變細變密。遠處要淡出，否則會出現摩爾紋。
* 倒影 = 以海平線鏡射「目前畫面」+ 波紋位移；菲涅耳：遠處接近全反射、近處透出深水色。
* 閃光只在太陽正下方的直帶內、且只在波面斜率朝向太陽處（-slope 大）出現；越近帶子越寬。
"""
import numpy as np

from pixelcanvas import *

NAME = "sea_sunset"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 9


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()
    hz = 0.55 * H
    sunx, suny = 0.5 * W, 0.45 * H

    log("sky, sun, clouds")
    cv.gradient([(0, "#2b4c7e"), (0.35, "#8a6f8e"), (0.5, "#e8a07a"), (0.55, "#ffc58a"), (1, "#ffc58a")])
    D = np.hypot(X - sunx, Y - suny)
    cv.paint("#ff9a55", np.exp(-(D / (0.35 * H)) ** 2) * 0.3, "add")
    cv.paint("#fff0d0", np.exp(-(D / (0.06 * H)) ** 2) * 0.4, "add")
    cv.paint((1.8, 1.6, 1.3), 1 - smoothstep(0.03 * H - 1, 0.03 * H + 1, D))
    c = fbm2d(H, W, 0.3 * W, 5, seed=seed - 4, aspect=0.15, mode="billow")
    cv.paint(lerp("#5a4a6a", "#ffb88a", smoothstep(0.1, 0.45, V)),
             smoothstep(0.35, 0.6, c) * (1 - smoothstep(0.3, 0.45, V)) * 0.8)

    log("sea")
    dy = np.maximum(Y - hz, 0.5)
    z = 0.45 * H / dy                                              # 畫面底部 z ≈ 1，越近海平線越大
    wx = (X - W / 2) / H * z
    far = smoothstep(1.0, 12.0, z)
    waves = fbm2d_at(wx * 5, z * 22, 5, seed=seed) * (1 - far)
    slope = np.gradient(waves, axis=0) * dy / (0.02 * H)           # 波面 y 方向斜率（近處放大）
    refl = cv.sample(X + waves * 0.03 * H / z, 2 * hz - Y + slope * 0.02 * H)
    t = 1 / z                                                      # 0 遠 → 1 近
    fres = 0.35 + 0.65 * (1 - t) ** 2
    water = lerp(lerp("#0b1c2e", "#1f3d55", waves * 0.5 + 0.5), refl * 0.9, fres)
    path = np.exp(-((X - sunx) / (0.02 * W + 0.12 * W * t)) ** 2)
    spark = smoothstep(0.15, 0.6, -slope) * path
    water = water + rgb((1.4, 1.05, 0.7)) * spark[..., None] * 0.9
    cv.paint(water, smoothstep(hz - 0.5, hz + 0.5, Y))

    log("post")
    cv.bloom(0.85, 0.015 * H, 0.3).soft_clip(0.85).contrast(0.08).vignette(0.3).grain(0.01, seed=seed)
    return cv
