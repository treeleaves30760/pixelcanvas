# -*- coding: utf-8 -*-
"""
黃昏城市天際線（city_dusk）— 幾何物件 + 重複圖樣（窗戶）+ 分層大氣透視。

構圖：三層樓群由遠到近，地面線 0.80 / 0.86 / 0.95；落日躲在遠樓後方（左三分線）。
光源：逆光 → 樓身全是剪影，只靠窗燈與霧色分層。
技巧：
* 每層一張「樓頂高度剖面」top(W,)，遮罩 = clamp01(Y - top + 0.5)（和山稜線同一招）。
* 窗戶 = 以 cell 為格的整數座標 → 用 perlin2d_at 在格點上取值決定開燈（同一格同一個值）。
* 越遠：越矮、越窄、越接近霧色、窗燈越淡；最遠一層不畫窗。
"""
import numpy as np

from pixelcanvas import *

NAME = "city_dusk"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 4

LAYERS = [  # (地面 V, 最矮, 最高（佔 H 比例）, 最窄, 最寬（1080p px）, 霧量, 窗格 px)
    (0.80, 0.08, 0.22, 36, 80, 0.70, 0),
    (0.86, 0.10, 0.32, 48, 120, 0.38, 12),
    (0.95, 0.12, 0.45, 80, 180, 0.00, 18),
]


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()

    log("sky")
    cv.gradient([(0, "#1a1440"), (0.45, "#6b3a6e"), (0.7, "#e0785a"), (0.8, "#f2b073"), (1, "#f2b073")])
    cv.glow(0.3 * W, 0.78 * H, 0.25 * H, "#ffb070", strength=0.5)
    haze = rgb("#d88a6a")
    rng = np.random.default_rng(seed)

    for k, (ground, hmin, hmax, wmin, wmax, fog, cell) in enumerate(LAYERS):
        log(f"buildings {k}")
        top = np.full(W, H, F32)
        win = np.zeros((H, W), F32)
        x = -rng.uniform(0, wmax * S)
        while x < W:
            bw = rng.uniform(wmin, wmax) * S
            bh = rng.uniform(hmin, hmax) ** 1.3 / hmax ** 0.3 * H     # 偏向矮樓，少數高樓
            x0, x1 = int(max(x, 0)), int(min(x + bw, W))
            top[x0:x1] = ground * H - bh
            if cell and x1 > x0:
                c = cell * S
                lx, yy = X[:, x0:x1] - x, Y[:, x0:x1]
                lit = perlin2d_at((lx // c) * 0.71 + x, (yy // c) * 0.37, seed=seed + k) > 0.25
                pane = (lx % c > c * 0.4) & (yy % c > c * 0.45) & (yy > top[x0] + c)
                win[:, x0:x1] = np.maximum(win[:, x0:x1], lit & pane)
            x += bw + rng.uniform(0, 8) * S
        mask = clamp01(Y - top[None, :] + 0.5)
        cv.paint(lerp(lerp("#141020", "#2a1d30", V), haze, fog), mask)
        cv.paint(lerp("#ffcf7a", haze, fog), win * mask * (1 - fog) * 0.9)

    log("post")
    cv.bloom(0.7, 0.01 * H, 0.5).soft_clip(0.85).vignette(0.3).grain(0.012, seed=seed)
    return cv
