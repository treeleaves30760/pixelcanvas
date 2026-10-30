# -*- coding: utf-8 -*-
"""
黃昏山川（mountains_sunset）— 用 scenery 配方組合的第一個範例場景。

渲染：python tools/render.py mountains_sunset            （全尺寸 1920x1080，約 25 秒）
      python tools/render.py mountains_sunset --scale 0.5 （草稿 960x540，約 6 秒）

場景檔的約定：提供 ``render(width, height, seed, log) -> Canvas``，以及 ``DEFAULT_SIZE``。
所有位置用比例（0~1）、所有像素單位以 1080p 為基準（scenery 內部會依解析度縮放）。
"""
import numpy as np

from pixelcanvas import F32, lerp, clamp01, smoothstep, fbm2d
from pixelcanvas.scenery import (Scene, sky, sun, cirrus, cumulus, mountain_layer, valley_floor, mist, haze_band,
                                 river_course, water_mask, water_surface, river_bank, pine_mask, rim_light,
                                 ground_edge, birds, god_rays, finish)

NAME = "mountains_sunset"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 2026

# ---- 調色盤 ----------------------------------------------------------------
SKY = [(0.00, "#0c1c3e"), (0.18, "#1e4576"), (0.33, "#3f6f9f"), (0.44, "#86a6c2"), (0.50, "#b9aba6"),
       (0.55, "#d6a784"), (0.615, "#e6a86c"), (0.70, "#efb97f"), (1.00, "#f3c89c")]
HAZE = [(0.00, "#5f7fae"), (0.40, "#8fa9c8"), (0.52, "#c1b0b0"), (0.575, "#e2b48f"), (0.615, "#e9b888"),
        (0.75, "#e0bb98"), (1.00, "#d4b89a")]

# ---- 山脈（遠 → 近）：base/amp 為高度比例，peaks = (x 比例, 高度比例, 半寬比例) ---------
LAYERS_FAR = [
    dict(base=0.600, amp=0.18, scale=560, seed=11, fog=0.80, shadow="#6e6e9d", lit="#c5b0bd", snow=0.55, relief=16,
         peaks=[(0.30, 0.10, 0.075), (0.84, 0.08, 0.06)], dips=[(0.60, 0.05, 0.06)]),
    dict(base=0.618, amp=0.24, scale=400, seed=12, fog=0.60, shadow="#4b5789", lit="#c0a9ad", snow=0.48, relief=22,
         peaks=[(0.46, 0.20, 0.085), (0.71, 0.11, 0.05), (0.08, 0.07, 0.08)], dips=[(0.60, 0.03, 0.05)]),
    dict(base=0.645, amp=0.16, scale=320, seed=13, fog=0.42, shadow="#334060", lit="#9a8f98", snow=0.40, relief=26,
         forest=0.35, treeline=0.55, peaks=[(0.17, 0.11, 0.06), (0.95, 0.10, 0.07)]),
    dict(base=0.690, amp=0.12, scale=250, seed=14, fog=0.27, shadow="#1f2d36", lit="#64716a", relief=28,
         forest=0.8, treeline=0.95),
]
LAYERS_NEAR = [
    dict(base=0.87, amp=0.07, scale=260, seed=15, fog=0.10, fog_down=0.25, fog_len=3.0, shadow="#10191a", lit="#3e5040",
         relief=22, forest=1.0, treeline=2.0, side="left", rise=0.40, reach=0.75, tree_px=12),
    dict(base=0.89, amp=0.07, scale=260, seed=16, fog=0.07, fog_down=0.2, fog_len=3.0, shadow="#0e1716", lit="#3b4d3d",
         relief=22, forest=1.0, treeline=2.0, side="right", rise=0.42, reach=0.72, tree_px=12),
]


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    sc = Scene(width, height, seed, log=log)
    W, H, S = sc.W, sc.H, sc.S
    sc.set_horizon(0.615)                               # 眼平線 = 水面鏡射軸
    sc.set_sun(0.60, 0.565, light_dir=(0.90, -0.42))    # 太陽坐在最遠山稜的鞍部上
    sc.set_haze(HAZE)

    # 1. 天空、太陽、雲
    sc.log("sky & clouds")
    sky(sc, SKY)
    disc = sun(sc)
    cirrus(sc)
    clouds = cumulus(sc)

    # 2. 遠山（大氣透視由 fog 參數控制）
    for i, L in enumerate(LAYERS_FAR):
        sc.log(f"mountain layer {i}")
        mountain_layer(sc, L)

    # 3. 谷底、霧、近景山坡
    sc.log("valley & near slopes")
    valley_floor(sc)
    mist(sc)
    near = np.zeros((H, W), F32)
    for L in LAYERS_NEAR:
        near = np.maximum(near, mountain_layer(sc, L))

    # 4. 河流（倒影取樣「目前畫面」，所以要在山與岸之後、前景之前）
    sc.log("river")
    center, half, t1 = river_course(sc)
    water, e = water_mask(sc, center, half, t1)
    water_surface(sc, water, e, t1)
    river_bank(sc, water, e, t1)
    haze_band(sc, exclude=near)

    # 5. 前景：沿岸小樹、兩側大松樹、地面、飛鳥
    sc.log("foreground")
    rng = np.random.default_rng(seed + 99)
    near_trees = []
    for i in range(7):                                   # 左前景
        hgt = rng.uniform(380, 640) * (1 - i * 0.06) * S
        near_trees.append(((30 + i * 62 + rng.uniform(-20, 20)) * S, H + rng.uniform(-10, 90) * S, hgt, hgt * 0.30))
    for i in range(5):                                   # 右前景
        hgt = rng.uniform(420, 700) * (1 - i * 0.07) * S
        near_trees.append((W - (30 + i * 70 + rng.uniform(-20, 20)) * S, H + rng.uniform(-10, 90) * S, hgt, hgt * 0.30))
    mid_trees = []
    for _ in range(60):                                  # 中景：沿著河岸、避開水面
        yb = rng.uniform(0.80, 0.96) * H
        tt = (yb - sc.horizon) / (H - sc.horizon)
        side = rng.choice([-1, 1])
        x = center[int(yb)] + side * (half[int(yb)] + rng.uniform(15, 160) * S)
        hgt = (rng.uniform(30, 70) + 120 * tt ** 1.5) * S
        if 0 < x < W and water[int(min(yb, H - 1)), int(x)] < 0.2 and water[int(min(yb - 10 * S, H - 1)), int(x)] < 0.2:
            mid_trees.append((x, yb, hgt, hgt * 0.32))
    tcol = lerp("#070d0c", "#13201a", fbm2d(H, W, sc.px(30), 3, seed=seed + 61) * 0.5 + 0.5)
    mid_mask = pine_mask(sc, mid_trees, seed_off=101)
    sc.cv.paint(lerp(tcol, sc.haze, 0.30), mid_mask)
    near_mask = pine_mask(sc, sorted(near_trees, key=lambda t: t[2]), seed_off=100)
    sc.cv.paint(tcol, near_mask)
    rim_light(sc, near_mask)
    sc.occlude(np.maximum(near_mask, mid_mask))
    ground_edge(sc, 0.955, "left", (0.18, 0.32), seed_off=62)
    ground_edge(sc, 0.965, "right", (0.16, 0.30), seed_off=63)
    birds(sc, [(0.27, 0.33, 12, 0.0), (0.295, 0.305, 10, 0.1), (0.255, 0.295, 8, -0.1), (0.31, 0.345, 7, 0.05)])

    # 6. 體積光與後製
    sc.log("light shafts & post")
    god_rays(sc, disc, clouds)
    finish(sc)
    return sc.cv


if __name__ == "__main__":
    import sys
    import time
    t0 = time.time()
    cv = render(log=lambda m: print(f"[{time.time() - t0:5.1f}s] {m}", flush=True))
    out = sys.argv[1] if len(sys.argv) > 1 else f"{NAME}.png"
    cv.save(out, preview=out.rsplit(".", 1)[0] + "_preview.jpg").stats()
