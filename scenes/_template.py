# -*- coding: utf-8 -*-
"""
新場景範本 — 複製成 scenes/<名字>.py 後修改（或 python tools/new_scene.py <名字>）。

約定：
* 提供 ``render(width, height, seed, log) -> Canvas`` 與 ``DEFAULT_SIZE``。
* 位置用比例（0~1）；像素單位（噪聲 scale、sigma、樹高）以 1080p 為基準，用 ``sc.px()`` 換算。
* 順序：天空 → 遠 → 近 → 水面（倒影取樣目前畫面）→ 前景 → 光線 → 後製。
* 只用低階 API 也完全可以：``sc.cv`` 就是 Canvas，``sc.X/Y/U/V`` 是座標網格。
"""
import numpy as np

from pixelcanvas import *                 # Canvas、噪聲、遮罩、筆刷、濾鏡、顏色（低階）
from pixelcanvas.scenery import *         # Scene 與景物配方（高階）

NAME = "my_scene"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

SKY = [(0.0, "#0c1c3e"), (0.5, "#86a6c2"), (0.62, "#e6a86c"), (1.0, "#f3c89c")]
HAZE = [(0.0, "#5f7fae"), (0.6, "#e2b48f"), (1.0, "#d4b89a")]
LAYERS = [
    dict(base=0.60, amp=0.18, scale=500, seed=11, fog=0.75, shadow="#6e6e9d", lit="#c5b0bd", snow=0.55,
         peaks=[(0.35, 0.12, 0.08)]),
    dict(base=0.66, amp=0.14, scale=300, seed=12, fog=0.40, shadow="#334060", lit="#9a8f98", forest=0.5, treeline=0.7),
    dict(base=0.80, amp=0.10, scale=260, seed=13, fog=0.10, fog_down=0.3, shadow="#10191a", lit="#3e5040",
         forest=1.0, treeline=2.0, side="left", rise=0.3, reach=0.7, tree_px=10),
]


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    sc = Scene(width, height, seed, log=log)
    sc.set_horizon(0.62)
    sc.set_sun(0.62, 0.56, light_dir=(0.9, -0.4))
    sc.set_haze(HAZE)

    sc.log("sky")
    sky(sc, SKY)
    disc = sun(sc)
    clouds = cumulus(sc, coverage=0.58)

    for i, L in enumerate(LAYERS):
        sc.log(f"layer {i}")
        mountain_layer(sc, L)

    # 在這裡加東西：valley_floor / mist / river_course+water_mask+water_surface / pine_mask / birds ...
    # 或直接用低階 API，例如：
    #   X, Y = sc.X, sc.Y
    #   m = clamp01(Y - (0.9 * sc.H + sc.px(20) * fbm1d(sc.W, sc.px(90), seed=seed + 5))[None, :] + 0.5)
    #   sc.cv.paint("#0a100e", m); sc.occlude(m)

    sc.log("post")
    god_rays(sc, disc, clouds)
    finish(sc)
    return sc.cv
