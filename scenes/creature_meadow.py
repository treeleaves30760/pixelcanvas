# -*- coding: utf-8 -*-
"""
Summer meadow (creature_meadow) — animals in a landscape with pixelcanvas.creature.

Composition: horizon at V=0.52 behind rolling hills and a band of trees. A bay horse gallops across the middle
ground from right to left (left third), a red fox sits in the tall grass in the right foreground and watches it,
a few crows fly over the hills, and a monarch butterfly hovers over the flowers in the left foreground.
Light: late-morning sun from the upper left; the horse casts a soft shadow on the grass.
"""
import numpy as np

from pixelcanvas import Canvas, mix, fbm2d, clamp01, smoothstep, lerp
from pixelcanvas.creature import Creature, flock
from pixelcanvas.form import set_light
from pixelcanvas.flora import forest, meadow, grass, flower, LIGHT

NAME = "creature_meadow"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 11

HORIZON = 0.52
SKY = [(0.0, "#6f9fd0"), (0.35, "#9cc0e0"), (0.52, "#e6ecd8")]
HILLS = [("#9db6a6", 0.43, 0.05, 1.3), ("#7f9e6e", 0.48, 0.045, 2.1)]   # colour, base line, amplitude, frequency
FIELD = [(0.5, "#a9b86a"), (0.66, "#7f9a42"), (1.0, "#4e6e26")]
HAZE = "#cfdcd2"
HORSE = dict(at=(0.37, 0.76), size=0.31, phase=0.25, coat=dict(base="#8a4a24", belly="#9a5830", mane="#1e140e",
                                                                  tail="#1e140e"))
FOX = dict(at=(0.8, 0.93), size=0.2)
BUTTERFLY = dict(at=(0.17, 0.66), size=0.075, phase=0.75)
BIRDS = dict(region=(0.55, 0.1, 0.92, 0.3), n=5, size=0.035)


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = X / W, Y / H
    set_light(direction=(-0.7, -0.62), elevation=42)

    log("sky")
    cv.paint(mix(SKY, V))
    n = fbm2d(H, W, 260 * S, 5, seed=seed + 1, aspect=0.4)
    clouds = smoothstep(0.08, 0.38, n - 0.1 + 0.25 * V) * (1 - smoothstep(0.3, 0.46, V))
    cv.paint(lerp("#dfe8f2", "#ffffff", smoothstep(0.2, 0.6, n)), clouds * 0.8)
    cv.glow(0.08 * W, 0.05 * H, 0.3 * H, "#fff1cc", 0.22)

    log("hills")
    for i, (col, base, amp, freq) in enumerate(HILLS):
        line = base - amp * (0.55 + 0.45 * np.sin(U * freq * np.pi + i * 1.7)) \
            - amp * 0.35 * fbm2d(H, W, 300 * S, 3, seed=seed + 10 + i)[0:1, :]
        m = clamp01((V - line) * H + 0.5)
        shade = 1 - 0.08 * smoothstep(0.0, 0.05, V - line)
        cv.paint(lerp(col, HAZE, 0.35 - 0.15 * i) * shade[..., None], m)

    log("trees")
    forest(cv, region=(-0.05, 0.47, 1.05, 0.53), n=40, species=("oak", "poplar", "birch", "maple"),
           height=(0.05, 0.11), seed=seed + 3, haze=(HAZE, 0.5), light=LIGHT)

    log("field")
    ground = clamp01((V - HORIZON) * H + 0.5)
    col = mix(FIELD, V)
    tex = fbm2d(H, W, 60 * S, 4, seed=seed + 4, aspect=3.0)
    col = col * (0.9 + 0.18 * tex)[..., None]
    col = lerp(col, HAZE, (1 - smoothstep(HORIZON, 0.62, V)) * 0.5)
    cv.paint(col, ground)
    meadow(cv, region=(0.0, 0.54, 1.0, 0.7), seed=seed + 5, density=0.8, grass_height=0.03, light=LIGHT)

    log("birds")
    flock(cv, "crow", n=BIRDS["n"], region=BIRDS["region"], size=BIRDS["size"], seed=seed + 6, facing="left")

    log("horse")
    h = Creature("horse", size=HORSE["size"], facing="left", coat=HORSE["coat"], seed=seed)
    h.pose("gallop", phase=HORSE["phase"])
    h.draw(cv, at=HORSE["at"], cast=dict(opacity=0.35, length=0.6))

    log("meadow")
    meadow(cv, region=(0.0, 0.7, 1.0, 0.86), seed=seed + 7, density=1.0, grass_height=0.05, light=LIGHT)

    log("fox")
    fox = Creature("fox", size=FOX["size"], facing="left", view="three-quarter", seed=seed)
    fox.pose("sit", head_turn=12.0)
    fox.draw(cv, at=FOX["at"])

    log("foreground grass")
    grass(cv, region=(0.0, 0.88, 1.0, 1.02), tufts=40, height=0.09, width=0.06, blades=24, seed=seed + 8, wind=0.12)
    grass(cv, region=(0.68, 0.9, 0.95, 1.0), tufts=10, height=0.1, width=0.05, blades=20, seed=seed + 9, wind=0.1)
    for i, (u, v, kind, hgt) in enumerate(((0.08, 0.95, "daisy", 0.2), (0.15, 0.97, "poppy", 0.26),
                                           (0.24, 0.96, "cornflower", 0.2), (0.11, 0.99, "daisy", 0.16),
                                           (0.03, 0.99, "poppy", 0.2))):
        flower(cv, kind, at=(u, v), height=hgt, seed=seed + 20 + i, light=LIGHT)

    log("butterfly")
    b = Creature("butterfly", size=BUTTERFLY["size"], facing="right", seed=seed)
    b.pose("fly", phase=BUTTERFLY["phase"])
    b.draw(cv, at=BUTTERFLY["at"], shadow=False)

    log("post")
    vig = 1 - 0.18 * smoothstep(0.55, 1.1, np.hypot(U - 0.5, (V - 0.5) * 0.9) * 1.6)
    cv.paint(np.ones((H, W, 3), np.float32) * vig[..., None], mode="multiply")
    return cv
