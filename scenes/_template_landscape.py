# -*- coding: utf-8 -*-
"""Mountains, sky, clouds and a lake at sunset (scenery)

* Format: 1920x1080 (phone wallpaper 1080x1920 works too)
* Focal point: <the low sun over the far ridge at SUN>
* Light: <low sun, light travelling LIGHT_DIR; lit cloud bottoms, rim light on near trees>
* Far / middle / near: <pale snowy peaks / forested ridge / dark pines on the shore>
* Palette: <SKY stops + HAZE (atmospheric perspective target)>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/api.md (scenery recipes) and elements.md (more skies, weather, effects).
import numpy as np

from pixelcanvas import clamp01, fbm1d
from pixelcanvas.scenery import (Scene, sky, sun, cumulus, mountain_layer, mist, pine_mask, scatter_trees, rim_light,
                                 water_surface, god_rays, finish)

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_landscape"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

HORIZON = 0.64                      # eye level = the lake's mirror line
SUN = (0.62, 0.47)                  # sun position (u, v)
LIGHT_DIR = (0.9, -0.4)             # direction the light travels
CLOUDS = 0.5                        # cumulus coverage: smaller = more cloud
SKY = [(0.0, "#16294f"), (0.4, "#5f7fae"), (0.58, "#e2a77c"), (0.64, "#f3c89c")]
HAZE = [(0.0, "#6f86b0"), (0.55, "#d9ab8c"), (1.0, "#d4b89a")]
RANGES = [                          # far -> near; fog: 0.8 far and pale ... 0.1 near and dark
    dict(base=0.56, amp=0.2, scale=480, seed=11, fog=0.72, shadow="#6e6e9d", lit="#c9b3bd", snow=0.5,
         peaks=[(0.3, 0.12, 0.08)]),
    dict(base=0.61, amp=0.12, scale=300, seed=12, fog=0.38, shadow="#26344a", lit="#7d7a84", relief=24),
]


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    sc = Scene(width, height, seed, log=log)
    sc.set_horizon(HORIZON)
    sc.set_sun(*SUN, light_dir=LIGHT_DIR)
    sc.set_haze(HAZE)

    sc.log("sky")                                        # gradient + sun disc and glow
    sky(sc, SKY)
    disc = sun(sc)
    sc.log("clouds")                                     # coverage / band / softness shape the clouds
    clouds = cumulus(sc, coverage=CLOUDS, softness=0.3)
    for i, L in enumerate(RANGES):                       # one layer per mountain range
        sc.log(f"mountains {i + 1}")
        mountain_layer(sc, L)
    sc.log("mist")
    mist(sc, v_range=(HORIZON - 0.06, HORIZON - 0.02, HORIZON, HORIZON + 0.02), strength=0.5)

    sc.log("lake")                                       # mirrors everything painted above the horizon
    depth = np.clip((np.arange(sc.H) - sc.horizon) / (sc.H - sc.horizon), 0, 1)     # 0 at the horizon -> 1 at the bottom
    water = clamp01(sc.Y - sc.horizon + 0.5)
    water_surface(sc, water, np.full_like(water, -1e4), depth, ripple=0.8)
    sc.log("near shore")                                 # dark bank across the bottom
    shore = (0.92 + 0.03 * fbm1d(sc.W, sc.px(300), 4, seed=seed + 3)) * sc.H
    bank = clamp01(sc.Y - shore[None, :] + 0.5)
    sc.cv.paint("#10161a", bank)
    sc.log("pines")                                      # backlit silhouettes; by day use flora.forest(cv, region=...)
    pines = pine_mask(sc, scatter_trees(sc, 12, (0.0, 0.26), (0.93, 1.0), height=(240, 520)))
    sc.cv.paint("#0b1012", pines)
    rim_light(sc, pines)
    sc.occlude(np.maximum(bank, pines))

    sc.log("post")                                       # light shafts, bloom, grade, vignette
    god_rays(sc, disc, clouds)
    finish(sc)
    return sc.cv
