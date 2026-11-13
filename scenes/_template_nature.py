# -*- coding: utf-8 -*-
"""Close natural scene: tree, rocks, flowers, grass, clouds (flora + elements + texture)

* Format: 1920x1080
* Focal point: <the old oak on the hill at TREE_AT>
* Light: <afternoon sun from the upper left (LIGHT); every plant casts its shadow to the right>
* Far / middle / near: <hazy tree line / hill with the oak / meadow, rocks, grass tufts>
* Palette: <blue sky, warm greens, grey rocks, flower accents>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/flora.md (plants, rocks), elements.md (sky, weather, fire, water), textures.md.
import numpy as np

from pixelcanvas import Canvas, clamp01
from pixelcanvas.elements import sky, cumulus
from pixelcanvas.flora import forest, plant, rock, meadow, grass
from pixelcanvas.texture import fill, plane_uv

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_nature"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

HORIZON = 0.6                       # where the far tree line meets the ground
LIGHT = (-0.6, -0.7)                # direction towards the light (upper left)
TREE = "old oak"                    # any flora preset or phrase: "cherry", "weeping willow", "palm" ...
TREE_AT, TREE_HEIGHT = (0.36, 0.78), 0.66
FLOWERS = ("poppy", "daisy", "cornflower")
GROUND = ("moss", "#94b45a")        # texture + tint: ("sand", "#e0c890"), ("snow", "#ffffff"), ("dry_earth", None)
SKY = "day"                         # elements.sky kinds: noon day morning golden sunset dusk night storm overcast


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    X, Y = cv.xy()

    # >>> setting: new_scene.py --prompt draws the request's place here (room, street, park, beach, night)
    log("sky")                                           # sky kind + clouds (at = base centre, size = width in px)
    sky(cv, SKY, horizon=HORIZON)
    cumulus(cv, at=(0.72, 0.36), size=620, seed=seed, light=(-LIGHT[0], LIGHT[1]))
    cumulus(cv, at=(0.16, 0.3), size=380, seed=seed + 1, light=(-LIGHT[0], LIGHT[1]))
    log("distant trees")                                 # small and hazy = far away
    forest(cv, region=(-0.05, HORIZON + 0.01, 1.05, HORIZON + 0.05), n=40, height=(0.07, 0.13),
           haze=("#b9cde0", 0.55), seed=seed + 2, light=LIGHT)

    log("ground")                                        # a hill under the tree; texture name sets the material
    hill = HORIZON + 0.04 - 0.12 * np.exp(-((X / width - TREE_AT[0]) / 0.3) ** 2)
    ground = clamp01(Y - hill * height + 0.5)
    fill(cv, ground, GROUND[0], uv=plane_uv(height, width, horizon=HORIZON - 0.1), tint=GROUND[1], light=LIGHT,
         seed=seed)
    # <<< setting
    # >>> subjects: new_scene.py --prompt writes one call per subject of the request here
    log("tree")
    plant(cv, TREE, at=TREE_AT, height=TREE_HEIGHT, seed=seed + 3, light=LIGHT, shadow=True)
    log("rocks")
    rock(cv, at=(0.7, 0.84), size=0.2, moss=0.4, seed=seed + 4, light=LIGHT, shadow=True)
    rock(cv, at=(0.81, 0.86), size=0.09, seed=seed + 5, light=LIGHT, shadow=True)
    log("flowers")                                       # grass_height also sets the flower size
    meadow(cv, region=(0.0, 0.87, 1.0, 1.0), flowers=FLOWERS, grass_height=0.12, density=0.9, seed=seed + 6,
           light=LIGHT)
    log("grass")
    grass(cv, region=(0.0, 0.95, 1.0, 1.03), tufts=50, height=0.08, seed=seed + 7, light=LIGHT)
    # <<< subjects

    log("post")                                          # sun glow from the light side, gentle vignette
    cv.glow(0.05 * width, 0.05 * height, 0.45 * height, "#fff0c8", 0.25, mode="screen")
    cv.vignette(0.22, 0.7).soft_clip(0.9).grain(0.008, seed)
    return cv
