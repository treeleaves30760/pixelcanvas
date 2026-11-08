# -*- coding: utf-8 -*-
"""
Design / poster template — copy via ``python tools/new_scene.py <name> --poster``.

Write the plan here before coding (then check it against the render with ``look.py check``):
* Format: 1080x1350 portrait poster (change DEFAULT_SIZE for square 1080x1080, story 1080x1920, thumb 1280x720).
* Focal element: <what is biggest / brightest / most saturated?>  at (U, V) = (0.5, 0.55)
* Text hierarchy: title (top third) → subtitle → info (bottom); margins MARGIN of the short side.
* Palette: background BG, accent ACCENT, ink INK. Light from the upper left.

Conventions: positions are fractions (0~1) of W / H; pixel sizes are designed at 1080 px on the short side and
multiplied by S, so drafts (--scale 0.5) and full size look identical. Each log("...") call is one step in
``render.py --stages``.
"""
import numpy as np

from pixelcanvas import *
from pixelcanvas import sdf

NAME = "my_poster"
DEFAULT_SIZE = (1080, 1350)
DEFAULT_SEED = 1

TITLE = "TITLE HERE"
SUBTITLE = "a short subtitle"
INFO = "DATE · PLACE"
BG = [(0.0, "#1b1f3a"), (1.0, "#0d0f1e")]
ACCENT = "#ffb347"
INK = "#f4efe6"
MARGIN = 0.07


def render(width=1080, height=1350, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = min(W, H) / 1080
    m = MARGIN * min(W, H)
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()

    log("background")
    cv.gradient(BG)
    cv.paint("#ffffff", (fbm2d(H, W, 300 * S, 4, seed=seed) * 0.5 + 0.5) * 0.04, "add")   # subtle texture

    log("focal shape")
    cx, cy, r = 0.5 * W, 0.55 * H, 0.28 * min(W, H)
    d = sdf.sd_circle(X, Y, cx, cy, r)
    cv.paint(ACCENT, sdf.glow(d, 60 * S) * 0.35, "add")
    cv.paint(lerp(ACCENT, "#ffffff", 0.15) * sdf.bevel_light(d, 30 * S, strength=0.3)[..., None], sdf.fill(d))

    log("text")
    cv.text(TITLE, box=(m, m, W - m, m + 0.14 * H), color=INK, bold=True, shadow=dict(blur=10 * S, opacity=0.5))
    t = cv.last_bbox
    cv.text(SUBTITLE, (W / 2, t[3] + 24 * S), 40 * S, lerp(INK, ACCENT, 0.4), anchor="mt")
    cv.text(INFO, (W / 2, H - m), 30 * S, INK, anchor="mb", tracking=0.25, bold=True)

    log("post")
    cv.vignette(0.3, 0.6).grain(0.01, seed).soft_clip(0.85)
    return cv
