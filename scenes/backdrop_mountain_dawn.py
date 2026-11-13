# -*- coding: utf-8 -*-
"""Desktop wallpaper: layered mountain ranges at dawn, mist between the ridges, a low sun, calm sky for icons

* Format: 2560x1440 desktop, no text, nothing important at the edges
* Focal point: the sun low over the ridges, left of centre
* Light: low sun behind the far range; each range paler with distance
* Palette: peach and lavender sky, slate-blue ranges
"""
# Read skills/pixelcanvas/references/backdrops.md (backdrop names and options, palettes).
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop

PROMPT = "A calm desktop wallpaper of layered mountains at dawn with mist between the ridges, peach and lavender."
NAME = "backdrop_mountain_dawn"
KIND = "painting"
DEFAULT_SIZE = (2560, 1440)
DEFAULT_SEED = 3

BACKDROP = "mountains"
PALETTE = "peach and lavender"
OPTIONS = dict(layers=5, sun_x=0.36, mist=0.6)
GLOW = 0.25                                              # soft bloom around the sun, 0 = none
VIGNETTE = 0.18
GRAIN = 0.008                                            # fine grain keeps the gradients from banding


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    log("wallpaper")                                     # one call paints sky, sun, ranges and mist
    bd = backdrop(cv, BACKDROP, palette=PALETTE, seed=seed, **OPTIONS)
    log("glow")
    cv.glow(OPTIONS["sun_x"] * width, 0.45 * height, 0.5 * height, bd.palette.accent, GLOW, mode="screen")
    log("post")
    cv.vignette(VIGNETTE, 0.7).grain(GRAIN, seed)
    return cv
