# -*- coding: utf-8 -*-
"""Abstract or scenic wallpaper from a palette, for desktop or phone (backdrop + palette)

* Format: 2560x1440 desktop (also 1080x1920 phone, 3840x2160 4K): no text, nothing important at the edges
* Focal point: <the sun and the ridge line, kept in the middle third>
* Light: <low sun behind the hills; layers get paler with distance>
* Layers: <one backdrop call, then a gentle vignette and fine grain>
* Palette: <PALETTE: a name, a phrase such as "soft blue and peach", or a list of hex colours>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/backdrops.md (backdrop names and options, palettes).
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_wallpaper"
DEFAULT_SIZE = (2560, 1440)
DEFAULT_SEED = 3

# >>> data: new_scene.py --prompt writes the request's motif, palette and mood here
BACKDROP = "hills"                      # hills mountains dunes ocean synthwave starry forest | mesh aurora waves ribbons
PALETTE = "sunset"                      # names, moods and colour words: "lavender", "soft blue and peach", ["#123", ...]
OPTIONS = dict(layers=6, sun_x=0.7)     # backdrop options; describe("hills") lists them for each backdrop
# <<< data
VIGNETTE = 0.2                          # 0 = none
GLOW = 0.25                             # soft light bloom in the palette's accent colour, 0 = none
GRAIN = 0.008                           # fine film grain keeps gradients from banding


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    log("wallpaper")                                     # one call paints the whole picture from palette + seed
    bd = backdrop(cv, BACKDROP, palette=PALETTE, seed=seed, **OPTIONS)
    log("glow")                                          # bd.palette gives the colours that were used
    cv.glow(OPTIONS.get("sun_x", 0.5) * width, 0.35 * height, 0.5 * height, bd.palette.accent, GLOW, mode="screen")
    log("post")                                          # darkened corners, then grain
    cv.vignette(VIGNETTE, 0.7).grain(GRAIN, seed)
    return cv
