# -*- coding: utf-8 -*-
"""Any picture finished in a named medium: oil, watercolour, ukiyo-e ... (stylize)

* Format: 1920x1080
* Medium: <the style the request names; pick(PROMPT) finds it, STYLE is the fallback>
* Focal point: <the cottage beside the oak, left of centre>
* Light: <golden late sun from the right>
* Far / middle / near: <sky and hills / cottage and tree / meadow>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/styles.md (42 styles, parameters, `python tools/styles.py sheet` to compare).
from pixelcanvas import Canvas, sdf, clamp01, fbm1d
from pixelcanvas.elements import sky, sun
from pixelcanvas.flora import tree
from pixelcanvas.stylize import apply, pick

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_styled"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

STYLE = "oil"                       # used when PROMPT names no medium; see styles.md for all 42
STRENGTH = 1.0                      # < 1 mixes the unstyled picture back in
HILLS = ["#9fb48a", "#6f9150", "#4a7038"]      # far -> near
WALL, ROOF = "#efe2c8", "#b8543a"
LIGHT = (0.7, -0.6)                 # direction towards the sun (from the right)


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    X, Y = cv.xy()
    H, W = height, width

    # >>> setting: new_scene.py --prompt draws the request's place here (room, street, park, beach, night)
    log("sky")
    sky(cv, "golden", horizon=0.62)
    sun(cv, at=(0.8, 0.3), size=60, rays=0)
    log("hills")                                         # pale far hill -> dark near hill
    for i, col in enumerate(HILLS):
        ridge = (0.5 + 0.12 * i + 0.1 * (i == 2) + 0.05 * fbm1d(W, 600 * H / 1080, 3, seed=seed + i)) * H
        cv.paint(col, clamp01(Y - ridge[None, :] + 0.5))
    # <<< setting
    # >>> subjects: new_scene.py --prompt writes one call per subject of the request here
    log("cottage")                                       # sdf shapes: walls, roof, door, window
    x0, y0, s = 0.42 * W, 0.76 * H, 0.15 * H
    cv.paint(WALL, sdf.fill(sdf.sd_box(X, Y, x0, y0 - 0.5 * s, 0.7 * s, 0.5 * s)))
    cv.paint(ROOF, sdf.fill(sdf.sd_polygon(X, Y, [(x0 - 0.85 * s, y0 - s), (x0, y0 - 1.6 * s), (x0 + 0.85 * s, y0 - s)])))
    cv.paint("#5a3a2a", sdf.fill(sdf.sd_box(X, Y, x0 - 0.25 * s, y0 - 0.25 * s, 0.14 * s, 0.25 * s)))
    cv.paint("#f3c96b", sdf.fill(sdf.sd_box(X, Y, x0 + 0.3 * s, y0 - 0.55 * s, 0.15 * s, 0.13 * s)))
    log("tree")
    tree(cv, "oak", at=(0.24, 0.8), height=0.6, seed=seed, light=LIGHT)
    # <<< subjects

    log("medium")                                        # the named medium is a requirement: always its own step
    apply(cv, pick(PROMPT) or STYLE, strength=STRENGTH, seed=seed)
    return cv
