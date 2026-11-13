# -*- coding: utf-8 -*-
"""Objects with volume on a table, lit by a window (form)

* Format: 1920x1080 (1080x1350 portrait works too)
* Focal point: <the mug just left of centre; brightest highlight on its rim>
* Light: <window on the upper left (LIGHT_DIR); shadows fall right and back>
* Back / middle / front: <bottle / mug and oranges / lemon at the front edge>
* Palette: <warm plaster wall, walnut table, one saturated accent (the oranges)>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/forms.md (solids, materials, shadows, tabletop).
import numpy as np

from pixelcanvas import Canvas, mix
from pixelcanvas.form import set_light, tabletop, lathe, cylinder, ball, reflection

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_still_life"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

LIGHT_DIR, ELEVATION = (-0.85, -0.5), 32         # direction towards the window; sun height in degrees
WALL = [(0.0, "#d8c6ac"), (1.0, "#7d6b5a")]      # lit corner -> shaded corner
TABLE = dict(top=0.6, front=0.96, horizon=0.42)  # table back edge, front edge, eye level (fractions)
MUG, BOTTLE, FRUIT = "#e9e1d3", "#5f9a72", "#f07a18"
VARNISH = 0.15                                    # reflections in the table top


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    U, V = cv.uv()
    set_light(direction=LIGHT_DIR, elevation=ELEVATION)

    # >>> setting: new_scene.py --prompt draws the request's place here (room, street, park, beach, night)
    log("wall")                                          # gradient away from the window + a soft light patch
    cv.paint(mix(WALL, np.clip(U * 0.9 + V * 0.3, 0, 1)))
    cv.glow(0.2 * width, 0.25 * height, 0.25 * height, "#ffe2b8", 0.25)
    log("table")                                         # kind="wood" | "marble" | "cloth" ...
    table = tabletop(cv, kind="wood", seed=seed, **TABLE)

    # <<< setting
    # >>> subjects: new_scene.py --prompt writes one call per subject of the request here
    log("bottle")                                        # lathe = any round object from (radius, height) pairs
    bottle = lathe(cv, table.place(0.3, 0.75), [(0.085, 0), (0.094, 0.03), (0.094, 0.26), (0.036, 0.4),
                                                (0.028, 0.53)], base=BOTTLE, material="glass", cast=True)
    reflection(cv, bottle, VARNISH)
    log("oranges")                                       # table.place(u, depth): depth 0 = front edge, 1 = back
    for u, depth in ((0.66, 0.62), (0.76, 0.7)):
        reflection(cv, ball(cv, table.place(u, depth), 0.085, base=FRUIT, material="fruit", cast=True), VARNISH)
    log("mug")
    mug = cylinder(cv, table.place(0.47, 0.42), 0.1, 0.21, base=MUG, material="ceramic", open=True, fill=0.85,
                   fill_color="#3b1f12", handle=True, cast=True)
    reflection(cv, mug, VARNISH)
    log("lemon")                                         # radius=(rx, ry) makes an ellipsoid
    reflection(cv, ball(cv, table.place(0.72, 0.16), (0.1, 0.07), base="#f2c84b", material="fruit", cast=True),
               VARNISH)
    # <<< subjects

    log("post")
    cv.vignette(0.3, 0.62).soft_clip(0.86).grain(0.01, seed)
    return cv
