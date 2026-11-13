# -*- coding: utf-8 -*-
"""Poster / flyer / cover with strong typography (layout.template + text)

* Format: 1080x1350 portrait (story 1080x1920, square 1080x1080, A4 2480x3508)
* Focal point: <the striped sun in the picture area, centre>
* Text hierarchy: <kicker -> TITLE -> subtitle -> details in columns -> footer>
* Palette: <theme THEME (midnight, paper, neon, forest, sunset, mono, pastel, luxury, ocean)>
* Light: <glow around the focal shape>
* Exact text: <TITLE, SUBTITLE, DETAILS copied verbatim from the request>
"""
# Read skills/pixelcanvas/references/typography.md (sections 9 templates, 11 layout rules).
import numpy as np

from pixelcanvas import Canvas, sdf, layout

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "design"
NAME = "my_poster"
DEFAULT_SIZE = (1080, 1350)
DEFAULT_SEED = 1

TEMPLATE = "poster"                 # poster, flyer, book_cover, magazine, quote_card, slide, menu ... (typography section 9)
THEME = "midnight"
KICKER = "Summer series 2026"
TITLE = "Night Market"
SUBTITLE = "Street food, music and lanterns by the river"
DETAILS = [("When", "Fridays 7 pm"), ("Where", "Pier 4"), ("Entry", "Free")]
FOOTER = "nightmarket.example"
SUN = ("#ffcf6b", "#ff5e7e")        # focal shape gradient, top -> bottom


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    th = layout.theme(THEME)
    X, Y = cv.xy()

    log("background")
    cv.gradient(th["bg"])
    log("text")                                          # the template places and sizes every line
    boxes = layout.template(TEMPLATE, cv, title=TITLE, subtitle=SUBTITLE, kicker=KICKER, details=DETAILS,
                            footer=FOOTER, theme=THEME, seed=seed, background=False)
    pic = boxes["image"]                                 # the picture's area, also as fractions u0, v0, u1, v1
    u0, v0, u1, v1 = pic.x0 / width, pic.y0 / height, pic.x1 / width, pic.y1 / height
    # >>> subjects stage: new_scene.py --prompt writes one call per subject of the request here
    log("focal shape")
    cx, cy, r = pic.cx, pic.cy, 0.42 * pic.short
    sun = sdf.sd_circle(X, Y, cx, cy, r)
    band = np.abs(((Y - cy) / r * 6) % 1 - 0.5) * 2 - np.clip((Y - cy) / r * 0.7, 0, 1)
    stripes = np.clip(band * r / 12 + 0.5, 0, 1)                                            # retro cut lines
    cv.paint(SUN[0], sdf.glow(sun, 0.3 * r) * 0.35, "add")
    cv.linear((cx, cy - r), (cx, cy + r), [(0, SUN[0]), (1, SUN[1])], mask=sdf.fill(sun) * stripes)
    # <<< subjects

    log("post")
    cv.vignette(0.3, 0.65).grain(0.01, seed).soft_clip(0.9)
    return cv
