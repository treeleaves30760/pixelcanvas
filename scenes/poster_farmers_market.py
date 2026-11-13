# -*- coding: utf-8 -*-
"""Bilingual flyer for a weekend farmers' market in Taipei: big Traditional Chinese title, vegetables, warm paper

* Format: 1080x1350 portrait
* Focal point: the title 週末小農市集, above a woven basket overflowing with vegetables
* Text hierarchy: date chip -> Chinese title -> English line -> location in the footer band
* Palette: warm paper, brick-red accent, vegetable colours
* Exact text: 週末小農市集; Weekend Farmers Market; 11/21–11/22 10:00–17:00; 華山文創園區
"""
# Read skills/pixelcanvas/references/typography.md (templates: flyer) and pictograms.md.
import numpy as np

from pixelcanvas import Canvas, layout, sdf
from pixelcanvas.backdrop import backdrop
from pixelcanvas.pictograms import pictogram

PROMPT = ("A flyer for a weekend farmers' market in Taipei: title '週末小農市集' with the English line 'Weekend "
          "Farmers Market', dates '11/21–11/22 10:00–17:00', location '華山文創園區', illustrated vegetables and a "
          "warm paper-like background.")
NAME = "poster_farmers_market"
KIND = "design"
DEFAULT_SIZE = (1080, 1350)
DEFAULT_SEED = 1

TITLE, SUBTITLE = "週末小農市集", "Weekend Farmers Market"
DATE, PLACE = "11/21–11/22 10:00–17:00", "華山文創園區"
BACK = [("broccoli", "#4f9a3c", "#2f6a2a"), ("corn", "#f2c53d", "#4d9a3a"), ("cabbage", "#9cc56a", "#5f8f3a"),
        ("eggplant", "#6a3f8a", "#3f8a3a")]              # (name, colour, detail colour), the taller back row
FRONT = [("carrot", "#f08a24", "#3f9a3a"), ("tomato", "#d9402f", "#3f8a3a"), ("radish", "#d8456a", "#3f9a3a"),
         ("pepper", "#3f9a4a", "#2a6a30"), ("mushroom", "#c9a27a", "#8a5a3a")]
ROWS = [(BACK, 0.50, 0.50, 0.20), (FRONT, 0.36, 0.40, 0.14)]   # (row, size, lift, spread) as picture-box fractions
BASKET = (0.10, 0.90, 0.44, 0.04)                        # left, right, rim and bottom as picture-box fractions
WEAVE_ROWS = 6
WICKER = ("#c08a4e", "#8a5a2e", "#e0b57a")               # body, weave shadow, rim highlight


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    S = height / 1080
    log("paper")
    backdrop(cv, "warm_paper", seed=seed)
    log("text")
    boxes = layout.template("flyer", cv, title=TITLE, subtitle=SUBTITLE, date=DATE, place=PLACE, theme="paper",
                            background=False, seed=seed)
    pic = boxes["image"]
    for row, size, lift, spread in ROWS:                 # back row first; the basket hides their lower halves
        n = len(row)
        for i, (name, color, detail) in enumerate(row):
            log(name)
            u = (pic.x0 + pic.w * (spread + (1 - 2 * spread) * i / (n - 1))) / width
            v = (pic.y1 - pic.h * lift) / height
            pictogram(cv, name, at=(u, v), size=size * pic.h / height, color=color, detail_color=detail,
                      style="shaded", shadow=True, anchor="bm")

    log("basket")                                        # a woven basket drawn freehand with sdf
    X, Y = cv.xy()
    l, r = pic.x0 + pic.w * BASKET[0], pic.x0 + pic.w * BASKET[1]
    top, bot = pic.y1 - pic.h * BASKET[2], pic.y1 - pic.h * BASKET[3]
    taper = 0.07 * (r - l)
    body = sdf.sd_polygon(X, Y, [(l, top), (r, top), (r - taper, bot), (l + taper, bot)])
    shadow = sdf.fill(sdf.sd_ellipse(X, Y, (l + r) / 2, bot, 0.5 * (r - l), 18 * S), aa=30 * S)
    cv.paint("#5a4030", shadow * 0.35)
    inside = sdf.fill(sdf.offset(body, -2 * S))
    row = (Y - top) / ((bot - top) / WEAVE_ROWS)         # woven strips: each row offset by half a stitch
    col = (X - l) / ((bot - top) / WEAVE_ROWS) / 1.6 + 0.5 * np.floor(row)
    stitch = np.abs(np.sin(np.pi * col)) * np.sin(np.pi * np.clip(row % 1.0, 0, 1))
    cv.paint(WICKER[0], inside)
    cv.paint(WICKER[1], inside * np.clip(1 - 1.6 * stitch, 0, 1) * 0.6)
    cv.paint(WICKER[2], inside * np.clip(stitch - 0.75, 0, 1) * 1.6)
    cv.paint(WICKER[1], inside * np.clip((Y - top) / (bot - top), 0, 1) * 0.35)      # darker towards the base
    rim = sdf.sd_box(X, Y, (l + r) / 2, top, (r - l) / 2 + 10 * S, 16 * S) - 12 * S
    cv.paint(WICKER[1], sdf.fill(rim))
    cv.paint(WICKER[2], sdf.fill(rim + 5 * S) * np.clip((top - Y) / (16 * S) + 0.6, 0, 1))
    log("post")
    cv.vignette(0.15, 0.75)
    return cv
