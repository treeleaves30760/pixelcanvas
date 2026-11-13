# -*- coding: utf-8 -*-
"""Retouch and grade an existing image (Canvas.load + filters)

* Format: the output size (DEFAULT_SIZE); the photo is centre-cropped to fill it
* Subject: <what the photo shows and what must stay untouched (faces, product colours)>
* Exposure: <black / white points, gamma>
* Local edits: <darken the bright sky, lift the shadows, warm the subject>
* Grade: <warm highlights, cool shadows, saturation, vignette>
* Exact text: <none, or a caption>
"""
# Read skills/pixelcanvas/references/techniques.md section 11 (photo editing) and styles.md (photographic looks).
import os

from pixelcanvas import Canvas, luma, smoothstep
from pixelcanvas.elements import sky, sun, ocean

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_photo"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

PHOTO = "photo.jpg"                 # the image to edit (path from the project root); missing = a generated stand-in
LEVELS = (0.03, 0.97, 1.05)         # black point, white point, gamma
SKY_DARKEN = 0.85                   # multiplier for bright areas in the top half (1 = off)
SPLIT = ("#3d4a66", "#f2c58a", 0.08)   # shadow tint, highlight tint, amount
SATURATION = 1.08
VIGNETTE = 0.3


def load_photo(width, height, seed):
    """PHOTO centre-cropped to width x height."""
    if not os.path.exists(PHOTO):
        cv = Canvas(width, height)
        sky(cv, "sunset", horizon=0.55)
        sun(cv, at=(0.62, 0.42), size=30)
        ocean(cv, horizon=0.55, sun=(0.62, 0.42), seed=seed)
        return cv
    cv = Canvas.load(PHOTO)
    k = max(width / cv.w, height / cv.h)
    cv.resize(max(width, round(cv.w * k)), max(height, round(cv.h * k)))
    x0, y0 = (cv.w - width) // 2, (cv.h - height) // 2
    return cv.crop(x0, y0, x0 + width, y0 + height)


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    log("photo")
    cv = load_photo(width, height, seed)
    cv.levels(*LEVELS)
    U, V = cv.uv()

    log("sky")                                           # local edit = a mask first, then paint the adjusted pixels
    bright_top = smoothstep(0.55, 0.85, luma(cv.buf)) * (1 - smoothstep(0.3, 0.6, V))
    cv.paint(cv.buf * SKY_DARKEN, bright_top)

    log("post")                                          # global grade last
    cv.split_tone(*SPLIT).saturation(SATURATION).vignette(VIGNETTE).grain(0.01, seed)
    return cv
