# -*- coding: utf-8 -*-
"""Bare conventions on an empty canvas, for the low-level API (Canvas, sdf, noise)

* Format: 1920x1080
* Focal point: <what is biggest / brightest / most saturated, and where (u, v)>
* Light: <where it comes from; every lit side faces it>
* Far / middle / near (or text hierarchy): <...>
* Palette: <3-5 colours>
* Exact text: <copied verbatim from the request>
"""
# Read skills/pixelcanvas/references/api.md (Canvas, masks, sdf, noise, filters) and techniques.md (how to draw X).
from pixelcanvas import Canvas, sdf, fbm2d

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_blank"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

BG = [(0.0, "#1b2438"), (1.0, "#3b4a6b")]      # background gradient stops, top -> bottom
SUBJECT = "#ffb347"
SUBJECT_AT, SUBJECT_R = (0.5, 0.5), 0.18        # centre (u, v) and radius as a fraction of the height


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    X, Y = cv.xy()
    S = height / 1080                                    # pixel sizes are designed at 1080p and multiplied by S

    # >>> setting: new_scene.py --prompt draws the request's place here (room, street, park, beach, night)
    log("background")                                    # one log() per object a person may move or hide
    cv.gradient(BG)
    cv.paint("#ffffff", (fbm2d(height, width, 300 * S, 4, seed=seed) * 0.5 + 0.5) * 0.04, "add")

    # <<< setting
    # >>> subjects: new_scene.py --prompt writes one call per subject of the request here
    log("subject")                                       # mask first (sdf / noise), colour second, cv.paint draws
    d = sdf.sd_circle(X, Y, SUBJECT_AT[0] * width, SUBJECT_AT[1] * height, SUBJECT_R * height)
    cv.paint(SUBJECT, sdf.glow(d, 60 * S) * 0.3, "add")
    cv.paint(SUBJECT, sdf.fill(d))
    # <<< subjects

    log("post")                                          # post-processing last
    cv.vignette(0.25).soft_clip(0.85).grain(0.01, seed)
    return cv
