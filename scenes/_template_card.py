# -*- coding: utf-8 -*-
"""Social card / thumbnail / link preview: a headline beside an illustration (text + pictograms)

* Format: 1280x720 thumbnail (1200x630 link card, 1080x1080 square)
* Focal point: <the headline on the left; the illustration on the right>
* Text hierarchy: <badge -> HEADLINE (2-6 words, huge) -> one subtitle line>
* Light: <a warm glow behind the illustration; text gets a soft shadow for contrast>
* Palette: <two background colours, cream ink, one accent for the badge>
* Exact text: <HEADLINE, SUBTITLE, BADGE copied verbatim from the request>
"""
# Read skills/pixelcanvas/references/typography.md (sections 1, 6, 8) and pictograms.md (shaded style).
import numpy as np

from pixelcanvas import Canvas, lerp, pictograms
from pixelcanvas.layout import Layout, stack

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "design"
NAME = "my_card"
DEFAULT_SIZE = (1280, 720)
DEFAULT_SEED = 1

BADGE = "NEW EPISODE"
HEADLINE = "Chasing the Last Light"
SUBTITLE = "A weekend in the hills"
INK, ACCENT = "#fff6e8", "#e8613a"
BG = [(0, "#25275e"), (1, "#6e3466")]          # background, top -> bottom
GLOW = "#ffb36b"                               # light behind the illustration


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    X, Y = cv.xy()
    S = height / 1080
    u0, v0, u1, v1 = 0.56, 0.16, 0.95, 0.88      # the illustration's area (fractions); the text takes the left

    log("background")
    cv.gradient(BG)
    cv.paint(GLOW, np.exp(-((X / width - 0.75) ** 2 * (width / height) ** 2 + (Y / height - 0.5) ** 2) / 0.09) * 0.45,
             "add")
    # >>> subjects stage: new_scene.py --prompt writes one call per subject of the request here
    log("hills")                                         # pictograms in the shaded style read as illustrations
    pictograms.pictogram(cv, "mountain", at=(u0 + 0.42 * (u1 - u0), v1), size=0.62 * (v1 - v0), color="#8a5a9c",
                         style="shaded", shadow=True, anchor="bm")
    log("sun")
    pictograms.pictogram(cv, "sun", at=(u0 + 0.78 * (u1 - u0), v0 + 0.3 * (v1 - v0)), size=0.3 * (v1 - v0),
                         color="#ffc857", style="shaded")
    # <<< subjects

    log("text")                                          # stack sizes and spaces the lines; an empty line is skipped
    L = Layout(cv, margin=0.07)
    shadow = dict(dx=0, dy=4 * S, blur=18 * S, opacity=0.55)
    lines = [{"text": BADGE, "role": "kicker", "color": INK, "bold": True, "plate": dict(kind="pill", color=ACCENT)},
             {"text": HEADLINE, "role": "title", "color": INK, "font": "display", "shadow": shadow},
             {"text": SUBTITLE, "role": "subtitle", "color": lerp(INK, ACCENT, 0.25), "shadow": shadow}]
    stack(cv, L.sub(0, 0.04, 0.55, 0.9), [ln for ln in lines if ln["text"]], valign="middle")

    log("post")
    cv.vignette(0.25).soft_clip(0.85).grain(0.008, seed)
    return cv
