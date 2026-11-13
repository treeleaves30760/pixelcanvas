# -*- coding: utf-8 -*-
"""
A field guide sheet (creature_parade) — the pixelcanvas.creature showcase.

Thirty animals, one call each, on a warm paper sheet: mammals in natural poses (sitting, walking, trotting, galloping,
grazing, lying), birds standing, perched, swimming and flying, sea life and small creatures. Every cell is filled with
``Creature(...).draw(cv, fit=cell)``, so the sheet layout never has to know how big an animal is.
Light: soft studio light from the upper left; each animal casts a soft contact shadow on the paper.
"""
import numpy as np

from pixelcanvas import Canvas, fbm2d, smoothstep
from pixelcanvas.creature import Creature
from pixelcanvas.form import set_light

NAME = "creature_parade"
DEFAULT_SIZE = (1920, 1600)
DEFAULT_SEED = 3

PAPER = [(0.0, "#f4eddd"), (1.0, "#ece2cc")]
INK = "#3a2e24"
RULE = "#c9b896"
TITLE = "A Field Guide to Animals"
SUBTITLE = "thirty species, one call each — pixelcanvas.creature"
COLS, ROWS = 6, 5
STYLE = "soft"

# (species, pose, phase, view, options, label)
SHEET = [
    ("cat", "sit", 0.0, "three-quarter", {}, "Cat"),
    ("labrador", "walk", 0.15, "side", {}, "Labrador"),
    ("fox", "trot", 0.25, "three-quarter", {}, "Red fox"),
    ("horse", "gallop", 0.2, "side", {}, "Horse"),
    ("cow", "graze", 0.0, "three-quarter", {}, "Cow"),
    ("elephant", "walk", 0.2, "three-quarter", {}, "Elephant"),
    ("lion", "stand", 0.0, "three-quarter", {}, "Lion"),
    ("tiger", "walk", 0.35, "side", {}, "Tiger"),
    ("zebra", "stand", 0.0, "three-quarter", {}, "Zebra"),
    ("giraffe", "walk", 0.3, "side", {}, "Giraffe"),
    ("deer", "stand", 0.0, "three-quarter", {}, "Red deer"),
    ("bear", "walk", 0.6, "three-quarter", {}, "Brown bear"),
    ("wolf", "trot", 0.6, "side", {}, "Wolf"),
    ("corgi", "sit", 0.0, "three-quarter", {}, "Corgi"),
    ("sheep", "graze", 0.0, "side", {}, "Sheep"),
    ("pig", "stand", 0.0, "three-quarter", {}, "Pig"),
    ("rabbit", "sit", 0.0, "three-quarter", {}, "Rabbit"),
    ("squirrel", "sit", 0.0, "three-quarter", {}, "Squirrel"),
    ("owl", "perch", 0.0, "side", {}, "Owl"),
    ("parrot", "perch", 0.0, "three-quarter", {}, "Macaw"),
    ("flamingo", "stand", 0.0, "side", {}, "Flamingo"),
    ("duck", "swim", 0.0, "three-quarter", {}, "Mallard"),
    ("penguin", "stand", 0.0, "three-quarter", {}, "Penguin"),
    ("eagle", "fly", 0.15, "side", {}, "Bald eagle"),
    ("clownfish", "swim", 0.2, "side", {}, "Clownfish"),
    ("dolphin", "swim", 0.4, "side", {}, "Dolphin"),
    ("turtle", "swim", 0.3, "three-quarter", {}, "Sea turtle"),
    ("octopus", "stand", 0.2, "three-quarter", {}, "Octopus"),
    ("butterfly", "fly", 0.75, "side", {}, "Monarch"),
    ("frog", "stand", 0.0, "three-quarter", {}, "Tree frog"),
]


def render(width=1920, height=1600, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1600.0
    cv = Canvas(W, H)
    set_light(direction=(-0.65, -0.75), elevation=40)

    log("paper")
    cv.gradient(PAPER)
    grain = fbm2d(H, W, 3 * S + 1, 3, seed=seed) * 0.5 + fbm2d(H, W, 160 * S, 3, seed=seed + 1) * 0.5
    cv.paint("#d8c8a4", np.clip(grain * 0.18, 0, 1).astype(np.float32), mode="multiply")

    log("title")
    top = 0.085
    cv.text(TITLE, (W * 0.5, H * 0.045), size=54 * S, color=INK, font="serif", anchor="mm")
    cv.text(SUBTITLE, (W * 0.5, H * 0.075), size=24 * S, color="#5e4e3a", font="serif", italic=True, anchor="mm")

    log("grid")
    X, Y = cv.xy()
    x0, x1, y0, y1 = 0.03, 0.97, top + 0.01, 0.985
    cw, ch = (x1 - x0) / COLS, (y1 - y0) / ROWS
    lines = np.zeros((H, W), np.float32)
    for c in range(1, COLS):
        u = (x0 + c * cw) * W
        lines = np.maximum(lines, np.clip(1 - np.abs(X - u) / (0.8 * S + 0.4), 0, 1) * ((Y > y0 * H) & (Y < y1 * H)))
    for r in range(1, ROWS):
        v = (y0 + r * ch) * H
        lines = np.maximum(lines, np.clip(1 - np.abs(Y - v) / (0.8 * S + 0.4), 0, 1) * ((X > x0 * W) & (X < x1 * W)))
    cv.paint(RULE, lines * 0.8)

    for i, (sp, pose, phase, view, opt, label) in enumerate(SHEET[:COLS * ROWS]):
        r, c = divmod(i, COLS)
        u0, v0 = x0 + c * cw, y0 + r * ch
        log(label)
        cr = Creature(sp, view=view, style=STYLE, seed=seed + i, **opt).pose(pose, phase)
        pad_x, pad_top, pad_bot = cw * 0.1, ch * 0.1, ch * 0.2
        cr.draw(cv, fit=(u0 + pad_x, v0 + pad_top, u0 + cw - pad_x, v0 + ch - pad_bot))
        cv.text(label, (W * (u0 + cw / 2), H * (v0 + ch - ch * 0.08)), size=24 * S, color=INK, font="serif",
                anchor="mm")
        cv.text(f"{i + 1:02d}", (W * (u0 + 0.006), H * (v0 + 0.012)), size=17 * S, color="#7a6a52", font="serif",
                anchor="la")

    log("post")
    return cv
