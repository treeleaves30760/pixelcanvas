# -*- coding: utf-8 -*-
"""Sticker sheet of nine cute animals drawn freehand, one shared construction so they match.

Plan: pastel diagonal gradient with confetti -> 3 x 3 grid; every animal in its own frame (1 unit = 0.15 of the
height), slightly rotated, built from the same parts: head ellipse at (0, -0.12), body below, the same eyes,
blush, mouth and outline colour -> each wrapped in sticker() (ink line + white die-cut + soft shadow) with a
pill label inside the cut. Light from the upper left.
"""
import numpy as np

from pixelcanvas import Canvas, manifest
from pixelcanvas.illustrate import Sketch

NAME = "illustration_stickers"
KIND = "design"
PROMPT = ("A sticker sheet of nine cute animals — panda, fox, penguin, owl, whale, rabbit, cat, dog, frog — each "
          "with a white die-cut border and a tiny label, on a pastel background.")
DEFAULT_SIZE = (1600, 1600)
INK, BLUSH = "#3d3350", "#ff8fa8"


def part(sk, m, color, **kw):
    return sk.paint(m, color, shade=0.32, highlight=0.35, **kw)


def face(sk, y=-0.06, gap=0.16, r=0.055, mouth=0.1, eye=INK):
    for s in (-1, 1):
        sk.paint(sk.circle(s * gap, y, r), eye)
        sk.paint(sk.circle(s * gap - r * 0.3, y - r * 0.35, r * 0.38), "#ffffff")
        sk.spot(s * (gap + 0.09), y + 0.11, 0.075, 0.045, BLUSH, 0.7)
    if mouth:
        for s in (-1, 1):
            sk.paint(sk.arc(s * 0.03, y + mouth, 0.03, 10, 170, 0.016), INK)


def body(sk, color, belly=None, feet=None):
    for s in (-1, 1):
        part(sk, sk.ellipse(s * 0.17, 0.62, 0.11, 0.06), feet or color)
    part(sk, sk.ellipse(0, 0.38, 0.32, 0.27), color)
    if belly:
        part(sk, sk.ellipse(0, 0.42, 0.19, 0.18), belly)


def panda(sk):
    for s in (-1, 1):
        part(sk, sk.circle(s * 0.3, -0.46, 0.13), "#2f2b3c")
    body(sk, "#ffffff", feet="#2f2b3c")
    for s in (-1, 1):
        part(sk, sk.ellipse(s * 0.3, 0.34, 0.09, 0.15, angle=s * 25), "#2f2b3c")
    part(sk, sk.ellipse(0, -0.12, 0.46, 0.4), "#ffffff")
    for s in (-1, 1):
        sk.paint(sk.ellipse(s * 0.16, -0.07, 0.09, 0.12, angle=s * 30), "#2f2b3c")
    face(sk, eye="#ffffff", r=0.04, mouth=0.13)
    sk.paint(sk.ellipse(0, 0.03, 0.04, 0.026), INK)


def fox(sk):
    tail = sk.blob([(0.2, 0.5), (0.5, 0.2), (0.62, -0.1), (0.48, 0.0), (0.3, 0.3)])
    part(sk, tail, "#f08a3c")
    sk.paint(tail * sk.circle(0.6, -0.08, 0.13), "#fff4e4")
    body(sk, "#f4943f", belly="#fff4e4", feet="#4a3038")
    for s in (-1, 1):
        part(sk, sk.blob([(s * 0.1, -0.4), (s * 0.4, -0.75), (s * 0.42, -0.32)], tension=0.4), "#f4943f")
        sk.paint(sk.blob([(s * 0.22, -0.45), (s * 0.38, -0.66), (s * 0.38, -0.42)], tension=0.4), "#ffd9c4")
    head = part(sk, sk.ellipse(0, -0.12, 0.46, 0.38), "#f4943f")
    for s in (-1, 1):
        sk.paint(sk.ellipse(s * 0.2, 0.1, 0.24, 0.15, angle=s * -20) * head, "#fff4e4")
    face(sk, mouth=0.12)
    sk.paint(sk.ellipse(0, 0.02, 0.045, 0.03), INK)


def penguin(sk):
    for s in (-1, 1):
        part(sk, sk.ellipse(s * 0.17, 0.64, 0.12, 0.05), "#ffa64a")
        part(sk, sk.ellipse(s * 0.43, 0.2, 0.08, 0.22, angle=s * -20), "#36447a")
    part(sk, sk.ellipse(0, 0.05, 0.44, 0.58), "#3f4e82")
    part(sk, sk.ellipse(0, 0.3, 0.3, 0.34), "#ffffff")
    for s in (-1, 1):
        sk.paint(sk.circle(s * 0.15, -0.16, 0.16), "#ffffff")
    face(sk, y=-0.14, mouth=0)
    part(sk, sk.ellipse(0, -0.04, 0.08, 0.05), "#ffb340")
    sk.paint(sk.box(-0.32, 0.08, 0.32, 0.2, radius=0.06), "#ef5b70")
    sk.paint(sk.box(0.12, 0.12, 0.26, 0.42, radius=0.05), "#ef5b70", shade=0.3)


def owl(sk):
    for s in (-1, 1):
        part(sk, sk.blob([(s * 0.22, -0.45), (s * 0.44, -0.72), (s * 0.4, -0.38)], tension=0.4), "#7a4e2a")
        part(sk, sk.ellipse(s * 0.17, 0.66, 0.1, 0.04), "#ffa64a")
    part(sk, sk.ellipse(0, 0.02, 0.46, 0.62), "#b27a46")
    part(sk, sk.ellipse(0, 0.3, 0.3, 0.3), "#f8e9cc")
    for x, y in [(-0.12, 0.2), (0.0, 0.2), (0.12, 0.2), (-0.06, 0.32), (0.06, 0.32), (0, 0.44)]:
        sk.paint(sk.arc(x, y, 0.05, 20, 160, 0.014), "#c9a46a")
    for s in (-1, 1):
        part(sk, sk.circle(s * 0.18, -0.18, 0.17), "#fbf0d8")
        sk.paint(sk.circle(s * 0.18, -0.18, 0.1), "#f6b73c")
    face(sk, y=-0.18, gap=0.18, r=0.06, mouth=0)
    sk.paint(sk.blob([(-0.05, -0.06), (0.05, -0.06), (0, 0.04)], tension=0.3), "#ffa64a")


def whale(sk):
    for x, a in ((-0.12, -30), (0.0, 0), (0.12, 30)):
        sk.paint(sk.stroke([(0, -0.38), (x, -0.55), (x * 2.2, -0.62)], 0.05, taper="end"), "#9fd6ff")
    part(sk, sk.blob([(0.38, 0.18), (0.6, -0.12), (0.72, 0.0), (0.56, 0.14), (0.66, 0.28), (0.5, 0.3)]), "#3f8fe0")
    w = part(sk, sk.blob([(-0.55, 0.1), (-0.4, -0.3), (0.0, -0.4), (0.4, -0.2), (0.5, 0.2), (0.1, 0.45), (-0.4, 0.42)]),
             "#4d9cf0")
    sk.paint(w * sk.ellipse(-0.05, 0.42, 0.55, 0.2), "#e6f3ff")
    with sk.at(-0.18, 0.05, 1):
        face(sk, gap=0.12, mouth=0.12)


def rabbit(sk):
    for s in (-1, 1):
        part(sk, sk.ellipse(s * 0.15, -0.62, 0.1, 0.3, angle=s * 8), "#fbf3ee")
        sk.paint(sk.ellipse(s * 0.15, -0.6, 0.05, 0.22, angle=s * 8), "#ffc4d4")
    body(sk, "#fbf3ee", belly="#ffffff")
    part(sk, sk.ellipse(0, -0.12, 0.42, 0.36), "#fbf3ee")
    face(sk, mouth=0.13)
    sk.paint(sk.ellipse(0, 0.02, 0.04, 0.028), "#ff8fa6")
    sk.paint(sk.box(-0.03, 0.09, 0.03, 0.15, radius=0.01), "#ffffff", line="#d9c4c0", line_width=0.008)


def cat(sk):
    part(sk, sk.stroke([(0.25, 0.55), (0.55, 0.45), (0.6, 0.1), (0.5, -0.05)], 0.1, taper="end"), "#8c95ad")
    body(sk, "#a3acc2", belly="#f4f6fb", feet="#ffffff")
    for s in (-1, 1):
        part(sk, sk.blob([(s * 0.1, -0.38), (s * 0.36, -0.7), (s * 0.42, -0.3)], tension=0.4), "#a3acc2")
        sk.paint(sk.blob([(s * 0.22, -0.42), (s * 0.34, -0.6), (s * 0.37, -0.38)], tension=0.4), "#ffc4d4")
    part(sk, sk.ellipse(0, -0.12, 0.46, 0.37), "#aeb7cc")
    for dx in (-0.08, 0, 0.08):
        sk.paint(sk.stroke([(dx, -0.47), (dx * 0.9, -0.36)], 0.025, taper="end"), "#6a7490")
    face(sk, eye="#3f7a5a", mouth=0.12)
    sk.paint(sk.blob([(-0.04, 0.0), (0.04, 0.0), (0, 0.04)], tension=0.3), "#ff8fa6")
    for s in (-1, 1):
        for dy in (-0.02, 0.03):
            sk.paint(sk.stroke([(s * 0.26, 0.04 + dy), (s * 0.48, 0.02 + dy * 2)], 0.01, taper="end"), "#ffffff")


def dog(sk):
    part(sk, sk.stroke([(0.25, 0.5), (0.48, 0.38), (0.5, 0.15)], 0.09, taper="end"), "#d9a26a")
    body(sk, "#e8b57e", belly="#fff6e8", feet="#ffffff")
    for s in (-1, 1):
        part(sk, sk.ellipse(s * 0.42, -0.08, 0.13, 0.26, angle=s * -15), "#9a6238")
    part(sk, sk.ellipse(0, -0.12, 0.42, 0.37), "#ecbd88")
    sk.paint(sk.ellipse(0.17, -0.12, 0.13, 0.14), "#b9794a")
    part(sk, sk.ellipse(0, 0.08, 0.2, 0.14), "#fff6e8")
    face(sk, mouth=0.15)
    sk.paint(sk.ellipse(0, 0.03, 0.065, 0.045), INK, highlight=0.6)
    sk.paint(sk.blob([(-0.04, 0.16), (0.04, 0.16), (0.03, 0.24), (-0.03, 0.24)]), "#ff7d94")
    sk.paint(sk.box(-0.24, 0.17, 0.24, 0.24, radius=0.035), "#e0525f")
    part(sk, sk.circle(0, 0.29, 0.05), "#ffd35a")


def frog(sk):
    for s in (-1, 1):
        part(sk, sk.ellipse(s * 0.38, 0.55, 0.18, 0.08), "#6cbb5e")
    part(sk, sk.ellipse(0, 0.35, 0.34, 0.27), "#82cc6e")
    part(sk, sk.ellipse(0, 0.42, 0.22, 0.18), "#effacb")
    for s in (-1, 1):
        part(sk, sk.circle(s * 0.22, -0.38, 0.17), "#82cc6e")
        sk.paint(sk.circle(s * 0.22, -0.38, 0.12), "#ffffff")
    part(sk, sk.ellipse(0, -0.08, 0.5, 0.32), "#82cc6e")
    with sk.at(0, -0.32, 1):
        face(sk, gap=0.22, r=0.07, mouth=0)
    sk.paint(sk.arc(0, -0.12, 0.26, 25, 155, 0.022), "#2f6a3a")


ANIMALS = [("panda", panda, "#cdeee0"), ("fox", fox, "#ffd9bf"), ("penguin", penguin, "#cfe6ff"),
           ("owl", owl, "#fff0b8"), ("whale", whale, "#bfe0ff"), ("rabbit", rabbit, "#ffd6e6"),
           ("cat", cat, "#e3d9ff"), ("dog", dog, "#f5dcc0"), ("frog", frog, "#d8f2b8")]
TILT = [-4, 3, -2, 4, -3, 2, -3, 3, -2]


def render(width=1600, height=1600, seed=5, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sk = Sketch(cv)
    rng = np.random.default_rng(seed)

    log("pastel background")
    cv.linear((0, 0), (width, height), [(0, "#ffe9f1"), (0.5, "#e2eeff"), (1, "#e5fce9")])
    for _ in range(110):
        x, y, r = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.003, 0.009)
        c = ["#ffffff", "#ffc9dc", "#c9defc", "#cdeec4", "#fff0a8"][rng.integers(5)]
        if rng.random() < 0.35:
            with sk.at(x, y, r * 2.2, rotate=rng.uniform(0, 90)):
                sk.paint(sk.blob([(0, -1), (0.2, -0.2), (1, 0), (0.2, 0.2), (0, 1), (-0.2, 0.2), (-1, 0), (-0.2, -0.2)],
                                 tension=0.6), c)
        else:
            sk.paint(sk.circle(x, y, r), c, opacity=0.8)

    for i, (name, draw, pill) in enumerate(ANIMALS):
        log(name)
        cx, cy = (i % 3 + 0.5) / 3, (i // 3 + 0.5) / 3 - 0.025
        with sk.at(cx, cy, 0.152, rotate=TILT[i]):
            with sk.sticker(border=0.1, ink=INK, ink_width=0.025, name=name, kind="creature"):
                draw(sk)
                sk.paint(sk.box(-0.3, 0.8, 0.3, 1.0, radius=0.1), pill, line=INK, line_width=0.02)
                sk.text(name, 0, 0.9, 0.12, INK, anchor="mm", bold=True)
    manifest.note(cv, "style", "sticker")
    return cv
