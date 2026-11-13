# -*- coding: utf-8 -*-
"""A cheerful chef with a steaming pot in a small kitchen, drawn freehand as a storybook illustration.

Plan: warm wall with a dot pattern and window light -> three shelves of jars (one jar function, many colours) ->
wooden floor -> the chef in his own frame (1 unit = half the canvas height): shoes, trousers, jacket, sleeves,
pot, mitts, neckerchief, head (ears, face, cheeks, eyes, brows, nose, moustache, smile), toque -> steam -> paper
grain and a cosy grade. Light from the upper left; outlines are a darker tone of each fill.
"""
import numpy as np

from pixelcanvas import Canvas, manifest
from pixelcanvas.illustrate import Sketch, paper, grade

NAME = "illustration_chef"
KIND = "painting"
PROMPT = ("A cheerful chef in a white jacket and tall hat holding a steaming pot in a small kitchen, with shelves "
          "of jars behind. Storybook illustration with soft shading.")
DEFAULT_SIZE = (1080, 1350)
SKIN, CHEEK, HAIR, WHITE, RED, PANTS = "#f6c49c", "#ff8a8a", "#6b3f26", "#fbf9f4", "#d9473e", "#4b5878"
JARS = ["#d9473e", "#f0b232", "#7fb069", "#f3e6c8", "#c97a3a", "#e2866b", "#9bc4a0", "#b47bb5"]
SHELVES = (0.25, 0.45, 0.65)


def jar(sk, x, y, w, h, color):
    sk.paint(sk.box(x, y - h, x + w, y, radius=0.012), [(0, color), (1, color)], shade=0.45, highlight=0.5,
             line=True, line_width=0.002)
    sk.paint(sk.box(x + 0.15 * w, y - 0.62 * h, x + 0.85 * w, y - 0.3 * h, radius=0.004), "#fbf3de")
    sk.paint(sk.box(x + 0.05 * w, y - h - 0.016, x + 0.95 * w, y - h + 0.004, radius=0.005),
             [(0, "#e3c27a"), (1, "#9a7a38")], angle=0)


def chef(sk):
    sk.spot(0, 0.72, 0.5, 0.06, "#5a3216", 0.45)                                        # contact shadow
    for s in (-1, 1):
        leg = sk.paint(sk.box(s * 0.04, 0.42, s * 0.19, 0.69, radius=0.04), PANTS, shade=0.4, line=True)
        sk.pattern(leg, "stripes", "#8f9bb8", size=0.035, width=0.006, angle=90, opacity=0.8)
        sk.paint(sk.ellipse(s * 0.13, 0.7, 0.11, 0.045), "#5a3622", shade=0.4, highlight=0.4, line=True)
    jacket = sk.blob([(-0.1, -0.29), (0.1, -0.29), (0.32, -0.21), (0.4, -0.04), (0.38, 0.25), (0.34, 0.5),
                      (0, 0.53), (-0.34, 0.5), (-0.38, 0.25), (-0.4, -0.04), (-0.32, -0.21)])
    sk.paint(jacket, WHITE, volume=0.75, highlight=0.2, line=True, shadow=True, name="chef")
    sk.paint(sk.stroke([(0, -0.25), (0, 0.5)], 0.008), "#c9d0de")
    for s in (-1, 1):
        for v in (-0.15, -0.03, 0.09):
            sk.paint(sk.circle(s * 0.08, v, 0.02), [(0, "#ffffff"), (1, "#b5bfd2")], angle="radial", line=True)
        sk.paint(sk.stroke([(s * 0.3, -0.18), (s * 0.41, -0.04), (s * 0.41, 0.08), (s * 0.34, 0.15)],
                           [0.13, 0.12, 0.11, 0.1]), WHITE, volume=0.7, line=True)          # sleeve
    with sk.at(0, 0.15, 1):                                                              # the pot
        sk.paint(sk.ellipse(0, -0.1, 0.28, 0.05), "#2d3340")
        sk.paint(sk.ellipse(0, -0.095, 0.25, 0.038), [(0, "#f6c37a"), (1, "#d98a45")], angle="radial")
        for s in (-1, 1):
            sk.paint(sk.arc(s * 0.3, -0.02, 0.06, 90 - s * 180, 90, 0.022), "#4a5262")   # handles
        body = np.maximum(sk.box(-0.28, -0.1, 0.28, 0.12, radius=0.0), sk.ellipse(0, 0.12, 0.28, 0.07))
        body = body * (1 - sk.ellipse(0, -0.1, 0.28, 0.05))
        sk.paint(body, [(0, "#5c6476"), (0.25, "#d8dde6"), (0.4, "#f4f6fa"), (0.7, "#a9b0bf"), (1, "#545c6e")],
                 angle=0, line="#3a404c", name="pot")
        sk.paint(sk.outline(sk.ellipse(0, -0.1, 0.28, 0.05), 0.016), [(0, "#eef1f6"), (1, "#8a92a2")], angle=0)
        for s in (-1, 1):
            sk.paint(sk.ellipse(s * 0.33, -0.01, 0.075, 0.06, angle=s * 20), RED, shade=0.45, highlight=0.4,
                     line=True)                                                          # mitts on the handles
    sk.paint(sk.box(-0.07, -0.33, 0.07, -0.25, radius=0.03), "#e8a97e")                   # neck
    sk.paint(sk.blob([(-0.14, -0.29), (0.14, -0.29), (0.08, -0.19), (0, -0.16), (-0.08, -0.19)]), RED,
             shade=0.35, line=True)
    with sk.at(0, -0.5, 1):                                                               # head
        for s in (-1, 1):
            sk.paint(sk.ellipse(s * 0.2, 0.03, 0.04, 0.05), SKIN, shade=0.3, line=True)
            sk.paint(sk.blob([(s * 0.13, -0.13), (s * 0.2, -0.12), (s * 0.2, -0.02), (s * 0.15, -0.05)]), HAIR)
        face = sk.ellipse(0, 0, 0.2, 0.19)
        sk.paint(face, SKIN, shade=0.45, highlight=0.3, line=True, name="face")
        for s in (-1, 1):
            sk.spot(s * 0.12, 0.07, 0.06, 0.04, CHEEK, 0.75, clip=face)
            sk.paint(sk.ellipse(s * 0.075, -0.02, 0.032, 0.04), "#ffffff")
            sk.paint(sk.ellipse(s * 0.072, -0.01, 0.022, 0.03), [(0, "#6a4020"), (1, "#24160c")])
            sk.paint(sk.circle(s * 0.072 - 0.008, -0.022, 0.009), "#ffffff")
            sk.paint(sk.arc(s * 0.075, -0.02, 0.06, 215, 325, 0.014, taper="both"), HAIR)  # brows
            with sk.at(0, 0, 1, mirror=s < 0):                                            # moustache
                sk.paint(sk.blob([(0, 0.07), (0.06, 0.055), (0.13, 0.085), (0.15, 0.115), (0.08, 0.105), (0, 0.1)]),
                         HAIR, highlight=0.35)
        sk.paint(sk.ellipse(0, 0.05, 0.035, 0.028), "#f0a07c", highlight=0.6)
        mouth = sk.blob([(-0.07, 0.115), (0.07, 0.115), (0.04, 0.16), (0, 0.17), (-0.04, 0.16)])
        sk.paint(mouth, "#8f2d3a")
        sk.spot(0, 0.165, 0.04, 0.022, "#f2808e", 1, clip=mouth)
    with sk.at(0, -0.88, 1):                                                              # toque
        puff = np.maximum.reduce([sk.circle(x, y, r) for x, y, r in
                                  [(-0.13, 0.08, 0.11), (0.13, 0.08, 0.11), (0, -0.05, 0.15), (-0.12, -0.02, 0.11),
                                   (0.12, -0.02, 0.11), (0, 0.08, 0.12)]])
        for x, y, r in [(-0.13, 0.08, 0.11), (0.13, 0.08, 0.11), (0, -0.05, 0.15), (-0.12, -0.02, 0.11),
                        (0.12, -0.02, 0.11), (0, 0.08, 0.12)]:
            sk.paint(sk.circle(x, y, r), WHITE, volume=1.0)                             # each puff rounded
        sk.paint(sk.outline(puff, 0.005), "#6a6470")
        manifest.note(sk.cv, "object", "hat", mask=puff)
        sk.paint(sk.box(-0.21, 0.13, 0.21, 0.25, radius=0.03), WHITE, volume=0.6, line=True)


def render(width=1080, height=1350, seed=4, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sk = Sketch(cv)
    rng = np.random.default_rng(seed)

    log("wall")
    cv.gradient([(0, "#f6d9a6"), (1, "#e3b57a")])
    sk.pattern(sk.box(0, 0, 1, 0.86), "dots", "#d09a5c", size=0.05, width=0.011, opacity=0.5)
    cv.glow(0.15 * width, 0.08 * height, 0.3 * height, "#ffe9b8", 0.18)

    log("shelves of jars")
    for y in SHELVES:
        x = rng.uniform(0.0, 0.03)
        while x < 0.96:
            w, h = rng.uniform(0.06, 0.085), rng.uniform(0.07, 0.11)
            jar(sk, x, y, w, h, JARS[rng.integers(len(JARS))])
            x += w + rng.uniform(0.012, 0.03)
        sk.paint(sk.box(-0.01, y, 1.01, y + 0.018), [(0, "#c18a52"), (1, "#8f5f32")], shadow=dict(offset=(0, 0.02), blur=0.012))

    log("floor")
    floor = sk.paint(sk.box(0, 0.86, 1, 1), [(0, "#d6a86c"), (1, "#a8783f")])
    sk.pattern(floor, "checker", "#e8c48e", size=0.07, angle=45, opacity=0.7)
    sk.paint(sk.box(0, 0.855, 1, 0.865), "#7a5230")

    log("chef")
    with sk.at(0.5, 0.6, 0.5):
        chef(sk)

    log("steam")
    with sk.at(0.5, 0.6, 0.5):
        for x0, bend, top in ((-0.2, -1, -0.3), (0.02, 1, -0.3), (0.2, 1, -0.28)):
            pts = [(x0 + bend * 0.06 * np.sin(t * 5.5) * (0.4 + t) + bend * 0.12 * t, 0.02 + (top - 0.02) * t)
                   for t in np.linspace(0, 1, 7)]
            sk.paint(sk.stroke(pts, 0.05, taper="both"), "#ffffff", line="#b9c6d8", line_width=0.006,
                     opacity=0.85, glow=dict(color="#ffffff", radius=0.04, strength=0.25), name="steam")

    log("post")
    paper(cv, grain=0.025, fibres=0.015, blotch=0.015, seed=seed)
    grade(cv, "cosy", 0.6)
    cv.soft_clip(0.9)
    manifest.note(cv, "style", "storybook")
    return cv
