# -*- coding: utf-8 -*-
"""A great wave with Mount Fuji and two fishing boats, built as a ukiyo-e woodblock print (not filtered).

Plan: cream paper with a grey-blue bokashi band at the top -> small Fuji with a snow cap and ink line ->
far sea with rows of wave marks -> the great wave: flat Prussian blue, pale stripes following its back, a foam
crest and curling foam claws in white with ink line, spray -> foreground swell -> two boats with rowers ->
cartouche with vertical title and a red seal -> six-ink palette lock, misregistration and paper grain.
Inks: Prussian #1d3f78, deep #102a5c, pale #9cb8cc, foam #f4ecd8, paper #ecdcb8, ink #1a1d2c, tan #cfa56a, red #b8392c.
"""
import numpy as np

from pixelcanvas import Canvas, manifest
from pixelcanvas.illustrate import Sketch, paper, limit_palette, grade

NAME = "illustration_ukiyoe"
KIND = "painting"
PROMPT = "A great wave with Mount Fuji in the distance and two fishing boats, in the style of a ukiyo-e woodblock print."
DEFAULT_SIZE = (1920, 1080)
BLUE, DEEP, PALE, FOAM, PAPER, INK, TAN, RED = ("#1d3f78", "#102a5c", "#9cb8cc", "#f4ecd8", "#ecdcb8", "#1a1d2c",
                                                "#cfa56a", "#b8392c")
BACK = [(-0.02, 1.0), (0.04, 0.8), (0.11, 0.6), (0.2, 0.42), (0.3, 0.29), (0.42, 0.22), (0.53, 0.23), (0.61, 0.3),
        (0.635, 0.39)]
HOLLOW = [(0.6, 0.37), (0.55, 0.33), (0.49, 0.34), (0.44, 0.4), (0.42, 0.5), (0.44, 0.62), (0.49, 0.72), (0.52, 0.8),
          (0.5, 1.0)]


def claw(sk, x, y, angle, length, width, curl, seed):
    """One curling finger of foam: a tapered stroke whose direction turns as it grows."""
    with sk.at(x, y, 1):
        a = np.radians(angle) + curl * np.linspace(0, 1, 8) ** 2
        p = np.cumsum(np.stack([np.cos(a), np.sin(a)], 1) * length / 8, 0)
        sk.paint(sk.stroke([(0, 0)] + [tuple(q) for q in p], width, taper="end"), FOAM, line=INK, line_width=0.0018)


def boat(sk, x, y, size, tilt, rowers=6):
    with sk.at(x, y, size, rotate=tilt):
        hull = sk.blob([(-1, -0.08), (-0.3, 0.06), (0.6, 0.04), (1.05, -0.24), (0.95, 0.06), (0.2, 0.24), (-0.7, 0.18)])
        for i in range(rowers):
            u = -0.7 + 1.4 * i / (rowers - 1)
            sk.paint(sk.box(u - 0.05, -0.2, u + 0.05, 0.02, radius=0.03), "#3a5f99", line=INK, line_width=0.012)
            sk.paint(sk.circle(u, -0.25, 0.045), "#e0b48c", line=INK, line_width=0.01)
        sk.paint(hull, TAN, line=INK, line_width=0.014, name="fishing boat")
        sk.paint(sk.stroke([(-0.85, 0.04), (0.0, 0.13), (0.8, 0.0)], 0.012), "#8a6238")


def render(width=1920, height=1080, seed=7, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, PAPER)
    sk = Sketch(cv)
    rng = np.random.default_rng(seed)

    log("sky bokashi")
    cv.gradient([(0, "#9fb2c0"), (0.16, "#e3d3bf"), (0.5, "#efdcc0"), (1, PAPER)])
    for x0, y0, x1 in ((0.62, 0.17, 0.95), (0.7, 0.24, 1.0), (0.05, 0.27, 0.25)):
        sk.paint(sk.box(x0, y0, x1, y0 + 0.02, radius=0.008), "#e6cfb6", line="#a89080", line_width=0.0015)

    log("mount fuji")
    fuji = sk.polygon(sk.curve([(0.53, 0.73), (0.63, 0.61), (0.7, 0.5), (0.735, 0.43)]) +
                      sk.curve([(0.765, 0.43), (0.8, 0.5), (0.87, 0.61), (0.98, 0.73)]))
    sk.paint(fuji, [(0, "#4d6f95"), (1, "#a9bccb")], line=INK, line_width=0.002, name="mount fuji")
    cap = fuji * sk.polygon([(0.6, 0.3), (0.9, 0.3), (0.795, 0.5), (0.78, 0.48), (0.765, 0.52), (0.75, 0.485),
                             (0.735, 0.525), (0.72, 0.49), (0.7, 0.51)])
    sk.paint(cap, FOAM, line=INK, line_width=0.0015)

    log("far sea")
    sk.paint(sk.box(0, 0.72, 1, 1), [(0, "#3d6a9a"), (1, DEEP)])
    sk.pattern(sk.box(0, 0.72, 1, 1), "seigaiha", PALE, size=0.075, width=0.0022, rings=4, opacity=0.75)

    log("great wave")
    wave = sk.polygon(sk.curve(BACK) + sk.curve(HOLLOW))
    sk.paint(wave, [(0, BLUE), (1, DEEP)], name="great wave")
    stripes = np.maximum.reduce([sk.stroke([(x + k * 0.008, y + k * 0.016) for x, y in sk.curve(BACK)[::3]],
                                           0.0035 if k % 2 else 0.002, smooth=False) for k in range(1, 40)])
    sk.paint(stripes * wave, PALE, opacity=0.85)
    hollow = sk.polygon(sk.curve(HOLLOW[:7]))
    sk.paint(sk.grow(hollow, 0.02) * wave, "#0b1e48", opacity=0.6)
    sk.paint(sk.outline(wave, 0.003), INK)

    log("foam")
    crest = sk.curve(BACK[3:])
    band = sk.stroke([(x + 0.006, y + 0.012) for x, y in BACK[3:]], [0.008, 0.03, 0.05, 0.06, 0.06, 0.05], cap="flat")
    sk.paint(band * wave, FOAM, line=INK, line_width=0.002)
    for i, (x, y) in enumerate(sk.curve([(x + 0.008, y + 0.045) for x, y in BACK[4:8]])[2::5]):
        claw(sk, x, y - 0.012, 75 + rng.uniform(-10, 10), rng.uniform(0.03, 0.05), 0.014, -1.5, i)
    for i, (x, y) in enumerate(crest[6:-14:4]):
        claw(sk, x, y, -50 + rng.uniform(-10, 10), rng.uniform(0.03, 0.05), 0.012, 1.4, i)
    for i, (x, y) in enumerate(crest[-16::2]):
        claw(sk, x, y + 0.01, 10 + 9 * i + rng.uniform(-8, 8), rng.uniform(0.07, 0.11), 0.022, 2.2, i)
    for _ in range(90):
        x, y = rng.uniform(0.4, 0.72), rng.uniform(0.18, 0.55)
        if wave[int(y * height) - 1, min(int(x * width), width - 1)] < 0.5:
            sk.paint(sk.circle(x, y, rng.uniform(0.002, 0.0055)), FOAM, line=INK, line_width=0.0012)

    log("foreground swell")
    swell = sk.polygon([(0.45, 1.01)] + sk.curve([(0.45, 0.96), (0.62, 0.86), (0.8, 0.83), (1.01, 0.8)]) + [(1.01, 1.01)])
    sk.paint(swell, [(0, BLUE), (1, DEEP)], line=INK, line_width=0.002)
    for k in range(1, 9):
        sk.paint(sk.stroke([(0.5 + k * 0.01, 0.98 + k * 0.004), (0.65, 0.89 + k * 0.016), (1.0, 0.83 + k * 0.02)],
                           0.002) * swell, PALE, opacity=0.8)
    for x in np.arange(0.5, 1.0, 0.03):
        y = float(np.interp(x, [0.45, 0.62, 0.8, 1.0], [0.96, 0.86, 0.83, 0.8]))
        claw(sk, x, y + 0.004, -150 + rng.uniform(-15, 15), rng.uniform(0.03, 0.05), 0.014, 1.8, int(x * 100))

    log("boats")
    boat(sk, 0.37, 0.84, 0.15, -14)
    boat(sk, 0.75, 0.77, 0.12, -6, rowers=5)

    log("cartouche")
    sk.paint(sk.box(0.05, 0.07, 0.15, 0.42), "#e9d49a", line=INK, line_width=0.003)
    sk.paint(sk.outline(sk.box(0.056, 0.08, 0.144, 0.41), 0.0015), RED)
    sk.text("神奈川沖浪裏", 0.115, 0.1, 0.045, INK, vertical=True, anchor="mt", font="cjk-serif", bold=True)
    sk.text("富嶽三十六景", 0.075, 0.11, 0.026, BLUE, vertical=True, anchor="mt", font="cjk-serif")
    sk.paint(sk.box(0.165, 0.07, 0.19, 0.12), RED)
    sk.text("北斎", 0.1775, 0.075, 0.018, FOAM, vertical=True, anchor="mt", font="cjk-serif", bold=True)
    manifest.note(cv, "style", "ukiyo-e")

    log("print finish")
    limit_palette(cv, [BLUE, DEEP, PALE, FOAM, PAPER, INK, TAN, RED, "#4d6f95", "#8fa6b8", "#3a5f99", "#e0b48c",
                       "#e9d49a", "#3d6a9a", "#e3d3bf"], amount=0.5)
    cv.buf[..., 0] = np.roll(cv.buf[..., 0], max(1, width // 1600), axis=1)          # red block misregistered
    paper(cv, grain=0.03, fibres=0.012, blotch=0.03, seed=seed)
    grade(cv, "print", 0.7)
    return cv
