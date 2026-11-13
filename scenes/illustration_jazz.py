# -*- coding: utf-8 -*-
"""Jazz night poster drawn freehand: a saxophone silhouette across a gold moon, art-deco rays, gold type.

Plan: deep-blue night (gradient + stars) -> deco fan of thin gold rays -> moon (gold gradient, glow, two rings) ->
saxophone silhouette (one variable-width stroke for the body + bell + mouthpiece) with gold keys and a rim light ->
notes -> title BLUE / HOUR in gold serif with letter spacing -> date, venue, line-up between gold rules -> frame.
Palette: navy #070d2a / #13306e, gold #e8b852 / #ffe39a, ice #e2ecff. Light: the moon behind the sax.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch, grade

NAME = "illustration_jazz"
KIND = "design"
PROMPT = ("A poster for a jazz night called 'Blue Hour' on Friday 21:00 at The Velvet Room, with the line-up "
          "'Ana Reyes Trio · Kofi & The Lanterns · DJ Marlowe', a saxophone silhouette and a deep blue and gold palette.")
DEFAULT_SIZE = (1080, 1350)
NAVY, BLUE, GOLD, GOLD2, ICE, INK = "#070d2a", "#163a80", "#e8b852", "#ffe39a", "#e2ecff", "#050a1f"
MOON = (0.5, 0.5, 0.2)                        # centre x, y, radius (fraction of height)


def render(width=1080, height=1350, seed=3, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sk = Sketch(cv, light=(0.0, -1.0))
    mx, my, mr = MOON
    rng = np.random.default_rng(seed)

    log("night sky")
    cv.gradient([(0, "#050a22"), (0.5, BLUE), (1, NAVY)])
    cv.glow(mx * width, my * height, 0.3 * height, "#3f6fc0", 0.35)
    for _ in range(90):
        x, y = rng.uniform(0.04, 0.96), rng.uniform(0.04, 0.72)
        if np.hypot((x - mx) * width, (y - my) * height) > (mr + 0.06) * height:
            sk.spot(x, y, rng.uniform(0.0015, 0.004), color=ICE, opacity=rng.uniform(0.5, 1))

    log("deco rays")
    for a in np.linspace(200, 340, 15):
        t = np.radians(a)
        p0 = (mx + np.cos(t) * (mr + 0.03) * height / width, my + np.sin(t) * (mr + 0.03))
        p1 = (mx + np.cos(t) * 0.62 * height / width, my + np.sin(t) * 0.62)
        sk.paint(sk.stroke([p0, p1], 0.004, taper="end"), GOLD, opacity=0.35)

    log("moon")
    cv.glow(mx * width, my * height, 0.17 * height, "#f0a040", 0.5)
    moon = sk.circle(mx, my, mr)
    sk.paint(moon, [(0, "#ffe7a4"), (0.6, "#f3b85a"), (1, "#d9863a")], name="moon")
    for k, w in ((0.025, 0.003), (0.05, 0.0015)):
        sk.paint(sk.outline(sk.circle(mx, my, mr + k), w), GOLD, opacity=0.9)

    log("saxophone")
    with sk.at(0.5, 0.52, 0.38, rotate=-10):
        spine = [(-0.4, -0.58), (-0.24, -0.68), (-0.04, -0.66), (0.08, -0.52), (0.13, -0.25), (0.18, 0.1),
                 (0.19, 0.38), (0.12, 0.56), (-0.04, 0.63), (-0.19, 0.55), (-0.25, 0.36), (-0.27, 0.2), (-0.29, 0.06)]
        widths = [0.025, 0.035, 0.045, 0.06, 0.08, 0.1, 0.12, 0.13, 0.14, 0.15, 0.17, 0.22, 0.36]
        sax = sk.stroke(spine, widths, cap="flat")
        sk.paint(sax, INK, name="saxophone")
        sk.paint(sk.stroke([(0.09, -0.48), (0.17, -0.12), (0.21, 0.2), (0.19, 0.44)], 0.012, taper="both"),
                 "#5d86d6", opacity=0.8)                                       # moonlit rim
        sk.paint(sk.outline(sk.ellipse(-0.29, 0.06, 0.18, 0.04, angle=-4), 0.012), GOLD)
        for x, y in [(0.07, -0.4), (0.1, -0.26), (0.12, -0.12), (0.14, 0.02), (0.15, 0.16), (0.14, 0.3)]:
            sk.paint(sk.outline(sk.circle(x + 0.015, y, 0.028), 0.01), GOLD)
            sk.paint(sk.circle(x + 0.015, y, 0.011), GOLD2)

    log("notes")
    for x, y, s, c in [(0.18, 0.4, 0.06, GOLD), (0.84, 0.36, 0.05, GOLD), (0.86, 0.62, 0.045, ICE), (0.14, 0.66, 0.04, ICE)]:
        with sk.at(x, y, s, rotate=-12):
            sk.paint(np.maximum(sk.ellipse(0, 0.6, 0.38, 0.27, angle=-20), sk.box(0.26, -0.9, 0.36, 0.6)), c)
            sk.paint(sk.stroke([(0.31, -0.9), (0.7, -0.5), (0.62, -0.1)], 0.12, taper="end"), c)

    log("title")
    sk.text("LIVE JAZZ NIGHT", 0.5, 0.075, 0.022, GOLD2, anchor="mm", bold=True, tracking=0.45)
    sk.text("Blue Hour", 0.5, 0.19, 0.1, ICE, anchor="mm", font="serif", bold=True, case="upper", tracking=0.06,
            gradient=[(0, GOLD2), (1, GOLD)], shadow=dict(dx=0, dy=6, blur=8, color="#000000", opacity=0.6))

    log("details")
    for y in (0.775, 0.87):
        sk.paint(sk.box(0.14, y, 0.86, y + 0.002), GOLD)
    sk.text("Friday  ·  21:00", 0.5, 0.808, 0.04, GOLD2, anchor="mm", font="serif", bold=True, case="upper", tracking=0.12)
    sk.text("The Velvet Room", 0.5, 0.846, 0.026, ICE, anchor="mm", font="serif", case="upper", tracking=0.3)
    sk.text("Ana Reyes Trio · Kofi & The Lanterns · DJ Marlowe", 0.5, 0.92, 0.021, "#c4d2f2", anchor="mm",
            bold=True, case="upper", tracking=0.08)

    log("frame")
    sk.paint(sk.outline(sk.box(0.03, 0.025, 0.97, 0.975), 0.003, "inside"), GOLD)
    sk.paint(sk.outline(sk.box(0.042, 0.035, 0.958, 0.965), 0.0012, "inside"), "#8a7040")

    log("post")
    grade(cv, "warm night", 0.6)
    cv.grain(0.012, seed)
    return cv
