# -*- coding: utf-8 -*-
"""Studio product shot of white wireless earbuds in an open charging case (solid)

* Format: 1920x1080
* Focal point: the open case just left of centre, both buds seated in it, the lid open behind
* Light: soft key from the upper left, a warm glow on the sweep behind, gentle rim
* Palette: pale pink sweep, glossy white, one small green status light
"""
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting, charging_case, earbuds

PROMPT = ("A product shot of a pair of white wireless earbuds in an open charging case on a pale pink seamless "
          "backdrop, soft studio lighting with gentle shadows.")
NAME = "solid_earbuds"
KIND = "painting"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

BACKDROP = "#efc4cf"
LID_OPEN = 100           # degrees
SOFTNESS = 0.85
CAMERA = dict(azimuth=18, elevation=22, fill=0.62)
FOCUS_BLUR = 6


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    lit = Lighting("studio", sun=(-50, 40), softness=SOFTNESS, rim_power=0.5, ambient=0.5)
    lit.point((0.0, 0.12, -0.2), color="#ffe0d0", power=0.8, radius=0.2, shadow=False)
    sc = Scene(lighting=lit, ground=None)
    sc.backdrop(BACKDROP, distance=0.25, radius=0.2)

    sc.add(charging_case(open=LID_OPEN).rotate_y(-12), "charging case")
    sc.camera = Camera.frame(sc, **CAMERA)
    res = sc.render(cv, log=log)

    log("post")
    res.depth_of_field(cv, focus="charging case", blur=FOCUS_BLUR)
    cv.vignette(0.15).grain(0.006, seed=seed)
    return cv
