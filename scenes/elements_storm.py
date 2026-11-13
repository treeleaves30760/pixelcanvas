# -*- coding: utf-8 -*-
"""
Thunderstorm at sea (elements_storm) — built from pixelcanvas.elements.

Composition: low horizon at V 0.56; heavy storm clouds fill the sky in three layers; a branching bolt strikes
the sea right of centre (U 0.62) and lights the cloud base around it; rain curtains hang under the cloud base;
the sea below is a heavy swell with whitecaps that mirrors the flash. Near rain streaks slant to the right.
Order: sky → clouds → lightning (clouds lit) → sea (reflects sky and bolt) → rain sheets + streaks → post.
"""
import numpy as np

from pixelcanvas.color import clamp01, rgb
from pixelcanvas.scenery import Scene, finish
from pixelcanvas.elements import sky, storm_clouds, lightning, ocean, rain, glow

NAME = "elements_storm"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 7

HORIZON = 0.56
BOLT = ((0.6, 0.2), (0.655, HORIZON))
SWELL = 2.2
CHOP = 1.4
FOAM = 0.85
RAIN_ANGLE = 14


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    sc = Scene(width, height, seed, log=log)
    cv, W, H = sc.cv, sc.W, sc.H

    sc.log("sky")
    sky(cv, [(0.0, "#101318"), (0.6, "#262c36"), (1.0, "#4a5260")], horizon=HORIZON)
    glow(cv, at=(0.62, HORIZON - 0.02), radius=420, color="#3a4658", strength=0.6)

    sc.log("storm clouds")
    clouds = storm_clouds(cv, region=(0.0, 0.0, 1.0, 0.5), seed=seed, layers=3, color="#1d2129", light="#6a7484",
                          backlight=(0.62, 0.3), rim="#b8c4d8", scale=420, coverage=0.6)

    sc.log("lightning")
    bolt = lightning(cv, start=BOLT[0], end=BOLT[1], seed=seed + 1, branches=8, width=2.6, clouds=clouds.mask,
                     glow=1.0, flash=0.06, color=(0.78, 0.82, 1.0))

    sc.log("sea")
    ocean(cv, horizon=HORIZON, seed=seed + 2, swell=SWELL, chop=CHOP, foam=FOAM, camera=2.2, wind=25,
          deep="#05090e", body="#121c24", sss="#2a3c44", crest=0.42, reflect=0.75)
    X, Y = sc.X, sc.Y
    bx = BOLT[1][0] * W
    flash = np.exp(-((X - bx) / (W * 0.18)) ** 2) * np.exp(-((Y - HORIZON * H) / (H * 0.12)) ** 2) * (Y > HORIZON * H)
    cv.paint(rgb((0.55, 0.6, 0.8)), flash * 0.35, "add")

    sc.log("rain")
    rain(cv, density=2.0, angle=RAIN_ANGLE, length=80, opacity=0.3, seed=seed + 3, sheets=6,
         sheet_band=(0.34, HORIZON), sheet_color="#262b33", color="#aab4c2")

    sc.log("post")
    finish(sc, bloom=((0.85, 40, 0.25), (1.1, 8, 0.2)), knee=0.8, contrast=0.12, saturation=0.9,
           split=("#2a3446", "#c8d0e0", 0.05), vignette=(0.4, 0.55, 0.7), grain=0.014)
    return cv
