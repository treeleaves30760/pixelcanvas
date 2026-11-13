# -*- coding: utf-8 -*-
"""Looping animation driven by T = 0 -> 1: lanterns rising over a lake (elements)

* Format: 1280x720, FRAMES frames at FPS (a still at T = 0)
* Loop: <every motion is periodic in T: positions use (start + k * T) % 1 with whole k, wobbles sin(2 pi k T)>
* Focal point: <the lantern group rising over the lake, centre>
* Light: <moon upper right, warm lanterns; their reflections in the lake>
* Far / middle / near: <stars, moon, clouds / lanterns / lake and the dark shore>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/elements.md (sky, lanterns, lake, clouds); render.py --animate renders T = 0 -> 1.
import numpy as np

from pixelcanvas import Canvas, clamp01, fbm1d, smoothstep
from pixelcanvas.elements import sky, stars, moon, cumulus, lantern

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_animation"
DEFAULT_SIZE = (1280, 720)
DEFAULT_SEED = 1

T = 0.0                             # animation phase 0..1; T = 1 looks exactly like T = 0
FRAMES, FPS = 48, 24
HORIZON = 0.62
LANTERNS = [(0.45, 0.0, 150), (0.56, 0.2, 130), (0.36, 0.4, 140), (0.64, 0.6, 120), (0.5, 0.8, 110)]   # u, phase, px
CLOUDS = [(0.1, 0.2, 460), (0.45, 0.32, 320)]                                       # start u, v, width px


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    tau = 2 * np.pi * T

    log("sky")                                           # still parts first
    sky(cv, "twilight", horizon=HORIZON)
    stars(cv, density=0.6, milky_way=False, horizon=HORIZON, seed=seed)
    moon(cv, at=(0.84, 0.16), size=36, phase=0.6, seed=seed)
    log("clouds")                                        # drift one screen width per loop, wrap off-screen
    for i, (u0, v, size) in enumerate(CLOUDS):
        cumulus(cv, at=(((u0 + T) % 1) * 1.6 - 0.3, v), size=size, seed=seed + i, lit="#a49ac0", shadow="#2c2b4a")
    log("lanterns")                                      # rise once per loop; grow in, shrink with height
    for i, (u, phase, size) in enumerate(LANTERNS):
        t = (phase + T) % 1
        at = (u + 0.02 * np.sin(tau + 2 * i), HORIZON - 0.08 - 0.75 * t)
        s = size * (1 - 0.6 * t) * smoothstep(0, 0.12, t)
        if s > 2 and at[1] > -0.03:                      # skip lanterns that are too small or off the top
            lantern(cv, at=at, size=s, style="paper", paper="#f08a3a")

    log("lake")                                          # mirror of everything above, ripples move with T
    X, Y = cv.xy()
    hy, S = HORIZON * height, height / 1080
    ripple = 4 * S * np.sin(Y / (5 * S) + tau) * clamp01((Y - hy) / (0.3 * height))
    cv.paint(cv.sample(X + ripple, np.clip(2 * hy - Y, 0, height - 1)) * 0.75, clamp01(Y - hy + 0.5))
    log("shore")
    shore = (0.92 + 0.03 * fbm1d(width, 300 * S, 4, seed=seed + 5)) * height
    cv.paint("#06080e", clamp01(Y - shore[None, :] + 0.5))

    log("post")
    cv.bloom(0.85, 14 * S, 0.3).soft_clip(0.82).grain(0.01, seed)
    return cv
