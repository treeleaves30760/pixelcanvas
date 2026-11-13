# -*- coding: utf-8 -*-
"""
Character line-up (figure_cast) — eight people from pixelcanvas.figure in different poses, builds and all four
rendering styles, each labelled with its preset, pose and style.

Composition: a warm studio backdrop with a soft floor; the cast stands on one ground line at v = 0.83 with a
little depth stagger, light from the upper left (one direction for every shadow). Labels sit under the feet.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas.form import Light, set_light
from pixelcanvas.figure import person

NAME = "figure_cast"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 3

FLOOR = 0.83
BACK_TOP, BACK_BOTTOM = "#f3e9da", "#e6d6c0"
FLOOR_COLOR = "#dcc8ad"
INK = "#3a3040"
HEIGHT = 0.64

CAST = [   # preset, pose, style, view, facing, height factor, overrides
    ("office_worker", "talk", "flat", "three_quarter", 1, 1.0, {}),
    ("chef", "hands_on_hips", "soft", "front", 1, 1.0, {}),
    ("runner", "run", "flat", "side", 1, 0.98, {}),
    ("kid", "cheer", "anime", "front", 1, 0.72, {}),
    ("musician", "guitar", "line", "front", 1, 1.0, {}),
    ("grandma", "hold", "soft", "three_quarter", -1, 1.0, {}),
    ("doctor", "arms_crossed", "anime", "three_quarter", -1, 1.0, {}),
    ("astronaut", "wave", "line", "front", -1, 1.0, {}),
]


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080
    cv = Canvas(W, H)
    set_light(Light(direction=(-0.75, -0.6), elevation=40, color="#fff1df", ambient=0.4, shadow="#4a3640",
                    front=0.35))

    log("backdrop")
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top, bot = np.array([0.953, 0.914, 0.855]), np.array([0.902, 0.84, 0.753])
    back = top * (1 - y) + bot * y
    cv.paint(np.broadcast_to(back, (H, W, 3)).astype(np.float32))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    floor = np.clip((yy / H - (FLOOR - 0.1)) / 0.12, 0, 1)
    cv.paint(FLOOR_COLOR, floor * 0.85)
    glow = np.exp(-(((xx / W - 0.42) / 0.45) ** 2 + ((yy / H - 0.45) / 0.5) ** 2))
    cv.paint("#fff6ea", glow * 0.35)

    n = len(CAST)
    for i, (preset, pose, style, view, facing, hf, over) in enumerate(CAST):
        u = (i + 0.5) / n
        stagger = 0.012 if i % 2 else 0.0
        fig = person(preset, pose=pose, style=style, view=view, facing=facing, height=HEIGHT * hf, phase=0.32,
                     **over)
        fig.draw(cv, at=(u, FLOOR + stagger), log=log)

    log("labels")
    for i, (preset, pose, style, view, facing, hf, over) in enumerate(CAST):
        x = (i + 0.5) / n * W
        cv.text(preset.replace("_", " "), (x, (FLOOR + 0.06) * H), int(26 * S), INK, anchor="mt", bold=True)
        cv.text(f"{pose.replace('_', ' ')} · {style}", (x, (FLOOR + 0.06) * H + 34 * S), int(20 * S),
                INK, anchor="mt")

    log("post")
    cv.vignette(0.18, radius=0.7).grain(0.008, seed=seed)
    return cv
