# -*- coding: utf-8 -*-
"""
Campfire by a lake at night (elements_campfire) — built from pixelcanvas.elements.

Composition: horizon at V 0.52 (far shore with a pine line), the lake from 0.52 to the near shore (~0.70),
dark near ground below. The fire sits on the left third (U 0.36, V 0.83) on the near shore; the moon on the
upper right third with the Milky Way crossing the sky, both mirrored in the lake.
Light: one warm point light (the fire) tints the ground, rocks, the log seat and the shore; cool moonlight
everywhere else. Order: sky → stars → moon → far shore → lake (reflects all of the above) → near ground →
rocks and seat → campfire → smoke → firelight → post.
"""
import numpy as np

from pixelcanvas.color import clamp01, mix, rgb, smoothstep
from pixelcanvas.noise import fbm1d, fbm2d
from pixelcanvas.scenery import Scene, pine_mask, scatter_trees, finish
from pixelcanvas.elements import (sky, stars, moon, lake, campfire, smoke, illuminate, fog, Light, stone,
                                  firewood)

NAME = "elements_campfire"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 4

HORIZON = 0.5
SHORE = 0.75
FIRE = (0.36, 0.88)
FIRE_SIZE = 440
MOON = (0.76, 0.33)
MOON_PHASE = 0.42


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    sc = Scene(width, height, seed, log=log)
    cv, W, H, S, X, Y = sc.cv, sc.W, sc.H, sc.S, sc.X, sc.Y

    sc.log("sky")
    sky(cv, [(0.0, "#03050d"), (0.55, "#0a1430"), (0.9, "#1b2a4c"), (1.0, "#24355a")], horizon=HORIZON)

    sc.log("stars")
    stars(cv, seed=seed, horizon=HORIZON, band_angle=-32, band_center=(0.45, 0.22), brightness=1.1)

    sc.log("moon")
    moon(cv, at=MOON, size=46, phase=MOON_PHASE, seed=seed, glow=0.7, tilt=-20)

    sc.log("far shore")
    hy = HORIZON * H
    hill = hy - sc.px(70) * (0.5 + 0.5 * fbm1d(W, sc.px(520), 4, seed=seed + 1)) - sc.px(40) * np.exp(-((np.arange(W) / W - 0.15) / 0.2) ** 2)
    far = clamp01(Y - hill[None, :] + 0.5) * (Y < hy + 1)
    cv.paint(mix([(0, "#0c1424"), (1, "#070b14")], clamp01((Y - hill.min()) / max(hy - hill.min(), 1))), far)
    trees = scatter_trees(sc, 160, (0.0, 1.0), (HORIZON - 0.012, HORIZON + 0.002), height=(26, 70), width_ratio=0.3)
    pm = pine_mask(sc, trees) * (Y < hy + 1)
    cv.paint("#060a12", pm)
    fog(cv, v_range=(HORIZON - 0.06, HORIZON + 0.02), color="#2a3858", density=0.35, layers=2, seed=seed + 2)

    sc.log("lake")
    shore_y = SHORE * H + sc.px(22) * fbm1d(W, sc.px(260), 4, seed=seed + 3) - sc.px(30) * np.sin(np.arange(W) / W * 2.2)
    water = clamp01(shore_y[None, :] - Y + 0.5)
    lake(cv, horizon=HORIZON, color="#050a14", ripple=1.6, streak=2.5, seed=seed + 4, mask=water, tint="#c8d4e6")

    sc.log("near ground")
    ground = 1 - water
    ground = ground * (Y > hy)
    t = clamp01((Y - shore_y[None, :]) / (H - shore_y.min()))
    soil = fbm2d(H, W, sc.px(90), 5, seed=seed + 5, aspect=0.4)
    gcol = mix([(0, "#15141a"), (0.5, "#0e0d0d"), (1, "#080707")], t) * (1 + 0.25 * soil)[..., None]
    pebbles = smoothstep(0.55, 0.8, fbm2d(H, W, sc.px(9), 3, seed=seed + 6, aspect=0.5, mode="billow"))
    gcol = gcol * (1 + 0.35 * pebbles * (1 - t))[..., None]
    cv.paint(gcol, ground)
    edge = np.exp(-((Y - shore_y[None, :]) / sc.px(4)) ** 2) * (Y > shore_y[None, :] - sc.px(3))
    cv.paint("#3a4458", edge * 0.35)

    fx, fy = FIRE[0] * W, FIRE[1] * H
    fire_light = Light(fx, fy - sc.px(FIRE_SIZE * 0.3), (1.0, 0.5, 0.2), sc.px(FIRE_SIZE * 0.9), 1.6)

    sc.log("rocks")
    rng = np.random.default_rng(seed + 7)
    rocks = np.zeros((H, W), np.float32)
    for (u, v, r) in ((0.06, 0.9, 80), (0.14, 0.93, 95), (0.68, 0.95, 80), (0.78, 0.84, 45), (0.88, 0.95, 95),
                      (0.57, 0.8, 26), (0.22, 0.79, 22)):
        stone(cv, at=(u, v), size=r, seed=int(rng.integers(1 << 30)), lit_by=(fx, fy - sc.px(80), sc.px(380)),
              color="#3e3a36")

    sc.log("log seat")
    firewood(cv, start=((fx + sc.px(360)) / W, (fy + sc.px(50)) / H), end=((fx + sc.px(640)) / W, (fy + sc.px(5)) / H),
             width=60, seed=seed + 8, bark="#3a281a", light=(-0.9, -0.4))

    sc.log("firelight")
    lit_mask = clamp01(ground + 0.0)
    illuminate(cv, fire_light, mask=lit_mask, strength=2.2, ambient=0.03)
    # warm reflection of the fire on the near water
    win_y = shore_y.mean()
    refl = np.exp(-((X - fx) / sc.px(90)) ** 2) * np.exp(-((Y - win_y + sc.px(30)) / sc.px(60)) ** 2) * water
    rip = smoothstep(0.1, 0.6, fbm2d(H, W, sc.px(30), 3, seed=seed + 9, aspect=0.1))
    cv.paint(rgb((1.0, 0.5, 0.18)), refl * rip * 0.6, "add")

    sc.log("smoke")
    smoke(cv, at=(FIRE[0] + 0.005, FIRE[1] - 0.22), size=620, width=40, spread=0.5, wind=0.3, seed=seed + 10,
          color="#9a8a80", shade="#4a4650", opacity=0.45, light=(-0.6, 0.8), curl=1.6)

    sc.log("campfire")
    campfire(cv, at=FIRE, size=FIRE_SIZE, seed=seed + 11, embers_n=55)

    sc.log("post")
    finish(sc, bloom=((0.9, 40, 0.22), (1.2, 8, 0.15)), knee=0.8, contrast=0.08, saturation=1.05,
           split=("#2a3a5e", "#f2b070", 0.06), vignette=(0.35, 0.55, 0.7), grain=0.012)
    return cv
