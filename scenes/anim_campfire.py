# -*- coding: utf-8 -*-
"""
Animated night campfire (anim_campfire): a small loop built from pixelcanvas.elements.

    python tools/render.py anim_campfire --animate T 0 1 --loop              -> outputs/anim_campfire.gif + _frames.jpg
    python tools/render.py anim_campfire --animate T 0 1 --loop --frames 36 --fps 15 --scale 0.5

T in 0..1 is one full cycle: every moving part is driven by sin / cos of 2*pi*T (or by a phase that wraps at 1),
so the frame at T = 1 equals the frame at T = 0 and the GIF loops without a jump. --loop leaves the T = 1 frame out.

Composition: fire on the left third (U 0.40, V 0.80), low horizon at V 0.46 with a pine line, moon upper right.
Light: one warm point light (the fire) tints ground, stones and logs; its strength flickers with T.
Order: sky -> stars -> moon -> far shore -> ground -> stones and logs -> firelight -> smoke -> flames -> sparks -> post.
Motion: flame tongues sway and breathe (three flames, each with its own phase), smoke leans left and right and its turbulence drifts (phase=T),
thirty sparks rise on staggered phases and fade out near the top, the firelight pulses.
"""
import math

import numpy as np

from pixelcanvas.color import clamp01, mix, smoothstep
from pixelcanvas.noise import fbm1d
from pixelcanvas.scenery import Scene, pine_mask, scatter_trees, finish
from pixelcanvas.elements import sky, stars, moon, flame, smoke, stone, firewood, illuminate, Light

NAME = "anim_campfire"
DEFAULT_SIZE = (1280, 720)
DEFAULT_SEED = 4
KIND = "painting"

T = 0.0                      # animation phase 0..1 (one full loop); --animate T 0 1 --loop
FIRE = (0.40, 0.82)          # base of the fire as (u, v)
FIRE_SIZE = 340              # flame height in 1080p px
HORIZON = 0.46
SPARKS = 30                  # number of rising sparks
SPARK_RISE = 1.9             # how high a spark climbs, in flame heights
WIND = 0.12                  # steady lean of smoke and sparks (fraction of height)


def _tau(x):
    return 2.0 * math.pi * x


def render(width=1280, height=720, seed=DEFAULT_SEED, log=None):
    sc = Scene(width, height, seed, log=log)
    cv, W, H, S, X, Y = sc.cv, sc.W, sc.H, sc.S, sc.X, sc.Y
    hy = HORIZON * H
    fx, fy = FIRE[0] * W, FIRE[1] * H
    pulse = 1.0 + 0.10 * math.sin(_tau(T) * 2 + 0.6) + 0.06 * math.sin(_tau(T) * 5 + 2.0)

    sc.log("sky")
    sky(cv, [(0.0, "#03050d"), (0.55, "#0a1430"), (0.9, "#1b2a4c"), (1.0, "#24355a")], horizon=HORIZON)
    stars(cv, seed=seed, horizon=HORIZON, band_angle=-32, band_center=(0.45, 0.2), brightness=1.1, twinkle=0.0)
    moon(cv, at=(0.78, 0.26), size=44, phase=0.45, seed=seed, glow=0.7, tilt=-20)

    sc.log("far shore")
    xs = np.arange(W)
    hill = hy - S * 46 * (0.5 + 0.5 * fbm1d(W, S * 420, 4, seed=seed + 1))
    cv.paint("#0a101e", clamp01(Y - hill[None, :] + 0.5) * (Y < hy + 1))
    trees = scatter_trees(sc, 110, (0.0, 1.0), (HORIZON - 0.01, HORIZON + 0.002), height=(22, 56), width_ratio=0.3)
    cv.paint("#060a12", pine_mask(sc, trees) * (Y < hy + 1))

    sc.log("ground")
    t = clamp01((Y - hy) / (H - hy))
    ground = (Y > hy).astype(np.float32)
    cv.paint(mix([(0, "#121722"), (0.5, "#0d0d12"), (1, "#070709")], t), ground)

    sc.log("stones and logs")
    rng = np.random.default_rng(seed + 7)
    light_src = (fx, fy - S * 70, S * 360)
    for (u, v, r) in ((0.07, 0.92, 70), (0.2, 0.97, 90), (0.7, 0.93, 70), (0.86, 0.86, 44), (0.94, 0.98, 110),
                      (0.34, 0.86, 22), (0.47, 0.86, 22), (0.4, 0.89, 20)):
        stone(cv, at=(u, v), size=r, seed=int(rng.integers(1 << 30)), lit_by=light_src, color="#3e3a36")
    firewood(cv, start=(FIRE[0] - 0.09, FIRE[1] + 0.045), end=(FIRE[0] + 0.07, FIRE[1] - 0.005), width=40,
             seed=seed + 8, bark="#3a281a", light=(0.9, -0.4))
    firewood(cv, start=(FIRE[0] + 0.09, FIRE[1] + 0.04), end=(FIRE[0] - 0.06, FIRE[1] - 0.005), width=38,
             seed=seed + 9, bark="#34241a", light=(-0.9, -0.4))

    sc.log("firelight")
    fire_light = Light(fx, fy - S * FIRE_SIZE * 0.3, (1.0, 0.5, 0.2), S * FIRE_SIZE * 1.0, 1.6 * pulse)
    illuminate(cv, fire_light, mask=clamp01(ground), strength=2.2, ambient=0.03)
    glow = np.exp(-((X - fx) / (S * 260)) ** 2 - ((Y - fy) / (S * 80)) ** 2) * (Y > hy)
    cv.paint((1.0, 0.45, 0.12), glow * 0.28 * pulse, "add")

    sc.log("smoke")
    lean = math.sin(_tau(T))
    smoke(cv, at=(FIRE[0] + 0.004 * lean, FIRE[1] - 0.2), size=560, width=36, spread=0.5,
          wind=0.28 + WIND * 0.5 * lean, angle=6 * lean, sway=0.06 + 0.03 * math.cos(_tau(T)), seed=seed + 10,
          color="#9a8a80", shade="#4a4650", opacity=0.42, light=(-0.6, 0.8), curl=1.4 + 0.3 * math.sin(_tau(T) + 1),
          phase=T)

    sc.log("flames")
    base = (FIRE[0], FIRE[1] + 0.006)
    for i, (sz, wd, sd, ph, lean_amp) in enumerate(((FIRE_SIZE, 0.78, seed + 1, 0.0, 0.10),
                                                      (FIRE_SIZE * 0.66, 0.6, seed + 2, 1.9, 0.14),
                                                      (FIRE_SIZE * 0.42, 0.55, seed + 3, 3.7, 0.18))):
        a = _tau(T)
        breathe = 1.0 + 0.07 * math.sin(a * 2 + ph) + 0.04 * math.sin(a * 3 + ph * 1.7)
        flame(cv, at=(base[0] + (i - 1) * 0.012, base[1]), size=sz * breathe, width=sz * wd, seed=sd, tongues=7 - i,
              lean=lean_amp * math.sin(a + ph) + WIND * 0.4, intensity=(1.1 - 0.12 * i) * pulse, glow=0.8 if i == 0 else 0,
              turbulence=1.0 + 0.18 * math.sin(a * 2 + ph * 0.8))

    sc.log("sparks")
    ph = np.random.default_rng(seed + 21)
    n = SPARKS
    phase = ph.random(n)
    sx0 = ph.normal(0, 0.16, n)
    wob = ph.uniform(0.5, 1.5, n)
    wph = ph.uniform(0, _tau(1), n)
    size_k = ph.uniform(0.7, 1.4, n)
    acc = np.zeros((H, W), np.float32)
    top = S * FIRE_SIZE * SPARK_RISE
    for i in range(n):
        p = (T + phase[i]) % 1.0                         # 0 at the fire, 1 at the top: wraps, so the loop is seamless
        x = fx + S * FIRE_SIZE * sx0[i] * (0.4 + p) + math.sin(_tau(p) * wob[i] * 2 + wph[i]) * S * 14 * p + WIND * top * p * p
        y = fy - S * FIRE_SIZE * 0.35 - top * p ** 0.85
        fade = smoothstep(0.0, 0.08, p) * (1 - smoothstep(0.55, 1.0, p))
        sg = S * 2.1 * size_k[i] * (1 - 0.5 * p)
        r = int(sg * 5) + 2
        x0, x1, y0, y1 = int(x) - r, int(x) + r + 1, int(y) - r, int(y) + r + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = X[y0:y1, x0:x1], Y[y0:y1, x0:x1]
        acc[y0:y1, x0:x1] += (np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sg * sg)) * fade).astype(np.float32)
    cv.paint((1.0, 0.62, 0.22), clamp01(acc), "add")
    cv.paint((1.0, 0.85, 0.5), clamp01(acc - 0.6) * 0.8, "add")

    sc.log("post")
    finish(sc, bloom=((0.9, 30, 0.2), (1.2, 7, 0.14)), knee=0.8, contrast=0.08, saturation=1.05,
           split=("#2a3a5e", "#f2b070", 0.06), vignette=(0.35, 0.55, 0.7), grain=0.0)
    return cv
