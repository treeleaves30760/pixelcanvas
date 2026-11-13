# -*- coding: utf-8 -*-
"""
Sea sunset (sea_sunset) — perspective water: world-space back-projection, mirrored reflection, Fresnel, glints on wave faces.

Composition: horizon at V=0.55 (slightly below the middle), the sun centred just above the horizon; the light path pulls the eye to the foreground.
Light: low sun -> warm cloud undersides; only wave faces turned towards the sun light up.
Techniques:
* Perspective: each water pixel gets depth z ~ 1 / (Y - horizon) and lateral wx = (X - centre) * z;
  sampling noise at (wx, z) makes distant waves finer and denser. Fade them out in the distance, otherwise moire appears.
* Reflection = mirror the current canvas about the horizon + ripple displacement; Fresnel: near-total reflection far away, deep water colour showing through up close.
* Glints appear only in the vertical band below the sun and only where the wave slope faces the sun (large -slope); the band widens towards the viewer.
"""
import numpy as np

from pixelcanvas import *

NAME = "sea_sunset"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 9

# ---- tunables (try: python tools/render.py sea_sunset --sweep SUN_X 0.3 0.5 0.7) ------------------------------
HORIZON = 0.55             # horizon height as a fraction of the canvas
SUN_X, SUN_Y = 0.5, 0.45   # sun position (fractions)
SUN_RADIUS = 0.03          # sun disc radius (fraction of H)
SKY = [(0, "#2b4c7e"), (0.35, "#8a6f8e"), (0.5, "#e8a07a"), (0.55, "#ffc58a"), (1, "#ffc58a")]
CLOUD_EDGE = 0.35          # cloud density threshold: lower = more cloud, higher = clearer sky
CLOUD_SEED_OFFSET = -4     # seed offset of the cloud noise
SEA_DEEP, SEA_LIGHT = "#0b1c2e", "#1f3d55"    # water colour in the troughs / on the crests
WAVE_SCALE = 5             # wave frequency across the sea
GLINT = 0.9                # strength of the sun glints
GLINT_WIDTH = 0.12         # width of the glitter path at the bottom (fraction of W)
BLOOM = 0.3                # bloom strength


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()
    hz = HORIZON * H
    sunx, suny = SUN_X * W, SUN_Y * H

    log("sky, sun, clouds")
    cv.gradient(SKY)
    D = np.hypot(X - sunx, Y - suny)
    cv.paint("#ff9a55", np.exp(-(D / (0.35 * H)) ** 2) * 0.3, "add")
    cv.paint("#fff0d0", np.exp(-(D / (0.06 * H)) ** 2) * 0.4, "add")
    cv.paint((1.8, 1.6, 1.3), 1 - smoothstep(SUN_RADIUS * H - 1, SUN_RADIUS * H + 1, D))
    c = fbm2d(H, W, 0.3 * W, 5, seed=seed + CLOUD_SEED_OFFSET, aspect=0.15, mode="billow")
    cv.paint(lerp("#5a4a6a", "#ffb88a", smoothstep(0.1, 0.45, V)),
             smoothstep(CLOUD_EDGE, 0.6, c) * (1 - smoothstep(0.3, 0.45, V)) * 0.8)

    log("sea")
    dy = np.maximum(Y - hz, 0.5)
    z = 0.45 * H / dy                                              # z ~ 1 at the bottom of the frame, growing towards the horizon
    wx = (X - W / 2) / H * z
    far = smoothstep(1.0, 12.0, z)
    waves = fbm2d_at(wx * WAVE_SCALE, z * 22, 5, seed=seed) * (1 - far)
    slope = np.gradient(waves, axis=0) * dy / (0.02 * H)           # wave slope along y (amplified up close)
    refl = cv.sample(X + waves * 0.03 * H / z, 2 * hz - Y + slope * 0.02 * H)
    t = 1 / z                                                      # 0 far -> 1 near
    fres = 0.35 + 0.65 * (1 - t) ** 2
    water = lerp(lerp(SEA_DEEP, SEA_LIGHT, waves * 0.5 + 0.5), refl * 0.9, fres)
    path = np.exp(-((X - sunx) / (0.02 * W + GLINT_WIDTH * W * t)) ** 2)
    spark = smoothstep(0.15, 0.6, -slope) * path
    water = water + rgb((1.4, 1.05, 0.7)) * spark[..., None] * GLINT
    cv.paint(water, smoothstep(hz - 0.5, hz + 0.5, Y))

    log("post")
    cv.bloom(0.85, 0.015 * H, BLOOM).soft_clip(0.85).contrast(0.08).vignette(0.3).grain(0.01, seed=seed)
    return cv
