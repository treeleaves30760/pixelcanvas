# -*- coding: utf-8 -*-
"""
Garden path in the afternoon (flora_garden) — the pixelcanvas.flora showcase.

Composition: eye level at V=0.47 behind a clipped hedge; a flagstone path (cells with mortar, in perspective) runs
from the bottom centre to a gap in the hedge. A large oak frames the left side, a cherry tree in bloom stands behind
the right flower bed, stone steps with potted plants fill the bottom right, rocks and grass tufts sit along the path.
Light: warm afternoon sun from the upper left (LIGHT), every plant casts its ground shadow to the right.
"""
import numpy as np

from pixelcanvas import Canvas, mix, fbm2d, clamp01, smoothstep, lerp
from pixelcanvas.flora import (tree, forest, hedge, flower_bed, potted, topiary, rock, grass, bush, cellular,
                               strokes, pebbles, LIGHT)

NAME = "flora_garden"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 7

HORIZON = 0.47
SKY = [(0.0, "#7fa6cc"), (0.32, "#a9c3da"), (0.47, "#e9dcc0")]
HAZE = "#c9d2cf"
LAWN = [(0.47, "#8a9c5a"), (0.62, "#6e8a3e"), (1.0, "#4a6a2a")]
PATH_TOP, PATH_HALF = 0.56, (0.022, 0.2)          # where the path meets the hedge gap; half width far / near
OAK = dict(at=(0.13, 0.9), height=0.95, params=dict(trunk_h=0.36, first_len=0.28))
CHERRY = dict(at=(0.76, 0.6), height=0.42)


def _path(X, Y, W, H):
    """Path mask and the centre line (perspective)."""
    v = Y / H
    t = np.clip((v - PATH_TOP) / (1 - PATH_TOP), 0, 1)
    cx = (0.51 + 0.035 * np.sin(t * 3.2 + 0.4) - 0.02 * t) * W
    half = (PATH_HALF[0] + (PATH_HALF[1] - PATH_HALF[0]) * t ** 1.15) * W
    m = clamp01(half - np.abs(X - cx) + 0.5) * clamp01((v - PATH_TOP) * H + 0.5)
    return m, cx, half, t


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080
    cv = Canvas(W, H)
    X, Y = cv.xy()
    V = Y / H

    log("sky")
    cv.paint(mix(SKY, V))
    cv.glow(0.08 * W, 0.12 * H, 0.35 * H, "#ffe2b0", 0.35)
    n = fbm2d(H, W, 420 * S, 5, seed=seed + 40, aspect=0.45) * 0.6 + fbm2d(H, W, 90 * S, 3, seed=seed + 41) * 0.25
    band = smoothstep(0.02, 0.12, V) * (1 - smoothstep(0.26, 0.4, V))
    cl = smoothstep(0.02, 0.3, n) * band
    cv.paint(lerp("#c4d2e0", "#fbf4ea", smoothstep(0.1, 0.5, n)), cl * 0.85)

    log("distant trees")
    forest(cv, region=(-0.05, 0.43, 1.05, 0.5), n=46, species=("oak", "maple", "poplar", "birch"),
           height=(0.08, 0.16), seed=seed + 1, haze=(HAZE, 0.62), light=LIGHT)

    log("lawn")
    ground = clamp01((V - HORIZON) * H + 0.5)
    col = mix(LAWN, V)
    depth = np.clip((V - HORIZON) / (1 - HORIZON), 0.02, 1)
    gx = (X / W - 0.5) / depth                                   # ground-plane coordinates
    stripe = 0.5 + 0.5 * np.sign(np.sin(gx * 9.0))
    col = col * (0.95 + 0.06 * stripe * smoothstep(0.5, 0.7, V))[..., None]
    cov, sid = strokes(X, Y, -np.pi / 2 + 0.35 * fbm2d(H, W, 200 * S, 2, seed=seed), length=7 * S + 1,
                       width=1.3 * S + 0.6, seed=seed)
    col = col * (0.95 + 0.1 * (cov * (sid - 0.3)))[..., None]
    col = lerp(col, HAZE, (1 - smoothstep(HORIZON, 0.6, V)) * 0.45)
    cv.paint(col, ground)

    log("path")
    pm, pcx, phalf, pt = _path(X, Y, W, H)
    z = 1.0 / np.maximum(V - HORIZON, 0.01)                       # perspective flagstones: cells with mortar
    f1, f2, cid = cellular(H, W, 1.0, seed=seed + 3, X=(X - pcx) / W * z * 13, Y=z * 1.9)
    mortar = smoothstep(0.0, 0.09, f2 - f1)
    stone = mix([(0, "#9a8f7e"), (0.5, "#b9ad98"), (1, "#d2c6ae")], cid)
    lit = 0.92 + 0.1 * fbm2d(H, W, 30 * S, 3, seed=seed + 4)
    pcol = lerp(mix([(0, "#5a5244"), (1, "#6e6454")], cid), stone * lit[..., None], mortar)
    edge_sh = 1 - 0.25 * smoothstep(0.75, 1.0, np.abs(X - pcx) / np.maximum(phalf, 1))
    pcol = lerp(pcol * edge_sh[..., None], HAZE, (1 - smoothstep(PATH_TOP, 0.75, V)) * 0.35)
    cv.paint(pcol, pm)

    log("hedge")
    hedge(cv, box=(-0.02, 0.46, 0.475, 0.575), top=0.03, side=0.012, seed=seed + 5, light=LIGHT, shadow=0.35)
    hedge(cv, box=(0.555, 0.46, 1.02, 0.575), top=0.03, side=-0.012, seed=seed + 6, light=LIGHT, shadow=0.35)

    log("cherry tree")
    tree(cv, "cherry", seed=seed + 7, light=LIGHT, shadow=0.55, **CHERRY)

    log("flower beds")
    for (u0, v0, u1, v1) in ((0.285, 0.605, 0.452, 0.77), (0.585, 0.615, 0.805, 0.772)):
        e = fbm2d(H, W, 40 * S, 3, seed=seed + 30) * 0.06
        q = (np.abs((X / W - (u0 + u1) / 2) / ((u1 - u0) / 2)) ** 4 + np.abs((V - (v0 + v1) / 2) / ((v1 - v0) / 2)) ** 4)
        bed = clamp01((1 + e - q) * 40 * S)
        soil = mix([(0, "#3a2c20"), (1, "#5a4632")], fbm2d(H, W, 6 * S, 3, seed=seed + 31) * 0.5 + 0.5)
        cv.paint(soil, bed)
    flower_bed(cv, region=(0.29, 0.61, 0.45, 0.76), kinds=("tulip", "daisy", "tulip", "poppy", "daisy"), n=60, seed=seed + 8,
               height=(0.05, 0.1), light=LIGHT)
    flower_bed(cv, region=(0.59, 0.62, 0.8, 0.76), kinds=("lavender", "rose", "cosmos", "lavender", "peony"), n=64,
               seed=seed + 9, height=(0.05, 0.1), light=LIGHT)
    bush(cv, at=(0.9, 0.66), width=0.2, seed=seed + 10, flowers="#f2f0f6", light=LIGHT, shadow=0.4)

    log("oak")
    tree(cv, "oak", seed=seed + 11, light=LIGHT, shadow=0.7, fullness=1.15, **OAK)

    log("rocks")
    rock(cv, at=(0.39, 0.83), size=0.09, seed=seed + 12, moss=0.5, light=LIGHT, shadow=True)
    rock(cv, at=(0.355, 0.86), size=0.05, seed=seed + 13, light=LIGHT, shadow=True)
    pebbles(cv, region=(0.4, 0.8, 0.47, 0.88), n=26, seed=seed + 14, size=(0.008, 0.018), light=LIGHT)

    log("steps")
    _steps(cv, X, Y, W, H, S)
    potted(cv, "flowering", at=(0.71, 0.926), height=0.13, seed=seed + 15, light=LIGHT, shadow=0.4)
    topiary(cv, at=(0.81, 0.862), height=0.17, shape="ball", seed=seed + 16, light=LIGHT, shadow=0.4)
    potted(cv, "succulent", at=(0.925, 0.792), height=0.07, seed=seed + 17, light=LIGHT, shadow=0.4)
    potted(cv, "fern", at=(0.955, 0.926), height=0.15, seed=seed + 18, light=LIGHT, shadow=0.4)

    log("grass")
    grass(cv, region=(0.0, 0.9, 0.42, 1.0), tufts=26, height=0.06, width=0.05, blades=22, seed=seed + 19, wind=0.15)
    grass(cv, at=(0.6, 0.99), height=0.07, width=0.06, blades=30, seed=seed + 20, wind=0.15)
    for i, (u, v, h) in enumerate(((0.455, 0.7, 0.03), (0.565, 0.66, 0.025), (0.44, 0.79, 0.04), (0.63, 0.86, 0.045))):
        grass(cv, at=(u, v), height=h, width=h, blades=16, seed=seed + 21 + i, wind=0.15)

    log("post")
    cv.glow(0.05 * W, 0.08 * H, 0.5 * H, "#ffcf8a", 0.12, mode="screen")
    cv.vignette(0.22, 0.7, 0.6)
    cv.soft_clip(0.9)
    return cv


def _steps(cv, X, Y, W, H, S):
    """Three stone steps rising towards the right (box faces: lit tops, shaded risers)."""
    steps = [(0.66, 0.905, 1.02, 0.96), (0.76, 0.835, 1.02, 0.905), (0.86, 0.765, 1.02, 0.835)]
    f1, f2, cid = cellular(H, W, 46 * S, seed=31)
    joint = smoothstep(0.0, 2.0 * S, f2 - f1)
    tex = 0.93 + 0.1 * fbm2d(H, W, 8 * S, 3, seed=32)
    for (u0, v0, u1, v1) in steps:
        top_h = (v1 - v0) * 0.5
        topm = clamp01(X - u0 * W + 0.5) * clamp01(u1 * W - X + 0.5) * clamp01(Y - v0 * H + 0.5) * \
            clamp01((v0 + top_h) * H - Y + 0.5)
        riser = clamp01(X - u0 * W + 0.5) * clamp01(u1 * W - X + 0.5) * clamp01(Y - (v0 + top_h) * H + 0.5) * \
            clamp01(v1 * H - Y + 0.5)
        stone = mix([(0, "#8e8576"), (1, "#b8ad98")], cid) * (tex * joint * 0.12 + tex * 0.88)[..., None]
        cv.paint(stone * 1.12, topm)
        ao = 1 - 0.35 * smoothstep(0.0, 1.0, (Y / H - (v0 + top_h)) / max(v1 - v0 - top_h, 1e-3))
        cv.paint(stone * 0.62 * ao[..., None], riser)
        lip = clamp01(1.6 * S - np.abs(Y - (v0 + top_h) * H)) * clamp01(X - u0 * W) * clamp01(u1 * W - X)
        cv.paint("#e6dcc6", lip * 0.7)
