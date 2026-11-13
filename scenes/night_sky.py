# -*- coding: utf-8 -*-
"""
Night sky (night_sky) — low-level API only: gradients, domain warp, scattered points, glow, 1D ridge lines.

Composition: very low horizon (hills at V ~ 0.76-0.9), sky fills 3/4; the moon sits on the upper-right third, the Milky Way runs from upper left to lower right.
Light: weak moonlight -> near-black silhouettes; only the sky glow separates far and near hills.
Techniques:
* Milky Way = diagonal-band Gaussian (distance to the centre line) x warped noise; a second noise adds dark dust lanes.
* Stars = random positions + power-law brightness (most dim, a few bright) -> np.add.at into a mask -> slight blur, then add.
* Keep the moon disc <= 1, otherwise its markings vanish after soft_clip.
"""
import numpy as np

from pixelcanvas import *

NAME = "night_sky"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080                                                   # pixel constants are relative to 1080p
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()

    log("sky & milky way")
    cv.gradient([(0, "#02040c"), (0.55, "#0b1838"), (0.78, "#23385f"), (1, "#3a4a6a")])
    d = (V - 0.15 - 0.55 * U) / 0.16                               # distance to the Milky Way centre line
    band = np.exp(-d * d)
    dust = warp2d(H, W, 0.12 * H, 0.8, seed=seed + 2) * 0.5 + 0.5
    cv.paint("#8ea6d8", band * smoothstep(0.35, 0.8, dust) * 0.45, "add")
    cv.paint("#05070f", band * smoothstep(0.55, 0.75, fbm2d(H, W, 0.05 * H, 5, seed=seed + 3)) * 0.6)

    log("stars")
    rng = np.random.default_rng(seed)
    n = int(1620 * W / H)                                          # star count depends only on the aspect ratio: draft and full size show the same sky
    sx, sy = rng.uniform(0, W, n), rng.uniform(0, H * 0.8, n)
    b = rng.random(n) ** 6 * 1.6 + 0.15                            # power-law: most dim, a few bright
    b *= (1 - sy / (H * 0.8)) ** 0.7 * (1 + 1.5 * np.exp(-((sy / H - 0.15 - 0.55 * sx / W) / 0.16) ** 2))
    stars = np.zeros((H, W), F32)
    np.add.at(stars, (sy.astype(int), sx.astype(int)), b)
    cv.paint("#ffffff", gaussian_blur(stars, 1.2 * S) * 12 * S * S, "add")   # blur radius ~ S -> compensate brightness by S^2
    for i in rng.choice(n, 14, replace=False):                     # bright stars are picked at random so they do not all cluster in the Milky Way
        cv.glow(sx[i], sy[i], 6 * S, "#cfe0ff", strength=0.5)

    log("moon")
    mx, my, mr = 0.78 * W, 0.22 * H, 0.045 * H
    moon = circle_mask(H, W, mx, my, mr, feather=S)
    maria = fbm2d(H, W, mr * 0.9, 4, seed=seed + 6)
    cv.glow(mx, my, mr * 4, "#5f78b0", strength=0.35)
    cv.paint(lerp("#f2efe4", "#a29f95", smoothstep(-0.1, 0.35, maria)), moon)

    log("hills")
    for base, amp, col, s in [(0.76, 0.10, "#141d33", 10), (0.87, 0.07, "#04060c", 11)]:
        ridge = (base - amp * fbm1d(W, 0.35 * W, seed=seed + s)) * H
        cv.paint(col, clamp01(Y - ridge[None, :] + 0.5))          # 1px anti-aliased silhouette

    log("post")
    cv.bloom(0.8, 0.02 * H, 0.3).vignette(0.3).soft_clip(0.85).grain(0.012, seed=seed)
    return cv
