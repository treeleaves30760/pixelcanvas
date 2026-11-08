# -*- coding: utf-8 -*-
"""
Aurora over a mountain lake — README hero banner (wide 12:5) and gallery piece.

Composition:
* Horizon (lake edge) at V = 0.66; snowy ridge from V ≈ 0.42 at the sides down to 0.62, lower in the middle.
* Focal point: the aurora's brightest curtain fold at about (0.62, 0.35), right third; title block on the
  left third (U 0.06~0.46, V 0.16~0.42) over the darker, calmer sky.
* Light: the aurora itself (green below, magenta / violet above) + faint starlight. Ridges get a green rim on top,
  snow catches a little green; the lake mirrors everything with ripples, slightly darker.
* Near: pine silhouettes on both banks frame the lake, darkest values of the image.
* Palette: night #050816 → #0d1a3a, aurora #3dff9a / #2bd2c4 / #c04bff, snow #b9c7d8, pines #03060b.
"""
import numpy as np

from pixelcanvas import *
from pixelcanvas.scenery import Scene, pine_mask

NAME = "aurora_lake"
DEFAULT_SIZE = (2400, 1000)
DEFAULT_SEED = 11

HORIZON = 0.66
SKY = [(0.0, "#03050f"), (0.40, "#0a1430"), (0.58, "#132a48"), (0.66, "#1b3a52")]
AURORA_LOW, AURORA_MID, AURORA_HIGH = "#3dff9a", "#2bd2c4", "#c04bff"
AURORA_STRENGTH = 1.4
TITLE = "pixelcanvas"
TAGLINE = "Every pixel computed. No image model."
SUBLINE = "numpy + Pillow  ·  a drawing skill for AI agents"
SHOW_TEXT = True


def aurora(W, H, X, Y, S, seed):
    """Curtains: a wavy lower edge per column, bright just above it, fading upwards, with vertical rays."""
    U = X / W
    u = U[0]
    base = (0.52 + 0.035 * fbm1d(W, 0.3 * W, 4, seed=seed) - 0.09 * np.sin((u - 0.15) * 2 * np.pi * 0.9)) * H
    base = base + 0.05 * H * np.exp(-((u - 0.62) / 0.06) ** 2)                      # the fold dips at the focal point
    base = base + 0.022 * H * fbm1d(W, 0.05 * W, 3, seed=seed + 5)                 # small curtain folds
    h = (Y - base[None, :]) / (0.30 * H)                    # 0 at the lower edge, negative above
    up = np.clip(-h, 0, None)
    rays = fbm2d(H, W, 0.012 * W, 4, seed=seed + 1, aspect=40) * 0.5 + 0.5
    rays = smoothstep(0.35, 0.85, rays)
    patch = smoothstep(-0.2, 0.6, fbm2d(H, W, 0.25 * W, 3, seed=seed + 2, aspect=0.6))   # brighter / dimmer folds
    body = np.exp(-up * 2.2) * smoothstep(0.06, -0.02, h)  # sharp-ish lower edge, long fade upwards
    hem = np.exp(-((h + 0.03) / 0.035) ** 2) * (0.25 + 0.75 * rays) * (0.3 + 0.7 * patch)
    body = body + 1.3 * hem                                 # real aurorae have a bright, broken lower hem
    focus = 0.45 + 1.1 * np.exp(-((U - 0.62) / 0.14) ** 2)
    dens = body * (0.3 + 0.7 * rays) * (0.35 + 0.65 * patch) * focus
    h2 = (Y - base[None, :] + 0.13 * H) / (0.22 * H)                                 # fainter upper curtain
    up2 = np.clip(-h2, 0, None)
    dens = dens + 0.35 * np.exp(-up2 * 2.5) * smoothstep(0.05, -0.03, h2) * rays * (1 - patch * 0.5)
    col = mix([(0.0, AURORA_LOW), (0.25, AURORA_MID), (0.75, AURORA_HIGH), (1.0, AURORA_HIGH)], clamp01(up / 1.2))
    return dens.astype(F32), col


def render(width=2400, height=1000, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    sc = Scene(width, height, seed)
    cv, X, Y, U, V, W, H = sc.cv, sc.X, sc.Y, sc.U, sc.V, sc.W, sc.H
    S = H / 1080
    hz = HORIZON * H
    sky_mask = (Y < hz).astype(F32)

    log("sky & stars")
    cv.gradient(SKY)
    rng = np.random.default_rng(seed)
    n = int(2600 * W * H / (2400 * 1000))
    px, py = rng.uniform(0, W, n), rng.uniform(0, hz, n)
    b = rng.random(n) ** 7 * 1.6 + 0.08
    pts = np.zeros((H, W), F32)
    np.add.at(pts, (py.astype(int), px.astype(int)), b)
    cv.paint("#dfe8ff", gaussian_blur(pts, 0.7 * S) * 5 * S * S, "add")

    log("aurora")
    dens, acol = aurora(W, H, X, Y, S, seed + 20)
    glow = gaussian_blur(dens, 18 * S)
    cv.paint(acol, (dens * 1.1 + glow * 0.6) * AURORA_STRENGTH * sky_mask, "add")
    cv.paint("#1d6f63", gaussian_blur(dens, 80 * S) * 0.6 * sky_mask, "add")        # sky lit by the aurora
    cv.paint("#1f5a55", smoothstep(0.38, 0.66, V) * 0.25 * sky_mask, "add")           # green airglow at the horizon

    log("mountains")
    far = (0.585 - 0.12 * fbm1d(W, 0.22 * W, 6, seed=seed + 30, mode="ridged")
           + 0.05 * np.exp(-((U[0] - 0.2) / 0.18) ** 2)) * H                         # lower under the title
    m_far = clamp01(Y - far[None, :] + 0.5)
    rel = clamp01((Y - far[None, :]) / (0.12 * H))
    rock = fbm2d(H, W, 90 * S, 5, seed=seed + 31, mode="ridged")
    snow = smoothstep(0.55, 0.85, rock * 0.55 + (1 - rel) * 0.6) * (1 - rel)        # snow only near the crests
    far_col = lerp(lerp("#0d1428", "#1f2a46", rock), lerp("#3f5a70", "#7f9eb0", rock), snow)
    far_col = lerp(far_col, "#13283a", 0.15 + 0.45 * rel)                              # haze towards the lake
    cv.paint(far_col, m_far)
    rim = m_far * (1 - sample(m_far, X, Y - 3 * S))
    cv.paint(AURORA_LOW, rim * 0.55, "add")

    near = (0.60 - 0.07 * fbm1d(W, 0.12 * W, 6, seed=seed + 40, mode="ridged")
            + 0.05 * np.exp(-((U[0] - 0.55) / 0.2) ** 2)) * H
    near = np.minimum(near, hz - 2 * S)
    m_near = clamp01(Y - near[None, :] + 0.5)
    tex = fbm2d(H, W, 25 * S, 5, seed=seed + 41) * 0.5 + 0.5
    cv.paint(lerp("#060a14", "#16213a", tex * 0.6), m_near)
    cv.paint(AURORA_LOW, m_near * (1 - sample(m_near, X, Y - 2 * S)) * 0.35, "add")
    sc.occlude(np.maximum(m_far, m_near))

    log("lake reflection")
    rip = fbm2d(H, W, 40 * S, 4, seed=seed + 50, aspect=0.08)
    rip2 = fbm2d(H, W, 160 * S, 3, seed=seed + 51, aspect=0.2)
    depth = clamp01((Y - hz) / (H - hz))
    refl = cv.sample(X + rip2 * 6 * S, 2 * hz - Y + (rip * 7 + rip2 * 4) * S * (0.3 + depth))
    water = refl * (0.78 - 0.25 * depth)[..., None]
    water = lerp(water, "#050a14", 0.15 + 0.35 * depth)
    streak = smoothstep(0.55, 0.9, rip * 0.5 + 0.5) * 0.08 * (1 - depth)               # subtle horizontal sheen
    cv.paint(water, smoothstep(hz - 0.5, hz + 0.5, Y))
    cv.paint("#9fe8d0", streak * (Y > hz), "add")
    cv.paint("#0a1a24", band_mask(H, W, hz - 1.5 * S, hz + 2 * S, soft=1.5 * S) * 0.6)

    log("pines")
    trees = []
    for side, x0, x1 in [(-1, 0.0, 0.2), (1, 0.84, 1.0)]:
        for _ in range(26):
            x = rng.uniform(x0, x1) * W
            edge = (x / W - x1) / (x1 - x0) if side < 0 else (x0 - x / W) / (x1 - x0)
            yb = hz + (6 + 30 * rng.random()) * S
            hgt = (110 + 260 * clamp01(-edge * 1.2) ** 0.8 + 50 * rng.random()) * S
            trees.append((x, yb, hgt, hgt * 0.3))
    pm = pine_mask(sc, trees)
    bank = clamp01(Y - (hz + (8 + 14 * fbm1d(W, 80 * S, 4, seed=seed + 60)) * S)[None, :] + 0.5)
    bank = bank * (smoothstep(0.24, 0.16, U) + smoothstep(0.80, 0.88, U)) * (Y < hz + 60 * S)
    tm = np.maximum(pm, bank)
    cv.paint("#03060b", tm)
    cv.paint(AURORA_MID, tm * (1 - sample(tm, X + 1.5 * S, Y - 2 * S)) * 0.35, "add")
    tree_refl = sample(tm, X + rip * 3 * S, 2 * hz - Y + rip * 6 * S) * (Y > hz)
    cv.paint("#02050a", tree_refl * 0.75)

    if SHOW_TEXT:
        log("title")
        x0 = 0.065 * W
        cv.paint("#000000", smoothstep(0.55, 0.0, U) * smoothstep(0.62, 0.1, V) * 0.35)   # calm the title area
        cv.text(TITLE, box=(x0, 0.08 * H, 0.46 * W, 0.27 * H), box_anchor="lb", color="#f2fbff", bold=True,
                glow=dict(radius=40 * S, color="#2bd2c4", strength=0.45),
                shadow=dict(dx=0, dy=4 * S, blur=10 * S, color="#000000", opacity=0.6))
        t = cv.last_bbox
        cv.text(TAGLINE, (x0 + 4 * S, t[3] + 26 * S), 46 * S, "#bff7e4", anchor="lt")
        t2 = cv.last_bbox
        cv.text(SUBLINE.upper(), (x0 + 4 * S, t2[3] + 22 * S), 24 * S, "#8fa8c8", anchor="lt", tracking=0.18,
                bold=True)

    log("post")
    cv.bloom(0.7, 0.015 * H, 0.35).vignette(0.3, 0.65).grain(0.012, seed).soft_clip(0.85)
    return cv
