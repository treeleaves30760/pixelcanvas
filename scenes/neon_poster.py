# -*- coding: utf-8 -*-
"""
Neon poster — demonstrates typography (pixelcanvas.text), SDF shapes (pixelcanvas.sdf) and layout.

Composition (portrait 4:5):
* Top 0.08~0.30: title block — big neon title + small tracked kicker above it.
* Centre: a striped sunset disc (SDF circle minus horizontal bands) sitting on the horizon at V = 0.62,
  framed by two dark mountain silhouettes; the disc is the focal point, on the vertical centre line.
* Bottom 0.62~1: perspective grid floor fading into haze; info card (rounded SDF box, bevel) with date text.
* Light: the disc is the light source → rim light on mountain tops, magenta glow on the horizon.
* Palette: night violet #120a2a → magenta #ff3cac → orange #ffb347; cyan accent #2bd2ff for the grid.
"""
import numpy as np

from pixelcanvas import *
from pixelcanvas import sdf

NAME = "neon_poster"
DEFAULT_SIZE = (1080, 1350)
DEFAULT_SEED = 3

TITLE = "NEON NIGHTS"
KICKER = "CITY  ·  SYNTH  ·  AFTER DARK"
SUBTITLE = "霓虹之夜"
INFO = "SAT 12.19  ·  21:00\nWAREHOUSE NO.9"
HORIZON = 0.62
SUN = (0.5, 0.52, 0.17)           # centre x, centre y, radius (fractions of width)
GRID = "#2bd2ff"
SKY = [(0.0, "#0b0620"), (0.35, "#2a0f4a"), (0.55, "#7a1f6e"), (0.62, "#ff5f8a")]


def render(width=1080, height=1350, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = min(W / 1080, H / 1350)                       # design was laid out at 1080x1350
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()
    hz = HORIZON * H

    log("sky")
    cv.gradient(SKY)
    stars = (np.random.default_rng(seed).random((H, W)) > 0.9985) * (1 - smoothstep(0.2, 0.5, V))
    cv.paint("#ffffff", gaussian_blur(stars.astype(F32), 0.6 * S) * 2.5, "add")
    cv.glow(W / 2, hz, 0.35 * W, "#ff3cac", strength=0.35)

    log("sun disc")
    cx, cy, r = SUN[0] * W, SUN[1] * H, SUN[2] * W
    disc = sdf.sd_circle(X, Y, cx, cy, r)
    t = clamp01((Y - (cy - r)) / (2 * r))             # 0 top → 1 bottom of the disc
    gap = 2 + 14 * t ** 1.5                            # stripes get thicker towards the bottom
    period = 0.075 * r
    phase = np.mod(Y - (cy - r), period + gap * S)
    stripes = (phase > period) & (t > 0.45)
    sun_col = mix([(0, "#ffe66d"), (0.55, "#ffb347"), (1, "#ff3cac")], t)
    cv.glow(cx, cy, r * 1.3, "#ff7a59", strength=0.25)
    cv.paint(sun_col, sdf.fill(disc, 1.2 * S) * (1 - stripes) * (Y < hz))

    log("mountains")
    for side, base, amp, col, sd in [(-1, 0.50, 0.10, "#1a0b2e", 1), (1, 0.53, 0.09, "#14081f", 2)]:
        ridge = (base + amp * (1 - fbm1d(W, 0.12 * W, 5, seed=seed + sd, mode="ridged"))) * H
        side_w = smoothstep(0.42, 0.18, U) if side < 0 else smoothstep(0.58, 0.82, U)   # only at the sides
        ridge_y = hz - (hz - ridge[None, :]) * side_w
        m = clamp01(Y - ridge_y + 0.5) * (Y < hz + 1)
        cv.paint(col, m)
        rim = m * (1 - sample(m, X + side * 3 * S, Y - 3 * S))
        cv.paint("#ff6fb5", rim * 0.8, "add")

    log("grid floor")
    dy = np.maximum(Y - hz, 0.5)
    z = 0.12 * H / dy                                  # depth: 1 near the bottom → large at the horizon
    wx = (X - W / 2) / dy * 0.12 * H                   # world x
    fade = smoothstep(0.0, 0.16 * H, dy) ** 1.5         # fade out towards the horizon (avoids moire)
    lw = 1.2 * S
    gx = np.abs(np.mod(wx / (0.09 * H) + 0.5, 1) - 0.5) * (0.09 * H) * dy / (0.12 * H)   # px distance to line
    gz = np.abs(np.mod(z * 6 + 0.5, 1) - 0.5) / 6 * dy / np.maximum(z, 1e-6)
    lines = np.maximum(clamp01(lw - gx + 0.5), clamp01(lw - gz * 0.5 + 0.5)) * fade
    floor = smoothstep(hz - 0.5, hz + 0.5, Y)
    cv.paint(mix([(0, "#2a0f4a"), (1, "#05020d")], clamp01((Y - hz) / (H - hz))), floor)
    cv.paint(GRID, lines * floor * 0.9)
    cv.paint(GRID, gaussian_blur(lines * floor, 3 * S) * 0.8, "add")
    cv.paint("#ff3cac", band_mask(H, W, hz - 2 * S, hz + 2 * S, soft=4 * S) * 0.9, "add")

    log("title")
    cv.text(KICKER, (W / 2, 0.085 * H), 22 * S, color="#ffd1ec", anchor="mm", tracking=0.25, bold=True)
    cv.text(TITLE, box=(0.06 * W, 0.11 * H, 0.94 * W, 0.24 * H), color=(1.5, 0.75, 1.35), bold=True,
            glow=dict(radius=30 * S, color="#ff3cac", strength=0.9), outline=2.2 * S, outline_color="#fff2fb")
    title_box = cv.last_bbox
    cv.text(SUBTITLE, (W / 2, title_box[3] + 0.03 * H), 60 * S, color="#ffe66d", anchor="mt", tracking=0.4,
            shadow=dict(dx=0, dy=3 * S, blur=6 * S, color="#2a0030", opacity=0.7))

    log("info card")
    bx, by, bw, bh = W / 2, 0.86 * H, 0.30 * W, 0.055 * H
    card = sdf.sd_box(X, Y, bx, by, bw, bh, radius=bh)
    cv.paint("#000000", sdf.glow(sdf.offset(card, -6 * S), 18 * S) * 0.5)
    cv.paint(lerp("#2a0f4a", "#4a1a6e", 0.5) * sdf.bevel_light(card, 6 * S, strength=0.35)[..., None], sdf.fill(card))
    cv.paint(GRID, sdf.outline(card, 2 * S), opacity=0.9)
    cv.text(INFO, box=(bx - bw, by - bh, bx + bw, by + bh), padding=bh * 0.35, color="#e8f9ff", bold=True,
            line_spacing=1.25, align="center")

    log("post")
    cv.bloom(0.8, 0.02 * H, 0.35).vignette(0.35, 0.6).grain(0.012, seed).soft_clip(0.85)
    return cv
