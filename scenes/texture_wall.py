# -*- coding: utf-8 -*-
"""
Texture wall: an old rubble-stone wall with a brick arch, a weathered plank door with iron fittings, moss at the
base and cobbles in perspective, built entirely from pixelcanvas.texture.

Plan
* Eye level at V = HORIZON (above the picture's ground line), the wall stands on the ground at V = GROUND.
* Light from the upper left (LIGHT): the arch and the left jamb cast a shadow into the recessed door.
* Arch: one ring of radial bricks laid with ``polar_uv`` (brick length along the radius), brick jambs below.
* Door: vertical planks, weathered, two strap hinges with rivets and a ring pull, all with soft contact shadows.
* Ground: cobblestones through ``plane_uv`` (perspective), moss creeping in the joints near the wall base.
"""
import numpy as np

from pixelcanvas import Canvas, sdf
from pixelcanvas.color import F32, mix, clamp01, smoothstep
from pixelcanvas.filters import gaussian_blur
from pixelcanvas.texture import fill, patches, plane_uv, polar_uv, weather, texture

NAME = "texture_wall"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 7

LIGHT = (-0.62, -0.72)
HORIZON = 0.34
GROUND = 0.79
ARCH_X = 0.5
SPRING = 0.41            # spring line of the arch (fraction of H)
OPEN_HALF = 0.118        # half width of the opening (fraction of W)
BRICK_SCALE = 0.62       # bricks relative to their real 215 x 65 size
STONE_SIZE = 100
DOOR_PALETTE = ["#1c130c", "#2f2216", "#463423", "#5f4b35", "#76644e"]
BRICK_PALETTE = ["#4a1f16", "#65291b", "#7b3824", "#8c4a33", "#9a5c43"]
MOSS = 0.45
MOSS_PALETTE = ["#1c2a0a", "#3b5a14", "#668e22", "#9cc244", "#d2e47e"]


def _opening(X, Y, cx, cy, yb, hw):
    box = sdf.sd_box(X, Y, cx, (cy + yb) / 2 + 2, hw, (yb - cy) / 2 + 2)
    return np.where(Y > cy, box, np.minimum(box, sdf.sd_circle(X, Y, cx, cy, hw)))


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080
    cv = Canvas(W, H, "#000000")
    X, Y = cv.xy()
    V = Y / H
    yb = GROUND * H
    cx, cy = ARCH_X * W, SPRING * H
    hw = OPEN_HALF * W
    R = hw
    T = (215 + 4) * BRICK_SCALE * S
    opening = sdf.fill(_opening(X, Y, cx, cy, yb, hw))
    above = clamp01(yb - Y + 0.5)

    log("stone wall")
    wall = above * (1 - opening)
    wm = texture("stone_wall", H, W, seed=seed, mask=wall, light=LIGHT, size=STONE_SIZE, maps=True, bump=1.5,
                 mortar="#7d766a")
    cv.paint(wm.lit, wall)
    weather(cv, wall, amount=0.6, seed=seed)
    cv.paint("#000000", (smoothstep(yb - 170 * S, yb, Y) * 0.4) * wall)

    log("brick arch")
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    ring = sdf.fill(np.maximum(R - r, r - (R + T))) * clamp01(cy + 0.6 * T - Y)     # jambs cover the overlap
    Up, Vp = polar_uv(H, W, cx, cy, radius=R)
    fill(cv, ring, "brick", light=LIGHT, seed=seed + 1, uv=(Vp, Up), scale=BRICK_SCALE, bond="stack", mortar=10,
         shadow=0.55, weathering=0.5, palette=BRICK_PALETTE)

    log("brick jambs")
    jamb = np.zeros((H, W), F32)
    for side in (-1, 1):
        x0, x1 = cx + side * hw, cx + side * (hw + T)
        lo, hi = min(x0, x1), max(x0, x1)
        jamb = np.maximum(jamb, sdf.fill(sdf.sd_box(X, Y, (lo + hi) / 2, (cy + yb) / 2, (hi - lo) / 2, (yb - cy) / 2)))
    fill(cv, jamb, "brick", light=LIGHT, seed=seed + 2, uv=((X - (cx - hw - T)) / S, (Y - cy) / S),
         scale=BRICK_SCALE, bond="english", mortar=10, shadow=0.55, weathering=0.6, palette=BRICK_PALETTE)
    weather(cv, np.maximum(ring, jamb), amount=0.45, seed=seed + 1)

    log("door")
    door = opening * above
    fill(cv, door, "planks", light=LIGHT, seed=seed + 3, angle=90, width=120, length=5000, nails=False,
         palette=DOOR_PALETTE, vary=0.6, gap=3.5, knots=0.5, uv=((X - cx) / S + 10000, Y / S))
    weather(cv, door, amount=0.9, seed=seed + 2)
    dx, dy = 34 * S, 44 * S
    recess = gaussian_blur(1 - sdf.fill(_opening(X - dx, Y - dy, cx, cy, yb, hw)), 7 * S) * door
    cv.paint("#050302", recess * 0.8)
    cv.paint("#000000", gaussian_blur(1 - opening, 4 * S) * door * 0.6)

    log("iron fittings")
    ds = []
    straps = (0.3, 0.8)
    for vy in straps:
        y = cy + (yb - cy) * vy
        xl, xr = cx - hw + 4 * S, cx + hw * 0.5
        ds.append(sdf.sd_box(X, Y, (xl + xr) / 2, y, (xr - xl) / 2, 15 * S, radius=4 * S))
        ds.append(sdf.sd_circle(X, Y, xr, y, 24 * S))
        ds.append(sdf.sd_box(X, Y, xl + 10 * S, y, 12 * S, 30 * S, radius=5 * S))
    pull_x, pull_y = cx + hw * 0.64, cy + (yb - cy) * 0.52
    ds.append(sdf.sd_circle(X, Y, pull_x, pull_y, 19 * S))
    ds.append(sdf.sd_ring(X, Y, pull_x, pull_y + 32 * S, 32 * S, 6 * S))
    d_iron = sdf.union(*ds)
    iron = sdf.fill(d_iron) * door
    m = texture("rust", H, W, seed=seed + 4, mask=iron, light=LIGHT, amount=0.4, maps=True)
    shade = sdf.bevel_light(d_iron, 6 * S, light=LIGHT, strength=0.7)
    sh = gaussian_blur(iron, 5 * S)
    sh = np.roll(np.roll(sh, int(10 * S), 0), int(8 * S), 1)
    cv.paint("#000000", clamp01(sh - iron) * 0.85)
    cv.paint(m.lit * 0.6 * shade[..., None], iron)
    rivet = np.zeros((H, W), F32)
    for vy in straps:
        y = cy + (yb - cy) * vy
        for k in range(6):
            x = cx - hw + 40 * S + k * (hw * 1.42 / 6)
            rivet = np.maximum(rivet, sdf.fill(sdf.sd_circle(X, Y, x, y, 5.5 * S)))
    fill(cv, rivet * door, "hammered_metal", light=LIGHT, seed=seed, size=8,
         palette=["#151515", "#3a3632", "#6d665e", "#b0a698"], bump=2.0)

    log("cobbled ground")
    ground = clamp01(Y - yb + 0.5)
    uv = plane_uv(H, W, horizon=HORIZON, near=1.3, center=ARCH_X, shift=(0, 30))
    gm = texture("cobblestone", H, W, seed=seed + 5, uv=uv, mask=ground, light=LIGHT, size=78, gap=0.12, ss=2,
                 palette=["#77736c", "#8a847a", "#666a6e", "#958b7d", "#7c6d5d", "#a29c91"], mortar="#2e2a24",
                 maps=True)
    cv.paint(gm.lit, ground)
    cv.paint("#000000", ground * np.exp(-(Y - yb) / (30 * S)) * 0.6)
    thr = sdf.fill(sdf.sd_box(X, Y, cx, yb + 13 * S, hw + T * 0.7, 13 * S, radius=3 * S))
    fill(cv, thr, "flagstone", light=LIGHT, seed=seed + 6, size=700, shadow=0.7)

    log("moss")
    joint = 1 - smoothstep(0.3, 0.6, np.asarray(wm.height, F32))                # moss settles in the joints
    bias = 1.0 * smoothstep(GROUND - 0.3, GROUND - 0.02, V) - 0.55
    mm = patches(H, W, coverage=MOSS, size=70, seed=seed, bias=bias, softness=0.06) * wall * joint
    tufts = patches(H, W, coverage=0.22, size=26, seed=seed + 3, softness=0.03, bias=bias * 0.8 - 0.1)
    mm = np.maximum(mm, tufts * wall)
    joints = 1 - smoothstep(0.25, 0.55, np.asarray(gm.height, F32))
    mg = patches(H, W, coverage=MOSS + 0.1, size=50, seed=seed + 1, softness=0.08,
                 bias=1.1 * np.exp(-(Y - yb) / (90 * S)) - 0.6) * ground * (1 - thr) * joints
    fill(cv, np.maximum(mm, mg), "moss", light=LIGHT, seed=seed, size=36, opacity=0.95, palette=MOSS_PALETTE)

    log("lantern")
    lx, ly = cx + hw + T + 120 * S, cy - 10 * S
    arm = sdf.union(sdf.sd_box(X, Y, lx - 45 * S, ly - 70 * S, 45 * S, 5 * S),
                    sdf.sd_box(X, Y, lx - 88 * S, ly - 55 * S, 7 * S, 32 * S, radius=3 * S),
                    sdf.sd_segment(X, Y, lx - 85 * S, ly - 30 * S, lx - 40 * S, ly - 68 * S, 3 * S))
    cap = sdf.sd_polygon(X, Y, [(lx - 34 * S, ly - 40 * S), (lx + 34 * S, ly - 40 * S), (lx, ly - 70 * S)])
    body = sdf.sd_box(X, Y, lx, ly + 5 * S, 27 * S, 44 * S, radius=3 * S)
    glass = sdf.sd_box(X, Y, lx, ly + 5 * S, 19 * S, 36 * S)
    base = sdf.sd_box(X, Y, lx, ly + 54 * S, 32 * S, 6 * S, radius=2 * S)
    frame = np.maximum(sdf.fill(sdf.union(arm, cap, body, base)) - sdf.fill(glass), 0)
    sh = gaussian_blur(sdf.fill(sdf.union(arm, cap, body, base)), 6 * S)
    cv.paint("#000000", np.roll(np.roll(sh, int(16 * S), 0), int(12 * S), 1) * 0.55 * (1 - sdf.fill(body)))
    cv.glow(lx, ly + 5 * S, 300 * S, "#ff9a3c", 0.3)
    cv.glow(lx, ly + 5 * S, 90 * S, "#ffb35c", 0.35)
    fill(cv, frame, "hammered_metal", light=LIGHT, seed=seed + 8, size=10,
         palette=["#0e0d0c", "#262321", "#4a4540", "#8a8075"])
    flame = np.exp(-(((X - lx) / (7 * S)) ** 2 + ((Y - ly - 10 * S) / (14 * S)) ** 2))
    inner = np.exp(-(((X - lx) / (22 * S)) ** 2 + ((Y - ly - 5 * S) / (40 * S)) ** 2))
    cv.paint(mix([(0, "#e8862e"), (0.5, "#ffc672"), (1, "#fff6dc")], np.maximum(flame, 0.6 * inner))
             * (1.2 + 1.6 * flame + 0.5 * inner)[..., None], sdf.fill(glass))
    bars = np.maximum(sdf.fill(np.maximum(glass, np.abs(X - lx) - 1.0 * S)),
                      sdf.fill(np.maximum(glass, np.abs(Y - ly - 5 * S) - 1.0 * S)))
    cv.paint("#141210", bars)

    log("light")
    U = X / W
    warm = mix([(0, "#fff0d6"), (1, "#5c5a70")], clamp01(0.5 * U + 0.7 * V))
    cv.paint(warm, None, mode="multiply")
    cv.paint("#ffd59a", np.exp(-((X - 0.12 * W) ** 2 + (Y + 0.05 * H) ** 2) / (2 * (0.42 * W) ** 2)) * 0.22,
             mode="add")

    log("post")
    cv.vignette(0.4, 0.6).soft_clip(0.85).grain(0.012, seed=seed)
    return cv
