# -*- coding: utf-8 -*-
"""
Still life: a red ball on a table (still_life) — the minimal example of lighting from normals; any curved object can follow this pattern.

Composition: table edge at V=0.62; the ball is centred slightly above middle, its bottom resting on the table.
Light: upper left, in front (light vector L points at the light) -> highlight upper left, shadow stretched to the back right.
Techniques:
* Ball normal: dx, dy = (pixel - centre) / radius, dz = sqrt(1 - dx^2 - dy^2).
* Lighting = ambient + diffuse max(n.L, 0) + specular (r.v)^k + table bounce (downward faces get a little light).
* A cylinder uses the same recipe with a one-axis normal (nx across, nz = sqrt(1 - nx^2)); a draped cloth lights a height field through its gradient normal.
* The cast shadow is a flat ellipse anchored at the contact point and stretched away from the light; a narrow ambient-occlusion band at the contact keeps the ball from floating.
"""
import numpy as np

from pixelcanvas import *

NAME = "still_life"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 2


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = cv.uv()
    L = np.array([-0.55, -0.6, 0.58])
    L /= np.linalg.norm(L)

    log("wall & table")
    cv.paint(mix([(0, "#3b3a44"), (1, "#5a5560")], U * 0.3 + V * 0.7))
    wall_tex = 1 + 0.05 * (fbm2d(H, W, 0.02 * W, 5, seed=seed + 1) - 0.5)[..., None]       # plaster mottling
    cv.paint(rgb("#7a6f86") * wall_tex, clamp01(1.0 - np.hypot((X - 0.2 * W) / (0.75 * W), (Y - 0.05 * H) / (0.7 * H))) * 0.55)  # window light pool
    grain_wood = 1 + 0.06 * fbm2d(H, W, 0.4 * W, 4, seed=seed, aspect=0.05)[..., None]   # stretched wood grain
    table = smoothstep(0.618 * H - 1, 0.618 * H + 1, Y)
    cv.paint(mix([(0.62, "#7a634d"), (1, "#3a2c22")], V) * grain_wood, table)
    cv.paint("#e8c9a0", np.exp(-((Y - 0.622 * H) / (0.006 * H)) ** 2) * 0.35 * table)    # lit table edge

    def ball(cx, cy, r, base, bounce_col, shadow_len):
        """A lit sphere with contact shadow and occlusion (light from L)."""
        dx, dy = (X - cx) / r, (Y - cy) / r
        rr = dx * dx + dy * dy
        dz = np.sqrt(np.clip(1 - rr, 0, 1))
        ndl = np.clip(dx * L[0] + dy * L[1] + dz * L[2], 0, 1)
        spec = np.clip(2 * ndl * dz - L[2], 0, 1) ** 40
        bounce = np.clip(dy, 0, 1) * 0.25
        col = rgb(base) * (0.12 + 0.95 * ndl[..., None]) + rgb(bounce_col) * bounce[..., None] + spec[..., None] * 1.4
        sphere = 1 - smoothstep(1 - 2 / r, 1 + 1 / r, np.sqrt(rr))
        sx, sy = cx - L[0] / L[2] * r * 0.45, cy + r * 0.92
        e = ((X - sx) / (r * shadow_len)) ** 2 + ((Y - sy) / (r * 0.22)) ** 2
        cv.paint("#120c0a", feather(clamp01(1.2 - e), r * 0.08) * 0.75)
        ao = np.exp(-(((X - cx) / (r * 0.9)) ** 2 + ((Y - (cy + r)) / (r * 0.12)) ** 2))
        cv.paint("#000000", ao * 0.6)
        return col, sphere

    # window light on the wall
    sh = (X - 0.45 * Y) / H                                              # window frame projected onto the wall (sheared)
    pane = smoothstep(0.0, 0.02, np.abs(((sh * 3.2) % 1.0) - 0.5) - 0.07) * smoothstep(0.0, 0.02, np.abs(((Y / H * 3.2) % 1.0) - 0.5) - 0.07)
    patch = (1 - smoothstep(0.3, 0.9, U)) * (1 - smoothstep(0.28, 0.62, V))
    cv.paint("#e0c08e", feather(clamp01(pane * patch), 0.004 * H) * 0.09 * (1 - table), "add")
    cv.paint("#14121a", np.exp(-((Y - 0.608 * H) / (0.012 * H)) ** 2) * 0.25 * (1 - table))   # dark seam above the table edge

    # cloth
    cx0, cy0 = 0.27 * W, 0.855 * H
    cu, cvv = (X - cx0) / (0.2 * W), (Y - cy0) / (0.1 * H)
    inside = 1 - smoothstep(0.97, 1.0, np.maximum(np.abs(cu) ** 3.5 + np.abs(cvv + 0.15 * cu) ** 3.5, 0) ** (1 / 3.5))
    folds = 0.5 * np.sin(cu * 6.5 + 0.8 * cvv + 2.2 * fbm2d(H, W, 0.15 * W, 3, seed=seed + 5) + cvv * 1.5) * (1 - 0.5 * cvv ** 2)
    folds += 0.12 * fbm2d(H, W, 0.08 * W, 3, seed=seed + 6)
    gx, gy = np.gradient(folds * 0.02 * H, axis=1), np.gradient(folds * 0.02 * H, axis=0)
    nrm = np.stack([-gx * 1.2, -gy * 1.6, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    cloth_l = np.clip(nrm @ L, 0, 1)
    sh_m = feather(np.roll(np.roll(inside, int(0.012 * H), axis=0), int(0.014 * W), axis=1), 0.012 * H)
    cv.paint("#0c0806", sh_m * 0.5 * table)
    cv.paint(rgb("#e9d9bf") * (0.30 + 0.80 * cloth_l[..., None]) * (1 + 0.03 * fbm2d(H, W, 4, 2, seed=seed + 7)[..., None]),
             inside * table)

    # cylinder
    def jar(cx, ybase, rw, hgt, base):
        """A lit cylinder: one-axis normal, rim ellipse, cast shadow stretched away from the light."""
        ry = rw * 0.28
        nx = np.clip((X - cx) / rw, -1, 1)
        nz = np.sqrt(1 - nx * nx)
        ndl = np.clip(nx * L[0] + nz * L[2], 0, 1)
        spec = np.clip(2 * ndl * nz - L[2], 0, 1) ** 30
        body = (1 - smoothstep(1 - 1.5 / rw, 1 + 0.5 / rw, np.abs((X - cx) / rw))) *             smoothstep(ybase - hgt - 1, ybase - hgt + 1, Y) * (1 - smoothstep(-1, 1, Y - (ybase + ry * nz)))
        col = rgb(base) * (0.12 + 0.95 * ndl[..., None]) + spec[..., None] * 0.5
        # shadow: parallelogram sheared away from the light, fading with distance
        reach = np.clip((X - cx) / (hgt * 1.5), 0, 1)
        band = np.abs(Y - ybase - 0.3 * ry * reach) / (ry * (1.0 + 0.5 * reach))
        shadow = (1 - smoothstep(0.7, 1.0, band)) * smoothstep(-rw, -rw * 0.2, X - cx) * (1 - smoothstep(0.55, 1.0, reach))
        cv.paint("#120c0a", shadow * 0.7 * table)
        ao = np.exp(-(((X - cx) / (rw * 1.05)) ** 2 + ((Y - ybase) / (ry * 0.9)) ** 2))
        cv.paint("#000000", ao * 0.4)
        cv.paint(col, body)
        # open top: dark inside, bright rim
        e = ((X - cx) / rw) ** 2 + ((Y - (ybase - hgt)) / ry) ** 2
        rim = 1 - smoothstep(1 - 2 / rw, 1 + 1 / rw, np.sqrt(e))
        cv.paint(rgb(base) * 0.16, rim)
        cv.paint(rgb(base) * 1.1, np.exp(-((np.sqrt(e) - 1) / (1.5 / rw + 0.03)) ** 2) * (Y < ybase - hgt + ry * 0.2) * 0.5, "add")

    jar(0.30 * W, 0.665 * H, 0.075 * H, 0.2 * H, "#d9b45a")

    log("small sphere")
    col, m = ball(0.69 * W, 0.62 * H - 0.105 * H + 0.04 * H, 0.105 * H, "#2f7f86", "#d9a07a", 1.6)
    cv.paint(col, m)

    log("sphere")
    col, m = ball(0.5 * W, 0.47 * H, 0.2 * H, "#c0392b", "#d9a07a", 1.3)
    cv.paint(col, m)

    log("post")
    cv.vignette(0.35).soft_clip(0.85).grain(0.01, seed=seed)
    return cv
