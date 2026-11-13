# -*- coding: utf-8 -*-
"""
Still life by a window (form_still_life) — the object-form kit (pixelcanvas.form) end to end.

Composition: a wooden table seen from slightly above (horizon 0.33, table back edge 0.56, front edge 0.93);
a window on the left lights the scene (light from the upper left, shadows fall right and back).
Back to front: glass bottle and an orange, a second orange, a ceramic mug of coffee, an apple, a folded linen
cloth and a metal spoon lying flat. Every object is one call; shadows and reflections come from the shared light.
"""
import numpy as np

from pixelcanvas import Canvas, sdf, fbm2d, lerp, mix, smoothstep, clamp01
from pixelcanvas.form import Frame, Light, set_light, tabletop, lathe, cylinder, ball, drop_shadow, reflection

NAME = "form_still_life"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 3

LIGHT_DIR = (-0.85, -0.5)
LIGHT_ELEVATION = 32
WALL = ("#cdbba3", "#7d6b5a")
TABLE_TOP, TABLE_FRONT, HORIZON = 0.5, 0.95, 0.38
MUG_COLOR = "#e9e1d3"
VARNISH = 0.14            # strength of the reflections in the varnished table
BOTTLE_COLOR = "#5f9a72"
ORANGE = "#f27a14"


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    cv = Canvas(W, H)
    X, Y = cv.xy()
    U, V = X / W, Y / H
    light = set_light(Light(direction=LIGHT_DIR, elevation=LIGHT_ELEVATION, color="#fff0dc", intensity=1.1,
                            sky="#b4c6dc", ambient=0.34, bounce="#a87650", shadow="#3b2a2a", front=0.35))

    log("wall")
    cv.paint(mix([(0, WALL[0]), (1, WALL[1])], np.clip(U * 0.9 + V * 0.25, 0, 1)))
    plaster = fbm2d(H, W, 0.05 * H, 4, seed=seed)
    cv.buf *= (1 + 0.025 * plaster)[..., None]
    # sunlight through the window falling on the wall (soft parallelogram with mullion shadows)
    sx = U - 0.22 + (V - 0.1) * 0.35
    patch = smoothstep(-0.02, 0.03, sx) * smoothstep(0.36, 0.28, sx) * smoothstep(0.02, 0.08, V) \
        * smoothstep(0.58, 0.5, V)
    mull = smoothstep(0.012, 0.004, np.abs(sx - 0.16)) + smoothstep(0.012, 0.004, np.abs(V - 0.3))
    cv.paint("#ffe2b8", patch * (1 - 0.8 * clamp01(mull)) * 0.32, mode="add")

    log("window")
    wx1 = 0.15
    frame = clamp01((wx1 * W - X) + 0.5) * clamp01(Y - 0.03 * H + 0.5) * clamp01(0.62 * H - Y + 0.5)
    cv.paint("#efe7da", frame)
    glass = clamp01((wx1 - 0.025) * W - X + 0.5) * clamp01(Y - 0.06 * H + 0.5) * clamp01(0.59 * H - Y + 0.5)
    outside = mix([(0, "#dfeefa"), (0.6, "#f4f6ee"), (1, "#c9d8b0")], V / 0.6) * 1.25
    foliage = fbm2d(H, W, 0.08 * H, 4, seed=seed + 4)
    outside = lerp(outside, np.array([0.62, 0.74, 0.5]) * 1.15, smoothstep(0.1, 0.5, foliage + V * 0.6 - 0.3))
    cv.paint(outside, glass)
    bar = smoothstep(0.009 * H, 0.005 * H, np.abs(Y - 0.3 * H)) * glass
    cv.paint("#efe7da", bar)
    cv.paint("#000000", clamp01(1 - np.abs(X - wx1 * W) / (0.01 * W)) * frame * 0.15)

    log("table")
    surf = tabletop(cv, top=TABLE_TOP, front=TABLE_FRONT, horizon=HORIZON, kind="wood", edge=0.05, seed=seed,
                    colors=("#8a5a36", "#7a4c2c", "#946238", "#804f2e"), name="table")
    # window light pooling on the table
    pool = np.exp(-(((U - 0.3) / 0.35) ** 2 + ((V - 0.72) / 0.2) ** 2)) * smoothstep(TABLE_TOP, TABLE_TOP + 0.01, V) \
        * smoothstep(TABLE_FRONT + 0.003, TABLE_FRONT - 0.003, V)
    cv.paint("#ffd9a8", pool * 0.18, mode="add")
    cv.paint("#000000", clamp01((V - TABLE_TOP) / 0.04) * 0 + np.exp(-np.maximum(V - TABLE_TOP, 0) / 0.012)
             * smoothstep(TABLE_TOP - 0.002, TABLE_TOP + 0.002, V) * 0.25)

    log("bottle")
    bottle = lathe(cv, surf.place(0.3, 0.72),
                   [(0.085, 0.0), (0.094, 0.025), (0.094, 0.26), (0.084, 0.31), (0.036, 0.4), (0.028, 0.52),
                    (0.031, 0.535)],
                   base=BOTTLE_COLOR, material="glass", fill=0.42, fill_color="#bcd8b4", cast=dict(opacity=0.32), name="bottle")
    bottle.band(0.12, 0.36, "#efe4cc", material="matte")
    reflection(cv, bottle, VARNISH, blur=1.5)
    cx, cy = bottle["top"]
    lathe(cv, (cx / W, cy / H + 0.006), [(0.025, 0.0), (0.026, 0.035), (0.024, 0.05)], base="#b88a5a",
          material="wood", shadow=False, tilt=bottle["tilt"], name="bottle cork")

    log("orange back")
    reflection(cv, ball(cv, surf.place(0.765, 0.7), 0.085, base=ORANGE, material="citrus", cast=True, name="orange"), VARNISH)

    log("orange")
    reflection(cv, ball(cv, surf.place(0.64, 0.56), 0.09, base=ORANGE, material="citrus", seed=2, cast=True, name="orange"),
               VARNISH)

    log("mug")
    mug = cylinder(cv, surf.place(0.465, 0.42), 0.1, 0.21, base=MUG_COLOR, material="ceramic", open=True,
                   fill=0.86, fill_color="#3b1f12", inside="#e3d9c8", handle=True, cast=True, name="mug")
    mug.band(0.62, 0.7, "#3d6d9a")
    reflection(cv, mug, VARNISH, blur=1.0)

    log("apple")
    blush = 0.5 + 0.5 * fbm2d(H, W, 0.09 * H, 3, seed=seed + 2, aspect=2.5) + (0.69 - U) * 2.5
    apple_col = lerp("#d9b54c", "#b3222a", smoothstep(0.25, 0.8, blush))
    apple = ball(cv, surf.place(0.69, 0.3), (0.1, 0.088), base=apple_col, material="fruit", cast=True, name="apple")
    reflection(cv, apple, VARNISH)
    ax, ay = apple["top"]
    sf = Frame(cv, (ax / W, ay / H + 0.008), 0.04, rotate=12)
    sf.form(sdf.sd_segment(sf.X, sf.Y, 0, 0, 0.15, -0.9, 0.09), "#5a3a1c", material="wood")

    log("cloth")
    t = np.abs(np.sin(X / H * 26)) ** 10 + np.abs(np.sin(Y / H * 26 * 1.7)) ** 10
    gingham = lerp("#f1ece2", "#4f74a8", np.clip(t * 0.5, 0, 1))
    cspot = surf.place(0.85, 0.17)
    thick = 0.014 * cspot.k * H
    for layer, (dx, rot, sz) in enumerate(((0.0, -10, 1.0), (0.012, -7, 0.92))):
        def towel(fr):
            wob = 0.025 * np.sin(fr.X * 5 + layer) + 0.02 * np.sin(fr.Y * 7 + 2 * layer)
            return sdf.sd_box(fr.X, fr.Y, 0, 0, 1.0, 0.62, 0.12) + wob
        u, v = cspot[0] + dx, cspot[1] - layer * 2.2 * thick / H
        ext = (-1.25, -0.8, 1.25, 0.8)
        side = Frame(cv, (u, v + thick / H), 0.2 * sz * cspot.k, rotate=rot, flat=cspot.tilt, extent=ext)
        top = Frame(cv, (u, v), 0.2 * sz * cspot.k, rotate=rot, flat=cspot.tilt, extent=ext)
        drop_shadow(cv, side.fill(towel(side)), offset=(0.01, 0.004), blur=0.006, opacity=0.5)
        side.form(towel(side), gingham * 0.8, material="fabric", profile="cushion", width=side.px(0.12))
        top.form(towel(top), gingham, material="fabric", profile="cushion", width=top.px(0.16))

    log("spoon")
    sp = surf.place(0.42, 0.1)
    fr = Frame(cv, sp, 0.13, rotate=-18, flat=True, extent=(-1.4, -0.5, 1.4, 0.5))
    d = sdf.smooth_union(sdf.sd_ellipse(fr.X, fr.Y, -0.85, 0, 0.42, 0.26),
                         sdf.sd_segment(fr.X, fr.Y, -0.4, 0, 1.25, 0.04, 0.07), 0.12)
    drop_shadow(cv, fr.fill(d), offset=(0.008, 0.006), blur=0.005, opacity=0.6)
    fr.form(d, "#c9ccd2", material="metal", width=fr.px(0.14))
    fr.form(sdf.sd_ellipse(fr.X, fr.Y, -0.85, 0, 0.34, 0.19), "#c9ccd2", material="metal", height=-0.8,
            width=fr.px(0.2))                                       # the hollow of the bowl

    log("post")
    cv.vignette(0.3, radius=0.62).soft_clip(0.86).grain(0.012, seed=seed)
    return cv
