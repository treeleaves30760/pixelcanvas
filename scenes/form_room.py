# -*- coding: utf-8 -*-
"""
Living room in one-point perspective (form_room) — Room, Ground.box and lathe solids from pixelcanvas.form.

Composition: eye level 0.44, vanishing point slightly left of centre; a window in the left wall throws a sunlight
patch across the plank floor and the rug. Back wall: a sofa with cushions under a framed print. Right: a round side
table with a vase. Left: a floor lamp. Front: a coffee table with a bowl and a ball on the floor, so objects stand
at three depths. One light direction (from the window, upper left) drives every shadow.
"""
import numpy as np

from pixelcanvas import Canvas, sdf, smoothstep, lerp
from pixelcanvas.form import Room, Light, set_light, lathe, ball, cylinder, Frame

NAME = "form_room"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 4

HORIZON, VANISH = 0.44, 0.46
WALL, SIDE = "#dcccb4", "#d2c0a6"
SOFA = "#4f7f86"
CUSHION = "#d9a55b"
WOOD = "#8b5a36"
RUG = ("#b8553d", "#efe2c8", "#2f4f63")


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    cv = Canvas(W, H)
    set_light(Light(direction=(-0.9, -0.42), elevation=30, color="#ffe8c8", intensity=1.05, sky="#b9c8dc",
                    ambient=0.42, bounce="#b08560", shadow="#3a2a26", front=-0.2))

    log("room")
    room = Room(cv, horizon=HORIZON, vanishing=VANISH, back=0.5, width=5.2, height=2.9, eye=1.35)
    room.draw(wall=WALL, side=SIDE, floor="planks", window="left", seed=seed, sun=1.0,
              sun_dir=(1.0, -0.7, -0.4))
    zb = room.zb
    room.sun_beam(0.22)

    log("rug")
    room.rug(-1.2, 1.9, zb - 2.6, zb - 0.75, color=RUG[0], border=RUG[1], pattern_color=RUG[2], border_width=0.14)

    log("picture")
    room.picture("back", center=(0.25, 1.8), size=(1.25, 0.8), seed=seed)

    log("sofa")
    sx, sz = 0.25, zb - 0.5
    kw = dict(material="fabric", round=0.35)
    room.box(sx, sz + 0.28, (2.3, 0.26, 0.86), base=SOFA, shadow=False, **kw)                 # back rest
    room.box(sx, sz, (2.3, 0.92, 0.42), base=SOFA, cast=dict(opacity=0.35), **kw)              # base
    for dx in (-0.55, 0.55):
        room.box(sx + dx, sz - 0.04, (1.06, 0.72, 0.16), base=SOFA, lift=0.42, **kw)           # seat cushions
        room.box(sx + dx, sz + 0.14, (0.98, 0.2, 0.46), base=SOFA, lift=0.55, yaw=0, **kw)     # back cushions
    for dx in (-1.26, 1.26):
        room.box(sx + dx, sz, (0.24, 0.94, 0.64), base=SOFA, **kw)                             # arms
    room.box(sx - 0.78, sz + 0.02, (0.42, 0.14, 0.4), base=CUSHION, lift=0.56, yaw=-8, material="velvet",
             round=0.8)

    log("lamp")
    lamp = room.place(-1.85, zb - 0.55)
    lathe(cv, lamp, [(0.17, 0), (0.17, 0.025), (0.06, 0.04), (0.016, 0.06), (0.016, 1.42)], base="#2b2725",
          material="metal", smooth=False, cast=dict(opacity=0.3), name="floor lamp")
    top = lamp.up(1.36)
    shade = lathe(cv, top, [(0.24, 0.0), (0.15, 0.34)], base="#f3e3c4", material="fabric", smooth=False,
                  shadow=False, emit=0.25, open=True, inside="#fff2d8", name="lamp shade")
    lx, ly = shade["center"]
    cv.glow(lx, ly + 0.03 * H, 0.09 * H, (1.0, 0.8, 0.5), strength=0.22)

    log("side table")
    st = room.place(1.85, zb - 0.6)
    lathe(cv, st, [(0.2, 0), (0.2, 0.02), (0.035, 0.05), (0.035, 0.52), (0.3, 0.54), (0.3, 0.58)],
                  base=WOOD, material="wood", smooth=False, cast=dict(opacity=0.35), name="side table")
    vase = lathe(cv, st.up(0.58), [(0.06, 0), (0.1, 0.08), (0.08, 0.2), (0.04, 0.27), (0.05, 0.3)],
                 base="#2f6f8f", material="glossy", shadow=0.5, name="vase")
    vx, vy = vase["top"]
    S = H / 1080
    for i, (dx, dy, c) in enumerate(((-26, -62, "#e8a040"), (18, -74, "#f2d050"), (2, -92, "#e07050"),
                                     (-8, -48, "#f0b860"))):
        fx, fy = vx + dx * S, vy + dy * S
        stem = sdf.sd_segment(*cv.xy(), vx, vy, fx, fy, 1.6 * S)
        cv.paint("#4c6b34", sdf.fill(stem))
        ball(cv, (fx / W, (fy + 11 * S) / H), 11 / 1080, base=c, material="velvet", shadow=False, name="flower")

    log("coffee table")
    ct = (0.3, zb - 1.75)
    for dx in (-0.52, 0.52):
        for dz in (0.24, -0.24):
            room.box(ct[0] + dx, ct[1] + dz, (0.05, 0.05, 0.36), base="#5a3a24", material="wood", shadow=0.4)
    room.box(ct[0], ct[1], (1.2, 0.62, 0.06), base="#9a6a42", material="wood", lift=0.36, round=0.3,
             cast=dict(opacity=0.35))
    bowl = room.place(ct[0] - 0.15, ct[1]).up(0.42)
    lathe(cv, bowl, [(0.07, 0), (0.17, 0.04), (0.22, 0.1), (0.23, 0.11)], base="#e8e2d6", material="ceramic",
          open=True, inside="#efe8dc", shadow=0.5, name="bowl")
    cup = room.place(ct[0] + 0.32, ct[1] + 0.05).up(0.42)
    cylinder(cv, cup, 0.05, 0.1, base="#c8513a", material="ceramic", open=True, fill=0.8, handle=True,
             shadow=0.5, name="cup")

    log("fruit bowl")
    for dx, dz, c in ((-0.07, 0.04, "#d8452f"), (0.08, 0.03, "#f0a030"), (0.0, -0.05, "#e8c23a")):
        ball(cv, room.place(ct[0] - 0.15 + dx, ct[1] + dz).up(0.43), 0.065, base=c, material="fruit", shadow=0.4, name="fruit")

    log("plant")
    pot = room.place(2.15, zb - 2.0)
    potm = lathe(cv, pot, [(0.13, 0), (0.17, 0.3), (0.19, 0.33), (0.19, 0.36)], base="#b8643e", material="clay",
                 open=True, inside="#3b2a1e", smooth=False, cast=dict(opacity=0.35), name="plant pot")
    px_, py_ = potm["top"]
    rng = np.random.default_rng(seed)
    for i in range(11):
        ang = -80 + i * 16 + rng.uniform(-6, 6)
        ln = rng.uniform(0.75, 1.05)
        leaf = Frame(cv, (px_ / W, py_ / H), 0.5 * pot.k * ln, rotate=ang - 90, extent=(-0.3, -0.2, 1.2, 0.2))
        dleaf = sdf.sd_ellipse(leaf.X, leaf.Y, 0.55, 0.0, 0.5, 0.13)
        leaf.form(dleaf, lerp("#3f7a3a", "#6aa84f", rng.uniform()), material="fruit", profile="ridge",
                  width=leaf.px(0.13), ks=0.25)

    log("wall shelf")
    w2 = room.width / 2
    shz, shy = zb - 1.6, 1.55
    rng2 = np.random.default_rng(seed + 1)
    zc = shz - 0.5
    while zc < shz + 0.45:
        th = rng2.uniform(0.04, 0.07)
        hgt = rng2.uniform(0.2, 0.28)
        col = ("#7a2e2e", "#2e4a6a", "#c89a3a", "#3e6a48", "#e0d6c0", "#5a3a5e")[int(rng2.integers(6))]
        room.box(w2 - 0.12, zc + th / 2, (0.2, th, hgt), base=col, material="matte", lift=shy + 0.035,
                 shadow=False, round=0.2)
        zc += th + 0.004
    room.box(w2 - 0.13, shz, (0.26, 1.2, 0.035), base=WOOD, material="wood", lift=shy, shadow=False)

    log("toys")
    ball(cv, room.place(1.25, zb - 2.6), 0.16, base="#2f6fb8", material="plastic", cast=dict(opacity=0.45), name="blue ball")
    room.box(-0.95, zb - 2.35, (0.4, 0.4, 0.4), base="#e0b44a", material="matte", yaw=25, round=0.15,
             cast=dict(opacity=0.4))

    log("post")
    cv.vignette(0.28, radius=0.65).soft_clip(0.86).grain(0.01, seed=seed)
    return cv
