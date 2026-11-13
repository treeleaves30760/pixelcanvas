# -*- coding: utf-8 -*-
"""
A corner café (figure_cafe) — people from pixelcanvas.figure placed in a form.Room with one light.

Composition: eye level 0.4, vanishing point right of centre. Back wall: counter with an espresso machine, a
barista behind it holding a cup, a menu board and pendant lamps. Middle: a round table where two friends sit
facing each other and talk (side views, cups on the table). Front left: a man walks past with a tote bag, the
nearest and largest figure. Sunlight comes through a window in the left wall (upper left), so every shadow falls
to the right and towards the back.
"""
import numpy as np

from pixelcanvas.canvas import Canvas
from pixelcanvas.form import Room, Light, set_light, lathe, cylinder
from pixelcanvas.figure import person

NAME = "figure_cafe"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 7

HORIZON, VANISH = 0.4, 0.56
WALL, SIDE = "#e8d9c0", "#dccab0"
COUNTER, COUNTER_TOP = "#3f5f58", "#d9cdb8"
WOOD = "#8b5a36"
TABLE_X, TABLE_Z = 0.75, 2.0          # table position (z measured from the back wall)


def _chair(room, x, z, facing, seat_h=0.46, color="#6b3f2a"):
    """A simple bistro chair: four legs, a seat and a back rest on the side away from the table."""
    kw = dict(base=color, material="wood", shadow=0.35)
    for dx in (-0.17, 0.17):
        for dz in (-0.17, 0.17):
            room.box(x + dx, z + dz, (0.035, 0.035, seat_h), **kw)
    bx = x - facing * 0.19
    room.box(bx, z, (0.04, 0.4, 0.42), lift=seat_h, round=0.3, **kw)
    room.box(x, z, (0.42, 0.42, 0.05), lift=seat_h, round=0.4, base=color, material="wood", shadow=False)


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080
    cv = Canvas(W, H)
    set_light(Light(direction=(-0.85, -0.5), elevation=34, color="#ffe9cc", intensity=1.05, sky="#c4d2e2",
                    ambient=0.42, bounce="#b58a62", shadow="#3e2c2c", front=-0.25))

    log("room")
    room = Room(cv, horizon=HORIZON, vanishing=VANISH, back=0.6, width=5.2, height=3.0, eye=1.45)
    room.draw(wall=WALL, side=SIDE, floor="checker", floor_colors=("#e9e0cf", "#5b4a42"), window="left",
              seed=seed, sun=0.9, sun_dir=(1.0, -0.65, -0.35))
    room.sun_beam(0.16)
    zb = room.zb

    log("menu board")
    quad = room.panel("back", center=(-0.85, 1.975), size=(1.4, 0.75), color="#2d3431", frame=WOOD,
                      frame_width=0.03)
    x0, y0 = quad[0]
    x1, y1 = quad[2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = (x0 + x1) / 2
    cv.text("MENU", (cx, y0 + 0.12 * (y1 - y0)), int(0.16 * (y1 - y0)), "#f3ead8", anchor="mt", bold=True)
    for i, line in enumerate(("espresso  2.8", "flat white  3.6", "cappuccino  3.8", "chai latte  4.0")):
        cv.text(line, (cx, y0 + (0.34 + i * 0.15) * (y1 - y0)), int(0.1 * (y1 - y0)), "#efe8d8", anchor="mt")

    log("lamps")
    for lx in (-1.2, 0.2, 1.4):
        top = room.project(lx, 3.0, zb - 1.0)
        bot = room.project(lx, 2.25, zb - 1.0)
        cv.paint("#2a2524", np.clip(1.5 * S - np.abs(xx - top[0]), 0, 1) * ((yy > top[1]) & (yy < bot[1])))
        lathe(cv, room.place(lx, zb - 1.0).up(2.05), [(0.0, 0.0), (0.2, 0.0), (0.12, 0.16), (0.03, 0.22)],
              base="#2f4f4a", material="ceramic", shadow=False, smooth=False, name="pendant lamp")
        cv.glow(bot[0], bot[1] + 0.06 * H * room.place(lx, zb - 1.0).k, 0.05 * H, (1.0, 0.82, 0.55),
                strength=0.25)

    log("barista")
    barista = person("barista", view="front", height=1.66, pose="hold", item="cup")
    barista.draw(cv, room.place(-0.75, zb - 0.32), log=None, shadow=0.4)

    log("counter")
    room.box(-0.75, zb - 0.75, (2.6, 0.62, 1.02), base=COUNTER, top=COUNTER_TOP, material="matte", round=0.12,
             cast=dict(opacity=0.3))
    room.box(-1.45, zb - 0.8, (0.5, 0.36, 0.42), base="#c9ccd2", material="metal", lift=1.02, round=0.25)
    for i, dx in enumerate((-0.35, -0.2, 0.05)):
        cylinder(cv, room.place(-0.75 + dx, zb - 0.62).up(1.02), 0.045, 0.09, base=("#f2ede4", "#c8553d",
                                                                                  "#f2ede4")[i],
                 material="ceramic", open=True, fill=0.8, handle=True, shadow=0.5, name="cup")

    log("chairs")
    tz = zb - TABLE_Z
    _chair(room, TABLE_X - 0.62, tz + 0.05, 1)
    _chair(room, TABLE_X + 0.62, tz - 0.05, -1)

    log("friends")
    left = person("a woman with a bob in a mustard sweater", name="woman", style="flat", view="side", facing=1, height=1.62,
                  pose="sit", hair=dict(style="bob", color="#3a2618"), skin="medium",
                  outfit=dict(top="sweater", bottom="jeans", shoes="flats",
                              colors=dict(top="#e3a72f", bottom="#3a5a8a", shoes="#7a3b2a")),
                  face=dict(expression="laugh"))
    left.pose("sit", reach_r=("chest", (0.0, -0.5, 0.75)), hand_r="grip", palm_r="in", nod=-6, turn=0)
    left.draw(cv, room.place(TABLE_X - 0.55, tz + 0.05), shadow=0.5)
    right = person("man", name="man", view="side", facing=-1, height=1.78, skin="deep", hair=dict(style="curly", color="black"),
                   outfit=dict(top="shirt", bottom="trousers", shoes="sneakers", long_sleeves=False,
                               colors=dict(top="#2a9d8f", bottom="#383b45", shoes="#f5f2ec"), extras=["glasses"]),
                   face=dict(expression="smile"))
    right.pose("sit", reach_l=("chest", (0.1, -0.7, 0.8)), pole_l=(0.8, -0.6, -0.2), hand_l="open", palm_l="up",
               nod=-4)
    right.draw(cv, room.place(TABLE_X + 0.55, tz - 0.05), shadow=0.5)

    log("table")
    lathe(cv, room.place(TABLE_X, tz), [(0.24, 0.0), (0.24, 0.02), (0.05, 0.05), (0.035, 0.08), (0.035, 0.72),
                                        (0.4, 0.72), (0.4, 0.76)],
          base="#c9c2b6", material="ceramic", smooth=False, cast=dict(opacity=0.35), name="round table")
    for dx, col in ((-0.17, "#f2ede4"), (0.2, "#c8553d")):
        cylinder(cv, room.place(TABLE_X + dx, tz).up(0.76), 0.05, 0.1, base=col, material="ceramic", open=True,
                 fill=0.82, handle=True, shadow=0.5, name="cup")

    log("passer-by")
    walker = person("man", name="passer-by", view="three_quarter", facing=1, height=1.82, pose="walk", phase=0.62, item="bag",
                    skin="fair", hair=dict(style="side_part", color="#8b3b24"),
                    outfit=dict(top="coat", bottom="trousers", shoes="boots", extras=["scarf"],
                                colors=dict(top="#26405f", bottom="#383b45", shoes="#3a2a24", inner="#f1e6cf",
                                            scarf="#c8553d", bag="#d8c3a0")),
                    face=dict(expression="neutral"))
    walker.look_at((0.85, 0.3))
    walker.draw(cv, room.place(-1.35, zb - 2.3), shadow=0.6)

    log("post")
    cv.vignette(0.22, radius=0.7).soft_clip(0.8).grain(0.008, seed=seed)
    return cv
