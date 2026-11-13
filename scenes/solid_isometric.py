# -*- coding: utf-8 -*-
"""
Isometric cosy room (solid_isometric): a cut-away diorama rendered with pixelcanvas.solid.

Composition: true isometric camera looking at a room on a thick floor slab; two back walls (window on the left
wall). Late warm sun pours through the window onto the desk and the floor; a desk lamp adds a warm pool of light.
Furniture: bed with throw against the right wall, bookshelf, desk under the window with laptop, lamp and mug,
chair, rug, potted plant, a framed print over the bed. Soft pastel background.
"""
from pixelcanvas import Canvas
from pixelcanvas.solid import (Scene, Camera, Lighting, box, material, bed, table, chair, bookshelf, desk_lamp,
                               laptop, mug, potted_plant, rug, book, stripes, room, planks)

NAME = "solid_isometric"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 3

ROOM = 4.4                # floor size (m)
WALL_H = 2.7
WALL_LEFT = "#efe2cf"
WALL_RIGHT = "#d6e2e4"
FLOOR_A, FLOOR_B = "#c99a68", "#a8774a"
SUN = (-100, 33)          # world azimuth / elevation of the sun (light comes through the left window)
SUN_POWER = 2.6
LAMP_POWER = 0.9
BG_TOP, BG_BOTTOM = "#dbe6ee", "#a9c0d0"


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    R, t = ROOM / 2, 0.16

    lit = Lighting("interior", sun=SUN, sun_power=SUN_POWER, ambient=0.62, softness=0.25)
    sc = Scene(lighting=lit, ground=None)

    # ---- shell: floor slab and two walls with a window on the left wall, in one call
    sc.add(room(ROOM - 2 * t, ROOM - 2 * t, WALL_H, walls=WALL_LEFT, walls2=WALL_RIGHT, door=None,
                window=("left", 0.25, 1.25, 1.25), floor=planks(light=FLOOR_A, dark=FLOOR_B), thickness=0.16),
           "room")

    # ---- furniture
    sc.add(rug(2.1, 1.5, color="#c8664a", border="#f2e4cc").at(0.35, 0, 0.55), "rug")
    sc.add(bed(1.5, 2.05, duvet="#f1ede6", throw="#6f8fb0").place(x=0.95, z=-R + t + 1.07), "bed")
    sc.add(bookshelf(0.85, 1.9, 0.3, seed=seed).place(x=-1.15, z=-R + t + 0.16), "bookshelf")
    desk = table(1.3, 0.62, 0.74, color=None).rotate_y(90).place(x=-R + t + 0.33, z=0.3)
    sc.add(desk, "desk")
    sc.add(laptop(open_deg=108).rotate_y(90).place(x=-1.73, z=0.42, y=0.74), "laptop")
    lamp = desk_lamp("#f2c14e", angle=35).scale(1.25).rotate_y(-35).place(x=-1.86, z=-0.12, y=0.74)
    sc.add(lamp, "desk lamp")
    sc.light_from(lamp["bulb"], color="#ffcf8a", power=LAMP_POWER, radius=0.9)
    sc.add(mug("#f2b33d", size=1.0).rotate_y(140).place(x=-1.7, z=0.78, y=0.74), "mug")
    sc.add(book(0.045, 0.24, 0.17, "#8a3a4a").rotate_z(90).rotate_y(10).place(x=-1.8, z=-0.45, y=0.74), "book")
    sc.add(chair("#e0b98a", seat="#d96a4a").rotate_y(-100).place(x=-1.05, z=0.35), "chair")
    sc.add(potted_plant(0.95, pot="#e9e4da", leaves="#4c9a4a").place(x=-1.62, z=1.5), "plant")
    frame = (box(0.9, 0.62, 0.04) - box(0.8, 0.52, 0.2)).at(0.95, 1.6, -R + t + 0.02)
    sc.add(frame, "print frame", material("wood", "#3a2a20"))
    art = box(0.8, 0.52, 0.01).at(0.95, 1.6, -R + t + 0.01)
    sc.add(art, "print", material("matte", stripes(0.08, "#f2c36b", "#e07a5f", axis="x")))

    sc.camera = Camera.isometric(target=(-0.1, 0.85, 0.1), size=5.3)
    sc.render(cv, log=log, background=[(0, BG_TOP), (1, BG_BOTTOM)])

    log("post")
    cv.vignette(0.12).grain(0.008, seed=seed)
    return cv
