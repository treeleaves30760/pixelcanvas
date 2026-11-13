# -*- coding: utf-8 -*-
"""Isometric 3D room diorama with furniture (solid)

* Format: 1920x1080 (1080x1080 square works too)
* Focal point: <the desk under the window, lit by the sun and the lamp>
* Light: <warm sun through the left-wall window + the desk lamp>
* Contents: <bed, bookshelf, desk with laptop and lamp, chair, rug, plant>
* Palette: <cream / pale blue walls, honey floor, a warm accent on the rug>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/solid.md (room, prefabs, Camera.isometric).
from pixelcanvas import Canvas
from pixelcanvas.solid import (Scene, Camera, Lighting, room, planks, bed, bookshelf, table, chair, laptop, desk_lamp,
                               rug, potted_plant)

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_isometric"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

ROOM = 4.0                                   # floor size in metres; furniture sits within +-ROOM / 2
WALLS = ("#efe2cf", "#d6e2e4")               # left wall, right wall
FLOOR = ("#c99a68", "#a8774a")               # plank colours light, dark
SUN = (-100, 33)                             # world azimuth / elevation of the sun through the left window
LAMP_POWER = 0.9
BACKGROUND = [(0, "#dbe6ee"), (1, "#a9c0d0")]


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sc = Scene(lighting=Lighting("interior", sun=SUN, sun_power=2.6, ambient=0.6, softness=0.25), ground=None)
    R = ROOM / 2

    # shell: floor slab + two walls, window on the left wall (side, from, width, height in metres)
    sc.add(room(ROOM, ROOM, 2.6, walls=WALLS[0], walls2=WALLS[1], door=None, window=("left", 0.3, 1.2, 1.2),
                floor=planks(light=FLOOR[0], dark=FLOOR[1])), "room")
    # >>> subjects spread 2.4: new_scene.py --prompt writes one call per subject of the request here
    # furniture: .place(x, z) stands it on the floor, .rotate_y(deg) turns it; one named layer each
    sc.add(rug(2.0, 1.4, color="#c8664a", border="#f2e4cc").at(0.3, 0, 0.5), "rug")
    sc.add(bed(1.4, 2.0, duvet="#f1ede6", throw="#6f8fb0").place(x=0.9, z=-R + 1.0), "bed")
    sc.add(bookshelf(0.8, 1.8, 0.3, seed=seed).place(x=-0.9, z=-R + 0.15), "bookshelf")
    sc.add(table(1.2, 0.6, 0.74).rotate_y(90).place(x=-R + 0.3, z=0.4), "desk")
    sc.add(laptop().rotate_y(90).place(x=-R + 0.3, z=0.5, y=0.74), "laptop")
    lamp = sc.add(desk_lamp("#f2c14e").scale(1.4).rotate_y(-35).place(x=-R + 0.2, z=0.0, y=0.74), "desk lamp")
    sc.light_from(lamp["bulb"], color="#ffcf8a", power=LAMP_POWER, radius=0.9)
    sc.add(chair("#e0b98a", seat="#d96a4a").rotate_y(-100).place(x=-R + 0.95, z=0.45), "chair")
    sc.add(potted_plant(0.9).place(x=-R + 0.35, z=R - 0.4), "plant")
    # <<< subjects

    sc.camera = Camera.isometric(sc)                     # fitted to the scene and the canvas
    sc.render(cv, log=log, background=BACKGROUND)
    # >>> overlay: 2D calls of the request painted over the render
    # <<< overlay

    log("post")
    cv.vignette(0.12).grain(0.008, seed=seed)
    return cv
