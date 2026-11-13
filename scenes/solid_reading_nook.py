# -*- coding: utf-8 -*-
"""Isometric cutaway of a cosy reading nook on a rainy night (solid + clay + elements)

* Format: 1200x1200, the room filling the square
* Focal point: the rust armchair in the pool of lamp light, a ginger cat on its armrest
* Light: one warm floor lamp beside the chair; a cool blue night outside the rainy window
* Contents: armchair, floor lamp, bookshelf, round rug, side table with a mug, plant, cat, rain outside
* Palette: deep navy night, warm cream walls in lamplight, honey floor, rust chair, teal rug
"""
from pixelcanvas import Canvas, elements
from pixelcanvas.clay import animal
from pixelcanvas.solid import (Scene, Camera, Lighting, room, armchair, floor_lamp, bookshelf, rug, side_table,
                               potted_plant, mug)

PROMPT = ("An isometric cutaway of a cosy reading nook: an armchair, a floor lamp, a bookshelf full of books, "
          "a round rug, a cat on the armrest and a window with rain outside.")
NAME = "solid_reading_nook"
KIND = "painting"
DEFAULT_SIZE = (1200, 1200)
DEFAULT_SEED = 1

WALLS = ("#f0dcc0", "#d9c4a8")
CHAIR = "#c4623f"
RUG = ("#3f7f7a", "#efd9b0")
LAMP_POWER = 3.2
LAMP_GLOW = 0.25
CAT_SCALE = 1.5
FILL = 0.98
BACKGROUND = [(0, "#1b2340"), (1, "#0c1020")]


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sc = Scene(lighting=Lighting("night", ambient=0.3, sun_power=0.3), ground=None)

    sc.add(room(3.4, 3.4, 2.6, walls=WALLS[0], walls2=WALLS[1], door=None, window_size=(1.4, 1.3),
                view="#16233f"), "room")
    sc.add(rug(2.1, shape="round", color=RUG[0], border=RUG[1]).place(x=0.1, z=0.35), "round rug")
    sc.add(bookshelf(1.1, 1.9, 0.32, seed=3).place(x=0.4, z=-1.53), "bookshelf")
    chair = sc.add(armchair(CHAIR).rotate_y(-35).place(x=0.1, z=0.4), "armchair")
    sc.add(animal("cat", coat="ginger", pose="sit").scale(CAT_SCALE).face(-35).on(chair, "armrest_r", at=(0.5, 0.45)),
           "cat")
    lamp = sc.add(floor_lamp().place(x=-0.95, z=0.1), "floor lamp")
    sc.light_from(lamp, color="#ffc27a", power=LAMP_POWER, radius=2.4)
    table = sc.add(side_table().place(x=1.0, z=0.95), "side table")
    sc.add(mug("#2f5d8a").scale(2.0).on(table, "top", at=(0.35, 0.5)), "mug")
    sc.add(potted_plant(0.9).place(x=-1.25, z=-1.25), "plant")

    sc.camera = Camera.isometric(sc, fill=FILL)
    res = sc.render(cv, log=log, background=BACKGROUND)

    log("rain")
    elements.rain(cv, density=3.0, angle=10, length=50, opacity=0.8, mask=res.window_mask(), splashes=0, seed=seed)

    log("lamp glow")
    x, y = sc.project((-0.95, 1.45, 0.1))
    cv.glow(x, y, 0.16 * height, "#ffb860", LAMP_GLOW, mode="add")

    log("post")
    cv.bloom(0.9, 12, 0.25).soft_clip(0.85).vignette(0.3).grain(0.008, seed=seed)
    return cv
