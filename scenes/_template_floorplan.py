# -*- coding: utf-8 -*-
"""Architectural floor plan at 1:50: walls, doors, furniture, room areas (drafting)

* Format: A3 landscape at 150 dpi = 2480x1754 (also written as SVG / DXF with --svg / --dxf)
* Plan: <outside WIDTH x DEPTH mm; external walls EXT, partitions INT>
* Rooms: <studio with sofa and bed / bath with shower and toilet / store>
* Openings: <entrance door on the bottom wall, one window per room>
* Sheet items: <dimension chains, grid lines, north arrow, scale bar, room schedule>
* Exact text: <TITLE, room names copied from the request>
"""
# Read skills/pixelcanvas/references/drafting.md (plans: wall, door, window, furniture, room).
from pixelcanvas import Canvas
from pixelcanvas.drafting import Sheet

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "drawing"
NAME = "my_floorplan"
DEFAULT_SIZE = (2480, 1754)
DEFAULT_SEED = 1

# >>> data: new_scene.py --prompt writes the request's size, scale and rooms here
TITLE, DRAWING_NO = "STUDIO", "A-100"
WIDTH, DEPTH = 7000, 5000                   # outside size in mm (y up, origin = outside bottom-left corner)
EXT, INT = 250, 100                         # wall thicknesses in mm
SPLIT = 4500                                # x of the partition between the studio and the bath / store
STYLE = "white"                             # white | paper | blueprint
# <<< data


# >>> build: walls, openings, furniture and rooms
def build():
    sh = Sheet("A3", scale="1:50", title=TITLE, drawing_no=DRAWING_NO, style=STYLE)
    p = sh.view("PLAN", label="FLOOR PLAN")              # placed in the free area of the sheet
    h, mid = EXT / 2, DEPTH / 2
    # walls are centre lines; doors and windows find their wall from (x, y)
    p.wall([(h, h), (WIDTH - h, h), (WIDTH - h, DEPTH - h), (h, DEPTH - h)], thickness=EXT, closed=True)
    p.wall([(SPLIT, h), (SPLIT, DEPTH - h)], thickness=INT)
    p.wall([(SPLIT, mid), (WIDTH - h, mid)], thickness=INT)
    p.door(3000, h, 1000, toward=(3000, 1500), hinge="left")
    p.door(SPLIT, 1300, 800, toward=(5600, 1300))
    p.door(SPLIT, 3700, 800, toward=(5600, 3700), hinge="right")
    p.window(h, 2500, 1400); p.window(2200, DEPTH - h, 1800); p.window(WIDTH - h, 1250, 700)
    # furniture symbols: bed sofa table chair toilet basin shower bathtub counter sink fridge wardrobe desk ...
    p.furniture("bed", 1300, 3700, angle=90); p.furniture("sofa", 1500, 900)
    p.furniture("shower", 6350, 1850); p.furniture("toilet", 5300, 470, angle=180)
    p.room("STUDIO", 2700, 3000); p.room("BATH", 5750, 1200); p.room("STORE", 5750, 3700)
    p.dim_chain([(0, DEPTH), (SPLIT, DEPTH), (WIDTH, DEPTH)], side="top")
    p.dim_h((0, 0), (WIDTH, 0)); p.dim_v((0, 0), (0, DEPTH))
    p.gridlines(xs=[h, SPLIT, WIDTH - h], ys=[h, DEPTH - h])
    sh.north_arrow(380, 250); sh.scale_bar(300, 230, length=5); sh.room_schedule(p, x=296, y=200)
    return sh
# <<< build


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height, (1, 1, 1))
    build().draw(cv, log=log)                            # border, title block and the plan as their own layers
    return cv
