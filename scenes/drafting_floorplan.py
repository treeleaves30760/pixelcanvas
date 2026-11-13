# -*- coding: utf-8 -*-
"""
One-bedroom apartment floor plan on an A3 sheet at 1:50 (pixelcanvas.drafting).

* Plan: 9.60 x 7.20 m outside; 250 mm external walls, 100 mm partitions with clean joins and solid poche.
  Open living / kitchen on the left, bedroom top right, hall and bathroom bottom right.
* Annotation: doors with swing arcs, windows, furniture symbols, room names with areas, structural grid with
  bubbles, chained and overall dimension strings on all four sides, north arrow, scale bar, room schedule above
  the title block, notes.
* Model units are millimetres with y up (origin = outside bottom-left corner); the view scale is the sheet's 1:50.
"""
from pixelcanvas.drafting import Sheet

NAME = "drafting_floorplan"
DEFAULT_SIZE = (2480, 1754)            # A3 landscape at 150 dpi
DEFAULT_SEED = 1

STYLE = "white"                        # white | paper | blueprint
W, D = 9600, 7200                      # outside dimensions (mm)
EXT, INT = 250, 100                    # wall thicknesses
X1, X2, Y1 = 5600, 7300, 3100          # partition centre lines

ROOMS = [  # name, label position (the area is measured from the walls)
    ("LIVING / KITCHEN", (2750, 3550)),
    ("BEDROOM", (6550, 6250)),
    ("HALL", (6450, 1800)),
    ("BATH", (8350, 1750)),
]


def build():
    sh = Sheet("A3", scale="1:50", title="ONE-BEDROOM APARTMENT", drawing_no="A-101", author="pixelcanvas",
               date="2026-11-12", material="", tolerances="", finish="", revision="C", checked="AR",
               style=STYLE, fields=[("UNITS", "mm"), ("PROJECT", "RIVERSIDE FLATS"),
                                    ("LEVEL", "FLOOR PLAN - LEVEL 2"), ("STATUS", "FOR APPROVAL")])
    p = sh.view("FLOOR PLAN", origin=(56, 100), label="FLOOR PLAN")
    h = EXT / 2

    # ---- walls (centre lines) and openings
    p.wall([(h, h), (W - h, h), (W - h, D - h), (h, D - h)], thickness=EXT, closed=True)
    p.wall([(X1, h), (X1, D - h)], thickness=INT)
    p.wall([(X1, Y1), (W - h, Y1)], thickness=INT)
    p.wall([(X2, h), (X2, Y1)], thickness=INT)
    p.door(6450, h, 1000, toward=(6450, 1500), hinge="left")          # entrance
    p.door(X1, 2250, 900, toward=(4000, 2250), hinge="right")         # hall -> living
    p.door(6450, Y1, 900, toward=(6450, 4500), hinge="right")         # hall -> bedroom
    p.door(X2, 2400, 800, toward=(8400, 2400), hinge="left")          # hall -> bath
    p.window(h, 1900, 1800)
    p.window(h, 4700, 1600)
    p.window(2900, h, 2400)
    p.window(1700, D - h, 1200)
    p.window(7500, D - h, 1600)
    p.window(W - h, 1500, 600)

    # ---- furniture
    p.furniture("counter", 2050, 6650, 3600, 600, sink=1250, hob=2850)
    p.furniture("fridge", 4180, 6625, 640, 650)
    p.furniture("table", 3100, 4650, 1400, 850, seats=4)
    p.furniture("sofa", 700, 2300, 2100, 900, angle=90)
    p.furniture("table", 1750, 2300, 1000, 550, seats=0, angle=90)
    p.furniture("armchair", 2950, 1250, 850, 850, angle=180 + 30)
    p.furniture("bed", 8350, 5000, 1600, 2000, angle=-90)
    p.furniture("wardrobe", 5950, 4450, 1800, 600, angle=90)
    p.furniture("shower", 8900, 2600, 900, 900)
    p.furniture("toilet", 8350, 590, angle=180)
    p.furniture("basin", 7560, 1150, angle=90)

    # ---- labels, dimensions, grid
    for name, (x, y) in ROOMS:
        p.room(name, x, y)
    p.dim_chain([(0, 0), (X1, 0), (X2, 0), (W, 0)], axis="h", side="bottom")
    p.dim_h((0, 0), (W, 0), side="bottom")
    p.dim_chain([(0, D), (1100, D), (2300, D), (6700, D), (8300, D), (W, D)], axis="h", side="top")
    p.dim_h((0, D), (W, D), side="top")
    p.dim_chain([(0, 0), (0, 1000), (0, 2800), (0, 3900), (0, 5500), (0, D)], axis="v", side="left")
    p.dim_v((0, 0), (0, D), side="left")
    p.dim_chain([(W, 0), (W, Y1), (W, D)], axis="v", side="right")
    p.dim_v((W, 0), (W, D), side="right")
    p.gridlines(xs=[h, X1, X2, W - h], ys=[h, Y1, D - h])

    # ---- sheet furniture (right-hand column)
    sh.north_arrow(392, 266, size=16, angle=90)
    sh.scale_bar(296, 244, length=5)
    sh.room_schedule(p, x=296, y=222)
    sh.notes(["All dimensions in mm; verify on site.",
              "External walls 250 mm, partitions 100 mm.",
              "Door widths are clear openings.",
              "Room areas to internal wall faces."], x=296, y=165, h=3.0)
    return sh


def render(width=2480, height=1754, seed=1, log=None):
    from pixelcanvas import Canvas
    sh = build()
    cv = Canvas(width, height, (1, 1, 1))
    sh.draw(cv, log=log)
    return cv
