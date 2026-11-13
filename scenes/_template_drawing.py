# -*- coding: utf-8 -*-
"""Engineering part drawing on an A3 sheet: views, section, dimensions (drafting)

* Format: A3 landscape at 150 dpi = 2480x1754 (also written as SVG / DXF with --svg / --dxf)
* Part: <a clamp block LENGTH x HEIGHT x DEPTH mm with a bored hole>
* Views: <front (with cutting plane A-A), top below it, section A-A on the right (first angle)>
* Dimensions: <overall sizes, hole diameter with its fit, depth>
* Title block: <title, drawing number, material, scale>
* Exact text: <TITLE, DRAWING_NO, MATERIAL and notes copied from the request>
"""
# Read skills/pixelcanvas/references/drafting.md (Sheet, views, dimensions, sections, line types).
from pixelcanvas import Canvas
from pixelcanvas.drafting import Sheet

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "drawing"
NAME = "my_drawing"
DEFAULT_SIZE = (2480, 1754)
DEFAULT_SEED = 1

# >>> data: new_scene.py --prompt writes the request's part, sizes and features here
TITLE, DRAWING_NO, MATERIAL = "CLAMP BLOCK", "PC-2001", "S355"
LENGTH, HEIGHT, DEPTH = 80, 50, 30          # part size in mm
HOLE = 16                                   # bore diameter in mm
SCALE = "2:1"                               # drawing scale: the part fills the sheet
# <<< data


# >>> build: the views of the part
def build():
    sh = Sheet("A3", scale=SCALE, title=TITLE, drawing_no=DRAWING_NO, material=MATERIAL,
               style="white")                    # white | paper | blueprint
    cx, cy = LENGTH / 2, HEIGHT / 2
    fr = sh.view("FRONT", origin=(60, 150))              # model mm, y up; origin = sheet mm of model (0, 0)
    fr.rect(0, 0, LENGTH, HEIGHT, chamfer=3)
    fr.hole(cx, cy, HOLE)
    fr.cutting_plane([(cx, 0), (cx, HEIGHT)], "A", direction="right")
    fr.dim_h((0, HEIGHT), (LENGTH, HEIGHT), side="top")
    fr.dim_v((0, 0), (0, HEIGHT))
    fr.dim_diameter(cx, cy, HOLE, tol=(0.018, 0))
    top = sh.projected(fr, "top", gap=20)               # first angle: the top view goes below the front view
    top.rect(0, 0, LENGTH, DEPTH)
    for x in (cx - HOLE / 2, cx + HOLE / 2):
        top.line(x, 0, x, DEPTH, "hidden")              # line types: visible hidden center phantom dim thin
    top.centerline(cx, -3, cx, DEPTH + 3)
    top.dim_v((0, 0), (0, DEPTH), side="left")
    sec = sh.projected(fr, "left", gap=40).section_label("A")
    for y0, y1 in ((0, cy - HOLE / 2), (cy + HOLE / 2, HEIGHT)):
        sec.hatch([(0, y0), (DEPTH, y0), (DEPTH, y1), (0, y1)], material="steel", outline=True)
    sec.centerline(-3, cy, DEPTH + 3, cy)
    sec.dim_h((0, HEIGHT), (DEPTH, HEIGHT), side="top")
    sh.notes(["Untoleranced dimensions ISO 2768-m.", "Break sharp edges 0.5 x 45\u00b0."])
    return sh
# <<< build


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height, (1, 1, 1))
    build().draw(cv, log=log)                            # border, title block and every view as their own layers
    return cv
