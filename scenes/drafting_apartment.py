# -*- coding: utf-8 -*-
"""Floor plan of a two-bedroom apartment on an A3 sheet: walls, doors with swings, windows, furniture, schedule

* Format: A3 landscape at 150 dpi = 2480x1754
* Plan: 10600 x 7200 mm outside (about 76 m2 gross), 250 mm external walls, 100 mm partitions, balcony on the south side
* Rooms: living room with open kitchen (west), bedroom 1 and bedroom 2 (north), entry, hall, bathroom (south east), balcony
* Openings: entrance door, doors with swings to every room, balcony door, windows in every room
* Furniture: kitchen counter + fridge, dining table, sofa, armchair, two beds, wardrobes, desk, bath, toilet, basin
* Sheet items: room names with areas, dimension chains and overall dimensions, north arrow, scale bar, room schedule
"""
# Read skills/pixelcanvas/references/drafting.md (plans: wall, door, window, furniture, room).
from pixelcanvas import Canvas
from pixelcanvas.drafting import Sheet

PROMPT = ('A floor plan of a two-bedroom apartment of about 75 square metres: living room with open kitchen, two '
          'bedrooms, a bathroom, a balcony; doors with swings, windows, furniture, room names with areas, overall '
          'dimensions, a north arrow and a scale bar.')
KIND = "drawing"
NAME = "drafting_apartment"
DEFAULT_SIZE = (2480, 1754)
DEFAULT_SEED = 1

TITLE, DRAWING_NO = "TWO-BEDROOM APARTMENT", "A-201"
W, D = 10600, 7200                          # outside size in mm (y up, origin = outside bottom-left corner)
EXT, INT = 250, 100                         # wall thicknesses in mm
X1, X2 = 4800, 7700                         # living | bedrooms; bedroom 1 | bedroom 2 and entry | bath
Y1, Y2 = 3600, 2400                         # top of hall / bottom of bedrooms
STYLE = "white"                             # white | paper | blueprint
PLAN_ORIGIN = (72, 118)                     # sheet mm of the plan's outside bottom-left corner
H = EXT / 2
FLOORS = [((H, H, X1, D - H), "#f3e3c8"),                       # (x0, y0, x1, y1) in model mm, floor tint
          ((X1, Y1, X2, D - H), "#dbe6f4"), ((X2, Y1, W - H, D - H), "#e2ecd9"),
          ((X1, Y2, W - H, Y1), "#ece7e1"), ((X1, H, X2, Y2), "#ece7e1"),
          ((X2, H, W - H, Y2), "#d4ecec"), ((600, -1500, 3800, H), "#e3eed6")]


def build():
    sh = Sheet("A3", scale="1:50", title=TITLE, drawing_no=DRAWING_NO, style=STYLE, author="PC",
               date="2026-11-12", material="", tolerances="", revision="A",
               fields=[("UNITS", "mm"), ("PROJECT", "CITY APARTMENTS"), ("LEVEL", "FLOOR PLAN - LEVEL 3"),
                       ("STATUS", "FOR APPROVAL")])
    p = sh.view("PLAN", origin=PLAN_ORIGIN, label="FLOOR PLAN")
    h = EXT / 2
    p.wall([(h, h), (W - h, h), (W - h, D - h), (h, D - h)], thickness=EXT, closed=True)
    p.wall([(X1, h), (X1, D - h)], thickness=INT)                 # living | rest
    p.wall([(X1, Y1), (W - h, Y1)], thickness=INT)                # bedrooms | hall
    p.wall([(X1, Y2), (W - h, Y2)], thickness=INT)                # hall | entry + bath
    p.wall([(X2, h), (X2, Y2)], thickness=INT)                    # entry | bath
    p.wall([(X2, Y1), (X2, D - h)], thickness=INT)                # bedroom 1 | bedroom 2
    # balcony railing (thin walls) south of the living room
    p.wall([(600, h), (600, -1500), (3800, -1500), (3800, h)], thickness=60)
    # doors
    p.door(6300, h, 1000, toward=(6300, 1400), hinge="left")             # entrance
    p.door(6300, Y2, 800, toward=(6300, 1400), hinge="right")            # entry -> hall
    p.door(9100, Y2, 800, toward=(9100, 1400), hinge="left")             # bath
    p.door(6600, Y1, 800, toward=(6600, 4500), hinge="right")            # bedroom 1
    p.door(9100, Y1, 800, toward=(9100, 4500), hinge="left")             # bedroom 2
    p.door(X1, 3000, 900, toward=(3900, 3000), hinge="left")             # hall -> living
    p.opening(2200, h, 1600)             # balcony door
    # windows
    p.window(h, 2000, 1500); p.window(h, 5800, 1400); p.window(2500, D - h, 2400)
    p.window(6350, D - h, 1400); p.window(9100, D - h, 1400); p.window(W - h, 5300, 1200)
    p.window(W - h, 1200, 600); p.window(4300, h, 900)
    # furniture
    p.furniture("counter", 2200, 6600, 3400, 600, sink=1500, hob=2900)
    p.furniture("fridge", 4320, 6625, 640, 650)
    p.furniture("table", 3200, 4700, 1500, 850, seats=4)
    p.furniture("sofa", 700, 2900, 2100, 900, angle=90)
    p.furniture("table", 1750, 2900, 1000, 550, seats=0, angle=90)
    p.furniture("armchair", 3900, 1500, 850, 850, angle=150)
    p.furniture("bed", 6250, 6050, 1500, 1900, angle=0)
    p.furniture("wardrobe", 5150, 4500, 1800, 600, angle=90)
    p.furniture("bed", 9100, 6050, 1500, 1900, angle=0)
    p.furniture("wardrobe", 7950, 4500, 1800, 600, angle=90)
    p.furniture("desk", 10150, 4300, 1200, 600, angle=90)
    p.furniture("bathtub", 9550, 600, 1700, 750)
    p.furniture("toilet", 8200, 590, angle=180)
    p.furniture("basin", 7980, 1700, angle=90)
    # labels
    p.room("LIVING / KITCHEN", 2600, 3900); p.room("BEDROOM 1", 6350, 4700); p.room("BEDROOM 2", 9100, 4700)
    p.room("HALL", 8300, 3000); p.room("ENTRY", 6300, 1600); p.room("BATH", 9100, 1550)
    p.room("BALCONY", 2200, -800, area=3.2 * 1.5)
    # dimensions
    p.dim_chain([(0, D), (X1, D), (X2, D), (W, D)], axis="h", side="top")
    p.dim_h((0, D), (W, D), side="top")
    p.dim_chain([(0, 0), (X1, 0), (X2, 0), (W, 0)], axis="h", side="bottom")
    p.dim_h((0, 0), (W, 0), side="bottom")
    p.dim_chain([(0, 0), (0, Y2), (0, Y1), (0, D)], axis="v", side="left")
    p.dim_v((0, 0), (0, D), side="left")
    p.dim_v((W, 0), (W, D), side="right")
    sh.north_arrow(392, 262, size=16, angle=90)
    sh.scale_bar(300, 250, length=5)
    sh.room_schedule(p, x=300, y=225)
    sh.notes(["Dimensions in mm.", "External walls 250 mm, partitions 100 mm.", "Areas to internal wall faces."],
             x=300, y=150, h=3.0)
    return sh, p


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, (1, 1, 1))
    sh, plan = build()
    sh.draw(cv, log=log)                                 # border, title block and the plan as their own layers
    log("floor colours")                                 # multiply keeps walls, furniture and labels on top
    k = min(width / sh.W, height / sh.H)                 # sheet mm -> pixels (sheet y points up)
    X, Y = cv.xy()
    for (x0, y0, x1, y1), color in FLOORS:
        (a, b), (c, d) = plan.to_sheet(x0, y0), plan.to_sheet(x1, y1)
        box = (X >= a * k) & (X <= c * k) & (Y >= (sh.H - d) * k) & (Y <= (sh.H - b) * k)
        cv.paint(color, box.astype("float32"), mode="multiply")
    return cv
