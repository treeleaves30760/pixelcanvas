# -*- coding: utf-8 -*-
"""Two or three illustrated people in a simple setting, grounded with shadows (figure + form Room)

* Format: 1920x1080
* Focal point: <the two friends talking at the table, faces at eye level>
* Light: <sun through the left window; every person casts a soft shadow to the right>
* Back / middle / front: <window wall and plant / seated pair at the table / passer-by on the left>
* Palette: <warm cream walls, wood, one saturated top colour per person>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/figures.md (person, pose, anchors) and forms.md (Room).
from pixelcanvas import Canvas
from pixelcanvas.figure import person
from pixelcanvas.form import Room, set_light, lathe, cylinder

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_people"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

EYE_LEVEL, VANISH = 0.42, 0.52          # horizon and vanishing point (fractions)
WALL, FLOOR = "#f3dfbd", "planks"
TABLE_AT = (0.6, 3.2)                   # table centre: metres from the room centre, metres from the back wall
TOPS = ("#e3a72f", "#2a9d8f", "#c8553d")    # one shirt colour per person
STYLE = "flat"                          # flat, soft, anime, line


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    set_light(direction=(-0.85, -0.5), elevation=34, color="#ffe9cc", ambient=0.42, front=-0.25)
    # >>> setting: new_scene.py --prompt draws the request's place here (room, street, park, beach, night)
    log("room")                                          # walls, floor and window; people are placed on room.place
    room = Room(cv, horizon=EYE_LEVEL, vanishing=VANISH, back=0.6, width=5.2, height=3.0, eye=1.45)
    room.draw(wall=WALL, side=WALL, floor=FLOOR, window="left", seed=seed, sun=0.9, sun_dir=(1.0, -0.65, -0.35))
    # <<< setting
    # >>> subjects: new_scene.py --prompt writes one call per subject of the request here
    tx, tz = TABLE_AT[0], room.zb - TABLE_AT[1]
    log("woman")                                         # description, skin, hair, outfit, face, pose
    a = person("a woman with a bob in a sweater", style=STYLE, view="side", facing=1, height=1.62, skin="medium",
               hair=dict(style="bob", color="#3a2618"), face=dict(expression="laugh"),
               outfit=dict(top="sweater", bottom="jeans", colors=dict(top=TOPS[0], bottom="#3a5a8a")))
    a.pose("sit", seat="chair", reach_r=("chest", (0.0, -0.5, 0.75)), hand_r="grip", palm_r="in")
    a.draw(cv, room.place(tx - 0.55, tz), shadow=0.5)
    log("man")
    b = person("man", style=STYLE, view="side", facing=-1, height=1.78, skin="deep",
               hair=dict(style="curly", color="black"), face=dict(expression="smile"),
               outfit=dict(top="shirt", bottom="trousers", colors=dict(top=TOPS[1], bottom="#383b45")))
    b.pose("sit", seat="chair", reach_l=("chest", (0.1, -0.7, 0.8)), hand_l="open", palm_l="up")
    b.draw(cv, room.place(tx + 0.55, tz), shadow=0.5)
    log("table")                                         # lathe = round table; cylinder = cups
    lathe(cv, room.place(tx, tz), [(0.24, 0.0), (0.05, 0.05), (0.035, 0.72), (0.4, 0.72), (0.4, 0.76)],
          base="#c9c2b6", material="ceramic", smooth=False, cast=dict(opacity=0.35))
    for dx, col in ((-0.17, "#f2ede4"), (0.2, TOPS[2])):
        cylinder(cv, room.place(tx + dx, tz).up(0.76), 0.065, 0.13, base=col, material="ceramic", open=True,
                 fill=0.82, handle=True, shadow=0.5)
    log("passer-by")                                     # nearest person: biggest, three-quarter view
    c = person("man", view="three_quarter", facing=1, height=1.8, pose="walk", phase=0.6, item="bag",
               hair=dict(style="side_part", color="#8b3b24"),
               outfit=dict(top="coat", bottom="trousers", colors=dict(top="#26405f", bag="#d8c3a0")))
    c.draw(cv, room.place(-1.3, room.zb - 2.4), shadow=0.6)
    # <<< subjects

    log("post")
    cv.vignette(0.22, radius=0.7).soft_clip(0.8).grain(0.008, seed)
    return cv
