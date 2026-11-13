# -*- coding: utf-8 -*-
"""Room in one-point perspective with furniture at three depths (form Room)

* Format: 1920x1080
* Focal point: <the sofa under the framed print on the back wall>
* Light: <sun through the left window; patch on the floor, shadows to the right>
* Back / middle / front: <sofa and print / lamp and side table / ball and box on the floor>
* Palette: <warm walls WALL, teal sofa SOFA, wood WOOD>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/forms.md (Room, Ground, solids, materials).
from pixelcanvas import Canvas
from pixelcanvas.form import Room, set_light, lathe, ball

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_interior"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

EYE_LEVEL, VANISH = 0.44, 0.46          # horizon and vanishing point (fractions)
ROOM_WIDTH, EYE_HEIGHT = 5.2, 1.35      # metres: room width, camera height
FLOOR = "planks"                        # floor texture: planks, tiles, checker ...
WALL = "#dcccb4"
SOFA, WOOD = "#4f7f86", "#8b5a36"


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    set_light(direction=(-0.9, -0.4), elevation=30, front=-0.2)          # sunlight from the left window

    # >>> setting: new_scene.py --prompt draws the request's place here (room, street, park, beach, night)
    log("room")                                          # walls, floor, window and the sun patch in one call
    room = Room(cv, horizon=EYE_LEVEL, vanishing=VANISH, back=0.5, width=ROOM_WIDTH, eye=EYE_HEIGHT)
    room.draw(wall=WALL, floor=FLOOR, window="left", seed=seed, sun_dir=(1.0, -0.7, -0.4))
    z = room.zb                                          # depth of the back wall; place furniture at z - metres
    # <<< setting
    # >>> subjects: new_scene.py --prompt writes one call per subject of the request here
    log("picture")
    room.picture("back", center=(0.2, 1.8), size=(1.2, 0.8), seed=seed)
    log("sofa")                                          # room.box(x, z, (width, depth, height)) in metres
    room.box(0.2, z - 0.2, (2.3, 0.26, 0.86), base=SOFA, material="fabric", round=0.35)
    room.box(0.2, z - 0.5, (2.3, 0.92, 0.42), base=SOFA, material="fabric", round=0.35, cast=True)
    log("lamp")                                          # lathe profile = (radius, height) pairs in metres
    lamp = room.place(-1.8, z - 0.5)
    lathe(cv, lamp, [(0.16, 0), (0.016, 0.05), (0.016, 1.4)], base="#2b2725", material="metal", cast=True)
    lathe(cv, lamp.up(1.34), [(0.24, 0), (0.15, 0.34)], base="#f3e3c4", material="fabric", emit=0.25, open=True,
          shadow=False)
    log("side table")
    lathe(cv, room.place(1.8, z - 0.6), [(0.2, 0), (0.035, 0.05), (0.035, 0.52), (0.3, 0.54), (0.3, 0.58)],
          base=WOOD, material="wood", smooth=False, cast=True)
    log("ball")                                          # objects near the camera show the depth of the room
    ball(cv, room.place(1.2, z - 2.6), 0.16, base="#2f6fb8", material="plastic", cast=True)
    log("box")
    room.box(-0.95, z - 2.3, (0.4, 0.4, 0.4), base="#e0b44a", material="matte", yaw=25, round=0.15, cast=True)
    # <<< subjects

    log("post")
    cv.vignette(0.28, 0.65).soft_clip(0.86).grain(0.01, seed)
    return cv
