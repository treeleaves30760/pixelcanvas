# -*- coding: utf-8 -*-
"""Two or three posed 3D clay characters on a soft studio ground (clay + solid)

* Format: 1920x1080
* Focal point: <the chef waving in the middle, face at eye level>
* Light: <warm key light from the upper left (SUN); soft contact shadows to the right>
* Back / middle / front: <soft backdrop / the trio side by side / ground with shadows>
* Palette: <cream backdrop, one saturated top colour per character>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/clay.md (cast, Character, poses, expressions, hold).
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting
from pixelcanvas.clay import cast, Character, prop

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_characters"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

GROUND = "#e9ddcc"                         # floor colour
BACKDROP = [(0, "#f7ecde"), (1, "#e8d6c0")]    # soft wall behind, top to bottom
SUN = (-40, 38)                            # light azimuth / elevation relative to the camera (-90 = from the left)
AZIMUTH, ELEVATION = 8, 7                  # camera angle around the group / above the floor
GAP = 0.85                                 # metres between characters


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sc = Scene(lighting=Lighting("studio", sun=SUN, softness=0.6, ambient=0.42), ground=GROUND)

    # >>> setting: new_scene.py --prompt adds the request's place here (a room, or GROUND / BACKDROP colours)
    # <<< setting
    # >>> subjects spread 3: new_scene.py --prompt writes one call per subject of the request here
    log("characters")           # ready-made roles: cast(role, pose=...); or build one with Character(...)
    chef = cast("chef", pose="wave", hold=None).face(14).place(x=-GAP)           # .face(deg) turns toward the camera
    kid = Character("kid", build="kid", skin=3, hair=dict(style="pigtails", color="ginger"),
                    face=dict(expression="laugh"),
                    outfit=dict(top="hoodie", bottom="jeans", shoes="sneakers", colours=dict(top="#e8613a")))
    kid.pose("cheer").face(0)
    doctor = cast("doctor", pose="hold", skin=5).face(-14).place(x=GAP)
    doctor.express("smile")                       # expressions: smile happy laugh surprised wink shy calm ...
    doctor.hold(prop("book", color="#3f86d8"), hand="r")     # one held prop
    for who in (chef, kid.place(z=0.25), doctor):
        sc.add(who, who.name)                     # one named layer each
    # <<< subjects

    sc.camera = Camera.frame(sc, azimuth=AZIMUTH, elevation=ELEVATION, fill=0.9)   # fits the group
    res = sc.render(cv, log=log, background=BACKDROP)
    # >>> overlay: 2D calls of the request painted over the render
    # <<< overlay

    log("post")
    cv.vignette(0.14).grain(0.006, seed=seed)
    return cv
