# -*- coding: utf-8 -*-
"""Three-quarter 3D scene with prefabs on a floor (solid)

* Format: 1920x1080
* Focal point: <the house in the middle, the character waving in front of it>
* Light: <afternoon sun from the upper left (SUN); soft shadows to the right>
* Back / middle / front: <trees / house and car / character>
* Palette: <green lawn, warm walls, red roof, a bright car>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/solid.md (prefabs, materials, Lighting, Camera.frame).
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting, house, tree, car, character

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_scene3d"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

GROUND = "#8fb46a"                         # floor colour (or ("wood", "#d9b48c"), "checker")
SKY = [(0, "#9cc4e8"), (1, "#e8f0f0")]
SUN = (-50, 40)                            # light azimuth / elevation relative to the camera (-90 = from the left)
AZIMUTH, ELEVATION = 30, 9                 # camera angle around the scene / above the floor
HAZE = 0.03                                # depth fog density
BLUR = 4                                   # depth-of-field blur, px at 1080p


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sc = Scene(lighting=Lighting("sunny", sun=SUN, softness=0.5), ground=GROUND)

    # >>> setting: new_scene.py --prompt adds the request's place here (a room, or GROUND / SKY colours)
    # <<< setting
    # >>> subjects spread 8: new_scene.py --prompt writes one call per subject of the request here
    # sizes are metres; .place(x, z) stands a model on the floor, .rotate_y(deg) turns it; one named layer each
    sc.add(house(walls="#f1dcc0", roof="#c0503a").rotate_y(-10).place(x=0.0, z=-2.0), "house")
    sc.add(tree(5.0, kind="round").place(x=-6.0, z=0.0), "oak")
    sc.add(tree(6.0, kind="pine").place(x=5.0, z=-3.5), "pine")
    sc.add(car("#2f6fb8").rotate_y(-120).place(x=4.2, z=2.0), "car")
    sc.add(character(shirt="#f2b33d", pants="#3d5a80", pose="wave", height=1.7).rotate_y(20).place(x=-1.6, z=3.0),
           "character")
    # <<< subjects

    sc.camera = Camera.frame(sc, azimuth=AZIMUTH, elevation=ELEVATION)    # fits everything to the canvas
    res = sc.render(cv, log=log, background=SKY)
    # >>> overlay: 2D calls of the request painted over the render
    # <<< overlay

    log("post")                                          # haze by distance, soft focus on the middle
    res.fog(cv, SKY[1][1], density=HAZE, start=10.0)
    res.depth_of_field(cv, blur=BLUR)
    cv.vignette(0.15).grain(0.008, seed=seed)
    return cv
