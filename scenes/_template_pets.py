# -*- coding: utf-8 -*-
"""Two posed 3D clay animals on a rug in a small garden corner (clay + solid)

* Format: 1920x1080
* Focal point: <the corgi sitting on the rug, looking at the cat>
* Light: <warm afternoon sun from the upper left (SUN); soft shadows to the right>
* Back / middle / front: <grass and sky / the pair on the rug / ball and rug edge>
* Palette: <green grass, a red rug, warm fur colours>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/clay.md (animal, species, poses, coats, prop).
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting
from pixelcanvas.clay import animal, prop

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_pets"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

GRASS = "#8fb46a"                          # floor colour
RUG = "#d9573f"                            # picnic blanket / rug colour
SKY = [(0, "#a9cdea"), (1, "#f1ead6")]
SUN = (-45, 38)                            # light azimuth / elevation relative to the camera (-90 = from the left)
AZIMUTH, ELEVATION = 10, 16                 # camera angle around the scene / above the floor
SPACING = 0.8                             # metres between the two animals


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sc = Scene(lighting=Lighting("sunny", sun=SUN, softness=0.55, ambient=0.45), ground=GRASS)

    # >>> setting: new_scene.py --prompt adds the request's place here (a room, or GRASS / SKY colours)
    # <<< setting
    # >>> subjects spread 2: new_scene.py --prompt writes one call per subject of the request here
    log("rug")                                  # sizes are metres; .place(x, z) stands a model on the floor
    sc.add(prop("picnic_blanket", color=RUG, size=(1.7, 1.2)).place(z=0.15), "rug")
    log("animals")                              # animal(species, pose=..., coat=...); .face(deg) turns the body
    dog = animal("corgi", pose="sit", size=0.5).face(22).place(x=-SPACING / 2, z=0.1, y=0.012)
    cat = animal("cat", pose="stand", coat="calico", size=0.38).face(-25).place(x=SPACING / 2, z=0.3, y=0.012)
    dog.look_at(cat.anchor("head"))             # the corgi watches the cat
    sc.add(dog, "corgi")
    sc.add(cat, "cat")
    log("ball")
    sc.add(prop("ball", kind="beach", r=0.08).place(x=0.62, z=0.62, y=0.012), "ball")
    # <<< subjects

    sc.camera = Camera.frame(sc, azimuth=AZIMUTH, elevation=ELEVATION, fill=0.85)     # fits every model
    res = sc.render(cv, log=log, background=SKY)
    # >>> overlay: 2D calls of the request painted over the render
    # <<< overlay

    log("post")                                 # soft focus on the middle of the frame, vignette and grain
    res.depth_of_field(cv, blur=3)
    cv.vignette(0.15).grain(0.006, seed=seed)
    return cv
