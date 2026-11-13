# -*- coding: utf-8 -*-
"""3D studio product shot: backdrop, soft light, shallow focus (solid)

* Format: 1920x1080 (1080x1080 square and 1080x1350 portrait work too)
* Focal point: <the speaker, sharp, just left of centre>
* Light: <soft key from the upper left, warm glow on the backdrop, rim light>
* Back / middle / front: <backdrop sweep / product on a plinth / small prop in front, out of focus>
* Palette: <sage backdrop, charcoal product, one warm accent>
* Exact text: <none, or the product name>
"""
# Read skills/pixelcanvas/references/solid.md (prefabs, materials, lighting, camera).
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting, speaker, cup, cylinder, material

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_product"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

PRODUCT = "#1c1c1e"                  # product colour
BACKDROP = "#c3cdb9"
TURN = 20                            # product rotation in degrees
CAMERA = dict(azimuth=12, elevation=12, fill=0.7)    # view angle; fill = share of the frame the subject takes
SOFTNESS = 0.7                       # 0 = hard shadows, 1 = very soft
FOCUS_BLUR = 10                      # depth-of-field blur in px at 1080p


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    lit = Lighting("studio", sun=(-55, 35), softness=SOFTNESS, rim_power=0.6)
    lit.point((0.0, 0.25, -0.45), color="#ffe6c8", power=1.0, radius=0.25, shadow=False)   # warm pool on the backdrop
    sc = Scene(lighting=lit, ground=None)
    sc.backdrop(BACKDROP, distance=0.6, radius=0.35)

    # every sc.add is one named layer; swap speaker(...) for any prefab (bottle, mug, laptop ...) or your own shape
    sc.add(cylinder(0.11, 0.03, round=0.004).place(x=-0.02), "plinth", material("marble", "#efebe5"))
    # >>> subjects spread 0.5: new_scene.py --prompt writes one call per subject of the request here
    log("speaker")
    sc.add(speaker(PRODUCT).rotate_y(TURN).place(x=-0.02, y=0.03), "speaker")
    log("cup")
    sc.add(cup("#e9e4dc").rotate_y(-40).place(x=0.16, z=-0.06), "cup")
    # <<< subjects
    sc.camera = Camera.frame(sc, **CAMERA)
    res = sc.render(cv, log=log)                         # logs one step per object
    # >>> overlay: 2D calls of the request painted over the render
    # <<< overlay

    log("post")                                          # focus on the middle of the frame, grain, vignette
    res.depth_of_field(cv, blur=FOCUS_BLUR)
    cv.vignette(0.2).grain(0.008, seed=seed)
    return cv
