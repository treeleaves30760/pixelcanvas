# -*- coding: utf-8 -*-
"""
Studio product shot (solid_product): a 3D render with pixelcanvas.solid.

Composition: a terracotta mug in focus just right of centre, a glass bottle of oil behind it on the left, a chrome
sphere and a small gold bead on the right, all standing on a marble slab in front of a seamless sage backdrop.
Soft key light from the upper left, a warm glow on the backdrop behind the objects, soft shadows, reflections in the
chrome and the glaze, shallow depth of field focused on the mug.
"""
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting, mug, bottle, sphere, rounded_box, material, marble

NAME = "solid_product"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

BACKDROP = "#c3cdb9"
MUG = "#d26f48"
BOTTLE_TINT = "#e2f0e8"
LIQUID = "#e0a33c"
FOCUS_BLUR = 12.0         # px at 1080p
SOFTNESS = 0.7            # 0 = hard shadows, 1 = very soft
AMBIENT = 0.24
CAM_DISTANCE = 0.84
GLOW = 1.2               # warm light pooled on the backdrop


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)

    lit = Lighting("studio", sun=(-62, 36), softness=SOFTNESS, ambient=AMBIENT, rim_power=0.6)
    lit.point((0.02, 0.2, -0.4), color="#ffe6c8", power=GLOW, radius=0.22, shadow=False)
    sc = Scene(lighting=lit, ground=None)
    sc.backdrop(BACKDROP, distance=0.6, radius=0.35)

    top = 0.024
    slab = rounded_box(0.5, top, 0.25, 0.006).at(0.0, top / 2, 0.0)
    sc.add(slab, "marble slab", material("marble", marble("#f4f1ec", "#9a958f", scale=0.2)))
    sc.add(mug(MUG).rotate_y(-30).place(x=0.045, z=0.04, y=top), "mug")
    sc.add(bottle(BOTTLE_TINT, liquid=LIQUID, level=0.62, label="#f6eddc", height=0.25)
           .place(x=-0.115, z=-0.06, y=top), "bottle")
    sc.add(sphere(0.045).place(x=0.175, z=-0.035, y=top), "chrome sphere", "chrome")
    sc.add(sphere(0.019).place(x=0.13, z=0.08, y=top), "gold bead", "gold")
    sc.camera = Camera.orbit(target=(0.015, 0.115, 0.0), distance=CAM_DISTANCE, azimuth=8, elevation=12, fov=26)

    res = sc.render(cv, log=log)

    log("post")
    res.depth_of_field(cv, focus=(0.045, 0.05, 0.04), blur=FOCUS_BLUR)
    cv.vignette(0.22).grain(0.008, seed=seed)
    return cv
