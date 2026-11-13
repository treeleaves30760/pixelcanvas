# -*- coding: utf-8 -*-
"""
Toy scene (solid_toys): a three-quarter view of toys on a checker floor, rendered with pixelcanvas.solid.

Composition: a waving clay character in the middle, a rounded red car on the left driving toward the viewer, a
retro rocket behind on the right, a stack of letter blocks at the front right and a striped ball at the front
left. Warm key light from the upper left, soft contact shadows, glossy plastic, a gentle fade into the warm
background and a light depth of field.
"""
from pixelcanvas import Canvas
from pixelcanvas.solid import (Scene, Camera, Lighting, sphere, material, stripes, car, rocket, character, toy_block,
                               quadruped)

NAME = "solid_toys"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 5

FLOOR_A, FLOOR_B = "#f4e7d4", "#e6cfb2"
BG_TOP, BG_BOTTOM = "#f6dcc4", "#f1e2cf"
KEY = "#ffd6a6"
FOG = 0.035
BLUR = 5.0


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)

    lit = Lighting("studio", sun=(-48, 40), sun_color=KEY, sun_power=1.35, softness=0.45, ambient=0.36,
                   zenith="#e6ddd6", horizon="#f0d8c0", ground_color="#a88f78", rim_power=0.5)
    sc = Scene(lighting=lit)
    sc.floor(checker=(FLOOR_A, FLOOR_B), size=0.6, roughness=0.5, reflect=0.06)

    sc.add(character(shirt="#f2b33d", pants="#3d6fb0", pose="wave", height=1.15).rotate_y(18).place(x=0.0, z=0.1),
           "character")
    sc.add(car("#e2483a").scale(0.4).rotate_y(-62).place(x=-1.35, z=0.5), "car")
    sc.add(rocket(body="#f4f1ea", height=1.6).material(hull="plastic").rotate_y(20).place(x=1.3, z=-0.7), "rocket")
    sc.add(quadruped("#c98d5a", kind="dog", height=0.42).rotate_y(-30).place(x=-0.95, z=-0.45), "dog")
    blocks = [("A", "#3d8be0", 1.05, 0.85, 0, 8), ("B", "#46b37a", 1.44, 0.7, 0, -12),
              ("C", "#f2b33d", 1.25, 0.77, 0.34, 22)]
    for letter, col, x, z, y, rot in blocks:
        sc.add(toy_block(0.34, col, letter=letter).rotate_y(rot).place(x=x, z=z, y=y), f"block {letter}")
    sc.add(sphere(0.2).rotate_z(70).rotate_y(30).place(x=-0.25, z=1.35), "ball",
           material("plastic", stripes(0.08, "#ffffff", "#e2483a", axis="y"), roughness=0.25))

    sc.camera = Camera.orbit(target=(0.0, 0.48, 0.2), distance=5.0, azimuth=24, elevation=18, fov=34)
    res = sc.render(cv, log=log, background=[(0, BG_TOP), (1, BG_BOTTOM)])

    log("post")
    res.fog(cv, BG_BOTTOM, density=FOG, start=4.0)
    res.depth_of_field(cv, focus=(0.0, 0.5, 0.15), blur=BLUR)
    cv.vignette(0.16).grain(0.008, seed=seed)
    return cv
