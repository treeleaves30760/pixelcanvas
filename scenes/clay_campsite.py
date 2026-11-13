# -*- coding: utf-8 -*-
"""A clay hiker and her dog at a campsite under the Milky Way (clay + solid + elements)

* Format: 1920x1080
* Focal point: the campfire just right of centre; the hiker sits on a log warming her hands, the dog lies beside her
* Light: the fire and the glowing tent are the only warm lights; a cool moonlit night fills the rest
* Far / middle / near: Milky Way and moon / pines on the dark hills / tent, fire, hiker, dog on the grass
* Palette: navy night, amber firelight, orange tent, moss green
"""
from pixelcanvas import Canvas
from pixelcanvas.elements import sky, stars, moon, embers
from pixelcanvas.solid import Scene, Camera, Lighting, tent, tree, cylinder, material
from pixelcanvas.clay import Character, animal, prop

PROMPT = ("A clay hiker and her dog at a campsite at night: an orange tent glowing from inside, a campfire with "
          "rising sparks, pine trees and the Milky Way overhead.")
NAME = "clay_campsite"
KIND = "painting"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 3

HORIZON = 0.42
TENT = "#e8823a"
GRASS = "#2f3d2c"
FIRE_POWER = 2.4
TENT_POWER = 1.2
PINES = [(-5.0, -7.0, 2.8), (-3.6, -8.5, 3.4), (-2.0, -7.6, 2.4), (2.6, -8.0, 3.0), (4.0, -6.8, 2.6),
         (5.2, -9.0, 3.2), (-6.6, -8.2, 3.2), (0.8, -9.5, 2.6), (-7.6, -6.4, 2.6)]


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    log("night sky")
    sky(cv, "night", horizon=HORIZON)
    log("Milky Way")
    stars(cv, horizon=HORIZON, seed=seed)
    log("moon")
    moon(cv, at=(0.82, 0.16), size=34)

    sc = Scene(lighting=Lighting("night", ambient=0.35), ground=GRASS)
    camp = sc.add(tent(TENT, lit=True).rotate_y(25).place(x=-1.9, z=-1.1), "tent")
    sc.light_from(camp, power=TENT_POWER, radius=1.6)
    fire = sc.add(prop("campfire").scale(1.4).place(x=0.6, z=0.3), "campfire")
    sc.light_from(fire, power=FIRE_POWER, radius=2.4)
    log_seat = sc.add(cylinder(0.17, 1.1).rotate_x(90).rotate_y(-35).at(-0.35, 0.17, 0.65), "log", material("wood", "#6a4a32"))
    hiker = Character("Mara", build="adult", skin=2, hair=dict(style="ponytail", color="auburn"),
                      face=dict(expression="happy"), outfit="hiking")
    sc.add(hiker.pose("sit").face(70).sit_on(None, height=0.34).place(x=-0.35, z=0.65), "hiker")
    sc.add(animal("husky", pose="lie").face(-100).place(x=1.6, z=1.0), "dog")
    for i, (x, z, h) in enumerate(PINES):
        sc.add(tree(h, kind="pine", leaves="#1f2e26", seed=i).place(x=x, z=z), f"pine {i + 1}")

    sc.camera = Camera.orbit(target=(0.0, 0.8, 0.2), distance=4.8, azimuth=6, elevation=6, fov=36)
    res = sc.render(cv, log=log, background=False)

    log("haze")
    res.fog(cv, "#1a2238", density=0.06, start=6.0, sky=False)

    log("fire glow")
    fx, fy = sc.project((0.6, 0.3, 0.3))
    cv.glow(fx, fy, 0.22 * height, "#ff9a40", 0.35, mode="add")

    log("sparks")
    embers(cv, at=(fx / width, fy / height - 0.12), n=70, seed=seed)

    log("post")
    cv.bloom(0.9, 12, 0.3).soft_clip(0.82).vignette(0.25).grain(0.008, seed=seed)
    return cv
