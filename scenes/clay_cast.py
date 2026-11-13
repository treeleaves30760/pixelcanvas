# -*- coding: utf-8 -*-
"""
Meet the cast (clay_cast) — the pixelcanvas.clay character showcase.

Ten ready-made clay characters in a gentle arc on a seamless studio sweep, each placed with one call
(``cast(name, pose=...)``) in its own pose, outfit and build: an explorer kid, a chef, a doctor, an astronaut, a
grandma, a wizard, a knight, a princess, a robot friend and a superhero. A warm key light from the upper left, a
soft fill, a gentle rim, contact shadows on the sweep. Name labels sit under each figure (projected from 3D), and
the explorer's shirt carries a printed word that stays on the shirt (``print_on``).
"""
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting
from pixelcanvas.clay import cast, print_on

NAME = "clay_cast"
KIND = "design"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 3

SWEEP = "#f0dcc8"
BG_TOP, BG_BOTTOM = "#f6e7d6", "#eed8c2"
INK = "#5a4436"
TITLE = "Meet the cast"
SUBTITLE = "ten clay characters, one call each"
SPACING = 0.6
ARC = 0.9

# (cast name, pose, phase, yaw toward camera, label)
LINEUP = [
    ("explorer", "wave", 0.25, 20, "Explorer"),
    ("chef", "hold", 0.0, 14, "Chef"),
    ("doctor", "arms_crossed", 0.0, 8, "Doctor"),
    ("astronaut", "cheer", 0.0, 4, "Astronaut"),
    ("grandma", "hold", 0.0, 2, "Grandma"),
    ("wizard", "hold", 0.0, -2, "Wizard"),
    ("knight", "hold", 0.0, -6, "Knight"),
    ("princess", "contrapposto", 0.0, -10, "Princess"),
    ("robot_friend", "wave", 0.6, -14, "Robot"),
    ("superhero", "hands_on_hips", 0.0, -20, "Superhero"),
]
HELD = {"chef": ("spoon", {}), "grandma": ("bouquet", {})}


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    lit = Lighting("studio", sun=(-38, 40), sun_color="#fff0e0", sun_power=1.3, softness=0.62, ambient=0.42,
                   zenith="#f2e6dc", horizon="#f0dccb", ground_color="#c8a890", rim_power=0.55, fill_power=0.18)
    sc = Scene(lighting=lit, ground=None)
    sc.backdrop(SWEEP, distance=2.2, radius=1.6)
    n = len(LINEUP)
    spots = []
    for i, (name, pose, phase, yaw, label) in enumerate(LINEUP):
        t = (i - (n - 1) / 2) / ((n - 1) / 2)
        x = t * (n - 1) / 2 * SPACING
        z = -ARC * (1 - t * t) * 0.5 + 0.25
        hold = HELD.get(name)
        ch = cast(name, pose=pose, phase=phase, **({"hold": None} if hold else {}))
        if hold:
            ch.hold(hold[0], **hold[1])
        ch.face(yaw)
        if name == "explorer":
            ch.express("laugh")
        placed = ch.place(x=x, z=z)
        sc.add(placed, label)
        spots.append((x, z, label, placed))
    lo, hi = sc.bounds()
    box = (lo - [0.04, 0.3, 0], hi + [0.04, 0.36, 0])         # room for the labels below and the title above
    sc.camera = Camera.frame(box, azimuth=0, elevation=6, fov=28, fill=0.98)
    res = sc.render(cv, log=log, background=[(0, BG_TOP), (1, BG_BOTTOM)])
    print_on(cv, res, spots[0][3], "HI", where="belly", color="#fff6e8", size=0.08, layer="Explorer", log=log)
    log("labels")
    S = height / 1080
    for x, z, label, _ in spots:
        px, py = sc.project((x, 0.0, z + 0.32))
        cv.text(label, (px, py + 14 * S), 30 * S, color=INK, anchor="mt", bold=True)
    log("title")
    cv.text(TITLE, (width / 2, 70 * S), 64 * S, color=INK, anchor="mt", bold=True)
    cv.text(SUBTITLE, (width / 2, 148 * S), 30 * S, color="#6e5444", anchor="mt")
    log("post")
    cv.vignette(0.12).grain(0.006, seed=seed)
    return cv
