# -*- coding: utf-8 -*-
"""
The clay zoo (clay_zoo) — a field-guide sheet of thirty pixelcanvas.clay animals.

Every animal stands on its own little round pedestal and is placed with one call (``animal(name, pose=...)``):
pets, farm animals, savanna and forest animals, birds, sea life, small creatures, dinosaurs, a dragon and a
unicorn, in natural poses (sit, walk, trot, graze, beg, lie, fly, swim). Each cell is its own small 3D render
into a box of the sheet (``sc.render(cv, box=cell, background=False)``), so every animal is framed the same way.
Light: a warm key from the upper left, soft shadows on the pedestals, warm paper behind.
"""
from pixelcanvas import Canvas
from pixelcanvas.solid import Scene, Camera, Lighting, cylinder, material
from pixelcanvas.clay import animal

NAME = "clay_zoo"
KIND = "design"
DEFAULT_SIZE = (1920, 1600)
DEFAULT_SEED = 4

PAPER = [(0.0, "#f6eedf"), (1.0, "#eee2cc")]
INK = "#4a3a2e"
SOFT = "#6e5a48"
RULE = "#dccbb0"
PEDESTAL = "#e8d6bf"
TITLE = "The Clay Zoo"
SUBTITLE = "thirty animals, one call each — pixelcanvas.clay"
COLS, ROWS = 6, 5
YAW = -38

# (species, pose, phase, options, label)
SHEET = [
    ("cat", "sit", 0.0, dict(coat="calico"), "Calico cat"),
    ("corgi", "play_bow", 0.0, {}, "Corgi"),
    ("labrador", "walk", 0.2, {}, "Labrador"),
    ("fox", "sit", 0.0, {}, "Red fox"),
    ("rabbit", "stand", 0.0, {}, "Rabbit"),
    ("squirrel", "beg", 0.0, {}, "Squirrel"),
    ("pig", "stand", 0.0, {}, "Pig"),
    ("sheep", "graze", 0.0, {}, "Sheep"),
    ("cow", "stand", 0.0, {}, "Cow"),
    ("horse", "trot", 0.15, {}, "Horse"),
    ("zebra", "walk", 0.4, {}, "Zebra"),
    ("giraffe", "stand", 0.0, {}, "Giraffe"),
    ("elephant", "walk", 0.1, {}, "Elephant"),
    ("lion", "lie", 0.0, {}, "Lion"),
    ("tiger", "walk", 0.35, {}, "Tiger"),
    ("panda", "sit", 0.0, {}, "Panda"),
    ("bear", "stand", 0.0, {}, "Brown bear"),
    ("koala", "stand", 0.0, {}, "Koala"),
    ("penguin", "stand", 0.0, {}, "Penguin"),
    ("owl", "stand", 0.0, {}, "Owl"),
    ("duck", "stand", 0.0, {}, "Duck"),
    ("parrot", "stand", 0.0, {}, "Parrot"),
    ("chick", "stand", 0.0, {}, "Chick"),
    ("fish", "swim", 0.3, dict(coat="clown"), "Clownfish"),
    ("turtle", "stand", 0.2, {}, "Turtle"),
    ("octopus", "stand", 0.2, {}, "Octopus"),
    ("frog", "stand", 0.0, {}, "Frog"),
    ("ladybird", "stand", 0.0, {}, "Ladybird"),
    ("triceratops", "walk", 0.3, {}, "Triceratops"),
    ("dragon", "stand", 0.0, {}, "Dragon"),
]


def _cell_scene(spec, lit):
    species, pose, phase, opts, label = spec
    a = animal(species, pose=pose, phase=phase, **opts)
    lo, hi = a.rotate_y(YAW).bounds()
    k = 1.15 / max(hi[0] - lo[0], hi[2] - lo[2], (hi[1] - lo[1]) * 0.9)
    a = animal(species, pose=pose, phase=phase, size=a.height * k, **opts).rotate_y(YAW)
    lo, hi = a.bounds()
    sc = Scene(lighting=lit, ground=None)
    ped_h = 0.06
    ped_r = max(0.3, min(0.62, 0.5 * max(hi[0] - lo[0], hi[2] - lo[2]) * 0.85))
    sc.add(cylinder(ped_r, ped_h, round=0.02).at(0, ped_h / 2, 0), "pedestal", material("clay", PEDESTAL,
                                                                                       roughness=0.6))
    sc.add(a.place(x=0, z=0, y=ped_h if lo[1] < 0.2 else ped_h + 0.05), label)
    sc.camera = Camera.frame(sc, azimuth=0, elevation=16, fov=30, fill=0.86)
    return sc


def render(width=1920, height=1600, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    S = height / 1600
    log("paper")
    cv.gradient(PAPER)
    log("title")
    cv.text(TITLE, (width / 2, 46 * S), 70 * S, color=INK, anchor="mt", bold=True)
    cv.text(SUBTITLE, (width / 2, 128 * S), 28 * S, color=SOFT, anchor="mt")
    top, bottom, side = 190 * S, 30 * S, 40 * S
    cw = (width - 2 * side) / COLS
    chh = (height - top - bottom) / ROWS
    lit = Lighting("studio", sun=(-40, 44), sun_color="#fff0de", sun_power=1.3, softness=0.6, ambient=0.44,
                   zenith="#f2e8dc", horizon="#efe2d0", ground_color="#c8b49a", rim_power=0.5)
    log("rules")
    from pixelcanvas.masks import line_mask
    for r in range(1, ROWS):
        y = top + r * chh
        cv.paint(RULE, line_mask(cv.h, cv.w, [(side + 20 * S, y), (width - side - 20 * S, y)], 1.5 * S))
    for i, spec in enumerate(SHEET[:COLS * ROWS]):
        r, c = divmod(i, COLS)
        x0, y0 = side + c * cw, top + r * chh
        box = (x0 + 6 * S, y0 + 4 * S, x0 + cw - 6 * S, y0 + chh - 44 * S)
        sc = _cell_scene(spec, lit)
        sc.render(cv, log=log, box=box, background=False)
        cv.text(spec[4], (x0 + cw / 2, y0 + chh - 40 * S), 25 * S, color=INK, anchor="mt", bold=True)
    log("post")
    cv.grain(0.006, seed=seed)
    return cv
