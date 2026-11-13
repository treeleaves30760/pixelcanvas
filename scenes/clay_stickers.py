# -*- coding: utf-8 -*-
"""A sticker sheet of nine cute clay animals, each die-cut with a white border and a tiny label (clay + stylize)

* Format: 1600x1600
* Layout: a 3 x 3 grid; each animal is its own small 3D render into its cell, then cut out as a sticker
* Light: a soft key from the upper left on every animal; each sticker casts a short drop shadow
* Palette: peach-to-lilac pastel sheet, the animals' natural colours, white borders, cocoa labels
* Exact text: Panda, Fox, Penguin, Owl, Whale, Rabbit, Cat, Dog, Frog
"""
from pixelcanvas import Canvas, stylize
from pixelcanvas.solid import Scene, Camera, Lighting
from pixelcanvas.clay import animal

PROMPT = ("A sticker sheet of nine cute animals — panda, fox, penguin, owl, whale, rabbit, cat, dog, frog — each "
          "with a white die-cut border and a tiny label, on a pastel background.")
NAME = "clay_stickers"
KIND = "design"
DEFAULT_SIZE = (1600, 1600)
DEFAULT_SEED = 1

SHEET = [(0.0, "#fbe3dc"), (0.55, "#f3e2ee"), (1.0, "#e2dcf4")]
INK = "#5a4036"
CUTE = 1.6             # bigger heads and eyes: the vinyl-toy sticker look
BORDER = 16            # white die-cut border, px at 1080p
# (label, species, pose, options, turn, camera elevation): turned a little to the viewer's left
STICKERS = [
    ("Panda", "panda", "sit", {}, -25, 10), ("Fox", "fox", "sit", {}, 25, 10),
    ("Penguin", "penguin", "stand", {}, -25, 10), ("Owl", "owl", "stand", {}, 20, 10),
    ("Whale", "whale", "swim", {}, -60, 25), ("Rabbit", "rabbit", "sit", {}, -30, 10),
    ("Cat", "cat", "sit", dict(coat="ginger"), 25, 10), ("Dog", "labrador", "sit", {}, -25, 10),
    ("Frog", "frog", "sit", dict(cute=1.3), 20, 14),
]


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    S = height / 1080
    log("sheet")
    cv.gradient(SHEET)

    lit = Lighting("studio", sun=(-40, 45), softness=0.6, ambient=0.5, rim_power=0.5)
    margin = 0.05 * width
    cw, ch = (width - 2 * margin) / 3, (height - 2 * margin) / 3
    cuts = []
    for i, (label, species, pose, opts, turn, elevation) in enumerate(STICKERS):
        r, c = divmod(i, 3)
        x0, y0 = margin + c * cw, margin + r * ch
        sc = Scene(lighting=lit, ground=None)
        sc.add(animal(species, pose=pose, **{"cute": CUTE, **opts}).face(turn), label)
        sc.camera = Camera.frame(sc, azimuth=0, elevation=elevation, fill=0.97)
        res = sc.render(cv, log=log, box=(x0 + 14 * S, y0 + 10 * S, x0 + cw - 14 * S, y0 + ch - 58 * S),
                        background=False)
        cuts.append(res.mask(label))

    log("die-cut borders")
    stylize.apply(cv, "sticker", mask=cuts, border=BORDER)

    log("labels")
    for i, (label, *_rest) in enumerate(STICKERS):
        r, c = divmod(i, 3)
        x, y = margin + (c + 0.5) * cw, margin + (r + 1) * ch - 34 * S
        cv.text(label, (x, y), 26 * S, color=INK, anchor="mm", bold=True, plate="pill")

    log("post")
    cv.grain(0.006, seed=seed)
    return cv
