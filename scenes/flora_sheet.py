# -*- coding: utf-8 -*-
"""
Botanical plate (flora_sheet): every tree, shrub, ground plant, flower, potted plant, leaf and rock preset of
pixelcanvas.flora, each drawn with one ``plant(cv, name, ...)`` call and labelled with the name you would type.

Layout: paper background, one section per kind (title rule + grid of cells); each cell plants its preset on the
cell's baseline with a ground shadow, light from the upper left for the whole plate.
"""
import numpy as np

from pixelcanvas import Canvas, fbm2d, mix
from pixelcanvas.flora import plant, LIGHT

NAME = "flora_sheet"
DEFAULT_SIZE = (2400, 3500)
DEFAULT_SEED = 3

PAPER = "#f1eadb"
INK = "#3b3226"
RULE = "#b9a98c"
SECTIONS = [
    ("Trees", ["oak", "maple", "birch", "poplar", "cypress", "spruce", "fir", "pine", "willow", "cherry", "apple",
               "olive", "baobab", "bonsai", "palm", "coconut palm", "winter", "snowy tree", "dead", "autumn maple",
               "autumn oak", "autumn birch", "old oak", "cute", "bamboo", "saguaro", "prickly pear", "banana"], 10, 0.15, 0.115),
    ("Shrubs and ground plants", ["bush", "flowering bush", "berry bush", "hedge", "topiary ball", "topiary cone",
                                  "topiary spiral", "topiary tiers", "grass", "dry grass", "reeds", "cattails", "fern",
                                  "mushroom", "mushrooms", "pumpkin", "lily pads", "mossy rock"], 9, 0.095, 0.066),
    ("Flowers", ["rose", "tulip", "sunflower", "daisy", "poppy", "cosmos", "lily", "orchid", "lavender", "dandelion",
                 "dandelion flower", "lotus", "cherry blossom", "cornflower", "marigold", "hibiscus", "peony",
                 "bouquet", "vase"], 10, 0.112, 0.07),
    ("Potted plants", ["potted monstera", "potted snake plant", "potted succulent", "potted ficus", "potted pothos",
                       "potted fern", "potted cactus", "potted aloe", "potted flowering"], 9, 0.11, 0.085),
    ("Leaves and rocks", ["ovate leaf", "oak leaf", "maple leaf", "ginkgo leaf", "monstera leaf", "palm leaf",
                          "birch leaf", "heart leaf", "rock", "sandstone", "rock pile"], 11, 0.085, 0.055),
]
SCALE = {"oak": 0.9, "cherry": 0.72, "apple": 0.85, "olive": 0.85, "willow": 0.85, "baobab": 0.9, "autumn oak": 0.85,
         "autumn maple": 0.9, "hedge": 0.8, "lily pads": 1.05, "old oak": 0.75, "rock pile": 0.85, "bouquet": 0.85}
TITLE = "pixelcanvas.flora"
SUBTITLE = "one call per plant:  plant(cv, name, at=(u, v), height=h)"


def render(width=2400, height=3500, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    cv = Canvas(W, H, PAPER)
    U = W / 2400.0                                   # layout unit (the plate is designed at 2400 px wide)

    log("paper")
    tex = fbm2d(H, W, 260 * U, 4, seed=seed) * 0.5 + fbm2d(H, W, 6 * U + 1, 2, seed=seed + 1) * 0.5
    cv.paint(mix([(0, "#e9e0cc"), (0.5, PAPER), (1, "#f7f2e6")], tex * 0.5 + 0.5))

    log("title")
    cv.text(TITLE, (W / 2, 70 * U), size=max(int(64 * U), 6), color=INK, anchor="mm", font=None, bold=True)
    cv.text(SUBTITLE, (W / 2, 128 * U), size=max(int(28 * U), 5), color="#6a5c48", anchor="mm")

    y = 175 * U
    k = 0
    for title, names, cols, cell_h, plant_h in SECTIONS:
        log(title)
        cv.text(title, (60 * U, y + 22 * U), size=max(int(34 * U), 5), color=INK, anchor="lm", bold=True)
        cv.paint(RULE, _rule(H, W, y + 48 * U, 60 * U, W - 60 * U, max(1.5 * U, 0.6)))
        y += 64 * U
        cw = (W - 120 * U) / cols
        ch = cell_h * 2400 * U
        rows = (len(names) + cols - 1) // cols
        for i, name in enumerate(names):
            r, c = divmod(i, cols)
            x0 = 60 * U + c * cw
            y0 = y + r * ch
            base_v = (y0 + ch - 48 * U) / H
            hh = plant_h * 2400 * U / H
            if name.endswith(" leaf"):                       # a leaf is placed by its centre, not its base
                base_v -= hh * SCALE.get(name, 1.0) * 0.38
            if name == "lily pads":                           # pads spread below the base point
                base_v -= hh * 0.3
            plant(cv, name, at=((x0 + cw / 2) / W, base_v), height=hh * SCALE.get(name, 1.0), seed=seed + k,
                  shadow=True, light=LIGHT)
            cv.text(name, (x0 + cw / 2, y0 + ch - 22 * U), size=max(int(21 * U), 4), color="#4d4232", anchor="mm")
            k += 1
        y += rows * ch + 30 * U
    log("post")
    cv.vignette(0.12, 0.75, 0.6)
    return cv


def _rule(H, W, y, x0, x1, w):
    m = np.zeros((H, W), np.float32)
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    m[:] = np.clip(w / 2 + 0.5 - np.abs(yy - y), 0, 1) * ((xx >= x0) & (xx <= x1))
    return m
