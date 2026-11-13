# -*- coding: utf-8 -*-
"""Textbook plate: the parts of a Japanese stone lantern, a 3D render labelled with callouts and a height dimension

* Format: 1600x1200
* Focal point: the lantern in the middle, lit softly from the upper left
* Reading order: title -> lantern -> six callouts from the jewel at the top to the base -> height dimension
* Palette: warm off-white page, grey stone with moss, dark ink labels
* Exact text: the title and the six part names
"""
# Read skills/pixelcanvas/references/solid.md (Scene, project) and diagrams.md (Annotator).
import numpy as np

from pixelcanvas import Canvas, sdf, solid
from pixelcanvas.diagram import Annotator

PROMPT = ("A textbook figure of a Japanese stone lantern with callouts naming its parts from the jewel on top to "
          "the base, and its height.")
NAME = "diagram_stone_lantern"
KIND = "diagram"
DEFAULT_SIZE = (1600, 1200)
DEFAULT_SEED = 1

TITLE, SUBTITLE = "Anatomy of a stone lantern", "Kasuga-style tōrō, granite"
PAGE, INK = "#f5f2ea", "#2b2a28"
FIGURE_BOX = (0.3, 0.16, 0.7, 0.95)                      # where the lantern is rendered, canvas fractions
PARTS = [(0.97, "Hōju · jewel", "right"), (0.80, "Kasa · roof", "left"), (0.64, "Hibukuro · fire box", "right"),
         (0.53, "Chūdai · platform", "left"), (0.33, "Sao · post", "right"), (0.05, "Kiso · base", "left")]
HEIGHT_LABEL = "1.8 m"
DIM_X = 0.92                                             # the height dimension, right of the labels


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, PAGE)
    S = height / 1080

    log("title")
    cv.text(TITLE, (0.05 * width, 0.05 * height), 54 * S, INK, anchor="lt", bold=True)
    cv.text(SUBTITLE, (0.05 * width, cv.last_bbox[3] + 14 * S), 28 * S, "#6b675f", anchor="lt")

    log("lantern")
    sc = solid.Scene(lighting="studio", ground=None)
    sc.add(solid.stone_lantern(height=1.0), "stone lantern")
    sc.camera = solid.Camera.frame(sc, azimuth=25, elevation=12, fill=0.9)
    box = tuple(int(f * s) for f, s in zip(FIGURE_BOX, (width, height) * 2))
    res = sc.render(cv, background=False, box=box)

    log("contact shadow")                                # a soft pool under the base, painted behind it
    body = res.mask("stone lantern")
    rows = np.nonzero(body.max(axis=1) > 0.5)[0]
    top, bottom = rows[0], rows[-1]                      # the lantern's silhouette, top to bottom in pixels
    X, Y = cv.xy()
    pool = sdf.fill(sdf.sd_ellipse(X, Y, res.project((0, 0, 0))[0], bottom - 0.03 * height, 0.17 * width,
                                   0.03 * height), aa=40 * S)
    cv.paint("#5a5246", pool * (1 - body) * 0.4)

    log("labels")                                        # world points on the lantern's axis -> canvas pixels
    ann = Annotator(cv, theme="light", leader="dot")
    for y, text, side in PARTS:
        ann.callout(res.project((0, y, 0)), text, side=side)
    ann.dimension((DIM_X * width, bottom), (DIM_X * width, top), HEIGHT_LABEL)
    ann.draw(log=log)
    return cv
