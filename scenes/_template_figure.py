# -*- coding: utf-8 -*-
"""Labelled textbook figure: hand-drawn shapes with leader-line callouts (sdf + diagram)

* Format: 1920x1080 (the figure sits in the middle third; labels stack in columns left and right)
* Subject: <an animal cell: membrane, cytoplasm, nucleus, nucleolus, mitochondria, vacuole>
* Labels: <one callout per part, exact textbook names>
* Line style: <ink outline, slightly wobbly like a pen drawing; flat pastel fills>
* Palette: <one pastel per part, dark ink>
* Exact text: <TITLE and every label copied from the request>
"""
# Read skills/pixelcanvas/references/diagrams.md (Annotator, callouts) and typography.md section 10 (sdf shapes).
from pixelcanvas import Canvas, sdf, fbm2d, manifest
from pixelcanvas.diagram import Annotator

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "diagram"
NAME = "my_figure"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

INK, PAPER = "#2a2f3a", "#fbfaf6"
WOBBLE = 6                          # hand-drawn wobble of every outline, px at 1080p (0 = perfect geometry)
# >>> data: new_scene.py --prompt writes the request's title and named parts here
TITLE = "Animal cell"
PARTS = [  # label, (kind, centre u, v, half-width, half-height) as fractions of the square, fill
    ("Cell membrane", ("ellipse", 0.5, 0.55, 0.42, 0.33), "#f6d6cf"),
    ("Vacuole", ("ellipse", 0.62, 0.41, 0.07, 0.06), "#cfe3f2"),
    ("Nucleus", ("ellipse", 0.44, 0.52, 0.13, 0.11), "#c9b6e4"),
    ("Nucleolus", ("ellipse", 0.42, 0.5, 0.035, 0.03), "#7f62b0"),
    ("Mitochondrion", ("capsule", 0.62, 0.66, 0.06, 0.025), "#f2b36b"),
]
# <<< data


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, PAPER)
    X, Y = cv.xy()
    S, Q = height / 1080, min(height, 0.62 * width)       # shapes live on a centred Q x Q square, labels beside it
    ox, oy = (width - Q) / 2, (height - Q) / 2 + 0.04 * height
    wob = fbm2d(height, width, 60 * S, 3, seed=seed) * WOBBLE * S
    ann = Annotator(cv, theme="light")
    for label, (kind, u, v, a, b), color in PARTS:      # one layer per part; add rows to PARTS for more
        log(label.lower())
        x, y, a, b = ox + u * Q, oy + v * Q, a * Q, b * Q
        d = (sdf.sd_ellipse(X, Y, x, y, a, b) if kind == "ellipse"
             else sdf.sd_segment(X, Y, x - a, y, x + a, y, b)) + wob
        cv.paint(color, sdf.fill(d))
        cv.paint(INK, sdf.outline(d, 2.5 * S))
        manifest.note(cv, "object", label.lower(), mask=sdf.fill(d))
        ann.callout((x + 0.7 * a, y - 0.7 * b), label)          # leader to the upper-right edge
    log("title")
    cv.text(TITLE, (width / 2, 0.08 * height), 54 * S, INK, anchor="mm", bold=True)
    log("labels")                                        # callouts stack beside the figure, leader lines never cross
    ann.scale_bar((ox, oy + 0.9 * Q), 0.2 * Q, "10 \u00b5m")
    ann.draw()
    return cv
