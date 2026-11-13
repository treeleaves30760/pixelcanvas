# -*- coding: utf-8 -*-
"""Three-panel comic strip: two characters, speech bubbles pointing at their mouths, a caption (figure + text + vector)

* Format: 1920x1080: three panels in a row with gutters, caption box under them
* Focal point: <the punchline bubble in panel 3>
* Panels: <1 meet, 2 reply, 3 punchline>; each panel is a flat colour and both people stand on its floor line
* Light: <flat comic style, no cast light>
* Exact text: <LINES and CAPTION copied verbatim from the request>
"""
# Read skills/pixelcanvas/references/figures.md (person, anchors), typography.md (text), vector.md (shapes).
import numpy as np

from pixelcanvas import Canvas, sdf
from pixelcanvas.figure import person

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "design"
NAME = "my_comic"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

PAPER, INK = "#fbf6ea", "#1f1d24"
BACKS = ("#cfe6f2", "#f6e3c3", "#e3d4ef")               # panel colours, cycled
GUTTER, MARGIN = 30, 50                                 # in 1080p units
# >>> data: new_scene.py --prompt writes the request's panels (one line each) and caption here
LINES = ((0, "Did you see the sky?"), (1, "Looks like rain."), (0, "Good. I brought snacks."))   # (speaker, text)
CAPTION = "Two friends, one cloud."
# <<< data

def box(cv, cx, cy, w, h, color, r=0, edge=0):
    X, Y = cv.xy()
    for c, g in ((INK, edge), (color, 0)):
        if c is color or edge:
            cv.paint(c, sdf.fill(sdf.sd_box(X, Y, cx, cy, w / 2 + g, h / 2 + g, r)))


def bubble(cv, text, mouth, cx, cy, w, h, S):            # rounded box + a tail polygon ending at the mouth
    X, Y = cv.xy()
    for color, g in ((INK, 4 * S), ("#ffffff", 0)):
        tail = [(cx - 0.1 * w - g, cy), (cx + 0.1 * w + g, cy), (mouth[0], mouth[1] - 14 * S)]
        cv.paint(color, np.maximum(sdf.fill(sdf.sd_box(X, Y, cx, cy, w / 2 + g, h / 2 + g, 0.4 * h)),
                                   sdf.fill(sdf.sd_polygon(X, Y, tail))))
    cv.text(text, box=(cx - 0.42 * w, cy - 0.36 * h, cx + 0.42 * w, cy + 0.36 * h), size=0.3 * h, color=INK,
            bold=True, max_width=True, balance=True)


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv, S, n = Canvas(width, height, PAPER), height / 1080, len(LINES)      # one panel per line
    pw = (width - 2 * MARGIN * S - (n - 1) * GUTTER * S) / n
    top, bot = MARGIN * S, 0.84 * height
    who = [person("woman", style="line", view="three_quarter", facing=1, height=0.5,
                  hair=dict(style="bob", color="#3a2618"), outfit=dict(top="sweater", colors=dict(top="#e3a72f")),
                  face=dict(expression="smile")),
           person("man", style="line", view="three_quarter", facing=-1, height=0.54, skin="deep",
                  outfit=dict(top="hoodie", colors=dict(top="#2a9d8f")), face=dict(expression="laugh"))]
    for i, (speaker, text) in enumerate(LINES):
        x0 = MARGIN * S + i * (pw + GUTTER * S)
        log(f"panel {i + 1}")                             # panel, both people, then the speaker's bubble
        box(cv, x0 + pw / 2, (top + bot) / 2, pw, bot - top, BACKS[i % 3], edge=4 * S)
        for k, f in enumerate(who):
            f.pose("talk" if k == speaker else "stand").draw(cv, at=((x0 + pw * (0.28 + 0.44 * k)) / width, 0.8))
        gap = (GUTTER if i < n - 1 else MARGIN) * S - 6 * S      # shadows stay inside the panel: repaper the gutter
        box(cv, x0 + pw + 6 * S + gap / 2, height / 2, gap, height, PAPER)
        mx, my = who[speaker].anchor("mouth")                 # the tail points at the speaker's mouth
        bubble(cv, text, (mx, my), min(max(mx, x0 + 0.46 * pw), x0 + 0.54 * pw), top + 0.1 * (bot - top) + 45 * S,
               0.82 * pw, 90 * S, S)
    log("caption"); box(cv, width / 2, 0.925 * height, 0.5 * width, 0.07 * height, INK, r=6 * S)
    cv.text(CAPTION, (width / 2, 0.925 * height), 38 * S, "#ffffff", anchor="mm", bold=True)
    return cv
