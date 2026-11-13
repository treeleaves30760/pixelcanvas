# -*- coding: utf-8 -*-
"""
Backdrop catalogue: every backdrop of pixelcanvas.backdrop as a labelled rounded 16:9 tile (README image).

* Format: 1920x1380; all positions in 1920-wide design units scaled to the canvas, so drafts look identical.
* Layout: title band on top, then a COLS-column grid in family order (gradient, geometric, scenic, organic);
  each tile shows the backdrop in its own suggested palette, with its name and family underneath.
* Each family is one layer (log step); the header is its own layer.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas.backdrop import render as bd_render, names, BACKDROPS
from pixelcanvas.color import clamp01

NAME = "backdrop_sheet"
DEFAULT_SIZE = (1920, 1380)
DEFAULT_SEED = 1

BG = "#101318"
INK = "#eef1f5"
MUTED = "#8d96a3"
COLS = 8
MARGIN = 56
GAP_X = 18
GAP_Y = 54
TOP = 170
RADIUS = 14
TITLE = "pixelcanvas backdrops"
SUBTITLE = "one call each · any size · any palette · backdrop(cv, name, palette=..., seed=...)"
FAMILY_COLORS = {"gradient": "#b9a4ff", "geometric": "#7fd6c2", "scenic": "#ffb38a", "organic": "#e6c86e"}


def _rounded(w, h, r):
    """Anti-aliased rounded-rectangle mask of size (h, w)."""
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    qx = np.abs(X + 0.5 - w / 2) - w / 2 + r
    qy = np.abs(Y + 0.5 - h / 2) - h / 2 + r
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return clamp01(0.5 - d)


def render(width=1920, height=1380, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    k = width / 1920.0
    cv = Canvas(width, height, BG)

    log("header")
    cv.text(TITLE, (MARGIN * k, 92 * k), 54 * k, INK, bold=True, anchor="ls")
    cv.text(SUBTITLE, (MARGIN * k, 132 * k), 23 * k, MUTED, anchor="ls")
    cv.text(f"{len(BACKDROPS)} backdrops", ((1920 - MARGIN) * k, 92 * k), 26 * k, MUTED, anchor="rs")
    x = (1920 - MARGIN) * k
    for fam in reversed(list(FAMILY_COLORS)):
        cv.text(fam, (x, 132 * k), 21 * k, FAMILY_COLORS[fam], anchor="rs")
        x = cv.last_bbox[0] - 26 * k

    tw = (1920 - 2 * MARGIN - (COLS - 1) * GAP_X) / COLS
    th = tw * 9 / 16
    i = 0
    for fam in ("gradient", "geometric", "scenic", "organic"):
        log(f"{fam} tiles")
        for nm in names(fam):
            r, cidx = divmod(i, COLS)
            x0 = (MARGIN + cidx * (tw + GAP_X)) * k
            y0 = (TOP + r * (th + GAP_Y)) * k
            w, h = int(round(tw * k)), int(round(th * k))
            img, P = bd_render(nm, w, h, seed=seed)
            cv.paste(img, int(round(x0)), int(round(y0)), mask=_rounded(w, h, RADIUS * k))
            cv.text(nm.replace("_", " "), (x0 + 2 * k, y0 + h + 27 * k), 19 * k, INK, bold=True, anchor="ls")
            cv.text(P.name.replace("_", " "), (x0 + w - 2 * k, y0 + h + 27 * k), 16 * k,
                    FAMILY_COLORS[fam], anchor="rs")
            i += 1
    log("post")
    return cv
