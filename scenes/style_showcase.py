# -*- coding: utf-8 -*-
"""
Style showcase — one picture, twelve media: the mountains_sunset landscape finished with twelve very different
``pixelcanvas.stylize`` styles, laid out as a labelled 4 x 3 grid.

Layout: dark charcoal page, a title band at the top, twelve tiles with their style name underneath.
Each tile is its own log step, so ``--layers`` gives one editable layer per style.
How: the base picture is rendered once at twice the tile size, every style is applied at that size
(``stylize.apply``), then the result is box-downsampled into its tile.
"""
import importlib

import numpy as np

from pixelcanvas import Canvas
from pixelcanvas.filters import resize
from pixelcanvas.stylize import apply

NAME = "style_showcase"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 2026

BASE_SCENE = "mountains_sunset"
STYLES = ["oil", "watercolor", "ink_wash", "ukiyoe",
          "comic", "linocut", "stained_glass", "mosaic",
          "low_poly", "pixel_art", "risograph", "blueprint"]
COLS = 4
TITLE = "One picture, twelve media"
SUBTITLE = "pixelcanvas.stylize  -  apply(cv, name)"
PAGE = "#16161b"
INK = "#ecebe6"
MUTED = "#9a98a6"
OVERSAMPLE = 2.0          # styles run at this multiple of the tile size, then downsample (crisper marks)


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = H / 1080.0
    cv = Canvas(W, H, PAGE)

    rows = -(-len(STYLES) // COLS)
    top, side, gap, lab = 0.13 * H, 0.035 * W, 0.012 * W, 0.042 * H
    tw = (W - 2 * side - (COLS - 1) * gap) / COLS
    th = tw * 9 / 16
    avail = H - top - 0.025 * H
    if rows * (th + lab) + (rows - 1) * gap * 0.5 > avail:          # keep the grid on the page
        th = (avail - (rows - 1) * gap * 0.5) / rows - lab
        tw = th * 16 / 9
    x_start = (W - (COLS * tw + (COLS - 1) * gap)) / 2

    log("title")
    cv.text(TITLE, (W / 2, 0.055 * H), 46 * S, color=INK, anchor="mm", bold=True)
    cv.text(SUBTITLE, (W / 2, 0.098 * H), 21 * S, color=MUTED, anchor="mm")

    log("base picture")
    bw, bh = int(round(tw * OVERSAMPLE)), int(round(th * OVERSAMPLE))
    base_mod = importlib.import_module(f"scenes.{BASE_SCENE}")
    base = np.clip(base_mod.render(bw, bh, getattr(base_mod, "DEFAULT_SEED", seed), lambda m: None).buf, 0, 1)

    for i, name in enumerate(STYLES):
        log(f"style: {name}")
        r, c = divmod(i, COLS)
        x = int(round(x_start + c * (tw + gap)))
        y = int(round(top + r * (th + lab + gap * 0.5)))
        tile = resize(apply(base, name, seed=seed), int(round(tw)), int(round(th)), "box")
        cv.paste(np.clip(tile, 0, 1), x, y)
        cv.text(name, (x + 2 * S, y + th + 0.012 * H), 22 * S, color=INK, anchor="lt", bold=True)
    return cv
