# -*- coding: utf-8 -*-
"""Sheet of twelve app icons for a productivity suite: gradient rounded-square tiles, long shadows, labels

* Format: 1600x1200, a 4 x 3 grid
* Style: one consistent look: soft two-tone gradient per tile, white glyph, long shadow to the lower right
* Exact text: the twelve app names under their tiles
"""
# Read skills/pixelcanvas/references/vector.md (app_icon, app_icon_sheet).
from pixelcanvas import Canvas
from pixelcanvas.vector import app_icon_sheet

PROMPT = ("A sheet of twelve app icons in one consistent style for a productivity suite: notes, calendar, mail, "
          "tasks, files, chat, camera, music, maps, weather, settings, calculator. Rounded-square tiles with soft "
          "gradients and long shadows, each labelled.")
NAME = "vector_app_icons"
KIND = "design"
DEFAULT_SIZE = (1600, 1200)
DEFAULT_SEED = 1

APPS = ["notes", "calendar", "mail", "tasks", "files", "chat", "camera", "music", "maps", "weather", "settings",
        "calculator"]
COLS = 4
SHADOW = "long"                                          # long, soft or none
TILE = 0.72                                              # tile size as a share of its grid cell


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    log("icons")                                         # backdrop, tiles, glyphs, shadows and labels in one call
    app_icon_sheet(cv, APPS, cols=COLS, shadow=SHADOW, tile=TILE)
    return cv
