# -*- coding: utf-8 -*-
"""
Icon sheet: every built-in vector icon on a tidy grid with labels (pixelcanvas.vector).

* Format: 1920x1080; one Drawing in 1920x1080 user units, scaled to the canvas (drafts look identical).
* Layout: header (title + subtitle) in the top band, then a COLS-column grid of cards; each card holds one icon
  (ICON px, 24-unit grid, 2-unit round strokes) and its name.
* Style: THEME surface cards on the theme background (any of vector.THEMES); icons in the row accent with a soft duotone body tint
  (DUOTONE=False for pure line icons). Each card is its own layer (one top-level group per icon).
* The same Drawing is exported as SVG by tools (``vector.last_drawings()``).
"""
from pixelcanvas import Canvas
from pixelcanvas.vector import Drawing, icon_names, THEMES

NAME = "vector_icons"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "dark"
COLS = 8
ICON = 54                 # icon size in user units (drawing is 1920 x 1080)
CARD_W, CARD_H = 194, 124
GAP = 16
DUOTONE = True
ACCENT_LINES = True       # icon strokes in the row accent (False: theme ink)
TINT = 0.20               # duotone body: accent mixed into the card colour
TITLE = "pixelcanvas icons"
SUBTITLE = "24-unit grid · 2-unit round strokes · dwg.icon(name, x, y, size)"


def build():
    th = THEMES[THEME]
    dwg = Drawing(1920, 1080, theme=THEME, name="icons")
    names = icon_names()
    rows = (len(names) + COLS - 1) // COLS
    grid_w = COLS * CARD_W + (COLS - 1) * GAP
    grid_h = rows * CARD_H + (rows - 1) * GAP
    x0 = (1920 - grid_w) / 2
    y0 = 176 + (1080 - 176 - 56 - grid_h) / 2

    with dwg.group("header"):
        dwg.text(x0, 86, TITLE, size=50, weight="bold", anchor="ls")
        dwg.text(x0, 126, SUBTITLE, size=23, fill=th.muted, anchor="ls")
        dwg.text(x0 + grid_w, 86, f"{len(names)} icons", size=26, fill=th.muted, anchor="rs")
        dwg.line(x0, 152, x0 + grid_w, 152, stroke=th.grid, stroke_width=2)

    for i, name in enumerate(names):
        r, c = divmod(i, COLS)
        x = x0 + c * (CARD_W + GAP)
        y = y0 + r * (CARD_H + GAP)
        acc = th.accent(r)
        with dwg.group(name, shadow=True):
            dwg.rect(x, y, CARD_W, CARD_H, r=th.radius + 4, fill=th.surface, stroke=th.grid, stroke_width=1)
            dwg.icon(name, x + CARD_W / 2, y + 50, ICON, color=acc if ACCENT_LINES else th.ink, stroke_width=2,
                     fill=(_tint(acc, th.surface, TINT) if DUOTONE else None))
            dwg.text(x + CARD_W / 2, y + CARD_H - 17, name, size=19, fill=th.muted, anchor="ms")
    return dwg


def _tint(color, base, amount):
    """``amount`` of ``color`` mixed into ``base`` (soft duotone fill)."""
    c = [int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(base.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(int(round(q + (p - q) * amount)) for p, q in zip(c, b))


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, THEMES[THEME].background)
    build().draw(cv, log=log)
    return cv
