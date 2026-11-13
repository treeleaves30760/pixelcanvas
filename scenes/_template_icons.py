# -*- coding: utf-8 -*-
"""Sheet of icons / logo marks, also exported as SVG (vector)

* Format: 1920x1080 sheet (render.py --svg writes outputs/<name>.svg with the same geometry)
* Focal point: <the custom logo mark in the first card>
* Grid: <COLS columns of equal cards, one icon + its name per card>
* Style: <24-unit grid, 2-unit round strokes, duotone fill in the row accent>
* Palette: <theme THEME; accents cycle per card>
* Exact text: <TITLE and every icon name copied from the request>
"""
# Read skills/pixelcanvas/references/vector.md (Drawing, shapes, paths, icons, groups, SVG).
from pixelcanvas import Canvas, pictograms
from pixelcanvas.vector import Drawing, THEMES

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "design"
NAME = "my_icons"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "light"
# >>> data: new_scene.py --prompt writes the request's title and icons here
TITLE = "Product icons"
ICONS = ["logo", "home", "search", "cart", "user", "bell", "gear", "chart", "cloud", "lock", "mail", "heart"]  # or
# <<< data (label, glyph): glyph = an icon_names() name, "pictogram:<name>" or "logo"
COLS = 6
ICON = 96                           # icon size in drawing units (the drawing is 1080 units tall)


# >>> build: the logo mark, the sheet and render
def logo(dwg, x, y, s, color):
    """The custom mark: replace with your own shapes (path / circle / polygon), drawn in an s x s box at (x, y)."""
    dwg.circle(x, y, 0.5 * s, fill=color)
    dwg.path(f"M{x - 0.24 * s} {y + 0.18 * s} L{x - 0.06 * s} {y - 0.22 * s} L{x + 0.06 * s} {y + 0.02 * s} "
             f"L{x + 0.24 * s} {y - 0.18 * s}", stroke="#ffffff", stroke_width=0.08 * s, cap="round", join="round",
             fill=None)


def build(aspect=16 / 9):
    W = 1080 * aspect                                    # drawing units: 1080 tall, width follows the canvas
    dwg = Drawing(W, 1080, theme=THEME)
    th = dwg.theme
    with dwg.group("title"):
        dwg.text(100, 110, TITLE, size=56, weight="bold", anchor="ls")
    cols = min(COLS, len(ICONS))
    rows, gap = -(-len(ICONS) // cols), 32
    cw, ch = (W - 200 - (cols - 1) * gap) / cols, min(300, (860 - (rows - 1) * gap) / rows)
    s = min(ICON, 0.5 * cw, 0.45 * ch)
    y0 = 180 + (860 - rows * ch - (rows - 1) * gap) / 2                # grid centred under the title
    for i, item in enumerate(ICONS):                     # one group = one layer and one SVG <g> per card
        label, glyph = (item, item) if isinstance(item, str) else item
        x, y = 100 + (i % cols) * (cw + gap), y0 + (i // cols) * (ch + gap)
        with dwg.group(label, shadow=True):
            dwg.rect(x, y, cw, ch, r=th.radius + 8, fill=th.surface, stroke=th.grid)
            cx, cy = x + cw / 2, y + ch / 2 - 18
            if glyph == "logo":
                logo(dwg, cx, cy, s, th.accent(i))
            elif glyph.startswith("pictogram:"):         # any of the 500+ pictograms as a vector glyph
                dwg.embed(pictograms.drawing(glyph[10:], color=th.accent(i)), box=(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))
            else:                                        # built-in names: vector.icon_names()
                dwg.icon(glyph, cx, cy, s, color=th.accent(i), fill=th.background)
            dwg.text(cx, y + ch - 28, label, size=24, fill=th.muted, anchor="ms")
    return dwg


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height, THEMES[THEME].background)
    build(width / height).draw(cv, log=log)              # --svg exports the same drawing
    return cv
# <<< build
