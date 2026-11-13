# -*- coding: utf-8 -*-
"""
Pictogram catalogue: every pictogram of pixelcanvas.pictograms, grouped by category, labelled (README image).

* Format: 1920 px wide, as tall as the catalogue needs (computed from the number of pictograms at import; tiles
  about 100 px at full size); positions as canvas fractions and sizes from
  the canvas size, so drafts look identical. PAGE picks one group of categories ("all" draws every category).
* Layout: title band on top, then one block per category: a coloured category label and a grid of duotone
  pictograms on soft rounded tiles, each with its name below. Small categories share a shelf side by side.
* Each category is one layer (log step); the header is its own layer.
"""
from pixelcanvas import Canvas
from pixelcanvas.pictograms import sheet, names, categories

PROMPT = "A labelled catalogue of every pictogram, grouped by category, on a clean background."
NAME = "pictogram_sheet"
KIND = "design"
DEFAULT_SEED = 1

BG = "#f6f3ee"
INK = "#1f2a3a"
MUTED = "#566072"
TILE = "#ebe6dd"
COLS = 18                          # tiles per row (1920 px wide: about 100 px per tile)
MARGIN = 0.03                      # left / right margin (fraction of the width)
TOP = 210                          # header height in units of the tile width / 100 (see _layout)
HEAD = 40                          # category label height (same units)
ROW = 122                          # one row of tiles, label included (same units)
GAP = 26                           # space between shelves (same units)
CAT_GAP = 1                        # empty columns between two categories on one shelf
STYLE = "duotone"
PAGE = "all"                       # "all" or a key of PAGES
TITLE = "pixelcanvas pictograms"
SUBTITLE = "pictogram(cv, name, style=...) · silhouette, duotone, line, shaded, sticker, cutout · every one is a mask"
PAGES = {
    "culture": ["music", "arts", "transport", "buildings"],
    "everyday": ["home", "objects", "clothing", "kitchen", "tech", "tools", "food"],
    "living": ["animals", "people", "nature"],
    "symbols": ["sports", "science", "medical", "work", "money", "holidays", "symbols"],
}
CATEGORY_COLORS = {"music": "#b0452f", "arts": "#7546ad", "transport": "#245f9c", "buildings": "#1f7356",
                   "home": "#8a5a2b", "objects": "#9c3f5f", "clothing": "#7546ad", "kitchen": "#b0452f",
                   "tech": "#245f9c", "tools": "#5d6b2f", "food": "#c0612b", "animals": "#1f7356",
                   "people": "#245f9c", "nature": "#3b7d3a", "sports": "#b0452f", "science": "#3f5fa8",
                   "medical": "#b3303c", "work": "#4f5a6e", "money": "#8a6d1f", "holidays": "#7546ad",
                   "symbols": "#1f2a3a"}


def _order():
    cats = categories()
    if PAGE != "all":
        return [c for c in PAGES[PAGE] if c in cats]
    known = [c for group in PAGES.values() for c in group if c in cats]
    return known + [c for c in cats if c not in known]


def _shelves(cats):
    """Pack categories into shelves: a category that fits in the columns left on a one-row shelf joins it."""
    shelves = []
    for c in cats:
        n = len(names(c))
        rows = -(-n // COLS)
        last = shelves[-1] if shelves else None
        if (last is not None and rows == 1 and last["rows"] == 1
                and last["used"] + CAT_GAP + n <= COLS):
            last["items"].append((c, last["used"] + CAT_GAP, n))
            last["used"] += CAT_GAP + n
        else:
            shelves.append({"rows": rows, "used": min(n, COLS), "items": [(c, 0, n)]})
    return shelves


def _layout(width, cats):
    """Units: one unit = (tile width) / 100, so every length scales with the canvas width."""
    tile = width * (1 - 2 * MARGIN) / COLS
    u = tile / 100.0
    shelves = _shelves(cats)
    total = TOP + sum(HEAD + s["rows"] * ROW + GAP for s in shelves)
    return u, shelves, total


def fit_height(width=1920):
    """Canvas height (px) that holds every pictogram of the catalogue at ``width``."""
    u, _, total = _layout(width, _order())
    return int(-(-total * u // 1)) + 2


DEFAULT_SIZE = (1920, fit_height(1920))          # grows with the catalogue: new pictograms never run off the sheet


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, BG)
    W, H = width, height
    cats = _order()
    u, shelves, total = _layout(W, cats)
    k = min(u, H / total)                         # px per unit: fits the sheet into the canvas height
    tile = 100 * k
    x0 = (W - tile * COLS) / 2

    log("header")
    cv.text(TITLE, (x0, 120 * k), 64 * k, INK, bold=True, anchor="ls")
    cv.text(SUBTITLE, (x0, 168 * k), 26 * k, MUTED, anchor="ls")
    cv.text(f"{sum(len(names(c)) for c in cats)} pictograms", (x0 + tile * COLS, 120 * k), 30 * k, MUTED,
            anchor="rs")

    y = TOP * k
    for sh in shelves:
        for cat, col, n in sh["items"]:
            ns = names(cat)
            log(cat)
            colr = CATEGORY_COLORS.get(cat, INK)
            cx = x0 + col * tile
            cv.text(cat.upper(), (cx + 6 * k, y + HEAD * k * 0.72), 24 * k, colr, bold=True, anchor="ls",
                    tracking=0.12)
            cols = min(n, COLS)
            rows = -(-n // COLS)
            box = (cx / W, (y + HEAD * k) / H, (cx + cols * tile) / W, (y + HEAD * k + rows * ROW * k) / H)
            sheet(cv, ns, box=box, cols=cols, style=STYLE, color=INK, accent=None, tile_color=TILE,
                  label_color=MUTED, detail_color=TILE)
        y += (HEAD + sh["rows"] * ROW + GAP) * k
    return cv
