# -*- coding: utf-8 -*-
"""
Texture swatches: the pixelcanvas.texture catalogue ->every material on a rounded tile, grouped by family.

Layout: title band on top, then one section per family (header + rows of COLS tiles). Every tile is a single
``texture(name, h, w, scale=...)`` call with the same light from the upper left, a soft drop shadow towards the
lower right and a gentle bevel. ``VIEW`` sets how many texture units (1080p pixels) each tile spans, so the tile
shows a natural amount of the material at any canvas size.
"""
import numpy as np

from pixelcanvas import Canvas, sdf
from pixelcanvas.color import F32
from pixelcanvas.filters import gaussian_blur
from pixelcanvas.texture import FAMILIES, texture

NAME = "texture_swatches"
DEFAULT_SIZE = (1920, 2300)
DEFAULT_SEED = 3

COLS = 10
ORDER = ["wood", "stone", "metal", "fabric", "man-made", "organic", "liquid"]
OPAQUE = {"fire"}        # keep the tile behind these even though they carry an alpha map
TITLE = "pixelcanvas.texture"
SUBTITLE = 'every tile is one call:  texture("brick", h, w)'
BG = [(0.0, "#1c1e23"), (1.0, "#111215")]
INK = "#d9d9de"
ACCENT = "#e9b45c"
FAMILY_TITLES = {"wood": "WOOD", "stone": "STONE & GROUND", "fabric": "FABRIC", "metal": "METAL",
                 "man-made": "MAN-MADE", "organic": "ORGANIC", "liquid": "LIQUIDS, SKY & FIRE"}

VIEW = dict(  # texture units across one tile
    planks=800, wood_grain=420, end_grain=420, bark=320,
    cobblestone=520, pebbles=300, flagstone=800, brick=700, stone_wall=700, marble=900, granite=260, concrete=900,
    gravel=200, sand=500, soil=260, dry_earth=600, asphalt=150,
    linen=110, denim=150, tartan=330, stripes=220, polka=240, houndstooth=120, chevron=260, knit=150, felt=300,
    velvet=700, leather=200, carpet=130, rug=420, quilted=300,
    brushed_metal=420, hammered_metal=260, rust=520, galvanized=320, chrome=400, gold_leaf=360, carbon_fiber=160,
    perforated=200, corrugated=260,
    paper=300, cardboard=260, cork=160, crackle_glaze=420, glass_tiles=260, terrazzo=320, damask=420, wallpaper=320,
    tiles=420, subway_tiles=460, hex_tiles=250, herringbone=620, circuit=260, lined_paper=320, graph_paper=300,
    film_grain=110, brushstrokes=420,
    lawn=170, meadow=380, moss=260, fur=160, fur_long=260, fur_curly=190, feathers=280, fish_scales=210,
    reptile_scales=210, leaves=320, hay=320, wicker=220, rattan=240, honeycomb=230,
    water=650, caustics=380, ice=520, snow=650, lava=520, fire=380, mud=520, clouds=1300, starfield=300,
)


def _label(name):
    return name.replace("_", " ")


def render(width=1920, height=2300, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = W / 1920
    cv = Canvas(W, H)

    log("background")
    cv.gradient(BG)

    log("title")
    cv.text(TITLE, (70 * S, 62 * S), 74 * S, INK, bold=True, anchor="la")
    cv.text(SUBTITLE, (74 * S, 160 * S), 30 * S, ACCENT, anchor="la", font=None)
    n_all = sum(len(v) for v in FAMILIES.values())
    cv.text(f"{n_all} materials, relief-lit, any resolution", (W - 70 * S, 160 * S), 26 * S, "#8d8f96",
            anchor="ra")

    margin = 70 * S
    gap = 18 * S
    tile = (W - 2 * margin - (COLS - 1) * gap) / COLS
    label_h = 34 * S
    head_h = 40 * S
    row_h = head_h + tile + label_h + 16 * S
    y0 = 240 * S
    tiles = []
    slot = 0
    for fam in ORDER:
        names = FAMILIES.get(fam, [])
        for k, nm in enumerate(names):
            col, row = slot % COLS, slot // COLS
            x, y = margin + col * (tile + gap), y0 + row * row_h + head_h
            if k == 0 or col == 0:
                tiles.append(("head", fam if k == 0 else fam + "+", x, y - head_h))
            tiles.append(("tile", nm, x, y))
            slot += 1

    log("shadows")
    shadow = np.zeros((H, W), F32)
    tp = int(round(tile))
    r = 14 * S
    ly, lx = np.mgrid[0:tp, 0:tp].astype(F32)
    d_tile = sdf.sd_box(lx, ly, (tp - 1) / 2, (tp - 1) / 2, tp / 2 - 1, tp / 2 - 1, radius=r)
    tmask = sdf.fill(d_tile)
    bevel = sdf.bevel_light(d_tile, 10 * S, strength=0.35)
    off = int(round(8 * S))
    for kind, nm, x, yy in tiles:
        if kind != "tile":
            continue
        x0, y0 = int(round(x)) + off, int(round(yy)) + off
        y1, x1 = min(y0 + tp, H), min(x0 + tp, W)
        if y0 < H and x0 < W:
            shadow[y0:y1, x0:x1] = np.maximum(shadow[y0:y1, x0:x1], tmask[:y1 - y0, :x1 - x0])
    cv.paint("#000000", gaussian_blur(shadow, 9 * S) * 0.75)

    for kind, nm, x, yy in tiles:
        if kind == "head":
            cont = nm.endswith("+")
            fam = nm.rstrip("+")
            if not cont:
                log(f"{fam} tiles")
            cv.text(FAMILY_TITLES.get(fam, fam.upper()), (x, yy + 6 * S), 21 * S, ACCENT, bold=True, anchor="la",
                    tracking=0.18, opacity=0.45 if cont else 1.0)
            continue
        x0, y0 = int(round(x)), int(round(yy))
        m = texture(nm, tp, tp, scale=1080.0 / VIEW.get(nm, 400), seed=seed, maps=True)
        col = m.lit * bevel[..., None]
        a = tmask if m.alpha is None or nm in OPAQUE else tmask * m.alpha
        cv.paste(np.clip(col, 0, 1.6), x0, y0, mask=a)
        cv.text(_label(nm), (x0 + tp / 2, y0 + tp + 7 * S), 17 * S, INK, anchor="mt")

    log("post")
    cv.soft_clip(0.9)
    return cv
