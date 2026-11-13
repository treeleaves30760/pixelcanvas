# -*- coding: utf-8 -*-
"""
Palette library sheet: every curated palette of pixelcanvas.palette as a labelled swatch strip, grouped by family,
each with a tiny poster thumbnail that shows its roles in use (bg / fg / accent / muted).

* Format: 1920x1200; one column per family (mood, subject, era, ui); positions in 1920-wide design units.
* Row: name, swatch strip (one rounded segment per colour, in the palette's own order), poster thumbnail.
* Layers: header, one per family column.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas.color import clamp01, rgb
from pixelcanvas.palette import FAMILIES, palette

NAME = "palette_sheet"
DEFAULT_SIZE = (1920, 1200)
DEFAULT_SEED = 1

BG = "#f4f2ee"
INK = "#1b1d22"
MUTED = "#6b7078"
EDGE = "#d3cec4"
TITLE = "pixelcanvas palettes"
SUBTITLE = "palette(name) → bg · fg · accent · muted · ramp(n) · text_on(colour)"
FAMILY_TITLES = {"mood": "Mood", "subject": "Subject", "era": "Era & school", "ui": "UI (light / dark)"}
MARGIN = 56
COL_GAP = 40
TOP = 190
ROW_H = 53


def _rr(w, h, r):
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    qx = np.abs(X + 0.5 - w / 2) - w / 2 + r
    qy = np.abs(Y + 0.5 - h / 2) - h / 2 + r
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return clamp01(0.5 - d)


def _box(cv, x0, y0, x1, y1, color, r=0.0):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    block = np.broadcast_to(rgb(color)[:3], (h, w, 3))
    cv.paste(block, x0, y0, mask=_rr(w, h, r) if r else None)


def _poster(cv, p, x, y, w, h, k):
    """A tiny poster: bg card, accent disc, fg headline bars, muted body line."""
    _box(cv, x, y, x + w, y + h, p.bg, 6 * k)
    yy = np.mgrid[0:int(h), 0:int(w)][0]
    d = np.hypot(np.mgrid[0:int(h), 0:int(w)][1] - w * 0.74, yy - h * 0.42)
    disc = clamp01(h * 0.27 - d + 0.5) * _rr(int(w), int(h), 6 * k)
    cv.paste(np.broadcast_to(rgb(p.accent)[:3], disc.shape + (3,)), int(round(x)), int(round(y)), mask=disc)
    _box(cv, x + w * 0.1, y + h * 0.2, x + w * 0.52, y + h * 0.34, p.fg, 2 * k)
    _box(cv, x + w * 0.1, y + h * 0.42, x + w * 0.42, y + h * 0.53, p.fg, 2 * k)
    _box(cv, x + w * 0.1, y + h * 0.7, x + w * 0.88, y + h * 0.78, p.muted, 1.5 * k)


def render(width=1920, height=1200, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    k = width / 1920.0
    cv = Canvas(width, height, BG)

    log("header")
    total = sum(len(v) for v in FAMILIES.values())
    cv.text(TITLE, (MARGIN * k, 92 * k), 54 * k, INK, bold=True, anchor="ls")
    cv.text(SUBTITLE, (MARGIN * k, 132 * k), 23 * k, MUTED, anchor="ls")
    cv.text(f"{total} palettes", ((1920 - MARGIN) * k, 92 * k), 26 * k, MUTED, anchor="rs")

    cw = (1920 - 2 * MARGIN - 3 * COL_GAP) / 4
    name_w, poster_w, poster_h = 128, 62, 40
    strip_w = cw - name_w - poster_w - 16
    for ci, fam in enumerate(("mood", "subject", "era", "ui")):
        log(f"{fam} palettes")
        x0 = MARGIN + ci * (cw + COL_GAP)
        cv.text(FAMILY_TITLES[fam], (x0 * k, (TOP - 14) * k), 22 * k, INK, bold=True, anchor="ls")
        _box(cv, x0 * k, (TOP - 4) * k, (x0 + cw) * k, (TOP - 2) * k, "#d6d2ca")
        for ri, nm in enumerate(FAMILIES[fam]):
            p = palette(nm)
            y = TOP + 12 + ri * ROW_H
            cv.text(nm.replace("_", " "), (x0 * k, (y + poster_h * 0.68) * k), 18 * k, INK, anchor="ls")
            sx = x0 + name_w
            n = len(p.colors)
            seg = strip_w / n
            _box(cv, (sx - 1.5) * k, (y - 1.5) * k, (sx + strip_w + 1.5) * k, (y + poster_h + 1.5) * k, EDGE, 8.5 * k)
            px0 = x0 + cw - poster_w
            _box(cv, (px0 - 1.5) * k, (y - 1.5) * k, (px0 + poster_w + 1.5) * k, (y + poster_h + 1.5) * k, EDGE,
                 7.5 * k)
            for i, col in enumerate(p.colors):
                a = sx + i * seg
                b = a + seg + (0 if i == n - 1 else 1)
                if i == 0 or i == n - 1:
                    _box(cv, a * k, y * k, b * k, (y + poster_h) * k, col, 7 * k)
                    _box(cv, (a if i else a + 7) * k, y * k, ((b - 7) if i else b) * k, (y + poster_h) * k, col)
                else:
                    _box(cv, a * k, y * k, b * k, (y + poster_h) * k, col)
            _poster(cv, p, (x0 + cw - poster_w) * k, y * k, poster_w * k, poster_h * k, k)
    log("post")
    return cv
