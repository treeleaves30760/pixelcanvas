# -*- coding: utf-8 -*-
"""
Six finished wallpapers made with pixelcanvas.backdrop, shown as a 2x3 set (or one of them full-frame).

* Format: 1920x1080 sheet; WALL = 0..5 renders that wallpaper alone at the full canvas size instead, so
  ``--set WALL=3 --size 2560x1440`` (desktop) or ``--size 1170x2532`` (phone) gives a wallpaper to use.
* Each wallpaper = one backdrop call with a chosen palette, seed and options, then a light finish
  (fine grain, gentle vignette). Positions are fractions; sizes scale with the canvas.
* Layers: header, one per wallpaper, post.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas.backdrop import render as bd_render
from pixelcanvas.color import clamp01
from pixelcanvas.filters import grain, vignette

NAME = "backdrop_wallpapers"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

WALL = None             # None = the 2x3 set; 0..5 = one wallpaper full-frame
BG = "#0d0f14"
INK = "#eef1f5"
MUTED = "#8d96a3"
TITLE = "Wallpapers"
SUBTITLE = "each one backdrop() call with a palette, a seed and a few options"

WALLPAPERS = [
    ("Ember Hills", "hills", "sunset", 3, dict(layers=6, sun_x=0.7)),
    ("Night Drive", "synthwave", "synthwave", 2, {}),
    ("Lilac Peaks", "mountains", "lavender", 5, dict(sun_x=0.66)),
    ("Silk", "ribbons", "aurora", 4, {}),
    ("Dawn Range", "mountains", "soft blue and peach", 3, dict(sun_x=0.3)),
    ("Pine Mist", "forest", "winter", 2, dict(layers=6)),
]


def _finish(img, seed):
    img = vignette(img, 0.22, 0.6, 0.8)
    return np.clip(grain(img, 0.008, seed), 0, 1)


def _wall(i, w, h, seed):
    title, name, pal, s, opts = WALLPAPERS[i]
    img, _ = bd_render(name, w, h, palette=pal, seed=s + seed - 1, **opts)
    return _finish(img, s)


def _rounded(w, h, r):
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    qx = np.abs(X + 0.5 - w / 2) - w / 2 + r
    qy = np.abs(Y + 0.5 - h / 2) - h / 2 + r
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return clamp01(0.5 - d)


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    if WALL is not None:
        cv = Canvas(width, height)
        log(WALLPAPERS[int(WALL)][0])
        cv.paint(_wall(int(WALL), width, height, seed))
        log("post")
        return cv

    k = width / 1920.0
    cv = Canvas(width, height, BG)
    gap, cap = 40, 92
    tw = (1920 - 160 - 2 * gap) / 3
    th = tw * 9 / 16
    off = (1080 - (110 + 2 * th + cap + 46)) / 2
    top = off + 110
    log("header")
    cv.text(TITLE, (80 * k, (off + 52) * k), 50 * k, INK, bold=True, anchor="ls")
    cv.text(SUBTITLE, ((1920 - 80) * k, (off + 52) * k), 22 * k, MUTED, anchor="rs")
    for i, (title, name, pal, s, opts) in enumerate(WALLPAPERS):
        log(title)
        r, c = divmod(i, 3)
        x0, y0 = (80 + c * (tw + gap)) * k, (top + r * (th + cap)) * k
        w, h = int(round(tw * k)), int(round(th * k))
        cv.paste(_wall(i, w, h, seed), int(round(x0)), int(round(y0)), mask=_rounded(w, h, 16 * k))
        cv.text(title, (x0 + 2 * k, y0 + h + 36 * k), 24 * k, INK, bold=True, anchor="ls")
        cv.text(f"{name} · {pal.replace('_', ' ')}", (x0 + w - 2 * k, y0 + h + 36 * k), 19 * k, MUTED,
                anchor="rs")
    log("post")
    return cv
