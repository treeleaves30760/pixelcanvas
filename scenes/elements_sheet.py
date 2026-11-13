# -*- coding: utf-8 -*-
"""
Elements catalogue (elements_sheet) — every effect of pixelcanvas.elements on its own labelled tile.

Each tile is a small canvas drawn with one or two calls, then pasted into the grid. Sizes inside a tile are given
as fractions of the tile height (``T(f)`` → 1080p units), so the sheet looks the same at any scale.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas.color import clamp01
from pixelcanvas.elements import *          # noqa: F401,F403
from pixelcanvas import elements as E

NAME = "elements_sheet"
DEFAULT_SIZE = (1920, 2360)
DEFAULT_SEED = 3
COLS = 6
BG = "#0c0e13"
TITLE = "pixelcanvas.elements"
SUBTITLE = "fire · smoke · clouds · water · weather · sky · light — one call each"


def T(f):
    """Fraction of the tile height → the 1080p units the element functions take."""
    return f * 1080


def _ground(cv, v, top="#1a1612", bottom="#0b0907"):
    X, Y = cv.xy()
    m = clamp01(Y - v * cv.h + 0.5)
    cv.paint(E.mix([(0, top), (1, bottom)], clamp01((Y / cv.h - v) / max(1 - v, 1e-3))), m)
    return m


def _t_flame(cv, s):
    sky(cv, "night"); flame(cv, at=(0.5, 0.93), size=T(0.8), seed=s)


def _t_campfire(cv, s):
    sky(cv, "night"); _ground(cv, 0.62); campfire(cv, at=(0.5, 0.74), size=T(0.42), seed=s)


def _t_candle(cv, s):
    cv.paint("#120c0a"); candle(cv, at=(0.5, 0.42), size=T(0.26), body_height=T(0.5), seed=s)


def _t_torch(cv, s):
    cv.paint("#14100e"); torch(cv, at=(0.5, 0.42), size=T(0.32), angle=8, seed=s)


def _t_fireplace(cv, s):
    cv.paint("#241a14"); fireplace(cv, box=(0.18, 0.06, 0.82, 0.92), seed=s)


def _t_explosion(cv, s):
    sky(cv, "dusk"); explosion(cv, at=(0.5, 0.52), size=T(0.32), seed=s)


def _t_sparks(cv, s):
    cv.paint("#0a0a0e"); sparks(cv, at=(0.12, 0.88), n=90, spread=T(1.3), direction=-40, angle=50, length=T(0.08), gravity=0.35, seed=s)


def _t_embers(cv, s):
    cv.paint("#0d0806"); embers(cv, at=(0.5, 0.95), n=70, height=T(0.95), width=T(0.4), seed=s)


def _t_fireworks(cv, s):
    sky(cv, "night"); fireworks(cv, at=(0.33, 0.38), size=T(0.3), seed=s)
    fireworks(cv, at=(0.72, 0.32), size=T(0.26), kind="willow", color="#ffc65a", seed=s + 1)


def _t_smoke(cv, s):
    sky(cv, "overcast"); smoke(cv, at=(0.42, 1.0), size=T(1.05), wind=0.35, seed=s, color="#4a4a50", shade="#2a2a30", opacity=0.95)


def _t_steam(cv, s):
    cv.paint("#2a2420")
    X, Y = cv.xy()
    from pixelcanvas import sdf
    cup = sdf.fill(sdf.sd_box(X, Y, cv.w * 0.5, cv.h * 0.8, cv.h * 0.16, cv.h * 0.14, radius=cv.h * 0.05))
    cv.paint("#e8e2d8", cup)
    steam(cv, at=(0.5, 0.66), size=T(0.6), seed=s)


def _t_chimney(cv, s):
    sky(cv, "morning")
    X, Y = cv.xy()
    roof = clamp01(Y - (cv.h * 0.75 + (X - cv.w * 0.5) * 0.25) + 0.5)
    cv.paint("#3a2b26", roof)
    chim = clamp01(cv.h * 0.05 - abs(X - cv.w * 0.3) + 0.5 * 0) * clamp01(Y - cv.h * 0.55 + 0.5)
    cv.paint("#6a3a2c", np.clip(chim, 0, 1) * (abs(X - cv.w * 0.3) < cv.h * 0.05))
    chimney_smoke(cv, at=(0.3, 0.56), size=T(0.7), seed=s)


def _t_breath(cv, s):
    cv.paint("#1c2430"); breath(cv, at=(0.22, 0.55), size=T(0.6), direction=78, seed=s, opacity=0.5)


def _t_fog(cv, s):
    sky(cv, "dawn")
    X, Y = cv.xy()
    for i, (v, c) in enumerate(((0.55, "#6a6a80"), (0.66, "#4a4a5e"), (0.8, "#2c2c3a"))):
        ridge = v * cv.h + cv.h * 0.06 * E.fbm1d(cv.w, cv.h * 0.4, 3, seed=s + i)[None, :]
        cv.paint(c, clamp01(Y - ridge + 0.5))
        fog(cv, v_range=(v - 0.08, v + 0.15), density=0.5, layers=2, seed=s + i, scale=T(0.6))


def _t_dust(cv, s):
    cv.paint("#120f0c"); spotlight(cv, at=(0.75, -0.05), to=(0.35, 1.05), angle=26, strength=0.45, seed=s)


def _t_cumulus(cv, s):
    sky(cv, "day"); cumulus(cv, at=(0.5, 0.78), size=T(1.0), seed=s)


def _t_storm(cv, s):
    sky(cv, "storm"); storm_clouds(cv, region=(0, 0, 1, 0.85), seed=s, backlight=(0.7, 0.3), scale=T(0.6))


def _t_cirrus(cv, s):
    sky(cv, "noon"); cirrus(cv, region=(0, 0, 1, 1), seed=s, scale=T(0.9), opacity=0.7)


def _t_bands(cv, s):
    sky(cv, "sunset"); cloud_bands(cv, region=(0, 0.15, 1, 0.8), seed=s, scale=T(0.7))


def _t_ocean(cv, s):
    sky(cv, "golden", horizon=0.45); sun(cv, at=(0.5, 0.36), size=T(0.05))
    ocean(cv, horizon=0.45, sun=(0.5, 0.36), seed=s)


def _t_wave(cv, s):
    sky(cv, "day", horizon=0.6)
    ocean(cv, horizon=0.6, seed=s, foam=0.3)
    breaking_wave(cv, at=(0.45, 0.9), size=T(0.55), seed=s)


def _t_lake(cv, s):
    sky(cv, "dusk", horizon=0.55)
    X, Y = cv.xy()
    ridge = cv.h * 0.47 + cv.h * 0.06 * E.fbm1d(cv.w, cv.h * 0.5, 4, seed=s)[None, :]
    cv.paint("#1c1a2c", clamp01(Y - ridge + 0.5))
    moon(cv, at=(0.7, 0.2), size=T(0.06), phase=0.5)
    lake(cv, horizon=0.55, seed=s)


def _t_river(cv, s):
    cv.paint("#2e4a3a")
    X, Y = cv.xy()
    c = cv.w * 0.5 + cv.w * 0.12 * np.sin(Y / cv.h * 3)
    half = cv.w * (0.12 + 0.18 * Y / cv.h)
    water = clamp01(half - abs(X - c) + 0.5)
    cv.paint(E.mix([(0, "#7fa8b8"), (1, "#26505e")], Y / cv.h), water)
    river_flow(cv, mask=water, angle=90, seed=s, strength=0.8, scale=T(0.03))


def _t_waterfall(cv, s):
    cv.paint("#2c3a34")
    X, Y = cv.xy()
    cv.paint("#1d2723", clamp01(abs(X - cv.w * 0.5) - cv.w * 0.14) * 0 + (abs(X - cv.w * 0.5) > cv.w * 0.14))
    cv.paint("#2e5c64", clamp01(Y - cv.h * 0.86 + 0.5))
    waterfall(cv, box=(0.38, 0.08, 0.62, 0.88), seed=s)


def _t_splash(cv, s):
    cv.paint("#0e2a3a")
    X, Y = cv.xy()
    cv.paint(E.mix([(0, "#12425a"), (1, "#081c28")], clamp01((Y / cv.h - 0.6) / 0.4)), clamp01(Y - cv.h * 0.6 + 0.5))
    splash(cv, at=(0.5, 0.72), size=T(0.45), seed=s)


def _t_ripples(cv, s):
    cv.gradient([(0, "#1c3a52"), (1, "#0a1a28")])
    ripples(cv, at=(0.5, 0.55), size=T(0.75), rings=6, squash=0.35, strength=2.0, seed=s)


def _t_puddle(cv, s):
    sky(cv, "dusk", horizon=0.4)
    X, Y = cv.xy()
    cv.paint("#24201c", clamp01(Y - cv.h * 0.4 + 0.5))
    window_light(cv, box=(0.22, 0.12, 0.4, 0.34), glow=0.5)
    puddle(cv, at=(0.32, 0.62), size=T(1.0), horizon=0.4, seed=s, ripple=1.0)


def _t_underwater(cv, s):
    underwater(cv, seed=s, bubbles=18)


def _t_caustics(cv, s):
    cv.gradient([(0, "#2a7c8c"), (1, "#16485a")]); caustics(cv, scale=T(0.14), strength=0.7, seed=s)


def _t_droplets(cv, s):
    bokeh(cv, n=0)
    cv.gradient([(0, "#1a2a40"), (1, "#3a2a30")])
    bokeh(cv, n=26, size=T(0.12), seed=s, opacity=0.8)
    droplets(cv, size=T(0.028), seed=s, density=1.4)


def _t_rain(cv, s):
    sky(cv, "storm"); _ground(cv, 0.8, "#202428", "#121416")
    rain(cv, seed=s, ground=0.82, length=T(0.18), splashes=30)


def _t_wet(cv, s):
    sky(cv, "twilight", horizon=0.55)
    X, Y = cv.xy()
    cv.paint("#0e0e14", (Y < cv.h * 0.55) * clamp01(Y - cv.h * (0.3 + 0.1 * (np.floor(X / (cv.w * 0.12)) % 3)) + 0.5))
    for i, (u, c) in enumerate(((0.2, "#ffcc66"), (0.55, "#66d0ff"), (0.85, "#ff5c8a"))):
        glow(cv, at=(u, 0.42), radius=T(0.05), color=c, strength=1.5)
    ground = clamp01(Y - cv.h * 0.55 + 0.5)
    cv.paint("#15161c", ground)
    wet_ground(cv, ground, axis=0.55, strength=0.8)


def _t_snow(cv, s):
    sky(cv, "twilight"); _ground(cv, 0.8, "#b8c4d6", "#8894aa"); snowfall(cv, seed=s, size=T(0.005))


def _t_snowcap(cv, s):
    sky(cv, "twilight")
    X, Y = cv.xy()
    from pixelcanvas import sdf
    house = sdf.fill(sdf.sd_polygon(X, Y, [(0.25 * cv.w, 0.55 * cv.h), (0.5 * cv.w, 0.25 * cv.h), (0.75 * cv.w, 0.55 * cv.h),
                                           (0.7 * cv.w, 0.55 * cv.h), (0.7 * cv.w, 0.95 * cv.h), (0.3 * cv.w, 0.95 * cv.h), (0.3 * cv.w, 0.55 * cv.h)]))
    cv.paint("#3a2a24", house)
    window_light(cv, box=(0.44, 0.62, 0.56, 0.8), glow=0.4)
    snow_cap(cv, house, depth=T(0.05), seed=s)


def _t_lightning(cv, s):
    sky(cv, "storm")
    cl = storm_clouds(cv, region=(0, 0, 1, 0.45), seed=s, layers=2, scale=T(0.5))
    lightning(cv, start=(0.45, 0.2), end=(0.58, 1.0), seed=s, clouds=cl.mask, width=T(0.006))


def _t_rainbow(cv, s):
    sky(cv, "day"); g = _ground(cv, 0.82, "#3e5a34", "#26381f")
    rainbow(cv, center=(0.5, 1.05), radius=0.8, opacity=0.6, seed=s, mask=1 - g)


def _t_aurora(cv, s):
    sky(cv, "night"); stars(cv, seed=s, milky_way=False, density=0.8)
    aurora(cv, v_range=(0.3, 0.75), seed=s, height=T(0.35))
    X, Y = cv.xy()
    ridge = cv.h * 0.84 - cv.h * 0.08 * (0.5 + E.fbm1d(cv.w, cv.h * 0.5, 4, seed=s))[None, :]
    cv.paint("#04060a", clamp01(Y - ridge + 0.5))


def _t_sun(cv, s):
    sky(cv, "noon"); sun(cv, at=(0.62, 0.4), size=T(0.05), rays=0.8, flare=0.5, seed=s)


def _t_moon(cv, s):
    sky(cv, "night"); stars(cv, seed=s, milky_way=False, density=0.6)
    for i, ph in enumerate((0.15, 0.3, 0.5, 0.75)):
        moon(cv, at=(0.14 + 0.24 * i, 0.5), size=T(0.09), phase=ph, seed=s, glow=0.3)


def _t_stars(cv, s):
    sky(cv, "night"); stars(cv, seed=s, band_center=(0.5, 0.5))


def _t_shooting(cv, s):
    sky(cv, "night"); stars(cv, seed=s, milky_way=False)
    shooting_star(cv, start=(0.2, 0.15), end=(0.7, 0.6), width=T(0.006))


def _t_skies(cv, s):
    w = cv.w
    for i, k in enumerate(("dawn", "noon", "sunset", "dusk", "night")):
        sub = Canvas(int(w / 5) + 1, cv.h)
        sky(sub, k)
        cv.paste(sub, int(i * w / 5), 0)


def _t_godrays(cv, s):
    sky(cv, "golden"); sun(cv, at=(0.55, 0.22), size=T(0.06), rays=0)
    cl = storm_clouds(cv, region=(0, 0, 1, 0.75), layers=2, seed=s, coverage=0.5, color="#3a3440", light="#9a8070",
                      backlight=(0.55, 0.22), scale=T(0.5))
    god_rays(cv, at=(0.55, 0.22), occluder=cl.mask, strength=0.6, reach=0.5, seed=s)


def _t_haze(cv, s):
    sky(cv, "golden", horizon=0.6)
    X, Y = cv.xy()
    cv.paint("#5a4a3a", clamp01(Y - cv.h * 0.6 + 0.5))
    cv.paint("#1c1c20", (abs(X - cv.w * 0.5) < (Y - cv.h * 0.6) * 0.8) * (Y > cv.h * 0.6))
    for i in range(7):
        x = cv.w * (0.08 + 0.14 * i)
        cv.paint("#2a2420", (abs(X - x) < cv.h * 0.012) * (Y > cv.h * 0.5) * (Y < cv.h * 0.6))
    heat_haze(cv, region=(0, 0.35, 1, 0.75), strength=T(0.025), scale=T(0.02), seed=s)


def _t_glow(cv, s):
    cv.paint("#0a0b10")
    for i, c in enumerate(("#ffcf7a", "#7ad0ff", "#ff6aa8")):
        glow(cv, at=(0.22 + 0.28 * i, 0.5), radius=T(0.08), color=c, strength=1.4)


def _t_lamp(cv, s):
    cv.gradient([(0, "#1a1614"), (1, "#0c0a09")]); lamp(cv, at=(0.5, 0.2), size=T(0.32), seed=s)


def _t_lantern(cv, s):
    cv.paint("#0c0d14")
    lantern(cv, at=(0.32, 0.52), size=T(0.55), seed=s)
    lantern(cv, at=(0.72, 0.48), size=T(0.45), style="paper", seed=s)


def _t_neon(cv, s):
    cv.paint("#0b0a12"); neon(cv, text="OPEN", at=(0.5, 0.5), size=T(0.38), color="#ff3cac", width=T(0.012))


def _t_window(cv, s):
    cv.gradient([(0, "#151c2c"), (1, "#0c0e16")])
    window_light(cv, box=(0.3, 0.1, 0.55, 0.5), floor=(0.62, 0.98, 0.12), seed=s)


def _t_bokeh(cv, s):
    cv.gradient([(0, "#140e1a"), (1, "#0a0a12")]); bokeh(cv, n=40, size=T(0.09), seed=s, shape="hex", opacity=0.7)


def _t_spot(cv, s):
    cv.paint("#0a0a0e"); spotlight(cv, at=(0.5, -0.02), to=(0.5, 0.92), angle=30, seed=s)


def _t_flare(cv, s):
    sky(cv, "night"); glow(cv, at=(0.25, 0.25), radius=T(0.05), color="#ffffff", strength=2)
    lens_flare(cv, at=(0.25, 0.25), strength=1.0, seed=s)


def _t_volfog(cv, s):
    cv.paint("#0b0d14")
    lights = [glow(cv, at=(0.2 + 0.3 * i, 0.4), radius=T(0.03), color=c, strength=2.0)
              for i, c in enumerate(("#ffcf7a", "#ffd9a0", "#ffcf7a"))]
    volumetric_fog(cv, lights, density=1.0, seed=s)


TILES = [
    ("flame", _t_flame), ("campfire", _t_campfire), ("candle", _t_candle), ("torch", _t_torch),
    ("fireplace", _t_fireplace), ("explosion", _t_explosion), ("sparks", _t_sparks), ("embers", _t_embers),
    ("fireworks", _t_fireworks), ("smoke", _t_smoke), ("steam", _t_steam), ("chimney smoke", _t_chimney),
    ("breath", _t_breath), ("fog", _t_fog), ("dust motes", _t_dust), ("cumulus", _t_cumulus),
    ("storm clouds", _t_storm), ("cirrus", _t_cirrus), ("cloud bands", _t_bands), ("ocean", _t_ocean),
    ("breaking wave", _t_wave), ("lake", _t_lake), ("river", _t_river), ("waterfall", _t_waterfall),
    ("splash", _t_splash), ("ripples", _t_ripples), ("puddle", _t_puddle), ("underwater", _t_underwater),
    ("caustics", _t_caustics), ("droplets", _t_droplets), ("rain", _t_rain), ("wet ground", _t_wet),
    ("snow", _t_snow), ("snow cap", _t_snowcap), ("lightning", _t_lightning), ("rainbow", _t_rainbow),
    ("aurora", _t_aurora), ("sun", _t_sun), ("moon phases", _t_moon), ("stars", _t_stars),
    ("shooting star", _t_shooting), ("sky keywords", _t_skies), ("god rays", _t_godrays), ("heat haze", _t_haze),
    ("glow", _t_glow), ("lamp", _t_lamp), ("lantern", _t_lantern), ("neon", _t_neon),
    ("window light", _t_window), ("bokeh", _t_bokeh), ("spotlight", _t_spot), ("lens flare", _t_flare),
    ("volumetric fog", _t_volfog),
]


def render(width=1920, height=2360, seed=DEFAULT_SEED, log=None, only=None):
    log = log or (lambda m: None)
    S = height / 2360
    cv = Canvas(width, height, BG)
    log("title")
    head = int(150 * S)
    cv.text(TITLE, (int(40 * S), int(40 * S)), int(56 * S), "#f2ede4", bold=True)
    cv.text(SUBTITLE, (int(42 * S), int(108 * S)), int(26 * S), "#9aa0aa")
    tiles = [t for t in TILES if only is None or t[0] in only]
    rows = (len(tiles) + COLS - 1) // COLS
    gap = int(12 * S)
    lab = int(34 * S)
    tw = (width - gap * (COLS + 1)) // COLS
    th = min(int(tw * 0.64), (height - head - gap * (rows + 1)) // rows - lab)
    for i, (name, fn) in enumerate(tiles):
        log(name)
        r, c = divmod(i, COLS)
        x = gap + c * (tw + gap)
        y = head + gap + r * (th + lab + gap)
        t = Canvas(tw, th, "#000000")
        fn(t, seed + i)
        t.bloom(0.9, 10 * th / 300, 0.25)
        t.soft_clip(0.82)
        cv.paste(t, x, y)
        cv.text(name, (x + int(4 * S), y + th + int(6 * S)), int(22 * S), "#d8d2c8")
    return cv
