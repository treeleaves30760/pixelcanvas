# -*- coding: utf-8 -*-
"""Natural phenomena as one-call effects: fire, smoke, clouds, water, weather, sky and light.

Every function draws onto a :class:`~pixelcanvas.canvas.Canvas` through ``cv.paint`` / ``cv.paste`` with its own
alpha, so each call is its own editable layer. Positions are fractions of the canvas (``at=(u, v)``), sizes are
1080p pixels (scaled by ``S = H / 1080``), and results are deterministic by ``seed``. Emissive effects are HDR
(> 1) so they bloom in post. Each call returns an :class:`FX` with ``.mask`` (coverage) and ``.lights`` (point
lights you can pass to :func:`illuminate` to tint nearby objects).

    from pixelcanvas import Canvas
    from pixelcanvas.elements import element, sky, campfire, illuminate
    cv = Canvas(960, 540)
    sky(cv, "night")
    fire = campfire(cv, at=(0.5, 0.85), size=220)
    cv.soft_clip(0.85)

Small models can use the keyword registry: ``element(cv, "campfire", at=(0.5, 0.8))``, ``names()``,
``find("rain on a window")``.
"""
from __future__ import annotations

import math

import numpy as np

from .color import F32, rgb, as_color, mix, lerp, clamp01, smoothstep, grid, luma
from .noise import fbm2d, fbm2d_at, perlin2d_at, perlin1d_at, fbm1d
from .filters import gaussian_blur, sample, resize, light_shafts
from .masks import shape_mask, line_mask
from .form import check_frac as _check_frac, guard_fractions as _guard_fractions
from ._args import (reject_unknown as _reject, signature_options as _sigopts, unknown_name as _unknown_name,
                    check_choice as _choice, strict as _strict)

__all__ = [
    # core
    "FX", "Light", "illuminate", "light_map", "worley", "voronoi", "curl", "advect",
    # sky & weather
    "sky", "SKIES", "stars", "moon", "sun", "shooting_star", "aurora", "rainbow", "lightning", "god_rays",
    "heat_haze", "snowfall", "snow_cap",
    # clouds, smoke, vapour
    "cumulus", "storm_clouds", "cirrus", "cloud_bands", "smoke", "steam", "chimney_smoke", "breath", "fog",
    "dust_motes",
    # fire
    "flame", "campfire", "candle", "torch", "fireplace", "explosion", "sparks", "embers", "fireworks", "FIRE",
    "stone", "firewood",
    # water
    "ocean", "breaking_wave", "lake", "river_flow", "waterfall", "splash", "ripples", "puddle", "underwater",
    "droplets", "rain", "wet_ground", "caustics",
    # light
    "glow", "lamp", "lantern", "neon", "window_light", "bokeh", "spotlight", "lens_flare", "volumetric_fog",
    # registry
    "element", "names", "find", "REGISTRY",
]

FIRE = [(0.00, (0.30, 0.02, 0.00)), (0.14, (0.80, 0.12, 0.02)), (0.32, (1.00, 0.36, 0.04)),
        (0.55, (1.00, 0.62, 0.14)), (0.78, (1.00, 0.86, 0.45)), (1.00, (1.00, 0.98, 0.86))]


# =============================================================================
# core helpers
# =============================================================================
class Light:
    """A point light: position in pixels, colour (RGB 0~1, HDR ok), radius in pixels, strength."""

    def __init__(self, x, y, color, radius, strength=1.0):
        self.x, self.y = float(x), float(y)
        self.color = rgb(color)
        self.radius = float(radius)
        self.strength = float(strength)

    def __repr__(self):
        return f"Light(x={self.x:.0f}, y={self.y:.0f}, radius={self.radius:.0f}, strength={self.strength:.2f})"


class FX:
    """What an effect call returns: ``mask`` (H, W) coverage (or None), ``lights`` (list of Light), extras in
    ``info`` (free-form by design: any extra keyword is stored there). ``fx.light`` is the first light or None."""

    def __init__(self, mask=None, lights=(), **info):
        self.mask = mask
        self.lights = list(lights)
        self.info = info

    @property
    def light(self):
        return self.lights[0] if self.lights else None

    def __getattr__(self, k):
        info = self.__dict__.get("info", {})
        if k in info:
            return info[k]
        raise AttributeError(k)

    def __repr__(self):
        return f"FX(mask={'yes' if self.mask is not None else 'no'}, lights={self.lights}, info={list(self.info)})"


def _S(cv):
    return cv.h / 1080.0


def _xy(cv, at):
    """Fractions (u, v) → pixels. Values may lie outside 0~1 (off-canvas sources)."""
    _check_frac("elements", "at", (at[0], at[1]))
    return float(at[0]) * cv.w, float(at[1]) * cv.h


def _win(cv, x0, y0, x1, y1):
    """Clipped integer window + absolute pixel grids, or None if empty."""
    x0 = int(max(0, math.floor(x0)))
    y0 = int(max(0, math.floor(y0)))
    x1 = int(min(cv.w, math.ceil(x1)))
    y1 = int(min(cv.h, math.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    Y, X = np.mgrid[y0:y1, x0:x1]
    return (x0, y0, x1, y1, X.astype(F32), Y.astype(F32))


def _put(cv, color, alpha, win, mode="normal"):
    """Composite a local colour / alpha onto the canvas inside window ``win`` (one paint step)."""
    x0, y0, x1, y1 = win[:4]
    col = as_color(color)
    if col.ndim == 1:
        col = np.broadcast_to(col, (y1 - y0, x1 - x0, 3))
    cv.paste(np.ascontiguousarray(col, F32), x0, y0, mask=np.asarray(alpha, F32), mode=mode)


def _full(cv, local, win):
    """Embed a local (h, w) array into a full-frame zero array."""
    out = np.zeros((cv.h, cv.w), F32)
    if win is not None:
        out[win[1]:win[3], win[0]:win[2]] = local
    return out


def _ramp(palette, t):
    """Colour lookup: ``palette`` = list of colours (evenly spaced) or ``[(pos, colour), ...]`` stops."""
    p = list(palette)
    if p and isinstance(p[0], (tuple, list)) and len(p[0]) == 2 and not isinstance(p[0][0], (tuple, list, str)) \
            and np.ndim(p[0][1]) != 0:
        return mix(p, t)
    n = len(p)
    return mix([(i / max(n - 1, 1), c) for i, c in enumerate(p)], t)


def _blur(a, sigma):
    """Gaussian blur that downsamples for large sigma (fast glows)."""
    a = np.asarray(a, F32)
    if sigma <= 0:
        return a.copy()
    h, w = a.shape[:2]
    k = int(sigma // 6)
    if k >= 2 and h // k >= 8 and w // k >= 8:
        small = resize(a, max(w // k, 1), max(h // k, 1), "box")
        small = gaussian_blur(small, sigma / k)
        return resize(small, w, h, "bilinear")
    return gaussian_blur(a, sigma)


def _area(cv):
    """Canvas area in 1080p units relative to a 1920x1080 frame (resolution-independent particle counts)."""
    S = _S(cv)
    return cv.w * cv.h / (S * S) / (1920.0 * 1080.0)


def _mask_or_ones(cv, mask):
    return np.ones((cv.h, cv.w), F32) if mask is None else np.asarray(mask, F32)


def _region_mask(cv, region):
    """``region=(u0, v0, u1, v1)`` fractions → soft-edged mask (None → whole canvas)."""
    if region is None:
        return np.ones((cv.h, cv.w), F32)
    X, Y = cv.xy()
    u0, v0, u1, v1 = region
    f = 2.0
    m = smoothstep(u0 * cv.w - f, u0 * cv.w + f, X) * (1 - smoothstep(u1 * cv.w - f, u1 * cv.w + f, X))
    m = m * smoothstep(v0 * cv.h - f, v0 * cv.h + f, Y) * (1 - smoothstep(v1 * cv.h - f, v1 * cv.h + f, Y))
    return m.astype(F32)


_KX, _KY, _KS, _KM = np.uint32(374761393), np.uint32(668265263), np.uint32(1442695041), np.uint32(1274126177)


def _hashf(ix, iy, seed):
    """Deterministic hash of integer lattice coords → float 0~1."""
    with np.errstate(over="ignore"):
        h = np.asarray(ix).astype(np.int64).astype(np.uint32) * _KX + \
            np.asarray(iy).astype(np.int64).astype(np.uint32) * _KY + np.uint32(int(seed) & 0xFFFFFFFF) * _KS
        h = h ^ (h >> np.uint32(13))
        h = h * _KM
        h = h ^ (h >> np.uint32(16))
    return ((h & np.uint32(0xFFFFFF)).astype(F32) / F32(16777216.0))


def _splat(h, w, xs, ys, vals):
    """Bilinear point splat → (h, w) accumulation (sub-pixel stars, specks)."""
    out = np.zeros((h, w), np.float64)
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    vals = np.asarray(vals, np.float64)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    for ox, oy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + ox, y0 + oy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(out, (yi[ok], xi[ok]), (vals * wt)[ok])
    return out.astype(F32)


def _gauss_spot(X, Y, cx, cy, sigma):
    return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma * sigma)).astype(F32)


def _soft_glow(cv, x, y, radius, color, strength=1.0, mask=None):
    """Physically-flavoured glow: tight core + wide 1/(1+r²) falloff, added in a window (HDR)."""
    R = radius * 4.0
    win = _win(cv, x - R, y - R, x + R, y + R)
    if win is None:
        return None
    X, Y = win[4], win[5]
    d2 = ((X - x) ** 2 + (Y - y) ** 2) / (radius * radius)
    g = (0.55 * np.exp(-d2 * 4.0) + 0.45 / (1.0 + 6.0 * d2)) * (1 - smoothstep(9.0, 16.0, d2))
    g = (g * strength).astype(F32)
    if mask is not None:
        g = g * np.asarray(mask, F32)[win[1]:win[3], win[0]:win[2]]
    _put(cv, color, g, win, "add")
    return win


# =============================================================================
# noise primitives that make structure (cells, edges, flow)
# =============================================================================
def worley(X, Y, seed=0, jitter=0.9):
    """Cellular (Worley) noise at coordinates in cell units.

    Returns ``(f1, f2, edge, cell_id)``: distance to the nearest / second feature point, distance to the nearest
    cell border (true bisector distance — use for mortar, cracks, caustic webs), and a 0~1 random id per cell."""
    X = np.asarray(X, F32)
    Y = np.asarray(Y, F32)
    xi = np.floor(X).astype(np.int64)
    yi = np.floor(Y).astype(np.int64)
    f1 = np.full(X.shape, 1e9, F32)
    f2 = np.full(X.shape, 1e9, F32)
    p1x = np.zeros(X.shape, F32)
    p1y = np.zeros(X.shape, F32)
    cid = np.zeros(X.shape, F32)
    pts = []
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = xi + ox, yi + oy
            px = cx + 0.5 + (_hashf(cx, cy, seed) - 0.5) * jitter
            py = cy + 0.5 + (_hashf(cx, cy, seed + 7919) - 0.5) * jitter
            pts.append((px, py))
            d = (X - px) ** 2 + (Y - py) ** 2
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            f1 = np.where(closer, d, f1)
            p1x = np.where(closer, px, p1x)
            p1y = np.where(closer, py, p1y)
            cid = np.where(closer, _hashf(cx, cy, seed + 31337), cid)
    edge = np.full(X.shape, 1e9, F32)
    for px, py in pts:
        dx, dy = px - p1x, py - p1y
        L = np.sqrt(dx * dx + dy * dy)
        valid = L > 1e-5
        mx, my = (px + p1x) * 0.5, (py + p1y) * 0.5
        e = ((mx - X) * dx + (my - Y) * dy) / np.where(valid, L, 1.0)
        edge = np.where(valid, np.minimum(edge, e), edge)
    return np.sqrt(f1).astype(F32), np.sqrt(f2).astype(F32), np.maximum(edge, 0).astype(F32), cid.astype(F32)


def voronoi(cv, scale=60, seed=0, jitter=0.9, warp=0.0):
    """Full-canvas Voronoi cells: ``(cell_id, edge_px)`` where scale = cell size in 1080p px. ``warp`` bends cell
    borders (0~1). Use for cobbles, cracked earth, scales, caustics."""
    S = _S(cv)
    X, Y = cv.xy()
    sc = scale * S
    U, V = X / sc, Y / sc
    if warp:
        U = U + warp * 0.35 * fbm2d(cv.h, cv.w, sc * 1.3, 3, seed=seed + 3)
        V = V + warp * 0.35 * fbm2d(cv.h, cv.w, sc * 1.3, 3, seed=seed + 4)
    f1, f2, edge, cid = worley(U, V, seed, jitter)
    return cid, edge * sc


def curl(X, Y, seed=0, octaves=3):
    """Divergence-free flow (curl of fBm) at coordinates in cell units → (vx, vy). Swirls without sources/sinks,
    the motion of smoke and water."""
    e = 0.01
    dpy = fbm2d_at(X, Y + e, octaves, seed=seed) - fbm2d_at(X, Y - e, octaves, seed=seed)
    dpx = fbm2d_at(X + e, Y, octaves, seed=seed) - fbm2d_at(X - e, Y, octaves, seed=seed)
    return (dpy / (2 * e)).astype(F32), (-dpx / (2 * e)).astype(F32)


def advect(X, Y, steps=4, step=0.25, seed=0, octaves=3):
    """Trace coordinates backwards through curl flow → warped (X, Y). Sampling noise at the result gives curls
    and eddies (smoke, flame licks, marbled water)."""
    X = np.asarray(X, F32).copy()
    Y = np.asarray(Y, F32).copy()
    for i in range(int(steps)):
        vx, vy = curl(X, Y, seed + 17 * i, octaves)
        X -= vx * step
        Y -= vy * step
    return X, Y


# =============================================================================
# lights
# =============================================================================
def _lights(obj):
    if obj is None:
        return []
    if isinstance(obj, Light):
        return [obj]
    if isinstance(obj, FX):
        return obj.lights
    out = []
    for o in obj:
        out += _lights(o)
    return out


def light_map(cv, lights, falloff=1.0):
    """(H, W, 3) additive light colour from point lights (Light / FX / lists of them). Smooth inverse-square."""
    X, Y = cv.xy()
    acc = np.zeros((cv.h, cv.w, 3), F32)
    for L in _lights(lights):
        d2 = ((X - L.x) ** 2 + (Y - L.y) ** 2) / (L.radius * L.radius)
        f = L.strength / (1.0 + d2 * 2.5 * falloff) * (1 - smoothstep(4.0, 30.0, d2))
        acc += f[..., None] * L.color
    return acc


def illuminate(cv, lights, mask=None, strength=1.0, ambient=0.0):
    """Tint what is already painted with nearby lights (warm firelight on rocks, lamp light on a wall).

    ``lights`` = Light, FX or a list; ``mask`` limits the lit surfaces (e.g. the ground and rock masks).
    Multiplies existing colour by (1 + light) so dark albedo stays dark; ``ambient`` adds a little light."""
    Lm = light_map(cv, lights) * strength
    m = _mask_or_ones(cv, mask)
    lit = cv.buf * (1.0 + Lm) + Lm * ambient
    cv.paint(lit, m)
    return FX(m, _lights(lights))


# =============================================================================
# sky gradients, stars, moon, sun
# =============================================================================
SKIES = {
    "noon": [(0.0, "#2a63b0"), (0.55, "#6ea6dc"), (1.0, "#c6e0f2")],
    "day": [(0.0, "#3170bd"), (0.6, "#86b8e2"), (1.0, "#d6e8f2")],
    "morning": [(0.0, "#4a7cc0"), (0.6, "#a8c6e0"), (1.0, "#f1e4cc")],
    "golden": [(0.0, "#3c5a92"), (0.5, "#c69a84"), (0.8, "#f2b770"), (1.0, "#ffd9a0")],
    "sunset": [(0.0, "#1b2350"), (0.32, "#5b4478"), (0.6, "#d0705e"), (0.82, "#f2a35e"), (1.0, "#ffd08c")],
    "dawn": [(0.0, "#26365e"), (0.45, "#7f84ae"), (0.78, "#eab0a2"), (1.0, "#ffdcbc")],
    "dusk": [(0.0, "#0b1230"), (0.5, "#30366a"), (0.82, "#9c5f78"), (1.0, "#dd8c70")],
    "twilight": [(0.0, "#070b22"), (0.55, "#1e2a5a"), (0.85, "#4a4f86"), (1.0, "#b9788a")],
    "night": [(0.0, "#02040b"), (0.6, "#08122a"), (1.0, "#16264a")],
    "storm": [(0.0, "#15191f"), (0.6, "#323a44"), (1.0, "#4f5862")],
    "overcast": [(0.0, "#6f7880"), (0.6, "#9aa2a8"), (1.0, "#c3c7c8")],
    "underwater": [(0.0, "#2a9bb0"), (0.5, "#0f5d78"), (1.0, "#05223a")],
}


def sky(cv, kind="sunset", horizon=1.0, mask=None, glow=True):
    """Sky gradient by keyword (``SKIES``: noon / day / morning / golden / sunset / dawn / dusk / twilight / night /
    storm / overcast / underwater) or a list of stops. ``horizon`` = v fraction where the gradient ends (the sea
    or ground line). Sunsets and dawns get a warm horizon glow. Returns FX with the stops in ``info['stops']``."""
    if isinstance(kind, str) and kind not in SKIES:
        raise _unknown("sky kind", kind, SKIES)
    stops = SKIES[kind] if isinstance(kind, str) else kind
    X, Y = cv.xy()
    V = Y / max(horizon * cv.h, 1)
    col = mix(stops, V)
    if glow and isinstance(kind, str) and kind in ("sunset", "dawn", "golden", "dusk"):
        g = np.exp(-((1 - V) / 0.12) ** 2)[..., None]
        col = col + g * rgb("#ffb070") * (0.25 if kind != "dusk" else 0.12)
    m = _mask_or_ones(cv, mask)
    cv.paint(col, m)
    return FX(m, stops=stops)


def stars(cv, density=1.0, seed=0, milky_way=True, twinkle=1.0, mask=None, horizon=None, band_angle=-28,
          band_center=(0.55, 0.35), brightness=1.0):
    """Star field: power-law brightness, colour temperature, a few bright stars with diffraction spikes, and an
    optional Milky Way band with dust lanes. ``mask`` (sky area) or ``horizon`` (v fraction) limits it; stars fade
    towards the horizon."""
    H, W = cv.h, cv.w
    S = _S(cv)
    rng = np.random.default_rng(seed)
    X, Y = cv.xy()
    m = _mask_or_ones(cv, mask)
    if horizon is not None:
        m = m * (1 - smoothstep(horizon * H - 60 * S, horizon * H, Y))
        fade = 0.25 + 0.75 * smoothstep(horizon * H, 0.2 * horizon * H, Y)
    else:
        fade = 1.0
    # milky way
    acc = np.zeros((H, W, 3), F32)
    band = None
    if milky_way:
        a = math.radians(band_angle)
        bx, by = band_center[0] * W, band_center[1] * H
        d = (-(X - bx) * math.sin(a) + (Y - by) * math.cos(a))           # distance across the band
        t = ((X - bx) * math.cos(a) + (Y - by) * math.sin(a))            # along the band
        wb = 150 * S * (1 + 0.35 * fbm2d(H, W, 500 * S, 3, seed=seed + 5))
        n = fbm2d(H, W, 90 * S, 6, seed=seed + 1)
        core = np.exp(-(d / wb) ** 2) * (0.75 + 0.25 * np.cos(t / (W * 0.6)))
        clouds = clamp01(0.55 + 0.9 * n) * core
        lanes = smoothstep(0.55, 0.95, fbm2d(H, W, 160 * S, 5, seed=seed + 2, mode="ridged"))
        rift = np.exp(-((d - 0.15 * wb * fbm2d(H, W, 300 * S, 3, seed=seed + 6)) / (wb * 0.22)) ** 2)
        dust = clamp01(1 - 0.85 * lanes * core - 0.6 * rift * smoothstep(0.1, 0.6, core))
        band = clouds * dust
        col = mix([(0, "#3a4a8a"), (0.5, "#9a94b8"), (1, "#f2dcc0")], clamp01(core * 1.1))
        acc += col * (band * 0.42 * brightness)[..., None]
    # stars
    n_st = int(2600 * density * _area(cv))
    xs = rng.uniform(0, W, n_st)
    ys = rng.uniform(0, H, n_st)
    if band is not None:                                               # extra dense faint stars in the band
        n_b = int(n_st * 1.6)
        bxs = rng.uniform(0, W, n_b * 3)
        bys = rng.uniform(0, H, n_b * 3)
        keep = rng.uniform(0, 1, n_b * 3) < band[np.clip(bys.astype(int), 0, H - 1), np.clip(bxs.astype(int), 0, W - 1)]
        bxs, bys = bxs[keep][:n_b], bys[keep][:n_b]
        xs = np.concatenate([xs, bxs])
        ys = np.concatenate([ys, bys])
    mag = np.clip((rng.pareto(1.6, len(xs)) + 1) * 0.12, 0, 6) * brightness
    temp = rng.uniform(0, 1, len(xs))
    tcol = mix([(0, "#9fb8ff"), (0.45, "#e8eeff"), (0.75, "#fff3dc"), (1, "#ffc890")], temp)
    for c in range(3):
        acc[..., c] += gaussian_blur(_splat(H, W, xs, ys, mag * tcol[:, c]), 0.55 * max(S, 0.5)) * 2.2
    # bright stars: halo + spikes
    order = np.argsort(-mag)[: max(1, int(14 * density * twinkle))]
    for i in order:
        x, y, b = xs[i], ys[i], min(mag[i], 3.0)
        R = 40 * S * (0.6 + 0.2 * b)
        win = _win(cv, x - R, y - R, x + R, y + R)
        if win is None:
            continue
        Xl, Yl = win[4] - x, win[5] - y
        halo = _gauss_spot(Xl, Yl, 0, 0, 2.2 * S) * 1.4 + _gauss_spot(Xl, Yl, 0, 0, 7 * S) * 0.18
        spikes = (np.exp(-(Yl / (0.7 * S)) ** 2) * np.exp(-np.abs(Xl) / (R * 0.28)) +
                  np.exp(-(Xl / (0.7 * S)) ** 2) * np.exp(-np.abs(Yl) / (R * 0.28))) * 0.6 * twinkle
        acc[win[1]:win[3], win[0]:win[2]] += ((halo + spikes) * b * 0.6)[..., None] * tcol[i]
    acc = acc * (m * fade)[..., None]
    cv.paint(acc, None, "add")
    return FX(m, band=band)


def moon(cv, at=(0.75, 0.22), size=60, phase=0.5, seed=0, color="#f1ecdc", glow=0.6, tilt=0.0, earthshine=0.05):
    """Moon disc with maria, craters, phase terminator and a soft halo.

    ``size`` = radius (1080p px); ``phase`` 0 = new, 0.25 = first quarter (lit right), 0.5 = full, 0.75 = last
    quarter; ``tilt`` rotates the terminator (degrees). Returns FX with the disc mask and a cool light."""
    S = _S(cv)
    cx, cy = _xy(cv, at)
    r = size * S
    if glow:
        _soft_glow(cv, cx, cy, r * 2.2, rgb("#9fb4d8") * 0.35, glow)
    win = _win(cv, cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    nx, ny = (X - cx) / r, (Y - cy) / r
    rr = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1 - rr, 0, 1))
    e = 2 * math.pi * phase
    ta = math.radians(tilt)
    Lx, Ly, Lz = math.sin(e) * math.cos(ta), math.sin(e) * math.sin(ta), -math.cos(e)
    ndl = nx * Lx + ny * Ly + nz * Lz
    lit = smoothstep(-0.04, 0.12, ndl)
    # surface: maria (large dark seas) + craters (worley rims) on sphere coordinates
    su, sv = nx / (0.6 + 0.4 * nz), ny / (0.6 + 0.4 * nz)
    maria = smoothstep(0.05, 0.45, fbm2d_at(su * 1.6 + 3.1, sv * 1.6 + 1.7, 4, seed=seed))
    f1, f2, edge, cid = worley(su * 5.5, sv * 5.5, seed + 3)
    rad = 0.18 + 0.22 * cid
    crater = smoothstep(rad + 0.05, rad, f1)
    rim = np.exp(-((f1 - rad) / 0.045) ** 2) * (cid > 0.35)
    f1b, _, _, cidb = worley(su * 13, sv * 13, seed + 4)
    radb = 0.15 + 0.2 * cidb
    small = smoothstep(radb + 0.05, radb, f1b) * (cidb > 0.5)
    alb = 1.0 - 0.32 * maria - 0.12 * crater * (cid > 0.35) - 0.08 * small + 0.12 * rim
    alb = alb * (1 + 0.06 * fbm2d_at(su * 30, sv * 30, 3, seed=seed + 9))
    limb = 0.75 + 0.25 * nz
    shade = (lit * (0.35 + 0.65 * np.clip(ndl, 0, 1) ** 0.5) * 1.25 + earthshine) * limb
    col = rgb(color) * (alb * shade)[..., None]
    disc = clamp01(r - np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) + 0.5)
    _put(cv, col, disc, win)
    return FX(_full(cv, disc, win), [Light(cx, cy, "#b8c8e8", r * 12, 0.25 * (0.3 + abs(math.sin(e / 2))))])


def sun(cv, at=(0.5, 0.3), size=40, color="#fff2d0", glow=1.0, rays=0.35, halo="#ffb060", flare=False, seed=0):
    """Sun: HDR disc, layered glow, streaky rays (``rays`` 0~1) and an optional lens flare. ``size`` = disc
    radius (1080p px). Returns FX(disc mask, light)."""
    S = _S(cv)
    cx, cy = _xy(cv, at)
    r = size * S
    X, Y = cv.xy()
    D = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    g = (np.exp(-(D / (r * 16)) ** 2) * 0.16 + np.exp(-(D / (r * 4.5)) ** 2) * 0.28 + np.exp(-(D / (r * 1.6)) ** 2) * 0.45)
    cv.paint(rgb(halo), g * glow, "add")
    if rays:
        ang = np.arctan2(Y - cy, X - cx)
        n = (perlin1d_at((ang + math.pi) * 9.0, seed) + perlin1d_at((ang + math.pi) * 23.0, seed + 1) * 0.6)
        streak = clamp01(n) ** 2.0 * np.exp(-D / (r * 6)) * smoothstep(r * 0.8, r * 2, D)
        cv.paint(rgb(color), streak * rays * 0.6, "add")
    disc = clamp01(r - D + 0.5)
    cv.paint(rgb(color) * 2.2, disc)
    if flare:
        lens_flare(cv, at, strength=0.6 if flare is True else float(flare))
    return FX(disc, [Light(cx, cy, color, r * 30, 1.0)])


def shooting_star(cv, start=(0.25, 0.15), end=(0.45, 0.28), color="#dfe9ff", width=2.0, glow=1.0):
    """Meteor streak: tapered tail fading from a bright head at ``end``."""
    S = _S(cv)
    x0, y0 = _xy(cv, start)
    x1, y1 = _xy(cv, end)
    pad = 30 * S
    win = _win(cv, min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) + 1e-6
    t = ((X - x0) * dx + (Y - y0) * dy) / (L * L)
    tc = np.clip(t, 0, 1)
    d = np.sqrt((X - x0 - tc * dx) ** 2 + (Y - y0 - tc * dy) ** 2)
    wdt = width * S * (0.25 + 0.75 * tc)
    core = np.exp(-(d / wdt) ** 2) * tc ** 1.5 * (t <= 1.02)
    halo = np.exp(-(d / (wdt * 5)) ** 2) * tc ** 2 * 0.35
    head = _gauss_spot(X, Y, x1, y1, 3 * S) * 2.0 + _gauss_spot(X, Y, x1, y1, 12 * S) * 0.3
    a = (core * 1.8 + halo) * glow + head
    _put(cv, rgb(color), a, win, "add")
    return FX(_full(cv, clamp01(core), win), [Light(x1, y1, color, 60 * S, 0.3)])


# =============================================================================
# fire
# =============================================================================
def flame(cv, at=(0.5, 0.8), size=220, width=None, seed=0, lean=0.0, tongues=5, palette=None, intensity=1.0,
          blue_base=0.0, glow=0.7, turbulence=1.0, mask=None):
    """Flame with tapering, licking tongues: white-yellow core → orange → red → transparent tips.

    ``at`` = base centre, ``size`` = height (1080p px), ``width`` = base width (default 0.42 * size), ``lean`` =
    wind (+ right, fraction of height the tip drifts), ``tongues`` = number of flame fingers, ``palette`` = colour
    ramp cool → hot (default ``FIRE``), ``blue_base`` 0~1 adds the blue combustion zone (gas / candle).
    Returns FX(mask, [light])."""
    S = _S(cv)
    h = size * S
    wd = (width if width is not None else size * 0.55) * S
    bx, by = _xy(cv, at)
    hw = wd * 0.5
    pad_x = wd * 0.9 + abs(lean) * h * 1.2
    win = _win(cv, bx - pad_x, by - h * 1.3, bx + pad_x, by + h * 0.1)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    rng = np.random.default_rng(seed)
    u = (X - bx) / hw
    v = (by - Y) / h
    vp = np.clip(v, 0, None)
    tb = turbulence
    # advected flow: noise scrolls upward faster than the flame body → licking, curling edges
    ax, ay = advect(u * 0.8, v * 1.4 - 0.3, steps=2, step=0.18 * tb, seed=seed + 11, octaves=2)
    n1 = fbm2d_at(ax, ay * 1.3, 3, seed=seed)
    uw = u + n1 * 0.55 * tb * vp ** 1.15 - lean * (h / hw) * vp ** 2
    vw = v + 0.08 * tb * fbm2d_at(u * 0.9 + 4.1, v * 1.7, 2, seed=seed + 1) * vp
    F = np.full(u.shape, -1.0, F32)
    k = max(int(tongues), 1)
    for i in range(k):
        if i == 0:
            xi, hi, wi = 0.0, 1.0, 0.62
        else:
            side = -1 if i % 2 else 1
            xi = side * rng.uniform(0.2, 0.75) * (0.55 + 0.45 * (i / k))
            hi = rng.uniform(0.38, 0.8) * (1 - 0.45 * abs(xi))
            wi = rng.uniform(0.28, 0.45)
        sway = perlin1d_at(vw * 2.4 + i * 7.3, seed + 20 + i) * 0.32 * tb * np.clip(vw, 0, None) ** 1.2
        t = vw / hi
        tcl = np.clip(t, 0, 1)
        prof = wi * (1 - tcl) ** 0.85 * (0.45 + 0.55 * smoothstep(-0.15, 0.22, t))
        cxl = xi * (1 - 0.55 * tcl) + sway
        d = np.abs(uw - cxl) / np.maximum(prof, 1e-3)
        Fi = (1 - d) * (1 - tcl) ** 0.45 * (1 - 0.15 * abs(xi))
        Fi = np.where(t > 1, -1.0, Fi)
        F = np.maximum(F, Fi)
    # break the top into separate licks with vertically stretched noise
    base = F * smoothstep(-0.07, 0.03, v)
    # vertical gaps split the upper flame into separate tongues; small licks detach at the tips
    ero = fbm2d_at(uw * 2.3 + 9.2, vw * 0.85 - 0.4, 4, seed=seed + 3)
    F = F + ero * 0.5 * tb * smoothstep(0.08, 0.75, vw) - 0.08 * vw
    gap = smoothstep(0.55, 0.92, fbm2d_at(uw * 1.5 + 2.7, vw * 0.45 - 1.3, 3, seed=seed + 5, mode="ridged"))
    F = F - gap * 0.62 * tb * smoothstep(0.12, 0.55, vw) * smoothstep(0.0, 0.25, F)
    F = F * smoothstep(-0.07, 0.03, v)                                   # rounded base
    heat = clamp01(np.minimum(F * 1.15, base * 1.05 + 0.1))
    a = smoothstep(0.0, 0.08, F) * (0.55 + 0.45 * smoothstep(0.0, 0.3, F))
    pal = FIRE if palette is None else palette
    col = _ramp(pal, heat ** 1.15) * (intensity * (0.9 + 1.2 * heat ** 3))[..., None]
    if blue_base:
        bz = smoothstep(0.35, 0.0, v) * smoothstep(0.55, 0.05, heat + 0.25 * v) * blue_base
        col = lerp(col, rgb((0.18, 0.32, 1.0)) * 0.9, bz)
        a = np.maximum(a, smoothstep(0.0, 0.08, F + 0.15) * bz * 0.8)
    if mask is not None:
        a = a * np.asarray(mask, F32)[win[1]:win[3], win[0]:win[2]]
    lightc = rgb((1.0, 0.55, 0.2))
    if glow:
        _soft_glow(cv, bx, by - 0.35 * h, h * 0.7, lightc * 0.45, glow, mask)
    _put(cv, col, a, win)
    return FX(_full(cv, a, win), [Light(bx, by - 0.35 * h, lightc, h * 1.6, 1.2 * intensity)])


def sparks(cv, at=(0.5, 0.7), n=40, spread=180, direction=-90, angle=60, length=18, speed=1.0, size=2.0,
           color=(1.0, 0.62, 0.22), seed=0, gravity=0.0, curl_=0.0, intensity=2.4, region=None):
    """Sparks with motion streaks: ``n`` particles leaving ``at`` towards ``direction`` (degrees, -90 = up)
    within ±``angle``/2, spread up to ``spread`` px; each is a tapered streak ``length`` px long with a hot head.
    ``gravity`` > 0 bends paths down (fireworks, welding); ``curl_`` adds wobble (embers)."""
    S = _S(cv)
    rng = np.random.default_rng(seed)
    cx, cy = _xy(cv, at)
    acc = np.zeros((cv.h, cv.w), F32)
    pts_all = []
    for i in range(int(n)):
        th = math.radians(direction + rng.uniform(-angle / 2, angle / 2))
        dist = spread * S * rng.uniform(0.05, 1.0) ** 0.8 * speed
        if region is not None:
            px0 = rng.uniform(region[0], region[2]) * cv.w
            py0 = rng.uniform(region[1], region[3]) * cv.h
        else:
            px0, py0 = cx + rng.normal(0, 6 * S), cy + rng.normal(0, 3 * S)
        L = length * S * rng.uniform(0.5, 1.3)
        steps = 6
        pts = []
        wob = rng.uniform(0, 100)
        for k in range(steps + 1):
            s = dist - L * (1 - k / steps)
            s = max(s, 0)
            x = px0 + math.cos(th) * s + curl_ * S * 14 * math.sin(s / (40 * S) + wob)
            y = py0 + math.sin(th) * s + gravity * s * s / max(spread * S * speed, 1.0)
            pts.append((x, y))
        pts_all.append((pts, rng.uniform(0.4, 1.0)))
    for pts, b in pts_all:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 6 * S + size * S * 3
        win = _win(cv, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        if win is None:
            continue
        X, Y = win[4], win[5]
        loc = np.zeros(X.shape, F32)
        for k in range(len(pts) - 1):
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            dx, dy = bx - ax, by - ay
            ll = dx * dx + dy * dy + 1e-9
            t = np.clip(((X - ax) * dx + (Y - ay) * dy) / ll, 0, 1)
            d = np.sqrt((X - ax - t * dx) ** 2 + (Y - ay - t * dy) ** 2)
            tt = (k + t) / (len(pts) - 1)
            w = size * S * (0.25 + 0.75 * tt) * 0.6 + 0.35
            loc = np.maximum(loc, np.exp(-(d / w) ** 2) * (0.25 + 0.75 * tt ** 1.5))
        hx, hy = pts[-1]
        loc = loc + _gauss_spot(X, Y, hx, hy, size * S * 1.8) * 0.5
        acc[win[1]:win[3], win[0]:win[2]] = np.maximum(acc[win[1]:win[3], win[0]:win[2]], loc * b)
    col = rgb(color)
    hot = np.broadcast_to(col, (cv.h, cv.w, 3)) * (intensity * (0.6 + acc))[..., None]
    hot = lerp(hot, rgb((1.0, 0.95, 0.75)) * intensity * 1.2, smoothstep(0.7, 1.2, acc))
    cv.paint(hot, clamp01(acc))
    halo = _blur(acc, 4 * S)
    cv.paint(col * 0.8, halo * 0.8, "add")
    return FX(clamp01(acc), [Light(cx, cy, col, spread * S, 0.3)])


def embers(cv, at=(0.5, 0.7), n=50, height=320, width=120, seed=0, color=(1.0, 0.5, 0.15), wind=0.0,
           intensity=2.2):
    """Rising embers: small glowing specks drifting up and sideways from a fire with short curved streaks."""
    S = _S(cv)
    rng = np.random.default_rng(seed + 501)
    cx, cy = _xy(cv, at)
    acc = np.zeros((cv.h, cv.w), F32)
    for i in range(int(n)):
        t = rng.uniform(0.0, 1.0) ** 0.7
        x0 = cx + rng.normal(0, width * 0.25 * S)
        y = cy - t * height * S
        x = x0 + (wind * height * t * t + 28 * math.sin(t * 7 + i) * t) * S
        L = rng.uniform(4, 14) * S
        ang = -math.pi / 2 + rng.normal(0, 0.5) + wind * 0.6
        bx_, by_ = x - math.cos(ang) * L, y - math.sin(ang) * L
        r = rng.uniform(0.9, 2.6) ** 1.3 * S * (1.2 - 0.5 * t)
        pad = L + 4 * r + 2
        win = _win(cv, min(x, bx_) - pad, min(y, by_) - pad, max(x, bx_) + pad, max(y, by_) + pad)
        if win is None:
            continue
        X, Y = win[4], win[5]
        dx, dy = x - bx_, y - by_
        ll = dx * dx + dy * dy + 1e-9
        tt = np.clip(((X - bx_) * dx + (Y - by_) * dy) / ll, 0, 1)
        d = np.sqrt((X - bx_ - tt * dx) ** 2 + (Y - by_ - tt * dy) ** 2)
        streak = np.exp(-(d / (r * 0.8 + 0.3)) ** 2) * tt ** 2 * 0.6
        head = _gauss_spot(X, Y, x, y, r)
        b = (1 - t) ** 0.8 * rng.uniform(0.5, 1.0)
        acc[win[1]:win[3], win[0]:win[2]] += (streak + head) * b
    col = rgb(color)
    cv.paint(col * intensity * (0.7 + 0.6 * clamp01(acc))[..., None], clamp01(acc))
    cv.paint(col, _blur(acc, 6 * S) * 2.0, "add")
    return FX(clamp01(acc), [])


def _log_layer(cv, segs, seed, char_end=None, glow_near=None, bark="#3a2618", wood="#c9955c", light=(-0.5, -0.8)):
    """Draw logs as shaded cylinders with bark grain and glowing cracks near the fire. segs = [(x0,y0,x1,y1,r)]."""
    S = _S(cv)
    total = np.zeros((cv.h, cv.w), F32)
    for j, (x0, y0, x1, y1, r) in enumerate(segs):
        pad = r + 4
        win = _win(cv, min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad)
        if win is None:
            continue
        X, Y = win[4], win[5]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) + 1e-6
        ux, uy = dx / L, dy / L
        t = ((X - x0) * ux + (Y - y0) * uy)
        s = (-(X - x0) * uy + (Y - y0) * ux)                          # signed across
        tc = np.clip(t, 0, L)
        d = np.sqrt((X - x0 - tc * ux) ** 2 + (Y - y0 - tc * uy) ** 2)
        cover = clamp01(r - d + 0.5)
        sn = np.clip(s / r, -1, 1)
        # cylinder shading towards light
        nz = np.sqrt(np.clip(1 - sn * sn, 0, 1))
        lx, ly = light
        ndl = np.clip((sn * (-uy * lx + ux * ly)) * 0.8 + nz * 0.6, 0, 1)
        grain = fbm2d_at(t / (r * 3.0), sn * 3.0 + j * 5, 4, seed=seed + j)
        cracks = smoothstep(0.6, 0.85, fbm2d_at(t / (r * 1.4), sn * 2.0 + j * 3, 4, seed=seed + 40 + j, mode="ridged"))
        col = rgb(bark) * (0.35 + 0.95 * ndl)[..., None] * (1 + 0.35 * grain)[..., None]
        col = lerp(col, col * 0.4, cracks * 0.7)
        # charred, glowing parts near the fire point
        if glow_near is not None:
            gx, gy, gr = glow_near
            near = np.exp(-((X - gx) ** 2 + (Y - gy) ** 2) / (2 * gr * gr))
            ember = near * cracks * smoothstep(0.15, 0.8, near)
            col = lerp(col, col * 0.25, near * 0.8)
            col = col + rgb((1.6, 0.5, 0.1)) * (ember * 1.3)[..., None]
        # cut end (end grain) at the far end x1
        if char_end is None or char_end:
            ed = np.sqrt((X - x1) ** 2 + (Y - y1) ** 2)
            endcap = clamp01(r * 0.92 - ed + 0.5) * (t > L - r * 0.4)
            rings = 0.5 + 0.5 * np.sin(ed / max(r, 1) * 18)
            cap = rgb(wood) * (0.7 + 0.3 * rings)[..., None] * (0.6 + 0.4 * ndl)[..., None]
            col = lerp(col, cap, endcap)
            cover = np.maximum(cover, endcap)
        _put(cv, col, cover, win)
        sub = total[win[1]:win[3], win[0]:win[2]]
        np.maximum(sub, cover, out=sub)
    return total


def _glow_src(cv, light):
    """Light / FX / (x, y, radius) in px → (x, y, radius) or None."""
    if light is None:
        return None
    L = _lights(light) if isinstance(light, (Light, FX, list)) else None
    if L:
        return (L[0].x, L[0].y, L[0].radius)
    return tuple(light) if not L and isinstance(light, tuple) else None


def stone(cv, at=(0.5, 0.85), size=60, seed=0, color="#56514b", light=(-0.4, -0.9), lit_by=None):
    """A rounded rock: noisy silhouette, faceted shading towards ``light`` (direction to the light, screen
    coords) and a warm rim facing ``lit_by`` (a Light / FX such as a campfire). ``size`` = radius (1080p px)."""
    x, y = _xy(cv, at)
    r = size * _S(cv)
    win = _stone(cv, x, y, r, seed, glow_from=_glow_src(cv, lit_by), color=color, light=light)
    m = np.zeros((cv.h, cv.w), F32)
    if win is not None:
        X, Y = win[4], win[5]
        m[win[1]:win[3], win[0]:win[2]] = clamp01(r * 1.25 - np.hypot(X - x, (Y - y) * 1.5) + 0.5)
    return FX(m)


def firewood(cv, start=(0.4, 0.85), end=(0.6, 0.8), width=40, seed=0, bark="#3a2618", wood="#c9955c",
             light=(-0.5, -0.8), lit_by=None, cut_end=True):
    """A wooden log as a shaded cylinder with bark grain, cracks and a ringed cut end at ``end``; parts near
    ``lit_by`` (a Light / FX, e.g. a campfire) char and glow. ``width`` = diameter (1080p px)."""
    x0, y0 = _xy(cv, start)
    x1, y1 = _xy(cv, end)
    g = _glow_src(cv, lit_by)
    if g is not None:
        g = (g[0], g[1], g[2] * 0.12)
    m = _log_layer(cv, [(x0, y0, x1, y1, width * _S(cv) * 0.5)], seed, char_end=cut_end, glow_near=g,
                   bark=bark, wood=wood, light=light)
    return FX(m)


def campfire(cv, at=(0.5, 0.82), size=240, seed=0, logs=True, stones=True, embers_n=45, smoke_=False,
             ground_glow=True, intensity=1.0):
    """Campfire: crossed logs, ring of stones, licking flames, glowing coals, rising embers, warm ground glow.

    ``at`` = base centre of the flames on the ground, ``size`` = flame height (1080p px). Returns FX(mask,
    [light]) — pass the light to :func:`illuminate` to warm nearby rocks / people."""
    S = _S(cv)
    bx, by = _xy(cv, at)
    h = size * S
    rng = np.random.default_rng(seed)
    lightc = rgb((1.0, 0.55, 0.22))
    if ground_glow:
        win = _win(cv, bx - h * 2.6, by - h * 0.5, bx + h * 2.6, by + h * 0.8)
        if win is not None:
            X, Y = win[4], win[5]
            e = ((X - bx) / (h * 1.4)) ** 2 + ((Y - by) / (h * 0.36)) ** 2
            g = (np.exp(-e * 2.2) * 0.8 + 0.3 / (1 + 6 * e)) * smoothstep(by - h * 0.45, by + h * 0.05, Y)
            g = g * (1 - smoothstep(4.0, 9.0, e))
            _put(cv, lightc * 0.45 * intensity, g, win, "add")
    if smoke_:
        smoke(cv, at=(at[0], at[1] - 0.9 * h / cv.h), size=size * 2.4, width=size * 0.5, seed=seed + 3,
              color="#4b4846", opacity=0.45, wind=0.15)
    lw = h * 0.042
    back, front = [], []
    if logs:
        n_logs = 7
        for i in range(n_logs):
            a = 2 * math.pi * (i + rng.uniform(-0.2, 0.2)) / n_logs + 0.3
            R = h * rng.uniform(0.34, 0.42)
            ex, ey = bx + math.cos(a) * R, by + math.sin(a) * R * 0.3 + lw * 0.4
            tipx, tipy = bx + math.cos(a) * h * 0.04, by - h * rng.uniform(0.22, 0.3)
            seg = (tipx, tipy, ex, ey, lw * rng.uniform(0.85, 1.15))
            (back if math.sin(a) < 0 else front).append(seg)
        back.sort(key=lambda q: q[3])
        front.sort(key=lambda q: q[3])
    stones_back, stones_front = [], []
    if stones:
        k = 11
        for i in range(k):
            a = 2 * math.pi * i / k + rng.uniform(-0.1, 0.1)
            sx, sy = bx + math.cos(a) * h * 0.85, by + math.sin(a) * h * 0.24 + h * 0.06
            sr = h * rng.uniform(0.085, 0.12)
            (stones_back if math.sin(a) < 0 else stones_front).append((sx, sy, sr, rng.integers(1 << 30)))
    for st in stones_back:
        _stone(cv, *st, glow_from=(bx, by - h * 0.2, h * 1.4))
    _log_layer(cv, back, seed + 7, glow_near=(bx, by - h * 0.08, h * 0.16))
    # coals bed
    win = _win(cv, bx - h * 0.7, by - h * 0.25, bx + h * 0.7, by + h * 0.15)
    if win is not None:
        X, Y = win[4], win[5]
        e = ((X - bx) / (h * 0.42)) ** 2 + ((Y - by + h * 0.02) / (h * 0.095)) ** 2
        cu, cvv = (X - bx) / (h * 0.05), (Y - by) / (h * 0.035)
        cu = cu + 0.45 * fbm2d_at(cu * 0.5, cvv * 0.5, 3, seed=seed + 6)
        cvv = cvv + 0.45 * fbm2d_at(cu * 0.5 + 4, cvv * 0.5, 3, seed=seed + 7)
        f1, f2, edge, cid = worley(cu, cvv, seed + 5)
        chunk = smoothstep(0.03, 0.16, edge) * (0.75 + 0.25 * fbm2d_at(cu * 2, cvv * 2, 3, seed=seed + 8))
        hot = clamp01(1 - e) ** 0.7 * (0.3 + 0.7 * cid) * chunk
        col = lerp(rgb("#120805"), rgb((1.8, 0.55, 0.12)), hot ** 1.4)
        _put(cv, col, clamp01((1.15 - e) * 4), win)
    fx = flame(cv, at=(at[0], (by - h * 0.04) / cv.h), size=size, width=size * 0.8, seed=seed + 1, tongues=8,
               intensity=1.15 * intensity, glow=0.9)
    _log_layer(cv, front, seed + 9, glow_near=(bx, by - h * 0.12, h * 0.14))
    for j in range(2):                                                 # small tongues licking over the front logs
        fxj = bx + (j - 0.5) * h * 0.3 + rng.normal(0, h * 0.05)
        sz = size * rng.uniform(0.22, 0.34)
        flame(cv, at=(fxj / cv.w, (by - h * 0.01) / cv.h), size=sz, width=sz * rng.uniform(0.5, 0.8), seed=seed + 30 + j,
              tongues=3, glow=0, intensity=1.05 * intensity, lean=rng.uniform(-0.15, 0.15))
    for st in stones_front:
        _stone(cv, *st, glow_from=(bx, by - h * 0.2, h * 1.4))
    if embers_n:
        embers(cv, at=(at[0], (by - h * 0.4) / cv.h), n=embers_n, height=size * 1.8, width=size * 0.5, seed=seed + 2)
    return FX(fx.mask, [Light(bx, by - h * 0.35, lightc, h * 2.2, 1.3 * intensity)])


def _stone(cv, x, y, r, seed, glow_from=None, color="#56514b", light=(-0.4, -0.9)):
    """A rounded rock with noisy silhouette, shading and an optional warm rim from a nearby fire."""
    S = _S(cv)
    win = _win(cv, x - r * 1.5, y - r * 1.2, x + r * 1.5, y + r * 1.0)
    if win is None:
        return None
    X, Y = win[4], win[5]
    nx, ny = (X - x) / (r * 1.25), (Y - y) / (r * 0.8)
    ang = np.arctan2(ny, nx)
    wob = 1 + 0.12 * perlin1d_at((ang + math.pi) * 1.6, int(seed) % 9973) + 0.05 * perlin1d_at((ang + math.pi) * 5, int(seed) % 9973 + 1)
    rr = np.sqrt(nx * nx + ny * ny) / wob
    cover = clamp01((1 - rr) * r * 0.8 + 0.5) * (ny < 0.55)
    nz = np.sqrt(np.clip(1 - rr * rr, 0, 1))
    lx, ly = light
    ndl = clamp01(nx * lx + ny * ly + nz * 0.5)
    tex = fbm2d_at(X / (r * 0.35), Y / (r * 0.35), 5, seed=int(seed) % 9973)
    _, _, fe, fid = worley(X / (r * 0.7), Y / (r * 0.7), int(seed) % 9973 + 2)
    facet = (fid - 0.5) * 0.22 - 0.1 * (1 - smoothstep(0.0, 0.06, fe))
    col = rgb(color) * clamp01(0.2 + 0.6 * ndl + 0.18 * tex + facet * ndl)[..., None]
    col = col * (1 - 0.5 * smoothstep(0.1, 0.55, ny))[..., None]
    if glow_from is not None:
        gx, gy, gr = glow_from
        dxl, dyl = gx - x, gy - y
        L = math.hypot(dxl, dyl) + 1e-6
        face = clamp01((nx * dxl + ny * dyl) / L * 0.9 + nz * 0.3)
        warm = 1.0 / (1 + (L / gr) ** 2 * 2.5)
        col = col + rgb((1.0, 0.45, 0.15)) * (face * warm * 0.9)[..., None]
    _put(cv, col, cover, win)
    return win


def candle(cv, at=(0.5, 0.6), size=60, seed=0, body=True, body_height=None, wax="#efe3c8", glow=1.0, base=None,
           light_on=None):
    """Candle: teardrop flame (blue base, dark wick, white core, orange edge) with a halo, on an optional wax
    body. ``at`` = wick base, ``size`` = flame height (1080p px), ``body_height`` = wax height (1080p px,
    default 2.6 * size). ``base=(u, v)`` stands the candle on a surface instead: the bottom of the wax sits at
    ``base`` and ``at`` is ignored. ``light_on`` = mask of the surfaces the flame lights (sill, table, wall); it
    is lit before the wax is drawn, so the candle itself is not tinted twice."""
    S = _S(cv)
    h = size * S
    bh = (body_height or size * 2.6) * S
    if base is not None:
        bx_, by_ = _xy(cv, base)
        at = (bx_ / cv.w, (by_ - bh - h * 0.04) / cv.h)
    cx, cy = _xy(cv, at)
    lt = Light(cx, cy - h * 0.5, (1.0, 0.65, 0.3), h * 4, 1.0)
    if light_on is not None:
        illuminate(cv, lt, mask=light_on, strength=1.6, ambient=0.12)
    if base is not None and body:                                       # soft contact shadow on the surface
        bw_ = h * 0.42
        swin = _win(cv, bx_ - bw_ * 2, by_ - bw_ * 0.5, bx_ + bw_ * 2, by_ + bw_ * 0.6)
        if swin is not None:
            d2 = ((swin[4] - bx_) / (bw_ * 1.5)) ** 2 + ((swin[5] - by_ - bw_ * 0.08) / (bw_ * 0.32)) ** 2
            _put(cv, rgb("#000000"), np.exp(-d2 * 2.0) * 0.55, swin)
    if body:
        bw = h * 0.42
        ry = bw * 0.22                                              # perspective of the top
        win = _win(cv, cx - bw - 2, cy - ry - 2, cx + bw + 2, cy + bh + ry + 2)
        if win is not None:
            X, Y = win[4], win[5]
            sx = (X - cx) / bw
            top = cy + h * 0.04
            rim = top + ry * np.sqrt(np.clip(1 - sx * sx, 0, 1))          # front edge of the top ellipse
            cover = clamp01(bw - np.abs(X - cx) + 0.5) * clamp01(Y - (top - ry * np.sqrt(np.clip(1 - sx * sx, 0, 1))) + 0.5)
            cover = cover * clamp01(top + bh + ry * np.sqrt(np.clip(1 - sx * sx, 0, 1)) - Y + 0.5)
            nz = np.sqrt(np.clip(1 - sx * sx, 0, 1))
            dist = np.clip(Y - rim, 0, None)
            lit = 1.0 / (1.0 + (dist / (h * 1.4)) ** 2)                 # light from the flame falls off down the body
            col = rgb(wax) * (0.06 + (0.12 + 0.5 * lit) * nz ** 0.9)[..., None]
            sss = np.exp(-dist / (h * 0.45)) * (0.5 + 0.5 * nz)           # wax glowing through near the flame
            col = col + rgb((1.0, 0.58, 0.24)) * (sss * 0.4)[..., None]
            drip = smoothstep(0.55, 0.85, perlin1d_at(sx * 3 + 2, seed)) * clamp01(1 - dist / (h * (0.25 + 0.5 * (perlin1d_at(sx * 5, seed + 1) + 1))))
            col = col + rgb(wax) * (drip * 0.12)[..., None]
            pool = clamp01((rim - Y) + 0.5) * cover                     # the melted pool on top, lit by the flame
            pcol = rgb((1.0, 0.72, 0.42)) * (0.8 + 0.4 * (1 - np.clip(np.abs(sx), 0, 1)))[..., None]
            col = lerp(col, pcol, pool * 0.85)
            edge = np.exp(-((Y - rim) / max(1.2 * S, 0.6)) ** 2) * nz
            col = col + rgb((1.0, 0.8, 0.55)) * (edge * 0.5)[..., None]
            _put(cv, col, cover, win)
    if glow:
        _soft_glow(cv, cx, cy - h * 0.55, h * 1.8, rgb((1.0, 0.62, 0.3)) * 0.55, glow)
    win = _win(cv, cx - h * 0.4, cy - h * 1.2, cx + h * 0.4, cy + h * 0.12)
    X, Y = win[4], win[5]
    n = perlin1d_at((cy - Y) / h * 1.5, seed + 3) * 0.06 * h * np.clip((cy - Y) / h, 0, 1) ** 2
    u = (X - cx - n) / (h * 0.17)
    v = (cy - Y) / h                                                  # 0 at wick base, 1 at tip
    prof = np.where(v < 0.3, np.sqrt(np.clip(1 - ((0.3 - v) / 0.34) ** 2, 0, 1)), np.clip(1 - (v - 0.3) / 0.7, 0, 1) ** 0.85)
    prof = np.clip(prof, 0, None)
    d = np.abs(u) / np.maximum(prof, 1e-3)
    F = clamp01(1 - d) * (v > -0.04) * (v < 1)
    a = smoothstep(0.0, 0.25, F)
    core = smoothstep(0.25, 0.75, F) * smoothstep(0.12, 0.3, v) * smoothstep(0.9, 0.45, v)
    col = lerp(rgb((1.0, 0.5, 0.12)) * 1.4, rgb((1.0, 0.95, 0.8)) * 2.6, core)
    blue = smoothstep(0.28, 0.0, v) * (1 - core)
    col = lerp(col, rgb((0.25, 0.4, 1.0)) * 0.9, blue * 0.85)
    a = a * (1 - 0.45 * blue)
    _put(cv, col, a, win)
    # wick
    wwin = _win(cv, cx - 3 * S, cy - h * 0.2, cx + 3 * S, cy + h * 0.06)
    if wwin is not None:
        X, Y = wwin[4], wwin[5]
        wk = clamp01(1.3 * S - np.abs(X - cx) + 0.5) * (Y > cy - h * 0.16)
        _put(cv, rgb("#1a120c"), wk * 0.9, wwin)
        _put(cv, rgb((1.6, 0.5, 0.1)), wk * (Y < cy - h * 0.1) * 0.8, wwin)
    return FX(_full(cv, a, win), [lt])


def torch(cv, at=(0.5, 0.5), size=150, seed=0, angle=0.0, handle="#5a3a22", wrap="#3a2a1e"):
    """Hand torch: wooden handle, cloth-wrapped head, flame and sparks. ``at`` = head top, ``size`` = flame
    height; ``angle`` tilts the handle (degrees)."""
    S = _S(cv)
    cx, cy = _xy(cv, at)
    h = size * S
    a = math.radians(angle)
    L = h * 1.9
    ex, ey = cx - math.sin(a) * L, cy + math.cos(a) * L
    _log_layer(cv, [(ex, ey, cx, cy + h * 0.25, h * 0.07)], seed + 3, char_end=False, bark=handle)
    win = _win(cv, cx - h * 0.25, cy - h * 0.1, cx + h * 0.25, cy + h * 0.45)
    if win is not None:
        X, Y = win[4], win[5]
        u = (X - cx) / (h * 0.15)
        v = (Y - cy) / (h * 0.36)
        cover = clamp01((1 - np.abs(u) ** 2.5 - np.abs(v - 0.5) ** 6 * 4) * h * 0.1)
        bands = 0.5 + 0.5 * np.sin((v * 7 + u * 1.2) * math.pi)
        col = rgb(wrap) * (0.4 + 0.6 * np.sqrt(np.clip(1 - u * u, 0, 1)))[..., None] * (0.6 + 0.4 * bands)[..., None]
        col = col + rgb((1.4, 0.4, 0.08)) * (smoothstep(0.35, -0.1, v) * 0.6)[..., None]
        _put(cv, col, cover, win)
    fx = flame(cv, at=(cx / cv.w, (cy + h * 0.06) / cv.h), size=size, width=size * 0.42, seed=seed, tongues=5,
               lean=0.08, glow=0.8)
    sparks(cv, at=(cx / cv.w, (cy - h * 0.4) / cv.h), n=10, spread=size * 1.2, length=10, angle=50, seed=seed + 5,
           curl_=0.6, size=1.2)
    return fx


def fireplace(cv, box=(0.25, 0.35, 0.75, 0.9), seed=0, stone="#8c7a68", mortar="#3d342c", fire_size=None):
    """Fireplace: brick surround with mortar lines, arched dark firebox, logs and flames, warm light spilling on
    the bricks and hearth. ``box`` = (u0, v0, u1, v1) of the whole surround."""
    S = _S(cv)
    x0, y0 = _xy(cv, box[:2])
    x1, y1 = _xy(cv, box[2:])
    W_, H_ = x1 - x0, y1 - y0
    win = _win(cv, x0, y0, x1, y1)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    # bricks: running bond, per-brick colour, bevelled mortar
    bh = max(H_ / 14.0, 4)
    bw = bh * 2.2
    row = np.floor((Y - y0) / bh)
    xo = (X - x0) / bw + 0.5 * (row % 2)
    col_i = np.floor(xo)
    fy = (Y - y0) / bh - row
    fx_ = xo - col_i
    edge = np.minimum(np.minimum(fy, 1 - fy) * bh, np.minimum(fx_, 1 - fx_) * bw)
    bid = _hashf(col_i, row, seed)
    brick = rgb(stone) * (0.7 + 0.5 * bid)[..., None] * (1 + 0.15 * fbm2d_at(X / (bh * 0.6), Y / (bh * 0.6), 3, seed=seed + 1))[..., None]
    brick = brick * (0.85 + 0.15 * smoothstep(0, bh * 0.25, edge))[..., None]
    surround = lerp(rgb(mortar), brick, smoothstep(0.6 * S, 2.2 * S, edge))
    # firebox opening with an arch
    ox0, ox1 = x0 + W_ * 0.2, x1 - W_ * 0.2
    oy0, oy1 = y0 + H_ * 0.28, y1 - H_ * 0.08
    ocx = (ox0 + ox1) / 2
    rw = (ox1 - ox0) / 2
    arch = oy0 + rw * 0.35 - np.sqrt(np.clip(1 - ((X - ocx) / rw) ** 2, 0, 1)) * rw * 0.35
    opening = clamp01(rw - np.abs(X - ocx) + 0.5) * clamp01(Y - arch + 0.5) * clamp01(oy1 - Y + 0.5)
    _put(cv, surround, np.ones_like(X), win)
    inner_y = (Y - oy0) / (oy1 - oy0)
    soot = lerp(rgb("#1a0e08"), rgb("#050302"), clamp01(1 - inner_y))
    _put(cv, soot, opening, win)
    # hearth
    hwin = _win(cv, x0 - W_ * 0.04, oy1, x1 + W_ * 0.04, oy1 + H_ * 0.08)
    if hwin is not None:
        Xh, Yh = hwin[4], hwin[5]
        hc = rgb(stone) * 0.9 * (1 + 0.4 * smoothstep(oy1 + H_ * 0.02, oy1, Yh))[..., None]
        _put(cv, hc, np.ones_like(Xh), hwin)
    fs = fire_size or (oy1 - oy0) / S * 0.7
    fbx, fby = ocx, oy1 - H_ * 0.06
    lw = (ox1 - ox0) * 0.05
    _log_layer(cv, [(ox0 + rw * 0.25, fby - lw * 0.2, ox1 - rw * 0.2, fby - lw * 1.4, lw),
                    (ox1 - rw * 0.3, fby + lw * 0.1, ox0 + rw * 0.35, fby - lw * 0.6, lw * 1.05)], seed + 4,
               glow_near=(fbx, fby, rw * 0.5))
    fx = flame(cv, at=(fbx / cv.w, fby / cv.h), size=fs, width=fs * 0.9, seed=seed + 2, tongues=8, glow=0.6,
               mask=_full(cv, opening, win) if True else None)
    # firelight on the surround and floor
    lt = Light(fbx, fby - fs * S * 0.3, (1.0, 0.55, 0.2), W_ * 0.7, 1.4)
    m = _full(cv, 1 - opening, win)
    illuminate(cv, lt, mask=m, strength=0.9)
    return FX(_full(cv, np.ones_like(X), win), [lt], opening=_full(cv, opening, win))


def explosion(cv, at=(0.5, 0.55), size=300, seed=0, smoke_color="#2a2420", debris=40, shock=False, intensity=1.0):
    """Explosion / fireball: billowing domain-warped fireball with a white-hot core, orange body, dark smoke
    rim, shock ring and flying debris sparks. ``size`` = radius (1080p px)."""
    S = _S(cv)
    cx, cy = _xy(cv, at)
    r = size * S
    win = _win(cv, cx - r * 1.6, cy - r * 1.7, cx + r * 1.6, cy + r * 1.3)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    u, v = (X - cx) / r, (Y - cy) / r
    _soft_glow(cv, cx, cy, r * 1.4, rgb((1.0, 0.45, 0.12)) * 0.6, 1.0 * intensity)
    wx = u + 0.25 * fbm2d_at(u * 1.2, v * 1.2, 3, seed=seed + 1)
    wy = v + 0.25 * fbm2d_at(u * 1.2 + 5.2, v * 1.2, 3, seed=seed + 2)
    bil = fbm2d_at(wx * 2.2, wy * 2.2, 5, seed=seed, mode="billow")
    lobes = fbm2d_at(wx * 0.9 + 3.3, wy * 0.9, 3, seed=seed + 6)
    rr = np.sqrt(u * u + (v * 1.05 + 0.08 * u * u) ** 2)
    dens = 1.0 - rr + 0.5 * (bil - 0.3) + 0.35 * lobes - 0.15 * np.clip(v, 0, None)
    a = smoothstep(-0.02, 0.18, dens) * (0.75 + 0.25 * smoothstep(0.05, 0.3, dens))   # smoke thins at the edge
    heat = clamp01((dens - 0.05) * 1.5) * clamp01(1.3 - rr) * (0.55 + 0.7 * bil)
    smoke_part = smoothstep(0.35, 0.0, heat) * smoothstep(0.55, 1.0, rr + 0.2 * bil)
    fire = _ramp(FIRE, clamp01(heat * 1.3)) * (intensity * (1.0 + 2.0 * heat ** 2))[..., None]
    # shade the billows: brighter where the noise bulges toward the viewer
    shade, _ = _shade_from_density(dens, (0.0, 1.0), r * 0.03)
    sh = 0.45 + 0.5 * bil + 0.6 * shade
    smk = rgb(smoke_color) * sh[..., None] + rgb((0.9, 0.3, 0.06)) * (heat * 0.6 + shade * 0.15)[..., None]
    col = lerp(fire, smk, clamp01(smoke_part * 1.3))
    _put(cv, col, a, win)
    if shock:
        d = np.sqrt(u * u + (v * 2.6 - 1.4) ** 2)
        ring = np.exp(-((d - 1.35) / 0.06) ** 2) * (v > 0.3) * 0.35
        _put(cv, rgb((1.0, 0.75, 0.45)), ring * intensity, win, "add")
    if debris:
        sparks(cv, at=at, n=debris, spread=size * 1.5, direction=-90, angle=300, length=30, gravity=0.6,
               seed=seed + 4, size=1.8, intensity=2.4)
    return FX(_full(cv, a, win), [Light(cx, cy, (1.0, 0.55, 0.2), r * 3, 2.0 * intensity)])


def fireworks(cv, at=(0.5, 0.3), size=220, color="#ff5a8a", seed=0, kind="peony", trails=True, n=None,
              secondary=None, intensity=2.0):
    """Firework burst: ``kind`` = peony (round, dotted ends) / willow (drooping long trails) / ring / chrysanthemum
    (long straight streaks). ``secondary`` adds a second colour at the tips. ``size`` = burst radius."""
    S = _S(cv)
    rng = np.random.default_rng(seed)
    cx, cy = _xy(cv, at)
    r = size * S
    _choice("fireworks", "kind", kind, ("peony", "willow", "ring", "chrysanthemum"))
    n = n or {"peony": 70, "willow": 55, "ring": 48, "chrysanthemum": 90}.get(kind, 70)
    win = _win(cv, cx - r * 1.4, cy - r * 1.3, cx + r * 1.4, cy + r * (2.2 if kind == "willow" else 1.5))
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    acc = np.zeros(X.shape, F32)
    tip = np.zeros(X.shape, F32)
    grav = {"willow": 0.9, "peony": 0.25, "ring": 0.15, "chrysanthemum": 0.3}.get(kind, 0.25)
    for i in range(n):
        th = 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n
        if kind == "ring":
            rx, ry = math.cos(th), math.sin(th) * 0.45
        else:
            phi = math.acos(rng.uniform(-1, 1))                      # 3D sphere → projected disc
            rx, ry = math.cos(th) * math.sin(phi), math.sin(th) * math.sin(phi)
        sp = rng.uniform(0.85, 1.0)
        steps = 10
        pts = []
        tail0 = {"willow": 0.0, "chrysanthemum": 0.15, "peony": 0.55, "ring": 0.5}.get(kind, 0.5)
        for k in range(steps + 1):
            t = tail0 + (1 - tail0) * k / steps
            px = cx + rx * r * sp * t
            py = cy + ry * r * sp * t + grav * r * t * t * (1.0 if kind != "willow" else 1.3)
            pts.append((px, py))
        for k in range(steps):
            (ax_, ay_), (bx_, by_) = pts[k], pts[k + 1]
            dx, dy = bx_ - ax_, by_ - ay_
            ll = dx * dx + dy * dy + 1e-9
            t = np.clip(((X - ax_) * dx + (Y - ay_) * dy) / ll, 0, 1)
            d = np.sqrt((X - ax_ - t * dx) ** 2 + (Y - ay_ - t * dy) ** 2)
            fr = (k + t) / steps
            w = S * (0.6 + 1.0 * fr) if kind != "willow" else S * (0.5 + 0.6 * (1 - fr))
            bright = (fr ** 1.5) if kind != "willow" else (0.35 + 0.65 * (1 - fr) ** 0.5) * (1 - fr ** 3)
            acc = np.maximum(acc, np.exp(-(d / w) ** 2) * bright)
        if kind in ("peony", "ring", "chrysanthemum"):
            ex, ey = pts[-1]
            tip = np.maximum(tip, _gauss_spot(X, Y, ex, ey, 2.2 * S))
            if trails and kind == "peony":
                for q in range(3):                                   # glittering dots trailing the star
                    qx, qy = pts[-3 - 2 * q]
                    tip = np.maximum(tip, _gauss_spot(X, Y, qx + rng.normal(0, 2 * S), qy + rng.normal(0, 2 * S), 1.2 * S) * 0.7)
    c1 = rgb(color)
    c2 = rgb(secondary) if secondary is not None else lerp(c1, rgb("#ffffff"), 0.6)
    col = c1 * (acc * intensity)[..., None] + c2 * (tip * intensity * 1.3)[..., None]
    _put(cv, col, np.ones_like(acc), win, "add")
    halo = _blur(acc + tip, 10 * S)
    _put(cv, c1 * 0.9, halo * 0.9, win, "add")
    _soft_glow(cv, cx, cy, r * 0.6, c1 * 0.25, 0.8)
    return FX(_full(cv, clamp01(acc + tip), win), [Light(cx, cy, c1, r * 3, 0.8)])


# =============================================================================
# clouds
# =============================================================================
def _shade_from_density(D, light, sigma):
    """Lambert-like shading of a density field: brighter where the (blurred) surface faces ``light``."""
    Db = gaussian_blur(D, sigma)
    gy, gx = np.gradient(Db)
    lx, ly = light
    n = math.hypot(lx, ly) + 1e-9
    lx, ly = lx / n, ly / n
    g = np.sqrt(gx * gx + gy * gy) + 1e-6
    face = -(gx * lx + gy * ly) / g                      # -grad points outward
    mag = smoothstep(0.0, 0.02, g)
    return clamp01(0.5 + 0.5 * face * mag), Db


def cumulus(cv, at=(0.5, 0.4), size=520, seed=0, light=(0.6, -0.8), lit="#fffaf0", shadow="#8f9bb4",
            base="#6e7a92", rim=0.5, puffs=14, opacity=1.0, height=0.55):
    """Fair-weather cumulus: cauliflower puffs on a flat, shaded base; lit side towards ``light`` (direction to
    the sun, screen coords) and a silver rim. ``at`` = centre of the flat base, ``size`` = width (1080p px)."""
    S = _S(cv)
    rng = np.random.default_rng(seed)
    w = size * S
    bx, by = _xy(cv, at)
    hg = w * height
    win = _win(cv, bx - w * 0.62, by - hg * 1.25, bx + w * 0.62, by + w * 0.06)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    D = np.full(X.shape, -1.0, F32)
    for i in range(int(puffs)):
        x = float(np.clip(rng.normal(0, 0.2), -0.4, 0.4))
        dome = (1 - abs(x) / 0.48) ** 0.7
        r = w * rng.uniform(0.1, 0.17) * (0.55 + 0.6 * dome)
        y = by - r * rng.uniform(0.25, 0.6) - dome * hg * rng.uniform(0.25, 0.6)
        d = 1 - np.sqrt((X - bx - x * w) ** 2 + (Y - y) ** 2) / r
        D = np.maximum(D, d)
    sc = w * 0.06
    bil = fbm2d_at(X / sc, Y / sc, 5, seed=seed + 1, mode="billow")
    D = D + 0.45 * (bil - 0.32)
    wob = fbm2d_at(X / (w * 0.15), np.full_like(X, 3.3), 2, seed=seed + 2) * w * 0.015
    flat = smoothstep(by + w * 0.008, by - w * 0.03, Y + wob)
    D = D * flat - (1 - flat) * 0.5
    a = smoothstep(0.0, 0.14, D) * opacity
    shade, Db = _shade_from_density(D, light, w * 0.03)
    detail = clamp01(0.5 + (bil - 0.45) * 1.6)
    s = clamp01(shade * 0.75 + detail * 0.35 - 0.05)
    col = lerp(shadow, lit, s)
    under = smoothstep(by - hg * 0.35, by, Y)
    col = lerp(col, rgb(base), under * 0.75 * (1 - 0.5 * s))
    if rim:
        edge = smoothstep(0.0, 0.05, D) * (1 - smoothstep(0.05, 0.22, D))
        col = col + rgb(lit) * (edge * shade * rim * 0.9)[..., None]
    _put(cv, col, a, win)
    return FX(_full(cv, a, win))


def storm_clouds(cv, region=(0.0, 0.0, 1.0, 0.6), seed=0, layers=3, color="#252a33", light="#7a8494",
                 backlight=None, rim="#d6deea", scale=380, coverage=0.55):
    """Storm clouds: dark, heavy banks in layers — a pale far deck, then lumpy masses with scalloped, darker
    undersides; nearer layers darker and larger. Edges glow when ``backlight=(u, v)`` (a hidden sun / moon /
    lightning) is behind them. Returns FX(mask) to pass to :func:`lightning`."""
    S = _S(cv)
    H, W = cv.h, cv.w
    X, Y = cv.xy()
    u0, v0, u1, v1 = region
    total = np.zeros((H, W), F32)
    bl = None
    if backlight is not None:
        bxp, byp = _xy(cv, backlight)
        bl = np.exp(-(((X - bxp) ** 2 + (Y - byp) ** 2) / (0.5 * H) ** 2))
    span = (v1 - v0) * H
    for i in range(int(layers)):
        f = i / max(layers - 1, 1)                                     # 0 far (pale deck) → 1 near (dark masses)
        yb = v0 * H + span * (0.55 + 0.45 * f)
        sc = scale * S * (0.8 + 0.7 * f)
        edge_y = yb + span * 0.12 * fbm1d(W, sc * 1.2, 4, seed=seed + 10 * i)[None, :]
        n = fbm2d(H, W, sc, 6, seed=seed + 3 + 10 * i, aspect=0.6, mode="billow")
        big = fbm2d(H, W, sc * 2.5, 3, seed=seed + 4 + 10 * i, aspect=0.5)
        lumps = fbm2d(H, W, sc * 0.3, 4, seed=seed + 5 + 10 * i, mode="billow")
        cov = coverage + 0.25 * (1 - f)
        D = n * 0.9 + big * 0.45 + lumps * 0.3 + cov - 0.95
        below = smoothstep(edge_y - sc * 0.05, edge_y + sc * 0.25 * (0.4 + lumps), Y)
        D = D - below * 1.2
        if i == 0:
            D = np.maximum(D, (1 - smoothstep(yb - span * 0.3, yb, Y)) * (0.05 + 0.35 * clamp01(n + big * 0.5)))
        D = D * _region_mask(cv, (u0 - 0.2, -1, u1 + 0.2, 2))
        a = smoothstep(0.0, 0.1, D)
        shade, _ = _shade_from_density(D, (0.15, -1.0), sc * 0.12)
        thick = smoothstep(0.0, 0.6, D)
        tone = clamp01(0.3 + 0.4 * shade * smoothstep(0.0, 0.25, D) + 0.45 * (lumps - 0.4) + 0.25 * (n - 0.4) - 0.25 * thick * f)
        dark = rgb(color) * (1.15 - 0.45 * f)
        litc = lerp(rgb(light), dark, 0.15 + 0.5 * f)
        col = lerp(dark, litc, tone)
        under = smoothstep(edge_y - sc * 0.35, edge_y + sc * 0.1, Y)
        col = col * (1 - 0.35 * under)[..., None]
        if bl is not None:
            thin = smoothstep(0.0, 0.05, D) * (1 - smoothstep(0.05, 0.3, D))
            thin = _blur(thin, 2.5 * S) * smoothstep(0.0, 0.02, D)
            col = col + rgb(rim) * (thin * 0.45 * bl * bl * (1.0 - 0.3 * f) * (1 - total))[..., None]
        cv.paint(col, a)
        total = np.maximum(total, a)
    return FX(total)


def cirrus(cv, region=(0.0, 0.0, 1.0, 0.45), seed=0, color="#f4f1ec", angle=-12, opacity=0.55, scale=420):
    """Cirrus: high, wispy, hair-like streaks combed along ``angle`` (degrees) with feathered hooks."""
    S = _S(cv)
    H, W = cv.h, cv.w
    m = _region_mask(cv, region)
    m = gaussian_blur(m, 40 * S) if region is not None else m
    X, Y = cv.xy()
    a = math.radians(angle)
    k = 2
    Xs, Ys = X[::k, ::k], Y[::k, ::k]
    U = (Xs * math.cos(a) + Ys * math.sin(a)) / (scale * S)
    V = (-Xs * math.sin(a) + Ys * math.cos(a)) / (scale * S * 0.07)
    wx = fbm2d_at(U * 0.7, V * 0.08, 3, seed=seed + 1)
    U2, V2 = U + 0.6 * wx, V + 3.0 * fbm2d_at(U * 1.5 + 4, V * 0.05, 2, seed=seed + 2)
    streak = fbm2d_at(U2 * 2.0, V2, 5, seed=seed, mode="ridged")
    cover = smoothstep(-0.1, 0.45, fbm2d_at(U * 0.8 + 9, V * 0.06, 3, seed=seed + 3))
    al = smoothstep(0.35, 0.85, streak) * cover
    al = resize(al, W, H, "bilinear")
    cv.paint(color, al * m * opacity)
    return FX(al * m * opacity)


def cloud_bands(cv, region=(0.0, 0.3, 1.0, 0.62), seed=0, colors=("#3a2c52", "#a2506a", "#ff9a62", "#ffd9a0"),
                light_below=True, opacity=0.95, scale=340):
    """Sunset / dawn stratus bands: long horizontal clouds lit from below (warm undersides, cool tops).
    ``colors`` = shadow → lit ramp."""
    S = _S(cv)
    H, W = cv.h, cv.w
    m = _region_mask(cv, region)
    n = fbm2d(H, W, scale * S, 6, seed=seed, aspect=0.13)
    n2 = fbm2d(H, W, scale * S * 0.4, 4, seed=seed + 1, aspect=0.2, mode="billow")
    D = n * 1.1 + n2 * 0.35 - 0.12
    a = smoothstep(0.0, 0.25, D) * m * opacity
    shade, _ = _shade_from_density(D, (0.0, 1.0 if light_below else -1.0), 6 * S)
    X, Y = cv.xy()
    vpos = (Y / H - region[1]) / max(region[3] - region[1], 1e-6)
    t = clamp01(shade * 0.8 + (vpos if light_below else 1 - vpos) * 0.35 + (n2 - 0.4) * 0.4)
    col = _ramp(colors, t)
    edge = smoothstep(0.0, 0.06, D) * (1 - smoothstep(0.06, 0.3, D)) * shade
    col = col + _ramp(colors, np.ones_like(t)) * (edge * 0.6)[..., None]
    cv.paint(col, a)
    return FX(a)


# =============================================================================
# smoke and vapour
# =============================================================================
_SMOKE_PHASE_R = 1.5          # radius of the noise-space circle that ``smoke(phase=...)`` travels


def smoke(cv, at=(0.5, 0.8), size=600, width=None, spread=0.35, wind=0.25, seed=0, color="#9a9aa0",
          shade="#3e3f46", opacity=0.75, light=(0.8, -0.4), curl=1.0, angle=0.0, sway=0.06, mask=None, softness=1.0,
          phase=0.0, offset=(0.0, 0.0)):
    """Rising smoke column that widens, curls and thins out. ``at`` = source, ``size`` = column length,
    ``width`` = source width (default 0.06 * size), ``spread`` = extra width at the top (fraction of length),
    ``wind`` = sideways drift at the top (fraction of length), ``angle`` = plume direction (0 up, 90 right),
    ``curl`` = turbulence, ``softness`` = edge softness (1 = billowy column, 3 = a soft vapour puff).
    ``phase`` / ``offset`` move the turbulence for animation: ``offset=(ox, oy)`` slides the noise field, and
    ``phase`` (turns; 0 and 1 give the same picture) drives it once around a circle, so ``phase=T`` with T from 0
    to 1 is a seamless loop. The column itself (position, width, wind) stays put."""
    S = _S(cv)
    h = size * S
    bx, by = _xy(cv, at)
    ox, oy = (float(offset), 0.0) if np.isscalar(offset) else (float(offset[0]), float(offset[1]))
    if phase:
        ox += _SMOKE_PHASE_R * (math.cos(2 * math.pi * phase) - 1.0)
        oy += _SMOKE_PHASE_R * math.sin(2 * math.pi * phase)
    w0 = (width if width is not None else size * 0.06) * S
    w1 = w0 + h * spread
    ang = math.radians(angle)
    axd = (math.sin(ang), -math.cos(ang))                             # plume axis (screen)
    prp = (math.cos(ang), math.sin(ang))
    pad = w1 * 1.6 + abs(wind) * h
    ends = [(bx, by), (bx + axd[0] * h + prp[0] * wind * h, by + axd[1] * h + prp[1] * wind * h)]
    xs = [p[0] for p in ends]
    ys = [p[1] for p in ends]
    win = _win(cv, min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    dx, dy = X - bx, Y - by
    t = (dx * axd[0] + dy * axd[1]) / h                               # along the plume 0 → 1
    lat = dx * prp[0] + dy * prp[1]                                   # across
    tp = np.clip(t, 0, None)
    xc = wind * h * tp ** 1.7 + h * sway * (perlin1d_at(tp * 2.5 + ox, seed) + 0.5 * perlin1d_at(tp * 6.0 + oy, seed + 1)) * tp
    wt = w0 * (0.35 + 0.65 * smoothstep(-0.02, 0.12, t)) + (w1 - w0) * tp ** 0.85   # pinched at the source
    s = (lat - xc) / wt
    # along-plume coordinate in units of the local width, so puffs grow as the column widens
    g = max(w1 - w0, 1e-3)
    along = (h / g) * np.log1p(g * tp / max(w0, 1e-3)) * 0.42
    # laminar near the source, turbulent above: the warp ramps in along the column
    turb = curl * smoothstep(0.0, 0.18, t) * (0.6 + 0.6 * tp)
    U, V = s * 0.75, along - 0.0
    ax_, ay_ = advect(U * 0.7 + ox, V * 0.7 + oy, steps=2, step=0.16 * max(curl, 1e-3), seed=seed + 1, octaves=2)
    U = lerp(U, (ax_ - ox) / 0.7, clamp01(turb))
    V = lerp(V, (ay_ - oy) / 0.7, clamp01(turb))
    big = 0.5 + 0.5 * fbm2d_at(U * 1.1 + 3.1 + ox, V * 1.1 + oy, 3, seed=seed + 2)
    det = fbm2d_at(U * 3.0 + 1.7 + ox, V * 3.0 + oy, 4, seed=seed + 3, mode="billow")
    # lumpy, defined outline: the edge distance is pushed in and out by billows
    lump = fbm2d_at(np.sign(s) * 2.0 + V * 0.35 + ox, V * 1.6 + 5.3 + oy, 3, seed=seed + 4)      # bulges along each edge
    edge = 1.0 - np.abs(s) + 0.8 * (big - 0.45) * (0.45 + tp) + 0.3 * (det - 0.45) + 0.55 * lump * smoothstep(0.05, 0.4, tp)
    soft = (0.08 + 0.55 * tp) * softness                                          # sharper puffs low down, wispy up high
    dens = smoothstep(0.0, soft, edge) * (0.3 + 0.7 * smoothstep(0.15, 0.65, big + 0.3 * det))
    dens = dens * smoothstep(-0.03, 0.16, t + 0.06 * (det - 0.45)) * (1 - smoothstep(0.5, 1.0, t))
    thin = (w0 / np.maximum(wt, 1e-3)) ** 0.55                       # the same smoke spread wider is thinner
    al = clamp01(dens) * opacity * thin * (1 - 0.3 * tp)
    al = gaussian_blur(al, max(0.5, 0.6 * S))
    # self-shadowing: lit side towards ``light``, darker dense interior, bright billow tops
    lx, ly = light
    shade_, _ = _shade_from_density(al, (lx, ly), max(1.0, w0 * 0.25 + 2 * S))
    inner = smoothstep(0.2, 0.9, edge)
    sidelit = clamp01(0.5 + 0.5 * np.tanh(s * (lx * prp[0] + ly * prp[1]) * 1.2))
    tone = clamp01(0.15 + 0.55 * shade_ + 0.25 * sidelit + 0.35 * (det - 0.45) - 0.25 * inner * (1 - sidelit))
    col = lerp(shade, color, tone)
    if mask is not None:
        al = al * np.asarray(mask, F32)[win[1]:win[3], win[0]:win[2]]
    _put(cv, col, al, win)
    return FX(_full(cv, al, win))


def _bg_under(cv, x0, y0, x1, y1):
    """Mean colour of the canvas inside a pixel box (what a vapour effect is about to be drawn over)."""
    win = _win(cv, x0, y0, x1, y1)
    if win is None:
        return rgb("#808080")
    patch = np.asarray(cv.buf[win[1]:win[3], win[0]:win[2]], F32)
    return np.clip(patch.reshape(-1, 3).mean(0), 0, 1)


def _vapour_look(bg, color, opacity):
    """Colours for steam / breath over a ground of colour ``bg``: (core, shade, opacity, edge strength, edge colour).
    The core is tinted by the ground (a warm wall gives warm white vapour); on a light ground the shade turns into a
    soft darker edge and the opacity rises so the wisps keep their shape on cream walls and white jackets."""
    lb = float(luma(bg))
    core = rgb(color) if color is not None else lerp(bg, np.array([0.97, 0.97, 0.96], F32), 0.82)
    contrast = float(luma(core)) - lb
    ring = float(np.clip((0.3 - contrast) / 0.3, 0.0, 1.0))
    edge_col = lerp(bg * 0.55, np.array([0.45, 0.52, 0.66], F32) * max(lb, 0.35), 0.35)
    if color is None:                                  # on a light ground vapour is a cool soft grey, not white
        core = lerp(core, lerp(bg, np.array([0.8, 0.85, 0.92], F32), 0.6) * 0.96, ring)
    shade = lerp(lerp(bg, np.array([0.78, 0.8, 0.82], F32), 0.55), edge_col, ring)
    op = float(opacity) if opacity is not None else 0.75 + 0.2 * ring
    return core, shade, op, ring, edge_col


def _vapour_edge(cv, out, S, edge_col, ring):
    """A soft darker rim just outside the wisps (``out`` = their coverage) so pale vapour reads on pale grounds."""
    if ring < 0.02 or out.max() <= 1e-4:
        return
    ys, xs = np.nonzero(out > 1e-3)
    pad = int(8 * S) + 4
    y0, y1 = max(0, ys.min() - pad), min(cv.h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(cv.w, xs.max() + pad + 1)
    on = clamp01(out[y0:y1, x0:x1] * 2.4)
    g = gaussian_blur(on, max(1.0, 1.3 * S))
    rim = clamp01(g * 1.7 - on) * clamp01(0.8 * ring + 0.15)
    full = np.zeros((cv.h, cv.w), F32)
    full[y0:y1, x0:x1] = rim
    cv.paint(edge_col, full)


def steam(cv, at=(0.5, 0.5), size=180, seed=0, wisps=3, color=None, opacity=None, width=None):
    """Steam wisps rising from a cup / pot: thin, sinuous, translucent ribbons that fade quickly. The colour is
    taken from what is behind them (white vapour on dark grounds; on cream walls and white jackets the wisps get a
    soft darker edge and more body), ``color=`` sets the vapour colour and ``opacity=`` its strength."""
    S = _S(cv)
    rng = np.random.default_rng(seed)
    wd = (width or size * 0.35)
    bx, by = _xy(cv, at)
    bg = _bg_under(cv, bx - (wd * 0.8 + 24) * S, by - size * 1.1 * S, bx + (wd * 0.8 + 24) * S, by)
    core, shade, op, ring, edge_col = _vapour_look(bg, color, opacity)
    out = np.zeros((cv.h, cv.w), F32)
    for i in range(int(wisps)):
        off = (i - (wisps - 1) / 2) * wd / max(wisps, 1) * S / cv.w
        fx = smoke(cv, at=(at[0] + off, at[1]), size=size * rng.uniform(0.75, 1.1), width=size * 0.06,
                   spread=0.12, wind=rng.uniform(-0.2, 0.2), seed=seed + 7 * i, color=core, shade=shade,
                   opacity=op, curl=1.1, sway=0.16, softness=2.6)
        if fx.mask is not None:
            out = np.maximum(out, fx.mask)
    _vapour_edge(cv, out, S, edge_col, ring)
    return FX(out)


def chimney_smoke(cv, at=(0.5, 0.4), size=520, wind=0.9, seed=0, color="#8e8f96", opacity=0.6, **kw):
    """Chimney smoke bent over by the wind (``wind`` = drift as a fraction of the column length). Other options
    are those of ``smoke`` (``phase`` / ``offset`` animate it)."""
    kw = _reject("chimney_smoke", kw, ["at", "size", "wind", "seed", "color", "opacity"] + [
        n for n in _sigopts(smoke) if n not in ("cv", "at", "size", "width", "spread", "wind", "seed", "color",
                                               "opacity")])
    return smoke(cv, at=at, size=size, width=size * 0.05, spread=0.3, wind=wind, seed=seed, color=color,
                 opacity=opacity, **kw)


def breath(cv, at=(0.5, 0.5), size=120, direction=70, seed=0, color=None, opacity=None):
    """Visible breath on a cold day: a soft puff leaving the mouth along ``direction`` (degrees, 90 = right). The
    colour follows the ground behind it (see ``steam``); ``color=`` / ``opacity=`` override it."""
    S = _S(cv)
    bx, by = _xy(cv, at)
    r = size * S
    bg = _bg_under(cv, bx - r, by - r, bx + r, by + r)
    core, shade, op, ring, edge_col = _vapour_look(bg, color, opacity)
    fx = smoke(cv, at=at, size=size, width=size * 0.08, spread=0.55, wind=0.05, seed=seed, color=core,
               shade=shade, opacity=op, angle=direction, curl=0.7, softness=3.0)
    if fx.mask is not None:
        _vapour_edge(cv, fx.mask, S, edge_col, ring)
    return fx


def fog(cv, v_range=(0.5, 1.0), color="#c3c9d2", density=0.55, layers=3, seed=0, scale=420, mask=None):
    """Ground fog / mist in depth layers: soft horizontal banks, thicker low down, each layer its own drift."""
    S = _S(cv)
    H, W = cv.h, cv.w
    X, Y = cv.xy()
    m = _mask_or_ones(cv, mask)
    y0, y1 = v_range[0] * H, v_range[1] * H
    total = np.zeros((H, W), F32)
    for i in range(int(layers)):
        f = (i + 1) / layers
        yc = y0 + (y1 - y0) * (0.25 + 0.6 * f)
        band = np.exp(-((Y - yc) / ((y1 - y0) * 0.35)) ** 2) * smoothstep(y0 - 20 * S, y0 + (y1 - y0) * 0.25, Y)
        n = fbm2d(H, W, scale * S * (0.6 + 0.5 * f), 5, seed=seed + 11 * i, aspect=0.18)
        a = clamp01(band * (0.45 + 0.75 * n)) * density / layers * 1.6 * m
        cv.paint(color, a)
        total = np.maximum(total, a)
    return FX(total)


def dust_motes(cv, n=220, mask=None, region=None, color="#fff1d2", size=2.2, seed=0, intensity=1.2):
    """Dust motes floating in light: tiny bright specks plus a few large out-of-focus discs. Pass the light
    beam as ``mask`` so motes show only where light falls."""
    S = _S(cv)
    H, W = cv.h, cv.w
    rng = np.random.default_rng(seed)
    m = _mask_or_ones(cv, mask) * _region_mask(cv, region)
    xs = rng.uniform(0, W, n * 4)
    ys = rng.uniform(0, H, n * 4)
    p = m[np.clip(ys.astype(int), 0, H - 1), np.clip(xs.astype(int), 0, W - 1)]
    keep = rng.uniform(0, 1, n * 4) < p
    xs, ys = xs[keep][:n], ys[keep][:n]
    b = rng.uniform(0.2, 1.0, len(xs)) ** 2
    small = gaussian_blur(_splat(H, W, xs, ys, b * 3.0), size * S * 0.5)
    acc = small * 2.2
    for i in range(min(len(xs), max(3, n // 25))):
        r = rng.uniform(4, 12) * S
        win = _win(cv, xs[i] - r * 1.5, ys[i] - r * 1.5, xs[i] + r * 1.5, ys[i] + r * 1.5)
        if win is None:
            continue
        d = np.sqrt((win[4] - xs[i]) ** 2 + (win[5] - ys[i]) ** 2) / r
        acc[win[1]:win[3], win[0]:win[2]] += (clamp01((1 - d) * 4) * (0.5 + 0.5 * d) * 0.18 * b[i])
    cv.paint(color, acc * intensity * np.maximum(m, 0.0), "add")
    return FX(clamp01(acc))


# =============================================================================
# water
# =============================================================================
def _vblur(img, sigma):
    """Vertical-only Gaussian-like blur (three box passes via cumulative sums), any sigma, fast."""
    r = int(round(sigma * 0.85))
    if r < 1:
        return img
    a = np.asarray(img, np.float64)
    for _ in range(3):
        c = np.cumsum(np.pad(a, [(r + 1, r)] + [(0, 0)] * (a.ndim - 1), mode="edge"), axis=0)
        a = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
    return a.astype(F32)


def _vsmear(img, sigma):
    """Vertical-only blur (stretches reflections of lights into streaks)."""
    if sigma <= 0.5:
        return img
    h, w = img.shape[:2]
    k = max(int(sigma / 2), 1)
    small = resize(img, w, max(h // k, 2), "box")
    return resize(resize(small, w, h, "bilinear"), w, h, "bilinear")


def ocean(cv, horizon=0.5, sun=None, seed=0, deep="#06202e", body="#14506a", foam=0.6, swell=1.0, chop=1.0,
          glitter=1.0, wind=18.0, camera=3.0, mask=None, sss="#2a8a8c", reflect=1.0, crest=0.62):
    """Open sea in perspective: swell lines, choppy small waves, foam on the crests, reflections of the sky
    already on the canvas (paint the sky first) and a sun glitter path when ``sun=(u, v)`` is given.

    ``horizon`` = v fraction of the sea line, ``swell`` / ``chop`` = wave heights, ``foam`` 0~1 = whitecaps,
    ``wind`` = swell direction (degrees from straight ahead), ``camera`` = eye height (bigger = calmer look)."""
    S = _S(cv)
    H, W = cv.h, cv.w
    hy = horizon * H
    win = _win(cv, 0, hy, W, H)
    if win is None:
        return FX(None)
    pre = cv.buf.copy()
    X, Y = win[4], win[5]
    dyp = np.maximum(Y - hy, 0.6)
    f = float(H)
    z = camera * f / dyp
    x = (X - W * 0.5) * camera / dyp
    pz = z / dyp                                                     # metres per pixel (vertical)
    rng = np.random.default_rng(seed)

    def vis(lam):
        return smoothstep(2.5, 7.0, lam / pz)

    wd = math.radians(wind)
    warp = fbm2d_at(x / 40.0, z / 40.0, 3, seed=seed + 1)
    hs = np.zeros_like(X)
    crest_f = np.zeros_like(X)
    env = 0.6 + 0.4 * fbm2d_at(x / 25.0, z / 25.0, 2, seed=seed + 3)          # wave groups
    for lam, A, da in ((19.0, 0.5, 0.0), (12.0, 0.36, 0.55), (8.0, 0.26, -0.6), (5.5, 0.18, 1.0), (3.8, 0.12, -1.1)):
        k = 2 * math.pi / lam
        ph = k * (x * math.sin(wd + da) + z * math.cos(wd + da)) + warp * 3.0 + rng.uniform(0, 6.28)
        c = (np.sin(ph) + 1) * 0.5
        wave = c ** 2.5 * 2 - 0.6                                   # sharp crests, flat troughs
        v = vis(lam)
        hs = hs + A * swell * wave * v * env
        crest_f = np.maximum(crest_f, c ** 14 * v * min(A / 0.36, 1.0) * env)
    hc = np.zeros_like(X)
    lam, A = 2.6, 0.15
    for o in range(9):
        hc = hc + A * chop * vis(lam) * perlin2d_at(x / lam + 3.1 * o, z / lam * 0.7, seed + 10 + o)
        lam *= 0.5
        A *= 0.55
    h = hs + hc
    gy, gx = np.gradient(h)
    hx = gx / (camera / dyp)
    hz = gy / (-z / dyp)
    nrm = np.sqrt(hx * hx + 1 + hz * hz)
    nx, ny, nz = -hx / nrm, 1.0 / nrm, -hz / nrm
    vx, vy, vz = x, -camera * np.ones_like(x), z
    vl = np.sqrt(vx * vx + vy * vy + vz * vz)
    vx, vy, vz = vx / vl, vy / vl, vz / vl
    vdn = vx * nx + vy * ny + vz * nz
    rx, ry, rz = vx - 2 * vdn * nx, vy - 2 * vdn * ny, vz - 2 * vdn * nz
    rz = np.maximum(rz, 1e-3)
    sx = W * 0.5 + rx / rz * f
    sy = hy - np.maximum(ry, 0.002) / rz * f
    sy = np.clip(sy, 0, hy - 1.5)
    dep = clamp01((Y - hy) / max(H - hy, 1))
    # near water is rougher at the pixel scale: it reflects a blurred sky (no hard-edged cloud shapes)
    sxc, syc = np.clip(sx, 0, W - 1), sy
    refl = lerp(sample(pre, sxc, syc), sample(_blur(pre, 10 * S), sxc, syc), smoothstep(0.05, 0.6, dep))
    fres = 0.02 + 0.98 * (1 - clamp01(-vdn)) ** 5
    fres = clamp01(fres * reflect)
    water = lerp(rgb(body), rgb(deep), clamp01(0.3 + 0.7 * dep))
    face = clamp01(hs * 1.2 + 0.2) * clamp01(-hz * 2 + 0.3)        # translucent wave faces
    water = water + rgb(sss) * (face * 0.35 * (1 - dep * 0.5))[..., None]
    # rays reflected down or grazing hit the next wave, not the sky: fade them to the water colour instead of
    # smearing the horizon row across the wave backs
    refl = lerp(water * 0.85 + refl * 0.15, refl, smoothstep(-0.01, 0.06, ry))
    col = lerp(water, refl, fres)
    if foam:
        patch = fbm2d_at(x / 5.0, z / 4.0, 4, seed=seed + 21)
        streak = smoothstep(0.5, 0.85, fbm2d_at(x / 5.0 + 0.6 * patch, z / 0.45, 4, seed=seed + 24, mode="ridged")) * vis(0.5)
        cf = crest_f * (0.8 + 0.6 * patch) + hc * 1.0 - (1 - foam) * 0.5
        tex = fbm2d_at(x / 0.7, z / 0.5, 5, seed=seed + 25, mode="billow")
        wc = smoothstep(crest - 0.12, crest + 0.15, cf)
        whitecap = smoothstep(0.35, 0.65, wc * (0.35 + 1.1 * tex)) * (0.75 + 0.25 * tex)
        trail = smoothstep(crest - 0.35, crest - 0.05, cf) * streak * smoothstep(-0.2, 0.3, patch)
        # lacy texture inside the foam near the camera (faded out where it would alias)
        lace = smoothstep(0.25, 0.75, fbm2d_at(x / 0.22 + 0.4 * tex, z / 0.17, 3, seed=seed + 26, mode="ridged"))
        lv = vis(0.22)
        whitecap = whitecap * (1 - lv + lv * (0.35 + 0.8 * lace))
        fm = clamp01(whitecap + trail * 0.3) * foam * vis(1.0)
        sky_l = np.mean(pre[: max(int(hy), 1)], axis=(0, 1))
        foamc = (0.55 + 0.45 * clamp01(ny * 1.4 - 0.2))[..., None] * (rgb("#eef3f5") * 0.6 + sky_l * 0.5)
        col = lerp(col, foamc, fm)
    if sun is not None and glitter:
        su, sv = _xy(cv, sun)
        s = np.array([su - W * 0.5, hy - sv, f], np.float64)
        s = s / np.linalg.norm(s)
        rl = np.sqrt(rx * rx + ry * ry + rz * rz)
        cosr = (rx * s[0] + ry * s[1] + rz * s[2]) / rl
        spec = clamp01((cosr - 0.992) / 0.008) ** 4 * 5.0 + clamp01((cosr - 0.96) / 0.04) ** 3 * 0.18
        sunc = sample(pre, np.array([np.clip(su, 0, W - 1)], F32), np.array([np.clip(sv, 0, hy - 1)], F32))[0]
        sunc = sunc / max(float(sunc.max()), 1e-3)
        col = col + sunc * (spec * glitter)[..., None]
    a = clamp01(Y - hy + 0.5)
    if mask is not None:
        a = a * np.asarray(mask, F32)[win[1]:win[3], win[0]:win[2]]
    _put(cv, col, a, win)
    return FX(_full(cv, a, win), height=h)


def _upsample(a, shape):
    """Bilinear upsample of a half-resolution field back to ``shape`` (h, w)."""
    return resize(a, shape[1], shape[0], "bilinear")


def _fbm_half(X, Y, *args, **kw):
    """fbm2d_at evaluated on every second pixel and upsampled: 4x faster for smooth fields on big windows."""
    X = np.asarray(X, F32)
    if X.ndim != 2 or min(X.shape) < 16:
        return fbm2d_at(X, Y, *args, **kw)
    Y = np.asarray(Y, F32)
    return _upsample(fbm2d_at(X[::2, ::2], Y[::2, ::2], *args, **kw), X.shape)


def _spline(pts, n=16):
    """Catmull-Rom curve through ``pts`` → list of (x, y) (n samples per segment)."""
    P = np.asarray(pts, np.float64)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (3 * p1 - p0 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.asarray(out)


def breaking_wave(cv, at=(0.5, 0.62), size=420, seed=0, water="#0b4256", light="#4fd0c8", foam="#f2f7f7",
                  direction=1, spray=True):
    """A plunging breaker seen side-on: a long back slope rising to the crest, a concave translucent face lit
    from behind, a curling lip thrown forward over a shadowed tube, a falling curtain where the lip drops,
    white water exploding where it lands, foam streaks and spray blowing off the crest.
    ``at`` = foot of the face, ``size`` = wave height (1080p px), ``direction`` = 1 (breaks right) / -1."""
    S = _S(cv)
    bx, by = _xy(cv, at)
    h = size * S
    d = 1 if direction >= 0 else -1
    x0w, x1w = (bx - h * 2.8, bx + h * 1.9) if d > 0 else (bx - h * 1.9, bx + h * 2.8)
    win = _win(cv, x0w, by - h * 1.6, x1w, by + h * 0.4)
    if win is None:
        return FX(None)
    ox, oy = win[0], win[1]
    hh, ww = win[3] - win[1], win[2] - win[0]
    rng = np.random.default_rng(seed)

    def px(uv):
        uv = np.asarray(uv, np.float64)
        return [(bx + u * h * d - ox, by - v * h - oy) for u, v in uv]

    def poly(points):
        pts = px(points)
        return shape_mask(hh, ww, lambda dr, s: dr.polygon([(x * s, y * s) for x, y in pts], fill=255))

    # --- profile curves in wave units (u forward, v up; crest near (0.3, 1.17), foot of the face at (0.8, 0))
    def vback(q):
        q = np.asarray(q, np.float64)
        return 1.04 / (1 + np.exp(-(q + 0.85) * 2.5)) + 0.035 * np.sin(q * 2.1 + ph0) * smoothstep(-0.6, -1.4, q)
    ph0 = rng.uniform(0, 6)
    ub = np.linspace(-2.8, -0.38, 60)
    vb = vback(ub)
    back = np.stack([ub, vb], 1)
    outer = _spline([(-0.38, float(vback(-0.38))), (-0.08, 1.06), (0.28, 1.17), (0.56, 1.1), (0.76, 0.88),
                     (0.88, 0.62), (0.94, 0.36)], 16)
    tng = np.gradient(outer, axis=0)
    tng /= np.linalg.norm(tng, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([tng[:, 1], -tng[:, 0]], 1)                           # points into the curl (inside)
    sl = np.linspace(0, 1, len(outer))
    th = 0.3 * (1 - sl) ** 1.6 + 0.035
    inner = outer + nrm * th[:, None]
    tip = outer[-1]
    land = np.array([tip[0] + 0.05, -0.02])
    k0 = int(np.argmin(np.abs(inner[:, 0] - 0.04) + 5 * (inner[:, 1] < 0.6)))
    inner_t = inner[k0:]
    face = _spline([tuple(inner_t[0]), (0.11, 0.66), (0.15, 0.46), (0.24, 0.25), (0.45, 0.08), (0.8, 0.0)], 14)
    sil = poly(np.vstack([back, outer, [land], [(1.9, -0.02), (1.9, -0.5), (-2.8, -0.5)]]))
    tube = poly(np.vstack([face, [land], [tip], inner_t[::-1]]))
    lip = poly(np.vstack([outer, inner[::-1]]))
    curtain = poly([(tip[0] - 0.035, tip[1] + 0.02), (tip[0] + 0.03, tip[1] + 0.02), (land[0] + 0.05, land[1]),
                    (land[0] - 0.045, land[1])])
    tube = tube * (1 - lip)
    body = sil
    c = outer
    ub_all = np.concatenate([ub, outer[1:, 0]])
    vb_all = np.concatenate([vb, outer[1:, 1]])
    X, Y = win[4], win[5]
    u = (X - bx) / h * d
    v = (by - Y) / h
    surf = np.interp(u, ub_all, vb_all)
    below = clamp01((surf - v) / 0.7)                                    # depth under the surface line
    # --- body: deep water low down, turquoise light shining through the thin top of the face
    trans = smoothstep(0.25, 0.95, v) * smoothstep(-1.1, 0.05, u) * (1 - 0.7 * below)
    col = lerp(rgb(water), rgb(light) * 1.15, clamp01(trans))
    col = col * (1 - 0.4 * below * smoothstep(0.2, -1.6, u))[..., None]
    draw_up = smoothstep(0.55, 0.92, _fbm_half(u * 16 + 3, v * 1.6, 4, seed=seed + 6, mode="ridged"))
    on_face = smoothstep(-0.12, 0.05, u) * smoothstep(0.02, 0.25, v)
    col = lerp(col, rgb(light) * 1.35, draw_up * on_face * 0.3)
    chop = smoothstep(0.55, 0.9, _fbm_half(u * 2.2 + 0.6 * v, (v - surf) * 16 + u * 0.7, 4, seed=seed + 2, mode="ridged"))
    col = lerp(col, rgb(foam) * 0.85, chop * 0.28 * smoothstep(0.05, 0.5, v) * smoothstep(0.1, -0.15, u))
    sky_hi = np.exp(-((surf - v) / 0.04) ** 2) * smoothstep(0.1, -0.4, u)
    col = col + rgb(light) * (sky_hi * 0.25)[..., None]
    # --- tube: shadowed, lighter towards the open end, streaked along the curl
    cx_, cy_ = 0.5, 0.55
    rad = np.hypot(u - cx_, (v - cy_) * 1.1)
    ang = np.arctan2(v - cy_, u - cx_)
    # streaks run around the curl (concentric about the tube axis); the back wall is darkest, the water near
    # the lip glows where light comes through it
    swirl = 0.8 + 0.2 * _fbm_half(ang * 2.2 + 1.3, rad * 22.0, 3, seed=seed + 7) * smoothstep(0.05, 0.25, rad)
    curl_lines = smoothstep(0.55, 0.9, _fbm_half(ang * 1.6 + 4.0, rad * 55.0, 3, seed=seed + 14, mode="ridged")) * smoothstep(0.12, 0.3, rad)
    open_ = clamp01((u - 0.1) / 0.8)
    deep_ = smoothstep(0.42, 0.12, rad) * smoothstep(0.75, 0.25, u)
    tube_c = lerp(rgb(water) * 0.9, rgb(light) * 1.05, clamp01(0.15 + 0.35 * open_ + 0.55 * smoothstep(0.18, 0.44, rad)))
    tube_c = tube_c * (1 - 0.45 * deep_)[..., None] * swirl[..., None] + rgb(light) * (curl_lines * 0.18)[..., None]
    col = lerp(col, tube_c, tube)
    # --- lip: thin translucent water, foam boiling on its leading edge
    bil = _fbm_half(u * 6, v * 6, 5, seed=seed + 3, mode="billow")
    sl_px = np.interp(u, c[:, 0], sl)
    lip_c = lerp(rgb(light) * 1.3, rgb(light) * 0.8, smoothstep(0.4, 1.0, sl_px))
    lip_foam = clamp01(smoothstep(0.35, 0.9, sl_px) * (0.25 + bil) + smoothstep(1.02, 1.14, v) * bil * 0.8)
    lip_c = lerp(lip_c, rgb(foam), clamp01(lip_foam) * 0.9)
    flow = smoothstep(0.5, 0.9, _fbm_half(ang * 1.4 + 2.0, rad * 26, 4, seed=seed + 11, mode="ridged"))
    lip_c = lip_c * (0.88 + 0.12 * _fbm_half(ang * 3, rad * 10, 3, seed=seed + 12))[..., None] + rgb(foam) * (flow * 0.22)[..., None]
    lipw = clamp01(gaussian_blur(lip, 0.025 * h) * 1.4) * smoothstep(0.0, 0.25, sl_px)
    col = lerp(col, lip_c, lipw)
    a = clamp01(body)
    _put(cv, col, a, win)
    # --- falling curtain under the lip tip
    ripple_ = 0.6 + 0.4 * _fbm_half(u * 40, v * 3, 3, seed=seed + 9)
    cstreak = smoothstep(0.3, 0.8, 0.5 + 0.5 * _fbm_half(u * 60, v * 2.5, 3, seed=seed + 9))
    _put(cv, lerp(rgb(light), rgb(foam), 0.6) * ripple_[..., None], gaussian_blur(curtain, 1.0 * S) * cstreak * 0.6, win)
    # --- white water where the lip lands, rolling forward and up
    lx = land[0]
    ww_ = np.exp(-((u - lx - 0.12) / 0.42) ** 2) * np.exp(-((v - 0.12) / 0.32) ** 2) * 1.35
    ww_ = ww_ + smoothstep(-0.4, 0.3, u) * smoothstep(0.14, -0.06, v) * smoothstep(1.9, 1.2, u) * 0.9
    # churning white water: torn, streaky foam with blue shadow in its folds (not cotton-wool puffs)
    wu, wv = advect(u[::2, ::2] * 3.0, v[::2, ::2] * 3.0, steps=2, step=0.08, seed=seed + 15, octaves=2)
    bil2 = _upsample(fbm2d_at(wu * 2.2, wv * 2.6, 5, seed=seed + 10, mode="billow"), u.shape)
    bil3 = _fbm_half(u * 16 - v * 5, v * 20, 3, seed=seed + 13, mode="billow")
    wwm = smoothstep(0.38, 0.8, ww_ * (0.3 + 0.75 * bil2 + 0.2 * bil3))
    shade, _ = _shade_from_density(gaussian_blur(wwm, 2 * S), (-0.4 * d, -1.0), 7 * S)
    wcol = rgb(foam) * (0.8 + 0.25 * shade + 0.08 * (bil3 - 0.5))[..., None]
    wcol = lerp(wcol, rgb(light) * 0.75, clamp01(smoothstep(0.75, 0.15, wwm) * 0.45 + (1 - shade) * 0.25))
    _put(cv, wcol, wwm, win)
    if spray:
        mist = smoothstep(0.25, 0.8, _fbm_half(u * 3, v * 3, 4, seed=seed + 4, mode="billow"))
        land_mist = mist * np.exp(-((u - lx - 0.1) / 0.45) ** 2 - ((v - 0.5) / 0.32) ** 2)
        crest_mist = mist * np.exp(-((u + 0.15) / 0.35) ** 2 - ((v - 1.16) / 0.08) ** 2) * smoothstep(0.98, 1.08, v)
        _put(cv, rgb(foam), clamp01(land_mist * 0.5 + crest_mist * 0.6), win)
        sparks(cv, at=((bx + (lx + 0.1) * h * d) / cv.w, (by - 0.35 * h) / cv.h), n=70, spread=size * 0.8,
               direction=-90 + 35 * d, angle=110, length=5, size=1.5, color=(0.9, 0.95, 1.0),
               intensity=1.0, gravity=0.8, seed=seed + 5)
    return FX(_full(cv, clamp01(a + wwm), win))


def lake(cv, horizon=0.6, color="#0a1622", ripple=1.0, seed=0, reflect=0.92, streak=1.0, mask=None, tint="#dfe8f0"):
    """Calm lake: mirrors the current canvas about ``horizon`` (paint sky, moon, shore first) with horizontal
    ripple distortion, vertical streaks under lights and Fresnel darkening towards the viewer."""
    S = _S(cv)
    H, W = cv.h, cv.w
    hy = horizon * H
    win = _win(cv, 0, hy, W, H)
    if win is None:
        return FX(None)
    pre = cv.buf
    X, Y = win[4], win[5]
    h_, w_ = X.shape
    T = clamp01((Y - hy) / max(H - hy, 1))
    rip = fbm2d(h_, w_, 70 * S, 4, seed=seed, aspect=0.1)
    rip2 = fbm2d(h_, w_, 160 * S, 3, seed=seed + 1, aspect=0.15)
    dx = (rip * (2 + 10 * T) + rip2 * 8 * T) * S * ripple
    dy = np.abs(rip) * (1 + 10 * T) * S * ripple
    src_y = 2 * hy - Y + dy
    refl = sample(pre, X + dx, np.clip(src_y, 0, H - 1))
    if streak:
        sm = []
        for k in (-2, -1, 1, 2):
            sm.append(sample(pre, X + dx, np.clip(src_y + k * (2 + 6 * T) * S * streak, 0, H - 1)))
        refl = (refl * 2 + sum(sm)) / 6
    # bright reflections (moon, lamps, sun) break into a column of horizontal glints: sample them with a
    # strong vertical jitter so the disc stretches into a glitter path instead of a mirrored ball
    if streak:
        bright = smoothstep(0.5, 0.95, luma(refl))
        hl = refl * bright[..., None]
        col_ = _vblur(hl, 30 * S * streak)
        strips = smoothstep(0.05, 0.5, fbm2d(h_, w_, 26 * S, 3, seed=seed + 6, aspect=0.07))
        spread_ = col_ * (strips * 2.4 * (0.6 + 0.6 * T))[..., None]
        refl = np.maximum(refl * (1 - 0.55 * bright * (1 - strips))[..., None], np.minimum(spread_, refl.max() * 1.0))
    fres = reflect * (1 - 0.5 * T ** 1.3)
    col = lerp(np.broadcast_to(rgb(color), refl.shape), refl * rgb(tint), fres[..., None] if fres.ndim == 2 else fres)
    # thin horizontal glints: ridges of 1-D noise down the water, broken into dashes by a slow isotropic field
    # (a stretched 2-D ridged noise left a vertical stripe at every lattice column)
    lines = smoothstep(0.8, 0.97, 1.0 - np.abs(fbm1d(h_, 1.2 * S, 3, seed=seed + 2))[:, None] * 3.0)         * smoothstep(0.0, 0.3, fbm2d(h_, w_, 70 * S, 2, seed=seed + 3, aspect=0.4))
    col = col + (luma(refl) * lines * 0.12 * (0.3 + T))[..., None]
    a = clamp01(Y - hy + 0.5)
    if mask is not None:
        a = a * np.asarray(mask, F32)[win[1]:win[3], win[0]:win[2]]
    _put(cv, col, a, win)
    return FX(_full(cv, a, win))


def _source_mask(cv, of):
    """Coverage (H, W) of what ``reflect`` mirrors: an FX, a mask, a box (u0, v0, u1, v1) or a list of those."""
    if isinstance(of, (list, tuple)) and of and not (len(of) == 4 and all(np.isscalar(v) for v in of)):
        out = np.zeros((cv.h, cv.w), F32)
        for o in of:
            out = np.maximum(out, _source_mask(cv, o))
        return out
    if isinstance(of, FX):
        if of.mask is None:
            raise ValueError("reflect: the effect passed as of= has no mask to mirror")
        return np.asarray(of.mask, F32)
    if isinstance(of, (list, tuple)):
        u0, v0, u1, v1 = [float(v) for v in of]
        _check_frac("reflect", "of", (u0, v0, u1, v1))
        m = np.zeros((cv.h, cv.w), F32)
        m[max(0, int(v0 * cv.h)):int(v1 * cv.h) + 1, max(0, int(u0 * cv.w)):int(u1 * cv.w) + 1] = 1.0
        return m
    m = np.asarray(of, F32)
    if m.shape != (cv.h, cv.w):
        raise ValueError("reflect: of= is a mask of shape %s but the canvas is %s; pass a full-canvas mask, a box "
                         "(u0, v0, u1, v1) or the FX an effect returned" % (m.shape, (cv.h, cv.w)))
    return m


def reflect(cv, of, water=None, axis=None, strength=0.85, ripple=1.0, streak=1.0, reach=1.0, glow=1.0, seed=0,
            tint="#ffffff"):
    """Mirror things painted after ``lake`` into the water: a campfire, a tent, a lantern, a boat.

    ``of``: what to reflect, the FX a call returned (``elements.campfire`` ...), a full-canvas mask, a box
    ``(u0, v0, u1, v1)`` or a list of those. ``water``: the water area (the FX / mask ``lake`` returned; default
    everything below ``axis``). ``axis``: v fraction of the waterline the things stand on (default the lowest
    point of ``of``). Bright parts break into a glitter path, the ripples distort the picture, and the reflection
    fades with distance (``reach`` makes it longer, ``glow`` scales the halo of lights, ``strength`` the opacity)."""
    S = _S(cv)
    H, W = cv.h, cv.w
    src = _source_mask(cv, of)
    ys, xs = np.nonzero(src > 0.02)
    if len(ys) == 0:
        return FX(None)
    top, bot = float(ys.min()), float(ys.max()) + 1.0
    ay = float(axis) * H if axis is not None else bot
    if axis is not None:
        _check_frac("reflect", "axis", (axis,))
    height = max(ay - top, 8 * S)
    pad = 26 * S * glow
    x0, x1 = float(xs.min()) - pad * 1.6, float(xs.max()) + pad * 1.6
    win = _win(cv, x0, ay, x1, ay + height * 1.15 * reach + pad)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    h_, w_ = X.shape
    T = clamp01((Y - ay) / max(height * reach, 1.0))
    rip = fbm2d(h_, w_, 60 * S, 4, seed=seed, aspect=0.1)
    rip2 = fbm2d(h_, w_, 140 * S, 3, seed=seed + 1, aspect=0.15)
    dx = (rip * (2 + 9 * T) + rip2 * 7 * T) * S * ripple
    dy = np.abs(rip) * (1 + 8 * T) * S * ripple
    sy = np.clip(2 * ay - Y + dy, 0, H - 1)
    sx = np.clip(X + dx, 0, W - 1)
    halo = gaussian_blur(src, max(1.0, 9 * S * glow))
    cover = clamp01(np.maximum(src, halo * 0.9 * min(glow, 1.5)))
    a_src = sample(cover, sx, sy)
    col = sample(cv.buf, sx, sy)
    lum = luma(col)
    if streak:
        bright = smoothstep(0.45, 0.95, lum) * a_src
        strips = smoothstep(0.05, 0.5, fbm2d(h_, w_, 24 * S, 3, seed=seed + 6, aspect=0.07))
        glint = _vblur(col * bright[..., None], 22 * S * streak) * (strips * 2.2 * (0.6 + 0.6 * T))[..., None]
        col = np.maximum(col * (1 - 0.5 * bright * (1 - strips))[..., None], np.minimum(glint, col.max()))
        a_src = np.maximum(a_src, clamp01(luma(glint) * 1.4) * 0.8)
    fade = 1 - 0.85 * smoothstep(0.0, 1.0, T)
    a = clamp01(a_src * fade * strength) * clamp01(Y - ay + 0.5)
    if water is not None:
        wm = water.mask if isinstance(water, FX) else water
        if wm is not None:
            a = a * np.asarray(wm, F32)[win[1]:win[3], win[0]:win[2]]
    col = col * rgb(tint)
    _put(cv, col, a, win)
    return FX(_full(cv, a, win))


def wet_ground(cv, mask, axis=None, strength=0.55, darken=0.55, smear=10, seed=0):
    """Wet-ground reflection helper: darkens the surface under ``mask`` and adds a vertically smeared mirror of
    what is above ``axis`` (v fraction, default: the top of the mask) — rainy streets, wet sand, wet stone."""
    S = _S(cv)
    H, W = cv.h, cv.w
    m = np.asarray(mask, F32)
    if axis is None:
        rows = np.where(m.max(1) > 0.5)[0]
        ay = float(rows[0]) if len(rows) else H * 0.6
    else:
        ay = axis * H
    X, Y = cv.xy()
    dist = clamp01((Y - ay) / (H * 0.5))
    jit = fbm2d(H, W, 30 * S, 3, seed=seed, aspect=0.4) * (3 + 10 * dist) * S
    refl = sample(cv.buf, X + jit, np.clip(2 * ay - Y, 0, H - 1))
    # a rough wet surface stretches lights into long vertical streaks; the streaks grow with distance
    refl = 0.5 * _vblur(refl, smear * S * 1.5) + 0.9 * _vblur(refl, smear * S * 5)
    # wetness is patchy: puddled areas mirror more, damp areas less
    patch = smoothstep(-0.25, 0.35, fbm2d(H, W, 140 * S, 4, seed=seed + 1, aspect=0.35))
    grain = 0.85 + 0.15 * fbm2d(H, W, 4 * S, 2, seed=seed + 2)
    fres = strength * (1 - 0.6 * dist) * (0.6 + 0.4 * patch) * grain
    col = cv.buf * darken + refl * fres[..., None]
    cv.paint(col, m)
    return FX(m)


def ripples(cv, at=(0.5, 0.7), size=160, rings=5, squash=0.32, strength=1.0, seed=0, highlight="#dfeaf2"):
    """Ripple rings spreading from a point on water (``size`` = outer radius, ``squash`` = perspective)."""
    S = _S(cv)
    cx, cy = _xy(cv, at)
    R = size * S
    win = _win(cv, cx - R * 1.1, cy - R * squash * 1.1, cx + R * 1.1, cy + R * squash * 1.1)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    dx, dy = X - cx, (Y - cy) / squash
    rr = np.sqrt(dx * dx + dy * dy) / R
    env = smoothstep(1.0, 0.75, rr) * smoothstep(0.0, 0.08, rr)
    ph = rr * rings * 2 * math.pi
    wave = np.sin(ph) * env * (1.1 - rr)
    nxr, nyr = dx / (rr * R + 1e-3), dy / (rr * R + 1e-3)
    amp = 3.0 * S * strength
    src = sample(cv.buf, X + nxr * wave * amp, Y + nyr * wave * amp * squash)
    lit = clamp01(np.cos(ph) * -nyr) * env * (1.1 - rr)
    col = src + rgb(highlight) * (lit * 0.25 * strength)[..., None] - (clamp01(np.cos(ph) * nyr) * env * 0.08 * strength)[..., None]
    a = smoothstep(1.02, 0.9, rr)
    _put(cv, col, a, win)
    return FX(_full(cv, a, win))


def puddle(cv, at=(0.5, 0.8), size=260, seed=0, squash=0.28, horizon=None, tint="#9fb0c4", rim="#2a2620",
           ripple=0):
    """Puddle on the ground: an irregular pool that mirrors what is above (about ``horizon`` v fraction,
    default just above the puddle), with a dark wet rim and an optional ripple ring."""
    S = _S(cv)
    cx, cy = _xy(cv, at)
    R = size * S * 0.5
    win = _win(cv, cx - R * 1.4, cy - R * squash * 1.6, cx + R * 1.4, cy + R * squash * 1.6)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    dx, dy = (X - cx) / R, (Y - cy) / (R * squash)
    ang = np.arctan2(dy, dx)
    wob = 1 + 0.18 * fbm2d_at(np.cos(ang) * 1.3 + 5, np.sin(ang) * 1.3, 3, seed=seed)
    rr = np.sqrt(dx * dx + dy * dy) / wob
    a = clamp01((1 - rr) * R * squash * 0.6 + 0.5)
    wet = clamp01((1.25 - rr) * 4) * (1 - a)
    ay = (horizon * cv.h) if horizon is not None else (cy - R * squash * 1.3)
    refl = sample(cv.buf, X + 1.5 * S * fbm2d_at(X / (20 * S), Y / (6 * S), 2, seed=seed + 1), np.clip(2 * ay - Y, 0, cv.h - 1))
    refl = refl * rgb(tint) * 0.9
    edge = smoothstep(0.75, 1.0, rr) * a
    col = lerp(refl, refl * 0.5, edge * 0.6)
    _put(cv, cv.buf[win[1]:win[3], win[0]:win[2]] * 0.6, wet * 0.8, win)
    _put(cv, col, a, win)
    if ripple:
        ripples(cv, at=at, size=size * 0.3, squash=squash, rings=3, strength=ripple)
    return FX(_full(cv, a, win))


def river_flow(cv, mask=None, angle=90.0, seed=0, strength=0.5, scale=50, color="#d8e6ee", region=None):
    """Flow lines on a river surface: long streaks and eddies combed along ``angle`` (degrees, 90 = flowing
    down the picture / towards the viewer, 0 = left → right). Paint over water you already drew."""
    S = _S(cv)
    m = _mask_or_ones(cv, mask) * _region_mask(cv, region)
    rows = np.where(m.max(1) > 0.01)[0]
    cols = np.where(m.max(0) > 0.01)[0]
    if not len(rows):
        return FX(m)
    win = _win(cv, cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)
    X, Y = win[4], win[5]
    a = math.radians(angle)
    U = (X * math.cos(a) + Y * math.sin(a)) / (scale * S * 8)
    V = (-X * math.sin(a) + Y * math.cos(a)) / (scale * S)
    wx = fbm2d_at(U * 0.5, V * 0.3, 3, seed=seed + 1)
    st = fbm2d_at(U, V + wx * 1.5, 4, seed=seed, mode="ridged")
    lines = smoothstep(0.62, 0.92, st)
    dark = smoothstep(0.35, 0.1, st) * 0.5
    ml = m[win[1]:win[3], win[0]:win[2]]
    base = cv.buf[win[1]:win[3], win[0]:win[2]]
    col = base * (1 - dark * strength * 0.5)[..., None] + rgb(color) * (lines * strength * 0.5)[..., None]
    _put(cv, col, ml, win)
    return FX(m)


def waterfall(cv, box=(0.4, 0.2, 0.6, 0.8), seed=0, color="#e8f3f7", shadow="#5c8796", mist=True, lip="#3f7f8c",
              mist_height=0.35, fall=True):
    """Waterfall in ``box`` (u0, v0, u1, v1): a glassy brink curving over the top edge, falling ropes of water
    that aerate and turn white as they drop, ragged edges with gaps showing the rock behind, white water at the
    foot and a cloud of mist rising ``mist_height`` (fraction of the fall) above the pool. ``fall=False`` paints
    only the foam and mist: call it again after :func:`lake` so the white water sits on top of the pool."""
    S = _S(cv)
    x0, y0 = _xy(cv, box[:2])
    x1, y1 = _xy(cv, box[2:])
    wdt, hgt = x1 - x0, y1 - y0
    win = _win(cv, x0 - wdt * 0.8, y0 - 4 * S, x1 + wdt * 0.8, y1 + hgt * 0.18)
    cloud = None
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    u = (X - x0) / wdt
    v = (Y - y0) / hgt
    vc = np.clip(v, 0, 1)
    spread = 0.09 * vc ** 1.3                                          # the fall widens as it drops
    en_l = fbm2d_at(np.full_like(X, 1.0), Y / (35 * S), 3, seed=seed + 1) * (0.015 + 0.06 * vc)
    en_r = fbm2d_at(np.full_like(X, 7.0), Y / (35 * S), 3, seed=seed + 2) * (0.015 + 0.06 * vc)
    pinch = 0.03 * smoothstep(0.1, -0.02, v)                           # rounded shoulders at the brink
    ul = -spread + en_l + pinch
    ur = 1 + spread - en_r - pinch
    soft = (0.004 + 0.03 * vc)
    inside = smoothstep(ul - soft, ul + soft, u) * smoothstep(ur + soft, ur - soft, u)
    brink = 0.022 * np.clip(1 - (2 * u - 1) ** 2, 0, 1)                # top edge bulges over the rock lip
    inside = inside * smoothstep(-brink - 0.004, -brink + 0.006, v) * smoothstep(1.03, 0.99, v)
    # ropes: column-wise density, more gaps near the edges and lower down where the sheet breaks up
    rope = 0.5 + 0.5 * fbm2d_at(X / (9 * S), np.full_like(X, 0.0) + vc * 1.5, 4, seed=seed + 3)
    edge_d = np.minimum(u - ul, ur - u) / 0.25
    gaps = smoothstep(0.3, 0.6, rope + 0.3 * clamp01(edge_d) - 0.3 * vc * (1 - clamp01(edge_d)))
    # falling clumps: streaks stretched vertically, getting shorter and whiter as the water aerates
    st = fbm2d_at(X / (6 * S), Y / ((260 - 180 * vc) * S), 5, seed=seed)
    st2 = fbm2d_at(X / (2.5 * S), Y / (90 * S), 3, seed=seed + 4)
    clump = fbm2d_at(X / (22 * S), Y / (30 * S) - 0.0, 4, seed=seed + 5, mode="billow")
    aer = smoothstep(0.05, 0.75, vc)
    bright = clamp01(0.35 + 0.85 * st + 0.35 * st2 + 0.6 * aer * clump + 0.25 * aer)
    col = lerp(rgb(shadow), rgb(color), bright)
    glassy = smoothstep(0.12, 0.0, v)                                  # smooth water on the brink
    gl = lerp(rgb(lip), rgb(color), clamp01(0.25 + 0.6 * smoothstep(0.02, 0.1, v) + 0.25 * st2))
    col = lerp(col, gl, glassy * 0.85)
    col = col + rgb(color) * (np.exp(-((v + brink * 0.5 - 0.012) / 0.012) ** 2) * 0.45)[..., None]
    a = inside * clamp01(0.3 + 0.7 * gaps + 0.5 * glassy + 0.3 * aer * clump) * clamp01(0.7 + 0.5 * st + 0.25 * aer)
    if fall:
        _put(cv, col, a, win)
        # spray peeling off the edges lower down
        side = np.exp(-((np.minimum(np.abs(u - ul), np.abs(u - ur))) / (0.04 + 0.1 * vc)) ** 2) * (1 - inside) * aer
        _put(cv, rgb(color), side * smoothstep(0.4, 0.8, clump) * 0.45, win)
    else:
        a = a * 0
    if mist:
        bil = fbm2d_at(X / (45 * S), Y / (38 * S), 5, seed=seed + 6, mode="billow")
        bil2 = fbm2d_at(X / (16 * S), Y / (14 * S), 3, seed=seed + 7, mode="billow")
        foot = np.exp(-((v - 1.0) / 0.07) ** 2) * np.exp(-((u - 0.5) / 0.75) ** 4)
        foamb = smoothstep(0.3, 0.75, foot * (0.5 + 0.6 * bil + 0.35 * bil2))
        sh, _ = _shade_from_density(foamb, (0.0, -1.0), 3 * S)
        _put(cv, rgb(color) * (0.82 + 0.25 * sh)[..., None], foamb, win)
        mh = max(mist_height, 0.05)
        soft_n = 0.5 + 0.5 * fbm2d_at(X / (60 * S), Y / (45 * S), 4, seed=seed + 8)
        cloud = np.exp(-((u - 0.5) / 0.95) ** 4) * smoothstep(1.0 - mh, 1.0, v) * smoothstep(1.22, 1.0, v)
        cloud = clamp01(cloud * (0.2 + 1.1 * soft_n) - 0.1) * 0.7
        _put(cv, rgb(color) * 0.97, cloud, win)
        a = np.maximum(a, np.maximum(foamb, cloud))
    spray = _full(cv, clamp01(cloud / max(float(cloud.max()), 1e-6)), win) if cloud is not None else None
    return FX(_full(cv, a, win), spray=spray)


def splash(cv, at=(0.5, 0.7), size=140, seed=0, color="#d4ecf6", highlight=(1.4, 1.5, 1.6), squash=0.28):
    """Water splash (crown): a thin translucent wall rising from the impact ring, flaring outwards and breaking
    into fingers with beads at their tips, a scatter of flying drops and a disturbed ring on the surface.
    ``at`` = impact point on the water, ``size`` = crown radius (1080p px), ``squash`` = perspective."""
    S = _S(cv)
    rng = np.random.default_rng(seed)
    cx, cy = _xy(cv, at)
    R = size * S
    win = _win(cv, cx - R * 1.9, cy - R * 1.9, cx + R * 1.9, cy + R * 0.6)
    if win is None:
        return FX(None)
    ox, oy = win[0], win[1]
    hh, ww = win[3] - win[1], win[2] - win[0]
    n = 180
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rim_h = R * (0.42 + 0.12 * np.sin(th * 3 + rng.uniform(0, 6)) + 0.08 * np.sin(th * 7 + rng.uniform(0, 6)))
    k = int(rng.integers(11, 16))
    peaks = np.sort((np.arange(k) + rng.uniform(-0.25, 0.25, k)) / k * 2 * np.pi)
    scal = np.zeros(n)
    for pk in peaks:                                                   # scalloped top edge: points at the fingers
        dd = np.angle(np.exp(1j * (th - pk)))
        scal = np.maximum(scal, np.exp(-(dd / (0.5 * 2 * np.pi / k)) ** 2))
    top = rim_h * (0.7 + 0.3 * scal)

    def wall_pt(t, i, frac):
        """Point on the wall at angle index i, height fraction frac (flares out as it rises)."""
        r = R * (0.42 + 0.22 * frac ** 1.5)
        return (cx + r * math.cos(t) - ox, cy + r * squash * math.sin(t) - top[i] * frac - oy)

    def wall(front):
        def draw(d, s_):
            for i in range(n):
                j = (i + 1) % n
                t0, t1 = th[i], th[j] if j else 2 * np.pi
                if (math.sin((t0 + t1) / 2) > 0) != front:
                    continue
                q = [wall_pt(t0, i, 0), wall_pt(t1, j, 0), wall_pt(t1, j, 1), wall_pt(t0, i, 1)]
                d.polygon([(x * s_, y * s_) for x, y in q], fill=255)
        return shape_mask(hh, ww, draw)
    back_w, front_w = wall(False), wall(True)
    tips = []

    def fingers(d, s_):
        for pk in peaks:
            i = int(round(pk / (2 * np.pi) * n)) % n
            x0, y0 = wall_pt(pk, i, 1)
            L = R * rng.uniform(0.25, 0.55)
            out = math.cos(pk)
            pts = []
            for t in np.linspace(0, 1, 8):
                pts.append((x0 + out * L * 0.45 * t ** 1.4, y0 - L * t + L * 0.15 * t * t))
            P_ = np.asarray(pts)
            tg = np.gradient(P_, axis=0)
            tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
            nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
            wd = R * 0.024 * (1 - 0.7 * np.linspace(0, 1, len(P_)) ** 0.8)[:, None]
            outline = np.vstack([P_ + nr * wd, (P_ - nr * wd)[::-1]])
            d.polygon([(x * s_, y * s_) for x, y in outline], fill=255)
            xe, ye = pts[-1]
            rb = R * rng.uniform(0.035, 0.06)
            gap = rb * rng.uniform(0.6, 2.2)
            bxp, byp = xe + out * gap * 0.4, ye - gap
            d.ellipse([(bxp - rb) * s_, (byp - rb * 1.1) * s_, (bxp + rb) * s_, (byp + rb * 1.1) * s_], fill=255)
            tips.append((bxp, byp, rb))
    fing = shape_mask(hh, ww, fingers)
    drops = []

    def flying(d, s_):
        for _ in range(40):
            ang = rng.uniform(math.pi * 1.08, math.pi * 1.92)
            dist = R * rng.uniform(0.9, 1.6)
            x = cx - ox + math.cos(ang) * dist
            y = cy - oy + math.sin(ang) * dist * 0.75 - R * 0.1
            r = R * 0.035 * rng.uniform(0.25, 1.0) ** 1.5
            d.ellipse([(x - r) * s_, (y - r * 1.3) * s_, (x + r) * s_, (y + r * 1.3) * s_], fill=255)
            drops.append((x, y, r))
    fly = shape_mask(hh, ww, flying)
    X, Y = win[4] - ox, win[5] - oy
    hl = np.zeros(X.shape, F32)
    for (x, y, r) in tips + drops:
        hl = np.maximum(hl, _gauss_spot(X, Y, x - r * 0.35, y - r * 0.45, r * 0.32))
    under = cv.buf[win[1]:win[3], win[0]:win[2]]
    col = rgb(color)
    vrel = clamp01((cy - oy - Y) / (R * 0.8))
    # wall: see-through water, brighter at the silhouette sides (more water along the view) and the top edge
    side = np.abs((X - (cx - ox)) / (R * 0.6))
    fres = clamp01(side) ** 2
    wall_c = lerp(under * 0.8 + col * 0.25, col * 1.05, clamp01(0.25 + 0.55 * fres + 0.35 * vrel))
    ring_back = lerp(under * 0.7 + col * 0.18, col * 0.8, clamp01(0.2 + 0.5 * fres))
    surf_ring = np.sqrt(((X - (cx - ox)) / (R * 0.75)) ** 2 + ((Y - (cy - oy)) / (R * 0.75 * squash)) ** 2)
    ring = clamp01(1 - np.abs(surf_ring - 1) / 0.12) * 0.45 + clamp01(1 - np.abs(surf_ring - 1.35) / 0.06) * 0.2
    _put(cv, col, ring, win)
    _put(cv, ring_back, back_w * 0.75, win)
    _put(cv, wall_c, front_w * 0.7, win)
    edge = clamp01(front_w - gaussian_blur(front_w, 1.5 * S)) * 2
    _put(cv, col * 1.3, edge * 0.8, win)
    # fingers and drops are clear water: the scene shows through, bright along their silhouettes
    f_edge = clamp01((fing - gaussian_blur(fing, 1.2 * S)) * 3)
    fcol = lerp(under * 0.6 + col * 0.35, col * 1.3, clamp01(0.3 * vrel + f_edge))
    _put(cv, fcol, fing * 0.85, win)
    d_edge = clamp01((fly - gaussian_blur(fly, 1.0 * S)) * 3)
    _put(cv, lerp(under * 0.5 + col * 0.45, col * 1.3, d_edge), fly * 0.85, win)
    _put(cv, np.asarray(highlight, F32), clamp01(hl), win, "add")
    a = clamp01(np.maximum.reduce([back_w * 0.75, front_w * 0.7, fing, fly]))
    return FX(_full(cv, a, win))


def caustics(cv, mask=None, scale=70, color="#c8fbff", strength=0.6, seed=0, warp=0.7):
    """Caustics: the bright, wobbly light web that waves focus onto a pool floor / seabed."""
    S = _S(cv)
    H, W = cv.h, cv.w
    m = _mask_or_ones(cv, mask)
    k = 2
    h2, w2 = (H + k - 1) // k, (W + k - 1) // k
    Xs, Ys = grid(h2, w2)
    sc = scale * S / k
    U = Xs / sc + warp * fbm2d(h2, w2, sc * 1.7, 3, seed=seed + 1)
    V = Ys / sc + warp * fbm2d(h2, w2, sc * 1.7, 3, seed=seed + 2)
    _, _, e1, _ = worley(U, V, seed)
    _, _, e2, _ = worley(U * 1.3 + 7.1, V * 1.3 + 2.3, seed + 5)
    web = np.exp(-e1 / 0.035) + 0.6 * np.exp(-e2 / 0.03)
    web = resize(web, W, H, "bilinear")
    cv.paint(color, clamp01(web) * strength * m, "add")
    return FX(clamp01(web) * m)


def underwater(cv, seed=0, colors=("#4cc8d8", "#0f6a86", "#04253c"), rays=1.0, caustic=1.0, bubbles=26, floor=0.8,
               sand="#b8a27a", particles=160):
    """Underwater view: depth gradient, bright rippling surface, light shafts fanning down, a sandy floor with
    ripples and caustics, depth haze, rising bubbles and marine snow. Paints the whole canvas."""
    S = _S(cv)
    H, W = cv.h, cv.w
    rng = np.random.default_rng(seed)
    X, Y = cv.xy()
    V = Y / H
    cv.paint(mix([(0.0, colors[0]), (0.45, colors[1]), (1.0, colors[2])], V))
    # surface: Snell's window shimmer
    surf = fbm2d(H, W, 60 * S, 4, seed=seed + 1, aspect=0.35)
    sm = smoothstep(0.12, 0.0, V) * clamp01(0.5 + surf * 1.5)
    cv.paint(rgb((0.85, 1.1, 1.15)), sm * 0.8, "add")
    # light shafts radiating from above the frame
    if rays:
        ox, oy = W * 0.55, -H * 0.6
        th = np.arctan2(X - ox, Y - oy)
        n = fbm1d(4096, 18, 4, seed=seed + 2)
        idx = np.clip(((th + 1.2) / 2.4 * 4095).astype(int), 0, 4095)
        shafts = clamp01(n[idx] * 1.8) ** 1.5 * np.exp(-V / 0.55) * (0.75 + 0.25 * fbm2d(H, W, 200 * S, 3, seed=seed + 3))
        cv.paint(rgb((0.6, 0.95, 1.0)), shafts * 0.45 * rays, "add")
    # floor
    fy = floor * H + 30 * S * fbm1d(W, 500 * S, 3, seed=seed + 4)[None, :]
    fm = clamp01(Y - fy + 0.5)
    rip = np.sin((X * 0.02 + Y * 0.09) / S + 3 * fbm2d(H, W, 120 * S, 3, seed=seed + 5)) * 0.5 + 0.5
    depth = clamp01((Y - fy) / max(H - floor * H, 1))
    sandc = rgb(sand) * (0.55 + 0.25 * rip + 0.2 * depth)[..., None]
    sandc = lerp(sandc, rgb(colors[1]), clamp01(0.75 - depth * 0.9))
    cv.paint(sandc, fm)
    if caustic:
        caustics(cv, mask=fm * (0.3 + 0.7 * depth), scale=60, strength=0.5 * caustic, seed=seed + 6)
    # marine snow
    xs, ys = rng.uniform(0, W, particles), rng.uniform(0, H, particles)
    snow_ = gaussian_blur(_splat(H, W, xs, ys, rng.uniform(0.3, 1, particles) * 2), 0.8 * S)
    cv.paint(rgb((0.8, 0.95, 1.0)), clamp01(snow_) * 0.6, "add")
    # bubbles: columns rising, smaller on top
    hl = np.zeros((H, W), F32)
    body = np.zeros((H, W), F32)
    for c in range(3):
        bx0 = rng.uniform(0.15, 0.85) * W
        by0 = rng.uniform(0.75, 1.0) * H
        for i in range(bubbles // 3):
            t = i / max(bubbles // 3, 1)
            r = (14 - 10 * t) * S * rng.uniform(0.6, 1.2)
            bxp = bx0 + 18 * S * math.sin(t * 9 + c) + rng.normal(0, 4 * S)
            byp = by0 - t * by0 * 0.95
            win = _win(cv, bxp - r * 2, byp - r * 2, bxp + r * 2, byp + r * 2)
            if win is None:
                continue
            d = np.sqrt((win[4] - bxp) ** 2 + (win[5] - byp) ** 2) / r
            ring = clamp01((1 - d) * r + 0.5) * (0.15 + 0.85 * smoothstep(0.6, 1.0, d))
            body[win[1]:win[3], win[0]:win[2]] = np.maximum(body[win[1]:win[3], win[0]:win[2]], ring)
            hl[win[1]:win[3], win[0]:win[2]] += _gauss_spot(win[4], win[5], bxp - r * 0.35, byp - r * 0.4, r * 0.22)
    cv.paint(rgb((0.75, 0.95, 1.0)), body * 0.55)
    cv.paint(rgb((1.4, 1.5, 1.5)), clamp01(hl), "add")
    return FX(fm)


def droplets(cv, mask=None, region=None, density=1.0, size=16, seed=0, fog=0.35, trails=True):
    """Water drops / condensation on glass: a fogged film of tiny beads, larger drops that refract an
    upside-down view of the background with a bright highlight and dark rim, and clear trails below some."""
    S = _S(cv)
    H, W = cv.h, cv.w
    rng = np.random.default_rng(seed)
    m = _mask_or_ones(cv, mask) * _region_mask(cv, region)
    sharp = cv.buf.copy()
    soft = _blur(sharp, 7 * S)
    X, Y = cv.xy()
    # fogged film: the scene behind, scattered and lifted a little
    film = soft * (1 + 0.15 * fog) + rgb("#ffffff") * (0.05 * fog)
    trail = np.zeros((H, W), F32)
    n = int(90 * density * _area(cv))
    drops = []
    for i in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        if m[int(y), int(x)] < 0.5:
            continue
        r = size * S * (0.3 + 1.5 * rng.uniform(0, 1) ** 2.2)
        drops.append((x, y, r))
        if trails and r > size * S * 0.8 and rng.uniform() < 0.55 * (1 if trails is True else float(trails)):
            # a drop that has run down the glass: clear channel above it with beads left behind
            L = rng.uniform(0.08, 0.45) * H
            win = _win(cv, x - r * 3, y - L, x + r * 3, y)
            if win is not None:
                xw = x + 4 * S * perlin1d_at((win[5][:, :1] - y) / (30 * S), seed + i) +                     1.5 * S * perlin1d_at((win[5][:, :1] - y) / (6 * S), seed + 2 * i)
                tw = r * 0.55 * (0.45 + 0.55 * smoothstep(y - L, y, win[5]))
                tr = clamp01(tw - np.abs(win[4] - xw) + 0.5) * smoothstep(y - L, y - L + 20 * S, win[5])
                trail[win[1]:win[3], win[0]:win[2]] = np.maximum(trail[win[1]:win[3], win[0]:win[2]], tr)
                for _ in range(int(rng.integers(1, 5))):
                    yb = y - rng.uniform(0.15, 0.95) * L
                    xb = x + 4 * S * float(perlin1d_at(np.array([(yb - y) / (30 * S)], F32), seed + i)[0])
                    drops.append((xb, yb, r * rng.uniform(0.2, 0.4)))
    film = lerp(film, sharp, trail)
    cv.paint(film, m * clamp01(fog * 2.5))
    _micro_drops(cv, sharp, soft, m * (1 - 0.85 * trail), density, S, seed)
    for i, (x, y, r) in enumerate(drops):
        _lens_drop(cv, sharp, x, y, r, m, seed * 7919 + i)
    return FX(m)


def _drop_shade(src, dx, dy, rr):
    """Shade a water drop on glass from its local unit coords: the refracted view ``src``, a dark refraction
    rim (heavier on top), a bright caustic crescent low inside and a thin meniscus line on the lower edge."""
    rim = smoothstep(0.62, 0.97, rr)
    col = src * (1.25 - 0.8 * rim * (0.5 + 0.5 * clamp01(0.4 - dy)))[..., None] + (0.03 * (1 - rim))[..., None]
    caus = np.exp(-((rr - 0.74) / 0.12) ** 2) * clamp01(dy * 1.4 + 0.1) * smoothstep(-0.9, 0.2, -np.abs(dx))
    col = col + (src * 1.1 + 0.08) * caus[..., None]
    menisc = np.exp(-((rr - 0.97) / 0.05) ** 2) * clamp01(dy + 0.2)
    return col + (src * 0.6 + 0.05) * menisc[..., None]


def _micro_drops(cv, sharp, soft, m, density, S, seed):
    """Fine drizzle beads over the whole glass: one jittered drop per cell, two sizes, vectorised."""
    X, Y = cv.xy()
    H, W = cv.h, cv.w
    acc = np.zeros((H, W, 3), F32)
    cov = np.zeros((H, W), F32)
    for k, frac, sd in ((max(11 * S, 9.0), 0.5, 11), (max(5.5 * S, 6.0), 0.6, 23)):
        U, V = X / k, Y / k
        iu, iv = np.floor(U), np.floor(V)
        hx, hy = _hashf(iu, iv, seed + sd), _hashf(iu, iv, seed + sd + 1)
        hr, hp = _hashf(iu, iv, seed + sd + 2), _hashf(iu, iv, seed + sd + 3)
        rad = (0.12 + 0.22 * hr ** 1.5) * (hp < frac * min(density, 1.6))
        cxp, cyp = iu + 0.5 + (hx - 0.5) * 0.4, iv + 0.5 + (hy - 0.5) * 0.4
        dx, dy = (U - cxp) / np.maximum(rad, 1e-3), (V - cyp) / np.maximum(rad, 1e-3)
        rr = np.sqrt(dx * dx + dy * dy)
        c = clamp01((1 - rr) * rad * k + 0.5) * (rad > 0)
        src = sample(sharp, np.clip(X - dx * rad * k * 4, 0, W - 1), np.clip(Y - dy * rad * k * 4, 0, H - 1))
        col = _drop_shade(lerp(soft, src, 0.6), dx, dy, rr)
        acc = lerp(acc, col, c)
        cov = np.maximum(cov, c)
    cv.paint(acc, cov * m * 0.9)


def _lens_drop(cv, sharp, x, y, r, m, seed):
    """One larger drop: slightly irregular outline, heavier at the bottom, an inverted view of the scene."""
    H, W = cv.h, cv.w
    win = _win(cv, x - r * 1.35, y - r * 1.4, x + r * 1.35, y + r * 1.45)
    if win is None:
        return
    rng = np.random.default_rng(seed)
    Xl, Yl = win[4], win[5]
    sq = rng.uniform(0.85, 1.05)
    dx = (Xl - x) / r
    dy = (Yl - y) / (r * sq * (1.0 + 0.18 * np.clip((Yl - y) / r, 0, 1)))
    th = np.arctan2(dy, dx)
    wob = 1 + sum(rng.uniform(0.005, 0.025) * np.sin(kk * th + rng.uniform(0, 6.3)) for kk in (2, 3))
    rr = np.sqrt(dx * dx + dy * dy) / wob
    cover = clamp01((1 - rr) * r + 0.5)
    src = sample(sharp, np.clip(x - dx * r * 4.0, 0, W - 1), np.clip(y - dy * r * 4.0, 0, H - 1))
    col = _drop_shade(src * 1.1 + 0.015, dx, dy, rr)
    # specular: a short arc inside the upper-left edge
    gl = np.exp(-((rr - 0.68) / 0.09) ** 2) * np.exp(-((th + 2.1) / 0.35) ** 2)
    col = col + (gl * 1.1)[..., None]
    _put(cv, col, cover * m[win[1]:win[3], win[0]:win[2]], win)


def rain(cv, density=1.0, angle=12.0, layers=3, length=70, color="#cfd9e4", opacity=0.55, seed=0, mask=None,
         ground=None, splashes=60, sheets=0, sheet_band=(0.25, 0.55), sheet_color="#8a94a0"):
    """Rain: streaks in depth layers (far: short, faint, many; near: long, soft, sparse) slanted by ``angle``
    (degrees, + = falling to the right), crown splashes on the ground (``ground`` = v fraction or mask) and
    ``sheets`` distant rain curtains hanging between ``sheet_band`` = (cloud base v, horizon v)."""
    S = _S(cv)
    H, W = cv.h, cv.w
    rng = np.random.default_rng(seed)
    m = _mask_or_ones(cv, mask)
    a = math.radians(angle)
    sx, sy = math.sin(a), math.cos(a)
    total = np.zeros((H, W), F32)
    if sheets:
        X, Y = cv.xy()
        y0, y1 = sheet_band[0] * H, sheet_band[1] * H
        Xs = X - (Y - y0) * math.tan(a) * 0.6
        streak = 0.6 + 0.4 * fbm2d_at(Xs / (6 * S), Y / (200 * S), 3, seed=seed + 1)
        cur = np.zeros((H, W), F32)
        for k in range(int(sheets)):
            cx = rng.uniform(0.05, 0.95) * W
            wd = rng.uniform(60, 220) * S
            edge = fbm2d_at(np.full_like(Y, k * 3.1), Y / (120 * S), 3, seed=seed + 7 + k) * wd * 0.4
            band = smoothstep(wd, wd * 0.3, np.abs(Xs - cx - edge))
            cur = np.maximum(cur, band * rng.uniform(0.5, 1.0))
        vert = smoothstep(y0 - 20 * S, y0 + 60 * S, Y) * smoothstep(y1 + 2 * S, y1 - 30 * S, Y)
        sa = cur * vert * streak * 0.7
        cv.paint(sheet_color, sa * m)
        total = np.maximum(total, sa)
    spec = [(1500, 0.35, 0.8, 0.45, 0.4), (600, 0.65, 1.1, 0.7, 0.7), (130, 1.2, 2.0, 0.85, 1.4)]
    for li, (cnt, lf, wf, al, bl) in enumerate(spec[-layers:] if layers < 3 else spec):
        cnt = int(cnt * density * _area(cv))
        xs = rng.uniform(-0.2 * W, 1.1 * W, cnt)
        ys = rng.uniform(-0.1 * H, 1.05 * H, cnt)
        Ls = length * S * lf * rng.uniform(0.6, 1.3, cnt)
        wd = max(wf * S, 0.5)

        def draw(d, s, xs=xs, ys=ys, Ls=Ls, wd=wd):
            ww = max(1, int(round(wd * s)))
            for x, y, L in zip(xs, ys, Ls):
                d.line([(x * s, y * s), ((x + sx * L) * s, (y + sy * L) * s)], fill=255, width=ww)
        lm = shape_mask(H, W, draw, aa=2)
        lm = gaussian_blur(lm, bl * S) * (1 + 0.3 * bl)
        al_ = clamp01(lm) * al * opacity * m
        cv.paint(color, al_)
        total = np.maximum(total, al_)
    if ground is not None and splashes:
        gm = np.asarray(ground, F32) if not np.isscalar(ground) else None
        pts = []
        for i in range(int(splashes * 6)):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            if gm is not None:
                if gm[int(y), int(x)] < 0.5:
                    continue
            elif y < ground * H:
                continue
            pts.append((x, y))
            if len(pts) >= splashes:
                break

        srng = np.random.default_rng(seed + 99)
        spl = [(x, y, srng.uniform(0.7, 1.3), srng.uniform(0, 6.28)) for x, y in pts]

        def drs(d, s):
            for x, y, sz, ph in spl:
                k = (0.6 + 1.4 * (y / H)) * S * sz                     # nearer splashes are bigger
                r = 9 * k
                lw = max(1, int(round(s * 0.7 * k)))
                # impact ring on the surface, a short jet in the middle and a few beads thrown up and out
                d.ellipse([(x - r) * s, (y - r * 0.26) * s, (x + r) * s, (y + r * 0.26) * s], outline=90, width=lw)
                jet = r * (0.5 + 0.4 * math.sin(ph * 5) ** 2)
                d.line([(x * s, y * s), (x * s, (y - jet) * s)], fill=170, width=lw)
                for j in range(2 + int(ph) % 3):
                    t = ph * 1.7 + j * 2.4
                    bx_ = x + math.cos(t) * r * (0.5 + 0.4 * math.sin(ph + j) ** 2)
                    by_ = y - r * (0.35 + 0.7 * math.sin(ph * 3 + j) ** 2)
                    rd = max(0.7 * k, 0.5 / s)
                    d.ellipse([(bx_ - rd) * s, (by_ - rd * 1.2) * s, (bx_ + rd) * s, (by_ + rd * 1.2) * s], fill=220)
        sp = gaussian_blur(shape_mask(H, W, drs, aa=3), 0.5 * S)
        cv.paint(color, sp * opacity * 0.9 * m)
    return FX(total)


# =============================================================================
# weather and sky phenomena
# =============================================================================
def snowfall(cv, density=1.0, layers=3, wind=0.25, size=3.2, color="#f3f6ff", seed=0, bokeh=True, mask=None,
             opacity=0.9):
    """Falling snow in depth layers: tiny sharp far flakes, larger soft near flakes and a few big out-of-focus
    bokeh flakes in front. ``wind`` slants the slight motion blur."""
    S = _S(cv)
    H, W = cv.h, cv.w
    rng = np.random.default_rng(seed)
    m = _mask_or_ones(cv, mask)
    spec = [(1100, 0.45, 0.25, 0.7), (380, 0.8, 0.6, 0.85), (90, 1.5, 1.4, 0.95)]
    total = np.zeros((H, W), F32)
    for cnt, rf, bl, al in spec[:max(1, int(layers))]:
        cnt = int(cnt * density * _area(cv))
        xs = rng.uniform(0, W, cnt)
        ys = rng.uniform(0, H, cnt)
        rs = size * S * rf * rng.uniform(0.5, 1.3, cnt)

        def draw(d, s, xs=xs, ys=ys, rs=rs):
            for x, y, r in zip(xs, ys, rs):
                d.ellipse([(x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s], fill=255)
        fm = shape_mask(H, W, draw, aa=2)
        if wind:
            Xg, Yg = cv.xy()
            fm = 0.5 * fm + 0.5 * sample(fm, Xg - wind * 2 * rf * S, Yg - 2 * rf * S)
        fm = gaussian_blur(fm, bl * S)
        a = clamp01(fm * (1.2 + bl * 0.3)) * al * opacity * m
        cv.paint(color, a)
        total = np.maximum(total, a)
    if bokeh:
        n = int(14 * density)
        for i in range(n):
            x, y = rng.uniform(0, W), rng.uniform(0, H)
            r = rng.uniform(14, 34) * S
            win = _win(cv, x - r, y - r, x + r, y + r)
            if win is None:
                continue
            d = np.sqrt((win[4] - x) ** 2 + (win[5] - y) ** 2) / r
            disc = clamp01((1 - d) * r * 0.25) * (0.35 + 0.3 * smoothstep(0.6, 0.95, d))
            _put(cv, rgb(color), disc * rng.uniform(0.25, 0.5) * m[win[1]:win[3], win[0]:win[2]], win)
    return FX(total)


def snow_cap(cv, mask, depth=14, seed=0, color="#f2f6fc", shadow="#a3b2cc", light=(-0.4, -0.9), bulge=0.6):
    """Snow lying on top edges of ``mask`` (roofs, branches, rocks, letters): a band of snow hugging every
    upward-facing edge with a rounded, lumpy top that overhangs slightly and a blue shadowed underside."""
    S = _S(cv)
    H, W = cv.h, cv.w
    b = np.asarray(mask, F32) > 0.5
    rows = np.arange(H)[:, None]
    last_out = np.maximum.accumulate(np.where(~b, rows, -1), axis=0)
    dist_top = (rows - last_out).astype(F32)                          # px below the nearest top edge
    nb = np.where(b, rows, H + 10)
    next_in = np.minimum.accumulate(nb[::-1], axis=0)[::-1]
    dist_up = (next_in - rows).astype(F32)                            # px above the next surface below
    D = depth * S * (0.7 + 0.6 * clamp01(0.5 + fbm1d(W, 60 * S, 3, seed=seed)))[None, :]
    lump = D * bulge * (0.55 + 0.45 * clamp01(0.5 + fbm1d(W, 22 * S, 3, seed=seed + 1)))[None, :]
    cap_in = b & (dist_top <= D)
    over = (~b) & (dist_up <= lump) & (dist_up > 0)
    cap = (cap_in | over).astype(F32)
    cap = smoothstep(0.35, 0.65, gaussian_blur(cap, 0.9 * S))
    # shading: top bright, underside blue
    t = np.where(b, (dist_top + lump) / (D + lump), (lump - dist_up) / (D + lump))
    t = clamp01(t)
    sh = clamp01(1 - t * 1.1)
    col = lerp(shadow, color, smoothstep(0.0, 0.7, sh))
    sparkle = (fbm2d(H, W, 3 * S, 2, seed=seed + 2) > 0.55) * sh * 0.4
    col = col + sparkle[..., None]
    cv.paint(col, cap)
    return FX(cap)


def _bolt(rng, p0, p1, depth, rough):
    pts = [p0, p1]
    disp = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * rough
    for lev in range(depth):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            t = rng.uniform(0.35, 0.65)
            mx, my = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy) + 1e-9
            o = rng.normal(0, disp) if lev < 3 else rng.choice([-1, 1]) * abs(rng.normal(0, disp)) * 1.3
            new += [(mx - dy / L * o, my + dx / L * o), b]
        pts = new
        disp *= 0.5
    return pts


def lightning(cv, start=(0.45, 0.05), end=(0.55, 0.9), seed=0, branches=7, width=2.2, color=(0.78, 0.82, 1.0),
              glow=1.0, clouds=None, flash=0.12, rough=0.22):
    """Lightning: a jagged main channel with forking branches, white-hot core, violet glow; with ``clouds``
    (mask) the flash lights up the clouds around the bolt. Returns FX(core mask, [light])."""
    S = _S(cv)
    H, W = cv.h, cv.w
    rng = np.random.default_rng(seed)
    p0, p1 = _xy(cv, start), _xy(cv, end)
    main = _bolt(rng, p0, p1, 8, rough)
    paths = [(main, width * S)]
    L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    base_ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for i in range(int(branches)):
        k = int(rng.uniform(0.08, 0.7) * (len(main) - 1))
        a = base_ang + rng.choice([-1, 1]) * rng.uniform(0.35, 0.9)
        ln = L * rng.uniform(0.12, 0.35) * (1 - k / len(main))
        q0 = main[k]
        q1 = (q0[0] + math.cos(a) * ln, q0[1] + math.sin(a) * ln)
        br = _bolt(rng, q0, q1, 5, rough * 1.2)
        paths.append((br, width * S * 0.45))
        if rng.uniform() < 0.5:
            k2 = len(br) // 2
            a2 = a + rng.choice([-1, 1]) * rng.uniform(0.3, 0.7)
            q2 = (br[k2][0] + math.cos(a2) * ln * 0.4, br[k2][1] + math.sin(a2) * ln * 0.4)
            paths.append((_bolt(rng, br[k2], q2, 4, rough * 1.2), width * S * 0.25))

    def draw(d, s):
        for pts, w in paths:
            n = len(pts)
            for j in range(n - 1):
                ww = max(1, int(round(w * s * (1 - 0.5 * j / n))))
                d.line([(pts[j][0] * s, pts[j][1] * s), (pts[j + 1][0] * s, pts[j + 1][1] * s)], fill=255, width=ww)
    core = shape_mask(H, W, draw, aa=3)
    g1 = _blur(core, 4 * S)
    g2 = _blur(core, 30 * S)
    c = rgb(color)
    if clouds is not None:
        lit = _blur(core, 110 * S)
        lit = (lit / max(float(lit.max()), 1e-6))
        cv.paint(c * 0.55, np.asarray(clouds, F32) * lit * glow, "add")
    if flash:
        cv.paint(c, np.full((H, W), flash * glow, F32), "add")
    cv.paint(c * 1.2, clamp01(g2 * 5) * 0.35 * glow, "add")
    cv.paint(c * 1.4, clamp01(g1 * 1.5) * 0.7 * glow, "add")
    cv.paint(rgb((2.6, 2.6, 3.0)), core)
    mid = main[len(main) // 3]
    return FX(core, [Light(mid[0], mid[1], c, L * 0.8, 1.5)], path=main)


def rainbow(cv, center=(0.5, 1.15), radius=0.8, width=0.055, opacity=0.45, secondary=True, mask=None, seed=0):
    """Rainbow arc: spectral bands (red outside), brighter sky inside the bow, a faint reversed secondary bow
    with the darker Alexander band between. ``radius`` / ``width`` are fractions of the canvas height.
    ``mask`` limits the bow to spray or rain (``waterfall(...).spray``, ``fog(...).mask``); a faint mask is
    normalised so the bow stays visible."""
    H, W = cv.h, cv.w
    X, Y = cv.xy()
    cx, cy = _xy(cv, center)
    R = radius * H
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / R
    m = clamp01(_mask_or_ones(cv, mask))
    if mask is not None and np.any(m > 0.01):                     # a faint spray / fog mask still shows the bow
        top = float(np.percentile(m[m > 0.01], 98))
        if 0 < top < 0.9:
            m = clamp01(m / top)
    var = 0.6 + 0.4 * clamp01(0.5 + fbm2d(H, W, 300 * _S(cv), 3, seed=seed))
    spec = [(0.0, "#6a2cff"), (0.17, "#2c5cff"), (0.36, "#18c050"), (0.55, "#f0e030"), (0.75, "#ff8a1a"), (1.0, "#ff2a2a")]

    def bow(r0, w, k, rev=False):
        t = (d - (r0 - w / 2)) / w
        band = np.clip(np.sin(np.pi * clamp01(t)), 0, 1) ** 0.7
        col = mix(spec, 1 - t if rev else t)
        al = band * k * var * m
        cv.paint(col * 1.05, al * 0.75, "normal")                 # hue that holds on a bright sky
        cv.paint(col, al * 0.6, "add")                              # luminous on a dark one
    inside = smoothstep(1 - width * 0.5, 1 - width * 3, d) * smoothstep(0.55, 0.9, d)
    cv.paint(rgb("#ffffff"), inside * 0.05 * opacity * m * var, "screen")
    bow(1.0, width, opacity)
    if secondary:
        dark = smoothstep(1 + width * 0.5, 1 + width, d) * smoothstep(1.33 - width, 1.2, d)
        cv.paint(rgb("#000000"), dark * 0.06 * m, "normal")
        bow(1.33, width * 1.4, opacity * 0.4, rev=True)
    return FX(clamp01(smoothstep(1 - width, 1, d) * smoothstep(1 + width, 1, d)) * m)


def aurora(cv, v_range=(0.1, 0.5), colors=("#2bff8f", "#1fd4a8", "#9b4dff"), seed=0, strength=1.0, curtains=2,
           mask=None, height=300):
    """Aurora borealis: folded curtains with a sharp lower edge, fine vertical rays and a glow fading upwards,
    green at the base shifting to violet at the top (``colors`` = base → top)."""
    S = _S(cv)
    H, W = cv.h, cv.w
    X, Y = cv.xy()
    m = _mask_or_ones(cv, mask)
    acc = np.zeros((H, W, 3), F32)
    v0, v1 = v_range
    cols = np.arange(W)
    for c in range(int(curtains)):
        sd = seed + 31 * c
        # the curtain is a ribbon: s runs along it, x(s) folds back and forth; where the ribbon folds the
        # screen sees more of it edge-on, so brightness piles up into the bright vertical folds of a real aurora
        ns = 4 * W
        sv = np.linspace(-0.15 * W, 1.15 * W, ns).astype(F32)
        fold = 0.07 * W * perlin1d_at(sv / (260 * S) + 3.1 * c, sd) + 0.025 * W * perlin1d_at(sv / (90 * S), sd + 9)
        xs = sv + fold
        ybs = (v0 + (v1 - v0) * (0.55 + 0.45 * perlin1d_at(sv / (700 * S), sd + 1))) * H
        ybs = ybs + 40 * S * perlin1d_at(sv / (200 * S), sd + 2)
        ray = clamp01(0.3 + 0.9 * (0.5 + 0.5 * perlin1d_at(sv / (9 * S), sd + 4)) * (0.6 + 0.4 * perlin1d_at(sv / (3.5 * S), sd + 5)))
        ray_h = 0.45 + 0.75 * clamp01(0.5 + perlin1d_at(sv / (40 * S), sd + 7))
        bright = clamp01(0.3 + 1.4 * perlin1d_at(sv / (380 * S), sd + 6)) ** 1.3 * (1.0 if c == 0 else 0.6)
        wgt = ray * bright
        ok = (xs >= -1) & (xs < W)
        # splat the hem of every ribbon sample, then let it glow upwards with a one-sided exponential:
        # overlapping folds stay separate, each with its own bright lower edge
        hem = _splat(H, W, xs[ok], ybs[ok], wgt[ok])
        hem = gaussian_blur(hem, 1.3 * S)
        idx = np.clip(np.round(xs[ok]).astype(int), 0, W - 1)
        cnt = np.bincount(idx, minlength=W)
        has = cnt > 0
        rh = np.bincount(idx, ray_h[ok], W) / np.maximum(cnt, 1)
        if has.any():
            rh = np.interp(cols, cols[has], rh[has])
        colsum = hem.sum(0)
        hem = hem / max(float(np.mean(colsum[colsum > 1e-6])) if np.any(colsum > 1e-6) else 1.0, 1e-6)
        hgt = np.maximum(height * S * rh, 2.0).astype(F32)
        Ls, Ll = hgt * 0.38, hgt * 0.77
        ds, dl = np.exp(-1.0 / Ls).astype(F32), np.exp(-1.0 / Ll).astype(F32)
        short = np.empty_like(hem)
        long_ = np.empty_like(hem)
        rs = np.zeros(W, F32)
        rl = np.zeros(W, F32)
        for y in range(H - 1, -1, -1):
            rs = hem[y] + ds * rs
            rl = hem[y] + dl * rl
            short[y] = rs
            long_[y] = rl
        dist = np.log((long_ + 1e-6) / (short + 1e-6)) / (1.0 / Ls - 1.0 / Ll)[None, :]
        a = clamp01(dist / hgt[None, :] / 1.15)
        I = 0.85 * short + 0.15 * long_ + 0.5 * hem * hgt[None, :] * 0.05
        I = I * (1 - smoothstep(0.5, 1.0, a))
        inten = 0.7 * I / (1 + 0.3 * I)                                # folds bright, not blown
        col = _ramp(colors, clamp01(a * 1.0))
        acc += col * inten[..., None]
    acc = acc + _blur(acc, 18 * S) * 0.4
    cv.paint(acc * strength * 1.3, m, "add")
    return FX(clamp01(luma(acc)))


def god_rays(cv, at=(0.5, 0.3), occluder=None, strength=0.3, color="#ffe0a8", length=0.85, threshold=0.8,
             reach=0.3, streaks=0.7, seed=0):
    """God rays / crepuscular rays from a light at ``at`` streaming through gaps: bright pixels near the light
    that are not covered by ``occluder`` (clouds, trees, a window frame) are smeared radially; ``streaks`` 0~1
    breaks the beams up even where nothing occludes them."""
    H, W = cv.h, cv.w
    X, Y = cv.xy()
    cx, cy = _xy(cv, at)
    near = np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (reach * H) ** 2))
    lum = luma(cv.buf)
    ref = float(np.percentile(lum[near > 0.5], 98)) if np.any(near > 0.5) else float(lum.max())
    src = smoothstep(ref * threshold, ref * (threshold + 0.15) + 1e-3, lum) * near
    if occluder is not None:
        src = src * (1 - clamp01(np.asarray(occluder, F32)))
    if streaks:
        ang = np.arctan2(Y - cy, X - cx) + math.pi
        st = clamp01(0.5 + 0.9 * perlin1d_at(ang * 7.0, seed) + 0.5 * perlin1d_at(ang * 19.0, seed + 1))
        src = src * (1 - streaks + streaks * st)
    rays = light_shafts(src, (cx, cy), samples=48, decay=0.965, length=length, downscale=2)
    rays = rays / max(float(rays.max()), 1e-6)
    cv.paint(color, rays * strength, "add")
    return FX(rays)


def heat_haze(cv, mask=None, region=None, strength=3.0, scale=26, seed=0):
    """Heat haze / mirage shimmer: wavy displacement of what is already painted (above fire, hot roads)."""
    S = _S(cv)
    H, W = cv.h, cv.w
    m = _mask_or_ones(cv, mask) * _region_mask(cv, region)
    X, Y = cv.xy()
    dx = fbm2d(H, W, scale * S, 3, seed=seed, aspect=0.5) * strength * S
    dy = fbm2d(H, W, scale * S * 1.5, 3, seed=seed + 1, aspect=0.4) * strength * S * 0.5
    cv.paint(cv.sample(X + dx, Y + dy), m)
    return FX(m)


# =============================================================================
# light sources and light effects
# =============================================================================
def glow(cv, at=(0.5, 0.5), radius=60, color="#ffd9a0", strength=1.0, mask=None):
    """Soft light glow (tight core + wide falloff, HDR). ``radius`` in 1080p px. Returns FX with its Light."""
    S = _S(cv)
    x, y = _xy(cv, at)
    _soft_glow(cv, x, y, radius * S, rgb(color), strength, mask)
    return FX(None, [Light(x, y, color, radius * S * 3, strength)])


def lamp(cv, at=(0.5, 0.3), size=80, color="#ffd08a", shade="#2c2a2e", cord=True, cone=0.35, seed=0):
    """Pendant lamp: cord, conical shade with a lit rim, a hot bulb and a soft cone of light below.
    ``at`` = shade top centre, ``size`` = shade width."""
    S = _S(cv)
    cx, cy = _xy(cv, at)
    w = size * S
    hgt = w * 0.55
    if cord:
        cw = _win(cv, cx - 3 * S, 0, cx + 3 * S, cy + 2)
        if cw is not None:
            _put(cv, rgb("#151416"), clamp01(1.2 * S - np.abs(cw[4] - cx) + 0.5), cw)
    if cone:
        win = _win(cv, cx - w * 3, cy + hgt, cx + w * 3, cv.h)
        if win is not None:
            X, Y = win[4], win[5]
            t = (Y - cy - hgt) / (cv.h - cy - hgt + 1)
            half = w * 0.5 + (Y - cy - hgt) * 0.75
            e = smoothstep(half, half * 0.55, np.abs(X - cx)) * np.exp(-t * 2.5) * (1 - t)
            _put(cv, rgb(color), e * cone, win, "add")
    win = _win(cv, cx - w * 0.6, cy - 2, cx + w * 0.6, cy + hgt + w * 0.2)
    X, Y = win[4], win[5]
    v = (Y - cy) / hgt
    half = w * (0.14 + 0.36 * np.clip(v, 0, 1))
    body = clamp01(half - np.abs(X - cx) + 0.5) * clamp01(Y - cy + 0.5) * clamp01(cy + hgt - Y + 0.5)
    sx = (X - cx) / np.maximum(half, 1)
    col = rgb(shade) * (0.5 + 0.7 * clamp01(0.6 - sx * 0.8))[..., None]
    col = col + rgb(color) * (smoothstep(0.85, 1.0, v) * 0.5)[..., None]
    _put(cv, col, body, win)
    bx, by = cx, cy + hgt + w * 0.02
    bulb = clamp01(w * 0.13 - np.sqrt((X - bx) ** 2 + ((Y - by) * 1.3) ** 2) + 0.5) * (Y > cy + hgt - 1)
    _put(cv, rgb(color) * 3.0, bulb, win)
    _soft_glow(cv, bx, by, w * 0.7, rgb(color), 0.9)
    return FX(_full(cv, body, win), [Light(bx, by, color, w * 6, 1.2)])


def lantern(cv, at=(0.5, 0.5), size=110, style="glass", color="#ffb85a", frame="#2a2420", paper="#e0402a",
            seed=0):
    """Lantern: ``style='glass'`` (metal cap, handle ring, glowing glass panes with frame bars and a flame) or
    ``'paper'`` (round paper lantern with ribs, lit from inside). ``at`` = centre, ``size`` = height."""
    S = _S(cv)
    cx, cy = _xy(cv, at)
    h = size * S
    _choice("lantern", "style", style, ("glass", "paper"))
    w = h * (0.55 if style == "glass" else 0.9)
    _soft_glow(cv, cx, cy, h * 1.1, rgb(color) * 0.6, 0.9)
    win = _win(cv, cx - w, cy - h * 0.95, cx + w, cy + h * 0.65)
    X, Y = win[4], win[5]
    u, v = (X - cx) / (w * 0.5), (Y - cy) / (h * 0.5)
    if style == "paper":
        rr = np.sqrt(u * u + (v * 1.15) ** 2)
        body = clamp01((1 - rr) * w * 0.5 + 0.5)
        ribs = 0.75 + 0.25 * np.abs(np.cos(v * 7.5))
        glow_in = (1.6 - 1.1 * rr) * ribs
        col = rgb(paper) * glow_in[..., None] + rgb(color) * (clamp01(1 - rr * 1.3) * 0.9)[..., None]
        col = col * (0.55 + 0.45 * np.sqrt(clamp01(1 - u * u)))[..., None]
        _put(cv, col, body, win)
        for yy, ww in ((-0.92, 0.28), (0.92, 0.28)):
            cap = clamp01(ww * w * 0.5 - np.abs(X - cx) + 0.5) * clamp01(h * 0.06 - np.abs(Y - cy - yy * h * 0.5) + 0.5)
            _put(cv, rgb("#1a1512"), cap, win)
        tas = clamp01(1.4 * S - np.abs(X - cx) + 0.5) * (Y > cy + h * 0.5) * (Y < cy + h * 0.62)
        _put(cv, rgb(paper) * 0.7, tas, win)
    else:
        glass = clamp01((1 - np.abs(u)) * w * 0.5 + 0.5) * clamp01((0.7 - np.abs(v)) * h * 0.5 + 0.5)
        inner = np.exp(-(u * u * 1.5 + (v - 0.15) ** 2 * 2.0))
        col = rgb(color) * (0.55 + 1.3 * inner)[..., None]
        _put(cv, col, glass * 0.92, win)
        bars = (clamp01(2.2 * S - np.abs(np.abs(X - cx) - w * 0.5 + 1.5 * S) + 0.5) +
                clamp01(1.6 * S - np.abs(X - cx) + 0.5) * 0.9) * (np.abs(v) < 0.75)
        rails = clamp01(h * 0.035 - np.abs(np.abs(Y - cy) - h * 0.35) + 0.5) * (np.abs(u) < 1.12)
        capm = clamp01((1.15 - np.abs(u) - 0.6 * (v + 0.95)) * w * 0.3) * (v < -0.7) * (v > -1.05)
        basem = clamp01((1.1 - np.abs(u)) * w * 0.4) * (v > 0.7) * (v < 0.85)
        rd = np.sqrt((X - cx) ** 2 + (Y - cy + h * 0.6) ** 2)
        ring = clamp01(1.8 * S - np.abs(rd - h * 0.14) + 0.5) * (Y < cy - h * 0.55)
        fr = clamp01(bars + rails + capm + basem + ring)
        frc = rgb(frame) * (1 + 0.6 * smoothstep(0.6, -0.4, v))[..., None] + rgb(color) * (0.25 * inner)[..., None]
        _put(cv, frc, fr, win)
        candle(cv, at=(cx / cv.w, (cy + h * 0.22) / cv.h), size=size * 0.28, body=True, body_height=size * 0.1,
               glow=0.4, seed=seed)
    return FX(_full(cv, clamp01(u * 0 + 1) * (np.abs(u) < 1) * (np.abs(v) < 1), win),
              [Light(cx, cy, color, h * 4, 1.3)])


def neon(cv, points=None, text=None, at=(0.5, 0.5), size=140, color="#ff3cac", width=7, glow=1.0, closed=False,
         font=None, flicker=0.0, seed=0):
    """Neon tube: a glowing glass tube with a white-hot core, saturated edge and wide coloured glow. Give a path
    ``points=[(u, v), ...]`` (fractions) or a ``text`` string (outline tubes around the letters, centred at
    ``at``, ``size`` = letter height). Returns FX(tube mask, [light])."""
    S = _S(cv)
    H, W = cv.h, cv.w
    if text is not None:
        from .text import text_mask
        x, y = _xy(cv, at)
        st = max(1, int(round(width * S * 0.5)))
        outer = text_mask(H, W, text, (x, y), size=int(size * S), font=font, anchor="mm", stroke=st)
        inner = text_mask(H, W, text, (x, y), size=int(size * S), font=font, anchor="mm")
        tube = clamp01(outer - inner)
        tube = np.maximum(tube, smoothstep(0.5, 1.0, gaussian_blur(inner, 0.5)) * 0.0)
    else:
        pts = [_xy(cv, p) for p in points]
        tube = line_mask(H, W, pts, width=width * S, closed=closed)
    core = smoothstep(0.55, 0.95, gaussian_blur(tube, width * S * 0.18))
    c = rgb(color)
    if flicker:
        rng = np.random.default_rng(seed)
        glow = glow * (1 - flicker * rng.uniform())
    g1 = _blur(tube, 5 * S)
    g2 = _blur(tube, 26 * S)
    cv.paint(c, clamp01(g2 * 3) * 0.55 * glow, "add")
    cv.paint(c * 1.2, clamp01(g1 * 1.6) * 0.8 * glow, "add")
    cv.paint(c * 1.7, tube)
    cv.paint(lerp(c, rgb("#ffffff"), 0.7) * 2.2, core * tube)
    ys, xs = np.nonzero(tube > 0.5)
    cxl, cyl = (float(xs.mean()), float(ys.mean())) if len(xs) else (W / 2, H / 2)
    return FX(tube, [Light(cxl, cyl, c, max(W, H) * 0.25, 0.8)])


def window_light(cv, box=(0.4, 0.3, 0.55, 0.55), color="#ffc46a", panes=(2, 2), frame="#2a1f18", floor=None,
                 glow=0.8, interior=True, seed=0):
    """Lit window at night: warm glowing panes with mullions and a soft halo; with ``floor=(v0, v1, shift)``
    (fractions) it also casts the window's light patch with mullion shadows on the floor / ground."""
    S = _S(cv)
    x0, y0 = _xy(cv, box[:2])
    x1, y1 = _xy(cv, box[2:])
    c = rgb(color)
    _soft_glow(cv, (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.9, c * 0.35, glow)
    win = _win(cv, x0, y0, x1, y1)
    if win is None:
        return FX(None)
    X, Y = win[4], win[5]
    u, v = (X - x0) / (x1 - x0), (Y - y0) / (y1 - y0)
    inner = np.exp(-((u - 0.5) ** 2 * 3 + (v - 0.65) ** 2 * 2.5))
    col = c * (0.8 + 1.1 * inner)[..., None]
    if interior:
        n = fbm2d_at(X / (60 * S), Y / (60 * S), 3, seed=seed)
        col = col * (0.85 + 0.25 * n)[..., None] * (1 - 0.35 * smoothstep(0.55, 0.2, v) * smoothstep(0.3, 0.0, np.abs(u - 0.3)))[..., None]
    fw = max(2.5 * S, (x1 - x0) * 0.04)
    fr = np.zeros_like(X)
    for k in range(panes[0] + 1):
        xx = x0 + (x1 - x0) * k / panes[0]
        fr = np.maximum(fr, clamp01(fw * (0.9 if 0 < k < panes[0] else 1.4) - np.abs(X - xx) + 0.5))
    for k in range(panes[1] + 1):
        yy = y0 + (y1 - y0) * k / panes[1]
        fr = np.maximum(fr, clamp01(fw * (0.9 if 0 < k < panes[1] else 1.4) - np.abs(Y - yy) + 0.5))
    _put(cv, col, np.ones_like(X), win)
    _put(cv, rgb(frame), fr, win)
    lt = Light((x0 + x1) / 2, (y0 + y1) / 2, color, (x1 - x0) * 3, 1.0)
    if floor is not None:
        fv0, fv1, shift = floor
        fy0, fy1 = fv0 * cv.h, fv1 * cv.h
        sh = shift * cv.w
        fwn = _win(cv, min(x0, x0 + sh) - 10, fy0, max(x1, x1 + sh * 2) + 10, fy1)
        if fwn is not None:
            Xf, Yf = fwn[4], fwn[5]
            t = (Yf - fy0) / max(fy1 - fy0, 1)
            spread = 1 + 0.8 * t
            cxf = (x0 + x1) / 2 + sh * t
            uf = (Xf - cxf) / ((x1 - x0) * spread) + 0.5
            inside = smoothstep(-0.02, 0.02, uf) * smoothstep(1.02, 0.98, uf) * smoothstep(0, 0.03, t) * smoothstep(1, 0.97, t)
            mull = np.ones_like(Xf)
            for k in range(1, panes[0]):
                mull = mull * smoothstep(0.012, 0.03, np.abs(uf - k / panes[0]))
            for k in range(1, panes[1]):
                mull = mull * smoothstep(0.012, 0.03, np.abs(t - k / panes[1]))
            patch = gaussian_blur(inside * mull, 2 * S) * (1 - 0.5 * t)
            _put(cv, c * 0.55, patch, fwn, "add")
    return FX(_full(cv, np.ones_like(X), win), [lt])


def bokeh(cv, n=40, region=None, colors=("#ffd27a", "#ff8a5a", "#7ab8ff", "#ff5c8a"), size=40, seed=0,
          shape="circle", opacity=0.5, mask=None):
    """Bokeh: out-of-focus light discs with a brighter rim (``shape`` = circle / hexagon), additive."""
    S = _S(cv)
    rng = np.random.default_rng(seed)
    m = _mask_or_ones(cv, mask) * _region_mask(cv, region)
    u0, v0, u1, v1 = region or (0, 0, 1, 1)
    for i in range(int(n)):
        x, y = rng.uniform(u0, u1) * cv.w, rng.uniform(v0, v1) * cv.h
        r = size * S * rng.uniform(0.45, 1.3)
        win = _win(cv, x - r - 2, y - r - 2, x + r + 2, y + r + 2)
        if win is None:
            continue
        dx, dy = win[4] - x, win[5] - y
        if shape.startswith("hex"):
            ang = np.arctan2(dy, dx)
            d = np.sqrt(dx * dx + dy * dy) * np.cos((ang % (math.pi / 3)) - math.pi / 6) / math.cos(math.pi / 6)
        else:
            d = np.sqrt(dx * dx + dy * dy)
        rr = d / r
        disc = clamp01((1 - rr) * r * 0.35 + 0.5) * (0.55 + 0.45 * smoothstep(0.65, 0.97, rr))
        c = rgb(colors[i % len(colors)]) * rng.uniform(0.5, 1.2)
        _put(cv, c, disc * opacity * m[win[1]:win[3], win[0]:win[2]], win, "add")
    return FX(m)


def spotlight(cv, at=(0.5, 0.0), to=(0.5, 0.9), angle=24, color="#fff0d0", strength=0.7, dust=True, pool=True,
              seed=0, mask=None):
    """Spotlight / light beam: a soft-edged volumetric cone from ``at`` towards ``to`` (fractions) with
    ``angle`` (full cone, degrees), uneven haze inside, dust motes and a bright pool where it lands."""
    S = _S(cv)
    H, W = cv.h, cv.w
    X, Y = cv.xy()
    sx, sy = _xy(cv, at)
    tx, ty = _xy(cv, to)
    ax, ay = tx - sx, ty - sy
    L = math.hypot(ax, ay) + 1e-6
    ax, ay = ax / L, ay / L
    along = (X - sx) * ax + (Y - sy) * ay
    across = np.abs(-(X - sx) * ay + (Y - sy) * ax)
    half = math.tan(math.radians(angle / 2))
    rel = across / np.maximum(along * half, 1e-3)
    edge = smoothstep(1.0, 0.6, rel) * (along > 0)
    fall = 1.0 / (1 + (along / (L * 0.9)) ** 2 * 1.2) * smoothstep(L * 1.05, L * 0.9, along)
    haze = 0.7 + 0.3 * fbm2d(H, W, 140 * S, 4, seed=seed)
    beam = edge * fall * haze * smoothstep(0, 30 * S, along)
    m = _mask_or_ones(cv, mask)
    c = rgb(color)
    cv.paint(c, beam * strength * m, "add")
    if pool:
        rw = L * half
        pd = ((X - tx) / rw) ** 2 + ((Y - ty) / (rw * 0.3)) ** 2
        cv.paint(c, np.exp(-pd * 1.5) * strength * 0.9 * m, "add")
    if dust:
        dust_motes(cv, n=int(160 * _area(cv)) + 20, mask=beam * m, seed=seed + 1, color=color)
    _soft_glow(cv, sx, sy, 20 * S, c, 1.0)
    return FX(beam, [Light(tx, ty, color, L * half * 2, 0.8)])


def lens_flare(cv, at=(0.7, 0.3), strength=0.6, color=None, seed=0, streak=True):
    """Lens flare: ghost discs and rings along the line from the light through the image centre, a halo and a
    thin anamorphic streak. Add after the scene, before post."""
    S = _S(cv)
    H, W = cv.h, cv.w
    rng = np.random.default_rng(seed)
    sx, sy = _xy(cv, at)
    cx, cy = W / 2, H / 2
    tints = ["#7fd0ff", "#a0ff9a", "#ffb070", "#d090ff", "#80a0ff", "#ffe080"]
    for i, t in enumerate((0.45, 0.7, 1.05, 1.35, 1.6, 1.95, -0.25)):
        gx, gy = sx + (cx - sx) * t, sy + (cy - sy) * t
        r = (rng.uniform(14, 70) if i % 3 else rng.uniform(60, 110)) * S
        win = _win(cv, gx - r - 3, gy - r - 3, gx + r + 3, gy + r + 3)
        if win is None:
            continue
        d = np.sqrt((win[4] - gx) ** 2 + (win[5] - gy) ** 2) / r
        disc = clamp01((1 - d) * r * 0.2) * (0.35 + 0.65 * smoothstep(0.5, 0.95, d))
        c = rgb(color) if color is not None else rgb(tints[i % len(tints)])
        _put(cv, c, disc * strength * rng.uniform(0.08, 0.2), win, "add")
    X, Y = cv.xy()
    D = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2)
    ring = np.exp(-((D - 160 * S) / (8 * S)) ** 2) * 0.06
    cv.paint(rgb("#bcd8ff"), ring * strength, "add")
    if streak:
        st = np.exp(-((Y - sy) / (2.2 * S)) ** 2) * np.exp(-np.abs(X - sx) / (W * 0.22))
        cv.paint(rgb("#9cc4ff"), st * strength * 0.8, "add")
    return FX(None)


def volumetric_fog(cv, lights, density=0.5, seed=0, scale=220, color=None):
    """Volumetric fog around lights: each light gets a noisy, uneven halo as if seen through haze."""
    S = _S(cv)
    H, W = cv.h, cv.w
    X, Y = cv.xy()
    n = 0.55 + 0.45 * clamp01(0.5 + fbm2d(H, W, scale * S, 4, seed=seed))
    for L in _lights(lights):
        d2 = ((X - L.x) ** 2 + (Y - L.y) ** 2) / (L.radius * 1.2) ** 2
        g = 1.0 / (1 + d2 * 4) * (1 - smoothstep(3.0, 20.0, d2)) * n * n
        cv.paint(L.color if color is None else rgb(color), g * density * min(L.strength, 1.5), "add")
    return FX(None, _lights(lights))


# =============================================================================
# keyword registry for small models
# =============================================================================
def _takes(f, name):
    import inspect
    return name in inspect.signature(f).parameters


class _UnknownName(ValueError, KeyError):
    """Unknown registry name (a ValueError that is also a KeyError)."""
    __str__ = ValueError.__str__


def _unknown(kind, name, valid, hint=None):
    return _UnknownName(*_unknown_name(kind, name, list(valid), hint).args)


def _reg_at(f, **fixed):
    f = _strict(f)

    def call(cv, at=None, size=None, seed=0, **kw):
        args = dict(fixed)
        args.update(kw)
        if at is not None:
            args["at"] = at
        if size is not None:
            for key in ("size", "radius", "height"):
                if _takes(f, key):
                    args.setdefault(key, size)
                    break
        if _takes(f, "seed"):
            args["seed"] = seed
        return f(cv, **args)
    return call


def _reg_region(f, **fixed):
    """Wrap a full-frame / region effect so that ``at`` + ``size`` place it: they become ``region``,
    ``v_range``, ``center``/``radius``, ``start``/``end`` or a soft ``mask``, whichever the function takes."""
    f = _strict(f)

    def call(cv, at=None, size=None, seed=0, **kw):
        args = dict(fixed)
        args.update(kw)
        if at is not None or size is not None:
            u, v = at if at is not None else (0.5, 0.5)
            s = (size if size is not None else 400) / 1080.0
            w = s * cv.h / cv.w
            region = (u - w / 2, v - s / 2, u + w / 2, v + s / 2)
            if _takes(f, "region") and "region" not in args and "mask" not in args:
                args["region"] = region
            elif _takes(f, "v_range") and "v_range" not in args:
                args["v_range"] = (v - s / 2, v + s / 2)
            elif _takes(f, "center") and "center" not in args:
                args["center"] = (u, v)
                if size is not None:
                    args.setdefault("radius", s)
            elif _takes(f, "start") and "start" not in args:
                args["start"] = (u, v)
                args.setdefault("end", (u + w * 0.6, v + s * 0.35))
            elif _takes(f, "mask") and args.get("mask") is None:
                args["mask"] = _region_mask(cv, region)
        if _takes(f, "seed"):
            args["seed"] = seed
        return f(cv, **args)
    return call


def _reg_sky(cv, at=None, size=None, seed=0, kind="sunset", **kw):
    return sky(cv, kind, **kw)


def _reg_lightning(cv, at=None, size=None, seed=0, **kw):
    if at is not None and "start" not in kw:
        L = (size or 700) / 1080.0
        kw["start"] = (at[0], at[1])
        kw["end"] = (at[0] + 0.1 * L, at[1] + L)
    return lightning(cv, seed=seed, **kw)


def _reg_box(f, w=0.3, h=0.4):
    f = _strict(f)

    def call(cv, at=None, size=None, seed=0, **kw):
        if at is not None and "box" not in kw:
            s = (size or 400) / 1080.0
            aw = s * cv.h / cv.w * (w / h)
            kw["box"] = (at[0] - aw / 2, at[1] - s / 2, at[0] + aw / 2, at[1] + s / 2)
        return f(cv, seed=seed, **kw)
    return call


def _reg_points(cv, at=None, size=None, seed=0, **kw):
    if "points" not in kw and "text" not in kw:
        u, v = at or (0.5, 0.5)
        s = (size or 300) / 1080.0
        kw["points"] = [(u - s * 0.8, v + s * 0.2), (u - s * 0.3, v - s * 0.3), (u + s * 0.3, v + s * 0.3),
                        (u + s * 0.8, v - s * 0.2)]
    elif at is not None:
        kw["at"] = at
    if size is not None and "text" in kw:
        kw["size"] = size
    return neon(cv, seed=seed, **kw)


def _reg_vfog(f):
    f = _strict(f)

    def call(cv, at=None, size=None, seed=0, **kw):
        if "lights" not in kw:
            u, v = at if at is not None else (0.5, 0.4)
            kw["lights"] = Light(u * cv.w, v * cv.h, "#ffd9a0", (size or 300) * _S(cv))
        return f(cv, seed=seed, **kw)
    return call


def _reg_moonlit(f):
    f = _strict(f)

    def call(cv, at=None, size=None, seed=0, **kw):
        if at is not None:
            kw.setdefault("horizon", at[1])
        return f(cv, seed=seed, **kw)
    return call


REGISTRY = {
    # fire
    "flame": (_reg_at(flame), "flame fire burning tongue blaze"),
    "campfire": (_reg_at(campfire), "campfire bonfire camp fire logs night camping embers"),
    "candle": (_reg_at(candle), "candle candlelight wax wick birthday flame small"),
    "torch": (_reg_at(torch), "torch burning stick medieval dungeon"),
    "fireplace": (_reg_box(fireplace, 0.5, 0.55), "fireplace hearth chimney brick cozy living room logs"),
    "explosion": (_reg_at(explosion), "explosion fireball blast bomb boom burst"),
    "sparks": (_reg_at(sparks), "sparks spark welding grinder streaks particles"),
    "embers": (_reg_at(embers), "embers glowing ash rising cinders"),
    "fireworks": (_reg_at(fireworks), "fireworks firework festival celebration new year burst"),
    # smoke
    "smoke": (_reg_at(smoke), "smoke column plume rising smoke"),
    "steam": (_reg_at(steam), "steam vapour vapor coffee tea cup hot wisps"),
    "chimney smoke": (_reg_at(chimney_smoke), "chimney smoke wind house roof"),
    "breath": (_reg_at(breath), "breath cold winter exhale mist puff"),
    "fog": (_reg_region(fog), "fog mist haze morning low cloud"),
    "dust motes": (_reg_region(dust_motes), "dust motes particles floating specks sunbeam"),
    "cumulus": (_reg_at(cumulus), "cumulus cloud clouds fluffy summer fair weather"),
    "storm clouds": (_reg_region(storm_clouds), "storm clouds thunder dark cloud overcast heavy"),
    "cirrus": (_reg_region(cirrus), "cirrus wispy high clouds streaks mares tails"),
    "cloud bands": (_reg_region(cloud_bands), "sunset clouds bands stratus dawn evening lit clouds"),
    # water
    "ocean": (_reg_moonlit(ocean), "ocean sea waves swell water seascape"),
    "breaking wave": (_reg_at(breaking_wave), "breaking wave surf curl tube beach"),
    "lake": (_reg_moonlit(lake), "lake reflection calm water pond mirror"),
    "river": (_reg_region(river_flow), "river flow stream current flowing water lines"),
    "waterfall": (_reg_box(waterfall, 0.2, 0.6), "waterfall falls cascade cliff"),
    "splash": (_reg_at(splash), "splash droplets crown drop impact"),
    "ripples": (_reg_at(ripples), "ripples rings ripple drop pond"),
    "puddle": (_reg_at(puddle), "puddle wet street rain reflection"),
    "underwater": (_reg_region(underwater), "underwater sea floor diving ocean bottom caustics bubbles"),
    "droplets": (_reg_region(droplets), "droplets water drops condensation window glass rain on window wet"),
    "rain": (_reg_region(rain), "rain rainy rainfall downpour drizzle storm"),
    "wet ground": (_reg_region(wet_ground), "wet ground street reflection rainy pavement"),
    "caustics": (_reg_region(caustics), "caustics pool light water patterns"),
    # weather / sky
    "sky": (_reg_sky, "sky gradient sunset dawn noon night dusk"),
    "stars": (_reg_region(stars), "stars starry night sky milky way galaxy"),
    "moon": (_reg_at(moon), "moon crescent full moon lunar night"),
    "sun": (_reg_at(sun), "sun sunshine sunny daylight rays"),
    "shooting star": (_reg_region(shooting_star), "shooting star meteor falling star"),
    "aurora": (_reg_region(aurora), "aurora borealis northern lights polar"),
    "rainbow": (_reg_region(rainbow), "rainbow arc after rain spectrum"),
    "lightning": (_reg_lightning, "lightning bolt thunderstorm thunder strike"),
    "snow": (_reg_region(snowfall), "snow snowfall snowing winter flakes blizzard"),
    "snow cap": (_reg_region(snow_cap), "snow cap snow on roofs snowy edges"),
    "god rays": (_reg_at(god_rays), "god rays sunbeams crepuscular light rays through clouds"),
    "heat haze": (_reg_region(heat_haze), "heat haze mirage shimmer hot"),
    # light
    "glow": (_reg_at(glow), "glow light halo"),
    "lamp": (_reg_at(lamp), "lamp pendant light bulb ceiling"),
    "lantern": (_reg_at(lantern), "lantern paper lantern festival"),
    "neon": (_reg_points, "neon sign tube glow cyberpunk"),
    "window light": (_reg_box(window_light, 0.3, 0.4), "window light lit window night house warm"),
    "bokeh": (_reg_region(bokeh), "bokeh lights blur out of focus city lights"),
    "spotlight": (_reg_at(spotlight), "spotlight beam stage light cone"),
    "lens flare": (_reg_at(lens_flare), "lens flare camera"),
    "volumetric fog": (_reg_vfog(volumetric_fog), "volumetric fog light haze"),
}


def names():
    """All registry keywords (``element(cv, name, ...)``)."""
    return list(REGISTRY)


def find(text, n=5):
    """Best registry names for a free-text request, e.g. ``find("rain on a window")`` → ['droplets', 'rain', ...]."""
    words = [w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(w) > 1]
    scored = []
    for name, (_, kw) in REGISTRY.items():
        keys = kw.split()
        sc = 0.0
        for w in words:
            if w in name.split():
                sc += 3
            if w in keys:
                sc += 2 if keys.index(w) < 3 else 1
            elif any(k.startswith(w) or w.startswith(k) for k in keys if len(k) > 3 and len(w) > 3):
                sc += 0.5
        if " ".join(words) in kw:
            sc += 2
        for a, b in zip(words[:-1], words[1:]):
            if f"{a} {b}" in kw or f"{a} on {b}" in kw:
                sc += 3
        if sc:
            scored.append((sc, name))
    scored.sort(key=lambda t: (-t[0], names().index(t[1])))
    return [s for _, s in scored[:n]]


def element(cv, name, at=None, size=None, seed=0, **kw):
    """Draw any effect by keyword: ``element(cv, "campfire", at=(0.5, 0.8), size=240, seed=1)``.

    ``name`` may be a registry key or free text (the best :func:`find` match is used). ``at`` = (u, v)
    fractions, ``size`` = 1080p px; other keywords go to the underlying function. Returns its FX."""
    key = name if name in REGISTRY else (find(name, 1) or [None])[0]
    if key is None:
        raise _unknown("element", name, REGISTRY, "Try elements.find('...') or elements.names()")
    return REGISTRY[key][0](cv, at=at, size=size, seed=seed, **kw)


# ---- scene manifest: each top-level call records what it drew (pixelcanvas.manifest) ----
from . import manifest as _manifest  # noqa: E402

def _note(cv, kind, name, **kw):
    """Record one entry exactly as given (no merging with an earlier entry of a similar name: a sky and its stars
    are two things)."""
    raw = getattr(_manifest, "_note_raw", None)
    return raw(cv, kind, name, **kw) if raw else _manifest.note(cv, kind, name, **kw)


def _after_meta(kind, label, alias, meta):
    """``after`` hook: record the call as one entry carrying ``meta`` (backdrop / see_through ...)."""
    def after(cap, args, res):
        cv = cap.cv
        m = cap.mask() if cv is not None else None
        if m is None:
            return []
        _note(cv, kind, label, mask=m, extent=_manifest._extent_from_args(args, cv.w, cv.h),
              alias=list(alias) if alias else None, **meta)
        return []
    return after


def _register_meta(names, kind="effect", labels=None, alias=None, **meta):
    for n in names:
        fn = globals().get(n)
        if fn is None or getattr(fn, "__pc_manifest__", None):
            continue
        label = (labels or {}).get(n) or n.strip("_").replace("_", " ")
        al = (alias or {}).get(n)
        globals()[n] = _manifest.drawn(kind, name=label, after=_after_meta(kind, label, al, meta))(fn)


def _after_stars(cap, args, res):
    cv = cap.cv
    m = cap.mask() if cv is not None else None
    if m is None:
        return []
    _note(cv, "effect", "stars", mask=m, alias=["star field", "starfield"], backdrop=True)
    band = getattr(res, "band", None) if res is not None else None
    if args.get("milky_way") and band is not None and res.mask is not None:
        mw = clamp01((np.asarray(band, F32) - 0.12) * 4.0) * np.asarray(res.mask, F32)
        _note(cv, "effect", "Milky Way", mask=mw, alias=["stars", "night sky", "galaxy"], backdrop=True)
    return []


def _after_cumulus(cap, args, res):
    """One "cloud" entry boxed by what was painted: ``height`` is the cloud's height as a share of its width, not a
    canvas fraction, so the intended box is the painted cloud itself."""
    cv = cap.cv
    m = cap.mask() if cv is not None else None
    if m is None:
        return []
    _note(cv, "effect", "cloud", mask=m, alias=["clouds"])
    return []


_register_meta(["sky", "aurora"], backdrop=True)
globals()["stars"] = _manifest.drawn("effect", name="stars", after=_after_stars)(stars)
_register_meta(["ocean", "lake", "river_flow", "puddle", "ripples", "reflect"], see_through=True,
               labels={"river_flow": "river", "reflect": "reflection"},
               alias={"ocean": ["water", "sea", "waves"], "lake": ["water", "reflection", "reflections", "pond"],
                      "river_flow": ["water", "stream"], "puddle": ["water", "reflection"], "ripples": ["water"],
                      "reflect": ["reflections", "water", "lake"]})
_register_meta(["smoke", "steam", "chimney_smoke", "breath", "fog", "volumetric_fog", "rain", "snowfall", "droplets",
                "god_rays", "heat_haze", "dust_motes", "bokeh", "caustics", "spotlight", "lens_flare", "glow",
                "rainbow", "wet_ground"], see_through=True,
               alias={"fog": ["mist", "haze"], "volumetric_fog": ["mist", "haze", "fog"], "breath": ["mist", "vapour"],
                      "steam": ["vapour", "vapor", "mist"], "snowfall": ["snow"], "rain": ["rainfall"],
                      "smoke": ["plume"]})
_manifest.register(globals(), "effect", [
    "illuminate", "shooting_star", "lightning", "snow_cap", "storm_clouds", "cirrus", "cloud_bands",
    "flame", "campfire", "explosion", "sparks", "embers", "fireworks", "breaking_wave", "waterfall", "splash",
    "underwater", "neon", "window_light"],
    labels={"cumulus": "cloud", "illuminate": "lighting"},
    alias={"cumulus": "clouds", "cirrus": "clouds", "campfire": "fire", "flame": "fire"})
globals()["cumulus"] = _manifest.drawn("effect", name="cloud", after=_after_cumulus)(cumulus)
_manifest.register(globals(), "object", ["moon", "sun", "stone", "firewood", "candle", "torch", "fireplace",
                                         "lamp", "lantern"])
_manifest.register(globals(), "effect", ["element"], arg="name")


# ---- strict options: unknown keyword names raise TypeError naming the closest valid option; colour / color,
# ---- w / width ... are accepted (pixelcanvas._args). FX(**info) and element(**kw) forward / carry free-form extras.
for _n, _f in list(globals().items()):
    if (not _n.startswith("_") and callable(_f) and getattr(_f, "__module__", None) == __name__
            and not isinstance(_f, type)):
        globals()[_n] = _guard_fractions(_strict(_f, _n), _n)


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "fire": "use elements.flame(cv, ...), elements.campfire(cv, ...) or elements.torch(cv, ...)",
    "clouds": "use elements.cumulus(cv, ...) or elements.cloud_bands(cv, ...)",
    "cloud": "use elements.cumulus(cv, ...) or elements.cloud_bands(cv, ...)",
    "snow": "use elements.snowfall(cv, ...) for falling snow or elements.snow_cap(cv, mask) for snow lying on shapes",
    "sunset": "use elements.sky(cv, 'sunset') then elements.sun(cv, ...)",
    "night": "use elements.sky(cv, 'night') with elements.stars(cv, ...) and elements.moon(cv, ...)",
    "water": "use elements.lake(cv, ...), elements.ocean(cv, ...), elements.river_flow or elements.ripples",
    "sea": "use elements.ocean(cv, ...)",
    "waves": "use elements.ocean(cv, ...) or elements.breaking_wave(cv, ...)",
    "wave": "use elements.breaking_wave(cv, ...) or elements.ocean(cv, ...)",
    "star": "use elements.stars(cv, ...) for a star field or elements.shooting_star(cv, ...)",
    "glowing": "use elements.glow(cv, ...) or elements.neon(cv, ...)",
    "bloom": "use elements.glow(cv, ...) or pixelcanvas.filters.bloom(img)",
    "flare": "use elements.lens_flare(cv, ...)",
    "mist": "use elements.fog(cv, ...) or elements.volumetric_fog(cv, ...)",
    "haze": "use elements.fog(cv, ...) or elements.heat_haze(cv, ...)",
    "firework": "use elements.fireworks(cv, ...)",
    "spark": "use elements.sparks(cv, ...) or elements.embers(cv, ...)",
    "bubbles": "use elements.bokeh(cv, ...) or elements.droplets(cv, ...)",
    "raindrops": "use elements.rain(cv, ...) or elements.droplets(cv, ...)",
    "thunder": "use elements.lightning(cv, ...) or elements.storm_clouds(cv, ...)",
    "draw": "use elements.element(cv, 'campfire', **options) to draw any effect by name; elements.names() lists them",
    "effect": "use elements.element(cv, 'campfire', **options); elements.names() lists the effects, elements.find('words') searches",
    "list": "use elements.names() for every effect name, elements.find('words') to search",
}
__getattr__ = _module_getattr(__name__, _HINTS)
