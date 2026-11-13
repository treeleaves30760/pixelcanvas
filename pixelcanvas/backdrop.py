# -*- coding: utf-8 -*-
"""Abstract backdrops: one call paints a finished, palette-driven background and tells you where text will read.

    from pixelcanvas import Canvas
    from pixelcanvas.backdrop import backdrop
    cv = Canvas(1920, 1080)
    bd = backdrop(cv, "mesh", palette="dreamy", seed=3)      # or backdrop(cv, "soft pastel blobs")
    cv.text("Hello", box=bd.calm_px(cv), color=bd.ink)        # calm region + readable ink

Families: gradient, geometric, scenic, organic. ``names(family)``, ``find(description)``, ``describe(name)``.
Every backdrop is resolution independent (sizes in 1080p units of the short side), deterministic by ``seed``, takes
``palette=`` (name / description / Palette / colour list) and paints through ``cv.paint`` so it is one layer.
``quiet(cv, box)`` calms a busy area behind text without a visible box edge.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .color import F32, rgb, clamp01, smoothstep, grid
from . import noise as _n
from ._args import reject_unknown as _reject, unknown_name as _unknown_name, strict as _strict
from .filters import gaussian_blur, resize
from .palette import (Palette, palette as _palette, find as _pfind, _colour_words, srgb_to_oklab, oklab_to_srgb, mix_oklab,
                      contrast_ratio, ensure_contrast, query_tokens, to_hex)

__all__ = ["backdrop", "quiet", "names", "find", "describe", "FAMILIES", "BACKDROPS", "Backdrop", "render"]

_RES = getattr(Image, "Resampling", Image)


# ============================================================================================== context

class _Ctx:
    """Everything a backdrop generator needs: size, unit S (short side / 1080), palette, seed, options."""

    def __init__(self, h, w, P, seed, opts):
        self.h, self.w, self.P, self.seed, self.o = int(h), int(w), P, int(seed), opts
        self.S = min(self.h, self.w) / 1080.0
        self._xy = None

    def rng(self, salt=0):
        return np.random.default_rng((self.seed * 7919 + salt * 104729) & 0xFFFFFFFF)

    @property
    def X(self):
        if self._xy is None:
            self._xy = grid(self.h, self.w)
        return self._xy[0]

    @property
    def Y(self):
        if self._xy is None:
            self._xy = grid(self.h, self.w)
        return self._xy[1]

    @property
    def u(self):
        return (self.X + 0.5) / self.w

    @property
    def v(self):
        return (self.Y + 0.5) / self.h

    @property
    def aspect(self):
        return self.w / self.h

    def fbm(self, scale, octaves=4, seed=0, mode="plain", aspect=1.0, offset=(0.0, 0.0)):
        """fBm with the lowest-frequency cell ``scale`` in 1080p units."""
        s = max(scale * self.S, 1e-3)
        return _n.fbm2d(self.h, self.w, s, octaves, seed=self.seed * 31 + seed, mode=mode, aspect=aspect,
                        offset=(offset[0] * s, offset[1] * s))

    def fbm1(self, scale, octaves=4, seed=0, mode="plain"):
        return _n.fbm1d(self.w, max(scale * self.S, 1e-3), octaves, seed=self.seed * 31 + seed, mode=mode)

    def warp(self, scale, strength=1.0, octaves=4, seed=0, mode="plain", aspect=1.0):
        return _n.warp2d(self.h, self.w, max(scale * self.S, 1e-3), strength, octaves, seed=self.seed * 31 + seed,
                         aspect=aspect, mode=mode)

    def sized(self, h, w):
        return _Ctx(h, w, self.P, self.seed, self.o)

    def lowres(self, fn, target=300):
        """Compute a smooth field ``fn(ctx)`` at reduced resolution and scale it up (big speed-up)."""
        f = min(self.h, self.w) / float(target)
        if f <= 1.5:
            return fn(self)
        hh, ww = max(8, int(round(self.h / f))), max(8, int(round(self.w / f)))
        return resize(fn(self.sized(hh, ww)), self.w, self.h, "bicubic")

    def flat(self, col):
        return np.broadcast_to(_c(col), (self.h, self.w, 3)).astype(F32).copy()

    def lw(self, px):
        """A line width in 1080p units -> pixels, never thinner than 0.9 px."""
        return max(px * self.S, 0.9)


def _c(x):
    return np.asarray(rgb(x), F32)[:3]


def _mix(a, b, t):
    return mix_oklab(a, b, t)


def _ramp(cols, t, cyclic=False):
    """OKLab interpolation through ``cols`` at ``t`` (any shape) -> t.shape + (3,)."""
    lab = srgb_to_oklab(np.stack([_c(c) for c in cols])).astype(np.float64)
    if cyclic:
        lab = np.vstack([lab, lab[:1]])
        t = np.mod(t, 1.0)
    else:
        t = np.clip(t, 0, 1)
    x = np.linspace(0, 1, len(lab))
    t = np.asarray(t, np.float64)
    out = np.stack([np.interp(t, x, lab[:, k]) for k in range(3)], -1)
    return oklab_to_srgb(out)


def _vramp(c, cols, t1d):
    """A vertical gradient from a 1-D t (length h) -> (h, w, 3)."""
    col = _ramp(cols, t1d)
    return np.broadcast_to(col[:, None, :], (c.h, c.w, 3)).astype(F32).copy()


def _over(img, col, a):
    a = np.asarray(a, F32)
    if a.ndim == 2:
        a = a[..., None]
    col = np.asarray(col, F32)
    return (img + (col - img) * a).astype(F32)


def _screen(img, col, a):
    a = np.asarray(a, F32)
    if a.ndim == 2:
        a = a[..., None]
    return (1 - (1 - img) * (1 - np.asarray(col, F32) * a)).astype(F32)


def _blur(a, sigma):
    """Gaussian blur that stays fast for big sigmas (blurs a downscaled copy)."""
    if sigma <= 0:
        return a
    h, w = a.shape[:2]
    f = int(sigma // 6)
    if f >= 2 and h // f >= 8 and w // f >= 8:
        small = resize(a, w // f, h // f, "box")
        return resize(gaussian_blur(small, sigma / f), w, h, "bilinear")
    return gaussian_blur(a, sigma)


def _grain(c, img, amount=0.03, salt=5):
    n = c.rng(salt).normal(0, amount, (c.h, c.w)).astype(F32)
    return img + n[..., None]


def _aa(d):
    """Signed distance in px -> coverage (inside < 0)."""
    return np.clip(0.5 - d, 0, 1).astype(F32)


def _bands(p, widths, cols, px=1.0):
    """Repeating colour bands along ``p`` (units): ``widths`` per band, anti-aliased with ``px`` pixels per unit."""
    w = np.asarray(widths, np.float64)
    cum = np.concatenate([[0], np.cumsum(w)])
    period = cum[-1]
    pm = np.mod(p, period)
    idx = np.clip(np.searchsorted(cum, pm, "right") - 1, 0, len(w) - 1)
    tab = np.stack([_c(k) for k in cols])
    cur = tab[idx]
    prev = tab[(idx - 1) % len(w)]
    a = np.clip((pm - cum[idx]) * px + 0.5, 0, 1)[..., None]
    return (prev + (cur - prev) * a).astype(F32), idx


class _Pen:
    """Many anti-aliased flat shapes drawn with Pillow at ``ss``x supersampling -> (colour map, alpha)."""

    def __init__(self, c, ss=3):
        self.ss = ss
        W, H = c.w * ss, c.h * ss
        self.c = c
        self.im = Image.new("RGB", (W, H))
        self.al = Image.new("L", (W, H))
        self.d = ImageDraw.Draw(self.im)
        self.a = ImageDraw.Draw(self.al)

    def _p(self, pts):
        s = self.ss
        return [(float(x) * s, float(y) * s) for x, y in pts]

    @staticmethod
    def _col(col):
        return tuple(int(round(v * 255)) for v in np.clip(_c(col), 0, 1))

    def polygon(self, pts, col):
        p = self._p(pts)
        self.d.polygon(p, fill=self._col(col))
        self.a.polygon(p, fill=255)

    def ellipse(self, cx, cy, rx, ry, col, width=0):
        s = self.ss
        box = [(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s]
        if width:
            wd = max(1, int(round(width * s)))
            self.d.ellipse(box, outline=self._col(col), width=wd)
            self.a.ellipse(box, outline=255, width=wd)
        else:
            self.d.ellipse(box, fill=self._col(col))
            self.a.ellipse(box, fill=255)

    def line(self, pts, col, width, caps=True):
        p = self._p(pts)
        wd = max(1, int(round(width * self.ss)))
        self.d.line(p, fill=self._col(col), width=wd, joint="curve")
        self.a.line(p, fill=255, width=wd, joint="curve")
        if caps:
            r = width * self.ss / 2
            for x, y in (p[0], p[-1]):
                self.d.ellipse([x - r, y - r, x + r, y + r], fill=self._col(col))
                self.a.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def pieslice(self, cx, cy, r, start, end, col):
        s = self.ss
        box = [(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s]
        self.d.pieslice(box, start, end, fill=self._col(col))
        self.a.pieslice(box, start, end, fill=255)

    def arc(self, cx, cy, r, start, end, col, width):
        s = self.ss
        box = [(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s]
        wd = max(1, int(round(width * s)))
        self.d.arc(box, start, end, fill=self._col(col), width=wd)
        self.a.arc(box, start, end, fill=255, width=wd)

    def rect(self, x0, y0, x1, y1, col, radius=0):
        s = self.ss
        box = [x0 * s, y0 * s, x1 * s, y1 * s]
        if radius:
            self.d.rounded_rectangle(box, radius * s, fill=self._col(col))
            self.a.rounded_rectangle(box, radius * s, fill=255)
        else:
            self.d.rectangle(box, fill=self._col(col))
            self.a.rectangle(box, fill=255)

    def result(self):
        c = self.c
        im = self.im.resize((c.w, c.h), _RES.BOX) if self.ss > 1 else self.im
        al = self.al.resize((c.w, c.h), _RES.BOX) if self.ss > 1 else self.al
        a = np.asarray(al, F32) / 255.0
        col = np.asarray(im, F32) / 255.0 / np.maximum(a, 1e-4)[..., None]
        return np.clip(col, 0, 1).astype(F32), a

    def onto(self, img, opacity=1.0):
        col, a = self.result()
        return _over(img, col, a * opacity)


# ============================================================================================== registry

BACKDROPS = {}
FAMILIES = {"gradient": [], "geometric": [], "scenic": [], "organic": []}


def _bd(name, family, pal, words, desc, **defaults):
    def deco(fn):
        BACKDROPS[name] = dict(fn=fn, family=family, palette=pal, words=tuple(words.split()), desc=desc,
                               opts=defaults)
        FAMILIES[family].append(name)
        return fn
    return deco


# ============================================================================================== gradients

def _scatter_points(c, k, salt, margin=0.08):
    """k well-spread points (fractions), deterministic by seed."""
    rng = c.rng(salt)
    cols = int(math.ceil(math.sqrt(k * c.aspect)))
    rows = int(math.ceil(k / cols))
    cells = [(i, j) for j in range(rows) for i in range(cols)]
    rng.shuffle(cells)
    pts = []
    for i, j in cells[:k]:
        x = (i + rng.uniform(0.15, 0.85)) / cols
        y = (j + rng.uniform(0.15, 0.85)) / rows
        pts.append((margin + (1 - 2 * margin) * x, margin + (1 - 2 * margin) * y))
    return pts


def _mesh_field(c, cols, sigma=0.32, flow=1.0, salt=1):
    pts = _scatter_points(c, len(cols), salt, margin=0.0)
    u, v = c.u, c.v
    if flow:
        u = u + flow * 0.14 * c.fbm(700, 3, seed=11 + salt)
        v = v + flow * 0.14 * c.fbm(700, 3, seed=17 + salt)
    lab = srgb_to_oklab(np.stack([_c(k) for k in cols])).astype(np.float64)
    acc = np.zeros(u.shape + (3,))
    wsum = np.zeros(u.shape)
    a = c.aspect
    for (px, py), L in zip(pts, lab):
        d2 = ((u - px) * a) ** 2 + (v - py) ** 2
        wt = np.exp(-d2 / (2 * sigma * sigma)) + 1e-6
        acc += wt[..., None] * L
        wsum += wt
    return oklab_to_srgb(acc / wsum[..., None])


@_bd("mesh", "gradient", "dreamy", "mesh gradient soft smooth blend colour color points organic flow modern pastel abstract tasteful calm",
     "Smooth blend of several palette colours with a soft organic flow (the modern 'mesh gradient').",
     points=5, flow=1.0, grain=0.0)
def _mesh(c):
    P = c.P
    k = int(c.o["points"])
    pool = [P.bg] + [x for x in P.accents if x != P.bg] + P.colors
    cols = []
    for x in pool:
        if x not in cols and x != P.fg:
            cols.append(x)
    cols = (cols * 3)[:k]
    img = c.lowres(lambda cc: _mesh_field(cc, cols, flow=c.o["flow"]))
    if c.o["grain"]:
        img = _grain(c, img, c.o["grain"])
    return img


def _axis_t(c, angle):
    a = math.radians(angle)
    dx, dy = math.cos(a), math.sin(a)
    x, y = (c.u - 0.5) * c.aspect, c.v - 0.5
    p = x * dx + y * dy
    ext = abs(0.5 * c.aspect * dx) + abs(0.5 * dy)
    return (p / ext + 1) / 2


@_bd("linear", "gradient", "sunset", "linear gradient smooth straight banding free dithered",
     "Linear gradient through the palette (dark to light) in OKLab, dithered against banding.",
     angle=-60.0, colors=None)
def _linear(c):
    cols = c.o["colors"] or c.P.sorted()
    t = _axis_t(c, c.o["angle"])
    return _ramp(cols, smoothstep(0, 1, t) * 0.35 + t * 0.65)


@_bd("radial", "gradient", "night_city", "radial gradient glow centre center circle soft light",
     "Radial glow: accent at the centre fading through a mid tone to the background.",
     center=(0.32, 0.3), radius=1.0)
def _radial(c):
    P = c.P
    cx, cy = c.o["center"]
    d = np.hypot((c.u - cx) * c.aspect, c.v - cy) / (c.o["radius"] * max(c.aspect, 1))
    cols = [_mix(P.accent, P.light, 0.25), P.accent, _mix(P.accent, P.bg, 0.6), P.bg]
    return _ramp(cols, np.clip(d, 0, 1) ** 0.8)


@_bd("conic", "gradient", "candy", "conic angular sweep gradient colour wheel spin",
     "Conic (angular) sweep through the vivid colours around an off-centre point.",
     center=(0.62, 0.45), spin=20.0)
def _conic(c):
    P = c.P
    cx, cy = c.o["center"]
    ang = np.arctan2(c.v - cy, (c.u - cx) * c.aspect)
    t = (np.degrees(ang) + c.o["spin"]) / 360.0
    cols = [x for x in P.accents[:4]]
    if len(cols) < 3:
        cols = P.ramp(4)
    img = _ramp(cols, t, cyclic=True)
    r = np.hypot((c.u - cx) * c.aspect, c.v - cy)
    soft = np.exp(-(r / 0.07) ** 2)                               # soften the singular centre
    return _over(img, np.mean(np.stack([_c(x) for x in cols]), 0), soft * 0.6)


def _blobs(c, cols, n, salt, size=(0.28, 0.5), warp=0.12):
    """Large soft colour fields: list of (alpha (H, W), colour)."""
    rng = c.rng(salt)
    pts = _scatter_points(c, n, salt + 3, margin=-0.05)
    out = []
    u = c.u + warp * c.fbm(600, 3, seed=salt + 1)
    v = c.v + warp * c.fbm(600, 3, seed=salt + 2)
    for i, (px, py) in enumerate(pts):
        r = rng.uniform(*size)
        ang = rng.uniform(0, math.pi)
        sx = rng.uniform(1.0, 1.9)
        x, y = (u - px) * c.aspect, v - py
        xr = (x * math.cos(ang) + y * math.sin(ang)) / sx
        yr = -x * math.sin(ang) + y * math.cos(ang)
        a = np.exp(-(xr * xr + yr * yr) / (r * r) * 2.2)
        out.append((a.astype(F32), cols[i % len(cols)]))
    return out


@_bd("aurora", "gradient", "aurora", "aurora blobs blurred colour color fields glow soft modern ui background dark",
     "Large blurred colour fields drifting over the background (the 'aurora UI' look).",
     blobs=5, glow=0.9)
def _aurora(c):
    P = c.P

    def field(cc):
        img = cc.flat(P.bg)
        cols = P.accents[:4] or [P.accent]
        for a, col in _blobs(cc, cols, int(c.o["blobs"]), 3, size=(0.34, 0.6), warp=0.09):
            if P.mode == "dark":
                img = _screen(img, _c(col), a * c.o["glow"])
            else:
                img = _over(img, _c(col), a * c.o["glow"] * 0.8)
        return img
    return _grain(c, c.lowres(field), 0.012)


@_bd("grainy", "gradient", "energetic", "grainy noise gradient grain noisy textured gradient trendy dissolve",
     "Vivid gradient with a film-grain dissolve into a darker tone (the popular noise-textured gradient).",
     grain=0.06, shadow=0.55)
def _grainy(c):
    P = c.P
    cols = [x for x in P.accents[:3]] + [P.light]
    base = c.lowres(lambda cc: _mesh_field(cc, cols, sigma=0.36, flow=1.2, salt=4))
    sh = c.lowres(lambda cc: clamp01(0.5 + 0.9 * cc.fbm(900, 2, seed=8) + (cc.v - 0.5) * 0.9))
    rnd = c.rng(9).random((c.h, c.w)).astype(F32)
    k = clamp01((sh - rnd * 0.9 - 0.15) * 2.2) * c.o["shadow"]
    dark = _c(_mix(P.dark, P.accents[min(1, len(P.accents) - 1)], 0.25))
    img = _over(base, dark, k)
    return _grain(c, img, c.o["grain"])


@_bd("spotlight", "gradient", "moody", "spotlight stage vignette product showcase light cone dark studio",
     "A dark stage lit by a soft spotlight from above, with a floor pool of light (product shots, titles).",
     center=0.5, beam=0.32)
def _spotlight(c):
    P = c.P
    dark = _mix(P.dark, P.bg, 0.3) if P.mode == "dark" else _mix(P.dark, P.muted, 0.25)
    wall = _mix(dark, P.muted, 0.25)
    img = _vramp(c, [dark, wall, dark], (np.arange(c.h) + 0.5) / c.h)
    cx = c.o["center"]
    x = (c.u - cx) * c.aspect
    y = c.v + 0.15
    half = c.o["beam"] * (0.25 + y * 0.9)
    cone = smoothstep(1.0, 0.55, np.abs(x) / half) * smoothstep(1.25, 0.2, y)
    haze = 0.85 + 0.15 * c.fbm(300, 3, seed=2)
    light = _c(_mix(P.light, P.accent, 0.15))
    img = _over(img, light, cone * 0.38 * haze)
    pool = np.exp(-((x / (c.o["beam"] * 1.25)) ** 2 + ((c.v - 0.8) / 0.09) ** 2))
    img = _over(img, light, pool * 0.45)
    vig = clamp01(np.hypot((c.u - 0.5) * 1.2, c.v - 0.45) * 1.25)
    return _over(img, _c(P.dark), vig ** 2 * 0.6)


@_bd("duotone", "gradient", "pop_art", "duotone split two colour color halves bold diagonal",
     "Two bold colour fields split by a soft curve, with a shadow where one overlaps the other.",
     angle=-18.0, curve=0.06)
def _duotone(c):
    P = c.P
    a = math.radians(c.o["angle"])
    x, y = (c.u - 0.5) * c.aspect, c.v - 0.5
    along = x * math.cos(a) + y * math.sin(a)
    across = -x * math.sin(a) + y * math.cos(a)
    d = across - c.o["curve"] * np.sin(along * 4.2 + 0.6)
    B = _c(P.accents[1] if len(P.accents) > 1 else P.dark)
    dpx = d * c.h
    side = _aa(-dpx)                                              # 1 on the lower side
    ga = _ramp([_mix(P.accent, P.light, 0.18), P.accent], clamp01(c.v * 1.2))
    gb = _ramp([B, _mix(to_hex(B), P.dark, 0.25)], clamp01(c.v))
    img = _over(ga, gb, side)
    shadow = np.exp(-np.maximum(dpx, 0) / (40 * c.S)) * (dpx > -1)
    img = _over(img, _c(P.dark), shadow * 0.22 * (1 - side))
    return img


@_bd("glass", "gradient", "violet_dark", "glass frosted glassmorphism panel blur blobs card ui modern",
     "Frosted glass panels over vivid blurred blobs (glassmorphism): text goes on the big panel.",
     panels=2, blur=34.0)
def _glass(c):
    P = c.P

    def field(cc):
        img = _vramp(cc, [P.bg, _mix(P.bg, P.muted, 0.2)], (np.arange(cc.h) + 0.5) / cc.h)
        for a, col in _blobs(cc, P.accents[:4] or [P.accent], 5, 21, size=(0.18, 0.32), warp=0.06):
            img = _over(img, _c(col), clamp01(a * 1.6))
        return img
    img = c.lowres(field)
    boxes = [(0.12, 0.22, 0.6, 0.78, 34), (0.56, 0.12, 0.86, 0.44, 26)][:int(c.o["panels"])]
    blurred = _blur(img, c.o["blur"] * c.S)
    X, Y = c.X, c.Y
    for x0, y0, x1, y1, r in boxes:
        x0, x1, y0, y1 = x0 * c.w, x1 * c.w, y0 * c.h, y1 * c.h
        rr = r * c.S
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        qx, qy = np.abs(X - cx) - hw + rr, np.abs(Y - cy) - hh + rr
        d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rr
        sh = _blur(_aa(d - 0 * c.S).astype(F32), 28 * c.S)
        sh = np.roll(sh, int(round(16 * c.S)), 0)
        img = _over(img, _c(P.dark), sh * 0.22 * (1 - _aa(d)))
        tint = _c("#ffffff") if P.mode == "dark" else _c(P.bg)
        frost = _over(blurred, tint, 0.16 if P.mode == "dark" else 0.42)
        img = _over(img, frost, _aa(d))
        edge = clamp01(1 - np.abs(d + 0.8 * c.S) / max(1.1 * c.S, 0.7))
        lit = clamp01(0.75 - ((X - x0) / (2 * hw) + (Y - y0) / (2 * hh)) * 0.5)
        img = _over(img, _c("#ffffff"), edge * (0.18 + 0.5 * lit))
    return _grain(c, img, 0.008)


# ============================================================================================== geometric

def _fade_field(c, corner=(1.0, 1.0)):
    return clamp01(1 - np.hypot((c.u - corner[0]) * c.aspect, c.v - corner[1]) / (0.9 * max(c.aspect, 1)))


@_bd("dot_grid", "geometric", "slate", "dot grid dots pattern polka minimal subtle tech background",
     "Quiet grid of dots growing toward one corner, with a soft accent glow.",
     spacing=34.0, size=0.12, corner=(1.0, 1.0))
def _dot_grid(c):
    P = c.P
    sp = c.o["spacing"] * c.S
    img = _vramp(c, [P.bg, _mix(P.bg, P.muted, 0.08)], (np.arange(c.h) + 0.5) / c.h)
    glow = np.exp(-(((c.u - 0.15) * c.aspect) ** 2 + (c.v - 0.1) ** 2) / 0.12)
    img = _over(img, _c(P.accent), glow * 0.12)
    fx = np.mod(c.X - c.w / 2, sp) - sp / 2
    fy = np.mod(c.Y - c.h / 2, sp) - sp / 2
    f = _fade_field(c, c.o["corner"])
    r = sp * c.o["size"] * (0.55 + 1.5 * f)
    r = np.maximum(r, 0.6)
    dots = _aa(np.hypot(fx, fy) - r)
    ink = _c(_mix(P.bg, P.fg, 0.38))
    return _over(img, ink, dots * (0.6 + 0.4 * f))


@_bd("line_grid", "geometric", "indigo_dark", "line grid lines graph squares tech saas hero fade",
     "Fine square grid that fades out toward the edges, over a soft accent glow (tech hero sections).",
     spacing=64.0, major=4)
def _line_grid(c):
    P = c.P
    sp = c.o["spacing"] * c.S
    img = c.flat(P.bg)
    glow = np.exp(-(((c.u - 0.5) * c.aspect) ** 2 / 0.5 + (c.v + 0.05) ** 2 / 0.22))
    img = _over(img, _c(P.accent), glow * (0.35 if P.mode == "dark" else 0.18))
    lw = c.lw(1.2)

    def lines(p, s):
        return clamp01(lw / 2 + 0.5 - np.abs(np.mod(p + s / 2, s) - s / 2))
    minor = np.maximum(lines(c.X - c.w / 2, sp), lines(c.Y - c.h * 0.2, sp))
    M = sp * c.o["major"]
    major = np.maximum(lines(c.X - c.w / 2, M), lines(c.Y - c.h * 0.2, M))
    fade = clamp01(1.2 - np.hypot((c.u - 0.5) * 1.15, (c.v - 0.35) * 1.35) * 2.0)
    ink = _c(P.fg)
    img = _over(img, ink, minor * 0.07 * fade)
    return _over(img, ink, major * 0.13 * fade)


@_bd("isometric", "geometric", "nordic", "isometric cubes blocks tumbling 3d boxes pattern",
     "Isometric cubes (tumbling blocks) with three lit faces and a few accent cubes.",
     size=64.0, accents=0.12)
def _isometric(c):
    P = c.P
    a = c.o["size"] * c.S
    rng = c.rng(3)
    pen = _Pen(c)
    dx, dy = a * math.sqrt(3), a * 1.5
    base = [_mix(P.bg, P.muted, k) for k in (0.05, 0.35, 0.6)]
    rows = int(c.h / dy) + 3
    cols = int(c.w / dx) + 3
    for j in range(-1, rows):
        for i in range(-1, cols):
            cx = i * dx + (j % 2) * dx / 2
            cy = j * dy
            t = (cx / c.w + cy / c.h) / 2
            if rng.random() < c.o["accents"]:
                acc = P.accents[rng.integers(min(3, len(P.accents)))]
                faces = [_mix(acc, P.light, 0.35), acc, _mix(acc, P.dark, 0.35)]
            else:
                faces = [_mix(base[0], P.accent, 0.04 * t), _mix(base[1], P.accent, 0.05 * t), base[2]]
            top = (cx, cy - a)
            ul = (cx - dx / 2, cy - a / 2)
            ur = (cx + dx / 2, cy - a / 2)
            bl = (cx - dx / 2, cy + a / 2)
            br = (cx + dx / 2, cy + a / 2)
            bot = (cx, cy + a)
            ctr = (cx, cy)
            pen.polygon([top, ur, ctr, ul], faces[0])
            pen.polygon([ul, ctr, bot, bl], faces[1])
            pen.polygon([ctr, ur, br, bot], faces[2])
    return pen.onto(c.flat(P.bg))


@_bd("triangles", "geometric", "sunset", "triangles tessellation triangle mosaic geometric gradient facets",
     "Equilateral triangle tessellation shaded along a diagonal gradient with gentle per-tile variation.",
     size=110.0, angle=-35.0, jitter=0.06)
def _triangles(c):
    P = c.P
    s = c.o["size"] * c.S
    hgt = s * math.sqrt(3) / 2
    rng = c.rng(4)
    cols = P.sorted()
    pen = _Pen(c)
    a = math.radians(c.o["angle"])
    rows = int(c.h / hgt) + 2
    ncol = int(c.w / (s / 2)) + 4
    for j in range(-1, rows):
        y0, y1 = j * hgt, (j + 1) * hgt
        for i in range(-2, ncol):
            x = i * s / 2
            up = (i + j) % 2 == 0
            pts = [(x - s / 2, y1), (x + s / 2, y1), (x, y0)] if up else [(x - s / 2, y0), (x + s / 2, y0), (x, y1)]
            gx = sum(p[0] for p in pts) / 3 / c.w
            gy = sum(p[1] for p in pts) / 3 / c.h
            p = ((gx - 0.5) * c.aspect * math.cos(a) + (gy - 0.5) * math.sin(a))
            ext = abs(0.5 * c.aspect * math.cos(a)) + abs(0.5 * math.sin(a))
            t = (p / ext + 1) / 2 + rng.normal(0, c.o["jitter"])
            col = to_hex(_ramp(cols, np.array(t)))
            pen.polygon(pts, col)
    return pen.onto(c.flat(P.bg))


def _hex_cells(X, Y, R):
    """Pointy-top hex lattice of circumradius R -> (local x, local y, q, r)."""
    q = (math.sqrt(3) / 3 * X - Y / 3) / R
    r = (2 / 3 * Y) / R
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    m1 = (dq > dr) & (dq > ds)
    m2 = ~m1 & (dr > ds)
    rq = np.where(m1, -rr - rs, rq)
    rr = np.where(m2, -rq - rs, rr)
    cx = R * math.sqrt(3) * (rq + rr / 2)
    cy = R * 1.5 * rr
    return X - cx, Y - cy, rq.astype(np.int64), rr.astype(np.int64)


def _hash01(a, b, salt=0):
    h = (a * 73856093) ^ (b * 19349663) ^ (salt * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(F32) / 65535.0


@_bd("hexagons", "geometric", "teal_dark", "hexagons hexagon honeycomb hex grid tech science pattern",
     "Honeycomb of outlined hexagons with a scatter of filled cells drifting toward one side.",
     size=46.0, fill=0.22)
def _hexagons(c):
    P = c.P
    R = c.o["size"] * c.S
    lx, ly, q, r = _hex_cells(c.X, c.Y, R)
    ri = R * math.sqrt(3) / 2
    d = np.maximum(np.abs(lx), np.abs(lx) * 0.5 + np.abs(ly) * math.sqrt(3) / 2) - ri
    gap = max(3 * c.S, 1.0)
    rnd = _hash01(q, r, c.seed)
    rnd2 = _hash01(q, r, c.seed + 9)
    bias = clamp01(c.u * 1.3 - 0.1)
    filled = (rnd < c.o["fill"] * (0.3 + 1.4 * bias)).astype(F32)
    img = _vramp(c, [P.bg, _mix(P.bg, P.accent, 0.08)], (np.arange(c.h) + 0.5) / c.h)
    outline = clamp01(c.lw(1.3) / 2 + 0.5 - np.abs(d + gap))
    img = _over(img, _c(P.fg), outline * 0.12)
    cols = np.stack([_c(x) for x in (P.accents[:3] or [P.accent])])
    pick = cols[(rnd2 * len(cols)).astype(int) % len(cols)]
    shade = (0.55 + 0.45 * rnd2)[..., None]
    fillc = pick * shade + _c(P.bg) * (1 - shade)
    return _over(img, fillc, _aa(d + gap) * filled * (0.35 + 0.65 * bias))


@_bd("circles", "geometric", "mid_century", "concentric circles rings ripples target arcs retro",
     "Concentric rings sweeping from an off-canvas centre, stepping from accent to background.",
     center=(0.85, 0.95), step=52.0)
def _circles(c):
    P = c.P
    cx, cy = c.o["center"]
    d = np.hypot((c.u - cx) * c.w, (c.v - cy) * c.h)
    st = c.o["step"] * c.S
    far = math.hypot(max(cx, 1 - cx) * c.w, max(cy, 1 - cy) * c.h)
    n = int(far / st) + 2
    seq = [P.accent] + [x for x in P.accents[1:3]] + [P.muted]
    tab = []
    for k in range(n):
        t = k / max(n - 1, 1)
        base = to_hex(_ramp(seq, np.array(min(t * 1.6, 1.0))))
        tab.append(_mix(base, P.bg, min(1.0, t * 1.15) ** 0.9 * 0.85 + (0.06 if k % 2 else 0)))
    img, _ = _bands(d / st, [1.0] * n, tab, px=st)
    return img if d.max() < n * st else img


@_bd("sunburst", "geometric", "retro", "sunburst rays sun burst radiating retro vintage stripes",
     "Retro sunburst: alternating rays from a low centre with a warm glow.",
     rays=28, center=(0.5, 1.08))
def _sunburst(c):
    P = c.P
    cx, cy = c.o["center"]
    x, y = (c.u - cx) * c.w, (c.v - cy) * c.h
    ang = (np.arctan2(y, x) / (2 * math.pi)) % 1.0
    r = np.hypot(x, y)
    n = int(c.o["rays"]) * 2
    A = _mix(P.bg, P.accents[1] if len(P.accents) > 1 else P.accent, 0.18)
    B = _mix(P.bg, P.accent, 0.42)
    img, _ = _bands(ang * n, [1.0, 1.0], [A, B], px=r * 2 * math.pi / n)
    glow = np.exp(-(r / (0.75 * c.h)) ** 2)
    img = _over(img, _c(_mix(P.light, P.accent, 0.2)), glow * 0.55)
    return _over(img, _c(P.dark), clamp01(r / (1.6 * c.h)) ** 2 * 0.25)


@_bd("stripes", "geometric", "candy", "stripes diagonal stripe bands candy pattern bold",
     "Bold diagonal stripes in a rhythm of wide and thin bands.",
     angle=-35.0, width=70.0)
def _stripes(c):
    P = c.P
    a = math.radians(c.o["angle"])
    p = (c.X * math.cos(a) + c.Y * math.sin(a)) / c.S
    w = c.o["width"]
    acc = [x for x in P.accents if contrast_ratio(x, P.fg) > 2.5] or P.accents
    acc = (acc * 3)[:3]
    widths = [w, w * 0.32, w * 0.6, w * 0.32, w, w * 0.32]
    cols = [acc[0], P.bg, acc[1], P.bg, acc[2], P.bg]
    img, _ = _bands(p, widths, cols, px=c.S)
    return img


@_bd("chevrons", "geometric", "terracotta", "chevrons chevron zigzag herringbone pattern bands",
     "Zigzag chevron bands in warm tones.",
     width=60.0, period=180.0)
def _chevrons(c):
    P = c.P
    per = c.o["period"] * c.S
    tri = np.abs(np.mod(c.X / per, 1.0) - 0.5) * 2
    p = (c.Y + tri * per * 0.45) / c.S
    w = c.o["width"]
    srt = P.sorted()
    cols = [P.bg, srt[len(srt) // 2], P.bg, P.accent, P.bg, _mix(P.muted, P.bg, 0.3)]
    img, _ = _bands(p, [w * 0.9, w * 0.5, w * 0.9, w * 0.5, w * 0.9, w * 0.5], cols, px=c.S)
    return img


@_bd("checker", "geometric", "mid_century", "checker checkerboard squares chess board grid pattern",
     "Soft two-tone checkerboard with a light falloff toward the corners.",
     size=90.0, angle=0.0)
def _checker(c):
    P = c.P
    s = c.o["size"] * c.S
    a = math.radians(c.o["angle"])
    x = (c.X - c.w / 2) * math.cos(a) + (c.Y - c.h / 2) * math.sin(a)
    y = -(c.X - c.w / 2) * math.sin(a) + (c.Y - c.h / 2) * math.cos(a)
    fx, fy = x / s, y / s
    par = ((np.floor(fx) + np.floor(fy)) % 2) * 2 - 1
    ex = np.abs(np.mod(fx + 0.5, 1) - 0.5) * s
    ey = np.abs(np.mod(fy + 0.5, 1) - 0.5) * s
    k = 0.5 + 0.5 * par * clamp01(np.minimum(ex, ey) * 2)
    A = _c(P.bg)
    B = _c(_mix(P.bg, P.accent, 0.3))
    img = _over(np.broadcast_to(A, (c.h, c.w, 3)).copy(), B, k)
    return _over(img, _c(P.dark), clamp01(np.hypot(c.u - 0.5, c.v - 0.5) * 1.6) ** 2.5 * 0.18)


@_bd("waves", "geometric", "ocean", "waves wave layered bands section divider sea flowing curves bottom",
     "Layered wave bands along the bottom (the classic section divider), calm gradient above.",
     layers=5, height=0.45)
def _waves(c):
    P = c.P
    n = int(c.o["layers"])
    H = c.o["height"]
    img = _vramp(c, [P.bg, _mix(P.bg, P.muted, 0.25)], (np.arange(c.h) + 0.5) / c.h)
    x = (np.arange(c.w) + 0.5) / c.w * c.aspect
    rng = c.rng(6)
    cols = [_mix(P.muted, P.bg, 0.45), P.muted, P.accent, _mix(P.accent, P.dark, 0.45), P.dark]
    Y = c.Y
    for i in range(n):
        f = i / max(n - 1, 1)
        base = 1 - H + H * (0.08 + 0.8 * f)
        amp = 0.03 + 0.015 * (1 - f)
        ph = rng.uniform(0, 6.3, 2)
        fr = rng.uniform(0.55, 0.9)
        yc = (base + amp * np.sin(x * fr * math.pi + ph[0]) + amp * 0.35 * np.sin(x * fr * 2.1 * math.pi + ph[1])) * c.h
        col = to_hex(_ramp(cols, np.array(f)))
        m = clamp01(Y - yc[None, :] + 0.5)
        sh = np.exp(-np.maximum(yc[None, :] - Y, 0) / (26 * c.S)) * (1 - m)
        img = _over(img, _c(P.dark), sh * 0.12)
        body = _ramp([col, _mix(col, P.dark, 0.18)], clamp01((Y - yc[None, :]) / (0.4 * c.h)))
        img = _over(img, body, m)
    return img


@_bd("topographic", "geometric", "earthy", "topographic contour lines topo map elevation terrain outdoor",
     "Topographic contour lines over a gentle elevation tint; every fifth line is an index line.",
     lines=14, scale=760.0)
def _topographic(c):
    P = c.P

    def height(cc):
        return cc.warp(c.o["scale"], 0.8, 3, seed=3) * 0.5 + 0.5
    h = c.lowres(height, target=360)
    img = _ramp([P.bg, _mix(P.bg, P.muted, 0.35)], clamp01((h - 0.25) * 1.6))
    f = h * c.o["lines"]
    gy, gx = np.gradient(f)
    g = np.hypot(gx, gy) + 1e-6
    dist = np.abs(np.mod(f + 0.5, 1) - 0.5) / g
    line = clamp01(c.lw(1.1) / 2 + 0.5 - dist)
    idx = (np.round(f).astype(int) % 5 == 0)
    bold = clamp01(c.lw(2.2) / 2 + 0.5 - dist) * idx
    ink = _c(_mix(P.muted, P.fg, 0.35))
    img = _over(img, ink, line * 0.45)
    return _over(img, _c(P.accent if P.mode == "dark" else _mix(P.accent, P.fg, 0.2)), bold * 0.75)


@_bd("ribbons", "geometric", "synthwave", "ribbons flowing lines curves waves silk motion lines",
     "Two flowing ribbons made of fine parallel lines that twist, glowing on a dark ground.",
     lines=26, ribbons=2)
def _ribbons(c):
    P = c.P
    lightm = P.mode == "light"
    dark = P.bg if not lightm else P.light
    img = _vramp(c, [dark, _mix(dark, P.muted, 0.12 if lightm else 0.15)], (np.arange(c.h) + 0.5) / c.h)
    X, Y = c.X, c.Y
    x = (np.arange(c.w) + 0.5) / c.w
    rng = c.rng(7)
    n = int(c.o["lines"])
    acc = P.accents[:3] if len(P.accents) >= 2 else [P.accent, P.light]
    ends = P.dark if lightm else P.light
    total = np.zeros((c.h, c.w), F32)
    colmap = np.zeros((c.h, c.w, 3), F32)
    for k in range(int(c.o["ribbons"])):
        ph = rng.uniform(0, 6.3, 4)
        yc = (0.5 + (k - 0.5) * 0.18 + 0.16 * np.sin(x * 3.1 + ph[0]) + 0.06 * np.sin(x * 7.3 + ph[1])) * c.h
        wid = (0.22 + 0.08 * np.sin(x * 2.2 + ph[2])) * c.h
        tw = np.cos(x * (2.6 + k) + ph[3])
        half = wid * (0.12 + 0.88 * np.abs(tw)) / 2
        s = (Y - yc[None, :]) / half[None, :]
        inside = clamp01((1 - np.abs(s)) * half[None, :] / 2)
        f = (s + 1) / 2 * (n - 1)
        spacing = (2 * half / (n - 1))[None, :]
        dist = np.abs(np.mod(f + 0.5, 1) - 0.5) * spacing
        lw = np.minimum(c.lw(1.6), 0.5 * spacing + 0.4)
        fade = smoothstep(2.4, 5.0, spacing)              # lines that would alias fuse into a soft silk sheet
        ln = (clamp01(lw / 2 + 0.5 - dist) * fade + (1 - fade) * 0.62) * inside
        ln = np.maximum(ln, inside * 0.14)
        grad = _ramp([acc[k % len(acc)], _mix(acc[(k + 1) % len(acc)], ends, 0.2), ends],
                     np.broadcast_to(x[None, :] * 0.9 + 0.05 * k, ln.shape))
        colmap = colmap * (1 - ln[..., None]) + grad * ln[..., None]
        total = np.maximum(total, ln)
    if lightm:                                                    # a soft contact shadow instead of a glow
        sh = _blur(total[..., None] * np.ones(3, F32), 16 * c.S)
        img = img * (1 - 0.16 * sh)
    else:
        img = img + _blur(colmap * total[..., None], 14 * c.S) * 1.2
    return _over(img, colmap, total)


@_bd("low_poly", "geometric", "ocean", "low poly polygon facets triangles crystal geometric faceted",
     "Low-poly facets: jittered triangles coloured by a gradient and lit from the upper left.",
     cells=11, jitter=0.38)
def _low_poly(c):
    P = c.P
    rng = c.rng(8)
    rows = int(c.o["cells"])
    cols_n = max(2, int(round(rows * c.aspect)))
    gx = np.linspace(0, c.w, cols_n + 1)
    gy = np.linspace(0, c.h, rows + 1)
    jx = c.w / cols_n * c.o["jitter"]
    jy = c.h / rows * c.o["jitter"]
    PX = gx[None, :] + rng.uniform(-jx, jx, (rows + 1, cols_n + 1))
    PY = gy[:, None] + rng.uniform(-jy, jy, (rows + 1, cols_n + 1))
    PX[:, 0], PX[:, -1], PY[0, :], PY[-1, :] = -2, c.w + 2, -2, c.h + 2
    Z = rng.uniform(0, 1, PX.shape) * c.h * 0.12
    seq = P.sorted()[1:] if len(P) > 4 else P.sorted()
    light = np.array([-0.5, -0.6, 0.62])
    light /= np.linalg.norm(light)
    pen = _Pen(c, ss=2)
    for j in range(rows):
        for i in range(cols_n):
            a, b = (j, i), (j, i + 1)
            d, e = (j + 1, i), (j + 1, i + 1)
            tris = [(a, b, e), (a, e, d)] if rng.random() < 0.5 else [(a, b, d), (b, e, d)]
            for tri in tris:
                pts = np.array([[PX[k], PY[k], Z[k]] for k in tri])
                nrm = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                nrm /= (np.linalg.norm(nrm) + 1e-9)
                if nrm[2] < 0:
                    nrm = -nrm
                lam = float(np.dot(nrm, light))
                cx, cy = pts[:, 0].mean() / c.w, pts[:, 1].mean() / c.h
                t = 1 - (cx * 0.45 + (1 - cy) * 0.55)
                base = _ramp(seq, np.array(t))
                lab = srgb_to_oklab(base)
                lab[0] += (lam - 0.75) * 0.35
                pen.polygon([(p[0], p[1]) for p in pts], to_hex(oklab_to_srgb(lab)))
    return pen.onto(c.flat(P.bg))


@_bd("voronoi", "geometric", "autumn", "voronoi mosaic cells stained glass tiles organic cells",
     "Voronoi mosaic: cells coloured along a gradient, separated by light grout lines.",
     cells=9.0, gap=0.05)
def _voronoi(c):
    from .texture import cellular
    P = c.P
    k = c.o["cells"] / c.h
    cl = cellular(c.X * k, c.Y * k, 0.9, seed=c.seed + 2)
    t = (cl.cx / k / c.w) * 0.55 + (1 - cl.cy / k / c.h) * 0.45
    seq = [x for x in P.colors if x != P.bg]
    col = _ramp(seq, clamp01(t + (cl.rand - 0.5) * 0.18))
    lab = srgb_to_oklab(col)
    lab[..., 0] += (cl.r(3) - 0.5) * 0.06 - cl.f1 * 0.08
    col = oklab_to_srgb(lab)
    edge_px = cl.edge / k
    gap = c.o["gap"] / k
    m = clamp01(edge_px - gap / 2 + 0.5)
    return _over(c.flat(_mix(P.bg, "#ffffff", 0.3)), col, m)


def _clear_ok(c, x, y, clear):
    if not clear:
        return True
    return ((x / c.w - 0.5) * c.aspect / (clear * c.aspect * 0.75)) ** 2 + ((y / c.h - 0.5) / clear) ** 2 > 1


@_bd("memphis", "geometric", "memphis", "memphis confetti shapes squiggles 80s playful party fun pattern",
     "Memphis-style confetti: squiggles, triangles, rings, dots and zigzags, with a clear centre for text.",
     density=1.0, clear=0.32)
def _memphis(c):
    P = c.P
    rng = c.rng(10)
    pen = _Pen(c)
    S = c.S
    acc = [x for x in P.accents if contrast_ratio(x, P.bg) > 1.3][:4] or [P.accent]
    ink = P.fg
    step = 150 * S / math.sqrt(c.o["density"])
    for gy in np.arange(step / 2, c.h + step, step):
        for gx in np.arange(step / 2, c.w + step, step):
            x = gx + rng.uniform(-0.4, 0.4) * step
            y = gy + rng.uniform(-0.4, 0.4) * step
            if not _clear_ok(c, x, y, c.o["clear"]) or rng.random() < 0.18:
                continue
            col = acc[rng.integers(len(acc))]
            sz = rng.uniform(30, 58) * S
            kind = rng.integers(8)
            rot = rng.uniform(0, 2 * math.pi)
            if kind == 0:                                           # triangle
                pts = [(x + sz * math.cos(rot + k * 2.094), y + sz * math.sin(rot + k * 2.094)) for k in range(3)]
                pen.polygon(pts, col)
            elif kind == 1:                                         # ring
                pen.ellipse(x, y, sz * 0.7, sz * 0.7, col, width=7 * S)
            elif kind == 2:                                         # squiggle
                tt = np.linspace(-1, 1, 40)
                pts = [(x + t * sz * 1.4 * math.cos(rot) - math.sin(t * 9) * sz * 0.22 * math.sin(rot),
                        y + t * sz * 1.4 * math.sin(rot) + math.sin(t * 9) * sz * 0.22 * math.cos(rot)) for t in tt]
                pen.line(pts, col, 6 * S)
            elif kind == 3:                                         # zigzag
                pts = [(x + (k - 2.5) * sz * 0.45 * math.cos(rot) - (k % 2 - 0.5) * sz * 0.5 * math.sin(rot),
                        y + (k - 2.5) * sz * 0.45 * math.sin(rot) + (k % 2 - 0.5) * sz * 0.5 * math.cos(rot))
                       for k in range(6)]
                pen.line(pts, ink, 4.5 * S)
            elif kind == 4:                                         # dot cluster
                for a in range(3):
                    for b in range(3):
                        pen.ellipse(x + (a - 1) * sz * 0.5, y + (b - 1) * sz * 0.5, 3.6 * S, 3.6 * S, ink)
            elif kind == 5:                                         # half disc
                pen.pieslice(x, y, sz * 0.8, math.degrees(rot), math.degrees(rot) + 180, col)
            elif kind == 6:                                         # plus
                for a in (0, math.pi / 2):
                    pen.line([(x - sz * 0.5 * math.cos(rot + a), y - sz * 0.5 * math.sin(rot + a)),
                              (x + sz * 0.5 * math.cos(rot + a), y + sz * 0.5 * math.sin(rot + a))], col, 7 * S)
            else:                                                   # filled circle with offset ink ring
                pen.ellipse(x, y, sz * 0.45, sz * 0.45, col)
                pen.ellipse(x + 5 * S, y + 5 * S, sz * 0.45, sz * 0.45, ink, width=2.5 * S)
    img = c.flat(P.bg)
    return pen.onto(img)


@_bd("bauhaus", "geometric", "bauhaus", "bauhaus geometric composition circles quarter arcs bars grid modernist poster",
     "Bauhaus composition on a grid: half and quarter discs, circles and bars in primary colours.",
     rows=4, empty=0.12)
def _bauhaus(c):
    P = c.P
    rng = c.rng(11)
    rows = int(c.o["rows"])
    cs = c.h / rows
    cols_n = int(math.ceil(c.w / cs))
    ox = (c.w - cols_n * cs) / 2
    pen = _Pen(c)
    pal = [x for x in P.colors if x != P.bg]
    prev_row = [None] * cols_n
    for j in range(rows):
        left = None
        for i in range(cols_n):
            x0, y0 = ox + i * cs, j * cs
            choices = [k for k in pal if k not in (left, prev_row[i])]
            bgc = P.bg if rng.random() < 0.45 else choices[rng.integers(len(choices))]
            fgc = [k for k in pal + [P.bg] if k != bgc][rng.integers(len(pal))]
            pen.rect(x0, y0, x0 + cs + 1, y0 + cs + 1, bgc)
            left, prev_row[i] = bgc, bgc
            m = rng.integers(7)
            if rng.random() < c.o["empty"]:
                continue
            if m == 0:
                pen.ellipse(x0 + cs / 2, y0 + cs / 2, cs * 0.38, cs * 0.38, fgc)
            elif m == 1:
                corner = rng.integers(4)
                cx, cy = x0 + cs * (corner % 2), y0 + cs * (corner // 2)
                st = [270, 180, 0, 90][corner]
                st = {0: 0, 1: 90, 2: 270, 3: 180}[corner]
                pen.pieslice(cx, cy, cs, st, st + 90, fgc)
            elif m == 2:
                side = rng.integers(4)
                cx, cy = x0 + cs / 2, y0 + cs / 2
                if side == 0:
                    pen.pieslice(cx, y0 + cs, cs / 2, 180, 360, fgc)
                elif side == 1:
                    pen.pieslice(cx, y0, cs / 2, 0, 180, fgc)
                elif side == 2:
                    pen.pieslice(x0, cy, cs / 2, 270, 450, fgc)
                else:
                    pen.pieslice(x0 + cs, cy, cs / 2, 90, 270, fgc)
            elif m == 3:
                for k in range(3):
                    yy = y0 + cs * (0.2 + 0.25 * k)
                    pen.rect(x0 + cs * 0.15, yy, x0 + cs * 0.85, yy + cs * 0.12, fgc)
            elif m == 4:
                pen.ellipse(x0 + cs / 2, y0 + cs / 2, cs * 0.42, cs * 0.42, fgc)
                ic = [k for k in pal + [P.bg] if k not in (fgc,)][rng.integers(len(pal))]
                pen.ellipse(x0 + cs / 2, y0 + cs / 2, cs * 0.17, cs * 0.17, ic)
            elif m == 5:
                pen.polygon([(x0, y0 + cs), (x0 + cs, y0 + cs), (x0 + cs, y0)], fgc)
            else:
                pen.rect(x0 + cs * 0.42, y0, x0 + cs * 0.58, y0 + cs, fgc)
    img = pen.onto(c.flat(P.bg))
    return _grain(c, img, 0.012)


@_bd("art_deco", "geometric", "art_deco", "art deco fans scallop gatsby 1920s gold pattern ornamental",
     "Art-deco fans: overlapping scallops with gold rims, arcs and fine rays on a deep ground.",
     size=130.0)
def _art_deco(c):
    P = c.P
    R = c.o["size"] * c.S
    gold = _c(P.accent)
    gold2 = _c(P.light if P.mode == "dark" else P.accents[1] if len(P.accents) > 1 else P.accent)
    img = c.flat(P.bg)
    X = c.X
    lw = c.lw(2.0)
    rows = int(c.h / (R / 2)) + 3
    for j in range(-1, rows):
        cy = j * R / 2
        y0, y1 = int(max(cy - R - 2, 0)), int(min(cy + R + 2, c.h))
        if y0 >= y1:
            continue
        off = (j % 2) * R
        Xs, Ys = X[y0:y1], c.Y[y0:y1]
        i = np.round((Xs - off) / (2 * R))
        dx = Xs - (i * 2 * R + off)
        dy = Ys - cy
        r = np.hypot(dx, dy)
        inside = _aa(r - R)
        t = r / R
        fill = _ramp([_mix(P.bg, P.muted, 0.6), P.bg], np.clip(t, 0, 1))
        ang = np.arctan2(dy, dx)
        rays = np.abs(np.mod(ang / (math.pi / 12) + 0.5, 1) - 0.5) * (math.pi / 12) * r
        ray_m = clamp01(lw * 0.35 + 0.5 - rays) * ((t > 0.36) & (t < 0.62))
        arcs = np.minimum.reduce([np.abs(r - R * k) for k in (0.36, 0.62, 0.8)])
        arc_m = clamp01(lw * 0.5 + 0.5 - arcs)
        rim = clamp01(lw + 0.5 - np.abs(r - (R - lw * 1.2)))
        sub = _over(fill, gold * 0.75, ray_m * 0.7)
        sub = _over(sub, gold, arc_m * 0.8)
        sub = _over(sub, gold2, rim)
        img[y0:y1] = _over(img[y0:y1], sub, inside)
    return img


@_bd("truchet", "geometric", "risograph", "truchet tiles maze arcs curves pattern generative",
     "Truchet tiles: quarter-circle arcs that join into endless meanders, with a soft drop shadow.",
     size=84.0, width=0.2)
def _truchet(c):
    P = c.P
    s = c.o["size"] * c.S
    fx, fy = c.X / s, c.Y / s
    ix, iy = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    lx, ly = fx - ix, fy - iy
    flip = _hash01(ix, iy, c.seed) < 0.5
    lx = np.where(flip, 1 - lx, lx)
    d1 = np.abs(np.hypot(lx, ly) - 0.5)
    d2 = np.abs(np.hypot(1 - lx, 1 - ly) - 0.5)
    d = np.minimum(d1, d2) * s
    hw = c.o["width"] * s / 2
    m = _aa(d - hw)
    sh = np.roll(np.roll(_blur(m, 4 * c.S), int(round(5 * c.S)), 0), int(round(3 * c.S)), 1)
    img = c.flat(P.bg)
    img = _over(img, _c(P.dark), sh * 0.18)
    acc = P.accents[:3] if len(P.accents) >= 2 else [P.accent, P.muted]
    col = _ramp(acc, clamp01(c.u * 0.7 + c.v * 0.3))
    return _over(img, col, m)


@_bd("halftone", "geometric", "pop_art", "halftone dots fade pop art comic print dot gradient",
     "Halftone dot field fading from large to small across the canvas (pop-art print).",
     spacing=24.0, angle=45.0, corner=(1.0, 1.0))
def _halftone(c):
    P = c.P
    sp = c.o["spacing"] * c.S
    a = math.radians(c.o["angle"])
    x = c.X * math.cos(a) + c.Y * math.sin(a)
    y = -c.X * math.sin(a) + c.Y * math.cos(a)
    fx = np.mod(x, sp) - sp / 2
    fy = np.mod(y, sp) - sp / 2
    cov = clamp01(_fade_field(c, c.o["corner"]) * 1.25 - 0.05)
    r = sp * 0.62 * np.sqrt(cov)
    dots = _aa(np.hypot(fx, fy) - r)
    img = c.flat(P.bg)
    return _over(img, _c(P.accent), dots)


@_bd("circuit", "geometric", "cyber", "circuit board traces pcb electronics tech chip hardware",
     "Circuit traces with 45-degree bends ending in pads, glowing softly on a dark board.",
     density=1.0)
def _circuit(c):
    P = c.P
    rng = c.rng(12)
    g = 34 * c.S
    nx, ny = int(c.w / g) + 2, int(c.h / g) + 2
    occ = np.zeros((ny, nx), bool)
    dark = P.bg if P.mode == "dark" else _mix(P.dark, P.accent, 0.12)
    img = _vramp(c, [dark, _mix(dark, P.accent, 0.08)], (np.arange(c.h) + 0.5) / c.h)
    pen = _Pen(c, ss=2)
    pads = _Pen(c, ss=3)
    dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    n = int(nx * ny * 0.09 * c.o["density"])
    acc = P.accents[:2] or [P.accent]
    for _ in range(n):
        x, y = int(rng.integers(nx)), int(rng.integers(ny))
        if occ[y, x]:
            continue
        di = int(rng.integers(0, 8, 1)[0] // 2 * 2)
        pts = [(x, y)]
        occ[y, x] = True
        for _s in range(int(rng.integers(3, 14))):
            if rng.random() < 0.25:
                di = (di + (1 if rng.random() < 0.5 else -1)) % 8
            dx, dy = dirs[di]
            x2, y2 = x + dx, y + dy
            if not (0 <= x2 < nx and 0 <= y2 < ny) or occ[y2, x2]:
                break
            x, y = x2, y2
            occ[y, x] = True
            pts.append((x, y))
        if len(pts) < 3:
            continue
        col = acc[int(rng.integers(len(acc)))]
        tone = _mix(col, dark, float(rng.uniform(0.25, 0.6)))
        P_ = [(px * g, py * g) for px, py in pts]
        pen.line(P_, tone, 2.6 * c.S, caps=False)
        for end in (P_[0], P_[-1]):
            pads.ellipse(end[0], end[1], 6.5 * c.S, 6.5 * c.S, tone, width=2.4 * c.S)
    col, a = pen.result()
    pcol, pa = pads.result()
    img = img + _blur(col * a[..., None], 8 * c.S) * 0.8
    img = _over(img, col, a)
    return _over(img, pcol, pa)


@_bd("blueprint", "geometric", "corporate", "blueprint grid engineering drawing technical plan architecture",
     "Blueprint sheet: blue ground, fine and major grid, faint construction circles and a title block.",
     spacing=24.0)
def _blueprint(c):
    P = c.P
    blue = _mix(P.accent, P.dark, 0.5)
    img = _vramp(c, [_mix(blue, "#ffffff", 0.04), blue], (np.arange(c.h) + 0.5) / c.h)
    img = img * (1 + 0.035 * c.fbm(200, 3, seed=4))[..., None]
    sp = c.o["spacing"] * c.S
    white = _c(_mix(P.light, "#ffffff", 0.6))

    def lines(p, s, w):
        return clamp01(w / 2 + 0.5 - np.abs(np.mod(p + s / 2, s) - s / 2))
    lw = c.lw(1.0)
    img = _over(img, white, np.maximum(lines(c.X, sp, lw), lines(c.Y, sp, lw)) * 0.1)
    img = _over(img, white, np.maximum(lines(c.X, sp * 5, lw * 1.3), lines(c.Y, sp * 5, lw * 1.3)) * 0.2)
    pen = _Pen(c, ss=2)
    S = c.S
    m = 40 * S
    pen.rect(m, m, c.w - m, c.h - m, "#ffffff")
    pen.rect(m + 3 * S, m + 3 * S, c.w - m - 3 * S, c.h - m - 3 * S, "#000000")
    cx, cy = c.w * 0.72, c.h * 0.46
    for r in (0.22, 0.14):
        pen.ellipse(cx, cy, r * c.h, r * c.h, "#ffffff", width=2 * S)
    pen.line([(cx - 0.3 * c.h, cy), (cx + 0.3 * c.h, cy)], "#ffffff", 1.4 * S, caps=False)
    pen.line([(cx, cy - 0.3 * c.h), (cx, cy + 0.3 * c.h)], "#ffffff", 1.4 * S, caps=False)
    y = cy + 0.3 * c.h
    pen.line([(cx - 0.22 * c.h, y), (cx + 0.22 * c.h, y)], "#ffffff", 1.4 * S, caps=False)
    for xx in (cx - 0.22 * c.h, cx + 0.22 * c.h):
        pen.line([(xx, y - 10 * S), (xx, y + 10 * S)], "#ffffff", 1.4 * S, caps=False)
    bx0, by0 = c.w - m - 300 * S, c.h - m - 110 * S
    pen.rect(bx0, by0, c.w - m, c.h - m, "#ffffff")
    pen.rect(bx0 + 2.5 * S, by0 + 2.5 * S, c.w - m - 2.5 * S, c.h - m - 2.5 * S, "#000000")
    pen.rect(bx0, by0 + 52 * S, c.w - m, by0 + 54.5 * S, "#ffffff")
    col, a = pen.result()
    wmask = a * (col.mean(-1) > 0.5)
    return _over(img, white, wmask * 0.55)


@_bd("origami", "geometric", "lavender", "origami folds folded paper creases facets paper art",
     "Folded paper: flat facets with crisp creases and soft light across each fold.",
     folds=6)
def _origami(c):
    P = c.P
    rng = c.rng(13)
    a2 = P.accents[1] if len(P.accents) > 1 else P.muted
    base = _ramp([_mix(P.bg, P.accent, 0.12), _mix(P.bg, a2, 0.45)], clamp01(c.u * 0.5 + c.v * 0.5))
    lab = srgb_to_oklab(base).astype(np.float64)
    shade = np.zeros((c.h, c.w))
    for k in range(int(c.o["folds"])):
        px, py = rng.uniform(0.1, 0.9) * c.w, rng.uniform(0.1, 0.9) * c.h
        ang = rng.uniform(0, math.pi)
        nx, ny = math.cos(ang), math.sin(ang)
        d = (c.X - px) * nx + (c.Y - py) * ny
        s = np.clip(d / 0.8, -1, 1)
        amp = rng.uniform(0.025, 0.06)
        L = rng.uniform(0.25, 0.6) * c.h
        shade += amp * s * (0.55 + 0.45 * np.exp(-np.abs(d) / L))
    lab[..., 0] += shade
    return oklab_to_srgb(lab)


_DEFAULT_ICONS = ["star", "heart", "cloud", "bolt", "bulb", "camera", "globe", "sparkles", "chart", "bell",
                  "clock", "code", "pin", "tag", "search", "gear"]


@_bd("icons", "geometric", "playful", "icons scattered icon pattern doodles symbols wallpaper pattern",
     "Scattered vector icons on a loose grid, tilted, mostly tone-on-tone with a few accents.",
     icons=None, size=54.0, clear=0.3)
def _icons(c):
    from .vector import Drawing
    from .canvas import Canvas
    P = c.P
    rng = c.rng(14)
    names_ = list(c.o["icons"] or _DEFAULT_ICONS)
    dwg = Drawing(c.w, c.h, background=False)
    step = c.o["size"] * c.S * 2.9
    tone = _mix(P.bg, P.fg, 0.22)
    acc = P.accents[:3] or [P.accent]
    j = 0
    for gy in np.arange(step / 2, c.h + step / 2, step * 0.866):
        off = (j % 2) * step / 2
        j += 1
        for gx in np.arange(step / 2 - off, c.w + step / 2, step):
            x = gx + rng.uniform(-0.18, 0.18) * step
            y = gy + rng.uniform(-0.18, 0.18) * step
            if not _clear_ok(c, x, y, c.o["clear"]):
                continue
            nm = names_[int(rng.integers(len(names_)))]
            col = acc[int(rng.integers(len(acc)))] if rng.random() < 0.18 else tone
            with dwg.group(rotate=float(rng.uniform(-18, 18)), origin=(float(x), float(y))):
                dwg.icon(nm, float(x), float(y), c.o["size"] * c.S, color=col, stroke_width=2.0)
    cv = Canvas(c.w, c.h, P.bg)
    dwg.draw(cv)
    return cv.buf


# ============================================================================================== scenic

def _scenic_colors(P):
    srt = P.sorted()
    mid = srt[len(srt) // 2]
    return dict(light=P.light, dark=P.dark, mid=mid, top=_mix(mid, P.light, 0.3), haze=P.light)


def _ridge_layer(c, img, top, col, haze=None, mist=0.0, shade=0.12):
    """Paint the area below the 1-D ridge ``top`` (px per column) with ``col`` (darker toward the bottom)."""
    Y = c.Y
    m = clamp01(Y - top[None, :] + 0.5)
    depth = clamp01((Y - top[None, :]) / (0.35 * c.h))
    body = _ramp([col, _mix(col, "#000000", shade)], depth)
    if haze is not None and mist:
        body = _over(body, _c(haze), (clamp01(depth * 2.5) ** 1.5 * mist)[..., None] * np.ones(3, F32))
    return _over(img, body, m)


def _sun(c, img, x, y, r, col, glow=0.5):
    d = np.hypot((c.u - x) * c.w, (c.v - y) * c.h)
    img = _screen(img, _c(col), np.exp(-(d / (r * 4.5)) ** 2) * glow * 1.6)
    return _over(img, _c(_mix(col, "#ffffff", 0.6)), _aa(d - r))


@_bd("hills", "scenic", "sunset", "hills rolling layered silhouettes landscape wallpaper atmospheric fade valley",
     "Layered hill silhouettes fading into haze under a soft sun (the classic wallpaper).",
     layers=5, sun=True, sun_x=0.68)
def _hills(c):
    P = c.P
    k = _scenic_colors(P)
    t = (np.arange(c.h) + 0.5) / c.h
    img = _vramp(c, [k["top"], _mix(k["top"], k["light"], 0.55), k["light"]], clamp01(t / 0.62))
    if c.o["sun"]:
        img = _sun(c, img, c.o["sun_x"], 0.42, 0.065 * c.h, _mix(k["light"], "#ffffff", 0.6), 0.35)
    n = int(c.o["layers"])
    far, near = _mix(k["mid"], k["light"], 0.55), k["dark"]
    for i in range(n):
        f = i / max(n - 1, 1)
        base = (0.5 + 0.38 * f ** 1.1) * c.h
        amp = (0.05 + 0.04 * f) * c.h
        rid = base - amp * (c.fbm1(900 - 400 * f, 3, seed=40 + i) * 1.4 + 0.25 * np.sin(np.arange(c.w) / c.w * 3 + i))
        col = to_hex(_ramp([far, k["mid"], near], np.array(f ** 0.85)))
        img = _ridge_layer(c, img, rid, col, haze=k["light"], mist=0.35 * (1 - f), shade=0.1)
    return img


def _glow_colour(P):
    """The warm light colour of a palette (peach, amber, rose ...) for horizon glow; its light colour if none is warm."""
    best, bs = None, 0.0
    for col in P.colors:
        L, C, h = (float(v) for v in srgb_to_oklab_ch(col))
        warm = max(0.0, math.cos(math.radians(h - 50)))
        sc = warm * min(C / 0.06, 1.0) * (1.0 if L > 0.55 else 0.4)
        if sc > bs and C > 0.03:
            best, bs = col, sc
    return best if best is not None and bs > 0.25 else P.light


def srgb_to_oklab_ch(col):
    lab = srgb_to_oklab(_c(col)).astype(np.float64)
    return lab[0], math.hypot(lab[1], lab[2]), math.degrees(math.atan2(lab[2], lab[1])) % 360


def _peak_ridge(c, rng, n_peaks, base, height, steep, rough, salt, sun_px):
    """A mountain range: top edge (px per column) with straight slopes meeting in sharp peaks and V valleys, and the
    lit amount (0 shaded ... 1 lit) of each column: the side of every peak that faces the sun is lit."""
    xs = (np.arange(c.w) + 0.5)
    px = (np.linspace(-0.12, 1.12, n_peaks) + (rng.random(n_peaks) - 0.5) * 1.1 / n_peaks) * c.w
    hp = height * (0.3 + 0.7 * rng.random(n_peaks) ** 1.3)
    hp[int(rng.integers(n_peaks))] = height * 1.05                    # one dominant peak per range
    sl = steep * (0.75 + 0.6 * rng.random(n_peaks))                   # px of height per px across
    cone = hp[:, None] - sl[:, None] * np.abs(xs[None, :] - px[:, None])
    top = np.maximum(cone.max(0), (0.2 + 0.1 * (c.fbm1(260, 2, seed=salt + 2) - 0.5)) * height)
    arg = cone.argmax(0)
    jit = (c.fbm1(40, 3, seed=salt + 5) - 0.5) * 36 * c.S            # the light / shadow split wanders a little
    dx = xs - px[arg] + jit
    lit = smoothstep(-2.0 * c.S, 2.0 * c.S, dx * np.sign(sun_px - px[arg] + 1e-3))
    top = top + rough * (c.fbm1(18, 3, seed=salt + 9) - 0.5) * height * 0.025
    lit = 0.5 + (lit - 0.5) * smoothstep(0.2 * height, 0.55 * height, top)     # no hard split down in the valleys
    top = top + rough * (c.fbm1(90, 4, seed=salt, mode="ridged") - 0.5) * height * 0.07
    return base - top, lit.astype(F32)


@_bd("mountains", "scenic", "winter", "mountains peaks alpine range sun layered landscape snow misty mist dawn fog",
     "Layered mountain ranges with sharp lit peaks, mist pooling between them and a low sun glowing at the horizon.",
     layers=4, sun_x=0.36, mist=0.55, sun_y=0.5)
def _mountains(c):
    P = c.P
    k = _scenic_colors(P)
    glow = _glow_colour(P)
    t = (np.arange(c.h) + 0.5) / c.h
    top_c = _mix(k["mid"], k["light"], 0.35)
    hz = c.o["sun_y"] + 0.12
    sky = _vramp(c, [top_c, _mix(top_c, k["light"], 0.6), _mix(k["light"], glow, 0.85)], clamp01(t / max(hz, 0.3)) ** 1.1)
    sx, sy = c.o["sun_x"], c.o["sun_y"]
    d = np.hypot((c.u - sx) * c.w, (c.v - sy) * c.h)
    sky = _screen(sky, _c(glow), np.exp(-(d / (0.42 * c.h)) ** 2) * 0.55)
    sky = _screen(sky, _c(_mix(glow, "#ffffff", 0.4)), np.exp(-(d / (0.09 * c.h)) ** 2) * 0.55)
    sky = _over(sky, _c(_mix(glow, "#ffffff", 0.55)), _aa(d - 0.05 * c.h))
    img = sky
    n = int(c.o["layers"])
    mist = float(c.o["mist"])
    rng = c.rng(60)
    far = _mix(_mix(k["mid"], k["light"], 0.5), glow, 0.3)
    near = _mix(k["dark"], k["mid"], 0.22)
    haze = _mix(k["light"], glow, 0.6)
    Y = c.Y
    xs = np.arange(c.w, dtype=np.float64)
    for i in range(n):
        f = i / max(n - 1, 1)
        base = (0.50 + 0.34 * f) * c.h + 0.1 * c.h
        height = (0.30 - 0.11 * f) * c.h
        rid, lit = _peak_ridge(c, rng, int(round(5 + 3 * f)), base, height, (0.55 + 0.2 * f) * (c.h / c.w) * 1.78,
                               0.5 + 0.5 * f, 60 + i * 3, sx * c.w)
        col = to_hex(_ramp([far, k["mid"], near], np.array(f ** 0.85)))
        depth = clamp01((Y - rid[None, :]) / (height * 1.25))
        lite = _mix(col, glow, 0.34 - 0.22 * f)
        shd = _mix(col, k["dark"], 0.22 * (0.5 + 0.5 * f))
        face = _ramp([shd, lite], 0.5 + (np.broadcast_to(lit[None, :], depth.shape) - 0.5) * (1 - smoothstep(0.15, 0.75, depth)) ** 1.2)
        pool = _mix(haze, col, 0.25 * f)
        body = _over(face, _c(pool), (clamp01(depth * 1.1) ** 1.3 * mist * (1 - 0.8 * f))[..., None])
        img = _over(img, body, clamp01(Y - rid[None, :] + 0.5))
        if i == 0 and k["light"] != P.dark:                           # a dusting of snow on the farthest range
            line = np.percentile(rid, 22)
            snow = clamp01((line - c.Y + 0.02 * c.h * c.fbm(70, 3, seed=5)) / 1.5) * clamp01(c.Y - rid[None, :] + 0.5)
            img = _over(img, _c(_mix(k["light"], "#ffffff", 0.6)), snow * 0.55)
    return img


@_bd("dunes", "scenic", "desert", "dunes sand desert sahara dune ridges sunlit warm",
     "Sand dunes with sunlit and shaded faces under a warm sky.",
     layers=4)
def _dunes(c):
    P = c.P
    k = _scenic_colors(P)
    t = (np.arange(c.h) + 0.5) / c.h
    img = _vramp(c, [_mix(k["top"], k["light"], 0.2), k["light"]], clamp01(t / 0.5))
    img = _sun(c, img, 0.78, 0.22, 0.045 * c.h, _mix(k["light"], "#ffffff", 0.7), 0.3)
    n = int(c.o["layers"])
    Y = c.Y
    xs = np.arange(c.w, dtype=F32)
    rng = c.rng(15)
    for i in range(n):
        f = i / max(n - 1, 1)
        base = (0.48 + 0.4 * f) * c.h
        ph = rng.uniform(0, 6.3, 2)
        wl = (1.3 + 0.6 * rng.random()) * math.pi
        u = xs / c.w * wl + ph[0]
        rid = base - (0.06 + 0.03 * f) * c.h * (np.sin(u) + 0.35 * np.sin(2 * u + ph[1]))
        slope = np.gradient(rid)
        lit = 0.5 - 0.5 * np.tanh(slope / max(np.abs(slope).max(), 1e-6) * 1.8)                       # smooth sunlit / shaded faces, no clipping
        col = to_hex(_ramp([_mix(k["mid"], k["light"], 0.5), k["mid"], _mix(k["dark"], k["mid"], 0.3)],
                           np.array(f)))
        litc = _mix(col, k["light"], 0.35)
        shc = _mix(col, k["dark"], 0.35)
        dep = np.maximum(Y - rid[None, :], 0.0)
        depth = clamp01(dep / (0.12 * c.h))
        # the light falls at a slant: each face is read from further along the ridge as it goes down, so the
        # shading drifts diagonally instead of standing in straight columns
        src = np.clip((xs[None, :] + 0.45 * dep).astype(np.int32), 0, c.w - 1)
        litd = lit[src]
        face = litd * (1 - depth * 0.6) + 0.5 * depth * 0.6
        body = _ramp([shc, col, litc], face)
        m = clamp01(Y - rid[None, :] + 0.5)
        img = _over(img, body, m)
    return img


@_bd("ocean", "scenic", "sunset", "ocean horizon sea sun sunset beach water reflection seascape",
     "Ocean horizon with a low sun, its glittering path on the water and a calm sky above.",
     horizon=0.6, sun_x=0.5)
def _ocean(c):
    P = c.P
    k = _scenic_colors(P)
    hz = c.o["horizon"]
    t = (np.arange(c.h) + 0.5) / c.h
    sky = _vramp(c, [k["top"], _mix(k["top"], k["light"], 0.5), k["light"]], clamp01(t / hz))
    sea = _vramp(c, [_mix(k["mid"], k["light"], 0.4), k["mid"], k["dark"]], clamp01((t - hz) / (1 - hz)))
    yh = hz * c.h
    img = _over(sky, sea, clamp01(c.Y - yh + 0.5))
    sx = c.o["sun_x"]
    sunc = _mix(k["light"], "#ffffff", 0.65)
    img = _sun(c, img, sx, hz - 0.07, 0.055 * c.h, sunc, 0.45)
    below = c.Y > yh
    depth = clamp01((c.Y - yh) / (c.h - yh))
    streak = c.fbm(40, 3, seed=8, aspect=0.08)
    width = (0.03 + 0.2 * depth) * c.w
    path = np.exp(-((c.X - sx * c.w) / width) ** 2)
    glit = clamp01((streak - 0.05) * 4) * path * below
    img = _over(img, _c(sunc), glit * 0.85)
    waves = clamp01((c.fbm(60, 3, seed=9, aspect=0.1) - 0.2) * 3) * below * 0.12 * depth
    return _over(img, _c(k["light"]), waves)


@_bd("synthwave", "scenic", "synthwave", "synthwave outrun retrowave 80s sun grid neon perspective vaporwave",
     "Synthwave: striped neon sun over dark mountains and a glowing perspective grid.",
     horizon=0.62)
def _synthwave(c):
    P = c.P
    hz = c.o["horizon"]
    srt = P.sorted()
    deep = P.bg if P.mode == "dark" else P.dark
    pink = P.accent
    warm = P.accents[1] if len(P.accents) > 1 else P.light
    cyan = P.accents[2] if len(P.accents) > 2 else P.light
    t = (np.arange(c.h) + 0.5) / c.h
    img = _vramp(c, [deep, srt[1], _mix(pink, srt[1], 0.45)], clamp01(t / hz) ** 1.4)
    rng = c.rng(16)
    ns = int(260 * c.aspect * min(1.0, 0.25 + c.S))
    sx, sy = rng.uniform(0, c.w, ns), rng.uniform(0, hz * 0.8 * c.h, ns)
    stars = np.zeros((c.h, c.w), F32)
    stars[sy.astype(int).clip(0, c.h - 1), sx.astype(int).clip(0, c.w - 1)] = rng.uniform(0.3, 1, ns)
    img = img + _blur(stars, 0.6 * c.S + 0.3)[..., None] * 0.8
    cx, cy, R = 0.5 * c.w, (hz - 0.17) * c.h, 0.25 * c.h
    d = np.hypot(c.X - cx, c.Y - cy)
    rel = (c.Y - (cy - R)) / (2 * R)
    sun = _ramp([warm, pink], clamp01(rel * 1.15))
    gaps = np.zeros_like(rel)
    for kk in range(6):
        gy = 0.52 + kk * 0.085
        gw = 0.008 + kk * 0.006
        gaps = np.maximum(gaps, clamp01((gw * 2 * R - np.abs(c.Y - (cy - R + gy * 2 * R))) + 0.5))
    m = _aa(d - R) * (1 - gaps)
    img = img + _c(pink) * (np.exp(-(d / (R * 1.6)) ** 2) * 0.35)[..., None]
    img = _over(img, sun, m)
    xs = np.arange(c.w, dtype=F32)
    yh = hz * c.h
    mnt = yh - c.h * (0.11 * clamp01(c.fbm1(260, 4, seed=70, mode="ridged") * 1.3) * (0.4 + np.abs(xs / c.w - 0.5) * 1.6))
    mm = clamp01(c.Y - mnt[None, :] + 0.5) * (c.Y < yh + 1)
    img = _over(img, _c(_mix(deep, srt[1], 0.5)), mm)
    rim = clamp01(1.0 - np.abs(c.Y - mnt[None, :]) / c.lw(1.4)) * (c.Y < yh)
    img = img + _c(cyan) * (rim * 0.5)[..., None]
    gnd = c.Y >= yh
    z = (c.Y - yh + 0.5) / c.h
    ground = _vramp(c, [deep, _mix(deep, srt[1], 0.6)], clamp01((t - hz) / (1 - hz)))
    img = np.where(gnd[..., None], ground, img)
    zc = np.maximum(z, 1e-3)
    kz, kx = 0.9, 1.1
    D = kz / zc
    hl = np.mod(D, 1.0)
    dDdy = kz / (zc ** 2 * c.h)
    hdist = np.minimum(hl, 1 - hl) / np.maximum(dDdy, 1e-6)
    xw = (c.X - cx) / (zc * c.h) * kx
    vl = np.mod(xw + 0.5, 1.0)
    vdist = np.abs(vl - 0.5) * (zc * c.h / kx)
    lw = c.lw(1.8)
    hm = clamp01(lw / 2 + 0.5 - hdist) * clamp01(1.5 - dDdy * lw * 2.5)
    vm = clamp01(lw / 2 + 0.5 - vdist) * clamp01(zc * c.h / kx / (lw * 3))
    grid_m = np.maximum(hm, vm) * gnd * smoothstep(0.004, 0.07, z)
    glow = _blur(grid_m, 5 * c.S)
    img = img + _c(pink) * (glow * 0.9 + grid_m * 0.6)[..., None]
    img = img + _c(pink) * (np.exp(-np.abs(c.Y - yh) / (0.03 * c.h)) * 0.25)[..., None]
    return img


@_bd("starry", "scenic", "deep_space", "starry night stars sky milky way galaxy night gradient cosmos",
     "Night gradient full of stars with a faint milky-way band and a dark hill line.",
     stars=1.0, ground=True)
def _starry(c):
    P = c.P
    srt = P.sorted()
    t = (np.arange(c.h) + 0.5) / c.h
    img = _vramp(c, [srt[0], srt[1], _mix(srt[2], P.accent, 0.3)], clamp01(t) ** 1.2)

    def band(cc):
        x, y = (cc.u - 0.5) * cc.aspect, cc.v - 0.45
        dd = (x * 0.42 + y * 0.9)
        b = np.exp(-(dd / 0.16) ** 2)
        neb = cc.warp(260, 1.2, 4, seed=3) * 0.5 + 0.5
        dust = clamp01(cc.fbm(90, 4, seed=4) * 2)
        return (b * neb * (1 - dust * 0.6)).astype(F32)
    b = c.lowres(band)
    img = img + _c(_mix(P.accent, P.light, 0.5)) * (b * 0.28)[..., None]
    rng = c.rng(17)
    n = int(1400 * c.aspect * c.o["stars"] * min(1.0, 0.25 + c.S))
    sx, sy = rng.uniform(0, c.w, n), rng.uniform(0, c.h, n)
    br = rng.pareto(2.2, n).clip(0, 6) * 0.25 + 0.15
    st = np.zeros((c.h, c.w), F32)
    np.add.at(st, (sy.astype(int).clip(0, c.h - 1), sx.astype(int).clip(0, c.w - 1)), br.astype(F32))
    soft = _blur(st, max(0.55 * c.S * 2, 0.45))
    halo = _blur(st * (st > 0.9), 4 * c.S + 0.5)
    img = img + (_c(P.light) * (soft * 1.6 + halo * 2.0)[..., None]) * (1 - 0.6 * clamp01((c.v - 0.75) * 4))[..., None]
    if c.o["ground"]:
        rid = (0.9 - 0.04 * (c.fbm1(500, 4, seed=81) + 0.5)) * c.h
        img = _ridge_layer(c, img, rid, srt[0], shade=0.0)
    return img


@_bd("clouds", "scenic", "dreamy", "clouds cloud bands sky pastel layered stylised paper cut heaven",
     "Stylised layered cloud bands in pastel tones with soft drop shadows.",
     layers=5)
def _clouds(c):
    P = c.P
    srt = P.sorted()
    t = (np.arange(c.h) + 0.5) / c.h
    top = _mix(P.accent, P.light, 0.35)
    img = _vramp(c, [top, _mix(top, P.light, 0.7)], t)
    n = int(c.o["layers"])
    rng = c.rng(18)
    xs = np.arange(c.w, dtype=np.float64)
    Y = c.Y
    for i in range(n):
        f = i / max(n - 1, 1)
        base = (0.42 + 0.5 * f) * c.h
        edge = np.full(c.w, base + 0.5 * c.h)
        x = -rng.uniform(0, 80) * c.S
        while x < c.w + 120 * c.S:
            r = rng.uniform(60, 130) * c.S * (1.1 - 0.3 * f)
            cy = base + rng.uniform(-0.25, 0.35) * r
            inside = np.abs(xs - x) < r
            prof = cy - np.sqrt(np.maximum(r * r - (xs - x) ** 2, 0))
            edge = np.where(inside, np.minimum(edge, prof), edge)
            x += r * rng.uniform(0.9, 1.4)
        col = to_hex(_ramp([_mix(top, srt[len(srt) // 2], 0.25), _mix(P.light, "#ffffff", 0.5)], np.array(f)))
        sh = np.exp(-np.maximum(edge[None, :] - Y, 0) / (22 * c.S)) * (Y < edge[None, :])
        img = _over(img, _c(_mix(top, P.dark, 0.4)), sh * 0.18)
        m = clamp01(Y - edge[None, :] + 0.5)
        img = _over(img, _c(col), m)
    return img


@_bd("skyline", "scenic", "night_city", "city skyline buildings urban silhouette downtown night windows",
     "City skyline in three hazy bands, the nearest with lit windows, under a dusk gradient.",
     layers=3, windows=0.35)
def _skyline(c):
    P = c.P
    srt = P.sorted()
    t = (np.arange(c.h) + 0.5) / c.h
    img = _vramp(c, [srt[0], srt[2], _mix(P.accent, srt[2], 0.4)], clamp01(t / 0.85) ** 1.3)
    img = _sun(c, img, 0.3, 0.6, 0.05 * c.h, _mix(P.accents[-1] if P.accents else P.light, P.light, 0.3), 0.25)
    rng = c.rng(19)
    n = int(c.o["layers"])
    Y = c.Y
    xs = np.arange(c.w)
    for i in range(n):
        f = i / max(n - 1, 1)
        top = np.full(c.w, float(c.h))
        x = 0.0
        while x < c.w:
            bw = rng.uniform(34, 95) * c.S * (0.8 + 0.4 * f)
            bh = rng.uniform(0.08, 0.32) * c.h * (1.15 - 0.3 * f)
            if rng.random() < 0.12:
                bh *= 1.5
            x0, x1 = int(x), int(min(x + bw, c.w))
            top[x0:x1] = (0.92 - 0.1 * (1 - f)) * c.h - bh
            if rng.random() < 0.2 and x1 - x0 > 6:
                mid = (x0 + x1) // 2
                top[max(mid - 1, 0):mid + 1] -= 0.04 * c.h
            x += bw + rng.uniform(0, 6) * c.S
        col = to_hex(_ramp([_mix(srt[2], srt[1], 0.3), srt[1], srt[0]], np.array(f)))
        m = clamp01(Y - top[None, :] + 0.5)
        img = _over(img, _c(col), m)
        if i >= n - 2:
            gw, gh = 11 * c.S, 15 * c.S
            cx = np.floor(c.X / gw)
            cy = np.floor((Y - top[None, :]) / gh)
            lit = _hash01(cx.astype(np.int64), cy.astype(np.int64), c.seed + i) < c.o["windows"] * (0.6 + 0.4 * f)
            lx, ly = np.mod(c.X, gw), np.mod(Y - top[None, :], gh)
            win = ((lx > gw * 0.3) & (lx < gw * 0.75) & (ly > gh * 0.35) & (ly < gh * 0.75)).astype(F32)
            inner = (Y > top[None, :] + gh) & (Y < c.h)
            wc = _c(_mix(P.accents[-1] if len(P.accents) > 1 else P.light, P.light, 0.4))
            img = _over(img, wc, win * lit * inner * (0.55 + 0.4 * f) * m)
    return img


@_bd("forest", "scenic", "forest", "forest pine trees woods silhouette layers mist fog misty nature",
     "Pine-forest silhouettes in receding layers with mist between them.",
     layers=5)
def _forest(c):
    P = c.P
    k = _scenic_colors(P)
    t = (np.arange(c.h) + 0.5) / c.h
    img = _vramp(c, [k["top"], k["light"]], clamp01(t / 0.7))
    img = _sun(c, img, 0.62, 0.3, 0.05 * c.h, _mix(k["light"], "#ffffff", 0.6), 0.4)
    rng = c.rng(20)
    n = int(c.o["layers"])
    xs = np.arange(c.w, dtype=np.float64)
    far, near = _mix(k["mid"], k["light"], 0.55), k["dark"]
    for i in range(n):
        f = i / max(n - 1, 1)
        ground = (0.55 + 0.38 * f) * c.h
        top = np.full(c.w, ground) - (c.fbm1(600, 2, seed=90 + i) * 0.04 * c.h)
        x = rng.uniform(-20, 0) * c.S
        sc = (0.6 + 0.9 * f)
        while x < c.w + 40 * c.S:
            ht = rng.uniform(70, 160) * c.S * sc
            hw = ht * rng.uniform(0.22, 0.3)
            dx = np.abs(xs - x)
            tier = ht / 5
            prof = ground - ht + dx * (ht / hw) * 0.9 + np.mod(dx * (ht / hw) * 0.9, tier) * 0.0
            yy = ground - ht + dx * ht / hw
            saw = np.mod(yy - (ground - ht), tier) / tier
            prof = yy - saw * tier * 0.45
            prof = np.where(dx < hw, prof, ground + 1e4)
            top = np.minimum(top, prof)
            x += hw * rng.uniform(0.7, 1.3)
        col = to_hex(_ramp([far, k["mid"], near], np.array(f ** 0.9)))
        img = _ridge_layer(c, img, top, col, haze=k["light"], mist=0.5 * (1 - f), shade=0.05)
    return img


@_bd("underwater", "scenic", "ocean", "underwater sea deep light rays god rays caustics diving aquatic",
     "Underwater light: rays slanting down from the surface, shimmer near the top, drifting particles.",
     rays=1.0)
def _underwater(c):
    P = c.P
    srt = P.sorted()
    t = (np.arange(c.h) + 0.5) / c.h
    img = _vramp(c, [srt[-2], srt[len(srt) // 2], srt[1], srt[0]], clamp01(t) ** 0.9)
    ox, oy = 0.35 * c.w, -0.5 * c.h
    ang = np.arctan2(c.X - ox, c.Y - oy)
    a1 = (ang - ang.min()) / max(float(np.ptp(ang)), 1e-6)
    ray = np.interp(a1, np.linspace(0, 1, 2048), np.clip(_n.fbm1d(2048, 22, 3, seed=c.seed * 31 + 7) * 2.2, 0, 1))
    fall = np.exp(-c.v * 2.3) * (0.4 + 0.6 * np.exp(-((c.u - 0.4) / 0.5) ** 2))
    img = img + _c(P.light) * (ray * fall * 0.28 * c.o["rays"])[..., None]

    def shimmer(cc):
        from .texture import cellular
        k = 7.0 / cc.h
        cl = cellular(cc.X * k + cc.fbm(200, 2, seed=3) * 0.3, cc.Y * k * 2.2, 1.0, seed=c.seed + 5)
        return (np.exp(-cl.edge * 14) * np.exp(-cc.v * 9)).astype(F32)
    img = img + _c(P.light) * (c.lowres(shimmer, 420) * 0.45)[..., None]
    rng = c.rng(21)
    n = int(180 * c.aspect * min(1.0, 0.25 + c.S))
    pts = np.zeros((c.h, c.w), F32)
    py, px = rng.uniform(0, c.h, n).astype(int), rng.uniform(0, c.w, n).astype(int)
    pts[py.clip(0, c.h - 1), px.clip(0, c.w - 1)] = rng.uniform(0.3, 1, n)
    img = img + _c(P.light) * (_blur(pts, 1.2 * c.S + 0.4) * 2.0)[..., None]
    return img


# ============================================================================================== organic

@_bd("watercolor", "organic", "sakura", "watercolor watercolour wash paint pigment painted soft art paper",
     "Watercolour washes on paper: pigment pools darker at the edges, with granulation.",
     washes=5)
def _watercolor(c):
    P = c.P
    paper = _mix(P.bg, "#ffffff", 0.55)
    img = c.flat(paper) * (1 + 0.025 * c.fbm(7, 2, seed=1))[..., None]
    acc = P.accents[:4] or [P.accent]
    rng = c.rng(22)
    pts = _scatter_points(c, int(c.o["washes"]), 23, margin=0.0)

    def blob(cc, px, py, r, s):
        x, y = (cc.u - px) * cc.aspect, cc.v - py
        dd = np.hypot(x * 0.8, y) / r + 0.32 * cc.fbm(420, 4, seed=s) + 0.04 * cc.fbm(50, 2, seed=s + 1)
        inner = cc.fbm(300, 3, seed=s + 2)
        return np.stack([dd, inner], -1).astype(F32)
    for i, (px, py) in enumerate(pts):
        r = rng.uniform(0.3, 0.5)
        f = c.lowres(lambda cc: blob(cc, px, py, r, 100 + i), 420)
        dd, inner = f[..., 0], f[..., 1]
        m = smoothstep(1.0, 0.975, dd)
        edge = np.exp(-np.maximum(1.0 - dd, 0) / 0.035) * m
        gran = 0.9 + 0.2 * c.fbm(4, 2, seed=200 + i)
        dens = (0.22 + 0.22 * clamp01(0.5 + inner) + 0.35 * edge) * m * gran * rng.uniform(0.7, 1.0)
        col = _c(acc[i % len(acc)])
        img = img * (1 - dens[..., None] * (1 - col))
    return img


@_bd("ink", "organic", "indigo", "ink in water smoke plume swirl fluid alcohol ink abstract",
     "Ink drifting through water: soft plumes with fine filaments.",
     plumes=3)
def _ink(c):
    P = c.P
    bg = P.bg
    img = _vramp(c, [bg, _mix(bg, P.muted, 0.08)], (np.arange(c.h) + 0.5) / c.h)
    acc = P.accents[:3] or [P.accent]
    rng = c.rng(24)
    dark_bg = P.mode == "dark"
    n = int(c.o["plumes"])
    for i in range(n):
        px = (i + 0.5) / n + rng.uniform(-0.12, 0.12) / n * 2
        py = rng.uniform(-0.05, 0.15)

        def field(cc, px=px, py=py, i=i):
            wx = cc.fbm(420, 4, seed=300 + i) * 0.22
            wy = cc.fbm(420, 4, seed=305 + i) * 0.12
            x, y = (cc.u + wx - px) * cc.aspect, cc.v + wy - py
            depth = np.maximum(y, 0)
            wid = 0.06 + depth * 0.36
            drift = 0.12 * np.sin(depth * 5 + i * 2.1) * depth
            shape = np.exp(-((x - drift) / wid) ** 2) * smoothstep(-0.02, 0.08, y) * np.exp(-depth * 0.55)
            fil = cc.warp(240, 2.6, 5, seed=310 + i, mode="ridged", aspect=1.8)
            soft = clamp01(0.55 + cc.warp(500, 1.8, 4, seed=320 + i))
            return np.stack([shape * soft, shape * fil ** 4], -1).astype(F32)
        fl = c.lowres(field, 420)
        dens = clamp01(fl[..., 0] * 0.9 + fl[..., 1] * 1.4) ** 0.85
        col = _c(acc[i % len(acc)])
        if dark_bg:
            img = _screen(img, col, dens * 0.9)
        else:
            img = img * (1 - dens[..., None] * 0.95 * (1 - col * 0.9))
    return img


@_bd("marble", "organic", "luxurious", "marble swirl stone veins luxury elegant gold veins",
     "Marble swirl: soft cloudy veins with a few fine gold lines.",
     scale=420.0, veins=1.0)
def _marble(c):
    P = c.P
    light = _mix(P.light, "#ffffff", 0.55)
    cloud = _mix(light, P.muted, 0.18)
    vein = _mix(P.muted, P.dark, 0.25)

    def fields(cc):
        w = cc.warp(c.o["scale"], 1.6, 6, seed=1)
        w2 = cc.fbm(c.o["scale"] * 0.5, 5, seed=2)
        x = (cc.u * cc.aspect * 0.7 + cc.v * 0.5) * 4
        s1 = np.sin(x * math.pi + w * 3.2)
        s2 = np.sin(x * 1.9 * math.pi + w * 4.0 + w2 * 1.5 + 1.3)
        return np.stack([s1, s2, w], -1).astype(F32)
    f = c.lowres(fields, 480)
    img = _ramp([light, cloud], clamp01(0.5 + 0.8 * f[..., 2]))
    soft = np.exp(-np.abs(f[..., 0]) * 5)
    img = _over(img, _c(_mix(vein, cloud, 0.45)), soft * 0.55 * c.o["veins"])
    thin = np.exp(-np.abs(f[..., 0]) * 40)
    img = _over(img, _c(vein), thin * 0.5 * c.o["veins"])
    fine = np.exp(-np.abs(f[..., 1]) * 45) * clamp01(0.3 + f[..., 2] * 2)
    img = _over(img, _c(P.accent), fine * 0.9 * c.o["veins"])
    return img * (1 + 0.012 * c.fbm(20, 3, seed=7))[..., None]


@_bd("fluid", "organic", "sunset", "fluid liquid paint acrylic pour swirl flowing glossy abstract colorful",
     "Liquid paint: swirling bands of colour with fine lacing and a glossy sheen.",
     scale=950.0, bands=0.9)
def _fluid(c):
    P = c.P
    acc = (P.accents[:4] if len(P.accents) >= 3 else P.ramp(4))

    def field(cc):
        return cc.warp(c.o["scale"], 2.4, 3, seed=2).astype(F32)
    f = c.lowres(field, 420)
    t = f * c.o["bands"] + 0.5
    img = _ramp(acc + [P.light], t, cyclic=True)
    ph = np.mod(t * (len(acc) + 1), 1.0)
    lace = np.exp(-np.minimum(ph, 1 - ph) * 60)
    img = _over(img, _c(P.light), lace * 0.5)
    gy, gx = np.gradient(f * c.h)
    spec = clamp01(-(gx * 0.6 + gy * 0.8) * 0.9)
    return img + (spec ** 3 * 0.25)[..., None]


@_bd("smoke", "organic", "moody", "smoke wisps fog mist incense vapor dark atmospheric",
     "Smoke wisps curling up through a dark space.",
     density=1.0)
def _smoke(c):
    P = c.P
    lightm = P.mode == "light"
    dark = P.bg if not lightm else P.light
    img = _vramp(c, [dark, _mix(dark, P.muted, 0.12 if lightm else 0.15)], (np.arange(c.h) + 0.5) / c.h)

    def field(cc):
        a = cc.warp(300, 2.4, 5, seed=4, mode="ridged", aspect=1.8)
        b = cc.warp(520, 1.6, 4, seed=5, aspect=1.4) * 0.5 + 0.5
        col = np.exp(-((cc.u - 0.45 - 0.25 * (cc.v - 0.5)) * cc.aspect / 0.35) ** 2)
        return (a ** 4 * 1.6 * b * col + 0.25 * b * col).astype(F32)
    s = clamp01(c.lowres(field, 420) * c.o["density"])
    img = _over(img, _c(_mix(P.light, P.muted, 0.3)), s * 0.75)
    img = _screen(img, _c(P.accent), s * clamp01(1 - c.v * 1.3) * 0.25)
    return _over(img, _c(dark), clamp01(np.hypot(c.u - 0.5, c.v - 0.5) * 1.5) ** 2 * 0.5)


@_bd("bokeh", "organic", "night_city", "bokeh lights blur defocus city lights sparkle festive glow circles",
     "Defocused light discs of several sizes floating over a dark gradient.",
     count=70)
def _bokeh(c):
    P = c.P
    dark = P.bg if P.mode == "dark" else P.dark
    img = _vramp(c, [dark, _mix(dark, P.muted, 0.3)], (np.arange(c.h) + 0.5) / c.h)
    rng = c.rng(25)
    acc = P.accents[:4] or [P.accent]
    n = int(c.o["count"] * c.aspect / 1.78)
    for _ in range(n):
        r = (rng.pareto(1.8) * 18 + 14) * c.S
        r = min(r, 140 * c.S)
        x, y = rng.uniform(0, c.w), rng.uniform(0, c.h) * 0.9 + rng.uniform(0, c.h) * 0.1
        soft = 1.2 + r * 0.06
        x0, x1 = int(max(x - r - 3 * soft, 0)), int(min(x + r + 3 * soft, c.w))
        y0, y1 = int(max(y - r - 3 * soft, 0)), int(min(y + r + 3 * soft, c.h))
        if x0 >= x1 or y0 >= y1:
            continue
        d = np.hypot(c.X[y0:y1, x0:x1] - x, c.Y[y0:y1, x0:x1] - y) - r
        disc = clamp01(0.5 - d / soft)
        rim = np.exp(-((d + soft) / soft) ** 2) * 0.5
        a = (disc * 0.55 + rim) * rng.uniform(0.12, 0.45) * (1 - 0.5 * min(r / (140 * c.S), 1))
        col = _c(acc[int(rng.integers(len(acc)))])
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] + col * a[..., None]
    return img


@_bd("light_leaks", "organic", "retro", "light leaks film leak analog warm glow burn photographic vintage",
     "Film light leaks: warm glows bleeding in from the edges over a dark frame, with grain.",
     leaks=3)
def _light_leaks(c):
    P = c.P
    dark = _mix(P.dark, P.muted, 0.12)
    img = _vramp(c, [dark, _mix(dark, P.accent, 0.12)], (np.arange(c.h) + 0.5) / c.h)
    rng = c.rng(26)
    warm = [x for x in P.accents[:4]] or [P.accent]
    sides = [(0.0, rng.uniform(0.2, 0.8)), (1.0, rng.uniform(0.2, 0.8)), (rng.uniform(0.2, 0.8), 0.0),
             (rng.uniform(0.2, 0.8), 1.0)]
    rng.shuffle(sides)
    for i in range(int(c.o["leaks"])):
        sx, sy = sides[i % 4]
        sw = rng.uniform(0.25, 0.45)

        def leak(cc, sx=sx, sy=sy, sw=sw, i=i):
            x, y = (cc.u - sx) * cc.aspect, cc.v - sy
            d = np.hypot(x * (1.4 if sx in (0.0, 1.0) else 0.7), y * (0.7 if sx in (0.0, 1.0) else 1.4))
            return (np.exp(-(d / sw) ** 2) * (0.75 + 0.25 * cc.fbm(300, 3, seed=400 + i))).astype(F32)
        a = c.lowres(leak, 300)
        col = _c(warm[i % len(warm)])
        img = _screen(img, col, clamp01(a * 1.2))
        img = img + _c(P.light) * (clamp01(a - 0.75) * 1.2)[..., None]
    return _grain(c, img, 0.035)


def _tex(c, name, **kw):
    from .texture import texture
    return np.asarray(texture(name, c.h, c.w, seed=c.seed, **kw), F32)[..., :3]


@_bd("paper", "organic", "minimal", "paper plain sheet white paper stationery clean texture",
     "Plain paper with soft fibres, tinted to the palette background.",
     tint=None)
def _paper(c):
    col = c.o["tint"] or _mix(c.P.bg, "#ffffff", 0.2)
    img = _tex(c, "paper", color=col)
    return _over(img, _c(c.P.dark), clamp01(np.hypot(c.u - 0.5, c.v - 0.5) * 1.5) ** 3 * 0.12)


@_bd("kraft", "organic", "coffee", "kraft paper brown paper craft packaging recycled cardboard",
     "Brown kraft paper with visible fibres and flecks.",
     tint="#b58b5a")
def _kraft(c):
    img = _tex(c, "paper", color=c.o["tint"])
    fib = c.fbm(14, 2, seed=3, aspect=0.25)
    img = img * (1 + 0.05 * fib)[..., None]
    rng = c.rng(27)
    fl = np.zeros((c.h, c.w), F32)
    n = int(260 * c.aspect)
    fl[rng.integers(0, c.h, n), rng.integers(0, c.w, n)] = 1
    fl = _blur(fl, 0.7 * c.S + 0.3)
    return _over(img, _c(_mix(c.o["tint"], "#2a1a0e", 0.6)), clamp01(fl * 3) * 0.3)


@_bd("crumpled", "organic", "minimal", "crumpled paper wrinkled creased crushed paper texture",
     "Crumpled paper: a web of creases catching the light from the upper left.",
     tint=None, creases=36)
def _crumpled(c):
    col = c.o["tint"] or _mix(c.P.bg, "#ffffff", 0.2)
    img = _tex(c, "paper", color=col)
    rng = c.rng(28)
    shade = np.zeros((c.h, c.w), F32)
    for k in range(int(c.o["creases"])):
        px, py = rng.uniform(0, c.w), rng.uniform(0, c.h)
        ang = rng.uniform(0, math.pi)
        nx, ny = math.cos(ang), math.sin(ang)
        d = (c.X - px) * nx + (c.Y - py) * ny
        L = rng.uniform(0.05, 0.3) * c.h
        facing = (nx * -0.6 + ny * -0.8)
        reach = np.exp(-((c.X - px) * ny - (c.Y - py) * nx) ** 2 / (rng.uniform(0.2, 0.6) * c.h) ** 2)
        shade += (np.clip(d / 0.9, -1, 1) * facing * np.exp(-np.abs(d) / L) * reach * rng.uniform(0.02, 0.06)).astype(F32)
    shade += 0.03 * c.fbm(120, 3, seed=9)
    return img * (1 + shade)[..., None]


@_bd("warm_paper", "organic", "coffee",
     "warm paper cream paper parchment ivory beige old paper natural paper flyer menu poster stationery recycled",
     "Warm cream paper: a light-to-deeper vertical tone, soft fibres, gentle mottling and warm edges.",
     tint="#f3e5c8", fibres=1.0, edges=1.0)
def _warm_paper(c):
    base = _c(c.o["tint"])
    top, bottom = base + (1 - base) * 0.42, base * 0.94
    v = c.v[..., None]
    img = (top * (1 - v) + bottom * v).astype(F32)
    fib = (0.022 * c.fbm(9, 2, seed=3, aspect=0.22) + 0.012 * c.fbm(3.5, 1, seed=4)) * float(c.o["fibres"])
    mott = c.lowres(lambda cc: cc.fbm(260, 4, seed=5), 300) * 0.035
    img = img * (1 + fib + mott)[..., None]
    edge = clamp01(np.hypot((c.u - 0.5) * 1.15, c.v - 0.5) * 1.55) ** 3 * 0.16 * float(c.o["edges"])
    return _over(img, _c(_mix(c.o["tint"], "#6b4a22", 0.75)), edge)


@_bd("concrete", "organic", "slate", "concrete wall cement brutalist industrial raw grey gray",
     "Smooth cast concrete with soft stains, toned toward the palette.",
     tint=0.35)
def _concrete(c):
    P = c.P
    img = _tex(c, "concrete", tint=(_mix(P.muted, P.bg, 0.4), float(c.o["tint"])))
    return img


@_bd("fabric", "organic", "terracotta", "fabric cloth linen textile weave canvas woven material",
     "Woven fabric (linen by default) dyed in the palette accent.",
     weave="linen", color=None)
def _fabric(c):
    P = c.P
    col = c.o["color"] or _mix(P.muted, P.bg, 0.2)
    return _tex(c, c.o["weave"], tint=col)


@_bd("wood_desk", "organic", "coffee", "wood desk table top view wooden planks workspace flat lay tabletop",
     "Top view of a wooden desk: long boards with a soft light falloff (flat-lay backgrounds).",
     tint=None)
def _wood_desk(c):
    kw = {"tint": (c.o["tint"], 0.5)} if c.o["tint"] else {}
    img = _tex(c, "planks", angle=0, **kw)
    light = np.exp(-(((c.u - 0.3) * c.aspect) ** 2 + (c.v - 0.2) ** 2) / 1.2)
    return img * (0.82 + 0.25 * light)[..., None]


# ============================================================================================== calm region

def _calm_box(img):
    """Largest low-detail rectangle (fractions x0, y0, x1, y1) of an image."""
    h, w = img.shape[:2]
    gh = 14
    gw = int(np.clip(round(gh * w / h), 5, 40))
    sub = 6
    small = resize(np.clip(img, 0, 1), gw * sub, gh * sub, "box")
    lab = srgb_to_oklab(small).astype(np.float64)
    hp = np.abs(lab - gaussian_blur(lab.astype(F32), sub * 0.5)).sum(-1)       # local detail, not smooth slopes
    D = hp.reshape(gh, sub, gw, sub).mean((1, 3))
    D = 0.6 * D + 0.4 * np.maximum(D, gaussian_blur(D.astype(F32), 0.8))      # busy neighbours count a little
    p75 = np.percentile(D, 75)
    thr = max(0.008, D.min() + 0.3 * (p75 - D.min()))
    if p75 < 0.03 or (p75 - D.min()) < 0.45 * p75:                # evenly textured: no area is calmer
        thr = max(thr, np.percentile(D, 90) * 1.05)
    good = D <= thr
    best, bscore = (0, 0, gw, gh), -1.0
    cw, ch = w / gw, h / gh
    for r0 in range(gh):
        colok = np.ones(gw, bool)
        for r1 in range(r0 + 1, gh + 1):
            colok &= good[r1 - 1]
            if not colok.any():
                break
            run = 0
            for x in range(gw + 1):
                if x < gw and colok[x]:
                    run += 1
                    continue
                if run:
                    c0, c1 = x - run, x
                    bw, bh = run * cw, (r1 - r0) * ch
                    a = bw / bh
                    pen = 1.0 if 0.7 <= a <= 5 else (a / 5) ** -0.6 if a > 5 else (a / 0.7) ** 0.6
                    cx, cy = (c0 + c1) / 2 / gw, (r0 + r1) / 2 / gh
                    score = bw * bh * pen * (1 - 0.25 * math.hypot(cx - 0.5, cy - 0.5))
                    if score > bscore:
                        best, bscore = (c0, r0, c1, r1), score
                run = 0
    c0, r0, c1, r1 = best
    if (c1 - c0) * (r1 - r0) < 0.07 * gw * gh:                   # no real calm area: centre box, needs quiet()
        return (0.2, 0.25, 0.8, 0.75), True
    ins = 0.25
    x0, x1 = (c0 + (ins if c0 > 0 else 0)) / gw, (c1 - (ins if c1 < gw else 0)) / gw
    y0, y1 = (r0 + (ins if r0 > 0 else 0)) / gh, (r1 - (ins if r1 < gh else 0)) / gh
    mx, my = 0.07 * min(w, h) / w, 0.07 * min(w, h) / h               # type never touches the canvas edge
    x0, x1, y0, y1 = max(x0, mx), min(x1, 1 - mx), max(y0, my), min(y1, 1 - my)
    return (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)), False


def _ink_for(img, box, P):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), max(int(box[2] * w), int(box[0] * w) + 1), \
        max(int(box[3] * h), int(box[1] * h) + 1)
    reg = np.clip(img[y0:y1, x0:x1].reshape(-1, 3), 0, 1)
    lum = reg @ np.array([0.2126, 0.7152, 0.0722], F32)
    lo = reg[np.argsort(lum)[int(len(lum) * 0.08)]]
    hi = reg[np.argsort(lum)[int(len(lum) * 0.92)]]
    mean = reg.mean(0)
    best, bs = None, -1
    for ink in [P.fg, P.bg, P.dark, P.light, P.text_on(to_hex(mean))]:
        s = min(contrast_ratio(ink, to_hex(lo)), contrast_ratio(ink, to_hex(hi)))
        if s > bs:
            best, bs = ink, s
    if bs < 4.5:
        best = ensure_contrast(best, to_hex(mean), 4.5)
    return to_hex(best)


# ============================================================================================== public API

class Backdrop:
    """What ``backdrop()`` painted: ``name``, ``palette``, ``calm`` (x0, y0, x1, y1 fractions of the canvas: the
    largest quiet area), ``ink`` (a colour that reads there), ``seed``, ``busy`` (True when the whole backdrop is
    evenly detailed: ``calm`` is then a centred box and text needs ``quiet(cv, bd.calm_px(cv))`` first).
    ``calm_px(cv)`` gives the box in pixels."""

    def __init__(self, name, palette, calm, ink, seed, box, busy=False):
        self.name, self.palette, self.calm, self.ink, self.seed, self.box = name, palette, calm, ink, seed, box
        self.busy = bool(busy)

    def calm_px(self, cv, pad=0.0):
        """The calm box in pixels of ``cv`` (optionally shrunk by ``pad`` as a fraction of its short side)."""
        x0, y0, x1, y1 = self.calm
        x0, x1, y0, y1 = x0 * cv.w, x1 * cv.w, y0 * cv.h, y1 * cv.h
        p = pad * min(x1 - x0, y1 - y0)
        return (int(x0 + p), int(y0 + p), int(x1 - p), int(y1 - p))

    def __repr__(self):
        return (f"Backdrop({self.name!r}, palette={self.palette.name!r}, calm={self.calm}, ink={self.ink!r}"
                f"{', busy=True' if self.busy else ''})")


class _UnknownName(ValueError, KeyError):
    """Unknown registry name (a ValueError that is also a KeyError)."""
    __str__ = ValueError.__str__


def _resolve(name):
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key in BACKDROPS:
        return key, False
    alias = {"gradient": "linear", "noise": "grainy", "grain": "grainy", "watercolour": "watercolor",
             "glassmorphism": "glass", "topo": "topographic", "contours": "topographic", "hex": "hexagons",
             "dots": "dot_grid", "grid": "line_grid", "lowpoly": "low_poly", "city": "skyline", "stars": "starry",
             "sea": "ocean", "wood": "wood_desk", "outrun": "synthwave", "confetti": "memphis",
             "cream": "warm_paper", "cream_paper": "warm_paper", "parchment": "warm_paper", "warm": "warm_paper",
             "paper_warm": "warm_paper", "old_paper": "warm_paper"}
    if key in alias:
        return alias[key], False
    hit = find(name)
    if hit is None:
        raise _UnknownName(*_unknown_name("backdrop", name, list(BACKDROPS),
                                          "Try backdrop.find('...') or backdrop.names()").args)
    return hit, True


def _box_px(cv, box):
    x0, y0, x1, y1 = box
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0, x1, y0, y1 = x0 * cv.w, x1 * cv.w, y0 * cv.h, y1 * cv.h
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def render(name, w, h, palette=None, seed=0, **opts):
    """Compute a backdrop as an (h, w, 3) float32 array without painting it. Returns (array, Palette)."""
    key, _ = _resolve(name)
    spec = BACKDROPS[key]
    opts = _reject(f"backdrop '{key}'", opts, spec["opts"]) if opts else opts
    o = dict(spec["opts"])
    o.update(opts)
    P = _palette(palette if palette is not None else spec["palette"])
    img = spec["fn"](_Ctx(h, w, P, seed, o))
    img = np.clip(np.asarray(img, F32), 0, 1)
    rng = np.random.default_rng(seed + 977)
    img = img + ((rng.random((h, w), np.float32) - rng.random((h, w), np.float32)) / 255.0)[..., None]
    return np.clip(img, 0, 1).astype(F32), P


def backdrop(cv, name="mesh", palette=None, seed=0, strength=1.0, box=None, **opts):
    """Paint a backdrop onto ``cv`` (one ``cv.paint`` / ``cv.paste`` call = one layer) -> Backdrop.

    name: a backdrop name (``names()``) or a description ("soft pastel blobs", "夜晚城市"). A description also picks
    a palette from its colour / mood words when ``palette`` is None.
    palette: name, description, Palette or list of colours (default: the backdrop's own suggestion).
    strength: opacity over what is already on the canvas (1 = opaque).
    box: (x0, y0, x1, y1) fractions or pixels -> paint only that rectangle (cards, banners, panels).
    opts: the backdrop's own options, see ``describe(name)``; they are checked (an unknown name raises a TypeError
    naming the closest valid option) and both spellings (colour / color ...) are accepted.
    """
    key, fuzzy = _resolve(name)
    if palette is None and fuzzy:
        own = set(BACKDROPS[key]["words"]) | set(key.split("_"))
        rest = [t for t in query_tokens(name) if t not in own]
        text = " ".join(rest)
        palette = text if rest and (_colour_words(text) or _pfind(text)) else None
    if box is None:
        x0, y0, x1, y1 = 0, 0, cv.w, cv.h
    else:
        x0, y0, x1, y1 = _box_px(cv, box)
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    img, P = render(key, w, h, palette, seed, **opts)
    if box is None:
        cv.paint(img, None, opacity=float(strength))
    else:
        cv.paste(img, x0, y0, opacity=float(strength))
    calm, busy = _calm_box(img)
    ink = _ink_for(img, calm, P)
    cx0, cy0 = x0 + calm[0] * w, y0 + calm[1] * h
    cx1, cy1 = x0 + calm[2] * w, y0 + calm[3] * h
    calm_c = (round(cx0 / cv.w, 4), round(cy0 / cv.h, 4), round(cx1 / cv.w, 4), round(cy1 / cv.h, 4))
    return Backdrop(key, P, calm_c, ink, seed, (x0, y0, x1, y1), busy)


def quiet(cv, box, amount=0.6, tone=None):
    """Calm a busy area behind text: blur + tone toward the local average (or ``tone``), feathered so no box edge
    shows. ``box`` = (x0, y0, x1, y1) fractions or pixels; ``amount`` 0~1. Returns the pixel box."""
    x0, y0, x1, y1 = _box_px(cv, box)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    fe = max(0.22 * min(bw, bh), 6.0)
    ex0, ey0 = int(max(x0 - fe * 2, 0)), int(max(y0 - fe * 2, 0))
    ex1, ey1 = int(min(x1 + fe * 2, cv.w)), int(min(y1 + fe * 2, cv.h))
    sub = cv.buf[ey0:ey1, ex0:ex1]
    sigma = max(2.0, 0.06 * min(bw, bh)) * (0.4 + amount)
    bl = _blur(sub, sigma)
    target = _c(tone) if tone is not None else bl[max(y0 - ey0, 0):y1 - ey0, max(x0 - ex0, 0):x1 - ex0].reshape(-1, 3).mean(0)
    calm = bl + (target - bl) * (0.6 * amount)
    X, Y = grid(ey1 - ey0, ex1 - ex0)
    cx, cy, hw, hh = (x0 + x1) / 2 - ex0, (y0 + y1) / 2 - ey0, bw / 2, bh / 2
    qx, qy = np.abs(X - cx) - hw + fe, np.abs(Y - cy) - hh + fe
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - fe
    m = smoothstep(fe * 1.2, -fe * 0.3, d) * min(1.0, amount * 1.4)
    cv.paste(calm, ex0, ey0, mask=m)
    return (x0, y0, x1, y1)


def names(family=None):
    """All backdrop names, or those of one family: gradient / geometric / scenic / organic."""
    if family is None:
        return list(BACKDROPS)
    if family not in FAMILIES:
        raise _UnknownName(*_unknown_name("backdrop family", family, list(FAMILIES)).args)
    return list(FAMILIES[family])


def find(query, n=1):
    """Best backdrop name for a description ("soft pastel blobs", "星空") -> name (or a list of ``n``), None if
    nothing matches."""
    toks = query_tokens(query)
    stop = {"a", "an", "the", "of", "with", "and", "in", "on", "for", "background", "backdrop", "wallpaper",
            "pattern", "soft", "dark", "light", "warm", "cool", "modern", "abstract"}
    scored = []
    for i, (nm, sp) in enumerate(BACKDROPS.items()):
        words = set(sp["words"])
        parts = set(nm.split("_"))
        s = 0.0
        for t in toks:
            if t in parts or t == nm:
                s += 3
            elif t in words:
                s += 1 if t in stop else 2
            elif len(t) >= 4 and any(wd.startswith(t) or t.startswith(wd) for wd in words if len(wd) >= 4):
                s += 0.6
        if s > 0:
            scored.append((-s, i, nm))
    scored.sort()
    res = [nm for _, _, nm in scored[:n]]
    return (res[0] if res else None) if n == 1 else res


def describe(name=None):
    """Text description of one backdrop (family, palette, options) or a one-line catalogue of all of them."""
    if name is None:
        return "\n".join(f"{nm:<12} {sp['family']:<9} {sp['desc']}" for nm, sp in BACKDROPS.items())
    key, _ = _resolve(name)
    sp = BACKDROPS[key]
    opts = ", ".join(f"{k}={v!r}" for k, v in sp["opts"].items()) or "none"
    return f"{key} ({sp['family']}): {sp['desc']}\n  default palette: {sp['palette']}\n  options: {opts}"


# ---- scene manifest: the backdrop is recorded as "backdrop: <name>" (pixelcanvas.manifest) ----
from . import manifest as _manifest  # noqa: E402

def _after_backdrop(cap, args, res):
    m = cap.mask()
    if m is None:
        return []
    key = _resolve(args["name"])[0]
    return [dict(kind="background", name="backdrop: " + key, mask=m,
                 alias=["backdrop", "background", str(args["name"])], meta={"backdrop": True, "call": "backdrop"})]


backdrop = _manifest.drawn("background", after=_after_backdrop)(backdrop)


# ---- strict options: unknown keyword names raise TypeError naming the closest valid option; colour / color,
# ---- w / width ... are accepted (pixelcanvas._args). Backdrop options are checked in ``render``.
for _n_, _f in list(globals().items()):
    if (not _n_.startswith("_") and callable(_f) and getattr(_f, "__module__", None) == __name__
            and not isinstance(_f, type)):
        globals()[_n_] = _strict(_f, _n_)
