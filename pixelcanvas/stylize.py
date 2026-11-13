# -*- coding: utf-8 -*-
"""
stylize — turn a finished render into a *medium* or *look* with one call: oil, watercolour, pencil, charcoal,
sumi-e, engraving, comic, manga, ukiyo-e, risograph, linocut, stained glass, mosaic, low-poly, pixel art, paper cut,
cross-stitch, blueprint, synthwave, vintage film, noir, tilt-shift, glitch, thermal, sticker, clay and more.

>>> from pixelcanvas.stylize import apply, pick, names
>>> apply(cv, "ukiyoe", seed=seed)                    # in place on a Canvas (or array in → array out)
>>> apply(cv, pick("a risograph print of a fox"))     # pick() maps a request (en / zh / ja) to a style name
>>> names()                                           # every style; describe(name) gives one line each

Every style is also a plain function ``fn(img, ..., seed=0) -> (H, W, 3) float32`` usable with ``cv.apply``.
Sizes in the registry styles are in 1080p pixels and scale with the image height (``S = H / 1080``), so a
960x540 draft looks like the full-size render. The four original filters (``oil_paint``, ``watercolor``,
``pencil_sketch``, ``cel_shade``) keep their absolute-pixel parameters.

Apply the medium as the last step, in its own ``log("medium")`` step.
"""
from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image, ImageDraw

from .color import F32, clamp01, luma, smoothstep, mix, rgb, grid
from .filters import gaussian_blur, sample, resize
from .noise import fbm2d
from . import manifest as _manifest
from ._args import (reject_unknown as _reject, signature_options as _sigopts, unknown_name as _unknown_name,
                    strict as _strict)

__all__ = ["kuwahara", "flow_field", "lic", "oil_paint", "watercolor", "pencil_sketch", "cel_shade", "canvas_texture",
           "MEDIA", "STYLES", "PALETTES", "apply", "options", "names", "pick", "describe",
           "oil", "impressionist", "pointillism", "gouache", "ink_wash", "colored_pencil", "charcoal", "pastel",
           "crayon", "storybook", "pen_ink", "chalkboard", "blueprint", "comic", "manga", "anime", "ukiyoe", "risograph",
           "linocut", "engraving", "cyanotype", "duotone", "stained_glass", "mosaic", "low_poly", "pixel_art",
           "paper_cut", "collage", "cross_stitch", "clay", "sticker", "photo_finish", "vintage", "sepia", "noir",
           "tilt_shift", "synthwave", "vaporwave", "glitch", "thermal", "infrared"]


def _img(a):
    a = np.asarray(a, F32)
    return np.repeat(a[..., None], 3, -1) if a.ndim == 2 else a


def _box_mean(a, r):
    """Mean over the (r+1)x(r+1) window whose top-left corner is each pixel (edge padded) — integral image."""
    r = int(r)
    pad = [(0, r), (0, r)] + [(0, 0)] * (a.ndim - 2)
    p = np.pad(a, pad, mode="edge").astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, [(1, 0), (1, 0)] + [(0, 0)] * (a.ndim - 2))
    H, W = a.shape[:2]
    k = r + 1
    s = c[k:k + H, k:k + W] - c[0:H, k:k + W] - c[k:k + H, 0:W] + c[0:H, 0:W]
    return (s / (k * k)).astype(F32)


def kuwahara(img, radius=4):
    """Edge-preserving painterly smoothing: each pixel takes the mean of its least-varied quadrant."""
    img = _img(img)
    r = max(int(round(radius)), 1)
    H, W = img.shape[:2]
    L = luma(img)
    m = _box_mean(img, r)
    m1 = _box_mean(L, r)
    m2 = _box_mean(L * L, r)
    var = m2 - m1 * m1
    P = r
    mp = np.pad(m, [(P, P), (P, P), (0, 0)], mode="edge")
    vp = np.pad(var, [(P, P), (P, P)], mode="edge")
    best_v = np.full((H, W), np.inf, F32)
    out = np.empty_like(img)
    for dy in (-r, 0):
        for dx in (-r, 0):
            v = vp[P + dy:P + dy + H, P + dx:P + dx + W]
            mm = mp[P + dy:P + dy + H, P + dx:P + dx + W]
            sel = v < best_v
            best_v = np.where(sel, v, best_v)
            out = np.where(sel[..., None], mm, out)
    return out


def flow_field(img, sigma=3.0):
    """Unit vectors (vx, vy) running ALONG edges (perpendicular to the gradient), smoothed by a structure tensor."""
    L = gaussian_blur(luma(_img(img)), max(sigma * 0.5, 0.5))
    gy, gx = np.gradient(L)
    jxx, jxy, jyy = (gaussian_blur(a, sigma) for a in (gx * gx, gx * gy, gy * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return np.cos(theta).astype(F32), np.sin(theta).astype(F32), clamp01(coherence)


def lic(tex, vx, vy, length=12.0, steps=8):
    """Line-integral convolution: smear ``tex`` along the flow (vx, vy) over ``length`` px each way → strokes."""
    H, W = tex.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(F32)
    acc = tex.astype(F32).copy()
    n = 1.0
    h = float(length) / steps
    for sgn in (1.0, -1.0):
        X, Y = xs.copy(), ys.copy()
        ux, uy = vx * sgn, vy * sgn
        for i in range(steps):
            X = X + ux * h
            Y = Y + uy * h
            w = 1.0 - (i + 1) / (steps + 1)
            acc += sample(tex, X, Y) * w
            n += w
            nvx, nvy = sample(vx, X, Y), sample(vy, X, Y)
            flip = np.where(nvx * ux + nvy * uy < 0, -1.0, 1.0).astype(F32)   # keep heading consistent
            ux, uy = nvx * flip, nvy * flip
    return (acc / n).astype(F32)


def oil_paint(img, brush=6.0, strength=1.0, impasto=0.35, light=(-0.6, -0.8), seed=0, steps=8):
    """Oil / acrylic painting: Kuwahara colour blocks + directional brush strokes (LIC) + impasto relief.

    ``brush``: stroke width in px (≈ 4~10 at 1080p; bigger = looser, more impressionist). ``impasto``: relief light.
    """
    img = _img(img)
    H, W = img.shape[:2]
    base = kuwahara(img, max(brush * 0.7, 1.5))
    base = kuwahara(base, max(brush * 0.35, 1))
    vx, vy, coh = flow_field(img, sigma=brush * 1.2)
    rng = np.random.default_rng(seed)
    # stroke texture: noise at stroke-width scale smeared along the flow
    noise = gaussian_blur(rng.random((H, W), dtype=np.float32), max(brush * 0.25, 0.6))
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    strokes = lic(noise, vx, vy, length=brush * 2.5, steps=steps)
    strokes = strokes / (strokes.std() + 1e-6)
    out = base * (1.0 + 0.045 * strength * strokes[..., None])
    # impasto: light the stroke height field
    gy, gx = np.gradient(gaussian_blur(strokes, 0.8))
    lx, ly = light
    shade = clamp01(0.5 + (gx * lx + gy * ly) * 0.9)
    out = out + (shade[..., None] - 0.5) * impasto * strength * 0.25
    return clamp01(out).astype(F32)


def watercolor(img, wash=6.0, transparency=0.35, edge=0.8, granulation=1.0, bleed=6.0, paper=1.0, deckle=0.05,
               seed=0):
    """Watercolour: transparent, desaturated washes with soft wet edges, pigment pooling at wash borders,
    granulation, blotchy density, visible cold-press paper and an unpainted white border (``deckle``, fraction)."""
    img = _img(img)
    H, W = img.shape[:2]
    S = H / 1080.0
    dx = fbm2d(H, W, 70 * S, 4, seed=seed + 11) * bleed * S
    dy = fbm2d(H, W, 70 * S, 4, seed=seed + 12) * bleed * S
    X, Y = np.meshgrid(np.arange(W, dtype=F32), np.arange(H, dtype=F32))
    w = sample(gaussian_blur(kuwahara(img, max(wash * S, 1)), 1.5 * S + 0.5), X + dx, Y + dy)
    m = w.mean(-1, keepdims=True)
    w = m + (w - m) * 0.8                                            # watercolour pigments are softer
    pig = (1.0 - w) * (1.0 - transparency)                           # pigment density; paper glows through
    L = luma(1.0 - pig)
    ring = clamp01(np.abs(L - gaussian_blur(L, 4.0 * S + 1)) * 10.0)   # pigment pools along wash borders
    gran = fbm2d(H, W, 2.2 * S + 1.0, 3, seed=seed + 3) * 0.5 + 0.5
    blot = fbm2d(H, W, 120 * S, 4, seed=seed + 5) * 0.5 + 0.5
    pig = pig * (1.0 + granulation * 0.6 * (gran - 0.5))[..., None] * (0.75 + 0.5 * blot)[..., None]
    pig = pig * (1.0 + edge * ring[..., None])
    if deckle:                                                       # paint stops short of the paper edge, raggedly
        U, V = X / W, Y / H
        d = np.minimum(np.minimum(U, 1 - U), np.minimum(V, 1 - V))
        rag = fbm2d(H, W, 40 * S, 4, seed=seed + 21) * deckle * 0.6
        pig = pig * smoothstep(deckle * 0.5, deckle * 1.1, d + rag)[..., None]
    out = 1.0 - pig
    tex = fbm2d(H, W, 1.6 * S + 0.8, 2, seed=seed + 9) * 0.5 + 0.5
    tooth = fbm2d(H, W, 6 * S + 1, 2, seed=seed + 10) * 0.5 + 0.5
    out = out * (1.0 - paper * 0.05 * ((tex + tooth)[..., None] - 1.0))
    return clamp01(out * np.array([1.0, 0.99, 0.965], F32)).astype(F32)


def pencil_sketch(img, line=1.0, hatching=0.8, colored=0.0, seed=0):
    """Graphite drawing: dodge-blend contour lines + directional hatching in the darks on paper. ``colored`` 0~1
    keeps some of the original colour (coloured pencil)."""
    img = _img(img)
    H, W = img.shape[:2]
    S = H / 1080.0
    L = luma(img)
    inv = gaussian_blur(1.0 - L, 6.0 * S + 1)
    dodge = clamp01(L / np.maximum(1.0 - inv, 1e-3))                 # colour-dodge: flat areas → white, edges → lines
    lines = 1.0 - (1.0 - dodge) ** 1.0 * line
    vx, vy, _ = flow_field(img, sigma=6 * S + 1)
    rng = np.random.default_rng(seed)
    n = rng.random((H, W), dtype=np.float32)
    # hatch along a fixed 45° + along the flow, both only where the tone is dark
    ang = np.float32(np.pi / 4)
    h1 = lic(n, np.full_like(L, np.cos(ang)), np.full_like(L, np.sin(ang)), length=10 * S + 3, steps=6)
    h2 = lic(n, vx, vy, length=10 * S + 3, steps=6)
    h1 = clamp01((h1 - 0.5) * 6 + 0.5)
    h2 = clamp01((h2 - 0.5) * 6 + 0.5)
    tone = 1.0 - L
    hatch = 1.0 - hatching * (smoothstep(0.25, 0.75, tone) * (1 - h1) * 0.45 + smoothstep(0.5, 0.95, tone) * (1 - h2) * 0.45)
    g = clamp01(lines * hatch * (0.97 + 0.03 * (fbm2d(H, W, 2 * S + 1, 2, seed=seed + 1) * 0.5 + 0.5)))
    paper = np.array([0.97, 0.96, 0.93], F32)
    out = g[..., None] * paper
    if colored:
        out = out * (1 - colored) + out * (0.55 + 0.45 * img) * colored
    return clamp01(out).astype(F32)


def cel_shade(img, levels=3, softness=0.04, outline=2.0, outline_color=(0.08, 0.06, 0.1), saturation=1.15):
    """Anime / cartoon look: flat colour bands (luminance quantised, hue kept) + dark outlines on region edges."""
    img = _img(img)
    H, W = img.shape[:2]
    base = kuwahara(img, 2)
    L = luma(base)[..., None]
    q = np.zeros_like(L)
    for k in range(1, int(levels)):
        q += smoothstep(k / levels - softness, k / levels + softness, L)
    q = (q + 0.5) / levels
    chroma = base / np.maximum(L, 1e-3)
    out = clamp01(chroma * q)
    m = out.mean(-1, keepdims=True)
    out = clamp01(m + (out - m) * saturation)
    if outline:
        e = np.zeros((H, W), F32)
        for c in range(3):
            gy, gx = np.gradient(gaussian_blur(base[..., c], 0.7))
            e = np.maximum(e, np.hypot(gx, gy))
        e = gaussian_blur(clamp01((e - 0.04) * 12.0), max(outline * 0.35, 0.4))
        e = clamp01(e * 1.6)
        out = out * (1 - e[..., None]) + np.asarray(outline_color, F32) * e[..., None]
    return out.astype(F32)


def canvas_texture(img, strength=0.06, weave=3.0, seed=0):
    """Woven canvas / paper tooth, multiplied in (use after oil_paint for 'painting on canvas')."""
    img = _img(img)
    H, W = img.shape[:2]
    y, x = np.mgrid[0:H, 0:W].astype(F32)
    p = max(float(weave), 1.0)
    wv = (np.sin(x * np.pi / p) * np.sin(y * np.pi / p)) * 0.5
    n = fbm2d(H, W, p * 1.5, 2, seed=seed) * 0.5
    t = (wv + n)[..., None]
    return clamp01(img * (1.0 + strength * t)).astype(F32)



# =================================================================================== shared building blocks

def _S(img):
    return max(img.shape[0], 1) / 1080.0


def _prep(img):
    img = clamp01(_img(img))
    return img, img.shape[0], img.shape[1], _S(img)


def _blur(a, sigma):
    """Gaussian blur; large sigmas run on a box-downsampled copy (same look, much faster)."""
    a = np.asarray(a, F32)
    if sigma <= 0.3:
        return a
    if sigma > 8:
        H, W = a.shape[:2]
        f = max(int(sigma / 4), 2)
        h, w = max(H // f, 4), max(W // f, 4)
        small = resize(a, w, h, "box")
        return resize(gaussian_blur(small, sigma * w / W), W, H, "bilinear")
    return gaussian_blur(a, sigma)


def _hash(a, b=0, seed=0):
    """Deterministic integer hash → float32 in [0, 1)."""
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    h = (a * 374761393 + b * 668265263 + ((int(seed) * 2246822519 + 97531) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return ((h & 0xFFFFFF).astype(np.float64) / 16777216.0).astype(F32)


def _vnoise(u, v, seed=0):
    """Value noise at arbitrary lattice coordinates, about -1~1 (cheap; good for anisotropic streaks)."""
    fu0 = np.floor(u)
    fv0 = np.floor(v)
    iu = fu0.astype(np.int64)
    iv = fv0.astype(np.int64)
    fu = (u - fu0).astype(F32)
    fv = (v - fv0).astype(F32)
    su = fu * fu * (3 - 2 * fu)
    sv = fv * fv * (3 - 2 * fv)
    a = _hash(iu, iv, seed)
    b = _hash(iu + 1, iv, seed)
    c = _hash(iu, iv + 1, seed)
    d = _hash(iu + 1, iv + 1, seed)
    return ((a + (b - a) * su + (c - a) * sv + (a - b - c + d) * su * sv) * 2.0 - 1.0).astype(F32)


def _streaks(X, Y, angle, length, width, seed=0, octaves=2):
    """Anisotropic noise stretched along ``angle`` (radians, scalar or array): strokes, scribbles, wood grain."""
    c, s = np.cos(angle), np.sin(angle)
    u = (X * c + Y * s) / length
    v = (Y * c - X * s) / width
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        f = 2.0 ** o
        tot = tot + amp * _vnoise(u * f, v * f, seed + 31 * o)
        norm += amp
        amp *= 0.5
    return (tot / norm).astype(F32)


def _streaks_flow(X, Y, angle, length, width, seed=0, n=6):
    """Streaks whose direction follows a smooth angle field: fixed-angle streaks blended by angle (no swirl)."""
    angle = np.asarray(angle, F32)
    tot = np.zeros(X.shape, F32)
    wsum = np.zeros(X.shape, F32)
    w2 = np.zeros(X.shape, F32)
    for i in range(n):
        th = math.pi * i / n
        w = np.maximum(np.cos(2 * (angle - th)), 0) ** 3
        if not np.any(w > 1e-3):
            continue
        tot += w * _streaks(X, Y, th, length, width, seed + 13 * i)
        wsum += w
        w2 += w * w
    return (tot / np.sqrt(np.maximum(w2, 1e-6))).astype(F32) * 0.85


def _clean(img, S):
    """Remove render grain / noise before line extraction (keeps edges)."""
    return kuwahara(img, max(1.5 * S, 1))


def _tooth(H, W, S, seed=0, scale=1.0):
    """Paper tooth: fine grain 0~1 (peaks catch dry media, valleys stay white)."""
    t = fbm2d(H, W, 2.2 * S * scale + 0.8, 2, seed=seed) * 0.7 + fbm2d(H, W, 9 * S * scale + 1, 2, seed=seed + 1) * 0.3
    return clamp01(t * 1.1 + 0.5)


def _paper(H, W, S, color, seed=0, amount=0.05):
    """Paper colour map with grain and soft mottling."""
    g = fbm2d(H, W, 1.8 * S + 0.8, 2, seed=seed + 101) * 0.55 + fbm2d(H, W, 60 * S + 2, 3, seed=seed + 102) * 0.45
    return (rgb(color) * (1.0 + amount * g)[..., None]).astype(F32)


def _dog(L, sigma, k=1.6):
    return _blur(L, sigma) - _blur(L, sigma * k)


def _edges(img, S, width=1.0, lo=0.010, hi=0.035, chroma=0.6, adapt=1.0):
    """Ink-line mask (0~1) on the dark side of real edges: difference of Gaussians on luma (+ colour edges).
    ``adapt`` > 0 lowers the thresholds on low-contrast pictures so their main edges still draw."""
    L = luma(img)
    s = max(0.9 * width * S, 0.55)
    d = -_dog(L, s)
    if chroma:
        a = img[..., 0] - img[..., 1]
        b = 0.5 * (img[..., 0] + img[..., 1]) - img[..., 2]
        d = np.maximum(d, chroma * 0.5 * np.maximum(np.abs(_dog(a, s)), np.abs(_dog(b, s))))
    if adapt:
        ref = float(np.percentile(d[::3, ::3], 99.0))
        k = min(max(ref / (hi * 2.0), adapt * 0.25), 1.0)
        lo, hi = lo * k, hi * k
    return smoothstep(lo, hi, d)


def _kmeans(img, k, seed=0, iters=8, n=16000):
    """Palette of ``k`` colours (k-means++ on a pixel sample)."""
    flat = np.asarray(img, F32)
    flat = flat.reshape(-1, flat.shape[-1])
    rng = np.random.default_rng(seed)
    samp = flat[rng.integers(0, flat.shape[0], min(n, flat.shape[0]))].astype(np.float64)
    c = samp[[int(rng.integers(len(samp)))]]
    for _ in range(1, k):
        d = ((samp[:, None, :] - c[None]) ** 2).sum(-1).min(1)
        tot = d.sum()
        pick_i = int(rng.integers(len(samp))) if tot <= 1e-12 else int(rng.choice(len(samp), p=d / tot))
        c = np.vstack([c, samp[pick_i]])
    for _ in range(iters):
        lab = ((samp[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = samp[m].mean(0)
    return c.astype(F32)


def _assign(img, pal):
    """Nearest palette index per pixel, (H, W) int32."""
    img = np.asarray(img, F32)
    best = None
    lab = np.zeros(img.shape[:2], np.int32)
    for j, p in enumerate(np.asarray(pal, F32)):
        d = ((img - p) ** 2).sum(-1)
        if best is None:
            best = d
        else:
            m = d < best
            best = np.where(m, d, best)
            lab[m] = j
    return lab


def _smooth_labels(lab, k, sigma):
    """Majority-vote smoothing of a label map (rounded region outlines, no speckles)."""
    if sigma <= 0.3:
        return lab
    best = np.full(lab.shape, -1.0, F32)
    out = lab.copy()
    for j in range(int(k)):
        m = lab == j
        if not m.any():
            continue
        b = _blur(m.astype(F32), sigma)
        sel = b > best
        best = np.where(sel, b, best)
        out[sel] = j
    return out


def _segment(img, k, S, seed=0, smooth=2.5, pre=3.0):
    """Edge-preserving smoothing + k-means + majority smoothing → (base, palette, labels)."""
    base = kuwahara(img, max(pre * S, 1))
    if pre >= 2:
        base = kuwahara(base, max(pre * 0.5 * S, 1))
    step = max(int(round(2 * S)), 1)
    pal = _kmeans(base[::step, ::step], k, seed)
    lab = _smooth_labels(_assign(base, pal), k, smooth * S)
    return base, pal, lab


def _region_smooth(img, lab, k, sigma):
    """Colour averaged locally *within* each region (normalised blur per label): flat but local, never mixing
    neighbouring regions."""
    out = np.zeros_like(img)
    for j in range(int(k)):
        m = (lab == j)
        if not m.any():
            continue
        mf = m.astype(F32)
        num = _blur(img * mf[..., None], sigma)
        den = _blur(mf, sigma)[..., None]
        out = np.where(m[..., None], num / np.maximum(den, 1e-4), out)
    return out


def _boundary(lab, pal=None, lo=0.1, hi=0.22):
    """1 on pixels whose 4-neighbour has a different label. With ``pal``, the boundary is weighted by how
    different the two colours are, so steps inside a smooth gradient draw no line."""
    if pal is None:
        b = np.zeros(lab.shape, bool)
        dx = lab[:, 1:] != lab[:, :-1]
        b[:, 1:] |= dx
        b[:, :-1] |= dx
        dy = lab[1:, :] != lab[:-1, :]
        b[1:, :] |= dy
        b[:-1, :] |= dy
        return b.astype(F32)
    pal = np.asarray(pal, F32)
    diff = np.sqrt(((pal[:, None] - pal[None]) ** 2).sum(-1))
    w = smoothstep(lo, hi, diff)
    b = np.zeros(lab.shape, F32)
    wx = w[lab[:, 1:], lab[:, :-1]]
    b[:, 1:] = np.maximum(b[:, 1:], wx)
    b[:, :-1] = np.maximum(b[:, :-1], wx)
    wy = w[lab[1:, :], lab[:-1, :]]
    b[1:, :] = np.maximum(b[1:, :], wy)
    b[:-1, :] = np.maximum(b[:-1, :], wy)
    return b


def _dilate(mask, r):
    """Grow a 0~1 mask by ``r`` px (max filter) with a soft edge."""
    r = float(r)
    m = np.asarray(mask, F32)
    n = int(round(r))
    if n >= 1:
        from PIL import ImageFilter
        im = Image.fromarray((clamp01(m) * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(2 * n + 1))
        m = np.asarray(im, F32) / 255.0
    return clamp01(_blur(m, 0.6) * 1.3)


def _line(mask_b, width):
    """Boundary pixels → an anti-aliased line of about ``width`` px."""
    return _dilate(mask_b, max(width * 0.5 - 0.5, 0))


def _region_mean(img, ids, n):
    ids = ids.ravel()
    cnt = np.bincount(ids, minlength=n).astype(np.float64)
    cols = np.stack([np.bincount(ids, img[..., c].ravel(), minlength=n) for c in range(3)], -1)
    return (cols / np.maximum(cnt, 1)[:, None]).astype(F32), cnt


def _halftone(cover, X, Y, spacing, angle_deg=45.0):
    """Amplitude-modulated dot screen: ``cover`` (0~1 ink fraction) → anti-aliased ink mask."""
    a = math.radians(angle_deg)
    k = 2 * math.pi / spacing
    u = (X * math.cos(a) + Y * math.sin(a)) * k
    v = (Y * math.cos(a) - X * math.sin(a)) * k
    f = 0.5 + 0.25 * (np.cos(u) + np.cos(v))
    fw = 0.5 * math.pi / spacing
    return clamp01((f - (1.0 - np.asarray(cover, F32))) / fw + 0.5)


def _shift(a, dx, dy):
    """Shift an array by whole pixels (edge pixels repeat)."""
    dx, dy = int(round(dx)), int(round(dy))
    H, W = a.shape[:2]
    ys = np.clip(np.arange(H) - dy, 0, H - 1)
    xs = np.clip(np.arange(W) - dx, 0, W - 1)
    return a[ys][:, xs]


def _sat(img, s):
    m = luma(img)[..., None]
    return m + (img - m) * s


def _gradmap(L, stops):
    return mix(stops, clamp01(L))


def _cells(H, W, cell, seed=0, jitter=0.9, reach=1, points=None):
    """Voronoi cells of a jittered grid. Returns (d1, d2, id, px, py): distance to the nearest / second point,
    nearest cell id (index into px.ravel()), and the point arrays. ``points`` = (px, py) overrides the jitter."""
    m = reach
    gh, gw = int(H // cell) + 2 * m + 2, int(W // cell) + 2 * m + 2
    if points is None:
        rng = np.random.default_rng(seed)
        jy, jx = np.mgrid[0:gh, 0:gw].astype(F32)
        px = ((jx - m + 0.5 + (rng.random((gh, gw), dtype=np.float32) - 0.5) * jitter) * cell).astype(F32)
        py = ((jy - m + 0.5 + (rng.random((gh, gw), dtype=np.float32) - 0.5) * jitter) * cell).astype(F32)
    else:
        px, py = points
    X, Y = grid(H, W)
    cx0 = (X // cell).astype(np.int64) + m
    cy0 = (Y // cell).astype(np.int64) + m
    fpx, fpy = px.ravel(), py.ravel()
    d1 = np.full((H, W), np.inf, F32)
    d2 = np.full((H, W), np.inf, F32)
    i1 = np.zeros((H, W), np.int64)
    for oy in range(-m, m + 1):
        for ox in range(-m, m + 1):
            idx = (cy0 + oy) * gw + (cx0 + ox)
            d = np.hypot(X - fpx[idx], Y - fpy[idx])
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            i1 = np.where(closer, idx, i1)
            d1 = np.where(closer, d, d1)
    return d1, d2, i1, px, py


def _dabs(H, W, cell, length, width, angle_of, seed=0, keep=1.0, reach=1):
    """Elliptical brush dabs on a jittered grid, each with its own angle and size.
    Returns (d, id, along, across, alive) — d < 1 inside the winning dab (normalised distance)."""
    m = reach
    gh, gw = int(H // cell) + 2 * m + 2, int(W // cell) + 2 * m + 2
    rng = np.random.default_rng(seed)
    jy, jx = np.mgrid[0:gh, 0:gw].astype(F32)
    px = ((jx - m + rng.random((gh, gw), dtype=np.float32)) * cell).ravel()
    py = ((jy - m + rng.random((gh, gw), dtype=np.float32)) * cell).ravel()
    ang = np.asarray(angle_of(px, py), F32).ravel()
    sz = (0.75 + 0.5 * rng.random(gh * gw)).astype(F32)
    alive = rng.random(gh * gw) < keep
    L = np.where(alive, length * sz, 1e-3).astype(F32)
    Wd = np.where(alive, width * sz, 1e-3).astype(F32)
    ca, sa = np.cos(ang), np.sin(ang)
    X, Y = grid(H, W)
    cx0 = (X // cell).astype(np.int64) + m
    cy0 = (Y // cell).astype(np.int64) + m
    best = np.full((H, W), np.inf, F32)
    bid = np.zeros((H, W), np.int64)
    bal = np.zeros((H, W), F32)
    bac = np.zeros((H, W), F32)
    for oy in range(-m, m + 1):
        for ox in range(-m, m + 1):
            idx = (cy0 + oy) * gw + (cx0 + ox)
            dx = X - px[idx]
            dy = Y - py[idx]
            c, s = ca[idx], sa[idx]
            al = (dx * c + dy * s) / L[idx]
            ac = (dy * c - dx * s) / Wd[idx]
            d = al * al + ac * ac
            sel = d < best
            best = np.where(sel, d, best)
            bid = np.where(sel, idx, bid)
            bal = np.where(sel, al, bal)
            bac = np.where(sel, ac, bac)
    return np.sqrt(best), bid, bal, bac, (px, py, Wd)


def _at(a, px, py):
    """Nearest-pixel lookup of a map at point coordinates (clamped)."""
    H, W = a.shape[:2]
    return a[np.clip(np.asarray(py).astype(np.int64), 0, H - 1), np.clip(np.asarray(px).astype(np.int64), 0, W - 1)]


def _flow_angle(img, S, sigma=6.0, seed=0, fallback=0.6):
    """Angle field along edges where the image has structure, a slow noisy default elsewhere."""
    vx, vy, coh = flow_field(img, sigma=sigma * S + 1)
    H, W = img.shape[:2]
    ang = np.arctan2(vy, vx)
    wander = fallback + 0.5 * fbm2d(H, W, 200 * S + 4, 2, seed=seed + 77)
    return np.where(coh > 0.25, ang, wander).astype(F32)


def _dist_to(mask, factor):
    """Approximate distance (px) to the nearest ``mask`` pixel — jump flooding on a ``factor``-downsampled grid."""
    H, W = mask.shape
    f = max(int(factor), 1)
    h, w = -(-H // f), -(-W // f)
    pad = np.zeros((h * f, w * f), bool)
    pad[:H, :W] = mask > 0.5
    small = pad.reshape(h, f, w, f).any(axis=(1, 3))
    Ys, Xs = np.mgrid[0:h, 0:w].astype(F32)
    BIG = F32(1e7)
    sx = np.where(small, Xs, BIG).astype(F32)
    sy = np.where(small, Ys, BIG).astype(F32)
    step = 1
    while step * 2 < max(h, w):
        step *= 2
    while step >= 1:
        bd = (Xs - sx) ** 2 + (Ys - sy) ** 2
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if (dx == 0 and dy == 0) or abs(dy) >= h or abs(dx) >= w:
                    continue
                tx = np.full_like(sx, BIG)
                ty = np.full_like(sy, BIG)
                ys0, ys1 = max(dy, 0), h + min(dy, 0)
                xs0, xs1 = max(dx, 0), w + min(dx, 0)
                tx[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx] = sx[ys0:ys1, xs0:xs1]
                ty[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx] = sy[ys0:ys1, xs0:xs1]
                d = (Xs - tx) ** 2 + (Ys - ty) ** 2
                sel = d < bd
                sx = np.where(sel, tx, sx)
                sy = np.where(sel, ty, sy)
                bd = np.where(sel, d, bd)
        step //= 2
    d = np.sqrt((Xs - sx) ** 2 + (Ys - sy) ** 2) * f
    return resize(d.astype(F32), W, H, "bilinear")


def _seal(out, S, color=(0.72, 0.13, 0.1), seed=0, glyph="印"):
    """Red artist's seal stamped in the lower right corner."""
    H, W = out.shape[:2]
    s = int(round(64 * S)) + 6
    x0, y0 = W - int(70 * S) - s, H - int(70 * S) - s
    if x0 < 0 or y0 < 0:
        return out
    X, Y = grid(s, s)
    sq = clamp01(np.minimum(np.minimum(X, s - 1 - X), np.minimum(Y, s - 1 - Y)) + 0.5)
    carve = None
    try:
        from .text import text_mask, find_font
        if find_font(style="cjk") is not None:
            carve = text_mask(s, s, glyph, (s / 2, s / 2), size=s * 0.78, anchor="mm", bold=True)
    except Exception:  # pragma: no cover - font problems fall back to a geometric seal
        carve = None
    inner = clamp01(np.minimum(np.minimum(X, s - 1 - X), np.minimum(Y, s - 1 - Y)) - s * 0.08 + 0.5)
    frame = sq * (1 - inner) + inner * clamp01(np.minimum(np.minimum(X, s - 1 - X), np.minimum(Y, s - 1 - Y)) - s * 0.13 + 0.5)
    if carve is None:
        carve = clamp01(1 - np.abs(np.abs(X - s / 2) - s * 0.18) / (s * 0.05)) * (np.abs(Y - s / 2) < s * 0.3)
        carve = np.maximum(carve, clamp01(1 - np.abs(Y - s / 2) / (s * 0.05)) * (np.abs(X - s / 2) < s * 0.3))
    ink = frame * (1 - clamp01(carve * 1.2) * inner)
    rough = fbm2d(s, s, 3 * S + 1, 2, seed=seed + 5)
    ink = ink * smoothstep(-0.6, -0.1, rough) * 0.92
    reg = out[y0:y0 + s, x0:x0 + s]
    out[y0:y0 + s, x0:x0 + s] = reg * (1 - ink[..., None]) + reg * np.asarray(color, F32) * ink[..., None] * 1.05
    return out


# =================================================================================== drawing media

def colored_pencil(img, stroke=1.0, pressure=1.0, outline=1.0, seed=0):
    """Coloured pencil: hatched colour strokes on white paper, paper tooth sparkling through, pencil outlines."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = _blur(kuwahara(img, max(2 * S, 1)), 0.8 * S)
    L = luma(base)
    col = clamp01(_sat(base, 1.45))
    col = 1 - (1 - col) * 0.88
    sat = base.max(-1) - base.min(-1)
    need = clamp01(0.22 + (1 - L) * 0.7 + sat * 0.6) * pressure
    tooth = _tooth(H, W, S, seed)
    h1 = _streaks(X, Y, 0.9, 30 * S * stroke, 1.5 * S * stroke + 0.5, seed + 1) * 0.5 + 0.5
    h2 = _streaks(X, Y, -0.5, 30 * S * stroke, 1.5 * S * stroke + 0.5, seed + 2) * 0.5 + 0.5
    t1 = 0.6 * h1 + 0.4 * tooth
    t2 = 0.6 * h2 + 0.4 * tooth
    d1 = smoothstep(1.0 - need * 0.8 - 0.08, 1.0 - need * 0.8 + 0.08, t1)
    d2 = smoothstep(1.45 - need * 0.8 - 0.08, 1.45 - need * 0.8 + 0.08, t2)
    cover = (1 - (1 - d1) * (1 - d2)) * smoothstep(0.12, 0.42, tooth)
    paper = _paper(H, W, S, "#fbf8f1", seed, 0.03)
    out = paper * (1 - cover[..., None] * (1 - col) * 0.97)
    out = out * (1 - (d1 * d2)[..., None] * 0.18)                       # burnished overlap gets a bit darker
    if outline:
        e = _edges(base, S, width=1.0, lo=0.012, hi=0.04) * smoothstep(0.25, 0.6, tooth * 0.6 + h1 * 0.6)
        out = out * (1 - e[..., None] * outline * 0.75 * (1 - col * 0.45))
    return clamp01(out)


def charcoal(img, contrast=1.35, smudge=1.0, paper="#dcd6ca", seed=0):
    """Charcoal: broad smudged darks on toned paper, grainy tooth, bold broken contours, chalk-lifted lights."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    L0 = luma(_blur(img, 1.5 * S))
    lo, hi = np.percentile(L0[::4, ::4], [2, 98])
    L = clamp01(((L0 - lo) / max(hi - lo, 1e-3) - 0.5) * contrast + 0.55)
    dark = 1 - L
    tooth = _tooth(H, W, S, seed, scale=1.7)
    ang = 0.7 + 0.6 * fbm2d(H, W, 260 * S + 4, 2, seed=seed + 3)
    st = _streaks_flow(X, Y, ang, 80 * S, 6 * S + 1, seed + 4) * 0.5 + 0.5
    cov = smoothstep(0.18, 0.92, dark + 0.38 * (tooth - 0.5) + 0.22 * (st - 0.5))
    cov = np.maximum(cov * 0.9, _blur(cov, 7 * S * smudge + 0.5) * 0.85)
    e = _edges(_blur(img, 1.2 * S), S, width=2.2, lo=0.012, hi=0.04) * smoothstep(0.3, 0.65, tooth * 0.7 + st * 0.5)
    cov = np.maximum(cov, e * 0.95)
    cov = cov * (0.85 + 0.15 * smoothstep(0.2, 0.8, tooth))
    chalk = smoothstep(0.82, 0.97, L) * smoothstep(0.35, 0.7, tooth)
    pap = _paper(H, W, S, paper, seed, 0.06)
    out = pap * (1 - cov[..., None]) + np.array([0.075, 0.07, 0.075], F32) * cov[..., None]
    out = out + (np.array([0.97, 0.96, 0.94], F32) - out) * (chalk * 0.75)[..., None]
    return clamp01(out)


def ink_wash(img, tones=3, bleed=1.0, dry=1.0, seal=True, paper="#f3eee2", seed=0):
    """Sumi-e ink wash: a few graded ink tones with wet blooming edges, dry-brush contours, empty rice paper,
    and a red seal (``seal=False`` to omit)."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = _blur(kuwahara(img, max(3 * S, 1)), 2 * S)
    L = luma(base)
    lo, hi = np.percentile(L[::4, ::4], [3, 97])
    dark = clamp01(1 - (L - lo) / max(hi - lo, 1e-3))
    n = fbm2d(H, W, 30 * S + 2, 3, seed=seed + 1) * 0.07 * bleed
    ink = np.zeros((H, W), F32)
    tones = max(int(tones), 1)
    for i in range(tones):
        t = 0.3 + 0.58 * i / max(tones - 1, 1)
        ink += smoothstep(t - 0.02, t + 0.02, dark + n)
    ink = ink / tones
    rim = clamp01((ink - _blur(ink, 3 * S + 0.5)) * 3.0)              # pigment pools at the wet edge
    ink = clamp01(ink * 0.82 + rim * 0.35)
    ang = _flow_angle(img, S, 5, seed)
    drs = _streaks_flow(X, Y, ang, 26 * S, 1.3 * S + 0.4, seed + 2) * 0.5 + 0.5
    e = _edges(base, S, width=1.7, lo=0.012, hi=0.04) * smoothstep(0.32, 0.6, drs)
    ink = np.maximum(ink, e * 0.95 * dry)
    fib = fbm2d(H, W, 40 * S + 2, 2, seed=seed + 5, aspect=0.12) * 0.5 + 0.5
    ink = ink * (0.9 + 0.1 * fib)
    pap = _paper(H, W, S, paper, seed, 0.045)
    out = pap * (1 - ink[..., None]) + np.array([0.06, 0.06, 0.07], F32) * ink[..., None]
    if seal:
        out = _seal(out, S, seed=seed)
    return clamp01(out)


def pen_ink(img, spacing=7.0, line=1.0, outline=1.0, seed=0):
    """Pen-and-ink: confident contour lines + hand cross-hatching in tonal tiers, black ink on white paper."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = kuwahara(img, max(2 * S, 1))
    L = luma(_blur(base, 1.5 * S))
    lo, hi = np.percentile(L[::4, ::4], [2, 98])
    tone = clamp01(1 - (L - lo) / max(hi - lo, 1e-3))
    sp = spacing * S + 1.5
    lw = 0.55 * S * line + 0.45
    wob = fbm2d(H, W, 70 * S + 2, 2, seed=seed + 1) * 1.5 * S

    def hatch(angle, thr, k):
        a = math.radians(angle)
        u = X * math.cos(a) + Y * math.sin(a)
        v = Y * math.cos(a) - X * math.sin(a) + wob
        lane = np.floor(v / sp)
        fv = (v / sp - lane - 0.5) * sp
        seglen = 11 * sp
        off = _hash(lane, 1, seed + k) * seglen
        segi = np.floor((u + off) / seglen)
        fr = (u + off) / seglen - segi
        r = _hash(lane, segi, seed + k)
        t = thr + (r - 0.5) * 0.14
        on = smoothstep(t, t + 0.05, tone)
        ends = smoothstep(0.0, 0.05, fr) * smoothstep(1.0, 0.95, fr)
        return clamp01(lw * (0.8 + 0.5 * on) - np.abs(fv) + 0.5) * on * ends

    h = np.maximum(np.maximum(hatch(45, 0.2, 1), hatch(-45, 0.45, 2)), hatch(5, 0.68, 3))
    solid = smoothstep(0.9, 0.97, tone)
    e = _edges(base, S, width=1.15 * outline, lo=0.012, hi=0.03) * (outline > 0)
    ink = np.maximum(np.maximum(h, solid), e)
    pap = _paper(H, W, S, "#fbfaf6", seed, 0.02)
    return clamp01(pap * (1 - ink[..., None]) + np.array([0.05, 0.05, 0.06], F32) * ink[..., None])


def engraving(img, spacing=8.0, angle=-28.0, relief=1.0, ink="#15191c", paper="#f2ecdc", seed=0):
    """Engraving / etching: parallel lines that swell with darkness and bow over the forms (banknote style),
    a cross-cut in the shadows, fine contour lines."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = _blur(img, 1.0 * S)
    L = luma(base)
    lo, hi = np.percentile(L[::4, ::4], [1, 99])
    tone = clamp01(1 - (L - lo) / max(hi - lo, 1e-3)) ** 1.2
    big = _blur(L, 22 * S + 1)
    sp = spacing * S + 1.2
    a = math.radians(angle)
    v = Y * math.cos(a) - X * math.sin(a) + relief * big * 55 * S + fbm2d(H, W, 300 * S + 8, 2, seed=seed) * 10 * S
    fv = (v / sp - np.floor(v / sp) - 0.5) * sp
    half = 0.5 * sp * (0.08 + 0.8 * tone)
    l1 = clamp01(half - np.abs(fv) + 0.5)
    a2 = a + math.radians(62)
    v2 = Y * math.cos(a2) - X * math.sin(a2) + relief * big * 30 * S
    fv2 = (v2 / sp - np.floor(v2 / sp) - 0.5) * sp
    half2 = 0.5 * sp * clamp01((tone - 0.55) / 0.45) * 0.9
    l2 = clamp01(half2 - np.abs(fv2) + 0.5)
    e = _edges(base, S, width=0.8, lo=0.014, hi=0.04, chroma=0.3)
    amt = np.maximum(np.maximum(l1, l2), e * 0.85)
    pap = _paper(H, W, S, paper, seed, 0.035)
    return clamp01(pap * (1 - amt[..., None]) + rgb(ink) * amt[..., None])


def pastel(img, paper="#9a8f7e", coverage=1.0, stroke=1.0, seed=0):
    """Soft pastel / chalk: powdery strokes on toned paper, the paper tooth showing through, smudged blends."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = _blur(kuwahara(img, max(3 * S, 1)), 2 * S)
    col = clamp01(0.1 + _sat(base, 1.3) * 0.95) * 0.88 + 0.1
    p = rgb(paper)
    need = clamp01(np.abs(col - p).mean(-1) * 2.5)
    c0 = clamp01((0.36 + 0.4 * need) * coverage)
    tooth = _tooth(H, W, S, seed, scale=1.0)
    ang = 0.8 + 0.9 * fbm2d(H, W, 240 * S + 4, 2, seed=seed + 3)
    st = _streaks_flow(X, Y, ang, 40 * S * stroke, 4.5 * S * stroke + 0.8, seed + 4) * 0.5 + 0.5
    t = 0.68 * tooth + 0.32 * st
    col = clamp01(col * (1 + 0.14 * (st[..., None] - 0.5) * np.array([1.0, 0.6, -0.4], F32)))
    dep = smoothstep(1 - c0 - 0.07, 1 - c0 + 0.07, t)
    dep = np.maximum(dep, _blur(dep, 1.5 * S) * 0.55)
    pap = _paper(H, W, S, paper, seed, 0.08)
    out = pap * (1 - dep[..., None]) + col * dep[..., None]
    dust = smoothstep(0.86, 0.98, tooth) * dep * 0.12
    return clamp01(out + dust[..., None])


def storybook(img, softness=1.0, outline=1.0, warmth=1.0, paper="#f3e6cb", seed=0):
    """Storybook / picture-book illustration: soft painterly shading, gentle brown outlines, warm paper
    with a faint tooth and a light vignette."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    soft = kuwahara(img, max(int(round(2.5 * S * softness)), 1))
    soft = soft * 0.65 + _blur(img, 1.6 * S * softness) * 0.35
    soft = clamp01(_sat(soft, 1.04))
    soft = 0.05 + soft * 0.9                                           # lifted blacks, nothing is pure ink
    tint = np.array([1.0, 0.965, 0.89], F32) ** warmth
    out = soft * tint
    if outline:
        e = _blur(_edges(soft, S, width=1.3 * outline, adapt=1.0), 0.6 * S + 0.3)
        brown = np.array([0.30, 0.20, 0.15], F32)
        ink = clamp01(out * 0.35 + brown * 0.65)
        out = out * (1 - 0.55 * e[..., None]) + ink * (0.55 * e[..., None])
    pap = _paper(H, W, S, paper, seed, 0.05)
    tooth = _tooth(H, W, S, seed + 2, scale=1.2)
    out = out * (0.8 + 0.2 * pap / np.maximum(pap.max(), 1e-6)) * (0.97 + 0.06 * tooth[..., None])
    vx = (X / W - 0.5) * 2
    vy = (Y / H - 0.5) * 2
    vig = 1 - 0.14 * smoothstep(0.55, 1.5, np.sqrt(vx * vx + vy * vy))
    out = out * vig[..., None] + pap * 0.04 * (1 - vig[..., None])
    return clamp01(out)


def crayon(img, colors=14, pressure=1.0, outline=1.0, seed=0):
    """Wax crayon: bright simplified colours scribbled back and forth, white paper tooth everywhere, crayon outlines."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base, pal, lab = _segment(img, colors, S, seed, smooth=3.0)
    pal = clamp01(_sat(pal, 1.35))
    flat = pal[lab]
    flat = clamp01(flat * 0.8 + _sat(_blur(base, 3 * S), 1.3) * 0.2)
    ang = 0.6 + (_hash(lab, 3, seed) - 0.5) * 1.6
    scr = _streaks(X, Y, ang, 34 * S, 1.5 * S + 0.5, seed + 1) * 0.5 + 0.5
    tooth = _tooth(H, W, S, seed + 2, scale=1.5)
    Lf = luma(flat)
    press = clamp01(0.5 + 0.5 * (1 - Lf)) * pressure
    cov = smoothstep(0.74 - 0.4 * press, 0.9 - 0.4 * press, 0.6 * scr + 0.55 * tooth)
    wax = flat * (1 + 0.1 * (scr[..., None] - 0.5))
    pap = _paper(H, W, S, "#fcfbf7", seed, 0.02)
    out = pap * (1 - cov[..., None]) + wax * cov[..., None]
    if outline:
        b = _line(_boundary(lab, pal), 2.0 * S * outline + 0.5) * smoothstep(0.3, 0.6, tooth * 0.7 + scr * 0.5)
        out = out * (1 - b[..., None]) + clamp01(flat * 0.45)[...] * b[..., None]
    return clamp01(out)


def chalkboard(img, board="#25332d", color=0.6, frame=True, seed=0):
    """Chalk drawing on a slate board: chalky contour strokes, side-of-chalk shading, erased haze, wooden frame."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = _blur(kuwahara(img, max(2.5 * S, 1)), 1.5 * S)
    L = luma(base)
    lo, hi = np.percentile(L[::4, ::4], [2, 98])
    Ln = clamp01((L - lo) / max(hi - lo, 1e-3))
    tooth = _tooth(H, W, S, seed, scale=1.3)
    st = _streaks(X, Y, 0.45, 30 * S, 2.5 * S + 0.6, seed + 2) * 0.5 + 0.5
    fill = smoothstep(0.5, 0.85, Ln * 0.8 + 0.35 * tooth + 0.3 * st - 0.15) * (0.35 + 0.5 * Ln)
    e = _edges(base, S, width=1.8, lo=0.008, hi=0.025)
    e = np.maximum(e, _edges(1 - base, S, width=1.8, lo=0.008, hi=0.025))
    e = e * smoothstep(0.15, 0.45, tooth)
    chalk = np.maximum(fill, e)
    tint = clamp01(0.62 + _sat(base, 1.6) * 0.5)
    tint = tint * color + np.array([0.94, 0.94, 0.92], F32) * (1 - color)
    haze = smoothstep(0.1, 0.9, fbm2d(H, W, 160 * S + 4, 3, seed=seed + 4) * 0.5 + 0.5) * 0.08
    bd = _paper(H, W, S, board, seed, 0.07) * (1 + haze[..., None] * 1.6)
    out = bd * (1 - chalk[..., None]) + tint * chalk[..., None]
    if frame:
        fw = 22 * S + 2
        d = np.minimum(np.minimum(X, W - 1 - X), np.minimum(Y, H - 1 - Y))
        fm = clamp01(fw - d + 0.5)
        wood = rgb("#7a5233") * (1 + 0.18 * fbm2d(H, W, 90 * S + 4, 3, seed=seed + 6, aspect=0.06))[..., None]
        bevel = clamp01(1 - np.abs(d - fw * 0.5) / fw) * 0.25
        out = out * (1 - fm[..., None]) + wood * (0.85 + bevel[..., None]) * fm[..., None]
        out = out * (1 - (clamp01(fw * 1.6 - d) * (1 - fm) * 0.3)[..., None])
    return clamp01(out)


# =================================================================================== paint media

def gouache(img, colors=12, gap=1.0, border=0.03, paper="#f6f1e6", seed=0):
    """Gouache / poster paint: flat opaque matte shapes, faint brush texture, slivers of paper between shapes,
    a clean taped margin."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base, pal, lab = _segment(img, colors, S, seed, smooth=3.0, pre=4.0)
    means, _ = _region_mean(base, lab, len(pal))
    flat = means[lab]
    flat = clamp01(_sat(0.05 + flat * 0.9, 1.05))
    ang = _hash(lab, 5, seed) * math.pi
    st = _streaks(X, Y, ang, 30 * S, 3.5 * S + 0.6, seed + 1)
    mot = fbm2d(H, W, 50 * S + 2, 3, seed=seed + 2)
    out = flat * (1 + 0.07 * st[..., None] + 0.03 * mot[..., None])
    pap = _paper(H, W, S, paper, seed, 0.03)
    if gap:
        dry = fbm2d(H, W, 12 * S + 1, 2, seed=seed + 3) * 0.5 + 0.5
        g = _line(_boundary(lab, pal, 0.04, 0.1), 1.6 * S * gap + 0.4) * smoothstep(0.58, 0.72, dry)
        out = out * (1 - g[..., None] * 0.85) + pap * (g[..., None] * 0.85)
    if border:
        d = np.minimum(np.minimum(X, W - 1 - X), np.minimum(Y, H - 1 - Y))
        bm = clamp01(border * H - d + 0.5)
        out = out * (1 - bm[..., None]) + pap * bm[..., None]
    return clamp01(out)


def impressionist(img, brush=20.0, jitter=0.08, ground="#c8ab7e", seed=0):
    """Impressionist: short thick dabs of broken colour that follow the forms, canvas ground glinting between
    them, lit impasto ridges."""
    img, H, W, S = _prep(img)
    vx, vy, coh = flow_field(img, sigma=7 * S + 1)
    ang_map = np.arctan2(vy, vx)
    wander = 0.5 + 0.7 * fbm2d(H, W, 220 * S + 4, 2, seed=seed + 7)
    ang_map = np.where(coh > 0.2, ang_map, wander)
    rng = np.random.default_rng(seed)
    under = _sat(_blur(img, 8 * S + 1), 1.2)
    out = _paper(H, W, S, ground, seed, 0.08) * 0.5 + under * 0.5
    height = np.zeros((H, W), F32)
    b = brush * S + 2
    for li, (cs, bl, keep) in enumerate(((1.0, 5.0, 1.0), (0.7, 2.5, 0.9), (0.5, 1.2, 0.75))):
        src = _sat(_blur(img, bl * S), 1.4)
        cell = b * cs
        d, idx, al, ac, (px, py, wd) = _dabs(H, W, cell, cell * 1.15, cell * 0.42,
                                              lambda x, y: _at(ang_map, x, y), seed=seed + 11 * li, keep=keep)
        col = _at(src, px, py)
        jit = (rng.random((len(px), 3)).astype(F32) - 0.5) * 2 * jitter
        jit += (rng.random((len(px), 1)).astype(F32) - 0.5) * jitter
        col = clamp01(col + jit)
        bristle = 1 + 0.09 * np.sin(ac * 9.0 + _hash(idx, 2, seed) * 6.28)
        aa = 1.5 / np.maximum(wd[idx], 1.0)
        cov = smoothstep(1.0, 1.0 - aa, d)
        if li > 0:                                                    # finer dabs only where there is detail
            det = np.abs(_blur(img, bl * S) - _blur(img, bl * S * 4)).sum(-1)
            cov = cov * smoothstep(0.02, 0.08, det + (2 - li) * 0.03)
        out = out * (1 - cov[..., None]) + col[idx] * bristle[..., None] * cov[..., None]
        height = np.where(cov > 0.5, (1 - d * d) * (0.8 + 0.2 * bristle), height)
    gy, gx = np.gradient(_blur(height, 0.8 * S + 0.3))
    shade = clamp01(0.5 - (gx * 0.6 + gy * 0.8) * 1.2 / (S + 0.3))
    return clamp01(out * (0.88 + 0.24 * shade[..., None]))


def pointillism(img, dot=8.0, ground="#f7f3ea", seed=0):
    """Pointillism (Seurat): small round dots of pure, varied colour that mix in the eye, white ground between."""
    img, H, W, S = _prep(img)
    rng = np.random.default_rng(seed)
    src = _sat(_blur(img, 1.5 * S), 1.45)
    out = _paper(H, W, S, ground, seed, 0.03)
    cell = dot * S + 2.0
    for li in range(3):
        d1, d2, idx, px, py = _cells(H, W, cell, seed=seed + 17 * li, jitter=1.0)
        fx, fy = px.ravel(), py.ravel()
        col = _at(src, fx, fy)
        n = len(fx)
        hue = (rng.random((n, 3)).astype(F32) - 0.5) * 0.24
        col = clamp01(col + hue + (rng.random((n, 1)).astype(F32) - 0.5) * 0.12)
        r = cell * (0.36 + 0.1 * _hash(idx, 4, seed + li))
        cov = clamp01(r - d1 + 0.5)
        if li == 2:
            cov = cov * (_hash(idx, 9, seed) < 0.6)
        out = out * (1 - cov[..., None]) + col[idx] * cov[..., None]
    return clamp01(out)


def oil(img, brush=7.0, impasto=0.6, canvas=0.06, seed=0):
    """Oil / impasto painting: blended colour blocks laid with visible loaded-brush strokes that follow the
    forms, bristle grooves and lit paint ridges, on canvas."""
    img, H, W, S = _prep(img)
    base = oil_paint(img, brush=brush * S + 1, impasto=0.0, seed=seed)
    ang_map = _flow_angle(img, S, 6, seed)
    out = base.copy()
    height = np.zeros((H, W), F32)
    b = brush * S + 1.5
    for li, (cs, keep) in enumerate(((2.8, 1.0), (1.9, 0.85))):
        cell = b * cs
        d, idx, al, ac, (px, py, wd) = _dabs(H, W, cell, cell * 1.2, cell * 0.42,
                                              lambda x, y: _at(ang_map, x, y), seed=seed + 5 + 11 * li, keep=keep)
        src = _blur(base, 0.6 * cell)
        col = _at(src, px, py) * 0.6 + _at(base, px, py) * 0.4
        groove = _hash(np.floor(ac * 6 + _hash(idx, 3, seed) * 6).astype(np.int64), idx, seed + li)
        bristle = 1 + 0.07 * (groove - 0.5) + 0.04 * np.sin(ac * 11.0)
        aa = 1.5 / np.maximum(wd[idx], 1.0)
        tail = smoothstep(1.0, 0.55, np.abs(al))                       # strokes thin out at the ends
        cov = smoothstep(1.0, 1.0 - aa, d) * (0.55 + 0.45 * tail)
        mixc = col[idx] * 0.75 + base * 0.25
        out = out * (1 - cov[..., None]) + mixc * bristle[..., None] * cov[..., None]
        height = np.where(cov > 0.4, (1 - d * d) * 0.7 + 0.3 * groove, height)
    gy, gx = np.gradient(_blur(height, 0.7 * S + 0.3))
    shade = (gx * 0.6 + gy * 0.8) / (S + 0.3)
    out = out * (1 - impasto * 0.35 * np.clip(shade, -1, 1))[..., None]
    out = out + impasto * 0.12 * smoothstep(0.08, 0.25, -shade)[..., None]
    return canvas_texture(clamp01(out), strength=canvas, weave=2.6 * S + 1, seed=seed) if canvas else clamp01(out)


# =================================================================================== prints

def comic(img, colors=8, outline=1.0, dots=9.0, saturation=1.4, whites=True, seed=0):
    """Comic / pop art: bold black outlines, flat saturated fills, Ben-Day halftone dots in light areas and
    shadows; ``whites`` keeps highlights clean white (room for speech bubbles)."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base, pal, lab = _segment(img, colors, S, seed, smooth=2.0)
    pal = clamp01(_sat(pal, saturation))
    flat = pal[lab]
    L = luma(_blur(base, 2 * S))
    Lp = luma(pal)
    sp = dots * S + 2
    light = smoothstep(0.55, 0.7, Lp)[lab]
    dense = clamp01(1.0 - (Lp - 0.55) / 0.45 * 0.75)[lab]
    dot_cov = _halftone(dense * 0.6 + 0.12, X, Y, sp, 45)
    pure = clamp01(_sat(flat, 1.6) * 0.95)
    out = flat * (1 - light[..., None]) + light[..., None] * (1 - dot_cov[..., None] * (1 - pure))
    shade = clamp01((luma(flat) - L) * 5.0)
    sd = _halftone(shade * 0.45, X, Y, sp * 0.8, 15)
    out = out * (1 - sd[..., None] * 0.8)
    if whites:
        wm = smoothstep(0.88, 0.94, L)
        out = out * (1 - wm[..., None]) + wm[..., None]
    e = _edges(base, S, width=2.4 * outline, lo=0.008, hi=0.03, chroma=0.8)
    e = np.maximum(e, _line(_boundary(lab, pal), 1.4 * S * outline) * 0.85)
    out = out * (1 - e[..., None]) + np.array([0.04, 0.04, 0.05], F32) * e[..., None]
    return clamp01(out)


def manga(img, outline=1.0, tone=6.0, seed=0):
    """Manga: black-and-white ink lines, cut screentone dots for the mid tones, solid blacks, clean white paper."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = kuwahara(img, max(2 * S, 1))
    L = luma(_blur(base, 1.2 * S))
    lo, hi = np.percentile(L[::4, ::4], [2, 98])
    Ln = clamp01((L - lo) / max(hi - lo, 1e-3))
    sp = tone * S + 1.8
    black = smoothstep(0.15, 0.12, Ln)
    lev1 = smoothstep(0.66, 0.68, 1 - Ln)                              # light tone
    lev2 = smoothstep(0.38, 0.40, 1 - Ln)
    grad = clamp01((0.8 - Ln) / 0.65)
    t_light = _halftone(0.16 + 0.0 * grad, X, Y, sp, 45)
    t_dark = _halftone(0.42, X, Y, sp, 45)
    t_grad = _halftone(grad * 0.55, X, Y, sp, 45)
    flatness = smoothstep(0.004, 0.0015, np.hypot(*np.gradient(_blur(Ln, 3 * S))))
    tone_ink = np.where(lev2 > 0.5, t_dark, t_light) * lev1
    tone_ink = tone_ink * flatness + t_grad * (1 - flatness) * smoothstep(0.82, 0.78, Ln)
    e = _edges(base, S, width=1.1 * outline, lo=0.01, hi=0.03)
    ink = np.maximum(np.maximum(tone_ink, black), e)
    return clamp01(np.array([0.975, 0.97, 0.96], F32) * (1 - ink[..., None]) + 0.03 * ink[..., None])


_UKIYO_PIGMENTS = ["#1d3c6e", "#2f4f6f", "#5f86a8", "#a9c1cf", "#c2452d", "#d98b6c", "#d9a441", "#6e8b4e",
                   "#3d5a3a", "#2b2522", "#e9dcbf", "#b77a4b", "#5d566e", "#8c8070", "#c96f7d"]


def ukiyoe(img, colors=9, keyline=1.0, bokashi=1.0, paper="#efe3c6", cartouche=True, seed=0):
    """Ukiyo-e woodblock print: flat areas in traditional pigments, a Prussian-blue bokashi sky, bokashi at the
    top of each block, sumi key-lines, wood grain, washi paper and a title cartouche (``cartouche`` = its text,
    or False)."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base, pal, lab = _segment(img, colors, S, seed, smooth=3.5, pre=4.0)
    pig = np.stack([rgb(c) for c in _UKIYO_PIGMENTS])
    near = pig[((pal[:, None] - pig[None]) ** 2).sum(-1).argmin(1)]
    pal2 = clamp01(pal * 0.55 + near * 0.45)
    pal2 = clamp01(_sat(0.06 + pal2 * 0.94, 0.9))
    flat = pal2[lab]
    if bokashi:
        yy = np.arange(H, dtype=np.int64)[:, None] * np.ones((1, W), np.int64)
        change = np.zeros((H, W), bool)
        change[0] = True
        change[1:] = lab[1:] != lab[:-1]
        start = np.maximum.accumulate(np.where(change, yy, 0), axis=0)
        endc = np.zeros((H, W), bool)
        endc[-1] = True
        endc[:-1] = lab[1:] != lab[:-1]
        end = np.minimum.accumulate(np.where(endc, yy, H - 1)[::-1], axis=0)[::-1]
        dist = (yy - start).astype(F32)
        tall = smoothstep(0.12 * H, 0.3 * H, (end - start).astype(F32))       # only big blocks get a bokashi
        bk = _blur(np.exp(-dist / (0.09 * H)) * tall, 5 * S + 1) * bokashi
        deep = clamp01(flat * 0.55 + np.array([0.03, 0.06, 0.16], F32) * 0.3)
        flat = flat + (deep - flat) * (bk * 0.55)[..., None]
    if bokashi:                                                       # the classic deep-blue band at the top
        top = np.exp(-(np.arange(H, dtype=F32) / H) / 0.16)[:, None, None] * 0.75 * bokashi
        flat = flat * (1 - top) + np.array([0.07, 0.16, 0.36], F32) * top
    flat = flat * 0.88 + rgb(paper) * 0.12
    grain = fbm2d(H, W, 260 * S + 8, 3, seed=seed + 1, aspect=0.025)
    mot = fbm2d(H, W, 40 * S + 2, 3, seed=seed + 2)
    flat = flat * (1 + 0.04 * grain[..., None] + 0.035 * mot[..., None])
    pap = _paper(H, W, S, paper, seed, 0.05)
    fib = smoothstep(0.55, 0.9, fbm2d(H, W, 30 * S + 2, 2, seed=seed + 4, aspect=0.1) * 0.5 + 0.5) * 0.05
    out = flat * (pap / rgb(paper)) * 0.97 + pap * 0.03 + fib[..., None]
    if keyline:
        rough = fbm2d(H, W, 6 * S + 1, 2, seed=seed + 5) * 0.5 + 0.5
        k = _line(_boundary(lab, pal), 1.6 * S * keyline + 0.4) * smoothstep(0.15, 0.35, rough)
        k = np.maximum(k, _edges(base, S, width=1.2, lo=0.02, hi=0.05) * 0.9)
        out = out * (1 - k[..., None] * 0.92) + np.array([0.13, 0.11, 0.1], F32) * k[..., None] * 0.92
    if cartouche:
        out = _cartouche(out, S, str(cartouche) if isinstance(cartouche, str) else "富嶽景", seed)
    return clamp01(out)


def _cartouche(out, S, text, seed=0):
    """Ukiyo-e title cartouche: a pale vertical panel with a key-line frame and vertical text, plus a seal."""
    H, W = out.shape[:2]
    bw, bh = int(46 * S) + 8, int(min(len(text), 4) * 46 * S + 30 * S) + 12
    x0, y0 = W - int(60 * S) - bw, int(48 * S) + 4
    if x0 < 0 or y0 + bh > H:
        return out
    X, Y = grid(bh, bw)
    d = np.minimum(np.minimum(X, bw - 1 - X), np.minimum(Y, bh - 1 - Y))
    panel = clamp01(d + 0.5)
    lw = 2.2 * S + 0.6
    frame = clamp01(lw - np.abs(d - lw * 1.2) + 0.5)
    glyphs = np.zeros((bh, bw), F32)
    try:
        from .text import text_mask, find_font
        if find_font(style="cjk") is not None:
            for i, ch in enumerate(text[:4]):
                glyphs = np.maximum(glyphs, text_mask(bh, bw, ch, (bw / 2, 15 * S + 6 + (i + 0.5) * 46 * S),
                                                      size=34 * S + 4, anchor="mm", bold=True))
    except Exception:  # pragma: no cover - no CJK font: plain panel
        pass
    reg = out[y0:y0 + bh, x0:x0 + bw]
    pc = np.array([0.91, 0.84, 0.62], F32) * (1 + 0.03 * fbm2d(bh, bw, 8 * S + 1, 2, seed=seed + 9))[..., None]
    reg = reg * (1 - panel[..., None]) + pc * panel[..., None]
    ink = np.maximum(frame, glyphs * 0.95)
    out[y0:y0 + bh, x0:x0 + bw] = reg * (1 - ink[..., None]) + np.array([0.12, 0.1, 0.09], F32) * ink[..., None]
    return out


_INK_LUTS = {}


def _ink_lut(ink, paper):
    """Coverage of each spot ink (0~1) that best reproduces every colour of a 32^3 grid when the inks are
    overprinted (multiplied) on ``paper`` — solved once per ink set by projected gradient descent."""
    key = (np.asarray(ink, F32).tobytes(), np.asarray(paper, F32).tobytes())
    if key in _INK_LUTS:
        return _INK_LUTS[key]
    g = np.linspace(0, 1, 32, dtype=np.float64)
    t = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) / np.maximum(paper, 1e-3)
    t = np.clip(t, 0.0, 1.0)
    k = 1.0 - np.asarray(ink, np.float64)                                 # (n, 3) light each ink removes
    n = len(k)
    A = -np.log(np.clip(1 - k, 0.03, 1.0))
    m = np.clip(-np.log(np.clip(t, 0.03, 1)) @ np.linalg.pinv(A), 0, 1)
    for _ in range(160):
        d = np.maximum(1 - m[:, :, None] * k[None], 1e-4)                 # (N, n, 3)
        f = np.prod(d, axis=1)                                            # printed colour
        grad = ((f - t)[:, None, :] * -k[None] * (f[:, None, :] / d)).sum(-1)
        m = np.clip(m - 0.8 * grad, 0, 1)
    lut = m.reshape(32, 32, 32, n).astype(F32)
    _INK_LUTS[key] = lut
    return lut


def risograph(img, inks=("#ff48b0", "#0078bf"), grain=1.0, misregister=1.0, paper="#f5f0e6", seed=0):
    """Risograph: the picture separated into 2–3 spot inks (default fluorescent pink + blue), grainy
    stochastic screening, uneven ink and slightly misregistered layers, overprinted on off-white paper."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    p = rgb(paper)
    ink = np.stack([rgb(c) for c in inks])
    lut = _ink_lut(ink, p)
    lift = clamp01(_blur(img, 0.8 * S)) ** 0.7                       # riso prints read light and bright
    q = np.clip(np.rint(lift * 31), 0, 31).astype(np.int64)
    cov = lut[q[..., 0], q[..., 1], q[..., 2]]
    rng = np.random.default_rng(seed)
    out = _paper(H, W, S, paper, seed, 0.025)
    for i in range(len(inks)):
        c = cov[..., i] * (0.9 + 0.12 * fbm2d(H, W, 90 * S + 2, 3, seed=seed + 10 + i))
        n = _blur(rng.random((H, W), dtype=np.float32), 0.6 * S + 0.3)
        n = (n - n.mean()) / (n.std() + 1e-6)
        m = _halftone(clamp01(c + n * 0.12 * grain), X, Y, 6.5 * S + 2, (15, 75, 0, 45)[i % 4])
        m = m * (1 - smoothstep(0.6, 1.0, n * 0.4 * grain + 0.5) * 0.35)
        dx, dy = (rng.random(2) - 0.5) * 12 * S * misregister
        m = _shift(m, dx, dy)
        out = out * (1 - m[..., None] * (1 - ink[i]))
    return clamp01(out)


def linocut(img, threshold=0.5, gouge=1.0, ink="#151312", paper="#efe8d8", color=None, seed=0):
    """Linocut / woodcut: carved two-tone print — solid black masses, white gouge lines that follow the forms
    in the mid tones, rough cut edges, uneven ink; ``color`` adds a second flat colour block."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = _blur(kuwahara(img, max(3 * S, 1)), 2 * S)
    L = luma(base)
    lo, hi = np.percentile(L[::4, ::4], [2, 98])
    Ln = clamp01((L - lo) / max(hi - lo, 1e-3)) + (0.5 - threshold)
    tone = clamp01(1 - Ln)
    big = _blur(L, 18 * S + 1)
    sp = 11 * S * gouge + 2
    ang = _flow_angle(img, S, 10, seed)
    a = 0.35
    v = Y * math.cos(a) - X * math.sin(a) + big * 70 * S + fbm2d(H, W, 160 * S + 4, 2, seed=seed) * 14 * S
    fv = (v / sp - np.floor(v / sp) - 0.5) * sp
    rough = fbm2d(H, W, 2.5 * S + 1, 2, seed=seed + 1) * 1.2 * S
    half = 0.5 * sp * smoothstep(0.22, 0.85, tone)
    stripes = clamp01(half - np.abs(fv) + rough + 0.5)
    solid = smoothstep(0.7, 0.74, tone + rough * 0.02)
    inkm = np.maximum(stripes, solid)
    e = _edges(base, S, width=2.2, lo=0.01, hi=0.03, chroma=0.3)
    inkm = np.maximum(inkm, e)
    del ang
    tooth = _tooth(H, W, S, seed + 3, scale=1.5)
    inkm = inkm * (1 - smoothstep(0.78, 0.95, tooth) * 0.5)
    pap = _paper(H, W, S, paper, seed, 0.04)
    out = pap
    if color is not None:
        cm = smoothstep(0.3, 0.4, tone) * smoothstep(0.8, 0.7, tone)
        out = out * (1 - cm[..., None] * (1 - rgb(color)))
    return clamp01(out * (1 - inkm[..., None]) + rgb(ink) * inkm[..., None])


# =================================================================================== crafts / cells

def stained_glass(img, cell=40.0, lead=1.0, glow=1.0, colors=10, seed=0):
    """Stained glass: pieces cut along the picture's regions and subdivided into cells, dark lead cames,
    saturated back-lit glass that glows brighter in the middle of each piece, mottled glass texture."""
    img, H, W, S = _prep(img)
    base, pal, lab = _segment(img, colors, S, seed, smooth=5.0)
    c = cell * S + 4
    d1, d2, cid, px, py = _cells(H, W, c, seed=seed, jitter=0.9)
    k = len(pal)
    ncell = px.size
    pid = cid * k + lab
    cnt = np.bincount(pid.ravel(), minlength=ncell * k)
    major = cnt.reshape(ncell, k).argmax(1)
    small = cnt[pid] < 0.12 * c * c                                 # slivers join their cell's main piece
    lab = np.where(small, major[cid], lab)
    pid = cid * k + lab
    uniq, inv = np.unique(pid.ravel(), return_inverse=True)
    inv = inv.reshape(H, W)
    cols, _ = _region_mean(base, inv, len(uniq))
    v = cols.max(-1, keepdims=True)
    cols = clamp01(_sat(cols / np.maximum(v, 1e-3) * (0.38 + 0.62 * v), 1.6))   # back-lit: dark glass still glows
    glass = cols[inv]
    lw = 1.5 * S * lead + 0.7
    bdist = (d2 - d1) * 0.5
    seg_line = _line(_boundary(lab), 2 * lw)
    leadm = np.maximum(clamp01(lw - bdist + 0.5), seg_line)
    inner = smoothstep(0, c * 0.4, bdist) * (1 - _blur(seg_line, 4 * S + 1))
    tex = 1 + 0.1 * fbm2d(H, W, 14 * S + 1, 3, seed=seed + 2) + 0.06 * fbm2d(H, W, 80 * S + 2, 2, seed=seed + 3, aspect=0.2)
    glass = glass * (0.72 + 0.4 * inner * glow)[..., None] * tex[..., None]
    halo = _blur(glass * (1 - leadm)[..., None], 4 * S + 1)
    ridge = clamp01(1 - np.abs(_blur(leadm, 0.7 * S) - 0.7) * 3)
    leadc = np.array([0.06, 0.06, 0.07], F32) + ridge[..., None] * 0.12
    out = glass * (1 - leadm[..., None]) + leadc * leadm[..., None]
    return clamp01(out + halo * (1 - leadm[..., None]) * 0.2 * glow)


def mosaic(img, tile=17.0, grout="#cfc8b8", seed=0):
    """Mosaic: small stone / glass tesserae laid in rows that follow the outlines of the picture (andamento),
    each tile one colour with a bevel, light grout between."""
    img, H, W, S = _prep(img)
    t = tile * S + 3
    base, pal, lab = _segment(img, 8, S, seed, smooth=3.0)
    edge = _boundary(lab)
    edge[0, :] = edge[-1, :] = 1
    edge[:, 0] = edge[:, -1] = 1
    dist = _dist_to(edge, max(int(t / 4), 1))
    gy, gx = np.gradient(_blur(dist, t * 0.3))
    gn = np.hypot(gx, gy) + 1e-6
    m = 2
    gh, gw = int(H // t) + 2 * m + 2, int(W // t) + 2 * m + 2
    rng = np.random.default_rng(seed)
    jy, jx = np.mgrid[0:gh, 0:gw].astype(F32)
    px = ((jx - m + 0.5 + (rng.random((gh, gw), dtype=np.float32) - 0.5) * 0.7) * t).astype(F32)
    py = ((jy - m + 0.5 + (rng.random((gh, gw), dtype=np.float32) - 0.5) * 0.7) * t).astype(F32)
    dpt = _at(dist, px, py)
    target = (np.floor(dpt / t) + 0.5) * t
    move = np.clip(dpt - target, -0.5 * t, 0.5 * t)
    ux, uy = _at(gx / gn, px, py), _at(gy / gn, px, py)
    px, py = px - ux * move, py - uy * move
    far = np.zeros(px.shape, bool)                                  # drop points that landed on a neighbour
    for oy, ox in ((0, 1), (1, -1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, -1), (1, -2), (2, 2), (2, -2)):
        qx = np.roll(np.roll(px, -oy, 0), -ox, 1)
        qy = np.roll(np.roll(py, -oy, 0), -ox, 1)
        far |= np.hypot(px - qx, py - qy) < 0.55 * t
    px = np.where(far, F32(-1e6), px)
    py = np.where(far, F32(-1e6), py)
    d1, d2, cid, _, _ = _cells(H, W, t, points=(px, py), reach=m)
    uniq, inv = np.unique(cid.ravel(), return_inverse=True)
    inv = inv.reshape(H, W)
    cols, _ = _region_mean(base, inv, len(uniq))
    var = (rng.random((len(uniq), 1)).astype(F32) - 0.5) * 0.12
    cols = clamp01(_sat(cols, 1.35) * (1 + var))
    tilec = cols[inv]
    bd = (d2 - d1) * 0.5
    gw_ = 0.9 * S + 0.5
    gm = clamp01(gw_ - bd + 0.5)
    bevel = smoothstep(0, 0.3 * t, bd)
    tex = 1 + 0.06 * fbm2d(H, W, 4 * S + 1, 2, seed=seed + 3)
    out = tilec * (0.78 + 0.22 * bevel)[..., None] * tex[..., None]
    gc = _paper(H, W, S, grout, seed, 0.08)
    return clamp01(out * (1 - gm[..., None]) + gc * gm[..., None])


def low_poly(img, cell=46.0, facet=0.06, seed=0):
    """Low-poly: the picture rebuilt from flat triangles whose corners snap to real edges, each facet one colour
    with slight shading."""
    img, H, W, S = _prep(img)
    t = cell * S + 6
    rng = np.random.default_rng(seed)
    nx, ny = int(math.ceil(W / t)) + 1, int(math.ceil(H / t)) + 1
    gxs = np.linspace(0, W, nx, dtype=F32)
    gys = np.linspace(0, H, ny, dtype=F32)
    PX, PY = np.meshgrid(gxs, gys)
    cw, ch = W / (nx - 1), H / (ny - 1)
    Lb = luma(_blur(img, 1.5 * S))
    gy_, gx_ = np.gradient(Lb)
    G = np.hypot(gx_, gy_)
    best = np.full(PX.shape, -1.0, F32)
    bx, by = PX.copy(), PY.copy()
    for oy in np.linspace(-0.38, 0.38, 6):
        for ox in np.linspace(-0.38, 0.38, 6):
            qx = PX + ox * cw
            qy = PY + oy * ch
            g = _at(G, qx, qy) + rng.random(PX.shape).astype(F32) * 0.002
            sel = g > best
            best = np.where(sel, g, best)
            bx = np.where(sel, qx, bx)
            by = np.where(sel, qy, by)
    jx = PX + (rng.random(PX.shape).astype(F32) - 0.5) * 0.5 * cw
    jy = PY + (rng.random(PY.shape).astype(F32) - 0.5) * 0.5 * ch
    strong = best > np.percentile(best, 45)
    PXn = np.where(strong, bx, jx)
    PYn = np.where(strong, by, jy)
    PXn[:, 0], PXn[:, -1], PYn[0, :], PYn[-1, :] = 0, W, 0, H
    PXn = np.clip(PXn, 0, W)
    PYn = np.clip(PYn, 0, H)
    tris = []
    src = _blur(img, 2 * S)
    for j in range(ny - 1):
        for i in range(nx - 1):
            p00 = (PXn[j, i], PYn[j, i])
            p10 = (PXn[j, i + 1], PYn[j, i + 1])
            p01 = (PXn[j + 1, i], PYn[j + 1, i])
            p11 = (PXn[j + 1, i + 1], PYn[j + 1, i + 1])
            ca = np.abs(_at(src, p00[0], p00[1]) - _at(src, p11[0], p11[1])).sum()
            cb = np.abs(_at(src, p10[0], p10[1]) - _at(src, p01[0], p01[1])).sum()
            if ca <= cb:
                tris += [(p00, p10, p11), (p00, p11, p01)]
            else:
                tris += [(p00, p10, p01), (p10, p11, p01)]
    idim = Image.new("I", (W, H), 0)
    dr = ImageDraw.Draw(idim)
    for k, tri in enumerate(tris):
        dr.polygon([(float(x), float(y)) for x, y in tri], fill=k)
    ids = np.asarray(idim, np.int64)
    cols, cnt = _region_mean(img, ids, len(tris))
    miss = cnt == 0
    if miss.any():
        cen = np.array([[sum(p[0] for p in tr) / 3, sum(p[1] for p in tr) / 3] for tr in tris], F32)
        cols[miss] = _at(img, cen[miss, 0], cen[miss, 1])
    cols = clamp01(cols * (1 + (rng.random((len(tris), 1)).astype(F32) - 0.5) * 2 * facet))
    big = Image.new("RGB", (W * 2, H * 2))
    d2 = ImageDraw.Draw(big)
    c8 = (cols * 255 + 0.5).astype(np.uint8)
    for k, tri in enumerate(tris):
        d2.polygon([(float(x) * 2, float(y) * 2) for x, y in tri], fill=tuple(int(v) for v in c8[k]))
    return np.asarray(big.resize((W, H), getattr(Image, "Resampling", Image).BOX), F32) / 255.0


PALETTES = {
    "pico8": ["#000000", "#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f", "#c2c3c7", "#fff1e8",
              "#ff004d", "#ffa300", "#ffec27", "#00e436", "#29adff", "#83769c", "#ff77a8", "#ffccaa"],
    "gameboy": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"],
    "nes": ["#000000", "#fcfcfc", "#bcbcbc", "#7c7c7c", "#a80020", "#f83800", "#fca044", "#f8b800",
            "#00a800", "#58d854", "#0058f8", "#3cbcfc", "#6844fc", "#d800cc", "#f878f8", "#ac7c00"],
    "c64": ["#000000", "#ffffff", "#880000", "#aaffee", "#cc44cc", "#00cc55", "#0000aa", "#eeee77",
            "#dd8855", "#664400", "#ff7777", "#333333", "#777777", "#aaff66", "#0088ff", "#bbbbbb"],
}


def _palette_of(img, k, seed=0, levels=24, power=0.5):
    """Palette that keeps small, distinct colours (highlights, eyes, a red apple on a big wall): weighted
    k-means over the image's colour histogram, each bin weighted by count**power instead of count."""
    flat = clamp01(np.asarray(img, F32).reshape(-1, 3))
    q = np.minimum((flat * levels).astype(np.int64), levels - 1)
    key = (q[:, 0] * levels + q[:, 1]) * levels + q[:, 2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.stack([np.bincount(inv, flat[:, c], minlength=len(uniq)) for c in range(3)], -1)
    pts = (sums / cnt[:, None]).astype(np.float64)
    wts = cnt.astype(np.float64) ** power
    k = int(min(k, len(pts)))
    rng = np.random.default_rng(seed)
    c = pts[[int(np.argmax(wts))]]
    for _ in range(1, k):                                          # weighted k-means++ (farthest-first flavour)
        d = ((pts[:, None, :] - c[None]) ** 2).sum(-1).min(1) * wts
        c = np.vstack([c, pts[int(np.argmax(d)) if rng.random() < 0.5 else int(rng.choice(len(pts), p=d / d.sum()))]])
    for _ in range(12):
        lab = ((pts[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = (pts[m] * wts[m, None]).sum(0) / wts[m].sum()
    return c.astype(F32)


def pixel_art(img, pixel=12.0, colors=16, dither=0.35, outline=1.0, palette=None, seed=0):
    """Pixel art: a coarse grid of crisp nearest-neighbour pixels in a small palette that keeps the highlights,
    ordered dithering in the gradients and dark sprite outlines in each colour's own hue.

    pixel: pixel size in 1080p px. colors: palette size. palette: a list of colours or a preset name
    ("pico8", "gameboy", "nes", "c64") instead of a palette taken from the picture. outline: 0 off."""
    img, H, W, S = _prep(img)
    p = max(int(round(pixel * S)), 1)
    h, w = -(-H // p), -(-W // p)
    pad = np.pad(_blur(img, 0.15 * p) if p > 2 else img, ((0, h * p - H), (0, w * p - W), (0, 0)), mode="edge")
    small = pad.reshape(h, p, w, p, 3).mean(axis=(1, 3))
    small = clamp01(_sat(small, 1.2))
    m = luma(small).mean()
    small = clamp01(m + (small - m) * 1.08)
    if palette is not None:
        if isinstance(palette, str) and palette not in PALETTES:
            raise _unknown_name("palette", palette, list(PALETTES))
        pal = np.array([rgb(c) for c in (PALETTES[palette] if isinstance(palette, str) else palette)], F32)
    else:
        pal = _palette_of(small, colors, seed)
    if dither:
        bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], F32) / 16.0 - 0.47
        th = np.tile(bayer, (h // 4 + 1, w // 4 + 1))[:h, :w]
        dist = np.sqrt(((pal[:, None] - pal[None]) ** 2).sum(-1)) + np.eye(len(pal)) * 9
        nn = dist.min(1)                                         # gap to each colour's nearest neighbour
        lab0 = _assign(small, pal)
        gap = np.minimum(nn[lab0], 0.25)
        flat = _blur(np.abs(small - _blur(small, 1.0)).sum(-1), 2.5) < 0.02   # no dither near hard edges
        sm = _blur(small, 2.0)
        gy, gx = np.gradient(luma(sm))
        ramp = smoothstep(0.002, 0.008, np.hypot(gx, gy))         # and none on flat fills (no lone dots)
        small = small + (th * gap * dither * 1.6 * flat * ramp)[..., None]
    lab = _assign(small, pal)
    q = pal[lab]
    if outline:
        Lq = luma(q)
        Lp = luma(pal)
        dist = np.sqrt(((pal[:, None] - pal[None]) ** 2).sum(-1))
        strong = np.zeros((h, w), bool)
        glint = (Lp > 0.72) & (pal.max(-1) - pal.min(-1) < 0.3)    # highlights get no dark ring around them
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            nl = _shift(lab, dx, dy)
            diag = dx != 0 and dy != 0
            strong |= (dist[lab, nl] > (0.42 if diag else 0.3) / outline) & (Lp[nl] > Lq + 0.04) & ~glint[nl]
        darker = pal[_assign(q * 0.42, pal)] if palette is not None else q * 0.42
        q = np.where(strong[..., None], darker, q)
    up = np.repeat(np.repeat(q, p, 0), p, 1)[:H, :W]
    return clamp01(up)


def paper_cut(img, layers=6, shadow=1.0, seed=0):
    """Paper cut / shadow box: the picture split into stacked sheets of coloured paper with smooth cut edges,
    each casting a soft shadow onto the sheets behind it. Every colour region of the picture is its own piece of
    paper; light regions sit at the back, dark ones in front (``layers`` = number of sheet depths)."""
    img, H, W, S = _prep(img)
    n = max(int(layers), 2)
    k = int(min(max(2 * n, 8), 16))
    base, pal, lab = _segment(img, k, S, seed, smooth=5.0, pre=4.0)
    v = pal.max(-1, keepdims=True)
    paper = clamp01(_sat(pal / np.maximum(v, 1e-3) * (0.12 + 0.88 * v), 1.25) * 0.96 + 0.03)   # matte card stock
    # depth: lighter colours further back; ties broken by height (lower in the picture = nearer)
    Yc = np.repeat(grid(H, W)[1][..., None] / H, 3, -1)
    ymean, cnt = _region_mean(Yc, lab, k)
    score = luma(pal) * 0.8 + (1 - ymean[:, 0]) * 0.2
    order = np.argsort(np.argsort(-score))                       # 0 = furthest back
    depth_of = np.minimum((order * n) // max(int((cnt > 0).sum()), 1), n - 1)
    depth_of = np.where(cnt > 0, depth_of, 0)
    dep = depth_of[lab].astype(F32)
    col = paper[lab]
    fib = fbm2d(H, W, 1.4 * S + 0.6, 2, seed=seed + 7)
    out = None
    for i in range(n):
        if i == 0:
            out = col * (1 + 0.03 * fib)[..., None]
            continue
        ind = (dep >= i).astype(F32)
        if not ind.any():
            continue
        m = clamp01((_blur(ind, 2.5 * S + 0.5) - 0.5) * (1.6 * S + 1.0) + 0.5)        # smooth knife-cut edge
        sh = _blur(_shift(m, 5 * S + 1, 9 * S + 1), 9 * S + 1) * (1 - m) * 0.62 * shadow
        out = out * (1 - sh[..., None])
        lit = clamp01(m - _shift(m, 1.4 * S + 0.6, 1.4 * S + 0.6)) * 0.22
        sheet = col * (1 + 0.035 * _shift(fib, 17 * i, 11 * i))[..., None] + lit[..., None]
        out = out * (1 - m[..., None]) + sheet * m[..., None]
    return clamp01(out)


def collage(img, pieces=7, offset=1.0, seed=0):
    """Torn-paper collage: the picture's regions torn from different papers (halftone print, newsprint, plain
    stock), white fibrous torn edges, slightly shifted pieces with shadows, on kraft card."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base, pal, lab = _segment(img, pieces, S, seed, smooth=4.0)
    rng = np.random.default_rng(seed)
    nA = fbm2d(H, W, 36 * S + 2, 4, seed=seed + 1) * 0.16
    nB = fbm2d(H, W, 5 * S + 1, 2, seed=seed + 2) * 0.05
    fib = fbm2d(H, W, 1.6 * S + 0.6, 2, seed=seed + 3) * 0.5 + 0.5
    out = _paper(H, W, S, "#b39468", seed, 0.08)
    areas = np.bincount(lab.ravel(), minlength=len(pal))
    detail = _blur(img, 1.2 * S)
    for k, i in enumerate(np.argsort(-areas)):
        if areas[i] == 0:
            continue
        m = _blur((lab == i).astype(F32), 6 * S + 0.5)
        ox, oy = (rng.random(2) - 0.5) * 2 * 60 * S
        n = _shift(nA, ox, oy) + nB
        inner = smoothstep(0.40, 0.43, m + n)                    # pieces overlap their neighbours: no gaps
        outer = smoothstep(0.25, 0.29, m + n + 0.1 * (fib - 0.5))
        dx, dy = (rng.random(2) - 0.5) * 2 * 10 * S * offset
        inner, outer = _shift(inner, dx, dy), _shift(outer, dx, dy)
        kind = k % 4
        c = pal[i]
        tint = 1 + (rng.random(3).astype(F32) - 0.5) * 0.16
        content = clamp01((_shift(detail, dx, dy) * 0.35 + c * 0.65) * tint)
        if kind == 1:
            dots = _halftone(clamp01(1 - luma(content)) * 0.8, X, Y, 6 * S + 2, 30)
            content = clamp01(_sat(content, 1.2) * 1.15) * (1 - dots[..., None] * 0.45)
        elif kind == 2:
            lines = clamp01(np.abs(((Y + oy) / (5 * S + 1.5)) % 1.0 - 0.5) * 4 - 1.2)
            content = content * (0.9 + 0.1 * lines[..., None])
        elif kind == 3:
            content = clamp01(_sat(content, 0.7) * 0.95 + 0.05)
        sh = _blur(_shift(outer, 3 * S, 5 * S), 5 * S + 1) * 0.5
        out = out * (1 - sh[..., None])
        rim = np.array([0.96, 0.95, 0.92], F32) * (0.92 + 0.08 * fib[..., None])
        out = out * (1 - outer[..., None]) + rim * outer[..., None]
        out = out * (1 - inner[..., None]) + content * inner[..., None]
    return clamp01(out)


def cross_stitch(img, stitch=15.0, colors=20, fabric="#efe9dc", seed=0):
    """Embroidery / cross-stitch: one X of twisted thread per cell on Aida cloth, a limited thread palette,
    pale cells left unstitched."""
    img, H, W, S = _prep(img)
    c = stitch * S + 3
    h, w = int(math.ceil(H / c)), int(math.ceil(W / c))
    small = _sat(resize(img, w, h, "box"), 1.15)
    pal = _kmeans(small, colors, seed)
    q = pal[_assign(small, pal)]
    fab = rgb(fabric)
    skip = np.abs(q - fab).max(-1) < 0.07
    X, Y = grid(H, W)
    gx, gy = X / c, Y / c
    ix = np.clip(gx.astype(np.int64), 0, w - 1)
    iy = np.clip(gy.astype(np.int64), 0, h - 1)
    fx, fy = gx - np.floor(gx), gy - np.floor(gy)
    col = q[iy, ix]
    on = (~skip[iy, ix]).astype(F32)
    hw = 0.3
    dA = np.abs(fx + fy - 1) / 1.4142
    dB = np.abs(fx - fy) / 1.4142
    tA = (fx - fy + 1) * 0.5                                          # position along each thread, 0~1
    tB = (fx + fy) * 0.5
    pA = smoothstep(0.0, 0.3, np.minimum(tA, 1 - tA))                 # threads pinch into the holes
    pB = smoothstep(0.0, 0.3, np.minimum(tB, 1 - tB))
    hA = hw * (0.35 + 0.65 * pA)
    hB = hw * (0.35 + 0.65 * pB)
    aa = 1.0 / c
    cA = clamp01((hA - dA) / aa + 0.5) * on
    cB = clamp01((hB - dB) / aa + 0.5) * on
    sA = np.sqrt(clamp01(1 - (dA / hA) ** 2)) * (0.65 + 0.35 * pA) * (1 - 0.3 * (fx + fy - 1) / 1.4142 / hA)
    sB = np.sqrt(clamp01(1 - (dB / hB) ** 2)) * (0.65 + 0.35 * pB) * (1 + 0.3 * (fx - fy) / 1.4142 / hB)
    twA = 1 + 0.16 * np.sin(((fx - fy) * 5 + dA * 9) * 2 * math.pi)
    twB = 1 + 0.16 * np.sin(((fx + fy) * 5 + dB * 9) * 2 * math.pi)
    weave = 1 + 0.05 * np.sin(X * math.pi / (c / 4)) * np.sin(Y * math.pi / (c / 4))
    corner = np.hypot(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    hole = smoothstep(0.13, 0.07, corner) * 0.35
    cloth = _paper(H, W, S, fabric, seed, 0.03) * (weave * (1 - hole))[..., None]
    out = cloth * (1 - cA[..., None] * 0.95)
    out = out * (1 - cA[..., None]) + col * (0.6 + 0.45 * sA * twA)[..., None] * cA[..., None]
    shB = clamp01((hw * 1.5 - dB) / (hw * 0.5)) * cA * (1 - cB) * 0.3
    out = out * (1 - shB[..., None])
    out = out * (1 - cB[..., None]) + col * (0.65 + 0.45 * sB * twB)[..., None] * cB[..., None]
    return clamp01(out)


def clay(img, colors=10, softness=1.0, seed=0):
    """Claymation / soft plasticine: simplified rounded blobs of saturated clay, soft studio light with a gentle
    sheen, darker creases where pieces meet, faint fingerprint texture."""
    img, H, W, S = _prep(img)
    base = kuwahara(kuwahara(img, max(4 * S, 1)), max(2 * S, 1))
    Lb = luma(base)
    # pieces are cut by hue, not by light: the shadow side of a red ball is still the same red lump of clay
    feat = np.concatenate([base / (Lb[..., None] + 0.05) * 0.16, Lb[..., None] * 0.35], -1)
    step = max(int(round(2 * S)), 1)
    fpal = _kmeans(feat[::step, ::step], colors, seed)
    lab = _smooth_labels(_assign(feat, fpal), len(fpal), 6.0 * S)
    pal, _ = _region_mean(base, lab, len(fpal))
    hue = clamp01(_sat(pal, 1.35))
    hue = hue / np.maximum(luma(hue)[:, None], 0.04)                       # colour per unit of light
    Lr, _ = _region_mean(np.repeat(Lb[..., None], 3, -1), lab, len(fpal))
    shade = _blur(Lb, 6 * S + 1)                                            # the form's soft light, kept
    Lmix = Lr[lab, 0] * 0.5 + shade * 0.5
    col = clamp01(_blur(hue[lab], 2.0 * S + 0.5) * (0.06 + 0.94 * Lmix)[..., None])
    # pieces meet only where the colours really differ; steps inside a gradient stay one smooth sheet
    seam = _boundary(lab, fpal, lo=0.065, hi=0.115)
    R = 16 * S * softness + 2
    dist = _dist_to(seam > 0.5, max(int(2 * S), 1))
    dist = _blur(dist, 1.5 * S + 0.5)
    t = clamp01(dist / R)
    hgt = np.sqrt(1 - (1 - t) ** 2) * R                                    # rounded bead profile at each seam
    lump = fbm2d(H, W, 120 * S + 4, 2, seed=seed + 1)
    thumb = fbm2d(H, W, 14 * S + 2, 2, seed=seed + 2)
    hgt = hgt + lump * 3 * S + np.sin(thumb * 30) * 0.12 * S               # hand-pressed lumps + fingerprints
    gy, gx = np.gradient(_blur(hgt, 1.2 * S + 0.3))
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1)
    nx, ny = -gx * nz, -gy * nz
    lx, ly, lz = -0.42, -0.55, 0.72
    ndl = clamp01(nx * lx + ny * ly + nz * lz)
    hz = np.array([lx, ly, lz + 1.0])
    hz = hz / np.linalg.norm(hz)
    spec = clamp01(nx * hz[0] + ny * hz[1] + nz * hz[2]) ** 18 * 0.16
    ao = 0.62 + 0.38 * smoothstep(0.0, 0.45, t)
    out = col * ((0.42 + 0.68 * ndl) * ao)[..., None] + spec[..., None] * (0.6 + 0.4 * col)
    return clamp01(out)


def _sticker_cut(out, src, m, b, S, shadow, X, Y):
    """Die-cut one subject (mask ``m``) onto ``out``: soft shadow, white border, the subject, a faint gloss."""
    H, W = m.shape
    ys, xs = np.nonzero(m > 0.02)
    if not len(ys):
        return out
    pad = int(b + 40 * S + 12)
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + 1 + pad, H)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + 1 + pad, W)
    sl = (slice(y0, y1), slice(x0, x1))
    mm = m[sl]
    d = _dist_to(mm > 0.5, 1)
    outer = np.maximum(clamp01(b - d + 0.5), mm)
    sh = _blur(_shift(outer, 4 * S, 9 * S), 10 * S + 1) * 0.45 * shadow
    white = np.array([0.985, 0.985, 0.98], F32)
    o = out[sl] * (1 - sh[..., None])
    o = o * (1 - outer[..., None]) + white * outer[..., None]
    o = o * (1 - mm[..., None]) + src[sl] * mm[..., None]
    gx, gy = X[sl], Y[sl]
    gloss = smoothstep(0.0, 0.5, np.sin((gx + gy * 0.6) / (W + H) * 7.0 - 0.6)) * 0.1 * outer
    out[sl] = o + gloss[..., None]
    return out


def sticker(img, mask=None, border=14.0, shadow=1.0, background=None, seed=0):
    """Die-cut sticker: a thick white border, a soft drop shadow and a faint gloss around a subject.

    ``mask`` is the subject (0~1), or a list of masks for several stickers at once. With a mask the rest of the
    picture stays exactly as it is (a sheet of nine stickers is nine masks, or nine calls); pass ``background``
    (a colour) to replace everything outside the cut with a plain sheet instead. Without a mask the whole picture
    becomes one rounded-corner sticker on a plain ``background`` (default "#e8e4dc")."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    b = border * S + 1
    if mask is None:
        sx, sy = 0.86 * W, 0.86 * H
        ox, oy = (W - sx) / 2, (H - sy) / 2
        u = (X - ox) / sx * W
        v = (Y - oy) / sy * H
        src = sample(img, u, v)
        r = 0.06 * H
        qx = np.maximum(np.abs(X - W / 2) - (sx / 2 - r), 0)
        qy = np.maximum(np.abs(Y - H / 2) - (sy / 2 - r), 0)
        sd = np.hypot(qx, qy) - r
        m = clamp01(-sd + 0.5)
        masks = [m]
        out = _paper(H, W, S, background or "#e8e4dc", seed, 0.03)
        src_all = src
    else:
        masks = [mask] if (hasattr(mask, "ndim") and np.ndim(mask) == 2) else list(mask)
        masks = [clamp01(np.asarray(q, F32)) for q in masks]
        for q in masks:
            if q.shape != (H, W):
                raise ValueError("sticker: mask shape %s does not match the picture %s" % (q.shape, (H, W)))
        src_all = img
        out = img.copy() if background is None else _paper(H, W, S, background, seed, 0.03)
    for m in masks:
        out = _sticker_cut(out, src_all, m, b, S, shadow, X, Y)
    return clamp01(out)


# =================================================================================== photographic looks

def blueprint(img, paper="#1d4f91", grid_size=24.0, seed=0):
    """Blueprint: white technical line work (the picture's edges) on Prussian-blue paper with a drafting grid,
    border frame and title block."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    base = kuwahara(img, max(2 * S, 1))
    e = np.maximum(_edges(base, S, width=0.8, lo=0.01, hi=0.03), _edges(1 - base, S, width=0.8, lo=0.01, hi=0.03))
    e = np.maximum(e, _edges(base, S, width=2.2, lo=0.012, hi=0.03) * 0.6)
    bgc = _paper(H, W, S, paper, seed, 0.08)
    g = grid_size * S + 4
    lw = 0.5 * S + 0.35
    minor = np.maximum(clamp01(lw - np.abs((X % g) - g / 2 + g / 2 * 0) + 0.5) * 0, 0)
    fx = np.abs(((X / g) % 1.0) - 0.5) * g
    fy = np.abs(((Y / g) % 1.0) - 0.5) * g
    minor = np.maximum(clamp01(lw - (g / 2 - fx) + 0.5), clamp01(lw - (g / 2 - fy) + 0.5)) * 0.16
    G = g * 5
    fX = np.abs(((X / G) % 1.0) - 0.5) * G
    fY = np.abs(((Y / G) % 1.0) - 0.5) * G
    major = np.maximum(clamp01(lw * 1.6 - (G / 2 - fX) + 0.5), clamp01(lw * 1.6 - (G / 2 - fY) + 0.5)) * 0.3
    m = 28 * S + 4
    d = np.minimum(np.minimum(X - m, W - 1 - m - X), np.minimum(Y - m, H - 1 - m - Y))
    frame = clamp01(1.6 * S + 0.6 - np.abs(d) + 0.5)
    tbw, tbh = 0.22 * W, 0.09 * H
    tx0, ty0 = W - m - tbw, H - m - tbh
    inb = (X > tx0) & (Y > ty0) & (X < W - m) & (Y < H - m)
    tb = (clamp01(1.2 * S + 0.5 - np.abs(X - tx0) + 0.5) * (Y > ty0) * (Y < H - m)
          + clamp01(1.2 * S + 0.5 - np.abs(Y - ty0) + 0.5) * (X > tx0) * (X < W - m)
          + clamp01(0.9 * S + 0.4 - np.abs(Y - (ty0 + tbh * 0.5)) + 0.5) * (X > tx0) * (X < W - m))
    wash = (1 - luma(_blur(img, 3 * S))) * 0.0
    lines = np.maximum.reduce([e * 0.92, minor, major, frame * 0.9, clamp01(tb) * 0.85])
    lines = lines * np.where(inb, 1.0, 1.0)
    del wash
    ink = np.array([0.93, 0.96, 1.0], F32)
    return clamp01(bgc * (1 - lines[..., None]) + ink * lines[..., None])


def cyanotype(img, contrast=1.2, seed=0):
    """Cyanotype: a sun print in Prussian blue and paper white, with brushed emulsion edges and uneven exposure."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    L = luma(_blur(img, 0.8 * S))
    lo, hi = np.percentile(L[::4, ::4], [1, 99])
    Ln = clamp01(((L - lo) / max(hi - lo, 1e-3) - 0.5) * contrast + 0.5)
    Ln = clamp01(Ln + fbm2d(H, W, 120 * S + 4, 3, seed=seed + 1) * 0.05)
    col = mix([(0.0, "#0b2a5b"), (0.45, "#1f5c99"), (0.8, "#8fb3cf"), (1.0, "#eef2ef")], Ln)
    edge = fbm2d(H, W, 25 * S + 2, 4, seed=seed + 2) * 0.03
    sx = fbm2d(H, W, 6 * S + 1, 2, seed=seed + 3, aspect=12) * 0.012
    u = np.minimum(X / W, 1 - X / W)
    v = np.minimum(Y / H, 1 - Y / H)
    emul = smoothstep(0.025, 0.05, np.minimum(u + sx, v + edge * 0.6) + edge)
    pap = _paper(H, W, S, "#f1efe6", seed, 0.04)
    return clamp01(pap * (1 - emul[..., None]) + col * emul[..., None])


def synthwave(img, glow=1.0, seed=0):
    """Neon / synthwave grade: deep indigo shadows, magenta mids, sunset-orange highlights, glowing neon edges,
    bloom and faint scanlines."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    L = luma(img)
    lo, hi = np.percentile(L[::4, ::4], [1, 99.5])
    Ln = clamp01((L - lo) / max(hi - lo, 1e-3))
    g = mix([(0.0, "#0a0420"), (0.3, "#2c0b5e"), (0.55, "#b0218f"), (0.78, "#ff6a3d"), (1.0, "#ffe27a")], Ln)
    out = g * 0.85 + _sat(img, 1.4) * 0.15
    cl = _clean(img, S)
    e = np.maximum(_edges(cl, S, width=1.2, lo=0.03, hi=0.07, adapt=0.4),
                   _edges(1 - cl, S, width=1.2, lo=0.03, hi=0.07, adapt=0.4))
    neon = mix([(0.0, "#22e4ff"), (1.0, "#ff3fd8")], (Y / H) * 0.6 + Ln * 0.4)
    tube = e[..., None] * neon
    out = out * (1 - e[..., None] * 0.7) + tube
    out = out + _blur(tube, 6 * S + 1) * 1.2 * glow + _blur(tube, 22 * S + 2) * 0.8 * glow
    hl = clamp01((luma(out) - 0.7) / 0.3)[..., None]
    out = out + _blur(out * hl, 20 * S + 2) * 0.35 * glow
    scan = 1 - 0.06 * (np.sin(Y * math.pi / (2.0 * S + 1)) ** 2)
    return clamp01(out * scan[..., None])


def vaporwave(img, seed=0):
    """Vaporwave: pastel pink-lavender-teal gradient map, VHS chroma bleed and tracking noise, RGB split,
    CRT scanlines."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    L = luma(img)
    lo, hi = np.percentile(L[::4, ::4], [1, 99])
    Ln = clamp01((L - lo) / max(hi - lo, 1e-3))
    hue = (img[..., 2] - img[..., 0]) * 0.25
    g = mix([(0.0, "#2a1b4f"), (0.3, "#7a5cc4"), (0.55, "#f49ac2"), (0.8, "#8ee6e2"), (1.0, "#fff3f9")],
            clamp01(Ln + hue))
    Lg = luma(g)[..., None]
    chroma = g - Lg
    chroma = np.stack([_blur(chroma[..., c], 0) for c in range(3)], -1)
    ch = np.stack([np.asarray(resize(chroma[..., c], max(W // int(6 * S + 2), 4), H, "box"))
                   for c in range(3)], -1)
    ch = resize(ch, W, H, "bilinear")
    out = Lg + ch
    s = 4 * S + 1
    out = np.stack([_shift(out[..., 0], s, 0), out[..., 1], _shift(out[..., 2], -s, 0)], -1)
    rng = np.random.default_rng(seed)
    yb = rng.random() * 0.7 + 0.15
    band = smoothstep(0.035, 0.0, np.abs(Y / H - yb))
    jitter = (rng.random(H).astype(F32) - 0.5) * 18 * S
    rows = np.clip((X + band * jitter[:, None]).astype(np.int64), 0, W - 1)
    out = out[np.arange(H)[:, None], rows]
    out = out + band[..., None] * (rng.random((H, W, 1)).astype(F32) * 0.25)
    scan = 1 - 0.1 * (np.sin(Y * math.pi / (1.6 * S + 1)) ** 2)
    return clamp01(out * scan[..., None])


def duotone(img, dark="#1d1a4f", light="#ff8a5c", mid=None, contrast=1.1, seed=0):
    """Duotone / tritone: the picture's values mapped onto two (or three, with ``mid``) ink colours."""
    img, H, W, S = _prep(img)
    L = luma(img)
    lo, hi = np.percentile(L[::4, ::4], [1, 99])
    Ln = clamp01(((L - lo) / max(hi - lo, 1e-3) - 0.5) * contrast + 0.5)
    stops = [(0.0, dark), (1.0, light)] if mid is None else [(0.0, dark), (0.5, mid), (1.0, light)]
    return clamp01(mix(stops, Ln))


def vintage(img, fade=1.0, leak=1.0, grain=1.0, seed=0):
    """Vintage film photo: lifted faded blacks, warm cast, soft halation around highlights, an orange light leak,
    film grain, dust and a vignette."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    rng = np.random.default_rng(seed)
    out = _blur(img, 0.7 * S)
    out = _sat(out, 1 - 0.3 * fade)
    out = 0.07 * fade + out * (1 - 0.15 * fade)
    out = out * np.array([1.06, 1.0, 0.86], F32) + np.array([0.0, 0.01, 0.03], F32) * fade
    hl = clamp01((luma(img) - 0.65) / 0.35)[..., None]
    out = out + _blur(img * hl, 10 * S + 1) * np.array([0.5, 0.18, 0.08], F32)
    if leak:
        cx, cy = (0.0 if rng.random() < 0.5 else 1.0) * W, rng.uniform(0.1, 0.6) * H
        r = np.hypot((X - cx) / W, (Y - cy) / H * 0.8)
        lk = np.exp(-(r / 0.28) ** 2)[..., None] * leak
        lc = np.array([1.0, 0.45, 0.12], F32)
        out = 1 - (1 - out) * (1 - lk * lc * 0.8)
    n = rng.normal(0, 1, (H, W)).astype(F32)
    n = _blur(n, 0.7 * S + 0.3)
    n = n / (n.std() + 1e-6)
    out = out + (n * 0.035 * grain)[..., None]
    dust = (rng.random((H, W)) > 0.99985).astype(F32)
    dust = clamp01(_blur(dust, 1.0 * S + 0.4) * 6)
    out = out + dust[..., None] * 0.5
    u, v = X / W - 0.5, Y / H - 0.5
    vig = 1 - 0.55 * smoothstep(0.25, 0.85, np.hypot(u, v * 1.1) * 1.4)
    return clamp01(out * vig[..., None])


def sepia(img, tone="sepia", tarnish=1.0, seed=0):
    """Sepia / daguerreotype: an antique monochrome plate — soft focus, sepia (or ``tone="silver"``) toning,
    oval darkening, tarnished iridescent rim, scratches and dust."""
    from ._args import check_choice
    check_choice("sepia", "tone", tone, ("sepia", "silver"))
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    rng = np.random.default_rng(seed)
    L = luma(img)
    L = L * 0.7 + luma(_blur(img, 3 * S + 0.5)) * 0.3
    lo, hi = np.percentile(L[::4, ::4], [1, 99])
    Ln = clamp01((L - lo) / max(hi - lo, 1e-3)) * 0.88 + 0.06
    stops = ([(0.0, "#1f140c"), (0.5, "#8a6a48"), (1.0, "#f1e2c4")] if tone == "sepia"
             else [(0.0, "#141618"), (0.5, "#7d8186"), (1.0, "#eef0ee")])
    out = mix(stops, Ln)
    u, v = X / W - 0.5, Y / H - 0.5
    r = np.hypot(u * 1.0, v * 1.25) * 1.9
    out = out * (1 - 0.6 * smoothstep(0.55, 1.05, r))[..., None]
    if tarnish:
        ring = smoothstep(0.75, 1.15, r + fbm2d(H, W, 60 * S + 2, 3, seed=seed + 1) * 0.12) * tarnish
        irid = mix([(0.0, "#3a5c8a"), (0.5, "#8a4f7a"), (1.0, "#c7a14a")],
                   clamp01(r - 0.75 + fbm2d(H, W, 40 * S + 2, 2, seed=seed + 2) * 0.3))
        out = out * (1 - ring[..., None] * 0.45) + irid * ring[..., None] * 0.35
    img_s = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img_s)
    for _ in range(int(rng.integers(6, 12))):
        x0 = rng.uniform(0, W)
        y0 = rng.uniform(0, H)
        ln = rng.uniform(0.1, 0.5) * H
        a = rng.uniform(-0.3, 0.3) + math.pi / 2 * (rng.random() < 0.7)
        dr.line([(x0, y0), (x0 + math.cos(a) * ln, y0 + math.sin(a) * ln)], fill=int(rng.uniform(60, 160)),
                width=max(int(S), 1))
    sc = np.asarray(img_s, F32) / 255.0
    dust = clamp01(_blur((rng.random((H, W)) > 0.9998).astype(F32), 1.2 * S + 0.4) * 5)
    out = out + (sc * 0.35 + dust * 0.5)[..., None] * np.array([0.9, 0.85, 0.75], F32)
    return clamp01(out)


def noir(img, blinds=0.5, contrast=1.5, grain=1.0, seed=0):
    """Film noir: hard black-and-white with crushed blacks, venetian-blind light falling across the lit areas,
    heavy vignette and grain."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    L = img[..., 0] * 0.45 + img[..., 1] * 0.45 + img[..., 2] * 0.1
    lo, hi = np.percentile(L[::4, ::4], [2, 99])
    Ln = clamp01((L - lo) / max(hi - lo, 1e-3))
    Ln = clamp01((Ln - 0.5) * contrast + 0.45)
    Ln = Ln * Ln * (3 - 2 * Ln)
    if blinds:
        a = math.radians(-24)
        v = Y * math.cos(a) - X * math.sin(a)
        p = 46 * S + 6
        st = smoothstep(0.38, 0.46, np.abs((v / p) % 1.0 - 0.5))
        pool = smoothstep(0.9, 0.2, np.hypot(X / W - 0.62, (Y / H - 0.45) * 1.2) * 1.6)
        Ln = Ln * (1 - blinds * 0.75 * st * pool) * (0.8 + 0.35 * pool)
    u, v2 = X / W - 0.5, Y / H - 0.5
    Ln = Ln * (1 - 0.65 * smoothstep(0.2, 0.75, np.hypot(u, v2)))
    rng = np.random.default_rng(seed)
    n = _blur(rng.normal(0, 1, (H, W)).astype(F32), 0.6 * S + 0.3)
    Ln = Ln + n / (n.std() + 1e-6) * 0.03 * grain
    out = Ln[..., None] * np.array([1.0, 1.0, 0.98], F32)
    return clamp01(out)


def tilt_shift(img, focus=0.62, band=0.1, blur=12.0, seed=0):
    """Tilt-shift miniature: a sharp horizontal band at ``focus`` (0~1 down the frame), blur growing above
    and below, toy-like saturation and contrast."""
    img, H, W, S = _prep(img)
    Y = np.arange(H, dtype=F32)[:, None] / H
    d = clamp01((np.abs(Y - focus) - band * 0.5) / 0.28)
    levels = [img] + [_blur(img, blur * S * f) for f in (0.33, 0.66, 1.0)]
    t = d * 3
    out = np.zeros_like(img)
    for i, lv in enumerate(levels):
        w = clamp01(1 - np.abs(t - i))
        out += lv * w[..., None]
    out = clamp01(_sat(out, 1.55))
    out = out + 0.35 * (out * out * (3 - 2 * out) - out)
    return clamp01(out * 1.03)


def photo_finish(img, depth=None, focus=0.5, aperture=1.0, aberration=1.0, bloom=1.0, grain=1.0, vignette=1.0,
                 seed=0):
    """Photographic finish for realistic renders: lens blur by ``depth`` (0 near ~ 1 far, optional) around
    ``focus``, chromatic aberration, bloom, film grain and vignette."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    out = img
    if depth is not None:
        dep = np.asarray(depth, F32)
        coc = clamp01(np.abs(dep - focus) * 2.5) * 3
        lv = [img] + [_blur(img, s * S * aperture) for s in (2.0, 5.0, 10.0)]
        out = np.zeros_like(img)
        for i, l in enumerate(lv):
            out += l * clamp01(1 - np.abs(coc - i))[..., None]
    if aberration:
        k = 0.0016 * aberration
        cx, cy = W / 2, H / 2
        r = sample(out[..., 0], cx + (X - cx) * (1 + k), cy + (Y - cy) * (1 + k))
        b = sample(out[..., 2], cx + (X - cx) * (1 - k), cy + (Y - cy) * (1 - k))
        out = np.stack([r, out[..., 1], b], -1)
    if bloom:
        hl = clamp01((luma(out) - 0.7) / 0.3)[..., None]
        out = out + _blur(out * hl, 18 * S + 1) * 0.35 * bloom + _blur(out * hl, 4 * S + 1) * 0.15 * bloom
    if vignette:
        u, v = X / W - 0.5, Y / H - 0.5
        out = out * (1 - 0.28 * vignette * smoothstep(0.3, 0.8, np.hypot(u, v) * 1.25))[..., None]
    if grain:
        rng = np.random.default_rng(seed)
        n = _blur(rng.normal(0, 1, (H, W)).astype(F32), 0.5 * S + 0.25)
        out = out + (n / (n.std() + 1e-6) * 0.014 * grain)[..., None]
    return clamp01(out)


def glitch(img, amount=1.0, seed=0):
    """Digital glitch: displaced horizontal slices, RGB channel split, corrupted macro-blocks, pixel-sorted
    streaks and scanlines."""
    img, H, W, S = _prep(img)
    rng = np.random.default_rng(seed)
    out = img.copy()
    y = 0
    while y < H:
        hgt = int(rng.integers(max(int(4 * S), 2), max(int(60 * S), 4)))
        if rng.random() < 0.45 * amount:
            sh = int(rng.normal(0, 120 * S * amount))
            out[y:y + hgt] = np.roll(out[y:y + hgt], sh, axis=1)
        y += hgt
    s = int(round(12 * S * amount)) + 1
    out = np.stack([np.roll(out[..., 0], s, 1), out[..., 1], np.roll(out[..., 2], -s, 1)], -1)
    b = max(int(16 * S), 4)
    for _ in range(int(26 * amount) + 2):
        bx, by = int(rng.integers(0, max(W - 4 * b, 1))), int(rng.integers(0, max(H - b, 1)))
        bw, bh = int(rng.integers(b, 6 * b)), int(rng.integers(b // 2 + 1, 2 * b))
        blk = out[by:by + bh, bx:bx + bw]
        if blk.size == 0:
            continue
        sx = int(rng.integers(-3 * b, 3 * b))
        src = np.roll(out, sx, axis=1)[by:by + bh, bx:bx + bw]
        q = np.round(src * 4) / 4
        out[by:by + bh, bx:bx + bw] = q if rng.random() < 0.5 else src[:, :, ::-1]
    for _ in range(int(3 * amount) + 1):
        y0 = int(rng.integers(0, H - 1))
        hh = int(rng.integers(max(int(10 * S), 2), max(int(70 * S), 3)))
        x0 = int(rng.integers(0, W // 2))
        seg = out[y0:y0 + hh, x0:]
        if seg.size:
            key = luma(seg)
            order = np.argsort(-key, axis=1, kind="stable")
            out[y0:y0 + hh, x0:] = np.take_along_axis(seg, order[..., None], axis=1)
    Y = np.arange(H, dtype=F32)[:, None]
    scan = 1 - 0.08 * (np.sin(Y * math.pi / (1.5 * S + 1)) ** 2)
    return clamp01(out * scan[..., None])


def thermal(img, hud=True, seed=0):
    """Thermal camera: warm objects glow white-yellow, cool ones sink to blue-black (ironbow palette), soft
    low-resolution sensor look, crosshair and colour scale bar."""
    img, H, W, S = _prep(img)
    X, Y = grid(H, W)
    L = luma(img)
    warm = clamp01(img[..., 0] - img[..., 2] * 0.6 + 0.3)
    heat = L * 0.65 + warm * 0.35
    f = max(int(4 * S), 1)
    heat = resize(resize(heat, max(W // f, 4), max(H // f, 4), "box"), W, H, "bilinear")
    heat = _blur(heat, 2 * S + 0.5)
    lo, hi = np.percentile(heat[::4, ::4], [1, 99.5])
    hn = clamp01((heat - lo) / max(hi - lo, 1e-3))
    stops = [(0.0, "#05031a"), (0.2, "#2b0a6b"), (0.4, "#8a1a8c"), (0.58, "#d8315b"), (0.75, "#f7801e"),
             (0.9, "#ffd23f"), (1.0, "#fffbe6")]
    out = mix(stops, hn)
    if hud:
        c = np.array([0.95, 0.95, 0.95], F32)
        lw = 1.2 * S + 0.5
        cx, cy, r = W / 2, H / 2, 26 * S + 4
        cross = (clamp01(lw - np.abs(X - cx) + 0.5) * (np.abs(Y - cy) < r) * (np.abs(Y - cy) > r * 0.35)
                 + clamp01(lw - np.abs(Y - cy) + 0.5) * (np.abs(X - cx) < r) * (np.abs(X - cx) > r * 0.35))
        bx0, bx1 = W - 40 * S - 10, W - 26 * S - 6
        by0, by1 = 0.2 * H, 0.8 * H
        inbar = (X >= bx0) & (X <= bx1) & (Y >= by0) & (Y <= by1)
        barc = mix(stops, clamp01((by1 - Y) / (by1 - by0)))
        out = np.where(inbar[..., None], barc, out)
        frame = ((np.abs(X - bx0) < lw) | (np.abs(X - bx1) < lw)) & (Y >= by0) & (Y <= by1)
        frame |= ((np.abs(Y - by0) < lw) | (np.abs(Y - by1) < lw)) & (X >= bx0) & (X <= bx1)
        hudm = clamp01(cross + frame)
        out = out * (1 - hudm[..., None]) + c * hudm[..., None]
    return clamp01(out)


def infrared(img, glow=1.0, seed=0):
    """Infrared (Aerochrome-style false colour): foliage turns pink-red, skies deepen to blue-cyan,
    highlights bloom softly."""
    img, H, W, S = _prep(img)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    veg = clamp01((g - np.maximum(r, b) * 0.9) * 4 + (g - b) * 1.5)
    ir = clamp01(luma(img) * 0.8 + veg * 0.6)
    out = np.stack([ir, r * 0.85 + veg * 0.1, g * 0.9], -1)
    out = _sat(out, 1.25)
    hl = clamp01((luma(out) - 0.6) / 0.4)[..., None]
    out = out + _blur(out * hl, 12 * S + 1) * 0.35 * glow
    return clamp01(out)


def anime(img, colors=12, shadow="#6e6290", outline=1.0, saturation=1.35, seed=0):
    """Cel anime: flat base colours, one crisp cool shadow tone and a highlight band, thin dark colour
    outlines, clean saturated palette."""
    img, H, W, S = _prep(img)
    base, pal, lab = _segment(img, colors, S, seed, smooth=4.5, pre=4.0)
    flat = clamp01(_sat(_region_smooth(base, lab, len(pal), 60 * S + 2), saturation) * 1.05 + 0.02)
    L = luma(_blur(base, 4 * S))
    Lm = luma(_blur(base, 40 * S + 2))
    sh = smoothstep(-0.06, -0.08, L - Lm)
    tint = rgb(shadow) / max(float(rgb(shadow).mean()), 1e-3)
    shade = flat * 0.66 * np.sqrt(tint)
    out = flat * (1 - sh[..., None]) + clamp01(shade) * sh[..., None]
    hl = smoothstep(0.1, 0.12, L - Lm) * smoothstep(0.55, 0.65, L)
    out = out + (clamp01(out * 1.15 + 0.1) - out) * hl[..., None]
    e = _edges(base, S, width=1.0 * outline, lo=0.012, hi=0.035)
    e = np.maximum(e, _line(_boundary(lab, pal), 1.0 * S * outline) * 0.6)
    line_c = out * 0.3 + np.array([0.06, 0.04, 0.08], F32)
    return clamp01(out * (1 - e[..., None]) + line_c * e[..., None])


# =================================================================================== registry

def _wrap_watercolor(img, seed=0, **kw):
    img = clamp01(_img(img))
    lifted = clamp01(_sat(1 - (1 - img) * 0.85, 1.25))                 # watercolour keeps its lights
    return watercolor(lifted, seed=seed, **kw)


def _wrap_pencil(img, seed=0, **kw):
    img = clamp01(_img(img))
    S = _S(img)
    img = _clean(img, S)
    # a drawing starts from white paper: stretch the values, lift the mid tones and boost local contrast so the
    # subject separates from a dark background instead of everything turning into hatching
    L = luma(img)
    lo, med, hi = np.percentile(L[::3, ::3], [1, 50, 99.5])
    lo = min(lo, med - 0.3)                                   # a big dark backdrop is not the darkest graphite
    Ln = clamp01((L - lo) / max(hi - lo, 1e-3))
    Ln = clamp01(Ln + (Ln - _blur(Ln, 40 * S + 2)) * 0.9)
    Ln = 1 - (1 - Ln) ** 1.6
    img = clamp01(img * (Ln / np.maximum(L, 1e-3))[..., None])
    out = pencil_sketch(img, seed=seed, **kw)
    return (1 - (1 - out) * 0.86).astype(F32)                           # graphite never reaches pure black


def _style(fn, desc, keywords, **defaults):
    return {"fn": fn, "defaults": defaults, "keywords": tuple(keywords), "desc": desc}


STYLES = {
    "oil": _style(oil, "Oil / impasto painting: Kuwahara colour blocks, brush strokes along the forms, lit relief, canvas weave.",
                  ["oil painting", "oil paint", "oil on canvas", "impasto", "acrylic", "painting", "painted", "brushstroke",
                   "brush stroke", "van gogh", "rembrandt", "old master", "油畫", "油彩", "壓克力", "厚塗", "油絵", "アクリル画",
                   "厚塗り", "絵画"]),
    "watercolor": _style(_wrap_watercolor, "Watercolour: transparent washes, wet pigment-pooled edges, granulation, white paper border.",
                         ["watercolor", "watercolour", "aquarelle", "水彩", "水彩畫", "水彩画", "透明水彩"]),
    "pencil": _style(_wrap_pencil, "Graphite pencil drawing on white paper: values lifted towards the paper, dodge-blend contours, hatching in the shadows.",
                     ["pencil", "graphite", "sketch", "pencil drawing", "drawing", "素描", "鉛筆", "速寫", "草圖", "鉛筆画",
                      "スケッチ", "デッサン"]),
    "colored_pencil": _style(colored_pencil, "Coloured pencil: hatched colour strokes on white paper with paper tooth showing.",
                             ["colored pencil", "coloured pencil", "colour pencil", "color pencil", "prismacolor",
                              "彩色鉛筆", "色鉛筆", "色えんぴつ"]),
    "charcoal": _style(charcoal, "Charcoal: smudged broad darks on toned paper, grainy tooth, bold broken contours.",
                       ["charcoal", "conte", "炭筆", "木炭", "炭筆畫", "木炭画", "チャコール"]),
    "ink_wash": _style(ink_wash, "Sumi-e ink wash: graded ink tones with wet edges, dry-brush strokes, rice paper, red seal.",
                       ["ink wash", "sumi-e", "sumie", "sumi e", "chinese painting", "brush and ink", "shan shui",
                        "水墨", "水墨畫", "國畫", "墨畫", "山水畫", "水墨画", "墨絵", "墨彩"]),
    "pen_ink": _style(pen_ink, "Pen and ink: contour lines and cross-hatching in tonal tiers, black ink on white.",
                      ["pen and ink", "pen-and-ink", "ink drawing", "line drawing", "line art", "lineart", "cross-hatching",
                       "crosshatching", "hatching", "ink illustration", "鋼筆畫", "針筆", "線稿", "排線", "ペン画",
                       "線画", "ハッチング"]),
    "engraving": _style(engraving, "Engraving / etching: swelling parallel lines that bow over forms, banknote style.",
                        ["engraving", "etching", "intaglio", "banknote", "copperplate", "steel engraving", "gravure",
                         "woodblock engraving", "銅版畫", "蝕刻", "雕版", "鈔票", "銅版画", "エッチング", "版画風"]),
    "comic": _style(comic, "Comic / pop art: bold outlines, flat saturated fills, Ben-Day halftone dots.",
                    ["comic", "comic book", "pop art", "lichtenstein", "halftone", "ben-day", "benday", "cartoon",
                     "superhero", "美漫", "漫畫風", "普普", "波普", "網點", "アメコミ", "ポップアート", "コミック"]),
    "manga": _style(manga, "Manga: black-and-white ink lines with cut screentone dots and solid blacks.",
                    ["manga", "screentone", "black and white comic", "黑白漫畫", "日漫", "網點紙", "マンガ", "漫画",
                     "スクリーントーン"]),
    "cel": _style(anime, "Cel anime: flat colours, one crisp cool shadow tone, highlight band, thin colour outlines.",
                  ["anime", "cel shaded", "cel-shaded", "cel shading", "ghibli", "cartoon style", "toon", "動畫",
                   "動漫", "卡通", "賽璐璐", "二次元", "アニメ", "セル画", "セルシェーディング"]),
    "gouache": _style(gouache, "Gouache / poster paint: flat opaque matte shapes with paper slivers and a taped margin.",
                      ["gouache", "poster paint", "poster color", "tempera", "flat painting", "不透明水彩", "廣告顏料",
                       "水粉", "水粉畫", "ガッシュ", "ポスターカラー"]),
    "pastel": _style(pastel, "Soft pastel / chalk: powdery strokes on toned paper, tooth showing, smudged blends.",
                     ["pastel", "soft pastel", "chalk pastel", "chalk drawing", "粉彩", "色粉", "パステル"]),
    "storybook": _style(storybook, "Storybook illustration: soft painterly shading, gentle brown outlines, warm paper tone and tooth.",
                        ["storybook", "story book", "storybook illustration", "children's book", "childrens book",
                         "children's illustration", "picture book", "picturebook", "fairy tale illustration",
                         "fairytale illustration", "nursery", "bedtime story", "童書", "繪本", "故事書", "插畫繪本",
                         "絵本", "児童書", "童話"]),
    "crayon": _style(crayon, "Wax crayon: bright scribbled colour, white paper tooth everywhere, waxy outlines.",
                     ["crayon", "crayons", "wax crayon", "child's drawing", "kid's drawing", "children's drawing",
                      "蠟筆", "兒童畫", "クレヨン", "クレパス"]),
    "impressionist": _style(impressionist, "Impressionist: short thick dabs of broken colour following the forms.",
                            ["impressionist", "impressionism", "monet", "renoir", "plein air", "broken color",
                             "印象派", "印象主義", "莫內", "モネ", "印象派風"]),
    "pointillism": _style(pointillism, "Pointillism: dots of pure varied colour that mix in the eye (Seurat).",
                          ["pointillism", "pointillist", "seurat", "stippling", "stipple", "dot painting", "點描",
                           "點彩", "點描畫", "点描", "点描画"]),
    "ukiyoe": _style(ukiyoe, "Ukiyo-e woodblock: flat traditional pigments, bokashi gradients, key-lines, washi.",
                     ["ukiyo-e", "ukiyoe", "ukiyo e", "woodblock print", "japanese print", "hokusai", "hiroshige",
                      "浮世繪", "木版畫", "北齋", "浮世絵", "木版画", "北斎", "広重"]),
    "risograph": _style(risograph, "Risograph: 2-3 spot inks, grainy screening, misregistration, overprint.",
                        ["risograph", "riso", "riso print", "spot color print", "zine", "孔版", "リソグラフ", "リソ"]),
    "linocut": _style(linocut, "Linocut / woodcut: carved two-tone print with gouge lines and rough edges.",
                      ["linocut", "lino cut", "woodcut", "wood cut", "relief print", "block print", "版畫", "木刻",
                       "木刻版畫", "麻膠版", "リノカット", "木版", "版画"]),
    "stained_glass": _style(stained_glass, "Stained glass: glowing glass pieces between dark lead cames.",
                            ["stained glass", "church window", "leaded glass", "彩繪玻璃", "彩色玻璃", "花窗",
                             "ステンドグラス"]),
    "mosaic": _style(mosaic, "Mosaic: tesserae in rows that follow the outlines, bevelled tiles, light grout.",
                     ["mosaic", "tesserae", "tile art", "roman mosaic", "馬賽克", "鑲嵌", "モザイク"]),
    "low_poly": _style(low_poly, "Low-poly: flat-coloured triangles snapped to the picture's edges.",
                       ["low poly", "low-poly", "lowpoly", "polygon art", "triangulated", "faceted", "低多邊形",
                        "低面數", "ローポリ", "ポリゴン"]),
    "pixel_art": _style(pixel_art, "Pixel art: crisp coarse pixels, a small palette that keeps the highlights, ordered dither, sprite outlines; palette presets pico8 / gameboy / nes / c64.",
                        ["pixel art", "pixel-art", "pixelart", "8-bit", "8 bit", "16-bit", "16 bit", "sprite", "retro game",
                         "像素", "像素風", "像素畫", "ドット絵", "ピクセルアート"]),
    "paper_cut": _style(paper_cut, "Paper cut: every colour region a piece of card, stacked in depth with knife-cut edges and soft cast shadows (shadow box).",
                        ["paper cut", "papercut", "paper-cut", "cut paper", "paper craft", "papercraft", "shadow box",
                         "layered paper", "剪紙", "紙雕", "切り絵", "ペーパークラフト"]),
    "collage": _style(collage, "Collage: torn paper pieces with white fibrous edges and shadows on kraft card.",
                      ["collage", "torn paper", "photomontage", "scrapbook", "mixed media", "拼貼", "コラージュ"]),
    "cross_stitch": _style(cross_stitch, "Cross-stitch embroidery: thread X stitches on Aida cloth.",
                           ["cross-stitch", "cross stitch", "embroidery", "embroidered", "needlepoint", "stitch",
                            "十字繡", "刺繡", "刺繍", "クロスステッチ"]),
    "blueprint": _style(blueprint, "Blueprint: white technical lines on Prussian blue with grid, frame and title block.",
                        ["blueprint", "technical drawing", "schematic", "drafting", "藍圖", "工程圖", "設計圖", "青写真",
                         "設計図"]),
    "cyanotype": _style(cyanotype, "Cyanotype: Prussian-blue sun print with brushed emulsion edges.",
                        ["cyanotype", "sun print", "sunprint", "藍曬", "藍晒", "サイアノタイプ", "青焼き"]),
    "chalkboard": _style(chalkboard, "Chalkboard: chalk lines and shading on a slate board with a wooden frame.",
                         ["chalkboard", "blackboard", "chalk", "greenboard", "黑板", "粉筆", "黒板", "チョーク"]),
    "synthwave": _style(synthwave, "Neon / synthwave grade: indigo-magenta-orange map, glowing neon edges, bloom.",
                        ["synthwave", "neon", "retrowave", "outrun", "cyberpunk", "80s", "賽博龐克", "霓虹",
                         "サイバーパンク", "ネオン", "シンセウェイヴ"]),
    "vaporwave": _style(vaporwave, "Vaporwave: pastel pink-teal map, VHS chroma bleed, RGB split, scanlines.",
                        ["vaporwave", "vhs", "aesthetic", "y2k", "蒸汽波", "ヴェイパーウェイヴ"]),
    "duotone": _style(duotone, "Duotone / tritone: values mapped onto two or three ink colours.",
                      ["duotone", "tritone", "two-tone", "two tone", "雙色", "雙色調", "ダブルトーン", "デュオトーン"]),
    "vintage": _style(vintage, "Vintage film: faded blacks, warm cast, halation, light leak, grain, vignette.",
                      ["vintage", "retro photo", "film photo", "analog", "analogue", "35mm", "polaroid", "faded",
                       "kodak", "old film", "復古", "底片", "膠片", "懷舊", "フィルム写真", "レトロ", "ヴィンテージ"]),
    "sepia": _style(sepia, "Sepia / daguerreotype: antique toned plate, soft focus, tarnished rim, scratches.",
                    ["sepia", "daguerreotype", "tintype", "antique photo", "old photograph", "19th century photo",
                     "victorian photo", "老照片", "褐色", "古董照片", "銀版", "セピア", "古写真", "ダゲレオタイプ"]),
    "noir": _style(noir, "Film noir: hard black and white, crushed blacks, venetian-blind light, grain.",
                   ["film noir", "noir", "black and white", "black-and-white", "b&w", "monochrome photo", "黑色電影",
                    "黑白", "フィルムノワール", "白黒", "モノクロ"]),
    "tilt_shift": _style(tilt_shift, "Tilt-shift miniature: sharp band, graded blur, toy-like colour.",
                         ["tilt-shift", "tilt shift", "miniature", "diorama", "toy town", "移軸", "微縮", "ミニチュア",
                          "チルトシフト"]),
    "photo_finish": _style(photo_finish, "Photo finish: lens blur by depth, chromatic aberration, bloom, grain, vignette.",
                           ["photorealistic", "photo-realistic", "realistic photo", "photograph", "photo", "dslr",
                            "cinematic", "realistic", "寫實", "照片", "攝影", "写実", "写真", "フォトリアル"]),
    "glitch": _style(glitch, "Glitch: displaced slices, RGB split, corrupted blocks, pixel sorting.",
                     ["glitch", "datamosh", "pixel sort", "corrupted", "故障", "故障藝術", "グリッチ"]),
    "thermal": _style(thermal, "Thermal camera: ironbow heat palette, sensor softness, crosshair and scale bar.",
                      ["thermal", "heat map", "heatmap", "thermal camera", "flir", "熱成像", "熱像", "サーモグラフィ"]),
    "infrared": _style(infrared, "Infrared false colour: pink-red foliage, deep blue skies, soft glow.",
                       ["infrared", "aerochrome", "false color", "ir photo", "紅外線", "紅外", "赤外線"]),
    "sticker": _style(sticker, "Die-cut sticker: white border and drop shadow around the subject (pass mask=, one mask or a list); the rest of the picture stays.",
                      ["sticker", "die-cut", "die cut", "decal", "貼紙", "ステッカー", "シール"]),
    "clay": _style(clay, "Claymation: lumps of plasticine cut by colour, rounded beads at the seams, the form's soft light kept, waxy sheen, fingerprints.",
                   ["claymation", "clay", "plasticine", "play-doh", "stop motion", "stop-motion", "黏土", "黏土動畫",
                    "定格動畫", "粘土", "クレイ", "クレイアニメ"]),
}


def names():
    """All style names, in registry order."""
    return list(STYLES)


def describe(name):
    """One line: what the style looks like."""
    return STYLES[name]["desc"]


_GENERIC = {"painting", "painted", "drawing", "sketch", "photo", "photograph", "realistic", "cartoon", "stitch",
            "chalk", "clay", "aesthetic", "black and white", "black-and-white", "b&w", "黑白", "白黒", "モノクロ",
            "絵画", "照片", "写真", "攝影"}


def _kw_hit(text, kw):
    if all(ord(ch) < 128 for ch in kw):
        import re
        return re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", text) is not None
    return kw in text


def pick(prompt_text):
    """Best style name for a request (English, Traditional Chinese or Japanese keywords), or None.
    Longer (more specific) keyword matches win: "ink wash" → ink_wash, "pen and ink" → pen_ink."""
    t = str(prompt_text).lower()
    best, score = None, 0
    for name, st in STYLES.items():
        s = max(((1 if k in _GENERIC else len(k)) for k in st["keywords"] if _kw_hit(t, k.lower())), default=0)
        if s > score:
            best, score = name, s
    if best is None:                                    # "pixel-art", "water colour", "Ukiyo E": the style name itself
        for name in STYLES:
            parts = [re.escape(c) for c in name.replace("_", "")]
            if re.search(r"(?<![a-z0-9])" + r"[-_ ]?".join(parts) + r"(?![a-z0-9])", t.replace("colour", "color")):
                return name
    return best


def _style_key(s):
    return re.sub(r"[-_\s]+", "", str(s).lower().replace("colour", "color"))


def resolve(name):
    """The canonical style name for ``name``, matched ignoring case, hyphens, spaces and underscores
    ("ukiyo-e", "Ukiyo E", "pixel-art", "water colour" -> ukiyoe, pixel_art, watercolor). A real miss raises a
    ValueError that names the closest styles."""
    if isinstance(name, str):
        if name in STYLES:
            return name
        key = _style_key(name)
        for n in STYLES:
            if _style_key(n) == key:
                return n
    raise _UnknownName(*_unknown_name("style", name, list(STYLES), "Try stylize.pick('...') or stylize.names()").args)


class _UnknownName(ValueError, KeyError):
    """Unknown registry name (a ValueError that is also a KeyError)."""
    __str__ = ValueError.__str__


# ---- parameter guide: useful range and a short meaning per option (sizes in 1080p pixels)
# name: (low, high, meaning); low / high None = not a number (colour, preset, switch, array)
_PARAM = {
    "brush": (2, 60, "brush stroke size in pixels"), "impasto": (0, 1.5, "height of the paint relief"),
    "canvas": (0, 0.2, "strength of the canvas weave"), "wash": (1, 24, "size of the flat washes"),
    "transparency": (0, 0.8, "how much paper glows through"), "edge": (0, 2, "darkness of pooled wash edges"),
    "granulation": (0, 2.5, "pigment grain in the washes"), "bleed": (0, 24, "how far washes wander, pixels"),
    "paper": (None, None, "colour of the paper"), "deckle": (0, 0.12, "width of the unpainted paper border"),
    "line": (0.2, 3, "weight of the contour lines"), "hatching": (0, 2, "amount of hatching in shadows"),
    "colored": (0, 1, "how much colour is kept"), "stroke": (0.3, 3, "size of the strokes"),
    "pressure": (0.2, 2, "how hard the strokes press"), "outline": (0, 3, "weight of the outlines"),
    "contrast": (0.5, 2.5, "contrast of the values"), "smudge": (0, 2.5, "amount of smudged blending"),
    "tones": (2, 8, "number of ink tone levels"), "dry": (0, 2, "amount of dry-brush strokes"),
    "seal": (None, None, "stamp the red seal (bool)"), "spacing": (3, 30, "distance between lines, pixels"),
    "angle": (-90, 90, "direction of the lines, degrees"), "relief": (0, 2, "how much lines bow over forms"),
    "ink": (None, None, "colour of the ink"), "colors": (2, 64, "number of flat colours"),
    "dots": (3, 32, "halftone dot spacing in pixels"), "saturation": (0.5, 2, "colour saturation multiplier"),
    "whites": (None, None, "keep the white highlights (bool)"), "tone": (2, 24, "screentone dot spacing, pixels"),
    "shadow": (0, 2, "strength of the cast shadows"), "gap": (0, 2.5, "paper slivers between shapes"),
    "border": (0, 0.08, "width of the taped margin"), "coverage": (0.3, 1.5, "how fully pastel covers paper"),
    "jitter": (0, 0.25, "random colour change per dab"), "ground": (None, None, "colour of the ground showing"),
    "dot": (3, 32, "dot size in pixels"), "keyline": (0, 2.5, "weight of the key lines"),
    "bokashi": (0, 2, "strength of the colour fades"), "cartouche": (None, None, "add the title cartouche (bool)"),
    "inks": (None, None, "the spot ink colours"), "grain": (0, 2.5, "amount of grain"),
    "misregister": (0, 3, "offset between the ink layers"), "threshold": (0.2, 0.8, "value split ink / paper"),
    "gouge": (0, 2, "amount of carved gouge marks"), "color": (None, None, "ink colour, None for black"),
    "cell": (12, 160, "size of one piece, pixels"), "lead": (0, 2.5, "width of the lead lines"),
    "glow": (0, 2.5, "strength of the glow"), "tile": (6, 64, "tile size in pixels"),
    "grout": (None, None, "colour of the grout"), "facet": (0, 0.2, "shading across each triangle"),
    "pixel": (2, 64, "size of one art pixel"), "dither": (0, 1, "amount of ordered dithering"),
    "palette": (None, None, "palette preset or colour list"), "layers": (2, 16, "number of paper layers"),
    "pieces": (2, 30, "number of torn pieces"), "offset": (0, 3, "how far the pieces shift"),
    "stitch": (6, 60, "size of one stitch, pixels"), "fabric": (None, None, "colour of the cloth"),
    "grid_size": (8, 120, "grid cell size in pixels"), "board": (None, None, "colour of the board"),
    "frame": (None, None, "draw the wooden frame (bool)"), "dark": (None, None, "ink for the darks"),
    "light": (None, None, "ink for the lights"), "mid": (None, None, "optional ink for mid tones"),
    "fade": (0, 2, "how faded the blacks are"), "leak": (0, 2, "strength of the light leak"),
    "tarnish": (0, 2, "amount of tarnish at the rim"), "blinds": (0, 1, "strength of the blind shadows"),
    "focus": (0, 1, "where the sharp band sits"), "band": (0.02, 0.4, "height of the sharp band"),
    "blur": (0, 40, "maximum blur in pixels"), "depth": (None, None, "depth map, None for vertical"),
    "aperture": (0, 3, "strength of the lens blur"), "aberration": (0, 3, "colour fringe at the edges"),
    "bloom": (0, 3, "glow around bright areas"), "vignette": (0, 2, "darkening of the corners"),
    "amount": (0, 3, "how much glitch damage"), "hud": (None, None, "draw crosshair and scale (bool)"),
    "mask": (None, None, "subject mask or list of masks; the rest of the picture stays"),
    "background": (None, None, "colour that replaces everything outside the sticker (None keeps the picture)"),
    "warmth": (0, 2, "strength of the warm paper tint"),
    "softness": (0, 2.5, "roundness of the clay seams"),
}
_PARAM_STYLE = {                                       # style-specific meanings / ranges
    "watercolor.paper": (0, 2, "strength of the paper texture"), "watercolor.bleed": (0, 24, "how far washes wander, pixels"),
    "chalkboard.color": (0, 1, "how much colour the chalk keeps"), "sticker.border": (0, 120, "white border width, pixels"),
    "sticker.shadow": (0, 2, "strength of the drop shadow"), "storybook.softness": (0, 2.5, "softness of the shading"), "cel.shadow": (None, None, "colour of the shadow tone"),
    "sepia.tone": (None, None, "tone preset name"), "low_poly.cell": (16, 240, "triangle size in pixels"),
    "photo_finish.focus": (0, 1, "depth that stays sharp"), "cyanotype.contrast": (0.5, 2.5, "contrast of the print"),
    "ink_wash.bleed": (0, 2.5, "how far wet ink spreads"), "impressionist.brush": (6, 64, "dab size in pixels"),
}
_MILD = 0.5                       # overshoot up to half the useful span is clamped with a note; beyond it is an error


def _param_info(name, p):
    return _PARAM_STYLE.get(f"{name}.{p}") or _PARAM.get(p) or (None, None, "style option")


def options(name):
    """Options the style ``name`` accepts besides ``strength`` and ``seed`` (``apply(cv, name, **options)``):
    ``{option: {"default": value, "range": (low, high) or None, "means": "short meaning"}}``. Iterating the
    result gives the names."""
    name = resolve(name)
    import inspect
    fn = STYLES[name]["fn"]
    fn = {_wrap_watercolor: watercolor, _wrap_pencil: pencil_sketch}.get(fn, fn)
    try:
        params = inspect.signature(inspect.unwrap(fn)).parameters
    except (TypeError, ValueError):
        params = {}
    out = {}
    for n in _sigopts(fn)[1:]:
        if n == "seed":
            continue
        lo, hi, means = _param_info(name, n)
        d = STYLES[name]["defaults"].get(n, params[n].default if n in params else None)
        out[n] = {"default": d, "range": None if lo is None else (lo, hi), "means": means}
    return out


def option_lines(name):
    """One line per option of style ``name``: ``deckle=0.05  (0-0.12)  width of the unpainted paper border``."""
    lines = []
    for n, o in options(name).items():
        rng = f"  ({o['range'][0]:g}-{o['range'][1]:g})" if o["range"] else ""
        lines.append(f"{n}={o['default']!r}{rng}  {o['means']}")
    return lines


def _guard(name, given):
    """Check numeric options against their useful range: mild overshoot is clamped (returns the notes),
    nonsense raises a ValueError naming the range."""
    opts = options(name)
    out, notes = dict(given), []
    for k, v in given.items():
        o = opts.get(k)
        if o is None or o["range"] is None or isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
            continue
        lo, hi = o["range"]
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"apply({name!r}): {k}={v} is not a number; useful range {lo:g}-{hi:g} ({o['means']})")
        if lo <= v <= hi:
            continue
        span = hi - lo
        neg = v < 0 <= lo                                # a negative size / amount is never meant
        if neg or v > hi + _MILD * span or v < lo - _MILD * span:
            raise ValueError(f"apply({name!r}): {k}={given[k]!r} is outside the useful range {lo:g}-{hi:g} "
                             f"({o['means']}; default {o['default']!r}). Use a value in {lo:g}-{hi:g}.")
        c = min(max(v, lo), hi)
        if isinstance(given[k], (int, np.integer)) and float(c).is_integer():
            c = int(c)
        out[k] = c
        notes.append(f"stylize {name}: {k}={given[k]!r} is outside {lo:g}-{hi:g} ({o['means']}); using {c!r}")
    return out, notes


def apply(cv_or_array, name=None, strength=1.0, seed=0, prompt=None, cv=None, **params):
    """Apply style ``name`` with its defaults (overridden by ``params``). A Canvas is changed in place and
    returned; an array gives a new (H, W, 3) array (pass ``cv=main_canvas`` when that array is a sticker or
    layer that gets pasted into ``main_canvas``, so the report counts the style as applied). ``strength`` < 1 mixes the original back in.

    ``prompt="a watercolor of a fox"`` picks the style from a request sentence (:func:`pick`) and applies it; it
    returns None and leaves the picture alone when the sentence names no medium. ``name`` wins when both are given.
    ``params`` are the style's own options (see :func:`options`); unknown names raise a TypeError naming the
    closest valid one, and an unknown style raises a ValueError with the closest style names."""
    if name is None:
        name = pick(prompt) if prompt is not None else None
        if name is None:
            if prompt is None:
                raise TypeError("apply: give a style name or prompt='...'. Styles: " + ", ".join(list(STYLES)[:15]) + ", …")
            return None
    name = resolve(name)                                    # watercolour, ink-wash, Ink Wash, ukiyo-e ...
    st = STYLES[name]
    kw = dict(st["defaults"])
    notes = []
    if params:
        try:
            given = _reject("apply(%r)" % name, params, list(options(name)))
        except TypeError as e:
            raise TypeError(str(e).split(" Options:")[0] + "\noptions of %r (default, useful range, meaning):\n  "
                            % name + "\n  ".join(option_lines(name))) from None
        given, notes = _guard(name, given)
        kw.update(given)
    s = float(strength)
    if not 0.0 <= s <= 1.0:
        if s < 0 or s > 1.5:
            raise ValueError(f"apply({name!r}): strength={strength!r} is outside 0-1 (0 = original, 1 = full style)")
        notes.append(f"stylize {name}: strength={strength!r} is outside 0-1; using 1.0")
        s = 1.0
    import sys as _sys
    for n in notes:
        print(n, file=_sys.stderr)
    is_canvas = hasattr(cv_or_array, "buf")
    src = np.asarray(cv_or_array.buf if is_canvas else cv_or_array, F32)
    out = np.asarray(st["fn"](src, seed=seed, **kw), F32)
    if s < 1.0:
        out = _img(src) * (1 - s) + out * s
    if is_canvas:
        cv_or_array.buf = out.astype(F32)
        # recorded in cv.styles_applied; the picture before the style feeds the mostly_blank check
        rec = dict(params or {})
        if rec.get("mask") is not None:
            rec["mask"] = "masked region"                    # a style applied to part of the picture is recorded as such
        _manifest.style_applied(cv_or_array, name, before=src, params=rec, notes=notes)
        return cv_or_array
    if cv is not None and hasattr(cv, "buf"):
        rec = dict(params or {})
        if rec.get("mask") is not None:
            rec["mask"] = "masked region"
        _manifest.style_applied(cv, name, before=None, params=rec, notes=notes)
    return out


# prompt words → which medium to apply (used by the skill's prompt checklist); keys are function names
MEDIA = {
    "oil_paint": ("painting", "oil", "acrylic", "impasto", "brushstroke", "brush stroke", "canvas", "van gogh"),
    "watercolor": ("watercolor", "watercolour", "aquarelle"),
    "pencil_sketch": ("pencil", "sketch", "graphite", "drawing"),
    "cel_shade": ("anime", "cel-shaded", "cel shaded", "cartoon", "ghibli", "pixar"),
}
for _name, _st in STYLES.items():
    _fn = _st["fn"]
    if _fn in (oil, _wrap_watercolor, _wrap_pencil, anime):
        continue
    MEDIA[_fn.__name__] = tuple(k for k in _st["keywords"] if all(ord(c) < 128 for c in k))
MEDIA["anime"] = STYLES["cel"]["keywords"][:8]
del _name, _st, _fn


# ---- strict options: unknown keyword names raise TypeError naming the closest valid option; colour / color,
# ---- w / width ... are accepted (pixelcanvas._args). ``apply(**params)`` checks against the chosen style.
for _n, _f in list(globals().items()):
    if (not _n.startswith("_") and callable(_f) and getattr(_f, "__module__", None) == __name__
            and not isinstance(_f, type)):
        globals()[_n] = _strict(_f, _n)
