# -*- coding: utf-8 -*-
"""
stylize — make a finished render read as a *medium*: oil painting, watercolour, pencil sketch, cel-shaded anime,
canvas texture. Array in → array out ((H, W, 3) float 0~1), like ``pixelcanvas.filters``; use with ``cv.apply``:

>>> from pixelcanvas import stylize
>>> cv.apply(stylize.oil_paint, brush=6 * S, seed=seed)      # "a painting of …", "impressionist", "brushstrokes"
>>> cv.apply(stylize.watercolor, seed=seed)                  # "watercolour", "ink wash"
>>> cv.apply(stylize.pencil_sketch, seed=seed)               # "pencil drawing", "sketch", "charcoal"
>>> cv.apply(stylize.cel_shade, levels=3, outline=2 * S)     # "anime", "cartoon", "comic", "cel-shaded"
>>> cv.apply(stylize.canvas_texture, strength=0.06)          # paint on canvas / paper grain

Why: a prompt that names a medium is a requirement like any object. Judges (and people) answer "is this a painting?"
from brush marks, paper and edges — a clean vector render says "digital illustration". Apply the medium as the last
step before / instead of photographic post (bloom, grain), in its own ``log("medium")`` step.

Pixel sizes are absolute — multiply by ``S = H / 1080`` like everything else.
"""
from __future__ import annotations

import numpy as np

from .color import F32, clamp01, luma, smoothstep
from .filters import gaussian_blur, sample
from .noise import fbm2d

__all__ = ["kuwahara", "flow_field", "lic", "oil_paint", "watercolor", "pencil_sketch", "cel_shade", "canvas_texture",
           "MEDIA"]


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


# prompt words → which medium to apply (used by the skill's prompt checklist)
MEDIA = {
    "oil_paint": ("painting", "oil", "acrylic", "impasto", "impressionist", "brushstroke", "brush stroke", "canvas",
                  "van gogh", "monet", "gouache"),
    "watercolor": ("watercolor", "watercolour", "ink wash", "aquarelle"),
    "pencil_sketch": ("pencil", "sketch", "graphite", "charcoal", "drawing", "line art", "hatching"),
    "cel_shade": ("anime", "manga", "cartoon", "comic", "cel-shaded", "cel shaded", "ghibli", "pixar"),
}
