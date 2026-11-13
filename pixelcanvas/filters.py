# -*- coding: utf-8 -*-
"""Blend modes and filters / post-processing (pure array-in, array-out functions; Canvas has wrappers of the same names)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .color import F32, as_color, rgb, luma, clamp01, smoothstep, grid

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

_RES = getattr(Image, "Resampling", Image)

__all__ = [
    "blend", "BLEND_MODES", "check_mode", "gaussian_blur", "sample", "resize", "radial_blur", "light_shafts", "bloom", "vignette",
    "grain", "levels", "contrast", "saturation", "soft_clip", "filmic", "split_tone",
]


BLEND_MODES = ("normal", "add", "multiply", "screen", "overlay", "softlight", "lighten", "darken")


def check_mode(mode, owner="blend"):
    """Return ``mode`` if it is a blend mode; else raise ValueError with the closest mode names
    (a number in the mode slot is the opacity: ``cv.paint(color, mask, 0.9)``)."""
    if mode in BLEND_MODES:
        return mode
    from ._args import check_choice
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        raise ValueError("%s: blend mode %r is a number; opacity is the next argument (cv.paint(color, mask, %r)) "
                         "or opacity=%r. Choices: %s" % (owner, mode, mode, mode, ", ".join(BLEND_MODES)))
    return check_choice(owner, "blend mode", mode, BLEND_MODES)


def blend(base, top, alpha=1.0, mode="normal"):
    """Blend top onto base with alpha (scalar or (H, W) mask) in the given mode; returns a new array (not clipped to 0~1).

    mode: normal / add / multiply / screen / overlay / softlight / lighten / darken
    """
    base = np.asarray(base, F32)
    top = as_color(top)
    if top.ndim == 1:
        top = np.broadcast_to(top, base.shape)
    a = np.asarray(alpha, F32)
    if a.ndim == 2:
        a = a[..., None]
    if mode == "normal":
        res = top
    elif mode == "add":
        res = base + top
    elif mode == "multiply":
        res = base * top
    elif mode == "screen":
        res = 1.0 - (1.0 - base) * (1.0 - top)
    elif mode == "overlay":
        res = np.where(base < 0.5, 2.0 * base * top, 1.0 - 2.0 * (1.0 - base) * (1.0 - top))
    elif mode == "softlight":
        res = (1.0 - 2.0 * top) * base * base + 2.0 * top * base
    elif mode == "lighten":
        res = np.maximum(base, top)
    elif mode == "darken":
        res = np.minimum(base, top)
    else:
        check_mode(mode, "blend")
    return (base + (res - base) * a).astype(F32)


def _box1d(a, r, axis):
    if r <= 0:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    zero = np.zeros_like(np.take(c, [0], axis=axis))
    c = np.concatenate([zero, c], axis=axis)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(F32)


def _conv1d(a, k, axis):
    """Correlate ``a`` with the odd-length kernel ``k`` along ``axis`` (edges replicated)."""
    r = len(k) // 2
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    out = None
    for i, w in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + n)
        t = p[tuple(sl)] * F32(w)
        out = t if out is None else out + t
    return out


def _box_radii(sigma, n=3):
    """Radii of ``n`` successive box blurs whose combined variance matches a Gaussian of ``sigma``."""
    wl = int(math.floor(math.sqrt(12.0 * sigma * sigma / n + 1.0)))
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 1)
    wu = wl + 2
    m = int(round((12.0 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4.0 * wl - 4.0)))
    return [((wl if i < m else wu) - 1) // 2 for i in range(n)]


def gaussian_blur(a, sigma):
    """Gaussian blur for (H, W) or (H, W, C). Without scipy: the same truncated kernel for sigma up to 6, and
    three variance-matched box blurs for wider blurs."""
    a = np.asarray(a, F32)
    if sigma <= 0:
        return a.copy()
    if _ndi is not None:
        if a.ndim == 3:
            return np.stack([_ndi.gaussian_filter(a[..., c], sigma, mode="nearest") for c in range(a.shape[2])], -1).astype(F32)
        return _ndi.gaussian_filter(a, sigma, mode="nearest").astype(F32)
    if sigma <= 6.0:
        r = int(4.0 * sigma + 0.5)
        if r < 1:
            return a.copy()
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / float(sigma)) ** 2)
        k /= k.sum()
        return _conv1d(_conv1d(a, k, 0), k, 1).astype(F32)
    out = a
    for r in _box_radii(sigma):
        out = _box1d(_box1d(out, r, 0), r, 1)
    return out


def sample(img, X, Y, order=1):
    """Bilinear sampling: read img at float pixel coordinates (X, Y) (edges clamp to nearest). Used for reflections, warps, displacement, refraction."""
    img = np.asarray(img, F32)
    X = np.asarray(X, F32)
    Y = np.asarray(Y, F32)
    if _ndi is not None:
        coords = [Y, X]
        if img.ndim == 2:
            return _ndi.map_coordinates(img, coords, order=order, mode="nearest").astype(F32)
        return np.stack([_ndi.map_coordinates(img[..., c], coords, order=order, mode="nearest") for c in range(img.shape[2])], -1).astype(F32)
    h, w = img.shape[:2]
    X = np.clip(X, 0, w - 1)
    Y = np.clip(Y, 0, h - 1)
    if order == 0:
        return img[np.rint(Y).astype(int), np.rint(X).astype(int)]
    x0 = np.minimum(np.floor(X).astype(int), max(w - 2, 0))
    y0 = np.minimum(np.floor(Y).astype(int), max(h - 2, 0))
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (X - x0).astype(F32)
    fy = (Y - y0).astype(F32)
    if img.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(F32)


def resize(arr, w, h, method="bilinear"):
    """Resize a float array of (H, W) or (H, W, C). method: bilinear / bicubic / lanczos / box / nearest"""
    arr = np.asarray(arr, F32)
    table = {"bilinear": _RES.BILINEAR, "bicubic": _RES.BICUBIC, "lanczos": _RES.LANCZOS, "box": _RES.BOX, "nearest": _RES.NEAREST}
    if method not in table:
        from ._args import check_choice
        check_choice("resize", "method", method, tuple(table))
    m = table[method]
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(np.ascontiguousarray(arr)).resize((w, h), m), F32)
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(arr[..., c])).resize((w, h), m), F32) for c in range(arr.shape[2])], -1)


def radial_blur(img, center, strength=0.3, samples=24, decay=1.0, downscale=1):
    """Radial (zoom) blur: scale the image toward center several times and average with weights. ``strength`` = maximum zoom amount (0~1).

    With ``downscale`` > 1 the work is done at reduced size and scaled back up (much faster, good for soft glow).
    """
    img = np.asarray(img, F32)
    h, w = img.shape[:2]
    cx, cy = center
    if downscale > 1 and w // downscale >= 4 and h // downscale >= 4:
        small = resize(img, w // downscale, h // downscale, "box")
        out = radial_blur(small, (cx / downscale, cy / downscale), strength, samples, decay, 1)
        return resize(out, w, h, "bilinear")
    X, Y = grid(h, w)
    acc = np.zeros_like(img)
    wsum, wgt = 0.0, 1.0
    for i in range(samples):
        s = 1.0 - strength * i / max(samples - 1, 1)
        acc += wgt * sample(img, cx + (X - cx) * s, cy + (Y - cy) * s)
        wsum += wgt
        wgt *= decay
    return (acc / wsum).astype(F32)


def light_shafts(source_mask, center, samples=32, decay=0.95, length=0.9, downscale=2):
    """Volumetric light / god rays: radially blur a light-source mask (e.g. the sky around the sun that is not blocked by mountains) toward the light.

    Returns (H, W) ray intensity, which can then be added with ``cv.paint(color, rays * strength, 'add')``.
    """
    return radial_blur(np.asarray(source_mask, F32), center, strength=length, samples=samples, decay=decay, downscale=downscale)


def bloom(img, threshold=0.75, sigma=30, strength=0.4):
    """Bloom: blur the parts brighter than threshold and add them back."""
    img = np.asarray(img, F32)
    l = luma(img)
    k = clamp01((l - threshold) / max(1.0 - threshold, 1e-6))[..., None]
    return (img + gaussian_blur(img * k, sigma) * strength).astype(F32)


def vignette(img, strength=0.35, radius=0.55, softness=0.6):
    """Vignette: ``radius`` is where darkening starts (0 centre ~ 1 corner), ``softness`` is the transition width."""
    img = np.asarray(img, F32)
    h, w = img.shape[:2]
    X, Y = grid(h, w)
    d = np.sqrt(((X / w - 0.5) * 2) ** 2 + ((Y / h - 0.5) * 2) ** 2) / 1.4142
    v = 1.0 - strength * smoothstep(radius - softness * 0.5, radius + softness * 0.5, d)
    return (img * v[..., None]).astype(F32)


def grain(img, amount=0.02, seed=0, mono=True):
    """Film grain."""
    img = np.asarray(img, F32)
    rng = np.random.default_rng(seed)
    if mono:
        n = rng.normal(0, amount, img.shape[:2]).astype(F32)[..., None]
    else:
        n = rng.normal(0, amount, img.shape).astype(F32)
    return (img + n).astype(F32)


def levels(img, black=0.0, white=1.0, gamma=1.0):
    """Levels: black->0, white->1, then apply gamma."""
    x = clamp01((np.asarray(img, F32) - black) / max(white - black, 1e-6))
    return (x ** (1.0 / gamma)).astype(F32)


def contrast(img, amount=0.2):
    """S-curve contrast (a negative amount lowers contrast)."""
    x = clamp01(img)
    return (x + amount * (x * x * (3.0 - 2.0 * x) - x)).astype(F32)


def saturation(img, s=1.0):
    """Saturation multiplier."""
    img = np.asarray(img, F32)
    l = luma(img)[..., None]
    return (l + (img - l) * s).astype(F32)


def soft_clip(img, knee=0.8):
    """Soft clip: unchanged below knee, smoothly compressed toward 1 above it (handles HDR highlights > 1)."""
    x = np.asarray(img, F32)
    y = x.copy()
    over = x > knee
    y[over] = knee + (1.0 - knee) * (1.0 - np.exp(-(x[over] - knee) / (1.0 - knee)))
    return np.clip(y, 0.0, 1.0).astype(F32)


def filmic(img):
    """ACES-style filmic tone mapping (slightly lifts midtones and compresses highlights)."""
    x = np.maximum(np.asarray(img, F32), 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return clamp01((x * (a * x + b)) / (x * (c * x + d) + e))


def split_tone(img, shadows="#405070", highlights="#f0c080", amount=0.1):
    """Split toning: shadows shift toward ``shadows``, highlights toward ``highlights`` (0.5 grey is neutral)."""
    img = np.asarray(img, F32)
    l = luma(clamp01(img))[..., None]
    off = (rgb(shadows) - 0.5) * (1 - l) + (rgb(highlights) - 0.5) * l
    return (img + off * amount * 2).astype(F32)


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "blur": "use filters.gaussian_blur(a, sigma) (sigma in pixels)",
    "boxblur": "use filters.gaussian_blur(a, sigma)",
    "glow": "use filters.bloom(img, threshold=0.75, sigma=30, strength=0.4)",
    "brightness": "use filters.levels(img, black, white, gamma) or filters.contrast(img, amount)",
    "gamma": "use filters.levels(img, black=0, white=1, gamma=1.0)",
    "saturate": "use filters.saturation(img, s)",
    "desaturate": "use filters.saturation(img, 0.0)",
    "grayscale": "use filters.saturation(img, 0.0), or pixelcanvas.color.luma(img) for an (H, W) array",
    "noise": "use filters.grain(img, amount=0.02, seed=0)",
    "film": "use filters.filmic(img) (tone curve) and filters.grain(img)",
    "tonemap": "use filters.filmic(img) or filters.soft_clip(img, knee)",
    "sharpen": "use img + (img - filters.gaussian_blur(img, 2)) * amount",
    "zoom_blur": "use filters.radial_blur(img, center, strength)",
    "motion_blur": "use filters.radial_blur(img, center, strength) for zoom blur, or filters.sample for custom warps",
    "shafts": "use filters.light_shafts(source_mask, center)",
    "godrays": "use filters.light_shafts(source_mask, center)",
    "scale": "use filters.resize(arr, w, h, method='bilinear')",
    "interpolate": "use filters.sample(img, X, Y) for bilinear lookups at float coordinates",
    "warp": "use filters.sample(img, X + dx, Y + dy)",
    "composite": "use filters.blend(base, top, alpha, mode) (modes: normal, add, multiply, screen, overlay, softlight, lighten, darken)",
    "multiply": "use filters.blend(base, top, alpha, mode='multiply')",
    "screen": "use filters.blend(base, top, alpha, mode='screen')",
    "overlay": "use filters.blend(base, top, alpha, mode='overlay')",
    "tint": "use filters.split_tone(img, shadows, highlights, amount)",
    "colorgrade": "use filters.split_tone(img, shadows, highlights, amount), filters.levels, filters.saturation",
}
__getattr__ = _module_getattr(__name__, _HINTS)
