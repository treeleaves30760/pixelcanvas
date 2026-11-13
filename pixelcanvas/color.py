# -*- coding: utf-8 -*-
"""Colour and basic math helpers (shared by every module).

Convention: a colour is ``'#rrggbb'`` / ``(r, g, b)`` (floats 0~1 with HDR above 1 allowed, or integers 0~255) / a full ``(H, W, 3)`` colour map;
values may exceed 1 (HDR) and are brought back to 0~1 by ``soft_clip`` / ``filmic``.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32

__all__ = ["F32", "rgb", "as_color", "lerp", "mix", "hsv_to_rgb", "luma", "clamp01", "smoothstep", "normalize", "grid"]


_NAMES = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
    "yellow": "#ffff00", "orange": "#ffa500", "purple": "#800080", "pink": "#ffc0cb", "brown": "#a52a2a",
    "gray": "#808080", "grey": "#808080", "cyan": "#00ffff", "magenta": "#ff00ff", "navy": "#000080",
    "teal": "#008080", "gold": "#ffd700", "silver": "#c0c0c0", "beige": "#f5f5dc", "maroon": "#800000",
    "olive": "#808000", "lime": "#00ff00", "indigo": "#4b0082", "violet": "#ee82ee", "turquoise": "#40e0d0",
    "coral": "#ff7f50", "salmon": "#fa8072", "khaki": "#f0e68c", "crimson": "#dc143c", "lavender": "#e6e6fa",
    "tan": "#d2b48c", "ivory": "#fffff0", "mint": "#98ff98",
}


def _parse_str(c):
    s = c.strip()
    key = s.lower()
    if key in _NAMES:
        s = _NAMES[key]
    h = s[1:] if s.startswith("#") else s
    if s.startswith("#") and len(h) in (3, 4, 6, 8) and all(ch in "0123456789abcdefABCDEF" for ch in h):
        if len(h) in (3, 4):
            h = "".join(ch * 2 for ch in h)
        return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], F32)
    raise ValueError(
        f"invalid colour {c!r}: use '#rgb', '#rgba', '#rrggbb', '#rrggbbaa', a CSS colour name "
        f"(e.g. 'navy', 'gold', 'coral'), or an (r, g, b) tuple")


def rgb(c):
    """Convert a colour to a float32 ndarray(3).

    Accepts ``'#rgb'`` / ``'#rgba'`` / ``'#rrggbb'`` / ``'#rrggbbaa'`` (alpha is ignored), a CSS colour name
    (``'navy'``, ``'gold'``, ``'coral'`` ...) or ``(r, g, b)``.
    The tuple rule: **floats are 0~1** (HDR values above 1 are allowed, e.g. ``(1.6, 0.62, 0.16)`` is a bright orange
    for ``mode="add"``), **integers are 0~255** (``(255, 128, 0)``); integers that are all 0 or 1 read as 0~1 so
    ``(0, 0, 0)`` and ``(1, 1, 1)`` are black and white. One float anywhere makes the whole tuple 0~1.
    An ndarray input is returned as is (so a full colour map can be passed through).
    Malformed strings, wrong lengths and integers outside 0~255 raise ValueError naming the bad value.
    """
    if isinstance(c, np.ndarray):
        return c.astype(F32, copy=False)
    if isinstance(c, str):
        return _parse_str(c)
    if c is None or np.isscalar(c):                      # kept as before: a placeholder, not a colour
        return np.asarray(np.nan if c is None else c, F32)
    try:
        items = list(c)
        a = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"invalid colour {c!r}: use '#rrggbb', a CSS colour name or an (r, g, b) tuple "
                         f"(floats 0-1, integers 0-255)") from None
    if a.ndim != 1 or len(a) not in (3, 4):
        raise ValueError(f"invalid colour {c!r}: an (r, g, b) tuple needs 3 numbers "
                         f"(floats 0-1, HDR above 1 allowed; integers 0-255)")
    a = a[:3]
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in items):
        if a.max() > 255 or a.min() < 0:
            raise ValueError(f"invalid colour {c!r}: integer components are 0-255 (use floats for 0-1 / HDR colours)")
        if a.max() > 1.0:
            a = a / 255.0
    return a.astype(F32)


def as_color(c):
    """Accept a single colour / a full (H, W, 3) colour map / a Canvas and return an ndarray."""
    if hasattr(c, "buf"):
        return np.asarray(c.buf, F32)
    if isinstance(c, np.ndarray):
        return c.astype(F32, copy=False)
    return rgb(c)


def _operand(x):
    if isinstance(x, (str, tuple, list)) or hasattr(x, "buf"):
        return as_color(x)
    return np.asarray(x, F32)


def lerp(a, b, t):
    """Linear interpolation a + (b - a) * t. a, b may be a colour, a colour map or any array; t a scalar or an (H, W) mask."""
    a = _operand(a)
    b = _operand(b)
    t = np.asarray(t, F32)
    if t.ndim == 2 and (a.ndim != 2 or b.ndim != 2):
        t = t[..., None]
    return (a + (b - a) * t).astype(F32)


def mix(stops, t):
    """Multi-stop gradient. ``stops = [(position, colour), ...]`` with increasing positions (usually 0~1); ``t`` is a scalar or array.

    Returns colours of shape ``t.shape + (3,)``. Example: ``mix([(0, '#000'), (1, '#fff')], V)`` gives a full vertical gradient.
    """
    pos = np.array([p for p, _ in stops], F32)
    cols = np.stack([rgb(c) for _, c in stops]).astype(F32)
    t = np.clip(np.asarray(t, F32), pos[0], pos[-1])
    out = np.empty(np.shape(t) + (3,), F32)
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, cols[:, ch])
    return out


def hsv_to_rgb(h, s, v):
    """HSV -> RGB (all 0~1, arrays allowed)."""
    h = np.asarray(h, F32) % 1.0
    s = np.asarray(s, F32)
    v = np.asarray(v, F32)
    h, s, v = np.broadcast_arrays(h, s, v)
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1).astype(F32)


def luma(img):
    """Luminance (H, W)."""
    img = np.asarray(img, F32)
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def clamp01(a):
    return np.clip(np.asarray(a, F32), 0.0, 1.0)


def smoothstep(e0, e1, x):
    """Smooth step: x<=e0 -> 0, x>=e1 -> 1, S-curve in between. e0/e1 may be arrays; e0 > e1 also works (reversed)."""
    x = np.asarray(x, F32)
    e0 = np.asarray(e0, F32)
    e1 = np.asarray(e1, F32)
    d = e1 - e0
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    t = np.clip((x - e0) / d, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(F32)


def normalize(a, lo=0.0, hi=1.0):
    """Linearly map an array to lo~hi."""
    a = np.asarray(a, F32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-12:
        return np.full_like(a, lo)
    return ((a - mn) / (mx - mn) * (hi - lo) + lo).astype(F32)


def grid(h, w):
    """Return the pixel coordinate grid (X, Y), both float32 (h, w)."""
    Y, X = np.mgrid[0:h, 0:w]
    return X.astype(F32), Y.astype(F32)


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "hex": "colours are '#rrggbb' strings; color.rgb('#ff8800') returns a float (r, g, b) triple",
    "hex_to_rgb": "use color.rgb('#rrggbb') -> (r, g, b) floats 0..1",
    "rgb_to_hex": "colours are floats 0..1; format with '#%02x%02x%02x' % tuple(int(round(v * 255)) for v in rgb)",
    "hsl_to_rgb": "use color.hsv_to_rgb(h, s, v) (h, s, v in 0..1)",
    "hsv": "use color.hsv_to_rgb(h, s, v) (h, s, v in 0..1)",
    "hsl": "use color.hsv_to_rgb(h, s, v) (h, s, v in 0..1)",
    "blend": "use color.lerp(a, b, t) for a mix; blend modes are in pixelcanvas.filters.blend(base, top, alpha, mode)",
    "gradient": "use color.mix([(0.0, '#112244'), (1.0, '#ffcc88')], t) for a colour ramp (t is a float or an array)",
    "ramp": "use color.mix([(0.0, '#112244'), (1.0, '#ffcc88')], t) for a colour ramp",
    "clamp": "use color.clamp01(a) (clips to 0..1)",
    "clip": "use color.clamp01(a) (clips to 0..1)",
    "luminance": "use color.luma(img)",
    "brightness": "use color.luma(img)",
    "parse": "use color.rgb('#rrggbb') or color.as_color(c)",
    "color": "use color.rgb(c) for a (r, g, b) triple or color.as_color(c) for a float array",
    "mix_colors": "use color.lerp(a, b, t) or color.mix(stops, t)",
    "interpolate": "use color.lerp(a, b, t) or color.mix(stops, t)",
}
__getattr__ = _module_getattr(__name__, _HINTS)
