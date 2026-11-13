# -*- coding: utf-8 -*-
"""Curated colour palettes and colour tools in a perceptual space (OKLab / OKLCH).

    from pixelcanvas.palette import palette, find
    p = palette("sunset")                 # or palette("warm evening sky"), palette(["#123", "#f80", ...])
    p.bg, p.fg, p.accent, p.muted         # roles: background, readable ink, accent, quiet mid colour
    p.ramp(7), p.stops()                  # smooth OKLab ramp / gradient stops (dark -> light)
    p.text_on("#3a6ea5")                  # ink that passes WCAG 4.5 on that colour

About 60 hand-tuned palettes in four families (mood, subject, era, ui), plus generators: ``harmony``,
``tints_shades``, ``from_image`` (k-means in OKLab), ``mix_oklab``, ``contrast_ratio``, ``ensure_contrast`` and
colour-blind-safe categorical sets (``categorical``).
"""
from __future__ import annotations

import difflib
import re

import numpy as np

from .color import F32, rgb
from ._args import (unknown_name as _unknown_name, check_choice as _choice, strict as _strict,
                    strict_methods as _strict_methods)


class _UnknownName(ValueError, KeyError):
    """Unknown registry name (a ValueError that is also a KeyError)."""
    __str__ = ValueError.__str__

__all__ = ["Palette", "palette", "find", "names", "describe", "FAMILIES", "PALETTES", "srgb_to_oklab", "oklab_to_srgb",
           "oklab_to_oklch", "oklch_to_oklab", "to_hex", "oklch", "mix_oklab", "contrast_ratio", "ensure_contrast",
           "harmony", "tints_shades", "from_image", "categorical", "CATEGORICAL", "ZH_KEYWORDS", "query_tokens"]


# ---------------------------------------------------------------------------------------------- colour spaces

def _to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4)


def _from_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.maximum(c, 0) ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M2I = np.linalg.inv(_M2)
_M1I = np.linalg.inv(_M1)


def srgb_to_oklab(c):
    """sRGB (..., 3) in 0~1 -> OKLab (..., 3): L 0~1, a / b about -0.4~0.4."""
    lms = _to_linear(c) @ _M1.T
    return (np.cbrt(lms) @ _M2.T).astype(F32)


def oklab_to_srgb(lab, clip=True):
    """OKLab (..., 3) -> sRGB (..., 3); ``clip`` clamps to 0~1."""
    lms = (np.asarray(lab, np.float64) @ _M2I.T) ** 3
    out = _from_linear(lms @ _M1I.T)
    return (np.clip(out, 0, 1) if clip else out).astype(F32)


def oklab_to_oklch(lab):
    lab = np.asarray(lab, np.float64)
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360
    return np.stack([lab[..., 0], C, h], -1).astype(F32)


def oklch_to_oklab(lch):
    lch = np.asarray(lch, np.float64)
    hr = np.radians(lch[..., 2])
    return np.stack([lch[..., 0], lch[..., 1] * np.cos(hr), lch[..., 1] * np.sin(hr)], -1).astype(F32)


def _in_gamut(lab):
    s = oklab_to_srgb(lab, clip=False)
    return np.all((s >= -1e-4) & (s <= 1 + 1e-4), -1)


def oklch(L, C, h):
    """A colour from OKLCH (lightness 0~1, chroma 0~0.37, hue degrees) -> '#rrggbb', chroma reduced into gamut."""
    lo, hi = 0.0, float(C)
    if _in_gamut(oklch_to_oklab([L, hi, h])):
        return to_hex(oklab_to_srgb(oklch_to_oklab([L, hi, h])))
    for _ in range(24):
        mid = (lo + hi) / 2
        if _in_gamut(oklch_to_oklab([L, mid, h])):
            lo = mid
        else:
            hi = mid
    return to_hex(oklab_to_srgb(oklch_to_oklab([L, lo, h])))


def to_hex(c):
    """Any colour (hex, CSS name, tuple, ndarray(3)) -> '#rrggbb'."""
    a = np.clip(np.asarray(rgb(c), np.float64), 0, 1)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in a[:3])


def _lab(c):
    return srgb_to_oklab(rgb(c)[:3]).astype(np.float64)


def mix_oklab(a, b, t=0.5):
    """Perceptual mix of two colours (t = 0 -> a, 1 -> b) -> '#rrggbb'."""
    return to_hex(oklab_to_srgb(_lab(a) + (_lab(b) - _lab(a)) * float(t)))


def _lum(c):
    lin = _to_linear(np.clip(rgb(c)[:3], 0, 1))
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def contrast_ratio(a, b):
    """WCAG contrast ratio between two colours (1 ... 21). Body text wants >= 4.5, large text >= 3."""
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def ensure_contrast(fg, bg, ratio=4.5):
    """Return ``fg`` moved in lightness (hue kept) until it reaches ``ratio`` against ``bg``. -> '#rrggbb'."""
    if contrast_ratio(fg, bg) >= ratio:
        return to_hex(fg)
    L, C, h = (float(v) for v in oklab_to_oklch(_lab(fg)))
    best = None
    for target in (0.0, 1.0):                                  # try darker and lighter, keep the closer success
        lo, hi = L, target
        if contrast_ratio(oklch(target, C, h), bg) < ratio:
            continue
        for _ in range(22):
            mid = (lo + hi) / 2
            if contrast_ratio(oklch(mid, C, h), bg) >= ratio:
                hi = mid
            else:
                lo = mid
        cand = oklch(hi, C, h)
        if best is None or abs(hi - L) < best[0]:
            best = (abs(hi - L), cand)
    if best is None:
        return "#000000" if contrast_ratio("#000000", bg) >= contrast_ratio("#ffffff", bg) else "#ffffff"
    return best[1]


# ---------------------------------------------------------------------------------------------- Palette

class _Swatch(str):
    """A '#rrggbb' string that can also be called: ``p.muted`` is a colour, ``p.muted(0.5)`` a softer Palette."""

    def __call__(self, amount=0.45):
        return self._pal.desaturated(amount)


class Palette:
    """A named set of 5-7 colours with roles.

    ``colors`` (as given), ``bg`` / ``fg`` (readable ink on bg, WCAG >= 4.5) / ``accent`` (most vivid) /
    ``muted`` (quiet mid colour; ``muted(amount)`` returns a desaturated Palette) / ``dark`` / ``light``,
    ``accents`` (vivid colours, most vivid first), ``mode`` ('light' = light background, 'dark').
    """

    def __init__(self, colors, name="custom", mode=None, tags=(), family="custom", accent=None, bg=None):
        cols = [to_hex(c) for c in colors]
        if len(cols) < 2:
            raise ValueError("a palette needs at least two colours")
        self.colors = cols
        self.name = name
        self.family = family
        self.tags = tuple(tags)
        lab = np.stack([_lab(c) for c in cols])
        lch = oklab_to_oklch(lab)
        self._L, self._C = lch[:, 0].astype(float), lch[:, 1].astype(float)
        if mode is None:
            mode = "light" if self._L.max() > 0.88 or self._L.min() > 0.32 else "dark"
        self.mode = mode
        ib = int(bg) if bg is not None else int(np.argmax(self._L) if mode == "light" else np.argmin(self._L))
        self.bg = cols[ib]
        cr = [contrast_ratio(c, self.bg) for c in cols]
        ifg = int(np.argmax(cr))
        self.fg = ensure_contrast(cols[ifg], self.bg, 4.5)
        others = [i for i in range(len(cols)) if i not in (ib, ifg)] or [ifg]
        if accent is not None:
            ia = int(accent)
        else:
            vivid = [i for i in others if cr[i] >= 1.6] or others
            ia = max(vivid, key=lambda i: self._C[i])
        self.accent = cols[ia]
        order = sorted([i for i in range(len(cols)) if i != ib], key=lambda i: -self._C[i])
        self.accents = [cols[i] for i in order if self._C[i] > 0.06] or [self.accent]
        if self.accent in self.accents:
            self.accents.remove(self.accent)
        self.accents.insert(0, self.accent)
        rest = [i for i in others if i != ia]
        if rest:
            im = min(rest, key=lambda i: self._C[i] * 2 + abs(self._L[i] - 0.6))
            muted = cols[im]
        else:
            muted = mix_oklab(self.bg, self.fg, 0.45)
        self.muted = _Swatch(muted)
        self.muted._pal = self
        self.dark = cols[int(np.argmin(self._L))]
        self.light = cols[int(np.argmax(self._L))]

    # -- container behaviour
    def __iter__(self):
        return iter(self.colors)

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, i):
        return self.colors[i]

    def __repr__(self):
        return f"Palette({self.name!r}, {self.mode}, {self.colors})"

    @property
    def array(self):
        """Colours as a float32 (n, 3) array (0~1)."""
        return np.stack([rgb(c)[:3] for c in self.colors]).astype(F32)

    def sorted(self):
        """Colours ordered dark -> light."""
        return [self.colors[i] for i in np.argsort(self._L, kind="stable")]

    def ramp(self, n=7, order="lightness"):
        """``n`` colours interpolated smoothly in OKLab through the palette (``order`` = 'lightness' or 'given')."""
        cols = self.sorted() if order == "lightness" else list(self.colors)
        lab = np.stack([_lab(c) for c in cols])
        x = np.linspace(0, 1, len(cols))
        t = np.linspace(0, 1, max(int(n), 1)) if n > 1 else np.array([0.5])
        out = np.stack([np.interp(t, x, lab[:, k]) for k in range(3)], -1)
        return [to_hex(c) for c in oklab_to_srgb(out)]

    def stops(self, order="lightness"):
        """Gradient stops ``[(0, c0), ..., (1, cn)]`` for ``cv.gradient`` (dark -> light by default)."""
        cols = self.sorted() if order == "lightness" else list(self.colors)
        n = len(cols)
        return [(i / (n - 1), c) for i, c in enumerate(cols)]

    def text_on(self, color, ratio=4.5):
        """Ink that reads on ``color``: the palette colour with the best contrast, pushed to ``ratio`` if needed."""
        cands = [self.fg, self.bg, self.dark, self.light]
        best = max(cands, key=lambda c: contrast_ratio(c, color))
        return ensure_contrast(best, color, ratio)

    def _map(self, fn, suffix):
        lch = [oklab_to_oklch(_lab(c)) for c in self.colors]
        cols = [oklch(*fn(float(L), float(C), float(h))) for L, C, h in lch]
        return Palette(cols, f"{self.name}{suffix}", self.mode, self.tags, self.family)

    def lighter(self, amount=0.08):
        """Every colour lighter by ``amount`` OKLab lightness."""
        return self._map(lambda L, C, h: (min(L + amount, 0.995), C, h), "+light")

    def darker(self, amount=0.08):
        """Every colour darker by ``amount`` OKLab lightness."""
        return self._map(lambda L, C, h: (max(L - amount, 0.02), C, h), "+dark")

    def desaturated(self, amount=0.45):
        """Chroma reduced by ``amount`` (0~1) -> a quieter palette (same as ``p.muted(amount)``)."""
        return self._map(lambda L, C, h: (L, C * (1 - amount), h), "+muted")

    def saturated(self, amount=0.3):
        """Chroma increased by ``amount`` (fraction), clipped into gamut."""
        return self._map(lambda L, C, h: (L, C * (1 + amount), h), "+vivid")

    def inverted(self):
        """The same colours with the other mode (light background <-> dark background)."""
        return Palette(self.colors, self.name + "+inv", "dark" if self.mode == "light" else "light", self.tags,
                       self.family)

    def shuffle(self, seed=0):
        """Same colours in a seeded random order (changes which colour patterns pick first)."""
        idx = np.random.default_rng(seed).permutation(len(self.colors))
        return Palette([self.colors[i] for i in idx], self.name, self.mode, self.tags, self.family)

    def pick(self, i):
        """Colour ``i`` of the vivid accents, cycling."""
        return self.accents[i % len(self.accents)]

    def to_dict(self):
        return dict(name=self.name, family=self.family, mode=self.mode, colors=list(self.colors), bg=self.bg,
                    fg=self.fg, accent=self.accent, muted=str(self.muted), dark=self.dark, light=self.light)


# ---------------------------------------------------------------------------------------------- the library

_DEFS = {
    "mood": [
        ("calm", "light", "#f4f1ea #dfe7e4 #a9c5c0 #6f9a9a #3f6870 #23394a",
         "calm serene peaceful quiet relaxing spa wellness soft sage zen"),
        ("energetic", "light", "#fff4e0 #ffd23f #ff8a1f #f2385a #3b2bd1 #14112b",
         "energetic vibrant bold sport dynamic loud fitness festival"),
        ("luxurious", "dark", "#0e0c0a #1f1a17 #4a3a2a #8c6a3e #c9a45c #f1e3c2",
         "luxurious luxury gold black premium elegant jewelry rich"),
        ("playful", "light", "#fffaf0 #ffd6e0 #ffc94d #4dd4c6 #ff6b6b #5b4bd8 #2d2147",
         "playful fun kids children cheerful toy party happy"),
        ("corporate", "light", "#f6f8fb #dce3ec #8fa3bc #2f6fde #1d3d6e #0f1b2d",
         "corporate business professional finance trust office report"),
        ("earthy", "light", "#f2ebe0 #d9c3a5 #b08a5c #7c8a5c #5a4632 #2e241b",
         "earthy natural organic soil clay rustic handmade"),
        ("dreamy", "light", "#fbf6ff #e8dcff #ffd0e8 #b9d8ff #9a8cf0 #4b3f7a",
         "dreamy pastel soft fantasy ethereal cloud whimsical"),
        ("retro", "light", "#f6ead0 #f2b134 #e5652e #c8372d #2d7d7a #2b2a28",
         "retro 70s seventies vintage groovy warm"),
        ("cyber", "dark", "#07070f #151530 #2b1a5c #ff2bd6 #00f0ff #e8f7ff",
         "cyber cyberpunk neon tech futuristic hacker glow electric"),
        ("minimal", "light", "#fafafa #ececec #c8c8c8 #8a8a8a #ff5a36 #1a1a1a",
         "minimal minimalist clean simple modern white gray grey"),
        ("romantic", "light", "#fff3f1 #f7d4d0 #e8a0a5 #c9607a #8a2f4f #3d1424",
         "romantic love wedding valentine rose blush pink"),
        ("moody", "dark", "#0d1117 #1b2430 #34404f #6b7b8c #c9a96e #e8e4dc",
         "moody dark cinematic noir dramatic night"),
    ],
    "subject": [
        ("sunset", "dark", "#1a1335 #4a2161 #a3306b #e2574c #f59a5a #fcd59c",
         "sunset dusk golden hour warm evening sky orange sun"),
        ("ocean", "light", "#e9f6f7 #9fd8df #4fb0c6 #1f7fa6 #0e4c74 #08263f",
         "ocean sea water marine blue beach wave coast"),
        ("forest", "dark", "#0f1d17 #1f3a2c #35593f #5f8a55 #a9c27a #ecf0d8",
         "forest woods trees green nature pine jungle"),
        ("desert", "light", "#f7ead6 #ecc9a0 #d99a6c #b8653f #7d3c2a #3a2420",
         "desert sand dunes arid sahara canyon warm"),
        ("autumn", "light", "#f7efe3 #f0b35a #d9722f #a63d21 #6b2f2a #2f2420",
         "autumn fall leaves harvest pumpkin maple warm"),
        ("spring", "light", "#fbfff6 #d9f2c4 #9fdb8a #ffc8dd #ff8fab #2f5d3a",
         "spring blossom fresh flowers garden easter"),
        ("summer", "light", "#fffbea #ffe066 #ffa94d #4ecdc4 #1a9fb8 #12355b",
         "summer sunny beach bright holiday pool"),
        ("winter", "light", "#f5f9fc #dbe8f1 #a8c4d9 #6c8fae #3a5578 #1b2a41",
         "winter snow ice cold frost christmas cool blue"),
        ("night_city", "dark", "#070b1a #141b38 #2c2f6b #5b3fa0 #ff4f9a #ffc857",
         "night city urban neon lights skyline downtown"),
        ("candy", "light", "#fff5fb #ffc4e1 #ff8cc6 #b5f0e0 #8ad8ff #a77bff #4a2a6a",
         "candy sweet pastel pink sugar dessert bubblegum"),
        ("coffee", "light", "#f5ede3 #dcc3a8 #b08968 #7f5539 #4a3123 #241812",
         "coffee cafe latte brown espresso chocolate bakery"),
        ("tropical", "light", "#fdf8e8 #ffcf56 #ff6f59 #2ec4b6 #0f8b6d #0b3c49",
         "tropical jungle palm island paradise exotic"),
        ("lavender", "light", "#f8f5fc #e3d9f2 #c3aee3 #9a7fc9 #6a52a3 #2e2346",
         "lavender purple violet floral provence lilac"),
        ("aurora", "dark", "#050a18 #0b1f3a #0f4c5c #19c3a1 #7cf29b #b388ff",
         "aurora northern lights polar arctic glow teal"),
        ("sakura", "light", "#fff7f8 #fde2e7 #f7b9c6 #e8879f #b55a7a #4a2c3d",
         "sakura cherry blossom japan pink spring petal"),
        ("deep_space", "dark", "#03040b #0b1030 #1d1f5e #4b2a8c #e05ea8 #f5f0ff",
         "space galaxy cosmos stars universe nebula night"),
    ],
    "era": [
        ("bauhaus", "light", "#f2ece0 #e63a2e #f2b705 #1e5aa8 #1a1a1a",
         "bauhaus modernist primary geometric constructivist"),
        ("memphis", "light", "#fff8ef #ff6fb5 #ffd23f #2ec4f1 #5b3ee6 #111111",
         "memphis 80s eighties postmodern confetti funky"),
        ("art_deco", "dark", "#0f1418 #1c2a33 #2f4a52 #b08d57 #e3c88f #f4ead5",
         "art deco gatsby 1920s twenties gold geometric elegant"),
        ("mid_century", "light", "#f3e9d2 #e9a23b #d9572b #2c7873 #6b8f71 #2b2d2f",
         "mid century modern 50s 60s fifties sixties atomic"),
        ("ukiyo_e", "light", "#efe4cc #d8c39a #c2452d #2f5d7c #1c2f45 #22201c",
         "ukiyo-e ukiyoe japanese woodblock print wave hokusai edo"),
        ("pop_art", "light", "#fff5d6 #ffde00 #ff3b3b #1e90ff #ff8fc8 #111111",
         "pop art comic warhol lichtenstein halftone bold"),
        ("vaporwave", "dark", "#1b1035 #3d1f6b #ff71ce #01cdfe #05ffa1 #fffb96",
         "vaporwave aesthetic 90s nineties pastel neon"),
        ("synthwave", "dark", "#0d0221 #261447 #6b1f7a #ff2a6d #ff9e3d #05d9e8",
         "synthwave outrun retrowave 80s eighties neon grid sunset"),
        ("nordic", "light", "#f4f2ee #e2ddd5 #b9b2a6 #7d8a86 #3f4a4c #c27c5b",
         "nordic scandinavian hygge muted interior calm"),
        ("terracotta", "light", "#f6ece2 #e7c3a7 #d08159 #b4532f #6f7c5a #3a2a24",
         "terracotta mediterranean clay pottery warm boho"),
        ("swiss", "light", "#f5f5f2 #d9d9d4 #8f8f8a #e10600 #111111",
         "swiss international typographic style grid red helvetica"),
        ("risograph", "light", "#f8f3e6 #ff48b0 #0078bf #ffe800 #00a95c #3a3a3a",
         "risograph riso print zine indie"),
        ("art_nouveau", "light", "#f0e6cf #c9b27c #8a9a5b #4f6b4a #a3573a #2e2a24",
         "art nouveau organic mucha botanical ornamental"),
        ("y2k", "light", "#f2f5ff #c7d2fe #9be7ff #ff9af3 #b8ff5c #2a2a5a",
         "y2k millennium chrome 2000s bubbly cyber pastel"),
    ],
}

_TW = {
    "indigo": "#eef2ff #e0e7ff #c7d2fe #a5b4fc #818cf8 #6366f1 #4f46e5 #4338ca #3730a3 #312e81 #1e1b4b",
    "emerald": "#ecfdf5 #d1fae5 #a7f3d0 #6ee7b7 #34d399 #10b981 #059669 #047857 #065f46 #064e3b #022c22",
    "rose": "#fff1f2 #ffe4e6 #fecdd3 #fda4af #fb7185 #f43f5e #e11d48 #be123c #9f1239 #881337 #4c0519",
    "amber": "#fffbeb #fef3c7 #fde68a #fcd34d #fbbf24 #f59e0b #d97706 #b45309 #92400e #78350f #451a03",
    "slate": "#f8fafc #f1f5f9 #e2e8f0 #cbd5e1 #94a3b8 #64748b #475569 #334155 #1e293b #0f172a #020617",
    "sky": "#f0f9ff #e0f2fe #bae6fd #7dd3fc #38bdf8 #0ea5e9 #0284c7 #0369a1 #075985 #0c4a6e #082f49",
    "violet": "#f5f3ff #ede9fe #ddd6fe #c4b5fd #a78bfa #8b5cf6 #7c3aed #6d28d9 #5b21b6 #4c1d95 #2e1065",
    "teal": "#f0fdfa #ccfbf1 #99f6e4 #5eead4 #2dd4bf #14b8a6 #0d9488 #0f766e #115e59 #134e4a #042f2e",
    "orange": "#fff7ed #ffedd5 #fed7aa #fdba74 #fb923c #f97316 #ea580c #c2410c #9a3412 #7c2d12 #431407",
}
_UI_WORDS = {"indigo": "indigo blue saas app", "emerald": "emerald green success money eco",
             "rose": "rose red pink fashion beauty", "amber": "amber yellow gold warning honey",
             "slate": "slate gray grey neutral dashboard", "sky": "sky blue light airy weather",
             "violet": "violet purple creative ai", "teal": "teal cyan health medical fresh",
             "orange": "orange tangerine food energy"}

PALETTES = {}
FAMILIES = {}


def _register():
    for fam, rows in _DEFS.items():
        for name, mode, cols, words in rows:
            PALETTES[name] = Palette(cols.split(), name, mode, words.split(), fam)
            FAMILIES.setdefault(fam, []).append(name)
    sl = _TW["slate"].split()
    for brand, scale in _TW.items():
        s = scale.split()
        acc_l = s[6] if brand not in ("amber",) else s[7]
        if brand == "slate":
            light = [s[0], s[2], s[4], "#3b82f6", s[6], s[9]]
            dark = [s[10], s[8], s[6], "#60a5fa", s[3], s[0]]
        else:
            light = [sl[0], sl[2], s[2], acc_l, sl[5], sl[9]]
            dark = [sl[10], sl[8], sl[5], s[4], s[2], sl[0]]
        words = ("ui interface app web brand " + _UI_WORDS[brand]).split()
        PALETTES[brand] = Palette(light, brand, "light", words + ["light"], "ui", accent=3)
        PALETTES[brand + "_dark"] = Palette(dark, brand + "_dark", "dark", words + ["dark", "night"], "ui", accent=3)
        FAMILIES.setdefault("ui", []).extend([brand, brand + "_dark"])


_register()

# Traditional Chinese keywords -> English search words (used by palette.find and backdrop.find)
ZH_KEYWORDS = {
    "夕陽": "sunset", "日落": "sunset", "黃昏": "sunset dusk", "晚霞": "sunset", "日出": "sunrise sun", "太陽": "sun",
    "海洋": "ocean sea", "大海": "ocean sea", "海": "ocean sea", "海浪": "wave ocean", "波浪": "wave", "水": "water",
    "森林": "forest", "樹": "tree forest", "山": "mountain", "丘陵": "hills", "沙漠": "desert", "沙丘": "dunes desert",
    "秋": "autumn", "春": "spring", "夏": "summer", "冬": "winter", "雪": "snow winter", "夜": "night",
    "城市": "city", "都市": "city urban", "糖果": "candy", "咖啡": "coffee", "熱帶": "tropical", "平靜": "calm",
    "寧靜": "calm serene", "活力": "energetic", "奢華": "luxurious luxury", "可愛": "playful cute", "童趣": "playful kids",
    "商務": "corporate business", "企業": "corporate", "大地": "earthy", "自然": "nature", "夢幻": "dreamy",
    "復古": "retro vintage", "懷舊": "retro vintage", "賽博": "cyber", "科技": "tech", "極簡": "minimal",
    "簡約": "minimal", "浪漫": "romantic", "暖": "warm", "溫暖": "warm", "冷": "cool cold", "粉彩": "pastel",
    "霓虹": "neon", "金": "gold", "藍": "blue", "綠": "green", "紅": "red", "粉紅": "pink", "粉": "pink",
    "紫": "purple", "黃": "yellow", "橙": "orange", "橘": "orange", "黑": "black dark", "白": "white", "灰": "gray",
    "深色": "dark", "淺色": "light", "星空": "starry stars night", "星": "stars", "宇宙": "space galaxy",
    "極光": "aurora", "櫻花": "sakura cherry", "薰衣草": "lavender", "包浩斯": "bauhaus", "浮世繪": "ukiyo-e",
    "普普": "pop art", "蒸汽波": "vaporwave", "北歐": "nordic", "陶土": "terracotta", "漸層": "gradient",
    "網格": "grid", "點": "dots", "條紋": "stripes", "幾何": "geometric", "水彩": "watercolor", "墨": "ink",
    "大理石": "marble", "煙": "smoke", "紙": "paper", "木": "wood", "雲": "clouds", "光": "light glow",
    "玻璃": "glass", "等高線": "topographic contour", "六角": "hexagon", "三角": "triangle", "圓": "circle",
    "水下": "underwater", "海底": "underwater", "天際線": "skyline city", "柔和": "soft pastel",
}


def query_tokens(q):
    """Lower-case English words of a query, with Traditional Chinese keywords translated."""
    q = str(q)
    extra = []
    for zh in sorted(ZH_KEYWORDS, key=len, reverse=True):
        if zh in q:
            extra.append(ZH_KEYWORDS[zh])
            q = q.replace(zh, " ")
    words = re.findall(r"[a-z0-9]+", (q + " " + " ".join(extra)).lower().replace("-", " "))
    out = []
    for w in words:
        out.append(w)
        if len(w) > 4 and w.endswith("s"):
            out.append(w[:-1])
    return out


_STOP = {"a", "an", "the", "of", "over", "with", "and", "in", "on", "for", "to", "at", "by", "my", "some", "very"}


def _score(tokens, name, words):
    nparts = set(name.replace("_", " ").split())
    ws = set(words)
    s = 0.0
    for t in tokens:
        if t in _STOP:
            continue
        if t == name or t in nparts:
            s += 3
        elif t in ws:
            s += 2
        elif len(t) >= 4 and any(w.startswith(t) or t.startswith(w) for w in ws | nparts if len(w) >= 4):
            s += 0.8
    return s


def find(query, n=1):
    """Best palette name for a description ("warm sunset over the sea", "夢幻粉彩") -> name, or a list of ``n``.
    Returns None (or []) when nothing matches."""
    toks = query_tokens(query)
    dark = "dark" in toks or "night" in toks
    scored = []
    for i, (name, p) in enumerate(PALETTES.items()):
        s = _score(toks, name, p.tags + (p.family,))
        if s > 0 and p.family == "ui":
            s += 0.3 if (p.mode == "dark") == dark else 0
        if s > 0:
            scored.append((-s, i, name))
    scored.sort()
    res = [nm for _, _, nm in scored[:n]]
    return (res[0] if res else None) if n == 1 else res


def names(tag=None):
    """All palette names, or those whose family or tags contain ``tag`` (e.g. 'ui', 'mood', 'warm', 'dark')."""
    if tag is None:
        return list(PALETTES)
    tag = tag.lower()
    return [n for n, p in PALETTES.items() if tag == p.family or tag in p.tags or tag in n]


def describe(name=None):
    """One line per palette (or for one palette): name, family, mode, colours, tags."""
    rows = [name] if name else list(PALETTES)
    out = []
    for n in rows:
        p = palette(n)
        out.append(f"{p.name:<14} {p.family:<8} {p.mode:<5} {' '.join(p.colors)}  [{' '.join(p.tags[:6])}]")
    return "\n".join(out)


COLOR_WORDS = {
    "sage": "#9caf88", "cream": "#f6efdd", "ivory": "#f8f4e6", "peach": "#ffcba4", "blush": "#f4c6c6",
    "coral": "#ff7f6e", "salmon": "#fa8f7d", "mint": "#a8e6cf", "seafoam": "#9fe0c9", "navy": "#1f2f57",
    "baby blue": "#a7cdf0", "powder blue": "#b6cfe8", "sky blue": "#8ec5e8", "periwinkle": "#9aa5e8",
    "lavender": "#b8a9e0", "lilac": "#c9a8e0", "mauve": "#b48ca8", "plum": "#6e3a6e", "burgundy": "#6d1f35",
    "wine": "#6d1f35", "maroon": "#6b1d2a", "terracotta": "#c4694a", "rust": "#b4532a", "clay": "#b9715a",
    "mustard": "#d9a52a", "amber": "#f0a31c", "gold": "#d4a937", "honey": "#e8b04a", "lemon": "#f5e26a",
    "olive": "#7a8040", "moss": "#6f8a4a", "forest green": "#2f5d3a", "emerald": "#2aa876",
    "teal": "#2a8c8c", "turquoise": "#40c4c0", "aqua": "#6fd6d6", "cyan": "#35c4e0", "indigo": "#3f3fa0",
    "charcoal": "#2e3138", "slate": "#5a6a7c", "graphite": "#3a3d44", "beige": "#e3d5bc", "sand": "#dcc9a0",
    "taupe": "#a89886", "tan": "#c9a67c", "chocolate": "#5a3825", "coffee": "#6a4632", "espresso": "#3a2418",
    "rose": "#e0728a", "fuchsia": "#d63fa0", "magenta": "#c43fa0", "crimson": "#c0243c", "scarlet": "#e03a2c",
    "tangerine": "#ff8a2a", "apricot": "#fbc08a", "lime": "#a6d83c", "pine": "#2d5a46", "ocean": "#2f7fa8",
    "silver": "#c3c8cf", "grey": "#8a8f98", "gray": "#8a8f98", "white": "#ffffff", "black": "#14161a",
    "red": "#d63a3a", "orange": "#f0822a", "yellow": "#f2cf3a", "green": "#3fa35a", "blue": "#3a7bd5",
    "purple": "#7a4ac8", "pink": "#f08ab4", "brown": "#8a5a3a", "violet": "#7c4fd0", "bronze": "#a97142",
}
_BASIC_COLOURS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "gray", "grey", "black",
                  "white", "gold", "violet", "teal", "indigo", "rose", "amber", "slate", "emerald", "lavender",
                  "coffee", "ocean"}
_SHADE_WORDS = {"soft": (0.55, 0.6), "pastel": (0.7, 0.55), "light": (0.5, 0.8), "pale": (0.7, 0.4),
                "dusty": (0.15, 0.55), "muted": (0.1, 0.5), "dark": (-0.5, 1.0), "deep": (-0.4, 1.05),
                "bright": (0.1, 1.35), "vivid": (0.0, 1.4)}
_NOUN_AFTER = {"green", "blue", "grey", "gray", "pink", "brown", "red", "orange"}
_NOUN_BEFORE = {"sage", "mint", "olive", "moss", "navy", "peach", "coral", "rose", "sky", "baby", "powder"}


def _colour_words(text):
    """Colours named in a description ("soft blue and peach", "sage green, cream") -> [(word, hex), ...] in order.
    Shade words (soft, pastel, light, pale, dusty, muted, dark, deep, bright, vivid) adjust the next colour."""
    toks = re.findall(r"[a-z]+", str(text).lower())
    out, i, shade = [], 0, None
    while i < len(toks):
        w = toks[i]
        hit, hexv = None, None
        if i + 1 < len(toks) and (w + " " + toks[i + 1]) in COLOR_WORDS:
            hit, hexv, i = w + " " + toks[i + 1], COLOR_WORDS[w + " " + toks[i + 1]], i + 2
        elif w in COLOR_WORDS:
            hit, hexv, i = w, COLOR_WORDS[w], i + 1
            if w in _NOUN_BEFORE and i < len(toks) and toks[i] in _NOUN_AFTER:
                i += 1                                       # "sage green", "navy blue": the second word is a noun
        elif w in _SHADE_WORDS:
            shade = _SHADE_WORDS[w]
            i += 1
            continue
        elif len(w) >= 4:
            try:
                hexv, hit = to_hex(w), w
            except ValueError:
                pass
            i += 1
        else:
            i += 1
        if hit is None:
            continue
        if shade is not None:
            L, C, h = (float(v) for v in oklab_to_oklch(_lab(hexv)))
            dl, cm = shade
            L = L + (0.9 - L) * dl if dl > 0 else L * (1 + dl)
            hexv = oklch(min(max(L, 0.05), 0.97), C * cm, h)
            shade = None
        out.append((hit, hexv))
    return out


def _palette_from_colours(cols, dark=False, name="colours"):
    """A 4-7 colour Palette that contains the given colours plus a tinted light, a deep ink and a mid shade."""
    lch = [oklab_to_oklch(_lab(c)) for c in cols]
    k = int(np.argmax([float(v[1]) for v in lch]))
    hue = float(lch[k][2])
    chroma = max(float(lch[k][1]), 0.05)
    cand = list(cols)
    Ls = [float(v[0]) for v in lch]
    if max(Ls) < 0.92:
        cand.append(oklch(0.965, 0.016, hue))
    if min(Ls) > 0.34:
        cand.append(oklch(0.27, min(chroma * 0.55, 0.06), hue))
    if not any(0.42 <= L <= 0.66 for L in Ls):
        cand.append(oklch(0.56, chroma * 0.8, hue))
    out = []
    for c in sorted(cand, key=lambda c: float(_lab(c)[0])):         # drop near-duplicates
        if all(float(np.linalg.norm(_lab(c) - _lab(o))) > 0.045 for o in out):
            out.append(to_hex(c))
    if len(out) < 4:                                                 # one or two colours: fill in with tints
        out = tints_shades(out[len(out) // 2], 5)
    if len(out) > 7:
        out = [out[i] for i in np.round(np.linspace(0, len(out) - 1, 7)).astype(int)]
    return Palette(out, name, "dark" if dark else "light", ("colours",), "generated")


def _mood_hit(spec):
    """True when the words besides the colour names pick a non-UI palette (a subject or mood: 'warm sunset')."""
    toks = [t for t in query_tokens(spec) if t not in _STOP and t not in COLOR_WORDS and t not in _SHADE_WORDS]
    h = find(" ".join(toks)) if toks else None
    return h is not None and PALETTES[h].family != "ui"


def palette(spec=None, n=None):
    """Get a Palette from a name ('sunset'), a description ('soft pastel dreamy', 'calm, sage green, cream': named
    colours are kept, with a tinted light and a deep ink added), a list of colours, a single colour (an analogous
    harmony around it) or a Palette. ``n`` resamples it to n colours (OKLab ramp)."""
    if spec is None:
        spec = "calm"
    if isinstance(spec, Palette):
        p = spec
    elif isinstance(spec, str):
        key = spec.strip().lower().replace(" ", "_").replace("-", "_")
        if key in PALETTES:
            p = PALETTES[key]
        else:
            try:
                rgb(spec)
                p = harmony(spec, "analogous")
            except ValueError:
                found = _colour_words(spec)
                only_basic = len(found) == 1 and found[0][0] in _BASIC_COLOURS
                if found and not (only_basic and _mood_hit(spec)):
                    dark = any(t in spec.lower().split() for t in ("dark", "night", "midnight", "dusk"))
                    p = _palette_from_colours([c for _, c in found], dark, name=" ".join(w for w, _ in found))
                else:
                    hit = find(spec)
                    if hit is None:
                        raise _UnknownName(*_unknown_name("palette", spec, list(PALETTES),
                                                          "Try palette.find('...') or palette.names()").args)
                    p = PALETTES[hit]
    else:
        cols = list(spec)
        p = Palette(cols)
    if n is not None and n != len(p):
        if n < len(p):
            idx = np.round(np.linspace(0, len(p) - 1, n)).astype(int)
            cols = [p.sorted()[i] for i in idx]
        else:
            cols = p.ramp(n)
        p = Palette(cols, p.name, p.mode, p.tags, p.family)
    return p


# ---------------------------------------------------------------------------------------------- generators

_HUES = {"analogous": [-32, -16, 0, 16, 32], "complementary": [0, 180], "split": [0, 150, 210],
         "triadic": [0, 120, 240], "tetradic": [0, 90, 180, 270], "monochrome": [0]}


def harmony(base, kind="analogous", n=5):
    """A harmonious Palette from one colour. kind: analogous / complementary / split / triadic / tetradic /
    monochrome. Lightness is spread from a usable dark to a usable light."""
    _choice("harmony", "kind", kind, list(_HUES))
    L0, C0, h0 = (float(v) for v in oklab_to_oklch(_lab(base)))
    C0 = max(C0, 0.04)
    n = max(int(n), 2)
    Ls = np.linspace(0.22, 0.96, n)
    hues = _HUES[kind]
    order = sorted(range(n), key=lambda i: abs(Ls[i] - L0))        # base hue at the lightness nearest the base
    cols = [None] * n
    for k, i in enumerate(order):
        h = h0 + hues[(k * (len(hues) // 2 + 1)) % len(hues)] if len(hues) > 1 else h0
        if k == 0:
            cols[i] = to_hex(base)
            continue
        C = C0 * (1 - 0.85 * abs(Ls[i] - 0.62) / 0.4) if kind != "monochrome" else C0 * (1 - abs(Ls[i] - 0.6))
        cols[i] = oklch(float(Ls[i]), max(C, 0.012), h)
    return Palette(cols, f"{kind}:{to_hex(base)}", None, (kind,), "generated")


def tints_shades(base, n=9):
    """``n`` colours of one hue from dark shade to light tint (OKLCH, chroma eased at both ends) -> list of hex."""
    L0, C0, h = (float(v) for v in oklab_to_oklch(_lab(base)))
    Ls = np.linspace(0.16, 0.97, max(int(n), 2))
    out = [oklch(float(L), C0 * (1 - (abs(L - L0) / max(L0, 1 - L0)) ** 2 * 0.8), h) for L in Ls]
    out[int(np.argmin(np.abs(Ls - L0)))] = to_hex(base)
    return out


def from_image(img, n=5, seed=0):
    """Dominant colours of an image (path, PIL image, Canvas or array) by k-means in OKLab -> Palette."""
    from PIL import Image
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    if hasattr(img, "buf"):
        arr = np.asarray(img.buf, F32)
    elif isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"), F32) / 255.0
    else:
        arr = np.asarray(img, F32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    h, w = arr.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / 96)))
    pts = srgb_to_oklab(np.clip(arr[::step, ::step, :3], 0, 1)).reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(seed)
    k = min(int(n), len(pts))
    cent = [pts[rng.integers(len(pts))]]
    for _ in range(1, k):                                         # k-means++ seeding
        d = np.min([((pts - c) ** 2).sum(1) for c in cent], 0)
        cent.append(pts[rng.choice(len(pts), p=d / d.sum())] if d.sum() > 0 else pts[rng.integers(len(pts))])
    cent = np.array(cent)
    for _ in range(16):
        lab = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if np.any(lab == j):
                cent[j] = pts[lab == j].mean(0)
    cent = cent[np.argsort(cent[:, 0])]
    return Palette([to_hex(c) for c in oklab_to_srgb(cent)], "from_image", None, ("image",), "generated")


CATEGORICAL = {
    "okabe_ito": ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7", "#000000"],
    "tol_bright": ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb"],
    "tol_vibrant": ["#ee7733", "#0077bb", "#33bbee", "#ee3377", "#cc3311", "#009988", "#bbbbbb"],
    "tol_muted": ["#332288", "#88ccee", "#44aa99", "#117733", "#999933", "#ddcc77", "#cc6677", "#882255", "#aa4499"],
    "ibm": ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000"],
}


def categorical(n=8, scheme="okabe_ito"):
    """Colour-blind-safe categorical colours (Okabe-Ito, Paul Tol bright / vibrant / muted, IBM) -> list of hex.
    More than the scheme holds cycles through it with lighter variants."""
    if scheme not in CATEGORICAL:
        raise _UnknownName(*_unknown_name("categorical scheme", scheme, list(CATEGORICAL)).args)
    base = CATEGORICAL[scheme]
    out = []
    for i in range(int(n)):
        c = base[i % len(base)]
        rnd = i // len(base)
        out.append(c if rnd == 0 else mix_oklab(c, "#ffffff", min(0.25 * rnd, 0.7)))
    return out


# ---- strict options: unknown keyword names raise TypeError naming the closest valid option; colour / color,
# ---- w / width ... are accepted (pixelcanvas._args)
_strict_methods(Palette)
for _n, _f in list(globals().items()):
    if (not _n.startswith("_") and callable(_f) and getattr(_f, "__module__", None) == __name__
            and not isinstance(_f, type)):
        globals()[_n] = _strict(_f, _n)
