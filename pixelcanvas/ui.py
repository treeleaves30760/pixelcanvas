# -*- coding: utf-8 -*-
"""UI mock-ups: describe a screen declaratively, get a Figma-grade picture (raster *and* SVG).

Quick start::

    from pixelcanvas.ui import Screen
    s = Screen("phone", theme="dark", accent="#7c5cff")
    s.app_bar("Wallet", trailing=["bell"])
    with s.card():
        s.caption("Total balance")
        s.heading("$12,480.20", level=1)
    s.list_row("Coffee Lab", "Today, 8:12", leading="cart", value="-$4.50")
    s.tab_bar([("home", "Home"), ("chart", "Stats"), ("user", "Profile")])
    cv = s.render(scale=2)                       # Canvas; s.save_svg("wallet.svg") for the same picture as SVG

Everything is laid out by a flexbox-like box model (``vstack`` / ``hstack`` / ``grid`` / ``spacer``; fixed, fill and
hug sizing; ``max_w``; scroll-clipped regions; overlays). Text wraps and truncates with an ellipsis inside its box.
Built on ``pixelcanvas.vector``: every screen is one Drawing (one layer per screen section), so ``draw`` paints a
Canvas and ``save_svg`` writes the same geometry. Glass blur and Canvas images are resolved at draw time.
"""
from __future__ import annotations

import base64
import colorsys
import copy as _copy
import io
import math
import re
import zlib
from functools import lru_cache

import numpy as np
from PIL import Image

from . import vector as _v
from .vector import Drawing, Group, ICONS, Theme, linear_gradient, radial_gradient, measure_text
from .filters import gaussian_blur

__all__ = ["Screen", "Presentation", "Node", "template", "TEMPLATES", "DEVICES", "STYLES", "TYPE", "tokens",
           "UI_ICONS", "icon_names", "annotate", "showcase", "mix", "alpha", "window", "WINDOW_KINDS", "COMPONENT_NAMES"]

_RES = getattr(Image, "Resampling", Image)
_TRANSFORM = getattr(Image, "Transform", Image)

STYLES = ("modern", "glass", "material", "wireframe")
DEFAULT_ACCENT = "#6366f1"

# --------------------------------------------------------------------------------------------- colours


def _rgb(c):
    if isinstance(c, str):
        h = c.strip().lstrip("#")
        if len(h) in (3, 4):
            h = "".join(ch * 2 for ch in h)
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    a = [float(v) for v in list(c)[:3]]
    if max(a) > 1.0:
        a = [v / 255.0 for v in a]
    return tuple(a)


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(round(min(max(v, 0.0), 1.0) * 255)) for v in c)


def mix(a, b, t):
    """``t`` of colour ``b`` mixed into colour ``a`` → '#rrggbb'."""
    A, B = _rgb(a), _rgb(b)
    return _hex([x + (y - x) * t for x, y in zip(A, B)])


def alpha(c, a):
    """Colour ``c`` with opacity ``a`` (0~1) → '#rrggbbaa'."""
    return _hex(_rgb(c)) + "%02x" % int(round(min(max(a, 0.0), 1.0) * 255))


def _lum(c):
    r, g, b = _rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _on(c, dark="#0f1115", light="#ffffff"):
    """Readable text colour on background ``c``."""
    return dark if _lum(c) > 0.62 else light


def _hue(c, dh=0.0, sat=None, light=None):
    r, g, b = _rgb(c)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + dh / 360.0) % 1.0
    if sat is not None:
        s = sat
    if light is not None:
        l = light
    return _hex(colorsys.hls_to_rgb(h, l, s))


def _hash(s):
    return zlib.crc32(str(s).encode("utf-8")) & 0xFFFFFFFF


# --------------------------------------------------------------------------------------------- design tokens

TYPE = {                     # role: (size, weight, line-height ratio)
    "display": (44, "bold", 1.12), "h1": (32, "bold", 1.18), "h2": (26, "bold", 1.22), "h3": (20, "bold", 1.3),
    "title": (17, "bold", 1.3), "body": (15, "normal", 1.45), "label": (14, "bold", 1.3),
    "small": (13, "normal", 1.4), "caption": (12, "normal", 1.35), "overline": (11, "bold", 1.3),
    "link": (15, "bold", 1.45),
}

_RADII = {
    "modern": dict(card=16, control=12, button=12, chip=999, small=8, sheet=28, fab=18),
    "material": dict(card=16, control=6, button=999, chip=8, small=8, sheet=28, fab=16),
    "glass": dict(card=24, control=14, button=999, chip=999, small=10, sheet=32, fab=999),
    "wireframe": dict(card=6, control=4, button=4, chip=999, small=4, sheet=8, fab=999),
}


class Tokens(dict):
    """Design tokens of one screen: colours, radii, shadows (attribute access: ``T.accent``, ``T.r['card']``)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def soft(self, c, k=1.0):
        """Pale tint of ``c`` for chips, selected rows, icon backgrounds."""
        if self["style"] == "glass":
            return alpha(c, 0.26 * k)
        return mix(self["surface"] if self["style"] != "material" else self["bg"], c,
                   (0.2 if self["dark"] else 0.12) * k)

    def tone(self, name):
        """'accent' / 'success' / 'warning' / 'danger' / 'info' / 'neutral' / a colour → colour."""
        if name in (None, "accent", "primary"):
            return self["accent"]
        if name in ("success", "warning", "danger", "info"):
            return self[name]
        if name in ("neutral", "muted", "grey", "gray"):
            return self["ink2"]
        return name


def tokens(theme="light", style="modern", accent=None):
    """Design tokens for ``theme`` (light / dark), ``style`` (modern / glass / material / wireframe) and ``accent``."""
    if style not in STYLES:
        from ._args import check_choice
        check_choice("ui", "style", style, list(STYLES))
    if str(theme).lower() not in ("light", "dark"):
        from ._args import check_choice
        check_choice("ui", "theme", theme, ["light", "dark"])
    dark = str(theme).lower() == "dark"
    acc = accent or DEFAULT_ACCENT
    if dark:
        c = dict(bg="#0c0e13", surface="#161a22", surface2="#1e232d", surface3="#262c38", border="#272d39",
                 border_strong="#3a4252", ink="#f2f4f8", ink2="#a3abba", ink3="#6c7486",
                 success="#34d399", warning="#fbbf24", danger="#f87171", info="#38bdf8")
    else:
        c = dict(bg="#f5f6f8", surface="#ffffff", surface2="#f1f3f6", surface3="#e7eaef", border="#e4e7ec",
                 border_strong="#cfd4dc", ink="#101828", ink2="#5d6676", ink3="#98a2b3",
                 success="#16a34a", warning="#d97706", danger="#dc2626", info="#0284c7")
    sh = dict(dx=0, dy=4, blur=10, color="#101828", opacity=0.07) if not dark else \
        dict(dx=0, dy=4, blur=12, color="#000000", opacity=0.28)
    sh_lg = dict(dx=0, dy=16, blur=32, color="#101828", opacity=0.16) if not dark else \
        dict(dx=0, dy=16, blur=36, color="#000000", opacity=0.55)
    card_border = True
    if style == "modern" and accent:               # an explicit brand colour tints the neutrals a little
        k = 0.07 if dark else 0.018
        for key in ("bg", "surface", "surface2", "surface3", "border"):
            c[key] = mix(c[key], acc, k * (1.6 if key in ("surface2", "surface3", "border") else 1.0))
    if style == "material":
        c["bg"] = mix(c["bg"], acc, 0.05 if not dark else 0.06)
        c["surface"] = mix(c["surface"], acc, 0.035 if not dark else 0.07)
        c["surface2"] = mix(c["surface2"], acc, 0.09 if not dark else 0.1)
        c["surface3"] = mix(c["surface3"], acc, 0.12)
        sh = dict(dx=0, dy=1, blur=3, color="#000000", opacity=0.14 if not dark else 0.4)
        card_border = False
    elif style == "glass":
        if dark:
            c.update(bg="#120e2a", surface="#ffffff1c", surface2="#ffffff14", surface3="#ffffff26",
                     border="#ffffff2e", border_strong="#ffffff4d", ink="#ffffff", ink2="#cfd2ea", ink3="#9a9fc4")
        else:
            c.update(bg="#eef0fb", surface="#ffffff8c", surface2="#ffffff66", surface3="#ffffffa6",
                     border="#ffffffb3", border_strong="#ffffffd9", ink="#141a2e", ink2="#4a5170", ink3="#7d84a3")
        sh = dict(dx=0, dy=8, blur=24, color="#0b0820", opacity=0.18 if not dark else 0.32)
    elif style == "wireframe":
        if dark:
            c.update(bg="#1b1c1f", surface="#232428", surface2="#2b2d31", surface3="#34363b", border="#45474d",
                     border_strong="#6a6d74", ink="#e6e7e9", ink2="#a2a5ab", ink3="#75787e")
            acc = "#d4d6da"
        else:
            c.update(bg="#f3f3f3", surface="#ffffff", surface2="#ececec", surface3="#e2e2e2", border="#c9c9c9",
                     border_strong="#9d9d9d", ink="#2f2f2f", ink2="#6b6b6b", ink3="#9a9a9a")
            acc = "#4a4a4a"
        c.update(success=c["ink2"], warning=c["ink2"], danger=c["ink"], info=c["ink2"])
        sh = None
        sh_lg = dict(dx=0, dy=6, blur=16, color="#000000", opacity=0.12)
    on_acc = _on(acc)
    T = Tokens(c)
    T.update(dark=dark, style=style, theme="dark" if dark else "light", accent=acc, on_accent=on_acc,
             r=dict(_RADII[style]), shadow=sh, shadow_lg=sh_lg, card_border=card_border,
             wire=style == "wireframe", glass=style == "glass", font="sans",
             scrim="#000000" + ("8c" if dark else "59"), placeholder=c["surface3"])
    T["accent_soft"] = T.soft(acc)
    T["accent_ink"] = acc if style != "glass" else (mix(acc, "#ffffff", 0.45) if dark else mix(acc, "#000000", 0.2))
    if style == "glass":
        T["bar"] = "#ffffff1f" if dark else "#ffffff99"
    else:
        T["bar"] = alpha(c["surface"] if style != "material" else c["surface2"], 0.94)
    return T


# --------------------------------------------------------------------------------------------- icons
# Same 24-unit grid and round 2-unit strokes as pixelcanvas.vector icons. Element: (kind, args, flags);
# flags: "f" = fillable body (duotone), "s" = solid.


def _P(d, f=""):
    return ("path", d, f)


def _L(x1, y1, x2, y2):
    return ("path", f"M{x1} {y1}L{x2} {y2}", "")


def _R(x, y, w, h, r=0.0, f=""):
    return ("rect", (x, y, w, h, r), f)


def _C(cx, cy, r, f=""):
    return ("circle", (cx, cy, r), f)


def _S(cx, cy, r=1.1):
    return ("circle", (cx, cy, r), "s")


def _rays(cx, cy, r0, r1, n=8, a0=0.0):
    out = []
    for i in range(n):
        a = math.radians(a0 + i * 360.0 / n)
        out.append(_L(round(cx + r0 * math.cos(a), 2), round(cy + r0 * math.sin(a), 2),
                      round(cx + r1 * math.cos(a), 2), round(cy + r1 * math.sin(a), 2)))
    return out


UI_ICONS = {
    "chevron-right": [_P("M9 5.5l6.5 6.5L9 18.5")],
    "chevron-left": [_P("M15 5.5 8.5 12l6.5 6.5")],
    "chevron-down": [_P("M5.5 9l6.5 6.5L18.5 9")],
    "chevron-up": [_P("M5.5 15 12 8.5l6.5 6.5")],
    "plus": [_L(12, 5, 12, 19), _L(5, 12, 19, 12)],
    "minus": [_L(5, 12, 19, 12)],
    "close": [_L(6.5, 6.5, 17.5, 17.5), _L(17.5, 6.5, 6.5, 17.5)],
    "tick": [_P("M5 12.5l4.5 4.5L19 7.5")],
    "menu": [_L(4, 7, 20, 7), _L(4, 12, 20, 12), _L(4, 17, 20, 17)],
    "more": [_S(5.5, 12, 1.6), _S(12, 12, 1.6), _S(18.5, 12, 1.6)],
    "more-vertical": [_S(12, 5.5, 1.6), _S(12, 12, 1.6), _S(12, 18.5, 1.6)],
    "arrow-left": [_L(19.5, 12, 5, 12), _P("M11 6l-6 6 6 6")],
    "arrow-right": [_L(4.5, 12, 19, 12), _P("M13 6l6 6-6 6")],
    "arrow-up": [_L(12, 19.5, 12, 5), _P("M6 11l6-6 6 6")],
    "arrow-down": [_L(12, 4.5, 12, 19), _P("M6 13l6 6 6-6")],
    "arrow-up-right": [_L(7, 17, 17, 7), _P("M8.5 7H17v8.5")],
    "send": [_P("M21 3 3.5 10.2l7 3.3 3.3 7Z", "f"), _L(21, 3, 10.5, 13.5)],
    "mic": [_R(9, 2.5, 6, 12, 3, "f"), _P("M5.5 11a6.5 6.5 0 0 0 13 0"), _L(12, 17.5, 12, 21.5)],
    "filter": [_P("M3.5 4.5h17l-6.5 8v6l-4 2v-8Z", "f")],
    "edit": [_P("M15.5 4.5l4 4L8.5 19.5l-5 1 1-5Z", "f"), _L(13.5, 6.5, 17.5, 10.5)],
    "trash": [_L(3.5, 6.5, 20.5, 6.5), _P("M9 6.5v-2a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"),
              _P("M5.5 6.5l1 13.2a2 2 0 0 0 2 1.8h7a2 2 0 0 0 2-1.8l1-13.2", "f"), _L(10, 10.5, 10, 17),
              _L(14, 10.5, 14, 17)],
    "eye": [_P("M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z", "f"), _C(12, 12, 3)],
    "eye-off": [_P("M10 5.7A10 10 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-2.4 3.2"),
                _P("M6.3 6.9C3.9 8.6 2.5 12 2.5 12S6 18.5 12 18.5a9.6 9.6 0 0 0 5-1.4"),
                _P("M9.9 9.9a3 3 0 0 0 4.2 4.2"), _L(3, 3, 21, 21)],
    "bookmark": [_P("M6 4.5a1.5 1.5 0 0 1 1.5-1.5h9A1.5 1.5 0 0 1 18 4.5V21l-6-4.5L6 21Z", "f")],
    "share": [_C(18, 5.5, 2.75, "f"), _C(6, 12, 2.75, "f"), _C(18, 18.5, 2.75, "f"), _L(8.4, 10.7, 15.6, 6.8),
              _L(8.4, 13.3, 15.6, 17.2)],
    "share-up": [_P("M8 9.5H6.5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2H16"),
                 _L(12, 14, 12, 2.5), _P("M8 6l4-3.5L16 6")],
    "sliders": [_L(4, 6.5, 6.6, 6.5), _L(11.4, 6.5, 20, 6.5), _C(9, 6.5, 2.4, "f"),
                _L(4, 12, 12.6, 12), _L(17.4, 12, 20, 12), _C(15, 12, 2.4, "f"),
                _L(4, 17.5, 4.6, 17.5), _L(9.4, 17.5, 20, 17.5), _C(7, 17.5, 2.4, "f")],
    "moon": [_P("M20 14.6A8.5 8.5 0 1 1 9.4 4a6.8 6.8 0 0 0 10.6 10.6Z", "f")],
    "sun": [_C(12, 12, 4, "f")] + _rays(12, 12, 7, 9.5, 8, 0),
    "credit-card": [_R(2.5, 5, 19, 14, 2.5, "f"), _L(2.5, 9.5, 21.5, 9.5), _L(6, 15, 10, 15)],
    "wallet": [_P("M3.5 8V18a2.5 2.5 0 0 0 2.5 2.5h12.5a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2H6A2.5 2.5 0 0 1 3.5 5.5 2.5 "
                  "2.5 0 0 1 6 3h10.5a1.5 1.5 0 0 1 1.5 1.5V8", "f"), _S(16.5, 14.25, 1.25)],
    "map": [_P("M3 6.2 9 3.5l6 2.7 6-2.7v14.3L15 20.5l-6-2.7-6 2.7Z", "f"), _L(9, 3.5, 9, 17.8),
            _L(15, 6.2, 15, 20.5)],
    "compass": [_C(12, 12, 9.5, "f"), _P("M15.5 8.5l-2.2 4.8-4.8 2.2 2.2-4.8Z")],
    "music": [_P("M9 18V5.5l11-2V16"), _C(6.5, 18, 2.5, "f"), _C(17.5, 16, 2.5, "f")],
    "pause": [_R(6.5, 4.5, 3.5, 15, 1.2, "s"), _R(14, 4.5, 3.5, 15, 1.2, "s")],
    "play-fill": [_P("M7.5 4.8v14.4a1 1 0 0 0 1.5.86l11.6-7.2a1 1 0 0 0 0-1.72L9 3.94a1 1 0 0 0-1.5.86Z", "s")],
    "skip-forward": [_P("M5 5.5v13l10-6.5Z", "s"), _R(16.5, 5, 2.5, 14, 1, "s")],
    "skip-back": [_P("M19 5.5v13l-10-6.5Z", "s"), _R(5, 5, 2.5, 14, 1, "s")],
    "shuffle": [_P("M3 7h3.5c4.5 0 6.5 10 11 10H21"), _P("M3 17h3.5c1.6 0 2.8-1.2 3.8-2.8"),
                _P("M13.8 9.6C14.8 8.1 16 7 17.5 7H21"), _P("M18.5 4.5 21 7l-2.5 2.5"),
                _P("M18.5 14.5 21 17l-2.5 2.5")],
    "repeat": [_P("M4 11V9.5A2.5 2.5 0 0 1 6.5 7H20"), _P("M17 4l3 3-3 3"), _P("M20 13v1.5a2.5 2.5 0 0 1-2.5 2.5H4"),
               _P("M7 20l-3-3 3-3")],
    "volume": [_P("M3.5 9.2h3.6L12 5v14l-4.9-4.2H3.5Z", "f"), _P("M15.5 8.8a4.6 4.6 0 0 1 0 6.4"),
               _P("M18.4 5.9a8.6 8.6 0 0 1 0 12.2")],
    "battery": [_R(2.5, 7, 17, 10, 2.5), _R(4.75, 9.25, 10, 5.5, 1, "s"), _P("M21.5 10.5v3")],
    "signal": [_R(3, 15, 3, 5, 1, "s"), _R(8, 11.5, 3, 8.5, 1, "s"), _R(13, 8, 3, 12, 1, "s"),
               _R(18, 4, 3, 16, 1, "s")],
    "bluetooth": [_P("M7 7.5l10 9-5 4.5V3l5 4.5-10 9")],
    "message": [_P("M4 5.5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-7.5L6 20.5v-4H6a2 2 0 0 1-2-2Z", "f")],
    "grid": [_R(3.5, 3.5, 7, 7, 1.5, "f"), _R(13.5, 3.5, 7, 7, 1.5, "f"), _R(3.5, 13.5, 7, 7, 1.5, "f"),
             _R(13.5, 13.5, 7, 7, 1.5, "f")],
    "list": [_L(9, 6, 20.5, 6), _L(9, 12, 20.5, 12), _L(9, 18, 20.5, 18), _S(4.5, 6, 1.3), _S(4.5, 12, 1.3),
             _S(4.5, 18, 1.3)],
    "logout": [_P("M9.5 20.5H6a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2h3.5"), _L(10, 12, 20.5, 12), _P("M16.5 7.5 21 12l-4.5 4.5")],
    "copy": [_R(8.5, 8.5, 12, 12, 2, "f"), _P("M15.5 5.5v-1a1.5 1.5 0 0 0-1.5-1.5H5A1.5 1.5 0 0 0 3.5 4.5V14A1.5 "
                                             "1.5 0 0 0 5 15.5h1")],
    "external": [_P("M13.5 4h6.5v6.5"), _L(20, 4, 11, 13), _P("M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 "
                                                            "18.5v-11A1.5 1.5 0 0 1 5.5 6H10")],
    "help": [_C(12, 12, 9.5, "f"), _P("M9.4 9.3a2.7 2.7 0 0 1 5.2.9c0 1.8-2.6 2.4-2.6 4"), _S(12, 17.2, 1.15)],
    "smile": [_C(12, 12, 9.5, "f"), _P("M8.3 14.2a4.6 4.6 0 0 0 7.4 0"), _S(9, 9.8, 1.15), _S(15, 9.8, 1.15)],
    "paperclip": [_P("M20 11.5 12 19.5a5 5 0 0 1-7.1-7.1l8.4-8.4a3.3 3.3 0 0 1 4.7 4.7L9.6 17.1a1.7 1.7 0 0 1-2.4-2.4"
                     "l7.4-7.4")],
    "video": [_R(2.5, 6, 13, 12, 2.5, "f"), _P("M15.5 10.5 21.5 7v10l-6-3.5")],
    "rain": [_P("M7 15.5h10a4.5 4.5 0 0 0 .5-8.97A5.8 5.8 0 0 0 6.3 7.7 3.9 3.9 0 0 0 7 15.5Z", "f"),
             _L(8, 18, 7, 21), _L(12, 18, 11, 21), _L(16, 18, 15, 21)],
    "cloud-sun": [_P("M5.6 10.7A4 4 0 1 1 12.4 6.6"), _L(8.5, 1.8, 8.5, 2.6), _L(2.3, 8, 3.1, 8),
                  _L(4.1, 3.6, 4.7, 4.2), _L(12.9, 3.6, 12.3, 4.2),
                  _P("M9 20.5h8.5a4.2 4.2 0 0 0 .5-8.37 5.4 5.4 0 0 0-10.3.9A3.8 3.8 0 0 0 9 20.5Z", "f")],
    "wind": [_P("M3 8.5h10.5a2.75 2.75 0 1 0-2.6-3.6"), _P("M3 12.5h15.5a2.75 2.75 0 1 1-2.6 3.6"),
             _L(3, 16.5, 10, 16.5)],
    "drop": [_P("M12 3s6.5 7 6.5 11.5a6.5 6.5 0 0 1-13 0C5.5 10 12 3 12 3Z", "f")],
    "thermometer": [_P("M14 14.3V5a2 2 0 0 0-4 0v9.3a4 4 0 1 0 4 0Z", "f"), _S(12, 17.5, 1.6)],
    "flag": [_P("M5 21V4"), _P("M5 4.5h12.5l-2.5 4 2.5 4H5", "f")],
    "gift": [_R(3.5, 8, 17, 4.5, 1.2, "f"), _P("M5 12.5v7a1.5 1.5 0 0 0 1.5 1.5h11a1.5 1.5 0 0 0 1.5-1.5v-7"),
             _L(12, 8, 12, 21), _P("M12 8S10.5 3.5 8 3.5a2.25 2.25 0 0 0 0 4.5Z"),
             _P("M12 8s1.5-4.5 4-4.5a2.25 2.25 0 0 1 0 4.5Z")],
    "briefcase": [_R(3, 7, 18, 13, 2.5, "f"), _P("M8.5 7V5.5a2 2 0 0 1 2-2h3a2 2 0 0 1 2 2V7"), _L(3, 12.5, 21, 12.5)],
    "plane": [_P("M10.5 13.5 3 11l1.5-1.5 7.5.5 4.5-4.5a2.1 2.1 0 0 1 3 3L15 13l.5 7.5L14 22l-2.5-7.5-3.5 3.5v2.5L6.5 "
                 "21.5 5 18.5 2.5 17.5 4 16l2.5.2Z", "f")],
    "inbox": [_P("M3 13.5 5.6 5.4A2 2 0 0 1 7.5 4h9a2 2 0 0 1 1.9 1.4L21 13.5V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z", "f"),
              _P("M3 13.5h5l1.5 2.5h5l1.5-2.5h5")],
    "dollar": [_L(12, 2.5, 12, 21.5), _P("M16.5 6.5H10a3 3 0 0 0 0 6h4a3 3 0 0 1 0 6H6.5")],
    "percent": [_L(19, 5, 5, 19), _C(7, 7, 2.5), _C(17, 17, 2.5)],
    "activity": [_P("M2.5 12h4l3-8 5 16 3-8h4")],
    "target": [_C(12, 12, 9.5, "f"), _C(12, 12, 5.5), _S(12, 12, 1.6)],
    "fire": [_P("M12 21.5a7 7 0 0 0 7-7c0-4-3-6.5-4-9.5-1 2-2 3-3.5 3.5C11 6 10 4 9 2.5 8 7 5 9.5 5 14.5a7 7 0 0 0 7 7Z",
                "f")],
    "headphones": [_P("M3.5 17v-4.5a8.5 8.5 0 0 1 17 0V17"), _R(3.5, 13.5, 4.5, 7, 2, "f"), _R(16, 13.5, 4.5, 7, 2, "f")],
    "dot": [_S(12, 12, 4)],
    "location": [_P("M20.5 3.5 3.5 10.8l7 2.7 2.7 7Z", "f")],
    "at": [_C(12, 12, 4), _P("M16 8v5.5a2.5 2.5 0 0 0 5 0V12a9 9 0 1 0-3.5 7.1")],
    "phone-call": [_P("M5 3.5h3.5l1.7 4.3-2.2 1.4a11 11 0 0 0 6.8 6.8l1.4-2.2 4.3 1.7V19a1.5 1.5 0 0 1-1.5 1.5A16 16 "
                      "0 0 1 3.5 5 1.5 1.5 0 0 1 5 3.5Z", "f")],
}
for _k, _spec_ in UI_ICONS.items():
    ICONS.setdefault(_k, _spec_)

_ICON_ALIAS = {"back": "chevron-left", "x": "close", "dots": "more", "kebab": "more-vertical", "hamburger": "menu",
               "settings": "gear", "notifications": "bell", "profile": "user", "person": "user", "chat": "message",
               "comment": "message", "delete": "trash", "pencil": "edit", "card": "credit-card", "note": "music",
               "music-note": "music", "speaker": "volume", "next": "skip-forward", "previous": "skip-back",
               "prev": "skip-back", "add": "plus", "remove": "minus", "done": "tick", "ok": "tick",
               "calendar-days": "calendar", "stats": "chart", "analytics": "chart", "money": "dollar",
               "favorite": "heart", "like": "heart", "weather": "cloud-sun", "location-arrow": "location"}


def _icon_name(name):
    if name is None:
        return None
    n = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    n = _ICON_ALIAS.get(n, n)
    return n if n in ICONS else None


def icon_names():
    """All icon names usable in UI calls (vector built-ins + UI_ICONS)."""
    return sorted(ICONS)


# --------------------------------------------------------------------------------------------- text helpers

_CJK_CH = "⺀-鿿가-힯぀-ヿ＀-￯"
_TOK = re.compile(rf"[{_CJK_CH}][\s]*|[^\s{_CJK_CH}]+\s*|\s+")


@lru_cache(maxsize=16384)
def _tw(text, size, bold=False, family="sans", ls=0.0):
    """Width of one line of text in units."""
    if not text:
        return 0.0
    return float(measure_text(text, float(size), family, "bold" if bold else "normal", letter_spacing=ls)[0])


def _ellipsize(s, size, width, bold=False, family="sans", force=False, ls=0.0):
    s = s.rstrip()
    width = width + 1.0
    if not force and _tw(s, size, bold, family, ls) <= width:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _tw(s[:mid].rstrip() + "…", size, bold, family, ls) <= width:
            lo = mid
        else:
            hi = mid - 1
    return (s[:lo].rstrip() + "…") if lo > 0 else ("…" if width >= _tw("…", size, bold, family, ls)
                                                          else "")


@lru_cache(maxsize=4096)
def wrap(text, size, width, bold=False, max_lines=None, family="sans", ls=0.0):
    """Lines of ``text`` wrapped to ``width`` units; the last allowed line ends with an ellipsis if cut."""
    width = max(float(width), 1.0)
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for tok in _TOK.findall(para):
            cand = cur + tok
            if not cur.strip() or _tw(cand.rstrip(), size, bold, family, ls) <= width:
                cur = cand
                continue
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        lines.append(cur.rstrip())
    out = []
    for ln in lines:                        # break words longer than the box
        while ln and _tw(ln, size, bold, family, ls) > width and len(ln) > 1:
            lo, hi = 1, len(ln)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _tw(ln[:mid], size, bold, family, ls) <= width:
                    lo = mid
                else:
                    hi = mid - 1
            out.append(ln[:lo])
            ln = ln[lo:].lstrip()
        out.append(ln)
    if max_lines and len(out) > max_lines:
        out = out[:max_lines]
        out[-1] = _ellipsize(out[-1], size, width, bold, family, force=True, ls=ls)
    return tuple(out)


def _lh(size, ratio):
    return float(round(size * ratio / 2.0) * 2)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pad4(p):
    if p is None:
        return (0.0, 0.0, 0.0, 0.0)
    if _num(p):
        return (float(p),) * 4
    p = [float(v) for v in p]
    if len(p) == 2:
        return (p[0], p[1], p[0], p[1])
    if len(p) == 3:
        return (p[0], p[1], p[2], p[1])
    return tuple(p[:4])


# --------------------------------------------------------------------------------------------- nodes & layout


_NODE_OPTS = None


def _node_options():
    """Option names each component kind reads (found in this module's source once, on first use), plus names
    every kind shares and the keywords the Screen methods pass explicitly."""
    global _NODE_OPTS
    if _NODE_OPTS is None:
        import ast
        per, common = {}, {"w", "h", "name", "max_w", "span", "grow", "pad", "bg", "radius", "self_align"}
        try:
            with open(__file__, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError):
            tree = None

        def const(n):
            return n.value if isinstance(n, ast.Constant) and isinstance(n.value, str) else None

        def is_p(b):
            return (isinstance(b, ast.Name) and b.id in ("p", "kw", "kw2")) or (isinstance(b, ast.Attribute)
                                                                               and b.attr == "p")

        def read_keys(node):
            out = set()
            for n in ast.walk(node):
                k = b = None
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args and                         n.func.attr in ("get", "pop", "setdefault"):
                    k, b = const(n.args[0]), n.func.value
                elif isinstance(n, ast.Subscript):
                    k, b = const(n.slice), n.value
                elif isinstance(n, ast.Compare) and isinstance(n.ops[0], (ast.In, ast.NotIn)):
                    k, b = const(n.left), n.comparators[0]
                if k is not None and is_p(b):
                    out.add(k)
                if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in ("Node", "_add"):
                    a = n.args[1:2] if getattr(n.func, "id", "") == "Node" else n.args[:1]
                    kind = const(a[0]) if a else None
                    if kind:
                        per.setdefault(kind, set()).update(kw.arg for kw in n.keywords if kw.arg)
            return out
        if tree is not None:
            for n in tree.body:
                kind = None
                if isinstance(n, ast.ClassDef):
                    for d in n.decorator_list:
                        if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "_spec":
                            kind = const(d.args[0])
                keys = read_keys(n)
                if kind:
                    per.setdefault(kind, set()).update(keys)
                else:
                    common |= keys
        _NODE_OPTS = (per, common) if tree is not None else False
    return _NODE_OPTS


def _check_node_options(ui, ctype, p):
    """Reject component options the component does not read (TypeError naming the closest valid one)."""
    opts = _node_options()
    if not opts or ctype not in opts[0]:
        return p
    import sys
    from ._args import reject_unknown, signature_options
    per, common = opts
    owner, first = "ui." + ctype, []
    f = sys._getframe(2)
    while f is not None:                      # outermost Screen method on the stack names the call
        if f.f_code.co_filename == __file__ and isinstance(f.f_locals.get("self"), Screen):
            owner = f.f_code.co_name
            first = signature_options(getattr(Screen, owner, None) or (lambda: 0))
        f = f.f_back
    allowed = list(dict.fromkeys(first + sorted(per[ctype]) + sorted(common)))
    p = reject_unknown(owner, p, allowed)
    if ctype in ("icon", "icon_badge", "icon_button", "button", "fab"):
        for k in ("icon", "icon_right"):
            v = p.get(k)
            if isinstance(v, str) and v and _icon_name(v) is None:
                from ._args import unknown_name
                raise unknown_name("icon", v, ICONS, "Try ui.icon_names()")
    return p


class Node:
    """One element of a screen (a component or a container). Containers are context managers:
    ``with s.vstack(gap=12): ...``. ``node.frame`` = (x, y, w, h) in screen units after layout."""

    def __init__(self, ui, ctype, w=None, h=None, name=None, max_w=None, span=1, grow=1.0, **p):
        if p:
            p = _check_node_options(ui, ctype, p)
        self.ui, self.kind, self.p = ui, ctype, p
        spec = _SPEC[ctype]
        self.w = w if w is not None else spec.w
        self.h = h if h is not None else spec.h
        self.name, self.max_w, self.span, self.grow = name, max_w, max(1, int(span)), float(grow)
        self.children, self.frame, self.parent = [], None, None
        self.lpad = None
        if not spec.container and "pad" in p:
            self.lpad = _pad4(p.pop("pad"))

    def __enter__(self):
        self.ui._stack.append(self)
        return self

    def __exit__(self, *exc):
        if self.ui._stack and self.ui._stack[-1] is self:
            self.ui._stack.pop()
        return False

    def key(self):
        """Lookup key: the explicit name or 'kind:text'."""
        if self.name:
            return str(self.name)
        t = self.p.get("text") or self.p.get("title") or self.p.get("label")
        return f"{self.kind}:{t}" if isinstance(t, str) and t else self.kind

    def box(self):
        """Pixel bbox (x0, y0, x1, y1) of this node on the canvas it was last drawn on."""
        return self.ui.box(self)

    def __repr__(self):
        return f"<Node {self.key()} frame={self.frame}>"


class _Spec:
    w = "auto"
    h = "auto"
    container = False

    def size(self, ui, n, aw):
        return (aw, 40.0)

    def paint(self, ui, g, n, x, y, w, h):
        pass


_SPEC = {}


def _spec(kind, w="auto", h="auto"):
    def deco(cls):
        inst = cls()
        inst.w, inst.h, inst.kind = w, h, kind
        _SPEC[kind] = inst
        return cls
    return deco


def _kids(n):
    return list(n.children) + list(n.p.get("tail") or [])


def _measure(ui, n, aw):
    """Natural (w, h) of node ``n`` when given at most ``aw`` units of width."""
    aw = max(float(aw), 0.0)
    if n.max_w:
        aw = min(aw, float(n.max_w))
    if _num(n.w):
        aw = min(aw, float(n.w))
    key = (id(n), round(aw, 2))
    hit = ui._mcache.get(key)
    if hit is not None:
        return hit
    if n.lpad:
        t, r, b, l = n.lpad
        w, h = _SPEC[n.kind].size(ui, n, max(aw - l - r, 0.0))
        w, h = w + l + r, h + t + b
    else:
        w, h = _SPEC[n.kind].size(ui, n, aw)
    if _num(n.w):
        w = aw
    if _num(n.h):
        h = float(n.h)
    res = (min(float(w), aw), max(float(h), 0.0))
    ui._mcache[key] = res
    return res


def _layout(ui, n, x, y, w, h):
    n.frame = (float(x), float(y), max(float(w), 0.0), max(float(h), 0.0))
    ui._register(n)
    sp = _SPEC[n.kind]
    if sp.container:
        sp.arrange(ui, n, *n.frame)


def _paint(ui, g, n):
    if n.frame is None:
        return
    x, y, w, h = n.frame
    if w < 0.5 or h < 0.5:
        return
    grp = g.group(n.key())
    if n.p.get("bg") and not _SPEC[n.kind].container:
        grp.rect(x, y, w, h, r=n.p.get("radius", min(ui.T.r["small"] + 4, h / 2)), fill=n.p["bg"])
    if n.lpad:
        t, r, b, l = n.lpad
        x, y, w, h = x + l, y + t, max(w - l - r, 0.0), max(h - t - b, 0.0)
    _SPEC[n.kind].paint(ui, grp, n, x, y, w, h)


def _fill_w(c):
    return c.w == "fill" or (c.kind == "spacer" and not _num(c.w))


def _fill_h(c):
    return c.h == "fill" or (c.kind == "spacer" and not _num(c.h))


@_spec("spacer")
class _Spacer(_Spec):
    def size(self, ui, n, aw):
        return (float(n.w) if _num(n.w) else 0.0, float(n.h) if _num(n.h) else 0.0)

    def paint(self, ui, g, n, x, y, w, h):
        d = n.p.get("dot")
        if d:
            g.rect(x, y, w, h, r=min(w, h) / 2, fill=ui.T.accent if d == "on" else ui.T.surface3)


@_spec("stack")
class _Stack(_Spec):
    container = True

    def _vw(self, ui, n, c, iw):
        if _num(c.w):
            return min(float(c.w), iw)
        if c.w == "fill" or (c.w == "auto" and n.p.get("align", "stretch") == "stretch"):
            return min(iw, float(c.max_w)) if c.max_w else iw
        return _measure(ui, c, iw)[0]

    def _hw(self, ui, n, kids, iw):
        gap = n.p.get("gap", 0.0)
        ws, flex = [], []
        for i, c in enumerate(kids):
            if _num(c.w):
                ws.append(min(float(c.w), iw))
            elif _fill_w(c):
                ws.append(0.0)
                flex.append(i)
            else:
                ws.append(_measure(ui, c, iw)[0])
        rem = iw - sum(ws) - gap * max(len(kids) - 1, 0)
        if rem >= 0:
            tot = sum(kids[i].grow for i in flex) or 1.0
            for i in flex:
                ws[i] = rem * kids[i].grow / tot
                if kids[i].max_w:
                    ws[i] = min(ws[i], float(kids[i].max_w))
        else:                                 # shrink the hug children that can give way (text, stacks)
            over = -rem
            shr = [i for i, c in enumerate(kids) if not _num(c.w) and i not in flex and ws[i] > 28
                   and c.kind in _SHRINK]
            tot = sum(ws[i] for i in shr)
            if tot > 0:
                for i in shr:
                    ws[i] = max(ws[i] - over * ws[i] / tot, 24.0)
        return ws

    def _flow(self, ui, n, kids, iw):
        gap, rg = n.p.get("gap", 0.0), n.p.get("row_gap", n.p.get("gap", 0.0))
        rows, cur, cw = [], [], 0.0
        for c in kids:
            w, h = _measure(ui, c, iw)
            if cur and cw + gap + w > iw + 0.01:
                rows.append(cur)
                cur, cw = [], 0.0
            cur.append((c, w, h))
            cw += (gap if len(cur) > 1 else 0) + w
        if cur:
            rows.append(cur)
        return rows, gap, rg

    def size(self, ui, n, aw):
        t, r, b, l = n.p["pad"]
        iw = max(aw - l - r, 0.0)
        kids = _kids(n)
        gap = n.p.get("gap", 0.0)
        if not kids:
            return (min(aw, l + r), t + b)
        if n.p.get("axis") == "v":
            W = H = 0.0
            for c in kids:
                cw = self._vw(ui, n, c, iw)
                w_, h_ = _measure(ui, c, cw)
                if _fill_h(c) and not _num(c.h):
                    h_ = 0.0 if c.kind == "spacer" else h_
                W = max(W, w_)
                H += h_
            H += gap * (len(kids) - 1)
            return (min(aw, W + l + r), H + t + b)
        if n.p.get("wrap"):
            rows, g_, rg = self._flow(ui, n, kids, iw)
            W = max(sum(w for _, w, _ in row) + g_ * (len(row) - 1) for row in rows)
            H = sum(max(h for _, _, h in row) for row in rows) + rg * (len(rows) - 1)
            return (min(aw, W + l + r), H + t + b)
        ws = self._hw(ui, n, kids, iw)
        hs = [_measure(ui, c, w)[1] for c, w in zip(kids, ws) if c.kind != "spacer" or _num(c.h)]
        H = max(hs) if hs else 0.0
        W = iw if any(_fill_w(c) for c in kids) else sum(ws) + gap * (len(kids) - 1)
        return (min(aw, W + l + r), H + t + b)

    def arrange(self, ui, n, x, y, w, h):
        t, r, b, l = n.p["pad"]
        ix, iy, iw, ih = x + l, y + t, max(w - l - r, 0.0), max(h - t - b, 0.0)
        kids = _kids(n)
        if not kids:
            return
        gap = n.p.get("gap", 0.0)
        align = n.p.get("align", "stretch" if n.p.get("axis") == "v" else "center")
        justify = n.p.get("justify", "start")
        scroll = bool(n.p.get("scroll"))
        if n.p.get("axis") == "v":
            sizes = []
            for c in kids:
                cw = self._vw(ui, n, c, iw)
                if _num(c.h):
                    ch = float(c.h)
                elif _fill_h(c):
                    ch = 0.0 if c.kind == "spacer" else _measure(ui, c, cw)[1]
                else:
                    ch = _measure(ui, c, cw)[1]
                sizes.append([cw, ch])
            total = sum(s[1] for s in sizes) + gap * (len(kids) - 1)
            extra = ih - total
            fills = [i for i, c in enumerate(kids) if _fill_h(c) and not _num(c.h)]
            g2, off = gap, 0.0
            if fills and not scroll:
                tot = sum(kids[i].grow for i in fills) or 1.0
                base = sum(s[1] for i, s in enumerate(sizes) if i not in fills) + gap * (len(kids) - 1)
                avail = max(ih - base, 0.0)
                for i in fills:
                    sizes[i][1] = avail * kids[i].grow / tot
            elif extra > 0 and not scroll:
                if justify == "center":
                    off = extra / 2
                elif justify == "end":
                    off = extra
                elif justify == "between" and len(kids) > 1:
                    g2 = gap + extra / (len(kids) - 1)
            cy = iy + off
            for c, (cw, ch) in zip(kids, sizes):
                if c.max_w and cw > c.max_w:
                    cw = float(c.max_w)
                a = c.p.get("self_align") or align
                if a == "center":
                    cx = ix + (iw - cw) / 2
                elif a == "end":
                    cx = ix + iw - cw
                else:
                    cx = ix
                _layout(ui, c, cx, cy, cw, ch)
                cy += ch + g2
            return
        if n.p.get("wrap"):
            rows, g_, rg = self._flow(ui, n, kids, iw)
            cy = iy
            for row in rows:
                rh = max(hh for _, _, hh in row)
                rw = sum(ww for _, ww, _ in row) + g_ * (len(row) - 1)
                cx = ix + ((iw - rw) / 2 if justify == "center" else (iw - rw if justify == "end" else 0.0))
                for c, ww, hh in row:
                    _layout(ui, c, cx, cy + (rh - hh) / 2, ww, hh)
                    cx += ww + g_
                cy += rh + rg
            return
        ws = self._hw(ui, n, kids, iw)
        used = sum(ws) + gap * (len(kids) - 1)
        extra = iw - used
        g2, off = gap, 0.0
        if extra > 0.01 and not any(_fill_w(c) for c in kids):
            if justify == "center":
                off = extra / 2
            elif justify == "end":
                off = extra
            elif justify == "between" and len(kids) > 1:
                g2 = gap + extra / (len(kids) - 1)
        cx = ix + off
        for c, cw in zip(kids, ws):
            if _num(c.h):
                ch = min(float(c.h), max(ih, float(c.h)))
            elif c.h == "fill" or (align == "stretch" and c.kind != "spacer"):
                ch = ih
            else:
                ch = _measure(ui, c, cw)[1]
            a = c.p.get("self_align") or align
            if a == "start":
                cy = iy
            elif a == "end":
                cy = iy + ih - ch
            else:
                cy = iy + (ih - ch) / 2
            _layout(ui, c, cx, cy, cw, ch)
            cx += cw + g2

    def paint(self, ui, g, n, x, y, w, h):
        p = n.p
        surface, bg = p.get("surface"), p.get("bg")
        rad = p.get("radius")
        if surface or bg or p.get("border"):
            ui.panel(g, x, y, w, h, r=rad, kind=surface or "plain", fill=bg, border=p.get("border"),
                     shadow=p.get("shadow"))
        target = g
        if p.get("clip") or p.get("scroll"):
            rr = ui.T.r["card"] if (rad is None and surface) else (rad or 0)
            target = g.group(f"{n.key()} content", clip=g.rect(x, y, w, h, r=rr, add=False))
        kids = _kids(n)
        for c in kids:
            _paint(ui, target, c)
        if p.get("dividers") and len(kids) > 1:
            inset = p.get("divider_inset", 0.0)
            t_, r_, b_, l_ = p["pad"]
            for a, b2 in zip(kids[:-1], kids[1:]):
                if a.frame is None or b2.frame is None:
                    continue
                yy = (a.frame[1] + a.frame[3] + b2.frame[1]) / 2
                ins = inset(a) if callable(inset) else inset
                ui.hline(target, x + l_ + ins, x + w - (r_ if p.get("divider_full") is None else 0), yy)


@_spec("grid")
class _Grid(_Spec):
    container = True

    def _rows(self, ui, n, iw):
        cols = max(int(n.p.get("cols", 2)), 1)
        gap = n.p.get("gap", 0.0)
        colw = (iw - gap * (cols - 1)) / cols
        rows, cur, used = [], [], 0
        for c in _kids(n):
            sp = min(c.span, cols)
            if used + sp > cols:
                rows.append(cur)
                cur, used = [], 0
            cur.append((c, used, sp))
            used += sp
        if cur:
            rows.append(cur)
        out = []
        for row in rows:
            items = []
            for c, col, sp in row:
                cw = colw * sp + gap * (sp - 1)
                items.append((c, col, cw, _measure(ui, c, cw)[1]))
            out.append(items)
        return out, colw, gap

    def size(self, ui, n, aw):
        t, r, b, l = n.p["pad"]
        rows, _, gap = self._rows(ui, n, max(aw - l - r, 0.0))
        rg = n.p.get("row_gap", gap)
        H = sum(max(it[3] for it in row) for row in rows) + rg * max(len(rows) - 1, 0)
        return (aw, H + t + b)

    def arrange(self, ui, n, x, y, w, h):
        t, r, b, l = n.p["pad"]
        rows, colw, gap = self._rows(ui, n, max(w - l - r, 0.0))
        rg = n.p.get("row_gap", gap)
        cy = y + t
        for row in rows:
            rh = max(it[3] for it in row)
            for c, col, cw, ch in row:
                _layout(ui, c, x + l + col * (colw + gap), cy, cw, rh if n.p.get("equal", True) else ch)
            cy += rh + rg

    paint = _Stack.paint


_SHRINK = {"text", "stack", "grid", "badge", "chip", "button", "breadcrumb", "field", "logo"}

# --------------------------------------------------------------------------------------------- paint helpers


def _txt(ui, g, x, y, text, size=15, color=None, bold=False, anchor="lc", ls=0.0, maxw=None, opacity=1.0,
         family=None):
    """One line of text; ``anchor`` like vector ('lc' = left, capitals centred on y); ``maxw`` truncates with …."""
    if text is None or text == "":
        return 0.0
    text = str(text)
    fam = family or ui.T.font
    if maxw is not None:
        text = _ellipsize(text, size, maxw, bold, fam, ls=ls)
        if not text:
            return 0.0
    if text[0] in "(\"'“‘[{" and anchor[0] == "l":     # keep the first glyph from being cropped
        x -= _tw(" x", size, bold, fam, ls) - _tw("x", size, bold, fam, ls)
        text = " " + text
    g.text(x, y, text, size=size, fill=color or ui.T.ink, anchor=anchor, weight="bold" if bold else "normal",
           family=fam, letter_spacing=ls, opacity=opacity)
    return _tw(text.lstrip(" "), size, bold, fam, ls)


def _bar(ui, g, x, cy, w, size, color=None):
    """Wireframe placeholder line for a line of text."""
    hh = max(round(size * 0.5), 4)
    g.rect(x, cy - hh / 2, max(w, hh), hh, r=hh / 2, fill=color or ui.T.placeholder)


def _ico(ui, g, name, cx, cy, size=20, color=None, sw=None, fill=None):
    nm = _icon_name(name)
    col = color or ui.T.ink
    if nm is None:
        g.circle(cx, cy, size * 0.32, fill=None, stroke=col, stroke_width=max(size / 12, 1.2))
        return
    g.icon(nm, cx, cy, size, color=col, stroke_width=sw or (1.75 if size >= 18 else 2.0), fill=fill)


_PALETTE = ["#6366f1", "#f97316", "#10b981", "#ec4899", "#0ea5e9", "#eab308", "#8b5cf6", "#ef4444", "#14b8a6",
            "#f43f5e"]
_SKIN = ["#f2c9a5", "#e0ac84", "#c68a62", "#a86b48", "#8a5236", "#f6d7bd", "#5e3a26"]
_HAIR = ["#2b2118", "#4a3020", "#7a4a26", "#c08a4a", "#1f1f24", "#a8432a", "#d9c29a", "#3d2b5a"]
_SHIRT = ["#6366f1", "#0ea5e9", "#10b981", "#f97316", "#ec4899", "#334155", "#8b5cf6", "#eab308", "#e11d48"]


def _initials(name):
    parts = [p for p in re.split(r"\s+", str(name).strip()) if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper() if len(parts[0]) > 1 and not _v._text.has_cjk(parts[0]) else parts[0][:1].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _avatar(ui, g, cx, cy, d, name="Alex Morgan", kind="portrait", status=None, ring=None, color=None):
    """Avatar disc: 'initials' (tinted disc + letters) or 'portrait' (abstract person)."""
    T = ui.T
    r = d / 2
    hsh = _hash(name)
    if ring:
        g.circle(cx, cy, r + max(d * 0.06, 1.5), fill=ring)
    if T.wire:
        g.circle(cx, cy, r, fill=T.surface2, stroke=T.border_strong, stroke_width=ui.hl)
        g.circle(cx, cy - r * 0.18, r * 0.32, fill=None, stroke=T.ink3, stroke_width=max(ui.hl, d / 30))
        g.path(f"M{cx - r * 0.6} {cy + r * 0.62}a{r * 0.6} {r * 0.5} 0 0 1 {r * 1.2} 0", fill=None, stroke=T.ink3,
               stroke_width=max(ui.hl, d / 30))
    elif kind == "initials":
        c = color or _PALETTE[hsh % len(_PALETTE)]
        bgc = mix(T.surface if not T.glass else ("#1d1840" if T.dark else "#ffffff"), c, 0.22 if T.dark else 0.16)
        g.circle(cx, cy, r, fill=bgc)
        _txt(ui, g, cx, cy, _initials(name), size=d * 0.38, color=mix(c, T.ink, 0.15 if T.dark else 0.3), bold=True,
             anchor="mc")
    else:
        rng = np.random.default_rng(hsh)
        bgc = _PALETTE[hsh % len(_PALETTE)]
        bg_hi, bg_lo = mix(bgc, "#ffffff", 0.72), mix(bgc, "#ffffff", 0.45)
        skin = _SKIN[(hsh >> 3) % len(_SKIN)]
        hair = _HAIR[(hsh >> 6) % len(_HAIR)]
        shirt = _SHIRT[(hsh >> 9) % len(_SHIRT)]
        style = (hsh >> 12) % 4
        with g.group(None, clip=g.circle(cx, cy, r, add=False)) as c:
            c.circle(cx, cy, r, fill=linear_gradient([bg_hi, bg_lo]))
            hy = cy - d * 0.06
            if style in (1, 3):                                     # long hair behind the head
                c.ellipse(cx, hy + d * 0.12, d * 0.25, d * 0.33, fill=hair)
            c.ellipse(cx, cy + d * 0.6, d * 0.38, d * 0.3, fill=shirt)
            c.rect(cx - d * 0.075, hy + d * 0.12, d * 0.15, d * 0.2, r=d * 0.05, fill=mix(skin, "#000000", 0.12))
            c.ellipse(cx, hy, d * 0.175, d * 0.215, fill=skin)
            if style == 0:                                          # short crop
                c.path(f"M{cx - d * 0.18} {hy - d * 0.02}C{cx - d * 0.2} {hy - d * 0.26} {cx + d * 0.2} "
                       f"{hy - d * 0.28} {cx + d * 0.18} {hy - d * 0.02}C{cx + d * 0.12} {hy - d * 0.12} "
                       f"{cx - d * 0.08} {hy - d * 0.16} {cx - d * 0.18} {hy - d * 0.02}Z", fill=hair)
            elif style == 2:                                        # bun
                c.circle(cx + d * 0.02, hy - d * 0.25, d * 0.09, fill=hair)
                c.path(f"M{cx - d * 0.185} {hy}C{cx - d * 0.19} {hy - d * 0.27} {cx + d * 0.19} {hy - d * 0.27} "
                       f"{cx + d * 0.185} {hy}C{cx + d * 0.1} {hy - d * 0.1} {cx - d * 0.1} {hy - d * 0.1} "
                       f"{cx - d * 0.185} {hy}Z", fill=hair)
            else:                                                   # fringe over long hair
                c.path(f"M{cx - d * 0.19} {hy + d * 0.04}C{cx - d * 0.22} {hy - d * 0.3} {cx + d * 0.22} "
                       f"{hy - d * 0.3} {cx + d * 0.19} {hy + d * 0.04}C{cx + d * 0.14} {hy - d * 0.1} "
                       f"{cx - d * 0.02} {hy - d * 0.12} {cx - d * 0.19} {hy + d * 0.04}Z", fill=hair)
            _ = rng
    if status:
        col = {"online": T.success, "busy": T.danger, "away": T.warning}.get(status, status)
        sr = max(d * 0.13, 3.5)
        a = math.radians(45)
        sx, sy = cx + r * math.cos(a), cy + r * math.sin(a)
        g.circle(sx, sy, sr + max(d * 0.04, 1.5), fill=T.surface if not T.glass else T.bg)
        g.circle(sx, sy, sr, fill=col)


def _btn_colors(ui, variant):
    T = ui.T
    if variant == "primary":
        return T.accent, T.on_accent, None
    if variant in ("destructive", "danger"):
        return T.danger, "#ffffff" if not T.wire else T.bg, None
    if variant in ("secondary", "tonal"):
        if T.style == "material" or variant == "tonal":
            return T.accent_soft, T.accent_ink, None
        if T.glass:
            return ("#ffffff24" if T.dark else "#ffffffb3"), T.ink, T.border
        return (T.surface3 if T.dark else T.surface2), T.ink, (None if T.dark else None)
    if variant == "outline":
        return None, T.ink, T.border_strong
    if variant == "inverse":
        return T.ink, T.bg if not T.glass else "#141a2e", None
    if variant == "white":
        return "#ffffff", "#101828", None
    return None, T.accent_ink, None                      # ghost / link


_BTN = {"sm": (32, 13, 12), "md": (44, 15, 18), "lg": (52, 16, 24), "xl": (56, 17, 28)}


def _round_r(ui, kind, h):
    return min(ui.T.r[kind], h / 2)


# --------------------------------------------------------------------------------------------- leaf components


@_spec("text")
class _Text(_Spec):
    def font(self, ui, n):
        role = n.p.get("role", "body")
        size, wt, ratio = TYPE.get(role, TYPE["body"])
        size = float(n.p.get("size") or size)
        bold = n.p.get("bold")
        bold = (wt == "bold") if bold is None else bool(bold)
        lh = _lh(size, n.p.get("lh") or ratio)
        ls = n.p.get("ls")
        if ls is None:
            ls = 0.7 if role == "overline" else (-0.012 * size if size >= 26 else 0.0)
        return size, bold, lh, ls

    def _text(self, n):
        t = str(n.p.get("text", ""))
        return t.upper() if n.p.get("role") == "overline" else t

    def size(self, ui, n, aw):
        size, bold, lh, ls = self.font(ui, n)
        lines = wrap(self._text(n), size, aw, bold, n.p.get("max_lines"), ui.T.font, ls)
        W = max((_tw(ln, size, bold, ui.T.font, ls) for ln in lines), default=0.0)
        return (min(aw, W + 1.0), lh * len(lines))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        size, bold, lh, ls = self.font(ui, n)
        role = n.p.get("role", "body")
        lines = wrap(self._text(n), size, w + 0.5, bold, n.p.get("max_lines"), T.font, ls)
        col = n.p.get("color")
        if col is None:
            col = {"small": T.ink2, "caption": T.ink2, "overline": T.accent_ink if not T.wire else T.ink2,
                   "link": T.accent_ink}.get(role, T.ink)
        else:
            col = T.tone(col)
        align = n.p.get("align", "left")
        bars = T.wire and role in ("body", "small", "caption") and not n.p.get("real")
        for i, ln in enumerate(lines):
            cy = y + i * lh + lh / 2
            if cy > y + h + 1:
                break
            if bars:
                lw = _tw(ln, size, bold, T.font, ls) * (0.92 if i == len(lines) - 1 else 1.0)
                bx = x if align == "left" else (x + (w - lw) / 2 if align == "center" else x + w - lw)
                _bar(ui, g, bx, cy, lw, size)
                continue
            if align == "center":
                _txt(ui, g, x + w / 2, cy, ln, size, col, bold, "mc", ls)
            elif align == "right":
                _txt(ui, g, x + w, cy, ln, size, col, bold, "rc", ls)
            else:
                _txt(ui, g, x, cy, ln, size, col, bold, "lc", ls)
            if n.p.get("underline"):
                lw = _tw(ln, size, bold, T.font, ls)
                lx = x if align == "left" else (x + (w - lw) / 2 if align == "center" else x + w - lw)
                g.line(lx, cy + size * 0.55, lx + lw, cy + size * 0.55, stroke=col, stroke_width=max(ui.hl, 1))


@_spec("divider")
class _Divider(_Spec):
    def size(self, ui, n, aw):
        if n.p.get("vertical"):
            return (1.0, 24.0)
        return (aw, 20.0 if n.p.get("label") else 1.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        if n.p.get("vertical"):
            g.line(x + w / 2, y, x + w / 2, y + h, stroke=T.border, stroke_width=ui.hl)
            return
        ins = n.p.get("inset", 0.0)
        cy = y + h / 2
        lab = n.p.get("label")
        if lab:
            tw = _tw(str(lab), 12, False, T.font)
            _txt(ui, g, x + w / 2, cy, lab, 12, T.ink3, anchor="mc")
            ui.hline(g, x + ins, x + w / 2 - tw / 2 - 12, cy)
            ui.hline(g, x + w / 2 + tw / 2 + 12, x + w, cy)
        else:
            ui.hline(g, x + ins, x + w, cy)


@_spec("icon", w="hug")
class _Icon(_Spec):
    def size(self, ui, n, aw):
        s = float(n.p.get("size", 24))
        return (s, s)

    def paint(self, ui, g, n, x, y, w, h):
        s = float(n.p.get("size", 24))
        _ico(ui, g, n.p.get("icon"), x + w / 2, y + h / 2, s, ui.T.tone(n.p.get("color")) if n.p.get("color")
             else ui.T.ink, fill=ui.T.tone(n.p.get("fill")) if n.p.get("fill") else None)


@_spec("icon_badge", w="hug")
class _IconBadge(_Spec):
    def size(self, ui, n, aw):
        s = float(n.p.get("size", 40))
        return (s, s)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        s = float(n.p.get("size", 40))
        cx, cy = x + w / 2, y + h / 2
        col = T.tone(n.p.get("tone"))
        solid = n.p.get("solid", False)
        fill = col if solid else T.soft(col, 1.3)
        if T.wire:
            fill = T.surface2
        if n.p.get("shape", "rounded") == "circle":
            g.circle(cx, cy, s / 2, fill=fill)
        else:
            g.rect(cx - s / 2, cy - s / 2, s, s, r=min(T.r["small"] * s / 36, s / 2) if T.r["small"] < 900 else s / 2,
                   fill=fill)
        _ico(ui, g, n.p.get("icon"), cx, cy, s * 0.5, _on(col) if solid else (col if not T.wire else T.ink2))


@_spec("button", w="hug")
class _Button(_Spec):
    def dims(self, n):
        return _BTN.get(n.p.get("size", "md"), _BTN["md"])

    def size(self, ui, n, aw):
        H, fs, px = self.dims(n)
        tw = _tw(str(n.p.get("text") or ""), fs, True, ui.T.font)
        iw = (fs + 4 + (8 if n.p.get("text") else 0)) if n.p.get("icon") else 0
        iw += (fs + 4 + 8) if n.p.get("icon_right") else 0
        if n.p.get("variant") == "link":
            px = 0
        if n.p.get("full"):
            return (aw, H)
        return (min(aw, tw + iw + 2 * px), H)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        H, fs, px = self.dims(n)
        variant = n.p.get("variant", "primary")
        bh = min(h, H)
        by = y + (h - bh) / 2
        fill, fg, br = _btn_colors(ui, variant)
        r = _round_r(ui, "button", bh)
        if fill or br:
            g.rect(x, by, w, bh, r=r, fill=fill, stroke=br, stroke_width=ui.hl if br else None)
        if variant == "link":
            px = 0
        text, ic, icr = n.p.get("text"), n.p.get("icon"), n.p.get("icon_right")
        isz = fs + 4
        iw = (isz + (8 if text else 0)) if ic else 0
        irw = (isz + 8) if icr else 0
        room = max(w - 2 * px - iw - irw, 0)
        text = _ellipsize(str(text), fs, room, True, T.font) if text else ""
        tw = _tw(text, fs, True, T.font)
        sx = x + (w - (iw + tw + irw)) / 2
        cy = by + bh / 2
        if ic:
            _ico(ui, g, ic, sx + isz / 2, cy, isz, fg, sw=2.0)
        if text:
            _txt(ui, g, sx + iw, cy, text, fs, fg, True, "lc")
        if icr:
            _ico(ui, g, icr, sx + iw + tw + 8 + isz / 2, cy, isz, fg, sw=2.0)


@_spec("icon_button", w="hug")
class _IconButton(_Spec):
    def size(self, ui, n, aw):
        s = float({"sm": 32, "md": 40, "lg": 48, "xl": 56}.get(n.p.get("size", "md"), n.p.get("size", 40))
                  if not _num(n.p.get("size")) else n.p.get("size"))
        return (s, s)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        s = min(w, h)
        cx, cy = x + w / 2, y + h / 2
        v = n.p.get("variant", "ghost")
        fill, fg, br = {"filled": (T.accent, T.on_accent, None), "tonal": (T.accent_soft, T.accent_ink, None),
                        "outline": (None, T.ink, T.border_strong),
                        "surface": (T.surface2 if not T.glass else T.surface, T.ink, None if not T.glass else T.border),
                        "inverse": (T.ink, T.bg, None)}.get(v, (None, T.ink, None))
        if n.p.get("color"):
            fg = T.tone(n.p["color"])
        if fill or br:
            if T.r["button"] >= 900 or n.p.get("shape") == "circle" or T.style == "material":
                g.circle(cx, cy, s / 2, fill=fill, stroke=br, stroke_width=ui.hl if br else None)
            else:
                rr = min(T.r["control"], s / 2)
                g.rect(cx - s / 2, cy - s / 2, s, s, r=rr, fill=fill, stroke=br, stroke_width=ui.hl if br else None)
        _ico(ui, g, n.p.get("icon"), cx, cy, s * (0.5 if fill or br else 0.6), fg)
        b = n.p.get("badge")
        if b:
            bx, by = cx + s * 0.22, cy - s * 0.24
            if b is True:
                g.circle(bx, by, 4.5, fill=T.danger, stroke=T.bg if not T.glass else None, stroke_width=1.5)
            else:
                t = str(b)
                bw = max(16.0, _tw(t, 10, True, T.font) + 8)
                g.rect(bx - 4, by - 8, bw, 16, r=8, fill=T.danger, stroke=T.bg if not T.glass else None,
                       stroke_width=1.5)
                _txt(ui, g, bx - 4 + bw / 2, by, t, 10, "#ffffff", True, "mc")


@_spec("field")
class _Field(_Spec):
    def fh(self, n):
        k = n.p.get("kind", "input")
        if k == "textarea":
            return 24.0 * n.p.get("rows", 3) + 24.0
        if k == "search":
            return 44.0
        return float(n.p.get("field_h", 48))

    def size(self, ui, n, aw):
        h = self.fh(n) + (28.0 if n.p.get("label") else 0.0) + (22.0 if (n.p.get("helper") or n.p.get("error"))
                                                                  else 0.0)
        return (aw if n.w == "fill" else min(aw, float(n.p.get("natural_w", 280))), h)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        k = p.get("kind", "input")
        fy = y
        if p.get("label"):
            _txt(ui, g, x, y + 9, p["label"], 13, T.ink2 if not T.glass else T.ink, True, "lc", maxw=w)
            fy += 28
        fh = self.fh(n)
        err = p.get("error")
        focused = p.get("focused")
        r = min(T.r["control"], fh / 2) if k != "search" else min(max(T.r["control"], 10), fh / 2)
        if T.r["button"] >= 900 and k == "search":
            r = fh / 2
        if T.glass:
            fill, br = ("#ffffff14" if T.dark else "#ffffff73"), T.border
        elif T.wire:
            fill, br = T.surface, T.border_strong
        elif k == "search":
            fill, br = (T.surface2, None) if T.dark else (T.surface, T.border_strong)
        elif T.style == "material":
            fill, br = None, T.border_strong if not T.dark else T.ink3
        else:
            fill, br = (T.surface, T.border_strong) if not T.dark else (T.surface2, T.border)
        if err:
            br = T.danger
        if focused and not err:
            br = T.accent
            g.rect(x - 3, fy - 3, w + 6, fh + 6, r=r + 3, fill=None, stroke=alpha(T.accent, 0.22), stroke_width=4)
        g.rect(x, fy, w, fh, r=r, fill=fill, stroke=br, stroke_width=(ui.hl * (1.6 if (focused or err) else 1.0))
               if br else None)
        lead = p.get("icon") or ("search" if k == "search" else None)
        trail = p.get("trailing") or {"select": "chevron-down", "date": "calendar", "password": "eye"}.get(k)
        tx = x + 14
        cy = fy + (fh / 2 if k != "textarea" else 24)
        if lead:
            _ico(ui, g, lead, x + 14 + 10, cy, 20, T.ink3)
            tx = x + 14 + 20 + 10
        right = x + w - 14
        if trail:
            if _icon_name(trail):
                _ico(ui, g, trail, right - 10, cy, 20, T.ink3)
                right -= 30
            else:
                tw_ = _txt(ui, g, right, cy, trail, 14, T.ink2, False, "rc")
                right -= tw_ + 10
        val = p.get("value")
        if k == "password" and val:
            val = "•" * min(len(str(val)), 14)
        shown, col = (val, T.ink) if val not in (None, "") else (p.get("placeholder") or "", T.ink3)
        if k == "textarea":
            lines = wrap(str(shown), 15, right - tx, False, max(int((fh - 16) // 22), 1), T.font)
            for i, ln in enumerate(lines):
                if T.wire and val:
                    _bar(ui, g, tx, fy + 24 + i * 22, _tw(ln, 15, False, T.font), 15)
                else:
                    _txt(ui, g, tx, fy + 24 + i * 22, ln, 15, col)
        else:
            if T.wire and val and k != "password":
                _bar(ui, g, tx, cy, min(_tw(str(shown), 15, False, T.font), right - tx), 15)
            else:
                tw_ = _txt(ui, g, tx, cy, shown, 15, col, maxw=right - tx)
                if focused and val is not None:
                    g.line(tx + tw_ + 2, cy - 9, tx + tw_ + 2, cy + 9, stroke=T.accent, stroke_width=1.5)
        msg = err or p.get("helper")
        if msg:
            _txt(ui, g, x + 2, fy + fh + 13, msg, 12, T.danger if err else T.ink3, maxw=w - 2)


def _label_block(ui, n, aw):
    """(lines, sub_lines, height) of a label + optional sub-label at width ``aw``."""
    T = ui.T
    lab = wrap(str(n.p.get("label") or ""), 15, max(aw, 10), False, 2, T.font) if n.p.get("label") else ()
    sub = wrap(str(n.p.get("sub")), 13, max(aw, 10), False, 2, T.font) if n.p.get("sub") else ()
    return lab, sub, len(lab) * 22.0 + len(sub) * 18.0


def _paint_label(ui, g, n, x, y, w, h, cy0=None):
    T = ui.T
    lab, sub, th = _label_block(ui, n, w)
    ty = y + (h - th) / 2 if cy0 is None else cy0 - 11
    for i, ln in enumerate(lab):
        _txt(ui, g, x, ty + 11 + i * 22, ln, 15, T.ink)
    for i, ln in enumerate(sub):
        yy = ty + len(lab) * 22 + 9 + i * 18
        if T.wire:
            _bar(ui, g, x, yy, _tw(ln, 13, False, T.font), 13)
        else:
            _txt(ui, g, x, yy, ln, 13, T.ink2)


@_spec("check")
class _Check(_Spec):
    def size(self, ui, n, aw):
        lab, sub, th = _label_block(ui, n, aw - 34)
        tw = max([_tw(t, 15, False, ui.T.font) for t in lab] + [_tw(t, 13, False, ui.T.font) for t in sub] + [0])
        return (min(aw, 34 + tw + 2) if lab else 22.0, max(24.0, th))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        on = bool(n.p.get("checked"))
        lab, sub, th = _label_block(ui, n, w - 34)
        cy = (y + (h - th) / 2 + 11) if lab else y + h / 2
        if n.p.get("radio"):
            if on:
                g.circle(x + 11, cy, 10.25, fill=None, stroke=T.accent, stroke_width=2)
                g.circle(x + 11, cy, 5.5, fill=T.accent)
            else:
                g.circle(x + 11, cy, 10.25, fill=None if not T.glass else T.surface2, stroke=T.border_strong
                         if not T.glass else T.border_strong, stroke_width=1.5)
        else:
            rr = 6 if T.style != "material" else 3
            if T.wire:
                rr = 3
            if on:
                g.rect(x, cy - 11, 22, 22, r=rr, fill=T.accent)
                g.path(f"M{x + 6} {cy + 0.5}l3.6 3.6 6.4-7", fill=None, stroke=T.on_accent, stroke_width=2.2,
                       cap="round", join="round")
            else:
                g.rect(x + 0.75, cy - 10.25, 20.5, 20.5, r=rr, fill=None if not T.glass else T.surface2,
                       stroke=T.border_strong, stroke_width=1.5)
        if lab:
            _paint_label(ui, g, n, x + 34, y, w - 34, h, cy0=cy)


def _switch(ui, g, x, cy, on, small=False):
    T = ui.T
    if T.style == "material":
        tw, th = 52, 32
        g.rect(x, cy - th / 2, tw, th, r=th / 2, fill=T.accent if on else T.surface3,
               stroke=None if on else T.ink3, stroke_width=2 if not on else None)
        if on:
            g.circle(x + tw - 16, cy, 12, fill=T.on_accent)
            g.path(f"M{x + tw - 21} {cy}l3.5 3.5 6.5-7", fill=None, stroke=T.accent, stroke_width=2, cap="round",
                   join="round")
        else:
            g.circle(x + 16, cy, 8, fill=T.ink3)
        return tw, th
    tw, th = (44, 26) if (small or T.wire) else (51, 31)
    track = T.accent if on else (T.surface3 if not T.glass else ("#ffffff2e" if T.dark else "#00000014"))
    if T.wire:
        track = T.ink2 if on else T.surface3
    g.rect(x, cy - th / 2, tw, th, r=th / 2, fill=track)
    kr = th / 2 - 2
    kx = x + tw - th / 2 if on else x + th / 2
    with g.group(None, shadow=dict(dx=0, dy=1.5, blur=2.5, color="#000000", opacity=0.22)) as k:
        k.circle(kx, cy, kr, fill="#ffffff")
    return tw, th


@_spec("toggle")
class _Toggle(_Spec):
    def size(self, ui, n, aw):
        if not n.p.get("label"):
            return (52.0, 32.0)
        lab, sub, th = _label_block(ui, n, aw - 68)
        return (aw, max(32.0, th))

    def paint(self, ui, g, n, x, y, w, h):
        if not n.p.get("label"):
            _switch(ui, g, x, y + h / 2, bool(n.p.get("on", True)))
            return
        sw = 52 if ui.T.style == "material" else (44 if ui.T.wire else 51)
        _switch(ui, g, x + w - sw, y + h / 2, bool(n.p.get("on", True)))
        _paint_label(ui, g, n, x, y, w - sw - 16, h)


@_spec("slider")
class _Slider(_Spec):
    def size(self, ui, n, aw):
        h = 28.0 + (28.0 if n.p.get("label") else 0.0) + (20.0 if n.p.get("min_label") else 0.0)
        return (aw if n.w != "hug" else min(aw, 240.0), h)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        v = min(max(float(n.p.get("value", 0.5)), 0.0), 1.0)
        yy = y
        if n.p.get("label"):
            _txt(ui, g, x, yy + 10, n.p["label"], 15, T.ink, maxw=w * 0.65)
            vt = n.p.get("value_text")
            if vt is None:
                vt = f"{int(round(v * 100))}%"
            if vt:
                _txt(ui, g, x + w, yy + 10, vt, 15, T.ink2, True, "rc")
            yy += 28
        cy = yy + 14
        kr = 11.0
        th = 6.0 if T.style == "material" else 4.0
        x0, x1 = x + kr, x + w - kr
        kx = x0 + (x1 - x0) * v
        tone = T.tone(n.p.get("tone"))
        g.rect(x0, cy - th / 2, x1 - x0, th, r=th / 2, fill=T.surface3 if not T.glass else T.border)
        g.rect(x0, cy - th / 2, max(kx - x0, th), th, r=th / 2, fill=tone)
        if T.style == "material":
            g.rect(kx - 2, cy - 12, 4, 24, r=2, fill=tone)
        else:
            with g.group(None, shadow=dict(dx=0, dy=1.5, blur=3, color="#000000", opacity=0.25)) as k:
                k.circle(kx, cy, kr, fill="#ffffff", stroke=None if T.dark else T.border, stroke_width=ui.hl)
        if n.p.get("min_label"):
            _txt(ui, g, x, cy + 24, n.p["min_label"], 12, T.ink3)
            _txt(ui, g, x + w, cy + 24, n.p.get("max_label", ""), 12, T.ink3, anchor="rc")


@_spec("stepper")
class _Stepper(_Spec):
    def size(self, ui, n, aw):
        if not n.p.get("label"):
            return (112.0, 36.0)
        lab, sub, th = _label_block(ui, n, aw - 128)
        return (aw, max(36.0, th))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        cw, ch = 112.0, 36.0
        cx0 = x + w - cw
        cy = y + h / 2
        r = min(T.r["control"], ch / 2)
        g.rect(cx0, cy - ch / 2, cw, ch, r=r, fill=T.surface2 if not T.glass else T.surface,
               stroke=T.border if (not T.dark and not T.glass) else None, stroke_width=ui.hl)
        g.line(cx0 + 36, cy - 10, cx0 + 36, cy + 10, stroke=T.border_strong, stroke_width=ui.hl)
        g.line(cx0 + cw - 36, cy - 10, cx0 + cw - 36, cy + 10, stroke=T.border_strong, stroke_width=ui.hl)
        _ico(ui, g, "minus", cx0 + 18, cy, 16, T.ink2)
        _ico(ui, g, "plus", cx0 + cw - 18, cy, 16, T.ink)
        _txt(ui, g, cx0 + cw / 2, cy, str(n.p.get("value", 1)), 15, T.ink, True, "mc")
        if n.p.get("label"):
            _paint_label(ui, g, n, x, y, w - cw - 16, h)


@_spec("segmented")
class _Segmented(_Spec):
    def size(self, ui, n, aw):
        items = n.p.get("items", [])
        H = 36.0 if n.p.get("size", "md") == "md" else 30.0
        if n.w == "hug":
            return (min(aw, sum(_tw(str(t), 13, True, ui.T.font) + 32 for t in items) + 4), H)
        return (aw, H)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        items = [str(t) for t in n.p.get("items", [])]
        act = n.p.get("active", 0)
        if not items:
            return
        k = len(items)
        if T.style == "material":
            r = h / 2
            g.rect(x, y, w, h, r=r, fill=None, stroke=T.border_strong, stroke_width=ui.hl)
            sw = w / k
            for i, t in enumerate(items):
                sx = x + i * sw
                if i == act:
                    rr = (r if i == 0 else 0, r if i == k - 1 else 0, r if i == k - 1 else 0, r if i == 0 else 0)
                    g.rect(sx, y, sw, h, r=rr, fill=T.accent_soft)
                if i:
                    g.line(sx, y, sx, y + h, stroke=T.border_strong, stroke_width=ui.hl)
                tw = _tw(t, 13, True, T.font) + (22 if i == act else 0)
                tx = sx + (sw - tw) / 2
                if i == act:
                    _ico(ui, g, "tick", tx + 8, y + h / 2, 16, T.accent_ink)
                    tx += 22
                _txt(ui, g, tx, y + h / 2, t, 13, T.accent_ink if i == act else T.ink, True, maxw=sw - 12)
            return
        r = min(T.r["control"] if T.r["button"] < 900 else h / 2, h / 2)
        g.rect(x, y, w, h, r=r, fill=T.surface2 if not T.glass else T.surface2,
               stroke=T.border if T.glass else None, stroke_width=ui.hl)
        sw = (w - 4) / k
        for i, t in enumerate(items):
            sx = x + 2 + i * sw
            if i == act:
                sel = T.surface if not T.dark else T.surface3
                if T.glass:
                    sel = "#ffffff38" if T.dark else "#ffffffe6"
                with g.group(None, shadow=None if (T.dark or T.wire) else dict(dx=0, dy=1, blur=3, color="#101828",
                                                                                opacity=0.12)) as s_:
                    s_.rect(sx, y + 2, sw, h - 4, r=max(r - 2, 0), fill=sel)
            elif i and i != act and i - 1 != act:
                g.line(sx, y + h * 0.3, sx, y + h * 0.7, stroke=T.border_strong, stroke_width=ui.hl)
            _txt(ui, g, sx + sw / 2, y + h / 2, t, 13, T.ink if i == act else T.ink2, i == act, "mc", maxw=sw - 10)


@_spec("tabs")
class _Tabs(_Spec):
    def size(self, ui, n, aw):
        return (aw, 44.0 if n.p.get("variant", "underline") == "underline" else 36.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        items = [str(t) for t in n.p.get("items", [])]
        act = n.p.get("active", 0)
        counts = n.p.get("counts") or {}
        var = n.p.get("variant", "underline")
        eq = n.p.get("fill", False)
        widths = []
        for i, t in enumerate(items):
            tw = _tw(t, 15, i == act, T.font)
            c = counts.get(i, counts.get(t)) if isinstance(counts, dict) else None
            if c is not None:
                tw += 8 + max(20, _tw(str(c), 11, True, T.font) + 10)
            widths.append(tw)
        pad = 14 if var == "pills" else 0
        gap = 24 if var == "underline" else 6
        if eq:
            sw = w / max(len(items), 1)
            xs = [x + i * sw for i in range(len(items))]
            ws_ = [sw] * len(items)
        else:
            xs, cx = [], x
            for tw in widths:
                xs.append(cx)
                cx += tw + 2 * pad + gap
            ws_ = [tw + 2 * pad for tw in widths]
        if var == "underline":
            ui.hline(g, x, x + w, y + h - 0.5)
        for i, t in enumerate(items):
            if xs[i] + ws_[i] > x + w + 0.5:
                break
            on = i == act
            cy = y + (h / 2 if var != "underline" else h / 2 - 1)
            if var == "pills" and on:
                g.rect(xs[i], y, ws_[i], h, r=min(T.r["button"], h / 2), fill=T.accent_soft)
            tx = xs[i] + (ws_[i] - widths[i]) / 2
            col = (T.accent_ink if var != "underline" or T.style == "material" else T.ink) if on else T.ink2
            tw = _txt(ui, g, tx, cy, t, 15, col, on)
            c = counts.get(i, counts.get(t)) if isinstance(counts, dict) else None
            if c is not None:
                bw = max(20, _tw(str(c), 11, True, T.font) + 10)
                g.rect(tx + tw + 8, cy - 9, bw, 18, r=9, fill=T.accent_soft if on else T.surface2)
                _txt(ui, g, tx + tw + 8 + bw / 2, cy, str(c), 11, T.accent_ink if on else T.ink2, True, "mc")
            if var == "underline" and on:
                g.rect(xs[i] + (0 if eq else 0), y + h - 2.5, ws_[i], 2.5, r=1.25, fill=T.accent)


@_spec("breadcrumb")
class _Breadcrumb(_Spec):
    def size(self, ui, n, aw):
        return (aw, 20.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        items = [str(t) for t in n.p.get("items", [])]
        while len(items) > 2:
            tot = sum(_tw(t, 14, False, T.font) for t in items) + 28 * (len(items) - 1)
            if tot <= w:
                break
            mid = len(items) // 2
            if items[mid] == "…":
                items.pop(mid - 1 if mid > 1 else mid + 1)
            else:
                items[mid] = "…"
        cx, cy = x, y + h / 2
        for i, t in enumerate(items):
            last = i == len(items) - 1
            cx += _txt(ui, g, cx, cy, t, 14, T.ink if last else T.ink2, last, maxw=x + w - cx)
            if not last:
                _ico(ui, g, "chevron-right", cx + 14, cy, 14, T.ink3)
                cx += 28


@_spec("pagination", w="hug")
class _Pagination(_Spec):
    def seq(self, n):
        p, N = int(n.p.get("page", 1)), int(n.p.get("pages", 10))
        if N <= 7:
            return list(range(1, N + 1))
        if p <= 4:
            return [1, 2, 3, 4, 5, None, N]
        if p >= N - 3:
            return [1, None, N - 4, N - 3, N - 2, N - 1, N]
        return [1, None, p - 1, p, p + 1, None, N]

    def size(self, ui, n, aw):
        k = len(self.seq(n)) + 2
        return (min(aw, k * 36 + (k - 1) * 6), 36.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        seq = [("prev",)] + [(v,) for v in self.seq(n)] + [("next",)]
        p = int(n.p.get("page", 1))
        cx = x
        r = min(T.r["small"], 18)
        for (v,) in seq:
            if cx + 36 > x + w + 0.5:
                break
            if v in ("prev", "next"):
                g.rect(cx, y, 36, 36, r=r, fill=None, stroke=T.border_strong if not T.glass else T.border,
                       stroke_width=ui.hl)
                _ico(ui, g, "chevron-left" if v == "prev" else "chevron-right", cx + 18, y + 18, 16, T.ink2)
            elif v is None:
                _txt(ui, g, cx + 18, y + 18, "…", 14, T.ink3, anchor="mc")
            else:
                on = v == p
                if on:
                    g.rect(cx, y, 36, 36, r=r, fill=T.accent)
                _txt(ui, g, cx + 18, y + 18, str(v), 14, T.on_accent if on else T.ink2, on, "mc")
            cx += 42


# ---- navigation bars


def _bar_item(ui, g, item, cx, cy, size=24, color=None):
    """Leading / trailing item of an app bar: icon name, 'avatar:Name', 'button:Text' or plain text. → width."""
    T = ui.T
    s = str(item)
    if s.startswith("avatar:"):
        _avatar(ui, g, cx - 16, cy, 32, s[7:], "portrait")
        return 32.0
    if s.startswith("button:"):
        t = s[7:]
        bw = _tw(t, 13, True, T.font) + 24
        g.rect(cx - bw, cy - 16, bw, 32, r=_round_r(ui, "button", 32), fill=T.accent)
        _txt(ui, g, cx - bw / 2, cy, t, 13, T.on_accent, True, "mc")
        return bw
    if _icon_name(s):
        _ico(ui, g, s, cx - size / 2, cy, size, color or T.ink)
        return float(size)
    return _txt(ui, g, cx, cy, s, 16, T.accent_ink, True, "rc")


@_spec("app_bar")
class _AppBar(_Spec):
    def size(self, ui, n, aw):
        if ui.wide:
            return (aw, 64.0)
        return (aw, 52.0 + (52.0 if n.p.get("large") else 0.0))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        title, lead, trail = p.get("title") or "", p.get("leading"), list(p.get("trailing") or [])
        if ui.wide:
            if not p.get("transparent"):
                ui.panel(g, x, y, w, h, r=0, kind="bar" if T.glass else "plain",
                         fill=None if T.glass else T.surface, border=None, shadow=None)
                ui.hline(g, x, x + w, y + h - 0.5)
            cy = y + h / 2
            tx = x + 24
            if lead:
                _bar_item(ui, g, lead, tx + 24, cy, 22)
                tx += 40
            right = x + w - 24
            for it in reversed(trail):
                right -= _bar_item(ui, g, it, right, cy, 22) + 16
            _txt(ui, g, tx, cy - (8 if p.get("subtitle") else 0), title, 20, T.ink, True, maxw=right - tx - 16)
            if p.get("subtitle"):
                _txt(ui, g, tx, cy + 12, p["subtitle"], 13, T.ink2, maxw=right - tx - 16)
            return
        if T.glass and not p.get("transparent"):
            ui.panel(g, x + 12, y + 2, w - 24, 48, r=24, kind="bar")
        elif T.style == "material" and not p.get("transparent"):
            g.rect(x, y, w, h, fill=T.bg)
        cy = y + 26
        lx = x + 16 + (4 if T.glass else 0)
        title_left = lx
        if lead:
            ls = str(lead)
            if ls.startswith("back:"):
                _ico(ui, g, "chevron-left", lx + 10, cy, 24, T.accent_ink)
                title_left = lx + 22 + _txt(ui, g, lx + 22, cy, ls[5:], 16, T.accent_ink) + 12
            else:
                wid = _bar_item(ui, g, "chevron-left" if ls == "back" else ls, lx + (32 if ls.startswith("avatar:")
                                                                                      else 24), cy, 24)
                title_left = lx + wid + 14
        right = x + w - 16 - (4 if T.glass else 0)
        for it in reversed(trail):
            right -= _bar_item(ui, g, it, right, cy, 24) + 18
        right += 18
        if p.get("large"):
            _txt(ui, g, x + 20, y + 52 + 24, title, 30, T.ink, True, ls=-0.4, maxw=w - 40)
            if p.get("subtitle"):
                _txt(ui, g, x + 20, y + 52 + 4, p["subtitle"], 13, T.ink2, maxw=w - 40)
            return
        center = p.get("center")
        if center is None:
            center = T.style not in ("material",) and bool(lead or trail)
        if center:
            room = 2 * min(x + w / 2 - title_left, right - (x + w / 2)) - 8
            if p.get("subtitle"):
                _txt(ui, g, x + w / 2, cy - 8, title, 16, T.ink, True, "mc", maxw=room)
                _txt(ui, g, x + w / 2, cy + 10, p["subtitle"], 12, T.ink2, False, "mc", maxw=room)
            else:
                _txt(ui, g, x + w / 2, cy, title, 17, T.ink, True, "mc", maxw=room)
        else:
            if p.get("subtitle"):
                _txt(ui, g, title_left, cy - 8, title, 17, T.ink, True, maxw=right - title_left - 8)
                _txt(ui, g, title_left, cy + 11, p["subtitle"], 12, T.ink2, maxw=right - title_left - 8)
            else:
                _txt(ui, g, title_left, cy, title, 20 if T.style == "material" else 18, T.ink, True,
                     maxw=right - title_left - 8)


def _items(items):
    """[(icon, label), ...] from tuples, dicts or plain strings."""
    out = []
    for it in items or []:
        if isinstance(it, dict):
            out.append((it.get("icon"), it.get("label", ""), it.get("count"), it))
        elif isinstance(it, (tuple, list)):
            out.append((it[0], it[1] if len(it) > 1 else "", it[2] if len(it) > 2 else None, {}))
        else:
            s = str(it)
            out.append((s if _icon_name(s) else None, s.replace("-", " ").title() if _icon_name(s) else s, None, {}))
    return out


@_spec("tab_bar")
class _TabBar(_Spec):
    def size(self, ui, n, aw):
        sb = ui.safe_bottom
        if ui.T.style == "material":
            return (aw, 80.0 + max(sb - 14, 0))
        if ui.T.glass:
            return (aw, 64.0 + max(sb, 12.0) + 6)
        return (aw, 50.0 + sb + (0 if sb else 6))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        items = _items(n.p.get("items"))
        act = n.p.get("active", 0)
        badges = n.p.get("badges") or {}
        k = max(len(items), 1)
        if T.style == "material":
            g.rect(x, y, w, h, fill=T.surface2)
            sw = w / k
            for i, (ic, lab, cnt, _) in enumerate(items):
                cx = x + sw * (i + 0.5)
                on = i == act
                if on:
                    g.rect(cx - 32, y + 12, 64, 32, r=16, fill=mix(T.surface2, T.accent, 0.22 if T.dark else 0.18))
                _ico(ui, g, ic, cx, y + 28, 24, T.accent_ink if on else T.ink2)
                _txt(ui, g, cx, y + 58, lab, 12, T.ink if on else T.ink2, on, "mc", maxw=sw - 6)
                self._badge(ui, g, badges, i, cx + 12, y + 18)
            return
        if T.glass:
            bh = 64.0
            by = y + 6
            bx0, bw = x + 20, w - 40
            ui.panel(g, bx0, by, bw, bh, r=bh / 2, kind="bar")
            sw = (bw - 16) / k
            for i, (ic, lab, cnt, _) in enumerate(items):
                cx = bx0 + 8 + sw * (i + 0.5)
                on = i == act
                if on:
                    g.rect(cx - sw / 2 + 2, by + 6, sw - 4, bh - 12, r=(bh - 12) / 2,
                           fill="#ffffff2e" if T.dark else "#ffffffcc")
                _ico(ui, g, ic, cx, by + 24, 22, T.accent_ink if on else T.ink2)
                _txt(ui, g, cx, by + 46, lab, 10.5, T.accent_ink if on else T.ink2, True, "mc", maxw=sw - 6)
                self._badge(ui, g, badges, i, cx + 11, by + 15)
            return
        g.rect(x, y, w, h, fill=T.bar)
        ui.hline(g, x, x + w, y + 0.5)
        sw = w / k
        for i, (ic, lab, cnt, _) in enumerate(items):
            cx = x + sw * (i + 0.5)
            on = i == act
            col = T.accent_ink if on else T.ink3
            _ico(ui, g, ic, cx, y + 19, 24, col, sw=2.0 if on else 1.75,
                 fill=T.soft(T.accent, 1.6) if (on and not T.wire) else None)
            _txt(ui, g, cx, y + 40, lab, 10.5, col, True, "mc", maxw=sw - 6)
            self._badge(ui, g, badges, i, cx + 12, y + 10)

    def _badge(self, ui, g, badges, i, bx, by):
        b = badges.get(i) if isinstance(badges, dict) else None
        if not b:
            return
        T = ui.T
        t = "" if b is True else str(b)
        bw = 10.0 if not t else max(18.0, _tw(t, 10, True, T.font) + 9)
        g.rect(bx - 2, by - (5 if not t else 9), bw, 10 if not t else 18, r=9, fill=T.danger)
        if t:
            _txt(ui, g, bx - 2 + bw / 2, by, t, 10, "#ffffff", True, "mc")


@_spec("sidebar", w=256, h="fill")
class _Sidebar(_Spec):
    def size(self, ui, n, aw):
        k = len(n.p.get("items") or [])
        return (aw, 72.0 + k * 40.0 + (72.0 if n.p.get("user") else 0.0))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        if T.glass:
            ui.panel(g, x + 12, y + 12, w - 12, h - 24, r=T.r["card"], kind="card")
        else:
            g.rect(x, y, w, h, fill=T.surface if T.style != "material" else T.surface2)
            g.line(x + w - 0.5, y, x + w - 0.5, y + h, stroke=T.border, stroke_width=ui.hl)
        ix = x + (28 if T.glass else 16)
        iw = w - (40 if T.glass else 32)
        cy = y + 36 + (8 if T.glass else 0)
        brand = p.get("brand")
        if brand:
            _logo_mark(ui, g, ix + 2, cy - 14, 28, _hash(brand) % 6, T.accent)
            _txt(ui, g, ix + 40, cy, brand, 17, T.ink, True, maxw=iw - 40)
        cy += 44
        act = p.get("active", 0)
        idx = 0
        for it in p.get("items") or []:
            if isinstance(it, str) and it.startswith("#"):
                cy += 8
                _txt(ui, g, ix + 12, cy + 4, it.lstrip("# ").upper(), 11, T.ink3, True, ls=0.6, maxw=iw)
                cy += 28
                continue
            (ic, lab, cnt, _), = _items([it])
            on = (idx == act) or (lab == act)
            if on:
                g.rect(ix, cy, iw, 38, r=min(T.r["small"] + 2, 19), fill=T.accent_soft if not T.wire else T.surface2)
            col = T.accent_ink if on else T.ink2
            if ic:
                _ico(ui, g, ic, ix + 22, cy + 19, 20, col)
            right = ix + iw - 12
            if cnt is not None:
                t = str(cnt)
                bw = max(22, _tw(t, 11, True, T.font) + 12)
                g.rect(right - bw, cy + 10, bw, 18, r=9, fill=T.accent if on else T.surface2 if not T.glass
                       else T.surface3)
                _txt(ui, g, right - bw / 2, cy + 19, t, 11, T.on_accent if on else T.ink2, True, "mc")
                right -= bw + 8
            _txt(ui, g, ix + (44 if ic else 12), cy + 19, lab, 14, T.ink if on else T.ink2, on,
                 maxw=right - ix - 44)
            cy += 40
            idx += 1
        user = p.get("user")
        if user:
            name, email = (user, None) if isinstance(user, str) else (user[0], user[1] if len(user) > 1 else None)
            uy = y + h - (76 if T.glass else 64)
            ui.hline(g, ix, ix + iw, uy - 8)
            _avatar(ui, g, ix + 20, uy + 26, 36, name, "portrait")
            _txt(ui, g, ix + 46, uy + 18 if email else uy + 26, name, 14, T.ink, True, maxw=iw - 72)
            if email:
                _txt(ui, g, ix + 46, uy + 36, email, 12, T.ink3, maxw=iw - 72)
            _ico(ui, g, "more-vertical", ix + iw - 10, uy + 26, 18, T.ink3)


def _logo_mark(ui, g, x, y, s, kind=0, color=None):
    """Small abstract brand mark in an s x s box."""
    T = ui.T
    c = color or T.accent
    if T.wire:
        g.rect(x, y, s, s, r=s * 0.25, fill=T.surface3, stroke=T.border_strong, stroke_width=ui.hl)
        return
    on = _on(c)
    if kind == 0:
        g.rect(x, y, s, s, r=s * 0.28, fill=linear_gradient([mix(c, "#ffffff", 0.18), c], start=(0, 0), end=(1, 1)))
        g.path(f"M{x + s * 0.28} {y + s * 0.68}l{s * 0.22}-{s * 0.38} {s * 0.22} {s * 0.38}Z", fill=on)
    elif kind == 1:
        g.circle(x + s / 2, y + s / 2, s / 2, fill=c)
        g.circle(x + s * 0.62, y + s * 0.4, s * 0.2, fill=on)
    elif kind == 2:
        g.rect(x, y, s, s, r=s * 0.28, fill=c)
        g.rect(x + s * 0.24, y + s * 0.26, s * 0.18, s * 0.48, r=s * 0.06, fill=on)
        g.rect(x + s * 0.5, y + s * 0.26, s * 0.26, s * 0.2, r=s * 0.06, fill=on, opacity=0.75)
    elif kind == 3:
        pts = [(x + s / 2 + s / 2 * math.cos(math.radians(a)), y + s / 2 + s / 2 * math.sin(math.radians(a)))
               for a in range(-90, 270, 60)]
        g.polygon(pts, fill=c)
        g.circle(x + s / 2, y + s / 2, s * 0.18, fill=on)
    elif kind == 4:
        g.circle(x + s * 0.38, y + s / 2, s * 0.36, fill=c, opacity=0.85)
        g.circle(x + s * 0.64, y + s / 2, s * 0.36, fill=_hue(c, 40), opacity=0.85)
    else:
        g.rect(x, y, s, s, r=s * 0.5, fill=c)
        g.path(f"M{x + s * 0.3} {y + s * 0.5}h{s * 0.4}M{x + s * 0.5} {y + s * 0.3}v{s * 0.4}", fill=None, stroke=on,
               stroke_width=s * 0.1, cap="round")


@_spec("navbar")
class _Navbar(_Spec):
    def size(self, ui, n, aw):
        return (aw, 72.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        if p.get("bg") is not False:
            if T.glass:
                ui.panel(g, x, y, w, h, r=0, kind="bar")
            else:
                g.rect(x, y, w, h, fill=p.get("bg") or alpha(T.bg, 0.92))
            if p.get("border", True):
                ui.hline(g, x, x + w, y + h - 0.5)
        mw = min(w - 48, float(p.get("max_w") or ui.content_w))
        x0 = x + (w - mw) / 2
        cy = y + h / 2
        brand = p.get("brand") or "Brand"
        _logo_mark(ui, g, x0, cy - 14, 28, _hash(brand) % 6, T.accent)
        bx = x0 + 38 + _txt(ui, g, x0 + 38, cy, brand, 18, T.ink, True)
        right = x0 + mw
        acts = p.get("actions") or []
        for a in reversed(acts):
            t, v = (a, "primary") if isinstance(a, str) else (a[0], a[1] if len(a) > 1 else "primary")
            bw = _tw(t, 14, True, T.font) + 32
            fill, fg, br = _btn_colors(ui, v)
            if fill or br:
                g.rect(right - bw, cy - 19, bw, 38, r=_round_r(ui, "button", 38), fill=fill, stroke=br,
                       stroke_width=ui.hl if br else None)
            else:
                bw -= 16
            _txt(ui, g, right - bw / 2, cy, t, 14, fg, True, "mc")
            right -= bw + 12
        links = [str(l) for l in p.get("links") or []]
        lw = [_tw(l, 15, False, T.font) for l in links]
        tot = sum(lw) + 32 * max(len(links) - 1, 0)
        lx = x + w / 2 - tot / 2 if p.get("center_links", True) else bx + 48
        lx = max(lx, bx + 40)
        act = p.get("active")
        for l, ww in zip(links, lw):
            if lx + ww > right - 16:
                break
            _txt(ui, g, lx, cy, l, 15, T.ink if l == act else T.ink2, l == act)
            lx += ww + 32


# ---- lists, tiles, tables


@_spec("list_row")
class _ListRow(_Spec):
    def _lead_w(self, n):
        lead = n.p.get("leading")
        if not lead:
            return 0.0
        s = str(lead)
        return (44.0 if n.p.get("subtitle") else 40.0) if s.startswith(("avatar:", "portrait:", "initials:")) \
            else 36.0

    def size(self, ui, n, aw):
        base = 64.0 if n.p.get("subtitle") else 52.0
        if n.p.get("leading") and str(n.p["leading"]).startswith(("avatar", "portrait", "initials")) \
                and n.p.get("subtitle"):
            base = 68.0
        return (aw, base + (18.0 if n.p.get("lines", 1) > 1 and n.p.get("subtitle") else 0.0))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        cy = y + h / 2
        tx = x
        lead = p.get("leading")
        if lead:
            s = str(lead)
            lw = self._lead_w(n)
            if s.startswith(("avatar:", "portrait:")):
                _avatar(ui, g, x + lw / 2, cy, lw, s.split(":", 1)[1], "portrait", p.get("status"))
            elif s.startswith("initials:"):
                _avatar(ui, g, x + lw / 2, cy, lw, s.split(":", 1)[1], "initials", p.get("status"))
            elif s.startswith("img:"):
                ui.image_art(g, x, cy - lw / 2, lw, lw, seed=_hash(s), r=T.r["small"],
                             scene=s[4:] if s[4:] in _SCAPES else None)
            else:
                col = T.tone(p.get("tone") or (_PALETTE[_hash(s) % len(_PALETTE)] if p.get("colorful") else None))
                solid = p.get("solid", False)
                fill = col if solid else T.soft(col, 1.25)
                if T.wire:
                    fill = T.surface2
                g.rect(x, cy - 18, 36, 36, r=min(T.r["small"] + 2, 18), fill=fill)
                _ico(ui, g, s, x + 18, cy, 20, _on(col) if solid and not T.wire else (col if not T.wire else T.ink2))
            tx = x + lw + 12
        right = x + w
        tr = p.get("trailing")
        meta = p.get("meta")
        if p.get("toggle") is not None:
            sw = 52 if T.style == "material" else (44 if T.wire else 51)
            _switch(ui, g, right - sw, cy, bool(p["toggle"]))
            right -= sw + 12
        if p.get("chevron") or tr == "chevron":
            _ico(ui, g, "chevron-right", right - 8, cy, 16, T.ink3, sw=2.2)
            right -= 24
        elif tr == "check":
            _ico(ui, g, "tick", right - 10, cy, 20, T.accent_ink, sw=2.2)
            right -= 28
        elif tr and _icon_name(tr):
            _ico(ui, g, tr, right - 11, cy, 22, T.ink2)
            right -= 32
        badge = p.get("badge")
        if badge is not None and not meta:
            right -= _pill(ui, g, right, cy, badge, p.get("badge_tone", "accent"), anchor="r") + 10
        val = p.get("value")
        vcol = T.tone(p.get("value_tone")) if p.get("value_tone") else (T.ink if p.get("value_strong") else T.ink2)
        sub = p.get("subtitle")
        if meta:
            _txt(ui, g, right, (cy - 10) if sub else cy, meta, 12, T.ink3, anchor="rc")
            mw = _tw(str(meta), 12, False, T.font)
            if badge is not None and sub:
                _pill(ui, g, right, cy + 10, badge, p.get("badge_tone", "accent"), anchor="r", count=True)
            right_t = right - mw - 12
        else:
            right_t = right
        if val is not None:
            vw = _txt(ui, g, right, cy if not p.get("value_sub") else cy - 9, val, 15, vcol,
                      bool(p.get("value_strong")), "rc")
            if p.get("value_sub"):
                _txt(ui, g, right, cy + 10, p["value_sub"], 12, T.ink3, anchor="rc")
                vw = max(vw, _tw(str(p["value_sub"]), 12, False, T.font))
            right_t = min(right_t, right - vw - 12)
        title = p.get("title") or ""
        bold = p.get("bold", bool(sub))
        if sub:
            _txt(ui, g, tx, cy - 10, title, 15, T.ink, bold, maxw=right_t - tx)
            if T.wire:
                _bar(ui, g, tx, cy + 11, min(_tw(str(sub), 13, False, T.font), right - tx), 13)
            else:
                _txt(ui, g, tx, cy + 11, sub, 13, T.ink2, maxw=(right if meta and badge is None else right_t) - tx
                     if not (meta and badge is not None) else right - tx - 30)
        else:
            _txt(ui, g, tx, cy, title, 15, T.tone(p.get("title_tone")) if p.get("title_tone") else T.ink, bold,
                 maxw=right_t - tx)


def _pill(ui, g, x, cy, text, tone="accent", anchor="l", solid=False, count=False, size=12):
    """Badge pill at (x, cy); returns its width."""
    T = ui.T
    t = str(text)
    col = T.tone(tone)
    h = 20.0 if size <= 12 else 24.0
    if count or (t.isdigit() and len(t) <= 3):
        w = max(h, _tw(t, 11, True, T.font) + 10)
        x0 = x - w if anchor == "r" else x
        g.rect(x0, cy - h / 2, w, h, r=h / 2, fill=col if not T.wire else T.ink2)
        _txt(ui, g, x0 + w / 2, cy, t, 11, _on(col) if not T.wire else T.bg, True, "mc")
        return w
    w = _tw(t, size, True, T.font) + 16
    x0 = x - w if anchor == "r" else x
    fill = col if solid else T.soft(col, 1.2 if not T.dark else 1.0)
    if T.wire:
        fill = T.surface2
    g.rect(x0, cy - h / 2, w, h, r=h / 2, fill=fill)
    fg = _on(col) if solid else (mix(col, T.ink, 0.15) if not T.dark else mix(col, "#ffffff", 0.2))
    if T.glass and not solid:
        fg = T.ink if T.dark else mix(col, "#000000", 0.25)
    _txt(ui, g, x0 + w / 2, cy, t, size, fg if not T.wire else T.ink2, True, "mc")
    return w


@_spec("badge", w="hug")
class _Badge(_Spec):
    def size(self, ui, n, aw):
        t = str(n.p.get("text", ""))
        if t.isdigit() and len(t) <= 3:
            return (max(20.0, _tw(t, 11, True, ui.T.font) + 10), 20.0)
        return (min(aw, _tw(t, 12, True, ui.T.font) + 16 + (12 if n.p.get("dot") else 0)), 20.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        t = str(n.p.get("text", ""))
        if n.p.get("dot"):
            col = T.tone(n.p.get("tone"))
            g.rect(x, y + h / 2 - 10, w, 20, r=10, fill=T.soft(col, 1.2) if not T.wire else T.surface2)
            g.circle(x + 11, y + h / 2, 3.5, fill=col)
            _txt(ui, g, x + 19, y + h / 2, t, 12, mix(col, T.ink, 0.15) if not T.wire else T.ink2, True,
                 maxw=w - 24)
            return
        _pill(ui, g, x, y + h / 2, _ellipsize(t, 12, w - 16, True, T.font), n.p.get("tone", "accent"),
              solid=n.p.get("solid", False))


@_spec("chip", w="hug")
class _Chip(_Spec):
    def size(self, ui, n, aw):
        t = str(n.p.get("text", ""))
        w = _tw(t, 13, bool(n.p.get("selected")), ui.T.font) + 30
        if n.p.get("icon") or n.p.get("selected"):
            w += 22
        if n.p.get("close"):
            w += 20
        return (min(aw, w), 32.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        on = bool(n.p.get("selected"))
        r = min(T.r["chip"], 16)
        if on:
            g.rect(x, y, w, h, r=r, fill=T.accent_soft if not T.wire else T.surface3,
                   stroke=None if T.style != "material" else None)
        else:
            g.rect(x, y, w, h, r=r, fill=None if not T.glass else T.surface2,
                   stroke=T.border_strong if not T.glass else T.border, stroke_width=ui.hl)
        cx = x + 14
        fg = T.accent_ink if on else T.ink
        ic = "tick" if (on and not n.p.get("icon")) else n.p.get("icon")
        if ic:
            _ico(ui, g, ic, cx + 7, y + h / 2, 16, fg, sw=2.0)
            cx += 22
        room = x + w - 14 - (20 if n.p.get("close") else 0) - cx
        _txt(ui, g, cx, y + h / 2, n.p.get("text", ""), 13, fg, on, maxw=room)
        if n.p.get("close"):
            _ico(ui, g, "close", x + w - 18, y + h / 2, 14, T.ink2, sw=2.2)


@_spec("avatar", w="hug")
class _Avatar(_Spec):
    def size(self, ui, n, aw):
        s = float(n.p.get("size", 40))
        return (s, s)

    def paint(self, ui, g, n, x, y, w, h):
        s = min(w, h)
        _avatar(ui, g, x + w / 2, y + h / 2, s, n.p.get("person", "Alex Morgan"), n.p.get("kind", "portrait"),
                n.p.get("status"), n.p.get("ring"))


@_spec("avatar_group", w="hug")
class _AvatarGroup(_Spec):
    def size(self, ui, n, aw):
        s = float(n.p.get("size", 32))
        names = n.p.get("names") or []
        k = min(len(names), n.p.get("max", 4)) + (1 if len(names) > n.p.get("max", 4) else 0)
        return (s + (k - 1) * s * 0.68 if k else 0.0, s)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        s = float(n.p.get("size", 32))
        names = n.p.get("names") or []
        mx = n.p.get("max", 4)
        ring = n.p.get("ring") or (T.surface if not T.glass else "#ffffff")
        cx = x + s / 2
        for nm in names[:mx]:
            _avatar(ui, g, cx, y + h / 2, s, nm, n.p.get("kind", "portrait"), ring=ring)
            cx += s * 0.68
        if len(names) > mx:
            g.circle(cx, y + h / 2, s / 2 + max(s * 0.06, 1.5), fill=ring)
            g.circle(cx, y + h / 2, s / 2, fill=T.surface3)
            _txt(ui, g, cx, y + h / 2, f"+{len(names) - mx}", s * 0.34, T.ink2, True, "mc")


@_spec("stat")
class _Stat(_Spec):
    def size(self, ui, n, aw):
        h = 20 + 20 + 8 + 34 + 20
        if n.p.get("delta") is not None or n.p.get("note"):
            h += 26
        if n.p.get("spark"):
            h += 48
        return (aw if n.w != "hug" else min(aw, 200.0), float(h))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        if p.get("card", True):
            ui.panel(g, x, y, w, h)
        px = 18 if w > 150 else 14
        x0, y0, x1 = x + px, y + 20, x + w - px
        ic = p.get("icon")
        if ic:
            col = T.tone(p.get("tone"))
            g.rect(x1 - 32, y0 - 6, 32, 32, r=min(T.r["small"], 16), fill=T.soft(col, 1.3) if not T.wire
                   else T.surface2)
            _ico(ui, g, ic, x1 - 16, y0 + 10, 18, col if not T.wire else T.ink2)
        _txt(ui, g, x0, y0 + 9, p.get("label", ""), 13, T.ink2, maxw=(x1 - x0) - (40 if ic else 0))
        vy = y0 + 28 + 17
        vs = float(p.get("value_size") or (28 if w > 150 else 24))
        _txt(ui, g, x0, vy, p.get("value", ""), vs, T.ink, True, ls=-0.3, maxw=x1 - x0)
        yy = vy + 30
        d = p.get("delta")
        if d is not None:
            ds = str(d)
            good = p.get("good", "up")
            up = not ds.strip().startswith(("-", "−"))
            ok = up == (good == "up")
            col = T.success if ok else T.danger
            pw = _tw(ds, 12, True, T.font) + 30
            g.rect(x0, yy - 10, pw, 20, r=10, fill=T.soft(col, 1.2) if not T.wire else T.surface2)
            _ico(ui, g, "arrow-up-right" if up else "arrow-down", x0 + 11, yy, 12, col if not T.wire else T.ink2,
                 sw=2.6)
            _txt(ui, g, x0 + 21, yy, ds, 12, col if not T.wire else T.ink2, True)
            if p.get("note"):
                _txt(ui, g, x0 + pw + 8, yy, p["note"], 12, T.ink3, maxw=x1 - x0 - pw - 8)
            yy += 26
        elif p.get("note"):
            _txt(ui, g, x0, yy, p["note"], 12, T.ink3, maxw=x1 - x0)
            yy += 26
        sp = p.get("spark")
        if sp:
            _sparkline(ui, g, x0, y + h - 18 - 40, x1 - x0, 40, sp, T.tone(p.get("tone")) if p.get("tone")
                       else T.accent)


def _sparkline(ui, g, x, y, w, h, values, color, fill=True, width=2.0, dot=True):
    v = np.asarray(values, np.float64)
    if v.size < 2:
        return
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo or 1.0
    pts = [(x + w * i / (v.size - 1), y + h - (val - lo) / rng * h * 0.9 - h * 0.05) for i, val in enumerate(v)]
    col = color if not ui.T.wire else ui.T.ink2
    if fill:
        g.path(_smooth_area(pts, y + h), fill=linear_gradient([(0, col, 0.28), (1, col, 0.0)]), stroke=None)
    g.path(_smooth_d(pts), fill=None, stroke=col, stroke_width=width, cap="round", join="round")
    if dot:
        g.circle(pts[-1][0], pts[-1][1], width * 1.8, fill=col, stroke=ui.T.surface if not ui.T.glass else None,
                 stroke_width=1.5)


def _smooth_d(pts):
    """Monotone-ish smooth path through points (Catmull-Rom → cubic Bezier, clamped)."""
    P = [tuple(map(float, p)) for p in pts]
    d = f"M{P[0][0]:.2f} {P[0][1]:.2f}"
    for i in range(len(P) - 1):
        p0 = P[i - 1] if i > 0 else P[i]
        p1, p2 = P[i], P[i + 1]
        p3 = P[i + 2] if i + 2 < len(P) else P[i + 1]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        lo, hi = min(p1[1], p2[1]), max(p1[1], p2[1])
        c1 = (c1[0], min(max(c1[1], lo), hi))
        c2 = (c2[0], min(max(c2[1], lo), hi))
        d += f"C{c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p2[0]:.2f} {p2[1]:.2f}"
    return d


def _smooth_area(pts, base):
    d = _smooth_d(pts)
    return d + f"L{pts[-1][0]:.2f} {base:.2f}L{pts[0][0]:.2f} {base:.2f}Z"


_NUMLIKE = re.compile(r"^[\s$€£¥+\-−]*[\d.,]+\s*[%kKMBx]?\s*$")


@_spec("table")
class _Table(_Spec):
    def dims(self, n):
        rh = 44.0 if n.p.get("compact") else 56.0
        return 40.0, rh

    def size(self, ui, n, aw):
        hh, rh = self.dims(n)
        return (aw, (hh if n.p.get("header", True) else 0.0) + rh * len(n.p.get("rows") or []))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        cols = [str(c) for c in p.get("columns") or []]
        rows = p.get("rows") or []
        k = max(len(cols), max((len(r) for r in rows), default=0), 1)
        hh, rh = self.dims(n)
        if p.get("card", True):
            ui.panel(g, x, y, w, h)
        wts = p.get("widths") or [2.0 if i == 0 else 1.0 for i in range(k)]
        tot = float(sum(wts))
        px = 16.0
        cw = [(w - 2 * px) * wt / tot for wt in wts]
        cx = [x + px + sum(cw[:i]) for i in range(k)]
        aligns = p.get("align") or [None] * k
        aligns = list(aligns) + [None] * (k - len(aligns))
        for j in range(k):
            if aligns[j] is None:
                vals = [r[j] for r in rows if j < len(r)]
                aligns[j] = "right" if vals and all(isinstance(v, (int, float)) or (isinstance(v, str)
                                                    and _NUMLIKE.match(v)) for v in vals) else "left"
        yy = y
        clip = g.group(None, clip=g.rect(x, y, w, h, r=T.r["card"] if p.get("card", True) else 0, add=False))
        if p.get("header", True):
            clip.rect(x, y, w, hh, fill=T.surface2 if not T.glass else "#ffffff10" if T.dark else "#ffffff40")
            for j, c in enumerate(cols):
                a = aligns[j]
                tx = cx[j] + (cw[j] - 8 if a == "right" else 0)
                _txt(ui, clip, tx, y + hh / 2, c.upper(), 11, T.ink2, True, "rc" if a == "right" else "lc", ls=0.5,
                     maxw=cw[j] - 8)
            ui.hline(clip, x, x + w, y + hh - 0.5)
            yy += hh
        sel = p.get("selected")
        for i, r in enumerate(rows):
            if sel == i:
                clip.rect(x, yy, w, rh, fill=T.accent_soft)
            elif p.get("zebra") and i % 2:
                clip.rect(x, yy, w, rh, fill=alpha(T.surface2, 0.6) if not T.glass else "#ffffff0a")
            cy = yy + rh / 2
            for j in range(k):
                v = r[j] if j < len(r) else ""
                self._cell(ui, clip, v, cx[j], cy, cw[j] - 8, aligns[j], j == 0)
            yy += rh
            if i < len(rows) - 1:
                ui.hline(clip, x, x + w, yy - 0.5)

    def _cell(self, ui, g, v, x, cy, w, align, first):
        T = ui.T
        if isinstance(v, dict):
            if "badge" in v:
                if align == "right":
                    _pill(ui, g, x + w, cy, v["badge"], v.get("tone", "neutral"), anchor="r")
                else:
                    _pill(ui, g, x, cy, v["badge"], v.get("tone", "neutral"))
                return
            if "avatar" in v:
                nm = v.get("avatar")
                _avatar(ui, g, x + 16, cy, 32, nm, v.get("kind", "portrait"))
                if v.get("sub"):
                    _txt(ui, g, x + 42, cy - 8, v.get("text", nm), 14, T.ink, True, maxw=w - 42)
                    _txt(ui, g, x + 42, cy + 10, v["sub"], 12, T.ink3, maxw=w - 42)
                else:
                    _txt(ui, g, x + 42, cy, v.get("text", nm), 14, T.ink, True, maxw=w - 42)
                return
            if "progress" in v:
                pv = min(max(float(v["progress"]), 0), 1)
                bw = min(w - 44, 120)
                g.rect(x, cy - 3, bw, 6, r=3, fill=T.surface3)
                g.rect(x, cy - 3, max(bw * pv, 6), 6, r=3, fill=T.tone(v.get("tone")))
                _txt(ui, g, x + bw + 8, cy, f"{int(round(pv * 100))}%", 12, T.ink2)
                return
            if "icon" in v:
                _ico(ui, g, v["icon"], x + 10, cy, 18, T.tone(v.get("tone")) if v.get("tone") else T.ink2)
                _txt(ui, g, x + 26, cy, v.get("text", ""), 14, T.ink, maxw=w - 26)
                return
            v = v.get("text", "")
        if v is True or v is False:
            _ico(ui, g, "tick" if v else "minus", x + (w - 10 if align == "right" else 10), cy, 18,
                 T.success if v else T.ink3, sw=2.2)
            return
        s = str(v)
        col = T.ink if first else T.ink2
        if isinstance(v, str) and s.startswith(("+", "-", "−")) and _NUMLIKE.match(s):
            col = T.success if s.startswith("+") else T.danger
        if T.wire and not first and not _NUMLIKE.match(s):
            bw = min(_tw(s, 14, False, T.font), w)
            _bar(ui, g, x + (w - bw if align == "right" else 0), cy, bw, 14)
            return
        _txt(ui, g, x + (w if align == "right" else 0), cy, s, 14, col, False, "rc" if align == "right" else "lc",
             maxw=w)


@_spec("progress")
class _Progress(_Spec):
    def size(self, ui, n, aw):
        t = float(n.p.get("thickness", 8))
        return (aw if n.w != "hug" else min(aw, 200.0), t + (26.0 if n.p.get("label") else 0.0))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        v = min(max(float(n.p.get("value", 0.5)), 0.0), 1.0)
        t = float(n.p.get("thickness", 8))
        yy = y
        if n.p.get("label"):
            _txt(ui, g, x, y + 9, n.p["label"], 13, T.ink, True, maxw=w * 0.7)
            vt = n.p.get("value_text")
            vt = f"{int(round(v * 100))}%" if vt is None else vt
            _txt(ui, g, x + w, y + 9, vt, 13, T.ink2, anchor="rc")
            yy = y + 26
        col = T.tone(n.p.get("tone"))
        g.rect(x, yy, w, t, r=t / 2, fill=T.surface3 if not T.glass else T.border)
        if v > 0:
            g.rect(x, yy, max(w * v, t), t, r=t / 2, fill=col if not T.wire else T.ink2)


@_spec("ring", w="hug")
class _Ring(_Spec):
    def size(self, ui, n, aw):
        s = float(n.p.get("size", 96))
        return (s, s)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        s = min(w, h)
        cx, cy = x + w / 2, y + h / 2
        th = float(n.p.get("thickness", max(s * 0.1, 6)))
        r = s / 2 - th / 2
        v = min(max(float(n.p.get("value", 0.5)), 0.0), 1.0)
        col = T.tone(n.p.get("tone")) if not T.wire else T.ink2
        if T.dark and not T.wire and s >= 140:       # soft inner glow: large rings read as a hero element
            g.circle(cx, cy, r, fill=radial_gradient([(0, col, 0.0), (0.7, col, 0.05), (1, col, 0.22)]))
        g.circle(cx, cy, r, fill=None, stroke=T.surface3 if not T.glass else T.border, stroke_width=th)
        glow = T.dark and not T.wire and s >= 140
        if v >= 0.999:
            g.circle(cx, cy, r, fill=None, stroke=col, stroke_width=th)
        elif v > 0:
            if glow:
                for k, a in ((2.1, 0.07), (1.65, 0.1), (1.3, 0.16)):
                    g.arc(cx, cy, r, -90, -90 + 360 * v, fill=None, stroke=col, stroke_width=th * k,
                          stroke_opacity=a, cap="round")
            g.arc(cx, cy, r, -90, -90 + 360 * v, fill=None, stroke=col, stroke_width=th, cap="round")
        lab = n.p.get("label")
        if lab is None:
            lab = f"{int(round(v * 100))}%"
        sub = n.p.get("sub")
        fs = s * (0.24 if not sub else 0.22)
        _txt(ui, g, cx, cy - (s * 0.07 if sub else 0), lab, fs, T.ink, True, "mc", maxw=s - 2 * th - 8)
        if sub:
            _txt(ui, g, cx, cy + s * 0.15, sub, max(s * 0.11, 10), T.ink2, False, "mc", maxw=s - 2 * th - 8)


@_spec("skeleton")
class _Skeleton(_Spec):
    def size(self, ui, n, aw):
        k = int(n.p.get("lines", 3))
        h = (n.p.get("image") or 0) + (16 if n.p.get("image") else 0) + max(k * 22 - 8, 0)
        if n.p.get("avatar"):
            h = max(h, 44)
        return (aw, float(h))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        c = T.placeholder if not T.glass else T.surface3
        c2 = mix(c, T.surface if not T.glass else "#ffffff", 0.35) if not T.glass else c
        yy = y
        if n.p.get("image"):
            ih = float(n.p["image"])
            g.rect(x, y, w, ih, r=T.r["small"] + 4, fill=linear_gradient([(0, c), (0.5, c2), (1, c)], start=(0, 0),
                                                                          end=(1, 0)))
            yy += ih + 16
        tx = x
        if n.p.get("avatar"):
            g.circle(x + 22, yy + 22, 22, fill=c)
            tx = x + 56
        for i in range(int(n.p.get("lines", 3))):
            frac = [0.62, 0.92, 0.78, 0.86, 0.54][i % 5] if i else 0.48
            g.rect(tx, yy + i * 22, (x + w - tx) * frac, 12, r=6, fill=c if i else mix(c, T.ink3, 0.18))


@_spec("tooltip", w="hug")
class _Tooltip(_Spec):
    def size(self, ui, n, aw):
        lines = wrap(str(n.p.get("text", "")), 13, min(aw, 240) - 24, False, 3, ui.T.font)
        W = max(_tw(l, 13, False, ui.T.font) for l in lines) + 24
        return (min(aw, W), 18.0 * len(lines) + 16 + 6)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        bg = "#1f2430" if not T.dark else "#f2f4f8"
        fg = "#ffffff" if not T.dark else "#101828"
        side = n.p.get("arrow", "bottom")
        bh = h - 6
        by = y if side != "top" else y + 6
        with g.group(None, shadow=T.shadow_lg if not T.wire else None) as s_:
            s_.rect(x, by, w, bh, r=8, fill=bg)
            ax = x + w * float(n.p.get("arrow_at", 0.5))
            if side == "bottom":
                s_.polygon([(ax - 6, by + bh - 0.5), (ax + 6, by + bh - 0.5), (ax, by + bh + 6)], fill=bg)
            elif side == "top":
                s_.polygon([(ax - 6, by + 0.5), (ax + 6, by + 0.5), (ax, by - 6)], fill=bg)
        lines = wrap(str(n.p.get("text", "")), 13, w - 24, False, 3, T.font)
        for i, ln in enumerate(lines):
            _txt(ui, g, x + 12, by + 8 + 9 + i * 18, ln, 13, fg)


@_spec("toast")
class _Toast(_Spec):
    def size(self, ui, n, aw):
        return (min(aw, 420.0), 56.0 if not n.p.get("sub") else 68.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        dark_bg = "#1b1f29" if not T.dark else "#2a303c"
        if T.glass:
            ui.panel(g, x, y, w, h, r=min(T.r["card"], 20), kind="raised")
            fg, fg2 = T.ink, T.ink2
        else:
            with g.group(None, shadow=T.shadow_lg) as s_:
                s_.rect(x, y, w, h, r=14, fill=dark_bg)
            fg, fg2 = "#ffffff", "#b7bdc9"
        col = T.tone(n.p.get("tone", "success"))
        cy = y + h / 2
        ic = n.p.get("icon", "check")
        tx = x + 16
        if ic:
            _ico(ui, g, ic, x + 28, cy, 22, col if not T.wire else fg)
            tx = x + 50
        right = x + w - 16
        act = n.p.get("action")
        if act:
            right -= _txt(ui, g, right, cy, act, 14, mix(T.accent, "#ffffff", 0.35) if not T.wire else fg, True,
                          "rc") + 14
        if n.p.get("sub"):
            _txt(ui, g, tx, cy - 9, n.p.get("text", ""), 14, fg, True, maxw=right - tx)
            _txt(ui, g, tx, cy + 10, n.p["sub"], 12, fg2, maxw=right - tx)
        else:
            _txt(ui, g, tx, cy, n.p.get("text", ""), 14, fg, True, maxw=right - tx)


@_spec("banner")
class _Banner(_Spec):
    def _lines(self, ui, n, w):
        tw = w - 16 - 36 - (36 if n.p.get("close") else 16)
        tl = wrap(str(n.p.get("title") or ""), 14, tw, True, 2, ui.T.font) if n.p.get("title") else ()
        bl = wrap(str(n.p.get("text") or ""), 13, tw, False, 3, ui.T.font) if n.p.get("text") else ()
        return tl, bl

    def size(self, ui, n, aw):
        tl, bl = self._lines(ui, n, aw)
        h = 28 + len(tl) * 20 + len(bl) * 19 + (30 if n.p.get("action") else 0)
        return (aw, float(max(h, 52)))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        col = T.tone(n.p.get("tone", "info"))
        g.rect(x, y, w, h, r=min(T.r["control"], 14), fill=T.soft(col, 1.1) if not T.wire else T.surface2,
               stroke=alpha(col, 0.35) if not T.wire else T.border_strong, stroke_width=ui.hl)
        ic = n.p.get("icon") or {"success": "check", "warning": "alert", "danger": "alert"}.get(n.p.get("tone"), "info")
        _ico(ui, g, ic, x + 26, y + 24, 20, col if not T.wire else T.ink2)
        tl, bl = self._lines(ui, n, w)
        yy = y + 24
        for ln in tl:
            _txt(ui, g, x + 48, yy, ln, 14, T.ink, True)
            yy += 20
        if tl:
            yy += 1
        for ln in bl:
            if T.wire:
                _bar(ui, g, x + 48, yy, _tw(ln, 13, False, T.font), 13)
            else:
                _txt(ui, g, x + 48, yy, ln, 13, T.ink2)
            yy += 19
        if n.p.get("action"):
            _txt(ui, g, x + 48, yy + 8, n.p["action"], 13, col if not T.wire else T.ink, True)
        if n.p.get("close"):
            _ico(ui, g, "close", x + w - 22, y + 24, 16, T.ink2)


@_spec("menu", w=232)
class _Menu(_Spec):
    def size(self, ui, n, aw):
        h = 12.0
        for it in n.p.get("items") or []:
            h += 9.0 if it in ("-", None) else 40.0
        return (aw, h)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        ui.panel(g, x, y, w, h, r=min(T.r["card"], 14), kind="raised")
        yy = y + 6
        act = n.p.get("active")
        for i, it in enumerate(n.p.get("items") or []):
            if it in ("-", None):
                ui.hline(g, x + 8, x + w - 8, yy + 4.5)
                yy += 9
                continue
            if isinstance(it, dict):
                ic, lab, sc, danger = it.get("icon"), it.get("label", ""), it.get("shortcut"), it.get("danger")
            else:
                it = list(it) if isinstance(it, (tuple, list)) else [None, it]
                ic, lab = it[0], it[1] if len(it) > 1 else ""
                sc = it[2] if len(it) > 2 else None
                danger = False
            if act == i or act == lab:
                g.rect(x + 6, yy, w - 12, 40, r=min(T.r["small"], 10), fill=T.surface2 if not T.glass else T.surface3)
            col = T.danger if danger else T.ink
            tx = x + 18
            if ic:
                _ico(ui, g, ic, x + 28, yy + 20, 18, T.danger if danger else T.ink2)
                tx = x + 48
            right = x + w - 16
            if sc:
                right -= _txt(ui, g, right, yy + 20, sc, 12, T.ink3, anchor="rc") + 8
            _txt(ui, g, tx, yy + 20, lab, 14, col, maxw=right - tx)
            yy += 40


# ---- conversation, feeds, calendar, boards


@_spec("message")
class _Message(_Spec):
    def _layout(self, ui, n, aw):
        mw = min(aw * 0.78, 520.0) - (40 if (n.p.get("avatar") and not n.p.get("me")) else 0)
        lines = wrap(str(n.p.get("text", "")), 15, mw - 28, False, None, ui.T.font)
        bw = max(_tw(l, 15, False, ui.T.font) for l in lines) + 28
        bh = len(lines) * 21 + 20
        return lines, bw, bh

    def size(self, ui, n, aw):
        lines, bw, bh = self._layout(ui, n, aw)
        return (aw, bh + (20.0 if n.p.get("time") else 0.0))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        me = bool(n.p.get("me"))
        lines, bw, bh = self._layout(ui, n, w)
        av = n.p.get("avatar") and not me
        bx = x + w - bw if me else x + (40 if av else 0)
        R = 18.0
        rr = (R, R, 6, R) if me else (R, R, R, 6)
        if T.wire:
            fill, fg = (T.surface3 if me else T.surface), T.ink
        elif me:
            fill, fg = T.accent, T.on_accent
        else:
            fill = T.surface if (not T.dark and T.style == "modern") else (T.surface2 if not T.glass else T.surface)
            fg = T.ink
        if T.glass and not me:
            ui.panel(g, bx, y, bw, bh, r=rr, kind="card", shadow=False)
        else:
            g.rect(bx, y, bw, bh, r=rr, fill=fill, stroke=(T.border if (not me and not T.dark and not T.wire)
                                                           else (T.border_strong if T.wire else None)),
                   stroke_width=ui.hl)
        if av:
            _avatar(ui, g, x + 15, y + bh - 15, 30, n.p["avatar"], "portrait")
        for i, ln in enumerate(lines):
            if T.wire:
                _bar(ui, g, bx + 14, y + 10 + 10.5 + i * 21, _tw(ln, 15, False, T.font), 15)
            else:
                _txt(ui, g, bx + 14, y + 10 + 10.5 + i * 21, ln, 15, fg)
        t = n.p.get("time")
        if t:
            ty = y + bh + 11
            if me:
                st = n.p.get("status", "read")
                tw = _txt(ui, g, x + w - (20 if st else 0), ty, t, 11, T.ink3, anchor="rc")
                if st:
                    col = T.accent_ink if st == "read" else T.ink3
                    g.path(f"M{x + w - 15} {ty}l2.5 2.5 4.5-5" + (f"M{x + w - 10.5} {ty + 2.5}l4.5-5" if st == "read"
                                                                 else ""), fill=None, stroke=col, stroke_width=1.5,
                           cap="round", join="round")
                _ = tw
            else:
                _txt(ui, g, bx + 4, ty, t, 11, T.ink3)


@_spec("notification")
class _Notification(_Spec):
    def size(self, ui, n, aw):
        lines = wrap(str(n.p.get("text", "")), 13, aw - 56 - 24, False, 2, ui.T.font) if n.p.get("text") else ()
        return (aw, float(max(64, 24 + 20 + len(lines) * 18 + 4)))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        if p.get("unread"):
            g.circle(x + w - 6, y + 22, 4.5, fill=T.accent)
        lead = p.get("avatar")
        if lead:
            _avatar(ui, g, x + 22, y + 32, 44, lead, "portrait")
        else:
            col = T.tone(p.get("tone"))
            g.circle(x + 22, y + 32, 22, fill=T.soft(col, 1.3) if not T.wire else T.surface2)
            _ico(ui, g, p.get("icon", "bell"), x + 22, y + 32, 20, col if not T.wire else T.ink2)
        tx = x + 56
        right = x + w - (18 if p.get("unread") else 0)
        tm = p.get("time")
        tmw = _txt(ui, g, right, y + 21, tm, 12, T.ink3, anchor="rc") if tm else 0
        _txt(ui, g, tx, y + 21, p.get("title", ""), 14, T.ink, True, maxw=right - tx - tmw - 10)
        if p.get("text"):
            lines = wrap(str(p["text"]), 13, w - 56 - 24, False, 2, T.font)
            for i, ln in enumerate(lines):
                if T.wire:
                    _bar(ui, g, tx, y + 41 + i * 18, _tw(ln, 13, False, T.font), 13)
                else:
                    _txt(ui, g, tx, y + 41 + i * 18, ln, 13, T.ink2)


@_spec("calendar")
class _Calendar(_Spec):
    def _month(self, n):
        import calendar as _cal
        yv, mv = int(n.p.get("year", 2026)), int(n.p.get("month", 11))
        first = int(n.p.get("first_weekday", 0))
        c = _cal.Calendar(firstweekday=first)
        weeks = c.monthdayscalendar(yv, mv)
        return yv, mv, weeks, first

    def size(self, ui, n, aw):
        _, _, weeks, _ = self._month(n)
        cell = min(max(aw / 7.0 * 0.82, 30.0), 48.0)
        head = 44.0 if n.p.get("header", True) else 0.0
        return (aw, head + 28.0 + len(weeks) * cell)

    def paint(self, ui, g, n, x, y, w, h):
        import calendar as _cal
        T = ui.T
        yv, mv, weeks, first = self._month(n)
        cw = w / 7.0
        cell = min(max(cw * 0.82, 30.0), 48.0)
        yy = y
        if n.p.get("header", True):
            _txt(ui, g, x + 4, y + 18, f"{_cal.month_name[mv]} {yv}", 17, T.ink, True)
            _ico(ui, g, "chevron-left", x + w - 52, y + 18, 18, T.ink2)
            _ico(ui, g, "chevron-right", x + w - 16, y + 18, 18, T.ink2)
            yy += 44
        names = [_cal.day_abbr[(first + i) % 7][:2] if w < 420 else _cal.day_abbr[(first + i) % 7] for i in range(7)]
        for i, nm in enumerate(names):
            _txt(ui, g, x + cw * (i + 0.5), yy + 10, nm.upper() if w >= 420 else nm, 11, T.ink3, True, "mc")
        yy += 28
        sel = n.p.get("selected")
        rng = n.p.get("range")
        today = n.p.get("today")
        events = n.p.get("events") or {}
        rr = cell * 0.42
        for wi, week in enumerate(weeks):
            cy = yy + wi * cell + cell / 2
            if rng:
                a, b = rng
                ds = [i for i, d in enumerate(week) if d and a <= d <= b]
                if ds:
                    i0, i1 = ds[0], ds[-1]
                    g.rect(x + cw * (i0 + 0.5) - rr, cy - rr, cw * (i1 - i0) + 2 * rr, 2 * rr, r=rr,
                           fill=T.accent_soft)
            for i, d in enumerate(week):
                if not d:
                    continue
                cx = x + cw * (i + 0.5)
                on = d == sel or (rng and d in rng)
                if on:
                    g.circle(cx, cy, rr, fill=T.accent)
                elif d == today:
                    g.circle(cx, cy, rr, fill=None, stroke=T.accent, stroke_width=1.5)
                wk = (first + i) % 7 >= 5
                col = T.on_accent if on else (T.ink3 if (wk and n.p.get("dim_weekends", True)) else T.ink)
                _txt(ui, g, cx, cy - (2 if d in events else 0), str(d), 14, col, on or d == today, "mc")
                ev = events.get(d)
                if ev:
                    cols = ev if isinstance(ev, (list, tuple)) else [T.accent] * min(int(ev) if _num(ev) else 1, 3)
                    k = len(cols)
                    for j, c in enumerate(cols[:3]):
                        g.circle(cx + (j - (k - 1) / 2) * 6, cy + cell * 0.27, 2.2,
                                 fill=(T.on_accent if on else T.tone(c)))


@_spec("kcard")
class _KCard(_Spec):
    def size(self, ui, n, aw):
        lines = wrap(str(n.p.get("title", "")), 14, aw - 28, True, 3, ui.T.font)
        h = 14 + (30 if n.p.get("tag") else 0) + len(lines) * 20 + 14 + 28 + 14
        if n.p.get("progress") is not None:
            h += 18
        if n.p.get("image"):
            h += 96
        return (aw, float(h))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        p = n.p
        ui.panel(g, x, y, w, h, r=min(T.r["card"], 14))
        yy = y + 14
        if p.get("image"):
            ui.image_art(g, x + 8, y + 8, w - 16, 88, seed=_hash(p.get("title")), r=min(T.r["card"], 14) - 4)
            yy += 96 - 6
        if p.get("tag"):
            tg = [p["tag"]] if isinstance(p["tag"], str) else list(p["tag"])
            tones = ["accent", "success", "warning", "info", "danger"]
            reg = ui.__dict__.setdefault("_tag_tones", {})
            tx = x + 14
            for i, tt in enumerate(tg[:3]):
                tone_i = p.get("tone", "accent") if i == 0 and isinstance(p["tag"], str) and p.get("tone") != "accent"                     else reg.setdefault(tt, tones[len(reg) % 5])
                tx += _pill(ui, g, tx, yy + 10, tt, tone_i) + 6
            yy += 30
        for ln in wrap(str(p.get("title", "")), 14, w - 28, True, 3, T.font):
            _txt(ui, g, x + 14, yy + 10, ln, 14, T.ink, True)
            yy += 20
        if p.get("progress") is not None:
            pv = min(max(float(p["progress"]), 0), 1)
            g.rect(x + 14, yy + 10, w - 28, 4, r=2, fill=T.surface3)
            g.rect(x + 14, yy + 10, max((w - 28) * pv, 4), 4, r=2, fill=T.tone(p.get("tone")))
            yy += 18
        fy = y + h - 14 - 14
        names = p.get("people") or []
        if names:
            ax = x + 14 + 14
            for nm in names[:3]:
                _avatar(ui, g, ax, fy, 28, nm, "portrait", ring=T.surface if not T.glass else None)
                ax += 18
        right = x + w - 14
        if p.get("comments"):
            tw = _txt(ui, g, right, fy, str(p["comments"]), 12, T.ink3, anchor="rc")
            _ico(ui, g, "message", right - tw - 10, fy, 14, T.ink3)
            right -= tw + 28
        if p.get("due"):
            tw = _txt(ui, g, right, fy, p["due"], 12, T.ink3, anchor="rc")
            _ico(ui, g, "clock", right - tw - 10, fy, 14, T.ink3)


@_spec("timeline")
class _Timeline(_Spec):
    def _items(self, n):
        out = []
        for it in n.p.get("items") or []:
            if isinstance(it, dict):
                out.append(it)
            else:
                it = list(it)
                out.append(dict(time=it[0] if len(it) > 0 else "", title=it[1] if len(it) > 1 else "",
                                text=it[2] if len(it) > 2 else None, icon=it[3] if len(it) > 3 else None))
        return out

    def _heights(self, ui, n, aw):
        hs = []
        for it in self._items(n):
            tl = wrap(str(it.get("text") or ""), 13, aw - 40, False, 3, ui.T.font) if it.get("text") else ()
            hs.append((tl, 22 + 6 + len(tl) * 19 + 20))
        return hs

    def size(self, ui, n, aw):
        return (aw, float(sum(h for _, h in self._heights(ui, n, aw))))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        its = self._items(n)
        hs = self._heights(ui, n, w)
        yy = y
        for i, (it, (tl, hh)) in enumerate(zip(its, hs)):
            cx = x + 12
            if i < len(its) - 1:
                g.line(cx, yy + 26, cx, yy + hh + 2, stroke=T.border_strong if not T.glass else T.border,
                       stroke_width=1.5)
            col = T.tone(it.get("tone")) if it.get("tone") else (T.accent if i == 0 else T.ink3)
            if it.get("icon"):
                g.circle(cx, yy + 11, 12, fill=T.soft(col, 1.3) if not T.wire else T.surface2)
                _ico(ui, g, it["icon"], cx, yy + 11, 14, col if not T.wire else T.ink2, sw=2.2)
            else:
                g.circle(cx, yy + 11, 6, fill=T.surface if not T.glass else T.bg, stroke=col, stroke_width=2.5)
            tm = it.get("time")
            tmw = _txt(ui, g, x + w, yy + 11, tm, 12, T.ink3, anchor="rc") if tm else 0
            _txt(ui, g, x + 36, yy + 11, it.get("title", ""), 14, T.ink, True, maxw=w - 36 - tmw - 10)
            for j, ln in enumerate(tl):
                if T.wire:
                    _bar(ui, g, x + 36, yy + 34 + j * 19, _tw(ln, 13, False, T.font), 13)
                else:
                    _txt(ui, g, x + 36, yy + 34 + j * 19, ln, 13, T.ink2)
            yy += hh


@_spec("logo", w="hug")
class _Logo(_Spec):
    def size(self, ui, n, aw):
        return (min(aw, _tw(str(n.p.get("text", "")), 18, True, ui.T.font) + 34), 32.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        col = n.p.get("color") or T.ink3
        _logo_mark(ui, g, x, y + h / 2 - 12, 24, int(n.p.get("mark", _hash(n.p.get("text")) % 6)), col)
        _txt(ui, g, x + 32, y + h / 2, n.p.get("text", ""), 18, col, True, maxw=w - 32, ls=-0.2)


# ---- media: images, maps, video, audio, code, terminal, charts, embedded screens

_SCAPES = {   # sky top, sky bottom, sun, far, mid, near
    "dawn": ("#9db8f5", "#ffd8c8", "#fff6ea", "#c5b8ea", "#8f87cc", "#5a579f"),
    "ocean": ("#79cfe4", "#e8f8f4", "#ffffff", "#9ccadb", "#4ea2b7", "#226f80"),
    "forest": ("#bfe3c4", "#f6f2d8", "#fff8d6", "#a3cd9f", "#5d9e6b", "#2e6a4a"),
    "dusk": ("#2c2f72", "#f2907c", "#ffd9a8", "#7b5192", "#4b3572", "#241b47"),
    "desert": ("#f5c29b", "#fde9d2", "#fff3df", "#e9aa7f", "#c97c5b", "#8b4d3b"),
    "lilac": ("#d9c7ff", "#ffe1ef", "#fff7fb", "#c2a8f0", "#9b7fd9", "#6b55b5"),
}


def _ridge(rng, x, y, w, h, base, amp, n=6):
    xs = np.linspace(x - w * 0.05, x + w * 1.05, n)
    ys = base - amp * (0.35 + 0.65 * rng.random(n))
    pts = list(zip(xs, ys))
    return _smooth_d(pts) + f"L{x + w * 1.05:.2f} {y + h + 2:.2f}L{x - w * 0.05:.2f} {y + h + 2:.2f}Z"


def _landscape(ui, g, x, y, w, h, seed=1, r=12, kind=None):
    """Abstract landscape placeholder (soft gradient sky, sun, three ridges)."""
    T = ui.T
    if T.wire:
        g.rect(x, y, w, h, r=r, fill=T.surface2, stroke=T.border_strong, stroke_width=ui.hl)
        with g.group(None, clip=g.rect(x, y, w, h, r=r, add=False)) as c:
            c.line(x, y, x + w, y + h, stroke=T.border_strong, stroke_width=ui.hl)
            c.line(x + w, y, x, y + h, stroke=T.border_strong, stroke_width=ui.hl)
        s = min(w, h) * 0.22
        if s >= 12:
            g.rect(x + w / 2 - s * 0.75, y + h / 2 - s * 0.6, s * 1.5, s * 1.2, r=4, fill=T.surface2)
            _ico(ui, g, "image", x + w / 2, y + h / 2, s, T.ink3)
        return
    names = list(_SCAPES)
    kind = kind if kind in _SCAPES else names[int(seed) % len(names)]
    s0, s1, sun, far, mid, near = _SCAPES[kind]
    if T.dark and kind not in ("dusk",):
        s0, s1 = mix(s0, "#1a1d3a", 0.35), mix(s1, "#1a1d3a", 0.25)
        far, mid, near = mix(far, "#1a1d3a", 0.25), mix(mid, "#1a1d3a", 0.3), mix(near, "#0d0f1f", 0.35)
    rng = np.random.default_rng(int(seed) * 7919 + 13)
    with g.group(None, clip=g.rect(x, y, w, h, r=r, add=False)) as c:
        c.rect(x, y, w, h, fill=linear_gradient([s0, s1]))
        sx, sy = x + w * (0.25 + 0.5 * rng.random()), y + h * (0.28 + 0.12 * rng.random())
        sr = min(w, h) * 0.11
        c.circle(sx, sy, sr * 3.2, fill=radial_gradient([(0, sun, 0.55), (1, sun, 0.0)]))
        c.circle(sx, sy, sr, fill=sun)
        c.path(_ridge(rng, x, y, w, h, y + h * 0.62, h * 0.26, 6), fill=far, stroke=None)
        c.path(_ridge(rng, x, y, w, h, y + h * 0.76, h * 0.22, 5), fill=mid, stroke=None)
        c.path(_ridge(rng, x, y, w, h, y + h * 0.92, h * 0.16, 4), fill=near, stroke=None)


def _map_art(ui, g, x, y, w, h, seed=3, r=12, pin=True, route=True):
    T = ui.T
    if T.dark:
        land, water, park, road, major, casing = "#1b2130", "#1c3354", "#1d3a30", "#2c3548", "#465572", "#151a26"
    else:
        land, water, park, road, major, casing = "#eef0ea", "#a9d3f3", "#cde8c4", "#ffffff", "#ffe08a", "#dcdfd6"
    if T.wire:
        land, water, park, road, major, casing = T.surface2, T.surface3, T.surface3, T.surface, T.border_strong, \
            T.border
    rng = np.random.default_rng(int(seed) * 31 + 5)
    with g.group(None, clip=g.rect(x, y, w, h, r=r, add=False)) as c:
        c.rect(x, y, w, h, fill=land)
        wy = y + h * (0.15 + 0.2 * rng.random())
        pts = [(x - 20, wy), (x + w * 0.3, wy + h * 0.08), (x + w * 0.55, wy - h * 0.06), (x + w * 0.8, wy + h * 0.1),
               (x + w + 20, wy)]
        top = [(px, py - h * 0.09) for px, py in pts]
        bot = [(px, py + h * 0.07) for px, py in reversed(pts)]
        c.path(_smooth_d(top) + "L" + _smooth_d(bot)[1:] + "Z", fill=water, stroke=None)
        for _ in range(2):
            px, py = x + w * (0.15 + 0.7 * rng.random()), y + h * (0.55 + 0.3 * rng.random())
            c.rect(px - w * 0.09, py - h * 0.06, w * 0.18, h * 0.12, r=min(w, h) * 0.04, fill=park)
        ang = math.radians(8 + 10 * rng.random())
        step = max(min(w, h) / 5.5, 28)
        L = max(w, h) * 1.6
        cx, cy = x + w / 2, y + h / 2
        for k in range(-8, 9):
            for a in (ang, ang + math.pi / 2):
                ox, oy = -math.sin(a) * k * step, math.cos(a) * k * step
                x0, y0 = cx + ox - math.cos(a) * L / 2, cy + oy - math.sin(a) * L / 2
                x1, y1 = cx + ox + math.cos(a) * L / 2, cy + oy + math.sin(a) * L / 2
                c.line(x0, y0, x1, y1, stroke=casing, stroke_width=6)
                c.line(x0, y0, x1, y1, stroke=road, stroke_width=4)
        mj = [(x - 10, y + h * 0.78), (x + w * 0.35, y + h * 0.66), (x + w * 0.7, y + h * 0.72), (x + w + 10,
                                                                                               y + h * 0.5)]
        c.path(_smooth_d(mj), fill=None, stroke=casing, stroke_width=10, cap="round")
        c.path(_smooth_d(mj), fill=None, stroke=major, stroke_width=7, cap="round")
        mj2 = [(x + w * 0.62, y - 10), (x + w * 0.58, y + h * 0.4), (x + w * 0.66, y + h + 10)]
        c.path(_smooth_d(mj2), fill=None, stroke=casing, stroke_width=10, cap="round")
        c.path(_smooth_d(mj2), fill=None, stroke=major, stroke_width=7, cap="round")
        a = (x + w * 0.2, y + h * 0.84)
        b = (x + w * 0.66, y + h * 0.36)
        if route:
            rp = [a, (x + w * 0.36, y + h * 0.68), (x + w * 0.59, y + h * 0.62), (x + w * 0.6, y + h * 0.45), b]
            c.path(_smooth_d(rp), fill=None, stroke="#ffffff" if not T.dark else "#0b0e16", stroke_width=8,
                   cap="round", join="round")
            c.path(_smooth_d(rp), fill=None, stroke=T.accent, stroke_width=5, cap="round", join="round")
            c.circle(a[0], a[1], 7, fill="#ffffff", stroke=T.accent, stroke_width=3)
        if pin:
            px, py = b
            c.ellipse(px, py + 2, 9, 3.5, fill="#000000", opacity=0.2)
            c.path(f"M{px} {py}c-3.5-6-13-11-13-19a13 13 0 0 1 26 0c0 8-9.5 13-13 19Z", fill=T.accent if not T.wire
                   else T.ink2, stroke="#ffffff", stroke_width=2)
            c.circle(px, py - 19, 4.8, fill="#ffffff")


@_spec("image")
class _Image(_Spec):
    def size(self, ui, n, aw):
        if n.p.get("aspect"):
            return (aw, aw / float(n.p["aspect"]))
        src = n.p.get("canvas")
        if src is not None and n.h == "auto" and not n.p.get("height"):
            return (aw, aw * src.h / max(src.w, 1))
        return (aw, float(n.p.get("height", 180)))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        r = n.p.get("radius")
        r = T.r["card"] if r is None else r
        r = min(r, w / 2, h / 2)
        src = n.p.get("canvas")
        if src is not None and not T.wire:
            g.rect(x, y, w, h, r=r, fill=T.surface2)
            ui._hook(g, "image", (x, y, w, h), r, canvas=src, fit=n.p.get("fit", "cover"))
        else:
            _landscape(ui, g, x, y, w, h, n.p.get("seed", 1), r, n.p.get("scene"))
        cap = n.p.get("caption")
        if cap:
            with g.group(None, clip=g.rect(x, y, w, h, r=r, add=False)) as c:
                c.rect(x, y + h * 0.5, w, h * 0.5, fill=linear_gradient([(0, "#000000", 0.0), (1, "#000000", 0.55)]))
            sub = n.p.get("sub")
            _txt(ui, g, x + 16, y + h - (38 if sub else 20), cap, 17, "#ffffff", True, maxw=w - 32)
            if sub:
                _txt(ui, g, x + 16, y + h - 18, sub, 13, "#e8eaf0", maxw=w - 32)
        if n.p.get("badge"):
            _pill(ui, g, x + 12, y + 22, n.p["badge"], "accent", solid=True)
        if n.p.get("icon"):
            g.circle(x + w - 26, y + 26, 16, fill="#ffffffd9" if not T.dark else "#00000080")
            _ico(ui, g, n.p["icon"], x + w - 26, y + 26, 16, "#101828" if not T.dark else "#ffffff")


@_spec("map")
class _Map(_Spec):
    def size(self, ui, n, aw):
        return (aw, float(n.p.get("height", 200)))

    def paint(self, ui, g, n, x, y, w, h):
        r = n.p.get("radius")
        r = ui.T.r["card"] if r is None else r
        _map_art(ui, g, x, y, w, h, n.p.get("seed", 3), min(r, w / 2, h / 2), n.p.get("pin", True),
                 n.p.get("route", True))


@_spec("video")
class _Video(_Spec):
    def size(self, ui, n, aw):
        return (aw, float(n.p.get("height") or aw * 9 / 16))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        r = n.p.get("radius")
        r = min(T.r["card"] if r is None else r, w / 2, h / 2)
        src = n.p.get("canvas")
        if src is not None and not T.wire:
            g.rect(x, y, w, h, r=r, fill="#000000")
            ui._hook(g, "image", (x, y, w, h), r, canvas=src, fit="cover")
        else:
            _landscape(ui, g, x, y, w, h, n.p.get("seed", 4), r, n.p.get("scene", "dusk"))
        with g.group(None, clip=g.rect(x, y, w, h, r=r, add=False)) as c:
            c.rect(x, y + h * 0.55, w, h * 0.45, fill=linear_gradient([(0, "#000000", 0.0), (1, "#000000", 0.6)]))
        pr = min(w, h) * 0.12
        g.circle(x + w / 2, y + h / 2, pr, fill="#ffffff40", stroke="#ffffff99", stroke_width=1.5)
        _ico(ui, g, "play-fill", x + w / 2 + pr * 0.08, y + h / 2, pr * 0.95, "#ffffff")
        if n.p.get("title"):
            _txt(ui, g, x + 16, y + 22, n.p["title"], 15, "#ffffff", True, maxw=w - 32)
        v = min(max(float(n.p.get("progress", 0.32)), 0), 1)
        by = y + h - 16
        x0, x1 = x + 16, x + w - 16
        tl = n.p.get("time", "1:24")
        dur = n.p.get("duration", "4:05")
        if w > 240:
            t0 = _txt(ui, g, x0, by, tl, 12, "#ffffff", True)
            t1 = _txt(ui, g, x1, by, dur, 12, "#ffffffcc", anchor="rc")
            bx0, bx1 = x0 + t0 + 12, x1 - t1 - 12
        else:
            bx0, bx1 = x0, x1
        g.rect(bx0, by - 2, bx1 - bx0, 4, r=2, fill="#ffffff4d")
        g.rect(bx0, by - 2, max((bx1 - bx0) * v, 4), 4, r=2, fill=T.accent if not T.wire else "#ffffff")
        g.circle(bx0 + (bx1 - bx0) * v, by, 6, fill="#ffffff")


@_spec("waveform")
class _Waveform(_Spec):
    def size(self, ui, n, aw):
        return (aw, float(n.p.get("height", 40)))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        k = int(n.p.get("bars", max(int(w / 6), 12)))
        rng = np.random.default_rng(int(n.p.get("seed", 2)))
        raw = rng.random(k)
        sm = np.convolve(raw, np.ones(3) / 3, mode="same")
        env = np.sin(np.linspace(0.15, np.pi - 0.15, k)) ** 0.6
        hs = (0.18 + 0.82 * (0.55 * sm + 0.45 * raw) * env) * h
        bw = w / k
        v = float(n.p.get("progress", 0.35))
        col = T.tone(n.p.get("tone")) if not T.wire else T.ink2
        rest = T.border_strong if not T.glass else T.border_strong
        for i in range(k):
            bx = x + i * bw + bw * 0.2
            bh = max(float(hs[i]), 3)
            g.rect(bx, y + (h - bh) / 2, bw * 0.6, bh, r=bw * 0.3, fill=col if (i + 0.5) / k <= v else rest)


def _syntax_palette(dark):
    if dark:
        return dict(bg="#0f1420", bar="#151b2a", fg="#d6deeb", kw="#c792ea", str="#c3e88d", num="#f78c6c",
                    com="#6b7398", fn="#82aaff", ty="#ffcb6b", pu="#89ddff", ln="#4b5370", hl="#1d2538")
    return dict(bg="#fbfcfe", bar="#f2f4f8", fg="#1e293b", kw="#7c3aed", str="#15803d", num="#c2410c", com="#94a3b8",
                fn="#2563eb", ty="#b45309", pu="#0e7490", ln="#b6bfcc", hl="#eef2ff")


_KW = set("""def class return if elif else for while in not and or is import from as with try except finally raise
lambda yield async await pass break continue global nonlocal None True False const let var function new this typeof
instanceof export default extends interface type enum public private static void int float bool string fn pub mut impl
struct match use mod self echo then fi do done case esac""".split())
_TOK_RE = re.compile(r"(?P<com>#.*$|//.*$|--\s.*$)|(?P<str>\"(?:[^\"\\]|\\.)*\"?|'(?:[^'\\]|\\.)*'?|`[^`]*`?)|"
                     r"(?P<num>\b\d+(?:\.\d+)?\b)|(?P<word>@?[A-Za-z_][A-Za-z0-9_]*)|(?P<pu>[^\sA-Za-z0-9_])|"
                     r"(?P<sp>\s+)")


def _tokenize(line):
    out = []
    toks = list(_TOK_RE.finditer(line))
    for i, m in enumerate(toks):
        kind, s = m.lastgroup, m.group()
        if kind == "word":
            nxt = line[m.end():m.end() + 1]
            if s.startswith("@"):
                kind = "fn"
            elif s in _KW:
                kind = "kw"
            elif nxt == "(":
                kind = "fn"
            elif s[:1].isupper():
                kind = "ty"
            else:
                kind = "fg"
        elif kind == "sp":
            kind = "fg"
        out.append((s, kind))
    return out


@_spec("code")
class _Code(_Spec):
    def lines(self, n):
        return str(n.p.get("code", "")).expandtabs(4).rstrip("\n").split("\n")

    def size(self, ui, n, aw):
        k = len(self.lines(n))
        return (aw, 16 * 2 + k * 20.0 + (36.0 if n.p.get("title") else 0.0))

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        dark = n.p.get("dark", True if not T.wire else T.dark)
        P = _syntax_palette(dark)
        if T.wire:
            P = dict(P, kw=T.ink, str=T.ink2, num=T.ink2, com=T.ink3, fn=T.ink, ty=T.ink, pu=T.ink2,
                     fg=T.ink, bg=T.surface2, bar=T.surface3, ln=T.ink3, hl=T.surface3)
        r = min(T.r["card"], 14)
        g.rect(x, y, w, h, r=r, fill=P["bg"], stroke=T.border if not dark else "#ffffff14", stroke_width=ui.hl)
        yy = y + 16
        if n.p.get("title"):
            g.rect(x, y, w, 36, r=(r, r, 0, 0), fill=P["bar"])
            for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
                g.circle(x + 18 + i * 16, y + 18, 5, fill=c if not T.wire else T.border_strong)
            _txt(ui, g, x + w / 2, y + 18, n.p["title"], 12, P["com"], True, "mc", maxw=w - 140)
            yy = y + 36 + 14
        fs = float(n.p.get("font_size", 13))
        cw = _tw("M" * 20, fs, False, "mono") / 20.0
        lines = self.lines(n)
        lnw = (len(str(len(lines))) * cw + 24) if n.p.get("line_numbers", True) else 12
        hl = set(n.p.get("highlight") or [])
        maxc = int((w - lnw - 16) // cw)
        for i, ln in enumerate(lines):
            cy = yy + i * 20 + 10
            if cy > y + h:
                break
            if (i + 1) in hl:
                g.rect(x + 1, cy - 10, w - 2, 20, fill=P["hl"])
                g.rect(x + 1, cy - 10, 3, 20, fill=T.accent if not T.wire else T.ink2)
            if n.p.get("line_numbers", True):
                _txt(ui, g, x + lnw - 12, cy, str(i + 1), fs, P["ln"], anchor="rc", family="mono")
            col = 0
            if len(ln) > maxc:
                ln = ln[:max(maxc - 1, 0)] + "…"
            for s, kind in _tokenize(ln):
                if s.strip():                 # a leading NBSP keeps edge glyphs like ( and " from being cropped
                    _txt(ui, g, x + lnw + (col - 1) * cw, cy, " " + s, fs, P.get(kind, P["fg"]), family="mono")
                col += len(s)


@_spec("terminal")
class _Terminal(_Spec):
    def lines(self, n):
        ls = n.p.get("lines")
        if isinstance(ls, str):
            ls = ls.rstrip("\n").split("\n")
        return list(ls or [])

    def size(self, ui, n, aw):
        return (aw, 32 + 14 * 2 + len(self.lines(n)) * 20.0 + 20)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        r = min(T.r["card"], 12)
        bg, bar, fg, mut = ("#0d1117", "#161b22", "#e6edf3", "#8b949e") if not T.wire else \
            (T.surface2, T.surface3, T.ink, T.ink3)
        g.rect(x, y, w, h, r=r, fill=bg, stroke="#ffffff14" if not T.wire else T.border_strong, stroke_width=ui.hl)
        g.rect(x, y, w, 32, r=(r, r, 0, 0), fill=bar)
        for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            g.circle(x + 16 + i * 16, y + 16, 5, fill=c if not T.wire else T.border_strong)
        _txt(ui, g, x + w / 2, y + 16, n.p.get("title", "zsh"), 12, mut, True, "mc", maxw=w - 120)
        fs = 13.0
        cw = _tw("M" * 20, fs, False, "mono") / 20.0
        maxc = int((w - 32) // cw)
        green = "#3fb950" if not T.wire else T.ink
        prompt = n.p.get("prompt", "~/app $")
        yy = y + 32 + 14 + 10
        lines = self.lines(n)
        for i, ln in enumerate(lines):
            s = str(ln)
            if s.startswith("$ "):
                pw = _txt(ui, g, x + 16, yy, prompt, fs, green, True, family="mono")
                cmd = s[2:]
                cmax = max(maxc - len(prompt) - 1, 1)
                _txt(ui, g, x + 16 + pw + cw, yy, cmd if len(cmd) <= cmax else cmd[:cmax - 1] + "…", fs, fg,
                     family="mono")
            else:
                low = s.lower()
                col = mut
                if s.startswith(("✓", "✔", "ok", "done", "+")) or "success" in low or "passed" in low:
                    col = green
                elif s.startswith(("✗", "✘", "error", "ERR", "-")) or "failed" in low:
                    col = "#f85149" if not T.wire else T.ink
                elif s.startswith(("warn", "!")):
                    col = "#d29922" if not T.wire else T.ink2
                _txt(ui, g, x + 16, yy, s if len(s) <= maxc else s[:maxc - 1] + "…", fs, col, family="mono")
            yy += 20
        pw = _txt(ui, g, x + 16, yy, prompt, fs, green, True, family="mono")
        g.rect(x + 16 + pw + cw, yy - 8, cw, 16, fill=fg, opacity=0.85)


@_spec("chart")
class _Chart(_Spec):
    def size(self, ui, n, aw):
        return (aw, float(n.p.get("height", 200)))

    def paint(self, ui, g, n, x, y, w, h):
        from . import chart as _ch
        T = ui.T
        f = float(n.p.get("scale", 1.6))
        acc = [T.accent, _hue(T.accent, 150, None, None), _hue(T.accent, 50), _hue(T.accent, -60), T.success,
               T.warning, T.info, T.danger]
        if T.wire:
            acc = [T.ink2, T.ink3, T.border_strong, T.ink, T.ink2, T.ink3, T.ink2, T.ink3]
        th = Theme(dict(_v.THEMES["dark" if T.dark else "light"]))
        th.update(name="ui", background=T.surface if not T.glass else ("#1a1540" if T.dark else "#ffffff"),
                  surface=T.surface2 if not T.glass else "#ffffff10", ink=T.ink, muted=T.ink2,
                  grid=T.border if not T.glass else ("#ffffff26" if T.dark else "#00000014"), accents=acc,
                  font=dict(family=T.font, weight="normal", title_weight="bold"), dark=T.dark)
        fig = _ch.Figure(None, th, size=(w * f, h * f), background=False)
        axkw = dict(frame="minimal", legend=n.p.get("legend", "auto"), pad=n.p.get("pad"))
        axkw.update(n.p.get("axes") or {})
        ax = fig.axes((0, 0, 1, 1), **{k: v for k, v in axkw.items() if v is not None})
        kind = n.p.get("kind", "line")
        data = n.p.get("data")
        xs = n.p.get("x")
        kw = dict(n.p.get("series") or {})
        if kind in ("line", "area"):
            if xs is None:
                first = next(iter(data.values())) if isinstance(data, dict) else data
                xs = list(range(len(first)))
            kw.setdefault("smooth", True)
            if kind == "line":
                kw.setdefault("fill", not isinstance(data, dict) or len(data) == 1)
            getattr(ax, kind)(xs, data, **kw)
        elif kind in ("bar", "barh", "stacked"):
            if kind == "stacked":
                kw["stacked"] = True
            if isinstance(data, dict) and xs is not None:
                ax.bar(xs, data, horizontal=kind == "barh", **kw)
            elif xs is not None:
                ax.bar(xs, data, horizontal=kind == "barh", **kw)
            else:
                ax.bar(data, horizontal=kind == "barh", **kw)
        elif kind in ("pie", "donut"):
            (ax.donut if kind == "donut" else ax.pie)(data, **kw)
        elif kind == "spark":
            ax.sparkline(data, **kw)
        elif kind == "scatter":
            ax.scatter(xs, data, **kw)
        else:
            getattr(ax, kind)(data, **kw)
        d = ax.drawing(standalone=False)
        g.embed(d, box=(x, y, x + w, y + h))


@_spec("shot")
class _Shot(_Spec):
    def _aspect(self, n):
        src = n.p.get("src")
        if isinstance(src, Screen):
            d = src._device_box(frame=n.p.get("frame", True))
            return (d[2] - d[0]) / max(d[3] - d[1], 1)
        if isinstance(src, Drawing):
            return src.width / max(src.height, 1)
        if src is not None and hasattr(src, "buf"):
            return src.w / max(src.h, 1)
        return 16 / 10

    def size(self, ui, n, aw):
        h = float(n.p.get("height") or aw / self._aspect(n))
        if n.p.get("max_h"):                                  # a tall shot (phone) fits the box instead of overflowing
            h = min(h, float(n.p["max_h"]))
        return (aw, h)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        src = n.p.get("src")
        r = n.p.get("radius", 0)
        if isinstance(src, Screen):
            fr = n.p.get("frame", True)
            b = src._device_box(frame=fr)
            kk = min(w / max(b[2] - b[0], 1), h / max(b[3] - b[1], 1))
            d = src._drawing(frame=fr, margin=0, k=ui._k * kk)
            if n.p.get("shadow", True):
                with g.group(None, shadow=T.shadow_lg) as s_:
                    s_.embed(d, box=(x, y, x + w, y + h))
            else:
                g.embed(d, box=(x, y, x + w, y + h))
        elif isinstance(src, Drawing):
            g.embed(src, box=(x, y, x + w, y + h))
        elif src is not None and hasattr(src, "buf"):
            g.rect(x, y, w, h, r=r, fill=T.surface2)
            ui._hook(g, "image", (x, y, w, h), r, canvas=src, fit=n.p.get("fit", "contain"))
        else:
            _landscape(ui, g, x, y, w, h, 2, r or T.r["card"])

@_spec("fab", w="hug")
class _Fab(_Spec):
    def size(self, ui, n, aw):
        lab = n.p.get("label")
        if lab:
            return (min(aw, 56 + _tw(str(lab), 15, True, ui.T.font) + 20), 56.0)
        return (56.0, 56.0)

    def paint(self, ui, g, n, x, y, w, h):
        T = ui.T
        r = h / 2 if T.style != "material" else 16
        fill = T.accent if T.style != "material" else mix(T.surface2, T.accent, 0.3)
        fg = T.on_accent if T.style != "material" else T.accent_ink
        if T.wire:
            fill, fg = T.ink, T.bg
        with g.group(None, shadow=dict(T.shadow_lg, dy=8, blur=18, opacity=(0.3 if not T.dark else 0.6))) as s_:
            s_.rect(x, y, w, h, r=r, fill=fill)
        ic = n.p.get("icon", "plus")
        if n.p.get("label"):
            _ico(ui, g, ic, x + 28, y + h / 2, 24, fg, sw=2.2)
            _txt(ui, g, x + 50, y + h / 2, n.p["label"], 15, fg, True, maxw=w - 64)
        else:
            _ico(ui, g, ic, x + w / 2, y + h / 2, 26, fg, sw=2.2)


# --------------------------------------------------------------------------------------------- devices

DEVICES = {
    "phone": dict(size=(390, 844), radius=47, status=54, safe=34, wide=False, ext=(16, 13, 16, 13), margin=48),
    "tablet": dict(size=(834, 1194), radius=24, status=30, safe=20, wide=True, ext=(20, 20, 20, 20), margin=56),
    "laptop": dict(size=(1280, 800), radius=6, status=28, safe=0, wide=True, ext=(90, 26, 90, 40), margin=56),
    "browser": dict(size=(1280, 800), radius=12, status=0, safe=0, wide=True, ext=(0, 88, 0, 0), margin=36),
    "desktop_window": dict(size=(1120, 720), radius=12, status=0, safe=0, wide=True, ext=(0, 40, 0, 0), margin=36),
    "watch": dict(size=(198, 242), radius=48, status=34, safe=0, wide=False, ext=(16, 92, 26, 92), margin=32),
    "plain": dict(size=(390, 844), radius=0, status=0, safe=0, wide=False, ext=(0, 0, 0, 0), margin=0),
}

_IMAGES = {}
_IMG_N = [0]


class _MatGroup(Group):
    """A group with an explicit affine matrix (device mapping for coverage masks)."""

    @property
    def matrix(self):
        return self._m


class _MaskCanvas:
    """Minimal canvas stand-in that records the coverage of a Drawing (no layer recording)."""

    def __init__(self, w, h):
        self.w, self.h = int(w), int(h)
        self.region = None

    def paint(self, *a, **k):
        return self

    def paste(self, other, x=0, y=0, mask=None, **k):
        if mask is None:
            return self
        m = np.asarray(mask, np.float32)
        if self.region is None:
            self.region = (int(x), int(y), m.copy())
        else:
            x0, y0, m0 = self.region
            X0, Y0 = min(x0, x), min(y0, y)
            X1, Y1 = max(x0 + m0.shape[1], x + m.shape[1]), max(y0 + m0.shape[0], y + m.shape[0])
            out = np.zeros((Y1 - Y0, X1 - X0), np.float32)
            out[y0 - Y0:y0 - Y0 + m0.shape[0], x0 - X0:x0 - X0 + m0.shape[1]] = m0
            s = out[y - Y0:y - Y0 + m.shape[0], x - X0:x - X0 + m.shape[1]]
            s[:] = s + m * (1 - s)
            self.region = (X0, Y0, out)
        return self


def _mm(m, n):
    """Affine product m · n (apply n first), SVG order (a, b, c, d, e, f)."""
    return (m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1], m[0] * n[2] + m[2] * n[3],
            m[1] * n[2] + m[3] * n[3], m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5])


def _ap(m, x, y):
    return (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])


def _minv(m):
    a, b, c, d, e, f = m
    det = a * d - b * c
    if abs(det) < 1e-12:
        det = 1e-12
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    return (ia, ib, ic, id_, -(ia * e + ic * f), -(ib * e + id_ * f))


def _world(item):
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    g = item.parent
    while g is not None:
        m = _mm(g.matrix, m)
        g = g.parent
    return m


def _coverage(W, H, M, build):
    d = Drawing(W, H, background=False, name="mask")
    mg = _MatGroup(d, None)
    mg._m = M
    d.add(mg)
    build(mg)
    mc = _MaskCanvas(W, H)
    d.draw(mc)
    reg = getattr(_v, "_REGISTRY", None)
    if reg is not None and d in reg:
        reg.remove(d)
    return mc.region


def _is_hook(it):
    return isinstance(it, Group) and getattr(it, "_ui_hook", None) is not None


def _has_hook(it):
    if _is_hook(it):
        return True
    return isinstance(it, Group) and any(_has_hook(c) for c in it.children)


def _leaves(it, out):
    if isinstance(it, Group) and not _is_hook(it):
        for c in it.children:
            _leaves(c, out)
    else:
        out.append(it)


def _first_index(it, idx, memo):
    if isinstance(it, Group) and not _is_hook(it):
        v = min((_first_index(c, idx, memo) for c in it.children), default=10 ** 9)
        memo[id(it)] = v
        return v
    return idx.get(id(it), 10 ** 9)


def _prune(grp, idx, lo, hi, first):
    kids = []
    for c in grp.children:
        if isinstance(c, Group) and not _is_hook(c):
            sub = _prune(c, idx, lo, hi, first)
            if sub is not None:
                kids.append(sub)
        elif not _is_hook(c) and lo <= idx.get(id(c), -1) < hi:
            kids.append(c)
    if not kids:
        return None
    cp = _copy.copy(grp)
    cp.children = kids
    if first.get(id(grp), 0) < lo:
        cp.shadow = None
    return cp


def _do_hook(cv, M, hook):
    h = hook._ui_hook
    x, y, w, hh = h["rect"]
    r = h.get("r", 0)
    reg = _coverage(cv.w, cv.h, M, lambda grp: grp.rect(x, y, w, hh, r=r, fill="#ffffff"))
    if reg is None:
        return
    x0, y0, mask = reg
    mh, mw = mask.shape
    s = math.sqrt(abs(M[0] * M[3] - M[1] * M[2]))
    if h["kind"] == "glass":
        sig = max(h.get("sigma", 20.0) * s, 0.5)
        pad = int(sig * 3) + 2
        X0, Y0 = max(x0 - pad, 0), max(y0 - pad, 0)
        X1, Y1 = min(x0 + mw + pad, cv.w), min(y0 + mh + pad, cv.h)
        bl = gaussian_blur(cv.buf[Y0:Y1, X0:X1], sig)
        sub = bl[y0 - Y0:y0 - Y0 + mh, x0 - X0:x0 - X0 + mw]
        sat = h.get("saturate", 1.15)
        if sat != 1.0:
            lum = sub.mean(-1, keepdims=True)
            sub = lum + (sub - lum) * sat
        cv.paste(sub, x0, y0, mask=mask)
    elif h["kind"] == "image":
        src = _IMAGES.get(h["key"])
        if src is None:
            return
        buf = src.buf if hasattr(src, "buf") else np.asarray(src, np.float32)
        if buf.ndim == 2:
            buf = np.repeat(buf[..., None], 3, -1)
        sh, sw_ = buf.shape[:2]
        tw, th_ = max(w * s, 1), max(hh * s, 1)
        k = min(sw_ / tw, sh / th_) if h.get("fit", "cover") == "cover" else max(sw_ / tw, sh / th_)
        if k > 2.0:                                      # prefilter big sources
            im = Image.fromarray((np.clip(buf, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
            nw, nh = max(int(sw_ / k * 1.5), 1), max(int(sh / k * 1.5), 1)
            buf = np.asarray(im.resize((nw, nh), _RES.LANCZOS), np.float32) / 255.0
            sh, sw_ = buf.shape[:2]
        Mi = _minv(M)
        Y, X = np.mgrid[y0:y0 + mh, x0:x0 + mw].astype(np.float64)
        X += 0.5
        Y += 0.5
        lx = Mi[0] * X + Mi[2] * Y + Mi[4]
        ly = Mi[1] * X + Mi[3] * Y + Mi[5]
        u, v = (lx - x) / max(w, 1e-9), (ly - y) / max(hh, 1e-9)
        ra, sa = w / max(hh, 1e-9), sw_ / max(sh, 1)
        cover = h.get("fit", "cover") == "cover"
        if (sa > ra) == cover:                              # scale by height
            sx = sh * ra
            px = (u - 0.5) * sx + sw_ / 2
            py = v * sh
        else:
            sy = sw_ / ra
            px = u * sw_
            py = (v - 0.5) * sy + sh / 2
        inside = ((px >= 0) & (px <= sw_) & (py >= 0) & (py <= sh)).astype(np.float32)
        from .filters import sample
        img = sample(buf, px - 0.5, py - 0.5)
        cv.paste(img, x0, y0, mask=mask * inside)


def _draw_dwg(dwg, cv, box=None, log=None):
    """Paint a Drawing, resolving UI hooks (glass blur, Canvas images) in paint order. Returns the device matrix."""
    W, H = cv.w, cv.h
    bx0, by0, bx1, by1 = box if box is not None else (0, 0, W, H)
    k = min((bx1 - bx0) / dwg.width, (by1 - by0) / dwg.height)
    ox = bx0 + ((bx1 - bx0) - dwg.width * k) / 2
    oy = by0 + ((by1 - by0) - dwg.height * k) / 2
    M0 = (k, 0.0, 0.0, k, ox, oy)
    if not _has_hook(dwg.root):
        dwg.draw(cv, log=log, box=box)
        return M0
    children, bg = list(dwg.root.children), dwg.background
    try:
        dwg.root.children = []
        dwg.draw(cv, log=log, box=box)                      # background only
        dwg.background = None
        run = []

        def flush():
            if run:
                dwg.root.children = list(run)
                dwg.draw(cv, log=log, box=box)
                run.clear()

        for ch in children:
            if not isinstance(ch, Group):
                run.append(ch)
                continue
            flush()
            if not _has_hook(ch):
                dwg.root.children = [ch]
                dwg.draw(cv, log=log, box=box)
                continue
            leaves = []
            _leaves(ch, leaves)
            idx = {id(l): i for i, l in enumerate(leaves)}
            first = {}
            _first_index(ch, idx, first)
            cuts = [i for i, l in enumerate(leaves) if _is_hook(l)]
            lo = 0
            for j, c in enumerate(cuts + [len(leaves)]):
                part = _prune(ch, idx, lo, c, first)
                if part is not None:
                    dwg.root.children = [part]
                    dwg.draw(cv, log=log if lo == 0 else None, box=box)
                elif lo == 0 and log:
                    log(ch.name or "group")
                if c < len(leaves):
                    hk = leaves[c]
                    _do_hook(cv, _mm(_mm(M0, _world(hk)), hk.matrix), hk)
                lo = c + 1
        flush()
    finally:
        dwg.root.children = children
        dwg.background = bg
    return M0


def _png_b64(src, max_side=1400):
    buf = src.buf if hasattr(src, "buf") else np.asarray(src, np.float32)
    im = Image.fromarray((np.clip(buf, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    k = max(im.size) / float(max_side)
    if k > 1:
        im = im.resize((max(int(im.size[0] / k), 1), max(int(im.size[1] / k), 1)), _RES.LANCZOS)
    bio = io.BytesIO()
    im.save(bio, "PNG")
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _svg_with_images(dwg):
    """SVG text of ``dwg`` with Canvas images embedded as <image> (glass panels stay translucent)."""
    svg = dwg.to_svg()
    hooks = []

    def walk(it):
        if _is_hook(it):
            hooks.append(it)
        elif isinstance(it, Group):
            for c in it.children:
                walk(c)
    walk(dwg.root)
    used = False
    for hk in hooks:
        h = hk._ui_hook
        if h["kind"] != "image" or h["key"] not in _IMAGES:
            continue
        x, y, w, hh = h["rect"]
        par = "xMidYMid slice" if h.get("fit", "cover") == "cover" else "xMidYMid meet"
        tag = (f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{hh:.2f}" preserveAspectRatio="{par}" '
               f'xlink:href="data:image/png;base64,{_png_b64(_IMAGES[h["key"]])}"/>')
        pat = re.compile(r'<g id="' + re.escape(hk.name) + r'(?:-\d+)?"></g>')
        svg, k = pat.subn(lambda m: tag, svg, count=1)
        used = used or k > 0
    if used and "xmlns:xlink" not in svg:
        svg = svg.replace('<svg xmlns="http://www.w3.org/2000/svg"',
                          '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"', 1)
    return svg


def _backdrop(g, x, y, w, h, T, kind="soft", r=0, seed=0):
    """Colourful soft backdrop: base gradient + blurred colour blobs (glass screens and presentations)."""
    acc = T.accent if not T.wire else "#9a9a9a"
    if isinstance(kind, str) and kind.startswith("#"):
        g.rect(x, y, w, h, r=r, fill=kind)
        return
    if T.wire:
        g.rect(x, y, w, h, r=r, fill=T.bg if kind != "soft" else mix(T.bg, "#000000", 0.04))
        return
    if T.dark:
        base = linear_gradient([mix("#0d0b1f", acc, 0.12), mix("#081226", acc, 0.06)], start=(0, 0), end=(1, 1))
        blobs = [(0.12, 0.15, 0.55, acc, 0.7), (0.92, 0.42, 0.5, _hue(acc, -40, light=0.45), 0.5),
                 (0.35, 0.95, 0.55, _hue(acc, -80, light=0.42), 0.55)]
    else:
        base = linear_gradient([mix("#f6f4ff", acc, 0.1), mix("#eef6ff", acc, 0.05)], start=(0, 0), end=(1, 1))
        blobs = [(0.1, 0.12, 0.55, mix(acc, "#ffffff", 0.35), 0.55),
                 (0.95, 0.4, 0.5, _hue(acc, -40, light=0.8), 0.6), (0.3, 0.98, 0.55, _hue(acc, -80, light=0.82), 0.55)]
    if kind == "flat":
        g.rect(x, y, w, h, r=r, fill=T.bg)
        return
    strength = 1.0 if kind in ("glass", "aurora", "vivid") else 0.55
    with g.group(None, clip=g.rect(x, y, w, h, r=r, add=False)) as c:
        c.rect(x, y, w, h, fill=base)
        rng = np.random.default_rng(seed)
        for fx, fy, fr, col, a in blobs:
            fx += (rng.random() - 0.5) * 0.08
            fy += (rng.random() - 0.5) * 0.08
            rr = fr * max(w, h) * (0.9 if kind == "glass" else 1.0)
            c.circle(x + fx * w, y + fy * h, rr, fill=radial_gradient([(0, col, a * strength),
                                                                        (0.55, col, a * strength * 0.45),
                                                                        (1, col, 0.0)]))


def _safe_name(s):
    return re.sub(r"\s+", " ", str(s)).strip()[:48] or "item"


class Screen:
    """A UI screen in a device frame. Build it with component calls, then ``draw`` / ``render`` / ``save_svg``.

    ``Screen(device="phone", theme="light", accent=None, style="modern", size=None, height=None, title=None,
    url=None, status_bar=True, time="9:41", pad=None, gap=None, margin=None, backdrop=None, name=None)``

    * ``device``: phone / tablet / laptop / browser / desktop_window / watch / plain.
    * ``theme``: light / dark; ``style``: modern / glass / material / wireframe; ``accent``: one brand colour.
    * ``size=(w, h)``: viewport size in points; ``height="auto"`` grows the viewport to fit (long web pages).
    * ``title`` / ``url``: browser tab title and address, window title. ``backdrop``: colour or "soft" around the
      device when rendered on its own (``margin`` points).
    """

    def __init__(self, device="phone", theme="light", accent=None, style="modern", size=None, height=None,
                 title=None, url=None, status_bar=True, time="9:41", pad=None, gap=None, margin=None, backdrop=None,
                 name=None, frame=True, content_w=1160, seed=1):
        if device not in DEVICES:
            from ._args import unknown_name
            raise unknown_name("device", device, DEVICES)
        self.device = device
        self.D = DEVICES[device]
        self.T = tokens(theme, style, accent)
        self._fit_h = size is None and height is None and device in ("browser", "desktop_window", "laptop")
        vw, vh = size if size is not None else self.D["size"]
        self.vw, self.vh = float(vw), (float(vh) if vh not in (None, "auto") else 0.0)
        self.auto_h = height == "auto" or vh == "auto"
        if _num(height):
            self.vh = float(height)
        self.window_title, self.url, self.time = title, url, time
        self.status_bar = status_bar and self.D["status"] > 0
        self.wide = self.D["wide"] or (device == "plain" and self.vw >= 700)
        self.frame = frame and device != "plain"
        self.margin = float(self.D["margin"] if margin is None else margin)
        self.backdrop = backdrop
        self.name = name or f"{device} screen"
        self.content_w = float(content_w)
        self.seed = seed
        if pad is None:
            pad = {"watch": (2, 12, 12, 12)}.get(device, 0 if self.wide else (8, 20, 24, 20))
        if gap is None:
            gap = 8 if device == "watch" else (0 if self.wide else 16)
        self.root = Node(self, "stack", axis="v", gap=float(gap), pad=_pad4(pad), align="stretch", justify="start")
        self._header, self._footer, self.overlays = [], [], []
        self._stack = [self.root]
        self._frames = {}
        self._mcache = {}
        self._laid = False
        self._last = None
        self._k = 1.0
        self.hl = 1.0
        self._vp_marker = None

    # ------------------------------------------------------------------ geometry
    @property
    def top_inset(self):
        return float(self.D["status"]) if (self.status_bar or self.device == "phone") else 0.0

    @property
    def safe_bottom(self):
        return float(self.D["safe"])

    def _ext(self, frame=True):
        return self.D["ext"] if (frame and self.frame) else (0, 0, 0, 0)

    def _device_box(self, frame=True):
        self._layout_all()
        l, t, r, b = self._ext(frame)
        return (-l, -t, self.vw + r, self.vh + b)

    # ------------------------------------------------------------------ building
    def _add(self, ctype, **kw):
        n = Node(self, ctype, **kw)
        parent = self._stack[-1]
        n.parent = parent
        parent.children.append(n)
        self._laid = False
        return n

    def _container(self, axis, gap, pad, align, justify, ctype="stack", **kw):
        return self._add(ctype, axis=axis, gap=float(gap), pad=_pad4(pad), align=align, justify=justify, **kw)

    def vstack(self, gap=12, pad=0, align="stretch", justify="start", **kw):
        """Vertical stack (context manager). ``align`` stretch / start / center / end; ``justify`` start /
        center / end / between. Sizing keywords for every call: ``w`` / ``h`` = number, "fill", "hug";
        ``max_w``; ``grow``; ``span`` (grid); ``name``. Surface: ``surface="card"|"inset"|"tint"``, ``bg``,
        ``radius``, ``border``, ``shadow``, ``clip``, ``dividers``."""
        return self._container("v", gap, pad, align, justify, **kw)

    column = vstack

    def hstack(self, gap=12, pad=0, align="center", justify="start", wrap=False, **kw):
        """Horizontal stack (context manager). ``wrap=True`` flows children onto new lines (chips, tags)."""
        return self._container("h", gap, pad, align, justify, wrap=wrap, **kw)

    row = hstack

    def grid(self, cols=2, gap=12, pad=0, row_gap=None, equal=True, **kw):
        """Grid of ``cols`` equal columns (context manager); children may take ``span=2``. Rows are equal height."""
        kw2 = dict(cols=int(cols), gap=float(gap), pad=_pad4(pad), equal=equal)
        if row_gap is not None:
            kw2["row_gap"] = float(row_gap)
        return self._add("grid", **kw2, **kw)

    def spacer(self, size=None, **kw):
        """Flexible space (fills the main axis) or a fixed gap of ``size``."""
        if size is not None:
            kw.setdefault("w", float(size))
            kw.setdefault("h", float(size))
        return self._add("spacer", **kw)

    def scroll(self, h="fill", gap=12, pad=0, **kw):
        """Scroll region: content keeps its natural height and is clipped to the region."""
        return self._container("v", gap, pad, "stretch", "start", scroll=True, h=h, **kw)

    def card(self, pad=16, gap=12, align="stretch", **kw):
        """Card surface (context manager): a vstack on the style's card surface."""
        kw.setdefault("surface", "card")
        return self._container("v", gap, pad, align, kw.pop("justify", "start"), **kw)

    def list_group(self, title=None, pad=16, **kw):
        """Grouped list (context manager): rows inside one card with inset dividers. ``title`` above it."""
        if title:
            self.text(title, role="overline", color=self.T.ink2)

        def inset(row):
            if row.kind != "list_row" or not row.p.get("leading"):
                return 0.0
            return _SPEC["list_row"]._lead_w(row) + 12

        kw.setdefault("surface", "card")
        return self._container("v", 0, (2, pad, 2, pad), "stretch", "start", dividers=True, divider_inset=inset,
                               **kw)

    def section(self, bg=None, pad=(72, 32), gap=24, max_w=None, align="stretch", **kw):
        """Full-bleed page band with centred content of ``max_w`` (landing pages). Returns the inner column."""
        outer = self._container("v", 0, pad, "stretch", "start", bg=bg, name=kw.pop("name", None))
        self._stack.append(outer)
        inner = self._container("v", gap, 0, align, kw.pop("justify", "start"), max_w=max_w or self.content_w,
                                self_align="center", **kw)
        self._stack.pop()
        return inner

    # ------------------------------------------------------------------ text
    def text(self, text, role="body", **kw):
        """Paragraph text that wraps in its box. ``role``: display h1 h2 h3 title body label small caption
        overline link; ``color`` (or a tone: accent / success / danger / muted), ``align``, ``max_lines``,
        ``size``, ``bold``."""
        return self._add("text", text=str(text), role=role, **kw)

    def heading(self, text, level=2, **kw):
        """Heading: level 0 = display, 1–3 = h1–h3, 4 = title."""
        role = {0: "display", 1: "h1", 2: "h2", 3: "h3", 4: "title"}.get(int(level), "h2")
        return self.text(text, role=role, **kw)

    def title(self, text, **kw):
        return self.text(text, role="title", **kw)

    def caption(self, text, **kw):
        return self.text(text, role="caption", **kw)

    def small(self, text, **kw):
        return self.text(text, role="small", **kw)

    def label(self, text, **kw):
        return self.text(text, role="label", **kw)

    def overline(self, text, **kw):
        return self.text(text, role="overline", **kw)

    def link(self, text, **kw):
        kw.setdefault("color", "accent")
        return self.text(text, role="link", **kw)

    def divider(self, label=None, inset=0, vertical=False, **kw):
        if vertical:
            kw.setdefault("h", "fill")
            kw.setdefault("w", 1)
        return self._add("divider", label=label, inset=inset, vertical=vertical, **kw)

    def icon(self, name, size=24, color=None, **kw):
        return self._add("icon", icon=name, size=size, color=color, **kw)

    def icon_badge(self, name, size=40, tone="accent", shape="rounded", solid=False, **kw):
        """Icon on a tinted square / circle (feature lists, settings rows)."""
        return self._add("icon_badge", icon=name, size=size, tone=tone, shape=shape, solid=solid, **kw)

    # ------------------------------------------------------------------ controls
    def button(self, text=None, variant="primary", icon=None, icon_right=None, size="md", full=False, **kw):
        """Button. ``variant``: primary secondary outline ghost link destructive tonal inverse white;
        ``size`` sm / md / lg / xl; ``full=True`` (or ``w="fill"``) spans the width."""
        if full:
            kw.setdefault("w", "fill")
        return self._add("button", text=text, variant=variant, icon=icon, icon_right=icon_right, size=size,
                         full=full, **kw)

    def icon_button(self, icon, variant="ghost", size="md", badge=None, color=None, **kw):
        """Round / square icon button: variant ghost filled tonal outline surface inverse; ``badge`` count / True."""
        return self._add("icon_button", icon=icon, variant=variant, size=size, badge=badge, color=color, **kw)

    def input(self, label=None, value=None, placeholder=None, icon=None, trailing=None, helper=None, error=None,
              focused=False, kind="input", **kw):
        """Text field with optional label, leading icon, trailing icon / text, helper or error message."""
        kw.setdefault("w", "fill")
        return self._add("field", kind=kind, label=label, value=value, placeholder=placeholder, icon=icon,
                         trailing=trailing, helper=helper, error=error, focused=focused, **kw)

    def text_area(self, label=None, value=None, placeholder=None, rows=4, **kw):
        return self.input(label, value, placeholder, kind="textarea", rows=rows, **kw)

    textarea = text_area

    def search(self, placeholder="Search", value=None, trailing=None, **kw):
        return self.input(None, value, placeholder, kind="search", trailing=trailing, **kw)

    def select(self, label=None, value=None, placeholder="Select", **kw):
        return self.input(label, value, placeholder, kind="select", **kw)

    def date_field(self, label=None, value=None, placeholder="dd / mm / yyyy", **kw):
        return self.input(label, value, placeholder, kind="date", **kw)

    def password(self, label="Password", value="secret12", **kw):
        return self.input(label, value, kind="password", **kw)

    def checkbox(self, label=None, checked=False, sub=None, **kw):
        return self._add("check", label=label, checked=checked, sub=sub, **kw)

    def radio(self, label=None, selected=False, sub=None, **kw):
        return self._add("check", label=label, checked=selected, sub=sub, radio=True, **kw)

    def toggle(self, label=None, on=True, sub=None, **kw):
        """Switch; with a ``label`` it is a full-width settings row."""
        if label is None:
            kw.setdefault("w", "hug")
        return self._add("toggle", label=label, on=on, sub=sub, **kw)

    switch = toggle

    def slider(self, value=0.5, label=None, value_text=None, min_label=None, max_label=None, tone=None, **kw):
        return self._add("slider", value=value, label=label, value_text=value_text, min_label=min_label,
                         max_label=max_label, tone=tone, **kw)

    def stepper(self, value=1, label=None, sub=None, **kw):
        if label is None:
            kw.setdefault("w", "hug")
        return self._add("stepper", value=value, label=label, sub=sub, **kw)

    def segmented(self, items, active=0, **kw):
        return self._add("segmented", items=list(items), active=active, **kw)

    def tabs(self, items, active=0, variant="underline", counts=None, fill=False, **kw):
        """Tabs: ``variant`` underline / pills; ``counts`` {index: n}."""
        return self._add("tabs", items=list(items), active=active, variant=variant, counts=counts, fill=fill, **kw)

    def breadcrumb(self, items, **kw):
        return self._add("breadcrumb", items=list(items), **kw)

    def pagination(self, page=1, pages=10, **kw):
        return self._add("pagination", page=page, pages=pages, **kw)

    # ------------------------------------------------------------------ navigation
    def _slot(self):
        return self._stack[-1] is self.root

    def app_bar(self, title="", leading=None, trailing=None, subtitle=None, large=False, center=None, **kw):
        """Top bar. ``leading``: "back" / "back:Label" / "menu" / "close" / "avatar:Name" / an icon; ``trailing``:
        list of icons, "avatar:Name", "button:Text" or text actions. ``large=True``: iOS large title.
        At the top level it is pinned above the scrolling content."""
        n = Node(self, "app_bar", title=title, leading=leading, trailing=trailing, subtitle=subtitle, large=large,
                 center=center, **kw)
        return self._place(n, "header")

    top_bar = app_bar

    def navbar(self, brand="Brand", links=(), actions=(), active=None, **kw):
        """Website navigation bar: logo + name, links, action buttons [("Sign in", "ghost"), "Start free"]."""
        n = Node(self, "navbar", brand=brand, links=list(links), actions=list(actions), active=active, **kw)
        return self._place(n, "header")

    def tab_bar(self, items, active=0, badges=None, **kw):
        """Bottom tab bar: items [(icon, label), ...] or icon names; pinned to the bottom at the top level."""
        n = Node(self, "tab_bar", items=list(items), active=active, badges=badges, **kw)
        return self._place(n, "footer")

    bottom_nav = tab_bar

    def _place(self, n, slot):
        self._laid = False
        if self._slot():
            (self._header if slot == "header" else self._footer).append(n)
            return n
        parent = self._stack[-1]
        n.parent = parent
        parent.children.append(n)
        return n

    def sidebar(self, items, active=0, brand=None, user=None, w=256, **kw):
        """Sidebar navigation: items [(icon, label[, count])], "# Section" headers, brand, user=(name, email)."""
        return self._add("sidebar", items=list(items), active=active, brand=brand, user=user, w=w, **kw)

    # ------------------------------------------------------------------ data display
    def list_row(self, title, subtitle=None, leading=None, trailing=None, value=None, chevron=False, toggle=None,
                 badge=None, meta=None, tone=None, **kw):
        """List row: ``leading`` icon name / "avatar:Name" / "initials:Name" / "img:seed"; ``trailing``
        "chevron" / "check" / icon; ``value`` text; ``toggle`` True/False; ``badge``; ``meta`` (time)."""
        return self._add("list_row", title=title, subtitle=subtitle, leading=leading, trailing=trailing, value=value,
                         chevron=chevron, toggle=toggle, badge=badge, meta=meta, tone=tone, **kw)

    row_item = list_row

    def stat(self, label, value, delta=None, icon=None, spark=None, note=None, tone=None, **kw):
        """Stat tile: label, big value, ▲▼ delta pill (``good="up"|"down"``), icon, sparkline."""
        return self._add("stat", label=label, value=value, delta=delta, icon=icon, spark=spark, note=note,
                         tone=tone, **kw)

    def table(self, columns, rows, align=None, widths=None, zebra=False, compact=False, selected=None, **kw):
        """Data table. Cells: text / numbers (right-aligned) / True-False / {"badge": "Paid", "tone": "success"} /
        {"avatar": "Ana Silva", "sub": "ana@mail.com"} / {"progress": 0.6} / {"icon": "file", "text": ...}."""
        return self._add("table", columns=list(columns), rows=[list(r) for r in rows], align=align, widths=widths,
                         zebra=zebra, compact=compact, selected=selected, **kw)

    def badge(self, text, tone="accent", solid=False, dot=False, **kw):
        return self._add("badge", text=str(text), tone=tone, solid=solid, dot=dot, **kw)

    def chip(self, text, selected=False, icon=None, close=False, **kw):
        return self._add("chip", text=str(text), selected=selected, icon=icon, close=close, **kw)

    tag = chip

    def chips(self, items, selected=(), gap=8, **kw):
        """A wrapping row of chips; ``selected`` = indices or labels."""
        sel = set(selected) if isinstance(selected, (list, tuple, set)) else {selected}
        with self.hstack(gap=gap, wrap=True, **kw) as n:
            for i, t in enumerate(items):
                self.chip(t, selected=(i in sel or t in sel))
        return n

    def avatar(self, name="Alex Morgan", size=40, kind="portrait", status=None, **kw):
        """Avatar: ``kind`` portrait (abstract person) / initials; ``status`` online / busy / away."""
        return self._add("avatar", person=name, size=size, kind=kind, status=status, **kw)

    def avatar_group(self, names, size=32, max=4, **kw):
        return self._add("avatar_group", names=list(names), size=size, max=max, **kw)

    def progress(self, value=0.5, label=None, value_text=None, tone=None, thickness=8, **kw):
        return self._add("progress", value=value, label=label, value_text=value_text, tone=tone,
                         thickness=thickness, **kw)

    def ring(self, value=0.72, label=None, sub=None, size=96, tone=None, thickness=None, **kw):
        """Progress ring with a centre label (default the percentage) and optional ``sub`` line."""
        p = dict(value=value, label=label, sub=sub, size=size, tone=tone)
        if thickness:
            p["thickness"] = thickness
        return self._add("ring", **p, **kw)

    progress_ring = ring

    def skeleton(self, lines=3, avatar=False, image=None, **kw):
        """Loading placeholder: ``lines`` of text, optional ``avatar`` disc and ``image`` height."""
        return self._add("skeleton", lines=lines, avatar=avatar, image=image, **kw)

    def empty_state(self, title="Nothing here yet", text=None, icon="inbox", action=None, **kw):
        """Centred empty state: icon disc, title, text, optional action button."""
        with self.vstack(gap=10, pad=(32, 16), align="center", **kw) as n:
            self.icon_badge(icon, size=64, shape="circle")
            self.spacer(4)
            self.heading(title, level=3, align="center")
            if text:
                self.text(text, role="small", align="center", max_w=300)
            if action:
                self.spacer(6)
                self.button(action, variant="primary")
        return n

    def tooltip(self, text, arrow="bottom", near=None, at=None, **kw):
        """Dark tooltip bubble; ``near="component name"`` floats it above that component."""
        n = Node(self, "tooltip", text=text, arrow=arrow, **kw)
        if near or at:
            self.overlays.append((n, dict(kind="near", near=near, at=at, side="above")))
            self._laid = False
            return n
        return self._attach(n)

    def toast(self, text, icon="check", action=None, tone="success", sub=None, position="bottom", **kw):
        """Floating toast (overlay at ``position`` bottom / top, or "inline" in the flow) with icon, text, action."""
        n = Node(self, "toast", text=text, icon=icon, action=action, tone=tone, sub=sub, **kw)
        if position == "inline":
            return self._attach(n)
        self.overlays.append((n, dict(kind="toast", pos=position)))
        self._laid = False
        return n

    def banner(self, title=None, text=None, tone="info", icon=None, action=None, close=False, **kw):
        """Inline alert banner tinted by ``tone`` (info / success / warning / danger)."""
        return self._add("banner", title=title, text=text, tone=tone, icon=icon, action=action, close=close, **kw)

    alert = banner

    def menu(self, items, active=None, near=None, at=None, w=232, **kw):
        """Dropdown menu: items [(icon, label, shortcut), "-", {"label": "Delete", "icon": "trash", "danger": True}].
        ``near="name"`` drops it below that component (overlay); otherwise inline."""
        n = Node(self, "menu", items=list(items), active=active, w=w, **kw)
        if near or at:
            self.overlays.append((n, dict(kind="near", near=near, at=at, side="below")))
            self._laid = False
            return n
        return self._attach(n)

    dropdown = menu

    def _attach(self, n):
        parent = self._stack[-1]
        n.parent = parent
        parent.children.append(n)
        self._laid = False
        return n

    def fab(self, icon="plus", label=None, position="bottom-right", **kw):
        """Floating action button (overlay); ``label`` → extended FAB."""
        n = Node(self, "fab", icon=icon, label=label, **kw)
        self.overlays.append((n, dict(kind="fab", pos=position)))
        self._laid = False
        return n

    def modal(self, title=None, text=None, actions=None, icon=None, tone="accent", w=None, scrim=True, **kw):
        """Centred dialog over a scrim (context manager for extra content). ``actions``: ["Cancel",
        ("Delete", "destructive")] — the last one is primary unless given a variant."""
        n = Node(self, "stack", axis="v", gap=12.0, pad=_pad4(24), align="stretch", justify="start",
                 surface="raised", radius=self.T.r["sheet"] if self.T.r["sheet"] < 30 else 24, **kw)
        self.overlays.append((n, dict(kind="modal", w=w, scrim=scrim)))
        self._laid = False
        self._stack.append(n)
        if icon:
            self.icon_badge(icon, size=48, tone=tone, shape="circle", self_align="center")
        if title:
            self.heading(title, level=3, align="center" if icon else "left")
        if text:
            self.text(text, role="small", align="center" if icon else "left", color=self.T.ink2)
        self._stack.pop()
        if actions:
            n.p["tail"] = [self._actions_row(actions, stretch=True)]
        return n

    dialog = modal

    def sheet(self, title=None, actions=None, scrim=True, **kw):
        """Bottom sheet with grab handle (context manager for its content)."""
        sb = self.safe_bottom
        n = Node(self, "stack", axis="v", gap=14.0, pad=_pad4((28, 20, sb + 16, 20)), align="stretch",
                 justify="start", surface="sheet", **kw)
        self.overlays.append((n, dict(kind="sheet", scrim=scrim)))
        self._laid = False
        if title:
            self._stack.append(n)
            self.heading(title, level=3)
            self._stack.pop()
        if actions:
            n.p["tail"] = [self._actions_row(actions, stretch=True)]
        return n

    bottom_sheet = sheet

    def _actions_row(self, actions, stretch=True):
        row = Node(self, "stack", axis="h", gap=12.0, pad=_pad4(0), align="center", justify="end")
        self._stack.append(row)
        acts = list(actions)
        for i, a in enumerate(acts):
            if isinstance(a, (tuple, list)):
                t, v = a[0], a[1]
            else:
                t, v = a, ("primary" if i == len(acts) - 1 else "secondary")
            self.button(t, variant=v, w="fill" if stretch else None)
        self._stack.pop()
        return row

    def overlay(self, at=(0, 0), w=None, gap=8, pad=0, **kw):
        """Free overlay container at viewport point ``at`` (context manager)."""
        n = Node(self, "stack", axis="v", gap=float(gap), pad=_pad4(pad), align="stretch", justify="start", **kw)
        self.overlays.append((n, dict(kind="near", at=at, w=w)))
        self._laid = False
        return n

    # ------------------------------------------------------------------ conversation / feeds / planning
    def message(self, text, me=False, time=None, status="read", avatar=None, **kw):
        """Chat bubble (``me=True`` → right, accent). ``time`` under the bubble; ``avatar`` name for others."""
        return self._add("message", text=text, me=me, time=time, status=status, avatar=avatar, **kw)

    chat_bubble = message

    def notification(self, title, text=None, time=None, icon="bell", avatar=None, unread=False, tone=None, **kw):
        return self._add("notification", title=title, text=text, time=time, icon=icon, avatar=avatar,
                         unread=unread, tone=tone, **kw)

    def calendar(self, year=2026, month=11, selected=None, today=None, events=None, range=None, first_weekday=0,
                 header=True, **kw):
        """Month grid; ``events`` {day: count or [tones]}; ``range=(a, b)`` highlights a span."""
        return self._add("calendar", year=year, month=month, selected=selected, today=today, events=events,
                         range=range, first_weekday=first_weekday, header=header, **kw)

    def kanban_card(self, title, tag=None, tone="accent", people=(), due=None, comments=None, progress=None,
                    image=False, tags=None, avatar=None, avatars=None, assignee=None, **kw):
        """Board card: ``tag`` / ``tags`` (one label or a list, each tinted by its own tone), ``people`` /
        ``avatar`` / ``avatars`` / ``assignee`` (names), ``due``, ``comments``, ``progress`` 0..1, ``image``."""
        tag = tags if tags is not None else tag
        if isinstance(avatar, str): avatar = [avatar]
        if isinstance(assignee, str): assignee = [assignee]
        if isinstance(people, str): people = [people]
        people = list(people or []) + list(avatar or []) + list(avatars or []) + list(assignee or [])
        return self._add("kcard", title=title, tag=tag, tone=tone, people=list(people), due=due, comments=comments,
                         progress=progress, image=image, **kw)

    def kanban(self, columns, gap=16, col_w=None, **kw):
        """Board: ``columns`` {"To do": [card dicts or titles], ...}; cards take kanban_card keywords."""
        tones = ["accent", "warning", "info", "success", "danger"]
        kw.setdefault("align", "stretch")
        with self.hstack(gap=gap, **kw) as board:
            for ci, (name, cards) in enumerate(columns.items()):
                with self.vstack(gap=10, w=col_w or "fill", pad=(12, 10), surface="inset",
                                 radius=self.T.r["card"]):
                    with self.hstack(gap=8, pad=(0, 4)):
                        self.badge(name, tone=tones[ci % len(tones)], dot=True)
                        self.text(str(len(cards)), role="small", color=self.T.ink3)
                        self.spacer()
                        self.icon("plus", 18, color=self.T.ink3)
                    for c in cards:
                        if isinstance(c, dict):
                            self.kanban_card(**c)
                        else:
                            self.kanban_card(str(c))
        return board

    def timeline(self, items, **kw):
        """Activity feed: items [(time, title, text, icon)] or dicts with time / title / text / icon / tone."""
        return self._add("timeline", items=list(items), **kw)

    activity = timeline

    # ------------------------------------------------------------------ marketing
    def hero(self, title, subtitle=None, actions=("Get started", ("Learn more", "outline")), kicker=None, media=None,
             note=None, layout="split", **kw):
        """Hero section: kicker chip, headline, sub-headline, buttons, note and a ``media`` slot — a Screen
        (product shot), a Canvas, "image", "map", or None. ``layout`` split / center."""
        center = layout == "center" or not self.wide
        with self.hstack(gap=56, align="center", **kw) if (self.wide and not center) else \
                self.vstack(gap=40, align="stretch", **kw) as n:
            with self.vstack(gap=20, align="center" if center else "start", w="fill",
                             max_w=760 if center and self.wide else None, self_align="center" if center else None):
                if kicker:
                    self.badge(kicker, tone="accent", dot=True)
                self.text(title, role="display", size=(60 if self.wide else 36), align="center" if center else "left",
                          lh=1.08)
                if subtitle:
                    self.text(subtitle, role="body", size=18 if self.wide else 16, color=self.T.ink2,
                              align="center" if center else "left", max_w=560)
                if actions:
                    with self.hstack(gap=12, justify="center" if center else "start", w="fill" if center else None):
                        acts = list(actions)
                        for i, a in enumerate(acts):
                            t, v = (a, "primary" if i == 0 else "outline") if isinstance(a, str) else (a[0], a[1])
                            self.button(t, variant=v, size="lg" if self.wide else "md",
                                        icon_right="arrow-right" if i == 0 and v == "primary" else None)
                if note:
                    self.caption(note, color=self.T.ink3, align="center" if center else "left")
            if media is not None:
                fit = {"max_h": 560 if self.wide else 480} if isinstance(media, (Screen, Drawing)) else {}
                self.media(media, w="fill", grow=1.25, **fit)
        return n

    def media(self, media, height=None, **kw):
        """Illustration slot: a Screen (product shot), a Canvas / array, a vector Drawing, "image", "map",
        "video", or None."""
        if isinstance(media, Screen) or isinstance(media, Drawing):
            return self._add("shot", src=media, height=height, **kw)
        if hasattr(media, "buf") or isinstance(media, np.ndarray):
            return self.image(canvas=media, height=height or 320, **kw)
        if media == "map":
            return self._add("map", height=height or 320, **kw)
        if media == "video":
            return self._add("video", height=height, **kw)
        return self._add("image", height=height or 360, seed=kw.pop("seed", 2), **kw)

    def logo_row(self, names, title=None, **kw):
        """Row of (invented) customer wordmarks with a caption above."""
        with self.vstack(gap=20, align="center", **kw) as n:
            if title:
                self.caption(title, color=self.T.ink3, align="center")
            with self.hstack(gap=40, justify="center", wrap=True, w="fill"):
                for i, nm in enumerate(names):
                    self._add("logo", text=nm, mark=i % 6)
        return n

    def feature_grid(self, items, cols=None, **kw):
        """Feature grid: items [(icon, title, text)]."""
        cols = cols or (3 if self.wide else 1)
        with self.grid(cols=cols, gap=24 if self.wide else 16, **kw) as n:
            for it in items:
                ic, t, d = (list(it) + [None, None, None])[:3]
                with self.card(pad=24 if self.wide else 18, gap=10, surface=kw.get("item_surface", "card")):
                    self.icon_badge(ic or "sparkles", size=44)
                    self.spacer(4)
                    self.title(t or "")
                    if d:
                        self.text(d, role="small")
        return n

    def pricing(self, tiers, highlight=None, **kw):
        """Pricing table: tiers [{"name", "price", "period", "desc", "features": [...], "cta"}]; ``highlight``
        = index or name of the recommended tier."""
        cols = len(tiers) if self.wide else 1
        with self.grid(cols=cols, gap=20, **kw) as n:
            for i, t in enumerate(tiers):
                hi = (highlight == i or highlight == t.get("name") or t.get("highlight"))
                with self.card(pad=28 if self.wide else 22, gap=14, border=self.T.accent if hi else None,
                               name=f"tier {t.get('name', i)}"):
                    with self.hstack(gap=8):
                        self.title(t.get("name", "Plan"))
                        self.spacer()
                        if hi:
                            self.badge(t.get("badge", "Most popular"), tone="accent", solid=True)
                    if t.get("desc"):
                        self.text(t["desc"], role="small", max_lines=2)
                    with self.hstack(gap=6, align="end"):
                        self.text(t.get("price", "$0"), role="h1", size=40 if self.wide else 34)
                        if t.get("period", "/mo"):
                            self.text(t.get("period", "/mo"), role="small", color=self.T.ink3, pad=(0, 0, 8, 0))
                    self.button(t.get("cta", "Get started"), variant="primary" if hi else "secondary", full=True)
                    self.divider()
                    for f in t.get("features", []):
                        with self.hstack(gap=10, align="start"):
                            self.icon("tick", 18, color="success" if not self.T.wire else None)
                            self.text(f, role="small", color=self.T.ink)
        return n

    def testimonial(self, quote, name, role=None, rating=5, **kw):
        with self.card(pad=24, gap=16, **kw) as n:
            if rating:
                with self.hstack(gap=2):
                    for i in range(5):
                        self.icon("star", 18, color="#f5b301" if i < rating and not self.T.wire else self.T.ink3,
                                  fill="#f5b301" if i < rating and not self.T.wire else None)
            self.text(f"“{quote}”", role="body", size=17 if self.wide else 15, color=self.T.ink)
            with self.hstack(gap=12):
                self.avatar(name, 40)
                with self.vstack(gap=0):
                    self.label(name)
                    if role:
                        self.caption(role, color=self.T.ink3)
        return n

    def footer(self, brand="Brand", columns=None, note=None, tagline=None, **kw):
        """Site footer: brand + tagline, link columns {title: [links]}, bottom note."""
        T = self.T
        columns = columns or {"Product": ["Features", "Pricing", "Changelog"], "Company": ["About", "Careers",
                                                                                           "Contact"],
                              "Resources": ["Docs", "Guides", "Status"]}
        inner = self.section(bg=T.surface if not T.glass else None, pad=(48 if self.wide else 32, 32), gap=32, **kw)
        with inner:
            with self.hstack(gap=40, align="start", wrap=not self.wide):
                with self.vstack(gap=10, w=280 if self.wide else "fill"):
                    with self.hstack(gap=10):
                        self._add("logo", text=brand, mark=_hash(brand) % 6, color=T.accent)
                    self.text(tagline or "Tools for teams who ship.", role="small", max_w=260)
                if self.wide:
                    self.spacer()
                for title, links in columns.items():
                    with self.vstack(gap=10, w=160 if self.wide else 140):
                        self.label(title)
                        for l in links:
                            self.text(l, role="small")
            self.divider()
            with self.hstack(gap=16):
                self.caption(note or f"© 2026 {brand}. All rights reserved.", color=T.ink3)
                self.spacer()
                for ic in ("globe", "mail", "link"):
                    self.icon(ic, 18, color=T.ink3)
        return inner

    # ------------------------------------------------------------------ media
    def image(self, height=180, seed=1, scene=None, canvas=None, caption=None, sub=None, radius=None, fit="cover",
              aspect=None, badge=None, icon=None, **kw):
        """Image placeholder: soft-gradient abstract landscape (``scene`` dawn ocean forest dusk desert lilac),
        or ``canvas=`` a Canvas / array painted in (cover / contain). ``caption`` over a dark fade."""
        if canvas is not None:
            _IMG_N[0] += 1
            _IMAGES[_IMG_N[0]] = canvas
            canvas = _CanvasRef(_IMG_N[0], canvas)
        return self._add("image", height=height, seed=seed, scene=scene, canvas=canvas, caption=caption, sub=sub,
                         radius=radius, fit=fit, aspect=aspect, badge=badge, icon=icon, **kw)

    def map(self, height=200, seed=3, pin=True, route=True, radius=None, **kw):
        """Map placeholder: water, parks, street grid, highways, a route and a pin."""
        return self._add("map", height=height, seed=seed, pin=pin, route=route, radius=radius, **kw)

    def video(self, height=None, title=None, progress=0.32, time="1:24", duration="4:05", seed=4, canvas=None, **kw):
        """Video player frame: poster, play button, scrubber with times."""
        if canvas is not None:
            _IMG_N[0] += 1
            _IMAGES[_IMG_N[0]] = canvas
            canvas = _CanvasRef(_IMG_N[0], canvas)
        return self._add("video", height=height, title=title, progress=progress, time=time, duration=duration,
                         seed=seed, canvas=canvas, **kw)

    def waveform(self, progress=0.35, height=40, bars=None, seed=2, tone=None, **kw):
        p = dict(progress=progress, height=height, seed=seed, tone=tone)
        if bars:
            p["bars"] = bars
        return self._add("waveform", **p, **kw)

    def audio(self, progress=0.35, time="0:42", duration="2:15", **kw):
        """Voice-message / audio row: play button, waveform, time."""
        with self.hstack(gap=12, **kw) as n:
            self.icon_button("play-fill", variant="filled", size="md")
            self.waveform(progress, height=32, w="fill")
            self.caption(f"{time} / {duration}" if duration else time, color=self.T.ink3)
        return n

    def code(self, code, lang="python", title=None, line_numbers=True, highlight=None, dark=None, **kw):
        """Code block with syntax colours and line numbers (``title`` adds a window bar with the file name)."""
        p = dict(code=code, lang=lang, title=title, line_numbers=line_numbers, highlight=highlight)
        if dark is not None:
            p["dark"] = dark
        return self._add("code", **p, **kw)

    def terminal(self, lines, title="zsh", prompt="~/app $", **kw):
        """Terminal window: lines starting with "$ " are commands; ✓ / error / warn lines are coloured."""
        return self._add("terminal", lines=lines, title=title, prompt=prompt, **kw)

    def chart(self, kind, data, x=None, height=200, scale=None, axes=None, legend="auto", **series):
        """Chart from pixelcanvas.chart in this box: kind line / area / bar / barh / stacked / pie / donut /
        spark / scatter; ``data`` list or {series: values}; ``axes`` dict of Axes options (yfmt, title ...)."""
        scale = scale or (1.3 if self.wide else 1.6)
        p = dict(kind=kind, data=data, x=x, height=height, scale=scale, axes=axes, legend=legend, series=series)
        sizing = {k: p["series"].pop(k) for k in ("w", "h", "name", "max_w", "span", "grow") if k in p["series"]}
        return self._add("chart", **p, **sizing)

    def screenshot(self, src, height=None, frame=True, shadow=True, **kw):
        """Embed another Screen (a product shot in a hero), a vector Drawing, or a Canvas."""
        if hasattr(src, "buf") or isinstance(src, np.ndarray):
            return self.image(canvas=src, height=height, fit="contain", radius=kw.pop("radius", 0), **kw)
        return self._add("shot", src=src, height=height, frame=frame, shadow=shadow, **kw)

    # ------------------------------------------------------------------ painting helpers used by components
    def panel(self, g, x, y, w, h, r=None, kind="card", fill=None, border=None, shadow=None):
        """Draw a surface in this screen's style (card / raised / inset / tint / accent / bar / sheet / plain)."""
        T = self.T
        r = (0.0 if kind == "plain" else T.r["card"]) if r is None else r
        if isinstance(r, (tuple, list)):
            rr = tuple(min(v, w / 2, h / 2) for v in r)
        else:
            rr = min(float(r), w / 2, h / 2)
        f, b, sh = fill, border, None
        glass = False
        if kind == "card":
            f = fill or T.surface
            b = border if border is not None else (T.border if T.card_border else None)
            sh = T.shadow if shadow is None else shadow
            glass = T.glass and fill is None
        elif kind == "raised":
            f = fill or (T.surface if (not T.dark or T.glass) else T.surface2)
            b = border if border is not None else (T.border if T.style != "material" else None)
            sh = T.shadow_lg if shadow is None else shadow
            glass = T.glass and fill is None
            if glass:
                f = fill or ("#ffffff2e" if T.dark else "#ffffffb8")
        elif kind == "sheet":
            f = fill or (T.surface if not T.glass else ("#ffffff26" if T.dark else "#ffffffbf"))
            b = border if border is not None else (T.border if T.glass else None)
            sh = dict(dx=0, dy=-6, blur=24, color="#000000", opacity=0.18 if not T.dark else 0.5)
            rr = (T.r["sheet"], T.r["sheet"], 0, 0) if not isinstance(r, tuple) else rr
            if isinstance(rr, tuple):
                rr = tuple(min(v, w / 2, h / 2) for v in rr)
            glass = T.glass
        elif kind == "inset":
            f = fill or (T.surface2 if not T.glass else T.surface2)
            b = border
            glass = T.glass and fill is None
        elif kind == "tint":
            f = fill or T.accent_soft
        elif kind == "accent":
            f = fill or T.accent
        elif kind == "bar":
            f = fill or T.bar
            b = border if border is not None else (T.border if T.glass else None)
            glass = T.glass
            sh = T.shadow if T.glass else None
        if T.wire:
            sh = None if kind != "raised" else T.shadow_lg
            if kind in ("card", "raised", "sheet") and b is None:
                b = T.border_strong
        if shadow is False:
            sh = None
        if glass:
            self._hook(g, "glass", (x, y, w, h), rr)
        if sh:
            with g.group(None, shadow=sh) as s_:
                s_.rect(x, y, w, h, r=rr, fill=f or T.surface, stroke=b, stroke_width=self.hl if b else None)
        elif f or b:
            g.rect(x, y, w, h, r=rr, fill=f, stroke=b, stroke_width=(self.hl * (1.5 if b == T.accent else 1.0))
                   if b else None)

    def hline(self, g, x0, x1, y, color=None):
        g.line(x0, y, x1, y, stroke=color or self.T.border, stroke_width=self.hl)

    def image_art(self, g, x, y, w, h, seed=1, r=12, scene=None):
        _landscape(self, g, x, y, w, h, seed, r, scene)

    def _hook(self, g, kind, rect, r, **data):
        name = f"backdrop-blur-{id(g) % 99991}" if kind == "glass" else f"uiimg-{data['canvas'].key}"
        hg = g.group(name)
        info = dict(kind=kind, rect=tuple(float(v) for v in rect), r=r, sigma=data.pop("sigma", 18.0))
        if kind == "image":
            info.update(key=data["canvas"].key, fit=data.get("fit", "cover"))
        hg._ui_hook = info
        return hg

    # ------------------------------------------------------------------ layout
    def _register(self, n):
        self._frames.setdefault(n.key(), []).append(n)
        if n.name:
            self._frames.setdefault(str(n.name).lower(), []).append(n)
            t = n.p.get("text") or n.p.get("title") or n.p.get("label")
            auto = f"{n.kind}:{t}" if isinstance(t, str) and t else None
            if auto:
                self._frames.setdefault(auto, []).append(n)

    def _layout_all(self):
        if self._laid:
            return
        self._mcache = {}
        self._frames = {}
        vw = self.vw
        top = self.top_inset
        hh = []
        for n in self._header:
            h = _measure(self, n, vw)[1]
            hh.append(h)
        fh = [_measure(self, n, vw)[1] for n in self._footer]
        body_top = top + sum(hh)
        if self.auto_h:
            natural = _measure(self, self.root, vw)[1]
            fill_min = 0.0
            self.vh = max(body_top + natural + sum(fh) + (self.safe_bottom if not self._footer else 0), fill_min)
        vh = self.vh
        y = top
        for n, h in zip(self._header, hh):
            _layout(self, n, 0, y, vw, h)
            y += h
        yb = vh
        for n, h in zip(reversed(self._footer), reversed(fh)):
            yb -= h
            _layout(self, n, 0, yb, vw, h)
        body_bottom = yb if self._footer else vh - self.safe_bottom
        if self._footer and self.T.glass and self._footer[-1].kind == "tab_bar":
            body_bottom = yb + 6
        self.body = (0.0, body_top, vw, max(body_bottom - body_top, 0.0))
        _layout(self, self.root, 0, body_top, vw, max(body_bottom - body_top, 0.0))
        for n, pl in self.overlays:
            self._place_overlay(n, pl, body_bottom)
        self._laid = True

    def _place_overlay(self, n, pl, body_bottom):
        vw, vh = self.vw, self.vh
        k = pl.get("kind")
        if k == "modal":
            w = min(vw - 48, float(pl.get("w") or (420 if self.wide else 342)))
            h = _measure(self, n, w)[1]
            h = min(h, vh - 80)
            _layout(self, n, (vw - w) / 2, max((vh - h) / 2, 40), w, h)
        elif k == "sheet":
            h = min(_measure(self, n, vw)[1], vh * 0.9)
            _layout(self, n, 0, vh - h, vw, h)
        elif k == "fab":
            w, h = _measure(self, n, vw - 32)
            pos = pl.get("pos", "bottom-right")
            x = vw - 20 - w if "right" in pos else ((vw - w) / 2 if "center" in pos else 20)
            _layout(self, n, x, body_bottom - 20 - h, w, h)
        elif k == "toast":
            w, h = _measure(self, n, min(vw - 32, 420))
            w = min(vw - 32, 420.0) if not self.wide else w
            y = (self.top_inset + 8) if pl.get("pos") == "top" else body_bottom - 16 - h
            _layout(self, n, (vw - w) / 2, y, w, h)
        else:
            tgt = self._find(pl.get("near")) if pl.get("near") else None
            w = pl.get("w")
            w = float(w) if w else (float(n.w) if _num(n.w) else None)
            if w is None:
                w = _measure(self, n, min(vw - 32, 320))[0]
            h = _measure(self, n, w)[1]
            if tgt is not None and tgt.frame is not None:
                fx, fy, fw, fh = tgt.frame
                if pl.get("side") == "above":
                    x, y = fx + fw / 2 - w / 2, fy - h - 4
                else:
                    x, y = fx + fw - w if fx + w > vw - 12 else fx, fy + fh + 6
            else:
                x, y = pl.get("at") or (16, 100)
            x = min(max(x, 12), vw - w - 12)
            y = min(max(y, 8), vh - h - 8)
            _layout(self, n, x, y, w, h)

    def _find(self, name):
        self._layout_all() if not self._laid else None
        if isinstance(name, Node):
            return name
        key = str(name)
        hit = self._frames.get(key) or self._frames.get(key.lower())
        if hit:
            return hit[0]
        low = key.lower()
        for k, ns in self._frames.items():
            if low in k.lower():
                return ns[0]
        return None

    def names(self):
        """Lookup keys of every laid-out component (for ``box``)."""
        self._layout_all()
        return sorted(self._frames)

    # ------------------------------------------------------------------ emission
    def _emit(self, parent, off=(0.0, 0.0), frame=True):
        """Add this screen's layer groups to ``parent`` (Drawing or Group); viewport origin at ``off``."""
        self._layout_all()
        T = self.T
        vw, vh = self.vw, self.vh
        R = float(self.D["radius"])
        dev = self.device
        fr = frame and self.frame
        tr = (float(off[0]), float(off[1]))
        if fr:
            with parent.group("device", translate=tr) as g:
                self._device_body(g)
        vr = R if dev in ("phone", "tablet", "watch", "laptop") else 0.0
        vrad = (0, 0, R, R) if dev in ("browser", "desktop_window") and fr else (vr if fr else 0.0)
        with parent.group("screen", translate=tr) as g:
            if T.glass or (self.backdrop in ("glass",)):
                _backdrop(g, 0, 0, vw, vh, T, "glass", vrad, self.seed)
            else:
                g.rect(0, 0, vw, vh, r=vrad, fill=T.bg)
        vclip = (0, 0, vw, vh)

        def clip_of(rect, rounded_bottom=False):
            x, y, w, h = rect
            if rounded_bottom and vr and y + h > vh - vr:
                return parent.rect(x, y, w, h, r=(0, 0, vr, vr), add=False)
            if isinstance(vrad, tuple) and y + h > vh - R:
                return parent.rect(x, y, w, h, r=vrad, add=False)
            return parent.rect(x, y, w, h, add=False)

        bx, by, bw, bh = self.body
        for n in _kids(self.root):
            if n.frame is None:
                continue
            with parent.group(_safe_name(n.key()), translate=tr, clip=clip_of((bx, by, bw, bh), True)) as g:
                _paint(self, g, n)
        for n in self._header:
            with parent.group(_safe_name(n.key()), translate=tr, clip=clip_of(vclip)) as g:
                _paint(self, g, n)
        for n in self._footer:
            with parent.group(_safe_name(n.key()), translate=tr, clip=clip_of(vclip, True)) as g:
                _paint(self, g, n)
        if self.status_bar:
            with parent.group("status bar", translate=tr) as g:
                self._status(g)
        for n, pl in self.overlays:
            with parent.group(_safe_name(n.key() if n.kind != "stack" else pl.get("kind", "overlay")), translate=tr,
                              clip=clip_of(vclip, True)) as g:
                if pl.get("scrim"):
                    g.rect(0, 0, vw, vh, fill=T.scrim)
                _paint(self, g, n)
        if fr:
            with parent.group("device glass", translate=tr) as g:
                self._device_top(g)
        self._vp_marker = parent.group("viewport-origin", translate=tr)
        self._vp_marker._ui_vp = True

    def _device_body(self, g):
        T = self.T
        vw, vh = self.vw, self.vh
        dev = self.device
        metal = linear_gradient([(0, "#4a4c53"), (0.5, "#1f2024"), (1, "#3a3c42")], start=(0, 0), end=(1, 1))
        if dev == "phone":
            g.rect(-14, -13, vw + 28, vh + 26, r=60, fill=metal)
            g.rect(-12.5, -11.5, vw + 25, vh + 23, r=58.5, fill=None, stroke="#6b6e76", stroke_width=1, opacity=0.6)
            for (yy, hh) in ((122, 32), (182, 62), (258, 62)):
                g.rect(-16.5, yy, 4, hh, r=2, fill="#2b2c31")
            g.rect(vw + 12.5, 210, 4, 98, r=2, fill="#2b2c31")
            g.rect(-9, -9, vw + 18, vh + 18, r=56, fill="#050506")
        elif dev == "tablet":
            g.rect(-20, -20, vw + 40, vh + 40, r=44, fill=metal)
            g.rect(-17, -17, vw + 34, vh + 34, r=41, fill="#050506")
            g.circle(vw / 2, -9, 3, fill="#1b2433")
        elif dev == "watch":
            strap = "#24252b" if T.dark else "#2e3036"
            g.rect(vw * 0.13, -92, vw * 0.74, 110, r=(10, 10, 0, 0), fill=strap)
            g.rect(vw * 0.13, vh - 18, vw * 0.74, 110, r=(0, 0, 10, 10), fill=strap)
            g.rect(-16, -16, vw + 32, vh + 32, r=60, fill=metal)
            g.rect(vw + 14, vh * 0.22, 12, 40, r=5, fill=linear_gradient(["#55575e", "#2a2b30"], start=(0, 0),
                                                                         end=(1, 0)))
            g.rect(vw + 14, vh * 0.52, 6, 48, r=3, fill="#2b2c31")
            g.rect(-8, -8, vw + 16, vh + 16, r=54, fill="#050506")
        elif dev == "laptop":
            g.rect(-26, -26, vw + 52, vh + 46, r=(22, 22, 4, 4), fill="#0d0d10", stroke="#3b3c42", stroke_width=1.5)
            g.circle(vw / 2, -13, 3.2, fill="#1d2430")
            base = linear_gradient([(0, "#d9dce1" if not T.dark else "#5b5e66"), (1, "#a4a8b0" if not T.dark
                                                                                  else "#2f3137")])
            g.rect(-90, vh + 20, vw + 180, 16, r=(4, 4, 14, 14), fill=base)
            g.rect(vw / 2 - 70, vh + 20, 140, 6, r=(0, 0, 6, 6), fill="#00000022")
            g.rect(-90, vh + 20, vw + 180, 2, fill="#ffffff55")
        elif dev in ("browser", "desktop_window"):
            chrome_h = self.D["ext"][1]
            strip = ("#dfe3e8" if not T.dark else "#191c22") if dev == "browser" else \
                ("#eceef1" if not T.dark else "#22262e")
            tool = "#ffffff" if not T.dark else "#262a32"
            g.rect(0, -chrome_h, vw, vh + chrome_h, r=12, fill=strip, stroke="#00000026" if not T.dark
                   else "#ffffff1f", stroke_width=1)
            for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
                g.circle(20 + i * 20, -chrome_h + 20, 6, fill=c)
            if dev == "desktop_window":
                _txt(self, g, vw / 2, -chrome_h / 2, self.window_title or "Untitled", 13, "#4b5563" if not T.dark
                     else "#c4c9d4", True, "mc", maxw=vw - 200)
                ui_line = "#00000022" if not T.dark else "#ffffff1a"
                g.line(0, -0.5, vw, -0.5, stroke=ui_line, stroke_width=1)
                return
            tx = 84
            tw = min(240.0, vw * 0.3)
            g.path(f"M{tx - 8} -48a8 8 0 0 0 8-8v-16a10 10 0 0 1 10-10h{tw - 20}a10 10 0 0 1 10 10v16a8 8 0 0 0 8 8Z",
                   fill=tool)
            _logo_mark(self, g, tx + 12, -76, 16, _hash(self.window_title or "x") % 6, T.accent)
            _txt(self, g, tx + 36, -68, self.window_title or "New Tab", 12.5, "#374151" if not T.dark else "#d6dae2",
                 maxw=tw - 70)
            _ico(self, g, "close", tx + tw - 18, -68, 12, "#6b7280" if not T.dark else "#9aa1ad", sw=2.2)
            _ico(self, g, "plus", tx + tw + 20, -68, 14, "#6b7280" if not T.dark else "#9aa1ad", sw=2.2)
            g.rect(0, -48, vw, 48, fill=tool)
            g.line(0, -0.5, vw, -0.5, stroke="#00000020" if not T.dark else "#ffffff14", stroke_width=1)
            ic = "#5f6672" if not T.dark else "#aab1bd"
            for i, nm in enumerate(("arrow-left", "arrow-right", "refresh")):
                _ico(self, g, nm, 24 + i * 34, -24, 17, ic if i != 1 else alpha(ic, 0.45), sw=2.0)
            ux0, ux1 = 128, vw - 96
            g.rect(ux0, -40, ux1 - ux0, 32, r=16, fill="#eef1f4" if not T.dark else "#1a1d23")
            _ico(self, g, "lock", ux0 + 20, -24, 13, ic, sw=2.2)
            url = self.url or "acme.app"
            _txt(self, g, ux0 + 36, -24, url, 13, "#374151" if not T.dark else "#d6dae2", maxw=ux1 - ux0 - 80)
            _ico(self, g, "star", ux1 - 20, -24, 14, ic, sw=2.0)
            _ico(self, g, "user", vw - 64, -24, 17, ic)
            _ico(self, g, "more-vertical", vw - 30, -24, 17, ic)

    def _device_top(self, g):
        vw, vh = self.vw, self.vh
        T = self.T
        if self.device == "phone":
            g.rect(vw / 2 - 62, 11, 124, 36, r=18, fill="#000000")
            g.circle(vw / 2 + 40, 29, 5, fill="#0d1220")
            g.circle(vw / 2 + 40, 29, 2.2, fill="#1d2a44")
            dark_ui = T.dark or (T.glass and T.dark)
            g.rect(vw / 2 - 67, vh - 13, 134, 5, r=2.5, fill="#ffffff" if dark_ui else "#0b0b0d")

    def _status(self, g):
        T = self.T
        vw = self.vw
        col = T.ink
        dev = self.device
        if dev == "phone":
            _txt(self, g, 78, 30, self.time, 16, col, True, "mc")
            x = vw - 98
            for i, hh in enumerate((4, 6, 8.5, 11)):
                g.rect(x + i * 5, 35.5 - hh, 3.2, hh, r=1, fill=col)
            _ico(self, g, "wifi", vw - 65, 30.5, 17, col, sw=2.4)
            g.rect(vw - 52, 24, 25, 12.5, r=3.8, fill=None, stroke=col, stroke_width=1, opacity=0.45)
            g.rect(vw - 50, 26, 19, 8.5, r=2, fill=col)
            g.rect(vw - 25.6, 28.2, 1.6, 4.2, r=0.8, fill=col, opacity=0.45)
        elif dev == "tablet":
            _txt(self, g, 24, 15, f"{self.time}  Thu 12 Nov", 13, col, True)
            _ico(self, g, "wifi", vw - 70, 15, 15, col, sw=2.4)
            g.rect(vw - 52, 9.5, 24, 11, r=3.5, fill=None, stroke=col, stroke_width=1, opacity=0.45)
            g.rect(vw - 50, 11.5, 17, 7, r=2, fill=col)
        elif dev == "watch":
            _txt(self, g, vw - 22, 20, self.time, 15, col, True, "rc")
            if self.window_title:
                _txt(self, g, 22, 20, self.window_title, 15, T.accent_ink, True, maxw=vw * 0.5)
        elif dev == "laptop":
            g.rect(0, 0, vw, 28, fill=alpha(T.surface if not T.glass else "#ffffff", 0.7 if not T.glass else 0.12))
            _logo_mark(self, g, 14, 7, 14, 1, T.ink)
            x = 40 + _txt(self, g, 40, 14, self.window_title or "Studio", 13, col, True) + 18
            for m in ("File", "Edit", "View", "Window", "Help"):
                x += _txt(self, g, x, 14, m, 13, col) + 18
            _txt(self, g, vw - 16, 14, f"Thu 12 Nov  {self.time}", 13, col, False, "rc")
            _ico(self, g, "wifi", vw - 160, 14, 14, col, sw=2.4)

    def _drawing(self, frame=True, margin=None, k=1.0, background=None, fit=None):
        """The screen as a vector Drawing (device frame + ``margin`` points of backdrop)."""
        self._k = float(k)
        self.hl = 1.0 if k >= 1 else min(1.0 / max(k, 1e-6), 3.0)
        self._layout_all()
        x0, y0, x1, y1 = self._device_box(frame)
        m = self.margin if margin is None else float(margin)
        W, H = x1 - x0 + 2 * m, y1 - y0 + 2 * m
        ox = oy = 0.0
        if fit is not None:                      # grow the backdrop so it covers a whole target of this aspect
            ox, oy = max(fit[0] - W, 0.0) / 2, max(fit[1] - H, 0.0) / 2
            W, H = W + 2 * ox, H + 2 * oy
        bg = background
        if bg is None:
            bg = False if m == 0 else (self.backdrop if isinstance(self.backdrop, str) and self.backdrop.startswith("#")
                                       else None)
        dwg = Drawing(W, H, background=bg if bg not in (None,) else False, name=self.name)
        if m > 0 and bg is None:
            with dwg.group("backdrop") as g:
                _backdrop(g, 0, 0, W, H, self.T, self.backdrop or "soft", 0, self.seed + 7)
        self._emit(dwg, (m - x0 + ox, m - y0 + oy), frame)
        return dwg

    def size(self, frame=True, margin=None):
        """(width, height) in points of the standalone picture."""
        x0, y0, x1, y1 = self._device_box(frame)
        m = self.margin if margin is None else float(margin)
        return (x1 - x0 + 2 * m, y1 - y0 + 2 * m)

    def build(self, k=1.0, frame=True, margin=None):
        """The vector Drawing of this screen (for embedding or custom export)."""
        return self._drawing(frame, margin, k)

    def draw(self, cv, box=None, log=None, frame=True, margin=None):
        """Paint onto Canvas ``cv`` scaled to fit ``box`` (default the whole canvas). One layer per section."""
        if box is None and self._fit_h and not self.auto_h:
            # default-size desktop screens take the canvas aspect: the window fills a 16:9 / 4:3 canvas
            x0, y0, x1, y1 = self._device_box(frame)
            m = self.margin if margin is None else float(margin)
            vh = (self.vw + (x1 - x0 - self.vw) + 2 * m) * cv.h / cv.w - (y1 - y0 - self.vh) - 2 * m
            if vh >= 360 and abs(vh - self.vh) > 1:
                self.vh = float(vh)
                self._laid = False
        W, H = self.size(frame, margin)
        bx0, by0, bx1, by1 = box if box is not None else (0, 0, cv.w, cv.h)
        k = min((bx1 - bx0) / W, (by1 - by0) / H)
        d = self._drawing(frame, margin, k, fit=((bx1 - bx0) / k, (by1 - by0) / k) if box is None else None)
        M0 = _draw_dwg(d, cv, box, log)
        self._set_last(M0)
        return cv

    def _set_last(self, M0):
        mk = self._vp_marker
        self._last = _mm(_mm(M0, _world(mk)), mk.matrix) if mk is not None else M0

    def render(self, scale=2.0, log=None, frame=True, margin=None):
        """New Canvas with this screen at ``scale`` px per point (2 = retina)."""
        from .canvas import Canvas
        W, H = self.size(frame, margin)
        cv = Canvas(max(int(round(W * scale)), 1), max(int(round(H * scale)), 1), (1.0, 1.0, 1.0))
        return self.draw(cv, log=log, frame=frame, margin=margin)

    def to_svg(self, frame=True, margin=None):
        return _svg_with_images(self._drawing(frame, margin, 1.0))

    def save_svg(self, path, frame=True, margin=None):
        """Write the screen as SVG (Canvas images embedded); returns the path."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_svg(frame, margin))
        return path

    def box(self, name):
        """Pixel bbox (x0, y0, x1, y1) of a component on the canvas this screen was last drawn on (before any
        draw: in ``render(scale=1)`` pixels). ``name``: the ``name=`` you gave, "kind:text" (e.g.
        "button:Sign in"), a substring of either, or the Node."""
        n = self._find(name)
        if n is None or n.frame is None:
            raise KeyError(f"no component {name!r}; known: {', '.join(self.names()[:40])}")
        x, y, w, h = n.frame
        M = self._last
        if M is None:
            x0, y0, _, _ = self._device_box(True)
            m = self.margin
            M = (1.0, 0.0, 0.0, 1.0, m - x0, m - y0)
        pts = [_ap(M, px, py) for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return (int(math.floor(min(xs))), int(math.floor(min(ys))), int(math.ceil(max(xs))), int(math.ceil(max(ys))))

    def center(self, name):
        x0, y0, x1, y1 = self.box(name)
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class _CanvasRef:
    """Reference to a Canvas registered for draw-time painting (keeps drawings light to copy)."""

    def __init__(self, key, canvas):
        self.key = key
        self.w = canvas.w if hasattr(canvas, "w") else np.asarray(canvas).shape[1]
        self.h = canvas.h if hasattr(canvas, "h") else np.asarray(canvas).shape[0]

    def __deepcopy__(self, memo):
        return self


# --------------------------------------------------------------------------------------------- presentation


class Presentation:
    """Several screens side by side on a soft backdrop with shadows (app-store / case-study shot).

    ``Presentation(screens, size=(1920, 1080), background="soft", tilt=0, stagger=0.0, gap=None, title=None,
    subtitle=None, labels=None, theme=None, accent=None, pad=0.08)``

    * ``background``: "soft" (gradient + colour glows), "vivid", "flat", or a colour.
    * ``tilt``: degrees of in-plane rotation, or "iso" for a lying-on-the-desk isometric view.
    * ``stagger``: raise every other screen by this fraction of the height.
    """

    def __init__(self, screens, size=(1920, 1080), background="soft", tilt=0, stagger=0.0, gap=None, title=None,
                 subtitle=None, labels=None, theme=None, accent=None, pad=0.08, shadow=True, name="presentation"):
        self.screens = list(screens)
        if not self.screens:
            raise ValueError("Presentation needs at least one Screen")
        s0 = self.screens[0]
        self.W, self.H = float(size[0]), float(size[1])
        self.T = tokens(theme or s0.T.theme, "modern" if not s0.T.wire else "wireframe", accent or s0.T.accent)
        self.background, self.tilt, self.stagger = background, tilt, float(stagger)
        self.gap, self.title, self.subtitle, self.labels = gap, title, subtitle, labels
        self.pad, self.shadow, self.name = float(pad), shadow, name
        for i, s in enumerate(self.screens):
            if s.name == f"{s.device} screen":
                s.name = f"screen {i + 1}"

    def _layout(self):
        W, H = self.W, self.H
        pad = self.pad * H
        top = pad + (H * 0.13 if self.title else 0.0)
        bottom = pad + (H * 0.05 if self.labels else 0.0)
        boxes = [s._device_box(True) for s in self.screens]
        dws = [b[2] - b[0] for b in boxes]
        dhs = [b[3] - b[1] for b in boxes]
        n = len(boxes)
        iso = self.tilt == "iso"
        gap = self.gap if self.gap is not None else max(W * (0.045 if not iso else 0.01), 32.0 if not iso else 0.0)
        avail_w = W - 2 * pad - gap * (n - 1)
        avail_h = H - top - bottom - self.stagger * H
        k = min(avail_w / sum(dws), avail_h / max(dhs))
        if iso:
            k *= 0.9
        tot = sum(dws) * k + gap * (n - 1)
        x = (W - tot) / 2
        out = []
        for i, (b, dw, dh) in enumerate(zip(boxes, dws, dhs)):
            off = (self.stagger * H / 2) * (1 if i % 2 else -1) if self.stagger else 0.0
            if n % 2 == 1 and self.stagger:
                off = -self.stagger * H / 2 if i == n // 2 else self.stagger * H / 2
            cy = top + avail_h / 2 + self.stagger * H / 2 + off
            out.append((x + dw * k / 2, cy, k, b))
            x += dw * k + gap
        return out

    def _drawing(self, kpx=1.0):
        T = self.T
        dwg = Drawing(self.W, self.H, background=T.bg if not str(self.background).startswith("#") else
                      self.background, name=self.name)
        with dwg.group("backdrop") as g:
            _backdrop(g, 0, 0, self.W, self.H, T, self.background, 0, 11)
        if self.title:
            with dwg.group("title") as g:
                _txt(self.screens[0], g, self.W / 2, self.pad * self.H + self.H * 0.035, self.title,
                     self.H * 0.042, T.ink, True, "mc", ls=-0.5)
                if self.subtitle:
                    _txt(self.screens[0], g, self.W / 2, self.pad * self.H + self.H * 0.085, self.subtitle,
                         self.H * 0.02, T.ink2, False, "mc")
        lay = self._layout()
        iso = self.tilt == "iso"
        for i, (s, (cx, cy, k, b)) in enumerate(zip(self.screens, lay)):
            s._k = kpx * k
            s.hl = 1.0 if s._k >= 1 else min(1.0 / max(s._k, 1e-6), 3.0)
            s._laid = False
            dcx, dcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            sh = dict(dx=0, dy=26, blur=34, color="#0b0f1a", opacity=0.32 if not T.dark else 0.6) \
                if self.shadow else None
            if iso:
                with dwg.group(s.name, translate=(cx, cy), scale=(1.0, 0.62), shadow=dict(sh, dy=40, blur=40)
                               if sh else None) as g0:
                    with g0.group(None, rotate=-38, scale=k) as g1:
                        s._emit(g1, (-dcx, -dcy), True)
            else:
                rot = float(self.tilt or 0)
                with dwg.group(s.name, translate=(cx, cy), rotate=rot, scale=k, shadow=sh) as g0:
                    s._emit(g0, (-dcx, -dcy), True)
        if self.labels:
            with dwg.group("labels") as g:
                for lab, (cx, cy, k, b) in zip(self.labels, lay):
                    _txt(self.screens[0], g, cx, self.H - self.pad * self.H * 0.9, lab, self.H * 0.018, T.ink2, True,
                         "mc")
        return dwg

    def build(self, k=1.0):
        return self._drawing(k)

    def draw(self, cv, box=None, log=None):
        """Paint onto ``cv`` (fit into ``box``); one layer per screen, plus backdrop / title / labels."""
        bx0, by0, bx1, by1 = box if box is not None else (0, 0, cv.w, cv.h)
        kpx = min((bx1 - bx0) / self.W, (by1 - by0) / self.H)
        d = self._drawing(kpx)
        M0 = _draw_dwg(d, cv, box, log)
        for s in self.screens:
            s._set_last(M0)
        return cv

    def render(self, scale=1.0, log=None):
        from .canvas import Canvas
        cv = Canvas(int(round(self.W * scale)), int(round(self.H * scale)))
        return self.draw(cv, log=log)

    def to_svg(self):
        return _svg_with_images(self._drawing(1.0))

    def save_svg(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_svg())
        return path

    def box(self, name, screen=None):
        """Pixel bbox of a component; ``screen`` index / name, or "screen 2/button:Save" style names."""
        if screen is None and isinstance(name, str) and "/" in name:
            screen, name = name.split("/", 1)
        cands = self.screens
        if screen is not None:
            cands = [self.screens[screen]] if isinstance(screen, int) else [s for s in self.screens
                                                                           if s.name == screen]
        for s in cands:
            if s._find(name) is not None:
                return s.box(name)
        raise KeyError(f"no component {name!r}")


def showcase(screens, **kw):
    """``Presentation(screens, **kw)`` — several screens on a presentation backdrop."""
    return Presentation(screens, **kw)


# --------------------------------------------------------------------------------------------- annotations


def annotate(cv, target, notes, kind="callout", theme=None, log=None, **kw):
    """Label components of a drawn Screen / Presentation on ``cv``: ``notes`` {component name: text} or
    [(name, text), ...]. ``kind`` callout (leader lines, pixelcanvas.diagram) / marker (numbered badges +
    legend). Returns the list of (name, bbox)."""
    items = list(notes.items()) if isinstance(notes, dict) else list(notes)
    th = theme or (target.T.theme if hasattr(target, "T") else "light")
    pts, out = [], []
    for nm, text in items:
        b = target.box(nm)
        out.append((nm, b))
        pts.append((((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0), str(text)))
    try:
        from . import diagram as _dg
        if kind == "marker":
            _dg.markers(cv, pts, theme=th, log=log, **kw)
        else:
            _dg.callouts(cv, pts, theme=th, log=log, **kw)
        return out
    except ImportError:
        pass
    S = cv.h / 1080.0
    dwg = Drawing(cv.w, cv.h, background=False, name="annotations")
    acc = target.T.accent if hasattr(target, "T") else DEFAULT_ACCENT
    with dwg.group("annotations") as g:
        for i, ((px, py), text) in enumerate(pts):
            g.circle(px, py, 15 * S, fill=acc, stroke="#ffffff", stroke_width=2.5 * S)
            g.text(px, py, str(i + 1), size=15 * S, fill="#ffffff", anchor="mc", weight="bold")
            g.text(px + 22 * S, py, text, size=16 * S, fill="#101828", anchor="lc", halo="#ffffff")
    dwg.draw(cv, log=log)
    return out

# --------------------------------------------------------------------------------------------- templates


def _c(content, defaults):
    d = dict(defaults)
    d.update(content or {})
    return d


def _wide_shell(s, c, active, items=None):
    """Sidebar + main column for wide templates; returns the main column node (context manager)."""
    items = items or [("home", "Overview"), ("chart", "Analytics"), ("users", "Customers"), ("cart", "Orders", 12),
                      ("message", "Messages", 3), "# Workspace", ("folder", "Projects"), ("gear", "Settings")]
    row = s.hstack(gap=0, align="stretch", h="fill")
    s._stack.append(row)
    s.sidebar(items, active=active, brand=c.get("brand", "Northwave"), user=(c.get("user", "Ana Silva"),
                                                                              "ana@northwave.io"))
    main = s.vstack(gap=24, pad=(28, 32), w="fill", scroll=True)
    s._stack.pop()
    return main


def _t_login(s, c, wide):
    c = _c(c, dict(brand="Northwave", title="Welcome back", subtitle="Sign in to continue to your workspace.",
                   email="ana@northwave.io", cta="Sign in"))
    outer = s.vstack(gap=0, align="center", justify="center", h="fill", pad=(0, 24)) if wide else s.vstack(gap=16)
    with outer:
        box = s.card(pad=32, gap=16, w=400) if wide else s.vstack(gap=16)
        with box:
            s.spacer(8 if not wide else 0)
            with s.hstack(gap=10):
                s._add("logo", text=c["brand"], color=s.T.accent)
            s.spacer(4)
            s.heading(c["title"], level=1 if not wide else 2)
            s.text(c["subtitle"], role="body", color=s.T.ink2)
            s.spacer(4)
            s.input("Email", value=c["email"], icon="mail")
            s.input("Password", value="••••••••••", icon="lock", trailing="eye", focused=True)
            with s.hstack():
                s.checkbox("Remember me", checked=True)
                s.spacer()
                s.link("Forgot password?", size=14)
            s.button(c["cta"], full=True, size="lg")
            s.divider(label="or continue with")
            with s.hstack(gap=12):
                s.button("Google", variant="outline", icon="globe", w="fill")
                s.button("Passkey", variant="outline", icon="key", w="fill")
            with s.hstack(gap=4, justify="center", w="fill"):
                s.small("New here?")
                s.link("Create an account", size=13)


def _t_dashboard(s, c, wide):
    c = _c(c, dict(title="Overview", user="Ana Silva", kpis=[("Revenue", "$48,210", "+12.4%", "dollar"),
                                                               ("Orders", "1,284", "+8.1%", "cart"),
                                                               ("Visitors", "32.6k", "-2.3%", "users"),
                                                               ("Conversion", "3.9%", "+0.4%", "target")]))
    spark = [[3, 4, 3.6, 5, 4.8, 6, 6.4], [5, 4, 5, 6, 5.5, 7, 7.2], [7, 6.5, 6, 6.2, 5.8, 5.5, 5.4],
             [2, 2.4, 2.2, 2.8, 3, 3.1, 3.4]]
    if wide:
        with _wide_shell(s, c, 0):
            with s.hstack(gap=12):
                with s.vstack(gap=4, w="fill"):
                    s.heading(c["title"], level=2)
                    s.small("Thursday, 12 November 2026")
                s.search("Search", w=260)
                s.icon_button("bell", variant="surface", badge=True)
                s.button("Export", icon="download", variant="secondary")
            with s.grid(cols=4, gap=16):
                for (l, v, d, ic), sp in zip(c["kpis"], spark):
                    s.stat(l, v, delta=d, icon=ic, spark=sp, note="vs last week")
            with s.hstack(gap=16, align="stretch"):
                with s.card(pad=20, gap=8, w="fill", grow=2):
                    with s.hstack():
                        s.title("Revenue")
                        s.spacer()
                        s.segmented(["7D", "30D", "90D"], active=1, w=180, size="sm")
                    s.chart("area", {"2026": [32, 38, 35, 44, 41, 52, 49, 58, 61, 57, 66, 72],
                                     "2025": [28, 30, 31, 33, 34, 36, 38, 37, 40, 42, 41, 45]},
                            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                            height=230, axes=dict(yfmt="$k"))
                with s.card(pad=20, gap=8, w="fill"):
                    s.title("Sales by channel")
                    s.chart("donut", {"Search": 38, "Direct": 26, "Social": 21, "Email": 15}, height=230)
            with s.card(pad=0, gap=0):
                with s.hstack(pad=(16, 20)):
                    s.title("Recent orders")
                    s.spacer()
                    s.button("View all", variant="ghost", size="sm", icon_right="arrow-right")
                s.table(["Customer", "Order", "Date", "Status", "Amount"],
                        [[{"avatar": "Liam Carter", "sub": "liam@mail.com"}, "#3021", "12 Nov", {"badge": "Paid",
                                                                                               "tone": "success"},
                          "$249.00"],
                         [{"avatar": "Maya Chen", "sub": "maya@mail.com"}, "#3020", "12 Nov",
                          {"badge": "Pending", "tone": "warning"}, "$89.50"],
                         [{"avatar": "Noah Patel", "sub": "noah@mail.com"}, "#3019", "11 Nov",
                          {"badge": "Refunded", "tone": "danger"}, "$1,120.00"]], card=False)
        return
    s.app_bar(c["title"], leading=f"avatar:{c['user']}", trailing=["search", "bell"])
    with s.grid(cols=2, gap=12):
        for (l, v, d, ic), sp in zip(c["kpis"], spark):
            s.stat(l, v, delta=d, icon=ic)
    with s.card(pad=16, gap=8):
        with s.hstack():
            s.title("Revenue")
            s.spacer()
            s.badge("+12.4%", tone="success")
        s.chart("area", [32, 38, 35, 44, 41, 52, 49, 58, 61, 57, 66, 72], height=150, axes=dict(yfmt="$k"))
    with s.list_group("Recent orders"):
        s.list_row("Liam Carter", "#3021 · Paid", leading="avatar:Liam Carter", value="$249.00", value_strong=True)
        s.list_row("Maya Chen", "#3020 · Pending", leading="avatar:Maya Chen", value="$89.50", value_strong=True)
    s.tab_bar([("home", "Home"), ("chart", "Stats"), ("cart", "Orders"), ("user", "Profile")], active=0)


def _t_settings(s, c, wide):
    c = _c(c, dict(name="Ana Silva", email="ana@northwave.io"))
    if wide:
        with _wide_shell(s, c, 6):
            s.heading("Settings", level=2)
            s.tabs(["Profile", "Account", "Notifications", "Billing", "Security"], active=2)
            with s.hstack(gap=24, align="start"):
                with s.vstack(gap=6, w=280):
                    s.title("Notifications")
                    s.small("Choose what we email you about and how often.")
                with s.card(pad=(4, 20), gap=0, w="fill", dividers=True):
                    for t, sub, on in (("Product updates", "New features and improvements", True),
                                       ("Weekly digest", "A summary of your workspace every Monday", True),
                                       ("Mentions", "When someone mentions you in a comment", True),
                                       ("Marketing", "Tips, offers and events", False)):
                        s.toggle(t, on=on, sub=sub, pad=(14, 0))
            with s.hstack(gap=24, align="start"):
                with s.vstack(gap=6, w=280):
                    s.title("Appearance")
                    s.small("Theme and density of the interface.")
                with s.card(pad=20, gap=16, w="fill"):
                    s.segmented(["Light", "Dark", "System"], active=2)
                    s.slider(0.6, label="Interface density", value_text="Comfortable")
                    s.select("Language", value="English (UK)")
            with s.hstack(gap=12, justify="end", w="fill"):
                s.button("Cancel", variant="secondary")
                s.button("Save changes")
        return
    s.app_bar("Settings", large=True)
    with s.card(pad=16, gap=0):
        with s.hstack(gap=14):
            s.avatar(c["name"], 56)
            with s.vstack(gap=2, w="fill"):
                s.title(c["name"])
                s.small(c["email"])
            s.icon("chevron-right", 18, color=s.T.ink3)
    with s.list_group("Preferences"):
        s.list_row("Notifications", leading="bell", tone="danger", chevron=True, value="On", solid=True)
        s.list_row("Dark mode", leading="moon", tone="info", toggle=s.T.dark, solid=True)
        s.list_row("Language", leading="globe", tone="success", value="English", chevron=True, solid=True)
    with s.list_group("Account"):
        s.list_row("Privacy", leading="lock", tone="accent", chevron=True, solid=True)
        s.list_row("Payment methods", leading="credit-card", tone="warning", chevron=True, solid=True)
        s.list_row("Help & support", leading="help", tone="info", chevron=True, solid=True)
    s.button("Sign out", variant="ghost", icon="logout", self_align="center")
    s.tab_bar([("home", "Home"), ("search", "Search"), ("bell", "Inbox"), ("gear", "Settings")], active=3)


def _t_chat(s, c, wide):
    c = _c(c, dict(contact="Maya Chen", status="Online", messages=[
        (False, "Hey! Did you get a chance to look at the new onboarding flow?", "9:12"),
        (True, "Yes — the progress steps feel much clearer now.", "9:14"),
        (False, "Great. I moved the plan picker to step two.", "9:15"),
        (True, "Love it. Can you share the prototype link?", "9:16"),
        (False, "Sending it now \U0001f680" if False else "Sending it now.", "9:16")]))
    if wide:
        row = s.hstack(gap=0, align="stretch", h="fill")
        with row:
            s.sidebar([("message", "Inbox", 4), ("users", "Contacts"), ("bookmark", "Saved"), ("gear", "Settings")],
                      brand="Northwave", user=("Ana Silva", "ana@northwave.io"), w=220)
            with s.vstack(gap=0, w=320, h="fill", surface="plain", bg=s.T.surface, pad=(16, 12)):
                s.search("Search messages")
                s.spacer(8)
                for i, (nm, last, tm, b) in enumerate((("Maya Chen", "Sending it now.", "9:16", 2),
                                                       ("Liam Carter", "Deploy is green", "8:40", None),
                                                       ("Design team", "Noah: new icons are in", "Yesterday", 5),
                                                       ("Sofia Rossi", "Thanks!", "Tue", None))):
                    s.list_row(nm, last, leading=f"avatar:{nm}", meta=tm, badge=b,
                               bg=s.T.accent_soft if i == 0 else None, pad=(0, 8))
            with s.vstack(gap=0, w="fill", h="fill"):
                with s.hstack(gap=12, pad=(14, 24), bg=s.T.surface):
                    s.avatar(c["contact"], 40, status="online")
                    with s.vstack(gap=0, w="fill"):
                        s.title(c["contact"])
                        s.caption(c["status"], color="success")
                    s.icon_button("phone-call")
                    s.icon_button("video")
                    s.icon_button("more-vertical")
                with s.scroll(gap=10, pad=(20, 24)):
                    s.divider(label="Today")
                    for me, t, tm in c["messages"]:
                        s.message(t, me=me, time=tm, avatar=None if me else c["contact"])
                with s.hstack(gap=10, pad=(12, 24)):
                    s.icon_button("paperclip")
                    s.input(placeholder="Write a message…", w="fill")
                    s.icon_button("send", variant="filled")
        return
    s.app_bar(c["contact"], leading="back", subtitle=c["status"], trailing=["phone-call", "video"])
    with s.scroll(gap=10, pad=(0, 0)):
        s.divider(label="Today")
        for me, t, tm in c["messages"]:
            s.message(t, me=me, time=tm)
        s.audio(0.4, "0:12", "0:38", w="fill")
    with s.hstack(gap=10):
        s.icon_button("plus", variant="surface")
        s.input(placeholder="Message", w="fill", trailing="mic", field_h=44)
        s.icon_button("send", variant="filled")


def _t_feed(s, c, wide):
    c = _c(c, dict(title="Discover", posts=[("Sofia Rossi", "2h", "Morning light over the lake — shot on a 35mm.",
                                             "dawn", "1.2k", "86"),
                                            ("Liam Carter", "5h", "New studio, new plants, same coffee.", "forest",
                                             "642", "31")]))
    if not wide:
        s.app_bar(c["title"], trailing=["search", "bell"], center=False)
    with (s.vstack(gap=20, max_w=640, pad=(24, 0), self_align="center") if wide else s.vstack(gap=16)):
        if wide:
            s.heading(c["title"], level=2)
        with s.hstack(gap=14, pad=(0, 0)):
            for nm in ("Your story", "Maya Chen", "Noah Patel", "Sofia Rossi", "Liam Carter")[:5 if wide else 4]:
                with s.vstack(gap=6, align="center", w=64):
                    s.avatar(nm, 58, ring=s.T.accent)
                    s.caption(nm.split()[0], align="center", max_lines=1)
        s.chips(["For you", "Following", "Design", "Travel", "Food"], selected=[0])
        for i, (nm, tm, text, scene, likes, comments) in enumerate(c["posts"]):
            with s.card(pad=14, gap=12):
                with s.hstack(gap=10):
                    s.avatar(nm, 36)
                    with s.vstack(gap=0, w="fill"):
                        s.label(nm)
                        s.caption(tm, color=s.T.ink3)
                    s.icon("more", 20, color=s.T.ink2)
                s.text(text, role="small", color=s.T.ink)
                s.image(height=200 if not wide else 320, scene=scene, seed=i + 3, radius=s.T.r["card"] - 4)
                with s.hstack(gap=18):
                    with s.hstack(gap=6):
                        s.icon("heart", 20, color="danger")
                        s.small(likes)
                    with s.hstack(gap=6):
                        s.icon("message", 20, color=s.T.ink2)
                        s.small(comments)
                    s.icon("share", 20, color=s.T.ink2)
                    s.spacer()
                    s.icon("bookmark", 20, color=s.T.ink2)
    if not wide:
        s.tab_bar([("home", "Home"), ("search", "Explore"), ("plus", "Post"), ("heart", "Activity"),
                   ("user", "Profile")], active=0)


def _t_profile(s, c, wide):
    c = _c(c, dict(name="Maya Chen", role="Product designer · Lisbon", bio="Designing calm software for busy "
                   "teams. Previously at two startups you have not heard of yet.",
                   stats=[("Projects", "48"), ("Followers", "12.4k"), ("Following", "312")]))
    if not wide:
        s.app_bar("", leading="back", trailing=["share-up", "more"])
    with (s.vstack(gap=20, max_w=720, pad=(32, 0), self_align="center") if wide else s.vstack(gap=16)):
        s.image(height=140 if not wide else 200, scene="lilac", seed=5)
        with s.vstack(gap=8, align="center"):
            s.avatar(c["name"], 96, ring=s.T.bg, status="online")
            s.heading(c["name"], level=2, align="center")
            s.small(c["role"], align="center")
            s.text(c["bio"], role="small", align="center", color=s.T.ink2, max_w=320)
        with s.hstack(gap=12, justify="center", w="fill"):
            s.button("Follow", icon="plus", w="fill" if not wide else None)
            s.button("Message", variant="secondary", icon="message", w="fill" if not wide else None)
        with s.card(pad=(14, 8), gap=0):
            with s.grid(cols=3, gap=0):
                for k, v in c["stats"]:
                    with s.vstack(gap=2, align="center"):
                        s.title(v, align="center")
                        s.caption(k, align="center", color=s.T.ink3)
        s.tabs(["Work", "Collections", "About"], active=0, fill=True)
        with s.grid(cols=3 if wide else 2, gap=10):
            for i, sc in enumerate(["dawn", "ocean", "dusk", "forest", "desert", "lilac"][:6 if wide else 4]):
                s.image(height=130, scene=sc, seed=i + 11, radius=s.T.r["small"] + 4)


def _t_checkout(s, c, wide):
    c = _c(c, dict(items=[("Aero running shoe", "Size 42 · Sand", "$129.00", "desert"),
                          ("Merino crew socks", "2 pairs", "$24.00", "lilac")], total="$162.40"))
    if not wide:
        s.app_bar("Checkout", leading="back")
    with (s.hstack(gap=32, align="start", pad=(40, 48)) if wide else s.vstack(gap=16)):
        with s.vstack(gap=16, w="fill", grow=1.4):
            if wide:
                s.breadcrumb(["Cart", "Shipping", "Payment"])
                s.heading("Payment", level=2)
            s.segmented(["Card", "Apple Pay", "PayPal"] if False else ["Card", "Wallet", "Bank"], active=0)
            with s.card(pad=18, gap=14):
                s.input("Card number", value="4242 4242 4242 4242", icon="credit-card")
                with s.hstack(gap=12):
                    s.input("Expiry", value="08 / 28", w="fill")
                    s.input("CVC", value="•••", w="fill", trailing="help")
                s.input("Name on card", value="Ana Silva")
                s.checkbox("Billing address same as shipping", checked=True)
        with s.card(pad=18, gap=12, w="fill" if wide else None):
            s.title("Order summary")
            for nm, sub, price, sc in c["items"]:
                with s.hstack(gap=12):
                    s.image(height=52, w=52, scene=sc, seed=_hash(nm) % 9, radius=10)
                    with s.vstack(gap=2, w="fill"):
                        s.label(nm, max_lines=1)
                        s.caption(sub, color=s.T.ink3)
                    s.label(price)
            s.divider()
            for k, v in (("Subtotal", "$153.00"), ("Shipping", "Free"), ("Tax", "$9.40")):
                with s.hstack():
                    s.small(k)
                    s.spacer()
                    s.small(v, color=s.T.ink)
            s.divider()
            with s.hstack():
                s.title("Total")
                s.spacer()
                s.title(c["total"])
            s.button(f"Pay {c['total']}", full=True, size="lg", icon="lock")
            s.caption("Payments are encrypted end to end.", align="center", color=s.T.ink3)


def _t_onboarding(s, c, wide):
    c = _c(c, dict(title="Plan your week in minutes", text="Bring tasks, events and notes into one calm timeline "
                   "that adapts as your day changes.", step=1, steps=3))
    with s.vstack(gap=18, align="center", h="fill", pad=(24 if wide else 8, 8)):
        s.spacer()
        with s.vstack(gap=0, w="fill"):
            s.image(height=300 if not wide else 340, scene="lilac", seed=8, radius=28)
        s.spacer(8)
        with s.hstack(gap=6, justify="center", w="fill"):
            for i in range(c["steps"]):
                s._add("spacer", w=24 if i == c["step"] - 1 else 8, h=8, dot="on" if i == c["step"] - 1 else "off")
        s.heading(c["title"], level=1, align="center")
        s.text(c["text"], role="body", align="center", color=s.T.ink2)
        s.spacer()
        s.button("Get started", full=True, size="lg")
        s.button("I already have an account", variant="ghost", full=True)


def _t_landing(s, c, wide):
    c = _c(c, dict(brand="Northwave", title="Ship product updates your customers actually read",
                   subtitle="Northwave turns release notes into beautiful in-app announcements, emails and a public "
                            "changelog — from one editor.",
                   kicker="New: AI summaries", logos=["Halcyon", "Quillstack", "Orbital", "Fernhill", "Larkspur"]))
    s.navbar(c["brand"], links=["Product", "Customers", "Pricing", "Docs"], actions=[("Sign in", "ghost"),
                                                                                      "Start free"] if wide else
             ["Start free"])
    with s.section(pad=(72 if wide else 40, 32 if wide else 20)):
        shot = template("dashboard", "browser", theme=s.T.theme, accent=s.T.accent, size=(1280, 760))
        s.hero(c["title"], c["subtitle"], kicker=c["kicker"], note="Free 14-day trial · No credit card",
               media=shot if wide else None, layout="split" if wide else "center")
    with s.section(pad=(24, 32)):
        s.logo_row(c["logos"][:5 if wide else 4], title="Trusted by product teams at")
    with s.section(pad=(64 if wide else 40, 32 if wide else 20), gap=32):
        with s.vstack(gap=12, align="center"):
            s.overline("Features", align="center")
            s.heading("Everything you need to announce", level=1 if wide else 2, align="center")
        s.feature_grid([("edit", "One editor", "Write once, publish to in-app, email and web."),
                        ("sparkles", "AI summaries", "Turn long changelogs into crisp highlights."),
                        ("target", "Targeting", "Show updates to the right segment at the right time."),
                        ("chart", "Analytics", "See reads, clicks and reactions per release."),
                        ("globe", "Localised", "Translate every post into 30 languages."),
                        ("shield", "SSO & audit", "Enterprise controls without the setup pain.")][:6 if wide else 3])
    with s.section(pad=(64 if wide else 40, 32 if wide else 20), gap=32, bg=s.T.surface if not s.T.glass else None):
        with s.vstack(gap=12, align="center"):
            s.overline("Pricing", align="center")
            s.heading("Simple plans that grow with you", level=1 if wide else 2, align="center")
        s.pricing(_PRICING, highlight=1)
    s.footer(c["brand"])


_PRICING = [dict(name="Starter", price="$0", period="/mo", desc="For side projects and small teams.",
                 features=["1 project", "In-app widget", "Public changelog"], cta="Start free"),
            dict(name="Growth", price="$49", period="/mo", desc="For growing products with many releases.",
                 features=["10 projects", "Email digests", "AI summaries", "Segments"], cta="Start trial"),
            dict(name="Scale", price="$199", period="/mo", desc="For companies with many teams.",
                 features=["Unlimited projects", "SSO & audit log", "Priority support"], cta="Contact sales")]


def _t_pricing(s, c, wide):
    c = _c(c, dict(title="Pricing that scales with you", tiers=_PRICING, highlight=1))
    if wide:
        s.navbar(c.get("brand", "Northwave"), links=["Product", "Customers", "Pricing", "Docs"],
                 actions=[("Sign in", "ghost"), "Start free"], active="Pricing")
    else:
        s.app_bar("Plans", leading="close")
    with s.section(pad=(56 if wide else 8, 32 if wide else 0), gap=28):
        with s.vstack(gap=12, align="center"):
            s.heading(c["title"], level=1 if wide else 2, align="center")
            s.text("Start free. Upgrade when your team grows.", align="center", color=s.T.ink2)
            s.segmented(["Monthly", "Yearly · save 20%"], active=0, w=300)
        s.pricing(c["tiers"], highlight=c["highlight"])


def _t_music(s, c, wide):
    c = _c(c, dict(track="Northern Lights", artist="Aurora Fields", album="Night Drive", progress=0.38,
                   time="1:24", duration="3:42"))
    s.app_bar("Now playing", leading="chevron-down", trailing=["more"], subtitle=c["album"])
    with s.vstack(gap=20, pad=(12, 8), h="fill"):
        s.image(aspect=1.0, scene="dusk", seed=3, radius=28)
        with s.hstack(gap=12):
            with s.vstack(gap=2, w="fill"):
                s.heading(c["track"], level=3, max_lines=1)
                s.small(c["artist"])
            s.icon("heart", 24, color="accent")
        with s.vstack(gap=6):
            s.waveform(c["progress"], height=36)
            with s.hstack():
                s.caption(c["time"], color=s.T.ink3)
                s.spacer()
                s.caption(c["duration"], color=s.T.ink3)
        with s.hstack(gap=0, justify="between", w="fill"):
            s.icon_button("shuffle", color=s.T.ink2)
            s.icon_button("skip-back", size="lg")
            s.icon_button("pause", variant="filled", size=72)
            s.icon_button("skip-forward", size="lg")
            s.icon_button("repeat", color="accent")
        s.spacer()
        with s.hstack(gap=12):
            s.icon("volume", 18, color=s.T.ink3)
            s.slider(0.7, w="fill")
            s.icon("headphones", 18, color=s.T.ink3)


def _t_weather(s, c, wide):
    c = _c(c, dict(city="Lisbon", temp="21°", cond="Partly cloudy", hi="24°", lo="15°"))
    with s.vstack(gap=16, pad=(8, 0)):
        with s.hstack():
            s.icon("pin", 18, color="accent")
            s.title(c["city"])
            s.spacer()
            s.icon_button("sliders", variant="surface", size="sm")
        with s.vstack(gap=0, align="center"):
            s.icon("cloud-sun", 96, color="accent")
            s.text(c["temp"], role="display", size=88, align="center", lh=1.0)
            s.text(c["cond"], role="title", align="center", color=s.T.ink2)
            s.small(f"H {c['hi']}  ·  L {c['lo']}", align="center")
        with s.card(pad=16, gap=12):
            s.caption("HOURLY FORECAST", color=s.T.ink3)
            with s.hstack(gap=0, justify="between", w="fill"):
                for tm, ic, t in (("Now", "cloud-sun", "21°"), ("13", "sun", "23°"), ("14", "sun", "24°"),
                                  ("15", "cloud", "22°"), ("16", "rain", "19°"), ("17", "rain", "18°")):
                    with s.vstack(gap=8, align="center", w=44):
                        s.caption(tm, align="center", color=s.T.ink2)
                        s.icon(ic, 24, color="accent" if ic in ("sun", "cloud-sun") else s.T.ink2)
                        s.label(t, align="center")
        with s.grid(cols=2, gap=12):
            for ic, lab, v, sub in (("wind", "Wind", "12 km/h", "NW breeze"), ("drop", "Humidity", "64%",
                                                                                 "Dew point 14°"),
                                    ("sun", "UV index", "5", "Moderate"), ("thermometer", "Feels like", "22°",
                                                                           "Similar to actual")):
                with s.card(pad=14, gap=6):
                    with s.hstack(gap=6):
                        s.icon(ic, 16, color=s.T.ink3)
                        s.caption(lab.upper(), color=s.T.ink3)
                    s.heading(v, level=3)
                    s.caption(sub, color=s.T.ink2)


def _t_calendar(s, c, wide):
    c = _c(c, dict(year=2026, month=11, selected=12, events={3: 1, 7: 2, 12: 3, 18: 1, 21: 2, 26: 1}))
    ev = [("09:00", "Design review", "Q4 roadmap · Room 2", "accent"), ("11:30", "Lunch with Maya", "Café Lumen",
                                                                         "success"),
          ("14:00", "Sprint planning", "Product team · 45 min", "warning"), ("17:15", "Yoga", "Studio North",
                                                                             "info")]
    body = _wide_shell(s, c, 1, [("home", "Home"), ("calendar", "Calendar"), ("tick", "Tasks", 5),
                                 ("users", "People"), ("gear", "Settings")]) if wide else s.vstack(gap=16)
    if not wide:
        s.app_bar("Calendar", leading="menu", trailing=["search", "plus"])
    with body:
        with (s.hstack(gap=24, align="start") if wide else s.vstack(gap=16)):
            with s.card(pad=16, gap=8, w=380 if wide else None):
                s.calendar(c["year"], c["month"], selected=c["selected"], today=c["selected"], events=c["events"])
            with s.vstack(gap=12, w="fill"):
                with s.hstack():
                    s.title("Thursday, 12 November")
                    s.spacer()
                    s.badge("4 events", tone="neutral")
                for tm, t, sub, tone in ev:
                    with s.hstack(gap=12, align="stretch"):
                        s.caption(tm, w=44, color=s.T.ink3)
                        with s.card(pad=(10, 14), gap=2, w="fill", surface="tint", bg=s.T.soft(s.T.tone(tone))):
                            s.label(t)
                            s.caption(sub, color=s.T.ink2)
    if not wide:
        s.fab("plus")


def _t_kanban(s, c, wide):
    c = _c(c, dict(title="Website relaunch", columns={
        "To do": [dict(title="Write launch announcement", tag="Content", tone="info", people=["Maya Chen"],
                       due="Nov 18", comments=3),
                  dict(title="Audit image weights on the home page", tag="Perf", tone="warning",
                       people=["Noah Patel"], due="Nov 20")],
        "In progress": [dict(title="New pricing page", tag="Design", tone="accent", people=["Sofia Rossi",
                                                                                             "Liam Carter"],
                             progress=0.6, comments=8, image=True),
                        dict(title="Migrate blog to MDX", tag="Eng", tone="success", people=["Liam Carter"],
                             progress=0.3)],
        "Review": [dict(title="Hero illustration v3", tag="Design", tone="accent", people=["Maya Chen"],
                        due="Nov 14", comments=5)],
        "Done": [dict(title="Set up analytics events", tag="Eng", tone="success", people=["Noah Patel"],
                      due="Nov 9")]}))
    if wide:
        with _wide_shell(s, c, 5):
            with s.hstack(gap=12):
                with s.vstack(gap=4, w="fill"):
                    s.breadcrumb(["Projects", "Marketing", c["title"]])
                    s.heading(c["title"], level=2)
                s.avatar_group(["Maya Chen", "Noah Patel", "Sofia Rossi", "Liam Carter", "Ana Silva"], max=4)
                s.button("Share", variant="secondary", icon="share-up")
                s.button("New task", icon="plus")
            s.tabs(["Board", "List", "Timeline", "Files"], active=0)
            s.kanban(c["columns"])
        return
    s.app_bar(c["title"], leading="back", trailing=["filter", "more"])
    cols = dict(list(c["columns"].items())[:2])
    with s.scroll(gap=0):
        s.kanban(cols, col_w=290)
    s.fab("plus")


TEMPLATES = {"login": _t_login, "dashboard": _t_dashboard, "settings": _t_settings, "chat": _t_chat, "feed": _t_feed,
             "profile": _t_profile, "checkout": _t_checkout, "onboarding": _t_onboarding, "landing": _t_landing,
             "pricing": _t_pricing, "music_player": _t_music, "weather": _t_weather, "calendar": _t_calendar,
             "kanban": _t_kanban}


def template(name, screen="phone", theme="light", accent=None, style="modern", content=None, **screen_kw):
    """A finished screen with believable sample content: ``name`` one of TEMPLATES (login dashboard settings chat
    feed profile checkout onboarding landing pricing music_player weather calendar kanban); ``screen`` phone /
    browser / tablet / laptop / ...; ``content`` dict overrides the sample copy (keys per template, e.g. title,
    brand, kpis, messages, items). Returns the Screen — add more components, then draw / render / save_svg."""
    if name not in TEMPLATES:
        from ._args import unknown_name
        raise unknown_name("template", name, TEMPLATES)
    kw = dict(screen_kw)
    if name in ("landing", "pricing") and screen in ("browser", "plain") and "height" not in kw and "size" not in kw:
        kw["height"] = "auto"
    if screen == "browser":
        kw.setdefault("url", {"landing": "northwave.io", "pricing": "northwave.io/pricing"}.get(
            name, f"app.northwave.io/{name.replace('_', '-')}"))
        kw.setdefault("title", {"landing": "Northwave — release notes", "pricing": "Pricing · Northwave"}.get(
            name, f"{name.replace('_', ' ').title()} · Northwave"))
    s = Screen(screen, theme=theme, accent=accent, style=style, **kw)
    if s.wide and name in ("music_player", "weather", "onboarding"):       # phone-shaped content: centre a column
        side = max((s.vw - 440) / 2, 24)
        s.root.p["pad"] = (24.0, side, 24.0, side)
        s.root.p["gap"] = 16.0
    TEMPLATES[name](s, content, s.wide)
    return s


# ---- scene manifest: one entry per screen and per named top-level section (not per glyph) ----
from . import manifest as _manifest  # noqa: E402

_STRUCTURAL = {"backdrop", "device", "screen", "status bar", "device glass", "device shadow"}


def _after_screen(cap, args, res):
    s = args.get("self")
    tiles = cap.tiles
    shell = [t for t in tiles if t[0] in _STRUCTURAL and t[0] != "backdrop"]
    out = []
    u = _manifest._union_tiles(shell or tiles)
    if u is not None:
        out.append(dict(kind="screen", name=str(getattr(s, "name", "screen")), crop=u,
                        meta={"device": getattr(s, "device", "")}))
    for nm, x0, y0, a in tiles:
        if nm not in _STRUCTURAL:
            out.append(dict(kind="section", name=nm, crop=(a, x0, y0), meta={"screen": getattr(s, "name", "")}))
    out += _component_entries(s, cap.cv)
    return out or None


def _after_presentation(cap, args, res):
    p = args.get("self")
    names = {getattr(s, "name", None) for s in getattr(p, "screens", [])}
    out = []
    for nm, x0, y0, a in cap.tiles:
        kind = "screen" if nm in names else ("text" if nm in ("title", "labels") else "shape")
        meta = {"presentation": getattr(p, "name", "")}
        if nm == "backdrop":
            meta["backdrop"] = True
        out.append(dict(kind=kind, name=nm, crop=(a, x0, y0), meta=meta))
    for s in getattr(p, "screens", []):
        out += _component_entries(s, cap.cv)
    return out or None


# components a request names ("a tab bar", "a big circular timer", "three cards") -> (entry name, aliases)
COMPONENT_NAMES = {
    "ring": ("ring", ["progress ring", "timer", "dial", "gauge", "circle", "circular", "countdown"]),
    "tab_bar": ("tab bar", ["tabs", "bottom bar", "navigation", "nav bar", "tab", "menu"]),
    "app_bar": ("app bar", ["top bar", "toolbar", "header"]),
    "navbar": ("navigation bar", ["navbar", "nav", "menu bar", "header", "navigation"]),
    "sidebar": ("sidebar", ["side bar", "navigation", "menu"]),
    "card": ("card", ["tile", "panel"]),
    "button": ("button", ["cta", "call to action"]),
    "icon_button": ("button", ["icon button"]),
    "fab": ("button", ["floating action button", "fab"]),
    "list_row": ("list row", ["row", "list", "item", "entry"]),
    "avatar": ("avatar", ["profile picture", "profile photo", "portrait", "user photo"]),
    "avatar_group": ("avatar group", ["avatars", "people"]),
    "chart": ("chart", ["graph", "plot"]),
    "terminal": ("terminal", ["console", "shell", "terminal window"]),
    "code": ("code editor", ["code", "editor", "snippet"]),
    "stat": ("stat", ["kpi", "metric", "stat tile"]),
    "calendar": ("calendar", ["month view", "date picker"]),
    "map": ("map", ["route"]),
    "image": ("image", ["photo", "picture"]),
    "video": ("video player", ["video", "player"]),
    "progress": ("progress bar", ["progress"]),
    "toggle": ("toggle", ["switch"]),
    "slider": ("slider", []),
    "table": ("table", ["data table"]),
    "message": ("message", ["chat bubble", "bubble"]),
    "notification": ("notification", ["alert"]),
    "kcard": ("kanban card", ["card", "task"]),
    "timeline": ("timeline", []),
    "waveform": ("waveform", ["audio"]),
    "chip": ("chip", ["tag", "pill"]),
    "badge": ("badge", ["tag", "pill"]),
    "segmented": ("segmented control", ["toggle"]),
    "tabs": ("tabs", ["tab"]),
    "field": ("text field", ["input", "field", "form"]),
    "search": ("search field", ["search", "search bar"]),
}


def _walk_nodes(s):
    stack = [s.root] + list(s._header) + list(s._footer) + [n for n, _ in s.overlays]
    seen = set()
    while stack:
        n = stack.pop(0)
        if id(n) in seen:
            continue
        seen.add(id(n))
        yield n
        stack.extend(_kids(n))


def _component_entries(s, cv):
    """One manifest entry per kind of named component on a drawn screen: name, aliases, count, union mask."""
    if cv is None or not hasattr(cv, "buf"):
        return []
    groups, order = {}, []
    for n in _walk_nodes(s):
        kind = "card" if n.p.get("surface") == "card" and n.kind == "stack" else n.kind
        if kind not in COMPONENT_NAMES or n.frame is None:
            continue
        try:
            b = s.box(n)
        except Exception:                                # noqa: BLE001 - recording never breaks a drawing
            continue
        name, aliases = COMPONENT_NAMES[kind]
        if name not in groups:
            groups[name] = dict(boxes=[], aliases=list(aliases), words=[])
            order.append(name)
        g = groups[name]
        g["boxes"].append(b)
        for key in ("label", "text", "title"):
            v = n.p.get(key)
            if isinstance(v, str) and v and len(v) < 40 and v not in g["words"]:
                g["words"].append(v)
    out = []
    H, W = cv.h, cv.w
    for name in order:
        g = groups[name]
        m = np.zeros((H, W), np.float32)
        for x0, y0, x1, y1 in g["boxes"]:
            m[max(int(y0), 0):max(min(int(y1), H), 0), max(int(x0), 0):max(min(int(x1), W), 0)] = 1.0
        if not m.any():
            continue
        meta = {"screen": getattr(s, "name", ""), "count": len(g["boxes"]), "see_through": True,
                "component": name}
        out.append(dict(kind="object", name=name, mask=m, alias=g["aliases"] + ["component"], meta=meta))
    return out


# --------------------------------------------------------------------------------------------- windows
# A standalone window that hugs its content and fits any box of any canvas: terminal, browser, code editor, card.

WINDOW_KINDS = ("terminal", "browser", "code", "card")
_WINDOW_NAMES = {"terminal": ("terminal window", ["terminal", "console", "shell", "window", "command line"]),
                 "browser": ("browser window", ["browser", "web page", "website", "window"]),
                 "code": ("code editor", ["editor", "code", "snippet", "window"]),
                 "card": ("card", ["panel", "window"])}


def _window_palette(theme, accent):
    dark = theme == "dark"
    if dark:
        P = dict(bg="#0e1420", bar="#182030", fg="#e6edf3", mut="#7d8aa0", border="#ffffff1a", prompt="#3fd18b",
                 flag="#8ab4ff", str="#f2c97d", ok="#3fd18b", err="#ff6b6b", warn="#f2c94c", cursor="#e6edf3",
                 url="#222b3c", page="#f7f8fb", sub="#1a2232")
    else:
        P = dict(bg="#ffffff", bar="#eef1f5", fg="#1b2333", mut="#6b7486", border="#0000001f", prompt="#16a34a",
                 flag="#2563eb", str="#b45309", ok="#16a34a", err="#dc2626", warn="#d97706", cursor="#1b2333",
                 url="#ffffff", page="#ffffff", sub="#f3f5f8")
    if accent:
        P["prompt"] = accent
        P["ok"] = accent if theme == "dark" else P["ok"]
    return P


def _cmd_tokens(cmd):
    out = []
    for i, m in enumerate(re.finditer(r'"[^"]*"?|\'[^\']*\'?|\S+|\s+', cmd)):
        s = m.group()
        if s.isspace():
            out.append((s, "fg"))
        elif s[0] in "\"'":
            out.append((s, "str"))
        elif s.startswith("-"):
            out.append((s, "flag"))
        elif not out or all(k == "fg" and t.isspace() for t, k in out):
            out.append((s, "prog"))
        else:
            out.append((s, "fg"))
    return out


class WindowBox(tuple):
    """Pixel box (x0, y0, x1, y1) of a drawn window, plus ``kind``, ``content`` (box of the content area) and
    ``commands`` (number of command lines of a terminal)."""

    def __new__(cls, box, kind, content, commands=0):
        self = super().__new__(cls, tuple(int(round(v)) for v in box))
        self.kind, self.content, self.commands = kind, tuple(int(round(v)) for v in content), int(commands)
        return self


def _window_drawing(kind, content, title, theme, accent, prompt, cursor, font_size, width, shadow, radius, url):
    P = _window_palette(theme, accent)
    fs = float(font_size)
    mono = "mono"
    lh = round(fs * 1.75)
    pad = round(fs * 1.5)
    bar = round(fs * 2.6) if kind != "card" else 0
    lines = content
    if isinstance(lines, str):
        lines = lines.rstrip("\n").split("\n")
    lines = list(lines or [])
    cw = _tw("M" * 20, fs, False, mono) / 20.0
    # ---- natural content size
    if kind == "terminal":
        pr = str(prompt)
        rows = [str(s) for s in lines] + ([None] if cursor else [])
        widths = [(_tw(pr + " " + s[2:], fs, False, mono) if s.startswith("$ ") else _tw(s, fs, False, mono))
                  for s in rows if s is not None]
        cw_need = max(widths + [cw * 28])
        ch_need = len(rows) * lh
    elif kind == "code":
        code_lines = [str(s).expandtabs(4) for s in lines] or [""]
        lnw = len(str(len(code_lines))) * cw + fs * 1.6
        cw_need = lnw + max([cw * len(s) for s in code_lines] + [cw * 30])
        ch_need = len(code_lines) * lh
    elif kind == "browser":
        page = content if isinstance(content, dict) else {}
        if isinstance(content, (list, tuple)) and content and isinstance(content[0], str):
            page = {"title": content[0], "text": " ".join(str(s) for s in content[1:])}
        blank = bool(page.get("blank"))
        ttl = str(page.get("title", title or "Build faster"))
        cw_need = max(_tw(ttl, fs * 2.6, True) + 2 * pad, fs * 34)
        ch_need = fs * 21.5
    else:                                                # card
        body = [str(s) for s in lines]
        ttl = title or ""
        cw_need = max([_tw(s, fs, False) for s in body] + [_tw(ttl, fs * 1.35, True), fs * 18])
        ch_need = (fs * 1.35 * 1.6 if ttl else 0) + len(body) * fs * 1.6
    W = float(width) if width else round(cw_need + 2 * pad)
    H = round(bar + pad + ch_need + pad * (0.85 if kind in ("terminal", "code") else 1.0))
    m = round(fs * 3.2) if shadow else round(fs * 0.5)
    r = float(radius) if radius is not None else round(fs * 0.85)
    d = Drawing(W + 2 * m, H + 2 * m, background=False, name=_WINDOW_NAMES[kind][0])
    sh = dict(dx=0, dy=fs * 1.1, blur=fs * 2.0, color="#05070d", opacity=0.45 if theme == "dark" else 0.22) \
        if shadow else None
    with d.group("window frame", shadow=sh) as g:
        g.rect(m, m, W, H, r=r, fill=P["bg"] if kind != "browser" else P["page"], stroke=P["border"],
               stroke_width=max(fs / 16, 1))
        if bar:
            g.rect(m, m, W, bar, r=(r, r, 0, 0), fill=P["bar"])
            g.line(m, m + bar, m + W, m + bar, stroke=P["border"], stroke_width=max(fs / 16, 1))
            for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
                g.circle(m + pad * 0.9 + i * fs * 1.3, m + bar / 2, fs * 0.38, fill=c)
    cx0, cy0 = m + pad, m + bar + pad
    commands = 0
    with d.group("window content") as g:
        if kind == "terminal":
            if bar:
                g.text(m + W / 2, m + bar / 2, title or "zsh", size=fs * 0.82, fill=P["mut"], anchor="mc",
                       weight="bold")
            y = cy0 + lh / 2
            cols = dict(P, prog=mix(P["fg"], "#ffffff", 0.3) if theme == "dark" else "#0f172a")
            for s in rows:
                if s is None or s.startswith("$ "):
                    x = cx0
                    g.text(x, y, prompt, size=fs, fill=P["prompt"], anchor="lc", family=mono, weight="bold")
                    x += _tw(str(prompt) + " ", fs, True, mono)
                    if s is None:
                        g.rect(x, y - fs * 0.62, cw * 0.95, fs * 1.24, fill=P["cursor"], opacity=0.85)
                    else:
                        commands += 1
                        for t, k in _cmd_tokens(s[2:]):
                            if not t.isspace():
                                g.text(x, y, t, size=fs, fill=cols.get(k, P["fg"]), anchor="lc", family=mono,
                                       weight="bold" if k == "prog" else "normal")
                            x += _tw(t, fs, False, mono) if not t.isspace() else cw * len(t)
                else:
                    low = s.lower()
                    col = P["mut"]
                    if s.startswith(("✓", "✔", "ok", "done", "+")) or "success" in low or "passed" in low:
                        col = P["ok"]
                    elif s.startswith(("✗", "✘", "error", "ERR")) or "failed" in low:
                        col = P["err"]
                    elif s.startswith(("warn", "!")):
                        col = P["warn"]
                    g.text(cx0, y, s, size=fs, fill=col, anchor="lc", family=mono)
                y += lh
        elif kind == "code":
            SP = _syntax_palette(theme == "dark")
            if bar:
                tw_ = _tw(title or "main.py", fs * 0.82, True) + fs * 2.4
                g.rect(m + pad * 0.9 + fs * 4.6, m + bar * 0.22, tw_, bar * 0.78, r=(fs * 0.4, fs * 0.4, 0, 0),
                       fill=P["bg"])
                g.text(m + pad * 0.9 + fs * 4.6 + tw_ / 2, m + bar * 0.61, title or "main.py", size=fs * 0.82,
                       fill=P["fg"], anchor="mc", weight="bold")
            y = cy0 + lh / 2
            for i, ln in enumerate(code_lines):
                g.text(cx0 + lnw - fs * 1.1, y, str(i + 1), size=fs, fill=SP["ln"], anchor="rc", family=mono)
                col = 0
                for t, k in _tokenize(ln):
                    if t.strip():                # a leading NBSP keeps edge glyphs like ( and " whole
                        g.text(cx0 + lnw + (col - 1) * cw, y, " " + t, size=fs, fill=SP.get(k, SP["fg"]),
                               anchor="lc", family=mono)
                    col += len(t)
                y += lh
        elif kind == "browser":
            ux0 = m + pad * 0.9 + fs * 4.8
            g.rect(ux0, m + bar * 0.2, W - (ux0 - m) - pad, bar * 0.6, r=bar * 0.3, fill=P["url"],
                   stroke=P["border"], stroke_width=max(fs / 16, 1))
            g.text(ux0 + fs * 1.1, m + bar / 2, url or "example.com", size=fs * 0.8, fill=P["mut"], anchor="lc")
            acc = accent or "#6366f1"
            ink, mut = "#111827", "#6b7280"
            if not blank:
                y = cy0 + fs * 0.8
                g.text(cx0, y, ttl, size=fs * 2.6, fill=ink, anchor="lt", weight="bold")
                y += fs * 3.8
                txt = str(page.get("text", "Ship pages, docs and dashboards from one place."))
                g.text(cx0, y, txt, size=fs * 1.2, fill=mut, anchor="lt", max_width=W - 2 * pad)
                y += fs * 3.0
                btn = str(page.get("button", "Get started"))
                bw = _tw(btn, fs, True) + fs * 2.4
                g.rect(cx0, y, bw, fs * 2.4, r=fs * 1.2, fill=acc)
                g.text(cx0 + bw / 2, y + fs * 1.2, btn, size=fs, fill=_on(acc), anchor="mc", weight="bold")
                y += fs * 4.0
                cwid = (W - 2 * pad - 2 * fs) / 3
                for i in range(3):
                    x = cx0 + i * (cwid + fs)
                    g.rect(x, y, cwid, m + H - pad - y, r=fs * 0.6, fill=P["sub"])
                    g.rect(x + fs, y + fs, fs * 2, fs * 2, r=fs * 0.5, fill=mix(acc, "#ffffff", 0.75))
                    g.rect(x + fs, y + fs * 4, cwid * 0.6, fs * 0.6, r=fs * 0.3, fill="#d7dbe3")
                    g.rect(x + fs, y + fs * 5.2, cwid * 0.4, fs * 0.6, r=fs * 0.3, fill="#e3e6ec")
        else:
            y = cy0
            if ttl:
                g.text(cx0, y, ttl, size=fs * 1.35, fill=P["fg"], anchor="lt", weight="bold")
                y += fs * 1.35 * 1.6
            for s in body:
                g.text(cx0, y, s, size=fs, fill=P["mut"], anchor="lt")
                y += fs * 1.6
    return d, (m, m, m + W, m + H), (cx0, cy0, m + W - pad, m + H - pad), commands


def window(cv, kind="terminal", content=None, box=None, title=None, theme="dark", accent=None, prompt="$",
           cursor=True, font_size=16, width=None, shadow=True, radius=None, url=None, align="center", log=None):
    """A standalone window drawn into ``box`` of ``cv`` (fractions or pixels; default a centred box). The window
    hugs its content and keeps its proportions: the box only sets where and how large it is.

    ``kind``: terminal / browser / code / card. ``content``: terminal and code take a list of lines (or one string;
    terminal lines starting with "$ " are commands, others output: ✓ / error / warn lines are coloured); browser
    takes ``{"title", "text", "button"}`` or a Canvas / array shown as the page; card takes body lines.
    ``title``: window title (terminal), file name (code) or card heading. ``theme`` dark / light; ``accent``:
    prompt / button colour; ``prompt``: the prompt string ("$", "~/app ❯"); ``cursor``: blinking-block line at the
    end; ``font_size``: points (the whole window scales with the box); ``width``: fixed width in points;
    ``align``: center / left / right / top / bottom inside the box. Returns a ``WindowBox`` (pixel box)."""
    _check = __import__("pixelcanvas._args", fromlist=["check_choice"]).check_choice
    _check("ui.window", "kind", kind, WINDOW_KINDS)
    _check("ui.window", "theme", theme, ("dark", "light"))
    _check("ui.window", "align", align, ("center", "left", "right", "top", "bottom"))
    img = None
    if kind == "browser" and (hasattr(content, "buf") or isinstance(content, np.ndarray)):
        img, content = content, {"blank": True}
    if content is None:
        content = {"terminal": ["$ git clone https://github.com/acme/app", "$ cd app && make", "✓ built in 1.2s"],
                   "code": ["def hello(name):", "    return f\"Hello, {name}!\""],
                   "browser": {}, "card": ["A short description."]}[kind]
    d, wb, cb, ncmd = _window_drawing(kind, content, title, theme, accent, prompt, cursor, font_size, width, shadow,
                                      radius, url)
    if box is None:
        box = (0.15, 0.15, 0.85, 0.85)
    x0, y0, x1, y1 = (float(v) for v in box)
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0, x1, y0, y1 = x0 * cv.w, x1 * cv.w, y0 * cv.h, y1 * cv.h
    k = min((x1 - x0) / d.width, (y1 - y0) / d.height)
    fw, fh = d.width * k, d.height * k
    ox = x0 if align == "left" else (x1 - fw if align == "right" else x0 + (x1 - x0 - fw) / 2)
    oy = y0 if align == "top" else (y1 - fh if align == "bottom" else y0 + (y1 - y0 - fh) / 2)
    d.draw(cv, log=log, box=(ox, oy, ox + fw, oy + fh))
    px = lambda b: (ox + b[0] * k, oy + b[1] * k, ox + b[2] * k, oy + b[3] * k)          # noqa: E731
    W_, C_ = px(wb), px(cb)
    if img is not None:
        from .canvas import Canvas as _Cv
        from . import sdf as _sdf
        bar_px = (C_[1] - W_[1]) - 1.5 * float(font_size) * k
        vx0, vy0 = int(np.floor(W_[0] + 0.5)), int(np.floor(W_[1] + bar_px + 0.5))
        vx1, vy1 = int(np.ceil(W_[2] - 0.5)), int(np.ceil(W_[3] - 0.5))
        src = img.buf if hasattr(img, "buf") else np.asarray(img, np.float32)
        im = Image.fromarray((np.clip(src[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
        tw_, th_ = max(vx1 - vx0, 1), max(vy1 - vy0, 1)
        sc = max(tw_ / im.width, th_ / im.height)                      # cover: fill the page, crop the overflow
        iw, ih = max(int(round(im.width * sc)), tw_), max(int(round(im.height * sc)), th_)
        im = im.resize((iw, ih), _RES.LANCZOS)
        cx_, cy_ = (iw - tw_) // 2, (ih - th_) // 2
        im = im.crop((cx_, cy_, cx_ + tw_, cy_ + th_))
        arr = np.asarray(im, np.float32) / 255.0
        yy, xx = np.mgrid[0:arr.shape[0], 0:arr.shape[1]].astype(np.float32)
        rr = (float(radius) if radius is not None else round(float(font_size) * 0.85)) * k
        dd = _sdf.sd_box(xx + 0.5, yy + 0.5, arr.shape[1] / 2, arr.shape[0] / 2 - rr, arr.shape[1] / 2,
                         arr.shape[0] / 2 + rr, radius=rr)
        cv.paste(arr, vx0, vy0, mask=_sdf.fill(dd))
        del _Cv
    return WindowBox(W_, kind, C_, ncmd)


def _after_window(cap, args, res):
    u = _manifest._union_tiles(cap.tiles)
    if u is None or res is None:
        return None
    kind = args.get("kind", "terminal")
    name, aliases = _WINDOW_NAMES.get(kind, ("window", []))
    out = [dict(kind="object", name=name, crop=u, alias=aliases, meta={"window": kind})]
    if getattr(res, "commands", 0):
        out.append(dict(kind="object", name="command", bbox=res.content, alias=["commands", "command line"],
                        meta={"count": int(res.commands), "see_through": True}))
    return out


from ._args import strict as _strict_fn  # noqa: E402

window = _strict_fn(_manifest.drawn("object", after=_after_window)(window), "ui.window")


_manifest.wrap_method(Screen, "draw", "screen", after=_after_screen)
_manifest.wrap_method(Presentation, "draw", "screen", after=_after_presentation)
