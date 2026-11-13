# -*- coding: utf-8 -*-
"""Layout: boxes, grids and type stacks that keep designs tidy, plus complete typographic templates.

* ``Layout(cv, margin=0.06)`` → the canvas minus a margin, as a ``Box``. Boxes split into ``rows`` / ``cols`` /
  ``grid`` / ``split`` / ``golden`` / ``thirds`` / ``inset`` and pass straight to ``cv.text(box=...)``.
* ``stack(cv, box, items)`` sets a list of text items (kicker, title, subtitle, body, footer ...) into a box with
  consistent spacing and returns their bboxes.
* ``template(name, cv, title=..., ...)`` draws a finished layout from content alone — poster, flyer, book_cover,
  magazine, business_card, quote_card, thumbnail, slide, certificate, menu, badge, label — and returns the boxes it
  used (``boxes["image"]`` is the free area for a picture).
"""
from __future__ import annotations

import math

import numpy as np

from . import text as _t
from ._args import (reject_unknown as _reject_unknown, unknown_name as _unknown_name, check_choice as _check_choice,
                    signature_options as _sig_options, strict as _strict, strict_methods as _strict_methods)
from .color import F32, lerp, mix, rgb

__all__ = ["Box", "Layout", "stack", "align_boxes", "distribute", "template", "TEMPLATES", "THEMES", "theme", "THEME_KEYS",
           "role_color"]

PHI = (1 + 5 ** 0.5) / 2


# --------------------------------------------------------------------------------------------- boxes

class Box(tuple):
    """An (x0, y0, x1, y1) rectangle in pixels with layout helpers. Works anywhere a bbox tuple does."""

    def __new__(cls, x0, y0=None, x1=None, y1=None):
        if y0 is None:
            x0, y0, x1, y1 = x0
        return super().__new__(cls, (float(x0), float(y0), float(x1), float(y1)))

    x0 = property(lambda s: s[0])
    y0 = property(lambda s: s[1])
    x1 = property(lambda s: s[2])
    y1 = property(lambda s: s[3])
    w = property(lambda s: s[2] - s[0])
    h = property(lambda s: s[3] - s[1])
    cx = property(lambda s: (s[0] + s[2]) / 2)
    cy = property(lambda s: (s[1] + s[3]) / 2)
    center = property(lambda s: ((s[0] + s[2]) / 2, (s[1] + s[3]) / 2))
    short = property(lambda s: min(s[2] - s[0], s[3] - s[1]))

    def __repr__(self):
        return "Box(%.0f, %.0f, %.0f, %.0f)" % tuple(self)

    def _len(self, v, ref):
        v = float(v)
        return v * ref if abs(v) <= 1.0 else v

    def inset(self, left, top=None, right=None, bottom=None):
        """Shrink by margins. Values ≤ 1 are fractions of the box's shorter side, larger values are pixels.
        One value = all sides; two = (horizontal, vertical); four = left, top, right, bottom."""
        if top is None:
            top = right = bottom = left
        elif right is None:
            right, bottom = left, top
        s = self.short
        l_, t_, r_, b_ = (self._len(v, s) for v in (left, top, right, bottom))
        return Box(self[0] + l_, self[1] + t_, self[2] - r_, self[3] - b_)

    def pad(self, v):
        """Grow by ``v`` (fraction of the shorter side, or pixels)."""
        d = self._len(v, self.short)
        return Box(self[0] - d, self[1] - d, self[2] + d, self[3] + d)

    def sub(self, u0, v0, u1, v1):
        """Sub-box by fractions of this box (0~1)."""
        return Box(self[0] + u0 * self.w, self[1] + v0 * self.h, self[0] + u1 * self.w, self[1] + v1 * self.h)

    def point(self, u, v):
        """(x, y) at fractions (u, v) of the box."""
        return (self[0] + u * self.w, self[1] + v * self.h)

    def _parts(self, weights, total, gap):
        if isinstance(weights, int):
            weights = [1] * weights
        weights = [float(w) for w in weights]
        g = self._len(gap, self.short) if gap else 0.0
        avail = total - g * (len(weights) - 1)
        s = sum(weights) or 1.0
        out, pos = [], 0.0
        for w in weights:
            size = avail * w / s
            out.append((pos, pos + size))
            pos += size + g
        return out

    def rows(self, weights=2, gap=0.0):
        """Split top-to-bottom: ``rows(3)`` or ``rows([2, 1, 1], gap=0.02)``."""
        return [Box(self[0], self[1] + a, self[2], self[1] + b) for a, b in self._parts(weights, self.h, gap)]

    def cols(self, weights=2, gap=0.0):
        """Split left-to-right: ``cols(3)`` or ``cols([1, 2], gap=24)``."""
        return [Box(self[0] + a, self[1], self[0] + b, self[3]) for a, b in self._parts(weights, self.w, gap)]

    def grid(self, rows=2, cols=2, gap=0.0):
        """rows x cols cells as a list of rows (``g[r][c]``)."""
        return [r.cols(cols, gap) for r in self.rows(rows, gap)]

    def cells(self, rows=2, cols=2, gap=0.0):
        """Same as ``grid`` but flat, row by row."""
        return [c for r in self.grid(rows, cols, gap) for c in r]

    def split(self, frac=0.5, axis="y", gap=0.0):
        """Two boxes split at ``frac`` along ``axis`` ("y" = top / bottom, "x" = left / right)."""
        _check_choice("Box.split", "axis", axis, ("x", "y"))
        w = [frac, 1 - frac]
        return tuple(self.rows(w, gap) if axis == "y" else self.cols(w, gap))

    def golden(self, axis="x", first="large", gap=0.0):
        """Golden-ratio split (0.618 : 0.382); ``first="small"`` puts the small part first."""
        _check_choice("Box.golden", "first", first, ("large", "small"))
        f = 1 / PHI if first == "large" else 1 - 1 / PHI
        return self.split(f, axis, gap)

    def thirds(self):
        """Rule-of-thirds guides: dict(x=[x1, x2], y=[y1, y2], points=[4 intersections], cells=3x3 rows)."""
        xs = [self[0] + self.w / 3, self[0] + 2 * self.w / 3]
        ys = [self[1] + self.h / 3, self[1] + 2 * self.h / 3]
        return dict(x=xs, y=ys, points=[(x, y) for y in ys for x in xs], cells=self.grid(3, 3))

    def take(self, side, amount, gap=0.0):
        """Cut a strip off one side: returns (strip, rest). ``side`` = top / bottom / left / right."""
        _check_choice("Box.take", "side", side, ("top", "bottom", "left", "right"))
        a = self._len(amount, self.h if side in ("top", "bottom") else self.w) if amount <= 1 else float(amount)
        g = self._len(gap, self.short) if gap else 0.0
        x0, y0, x1, y1 = self
        if side == "top":
            return Box(x0, y0, x1, y0 + a), Box(x0, y0 + a + g, x1, y1)
        if side == "bottom":
            return Box(x0, y1 - a, x1, y1), Box(x0, y0, x1, y1 - a - g)
        if side == "left":
            return Box(x0, y0, x0 + a, y1), Box(x0 + a + g, y0, x1, y1)
        return Box(x1 - a, y0, x1, y1), Box(x0, y0, x1 - a - g, y1)

    def fit(self, aspect, align="mm"):
        """Largest box of width / height ratio ``aspect`` inside this one."""
        _check_choice("Box.fit", "align", align, tuple(a + b for a in "lmr" for b in "tmb"))
        w, h = self.w, self.h
        if w / max(h, 1e-9) > aspect:
            nw, nh = h * aspect, h
        else:
            nw, nh = w, w / aspect
        fx = {"l": 0.0, "m": 0.5, "r": 1.0}[align[0]]
        fy = {"t": 0.0, "m": 0.5, "b": 1.0}[align[1]]
        x0 = self[0] + (w - nw) * fx
        y0 = self[1] + (h - nh) * fy
        return Box(x0, y0, x0 + nw, y0 + nh)

    def square(self, align="mm"):
        return self.fit(1.0, align)

    def mask(self, cv, radius=0.0, aa=1.0):
        """Anti-aliased (H, W) mask of this box (rounded corners: ``radius`` px)."""
        from . import sdf
        X, Y = cv.xy()
        return sdf.fill(sdf.sd_box(X, Y, self.cx, self.cy, self.w / 2, self.h / 2, radius=radius), aa)


class Layout(Box):
    """The canvas minus a margin. ``Layout(cv, margin=0.06)`` (fraction of the short side, or pixels).

    ``.page`` is the full canvas, ``.S`` the 1080p scale factor, ``.m`` the margin in pixels.
    """

    def __new__(cls, cv, margin=0.06):
        W, H = cv.w, cv.h
        m = margin * min(W, H) if margin <= 1 else float(margin)
        self = super().__new__(cls, m, m, W - m, H - m)
        self.cv, self.m, self.S = cv, m, min(W, H) / 1080
        self.page = Box(0, 0, W, H)
        return self

    def baseline_grid(self, step=None, box=None):
        """y positions of a baseline grid (``step`` px, default 32 at 1080p) inside ``box`` (default: the layout)."""
        step = step or 32 * self.S
        b = box or self
        n = int(b.h // step)
        return [b[1] + (i + 1) * step for i in range(n)]

    def snap(self, y, step=None):
        """Snap a y coordinate to the baseline grid."""
        step = step or 32 * self.S
        return self[1] + round((y - self[1]) / step) * step


def align_boxes(boxes, edge="left", to=None):
    """Align bboxes on one edge (left / right / top / bottom / cx / cy); returns new Boxes.
    ``to`` = coordinate to align to (default: the extreme / mean of the boxes)."""
    _check_choice("align_boxes", "edge", edge, ("left", "right", "top", "bottom", "cx", "cy"))
    bs = [Box(b) for b in boxes]
    if not bs:
        return []
    if edge in ("left", "right", "cx"):
        vals = [b[0] for b in bs] if edge == "left" else [b[2] for b in bs] if edge == "right" else [b.cx for b in bs]
        t = to if to is not None else (min(vals) if edge == "left" else max(vals) if edge == "right" else sum(vals) / len(vals))
        return [Box(b[0] + t - v, b[1], b[2] + t - v, b[3]) for b, v in zip(bs, vals)]
    vals = [b[1] for b in bs] if edge == "top" else [b[3] for b in bs] if edge == "bottom" else [b.cy for b in bs]
    t = to if to is not None else (min(vals) if edge == "top" else max(vals) if edge == "bottom" else sum(vals) / len(vals))
    return [Box(b[0], b[1] + t - v, b[2], b[3] + t - v) for b, v in zip(bs, vals)]


def distribute(boxes, start, end, axis="x", mode="space"):
    """Spread bboxes evenly between ``start`` and ``end`` along ``axis``; ``mode="space"`` equal gaps,
    ``"center"`` equally spaced centres. Returns new Boxes in the given order."""
    _check_choice("distribute", "axis", axis, ("x", "y"))
    _check_choice("distribute", "mode", mode, ("space", "center"))
    bs = [Box(b) for b in boxes]
    n = len(bs)
    if n == 0:
        return []
    size = [b.w if axis == "x" else b.h for b in bs]
    out = []
    if mode == "center" or n == 1:
        for i, (b, s) in enumerate(zip(bs, size)):
            c = start + (end - start) * ((i + 0.5) / n)
            out.append(_place(b, axis, c - s / 2))
        return out
    gap = (end - start - sum(size)) / (n - 1)
    pos = start
    for b, s in zip(bs, size):
        out.append(_place(b, axis, pos))
        pos += s + gap
    return out


def _place(b, axis, p):
    if axis == "x":
        return Box(p, b[1], p + b.w, b[3])
    return Box(b[0], p, b[2], p + b.h)


# --------------------------------------------------------------------------------------------- type stacks

ROLES = {
    "kicker": dict(size=24, bold=True, case="upper", tracking=0.2, ink="accent"),
    "eyebrow": dict(size=24, bold=True, case="upper", tracking=0.2, ink="accent"),
    "title": dict(size=None, bold=True, balance=True, max_size=150),
    "headline": dict(size=None, bold=True, balance=True, max_size=110),
    "subtitle": dict(size=42, ink="soft"),
    "body": dict(size=28, ink="soft"),
    "quote": dict(size=None, balance=True, max_size=86, italic=True, font="serif"),
    "caption": dict(size=20, ink="muted"),
    "footer": dict(size=21, case="upper", tracking=0.14, bold=True, ink="muted"),
    "meta": dict(size=22, ink="muted"),
}


def _item(it):
    if isinstance(it, dict):
        d = dict(it)
    elif isinstance(it, (tuple, list)) and len(it) == 2 and isinstance(it[1], (str, dict)):
        d = {"text": it[0]}
        d.update({"role": it[1]} if isinstance(it[1], str) else it[1])
    else:
        d = {"text": str(it)}
    d.setdefault("role", "body")
    if d["role"] not in ROLES:
        raise _unknown_name("role", d["role"], ROLES)
    return d


def stack(cv, box, items, gap=None, align="left", valign="top", theme=None, scale=None, draw=True, **common):
    """Set text items top-to-bottom inside ``box`` with consistent spacing; returns their bboxes.

    ``items``: strings (body), ``(text, role)`` pairs or dicts with ``text`` + ``role`` + any ``cv.text`` keyword.
    Roles: kicker, title, headline, subtitle, body, quote, caption, footer, meta (sizes are 1080p pixels and scale
    with the canvas; ``title`` / ``headline`` / ``quote`` auto-fit the box width with balanced lines).
    Everything shrinks together when the stack would overflow the box. ``align`` = left / center / right,
    ``valign`` = top / middle / bottom, ``gap`` px between items (default from the sizes).
    """
    box = Box(box)
    H, W = cv.h, cv.w
    S = min(W, H) / 1080 if scale is None else scale
    _check_choice("stack", "align", align, ("left", "center", "right"))
    _check_choice("stack", "valign", valign, ("top", "middle", "bottom"))
    if isinstance(theme, str) and theme and theme not in THEMES:
        raise _unknown_name("theme", theme, THEMES, "Try layout.THEMES for the list")
    th = theme if isinstance(theme, dict) else THEMES.get(theme or "", None)
    _text_kw = [n for n in _sig_options(_t.draw_text) if n not in ("cv", "xy")] + ["max_size", "ink", "role"]
    common = _reject_unknown("stack", common, _text_kw)
    its = [_reject_unknown("stack item", _item(i), _text_kw) for i in items if i]
    specs = []
    for d in its:
        role = ROLES.get(d.get("role"), ROLES["body"])
        kw = {k: v for k, v in role.items() if k not in ("size", "max_size", "ink")}
        kw.update(common)
        kw.update({k: v for k, v in d.items() if k not in ("text", "role", "size", "max_size", "ink")})
        ink = d.get("ink", role.get("ink"))
        if "color" not in kw:
            kw["color"] = _ink(th, ink) if th else ("#ffffff" if ink is None else kw.get("color", "#ffffff"))
        if th and "font" not in kw:
            kw["font"] = th["title_font"] if d.get("role") in ("title", "headline") else th["body_font"]
        size = d.get("size", role.get("size"))
        size = None if size is None else float(size) * (S if d.get("size") is None else 1.0)
        mx = float(d.get("max_size", role.get("max_size") or 150)) * S
        specs.append([d["text"], kw, size, mx])

    def measure(sc):
        sizes, heights = [], []
        for text_, kw, size, mx in specs:
            sz = size * sc if size is not None else None
            if sz is None:
                lay = {k: v for k, v in kw.items() if k in _t._LAYOUT_KW}
                sz = _t.fit_text_size(text_, box.w, box.h * 0.55 * sc, kw.get("font"), kw.get("bold", False),
                                      max_size=mx * sc, wrap=True, line_spacing=_t._auto_leading(str(text_), mx, H),
                                      tracking=kw.get("tracking") or 0.0, **lay)
            ls = kw.get("line_spacing") or _t._auto_leading(_plain(text_), sz, H)
            tr = kw.get("tracking")
            tr = _t._auto_tracking(_plain(text_), sz, H) if tr is None else tr
            lay = {k: v for k, v in kw.items() if k in _t._LAYOUT_KW}
            bb = _t.text_bbox(text_, (0, 0), sz, kw.get("font"), kw.get("bold", False), "la", align, box.w, ls, tr, **lay)
            sizes.append((sz, ls, tr))
            heights.append(bb[3] - bb[1])
        gaps = []
        for i in range(len(specs) - 1):
            if gap is not None:
                gaps.append(float(gap))
            else:
                a, b = sizes[i][0], sizes[i + 1][0]
                gaps.append(0.42 * b + 0.22 * a)
        return sizes, heights, gaps
    sc = 1.0
    for _ in range(10):
        sizes, heights, gaps = measure(sc)
        total = sum(heights) + sum(gaps)
        if total <= box.h * 1.001 or sc < 0.15:
            break
        sc *= max(min(box.h / total, 0.97), 0.5)
    total = sum(heights) + sum(gaps)
    y = box[1] + {"top": 0.0, "middle": (box.h - total) / 2, "bottom": box.h - total}.get(valign, 0.0)
    x = {"left": box[0], "center": box.cx, "right": box[2]}.get(align, box[0])
    anchor = {"left": "lt", "center": "mt", "right": "rt"}.get(align, "lt")
    out = []
    for i, ((text_, kw, _, _), (sz, ls, tr), h) in enumerate(zip(specs, sizes, heights)):
        if draw:
            dkw = {k: v for k, v in kw.items() if k not in ("tracking", "line_spacing")}
            bb = _t.draw_text(cv, text_, (x, y), sz, anchor=anchor, align=align, max_width=box.w, line_spacing=ls,
                              tracking=tr, **dkw)
            out.append(Box(bb))
        else:
            out.append(Box(x - (box.w / 2 if align == "center" else box.w if align == "right" else 0), y,
                           x + (box.w / 2 if align == "center" else 0 if align == "right" else box.w), y + h))
        y += h + (gaps[i] if i < len(gaps) else 0.0)
    return out


def _plain(t):
    if isinstance(t, (list, tuple)):
        return " ".join(x[0] if isinstance(x, (tuple, list)) else str(x) for x in t)
    return str(t)


# --------------------------------------------------------------------------------------------- themes

THEMES = {
    "midnight": dict(bg=[(0, "#151a36"), (1, "#0b0d1c")], ink="#f4efe6", soft="#c9cbe0", muted="#8d93b8",
                     accent="#ffb347", accent2="#7aa2ff", panel="#1f2547", title_font="display", body_font="sans",
                     dark=True),
    "paper": dict(bg=[(0, "#f6f1e7"), (1, "#ebe2d1")], ink="#1f1a17", soft="#3d342e", muted="#62564a",
                  accent="#a3361f", accent2="#2f5d62", panel="#fffaf0", title_font="serif bold", body_font="serif",
                  dark=False),
    "neon": dict(bg=[(0, "#12052a"), (1, "#05010f")], ink="#fff0fa", soft="#e8c8ff", muted="#a68bd0",
                 accent="#ff3cac", accent2="#2bd2ff", panel="#22104a", title_font="display", body_font="sans",
                 dark=True),
    "forest": dict(bg=[(0, "#17352b"), (1, "#0c1f19")], ink="#f2f0e3", soft="#cfd8c6", muted="#8fa596",
                   accent="#e9c46a", accent2="#f4a261", panel="#21463a", title_font="serif bold", body_font="sans",
                   dark=True),
    "sunset": dict(bg=[(0, "#e8663f"), (0.5, "#c23b62"), (1, "#3e1a55")], ink="#fff8ef", soft="#ffe6d6",
                   muted="#ffd6c8", accent="#ffe066", accent2="#2b1630", panel="#3a1747", title_font="display",
                   body_font="sans", dark=True),
    "mono": dict(bg=[(0, "#fafafa"), (1, "#efefef")], ink="#111111", soft="#333333", muted="#666666",
                 accent="#c81e16", accent2="#111111", panel="#ffffff", title_font="sans black", body_font="sans",
                 dark=False),
    "pastel": dict(bg=[(0, "#fdf0f6"), (1, "#e4f1fb")], ink="#2d3047", soft="#4a4e69", muted="#8d8fa8",
                   accent="#e76f9a", accent2="#5aa9e6", panel="#ffffff", title_font="rounded bold", body_font="rounded",
                   dark=False),
    "luxury": dict(bg=[(0, "#1b1813"), (1, "#0b0a08")], ink="#f3e6c8", soft="#d9c9a5", muted="#ab9b7a",
                   accent="#e2bd68", accent2="#f3e6c8", panel="#26221b", title_font="serif", body_font="serif",
                   dark=True, metal="gold"),
    "ocean": dict(bg=[(0, "#0f4c75"), (1, "#0b2545")], ink="#f1faff", soft="#cfe8f7", muted="#8fb6d1",
                  accent="#ffd166", accent2="#06d6a0", panel="#13406a", title_font="sans black", body_font="sans",
                  dark=True),
}


# keys a theme may carry: the base keys of THEMES plus per-element colours (None = derived from the base keys)
THEME_KEYS = {
    "bg": "background gradient stops [(0, top), (1, bottom)] (or one colour)",
    "ink": "main text: title (unless 'title' is set), item names, place line",
    "soft": "subtitle and secondary lines",
    "muted": "captions, footer, dotted menu leaders",
    "accent": "kicker, rules, date pill, menu headings, flyer band, frame",
    "accent2": "second accent (decorations, links)",
    "panel": "surfaces: business card / label panel, badge disc, text on accent pills",
    "title_font": "font of titles and headlines",
    "body_font": "font of everything else",
    "dark": "True for a dark page (texture and vignette strength)",
    "metal": "metal effect for titles (luxury: gold)",
    "title": "title colour (default: ink)",
    "kicker": "kicker / eyebrow colour (default: accent)",
    "subtitle": "subtitle colour (default: soft)",
    "body": "details and body text colour (default: ink)",
    "band": "flyer footer band colour (default: accent)",
}
_ROLE_DEFAULT = {"title": "ink", "kicker": "accent", "subtitle": "soft", "body": "ink", "band": "accent"}


def theme(name=None, **over):
    """A theme dict (``THEMES[name]`` with overrides). Keys (see ``THEME_KEYS``): bg, ink, soft, muted, accent,
    accent2, panel, title_font, body_font, dark, metal, and the per-element colours title, kicker, subtitle, body,
    band (``title_color=`` style spellings work too)."""
    if isinstance(name, dict):
        t = dict(THEMES["midnight"])
        t.update(name)
    else:
        if name and name not in THEMES:
            raise _unknown_name("theme", name, THEMES, "Try layout.THEMES for the list")
        t = dict(THEMES.get(name or "midnight", THEMES["midnight"]))
    t.update(_reject_unknown("theme", {_theme_key(k): v for k, v in over.items()}, list(THEME_KEYS)))
    return t


def _theme_key(k):
    """title_color / title_colour -> title (any THEME_KEYS key with a colour suffix)."""
    for suf in ("_color", "_colour"):
        if k.endswith(suf) and k[:-len(suf)] in THEME_KEYS:
            return k[:-len(suf)]
    return k


def role_color(th, role):
    """The colour a theme gives an element: title / kicker / subtitle / body / band (falls back to the base key)."""
    v = th.get(role)
    return v if v is not None else th[_ROLE_DEFAULT.get(role, role)]


def _ink(th, ink):
    if ink is None:
        return th["ink"]
    return th.get(ink, ink)


def _bg(cv, th, seed=0, texture=0.035):
    from .noise import fbm2d
    cv.gradient(th["bg"])
    H, W = cv.h, cv.w
    S = min(W, H) / 1080
    n = fbm2d(H, W, 420 * S, 4, seed=seed + 11) * 0.5 + 0.5
    cv.paint("#ffffff" if th.get("dark") else "#000000", n * texture * (0.8 if th.get("dark") else 0.5))
    X, Y = cv.xy()
    lift = np.exp(-(((X - 0.5 * W) / (0.75 * W)) ** 2 + ((Y - 0.38 * H) / (0.6 * H)) ** 2)).astype(F32)
    if th.get("dark"):                                   # depth: a soft pool of light behind the content
        cv.paint(role_color(th, "kicker"), lift * 0.07, "add")
        cv.paint("#ffffff", lift * 0.04, "add")
    else:
        cv.paint("#ffffff", lift * 0.22)
        fib = fbm2d(H, W, 9 * S, 2, seed=seed + 23, aspect=0.25) * 0.5 + 0.5
        cv.paint("#6b4a22", fib * 0.035)
    d = ((X - W / 2) / W) ** 2 + ((Y - H / 2) / H) ** 2
    cv.paint("#000000", np.clip(d * 1.3, 0, 1) * (0.35 if th.get("dark") else 0.08))


def _rule(cv, x0, y0, x1, w, color, opacity=1.0):
    from .masks import rect_mask
    cv.paint(color, rect_mask(cv.h, cv.w, x0, y0 - w / 2, x1, y0 + w / 2, feather=0.6), opacity=opacity)


def _title_effect(th):
    if th.get("metal"):
        return th["metal"]
    return None


# --------------------------------------------------------------------------------------------- templates

def _setup(cv, theme_, margin, seed, background):
    th = theme(theme_)
    if background:
        _bg(cv, th, seed)
    L = Layout(cv, margin)
    return th, L, L.S


def _title_style(th, S):
    """Designed title by default: a soft vertical gradient of the title colour and a soft drop shadow."""
    col = rgb(role_color(th, "title"))
    if th.get("metal"):
        return {"effect": th["metal"]}
    hi = np.clip(col + (1 - col) * 0.35, 0, 1)
    lo = col * 0.86
    return {"gradient": [(0, hi), (0.55, col), (1, lo)],
            "shadow": dict(dx=0, dy=6 * S, blur=22 * S, opacity=0.5 if th.get("dark") else 0.18)}


def _frame(cv, L, th, S, k=0.45):
    from . import sdf
    X, Y = cv.xy()
    f = L.inset(-k * L.m / max(L.short, 1))
    cv.paint(role_color(th, "kicker"), sdf.outline(sdf.sd_box(X, Y, f.cx, f.cy, f.w / 2, f.h / 2), max(1.6 * S, 1)),
             opacity=0.55)


def _image_glow(cv, box, th):
    from . import sdf
    X, Y = cv.xy()
    r = 0.55 * box.short
    d = sdf.sd_circle(X, Y, box.cx, box.cy, r * 0.25)
    cv.paint(role_color(th, "kicker"), sdf.glow(d, r) * (0.22 if th.get("dark") else 0.12), "add" if th.get("dark")
             else "normal")


def _detail_parts(det):
    if isinstance(det, (tuple, list)):
        return str(det[0]), str(det[1])
    parts = str(det).split("\n", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def t_poster(cv, title, subtitle=None, details=None, kicker=None, footer=None, theme="midnight", margin=0.07, seed=0,
             background=True, frame=True, glow=True, **extra):
    """Poster: kicker, big gradient title, subtitle, a large picture area, details centred under a rule, a fine
    frame. ``frame`` / ``glow`` (soft light behind the picture area) can be switched off."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    portrait = cv.h >= cv.w
    if frame:
        _frame(cv, L, th, S)
    top, rest = L.take("top", 0.34 if portrait else 0.44)
    items = []
    if kicker:
        items.append({"text": kicker, "role": "kicker", "color": role_color(th, "kicker"), "size": 30 * S})
    items.append(dict({"text": title, "role": "title", "color": role_color(th, "title"), "font": th["title_font"],
                       "max_size": 230 if portrait else 160}, **_title_style(th, S)))
    if subtitle:
        items.append({"text": subtitle, "role": "subtitle", "color": role_color(th, "subtitle"), "size": 46 * S})
    boxes = stack(cv, top, items, align="center" if portrait else "left", theme=th)
    details = details or []
    single = len(details) == 1
    info_h = ((0.11 if single else 0.15) if portrait else 0.2) * cv.h if details else 0
    bottom, mid = rest.take("bottom", info_h, gap=0.025) if details else (None, rest)
    image = Box(mid[0], boxes[-1][3] + 0.035 * cv.h, mid[2], mid[3])
    res = {"title": boxes[1 if kicker else 0], "items": boxes, "image": image}
    if glow:
        _image_glow(cv, image, th)
    if details:
        acc = role_color(th, "kicker")
        if portrait:
            _rule(cv, bottom.cx - 0.18 * bottom.w, bottom[1], bottom.cx + 0.18 * bottom.w, max(2 * S, 1), acc, 0.9)
            _t.draw_text(cv, "◆", (bottom.cx, bottom[1]), 14 * S, acc, anchor="mm", audit=False)
        else:
            _rule(cv, bottom[0], bottom[1], bottom[2], max(2 * S, 1), acc, 0.9)
        n = min(len(details), 4)
        cols = bottom.inset(0, 0.012 * cv.h / bottom.short if bottom.short else 0, 0, 0).cols(n, gap=0.04 * cv.w)
        al = "center" if portrait else "left"
        for c, det in zip(cols, details[:n]):
            head, body = _detail_parts(det)
            stack(cv, c.inset(0, 0.028 * cv.h, 0, 0),
                  [{"text": head, "role": "kicker", "color": acc},
                   {"text": body, "role": "body", "color": role_color(th, "body"), "size": (36 if single else 30) * S}],
                  align=al, theme=th, gap=12 * S)
        res["details"] = cols
    if footer:
        _t.draw_text(cv, footer, (cv.w / 2, cv.h - L.m * (0.27 if frame else 0.5)), 19 * S, th["muted"], anchor="mm",
                     case="upper", tracking=0.2, font=th["body_font"])
    res["layout"] = L
    return res


def t_flyer(cv, title, subtitle=None, details=None, date=None, place=None, kicker=None, cta=None, theme="sunset",
            margin=0.07, seed=0, background=True, **extra):
    """Event flyer: date pill, big stacked title, subtitle, bullet details, a full-width picture area and a colour
    band (theme ``band``, default the accent) with the place and a call to action."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    band_h = 0.12 * cv.h
    from .masks import rect_mask
    band = Box(0, cv.h - band_h, cv.w, cv.h)
    bcol = role_color(th, "band")
    on_band = max((th["panel"], th["ink"], "#ffffff", "#111111"), key=lambda c: _t.contrast_ratio(c, bcol))
    cv.paint(bcol, rect_mask(cv.h, cv.w, *band), opacity=0.96)
    _rule(cv, 0, band[1] + max(3 * S, 1.5), cv.w, max(1.5 * S, 1), on_band, 0.35)
    body = Box(L[0], L[1], L[2], band[1] - 0.04 * cv.h)
    y = body[1]
    res = {}
    acc = role_color(th, "kicker")
    on_acc = max((th["panel"], th["ink"], "#ffffff", "#111111"), key=lambda c: _t.contrast_ratio(c, acc))
    if date:
        bb = _t.draw_text(cv, date, (body[0] + 0.3 * 34 * S * 1.5, y + 34 * S * 0.32), 34 * S, on_acc, anchor="lt",
                          bold=True, plate=dict(kind="pill", color=acc, opacity=1.0), case="upper", tracking=0.08,
                          font=th["body_font"])
        res["date"] = Box(bb)
        y = bb[3] + 0.04 * cv.h
    items = []
    if kicker:
        items.append({"text": kicker, "role": "kicker", "color": acc})
    items.append(dict({"text": title, "role": "title", "color": role_color(th, "title"), "font": th["title_font"],
                       "max_size": 185}, **_title_style(th, S)))
    if subtitle:
        items.append({"text": subtitle, "role": "subtitle", "color": role_color(th, "subtitle"), "size": 50 * S})
    if details:
        items.append({"text": "\n".join("- " + str(d) for d in details), "role": "body", "color": role_color(th, "body"),
                      "size": 30 * S, "para_spacing": 0.25})
    top = Box(body[0], y, body[2], body[1] + 0.62 * body.h)
    boxes = stack(cv, top, items, align="left", theme=th)
    res["title"] = boxes[1 if kicker else 0]
    res["items"] = boxes
    inner = band.inset(L.m, band.h * 0.2, L.m, band.h * 0.2)
    if place:
        _t.draw_text(cv, place, (inner[0], inner.cy), 34 * S, on_band, anchor="lm", bold=True, font=th["body_font"],
                     box=None)
    if cta:
        _t.draw_text(cv, cta, (inner[2] - 0.5 * 28 * S, inner.cy), 28 * S, bcol, anchor="rm", bold=True,
                     plate=dict(kind="pill", color=on_band, opacity=1), case="upper", tracking=0.1,
                     font=th["body_font"])
    res["band"] = band
    res["image"] = Box(L[0], boxes[-1][3] + 0.03 * cv.h, L[2], band[1] - 0.025 * cv.h)
    res["layout"] = L
    return res


def t_book_cover(cv, title, author=None, subtitle=None, series=None, theme="paper", margin=0.08, seed=0, background=True, **extra):
    """Book cover: serif title in the upper third between fine rules, picture area, author in spaced caps."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    from . import sdf
    X, Y = cv.xy()
    frame = L.inset(-0.35 * L.m / L.short if L.short else 0)
    cv.paint(th["accent"], sdf.outline(sdf.sd_box(X, Y, frame.cx, frame.cy, frame.w / 2, frame.h / 2), max(1.5 * S, 1)),
             opacity=0.7)
    inner = L.inset(0.04)
    res = {}
    top = inner.sub(0, 0.02, 1, 0.42)
    items = []
    if series:
        items.append({"text": series, "role": "kicker", "color": th["accent"], "font": th["body_font"]})
    items.append({"text": title, "role": "title", "color": role_color(th, "title"), "font": th["title_font"], "max_size": 170,
                  "case": None})
    if subtitle:
        items.append({"text": subtitle, "role": "subtitle", "color": th["soft"], "font": th["body_font"], "italic": True,
                      "size": 38 * S})
    boxes = stack(cv, top, items, align="center", valign="middle", theme=th)
    tb = boxes[1 if series else 0]
    _rule(cv, inner.cx - 0.12 * inner.w, tb[1] - 0.035 * cv.h, inner.cx + 0.12 * inner.w, max(2 * S, 1), th["accent"])
    _t.draw_text(cv, "◆", (inner.cx, boxes[-1][3] + 0.04 * cv.h), 22 * S, th["accent"], anchor="mm")
    res["title"] = tb
    res["items"] = boxes
    if author:
        bb = _t.draw_text(cv, author, (inner.cx, inner[3]), 34 * S, th["ink"], anchor="mb", case="upper", tracking=0.28,
                          font=th["body_font"], bold=False)
        res["author"] = Box(bb)
        _rule(cv, inner.cx - 0.05 * inner.w, bb[1] - 0.03 * cv.h, inner.cx + 0.05 * inner.w, max(1.5 * S, 1), th["accent"])
    res["image"] = Box(inner[0], boxes[-1][3] + 0.08 * cv.h, inner[2], (res["author"][1] if author else inner[3]) - 0.07 * cv.h)
    res["layout"] = L
    return res


def t_magazine(cv, title, headline=None, cover_lines=None, issue=None, kicker=None, theme="mono", margin=0.05, seed=0,
               background=True, **extra):
    """Magazine cover: full-width masthead, issue line, cover lines down the left, headline lower left.
    ``boxes["image"]`` is the whole page behind the type (draw the photo first, then call with background=False)."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    res = {"image": Box(0, 0, cv.w, cv.h)}
    mast = Box(L[0], L[1], L[2], L[1] + 0.21 * cv.h)
    bb = _t.draw_text(cv, title.upper(), box=mast, box_anchor="mt", color=th["accent"], font=th["title_font"],
                      tracking=-0.02, bold=True)
    res["masthead"] = Box(bb)
    if issue:
        _t.draw_text(cv, issue, (L[2], bb[3] + 0.012 * cv.h), 20 * S, th["ink"], anchor="rt", case="upper", tracking=0.16,
                     font=th["body_font"], bold=True)
        _rule(cv, L[0], bb[3] + 0.006 * cv.h, L[2], max(2 * S, 1), th["ink"])
    lines = cover_lines or []
    if lines:
        col = Box(L[0], bb[3] + 0.06 * cv.h, L[0] + 0.36 * L.w, L[1] + 0.62 * L.h)
        items = []
        for ln in lines:
            if isinstance(ln, (tuple, list)):
                items.append({"text": ln[0], "role": "kicker", "color": th["accent"], "size": 26 * S})
                items.append({"text": ln[1], "role": "body", "color": th["ink"], "size": 30 * S, "bold": True})
            else:
                items.append({"text": ln, "role": "body", "color": th["ink"], "size": 30 * S, "bold": True})
        res["cover_lines"] = stack(cv, col, items, theme=th, gap=14 * S)
    if headline:
        hb = Box(L[0], L[1] + 0.66 * L.h, L[0] + 0.7 * L.w, L[3])
        items = []
        if kicker:
            items.append({"text": kicker, "role": "kicker", "color": role_color(th, "kicker")})
        items.append({"text": headline, "role": "headline", "color": th["ink"], "font": th["title_font"], "max_size": 120})
        res["headline"] = stack(cv, hb, items, valign="bottom", theme=th)
    res["layout"] = L
    return res


def t_business_card(cv, name, role=None, details=None, company=None, theme="paper", margin=0.09, seed=0,
                    background=True, **extra):
    """Business card (e.g. 1050x600): name + role top left, accent bar, contact details bottom right."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    S = S * 1.8                                        # a card is read from 30 cm, not 3 m
    from .masks import rect_mask
    cv.paint(th["accent"], rect_mask(cv.h, cv.w, 0, 0, 0.022 * cv.w, cv.h))
    res = {}
    left = L.sub(0.03, 0, 0.62, 0.55)
    items = [{"text": name, "role": "headline", "color": th["ink"], "font": th["title_font"], "size": 0.13 * cv.h}]
    if role:
        items.append({"text": role, "role": "kicker", "color": th["accent"], "size": 0.042 * cv.h})
    res["name"] = stack(cv, left, items, theme=th)
    if company:
        bb = _t.draw_text(cv, company, (L[0] + 0.03 * L.w, L[3]), 0.05 * cv.h, th["ink"], anchor="lb", bold=True,
                          font=th["body_font"], case="smallcaps", tracking=0.06)
        res["company"] = Box(bb)
    if details:
        txt = "\n".join(str(d) for d in details)
        bb = _t.draw_text(cv, txt, (L[2], L[3]), 0.042 * cv.h, th["soft"], anchor="rb", align="right", line_spacing=1.5,
                          font=th["body_font"])
        res["details"] = Box(bb)
    res["image"] = Box(L[0] + 0.66 * L.w, L[1], L[2], L[1] + 0.45 * L.h)
    res["layout"] = L
    return res


def t_quote_card(cv, quote, author=None, source=None, theme="forest", margin=0.1, seed=0, background=True, **extra):
    """Social-media quote card: a large quotation mark, balanced serif quote, author line."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    res = {}
    _t.draw_text(cv, "“", (L[0] - 0.02 * L.w, L[1] - 0.04 * L.h), 300 * S, th["accent"], anchor="lt", font="serif bold",
                 opacity=0.9, audit=False)
    body = L.sub(0.0, 0.16, 1.0, 0.82)
    items = [{"text": quote, "role": "quote", "color": th["ink"], "font": "serif italic", "max_size": 92}]
    if author:
        items.append({"text": "— " + author, "role": "kicker", "color": th["accent"], "size": 28 * S, "case": None,
                      "tracking": 0.04})
    if source:
        items.append({"text": source, "role": "caption", "color": th["muted"]})
    res["items"] = stack(cv, body, items, valign="middle", theme=th)
    res["quote"] = res["items"][0]
    res["layout"] = L
    res["image"] = Box(L[0], L[3] - 0.12 * L.h, L[2], L[3])
    return res


def t_thumbnail(cv, title, subtitle=None, badge=None, theme="sunset", margin=0.05, seed=0, background=True,
                effect="sticker", layout="left", **extra):
    """Video thumbnail (1280x720): huge 2-4 word title with a sticker outline. ``layout="left"``: title left,
    picture area right; ``layout="top"``: title across the top, ``boxes["left"]`` / ``boxes["right"]`` for two
    subjects below it (``boxes["image"]`` = the whole band under the title)."""
    _check_choice("thumbnail", "layout", layout, ("left", "top"))
    th, L, S = _setup(cv, theme, margin, seed, background)
    res = {}
    eff = effect
    if eff == "sticker":
        eff = dict(kind="sticker", color="#111111")
    words = str(title)
    tcol = th.get("title") or th["accent"]
    if layout == "top":
        top = Box(L[0] + 0.04 * L.w, L[1], L[2] - 0.04 * L.w, L[1] + (0.36 if subtitle else 0.42) * L.h)
        bb = _t.draw_text(cv, words.upper(), box=top, box_anchor="mt", color=tcol, tracking=0.02, font="display",
                          balance=True, line_spacing=0.98, effect=eff, outline=0,
                          shadow=dict(dx=0, dy=8 * S, blur=10 * S, opacity=0.4))
        res["title"] = Box(bb)
        y = bb[3]
        if subtitle:
            sb = _t.draw_text(cv, subtitle, (L.cx, bb[3] + 0.035 * cv.h), 40 * S, "#ffffff", anchor="mt", bold=True,
                              plate=dict(kind="box", color="#111111", opacity=1, radius=8 * S), font=th["body_font"])
            res["subtitle"] = Box(sb)
            y = sb[3]
        if badge:
            bx = _t.draw_text(cv, badge, (L[2], L[3]), 36 * S, "#ffffff", anchor="rb", bold=True,
                              plate=dict(kind="pill", color="#e63946", opacity=1), case="upper", tracking=0.06,
                              font=th["body_font"])
            res["badge"] = Box(bx)
        band = Box(0, y + 0.02 * cv.h, cv.w, cv.h)
        res["image"] = band
        res["left"], res["right"] = Box(L[0], band[1], L[0] + 0.42 * L.w, cv.h), Box(L[2] - 0.42 * L.w, band[1],
                                                                                      L[2], cv.h)
        res["middle"] = Box(L[0] + 0.42 * L.w, band[1], L[2] - 0.42 * L.w, cv.h)
        res["layout"] = L
        return res
    left, right = L.cols([0.58, 0.42], gap=0.02)
    box = Box(left[0], left[1] + 0.1 * left.h, left[2], left[3] - (0.2 * left.h if subtitle else 0.08 * left.h))
    bb = _t.draw_text(cv, words.upper(), box=box, box_anchor="lm", color=tcol, tracking=0.02,
                      font="display", balance=True, line_spacing=0.98, effect=eff, outline=0,
                      shadow=dict(dx=0, dy=8 * S, blur=10 * S, opacity=0.4))
    res["title"] = Box(bb)
    if subtitle:
        sb = _t.draw_text(cv, subtitle, (bb[0] + 0.4 * 44 * S, bb[3] + 0.06 * cv.h), 44 * S, "#ffffff",
                          anchor="lt", bold=True, plate=dict(kind="box", color="#111111", opacity=1, radius=8 * S),
                          font=th["body_font"])
        res["subtitle"] = Box(sb)
    if badge:
        bx = _t.draw_text(cv, badge, (L[2], L[1] + 0.02 * cv.h), 40 * S, "#ffffff", anchor="rt", bold=True,
                          plate=dict(kind="pill", color="#e63946", opacity=1), case="upper", tracking=0.06,
                          font=th["body_font"], rotate=0)
        res["badge"] = Box(bx)
    res["image"] = right
    res["layout"] = L
    return res


def t_slide(cv, title, subtitle=None, kicker=None, footer=None, page=None, theme="midnight", margin=0.08, seed=0,
            background=True, **extra):
    """Slide title: left-aligned at the margin, accent rule, subtitle, footer line with page number."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    res = {}
    content = L.sub(0, 0.18, 0.72, 0.86)
    items = []
    if kicker:
        items.append({"text": kicker, "role": "kicker", "color": role_color(th, "kicker")})
    items.append({"text": title, "role": "title", "color": role_color(th, "title"), "font": th["title_font"], "max_size": 160})
    if subtitle:
        items.append({"text": subtitle, "role": "subtitle", "color": role_color(th, "subtitle")})
    details = extra.get("details")
    if details:
        items.append({"text": "\n".join("- " + str(d) for d in details), "role": "body", "color": th["soft"],
                      "para_spacing": 0.3})
    boxes = stack(cv, content, items, valign="middle", theme=th)
    tb = boxes[1 if kicker else 0]
    _rule(cv, L[0], tb[1] - 0.045 * cv.h, L[0] + 0.08 * L.w, max(5 * S, 2), th["accent"])
    res["title"] = tb
    res["items"] = boxes
    if footer or page is not None:
        _rule(cv, L[0], L[3] - 0.035 * cv.h, L[2], max(1.2 * S, 1), th["muted"], 0.5)
        if footer:
            _t.draw_text(cv, footer, (L[0], L[3]), 20 * S, th["muted"], anchor="lb", case="upper", tracking=0.14,
                         font=th["body_font"])
        if page is not None:
            _t.draw_text(cv, str(page), (L[2], L[3]), 20 * S, th["soft"], anchor="rb", tabular=True, font=th["body_font"])
    res["image"] = Box(L[0] + 0.76 * L.w, L[1], L[2], L[3] - 0.06 * cv.h)
    res["layout"] = L
    return res


def t_certificate(cv, title="Certificate of Achievement", name="Recipient Name", details=None, date=None, signer=None,
                  org=None, theme="luxury", margin=0.05, seed=0, background=True, **extra):
    """Certificate: double frame, spaced-caps title, script name, citation, date / signature lines, gold seal."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    from . import sdf
    X, Y = cv.xy()
    for k, w_ in ((0.0, 3.0), (0.018, 1.2)):
        b = L.inset(k)
        cv.paint(th["accent"], sdf.outline(sdf.sd_box(X, Y, b.cx, b.cy, b.w / 2, b.h / 2), max(w_ * S, 1)), opacity=0.85)
    inner = L.inset(0.09)
    res = {}
    items = []
    if org:
        items.append({"text": org, "role": "kicker", "color": th["accent"], "size": 24 * S})
    items.append({"text": title, "role": "headline", "color": th["ink"], "font": "serif", "case": "upper",
                  "tracking": 0.12, "max_size": 70, "bold": False})
    items.append({"text": "is proudly presented to", "role": "meta", "color": th["muted"], "italic": True,
                  "font": "serif", "size": 28 * S})
    items.append({"text": name, "role": "title", "color": th["accent"], "font": "script", "max_size": 130, "bold": False,
                  "effect": th.get("metal")})
    if details:
        items.append({"text": details, "role": "body", "color": th["soft"], "font": "serif", "size": 28 * S})
    top = inner.sub(0, 0, 1, 0.74)
    res["items"] = stack(cv, top, items, align="center", valign="middle", theme=th)
    foot = inner.sub(0, 0.78, 1, 1)
    c1, c2, c3 = foot.cols([1, 0.8, 1], gap=0.04)
    for c, lab, val in ((c1, "Date", date), (c3, "Signature", signer)):
        if val is None:
            continue
        _rule(cv, c[0] + 0.1 * c.w, c[1] + 0.45 * c.h, c[2] - 0.1 * c.w, max(1.2 * S, 1), th["muted"])
        _t.draw_text(cv, str(val), (c.cx, c[1] + 0.4 * c.h), 28 * S, th["ink"], anchor="mb",
                     font="script" if lab == "Signature" else "serif")
        _t.draw_text(cv, lab, (c.cx, c[1] + 0.52 * c.h), 18 * S, th["muted"], anchor="mt", case="upper", tracking=0.2,
                     font="serif")
    r = min(c2.w * 0.45, c2.h * 0.62)
    seal_c = (c2.cx, c2.cy)
    _seal(cv, seal_c, r, th, top_text=(org or "AWARD").upper(), bottom_text=str(date or "").upper(), center="★")
    res["seal"] = Box(seal_c[0] - r, seal_c[1] - r, seal_c[0] + r, seal_c[1] + r)
    res["layout"] = L
    res["image"] = res["seal"]
    return res


def _seal(cv, c, r, th, top_text="", bottom_text="", center="★", center_size=None):
    from . import sdf
    X, Y = cv.xy()
    cx, cy = c
    S = min(cv.w, cv.h) / 1080
    star = sdf.sd_star(X, Y, cx, cy, r, r * 0.9, n=28)
    cv.paint("#000000", sdf.glow(star, r * 0.12) * 0.35)
    base = rgb(th["accent"])
    cv.paint(base * sdf.bevel_light(star, r * 0.08, strength=0.4)[..., None], sdf.fill(star))
    cv.paint(base * 0.6, sdf.outline(sdf.sd_circle(X, Y, cx, cy, r * 0.78), max(r * 0.012, 1)))
    cv.paint(base * 0.6, sdf.outline(sdf.sd_circle(X, Y, cx, cy, r * 0.52), max(r * 0.012, 1)))
    ink = lerp(base, "#000000", 0.62)
    if top_text:
        _t.text_on_arc(cv, top_text, (cx, cy), r * 0.65, r * 0.15, angle=-90, color=ink, bold=True, audit=False)
    if bottom_text:
        _t.text_on_arc(cv, bottom_text, (cx, cy), r * 0.65, r * 0.13, angle=90, color=ink, bold=True, audit=False)
    if center:
        _t.draw_text(cv, center, (cx, cy), center_size or r * 0.42, ink, anchor="mm", bold=True)


def t_menu(cv, title, sections=None, subtitle=None, footer=None, theme="paper", margin=0.08, seed=0, background=True, **extra):
    """Menu: centred title, sections with dotted leaders and right-aligned prices.
    ``sections`` = [(heading, [(item, price) or (item, price, description), ...]), ...]."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    res = {}
    head = L.sub(0, 0, 1, 0.16)
    items = [{"text": title, "role": "title", "color": role_color(th, "title"), "font": th["title_font"], "max_size": 120}]
    if subtitle:
        items.append({"text": subtitle, "role": "kicker", "color": th["accent"], "size": 22 * S})
    hb = stack(cv, head, items, align="center", valign="middle", theme=th)
    res["title"] = hb[0]
    sections = sections or []
    body = Box(L[0], hb[-1][3] + 0.05 * cv.h, L[2], L[3] - (0.06 * cv.h if footer else 0))
    ncols = 2 if cv.w > cv.h * 1.1 and len(sections) > 1 else 1
    cols = body.cols(ncols, gap=0.08)
    per = math.ceil(len(sections) / ncols) if sections else 0
    n_lines = sum(1 + sum(2 if len(it) > 2 else 1 for it in sec[1]) for sec in sections) / max(ncols, 1) + per * 0.8
    size = min(30 * S, body.h / max(n_lines, 1) / 1.55)
    acc, ink = role_color(th, "kicker"), role_color(th, "body")

    def run(size, extra, draw):
        """Lay the columns out at ``size`` with ``extra`` px between sections; returns the tallest column end."""
        end = 0.0
        for ci, col in enumerate(cols):
            y = col[1]
            secs = sections[ci * per:(ci + 1) * per]
            for si, sec in enumerate(secs):
                heading, entries = sec[0], sec[1]
                hs = size * 1.05
                if draw:
                    bb = _t.draw_text(cv, heading, (col.cx, y), hs, acc, anchor="mt", bold=True, case="upper",
                                      tracking=0.22, font=th["body_font"])
                    rw = 0.06 * col.w
                    _rule(cv, bb[0] - rw - size * 0.6, (bb[1] + bb[3]) / 2, bb[0] - size * 0.6, max(1.2 * S, 1), acc,
                          0.7)
                    _rule(cv, bb[2] + size * 0.6, (bb[1] + bb[3]) / 2, bb[2] + size * 0.6 + rw, max(1.2 * S, 1), acc,
                          0.7)
                    res["sections"].append(heading)
                else:
                    bb = _t.text_bbox(heading, (col.cx, y), hs, th["body_font"], True, "mt", tracking=0.22,
                                      case="upper")
                y = bb[3] + size * 0.75
                for it in entries:
                    name, price = str(it[0]), str(it[1])
                    if draw:
                        nb = _t.draw_text(cv, name, (col[0], y), size, ink, anchor="lt", font=th["body_font"])
                        pb = _t.draw_text(cv, price, (col[2], y), size, ink, anchor="rt", font=th["body_font"],
                                          tabular=True)
                        base = nb[1] + (nb[3] - nb[1]) * 0.72
                        _leader(cv, nb[2] + size * 0.4, pb[0] - size * 0.4, base, size, th["muted"])
                    else:
                        nb = _t.text_bbox(name, (col[0], y), size, th["body_font"], False, "lt")
                    y = nb[3] + size * 0.15
                    if len(it) > 2 and it[2]:
                        if draw:
                            db = _t.draw_text(cv, str(it[2]), (col[0], y), size * 0.72, th["muted"], anchor="lt",
                                              italic=True, max_width=col.w * 0.8, font=th["body_font"])
                        else:
                            db = _t.text_bbox(str(it[2]), (col[0], y), size * 0.72, th["body_font"], False, "lt",
                                              max_width=col.w * 0.8, italic=True)
                        y = db[3] + size * 0.15
                    y += size * 0.42
                if si < len(secs) - 1:
                    y += size * 0.8 + extra
            end = max(end, y - col[1])
        return end

    res["sections"] = []
    if sections:                                         # fill the sheet: grow the type, then spread the sections
        for _ in range(4):
            used = run(size, 0.0, False)
            grow = body.h / max(used, 1.0)
            new = min(size * grow ** 0.9, 40 * S, size * 1.6)
            if abs(new - size) < 0.25:
                break
            size = new
        used = run(size, 0.0, False)
        gaps = max(max((len(sections[ci * per:(ci + 1) * per]) for ci in range(ncols)), default=1) - 1, 1)
        extra = max(body.h - used, 0.0) / gaps * 0.92
        run(size, extra, True)
    if footer:
        _t.draw_text(cv, footer, (L.cx, L[3]), 20 * S, th["muted"], anchor="mb", case="upper", tracking=0.2,
                     font=th["body_font"])
    res["layout"] = L
    res["image"] = Box(L[0], L[1], L[0] + 0.2 * L.w, L[1] + 0.12 * L.h)
    return res


def _leader(cv, x0, x1, y, size, color):
    if x1 - x0 < size:
        return
    from . import sdf
    X, Y = cv.xy()
    step = size * 0.32
    n = int((x1 - x0) / step)
    if n < 2:
        return
    xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    r = max(size * 0.045, 0.7)
    y0, y1 = int(y - r - 2), int(y + r + 3)
    sub = np.zeros((cv.h, cv.w), F32)
    yy = np.arange(max(y0, 0), min(y1, cv.h), dtype=F32)[:, None]
    xx = np.arange(cv.w, dtype=F32)[None, :]
    if len(yy) == 0:
        return
    k = np.round((xx - x0) / ((x1 - x0) / n) - 0.5)
    k = np.clip(k, 0, n - 1)
    cxs = x0 + (k + 0.5) * (x1 - x0) / n
    d = np.sqrt((xx - cxs) ** 2 + (yy - y) ** 2) - r
    m = np.clip(0.5 - d, 0, 1) * ((xx >= x0 - r) & (xx <= x1 + r))
    sub[max(y0, 0):min(y1, cv.h)] = m
    cv.paint(color, sub, opacity=0.7)
    del xs, X, Y, sdf


def t_badge(cv, title, subtitle=None, center=None, theme="forest", seed=0, background=True, radius=0.4, **extra):
    """Badge / seal: circular emblem with arc text top and bottom and a centre word or symbol."""
    th, L, S = _setup(cv, theme, 0.05, seed, background)
    from . import sdf
    X, Y = cv.xy()
    cx, cy = cv.w / 2, cv.h / 2
    R = radius * min(cv.w, cv.h)
    outer = sdf.sd_circle(X, Y, cx, cy, R)
    cv.paint("#000000", sdf.glow(outer, R * 0.08) * 0.4)
    cv.paint(th["panel"], sdf.fill(outer))
    for rr, w_ in ((0.96, 0.012), (0.66, 0.008), (0.62, 0.004)):
        cv.paint(th["accent"], sdf.outline(sdf.sd_circle(X, Y, cx, cy, R * rr), max(R * w_, 1)))
    band_r = R * 0.81
    _t.text_on_arc(cv, str(title).upper(), (cx, cy), band_r, R * 0.13, angle=-90, color=th["ink"], bold=True,
                   font=th["body_font"])
    if subtitle:
        _t.text_on_arc(cv, str(subtitle).upper(), (cx, cy), band_r, R * 0.1, angle=90, color=th["accent"], bold=True,
                       font=th["body_font"])
    for a in (180, 0):
        _t.draw_text(cv, "★", (cx + band_r * math.cos(math.radians(a)), cy), R * 0.08, th["accent"], anchor="mm")
    res = {}
    cb = Box(cx - R * 0.5, cy - R * 0.4, cx + R * 0.5, cy + R * 0.4)
    if center:
        bb = _t.draw_text(cv, center, box=cb, color=th["ink"], font=th["title_font"], balance=True, line_spacing=1.0)
        res["center"] = Box(bb)
    res["image"] = cb
    res["badge"] = Box(cx - R, cy - R, cx + R, cy + R)
    res["layout"] = L
    return res


def t_label(cv, title, subtitle=None, details=None, brand=None, est=None, theme="paper", margin=0.08, seed=0,
            background=True, **extra):
    """Product label (jar / bottle / soap): framed panel, brand in small caps, product name, details."""
    th, L, S = _setup(cv, theme, margin, seed, background)
    from . import sdf
    X, Y = cv.xy()
    panel = L.inset(0.02)
    d = sdf.sd_box(X, Y, panel.cx, panel.cy, panel.w / 2, panel.h / 2, radius=0.1 * panel.short)
    cv.paint("#000000", sdf.glow(d, 20 * S) * 0.25)
    cv.paint(th["panel"], sdf.fill(d))
    cv.paint(th["accent"], sdf.outline(sdf.offset(d, 14 * S), max(2.5 * S, 1)))
    cv.paint(th["accent"], sdf.outline(sdf.offset(d, 22 * S), max(1 * S, 1)))
    inner = panel.inset(0.12)
    items = []
    if brand:
        items.append({"text": brand, "role": "kicker", "color": th["accent"], "size": 26 * S})
    items.append({"text": title, "role": "title", "color": role_color(th, "title"), "font": th["title_font"], "max_size": 150})
    if subtitle:
        items.append({"text": subtitle, "role": "subtitle", "color": th["soft"], "italic": True, "font": "serif"})
    if details:
        items.append({"text": "  ·  ".join(details) if isinstance(details, (list, tuple)) else details, "role": "caption",
                      "color": th["muted"], "case": "upper", "tracking": 0.12, "size": 20 * S})
    res = {"items": stack(cv, inner, items, align="center", valign="middle", theme=th)}
    if est:
        _t.draw_text(cv, est, (panel.cx, panel[3] - 0.05 * panel.h), 18 * S, th["soft"], anchor="mb", case="upper",
                     tracking=0.3, font=th["body_font"])
    res["panel"] = panel
    res["image"] = inner
    res["layout"] = L
    return res


TEMPLATES = {
    "poster": t_poster, "flyer": t_flyer, "event_flyer": t_flyer, "book_cover": t_book_cover, "book": t_book_cover,
    "magazine": t_magazine, "magazine_cover": t_magazine, "business_card": t_business_card, "card": t_business_card,
    "quote_card": t_quote_card, "quote": t_quote_card, "thumbnail": t_thumbnail, "youtube": t_thumbnail,
    "slide": t_slide, "slide_title": t_slide, "certificate": t_certificate, "menu": t_menu, "badge": t_badge,
    "seal": t_badge, "label": t_label, "product_label": t_label,
}


_EXTRA_CONTENT = {"t_slide": ("details",)}      # keywords a template reads from its ``**extra``


def template(kind, cv, /, **content):
    """Draw a complete typographic layout from content alone; returns a dict of the boxes it used.

    ``layout.template("poster", cv, title="Night Market", subtitle="...", details=["FRI 8 PM", "Pier 4"],
    theme="midnight")``. Templates: poster, flyer, book_cover, magazine, business_card, quote_card, thumbnail,
    slide, certificate, menu, badge, label. Themes: midnight, paper, neon, forest, sunset, mono, pastel, luxury,
    ocean (or a dict). ``boxes["image"]`` is the free area for a picture; ``background=False`` keeps what is
    already on the canvas (draw the picture first).
    """
    fn = TEMPLATES.get(str(kind).lower().replace("-", "_").replace(" ", "_"))
    if fn is None:
        raise _unknown_name("template", kind, TEMPLATES, "Try layout.TEMPLATES for the list")
    allowed = [n for n in _sig_options(fn) if n not in ("cv", "extra")] + list(_EXTRA_CONTENT.get(fn.__name__, ()))
    over = {}
    for k in list(content):                              # colour / font overrides: title_color="#f0d078", accent=...
        tk = _theme_key(k)
        if k not in allowed and tk in THEME_KEYS:
            over[tk] = content.pop(k)
    content = _reject_unknown("template '%s'" % kind, content, allowed + [k + "_color" for k in _ROLE_DEFAULT])
    if over:
        import inspect
        base = content.get("theme", inspect.signature(getattr(fn, "__wrapped__", fn)).parameters["theme"].default)
        content["theme"] = theme(base, **over)
    return fn(cv, **content)


# --------------------------------------------------------------------------------------------- checked options

def _strict_templates():
    """The ``t_*`` template functions take ``**extra`` for call-compatibility only: unknown content names raise."""
    import functools
    for name, fn in list(globals().items()):
        if not name.startswith("t_") or not callable(fn):
            continue
        allowed = [n for n in _sig_options(fn) if n != "extra"] + list(_EXTRA_CONTENT.get(name, ()))
        known = frozenset(allowed)

        def make(fn, name, allowed, known):
            @functools.wraps(fn)
            def wrapper(*a, **k):
                if k and not known.issuperset(k):
                    k = _reject_unknown(name, k, allowed)
                return fn(*a, **k)
            return wrapper
        globals()[name] = make(fn, name, allowed, known)


_strict_templates()
TEMPLATES.update({k: globals()[v.__name__] for k, v in TEMPLATES.items()})
align_boxes = _strict(align_boxes, "align_boxes")
distribute = _strict(distribute, "distribute")
_strict_methods(Box)
_strict_methods(Layout)
