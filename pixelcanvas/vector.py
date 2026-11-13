# -*- coding: utf-8 -*-
"""Retained-mode vector drawing: one ``Drawing`` → anti-aliased raster on a Canvas *and* a standalone SVG.

Quick start::

    from pixelcanvas.vector import Drawing
    dwg = Drawing(800, 450, theme="light")
    with dwg.group("card", shadow=True):
        dwg.rect(40, 40, 320, 180, r=14, fill=dwg.theme.surface, stroke=dwg.theme.grid)
        dwg.text(200, 130, "Hello vectors", size=32, anchor="mm", weight="bold")
    dwg.line(380, 130, 600, 130, stroke=dwg.theme.accents[0], stroke_width=3, arrow="end")
    cv = dwg.render()                  # Canvas (anti-aliased, one layer per top-level group)
    dwg.save_svg("card.svg")           # same geometry as SVG 1.1

Coordinates are user units (``units="px"`` by default, or ``"mm"`` / ``"cm"`` / ``"in"`` / ``"pt"``); x right,
y down, angles in degrees clockwise (SVG convention). ``dwg.draw(cv)`` scales the drawing to fit the canvas (or
``box``), so the same drawing works for drafts and full-size renders.
"""
from __future__ import annotations

import copy as _copy
import math
import re
from functools import lru_cache

import numpy as np
from PIL import Image

from .color import F32, rgb as _rgb
from . import text as _text
from . import manifest as _manifest

_RES = getattr(Image, "Resampling", Image)
_TRANSFORM = getattr(Image, "Transform", Image)

__all__ = ["Drawing", "Group", "Shape", "Text", "Path", "Theme", "THEMES", "theme", "linear_gradient",
           "radial_gradient", "hatch", "measure_text", "icon", "icon_names", "ICONS", "last_drawings",
           "clear_drawings", "MARKERS", "DASHES"]

_SS = 6                     # vertical sub-samples per pixel (horizontal coverage is exact)
_TOL = 0.12                 # curve flattening tolerance, device px
_CORNER = 0.30              # turn angle (rad) above which a vertex is a corner (gets a real join)
_UNITS = {"px": 1.0, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96.0, "pt": 96 / 72}
MARKERS = ("arrow", "triangle", "open", "dot", "bar", "diamond", "none")
DASHES = {"dash": (4, 3), "longdash": (8, 4), "dot": (0, 2), "dashdot": (6, 3, 0, 3), "center": (12, 3, 2, 3),
          "hidden": (3, 2)}

# --------------------------------------------------------------------------------------------- themes


class Theme(dict):
    """A theme dict with attribute access: ``t.ink``, ``t.accents[2]``, ``t.accent(9)`` (wraps around)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def accent(self, i):
        return self["accents"][int(i) % len(self["accents"])]


def _theme(name, background, surface, ink, muted, grid, accents, stroke_width=1.5, radius=8, shadow=None,
           font=None, dark=False):
    sh = dict(dx=0, dy=2, blur=6, color="#0f172a", opacity=0.16)
    sh.update(shadow or {})
    fo = dict(family="sans", weight="normal", title_weight="bold")
    fo.update(font or {})
    return Theme(name=name, background=background, surface=surface, ink=ink, muted=muted, grid=grid,
                 accents=list(accents), stroke_width=stroke_width, radius=radius, shadow=sh, font=fo, dark=dark)


THEMES = {
    "light": _theme("light", "#f6f7f9", "#ffffff", "#1d2633", "#667385", "#e1e5eb",
                    ["#2f6fdf", "#f08a24", "#1f9d74", "#e0464e", "#7e5bd0", "#1aa3c2", "#d6588f", "#8a9a2a"]),
    "dark": _theme("dark", "#10151c", "#1a212b", "#e8edf3", "#8b98a8", "#2a333f",
                   ["#5aa5ff", "#ffa552", "#4fd08a", "#ff7a72", "#b48cff", "#3cc8d6", "#f57fb6", "#d8c35a"],
                   shadow=dict(color="#000000", opacity=0.45, blur=8, dy=3), dark=True),
    "blueprint": _theme("blueprint", "#163d6e", "#1b4a84", "#eaf2ff", "#9fbbe0", "#2c5d98",
                        ["#ffffff", "#8fd6ff", "#ffd36e", "#7fe3bb", "#ff9e88", "#c6aaff", "#64c9f5", "#eef3a2"],
                        stroke_width=1.25, radius=2, shadow=dict(color="#06162b", opacity=0.35),
                        font=dict(family="mono"), dark=True),
    "paper": _theme("paper", "#f3eee3", "#fbf8f1", "#2b2621", "#776c60", "#ded5c4",
                    ["#b4533b", "#3d6d8c", "#5b8a4c", "#cf9634", "#7a5a8b", "#3a8a84", "#a6455f", "#6c6a35"],
                    radius=4, shadow=dict(color="#3b2f22", opacity=0.14), font=dict(family="serif")),
    "pastel": _theme("pastel", "#fbf9ff", "#ffffff", "#3c3950", "#8a86a2", "#ebe7f5",
                     ["#86b3f7", "#f6b48a", "#93d8ad", "#f3a0b4", "#c1abf1", "#83d2dc", "#f4d47c", "#b6d487"],
                     stroke_width=1.75, radius=14, shadow=dict(color="#5b4b8a", opacity=0.14, blur=10, dy=4)),
    "neon": _theme("neon", "#0b0b19", "#15152b", "#f2f0ff", "#9390b8", "#25254a",
                   ["#00e5ff", "#ff2bd6", "#b6ff3b", "#ffb800", "#8a6bff", "#ff5c6c", "#00ffa8", "#ff8a1f"],
                   stroke_width=2, radius=10, shadow=dict(color="#000000", opacity=0.6, blur=10, dy=3), dark=True),
    "mono": _theme("mono", "#ffffff", "#f4f4f4", "#121212", "#6f6f6f", "#e3e3e3",
                   ["#121212", "#4a4a4a", "#7d7d7d", "#a8a8a8", "#2e2e2e", "#636363", "#929292", "#c4c4c4"],
                   stroke_width=1.5, radius=4, shadow=dict(color="#000000", opacity=0.14)),
}


class _UnknownName(ValueError, KeyError):
    """Unknown registry name (a ValueError that is also a KeyError, so older ``except KeyError`` code keeps working)."""
    __str__ = ValueError.__str__


def _unknown(kind, name, valid, hint=None):
    from ._args import unknown_name
    return _UnknownName(*unknown_name(kind, name, list(valid), hint).args)


def _find_theme(name):
    if name not in THEMES:
        raise _unknown("theme", name, THEMES)
    return THEMES[name]


def theme(name="light"):
    """A copy of ``THEMES[name]`` (or the Theme you pass)."""
    t = _find_theme(name) if isinstance(name, str) else name
    return Theme(_copy.deepcopy(dict(t)))


# --------------------------------------------------------------------------------------------- colours

_NAMED = {"black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
          "gray": "#808080", "grey": "#808080", "yellow": "#ffff00", "orange": "#ffa500", "transparent": None,
          "none": None}


def _color(c):
    """→ (rgb float32[3], alpha) or None."""
    if c is None or c is False:
        return None
    if isinstance(c, str):
        s = c.strip().lower()
        if s in _NAMED:
            if _NAMED[s] is None:
                return None
            s = _NAMED[s]
        h = s.lstrip("#")
        if len(h) in (4, 8):
            a = int(h[-1] * 2 if len(h) == 4 else h[-2:], 16) / 255.0
            return _rgb("#" + h[:-1] if len(h) == 4 else "#" + h[:6]), a
        return _rgb(s), 1.0
    a = np.asarray(c, np.float64).ravel()
    if a.size == 4:
        col = _rgb(tuple(a[:3]))
        al = a[3] / 255.0 if a[3] > 1 else a[3]
        return col, float(al)
    return _rgb(tuple(a[:3]) if a.size >= 3 else c), 1.0


def _hex(col):
    c = np.clip(np.asarray(col, np.float64), 0, 1)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)


class _Gradient:
    def __init__(self, kind, stops, units, **geo):
        def is_stop(s):
            return (isinstance(s, (tuple, list)) and len(s) in (2, 3) and isinstance(s[0], (int, float))
                    and not isinstance(s[1], (int, float)))

        if not all(is_stop(s) for s in stops):                 # plain colours → evenly spaced
            n = len(stops)
            stops = [(i / max(n - 1, 1), s) for i, s in enumerate(stops)]
        norm = []
        for st in stops:
            pos, col = float(st[0]), st[1]
            c = _color(col)
            if c is None:
                c = (np.zeros(3, F32), 0.0)
            op = float(st[2]) if len(st) > 2 else 1.0
            norm.append((pos, c[0], c[1] * op))
        self.kind, self.stops, self.units = kind, norm, units
        self.__dict__.update(geo)


def linear_gradient(stops, start=(0, 0), end=(0, 1), units="bbox"):
    """Linear gradient paint for ``fill=`` / ``stroke=``.

    ``stops``: ``[(pos, colour), ...]`` or ``(pos, colour, opacity)`` or just a list of colours (evenly spaced).
    ``start`` / ``end``: points in bbox fractions (``units="bbox"``, default: top → bottom) or user units
    (``units="user"``)."""
    return _Gradient("linear", list(stops), units, start=tuple(start), end=tuple(end))


def radial_gradient(stops, center=(0.5, 0.5), radius=0.5, units="bbox"):
    """Radial gradient paint: position 0 at ``center``, 1 at ``radius`` (bbox fractions or user units)."""
    return _Gradient("radial", list(stops), units, center=tuple(center), radius=float(radius))


class _Hatch:
    def __init__(self, angle=45, spacing=None, width=None, color=None, cross=False, opacity=1.0):
        self.angle, self.spacing, self.width = float(angle), spacing, width
        self.color, self.cross, self.opacity = color, bool(cross), float(opacity)


def hatch(angle=45, spacing=None, width=None, color=None, cross=False, opacity=1.0):
    """Hatch fill for ``hatch=``: parallel lines at ``angle`` degrees, ``spacing`` apart, ``width`` thick
    (user units; defaults scale with the drawing's stroke width), clipped to the shape. ``cross=True`` adds the
    perpendicular set. ``color`` defaults to the shape's stroke (or ink)."""
    return _Hatch(angle, spacing, width, color, cross, opacity)


# --------------------------------------------------------------------------------------------- affine maths
# m = (a, b, c, d, e, f):  x' = a x + c y + e,   y' = b x + d y + f    (SVG matrix order)

_I = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mul(m, n):
    """m ∘ n (apply n first)."""
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D, a * E + c * F + e, b * E + d * F + f)


def _apply(m, pts):
    p = np.asarray(pts, np.float64)
    a, b, c, d, e, f = m
    out = np.empty_like(p)
    out[..., 0] = a * p[..., 0] + c * p[..., 1] + e
    out[..., 1] = b * p[..., 0] + d * p[..., 1] + f
    return out


def _inv(m):
    a, b, c, d, e, f = m
    det = a * d - b * c
    if abs(det) < 1e-15:
        det = 1e-15
    ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
    return (ia, ib, ic, id_, -(ia * e + ic * f), -(ib * e + id_ * f))


def _mscale(m):
    return math.sqrt(abs(m[0] * m[3] - m[1] * m[2]))


def _conformal(m):
    a, b, c, d = m[:4]
    s = max(abs(a) + abs(b) + abs(c) + abs(d), 1e-12)
    return abs(a - d) < 1e-6 * s and abs(b + c) < 1e-6 * s and a * d - b * c > 0


def _upright(m):
    return abs(m[1]) < 1e-9 and abs(m[2]) < 1e-9 and m[0] > 0 and abs(m[0] - m[3]) < 1e-9 * max(m[0], 1)


def _translate(tx, ty):
    return (1.0, 0.0, 0.0, 1.0, float(tx), float(ty))


def _rotate(deg, cx=0.0, cy=0.0):
    t = math.radians(deg)
    co, si = math.cos(t), math.sin(t)
    r = (co, si, -si, co, 0.0, 0.0)
    if cx or cy:
        return _mul(_translate(cx, cy), _mul(r, _translate(-cx, -cy)))
    return r


def _scale(sx, sy=None):
    return (float(sx), 0.0, 0.0, float(sx if sy is None else sy), 0.0, 0.0)


# --------------------------------------------------------------------------------------------- paths

def _n(v):
    s = "%.3f" % v
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _ellipse_arc(cx, cy, rx, ry, phi, th1, dth):
    """Cubic segments approximating an ellipse arc (angles in radians)."""
    n = max(1, int(math.ceil(abs(dth) / (math.pi / 2) - 1e-9)))
    d = dth / n
    k = 4.0 / 3.0 * math.tan(d / 4)
    cp, sp = math.cos(phi), math.sin(phi)

    def pt(t):
        x, y = rx * math.cos(t), ry * math.sin(t)
        return cx + x * cp - y * sp, cy + x * sp + y * cp

    def dv(t):
        x, y = -rx * math.sin(t), ry * math.cos(t)
        return x * cp - y * sp, x * sp + y * cp

    segs = []
    for i in range(n):
        t0, t1 = th1 + i * d, th1 + (i + 1) * d
        p0, p3, d0, d1 = pt(t0), pt(t1), dv(t0), dv(t1)
        segs.append(("C", p0[0] + k * d0[0], p0[1] + k * d0[1], p3[0] - k * d1[0], p3[1] - k * d1[1], p3[0], p3[1]))
    return segs


def _arc_segments(x1, y1, rx, ry, rot, large, sweep, x2, y2):
    """SVG endpoint arc → cubic segments (SVG 1.1 implementation notes F.6.5)."""
    if abs(x1 - x2) < 1e-12 and abs(y1 - y2) < 1e-12:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12:
        return [("L", x2, y2)]
    phi = math.radians(rot)
    cp, sp = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) / 2, (y1 - y2) / 2
    x1p, y1p = cp * dx2 + sp * dy2, -sp * dx2 + cp * dy2
    lam = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(num, 0.0) / den) if den > 0 else 0.0
    if bool(large) == bool(sweep):
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2

    def ang(ux, uy, vx, vy):
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    th1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi
    segs = _ellipse_arc(cx, cy, rx, ry, phi, th1, dth)
    if segs:                                                 # land exactly on the endpoint
        s = segs[-1]
        segs[-1] = s[:5] + (x2, y2)
    return segs


_CMD_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARGS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def _parse_d(path, d):
    i, n = 0, len(d)
    cmd = None
    last_c = last_q = None

    def skip(i):
        while i < n and (d[i].isspace() or d[i] == ","):
            i += 1
        return i

    while True:
        i = skip(i)
        if i >= n:
            break
        if _CMD_RE.match(d, i):
            cmd = d[i]
            i += 1
            if cmd in "Zz":
                path.close()
                last_c = last_q = None
                continue
        elif cmd is None:
            raise ValueError(f"bad path data near {d[i:i + 12]!r}")
        up = cmd.upper()
        args = []
        for k in range(_ARGS[up]):
            i = skip(i)
            if up == "A" and k in (3, 4):                   # flags may be written without separators
                if i < n and d[i] in "01":
                    args.append(float(d[i]))
                    i += 1
                    continue
            m = _NUM_RE.match(d, i)
            if not m:
                raise ValueError(f"bad path data near {d[i:i + 12]!r}")
            args.append(float(m.group()))
            i = m.end()
        rel = cmd.islower()
        cx, cy = path._pen if path._pen is not None else (0.0, 0.0)
        ox, oy = (cx, cy) if rel else (0.0, 0.0)
        if up == "M":
            path.move_to(args[0] + ox, args[1] + oy)
            cmd = "l" if rel else "L"                       # implicit lineto after moveto
            last_c = last_q = None
        elif up == "L":
            path.line_to(args[0] + ox, args[1] + oy)
            last_c = last_q = None
        elif up == "H":
            path.line_to(args[0] + ox, cy)
            last_c = last_q = None
        elif up == "V":
            path.line_to(cx, args[0] + (cy if rel else 0.0))
            last_c = last_q = None
        elif up == "C":
            x1, y1, x2, y2, x, y = (args[0] + ox, args[1] + oy, args[2] + ox, args[3] + oy, args[4] + ox, args[5] + oy)
            path.cubic_to(x1, y1, x2, y2, x, y)
            last_c, last_q = (x2, y2), None
        elif up == "S":
            x1, y1 = (2 * cx - last_c[0], 2 * cy - last_c[1]) if last_c else (cx, cy)
            x2, y2, x, y = args[0] + ox, args[1] + oy, args[2] + ox, args[3] + oy
            path.cubic_to(x1, y1, x2, y2, x, y)
            last_c, last_q = (x2, y2), None
        elif up == "Q":
            qx, qy, x, y = args[0] + ox, args[1] + oy, args[2] + ox, args[3] + oy
            path.quad_to(qx, qy, x, y)
            last_q, last_c = (qx, qy), None
        elif up == "T":
            qx, qy = (2 * cx - last_q[0], 2 * cy - last_q[1]) if last_q else (cx, cy)
            x, y = args[0] + ox, args[1] + oy
            path.quad_to(qx, qy, x, y)
            last_q, last_c = (qx, qy), None
        elif up == "A":
            path.arc_to(args[0], args[1], args[2], args[3], args[4], args[5] + ox, args[6] + oy)
            last_c = last_q = None
    return path


def _cubic_split(p0, c1, c2, p3, t):
    """de Casteljau split → (left, right) control tuples."""
    lerp = lambda a, b: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    a, b, c = lerp(p0, c1), lerp(c1, c2), lerp(c2, p3)
    d, e = lerp(a, b), lerp(b, c)
    f = lerp(d, e)
    return (p0, a, d, f), (f, e, c, p3)


def _cubic_pt(p0, c1, c2, p3, t):
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * p3[1])


class Path:
    """Path geometry: subpaths of line and cubic segments (quadratics and arcs are converted exactly / to cubics).

    Build with ``move_to`` / ``line_to`` / ``quad_to`` / ``cubic_to`` / ``arc_to`` / ``close`` (all chainable) or
    from SVG path data ``Path("M0 0 L10 0 A5 5 0 0 1 20 0 Z")``."""

    def __init__(self, d=None):
        self.subs = []                      # [start (x, y), [segments], closed]
        self._pen = None
        if d:
            _parse_d(self, d)

    def _ensure_open(self):
        if self._pen is None:
            raise ValueError("path: call move_to() first")
        if self.subs[-1][2]:                # drawing after close → new subpath at the pen
            self.subs.append([self._pen, [], False])

    def move_to(self, x, y):
        self._pen = (float(x), float(y))
        self.subs.append([self._pen, [], False])
        return self

    def line_to(self, x, y):
        if self._pen is None:
            return self.move_to(x, y)
        self._ensure_open()
        self.subs[-1][1].append(("L", float(x), float(y)))
        self._pen = (float(x), float(y))
        return self

    def cubic_to(self, x1, y1, x2, y2, x, y):
        self._ensure_open()
        self.subs[-1][1].append(("C", float(x1), float(y1), float(x2), float(y2), float(x), float(y)))
        self._pen = (float(x), float(y))
        return self

    def quad_to(self, qx, qy, x, y):
        self._ensure_open()
        x0, y0 = self._pen
        return self.cubic_to(x0 + 2 / 3 * (qx - x0), y0 + 2 / 3 * (qy - y0),
                             x + 2 / 3 * (qx - x), y + 2 / 3 * (qy - y), x, y)

    def arc_to(self, rx, ry, rotation, large, sweep, x, y):
        """SVG-style elliptical arc from the pen to (x, y)."""
        self._ensure_open()
        x0, y0 = self._pen
        self.subs[-1][1].extend(_arc_segments(x0, y0, rx, ry, rotation, large, sweep, float(x), float(y)))
        self._pen = (float(x), float(y))
        return self

    def bezier_to(self, *pts):
        """Bezier from the pen through control points: 1 point → line, 2 → quadratic, 3 → cubic,
        3k → chain of cubics. Points are (x, y) pairs."""
        pts = [tuple(p) for p in pts]
        if len(pts) == 1:
            return self.line_to(*pts[0])
        if len(pts) == 2:
            return self.quad_to(*pts[0], *pts[1])
        if len(pts) % 3:
            raise ValueError("bezier_to: give 1, 2 or a multiple of 3 points")
        for i in range(0, len(pts), 3):
            self.cubic_to(*pts[i], *pts[i + 1], *pts[i + 2])
        return self

    def close(self):
        if self.subs:
            self.subs[-1][2] = True
            self._pen = self.subs[-1][0]
        return self

    # ---- queries
    @property
    def closed(self):
        return any(s[2] for s in self.subs)

    def copy(self):
        p = Path()
        p.subs = [[s[0], list(s[1]), s[2]] for s in self.subs]
        p._pen = self._pen
        return p

    def transformed(self, m):
        p = Path()
        for start, segs, closed in self.subs:
            st = tuple(_apply(m, start))
            ns = []
            for sg in segs:
                q = _apply(m, np.array(sg[1:], np.float64).reshape(-1, 2)).ravel()
                ns.append((sg[0],) + tuple(float(v) for v in q))
            p.subs.append([st, ns, closed])
        if self._pen is not None:
            p._pen = tuple(_apply(m, self._pen))
        return p

    def flatten(self, tol=0.1):
        """→ list of (points (n, 2) float64, closed)."""
        out = []
        for start, segs, closed in self.subs:
            pts = [start]
            cur = start
            for sg in segs:
                if sg[0] == "L":
                    cur = (sg[1], sg[2])
                    pts.append(cur)
                else:
                    p0 = cur
                    x1, y1, x2, y2, x, y = sg[1:]
                    ddx = max(abs(p0[0] - 2 * x1 + x2), abs(x1 - 2 * x2 + x))
                    ddy = max(abs(p0[1] - 2 * y1 + y2), abs(y1 - 2 * y2 + y))
                    dd = math.hypot(ddx, ddy)
                    nseg = int(min(max(math.ceil(math.sqrt(0.75 * dd / max(tol, 1e-9))), 1), 512))
                    if nseg == 1:
                        pts.append((x, y))
                    else:
                        t = np.arange(1, nseg + 1) / nseg
                        u = 1 - t
                        bx = u ** 3 * p0[0] + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t ** 3 * x
                        by = u ** 3 * p0[1] + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t ** 3 * y
                        pts.extend(zip(bx.tolist(), by.tolist()))
                    cur = (x, y)
            out.append((np.asarray(pts, np.float64), closed))
        return out

    def bbox(self):
        """Geometric bounding box (x0, y0, x1, y1) of the path."""
        pl = self.flatten(0.05)
        if not pl:
            return (0.0, 0.0, 0.0, 0.0)
        P = np.concatenate([p for p, _ in pl])
        return (float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max()))

    def to_d(self):
        """SVG path data string."""
        out = []
        for start, segs, closed in self.subs:
            out.append(f"M{_n(start[0])} {_n(start[1])}")
            for sg in segs:
                if sg[0] == "L":
                    out.append(f"L{_n(sg[1])} {_n(sg[2])}")
                else:
                    out.append("C" + " ".join(_n(v) for v in sg[1:]))
            if closed:
                out.append("Z")
        return "".join(out)

    def _end(self, which):
        """(tip, prev-control-point-ish, segment index info) for marker placement."""
        if not self.subs:
            return None
        if which == "end":
            start, segs, _ = self.subs[-1]
            if not segs:
                return None
            prev = segs[-2][-2:] if len(segs) > 1 else start
            return prev, segs[-1]
        start, segs, _ = self.subs[0]
        if not segs:
            return None
        return start, segs[0]

    def trimmed(self, at_start=0.0, at_end=0.0):
        """Copy with the first subpath shortened by ``at_start`` and the last by ``at_end`` (chord distance)."""
        p = self.copy()
        if at_end > 0 and p.subs and p.subs[-1][1]:
            start, segs, closed = p.subs[-1]
            prev = segs[-2][-2:] if len(segs) > 1 else start
            segs[-1] = _trim_seg(prev, segs[-1], at_end, from_end=True)
        if at_start > 0 and p.subs and p.subs[0][1]:
            start, segs, closed = p.subs[0]
            new_start, seg0 = _trim_seg(start, segs[0], at_start, from_end=False)
            p.subs[0][0] = new_start
            segs[0] = seg0
        return p


def _trim_seg(p0, sg, dist, from_end):
    """Shorten one segment by chord distance ``dist`` at its end (or start). Returns the new segment
    (or (new_start, segment) when trimming the start)."""
    p3 = sg[-2:]
    if sg[0] == "L":
        L = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        f = min(dist / L, 0.95) if L > 0 else 0.0
        if from_end:
            return ("L", p3[0] + (p0[0] - p3[0]) * f, p3[1] + (p0[1] - p3[1]) * f)
        return (p0[0] + (p3[0] - p0[0]) * f, p0[1] + (p3[1] - p0[1]) * f), sg
    c1, c2 = sg[1:3], sg[3:5]
    tip = p3 if from_end else p0
    lo, hi = 0.0, 1.0                                        # parameter measured from the tip
    for _ in range(40):
        mid = (lo + hi) / 2
        t = 1 - mid if from_end else mid
        q = _cubic_pt(p0, c1, c2, p3, t)
        if math.hypot(q[0] - tip[0], q[1] - tip[1]) < dist:
            lo = mid
        else:
            hi = mid
    s = min((lo + hi) / 2, 0.95)
    if from_end:
        left, _ = _cubic_split(p0, c1, c2, p3, 1 - s)
        return ("C",) + left[1] + left[2] + left[3]
    _, right = _cubic_split(p0, c1, c2, p3, s)
    return right[0], ("C",) + right[1] + right[2] + right[3]


def _catmull(points, closed=False, tension=0.5):
    """Smooth cubic path through the points (Catmull-Rom → Bezier)."""
    P = [tuple(map(float, p)) for p in points]
    path = Path().move_to(*P[0])
    n = len(P)
    if n < 3:
        for q in P[1:]:
            path.line_to(*q)
        return path.close() if closed else path
    k = tension / 3.0 * 2.0 if tension != 0.5 else 1 / 3.0
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = P[i - 1] if (i > 0 or closed) else P[i]
        p1, p2 = P[i], P[(i + 1) % n]
        p3 = P[(i + 2) % n] if (i + 2 < n or closed) else p2
        c1 = (p1[0] + (p2[0] - p0[0]) * k / 2, p1[1] + (p2[1] - p0[1]) * k / 2)
        c2 = (p2[0] - (p3[0] - p1[0]) * k / 2, p2[1] - (p3[1] - p1[1]) * k / 2)
        path.cubic_to(*c1, *c2, *p2)
    return path.close() if closed else path


# --------------------------------------------------------------------------------------------- raster core

def _dedupe(P):
    if len(P) < 2:
        return P
    d = np.abs(np.diff(P, axis=0)).sum(1) > 1e-9
    return P[np.concatenate([[True], d])]


def _signed_area(batch):
    x, y = batch[..., 0], batch[..., 1]
    s = (x[..., :-1] * y[..., 1:] - x[..., 1:] * y[..., :-1]).sum(-1)
    return 0.5 * (s + x[..., -1] * y[..., 0] - x[..., 0] * y[..., -1])


def _pos(batch):
    """Orient every polygon of a (m, k, 2) batch counter-clockwise-positive (union under nonzero)."""
    if batch.size == 0:
        return batch
    neg = _signed_area(batch) < 0
    if neg.any():
        batch = batch.copy()
        batch[neg] = batch[neg, ::-1]
    return batch


def _circles(C, r, tol):
    C = np.asarray(C, np.float64).reshape(-1, 2)
    k = int(np.clip(math.ceil(math.pi / math.acos(max(1 - tol / max(r, 1e-9), -1.0))), 8, 128))
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ring = np.stack([np.cos(t), np.sin(t)], -1) * r
    return C[:, None, :] + ring[None]


def _square_at(P, hw):
    P = np.asarray(P, np.float64).reshape(-1, 2)
    off = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64) * hw
    return P[:, None, :] + off[None]


def _stroke_segments(A, B, T, hw, cap, tol, out):
    """Vectorised stroke of many independent 2-point pieces A→B with unit directions T."""
    N = np.stack([-T[:, 1], T[:, 0]], -1)
    A2, B2 = A, B
    if cap == "square":
        A2, B2 = A - T * hw, B + T * hw
    L = np.hypot(*(B2 - A2).T)
    ok = L > 1e-9
    if ok.any():
        q = np.stack([A2 + N * hw, B2 + N * hw, B2 - N * hw, A2 - N * hw], 1)[ok]
        out.append(_pos(q))
    if cap == "round":
        out.append(_pos(_circles(np.concatenate([A, B]), hw, tol)))


def _stroke_polyline(P, closed, hw, cap, join, ml, tol, out):
    """Stroke outline polygons (union under nonzero) of one polyline, appended to ``out`` as (m, k, 2) batches."""
    P = _dedupe(np.asarray(P, np.float64))
    if closed and len(P) > 1 and np.abs(P[0] - P[-1]).sum() < 1e-9:
        P = P[:-1]
    n = len(P)
    if n == 0 or hw <= 0:
        return
    if n == 1:
        if cap == "round":
            out.append(_pos(_circles(P, hw, tol)))
        elif cap == "square":
            out.append(_pos(_square_at(P, hw)))
        return
    if closed and n == 2:
        closed, cap = False, ("round" if join == "round" else cap)
    D = (np.roll(P, -1, 0) - P) if closed else (P[1:] - P[:-1])
    L = np.hypot(D[:, 0], D[:, 1])
    T = D / L[:, None]
    N = np.stack([-T[:, 1], T[:, 0]], -1)
    if closed:
        Tin, Nin, Tout, Nout, V = np.roll(T, 1, 0), np.roll(N, 1, 0), T, N, P
    else:
        Tin, Nin, Tout, Nout, V = T[:-1], N[:-1], T[1:], N[1:], P[1:-1]
    dot = (Tin * Tout).sum(1)
    cross = Tin[:, 0] * Tout[:, 1] - Tin[:, 1] * Tout[:, 0]
    ang = np.arctan2(cross, dot)
    corner = np.abs(ang) > _CORNER
    denom = np.maximum(1 + dot, 1e-6)
    Moff = (Nin + Nout) / denom[:, None]
    if corner.any():
        cv = V[corner]
        if join == "round":
            out.append(_pos(_circles(cv, hw, tol)))
        else:
            sg = -np.sign(cross[corner])[:, None]
            a = cv + sg * hw * Nin[corner]
            b = cv + sg * hw * Nout[corner]
            tip = cv + sg * hw * Moff[corner]
            ratio = np.sqrt(2 / denom[corner])
            use = (ratio <= ml)[:, None] if join == "miter" else np.zeros((len(cv), 1), bool)
            out.append(_pos(np.stack([cv, a, np.where(use, tip, a), b], 1)))
    if closed:
        if not corner.any():
            outer, inner = P + hw * Moff, P - hw * Moff
            a1, a2 = _signed_area(outer[None])[0], _signed_area(inner[None])[0]
            big, small = (outer, inner) if abs(a1) >= abs(a2) else (inner, outer)
            big = _pos(big[None])
            small = small[None]
            if _signed_area(small)[0] > 0:
                small = small[:, ::-1]
            out.append(big)
            out.append(small)
            return
        c0 = int(np.nonzero(corner)[0][0])
        idx = (np.arange(n + 1) + c0) % n
        Pp, Nseg, cint, Mint = P[idx], N[idx[:-1]], corner[idx[1:-1]], Moff[idx[1:-1]]
    else:
        Pp, Nseg, cint, Mint = P, N, corner, Moff
    m = len(Pp)
    cuts = np.nonzero(cint)[0] + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [m - 1]])
    single = ends - starts == 1
    if single.any():
        s, e = starts[single], ends[single]
        Ns = Nseg[s]
        q = np.stack([Pp[s] + hw * Ns, Pp[e] + hw * Ns, Pp[e] - hw * Ns, Pp[s] - hw * Ns], 1)
        out.append(_pos(q))
    for s, e in zip(starts[~single], ends[~single]):
        Off = np.empty((e - s + 1, 2))
        Off[0], Off[-1] = Nseg[s], Nseg[e - 1]
        Off[1:-1] = Mint[s:e - 1]
        seg = Pp[s:e + 1]
        poly = np.concatenate([seg + hw * Off, (seg - hw * Off)[::-1]])
        out.append(_pos(poly[None]))
    if not closed:
        t0, t1 = -T[0], T[-1]
        if cap == "square":
            for p, t in ((P[0], t0), (P[-1], t1)):
                nn = np.array([-t[1], t[0]])
                out.append(_pos(np.array([[p + nn * hw, p + nn * hw + t * hw, p - nn * hw + t * hw, p - nn * hw]])))
        elif cap == "round":
            out.append(_pos(_circles(np.stack([P[0], P[-1]]), hw, tol)))


def _dash(P, closed, pattern, offset):
    """Split a polyline into dashes. Returns (list of long pieces, (A, B, T) arrays of 2-point pieces)."""
    P = _dedupe(np.asarray(P, np.float64))
    if closed:
        P = np.concatenate([P, P[:1]])
    if len(P) < 2:
        return [], None
    seg = np.diff(P, axis=0)
    sl = np.hypot(seg[:, 0], seg[:, 1])
    keep = sl > 1e-12
    seg, sl = seg[keep], sl[keep]
    P = np.concatenate([P[:1], P[1:][keep]])
    cum = np.concatenate([[0.0], np.cumsum(sl)])
    total = cum[-1]
    pat = [float(v) for v in pattern]
    if len(pat) % 2:
        pat = pat * 2
    period = sum(pat)
    if period <= 1e-12 or total <= 0:
        return [(P, False)], None
    pc = np.concatenate([[0.0], np.cumsum(pat)])
    start = -(float(offset) % period)
    reps = int(math.ceil((total - start) / period)) + 1
    if reps * len(pat) > 200000:
        return [(P, False)], None
    base = start + period * np.arange(reps)
    s = (base[:, None] + pc[None, 0:-1:2]).ravel()
    e = s + np.tile(np.asarray(pat[0::2]), reps)
    s, e = np.maximum(s, 0.0), np.minimum(e, total)
    ok = (e > s) | ((e == s) & (s > 0) & (s < total))
    s, e = s[ok], e[ok]
    nseg = len(sl)
    i0 = np.clip(np.searchsorted(cum, s, "right") - 1, 0, nseg - 1)
    i1 = np.clip(np.searchsorted(cum, e, "left") - 1, 0, nseg - 1)
    i1 = np.maximum(i1, i0)
    T = seg / sl[:, None]
    A = P[i0] + T[i0] * (s - cum[i0])[:, None]
    B = P[i1] + T[i1] * (e - cum[i1])[:, None]
    two = i0 == i1
    longs = []
    for k in np.nonzero(~two)[0]:
        longs.append((np.concatenate([A[k:k + 1], P[i0[k] + 1:i1[k] + 1], B[k:k + 1]]), False))
    return longs, (A[two], B[two], T[i0[two]])


def _poly_len(P, closed):
    P = np.asarray(P, np.float64)
    if closed and len(P) > 1:
        P = np.concatenate([P, P[:1]])
    return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0


def _fit_dash(pattern, L, closed=False, center=False):
    """Scale a dash pattern to length ``L``: an open line starts and ends on a full dash, a closed one holds whole
    periods. ``center``: an even number of gaps (centre lines). None → too short, draw it solid."""
    pat = [max(float(v), 0.0) for v in pattern]
    if len(pat) % 2:
        pat = pat * 2
    P, a = sum(pat), pat[0]
    if P <= 1e-12 or L <= 0:
        return None
    if closed:
        n = max(1, round(L / P))
        return tuple(v * L / (n * P) for v in pat)
    if L <= a * 1.2:
        return None
    n = (L - a) / P
    n = max(1, int(round(n / 2)) * 2) if center else max(1, int(round(n)))
    k = L / (n * P + a)
    return tuple(v * k for v in pat)


def _box_snap(d, box, f):
    """Map device points so the box edges (X0, Y0, X1, Y1) land on ``f(edge)`` (pixel-aligned rectangles)."""
    X0, Y0, X1, Y1 = box
    out = d.copy()
    for ax, (a, b) in enumerate(((X0, X1), (Y0, Y1))):
        A, B = f(a), f(b)
        if B - A >= 1 and b - a > 1e-9:
            out[:, ax] = A + (d[:, ax] - a) * ((B - A) / (b - a))
    return out


def _snap(P, hw2):
    """Pixel-align an axis-aligned device polyline so odd widths sit on pixel centres (crisp 1 px lines)."""
    w = 2 * hw2
    k = round(w)
    if k < 1 or abs(w - k) > 0.08 or k > 9 or len(P) < 2:
        return P
    d = np.diff(P, axis=0)
    hor, ver = np.abs(d[:, 1]) < 1e-6, np.abs(d[:, 0]) < 1e-6
    if not np.all(hor | ver):
        return P
    n = len(P)
    pos_x = np.zeros(n, bool)
    pos_y = np.zeros(n, bool)
    pos_x[:-1] |= ver
    pos_x[1:] |= ver
    pos_y[:-1] |= hor
    pos_y[1:] |= hor
    f = (lambda v: np.floor(v) + 0.5) if k % 2 else np.round
    Q = P.copy()
    Q[:, 0] = np.where(pos_x, f(P[:, 0]), np.round(P[:, 0]))
    Q[:, 1] = np.where(pos_y, f(P[:, 1]), np.round(P[:, 1]))
    return Q


def _raster(jobs, bounds, ss=_SS):
    """Rasterise coverage jobs in one vectorised pass.

    ``jobs``: list of (batches, evenodd) where batches is a list of (m, k, 2) device-px polygon arrays.
    Returns, per job, ``(x0, y0, coverage float32 (h, w))`` clipped to ``bounds`` or None.
    Vertical: ``ss`` sample rows per pixel; horizontal: exact span coverage."""
    out = [None] * len(jobs)
    # chunk by area so the accumulation buffer stays small
    chunk, area = [], 0
    for j, (batches, eo) in enumerate(jobs):
        ext = [b.reshape(-1, 2) for b in batches if b.size]
        if not ext:
            continue
        E = np.concatenate(ext) if len(ext) > 1 else ext[0]
        x0, y0 = E.min(0)
        x1, y1 = E.max(0)
        bx0 = int(max(math.floor(x0), bounds[0]))
        by0 = int(max(math.floor(y0), bounds[1]))
        bx1 = int(min(math.ceil(x1), bounds[2]))
        by1 = int(min(math.ceil(y1), bounds[3]))
        if bx1 <= bx0 or by1 <= by0:
            continue
        a = (bx1 - bx0 + 1) * (by1 - by0)
        if area + a > 6_000_000 and chunk:
            _raster_chunk(jobs, chunk, out, ss)
            chunk, area = [], 0
        chunk.append((j, bx0, by0, bx1 - bx0, by1 - by0))
        area += a
    if chunk:
        _raster_chunk(jobs, chunk, out, ss)
    return out


def _raster_chunk(jobs, chunk, out, ss):
    nj = len(chunk)
    jx0 = np.array([c[1] for c in chunk], np.float64)
    jy0 = np.array([c[2] for c in chunk], np.float64)
    jw = np.array([c[3] for c in chunk], np.int64)
    jh = np.array([c[4] for c in chunk], np.int64)
    eo = np.array([jobs[c[0]][1] for c in chunk], bool)
    rowoff = np.concatenate([[0], np.cumsum(jh * ss)])[:-1]
    pixoff = np.concatenate([[0], np.cumsum(jh * (jw + 1))])
    Ps, Qs, Js = [], [], []
    for k, c in enumerate(chunk):
        for b in jobs[c[0]][0]:
            if not b.size:
                continue
            Ps.append(b.reshape(-1, 2))
            Qs.append(np.roll(b, -1, axis=1).reshape(-1, 2))
            Js.append(np.full(b.shape[0] * b.shape[1], k, np.int64))
    P, Q, J = np.concatenate(Ps), np.concatenate(Qs), np.concatenate(Js)
    up = Q[:, 1] > P[:, 1]
    keep = Q[:, 1] != P[:, 1]
    P, Q, J, up = P[keep], Q[keep], J[keep], up[keep]
    ya = np.where(up, P[:, 1], Q[:, 1]) - jy0[J]
    yb = np.where(up, Q[:, 1], P[:, 1]) - jy0[J]
    xa = np.where(up, P[:, 0], Q[:, 0]) - jx0[J]
    xb = np.where(up, Q[:, 0], P[:, 0]) - jx0[J]
    wd = np.where(up, 1, -1).astype(np.int32)
    r0 = np.maximum(np.ceil(ya * ss - 0.5).astype(np.int64), 0)
    r1 = np.minimum(np.ceil(yb * ss - 0.5).astype(np.int64) - 1, jh[J] * ss - 1)
    cnt = r1 - r0 + 1
    m = cnt > 0
    if not m.any():
        return
    xa, ya, xb, yb, wd, r0, cnt, J = xa[m], ya[m], xb[m], yb[m], wd[m], r0[m], cnt[m], J[m]
    tot = int(cnt.sum())
    idx = np.repeat(np.arange(len(cnt)), cnt)
    first = np.cumsum(cnt) - cnt
    r = np.arange(tot, dtype=np.int64) - first[idx] + r0[idx]
    slope = (xb - xa) / (yb - ya)
    x = xa[idx] + ((r + 0.5) / ss - ya[idx]) * slope[idx]
    jj = J[idx]
    maxw = float(jw.max())
    xc = np.clip(x, -1.0, maxw + 1.0)
    key = rowoff[jj] + r
    order = np.argsort(key.astype(np.float64) * (maxw + 4.0) + (xc + 1.5), kind="quicksort")
    ks, xs, ws, js, rs = key[order], xc[order], wd[idx][order], jj[order], r[order]
    wc = np.cumsum(ws)
    inside = np.where(eo[js], (wc & 1) != 0, wc != 0)
    span = np.nonzero(inside[:-1] & (ks[:-1] == ks[1:]))[0]
    if not len(span):
        return
    sj = js[span]
    wj = jw[sj].astype(np.float64)
    sa = np.clip(xs[span], 0.0, wj)
    sb = np.clip(xs[span + 1], 0.0, wj)
    ok = sb > sa
    sa, sb, sj, sr = sa[ok], sb[ok], sj[ok], rs[span][ok]
    base = pixoff[sj] + (sr // ss) * (jw[sj] + 1)
    fa = np.floor(sa).astype(np.int64)
    fb = np.floor(sb).astype(np.int64)
    total = int(pixoff[-1])
    D = np.bincount(base + fa, minlength=total).astype(np.float64)
    D -= np.bincount(base + fb, minlength=total)
    Pw = np.bincount(base + fb, weights=sb - fb, minlength=total)
    Pw -= np.bincount(base + fa, weights=sa - fa, minlength=total)
    acc = (np.cumsum(D) + Pw) * (1.0 / ss)
    for k, c in enumerate(chunk):
        w, h = int(jw[k]), int(jh[k])
        a = acc[pixoff[k]:pixoff[k + 1]].reshape(h, w + 1)[:, :w]
        out[c[0]] = (c[1], c[2], np.clip(a, 0.0, 1.0).astype(F32))


# --------------------------------------------------------------------------------------------- fonts

_STACKS = {
    "sans": "'Segoe UI', 'Helvetica Neue', Arial, 'Noto Sans', 'Microsoft JhengHei', 'PingFang TC', "
            "'Noto Sans CJK TC', sans-serif",
    "serif": "Georgia, 'Times New Roman', 'Noto Serif', 'Noto Serif CJK TC', 'PMingLiU', serif",
    "mono": "Consolas, 'Cascadia Mono', Menlo, 'DejaVu Sans Mono', 'Noto Sans Mono', monospace",
}


def _is_bold(weight):
    if isinstance(weight, (int, float)):
        return weight >= 600
    return str(weight).lower() in ("bold", "semibold", "heavy", "black", "extrabold")


@lru_cache(maxsize=256)
def _font_path(family, bold, cjk):
    fam = (family or "sans").lower()
    if fam == "sans":
        return None                                    # text module's automatic choice (CJK aware)
    if fam in ("serif", "mono", "cjk", "cjk-serif"):
        return _text.find_font(None, bold=bold, style=fam, text="中" if cjk else None)
    return _text.find_font(family, bold=bold)


@lru_cache(maxsize=1)
def _italic_ok():
    import inspect
    try:
        return "kw" in inspect.signature(_text.text_bbox).parameters
    except (TypeError, ValueError):
        return False


def _tkw(family, weight, style, text):
    """Keyword arguments for pixelcanvas.text measuring / rendering."""
    bold = _is_bold(weight)
    kw = dict(font=_font_path(family, bold, _text.has_cjk(text)), bold=bold)
    if str(style).lower() in ("italic", "oblique") and _italic_ok():
        kw["italic"] = True
    return kw


def _text_lines(text, size, kw, max_width, tracking, line_spacing, anchor, xy, markup=False):
    """Per-line layout [(line, x_left, top_y, width)] and block bbox, from the public text API."""
    f = _text.load_font(kw["font"], size, kw["bold"], text=text)
    lines = _text.wrap_text(text, f, max_width, tracking * size) if max_width else str(text).split("\n")
    bb = _text.text_bbox(text, xy, size, anchor=anchor, max_width=max_width, line_spacing=line_spacing,
                         tracking=tracking, markup=markup, **kw)
    out = []
    for i, ln in enumerate(lines):
        if ln:
            x0, _, x1, _ = _text.text_bbox(ln, (0, 0), size, anchor="la", line_spacing=line_spacing,
                                           tracking=tracking, markup=markup, **kw)
            w = x1 - x0
        else:
            w = 0.0
        out.append((ln, w, bb[1] + i * size * line_spacing))
    asc = f.getmetrics()[0]
    return out, bb, asc


_ANCHORS = {"center": "mm", "middle": "mm", "left": "lm", "right": "rm", "top": "mt", "bottom": "mb",
            "top-left": "lt", "top-right": "rt", "bottom-left": "lb", "bottom-right": "rb", "baseline": "ls"}


def _anchor(a):
    a = _ANCHORS.get(str(a).lower(), str(a).lower())
    if len(a) != 2 or a[0] not in "lmr" or a[1] not in "tamsbdc":
        raise ValueError(f"anchor must be like 'lt', 'mm', 'mc', 'rb', 'ls' (got {a!r})")
    return a


@lru_cache(maxsize=128)
def _cap_ratio(font, bold, cjk):
    """Capital height / font size for this font (from the glyph 'H')."""
    f = _text.load_font(font, 200, bold, text="\u4e2d" if cjk else "H")
    b = f.getbbox("H", anchor="ls")
    return min(max(-b[1] / 200.0, 0.4), 1.0)


def measure_text(text, size=16, family="sans", weight="normal", style="normal", letter_spacing=0.0,
                 line_spacing=1.2, max_width=None, markup=False):
    """(width, height) in px of ``text`` at ``size`` px — same metrics the raster uses (multi-line aware).
    Trailing spaces count with their advance width (``"1. "`` is wider than ``"1."``)."""
    s = str(text)
    kw = _tkw(family, weight, style, s)
    tr = letter_spacing / max(size, 1e-9)
    lines = s.split("\n")
    body = "\n".join(ln.rstrip() for ln in lines)

    def width_height(t):
        b = _text.text_bbox(t, (0, 0), size, anchor="la", max_width=max_width, line_spacing=line_spacing,
                            tracking=tr, markup=markup, **kw)
        return b[2] - b[0], b[3] - b[1]
    w, h = width_height(body if body.strip() else (s or " "))
    if max_width is None and body != s:
        f = _text.load_font(kw["font"], size, kw["bold"], text=s)
        for ln in lines:
            core = ln.rstrip()
            tail = ln[len(core):]
            if tail:
                cw = width_height(core)[0] if core.strip() else 0.0
                w = max(w, cw + float(f.getlength(tail)) + letter_spacing * len(tail))
    return (w, h)


# --------------------------------------------------------------------------------------------- items

_D = object()                       # "not given" sentinel

_STYLE = dict(stroke=_D, stroke_width=None, dash=None, dash_offset=0.0, cap="butt", join="miter", miter_limit=4.0,
              fill=_D, fill_opacity=1.0, stroke_opacity=1.0, opacity=1.0, hatch=None, fill_rule="nonzero",
              marker_start=None, marker_end=None, arrow=None, marker_size=None, dash_fit=False)


_STYLE_ALIASES = {"stroke_color": "stroke", "fill_color": "fill", "line_width": "stroke_width", "lw": "stroke_width",
                  "alpha": "opacity", "dasharray": "dash", "stroke_dasharray": "dash", "linecap": "cap",
                  "linejoin": "join", "arrowhead": "arrow"}


def _split_style(kw, extra=(), owner="vector shape"):
    """Split keywords into style and ``extra`` options; unknown names raise TypeError (aliases accepted)."""
    from ._args import reject_unknown
    main = "stroke" if owner in ("line", "polyline", "arc", "bezier", "spline") else "fill"   # what ``color=`` means
    kw = reject_unknown(owner, kw, list(_STYLE) + list(extra), dict(_STYLE_ALIASES, color=main, colour=main))
    style, other = {}, {}
    for k, v in kw.items():
        (style if k in _STYLE else other)[k] = v
    return style, other


class Item:
    """Base class of everything in a drawing. ``name`` becomes the SVG id (and the layer name for groups)."""
    parent = None
    name = None
    drawing = None

    def _world(self):
        """Matrix from this item's parent coordinates to drawing coordinates."""
        m = _I
        g = self.parent
        while g is not None:
            m = _mul(g.matrix, m)
            g = g.parent
        return m

    def bbox(self, stroke=False):
        """Bounding box (x0, y0, x1, y1) in drawing units (after group transforms)."""
        return self._bbox(self._world(), stroke)

    def move(self, dx, dy):
        """Move by (dx, dy) in drawing units. Returns self."""
        m = self._world()
        lx, ly = m[0], m[1]
        inv = _inv((m[0], m[1], m[2], m[3], 0.0, 0.0))
        v = _apply(inv, (dx, dy))
        self._translate(float(v[0]), float(v[1]))
        return self

    def place(self, ref, side="right", gap=0.0, align="center"):
        """Move this item next to ``ref`` (an item or an (x0, y0, x1, y1) box in drawing units).

        ``side``: right / left / above / below / center (centred on ref). ``align`` along the other axis:
        start / center / end (top/left, middle, bottom/right). Returns self."""
        rb = ref.bbox() if isinstance(ref, Item) else tuple(ref)
        b = self.bbox()
        w, h = b[2] - b[0], b[3] - b[1]

        def along(lo, hi, size):
            return {"start": lo, "center": (lo + hi - size) / 2, "end": hi - size}[align]

        if side == "right":
            x, y = rb[2] + gap, along(rb[1], rb[3], h)
        elif side == "left":
            x, y = rb[0] - gap - w, along(rb[1], rb[3], h)
        elif side == "below":
            x, y = along(rb[0], rb[2], w), rb[3] + gap
        elif side == "above":
            x, y = along(rb[0], rb[2], w), rb[1] - gap - h
        elif side in ("center", "inside"):
            x, y = (rb[0] + rb[2] - w) / 2, (rb[1] + rb[3] - h) / 2
        else:
            raise ValueError("side must be right / left / above / below / center")
        return self.move(x - b[0], y - b[1])

    def remove(self):
        """Take this item out of the drawing."""
        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent = None
        return self


class Shape(Item):
    """A path-based primitive (line, rect, circle, path, ...). Path-building methods are chainable on
    shapes made with ``dwg.path()``."""

    def __init__(self, kind, path, params, style, name=None):
        self.kind, self.path, self.params, self.style, self.name = kind, path, params, style, name

    # path building (for dwg.path())
    def move_to(self, x, y):
        self.path.move_to(x, y)
        return self

    def line_to(self, x, y):
        self.path.line_to(x, y)
        return self

    def quad_to(self, qx, qy, x, y):
        self.path.quad_to(qx, qy, x, y)
        return self

    def cubic_to(self, x1, y1, x2, y2, x, y):
        self.path.cubic_to(x1, y1, x2, y2, x, y)
        return self

    def arc_to(self, rx, ry, rotation, large, sweep, x, y):
        self.path.arc_to(rx, ry, rotation, large, sweep, x, y)
        return self

    def bezier_to(self, *pts):
        self.path.bezier_to(*pts)
        return self

    def close(self):
        self.path.close()
        return self

    def _translate(self, dx, dy):
        self.path = self.path.transformed(_translate(dx, dy))
        p = self.params
        for kx, ky in (("x", "y"), ("cx", "cy"), ("x1", "y1"), ("x2", "y2")):
            if kx in p:
                p[kx] += dx
                p[ky] += dy
        if "points" in p:
            p["points"] = [(x + dx, y + dy) for x, y in p["points"]]

    def _bbox(self, m, stroke=False):
        pts = [_apply(m, p) for p, _ in self.path.flatten(0.05) if len(p)]
        if not pts:
            return (0.0, 0.0, 0.0, 0.0)
        P = np.concatenate(pts)
        b = [P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()]
        if stroke:
            st = _resolve(self, self.drawing)
            if st["stroke"] is not None:
                pad = st["stroke_width"] / 2 * _mscale(m)
                if st["marker_start"] or st["marker_end"]:
                    pad = max(pad, _marker_len(st) * 0.6 * _mscale(m))
                b = [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]
        return tuple(float(v) for v in b)


class Text(Item):
    """A text label (multi-line, anchored, optionally rotated)."""

    def __init__(self, x, y, text, size, fill, anchor, family, weight, style, rotate, letter_spacing,
                 line_spacing, align, max_width, halo, halo_width, opacity, name, markup=False):
        self.x, self.y, self.size = float(x), float(y), float(size)
        self.text = "\n".join(ln.rstrip() for ln in str(text).split("\n"))   # trailing spaces are invisible
        a = _anchor(anchor)
        self.cap = a[1] == "c"                              # 'c': middle of the capital height, first line
        self.fill, self.family, self.weight, self.style = fill, family, weight, style
        self.anchor = a[0] + "s" if self.cap else a
        self.rotate, self.letter_spacing, self.line_spacing = float(rotate), float(letter_spacing), line_spacing
        self.align, self.max_width, self.halo, self.halo_width = align, max_width, halo, halo_width
        self.opacity, self.name, self.markup = float(opacity), name, bool(markup)

    def _kw(self):
        return _tkw(self.family, self.weight, self.style, self.text)

    def _ay(self):
        """y of the layout anchor point (moved down to the baseline for cap-centred 'c' anchors)."""
        if not self.cap:
            return self.y
        kw = self._kw()
        return self.y + 0.5 * self.size * _cap_ratio(kw["font"], kw["bold"], _text.has_cjk(self.text))

    def _tracking(self):
        return self.letter_spacing / max(self.size, 1e-9)

    def _halign(self):
        return self.align or {"l": "left", "m": "center", "r": "right"}[self.anchor[0]]

    def _layout(self, k):
        """Layout at k px per unit → (None, lines [(text, x_left, baseline_y, width)], bbox) in user units."""
        size = self.size * k
        rows, bb, asc = _text_lines(self.text, size, self._kw(), None if self.max_width is None else
                                    self.max_width * k, self._tracking(), self.line_spacing, self.anchor,
                                    (self.x * k, self._ay() * k), self.markup)
        bw = bb[2] - bb[0]
        off = {"left": 0.0, "center": 0.5, "right": 1.0}.get(self._halign(), 0.0)
        lines = [(ln, (bb[0] + (bw - w) * off) / k, (top + asc) / k, w / k) for ln, w, top in rows]
        return None, lines, tuple(v / k for v in bb)

    def _matrix(self):
        return _rotate(self.rotate, self.x, self.y) if self.rotate else _I

    def _translate(self, dx, dy):
        self.x += dx
        self.y += dy

    def _bbox(self, m, stroke=False):
        k = self.drawing._ppu if self.drawing else 1.0
        x0, y0, x1, y1 = self._layout(max(k, 4.0 / max(self.size, 1e-9), 1.0))[2]
        P = _apply(_mul(m, self._matrix()), np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        return (float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max()))


class _Builder:
    """Primitive constructors shared by Drawing and Group."""

    # --- plumbing
    def _target(self):
        raise NotImplementedError

    def _dwg(self):
        raise NotImplementedError

    def _add(self, item, add=True):
        item.drawing = self._dwg()
        if add:
            tgt = self._target()
            item.parent = tgt
            tgt.children.append(item)
        return item

    def _shape(self, kind, path, params, kw):
        style, other = _split_style(kw, ("name", "add"), owner=kind)
        return self._add(Shape(kind, path, params, style, other.get("name")), other.get("add", True))

    # --- primitives
    def line(self, x1, y1, x2, y2, **style):
        """Straight line. Style: stroke, stroke_width, dash, cap, arrow / marker_start / marker_end, ..."""
        p = Path().move_to(x1, y1).line_to(x2, y2)
        return self._shape("line", p, dict(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)), style)

    def polyline(self, points, **style):
        """Open polyline through ``points`` [(x, y), ...]."""
        pts = [(float(x), float(y)) for x, y in points]
        p = Path().move_to(*pts[0])
        for q in pts[1:]:
            p.line_to(*q)
        return self._shape("polyline", p, dict(points=pts), style)

    def polygon(self, points, **style):
        """Closed polygon."""
        pts = [(float(x), float(y)) for x, y in points]
        p = Path().move_to(*pts[0])
        for q in pts[1:]:
            p.line_to(*q)
        return self._shape("polygon", p.close(), dict(points=pts), style)

    def rect(self, x, y, w, h, r=0.0, **style):
        """Rectangle at top-left (x, y), size w x h. ``r``: corner radius — a number, (rx, ry) for elliptic
        corners, or (top_left, top_right, bottom_right, bottom_left), e.g. ``r=(6, 6, 0, 0)`` for a bar rounded on
        top. Upright rectangles land on whole pixels (crisp edges) unless ``Drawing(crisp=False)``."""
        x, y, w, h = float(x), float(y), float(w), float(h)
        if w < 0:
            x, w = x + w, -w
        if h < 0:
            y, h = y + h, -h
        if isinstance(r, (tuple, list)) and len(r) == 4:
            radii = [min(max(float(v), 0.0), w / 2, h / 2) for v in r]
            if len(set(radii)) > 1:
                tl, tr, br, bl = radii
                p = Path().move_to(x + tl, y).line_to(x + w - tr, y)
                if tr:
                    p.arc_to(tr, tr, 0, 0, 1, x + w, y + tr)
                p.line_to(x + w, y + h - br)
                if br:
                    p.arc_to(br, br, 0, 0, 1, x + w - br, y + h)
                p.line_to(x + bl, y + h)
                if bl:
                    p.arc_to(bl, bl, 0, 0, 1, x, y + h - bl)
                p.line_to(x, y + tl)
                if tl:
                    p.arc_to(tl, tl, 0, 0, 1, x + tl, y)
                return self._shape("rect", p.close(), dict(x=x, y=y, w=w, h=h, rx=max(radii), ry=max(radii),
                                                           radii=tuple(radii)), style)
            r = radii[0]
        rx, ry = (r, r) if not isinstance(r, (tuple, list)) else r
        rx, ry = min(max(float(rx), 0.0), w / 2), min(max(float(ry), 0.0), h / 2)
        p = Path()
        if rx > 0 and ry > 0:
            p.move_to(x + rx, y).line_to(x + w - rx, y).arc_to(rx, ry, 0, 0, 1, x + w, y + ry)
            p.line_to(x + w, y + h - ry).arc_to(rx, ry, 0, 0, 1, x + w - rx, y + h)
            p.line_to(x + rx, y + h).arc_to(rx, ry, 0, 0, 1, x, y + h - ry)
            p.line_to(x, y + ry).arc_to(rx, ry, 0, 0, 1, x + rx, y).close()
        else:
            p.move_to(x, y).line_to(x + w, y).line_to(x + w, y + h).line_to(x, y + h).close()
        return self._shape("rect", p, dict(x=x, y=y, w=w, h=h, rx=rx, ry=ry), style)

    def circle(self, cx, cy, r, **style):
        """Circle centred on (cx, cy)."""
        return self.ellipse(cx, cy, r, r, _kind="circle", **style)

    def ellipse(self, cx, cy, rx, ry, _kind="ellipse", **style):
        """Axis-aligned ellipse."""
        cx, cy, rx, ry = float(cx), float(cy), abs(float(rx)), abs(float(ry))
        p = Path().move_to(cx + rx, cy)
        p.subs[-1][1].extend(_ellipse_arc(cx, cy, rx, ry, 0.0, 0.0, 2 * math.pi))
        p._pen = (cx + rx, cy)
        p.subs[-1][1][-1] = p.subs[-1][1][-1][:5] + (cx + rx, cy)
        return self._shape(_kind, p.close(), dict(cx=cx, cy=cy, rx=rx, ry=ry), style)

    def arc(self, cx, cy, r, start, end, **style):
        """Open circular arc from angle ``start`` to ``end`` (degrees, clockwise from +x)."""
        a0, a1 = math.radians(start), math.radians(end)
        p = Path().move_to(cx + r * math.cos(a0), cy + r * math.sin(a0))
        p.subs[-1][1].extend(_ellipse_arc(cx, cy, r, r, 0.0, a0, a1 - a0))
        p._pen = (cx + r * math.cos(a1), cy + r * math.sin(a1))
        return self._shape("arc", p, {}, style)

    def pie(self, cx, cy, r, start, end, inner=0.0, **style):
        """Pie slice / sector from ``start`` to ``end`` degrees (clockwise from +x); ``inner`` > 0 → donut slice."""
        a0, a1 = math.radians(start), math.radians(end)
        p = Path()
        if inner > 0:
            p.move_to(cx + r * math.cos(a0), cy + r * math.sin(a0))
            p.subs[-1][1].extend(_ellipse_arc(cx, cy, r, r, 0.0, a0, a1 - a0))
            p.subs[-1][1].append(("L", cx + inner * math.cos(a1), cy + inner * math.sin(a1)))
            p.subs[-1][1].extend(_ellipse_arc(cx, cy, inner, inner, 0.0, a1, a0 - a1))
        else:
            p.move_to(cx, cy).line_to(cx + r * math.cos(a0), cy + r * math.sin(a0))
            p.subs[-1][1].extend(_ellipse_arc(cx, cy, r, r, 0.0, a0, a1 - a0))
        p._pen = p.subs[-1][0]
        return self._shape("pie", p.close(), {}, style)

    sector = pie

    def path(self, d=None, **style):
        """Free path: SVG path data ``d`` ("M10 10 C ...") and/or chain ``.move_to().line_to().cubic_to()...``."""
        return self._shape("path", Path(d) if isinstance(d, str) else (d.copy() if isinstance(d, Path) else Path()),
                           {}, style)

    def bezier(self, points, **style):
        """Bezier curve from control points: 3 → quadratic, 4 → cubic, 3k+1 → chain of cubics."""
        pts = [tuple(map(float, p)) for p in points]
        p = Path().move_to(*pts[0]).bezier_to(*pts[1:])
        return self._shape("path", p, {}, style)

    def spline(self, points, closed=False, tension=0.5, **style):
        """Smooth curve through all ``points`` (Catmull-Rom); good for chart lines and organic outlines."""
        return self._shape("path", _catmull(points, closed, tension), {}, style)

    def text(self, x, y, text, size=None, fill=None, anchor="lt", family=None, weight=None, style="normal",
             rotate=0.0, letter_spacing=0.0, line_spacing=1.2, align=None, max_width=None, halo=None,
             halo_width=None, opacity=1.0, name=None, add=True, markup=False):
        """Text at (x, y). ``anchor``: two letters, horizontal l/m/r + vertical t/m/b/s/c (s = first baseline,
        c = middle of the capital height: 'mc' centres numbers and capitals optically in circles and boxes), e.g.
        'lt', 'mm', 'mc', 'rb', 'ls' (or 'center', 'left', 'top-right'...). ``size`` in user units; ``family``:
        sans / serif / mono / a font name; ``weight``: normal / bold / 100-900; ``style``: normal / italic;
        ``rotate`` degrees clockwise about (x, y); ``"\\n"`` for multi-line; ``max_width`` wraps;
        ``halo`` = outline colour behind the glyphs (labels over busy backgrounds). Text is plain by default
        (``**``, ``__``, ``~~``, ``==`` print as typed); ``markup=True`` turns on pixelcanvas.text light markup."""
        d = self._dwg()
        th = d.theme
        size = size if size is not None else (d._default_sw * 9.5)
        fam = family or (th["font"]["family"] if th else "sans")
        wt = weight or (th["font"]["weight"] if th else "normal")
        hw = halo_width if halo_width is not None else size * 0.12
        t = Text(x, y, text, size, fill, anchor, fam, wt, style, rotate, letter_spacing, line_spacing, align,
                 max_width, halo, hw, opacity, name, markup)
        return self._add(t, add)

    def icon(self, name, x, y, size=24, color=None, stroke_width=2.0, fill=None, anchor="mm", opacity=1.0,
             label=None):
        """Place icon ``name`` (see ``icon_names()``) at (x, y), ``size`` units square. ``anchor`` as for text
        (default centred on x, y). ``stroke_width`` is in 24-grid units; ``fill`` tints the icon body (duotone).
        Returns the icon's Group."""
        if name not in ICONS:
            raise _unknown("icon", name, ICONS, "Try icon_names()")
        a = _anchor(anchor)
        ox = {"l": 0.0, "m": 0.5, "r": 1.0}[a[0]] * size
        oy = {"t": 0.0, "a": 0.0, "m": 0.5, "c": 0.5, "s": 1.0, "b": 1.0, "d": 1.0}[a[1]] * size
        k = size / 24.0
        g = Group(self._dwg(), label or f"icon {name}", translate=(x - ox, y - oy), scale=k, opacity=opacity)
        self._add(g)
        _build_icon(g, name, color, stroke_width, fill)
        return g

    def group(self, name=None, translate=None, rotate=0.0, scale=1.0, origin=None, opacity=1.0, clip=None,
              shadow=None):
        """Group context manager: ``with dwg.group("nodes", translate=(10, 0), opacity=0.9): ...``.

        Transform = translate, then rotate (degrees, about ``origin``), then scale (number or (sx, sy)).
        ``clip``: (x0, y0, x1, y1) box or a shape made with ``add=False`` (group-local coordinates).
        ``shadow``: True (theme shadow) or dict(dx, dy, blur, color, opacity) in user units.
        Each top-level group is painted as its own layer step named ``name``."""
        g = Group(self._dwg(), name, translate=translate, rotate=rotate, scale=scale, origin=origin,
                  opacity=opacity, clip=clip, shadow=shadow)
        return self._add(g)

    def embed(self, drawing, x=0.0, y=0.0, w=None, h=None, name=None, box=None, align="center"):
        """Insert a copy of another Drawing's content at (x, y), scaled to width ``w`` (or height ``h``).

        ``box=(x0, y0, x1, y1)`` instead scales it uniformly to fit the box; ``align`` places it inside:
        center / top / bottom / left / right / top-left / bottom-right ... Returns the Group."""
        if box is not None:
            x0, y0, x1, y1 = (float(v) for v in box)
            k = min((x1 - x0) / drawing.width, (y1 - y0) / drawing.height)
            fw, fh = drawing.width * k, drawing.height * k
            al = str(align).lower()
            x = x0 if "left" in al else (x1 - fw if "right" in al else x0 + (x1 - x0 - fw) / 2)
            y = y0 if "top" in al else (y1 - fh if "bottom" in al else y0 + (y1 - y0 - fh) / 2)
        else:
            k = (w / drawing.width) if w else ((h / drawing.height) if h else 1.0)
        g = Group(self._dwg(), name or drawing.name, translate=(x, y), scale=k)
        self._add(g)
        for ch in drawing.root.children:
            c = _clone(ch, self._dwg())
            c.parent = g
            g.children.append(c)
        return g


def _clone(item, dwg):
    par = item.parent
    item.parent = None
    try:
        c = _copy.deepcopy(item, {id(dwg): dwg, id(item.drawing): dwg} if item.drawing is not None else {})
    finally:
        item.parent = par
    _reparent(c, dwg)
    return c


def _reparent(item, dwg):
    item.drawing = dwg
    if isinstance(item, Group):
        for ch in item.children:
            ch.parent = item
            _reparent(ch, dwg)


class Group(Item, _Builder):
    """A named group with its own transform, opacity, clip and shadow. Use as a context manager."""

    def __init__(self, drawing, name=None, translate=None, rotate=0.0, scale=1.0, origin=None, opacity=1.0,
                 clip=None, shadow=None):
        self.drawing, self.name, self.children = drawing, name, []
        self.tx, self.ty = (float(translate[0]), float(translate[1])) if translate is not None else (0.0, 0.0)
        self.rot = float(rotate or 0.0)
        self.origin = tuple(origin) if origin is not None else (0.0, 0.0)
        self.sx, self.sy = (scale, scale) if not isinstance(scale, (tuple, list)) else scale
        self.opacity = float(opacity)
        self.clip = clip
        self.shadow = shadow

    def _target(self):
        return self

    def _dwg(self):
        return self.drawing

    def __enter__(self):
        self.drawing._stack.append(self)
        return self

    def __exit__(self, *exc):
        self.drawing._stack.remove(self)
        return False

    @property
    def matrix(self):
        m = _translate(self.tx, self.ty)
        if self.rot:
            m = _mul(m, _rotate(self.rot, *self.origin))
        if self.sx != 1 or self.sy != 1:
            m = _mul(m, _scale(self.sx, self.sy))
        return m

    def _translate(self, dx, dy):
        self.tx += dx
        self.ty += dy

    def _bbox(self, m, stroke=False):
        mm = _mul(m, self.matrix)
        bs = [c._bbox(mm, stroke) for c in self.children]
        if not bs:
            return (mm[4], mm[5], mm[4], mm[5])
        B = np.array(bs)
        return (float(B[:, 0].min()), float(B[:, 1].min()), float(B[:, 2].max()), float(B[:, 3].max()))

    def _needs_tile(self):
        return self.opacity < 1.0 or self.clip is not None or bool(self.shadow)

    def _shadow(self):
        if not self.shadow:
            return None
        th = self.drawing.theme or THEMES["light"]
        s = dict(th["shadow"])
        u = self.drawing._default_sw / 1.5
        s.update(dx=s["dx"] * u, dy=s["dy"] * u, blur=s["blur"] * u)
        if isinstance(self.shadow, dict):
            s.update(self.shadow)
        return s

    def find(self, name):
        """First descendant with this name (or None)."""
        for c in self.children:
            if c.name == name:
                return c
            if isinstance(c, Group):
                r = c.find(name)
                if r is not None:
                    return r
        return None


def _clip_path(clip):
    if isinstance(clip, Shape):
        return clip.path, clip.style.get("fill_rule", "nonzero") == "evenodd"
    x0, y0, x1, y1 = (float(v) for v in clip)
    return Path().move_to(x0, y0).line_to(x1, y0).line_to(x1, y1).line_to(x0, y1).close(), False


# --------------------------------------------------------------------------------------------- style resolution

def _marker_len(st):
    return st["marker_size"] if st["marker_size"] else st["stroke_width"] * 5.0


def _resolve(shape, dwg):
    s = dict(_STYLE)
    s.update(shape.style)
    ink = dwg.theme["ink"] if dwg is not None and dwg.theme else "#1d2633"
    closed = shape.kind in ("rect", "circle", "ellipse", "polygon", "pie") or \
        (shape.kind == "path" and shape.path.closed)
    if s["fill"] is _D:
        s["fill"] = ink if (closed and s["stroke"] in (_D, None) and not s["hatch"]) else None
    if s["stroke"] is _D:
        s["stroke"] = ink if (not closed or s["fill"] is None) and not (closed and s["hatch"]) else None
    if s["stroke_width"] is None:
        s["stroke_width"] = dwg._default_sw if dwg is not None else 1.5
    if not s["marker_size"]:                     # grows with the stroke, never vanishingly small
        s["marker_size"] = 4.0 * s["stroke_width"] + 2.7 * (dwg._default_sw if dwg is not None else 1.5)
    if s["arrow"]:
        a = str(s["arrow"])
        if a in ("end", "both", "true", "True") and not s["marker_end"]:
            s["marker_end"] = "arrow"
        if a in ("start", "both") and not s["marker_start"]:
            s["marker_start"] = "arrow"
    for k in ("marker_start", "marker_end"):
        if s[k] in ("none", False):
            s[k] = None
        if s[k] and s[k] not in MARKERS:
            raise _unknown("marker", s[k], MARKERS)
    if shape.kind in ("rect", "circle", "ellipse", "polygon", "pie"):
        s["marker_start"] = s["marker_end"] = None
    if isinstance(s["dash"], str):
        pat = DASHES[s["dash"]]
        sw = s["stroke_width"]
        s["dash"] = tuple(max(v, 0.0) * sw for v in pat)
        if 0 in pat and "cap" not in shape.style:
            s["cap"] = "round"
    if s["hatch"] is True:
        s["hatch"] = _Hatch()
    elif isinstance(s["hatch"], (int, float)) and not isinstance(s["hatch"], bool):
        s["hatch"] = _Hatch(angle=s["hatch"])
    elif isinstance(s["hatch"], dict):
        s["hatch"] = _Hatch(**s["hatch"])
    return s


def _paint(p):
    """Normalise a paint → ('solid', rgb, alpha) / ('grad', gradient) / None."""
    if p is None:
        return None
    if isinstance(p, _Gradient):
        return ("grad", p)
    c = _color(p)
    if c is None:
        return None
    return ("solid", c[0], c[1])


def _markers_geom(path, st):
    """Marker geometry in local units → (trimmed path, [('fill', poly) | ('stroke', pts, closed)])."""
    sw = st["stroke_width"]
    L = _marker_len(st)
    shapes = []
    trim = {}
    for which, kind in (("start", st["marker_start"]), ("end", st["marker_end"])):
        if not kind:
            continue
        info = path._end(which)
        if info is None:
            continue
        p0, sg = info                                   # segment start point, segment
        tmp = Path().move_to(*p0)
        tmp.subs[-1][1].append(sg)
        # direction: chord from the point at distance L along the curve to the tip
        if which == "end":
            tip = np.array(sg[-2:])
            other = np.array(sg[3:5]) if sg[0] == "C" else np.array(p0)
            back = np.array(tmp.trimmed(at_end=L).subs[0][1][-1][-2:])
        else:
            tip = np.array(p0)
            other = np.array(sg[1:3])
            back = np.array(tmp.trimmed(at_start=L).subs[0][0])
        d = tip - back
        if np.hypot(*d) < 1e-9:
            d = tip - other
        if np.hypot(*d) < 1e-9:
            continue
        d = d / np.hypot(*d)
        n = np.array([-d[1], d[0]])
        if kind == "arrow":
            shapes.append(("fill", np.array([tip, tip - d * L + n * L * 0.36, tip - d * L * 0.84,
                                             tip - d * L - n * L * 0.36])))
            trim[which] = L * 0.62
        elif kind == "triangle":
            shapes.append(("fill", np.array([tip, tip - d * L + n * L * 0.3, tip - d * L - n * L * 0.3])))
            trim[which] = L * 0.8
        elif kind == "open":
            t2 = tip - d * sw * 0.5
            shapes.append(("stroke", np.array([t2 - d * L + n * L * 0.4, t2, t2 - d * L - n * L * 0.4]), False))
            trim[which] = sw * 0.5
        elif kind == "dot":
            r = max(L * 0.3, sw * 1.2)
            t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            shapes.append(("fill", tip + np.stack([np.cos(t), np.sin(t)], -1) * r))
        elif kind == "bar":
            shapes.append(("stroke", np.array([tip + n * L * 0.45, tip - n * L * 0.45]), False))
        elif kind == "diamond":
            shapes.append(("fill", np.array([tip, tip - d * L * 0.5 + n * L * 0.3, tip - d * L,
                                             tip - d * L * 0.5 - n * L * 0.3])))
            trim[which] = L * 0.5
    sp = path.trimmed(trim.get("start", 0.0), trim.get("end", 0.0)) if trim else path
    return sp, shapes


# --------------------------------------------------------------------------------------------- raster drawing

class _RCtx:
    def __init__(self, dwg, bounds, W, H):
        self.dwg, self.bounds, self.W, self.H = dwg, bounds, W, H

    # ---- collect paint operations
    def collect(self, item, M, ops, jobs):
        if isinstance(item, Group):
            Mc = _mul(M, item.matrix)
            if item._needs_tile():
                ops.append(("group", item, Mc))
            else:
                for c in item.children:
                    self.collect(c, Mc, ops, jobs)
        elif isinstance(item, Shape):
            self._shape(item, M, ops, jobs)
        elif isinstance(item, Text):
            self._text(item, M, ops)

    def _shape(self, sh, M, ops, jobs):
        st = _resolve(sh, self.dwg)
        s = _mscale(M)
        if s < 1e-9:
            return
        tol = _TOL / s
        fill, strk = _paint(st["fill"]), _paint(st["stroke"])
        op = st["opacity"]
        crisp = self.dwg.crisp
        dbox = None
        if crisp and sh.kind == "rect" and _upright(M):          # device box of a crisp rectangle
            q = sh.params
            dbox = (M[0] * q["x"] + M[4], M[3] * q["y"] + M[5], M[0] * (q["x"] + q["w"]) + M[4],
                    M[3] * (q["y"] + q["h"]) + M[5])
        if (fill or st["hatch"]) and sh.kind not in ("line", "polyline", "arc"):
            polys = []
            for p, _ in sh.path.flatten(tol):
                if len(p) >= 3:
                    d = _apply(M, p)
                    if dbox is not None:
                        d = _box_snap(d, dbox, np.round)
                    polys.append(d[None])
            if polys:
                jobs.append((polys, st["fill_rule"] == "evenodd"))
                jid = len(jobs) - 1
                if fill:
                    ops.append(("paint", jid, fill, st["fill_opacity"] * op, M, sh))
                if st["hatch"]:
                    h = st["hatch"]
                    col = h.color or (st["stroke"] if st["stroke"] is not None else
                                      (self.dwg.theme["ink"] if self.dwg.theme else "#1d2633"))
                    hp = _paint(col)
                    if hp:
                        ops.append(("hatch", jid, hp, h.opacity * op, M, sh, h))
        if strk and st["stroke_width"] > 0:
            hw = st["stroke_width"] / 2
            path, marks = (_markers_geom(sh.path, st) if (st["marker_start"] or st["marker_end"])
                           else (sh.path, []))
            out = []
            conf = _conformal(M)
            lines = path.flatten(tol)
            pieces, twos = [], []
            if st["dash"]:
                for p, c in lines:
                    pat, off = st["dash"], st["dash_offset"]
                    if st["dash_fit"]:
                        pat, off = _fit_dash(pat, _poly_len(p, c), c, st["dash_fit"] == "center"), 0.0
                        if pat is None:
                            pieces.append((p, c))
                            continue
                    lg, tw = _dash(p, c, pat, off)
                    pieces += lg
                    if tw is not None and len(tw[0]):
                        twos.append(tw)
            else:
                pieces = lines
            cap, join, ml = st["cap"], st["join"], st["miter_limit"]
            if conf:
                hwd = hw * s
                lin = (M[0], M[1], M[2], M[3], 0.0, 0.0)
                rbox = None
                if dbox is not None and not marks:
                    k_ = round(2 * hwd)
                    if 1 <= k_ <= 9 and abs(2 * hwd - k_) <= 0.08:
                        rbox = (lambda v: np.floor(v) + 0.5) if k_ % 2 else np.round
                for p, c in pieces:
                    d = _apply(M, p)
                    if rbox is not None:
                        d = _box_snap(d, dbox, rbox)
                    elif crisp:
                        d = _snap(d, hwd)
                    _stroke_polyline(d, c, hwd, cap, join, ml, _TOL, out)
                for A, B, T in twos:
                    Td = _apply(lin, T)
                    Td /= np.maximum(np.hypot(Td[:, 0], Td[:, 1]), 1e-12)[:, None]
                    _stroke_segments(_apply(M, A), _apply(M, B), Td, hwd, cap, _TOL, out)
            else:
                loc = []
                for p, c in pieces:
                    _stroke_polyline(p, c, hw, cap, join, ml, tol, loc)
                for A, B, T in twos:
                    _stroke_segments(A, B, T, hw, cap, tol, loc)
                out += [_pos(_apply(M, b)) for b in loc]
            for mk in marks:
                if mk[0] == "fill":
                    out.append(_pos(_apply(M, mk[1])[None]))
                else:
                    loc = []
                    _stroke_polyline(mk[1], mk[2], hw, "round" if len(mk[1]) > 2 else "butt", "round", 4.0, tol, loc)
                    out += [_pos(_apply(M, b)) for b in loc]
            if out:
                jobs.append((out, False))
                ops.append(("paint", len(jobs) - 1, strk, st["stroke_opacity"] * op, M, sh))

    def _text(self, t, M, ops):
        T = _mul(M, t._matrix())
        s = _mscale(T)
        size = t.size * s
        if size < 1.0 or not t.text.strip():
            return
        track = t.letter_spacing / max(t.size, 1e-9)
        mw = None if t.max_width is None else t.max_width * s
        layers = []
        if t.halo:
            layers.append((_paint(t.halo), t.halo_width * s))
        layers.append((_paint(t.fill if t.fill is not None else
                              (self.dwg.theme["ink"] if self.dwg.theme else "#1d2633")), 0.0))
        bx0, by0, bx1, by1 = self.bounds
        tk = dict(anchor=t.anchor, align=t.align, max_width=mw, line_spacing=t.line_spacing, tracking=track,
                  markup=t.markup, **t._kw())
        ty = t._ay()
        pad = int(math.ceil(max(lw for _, lw in layers) + 2))
        if _upright(T):
            ax, ay = _apply(T, (t.x, ty))
            bb = _text.text_bbox(t.text, (ax, ay), size, **tk)
            x0 = max(int(math.floor(bb[0])) - pad, bx0)
            y0 = max(int(math.floor(bb[1])) - pad, by0)
            x1 = min(int(math.ceil(bb[2])) + pad, bx1)
            y1 = min(int(math.ceil(bb[3])) + pad, by1)
            if x1 <= x0 or y1 <= y0:
                return
            for paint, lw in layers:
                if paint is None:
                    continue
                cov = _text.text_mask(y1 - y0, x1 - x0, t.text, (ax - x0, ay - y0), size, stroke=lw, aa=3, **tk)
                ops.append(("cov", x0, y0, np.asarray(cov, F32), paint, t.opacity, M, t))
            return
        bb = _text.text_bbox(t.text, (0.0, 0.0), size, **tk)
        iw = int(math.ceil(bb[2] - bb[0])) + 2 * pad
        ih = int(math.ceil(bb[3] - bb[1])) + 2 * pad
        ox, oy = -bb[0] + pad, -bb[1] + pad                    # anchor position inside the image
        # device point q -> text image: i = (T^-1(q) - (x, y)) * s + (ox, oy)
        Ti = _inv(T)
        g = _mul((s, 0.0, 0.0, s, ox - t.x * s, oy - ty * s), Ti)
        corners = _apply(T, np.array([[t.x + (u - ox) / s, ty + (v - oy) / s] for u, v in
                                      ((0, 0), (iw, 0), (iw, ih), (0, ih))]))
        x0 = max(int(math.floor(corners[:, 0].min())), bx0)
        y0 = max(int(math.floor(corners[:, 1].min())), by0)
        x1 = min(int(math.ceil(corners[:, 0].max())), bx1)
        y1 = min(int(math.ceil(corners[:, 1].max())), by1)
        if x1 <= x0 or y1 <= y0:
            return
        g = _mul(g, _translate(x0, y0))
        for paint, lw in layers:
            if paint is None:
                continue
            cov = _text.text_mask(ih, iw, t.text, (ox, oy), size, stroke=lw, aa=3, **tk)
            im = Image.fromarray(np.asarray(cov, np.float32), "F")
            im = im.transform((x1 - x0, y1 - y0), _TRANSFORM.AFFINE, (g[0], g[2], g[4], g[1], g[3], g[5]),
                              resample=_RES.BILINEAR)
            ops.append(("cov", x0, y0, np.clip(np.asarray(im, F32), 0, 1), paint, t.opacity, M, t))

    # ---- render a list of items into one premultiplied tile
    def render_items(self, items, M, shadow=None, clip=None, opacity=1.0):
        ops, jobs = [], []
        for it in items:
            self.collect(it, M, ops, jobs)
        covs = _raster(jobs, self.bounds)
        layers = []
        for op in ops:
            if op[0] in ("paint", "hatch"):
                c = covs[op[1]]
                if c is not None:
                    layers.append((op, c[0], c[1], c[2].shape[1], c[2].shape[0], c[2]))
            elif op[0] == "cov":
                layers.append((op, op[1], op[2], op[3].shape[1], op[3].shape[0], op[3]))
            else:
                tile = self.render_group(op[1], op[2])
                if tile is not None:
                    layers.append((("tile",), tile[0], tile[1], tile[2].shape[1], tile[2].shape[0], tile))
        if not layers:
            return None
        tx0 = min(l[1] for l in layers)
        ty0 = min(l[2] for l in layers)
        tx1 = max(l[1] + l[3] for l in layers)
        ty1 = max(l[2] + l[4] for l in layers)
        prem = np.zeros((ty1 - ty0, tx1 - tx0, 3), F32)
        alpha = np.zeros((ty1 - ty0, tx1 - tx0), F32)
        for op, x0, y0, w, h, data in layers:
            sl = (slice(y0 - ty0, y0 - ty0 + h), slice(x0 - tx0, x0 - tx0 + w))
            if op[0] == "tile":
                _, _, tp, ta = data
                inv = 1.0 - ta
                prem[sl] = prem[sl] * inv[..., None] + tp
                alpha[sl] = alpha[sl] * inv + ta
                continue
            cov = data
            paint = op[2] if op[0] != "cov" else op[4]
            a = op[3] if op[0] != "cov" else op[5]
            Mp = op[4] if op[0] != "cov" else op[6]
            item = op[5] if op[0] != "cov" else op[7]
            if op[0] == "hatch":
                cov = cov * self._hatch_cov(op[6], x0, y0, w, h, Mp)
            col, al = self._paint_field(paint, x0, y0, w, h, Mp, item)
            A = cov * (al * a)
            inv = 1.0 - A
            P = prem[sl]
            P *= inv[..., None]
            P += (col * A[..., None]) if np.ndim(col) == 3 else (A[..., None] * col)
            alpha[sl] = alpha[sl] * inv + A
        return [tx0, ty0, prem, alpha]

    def _local_grid(self, x0, y0, w, h, M):
        Y, X = np.mgrid[0:h, 0:w].astype(np.float64)
        Mi = _inv(M)
        X += x0 + 0.5
        Y += y0 + 0.5
        return Mi[0] * X + Mi[2] * Y + Mi[4], Mi[1] * X + Mi[3] * Y + Mi[5]

    def _paint_field(self, paint, x0, y0, w, h, M, item):
        if paint[0] == "solid":
            return paint[1], paint[2]
        g = paint[1]
        lx, ly = self._local_grid(x0, y0, w, h, M)
        if g.units == "bbox":
            if isinstance(item, Text):
                k = max(self.dwg._ppu, 1.0)
                bx0, by0, bx1, by1 = item._layout(k)[2]
            else:
                bx0, by0, bx1, by1 = item.path.bbox()
            lx = (lx - bx0) / max(bx1 - bx0, 1e-9)
            ly = (ly - by0) / max(by1 - by0, 1e-9)
        if g.kind == "linear":
            (sx, sy), (ex, ey) = g.start, g.end
            dx, dy = ex - sx, ey - sy
            t = ((lx - sx) * dx + (ly - sy) * dy) / max(dx * dx + dy * dy, 1e-12)
        else:
            t = np.hypot(lx - g.center[0], ly - g.center[1]) / max(g.radius, 1e-12)
        pos = np.array([s[0] for s in g.stops])
        t = np.clip(t, pos[0], pos[-1])
        cols = np.stack([s[1] for s in g.stops])
        col = np.stack([np.interp(t, pos, cols[:, c]) for c in range(3)], -1).astype(F32)
        al = np.interp(t, pos, np.array([s[2] for s in g.stops])).astype(F32)
        return col, al

    def _hatch_cov(self, h, x0, y0, w, h_, M):
        lx, ly = self._local_grid(x0, y0, w, h_, M)
        sw = self.dwg._default_sw
        sp = h.spacing if h.spacing else sw * 4.0
        lw = h.width if h.width else sw * 0.6
        s = _mscale(M)
        th = math.radians(h.angle)

        def lines(v):
            d = np.abs(np.mod(v / sp, 1.0) - 0.5) * sp * s
            return np.clip(lw * s / 2 - d + 0.5, 0.0, 1.0)

        cov = lines(-math.sin(th) * lx + math.cos(th) * ly)
        if h.cross:
            cov = np.maximum(cov, lines(math.cos(th) * lx + math.sin(th) * ly))
        return cov.astype(F32)

    def render_group(self, g, M):
        """Render a group (M already includes its own transform) → [x0, y0, prem, alpha] or None."""
        tile = self.render_items(g.children, M)
        if tile is None:
            return None
        sh = g._shadow()
        s = _mscale(M)
        if sh:
            tile = self._apply_shadow(tile, sh, M)
        if g.clip is not None:
            path, eo = _clip_path(g.clip)
            polys = [_apply(M, p)[None] for p, _ in path.flatten(_TOL / max(s, 1e-9)) if len(p) >= 3]
            c = _raster([(polys, eo)], self.bounds)[0]
            x0, y0, P, A = tile
            mask = np.zeros_like(A)
            if c is not None:
                cx0, cy0, cv = c
                ix0, iy0 = max(cx0, x0), max(cy0, y0)
                ix1 = min(cx0 + cv.shape[1], x0 + A.shape[1])
                iy1 = min(cy0 + cv.shape[0], y0 + A.shape[0])
                if ix1 > ix0 and iy1 > iy0:
                    mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = cv[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0]
            P *= mask[..., None]
            A *= mask
        if g.opacity < 1.0:
            tile[2] *= g.opacity
            tile[3] *= g.opacity
        return tile

    def _apply_shadow(self, tile, sh, M):
        from .filters import gaussian_blur
        x0, y0, P, A = tile
        lin = (M[0], M[1], M[2], M[3], 0.0, 0.0)
        dx, dy = _apply(lin, (sh["dx"], sh["dy"]))
        blur = sh["blur"] * _mscale(M)
        pad = int(math.ceil(blur * 3 + max(abs(dx), abs(dy)))) + 1
        bx0, by0, bx1, by1 = self.bounds
        nx0, ny0 = max(x0 - pad, bx0), max(y0 - pad, by0)
        nx1, ny1 = min(x0 + A.shape[1] + pad, bx1), min(y0 + A.shape[0] + pad, by1)
        H, W = ny1 - ny0, nx1 - nx0
        P2 = np.zeros((H, W, 3), F32)
        A2 = np.zeros((H, W), F32)
        sl = (slice(y0 - ny0, y0 - ny0 + A.shape[0]), slice(x0 - nx0, x0 - nx0 + A.shape[1]))
        P2[sl], A2[sl] = P, A
        S = np.zeros_like(A2)
        ix, iy = int(round(dx)), int(round(dy))
        src = A2[max(0, -iy):H - max(0, iy), max(0, -ix):W - max(0, ix)]
        S[max(0, iy):max(0, iy) + src.shape[0], max(0, ix):max(0, ix) + src.shape[1]] = src
        if blur > 0.3:
            S = gaussian_blur(S, blur)
        c = _color(sh["color"]) or (np.zeros(3, F32), 1.0)
        S = np.clip(S * float(sh["opacity"]) * c[1], 0, 1) * (1.0 - A2)
        P2 += S[..., None] * c[0]
        A2 += S
        return [nx0, ny0, P2, A2]


# --------------------------------------------------------------------------------------------- SVG writing

def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


class _SCtx:
    def __init__(self, dwg):
        self.dwg, self.defs, self.ids, self._n = dwg, [], set(), 0
        self._grad = {}

    def uid(self, base):
        base = re.sub(r"[^A-Za-z0-9_\-]+", "-", str(base)).strip("-") or "g"
        if not re.match(r"[A-Za-z_]", base):
            base = "id-" + base
        cand, k = base, 2
        while cand in self.ids:
            cand = f"{base}-{k}"
            k += 1
        self.ids.add(cand)
        return cand

    def paint(self, p, attr):
        """→ attribute string for fill / stroke paint (registers gradient defs)."""
        pp = _paint(p)
        if pp is None:
            return f' {attr}="none"', 1.0
        if pp[0] == "solid":
            return f' {attr}="{_hex(pp[1])}"', pp[2]
        g = pp[1]
        key = id(g)
        if key not in self._grad:
            gid = self.uid("grad")
            units = "objectBoundingBox" if g.units == "bbox" else "userSpaceOnUse"
            stops = "".join(f'<stop offset="{_n(pos)}" stop-color="{_hex(c)}"'
                            + (f' stop-opacity="{_n(a)}"' if a < 1 else "") + "/>" for pos, c, a in g.stops)
            if g.kind == "linear":
                self.defs.append(f'<linearGradient id="{gid}" gradientUnits="{units}" x1="{_n(g.start[0])}" '
                                 f'y1="{_n(g.start[1])}" x2="{_n(g.end[0])}" y2="{_n(g.end[1])}">{stops}'
                                 f'</linearGradient>')
            else:
                self.defs.append(f'<radialGradient id="{gid}" gradientUnits="{units}" cx="{_n(g.center[0])}" '
                                 f'cy="{_n(g.center[1])}" r="{_n(g.radius)}">{stops}</radialGradient>')
            self._grad[key] = gid
        return f' {attr}="url(#{self._grad[key]})"', 1.0


def _stroke_attrs(st, ctx):
    a, al = ctx.paint(st["stroke"], "stroke")
    if st["stroke"] is None or _paint(st["stroke"]) is None or st["stroke_width"] <= 0:
        return ' stroke="none"'
    s = a + f' stroke-width="{_n(st["stroke_width"])}"'
    if st["cap"] != "butt":
        s += f' stroke-linecap="{st["cap"]}"'
    if st["join"] != "miter":
        s += f' stroke-linejoin="{st["join"]}"'
    elif st["miter_limit"] != 4:
        s += f' stroke-miterlimit="{_n(st["miter_limit"])}"'
    if st["dash"]:
        pat = list(st["dash"])
        s += ' stroke-dasharray="' + ",".join(_n(max(v, 0.0)) for v in pat) + '"'
        if st["dash_offset"]:
            s += f' stroke-dashoffset="{_n(st["dash_offset"])}"'
    o = al * st["stroke_opacity"] * st["opacity"]
    if o < 1:
        s += f' stroke-opacity="{_n(o)}"'
    return s


def _geom_el(sh, attrs, path=None):
    """Native SVG element for a shape (or <path> for ``path``)."""
    p = sh.params
    if path is None:
        if sh.kind == "rect" and "radii" not in p:
            r = ""
            if p["rx"] > 0:
                r = f' rx="{_n(p["rx"])}" ry="{_n(p["ry"])}"'
            return (f'<rect x="{_n(p["x"])}" y="{_n(p["y"])}" width="{_n(p["w"])}" height="{_n(p["h"])}"{r}'
                    f'{attrs}/>')
        if sh.kind == "circle":
            return f'<circle cx="{_n(p["cx"])}" cy="{_n(p["cy"])}" r="{_n(p["rx"])}"{attrs}/>'
        if sh.kind == "ellipse":
            return (f'<ellipse cx="{_n(p["cx"])}" cy="{_n(p["cy"])}" rx="{_n(p["rx"])}" ry="{_n(p["ry"])}"'
                    f'{attrs}/>')
        if sh.kind == "line":
            return (f'<line x1="{_n(p["x1"])}" y1="{_n(p["y1"])}" x2="{_n(p["x2"])}" y2="{_n(p["y2"])}"'
                    f'{attrs}/>')
        if sh.kind in ("polyline", "polygon"):
            pts = " ".join(f"{_n(x)},{_n(y)}" for x, y in p["points"])
            return f'<{sh.kind} points="{pts}"{attrs}/>'
        path = sh.path
    return f'<path d="{path.to_d()}"{attrs}/>'


def _svg_shape(sh, ctx):
    st = _resolve(sh, ctx.dwg)
    idattr = f' id="{ctx.uid(sh.name)}"' if sh.name else ""
    op = st["opacity"]
    has_fill = _paint(st["fill"]) is not None and sh.kind not in ("line", "polyline", "arc")
    fa, fal = ctx.paint(st["fill"], "fill") if has_fill else (' fill="none"', 1.0)
    if has_fill:
        o = fal * st["fill_opacity"] * op
        if o < 1:
            fa += f' fill-opacity="{_n(o)}"'
        if st["fill_rule"] == "evenodd":
            fa += ' fill-rule="evenodd"'
    markers = st["marker_start"] or st["marker_end"]
    multi = None                                      # dash_fit on several subpaths: one <path> each
    if st["dash"] and st["dash_fit"]:
        spath = _markers_geom(sh.path, st)[0] if markers else sh.path
        subs = spath.flatten(0.05)
        fits = [_fit_dash(st["dash"], _poly_len(q, c), c, st["dash_fit"] == "center") for q, c in subs]
        if len(fits) == 1:
            st = dict(st, dash=fits[0], dash_offset=0.0)
        elif fits:
            multi = []
            for (start, segs, closed), ft in zip(spath.subs, fits):
                one = Path()
                one.subs = [[start, list(segs), closed]]
                multi.append(f'<path d="{one.to_d()}" fill="none"'
                             + _stroke_attrs(dict(st, dash=ft, dash_offset=0.0), ctx) + "/>")
    sa = _stroke_attrs(st, ctx)
    hat = st["hatch"] if sh.kind not in ("line", "polyline", "arc") else None
    if not markers and not hat and multi is None:
        return _geom_el(sh, idattr + fa + sa)
    parts = []
    if has_fill:
        parts.append(_geom_el(sh, fa + ' stroke="none"'))
    if hat:
        sw = ctx.dwg._default_sw
        sp = hat.spacing if hat.spacing else sw * 4.0
        lw = hat.width if hat.width else sw * 0.6
        col = hat.color or (st["stroke"] if st["stroke"] is not None else
                            (ctx.dwg.theme["ink"] if ctx.dwg.theme else "#1d2633"))
        c = _color(col) or (np.zeros(3), 1.0)
        pid = ctx.uid("hatch")
        rects = f'<rect x="0" y="{_n(sp / 2 - lw / 2)}" width="{_n(sp)}" height="{_n(lw)}"/>'
        if hat.cross:
            rects += f'<rect x="{_n(sp / 2 - lw / 2)}" y="0" width="{_n(lw)}" height="{_n(sp)}"/>'
        ctx.defs.append(f'<pattern id="{pid}" patternUnits="userSpaceOnUse" width="{_n(sp)}" height="{_n(sp)}" '
                        f'patternTransform="rotate({_n(hat.angle)})"><g fill="{_hex(c[0])}">{rects}</g></pattern>')
        o = c[1] * hat.opacity * op
        rule = ' fill-rule="evenodd"' if st["fill_rule"] == "evenodd" else ""
        parts.append(_geom_el(sh, f' fill="url(#{pid})"' + (f' fill-opacity="{_n(o)}"' if o < 1 else "")
                              + rule + ' stroke="none"'))
    if sa != ' stroke="none"':
        if markers:
            tp, marks = _markers_geom(sh.path, st)
            parts.append("".join(multi) if multi else _geom_el(sh, ' fill="none"' + sa, path=tp))
            pc = _paint(st["stroke"])
            col = _hex(pc[1]) if pc[0] == "solid" else "#000000"
            o = (pc[2] if pc[0] == "solid" else 1.0) * st["stroke_opacity"] * op
            oa = f' fill-opacity="{_n(o)}"' if o < 1 else ""
            soa = f' stroke-opacity="{_n(o)}"' if o < 1 else ""
            for mk in marks:
                pts = mk[1]
                d = "M" + " L".join(f"{_n(x)} {_n(y)}" for x, y in pts)
                if mk[0] == "fill":
                    parts.append(f'<path d="{d}Z" fill="{col}"{oa} stroke="none"/>')
                else:
                    capj = ' stroke-linecap="round" stroke-linejoin="round"' if len(pts) > 2 else ""
                    parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{_n(st["stroke_width"])}"'
                                 f'{capj}{soa}/>')
        else:
            parts.append("".join(multi) if multi else _geom_el(sh, ' fill="none"' + sa))
    return f"<g{idattr}>" + "".join(parts) + "</g>"


def _svg_text(t, ctx):
    k = max(ctx.dwg._ppu, 1.0)
    f, lines, bb = t._layout(k)
    th = ctx.dwg.theme
    fam = (t.family or "sans")
    stack = _STACKS.get(fam.lower(), f"'{fam}', " + _STACKS["sans"])
    if fam.lower() == "sans" and _text.has_cjk(t.text):
        stack = _STACKS["sans"]
    ha = t.anchor[0]
    al = t.align or {"l": "left", "m": "center", "r": "right"}[ha]
    ta = {"left": "start", "center": "middle", "right": "end"}[al]
    attrs = (f' font-family="{_esc(stack)}" font-size="{_n(t.size)}"'
             + (' font-weight="bold"' if _is_bold(t.weight) else "")
             + (' font-style="italic"' if str(t.style).lower() in ("italic", "oblique") else "")
             + (f' letter-spacing="{_n(t.letter_spacing)}"' if t.letter_spacing else "")
             + f' text-anchor="{ta}"')
    if t.rotate:
        attrs += f' transform="rotate({_n(t.rotate)} {_n(t.x)} {_n(t.y)})"'
    if t.opacity < 1:
        attrs += f' opacity="{_n(t.opacity)}"'
    spans = []
    for ln, lx, by, lw in lines:
        if t.markup:
            ln = "".join(sp for sp, _ in _text.parse_markup(ln))
        x = {"start": lx, "middle": lx + lw / 2, "end": lx + lw}[ta]
        spans.append(f'<tspan x="{_n(x)}" y="{_n(by)}">{_esc(ln)}</tspan>')
    body = "".join(spans)
    fill = t.fill if t.fill is not None else (th["ink"] if th else "#1d2633")
    fa, _ = ctx.paint(fill, "fill")
    pc = _paint(fill)
    if pc and pc[0] == "solid" and pc[2] < 1:
        fa += f' fill-opacity="{_n(pc[2])}"'
    idattr = f' id="{ctx.uid(t.name)}"' if t.name else ""
    out = ""
    if t.halo and _paint(t.halo):
        hp = _paint(t.halo)
        ha_ = f' fill="none" stroke="{_hex(hp[1])}" stroke-width="{_n(2 * t.halo_width)}" stroke-linejoin="round"'
        if hp[2] < 1:
            ha_ += f' stroke-opacity="{_n(hp[2])}"'
        out += f'<text{attrs}{ha_} xml:space="preserve">{body}</text>'
    out += f'<text{idattr}{attrs}{fa} xml:space="preserve">{body}</text>'
    return f"<g>{out}</g>" if t.halo else out


def _svg_group(g, ctx):
    attrs = f' id="{ctx.uid(g.name)}"' if g.name else ""
    tr = []
    if g.tx or g.ty:
        tr.append(f"translate({_n(g.tx)} {_n(g.ty)})")
    if g.rot:
        tr.append(f"rotate({_n(g.rot)} {_n(g.origin[0])} {_n(g.origin[1])})" if any(g.origin)
                  else f"rotate({_n(g.rot)})")
    if g.sx != 1 or g.sy != 1:
        tr.append(f"scale({_n(g.sx)}" + (f" {_n(g.sy)})" if g.sy != g.sx else ")"))
    if tr:
        attrs += f' transform="{" ".join(tr)}"'
    if g.opacity < 1:
        attrs += f' opacity="{_n(g.opacity)}"'
    if g.clip is not None:
        cid = ctx.uid("clip")
        path, eo = _clip_path(g.clip)
        rule = ' clip-rule="evenodd"' if eo else ""
        ctx.defs.append(f'<clipPath id="{cid}" clipPathUnits="userSpaceOnUse"><path d="{path.to_d()}"{rule}/>'
                        f'</clipPath>')
        attrs += f' clip-path="url(#{cid})"'
    sh = g._shadow()
    inner = "".join(_svg_item(c, ctx) for c in g.children)
    if sh:
        fid = ctx.uid("shadow")
        c = _color(sh["color"]) or (np.zeros(3), 1.0)
        bs = [c._bbox(_I, True) for c in g.children]
        bb = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)) \
            if bs else (0, 0, 0, 0)
        pad = sh["blur"] * 3 + max(abs(sh["dx"]), abs(sh["dy"]))
        ctx.defs.append(
            f'<filter id="{fid}" filterUnits="userSpaceOnUse" x="{_n(bb[0] - pad)}" y="{_n(bb[1] - pad)}" '
            f'width="{_n(bb[2] - bb[0] + 2 * pad)}" height="{_n(bb[3] - bb[1] + 2 * pad)}" '
            f'color-interpolation-filters="sRGB">'
            f'<feGaussianBlur in="SourceAlpha" stdDeviation="{_n(sh["blur"])}"/>'
            f'<feOffset dx="{_n(sh["dx"])}" dy="{_n(sh["dy"])}" result="o"/>'
            f'<feFlood flood-color="{_hex(c[0])}" flood-opacity="{_n(float(sh["opacity"]) * c[1])}"/>'
            f'<feComposite in2="o" operator="in"/>'
            f'<feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
        if g.clip is not None:                 # filter first, then clip: keep SVG order explicit
            return f"<g{attrs}><g filter=\"url(#{fid})\">{inner}</g></g>"
        attrs += f' filter="url(#{fid})"'
    return f"<g{attrs}>{inner}</g>"


def _svg_item(it, ctx):
    if isinstance(it, Group):
        return _svg_group(it, ctx)
    if isinstance(it, Text):
        return _svg_text(it, ctx)
    return _svg_shape(it, ctx)


# --------------------------------------------------------------------------------------------- Drawing

_REGISTRY = []


def last_drawings():
    """Drawings drawn (``draw`` / ``render``) since the last ``clear_drawings()``, oldest first."""
    return list(_REGISTRY)


def clear_drawings():
    """Forget the drawings recorded by ``last_drawings()`` (call before rendering a scene)."""
    _REGISTRY.clear()


class Drawing(_Builder):
    """A retained-mode vector drawing.

    ``Drawing(width, height, units="px", px_per_unit=None, background=None, theme=None, name="drawing")``

    * ``units``: px / mm / cm / in / pt. ``px_per_unit`` sets the raster density of ``render()`` (default: CSS
      96 dpi, i.e. 1 for px, 3.78 for mm).
    * ``theme``: a THEMES name or Theme → default ink, stroke width, font and background.
    * ``background``: None → the theme background (none without a theme); a colour; ``False`` (or ``"none"``)
      → transparent, for drawings painted over an existing canvas or embedded in another drawing.
    * ``crisp``: pixel-align axis-aligned strokes and plain rectangles (sharp 1 px lines). Default True.
    """

    def __init__(self, width, height, units="px", px_per_unit=None, background=None, theme=None, name="drawing",
                 crisp=True):
        if units not in _UNITS:
            raise ValueError(f"units must be one of {list(_UNITS)}")
        self.width, self.height, self.units = float(width), float(height), units
        self.px_per_unit = float(px_per_unit) if px_per_unit else _UNITS[units]
        self.theme = (_find_theme(theme) if isinstance(theme, str) else theme) if theme is not None else None
        self.background = background if background is not None else (self.theme["background"] if self.theme
                                                                     else None)
        self.name, self.crisp = name, crisp
        self.root = Group(self, None)
        self._stack = [self.root]
        self._default_sw = (self.theme["stroke_width"] if self.theme else 1.5) / _UNITS[units]

    @property
    def _ppu(self):
        return self.px_per_unit

    def _target(self):
        return self._stack[-1]

    def _dwg(self):
        return self

    @property
    def items(self):
        """Top-level items, bottom to top."""
        return list(self.root.children)

    def find(self, name):
        """First item with this name (searches groups)."""
        return self.root.find(name)

    def add(self, item):
        """Add an item created with ``add=False`` (or removed) to the current group."""
        return self._add(item)

    def measure_text(self, text, size=None, family=None, weight=None, style="normal", letter_spacing=0.0,
                     line_spacing=1.2, max_width=None, markup=False):
        """(width, height) of ``text`` in drawing units."""
        k = max(self._ppu, 1.0)
        size = size if size is not None else self._default_sw * 9.5
        fam = family or (self.theme["font"]["family"] if self.theme else "sans")
        wt = weight or (self.theme["font"]["weight"] if self.theme else "normal")
        w, h = measure_text(text, size * k, fam, wt, style, letter_spacing * k, line_spacing,
                            None if max_width is None else max_width * k, markup)
        return (w / k, h / k)

    # ---- raster
    def draw(self, cv, log=None, box=None):
        """Paint onto Canvas ``cv``, scaled to fit ``box`` (x0, y0, x1, y1 px; default the whole canvas).
        Each top-level group is one step (``log(name)``); loose items are painted under ``log(self.name)``."""
        if self not in _REGISTRY:
            _REGISTRY.append(self)
            del _REGISTRY[:-64]
        _manifest.drawing_drawn(cv, self, box)
        W, H = cv.w, cv.h
        bx0, by0, bx1, by1 = box if box is not None else (0, 0, W, H)
        k = min((bx1 - bx0) / self.width, (by1 - by0) / self.height)
        ox = bx0 + ((bx1 - bx0) - self.width * k) / 2
        oy = by0 + ((by1 - by0) - self.height * k) / 2
        M0 = (k, 0.0, 0.0, k, ox, oy)
        bounds = (max(0, int(math.floor(ox + 1e-6))), max(0, int(math.floor(oy + 1e-6))),
                  min(W, int(math.ceil(ox + self.width * k - 1e-6))), min(H, int(math.ceil(oy + self.height * k - 1e-6))))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return cv
        ctx = _RCtx(self, bounds, W, H)
        bg = _color(self.background)
        if bg is not None:
            if log:
                log("background")
            x0, y0, x1, y1 = bounds
            if (x0, y0, x1, y1) == (0, 0, W, H) and bg[1] >= 1:
                cv.paint(bg[0])
            else:
                cv.paste(np.broadcast_to(bg[0], (y1 - y0, x1 - x0, 3)), x0, y0, opacity=bg[1])
        run, n = [], [0]

        def paste(tile, name):
            if tile is None:
                return
            x0, y0, P, A = tile
            if log:
                log(name)
            col = P / np.maximum(A, 1e-6)[..., None]
            _manifest.tile(cv, name, x0, y0, A)
            cv.paste(col, int(x0), int(y0), mask=np.clip(A, 0, 1))

        def flush():
            if run:
                paste(ctx.render_items(run, M0), self.name)
                run.clear()

        for ch in self.root.children:
            if isinstance(ch, Group):
                flush()
                n[0] += 1
                paste(ctx.render_group(ch, _mul(M0, ch.matrix)), ch.name or f"group {n[0]}")
            else:
                run.append(ch)
        flush()
        return cv

    def render(self, scale=1.0, log=None):
        """New Canvas of ``width * px_per_unit * scale`` px with the drawing painted on it."""
        from .canvas import Canvas
        W = max(1, int(round(self.width * self.px_per_unit * scale)))
        H = max(1, int(round(self.height * self.px_per_unit * scale)))
        bg = _color(self.background)
        cv = Canvas(W, H, bg[0] if bg else (1.0, 1.0, 1.0))
        return self.draw(cv, log)

    # ---- SVG
    def to_svg(self):
        """Standalone SVG 1.1 document (string)."""
        ctx = _SCtx(self)
        body = []
        bg = _color(self.background)
        if bg is not None:
            body.append(f'<rect width="{_n(self.width)}" height="{_n(self.height)}" fill="{_hex(bg[0])}"'
                        + (f' fill-opacity="{_n(bg[1])}"' if bg[1] < 1 else "") + "/>")
        for it in self.root.children:
            body.append(_svg_item(it, ctx))
        u = "" if self.units == "px" else self.units
        defs = ("<defs>" + "".join(ctx.defs) + "</defs>\n") if ctx.defs else ""
        return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{_n(self.width)}{u}" '
                f'height="{_n(self.height)}{u}" viewBox="0 0 {_n(self.width)} {_n(self.height)}">\n'
                + defs + "\n".join(body) + "\n</svg>\n")

    def save_svg(self, path):
        """Write ``to_svg()`` to ``path``; returns the path."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_svg())
        return path


# --------------------------------------------------------------------------------------------- icons
# 24-unit grid, 2-unit round strokes. Element: (kind, args, flags) where flags ⊂ {"f": fillable body, "s": solid}.

def _c(cx, cy, r):
    return f"M{cx - r} {cy}a{r} {r} 0 1 0 {2 * r} 0a{r} {r} 0 1 0 {-2 * r} 0Z"


def _gear_d(cx=12, cy=12, r_out=9.6, r_in=7.4, teeth=8):
    pts = []
    step = 2 * math.pi / teeth
    for i in range(teeth):
        a = i * step - math.pi / 2
        for da, r in ((-0.30, r_in), (-0.17, r_out), (0.17, r_out), (0.30, r_in)):
            t = a + da * step * 1.15
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return "M" + " L".join(f"{x:.3f} {y:.3f}" for x, y in pts) + "Z"


def _star_d(cx=12, cy=12.6, ro=9.8, ri=4.3, n=5):
    pts = []
    for i in range(2 * n):
        r = ro if i % 2 == 0 else ri
        t = -math.pi / 2 + i * math.pi / n
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return "M" + " L".join(f"{x:.3f} {y:.3f}" for x, y in pts) + "Z"


def _R(x, y, w, h, r=0.0, f=""):
    return ("rect", (x, y, w, h, r), f)


def _P(d, f=""):
    return ("path", d, f)


def _C(cx, cy, r, f=""):
    return ("circle", (cx, cy, r), f)


def _L(x1, y1, x2, y2):
    return ("path", f"M{x1} {y1}L{x2} {y2}", "")


def _S(cx, cy, r=1.1):
    return ("circle", (cx, cy, r), "s")


ICONS = {
    "cloud": [_P("M7 19h10a5 5 0 0 0 .6-9.97A6.5 6.5 0 0 0 5.2 10.8 4.2 4.2 0 0 0 7 19Z", "f")],
    "server": [_R(3, 3.5, 18, 7.5, 2, "f"), _R(3, 13, 18, 7.5, 2, "f"), _S(7, 7.25), _S(7, 16.75),
               _L(11, 7.25, 17, 7.25), _L(11, 16.75, 17, 16.75)],
    "database": [_P("M4 5.5v13c0 1.66 3.58 3 8 3s8-1.34 8-3v-13", "f"), _P("M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3"),
                 ("ellipse", (12, 5.5, 8, 3), "f")],
    "user": [_C(12, 8, 4.25, "f"), _P("M4 21v-.5A6.5 6.5 0 0 1 10.5 14h3a6.5 6.5 0 0 1 6.5 6.5v.5", "f")],
    "users": [_C(9, 8, 3.75, "f"), _P("M2.5 20.5v-.5a5.5 5.5 0 0 1 5.5-5.5h2a5.5 5.5 0 0 1 5.5 5.5v.5", "f"),
              _P("M15.5 4.4a3.75 3.75 0 0 1 0 7.2"), _P("M18 14.6a5.5 5.5 0 0 1 3.5 5.1v.8")],
    "lock": [_R(4.5, 10.5, 15, 11, 2.5, "f"), _P("M8 10.5V7.5a4 4 0 0 1 8 0v3"), _S(12, 15.4, 1.5),
             _L(12, 16, 12, 18)],
    "key": [_C(7.5, 15.5, 4.5, "f"), _L(10.7, 12.3, 20.5, 2.5), _L(16.5, 6.5, 19.5, 9.5), _L(14, 9, 16, 11)],
    "shield": [_P("M12 2.5 19.5 5.5v6c0 4.8-3.2 8.3-7.5 10-4.3-1.7-7.5-5.2-7.5-10v-6Z", "f"),
               _P("M8.8 12.2l2.2 2.2 4.2-4.4")],
    "gear": [_P(_gear_d(), "f"), _C(12, 12, 3)],
    "globe": [_C(12, 12, 9.5, "f"), ("ellipse", (12, 12, 4, 9.5), ""), _L(2.5, 12, 21.5, 12)],
    "file": [_P("M14 2.5H6.5a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V8Z", "f"), _P("M14 2.5V8h5.5"),
             _L(8.5, 13, 15.5, 13), _L(8.5, 17, 13, 17)],
    "folder": [_P("M3 6.5a2 2 0 0 1 2-2h4.2l2 2.5H19a2 2 0 0 1 2 2v9.5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z", "f")],
    "phone": [_R(6, 2, 12, 20, 2.5, "f"), _L(10.5, 18.5, 13.5, 18.5)],
    "monitor": [_R(2.5, 3.5, 19, 13, 2, "f"), _L(8, 21, 16, 21), _L(12, 16.5, 12, 21)],
    "chip": [_R(6, 6, 12, 12, 2, "f"), _R(9.5, 9.5, 5, 5, 0.75), _L(9.5, 2.5, 9.5, 6), _L(14.5, 2.5, 14.5, 6),
             _L(9.5, 18, 9.5, 21.5), _L(14.5, 18, 14.5, 21.5), _L(2.5, 9.5, 6, 9.5), _L(2.5, 14.5, 6, 14.5),
             _L(18, 9.5, 21.5, 9.5), _L(18, 14.5, 21.5, 14.5)],
    "bolt": [_P("M13.5 2 4.5 13.5h7l-1 8.5 9-11.5h-7Z", "f")],
    "heart": [_P("M12 20.5S3 15.3 3 8.9A4.9 4.9 0 0 1 12 6.2a4.9 4.9 0 0 1 9 2.7c0 6.4-9 11.6-9 11.6Z", "f")],
    "star": [_P(_star_d(), "f")],
    "cart": [_P("M2.5 3h2.6l2.4 11.4a1.6 1.6 0 0 0 1.6 1.3h8.6a1.6 1.6 0 0 0 1.55-1.2L21 7H6.1"),
             _C(9.5, 20, 1.5), _C(17.5, 20, 1.5)],
    "camera": [_P("M3 8.5a2 2 0 0 1 2-2h2.5L9.3 4h5.4l1.8 2.5H19a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z",
                  "f"), _C(12, 13, 3.75)],
    "mail": [_R(2.5, 5, 19, 14, 2.5, "f"), _P("M3.5 7.5 12 13.5l8.5-6")],
    "home": [_P("M3.5 10.5 12 3.5l8.5 7V19a2 2 0 0 1-2 2H15v-6.5H9V21H5.5a2 2 0 0 1-2-2Z", "f")],
    "search": [_C(10.5, 10.5, 6.75, "f"), _L(15.5, 15.5, 21, 21)],
    "bell": [_P("M6 9.5a6 6 0 0 1 12 0c0 5.6 2.5 7.5 2.5 7.5h-17S6 15.1 6 9.5Z", "f"),
             _P("M10.2 20.2a2.1 2.1 0 0 0 3.6 0")],
    "calendar": [_R(3, 4.5, 18, 17, 2.5, "f"), _L(3, 10, 21, 10), _L(8, 2.5, 8, 6.5), _L(16, 2.5, 16, 6.5),
                 _S(8, 14, 1.05), _S(12, 14, 1.05), _S(16, 14, 1.05), _S(8, 17.75, 1.05), _S(12, 17.75, 1.05)],
    "chart": [_R(3, 3, 18, 18, 2.5, "f"), _L(8, 16.5, 8, 12.5), _L(12, 16.5, 12, 7.5), _L(16, 16.5, 16, 10.5)],
    "trend": [_P("M3 17.5l6-6 4 4 8-8"), _P("M15 7.5h6v6")],
    "code": [_P("M8 7 3 12l5 5"), _P("M16 7l5 5-5 5"), _L(14, 4, 10, 20)],
    "terminal": [_R(2.5, 4, 19, 16, 2.5, "f"), _P("M7 9.5l3 2.5-3 2.5"), _L(12.5, 15, 17, 15)],
    "clock": [_C(12, 12, 9.5, "f"), _P("M12 6.5V12l3.5 2")],
    "wifi": [_P("M2.5 8.8a14 14 0 0 1 19 0"), _P("M5.6 12.4a9.4 9.4 0 0 1 12.8 0"), _P("M8.8 15.9a4.8 4.8 0 0 1 6.4 0"),
             _S(12, 19.5, 1.25)],
    "link": [_P("M10 13.5a4.5 4.5 0 0 0 6.4.4l3-3a4.5 4.5 0 0 0-6.4-6.4l-1.6 1.6"),
             _P("M14 10.5a4.5 4.5 0 0 0-6.4-.4l-3 3a4.5 4.5 0 0 0 6.4 6.4l1.6-1.6")],
    "check": [_C(12, 12, 9.5, "f"), _P("M7.8 12.3l2.9 2.9 5.5-5.6")],
    "alert": [_P("M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z", "f"),
              _L(12, 9, 12, 13.5), _S(12, 17, 1.2)],
    "info": [_C(12, 12, 9.5, "f"), _L(12, 11, 12, 16.5), _S(12, 7.75, 1.2)],
    "image": [_R(3, 3.5, 18, 17, 2.5, "f"), _C(9, 9.5, 2), _P("M21 15.5l-4.8-4.8L6 20.5")],
    "pin": [_P("M12 21.5s7-6.2 7-12a7 7 0 0 0-14 0c0 5.8 7 12 7 12Z", "f"), _C(12, 9.5, 2.5)],
    "bulb": [_P("M9 17.5v-1.7c0-1.3-.6-2.3-1.6-3.3A6.5 6.5 0 1 1 18.5 9c0 1.4-.6 2.5-1.9 3.5-1 .9-1.6 2-1.6 3.3v1.7Z",
                "f"), _L(9.5, 21, 14.5, 21)],
    "download": [_P("M3.5 15.5v3a2 2 0 0 0 2 2h13a2 2 0 0 0 2-2v-3"), _L(12, 3.5, 12, 15),
                 _P("M7 10.5l5 5 5-5")],
    "upload": [_P("M3.5 15.5v3a2 2 0 0 0 2 2h13a2 2 0 0 0 2-2v-3"), _L(12, 15, 12, 3.5), _P("M7 8.5l5-5 5 5")],
    "cube": [_P("M12 2.5l8.5 4.75v9.5L12 21.5l-8.5-4.75v-9.5Z", "f"), _P("M3.8 7.4 12 12l8.2-4.6"),
             _L(12, 12, 12, 21.3)],
    "bot": [_R(4, 8, 16, 12, 3, "f"), _L(12, 4.5, 12, 8), _S(12, 3.6, 1.3), _S(9, 13.5, 1.4), _S(15, 13.5, 1.4),
            _L(2, 13, 2, 15.5), _L(22, 13, 22, 15.5)],
    "play": [_C(12, 12, 9.5, "f"), _P("M10 8.5v7l6-3.5Z")],
    "layers": [_P("M12 2.5 21.5 7.5 12 12.5 2.5 7.5Z", "f"), _P("M2.5 12l9.5 5 9.5-5"), _P("M2.5 16.5l9.5 5 9.5-5")],
    "network": [_C(12, 5, 2.5, "f"), _C(5, 19, 2.5, "f"), _C(19, 19, 2.5, "f"), _L(10.8, 7.3, 6.2, 16.7),
                _L(13.2, 7.3, 17.8, 16.7), _L(7.5, 19, 16.5, 19)],
    "sparkles": [_P("M10 3c.7 4.1 2.4 5.8 6.5 6.5-4.1.7-5.8 2.4-6.5 6.5-.7-4.1-2.4-5.8-6.5-6.5 4.1-.7 5.8-2.4 6.5-6.5Z",
                    "f"),
                 _P("M18 14.5c.3 1.6 1 2.2 2.5 2.5-1.5.3-2.2.9-2.5 2.5-.3-1.6-1-2.2-2.5-2.5 1.5-.3 2.2-.9 2.5-2.5Z")],
    "refresh": [_P("M20.5 12a8.5 8.5 0 1 1-2.6-6.1l2.6 2.3"), _P("M20.5 3.5v4.7h-4.7")],
    "tag": [_P("M3 3.5h8l9.4 9.4a2 2 0 0 1 0 2.8l-4.7 4.7a2 2 0 0 1-2.8 0L3 11Z", "f"), _S(7.6, 8.1, 1.45)],
}


def icon_names():
    """Sorted list of built-in icon names."""
    return sorted(ICONS)


def _build_icon(g, name, color, sw, fill):
    dwg = g.drawing
    col = color if color is not None else (dwg.theme["ink"] if dwg.theme else "#1d2633")
    base = dict(stroke=col, stroke_width=sw, cap="round", join="round")
    for kind, args, flags in ICONS[name]:
        if "s" in flags:
            style = dict(fill=col, stroke=None)
        else:
            style = dict(base, fill=fill if ("f" in flags and fill is not None) else None)
        if kind == "path":
            g.path(args, **style)
        elif kind == "rect":
            x, y, w, h, r = args
            g.rect(x, y, w, h, r=r, **style)
        elif kind == "circle":
            g.circle(*args, **style)
        elif kind == "ellipse":
            g.ellipse(*args, **style)


def icon(name, size=24, color="#1d2633", stroke_width=2.0, fill=None, background=None):
    """A standalone ``size`` x ``size`` Drawing of one icon (``.render()``, ``.save_svg()``, or ``embed`` it)."""
    d = Drawing(size, size, background=background, name=name)
    d.icon(name, size / 2, size / 2, size, color=color, stroke_width=stroke_width, fill=fill)
    return d


from ._args import strict_methods as _strict_methods  # noqa: E402

for _c in (_Builder, Group, Drawing):                      # explicit signatures: helpful errors, colour / lw spellings
    _strict_methods(_c, aliases={"color": "fill", "colour": "fill"})

# ---- scene manifest: a drawing painted on a canvas records one entry per top-level group ----
_manifest.wrap_method(Drawing, "draw", "shape", after=_manifest.after_drawing)


# --------------------------------------------------------------------------------------------- app icon tiles
# Rounded-square app icons: a glyph (pictogram or line icon), a two-colour gradient, a long / soft shadow, a label.

APP_GLYPHS = {"notes": "pencil", "note": "pencil", "calendar": "calendar", "mail": "envelope", "email": "envelope",
              "inbox": "envelope", "tasks": "clipboard", "todo": "clipboard", "to-do": "clipboard",
              "reminders": "clipboard", "files": "folder", "documents": "folder", "chat": "speech bubble",
              "messages": "speech bubble", "camera": "camera", "photos": "camera", "music": "music note",
              "maps": "map", "map": "map", "weather": "weather", "settings": "gear", "calculator": "calculator",
              "clock": "clock", "home": "home", "search": "search", "cloud": "cloud", "security": "shield",
              "store": "cart", "shop": "cart", "contacts": "user", "people": "users", "stats": "chart",
              "analytics": "chart", "terminal": "terminal", "code": "code", "news": "file", "health": "heart",
              "favorites": "star", "alarm": "bell", "notifications": "bell", "browser": "globe", "web": "globe"}
APP_COLORS = [("#ffe27a", "#f0a30a"), ("#ff8a80", "#e0393e"), ("#6ec3ff", "#1e6fe0"), ("#7be0b0", "#18a06a"),
              ("#8fb2ff", "#3f58d4"), ("#79e08a", "#1fa24a"), ("#b4b9c6", "#555d70"), ("#ff8fc0", "#d4286e"),
              ("#6fe0d6", "#119a9c"), ("#7fd0ff", "#2b86d8"), ("#c0a8ff", "#6a43d6"), ("#ffb07a", "#e5601c")]
APP_SHADOWS = ("long", "soft", "none")


def _glyph_names():
    try:
        from . import pictograms as _pg
        pics = list(_pg.names())
    except Exception:                                    # noqa: BLE001
        pics = []
    return pics, sorted(set(pics) | set(ICONS) | set(APP_GLYPHS))


def _resolve_glyph(glyph):
    g = str(glyph).strip()
    pics, allnames = _glyph_names()
    low = g.lower()
    low = APP_GLYPHS.get(low, low)
    if low == "weather" or low in pics or low in ICONS:
        return low
    raise _unknown("app icon glyph", glyph, allnames, "Glyphs: any pictogram (pictograms.names()), any vector "
                   "icon (icon_names()) or an app word (notes, tasks, chat, weather ...): see vector.APP_GLYPHS")


def _icon_alpha(name, h, w, box, stroke_width=2.0):
    """Anti-aliased alpha (h, w) of vector icon ``name`` drawn in ``box`` (x0, y0, x1, y1 px)."""
    d = Drawing(w, h, background=False)
    s = min(box[2] - box[0], box[3] - box[1])
    g = d.icon(name, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, s, color="#ffffff", stroke_width=stroke_width)
    tile = _RCtx(d, (0, 0, w, h), w, h).render_group(g, g.matrix)
    out = np.zeros((h, w), F32)
    if tile is not None:
        x0, y0, _, A = tile
        out[y0:y0 + A.shape[0], x0:x0 + A.shape[1]] = np.clip(A, 0, 1)
    return out


def _glyph_alpha(glyph, h, w, box):
    if glyph == "weather":
        x0, y0, x1, y1 = box
        s = x1 - x0
        sun = _glyph_alpha("sun", h, w, (x0 + 0.02 * s, y0 - 0.06 * s, x0 + 0.62 * s, y0 + 0.54 * s))
        cloud = _glyph_alpha("cloud", h, w, (x0 + 0.12 * s, y0 + 0.14 * s, x1 + 0.04 * s, y1 + 0.1 * s))
        halo = np.clip(_shift_max(cloud, 0.05 * s), 0, 1)
        return np.maximum(sun * (1 - halo), cloud)
    from . import pictograms as _pg
    if glyph in _pg.names():
        return np.asarray(_pg.mask(glyph, h, w, box=box, role="silhouette"), F32)
    return _icon_alpha(glyph, h, w, box, stroke_width=2.3)


def _glyph_layers(glyph, h, w, box):
    """(main, accent) alpha masks of a glyph: ``accent`` holds the part drawn in the second colour (the sun behind
    a cloud, a calendar's header band, the inner details of a pictogram) and is already cut out of ``main``."""
    from . import pictograms as _pg
    if glyph == "weather":
        x0, y0, x1, y1 = box
        s = x1 - x0
        sun = _glyph_alpha("sun", h, w, (x0 + 0.02 * s, y0 - 0.06 * s, x0 + 0.62 * s, y0 + 0.54 * s))
        cloud = _glyph_alpha("cloud", h, w, (x0 + 0.12 * s, y0 + 0.14 * s, x1 + 0.04 * s, y1 + 0.1 * s))
        halo = np.clip(_shift_max(cloud, 0.05 * s), 0, 1)
        return cloud, sun * (1 - halo)
    if glyph in _pg.names():
        sil = np.asarray(_pg.mask(glyph, h, w, box=box, role="silhouette"), F32)
        acc = np.asarray(_pg.mask(glyph, h, w, box=box, role="accent"), F32) * sil
        if glyph == "calendar":                          # header band across the top of the page
            body = np.asarray(_pg.mask(glyph, h, w, box=box, role="body"), F32)
            ys = np.arange(h, dtype=F32)[:, None] + 0.5
            y0, y1 = box[1], box[3]
            band = np.clip((ys - (y0 + 0.16 * (y1 - y0))) / 1.0, 0, 1) * np.clip(((y0 + 0.40 * (y1 - y0)) - ys) / 1.0, 0, 1)
            acc = np.maximum(acc, body * band)
        else:                                            # inner details take the second colour instead of cut-outs
            det = np.asarray(_pg.mask(glyph, h, w, box=box, role="detail"), F32)
            body = np.asarray(_pg.mask(glyph, h, w, box=box, role="body"), F32)
            acc = np.maximum(acc, det * body)
        main = sil * (1 - acc)
        return main, np.clip(acc, 0, 1)
    return _icon_alpha(glyph, h, w, box, stroke_width=2.3), None


def _shift(a, dx, dy):
    """``a`` shifted by whole pixels (dx, dy) with zero fill (no wrap-around)."""
    out = np.zeros_like(a)
    h, w = a.shape
    dx, dy = int(round(dx)), int(round(dy))
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def _shift_max(a, r):
    """Grow a mask by about ``r`` px (max of 8 shifted copies)."""
    out = a.copy()
    for k in range(8):
        t = k * math.pi / 4
        out = np.maximum(out, _shift(a, r * math.cos(t), r * math.sin(t)))
    return out


def _sweep(a, length, fade=1.4):
    """Long shadow of mask ``a``: copies swept down-right at 45 degrees over ``length`` px, fading with distance."""
    out = np.zeros_like(a)
    n = max(int(length / 1.0), 1)
    for i in range(1, n + 1):
        d = i * length / n
        np.maximum(out, _shift(a, d, d) * (1.0 - i / (n + 1)) ** fade, out=out)
    return out


def app_icon(cv, box, glyph, colors=None, shadow="long", label=None, radius=0.23, glyph_color="#ffffff",
             glyph_size=0.56, label_color="#2b3140", cell=None, name=None, accent_color=None):
    """One app icon tile: rounded square with a two-colour diagonal gradient, a glyph (white by default) and a shadow.

    ``box``: the tile (x0, y0, x1, y1) in pixels or canvas fractions (kept square, centred). ``glyph``: a
    pictogram name, a vector icon name or an app word (notes, tasks, chat, weather, settings ... see
    ``APP_GLYPHS``). ``colors``: (light, deep) or one colour. ``shadow``: long (glyph shadow across the tile + tile
    shadow, both kept inside ``cell``) / soft / none. ``label``: text under the tile. ``cell``: the area the shadow
    may use (default: the tile grown by 30%). ``glyph_color``: colour of the glyph. ``accent_color``: a second glyph
    colour for its details (the calendar header, the sun behind a cloud, the hands of a clock, a camera lens);
    None keeps the whole glyph in ``glyph_color``. Returns the tile box in pixels."""
    _check_choice("app_icon", "shadow", shadow, APP_SHADOWS)
    g = _resolve_glyph(glyph)
    if colors is None:
        colors = APP_COLORS[sum(map(ord, str(glyph))) % len(APP_COLORS)]
    if isinstance(colors, str):
        c1 = _rgb(colors)
        c0 = np.clip(c1 + (1 - c1) * 0.45, 0, 1)
    else:
        c0, c1 = _rgb(colors[0]), _rgb(colors[1])
    W, H = cv.w, cv.h
    bx = [float(v) for v in box]
    if max(abs(v) for v in bx) <= 1.0:
        bx = [bx[0] * W, bx[1] * H, bx[2] * W, bx[3] * H]
    T = min(bx[2] - bx[0], bx[3] - bx[1])
    tx0, ty0 = (bx[0] + bx[2] - T) / 2, (bx[1] + bx[3] - T) / 2
    if cell is None:
        cell = (tx0 - 0.3 * T, ty0 - 0.3 * T, tx0 + 1.3 * T, ty0 + 1.3 * T + (0.3 * T if label else 0))
    cx0, cy0 = max(int(math.floor(cell[0])), 0), max(int(math.floor(cell[1])), 0)
    cx1, cy1 = min(int(math.ceil(cell[2])), W), min(int(math.ceil(cell[3])), H)
    h, w = cy1 - cy0, cx1 - cx0
    if h <= 0 or w <= 0:
        return (tx0, ty0, tx0 + T, ty0 + T)
    from . import sdf as _sdf
    Y, X = np.mgrid[0:h, 0:w].astype(F32)
    X += 0.5 + cx0
    Y += 0.5 + cy0
    r = float(radius) * T if radius <= 0.5 else float(radius)
    tile = _sdf.fill(_sdf.sd_box(X, Y, tx0 + T / 2, ty0 + T / 2, T / 2, T / 2, radius=r)).astype(F32)
    # cell edge fade: shadows never reach the neighbours or the label of the next row
    ex = np.minimum(np.minimum(X - cx0, cx1 - X), np.minimum(Y - cy0, cy1 - Y))
    edge = np.clip(ex / max(0.08 * T, 1.0), 0, 1).astype(F32)
    ink = np.array([0.08, 0.1, 0.16], F32)
    if shadow == "long":
        sh = _sweep(tile, 0.42 * T, fade=1.6) * (1 - tile) * edge
        cv.paste(np.broadcast_to(ink, (h, w, 3)), cx0, cy0, mask=sh * 0.16)
    if shadow in ("long", "soft"):
        from .filters import gaussian_blur
        drop = gaussian_blur(_shift(tile, 0, 0.05 * T), max(0.06 * T, 0.5)) * (1 - tile) * edge
        cv.paste(np.broadcast_to(ink, (h, w, 3)), cx0, cy0, mask=drop * (0.22 if shadow == "soft" else 0.14))
    t = np.clip(((X - tx0) + (Y - ty0)) / (2 * T), 0, 1)[..., None]
    img = c0 * (1 - t) + c1 * t
    sheen = np.clip(1 - np.hypot(X - (tx0 + 0.25 * T), Y - (ty0 + 0.1 * T)) / (0.85 * T), 0, 1) ** 1.5
    img = img + (1 - img) * (sheen * 0.22)[..., None]
    gs = float(glyph_size) * T
    gbox = (tx0 + (T - gs) / 2 - cx0, ty0 + (T - gs) / 2 - cy0, tx0 + (T + gs) / 2 - cx0, ty0 + (T + gs) / 2 - cy0)
    if accent_color is not None:
        gm_main, gm_acc = _glyph_layers(g, h, w, gbox)
        gm = gm_main if gm_acc is None else np.maximum(gm_main, gm_acc)
    else:
        gm, gm_acc = _glyph_alpha(g, h, w, gbox), None
    if shadow == "long":
        gsh = _sweep(gm, 0.75 * T, fade=1.2) * (1 - gm) * tile
        deep = c1 * 0.62
        img = img * (1 - 0.42 * gsh[..., None]) + deep * (0.42 * gsh[..., None])
    rim = np.clip(tile - _sdf.fill(_sdf.sd_box(X, Y, tx0 + T / 2, ty0 + T / 2 + 0.012 * T, T / 2, T / 2,
                                               radius=r)), 0, 1)
    img = img + (1 - img) * (rim * 0.35)[..., None]
    gc = _rgb(glyph_color)
    img = img * (1 - gm[..., None]) + gc * gm[..., None]
    if gm_acc is not None:
        img = img * (1 - gm_acc[..., None]) + _rgb(accent_color) * gm_acc[..., None]
    cv.paste(img.astype(F32), cx0, cy0, mask=tile)
    if label:
        cv.text(str(label), (tx0 + T / 2, ty0 + T * 1.1), max(T * 0.13, 6.0), label_color, anchor="mt", bold=True)
    return (tx0, ty0, tx0 + T, ty0 + T)


def _app_entry(it, i):
    if isinstance(it, dict):
        d = dict(it)
        glyph = d.pop("glyph", d.get("label"))
        return d.pop("label", str(glyph).title()), glyph, d.pop("colors", None)
    if isinstance(it, str):
        return it.title() if it.lower() in APP_GLYPHS else it, it, None
    it = list(it)
    label, glyph = it[0], (it[1] if len(it) > 1 else it[0])
    cols = (it[2], it[3]) if len(it) > 3 else (it[2] if len(it) > 2 else None)
    return label, glyph, cols


def app_icon_sheet(cv, apps, cols=None, box=None, background=True, shadow="long", tile=0.6, radius=0.23,
                   glyph_color="#ffffff", label_color="#2b3140", labels=True, accent_color=None):
    """A labelled sheet of app icons in a grid. ``apps``: app words ("notes", "weather" ...), (label, glyph) or
    (label, glyph, light, deep) tuples, or dicts with label / glyph / colors. ``cols``: columns (default from the
    canvas shape); ``box``: area (fractions or pixels, default the canvas minus a margin); ``background``: True
    (soft grey gradient), a colour, or False; ``tile``: tile size as a share of the cell. Each shadow stays inside
    its own cell. ``glyph_color`` / ``accent_color``: glyph colour and the second colour for glyph details (see
    ``app_icon``). Returns the tile boxes in pixels."""
    _check_choice("app_icon_sheet", "shadow", shadow, APP_SHADOWS)
    items = [_app_entry(a, i) for i, a in enumerate(apps)]
    n = len(items)
    if n == 0:
        return []
    W, H = cv.w, cv.h
    if box is None:
        m = 0.04 * min(W, H)
        box = (m, m, W - m, H - m)
    bx = [float(v) for v in box]
    if max(abs(v) for v in bx) <= 1.0:
        bx = [bx[0] * W, bx[1] * H, bx[2] * W, bx[3] * H]
    if background is not False and background is not None:
        if background is True:
            cv.gradient([(0, "#f1f3f7"), (1, "#dde2ea")], axis="y")
            cv.grain(0.006, 7)                           # dither: no banding in the soft gradient
        else:
            cv.paint(background)
    if cols is None:
        aspect = (bx[2] - bx[0]) / max(bx[3] - bx[1], 1.0)
        cols = max(1, min(n, int(round(math.sqrt(n * aspect * 1.1)))))
    rows = int(math.ceil(n / cols))
    cw, ch = (bx[2] - bx[0]) / cols, (bx[3] - bx[1]) / rows
    out = []
    for i, (label, glyph, colors) in enumerate(items):
        r_, c_ = divmod(i, cols)
        x0, y0 = bx[0] + c_ * cw, bx[1] + r_ * ch
        T = float(tile) * min(cw, ch / (1.3 if labels else 1.0))
        block = T * (1.32 if labels else 1.0)
        tx0, ty0 = x0 + (cw - T) / 2, y0 + (ch - block) / 2
        cols_i = colors if colors is not None else APP_COLORS[i % len(APP_COLORS)]
        out.append(app_icon(cv, (tx0, ty0, tx0 + T, ty0 + T), glyph, colors=cols_i, shadow=shadow,
                            label=label if labels else None, radius=radius, glyph_color=glyph_color,
                            label_color=label_color, cell=(x0, y0, x0 + cw, y0 + ch), name=label,
                            accent_color=accent_color))
    return out


def _after_app_icon(cap, args, res):
    if res is None:
        return None
    nm = args.get("name") or args.get("label") or str(args.get("glyph"))
    return [dict(kind="object", name=str(nm), bbox=tuple(res), alias=["app icon", "icon", "tile", "app",
                                                                     str(args.get("glyph"))])]


from ._args import check_choice as _check_choice, strict as _strict_fn  # noqa: E402

app_icon = _strict_fn(_manifest.drawn("object", after=_after_app_icon)(app_icon), "app_icon")
app_icon_sheet = _strict_fn(app_icon_sheet, "app_icon_sheet")
__all__ += ["app_icon", "app_icon_sheet", "APP_GLYPHS", "APP_COLORS"]
