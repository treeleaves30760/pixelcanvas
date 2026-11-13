# -*- coding: utf-8 -*-
"""Engineering drawing sheets in millimetres: ISO line types, dimensions, sections, plans, isometric views.

Quick start::

    from pixelcanvas.drafting import Sheet, Profile
    sh = Sheet("A4", title="SPACER PLATE", drawing_no="PC-0001", material="S235JR")
    v = sh.view("FRONT", origin=(80, 110))                  # model (0, 0) lands at sheet (80, 110) mm
    v.profile(Profile(0, 0).to(100, 0, fillet=8).to(100, 60, fillet=8).to(0, 60).close())
    v.hole(50, 30, 20)                                       # circle + centre lines
    v.dim_h((0, 0), (100, 0))                                # stacked below the part automatically
    v.dim_v((100, 0), (100, 60), side="right")
    v.dim_diameter(50, 30, 20)
    cv = sh.draw()                                           # Canvas, 150 dpi by default
    sh.save_svg("plate.svg"); sh.save_dxf("plate.dxf")

Coordinates are millimetres with **y up** (like CAD): sheet coordinates start at the bottom-left corner of the
paper, view coordinates are model millimetres mapped by the view's origin and scale. Text heights, arrow sizes,
dimension offsets and line widths are paper millimetres. Angles are degrees counter-clockwise from +x.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import vector as _vec
from ._args import (check_choice as _check_choice, reject_unknown as _reject_unknown,
                    signature_options as _signature_options, strict_methods as _strict_methods,
                    unknown_name as _unknown_name)
from .color import rgb as _rgb

__all__ = ["Sheet", "View", "Profile", "iso", "fillet", "chamfer", "SHEET_SIZES", "LINE_TYPES", "MATERIALS",
           "FURNITURE", "FLOOR_PALETTES", "GDT_SYMBOLS", "parse_dxf"]

SHEET_SIZES = {"A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (297, 210)}

# name: (weight, dash pattern in mm or None, DXF linetype, DXF colour)
LINE_TYPES = {
    "visible": ("thick", None, "CONTINUOUS", 7),
    "hidden": ("thin", (3.0, 1.0), "HIDDEN", 4),
    "center": ("thin", (10.0, 1.5, 1.5, 1.5), "CENTER", 1),
    "phantom": ("thin", (10.0, 1.5, 1.5, 1.5, 1.5, 1.5), "PHANTOM", 6),
    "dim": ("thin", None, "CONTINUOUS", 3),
    "thin": ("thin", None, "CONTINUOUS", 7),
    "hatch": ("thin", None, "CONTINUOUS", 8),
    "cutting": ("thin", (10.0, 1.5, 1.5, 1.5), "CENTER", 1),
    "wall": ("thick", None, "CONTINUOUS", 7),
    "furniture": ("thin", None, "CONTINUOUS", 5),
    "frame": ("frame", None, "CONTINUOUS", 7),
    "text": ("thin", None, "CONTINUOUS", 7),
}
_ALIAS = {"dimension": "dim", "extension": "dim", "leader": "dim", "centre": "center", "chain": "center",
          "section": "cutting", "outline": "visible", "thick": "visible", "dashed": "hidden", "border": "frame"}

MATERIALS = ("general", "steel", "aluminium", "plastic", "concrete", "insulation", "earth")
GDT_SYMBOLS = ("straightness", "flatness", "circularity", "cylindricity", "profile_line", "profile_surface",
               "parallelism", "perpendicularity", "angularity", "position", "concentricity", "symmetry",
               "runout", "total_runout")
FURNITURE = ("bed", "single_bed", "sofa", "armchair", "table", "chair", "toilet", "basin", "shower", "bathtub",
             "counter", "sink", "hob", "fridge", "wardrobe", "desk")

# pale floor tints per room type (room.fill / colors(floors=...)); the type is read from the room name
FLOOR_PALETTES = {
    "pastel": dict(living="#f3e3c8", kitchen="#f5ecd2", dining="#f2e2cf", bedroom="#dbe6f4", bath="#d4ecec",
                   hall="#ece7e1", balcony="#e3eed6", office="#e6e1f1", storage="#e8e8e2", other="#efebe3"),
    "warm": dict(living="#f4dfc4", kitchen="#f6e8c9", dining="#f1d9c6", bedroom="#f2dfd6", bath="#e8e3d6",
                 hall="#efe8dc", balcony="#e6ebcf", office="#efe0cf", storage="#e9e4da", other="#f1ebe0"),
    "cool": dict(living="#dde9ee", kitchen="#e1eee6", dining="#e0e6ef", bedroom="#dfe2f2", bath="#d3eaef",
                 hall="#e6e9ec", balcony="#dcecdc", office="#e4e3f0", storage="#e4e7e9", other="#e9edf0"),
    "grey": dict(living="#ececec", kitchen="#e9e9e9", dining="#ebebeb", bedroom="#efefef", bath="#e4e6e8",
                 hall="#f2f2f2", balcony="#e6e8e4", office="#ededed", storage="#e5e5e5", other="#f0f0f0"),
}
_ROOM_TYPES = {
    "living": "living", "lounge": "living", "family": "living", "sitting": "living", "salon": "living",
    "kitchen": "kitchen", "kitchenette": "kitchen", "cook": "kitchen", "dining": "dining", "eat": "dining",
    "bed": "bedroom", "bedroom": "bedroom", "master": "bedroom", "guest": "bedroom", "nursery": "bedroom",
    "kids": "bedroom", "sleeping": "bedroom", "bath": "bath", "bathroom": "bath", "wc": "bath", "toilet": "bath",
    "shower": "bath", "ensuite": "bath", "washroom": "bath", "laundry": "bath", "powder": "bath",
    "hall": "hall", "hallway": "hall", "corridor": "hall", "entry": "hall", "entrance": "hall", "foyer": "hall",
    "lobby": "hall", "landing": "hall", "stair": "hall", "stairs": "hall", "vestibule": "hall",
    "balcony": "balcony", "terrace": "balcony", "patio": "balcony", "deck": "balcony", "loggia": "balcony",
    "garden": "balcony", "veranda": "balcony", "porch": "balcony",
    "office": "office", "study": "office", "studio": "office", "library": "office", "meeting": "office",
    "work": "office", "den": "office", "store": "storage", "storage": "storage", "closet": "storage",
    "utility": "storage", "pantry": "storage", "garage": "storage", "plant": "storage", "wardrobe": "storage",
}


def _room_type(name):
    """Room type (a FLOOR_PALETTES key) read from the first telling word of a room name."""
    import re
    for w in re.split(r"[^a-z]+", str(name).lower()):
        if w in _ROOM_TYPES:
            return _ROOM_TYPES[w]
        if w.endswith("s") and w[:-1] in _ROOM_TYPES:
            return _ROOM_TYPES[w[:-1]]
    return "other"


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, float(v))) * 255)) for v in c[:3])


def _color_hex(c, what="color"):
    """A colour option → '#rrggbb' (None stays None); a bad colour raises ValueError naming the option."""
    if c is None or c is False:
        return None
    try:
        return _hex(_rgb(c))
    except ValueError as e:
        raise ValueError(f"{what}: {e}") from None


def _floor_color(name, fill, plan):
    """Resolved floor tint of a room: its own ``fill`` first, then the plan palette; None = no tint."""
    if fill is False:
        return None
    if fill is None or fill is True:
        pal = plan if plan is not None else ("pastel" if fill is True else None)
    else:
        pal = fill
    if pal is None:
        return None
    if isinstance(pal, dict):
        low = str(name).lower()
        c = pal.get(low, pal.get(_room_type(name), pal.get("other")))
        return c if c is None else _color_hex(c)
    if isinstance(pal, str) and pal in FLOOR_PALETTES:
        return FLOOR_PALETTES[pal][_room_type(name)]
    return _color_hex(pal, "room fill")

_STYLES = {
    "white": dict(paper="#ffffff", ink="#111418", muted="#5a6270", poche="#3b3f45", fill="#e9ecef"),
    "paper": dict(paper="#f7f4ea", ink="#1b1d22", muted="#62666e", poche="#3d3e42", fill="#e6e1d2"),
    "blueprint": dict(paper="#1d4f8f", ink="#f2f7ff", muted="#b9cdec", poche="#c9daf3", fill="#2c62a6"),
}

_METRIC_PITCH = {1.6: 0.35, 2: 0.4, 2.5: 0.45, 3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 14: 2.0,
                 16: 2.0, 20: 2.5, 24: 3.0, 30: 3.5, 36: 4.0}


# --------------------------------------------------------------------------------------------- small maths

def _v(x, y):
    return np.array([float(x), float(y)])


def _unit(v):
    n = math.hypot(v[0], v[1])
    return np.array([v[0] / n, v[1] / n]) if n > 1e-12 else np.array([1.0, 0.0])


def _rot90(v):
    return np.array([-v[1], v[0]])


def _ang(v):
    return math.degrees(math.atan2(v[1], v[0]))


def _dir(a):
    r = math.radians(a)
    return np.array([math.cos(r), math.sin(r)])


def _fmt(v, decimals=None):
    if decimals is not None:
        return f"{v:.{decimals}f}"
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _scale_factor(scale):
    if scale is None:
        return None
    if isinstance(scale, (int, float)):
        return float(scale)
    a, b = str(scale).replace(" ", "").split(":")
    return float(a) / float(b)


def _scale_text(f):
    if f >= 1:
        return f"{_fmt(f)}:1"
    return f"1:{_fmt(1 / f)}"


def iso(x, y, z):
    """Isometric projection of model point (x, y, z) → 2-D (u, v), true lengths along the axes.

    x runs to the lower right, y (depth) to the upper right, z up; the viewer looks from the front-right-top,
    so the -y face (front) is on the left and the +x face on the right."""
    c, s = math.cos(math.pi / 6), 0.5
    return ((x + y) * c, (y - x) * s + z)


def fillet(p_prev, p, p_next, r):
    """Round the corner at ``p`` between straight edges → (t1, t2, centre, ccw): tangent points on the incoming
    and outgoing edges, arc centre and turning direction (ccw=True: arc runs counter-clockwise t1 → t2)."""
    a, b, c = np.asarray(p_prev, float), np.asarray(p, float), np.asarray(p_next, float)
    u1, u2 = _unit(b - a), _unit(c - b)
    cosang = float(np.clip(-u1 @ u2, -1, 1))
    th = math.acos(cosang)                             # interior angle
    if th < 1e-6 or abs(th - math.pi) < 1e-6:
        return b, b, b, True
    d = r / math.tan(th / 2)
    t1, t2 = b - u1 * d, b + u2 * d
    bis = _unit(-u1 + u2)
    ctr = b + bis * (r / math.sin(th / 2))
    ccw = (u1[0] * u2[1] - u1[1] * u2[0]) > 0
    return t1, t2, ctr, ccw


def chamfer(p_prev, p, p_next, c, c2=None):
    """Cut the corner at ``p`` by ``c`` along the incoming edge and ``c2`` (default ``c``) along the outgoing
    one → (t1, t2)."""
    a, b, d = np.asarray(p_prev, float), np.asarray(p, float), np.asarray(p_next, float)
    return b - _unit(b - a) * c, b + _unit(d - b) * (c if c2 is None else c2)


def _arc_pts(c, r, a0, a1, step_deg=4.0):
    n = max(2, int(math.ceil(abs(a1 - a0) / step_deg)) + 1)
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], -1)


def _circle_ring(cx, cy, r, n=72):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], -1)


def _slot_ring(x1, y1, x2, y2, w, n=24):
    a, b = _v(x1, y1), _v(x2, y2)
    u = _unit(b - a)
    an = _ang(u)
    r = w / 2
    p1 = _arc_pts(b, r, an - 90, an + 90, 180 / n)
    p2 = _arc_pts(a, r, an + 90, an + 270, 180 / n)
    return np.concatenate([p1, p2])


def _area(pts):
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# --------------------------------------------------------------------------------------------- Profile

class Profile:
    """Closed or open outline built from straight edges and arcs, with fillets and chamfers at corners.

    ``Profile(0, 0).to(80, 0, fillet=5).to(80, 40, chamfer=3).arc_to(40, 40, r=20).to(0, 40).close(fillet=2)``

    ``fillet=`` / ``chamfer=`` on ``to()`` modify the corner at the point just added; ``close(fillet=)`` the corner
    at the start point. ``by(dx, dy)`` adds a relative point. ``segments()`` → [('line', p0, p1) |
    ('arc', centre, r, a0, a1)] with arcs counter-clockwise from a0 to a1; ``points()`` → polygon (N, 2)."""

    def __init__(self, x=0.0, y=0.0):
        self.pts = [_v(x, y)]
        self.mods = [(0.0, 0.0)]
        self.edges = []                                  # None (line) or (r, cw, large)
        self.closed = False

    def to(self, x, y, fillet=0.0, chamfer=0.0):
        """Straight edge to (x, y); round (``fillet``) or cut (``chamfer``) the corner at (x, y)."""
        self.pts.append(_v(x, y))
        self.mods.append((float(fillet), float(chamfer)))
        self.edges.append(None)
        return self

    def by(self, dx, dy, fillet=0.0, chamfer=0.0):
        """Straight edge by a relative offset."""
        p = self.pts[-1]
        return self.to(p[0] + dx, p[1] + dy, fillet, chamfer)

    def arc_to(self, x, y, r, cw=False, large=False):
        """Circular arc of radius ``r`` to (x, y); ``cw`` turns clockwise, ``large`` takes the long way round."""
        self.pts.append(_v(x, y))
        self.mods.append((0.0, 0.0))
        self.edges.append((float(r), bool(cw), bool(large)))
        return self

    def close(self, fillet=0.0, chamfer=0.0):
        """Close the outline back to the start point."""
        if np.allclose(self.pts[-1], self.pts[0]):
            self.pts.pop()
            e = self.edges.pop()
            self.mods.pop()
            self.edges.append(e)
        else:
            self.edges.append(None)
        self.mods[0] = (float(fillet), float(chamfer))
        self.closed = True
        return self

    @staticmethod
    def _arc_centre(p0, p1, r, cw, large):
        chord = p1 - p0
        L = math.hypot(*chord)
        r = max(r, L / 2)
        h = math.sqrt(max(r * r - L * L / 4, 0.0))
        m = (p0 + p1) / 2
        n = _rot90(_unit(chord))
        left = (not cw) != large                         # ccw short arc → centre on the left of the chord
        c = m + n * h if left else m - n * h
        return c, r

    def segments(self):
        """Edges as [('line', p0, p1) | ('arc', centre, r, a0, a1)], arcs counter-clockwise a0 → a1."""
        return _profile_segments(self)

    def _segs(self):
        """Internal: like segments() but arcs carry a 6th item, True when traversed clockwise."""
        n = len(self.pts)
        ne = len(self.edges)
        start = [p.copy() for p in self.pts]              # trimmed start of edge i (at vertex i)
        end = [self.pts[(i + 1) % n].copy() for i in range(ne)]   # trimmed end of edge i
        corners = {}
        for i in range(n):
            f, c = self.mods[i]
            if not (f > 0 or c > 0):
                continue
            ein, eout = (i - 1) % ne if (self.closed or i > 0) else None, i if i < ne else None
            if ein is None or eout is None or (not self.closed and i == 0):
                continue
            if self.edges[ein] is not None or self.edges[eout] is not None:
                continue
            pp, p, pn = self.pts[(i - 1) % n], self.pts[i], self.pts[(i + 1) % n]
            if f > 0:
                t1, t2, ctr, ccw = fillet(pp, p, pn, f)
                corners[i] = ("arc", t1, t2, ctr, f, ccw)
            else:
                t1, t2 = chamfer(pp, p, pn, c)
                corners[i] = ("line", t1, t2)
            end[ein] = t1
            start[eout] = t2
        segs = []
        if self.closed and 0 in corners:                     # corner at the start point comes first
            cn = corners[0]
            if cn[0] == "line":
                segs.append(("line", cn[1], cn[2]))
            else:
                _, t1, t2, ctr, r, ccw = cn
                a0, a1 = _ang(t1 - ctr), _ang(t2 - ctr)
                if not ccw:
                    a0, a1 = a1, a0
                while a1 <= a0:
                    a1 += 360
                segs.append(("arc", ctr, r, a0, a1, not ccw))
        for i in range(ne):
            e = self.edges[i]
            p0, p1 = start[i], end[i]
            if e is None:
                if math.hypot(*(p1 - p0)) > 1e-9:
                    segs.append(("line", p0, p1))
            else:
                r, cw, large = e
                ctr, r = self._arc_centre(self.pts[i], self.pts[(i + 1) % n], r, cw, large)
                a0, a1 = _ang(self.pts[i] - ctr), _ang(self.pts[(i + 1) % n] - ctr)
                if cw:
                    a0, a1 = a1, a0
                while a1 <= a0:
                    a1 += 360
                segs.append(("arc", ctr, r, a0, a1, cw))
            j = (i + 1) % n
            if j in corners and j != 0 and (self.closed or j < n - 1):
                cn = corners[j]
                if cn[0] == "line":
                    segs.append(("line", cn[1], cn[2]))
                else:
                    _, t1, t2, ctr, r, ccw = cn
                    a0, a1 = _ang(t1 - ctr), _ang(t2 - ctr)
                    if not ccw:
                        a0, a1 = a1, a0
                    while a1 <= a0:
                        a1 += 360
                    segs.append(("arc", ctr, r, a0, a1, not ccw))
        return segs

    def points(self, step_deg=4.0):
        """Outline as an (N, 2) polygon (arcs sampled every ``step_deg`` degrees)."""
        out = []
        for s in self._segs():
            if s[0] == "line":
                out.append(np.array([s[1]]))
            else:
                _, c, r, a0, a1, rev = s
                p = _arc_pts(c, r, a0, a1, step_deg)
                if rev:
                    p = p[::-1]
                out.append(p[:-1])
        if not self.closed:
            last = self._segs()[-1]
            if last[0] == "line":
                out.append(np.array([last[2]]))
            else:
                _, c, r, a0, a1, rev = last
                out.append(np.array([c + r * _dir(a0 if rev else a1)]))
        return np.concatenate(out) if out else np.zeros((0, 2))

    def area(self):
        """Enclosed area (model units²)."""
        return abs(_area(self.points()))


def _profile_segments(p):
    segs = p._segs()
    return [(s[0], s[1], s[2]) if s[0] == "line" else ("arc", s[1], s[2], s[3], s[4]) for s in segs]


# --------------------------------------------------------------------------------------------- polygon helpers

def _rings_of(region, holes=()):
    """Region spec → list of (N, 2) rings (outer + holes, even-odd)."""
    rings = []

    def one(r):
        if isinstance(r, Profile):
            return [r.points()]
        if isinstance(r, tuple) and len(r) == 3 and all(isinstance(v, (int, float)) for v in r):
            return [_circle_ring(*r)]                     # (cx, cy, r) circle
        try:
            a = np.asarray(r, float)
        except (ValueError, TypeError):
            a = None
        if a is not None and a.ndim == 2 and a.shape[1] == 2:
            return [a]
        out = []
        for q in r:
            out += one(q)
        return out

    rings += one(region)
    for h in holes:
        rings += one(h)
    return rings


def _edges_of(rings):
    E = []
    for r in rings:
        r = np.asarray(r, float)
        E.append(np.concatenate([r, np.roll(r, -1, 0)], 1))
    return np.concatenate(E) if E else np.zeros((0, 4))


def _inside(P, E):
    """Even-odd point-in-region for points P (N, 2) against edges E (M, 4)."""
    P = np.asarray(P, float).reshape(-1, 2)
    if not len(E):
        return np.zeros(len(P), bool)
    x, y = P[:, 0:1], P[:, 1:2]
    x0, y0, x1, y1 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
    cond = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (np.sum(cond & (x < xi), 1) % 2) == 1


def _seg_dist(P, E):
    """Distance from points P (N, 2) to the nearest of edges E (M, 4)."""
    a, b = E[None, :, :2], E[None, :, 2:]
    d = b - a
    w = P[:, None, :] - a
    t = np.clip(np.sum(w * d, -1) / np.maximum(np.sum(d * d, -1), 1e-18), 0, 1)
    q = a + d * t[..., None]
    return np.sqrt(np.min(np.sum((P[:, None, :] - q) ** 2, -1), 1))


def _clip_segments(segs, E, keep_inside=True):
    """Clip segments [(x0, y0, x1, y1)] to the region with edges E → list of kept pieces."""
    out = []
    if not len(segs):
        return out
    S = np.asarray(segs, float)
    ex0, ey0, ex1, ey1 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
    ed = np.stack([ex1 - ex0, ey1 - ey0], -1)
    for s in S:
        p, q = s[:2], s[2:]
        d = q - p
        den = d[0] * ed[:, 1] - d[1] * ed[:, 0]
        ok = np.abs(den) > 1e-12
        wx, wy = ex0 - p[0], ey0 - p[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (wx * ed[:, 1] - wy * ed[:, 0]) / den
            u = (wx * d[1] - wy * d[0]) / den
        hit = ok & (t > 0) & (t < 1) & (u >= 0) & (u <= 1)
        ts = np.concatenate([[0.0], np.sort(t[hit]), [1.0]])
        mids = (ts[:-1] + ts[1:]) / 2
        M = p[None] + mids[:, None] * d[None]
        ins = _inside(M, E)
        for k in range(len(mids)):
            if ins[k] == keep_inside and ts[k + 1] - ts[k] > 1e-9:
                a, b = p + d * ts[k], p + d * ts[k + 1]
                out.append((a[0], a[1], b[0], b[1]))
    return out


def _clip_poly_convex(poly, clip):
    """Sutherland-Hodgman: clip polygon by a convex CCW polygon."""
    out = [tuple(p) for p in poly]
    C = [tuple(p) for p in clip]
    for i in range(len(C)):
        a, b = np.array(C[i]), np.array(C[(i + 1) % len(C)])
        inp, out = out, []
        if not inp:
            break

        def side(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        for j in range(len(inp)):
            P, Q = np.array(inp[j]), np.array(inp[(j + 1) % len(inp)])
            sp, sq = side(P), side(Q)
            if sp >= 0:
                out.append(tuple(P))
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append(tuple(P + (Q - P) * t))
    return out


# --------------------------------------------------------------------------------------------- text metrics

@lru_cache(maxsize=16)
def _cap_ratio(family, bold):
    from . import text as _text
    kw = _vec._tkw(family, "bold" if bold else "normal", "normal", "H")
    f = _text.load_font(kw["font"], 200, kw["bold"], text="H")
    b = f.getbbox("H")
    return max(0.55, min(0.85, (b[3] - b[1]) / 200.0))


@lru_cache(maxsize=8192)
def _em_width(s, family, bold):
    w, _ = _vec.measure_text(s, 100, family, "bold" if bold else "normal")
    return w / 100.0


def _tag(ents, **kw):
    """Mark entities with what they draw (``o``) for the scene manifest."""
    for e in ents:
        e.update(kw)
    return ents


_OBJ_OF = {m: "dimension" for m in ("dim", "dim_h", "dim_v", "dim_aligned", "dim_chain", "dim_baseline",
                                    "dim_radius", "dim_diameter", "dim_angle", "dim_chamfer")}


def _E(t, lt, **kw):
    kw["t"], kw["lt"] = t, lt
    return kw


# --------------------------------------------------------------------------------------------- View

class View:
    """A view on the sheet: model millimetres mapped to the sheet by ``origin`` and ``scale``.

    Geometry calls take model coordinates (y up). Sizes that belong to the paper (text height, arrow size,
    dimension offsets, centre-line overshoot) are paper millimetres. Every call returns the view (chainable)
    unless stated otherwise."""

    def __init__(self, sheet, name, origin=(0.0, 0.0), scale=None, label=None):
        self.sheet = sheet
        self.name = name
        self._ox, self._oy = float(origin[0]), float(origin[1])
        self.s = _scale_factor(scale) if scale is not None else sheet.s
        self.label_text = label
        self._label0 = label
        self._sheet_scale = scale is None                     # follows the sheet's scale (refitted with it)
        self._calls, self._replaying, self._rec_depth = [], False, 0
        self._reset_state()
        self._auto = None                                    # True / (x, y): centre this view (group) on the sheet
        self._follow = None                                  # (base view, side, gap or None) for projected views
        self._shift = [0.0, 0.0]                             # paper offset applied when the sheet is collected

    def _reset_state(self):
        self._geom, self._anno, self._lazy = [], [], []
        self._walls, self._openings = [], []
        self._iso = None
        self._n = 0
        self._n_dims = 0
        self._oids = 0
        self._loops_key = None
        self.rooms = {}                                      # room name -> area in m² (from room())
        self._floors = None                                  # plan palette for room floors (colors())
        self._furn_col, self._furn_fill = None, None         # plan furniture line / fill colour (colors())

    def _next_oid(self):
        self._oids += 1
        return self._oids

    def _replay(self):
        """Redraw every call on a clean slate (used when the sheet's scale changes)."""
        calls = list(self._calls)
        self._reset_state()
        self.label_text = self._label0
        self._replaying = True
        try:
            for fn, a, k in calls:
                fn(self, *a, **k)
        finally:
            self._replaying = False

    # ---- mapping
    @property
    def ox(self):
        """Sheet x (mm from the paper's left edge) of model (0, 0); set once the view is placed (after a draw)."""
        return self._ox + self._shift[0]

    @property
    def oy(self):
        """Sheet y (mm from the paper's bottom edge) of model (0, 0); set once the view is placed."""
        return self._oy + self._shift[1]

    def to_sheet(self, x, y):
        """Model point → sheet millimetres (from the paper's bottom-left corner), where the view was placed."""
        return (self.ox + self.s * x, self.oy + self.s * y)

    def to_px(self, x, y, cv):
        """Model point → pixel (x, y) on the canvas ``cv`` the sheet was drawn on (or a (width, height) size)."""
        W, H = (cv.w, cv.h) if hasattr(cv, "w") else (float(cv[0]), float(cv[1]))
        sh = self.sheet
        k = min(W / sh.W, H / sh.H)
        X, Y = self.to_sheet(x, y)
        return ((W - sh.W * k) / 2 + k * X, (H - sh.H * k) / 2 + k * (sh.H - Y))

    def _to_paper(self, x, y):
        """Model point → paper mm before the view is placed (layout space)."""
        return (self._ox + self.s * x, self._oy + self.s * y)

    def _P(self, x, y):
        return np.array(self._to_paper(x, y))

    def _paper(self, mm):
        """Paper millimetres → model units."""
        return mm / self.s

    def _later(self, prio, fn, key=0.0):
        self._n += 1
        self._lazy.append((prio, key, self._n, fn, getattr(self, "_cur_obj", None)))

    def _lt(self, kind):
        k = _ALIAS.get(kind, kind)
        if k not in LINE_TYPES:
            raise _unknown_name("line type", kind, list(LINE_TYPES) + list(_ALIAS))
        return k

    # ---- primitive geometry (model coordinates)
    def line(self, x1, y1, x2, y2, kind="visible", width=None):
        """Straight line. ``kind``: visible hidden center phantom thin dim hatch cutting wall furniture."""
        self._geom.append(_E("line", self._lt(kind), p=(x1, y1, x2, y2), w=width))
        return self

    def polyline(self, points, kind="visible", closed=False, width=None):
        """Polyline through ``points``; ``closed=True`` joins the last point to the first."""
        pts = [tuple(map(float, p)) for p in points]
        self._geom.append(_E("poly", self._lt(kind), pts=pts, closed=closed, w=width))
        return self

    def polygon(self, points, kind="visible", width=None):
        """Closed polygon outline."""
        return self.polyline(points, kind, closed=True, width=width)

    def rect(self, x, y, w, h, kind="visible", r=0.0, chamfer=0.0):
        """Rectangle with lower-left corner (x, y); ``r`` rounds and ``chamfer`` cuts all four corners."""
        p = Profile(x, y).to(x + w, y, r, chamfer).to(x + w, y + h, r, chamfer).to(x, y + h, r, chamfer)
        g0 = len(self._geom)
        self.profile(p.close(r, chamfer), kind)
        if chamfer:
            for e in self._geom[g0:]:
                if e["t"] == "line" and abs(math.hypot(e["p"][2] - e["p"][0], e["p"][3] - e["p"][1])
                                            - chamfer * math.sqrt(2)) < 1e-6:
                    e.update(o="chamfer", ov=self.name)
        return self

    def circle(self, cx, cy, r, kind="visible"):
        """Full circle."""
        self._geom.append(_E("circle", self._lt(kind), c=(cx, cy), r=float(r)))
        return self

    def arc(self, cx, cy, r, start, end, kind="visible"):
        """Arc counter-clockwise from ``start`` to ``end`` degrees."""
        a0, a1 = float(start), float(end)
        while a1 <= a0:
            a1 += 360
        self._geom.append(_E("arc", self._lt(kind), c=(cx, cy), r=float(r), a=(a0, a1)))
        return self

    def profile(self, profile, kind="visible"):
        """Draw a ``Profile`` outline (lines + arcs)."""
        for s in _profile_segments(profile):
            if s[0] == "line":
                self.line(s[1][0], s[1][1], s[2][0], s[2][1], kind)
            else:
                _, c, r, a0, a1 = s
                self.arc(c[0], c[1], r, a0, a1, kind)
        return self

    def slot(self, x1, y1, x2, y2, width, kind="visible", centerlines=True):
        """Obround slot between arc centres (x1, y1) and (x2, y2), ``width`` wide, with centre lines."""
        a, b = _v(x1, y1), _v(x2, y2)
        u = _unit(b - a)
        n = _rot90(u) * width / 2
        an = _ang(u)
        self.line(*(a + n), *(b + n), kind=kind).line(*(a - n), *(b - n), kind=kind)
        self.arc(b[0], b[1], width / 2, an - 90, an + 90, kind).arc(a[0], a[1], width / 2, an + 90, an + 270, kind)
        if centerlines:
            ext = self._paper(2.5)
            p, q = a - u * (width / 2 + ext), b + u * (width / 2 + ext)
            self.line(*p, *q, kind="center")
            for c in (a, b):
                m = _rot90(u) * (width / 2 + ext)
                self.line(*(c - m), *(c + m), kind="center")
        return self

    def centerline(self, x1, y1, x2, y2, extend=3.0):
        """Centre line from (x1, y1) to (x2, y2), extended ``extend`` paper mm beyond both ends."""
        a, b = _v(x1, y1), _v(x2, y2)
        u = _unit(b - a) * self._paper(extend)
        return self.line(*(a - u), *(b + u), kind="center")

    def centre_mark(self, cx, cy, r, extend=2.5, angle=0.0):
        """Two crossing centre lines through (cx, cy) reaching ``extend`` paper mm past radius ``r``."""
        L = r + self._paper(extend)
        for a in (angle, angle + 90):
            d = _dir(a) * L
            self.line(cx - d[0], cy - d[1], cx + d[0], cy + d[1], kind="center")
        return self

    center_mark = centre_mark

    def hole(self, cx, cy, d, thread=None, marks=True, kind="visible"):
        """Hole of diameter ``d`` with centre lines. ``thread="M8"`` (or True) draws the conventional
        thread: minor diameter thick full circle, nominal diameter thin three-quarter circle."""
        r = d / 2
        if thread:
            nom = d
            if isinstance(thread, str) and thread.upper().startswith("M"):
                nom = float(thread[1:].split("x")[0].split("X")[0])
            pitch = _METRIC_PITCH.get(nom, nom * 0.15)
            minor = nom - 1.0825 * pitch
            self.circle(cx, cy, minor / 2, kind)
            self.arc(cx, cy, nom / 2, 100, 350, "thin")
            r = nom / 2
        else:
            self.circle(cx, cy, r, kind)
        _tag(self._geom[-2:] if thread else self._geom[-1:], o="hole", ov=self.name, d=float(d))
        if marks:
            self.centre_mark(cx, cy, r)
        return self

    def bolt_circle(self, cx, cy, pcd, n, d, start=90.0, thread=None):
        """``n`` holes of diameter ``d`` on pitch circle ``pcd`` (centre-line circle + radial centre lines).
        Returns the list of hole centres."""
        self.circle(cx, cy, pcd / 2, "center")
        cs = []
        for i in range(n):
            a = start + 360.0 * i / n
            p = _v(cx, cy) + _dir(a) * pcd / 2
            self.hole(p[0], p[1], d, thread=thread, marks=False)
            rr = d / 2 + self._paper(2.5)
            q = _dir(a) * rr
            self.line(*(p - q), *(p + q), kind="center")
            cs.append((float(p[0]), float(p[1])))
        self.centre_mark(cx, cy, pcd / 2)
        return cs

    def thread(self, x1, y1, x2, y2, d, internal=False, pitch=None, end=True):
        """Threaded length along the axis (x1, y1) → (x2, y2), seen from the side (conventional
        representation): crests thick, roots thin, thick thread-end line at (x2, y2)."""
        a, b = _v(x1, y1), _v(x2, y2)
        u = _unit(b - a)
        n = _rot90(u)
        p = pitch or _METRIC_PITCH.get(d, d * 0.15)
        minor = d - 1.2269 * p
        ro, ri = d / 2, minor / 2
        thick_r, thin_r = (ri, ro) if internal else (ro, ri)
        for sgn in (1, -1):
            self.line(*(a + n * thick_r * sgn), *(b + n * thick_r * sgn), "visible")
            self.line(*(a + n * thin_r * sgn), *(b + n * thin_r * sgn), "thin")
        if end:
            self.line(*(b + n * ro), *(b - n * ro), "visible")
        return self

    def break_line(self, x1, y1, x2, y2, zigzags=1, extend=2.0):
        """Thin break line with a zigzag (ISO 128 type 01.1), extended ``extend`` paper mm past both ends."""
        a, b = _v(x1, y1), _v(x2, y2)
        u = _unit(b - a)
        n = _rot90(u)
        e = self._paper(extend)
        a, b = a - u * e, b + u * e
        L = math.hypot(*(b - a))
        z = self._paper(3.0)
        pts = [a]
        for k in range(zigzags):
            m = a + u * L * (k + 1) / (zigzags + 1)
            pts += [m - u * z * 0.5, m + n * z - u * z * 0.15, m - n * z + u * z * 0.15, m + u * z * 0.5]
        pts.append(b)
        return self.polyline(pts, "thin")

    # ---- hatching / sections
    def hatch(self, region, holes=(), material="general", angle=45.0, spacing=None, outline=False, mirror=None):
        """Section hatching inside ``region`` (Profile, point list, (cx, cy, r) circle, or a list of these)
        minus ``holes``. ``material``: general steel aluminium plastic concrete insulation earth.
        ``spacing`` in paper mm (default 2.5 – 3). ``outline=True`` also draws the boundary as visible lines.
        ``mirror="x"`` adds the mirror image about the x axis (y → -y), ``"y"`` about the y axis: draw half of a
        turned part and get the whole section, hatched in one direction as one part."""
        rings = _rings_of(region, holes)
        if mirror:
            if mirror not in ("x", "y"):
                _check_choice("region", "mirror", mirror, ["x", "y"])
            f = np.array([1.0, -1.0]) if mirror == "x" else np.array([-1.0, 1.0])
            rings = list(rings) + [np.asarray(r, float)[::-1] * f for r in rings]
        E = _edges_of(rings)
        sp = self._paper(spacing if spacing is not None else (3.0 if material in ("steel", "aluminium") else 2.5))
        for s in _hatch_pattern(rings, E, material, angle, sp, self._paper):
            self._geom.append(_E("line", "hatch", p=tuple(float(v) for v in s), w=None))
        if outline:
            for r in rings:
                self.polyline(r, "visible", closed=True)
        return self

    def section(self, outline, gaps=(), material="steel", beyond=(), angle=45.0, spacing=None, centerlines=True,
                mirror=None):
        """The whole cut body of a section view: its outline, the hatching on the cut faces, ``gaps`` (bores and
        holes the plane cuts through, left open with their edges drawn) and the visible edges ``beyond`` the plane.

        ``outline``: the cut body as (x0, y0, x1, y1), a point list or a Profile. ``gaps``: (x0, y0, x1, y1)
        rectangles, point lists or (cx, cy, r) circles; a rectangle gets its centre line when ``centerlines``.
        ``beyond``: lines (x1, y1, x2, y2) or (x1, y1, x2, y2, kind) seen past the cut. ``material`` and
        ``angle`` / ``spacing`` as ``hatch``; ``mirror="x"`` completes a half section of a turned part."""
        def ring(g):
            if isinstance(g, (tuple, list)) and len(g) == 4 and all(isinstance(v, (int, float)) for v in g):
                return _rect_pts(*g)
            return g
        region = ring(outline)
        holes = [ring(g) for g in gaps]
        self.hatch(region, holes=holes, material=material, angle=angle, spacing=spacing, outline=True, mirror=mirror)
        for g in gaps:
            if centerlines and isinstance(g, (tuple, list)) and len(g) == 4 and all(isinstance(v, (int, float))
                                                                                  for v in g):
                x0, y0, x1, y1 = g
                if (x1 - x0) >= (y1 - y0):
                    self.centerline(x0, (y0 + y1) / 2, x1, (y0 + y1) / 2, extend=4.0)
                else:
                    self.centerline((x0 + x1) / 2, y0, (x0 + x1) / 2, y1, extend=4.0)
        for b in beyond:
            kind = b[4] if len(b) > 4 else "visible"
            self.line(b[0], b[1], b[2], b[3], kind)
        return self

    def cutting_plane(self, points, letter="A", direction="left", extend=6.0):
        """Cutting-plane line through model ``points`` (2 or more) with thick ends, arrows pointing in the
        viewing ``direction`` ('left' 'right' 'up' 'down' or an angle / vector) and the section ``letter``."""
        P = [self._P(*p) for p in points]
        u0, u1 = _unit(P[1] - P[0]), _unit(P[-1] - P[-2])
        P[0] = P[0] - u0 * extend
        P[-1] = P[-1] + u1 * extend
        if isinstance(direction, str):
            d = {"left": _v(-1, 0), "right": _v(1, 0), "up": _v(0, 1), "down": _v(0, -1)}[direction]
        elif isinstance(direction, (int, float)):
            d = _dir(direction)
        else:
            d = _unit(np.asarray(direction, float))
        sh = self.sheet
        a = self._anno
        a.append(_E("poly", "cutting", pts=[tuple(p) for p in P], closed=False, w=None))
        thick = sh.thick * 1.4
        L = 6.0
        for i, (end, u) in enumerate(((P[0], u0), (P[-1], -u1))):
            a.append(_E("line", "cutting", p=(*end, *(end + u * L)), w=thick))
        for k in range(1, len(P) - 1):                       # thick marks at changes of direction
            a.append(_E("poly", "cutting", pts=[tuple(P[k] - _unit(P[k] - P[k - 1]) * 4), tuple(P[k]),
                                                 tuple(P[k] + _unit(P[k + 1] - P[k]) * 4)], closed=False,
                        w=thick))
        AL, AW = sh.arrow
        for end, u in ((P[0], u0), (P[-1], -u1)):
            tip = end + u * 2.5
            tail = tip - d * 10.0
            a.append(_E("line", "cutting", p=(*tail, *(tip - d * AL * 1.4)), w=sh.thin))
            a.append(_arrow(tip, d, AL * 1.6, AW * 1.6, "cutting"))
            lp = tail - d * (sh.text_h * 0.9)
            a += _tag(sh._text(lp, letter, sh.text_h * 1.4, ha="m", va="m", bold=True), mv="cut",
                      gid=self._next_oid(), pull=tuple(d))
        return self

    def section_label(self, letter="A"):
        """Label this view "SECTION A-A" (placed under the view)."""
        self.label_text = f"SECTION {letter}-{letter}"
        return self

    def label(self, text):
        """Label placed centred under the view (with the scale when it differs from the sheet scale)."""
        self.label_text = text
        return self

    def detail_circle(self, cx, cy, r, letter="B"):
        """Thin circle marking an area shown enlarged in a detail view, with its letter."""
        self.circle(cx, cy, r, "thin")
        p = self._P(cx, cy) + _dir(45) * (r * self.s + 2.0)
        self._anno += self.sheet._text(p, letter, self.sheet.text_h * 1.4, ha="l", va="b", bold=True)
        return self

    # ---- text & notes
    def text(self, x, y, s, h=None, anchor="lb", rot=0.0, bold=False, kind="text"):
        """Text at model point (x, y); ``h`` = capital height in paper mm (default 3.5);
        ``anchor`` = horizontal l/m/r + vertical b/m/t."""
        ha, va = _anchor2(anchor)
        self._anno += self.sheet._text(self._P(x, y), s, h or self.sheet.text_h, ha, va, rot, bold, lt=kind)
        return self

    def leader(self, x, y, text, dx=10.0, dy=8.0, arrow="arrow", h=None, clear=True):
        """Leader from model point (x, y) to a note offset (dx, dy) paper mm away. ``arrow``: 'arrow' (ends on
        an edge), 'dot' (ends inside a surface) or None. ``clear=True`` lengthens the leader along its direction
        until the note is clear of the part's lines and of the other annotations."""
        sh = self.sheet
        h = h or sh.text_h
        T = self._P(x, y)
        D = _v(dx, dy)
        L0 = math.hypot(dx, dy)
        u = D / L0 if L0 > 1e-9 else _v(1, 0)
        sgn = 1.0 if dx >= 0 else -1.0
        lines = str(text).split("\n")
        tw = max(sh._tw(ln, h) for ln in lines)

        def build(ext):
            K = T + u * (L0 + ext)
            S = K + _v(sgn * 3.0, 0)
            a = []
            if arrow == "arrow":
                d = _unit(T - K)
                a.append(_E("poly", "dim", pts=[tuple(S), tuple(K), tuple(T - d * sh.arrow[0] * 0.9)],
                            closed=False, w=None))
                a.append(_arrow(T, d, *sh.arrow))
            else:
                a.append(_E("poly", "dim", pts=[tuple(S), tuple(K), tuple(T)], closed=False, w=None))
                if arrow == "dot":
                    a.append(_E("dot", "dim", c=tuple(T), r=0.75))
            for i, ln in enumerate(lines):
                p = S + _v(sgn * 1.0, -(i - (len(lines) - 1) / 2) * h * 1.6)
                a += sh._text(p, ln, h, ha="l" if sgn > 0 else "r", va="m", halo=True)
            xa, xb = sorted((K[0] + sgn * 1.0, S[0] + sgn * (2.0 + tw)))
            hh = len(lines) * h * 1.6 / 2 + 0.6
            return a, (xa, K[1] - hh, xb, K[1] + hh)

        if not clear:
            self._anno += build(0.0)[0]
            return self

        def run():
            P = self._obstacle_points()
            best = None
            for ext in np.arange(0.0, 40.01, 1.5):
                ents, (x0, y0, x1, y1) = build(ext)
                hits = int(np.count_nonzero((P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)))
                if best is None or hits < best[0]:
                    best = (hits, ents)
                if hits == 0:
                    break
            self._anno.extend(best[1])
        self._later(3, run)
        return self

    def _obstacle_points(self):
        """Sample points of everything drawn in this view so far (paper mm)."""
        geom = [_map_entity(e, self) for e in self._out_geom if e["lt"] != "text"]
        P = _sample_ents(geom + [e for e in self._anno if e["t"] != "text"])
        boxes = [_text_corners(e, self.sheet, pad=0.3) for e in self._anno if e["t"] == "text" and e["s"].strip()]
        for c in boxes:
            o, ex, ey = c[0], c[1] - c[0], c[3] - c[0]
            na = max(3, int(np.hypot(*ex) / 1.2) + 1)
            nb = max(3, int(np.hypot(*ey) / 1.2) + 1)
            A, B = np.meshgrid(np.linspace(0, 1, na), np.linspace(0, 1, nb))
            P = np.vstack([P, o + A.reshape(-1, 1) * ex + B.reshape(-1, 1) * ey])
        return P

    def balloon(self, x, y, n, dx=12.0, dy=12.0, d=8.0):
        """Item balloon: circle with number ``n`` at (dx, dy) paper mm from model point (x, y), leader ending in
        a dot on the part."""
        sh = self.sheet
        T = self._P(x, y)
        C = T + _v(dx, dy)
        u = _unit(T - C)
        self._anno.append(_E("line", "dim", p=(*(C + u * d / 2), *T), w=None))
        self._anno.append(_E("dot", "dim", c=tuple(T), r=0.75))
        self._anno.append(_E("circle", "dim", c=tuple(C), r=d / 2, w=sh.thin, fill="paper"))
        self._anno += sh._text(C, str(n), sh.text_h * 1.15, "m", "m")
        return self

    def surface_finish(self, x, y, value="Ra 3.2", angle=90.0, removal=True, process=None):
        """Surface texture symbol (ISO 1302 / 21920) standing on model point (x, y), pointing into the surface;
        ``angle`` = direction from the surface to the symbol (90 = above). ``value`` is written on the long leg."""
        sh = self.sheet
        h = sh.text_h
        T = self._P(x, y)
        up = _dir(angle)
        rt = -_rot90(up)                                 # to the right when looking along -up
        if not (-90 + 1e-6 < _ang(rt) <= 90 + 1e-6):     # keep the value readable (mirror the symbol)
            rt = -rt
        h1 = h * 1.4
        h2 = h1 * 2.0

        def P(a, b):
            return T + rt * a + up * b

        c60 = 1 / math.tan(math.radians(60))
        L = P(-h1 * c60, h1)
        R = P(h2 * c60, h2)
        a = self._anno
        a.append(_E("poly", "dim", pts=[tuple(L), tuple(T), tuple(R)], closed=False, w=None))
        if removal:
            a.append(_E("line", "dim", p=(*L, *P(h1 * c60, h1)), w=None))
        tw = sh._tw(value, h) if value else 0
        if value or process:
            ext = max(tw + 2.0, 6.0)
            a.append(_E("line", "dim", p=(*R, *(R + rt * ext)), w=None))
            rot = _ang(rt)
            if value:
                a += sh._text(R + rt * (ext / 2) - up * (h * 0.35), value, h * 0.85, "m", "t", rot=rot)
            if process:
                a += sh._text(R + rt * (ext / 2) + up * 0.8, process, h * 0.85, "m", "b", rot=rot)
        return self

    def datum(self, x, y, letter="A", angle=-90.0, length=7.0):
        """Datum feature symbol: filled triangle on model point (x, y), stem towards ``angle`` (degrees,
        direction away from the surface), square frame with the datum ``letter``."""
        sh = self.sheet
        T = self._P(x, y)
        d = _dir(angle)
        n = _rot90(d)
        s = sh.text_h * 0.95
        tip = T + d * s * 0.866
        tri = [tuple(T + n * s / 2), tuple(T - n * s / 2), tuple(tip)]
        a = self._anno
        a.append(_E("solid", "dim", pts=tri))
        box = sh.text_h * 2.0
        stem_end = T + d * length
        a.append(_E("line", "dim", p=(*tip, *stem_end), w=None))
        c = stem_end + d * box / 2
        sq = [c + (-box / 2, -box / 2), c + (box / 2, -box / 2), c + (box / 2, box / 2), c + (-box / 2, box / 2)]
        a.append(_E("poly", "dim", pts=[tuple(p) for p in sq], closed=True, w=None, fill="paper"))
        a += sh._text(c, letter, sh.text_h * 1.15, "m", "m")
        return self

    def fcf(self, x, y, symbol, tolerance, datums=(), diameter=False, leader=None, modifier=None):
        """Feature control frame (GD&T) with its left-middle at model point (x, y): characteristic ``symbol``
        (see GDT_SYMBOLS), ``tolerance`` value, optional Ø prefix and datum letters. ``leader=(lx, ly)``
        draws an arrow from the frame to that model point."""
        sh = self.sheet
        h = sh.text_h
        H = h * 2.0
        O = self._P(x, y)
        tol = ("Ø" if diameter else "") + str(tolerance) + (f" {modifier}" if modifier else "")
        widths = [H, max(sh._tw(tol, h) + 3.0, H)] + [max(sh._tw(dd, h) + 3.0, H) for dd in datums]
        a = self._anno
        x0 = O[0]
        xs = [x0]
        for w in widths:
            xs.append(xs[-1] + w)
        y0, y1 = O[1] - H / 2, O[1] + H / 2
        a.append(_E("poly", "dim", pts=[(xs[0], y0), (xs[-1], y0), (xs[-1], y1), (xs[0], y1)], closed=True,
                    w=None, fill="paper"))
        for xx in xs[1:-1]:
            a.append(_E("line", "dim", p=(xx, y0, xx, y1), w=None))
        a += _gdt_symbol(symbol, _v((xs[0] + xs[1]) / 2, O[1]), H * 0.62)
        a += sh._text(_v((xs[1] + xs[2]) / 2, O[1]), tol, h, "m", "m")
        for i, dd in enumerate(datums):
            a += sh._text(_v((xs[2 + i] + xs[3 + i]) / 2, O[1]), dd, h, "m", "m")
        if leader is not None:
            T = self._P(*leader)
            S = _v(xs[-1], O[1]) if T[0] > xs[-1] else _v(xs[0], O[1])
            if abs(T[1] - O[1]) < 1.0:                            # straight across to a vertical surface
                K = S
            elif xs[0] - 2 < T[0] < xs[-1] + 2:                   # target under / over the frame: drop down
                S = _v(T[0], y0 if T[1] < O[1] else y1)
                K = S
            else:
                K = _v(T[0], O[1])
            d = _unit(T - K)
            a.append(_E("poly", "dim", pts=[tuple(S), tuple(K), tuple(T - d * sh.arrow[0] * 0.9)], closed=False,
                        w=None))
            a.append(_arrow(T, d, *sh.arrow))
        return self

    # ---- dimensions (ISO 129)
    def _part_bbox(self):
        if getattr(self, "_bb", None) is not None:
            return self._bb
        pts = []
        for e in self._out_geom:
            if e["lt"] not in ("visible", "hidden", "wall"):
                continue
            pts.append(_entity_pts(e))
        if not pts:
            P = np.array([[self._ox, self._oy]])
        else:
            P = np.concatenate(pts) * self.s + np.array([self._ox, self._oy])
        self._bb = (P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max())
        return self._bb

    def _tier(self, side, lo, hi, offset=None):
        sh = self.sheet
        bb = self._part_bbox()
        edge = {"bottom": bb[1], "top": bb[3], "left": bb[0], "right": bb[2]}[side]
        sgn = -1.0 if side in ("bottom", "left") else 1.0
        if offset is not None:
            c = edge + sgn * offset
        else:
            tiers = self._tiers.setdefault(side, [])
            k = 0
            while True:
                if k >= len(tiers):
                    tiers.append([])
                if all(hi < a - 1.5 or lo > b + 1.5 for a, b in tiers[k]):
                    tiers[k].append((lo, hi))
                    break
                k += 1
            c = edge + sgn * (sh.dim_first + k * sh.dim_step)
        ext = self._extent.get(side)
        self._extent[side] = c if ext is None else (min(ext, c) if sgn < 0 else max(ext, c))
        return c

    def _value_text(self, value, text, prefix, suffix, decimals):
        if text is not None:
            t = str(text)
            return t.replace("{}", _fmt(value, decimals)) if "{}" in t else t
        return f"{prefix}{_fmt(value, decimals)}{suffix}"

    def _linear(self, p1, p2, side, axis, value, text, tol, limits, prefix, suffix, decimals, offset, group=None):
        """Lazily placed horizontal / vertical dimension between model points."""
        sh = self.sheet
        P1, P2 = self._P(*p1), self._P(*p2)
        if axis == "h":
            u, k = _v(1, 0), 0
        else:
            u, k = _v(0, 1), 1
        if P2[k] < P1[k]:
            P1, P2 = P2, P1
        L = P2[k] - P1[k]
        label = self._value_text(value, text, prefix, suffix, decimals)
        parts, tw = sh._dim_parts(label, tol, limits)
        lay = _dim_layout(L, tw, sh)
        lo, hi = P1[k] + lay["lo"], P1[k] + lay["hi"]

        def run():
            c = self._tier(side, lo, hi, offset) if group is None else group["c"]
            n = _v(0, 1) if axis == "h" else _v(1, 0)
            q1, q2 = P1.copy(), P2.copy()
            q1[1 - k] = c
            q2[1 - k] = c
            for P, q in ((P1, q1), (P2, q2)):
                self._ext_line(P, q)
            self._anno.extend(_dim_line(q1, q2, parts, tw, lay, sh))
        lay["outer"] = {"bottom": _v(0, -1), "top": _v(0, 1), "left": _v(-1, 0), "right": _v(1, 0)}[side]
        run.meta = (P1[k], P2[k], tw, lay)
        return run, (hi - lo), (lo, hi)

    def _ext_line(self, P, q):
        sh = self.sheet
        d = q - P
        L = math.hypot(*d)
        if L < sh.ext_gap + 0.5:
            return
        u = d / L
        self._out_anno.append(_E("line", "dim", p=(*(P + u * sh.ext_gap), *(q + u * sh.ext_over)), w=None))

    def dim_h(self, p1, p2, side="bottom", text=None, tol=None, limits=False, prefix="", suffix="",
              decimals=None, offset=None):
        """Horizontal dimension between model points ``p1`` and ``p2``, placed on ``side`` ('bottom' / 'top')
        of the view and stacked outwards automatically (``offset`` = fixed paper distance from the part).
        ``text``: replaces the value ('{}' = the value); ``tol``: 0.1 → ±0.1, (0.1, -0.05) → stacked deviations,
        with ``limits=True`` → limit dimensions; ``prefix`` e.g. 'Ø'."""
        side = {"below": "bottom", "above": "top"}.get(side, side)
        value = abs(p2[0] - p1[0])
        run, span, _ = self._linear(p1, p2, side, "h", value, text, tol, limits, prefix, suffix, decimals, offset)
        self._later(1, run, span)
        return self

    def dim_v(self, p1, p2, side="left", text=None, tol=None, limits=False, prefix="", suffix="",
              decimals=None, offset=None):
        """Vertical dimension between model points (``side`` 'left' / 'right'); options as ``dim_h``."""
        value = abs(p2[1] - p1[1])
        run, span, _ = self._linear(p1, p2, side, "v", value, text, tol, limits, prefix, suffix, decimals, offset)
        self._later(1, run, span)
        return self

    def dim_aligned(self, p1, p2, offset=10.0, text=None, tol=None, limits=False, prefix="", suffix="",
                    decimals=None):
        """Dimension parallel to p1 → p2, ``offset`` paper mm to the left of that direction (negative: right)."""
        sh = self.sheet
        P1, P2 = self._P(*p1), self._P(*p2)
        u = _unit(P2 - P1)
        n = _rot90(u)
        value = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        label = self._value_text(value, text, prefix, suffix, decimals)
        parts, tw = sh._dim_parts(label, tol, limits)
        L = math.hypot(*(P2 - P1))
        lay = _dim_layout(L, tw, sh)

        def run():
            q1, q2 = P1 + n * offset, P2 + n * offset
            self._ext_line(P1, q1)
            self._ext_line(P2, q2)
            self._out_anno.extend(_dim_line(q1, q2, parts, tw, lay, sh))
        self._later(1, run, L)
        return self

    def dim(self, p1, p2, side=None, **kw):
        """Linear dimension: horizontal when y is equal, vertical when x is equal, aligned otherwise."""
        if abs(p1[1] - p2[1]) < 1e-9:
            return self.dim_h(p1, p2, side or "bottom", **kw)
        if abs(p1[0] - p2[0]) < 1e-9:
            return self.dim_v(p1, p2, side or "left", **kw)
        return self.dim_aligned(p1, p2, **kw)

    def dim_chain(self, points, axis="h", side=None, offset=None, decimals=None):
        """Chain of consecutive dimensions through model ``points`` on one line (``axis`` 'h' or 'v')."""
        side = side or ("bottom" if axis == "h" else "left")
        side = {"below": "bottom", "above": "top"}.get(side, side)
        k = 0 if axis == "h" else 1
        pts = sorted(points, key=lambda p: p[k])
        group = {}
        runs = []
        lo, hi = 1e18, -1e18
        for a, b in zip(pts[:-1], pts[1:]):
            value = abs(b[k] - a[k])
            run, span, (l0, h0) = self._linear(a, b, side, axis, value, None, None, False, "", "", decimals,
                                               offset, group=group)
            runs.append(run)
            lo, hi = min(lo, l0), max(hi, h0)

        def run_all():
            group["c"] = self._tier(side, lo, hi, offset)
            second = _layout_chain([r.meta for r in runs], self.sheet)
            if second and offset is None:                    # keep the next tier clear of the second row
                self._tier(side, min(a for a, _ in second) - 1.0, max(b for _, b in second) + 1.0)
            for r in runs:
                r()
        self._later(1, run_all, hi - lo)
        return self

    def dim_baseline(self, points, axis="h", side=None, decimals=None):
        """Baseline (parallel) dimensions from the first point to every other point, stacked outwards."""
        side = side or ("bottom" if axis == "h" else "left")
        base = points[0]
        for p in points[1:]:
            if axis == "h":
                self.dim_h(base, p, side, decimals=decimals)
            else:
                self.dim_v(base, p, side, decimals=decimals)
        return self

    def dim_radius(self, cx, cy, r, angle=45.0, text=None, tol=None, inside=None, leader=6.0):
        """Radius dimension "R r": dimension line from the centre with one arrowhead on the arc at ``angle``,
        text on a horizontal shoulder outside. Small radii get the arrow from outside (``inside`` overrides)."""
        sh = self.sheet
        C = self._P(cx, cy)
        rp = r * self.s
        d = _dir(angle)
        A = C + d * rp
        label = text if text is not None else f"R{_fmt(r)}"
        parts, tw = sh._dim_parts(label, tol, False)
        inside = rp >= 7.0 if inside is None else inside
        AL, AW = sh.arrow
        sgn = 1.0 if d[0] >= 0 else -1.0

        def build(ext):
            a = []
            K = A + d * (leader + ext)
            S = K + _v(sgn * (tw + 2.0), 0)
            if inside:
                a.append(_E("poly", "dim", pts=[tuple(C), tuple(A - d * AL * 0.9)], closed=False, w=None))
                a.append(_arrow(A, d, AL, AW))
                a.append(_E("poly", "dim", pts=[tuple(A), tuple(K), tuple(S)], closed=False, w=None))
            else:
                a.append(_E("poly", "dim", pts=[tuple(S), tuple(K), tuple(A + d * AL * 0.9)], closed=False,
                            w=None))
                a.append(_arrow(A, -d, AL, AW))
            a += sh._place_parts(parts, tw, (K + S) / 2 + _v(0, sh.text_gap), _v(1, 0))
            return a, K, S
        self._shoulder(build, parts)
        return self

    def dim_diameter(self, cx, cy, d, angle=45.0, text=None, tol=None, limits=False, inside=None, leader=6.0):
        """Diameter dimension "Ød" across a circle at ``angle``: arrowheads on both sides, text on a shoulder
        outside. Small circles get one arrow from outside (``inside`` overrides)."""
        sh = self.sheet
        C = self._P(cx, cy)
        rp = d * self.s / 2
        u = _dir(angle)
        A2, A1 = C + u * rp, C - u * rp
        label = text if text is not None else f"Ø{_fmt(d)}"
        parts, tw = sh._dim_parts(label, tol, limits)
        AL, AW = sh.arrow
        inside = 2 * rp >= 2 * AL + 4.0 if inside is None else inside
        sgn = 1.0 if u[0] >= 0 else -1.0

        def build(ext):
            K = A2 + u * (leader + ext)
            S = K + _v(sgn * (tw + 2.0), 0)
            a = []
            if inside:
                a.append(_E("poly", "dim", pts=[tuple(A1 + u * AL * 0.9), tuple(A2 - u * AL * 0.9)],
                            closed=False, w=None))
                a.append(_arrow(A1, -u, AL, AW))
                a.append(_arrow(A2, u, AL, AW))
                a.append(_E("poly", "dim", pts=[tuple(A2), tuple(K), tuple(S)], closed=False, w=None))
            else:
                a.append(_E("poly", "dim", pts=[tuple(S), tuple(K), tuple(A2 + u * AL * 0.9)], closed=False,
                            w=None))
                a.append(_arrow(A2, -u, AL, AW))
                B = A1 - u * (AL + 2.0)
                a.append(_E("line", "dim", p=(*B, *(A1 - u * AL * 0.9)), w=None))
                a.append(_arrow(A1, u, AL, AW))
            a += sh._place_parts(parts, tw, (K + S) / 2 + _v(0, sh.text_gap), _v(1, 0))
            return a, K, S
        self._shoulder(build, parts)
        return self

    def _shoulder(self, build, parts):
        """Place a radial dimension after the linear ones: lengthen its leader until the value on the shoulder is
        clear of the drawing (``build(ext)`` -> (entities, knee, shoulder end))."""
        sh = self.sheet
        top = max(dy + hz for _, dy, _, hz in parts) + sh.text_gap + 0.4
        bot = min(dy - hz * 0.25 for _, dy, _, hz in parts) + sh.text_gap

        def run():
            P = self._obstacle_points()
            best = None
            for ext in np.arange(0.0, 30.01, 1.5):
                ents, K, S = build(ext)
                x0, x1 = sorted((K[0], S[0]))
                x0, x1 = x0 + 0.8, x1 + 0.4
                y0, y1 = K[1] + max(bot, 0.3), K[1] + top
                hits = int(np.count_nonzero((P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)))
                if best is None or hits < best[0]:
                    best = (hits, ents)
                if hits == 0:
                    break
            self._anno.extend(best[1])
        self._later(2, run)

    def dim_angle(self, cx, cy, start, end, r=None, text=None, ext_from=None, decimals=None):
        """Angular dimension at vertex (cx, cy) from direction ``start`` to ``end`` (degrees, counter-
        clockwise): arc of paper radius ``r`` with arrowheads, value in degrees. ``ext_from`` (model radius)
        draws extension lines along both rays from that radius to the arc."""
        sh = self.sheet
        C = self._P(cx, cy)
        a0, a1 = float(start), float(end)
        while a1 <= a0:
            a1 += 360
        r = r or 20.0
        val = a1 - a0
        label = text if text is not None else f"{_fmt(val, decimals)}°"
        parts, tw = sh._dim_parts(label, None, False)
        AL, AW = sh.arrow
        a = []
        span = math.radians(val) * r
        inside = span >= 2 * AL + 1.0
        if ext_from is not None:
            for ang in (a0, a1):
                d = _dir(ang)
                a.append(_E("line", "dim", p=(*(C + d * (ext_from * self.s + sh.ext_gap)),
                                               *(C + d * (r + sh.ext_over))), w=None))
        da = math.degrees(AL * 0.9 / r)
        if inside:
            a.append(_E("arc", "dim", c=tuple(C), r=r, a=(a0 + da, a1 - da), w=None))
            a.append(_arrow(C + _dir(a0) * r, -_rot90(_dir(a0 + math.degrees(AL / r) * 0.5)), AL, AW))
            a.append(_arrow(C + _dir(a1) * r, _rot90(_dir(a1 - math.degrees(AL / r) * 0.5)), AL, AW))
        else:
            ex = math.degrees((AL + 2.5) / r)
            a.append(_E("arc", "dim", c=tuple(C), r=r, a=(a0 - ex, a1 + ex), w=None))
            a.append(_arrow(C + _dir(a0) * r, _rot90(_dir(a0 - math.degrees(AL / r) * 0.5)), AL, AW))
            a.append(_arrow(C + _dir(a1) * r, -_rot90(_dir(a1 + math.degrees(AL / r) * 0.5)), AL, AW))
        m = _dir((a0 + a1) / 2)
        h = sh.text_h
        ext = abs(m[0]) * tw / 2 + abs(m[1]) * h / 2
        tc = C + m * (r + 1.5 + ext)
        a += sh._place_parts(parts, tw, tc - _v(0, h / 2), _v(1, 0))
        self._anno += a
        return self

    def dim_chamfer(self, x, y, size, angle=45.0, direction=45.0, length=10.0, text=None):
        """Chamfer note "size x angle°" on a leader from model point (x, y) towards ``direction``."""
        label = text or f"{_fmt(size)} x {_fmt(angle)}°"
        d = _dir(direction) * length
        return self.leader(x, y, label, dx=d[0], dy=d[1])

    def callout(self, x, y, text, angle=45.0, length=12.0, r=0.0):
        """Leader note pointing at a circle edge: arrow at radius ``r`` from (x, y) in ``angle``."""
        p = _v(x, y) + _dir(angle) * r
        d = _dir(angle) * length
        return self.leader(p[0], p[1], text, dx=d[0], dy=d[1])

    def hole_callout(self, cx, cy, d, n=1, thru=True, depth=None, thread=None, angle=45.0, length=12.0,
                     text=None, cbore=None):
        """Hole note on a leader, e.g. "6x Ø9 THRU", "4x M6 - 6H THRU", "Ø6 x 10 DEEP";
        ``cbore=(d2, depth2)`` adds a counterbore line."""
        if text is None:
            head = f"{n}x " if n > 1 else ""
            body = (thread if isinstance(thread, str) else f"M{_fmt(d)}") if thread else f"Ø{_fmt(d)}"
            tail = f" x {_fmt(depth)} DEEP" if depth else (" THRU" if thru else "")
            text = head + body + tail
            if cbore:
                text += f"\n⌴ Ø{_fmt(cbore[0])} x {_fmt(cbore[1])}"
        return self.callout(cx, cy, text, angle, length, r=d / 2)

    # ---- architectural plans (model mm, typically scale 1:50 / 1:100)
    def wall(self, points, thickness=200.0, closed=False):
        """Wall along the centre-line ``points`` (model mm) with ``thickness``; mitred corners, clean joins with
        other walls, solid poche. Doors and windows cut openings into it. Returns the view."""
        pts = [_v(*p) for p in points]
        self._walls.append(dict(pts=pts, t=float(thickness), closed=bool(closed)))
        if len(self._walls) == 1:
            self._later(0, self._build_walls)
        return self

    def _nearest_wall(self, x, y):
        best = None
        P = _v(x, y)
        for wi, w in enumerate(self._walls):
            pts = w["pts"]
            n = len(pts)
            segs = range(n if w["closed"] else n - 1)
            for si in segs:
                a, b = pts[si], pts[(si + 1) % n]
                d = b - a
                L = math.hypot(*d)
                t = float(np.clip((P - a) @ d / (L * L), 0, 1))
                q = a + d * t
                dist = math.hypot(*(P - q))
                if best is None or dist < best[0]:
                    best = (dist, wi, si, t * L, L)
        if best is None:
            raise ValueError("add walls before doors / windows")
        return best[1:]

    def _opening(self, x, y, width, kind, **kw):
        wi, si, s, L = self._nearest_wall(x, y)
        s = float(np.clip(s, width / 2, L - width / 2))
        self._openings.append(dict(w=wi, s=si, a=s - width / 2, b=s + width / 2, kind=kind, width=width, **kw))
        return self

    def door(self, x, y, width=900.0, toward=None, hinge="left", double=False, style="swing"):
        """Door in the nearest wall, centred at the projection of (x, y). ``style``: 'swing' (leaf and arc;
        ``toward``: a model point inside the room it swings into, ``hinge``: 'left' / 'right' as seen when
        walking through towards that room), 'double' (a pair of leaves, same as ``double=True``) or 'sliding'
        (two overlapping panels in the wall, no arc: balcony doors, wardrobes, wide openings)."""
        if style not in ("swing", "double", "sliding"):
            _check_choice("door", "style", style, ["swing", "double", "sliding"])
        return self._opening(x, y, width, "door", toward=toward, hinge=hinge, double=bool(double) or style == "double",
                             sliding=style == "sliding")

    def sliding_door(self, x, y, width=1600.0):
        """Sliding door in the nearest wall: two overlapping panels, no swing arc. The door for balconies and
        any opening wider than about 1.2 m."""
        return self.door(x, y, width, style="sliding")

    def window(self, x, y, width=1200.0, sill=True):
        """Window in the nearest wall, centred at the projection of (x, y)."""
        return self._opening(x, y, width, "window", sill=sill)

    def opening(self, x, y, width=900.0):
        """Plain opening (no door leaf) in the nearest wall."""
        return self._opening(x, y, width, "opening")

    def _wall_quads(self, cuts=True):
        """Convex pieces (clockwise point lists) of every wall, split at openings when ``cuts``."""
        quads = []
        for wi, w in enumerate(self._walls):
            pts, t, closed = w["pts"], w["t"], w["closed"]
            n = len(pts)
            nseg = n if closed else n - 1
            half = t / 2
            for si in range(nseg):
                a, b = pts[si], pts[(si + 1) % n]
                u = _unit(b - a)
                nn = _rot90(u)
                L = math.hypot(*(b - a))

                def miter(i_vertex, at_start):
                    has_prev = closed or i_vertex > 0
                    has_next = closed or i_vertex < n - 1
                    v = pts[i_vertex % n]
                    if at_start and has_prev:
                        p_prev = pts[(i_vertex - 1) % n]
                        u2 = _unit(v - p_prev)
                    elif not at_start and has_next:
                        u2 = _unit(pts[(i_vertex + 1) % n] - v)
                    else:
                        return None
                    m = _unit(_rot90(u) + _rot90(u2))
                    cosv = m @ _rot90(u)
                    if abs(cosv) < 0.2:
                        return None
                    return m * (half / cosv)

                ms, me = miter(si, True), miter(si + 1, False)
                ls = a + (ms if ms is not None else nn * half)
                rs = a - (ms if ms is not None else nn * half)
                le = b + (me if me is not None else nn * half)
                re = b - (me if me is not None else nn * half)
                cut = sorted((o["a"], o["b"]) for o in self._openings if o["w"] == wi and o["s"] == si)                     if cuts else []
                pieces = []
                cur = (ls, rs)
                for (ca, cb) in cut:
                    pa_l, pa_r = a + u * ca + nn * half, a + u * ca - nn * half
                    pieces.append([cur[0], pa_l, pa_r, cur[1]])
                    cur = (a + u * cb + nn * half, a + u * cb - nn * half)
                pieces.append([cur[0], le, re, cur[1]])
                for q in pieces:
                    q = [np.asarray(p, float) for p in q]
                    if _area(q) > 0:
                        q = q[::-1]
                    if abs(_area(q)) > 1e-6:
                        quads.append(q)
        return quads

    def _build_walls(self):
        quads = self._wall_quads()
        if not quads:
            return
        poly = [[tuple(p) for p in q] for q in quads]
        self._out_geom.append(_E("solid", "wall", rings=poly, fill="poche", o="wall", ov=self.name))
        for s in _union_outline(quads):
            self._out_geom.append(_E("line", "wall", p=s, w=None, o="wall", ov=self.name))
        for oi, o in enumerate(self._openings):
            n0 = len(self._out_geom)
            self._opening_symbol(o)
            _tag(self._out_geom[n0:], o=o["kind"], ov=self.name, oi=oi, sub="sliding" if o.get("sliding") else (
                "double" if o.get("double") else "swing"))

    def _opening_symbol(self, o):
        w = self._walls[o["w"]]
        pts, t = w["pts"], w["t"]
        n = len(pts)
        a, b = pts[o["s"]], pts[(o["s"] + 1) % n]
        u = _unit(b - a)
        nn = _rot90(u)
        half = t / 2
        A, B = a + u * o["a"], a + u * o["b"]
        g = self._out_geom
        if o["kind"] == "window":
            for off in (half, -half):
                g.append(_E("line", "thin", p=(*(A + nn * off), *(B + nn * off)), w=None))
            for off in (t * 0.09, -t * 0.09):
                g.append(_E("line", "visible", p=(*(A + nn * off), *(B + nn * off)), w=self.sheet.thin * 1.2))
            if o.get("sill"):
                for P in (A, B):
                    g.append(_E("line", "thin", p=(*(P + nn * half), *(P - nn * half)), w=None))
        elif o["kind"] == "door":
            side = 1.0
            if o.get("toward") is not None:
                side = 1.0 if (_v(*o["toward"]) - (A + B) / 2) @ nn >= 0 else -1.0
            if o.get("sliding"):
                M = (A + B) / 2
                W = o["width"]
                lt_ = self._paper(0.9)
                for off, sgn in ((t * 0.14, -1.0), (-t * 0.14, 1.0)):
                    c = M + nn * off + u * (sgn * W * 0.225)
                    r0, r1 = c - u * W * 0.275, c + u * W * 0.275
                    panel = [r0 - nn * lt_ / 2, r1 - nn * lt_ / 2, r1 + nn * lt_ / 2, r0 + nn * lt_ / 2]
                    g.append(_E("poly", "visible", pts=[tuple(p_) for p_ in panel], closed=True,
                                w=self.sheet.thin * 1.4))
                for P_ in (A, B):
                    g.append(_E("line", "thin", p=(*(P_ + nn * half), *(P_ - nn * half)), w=None))
                return
            N = nn * side                                       # into the room
            right = _rot90(N) * -1                              # walking along N, right-hand side
            hinge_left = o.get("hinge", "left") == "left"
            face = N * half
            W = o["width"]
            M = (A + B) / 2
            leaves = [(M - right * W / 2, right, W)] if hinge_left else [(M + right * W / 2, -right, W)]
            if o.get("double"):
                leaves = [(M - right * W / 2, right, W / 2), (M + right * W / 2, -right, W / 2)]
            lt_ = self._paper(0.9)
            for H, along, lw in leaves:
                Hp = H + face
                tip = Hp + N * lw
                leaf = [Hp, Hp + along * lt_, tip + along * lt_, tip]
                g.append(_E("poly", "visible", pts=[tuple(p) for p in leaf], closed=True, w=self.sheet.thin * 1.4))
                a0 = _ang(N)
                a1 = _ang(along)
                lo, hi = (a0, a1) if ((a1 - a0) % 360) <= 180 else (a1, a0)
                g.append(_E("arc", "thin", c=tuple(Hp), r=lw, a=(lo, lo + ((hi - lo) % 360)), w=None))

    def stairs(self, x, y, width, length, steps=None, axis="y", up=True, label=None):
        """Straight flight with lower-left corner (x, y): ``width`` across, ``length`` along ``axis`` ('x'/'y'),
        treads, walking line with arrow and ``label``."""
        steps = steps or max(3, int(round(length / 280.0)))
        if axis == "y":
            self.rect(x, y, width, length, "visible")
            for i in range(1, steps):
                yy = y + length * i / steps
                self.line(x, yy, x + width, yy, "thin")
            p0, p1 = _v(x + width / 2, y + length * 0.5 / steps), _v(x + width / 2, y + length * (1 - 0.3 / steps))
        else:
            self.rect(x, y, length, width, "visible")
            for i in range(1, steps):
                xx = x + length * i / steps
                self.line(xx, y, xx, y + width, "thin")
            p0, p1 = _v(x + length * 0.5 / steps, y + width / 2), _v(x + length * (1 - 0.3 / steps), y + width / 2)
        if not up:
            p0, p1 = p1, p0
        P0, P1 = self._P(*p0), self._P(*p1)
        d = _unit(P1 - P0)
        self._anno.append(_E("line", "dim", p=(*P0, *(P1 - d * self.sheet.arrow[0] * 0.9)), w=None))
        self._anno.append(_E("dot", "dim", c=tuple(P0), r=0.8))
        self._anno.append(_arrow(P1, d, *self.sheet.arrow))
        label = ("UP" if up else "DN") if label is None else label
        if label:
            rot = 0.0 if abs(d[0]) > 0.5 else 90.0
            along = _dir(rot)
            ha = "l" if d @ along > 0 else "r"
            q = P0 + d * 2.0 + _rot90(along) * 1.2
            self._anno += self.sheet._text(q, label, self.sheet.text_h * 0.8, ha, "b", rot=rot)
        return self

    def room(self, name, x, y, area="auto", h=None, clear=True, fill=None, tint=None):
        """Room label centred at (x, y): ``name`` plus the floor area in m². ``area="auto"`` measures the room
        enclosed by the walls around (x, y) to their inner faces (doorways count as closed); or pass a number
        in m², a (w, d) size in mm, a point list / Profile in mm, or None for no area line. ``clear=True`` nudges
        the label to the nearest spot free of furniture and other lines. The area is kept in
        ``view.rooms[name]``; ``sh.room_schedule(view)`` lists them in a table.

        ``fill`` (alias ``tint``) colours the floor under walls, furniture and labels: a colour, a palette name
        (pastel warm cool grey: the pale colour for this room's type, read from its name), True (the plan's
        palette from ``colors()``, else pastel) or False (no tint even when the plan has a palette). Default:
        the plan's palette when ``colors(floors=...)`` was set, else a plain white floor."""
        sh = self.sheet
        h = h or sh.text_h * 1.15
        C = self._P(x, y)
        if fill is None:
            fill = tint
        if isinstance(fill, str) and fill not in FLOOR_PALETTES:
            try:
                _rgb(fill)
            except ValueError:
                raise _unknown_name("room fill", fill, list(FLOOR_PALETTES),
                                    "Use a colour like '#dbe6f4', a palette name, True or False.") from None
        floor_pts = None
        if area is not None and not isinstance(area, (int, float, str)):
            if isinstance(area, Profile):
                floor_pts = area.points()
            elif not (len(area) == 2 and isinstance(area[0], (int, float))):
                floor_pts = [tuple(map(float, q)) for q in area]
        if isinstance(area, str):
            area = self.room_area(x, y) if area == "auto" else None
        elif area is not None and not isinstance(area, (int, float)):
            if isinstance(area, Profile):
                area = area.area() / 1e6
            elif len(area) == 2 and isinstance(area[0], (int, float)):
                area = area[0] * area[1] / 1e6
            else:
                area = abs(_area(area)) / 1e6
        self.rooms[name] = area

        gid = self._next_oid()
        floor = {}

        def paint_floor():                                   # resolved at layout: colors() may come later
            col = _floor_color(name, fill, self._floors)
            if col is None:
                return
            ring = self._room_ring(x, y)
            if ring is None and floor_pts is not None:
                ring = floor_pts
            if ring is None:
                return
            R = [tuple(self._P(*q)) for q in np.asarray(ring, float)]
            floor["c"] = col
            self._anno.append(_E("solid", "hatch", rings=[R], fill="floor", color=col, o="room", on=name,
                                 ov=self.name, gid=gid))
        self._later(0, paint_floor)

        def build(o, f=1.0):
            out = sh._text(C + o + _v(0, 0.6 * f), name, h * f, "m", "b", bold=True, halo="box")
            if area is not None:
                out += sh._text(C + o - _v(0, 0.8 * f), f"{area:.1f} m²", h * 0.8 * f, "m", "t", halo="box")
            if floor.get("c"):
                for e in out:
                    e["bg"] = floor["c"]
            return _tag(out, o="room", on=name, ov=self.name, gid=gid, mv="room")

        if not clear:
            self._later(3, lambda: self._anno.extend(build(_v(0, 0))))
            return self

        def run():                                           # move the label to free floor inside the room
            P = self._obstacle_points()
            ring = self._room_ring(x, y)
            E = _edges_of([ring]) if ring is not None else None
            furn = []                                        # furniture outlines count as filled
            for e in self._out_geom:
                if e["lt"] == "furniture" and e["t"] in ("poly", "circle"):
                    Q = _entity_pts(e) * self.s + np.array([self._ox, self._oy])
                    furn.append((*Q.min(0), *Q.max(0)))
            F = np.array(furn) if furn else np.zeros((0, 4))
            tried = []
            for f in (1.0, 0.85, 0.72):                      # shrink the label until it finds clear floor
                boxes = [_text_corners(e, sh, pad=1.2, desc=0.15) for e in build(_v(0, 0), f)]
                B = np.concatenate(boxes)
                bx0, by0 = B.min(0) - C
                bx1, by1 = B.max(0) - C
                if ring is not None:
                    R = np.asarray(ring, float) * self.s + np.array([self._ox, self._oy])
                    gx = np.arange(R[:, 0].min() - bx0, R[:, 0].max() - bx1 + 1e-9, 1.0) - C[0]
                    gy = np.arange(R[:, 1].min() - by0, R[:, 1].max() - by1 + 1e-9, 1.0) - C[1]
                else:
                    gx, gy = np.arange(-24, 24.1, 1.0), np.arange(-18, 18.1, 1.0)
                if not len(gx) or not len(gy):
                    gx, gy = np.array([0.0]), np.array([0.0])
                DX, DY = [a.ravel() for a in np.meshgrid(gx, gy)]
                if E is not None:                            # the whole box stays on this room's floor
                    ok = np.ones(len(DX), bool)
                    for fx, fy in ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1), ((bx0 + bx1) / 2, by0),
                                   ((bx0 + bx1) / 2, by1)):
                        Q = (np.stack([C[0] + DX + fx, C[1] + DY + fy], 1) - (self._ox, self._oy)) / self.s
                        ok &= _inside(Q, E)
                    if ok.any():
                        DX, DY = DX[ok], DY[ok]
                X0, Y0, X1, Y1 = C[0] + DX + bx0, C[1] + DY + by0, C[0] + DX + bx1, C[1] + DY + by1
                hits = np.zeros(len(DX))
                for k in range(0, len(P), 4096):
                    p = P[k:k + 4096]
                    hits += ((p[None, :, 0] > X0[:, None]) & (p[None, :, 0] < X1[:, None]) &
                             (p[None, :, 1] > Y0[:, None]) & (p[None, :, 1] < Y1[:, None])).sum(1)
                for fb in F:
                    ox_ = np.clip(np.minimum(X1, fb[2]) - np.maximum(X0, fb[0]), 0, None)
                    oy_ = np.clip(np.minimum(Y1, fb[3]) - np.maximum(Y0, fb[1]), 0, None)
                    hits += ox_ * oy_ * 2.0
                score = hits * 1e3 + DX ** 2 * 0.6 + DY ** 2
                i = int(np.argmin(score))
                tried.append((hits[i], f, DX[i], DY[i]))
                if hits[i] == 0:
                    break
            h0, f, dx, dy = tried[-1] if tried[-1][0] == 0 else min(tried, key=lambda t: (t[0], -t[1]))
            self._anno.extend(build(_v(dx, dy), f))
        self._later(3, run)
        return self

    def room_area(self, x, y):
        """Floor area in m² of the room enclosed by walls around model point (x, y), measured to the inner wall
        faces with doorways closed; None when (x, y) is not enclosed."""
        ring = self._room_ring(x, y)
        return None if ring is None else abs(_area(ring)) / 1e6

    def _room_ring(self, x, y):
        if not self._walls:
            return None
        key = tuple((len(w["pts"]), w["t"]) for w in self._walls)
        if getattr(self, "_loops_key", None) != key:
            self._loops = [r for r in _loops(_union_outline(self._wall_quads(cuts=False))) if len(r) >= 3]
            self._loops_key = key
        P = np.array([[float(x), float(y)]])
        hits = [r for r in self._loops if _inside(P, _edges_of([r]))[0]]
        if len(hits) < 2:                                            # outside, or inside the outer outline only
            return None
        for q in self._wall_quads(cuts=False):                       # inside a wall: not a room
            if _inside(P, _edges_of([np.asarray(q, float)]))[0]:
                return None
        return min(hits, key=lambda r: abs(_area(r)))

    def furniture(self, kind, x, y, w=None, d=None, angle=0.0, **opts):
        """Furniture symbol centred on model point (x, y), rotated ``angle`` degrees; local width ``w`` along x,
        depth ``d`` along y with the back / head at +y. Kinds: bed single_bed sofa armchair table chair toilet
        basin shower bathtub counter sink hob fridge wardrobe desk. ``table`` takes ``seats=4``.
        ``color`` = line colour, ``fill`` = colour inside the outlines (default: the plan's ``colors()``,
        else black lines on white)."""
        if kind not in _FURN:
            raise _unknown_name("furniture", kind, FURNITURE)
        extra = _signature_options(_FURN[kind][1])[2:]
        opts = _reject_unknown(f"furniture('{kind}')", opts, ["w", "d", "angle", "color", "fill"] + extra,
                               {"width": "w", "depth": "d", "rotation": "angle", "rotate": "angle",
                                "colour": "color", "stroke": "color", "line_color": "color"})
        w, d, angle = opts.pop("w", w), opts.pop("d", d), opts.pop("angle", angle)
        color, ffill = _color_hex(opts.pop("color", None), "color"), _color_hex(opts.pop("fill", None), "fill")
        dw, dd = _FURN[kind][0]
        w = float(w or dw)
        d = float(d or dd)
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        g0 = len(self._geom)

        def T(p):
            return (x + p[0] * ca - p[1] * sa, y + p[0] * sa + p[1] * ca)

        for item in _FURN[kind][1](w, d, **opts):
            k = item[0]
            if k == "poly":
                self.polyline([T(p) for p in item[1]], "furniture", closed=item[2])
            elif k == "line":
                self.line(*T(item[1]), *T(item[2]), "furniture")
            elif k == "circle":
                c = T(item[1])
                self.circle(c[0], c[1], item[2], "furniture")
            elif k == "arc":
                c = T(item[1])
                self.arc(c[0], c[1], item[2], item[3] + angle, item[4] + angle, "furniture")
        _tag(self._geom[g0:], o="furniture", on=kind, ov=self.name, oid=self._next_oid())
        for e in self._geom[g0:]:
            if color:
                e["color"] = color
            if ffill and (e["t"] == "circle" or (e["t"] == "poly" and e.get("closed"))):
                e["fill"] = ffill
            e["own_col"] = bool(color) or bool(ffill)
        return self

    def colors(self, floors="pastel", furniture=None, furniture_fill=None):
        """Colour the whole plan: ``floors`` = palette for every room's floor tint (pastel warm cool grey, a
        dict {room type or name: colour} with types living kitchen dining bedroom bath hall balcony office
        storage other, or None); ``furniture`` = furniture line colour; ``furniture_fill`` = colour inside
        furniture outlines (default: a pale mix of ``furniture``). A room's own ``fill`` wins. Returns the view."""
        if isinstance(floors, str) and floors not in FLOOR_PALETTES:
            raise _unknown_name("floor palette", floors, list(FLOOR_PALETTES))
        if isinstance(floors, dict):
            floors = {str(k).lower(): _color_hex(v, "floors") for k, v in floors.items()}
        self._floors = floors
        self._furn_col = _color_hex(furniture, "furniture")
        ff = _color_hex(furniture_fill, "furniture_fill")
        if ff is None and self._furn_col is not None:
            c = _rgb(self._furn_col)
            ff = _hex(tuple(0.86 + 0.14 * float(v) for v in c))
        self._furn_fill = ff
        return self

    def gridlines(self, xs=(), ys=(), x_labels=None, y_labels=None, bubble=8.0, ends="both"):
        """Structural grid: centre lines at model ``xs`` (vertical) and ``ys`` (horizontal) with labelled
        bubbles beyond the dimension strings (labels default 1 2 3 … and A B C …)."""
        x_labels = list(x_labels) if x_labels else [str(i + 1) for i in range(len(xs))]
        y_labels = list(y_labels) if y_labels else [chr(65 + i) for i in range(len(ys))]

        def run():
            sh = self.sheet
            bb = self._part_bbox()
            ex = self._extent
            r = bubble / 2
            bot = min(ex.get("bottom", bb[1] - 4), bb[1] - 4) - 6
            top = max(ex.get("top", bb[3] + 4), bb[3] + 4) + 6
            lef = min(ex.get("left", bb[0] - 4), bb[0] - 4) - 6
            rig = max(ex.get("right", bb[2] + 4), bb[2] + 4) + 6
            a = self._out_anno
            boxes = [_text_corners(e, sh, pad=0.8) for e in a if e["t"] == "text" and e["s"].strip()]
            boxes = [(*c.min(0), *c.max(0)) for c in boxes]

            def gapped(p, q):                                # break the grid line around dimension values
                k = 1 if p[0] == q[0] else 0
                c = p[1 - k]
                lo, hi = sorted((p[k], q[k]))
                cuts = sorted((b[k], b[k + 2]) for b in boxes if b[1 - k] < c < b[3 - k] and b[k + 2] > lo
                              and b[k] < hi)
                cur = lo
                for c0, c1 in cuts + [(hi, hi)]:
                    if c0 - cur > 1.0:
                        pt0, pt1 = [0.0, 0.0], [0.0, 0.0]
                        pt0[k], pt1[k], pt0[1 - k], pt1[1 - k] = cur, c0, c, c
                        a.append(_E("line", "center", p=(*pt0, *pt1), w=None))
                    cur = max(cur, c1)

            for xv, lab in zip(xs, x_labels):
                X = self._to_paper(xv, 0)[0]
                y0 = bot if ends in ("both", "start") else bb[1] - 4
                y1 = top if ends in ("both", "end") else bb[3] + 4
                gapped((X, y0), (X, y1))
                for yy, sg in ((y0, -1), (y1, 1)):
                    if (sg < 0 and ends in ("both", "start")) or (sg > 0 and ends in ("both", "end")):
                        c = _v(X, yy + sg * r)
                        a.append(_E("circle", "thin", c=tuple(c), r=r, w=sh.thin, fill="paper"))
                        a.extend(sh._text(c, lab, sh.text_h * 1.1, "m", "m", bold=True))
            for yv, lab in zip(ys, y_labels):
                Y = self._to_paper(0, yv)[1]
                x0 = lef if ends in ("both", "start") else bb[0] - 4
                x1 = rig if ends in ("both", "end") else bb[2] + 4
                gapped((x0, Y), (x1, Y))
                for xx, sg in ((x0, -1), (x1, 1)):
                    if (sg < 0 and ends in ("both", "start")) or (sg > 0 and ends in ("both", "end")):
                        c = _v(xx + sg * r, Y)
                        a.append(_E("circle", "thin", c=tuple(c), r=r, w=sh.thin, fill="paper"))
                        a.extend(sh._text(c, lab, sh.text_h * 1.1, "m", "m", bold=True))
        self._later(2, run)
        return self

    # ---- isometric pictorial
    def _isoscene(self):
        if self._iso is None:
            self._iso = _IsoScene()
            self._later(0, self._build_iso)
        return self._iso

    def iso_extrude(self, profile, depth, plane="xy", at=0.0, holes=()):
        """Prism for an isometric view: a 2-D ``profile`` (Profile or points) in ``plane`` ('xy' extrudes along
        z, 'xz' along y, 'yz' along x) from ``at`` to ``at + depth``; ``holes`` = through holes in the profile
        plane as (a, b, r) circles or point lists."""
        rings = [profile.points() if isinstance(profile, Profile) else np.asarray(profile, float)]
        for h in holes:
            rings += _rings_of(h)
        self._isoscene().prism(rings, plane, at, at + depth)
        return self

    def iso_box(self, x, y, z, dx, dy, dz):
        """Rectangular block from corner (x, y, z) with sizes dx, dy, dz."""
        ring = np.array([[x, y], [x + dx, y], [x + dx, y + dy], [x, y + dy]], float)
        self._isoscene().prism([ring], "xy", z, z + dz)
        return self

    def iso_cylinder(self, x, y, z, r, h, axis="z"):
        """Cylinder of radius ``r`` and length ``h`` starting at (x, y, z) along ``axis``."""
        plane = {"z": "xy", "y": "xz", "x": "yz"}[axis]
        a, b, w = {"z": (x, y, z), "y": (x, z, y), "x": (y, z, x)}[axis]
        self._isoscene().prism([_circle_ring(a, b, r, 96)], plane, w, w + h)
        return self

    def iso_hole(self, x, y, z, d, axis="z"):
        """Through hole of diameter ``d`` along ``axis`` through point (x, y, z), cut into every face of the
        solids added so far that it pierces."""
        self._isoscene().hole(_v(0, 0), (x, y, z), axis, _circle_ring(0, 0, d / 2, 96))
        return self

    def iso_slot(self, x1, y1, z1, x2, y2, z2, width, axis="z"):
        """Through slot (obround) between two arc centres, along ``axis``."""
        p1, p2 = np.array([x1, y1, z1], float), np.array([x2, y2, z2], float)
        ax = "xyz".index(axis)
        keep = [i for i in range(3) if i != ax]
        ring = _slot_ring(p1[keep[0]], p1[keep[1]], p2[keep[0]], p2[keep[1]], width)
        c = (p1 + p2) / 2
        ring = ring - np.array([c[keep[0]], c[keep[1]]])
        self._isoscene().hole(_v(0, 0), tuple(c), axis, ring)
        return self

    def iso_line(self, p, q, kind="visible"):
        """Line between 3-D points (drawn without hidden-line removal)."""
        a, b = iso(*p), iso(*q)
        return self.line(a[0], a[1], b[0], b[1], kind)

    def _build_iso(self):
        step = self._paper(0.35)
        for run in self._iso.visible_runs(step):
            run = _simplify(run, step * 0.02)
            if len(run) >= 2:
                self._out_geom.append(_E("poly", "visible", pts=[tuple(p) for p in run], closed=False, w=None))

    # ---- finalize
    def _finalize(self):
        self._out_geom = list(self._geom)
        self._out_anno = list(self._anno)
        saved_anno = self._anno
        self._anno = self._out_anno                          # lazy calls append to the output lists
        self._tiers, self._extent, self._bb = {}, {}, None
        try:
            for prio, key, _, fn, obj in sorted(self._lazy, key=lambda t: (t[0], t[1], t[2])):
                if prio >= 1 and self._bb is None:
                    self._part_bbox()
                n0, g0 = len(self._out_anno), len(self._out_geom)
                fn()
                if obj:
                    _tag(self._out_anno[n0:] + self._out_geom[g0:], o=obj, ov=self.name)
            if self.label_text:
                self._place_label()
            if self._furn_col or self._furn_fill:
                out = []
                for e in self._out_geom:
                    if e["lt"] == "furniture" and not e.get("own_col"):
                        e = dict(e)
                        if self._furn_col:
                            e["color"] = self._furn_col
                        if self._furn_fill and (e["t"] == "circle" or (e["t"] == "poly" and e.get("closed"))):
                            e["fill"] = self._furn_fill
                    out.append(e)
                self._out_geom = out
        finally:
            self._anno = saved_anno
        return self._out_geom, self._out_anno

    def _place_label(self):
        sh = self.sheet
        pts = [_entity_pts(e) * self.s + np.array([self._ox, self._oy]) for e in self._out_geom]
        pts += [_entity_pts(e) for e in self._out_anno if e["t"] != "text"]
        if not pts:
            return
        P = np.concatenate(pts)
        x = (P[:, 0].min() + P[:, 0].max()) / 2
        bb = self._part_bbox()
        x = (bb[0] + bb[2]) / 2
        y = P[:, 1].min() - 6.0
        t = self.label_text
        if abs(self.s - sh.s) > 1e-9:
            t += f" ({_scale_text(self.s)})"
        self._out_anno += _tag(sh._text(_v(x, y), t, sh.text_h * 1.4, "m", "t", bold=True), mv="title",
                               gid=self._next_oid(), o="title", on=t, ov=self.name)


def _simplify(P, tol):
    """Drop polyline points that lie within ``tol`` of the chord between their kept neighbours."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    keep = [0]
    i = 0
    n = len(P)
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            a, b = P[i], P[j + 1]
            d = b - a
            L = math.hypot(*d)
            seg = P[i + 1:j + 1]
            if L < 1e-12:
                break
            dist = np.abs((seg[:, 0] - a[0]) * d[1] - (seg[:, 1] - a[1]) * d[0]) / L
            if dist.max() > tol:
                break
            j += 1
        keep.append(j)
        i = j
    return P[keep]


def _anchor2(a):
    a = str(a).lower()
    m = {"center": "mm", "left": "lm", "right": "rm"}
    a = m.get(a, a)
    return a[0], a[1]


def _entity_pts(e):
    t = e["t"]
    if t == "line":
        x1, y1, x2, y2 = e["p"]
        return np.array([[x1, y1], [x2, y2]], float)
    if t == "poly":
        return np.asarray(e["pts"], float)
    if t in ("circle", "dot"):
        c, r = e["c"], e["r"]
        return np.array([[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]], float)
    if t == "arc":
        return _arc_pts(e["c"], e["r"], *e["a"], step_deg=10)
    if t == "solid":
        if "rings" in e:
            return np.concatenate([np.asarray(r, float) for r in e["rings"]])
        return np.asarray(e["pts"], float)
    if t == "text":
        return np.asarray([e["p"]], float)
    return np.zeros((0, 2))


def _text_corners(e, sheet, pad=0.0, desc=0.25):
    """Corners (4, 2) of a text entity's box in paper mm (``desc`` x height below the baseline)."""
    h = e["h"]
    w = sheet._tw(e["s"], h, e.get("bold", False))
    x0 = {"l": 0.0, "m": -w / 2, "r": -w}[e["ha"]]
    y0 = {"b": 0.0, "m": -h / 2, "t": -h}[e["va"]]
    loc = np.array([[x0 - pad, y0 - desc * h - pad], [x0 + w + pad, y0 - desc * h - pad],
                    [x0 + w + pad, y0 + h + pad], [x0 - pad, y0 + h + pad]])
    a = math.radians(e.get("rot", 0.0))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return loc @ R.T + np.asarray(e["p"], float)


def _ents_bbox(ents, sheet):
    """(x0, y0, x1, y1) paper bbox of entities, text boxes included; None when empty."""
    pts = []
    for e in ents:
        if e["t"] == "text":
            if e["s"].strip():
                pts.append(_text_corners(e, sheet))
        else:
            P = _entity_pts(e)
            if len(P):
                pts.append(P)
    if not pts:
        return None
    P = np.concatenate(pts)
    return (float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max()))


def _shift_entity(e, dx, dy):
    """Copy of a paper entity moved by (dx, dy)."""
    n = dict(e)
    t = e["t"]
    if t == "line":
        x1, y1, x2, y2 = e["p"]
        n["p"] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    elif t == "text":
        n["p"] = (e["p"][0] + dx, e["p"][1] + dy)
    elif t == "poly":
        n["pts"] = [(x + dx, y + dy) for x, y in e["pts"]]
    elif t in ("circle", "arc", "dot"):
        n["c"] = (e["c"][0] + dx, e["c"][1] + dy)
    elif t == "solid":
        if "rings" in e:
            n["rings"] = [[(x + dx, y + dy) for x, y in r] for r in e["rings"]]
        else:
            n["pts"] = [(x + dx, y + dy) for x, y in e["pts"]]
    return n


def _sample_ents(ents, step=0.6):
    """Points along paper entities (for collision tests); text contributes a grid over its box."""
    out = []
    for e in ents:
        t = e["t"]
        if t == "line":
            x1, y1, x2, y2 = e["p"]
            n = max(2, int(math.hypot(x2 - x1, y2 - y1) / step) + 1)
            s = np.linspace(0, 1, n)[:, None]
            out.append(np.array([x1, y1]) * (1 - s) + np.array([x2, y2]) * s)
        elif t in ("poly", "solid"):
            rings = e.get("rings") or [e["pts"]]
            for ring in rings:
                P = np.asarray(ring, float)
                if t == "solid" or e.get("closed"):
                    P = np.vstack([P, P[:1]])
                for a, b in zip(P[:-1], P[1:]):
                    n = max(2, int(math.hypot(*(b - a)) / step) + 1)
                    s = np.linspace(0, 1, n)[:, None]
                    out.append(a * (1 - s) + b * s)
        elif t in ("circle", "arc"):
            a0, a1 = e.get("a", (0.0, 360.0)) if t == "arc" else (0.0, 360.0)
            n = max(4, int(math.radians(a1 - a0) * e["r"] / step) + 1)
            A = np.radians(np.linspace(a0, a1, n))
            out.append(np.c_[e["c"][0] + e["r"] * np.cos(A), e["c"][1] + e["r"] * np.sin(A)])
        elif t == "dot":
            out.append(np.array([e["c"]], float))
    return np.concatenate(out) if out else np.zeros((0, 2))


def _arrow(tip, d, L, W, lt="dim"):
    """Closed filled arrowhead with its tip at ``tip`` pointing along unit ``d``."""
    tip = np.asarray(tip, float)
    d = np.asarray(d, float)
    n = _rot90(d)
    return _E("solid", lt, pts=[tuple(tip), tuple(tip - d * L + n * W / 2), tuple(tip - d * L - n * W / 2)])


def _out_offset(lay, sh):
    """Distance from a dimension end to the near edge of text placed outside it."""
    if lay.get("tick"):
        return sh.ext_over + 0.8
    return sh.arrow[0] + 2.0 if not lay["inside"] else 1.5


def _dim_layout(L, tw, sh):
    """Decide terminator and text placement for a dimension line of paper length L with text width tw."""
    AL = sh.arrow[0]
    if sh.terminator == "tick":                              # oblique strokes never need to flip outside
        inside, text_in = True, tw + 1.0 <= L
    else:
        inside = L >= 2 * AL + 1.0
        text_in = tw + 2.0 <= L
    lay = dict(inside=inside, text_in=text_in, tick=sh.terminator == "tick")
    lo, hi = -sh.ext_over, L + sh.ext_over
    if not inside:
        lo, hi = -(AL + 2.5), L + AL + 2.5
    if not text_in:
        hi = max(hi, L + _out_offset(lay, sh) + tw + 1.0)
    lay.update(lo=lo, hi=hi)
    return lay


def _layout_chain(segs, sh):
    """Text placement for the segments of a chain dimension so that no two values collide.

    ``segs``: [(a, b, tw, lay)] along the chain (paper mm, a < b). Values that fit stay centred; the others go
    just outside the segment on whichever side is free, else on a second row on the outer side of the
    dimension line, else outside anyway. Sets ``lay["place"]`` to 'in', 'out+' / 'out-' or 'outer' and returns
    the intervals used on the second row."""
    pad = 0.8
    ends = sorted({round(v, 6) for a, b, _, _ in segs for v in (a, b)})
    above, below = [], []

    def free(lo, hi, taken, skip=()):
        if any(lo < y + pad and hi > x - pad for x, y in taken):
            return False
        return not any(lo - 0.3 < e < hi + 0.3 for e in ends if e not in skip)

    for a, b, tw, lay in segs:
        if lay["text_in"]:
            m = (a + b) / 2
            above.append((m - tw / 2, m + tw / 2))
            lay["place"] = "in"
    n = len(segs)
    for i, (a, b, tw, lay) in enumerate(segs):
        if lay["text_in"]:
            continue
        off = _out_offset(lay, sh)
        cands = [("out+", b + off, b + off + tw), ("out-", a - off - tw, a - off)]
        if i == 0 and n > 1:
            cands.reverse()
        for name, lo, hi in cands:
            if free(lo, hi, above):
                lay["place"] = name
                above.append((lo, hi))
                break
        else:
            m = (a + b) / 2
            lo, hi = m - tw / 2, m + tw / 2
            if not any(lo < y + pad and hi > x - pad for x, y in below):
                lay["place"] = "outer"
                below.append((lo, hi))
            else:
                lay["place"] = cands[0][0]
    return below


def _dim_line(q1, q2, parts, tw, lay, sh):
    """Dimension line q1 → q2 (paper) with arrowheads (or oblique ticks) and text above."""
    out = []
    d = q2 - q1
    L = math.hypot(*d)
    if L < 1e-9:
        return out
    u = d / L
    ur = u if -90 < _ang(u) <= 90 + 1e-9 else -u
    if abs(_ang(u) - (-90)) < 1e-6:
        ur = -u
    AL, AW = sh.arrow
    a, b = q1.copy(), q2.copy()
    up = _rot90(ur)
    if lay.get("tick"):
        t = _unit(ur + up) * 1.6
        for q in (q1, q2):
            out.append(_E("line", "dim", p=(*(q - t), *(q + t)), w=sh.thick * 0.9))
        a, b = q1 - u * sh.ext_over * 0.75, q2 + u * sh.ext_over * 0.75
    elif lay["inside"]:
        out.append(_arrow(q1, -u, AL, AW))
        out.append(_arrow(q2, u, AL, AW))
        a, b = q1 + u * AL * 0.9, q2 - u * AL * 0.9
    else:
        out.append(_arrow(q1, u, AL, AW))
        out.append(_arrow(q2, -u, AL, AW))
        a, b = q1 - u * (AL + 2.5), q2 + u * (AL + 2.5)
    place = lay.get("place") or ("in" if lay["text_in"] else "out")
    anchor_up = up * sh.text_gap
    if place in ("in", "outer"):
        c = (q1 + q2) / 2
        if place == "outer":                                 # second row, away from the part
            sgn = 1.0 if (lay.get("outer") @ up) > 0 else -1.0
            if sgn > 0:
                anchor_up = up * (sh.text_gap + sh.text_h + 1.8)
            else:
                anchor_up = -up * (sh.ext_over + 0.8 + sh.text_h)
    else:
        if place == "out":
            e = q2 if (q2 - q1) @ ur > 0 else q1
            sgn = 1.0
        else:                                                # 'out+' = past q2 along u, 'out-' = before q1
            e = q2 if place == "out+" else q1
            sgn = 1.0 if ((place == "out+") == (u @ ur > 0)) else -1.0
        off = _out_offset(lay, sh) + tw / 2
        c = e + ur * sgn * off
        tail = e + ur * sgn * (off + tw / 2 + 0.5)
        if (tail - a) @ u > (b - a) @ u:
            b = tail
        elif (tail - a) @ u < 0:
            a = tail
    out.insert(0, _E("line", "dim", p=(*a, *b), w=None))
    out += sh._place_parts(parts, tw, c + anchor_up, ur)
    return out


def _union_outline(quads):
    """Boundary segments of the union of convex polygons (shared / interior edges removed)."""
    polys = [np.asarray(q, float) for q in quads]
    E_all = [(np.concatenate([p, np.roll(p, -1, 0)], 1)) for p in polys]
    eps = 1e-6 * max(1.0, max(float(np.abs(p).max()) for p in polys))
    verts = np.concatenate(polys)

    def inside_any(P):
        res = np.zeros(len(P), bool)
        for p in polys:
            q = np.roll(p, -1, 0)
            cr = (q[:, 0] - p[:, 0])[None] * (P[:, 1:2] - p[None, :, 1]) - \
                (q[:, 1] - p[:, 1])[None] * (P[:, 0:1] - p[None, :, 0])
            res |= np.all(cr > eps * 10, 1) | np.all(cr < -eps * 10, 1)
        return res

    out = []
    for pi, E in enumerate(E_all):
        for e in E:
            p, q = e[:2], e[2:]
            d = q - p
            L = math.hypot(*d)
            if L < eps:
                continue
            u = d / L
            ts = [0.0, 1.0]
            for pj, F in enumerate(E_all):
                if pj == pi:
                    continue
                for f in F:
                    a, b = f[:2], f[2:]
                    ed = b - a
                    den = d[0] * ed[1] - d[1] * ed[0]
                    if abs(den) > 1e-12:
                        w = a - p
                        t = (w[0] * ed[1] - w[1] * ed[0]) / den
                        s = (w[0] * d[1] - w[1] * d[0]) / den
                        if 0 < t < 1 and -1e-9 <= s <= 1 + 1e-9:
                            ts.append(t)
            for v in verts:
                w = v - p
                t = (w @ d) / (L * L)
                if 0 < t < 1 and abs(w[0] * u[1] - w[1] * u[0]) < eps * 10:
                    ts.append(t)
            ts = np.unique(np.round(np.array(ts), 12))
            n = _rot90(u)
            mids = (ts[:-1] + ts[1:]) / 2
            M = p[None] + mids[:, None] * d[None]
            off = max(L * 1e-4, 1e-3)
            inl = inside_any(M + n * off)
            inr = inside_any(M - n * off)
            for k in range(len(mids)):
                if inl[k] != inr[k] and ts[k + 1] - ts[k] > 1e-9:
                    A, B = p + d * ts[k], p + d * ts[k + 1]
                    out.append((A[0], A[1], B[0], B[1]))
    return _merge_segments(out)


def _loops(segs, tol=1e-6):
    """Chain boundary segments (x1, y1, x2, y2) into closed point rings."""
    def k(x, y):
        return (round(x / tol) * tol, round(y / tol) * tol) if tol else (x, y)
    adj = {}
    for i, (x1, y1, x2, y2) in enumerate(segs):
        a, b = k(round(x1, 4), round(y1, 4)), k(round(x2, 4), round(y2, 4))
        adj.setdefault(a, []).append((i, b))
        adj.setdefault(b, []).append((i, a))
    used = set()
    rings = []
    for i, (x1, y1, x2, y2) in enumerate(segs):
        if i in used:
            continue
        used.add(i)
        start = k(round(x1, 4), round(y1, 4))
        cur = k(round(x2, 4), round(y2, 4))
        ring = [start]
        for _ in range(len(segs) + 1):
            if cur == start:
                break
            ring.append(cur)
            nxt = [(j, q) for j, q in adj.get(cur, []) if j not in used]
            if not nxt:
                break
            j, q = nxt[0]
            used.add(j)
            cur = q
        if cur == start and len(ring) >= 3:
            rings.append(np.array(ring, float))
    return rings


def _merge_segments(segs, tol=1e-6):
    segs = [tuple(round(v, 6) for v in s) for s in segs]
    uniq = []
    seen = set()
    for s in segs:
        k = s if (s[0], s[1]) <= (s[2], s[3]) else (s[2], s[3], s[0], s[1])
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    changed = True
    while changed:
        changed = False
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                pa, qa, pb, qb = _v(*a[:2]), _v(*a[2:]), _v(*b[:2]), _v(*b[2:])
                da, db = qa - pa, qb - pb
                if abs(da[0] * db[1] - da[1] * db[0]) > 1e-6 * math.hypot(*da) * math.hypot(*db):
                    continue
                for x, y in ((qa, pb), (qa, qb), (pa, pb), (pa, qb)):
                    if math.hypot(*(x - y)) < 1e-6:
                        pts = [pa, qa, pb, qb]
                        u = _unit(da)
                        pr = [p @ u for p in pts]
                        lo, hi = pts[int(np.argmin(pr))], pts[int(np.argmax(pr))]
                        uniq[i] = (lo[0], lo[1], hi[0], hi[1])
                        uniq.pop(j)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return uniq


# --------------------------------------------------------------------------------------------- hatch patterns

def _hatch_pattern(rings, E, material, angle, sp, paper):
    P = np.concatenate(rings)
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    R = math.hypot(x1 - x0, y1 - y0) / 2 + sp

    def family(ang, spacing, offset=0.0, dash=None):
        d = _dir(ang)
        n = _rot90(d)
        k = np.arange(-math.ceil(R / spacing) - 1, math.ceil(R / spacing) + 2)
        segs = []
        for i in k:
            c = _v(cx, cy) + n * (i * spacing + offset)
            segs.append((*(c - d * R), *(c + d * R)))
        pieces = _clip_segments(segs, E)
        if dash is None:
            return pieces
        out = []
        on, off = dash
        for s in pieces:
            a, b = _v(*s[:2]), _v(*s[2:])
            L = math.hypot(*(b - a))
            u = (b - a) / max(L, 1e-12)
            t0 = (((a - _v(cx, cy)) @ u) % (on + off))
            t = -t0
            while t < L:
                s0, s1 = max(t, 0), min(t + on, L)
                if s1 > s0:
                    out.append((*(a + u * s0), *(a + u * s1)))
                t += on + off
        return out

    m = material
    if m in ("general", None):
        return family(angle, sp)
    if m == "steel":
        return family(angle, sp) + family(angle, sp, offset=sp * 0.28)
    if m in ("aluminium", "aluminum"):
        return family(angle, sp * 2) + family(angle, sp * 2, offset=sp, dash=(sp * 1.2, sp * 0.6))
    if m == "plastic":
        return family(angle, sp * 1.2) + family(angle + 90, sp * 1.2)
    if m == "concrete":
        rng = np.random.default_rng(7)
        area = abs(sum(_area(r) for r in rings)) or (x1 - x0) * (y1 - y0)
        unit = paper(1.0)
        n = int(area / (unit * unit * 9.0)) + 4
        pts = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], -1)
        pts = pts[_inside(pts, E)]
        out = []
        for i, p in enumerate(pts):
            if i % 3 == 0:                                   # small aggregate triangle
                s = unit * rng.uniform(0.7, 1.3)
                a = rng.uniform(0, 360)
                tri = [p + _dir(a + k * 120) * s for k in range(3)]
                for k in range(3):
                    out.append((*tri[k], *tri[(k + 1) % 3]))
            else:                                            # stipple dot
                d = _dir(rng.uniform(0, 180)) * unit * 0.12
                out.append((*(p - d), *(p + d)))
        return _clip_segments(out, E)
    if m == "insulation":                                    # soft batt: looping prolate cycloid
        w, h = x1 - x0, y1 - y0
        horiz = w >= h
        L, T = (w, h) if horiz else (h, w)
        lam = T * 0.55
        R = T * 0.46
        th = np.linspace(0, (L / lam + 1) * 2 * np.pi, int((L / lam + 1) * 48))
        a = lam / (2 * np.pi) * th - R * np.sin(th) - lam * 0.25
        b = T / 2 - R * np.cos(th)
        X, Y = (a, b) if horiz else (b, a)
        X, Y = X + x0, Y + y0
        segs = [(X[k], Y[k], X[k + 1], Y[k + 1]) for k in range(len(X) - 1)]
        return _clip_segments(segs, E)
    if m == "earth":
        c = sp * 2.4
        out = []
        nx, ny = int((x1 - x0) / c) + 2, int((y1 - y0) / c) + 2
        for i in range(-1, nx):
            for j in range(-1, ny):
                ox, oy = x0 + i * c, y0 + j * c
                if (i + j) % 2 == 0:
                    for k in range(3):
                        yy = oy + c * (0.25 + 0.25 * k)
                        out.append((ox + c * 0.1, yy, ox + c * 0.9, yy))
                else:
                    for k in range(3):
                        xx = ox + c * (0.25 + 0.25 * k)
                        out.append((xx, oy + c * 0.1, xx, oy + c * 0.9))
        a = math.radians(angle - 45)
        ca, sa = math.cos(a), math.sin(a)
        rot = []
        for s in out:
            p = [(s[0] - cx, s[1] - cy), (s[2] - cx, s[3] - cy)]
            q = [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in p]
            rot.append((*q[0], *q[1]))
        return _clip_segments(rot, E)
    raise _unknown_name("material", material, MATERIALS)


# --------------------------------------------------------------------------------------------- GD&T symbols

def _gdt_symbol(name, c, s):
    """Characteristic symbol geometry (paper), centred on c, about s mm tall."""
    out = []
    h = s / 2

    def L(a, b):
        out.append(_E("line", "dim", p=(c[0] + a[0], c[1] + a[1], c[0] + b[0], c[1] + b[1]), w=None))

    def C(r, dx=0.0, dy=0.0):
        out.append(_E("circle", "dim", c=(c[0] + dx, c[1] + dy), r=r, w=None))

    def A(r, a0, a1, dy=0.0):
        out.append(_E("arc", "dim", c=(c[0], c[1] + dy), r=r, a=(a0, a1), w=None))

    if name == "straightness":
        L((-h, 0), (h, 0))
    elif name == "flatness":
        L((-h, -h * 0.5), (h * 0.5, -h * 0.5)), L((h * 0.5, -h * 0.5), (h, h * 0.5))
        L((h, h * 0.5), (-h * 0.5, h * 0.5)), L((-h * 0.5, h * 0.5), (-h, -h * 0.5))
    elif name == "circularity":
        C(h * 0.85)
    elif name == "cylindricity":
        C(h * 0.6)
        L((-h * 0.35, -h * 0.9), (-h * 0.95, h * 0.9)), L((h * 0.95, -h * 0.9), (h * 0.35, h * 0.9))
    elif name == "profile_line":
        A(h * 0.85, 0, 180, -h * 0.4)
    elif name == "profile_surface":
        A(h * 0.85, 0, 180, -h * 0.4)
        L((-h * 0.85, -h * 0.4), (h * 0.85, -h * 0.4))
    elif name == "parallelism":
        L((-h * 0.7, -h), (-h * 0.1, h)), L((h * 0.1, -h), (h * 0.7, h))
    elif name == "perpendicularity":
        L((-h, -h * 0.85), (h, -h * 0.85)), L((0, -h * 0.85), (0, h * 0.95))
    elif name == "angularity":
        L((-h, -h * 0.85), (h, -h * 0.85)), L((-h, -h * 0.85), (h * 0.8, h * 0.6))
    elif name == "position":
        C(h * 0.62)
        L((-h, 0), (h, 0)), L((0, -h), (0, h))
    elif name == "concentricity":
        C(h * 0.85), C(h * 0.45)
    elif name == "symmetry":
        L((-h, 0), (h, 0)), L((-h * 0.6, h * 0.6), (h * 0.6, h * 0.6)), L((-h * 0.6, -h * 0.6), (h * 0.6, -h * 0.6))
    elif name in ("runout", "total_runout"):
        tip = _v(h * 0.3, h)
        L((-h * 0.3, -h), tuple(tip))
        out.append(_E("solid", "dim", pts=[tuple(c + tip), tuple(c + tip + _v(-h * 0.55, -h * 0.25)),
                                           tuple(c + tip + _v(-h * 0.05, -h * 0.6))]))
        if name == "total_runout":
            L((h * 0.3, -h), (h * 0.9, h))
            L((-h * 0.6, -h), (h * 0.6, -h))
    else:
        raise _unknown_name("GD&T symbol", name, GDT_SYMBOLS)
    return out


# --------------------------------------------------------------------------------------------- furniture

def _rect_pts(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _rounded(x0, y0, x1, y1, r, n=4):
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _ellipse_pts(cx, cy, rx, ry, a0=0, a1=360, n=40):
    t = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * math.cos(v), cy + ry * math.sin(v)) for v in t]


def _f_bed(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    out = [("poly", _rect_pts(x0, y0, x1, y1), True)]
    pw = (w - 150) / (2 if w > 1100 else 1)
    for k in range(2 if w > 1100 else 1):
        px = x0 + 50 + k * (pw + 50)
        out.append(("poly", _rounded(px, y1 - 450, px + pw, y1 - 80, 60), True))
    out.append(("line", (x0, y1 - 600), (x1, y1 - 600)))
    out.append(("poly", [(x0, y1 - 600), (x0 + w * 0.35, y1 - 780), (x1, y1 - 600)], False))
    return out


def _f_sofa(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    arm, back = min(220, w * 0.12), min(220, d * 0.25)
    out = [("poly", _rounded(x0, y0, x1, y1, 60), True),
           ("poly", _rounded(x0 + 10, y1 - back, x1 - 10, y1 - 10, 50), True),
           ("poly", _rounded(x0 + 10, y0 + 10, x0 + arm, y1 - back, 50), True),
           ("poly", _rounded(x1 - arm, y0 + 10, x1 - 10, y1 - back, 50), True)]
    n = max(1, int(round((w - 2 * arm) / 700)))
    cw = (w - 2 * arm) / n
    for k in range(n):
        a = x0 + arm + k * cw
        out.append(("poly", _rounded(a + 15, y0 + 40, a + cw - 15, y1 - back - 15, 60), True))
    return out


def _f_armchair(w, d, **o):
    return _f_sofa(w, d)


def _f_chair(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    return [("poly", _rounded(x0, y0, x1, y1 - 70, 50), True), ("poly", _rounded(x0, y1 - 80, x1, y1, 30), True)]


def _f_table(w, d, seats=4, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    out = []
    cw, cd = 440, 440
    per_side = max(1, seats // 2)
    for side in (-1, 1):
        for k in range(per_side):
            cx = x0 + w * (k + 0.5) / per_side
            cy = side * (d / 2 + cd * 0.25)
            pts = _rounded(cx - cw / 2, cy - cd / 2, cx + cw / 2, cy + cd / 2, 50)
            if side < 0:
                out.append(("poly", pts, True))
                out.append(("line", (cx - cw / 2, cy - cd / 2 + 70), (cx + cw / 2, cy - cd / 2 + 70)))
            else:
                out.append(("poly", pts, True))
                out.append(("line", (cx - cw / 2, cy + cd / 2 - 70), (cx + cw / 2, cy + cd / 2 - 70)))
    if seats % 2 or seats > 4 and w > 1600:
        pass
    out.append(("poly", _rect_pts(x0, y0, x1, y1), True))
    return out


def _f_desk(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    return [("poly", _rect_pts(x0, y0, x1, y1), True)] + \
        [("poly", _rounded(-220, y0 - 330, 220, y0 + 110, 60), True)]


def _f_toilet(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    tank = 180
    return [("poly", _rounded(x0, y1 - tank, x1, y1, 30), True),
            ("poly", _ellipse_pts(0, y0 + (d - tank) * 0.48, w * 0.44, (d - tank) * 0.5, 0, 360), True),
            ("poly", _ellipse_pts(0, y0 + (d - tank) * 0.45, w * 0.3, (d - tank) * 0.34, 0, 360), True)]


def _f_basin(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    return [("poly", _rounded(x0, y0, x1, y1, 40), True),
            ("poly", _ellipse_pts(0, -d * 0.05, w * 0.36, d * 0.32), True), ("circle", (0, d * 0.36), 22)]


def _f_shower(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    return [("poly", _rect_pts(x0, y0, x1, y1), True), ("poly", _rect_pts(x0 + 40, y0 + 40, x1 - 40, y1 - 40), True),
            ("line", (x0 + 40, y0 + 40), (x1 - 40, y1 - 40)), ("line", (x0 + 40, y1 - 40), (x1 - 40, y0 + 40)),
            ("circle", (0, 0), 45)]


def _f_bathtub(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    return [("poly", _rect_pts(x0, y0, x1, y1), True), ("poly", _rounded(x0 + 70, y0 + 70, x1 - 70, y1 - 70, 180), True),
            ("circle", (x1 - 200, 0), 30)]


def _f_counter(w, d, sink=None, hob=None, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    out = [("poly", _rect_pts(x0, y0, x1, y1), True), ("line", (x0, y0 + 40), (x1, y0 + 40))]
    if sink is not None:
        sx = x0 + sink
        out += [("poly", _rounded(sx - 280, -d / 2 + 90, sx + 280, d / 2 - 80, 50), True),
                ("circle", (sx, d / 2 - 140), 25)]
    if hob is not None:
        hx = x0 + hob
        out.append(("poly", _rect_pts(hx - 290, -d / 2 + 70, hx + 290, d / 2 - 50), True))
        for dx, dy, r in ((-130, -110, 80), (140, -110, 95), (-130, 120, 95), (140, 120, 80)):
            out.append(("circle", (hx + dx, dy + 10), r))
    return out


def _f_sink(w, d, **o):
    return _f_counter(w, d, sink=w / 2)


def _f_hob(w, d, **o):
    return _f_counter(w, d, hob=w / 2)


def _f_fridge(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    return [("poly", _rect_pts(x0, y0, x1, y1), True), ("line", (x0, y0 + 60), (x1, y0 + 60)),
            ("line", (x0, y0), (x1, y1)), ("line", (x0, y1), (x1, y0))]


def _f_wardrobe(w, d, **o):
    x0, y0, x1, y1 = -w / 2, -d / 2, w / 2, d / 2
    out = [("poly", _rect_pts(x0, y0, x1, y1), True), ("line", (x0 + 40, 0), (x1 - 40, 0))]
    n = max(2, int(w / 120))
    for k in range(n):
        x = x0 + w * (k + 0.5) / n
        out.append(("line", (x - 70, -d * 0.32), (x + 70, d * 0.32)))
    return out


def _f_single_bed(w, d, **o):
    return _f_bed(w, d)


_FURN = {
    "bed": ((1600, 2000), _f_bed), "single_bed": ((900, 2000), _f_single_bed), "sofa": ((2100, 900), _f_sofa),
    "armchair": ((850, 850), _f_armchair), "table": ((1400, 850), _f_table), "chair": ((450, 480), _f_chair),
    "toilet": ((380, 680), _f_toilet), "basin": ((550, 420), _f_basin), "shower": ((900, 900), _f_shower),
    "bathtub": ((1700, 750), _f_bathtub), "counter": ((2400, 600), _f_counter), "sink": ((800, 600), _f_sink),
    "hob": ((600, 600), _f_hob), "fridge": ((600, 650), _f_fridge), "wardrobe": ((1800, 600), _f_wardrobe),
    "desk": ((1200, 600), _f_desk),
}


# --------------------------------------------------------------------------------------------- isometric scene

_VIEW_DIR = np.array([1.0, -1.0, 1.0]) / math.sqrt(3)


def _iso_np(P):
    c = math.cos(math.pi / 6)
    return np.stack([(P[:, 0] + P[:, 1]) * c, (P[:, 1] - P[:, 0]) * 0.5 + P[:, 2]], -1)


def _to3d(ab, plane, w):
    ab = np.asarray(ab, float)
    W = np.full(len(ab), float(w))
    if plane == "xy":
        return np.stack([ab[:, 0], ab[:, 1], W], -1)
    if plane == "xz":
        return np.stack([ab[:, 0], W, ab[:, 1]], -1)
    if plane == "yz":
        return np.stack([W, ab[:, 0], ab[:, 1]], -1)
    _check_choice("iso", "plane", plane, ["xy", "xz", "yz"])


class _IsoScene:
    """Planar faces + edges in 3-D with hidden-line removal for the isometric view."""

    def __init__(self):
        self.faces = []                                   # dict(rings=[(N,3)], n, c, axis, plane coord)
        self.edges = []                                   # (N, 3) arrays (polylines; closed rings repeat first)
        self.graze = set()                                # silhouette edges of curved sides

    def _face(self, rings3):
        R = rings3[0]
        n = np.zeros(3)
        for i in range(len(R)):                           # Newell normal
            a, b = R[i], R[(i + 1) % len(R)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                           (a[0] - b[0]) * (a[1] + b[1])])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            return
        n /= nn
        self.faces.append(dict(rings=[np.asarray(r, float) for r in rings3], n=n, c=float(n @ R[0])))

    def prism(self, rings2, plane, w0, w1):
        rings2 = [np.asarray(r, float) for r in rings2]
        caps = [[_to3d(r, plane, w) for r in rings2] for w in (w0, w1)]
        for cap in caps:
            self._face(cap)
            for r in cap:
                self.edges.append(np.concatenate([r, r[:1]]))
        axis = {"xy": 2, "xz": 1, "yz": 0}[plane]
        for ri, r in enumerate(rings2):
            A, B = caps[0][ri], caps[1][ri]
            m = len(r)
            facing = []
            for i in range(m):
                j = (i + 1) % m
                quad = np.array([A[i], A[j], B[j], B[i]])
                self._face([quad])
                facing.append(self.faces[-1]["n"] @ _VIEW_DIR if self.faces else 0.0)
            for i in range(m):                               # side edges: sharp corners + silhouettes
                p, q, s = r[i - 1], r[i], r[(i + 1) % m]
                u1, u2 = _unit(q - p), _unit(s - q)
                turn = math.degrees(math.acos(float(np.clip(u1 @ u2, -1, 1))))
                sil = (facing[i - 1] > 1e-9) != (facing[i] > 1e-9)
                if turn > 25 or sil:
                    self.edges.append(np.array([A[i], B[i]]))
                    if sil and turn <= 25:
                        self.graze.add(len(self.edges) - 1)

    def hole(self, _unused, centre, axis, ring):
        ax = "xyz".index(axis)
        keep = [i for i in range(3) if i != ax]
        c = np.asarray(centre, float)
        for f in self.faces:
            if abs(abs(f["n"][ax]) - 1) > 1e-6:
                continue
            w = f["c"] / f["n"][ax]                        # plane coordinate along the axis
            outer = f["rings"][0]
            E = _edges_of([outer[:, keep]])
            if not _inside(np.array([[c[keep[0]], c[keep[1]]]]), E)[0]:
                continue
            P = np.zeros((len(ring), 3))
            P[:, keep[0]] = ring[:, 0] + c[keep[0]]
            P[:, keep[1]] = ring[:, 1] + c[keep[1]]
            P[:, ax] = w
            f["rings"].append(P)
            self.edges.append(np.concatenate([P, P[:1]]))

    def visible_runs(self, step):
        runs = []
        F = []
        tol = step * 0.8
        for f in self.faces:
            nd = f["n"] @ _VIEW_DIR
            if abs(nd) < 1e-9:
                continue
            rings2 = [_iso_np(r) for r in f["rings"]]
            E = _edges_of(rings2)
            allp = np.concatenate(rings2)
            F.append((f, nd, E, allp.min(0), allp.max(0)))
        for ei, e in enumerate(self.edges):
            seg = []
            for i in range(len(e) - 1):
                a, b = e[i], e[i + 1]
                L = np.linalg.norm(b - a)
                k = max(1, int(math.ceil(L / step)))
                t = np.linspace(0, 1, k + 1)[:-1] if i < len(e) - 2 else np.linspace(0, 1, k + 1)
                seg.append(a[None] + t[:, None] * (b - a)[None])
            P = np.concatenate(seg)
            S = _iso_np(P)
            vis = np.ones(len(P), bool)
            scale = max(1.0, float(np.abs(P).max()))
            for f, nd, E, lo, hi in F:
                m = vis & (S[:, 0] >= lo[0]) & (S[:, 0] <= hi[0]) & (S[:, 1] >= lo[1]) & (S[:, 1] <= hi[1])
                if not m.any():
                    continue
                idx = np.nonzero(m)[0]
                t = (f["c"] - P[idx] @ f["n"]) / nd             # distance towards the viewer to the plane
                cand = t > 2e-3 * scale
                if not cand.any():
                    continue
                idx = idx[cand]
                ins = _inside(S[idx], E)
                if ins.any() and ei in self.graze:              # grazing silhouettes stay visible
                    j = idx[ins]
                    ins_j = _seg_dist(S[j], E) > tol
                    vis[j[ins_j]] = False
                elif ins.any():
                    vis[idx[ins]] = False
            # split into runs
            k = 0
            n = len(P)
            while k < n:
                if not vis[k]:
                    k += 1
                    continue
                j = k
                while j + 1 < n and vis[j + 1]:
                    j += 1
                if j > k + 2:
                    runs.append(S[k:j + 1])
                k = j + 1
        return runs


# --------------------------------------------------------------------------------------------- Sheet

class Sheet:
    """A drawing sheet with border, zone marks and title block; holds views and sheet-level annotations.

    ``Sheet(size="A3", orientation="landscape", scale=None, projection="first", title=..., drawing_no=...,
    author=..., date=..., material=..., units="mm", revision="A", checked=None, approved=None, company=None,
    tolerances="ISO 2768-m", finish=None, sheet_no="1/1", dpi=150, line_width=0.5, style="white", font="sans",
    fields=None)``

    * ``scale``: '1:2', '2:1' or a factor. Leave it out and a plan (walls) gets the largest standard scale that
      fits the sheet, other drawings 1:1; ``"auto"`` picks the largest standard scale that fits for any drawing.
      Views placed automatically always fit: when they do not at the given scale the sheet drops to the largest
      standard scale that fits and reports it in ``sheet.fit_report`` (``fit="error"`` raises a ValueError that
      names that scale instead). The scale bar and the title block follow the chosen scale.
    * ``size``: A0 – A4 or (width, height) mm; ``projection``: 'first' (ISO E) or 'third' (ISO A / ASME).
    * ``dpi``: raster density of ``draw()`` (150 → A3 landscape is 2480 px wide); ``draw(cv)`` fits any canvas.
    * ``line_width``: thick lines (0.5 mm); thin lines are half of it (ISO 128 2:1).
    * ``style``: 'white' (black ink on white), 'paper' (warm paper) or 'blueprint' (white on blue).
    * ``fields``: replace the title block's top row with your own [(label, value), ...] cells.
    * ``arrows``: linear-dimension terminators, 'arrow' (filled 3 x 1 mm), 'tick' (oblique strokes, the
      architectural convention) or 'auto' (ticks at 1:20 and smaller scales, arrows otherwise)."""

    def __init__(self, size="A3", orientation="landscape", scale=None, projection="first", title="",
                 drawing_no="", author="", date="", material="", units="mm", revision="A", checked="",
                 approved="", company="", tolerances="ISO 2768-m", finish="", sheet_no="1/1", dpi=150,
                 line_width=0.5, style="white", font="sans", text_height=3.5, title_block=True, zones=True,
                 mass="", fields=None, arrows="auto", fit="shrink"):
        if isinstance(size, str):
            if size.upper() not in SHEET_SIZES:
                raise _unknown_name("sheet size", size, SHEET_SIZES)
            W, H = SHEET_SIZES[size.upper()]
            self.size_name = size.upper()
        else:
            W, H = float(size[0]), float(size[1])
            self.size_name = f"{_fmt(W)}x{_fmt(H)}"
        if (orientation == "portrait") == (W > H):
            W, H = H, W
        self.W, self.H = float(W), float(H)
        if fit not in ("shrink", "error"):
            _check_choice("Sheet", "fit", fit, ["shrink", "error"])
        self.fit = fit
        self._scale_mode = "auto" if (isinstance(scale, str) and scale.lower() == "auto") else (
            "plans" if scale is None else "fixed")
        if self._scale_mode != "fixed":
            scale = "1:1"
        self.scale_text = scale if isinstance(scale, str) else _scale_text(float(scale))
        self.s = _scale_factor(scale)
        self.fit_report = ""                              # what the last layout did about the sheet scale
        self._bars = []
        self._arrows = arrows
        if projection not in ("first", "third"):
            _check_choice("Sheet", "projection", projection, ["first", "third"])
        self.projection = projection
        self.info = dict(title=title, drawing_no=drawing_no, author=author, date=date, material=material,
                         units=units, revision=revision, checked=checked, approved=approved, company=company,
                         tolerances=tolerances, finish=finish, sheet_no=sheet_no, mass=mass)
        self.fields = list(fields) if fields else None
        self.dpi = float(dpi)
        self.thick = float(line_width)
        self.thin = self.thick / 2
        if style not in _STYLES:
            raise _unknown_name("sheet style", style, _STYLES)
        self.style = style
        self.font = font
        self.text_h = float(text_height)
        self.arrow = (3.0, 1.0)                           # length, width (paper mm)
        if arrows not in ("auto", "arrow", "tick"):
            _check_choice("Sheet", "arrows", arrows, ["auto", "arrow", "tick"])
        # linear dimension terminators: filled arrows, or oblique ticks (architectural; 'auto' = scales 1:20+)
        self.terminator = self._terminator()
        self.ext_gap, self.ext_over = 1.0, 2.0
        self.text_gap = 1.0
        self.dim_first, self.dim_step = 10.0, 8.0
        self.views = []
        self._sheet = View(self, "sheet", (0, 0), 1.0)
        self._sheet.s = 1.0
        self._groups = {}
        self.margins = (20.0, 10.0, 10.0, 10.0)           # left, bottom, right, top
        self.show_title_block = title_block
        self.show_zones = zones
        self._parts = None
        self._tb = (self.W - self.margins[2] - 180.0, self.margins[1], self.W - self.margins[2],
                    self.margins[1] + 40.0) if title_block else None

    def _terminator(self):
        return self._arrows if self._arrows != "auto" else ("tick" if self.s <= 1 / 20 + 1e-12 else "arrow")

    # ---- views
    def view(self, name, origin=None, scale=None, label=None, center=None):
        """New View. ``origin`` = sheet mm (from the paper's bottom-left corner) where model (0, 0) lands; leave it
        out and the view, with its dimensions and its projected views, is centred in the free drawing area
        (``center=(x, y)`` centres it on that sheet point instead). ``scale`` ('1:2', '2:1' or a factor) defaults
        to the sheet scale; ``label`` prints under the view."""
        v = View(self, name, origin if origin is not None else (0.0, 0.0), scale, label)
        if center is not None and center is not False:
            v._auto = True if center is True else (float(center[0]), float(center[1]))
        elif origin is None:
            v._auto = True
        self.views.append(v)
        return v

    def projected(self, base, which, distance=None, name=None, label=None, gap=12.0):
        """View aligned with ``base`` for the sheet's projection method: ``which`` = 'top' 'bottom' 'left' 'right'
        (view from that side). By default it is placed ``gap`` paper mm clear of ``base``, dimensions and labels
        included; ``distance`` = fixed paper mm between the two view origins instead.

        Model axes: front (X, Z) · top (X, Y) · right (Y, Z) · left (-Y, Z), Y pointing away from the viewer of
        the front view. First angle puts the top view below and the right view on the left; third angle the
        opposite."""
        if which not in ("top", "bottom", "left", "right"):
            _check_choice("projected", "which", which, ["top", "bottom", "left", "right"])
        sgn = 1.0 if self.projection == "third" else -1.0
        off = {"top": (0, sgn), "bottom": (0, -sgn), "right": (sgn, 0), "left": (-sgn, 0)}[which]
        dist = 0.0 if distance is None else float(distance)
        o = (base._ox + off[0] * dist, base._oy + off[1] * dist)
        v = View(self, name or f"{which.upper()} VIEW", o, base.s, label)
        v._sheet_scale = base._sheet_scale
        v._follow = (base, off, None if distance is not None else float(gap))
        self.views.append(v)
        return v

    def detail(self, source, cx, cy, r, letter="B", origin=(0.0, 0.0), scale="2:1"):
        """Detail view: marks a circle of model radius ``r`` at (cx, cy) on ``source`` and shows that area
        enlarged at ``origin`` (the detail centre on the sheet) with label "DETAIL B (2:1)". Returns the View
        (add dimensions to it like any view)."""
        source.detail_circle(cx, cy, r, letter)
        f = _scale_factor(scale)
        v = View(self, f"DETAIL {letter}", (origin[0] - cx * f, origin[1] - cy * f), f, f"DETAIL {letter}")
        self.views.append(v)

        def run():
            geom, _ = source._finalize() if source not in _BUSY else (source._geom, [])
            for e in _clip_to_circle(geom, (cx, cy), r):
                v._out_geom.append(e)
            v._out_anno.append(_E("circle", "thin", c=v._to_paper(cx, cy), r=r * f, w=None))
        v._later(0, run)
        return v

    # ---- sheet-level annotations (paper coordinates)
    def text(self, x, y, s, h=None, anchor="lb", rot=0.0, bold=False):
        """Text on the sheet at (x, y) mm; ``anchor`` horizontal l/m/r + vertical b/m/t."""
        ha, va = _anchor2(anchor)
        self._add("notes", self._text(_v(x, y), s, h or self.text_h, ha, va, rot, bold))
        return self

    def notes(self, lines, x=None, y=None, title="NOTES", h=None, numbered=True, width=None):
        """Notes block: ``title`` then numbered ``lines`` with their top-left at (x, y) mm (default: above the
        title block / parts list, aligned with its left edge). Long lines wrap at ``width`` mm (default: the
        drawing frame), continuation lines aligned after the number."""
        h = h or self.text_h
        x_ = x if x is not None else (self._tb or (self.W - 190, 10))[0]
        limit = width if width is not None else self.W - self.margins[2] - 3.0 - x_
        rows = []                                            # (number or "", text, indent)
        for i, ln in enumerate(lines):
            lead = f"{i + 1}." if numbered else ""
            ind = self._tw(lead, h) + h * 0.45 if lead else 0.0
            cur = ""
            for wd in str(ln).split():                       # wrap to the frame (or ``width``)
                t = (cur + " " + wd) if cur else wd
                if cur and ind + self._tw(t, h) > limit:
                    rows.append((lead, cur, ind))
                    lead, cur = "", wd
                else:
                    cur = t
            rows.append((lead, cur, ind))
        if x is None or y is None:
            tb = self._tb or (self.W - 190, 10, self.W - 10, 10)
            top = tb[3] + (self._parts_height() if self._parts else 0)
            y = top + 6 + (len(rows) + 1) * h * 1.7 if y is None else y
        x = x_
        out = self._text(_v(x, y), title, h * 1.15, "l", "t", bold=True)
        for r, (lead, t, ind) in enumerate(rows):
            yy = y - (r + 1) * h * 1.7 - 0.8
            if lead:
                out += self._text(_v(x, yy), lead, h, "l", "t")
            out += self._text(_v(x + ind, yy), t, h, "l", "t")
        self._add("notes", out)
        return self

    def parts_list(self, rows, columns=("ITEM", "QTY", "DESCRIPTION", "MATERIAL"), widths=None, row_h=7.0):
        """Parts list directly above the title block: header row at the bottom, item 1 above it
        (ISO 7573). ``rows`` = list of tuples matching ``columns``."""
        self._parts = dict(rows=[tuple(str(c) for c in r) for r in rows], columns=columns, widths=widths,
                           row_h=row_h)
        return self

    def table(self, rows, columns, x, y, widths=None, row_h=6.5, title=None, h=None):
        """Free-standing table with its top-left corner at (x, y) mm: optional bold ``title``, header row of
        ``columns``, then ``rows``. ``widths`` per column (default: equal, 30 mm)."""
        h = h or self.text_h * 0.8
        widths = list(widths or [30.0] * len(columns))
        out = []
        if title:
            out += self._text(_v(x, y + 2.0), title, self.text_h * 1.15, "l", "b", bold=True)
        allrows = [tuple(columns)] + [tuple(str(c) for c in r) for r in rows]
        for i, row in enumerate(allrows):
            yt = y - i * row_h
            xx = x
            for j, wv in enumerate(widths):
                out.append(_E("poly", "thin", pts=_rect_pts(xx, yt - row_h, xx + wv, yt), closed=True,
                              w=self.thin * (1.4 if i == 0 else 1.0)))
                val = row[j] if j < len(row) else ""
                out += self._text(_v(xx + 1.8, yt - row_h / 2), val, h * (0.85 if i == 0 else 1.0), "l", "m",
                                  bold=(i == 0))
                xx += wv
        W = sum(widths)
        H = row_h * len(allrows)
        out.append(_E("poly", "visible", pts=_rect_pts(x, y - H, x + W, y), closed=True, w=self.thick * 0.8))
        self._add("table", out)
        return self

    def room_schedule(self, view, x=None, y=None, title="ROOM SCHEDULE", total="NET FLOOR AREA",
                      widths=(12.0, 64.0, 30.0)):
        """Table of the rooms labelled with ``view.room(...)`` (number, name, area) with its top-left at
        (x, y) mm (default: above the title block, aligned with its left edge) and a bold total line
        (``total=None`` leaves it out)."""
        rooms = [(n, a) for n, a in view.rooms.items() if a is not None]
        rows = [(i + 1, n, f"{a:.1f} m²") for i, (n, a) in enumerate(rooms)]
        row_h = 6.5
        H = row_h * (len(rows) + 1)
        if x is None or y is None:
            tb = self._tb or (self.W - 190, 10, self.W - 10, 10)
            top = tb[3] + (self._parts_height() if self._parts else 0)
            x = tb[0] if x is None else x
            y = top + 14 + H if y is None else y
        self.table(rows, ("NO.", "ROOM", "AREA"), x, y, widths=widths, row_h=row_h, title=title)
        if total and rooms:
            s = sum(a for _, a in rooms)
            self._add("table", self._text(_v(x + sum(widths) - 1.8, y - H - 5.0), f"{total}  {s:.1f} m²",
                                          self.text_h * 0.85, "r", "t", bold=True))
        return self

    def _parts_height(self):
        p = self._parts
        return (len(p["rows"]) + 1) * p["row_h"] if p else 0.0

    def north_arrow(self, x, y, size=16.0, angle=90.0):
        """North arrow centred at (x, y) mm, pointing to ``angle`` (degrees, 90 = up)."""
        c = _v(x, y)
        d = _dir(angle)
        n = _rot90(d)
        r = size / 2
        out = [_E("circle", "thin", c=tuple(c), r=r, w=self.thin)]
        tip, tail = c + d * r * 1.05, c - d * r * 0.75
        out.append(_E("solid", "thin", pts=[tuple(tip), tuple(tail - n * r * 0.42), tuple(c - d * r * 0.35)]))
        out.append(_E("poly", "thin", pts=[tuple(tip), tuple(tail + n * r * 0.42), tuple(c - d * r * 0.35)],
                      closed=True, w=self.thin))
        out += self._text(c + d * (r + 2.0), "N", self.text_h * 1.4, "m", "b", rot=angle - 90, bold=True)
        self._add("north arrow", out)
        return self

    def scale_bar(self, x, y, length=None, divisions=None, unit="m", h=2.5):
        """Scale bar with its lower-left at (x, y) mm for the sheet scale: alternating blocks with labels
        (``length`` in ``unit``, default 5 m). It follows the scale the sheet settles on."""
        if unit not in ("m", "cm", "mm"):
            _check_choice("scale_bar", "unit", unit, ["m", "cm", "mm"])
        self._bars.append((x, y, length, divisions, unit, h))
        return self

    def _build_scale_bars(self):
        self._groups.pop("scale bar", None)
        for x, y, length, divisions, unit, h in self._bars:
            k = {"m": 1000.0, "cm": 10.0, "mm": 1.0}[unit]
            length = length or 5
            divisions = divisions or int(length)
            Lp = length * k * self.s
            out = []
            seg = Lp / divisions
            for i in range(divisions):
                x0 = x + i * seg
                pts = [(x0, y), (x0 + seg, y), (x0 + seg, y + h), (x0, y + h)]
                if i % 2 == 0:
                    out.append(_E("solid", "thin", pts=pts))
                out.append(_E("poly", "thin", pts=pts, closed=True, w=self.thin))
            for i in range(divisions + 1):
                v = length * i / divisions
                out += self._text(_v(x + i * seg, y + h + 1.2), _fmt(v), self.text_h * 0.75, "m", "b")
            out += self._text(_v(x + Lp + 2.5, y + h / 2), unit, self.text_h * 0.75, "l", "m")
            out += self._text(_v(x, y - 2.0), f"SCALE {self.scale_text}", self.text_h * 0.85, "l", "t", bold=True)
            self._add("scale bar", out)

    def _add(self, group, ents):
        self._groups.setdefault(group, []).extend(ents)

    # ---- text helpers
    def _tw(self, s, h, bold=False):
        return _em_width(str(s), self.font, bold) * h / _cap_ratio(self.font, bold)

    def _text(self, p, s, h, ha="l", va="b", rot=0.0, bold=False, halo=False, lt="text", color=None):
        return [_E("text", lt, p=(float(p[0]), float(p[1])), s=str(s), h=float(h), rot=float(rot), ha=ha, va=va,
                   bold=bool(bold), halo=halo, color=color)]

    def _dim_parts(self, label, tol, limits):
        """Text pieces of a dimension value in a local frame (x along the reading direction, baseline 0)."""
        h = self.text_h
        parts = []
        if limits and tol is not None:
            try:
                base = float(label.lstrip("ØR"))
                pre = label[:len(label) - len(label.lstrip("ØR"))]
            except ValueError:
                base, pre = None, ""
            if base is not None:
                up, lo = (tol, -tol) if isinstance(tol, (int, float)) else tol
                t1, t2 = f"{pre}{_fmt(base + up)}", f"{pre}{_fmt(base + lo)}"
                hh = h * 0.85
                w = max(self._tw(t1, hh), self._tw(t2, hh))
                parts = [(0.0, hh * 0.6 + 0.2, t1, hh), (0.0, -hh * 0.55, t2, hh)]
                return [(x - w / 2, y, s, hz) for x, y, s, hz in parts], w
        w0 = self._tw(label, h)
        parts.append((0.0, 0.0, label, h))
        tw = w0
        if tol is not None:
            if isinstance(tol, (int, float)):
                t = f"±{_fmt(tol)}"
                parts.append((w0 + 0.4, 0.0, t, h))
                tw = w0 + 0.4 + self._tw(t, h)
            else:
                up, lo = tol
                th = h * 0.7
                t1 = ("+" if up > 0 else "") + _fmt(up) if up else "0"
                t2 = ("+" if lo > 0 else "") + _fmt(lo) if lo else "0"
                parts += [(w0 + 0.6, h * 0.62, t1, th), (w0 + 0.6, -h * 0.18, t2, th)]
                tw = w0 + 0.6 + max(self._tw(t1, th), self._tw(t2, th))
        return [(x - tw / 2, y, s, hz) for x, y, s, hz in parts], tw

    def _place_parts(self, parts, tw, anchor, ur):
        up = _rot90(ur)
        rot = _ang(ur)
        out = []
        for dx, dy, s, h in parts:
            p = anchor + ur * dx + up * dy
            out += self._text(p, s, h, "l", "b", rot, halo="thin")
        return out

    # ---- border / title block
    def _border(self):
        W, H = self.W, self.H
        l, b, r, t = self.margins
        fx0, fy0, fx1, fy1 = l, b, W - r, H - t
        out = [_E("poly", "frame", pts=_rect_pts(fx0, fy0, fx1, fy1), closed=True, w=self.thick * 1.4)]
        if self.show_zones:
            ox0, oy0, ox1, oy1 = l - 5, b - 5, W - r + 5, H - t + 5
            out.append(_E("poly", "thin", pts=_rect_pts(ox0, oy0, ox1, oy1), closed=True, w=self.thin))
            nc = max(2, int(round((fx1 - fx0) / 100.0)) * 2)        # even counts: centring marks on lines
            nr = max(2, int(round((fy1 - fy0) / 100.0)) * 2)
            hz = 3.0
            for i in range(nc + 1):
                x = fx0 + (fx1 - fx0) * i / nc
                if 0 < i < nc:
                    out.append(_E("line", "thin", p=(x, oy0, x, fy0), w=self.thin))
                    out.append(_E("line", "thin", p=(x, fy1, x, oy1), w=self.thin))
                if i < nc:
                    xc = x + (fx1 - fx0) / nc / 2
                    for yc in ((oy0 + fy0) / 2, (fy1 + oy1) / 2):
                        out += self._text(_v(xc, yc), str(i + 1), hz, "m", "m", color="muted")
            for j in range(nr + 1):
                y = fy1 - (fy1 - fy0) * j / nr
                if 0 < j < nr:
                    out.append(_E("line", "thin", p=(ox0, y, fx0, y), w=self.thin))
                    out.append(_E("line", "thin", p=(fx1, y, ox1, y), w=self.thin))
                if j < nr:
                    yc = y - (fy1 - fy0) / nr / 2
                    for xc in ((ox0 + fx0) / 2, (fx1 + ox1) / 2):
                        out += self._text(_v(xc, yc), chr(65 + j), hz, "m", "m", color="muted")
            cw = self.thick * 1.4                                       # centring marks
            mx, my = (fx0 + fx1) / 2, (fy0 + fy1) / 2
            out += [_E("line", "frame", p=(mx, oy0 - 2, mx, fy0 + 5), w=cw),
                    _E("line", "frame", p=(mx, oy1 + 2, mx, fy1 - 5), w=cw),
                    _E("line", "frame", p=(ox0 - 2, my, fx0 + 5, my), w=cw),
                    _E("line", "frame", p=(ox1 + 2, my, fx1 - 5, my), w=cw)]
        return out

    def _title_block(self):
        x0, y0, x1, y1 = self._tb
        I = self.info
        out = []
        th = self.thin

        def cell(cx0, cy0, cx1, cy1, label, value, vh=3.5, bold=False, align="l"):
            out.append(_E("poly", "thin", pts=_rect_pts(cx0, cy0, cx1, cy1), closed=True, w=th))
            if label:
                out.extend(self._text(_v(cx0 + 1.2, cy1 - 1.2), label, 1.8, "l", "t", color="muted"))
            if value:
                v = str(value)
                maxw = cx1 - cx0 - 3.0
                hh = vh
                while hh > 1.8 and self._tw(v, hh, bold) > maxw:
                    hh *= 0.92
                yy = cy0 + (cy1 - cy0 - (2.6 if label else 0)) / 2
                if align == "m":
                    out.extend(self._text(_v((cx0 + cx1) / 2, yy), v, hh, "m", "m", bold=bold))
                else:
                    out.extend(self._text(_v(cx0 + 2.0, yy), v, hh, "l", "m", bold=bold))

        # bottom row (16): projection | scale | size | drawing no | rev | sheet
        r0, r1, r2, r3 = y0, y0 + 15, y0 + 30, y1
        xs = [x0, x0 + 30, x0 + 50, x0 + 70, x0 + 150, x0 + 162, x1]
        cell(xs[0], r0, xs[1], r1, "PROJECTION", None)
        out.extend(_projection_symbol(_v((xs[0] + xs[1]) / 2, r0 + 6.2), 16.0, self.projection, th, self.thick))
        cell(xs[1], r0, xs[2], r1, "SCALE", self.scale_text, 3.5, True, "m")
        cell(xs[2], r0, xs[3], r1, "SIZE", self.size_name, 3.5, True, "m")
        cell(xs[3], r0, xs[4], r1, "DRAWING NO.", I["drawing_no"], 5.0, True)
        cell(xs[4], r0, xs[5], r1, "REV", I["revision"], 5.0, True, "m")
        cell(xs[5], r0, xs[6], r1, "SHEET", I["sheet_no"], 3.5, False, "m")
        # middle row (16): drawn/date, checked/approved | title
        cell(x0, r1 + 7.5, x0 + 35, r2, "DRAWN", I["author"], 2.8)
        cell(x0 + 35, r1 + 7.5, x0 + 70, r2, "DATE", I["date"], 2.8)
        cell(x0, r1, x0 + 35, r1 + 7.5, "CHECKED", I["checked"], 2.8)
        cell(x0 + 35, r1, x0 + 70, r1 + 7.5, "APPROVED", I["approved"], 2.8)
        cell(x0 + 70, r1, x1, r2, (I["company"] or "TITLE").upper() if I["company"] else "TITLE", I["title"], 6.0,
             True)
        # top row (10): units | tolerances | material | finish / mass  (or custom ``fields``)
        top = self.fields or [("UNITS", I["units"]), ("GENERAL TOLERANCES", I["tolerances"]),
                              ("MATERIAL", I["material"]), ("MASS" if I["mass"] else "FINISH", I["mass"] or I["finish"])]
        wts = [30, 60, 60, 30] if len(top) == 4 else [180.0 / len(top)] * len(top)
        xx = x0
        for (lab, val), wv in zip(top, wts):
            cell(xx, r2, min(xx + wv, x1), r3, str(lab).upper(), val, 2.8)
            xx += wv
        out.append(_E("poly", "visible", pts=_rect_pts(x0, y0, x1, y1), closed=True, w=self.thick))
        if self._parts:
            out += self._parts_table(x0, y1, x1)
        return out

    def _parts_table(self, x0, y0, x1):
        p = self._parts
        cols = list(p["columns"])
        W = x1 - x0
        widths = p["widths"] or ([14, 14] + [W - 28 - 50, 50] if len(cols) == 4 else [W / len(cols)] * len(cols))
        widths = list(widths)
        widths[-1] += W - sum(widths)
        rh = p["row_h"]
        out = []
        rows = [cols] + list(p["rows"])
        for i, row in enumerate(rows):
            yb = y0 + i * rh
            x = x0
            for j, wv in enumerate(widths):
                out.append(_E("poly", "thin" if i else "visible", pts=_rect_pts(x, yb, x + wv, yb + rh), closed=True,
                              w=self.thin if i else self.thin * 1.4))
                val = row[j] if j < len(row) else ""
                hh = 2.5 if i == 0 else 3.0
                al = "m" if (j < 2 and len(cols) == 4) else "l"
                px = x + wv / 2 if al == "m" else x + 2.0
                out += self._text(_v(px, yb + rh / 2), val, hh, al, "m", bold=(i == 0))
                x += wv
        return out

    # ---- automatic view placement
    def _frame_ok(self, b, obstacles):
        """True when box ``b`` lies inside the drawing frame and clear of the sheet-level ``obstacles``."""
        l, bt, r, t = self.margins
        if b[0] < l + 3.9 or b[1] < bt + 3.9 or b[2] > self.W - r - 3.9 or b[3] > self.H - t - 3.9:
            return False
        for ob in obstacles:
            if min(b[2], ob[2]) - max(b[0], ob[0]) > 0.05 and min(b[3], ob[3]) - max(b[1], ob[1]) > 0.05:
                return False
        return True

    def _place_views(self, fin, check_all=False):
        """Resolve projected-view gaps and centre views created without an origin (sets ``View._shift``).
        Returns whether the automatically placed views (all views with ``check_all``) fit the sheet."""
        views = self.views
        bb = {v: _ents_bbox(fin[v], self) for v in views}
        for v in views:
            v._shift = [0.0, 0.0]
        done = set()

        def resolve(v, depth=0):
            if v in done or depth > 64:
                return
            if v._follow is not None:
                base, off, gap = v._follow
                resolve(base, depth + 1)
                sx, sy = base._shift
                b0, b1 = bb.get(base), bb.get(v)
                if gap is not None and b0 is not None and b1 is not None:
                    if off[0] > 0:
                        sx += b0[2] + gap - b1[0]
                    elif off[0] < 0:
                        sx += b0[0] - gap - b1[2]
                    elif off[1] > 0:
                        sy += b0[3] + gap - b1[1]
                    else:
                        sy += b0[1] - gap - b1[3]
                v._shift = [sx, sy]
            done.add(v)

        for v in views:
            resolve(v)

        def root(v):
            seen = 0
            while v._follow is not None and seen < 64:
                v, seen = v._follow[0], seen + 1
            return v

        groups = {}
        for v in views:
            groups.setdefault(root(v), []).append(v)

        def group_bb(members):
            boxes = [(b[0] + m._shift[0], b[1] + m._shift[1], b[2] + m._shift[0], b[3] + m._shift[1])
                     for m in members for b in [bb.get(m)] if b is not None]
            if not boxes:
                return None
            B = np.array(boxes)
            return (B[:, 0].min(), B[:, 1].min(), B[:, 2].max(), B[:, 3].max())

        def move(members, dx, dy):
            for m in members:
                m._shift = [m._shift[0] + dx, m._shift[1] + dy]

        obstacles = []
        if self._tb:
            x0, y0, x1, y1 = self._tb
            obstacles.append((x0, y0, x1, y1 + self._parts_height()))
        for ents in self._groups.values():
            b = _ents_bbox(ents, self)
            if b is not None:
                obstacles.append(b)
        sheet_obs = list(obstacles)
        fixed = [r for r in groups if r._auto is None]
        for r in fixed:
            b = group_bb(groups[r])
            if b is not None:
                obstacles.append(b)
        for r in groups:                                     # centred on a given sheet point
            if isinstance(r._auto, tuple):
                b = group_bb(groups[r])
                if b is not None:
                    move(groups[r], r._auto[0] - (b[0] + b[2]) / 2, r._auto[1] - (b[1] + b[3]) / 2)
                    obstacles.append(group_bb(groups[r]))
        autos = [r for r in groups if r._auto is True and group_bb(groups[r]) is not None]
        if not autos:
            if not check_all:
                return True
            return all(self._frame_ok(group_bb(g), sheet_obs) for g in groups.values() if group_bb(g) is not None)
        x = 0.0                                              # several auto groups: one row, left to right
        for r in autos:
            b = group_bb(groups[r])
            move(groups[r], x - b[0], -(b[1] + b[3]) / 2)
            x += (b[2] - b[0]) + 20.0
        members = [m for r in autos for m in groups[r]]
        b = group_bb(members)
        px, py = self._free_spot(b[2] - b[0], b[3] - b[1], obstacles)
        move(members, px - b[0], py - b[1])
        fits = self._frame_ok(group_bb(members), obstacles)
        if check_all:
            fits = fits and all(self._frame_ok(group_bb(g), sheet_obs) for r, g in groups.items()
                                if r._auto is not True and group_bb(g) is not None)
        return fits

    def _free_spot(self, w, h, obstacles):
        """Lower-left corner for a w x h box inside the frame, as far as possible from the frame edges and from
        ``obstacles`` (centre of the largest free area)."""
        l, b, r, t = self.margins
        fx0, fy0, fx1, fy1 = l + 4.0, b + 4.0, self.W - r - 4.0, self.H - t - 4.0
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        if w > fx1 - fx0 or h > fy1 - fy0:
            return cx - w / 2, cy - h / 2
        X, Y = np.meshgrid(np.arange(fx0, fx1 - w + 1e-9, 1.0), np.arange(fy0, fy1 - h + 1e-9, 1.0))
        clear = np.minimum.reduce([X - fx0, fx1 - (X + w), Y - fy0, fy1 - (Y + h)])
        overlap = np.zeros_like(X)
        for ob in obstacles:
            dx = np.maximum(np.maximum(ob[0] - (X + w), X - ob[2]), 0.0)
            dy = np.maximum(np.maximum(ob[1] - (Y + h), Y - ob[3]), 0.0)
            clear = np.minimum(clear, np.hypot(dx, dy))
            ox = np.clip(np.minimum(X + w, ob[2]) - np.maximum(X, ob[0]), 0.0, None)
            oy = np.clip(np.minimum(Y + h, ob[3]) - np.maximum(Y, ob[1]), 0.0, None)
            overlap += ox * oy
        score = np.where(overlap > 0, -overlap, clear) - 1e-3 * np.hypot(X + w / 2 - cx, Y + h / 2 - cy)
        k = int(np.argmax(score))
        return float(X.flat[k]), float(Y.flat[k])

    # ---- output
    def _layout(self):
        """Finalize every view at the current scale and place them: ({view: paper entities}, fits)."""
        global _BUSY
        self._build_scale_bars()
        fin = {}
        for v in self.views:
            _BUSY.add(v)
            try:
                geom, anno = v._finalize()
            finally:
                _BUSY.discard(v)
            fin[v] = [_map_entity(e, v) for e in geom] + anno
        fits = self._place_views(fin, check_all=self._scale_mode == "auto" or (
            self._scale_mode == "plans" and self._is_plan()))
        return fin, fits

    def _is_plan(self):
        return any(v._walls for v in self.views)

    def _set_scale(self, f):
        """Settle the sheet on scale factor ``f``: views that follow the sheet are redrawn at it."""
        self.s = f
        self.scale_text = _scale_text(f)
        self.terminator = self._terminator()
        for v in self.views:
            if v._sheet_scale and abs(v.s - f) > 1e-12:
                v.s = f
                v._replay()

    def _fit_scale(self):
        """Layout at a scale that fits: the given scale when it does, else the largest standard one that does."""
        plan = self._scale_mode == "plans" and self._is_plan()
        searching = self._scale_mode == "auto" or plan
        if not searching:
            fin, fits = self._layout()
            if fits or not any(v._sheet_scale for v in self.views):
                return fin
        elif not any(v._sheet_scale for v in self.views):
            return self._layout()[0]
        if plan:
            cands = [c for c in _ARCH_SCALES]
        elif self._scale_mode == "auto":
            cands = [c for c in _STD_SCALES]
        else:
            cands = [c for c in _STD_SCALES if c < self.s - 1e-12]
        start, start_text = self.s, self.scale_text
        lo, hi = 0, len(cands) - 1                      # cands run from large to small; the last one fits if any does
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            self._set_scale(cands[mid])
            _, ok = self._layout()
            if ok:
                best, hi = mid, mid - 1
            else:
                lo = mid + 1
        if best is None:
            self._set_scale(start)
            self.scale_text = start_text
            names = ", ".join(_scale_text(c) for c in cands[-2:])
            raise ValueError(f"the views do not fit the {self.size_name} sheet at any standard scale down to "
                             f"{_scale_text(cands[-1])}: split them over more sheets or use a bigger size "
                             f"(tried {names})")
        chosen = cands[best]
        if not searching and self.fit == "error":
            self._set_scale(start)
            self.scale_text = start_text
            raise ValueError(f"the views do not fit the {self.size_name} sheet at {start_text}; the "
                             f"largest standard scale that fits is {_scale_text(chosen)} "
                             f"(use Sheet(scale='{_scale_text(chosen)}'))")
        self._set_scale(chosen)
        if not searching:
            self.fit_report = (f"the views do not fit at {start_text}; the largest standard scale that "
                               f"fits is {_scale_text(chosen)} and the sheet uses it")
            if not getattr(self, "_fit_warned", False):
                import warnings
                warnings.warn(self.fit_report, stacklevel=4)
                self._fit_warned = True
        else:
            self.fit_report = f"scale {_scale_text(chosen)}: the largest standard scale that fits the sheet"
        fin, _ = self._layout()
        return fin

    def _resolve_labels(self, groups):
        """Keep movable labels (cutting-plane letters, room names, view titles) off dimension values, other labels,
        lines and furniture: a label that collides moves to the nearest free spot, and shrinks when none is near."""
        mov = {}
        for gi, (_, ents) in enumerate(groups):
            for ei, e in enumerate(ents):
                if e["t"] == "text" and e.get("mv") and str(e["s"]).strip():
                    mov.setdefault((gi, e.get("ov"), e.get("gid")), []).append(ei)
        if not mov:
            return groups
        groups = [(n, list(ents)) for n, ents in groups]
        skip = {"hatch", "center", "phantom", "frame"}
        pts, lts, fur, tbox, owner = [], [], [], [], []
        for gi, (name, ents) in enumerate(groups):
            if name == "border":
                continue
            for ei, e in enumerate(ents):
                if e["t"] == "text":
                    if str(e["s"]).strip():
                        c = _text_corners(e, self, pad=0.5, desc=0.15)
                        tbox.append((*c.min(0), *c.max(0)))
                        owner.append((gi, e.get("ov"), e.get("gid")) if e.get("mv") else None)
                    continue
                if e["lt"] in skip:
                    continue
                P = _sample_ents([e], step=0.7)
                if len(P):
                    pts.append(P)
                    lts.append(np.full(len(P), e["lt"], dtype=object))
                if e["lt"] == "furniture" and e["t"] in ("poly", "circle") and (e["t"] == "circle" or e.get("closed")):
                    Q = _entity_pts(e)
                    fur.append((*Q.min(0), *Q.max(0)))
        if self._tb:
            tbox.append(self._tb)
            owner.append(None)
            tbox[-1] = (self._tb[0], self._tb[1], self._tb[2], self._tb[3] + self._parts_height())
        P = np.concatenate(pts) if pts else np.zeros((0, 2))
        LT = np.concatenate(lts) if lts else np.zeros(0, dtype=object)
        T = np.array(tbox) if tbox else np.zeros((0, 4))
        F = np.array(fur) if fur else np.zeros((0, 4))
        for key, idx in mov.items():
            gi = key[0]
            ents = [groups[gi][1][i] for i in idx]
            kind = ents[0]["mv"]
            drop = {"cutting"} if kind == "cut" else set()
            keep_t = np.array([o != key for o in owner], bool) if owner else np.zeros(0, bool)
            Tk = T[keep_t] if len(T) else T
            Pk = P[~np.isin(LT, list(drop))] if len(P) and drop else P
            cen = np.mean([e["p"] for e in ents], axis=0)

            def box_of(offset, f):
                bs = []
                for e in ents:
                    e2 = dict(e, h=e["h"] * f, p=tuple(cen + (np.asarray(e["p"]) - cen) * f + offset))
                    c = _text_corners(e2, self, pad=0.3, desc=0.15)
                    bs.append((*c.min(0), *c.max(0)))
                b = np.array(bs)
                return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()

            def hits(b):
                n = 0.0
                if len(Tk):
                    ox = np.clip(np.minimum(Tk[:, 2], b[2]) - np.maximum(Tk[:, 0], b[0]), 0, None)
                    oy = np.clip(np.minimum(Tk[:, 3], b[3]) - np.maximum(Tk[:, 1], b[1]), 0, None)
                    n += float((ox * oy).sum()) * 4.0
                if len(F):
                    ox = np.clip(np.minimum(F[:, 2], b[2]) - np.maximum(F[:, 0], b[0]), 0, None)
                    oy = np.clip(np.minimum(F[:, 3], b[3]) - np.maximum(F[:, 1], b[1]), 0, None)
                    n += float((ox * oy).sum()) * 2.0
                if len(Pk):
                    m = (Pk[:, 0] > b[0]) & (Pk[:, 0] < b[2]) & (Pk[:, 1] > b[1]) & (Pk[:, 1] < b[3])
                    n += float(m.sum()) * 0.7
                return n

            R = 6.0 if kind == "room" else 16.0
            cand = [(0.0, 0.0)]
            for r in np.arange(1.0, R + 1e-9, 1.0):
                for a in range(0, 360, 20):
                    cand.append((r * math.cos(math.radians(a)), r * math.sin(math.radians(a))))
            best = None
            for f in (1.0, 0.85, 0.7):
                for dx, dy in cand:
                    h = hits(box_of(np.array([dx, dy]), f))
                    if h <= 0.0:
                        best = (f, dx, dy)
                        break
                if best:
                    break
            if best is None or (best[1] == 0.0 and best[2] == 0.0 and best[0] == 1.0):
                continue
            f, dx, dy = best
            for i in idx:
                e = groups[gi][1][i]
                q = cen + (np.asarray(e["p"]) - cen) * f + np.array([dx, dy])
                groups[gi][1][i] = dict(e, h=e["h"] * f, p=(float(q[0]), float(q[1])))
        return groups

    def _collect(self):
        """[(group, [paper entities])] in drawing order."""
        groups = [("border", self._border())]
        fin = self._fit_scale()
        for v in self.views:
            dx, dy = v._shift
            ents = fin[v] if (dx, dy) == (0.0, 0.0) else [_shift_entity(e, dx, dy) for e in fin[v]]
            groups.append((v.name, ents))
        for g, ents in self._groups.items():
            groups.append((g, list(ents)))
        if self._tb:
            groups.append(("title block", self._title_block()))
        groups = self._resolve_labels(groups)
        self._last_groups = groups
        return groups

    def to_drawing(self, style=None, min_px=None, px_per_mm=None):
        """The sheet as a ``pixelcanvas.vector.Drawing`` in mm (one group per view / sheet element)."""
        st = _STYLES[style or self.style]
        ppu = px_per_mm or self.dpi / 25.4
        dwg = _vec.Drawing(self.W, self.H, units="mm", px_per_unit=ppu, background=st["paper"],
                           name="sheet")
        minw = (min_px / ppu) if min_px else 0.0
        for gname, ents in self._collect():
            with dwg.group(gname):
                _emit(dwg, ents, self, st, minw)
        return dwg

    def draw(self, cv=None, log=None, style=None):
        """Raster the sheet → Canvas. ``cv=None`` makes a new Canvas at ``dpi``; otherwise the sheet is fitted
        into ``cv``. ``style``: 'white' / 'paper' / 'blueprint' (default: the sheet's style). One layer per view."""
        from .canvas import Canvas
        st = _STYLES[style or self.style]
        if cv is None:
            ppu = self.dpi / 25.4
            cv = Canvas(int(round(self.W * ppu)), int(round(self.H * ppu)), st["paper"])
        k = min(cv.w / self.W, cv.h / self.H)
        dwg = self.to_drawing(style, min_px=0.85, px_per_mm=k)
        dwg.draw(cv, log=log)
        return cv

    def save_svg(self, path, style=None):
        """Write the sheet as SVG (width / height in mm). Returns the path."""
        return self.to_drawing(style).save_svg(path)

    def save_dxf(self, path):
        """Write the sheet as ASCII DXF R12 (layers per line type; LINE, CIRCLE, ARC, TEXT, SOLID entities).
        Returns the path."""
        txt = _dxf(self)
        with open(path, "w", encoding="cp1252", errors="replace", newline="\r\n") as fh:
            fh.write(txt)
        return path


_BUSY = set()

# standard drawing scales, largest first (ISO 5455 plus the common architectural 1:25 / 1:75 / 1:125 ...)
_STD_SCALES = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 1 / 2, 1 / 5, 1 / 10, 1 / 20, 1 / 25, 1 / 50, 1 / 75, 1 / 100,
               1 / 125, 1 / 150, 1 / 200, 1 / 250, 1 / 500, 1 / 1000]
_ARCH_SCALES = [c for c in _STD_SCALES if 1 / 500 - 1e-12 <= c <= 1 / 10 + 1e-12]


def _projection_symbol(c, w, method, thin, thick):
    """First / third angle projection symbol (ISO 5456-2), centred on c, about w mm wide."""
    out = []
    d = w / 16.0
    r_big, r_small = 3.0 * d, 1.75 * d
    L = 5.0 * d
    gap = 2.5 * d
    total = L + gap + 2 * r_big
    x0 = c[0] - total / 2
    if method == "first":
        cone_x0, circ_c = x0, x0 + L + gap + r_big
    else:
        circ_c, cone_x0 = x0 + r_big, x0 + 2 * r_big + gap
    cy = c[1]
    cone = [(cone_x0, cy - r_small), (cone_x0 + L, cy - r_big), (cone_x0 + L, cy + r_big), (cone_x0, cy + r_small)]
    out.append(_E("poly", "visible", pts=cone, closed=True, w=thick * 0.7))
    out.append(_E("circle", "visible", c=(circ_c, cy), r=r_big, w=thick * 0.7))
    out.append(_E("circle", "visible", c=(circ_c, cy), r=r_small, w=thick * 0.7))
    out.append(_E("line", "center", p=(x0 - d, cy, x0 + total + d, cy), w=thin * 0.7, dash=(3 * d, 0.6 * d, 0.6 * d, 0.6 * d)))
    out.append(_E("line", "center", p=(circ_c, cy - r_big - d, circ_c, cy + r_big + d), w=thin * 0.7,
                  dash=(3 * d, 0.6 * d, 0.6 * d, 0.6 * d)))
    return out


def _clip_to_circle(geom, c, r):
    """Clip model-space geometry entities to a circle (for detail views)."""
    C = _v(*c)
    out = []
    ring = _circle_ring(c[0], c[1], r, 128)
    E = _edges_of([ring])

    def clip_line(p, q, lt, w):
        d = q - p
        f = p - C
        A, B, Cc = d @ d, 2 * f @ d, f @ f - r * r
        disc = B * B - 4 * A * Cc
        if A < 1e-18 or disc <= 0:
            return
        sq = math.sqrt(disc)
        t0, t1 = max((-B - sq) / (2 * A), 0.0), min((-B + sq) / (2 * A), 1.0)
        if t1 > t0:
            a, b = p + d * t0, p + d * t1
            out.append(_E("line", lt, p=(*a, *b), w=w))

    def clip_arc(cc, rr, a0, a1, lt, w):
        cc = _v(*cc)
        dd = math.hypot(*(cc - C))
        cuts = [a0, a1]
        if dd > 1e-12 and abs(r - rr) < dd < r + rr:
            a = (rr * rr - r * r + dd * dd) / (2 * dd)
            base = _ang(C - cc)
            delta = math.degrees(math.acos(float(np.clip(a / rr, -1, 1))))
            for ang in (base - delta, base + delta):
                while ang < a0:
                    ang += 360
                while ang > a0 + 360:
                    ang -= 360
                if a0 < ang < a1:
                    cuts.append(ang)
        cuts.sort()
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            m = cc + _dir((s0 + s1) / 2) * rr
            if math.hypot(*(m - C)) < r and s1 - s0 > 1e-6:
                out.append(_E("arc", lt, c=tuple(cc), r=rr, a=(s0, s1), w=w))

    for e in geom:
        t, lt, w = e["t"], e["lt"], e.get("w")
        if t == "line":
            x1, y1, x2, y2 = e["p"]
            clip_line(_v(x1, y1), _v(x2, y2), lt, w)
        elif t == "poly":
            P = [np.asarray(p, float) for p in e["pts"]]
            if e.get("closed"):
                P = P + [P[0]]
            for a, b in zip(P[:-1], P[1:]):
                clip_line(a, b, lt, w)
        elif t == "circle":
            clip_arc(e["c"], e["r"], 0.0, 360.0, lt, w)
        elif t == "arc":
            clip_arc(e["c"], e["r"], e["a"][0], e["a"][1], lt, w)
        elif t == "solid":
            rings = e.get("rings") or [e["pts"]]
            kept = []
            for rg in rings:
                q = _clip_poly_convex(rg, ring)
                if len(q) >= 3:
                    kept.append(q)
            if kept:
                out.append(_E("solid", lt, rings=kept, fill=e.get("fill")))
    return out


def _map_entity(e, v):
    """Model-space entity of view v → paper-space copy."""
    s, ox, oy = v.s, v._ox, v._oy
    t = e["t"]
    n = dict(e)
    if t == "line":
        x1, y1, x2, y2 = e["p"]
        n["p"] = (ox + s * x1, oy + s * y1, ox + s * x2, oy + s * y2)
    elif t == "poly":
        n["pts"] = [(ox + s * x, oy + s * y) for x, y in e["pts"]]
    elif t in ("circle", "arc", "dot"):
        n["c"] = (ox + s * e["c"][0], oy + s * e["c"][1])
        n["r"] = e["r"] * s
    elif t == "solid":
        if "rings" in e:
            n["rings"] = [[(ox + s * x, oy + s * y) for x, y in r] for r in e["rings"]]
        else:
            n["pts"] = [(ox + s * x, oy + s * y) for x, y in e["pts"]]
    elif t == "text":
        n["p"] = (ox + s * e["p"][0], oy + s * e["p"][1])
    return n


# --------------------------------------------------------------------------------------------- raster / SVG emit

def _dash_fit(pattern, L, closed=False, center=False):
    """Scale a dash pattern so an open line starts and ends with a full dash (closed: whole periods)."""
    return _vec._fit_dash(pattern, L, closed, center)        # None: too short, draw continuous


def _path_len(pts):
    P = np.asarray(pts, float)
    return float(np.sum(np.hypot(*(P[1:] - P[:-1]).T)))


def _emit(dwg, ents, sheet, st, minw):
    H = sheet.H
    ink, muted, paper = st["ink"], st["muted"], st["paper"]

    def width(e):
        w = e.get("w")
        if w is None:
            kind = LINE_TYPES[e["lt"]][0]
            w = {"thick": sheet.thick, "thin": sheet.thin, "frame": sheet.thick * 1.4}[kind]
        return max(w, minw)

    def dash_of(e, L, closed=False):
        if e.get("dash") is not None:
            return e["dash"]
        pat = LINE_TYPES[e["lt"]][1]
        if pat is None:
            return None
        return _dash_fit(pat, L, closed, center=e["lt"] in ("center", "cutting", "phantom"))

    def one(e):
        t = e["t"]
        c = e.get("color")
        col = muted if c == "muted" else (c if c else ink)
        if t == "line":
            x1, y1, x2, y2 = e["p"]
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 1e-9:
                return
            ds = dash_of(e, L)
            dwg.line(x1, H - y1, x2, H - y2, stroke=col, stroke_width=width(e), dash=ds,
                     cap="butt" if ds else "round")
        elif t == "poly":
            pts = [(x, H - y) for x, y in e["pts"]]
            closed = e.get("closed")
            L = _path_len(pts + ([pts[0]] if closed else []))
            ds = dash_of(e, L, closed)
            fill = paper if e.get("fill") == "paper" else (e.get("fill") if e["lt"] == "furniture" else None)
            kw = dict(stroke=col, stroke_width=width(e), dash=ds, join="round", cap="butt" if ds else "round")
            if closed:
                dwg.polygon(pts, fill=fill, **kw)
            else:
                dwg.polyline(pts, **kw)
        elif t == "circle":
            cx, cy = e["c"]
            L = 2 * math.pi * e["r"]
            ds = dash_of(e, L, True)
            fill = paper if e.get("fill") == "paper" else (e.get("fill") if e["lt"] == "furniture" else None)
            dwg.circle(cx, H - cy, e["r"], fill=fill, stroke=col, stroke_width=width(e), dash=ds)
        elif t == "arc":
            cx, cy = e["c"]
            a0, a1 = e["a"]
            L = math.radians(a1 - a0) * e["r"]
            ds = dash_of(e, L)
            dwg.arc(cx, H - cy, e["r"], -a0, -a1, stroke=col, stroke_width=width(e), dash=ds,
                    cap="butt" if ds else "round")
        elif t == "dot":
            cx, cy = e["c"]
            dwg.circle(cx, H - cy, e["r"], fill=col, stroke=None)
        elif t == "solid":
            if e.get("fill") in ("poche", "floor"):
                return
            rings = e.get("rings") or [e["pts"]]
            for ring in rings:
                dwg.polygon([(x, H - y) for x, y in ring], fill=col, stroke=col, stroke_width=minw or 0.05,
                            join="round")
        elif t == "text":
            h = e["h"]
            rot = e["rot"]
            up = _dir(rot + 90)
            shift = {"b": 0.0, "m": h / 2, "t": h}[e["va"]]
            x, y = e["p"][0] - up[0] * shift, e["p"][1] - up[1] * shift
            size = h / _cap_ratio(sheet.font, e["bold"])
            halo = e.get("halo")
            if halo == "box":                                # knock-out box (labels over furniture / hatching)
                box = _text_corners(e, sheet, pad=0.9, desc=0.15)
                dwg.polygon([(bx, H - by) for bx, by in box], fill=e.get("bg") or paper, stroke=None)
                halo = None
            hw = None if not halo else size * (0.09 if halo == "thin" else 0.18)
            dwg.text(x, H - y, e["s"], size=size, fill=col, anchor=e["ha"] + "s", family=sheet.font,
                     weight="bold" if e["bold"] else "normal", rotate=-rot,
                     halo=paper if halo else None, halo_width=hw)

    fills = [e for e in ents if e["t"] == "solid" and e.get("fill") == "poche"]
    for e in ents:                                           # floor tints under everything
        if e["t"] == "solid" and e.get("fill") == "floor":
            for ring in e["rings"]:
                dwg.polygon([(x, H - y) for x, y in ring], fill=e["color"], stroke=None)
    for e in ents:                                           # furniture first: walls cover it
        if e["lt"] == "furniture":
            one(e)
    for e in fills:                                          # wall poche under every other line
        d = None
        for ring in e["rings"]:
            pts = [(x, H - y) for x, y in ring]
            if d is None:
                d = dwg.path(fill=st["poche"], stroke=None)
            d.move_to(*pts[0])
            for p in pts[1:]:
                d.line_to(*p)
            d.close()
    for e in ents:
        if e["lt"] != "furniture" and e["t"] != "text":
            one(e)
    for e in ents:                                           # text last: its halo keeps it readable
        if e["lt"] != "furniture" and e["t"] == "text":
            one(e)


# --------------------------------------------------------------------------------------------- DXF

def _dxf_text(s):
    return str(s).replace("Ø", "%%c").replace("±", "%%p").replace("°", "%%d").replace("⌴", "")


def _dxf(sheet):
    out = []

    def g(code, val):
        if isinstance(val, float):
            val = f"{val:.6f}".rstrip("0").rstrip(".") if "." in f"{val:.6f}" else f"{val:.6f}"
        out.append(f"{code:>3}\n{val}")

    groups = sheet._collect()
    layers = {}
    for _, ents in groups:
        for e in ents:
            layers.setdefault(e["lt"].upper(), e["lt"])
    g(0, "SECTION"), g(2, "HEADER")
    g(9, "$ACADVER"), g(1, "AC1009")
    g(9, "$DWGCODEPAGE"), g(3, "ANSI_1252")
    g(9, "$INSBASE"), g(10, 0.0), g(20, 0.0), g(30, 0.0)
    g(9, "$EXTMIN"), g(10, 0.0), g(20, 0.0), g(30, 0.0)
    g(9, "$EXTMAX"), g(10, float(sheet.W)), g(20, float(sheet.H)), g(30, 0.0)
    g(9, "$LIMMIN"), g(10, 0.0), g(20, 0.0)
    g(9, "$LIMMAX"), g(10, float(sheet.W)), g(20, float(sheet.H))
    g(9, "$LTSCALE"), g(40, 1.0)
    g(0, "ENDSEC")
    g(0, "SECTION"), g(2, "TABLES")
    ltypes = {"CONTINUOUS": ("Solid line", ())}
    for k, (wt, pat, name, col) in LINE_TYPES.items():
        if pat and name not in ltypes:
            desc = {"HIDDEN": "Hidden __ __ __", "CENTER": "Center ____ _ ____ _",
                    "PHANTOM": "Phantom ____ _ _ ____"}.get(name, name)
            ltypes[name] = (desc, pat)
    g(0, "TABLE"), g(2, "LTYPE"), g(70, len(ltypes))
    for name, (desc, pat) in ltypes.items():
        g(0, "LTYPE"), g(2, name), g(70, 0), g(3, desc), g(72, 65), g(73, len(pat)), g(40, float(sum(pat)))
        for i, v in enumerate(pat):
            g(49, float(v if i % 2 == 0 else -v))
    g(0, "ENDTAB")
    g(0, "TABLE"), g(2, "LAYER"), g(70, len(layers) + 1)
    g(0, "LAYER"), g(2, "0"), g(70, 0), g(62, 7), g(6, "CONTINUOUS")
    for L, lt in layers.items():
        wt, pat, name, col = LINE_TYPES[lt]
        g(0, "LAYER"), g(2, L), g(70, 0), g(62, col), g(6, name)
    g(0, "ENDTAB")
    g(0, "TABLE"), g(2, "STYLE"), g(70, 1)
    g(0, "STYLE"), g(2, "STANDARD"), g(70, 0), g(40, 0.0), g(41, 1.0), g(50, 0.0), g(71, 0), g(42, 3.5),
    g(3, "txt"), g(4, "")
    g(0, "ENDTAB")
    g(0, "ENDSEC")
    g(0, "SECTION"), g(2, "ENTITIES")

    def line(L, x1, y1, x2, y2):
        g(0, "LINE"), g(8, L), g(10, float(x1)), g(20, float(y1)), g(30, 0.0), g(11, float(x2)), g(21, float(y2)),
        g(31, 0.0)

    def solid(L, pts):
        pts = [tuple(map(float, p)) for p in pts]
        if len(pts) == 3:
            order = [pts[0], pts[1], pts[2], pts[2]]
        elif len(pts) == 4:
            order = [pts[0], pts[1], pts[3], pts[2]]
        else:
            for k in range(1, len(pts) - 1):
                solid(L, [pts[0], pts[k], pts[k + 1]])
            return
        g(0, "SOLID"), g(8, L)
        for i, (x, y) in enumerate(order):
            g(10 + i, x), g(20 + i, y), g(30 + i, 0.0)

    for _, ents in groups:
        for e in ents:
            L = e["lt"].upper()
            t = e["t"]
            if t == "line":
                line(L, *e["p"])
            elif t == "poly":
                P = list(e["pts"])
                if e.get("closed"):
                    P.append(P[0])
                for a, b in zip(P[:-1], P[1:]):
                    line(L, a[0], a[1], b[0], b[1])
            elif t in ("circle", "dot"):
                g(0, "CIRCLE"), g(8, L), g(10, float(e["c"][0])), g(20, float(e["c"][1])), g(30, 0.0),
                g(40, float(e["r"]))
            elif t == "arc":
                a0, a1 = e["a"]
                g(0, "ARC"), g(8, L), g(10, float(e["c"][0])), g(20, float(e["c"][1])), g(30, 0.0),
                g(40, float(e["r"])), g(50, float(a0 % 360)), g(51, float(a1 % 360) if (a1 - a0) < 360 else
                                                               float(a0 % 360 + 359.999))
            elif t == "solid":
                if e.get("fill") == "floor":                 # floor tints are raster / SVG colour only
                    continue
                for ring in (e.get("rings") or [e["pts"]]):
                    solid(L, ring)
            elif t == "text":
                h, rot = e["h"], e["rot"]
                up = _dir(rot + 90)
                shift = {"b": 0.0, "m": h / 2, "t": h}[e["va"]]
                x, y = e["p"][0] - up[0] * shift, e["p"][1] - up[1] * shift
                hj = {"l": 0, "m": 1, "r": 2}[e["ha"]]
                g(0, "TEXT"), g(8, L), g(10, float(x)), g(20, float(y)), g(30, 0.0), g(40, float(h)),
                g(1, _dxf_text(e["s"]))
                if rot:
                    g(50, float(rot % 360))
                if hj:
                    g(72, hj), g(11, float(x)), g(21, float(y)), g(31, 0.0)
    g(0, "ENDSEC")
    g(0, "EOF")
    return "\n".join(out) + "\n"


def parse_dxf(path):
    """Read an ASCII DXF back → dict(layers=[names], entities=[dict(type, layer, codes)]). Handy for tests and
    for checking what a CAD program will see."""
    with open(path, encoding="cp1252", errors="replace") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    pairs = [(int(lines[i].strip()), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
    ents, layers = [], []
    section = None
    cur = None
    for i, (c, v) in enumerate(pairs):
        if c == 0 and v == "SECTION":
            section = pairs[i + 1][1] if i + 1 < len(pairs) else None
            continue
        if section == "TABLES" and c == 0 and v == "LAYER":
            layers.append(pairs[i + 1][1])
        if section == "ENTITIES":
            if c == 0:
                if cur:
                    ents.append(cur)
                cur = None if v in ("ENDSEC", "EOF") else dict(type=v, layer=None, codes={})
            elif cur is not None:
                if c == 8:
                    cur["layer"] = v
                cur["codes"].setdefault(c, []).append(v)
    if cur:
        ents.append(cur)
    return dict(layers=layers, entities=ents)


# ---- scene manifest: views with their dimension counts; drawn sheets for render.py --dxf ----
# unknown options raise a helpful TypeError; colour / w / lw ... spellings are accepted
for _c in (Profile, View, Sheet):
    _strict_methods(_c)

from . import manifest as _manifest  # noqa: E402


def _record_calls(cls, skip=()):
    """Remember every top-level public call of a View so the view can be redrawn at another scale."""
    import functools

    def make(fn):
        @functools.wraps(fn)
        def w(self, *a, **k):
            if self._rec_depth:
                return fn(self, *a, **k)
            self._rec_depth = 1
            self._cur_obj = _OBJ_OF.get(fn.__name__)
            try:
                res = fn(self, *a, **k)
            finally:
                self._rec_depth = 0
                self._cur_obj = None
            if not self._replaying:
                self._calls.append((fn, a, k))
            return res
        return w

    for name, fn in list(cls.__dict__.items()):
        if name.startswith("_") or not callable(fn) or name in skip or isinstance(fn, (staticmethod, classmethod)):
            continue
        setattr(cls, name, make(fn))

_SHEETS = []


def last_sheets():
    """Sheets drawn (``Sheet.draw``) since the last ``clear_sheets()``, oldest first (render.py --dxf uses it)."""
    return list(_SHEETS)


def clear_sheets():
    """Forget the sheets recorded by ``last_sheets()``."""
    _SHEETS.clear()


def _remember_sheet(a):
    sh = a["self"]
    if sh not in _SHEETS:
        _SHEETS.append(sh)
        del _SHEETS[:-16]
    return a.get("cv")


_manifest.count_calls(View, ("dim", "dim_h", "dim_v", "dim_aligned", "dim_chain", "dim_baseline", "dim_radius",
                             "dim_diameter", "dim_angle", "dim_chamfer"), "_n_dims")
# ---- what a sheet drew, for the scene manifest (walls, doors, windows, furniture, rooms, holes, dimensions ...)
_ROOM_WORDS = {"bath": ["bathroom"], "wc": ["toilet", "bathroom"], "living": ["living room"],
               "lounge": ["living room"], "kitchen": ["kitchen"], "entry": ["entrance", "hall"],
               "hall": ["hallway", "corridor"], "balcony": ["balcony", "terrace"], "bedroom": ["bedroom"]}


def _room_alias(name):
    low = " ".join(str(name).lower().replace("/", " / ").replace("&", " / ").split())
    base = low.rstrip("0123456789 ").strip()
    out = ["room", "room name", "name", base, low]
    for part in base.split(" / "):
        part = part.strip()
        out.append(part)
        for w in part.split():
            out += _ROOM_WORDS.get(w, [])
    return [a for i, a in enumerate(out) if a and a not in out[:i] and a != low.strip()]


def _sheet_masks(sh, cv, ents_by_key):
    """{key: float mask (H, W)} rasterised from paper entities (mm) onto the canvas the sheet was drawn on."""
    from PIL import Image, ImageDraw
    W, H = sh.W, sh.H
    k = min(cv.w / W, cv.h / H)
    ox, oy = (cv.w - W * k) / 2, (cv.h - H * k) / 2

    def X(p):
        return (ox + k * p[0], oy + k * (H - p[1]))

    lw = max(2, int(round(0.3 * k)))
    out = {}
    for key, ents in ents_by_key.items():
        im = Image.new("L", (cv.w, cv.h), 0)
        d = ImageDraw.Draw(im)
        for e in ents:
            t = e["t"]
            try:
                if t == "line":
                    d.line([X(e["p"][:2]), X(e["p"][2:])], fill=255, width=lw)
                elif t == "poly":
                    pts = [X(p) for p in e["pts"]]
                    if e.get("closed") and (e["lt"] == "furniture" or e.get("o") == "door"):
                        d.polygon(pts, fill=255)
                    else:
                        d.line(pts + ([pts[0]] if e.get("closed") else []), fill=255, width=lw)
                elif t == "solid":
                    for ring in e.get("rings") or [e["pts"]]:
                        d.polygon([X(p) for p in ring], fill=255)
                elif t == "circle":
                    cx, cy = X(e["c"])
                    r = e["r"] * k
                    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255 if e.get("o") == "hole" else None,
                              outline=255, width=lw)
                elif t == "arc":
                    cx, cy = X(e["c"])
                    r = e["r"] * k
                    a0, a1 = e["a"]
                    d.arc([cx - r, cy - r, cx + r, cy + r], -a1, -a0, fill=255, width=lw)
                elif t == "text" and str(e["s"]).strip():
                    c = _text_corners(e, sh, pad=0.3, desc=0.15)
                    d.polygon([X(p) for p in c], fill=255)
            except Exception:                                 # noqa: BLE001 - a bad entity never breaks a render
                continue
        out[key] = np.asarray(im, np.float32) / 255.0
    return out


def _sheet_entries(sh, cv):
    """Manifest entries for everything the sheet drew: dicts (kind, name, mask, alias, meta) in drawing order."""
    groups = getattr(sh, "_last_groups", None)
    if not groups or cv is None or not hasattr(cv, "buf"):
        return []
    order, buckets, info = [], {}, {}

    def add(key, e, **meta):
        if key not in buckets:
            buckets[key] = []
            order.append(key)
            info[key] = dict(meta)
        buckets[key].append(e)

    holes = {}
    for gname, ents in groups:
        for e in ents:
            o = e.get("o")
            if o == "wall":
                add(("wall",), e, kind="object", alias=["walls", "partition", "partitions", "external wall"])
            elif o in ("door", "window"):
                add((o,), e, kind="symbol", alias=[o + "s"] + (["swing", "swings", "door swing"] if o == "door" else []))
                info[(o,)].setdefault("ids", set()).add((e.get("ov"), e.get("oi")))
                if e.get("sub") == "sliding":
                    info[(o,)]["alias"] = list(set(info[(o,)]["alias"]) | {"sliding door", "balcony door"})
            elif o == "furniture":
                add(("furniture", e["on"]), e, kind="furniture", alias=[e["on"].replace("_", " ") + "s", "furniture"])
                info[("furniture", e["on"])].setdefault("ids", set()).add((e.get("ov"), e.get("oid")))
            elif o == "hole":
                holes.setdefault(e.get("ov"), []).append(e)
            elif o == "chamfer":
                add(("chamfer",), e, kind="object", alias=["chamfers", "edge break"])
                info[("chamfer",)].setdefault("ids", set()).add((e.get("ov"), id(e)))
            elif o == "dimension":
                add(("dimension",), e, kind="annotation", alias=["dimensions", "dimension label", "overall dimensions",
                                                              "measurements", "dimension line"], see_through=True)
                if e["t"] == "text":
                    info[("dimension",)].setdefault("values", []).append(str(e["s"]))
            elif o == "room":
                add(("room", e.get("ov"), e.get("gid")), e, kind="room", name=e["on"], alias=_room_alias(e["on"]))
        if gname == "north arrow":
            for e in ents:
                add(("north arrow",), e, kind="symbol", alias=["compass", "north"])
        elif gname == "scale bar":
            for e in ents:
                add(("scale bar",), e, kind="symbol", alias=["scale"])
    for vname, hs in holes.items():                                    # a bore is the big hole among small ones
        circles = [h for h in hs if h["t"] == "circle"]
        if not circles:
            continue
        ds = sorted({round(h["d"], 3) for h in circles})
        big = ds[-1] if (len(ds) > 1 and ds[-1] >= 1.8 * ds[0]) or (len(ds) == 1 and ds[0] >= 40) else None
        for h in hs:
            is_bore = big is not None and abs(round(h["d"], 3) - big) < 1e-9
            nm = "bore" if is_bore else "hole"
            add((nm, vname), h, kind="object", name=nm,
                alias=(["bores", "through bore", "hole", "bore hole"] if is_bore else
                       ["holes", "mounting hole", "through hole"]))
            if h["t"] == "circle":
                info[(nm, vname)].setdefault("ids", set()).add(id(h))
    masks = _sheet_masks(sh, cv, buckets)
    out, merged = [], {}
    for key in order:
        m = dict(info[key])
        name = m.pop("name", None)
        ids = m.pop("ids", None)
        values = m.pop("values", None)
        kind = m.pop("kind")
        alias = m.pop("alias", [])
        meta = {}
        if key[0] == "dimension":
            meta["count"] = max(1, sum(int(v.__dict__.get("_n_dims", 0)) for v in sh.views))
            meta["values"] = (values or [])[:40]
        elif ids is not None and len(ids) > 1:
            meta["count"] = len(ids)
        if m.get("see_through"):
            meta["see_through"] = True
        if key[0] == "furniture":
            name = key[1].replace("_", " ")
        name = name or key[0]
        if key[0] in ("hole", "bore"):                                 # the same feature in several views: once
            prev = merged.get(name)
            if prev is not None and prev["meta"].get("count", 1) >= meta.get("count", 1):
                continue
            if prev is not None:
                out.remove(prev)
        e = dict(kind=kind, name=name, mask=masks[key], alias=alias, meta=meta)
        merged[name] = e
        out.append(e)
    small = [e for e in out if e["kind"] == "furniture" and float(e["mask"].sum()) < 0.0025 * cv.w * cv.h]
    if len(small) > 1:                                                 # small symbols read as one set of furniture
        mask = np.maximum.reduce([e["mask"] for e in small])
        kinds = [e["name"] for e in small]
        alias = sorted({a for e in small for a in e["alias"]} | set(kinds))
        count = sum(int(e["meta"].get("count", 1)) for e in small)
        at = out.index(small[0])
        out = [e for e in out if e not in small]
        out.insert(at, dict(kind="furniture", name="furniture", mask=mask, alias=alias, meta={"count": count,
                                                                                                   "kinds": kinds}))
    return out


def _after_sheet(cap, args, res):
    """Manifest hook of ``Sheet.draw``: the views and title block, then every nameable thing the sheet drew."""
    items = list(_manifest.after_sheet(cap, args, res) or [])
    sh = args.get("self")
    cv = cap.cv if cap.cv is not None else (res if hasattr(res, "buf") else None)
    try:
        items += _sheet_entries(sh, cv)
    except Exception:                                                  # noqa: BLE001 - reporting never breaks a drawing
        import os
        if os.environ.get("PIXELCANVAS_DEBUG"):
            raise
    return items or None


_record_calls(View, skip=("to_sheet", "to_px", "room_area"))
_manifest.wrap_method(Sheet, "draw", "view", after=_after_sheet, cv_of=_remember_sheet)
