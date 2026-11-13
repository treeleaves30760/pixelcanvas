# -*- coding: utf-8 -*-
"""
Declarative diagrams on top of ``pixelcanvas.vector``: describe nodes and edges, get a finished figure.

* ``Diagram``   flowcharts and architecture diagrams with automatic layered layout, orthogonal / curved routing,
                containers, icons and 12 node shapes.
* ``Tree``      org charts, hierarchies and mind maps from a nested dict.
* ``Sequence``  sequence diagrams (participants, lifelines, messages, activation bars, notes, frames).
* ``Timeline``  roadmaps: phases as bars, milestones on a line.
* ``Table``     comparison matrices with check / cross marks.
* ``callout``, ``callouts``, ``markers``, ``brace``, ``dimension``, ``scale_bar``: labels for figures drawn with
  the rest of the library, placed without collisions.

Every figure has ``build() -> Drawing``, ``draw(cv, box=None, log=None)``, ``render(scale=1)``,
``save_svg(path)`` and ``into(drawing, box)`` (embed into a bigger page).
"""
import math

import numpy as np

from . import vector as _v
from .vector import Drawing, Path, THEMES, measure_text

__all__ = ["Diagram", "Tree", "Sequence", "Timeline", "Table", "Annotator", "callout", "callouts", "markers",
           "brace", "dimension", "scale_bar", "SHAPES", "page"]

SHAPES = ("box", "rounded", "pill", "diamond", "ellipse", "cylinder", "parallelogram", "hexagon", "document",
          "cloud", "note", "circle")
_ALIASES = {"rect": "box", "rectangle": "box", "square": "box", "round": "rounded", "card": "rounded",
            "process": "rounded", "decision": "diamond", "rhombus": "diamond", "database": "cylinder",
            "db": "cylinder", "storage": "cylinder", "terminal": "pill", "start": "pill", "end": "pill",
            "stadium": "pill", "io": "parallelogram", "data": "parallelogram", "input": "parallelogram",
            "doc": "document", "oval": "ellipse", "sticky": "note", "comment": "note", "hex": "hexagon",
            "prep": "hexagon", "dot": "circle"}
_COLOR_NAMES = {"blue": 0, "orange": 1, "green": 2, "red": 3, "purple": 4, "violet": 4, "cyan": 5, "teal": 5,
                "pink": 6, "olive": 7, "lime": 7, "yellow": 7}
_K = 0.5523                                   # cubic circle constant


# --------------------------------------------------------------------------------------------- colour helpers

def _rgb(c):
    col = _v._color(c)
    return (np.array([0.5, 0.5, 0.5]), 1.0) if col is None else (np.asarray(col[0], float), col[1])


def _hex(c):
    return _v._hex(_rgb(c)[0])


def _mix(a, b, t):
    """``t`` of colour ``a`` over colour ``b`` → '#rrggbb'."""
    ca, cb = _rgb(a)[0], _rgb(b)[0]
    return _v._hex(cb + (ca - cb) * float(t))


def _lum(c):
    r, g, b = _rgb(c)[0]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _on(c, th):
    """Readable text colour on a fill ``c``."""
    return "#ffffff" if _lum(c) < 0.62 else ("#10151c" if th.dark else th.ink)


from ._args import (check_choice as _check_choice, reject_unknown as _reject_unknown,  # noqa: E402
                    strict_methods as _strict_methods, unknown_name as _unknown_name)


def _theme(theme):
    if isinstance(theme, _v.Theme):
        return theme
    if isinstance(theme, dict):
        return _v.Theme(theme)
    if theme not in THEMES:
        raise _unknown_name("theme", theme, THEMES)
    return THEMES[theme]


def _accent(th, color, default=0):
    """Resolve a colour option: None → accent ``default``; int → accent i; 'green' → theme green; else as is."""
    if color is None:
        return th.accent(default)
    if isinstance(color, bool):
        return th.accent(default)
    if isinstance(color, (int, np.integer)):
        return th.accent(int(color))
    if isinstance(color, str) and color.lower() in _COLOR_NAMES:
        return th.accent(_COLOR_NAMES[color.lower()])
    if isinstance(color, str) and color.lower() in ("ink", "muted", "grid", "surface", "background"):
        return th[color.lower()]
    return color


def _shadow(th, strength=1.0):
    s = th.shadow
    return dict(dx=0, dy=3 * strength, blur=9 * strength, color=s["color"],
                opacity=min(0.85, s["opacity"] * (1.25 if th.dark else 1.1)))


def _edge_color(th):
    return _mix(th.muted, th.background, 0.78 if th.dark else 0.72)


# --------------------------------------------------------------------------------------------- text helpers

def _fam(th, family=None):
    return family or th.font["family"]


def _measure(text, size, weight="normal", family="sans"):
    if not str(text):
        return (0.0, 0.0)
    return measure_text(str(text), size, family, weight)


def _is_cjk(ch):
    o = ord(ch)
    return 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF \
        or 0x3000 <= o <= 0x303F


def _tokens(line):
    out, cur = [], ""
    for ch in line:
        if ch == " ":
            if cur:
                out.append(cur)
            cur = ""
            out.append(" ")
        elif _is_cjk(ch):
            if cur:
                out.append(cur)
            cur = ""
            out.append(ch)
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _wrap(text, size, max_w, weight="normal", family="sans"):
    """Balanced wrap of ``text`` to lines no wider than ``max_w`` → string with '\\n'."""
    text = str(text)
    if max_w is None or max_w <= 0:
        return text
    lines = []
    for para in text.split("\n"):
        w = _measure(para, size, weight, family)[0]
        if w <= max_w or not para.strip():
            lines.append(para)
            continue
        n = int(math.ceil(w / max_w))
        toks = _tokens(para)

        def greedy(limit):
            out, cur = [], ""
            for tk in toks:
                trial = cur + tk
                if cur.strip() and tk != " " and _measure(trial.strip(), size, weight, family)[0] > limit:
                    out.append(cur.strip())
                    cur = tk
                else:
                    cur = trial
            if cur.strip():
                out.append(cur.strip())
            return out

        limit, best = w / n, None                       # narrowest width that still needs only n lines
        while limit < max_w:
            trial = greedy(limit)
            if len(trial) <= n:
                best = trial
                break
            limit *= 1.05
        lines += best or greedy(max_w)
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- geometry helpers

def _round_poly(pts, r, closed=True):
    """Path through ``pts`` with corners rounded by radius ``r`` (circle-like cubic corners)."""
    pts = [(float(x), float(y)) for x, y in pts]
    # drop duplicate / collinear points
    clean = []
    for p in pts:
        if clean and abs(p[0] - clean[-1][0]) < 1e-6 and abs(p[1] - clean[-1][1]) < 1e-6:
            continue
        clean.append(p)
    if closed and len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) < 1e-6 and abs(clean[0][1] - clean[-1][1]) < 1e-6:
        clean.pop()
    pts = clean
    if not closed:
        out = [pts[0]] if pts else []
        for i in range(1, len(pts) - 1):
            a, b, c = np.array(out[-1]), np.array(pts[i]), np.array(pts[i + 1])
            cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cr) < 1e-6 and np.dot(b - a, c - b) >= 0:
                continue
            out.append(pts[i])
        if len(pts) > 1:
            out.append(pts[-1])
        pts = out
    n = len(pts)
    p = Path()
    if n < 2:
        return p
    if r <= 0 or n < 3:
        p.move_to(*pts[0])
        for q in pts[1:]:
            p.line_to(*q)
        if closed:
            p.close()
        return p

    def corner(i):
        a = np.array(pts[(i - 1) % n])
        b = np.array(pts[i])
        c = np.array(pts[(i + 1) % n])
        la, lc = np.linalg.norm(a - b), np.linalg.norm(c - b)
        if la < 1e-9 or lc < 1e-9:
            return b, b, b
        fa = 0.5 if (closed or i - 1 > 0) else 1.0
        fc = 0.5 if (closed or i + 1 < n - 1) else 1.0
        d = min(r, la * fa, lc * fc)
        return b + (a - b) / la * d, b, b + (c - b) / lc * d

    if closed:
        _, _, p2 = corner(0)
        p.move_to(*p2)
        idx = list(range(1, n)) + [0]
    else:
        p.move_to(*pts[0])
        idx = list(range(1, n - 1))
    for i in idx:
        p1, b, p2 = corner(i)
        p.line_to(*p1)
        c1 = p1 + (b - p1) * _K
        c2 = p2 + (b - p2) * _K
        p.cubic_to(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
    if closed:
        p.close()
    else:
        p.line_to(*pts[-1])
    return p


def _rect_hit(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inflate(b, d):
    return (b[0] - d, b[1] - d, b[2] + d, b[3] + d)


def _seg_hits_box(p, q, box):
    """Does segment p-q intersect the axis-aligned box (x0, y0, x1, y1)? (Liang-Barsky)"""
    x0, y0, x1, y1 = box
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
        if abs(pp) < 1e-12:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return t1 - t0 > 1e-9


def _in_poly(p, poly):
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _rect_hits_poly(box, poly):
    """True when the axis-aligned ``box`` overlaps the closed polygon ``poly``."""
    x0, y0, x1, y1 = box
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    probes = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (xm, y0), (xm, y1), (x0, ym), (x1, ym), (xm, ym)]
    if any(_in_poly(q, poly) for q in probes):
        return True
    if any(x0 <= px <= x1 and y0 <= py <= y1 for px, py in poly):
        return True
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for a, b in zip(poly, poly[1:] + poly[:1]):
        for c, d in edges:
            if _seg_cross(a, b, c, d):
                return True
    return False


def _seg_cross(a, b, c, d):
    def o(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def _poly_len(pts):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def _point_at(pts, s):
    """Point and unit direction at arc length ``s`` along polyline ``pts``."""
    for a, b in zip(pts, pts[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-9:
            continue
        if s <= L:
            t = s / L
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        s -= L
    a, b = pts[-2], pts[-1]
    L = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1e-9)
    return b, ((b[0] - a[0]) / L, (b[1] - a[1]) / L)


def _arrowhead(g, tip, d, size, color, kind="arrow"):
    """Draw an arrowhead with its tip at ``tip`` pointing along unit vector ``d``."""
    ux, uy = d
    px, py = -uy, ux
    L, W = size, size * 0.42
    bx, by = tip[0] - ux * L, tip[1] - uy * L
    if kind == "open":
        g.polyline([(bx + px * W, by + py * W), tip, (bx - px * W, by - py * W)], stroke=color,
                   stroke_width=max(size * 0.16, 1.2), cap="round", join="round")
    elif kind == "dot":
        g.circle(tip[0] - ux * size * 0.35, tip[1] - uy * size * 0.35, size * 0.35, fill=color)
    elif kind == "diamond":
        g.polygon([tip, (tip[0] - ux * L * 0.5 + px * W, tip[1] - uy * L * 0.5 + py * W),
                   (tip[0] - ux * L, tip[1] - uy * L), (tip[0] - ux * L * 0.5 - px * W, tip[1] - uy * L * 0.5 - py * W)],
                  fill=color)
    else:
        nx, ny = tip[0] - ux * L * 0.78, tip[1] - uy * L * 0.78
        p = Path().move_to(*tip).line_to(bx + px * W, by + py * W).quad_to(nx, ny, bx - px * W, by - py * W).close()
        g.path(p, fill=color)


def _arrow_trim(kind, size):
    return {"arrow": size * 0.72, "open": size * 0.1, "dot": size * 0.6, "diamond": size * 0.95}.get(kind, 0.0)


def _trim_poly(pts, s0, s1):
    """Cut ``s0`` off the start and ``s1`` off the end of a polyline."""
    pts = list(pts)
    if s1 > 0:
        rev = _trim_poly(pts[::-1], s1, 0)
        pts = rev[::-1]
    if s0 <= 0:
        return pts
    acc = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if acc + L >= s0 and L > 1e-9:
            t = (s0 - acc) / L
            return [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)] + pts[i + 1:]
        acc += L
    return pts[-2:]


def _dash(style, sw):
    if style in (None, "solid"):
        return None
    if style == "dashed":
        return (sw * 4.0, sw * 3.0)
    if style == "dotted":
        return (0.01, sw * 2.6)
    return style


# --------------------------------------------------------------------------------------------- figure base

class _Figure:
    """Common output API: ``build()``, ``draw()``, ``render()``, ``save_svg()``, ``into()``."""

    def __init__(self, theme="light", title=None, subtitle=None, margin=40):
        self.theme = _theme(theme)
        self.title, self.subtitle, self.margin = title, subtitle, margin
        self.boxes = {}

    # subclasses implement _size() -> (w, h) and _paint(dwg, x, y)
    def _size(self):
        raise NotImplementedError

    def _paint(self, dwg, x, y):
        raise NotImplementedError

    _tfs, _sfs = 34.0, 18.0                     # title / subtitle size in figure units

    def _header_size(self):
        th = self.theme
        w = h = 0.0
        k = self._tfs / 34.0
        if self.title:
            tw, t_h = _measure(self.title, self._tfs, th.font["title_weight"], _fam(th))
            w, h = tw, t_h
        if self.subtitle:
            sw, sh = _measure(self.subtitle, self._sfs, "normal", _fam(th))
            w, h = max(w, sw), h + (10 * k if self.title else 0) + sh
        return w, (h + 28 * k if h else 0.0)

    def build(self, background=True, aspect=None, target=None):
        """Lay the figure out and return it as a ``vector.Drawing`` (user units ≈ px at natural size).

        ``aspect`` (width / height): grow the frame (and, for diagrams, the gaps) to exactly that shape, so
        the figure fills a box without empty bands; the title stays top-left. ``target=(width_px,
        height_px, S)``: lay the figure out for a box of that pixel size (diagrams size their text in
        pixels, ``S`` = pixels per 1080p pixel)."""
        th = self.theme
        frame = None
        if hasattr(self, "_fit"):
            frame = self._fit(*target) if target is not None else self._unfit()
            if frame is not None:
                aspect = None
        m = getattr(self, "_mu", None) or self.margin
        hw, hh = self._header_size()
        if hasattr(self, "_stretch") and frame is None:
            self._stretch(aspect, hw, hh)
        cw, ch = self._size()
        W0, H0 = max(cw, hw) + 2 * m, ch + hh + 2 * m
        W, H = W0, H0
        if frame is not None:
            W, H = max(W0, frame[0]), max(H0, frame[1])
        elif aspect:
            if W0 / H0 > aspect:
                H = W0 / aspect
            else:
                W = H0 * aspect
        dwg = Drawing(W, H, theme=th, background=(th.background if background else "none"),
                      name=type(self).__name__.lower())
        if self.title or self.subtitle:
            with dwg.group("title"):
                y = m
                if self.title:
                    dwg.text(m, y, self.title, size=self._tfs, weight=th.font["title_weight"], anchor="lt")
                    y += _measure(self.title, self._tfs, th.font["title_weight"], _fam(th))[1] + 10 * self._tfs / 34
                if self.subtitle:
                    dwg.text(m, y, self.subtitle, size=self._sfs, fill=th.muted, anchor="lt")
        ox = m + (W - 2 * m - cw) / 2
        if self.title or self.subtitle:
            ox = m + (W - W0) / 2 + (max(cw, hw) - cw) / 2 if cw < hw else m + (W - W0) / 2
        oy = m + hh + (H - H0) / 2
        self.origin = (ox, oy)
        self.header_box = (m, m, m + hw, m + hh - 28 * self._tfs / 34) if hh else None
        self._paint(dwg, ox, oy)
        return dwg

    def _aspect(self, box, cv):
        x0, y0, x1, y1 = box if box is not None else (0, 0, cv.w, cv.h)
        return max(x1 - x0, 1) / max(y1 - y0, 1)

    def draw(self, cv, box=None, log=None, background=None, fill=True):
        """Fit the figure into ``box`` (x0, y0, x1, y1 canvas px; default the whole canvas) and paint it.
        ``fill``: adapt the frame (and diagram gaps) to the box shape. ``background``: None → fill the canvas
        with the theme background when no box is given."""
        if background is None:
            background = box is None
        if background and box is None:
            if log:
                log("background")
            cv.paint(_rgb(self.theme.background)[0])
        bx0, by0, bx1, by1 = box if box is not None else (0, 0, cv.w, cv.h)
        target = (max(bx1 - bx0, 1), max(by1 - by0, 1), cv.h / 1080.0) if fill else None
        dwg = self.build(background=bool(background and box is not None),
                         aspect=self._aspect(box, cv) if fill else None,
                         target=target if hasattr(self, "_fit") else None)
        dwg.draw(cv, log=log, box=box)
        k = min((bx1 - bx0) / dwg.width, (by1 - by0) / dwg.height)
        self._px = (k, bx0 + ((bx1 - bx0) - dwg.width * k) / 2, by0 + ((by1 - by0) - dwg.height * k) / 2)
        if hasattr(self, "_record"):
            self._record()
        return cv

    def render(self, scale=1.0, log=None):
        """New Canvas at the natural size times ``scale``."""
        return self.build().render(scale=scale * getattr(self, "_unit_px", 1.0), log=log)

    def to_svg(self):
        return self.build().to_svg()

    def save_svg(self, path):
        """Write the figure as a standalone SVG file."""
        return self.build().save_svg(path)

    def into(self, target, box, name=None, align="center", fill=True, margin=8):
        """Embed the figure into a ``Drawing`` / ``Group`` scaled to fit ``box`` (x0, y0, x1, y1 user units).
        ``align``: center / top / left / top-left. Returns the embedded Group."""
        x0, y0, x1, y1 = box
        keep = self.margin
        if margin is not None:
            self.margin = margin
        try:
            tgt = None
            if fill and hasattr(self, "_fit"):
                root = target
                hgt = getattr(root, "height", None)
                tgt = (max(x1 - x0, 1), max(y1 - y0, 1), (hgt / 1080.0) if hgt else 1.0)
            sub = self.build(background=False, aspect=(x1 - x0) / max(y1 - y0, 1e-9) if fill else None,
                             target=tgt)
        finally:
            self.margin = keep
        g = target.embed(sub, box=box, align=align, name=name or sub.name)
        self.fit_scale, self.fit_offset = g.sx, (g.tx, g.ty)
        return g


def page(width=1920, height=1080, theme="light", title=None, subtitle=None, margin=64):
    """A blank themed ``Drawing`` with an optional title block → (drawing, content_box).

    Place figures with ``fig.into(dwg, box)``; ``content_box`` is the free area below the title."""
    th = _theme(theme)
    dwg = Drawing(width, height, theme=th, name="page")
    y = margin
    if title or subtitle:
        with dwg.group("title"):
            if title:
                dwg.text(margin, y, title, size=52, weight=th.font["title_weight"], anchor="lt")
                y += _measure(title, 52, th.font["title_weight"], _fam(th))[1] + 14
            if subtitle:
                dwg.text(margin, y, subtitle, size=24, fill=th.muted, anchor="lt")
                y += _measure(subtitle, 24, "normal", _fam(th))[1]
        y += 36
    return dwg, (margin, y, width - margin, height - margin)


# --------------------------------------------------------------------------------------------- node art

_ROW_SHAPES = ("box", "rounded", "pill", "note", "document", "parallelogram", "hexagon", "cylinder")
_POINTY = ("diamond", "hexagon", "circle")             # side ports sit on the corner point


class _NodeArt:
    """Measured content and outline of one node."""

    def __init__(self, th, label, sublabel=None, icon=None, shape="rounded", size=18, wrap=200, width=None,
                 height=None, min_w=None):
        self.th, self.shape, self.icon = th, shape, icon
        fam = _fam(th)
        self.fs, self.ss = size, round(size * 0.76, 1)
        self.label = _wrap(label or "", self.fs, wrap, "bold", fam) if label else ""
        self.sub = _wrap(sublabel, self.ss, wrap * 1.1, "normal", fam) if sublabel else ""
        lw, lh = _measure(self.label, self.fs, "bold", fam) if self.label else (0, 0)
        sw, sh = _measure(self.sub, self.ss, "normal", fam) if self.sub else (0, 0)
        self.lw, self.lh, self.sw, self.sh = lw, lh, sw, sh
        tw, th_ = max(lw, sw), lh + (sh + 5 if self.sub else 0)
        self.tw, self.tht = tw, th_
        self.row = icon is not None and shape in _ROW_SHAPES
        chip = size * 2.3 if self.row else size * 2.0
        self.chip = chip if icon else 0
        if icon and self.row:
            cw, ch = chip + size * 0.7 + tw, max(chip, th_)
        elif icon:
            cw, ch = max(chip, tw), chip + size * 0.45 + th_
        else:
            cw, ch = tw, th_
        self.cw, self.ch = cw, ch
        px, py = size * 1.1, size * 0.85
        s = shape
        if s in ("box", "rounded", "note"):
            W, H = cw + 2 * px, ch + 2 * py
        elif s == "document":
            W, H = cw + 2 * px, ch + 2 * py + size * 0.6
        elif s == "pill":
            H = ch + 2 * py
            W = cw + H * 0.9 + px * 0.6
        elif s == "parallelogram":
            H = ch + 2 * py
            W = cw + 2 * px + H * 0.5
        elif s == "hexagon":
            H = ch + 2 * py
            W = cw + 2 * px + H * 0.5
        elif s == "cylinder":
            W = cw + 2 * px
            cap = min(max(W * 0.09, 7.0), 14.0)
            H = ch + 2 * py + 2.4 * cap
        elif s == "diamond":
            W = cw + 1.7 * ch + size * 2.2
            H = max(ch + 0.42 * W, size * 4)
            f = cw / W + ch / H
            if f > 0.88:
                W, H = W * f / 0.88, H * f / 0.88
        elif s == "ellipse":
            W, H = cw * 1.3 + size * 1.8, ch * 1.45 + size * 1.3
        elif s == "circle":
            W = H = max(math.hypot(cw, ch) + size * 1.2, size * 3.4)
        elif s == "cloud":
            W, H = cw * 1.3 + size * 2.6, ch * 1.5 + size * 2.4
        else:
            W, H = cw + 2 * px, ch + 2 * py
        mw = min_w if min_w is not None else (size * 6.2 if s not in ("circle", "diamond") else 0)
        if s in ("box", "rounded", "note", "document", "cylinder", "parallelogram", "hexagon", "pill"):
            W = max(W, mw)
            H = max(H, size * 3.0)
        self.W = float(width) if width else W
        self.H = float(height) if height else H
        if s == "circle" and not (width or height):
            self.W = self.H = max(self.W, self.H)

    # ---- outline (unrounded polygon, for port and boundary tests)
    def outline(self, cx, cy, W=None, H=None):
        W, H = W or self.W, H or self.H
        x0, y0, x1, y1 = cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2
        s = self.shape
        if s == "diamond":
            return [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
        if s in ("ellipse", "circle", "cloud"):
            a = np.linspace(0, 2 * math.pi, 48, endpoint=False)
            return list(zip(cx + W / 2 * np.cos(a), cy + H / 2 * np.sin(a)))
        if s == "parallelogram":
            k = H * 0.25
            return [(x0 + k, y0), (x1, y0), (x1 - k, y1), (x0, y1)]
        if s == "hexagon":
            k = H * 0.25
            return [(x0 + k, y0), (x1 - k, y0), (x1, cy), (x1 - k, y1), (x0 + k, y1), (x0, cy)]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # ---- body
    def body(self, g, cx, cy, fill, stroke, sw=1.5, W=None, H=None, radius=None):
        W, H = W or self.W, H or self.H
        x0, y0 = cx - W / 2, cy - H / 2
        s = self.shape
        R = self.th.radius * 1.5 if radius is None else radius
        st = dict(fill=fill, stroke=stroke, stroke_width=sw, join="round")
        if s == "box":
            return g.rect(x0, y0, W, H, r=min(4, R), **st)
        if s in ("rounded",):
            return g.rect(x0, y0, W, H, r=min(R, H / 2), **st)
        if s == "pill":
            return g.rect(x0, y0, W, H, r=H / 2, **st)
        if s == "circle":
            return g.circle(cx, cy, min(W, H) / 2, **st)
        if s == "ellipse":
            return g.ellipse(cx, cy, W / 2, H / 2, **st)
        if s in ("diamond", "parallelogram", "hexagon"):
            return g.path(_round_poly(self.outline(cx, cy, W, H), min(R * 0.8, 9)), **st)
        if s == "note":
            f = min(18, H * 0.3)
            pts = [(x0, y0), (x0 + W - f, y0), (x0 + W, y0 + f), (x0 + W, y0 + H), (x0, y0 + H)]
            item = g.path(_round_poly(pts, 4), **st)
            g.path(Path().move_to(x0 + W - f, y0).line_to(x0 + W - f, y0 + f - 3)
                   .quad_to(x0 + W - f, y0 + f, x0 + W - f + 3, y0 + f).line_to(x0 + W, y0 + f).close(),
                   fill=_mix(stroke, fill, 0.35) if fill not in (None, "none") else stroke, stroke=stroke,
                   stroke_width=sw, join="round")
            return item
        if s == "document":
            wv = min(10, H * 0.14)
            p = Path().move_to(x0 + 6, y0).line_to(x0 + W - 6, y0).quad_to(x0 + W, y0, x0 + W, y0 + 6)
            p.line_to(x0 + W, y0 + H - wv)
            p.cubic_to(x0 + W * 0.75, y0 + H - wv * 2.6, x0 + W * 0.5, y0 + H - wv * 0.2, x0 + W * 0.25, y0 + H)
            p.cubic_to(x0 + W * 0.12, y0 + H + wv * 0.5, x0 + W * 0.04, y0 + H - wv * 0.3, x0, y0 + H - wv * 0.6)
            p.line_to(x0, y0 + 6).quad_to(x0, y0, x0 + 6, y0).close()
            return g.path(p, **st)
        if s == "cylinder":
            cap = min(max(W * 0.09, 7.0), 14.0)
            p = Path().move_to(x0, y0 + cap)
            p.arc_to(W / 2, cap, 0, 0, 1, x0 + W, y0 + cap)
            p.line_to(x0 + W, y0 + H - cap).arc_to(W / 2, cap, 0, 0, 1, x0, y0 + H - cap).close()
            item = g.path(p, **st)
            g.ellipse(cx, y0 + cap, W / 2, cap, fill=_mix(stroke, fill, 0.18) if fill not in (None, "none")
                      else "none", stroke=stroke, stroke_width=sw)
            return item
        if s == "cloud":
            return g.path(_cloud_path(cx, cy, W, H), **st)
        return g.rect(x0, y0, W, H, r=R, **st)

    # ---- content
    def content(self, g, cx, cy, accent, text_color=None, sub_color=None, chip_fill=None, icon_color=None,
                align=None, W=None, H=None):
        th = self.th
        W, H = W or self.W, H or self.H
        fam = _fam(th)
        tc = text_color or th.ink
        sc = sub_color or th.muted
        shift_y = 0.0
        if self.shape == "cylinder":
            shift_y = min(max(W * 0.09, 7.0), 14.0) * 0.6
        elif self.shape == "document":
            shift_y = -self.fs * 0.2
        cy = cy + shift_y
        if self.icon and self.row:
            left = cx - self.cw / 2
            if self.shape in ("box", "rounded", "note", "document", "cylinder") and align != "center":
                left = cx - W / 2 + self.fs * 1.1 + (0 if self.shape != "parallelogram" else H * 0.25)
            chip = self.chip
            if chip_fill is not None:
                g.rect(left, cy - chip / 2, chip, chip, r=chip * 0.28, fill=chip_fill)
            g.icon(self.icon, left + chip / 2, cy, chip * 0.56, color=icon_color or accent, stroke_width=2.0)
            tx = left + chip + self.fs * 0.7
            ty = cy - self.tht / 2
            if self.label:
                g.text(tx, ty, self.label, size=self.fs, weight="bold", fill=tc, anchor="lt", family=fam)
            if self.sub:
                g.text(tx, ty + self.lh + 5, self.sub, size=self.ss, fill=sc, anchor="lt", family=fam)
            return
        ty = cy - self.ch / 2
        if self.icon:
            chip = self.chip
            if chip_fill is not None:
                g.circle(cx, ty + chip / 2, chip / 2, fill=chip_fill)
            g.icon(self.icon, cx, ty + chip / 2, chip * 0.55, color=icon_color or accent, stroke_width=2.0)
            ty += chip + self.fs * 0.45
        if self.label:
            g.text(cx, ty, self.label, size=self.fs, weight="bold", fill=tc, anchor="mt", family=fam,
                   align="center")
        if self.sub:
            g.text(cx, ty + self.lh + 5, self.sub, size=self.ss, fill=sc, anchor="mt", family=fam, align="center")


def _cloud_path(cx, cy, W, H):
    """A friendly cloud outline inside a W x H box."""
    rx, ry = W / 2, H / 2
    # bumps around an ellipse, bottom flatter
    angs = [200, 245, 290, 335, 20, 70, 115, 160]
    pts = [(cx + rx * 0.86 * math.cos(math.radians(a)), cy + ry * 0.80 * math.sin(math.radians(a))) for a in angs]
    p = Path().move_to(*pts[0])
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        chord = math.hypot(b[0] - a[0], b[1] - a[1])
        big = 0.62 if i < 4 else 0.55
        p.arc_to(chord * big, chord * big, 0, 0, 1, b[0], b[1])
    return p.close()


def _style_colors(th, style, accent):
    """(fill, stroke, text, sub, chip, icon) for a node style."""
    bg, sf = th.background, th.surface
    if style == "solid":
        txt = _on(accent, th)
        return (accent, _mix("#000000", accent, 0.12), txt, _mix(txt, accent, 0.78), _mix("#ffffff", accent, 0.2),
                txt)
    if style == "tint":
        return (_mix(accent, sf, 0.16 if th.dark else 0.11), _mix(accent, sf, 0.7), th.ink, th.muted,
                _mix(accent, sf, 0.26 if th.dark else 0.2), accent)
    if style == "outline":
        return (bg, _mix(accent, bg, 0.85), th.ink, th.muted, _mix(accent, bg, 0.16), accent)
    if style == "plain":
        return (sf, th.grid, th.ink, th.muted, _mix(accent, sf, 0.14), accent)
    # card
    return (sf, _mix(accent, sf, 0.42 if th.dark else 0.36), th.ink, th.muted, _mix(accent, sf, 0.2 if th.dark
                                                                                          else 0.13), accent)


# --------------------------------------------------------------------------------------------- Diagram

class _N:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _GroupHandle:
    def __init__(self, dia, gid):
        self.dia, self.id = dia, gid

    def __enter__(self):
        self.dia._cur_group.append(self.id)
        return self

    def __exit__(self, *a):
        self.dia._cur_group.remove(self.id)
        return False

    def node(self, id, label=None, **kw):
        kw.setdefault("group", self.id)
        return self.dia.node(id, label, **kw)


class _Rec(dict):
    """Plain-data record (a dict) whose keys also read as attributes: ``n["label"] == n.label``."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


_DFS = 18.0                                     # label size in layout units; the fit maps units to pixels
_LANES = ("rows", "columns", "auto")


class Diagram(_Figure):
    """Flowchart / architecture diagram with automatic layered layout that fills the box it is drawn into.

    ``Diagram(theme="light", direction="auto", title=None, subtitle=None)``

    * ``direction``: auto (default: LR or TB, whichever fills the box), LR, TB, RL, BT.
    * ``lanes``: "rows" / "columns" / "auto": every group becomes a full-width band (rows) or a full-height
      column, members spread evenly across it; groups in the order they were made (or ``d.tiers([...])``).
    * ``routing``: "orthogonal" (rounded elbows, default), "curved", "straight".
    * ``node_style``: "card" (default), "tint", "solid", "outline", "plain" — per node with ``style=``.
    * ``font_size``: the smallest node label height in pixels on a 1080-px-tall canvas (26; scales with the
      canvas height). The layout grows around it: gaps tighten before text shrinks, and text grows past it up to
      ``grow`` times when the box has room (1.7; ``False`` keeps ``font_size`` exactly). ``floor=False`` lets
      text shrink to fit a crowded diagram. ``title_size``: title height the same way (40).
    * ``keep_order``: ``True`` keeps the nodes of every tier in the order they were added (left to right in a
      row of tiers, top to bottom in columns) instead of the order that crosses the fewest edges.
    * ``spacing``: scales every gap (1.0 default); ``wrap``: label wrap width in layout units (200).
    """

    def __init__(self, theme="light", direction="auto", title=None, subtitle=None, routing="orthogonal",
                 node_style="card", spacing=1.0, font_size=26, wrap=200, uniform=True, margin=40,
                 corner=12, lanes=None, title_size=40, grow=1.7, floor=True, keep_order=False):
        super().__init__(theme, title, subtitle, margin)
        d = str(direction).upper()
        if d not in ("AUTO", "LR", "TB", "RL", "BT", "TD"):
            raise _unknown_name("direction", direction, ["auto", "LR", "TB", "RL", "BT"])
        self._auto = d == "AUTO"
        self.direction = "LR" if self._auto else ("TB" if d == "TD" else d)
        _check_choice("Diagram", "routing", routing, ["orthogonal", "curved", "straight"])
        _check_choice("Diagram", "node_style", node_style, ["card", "tint", "solid", "outline", "plain"])
        if lanes is not None:
            _check_choice("Diagram", "lanes", lanes, list(_LANES))
        self.routing, self.node_style = routing, node_style
        self._sp_mul = 1.0                              # tightened by the font_size floor while fitting
        self.spacing, self.font_size, self.wrap, self.uniform = spacing, font_size, wrap, uniform
        self.title_size = title_size
        self.grow = 1.0 if grow is False else (1.7 if grow is True else float(grow))
        self.corner = corner
        self.lanes = lanes
        self.floor, self.keep_order = bool(floor), bool(keep_order)
        self._nodes, self._edges, self.groups = {}, [], {}
        self._buses, self._same, self._tier_order = {}, [], None
        self._cur_group = []
        self._kc = self._km = self._wg = 1.0
        self._bands, self._balance, self._stagger = 1, False, False
        self._mu = None
        self.fit = {}

    @property
    def spacing(self):
        """Gap multiplier (1.0); the font_size floor may tighten it while the diagram is fitted."""
        return self._spacing * self._sp_mul

    @spacing.setter
    def spacing(self, v):
        self._spacing = float(v)

    @property
    def _unit_px(self):
        return self.font_size / _DFS

    # ---- plain data for reports
    @property
    def nodes(self):
        """{id: record} with id, label, group, shape, sublabel (and ``box`` in canvas px after ``draw``).
        Buses appear with ``shape="bus"`` and their ``members``. Records are dicts with attribute access."""
        out = {}
        px = getattr(self, "_px", None)
        boxes = getattr(self, "boxes", {}) or {}
        for k, n in self._nodes.items():
            r = _Rec(id=k, label=n.label, group=n.group, shape=n.shape, sublabel=n.sublabel)
            if px and k in boxes:
                r["box"] = self._to_px(boxes[k])
            out[k] = r
        for k, b in self._buses.items():
            r = _Rec(id=k, label=b.label, group=None, shape="bus", sublabel=None, members=list(b.members))
            if px and k in getattr(self, "bus_boxes", {}):
                r["box"] = self._to_px(self.bus_boxes[k])
            out[k] = r
        return out

    @property
    def edges(self):
        """[record] with source, target, label, style, arrow (bus taps: target = the bus id, ``bus=True``)."""
        out = [_Rec(source=e.a, target=e.b, label=e.label, style=e.style, arrow=e.arrow,
                    kind=getattr(e, "kind", None), a=e.a, b=e.b) for e in self._edges]
        for k, b in self._buses.items():
            for m in b.members:
                out.append(_Rec(source=m, target=k, label=None, style=b.style, arrow=b.taps.get(m, b.arrow),
                                kind="bus", bus=True, a=m, b=k))
        return out

    def _to_px(self, b):
        k, x0, y0 = self._px
        return [round(x0 + b[0] * k, 1), round(y0 + b[1] * k, 1), round(x0 + b[2] * k, 1), round(y0 + b[3] * k, 1)]

    def _record(self):
        """After draw: the fit summary in canvas px (``d.fit``)."""
        if not getattr(self, "_px", None):
            return
        bs = list(getattr(self, "boxes", {}).values()) + list(getattr(self, "group_boxes", {}).values()) + \
            list(getattr(self, "bus_boxes", {}).values())
        if getattr(self, "header_box", None):
            bs.append(self.header_box)
        if bs:
            u = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
            self.fit["content"] = tuple(self._to_px(u))
        self.fit["font_px"] = round(_DFS * self._px[0], 1)

    # ---- building API
    def node(self, id, label=None, shape="rounded", icon=None, group=None, color=None, sublabel=None, at=None,
             pos=None, style=None, width=None, height=None, rank=None):
        """Add a node. ``shape``: box rounded pill diamond ellipse cylinder parallelogram hexagon document cloud
        note circle (aliases: database, decision, start, io, ...). ``icon``: a vector icon name. ``color``:
        accent index, theme colour name ('green') or '#hex'. ``at=(col, row)`` grid cell, ``pos=(u, v)``
        absolute centre. Returns the id."""
        shape = _ALIASES.get(shape, shape)
        if shape not in SHAPES:
            raise _unknown_name("shape", shape, list(SHAPES) + list(_ALIASES))
        if icon is not None and icon not in _v.ICONS:
            raise _v._unknown("icon", icon, _v.ICONS, "Try vector.icon_names()")
        if group is None and self._cur_group:
            group = self._cur_group[-1]
        if group is not None and group not in self.groups:
            self.group(group)
        self._nodes[id] = _N(id=id, label=str(id) if label is None else label, shape=shape, icon=icon,
                             group=group, color=color, sublabel=sublabel, at=at, pos=pos, style=style,
                             width=width, height=height, fixed_rank=rank if at is None else at[0], dummy=False)
        return id

    def edge(self, a, b, label=None, style="solid", arrow="end", color=None, width=None, label_pos=None,
             head="arrow", rank=None):
        """Connect node ``a`` to ``b``. ``style``: solid / dashed / dotted; ``arrow``: end / start / both / none;
        ``head``: arrow / open / dot / diamond; ``label_pos``: 0..1 along the edge (default middle, near the
        source for decisions); ``rank="same"`` keeps the two side by side (peers). Unknown ids become nodes;
        a bus id adds a tap. ``a`` or ``b`` may be a list of ids: one edge per pair
        (``edge("gw", ["users", "orders"])`` fans out)."""
        if rank is not None:
            _check_choice("edge", "rank", rank, ["same"])
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            for x in (a if isinstance(a, (list, tuple)) else [a]):
                for y in (b if isinstance(b, (list, tuple)) else [b]):
                    self.edge(x, y, label, style, arrow, color, width, label_pos, head, rank)
            return self
        if a in self._buses or b in self._buses:            # a tap onto a bus
            bus, m = (self._buses[b], a) if b in self._buses else (self._buses[a], b)
            if m not in self._nodes:
                self.node(m)
            if m not in bus.members:
                bus.members.append(m)
            bus.taps[m] = arrow if b == bus.id else {"end": "start", "start": "end"}.get(arrow, arrow)
            return self
        for n in (a, b):
            if n not in self._nodes:
                self.node(n)
        self._edges.append(_N(a=a, b=b, label=label, style=style, arrow=arrow, color=color, width=width,
                              label_pos=label_pos, head=head, idx=len(self._edges)))
        if rank == "same":
            self.same_rank(a, b)
        return self

    def chain(self, *ids, **kw):
        """Edges along a path of ids: ``chain("a", "b", "c")``; a list step fans out / in:
        ``chain(["web", "mobile"], "gw", ["users", "orders"])``."""
        for a, b in zip(ids, ids[1:]):
            self.edge(a, b, **kw)
        return self

    def same_rank(self, *ids):
        """Keep these nodes in one rank (side by side in TB, stacked in LR): peers, replicas, a pair."""
        ids = [i for x in ids for i in (x if isinstance(x, (list, tuple)) else [x])]
        for n in ids:
            if n not in self._nodes:
                self.node(n)
        if len(ids) > 1:
            self._same.append(list(ids))
        return self

    def bus(self, id, label=None, members=(), style="solid", color=None, arrow="none"):
        """A bus bar (pub/sub topic, event bus, network segment) with a short tap to each member. It sits in the
        gap after its members and never changes the ranks. ``style``: solid / dashed / dotted; ``arrow`` on
        the taps: none / end (toward the bus) / start / both; ``d.edge(node, id)`` adds a tap later."""
        if id in self._nodes:
            raise ValueError(f"bus id {id!r} is already a node id")
        for m in members:
            if m not in self._nodes:
                self.node(m)
        self._buses[id] = _N(id=id, label=str(id) if label is None else label, members=list(members),
                             style=style, color=color, arrow=arrow, taps={})
        return id

    def group(self, id, label=None, color=None, icon=None, style="solid"):
        """Container for nodes. Use ``with d.group("svc", "Services"):`` or ``node(..., group="svc")``."""
        if id not in self.groups:
            self.groups[id] = _N(id=id, label=id if label is None else label, color=color, icon=icon, style=style,
                                 order=len(self.groups))
        else:
            g = self.groups[id]
            if label is not None:
                g.label = label
            if color is not None:
                g.color = color
            if icon is not None:
                g.icon = icon
        return _GroupHandle(self, id)

    def tiers(self, groups, lanes="rows", keep_order=None):
        """Lay these groups out as tiers, in this order: ``lanes="rows"`` full-width bands top to bottom,
        ``"columns"`` full-height columns left to right, ``"auto"`` whichever fills the box.
        ``keep_order=True`` keeps the nodes inside every tier in the order they were added."""
        _check_choice("tiers", "lanes", lanes, list(_LANES))
        if keep_order is not None:
            self.keep_order = bool(keep_order)
        for g in groups:
            if g not in self.groups:
                self.group(g)
        self._tier_order = list(groups)
        self.lanes = lanes
        return self

    # ---- fitting the box
    def _lane_dir(self):
        return {"rows": "TB", "columns": "LR"}.get(self.lanes)

    def _candidates(self, aspect):
        """Layout variants worth trying: (direction, bands, balance)."""
        if self.lanes:
            if self.lanes == "auto":
                return [("TB", 1, False), ("LR", 1, False)]
            return [(self._lane_dir(), 1, False)]
        if not self._auto:
            return [(self.direction, 1, False)]
        out = [("LR", 1, False), ("TB", 1, False)]
        if self._flowlike():                    # a flowchart reads left to right on a wide box, top down on a tall one
            if aspect >= 1.2:
                out = out[:1]
            elif aspect <= 0.85:
                out = out[1:]
        if self._hubs():
            out += [("LR", 1, True), ("TB", 1, True)]
        if len(self._nodes) >= 7:
            out += [("LR", 1, "stagger"), ("TB", 1, "stagger")]
        return out

    def _flowlike(self):
        """A flowchart: a few nodes with decisions or loops, no containers, buses or fixed positions."""
        if self.groups or self._buses or not 4 <= len(self._nodes) <= 20:
            return False
        if any(n.fixed_rank is not None or n.pos is not None for n in self._nodes.values()):
            return False
        if any(n.shape == "diamond" for n in self._nodes.values()):
            return True
        adj = {}
        for e in self._edges:
            adj.setdefault(e.a, []).append(e.b)
        state = {}

        def cyc(v):
            state[v] = 1
            for w in adj.get(v, ()):
                if state.get(w) == 1 or (w not in state and cyc(w)):
                    return True
            state[v] = 2
            return False
        return any(v not in state and cyc(v) for v in self._nodes)

    def _s_cap(self, BW, BH, s0):
        """Largest px-per-unit scale: text grows at most ``grow`` times, no node beyond a share of the box."""
        arts = [n.art for n in self._nodes.values()]
        maxW = max(a.W for a in arts) / max(self._wg, 1.0)
        maxH = max(a.H for a in arts)
        ch = getattr(self, "_canvas_h", BH)                   # node at most ~1/5 of the canvas height, 1/3 wide
        return min(s0 * self.grow, 0.2 * ch / maxH, min(0.34 * ch * 16 / 9, 0.85 * BW) / maxW), maxW

    def _choose(self, Wa, Ha, BW, BH, s0):
        """Pick direction / bands / balance: the variant that gives the largest text and fills the box best
        once its gaps are stretched (two layouts per variant: natural and fully stretched)."""
        best = None
        for dirn, bal in [(c[0], c[2]) for c in self._candidates(Wa / Ha)]:
            self.direction, self._balance, self._stagger = dirn, bal is True, bal == "stagger"
            for nb in range(1, 5):
                self._bands = nb
                self._kc = self._km = self._wg = 1.0
                res = self._layout()
                if nb > 1 and res.get("bands", 1) < nb:
                    break
                if self._stagger and not self._staggered:
                    break
                cw, ch = res["W"], res["H"]
                s_cap, maxW = self._s_cap(BW, BH, s0)
                s = min(Wa / cw, Ha / ch, s_cap)
                LR = dirn in ("LR", "RL")
                self._km, self._kc = 5.0, 8.0
                self._wg = max(1.0, min(1.45, 0.85 * BW / (maxW * s)))
                big = self._layout()
                covx = min(1.0, big["W"] * s / Wa)
                covy = min(1.0, big["H"] * s / Ha)
                extra = max(0, res.get("cut_edges", 0) - (nb - 1)) + 2 * res.get("cut_back", 0)
                extra += 4 * max(0.0, 0.6 - res.get("band_balance", 1.0))       # bands of similar length
                if res.get("band_balance", 1.0) < 0.35:
                    extra += 100
                if self._flowlike():                  # a flowchart that leaves a band of the frame empty folds into rows
                    extra += 12 * max(0.0, 0.8 - min(covx, covy))
                score = (math.log(min(s / s0, 1.3)) + 1.2 * math.log(min(covx, covy)) - 0.15 * (nb - 1)
                         - 0.25 * extra + (0.08 if bal is True else 0.0) - (0.12 if self._stagger else 0.0))
                if best is None or score > best[0] + 1e-9:
                    best = (score, dirn, nb, bal)
                too_long = (cw / ch > Wa / Ha) if LR else (cw / ch < Wa / Ha)
                if not self._auto or self.lanes or not too_long or self._stagger:
                    break
        _, self.direction, self._bands, bal = best
        self._balance, self._stagger = bal is True, bal == "stagger"
        self._kc = self._km = self._wg = 1.0

    def _unfit(self):
        """Natural size (render / save_svg / build without a box): no stretching, sizes in layout units."""
        self._mu = None
        k = _DFS / max(self.font_size, 1e-6)
        self._tfs, self._sfs = self.title_size * k, self.title_size * 0.52 * k
        if self._nodes and (self._auto or self.lanes == "auto"):
            self._choose(1840.0, 880.0, 1920.0, 1080.0, 22.0 / _DFS)
        self._kc = self._km = self._wg = 1.0
        return None

    def _fit(self, BW, BH, S):
        """Lay out for a BW x BH px box: choose the shape, then grow nodes, text and gaps to fill it.
        Returns the frame (width, height) in layout units; ``units * s`` = px."""
        S = max(S, 1e-6)
        self._sp_mul = 1.0
        F = self.font_size * S
        self._canvas_h = 1080.0 * S
        m_px = min(self.margin * S, 0.06 * min(BW, BH))         # a small box keeps a proportional margin
        tpx, spx = self.title_size * S, self.title_size * 0.52 * S
        self._tfs, self._sfs = tpx, spx
        hw_px, hh_px = self._header_size()
        if hw_px > BW - 2 * m_px:                       # a long title or subtitle steps down to fit the width
            k = (BW - 2 * m_px) / hw_px
            tpx, spx = tpx * k, spx * k
            self._tfs, self._sfs = tpx, spx
            hw_px, hh_px = self._header_size()
        Wa = max(BW - 2 * m_px, BW * 0.3)
        Ha = max(BH - 2 * m_px - hh_px, BH * 0.3)
        if not self._nodes or all(n.pos is not None for n in self._nodes.values()):
            self._kc = self._km = self._wg = 1.0
            cw, ch = self._size() if self._nodes else (1.0, 1.0)
            s = min(Wa / max(cw, 1), Ha / max(ch, 1), F / _DFS * self.grow)
        else:
            if self._auto or self.lanes == "auto":
                self._choose(Wa, Ha, BW, BH, F / _DFS)
            self._kc = self._km = self._wg = 1.0
            res = self._layout()
            cw, ch = res["W"], res["H"]
            s_cap, maxW = self._s_cap(BW, BH, F / _DFS)
            s = min(Wa / cw, Ha / ch, s_cap)
            if s < F / _DFS * 0.999 and self.floor:       # font_size is a floor: tighten the gaps before text shrinks
                best = (s, 1.0)
                for lam in (0.88, 0.76, 0.66, 0.58, 0.5, 0.44):
                    self._km = self._kc = self._sp_mul = lam
                    r2 = self._layout()
                    s2 = min(Wa / r2["W"], Ha / r2["H"], s_cap)
                    if s2 > best[0] + 1e-9:
                        best = (s2, lam)
                    if s2 >= F / _DFS * 0.999:
                        break
                self._km = self._kc = self._sp_mul = best[1]
                res = self._layout()
                cw, ch = res["W"], res["H"]
                s = min(Wa / cw, Ha / ch, s_cap)
            LR = self.direction in ("LR", "RL")
            wg_max = max(1.0, min(1.45, 0.85 * BW / (maxW * s)))
            # x axis: main gaps (LR) or cross gaps (TB) plus wider nodes; y axis: the other gap
            for axis, D in (("x", Wa / s), ("y", Ha / s)):
                main = (axis == "x") == LR
                cap = 5.0 if main else 8.0

                def setk(lam, axis=axis, main=main):
                    if main:
                        self._km = lam
                    else:
                        self._kc = lam
                    if axis == "x":
                        self._wg = min(wg_max, 1.0 + (lam - 1.0) * 0.3)

                def dim(axis=axis):
                    r = self._layout()
                    return r["W"] if axis == "x" else r["H"]

                if dim() >= D * 0.985:
                    continue
                setk(cap)
                if dim() <= D:
                    continue
                lo, hi = 1.0, cap
                for _ in range(7):
                    mid = (lo + hi) / 2
                    setk(mid)
                    if dim() <= D:
                        lo = mid
                    else:
                        hi = mid
                setk(lo)
            res = self._layout()
            s = min(Wa / res["W"], Ha / res["H"], s)
        if self._nodes and self.title and tpx < 1.3 * _DFS * s:     # the title stays above the node labels
            k = min(1.3 * _DFS * s / tpx, (BW - 2 * m_px) / max(hw_px, 1.0))
            tpx, spx = tpx * max(k, 1.0), spx * max(k, 1.0)
            self._tfs, self._sfs = tpx, spx
            Ha2 = max(BH - 2 * m_px - self._header_size()[1], BH * 0.3)
            if self._nodes and not all(n.pos is not None for n in self._nodes.values()):
                s = min(s, Ha2 / self._res["H"], Wa / self._res["W"])
        self._mu = m_px / s
        self._tfs, self._sfs = tpx / s, spx / s
        self.fit = {"direction": self.direction, "bands": self._bands, "lanes": self.lanes,
                    "scale": round(s, 4), "font_px": round(_DFS * s, 1)}
        return (BW / s, BH / s)

    def _stretch(self, aspect, hw, hh, limit=2.4):
        """Grow the gaps so the laid-out diagram approaches ``aspect`` (width / height)."""
        self._kc = self._km = 1.0
        if not aspect or not self._nodes or all(n.pos is not None for n in self._nodes.values()):
            return
        LR = self.direction in ("LR", "RL")
        m = self.margin
        for _ in range(4):
            cw, ch = self._size()
            W, H = max(cw, hw) + 2 * m, ch + hh + 2 * m
            r = (W / H) / aspect
            if abs(r - 1) < 0.03:
                break
            taller = r > 1                          # need more height
            attr = ("_kc" if LR else "_km") if taller else ("_km" if LR else "_kc")
            k = getattr(self, attr)
            f = r if taller else 1 / r
            nk = min(limit, k * (1 + (f - 1) * 1.6))
            if nk <= k + 1e-3:
                break
            setattr(self, attr, nk)

    # ---- layout
    def _colors(self, n):
        th = self.theme
        if n.color is not None:
            return _accent(th, n.color)
        if n.group is not None:
            g = self.groups[n.group]
            return _accent(th, g.color, g.order % 8 if g.color is None else 0)
        default = {"diamond": 1, "pill": 2, "circle": 2, "cylinder": 4, "note": 7, "document": 5,
                   "parallelogram": 5, "cloud": 5, "hexagon": 6}.get(n.shape, 0)
        return th.accent(default)

    def _gcolor(self, g):
        return _accent(self.theme, g.color, g.order % 8 if g.color is None else 0)

    def _hubs(self):
        """Edges to flip so a hub with 4+ leaf neighbours on one side spreads them over both sides."""
        deg = {k: 0 for k in self._nodes}
        for e in self._edges:
            if e.a != e.b:
                deg[e.a] += 1
                deg[e.b] += 1
        for b in self._buses.values():
            for m in b.members:
                deg[m] += 1
        flip = set()
        for v, n in self._nodes.items():
            if n.fixed_rank is not None:
                continue
            outs = [e for e in self._edges if e.a == v and e.b != v]
            ins = [e for e in self._edges if e.b == v and e.a != v]
            ol = [e for e in outs if deg[e.b] == 1 and self._nodes[e.b].fixed_rank is None]
            il = [e for e in ins if deg[e.a] == 1 and self._nodes[e.a].fixed_rank is None]
            if len(ol) >= 4 and not ins:
                flip |= {e.idx for e in ol[(len(ol) + 1) // 2:]}
            elif len(il) >= 4 and not outs:
                flip |= {e.idx for e in il[(len(il) + 1) // 2:]}
        return flip

    def _layout(self):
        if self.lanes in ("rows", "columns") and self.groups:
            self.direction = self._lane_dir()
        flip = self._hubs() if (self._balance and not self.lanes) else set()
        for e in self._edges:
            e.rev = e.idx in flip
            if e.rev:
                e.a, e.b = e.b, e.a
        try:
            res = self._layout_core()
        finally:
            for e in self._edges:
                if e.rev:
                    e.a, e.b = e.b, e.a
        if flip:
            res["paths"] = [(e, pts[::-1] if e.rev else pts) for e, pts in res["paths"]]
            for e in self._edges:
                if e.rev:
                    e.keys = None
        return res

    def _layout_core(self):
        th = self.theme
        sp = self.spacing
        fs = _DFS
        LR = self.direction in ("LR", "RL")
        nodes = self._nodes
        for n in nodes.values():
            n.art = _NodeArt(th, n.label, n.sublabel, n.icon, n.shape, fs, self.wrap, n.width, n.height)
            if self._wg > 1.0 and n.width is None and n.shape in _ROW_SHAPES:
                n.art.W *= self._wg
        self._lane_mode = False
        if nodes and all(n.pos is not None for n in nodes.values()):
            return self._layout_manual()
        ids = list(nodes)
        for e in self._edges:                              # layout runs more than once (stretching): reset
            e.short = e.lshape = False
            e.keys = None
        # ---------------- same-rank sets (peers): one representative per set
        par = {k: k for k in ids}

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for grp in self._same:
            for x in grp[1:]:
                if x in par and grp[0] in par:
                    par[find(x)] = find(grp[0])
        rep = {k: find(k) for k in ids}
        sets = {}
        for k in ids:
            sets.setdefault(rep[k], []).append(k)
        sets = [v for v in sets.values() if len(v) > 1]
        # ---------------- ranks: inside each cycle, edges that go backwards in breadth-first order are loops
        reps = sorted(set(rep.values()), key=ids.index)
        out = {k: [] for k in reps}
        indeg = {k: 0 for k in reps}
        for e in self._edges:
            ra_, rb_ = rep[e.a], rep[e.b]
            if ra_ != rb_:
                out[ra_].append(rb_)
                indeg[rb_] += 1

        def reach(src):
            seen, stack = {src}, [src]
            while stack:
                x = stack.pop()
                for y in out[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            return seen

        reachable = {k: reach(k) for k in reps}
        bfs, seen = [], set()
        for r in [k for k in reps if indeg[k] == 0] + reps:
            if r in seen:
                continue
            seen.add(r)
            queue = [r]
            while queue:
                x = queue.pop(0)
                bfs.append(x)
                for y in out[x]:
                    if y not in seen:
                        seen.add(y)
                        queue.append(y)
        border = {k: i for i, k in enumerate(bfs)}
        back = set()
        for e in self._edges:
            ra_, rb_ = rep[e.a], rep[e.b]
            if ra_ != rb_ and ra_ in reachable[rb_] and border[rb_] <= border[ra_]:
                back.add(e.idx)
        fwd = [e for e in self._edges if rep[e.a] != rep[e.b] and e.idx not in back]
        rank = {}
        preds = {k: [] for k in ids}
        succs = {k: [] for k in ids}
        for e in fwd:
            preds[e.b].append(e.a)
            succs[e.a].append(e.b)
        # topological order
        deg = {k: len(preds[k]) for k in ids}
        topo, queue = [], [k for k in ids if deg[k] == 0]
        while queue:
            v = queue.pop(0)
            topo.append(v)
            for w in succs[v]:
                deg[w] -= 1
                if deg[w] == 0:
                    queue.append(w)
        lanes = self.lanes if self.lanes != "auto" else ("rows" if self.direction == "TB" else "columns")
        self._lane_mode = bool(lanes) and bool(self.groups)
        if self._lane_mode:
            order = [g for g in (self._tier_order or []) if g in self.groups]
            order += [g for g in sorted(self.groups, key=lambda g: self.groups[g].order) if g not in order]
            tier = {g: i for i, g in enumerate(order)}
            for v in ids:
                g = nodes[v].group
                if g in tier:
                    rank[v] = 2 * tier[g]
            loose = [v for v in ids if v not in rank]
            for _ in range(len(loose) + 1):              # ungrouped nodes get a band of their own between tiers
                moved = False
                for v in loose:
                    ins = [rank[e.a] for e in self._edges if e.b == v and e.a != v and e.a in rank]
                    if ins:
                        r = max(ins) + 1
                    else:
                        outs = [rank[e.b] for e in self._edges if e.a == v and e.b != v and e.b in rank]
                        r = (min(outs) - 1) if outs else -1
                    if r % 2 == 0:
                        r += 1 if ins else -1
                    if rank.get(v) != r:
                        rank[v] = r
                        moved = True
                if not moved:
                    break
            for S_ in sets:
                mx = max(rank[v] for v in S_)
                for v in S_:
                    rank[v] = mx
            used = sorted(set(rank.values()))
            remap = {r: i for i, r in enumerate(used)}
            for v in ids:
                rank[v] = remap[rank[v]]
            back = {e.idx for e in self._edges if rank[e.b] < rank[e.a]}
        else:
            for v in topo:
                fr = nodes[v].fixed_rank
                rank[v] = int(fr) if fr is not None else max([rank[u] + 1 for u in preds[v]] or [0])
            # pull sources next to their first successor (grouped sources stay level with their group)
            for v in reversed(topo):
                if not preds[v] and succs[v] and nodes[v].fixed_rank is None:
                    rank[v] = max(rank[v], min(rank[w] for w in succs[v]) - 1)
            for v in topo:
                g = nodes[v].group
                if g is None or preds[v] or nodes[v].fixed_rank is not None:
                    continue
                others = [rank[u] for u in ids if nodes[u].group == g and u != v]
                if others and min(others) < rank[v]:
                    rank[v] = min(others)
            # a group made only of sinks (a data tier) lines up in one column: its deepest member's rank
            tiers = {}
            for v in ids:
                if nodes[v].group is not None:
                    tiers.setdefault(nodes[v].group, []).append(v)
            for g, mem in tiers.items():
                if len(mem) > 1 and all(not succs[v] for v in mem):
                    deep = max(rank[v] for v in mem)
                    for v in mem:
                        if nodes[v].fixed_rank is None:
                            rank[v] = deep
            # peers share a rank; forward edges keep pointing forward
            if sets:
                for _ in range(2 * len(ids) + 2):
                    changed = False
                    for S_ in sets:
                        mx = max(rank[v] for v in S_)
                        for v in S_:
                            if rank[v] != mx:
                                rank[v], changed = mx, True
                    for e in fwd:
                        if rank[e.b] < rank[e.a] + 1 and nodes[e.b].fixed_rank is None:
                            rank[e.b], changed = rank[e.a] + 1, True
                    if not changed:
                        break
        lo = min(rank.values()) if rank else 0
        for v in rank:
            rank[v] -= lo
        self._staggered = 0
        if getattr(self, "_stagger", False) and not self._lane_mode:
            # a very wide rank splits into two staggered rows (every other node one rank further)
            peers = {v for S_ in sets for v in S_}
            byr = {}
            for v in ids:
                byr.setdefault(rank[v], []).append(v)
            shift, new = 0, {}
            for r in sorted(byr):
                mem = byr[r]
                mov = [v for v in mem if nodes[v].fixed_rank is None and v not in peers]
                for v in mem:
                    new[v] = r + shift
                if len(mem) >= 6 and len(mov) >= 4:
                    for i, v in enumerate(mov):
                        if i % 2:
                            new[v] = r + shift + 1
                    shift += 1
                    self._staggered += 1
            rank = new
        R = max(rank.values()) + 1 if rank else 0
        # ---------------- classify edges into hops
        self._back, self._flat, chains = [], [], []
        dummies = {}
        gspan = {}
        for v in ids:
            g = nodes[v].group
            if g is not None:
                lo_, hi_ = gspan.get(g, (rank[v], rank[v]))
                gspan[g] = (min(lo_, rank[v]), max(hi_, rank[v]))
        for e in self._edges:
            ra, rb = rank[e.a], rank[e.b]
            if e.a == e.b:
                self._back.append(e)
                e.kind = "self"
            elif e.idx in back or rb < ra:
                e.kind = "back"
                self._back.append(e)
            elif ra == rb:
                e.kind = "flat"
                self._flat.append(e)
            else:
                e.kind = "fwd"
                ch = [e.a]
                g_a, g_b = nodes[e.a].group, nodes[e.b].group
                for r in range(ra + 1, rb):
                    did = ("~", e.idx, r)
                    # a long edge stays inside the container it leaves or enters while it crosses that
                    # container's ranks, instead of detouring around the rigid block
                    if g_a is not None and r <= gspan[g_a][1]:
                        ga = g_a
                    elif g_b is not None and r >= gspan[g_b][0]:
                        ga = g_b
                    else:
                        ga = None
                    dummies[did] = _N(id=did, dummy=True, group=ga, edge=e)
                    rank[did] = r
                    ch.append(did)
                ch.append(e.b)
                e.chain = ch
                chains.append(e)
        allv = ids + list(dummies)
        info = dict(nodes)
        info.update(dummies)
        up = {k: [] for k in allv}
        down = {k: [] for k in allv}
        for e in chains:
            for u, w in zip(e.chain, e.chain[1:]):
                down[u].append(w)
                up[w].append(u)
        # ---------------- initial order: DFS from sources
        layers = [[] for _ in range(R)]
        seen = set()

        def visit(v):
            stack = [v]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                layers[rank[x]].append(x)
                for w in reversed(down[x]):
                    if w not in seen:
                        stack.append(w)

        for v in ids:
            if not up[v]:
                visit(v)
        for v in allv:
            visit(v)
        layers = self._order(layers, up, down, info, rank)
        if self.keep_order:                              # real nodes keep their declaration order inside a tier
            pos_in = {v: i for i, v in enumerate(ids)}
            for r, lay in enumerate(layers):
                it = iter(sorted([v for v in lay if v in nodes], key=pos_in.get))
                layers[r] = [next(it) if v in nodes else v for v in lay]
        # ---------------- sizes along both axes
        for v in ids:
            n = nodes[v]
            n.sm, n.sc = (n.art.W, n.art.H) if LR else (n.art.H, n.art.W)
        if self.uniform:
            for r in range(R):
                byshape = {}
                for v in layers[r]:
                    if v in nodes:
                        byshape.setdefault(nodes[v].shape, []).append(nodes[v])
                for lst in byshape.values():
                    W = max(n.art.W for n in lst)
                    H = max(n.art.H for n in lst)
                    for n in lst:
                        if n.width is None and n.height is None:
                            n.art.W, n.art.H = W, H
                            if n.shape == "circle":
                                n.art.W = n.art.H = max(W, H)
                            n.sm, n.sc = (n.art.W, n.art.H) if LR else (n.art.H, n.art.W)
        # a node with many edges on one face grows along that face so every edge gets its own port
        nin = {v: 0 for v in ids}
        nout = {v: 0 for v in ids}
        for e in chains:
            nout[e.a] += 1
            nin[e.b] += 1
        for b in self._buses.values():
            for x in b.members:
                if x in nout:
                    nout[x] += 1
        for v in ids:
            n = nodes[v]
            k = max(nin[v], nout[v])
            if k < 4 or n.width is not None or n.height is not None or n.shape not in _ROW_SHAPES:
                continue
            want = (k - 1) * 16 * sp + 2 * max(self.corner, 10) + 12
            if n.sc < want:
                if LR:
                    n.art.H = want
                else:
                    n.art.W = want
                n.sm, n.sc = (n.art.W, n.art.H) if LR else (n.art.H, n.art.W)
        for d in dummies.values():
            d.sm, d.sc = 0.0, 0.0
        # ---------------- cross coordinates
        node_gap = (30 if LR else 40) * sp * self._kc
        dummy_gap = 16 * sp * self._kc
        gpad = 18 * sp
        ghdr = 34 * max(self._sp_mul, 0.82) if any(g.label for g in self.groups.values()) else 0
        hdr_c = ghdr if LR else 0
        hdr_m = 0 if LR else ghdr
        self._gpad, self._ghdr = gpad, ghdr
        pos = self._cross(layers, info, up, down, rank, node_gap, dummy_gap, gpad, hdr_c)
        Cw = None
        if self._lane_mode:
            Cw = self._spread(layers, info, pos, node_gap, dummy_gap, gpad, hdr_c, LR)
        for v in allv:
            info[v].c = pos[v]
        # ---------------- group spans
        gspan = {}
        for v in ids:
            g = nodes[v].group
            if g is not None:
                r0, r1 = gspan.get(g, (rank[v], rank[v]))
                gspan[g] = (min(r0, rank[v]), max(r1, rank[v]))
        # ---------------- ports and hops (cross coordinates only)
        self._rank, self._layers, self._info = rank, layers, info
        gcross = {}
        for g in gspan:
            mem = [info[v] for v in allv if info[v].group == g]
            gcross[g] = (min(x.c - x.sc / 2 for x in mem) - gpad - hdr_c, max(x.c + x.sc / 2 for x in mem) + gpad)
            if Cw is not None:
                gcross[g] = (-gpad - hdr_c, Cw + gpad)
        self._faces = {}
        self._chains = chains
        # buses: a strip in the gap after their deepest member, one tap per member on its outgoing face
        bus_off = [0.0] * R
        bgeo = {}
        tidx = 10 ** 6
        for bid, b in self._buses.items():
            mem = [x for x in b.members if x in rank]
            if not mem:
                continue
            rb = max(rank[x] for x in mem)
            k = sum(1 for o in bgeo.values() if o.r == rb)
            bus_off[rb] += 58 * sp
            taps = []
            for x in mem:
                t = _N(idx=tidx, bus=bid, m=x)
                tidx += 1
                self._faces.setdefault((x, "out"), []).append((t, 0, info[x].c))
                taps.append(t)
            bgeo[bid] = _N(r=rb, k=k, members=mem, taps=taps, r0=min(rank[x] for x in mem))
        self._decisions(chains, info, rank)
        self._lanes(info, rank, layers, gspan, gcross)
        hops = self._hops(chains, info, rank, LR)
        # ---------------- main coordinates
        T = [max([info[v].sm for v in layers[r] if not info[v].dummy] or [0.0]) for r in range(R)]
        base_gap = (64 if LR else 52) * sp * self._km
        need = [base_gap] * max(R - 1, 0)
        ntr = [0] * max(R - 1, 0)
        track_sp = 12 * sp
        for h in hops:
            if h.jog:
                ntr[h.r] = max(ntr[h.r], h.track + 1)
        for r in range(R - 1):
            need[r] = max(need[r], (ntr[r] + 1) * track_sp + 36 * self._sp_mul)
        efs = round(fs * 0.76, 1)
        for e in chains:
            if e.label:
                lw, lh = _measure(e.label, efs, "normal", _fam(th))
                ext = (lw + 16) if LR else (lh + 10)
                rs = [rank[x] for x in e.chain[:-1]]
                r = rs[len(rs) // 2] if not getattr(e, "near_source", False) else rs[0]
                if not getattr(e, "lshape", False):
                    need[r] = max(need[r], ext + 40 * self._sp_mul)
        g_end = [0.0] * R
        g_start = [0.0] * R
        for g, (r0, r1) in gspan.items():
            gl = self.groups[g]
            extra = 0.0
            if gl.label and LR and r0 == r1:
                lw = _measure(str(gl.label).upper(), 13, "bold", _fam(th))[0] + (22 if gl.icon else 0)
                extra = max(0.0, (lw + 2 * gpad - T[r0]) / 2 - gpad)
            g_start[r0] = max(g_start[r0], gpad + hdr_m + extra)
            g_end[r1] = max(g_end[r1], gpad + extra)
        mpos = []
        m = g_start[0] if R else 0.0
        for r in range(R):
            m += T[r] / 2
            mpos.append(m)
            m += T[r] / 2
            if r < R - 1:
                m += g_end[r] + bus_off[r] + need[r] + g_start[r + 1]
        total_m = m + ((g_end[R - 1] + bus_off[R - 1]) if R else 0.0)
        for v in allv:
            info[v].m = mpos[rank[v]]
        self._T, self._mpos = T, mpos
        # tracks → main positions
        for h in hops:
            if h.jog:
                r = h.r
                a = mpos[r] + T[r] / 2 + g_end[r] + bus_off[r]
                b = mpos[r + 1] - T[r + 1] / 2 - g_start[r + 1]
                n = ntr[r]
                tsp = min(track_sp, (b - a - 24) / max(n, 1))
                h.mt = (a + b) / 2 + (h.track - (n - 1) / 2) * tsp
        # bus bars: main position, cross extent, label plate
        bfs_ = efs
        for bid, bg in bgeo.items():
            b = self._buses[bid]
            r = bg.r
            bg.mb = mpos[r] + T[r] / 2 + g_end[r] + (34 + 58 * bg.k) * sp
            bg.cps = [self._port.get((id(t), 0), info[t.m].c) for t in bg.taps]
            lw, lh = _measure(str(b.label), bfs_, "bold", _fam(th))
            bg.plate = (lw + 18, lh + 10)
            ext = (bg.plate[1] if LR else bg.plate[0]) + 14
            bg.c1 = max(bg.cps) + 26
            bg.c0 = min(bg.cps) - 26 - ext
            bg.pc = bg.c0 + ext / 2 + 2
        # ---------------- coordinates → x, y
        flip = self.direction in ("RL", "BT")
        cmin = min([info[v].c - info[v].sc / 2 for v in allv] + [bg.c0 for bg in bgeo.values()] or [0.0])
        group_boxes = {}
        for g, (r0, r1) in gspan.items():
            mem = [nodes[v] for v in ids if nodes[v].group == g] + [d for d in dummies.values() if d.group == g]
            c0 = min(x.c - x.sc / 2 for x in mem) - gpad - hdr_c
            c1 = max(x.c + x.sc / 2 for x in mem) + gpad
            if Cw is not None:
                c0, c1 = gcross[g]
            m0 = mpos[r0] - T[r0] / 2 - g_start[r0]
            m1 = mpos[r1] + T[r1] / 2 + g_end[r1]
            group_boxes[g] = [m0, c0, m1, c1]
            cmin = min(cmin, c0)
        cmax = max([info[v].c + info[v].sc / 2 for v in allv] + [b[3] for b in group_boxes.values()] +
                   [bg.c1 for bg in bgeo.values()] or [0.0])
        for e in self._back:                          # loop edges routed around the outside stay inside the frame
            if getattr(e, "lane", None) is not None and not getattr(e, "short", False) and e.kind != "self":
                cmin, cmax = min(cmin, e.lane - 10), max(cmax, e.lane + 10)
        self._cmin, self._cmax, self._total_m = cmin, cmax, total_m
        # ---------------- folding: long rank sequences wrap into bands (snake order)
        bands = self._cut_bands(R, chains, rank, gspan, bgeo, mpos, T, total_m) if (
            self._bands > 1 and not flip and not self._lane_mode and R > 2) else [(0, R - 1)]
        nb = len(bands)
        band_of = {}
        for j, (ra_, rb_) in enumerate(bands):
            for r in range(ra_, rb_ + 1):
                band_of[r] = j
        if nb > 1:
            starts = [mpos[a] - T[a] / 2 - g_start[a] for a, _ in bands]
            ends = [mpos[b_] + T[b_] / 2 + g_end[b_] + bus_off[b_] for _, b_ in bands]
            starts[0], ends[-1] = 0.0, total_m
            bounds = [(ends[j] + starts[j + 1]) / 2 for j in range(nb - 1)]
            lo_c = [1e18] * nb
            hi_c = [-1e18] * nb

            def widen(j, a, b):
                lo_c[j], hi_c[j] = min(lo_c[j], a), max(hi_c[j], b)

            for v in allv:
                x = info[v]
                widen(band_of[rank[v]], x.c - x.sc / 2, x.c + x.sc / 2)
            for g, (m0, c0, m1, c1) in group_boxes.items():
                widen(band_of[gspan[g][0]], c0, c1)
            for bg in bgeo.values():
                widen(band_of[bg.r], bg.c0, bg.c1)
            for e in self._back:
                ja, jb = band_of[rank[e.a]], band_of[rank[e.b]]
                if ja == jb and getattr(e, "lane", None) is not None and not getattr(e, "short", False):
                    widen(ja, e.lane - 8, e.lane + 8)
                if e.kind == "self":
                    x = info[e.a]
                    widen(ja, x.c - x.sc / 2 - 30 * sp, x.c + x.sc / 2)
            bgap = 64 * sp * self._kc
            offs, acc, gap_mid = [], 0.0, []
            for j in range(nb):
                offs.append(acc - lo_c[j])
                acc += hi_c[j] - lo_c[j] + bgap
                gap_mid.append(acc - bgap / 2)
            Lmax = max(ends[j] - starts[j] for j in range(nb))
            cross_total = acc - bgap

            def xy(mm, cc, j=None):
                if j is None:
                    j = 0
                    while j < nb - 1 and mm > bounds[j]:
                        j += 1
                loc = mm - starts[j]
                if j % 2 and LR:                         # rows snake back; columns restart at the top
                    loc = Lmax - loc
                cc = cc + offs[j]
                return (loc, cc) if LR else (cc, loc)
        else:
            def xy(mm, cc, j=None):
                if flip:
                    mm = total_m - mm
                cc = cc - cmin
                return (mm, cc) if LR else (cc, mm)

        self._xy = xy
        # node boxes
        boxes = {}
        for v in ids:
            n = nodes[v]
            x, y = xy(n.m, n.c, band_of.get(rank[v]))
            n.x, n.y = x, y
            boxes[v] = (x - n.art.W / 2, y - n.art.H / 2, x + n.art.W / 2, y + n.art.H / 2)
        gb = {}
        for g, (m0, c0, m1, c1) in group_boxes.items():
            j = band_of.get(gspan[g][0])
            p0, p1 = xy(m0, c0, j), xy(m1, c1, j)
            gb[g] = (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
        # edge paths (edges between two bands get an obstacle-aware orthogonal route)
        per_gap = {}
        for e in self._edges:
            if nb > 1 and e.a != e.b and band_of[rank[e.a]] != band_of[rank[e.b]]:
                per_gap.setdefault(min(band_of[rank[e.a]], band_of[rank[e.b]]), []).append(e)
        done, pending = {}, []
        for e in self._edges:
            if nb > 1 and e.a != e.b and band_of[rank[e.a]] != band_of[rank[e.b]]:
                e.keys = None
                e.folded = True
                pending.append(e)
                continue
            e.folded = False
            pts = self._edge_points(e, hops, info, rank, LR)
            if pts:
                j = band_of.get(rank[e.a])
                done[e.idx] = [xy(p[0], p[1], j) for p in pts]
        for e in pending:
            ja, jb = band_of[rank[e.a]], band_of[rank[e.b]]
            pts = None
            if abs(ja - jb) == 1:
                lst = per_gap[min(ja, jb)]
                k = lst.index(e)
                mid = gap_mid[min(ja, jb)] + (k - (len(lst) - 1) / 2) * 14 * sp
                pts = _route_fold(nodes[e.a], nodes[e.b], boxes, e.a, e.b, LR, mid)
            if pts is None:
                used = [(p_, q_) for q in done.values() for p_, q_ in zip(q, q[1:])]
                pts = _route_avoid(boxes, e.a, e.b, used) or _route_manual(nodes[e.a], nodes[e.b], boxes, e.a, e.b,
                                                                           "orthogonal")
            done[e.idx] = pts
        paths = [(e, done[e.idx]) for e in self._edges if e.idx in done]
        buses = {}
        for bid, bg in bgeo.items():
            j = band_of.get(bg.r)
            taps = []
            for t, cp in zip(bg.taps, bg.cps):
                x = info[t.m]
                j2 = band_of.get(rank[t.m])
                m0 = x.m + self._boundary(x, cp - x.c, 1)
                if j2 == j:
                    taps.append((t.m, [xy(m0, cp, j), xy(bg.mb, cp, j)]))
            # the label plate sits on the bar where no edge crosses it: its start, between taps, or its end
            pw, ph = bg.plate
            ext = (ph if LR else pw) / 2
            cps = sorted(bg.cps)
            cands = [bg.pc] + [(a + b) / 2 for a, b in zip(cps, cps[1:]) if b - a > 2 * ext + 20] +                 [max(cps) + 14 + ext]
            segs = [(p_, q_) for e_, pts_ in paths for p_, q_ in zip(pts_, pts_[1:])]
            pc = bg.pc
            for c in cands:
                x_, y_ = xy(bg.mb, c, j)
                box_ = (x_ - pw / 2 - 3, y_ - ph / 2 - 3, x_ + pw / 2 + 3, y_ + ph / 2 + 3)
                if not any(_seg_hits_box(p_, q_, box_) for p_, q_ in segs) and                         not any(_rect_hit(box_, b_) for b_ in boxes.values()):
                    pc = c
                    break
            c1 = max(bg.c1, pc + ext + 8)
            buses[bid] = dict(bar=[xy(bg.mb, bg.c0 if pc == bg.pc else min(cps) - 26, j), xy(bg.mb, c1, j)],
                              taps=taps, label=xy(bg.mb, pc, j), plate=bg.plate)
        if nb > 1:
            W, H = (Lmax, cross_total) if LR else (cross_total, Lmax)
        else:
            W = total_m if LR else cmax - cmin
            H = cmax - cmin if LR else total_m
        bal_ = 1.0
        if nb > 1:
            lens = [ends[j] - starts[j] for j in range(nb)]
            bal_ = min(lens) / max(max(lens), 1e-9)
        cut_edges = sum(1 for e in self._edges if getattr(e, "folded", False))
        cut_back = sum(1 for e in self._edges if getattr(e, "folded", False) and e.kind == "back")
        self._res = dict(boxes=boxes, groups=gb, paths=paths, W=W, H=H, buses=buses, bands=nb, cut_edges=cut_edges,
                         cut_back=cut_back, band_balance=bal_)
        return self._res

    def _cut_bands(self, R, chains, rank, gspan, bgeo, mpos, T, total_m):
        """Split ranks 0..R-1 into up to ``self._bands`` bands at gaps that no long edge, container or bus
        straddles; cuts near equal lengths. Returns [(first_rank, last_rank)]."""
        allowed = []
        for r in range(R - 1):
            if any(rank[e.a] <= r < rank[e.b] and rank[e.b] - rank[e.a] > 1 for e in chains):
                continue
            if any(r0 <= r < r1 for r0, r1 in gspan.values()):
                continue
            if any(bg.r0 <= r <= bg.r for bg in bgeo.values()):
                continue
            allowed.append(r)
        cross = {r: sum(1 for e in self._edges if e.a != e.b and min(rank[e.a], rank[e.b]) <= r
                        < max(rank[e.a], rank[e.b])) for r in allowed}
        cuts = []
        for j in range(1, self._bands):
            tgt = total_m * j / self._bands
            cand = [r for r in allowed if not cuts or r > cuts[-1]]
            if not cand:
                break
            cuts.append(min(cand, key=lambda r: abs(mpos[r] + T[r] / 2 - tgt) / max(total_m, 1.0)
                            + 0.06 * max(0, cross[r] - 1)))
        bands, start = [], 0
        for r in cuts:
            bands.append((start, r))
            start = r + 1
        bands.append((start, R - 1))
        return bands

    def _spread(self, layers, info, pos, node_gap, dummy_gap, gpad, hdr_c, LR):
        """Lanes: every rank spans the same width Cw with its nodes spread evenly; long edges pass through
        the gaps. Writes ``pos`` and returns Cw."""
        dg = dummy_gap + 6
        rows = []
        for L in layers:
            ns = [v for v in L if not info[v].dummy]
            nd = len(L) - len(ns)
            rows.append(sum(info[v].sc for v in ns) + max(len(ns) - 1, 0) * node_gap + nd * 2 * dg)
        Cw = max(rows + [1.0])
        for g in self.groups.values():                  # the band label fits on one line
            if g.label:
                lw = _measure(str(g.label).upper(), 13, "bold", _fam(self.theme))[0] + (24 if g.icon else 0) + 32
                Cw = max(Cw, lw - 2 * gpad)
        for L in layers:
            ns = [v for v in L if not info[v].dummy]
            cp = {}
            if ns:
                gap = (Cw - sum(info[v].sc for v in ns)) / len(ns)
                x = gap / 2
                for v in ns:
                    cp[v] = x + info[v].sc / 2
                    x += info[v].sc + gap
            slots = {}
            left = None
            for i, v in enumerate(L):
                if not info[v].dummy:
                    left = v
                    continue
                right = next((w for w in L[i + 1:] if not info[w].dummy), None)
                slots.setdefault((left, right), []).append(v)
            for (left, right), ds in slots.items():
                lo = cp[left] + info[left].sc / 2 if left is not None else None
                hi = cp[right] - info[right].sc / 2 if right is not None else None
                if lo is None:
                    lo = min(0.0, (hi if hi is not None else Cw) - 2 * dg * len(ds))
                if hi is None:
                    hi = max(Cw, lo + 2 * dg * len(ds))
                for k, v in enumerate(ds):
                    cp[v] = lo + (hi - lo) * (k + 1) / (len(ds) + 1)
            pos.update(cp)
        return Cw

    # ---- ordering (barycentre sweeps, groups kept contiguous)
    def _order(self, layers, up, down, info, rank):
        nodes = self._nodes
        R = len(layers)
        pinned = {v: nodes[v].at[1] for v in nodes if nodes[v].at is not None}

        def npos(L):
            return {v: (i + 0.5) / len(L) for i, v in enumerate(L)}

        def crossings(lys):
            total = 0
            for r in range(len(lys) - 1):
                p = {v: i for i, v in enumerate(lys[r + 1])}
                es = []
                for i, v in enumerate(lys[r]):
                    for w in down[v]:
                        if w in p:
                            es.append((i, p[w]))
                es.sort()
                for a in range(len(es)):
                    for b in range(a + 1, len(es)):
                        if es[a][0] < es[b][0] and es[a][1] > es[b][1]:
                            total += 1
            return total

        def gkeys(lys):
            P = {}
            for L in lys:
                P.update(npos(L))
            acc = {}
            for v, p in P.items():
                g = info[v].group
                if g is not None:
                    acc.setdefault(g, []).append(p)
            return {g: sum(v) / len(v) for g, v in acc.items()}

        def sort_layer(L, bary):
            gk = self._gk
            blocks, seen = [], set()
            for v in L:
                g = info[v].group
                if g is None:
                    blocks.append((pinned.get(v, bary[v]), [v]))
                elif g not in seen:
                    seen.add(g)
                    mem = sorted([x for x in L if info[x].group == g], key=lambda x: (pinned.get(x, bary[x])))
                    blocks.append((gk.get(g, 0.5) if not any(x in pinned for x in mem)
                                   else min(pinned.get(x, 1e9) for x in mem), mem))
            blocks.sort(key=lambda b: b[0])
            return [v for _, mem in blocks for v in mem]

        cur = [list(L) for L in layers]
        self._gk = gkeys(cur)
        cur = [sort_layer(L, {v: (i + 0.5) / len(L) for i, v in enumerate(L)}) for L in cur]
        best = [list(L) for L in cur]
        best_c = crossings(best)
        for it in range(14):
            self._gk = gkeys(cur)
            rng = range(1, R) if it % 2 == 0 else range(R - 2, -1, -1)
            for r in rng:
                ref = cur[r - 1] if it % 2 == 0 else cur[r + 1]
                p = npos(ref)
                nb = up if it % 2 == 0 else down
                old = npos(cur[r])
                bary = {}
                for v in cur[r]:
                    ps = [p[w] for w in nb[v] if w in p]
                    bary[v] = sum(ps) / len(ps) if ps else old[v]
                cur[r] = sort_layer(cur[r], bary)
            c = crossings(cur)
            if c < best_c or (c == best_c and it < 2):
                best_c, best = c, [list(L) for L in cur]
        self._crossings = best_c
        return best

    # ---- cross coordinates (groups are rigid blocks laid out first)
    def _cross(self, layers, info, up, down, rank, node_gap, dummy_gap, gpad, hdr_c):

        def gap(a, b):
            da, db = info[a].dummy, info[b].dummy
            if da and db:
                return dummy_gap * 0.85
            if da or db:
                return dummy_gap + 6
            return node_gap

        def weight(a, b):
            da, db = info[a].dummy, info[b].dummy
            return 8.0 if (da and db) else (2.0 if (da or db) else 1.0)

        # inner layout of each group
        rel, gext = {}, {}
        groups = {}
        for L in layers:
            for v in L:
                g = info[v].group
                if g is not None:
                    groups.setdefault(g, []).append(v)
        for g, mem in groups.items():
            ms = set(mem)
            sub_layers = [[v for v in L if v in ms] for L in layers]
            items = {v: dict(ranks=[rank[v]], lo=-info[v].sc / 2, hi=info[v].sc / 2) for v in mem}
            links = [(u, 0.0, w, 0.0, weight(u, w)) for u in mem for w in down[u] if w in ms]
            p = _place(items, sub_layers, links, gap)
            c0 = min(p[v] - info[v].sc / 2 for v in mem)
            for v in mem:
                rel[v] = p[v] - c0
            gext[g] = max(rel[v] + info[v].sc / 2 for v in mem)
        # global items
        items, item_of = {}, {}
        for L in layers:
            for v in L:
                g = info[v].group
                if g is None:
                    items[v] = dict(ranks=[rank[v]], lo=-info[v].sc / 2, hi=info[v].sc / 2)
                    item_of[v] = (v, 0.0)
                else:
                    key = ("G", g)
                    if key not in items:
                        items[key] = dict(ranks=[], lo=-gpad - hdr_c, hi=gext[g] + gpad)
                    if rank[v] not in items[key]["ranks"]:
                        items[key]["ranks"].append(rank[v])
                    item_of[v] = (key, rel[v])
        glayers = []
        for L in layers:
            row = []
            for v in L:
                it = item_of[v][0]
                if not row or row[-1] != it:
                    if it not in row:
                        row.append(it)
            glayers.append(row)
        # one global order of the group blocks (mean position over their ranks)
        def is_g(k):
            return isinstance(k, tuple) and len(k) == 2 and k[0] == "G"

        gpos = {}
        for row in glayers:
            for i, k in enumerate(row):
                if is_g(k):
                    gpos.setdefault(k, []).append((i + 0.5) / len(row))
        gkey = {k: sum(v) / len(v) for k, v in gpos.items()}
        # fill rank gaps in a group's span so nothing slips between its ranks
        for key, it in items.items():
            if is_g(key):
                r0, r1 = min(it["ranks"]), max(it["ranks"])
                for r in range(r0, r1 + 1):
                    if r not in it["ranks"]:
                        it["ranks"].append(r)
                        glayers[r].append(key)
        for r, row in enumerate(glayers):
            slots = [i for i, k in enumerate(row) if is_g(k)]
            gs = sorted([row[i] for i in slots], key=lambda k: gkey.get(k, 0.5))
            if not slots:
                continue
            # keep singletons in place relative to the (re-sorted) group blocks by their old index
            new = list(row)
            for i, k in zip(slots, gs):
                new[i] = k
            glayers[r] = new
        links = []
        for u in info:
            for w in down[u]:
                iu, ou = item_of[u]
                iw, ow = item_of[w]
                if iu != iw:
                    links.append((iu, ou, iw, ow, weight(u, w)))

        def igap(a, b):
            if isinstance(a, tuple) and a and a[0] == "G" or isinstance(b, tuple) and b and b[0] == "G":
                return node_gap
            return gap(a, b)

        p = _place(items, glayers, links, igap)
        pos = {}
        for v, (it, off) in item_of.items():
            if isinstance(it, tuple) and it and it[0] == "G":
                pos[v] = p[it] + off
            else:
                pos[v] = p[it]
        # pinned rows: exact grid
        pinned = {v: self._nodes[v].at[1] for v in self._nodes if self._nodes[v].at is not None}
        if pinned:
            pitch = max(info[v].sc for v in pinned) + node_gap
            for v, row in pinned.items():
                pos[v] = row * pitch
        return pos

    # ---- ports, hops, tracks
    def _hops(self, chains, info, rank, LR):
        nodes = self._nodes
        hops = []
        faces = self._faces                               # (node, face) -> list of (hop, end, key)
        lshape = self._lshape

        def reg(v, face, h, end, key):
            faces.setdefault((v, face), []).append((h, end, key))

        for e in chains:
            e.near_source = nodes[e.a].shape == "diamond"
            e.hops = []
            for k, (u, w) in enumerate(zip(e.chain, e.chain[1:])):
                h = _N(e=e, u=u, w=w, r=rank[u], k=k, jog=False, track=0, lside=None)
                if k == 0 and e.idx in lshape:
                    h.lside = lshape[e.idx]
                else:
                    reg(u, "out", h, 0, info[w].c)
                reg(w, "in", h, 1, info[u].c if h.lside is None else info[u].c + h.lside * 1e-3)
                e.hops.append(h)
                hops.append(h)
        for e in self._flat:
            a, b = info[e.a], info[e.b]
            side = 1 if b.c > a.c else -1
            e.side = side
            between = [v for v in self._layers[rank[e.a]] if min(a.c, b.c) < info[v].c < max(a.c, b.c)
                       and not info[v].dummy]
            e.adjacent = not between
            if e.adjacent:
                reg(e.a, "hi" if side > 0 else "lo", e, 0, 0)
                reg(e.b, "lo" if side > 0 else "hi", e, 1, 0)
            else:
                reg(e.a, "out", e, 0, b.c)
                reg(e.b, "out", e, 1, a.c)
        # port positions
        port = {}

        def usable(v, face):
            x = info[v]
            if x.dummy:
                return 0.0
            s = x.sc if face in ("in", "out") else x.sm
            k = 0.5 if x.shape in ("diamond", "circle", "ellipse", "cloud", "hexagon") else 1.0
            if x.shape == "pill" and face in ("lo", "hi"):
                k = 0.3
            return max(0.0, (s - 2 * max(self.corner, 10)) * k)

        def center(v, face):
            x = info[v]
            return x.c if face in ("in", "out") else 0.0      # side faces: offset from the rank centre

        singles = []
        for (v, face), lst in faces.items():
            lst.sort(key=lambda t: (t[2], t[0].e.idx if hasattr(t[0], "e") else t[0].idx))
            n = len(lst)
            u = usable(v, face)
            if n == 1:
                singles.append((v, face))
                continue
            spc = min(18 * self.spacing + 4, u / max(n - 1, 1))
            if not info[v].dummy and info[v].shape in ("diamond", "circle") and face in ("in", "out"):
                spc = 0.0                                 # merge into the corner / centre point
            for i, (h, end, key) in enumerate(lst):
                port[(id(h), end)] = center(v, face) + (i - (n - 1) / 2) * spc
        for (v, face) in singles:
            h, end, key = faces[(v, face)][0]
            port[(id(h), end)] = center(v, face)
        # straighten single-port faces
        for (v, face) in singles:
            h, end, key = faces[(v, face)][0]
            if not hasattr(h, "e") or face not in ("in", "out"):
                continue
            other = h.w if end == 0 else h.u
            oface = "in" if end == 0 else "out"
            if h.lside is not None:
                continue
            oc = port.get((id(h), 1 - end))
            o_single = info[other].dummy or len(faces.get((other, oface), [])) == 1
            half = usable(v, face) / 2
            c0 = info[v].c
            if info[other].dummy:
                target = info[other].c
                if abs(target - c0) <= half:
                    port[(id(h), end)] = target
            elif o_single:
                oh = usable(other, oface) / 2
                oc0 = info[other].c
                lo_, hi_ = max(c0 - half, oc0 - oh), min(c0 + half, oc0 + oh)
                if lo_ <= hi_:
                    c = min(max((c0 + oc0) / 2, lo_), hi_)
                    port[(id(h), end)] = c
                    port[(id(h), 1 - end)] = c
            elif oc is not None and abs(oc - c0) <= half:
                port[(id(h), end)] = oc
        def lane_clear(dm, c):
            for d_ in dm:
                for o in self._layers[rank[d_]]:
                    if o != d_ and abs(info[o].c - c) < info[o].sc / 2 + (10 if info[o].dummy else 14):
                        return False
            return True

        # a dummy chain whose ranks differ by a few units runs in one straight lane (no small kinks)
        for e in chains:
            dm = e.chain[1:-1]
            if len(dm) < 2:
                continue
            cs = [info[d_].c for d_ in dm]
            if max(cs) - min(cs) < 0.5 or max(cs) - min(cs) > 30:
                continue
            for c in sorted(set(cs), key=lambda c: sum(abs(c - x) for x in cs)):
                if lane_clear(dm, c):
                    for d_ in dm:
                        info[d_].c = c
                    break
        # slide dummy chains onto the source / target port when that keeps them clear (no tiny jogs)
        for e in chains:
            if len(e.chain) < 3 or getattr(e, "lshape", False):
                continue
            dm = e.chain[1:-1]
            h0, h1 = e.hops[0], e.hops[-1]
            cands = [port.get((id(h0), 0)), port.get((id(h1), 1))]
            dc = info[dm[0]].c
            for c in cands:
                if c is None or abs(c - dc) < 0.5 or abs(c - dc) > 40:
                    continue
                ok = True
                for d_ in dm:
                    for o in self._layers[rank[d_]]:
                        if o == d_:
                            continue
                        x = info[o]
                        if abs(x.c - c) < x.sc / 2 + (10 if x.dummy else 14):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                for d_ in dm:
                    info[d_].c = c
                b = e.chain[-1]
                if len(faces.get((b, "in"), [])) == 1 and abs(c - info[b].c) <= usable(b, "in") / 2:
                    port[(id(h1), 1)] = c
                a = e.chain[0]
                if len(faces.get((a, "out"), [])) == 1 and abs(c - info[a].c) <= usable(a, "out") / 2:
                    port[(id(h0), 0)] = c
                break
        self._port = port
        # hop cross coordinates + jogs
        for h in hops:
            u, w = info[h.u], info[h.w]
            c1 = u.c if u.dummy else port.get((id(h), 0), u.c)
            if h.lside is not None:
                c1 = u.c + h.lside * u.sc / 2
            c2 = w.c if w.dummy else port.get((id(h), 1), w.c)
            h.c1, h.c2 = c1, c2
            h.jog = h.lside is None and abs(c1 - c2) > 0.5
        # tracks per gap
        bygap = {}
        for h in hops:
            if h.jog:
                bygap.setdefault(h.r, []).append(h)
        for r, lst in bygap.items():
            down_ = sorted([h for h in lst if h.c2 > h.c1], key=lambda h: -h.c1)
            up_ = sorted([h for h in lst if h.c2 < h.c1], key=lambda h: h.c1)
            done = []
            eps = 6.0
            for h in down_ + up_:
                a0, a1 = min(h.c1, h.c2) - eps, max(h.c1, h.c2) + eps
                t = -1
                for o in done:
                    b0, b1 = min(o.c1, o.c2), max(o.c1, o.c2)
                    if a0 < b1 and b0 < a1:
                        t = max(t, o.track)
                h.track = t + 1
                done.append(h)
            # compact: reuse lowest free track when no conflict
        return hops

    def _lclear(self, v, tc, side, info, rank):
        n = info[v]
        a = n.c + side * n.sc / 2
        lo, hi = min(a, tc) - 10, max(a, tc) + 10
        for x in self._layers[rank[v]]:
            if x == v:
                continue
            o = info[x]
            if o.c + o.sc / 2 + 4 > lo and o.c - o.sc / 2 - 4 < hi:
                return False
        return True

    def _decisions(self, chains, info, rank):
        """Diamonds with 2+ successors send the outer branches out of their side corners (L-shaped)."""
        self._lshape = {}
        self._dog = {}
        self._used_side = set()
        for v, n in self._nodes.items():
            if n.shape != "diamond":
                continue
            outs = [e for e in chains if e.a == v]
            if len(outs) < 2:
                continue
            outs.sort(key=lambda e: (info[e.chain[1]].c, e.idx))
            for e, side in ((outs[0], -1), (outs[-1], 1)):
                w = e.chain[1]
                if side * (info[w].c - n.c) < n.sc * 0.22:
                    continue
                if self._lclear(v, info[w].c, side, info, rank):
                    self._lshape[e.idx] = side
                    self._used_side.add((v, side))
                    e.lshape = True
                    continue
                # a neighbour blocks the straight L: step out of the corner into the gap before it
                corner = n.c + side * n.sc / 2
                near = [info[o].c - side * info[o].sc / 2 for o in self._layers[rank[v]]
                        if o != v and side * (info[o].c - n.c) > 0]
                if not near:
                    continue
                edge_ = min(near, key=lambda c: side * c)
                if side * (edge_ - corner) < 44 * self.spacing:
                    continue
                xm = (corner + edge_) / 2
                if side * (info[w].c - xm) < 8:
                    continue
                self._lshape[e.idx] = side
                self._dog[e.idx] = xm
                self._used_side.add((v, side))
                e.lshape = True

    def _lanes(self, info, rank, layers, gspan, gcross):
        """Lanes (cross axis) for back edges routed around the outside; registers their face ports."""
        used = {1: [], -1: []}
        faces = self._faces

        def reg(v, face, e, end, key):
            faces.setdefault((v, face), []).append((e, end, key))

        def extreme(v, side):
            x = info[v]
            if (v, side) in self._used_side:
                return False
            return all((info[o].c + side * info[o].sc / 2 - (x.c + side * x.sc / 2)) * side <= 1
                       for o in layers[rank[v]] if o != v)

        for e in self._back:
            if e.kind == "self":
                continue
            ra, rb = sorted((rank[e.a], rank[e.b]))
            if self._short_back(e, info, rank, layers):
                continue
            best = None
            for side in (1, -1):
                ext = []
                for r in range(ra, rb + 1):
                    for v in layers[r]:
                        x = info[v]
                        ext.append(x.c + side * x.sc / 2)
                for g, (g0, g1) in gspan.items():
                    if g0 <= rb and ra <= g1:
                        ext.append(gcross[g][1] if side > 0 else gcross[g][0])
                lim = max(ext) if side > 0 else min(ext)
                k = sum(1 for (s0, s1) in used[side] if s0 <= rb and ra <= s1)
                lane = lim + side * (26 + 14 * k) * self.spacing
                a, b = info[e.a], info[e.b]
                cost = abs(lane - a.c) + abs(lane - b.c)
                for v in (e.a, e.b):
                    if not extreme(v, side):
                        cost += 90 + (150 if self._nodes[v].shape in ("diamond", "circle") else 0)
                if best is None or cost < best[0] - 1e-6:
                    best = (cost, side, lane)
            _, side, lane = best
            used[side].append((ra, rb))
            e.side, e.lane = side, lane
            e.ext_a, e.ext_b = extreme(e.a, side), extreme(e.b, side)
            fs = "hi" if side > 0 else "lo"
            reg(e.a, fs if e.ext_a else "out", e, 0, rank[e.b] if e.ext_a else lane)
            reg(e.b, fs if e.ext_b else "in", e, 1, rank[e.a] if e.ext_b else lane)

    def _short_back(self, e, info, rank, layers):
        """Route a back edge straight across when the target sits to one side of the source.

        The edge leaves the source's side facing the target, runs back through the gap between the
        two nodes and enters the target's facing side: no detour around the whole figure. Taken only
        when both sides are free and the path crosses no node and no forward edge."""
        a, b = info[e.a], info[e.b]
        if abs(b.c - a.c) < 1:
            return False
        s = 1 if b.c > a.c else -1
        ea = a.c + s * a.sc / 2
        eb = b.c - s * b.sc / 2
        if s * (eb - ea) < 44 * self.spacing:
            return False
        fa, fb = ("hi", "lo") if s > 0 else ("lo", "hi")
        if (e.a, s) in self._used_side or (e.b, -s) in self._used_side:
            return False
        if self._faces.get((e.a, fa)) or self._faces.get((e.b, fb)):
            return False
        cx = (ea + eb) / 2
        if not (self._lclear(e.a, cx, s, info, rank) and self._lclear(e.b, cx, -s, info, rank)):
            return False
        ra, rb = sorted((rank[e.a], rank[e.b]))
        for r in range(ra + 1, rb):
            for v in layers[r]:
                x = info[v]
                if abs(x.c - cx) < x.sc / 2 + 14:
                    return False
        for f in getattr(self, "_chains", []):          # forward hops whose jog would cut the vertical
            for u, w in zip(f.chain, f.chain[1:]):
                if ra <= rank[u] < rb:
                    c1, c2 = sorted((info[u].c, info[w].c))
                    if c1 - 8 < cx < c2 + 8 and c2 - c1 > 0.5:
                        return False
        for o in self._back:                            # other short back edges sharing the gap
            if getattr(o, "short", False):
                oa, ob = sorted((rank[o.a], rank[o.b]))
                if oa < rb and ra < ob and abs(o.cx - cx) < 20:
                    return False
        e.short, e.side, e.cx = True, s, cx
        self._used_side.add((e.a, s))
        self._used_side.add((e.b, -s))
        self._faces.setdefault((e.a, fa), []).append((e, 0, 0.0))
        self._faces.setdefault((e.b, fb), []).append((e, 1, 0.0))
        return True

    def _side_extent(self, n, m_off, side):
        """Cross-axis distance from the node centre to its outline on ``side`` at main offset ``m_off``."""
        if n.dummy:
            return 0.0
        if n.shape not in ("diamond", "ellipse", "circle", "cloud", "parallelogram", "hexagon"):
            return n.sc / 2
        LR = self.direction in ("LR", "RL")
        if self.direction in ("RL", "BT"):
            m_off = -m_off
        poly = n.art.outline(0.0, 0.0)
        best = 0.0
        for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
            if LR:
                a1, b1, a2, b2 = x1, y1, x2, y2
            else:
                a1, b1, a2, b2 = y1, x1, y2, x2
            if (a1 - m_off) * (a2 - m_off) <= 0 and abs(a2 - a1) > 1e-9:
                t = (m_off - a1) / (a2 - a1)
                cc = b1 + (b2 - b1) * t
                if cc * side > best:
                    best = cc * side
        return best

    def _boundary(self, n, c_off, direction):
        """Main-axis distance from the node centre to its outline at cross offset ``c_off``."""
        LR = self.direction in ("LR", "RL")
        if n.dummy:
            return 0.0
        if self.direction in ("RL", "BT"):
            direction = -direction
        if n.shape in ("box", "rounded", "note", "document", "pill", "cylinder"):
            if n.shape == "pill" and LR:
                r = n.art.H / 2
                inner = n.art.W / 2 - r
                return inner + math.sqrt(max(r * r - c_off * c_off, 0.0))
            if n.shape == "cylinder" and not LR:
                cap = min(max(n.art.W * 0.09, 7.0), 14.0)
                return n.sm / 2 - cap + cap * math.sqrt(max(1 - (2 * c_off / n.art.W) ** 2, 0.0))
            return n.sm / 2
        poly = n.art.outline(0.0, 0.0)
        best = 0.0
        for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
            if LR:
                a1, b1, a2, b2 = y1, x1, y2, x2
            else:
                a1, b1, a2, b2 = x1, y1, x2, y2
            if (a1 - c_off) * (a2 - c_off) <= 0 and abs(a2 - a1) > 1e-9:
                t = (c_off - a1) / (a2 - a1)
                mm = b1 + (b2 - b1) * t
                if mm * direction > best:
                    best = mm * direction
        return best

    def _edge_points(self, e, hops, info, rank, LR):
        n = info
        if e.kind == "fwd":
            pts, keys = [], []
            for h in e.hops:
                u, w = n[h.u], n[h.w]
                if h.lside is not None:
                    s = h.lside
                    p0 = (u.m, u.c + s * self._side_extent(u, 0.0, s))
                    m2 = w.m - self._boundary(w, h.c2 - w.c, -1)
                    seg = [p0, (u.m, h.c2), (m2, h.c2)]
                    kseg = [(p0[0], p0[1], "Q"), (u.m, h.c2, "ctrl"), (m2, h.c2, "L")]
                else:
                    m1 = u.m + (self._T[h.r] / 2 if u.dummy else self._boundary(u, h.c1 - u.c, 1))
                    m2 = w.m - (self._T[rank[h.w]] / 2 if w.dummy else self._boundary(w, h.c2 - w.c, -1))
                    if u.dummy:
                        kseg = [(u.m - self._T[h.r] / 2, h.c1, "L"), (m1, h.c1, "S")]
                    else:
                        kseg = [(m1, h.c1, "S")]
                    kseg.append((m2, h.c2, "L"))
                    if h.jog:
                        seg = [(m1, h.c1), (h.mt, h.c1), (h.mt, h.c2), (m2, h.c2)]
                    else:
                        seg = [(m1, h.c1), (m2, h.c2)]
                if pts and abs(pts[-1][0] - seg[0][0]) < 1e-6 and abs(pts[-1][1] - seg[0][1]) < 1e-6:
                    seg = seg[1:]
                if keys and abs(keys[-1][0] - kseg[0][0]) < 1e-6 and abs(keys[-1][1] - kseg[0][1]) < 1e-6:
                    keys[-1] = kseg[0]
                    kseg = kseg[1:]
                pts += seg
                keys += kseg
            e.keys = keys
            return pts
        a, b = n[e.a], n[e.b]
        if e.kind == "flat":
            if e.adjacent:
                s = e.side
                pa = a.m + self._port.get((id(e), 0), 0.0)
                pb = b.m + self._port.get((id(e), 1), 0.0)
                mm = (pa + pb) / 2 if abs(pa - pb) < 30 else None
                if mm is not None:
                    return [(mm, a.c + s * a.sc / 2), (mm, b.c - s * b.sc / 2)]
                return [(pa, a.c + s * a.sc / 2), (pa, (a.c + b.c) / 2), (pb, (a.c + b.c) / 2),
                        (pb, b.c - s * b.sc / 2)]
            ca = self._port.get((id(e), 0), a.c)
            cb = self._port.get((id(e), 1), b.c)
            m1 = a.m + self._boundary(a, ca - a.c, 1)
            m2 = b.m + self._boundary(b, cb - b.c, 1)
            mm = max(m1, m2) + 26 * self.spacing
            return [(m1, ca), (mm, ca), (mm, cb), (m2, cb)]
        if e.kind == "self":
            q = a.sc * 0.22
            L = 26 * self.spacing
            m1 = a.m + self._boundary(a, -q, 1)
            top = a.c - a.sc / 2 - L
            mo = a.sm * 0.18
            return [(m1, a.c - q), (m1 + L, a.c - q), (m1 + L, top), (a.m + mo, top),
                    (a.m + mo, a.c - self._side_extent(a, mo, -1))]
        P = self._port
        if getattr(e, "short", False):
            s = e.side
            pa = P.get((id(e), 0), 0.0) if a.shape not in _POINTY else 0.0
            pb = P.get((id(e), 1), 0.0) if b.shape not in _POINTY else 0.0
            return [(a.m + pa, a.c + s * self._side_extent(a, pa, s)), (a.m + pa, e.cx),
                    (b.m + pb, e.cx), (b.m + pb, b.c - s * self._side_extent(b, pb, -s))]
        # back edge around the outside: source a (lower rank side) → target b
        s, lane = e.side, e.lane
        T = self._T
        gapoff = 20 * self.spacing
        pts = []
        if e.ext_a:
            mm = a.m + P.get((id(e), 0), 0.0)
            pts += [(mm, a.c + s * self._side_extent(a, mm - a.m, s)), (mm, lane)]
        else:
            cc = P.get((id(e), 0), a.c)
            mo = a.m + T[rank[e.a]] / 2 + gapoff
            pts += [(a.m + self._boundary(a, cc - a.c, 1), cc), (mo, cc), (mo, lane)]
        if e.ext_b:
            mm = b.m + P.get((id(e), 1), 0.0)
            pts += [(mm, lane), (mm, b.c + s * self._side_extent(b, mm - b.m, s))]
        else:
            cc = P.get((id(e), 1), b.c)
            mo = b.m - T[rank[e.b]] / 2 - gapoff
            pts += [(mo, lane), (mo, cc), (b.m - self._boundary(b, cc - b.c, -1), cc)]
        return pts

    def _layout_manual(self):
        """All nodes have ``pos``: route edges with an obstacle-aware orthogonal router."""
        nodes = self._nodes
        boxes = {}
        for v, n in nodes.items():
            x, y = n.pos
            n.x, n.y = float(x), float(y)
            boxes[v] = (n.x - n.art.W / 2, n.y - n.art.H / 2, n.x + n.art.W / 2, n.y + n.art.H / 2)
        x0 = min(b[0] for b in boxes.values()) - 20
        y0 = min(b[1] for b in boxes.values()) - 20
        for v, n in nodes.items():
            n.x, n.y = n.x - x0, n.y - y0
            b = boxes[v]
            boxes[v] = (b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0)
        gb = {}
        for g in self.groups:
            mem = [boxes[v] for v in nodes if nodes[v].group == g]
            if mem:
                gb[g] = (min(b[0] for b in mem) - 18, min(b[1] for b in mem) - 18 - 34,
                         max(b[2] for b in mem) + 18, max(b[3] for b in mem) + 18)
        paths = []
        for e in self._edges:
            e.kind = "manual"
            paths.append((e, _route_manual(nodes[e.a], nodes[e.b], boxes, e.a, e.b, self.routing)))
        W = max([b[2] for b in boxes.values()] + [b[2] for b in gb.values()]) + 20
        H = max([b[3] for b in boxes.values()] + [b[3] for b in gb.values()]) + 20
        self._res = dict(boxes=boxes, groups=gb, paths=paths, W=W, H=H)
        return self._res

    # ---- drawing
    def _size(self):
        res = self._layout()
        return res["W"], res["H"]

    def _paint(self, dwg, ox, oy):
        th = self.theme
        res = self._res
        fam = _fam(th)
        LR = self.direction in ("LR", "RL")

        def sh(b):
            return (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy)

        self.boxes = {k: sh(b) for k, b in res["boxes"].items()}
        self.group_boxes = {k: sh(b) for k, b in res["groups"].items()}
        # containers
        if res["groups"]:
            with dwg.group("containers"):
                for gid, b in sorted(self.group_boxes.items(), key=lambda kv: self.groups[kv[0]].order):
                    g = self.groups[gid]
                    col = self._gcolor(g)
                    dwg.rect(b[0], b[1], b[2] - b[0], b[3] - b[1], r=16,
                             fill=_mix(col, th.background, 0.075 if th.dark else 0.055),
                             stroke=_mix(col, th.background, 0.45 if th.dark else 0.38), stroke_width=1.25,
                             dash=(6, 4) if g.style == "dashed" else None)
                    if g.label:
                        tx = b[0] + 16
                        if g.icon:
                            dwg.icon(g.icon, tx + 8, b[1] + 18, 17, color=col, stroke_width=2.2)
                            tx += 24
                        dwg.text(tx, b[1] + 18, str(g.label).upper(), size=13, weight="bold", fill=col,
                                 anchor="lm", letter_spacing=1.1, family=fam)
        # edges
        ecol = _edge_color(th)
        plates = []
        node_boxes = list(self.boxes.values())
        with dwg.group("edges"):
            for e, pts in res["paths"]:
                pts = [(x + ox, y + oy) for x, y in pts]
                col = _accent(th, e.color) if e.color is not None else ecol
                sw = e.width or 1.8
                hs = 9 + sw * 2.2
                arrow = e.arrow or "none"
                want_end, want_start = arrow in ("end", "both"), arrow in ("start", "both")
                tr = _arrow_trim(e.head, hs)
                p_end, p_start = pts[-1], pts[0]
                keys = getattr(e, "keys", None)
                if self.routing in ("curved", "straight") and e.kind == "fwd" and keys:
                    kx = [self._xy(k[0], k[1]) + (k[2] if self.routing == "curved" else
                                                  ("L" if k[2] != "ctrl" else "ctrl"),) for k in keys]
                    kx = [k for k in kx if not (self.routing == "straight" and k[2] == "ctrl")]
                    kx = [(k[0] + ox, k[1] + oy, k[2]) for k in kx]
                    path, start_dir, end_dir, poly = _keys_path(kx, LR, tr if want_start else 0,
                                                                tr if want_end else 0)
                    pts = poly
                else:
                    end_dir = start_dir = None
                    a, b = pts[-2], pts[-1]
                    L = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1e-9)
                    end_dir = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
                    a, b = pts[1], pts[0]
                    L = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1e-9)
                    start_dir = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
                    draw_pts = _trim_poly(pts, tr if want_start else 0, tr if want_end else 0)
                    path = _round_poly(draw_pts, self.corner, closed=False)
                cap = "round" if e.style == "dotted" else "butt"
                dwg.path(path, stroke=col, stroke_width=sw, fill=None, dash=_dash(e.style, sw), cap=cap,
                         join="round")
                if want_end:
                    _arrowhead(dwg, p_end, end_dir, hs, col, e.head)
                if want_start:
                    _arrowhead(dwg, p_start, start_dir, hs, col, e.head)
                if e.label:
                    plates.append((e, pts, col))
        self.bus_boxes, self._bus_plates = {}, []
        for bid, bg in (res.get("buses") or {}).items():
            self._paint_bus(dwg, bid, bg, ox, oy)
        if plates:
            with dwg.group("edge labels"):
                taken = list(self._bus_plates)
                node_shapes = []                          # (bbox, outline polygon for pointy / round shapes)
                for v, n in self._nodes.items():
                    poly = None
                    if n.shape in ("diamond", "ellipse", "circle", "cloud", "hexagon", "parallelogram"):
                        poly = n.art.outline(n.x + ox, n.y + oy)
                    node_shapes.append((self.boxes[v], poly))
                group_lines = []                          # container borders and headers, avoided softly
                for gid, b in self.group_boxes.items():
                    x0, y0, x1, y1 = b
                    group_lines += [(x0, y0 - 1, x1, y0 + 1), (x0, y1 - 1, x1, y1 + 1),
                                    (x0 - 1, y0, x0 + 1, y1), (x1 - 1, y0, x1 + 1, y1)]
                    if self.groups[gid].label:
                        hw = _measure(str(self.groups[gid].label).upper(), 13, "bold", fam)[0] + 56
                        group_lines.append((x0, y0, x0 + hw, y0 + 32))
                for e, pts, col in plates:
                    efs = round(_DFS * 0.76, 1)
                    lw, lh = _measure(e.label, efs, "normal", fam)
                    pw, ph = lw + 14, lh + 8
                    total = _poly_len(pts)
                    if e.label_pos is not None:
                        prefs = [float(e.label_pos)]
                    elif getattr(e, "near_source", False):
                        horiz = abs(pts[1][1] - pts[0][1]) < abs(pts[1][0] - pts[0][0])
                        d0 = (pw if horiz else ph) / 2 + 12
                        prefs = [min(d0 / max(total, 1e-9), 0.5)]
                    else:
                        prefs = [0.5]
                    cands = []
                    for p0 in prefs:
                        for dt in (0, 0.08, -0.08, 0.16, -0.16, 0.24, -0.24, 0.32, -0.32, 0.4, -0.4):
                            t = p0 + dt
                            if 0.0 < t < 1.0:
                                cands.append(t)
                    cands += [t / 20 for t in range(1, 20)]     # then anywhere along the edge
                    chosen = soft = None
                    for t in cands:
                        (cx, cy), _ = _point_at(pts, total * t)
                        box = (cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2)
                        if any(_rect_hit(_inflate(box, 3), nb) and (poly is None or _rect_hits_poly(
                                _inflate(box, 3), poly)) for nb, poly in node_shapes):
                            continue
                        if any(_rect_hit(_inflate(box, 2), tb) for tb in taken):
                            continue
                        if any(_rect_hit(_inflate(box, 4), gb) for gb in group_lines):
                            soft = soft or (cx, cy, box)          # straddles a container border or header
                            continue
                        chosen = (cx, cy, box)
                        break
                    if chosen is None:                      # beside the line, never on a node
                        for t in cands:
                            (cx, cy), dv = _point_at(pts, total * t)
                            for sgn in (1, -1):
                                horiz = abs(dv[0]) >= abs(dv[1])
                                qx, qy = (cx, cy + sgn * (ph / 2 + 5)) if horiz else (cx + sgn * (pw / 2 + 5), cy)
                                box = (qx - pw / 2, qy - ph / 2, qx + pw / 2, qy + ph / 2)
                                if any(_rect_hit(_inflate(box, 3), nb) and (poly is None or _rect_hits_poly(
                                        _inflate(box, 3), poly)) for nb, poly in node_shapes):
                                    continue
                                if any(_rect_hit(_inflate(box, 2), tb) for tb in taken):
                                    continue
                                chosen = (qx, qy, box)
                                break
                            if chosen:
                                break
                    chosen = chosen or soft
                    if chosen is None:
                        (cx, cy), _ = _point_at(pts, total * prefs[0])
                        chosen = (cx, cy, (cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2))
                    cx, cy, box = chosen
                    taken.append(box)
                    dwg.rect(box[0], box[1], pw, ph, r=ph / 2, fill=th.background,
                             stroke=_mix(col, th.background, 0.35), stroke_width=1)
                    dwg.text(cx, cy, e.label, size=efs, fill=_mix(th.ink, th.muted, 0.35), anchor="mm",
                             family=fam)
        self.label_boxes = list(taken[len(self._bus_plates):]) if plates else []
        # nodes (one layer each, soft shadow); plain cards get an accent strip unless the diagram uses icons
        iconic = any(n.icon for n in self._nodes.values())
        for v, n in self._nodes.items():
            acc = self._colors(n)
            style = n.style or self.node_style
            if not n.style and style == "card" and n.shape in ("diamond", "pill", "circle"):
                style = "tint"                            # decisions and terminals read at a glance
            fill, stroke, tc, sc, chip, icol = _style_colors(th, style, acc)
            cx, cy = n.x + ox, n.y + oy
            with dwg.group(f"node {v}", shadow=_shadow(th, 0.8 if style != "outline" else 0.0)
                           if style != "outline" else None) as g:
                n.art.body(g, cx, cy, fill, stroke, 1.5 if style != "outline" else 2.0)
                if style == "card" and n.shape in ("box", "rounded") and not n.icon and not iconic:
                    W, H = n.art.W, n.art.H
                    clip = dwg.rect(cx - W / 2, cy - H / 2, W, H, r=min(self.theme.radius * 1.5, H / 2), add=False)
                    with g.group(clip=clip) as cg:
                        cg.rect(cx - W / 2, cy - H / 2, W, 4, fill=acc)
                n.art.content(g, cx, cy, acc, tc, sc, chip, icol)


    def _paint_bus(self, dwg, bid, bg, ox, oy):
        """Bus bar with a label plate at its start and a tap (dot at the junction) to every member."""
        th = self.theme
        b = self._buses[bid]
        col = _accent(th, b.color, 5) if b.color is not None else th.accent(5)
        fam = _fam(th)
        (x0, y0), (x1, y1) = [(x + ox, y + oy) for x, y in bg["bar"]]
        xs, ys = [x0, x1], [y0, y1]
        with dwg.group(f"node {bid}") as g0, g0.group("lines") as g:
            for m, pts in bg["taps"]:
                (ax, ay), (bx, by) = [(x + ox, y + oy) for x, y in pts]
                xs += [ax, bx]
                ys += [ay, by]
                arrow = b.taps.get(m, b.arrow) or "none"
                L = max(math.hypot(bx - ax, by - ay), 1e-9)
                d = ((bx - ax) / L, (by - ay) / L)
                hs = 9 + 2.0 * 2.2
                tr = _arrow_trim("arrow", hs)
                p0 = (ax + d[0] * tr, ay + d[1] * tr) if arrow in ("start", "both") else (ax, ay)
                p1 = (bx - d[0] * tr, by - d[1] * tr) if arrow in ("end", "both") else (bx, by)
                g.path(Path().move_to(*p0).line_to(*p1), stroke=col, stroke_width=2.0, fill=None,
                       dash=_dash(b.style, 2.0), cap="butt")
                if arrow in ("end", "both"):
                    _arrowhead(g, (bx, by), d, hs, col, "arrow")
                if arrow in ("start", "both"):
                    _arrowhead(g, (ax, ay), (-d[0], -d[1]), hs, col, "arrow")
                g.circle(bx, by, 5.0, fill=col)
            g.path(Path().move_to(x0, y0).line_to(x1, y1), stroke=col, stroke_width=5.0, fill=None,
                   dash=_dash(b.style, 5.0), cap="round")
        with g0:
            g = g0
            lx, ly = bg["label"]
            lx, ly = lx + ox, ly + oy
            pw, ph = bg["plate"]
            g.rect(lx - pw / 2, ly - ph / 2, pw, ph, r=ph / 2, fill=_mix(col, th.background, 0.22 if th.dark else 0.14),
                   stroke=col, stroke_width=1.5)
            g.text(lx, ly, str(b.label), size=round(_DFS * 0.76, 1), weight="bold", fill=th.ink, anchor="mm",
                   family=fam)
            xs += [lx - pw / 2, lx + pw / 2]
            ys += [ly - ph / 2, ly + ph / 2]
            self._bus_plates.append((lx - pw / 2, ly - ph / 2, lx + pw / 2, ly + ph / 2))
        self.bus_boxes[bid] = (min(xs), min(ys), max(xs), max(ys))


def _keys_path(keys, LR, trim_start=0.0, trim_end=0.0):
    """Curved / straight edge from key points [(x, y, kind)] → (Path, start_dir, end_dir, polyline).

    kind = how to reach the NEXT point: "S" (S-curve with main-axis tangents), "L" (line), "Q" (smooth corner,
    the next point is the control corner)."""
    pts = [(k[0], k[1]) for k in keys]
    kinds = [k[2] for k in keys]

    def unit(a, b):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        return ((b[0] - a[0]) / L, (b[1] - a[1]) / L) if L > 1e-9 else (1.0, 0.0)

    def tangent(i, at_end):
        # direction of travel at the start (i = 0) or at the end of the last segment
        if at_end:
            j = len(pts) - 2
            if j >= 1 and kinds[j - 1] == "Q":
                return unit(pts[j], pts[j + 1])
            a, b = pts[j], pts[j + 1]
        else:
            a, b = pts[0], pts[1]
            if kinds[0] == "Q":
                return unit(pts[0], pts[1])
        if (kinds[len(pts) - 2] if at_end else kinds[0]) == "S":
            if LR:
                return (1.0 if b[0] >= a[0] else -1.0, 0.0)
            return (0.0, 1.0 if b[1] >= a[1] else -1.0)
        return unit(a, b)

    d0, d1 = tangent(0, False), tangent(0, True)
    if trim_start:
        pts[0] = (pts[0][0] + d0[0] * trim_start, pts[0][1] + d0[1] * trim_start)
    if trim_end:
        pts[-1] = (pts[-1][0] - d1[0] * trim_end, pts[-1][1] - d1[1] * trim_end)
    p = Path().move_to(*pts[0])
    poly = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        a, b = pts[i], pts[i + 1]
        k = kinds[i]
        if k == "Q" and i + 2 < len(pts):
            c = pts[i + 2]
            c1 = (a[0] + (b[0] - a[0]) * 0.7, a[1] + (b[1] - a[1]) * 0.7)
            c2 = (c[0] + (b[0] - c[0]) * 0.7, c[1] + (b[1] - c[1]) * 0.7)
            p.cubic_to(c1[0], c1[1], c2[0], c2[1], c[0], c[1])
            poly += [b, c]
            i += 2
            continue
        if k == "S" and (abs(a[1] - b[1]) > 0.5 if LR else abs(a[0] - b[0]) > 0.5):
            if LR:
                dm = (b[0] - a[0]) * 0.5
                p.cubic_to(a[0] + dm, a[1], b[0] - dm, b[1], b[0], b[1])
            else:
                dm = (b[1] - a[1]) * 0.5
                p.cubic_to(a[0], a[1] + dm, b[0], b[1] - dm, b[0], b[1])
            for t in (0.25, 0.5, 0.75):
                poly.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * (3 * t * t - 2 * t ** 3)) if LR
                            else (a[0] + (b[0] - a[0]) * (3 * t * t - 2 * t ** 3), a[1] + (b[1] - a[1]) * t))
        else:
            p.line_to(*b)
        poly.append(b)
        i += 1
    return p, (-d0[0], -d0[1]), d1, poly


def _route_fold(na, nb, boxes, ida, idb, LR, mid):
    """Edge between two folded bands: leave the face toward the other band, cross the gap at ``mid``
    (straight when the two nodes overlap), enter the facing side. None when another node is in the way."""
    ba, bb = boxes[ida], boxes[idb]
    if LR:                                      # bands stacked top to bottom: work in (x, y)
        A, B, cA, cB, flip = ba, bb, na.x, nb.x, False
    else:                                       # bands side by side: swap axes
        A, B, cA, cB, flip = (ba[1], ba[0], ba[3], ba[2]), (bb[1], bb[0], bb[3], bb[2]), na.y, nb.y, True
    down = (B[1] + B[3]) > (A[1] + A[3])
    ya, yb = (A[3], B[1]) if down else (A[1], B[3])
    lo, hi = max(A[0], B[0]) + 14, min(A[2], B[2]) - 14
    if hi - lo > 4:
        x = min(max((cA + cB) / 2, lo), hi)
        pts = [(x, ya), (x, yb)]
    else:
        pts = [(cA, ya), (cA, mid), (cB, mid), (cB, yb)]
    if flip:
        pts = [(y, x) for x, y in pts]
    obst = [_inflate(b, 4) for k, b in boxes.items() if k not in (ida, idb)]
    for p, q in zip(pts, pts[1:]):
        if any(_seg_hits_box(p, q, o) for o in obst):
            return None
    return pts


def _route_avoid(boxes, ida, idb, used=(), pad=9.0, bend=36.0):
    """Orthogonal route between two nodes that stays clear of every other node: Dijkstra over the grid spanned
    by the node boxes (lines ``pad`` outside each), with a cost per bend and for running along or entering
    where the ``used`` segments already are. None when no route exists."""
    import heapq
    A, B = boxes[ida], boxes[idb]
    obst = [_inflate(b, 3) for k, b in boxes.items() if k not in (ida, idb)] + [_inflate(A, -1), _inflate(B, -1)]
    stub = 12.0
    ports = []
    for box in (A, B):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        ports.append([((box[2], cy), (box[2] + stub, cy)), ((box[0], cy), (box[0] - stub, cy)),
                      ((cx, box[3]), (cx, box[3] + stub)), ((cx, box[1]), (cx, box[1] - stub))])
    xs, ys = set(), set()
    for b in boxes.values():
        xs.update((b[0] - pad, b[2] + pad))
        ys.update((b[1] - pad, b[3] + pad))
    for side in ports:
        for _, q in side:
            xs.add(q[0])
            ys.add(q[1])
    xs, ys = sorted(xs), sorted(ys)
    xi = {round(v, 3): i for i, v in enumerate(xs)}
    yi = {round(v, 3): i for i, v in enumerate(ys)}
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    free_cache = {}
    used = list(used)

    def shared(p, q):
        """Length of the move p->q that runs along an existing segment (within 7 px)."""
        if p[1] == q[1]:
            lo, hi = sorted((p[0], q[0]))
            return sum(max(0.0, min(hi, max(a[0], b[0])) - max(lo, min(a[0], b[0]))) for a, b in used
                       if abs(a[1] - b[1]) < 0.5 and abs(a[1] - p[1]) < 7)
        lo, hi = sorted((p[1], q[1]))
        return sum(max(0.0, min(hi, max(a[1], b[1])) - max(lo, min(a[1], b[1]))) for a, b in used
                   if abs(a[0] - b[0]) < 0.5 and abs(a[0] - p[0]) < 7)

    def busy(pt):
        return any(_seg_dist(pt, a, b) < 5 for a, b in used)

    def free(i0, j0, i1, j1):
        key = (i0, j0, i1, j1)
        if key not in free_cache:
            p, q = (xs[i0], ys[j0]), (xs[i1], ys[j1])
            free_cache[key] = not any(_seg_hits_box(p, q, o) for o in obst)
        return free_cache[key]

    best = None
    for pa, qa in ports[0]:
        for pb, qb in ports[1]:
            if qa == qb or any(_in_box(qa, o) or _in_box(qb, o) for o in obst):
                continue
            si, sj = xi[round(qa[0], 3)], yi[round(qa[1], 3)]
            ti, tj = xi[round(qb[0], 3)], yi[round(qb[1], 3)]
            d0 = (round(qa[0] - pa[0]), round(qa[1] - pa[1]))
            d0 = (int(math.copysign(1, d0[0])) if d0[0] else 0, int(math.copysign(1, d0[1])) if d0[1] else 0)
            dq = {(si, sj, d0): 0.0}
            prev = {}
            heap = [(0.0, si, sj, d0)]
            goal = None
            while heap:
                c, i, j, dd = heapq.heappop(heap)
                if c > dq.get((i, j, dd), 1e18) + 1e-9:
                    continue
                if (i, j) == (ti, tj):
                    goal = (i, j, dd)
                    break
                for nd in dirs:
                    if nd[0] == -dd[0] and nd[1] == -dd[1]:
                        continue
                    ni, nj = i + nd[0], j + nd[1]
                    if not (0 <= ni < len(xs) and 0 <= nj < len(ys)) or not free(i, j, ni, nj):
                        continue
                    cost = c + abs(xs[ni] - xs[i]) + abs(ys[nj] - ys[j]) + (bend if nd != dd else 0.0)
                    if used:
                        cost += 2.0 * shared((xs[i], ys[j]), (xs[ni], ys[nj]))
                    if cost < dq.get((ni, nj, nd), 1e18) - 1e-9:
                        dq[(ni, nj, nd)] = cost
                        prev[(ni, nj, nd)] = (i, j, dd)
                        heapq.heappush(heap, (cost, ni, nj, nd))
            if goal is None:
                continue
            total = dq[goal] + 2 * stub + (250.0 if used and (busy(pa) or busy(pb)) else 0.0)
            total += 0.0 * stub + (bend if (goal[2][0] and (pb[0] == qb[0])) or
                                           (goal[2][1] and (pb[1] == qb[1])) else 0.0)
            # the last move must run toward the target port
            if best is None or total < best[0]:
                seq, cur = [], goal
                while cur in prev:
                    seq.append((xs[cur[0]], ys[cur[1]]))
                    cur = prev[cur]
                seq.append((xs[cur[0]], ys[cur[1]]))
                pts = [pa] + seq[::-1] + [pb]
                best = (total, pts)
    if best is None:
        return None
    out = []
    for p in best[1]:
        if out and abs(out[-1][0] - p[0]) < 0.01 and abs(out[-1][1] - p[1]) < 0.01:
            continue
        while len(out) >= 2 and ((abs(out[-1][0] - out[-2][0]) < 0.01 and abs(out[-1][0] - p[0]) < 0.01) or
                                 (abs(out[-1][1] - out[-2][1]) < 0.01 and abs(out[-1][1] - p[1]) < 0.01)):
            out.pop()
        out.append(p)
    return out


def _seg_dist(p, a, b):
    """Distance from point p to the axis-aligned segment a-b."""
    x = min(max(p[0], min(a[0], b[0])), max(a[0], b[0]))
    y = min(max(p[1], min(a[1], b[1])), max(a[1], b[1]))
    return math.hypot(p[0] - x, p[1] - y)


def _in_box(p, b):
    return b[0] < p[0] < b[2] and b[1] < p[1] < b[3]


def _route_manual(na, nb, boxes, ida, idb, routing):
    """Orthogonal route between two manually placed nodes avoiding other node boxes."""
    ba, bb = boxes[ida], boxes[idb]
    ax, ay = na.x, na.y
    bx, by = nb.x, nb.y
    obst = [_inflate(b, 6) for k, b in boxes.items() if k not in (ida, idb)]

    def side_pt(n, b, s, toward):
        x0, y0, x1, y1 = b
        if s == "r":
            return (x1, n.y)
        if s == "l":
            return (x0, n.y)
        if s == "b":
            return (n.x, y1)
        return (n.x, y0)

    def clear(pts):
        for p, q in zip(pts, pts[1:]):
            for o in obst:
                if _seg_hits_box(p, q, o):
                    return False
        return True

    cands = []
    out = {"r": (1, 0), "l": (-1, 0), "b": (0, 1), "t": (0, -1)}
    for sa in "rlbt":
        for sb in "rlbt":
            p0 = side_pt(na, ba, sa, nb)
            p3 = side_pt(nb, bb, sb, na)
            da, db = out[sa], out[sb]
            # must leave away from own node and enter from outside
            if (p3[0] - p0[0]) * da[0] + (p3[1] - p0[1]) * da[1] < -1:
                continue
            if (p0[0] - p3[0]) * db[0] + (p0[1] - p3[1]) * db[1] < -1:
                continue
            q0 = (p0[0] + da[0] * 18, p0[1] + da[1] * 18)
            q3 = (p3[0] + db[0] * 18, p3[1] + db[1] * 18)
            shapes = []
            if da[0] and db[0]:
                for f in (0.5, 0.3, 0.7):
                    mx = q0[0] + (q3[0] - q0[0]) * f
                    shapes.append([p0, (mx, p0[1]), (mx, p3[1]), p3])
            elif da[1] and db[1]:
                for f in (0.5, 0.3, 0.7):
                    my = q0[1] + (q3[1] - q0[1]) * f
                    shapes.append([p0, (p0[0], my), (p3[0], my), p3])
            else:
                if da[0]:
                    shapes.append([p0, (p3[0], p0[1]), p3])
                else:
                    shapes.append([p0, (p0[0], p3[1]), p3])
            for pts in shapes:
                if abs(pts[0][0] - pts[-1][0]) < 1 and len(pts) > 2 and da[1]:
                    pts = [pts[0], pts[-1]]
                if abs(pts[0][1] - pts[-1][1]) < 1 and len(pts) > 2 and da[0]:
                    pts = [pts[0], pts[-1]]
                L = _poly_len(pts)
                bends = len(pts) - 2
                cost = L + bends * 40 + (0 if clear(pts) else 1e6)
                cands.append((cost, pts))
    if not cands:
        return [(ax, ay), (bx, by)]
    cands.sort(key=lambda c: c[0])
    pts = cands[0][1]
    if routing == "straight":
        return [pts[0], pts[-1]]
    return pts


def _place(items, layers, links, gap, iters=40):
    """1-D placement with ordering constraints per layer and attraction along links.

    ``items``: {id: dict(ranks, lo, hi)}; ``layers``: per rank list of item ids (order); ``links``:
    (i, off_i, j, off_j, w) wanting pos_i + off_i == pos_j + off_j; ``gap(a, b)`` min gap. Returns {id: pos}."""
    cons = {}
    for L in layers:
        for a, b in zip(L, L[1:]):
            d = items[a]["hi"] - items[b]["lo"] + gap(a, b)
            cons[(a, b)] = max(cons.get((a, b), -1e18), d)
    preds = {k: [] for k in items}
    succs = {k: [] for k in items}
    for (a, b), d in cons.items():
        if a == b:
            continue
        preds[b].append((a, d))
        succs[a].append((b, d))
    # topological order (Kahn); cycles broken by insertion order
    deg = {k: len(preds[k]) for k in items}
    order, queue = [], [k for k in items if deg[k] == 0]
    seen = set()
    while len(order) < len(items):
        if not queue:
            rest = [k for k in items if k not in seen]
            queue = [rest[0]]
            deg[rest[0]] = 0
        v = queue.pop(0)
        if v in seen:
            continue
        seen.add(v)
        order.append(v)
        for w, _ in succs[v]:
            deg[w] -= 1
            if deg[w] == 0 and w not in seen:
                queue.append(w)
    rank_of = {k: i for i, k in enumerate(order)}
    # drop constraints that point backwards (only possible with inconsistent input orders)
    preds = {k: [(a, d) for a, d in preds[k] if rank_of[a] < rank_of[k]] for k in items}
    succs = {k: [(b, d) for b, d in succs[k] if rank_of[b] > rank_of[k]] for k in items}
    pos = {}
    for v in order:
        pos[v] = max([pos[a] + d for a, d in preds[v] if a in pos] or [0.0])
    nb = {k: [] for k in items}
    for i, oi, j, oj, w in links:
        if i == j or i not in nb or j not in nb:
            continue
        nb[i].append((j, oj - oi, w))
        nb[j].append((i, oi - oj, w))

    def desired(v):
        lst = nb[v]
        if not lst:
            return None
        vals = sorted((pos[j] + off, w) for j, off, w in lst)
        tot = sum(w for _, w in vals)
        acc = 0.0
        # weighted median, averaged with the mean for balance
        med = vals[-1][0]
        for x, w in vals:
            acc += w
            if acc >= tot / 2 - 1e-9:
                med = x
                break
        if len(vals) == 2:
            med = (vals[0][0] * vals[0][1] + vals[1][0] * vals[1][1]) / tot
        return med

    for it in range(iters):
        seq = order if it % 2 == 0 else order[::-1]
        moved = 0.0
        for v in seq:
            d = desired(v)
            if d is None:
                continue
            lo = max([pos[a] + dd for a, dd in preds[v] if rank_of[a] != rank_of[v]] or [-1e18])
            hi = min([pos[b] - dd for b, dd in succs[v]] or [1e18])
            if lo > hi:
                nv = lo
            else:
                nv = min(max(d, lo), hi)
            moved += abs(nv - pos[v])
            pos[v] = nv
        if moved < 0.05:
            break
    # unlinked items: centre between neighbours where possible
    for v in order:
        if nb[v]:
            continue
        lo = max([pos[a] + dd for a, dd in preds[v]] or [-1e18])
        hi = min([pos[b] - dd for b, dd in succs[v]] or [1e18])
        if lo > -1e17 and hi < 1e17:
            pos[v] = (lo + hi) / 2 if lo <= hi else lo
        elif lo > -1e17:
            pos[v] = max(pos[v], lo)
    return pos


# --------------------------------------------------------------------------------------------- Tree

def _split_label(s):
    s = str(s)
    if "|" in s:
        a, b = s.split("|", 1)
        return a.strip(), b.strip() or None
    return s, None


class _TN:
    def __init__(self, key, depth):
        self.label, self.sub = _split_label(key)
        self.key, self.depth, self.children = key, depth, []


def _tree_from(data, depth=0):
    """Nested dict / list / str → list of _TN roots."""
    out = []
    if data is None:
        return out
    if isinstance(data, dict):
        for k, v in data.items():
            n = _TN(k, depth)
            n.children = _tree_from(v, depth + 1)
            out.append(n)
    elif isinstance(data, (list, tuple)):
        for item in data:
            out.extend(_tree_from(item, depth))
    else:
        out.append(_TN(data, depth))
    return out


def _initials(name):
    parts = [p for p in str(name).replace("-", " ").split() if p]
    if not parts:
        return "?"
    if any(_is_cjk(ch) for ch in parts[0]):
        return parts[0][:1]
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper()


class Tree(_Figure):
    """Org chart, hierarchy or mind map from a nested dict.

    ``Tree({"Ada | CEO": {"Grace | CTO": ["Web", "Data"], "Alan | CFO": None}}, kind="org")``

    * ``kind``: "org" (cards with initials avatars, elbow connectors), "tree" (rounded nodes, curved links),
      "mindmap" (centre topic, branches left and right, tapered curves).
    * ``direction``: TB or LR (org / tree). ``compact``: stack leaf-only teams in a column (org, TB).
    * Labels "Name | Role" become a title and a sublabel. ``icon``: one icon for every avatar / node.
    """

    def __init__(self, data, kind="org", theme="light", direction="TB", title=None, subtitle=None,
                 compact=True, icon=None, avatars=None, font_size=18, spacing=1.0, margin=40, colors=None):
        super().__init__(theme, title, subtitle, margin)
        if kind not in ("org", "tree", "mindmap"):
            raise _unknown_name("kind", kind, ["org", "tree", "mindmap"])
        roots = _tree_from(data)
        if len(roots) == 1:
            self.root = roots[0]
        else:
            self.root = _TN("", 0)
            self.root.children = roots
            self.root.hidden = True

            def bump(n):
                for c in n.children:
                    c.depth = n.depth + 1
                    bump(c)
            bump(self.root)
        self.kind, self.direction = kind, direction.upper()
        self.compact, self.icon, self.fs, self.spacing = compact, icon, font_size, spacing
        self.avatars = (kind == "org") if avatars is None else avatars
        self.colors = colors
        self._assign_branches()

    def _assign_branches(self):
        th = self.theme
        for i, c in enumerate(self.root.children):
            col = _accent(th, self.colors[i % len(self.colors)]) if self.colors else th.accent(i)

            def setc(n, col=col, b=i):
                n.color, n.branch = col, b
                for ch in n.children:
                    setc(ch)
            setc(c)
        self.root.color, self.root.branch = th.accent(0), -1

    def _all(self, n=None):
        n = n or self.root
        yield n
        for c in n.children:
            yield from self._all(c)

    # ---- org / tree layout (tidy, parent centred over children)
    def _measure_org(self, n):
        th = self.theme
        if getattr(n, "hidden", False):
            n.W = n.H = 0.0
            return
        if self.kind == "org" and self.avatars:
            fam = _fam(th)
            fs = self.fs
            lw, lh = _measure(n.label, fs, "bold", fam)
            sw, sh = _measure(n.sub, fs * 0.76, "normal", fam) if n.sub else (0, 0)
            av = fs * 2.4
            n.av = av
            n.W = max(fs * 10, fs * 1.0 + av + fs * 0.7 + max(lw, sw) + fs * 1.1)
            n.H = max(av + fs * 1.1, lh + (sh + 4 if n.sub else 0) + fs * 1.3)
            n.lw, n.lh, n.sw, n.sh = lw, lh, sw, sh
        else:
            n.art = _NodeArt(th, n.label, n.sub, self.icon, "rounded", self.fs, 200)
            n.W, n.H = n.art.W, n.art.H

    def _layout_org(self):
        LR = self.direction in ("LR", "RL")
        sp = self.spacing
        for n in self._all():
            self._measure_org(n)
        bydepth = {}
        for n in self._all():
            if not getattr(n, "hidden", False):
                bydepth.setdefault(n.depth, []).append(n)
        for lst in bydepth.values():                      # uniform card size per depth
            W = max(n.W for n in lst)
            H = max(n.H for n in lst)
            for n in lst:
                n.W, n.H = W, H
                if hasattr(n, "art"):
                    n.art.W, n.art.H = W, H
        for n in self._all():
            n.sc, n.sm = (n.H, n.W) if LR else (n.W, n.H)
        gap_c = (28 if not LR else 18) * sp
        gap_m = (56 if not LR else 70) * sp
        self._indent = indent = 26 * sp

        def stacked(n):
            return (self.compact and self.kind == "org" and not LR and n.depth >= 1 and len(n.children) >= 2
                    and all(not c.children for c in n.children))

        def width(n):                                     # cross extent of the subtree
            n.stack = stacked(n)
            if n.stack:
                n.span = max(n.sc, indent + max(c.sc for c in n.children))
                return n.span
            if not n.children:
                n.span = n.sc
                return n.span
            tot = sum(width(c) for c in n.children) + gap_c * (len(n.children) - 1)
            n.span = max(n.sc, tot)
            return n.span

        width(self.root)
        hidden = getattr(self.root, "hidden", False)
        mrow, m = {}, 0.0
        for d in range(0, max(bydepth) + 1 if bydepth else 1):
            if d == 0 and hidden:
                mrow[0] = -gap_m
                continue
            sm = max([n.sm for n in bydepth.get(d, [])] or [0])
            mrow[d] = m + sm / 2
            m += sm + gap_m

        def place(n, c0):
            n.m = mrow[n.depth]
            if n.stack:
                n.c = c0 + n.sc / 2
                x = c0 + indent
                mm = n.m + n.sm / 2 + gap_m * 0.45
                for c in n.children:
                    c.c, c.m, c.stack = x + c.sc / 2, mm + c.sm / 2, False
                    mm += c.sm + 14 * sp
                return
            n.c = c0 + n.span / 2
            if n.children:
                tot = sum(c.span for c in n.children) + gap_c * (len(n.children) - 1)
                x = c0 + (n.span - tot) / 2
                for c in n.children:
                    place(c, x)
                    x += c.span + gap_c
                n.c = (n.children[0].c + n.children[-1].c) / 2

        place(self.root, 0.0)
        nodes = [n for n in self._all() if not getattr(n, "hidden", False)]
        c0 = min(n.c - n.sc / 2 for n in nodes)
        m0 = min(n.m - n.sm / 2 for n in nodes)
        for n in self._all():
            n.c -= c0
            n.m -= m0
            n.x, n.y = (n.m, n.c) if LR else (n.c, n.m)
        W = max(n.x + n.W / 2 for n in nodes)
        H = max(n.y + n.H / 2 for n in nodes)
        return W, H

    # ---- mind map layout
    def _layout_mind(self):
        th = self.theme
        fam = _fam(th)
        sp = self.spacing
        fs = self.fs
        r = self.root
        rl, rh = _measure(r.label, fs * 1.35, "bold", fam)
        r.W, r.H = rl + fs * 2.6, rh + fs * 1.6
        for n in self._all():
            if n is r:
                continue
            if n.depth == 1:
                tw, t_h = _measure(n.label, fs, "bold", fam)
                n.W, n.H = tw + fs * 1.6, t_h + fs * 1.0
            else:
                tw, t_h = _measure(n.label, fs * 0.86, "normal", fam)
                n.W, n.H = tw + 10, t_h + fs * 0.6
            n.tw = tw
        vgap = 10 * sp
        hgap = 60 * sp

        def height(n):
            own = n.H + vgap * (1.6 if n.depth == 1 else 1)
            n.span = own if not n.children else max(own, sum(height(c) for c in n.children))
            return n.span

        kids = r.children
        for k in kids:
            height(k)
        total = sum(k.span for k in kids)
        right, acc = [], 0.0
        for k in kids:
            if not right or acc + k.span / 2 <= total / 2:
                right.append(k)
                acc += k.span
            else:
                break
        left = kids[len(right):]
        if not left and len(right) > 1:
            left = [right.pop()]
        left = left[::-1]                                  # clockwise reading order

        def place(n, x_edge, y0, side):
            n.side = side
            n.y = y0 + n.span / 2
            n.x = x_edge + side * n.W / 2
            if n.children:
                tot = sum(c.span for c in n.children)
                y = n.y - tot / 2
                for c in n.children:
                    place(c, x_edge + side * (n.W + hgap * 0.8), y, side)
                    y += c.span

        for side, lst in ((1, right), (-1, left)):
            tot = sum(k.span for k in lst)
            y = -tot / 2
            for k in lst:
                place(k, side * (r.W / 2 + hgap * 1.3), y, side)
                y += k.span
        r.x, r.y = 0.0, 0.0
        nodes = list(self._all())
        x0 = min(n.x - n.W / 2 for n in nodes)
        y0 = min(n.y - n.H / 2 for n in nodes)
        for n in nodes:
            n.x -= x0
            n.y -= y0
        W = max(n.x + n.W / 2 for n in nodes)
        H = max(n.y + n.H / 2 for n in nodes)
        return W, H

    def _size(self):
        if self.kind == "mindmap":
            return self._layout_mind()
        return self._layout_org()

    def _paint(self, dwg, ox, oy):
        if self.kind == "mindmap":
            return self._paint_mind(dwg, ox, oy)
        th = self.theme
        fam = _fam(th)
        LR = self.direction in ("LR", "RL")
        ecol = _edge_color(th)
        nodes = [n for n in self._all() if not getattr(n, "hidden", False)]
        self.boxes = {n.key: (n.x + ox - n.W / 2, n.y + oy - n.H / 2, n.x + ox + n.W / 2, n.y + oy + n.H / 2)
                      for n in nodes}
        with dwg.group("links"):
            for n in self._all():
                if getattr(n, "hidden", False) or not n.children:
                    continue
                for c in n.children:
                    if n.stack:
                        sx = n.x + ox - n.W / 2 + self._indent * 0.5
                        pts = [(sx, n.y + oy + n.H / 2), (sx, c.y + oy), (c.x + ox - c.W / 2, c.y + oy)]
                    elif LR:
                        mx = (n.x + n.W / 2 + c.x - c.W / 2) / 2 + ox
                        pts = [(n.x + ox + n.W / 2, n.y + oy), (mx, n.y + oy), (mx, c.y + oy),
                               (c.x + ox - c.W / 2, c.y + oy)]
                    else:
                        my = (n.y + n.H / 2 + c.y - c.H / 2) / 2 + oy
                        pts = [(n.x + ox, n.y + oy + n.H / 2), (n.x + ox, my), (c.x + ox, my),
                               (c.x + ox, c.y + oy - c.H / 2)]
                    if self.kind == "tree" and not n.stack:
                        p0, p3 = pts[0], pts[-1]
                        if LR:
                            mx = (p0[0] + p3[0]) / 2
                            path = Path().move_to(*p0).cubic_to(mx, p0[1], mx, p3[1], *p3)
                        else:
                            my = (p0[1] + p3[1]) / 2
                            path = Path().move_to(*p0).cubic_to(p0[0], my, p3[0], my, *p3)
                    else:
                        path = _round_poly(pts, 10, closed=False)
                    dwg.path(path, stroke=ecol, stroke_width=1.8, fill=None, join="round")
        for n in nodes:
            acc = n.color
            is_root = n.depth == 0
            with dwg.group(f"node {n.label}", shadow=_shadow(th, 0.8)) as g:
                cx, cy = n.x + ox, n.y + oy
                if self.kind == "org" and self.avatars:
                    self._org_card(dwg, g, n, cx, cy, acc, is_root, fam)
                else:
                    style = "solid" if is_root else ("tint" if n.depth == 1 else "card")
                    fill, stroke, tc, sc, chip, icol = _style_colors(th, style, acc)
                    n.art.body(g, cx, cy, fill, stroke, 1.5)
                    n.art.content(g, cx, cy, acc, tc, sc, chip, icol)

    def _org_card(self, dwg, g, n, cx, cy, acc, is_root, fam):
        th = self.theme
        R = th.radius * 1.5
        fill = acc if is_root else th.surface
        stroke = _mix("#000000", acc, 0.1) if is_root else _mix(acc, th.surface, 0.36)
        g.rect(cx - n.W / 2, cy - n.H / 2, n.W, n.H, r=R, fill=fill, stroke=stroke, stroke_width=1.5)
        if not is_root:
            clip = dwg.rect(cx - n.W / 2, cy - n.H / 2, n.W, n.H, r=R, add=False)
            with g.group(clip=clip) as cg:
                cg.rect(cx - n.W / 2, cy - n.H / 2, 5, n.H, fill=acc)
        av = n.av
        ax = cx - n.W / 2 + self.fs * 1.0 + av / 2
        if is_root:
            g.circle(ax, cy, av / 2, fill=_mix("#ffffff", acc, 0.22))
            ink = _on(acc, th)
        else:
            g.circle(ax, cy, av / 2, fill=_mix(acc, th.surface, 0.18 if not th.dark else 0.26))
            ink = th.ink
        if self.icon:
            g.icon(self.icon, ax, cy, av * 0.55, color=ink if is_root else acc)
        else:
            g.text(ax, cy, _initials(n.label), size=self.fs * 0.82, weight="bold", fill=ink if is_root else acc,
                   anchor="mm", family=fam)
        tx = ax + av / 2 + self.fs * 0.7
        tt = cy - (n.lh + (n.sh + 4 if n.sub else 0)) / 2
        g.text(tx, tt, n.label, size=self.fs, weight="bold", fill=ink, anchor="lt", family=fam)
        if n.sub:
            g.text(tx, tt + n.lh + 4, n.sub, size=self.fs * 0.76,
                   fill=_mix(ink, acc, 0.75) if is_root else th.muted, anchor="lt", family=fam)

    def _paint_mind(self, dwg, ox, oy):
        th = self.theme
        fam = _fam(th)
        fs = self.fs
        r = self.root
        widths = {1: 4.4, 2: 2.6, 3: 1.8}
        self.boxes = {n.key: (n.x + ox - n.W / 2, n.y + oy - n.H / 2, n.x + ox + n.W / 2, n.y + oy + n.H / 2)
                      for n in self._all()}

        def anchor_out(n):
            if n.depth == 1:
                return (n.x + ox + n.side * n.W / 2, n.y + oy)
            return (n.x + ox + n.side * n.W / 2, n.y + oy + n.H / 2 - 2)

        def anchor_in(n):
            if n.depth == 1:
                return (n.x + ox - n.side * n.W / 2, n.y + oy)
            return (n.x + ox - n.side * n.W / 2, n.y + oy + n.H / 2 - 2)

        with dwg.group("branches"):
            for n in self._all():
                for c in n.children:
                    col = c.color
                    sw = widths.get(c.depth, 1.6)
                    p0 = (r.x + ox + c.side * r.W * 0.32, r.y + oy) if n is r else anchor_out(n)
                    p3 = anchor_in(c)
                    dx = (p3[0] - p0[0]) * 0.55
                    path = Path().move_to(*p0).cubic_to(p0[0] + dx, p0[1], p3[0] - dx, p3[1], *p3)
                    dwg.path(path, stroke=col, stroke_width=sw, fill=None, cap="round")
                    if c.depth >= 2:
                        x_in, y_in = anchor_in(c)
                        dwg.line(x_in, y_in, x_in + c.side * c.W, y_in, stroke=col, stroke_width=sw, cap="round")
        with dwg.group("topics"):
            for n in self._all():
                if n.depth != 1:
                    continue
                cx, cy = n.x + ox, n.y + oy
                with dwg.group(f"topic {n.label}", shadow=_shadow(th, 0.6)) as g:
                    g.rect(cx - n.W / 2, cy - n.H / 2, n.W, n.H, r=n.H / 2,
                           fill=_mix(n.color, th.surface, 0.2 if th.dark else 0.14), stroke=n.color, stroke_width=2)
                    g.text(cx, cy, n.label, size=fs, weight="bold", anchor="mm", family=fam, fill=th.ink)
        with dwg.group("leaves"):
            for n in self._all():
                if n.depth < 2:
                    continue
                x_in = n.x + ox - n.side * n.W / 2
                dwg.text(x_in + n.side * 5, n.y + oy + n.H / 2 - 7, n.label, size=fs * 0.86,
                         anchor="lb" if n.side > 0 else "rb", family=fam, fill=th.ink)
        with dwg.group("centre", shadow=_shadow(th, 1.2)) as g:
            cx, cy = r.x + ox, r.y + oy
            col = th.accent(0)
            g.rect(cx - r.W / 2, cy - r.H / 2, r.W, r.H, r=r.H / 2, fill=col)
            g.text(cx, cy, r.label, size=fs * 1.35, weight="bold", anchor="mm", family=fam, fill=_on(col, th))


# --------------------------------------------------------------------------------------------- Sequence

class _FrameHandle:
    def __init__(self, seq, ev):
        self.seq, self.ev = seq, ev

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.seq._events.append(_N(kind="frame_end", start=self.ev))
        return False


class Sequence(_Figure):
    """Sequence diagram: participants across the top, messages down the page.

    ``s = Sequence(theme="light")``; ``s.participant("app", "App", icon="phone")``;
    ``s.message("app", "api", "POST /login", activate=True)``; ``s.reply("api", "app", "200 OK")``;
    ``s.note("api", "checks the token")``; ``with s.frame("loop", "every 30 s"): ...``; ``s.divider("later")``.

    * ``autonumber``: numbered badges on every message.
    """

    def __init__(self, theme="light", title=None, subtitle=None, autonumber=False, font_size=17, spacing=1.0,
                 margin=40, mirror=False):
        super().__init__(theme, title, subtitle, margin)
        self.parts, self._events = {}, []
        self.autonumber, self.fs, self.spacing, self.mirror = autonumber, font_size, spacing, mirror
        self._kx = self._ky = 1.0
        self._xmax, self._ymax = 2.2, 2.0

    def _stretch(self, aspect, hw, hh):
        """Adapt spacing so the figure approaches ``aspect`` (width / height)."""
        self._kx = self._ky = 1.0
        if not aspect:
            return
        m = self.margin
        for _ in range(4):
            cw, ch = self._size()
            W, H = max(cw, hw) + 2 * m, ch + hh + 2 * m
            r = (W / H) / aspect
            if abs(r - 1) < 0.03:
                break
            if r < 1:
                nk = min(self._xmax, self._kx * (1 + (1 / r - 1) * 1.3))
                if nk <= self._kx + 1e-3:
                    break
                self._kx = nk
            else:
                nk = min(self._ymax, self._ky * (1 + (r - 1) * 1.3))
                if nk <= self._ky + 1e-3:
                    break
                self._ky = nk

    def participant(self, id, label=None, icon=None, color=None, kind="box"):
        """Add a participant (column). ``kind``: box / actor (person badge) / database."""
        if icon is not None and icon not in _v.ICONS:
            raise _v._unknown("icon", icon, _v.ICONS, "Try vector.icon_names()")
        if kind == "actor" and icon is None:
            icon = "user"
        if kind == "database" and icon is None:
            icon = "database"
        self.parts[id] = _N(id=id, label=str(id) if label is None else label, icon=icon, color=color, kind=kind,
                            idx=len(self.parts))
        return id

    def _need(self, *ids):
        for i in ids:
            if i not in self.parts:
                self.participant(i)

    def message(self, a, b, label=None, reply=False, style=None, activate=False, deactivate=False, head=None,
                color=None):
        """Arrow from ``a`` to ``b``. ``reply=True`` → dashed return arrow that ends ``a``'s activation;
        ``activate=True`` starts an activation bar on ``b``; ``head``: "arrow" (sync) or "open" (async)."""
        self._need(a, b)
        self._events.append(_N(kind="msg", a=a, b=b, label=label, reply=reply,
                               style=style or ("dashed" if reply else "solid"), activate=activate,
                               deactivate=deactivate or reply, head=head or ("open" if reply else "arrow"),
                               color=color))
        return self

    def reply(self, a, b, label=None, **kw):
        """Dashed return message (ends the sender's activation)."""
        return self.message(a, b, label, reply=True, **kw)

    def activate(self, id):
        self._need(id)
        self._events.append(_N(kind="act", a=id))
        return self

    def deactivate(self, id):
        self._events.append(_N(kind="deact", a=id))
        return self

    def note(self, over, text, side="over"):
        """Sticky note. ``over``: a participant id or a pair (a, b) to span; ``side``: over / left / right."""
        ids = (over,) if isinstance(over, str) else tuple(over)
        self._need(*ids)
        self._events.append(_N(kind="note", ids=ids, text=text, side=side))
        return self

    def frame(self, kind, label=None):
        """Box around the messages inside ``with s.frame("alt", "cache hit"):`` (loop / alt / opt / par / ...)."""
        ev = _N(kind="frame", fkind=kind, label=label, else_rows=[])
        self._events.append(ev)
        return _FrameHandle(self, ev)

    def otherwise(self, label=None):
        """Dashed separator inside the current ``alt`` frame (the "else" branch)."""
        self._events.append(_N(kind="else", label=label))
        return self

    def divider(self, text):
        """Full-width separator with a label (a pause, a phase)."""
        self._events.append(_N(kind="divider", text=text))
        return self

    # ---- layout
    def _size(self):
        th = self.theme
        fam = _fam(th)
        fs = self.fs
        sp = self.spacing
        P = list(self.parts.values())
        for p in P:
            lw, lh = _measure(p.label, fs, "bold", fam)
            p.lw, p.lh = lw, lh
            # an actor is a header like the others (person badge + label)
            p.W = max(lw + fs * 2.4 + (fs * 1.8 if p.icon else 0), fs * 7.5)
            p.H = max(fs * 3.2, lh + fs * 1.6)
        n = len(P)
        lfs = round(fs * 0.86, 1)
        self.lfs = lfs
        gaps = [max((P[i].W + P[i + 1].W) / 2 + 36 * sp, 150 * sp) * self._kx for i in range(n - 1)]
        reqs = []
        for ev in self._events:
            if ev.kind == "msg" and ev.label:
                lw = _measure(ev.label, lfs, "normal", fam)[0] + (30 if self.autonumber else 0)
                ia, ib = self.parts[ev.a].idx, self.parts[ev.b].idx
                if ia == ib:
                    if ia < n - 1:
                        reqs.append((ia, ia + 1, lw + 70))
                else:
                    reqs.append((min(ia, ib), max(ia, ib), lw + 48))
            if ev.kind == "note":
                w = _measure(_wrap(ev.text, lfs, 240, "normal", fam), lfs, "normal", fam)[0] + 28
                idx = sorted(self.parts[i].idx for i in ev.ids)
                if len(idx) == 1 and ev.side in ("right", "left"):
                    i = idx[0]
                    j = i + 1 if ev.side == "right" else i - 1
                    if 0 <= j < n:
                        reqs.append((min(i, j), max(i, j), w + 30))
        for i, j, need in sorted(reqs, key=lambda r: r[1] - r[0]):
            have = sum(gaps[i:j])
            if have < need:
                extra = (need - have) / (j - i)
                for k in range(i, j):
                    gaps[k] += extra
        xs = [P[0].W / 2 if P else 0.0]
        for g in gaps:
            xs.append(xs[-1] + g)
        for p, x in zip(P, xs):
            p.x = x
        head_h = max([p.H for p in P] or [0])
        self.head_h = head_h
        # rows
        sp = self.spacing * self._ky
        y = head_h + 30 * sp
        rows = []
        depth = {p.id: [] for p in P}
        frames = []
        num = 0
        for ev in self._events:
            k = ev.kind
            if k == "msg":
                lh = _measure(ev.label, lfs, "normal", fam)[1] if ev.label else 0
                selfm = ev.a == ev.b
                y += lh + 12 * sp if ev.label else 6
                ev.y = y
                num += 1
                ev.num = num
                ev.depth_a = len(depth[ev.a])
                if ev.activate:
                    depth[ev.b].append(_N(y0=y, y1=None, level=len(depth[ev.b]), pid=ev.b))
                ev.depth_b = len(depth[ev.b])
                if ev.deactivate and depth[ev.a]:
                    bar = depth[ev.a].pop()
                    bar.y1 = y
                    rows.append(bar)
                ev.depth_a2 = len(depth[ev.a])
                y += (34 if selfm else 22) * sp
                if selfm:
                    ev.y2 = y - 14 * sp
            elif k == "act":
                depth[ev.a].append(_N(y0=y - 6, y1=None, level=len(depth[ev.a]), pid=ev.a))
            elif k == "deact":
                if depth[ev.a]:
                    bar = depth[ev.a].pop()
                    bar.y1 = y - 6
                    rows.append(bar)
            elif k == "note":
                txt = _wrap(ev.text, lfs, 240, "normal", fam)
                w, h = _measure(txt, lfs, "normal", fam)
                ev.txt, ev.w, ev.h = txt, w + 28, h + 18
                ev.y = y + 4
                y += ev.h + 18 * sp
            elif k == "frame":
                ev.y = y
                frames.append(ev)
                y += 40 * sp
                ev.ids = set()
                ev.else_rows = []                      # the layout can run more than once: collect afresh
            elif k == "else":
                if frames:
                    ev.y = y
                    frames[-1].else_rows.append(ev)
                    y += 30 * sp
            elif k == "frame_end":
                if frames:
                    f = frames.pop()
                    f.y1 = y + 4
                    y += 18 * sp
            elif k == "divider":
                ev.y = y + 16
                y += 44 * sp
            if k in ("msg", "note") and frames:
                ids = (ev.a, ev.b) if k == "msg" else ev.ids
                for f in frames:
                    f.ids.update(ids)
        for lst in depth.values():
            for bar in lst:
                bar.y1 = y
                rows.append(bar)
        self.bars = rows
        self.y_end = y + 10
        W = (xs[-1] + P[-1].W / 2) if P else 0
        for ev in self._events:
            if ev.kind == "msg" and ev.a == ev.b and ev.label:
                lw = _measure(ev.label, lfs, "normal", fam)[0]
                W = max(W, self.parts[ev.a].x + 46 + lw + 10)
            if ev.kind == "note":
                xs_ = [self.parts[i].x for i in ev.ids]
                if ev.side == "right":
                    W = max(W, max(xs_) + 20 + ev.w)
        H = self.y_end + (head_h + 20 if self.mirror else 0)
        # frames may extend left of the first participant
        self.x_shift = 0.0
        for ev in self._events:
            if ev.kind == "note" and ev.side == "left":
                x0 = min(self.parts[i].x for i in ev.ids) - 20 - ev.w
                self.x_shift = max(self.x_shift, -x0)
        return W + self.x_shift, H

    def _paint(self, dwg, ox, oy):
        th = self.theme
        fam = _fam(th)
        fs, lfs = self.fs, self.lfs
        ox += self.x_shift
        P = list(self.parts.values())
        acc = {p.id: _accent(th, p.color, p.idx) for p in P}
        ecol = _mix(th.ink, th.muted, 0.35)
        bar_w = 12
        y_top = oy + self.head_h
        # frames (behind everything)
        frames = [ev for ev in self._events if ev.kind == "frame" and hasattr(ev, "y1")]
        if frames:
            with dwg.group("frames"):
                for f in frames:
                    ids = f.ids or set(self.parts)
                    xs = [self.parts[i].x for i in ids]
                    x0 = min(xs) - 70 + ox
                    x1 = max(xs) + 70 + ox
                    for ev in self._events:
                        if ev.kind == "msg" and ev.a == ev.b and ev.a in ids and f.y <= ev.y <= f.y1 and ev.label:
                            x1 = max(x1, self.parts[ev.a].x + ox + 56 + _measure(ev.label, lfs, "normal", fam)[0])
                    y0, y1 = f.y + oy, f.y1 + oy
                    col = th.accent(4) if f.fkind in ("loop", "par") else th.accent(5)
                    dwg.rect(x0, y0, x1 - x0, y1 - y0, r=10, fill=_mix(col, th.background, 0.05),
                             stroke=_mix(col, th.background, 0.55), stroke_width=1.3)
                    kw = _measure(f.fkind.upper(), 12, "bold", fam)[0]
                    tag = Path().move_to(x0 + 10, y0).line_to(x0 + kw + 26, y0).line_to(x0 + kw + 26, y0 + 18)
                    tag.line_to(x0 + kw + 18, y0 + 26).line_to(x0, y0 + 26).line_to(x0, y0 + 10)
                    tag.quad_to(x0, y0, x0 + 10, y0).close()
                    dwg.path(tag, fill=_mix(col, th.background, 0.8))
                    dwg.text(x0 + 10, y0 + 13, f.fkind.upper(), size=12, weight="bold", fill=_on(col, th),
                             anchor="lm", letter_spacing=1, family=fam)
                    if f.label:
                        dwg.text(x0 + kw + 36, y0 + 13, f"[{f.label}]", size=lfs * 0.95, fill=th.muted, anchor="lm",
                                 family=fam, style="italic")
                    for e in f.else_rows:
                        ye = e.y + oy + 6
                        dwg.line(x0, ye, x1, ye, stroke=_mix(col, th.background, 0.5), stroke_width=1.2,
                                 dash=(5, 4))
                        if e.label:
                            dwg.text(x0 + 12, ye + 6, f"[{e.label}]", size=lfs * 0.95, fill=th.muted, anchor="lt",
                                     family=fam, style="italic")
        # lifelines
        with dwg.group("lifelines"):
            for p in P:
                x = p.x + ox
                dwg.line(x, y_top + 6, x, oy + self.y_end, stroke=_mix(th.muted, th.background, 0.55),
                         stroke_width=1.5, dash=(6, 5))
        # dividers
        divs = [ev for ev in self._events if ev.kind == "divider"]
        if divs:
            with dwg.group("dividers"):
                xa = P[0].x + ox - P[0].W / 2
                xb = P[-1].x + ox + P[-1].W / 2
                for ev in divs:
                    y = ev.y + oy
                    dwg.line(xa, y, xb, y, stroke=_mix(th.muted, th.background, 0.6), stroke_width=1.2, dash=(2, 4),
                             cap="round")
                    w, h = _measure(ev.text, lfs, "bold", fam)
                    cx = (xa + xb) / 2
                    dwg.rect(cx - w / 2 - 14, y - h / 2 - 6, w + 28, h + 12, r=(h + 12) / 2, fill=th.surface,
                             stroke=th.grid, stroke_width=1)
                    dwg.text(cx, y, ev.text, size=lfs, weight="bold", fill=th.muted, anchor="mm", family=fam)
        # activation bars
        if self.bars:
            with dwg.group("activations"):
                for bar in self.bars:
                    p = self.parts[bar.pid]
                    x = p.x + ox - bar_w / 2 + bar.level * bar_w * 0.6
                    dwg.rect(x, bar.y0 + oy - 4, bar_w, bar.y1 - bar.y0 + 8, r=3,
                             fill=_mix(acc[p.id], th.surface, 0.22 if th.dark else 0.16), stroke=acc[p.id],
                             stroke_width=1.3)
        # messages
        with dwg.group("messages"):
            for ev in self._events:
                if ev.kind != "msg":
                    continue
                pa, pb = self.parts[ev.a], self.parts[ev.b]
                col = _accent(th, ev.color) if ev.color is not None else ecol
                y = ev.y + oy
                xa, xb = pa.x + ox, pb.x + ox
                sw = 1.6
                hs = 11
                if ev.a == ev.b:
                    x0 = xa + (bar_w / 2 + bar_w * 0.6 * max(ev.depth_a - 1, 0) if ev.depth_a else 0)
                    y2 = ev.y2 + oy
                    xr = xa + 44
                    xe = xa + (bar_w / 2 + bar_w * 0.6 * max(ev.depth_b - 1, 0) if ev.depth_b else 0)
                    pts = [(x0, y), (xr, y), (xr, y2), (xe + _arrow_trim(ev.head, hs), y2)]
                    dwg.path(_round_poly(pts, 8, closed=False), stroke=col, stroke_width=sw, fill=None,
                             dash=_dash(ev.style, sw))
                    _arrowhead(dwg, (xe, y2), (-1.0, 0.0), hs, col, ev.head)
                    if ev.label:
                        dwg.text(xr + 8, (y + y2) / 2, ev.label, size=lfs, fill=th.ink, anchor="lm", family=fam)
                    lx = xa + 22
                else:
                    s = 1 if xb > xa else -1
                    da = ev.depth_a if not ev.reply else ev.depth_a
                    off_a = (bar_w / 2 + bar_w * 0.6 * max(da - 1, 0)) if da else 0
                    db = ev.depth_b
                    off_b = (bar_w / 2 + bar_w * 0.6 * max(db - 1, 0)) if db else 0
                    x0 = xa + s * off_a
                    x1 = xb - s * off_b
                    dwg.line(x0, y, x1 - s * _arrow_trim(ev.head, hs), y, stroke=col, stroke_width=sw,
                             dash=_dash(ev.style, sw))
                    _arrowhead(dwg, (x1, y), (float(s), 0.0), hs, col, ev.head)
                    lx = (x0 + x1) / 2
                    if ev.label:
                        dwg.text(lx + (12 if self.autonumber else 0), y - 7, ev.label, size=lfs,
                                 fill=th.ink if not ev.reply else th.muted, anchor="mb", family=fam)
                if self.autonumber:
                    if ev.a == ev.b:
                        bx, by = xa + 22, y
                    else:
                        lw = _measure(ev.label, lfs, "normal", fam)[0] if ev.label else 0
                        bx, by = lx - lw / 2 - 4, y - 7 - _measure("8", lfs, "normal", fam)[1] / 2
                        if not ev.label:
                            bx, by = (xa + xb) / 2, y - 14
                    dwg.circle(bx, by, 10, fill=acc[ev.a])
                    dwg.text(bx, by, str(ev.num), size=11, weight="bold", fill=_on(acc[ev.a], th), anchor="mm",
                             family=fam)
        # notes
        notes = [ev for ev in self._events if ev.kind == "note"]
        if notes:
            with dwg.group("notes"):
                ncol = th.accent(1) if not th.dark else th.accent(7)
                for ev in notes:
                    xs = [self.parts[i].x + ox for i in ev.ids]
                    if ev.side == "right":
                        x0 = max(xs) + 20
                    elif ev.side == "left":
                        x0 = min(xs) - 20 - ev.w
                    else:
                        w = max(ev.w, (max(xs) - min(xs)) + 80 if len(xs) > 1 else ev.w)
                        ev.w = w
                        x0 = (min(xs) + max(xs)) / 2 - w / 2
                    y0 = ev.y + oy
                    with dwg.group(shadow=_shadow(th, 0.5)) as g:
                        f = 12
                        pts = [(x0, y0), (x0 + ev.w - f, y0), (x0 + ev.w, y0 + f), (x0 + ev.w, y0 + ev.h),
                               (x0, y0 + ev.h)]
                        fill = _mix(ncol, th.surface, 0.2 if th.dark else 0.13)
                        g.path(_round_poly(pts, 3), fill=fill, stroke=_mix(ncol, th.surface, 0.55), stroke_width=1.2)
                        g.polygon([(x0 + ev.w - f, y0), (x0 + ev.w, y0 + f), (x0 + ev.w - f, y0 + f)],
                                  fill=_mix(ncol, th.surface, 0.4))
                    dwg.text(x0 + ev.w / 2, y0 + ev.h / 2, ev.txt, size=lfs, fill=th.ink, anchor="mm", align="center",
                             family=fam)
        # participants (headers, one layer each)
        for p in P:
            self._head(dwg, p, p.x + ox, oy, acc[p.id])
            if self.mirror:
                self._head(dwg, p, p.x + ox, oy + self.y_end + 16, acc[p.id], suffix=" (bottom)")

    def _head(self, dwg, p, x, y, col, suffix=""):
        th = self.theme
        fam = _fam(th)
        fs = self.fs
        with dwg.group(f"participant {p.label}{suffix}", shadow=_shadow(th, 0.8)) as g:
            H = self.head_h
            g.rect(x - p.W / 2, y, p.W, H, r=th.radius * 1.4, fill=th.surface, stroke=_mix(col, th.surface, 0.45),
                   stroke_width=1.5)
            clip = dwg.rect(x - p.W / 2, y, p.W, H, r=th.radius * 1.4, add=False)
            with g.group(clip=clip) as cg:
                cg.rect(x - p.W / 2, y, p.W, 4, fill=col)
            if p.icon:
                tw = p.lw + fs * 1.8
                ix = x - tw / 2 + fs * 0.65 + (fs * 0.2 if p.kind == "actor" else 0)
                if p.kind == "actor":
                    g.circle(ix, y + H / 2 + 1, fs * 0.85, fill=_mix(col, th.surface, 0.2), stroke=col, stroke_width=1.4)
                g.icon(p.icon, ix, y + H / 2 + 1, fs * 1.25, color=col)
                g.text(ix + fs * 1.15, y + H / 2 + 1, p.label, size=fs, weight="bold", anchor="lm", family=fam)
            else:
                g.text(x, y + H / 2 + 1, p.label, size=fs, weight="bold", anchor="mm", family=fam)


# --------------------------------------------------------------------------------------------- Timeline

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _when(v):
    """Date-like value → (months since year 0, is_date). Accepts 2026, 2026.5, '2026-03', '2026-03-15',
    'Q3 2026', '2026 Q3', 'Mar 2026', datetime.date; other numbers are plain values."""
    import datetime as _dt
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.year * 12 + v.month - 1 + (v.day - 1) / 31.0, True
    if isinstance(v, (int, float, np.integer, np.floating)):
        if 1000 <= float(v) <= 3000:
            return float(v) * 12, True
        return float(v), False
    s = str(v).strip()
    up = s.upper().replace("-", " ").replace("/", " ")
    parts = up.split()
    if len(parts) == 2 and (parts[0].startswith("Q") or parts[1].startswith("Q")):
        q, y = (parts[0], parts[1]) if parts[0].startswith("Q") else (parts[1], parts[0])
        return int(y) * 12 + (int(q[1:]) - 1) * 3, True
    if len(parts) == 2 and parts[0][:3].title() in _MONTHS:
        return int(parts[1]) * 12 + _MONTHS.index(parts[0][:3].title()), True
    nums = [p for p in s.replace("/", "-").split("-") if p]
    try:
        y = int(nums[0])
        m = int(nums[1]) if len(nums) > 1 else 1
        d = int(nums[2]) if len(nums) > 2 else 1
        return y * 12 + m - 1 + (d - 1) / 31.0, True
    except (ValueError, IndexError):
        raise ValueError(f"cannot read the date {v!r}; use 2026, '2026-03', '2026-03-15' or 'Q3 2026'") from None


def _date_label(m, unit):
    y, mo = int(m // 12), int(round(m % 12)) % 12
    if unit == "year":
        return str(y)
    if unit == "quarter":
        return f"Q{mo // 3 + 1} {y}"
    return _MONTHS[mo]


class Timeline(_Figure):
    """Roadmap / timeline: phases as bars above a time axis, milestones as dots with labels below.

    ``t = Timeline(theme="light", title="Roadmap")``; ``t.phase("2026-01", "2026-04", "Discovery")``;
    ``t.milestone("2026-05-10", "Beta", icon="star")``; ``t.marker("2026-03-01", "Today")``.
    Dates: 2026, '2026-03', '2026-03-15', 'Q3 2026', datetime.date — or plain numbers.
    """

    def __init__(self, theme="light", title=None, subtitle=None, start=None, end=None, width=1500,
                 font_size=17, margin=40, unit=None):
        super().__init__(theme, title, subtitle, margin)
        self.start, self.end, self.axis_w, self.fs, self.unit = start, end, width, font_size, unit
        self.phases, self.milestones, self.markers = [], [], []
        self._w0 = width

    def _stretch(self, aspect, hw, hh):
        """Choose the axis width so the timeline fills a box of this ``aspect``."""
        self.axis_w = self._w0
        if not aspect:
            return
        m = self.margin
        for _ in range(5):
            cw, ch = self._size()
            H = ch + hh + 2 * m
            w = max(500.0, aspect * H - 2 * m)
            if abs(w - self.axis_w) < 4:
                break
            self.axis_w = w

    def phase(self, start, end, label, color=None, sublabel=None, lane=None):
        """A bar from ``start`` to ``end`` (lanes are packed automatically; ``lane=`` forces one)."""
        self.phases.append(_N(a=start, b=end, label=label, color=color, sub=sublabel, lane=lane,
                              idx=len(self.phases)))
        return self

    def milestone(self, when, label, sublabel=None, icon=None, color=None):
        """A dot on the axis with a label card (date shown automatically)."""
        if icon is not None and icon not in _v.ICONS:
            raise _v._unknown("icon", icon, _v.ICONS, "Try vector.icon_names()")
        self.milestones.append(_N(t=when, label=label, sub=sublabel, icon=icon, color=color,
                                  idx=len(self.milestones)))
        return self

    def marker(self, when, label="Today", color=None):
        """A vertical line across the whole timeline (e.g. today)."""
        self.markers.append(_N(t=when, label=label, color=color))
        return self

    def _size(self):
        th = self.theme
        fam = _fam(th)
        fs = self.fs
        vals = []
        for p in self.phases:
            p.ta, da = _when(p.a)
            p.tb, db = _when(p.b)
            vals += [(p.ta, da), (p.tb, db)]
        for m in self.milestones + self.markers:
            m.tt, dm = _when(m.t)
            vals.append((m.tt, dm))
        if self.start is not None:
            vals.append(_when(self.start))
        if self.end is not None:
            vals.append(_when(self.end))
        if not vals:
            vals = [(0.0, False), (1.0, False)]
        self.is_date = any(d for _, d in vals)
        lo = _when(self.start)[0] if self.start is not None else min(v for v, _ in vals)
        hi = _when(self.end)[0] if self.end is not None else max(v for v, _ in vals)
        span = max(hi - lo, 1e-9)
        if self.is_date:
            unit = self.unit or ("month" if span <= 15 else ("quarter" if span <= 48 else "year"))
            step = {"month": 1, "quarter": 3, "year": 12}[unit]
            if self.start is None:
                lo = math.floor(lo / step) * step
            if self.end is None:
                hi = math.ceil(hi / step + 1e-9) * step
                if hi <= lo:
                    hi = lo + step
            ticks = list(np.arange(lo, hi + 1e-6, step))
            if unit == "month" and len(ticks) > 14:
                ticks = ticks[::2]
            if unit == "year" and len(ticks) > 14:
                ticks = ticks[::int(math.ceil(len(ticks) / 12))]
        else:
            unit = None
            raw = span / 8
            mag = 10 ** math.floor(math.log10(raw))
            step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=mag * 10)
            lo = math.floor(lo / step) * step if self.start is None else lo
            hi = math.ceil(hi / step) * step if self.end is None else hi
            ticks = list(np.arange(lo, hi + step * 1e-6, step))
        self.lo, self.hi, self.tick_unit, self.ticks = lo, hi, unit, ticks
        W = self.axis_w
        self.x = lambda t: (t - lo) / max(hi - lo, 1e-9) * W
        # phases → lanes
        bar_h = fs * 2.1
        self.bar_h = bar_h
        lanes = []
        for p in sorted(self.phases, key=lambda p: (p.ta, p.idx)):
            x0, x1 = self.x(p.ta), self.x(p.tb)
            lw = _measure(p.label, fs * 0.94, "bold", fam)[0]
            p.inside = lw + 28 <= x1 - x0
            ext1 = x1 if p.inside else x1 + 10 + lw
            if p.lane is None:
                k = 0
                while k < len(lanes) and lanes[k] > x0 - 6:
                    k += 1
                p.lane = k
            while len(lanes) <= p.lane:
                lanes.append(-1e9)
            lanes[p.lane] = max(lanes[p.lane], ext1)
            p.x0, p.x1 = x0, x1
        nl = len(lanes)
        self.lane_gap = 10
        top = 0.0
        self.phase_top = top
        axis_y = top + nl * (bar_h + self.lane_gap) + (26 if nl else 6)
        self.axis_y = axis_y
        # milestones → label cards with levels
        cards = []
        ms = sorted(self.milestones, key=lambda m: m.tt)
        for m in ms:
            date = self._fmt(m.tt)
            m.date = date
            lw, lh = _measure(m.label, fs, "bold", fam)
            dw, dh = _measure(date, fs * 0.74, "bold", fam)
            sw, sh = _measure(m.sub, fs * 0.78, "normal", fam) if m.sub else (0, 0)
            icw = fs * 1.6 if m.icon else 0
            m.cw = max(lw + icw, dw, sw) + 28
            m.chh = dh + 6 + lh + (sh + 4 if m.sub else 0) + 22
            m.lh, m.dh, m.sh, m.icw = lh, dh, sh, icw
            x = self.x(m.tt)
            cx = min(max(x, m.cw / 2), W - m.cw / 2)
            m.cx = cx
            level = 0
            while any(c[0] == level and abs(c[1] - cx) < (c[2] + m.cw) / 2 + 12 for c in cards):
                level += 1
            m.level = level
            cards.append((level, cx, m.cw))
        lv_h = max([m.chh for m in ms] or [0]) + 18
        self.lv_h = lv_h
        nlev = max([m.level for m in ms] or [-1]) + 1
        self.card_top = axis_y + 58
        H = self.card_top + nlev * lv_h + 4 if ms else axis_y + 40
        return W, H

    def _fmt(self, t):
        if not self.is_date:
            return f"{t:g}"
        y, mo = int(t // 12), int(t % 12)
        d = int(round((t - math.floor(t)) * 31)) + 1
        if self.tick_unit == "year" and mo == 0 and d == 1:
            return str(y)
        if d > 1:
            return f"{_MONTHS[mo]} {d}, {y}"
        return f"{_MONTHS[mo]} {y}"

    def _paint(self, dwg, ox, oy):
        th = self.theme
        fam = _fam(th)
        fs = self.fs
        W = self.axis_w
        X = lambda t: ox + self.x(t)
        ay = oy + self.axis_y
        # grid + ticks
        with dwg.group("axis"):
            for t in self.ticks:
                x = X(t)
                dwg.line(x, oy + self.phase_top - 4, x, ay, stroke=_mix(th.grid, th.background, 0.8), stroke_width=1)
            dwg.rect(ox - 4, ay - 3, W + 8, 6, r=3, fill=_mix(th.muted, th.background, 0.35))
            for t in self.ticks:
                dwg.circle(X(t), ay, 3.2, fill=th.background, stroke=_mix(th.muted, th.background, 0.6),
                           stroke_width=1.5)
        # phases
        if self.phases:
            with dwg.group("phases"):
                for p in self.phases:
                    col = _accent(th, p.color, p.idx)
                    y = oy + self.phase_top + p.lane * (self.bar_h + self.lane_gap)
                    x0, x1 = ox + p.x0, ox + p.x1
                    with dwg.group(f"phase {p.label}", shadow=_shadow(th, 0.5)) as g:
                        g.rect(x0, y, max(x1 - x0, self.bar_h), self.bar_h, r=self.bar_h / 2,
                               fill=_mix(col, th.surface, 0.24 if th.dark else 0.17), stroke=_mix(col, th.surface, 0.7),
                               stroke_width=1.3)
                        g.circle(x0 + self.bar_h / 2, y + self.bar_h / 2, self.bar_h * 0.17, fill=col)
                    if p.inside:
                        dwg.text(x0 + self.bar_h * 0.85, y + self.bar_h / 2, p.label, size=fs * 0.94, weight="bold",
                                 anchor="lm", family=fam)
                    else:
                        dwg.text(x1 + 10, y + self.bar_h / 2, p.label, size=fs * 0.94, weight="bold", anchor="lm",
                                 family=fam)
        # markers
        for mk in self.markers:
            col = _accent(th, mk.color, 3)
            x = X(mk.tt)
            with dwg.group(f"marker {mk.label}"):
                dwg.line(x, oy + self.phase_top - 22, x, oy + self._size_h() - 4, stroke=col, stroke_width=1.6,
                         dash=(5, 4))
                w, h = _measure(mk.label, fs * 0.72, "bold", fam)
                dwg.rect(x - w / 2 - 9, oy + self.phase_top - 40, w + 18, h + 10, r=(h + 10) / 2, fill=col)
                dwg.text(x, oy + self.phase_top - 40 + (h + 10) / 2, mk.label, size=fs * 0.72, weight="bold",
                         fill=_on(col, th), anchor="mm", family=fam)
        # milestones
        for m in self.milestones:
            col = _accent(th, m.color, m.idx + 2)
            x = X(m.tt)
            top = oy + self.card_top + m.level * self.lv_h
            cx = ox + m.cx
            with dwg.group(f"milestone {m.label}"):
                dwg.path(_round_poly([(x, ay), (x, top - 14), (cx, top - 14), (cx, top)], 8, closed=False)
                         if abs(cx - x) > 1 else Path().move_to(x, ay).line_to(x, top),
                         stroke=_mix(col, th.background, 0.6), stroke_width=1.5, fill=None)
                dwg.circle(x, ay, 9, fill=col, stroke=th.background, stroke_width=3.5)
                with dwg.group(shadow=_shadow(th, 0.7)) as g:
                    g.rect(cx - m.cw / 2, top, m.cw, m.chh, r=th.radius * 1.3, fill=th.surface,
                           stroke=_mix(col, th.surface, 0.4), stroke_width=1.3)
                ty = top + 11
                dwg.text(cx - m.cw / 2 + 14, ty, m.date.upper(), size=fs * 0.74, weight="bold", fill=col, anchor="lt",
                         letter_spacing=0.6, family=fam)
                ty += m.dh + 6
                tx = cx - m.cw / 2 + 14
                if m.icon:
                    dwg.icon(m.icon, tx + fs * 0.6, ty + m.lh / 2, fs * 1.1, color=col)
                    tx += m.icw
                dwg.text(tx, ty, m.label, size=fs, weight="bold", anchor="lt", family=fam)
                if m.sub:
                    dwg.text(cx - m.cw / 2 + 14, ty + m.lh + 4, m.sub, size=fs * 0.78, fill=th.muted, anchor="lt",
                             family=fam)

        with dwg.group("axis labels"):
            prev_year = None
            for t in self.ticks:
                x = X(t)
                if self.is_date:
                    lab = _date_label(t, self.tick_unit)
                    y = int(t // 12)
                    dwg.text(x, ay + 12, lab, size=fs * 0.74, fill=th.muted, anchor="mt", family=fam,
                             weight="bold" if self.tick_unit != "month" else "normal", halo=th.background)
                    if self.tick_unit == "month" and y != prev_year:
                        dwg.text(x, ay + 12 + fs * 0.95, str(y), size=fs * 0.66, fill=_mix(th.muted, th.background,
                                                                                         0.7), anchor="mt", family=fam,
                                 halo=th.background)
                    prev_year = y
                else:
                    dwg.text(x, ay + 12, f"{t:g}", size=fs * 0.74, fill=th.muted, anchor="mt", family=fam,
                             halo=th.background)

    def _size_h(self):
        ms = self.milestones
        nlev = max([m.level for m in ms] or [-1]) + 1
        return self.card_top + nlev * self.lv_h if ms else self.axis_y + 30


# --------------------------------------------------------------------------------------------- Table

_YES = {"yes", "y", "true", "✓", "✔", "check", "ok", "+"}
_NO = {"no", "n", "false", "✗", "✘", "x", "cross", "-x"}
_PART = {"partial", "~", "some", "half", "◐"}


def _cell_kind(v):
    if v is True:
        return "yes"
    if v is False:
        return "no"
    if v is None:
        return "none"
    if isinstance(v, (int, float, np.integer, np.floating)):
        return "num"
    s = str(v).strip()
    if s.lower() in _YES:
        return "yes"
    if s.lower() in _NO:
        return "no"
    if s.lower() in _PART:
        return "part"
    if s in ("", "-", "—", "–"):
        return "none"
    t = s.replace(",", "").replace("$", "").replace("%", "").replace("€", "").replace("£", "").replace("x", "")
    try:
        float(t)
        return "num"
    except ValueError:
        return "text"


class Table(_Figure):
    """Comparison table / matrix with a header row, zebra rows and check / cross marks.

    ``Table(["Feature", "Free", "Pro"], [["SSO", False, True], ["Seats", "3", "Unlimited"]], highlight="Pro")``

    * Cells: True / "yes" / "✓" → check, False / "no" / "✗" → cross, "partial" / "~" → half, None / "-" → dash,
      numbers right-aligned, text left-aligned (``align=["l", "c", "r", ...]`` overrides).
    * ``highlight``: a column index or name, drawn as the recommended column (``badge`` text on top).
    """

    def __init__(self, columns, rows, theme="light", title=None, subtitle=None, align=None, highlight=None,
                 badge="Recommended", zebra=True, font_size=17, max_col=300, margin=40, row_labels=True):
        super().__init__(theme, title, subtitle, margin)
        self.columns = [str(c) for c in columns]
        self.rows = [list(r) + [None] * (len(columns) - len(r)) for r in rows]
        self.align, self.zebra, self.fs, self.max_col = align, zebra, font_size, max_col
        self.badge, self.row_labels = badge, row_labels
        self._kx = self._ky = 1.0
        self._xmax, self._ymax = 2.4, 1.9
        if isinstance(highlight, str):
            highlight = self.columns.index(highlight)
        self.highlight = highlight

    def _stretch(self, aspect, hw, hh):
        """Adapt spacing so the figure approaches ``aspect`` (width / height)."""
        self._kx = self._ky = 1.0
        if not aspect:
            return
        m = self.margin
        for _ in range(4):
            cw, ch = self._size()
            W, H = max(cw, hw) + 2 * m, ch + hh + 2 * m
            r = (W / H) / aspect
            if abs(r - 1) < 0.03:
                break
            if r < 1:
                nk = min(self._xmax, self._kx * (1 + (1 / r - 1) * 1.3))
                if nk <= self._kx + 1e-3:
                    break
                self._kx = nk
            else:
                nk = min(self._ymax, self._ky * (1 + (r - 1) * 1.3))
                if nk <= self._ky + 1e-3:
                    break
                self._ky = nk

    def _size(self):
        th = self.theme
        fam = _fam(th)
        fs = self.fs
        nc = len(self.columns)
        self.pad_x, self.pad_y = fs * 1.1, fs * 0.62 * self._ky
        self.kinds = [[_cell_kind(v) for v in r] for r in self.rows]
        widths = []
        for j, c in enumerate(self.columns):
            w = _measure(c, fs * 0.82, "bold", fam)[0] * 1.06 + 6
            for i, r in enumerate(self.rows):
                k = self.kinds[i][j]
                if k in ("yes", "no", "part", "none"):
                    w = max(w, fs * 1.6)
                else:
                    txt = _wrap(r[j], fs, self.max_col, "bold" if (j == 0 and self.row_labels) else "normal", fam)
                    w = max(w, _measure(txt, fs, "bold" if (j == 0 and self.row_labels) else "normal", fam)[0])
            widths.append(min(w, self.max_col) + 2 * self.pad_x)
        if nc > 1:
            rest = max(widths[1:])
            widths = [widths[0]] + [max(w, min(rest, fs * 9)) for w in widths[1:]]
        if self._kx > 1:
            extra = sum(widths) * (self._kx - 1)
            widths = [widths[0] + extra * 0.25] + [w + extra * 0.75 / max(nc - 1, 1) for w in widths[1:]]                 if nc > 1 else [widths[0] + extra]
        self.widths = widths
        self.head_h = _measure("Ag", fs * 0.82, "bold", fam)[1] + 2 * self.pad_y + 4
        hs = []
        self.cells = []
        for i, r in enumerate(self.rows):
            h = fs * 1.45
            row = []
            for j, v in enumerate(r):
                k = self.kinds[i][j]
                if k in ("text", "num"):
                    wt = "bold" if (j == 0 and self.row_labels) else "normal"
                    txt = _wrap(v, fs, widths[j] - 2 * self.pad_x, wt, fam)
                    h = max(h, _measure(txt, fs, wt, fam)[1])
                    row.append(txt)
                else:
                    row.append(None)
            self.cells.append(row)
            hs.append(h + 2 * self.pad_y)
        self.row_h = hs
        self.badge_h = (fs * 1.9) if (self.highlight is not None and self.badge) else 0
        return sum(widths), self.badge_h + self.head_h + sum(hs)

    def _paint(self, dwg, ox, oy):
        th = self.theme
        fam = _fam(th)
        fs = self.fs
        W = sum(self.widths)
        top = oy + self.badge_h
        H = self.head_h + sum(self.row_h)
        R = th.radius * 1.8
        acc = th.accent(0)
        xs = [ox]
        for w in self.widths:
            xs.append(xs[-1] + w)
        aligns = []
        for j in range(len(self.columns)):
            if self.align and j < len(self.align) and self.align[j]:
                a = {"l": "left", "c": "center", "m": "center", "r": "right"}.get(self.align[j][0], "left")
            else:
                ks = [self.kinds[i][j] for i in range(len(self.rows))]
                if ks and all(k in ("yes", "no", "part", "none") for k in ks):
                    a = "center"
                elif ks and all(k in ("num", "none") for k in ks):
                    a = "right"
                else:
                    a = "left" if j == 0 else "center"
            aligns.append(a)
        with dwg.group("table card", shadow=_shadow(th, 1.0)) as g:
            g.rect(ox, top, W, H, r=R, fill=th.surface, stroke=th.grid, stroke_width=1.2)
        clip = dwg.rect(ox, top, W, H, r=R, add=False)
        with dwg.group("rows", clip=clip):
            head_fill = _mix(th.ink, th.surface, 0.06 if not th.dark else 0.08)
            dwg.rect(ox, top, W, self.head_h, fill=head_fill)
            y = top + self.head_h
            for i, h in enumerate(self.row_h):
                if self.zebra and i % 2 == 1:
                    dwg.rect(ox, y, W, h, fill=_mix(th.ink, th.surface, 0.025 if not th.dark else 0.035))
                y += h
            if self.highlight is not None:
                j = self.highlight
                dwg.rect(xs[j], top, self.widths[j], H, fill=_mix(acc, th.surface, 0.09 if not th.dark else 0.14))
                dwg.rect(xs[j], top, self.widths[j], self.head_h, fill=_mix(acc, th.surface, 0.2 if not th.dark
                                                                            else 0.3))
            y = top + self.head_h
            for i, h in enumerate(self.row_h[:-1]):
                y += h
                dwg.line(ox, y, ox + W, y, stroke=th.grid, stroke_width=1)
            dwg.line(ox, top + self.head_h, ox + W, top + self.head_h, stroke=_mix(th.muted, th.surface, 0.4),
                     stroke_width=1.2)
        if self.highlight is not None:
            j = self.highlight
            with dwg.group("highlight"):
                dwg.rect(xs[j] + 1, top + 1, self.widths[j] - 2, H - 2, r=R if j in (0, len(self.widths) - 1) else 4,
                         fill=None, stroke=acc, stroke_width=2.2)
                if self.badge:
                    bw = _measure(self.badge, fs * 0.72, "bold", fam)[0] + 22
                    bh = fs * 1.45
                    bx = xs[j] + self.widths[j] / 2
                    dwg.rect(bx - bw / 2, top - bh * 0.62, bw, bh, r=bh / 2, fill=acc)
                    dwg.text(bx, top - bh * 0.62 + bh / 2, self.badge, size=fs * 0.72, weight="bold",
                             fill=_on(acc, th), anchor="mm", family=fam, letter_spacing=0.4)
        good, bad, part = th.accent(2), th.accent(3), th.accent(1)
        with dwg.group("cells"):
            for j, c in enumerate(self.columns):
                x0, x1 = xs[j] + self.pad_x, xs[j + 1] - self.pad_x
                a = aligns[j]
                x = {"left": x0, "center": (x0 + x1) / 2, "right": x1}[a]
                anc = {"left": "l", "center": "m", "right": "r"}[a] + "m"
                hl = self.highlight == j
                dwg.text(x, top + self.head_h / 2, c.upper(), size=fs * 0.82, weight="bold", anchor=anc,
                         letter_spacing=0.8, fill=acc if hl else _mix(th.ink, th.muted, 0.4), family=fam)
            y = top + self.head_h
            for i, h in enumerate(self.row_h):
                cy = y + h / 2
                for j in range(len(self.columns)):
                    x0, x1 = xs[j] + self.pad_x, xs[j + 1] - self.pad_x
                    a = aligns[j]
                    k = self.kinds[i][j]
                    x = {"left": x0, "center": (x0 + x1) / 2, "right": x1}[a]
                    if k in ("yes", "no", "part"):
                        r = fs * 0.62
                        mx = x if a == "center" else (x0 + r if a == "left" else x1 - r)
                        col = {"yes": good, "no": bad, "part": part}[k]
                        if k == "yes":
                            dwg.circle(mx, cy, r, fill=col)
                            dwg.path(Path().move_to(mx - r * 0.45, cy + r * 0.02).line_to(mx - r * 0.12, cy + r * 0.34)
                                     .line_to(mx + r * 0.48, cy - r * 0.32), stroke=_on(col, th), stroke_width=r * 0.24,
                                     cap="round", join="round", fill=None)
                        elif k == "no":
                            dwg.circle(mx, cy, r, fill=_mix(col, th.surface, 0.18 if not th.dark else 0.28))
                            q = r * 0.34
                            for s_ in (1, -1):
                                dwg.line(mx - q, cy - s_ * q, mx + q, cy + s_ * q, stroke=col, stroke_width=r * 0.22,
                                         cap="round")
                        else:
                            dwg.circle(mx, cy, r, fill=_mix(col, th.surface, 0.22), stroke=col, stroke_width=1.4)
                            dwg.pie(mx, cy, r, 90, 270, fill=col)
                    elif k == "none":
                        dwg.line(x - fs * 0.4 if a == "center" else (x0 if a == "left" else x1 - fs * 0.8), cy,
                                 x + fs * 0.4 if a == "center" else (x0 + fs * 0.8 if a == "left" else x1), cy,
                                 stroke=_mix(th.muted, th.surface, 0.55), stroke_width=1.6, cap="round")
                    else:
                        anc = {"left": "l", "center": "m", "right": "r"}[a] + "m"
                        bold = (j == 0 and self.row_labels) or self.highlight == j
                        dwg.text(x, cy, self.cells[i][j], size=fs, anchor=anc, align=a, family=fam,
                                 weight="bold" if bold else "normal",
                                 fill=th.ink if (j == 0 or k == "num" or bold) else _mix(th.ink, th.muted, 0.25))
                y += h


# --------------------------------------------------------------------------------------------- annotations

import weakref as _weakref

_PLACED = _weakref.WeakKeyDictionary()          # target → label boxes already placed by the one-shot helpers


def _target_size(target):
    if hasattr(target, "w") and hasattr(target, "h") and hasattr(target, "paint"):
        return float(target.w), float(target.h)
    d = target.drawing if isinstance(target, _v.Group) else target
    return float(d.width), float(d.height)


def _isotonic(d, w=None):
    """Pool-adjacent-violators: non-decreasing fit to ``d`` (least squares)."""
    w = [1.0] * len(d) if w is None else list(w)
    blocks = []
    for i, v in enumerate(d):
        blocks.append([v * w[i], w[i], 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            a = blocks.pop()
            blocks[-1][0] += a[0]
            blocks[-1][1] += a[1]
            blocks[-1][2] += a[2]
    out = []
    for s, ww, n in blocks:
        out += [s / ww] * n
    return out


def _stack(desired, sizes, gap, lo, hi):
    """1-D non-overlapping positions (centres) closest to ``desired`` inside [lo, hi]."""
    n = len(desired)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: desired[i])
    off = [0.0]
    for a, b in zip(order, order[1:]):
        off.append(off[-1] + (sizes[a] + sizes[b]) / 2 + gap)
    z = _isotonic([desired[i] - o for i, o in zip(order, off)])
    pos = [zz + o for zz, o in zip(z, off)]
    first_lo = lo + sizes[order[0]] / 2
    last_hi = hi - sizes[order[-1]] / 2
    if pos[-1] > last_hi:
        sh = pos[-1] - last_hi
        pos = [p - sh for p in pos]
    if pos[0] < first_lo:
        sh = first_lo - pos[0]
        pos = [p + sh for p in pos]
    out = [0.0] * n
    for i, p in zip(order, pos):
        out[i] = p
    return out


class Annotator:
    """Collision-free labels for a figure on a Canvas (pixels or 0~1 fractions) or a vector Drawing (units).

    ``ann = Annotator(cv, theme="light")``; ``ann.callout((0.42, 0.31), "Left atrium")``; ...;
    ``ann.draw(log=log)``. Callouts with ``side="auto"`` stack in label columns left and right of the figure,
    sorted by height so leader lines never cross. Sizes scale with the canvas height (1080p units).

    Also: ``marker(point, text)`` (numbered badge + legend), ``brace(p0, p1, text)``, ``dimension(p0, p1, text)``,
    ``scale_bar(xy, length, label)``, ``label(point, text)`` (plain plate at a point).
    """

    def __init__(self, target, theme="light", font_size=24, color=None, plate=True, columns=None, lead=None,
                 legend="right", legend_title=None, max_width=340, bounds=None, figure=None, leader="dot",
                 leader_color=None, leader_width=1.6):
        _check_choice("Annotator", "leader", leader, list(_LEADERS))
        self.leader, self.leader_color, self.leader_width = leader, leader_color, float(leader_width)
        self.target, self.th = target, _theme(theme)
        self.figure = figure
        self.W, self.H = _target_size(target)
        self.S = self.H / 1080.0
        self.fs, self.color, self.plate, self.columns = font_size, color, plate, columns
        self.lead = lead
        self.legend_pos, self.legend_title, self.max_width = legend, legend_title, max_width
        self.bounds = bounds or (0, 0, self.W, self.H)
        self._callouts, self._markers, self._items = [], [], []

    # ---- coordinates
    def _p(self, p):
        x, y = float(p[0]), float(p[1])
        if 0 <= x <= 1 and 0 <= y <= 1:
            return x * self.W, y * self.H
        return x, y

    def _col(self, c, default=None):
        th = self.th
        if c is None and self.color is None:
            return default if default is not None else th.ink
        return _accent(th, c if c is not None else self.color)

    # ---- API
    def callout(self, point, text, side="auto", color=None, dot=True, leader=None, leader_color=None,
                leader_width=None):
        """Label ``text`` with a leader line to ``point``. ``side``: auto / left / right / top / bottom.
        ``leader``: "dot" (default) / "arrow" / "none" for the leader end, ``leader_color`` and
        ``leader_width`` (px at 1080p) style the line; each defaults to the Annotator's own setting."""
        if leader is not None:
            _check_choice("callout", "leader", leader, list(_LEADERS))
        elif not dot:
            leader = "none"
        self._callouts.append(_N(p=self._p(point), text=str(text), side=side, color=color, dot=dot, leader=leader,
                                 leader_color=leader_color, leader_width=leader_width))
        return self

    def marker(self, point, text=None, color=None):
        """Numbered badge at ``point``; ``text`` goes into the legend. Returns the number."""
        self._markers.append(_N(p=self._p(point), text=text, color=color, n=len(self._markers) + 1))
        return len(self._markers)

    def label(self, point, text, anchor="mm", color=None, size=None):
        """A text plate centred (``anchor``) on ``point``."""
        self._items.append(("label", self._p(point), text, anchor, color, size))
        return self

    def brace(self, p0, p1, text=None, side="auto", depth=None, color=None):
        """Curly brace spanning ``p0``–``p1`` with ``text`` at its tip. ``side``: left / right / above / below."""
        self._items.append(("brace", self._p(p0), self._p(p1), text, side, depth, color))
        return self

    def dimension(self, p0, p1, text=None, offset=0.0, scale=None, unit="", color=None):
        """Measurement arrow between two points with extension lines (``offset`` px/units sideways) and a
        label (``text``, or the length × ``scale`` + ``unit``)."""
        self._items.append(("dim", self._p(p0), self._p(p1), text, offset, scale, unit, color))
        return self

    def scale_bar(self, xy, length, label, segments=4, color=None):
        """Alternating scale bar starting at ``xy`` (left end), ``length`` px/units long."""
        self._items.append(("scale", self._p(xy), length, label, segments, color))
        return self

    # ---- output
    def draw(self, log=None, name="annotations"):
        """Paint everything (Canvas) or add it to the Drawing. Returns the label boxes."""
        th = self.th
        if isinstance(self.target, (Drawing, _v.Group)):
            dwg = self.target
            with dwg.group(name) as g:
                boxes = self._build(g)
            return boxes
        dwg = Drawing(self.W, self.H, theme=th, background=False, name=name)
        with dwg.group(name) as g:
            boxes = self._build(g)
        dwg.draw(self.target, log=log)
        return boxes

    def _figure_box(self):
        """Bounding box of what is already drawn (canvas pixels that differ from the background, or the
        drawing's items) so label columns sit outside the figure. ``figure=`` overrides."""
        if getattr(self, "figure", None) is not None:
            f = self.figure
            return self._p(f[:2]) + self._p(f[2:])
        t = self.target
        try:
            if isinstance(t, (Drawing, _v.Group)):
                return t.root.bbox() if isinstance(t, Drawing) else t.bbox()
            a = np.asarray(t.buf, np.float32)[::3, ::3, :3]
        except Exception:
            return None
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        bg = np.median(border, axis=0)
        diff = np.abs(a - bg).max(axis=2) > 0.06
        ys, xs = np.nonzero(diff)
        if len(xs) == 0:
            return None
        k = 3.0
        return (xs.min() * k, ys.min() * k, (xs.max() + 1) * k, (ys.max() + 1) * k)

    def _plate(self, g, cx, cy, text, anchor="mm", color=None, size=None, max_w=None):
        th = self.th
        S = self.S
        fam = _fam(th)
        fs = (size or self.fs) * S
        txt = _wrap(text, fs, (max_w or self.max_width) * S, "normal", fam)
        w, h = _measure(txt, fs, "normal", fam)
        pw, ph = w + 20 * S, h + 12 * S
        a = _v._anchor(anchor)
        x0 = cx - {"l": 0, "m": 0.5, "r": 1}[a[0]] * pw
        y0 = cy - {"t": 0, "a": 0, "m": 0.5, "s": 0.5, "b": 1, "d": 1}[a[1]] * ph
        if self.plate:
            with g.group(shadow=dict(dx=0, dy=2 * S, blur=6 * S, color=th.shadow["color"],
                                     opacity=th.shadow["opacity"] * 0.9)) as pg:
                pg.rect(x0, y0, pw, ph, r=min(8 * S, ph / 2), fill=th.surface, fill_opacity=0.96,
                        stroke=_mix(color, th.surface, 0.5) if color else th.grid, stroke_width=1.2 * S)
            g.text(x0 + pw / 2, y0 + ph / 2, txt, size=fs, anchor="mm", align="center", fill=th.ink, family=fam)
        else:
            g.text(x0 + pw / 2, y0 + ph / 2, txt, size=fs, anchor="mm", align="center", fill=th.ink, family=fam,
                   halo=th.background, halo_width=fs * 0.18)
        return (x0, y0, x0 + pw, y0 + ph)

    def _build(self, g):
        th = self.th
        S = self.S
        fam = _fam(th)
        boxes = []
        bx0, by0, bx1, by1 = self.bounds
        # ---- callouts
        cs = self._callouts
        if cs:
            fs = self.fs * S
            for c in cs:
                c.txt = _wrap(c.text, fs, self.max_width * S, "normal", fam)
                w, h = _measure(c.txt, fs, "normal", fam)
                c.w, c.h = w + 20 * S, h + 12 * S
                if c.side == "auto":
                    c.side = "left" if c.p[0] < (bx0 + bx1) / 2 else "right"
            groups = {}
            for c in cs:
                groups.setdefault(c.side, []).append(c)
            m = 16 * S
            lead = (self.lead if self.lead is not None else 44) * S
            fb = self._figure_box()
            for side, lst in groups.items():
                if side in ("left", "right"):
                    if self.columns:
                        col = self.columns[0] if side == "left" else self.columns[1]
                        col = self._p((col, 0.5))[0] if col <= 1 else col
                    elif side == "left":
                        col = min([c.p[0] for c in lst] + ([fb[0]] if fb else [])) - lead
                        col = max(col, bx0 + m + max(c.w for c in lst))
                    else:
                        col = max([c.p[0] for c in lst] + ([fb[2]] if fb else [])) + lead
                        col = min(col, bx1 - m - max(c.w for c in lst))
                    ys = _stack([c.p[1] for c in lst], [c.h for c in lst], 10 * S, by0 + m, by1 - m)
                    for c, y in zip(lst, ys):
                        if side == "left":
                            c.box = (col - c.w, y - c.h / 2, col, y + c.h / 2)
                            c.anchor = (col, y)
                        else:
                            c.box = (col, y - c.h / 2, col + c.w, y + c.h / 2)
                            c.anchor = (col, y)
                else:
                    if self.columns and len(self.columns) > 2:
                        row = self.columns[2] if side == "top" else self.columns[3]
                    elif side == "top":
                        row = min([c.p[1] for c in lst] + ([fb[1]] if fb else [])) - lead
                        row = max(row, by0 + m + max(c.h for c in lst))
                    else:
                        row = max([c.p[1] for c in lst] + ([fb[3]] if fb else [])) + lead
                        row = min(row, by1 - m - max(c.h for c in lst))
                    xs = _stack([c.p[0] for c in lst], [c.w for c in lst], 12 * S, bx0 + m, bx1 - m)
                    for c, x in zip(lst, xs):
                        if side == "top":
                            c.box = (x - c.w / 2, row - c.h, x + c.w / 2, row)
                        else:
                            c.box = (x - c.w / 2, row, x + c.w / 2, row + c.h)
                        c.anchor = (x, row)
            with g.group("leaders") as lg:
                for c in cs:
                    col = self._col(c.color, _mix(th.ink, th.background, 0.8))
                    px, py = c.p
                    ax, ay = c.anchor
                    if c.side in ("left", "right"):
                        s = 1 if c.side == "left" else -1
                        knee = (ax + s * 18 * S, ay)
                        pts = [(px, py), knee, (ax, ay)]
                    else:
                        s = 1 if c.side == "top" else -1
                        knee = (ax, ay + s * 18 * S)
                        pts = [(px, py), knee, (ax, ay)]
                    if c.leader_color is not None:
                        col = _accent(th, c.leader_color)
                    elif self.leader_color is not None and c.color is None:
                        col = _accent(th, self.leader_color)
                    lw = (c.leader_width if c.leader_width is not None else self.leader_width) * S
                    style = c.leader or self.leader
                    if style == "arrow":                 # the line stops at the arrowhead's base
                        d = np.array([px - pts[1][0], py - pts[1][1]], float)
                        n = float(np.hypot(*d))
                        u = d / n if n > 1e-6 else np.array([1.0, 0.0])
                        L, Wd = max(11 * S, lw * 5.5), max(5 * S, lw * 2.6)
                        L = min(L, 0.9 * n) if n > 1 else L
                        base = np.array([px, py]) - u * L
                        nrm = np.array([-u[1], u[0]])
                        lg.polyline([(float(base[0]), float(base[1])), knee, (ax, ay)],
                                    stroke=col, stroke_width=lw, join="round", cap="round")
                        lg.polygon([(px, py), tuple(base + nrm * Wd), tuple(base - nrm * Wd)], fill=col,
                                   stroke=col, stroke_width=lw * 0.4, join="round")
                    else:
                        lg.polyline(pts, stroke=col, stroke_width=lw, join="round", cap="round")
                        if style == "dot":
                            lg.circle(px, py, max(4.6 * S, lw * 2.9), fill=col, stroke=th.surface, stroke_width=lw)
            with g.group("labels") as tg:
                for c in cs:
                    b = c.box
                    col = self._col(c.color, None) if c.color or self.color else None
                    boxes.append(self._plate(tg, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, c.text, "mm", col))
        # ---- markers + legend
        if self._markers:
            r = 15 * S
            with g.group("markers") as mg:
                for mk in self._markers:
                    col = self._col(mk.color, th.accent(0))
                    x, y = mk.p
                    with mg.group(shadow=dict(dx=0, dy=1.5 * S, blur=4 * S, color=th.shadow["color"],
                                              opacity=0.35)) as sg:
                        sg.circle(x, y, r, fill=col, stroke=th.surface, stroke_width=2.2 * S)
                    mg.text(x, y, str(mk.n), size=r * 1.1, weight="bold", fill=_on(col, th), anchor="mm", family=fam)
            items = [mk for mk in self._markers if mk.text]
            if items and self.legend_pos:
                boxes.append(self._legend(g, items, r))
        # ---- braces, dimensions, labels, scale bars
        for it in self._items:
            kind = it[0]
            if kind == "label":
                _, p, text, anchor, color, size = it
                boxes.append(self._plate(g, p[0], p[1], text, anchor, self._col(color) if color else None, size))
            elif kind == "brace":
                boxes.append(self._brace(g, *it[1:]))
            elif kind == "dim":
                boxes.append(self._dim(g, *it[1:]))
            elif kind == "scale":
                boxes.append(self._scale(g, *it[1:]))
        return boxes

    def _legend(self, g, items, r):
        th = self.th
        S = self.S
        fam = _fam(th)
        fs = self.fs * 0.92 * S
        rows = []
        for mk in items:
            txt = _wrap(mk.text, fs, self.max_width * S, "normal", fam)
            w, h = _measure(txt, fs, "normal", fam)
            rows.append((mk, txt, w, max(h, 2 * r * 0.85)))
        pad = 18 * S
        tw = 0
        title_h = 0
        if self.legend_title:
            tw, title_h = _measure(self.legend_title, fs * 1.05, "bold", fam)
            title_h += 12 * S
        W = max([2 * r * 0.85 + 12 * S + w for _, _, w, _ in rows] + [tw]) + 2 * pad
        H = sum(h for *_, h in rows) + 10 * S * (len(rows) - 1) + 2 * pad + title_h
        pos = self.legend_pos
        m = 24 * S
        if isinstance(pos, (tuple, list)):
            x0, y0 = self._p(pos)
        elif pos == "left":
            x0, y0 = m, (self.H - H) / 2
        elif pos in ("bottom", "bottom-right"):
            x0, y0 = self.W - W - m, self.H - H - m
        elif pos == "bottom-left":
            x0, y0 = m, self.H - H - m
        elif pos == "top-left":
            x0, y0 = m, m
        elif pos == "top-right":
            x0, y0 = self.W - W - m, m
        else:
            x0, y0 = self.W - W - m, (self.H - H) / 2
        with g.group("legend") as lg:
            with lg.group(shadow=dict(dx=0, dy=3 * S, blur=10 * S, color=th.shadow["color"],
                                      opacity=th.shadow["opacity"])) as sg:
                sg.rect(x0, y0, W, H, r=12 * S, fill=th.surface, stroke=th.grid, stroke_width=1.2 * S)
            y = y0 + pad
            if self.legend_title:
                lg.text(x0 + pad, y, self.legend_title, size=fs * 1.05, weight="bold", anchor="lt", family=fam)
                y += title_h
            rr = r * 0.85
            for mk, txt, w, h in rows:
                col = self._col(mk.color, th.accent(0))
                lg.circle(x0 + pad + rr, y + h / 2, rr, fill=col)
                lg.text(x0 + pad + rr, y + h / 2, str(mk.n), size=rr * 1.1, weight="bold", fill=_on(col, th),
                        anchor="mm", family=fam)
                lg.text(x0 + pad + 2 * rr + 12 * S, y + h / 2, txt, size=fs, anchor="lm", family=fam)
                y += h + 10 * S
        return (x0, y0, x0 + W, y0 + H)

    def _brace(self, g, p0, p1, text, side, depth, color):
        th = self.th
        S = self.S
        col = self._col(color, _mix(th.ink, th.background, 0.85))
        ux, uy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(ux, uy)
        if L < 1e-6:
            return (p0[0], p0[1], p0[0], p0[1])
        ux, uy = ux / L, uy / L
        nx, ny = -uy, ux
        want = {"right": (1, 0), "left": (-1, 0), "below": (0, 1), "bottom": (0, 1), "above": (0, -1),
                "top": (0, -1)}.get(side)
        if want is None:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            if abs(ux) > abs(uy):
                want = (0, 1) if my > self.H / 2 else (0, -1)
            else:
                want = (1, 0) if mx > self.W / 2 else (-1, 0)
        if nx * want[0] + ny * want[1] < 0:
            nx, ny = -nx, -ny
        d = (depth if depth is not None else min(26.0, L * 0.18 / max(S, 1e-9))) * S
        rr = min(d / 2, L / 4)

        def P(s, h):
            return (p0[0] + ux * s + nx * h, p0[1] + uy * s + ny * h)

        path = Path().move_to(*P(0, 0))
        path.quad_to(*P(0, d / 2), *P(rr, d / 2))
        path.line_to(*P(L / 2 - rr, d / 2))
        path.quad_to(*P(L / 2, d / 2), *P(L / 2, d))
        path.quad_to(*P(L / 2, d / 2), *P(L / 2 + rr, d / 2))
        path.line_to(*P(L - rr, d / 2))
        path.quad_to(*P(L, d / 2), *P(L, 0))
        g.path(path, stroke=col, stroke_width=2.0 * S, fill=None, cap="round", join="round")
        if not text:
            return (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
        tip = P(L / 2, d + 10 * S)
        if abs(nx) > abs(ny):
            anchor = "lm" if nx > 0 else "rm"
        else:
            anchor = "mt" if ny > 0 else "mb"
        return self._plate(g, tip[0], tip[1], text, anchor, None)

    def _dim(self, g, p0, p1, text, offset, scale, unit, color):
        th = self.th
        S = self.S
        fam = _fam(th)
        col = self._col(color, _mix(th.ink, th.background, 0.85))
        ux, uy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(ux, uy)
        if L < 1e-6:
            return (p0[0], p0[1], p0[0], p0[1])
        ux, uy = ux / L, uy / L
        nx, ny = -uy, ux
        off = float(offset) if (abs(offset) > 1 or offset == 0) else float(offset) * self.H
        a = (p0[0] + nx * off, p0[1] + ny * off)
        b = (p1[0] + nx * off, p1[1] + ny * off)
        sw = 1.6 * S
        if abs(off) > 0.5:
            s = 1 if off > 0 else -1
            for p, q in ((p0, a), (p1, b)):
                g.line(p[0] + nx * s * 4 * S, p[1] + ny * s * 4 * S, q[0] + nx * s * 8 * S, q[1] + ny * s * 8 * S,
                       stroke=_mix(col, th.background, 0.6), stroke_width=1.1 * S)
        hs = 13 * S
        tr = _arrow_trim("arrow", hs)
        g.line(a[0] + ux * tr, a[1] + uy * tr, b[0] - ux * tr, b[1] - uy * tr, stroke=col, stroke_width=sw)
        _arrowhead(g, b, (ux, uy), hs, col)
        _arrowhead(g, a, (-ux, -uy), hs, col)
        if text is None:
            v = L * (scale if scale else 1.0)
            text = (f"{v:.3g}" if scale else f"{v:.0f}") + (f" {unit}" if unit else "")
        ang = math.degrees(math.atan2(uy, ux))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        fs = self.fs * 0.9 * S
        w, h = _measure(text, fs, "bold", fam)
        g.text(mx, my, text, size=fs, weight="bold", anchor="mm", rotate=ang, fill=th.ink, family=fam,
               halo=th.background, halo_width=fs * 0.35)
        return (mx - w / 2, my - h / 2, mx + w / 2, my + h / 2)

    def _scale(self, g, xy, length, label, segments, color):
        th = self.th
        S = self.S
        fam = _fam(th)
        col = self._col(color, th.ink)
        x, y = xy
        h = 9 * S
        n = max(int(segments), 1)
        seg = length / n
        g.rect(x, y - h, length, h, fill=th.surface, stroke=col, stroke_width=1.4 * S)
        for i in range(0, n, 2):
            g.rect(x + i * seg, y - h, seg, h, fill=col)
        fs = self.fs * 0.8 * S
        g.text(x, y + 6 * S, "0", size=fs, anchor="mt", fill=th.ink, family=fam, halo=th.background)
        g.text(x + length, y + 6 * S, str(label), size=fs, anchor="mt", fill=th.ink, family=fam, weight="bold",
               halo=th.background)
        return (x, y - h, x + length, y + 6 * S + fs * 1.2)


_LEADERS = ("dot", "arrow", "none")
_ANN_OPTS = ("font_size", "color", "plate", "columns", "lead", "legend", "legend_title", "max_width", "bounds",
             "leader", "leader_color", "leader_width")


def _one_shot(target, theme, kw, owner="annotation", extra=()):
    kw = _reject_unknown(owner, kw, _ANN_OPTS + tuple(extra))
    return Annotator(target, theme=theme, **kw)


def callout(target, point, text, side="right", theme="light", log=None, distance=None, **kw):
    """One label with a leader line, drawn right away. ``side``: right / left / top / bottom / auto.
    Later calls on the same canvas slide away from labels already placed."""
    a = _one_shot(target, theme, kw, "callout")
    S = a.S
    p = a._p(point)
    if side == "auto":
        side = "left" if p[0] < a.W / 2 else "right"
    fam = _fam(a.th)
    fs = a.fs * S
    txt = _wrap(str(text), fs, a.max_width * S, "normal", fam)
    w, h = _measure(txt, fs, "normal", fam)
    w, h = w + 20 * S, h + 12 * S
    dist = (distance if distance is not None else 90) * S
    placed = _PLACED.setdefault(target, []) if _weakable(target) else []
    dx, dy = {"right": (1, 0), "left": (-1, 0), "top": (0, -1), "bottom": (0, 1)}[side]
    cx, cy = p[0] + dx * (dist + w / 2), p[1] + dy * (dist + h / 2)
    step = (h + 8 * S) if dx else (w + 8 * S)
    best = None
    for k in range(0, 24):
        off = ((k + 1) // 2) * step * (1 if k % 2 else -1)
        bx, by = (cx, cy + off) if dx else (cx + off, cy)
        box = (bx - w / 2, by - h / 2, bx + w / 2, by + h / 2)
        if box[0] < 0 or box[1] < 0 or box[2] > a.W or box[3] > a.H:
            continue
        if not any(_rect_hit(_inflate(box, 4 * S), q) for q in placed):
            best = (bx, by)
            break
    if best is None:
        best = (cx, cy)
    bx, by = best
    a.callout(point, text, side=side, color=kw.get("color"))
    c = a._callouts[-1]
    a.columns = None
    # place exactly: column at the chosen box edge
    if dx:
        a.columns = (bx + w / 2, bx - w / 2)
    else:
        a.columns = (0, 0, by + h / 2, by - h / 2)
    a.bounds = (bx - w / 2 - 1, by - h / 2 - 1, bx + w / 2 + 1, by + h / 2 + 1) if not dx else \
        (0, by - h / 2 - 1, a.W, by + h / 2 + 1)
    if not dx:
        a.bounds = (bx - w / 2 - 1, 0, bx + w / 2 + 1, a.H)
    c.side = side
    boxes = a.draw(log=log, name=f"callout {str(text)[:24]}")
    if _weakable(target):
        placed.extend(boxes)
    return boxes[0] if boxes else None


def _weakable(t):
    try:
        _weakref.ref(t)
        return True
    except TypeError:
        return False


def callouts(target, items, side="auto", theme="light", log=None, **kw):
    """Many labels at once, stacked in columns without overlaps. ``items``: [(point, text), ...] or
    [(point, text, side), ...]. Returns the label boxes."""
    a = _one_shot(target, theme, kw, "callouts")
    for it in items:
        if isinstance(it, dict):
            a.callout(it["point"], it["text"], it.get("side", side), it.get("color"), leader=it.get("leader"),
                      leader_color=it.get("leader_color"), leader_width=it.get("leader_width"))
        else:
            a.callout(it[0], it[1], it[2] if len(it) > 2 else side)
    boxes = a.draw(log=log, name="callouts")
    if _weakable(target):
        _PLACED.setdefault(target, []).extend(boxes)
    return boxes


def markers(target, items, legend="right", theme="light", log=None, legend_title=None, **kw):
    """Numbered badges at points plus a legend. ``items``: [(point, text), ...]; ``legend``: right / left /
    top-left / top-right / bottom-left / bottom-right / (x, y) / None."""
    kw = _reject_unknown("markers", kw, ("font_size", "color", "max_width"))
    a = Annotator(target, theme=theme, legend=legend, legend_title=legend_title, **kw)
    for it in items:
        a.marker(it[0], it[1] if len(it) > 1 else None, it[2] if len(it) > 2 else None)
    return a.draw(log=log, name="markers")


def brace(target, p0, p1, text=None, side="auto", theme="light", log=None, **kw):
    """Curly brace from ``p0`` to ``p1`` with a label at its tip."""
    a = _one_shot(target, theme, kw, "brace", ("depth",))
    a.brace(p0, p1, text, side, kw.get("depth"), kw.get("color"))
    return a.draw(log=log, name=f"brace {text or ''}".strip())


def dimension(target, p0, p1, text=None, offset=0.0, theme="light", log=None, scale=None, unit="", **kw):
    """Double-headed measurement arrow between two points with a rotated label."""
    a = _one_shot(target, theme, kw, "dimension")
    a.dimension(p0, p1, text, offset, scale, unit, kw.get("color"))
    return a.draw(log=log, name="dimension")


def scale_bar(target, xy, length, label, segments=4, theme="light", log=None, **kw):
    """Scale bar (alternating segments) from ``xy`` to the right, ``length`` px long."""
    a = _one_shot(target, theme, kw, "scale_bar")
    a.scale_bar(xy, length, label, segments, kw.get("color"))
    return a.draw(log=log, name="scale bar")


# unknown options raise a helpful TypeError; colour / tags / w ... spellings are accepted
for _c in (_Figure, Diagram, Tree, Sequence, Timeline, Table, Annotator, _GroupHandle):
    _strict_methods(_c)

# ---- scene manifest: nodes, groups and annotations record what they drew (pixelcanvas.manifest) ----
from . import manifest as _manifest  # noqa: E402

def _after_figure(cap, args, res):
    """Manifest entries for a figure; diagram nodes also carry label, group and their outgoing edges, and
    containers their members (``meta`` of each OBJECTS entry)."""
    items = _manifest.after_figure(cap, args, res)
    fig = args.get("self")
    if not items or not isinstance(fig, Diagram):
        return items
    try:
        nodes = fig.nodes
        outs = {}
        for e in fig.edges:
            outs.setdefault(e["source"], []).append({"target": e["target"], "label": e["label"],
                                                     "style": e["style"], "arrow": e["arrow"]})
        for it in items:
            m = dict(it.get("meta") or {})
            k = m.get("id")
            if it.get("kind") == "node" and k in getattr(fig, "_buses", {}):
                b = fig._buses[k]
                lab = " ".join(str(b.label).split())
                it["alias"] = ["bus", "message bus", "event bus", "pub sub", lab.replace("/", " ")] + (
                    [lab.replace("/", " ") + " bus"] if "bus" not in lab.lower() else [])
                m.update(label=lab, shape="bus", members=list(b.members))
            elif it.get("kind") == "node" and k in nodes:
                n = nodes[k]
                m.update(label=" ".join(str(n["label"]).split()), group=n["group"], edges=outs.get(k, []))
                if n.get("members"):
                    m["members"] = list(n["members"])
            elif it.get("kind") == "group" and k in fig.groups:
                m["members"] = [x for x, n in nodes.items() if n["group"] == k]
            it["meta"] = m
    except Exception:                                    # noqa: BLE001 - recording never breaks a drawing
        pass
    return items


_manifest.wrap_method(_Figure, "draw", "node", after=_after_figure)
_manifest.wrap_method(Annotator, "draw", "annotation",
                      cv_of=lambda a: a["self"].target if hasattr(a["self"].target, "buf") else None)
