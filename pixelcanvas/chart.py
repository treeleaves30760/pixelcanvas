# -*- coding: utf-8 -*-
"""Charts on the vector core: line, bar, area, scatter, histogram, heatmap, pie / donut, radar, sparkline, KPI tiles.

A chart is drawn into a box of a Canvas through ``pixelcanvas.vector`` (anti-aliased, one layer per chart object)
and exports as SVG::

    from pixelcanvas import Canvas
    from pixelcanvas import chart
    cv = Canvas(1600, 900)
    ax = chart.axes(cv, theme="light", title="Monthly revenue", yfmt="$k")
    ax.line(["2026-01", "2026-02", "2026-03", "2026-04"], [120e3, 135e3, 128e3, 162e3], label="2026")
    ax.draw()                                   # paints onto cv; ax.save_svg("revenue.svg")

Several charts on one canvas share a ``Figure``: ``fig = chart.figure(cv, theme="dark")``,
``fig.axes(box)`` / ``fig.kpi(box, ...)`` / ``fig.header(...)``, then ``fig.draw(log=log)`` and ``fig.save_svg(path)``.
Boxes are (u0, v0, u1, v1) fractions of the canvas. Sizes (fonts, line widths) are in units of a 1080 px tall canvas,
so drafts and full-size renders look the same.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from functools import lru_cache

import numpy as np

from . import vector as _v
from .vector import Drawing, THEMES, Theme, linear_gradient

__all__ = ["Figure", "Axes", "KPI", "Header", "figure", "axes", "quick", "grid", "kpi", "sparkline", "header",
           "fmt_number", "nice_ticks", "PALETTE_ORDER", "SIZES", "THEMES"]

PALETTE_ORDER = (0, 1, 2, 4, 5, 6, 7, 3)    # theme accents: blue orange green purple cyan pink olive, red last
SIZES = dict(title=23, subtitle=15, tick=14, label=15, legend=14, value=13, note=14, panel=30, kpi=44)
_MINUS = "−"

# --------------------------------------------------------------------------------------------- colours


from ._args import (check_choice as _check_choice, reject_unknown as _reject_unknown,  # noqa: E402
                    signature_options as _signature_options, strict_methods as _strict_methods,
                    unknown_name as _unknown_name)


def _theme(t):
    if isinstance(t, Theme):
        return t
    if isinstance(t, dict):
        return Theme(t)
    t = t or "light"
    if t not in THEMES:
        raise _unknown_name("theme", t, THEMES)
    return THEMES[t]


def _rgbf(c):
    col = _v._color(c)
    return np.asarray(col[0], np.float64) if col else np.zeros(3)


def _mix(a, b, t):
    """``t`` of colour ``b`` mixed into ``a`` → '#rrggbb'."""
    return _v._hex(_rgbf(a) * (1 - t) + _rgbf(b) * t)


def _lum(c):
    r = _rgbf(c)
    r = np.where(r <= 0.04045, r / 12.92, ((r + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2])


def _ramp(stops, t):
    """Colour at ``t`` (0~1, array ok) on evenly spaced colour ``stops`` → (N, 3) floats."""
    cols = np.stack([_rgbf(c) for c in stops])
    t = np.clip(np.asarray(t, np.float64), 0, 1) * (len(stops) - 1)
    i = np.minimum(np.floor(t).astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    return cols[i] * (1 - f) + cols[i + 1] * f


# --------------------------------------------------------------------------------------------- text metrics


@lru_cache(maxsize=8192)
def _measure(text, size, family, bold, max_width=None):
    k = 3.0
    w, h = _v.measure_text(text, size * k, family, "bold" if bold else "normal",
                           max_width=None if max_width is None else max_width * k)
    return w / k, h / k


def _text(g, x, y, s, **kw):
    return g.text(x, y, str(s), **kw)


# --------------------------------------------------------------------------------------------- numbers


_FMT_RE = re.compile(r"^(?P<pre>[$€£¥₹₩+]*)(?:\.(?P<dec>\d+))?(?P<mode>[k%])?(?P<suf>.*)$", re.S)


def _parse_fmt(fmt):
    m = _FMT_RE.match(fmt or "")
    pre = m.group("pre")
    return dict(sign="+" in pre, pre=pre.replace("+", ""), dec=None if m.group("dec") is None else int(m.group("dec")),
                mode=m.group("mode") or "", suf=m.group("suf"))


def _compact_unit(a):
    for div, s in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return div, s
    return 1.0, ""


def _decimals_for(vals, limit=6):
    vals = [abs(v) for v in vals if np.isfinite(v)]
    for d in range(limit + 1):
        if all(abs(round(v, d) - v) <= 1e-9 * max(1.0, v) for v in vals):
            return d
    return limit


def _half_up(a, dec):
    k = 10.0 ** dec
    return math.floor(abs(a) * k + 0.5 + 1e-9) / k


def _join(v, dec, spec, unit=""):
    s = f"{_half_up(v, dec):,.{dec}f}"
    neg = v < 0 and float(s.replace(",", "")) != 0
    sign = _MINUS if neg else ("+" if spec["sign"] and v > 0 else "")
    tail = unit + ("%" if spec["mode"] == "%" else "") + spec["suf"]
    return sign + spec["pre"] + s + tail


def fmt_number(v, fmt=None):
    """Format one number for display.

    ``fmt``: None (auto: thousands separators, 2 significant decimals for small numbers, k/M/B above 100 000),
    a shorthand ``[+][currency][.N][k|%][suffix]`` — ``"$"`` → $1,250 · ``"$k"`` → $1.3k · ``"%"`` → 12.5%
    (value already in percent) · ``".1%"`` · ``"+%"`` (sign on positives) · ``"k"`` → 1.2M · ``" ms"`` → 12 ms —
    a Python format string (``"{:.1%}"`` multiplies by 100) or a callable."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, str):
        return v
    if callable(fmt):
        return str(fmt(v))
    if isinstance(fmt, str) and "{" in fmt:
        return fmt.format(v)
    spec = _parse_fmt(fmt)
    v = float(v)
    unit, div = "", 1.0
    if spec["mode"] == "k" or (fmt in (None, "", "auto") and abs(v) >= 1e5):
        div, unit = _compact_unit(abs(v))
    x = v / div
    dec = spec["dec"]
    if dec is None:
        a = abs(x)
        if a == 0 or abs(round(x) - x) < 1e-9:
            dec = 0
        elif a >= 100:
            dec = 0
        elif a >= 10:
            dec = 1
        elif a >= 1:
            dec = 1 if unit else 2
        else:
            dec = max(2, -int(math.floor(math.log10(a))) + 1)
        s = f"{_half_up(x, dec):.{dec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        dec = len(s.split(".")[1]) if "." in s else 0
    return _join(x, dec, spec, unit)


def _fmt_ticks(ticks, fmt):
    """Labels for a set of ticks with one shared unit and number of decimals."""
    ticks = [float(t) for t in ticks]
    if callable(fmt) or (isinstance(fmt, str) and "{" in fmt):
        return [fmt_number(t, fmt) for t in ticks]
    spec = _parse_fmt(fmt)
    amax = max([abs(t) for t in ticks] + [0.0])
    div, unit = 1.0, ""
    if spec["mode"] == "k" or (spec["mode"] == "" and spec["dec"] is None and amax >= 1e4):
        div, unit = _compact_unit(amax)
    xs = [t / div for t in ticks]
    dec = spec["dec"] if spec["dec"] is not None else _decimals_for(xs, 4)
    out = []
    for x in xs:
        if abs(x) < 1e-12 * max(1.0, amax / div):
            out.append(spec["pre"] + "0" + ("%" if spec["mode"] == "%" else "") + spec["suf"])
        else:
            out.append(_join(x, dec, spec, unit))
    return out


def _nice_step(raw, up=True):
    raw = max(abs(raw), 1e-300)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m == 2.5 and mag < 10:                     # 25, 250 ... but never 2.5 / 0.25 steps
            continue
        if m * mag >= raw * (1 - 1e-9):
            return m * mag
    return 10 * mag


def nice_ticks(lo, hi, n=5):
    """'Nice' tick values (steps 1, 2, 2.5, 5 x 10^k) covering [lo, hi] with about ``n`` intervals → (ticks, step)."""
    lo, hi = float(lo), float(hi)
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-12 * max(1.0, abs(hi)):
        d = abs(hi) * 0.1 or 1.0
        lo, hi = lo - d, hi + d
    best = None
    for m in (n, n + 1, n + 2):                       # the tightest cover among a few tick counts
        step = _nice_step((hi - lo) / max(int(m), 1))
        t0 = math.floor(lo / step + 1e-9) * step
        t1 = math.ceil(hi / step - 1e-9) * step
        k = int(round((t1 - t0) / step))
        waste = (t1 - t0) / (hi - lo)
        if best is None or (waste < best[0] - 0.08 and k <= max(n + 1, 8)):
            best = (waste, t0, step, k)
    _, t0, step, k = best
    ticks = [round(t0 + i * step, 12) for i in range(k + 1)]
    return ticks, step


def _log_ticks(lo, hi):
    """Log-axis ticks → (major, minor, d0, d1). Few decades get 1-2-5 (or 1-2-3-5) mantissas."""
    lo, hi = max(min(lo, hi), 1e-300), max(lo, hi, 1e-300)
    la, lb = math.log10(lo), math.log10(hi)
    if lb - la < 1e-9:
        la, lb = la - 0.5, lb + 0.5
    dec = lb - la
    a, b = math.floor(la + 1e-9), math.ceil(lb - 1e-9)
    if b <= a:
        b = a + 1
    if dec >= 2.5:
        step = max(1, int(math.ceil((b - a) / 7)))
        d0, d1 = 10.0 ** a, 10.0 ** b
        major = [10.0 ** e for e in range(a, b + 1, step)]
        minor = [m * 10.0 ** e for e in range(a, b) for m in range(2, 10)] if step == 1 else []
        return major, minor, d0, d1
    mant = (1, 2, 3, 5) if dec <= 1.3 else (1, 2, 5)
    cand = sorted(m * 10.0 ** e for e in range(a - 1, b + 1) for m in mant)
    d0 = max([c for c in cand if c <= lo * (1 + 1e-9)] or [10.0 ** a])
    d1 = min([c for c in cand if c >= hi * (1 - 1e-9)] or [10.0 ** b])
    major = [c for c in cand if d0 * (1 - 1e-9) <= c <= d1 * (1 + 1e-9)]
    minor = [m * 10.0 ** e for e in range(a - 1, b + 1) for m in range(2, 10)
             if d0 < m * 10.0 ** e < d1 and all(abs(m * 10.0 ** e - c) > 1e-9 * c for c in major)]
    return major, minor, d0, d1


def _log_labels(ticks, fmt):
    if fmt is not None:
        return [fmt_number(v, fmt) for v in ticks]
    big = max(ticks) >= 1e4 if ticks else False
    out = []
    for v in ticks:
        if v >= 1:
            out.append(fmt_number(v, "k") if big else fmt_number(v))
        else:
            out.append(f"{v:.{max(0, -int(math.floor(math.log10(v) + 1e-9)))}f}".rstrip("0").rstrip(".")
                       if abs(v - round(v, 6)) < 1e-12 else fmt_number(v))
    return out


# --------------------------------------------------------------------------------------------- dates

_DATE_RE = re.compile(r"^(\d{4})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?$")
_MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _parse_date(v):
    """Date-like value → day number (float) or None. Strings 'YYYY', 'YYYY-MM', 'YYYY-MM-DD', date objects."""
    if isinstance(v, _dt.datetime):
        return v.toordinal() + (v.hour * 3600 + v.minute * 60 + v.second) / 86400.0
    if isinstance(v, _dt.date):
        return float(v.toordinal())
    if isinstance(v, np.datetime64):
        return float(v.astype("datetime64[D]").astype(np.int64)) + 719163.0
    if not isinstance(v, str):
        return None
    m = _DATE_RE.match(v.strip())
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)
    try:
        return float(_dt.date(y, mo, d).toordinal())
    except ValueError:
        return None


def _date_labels(cats):
    """Readable labels for category names that are all ISO dates ('2026-03' → 'Mar', year on first / January)."""
    ms = [_DATE_RE.match(c.strip()) for c in cats]
    if not cats or not all(ms) or all(m.group(2) is None for m in ms):
        return list(cats)
    out, prev_y = [], None
    for m in ms:
        y, mo, d = int(m.group(1)), int(m.group(2) or 1), m.group(3)
        lab = _MON[mo - 1] + (f" {int(d)}" if d else "")
        if y != prev_y:
            lab += f"\n{y}"
        prev_y = y
        out.append(lab)
    return out


def _date(n):
    return _dt.date.fromordinal(int(math.floor(n + 1e-6)))


def _add_months(d, k):
    m = d.year * 12 + d.month - 1 + k
    return _dt.date(m // 12, m % 12 + 1, 1)


def _date_ticks(d0, d1, n):
    """Calendar ticks between day numbers d0..d1, about ``n`` of them → [(daynum, label)]."""
    span = max(d1 - d0, 1e-6)
    a, b = _date(d0), _date(d1)
    for unit, k in (("day", 1), ("day", 2), ("day", 7), ("day", 14), ("month", 1), ("month", 2), ("month", 3),
                    ("month", 6), ("year", 1), ("year", 2), ("year", 5), ("year", 10), ("year", 25), ("year", 50)):
        approx = {"day": 1, "month": 30.44, "year": 365.25}[unit] * k
        if span / approx <= max(n, 1) + 0.5:
            break
    out = []
    if unit == "day":
        if k == 7:
            t = a + _dt.timedelta(days=(7 - a.weekday()) % 7)
        else:
            t = a if (d0 - a.toordinal()) < 1e-6 else a + _dt.timedelta(days=1)
        prev_m = None
        while t.toordinal() <= d1 + 1e-6:
            lab = f"{_MON[t.month - 1]} {t.day}" if t.month != prev_m or k >= 7 or t.day == 1 else str(t.day)
            if k >= 7:
                lab = f"{_MON[t.month - 1]} {t.day}"
            prev_m = t.month
            out.append((float(t.toordinal()), lab))
            t += _dt.timedelta(days=k)
    elif unit == "month":
        t = _dt.date(a.year, a.month, 1)
        if t.toordinal() < d0 - 1e-6:
            t = _add_months(t, 1)
        while (t.month - 1) % k:
            t = _add_months(t, 1)
        first = True
        while t.toordinal() <= d1 + 1e-6:
            lab = _MON[t.month - 1]
            if first or t.month == 1:
                lab += f"\n{t.year}"
            first = False
            out.append((float(t.toordinal()), lab))
            t = _add_months(t, k)
    else:
        y = a.year if _dt.date(a.year, 1, 1).toordinal() >= d0 - 1e-6 else a.year + 1
        y = int(math.ceil(y / k) * k)
        while _dt.date(y, 1, 1).toordinal() <= d1 + 1e-6:
            out.append((float(_dt.date(y, 1, 1).toordinal()), str(y)))
            y += k
    return out


# --------------------------------------------------------------------------------------------- scales


class _Scale:
    """Maps domain values to positions. kind: linear / log / date / category (index domain -0.5 .. n-0.5)."""

    def __init__(self, kind, d0, d1, r0=0.0, r1=1.0, cats=None):
        self.kind, self.d0, self.d1, self.r0, self.r1, self.cats = kind, float(d0), float(d1), r0, r1, cats

    def _t(self, v):
        v = np.asarray(v, np.float64)
        if self.kind == "log":
            return (np.log10(np.maximum(v, 1e-300)) - math.log10(self.d0)) / (math.log10(self.d1) - math.log10(self.d0))
        return (v - self.d0) / ((self.d1 - self.d0) or 1.0)

    def __call__(self, v):
        r = self.r0 + self._t(v) * (self.r1 - self.r0)
        return float(r) if np.ndim(r) == 0 else r

    @property
    def step(self):
        """Band width (category scales)."""
        return abs(self.r1 - self.r0) / max(len(self.cats or [1]), 1)


# --------------------------------------------------------------------------------------------- layout helpers


def _aspect(obj):
    if obj is None:
        return 16 / 9
    if isinstance(obj, Figure):
        return obj.width / obj.height
    if hasattr(obj, "w") and hasattr(obj, "h"):
        return obj.w / obj.h
    if isinstance(obj, (tuple, list)):
        return obj[0] / obj[1]
    return 16 / 9


def grid(cv=None, rows=1, cols=1, box=(0, 0, 1, 1), gap=0.02, widths=None, heights=None):
    """Split ``box`` (canvas fractions) into ``rows`` x ``cols`` boxes, row-major → list of (u0, v0, u1, v1).

    ``gap``: space between cells as a fraction of the canvas height (same pixels across and down; give ``cv`` or
    a Figure for the aspect ratio). ``widths`` / ``heights``: relative sizes, e.g. ``widths=(2, 1)``."""
    asp = _aspect(cv)
    u0, v0, u1, v1 = box
    gx, gy = gap / asp, gap
    widths = list(widths or [1] * cols)
    heights = list(heights or [1] * rows)
    cw = (u1 - u0 - gx * (cols - 1)) / sum(widths)
    ch = (v1 - v0 - gy * (rows - 1)) / sum(heights)
    out = []
    y = v0
    for r in range(rows):
        x = u0
        for c in range(cols):
            out.append((x, y, x + cw * widths[c], y + ch * heights[r]))
            x += cw * widths[c] + gx
        y += ch * heights[r] + gy
    return out


def _monotone(px, py):
    """Fritsch–Carlson monotone cubic through (px, py) → list of bezier segments (c1x, c1y, c2x, c2y, x, y)."""
    n = len(px)
    h = np.diff(px)
    h = np.where(np.abs(h) < 1e-9, 1e-9, h)
    d = np.diff(py) / h
    m = np.zeros(n)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    segs = []
    for i in range(n - 1):
        segs.append((px[i] + h[i] / 3, py[i] + m[i] * h[i] / 3, px[i + 1] - h[i] / 3, py[i + 1] - m[i + 1] * h[i] / 3,
                     px[i + 1], py[i + 1]))
    return segs


def _curve(g, px, py, smooth, close_to=None, **style):
    """Polyline or monotone curve through the points; ``close_to`` = baseline y (area) or list of (x, y) back."""
    p = g.path(**style)
    p.move_to(px[0], py[0])
    if smooth and len(px) > 2:
        for s in _monotone(np.asarray(px, float), np.asarray(py, float)):
            p.cubic_to(*s)
    else:
        for x, y in zip(px[1:], py[1:]):
            p.line_to(x, y)
    if close_to is not None:
        if np.ndim(close_to) == 0:
            p.line_to(px[-1], close_to).line_to(px[0], close_to)
        else:
            bx, by = close_to
            if smooth and len(bx) > 2:
                p.line_to(bx[0], by[0])
                for s in _monotone(np.asarray(bx, float), np.asarray(by, float)):
                    p.cubic_to(*s)
            else:
                for x, y in zip(bx, by):
                    p.line_to(x, y)
        p.close()
    return p


def _triangle(g, cx, cy, s, up, **style):
    if up:
        pts = [(cx - s, cy + s * 0.6), (cx + s, cy + s * 0.6), (cx, cy - s * 0.75)]
    else:
        pts = [(cx - s, cy - s * 0.6), (cx + s, cy - s * 0.6), (cx, cy + s * 0.75)]
    return g.polygon(pts, **style)


class _Occ:
    """Coarse occupancy grid over the plot (plot-local units) for legend / callout placement."""

    def __init__(self, x0, y0, x1, y1, cell=6.0):
        self.x0, self.y0, self.cell = x0, y0, cell
        self.nx, self.ny = max(1, int((x1 - x0) / cell) + 1), max(1, int((y1 - y0) / cell) + 1)
        self.g = np.zeros((self.ny, self.nx), np.float32)

    def _ij(self, x, y):
        return (np.clip(((np.asarray(y) - self.y0) / self.cell).astype(int), 0, self.ny - 1),
                np.clip(((np.asarray(x) - self.x0) / self.cell).astype(int), 0, self.nx - 1))

    def rect(self, x0, y0, x1, y1, w=1.0):
        i0, j0 = self._ij(min(x0, x1), min(y0, y1))
        i1, j1 = self._ij(max(x0, x1), max(y0, y1))
        self.g[int(i0):int(i1) + 1, int(j0):int(j1) + 1] += w

    def polyline(self, px, py, w=1.0):
        px, py = np.asarray(px, float), np.asarray(py, float)
        if len(px) == 1:
            self.rect(px[0] - 3, py[0] - 3, px[0] + 3, py[0] + 3, w)
            return
        for i in range(len(px) - 1):
            L = math.hypot(px[i + 1] - px[i], py[i + 1] - py[i])
            t = np.linspace(0, 1, max(2, int(L / (self.cell * 0.5)) + 1))
            x = px[i] + (px[i + 1] - px[i]) * t
            y = py[i] + (py[i + 1] - py[i]) * t
            ii, jj = self._ij(x, y)
            self.g[ii, jj] += w

    def under(self, px, py, base, w=0.5):
        xs = np.arange(min(px), max(px), self.cell * 0.5)
        ys = np.interp(xs, px, py)
        for x, y in zip(xs, ys):
            self.rect(x, min(y, base), x, max(y, base), w)

    def score(self, x0, y0, x1, y1):
        i0, j0 = self._ij(x0, y0)
        i1, j1 = self._ij(x1, y1)
        return float(self.g[int(i0):int(i1) + 1, int(j0):int(j1) + 1].sum())


# --------------------------------------------------------------------------------------------- parts


class _Part:
    """Something drawn in a box of a Figure (an Axes, a KPI tile, a header ...)."""

    background = None

    def __init__(self, fig, box, name):
        self.fig, self.name = fig, name
        self.box = fig._box(box)                      # figure units (x0, y0, x1, y1)
        self.th = fig.theme

    @property
    def w(self):
        return self.box[2] - self.box[0]

    @property
    def h(self):
        return self.box[3] - self.box[1]

    def _emit(self, dwg, ox, oy):
        raise NotImplementedError

    def drawing(self, standalone=True):
        """This part alone as a vector Drawing (box-sized, theme background) — for SVG export or embedding."""
        dwg = Drawing(self.w, self.h, theme=self.th, name=self.name,
                      background=self.th.background if standalone else None)
        self._emit(dwg, 0.0, 0.0)
        return dwg

    def draw(self, cv=None, log=None):
        """Paint this part onto ``cv`` (default: the figure's canvas). Returns the canvas."""
        cv = cv if cv is not None else self.fig.cv
        if cv is None:
            raise ValueError("no canvas: pass cv=, or use fig.render()")
        f = self.fig
        dwg = Drawing(f.width, f.height, theme=self.th, name=self.name, background=False)
        full = self.box[0] <= 1 and self.box[1] <= 1 and self.box[2] >= f.width - 1 and self.box[3] >= f.height - 1
        bg = self.background
        if bg is None:                       # auto: the whole canvas, or a box that is still blank (black)
            k = cv.h / f.height
            x0, y0 = max(0, int(self.box[0] * k)), max(0, int(self.box[1] * k))
            x1, y1 = min(cv.w, int(math.ceil(self.box[2] * k))), min(cv.h, int(math.ceil(self.box[3] * k)))
            region = cv.buf[y0:y1, x0:x1]
            bg = full or (region.size > 0 and float(np.abs(region).max()) < 1e-6)
        if bg:
            bx = self.box
            if bg is True and not full and self.background is None and float(np.abs(cv.buf).max()) < 1e-6:
                bx = (0.0, 0.0, f.width, f.height)        # nothing drawn yet: the theme colour fills the whole canvas
            with dwg.group(f"{self.name} background"):
                dwg.rect(bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1], fill=self.th.background if bg is True else bg)
        self._emit(dwg, self.box[0], self.box[1])
        dwg.draw(cv, log=log, box=f._pxbox(cv))
        return cv

    def save_svg(self, path):
        """Write this part as a standalone SVG file; returns the path."""
        return self.drawing().save_svg(path)

    def render(self, scale=1.0):
        """This part alone as a new Canvas (box size x ``scale`` x figure pixels per unit)."""
        return self.drawing().render(scale=scale * self.fig.px_per_unit)


class Figure:
    """A set of charts, KPI tiles and headers sharing one canvas, theme and SVG file.

    ``Figure(cv=None, theme="light", size=None, background=True)``: with a Canvas, units are 1080 per canvas height
    (the width follows the aspect ratio); without one, ``size=(w, h)`` in px (default 1600 x 900).
    ``background``: True (theme background), "surface", a colour, or False (draw over what is on the canvas)."""

    def __init__(self, cv=None, theme="light", size=None, background=True):
        self.cv = cv
        self.theme = _theme(theme)
        if cv is not None:
            self.height = 1080.0
            self.width = 1080.0 * cv.w / cv.h
            self.px_per_unit = cv.h / 1080.0
        else:
            w, h = size or (1600, 900)
            self.width, self.height, self.px_per_unit = float(w), float(h), 1.0
        self.background = background
        self.parts = []
        self._n = 0

    # ---- geometry
    def _box(self, box):
        if box is None:
            box = (0, 0, 1, 1)
        b = [float(v) for v in box]
        if max(abs(v) for v in b) <= 1.0001:
            return (b[0] * self.width, b[1] * self.height, b[2] * self.width, b[3] * self.height)
        k = 1.0 / self.px_per_unit if self.cv is not None else 1.0       # pixel boxes
        return tuple(v * k for v in b)

    def _pxbox(self, cv):
        return (0, 0, cv.w, cv.h)

    def grid(self, rows=1, cols=1, box=(0, 0, 1, 1), gap=0.02, widths=None, heights=None):
        """``chart.grid`` with this figure's aspect ratio."""
        return grid(self, rows, cols, box, gap, widths, heights)

    def _name(self, base):
        self._n += 1
        names = {p.name for p in self.parts}
        nm = str(base) if base else f"chart {self._n}"
        while nm in names:
            nm = f"{nm} {self._n}"
        return nm

    def _bg(self):
        b = self.background
        if b is True:
            return self.theme.background
        if b == "surface":
            return self.theme.surface
        return b or None

    # ---- parts
    def axes(self, box=None, **kw):
        """New Axes in ``box`` (canvas fractions). Keywords as for ``Axes`` (title, subtitle, xlabel, ylabel, ...)."""
        ax = Axes(self, box, **kw)
        self.parts.append(ax)
        return ax

    def kpi(self, box, label, value, **kw):
        """KPI tile: label, big number, delta pill and an optional sparkline. See ``KPI``."""
        k = KPI(self, box, label, value, **kw)
        self.parts.append(k)
        return k

    def header(self, title, subtitle=None, right=None, box=None, **kw):
        """Dashboard / figure header: title, subtitle and right-aligned chips. See ``Header``."""
        hd = Header(self, box, title, subtitle, right, **kw)
        self.parts.append(hd)
        return hd

    def text(self, text, box, size=15, color=None, weight=None, align="left", italic=False, name=None):
        """Wrapped text block (captions, notes) in ``box``; ``color`` defaults to the theme muted ink."""
        t = _TextPart(self, box, text, size, color, weight, align, italic, name)
        self.parts.append(t)
        return t

    caption = text

    def sparkline(self, box, values, **kw):
        """A bare sparkline in ``box`` (see ``Axes.sparkline`` options)."""
        ax = self.axes(box, frame="none", grid=False, legend="none", pad=0)
        ax.sparkline(values, **kw)
        return ax

    # ---- output
    def build(self):
        """All parts as one vector Drawing (figure units). Each chart object is its own top-level group."""
        dwg = Drawing(self.width, self.height, theme=self.theme, background=self._bg(), name="figure")
        for p in self.parts:
            p._emit(dwg, p.box[0], p.box[1])
        return dwg

    def draw(self, cv=None, log=None):
        """Paint the whole figure onto ``cv`` (default the figure's canvas); one layer per chart object."""
        cv = cv if cv is not None else self.cv
        if cv is None:
            return self.render(log=log)
        self.build().draw(cv, log=log)
        return cv

    def render(self, scale=1.0, log=None):
        """New Canvas with the figure (size x px_per_unit x scale)."""
        d = self.build()
        d.px_per_unit = self.px_per_unit
        return d.render(scale=scale, log=log)

    def to_svg(self):
        return self.build().to_svg()

    def save_svg(self, path):
        """Write the whole figure as one SVG file; returns the path."""
        return self.build().save_svg(path)


def figure(cv=None, theme="light", size=None, background=True):
    """New ``Figure`` (see the class)."""
    return Figure(cv, theme, size, background)


def _fig_for(cv, theme):
    th = _theme(theme)
    figs = getattr(cv, "_chart_figures", None)
    if figs is None:
        figs = {}
        try:
            cv._chart_figures = figs
        except AttributeError:
            return Figure(cv, th, background=False)
    key = th.name
    if key not in figs:
        figs[key] = Figure(cv, th, background=False)
    return figs[key]


def axes(cv=None, box=None, theme="light", **kw):
    """Axes placed in ``box`` (canvas fractions, default the whole canvas) of Canvas ``cv``.

    Add series (``ax.line``, ``ax.bar`` ...) then ``ax.draw(log=log)``. Without ``cv`` the chart is standalone
    (``ax.render()`` / ``ax.save_svg(path)``; ``size=(w, h)`` keyword, default 1600 x 900)."""
    size = kw.pop("size", None)
    fig = _fig_for(cv, theme) if cv is not None else Figure(None, theme, size)
    return fig.axes(box, **kw)


def kpi(cv, box, label, value, theme="dark", **kw):
    """KPI tile on ``cv`` (call ``.draw(log=log)``). See ``KPI``."""
    return _fig_for(cv, theme).kpi(box, label, value, **kw)


def header(cv, title, subtitle=None, right=None, box=None, theme="dark", **kw):
    """Header on ``cv`` (call ``.draw(log=log)``). See ``Header``."""
    return _fig_for(cv, theme).header(title, subtitle, right, box, **kw)


def sparkline(cv, values, box, theme="light", **kw):
    """Sparkline in ``box`` of ``cv`` (call ``.draw(log=log)``)."""
    return _fig_for(cv, theme).sparkline(box, values, **kw)


# --------------------------------------------------------------------------------------------- Axes


def _as_list(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v)


def _is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


class Axes(_Part):
    """A chart in a box. Add series with the methods below, then ``draw(log=log)`` / ``save_svg(path)``.

    Keywords: ``title``, ``subtitle``, ``xlabel``, ``ylabel``, ``ylabel_right``; ``xscale`` auto / linear / log /
    date / category; ``yscale`` linear / log; ``xfmt`` / ``yfmt`` / ``yfmt_right`` number formats (see
    ``fmt_number``); ``xlim`` / ``ylim`` / ``ylim_right``; ``xticks`` / ``yticks`` explicit values;
    ``grid`` auto / "x" / "y" / "both" / False; ``frame`` "minimal" (baseline only) / "axes" (left + bottom axis
    lines with tick marks, paper style) / "box" / "none"; ``legend`` auto / top / right / bottom / none /
    tl / tr / bl / br; ``card`` (rounded surface card with shadow); ``panel`` ("a", "b" ... panel letter);
    ``font_scale``; ``palette`` (list of colours); ``pad``; ``background`` (True / colour)."""

    def __init__(self, fig, box=None, title=None, subtitle=None, xlabel=None, ylabel=None, ylabel_right=None,
                 xscale="auto", yscale="linear", xfmt=None, yfmt=None, yfmt_right=None, xlim=None, ylim=None,
                 ylim_right=None, xticks=None, yticks=None, grid="auto", frame="minimal", legend="auto", card=False,
                 panel=None, font_scale=None, palette=None, pad=None, background=None, name=None, zero=None):
        super().__init__(fig, box, fig._name(name or title))
        _check_choice("Axes", "xscale", xscale, ["auto", "linear", "log", "date", "category"])
        _check_choice("Axes", "yscale", yscale, ["linear", "log"])
        _check_choice("Axes", "frame", frame, ["minimal", "axes", "box", "none"])
        if isinstance(legend, str):
            _check_choice("Axes", "legend", legend, ["auto", "top", "right", "bottom", "none", "tl", "tr", "bl", "br"])
        self.title, self.subtitle, self.xlabel, self.ylabel, self.ylabel_right = (title, subtitle, xlabel, ylabel,
                                                                                  ylabel_right)
        self.xscale, self.yscale = xscale, yscale
        self.xfmt, self.yfmt, self.yfmt_right = xfmt, yfmt, yfmt_right
        self.xlim, self.ylim, self.ylim_right = xlim, ylim, ylim_right
        self.xticks, self.yticks = xticks, yticks
        self.grid, self.frame, self.legend, self.card, self.panel = grid, frame, legend, card, panel
        if font_scale is None:                       # taller boxes get bigger type (a full-canvas chart: 1.5x)
            font_scale = min(1.5, max(1.0, self.h / 600.0))
        self.fs = float(font_scale)
        self.palette = list(palette) if palette else [self.th.accent(i) for i in PALETTE_ORDER]
        self.pad = pad
        self.background = background
        self.zero = zero
        self.series = []
        self.marks = []                 # hline / vline / span / annotate / signif
        self._ci = 0

    # ---- colours
    def _color(self, color=None):
        if color is not None:
            if isinstance(color, int):
                return self.palette[color % len(self.palette)]
            return color
        c = self.palette[self._ci % len(self.palette)]
        self._ci += 1
        return c

    def _add(self, **s):
        self.series.append(s)
        return self

    # ---- series
    def line(self, x, y=None, label=None, color=None, smooth=False, markers=None, width=None, dash=None,
             fill="auto", axis="left", end_label=False, opacity=1.0):
        """Line series. ``y`` may be a dict {label: values} for several lines sharing ``x``; ``x`` alone → y with
        x = 0, 1, 2 ... ``x``: numbers, date strings ('2026-03', '2026-03-14') or category labels.
        ``smooth``: monotone curve (never overshoots). ``markers``: True / False / None (auto for short series).
        ``fill``: soft gradient area under the line ("auto" = when it is the only line). ``axis="right"``: second
        y axis. ``end_label``: label at the last point instead of the legend."""
        if isinstance(fill, str):
            _check_choice("Axes.line", "fill", fill, ["auto"])
        if y is None:
            x, y = list(range(len(_as_list(x)))), x
        if isinstance(y, dict):
            for k, v in y.items():
                self.line(x, v, label=k, color=None, smooth=smooth, markers=markers, width=width, dash=dash,
                          fill=fill, axis=axis, end_label=end_label, opacity=opacity)
            return self
        return self._add(kind="line", x=_as_list(x), y=np.asarray(y, np.float64), label=label,
                         color=self._color(color), smooth=smooth, markers=markers, width=width, dash=dash,
                         fill=fill, axis=axis, end_label=end_label, opacity=opacity)

    def area(self, x, y=None, label=None, color=None, stacked=False, smooth=False, opacity=0.32, axis="left"):
        """Area series (gradient fill under a line). ``y`` dict {label: values} + ``stacked=True`` → stacked areas."""
        if y is None:
            x, y = list(range(len(_as_list(x)))), x
        if isinstance(y, dict):
            base = None
            for k, v in y.items():
                v = np.asarray(v, np.float64)
                self._add(kind="area", x=_as_list(x), y=v + (base if (stacked and base is not None) else 0),
                          base=(base if stacked else None), label=k, color=self._color(None), smooth=smooth,
                          opacity=opacity, axis=axis)
                base = v + (base if (stacked and base is not None) else 0)
            return self
        return self._add(kind="area", x=_as_list(x), y=np.asarray(y, np.float64), base=None, label=label,
                         color=self._color(color), smooth=smooth, opacity=opacity, axis=axis)

    def bar(self, categories, values=None, label=None, stacked=False, horizontal=False, errors=None, color=None,
            value_labels="auto", fmt=None, highlight=None, radius=None, width=0.72, axis="left"):
        """Bars. ``categories`` list + ``values`` list, or a dict {category: value}; ``values`` as a dict
        {series: [..]} → grouped bars (``stacked=True`` stacks them). ``horizontal=True`` puts categories on the
        y axis (top to bottom). ``errors``: same shape as values, each ± value or (minus, plus).
        ``value_labels``: print values at the bar ends (``fmt`` number format); "auto" (default) = on for one
        series of up to 16 bars. ``highlight``: a category (or
        index) drawn in the accent while the others are toned down. ``radius``: rounded bar ends (default
        auto; 0 = square)."""
        if isinstance(categories, dict) and values is None:
            categories, values = list(categories.keys()), list(categories.values())
        cats = [str(c) for c in _as_list(categories)]
        if isinstance(values, dict):
            labels = list(values.keys())
            vals = [np.asarray(values[k], np.float64) for k in labels]
            errs = [None if errors is None else (errors.get(k) if isinstance(errors, dict) else errors[i])
                    for i, k in enumerate(labels)]
            cols = [self._color(color[i] if isinstance(color, (list, tuple)) else None) for i in range(len(labels))]
        else:
            labels = [label]
            vals = [np.asarray(values, np.float64)]
            errs = [errors]
            cols = [self._color(color)]
        if isinstance(value_labels, str):
            _check_choice("Axes.bar", "value_labels", value_labels, ["auto"])
            value_labels = len(vals) == 1 and len(cats) <= 16
        return self._add(kind="bar", cats=cats, vals=vals, labels=labels, errs=errs, colors=cols, stacked=stacked,
                         horizontal=horizontal, value_labels=value_labels, fmt=fmt, highlight=highlight,
                         radius=radius, width=width, axis=axis, label=label)

    def barh(self, categories, values=None, **kw):
        """Horizontal bars (``bar(..., horizontal=True)``)."""
        return self.bar(categories, values, horizontal=True, **kw)

    def scatter(self, x, y, size=None, color=None, label=None, trend=False, opacity=0.78, sizes=(4, 15), axis="left"):
        """Scatter points. ``size``: values mapped to radii ``sizes`` (area-true). ``color``: one colour, a list of
        category names (one legend entry each) or numbers (sequential colour ramp). ``trend``: least-squares line
        with R²."""
        x, y = _as_list(x), np.asarray(y, np.float64)
        groups = None
        cvals = None
        gcol = None
        if isinstance(color, (list, tuple, np.ndarray)) and len(color) == len(y) and not isinstance(color, str):
            cl = _as_list(color)
            if all(isinstance(c, str) and not c.startswith("#") for c in cl):
                groups = []
                for c in cl:
                    if c not in groups:
                        groups.append(c)
                groups.sort()
                gcol = {g: self._color(None) for g in groups}
                color = [gcol[c] for c in cl]
            elif all(_is_num(c) for c in cl):
                cvals = np.asarray(cl, np.float64)
                color = None
        col = color if isinstance(color, (list, tuple)) else self._color(color)
        return self._add(kind="scatter", x=x, y=y, size=None if size is None else np.asarray(size, np.float64),
                         color=col, groups=groups, gcolors=gcol, cvals=cvals, label=label, trend=trend, opacity=opacity,
                         sizes=sizes, axis=axis)

    def hist(self, values, bins=20, label=None, color=None, density=False, range=None, opacity=0.9):
        """Histogram of ``values`` (``bins`` count or edges)."""
        counts, edges = np.histogram(np.asarray(values, np.float64), bins=bins, range=range, density=density)
        return self._add(kind="hist", counts=counts.astype(np.float64), edges=edges, label=label,
                         color=self._color(color), opacity=opacity, axis="left")

    def heatmap(self, matrix, xlabels=None, ylabels=None, annotate=True, fmt=None, cmap=None, colorbar=None,
                vmin=None, vmax=None, gap=2.0):
        """Heatmap of ``matrix`` (rows top → bottom). ``cmap``: accent index, a colour, or a list of colours
        (low → high; 3 colours for diverging). ``annotate``: values in the cells (auto-contrast ink)."""
        M = np.asarray(matrix, np.float64)
        xl = [str(v) for v in (xlabels if xlabels is not None else range(M.shape[1]))]
        yl = [str(v) for v in (ylabels if ylabels is not None else range(M.shape[0]))]
        return self._add(kind="heatmap", M=M, xl=xl, yl=yl, annotate=annotate, fmt=fmt, cmap=cmap,
                         colorbar=(not annotate) if colorbar is None else colorbar, vmin=vmin, vmax=vmax, gap=gap,
                         label=None)

    def pie(self, values, labels=None, donut=0.0, center=None, colors=None, start=-90, pct=True, fmt=None,
            sort=False, legend_values=True):
        """Pie (``donut`` = hole fraction 0~0.9). ``values`` list or dict {label: value}. ``center``: text in the
        hole (str, (big, small) tuple, or None → the total). ``pct``: percentages on large slices."""
        if isinstance(values, dict):
            labels, values = list(values.keys()), list(values.values())
        vals = np.asarray(values, np.float64)
        labels = [str(l) for l in (labels or [f"#{i + 1}" for i in range(len(vals))])]
        if sort:
            o = np.argsort(-vals)
            vals, labels = vals[o], [labels[i] for i in o]
        cols = list(colors) if colors else [self.palette[i % len(self.palette)] for i in range(len(vals))]
        return self._add(kind="pie", vals=vals, labels=labels, donut=float(donut), center=center, colors=cols,
                         start=start, pct=pct, fmt=fmt, legend_values=legend_values, label=None)

    def donut(self, values, labels=None, hole=0.62, **kw):
        """Donut chart (``pie(..., donut=hole)``)."""
        return self.pie(values, labels, donut=hole, **kw)

    def radar(self, categories, values, label=None, max=None, levels=4, fill=True, fmt=None):
        """Radar / spider chart: ``values`` list (one series) or dict {series: values}; ``max`` = outer ring."""
        cats = [str(c) for c in categories]
        if not isinstance(values, dict):
            values = {label: values}
        ser = [(k, np.asarray(v, np.float64), self._color(None)) for k, v in values.items()]
        return self._add(kind="radar", cats=cats, ser=ser, vmax=max, levels=levels, fill=fill, fmt=fmt, label=None)

    def sparkline(self, values, x=None, color=None, fill=True, dot=True, width=2.2, smooth=True, band=None):
        """Minimal line for small boxes: no axes, gradient fill, end dot (``band=(lo, hi)`` shades a normal range)."""
        v = np.asarray(values, np.float64)
        return self._add(kind="spark", y=v, x=None if x is None else _as_list(x), color=self._color(color), fill=fill,
                         dot=dot, width=width, smooth=smooth, band=band, label=None)

    # ---- marks
    def band(self, x, lo, hi, color=None, opacity=0.16, label=None, smooth=False, axis="left"):
        """Shaded band between ``lo`` and ``hi`` along ``x`` (confidence interval, range)."""
        col = color if color is not None else (self.series[-1]["color"] if self.series and "color" in self.series[-1]
                                               and isinstance(self.series[-1]["color"], str) else self._color(None))
        return self._add(kind="band", x=_as_list(x), lo=np.asarray(lo, np.float64), hi=np.asarray(hi, np.float64),
                         color=col, opacity=opacity, label=label, smooth=smooth, axis=axis)

    def span(self, a, b, axis="x", color=None, opacity=0.08, label=None):
        """Shaded region from ``a`` to ``b`` across the plot (``axis="x"``: vertical strip; "y": horizontal)."""
        self.marks.append(dict(kind="span", a=a, b=b, axis=axis, color=color, opacity=opacity, label=label))
        return self

    def hline(self, y, label=None, color=None, dash="dash", width=1.3, axis="left"):
        """Horizontal reference line at ``y`` with an optional label at the right end."""
        self.marks.append(dict(kind="hline", v=y, label=label, color=color, dash=dash, width=width, axis=axis))
        return self

    def vline(self, x, label=None, color=None, dash="dash", width=1.3):
        """Vertical reference line at ``x`` (number, date string or category) with an optional label at the top."""
        self.marks.append(dict(kind="vline", v=x, label=label, color=color, dash=dash, width=width))
        return self

    def annotate(self, text, xy, xytext=None, offset=None, color=None, axis="left"):
        """Callout: ``text`` in a small pill with a leader line to data point ``xy`` (x, y). ``xytext``: pill
        position in data coordinates; ``offset``: (dx, dy) in units; neither → placed automatically where it
        covers no data."""
        self.marks.append(dict(kind="annotate", text=str(text), xy=xy, xytext=xytext, offset=offset, color=color,
                               axis=axis))
        return self

    def signif(self, a, b, text="*", level=None):
        """Significance bracket between bars ``a`` and ``b`` — a category name or (category, series label) —
        with ``text`` ("*", "**", "n.s.", "p = 0.01") above. Overlapping brackets stack automatically."""
        self.marks.append(dict(kind="signif", a=a, b=b, text=str(text), level=level))
        return self

    # ---- output convenience
    def drawing(self, standalone=True):
        return _Part.drawing(self, standalone)

    # ================================================================================== build
    def _S(self, k):
        return SIZES[k] * self.fs

    def _txt(self, g, x, y, s, size, fill=None, anchor="lt", bold=False, **kw):
        return _text(g,x, y, s, size=size, fill=fill if fill is not None else self.th.ink, anchor=anchor,
                      weight="bold" if bold else None, **kw)

    def _otxt(self, g, x, y, s, size, anchor="lt", **kw):
        """Text that also marks the occupancy grid (so legends and callouts avoid it)."""
        t = self._txt(g, x, y, s, size, anchor=anchor, **kw)
        occ = getattr(self, "_occ", None)
        if occ is not None:
            w, h = self._mw(s, size, kw.get("bold", False))
            a = _v._anchor(anchor)
            x0 = x - {"l": 0, "m": 0.5, "r": 1}[a[0]] * w
            y0 = y - {"t": 0, "a": 0, "m": 0.5, "s": 0.8, "b": 1, "d": 1}[a[1]] * h
            occ.rect(x0, y0, x0 + w, y0 + h, 2.0)
        return t

    def _mw(self, s, size, bold=False, max_width=None):
        return _measure(str(s), round(size, 3), self.th.font["family"], bold,
                        None if max_width is None else round(max_width, 2))

    def _mode(self):
        kinds = {s["kind"] for s in self.series}
        if "pie" in kinds:
            return "pie"
        if "radar" in kinds:
            return "radar"
        if kinds and kinds <= {"spark", "band"}:
            return "spark"
        return "cart"

    def _emit(self, dwg, ox, oy):
        th = self.th
        mode = self._mode()
        W, H = self.w, self.h
        pad = self.pad if self.pad is not None else (22 * self.fs if self.card else
                                                     max(6.0, min(26.0, 0.022 * min(W, H))) * self.fs)
        self._pad = pad
        if self.card:
            with dwg.group(f"{self.name} card", translate=(ox, oy), shadow=True) as g:
                g.rect(0, 0, W, H, r=th.radius + 6, fill=th.surface,
                       stroke=_mix(th.grid, th.surface, 0.35) if th.dark else None, stroke_width=1)
        elif self.background:
            with dwg.group(f"{self.name} background", translate=(ox, oy)) as g:
                g.rect(0, 0, W, H, fill=th.background if self.background is True else self.background)
        top = pad + self._header_h()
        if mode == "pie":
            self._emit_pie(dwg, ox, oy, pad, top)
        elif mode == "radar":
            self._emit_radar(dwg, ox, oy, pad, top)
        elif mode == "spark":
            self._emit_header(dwg, ox, oy, pad)
            self._emit_spark(dwg, ox, oy, pad, top)
        else:
            self._emit_cart(dwg, ox, oy, pad, top)

    # ---- header (title / subtitle / panel letter)
    def _title_x(self, pad):
        if self.panel:
            return pad + self._mw(self.panel, self._S("panel"), True)[0] + 12 * self.fs
        return pad

    def _header_h(self):
        h = 0.0
        if self.title:
            h += self._S("title") * 1.25
        if self.subtitle:
            h += self._S("subtitle") * 1.45
        if self.panel and not self.title:
            h = max(h, self._S("panel") * 1.15)
        if h:
            h += 14 * self.fs
        return h

    def _emit_header(self, dwg, ox, oy, pad, legend_row=None):
        if not (self.title or self.subtitle or self.panel):
            return
        th = self.th
        with dwg.group(f"{self.name} title", translate=(ox, oy)) as g:
            tx = self._title_x(pad)
            if self.panel:
                self._txt(g, pad, pad - 3 * self.fs, str(self.panel), self._S("panel"), bold=True)
            y = pad
            if self.title:
                self._txt(g, tx, y + (self._S("panel") - self._S("title")) * 0.35 if self.panel else y, self.title,
                          self._S("title"), bold=True)
                y += self._S("title") * 1.25 + (self._S("panel") - self._S("title")) * 0.35 * bool(self.panel)
            if self.subtitle:
                self._txt(g, tx, y + 2 * self.fs, self.subtitle, self._S("subtitle"), fill=th.muted)

    # ---- legend
    def _legend_entries(self):
        out = []
        for s in self.series:
            k = s["kind"]
            if k == "bar":
                if len(s["labels"]) > 1 or s["labels"][0]:
                    for lab, c in zip(s["labels"], s["colors"]):
                        if lab:
                            out.append(("box", c, str(lab)))
            elif k == "scatter" and s["groups"]:
                for gn in s["groups"]:
                    out.append(("dot", s["gcolors"][gn], gn))
            elif s.get("label") and not s.get("end_label"):
                kind = {"line": "line", "area": "box", "scatter": "dot", "hist": "box", "band": "band"}.get(k, "box")
                mk = False
                if k == "line":
                    mk = s["markers"] if s["markers"] is not None else (len(s["y"]) <= 14 and not s["dash"])
                out.append((kind, s["color"], str(s["label"]), s.get("dash"), mk))
        return out

    def _scatter_names(self, s):
        return s.get("_names") or []

    def _legend_size(self, entries, horizontal, max_w=None):
        fsz = self._S("legend")
        sw = 22 * self.fs
        gap_in, gap_out = 8 * self.fs, 20 * self.fs
        ws = [sw + gap_in + self._mw(e[2], fsz)[0] for e in entries]
        lh = fsz * 1.55
        if horizontal:
            rows, cur, rw = [[]], 0.0, []
            for i, w in enumerate(ws):
                if rows[-1] and max_w and cur + gap_out + w > max_w:
                    rw.append(cur)
                    rows.append([])
                    cur = 0.0
                cur += (gap_out if rows[-1] else 0) + w
                rows[-1].append(i)
            rw.append(cur)
            return max(rw), lh * len(rows), rows, ws
        return max(ws) if ws else 0, lh * len(ws), [[i] for i in range(len(ws))], ws

    def _draw_legend(self, g, entries, x, y, horizontal, max_w=None, align="left"):
        th = self.th
        fsz = self._S("legend")
        sw = 22 * self.fs
        gap_in, gap_out = 8 * self.fs, 20 * self.fs
        tw, thh, rows, ws = self._legend_size(entries, horizontal, max_w)
        lh = fsz * 1.55
        for r, row in enumerate(rows):
            rwid = sum(ws[i] for i in row) + gap_out * (len(row) - 1)
            cx = x + (tw - rwid if align == "right" else 0)
            cy = y + r * lh + lh / 2
            for i in row:
                kind, col, lab = entries[i][:3]
                dash = entries[i][3] if len(entries[i]) > 3 else None
                mk = entries[i][4] if len(entries[i]) > 4 else False
                if kind == "line":
                    g.line(cx, cy, cx + sw, cy, stroke=col, stroke_width=2.6 * self.fs, cap="round", dash=dash)
                    if mk:
                        g.circle(cx + sw / 2, cy, 3.6 * self.fs, fill=col, stroke=th.surface if self.card else
                             th.background, stroke_width=1.4 * self.fs)
                elif kind == "dot":
                    g.circle(cx + sw / 2, cy, 5.2 * self.fs, fill=col, fill_opacity=0.85)
                elif kind == "band":
                    g.rect(cx, cy - 6 * self.fs, sw, 12 * self.fs, r=3 * self.fs, fill=col, fill_opacity=0.28)
                else:
                    g.rect(cx + 3 * self.fs, cy - 7 * self.fs, sw - 6 * self.fs, 14 * self.fs, r=3.5 * self.fs, fill=col)
                if not horizontal:
                    pass
                self._txt(g, cx + sw + gap_in, cy, lab, fsz, fill=th.muted if th.dark else _mix(th.ink, th.muted, 0.35),
                          anchor="lm")
                cx += ws[i] + gap_out
        return tw, thh

    # ---- cartesian
    def _kinds(self):
        return [s["kind"] for s in self.series]

    def _horizontal(self):
        return any(s["kind"] == "bar" and s["horizontal"] for s in self.series)

    def _cat_union(self, lists):
        out = []
        seen = set()
        for L in lists:
            for c in L:
                c = str(c)
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out

    def _resolve_axes(self):
        """Decide scale kinds and categories; convert raw x values to numbers."""
        kinds = self._kinds()
        hz = self._horizontal()
        self._hz = hz
        for s_ in self.series:                       # a bar fmt also formats the value axis unless one was given
            if s_["kind"] == "bar" and s_["fmt"] is not None and s_["axis"] == "left":
                if s_["horizontal"] and self.xfmt is None:
                    self.xfmt = s_["fmt"]
                elif not s_["horizontal"] and self.yfmt is None:
                    self.yfmt = s_["fmt"]
        heat = "heatmap" in kinds
        xs_raw = [s["x"] for s in self.series if s["kind"] in ("line", "area", "scatter", "band")]
        xk = self.xscale
        if xk == "auto":
            if hz or ("hist" in kinds and "bar" not in kinds):
                xk = "linear"
            elif heat or "bar" in kinds:
                xk = "category"
            else:
                flat = [v for L in xs_raw for v in L]
                if flat and all(_is_num(v) for v in flat):
                    xk = "linear"
                elif flat and all(_parse_date(v) is not None for v in flat):
                    xk = "date"
                else:
                    xk = "category"
        yk = "category" if (hz or heat) else self.yscale
        self._xk, self._yk = xk, yk
        if xk == "category":
            lists = []
            for s in self.series:
                if s["kind"] == "bar" and not s["horizontal"]:
                    lists.append(s["cats"])
                elif s["kind"] == "heatmap":
                    lists.append(s["xl"])
                elif s["kind"] in ("line", "area", "scatter", "band"):
                    lists.append(s["x"])
            self._xcats = self._cat_union(lists)
        else:
            self._xcats = None
        if yk == "category":
            lists = []
            for s in self.series:
                if s["kind"] == "bar" and s["horizontal"]:
                    lists.append(s["cats"])
                elif s["kind"] == "heatmap":
                    lists.append(s["yl"])
            self._ycats = self._cat_union(lists)
        else:
            self._ycats = None

    def _xnum(self, v):
        """Raw x value → domain number."""
        if isinstance(v, (list, tuple, np.ndarray)):
            return np.array([self._xnum(u) for u in v], np.float64)
        if self._xk == "category":
            s = str(v)
            if s in self._xcats:
                return float(self._xcats.index(s))
            return float(v) if _is_num(v) else 0.0
        if self._xk == "date":
            d = _parse_date(v)
            return d if d is not None else float(v)
        return float(v)

    def _ycatnum(self, v):
        s = str(v)
        if s in self._ycats:
            return float(self._ycats.index(s))
        return float(v) if _is_num(v) else 0.0

    def _bar_stats(self, s):
        """Per category: (pos_top, neg_bottom) value extents including stacking and errors."""
        vals = np.stack(s["vals"])                       # (nser, ncat)
        hi_list, lo_list = [], []
        if s["stacked"]:
            pos = np.where(vals > 0, vals, 0).sum(0)
            neg = np.where(vals < 0, vals, 0).sum(0)
            hi_list += list(pos)
            lo_list += list(neg)
        else:
            for i, v in enumerate(s["vals"]):
                e = s["errs"][i]
                ep, em = self._err_arrays(e, len(v))
                hi_list += list(np.maximum(v + ep, 0))
                lo_list += list(np.minimum(v - em, 0))
        return hi_list, lo_list

    @staticmethod
    def _err_arrays(e, n):
        if e is None:
            return np.zeros(n), np.zeros(n)
        if np.ndim(e) == 0:
            a = np.full(n, float(e))
            return a, a
        arr = [x if x is not None else 0 for x in e]
        if all(np.ndim(x) == 0 for x in arr):
            a = np.asarray(arr, np.float64)
            return a, a
        em = np.array([float(x[0]) if np.ndim(x) else float(x) for x in arr])
        ep = np.array([float(x[1]) if np.ndim(x) else float(x) for x in arr])
        return ep, em

    def _extents(self):
        """Collect value extents per axis: dict axis -> (values list, include_zero, padded)."""
        ex = {"x": [[], False, False], "left": [[], False, False], "right": [[], False, False]}
        hz = self._hz
        for s in self.series:
            k = s["kind"]
            if k in ("line", "area", "scatter", "band"):
                if self._xk != "category":
                    ex["x"][0] += list(self._xnum(s["x"]))
                ax = ex[s.get("axis", "left")]
                if k == "band":
                    ax[0] += list(s["lo"]) + list(s["hi"])
                else:
                    ax[0] += list(s["y"])
                    if k == "area":
                        ax[1] = True
                        if s.get("base") is not None:
                            ax[0] += list(s["base"])
                    if k == "line":
                        y = s["y"][np.isfinite(s["y"])]
                        if len(y) and (self.zero or (self.zero is None and y.min() >= 0 and y.min() <= 0.45 * y.max())):
                            ax[1] = True
                    if k == "scatter":
                        ax[2] = True
                        ex["x"][2] = True
            elif k == "bar":
                hi, lo = self._bar_stats(s)
                ax = ex["x"] if s["horizontal"] else ex[s["axis"]]
                ax[0] += hi + lo
                ax[1] = True
                if s["value_labels"]:
                    ax.append("labels")
            elif k == "hist":
                ex["x"][0] += [float(s["edges"][0]), float(s["edges"][-1])]
                ex["left"][0] += list(s["counts"])
                ex["left"][1] = True
        for m in self.marks:
            if m["kind"] == "hline":
                ex[m["axis"]][0].append(float(m["v"]))
            elif m["kind"] == "vline" and self._xk != "category":
                ex["x"][0].append(float(self._xnum(m["v"])))
            elif m["kind"] == "span":
                if m["axis"] == "y":
                    ex["left"][0] += [float(m["a"]), float(m["b"])]
        return ex

    def _value_domain(self, vals, zero, padded, kind, n, lim, labels=False, extra_top=0.0, ticks=None):
        """→ (d0, d1, ticks)."""
        vals = [v for v in vals if np.isfinite(v)]
        if kind == "log":
            pos = [v for v in vals if v > 0] or [1.0, 10.0]
            lo, hi = (lim if lim else (min(pos), max(pos)))
            major, minor, d0, d1 = _log_ticks(lo, hi)
            if lim:
                d0, d1 = lim
                major = [t for t in major if d0 * (1 - 1e-9) <= t <= d1 * (1 + 1e-9)]
            self._minor = minor
            return d0, d1, ticks or major
        if not vals:
            vals = [0.0, 1.0]
        lo, hi = min(vals), max(vals)
        if zero:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        span = (hi - lo) or abs(hi) or 1.0
        if padded:
            lo, hi = lo - span * 0.05, hi + span * 0.05
        if labels:
            hi = hi + span * 0.1 if hi > 0 else hi
            lo = lo - span * 0.1 if lo < 0 else lo
        if extra_top:
            hi = hi + span * extra_top
        if lim:
            lo, hi = lim[0] if lim[0] is not None else lo, lim[1] if lim[1] is not None else hi
        if ticks is not None:
            t = [float(v) for v in ticks]
            return (min(lo, min(t)) if not lim else lo), (max(hi, max(t)) if not lim else hi), t
        t, step = nice_ticks(lo, hi, n)
        if lim:
            t = [v for v in t if lo - 1e-9 * span <= v <= hi + 1e-9 * span]
            return lo, hi, t
        return t[0], t[-1], t

    def _x_linear_domain(self, vals, padded, n, ticks=None):
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            vals = [0.0, 1.0]
        lo, hi = min(vals), max(vals)
        if self.xlim:
            lo = self._xnum(self.xlim[0]) if self.xlim[0] is not None else lo
            hi = self._xnum(self.xlim[1]) if self.xlim[1] is not None else hi
        if hi - lo < 1e-12:
            lo, hi = lo - 1, hi + 1
        span = hi - lo
        if self._xk == "log":
            major, minor, d0, d1 = _log_ticks(max(lo, 1e-300), hi)
            self._xminor = minor
            return d0, d1, ticks or major
        if padded and not self.xlim:
            lo, hi = lo - span * 0.04, hi + span * 0.04
        if ticks is not None:
            return lo, hi, [self._xnum(t) for t in ticks]
        if any(s["kind"] == "hist" for s in self.series) or (padded and not self.xlim):
            t, _ = nice_ticks(lo, hi, n)
            t = [v for v in t if lo - 1e-9 * span <= v <= hi + 1e-9 * span]
            return lo, hi, t
        t, _ = nice_ticks(lo, hi, n)
        t = [v for v in t if lo - 1e-9 * span <= v <= hi + 1e-9 * span]
        return lo, hi, t

    def _right_domain(self, vals, zero, nint):
        vals = [v for v in vals if np.isfinite(v)] or [0.0, 1.0]
        lo, hi = min(vals), max(vals)
        if zero or (lo >= 0 and lo <= 0.45 * hi):
            lo = min(lo, 0.0)
        if self.ylim_right:
            lo, hi = self.ylim_right
        nint = max(nint, 1)
        step = _nice_step((hi - lo) / nint)
        for _ in range(12):
            t0 = math.floor(lo / step + 1e-9) * step
            if t0 + step * nint >= hi - 1e-9 * abs(hi):
                break
            step = _nice_step(step * 1.0001)
        t = [round(t0 + i * step, 12) for i in range(nint + 1)]
        return t[0], t[-1], t

    def _x_ticks_layout(self, plot_w, n_override=None):
        """→ dict(ticks=[(pos_domain, label)], mode, height, over_l, over_r, rot, maxw)."""
        fsz = self._S("tick")
        gap = 10 * self.fs
        if self._xk == "category":
            cats = _date_labels(self._xcats)
            n = len(cats)
            step = plot_w / max(n, 1)
            ws = [self._mw(c, fsz)[0] for c in cats]
            lh = self._mw("Ag", fsz)[1]
            res = dict(ticks=[(float(i), c) for i, c in enumerate(cats)], mode="flat", rot=0.0, maxw=None)
            if not ws or max(ws) + gap <= step:
                hs = max([self._mw(c, fsz)[1] for c in cats] + [lh])
                res.update(height=hs, over_l=ws[0] / 2 if ws else 0, over_r=ws[-1] / 2 if ws else 0)
                return res
            if any(" " in c for c in cats):
                mw = step - gap * 0.6
                dims = [self._mw(c, fsz, max_width=mw) for c in cats]
                if max(d[0] for d in dims) <= mw + 0.5 and max(d[1] for d in dims) <= lh * 3.4 + 2:
                    res.update(mode="wrap", maxw=mw, height=max(d[1] for d in dims), over_l=dims[0][0] / 2,
                               over_r=dims[-1][0] / 2)
                    return res
            # rotate 45°, thin if needed
            ang = math.radians(45)
            keep = max(1, int(math.ceil((lh * 1.25) / (step * math.sin(ang)))))
            maxlen = 0.42 * self.h
            labels = []
            for i, c in enumerate(cats):
                if i % keep:
                    continue
                s = c
                while self._mw(s, fsz)[0] > maxlen and len(s) > 4:
                    s = s[:-2]
                labels.append((float(i), s if s == c else s.rstrip() + "…"))
            wmax = max(self._mw(s, fsz)[0] for _, s in labels)
            res.update(ticks=labels, mode="rot", rot=45.0, height=wmax * math.sin(ang) + lh * math.cos(ang),
                       over_l=wmax * math.cos(ang), over_r=lh * 0.5)
            return res
        # numeric / date
        n = n_override or max(2, min(10, int(plot_w / (105 * self.fs))))
        lh = self._mw("Ag", fsz)[1]
        for _ in range(10):
            ticks = self._x_numeric_ticks(n)
            if not ticks:
                break
            ws = [self._mw(l, fsz)[0] for _, l in ticks]
            xs = [self._xs_map(p) for p, _ in ticks] if hasattr(self, "_xs_tmp") else None
            if xs is None:
                break
            ok = all(xs[i + 1] - xs[i] >= (ws[i] + ws[i + 1]) / 2 + gap * 1.6 for i in range(len(xs) - 1))
            if ok or n <= 2:
                break
            n -= 1
        hs = max([self._mw(l, fsz)[1] for _, l in ticks] + [lh])
        ws = [self._mw(l, fsz)[0] for _, l in ticks] or [0]
        return dict(ticks=ticks, mode="flat", rot=0.0, maxw=None, height=hs, over_l=ws[0] / 2, over_r=ws[-1] / 2, n=n)

    def _x_numeric_ticks(self, n):
        d0, d1 = self._xdom
        if self._xk == "date":
            return [(p, l) for p, l in _date_ticks(d0, d1, n) if d0 - 1e-6 <= p <= d1 + 1e-6]
        if self._xk == "log":
            major = [t for t in _log_ticks(d0, d1)[0] if d0 * (1 - 1e-9) <= t <= d1 * (1 + 1e-9)]
            return list(zip(major, _log_labels(major, self.xfmt)))
        if self.xticks is not None:
            t = [self._xnum(v) for v in self.xticks]
        else:
            t, _ = nice_ticks(d0, d1, n)
            span = d1 - d0
            t = [v for v in t if d0 - 1e-9 * span <= v <= d1 + 1e-9 * span]
        return list(zip(t, _fmt_ticks(t, self.xfmt)))

    def _xs_map(self, p):
        return self._xs_tmp(p)

    def _y_ticks(self, plot_h):
        """Left/right y domains + labels (value axis), or categorical labels."""
        fsz = self._S("tick")
        ex = self._ex
        n = max(3, min(7, int(plot_h / (72 * self.fs))))
        out = {}
        if self._yk == "category":
            cats = self._ycats
            step = plot_h / max(len(cats), 1)
            maxw = self.w * 0.3
            labs = []
            lh = self._mw("Ag", fsz)[1]
            keep = max(1, int(math.ceil(lh * 1.1 / max(step, 1e-6))))
            for i, c in enumerate(cats):
                if i % keep:
                    continue
                wrap = self._mw(c, fsz)[0] > maxw and " " in c and step > lh * 2.2
                labs.append((float(i), c, maxw if wrap else None))
            out["left"] = (-0.5, len(cats) - 0.5, labs)
            return out
        extra = 0.0
        nsig = sum(1 for m in self.marks if m["kind"] == "signif")
        if nsig:
            extra = 0.11 * (1 + self._sig_levels())
        vals, zero, padded = ex["left"][:3]
        if not vals and ex["right"][0]:
            vals, zero, padded = ex["right"][:3]
        d0, d1, t = self._value_domain(vals, zero, padded, self._yk, n, self.ylim, labels="labels" in ex["left"],
                                       extra_top=extra, ticks=self.yticks)
        if self._yk == "log":
            labs = [(v, l, None) for v, l in zip(t, _log_labels(t, self.yfmt))]
        else:
            labs = [(v, l, None) for v, l in zip(t, _fmt_ticks(t, self.yfmt))]
        out["left"] = (d0, d1, labs)
        if ex["right"][0] and any(s.get("axis") == "right" for s in self.series):
            r0, r1, rt = self._right_domain(ex["right"][0], ex["right"][1], len(t) - 1)
            out["right"] = (r0, r1, [(v, l, None) for v, l in zip(rt, _fmt_ticks(rt, self.yfmt_right))])
        return out

    def _sig_levels(self):
        sig = [m for m in self.marks if m["kind"] == "signif"]
        return max(0, len(sig) - 1) if sig else 0

    def _emit_cart(self, dwg, ox, oy, pad, top):
        th, fs = self.th, self.fs
        W, H = self.w, self.h
        self._resolve_axes()
        self._ex = self._extents()
        for s in self.series:
            if s["kind"] == "scatter" and s["groups"]:
                pass
        fsz = self._S("tick")
        tgap = 9 * fs
        lab_sz = self._S("label")
        legend_entries = self._legend_entries()
        has_heat = any(s["kind"] == "heatmap" for s in self.series)
        heat = next((s for s in self.series if s["kind"] == "heatmap"), None)
        cb_w = (16 + 12) * fs + self._mw("0000", fsz)[0] if (heat is not None and heat["colorbar"]) else 0.0
        endlab_w = 0.0
        for s in self.series:
            if s["kind"] == "line" and s.get("end_label") and s.get("label"):
                endlab_w = max(endlab_w, self._mw(s["label"], self._S("legend"), True)[0] + 14 * fs)
        for s in self.series:
            if s["kind"] == "bar" and s["horizontal"] and s["value_labels"]:
                fmt = s["fmt"] if s["fmt"] is not None else self.xfmt
                tot = np.sum(np.stack(s["vals"]), 0) if s["stacked"] else np.concatenate(s["vals"])
                endlab_w = max([endlab_w] + [self._mw(fmt_number(v, fmt), self._S("value"))[0] + 12 * fs
                                             for v in tot if np.isfinite(v)])
        legend_pos = self.legend if legend_entries else "none"
        if has_heat:
            legend_pos = "none"
        extra_top, extra_right, extra_bottom = 0.0, 0.0, 0.0
        if legend_pos in ("top", "bottom", "right"):
            lw, lh_, _, _ = self._legend_size(legend_entries, legend_pos != "right", W - 2 * pad)
            if legend_pos == "top":
                extra_top = lh_ + 8 * fs
            elif legend_pos == "bottom":
                extra_bottom = lh_ + 10 * fs
            else:
                extra_right = lw + 18 * fs

        def layout(extra_top, extra_right, extra_bottom):
            xl_h = (lab_sz * 1.3 + 6 * fs) if self.xlabel else 0.0
            yl_w = (lab_sz * 1.3 + 8 * fs) if self.ylabel else 0.0
            yr_w = (lab_sz * 1.3 + 8 * fs) if self.ylabel_right else 0.0
            lh = self._mw("Ag", fsz)[1]
            bottom = lh + tgap + xl_h + extra_bottom
            plot_top = top + extra_top + (lh * 0.5 if self._yk != "category" else 0)
            res = None
            for _ in range(3):
                plot_h = max(20.0, H - pad - bottom - plot_top)
                yt = self._y_ticks(plot_h)
                lw = 0.0
                for v, l, mw in yt["left"][2]:
                    lw = max(lw, self._mw(l, fsz, max_width=mw)[0])
                rw = 0.0
                if "right" in yt:
                    for v, l, mw in yt["right"][2]:
                        rw = max(rw, self._mw(l, fsz)[0])
                left = pad + yl_w + (lw + tgap if lw else 0)
                right = pad + yr_w + (rw + tgap if rw else 0) + cb_w + endlab_w + extra_right
                plot_w = max(20.0, W - left - right)
                # x domain
                if self._xk == "category":
                    self._xdom = (-0.5, len(self._xcats) - 0.5)
                    xt = None
                else:
                    if self._xk in ("linear", "log", "date"):
                        d0, d1, _t = self._x_linear_domain(self._ex["x"][0], self._ex["x"][2] or (
                            "labels" in self._ex["x"]), max(2, int(plot_w / (105 * fs))))
                        if self._hz:
                            vals = [v for v in self._ex["x"][0] if np.isfinite(v)]
                            d0, d1, _t = self._value_domain(vals, True, False, "linear",
                                                            max(2, int(plot_w / (105 * fs))), self.xlim)
                        self._xdom = (d0, d1)
                sx = _Scale(self._xk if self._xk != "date" else "linear", *self._xdom, left, left + plot_w,
                            cats=self._xcats)
                self._xs_tmp = sx
                xt = self._x_ticks_layout(plot_w)
                if self._hz and self._xk == "linear":
                    xt["ticks"] = [(p, l) for p, l in xt["ticks"]]
                left2 = max(left, xt["over_l"] + pad - (0 if xt["mode"] != "rot" else 0) - 0)
                if self._xk != "category" and xt["ticks"]:
                    # overhang only matters if first/last tick sits at the plot edge
                    p0 = sx(xt["ticks"][0][0]) - left
                    left2 = max(left, pad + xt["over_l"] - p0)
                    pr = left + plot_w - sx(xt["ticks"][-1][0])
                    right = max(right, pad + xt["over_r"] - pr)
                elif xt["mode"] == "rot":
                    left2 = max(left, pad + xt["over_l"] - plot_w / len(self._xcats) / 2)
                new_bottom = xt["height"] + tgap + xl_h + extra_bottom
                if xt["mode"] == "rot":
                    new_bottom += 4 * fs
                changed = abs(new_bottom - bottom) > 0.5 or abs(left2 - left) > 0.5
                bottom = new_bottom
                left = left2
                plot_w = max(20.0, W - left - right)
                res = dict(left=left, right=right, top=plot_top, bottom=bottom, plot_w=plot_w,
                           plot_h=max(20.0, H - pad - bottom - plot_top), yt=yt, xt=xt, xl_h=xl_h, yl_w=yl_w,
                           yr_w=yr_w, lw=lw, rw=rw)
                if not changed:
                    break
            # final x ticks for the final plot width
            L = res
            if self._xk != "category":
                self._xs_tmp = _Scale(self._xk if self._xk != "date" else "linear", *self._xdom, L["left"],
                                      L["left"] + L["plot_w"], cats=self._xcats)
                L["xt"] = self._x_ticks_layout(L["plot_w"])
            yt = self._y_ticks(L["plot_h"])
            L["yt"] = yt
            return L

        L = layout(extra_top, extra_right, extra_bottom)
        px0, py0 = L["left"], L["top"]
        px1, py1 = px0 + L["plot_w"], py0 + L["plot_h"]
        self._plot = (px0, py0, px1, py1)
        # scales
        xk = self._xk
        sx = _Scale("log" if xk == "log" else "linear", *self._xdom, px0, px1, cats=self._xcats)
        yl = L["yt"]["left"]
        if self._yk == "category":
            sy = _Scale("linear", -0.5, len(self._ycats) - 0.5, py0, py1, cats=self._ycats)
        else:
            sy = _Scale("log" if self._yk == "log" else "linear", yl[0], yl[1], py1, py0)
        sy2 = None
        if "right" in L["yt"]:
            r = L["yt"]["right"]
            sy2 = _Scale("linear", r[0], r[1], py1, py0)
        self._sx, self._sy, self._sy2 = sx, sy, sy2
        occ = _Occ(px0, py0, px1, py1, 6.0 * fs)
        self._occ = occ

        # legend placement (inside / header) needs the data occupancy, computed while drawing
        groups = []                                          # (name, fn) drawn later in order
        self._emit_header(dwg, ox, oy, pad)
        self._emit_axes(dwg, ox, oy, L)
        mclip = (px0 - 8 * fs, py0 - 8 * fs, px1 + 8 * fs, py1 + 8 * fs)          # lines + markers
        tclip = (px0, py0, px1, py1)                                              # fills
        # spans under everything
        for i, m in enumerate(self.marks):
            if m["kind"] == "span":
                with dwg.group(f"{self.name} {m['label'] or 'span'}", translate=(ox, oy)) as g:
                    self._draw_span(g, m)
        for s in self.series:
            nm = s.get("label") or s["kind"]
            k = s["kind"]
            cl = None if k in ("heatmap", "bar", "hist") else (tclip if k in ("band", "area") else mclip)
            with dwg.group(f"{self.name} {nm}", translate=(ox, oy), clip=cl) as g:
                getattr(self, "_draw_" + s["kind"])(g, s)
        for m in self.marks:
            if m["kind"] == "span" and m["label"]:
                with dwg.group(f"{self.name} {m['label']} label", translate=(ox, oy)) as g:
                    self._draw_span(g, m, label=True)
            if m["kind"] in ("hline", "vline"):
                with dwg.group(f"{self.name} {m['label'] or m['kind']}", translate=(ox, oy)) as g:
                    self._draw_refline(g, m)
        sig = [m for m in self.marks if m["kind"] == "signif"]
        if sig:
            with dwg.group(f"{self.name} significance", translate=(ox, oy)) as g:
                self._draw_signif(g, sig)
        for m in self.marks:
            if m["kind"] == "annotate":
                with dwg.group(f"{self.name} note {m['text'][:24]}", translate=(ox, oy)) as g:
                    self._draw_annotate(g, m)
        for s in self.series:
            if s["kind"] == "line" and s.get("end_label") and s.get("label"):
                with dwg.group(f"{self.name} {s['label']} label", translate=(ox, oy)) as g:
                    self._draw_end_label(g, s)
        if heat is not None and heat["colorbar"]:
            with dwg.group(f"{self.name} colorbar", translate=(ox, oy)) as g:
                self._draw_colorbar(g, heat, L)
        if legend_pos != "none":
            with dwg.group(f"{self.name} legend", translate=(ox, oy)) as g:
                self._place_legend(g, legend_entries, legend_pos, L, pad)

    def _place_legend(self, g, entries, pos, L, pad):
        fs = self.fs
        px0, py0, px1, py1 = self._plot
        W = self.w
        if pos == "top":
            self._draw_legend(g, entries, px0, py0 - L["top"] + self._pad + self._header_h(), True, W - 2 * pad)
            return
        if pos == "bottom":
            lw, lh, _, _ = self._legend_size(entries, True, W - 2 * pad)
            self._draw_legend(g, entries, (px0 + px1 - lw) / 2, self.h - pad - lh, True, W - 2 * pad)
            return
        if pos == "right":
            lw, lh, _, _ = self._legend_size(entries, False)
            self._draw_legend(g, entries, W - pad - lw, py0, False)
            return
        # inside candidates
        horiz = len(entries) <= 4
        lw, lh, _, _ = self._legend_size(entries, horiz, (px1 - px0) * 0.9)
        if lw > (px1 - px0) * 0.9:
            horiz = False
            lw, lh, _, _ = self._legend_size(entries, False)
        m = 10 * fs
        cands = {"tr": (px1 - m - lw, py0 + m * 0.4), "tl": (px0 + m, py0 + m * 0.4),
                 "br": (px1 - m - lw, py1 - m - lh), "bl": (px0 + m, py1 - m - lh)}
        order = ["tr", "tl", "br", "bl"] if pos == "auto" else [pos]
        best, bs = None, 1e18
        for k in order:
            x, y = cands[k]
            sc = self._occ.score(x - 10 * fs, y - 8 * fs, x + lw + 10 * fs, y + lh + 8 * fs)
            if sc < bs - 1e-6:
                best, bs = k, sc
        if pos == "auto" and bs > 0:
            # try the title row (right aligned)
            hl, hh, _, _ = self._legend_size(entries, True)
            tx = self._title_x(pad) + (self._mw(self.title, self._S("title"), True)[0] if self.title else 0)
            if self.title and tx + 30 * fs + hl <= W - pad:
                self._draw_legend(g, entries, W - pad - hl, pad + (self._S("title") * 1.25 - hh) / 2, True,
                                  align="right")
                return
            # just above the plot (right aligned) when there is a header gap
            if not self.title and py0 - hh - 2 * fs >= pad * 0.5:
                self._draw_legend(g, entries, px1 - hl, py0 - hh - 2 * fs, True, align="right")
                return
        x, y = cands[best]
        bg = self.th.surface if self.card else (self.background if isinstance(self.background, str) else
                                                self.fig._bg() or self.th.background)
        if bs > 0:
            g.rect(x - 6 * fs, y - 3 * fs, lw + 12 * fs, lh + 6 * fs, r=6 * fs, fill=bg, fill_opacity=0.88)
        self._draw_legend(g, entries, x, y, horiz, (px1 - px0) * 0.9)

    def _emit_axes(self, dwg, ox, oy, L):
        th, fs = self.th, self.fs
        px0, py0, px1, py1 = self._plot
        sx, sy, sy2 = self._sx, self._sy, self._sy2
        fsz = self._S("tick")
        tgap = 9 * fs
        tick_col = th.muted
        frame = self.frame
        gmode = self.grid
        if gmode == "auto":
            if self._hz:
                gmode = "x"
            elif any(s["kind"] == "scatter" for s in self.series):
                gmode = "both"
            elif any(s["kind"] == "heatmap" for s in self.series):
                gmode = False
            else:
                gmode = "y"
        heat = any(s["kind"] == "heatmap" for s in self.series)
        grid_col = th.grid
        axis_col = _mix(th.ink, th.background, 0.62 if th.dark else 0.55)
        with dwg.group(f"{self.name} axes", translate=(ox, oy)) as g:
            gw = 1.0 * fs
            # grid
            if gmode in ("y", "both", True) and self._yk != "category":
                for v, l, _ in L["yt"]["left"][2]:
                    y = sy(v)
                    if frame in ("axes", "box") and abs(y - py1) < 0.5:
                        continue
                    g.line(px0, y, px1, y, stroke=grid_col, stroke_width=gw)
                if self._yk == "log":
                    for v in getattr(self, "_minor", []):
                        if sy.d0 <= v <= sy.d1:
                            y = sy(v)
                            g.line(px0, y, px1, y, stroke=grid_col, stroke_width=gw * 0.7, stroke_opacity=0.45)
            if gmode in ("x", "both", True) and self._xk != "category":
                for p, l in L["xt"]["ticks"]:
                    x = sx(p)
                    if frame in ("axes", "box") and abs(x - px0) < 0.5:
                        continue
                    g.line(x, py0, x, py1, stroke=grid_col, stroke_width=gw)
            # frame
            if frame in ("axes", "box"):
                g.line(px0, py1, px1, py1, stroke=axis_col, stroke_width=1.2 * fs, cap="square")
                g.line(px0, py0, px0, py1, stroke=axis_col, stroke_width=1.2 * fs, cap="square")
                if frame == "box":
                    g.line(px0, py0, px1, py0, stroke=axis_col, stroke_width=1.2 * fs)
                    g.line(px1, py0, px1, py1, stroke=axis_col, stroke_width=1.2 * fs)
                if sy2 is not None:
                    g.line(px1, py0, px1, py1, stroke=axis_col, stroke_width=1.2 * fs)
            elif frame == "minimal" and not heat:
                if self._hz:
                    x0v = sx(0.0) if sx.d0 <= 0 <= sx.d1 else px0
                    g.line(x0v, py0, x0v, py1, stroke=axis_col, stroke_width=1.3 * fs)
                elif self._yk != "category":
                    zy = sy(0.0) if (self._yk != "log" and sy.d0 <= 0 <= sy.d1) else py1
                    g.line(px0, zy, px1, zy, stroke=axis_col, stroke_width=1.3 * fs)
            tick_len = 5 * fs if frame in ("axes", "box") else 0
            # y tick labels
            if frame != "none":
                for v, l, mw in L["yt"]["left"][2]:
                    y = sy(v)
                    if tick_len:
                        g.line(px0 - tick_len, y, px0, y, stroke=axis_col, stroke_width=1.2 * fs)
                    if mw:
                        self._txt(g, px0 - tgap - tick_len, y, l, fsz, fill=tick_col, anchor="rm", max_width=mw,
                                  align="right")
                    else:
                        self._txt(g, px0 - tgap - tick_len, y, l, fsz, fill=tick_col, anchor="rm")
                if self._yk == "log" and tick_len:
                    for v in getattr(self, "_minor", []):
                        if sy.d0 <= v <= sy.d1:
                            g.line(px0 - tick_len * 0.55, sy(v), px0, sy(v), stroke=axis_col, stroke_width=1.0 * fs)
                if "right" in L["yt"]:
                    for v, l, _ in L["yt"]["right"][2]:
                        y = sy2(v)
                        if tick_len:
                            g.line(px1, y, px1 + tick_len, y, stroke=axis_col, stroke_width=1.2 * fs)
                        self._txt(g, px1 + tgap + tick_len, y, l, fsz, fill=tick_col, anchor="lm")
                # x tick labels
                xt = L["xt"]
                ty = py1 + tgap + tick_len
                for p, l in xt["ticks"]:
                    x = sx(p)
                    if tick_len:
                        g.line(x, py1, x, py1 + tick_len, stroke=axis_col, stroke_width=1.2 * fs)
                    if xt["mode"] == "rot":
                        self._txt(g, x + 4 * fs, ty + 2 * fs, l, fsz, fill=tick_col, anchor="rm", rotate=-45)
                    elif xt["mode"] == "wrap":
                        self._txt(g, x, ty, l, fsz, fill=tick_col, anchor="mt", max_width=xt["maxw"], align="center",
                                  line_spacing=1.1)
                    else:
                        self._txt(g, x, ty, l, fsz, fill=tick_col, anchor="mt", align="center", line_spacing=1.1)
                if self._xk == "log" and tick_len:
                    for v in getattr(self, "_xminor", []):
                        if sx.d0 <= v <= sx.d1:
                            g.line(sx(v), py1, sx(v), py1 + tick_len * 0.55, stroke=axis_col, stroke_width=1.0 * fs)
            lab_sz = self._S("label")
            lab_col = _mix(th.ink, th.muted, 0.45)
            if self.xlabel:
                self._txt(g, (px0 + px1) / 2, py1 + tgap + tick_len + L["xt"]["height"] + 8 * fs, self.xlabel,
                          lab_sz, fill=lab_col, anchor="mt")
            if self.ylabel:
                self._txt(g, self._pad + lab_sz * 0.6, (py0 + py1) / 2, self.ylabel, lab_sz, fill=lab_col,
                          anchor="mm", rotate=-90)
            if self.ylabel_right:
                self._txt(g, self.w - self._pad - lab_sz * 0.6, (py0 + py1) / 2, self.ylabel_right, lab_sz,
                          fill=lab_col, anchor="mm", rotate=90)

    # ---- series painters
    def _ys(self, s):
        return self._sy2 if (s.get("axis") == "right" and self._sy2 is not None) else self._sy

    def _xy(self, s, y):
        xs = self._sx(self._xnum(s["x"]))
        ys = self._ys(s)(y)
        xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
        ok = np.isfinite(ys) & np.isfinite(xs)
        return xs[ok], ys[ok]

    def _bgcol(self):
        if self.card:
            return self.th.surface
        if isinstance(self.background, str):
            return self.background
        return self.fig._bg() or self.th.background

    def _draw_line(self, g, s):
        fs = self.fs
        px, py = self._xy(s, s["y"])
        if not len(px):
            return
        o = np.argsort(px, kind="stable") if self._xk != "category" else np.arange(len(px))
        px, py = px[o], py[o]
        col = s["color"]
        w = (s["width"] or 2.6) * fs
        fill = s["fill"]
        if fill == "auto":
            fill = sum(1 for t in self.series if t["kind"] in ("line", "area")) == 1
        if fill:
            base = self._ys(s)(0.0) if self._ys(s).d0 <= 0 <= self._ys(s).d1 else self._plot[3]
            _curve(g, px, py, s["smooth"], close_to=base,
                   fill=linear_gradient([(0, col, 0.30), (1, col, 0.0)]), stroke=None, opacity=s["opacity"])
            self._occ.under(px, py, base, 0.3)
        _curve(g, px, py, s["smooth"], stroke=col, stroke_width=w, cap="round", join="round", fill=None,
               dash=s["dash"], opacity=s["opacity"])
        self._occ.polyline(px, py)
        mk = s["markers"]
        if mk is None:
            mk = len(px) <= 14 and not s["dash"]
        if mk:
            r = max(3.2, w * 1.35)
            for x, y in zip(px, py):
                g.circle(x, y, r, fill=col, stroke=self._bgcol(), stroke_width=1.6 * fs)

    def _draw_end_label(self, g, s):
        px, py = self._xy(s, s["y"])
        if not len(px):
            return
        i = int(np.argmax(px))
        g.circle(px[i], py[i], 4.2 * self.fs, fill=s["color"], stroke=self._bgcol(), stroke_width=1.6 * self.fs)
        self._txt(g, px[i] + 10 * self.fs, py[i], s["label"], self._S("legend"), fill=s["color"], anchor="lm",
                  bold=True)

    def _draw_area(self, g, s):
        px, py = self._xy(s, s["y"])
        if not len(px):
            return
        col = s["color"]
        sy = self._ys(s)
        if s.get("base") is not None:
            bx, by = self._xy(s, s["base"])
            close = (bx[::-1], by[::-1])
            fill = linear_gradient([(0, col, s["opacity"] + 0.25), (1, col, s["opacity"] + 0.1)])
        else:
            close = sy(0.0) if sy.d0 <= 0 <= sy.d1 else self._plot[3]
            fill = linear_gradient([(0, col, s["opacity"] + 0.12), (1, col, 0.02)])
        _curve(g, px, py, s["smooth"], close_to=close, fill=fill, stroke=None)
        _curve(g, px, py, s["smooth"], stroke=col, stroke_width=2.2 * self.fs, cap="round", join="round", fill=None)
        self._occ.under(px, py, self._plot[3], 0.6)

    def _draw_band(self, g, s):
        px, lo = self._xy(s, s["lo"])
        _, hi = self._xy(s, s["hi"])
        if not len(px):
            return
        _curve(g, px, hi, s["smooth"], close_to=(px[::-1], lo[::-1]), fill=s["color"], fill_opacity=s["opacity"],
               stroke=None)
        for x, a, b in zip(px, lo, hi):
            self._occ.rect(x, min(a, b), x, max(a, b), 0.2)

    def _draw_scatter(self, g, s):
        fs = self.fs
        px, py = self._xy(s, s["y"])
        n = len(px)
        if not n:
            return
        if s["size"] is not None:
            sz = s["size"]
            a = (sz - np.nanmin(sz)) / max(np.nanmax(sz) - np.nanmin(sz), 1e-12)
            r0, r1 = s["sizes"]
            rad = np.sqrt(r0 ** 2 + a * (r1 ** 2 - r0 ** 2)) * fs
        else:
            rad = np.full(n, 5.0 * fs)
        if s["cvals"] is not None:
            cv = s["cvals"]
            t = (cv - cv.min()) / max(cv.max() - cv.min(), 1e-12)
            base = s["color"] if isinstance(s["color"], str) else self.palette[0]
            stops = [_mix(self._bgcol(), base, 0.25), base, _mix(base, self.th.ink, 0.45)]
            cols = [_v._hex(c) for c in _ramp(stops, t)]
        elif isinstance(s["color"], (list, tuple)):
            cols = list(s["color"])
        else:
            cols = [s["color"]] * n
        order = np.argsort(-rad, kind="stable")
        ring = self._bgcol()
        for i in order:
            g.circle(px[i], py[i], rad[i], fill=cols[i], fill_opacity=s["opacity"], stroke=ring,
                     stroke_width=1.0 * fs, stroke_opacity=0.9)
            self._occ.rect(px[i] - rad[i], py[i] - rad[i], px[i] + rad[i], py[i] + rad[i])
        if s["trend"]:
            xs = self._xnum(s["x"])
            ys = s["y"]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() >= 2:
                xl = xs[ok] if self._xk != "log" else np.log10(xs[ok])
                yl = ys[ok] if self._yk != "log" else np.log10(ys[ok])
                k, b = np.polyfit(xl, yl, 1)
                r = np.corrcoef(xl, yl)[0, 1]
                xa = np.linspace(xl.min(), xl.max(), 32)
                ya = k * xa + b
                xx = self._sx(xa if self._xk != "log" else 10 ** xa)
                yy = self._ys(s)(ya if self._yk != "log" else 10 ** ya)
                tc = self.th.ink if not isinstance(s["color"], str) else _mix(s["color"], self.th.ink, 0.35)
                g.polyline(list(zip(xx, yy)), stroke=tc, stroke_width=1.8 * fs, dash="dash", cap="round")
                self._occ.polyline(xx, yy, 1.5)
                txt = f"R² = {r * r:.2f}"
                tsz = self._S("value")
                tw, thh = self._mw(txt, tsz)
                px0, py0, px1, py1 = self._plot
                m = 8 * fs
                cands = [(px0 + m, py0 + m), (px1 - m - tw, py1 - m - thh), (px1 - m - tw, py0 + m),
                         (px0 + m, py1 - m - thh)]
                best, bs = None, 1e18
                for i, (bx, by) in enumerate(cands):
                    sc = self._occ.score(bx - 3 * fs, by - 3 * fs, bx + tw + 3 * fs, by + thh + 3 * fs) + 0.01 * i
                    if sc < bs:
                        best, bs = (bx, by), sc
                self._txt(g, best[0], best[1], txt, tsz, fill=tc, anchor="lt", halo=self._bgcol())
                self._occ.rect(best[0], best[1], best[0] + tw, best[1] + thh, 3)

    def _scatter_names_set(self, s, names):
        s["_names"] = names

    def _draw_hist(self, g, s):
        fs = self.fs
        e, c = s["edges"], s["counts"]
        sy = self._sy
        base = sy(0.0)
        gap = min(1.2 * fs, (self._sx(e[1]) - self._sx(e[0])) * 0.12)
        for i in range(len(c)):
            x0, x1 = self._sx(e[i]) + gap / 2, self._sx(e[i + 1]) - gap / 2
            y = sy(c[i])
            if c[i] <= 0:
                continue
            r = min(3.0 * fs, (x1 - x0) / 3)
            g.rect(x0, y, x1 - x0, base - y, r=(r, r, 0, 0), fill=s["color"], fill_opacity=s["opacity"])
            self._occ.rect(x0, y, x1, base)

    def _bar_geom(self, s):
        """→ list of dict(i (series), j (category), v, x0, y0, x1, y1 (pixel rect), end (outer edge px), tip)."""
        hz = s["horizontal"]
        cats = s["cats"]
        nser = len(s["vals"])
        if hz:
            cat_scale, val_scale = self._sy, self._sx
            step = abs(self._sy.r1 - self._sy.r0) / max(len(self._ycats), 1)
            num = self._ycatnum
        else:
            cat_scale, val_scale = self._sx, self._ys(s)
            step = abs(self._sx.r1 - self._sx.r0) / max(len(self._xcats), 1)
            num = self._xnum
        gw = step * s["width"]
        out = []
        cum_pos = np.zeros(len(cats))
        cum_neg = np.zeros(len(cats))
        sub = gw if s["stacked"] or nser == 1 else gw / nser
        inner = 0 if (s["stacked"] or nser == 1) else min(3.0 * self.fs, sub * 0.14)
        for i, vals in enumerate(s["vals"]):
            for j, c in enumerate(cats):
                v = float(vals[j]) if j < len(vals) else float("nan")
                if not np.isfinite(v):
                    continue
                center = cat_scale(num(c))
                if s["stacked"]:
                    a = cum_pos[j] if v >= 0 else cum_neg[j]
                    b = a + v
                    if v >= 0:
                        cum_pos[j] = b
                    else:
                        cum_neg[j] = b
                    c0, c1 = center - gw / 2, center + gw / 2
                else:
                    a, b = 0.0, v
                    c0 = center - gw / 2 + i * sub + inner / 2
                    c1 = c0 + sub - inner
                va, vb = val_scale(a), val_scale(b)
                if hz:
                    out.append(dict(i=i, j=j, v=v, x0=min(va, vb), x1=max(va, vb), y0=c0, y1=c1, a=a, b=b,
                                    center=(c0 + c1) / 2, end=vb, cat=c))
                else:
                    out.append(dict(i=i, j=j, v=v, x0=c0, x1=c1, y0=min(va, vb), y1=max(va, vb), a=a, b=b,
                                    center=(c0 + c1) / 2, end=vb, cat=c))
        if s["stacked"]:
            # mark outermost segments
            for j in range(len(cats)):
                segs = [r for r in out if r["j"] == j]
                pos = [r for r in segs if r["v"] >= 0]
                neg = [r for r in segs if r["v"] < 0]
                if pos:
                    max(pos, key=lambda r: r["b"])["outer"] = True
                if neg:
                    min(neg, key=lambda r: r["b"])["outer"] = True
        else:
            for r in out:
                r["outer"] = True
        return out

    def _draw_bar(self, g, s):
        th, fs = self.th, self.fs
        hz = s["horizontal"]
        geom = self._bar_geom(s)
        s["_geom"] = geom
        hl = s["highlight"]
        if hl is not None and not isinstance(hl, (list, tuple, set)):
            hl = [hl]
        for r in geom:
            col = s["colors"][r["i"]]
            if hl is not None:
                on = (r["cat"] in [str(h) for h in hl]) or (r["j"] in [h for h in hl if isinstance(h, int)])
                if not on:
                    col = _mix(self._bgcol(), col, 0.42)
            thick = (r["y1"] - r["y0"]) if hz else (r["x1"] - r["x0"])
            rad = s["radius"] if s["radius"] is not None else min(thick * 0.2, 7.0 * fs)
            rad = rad if r.get("outer") else 0.0
            pos = r["v"] >= 0
            if hz:
                radii = (0, rad, rad, 0) if pos else (rad, 0, 0, rad)
            else:
                radii = (rad, rad, 0, 0) if pos else (0, 0, rad, rad)
            st = {}
            if s["stacked"]:
                st = dict(stroke=self._bgcol(), stroke_width=1.2 * fs)
            g.rect(r["x0"], r["y0"], r["x1"] - r["x0"], r["y1"] - r["y0"], r=radii, fill=col, **st)
            self._occ.rect(r["x0"], r["y0"], r["x1"], r["y1"])
        # error bars
        if not s["stacked"]:
            ecol = _mix(th.ink, self._bgcol(), 0.25)
            for i, e in enumerate(s["errs"]):
                if e is None:
                    continue
                ep, em = self._err_arrays(e, len(s["vals"][i]))
                for r in geom:
                    if r["i"] != i:
                        continue
                    j = r["j"]
                    vs = self._sx if hz else self._ys(s)
                    lo, hi = vs(r["v"] - em[j]), vs(r["v"] + ep[j])
                    cw = min((r["y1"] - r["y0"]) if hz else (r["x1"] - r["x0"]), 30 * fs) * 0.32
                    c = r["center"]
                    if hz:
                        g.line(lo, c, hi, c, stroke=ecol, stroke_width=1.5 * fs)
                        g.line(lo, c - cw / 2, lo, c + cw / 2, stroke=ecol, stroke_width=1.5 * fs)
                        g.line(hi, c - cw / 2, hi, c + cw / 2, stroke=ecol, stroke_width=1.5 * fs)
                    else:
                        g.line(c, lo, c, hi, stroke=ecol, stroke_width=1.5 * fs)
                        g.line(c - cw / 2, lo, c + cw / 2, lo, stroke=ecol, stroke_width=1.5 * fs)
                        g.line(c - cw / 2, hi, c + cw / 2, hi, stroke=ecol, stroke_width=1.5 * fs)
                    r["tip"] = min(lo, hi) if not hz else max(lo, hi)
                    if hz:
                        self._occ.rect(min(lo, hi), c - cw / 2, max(lo, hi), c + cw / 2)
                    else:
                        self._occ.rect(c - cw / 2, min(lo, hi), c + cw / 2, max(lo, hi))
        if s["value_labels"]:
            vsz = self._S("value")
            fmt = s["fmt"] if s["fmt"] is not None else (self.xfmt if hz else self.yfmt)
            lab_col = _mix(th.ink, th.muted, 0.3)
            if s["stacked"]:
                tot = {}
                for r in geom:
                    t = tot.setdefault(r["j"], [0.0, None])
                    t[0] += r["v"]
                    if r.get("outer") and r["v"] >= 0:
                        t[1] = r
                for r in geom:
                    if r["v"] == 0:
                        continue
                    seg = (r["x1"] - r["x0"]) if hz else (r["y1"] - r["y0"])
                    cross = (r["y1"] - r["y0"]) if hz else (r["x1"] - r["x0"])
                    txt = fmt_number(r["v"], fmt)
                    need = self._mw(txt, vsz)[0] + 10 * fs if hz else vsz * 1.9
                    if seg < need or cross < (vsz * 1.4 if hz else self._mw(txt, vsz)[0] + 8 * fs):
                        continue
                    ccol = s["colors"][r["i"]]
                    tcol = "#ffffff" if _lum(_rgbf(ccol)) < 0.42 else "#1a1f2b"
                    cx_ = (r["x0"] + r["x1"]) / 2 if hz else r["center"]
                    cy_ = r["center"] if hz else (r["y0"] + r["y1"]) / 2
                    self._txt(g, cx_, cy_, txt, vsz, fill=tcol, anchor="mm")
                for j, (v, r) in tot.items():
                    if r is None:
                        continue
                    if hz:
                        self._otxt(g, r["x1"] + 7 * fs, r["center"], fmt_number(v, fmt), vsz, fill=lab_col,
                                   anchor="lm")
                    else:
                        self._otxt(g, r["center"], r["y0"] - 6 * fs, fmt_number(v, fmt), vsz, fill=lab_col,
                                   anchor="mb")
                        r["lab_top"] = r["y0"] - 6 * fs - vsz * 1.2
            else:
                for r in geom:
                    txt = fmt_number(r["v"], fmt)
                    if hz:
                        x = (r.get("tip") or (r["x1"] if r["v"] >= 0 else r["x0"]))
                        if r["v"] >= 0:
                            self._otxt(g, x + 7 * fs, r["center"], txt, vsz, fill=lab_col, anchor="lm")
                        else:
                            self._otxt(g, x - 7 * fs, r["center"], txt, vsz, fill=lab_col, anchor="rm")
                    else:
                        if r["v"] >= 0:
                            y = r.get("tip", r["y0"])
                            self._otxt(g, r["center"], y - 5 * fs, txt, vsz, fill=lab_col, anchor="mb")
                            r["lab_top"] = y - 5 * fs - vsz * 1.2
                        else:
                            self._otxt(g, r["center"], r["y1"] + 5 * fs, txt, vsz, fill=lab_col, anchor="mt")

    def _cmap_stops(self, s):
        th = self.th
        cm = s["cmap"]
        bg = self._bgcol()
        if isinstance(cm, (list, tuple)):
            return list(cm)
        base = self.palette[cm] if isinstance(cm, int) else (cm or self.palette[0])
        if th.dark:
            return [_mix(bg, base, 0.08), _mix(bg, base, 0.45), base, _mix(base, "#ffffff", 0.55)]
        return [_mix(bg, base, 0.06), _mix(bg, base, 0.38), base, _mix(base, th.ink, 0.45)]

    def _draw_heatmap(self, g, s):
        fs = self.fs
        M = s["M"]
        vmin = s["vmin"] if s["vmin"] is not None else float(np.nanmin(M))
        vmax = s["vmax"] if s["vmax"] is not None else float(np.nanmax(M))
        stops = self._cmap_stops(s)
        t = (M - vmin) / max(vmax - vmin, 1e-12)
        cols = _ramp(stops, np.nan_to_num(t))
        sx, sy = self._sx, self._sy
        stepx = abs(sx.r1 - sx.r0) / max(len(self._xcats), 1)
        stepy = abs(sy.r1 - sy.r0) / max(len(self._ycats), 1)
        gp = min(s["gap"] * fs, stepx * 0.15, stepy * 0.15)
        rr = min(3.5 * fs, stepx * 0.2, stepy * 0.2)
        vsz = min(self._S("value"), stepy * 0.5, stepx * 0.36)
        fmt = s["fmt"]
        light_ink, dark_ink = (self.th.ink if self.th.dark else "#ffffff"), (self.th.background if self.th.dark
                                                                           else self.th.ink)
        if not self.th.dark:
            light_ink, dark_ink = "#ffffff", self.th.ink
        else:
            light_ink, dark_ink = "#ffffff", "#0d1117"
        for r, yl in enumerate(s["yl"]):
            cy = sy(self._ycatnum(yl))
            for c, xl in enumerate(s["xl"]):
                if not np.isfinite(M[r, c]):
                    continue
                cx = sx(self._xnum(xl))
                col = _v._hex(cols[r, c])
                g.rect(cx - stepx / 2 + gp / 2, cy - stepy / 2 + gp / 2, stepx - gp, stepy - gp, r=rr, fill=col)
                if s["annotate"] and vsz >= 7:
                    ink = dark_ink if _lum(col) > 0.36 else light_ink
                    _text(g,cx, cy, fmt_number(M[r, c], fmt), size=vsz, fill=ink, anchor="mm")

    def _draw_colorbar(self, g, s, L):
        fs = self.fs
        px0, py0, px1, py1 = self._plot
        M = s["M"]
        vmin = s["vmin"] if s["vmin"] is not None else float(np.nanmin(M))
        vmax = s["vmax"] if s["vmax"] is not None else float(np.nanmax(M))
        stops = self._cmap_stops(s)
        x = px1 + 16 * fs
        h = (py1 - py0) * 0.82
        y0 = py0 + (py1 - py0 - h) / 2
        g.rect(x, y0, 12 * fs, h, r=6 * fs,
               fill=linear_gradient([(i / (len(stops) - 1), c) for i, c in enumerate(stops[::-1])]))
        t, _ = nice_ticks(vmin, vmax, 4)
        t = [v for v in t if vmin - 1e-9 <= v <= vmax + 1e-9]
        for v, l in zip(t, _fmt_ticks(t, s["fmt"])):
            yy = y0 + h - (v - vmin) / max(vmax - vmin, 1e-12) * h
            self._txt(g, x + 12 * fs + 8 * fs, yy, l, self._S("tick") * 0.92, fill=self.th.muted, anchor="lm")

    def _draw_span(self, g, m, label=False):
        px0, py0, px1, py1 = self._plot
        col = m["color"] or self.th.ink
        halo = self._bgcol()
        if m["axis"] == "x":
            a, b = self._sx(self._xnum(m["a"])), self._sx(self._xnum(m["b"]))
            a, b = max(min(a, b), px0), min(max(a, b), px1)
            if label:
                self._otxt(g, (a + b) / 2, py0 + 6 * self.fs, m["label"], self._S("value"), fill=self.th.muted,
                           anchor="mt", halo=halo)
            else:
                g.rect(a, py0, b - a, py1 - py0, fill=col, fill_opacity=m["opacity"])
        else:
            a, b = self._sy(float(m["a"])), self._sy(float(m["b"]))
            a, b = max(min(a, b), py0), min(max(a, b), py1)
            if label:
                self._otxt(g, px1 - 6 * self.fs, a + 5 * self.fs, m["label"], self._S("value"), fill=self.th.muted,
                           anchor="rt", halo=halo)
            else:
                g.rect(px0, a, px1 - px0, b - a, fill=col, fill_opacity=m["opacity"])

    def _draw_refline(self, g, m):
        fs = self.fs
        px0, py0, px1, py1 = self._plot
        col = m["color"] or _mix(self.th.ink, self._bgcol(), 0.35)
        if m["kind"] == "hline":
            sy = self._sy2 if (m["axis"] == "right" and self._sy2 is not None) else self._sy
            y = sy(float(m["v"]))
            g.line(px0, y, px1, y, stroke=col, stroke_width=m["width"] * fs, dash=m["dash"])
            self._occ.polyline([px0, px1], [y, y], 0.5)
            if m["label"]:
                self._txt(g, px1 - 4 * fs, y - 5 * fs, m["label"], self._S("value"), fill=col, anchor="rb",
                          halo=self._bgcol())
                self._occ.rect(px1 - 120 * fs, y - 22 * fs, px1, y)
        else:
            x = self._sx(self._xnum(m["v"]))
            g.line(x, py0, x, py1, stroke=col, stroke_width=m["width"] * fs, dash=m["dash"])
            self._occ.polyline([x, x], [py0, py1], 0.5)
            if m["label"]:
                self._txt(g, x + 6 * fs, py0 + 4 * fs, m["label"], self._S("value"), fill=col, anchor="lt",
                          halo=self._bgcol())
                self._occ.rect(x, py0, x + 120 * fs, py0 + 22 * fs)

    def _bar_lookup(self, key):
        """Category name or (category, series label) → (center px, top px) over bars."""
        bars = [s for s in self.series if s["kind"] == "bar" and not s["horizontal"]]
        cat, ser = (key if isinstance(key, (tuple, list)) else (key, None))
        cat = str(cat)
        best = None
        for s in bars:
            geom = s.get("_geom") or self._bar_geom(s)
            sel = [r for r in geom if r["cat"] == cat and (ser is None or s["labels"][r["i"]] == ser)]
            if sel:
                c = sum(r["center"] for r in sel) / len(sel) if ser is None else sel[0]["center"]
                top = min(min(r.get("tip", r["y0"]), r["y0"], r.get("lab_top", r["y0"])) for r in sel)
                best = (c, top, sel)
        if best is None:
            x = self._sx(self._xnum(cat))
            return x, self._plot[3], []
        return best

    def _draw_signif(self, g, sig):
        fs = self.fs
        col = _mix(self.th.ink, self._bgcol(), 0.15)
        levels = []
        vsz = self._S("value") * 1.05
        for m in sig:
            xa, ta, _ = self._bar_lookup(m["a"])
            xb, tb, _ = self._bar_lookup(m["b"])
            x0, x1 = min(xa, xb), max(xa, xb)
            # top of anything between
            top = min(ta, tb)
            for s in self.series:
                if s["kind"] == "bar" and not s["horizontal"]:
                    for r in s.get("_geom", []):
                        if x0 - 1 <= r["center"] <= x1 + 1:
                            top = min(top, r.get("tip", r["y0"]), r["y0"], r.get("lab_top", r["y0"]))
            y = top - 9 * fs
            for (a, b, yy) in levels:
                if not (x1 < a - 2 or x0 > b + 2):
                    y = min(y, yy - vsz * 1.7)
            levels.append((x0, x1, y))
            hgt = 6 * fs
            g.polyline([(x0, y + hgt), (x0, y), (x1, y), (x1, y + hgt)], stroke=col, stroke_width=1.3 * fs,
                       join="miter")
            self._occ.rect(x0, y - vsz * 1.3, x1, y + hgt, 2.0)
            stars = m["text"].strip("*") == ""
            self._txt(g, (x0 + x1) / 2, y + (3 * fs if stars else -2 * fs), m["text"], vsz * (1.45 if stars else 1),
                      fill=col, anchor="mb", bold=stars)

    def _draw_annotate(self, g, m):
        th, fs = self.th, self.fs
        sy = self._sy2 if (m["axis"] == "right" and self._sy2 is not None) else self._sy
        x, y = m["xy"]
        ax_, ay_ = self._sx(self._xnum(x)), sy(float(y))
        px0, py0, px1, py1 = self._plot
        tsz = self._S("note")
        tw, thh = self._mw(m["text"], tsz, max_width=220 * fs)
        bw, bh = tw + 20 * fs, thh + 12 * fs
        if m["xytext"] is not None:
            tx, ty = self._sx(self._xnum(m["xytext"][0])), sy(float(m["xytext"][1]))
        elif m["offset"] is not None:
            tx, ty = ax_ + m["offset"][0] * fs, ay_ + m["offset"][1] * fs
        else:
            best, bs = None, 1e18
            cands = []
            for k in (1.0, 1.7):
                for dx, dy in ((-1, -1), (1, -1), (-1.4, -0.4), (1.4, -0.4), (0, -1.2), (-1.4, 0), (1.4, 0),
                               (-1, 1), (1, 1), (0, 1.2)):
                    cands.append((dx * k, dy * k, k))
            for dx, dy, k in cands:
                cx = ax_ + dx * (bw / 2 + 30 * fs)
                cy = ay_ + dy * (bh / 2 + 30 * fs)
                bx0, by0 = cx - bw / 2, cy - bh / 2
                if bx0 < px0 or bx0 + bw > px1 or by0 < py0 - 4 * fs or by0 + bh > py1:
                    pen = 1e6
                else:
                    pen = 0
                sc = self._occ.score(bx0 - 4 * fs, by0 - 4 * fs, bx0 + bw + 4 * fs, by0 + bh + 4 * fs) + pen + k
                if sc < bs:
                    best, bs = (cx, cy), sc
            tx, ty = best
        col = m["color"] or _mix(th.ink, self._bgcol(), 0.2)
        # leader from the pill edge to the point
        dx, dy = ax_ - tx, ay_ - ty
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        k = min((bw / 2) / max(abs(ux), 1e-9), (bh / 2) / max(abs(uy), 1e-9))
        sx_, sy_ = tx + ux * k, ty + uy * k
        ex, ey = ax_ - ux * 7 * fs, ay_ - uy * 7 * fs
        if math.hypot(ex - sx_, ey - sy_) > 4 * fs:
            g.line(sx_, sy_, ex, ey, stroke=col, stroke_width=1.3 * fs, cap="round")
        g.circle(ax_, ay_, 6.5 * fs, fill=None, stroke=col, stroke_width=1.6 * fs)
        g.circle(ax_, ay_, 2.4 * fs, fill=col)
        pill = th.surface if not self.card else (_mix(th.surface, th.ink, 0.06) if th.dark else th.background)
        g.rect(tx - bw / 2, ty - bh / 2, bw, bh, r=min(bh / 2, 9 * fs), fill=pill,
               stroke=_mix(th.grid, th.ink, 0.15), stroke_width=1 * fs)
        self._txt(g, tx, ty, m["text"], tsz, fill=th.ink, anchor="mm", max_width=220 * fs, align="center")
        self._occ.rect(tx - bw / 2, ty - bh / 2, tx + bw / 2, ty + bh / 2, 3)

    # ---- spark
    def _emit_spark(self, dwg, ox, oy, pad, top=None):
        top = pad if top is None else top
        fs = self.fs
        W, H = self.w, self.h
        with dwg.group(f"{self.name} sparkline", translate=(ox, oy)) as g:
            for s in self.series:
                if s["kind"] != "spark":
                    continue
                v = s["y"]
                n = len(v)
                x = np.arange(n, dtype=float) if s["x"] is None else np.asarray([_parse_date(a) if _parse_date(a)
                                                                                 is not None else float(a)
                                                                                 for a in s["x"]])
                lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
                if s["band"]:
                    lo, hi = min(lo, s["band"][0]), max(hi, s["band"][1])
                span = (hi - lo) or 1.0
                r = 4.0 * fs if s["dot"] else 0
                x0, x1 = pad, W - pad - r - 1
                y0, y1 = top + r + 1, H - pad - 1
                px = x0 + (x - x.min()) / max(x.max() - x.min(), 1e-12) * (x1 - x0)
                py = y1 - (v - lo) / span * (y1 - y0)
                col = s["color"]
                if s["band"]:
                    b0 = y1 - (s["band"][0] - lo) / span * (y1 - y0)
                    b1 = y1 - (s["band"][1] - lo) / span * (y1 - y0)
                    g.rect(x0, min(b0, b1), x1 - x0, abs(b1 - b0), fill=col, fill_opacity=0.08)
                if s["fill"]:
                    _curve(g, px, py, s["smooth"], close_to=H - pad,
                           fill=linear_gradient([(0, col, 0.34), (1, col, 0.0)]), stroke=None)
                _curve(g, px, py, s["smooth"], stroke=col, stroke_width=s["width"] * fs, cap="round", join="round",
                       fill=None)
                if s["dot"]:
                    g.circle(px[-1], py[-1], r * 2.1, fill=col, fill_opacity=0.18)
                    g.circle(px[-1], py[-1], r, fill=col, stroke=self._bgcol(), stroke_width=1.5 * fs)

    # ---- pie
    def _emit_pie(self, dwg, ox, oy, pad, top):
        th, fs = self.th, self.fs
        s = next(s for s in self.series if s["kind"] == "pie")
        W, H = self.w, self.h
        self._emit_header(dwg, ox, oy, pad)
        vals = np.maximum(s["vals"], 0)
        tot = float(vals.sum()) or 1.0
        fr = vals / tot
        lsz = self._S("legend")
        entries = []
        for lab, f, v in zip(s["labels"], fr, vals):
            entries.append((lab, f"{f * 100:.0f}%" if s["legend_values"] is True or s["legend_values"] == "pct"
                            else fmt_number(v, s["fmt"])))
        name_w = max(self._mw(e[0], lsz)[0] for e in entries)
        val_w = max(self._mw(e[1], lsz, True)[0] for e in entries)
        row_h = lsz * 2.0
        leg_w = 14 * fs + 10 * fs + name_w + 18 * fs + val_w
        leg_h = row_h * len(entries)
        avail_h = H - top - pad
        side = self.legend not in ("none", "bottom") and (W - 2 * pad - leg_w - 28 * fs) >= avail_h * 0.7
        if self.legend == "none":
            R = min(W - 2 * pad, avail_h) / 2
            cx, cy = W / 2, top + avail_h / 2
        elif side:
            R = min((W - 2 * pad - leg_w - 32 * fs) / 2, avail_h / 2)
            gapx = (W - 2 * pad - 2 * R - leg_w) / 3
            cx = pad + gapx + R
            cy = top + avail_h / 2
            lx, ly = cx + R + gapx, cy - leg_h / 2
        else:
            R = min(W - 2 * pad, avail_h - leg_h - 18 * fs) / 2
            cx, cy = W / 2, top + R
            lx, ly = W / 2 - leg_w / 2, cy + R + 18 * fs
        R = max(R, 10)
        inner = R * s["donut"]
        sep = self._bgcol()
        with dwg.group(f"{self.name} slices", translate=(ox, oy)) as g:
            a = float(s["start"])
            for i, f in enumerate(fr):
                if f <= 0:
                    continue
                b = a + f * 360.0
                g.pie(cx, cy, R, a, b, inner=inner, fill=s["colors"][i], stroke=sep, stroke_width=2.2 * fs,
                      join="round")
                if s["pct"] and f >= 0.06:
                    mid = math.radians((a + b) / 2)
                    rr = (R + inner) / 2 if inner > 0 else R * 0.64
                    col = s["colors"][i]
                    ink = "#ffffff" if _lum(col) < 0.42 else "#111418"
                    psz = max(10.0, min(self._S("value") * 1.05, (R - inner) * 0.42))
                    _text(g,cx + rr * math.cos(mid), cy + rr * math.sin(mid), f"{f * 100:.0f}%", size=psz, fill=ink,
                           anchor="mm", weight="bold")
                a = b
            if inner > 0:
                cen = s["center"]
                if cen is None:
                    cen = (fmt_number(tot, s["fmt"]), "Total")
                if isinstance(cen, str):
                    cen = (cen, None)
                big = min(inner * 0.4, 40 * fs)
                while cen[0] and self._mw(cen[0], round(big, 2), True)[0] > inner * 1.55 and big > 8:
                    big *= 0.92
                if cen[1]:
                    self._txt(g, cx, cy - big * 0.1, cen[0], big, anchor="ms", bold=True)
                    self._txt(g, cx, cy + big * 0.32, cen[1], big * 0.42, fill=th.muted, anchor="mt")
                else:
                    self._txt(g, cx, cy, cen[0], big, anchor="mm", bold=True)
        if self.legend != "none":
            with dwg.group(f"{self.name} legend", translate=(ox, oy)) as g:
                for i, (lab, val) in enumerate(entries):
                    y = ly + i * row_h + row_h / 2
                    g.rect(lx, y - 7 * fs, 14 * fs, 14 * fs, r=4 * fs, fill=s["colors"][i])
                    self._txt(g, lx + 24 * fs, y, lab, lsz, fill=_mix(th.ink, th.muted, 0.35), anchor="lm")
                    self._txt(g, lx + leg_w, y, val, lsz, anchor="rm", bold=True)

    # ---- radar
    def _emit_radar(self, dwg, ox, oy, pad, top):
        th, fs = self.th, self.fs
        s = next(s for s in self.series if s["kind"] == "radar")
        W, H = self.w, self.h
        self._emit_header(dwg, ox, oy, pad)
        cats = s["cats"]
        n = len(cats)
        lsz = self._S("tick")
        labw = max(self._mw(c, lsz)[0] for c in cats)
        entries = [("box", c, str(k)) for k, v, c in s["ser"] if k]
        leg_h = self._legend_size(entries, True, W - 2 * pad)[1] + 14 * fs if entries and self.legend != "none" else 0
        avail_h = H - top - pad - leg_h
        R = max(10.0, min((W - 2 * pad) / 2 - labw - 14 * fs, avail_h / 2 - lsz * 1.6))
        cx, cy = W / 2, top + avail_h / 2
        levels = s["levels"]
        vmax = s["vmax"]
        if not vmax:
            t, _ = nice_ticks(0, max(float(np.nanmax(v)) for _, v, _ in s["ser"]), levels)
            vmax, levels = t[-1], len(t) - 1
        s = dict(s, levels=levels)
        ang = [math.radians(-90 + 360.0 * i / n) for i in range(n)]
        with dwg.group(f"{self.name} axes", translate=(ox, oy)) as g:
            for L in range(1, s["levels"] + 1):
                rr = R * L / s["levels"]
                g.polygon([(cx + rr * math.cos(a), cy + rr * math.sin(a)) for a in ang], fill=None,
                          stroke=th.grid, stroke_width=1.1 * fs)
            for i, a in enumerate(ang):
                g.line(cx, cy, cx + R * math.cos(a), cy + R * math.sin(a), stroke=th.grid, stroke_width=1.1 * fs)
                lx, ly = cx + (R + 12 * fs) * math.cos(a), cy + (R + 12 * fs) * math.sin(a)
                ca = math.cos(a)
                h = "m" if abs(ca) < 0.2 else ("l" if ca > 0 else "r")
                sa = math.sin(a)
                v = "m" if abs(sa) < 0.3 else ("t" if sa > 0 else "b")
                self._txt(g, lx, ly, cats[i], lsz, fill=_mix(th.ink, th.muted, 0.4), anchor=h + v)
            for L in (range(1, s["levels"]) if R >= 100 * fs else ()):
                val = vmax * L / s["levels"]
                self._txt(g, cx + 5 * fs, cy - R * L / s["levels"] - 2 * fs, fmt_number(val, s["fmt"]),
                          lsz * 0.8, fill=th.muted, anchor="lb")
        for k, v, col in s["ser"]:
            with dwg.group(f"{self.name} {k or 'series'}", translate=(ox, oy)) as g:
                pts = [(cx + R * min(max(float(x) / vmax, 0), 1.15) * math.cos(a),
                        cy + R * min(max(float(x) / vmax, 0), 1.15) * math.sin(a)) for x, a in zip(v, ang)]
                g.polygon(pts, fill=col if s["fill"] else None, fill_opacity=0.16, stroke=col,
                          stroke_width=2.4 * fs, join="round")
                for x, y in pts:
                    g.circle(x, y, 3.6 * fs, fill=col, stroke=self._bgcol(), stroke_width=1.3 * fs)
        if entries and self.legend != "none":
            with dwg.group(f"{self.name} legend", translate=(ox, oy)) as g:
                lw, lh, _, _ = self._legend_size(entries, True, W - 2 * pad)
                self._draw_legend(g, entries, (W - lw) / 2, H - pad - lh, True, W - 2 * pad)


# --------------------------------------------------------------------------------------------- KPI / header / text


class KPI(_Part):
    """KPI tile: ``label``, big ``value``, ``delta`` pill (▲ green / ▼ red), optional sparkline and icon.

    ``fmt`` formats the value (see ``fmt_number``; a string value is shown as is). ``delta``: number (shown with
    ``delta_fmt``, default "+.1%" i.e. percent points as given) or a string. ``good="up"`` (or "down" when lower is
    better). ``note``: small text after the delta ("vs last month"). ``spark``: values for the sparkline.
    ``icon``: a vector icon name. ``color``: accent for the sparkline / icon (default the first accent)."""

    def __init__(self, fig, box, label, value, delta=None, spark=None, fmt=None, delta_fmt="+.1%", good="up",
                 note=None, icon=None, color=None, card=True, font_scale=1.0, name=None):
        super().__init__(fig, box, fig._name(name or label))
        self.label, self.value, self.delta, self.spark = label, value, delta, spark
        self.fmt, self.delta_fmt, self.good, self.note, self.icon = fmt, delta_fmt, good, note, icon
        self.color = color if color is not None else self.th.accent(0)
        if isinstance(self.color, int):
            self.color = self.th.accent(PALETTE_ORDER[self.color % 8])
        self.card, self.fs = card, float(font_scale)

    def _emit(self, dwg, ox, oy):
        th, fs = self.th, self.fs
        W, H = self.w, self.h
        pad = 22 * fs
        fam = th.font["family"]
        if self.card:
            with dwg.group(f"{self.name} card", translate=(ox, oy), shadow=True) as g:
                g.rect(0, 0, W, H, r=th.radius + 6, fill=th.surface,
                       stroke=_mix(th.grid, th.surface, 0.35) if th.dark else None, stroke_width=1)
        with dwg.group(f"{self.name} value", translate=(ox, oy)) as g:
            lsz = 15 * fs
            _text(g,pad, pad, str(self.label).upper(), size=lsz, fill=th.muted, anchor="lt", letter_spacing=1.1 * fs,
                   weight="bold")
            if self.icon:
                isz = 34 * fs
                g.circle(W - pad - isz / 2, pad + isz / 2 - 6 * fs, isz / 2 + 4 * fs, fill=self.color,
                         fill_opacity=0.14)
                g.icon(self.icon, W - pad - isz / 2, pad + isz / 2 - 6 * fs, isz * 0.62, color=self.color)
            vtxt = self.value if isinstance(self.value, str) else fmt_number(self.value, self.fmt)
            vsz = min(SIZES["kpi"] * fs, H * 0.3)
            vy = pad + lsz * 1.25 + 10 * fs
            # shrink to fit width
            while _measure(vtxt, round(vsz, 2), fam, True)[0] > W - 2 * pad and vsz > 12:
                vsz *= 0.92
            _text(g,pad, vy, vtxt, size=vsz, anchor="lt", weight="bold", fill=th.ink)
            dy = vy + vsz * 1.22 + 6 * fs
            if self.delta is not None:
                if isinstance(self.delta, str):
                    dtxt = self.delta
                    up = not self.delta.strip().startswith(("-", _MINUS))
                else:
                    dtxt = fmt_number(self.delta, self.delta_fmt)
                    up = self.delta >= 0
                goodness = up if self.good == "up" else not up
                dcol = th.accent(2) if goodness else th.accent(3)
                dsz = 14 * fs
                tw = _measure(dtxt, round(dsz, 2), fam, True)[0]
                pw, ph = tw + 30 * fs, dsz * 1.75
                g.rect(pad, dy, pw, ph, r=ph / 2, fill=dcol, fill_opacity=0.16)
                _triangle(g, pad + 12 * fs, dy + ph / 2, 4.6 * fs, up, fill=dcol)
                _text(g,pad + 22 * fs, dy + ph / 2, dtxt, size=dsz, fill=dcol, anchor="lm", weight="bold")
                if self.note:
                    _text(g,pad + pw + 10 * fs, dy + ph / 2, self.note, size=dsz, fill=th.muted, anchor="lm")
        if self.spark is not None and len(self.spark) > 1:
            v = np.asarray(self.spark, np.float64)
            top = dy + 28 * fs + 14 * fs if self.delta is not None else vy + vsz * 1.3 + 10 * fs
            y0, y1 = max(top, H * 0.56), H - pad * 0.75
            x0, x1 = pad, W - pad - 5 * fs
            bottom = H - 2 * fs
            if y1 - y0 < 34 * fs:                          # short, wide tile: sparkline on the right
                vw = _measure(vtxt, round(vsz, 2), fam, True)[0]
                x0 = max(pad + vw + 28 * fs, W * 0.5)
                y0 = pad + (46 * fs if self.icon else 8 * fs)
                y1 = H - pad * 0.9
                bottom = y1 + 6 * fs
            if y1 - y0 > 12 * fs and x1 - x0 > 30 * fs:
                with dwg.group(f"{self.name} sparkline", translate=(ox, oy)) as g:
                    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
                    span = (hi - lo) or 1.0
                    px = np.linspace(x0, x1, len(v))
                    py = y1 - (v - lo) / span * (y1 - y0)
                    _curve(g, px, py, True, close_to=bottom,
                           fill=linear_gradient([(0, self.color, 0.32), (1, self.color, 0.0)]), stroke=None)
                    _curve(g, px, py, True, stroke=self.color, stroke_width=2.4 * fs, cap="round", join="round",
                           fill=None)
                    g.circle(px[-1], py[-1], 9 * fs, fill=self.color, fill_opacity=0.18)
                    g.circle(px[-1], py[-1], 4.2 * fs, fill=self.color, stroke=th.surface if self.card else
                             th.background, stroke_width=1.6 * fs)


class Header(_Part):
    """Header band: bold ``title``, muted ``subtitle``, ``right`` chips (a string or list of strings) and a divider.
    ``box`` defaults to the top 11% of the figure. ``icon``: a vector icon in an accent tile left of the title."""

    def __init__(self, fig, box, title, subtitle=None, right=None, icon=None, divider=True, font_scale=1.0,
                 name=None):
        super().__init__(fig, box or (0.025, 0.02, 0.975, 0.12), fig._name(name or "header"))
        self.title, self.subtitle, self.icon, self.divider, self.fs = title, subtitle, icon, divider, float(font_scale)
        self.right = [right] if isinstance(right, str) else list(right or [])

    def _emit(self, dwg, ox, oy):
        th, fs = self.th, self.fs
        W, H = self.w, self.h
        fam = th.font["family"]
        tsz = min(36 * fs, H * 0.42)
        ssz = tsz * 0.5
        with dwg.group(f"{self.name}", translate=(ox, oy)) as g:
            x = 0.0
            block = tsz * 1.2 + (ssz * 1.5 if self.subtitle else 0)
            ty = (H - block) / 2 - (2 * fs if self.divider else 0)
            if self.icon:
                s = block * 0.98
                g.rect(0, ty + (block - s) / 2, s, s, r=s * 0.26, fill=th.accent(0))
                g.icon(self.icon, s / 2, ty + block / 2, s * 0.56, color="#ffffff")
                x = s + 18 * fs
            _text(g,x, ty, self.title, size=tsz, weight="bold", anchor="lt", fill=th.ink)
            if self.subtitle:
                _text(g,x, ty + tsz * 1.25, self.subtitle, size=ssz, anchor="lt", fill=th.muted)
            cx = W
            csz = 15 * fs
            for i, chip in enumerate(reversed(self.right)):
                tw = _measure(chip, round(csz, 2), fam, i == 0)[0]
                cw, ch = tw + 30 * fs, csz * 2.3
                cx -= cw
                first = i == 0
                g.rect(cx, (H - ch) / 2 - (2 * fs if self.divider else 0), cw, ch, r=ch / 2,
                       fill=th.accent(0) if first else th.surface, fill_opacity=0.16 if first else 1,
                       stroke=None if first else th.grid, stroke_width=1)
                _text(g,cx + cw / 2, H / 2 - (2 * fs if self.divider else 0), chip, size=csz, anchor="mm",
                       fill=th.accent(0) if first else th.muted, weight="bold" if first else None)
                cx -= 10 * fs
            if self.divider:
                g.line(0, H - 1, W, H - 1, stroke=th.grid, stroke_width=1.2 * fs)


class _TextPart(_Part):
    def __init__(self, fig, box, text, size, color, weight, align, italic, name):
        super().__init__(fig, box, fig._name(name or "text"))
        self.text, self.size, self.color, self.weight, self.align, self.italic = (text, size, color, weight, align,
                                                                                  italic)

    def _emit(self, dwg, ox, oy):
        th = self.th
        a = {"left": "lt", "center": "mt", "right": "rt"}[self.align]
        x = {"left": 0, "center": self.w / 2, "right": self.w}[self.align]
        with dwg.group(self.name, translate=(ox, oy)) as g:
            _text(g,x, 0, self.text, size=self.size, fill=self.color or th.muted, anchor=a, max_width=self.w,
                   align=self.align, weight=self.weight, style="italic" if self.italic else "normal",
                   line_spacing=1.35)


# --------------------------------------------------------------------------------------------- quick


def _series_dict(data):
    """dict {label: list} with all list values?"""
    return isinstance(data, dict) and data and all(isinstance(v, (list, tuple, np.ndarray)) for v in data.values())


def quick(cv, kind, data, title=None, box=None, theme="light", log=None, x=None, **kw):
    """One call → a finished chart drawn on ``cv``; returns the Axes (``.save_svg(path)`` works).

    ``kind``: line, area, bar, barh, stacked, scatter, hist, heatmap, pie, donut, radar, spark.
    ``data`` (plain lists / dicts):
      line / area: ``[3, 5, 4]`` · ``{"2026-01": 3, "2026-02": 5}`` · ``{"North": [..], "South": [..]}`` with ``x=``
      bar / barh / stacked: ``{"A": 3, "B": 5}`` · ``{"2025": [..], "2026": [..]}`` with ``x=`` categories
      scatter: ``{"x": [..], "y": [..]}`` (optional "size", "color") or a list of (x, y)
      hist: list of values (``bins=``) · heatmap: matrix (``x=`` column labels, ``y=`` row labels)
      pie / donut: ``{"label": value}`` · radar: ``{"series": [..]}`` with ``x=`` axis names
    Other keywords go to ``axes()`` (subtitle, xlabel, ylabel, yfmt, legend, card, ...) or the series call."""
    ax_keys = {"subtitle", "xlabel", "ylabel", "ylabel_right", "xscale", "yscale", "xfmt", "yfmt", "yfmt_right",
               "xlim", "ylim", "ylim_right", "xticks", "yticks", "grid", "frame", "legend", "card", "panel",
               "font_scale", "palette", "pad", "background", "name", "zero"}
    k = str(kind).lower()
    _check_choice("quick", "kind", k, ["line", "area", "bar", "barh", "stacked", "column", "scatter", "hist",
                                       "heatmap", "pie", "donut", "radar", "spark", "sparkline"])
    method = {"barh": "bar", "stacked": "bar", "column": "bar", "donut": "pie", "spark": "sparkline"}.get(k, k)
    kw = _reject_unknown("quick", kw, sorted(ax_keys) + _signature_options(getattr(Axes, method))
                         + ["y", "categories"])
    akw = {k: kw.pop(k) for k in list(kw) if k in ax_keys}
    ax = axes(cv, box=box, theme=theme, title=title, **akw)
    k = str(kind).lower()
    if k in ("line", "area"):
        fn = ax.line if k == "line" else ax.area
        if _series_dict(data):
            n = len(next(iter(data.values())))
            fn(x if x is not None else list(range(n)), dict(data), **kw)
        elif isinstance(data, dict):
            fn(list(data.keys()), list(data.values()), **kw)
        else:
            fn(x if x is not None else list(range(len(data))), list(data), **kw)
    elif k in ("bar", "barh", "stacked", "column"):
        if k == "barh":
            kw["horizontal"] = True
        if k == "stacked":
            kw["stacked"] = True
        if _series_dict(data):
            n = len(next(iter(data.values())))
            ax.bar(x if x is not None else [str(i + 1) for i in range(n)], dict(data), **kw)
        elif isinstance(data, dict):
            ax.bar(list(data.keys()), list(data.values()), **kw)
        else:
            ax.bar(x if x is not None else [str(i + 1) for i in range(len(data))], list(data), **kw)
    elif k == "scatter":
        if isinstance(data, dict):
            ax.scatter(data["x"], data["y"], size=data.get("size"), color=data.get("color"), **kw)
        else:
            pts = np.asarray(data, np.float64)
            ax.scatter(pts[:, 0], pts[:, 1], **kw)
    elif k == "hist":
        ax.hist(data, **kw)
    elif k == "heatmap":
        ax.heatmap(data, xlabels=x, ylabels=kw.pop("y", None), **kw)
    elif k in ("pie", "donut"):
        if k == "donut":
            kw.setdefault("donut", 0.62)
        ax.pie(data, **kw)
    elif k == "radar":
        cats = x if x is not None else kw.pop("categories")
        ax.radar(cats, data, **kw)
    elif k in ("spark", "sparkline"):
        ax.frame, ax.grid, ax.legend = "none", False, "none"
        ax.sparkline(data, **kw)
    else:
        raise ValueError("kind must be line, area, bar, barh, stacked, scatter, hist, heatmap, pie, donut, radar or "
                         "spark")
    ax.draw(log=log)
    return ax


# unknown options raise a helpful TypeError; colour / tags / w ... spellings are accepted
for _c in (Figure, Axes, KPI, Header, _TextPart):
    _strict_methods(_c)

# ---- scene manifest: each chart part records its title and series (pixelcanvas.manifest) ----
from . import manifest as _manifest  # noqa: E402

_MONTHS = {m.lower() for m in ("January February March April May June July August September October November "
                                "December Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec").split()}
_WEEKDAYS = {d.lower() for d in "Monday Tuesday Wednesday Thursday Friday Saturday Sunday Mon Tue Wed Thu Fri Sat "
                                "Sun".split()}


def _unit_of(xs, kind):
    """What one data point of a series is, as a request would count it: month / day / year / quarter / week /
    hour, else bar / point."""
    xs = [x for x in (xs or []) if x is not None]
    if xs and all(isinstance(x, str) for x in xs):
        ms = [_DATE_RE.match(x.strip()) for x in xs]
        if all(ms):
            if all(m.group(3) for m in ms):
                return "day"
            if all(m.group(2) for m in ms):
                return "month"
            return "year"
        low = [x.strip().lower() for x in xs]
        if all(x in _MONTHS for x in low):
            return "month"
        if all(x in _WEEKDAYS for x in low):
            return "day"
        if all(re.fullmatch(r"q[1-4]( \d{2,4})?|\d{4} q[1-4]", x) for x in low):
            return "quarter"
        if all(re.fullmatch(r"(w|wk|week) ?\d{1,2}", x) for x in low):
            return "week"
        if all(re.fullmatch(r"\d{1,2}(:\d\d)? ?(am|pm|h)?", x) for x in low):
            return "hour"
    if xs and all(isinstance(x, (int, np.integer)) and 1900 <= int(x) <= 2100 for x in xs):
        return "year"
    return {"bar": "bar", "hist": "bar"}.get(kind, "point")


_TYPE_ALIASES = {"line": ["line chart", "line graph", "trend", "graph"], "area": ["area chart", "trend", "graph"],
                 "bar": ["bar chart", "bars", "column chart", "bar graph"], "hist": ["histogram"],
                 "scatter": ["scatter plot", "scatter"], "heatmap": ["heatmap", "heat map"],
                 "pie": ["pie chart", "pie"], "donut": ["donut", "donut chart", "doughnut", "ring chart"],
                 "radar": ["radar chart", "spider chart"], "spark": ["sparkline"]}


def _by_word(title):
    m = re.search(r"\bby ([A-Za-z][A-Za-z\-]+)", str(title or ""))
    return [m.group(1).lower()] if m else []


def _part_extras(p, crop, title):
    """Aliases for one chart part and extra entries: data points with a count, marker lines."""
    if isinstance(p, KPI):
        return ["kpi", "kpi tile", "tile", "metric", "stat", "chart"], []
    if isinstance(p, Header):
        return ["header", "dashboard header", "title"], []
    if not isinstance(p, Axes):
        return ["caption", "note"], []
    al, extra = ["chart"], []
    kinds = []
    for s in p.series:
        k = s.get("kind")
        if k == "pie" and s.get("donut", 0) > 0:
            k = "donut"
        if k not in kinds:
            kinds.append(k)
        al += _TYPE_ALIASES.get(k, [])
    for s in p.series:
        k = s.get("kind")
        if k in ("line", "area", "bar", "scatter"):
            xs = s.get("cats") if k == "bar" else s.get("x")
            n = len(xs or [])
            unit = _unit_of(xs, k)
        elif k == "pie":
            n, unit = len(s.get("vals", [])), "slice"
        else:
            continue
        if n < 2:
            continue
        aliases = [unit + "s", "data point"] + _by_word(title)
        if unit == "slice":
            aliases += ["segment", "wedge", "share"]
        if unit == "bar":
            aliases += ["column", "category"]
        meta = {"count": int(n), "see_through": True, "chart": str(title or p.name)}
        if s.get("label"):
            meta["series"] = [str(s["label"])]
        extra.append(dict(kind="object", name=unit, crop=crop, alias=aliases, meta=meta))
        if k != "bar" or len(s.get("vals", [])) == 1:
            break                                        # one entry per chart: the first series sets the count
    for m in p.marks:
        if m.get("kind") in ("vline", "hline"):
            lab = m.get("label") or "marker"
            line = "vertical line" if m["kind"] == "vline" else "horizontal line"
            extra.append(dict(kind="object", name=str(lab), crop=crop,
                              alias=["marker", "reference line", line, "date", "event", "target"],
                              meta={"see_through": True, "chart": str(title or p.name)}))
    return al, extra


def _after_chart(cap, args, res):
    """chart parts: the title / series entry from manifest.after_chart_part, plus chart-type aliases ("donut",
    "bar chart", "KPI tile"), one data-point entry with ``count`` (12 monthly points -> "month" x12) and the marker
    lines."""
    out = _manifest.after_chart_part(cap, args, res)
    if not out:
        return out
    part = args.get("self")
    parts = [part] if not hasattr(part, "parts") else list(part.parts)
    names = {}
    for p in parts:
        names.setdefault(str(getattr(p, "title", None) or getattr(p, "label", None) or p.name), p)
    full = []
    for e in out:
        p = names.get(e.get("name"))
        full.append(e)
        if p is None:
            continue
        al, extra = _part_extras(p, e.get("crop"), e.get("name"))
        own = e.get("alias")
        own = [own] if isinstance(own, str) else list(own or [])
        e["alias"] = own + [a for a in al if a not in own]
        full += extra
    return full


_manifest.wrap_method(_Part, "draw", "chart", after=_after_chart,
                      cv_of=lambda a: a.get("cv") if a.get("cv") is not None else a["self"].fig.cv)
_manifest.wrap_method(Figure, "draw", "chart", after=_after_chart,
                      cv_of=lambda a: a.get("cv") if a.get("cv") is not None else a["self"].cv)
