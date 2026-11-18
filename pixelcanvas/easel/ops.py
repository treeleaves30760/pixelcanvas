"""Paint operations (the unit of every change to a layer) and their exact, compact JSON form.

An op is a :class:`Stroke` (a brush dragged along a path) or a :class:`Fill` (an area filled with a colour or a
linear gradient).  ``apply_ops`` paints a list of ops onto an RGBA float layer; the rendering lives in
``brush.py``.

JSON form: arrays are stored as base64 of their raw little-endian bytes (float32 for coordinates and widths, uint8
for per-point colours), so ``ops_from_jsonable(ops_to_jsonable(ops))`` reproduces every array bit for bit.  Plain
JSON lists are also accepted on input, so hand-written op files work too.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Iterator, Optional, Union

import numpy as np

__all__ = [
    "Stroke", "Fill", "Blend", "Spray", "Op", "apply_ops", "apply_ops_progressive", "ops_to_jsonable", "ops_from_jsonable",
    "mask_to_rle", "rle_to_mask", "op_bbox", "check_mark_rules", "mark_area", "stroke_length", "op_anchor",
    "owned_ops", "owned_stack", "is_partial", "Retouch", "mark_level", "level_counts", "LEVEL_NAMES",
    "Streak", "STREAK_STATION",
]

STREAK_STATION = 2.0    # px: mean spacing of a streak's colour stations at least this (colour changes along a stroke,
                        # not pixel by pixel)


def _as_color(c):
    """(r,g,b) tuple of floats 0..255, or [N,3] uint8 array (per point)."""
    if isinstance(c, np.ndarray) and c.ndim == 2:
        return np.clip(np.rint(c), 0, 255).astype(np.uint8)
    if isinstance(c, str):
        s = c.lstrip("#")
        return (float(int(s[0:2], 16)), float(int(s[2:4], 16)), float(int(s[4:6], 16)))
    c = tuple(float(v) for v in np.asarray(c, dtype=np.float64).reshape(-1)[:3])
    return c


@dataclass(eq=False)
class Stroke:
    """A brush stroke along ``pts``.

    pts: [N,2] float32 pixel coordinates (x, y), N >= 1, pixel centres at +0.5.
    width: diameter in px, a float or [N] float32 per point.  With a constant width the brush preset applies its
        own pressure taper at the ends (pen, pencil); with per-point widths the caller controls the taper.
    color: (r,g,b) 0..255 or [N,3] uint8 per point (the colour at each pixel is that of the nearest path point).
    opacity: of the whole stroke; overlapping parts of one stroke do not build up.
    brush: key into ``brush.PRESETS``.
    seed: texture / jitter seed (replay is exact).
    erase: remove alpha instead of adding paint.
    own: the mark belongs to the region it was made for and is painted whole, across the edge of a clip mask
        (a stage tool run on a region draws the shapes and lines anchored in it, each one complete).
    """
    pts: np.ndarray
    width: Union[np.ndarray, float] = 2.0
    color: Union[tuple, np.ndarray] = (0.0, 0.0, 0.0)
    opacity: float = 1.0
    brush: str = "pen"
    seed: int = 0
    erase: bool = False
    own: bool = False

    def __post_init__(self):
        p = np.asarray(self.pts, dtype=np.float32).reshape(-1, 2)
        if len(p) == 0:
            raise ValueError("Stroke needs at least one point")
        self.pts = np.ascontiguousarray(p)
        w = self.width
        if isinstance(w, np.ndarray) and w.ndim >= 1 and w.size > 1:
            w = np.ascontiguousarray(w.astype(np.float32).reshape(-1))
            if len(w) != len(p):
                raise ValueError(f"per-point width has {len(w)} values for {len(p)} points")
        else:
            w = float(np.asarray(w).reshape(-1)[0])
        self.width = w
        c = self.color
        if isinstance(c, np.ndarray) and c.ndim == 2:
            c = _as_color(c)
            if len(c) != len(p):
                raise ValueError(f"per-point colour has {len(c)} rows for {len(p)} points")
        else:
            c = _as_color(c)
        self.color = c
        self.opacity = float(self.opacity)
        self.brush = str(self.brush)
        self.seed = int(self.seed)
        self.erase = bool(self.erase)
        self.own = bool(self.own)


@dataclass(eq=False)
class Fill:
    """An area: polygons (outer rings, even-odd rule for holes) or a run-length encoded mask, filled with a colour
    or, when ``color2``, ``p0`` and ``p1`` are given, with a linear gradient from ``color`` at p0 to ``color2`` at p1.
    ``feather`` blurs the area's edge by that many px."""
    polygons: Optional[list] = None
    mask_rle: Optional[dict] = None
    color: tuple = (0.0, 0.0, 0.0)
    opacity: float = 1.0
    feather: float = 0.0
    color2: Optional[tuple] = None
    p0: Optional[tuple] = None
    p1: Optional[tuple] = None
    erase: bool = False
    radial: bool = False                # with p0/p1: radial gradient, centre p0, color2 reached at |p1 - p0|
    own: bool = False                   # painted whole across a clip mask's edge (see Stroke.own)

    def __post_init__(self):
        self.radial = bool(self.radial)
        self.own = bool(self.own)
        if self.polygons is not None:
            self.polygons = [np.ascontiguousarray(np.asarray(p, dtype=np.float32).reshape(-1, 2))
                             for p in self.polygons]
        if self.polygons is None and self.mask_rle is None:
            raise ValueError("Fill needs polygons or mask_rle")
        self.color = _as_color(self.color)
        if self.color2 is not None:
            self.color2 = _as_color(self.color2)
        if self.p0 is not None:
            self.p0 = tuple(float(v) for v in self.p0)
        if self.p1 is not None:
            self.p1 = tuple(float(v) for v in self.p1)
        self.opacity = float(self.opacity)
        self.feather = float(self.feather)
        self.erase = bool(self.erase)


@dataclass(eq=False)
class Blend:
    """A blending pass (a soft dry brush or blending stump worked over paint already down): inside the area the
    layer's own paint moves toward its local mean over ``radius`` px, except across colour steps larger than
    ``tol`` (0..255 per channel), which stay crisp.  It reads only the layer it is applied to.  ``strength`` 0..1;
    ``feather`` softens the area's edge as for a Fill."""
    polygons: Optional[list] = None
    mask_rle: Optional[dict] = None
    radius: float = 3.0
    tol: float = 12.0
    strength: float = 1.0
    feather: float = 1.0
    own: bool = False
    erase = False
    opacity = 1.0
    color2 = None
    p0 = None
    p1 = None
    radial = False
    color = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.polygons is not None:
            self.polygons = [np.ascontiguousarray(np.asarray(p, dtype=np.float32).reshape(-1, 2))
                             for p in self.polygons]
        if self.polygons is None and self.mask_rle is None:
            raise ValueError("Blend needs polygons or mask_rle")
        self.radius = float(max(0.5, self.radius))
        self.tol = float(max(0.5, self.tol))
        self.strength = float(np.clip(self.strength, 0.0, 1.0))
        self.feather = float(self.feather)
        self.own = bool(self.own)


@dataclass(eq=False)
class Spray:
    """An airbrush pass: soft round dabs sprayed inside a frisket (the area masked off for it, as an airbrush artist
    cuts a mask from film).  Each dab is one mark: one colour, a Gaussian-soft round footprint of diameter
    ``size`` (about 2 sigma each side of the centre), its own flow ``opacity``; overlapping dabs build up tone.
    pts: [N,2] dab centres.  size: [N] or float, diameter px (>= 3).  colors: [N,3] uint8, one colour per dab.
    opacity: [N] or float, peak opacity of each dab.  polygons / mask_rle: the frisket (even-odd), or both None for
    no frisket.  feather: px of soft frisket edge."""
    pts: np.ndarray = None
    size: Union[np.ndarray, float] = 8.0
    colors: np.ndarray = None
    opacity: Union[np.ndarray, float] = 0.3
    polygons: Optional[list] = None
    mask_rle: Optional[dict] = None
    feather: float = 0.0
    seed: int = 0
    own: bool = False
    erase = False
    color2 = None
    p0 = None
    p1 = None
    radial = False

    def __post_init__(self):
        p = np.ascontiguousarray(np.asarray(self.pts, dtype=np.float32).reshape(-1, 2))
        if len(p) == 0:
            raise ValueError("Spray needs at least one dab")
        self.pts = p
        n = len(p)
        s = np.asarray(self.size, dtype=np.float32).reshape(-1)
        self.size = np.ascontiguousarray(np.broadcast_to(s, (n,)) if s.size == 1 else s)
        c = np.asarray(self.colors)
        if c.ndim == 1:
            c = np.broadcast_to(c.reshape(1, 3), (n, 3))
        self.colors = np.ascontiguousarray(np.clip(np.rint(c), 0, 255).astype(np.uint8).reshape(-1, 3))
        o = np.asarray(self.opacity, dtype=np.float32).reshape(-1)
        self.opacity = np.ascontiguousarray(np.broadcast_to(o, (n,)) if o.size == 1 else o)
        if len(self.size) != n or len(self.colors) != n or len(self.opacity) != n:
            raise ValueError("Spray: pts, size, colors and opacity need one entry per dab")
        if self.polygons is not None:
            self.polygons = [np.ascontiguousarray(np.asarray(q, dtype=np.float32).reshape(-1, 2))
                             for q in self.polygons]
        self.feather = float(self.feather)
        self.seed = int(self.seed)
        self.own = bool(self.own)

    @property
    def color(self):
        return tuple(float(v) for v in np.median(self.colors, 0))


@dataclass(eq=False)
class Retouch:
    """Retouching marks (detail levels 1 and 0): hard-edged, axis-aligned marks laid pixel-exactly, each one colour.
    Mark k covers ``n[k]`` px along its axis (``axis`` 0 = horizontal, 1 = vertical) and ``size[k]`` px across,
    starting at pixel (``x[k]``, ``y[k]``): a 1 px hairline of any length >= 2 px, a 2 px dab (n = size = 2), or a
    1 px dab (n = size = 1, the retouching brush).  One op holds many marks (as a Spray holds many dabs); later marks
    lie over earlier ones.  colors: [N,3] uint8, one eyedropped colour per mark."""
    x: np.ndarray = None
    y: np.ndarray = None
    n: np.ndarray = None
    colors: np.ndarray = None
    axis: np.ndarray = None
    size: np.ndarray = None
    own: bool = False
    erase = False
    opacity = 1.0
    color2 = None
    p0 = None
    p1 = None
    radial = False
    polygons = None
    mask_rle = None
    feather = 0.0

    def __post_init__(self):
        self.x = np.ascontiguousarray(np.asarray(self.x, np.int32).reshape(-1))
        k = len(self.x)
        if k == 0:
            raise ValueError("Retouch needs at least one mark")
        self.y = np.ascontiguousarray(np.asarray(self.y, np.int32).reshape(-1))
        n = np.ones(k, np.int32) if self.n is None else np.asarray(self.n, np.int32).reshape(-1)
        self.n = np.ascontiguousarray(np.broadcast_to(n, (k,)) if n.size == 1 else n)
        a = np.zeros(k, np.uint8) if self.axis is None else np.asarray(self.axis, np.uint8).reshape(-1)
        self.axis = np.ascontiguousarray(np.broadcast_to(a, (k,)) if a.size == 1 else a)
        z = np.ones(k, np.uint8) if self.size is None else np.asarray(self.size, np.uint8).reshape(-1)
        self.size = np.ascontiguousarray(np.broadcast_to(z, (k,)) if z.size == 1 else z)
        c = np.asarray(self.colors)
        if c.ndim == 1:
            c = np.broadcast_to(c.reshape(1, 3), (k, 3))
        self.colors = np.ascontiguousarray(np.clip(np.rint(c), 0, 255).astype(np.uint8).reshape(-1, 3))
        if not (len(self.y) == len(self.n) == len(self.axis) == len(self.size) == len(self.colors) == k):
            raise ValueError("Retouch: x, y, n, axis, size and colors need one entry per mark")
        if (self.n < 1).any() or (self.size < 1).any():
            raise ValueError("Retouch: marks are at least 1 px")
        self.own = bool(self.own)

    @property
    def color(self):
        return tuple(float(v) for v in np.median(self.colors, 0))

    def pixels(self):
        """(xs, ys, mark index) of every pixel the marks cover, in mark order."""
        k = len(self.x)
        cnt = self.n.astype(np.int64) * self.size.astype(np.int64)
        idx = np.repeat(np.arange(k), cnt)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        j = np.arange(int(cnt.sum())) - np.repeat(start, cnt)
        ln = self.n.astype(np.int64)[idx]
        along, across = j % ln, j // ln
        hor = self.axis[idx] == 0
        xs = self.x[idx].astype(np.int64) + np.where(hor, along, across)
        ys = self.y[idx].astype(np.int64) + np.where(hor, across, along)
        return xs, ys, idx


@dataclass(eq=False)
class Streak:
    """Thin brush strokes whose colour and width change along the path (a loaded brush laying fine detail).

    Stroke k runs through ``n[k]`` stations (>= 2), stored one after another in ``pts`` [M,2] (pixel coordinates,
    centres at +0.5), each with its own colour ``colors`` [M,3] uint8 and width ``width`` [M] px; between stations
    position, colour and width are interpolated linearly, so one stroke carries a smooth change of colour along it.
    The edge is anti-aliased (coverage falls off over half a pixel either side of the width).  One op holds many
    strokes, as a Spray holds many dabs; strokes of one op that overlap mix like wet paint (the colour is the
    coverage-weighted mean) and the op is then laid over the layer."""
    pts: np.ndarray = None
    n: np.ndarray = None
    colors: np.ndarray = None
    width: np.ndarray = None
    own: bool = False
    erase = False
    opacity = 1.0
    color2 = None
    p0 = None
    p1 = None
    radial = False
    polygons = None
    mask_rle = None
    feather = 0.0

    def __post_init__(self):
        self.pts = np.ascontiguousarray(np.asarray(self.pts, np.float32).reshape(-1, 2))
        m = len(self.pts)
        self.n = np.ascontiguousarray(np.asarray(self.n, np.int32).reshape(-1))
        if not len(self.n):
            raise ValueError("Streak needs at least one stroke")
        if int(self.n.sum()) != m:
            raise ValueError(f"Streak: n sums to {int(self.n.sum())} stations for {m} points")
        if (self.n < 2).any():
            raise ValueError("Streak: a stroke has at least 2 stations")
        c = np.asarray(self.colors)
        if c.ndim == 1:
            c = np.broadcast_to(c.reshape(1, 3), (m, 3))
        self.colors = np.ascontiguousarray(np.clip(np.rint(c), 0, 255).astype(np.uint8).reshape(-1, 3))
        w = np.asarray(self.width, np.float32).reshape(-1)
        self.width = np.ascontiguousarray(np.broadcast_to(w, (m,)) if w.size == 1 else w)
        if len(self.colors) != m or len(self.width) != m:
            raise ValueError("Streak: pts, colors and width need one entry per station")
        self.own = bool(self.own)

    @property
    def color(self):
        return tuple(float(v) for v in np.median(self.colors, 0))

    def starts(self):
        return np.concatenate([[0], np.cumsum(self.n)[:-1]]).astype(np.int64)

    def lengths(self):
        """Path length of each stroke (px)."""
        d = np.sqrt((np.diff(self.pts.astype(np.float64), axis=0) ** 2).sum(1))
        seg_mark = np.repeat(np.arange(len(self.n)), self.n)[1:]
        same = seg_mark == np.repeat(np.arange(len(self.n)), self.n)[:-1]
        return np.bincount(seg_mark[same], weights=d[same], minlength=len(self.n))

    def mean_widths(self):
        mk = np.repeat(np.arange(len(self.n)), self.n)
        return np.bincount(mk, weights=self.width.astype(np.float64), minlength=len(self.n)) / self.n


Op = Union[Stroke, Fill, Blend, Spray, Retouch, Streak]


# ----------------------------------------------------------------------------------------------- run-length masks
def mask_to_rle(mask: np.ndarray, bbox: bool = True) -> dict:
    """Encode a boolean mask [H,W] as {"size":[H,W], "box":[x0,y0,x1,y1], "counts":[...]}: alternating run lengths
    (row-major, starting with a run of False) of the mask cropped to ``box``."""
    m = np.asarray(mask).astype(bool)
    H, W = m.shape
    if bbox and m.any():
        ys = np.flatnonzero(m.any(1))
        xs = np.flatnonzero(m.any(0))
        x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    flat = m[y0:y1, x0:x1].reshape(-1).astype(np.int8)
    if flat.size == 0:
        return {"size": [H, W], "box": [x0, y0, x1, y1], "counts": []}
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return {"size": [H, W], "box": [x0, y0, x1, y1], "counts": [int(r) for r in runs]}


def rle_to_mask(rle: dict) -> np.ndarray:
    H, W = rle["size"]
    x0, y0, x1, y1 = rle.get("box", [0, 0, W, H])
    counts = np.asarray(rle["counts"], dtype=np.int64)
    out = np.zeros((H, W), bool)
    n = (x1 - x0) * (y1 - y0)
    if n == 0 or counts.size == 0:
        return out
    vals = (np.arange(counts.size) % 2).astype(bool)
    flat = np.repeat(vals, counts)
    if flat.size < n:
        flat = np.concatenate([flat, np.zeros(n - flat.size, bool)])
    out[y0:y1, x0:x1] = flat[:n].reshape(y1 - y0, x1 - x0)
    return out


# ----------------------------------------------------------------------------------------------- JSON round trip
def _enc(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()).decode("ascii")


def _dec(v, dtype, shape_tail=()):
    if isinstance(v, str):
        a = np.frombuffer(base64.b64decode(v), dtype=np.dtype(dtype).newbyteorder("<")).astype(dtype)
    else:
        a = np.asarray(v, dtype=dtype)
    if shape_tail:
        a = a.reshape((-1,) + tuple(shape_tail))
    return np.array(a)  # writable copy


def ops_to_jsonable(ops) -> list:
    out = []
    for op in ops:
        if isinstance(op, Stroke):
            d = {"op": "stroke", "pts": _enc(op.pts, np.float32)}
            d["w"] = _enc(op.width, np.float32) if isinstance(op.width, np.ndarray) else op.width
            d["c"] = _enc(op.color, np.uint8) if isinstance(op.color, np.ndarray) else list(op.color)
            if op.opacity != 1.0:
                d["o"] = op.opacity
            d["b"] = op.brush
            if op.seed:
                d["s"] = op.seed
            if op.erase:
                d["e"] = 1
            if op.own:
                d["own"] = 1
        elif isinstance(op, Fill):
            d = {"op": "fill", "c": list(op.color)}
            if op.polygons is not None:
                d["poly"] = [_enc(p, np.float32) for p in op.polygons]
            if op.mask_rle is not None:
                d["rle"] = op.mask_rle
            if op.opacity != 1.0:
                d["o"] = op.opacity
            if op.feather:
                d["f"] = op.feather
            if op.color2 is not None:
                d["c2"] = list(op.color2)
            if op.p0 is not None:
                d["p0"] = list(op.p0)
            if op.p1 is not None:
                d["p1"] = list(op.p1)
            if op.radial:
                d["rad"] = 1
            if op.erase:
                d["e"] = 1
            if op.own:
                d["own"] = 1
        elif isinstance(op, Blend):
            d = {"op": "blend", "r": op.radius, "tol": op.tol}
            if op.polygons is not None:
                d["poly"] = [_enc(p, np.float32) for p in op.polygons]
            if op.mask_rle is not None:
                d["rle"] = op.mask_rle
            if op.strength != 1.0:
                d["k"] = op.strength
            d["f"] = op.feather
            if op.own:
                d["own"] = 1
        elif isinstance(op, Spray):
            d = {"op": "spray", "pts": _enc(op.pts, np.float32), "sz": _enc(op.size, np.float32),
                 "c": _enc(op.colors, np.uint8), "o": _enc(op.opacity, np.float32)}
            if op.polygons is not None:
                d["poly"] = [_enc(p, np.float32) for p in op.polygons]
            if op.mask_rle is not None:
                d["rle"] = op.mask_rle
            if op.feather:
                d["f"] = op.feather
            if op.seed:
                d["s"] = op.seed
            if op.own:
                d["own"] = 1
        elif isinstance(op, Retouch):
            d = {"op": "retouch", "x": _enc(op.x, np.int32), "y": _enc(op.y, np.int32), "n": _enc(op.n, np.int32),
                 "a": _enc(op.axis, np.uint8), "sz": _enc(op.size, np.uint8), "c": _enc(op.colors, np.uint8)}
            if op.own:
                d["own"] = 1
        elif isinstance(op, Streak):
            d = {"op": "streak", "pts": _enc(op.pts, np.float32), "n": _enc(op.n, np.int32),
                 "c": _enc(op.colors, np.uint8), "w": _enc(op.width, np.float32)}
            if op.own:
                d["own"] = 1
        else:
            raise TypeError(f"not an op: {type(op).__name__}")
        out.append(d)
    return out


def ops_from_jsonable(data) -> list:
    ops = []
    for d in data:
        kind = d.get("op")
        if kind == "stroke":
            w = d.get("w", 2.0)
            w = _dec(w, np.float32) if isinstance(w, (str, list)) else float(w)
            c = d.get("c", [0, 0, 0])
            if isinstance(c, str):
                c = _dec(c, np.uint8, (3,))
            elif isinstance(c, list) and c and isinstance(c[0], list):
                c = np.asarray(c, dtype=np.uint8)
            else:
                c = tuple(c)
            ops.append(Stroke(pts=_dec(d["pts"], np.float32, (2,)), width=w, color=c,
                              opacity=d.get("o", 1.0), brush=d.get("b", "pen"), seed=d.get("s", 0),
                              erase=bool(d.get("e", 0)), own=bool(d.get("own", 0))))
        elif kind == "fill":
            polys = d.get("poly")
            if polys is not None:
                polys = [_dec(p, np.float32, (2,)) for p in polys]
            ops.append(Fill(polygons=polys, mask_rle=d.get("rle"), color=tuple(d["c"]), opacity=d.get("o", 1.0),
                            feather=d.get("f", 0.0), color2=tuple(d["c2"]) if "c2" in d else None,
                            p0=tuple(d["p0"]) if "p0" in d else None, p1=tuple(d["p1"]) if "p1" in d else None,
                            erase=bool(d.get("e", 0)), radial=bool(d.get("rad", 0)),
                            own=bool(d.get("own", 0))))
        elif kind == "blend":
            polys = d.get("poly")
            if polys is not None:
                polys = [_dec(p, np.float32, (2,)) for p in polys]
            ops.append(Blend(polygons=polys, mask_rle=d.get("rle"), radius=d.get("r", 3.0), tol=d.get("tol", 12.0),
                             strength=d.get("k", 1.0), feather=d.get("f", 1.0), own=bool(d.get("own", 0))))
        elif kind == "spray":
            polys = d.get("poly")
            if polys is not None:
                polys = [_dec(p, np.float32, (2,)) for p in polys]
            sz, o = d.get("sz", 8.0), d.get("o", 0.3)
            ops.append(Spray(pts=_dec(d["pts"], np.float32, (2,)),
                             size=_dec(sz, np.float32) if isinstance(sz, (str, list)) else float(sz),
                             colors=_dec(d["c"], np.uint8, (3,)),
                             opacity=_dec(o, np.float32) if isinstance(o, (str, list)) else float(o),
                             polygons=polys, mask_rle=d.get("rle"), feather=d.get("f", 0.0), seed=d.get("s", 0),
                             own=bool(d.get("own", 0))))
        elif kind == "retouch":
            ops.append(Retouch(x=_dec(d["x"], np.int32), y=_dec(d["y"], np.int32), n=_dec(d["n"], np.int32),
                               axis=_dec(d["a"], np.uint8), size=_dec(d["sz"], np.uint8),
                               colors=_dec(d["c"], np.uint8, (3,)), own=bool(d.get("own", 0))))
        elif kind == "streak":
            ops.append(Streak(pts=_dec(d["pts"], np.float32, (2,)), n=_dec(d["n"], np.int32),
                              colors=_dec(d["c"], np.uint8, (3,)), width=_dec(d["w"], np.float32),
                              own=bool(d.get("own", 0))))
        else:
            raise ValueError(f"unknown op kind {kind!r}")
    return ops


# ----------------------------------------------------------------------------------------------- painting
def apply_ops(layer: np.ndarray, ops, clip=None, alpha_lock: bool = False):
    """Paint ``ops`` onto ``layer`` (float32 [H,W,4], straight alpha, 0..1) in place, in order.

    clip: bool / float mask [H,W] limiting where paint lands.  alpha_lock: paint only where the layer already has
    alpha (colour changes, alpha is kept).  Returns the changed bounding box (x0,y0,x1,y1) or None."""
    from . import brush
    return brush.apply_ops(layer, ops, clip=clip, alpha_lock=alpha_lock)


def apply_ops_progressive(layer: np.ndarray, ops, n_frames: int, clip=None) -> Iterator[np.ndarray]:
    """Paint ``ops`` in ``n_frames`` roughly equal chunks, yielding a copy of the layer after each chunk."""
    from . import brush
    return brush.apply_ops_progressive(layer, ops, n_frames, clip=clip)


def op_bbox(op, shape) -> Optional[tuple]:
    from . import brush
    return brush.op_bbox(op, shape)


# ----------------------------------------------------------------------------------------------- region ownership
def is_partial(mask) -> bool:
    """True when ``mask`` limits the picture (None or all-true means the whole picture)."""
    if mask is None:
        return False
    m = np.asarray(mask)
    return not bool((m if m.dtype == bool else m > 0.5).all())


def op_anchor(op) -> tuple:
    """The one point that decides which region a mark belongs to: a stroke's middle (half its length along the
    path), a fill's deepest interior point.  Every mark has exactly one anchor, so over a partition of the canvas
    every mark belongs to exactly one region."""
    if isinstance(op, Stroke):
        p = op.pts.astype(np.float64)
        if len(p) == 1:
            return float(p[0, 0]), float(p[0, 1])
        s = np.concatenate([[0.0], np.cumsum(np.sqrt((np.diff(p, axis=0) ** 2).sum(1)))])
        t = 0.5 * s[-1]
        return float(np.interp(t, s, p[:, 0])), float(np.interp(t, s, p[:, 1]))
    if isinstance(op, Retouch):
        return float(op.x[0]) + 0.5, float(op.y[0]) + 0.5
    if isinstance(op, Streak):
        p = op.pts[:int(op.n[0])].astype(np.float64)
        return float(p[:, 0].mean()), float(p[:, 1].mean())
    if isinstance(op, Spray) and op.polygons is None and op.mask_rle is None:
        c = op.pts.astype(np.float64).mean(0)
        return float(c[0]), float(c[1])
    import cv2
    if op.polygons is not None and len(op.polygons):
        allp = np.concatenate(op.polygons, 0)
        x0, y0 = np.floor(allp.min(0)).astype(int) - 1
        x1, y1 = np.ceil(allp.max(0)).astype(int) + 1
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for p in op.polygons:   # even-odd: XOR the rings
            r = np.zeros_like(m)
            cv2.fillPoly(r, [np.rint(p - 0.5 - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)], 1)
            m ^= r
    else:
        mm = rle_to_mask(op.mask_rle)
        ys, xs = np.nonzero(mm)
        if len(ys) == 0:
            return 0.0, 0.0
        y0, x0 = int(ys.min()), int(xs.min())
        m = mm[y0:ys.max() + 1, x0:xs.max() + 1].astype(np.uint8)
    if not m.any():
        c = np.concatenate(op.polygons, 0).mean(0)
        return float(c[0]), float(c[1])
    d = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    iy, ix = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(x0 + ix + 0.5), float(y0 + iy + 0.5)


def owned_ops(ops, mask) -> list:
    """The marks anchored inside ``mask``, flagged ``own`` so they are painted whole (see ``op_anchor``)."""
    m = np.asarray(mask)
    m = m if m.dtype == bool else m > 0.5
    H, W = m.shape
    out = []
    for op in ops:
        x, y = op_anchor(op)
        xi, yi = min(max(int(x), 0), W - 1), min(max(int(y), 0), H - 1)
        if m[yi, xi]:
            op.own = True
            out.append(op)
    return out


def owned_stack(ops, mask, shape) -> list:
    """For stacked marks (painted in order, later ones on top): the marks anchored in ``mask`` plus every later
    mark that overlaps one of them, all flagged ``own``.  Region calls then reproduce the whole picture's stacking
    whatever order they run in (a mark painted again is the same mark)."""
    m = np.asarray(mask)
    m = m if m.dtype == bool else m > 0.5
    H, W = m.shape
    boxes = [op_bbox(o, shape) for o in ops]
    own = []
    for i, op in enumerate(ops):
        x, y = op_anchor(op)
        own.append(bool(m[min(max(int(y), 0), H - 1), min(max(int(x), 0), W - 1)]))
    out = []
    active = []
    for i, op in enumerate(ops):
        b = boxes[i]
        take = own[i]
        if not take and b is not None:
            for a in active:
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    take = True
                    break
        if take:
            op.own = True
            out.append(op)
            if own[i] and b is not None:
                active.append(b)
    return out


def scale_ops(ops, s: float, shape=None) -> list:
    """The same marks on a canvas ``s`` times larger (positions, widths, sizes, radii and feathers times ``s``; a
    run-length mask resized nearest to ``shape`` (H, W) of the larger canvas).  Used to draw the coarse levels of a
    working canvas from the delivered-size picture."""
    import copy
    import cv2
    s = float(s)

    def rle(r):
        if r is None:
            return None
        m = rle_to_mask(r).astype(np.uint8)
        H, W = shape if shape is not None else (int(round(m.shape[0] * s)), int(round(m.shape[1] * s)))
        return mask_to_rle(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0)

    out = []
    for op in ops:
        o = copy.copy(op)
        if isinstance(op, Stroke):
            o.pts = op.pts * np.float32(s)
            o.width = op.width * np.float32(s) if isinstance(op.width, np.ndarray) else float(op.width) * s
        elif isinstance(op, Streak):
            o = Streak(pts=op.pts * np.float32(s), n=op.n, colors=op.colors, width=op.width * np.float32(s), own=op.own)
        elif isinstance(op, Retouch):
            raise ValueError("retouch marks are pixel marks: they are not scaled")
        else:
            if getattr(op, "polygons", None) is not None:
                o.polygons = [p * np.float32(s) for p in op.polygons]
            o.mask_rle = rle(getattr(op, "mask_rle", None))
            o.feather = float(op.feather) * s
            if isinstance(op, Spray):
                o.pts = op.pts * np.float32(s)
                o.size = op.size * np.float32(s)
            if isinstance(op, Blend):
                o.radius = float(op.radius) * s
            if isinstance(op, Fill):
                if op.p0 is not None:
                    o.p0 = tuple(float(v) * s for v in op.p0)
                if op.p1 is not None:
                    o.p1 = tuple(float(v) * s for v in op.p1)
        out.append(o)
    return out


# ----------------------------------------------------------------------------------------------- mark rules
def stroke_length(st: Stroke) -> float:
    p = st.pts.astype(np.float64)
    return float(np.sqrt((np.diff(p, axis=0) ** 2).sum(1)).sum()) if len(p) > 1 else 0.0


def _poly_area(p):
    p = np.asarray(p, np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def mark_area(op) -> float:
    """Approximate painted area of a mark in px^2 (strokes: length x mean width + the round caps)."""
    if isinstance(op, Stroke):
        w = float(np.mean(op.width))
        return stroke_length(op) * w + np.pi * w * w / 4.0
    if isinstance(op, Retouch):
        return float((op.n.astype(np.float64) * op.size).sum())
    if isinstance(op, Streak):
        w = op.mean_widths()
        return float((op.lengths() * w + np.pi * w * w / 4.0).sum())
    if isinstance(op, Spray):
        # the dabs' own footprints (diameter ``size``)
        return float((np.pi * op.size.astype(np.float64) ** 2 / 4.0).sum())
    if op.polygons is not None and len(op.polygons):
        areas = sorted((_poly_area(p) for p in op.polygons), reverse=True)
        return max(0.0, areas[0] - sum(areas[1:]))  # outer ring minus holes (even-odd)
    if op.mask_rle is not None:
        c = op.mask_rle.get("counts", [])
        return float(sum(c[1::2]))
    return 0.0


def _stair_share(p):
    """Share of a polygon's edges that are axis-aligned unit steps (a pixel-traced, unsmoothed outline)."""
    d = np.diff(np.vstack([p, p[:1]]).astype(np.float64), axis=0)
    unit = ((np.abs(d[:, 0]) < 1e-6) | (np.abs(d[:, 1]) < 1e-6)) & (np.abs(d).sum(1) <= 1.01)
    return float(unit.mean()) if len(d) else 0.0


LEVEL_NAMES = {5: "block-in", 4: "forms", 3: "edges", 2: "fine", 1: "hairline", 0: "retouch"}


def _scale_level(px: float) -> int:
    """Detail level of a mark whose characteristic size is ``px``: >= 32 block-in, 8-32 forms, 4-8 edges, 1.5-4 fine,
    1-1.5 hairline, below 1 px retouch."""
    if px >= 32.0:
        return 5
    if px >= 8.0:
        return 4
    if px >= 4.0:
        return 3
    if px >= 1.5:
        return 2
    if px >= 1.0 - 1e-6:
        return 1
    return 0


def mark_level(op) -> int:
    """The detail level (5 coarse .. 0 finest, DESIGN.md 10.1) a mark belongs to.

    Stroke: by its mean width (a single-point 1 px pixel-brush dab is level 0).  Fill: by the side of a square of its
    area (>= 12 px^2 is at least level 3, a smaller shape level 2).  Spray: by its smallest dab.  Blend: level 4 (it
    adds no detail).  Retouch: 1 px dabs are level 0, hairlines and 2 px dabs level 1 (an op takes its finest
    mark)."""
    if isinstance(op, Retouch):
        return 0 if bool(((op.n == 1) & (op.size == 1)).any()) else 1
    if isinstance(op, Streak):
        return _scale_level(float(op.mean_widths().min()))
    if isinstance(op, Stroke):
        w = float(np.mean(op.width))
        if op.brush == "pixel" and len(op.pts) == 1 and w <= 1.0 + 1e-6:
            return 0
        return _scale_level(w)
    if isinstance(op, Blend):
        return 4
    if isinstance(op, Spray):
        smin = float(op.size.min())
        return 5 if smin >= 32.0 else min(4, _scale_level(smin))
    a = mark_area(op)
    side = float(np.sqrt(max(a, 0.0)))
    if side >= 32.0:
        return 5
    if side >= 8.0:
        return 4
    return 3 if a >= 12.0 - 1e-6 else 2


def level_counts(ops) -> dict:
    """{level: number of marks}, finest last (a Spray counts its dabs, a Retouch its marks)."""
    out = {}
    for op in ops:
        if isinstance(op, Retouch):
            one = (op.n == 1) & (op.size == 1)
            for lv, k in ((0, int(one.sum())), (1, int((~one).sum()))):
                if k:
                    out[lv] = out.get(lv, 0) + k
            continue
        if isinstance(op, Streak):
            for w in op.mean_widths():
                lv = _scale_level(float(w))
                out[lv] = out.get(lv, 0) + 1
            continue
        lv = mark_level(op)
        out[lv] = out.get(lv, 0) + (len(op.pts) if isinstance(op, Spray) else 1)
    return dict(sorted(out.items(), reverse=True))


def check_mark_rules(ops, *, stage_tool: bool = True, min_level=None, levels: Optional[dict] = None) -> list:
    """Violations of DESIGN.md section 10.1 (empty list = every op is a mark a painter could make).

    At every detail level: one colour per mark (a stroke has no per-point colours; a fill has one colour or one
    two-colour gradient; each airbrush dab and each retouch mark has its own single colour); a mark is a mark (a
    stroke wider than 0, a fill or frisket with a smooth boundary or a feathered mask and an area); no image patch is
    pasted (no op carries an image).  ``min_level``: also flag marks finer than that level (a pass limited to "detail
    to level 2").  ``levels``: a dict that receives {level: number of marks}.  ``stage_tool`` is kept for
    compatibility (the pixel brush is the level-0 retouching brush)."""
    out = []
    if levels is not None:
        for k, v in level_counts(ops).items():
            levels[k] = levels.get(k, 0) + v
    for i, op in enumerate(ops):
        if min_level is not None and isinstance(op, (Stroke, Fill, Blend, Spray, Retouch, Streak)):
            lv = mark_level(op)
            if lv < int(min_level):
                out.append(f"op {i}: a level-{lv} mark ({LEVEL_NAMES.get(lv, lv)}) is finer than level "
                           f"{int(min_level)}")
        if isinstance(op, Stroke):
            if isinstance(op.color, np.ndarray):
                out.append(f"op {i}: stroke has per-point colours (one colour per stroke)")
            if float(np.min(op.width)) <= 0.0:
                out.append(f"op {i}: stroke width <= 0")
        elif isinstance(op, Blend):
            if mark_area(op) <= 0.0:
                out.append(f"op {i}: empty blend area")
        elif isinstance(op, Retouch):
            pass                     # one colour per mark by construction
        elif isinstance(op, Streak):
            # a loaded brush: the colour may change along the stroke, but smoothly (stations at least STREAK_STATION
            # px apart on average) and the stroke is a stroke (>= 2 px long, width > 0)
            if float(op.width.min()) <= 0.0:
                out.append(f"op {i}: streak width <= 0")
            ln = op.lengths()
            if (ln < 2.0 - 1e-3).any():
                out.append(f"op {i}: {int((ln < 2.0 - 1e-3).sum())} streaks shorter than 2 px")
            if ((op.n - 1) * STREAK_STATION > ln + 1e-3).any():
                out.append(f"op {i}: streak stations closer than {STREAK_STATION} px (colour changes per pixel)")
        elif isinstance(op, Spray):
            if float(op.size.min()) <= 0.0:
                out.append(f"op {i}: airbrush dab of size <= 0")
            if op.polygons is not None:
                for p in op.polygons:
                    if len(p) > 12 and _stair_share(p) > 0.5:
                        out.append(f"op {i}: frisket boundary is pixel-stepped (smooth it)")
            elif op.mask_rle is not None and op.feather < 0.75:
                out.append(f"op {i}: mask frisket without feather >= 0.75 has a pixel-stepped boundary")
        elif isinstance(op, Fill):
            if mark_area(op) <= 0.0:
                out.append(f"op {i}: empty fill")
            if op.polygons is not None:
                for p in op.polygons:
                    if len(p) < 3:
                        out.append(f"op {i}: polygon with fewer than 3 points")
                    elif len(p) > 12 and _stair_share(p) > 0.5:
                        out.append(f"op {i}: polygon boundary is pixel-stepped (smooth it)")
            elif op.mask_rle is not None and op.feather < 0.75:
                out.append(f"op {i}: mask fill without feather >= 0.75 has a pixel-stepped boundary")
            if (op.color2 is not None) != (op.p0 is not None and op.p1 is not None):
                out.append(f"op {i}: gradient needs color2, p0 and p1 together")
        else:
            out.append(f"op {i}: not an op ({type(op).__name__})")
    return out
