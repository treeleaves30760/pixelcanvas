"""``pixelcanvas scene <kit>``: a finished 2D background in the exact perspective of ``camera``.

A kit (corridor, room, classroom, street, platform, outdoor, field, beach, rooftop, stairs) is built as 3D geometry
in camera space (the ground is y = 0, the camera at x = 0 looking along +z, the same space ``pose`` and ``block``
use, so feet and contact shadows sit on its floor and its horizon is the camera's) and drawn as 2D: flat colour
per surface, a two-tone cel shade by the surface's side to the light, one soft depth haze (far things lighter and
thinner-lined), one light pool per lamp, and straight confident lines of one weight that thin with distance.

The picture goes on the paper layer (under every drawn figure; ``draw all``, ``flat auto`` and ``shade all`` leave
it alone), each surface is a region (wall, wainscot, floor, ceiling, door, window, lamp, sky, ground, ...) so
``recolor``, ``flat`` and the checks know it. The kit and its options are kept, so ``scene`` alone redraws it (after
the camera moved, for instance).

  pixelcanvas scene corridor [--time day|sunset|night] [--depth 24] [--width 3.2] [--height 3]
                   [--wall #e6dcc4] [--trim #7a4e30] [--floor #b07a4a] [--ceiling #f2ece0]
                   [--doors left|right|both|none] [--windows left|right|both|none] [--lamps 4] [--arch|--no-arch]
                   [--shift 0] [--light left|right]
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import coords
from . import draw_core as dc
from .coords import UsageError

KITS = ("corridor", "room", "classroom", "street", "platform", "outdoor", "field", "beach", "rooftop", "stairs")
USAGE = {"scene": "pixelcanvas scene corridor --time day --wall #e6dcc4 --trim #7a4e30 --floor #b07a4a --arch"}
SS = 2                                # supersampling: polygons and lines are drawn at 2x and reduced (clean edges)

# the base colours of each kit (any of them can be given as an option)
DEFAULTS = {
    "wall": "#e8dcc2", "trim": "#7c5034", "floor": "#b9814f", "ceiling": "#f3ede0", "door": "#8a5a3a",
    "window": "#bcd6ea", "lamp": "#fff1c8", "sky": "#8ec3ec", "ground": "#93b864", "road": "#6c6e76",
    "pavement": "#b8b2a6", "building": "#d8c8b0", "pole": "#5a5e66", "roof": "#5c6470", "platform": "#b4ada2",
    "edge": "#e6c23a", "rail": "#7a7068", "sand": "#ecd9a8", "sea": "#4f9ed0", "hill": "#7fa860",
    "tree": "#4f8a4a", "trunk": "#7a5a40", "fence": "#9aa4ac", "board": "#3e5a48", "desk": "#c89a64",
    "stairs": "#cfc4b0", "cloud": "#ffffff", "bench": "#8a6040", "line": "#3a2c26", "concrete": "#b8b4ac",
}
TIMES = {
    # sky top, sky at the horizon, multiplier on surface colours, haze colour weight, lamp strength
    "day": ((112, 172, 232), (214, 234, 246), (1.0, 1.0, 1.0), 1.0, 0.5),
    "sunset": ((96, 84, 150), (248, 178, 120), (1.0, 0.86, 0.78), 1.0, 0.6),
    "night": ((14, 20, 44), (42, 52, 86), (0.34, 0.36, 0.52), 0.6, 1.0),
}


def _rgb(c) -> np.ndarray:
    if isinstance(c, str):
        return np.asarray(coords.parse_colour(c), float)
    return np.asarray(c, float)


def _mix(a, b, k) -> np.ndarray:
    return np.asarray(a, float) * (1 - k) + np.asarray(b, float) * k


# ============================================================================================ the scene as 3D items

class Kit:
    """3D faces (flat polygons with a colour and a region) and lines, in the order a painter lays them."""

    def __init__(self, opts: dict):
        self.o = opts
        self.faces: List[dict] = []
        self.lines: List[dict] = []
        self.lamps: List[Tuple[np.ndarray, float]] = []        # (position, reach in m)
        self.sky = False
        self.ground: Optional[dict] = None                    # an infinite ground plane under the horizon
        self.clouds: List[Tuple[float, float, float]] = []    # (x, y, size) in 0-1000 units, above the horizon
        self.far: List[dict] = []                             # far silhouettes (hills, skyline, trees) in 2D units
        self.floor_y: Optional[float] = None                  # a floor that mirrors the lamps (polished)

    def col(self, key: str) -> np.ndarray:
        return _rgb(self.o.get(key) or DEFAULTS[key])

    def face(self, pts, rgb, region: str, order: int = 1, outline: bool = True, emit: bool = False,
             lw: float = 1.0) -> dict:
        P = np.asarray(pts, float)
        f = {"P": P, "rgb": np.asarray(rgb, float), "region": region, "order": order, "outline": outline,
             "emit": emit, "lw": lw, "lines": []}
        self.faces.append(f)
        return f

    def line(self, a, b, order: int = 2, lw: float = 1.0, rgb=None, region: str = "") -> None:
        self.lines.append({"a": np.asarray(a, float), "b": np.asarray(b, float), "order": order, "lw": lw,
                           "rgb": rgb, "region": region})

    def box(self, x0, y0, z0, x1, y1, z1, rgb, region: str, order: int = 3, lw: float = 1.0) -> None:
        """An axis-aligned box: its six faces (the hidden ones are culled when drawn)."""
        X, Y, Z = (x0, x1), (y0, y1), (z0, z1)
        c = lambda i, j, k: [X[i], Y[j], Z[k]]  # noqa: E731
        quads = [([c(0, 0, 0), c(0, 1, 0), c(0, 1, 1), c(0, 0, 1)], (-1, 0, 0)),
                 ([c(1, 0, 0), c(1, 0, 1), c(1, 1, 1), c(1, 1, 0)], (1, 0, 0)),
                 ([c(0, 0, 0), c(0, 0, 1), c(1, 0, 1), c(1, 0, 0)], (0, -1, 0)),
                 ([c(0, 1, 0), c(1, 1, 0), c(1, 1, 1), c(0, 1, 1)], (0, 1, 0)),
                 ([c(0, 0, 0), c(1, 0, 0), c(1, 1, 0), c(0, 1, 0)], (0, 0, -1)),
                 ([c(0, 0, 1), c(0, 1, 1), c(1, 1, 1), c(1, 0, 1)], (0, 0, 1))]
        grp = len(self.faces)
        for q, n in quads:
            f = self.face(q, rgb, region, order, lw=lw)
            f["normal"] = np.asarray(n, float)
            f["cull"] = True
            f["group"] = grp


def _opts_time(o: dict):
    return TIMES.get(str(o.get("time") or "day"), TIMES["day"])


def _side(o: dict, key: str, default: str) -> Tuple[bool, bool]:
    s = str(o.get(key) or default).lower()
    return s in ("left", "both", "l"), s in ("right", "both", "r")


def kit_corridor(k: Kit) -> None:
    o = k.o
    W = float(o.get("width") or 3.2)
    Hc = float(o.get("height") or 3.0)
    D = float(o.get("depth") or 24.0)
    sx = float(o.get("shift") or 0.0)
    xl, xr = sx - W / 2, sx + W / 2
    z0 = 0.3
    wall, trim, floor, ceil = k.col("wall"), k.col("trim"), k.col("floor"), k.col("ceiling")
    wh = 0.95                                   # the wainscot's height
    k.floor_y = 0.0
    k.face([[xl, 0, z0], [xr, 0, z0], [xr, 0, D], [xl, 0, D]], floor, "floor", 0)
    k.face([[xl, Hc, z0], [xl, Hc, D], [xr, Hc, D], [xr, Hc, z0]], ceil, "ceiling", 0)
    for x, nx in ((xl, 1), (xr, -1)):
        k.face([[x, wh, z0], [x, Hc, z0], [x, Hc, D], [x, wh, D]], wall, "wall", 0, outline=False)["normal"] = \
            np.array([nx, 0, 0.0])
        k.face([[x, 0, z0], [x, wh, z0], [x, wh, D], [x, 0, D]], trim, "wainscot", 0, outline=False)["normal"] = \
            np.array([nx, 0, 0.0])
        # the dado rail, the skirting and the wall's corners as lines
        k.line([x, wh, z0], [x, wh, D], 1, 1.0)
        k.line([x, wh + 0.05, z0], [x, wh + 0.05, D], 1, 0.6)
        k.line([x, 0, z0], [x, 0, D], 1, 1.1)
        k.line([x, Hc, z0], [x, Hc, D], 1, 1.1)
        # wainscot panels: vertical joints every 0.9 m
        for z in np.arange(z0 + 0.9, D, 0.9):
            k.line([x, 0.08, z], [x, wh - 0.08, z], 1, 0.45)
    k.face([[xl, 0, D], [xr, 0, D], [xr, Hc, D], [xl, Hc, D]], _mix(wall, (0, 0, 0), 0.04), "end-wall", 0)
    k.face([[xl, 0, D], [xr, 0, D], [xr, wh, D], [xl, wh, D]], trim, "wainscot", 0)
    # floor boards converging to the vanishing point, staggered end joints
    nb = max(6, int(round(W / 0.2)))
    xs = np.linspace(xl, xr, nb + 1)[1:-1]
    for i, x in enumerate(xs):
        k.line([x, 0, z0], [x, 0, D], 1, 0.7)
        for z in np.arange(z0 + (i % 3) * 0.6 + 0.4, D, 1.8):
            a, b = xs[i - 1] if i else xl, x
            k.line([a, 0, z], [b, 0, z], 1, 0.4)
    # the arched doorway in the end wall
    if o.get("arch", True):
        aw, ah = min(1.6, 0.5 * W), min(2.4, 0.8 * Hc)
        r = aw / 2
        pts = [[sx - r, 0, D - 0.01], [sx - r, ah - r, D - 0.01]]
        pts += [[sx + r * math.cos(t), ah - r + r * math.sin(t), D - 0.01] for t in np.linspace(math.pi, 0, 24)]
        pts += [[sx + r, 0, D - 0.01]]
        beyond = _mix(_mix(wall, (255, 248, 230), 0.5), (255, 255, 255), 0.25)
        # the moulding: a band of the trim colour round the opening
        R2 = r + 0.14
        outer = [[sx - R2, 0, D - 0.004], [sx - R2, ah - r, D - 0.004]]
        outer += [[sx + R2 * math.cos(t), ah - r + R2 * math.sin(t), D - 0.004] for t in np.linspace(math.pi, 0, 28)]
        outer += [[sx + R2, 0, D - 0.004]]
        k.face(outer, trim, "arch-frame", 1)["normal"] = np.array([0, 0, -1.0])
        f = k.face(pts, beyond, "arch", 1, emit=str(o.get("time")) == "night")
        f["normal"] = np.array([0, 0, -1.0])
        # the floor going on beyond the arch: a lighter trapezoid at the opening's foot
        fl2 = _mix(floor, (255, 240, 210), 0.35)
        g = k.face([[sx - r, 0, D - 0.012], [sx + r, 0, D - 0.012], [sx + 0.45 * r, 0.32 * ah, D - 0.012],
                    [sx - 0.45 * r, 0.32 * ah, D - 0.012]], fl2, "arch", 1, outline=False)
        g["normal"] = np.array([0, 0, -1.0])
        g["emit"] = f["emit"]
        k.line([sx - 0.45 * r, 0.32 * ah, D - 0.013], [sx + 0.45 * r, 0.32 * ah, D - 0.013], 2, 0.5)
    # doors on the side walls
    dl, dr = _side(o, "doors", "right")
    wl, wr = _side(o, "windows", "left")
    for x, on_d, on_w, nx in ((xl, dl, wl, 1), (xr, dr, wr, -1)):
        e = 0.012 * nx
        if on_d:
            for z in np.arange(3.0 + (0.0 if nx > 0 else 1.5), D - 1.5, 6.0):
                dw, dh = 1.0, 2.15
                f = k.face([[x + e, 0, z], [x + e, dh, z], [x + e, dh, z + dw], [x + e, 0, z + dw]], k.col("door"),
                           "door", 1)
                f["normal"] = np.array([nx, 0, 0.0])
                fr = 0.07
                k.face([[x + 2 * e, dh, z - fr], [x + 2 * e, dh + fr, z - fr], [x + 2 * e, dh + fr, z + dw + fr],
                        [x + 2 * e, dh, z + dw + fr]], trim, "door", 1)["normal"] = np.array([nx, 0, 0.0])
                k.line([x + 2 * e, 0.06, z + 0.12], [x + 2 * e, dh - 0.12, z + 0.12], 2, 0.4)
                k.line([x + 2 * e, 0.06, z + dw - 0.12], [x + 2 * e, dh - 0.12, z + dw - 0.12], 2, 0.4)
                k.line([x + 2 * e, 1.0, z + dw - 0.14], [x + 2 * e, 1.0, z + dw - 0.22], 2, 1.6,
                       rgb=(200, 170, 90))
        if on_w:
            for z in np.arange(2.0 + (0.0 if nx > 0 else 1.0), D - 1.5, 4.0):
                ww, y0, y1 = 2.0, wh + 0.35, min(Hc - 0.35, 2.5)
                f = k.face([[x + e, y0, z], [x + e, y1, z], [x + e, y1, z + ww], [x + e, y0, z + ww]],
                           k.col("window"), "window", 1, emit=str(o.get("time")) != "night")
                f["normal"] = np.array([nx, 0, 0.0])
                f["sheen"] = True
                k.line([x + 2 * e, y0, z + ww / 2], [x + 2 * e, y1, z + ww / 2], 2, 0.8)
                k.line([x + 2 * e, (y0 + y1) / 2, z], [x + 2 * e, (y0 + y1) / 2, z + ww], 2, 0.6)
    # hanging lamps on the ceiling's middle line
    # evenly spaced from 4 m: the first one inside the picture, the rest shrinking toward the far end
    every = float(o.get("lamp_every") or 4.0)
    zs = np.arange(4.0, 0.8 * D, every)
    if o.get("lamps") is not None:
        zs = np.linspace(4.0, D - 2.0, int(o["lamps"])) if int(o["lamps"]) > 1 else zs[:int(o["lamps"])]
    for z in zs:
        _hanging_lamp(k, sx, Hc, z, 0.7)


def _hanging_lamp(k: Kit, x: float, top: float, z: float, drop: float, r: float = 0.22) -> None:
    """A cord from the ceiling and a rounded shade facing the camera; it lights a pool on the floor."""
    y = top - drop
    k.line([x, top, z], [x, y + r, z], 4, 0.5)
    pts = [[x + r * 0.55 * math.cos(t), y + r * 0.9 * math.sin(t) + 0.2 * r, z] for t in np.linspace(0, 2 * math.pi,
                                                                                                    36)[:-1]]
    pts = [[px, max(py, y - 0.5 * r), pz] for px, py, pz in pts]
    f = k.face(pts, k.col("lamp"), "lamp", 4, emit=True)
    f["normal"] = np.array([0, 0, -1.0])
    f["billboard"] = True
    k.lamps.append((np.array([x, y, z]), 2.6))
    if k.o.get("polish", True) and k.floor_y is not None:
        # a polished floor mirrors the lamp: the same shape upside down under the floor, faint, no outline
        fy = k.floor_y
        ref = [[x + (px - x) * 0.8, fy - (py - fy - (y - fy)) * 0.0 - (y - fy) - (py - y) * 3.0, pz]
               for px, py, pz in pts]
        g = k.face(ref, _mix(k.col("floor"), k.col("lamp"), 0.6), "floor", 1, outline=False)
        g["normal"] = np.array([0, 0, -1.0])
        g["billboard"] = True
        g["glow"] = 0.45


def kit_room(k: Kit, classroom: bool = False) -> None:
    o = k.o
    W = float(o.get("width") or 8.0)
    Hc = float(o.get("height") or 3.0)
    D = float(o.get("depth") or 7.0)
    sx = float(o.get("shift") or 0.0)
    xl, xr = sx - W / 2, sx + W / 2
    z0 = 0.3
    wall, floor = k.col("wall"), k.col("floor")
    k.face([[xl, 0, z0], [xr, 0, z0], [xr, 0, D], [xl, 0, D]], floor, "floor", 0)
    k.face([[xl, Hc, z0], [xl, Hc, D], [xr, Hc, D], [xr, Hc, z0]], k.col("ceiling"), "ceiling", 0)
    for x, nx in ((xl, 1), (xr, -1)):
        k.face([[x, 0, z0], [x, Hc, z0], [x, Hc, D], [x, 0, D]], wall, "wall", 0)["normal"] = np.array([nx, 0, 0.0])
    k.face([[xl, 0, D], [xr, 0, D], [xr, Hc, D], [xl, Hc, D]], _mix(wall, (0, 0, 0), 0.03), "back-wall", 0)
    for x in np.arange(xl + 0.3, xr, 0.3):
        k.line([x, 0, z0], [x, 0, D], 1, 0.5)
    k.line([xl, 0.1, D - 0.01], [xr, 0.1, D - 0.01], 1, 0.6)
    wl, wr = _side(o, "windows", "left")
    for x, on, nx in ((xl, wl, 1), (xr, wr, -1)):
        if not on:
            continue
        e = 0.012 * nx
        for z in np.arange(1.2, D - 1.0, 2.2):
            y0, y1 = 0.9, 2.5
            f = k.face([[x + e, y0, z], [x + e, y1, z], [x + e, y1, z + 1.8], [x + e, y0, z + 1.8]], k.col("window"),
                       "window", 1, emit=True)
            f["normal"] = np.array([nx, 0, 0.0])
            f["sheen"] = True
            k.line([x + 2 * e, y0, z + 0.9], [x + 2 * e, y1, z + 0.9], 2, 0.8)
    if classroom:
        bw = min(4.0, 0.55 * W)
        k.face([[sx - bw / 2, 0.9, D - 0.02], [sx + bw / 2, 0.9, D - 0.02], [sx + bw / 2, 2.2, D - 0.02],
                [sx - bw / 2, 2.2, D - 0.02]], k.col("board"), "blackboard", 1)
        k.box(sx - bw / 2, 0.86, D - 0.12, sx + bw / 2, 0.9, D - 0.02, k.col("trim"), "blackboard", 2)
        # rows of desks in the back half of the room
        dk = k.col("desk")
        for z in np.arange(max(2.6, D * 0.45), D - 1.2, 1.3):
            for x in np.arange(xl + 1.0, xr - 0.9, 1.4):
                if abs(x - sx) < 0.5 and z < D * 0.6:
                    continue
                _desk(k, x, z, dk)
    else:
        if o.get("door", True):
            e = -0.012
            k.face([[xr + e, 0, D - 2.4], [xr + e, 2.1, D - 2.4], [xr + e, 2.1, D - 1.4], [xr + e, 0, D - 1.4]],
                   k.col("door"), "door", 1)["normal"] = np.array([-1, 0, 0.0])
    n = int(o.get("lamps") if o.get("lamps") is not None else 2)
    for i in range(n):
        z = z0 + (i + 1) * (D - z0) / (n + 1)
        k.face([[sx - 0.6, Hc - 0.01, z - 0.12], [sx + 0.6, Hc - 0.01, z - 0.12], [sx + 0.6, Hc - 0.01, z + 0.12],
                [sx - 0.6, Hc - 0.01, z + 0.12]], k.col("lamp"), "lamp", 2, emit=True)
        k.lamps.append((np.array([sx, Hc - 0.1, z]), 3.0))


def _desk(k: Kit, x: float, z: float, rgb) -> None:
    w, d, h, t = 0.65, 0.48, 0.72, 0.04
    leg = _mix(k.col("pole"), (255, 255, 255), 0.2)
    for lx in (x - w / 2 + 0.03, x + w / 2 - 0.06):
        k.box(lx, 0, z + d - 0.06, lx + 0.03, h - t, z + d - 0.03, leg, "desk", 3, 0.6)
        k.box(lx, 0, z + 0.03, lx + 0.03, h - t, z + 0.06, leg, "desk", 3, 0.6)
    k.box(x - w / 2, h - t, z, x + w / 2, h, z + d, rgb, "desk", 3, 0.8)
    k.box(x - w / 2 + 0.04, h - 0.22, z + 0.04, x + w / 2 - 0.04, h - t, z + d - 0.04, _mix(rgb, (0, 0, 0), 0.35),
          "desk", 3, 0.6)


def _sky_ground(k: Kit, ground_key: str = "ground") -> None:
    k.sky = True
    k.ground = {"rgb": k.col(ground_key), "region": ground_key}


def kit_outdoor(k: Kit, kind: str = "outdoor") -> None:
    o = k.o
    _sky_ground(k, "sand" if kind == "beach" else "ground")
    far = str(o.get("far") or {"outdoor": "hills", "field": "trees", "beach": "sea"}.get(kind, "hills")).lower()
    for w in far.replace(",", " ").split():
        k.far.append({"kind": w})
    if kind == "beach":
        k.far.append({"kind": "sea"})
    k.clouds = [(160, 0.45, 1.0), (520, 0.3, 0.8), (820, 0.55, 1.2)]
    if kind == "field":
        g = k.col("ground")
        rng = np.random.default_rng(3)
        for i in range(26):                                 # furrows / grass rows converging
            x = -26 + 2.0 * i
            k.line([x, 0, 1.0], [x, 0, 60.0], 1, 0.35, rgb=_mix(g, (0, 0, 0), 0.25))
        del rng


def kit_street(k: Kit) -> None:
    o = k.o
    _sky_ground(k, "road")
    sx = float(o.get("shift") or 0.0)
    D = float(o.get("depth") or 90.0)
    rw = float(o.get("width") or 7.0)
    pw = 3.0
    road, pave, bld = k.col("road"), k.col("pavement"), k.col("building")
    z0 = 0.3
    for side in (-1, 1):
        xa, xb = sx + side * rw / 2, sx + side * (rw / 2 + pw)
        lo, hi = min(xa, xb), max(xa, xb)
        k.box(lo, 0, z0, hi, 0.15, D, pave, "pavement", 0, 0.8)
        for z in np.arange(z0 + 1.5, D, 1.5):
            k.line([lo, 0.151, z], [hi, 0.151, z], 1, 0.35)
        # building fronts: a row of boxes of varied height, windows in a grid
        rng = np.random.default_rng(11 if side < 0 else 12)
        z = z0
        while z < D:
            w = float(rng.uniform(7, 12))
            h = float(rng.uniform(7, 16))
            x_front = sx + side * (rw / 2 + pw)
            x_back = x_front + side * 6
            col = _mix(bld, rng.choice([(255, 255, 255), (150, 120, 100), (120, 130, 150)]), 0.18)
            k.box(min(x_front, x_back), 0, z, max(x_front, x_back), h, z + w - 0.2, col, "building", 2, 0.9)
            for fy in np.arange(3.0, h - 1.0, 3.0):
                for fz in np.arange(z + 1.2, z + w - 1.6, 2.2):
                    e = -side * 0.02
                    q = [[x_front + e, fy, fz], [x_front + e, fy + 1.4, fz], [x_front + e, fy + 1.4, fz + 1.1],
                         [x_front + e, fy, fz + 1.1]]
                    f = k.face(q, k.col("window"), "window", 3, emit=str(o.get("time")) == "night", lw=0.6)
                    f["normal"] = np.array([-side, 0, 0.0])
            z += w
        for z in np.arange(6.0, D, 14.0):
            x = sx + side * (rw / 2 + 0.4)
            _pole(k, x, z, 6.5, lamp_to=-side)
    # lane marks
    for z in np.arange(2.0, D, 6.0):
        k.face([[sx - 0.08, 0.002, z], [sx + 0.08, 0.002, z], [sx + 0.08, 0.002, z + 3.0], [sx - 0.08, 0.002, z + 3.0]],
               (236, 232, 220), "road-mark", 1, outline=False)


def _pole(k: Kit, x: float, z: float, h: float, lamp_to: int = 0) -> None:
    r = 0.07
    k.box(x - r, 0, z - r, x + r, h, z + r, k.col("pole"), "pole", 3, 0.8)
    if lamp_to:
        x2 = x + lamp_to * 1.0
        k.box(min(x, x2), h - 0.08, z - 0.04, max(x, x2), h, z + 0.04, k.col("pole"), "pole", 3, 0.6)
        k.box(x2 - 0.25, h - 0.18, z - 0.12, x2 + 0.25, h - 0.08, z + 0.12, k.col("lamp"), "lamp", 3, 0.6)
        for f in k.faces[-6:]:
            f["emit"] = True
        k.lamps.append((np.array([x2, h - 0.2, z]), 4.0))


def kit_platform(k: Kit) -> None:
    o = k.o
    _sky_ground(k, "ground")
    k.ground["rgb"] = _mix(k.col("rail"), (0, 0, 0), 0.25)
    k.ground["region"] = "track-bed"
    sx = float(o.get("shift") or 0.0)
    D = float(o.get("depth") or 40.0)
    left = str(o.get("side") or "left").lower().startswith("l")     # the tracks' side
    sgn = -1 if left else 1
    edge_x = sx + sgn * 2.0                       # the platform edge
    far_x = sx - sgn * 4.0                        # the platform's back wall side
    lo, hi = min(edge_x, far_x), max(edge_x, far_x)
    z0 = 0.3
    # the platform sits 1 m above the track bed: the figures stand on its top (y = 0), so the tracks are below
    k.box(lo, -1.0, z0, hi, 0.0, D, k.col("platform"), "platform", 0, 1.0)
    # the yellow safety line along the edge
    e0, e1 = edge_x - sgn * 0.5, edge_x - sgn * 0.35
    k.face([[min(e0, e1), 0.002, z0], [max(e0, e1), 0.002, z0], [max(e0, e1), 0.002, D], [min(e0, e1), 0.002, D]],
           k.col("edge"), "edge-line", 1, outline=False)
    for z in np.arange(z0 + 1.2, D, 1.2):
        k.line([lo, 0.001, z], [hi, 0.001, z], 1, 0.3)
    # rails in the bed below the edge
    for dx in (1.2, 2.7):
        x = edge_x + sgn * dx
        k.line([x, -0.98, z0], [x, -0.98, D * 3], 1, 1.0, rgb=(150, 150, 160))
    for z in np.arange(z0, D * 2, 0.7):
        a, b = edge_x + sgn * 0.9, edge_x + sgn * 3.0
        k.line([a, -1.0, z], [b, -1.0, z], 1, 0.5)
    # the back wall of the platform
    k.face([[far_x, 0, z0], [far_x, 3.6, z0], [far_x, 3.6, D], [far_x, 0, D]], k.col("wall"), "wall", 0)["normal"] = \
        np.array([sgn * 1.0, 0, 0])
    # the roof on pillars, lamps hanging under it
    rh = 3.6
    k.box(lo - (1.0 if left else 0), rh, z0, hi + (0 if left else 1.0), rh + 0.25, D, k.col("roof"), "roof", 2, 1.0)
    px = far_x + sgn * 1.2
    for z in np.arange(5.0, D, 7.0):
        k.box(px - 0.12, 0, z - 0.12, px + 0.12, rh, z + 0.12, k.col("pole"), "pillar", 3, 0.9)
    for z in np.arange(3.0, D, 7.0):
        _hanging_lamp(k, (edge_x + far_x) / 2, rh, z, 0.55, r=0.2)
    # far: the opposite platform's lights and a dark city line
    k.far.append({"kind": "skyline"})


def kit_rooftop(k: Kit) -> None:
    o = k.o
    _sky_ground(k, "concrete")
    D = float(o.get("depth") or 9.0)
    W = 14.0
    sx = float(o.get("shift") or 0.0)
    k.ground = None
    k.face([[sx - W, 0, 0.3], [sx + W, 0, 0.3], [sx + W, 0, D], [sx - W, 0, D]], k.col("concrete"), "floor", 0)
    for z in np.arange(1.5, D, 1.5):
        k.line([sx - W, 0, z], [sx + W, 0, z], 1, 0.35)
    for x in np.arange(sx - W, sx + W, 1.5):
        k.line([x, 0, 0.3], [x, 0, D], 1, 0.35)
    k.box(sx - W, 0, D, sx + W, 0.3, D + 0.3, _mix(k.col("concrete"), (0, 0, 0), 0.1), "parapet", 1, 0.9)
    fc = k.col("fence")
    fh = 2.2
    for x in np.arange(sx - W, sx + W + 0.01, 2.0):
        k.box(x - 0.04, 0.3, D + 0.1, x + 0.04, fh + 0.3, D + 0.18, fc, "fence", 3, 0.7)
    for y in (0.35, fh + 0.25):
        k.box(sx - W, y, D + 0.12, sx + W, y + 0.05, D + 0.16, fc, "fence", 3, 0.6)
    for x in np.arange(sx - W, sx + W, 0.25):                  # the mesh: a diagonal lattice
        k.line([x, 0.35, D + 0.14], [x + 1.9, fh + 0.25, D + 0.14], 3, 0.25, rgb=_mix(fc, (60, 60, 70), 0.3))
        k.line([x + 1.9, 0.35, D + 0.14], [x, fh + 0.25, D + 0.14], 3, 0.25, rgb=_mix(fc, (60, 60, 70), 0.3))
    k.far.append({"kind": "skyline-low"})
    k.clouds = [(200, 0.35, 1.0), (700, 0.5, 1.3)]


def kit_stairs(k: Kit) -> None:
    o = k.o
    W = float(o.get("width") or 2.4)
    sx = float(o.get("shift") or 0.0)
    n = int(o.get("steps") or 12)
    z0 = float(o.get("at") or 3.0)
    run, rise = 0.3, 0.17
    xl, xr = sx - W / 2, sx + W / 2
    st = k.col("stairs")
    D = z0 + n * run + 6
    Hc = n * rise + 3.0
    k.face([[xl, 0, 0.3], [xr, 0, 0.3], [xr, 0, z0], [xl, 0, z0]], k.col("floor"), "floor", 0)
    for x, nx in ((xl, 1), (xr, -1)):
        k.face([[x, 0, 0.3], [x, Hc, 0.3], [x, Hc, D], [x, 0, D]], k.col("wall"), "wall", 0)["normal"] = \
            np.array([nx, 0, 0.0])
        # the handrail rising with the steps
        k.line([x - 0.08 * nx * -1, 0.9, z0 - 0.3], [x + 0.08 * nx, 0.9 + n * rise, z0 + n * run], 2, 1.0)
    for i in range(n):
        k.box(xl, 0, z0 + i * run, xr, (i + 1) * rise, z0 + n * run + 2, st, "stairs", 1, 0.8)
    k.face([[xl, n * rise, z0 + n * run + 2], [xr, n * rise, z0 + n * run + 2], [xr, Hc, D], [xl, Hc, D]],
           _mix(k.col("wall"), (0, 0, 0), 0.06), "end-wall", 0)
    k.face([[xl, Hc, 0.3], [xl, Hc, D], [xr, Hc, D], [xr, Hc, 0.3]], k.col("ceiling"), "ceiling", 0)
    _hanging_lamp(k, sx, Hc, z0 + n * run * 0.6, 0.6)


def build_kit(kit: str, opts: dict) -> Kit:
    k = Kit(opts)
    if kit == "corridor":
        kit_corridor(k)
    elif kit in ("room", "classroom"):
        kit_room(k, classroom=kit == "classroom")
    elif kit in ("outdoor", "field", "beach"):
        kit_outdoor(k, kit)
    elif kit == "street":
        kit_street(k)
    elif kit == "platform":
        kit_platform(k)
    elif kit == "rooftop":
        kit_rooftop(k)
    elif kit == "stairs":
        kit_stairs(k)
    else:
        raise UsageError(f"unknown scene kit {kit!r}; kits: {', '.join(KITS)}", USAGE["scene"])
    return k


# ============================================================================================ drawing

def _clip_near(P: np.ndarray, cam, near: float = 0.25) -> np.ndarray:
    """A polygon cut by the plane camera-z = near (Sutherland-Hodgman, one plane)."""
    Q = cam.to_cam(P)
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        za, zb = Q[i, 2], Q[(i + 1) % n, 2]
        if za >= near:
            out.append(a)
        if (za >= near) != (zb >= near):
            t = (near - za) / (zb - za)
            out.append(a + t * (b - a))
    return np.asarray(out) if out else np.zeros((0, 3))


def _seg_near(a, b, cam, near: float = 0.25):
    za, zb = cam.to_cam(np.array([a, b]))[:, 2]
    if za < near and zb < near:
        return None
    if za < near:
        a = a + (near - za) / (zb - za) * (b - a)
    elif zb < near:
        b = b + (near - zb) / (za - zb) * (a - b)
    return a, b


def render(kit: Kit, cam, W: int, H: int, light_left: bool = True, split: bool = False):
    """(RGB uint8 [H,W,3], region masks) of the kit seen by ``cam``. With ``split`` (and a time other than day) the
    surfaces keep their day colours and a third array is returned: the light as a multiply map [H,W,3] 0..1 (the
    time's tint, warm pools round the lamps, 1 on the sky and the lamps), laid over figures and scene alike."""
    o = kit.o
    sky_top, sky_hor, mult, haze_w, lamp_k = _opts_time(o)
    night = str(o.get("time") or "day") == "night"
    mult = np.asarray(mult, float)
    split = bool(split) and str(o.get("time") or "day") != "day"
    tint = mult.copy()
    if split:
        mult = np.ones(3)
    ss = SS if max(W, H) <= 2000 else 1          # a large canvas is fine enough as it is
    Ws, Hs = W * ss, H * ss
    cs = cam.resized(Ws, Hs)
    S = float(max(Ws, Hs))
    img = np.zeros((Hs, Ws, 3), np.float32)
    depth = np.full((Hs, Ws), 1e4, np.float32)
    rid = np.full((Hs, Ws), -1, np.int16)
    regions: List[str] = []

    def reg(name: str) -> int:
        if name not in regions:
            regions.append(name)
        return regions.index(name)
    hz = cs.horizon_px()
    ys = np.arange(Hs, dtype=np.float32)[:, None]
    uu, vv = np.meshgrid(np.arange(Ws, dtype=np.float32) + 0.5, np.arange(Hs, dtype=np.float32) + 0.5)
    rays = cs.rays(uu, vv).astype(np.float32)                    # camera-z = 1 scaled
    # the light: from the upper left (or right), a little toward the viewer
    L = np.array([-0.55 if light_left else 0.55, 0.75, -0.35])
    L /= np.linalg.norm(L)
    indoor = not kit.sky
    if indoor:
        haze = _mix(kit.col("wall"), (255, 255, 255), 0.35) * mult
    else:
        haze = np.asarray(sky_hor, float)
    # the sky: one soft gradient top to horizon
    if kit.sky:
        t = np.clip(ys / max(hz, 1.0), 0, 1) ** 1.4
        sky = (np.asarray(sky_top, float)[None, None, :] * (1 - t[..., None]) +
               np.asarray(sky_hor, float)[None, None, :] * t[..., None])
        img[:] = np.broadcast_to(sky, img.shape)
        rid[:] = reg("sky")
        if night:
            rng = np.random.default_rng(5)
            for _ in range(int(0.00004 * Ws * Hs)):
                x, y = int(rng.uniform(0, Ws)), int(rng.uniform(0, max(1, hz * 0.8)))
                cv2.circle(img, (x, y), max(1, int(S * 0.0009)), (230, 230, 210), -1, cv2.LINE_AA)
        _clouds(kit, img, rid, reg("cloud"), cs, hz, sky_hor, night)
        _far(kit, img, rid, reg, cs, hz, haze, tint)
    if kit.ground is not None:
        below = (vv > hz + 0.5)
        dy = rays[..., 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dy < -1e-6, (0.0 - cam.eye) / dy, 1e4)
        g = below & (t < 1e4)
        sea = None
        if any(f["kind"] == "sea" for f in kit.far):
            shore = 28.0
            sea = g & (t > shore)
            g2 = g & ~sea
        else:
            g2 = g
        img[g2] = kit.ground["rgb"] * mult
        rid[g2] = reg(kit.ground["region"])
        depth[g] = t[g]
        if sea is not None and sea.any():
            img[sea] = kit.col("sea") * mult
            rid[sea] = reg("sea")
    # faces, painter's order: by order, then far to near
    items = []
    for f in kit.faces:
        P = f["P"]
        Q = cs.to_cam(P)
        if Q[:, 2].max() < 0.25:
            continue
        items.append((f["order"], -float(Q[:, 2].mean()), "f", f))
    for ln in kit.lines:
        Q = cs.to_cam(np.array([ln["a"], ln["b"]]))
        items.append((ln["order"], -float(Q[:, 2].mean()), "l", ln))
    items.sort(key=lambda t: (t[0], t[1]))
    lw0 = 0.0021 * S
    ink = np.asarray(_rgb(o.get("line") or DEFAULTS["line"]), float)
    strokes = []                                       # face outlines: drawn after their face
    glows = []

    def fog(z):
        return np.clip(1.0 - np.exp(-np.asarray(z, float) / float(o.get("haze_m") or (22.0 if indoor else 70.0))),
                       0, 0.85 if indoor else 0.9) * (0.75 if indoor else 1.0) * haze_w

    def draw_seg(a, b, w_scale, rgb):
        seg = _seg_near(a, b, cs)
        if seg is None:
            return
        a, b = seg
        n = 12
        pts = np.linspace(a, b, n + 1)
        q, z = cs.project(pts)
        for i in range(n):
            zm = float((z[i] + z[i + 1]) / 2)
            w = lw0 * w_scale * float(np.clip(2.6 / max(zm, 0.3), 0.3, 1.0) ** 0.6)
            c = _mix(rgb, haze, float(fog(zm)) * 0.9)
            th = max(1, int(round(w)))
            p0 = tuple(int(round(v * 16)) for v in q[i])
            p1 = tuple(int(round(v * 16)) for v in q[i + 1])
            if max(abs(p0[0]), abs(p0[1]), abs(p1[0]), abs(p1[1])) > 2 ** 26:
                continue
            cv2.line(img, p0, p1, tuple(float(v) for v in c), th, cv2.LINE_AA, 4)

    for _, _, typ, it in items:
        if typ == "l":
            rgb = np.asarray(it["rgb"], float) * mult if it.get("rgb") is not None else ink
            draw_seg(it["a"], it["b"], it["lw"], rgb)
            continue
        f = it
        P = f["P"]
        nrm = f.get("normal")
        if nrm is None:
            e1, e2 = P[1] - P[0], P[2] - P[0]
            nrm = np.cross(e1, e2)
            nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        if f.get("billboard"):
            nrm = np.array([0, 0, -1.0])
        to_cam = cs.C - P.mean(0)
        if f.get("cull") and float(nrm @ to_cam) <= 0:
            continue
        if float(nrm @ to_cam) < 0:
            nrm = -nrm
        Pc = _clip_near(P, cs)
        if len(Pc) < 3:
            continue
        q, _ = cs.project(Pc)
        if not np.isfinite(q).all() or np.abs(q).max() > 1e6:
            continue
        m = np.zeros((Hs, Ws), np.uint8)
        cv2.fillPoly(m, [np.round(q * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
        mb = m > 0
        if not mb.any():
            continue
        # the depth of each pixel on the face's plane
        dn = rays[mb] @ nrm.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = float(nrm @ (P[0] - cs.C)) / dn
        t = np.where(np.isfinite(t) & (t > 0), t, 1e4).astype(np.float32)
        base = f["rgb"].astype(float)
        if f["emit"]:
            col = base if night else base
        else:
            lam = float(nrm @ L)
            col = base * mult * (1.0 if lam > -0.05 else 0.84)
            if abs(nrm[1]) > 0.9 and nrm[1] < 0:              # an underside: in shadow
                col = base * mult * 0.8
        if f.get("glow"):
            # a soft glint (a lamp mirrored in a polished floor): blurred and laid over later
            glows.append((mb, np.asarray(col, np.float32), float(f["glow"])))
            continue
        c = np.broadcast_to(col, (int(mb.sum()), 3)).astype(np.float32)
        if f.get("sheen"):
            # a glass pane: a lighter diagonal band
            yy, xx = np.nonzero(mb)
            ph = ((xx + yy) / (0.06 * S)) % 1.0
            c = c.copy()
            c[ph < 0.18] = _mix(col, (255, 255, 255), 0.45)
        img[mb] = c
        depth[mb] = t
        rid[mb] = reg(f["region"])
        if f["outline"]:
            strokes.append(None)
            for i in range(len(P)):
                draw_seg(P[i], P[(i + 1) % len(P)], f["lw"] * (0.8 if f["emit"] else 1.0), ink)
    for mb, col, k_g in glows:
        a_ = cv2.GaussianBlur(mb.astype(np.float32), (0, 0), 0.006 * S)[..., None] * k_g
        img[:] = img * (1 - a_) + col[None, None, :] * a_
    # depth haze and the lamps' light pools, per pixel, on every surface (not the sky)
    surf = depth < 1e3
    if surf.any():
        zz = depth[surf]
        k = fog(zz)[:, None].astype(np.float32)
        emit_ids = [regions.index(r) for r in ("lamp", "arch", "window") if r in regions]
        col = img[surf]
        if kit.lamps:
            Pw = cs.C[None, :] + rays[surf] * zz[:, None]
            pool = np.zeros(len(zz), np.float32)
            for Lp, reach in kit.lamps:
                # a pool on the floor under the lamp and on the walls beside it; nothing above the lamp
                d2 = ((Pw - Lp[None, :]) ** 2)
                dist = np.sqrt(d2[:, 0] + 0.25 * d2[:, 1] + d2[:, 2])
                below = np.clip((Lp[1] - Pw[:, 1]) / 0.6, 0, 1).astype(np.float32)
                pool = np.maximum(pool, below * np.exp(-(dist / reach) ** 2))
            lamp_rgb = np.asarray(_rgb(o.get("lamp") or DEFAULTS["lamp"]), float)
            glow = np.asarray(_mix(lamp_rgb, (255, 170, 70), 0.35 if night else 0.1), np.float32)
            add = (pool * lamp_k)[:, None]
            if split:
                pool_img = np.zeros((Hs, Ws), np.float32)
                pool_img[surf] = np.clip(pool * lamp_k, 0, 1)
                add = add * 0.0
            col = col * (1 - 0.55 * add) + glow[None, :] * (0.55 * add) + (glow[None, :] * 0.25 * add if night else 0)
        col = col * (1 - k) + np.asarray(haze, np.float32)[None, :] * k
        isem = np.isin(rid[surf], emit_ids)
        img[surf] = np.where(isem[:, None], img[surf], col)
    if night and kit.lamps:
        # a soft halo round each lamp
        halo = np.zeros((Hs, Ws), np.float32)
        for Lp, _ in kit.lamps:
            q, z = cs.project(Lp[None])
            if z[0] < 0.3:
                continue
            r = float(0.22 * cs.f / z[0])
            cv2.circle(halo, (int(q[0, 0]), int(q[0, 1])), max(2, int(r * 1.6)), 1.0, -1)
        halo = cv2.GaussianBlur(halo, (0, 0), 0.02 * S)
        img = img + halo[..., None] * np.array([255, 200, 120], np.float32)[None, None, :] * 0.35
    else:
        halo = None
    if split:
        # the sky and far things are drawn in the time's own colours: the paper holds them divided by the tint,
        # so the light layer (the tint everywhere) gives them back and darkens a figure in front of them alike
        far_ids = [regions.index(r) for r in ("sky", "cloud", "skyline", "hill", "tree", "sea") if r in regions]
        fm = np.isin(rid, far_ids)
        img[fm] = img[fm] / np.maximum(tint.astype(np.float32), 0.05)[None, :]
    out = cv2.resize(np.clip(img, 0, 255), (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8)
    rsmall = cv2.resize(rid, (W, H), interpolation=cv2.INTER_NEAREST)
    masks = {name: rsmall == i for i, name in enumerate(regions)}
    if not split:
        return out, masks
    lt = np.broadcast_to(tint.astype(np.float32), (Hs, Ws, 3)).copy()
    if kit.lamps and "pool_img" in locals():
        lamp_rgb = np.asarray(_rgb(o.get("lamp") or DEFAULTS["lamp"]), float) / 255.0
        warm = np.asarray(_mix(lamp_rgb, (1.0, 0.72, 0.4), 0.45), np.float32)
        a = cv2.GaussianBlur(pool_img, (0, 0), 0.01 * S)[..., None]
        lt = lt * (1 - a) + warm[None, None, :] * a
    if halo is not None:
        lt = lt * (1 - np.clip(halo, 0, 1)[..., None]) + np.clip(halo, 0, 1)[..., None]
    keep = np.isin(rid, [regions.index(r) for r in ("lamp",) if r in regions])
    lt[keep] = 1.0
    light = cv2.resize(lt, (W, H), interpolation=cv2.INTER_AREA)
    return out, masks, np.clip(light, 0, 1)


def _clouds(kit: Kit, img, rid, rk: int, cs, hz: float, sky_hor, night: bool) -> None:
    """Flat anime clouds: a cluster of round puffs on a flat base, a lighter top, no outline."""
    Hs, Ws = img.shape[:2]
    S = float(max(Ws, Hs))
    rng = np.random.default_rng(7)
    body = _mix((255, 255, 255), sky_hor, 0.15) if not night else _mix((70, 80, 120), sky_hor, 0.3)
    shade = _mix(body, (120, 130, 170), 0.25)
    for xu, yk, size in kit.clouds:
        cx = xu * Ws / 1000.0
        cy = hz * yk
        r = 0.05 * S * size
        m = np.zeros((Hs, Ws), np.uint8)
        for i in range(6):
            dx = (i - 2.5) * 0.55 * r
            rr = r * (0.55 + 0.45 * math.sin(math.pi * (i + 0.5) / 6)) * float(rng.uniform(0.85, 1.1))
            cv2.circle(m, (int(cx + dx), int(cy - rr * 0.35)), int(rr), 1, -1, cv2.LINE_AA)
        base_y = int(cy + 0.25 * r)
        m[base_y:] = 0
        mb = m > 0
        img[mb] = shade
        top = np.zeros_like(m)
        top[: max(0, base_y - int(0.35 * r))] = 1
        img[mb & (top > 0)] = body
        rid[mb] = rk


def _far(kit: Kit, img, rid, reg, cs, hz: float, haze, mult) -> None:
    """Far silhouettes on the horizon, lighter the farther (atmospheric perspective)."""
    Hs, Ws = img.shape[:2]
    S = float(max(Ws, Hs))
    rng = np.random.default_rng(9)
    xs = np.arange(Ws)
    for fdesc in kit.far:
        kind = fdesc["kind"]
        if kind == "hills":
            for layer, (amp, k_h) in enumerate(((0.07, 0.55), (0.045, 0.3))):
                ph = rng.uniform(0, 6.28)
                y = hz - amp * S * (0.6 + 0.4 * np.sin(xs / Ws * 6.28 * (1.2 + layer) + ph)) * \
                    (0.75 + 0.25 * np.sin(xs / Ws * 17 + ph))
                col = _mix(kit.col("hill") * mult, haze, k_h)
                m = (np.arange(Hs)[:, None] >= y[None, :]) & (np.arange(Hs)[:, None] <= hz + 1)
                img[m] = col
                rid[m] = reg("hill")
        elif kind in ("trees", "forest"):
            col = _mix(kit.col("tree") * mult, haze, 0.45)
            m = np.zeros((Hs, Ws), np.uint8)
            x = 0.0
            while x < Ws:
                r = float(rng.uniform(0.012, 0.025)) * S
                cv2.circle(m, (int(x), int(hz - r * 0.6)), int(r), 1, -1, cv2.LINE_AA)
                x += r * 1.1
            m[int(hz) + 1:] = 0
            img[m > 0] = col
            rid[m > 0] = reg("tree")
        elif kind.startswith("skyline"):
            low = kind.endswith("low")
            col = _mix(kit.col("building") * mult * 0.75, haze, 0.55 if not low else 0.45)
            x = 0
            while x < Ws:
                w = int(rng.uniform(0.03, 0.08) * S)
                h = float(rng.uniform(0.02, 0.09 if not low else 0.12)) * S
                y0 = int(hz - h)
                img[y0:int(hz) + 1, x:x + w] = col
                rid[y0:int(hz) + 1, x:x + w] = reg("skyline")
                if str(kit.o.get("time")) == "night":
                    for wy in range(y0 + 6, int(hz) - 4, max(4, int(0.008 * S))):
                        for wx in range(x + 4, x + w - 4, max(4, int(0.01 * S))):
                            if rng.random() < 0.35:
                                img[wy:wy + 3, wx:wx + 3] = (240, 200, 120)
                cv2.line(img, (x, y0), (x + w, y0), tuple(float(v) for v in _mix(col, (0, 0, 0), 0.15)),
                         max(1, int(0.001 * S)), cv2.LINE_AA)
                x += w + int(rng.uniform(0, 0.01) * S)


# ============================================================================================ the command

def scene_rec(p) -> Optional[dict]:
    return dc.dstate(p).get("scene3d")


def _parse(args: List[str], notes: List[str]) -> Tuple[str, dict]:
    from .cli import Spec, parse
    vals = ["time", "depth", "width", "height", "wall", "trim", "floor", "ceiling", "doors", "windows", "lamps",
            "shift", "light", "side", "far", "sky", "ground", "road", "building", "door", "window", "lamp",
            "line", "steps", "at", "platform", "roof", "sea", "sand", "hill", "tree", "fence", "concrete",
            "pavement", "board", "desk", "stairs", "haze", "lamp-every"]
    spec = {v: "value" for v in vals}
    spec.update({"arch": "flag", "no-arch": "flag", "polish": "flag", "no-polish": "flag"})
    a = parse(args, Spec(USAGE["scene"], ["kit"], spec, alias={"colour": "color", "tod": "time"}, min_pos=0))
    kit = str(a.get("kit") or "").lower()
    o = {}
    for v in vals:
        if a.get(v) is not None:
            o[v] = str(a[v])
    for v in ("depth", "width", "height", "shift", "lamps", "steps", "at"):
        if v in o:
            o[v] = float(coords.parse_float(o[v], v, -100, 1000, notes))
    if "lamps" in o:
        o["lamps"] = int(o["lamps"])
    if "lamp-every" in o:
        o["lamp_every"] = float(coords.parse_float(o.pop("lamp-every"), "lamp-every", 1, 50, notes))
    if "haze" in o:
        o["haze_m"] = float(o.pop("haze"))
    for v in DEFAULTS:
        if v in o:
            coords.parse_colour(o[v], USAGE["scene"])
    if "time" in o and o["time"] not in TIMES:
        raise UsageError(f"--time is one of {', '.join(TIMES)}", USAGE["scene"])
    if a.get("no-arch"):
        o["arch"] = False
    elif a.get("arch"):
        o["arch"] = True
    if a.get("no-polish"):
        o["polish"] = False
    elif a.get("polish"):
        o["polish"] = True
    return kit, o


def paint(p, rec: dict) -> Dict[str, np.ndarray]:
    """Draw the kit on the paper layer, save its regions; returns the region masks."""
    from . import draw_pose as DP
    cam = DP.camera_of(p)
    kit = build_kit(rec["kit"], rec["opts"])
    light_left = str(rec["opts"].get("light") or "left").lower().startswith("l")
    res = render(kit, cam, p.W, p.H, light_left, split=True)
    rgb, masks = res[0], res[1]
    arr = np.ones((p.H, p.W, 4), np.float32)
    arr[..., :3] = rgb.astype(np.float32) / 255.0
    p._write_layer_file("paper", arr)
    _light_layer(p)
    la = np.zeros((p.H, p.W, 4), np.float32)
    if len(res) == 3:
        la[..., :3] = res[2]
        la[..., 3] = 1.0
    p._write_layer_file("light", la if len(res) == 3 else None)
    d = dc.dstate(p)
    from .regions import clean_name
    for name, m in masks.items():
        if not m.any():
            continue
        rn = clean_name(name)
        p.save_region_mask(rn, m)
        p.state["regions"][rn] = {"focus": 0, "box": list(coords.px_box_to_units(coords.mask_bbox_px(m), p.W, p.H)),
                                  "area": round(float(m.mean()), 4), "how": "scene"}
        ys, xs = np.nonzero(m)
        d["palette"]["swatches"].setdefault(rn, dc.to_hex(tuple(int(v) for v in np.median(rgb[ys, xs], 0))))
    return masks


def _light_layer(p) -> None:
    """The scene's light: a multiply layer under the line art, over the scene and the figures (night, sunset)."""
    names = [e["name"] for e in p.state["layers"]]
    if "light" in names:
        return
    at = names.index("line") if "line" in names else len(names)
    p.state["layers"].insert(at, {"name": "light", "mode": "multiply", "opacity": 1.0, "visible": True,
                                  "clip": False, "alpha_lock": False, "empty": True})


def cmd_scene(ctx, args):
    from . import cli_draw
    p = ctx.project
    d = dc.dstate(p)
    kit, o = _parse(args, ctx.notes)
    old = d.get("scene3d")
    if not kit:
        if old is None:
            raise UsageError(f"say the kit: {', '.join(KITS)}", USAGE["scene"])
        kit, o = old["kit"], {**old["opts"], **o}
    elif old is not None and old.get("kit") == kit:
        o = {**old["opts"], **o}
    if kit not in KITS:
        raise UsageError(f"unknown scene kit {kit!r}; kits: {', '.join(KITS)}", USAGE["scene"])
    rec = {"kit": kit, "opts": o}
    before = dc.composite_u8(p, "view")
    names = [f"regions/{n}.png" for n in p.state.get("regions", {})]
    step = p.begin_step(["layers/paper.png", "layers/light.png"] + names + [f"regions/{n}.png" for n in
                                                        ("wall", "floor", "ceiling", "sky", "ground")])
    try:
        d["scene3d"] = rec
        masks = paint(p, rec)
        p.end_step(step, {"cmd": ctx.cmdline, "layer": "paper", "ops": None, "n_ops": len(masks), "bbox": None})
    except Exception:
        p.abort_step(step)
        raise
    rep = cli_draw.DReport(mutated=True)
    rep.step = step
    rep.add(f"step {step}  scene {kit}: drawn in the camera's perspective on the paper layer (under the figures; "
            f"draw all / flat auto / shade all keep it)")
    rep.add("options: " + (" ".join(f"--{k_} {v}" for k_, v in o.items() if k_ not in ("arch", "polish")) or "defaults") +
            (" --arch" if o.get("arch") else "") + (" --no-polish" if o.get("polish") is False else ""))
    rep.add("regions: " + ", ".join(sorted(m for m, v in masks.items() if v.any())))
    rep.add("change: pixelcanvas scene --wall #hex --time night ...  (only the options you give change); "
            "recolor <region> works on the regions")
    if not ctx.batch:
        cli_draw.show_before_after(ctx, rep, before, dc.composite_u8(p, "view"), (0, 0, p.W, p.H), step,
                                   f"the {kit} scene")
    cli_draw._tail(ctx, rep)
    rep.summary = f"scene {kit}"
    return rep


def flat_filter(p, labels, auto: dict) -> dict:
    """With a scene under the figures, ``flat auto`` fills only the drawn figures' and blocks' areas: an area that
    is mostly outside them keeps the scene showing (it is not painted over with a background colour)."""
    if scene_rec(p) is None:
        return auto
    from .draw_figure2d import load_maps
    mp = load_maps(p)
    if mp is None:
        return {n: v for n, v in auto.items() if v[0] != "background"}
    lab = mp["lab"]
    Hl, Wl = lab.shape
    small = cv2.resize(labels.astype(np.int32), (Wl, Hl), interpolation=cv2.INTER_NEAREST)
    inside = lab >= 0
    nA = int(labels.max()) + 1
    tot = np.bincount(small.ravel(), minlength=nA)
    ins = np.bincount(small.ravel(), weights=inside.ravel().astype(float), minlength=nA)
    out = {}
    for n, v in auto.items():
        if v[0] == "background":
            continue
        if n < nA and tot[n] > 0 and ins[n] < 0.5 * tot[n]:
            continue
        out[n] = v
    return out


FLOORS = ("floor", "ground", "road", "pavement", "platform", "sand", "concrete", "stairs", "edge-line", "road-mark")


def floor_mask(p):
    """(mask of the scene's floor regions [H,W], colour at a pixel (x, y) -> rgb) or None without a scene."""
    if scene_rec(p) is None:
        return None
    m = np.zeros((p.H, p.W), bool)
    for n in FLOORS:
        if n in p.state.get("regions", {}) and p.state["regions"][n].get("how") == "scene":
            try:
                m |= p.region_mask(n)
            except Exception:  # noqa: BLE001
                pass
    if not m.any():
        return None
    paper = p.layer("paper")

    def colour(x, y):
        x = int(np.clip(x, 0, p.W - 1))
        y = int(np.clip(y, 0, p.H - 1))
        return tuple(int(round(v * 255)) for v in paper[y, x, :3])
    return m, colour
