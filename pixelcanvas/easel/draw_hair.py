"""Hair built from locks, without a reference: ``easel locks <hair part> --cut long --part x,y``.

A cut is a set of numbers for a generator, not a stored drawing: how many locks, how long (in head heights from the
top of the skull), how pointed the tips are, how they curl and spread, where twin tails or a ponytail attach. The
generator reads the head construction (``sketch head``: box, yaw, chibi), the mannequin's neck and shoulders when
there is a figure, or the hair part's box when there is neither, and builds overlapping tapered locks, back to
front:

1. the back mass behind the head (the face, the skull, the neck and the shoulders hide its middle);
2. twin tails or a ponytail, when the cut has them (behind the head);
3. the crown: the top of the hair, with volume above the skull;
4. a few stray ends that break the dome;
5. side locks framing the face, in front of the shoulders, their inner edge through the corner of the jaw;
6. the fringe: pointed locks from the part over the forehead, tips near the brow line, curving away from the part.

The line layer has no depth, so every piece of an outline that lies inside a lock (or the face) in front of it is
cut away; each visible piece is one editable line record (part = the hair part, ``src`` = ``locks <part>``): weight
1.0 inside the hair, 1.4 on the outer silhouette, tapered where a piece ends freely, blunt where it runs behind
another lock. Lines of the neck and the clothes drawn before the locks are cut where a lock in front covers them (so
each lock stays one closed area down to its tip), and the back hair stops at the outline of the body and the clothes.
Running ``locks`` again puts those lines back and replaces the lock records. The lock list (centre paths, widths,
order) is kept in ``draw["locks"][part]``; ``easel locks shine <part>`` reads it in the detail stage: a tapered shadow
strip along each lock's side away from the light, a shine ring across the hair at a third of the head height (one
pointed piece per lock, bent along the lock) and one or two strands per lock along its centre path.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import coords
from . import draw_core as dc
from .coords import UsageError

USAGE = {"locks": "easel locks hair --cut long --part 500,90"}

CUT_NAMES = ("long", "bob", "short", "spiky", "twin", "ponytail", "wavy")

# Numbers for the generator. Lengths are head heights from the top of the skull; widths are fractions of the head
# width; curl > 0 flicks the tips outward, < 0 tucks them in; tip > 1 makes needle tips, < 1 blunt ones; jag varies
# the lengths; fvar varies the fringe lengths (0 = a straight cut).
CUTS: Dict[str, dict] = {
    "long": dict(length=2.3, back=7, side=2, side_len=1.75, fringe=5, stray=2, volume=1.08, spread=0.22,
                 curl=0.12, tip=1.0, wave=0.0, jag=0.05, fvar=1.0, bang=1.0, tails=None, tail_len=0.0,
                 ahoge=False),
    "bob": dict(length=1.06, back=6, side=1, side_len=1.0, fringe=6, stray=1, volume=1.13, spread=0.12,
                curl=-0.5, tip=0.8, wave=0.0, jag=0.025, fvar=0.35, bang=0.96, tails=None, tail_len=0.0,
                ahoge=False),
    "short": dict(length=0.8, back=5, side=1, side_len=0.74, fringe=5, stray=2, volume=1.1, spread=0.12,
                  curl=0.3, tip=1.2, wave=0.0, jag=0.06, fvar=1.0, bang=0.92, tails=None, tail_len=0.0,
                  ahoge=True),
    "spiky": dict(length=0.92, back=7, side=2, side_len=0.8, fringe=5, stray=3, volume=1.18, spread=0.6,
                  curl=0.45, tip=1.6, wave=0.0, jag=0.16, fvar=1.6, bang=0.95, tails=None, tail_len=0.0,
                  ahoge=True),
    "twin": dict(length=0.7, back=4, side=1, side_len=1.05, fringe=5, stray=1, volume=1.08, spread=0.04,
                 curl=0.1, tip=1.15, wave=0.0, jag=0.04, fvar=0.8, bang=1.0, tails="twin", tail_len=2.1,
                 ahoge=False),
    "ponytail": dict(length=0.74, back=4, side=1, side_len=1.0, fringe=5, stray=1, volume=1.05, spread=0.04,
                     curl=0.1, tip=1.15, wave=0.0, jag=0.04, fvar=0.8, bang=1.0, tails="pony", tail_len=2.1,
                     ahoge=False),
    "wavy": dict(length=2.2, back=7, side=2, side_len=1.65, fringe=5, stray=2, volume=1.13, spread=0.32,
                 curl=0.2, tip=1.1, wave=0.07, jag=0.05, fvar=1.0, bang=1.0, tails=None, tail_len=0.0,
                 ahoge=False),
}

INNER_WEIGHT = 1.0
OUTER_WEIGHT = 1.4

# depth of each layer (back to front); the neck and shoulders hide only the back mass and a ponytail
DEPTH = {"back": 10, "tail": 20, "body": 30, "head": 31, "crown": 40, "stray": 45, "side": 50, "fringe": 60}
# locks that hang in front of the neck and the clothes (twin tails too)
FRONT = ("side", "fringe")
HAIRISH = ("hair", "bang", "fringe", "tail", "mane", "lock", "braid", "ahoge")
NOT_BODY = ("background", "bg", "sky", "ground", "floor", "wall")


# ============================================================================================ small geometry

def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _arc_t(path: np.ndarray) -> np.ndarray:
    """Arc-length fraction 0..1 of every point of a polyline."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    return s / max(float(s[-1]), 1e-9)


def _normals(P: np.ndarray) -> np.ndarray:
    """Unit normals (-ty, tx): on a path going down they point to the picture's left."""
    t = np.gradient(np.asarray(P, float), axis=0)
    if len(P) >= 5:
        k = np.ones(5) / 5.0
        t = np.stack([np.convolve(np.pad(t[:, i], 2, mode="edge"), k, "valid") for i in range(2)], 1)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return np.stack([-t[:, 1], t[:, 0]], 1)


def _centre(waypoints, spacing: float = 1.5) -> np.ndarray:
    W = np.asarray(waypoints, float).reshape(-1, 2)
    return dc.resample(dc.catmull_rom(W, spacing=spacing), spacing)


def taper_profile(t, w_max: float, kind: str = "hang", root: Optional[float] = None, t_max: Optional[float] = None,
                  tip: float = 1.15) -> np.ndarray:
    """Width along a lock (``t`` = arc fractions 0..1 or a point count): from the root up to ``w_max`` at ``t_max``,
    then down to a point at the tip. ``kind`` hang (wide root, hangs from the head), leaf (narrow root, a fringe
    lock) or tuft (a short stray end). ``tip`` > 1 gives a needle tip, < 1 a blunt one."""
    t = np.linspace(0.0, 1.0, int(t)) if np.ndim(t) == 0 else np.asarray(t, float)
    r0, tm = {"leaf": (0.14, 0.58), "tuft": (0.55, 0.25)}.get(kind, (0.62, 0.24))
    r0 = r0 if root is None else float(root)
    tm = tm if t_max is None else float(t_max)
    rise = r0 + (1.0 - r0) * _ss(t / max(tm, 1e-6))
    x = np.clip((t - tm) / max(1.0 - tm, 1e-6), 0.0, 1.0)
    a = max(0.8, float(tip))
    fall = (1.0 - x) ** a * (1.0 + a * x)          # a smooth shoulder at t_max, a point at the tip
    w = w_max * rise * fall
    if len(w):
        w[-1] = 0.0
    return np.minimum(w, w_max)


def lock_shape(path, widths) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(left edge, right edge, closed polygon) of a lock: the centre path widened by ``widths``; edges run root to
    tip and meet at the tip when the last width is 0."""
    P = np.asarray(path, float).reshape(-1, 2)
    w = np.asarray(widths, float).reshape(-1)
    n = _normals(P)
    left = P + n * (w / 2.0)[:, None]
    right = P - n * (w / 2.0)[:, None]
    return left, right, np.vstack([left, right[::-1]])


def visible_runs(pts, front: np.ndarray, min_len: int = 1) -> List[Tuple[int, int]]:
    """Index ranges [a, b) of the points that are not inside ``front`` (the fill of everything in front). Points
    off the canvas count as visible (the renderer clips them)."""
    P = np.asarray(pts, float).reshape(-1, 2)
    H, W = front.shape
    xi = np.floor(P[:, 0]).astype(np.int64)
    yi = np.floor(P[:, 1]).astype(np.int64)
    on = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    hid = np.zeros(len(P), bool)
    hid[on] = front[yi[on], xi[on]]
    d = np.diff(np.concatenate([[0], (~hid).astype(np.int8), [0]]))
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(i), int(j)) for i, j in zip(a, b) if j - i >= min_len]


def _poly_mask(H, W, pts) -> np.ndarray:
    P = np.asarray(pts, float).reshape(-1, 2)
    if len(P) < 3:
        return np.zeros((H, W), bool)
    return dc.poly_mask(H, W, np.clip(P, -4 * max(H, W), 5 * max(H, W)))


def _ellipse_inside(pt, c, rx, ry) -> float:
    return ((pt[0] - c[0]) / rx) ** 2 + ((pt[1] - c[1]) / ry) ** 2


# ============================================================================================ head frame

def _overlap(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def frame(p, part: str, cut: dict) -> dict:
    """The head the hair grows on, in px: box, ball, jaw, centre line, brow, eye boxes, neck and shoulders."""
    from . import draw_sketch
    d = dc.dstate(p)
    prt = d["parts"].get(part, {})
    subj = prt.get("of")
    hb = prt.get("box")
    heads = [e for e in d["sketch"] if e["kind"] == "head"]
    el = None
    if heads:
        same = [e for e in heads if subj and e.get("subject") == subj]
        el = same[-1] if same else (max(heads, key=lambda e: _overlap(e["box"], hb)) if hb else heads[-1])
    figs = [e for e in d["sketch"] if e["kind"] == "figure"]
    fig = ([e for e in figs if subj and e.get("subject") == subj] or figs or [None])[-1]
    if el is not None:
        head = {"box": list(el["box"]), "yaw": float(el.get("yaw", 0)), "pitch": float(el.get("pitch", 0)),
                "chibi": bool(el.get("chibi"))}
        source, head_id = f"head {el['id']}", el["id"]
    elif fig is not None:
        hd = dc.u2px(p, [fig["joints"]["head"]])[0]
        ch = dc.u2px(p, [fig["joints"]["chin"]])[0]
        r = float(np.hypot(*(ch - hd)))
        box = [(hd[0] - 0.8 * r) * 1000 / p.W, (hd[1] - r) * 1000 / p.H, (hd[0] + 0.8 * r) * 1000 / p.W,
               ch[1] * 1000 / p.H]
        head = {"box": box, "yaw": 0.0, "pitch": 0.0, "chibi": float(fig.get("heads", 7)) < 4.2}
        source, head_id = f"figure {fig['id']}", None
    elif hb:
        head = {"box": _head_in_box(p, hb, cut), "yaw": 0.0, "pitch": 0.0, "chibi": False}
        source, head_id = "hair box", None
    else:
        raise UsageError(f"{part} has no box and there is no head construction to grow the hair on",
                         f"easel plan part {part} --box 150,40,850,920")
    x0, y0, x1, y1 = dc.u2px(p, [head["box"][:2], head["box"][2:]]).reshape(-1)
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    yaw = math.radians(head["yaw"])
    sy = math.sin(yaw)
    rx, ry = 0.5 * w, 0.4 * h
    by = y0 + ry
    shp = draw_sketch.head_shapes(p, head)
    ball, jl, jr = shp[0][0], shp[1][0], shp[2][0]
    g = {"S": float(max(p.W, p.H)), "x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": w, "h": h, "cx": cx, "rx": rx,
         "ry": ry, "by": by,
         "yaw": yaw, "sy": sy, "chibi": head["chibi"], "cxl": cx + sy * rx * 0.92, "jaw_l": jl[0], "jaw_r": jr[0],
         "chin": jl[-1], "ball": ball, "jaw": np.vstack([jl, jr[::-1]]), "source": source, "head_id": head_id,
         "head_box": head["box"], "fig": fig}
    gu = draw_sketch.head_guides(p, head)
    eye_y = dc.u2px(p, [gu["left eye"]])[0][1]
    brow = eye_y - (0.11 if head["chibi"] else 0.1) * h
    eyes = []
    for n, v in d["parts"].items():
        if n.startswith("eye") and (not subj or v.get("of") == subj):
            b = dc.box_px(p, v["box"])
            if _overlap(b, (x0, y0, x1, y1)) > 0:
                eyes.append(b)
    if eyes:
        brow = min(brow, min(b[1] for b in eyes) - 0.015 * h)
    g["brow"] = brow
    g["eyes"] = eyes
    g["shoulders"] = None
    g["neck"] = None
    if fig is not None:
        ls = dc.u2px(p, [fig["joints"]["lshoulder"]])[0]
        rs = dc.u2px(p, [fig["joints"]["rshoulder"]])[0]
        if rs[0] < ls[0]:
            ls, rs = rs, ls
        if max(ls[1], rs[1]) > y1 + 0.05 * h and rs[0] - ls[0] > 0.6 * w:
            g["shoulders"] = (ls, rs)
            nk = next((v for n, v in d["parts"].items() if n == "neck" and (not subj or v.get("of") == subj)), None)
            half = (0.16 if head["chibi"] else 0.2) * w
            if nk is not None:
                b = dc.box_px(p, nk["box"])
                if b[0] < g["chin"][0] < b[2] and b[2] - b[0] < 0.8 * w:
                    g["neck"] = (b[0], b[2])
            if g["neck"] is None:
                g["neck"] = (g["chin"][0] - half, g["chin"][0] + half)
    return g


def _head_in_box(p, hb, cut: dict) -> List[float]:
    """A head fitted in the hair part's box (units), from the cut's length and spread."""
    x0, y0, x1, y1 = dc.box_px(p, hb)
    bw, bh = x1 - x0, y1 - y0
    w = bw / (1.12 * (1.0 + cut["spread"]) + 0.12)
    L = cut["tail_len"] if cut["tails"] else cut["length"]
    w = min(w, bh / max(1.0, 1.25 * (L + 0.12)))
    h = 1.25 * w
    cx = (x0 + x1) / 2
    top = y0 + (0.4 * (cut["volume"] - 1) + 0.05) * h
    return [(cx - w / 2) * 1000 / p.W, top * 1000 / p.H, (cx + w / 2) * 1000 / p.W, (top + h) * 1000 / p.H]


def _body_shapes(g: dict, H: int, W: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Neck and shoulders (px polygons) from the mannequin, which hide the back mass, and the zone under the jaw
    where the back hair shows as one mass (its lock edges are left out there)."""
    if g["shoulders"] is None:
        return [], []
    ls, rs = g["shoulders"]
    nl, nr = g["neck"]
    w, h = g["w"], g["h"]
    ysh = (ls[1] + rs[1]) / 2
    chin = g["chin"]
    neck = np.array([[nl, chin[1] - 0.25 * h], [nr, chin[1] - 0.25 * h], [nr + 0.02 * w, ysh + 0.05 * h],
                     [nl - 0.02 * w, ysh + 0.05 * h]])
    top = ysh - 0.07 * h
    bottom = max(float(H) + 10.0, ysh + 2.5 * h)
    torso = dc.catmull_rom(np.array([[nl - 0.01 * w, top - 0.02 * h], [(nl + ls[0]) / 2, top + 0.02 * h],
                                     [ls[0], ls[1]], [ls[0] - 0.08 * w, ls[1] + 0.35 * h]]), spacing=3.0)
    torso_r = dc.catmull_rom(np.array([[rs[0] + 0.08 * w, rs[1] + 0.35 * h], [rs[0], rs[1]],
                                       [(nr + rs[0]) / 2, top + 0.02 * h], [nr + 0.01 * w, top - 0.02 * h]]),
                             spacing=3.0)
    poly = np.vstack([torso, [[ls[0] - 0.1 * w, bottom], [rs[0] + 0.1 * w, bottom]], torso_r])
    # under the jaw, between the side locks, the back hair is one dark mass: no lock edges there (the zone reaches
    # under the side locks, which hide it anyway, so no sliver of an edge is left beside them)
    jl, jr = g["jaw_l"], g["jaw_r"]
    low = ysh if g.get("back_tip") is None else min(ysh, g["back_tip"] - 0.28 * h)   # short hair: its tips show
    if low < max(jl[1], jr[1]) + 0.08 * h:
        return [neck, poly], []
    under = np.array([[jl[0] - 0.1 * w, jl[1]], [jr[0] + 0.1 * w, jr[1]], [jr[0] + 0.1 * w, low],
                      [jl[0] - 0.1 * w, low]])
    return [neck, poly], [under]


# ============================================================================================ the locks

def _lock(layer: str, waypoints, w_max: float, kind: str = "hang", root=None, t_max=None, tip=1.15,
          wave: float = 0.0, wave_phase: float = 0.0, drift=None, drift_from: Optional[float] = None,
          t0: Tuple[float, float] = (0.0, 0.0), **extra) -> dict:
    path = _centre(waypoints)
    t = _arc_t(path)
    if wave:
        # an S-wave across the lock, none at the root
        off = wave * np.sin(2 * math.pi * 1.7 * t + wave_phase) * _ss(t / 0.25)
        path = path + _normals(path) * off[:, None]
    if drift is not None and drift_from is not None:
        drop = np.maximum(0.0, path[:, 1] - drift_from)
        path = path.copy()
        path[:, 0] += drift(drop)
        path = dc.resample(path, 1.5)
        t = _arc_t(path)
    widths = taper_profile(t, w_max, kind, root, t_max, tip)
    k = {"layer": layer, "path": path, "width": widths, "t0": t0}
    k.update(extra)
    return k


def _flow_drift(flow_px: Optional[np.ndarray]):
    """x offset as a function of the drop below where a lock starts to hang, from a --flow curve."""
    if flow_px is None or len(flow_px) < 2:
        return None
    F = dc.catmull_rom(np.asarray(flow_px, float), spacing=2.0)
    dy = np.maximum.accumulate(np.maximum(F[:, 1] - F[0, 1], 0.0))
    dx = F[:, 0] - F[0, 0]
    if dy[-1] < 1.0:
        return None
    keep = np.concatenate([[True], np.diff(dy) > 1e-6])
    dy, dx = dy[keep], dx[keep]
    tail = (dx[-1] - dx[max(0, len(dx) - 6)]) / max(1e-6, dy[-1] - dy[max(0, len(dy) - 6)])

    def g(v):
        v = np.asarray(v, float)
        return np.where(v <= dy[-1], np.interp(v, dy, dx), dx[-1] + tail * (v - dy[-1]))
    return g


def generate(g: dict, c: dict, P: np.ndarray, rng: np.random.Generator, flow=None) -> List[dict]:
    """Every lock of the cut, in px (centre path, widths, layer, depth)."""
    x0, y0, x1, y1, w, h = g["x0"], g["y0"], g["x1"], g["y1"], g["w"], g["h"]
    cx, by, rx, ry, sy = g["cx"], g["by"], g["rx"], g["ry"], g["sy"]
    vol = c["volume"] + (0.05 if g["chibi"] else 0.0)
    cxd = cx - sy * 0.05 * w
    rxd = rx * (1 + 0.9 * (vol - 1)) + 0.025 * w
    ryd = ry * vol + 0.02 * h
    g["dome"] = (cxd, by, rxd, ryd)
    fat = 1.15 if g["chibi"] else 1.0
    drift = _flow_drift(flow)
    out: List[dict] = []
    jag = c["jag"]

    # ---- 1. back mass: behind the head, the outer locks continue the dome's sides
    K = int(c["back"])
    L_tip = y0 + c["length"] * h
    g["back_tip"] = L_tip if K > 0 else None
    if K > 0:
        spread = c["spread"]
        wk = (2 * rxd * (1 + 0.5 * spread)) / K * 1.65 * fat
        for k in range(K):
            u = -1.0 + 2.0 * k / (K - 1) if K > 1 else 0.0
            uj = u + (rng.uniform(-0.05, 0.05) if 0 < k < K - 1 else 0.0)
            side_pt = (cxd + uj * (rxd - 0.45 * wk), by + 0.04 * ryd)
            root = (cxd + 0.5 * rxd * uj, by - 0.55 * ryd * (1 - 0.4 * uj * uj))
            ty = L_tip - 0.05 * h * abs(u) + rng.normal(0, jag * h)
            ty = max(ty, side_pt[1] + 0.22 * h)
            tx = cxd + uj * rxd * (1 + spread) - sy * 0.04 * w
            curl = c["curl"] * (1 if u >= 0 else -1) if abs(u) > 0.15 else 0.0
            Lk = ty - side_pt[1]
            tip = (tx + curl * 0.16 * Lk, ty - abs(curl) * 0.06 * Lk)
            mid = (side_pt[0] + (tx - side_pt[0]) * 0.55 + uj * spread * 0.2 * rxd, side_pt[1] + 0.52 * Lk)
            out.append(_lock("back", [root, side_pt, mid, tip], wk * rng.uniform(0.9, 1.1), "hang", tip=c["tip"],
                             wave=c["wave"] * h, wave_phase=rng.uniform(0, 1.5), drift=drift, drift_from=by,
                             depth=DEPTH["back"] + 4 * abs(u), behind_body=True))

    # ---- 2. tails
    if c["tails"] == "twin":
        # gathered at the upper side of the head, a bulge just below the gather, then a long taper
        for tag, s in (("L", -1.0), ("R", 1.0)):
            A = (cxd + s * 0.82 * rxd, by - 0.6 * ryd)
            for j, (wf, off, ln) in enumerate(((0.3, 0.0, 1.0), (0.2, 0.05, 0.9), (0.18, -0.05, 0.84))):
                Lt = c["tail_len"] * ln + rng.normal(0, jag)
                O = (cxd + s * (rxd + 0.06 * w), by - 0.66 * ryd)
                B = (cxd + s * (rxd + 0.22 * w + off * w), by + 0.05 * ryd)
                D = (cxd + s * (rxd + 0.26 * w + off * w), y0 + (0.3 + 0.35 * Lt) * h)
                E = (cxd + s * (rxd + 0.16 * w + off * w), y0 + (0.2 + 0.62 * Lt) * h)       # swings back in
                T = (cxd + s * (rxd + 0.22 * w + off * 1.6 * w + c["curl"] * 0.2 * w), y0 + Lt * h)   # and out
                out.append(_lock("tail", [A, O, B, D, E, T], wf * w * fat, "hang", root=0.3, t_max=0.2,
                                 tip=c["tip"], wave=c["wave"] * h, wave_phase=rng.uniform(0, 2), drift=drift,
                                 drift_from=by, depth=DEPTH["tail"] + 3 - j, tail=tag, behind_body=False))
    elif c["tails"] == "pony":
        # gathered high at the back of the head: the tail shows over the top on the far side and falls behind
        s = -1.0 if sy >= 0 else 1.0
        A = (cxd + s * 0.35 * rxd, by - 0.8 * ryd)
        for j, (wf, off, ln) in enumerate(((0.28, 0.0, 1.0), (0.19, 0.05, 0.92), (0.17, -0.05, 0.86))):
            Lt = c["tail_len"] * ln + rng.normal(0, jag)
            U = (cxd + s * 0.78 * rxd, by - 1.02 * ryd)
            O = (cxd + s * (rxd + 0.14 * w + off * w), by - 0.55 * ryd)
            D = (cxd + s * (rxd + 0.2 * w + off * w), y0 + (0.3 + 0.4 * Lt) * h)
            T = (cxd + s * (rxd + 0.1 * w + off * 2 * w), y0 + Lt * h)
            out.append(_lock("tail", [A, U, O, D, T], wf * w * fat, "hang", root=0.35, t_max=0.24, tip=c["tip"],
                             wave=c["wave"] * h, drift=drift, drift_from=by,
                             depth=DEPTH["tail"] + 3 - j, tail="P", behind_body=True))

    # ---- 3. crown: the dome over the skull, its hairline across the forehead (hidden by the fringe)
    phi = np.linspace(math.pi + 0.14, -0.14, 90)
    arc = np.stack([cxd + rxd * np.cos(phi), by - ryd * np.sin(phi)], 1)
    xl_f, xr_f = x0 + 0.04 * w, x1 - 0.04 * w
    yt = y0 + 0.42 * h
    hair_line = dc.catmull_rom(np.array([[xl_f, yt], [x0 + 0.2 * w, y0 + 0.27 * h], [g["cxl"], y0 + 0.2 * h],
                                         [x1 - 0.2 * w, y0 + 0.27 * h], [xr_f, yt]]), spacing=1.5)
    crown_poly = np.vstack([arc, [[cxd + rxd * 0.99, by + 0.2 * ryd]], hair_line[::-1],
                            [[cxd - rxd * 0.99, by + 0.2 * ryd]]])
    out.append({"layer": "crown", "path": arc, "width": np.full(len(arc), 0.0), "poly": crown_poly,
                "edges_px": [(dc.resample(arc, 1.5), "dome"), (dc.resample(hair_line, 1.5), "hairline")],
                "depth": DEPTH["crown"], "behind_body": False})

    # ---- 4. stray ends breaking the dome: short locks lying along it, their tips lifting off the outline
    ns = int(c["stray"])
    angs = {1: [140], 2: [142, 36], 3: [150, 96, 30]}.get(min(ns, 3), [])
    for j, a_deg in enumerate(angs):
        a = math.radians(a_deg + rng.uniform(-5, 5))
        big = c["tip"] > 1.4
        ln = (0.2 if big else 0.12) * h
        base = np.array([cxd + rxd * math.cos(a), by - ryd * math.sin(a)])
        out_dir = np.array([math.cos(a), -math.sin(a)])
        tang = np.array([math.sin(a), math.cos(a)]) * (-1.0 if a_deg > 90 else 1.0)    # downhill along the dome
        if abs(a_deg - 90) < 20:
            tang = np.array([math.copysign(1.0, rng.uniform(-1, 1)), 0.0])
        root = base - out_dir * 0.22 * ln - tang * 0.5 * ln
        tip = base + out_dir * (0.5 if big else 0.12) * ln + tang * (0.5 if big else 0.75) * ln
        mid = base + out_dir * 0.02 * ln + tang * 0.12 * ln
        out.append(_lock("stray", [root, mid, tip], (0.11 if big else 0.04) * w * fat, "tuft", tip=c["tip"],
                         depth=DEPTH["stray"] + j, behind_body=False))
    if c.get("ahoge"):
        # one antenna lock standing up from the crown and curling over
        top = np.array([P[0] - 0.03 * w, by - ryd])
        pts = [top + [0.0, 0.05 * h], top + [-0.005 * w, -0.05 * h], top + [0.03 * w, -0.11 * h],
               top + [0.09 * w, -0.1 * h]]
        out.append(_lock("stray", pts, 0.06 * w * fat, "tuft", root=0.75, t_max=0.3, tip=1.3,
                         depth=DEPTH["stray"] + 3, behind_body=False))

    # ---- 5. side locks: from the part over the skull, past the temple, inner edge through the corner of the jaw
    n_side = int(c["side"])
    side_len = max(0.66, c["side_len"])
    for tag, s, jaw in (("L", -1.0, g["jaw_l"]), ("R", 1.0, g["jaw_r"])):
        for j in range(n_side):
            wm = (0.2 if j == 0 else 0.16) * w * fat
            out_ = 0.12 * w * j
            # the first lock comes from the part down over the side of the skull; the ones outside it start under
            # it at the temple, so only one lock fans out from the part on each side
            root = (P[0], P[1] + 0.012 * h) if j == 0 else (cx + s * 0.24 * w, y0 + 0.17 * h)
            A = (cx + s * (0.27 + 0.07 * j) * w, y0 + (0.15 + 0.03 * j) * h)
            T = (cx + s * (0.47 + 0.05 * j) * w, y0 + 0.34 * h)
            J = (jaw[0] + s * (0.45 * wm + out_), jaw[1] + 0.04 * h * j)
            # the outer locks end shorter and flick out a little more: no row of equal spikes
            ty = y0 + side_len * h * (1 - 0.12 * j) + rng.normal(0, jag * h * 0.5)
            ty = max(ty, jaw[1] + 0.12 * h)
            curl = c["curl"] + 0.12 * j
            Lr = ty - J[1]
            E = (J[0] + s * (0.03 * w + 0.04 * Lr / h * w * max(0.0, c["spread"])), J[1] + 0.55 * Lr)
            tip = (J[0] + s * (0.01 * w + curl * 0.12 * Lr), ty - abs(curl) * 0.05 * Lr)
            lk = _lock("side", [root, A, T, J, E, tip], wm, "hang", root=0.12, t_max=0.32, tip=c["tip"],
                       wave=c["wave"] * h * 0.8, wave_phase=rng.uniform(0, 1.5), drift=drift,
                       drift_from=g["chin"][1], depth=DEPTH["side"] + (n_side - j), behind_body=False, tail=tag)
            if j == 0:
                _fit_inner_edge(lk, jaw, s, P)
            out.append(lk)

    # ---- 6. fringe: from the part, over the forehead, tips near the brow line
    nf = int(c["fringe"])
    if nf > 0:
        # the outermost bangs lie over the side locks, so no window of forehead opens between them
        xl, xr = g["jaw_l"][0] - 0.05 * w, g["jaw_r"][0] + 0.05 * w
        sp = (xr - xl) / nf
        brow = g["brow"]
        ip = int(np.argmin([abs(xl + (i + 0.5) * sp - P[0]) for i in range(nf)]))
        for i in range(nf):
            tx = xl + (i + 0.5) * sp + rng.uniform(-0.1, 0.1) * sp
            off = abs(tx - g["cxl"]) / (0.5 * w)
            ty = brow - c["fvar"] * (0.05 * h * off * off + abs(rng.normal(0, jag * 0.35 * h)))
            ty = P[1] + (ty - P[1]) * c["bang"]
            ty = min(ty, brow)
            dx = tx - P[0]
            sg = 0.0 if abs(dx) < 0.25 * sp else math.copysign(1.0, dx)
            far = min(1.0, abs(dx) / (0.35 * w))
            # from the part the lock follows the skull, bulging away from the part, then falls to its tip
            root = (P[0], P[1] + 0.01 * h)
            mid = (P[0] + dx * 0.5 + sg * 0.03 * w * far, P[1] + (ty - P[1]) * 0.45)
            near = (tx - sg * 0.008 * w, ty - 0.17 * (ty - P[1]))
            # longer bangs lie in front, shorter ones peek out between them (no edge runs across a shorter lock)
            rank = (ty - P[1]) / max(1.0, brow - P[1]) * 4.0 + 0.3 * (nf - abs(i - ip)) / nf
            # the separation lines show on the lower part only and fade out upward (kept shorter than the lines
            # easel's gap check looks at); the outermost edges run all the way up to the part
            L_lock = dc.path_length(_centre([root, mid, near, (tx, ty)], 3.0))
            cap = 1.0 - 0.085 * g["S"] / max(L_lock, 1.0)
            sep = (max(cap, rng.uniform(0.36, 0.52)), max(cap, rng.uniform(0.36, 0.52)))
            t0 = (0.0 if i == 0 else sep[0], 0.0 if i == nf - 1 else sep[1])
            out.append(_lock("fringe", [root, mid, near, (tx, ty)], sp * 1.42 * fat, "hang", root=0.1, t_max=0.42,
                             tip=max(1.0, c["tip"]), depth=DEPTH["fringe"] + rank + 0.01 * i, behind_body=False,
                             t0=t0))
    for k in out:
        k.setdefault("t0", (0.0, 0.0))
    return out


def _fit_inner_edge(lk: dict, jaw, s: float, P) -> None:
    """Move a side lock sideways below the temple so its inner edge passes through the corner of the jaw: the face
    outline (``line trace``) then starts on the lock's edge and the face stays closed."""
    for _ in range(3):
        left, right, _poly = lock_shape(lk["path"], lk["width"])
        inner = right if s < 0 else left
        t = _arc_t(lk["path"])
        cand = np.flatnonzero(t > 0.15)
        if not len(cand):
            return
        i = cand[int(np.argmin(np.abs(inner[cand, 1] - jaw[1])))]
        err = float(jaw[0] - inner[i, 0])
        if abs(err) < 0.3:
            return
        ti = t[i]
        ramp = _ss((t - 0.12) / max(1e-6, ti - 0.12))
        path = lk["path"].copy()
        path[:, 0] += err * ramp
        lk["path"] = path


# ============================================================================================ outlines -> pieces

def _edges(k: dict) -> List[dict]:
    """The drawn outline of a shape: a lock's two edges (root to tip, open at the root), the crown's dome and
    hairline. ``out`` = outward normals, ``tip`` = the end where the edges meet."""
    if "edges_px" in k:
        res = []
        for pts, kind in k["edges_px"]:
            if kind == "dome":
                c = np.array(k["centre"]) if "centre" in k else np.array([pts[:, 0].mean(), pts[:, 1].max()])
                out = pts - c
            else:
                out = np.tile([[0.0, 1.0]], (len(pts), 1))
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
            res.append({"pts": pts, "out": out, "kind": kind, "tip0": False, "tip1": False})
        return res
    left, right, _ = lock_shape(k["path"], k["width"])
    t = _arc_t(k["path"])
    res = []
    for side, E, t0 in (("left", left, k["t0"][0]), ("right", right, k["t0"][1])):
        i0 = int(np.searchsorted(t, t0)) if t0 > 0 else 0
        E = E[i0:]
        if len(E) < 3:
            continue
        pts = dc.resample(E, 1.5)
        n = _normals(pts)
        out = n if side == "left" else -n
        res.append({"pts": pts, "out": out, "kind": side, "tip0": False, "tip1": True})
    return res


def cut_pieces(shapes: List[dict], H: int, W: int, extra_occluders: Sequence[Tuple[float, np.ndarray, bool]] = ()):
    """Visible pieces of every drawn outline, back shapes cut where anything in front covers them.

    ``extra_occluders`` = (depth, mask, hides_only_behind_body_shapes) for the head and the body. Returns (pieces,
    masks by shape index, union of everything)."""
    items = []
    for i, k in enumerate(shapes):
        poly = k["poly"] if "poly" in k else lock_shape(k["path"], k["width"])[2]
        k["mask"] = _poly_mask(H, W, poly)
        items.append((k["depth"], "shape", i, k["mask"], False))
    for dep, m, body in extra_occluders:
        items.append((dep, "occ", -1, m, body))
    items.sort(key=lambda it: -it[0])
    acc = np.zeros((H, W), bool)
    body = np.zeros((H, W), bool)
    pieces = []

    def by_body(q) -> bool:
        x, y = int(math.floor(q[0])), int(math.floor(q[1]))
        return 0 <= x < W and 0 <= y < H and bool(body[y, x]) and not bool(acc[y, x])
    for dep, kind, i, m, is_body in items:
        if kind == "shape":
            k = shapes[i]
            front = acc | body if k.get("behind_body") else acc
            for e in _edges(k):
                n = len(e["pts"])
                for a, b in visible_runs(e["pts"], front):
                    pieces.append({"shape": i, "pts": e["pts"][a:b], "out": e["out"][a:b], "kind": e["kind"],
                                   "layer": k["layer"], "tip0": a == 0 and e["tip0"], "tip1": b == n and e["tip1"],
                                   "nat0": a == 0, "nat1": b == n,
                                   "body0": a > 0 and k.get("behind_body", False) and by_body(e["pts"][a - 1]),
                                   "body1": b < n and k.get("behind_body", False) and by_body(e["pts"][b])})
            acc |= m
        elif is_body:
            body |= m
        else:
            acc |= m
    union = acc | body
    return pieces, union


def _min_dist(q, P: np.ndarray) -> Tuple[float, int]:
    if not len(P):
        return 1e9, -1
    d = np.hypot(P[:, 0] - q[0], P[:, 1] - q[1])
    j = int(np.argmin(d))
    return float(d[j]), j


def finish_pieces(pieces: List[dict], union: np.ndarray, lw: float, S: float, others: List[np.ndarray],
                  hair: Optional[np.ndarray] = None) -> List[dict]:
    """Drop crumbs, decide the taper of every end (free ends taper, ends that run behind a line stay blunt), pull
    ends that stop just short of a line onto it (no small gaps), and split each piece where it leaves the outer
    silhouette (weight 1.4 outside, 1.0 inside). ``hair`` = every lock's area: a short bang edge whose upper end
    meets another line with hair all round it stops short of that line instead."""
    H, W = union.shape
    min_len = max(4.0 * lw, 0.01 * S)
    keep = []
    for pc in pieces:
        L = dc.path_length(pc["pts"])
        if L >= min_len or (L >= 2 * lw and (pc["tip0"] or pc["tip1"])):
            keep.append(pc)
    pieces = keep
    ext = [o for o in others if len(o)]
    checked = 0.1 * S          # easel's gap check looks at the ends of lines at least this long
    reach = 0.065 * S          # ... and reports an end that stops within this distance of another line
    for k, pc in enumerate(pieces):
        rest = [pieces[j]["pts"] for j in range(len(pieces)) if j != k]
        pool = np.vstack(rest + ext) if (rest or ext) else np.zeros((0, 2))
        for end in (0, 1):
            if pc[f"tip{end}"]:
                pc[f"free{end}"] = True            # a lock tip: both edges meet there, each tapers to the point
                continue
            q = pc["pts"][0 if end == 0 else -1]
            dmin, j = _min_dist(q, pool)
            L_pc = dc.path_length(pc["pts"])
            touching = dmin <= max(1.5 * lw, 2.5)
            if (end == 0 and hair is not None and pc.get("layer") == "fringe" and L_pc < 1.6 * checked
                    and ((touching and _hair_round(pc["pts"], hair, 3.0 * lw))
                         or (pc.get("nat0") and not touching and L_pc >= checked))):
                # a bang coming out from under the bang in front stops short of its edge, and a separation line
                # fading out between two bangs stays free: the bang behind stays one area with the rest of the
                # fringe, not a closed window that flat auto would give to the face (what is left stays shorter
                # than the lines whose free ends easel's gap check looks at)
                cut = _trim_start(pc["pts"], pool, 2.5 * lw + 1.0, L_pc - checked + 2.0 * lw)
                if cut:
                    pc["pts"] = pc["pts"][cut:]
                    if pc["out"] is not None:
                        pc["out"] = pc["out"][cut:]
                    pc["free0"] = True
                    continue
            if dmin <= max(1.5 * lw, 2.5):
                pc[f"free{end}"] = False           # runs behind a line: a blunt end that meets it
                continue
            L = dc.path_length(pc["pts"])
            if pc[f"body{end}"]:
                # runs behind the neck or shoulders: a blunt end their line will close; pulled onto a line it
                # stops just short of
                if dmin < 0.035 * S and L > 2.5 * dmin:
                    pc["pts"] = _pull_end(pc["pts"], end, pool[j], min(3 * dmin, 0.6 * L))
                pc[f"free{end}"] = False
                continue
            if L < checked or dmin >= reach:
                pc[f"free{end}"] = True            # a lock edge fading out (a short one, or far from any line)
                continue
            # a long line stopping near another one: a line cut where it goes behind something carries on until it
            # meets the next line; a root or the end of an outline is pulled onto the nearest line. No open end is
            # left for a fill to leak through.
            longer = None if pc[f"nat{end}"] else _extend_to(pc["pts"], end, pool, reach, lw)
            if longer is not None:
                if pc["out"] is not None:
                    pc["out0"] = pc["out"]
                pc["pts"] = longer
                pc["out"] = None
            else:
                pc["pts"] = _pull_end(pc["pts"], end, pool[j], min(3 * dmin, 0.45 * L))
            pc[f"free{end}"] = False
    out = []
    d_out = 0.9 * lw + 2.5
    for pc in pieces:
        pts, nrm = pc["pts"], pc["out"]
        if nrm is None or len(nrm) != len(pts):
            ref = pc.get("out0")
            nrm = _normals(pts)
            if ref is not None and float(nrm.mean(0) @ np.asarray(ref).mean(0)) < 0:
                nrm = -nrm
        q = pts + nrm * d_out
        xi = np.clip(np.floor(q[:, 0]).astype(int), -1, W)
        yi = np.clip(np.floor(q[:, 1]).astype(int), -1, H)
        on = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        inside = np.zeros(len(pts), bool)
        inside[on] = union[yi[on], xi[on]]
        sil = ~inside
        if pc["kind"] == "hairline":
            sil[:] = False
        if len(sil) >= 9:
            from scipy.ndimage import median_filter
            sil = median_filter(sil.astype(np.uint8), size=9, mode="nearest").astype(bool)
        # split at changes, merging bits shorter than min_seg into the neighbour before them
        cuts = [0] + [i for i in range(1, len(sil)) if sil[i] != sil[i - 1]] + [len(sil)]
        segs = [(a, b, bool(sil[a])) for a, b in zip(cuts[:-1], cuts[1:])]
        min_seg = max(6, int(0.02 * S / 1.5))
        merged: List[list] = []
        for a, b, sv in segs:
            if merged and (b - a < min_seg or merged[-1][2] == sv):
                merged[-1][1] = b
            else:
                merged.append([a, b, sv])
        if len(merged) > 1 and merged[0][1] - merged[0][0] < min_seg:
            merged[1][0] = merged[0][0]
            merged.pop(0)
        for m_i, (a, b, sv) in enumerate(merged):
            a2 = max(0, a - 1) if m_i > 0 else a
            seg = pts[a2:b]
            if len(seg) < 2:
                continue
            free0 = pc["free0"] if m_i == 0 else False
            free1 = pc["free1"] if m_i == len(merged) - 1 else False
            out.append({"pts": seg, "weight": OUTER_WEIGHT if sv else INNER_WEIGHT, "free0": free0,
                        "free1": free1, "shape": pc["shape"], "kind": pc["kind"]})
    return out


def drop_floating(final: List[dict], lw: float, S: float, below: float, share: float = 0.25) -> List[dict]:
    """Drop the pieces that float apart from the hair: pieces are joined when they come within a few line widths
    of each other; a group much shorter than the largest one that lies wholly below ``below`` (the chin, px) is a
    back lock showing again beside a shoulder or an arm that hides its upper part, and is left out. Short lines on
    the head (a parting, a stray end) stay."""
    if len(final) < 2:
        return final
    from scipy.spatial import cKDTree
    tol = max(3.0 * lw, 0.012 * S)
    trees = [cKDTree(np.asarray(pc["pts"], float)) for pc in final]
    parent = list(range(len(final)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(final)):
        for j in range(i + 1, len(final)):
            if root(i) != root(j) and trees[i].count_neighbors(trees[j], tol) > 0:
                parent[root(j)] = root(i)
    length: Dict[int, float] = {}
    top: Dict[int, float] = {}
    for i, pc in enumerate(final):
        P = np.asarray(pc["pts"], float)
        length[root(i)] = length.get(root(i), 0.0) + dc.path_length(P)
        top[root(i)] = min(top.get(root(i), 1e9), float(P[:, 1].min()))
    if len(length) < 2:
        return final
    main = max(length.values())
    return [pc for i, pc in enumerate(final) if length[root(i)] >= share * main or top[root(i)] <= below]


def _hair_round(pts: np.ndarray, hair: np.ndarray, r: float) -> bool:
    """True when hair lies on both sides of a piece's first point and beyond it (cutting it short joins hair to
    hair, never hair to skin)."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return False
    tg = P[min(6, len(P) - 1)] - P[0]
    nt = float(np.hypot(*tg))
    if nt < 1e-6:
        return False
    tg = tg / nt
    nrm = np.array([-tg[1], tg[0]])
    H, W = hair.shape
    for s in (P[0] + nrm * r, P[0] - nrm * r, P[0] - tg * r):
        x, y = int(math.floor(s[0])), int(math.floor(s[1]))
        if not (0 <= x < W and 0 <= y < H and hair[y, x]):
            return False
    return True


def _trim_start(pts: np.ndarray, pool: np.ndarray, gap: float, min_drop: float = 0.0) -> int:
    """How many points to drop from the start so the piece begins ``gap`` px away from every pool point and at
    least ``min_drop`` px along it (0 when that would take more than 45% of it)."""
    P = np.asarray(pts, float)
    q = P[0]
    near = pool[(np.abs(pool[:, 0] - q[0]) < 6 * gap + 80) & (np.abs(pool[:, 1] - q[1]) < 6 * gap + 80)]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    for i in range(1, len(P)):
        if s[i] > 0.45 * s[-1]:
            return 0
        if s[i] < min_drop:
            continue
        if not len(near) or float(np.hypot(near[:, 0] - P[i, 0], near[:, 1] - P[i, 1]).min()) >= gap:
            return i if len(P) - i >= 3 else 0
    return 0


def _extend_to(pts: np.ndarray, end: int, pool: np.ndarray, reach: float, lw: float) -> Optional[np.ndarray]:
    """The line carried on along its own direction from ``end`` until it touches a point of ``pool``; None when it
    meets nothing within ``reach``."""
    P = np.asarray(pts, float)
    if len(P) < 4 or not len(pool):
        return None
    if end == 0:
        P = P[::-1]
    k = min(len(P) - 1, 8)
    d = P[-1] - P[-1 - k]
    nd = float(np.hypot(*d))
    if nd < 1e-6:
        return None
    d /= nd
    steps = np.arange(1, int(reach / 1.5) + 1) * 1.5
    cand = P[-1] + np.outer(steps, d)
    lo, hi = cand.min(0) - 3 * lw, cand.max(0) + 3 * lw
    near = pool[(pool[:, 0] >= lo[0]) & (pool[:, 0] <= hi[0]) & (pool[:, 1] >= lo[1]) & (pool[:, 1] <= hi[1])]
    if not len(near):
        return None
    D = np.hypot(cand[:, None, 0] - near[None, :, 0], cand[:, None, 1] - near[None, :, 1]).min(1)
    hit = np.flatnonzero(D < 0.8 * lw)
    if not len(hit):
        return None
    out = np.vstack([P, cand[: int(hit[0]) + 1]])
    return out if end == 1 else out[::-1]


def _pull_end(pts: np.ndarray, end: int, target, reach: float) -> np.ndarray:
    P = np.asarray(pts, float)
    q = P[0] if end == 0 else P[-1]
    off = np.asarray(target, float) - q
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    if end == 1:
        s = s[-1] - s
    wgt = np.clip(1 - s / max(reach, 1e-6), 0, 1) ** 2
    return P + wgt[:, None] * off[None, :]


def _taper(free0: bool, free1: bool) -> str:
    return {(True, True): "both", (True, False): "start", (False, True): "end"}.get((free0, free1), "none")


# ============================================================================================ the command

def _part_name(ctx, a, usage: str) -> str:
    p = ctx.project
    d = dc.dstate(p)
    name = str(a).strip().lower()
    if name not in d["parts"]:
        hair = [n for n in d["parts"] if any(w in n for w in ("hair", "bang", "fringe", "tail", "mane"))]
        known = ", ".join(d["parts"]) or "none yet"
        if hair:
            raise UsageError(f"{name} is not a planned part; hair parts: {', '.join(hair)} (parts: {known})",
                             usage.replace(" hair ", f" {hair[0]} ", 1))
        raise UsageError(f"{name} is not a planned part (parts: {known}); plan the hair first",
                         "easel plan part hair --box 150,40,850,920")
    return name


def _cut_params(a, ctx) -> Tuple[str, dict]:
    cut = str(a.get("cut") or "long").strip().lower().replace("-", "").replace("_", "")
    alias = {"pony": "ponytail", "twintails": "twin", "twintail": "twin", "pigtails": "twin", "curly": "wavy",
             "straight": "long", "bobcut": "bob", "spikey": "spiky", "pixie": "short"}
    cut = alias.get(cut, cut)
    if cut not in CUTS:
        raise UsageError(f"unknown cut {a.get('cut')!r}; cuts: {', '.join(CUT_NAMES)}",
                         "easel locks hair --cut long")
    c = dict(CUTS[cut])
    from .cli_draw import _num
    if a.get("length") is not None:
        L = _num(a, "length", None, 0.5, 6.0, ctx)
        key = "tail_len" if c["tails"] else "length"
        f = L / c[key]
        c[key] = L
        if c["tails"]:
            c["length"] = min(c["length"], L)
        else:
            c["side_len"] = max(0.66, c["side_len"] * min(1.0, f) if f < 1 else c["side_len"] * (0.5 + 0.5 * f))
    if a.get("volume") is not None:
        c["volume"] = _num(a, "volume", 1.08, 0.9, 1.6, ctx)
    if a.get("fringe") is not None:
        c["fringe"] = int(round(_num(a, "fringe", 5, 0, 12, ctx)))
    return cut, c


def _part_point(ctx, g: dict, a) -> Tuple[np.ndarray, Optional[str]]:
    p = ctx.project
    cxd, by, rxd, ryd = g["cx"] - g["sy"] * 0.05 * g["w"], g["by"], g["rx"] * 1.05, g["ry"] * 1.06
    if a.get("part"):
        u = coords.parse_point(a["part"], ctx.notes, "easel locks hair --part 520,90")
        P = dc.u2px(p, [u])[0]
    else:
        P = np.array([g["cxl"] + 0.06 * g["w"], g["y0"] + 0.045 * g["h"]])
    note = None
    low = g["y0"] + 0.3 * g["h"]
    if P[1] > low:                                   # a part sits on the top of the head, not on the face
        P = np.array([P[0], g["y0"] + 0.06 * g["h"]])
        note = "low"
    r = math.sqrt(_ellipse_inside(P, (cxd, by), rxd, ryd))
    if r > 0.92:
        ang = math.atan2((P[1] - by) / ryd, (P[0] - cxd) / rxd)
        P = np.array([cxd + 0.88 * rxd * math.cos(ang), by + 0.88 * ryd * math.sin(ang)])
        note = "out"
    if note:
        note = f"the part is moved onto the top of the head: {dc.fmt_pt(p, P)}"
    return P, note


def cmd_locks(ctx, args: List[str]):
    from . import cli_draw
    if args and args[0].lower() == "shine":
        return cmd_shine(ctx, args[1:])
    from .cli import Spec, parse
    p = ctx.project
    d = dc.dstate(p)
    if not args:
        raise UsageError("say which part is the hair: easel locks <hair part> [--cut long|bob|short|spiky|twin|"
                         "ponytail|wavy] [--part x,y]", USAGE["locks"])
    a = parse(args, Spec(USAGE["locks"], ["region"], {"cut": "value", "part": "value", "flow": "value",
                                                      "length": "value", "volume": "value", "fringe": "value",
                                                      "seed": "value"},
                         alias={"style": "cut", "parting": "part", "len": "length", "bangs": "fringe"}, min_pos=1))
    part = _part_name(ctx, a["region"], USAGE["locks"])
    cut, c = _cut_params(a, ctx)
    seed = int(cli_draw._num(a, "seed", 1, 0, 1e6, ctx))
    g = frame(p, part, c)
    P, pnote = _part_point(ctx, g, a)
    flow = dc.u2px(p, coords.parse_points(a["flow"], ctx.notes, 2, 'easel locks hair --flow "500,100 560,400 '
                                                                     '600,800"')) if a.get("flow") else None
    lines: List[str] = []
    new_ids: List[str] = []

    def mutate():
        res = _build(p, part, cut, c, seed, P, flow, g)
        new_ids.extend(res["ids"])
        lines.extend(_report(p, part, cut, c, g, P, res, pnote, d))
        note = _grow_box(p, part, res["shapes"])
        if note:
            lines.append(note)
        return res["ops"], res["bbox"]
    label = f"locks {part} cut {cut}"
    rep = cli_draw.draw_step(ctx, label=label, layer="line", owned=cli_draw._line_owned(p, mutate), lines=lines,
                             post=cli_draw._line_post(p), overlay=cli_draw._line_overlay(p, ids=[]),
                             note="the hair locks (gaps magenta)", area_extra=cli_draw._subject_box_px(p))
    if new_ids:
        rep.lines.insert(1, f"new: {new_ids[0]}-{new_ids[-1]} ({len(new_ids)} lock lines, part {part})")
    return rep


def _build(p, part: str, cut: str, c: dict, seed: int, P: np.ndarray, flow, g: dict) -> dict:
    """Generate the locks and put their lines in place of the part's earlier lock lines (the mutate of easel locks).
    Neck and clothes lines that pass under a lock in front are cut there; the back hair is cut where the body and
    the clothes hide it. Returns ops, touched box, shapes, pieces, new ids and what was hidden."""
    from . import cli_draw, draw_line
    d = dc.dstate(p)
    H, W = p.H, p.W
    S = float(max(H, W))
    lw = draw_line.base_width(p)
    rng = np.random.default_rng(seed * 7919 + CUT_NAMES.index(cut))
    shapes = generate(g, c, P, rng, flow)
    box = [None]

    def touch(path):
        box[0] = coords.union_box(box[0], cli_draw._path_box(p, path))

    for r in _restore_hidden(d, part):
        touch(draw_line.record_path(p, r))
    old = [r for r in d["lines"] if r.get("src") == f"locks {part}"]
    for r in old:
        touch(draw_line.record_path(p, r))
        d["lines"].remove(r)
    head_m = _poly_mask(H, W, g["ball"]) | _poly_mask(H, W, g["jaw"])
    occ = [(DEPTH["head"], head_m, False)]
    body, under = _body_shapes(g, H, W)
    for poly in body + under:
        occ.append((DEPTH["body"], _poly_mask(H, W, poly), True))
    names = _body_parts(p, part, g, head_m)
    body_recs = [r for r in d["lines"] if r.get("part") in names and not str(r.get("src", "")).startswith("locks ")]
    clothes, clothes_u = _clothes_solid(p, body_recs, H, W)
    if clothes.any():
        occ.append((DEPTH["body"], clothes, True))
    front = np.zeros((H, W), bool)
    for k in shapes:
        if k["layer"] in FRONT or (k["layer"] == "tail" and not k.get("behind_body")):
            front |= _poly_mask(H, W, lock_shape(k["path"], k["width"])[2])
    removed, added = _hide_lines(p, body_recs, front, part, lw)
    for r in removed:
        touch(draw_line.record_path(p, r))
        d["lines"].remove(r)
    d["lines"].extend(added)
    for k in shapes:
        if k["layer"] == "crown":
            k["centre"] = (g["dome"][0], g["dome"][1])
    pieces, union = cut_pieces(shapes, H, W, occ)
    hair = np.zeros((H, W), bool)
    for k in shapes:
        hair |= k["mask"]
    others = [draw_line.record_path(p, r) for r in d["lines"]]
    chin = float(np.asarray(g["jaw"], float)[:, 1].max())
    final = drop_floating(finish_pieces(pieces, union | head_m, lw, S, others, hair), lw, S, chin)
    ops, ids = [], []
    for pc in final:
        ctrl_u = dc.px2u(p, dc.rdp(pc["pts"], 0.6))
        path, _ = draw_line.fit(p, ctrl_u, closed=False, snap=False, join=False, bow=0.0)
        rec = draw_line.make_record(p, path, ctrl_u, weight=pc["weight"], taper=_taper(pc["free0"], pc["free1"]),
                                    color="line", closed=False, src=f"locks {part}")
        rec["part"] = part
        d["lines"].append(rec)
        ids.append(rec["id"])
        ops += draw_line.render_ops(p, rec)
        touch(path)
    for r in added:
        ops += draw_line.render_ops(p, r)
    d.setdefault("locks", {})[part] = _record(p, part, cut, c, g, P, seed, flow, shapes, body, ids, clothes_u,
                                              removed)
    return {"ops": ops, "bbox": box[0], "shapes": shapes, "final": final, "ids": ids, "n_old": len(old),
            "hidden": removed, "clothes": bool(clothes.any()), "body_lines": len(body_recs)}


def refit(p) -> List[str]:
    """Grow the recorded locks again over lines added since (``easel wear``, ``easel line`` on the neck or the
    clothes): the locks hide the new lines where they pass under them and the back hair goes behind the new
    clothes. Call it inside the mutate of the command that added the lines; returns lines for its report."""
    d = dc.dstate(p)
    out = []
    for part, rec in list((d.get("locks") or {}).items()):
        if part not in d["parts"] or not rec.get("params"):
            continue
        new = [r for r in d["lines"] if r.get("n", 0) > rec.get("n_at", 10 ** 9)
               and not str(r.get("src", "")).startswith("locks ")]
        if not new:
            continue
        c = dict(rec["params"])
        g = frame(p, part, c)
        head_m = _poly_mask(p.H, p.W, g["ball"]) | _poly_mask(p.H, p.W, g["jaw"])
        names = _body_parts(p, part, g, head_m)
        if not any(r.get("part") in names for r in new):
            continue
        P = dc.u2px(p, [rec["part"]])[0]
        flow = dc.u2px(p, rec["flow"]) if rec.get("flow") else None
        res = _build(p, part, rec["cut"], c, int(rec.get("seed", 1)), P, flow, g)
        n = len(res["hidden"])
        out.append(f"locks {part} laid again over the new lines" + (f": {n} lines that pass under the locks are cut"
                                                                    if n else ""))
    return out


def _body_parts(p, part: str, g: dict, head_m: np.ndarray) -> List[str]:
    """The parts of the hair's subject below the head (neck, clothes, shoulders): the side locks hang in front of
    them, the back hair behind them."""
    d = dc.dstate(p)
    subj = d["parts"].get(part, {}).get("of")
    low = min(g["jaw_l"][1], g["jaw_r"][1])
    out = []
    for n, v in d["parts"].items():
        if n == part or n in NOT_BODY or any(w in n for w in HAIRISH):
            continue
        if subj and v.get("of") and v.get("of") != subj:
            continue
        b = dc.box_px(p, v["box"])
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        xi, yi = int(np.clip(cx, 0, p.W - 1)), int(np.clip(cy, 0, p.H - 1))
        if cy > low and not head_m[yi, xi]:
            out.append(n)
    return out


def _clothes_solid(p, recs: List[dict], H: int, W: int) -> Tuple[np.ndarray, List[list]]:
    """The body and the clothes as one solid: the outlines easel wear keeps with its lines (``shape``) and closed
    clothes lines, else the span between the outermost neck and clothes lines, row by row. Returns (mask, polygons
    in units)."""
    from . import draw_line
    m = np.zeros((H, W), bool)
    polys: List[list] = []
    seen = set()
    for r in recs:
        sh = r.get("shape")
        if (not sh or len(sh) < 3) and r.get("closed") and len(r.get("path") or []) >= 3:
            sh = r["path"]
        if not sh or len(sh) < 3:
            continue
        key = (len(sh), tuple(sh[0]), tuple(sh[len(sh) // 2]))
        if key in seen:
            continue
        seen.add(key)
        m |= _poly_mask(H, W, dc.u2px(p, sh))
        polys.append(sh)
    if polys:
        return m, polys
    paths = [draw_line.record_path(p, r) for r in recs]
    paths = [q for q in paths if len(q) >= 2]
    if not paths:
        return m, []
    Q = np.vstack(paths)
    yi = np.round(Q[:, 1]).astype(int)
    ok = (yi >= 0) & (yi < H)
    if ok.sum() < 2:
        return m, []
    lo = np.full(H, np.inf)
    hi = np.full(H, -np.inf)
    np.minimum.at(lo, yi[ok], Q[ok, 0])
    np.maximum.at(hi, yi[ok], Q[ok, 0])
    rows = np.flatnonzero(np.isfinite(lo))
    ys = np.arange(rows[0], rows[-1] + 1)
    left = np.interp(ys, rows, lo[rows])
    right = np.interp(ys, rows, hi[rows])
    xs = np.arange(W)[None, :]
    m[ys[0]:ys[-1] + 1] = (xs >= left[:, None]) & (xs <= right[:, None])
    step = max(1, len(ys) // 60)
    poly = np.vstack([np.stack([left[::step], ys[::step]], 1), np.stack([right[::step], ys[::step]], 1)[::-1]])
    return m, [_u(p, poly)]


def _hide_lines(p, recs: List[dict], front: np.ndarray, part: str, lw: float) -> Tuple[List[dict], List[dict]]:
    """Cut neck and clothes lines where they pass under a lock in front: (records removed, pieces added). The
    longest piece keeps the record's id; every piece keeps its weight, colour, source and part."""
    from . import draw_line
    H, W = front.shape
    S = float(max(H, W))
    removed, added = [], []
    for r in recs:
        P = draw_line.record_path(p, r)
        if len(P) < 2:
            continue
        xi = np.floor(P[:, 0]).astype(np.int64)
        yi = np.floor(P[:, 1]).astype(np.int64)
        on = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        hid = np.zeros(len(P), bool)
        hid[on] = front[yi[on], xi[on]]
        if not hid.any():
            continue
        closed = bool(r.get("closed"))
        if closed:
            P = np.roll(P, -int(np.flatnonzero(hid)[0]), axis=0)
        tp = str(r.get("taper", "both"))
        wgt = float(r.get("weight", 1.0))
        keep = []
        for a, b in visible_runs(P, front):
            seg = P[a:b]
            L = dc.path_length(seg) if len(seg) >= 2 else 0.0
            if L >= max(4.0 * lw * wgt, 0.008 * S):
                keep.append((L, seg, not closed and a == 0 and tp in ("both", "start"),
                             not closed and b == len(P) and tp in ("both", "end")))
        removed.append(r)
        keep.sort(key=lambda q: -q[0])
        for j, (_L, seg, f0, f1) in enumerate(keep):
            nr = draw_line.make_record(p, seg, dc.px2u(p, dc.rdp(seg, 1.0)), weight=wgt, taper=_taper(f0, f1),
                                       color=r.get("color") or "line", closed=False, src=r.get("src", ""))
            for key, v in r.items():
                if key not in nr and key not in ("id", "n"):
                    nr[key] = copy.deepcopy(v)
            nr["part"] = r.get("part")
            nr["under_locks"] = part
            nr["from_id"] = r["id"]
            if j == 0:
                nr["id"], nr["n"] = r["id"], r["n"]
            added.append(nr)
    return removed, added


def _restore_hidden(d: dict, part: str) -> List[dict]:
    """Put back the lines the part's last locks cut (the new locks cut them again where they cover them). A line
    whose command drew it again since (easel wear once more) is left as that command made it."""
    prev = (d.get("locks") or {}).get(part) or {}
    hid = prev.get("hidden") or []
    if not hid:
        return []
    n_at = prev.get("n_at", 10 ** 9)
    redrawn = {r.get("src") for r in d["lines"] if r.get("n", 0) > n_at and r.get("src")}
    back = [r for r in hid if not (r.get("src") and r.get("src") in redrawn)]
    ids = {r["id"] for r in back}
    d["lines"][:] = [r for r in d["lines"] if not (r.get("under_locks") == part and r.get("from_id") in ids)]
    have = {r["id"] for r in d["lines"]}
    out = []
    for r in back:
        if r["id"] not in have:
            d["lines"].append(copy.deepcopy(r))
            out.append(r)
    return out


def _grow_box(p, part: str, shapes: List[dict]) -> Optional[str]:
    """The plan box of the hair part holds every lock, so flat auto gives their areas to the hair."""
    d = dc.dstate(p)
    pts = np.vstack([k["poly"] if "poly" in k else lock_shape(k["path"], k["width"])[2] for k in shapes])
    pts = np.clip(pts, [0, 0], [p.W, p.H])
    lo = dc.px2u(p, [pts.min(0)])[0]
    hi = dc.px2u(p, [pts.max(0)])[0]
    b = list(d["parts"][part]["box"])
    nb = [min(b[0], lo[0]), min(b[1], lo[1]), max(b[2], hi[0]), max(b[3], hi[1])]
    nb = [round(float(np.clip(v, 0, 1000)), 1) for v in nb]
    if max(abs(nb[i] - b[i]) for i in range(4)) > 5:
        d["parts"][part]["box"] = nb
        return f"plan part {part} box grown to hold every lock: {coords.fmt_box(nb)}"
    return None


def _u(p, pts) -> list:
    return dc.rnd(dc.px2u(p, pts), 1)


def _record(p, part, cut, c, g, P, seed, flow, shapes, body, ids, clothes=(), hidden=()) -> dict:
    S = float(max(p.W, p.H))
    locks = []
    for i, k in enumerate(sorted(shapes, key=lambda k: k["depth"])):
        path = np.asarray(k["path"], float)
        idx = _keep_idx(path, 0.004 * S)
        item = {"id": f"k{i + 1}", "layer": k["layer"], "order": i, "path": _u(p, path[idx]),
                "width": [round(float(v) * 1000.0 / S, 2) for v in np.asarray(k["width"])[idx]]}
        if k.get("tail"):
            item["tail"] = k["tail"]
        if k["layer"] == "crown":
            item["poly"] = _u(p, dc.rdp(k["poly"], 0.004 * S))
        locks.append(item)
    cxd, byd, rxd, ryd = g["dome"]
    return {"cut": cut, "part": [round(float(v), 1) for v in dc.px2u(p, [P])[0]], "seed": seed,
            "source": g["source"], "volume": round(c["volume"], 3),
            "length": round(c["tail_len"] if c["tails"] else c["length"], 2), "fringe": int(c["fringe"]),
            "flow": _u(p, flow) if flow is not None else None,
            "dome": [round(v, 1) for v in (cxd * 1000 / p.W, byd * 1000 / p.H, rxd * 1000 / p.W, ryd * 1000 / p.H)],
            "head_box": [round(float(v), 1) for v in g["head_box"]],
            "head": [_u(p, dc.rdp(np.asarray(q, float), 0.002 * S)) for q in (g["ball"], g["jaw"])],
            "head_id": g["head_id"],
            "body": [_u(p, dc.rdp(b, 0.002 * S)) for b in body], "clothes": [list(q) for q in clothes],
            "locks": locks, "lines": list(ids), "params": {k: v for k, v in c.items()},
            "hidden": [copy.deepcopy(r) for r in hidden],
            "n_at": max([int(r.get("n", 0)) for r in dc.dstate(p)["lines"]] or [0])}


def _keep_idx(path: np.ndarray, eps: float) -> np.ndarray:
    """Indices of a decimated path (root and tip always kept), so widths stay paired with points."""
    n = len(path)
    if n <= 2:
        return np.arange(n)
    step = max(1, int(round(eps / 1.5 * 4)))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return np.asarray(idx)


def _report(p, part, cut, c, g, P, res, pnote, d) -> List[str]:
    shapes, final, n_old = res["shapes"], res["final"], res["n_old"]
    cnt: Dict[str, int] = {}
    for k in shapes:
        cnt[k["layer"]] = cnt.get(k["layer"], 0) + 1
    words = []
    for lay, word in (("back", "back"), ("tail", "tail"), ("side", "side"), ("fringe", "fringe"),
                      ("stray", "stray end")):
        if cnt.get(lay):
            words.append(f"{cnt[lay]} {word}" + ("s" if lay in ("tail", "stray") and cnt[lay] > 1 else ""))
    yaw = math.degrees(g["yaw"])
    out = [f"locks {part}: cut {cut}, part {dc.fmt_pt(p, P)}, grown on the {g['source']}"
           + (f" (yaw {yaw:g})" if abs(yaw) > 0.5 else "") + ": " + ", ".join(words) + ", crown"]
    n_out = sum(1 for pc in final if pc["weight"] == OUTER_WEIGHT)
    out.append(f"each visible piece is one line: {len(final) - n_out} inside (weight {INNER_WEIGHT:g}), {n_out} on "
               f"the outer silhouette (weight {OUTER_WEIGHT:g}); pieces behind a lock in front are cut away")
    if n_old:
        out.append(f"replaced the {n_old} lines of the earlier  locks {part}")
    if pnote:
        out.append(pnote)
    if g["source"] == "hair box":
        out.append(f"no head construction: the head is fitted in the {part} box; sketch head first for a better fit")
    hand = [r["id"] for r in d["lines"] if r.get("part") == part and r.get("src") != f"locks {part}"
            and not str(r.get("src", "")).startswith("locks ")]
    if hand:
        out.append(f"older {part} lines {', '.join(hand[:8])} are still there; if the locks replace them:  "
                   f"easel line delete {','.join(hand[:8])}")
    hid = res["hidden"]
    if hid:
        names = ", ".join(sorted({str(r.get("part")) for r in hid}))
        out.append(f"the locks in front cover {len(hid)} {names} lines ({', '.join(r['id'] for r in hid[:6])}"
                   f"{' ...' if len(hid) > 6 else ''}): they are cut where the locks lie over them")
    if res["clothes"]:
        out.append("the back hair hangs behind the body and the clothes: its lines stop at their outline")
    elif g["shoulders"] is not None and c["back"]:
        out.append("the back hair runs behind the neck and shoulders: their lines close it")
    if not res["body_lines"] and g["shoulders"] is not None:
        out.append(f"no neck or clothes lines yet: draw them (easel wear ...) and then run this  easel locks {part}  "
                   "command again, so the locks lie over them")
    out.append(f"change it:  --cut {'|'.join(n for n in CUT_NAMES if n != cut)}  --part x,y (where it parts)  "
               "--volume 1.2  --fringe 3  --length 1.5  --flow \"x,y x,y x,y\"  --seed 2  (replaces these lines)")
    out.append(f"after flat and shade:  easel locks shine {part}   (shadow, shine and strands along these locks)")
    return out


# ============================================================================================ shine

def _shine_masks(p, rec: dict) -> Tuple[List[dict], np.ndarray]:
    """Each stored lock in px with its own visible area (the locks in front cut away)."""
    H, W = p.H, p.W
    S = float(max(H, W))
    items = []
    for k in rec["locks"]:
        path = dc.u2px(p, k["path"])
        widths = np.asarray(k["width"], float) * S / 1000.0
        if len(path) < 2:
            continue
        dense = dc.resample(path, 1.5)
        tt = _arc_t(dense)
        t_src = _arc_t(path)
        wd = np.interp(tt, t_src, widths)
        if k["layer"] == "crown":
            poly = dc.u2px(p, k["poly"]) if k.get("poly") else None
            if poly is None:
                continue
            m = _poly_mask(H, W, poly)
        else:
            m = _poly_mask(H, W, lock_shape(dense, wd)[2])
        items.append({"k": k, "path": dense, "width": wd, "mask": m, "order": k["order"]})
    acc = np.zeros((H, W), bool)
    for it in sorted(items, key=lambda it: -it["order"]):
        it["vis"] = it["mask"] & ~acc
        acc |= it["mask"]
    head = np.zeros((H, W), bool)
    for q in rec.get("head") or []:
        head |= _poly_mask(H, W, dc.u2px(p, q))
    body = np.zeros((H, W), bool)
    for q in list(rec.get("body") or []) + list(rec.get("clothes") or []):
        body |= _poly_mask(H, W, dc.u2px(p, q))
    for it in items:
        if it["k"]["layer"] in ("back", "tail"):
            it["vis"] &= ~head                      # behind the head
        if it["k"]["layer"] == "back" or it["k"].get("tail") == "P":
            it["vis"] &= ~body                      # behind the neck and shoulders
    union = np.zeros((H, W), bool)
    for it in items:
        union |= it["vis"]
    return items, union


def _lost_tips(p, items: List[dict], hair_r: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """The thin ends of the locks that flat gave to the area beside them (closing the gaps between lines swallows
    any part of a shape narrower than the gap): thin pieces of the locks' visible area outside the hair region,
    touching it, off the lines. A wide area given to another part is left alone."""
    import cv2
    from scipy import ndimage as ndi

    from .draw_flat import default_gap
    from .draw_line import base_width
    H, W = hair_r.shape
    lock = np.zeros((H, W), bool)
    for it in items:
        if it["k"]["layer"] != "crown":
            lock |= it["vis"]
    lost = lock & ~hair_r & ~lines
    if not lost.any():
        return lost
    r = int(default_gap(p) + base_width(p) / 2 + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    wide = cv2.morphologyEx(lost.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
    thin = lost & ~(cv2.dilate(wide.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    lab, n = ndi.label(thin)
    if not n:
        return thin
    near = cv2.dilate(hair_r.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    ids = np.unique(lab[near & (lab > 0)])
    return np.isin(lab, ids[ids > 0])


def shine_ops(p, rec: dict, clip: np.ndarray, base, shadow, light_rgb, light: np.ndarray, strands: int = 2,
              seed: int = 1, lost: Optional[np.ndarray] = None) -> Tuple[list, Dict[str, int]]:
    """The back mass a step darker, a tapered shadow strip along each lock's side away from the light (a crescent on
    the crown), the shine ring across the hair at a third of the head height (one pointed piece per lock, bent along
    the lock, so the ring breaks only at the lock edges), strands along the centre paths."""
    from .draw_line import base_width
    from .ops import Fill, Stroke, mask_to_rle
    H, W = p.H, p.W
    S = float(max(H, W))
    rng = np.random.default_rng(seed)
    items, _ = _shine_masks(p, rec)
    l2 = np.array([light[0], light[1]], float)
    l2 = l2 / (np.linalg.norm(l2) + 1e-9) if np.linalg.norm(l2) > 1e-6 else np.array([0.0, -1.0])
    hb = dc.box_px(p, rec["head_box"])
    hx0, hy0, hx1, hy1 = hb
    hw, hh = hx1 - hx0, hy1 - hy0
    cxd, byd, rxd, ryd = (rec["dome"][0] * W / 1000, rec["dome"][1] * H / 1000, rec["dome"][2] * W / 1000,
                          rec["dome"][3] * H / 1000)
    dark_line = dc.shadow_of(shadow, 1.0)
    hi = dc.light_of(light_rgb, 1.15)
    ops: list = []
    count = {"shadow": 0, "shine": 0, "strands": 0}
    fe = max(1.0, 0.0025 * S)
    crisp = max(1.0, 0.6 * fe)                     # cel edges: shadow strips and the ring stay sharp

    def fill(mask, rgb, opacity, feather=fe):
        mm = mask & clip
        if mm.sum() < 14:
            return False
        ops.append(Fill(mask_rle=mask_to_rle(mm), color=tuple(int(v) for v in rgb), opacity=float(opacity),
                        feather=max(1.0, float(feather))))
        return True

    # 0. the thin lock ends flat gave to the area beside them get the hair colour back: the colour the shaded hair
    #    has just above each end
    if lost is not None and lost.any():
        import cv2
        from scipy import ndimage as ndi
        pic = p.composite("export", upto="line")           # the colours under the line art: flat and shade
        lab, n_l = ndi.label(lost)
        for c_i in range(1, n_l + 1):
            m_c = lab == c_i
            ring = (cv2.dilate(m_c.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & clip & ~lost
            rgb = np.median(pic[ring], axis=0) * 255.0 if ring.sum() >= 6 else np.asarray(base, float)
            if fill(m_c, np.clip(rgb, 0, 255), 1.0, 0.8):
                count["tips"] = count.get("tips", 0) + 1
    # 1. the back mass and the tails behind the head sit in the head's shadow
    back = np.zeros((H, W), bool)
    for it in items:
        if it["k"]["layer"] in ("back",) or (it["k"]["layer"] == "tail" and it["k"].get("tail") == "P"):
            back |= it["vis"]
    if fill(back, shadow, 0.45, fe * 2):
        count["shadow"] += 1
    # 2. a shadow strip along each lock's side away from the light, thin at both ends; on the crown a crescent on
    #    the far side of the dome
    for it in items:
        k = it["k"]
        if k["layer"] == "crown":
            yy, xx = np.mgrid[0:H, 0:W]
            lx, ly = cxd + l2[0] * 0.16 * rxd, byd + l2[1] * 0.16 * ryd
            lit_core = ((xx - lx) / (0.97 * rxd)) ** 2 + ((yy - ly) / (0.97 * ryd)) ** 2 <= 1.0
            if fill(it["vis"] & ~lit_core, shadow, 0.72, fe * 2):
                count["shadow"] += 1
            continue
        if k["layer"] == "back":
            continue
        P = it["path"]
        wd = it["width"]
        n = _normals(P)
        facing = float(np.mean((n * l2).sum(1)))
        sd = -1.0 if facing > 0 else 1.0          # the side whose outward normal points away from the light
        if abs(facing) < 0.12:
            sd = 1.0 if np.mean(P[:, 0]) > cxd else -1.0
        t = _arc_t(P)
        prof = np.sin(np.pi * np.clip((t - 0.05) / 0.92, 0.0, 1.0)) ** 0.7
        bw = np.minimum(0.36 * wd, 0.06 * hw) * prof
        outer = P + n * (sd * wd / 2.0)[:, None]
        inner = outer - n * (sd * bw)[:, None]
        if fill(_poly_mask(H, W, np.vstack([outer, inner[::-1]])) & it["vis"], shadow, 0.72, crisp):
            count["shadow"] += 1
    # 3. the shine ring: an arc across the hair at a third of the head height, parallel to the top of the head; in
    #    each lock it crosses, one piece across the lock's full width that follows the lock and runs out to a point
    #    above and below the lock's middle
    phi = np.linspace(math.radians(8), math.radians(172), 400)
    y_ring = hy0 + hh / 3.0
    shift = l2[0] * 0.08 * hw                      # the ring slides a little toward the light
    arc = np.stack([cxd + shift + 0.98 * rxd * np.cos(phi), y_ring + 0.42 * ryd * (1 - np.sin(phi))], 1)
    part_pt = np.array(dc.u2px(p, [rec["part"]])[0])
    tooth = 0.045 * hw                             # one tooth of the ring per strand-width of a lock

    def half_band(x) -> float:
        u = min(1.0, abs(float(x) - (cxd + shift)) / max(rxd, 1.0))
        return 0.012 * hh * (1.0 - 0.5 * u * u)

    def lit_at(x) -> float:
        return float(np.clip(0.5 + 0.5 * ((float(x) - cxd) / max(rxd, 1.0)) * l2[0], 0, 1))

    for it in items:
        k = it["k"]
        m = it["vis"] & clip
        if k["layer"] == "tail" or not m.any():
            continue
        if k["layer"] == "crown":
            # no centre paths on the crown: pieces at a lock's spacing along the ring, pointing away from the part
            on = np.array([q for q in arc if 0 <= q[1] < H and 0 <= q[0] < W and m[int(q[1]), int(q[0])]])
            if len(on) < 4:
                continue
            gap = 0.09 * hw
            s_on = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(on, axis=0).T))])
            for s_c in np.arange(gap / 2, s_on[-1], gap):
                c0 = on[int(np.argmin(np.abs(s_on - s_c)))]
                v = c0 - part_pt
                v = v / (np.linalg.norm(v) + 1e-9)
                if v[1] < 0:
                    v = -v
                axis = np.stack([c0 - v * 0.15 * hh, c0 + v * 0.15 * hh])
                axis = dc.resample(axis, 1.5)
                piece = _ring_piece(axis, np.full(len(axis), gap), len(axis) // 2, half_band(c0[0]), tooth=tooth,
                                    rng=rng)
                if piece is not None and fill(_poly_mask(H, W, piece) & m, hi, 0.5 + 0.3 * lit_at(c0[0]), crisp):
                    count["shine"] += 1
            continue
        P = it["path"]
        wd = it["width"]
        dd = np.hypot(P[:, None, 0] - arc[None, :, 0], P[:, None, 1] - arc[None, :, 1])
        i, j = np.unravel_index(int(np.argmin(dd)), dd.shape)
        if dd[i, j] > 0.5 * max(2.0, float(wd[i])) + 2 or float(wd[i]) < 3:
            continue
        piece = _ring_piece(P, wd, int(i), half_band(P[i, 0]), tooth=tooth, rng=rng)
        if piece is not None and fill(_poly_mask(H, W, piece) & m, hi, 0.5 + 0.3 * lit_at(P[i, 0]), crisp):
            count["shine"] += 1
    # 4. a few strands along the centre paths
    sw = max(1.0, 0.42 * base_width(p))
    for it in items:
        k = it["k"]
        if k["layer"] == "crown":
            continue
        P = it["path"]
        wd = it["width"]
        if dc.path_length(P) < 0.06 * S:
            continue
        n = _normals(P)
        t = _arc_t(P)
        m = it["vis"] & clip
        for s_i in range(1 if k["layer"] in ("stray", "fringe") else strands):
            off = (0.22 if s_i == 0 else -0.25) + rng.uniform(-0.06, 0.06)
            ta, tb = rng.uniform(0.1, 0.3), rng.uniform(0.62, 0.88)
            sel = (t >= ta) & (t <= tb)
            Q = (P + n * (off * wd)[:, None])[sel]
            if len(Q) < 4:
                continue
            for a, b in visible_runs(Q, ~m):
                seg = Q[a:b]
                L = dc.path_length(seg)
                if L < max(0.04 * S, 6 * sw):
                    continue
                ops.append(Stroke(pts=seg.astype(np.float32), width=dc.taper_widths(seg, sw, "both", 0.2),
                                  color=tuple(int(v) for v in dark_line), opacity=0.7, brush="liner",
                                  seed=seed + len(ops)))
                count["strands"] += 1
    return ops, count


def _ring_piece(P: np.ndarray, wd: np.ndarray, i: int, b: float, spike: float = 4.0, tooth: float = 0.0,
                rng: Optional[np.random.Generator] = None) -> Optional[np.ndarray]:
    """The ring's piece in one lock: across the lock's full width a band ``b`` px either side of path point ``i``,
    its upper and lower edge cut into teeth that run ``spike`` * b further along the lock (one tooth per ``tooth``
    px of width, at least one), bent with the path."""
    P = np.asarray(P, float)
    if len(P) < 3 or b <= 0.5:
        return None
    rng = rng or np.random.default_rng(0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    n_all = _normals(P)
    wl = float(np.asarray(wd, float)[i])
    k = max(1, int(round(wl / tooth))) if tooth > 0 else 1
    v = np.linspace(-1.0, 1.0, 12 * k + 1)
    saw = 1.0 - np.abs(2.0 * ((v + 1.0) / 2.0 * k % 1.0) - 1.0)          # 0 between teeth, 1 at a tooth's point
    saw = saw ** 1.6
    reach_up = b * (1.0 + spike * saw * rng.uniform(0.75, 1.0))
    reach_dn = b * (1.0 + spike * saw * rng.uniform(0.75, 1.0))

    def at(off, vv):
        q = float(np.clip(s[i] + off, 0.0, s[-1]))
        j = int(np.clip(np.searchsorted(s, q), 0, len(P) - 1))
        return P[j] + n_all[j] * (0.52 * float(np.asarray(wd, float)[j]) * vv)

    top = [at(-ru, vv) for ru, vv in zip(reach_up, v)]
    bot = [at(rd, vv) for rd, vv in zip(reach_dn, v)]
    return np.vstack([np.array(top), np.array(bot)[::-1]])


def cmd_shine(ctx, args: List[str]):
    from . import cli, cli_draw
    from .cli import Spec, parse
    p = ctx.project
    d = dc.dstate(p)
    usage = "easel locks shine hair"
    a = parse(args, Spec(usage, ["region"], {"strands": "value", "color": "value"},
                         alias={"colour": "color", "strand": "strands"}, min_pos=1))
    part = _part_name(ctx, a["region"], usage)
    rec = d.get("locks", {}).get(part)
    if not rec:
        raise UsageError(f"{part} has no locks yet: build them first, then shine them",
                         f"easel locks {part} --cut long")
    strands = int(round(cli_draw._num(a, "strands", 2, 0, 3, ctx)))
    names = p.region_names()
    items, union = _shine_masks(p, rec)
    la = p.layer("line", create=False)
    on_line = la[..., 3] > 0.35 if la is not None else np.zeros((p.H, p.W), bool)
    lost = None
    if part in names:
        hair_r = p.region_mask(part).copy()
        region = cli.Region(part, hair_r, coords.mask_bbox_px(hair_r) or (0, 0, p.W, p.H), part, True)
        lost = _lost_tips(p, items, hair_r, on_line)
        clip = (hair_r | lost) & ~on_line
    else:
        clip = union & ~on_line
        ctx.notes.append(f"{part} has no flat area yet: the shine is clipped to the locks")
        region = cli.Region(part, union.copy(), coords.mask_bbox_px(union) or (0, 0, p.W, p.H), part, False)
    base, sh, li, nm = cli_draw.colour_set(p, region, a.get("color"))
    light = dc.light_vec(d)
    info: Dict[str, int] = {}

    def build():
        ops, cnt = shine_ops(p, rec, clip, base, sh, li, light, strands, seed=dc.stable_seed(p, ctx.cmdline, "shine"), lost=lost)
        info.update(cnt)
        return ops

    def post():
        cli_draw._detailed(p, part)
        return [f"{info.get('shadow', 0)} shadow strips on the sides away from the light, a shine ring of "
                f"{info.get('shine', 0)} pieces across the locks, {info.get('strands', 0)} strands along them ({nm})"
                + ("; the thin lock tips flat gave to the area beside them are hair again" if info.get("tips") else ""),
                "a stronger shine:  easel highlight " + part + "   more strands:  easel hair " + part +
                ' --flow "x,y x,y x,y"']
    return cli_draw.draw_step(ctx, label=f"locks shine {part}", layer="detail", build=build, clip=clip, post=post,
                              area_extra=region.box_px, note="lock shading, shine and strands")


COMMANDS = {"locks": cmd_locks}
