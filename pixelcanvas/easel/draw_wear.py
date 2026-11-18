"""Clothes and hands cut to the mannequin, as ordinary line records.

``easel wear <subject> <piece>`` dresses the subject's mannequin (``sketch figure``) and head construction
(``sketch head``): ``coat``, ``shirt``, ``dress``, ``robe`` (shoulders, torso sides with ease, a hem at ``--length``,
sleeves, an opening or a row of buttons), ``sleeves``, ``collar``, ``scarf`` (wrap, knot, two hanging ends), ``hood``
(up, or ``--down`` behind the neck), ``pants``, ``skirt``, ``boots``. ``easel hand <subject> <l|r>`` draws a mitten
hand from the wrist along the forearm, thumb toward the body; ``--grip x,y`` wraps the fingers round a handle there,
``--open`` spreads the fingers, ``--paw`` makes a round animal paw with toe bumps.

Every outline is a line record (L1, L2 ...) on the owned line layer: move, redo, weight or delete it by id like any
line. Silhouettes are closed shapes (weight 1.4), so ``flat auto`` fills each piece as its own areas; folds, cuffs and
seams are lighter open lines with tapered ends. Sizes come from the mannequin: head height, shoulder and hip joints,
limb joints; figures under 4.2 heads get chibi proportions; l = picture left. The first piece round the neck (or
``easel face``) draws the neck too: two line records of a part ``neck``, about half the jaw wide, that run into
shoulders sloping down from it; the clothes cut them where they cover them.

The line layer has no depth, so a piece that goes in front of earlier lines cuts them where its filled shape covers
them, inside the same undoable step (a cut line keeps its id, extra pieces get new ids). Order of the pieces, back to
front: pants, boots, skirt, shirt, dress, coat / robe, their sleeves, collar, hood, hand, scarf. A hand sits behind
its own cuff. Lines of the head (face, hair, eyes ...) stay in front of clothes, except under a hood that is up;
``--under`` names lines or parts to keep in front of the new piece, ``--over`` names lines or parts to cut anyway.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import coords
from . import draw_core as dc
from . import draw_line
from .coords import UsageError

USAGE = {
    "wear": "easel wear girl coat --length knee",
    "hand": "easel hand girl r --grip 640,520",
}

PIECES = ("coat", "shirt", "dress", "robe", "sleeves", "collar", "scarf", "hood", "pants", "skirt", "boots")
PIECE_ALIAS = {"jacket": "coat", "raincoat": "coat", "parka": "coat", "overcoat": "coat", "cloak": "robe",
               "gown": "dress", "kimono": "robe", "mantle": "robe", "sweater": "shirt", "blouse": "shirt",
               "top": "shirt", "tshirt": "shirt", "t-shirt": "shirt", "jumper": "shirt", "sleeve": "sleeves",
               "muffler": "scarf", "shawl": "scarf", "trousers": "pants", "jeans": "pants", "shorts": "pants",
               "boot": "boots", "shoes": "boots", "shoe": "boots", "skirts": "skirt", "hoods": "hood",
               "collars": "collar", "scarves": "scarf"}
# planned part names that mean the piece (the first planned part of the subject that matches names the lines)
PART_WORDS = {"coat": ("coat", "raincoat", "jacket", "parka", "overcoat", "cloak", "cape"),
              "shirt": ("shirt", "blouse", "sweater", "top", "tshirt", "jumper", "uniform", "clothes"),
              "dress": ("dress", "gown", "clothes"),
              "robe": ("robe", "cloak", "gown", "kimono", "mantle", "cape", "clothes"),
              "sleeves": ("sleeves", "sleeve"),
              "collar": ("collar",),
              "scarf": ("scarf", "muffler", "shawl"),
              "hood": ("hood",),
              "pants": ("pants", "trousers", "jeans", "shorts"),
              "skirt": ("skirt",),
              "boots": ("boots", "boot", "shoes", "shoe"),
              "hand": ("hand", "hands", "paw", "paws", "mitten", "glove", "gloves")}

GARMENTS = {
    # default length, hem flare (x hip width per thigh length below the hips), neckline dip (x neck width),
    # front opening, buttons, hem folds, cuff width (x sleeve), depth
    "coat": dict(length="knee", flare=0.12, dip=0.5, opening=True, buttons=4, folds=2, cuff=1.0, z=30.0),
    "shirt": dict(length="hip", flare=0.02, dip=0.65, opening=True, buttons=4, folds=0, cuff=0.92, z=20.0),
    "dress": dict(length="knee", flare=0.42, dip=0.9, opening=False, buttons=0, folds=3, cuff=1.0, z=22.0),
    "robe": dict(length="ankle", flare=0.24, dip=0.6, opening=True, buttons=0, folds=4, cuff=1.45, z=30.0),
}
Z = {"pants": 10.0, "boots": 12.0, "skirt": 14.0, "collar_dz": 8.0, "sleeve_dz": 5.0, "hood": 50.0,
     "hand": 55.0, "scarf": 60.0}
FITS = {"tight": 0.92, "regular": 1.0, "loose": 1.13}
LENGTHS = ("hip", "knee", "ankle", "floor")
SIL = 1.4          # silhouette weight
INNER = 0.7        # fold weight

HAIR_WORDS = ("hair", "bang", "fringe", "ponytail", "braid", "pigtail", "twintail", "lock", "ahoge")
HEAD_WORDS = ("face", "head", "eye", "brow", "mouth", "nose", "ear", "cheek", "muzzle", "snout", "beak", "lash",
              "jaw", "lip", "horn", "whisker", "iris", "pupil", "teeth", "tongue")


# ============================================================================================ small geometry

def _dist(a, b) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _unit(v, default=(0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(math.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else np.asarray(default, float)


def _perp(v) -> np.ndarray:
    """``v`` turned by +90 degrees on screen (x right, y down): right -> down."""
    return np.array([-float(v[1]), float(v[0])])


def _lerp(a, b, t):
    return a + (b - a) * t


def _ellipse(c, rx, ry, ang: float = 0.0, n: int = 16) -> np.ndarray:
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    x, y = rx * np.cos(t), ry * np.sin(t)
    ca, sa = math.cos(ang), math.sin(ang)
    return np.stack([c[0] + x * ca - y * sa, c[1] + x * sa + y * ca], 1)


def _capsule(A, B, ra: float, rb: float, n: int = 12) -> np.ndarray:
    """Closed outline of a capsule from A (radius ra) to B (radius rb)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = _unit(B - A, (1.0, 0.0))
    ang = math.atan2(d[1], d[0])
    cb = [B + rb * np.array([math.cos(a), math.sin(a)]) for a in np.linspace(ang - math.pi / 2, ang + math.pi / 2, n)]
    ca = [A + ra * np.array([math.cos(a), math.sin(a)])
          for a in np.linspace(ang + math.pi / 2, ang + 3 * math.pi / 2, n)]
    return np.array(cb + ca)


def _superellipse(c, ax, ay, A: float, B: float, pw: float = 3.2, n: int = 44) -> np.ndarray:
    """A rounded rectangle centred on c with half sizes A along ``ax`` and B along ``ay``."""
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    x = A * np.sign(ct) * np.abs(ct) ** (2.0 / pw)
    y = B * np.sign(st) * np.abs(st) ** (2.0 / pw)
    return np.asarray(c, float) + np.outer(x, ax) + np.outer(y, ay)


def _line_cross(p0, d0, p1, d1) -> Optional[np.ndarray]:
    """Where the line p0 + s d0 meets the line p1 + t d1 (None when parallel)."""
    den = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(den) < 1e-9:
        return None
    w = np.asarray(p1, float) - np.asarray(p0, float)
    s = (w[0] * d1[1] - w[1] * d1[0]) / den
    return np.asarray(p0, float) + s * np.asarray(d0, float)


def _shorten(P: np.ndarray, max_len: float) -> np.ndarray:
    """The polyline cut to ``max_len`` px from its start."""
    P = np.asarray(P, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= max_len or len(P) < 2:
        return P
    k = int(np.searchsorted(cum, max_len))
    u = (max_len - cum[k - 1]) / max(seg[k - 1], 1e-9)
    return np.vstack([P[:k], P[k - 1] + (P[k] - P[k - 1]) * u])


def _start_at_corner(P: np.ndarray) -> np.ndarray:
    """A closed loop's points rolled so the sharpest corner comes first (``draw_line.fit`` starts and ends a closed
    path with corners there)."""
    P = np.asarray(P, float)
    if len(P) < 4:
        return P
    v1 = P - np.roll(P, 1, 0)
    v2 = np.roll(P, -1, 0) - P
    cos = (v1 * v2).sum(1) / (np.hypot(*v1.T) * np.hypot(*v2.T) + 1e-9)
    i = int(np.argmin(cos))
    if cos[i] < math.cos(math.radians(draw_line.SHARP_DEG)):
        return np.roll(P, -i, 0)
    return P


# ============================================================================================ masks and cutting

def shape_mask(W: int, H: int, poly, erode: float = 0.0, dilate: float = 0.0) -> np.ndarray:
    """Filled polygon (px) as an (H, W) bool mask, optionally shrunk or grown by a few px."""
    m = dc.poly_mask(H, W, np.asarray(poly, float))
    if erode > 0:
        m = dc.erode(m, erode)
    if dilate > 0:
        m = dc.dilate(m, dilate)
    return m


def _sample(mask: np.ndarray, P: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    q = np.round(np.asarray(P, float)).astype(np.int64)
    ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
    out = np.zeros(len(q), bool)
    out[ok] = mask[q[ok, 1], q[ok, 0]]
    return out


def _dist_map(H: int, W: int, paths) -> Optional[np.ndarray]:
    """Distance (px) from every pixel to the nearest of ``paths``."""
    m = np.full((H, W), 255, np.uint8)
    any_ = False
    for P in paths:
        P = np.asarray(P, float)
        if len(P) < 2:
            continue
        q = np.round(P * 4).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [q], False, 0, 1, cv2.LINE_8, 2)
        any_ = True
    if not any_:
        return None
    return cv2.distanceTransform(m, cv2.DIST_L2, 3)


def _runs(flags: np.ndarray, closed: bool) -> List[Tuple[int, int]]:
    """Index runs [a, b) where ``flags`` is True; on a closed path a run may wrap (b > len)."""
    n = len(flags)
    if n == 0 or not flags.any():
        return []
    if flags.all():
        return [(0, n)]
    f = flags.astype(np.int8)
    edges = np.diff(np.concatenate([[0], f, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = list(zip(starts.tolist(), ends.tolist()))
    if closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n:
        a, _ = runs[-1]
        _, b = runs[0]
        runs = runs[1:-1] + [(a, n + b)]
    return runs


def _run_len(cum: np.ndarray, a: int, b: int) -> float:
    return float(cum[b - 1] - cum[a]) if b - 1 >= a else 0.0


def outside_runs(path, mask, closed: bool = False, min_cut: float = 0.0, min_keep: float = 0.0,
                 also: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Index runs [a, b) of ``path`` (px) that stay when the parts inside ``mask`` are cut away. Inside stretches
    shorter than ``min_cut`` px are not cut, left pieces shorter than ``min_keep`` px are dropped. ``also`` marks
    extra points to cut (lines that run along the new outline). A closed path's runs may wrap (b > len); the closed
    path's repeated last point is ignored. Nothing to cut -> [(0, n)]."""
    P = np.asarray(path, float)
    n = len(P)
    if closed and n > 2 and math.hypot(*(P[0] - P[-1])) < 1e-6:
        n -= 1
        P = P[:n]
    ins = _sample(mask, P) if mask is not None else np.zeros(n, bool)
    if also is not None:
        ins = ins | np.asarray(also[:n], bool)
    if not ins.any():
        return [(0, n)]
    PP = np.vstack([P, P]) if closed else P
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(PP, axis=0).T))])
    for a, b in _runs(ins, closed):
        if _run_len(cum, a, b) < min_cut:
            ins[np.arange(a, b) % n] = False
    if not ins.any():
        return [(0, n)]
    if ins.all():
        return []
    out = []
    for a, b in _runs(~ins, closed):
        if _run_len(cum, a, b) >= min_keep:
            out.append((a, b))
    return out


def _dup_flags(P: np.ndarray, dist: Optional[np.ndarray], tol: float, min_len: float, closed: bool) -> np.ndarray:
    """Points of P that run along other lines (closer than ``tol`` px for at least ``min_len`` px)."""
    if dist is None or len(P) < 2:
        return np.zeros(len(P), bool)
    H, W = dist.shape
    q = np.round(P).astype(np.int64)
    ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
    near = np.zeros(len(P), bool)
    near[ok] = dist[q[ok, 1], q[ok, 0]] < tol
    if not near.any():
        return near
    n = len(P)
    PP = np.vstack([P, P]) if closed else P
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(PP, axis=0).T))])
    out = np.zeros(n, bool)
    for a, b in _runs(near, closed):
        if _run_len(cum, a, b) >= min_len:
            out[np.arange(a, b) % n] = True
    return out


# ============================================================================================ the mannequin

def _clean_word(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).strip().lower()).strip("-")


def _is_hair(part: str) -> bool:
    return any(w in (part or "") for w in HAIR_WORDS)


def _is_head(part: str) -> bool:
    base = re.sub(r"(?:[-_](?:l|r|left|right)|[-_]?\d+)$", "", part or "")
    return any(base == w or base.startswith(w + "-") or base.endswith("-" + w) for w in HEAD_WORDS)


class Body:
    """One subject's mannequin in pixels, with the measures clothes are cut to (a frame along the torso: x to the
    picture right, y down the torso, origin between the shoulders)."""

    def __init__(self, p, el: dict, head: Optional[dict] = None):
        self.p, self.el, self.head = p, el, head
        self.W, self.H = p.W, p.H
        self.subject = el.get("subject") or el["id"]
        self.J = {k: dc.u2px(p, [v])[0] for k, v in el["joints"].items()}
        J = self.J
        self.heads = float(el.get("heads", 7.0))
        self.t = float(np.clip((self.heads - 2.5) / 4.0, 0.0, 1.0))
        self.chibi = self.heads < 4.2
        self.bust = el.get("pose") == "bust"
        self.hh = max(2.0, _dist(J["head"], J["chin"]))
        self.Hh = 2.0 * self.hh
        self.shc = (J["lshoulder"] + J["rshoulder"]) / 2.0
        v = (J["lhip"] + J["rhip"]) / 2.0 - self.shc
        if math.hypot(*v) < 0.5 * self.hh:
            v = np.array([0.0, self.Hh])
        self.dn = _unit(v)
        self.rt = np.array([self.dn[1], -self.dn[0]])
        self.sw = max(_dist(J["lshoulder"], J["rshoulder"]), 0.3 * self.Hh)
        self.nl = max(0.15 * self.hh, float((self.shc - J["chin"]) @ self.dn))
        nw = min(_lerp(0.13, 0.16, self.t) * self.Hh, 0.22 * self.sw)
        if head is not None:
            # a neck is about half as wide as the jaw it carries (the jaw spans ~0.84 of the head box)
            hx0, _, hx1, _ = dc.box_px(p, head["box"])
            nw = min(nw, 0.22 * (hx1 - hx0))
        self.nw = {"l": nw, "r": nw}
        self.y_ns = -0.6 * self.nl           # where the neck meets the shoulders (torso frame)

    def trapezius(self, s: str, us, start: Optional[np.ndarray] = None,
                  end: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Points (px) on side s's shoulder line, from the neck base (u = 0, or ``start`` in the torso frame) to the
        shoulder seam (u = 1, or ``end`` in px): it leaves the neck steeply and flattens toward the shoulder (the
        trapezius), so the shoulders slope down from the neck instead of running level."""
        sg = self.sg(s)
        SP = self.loc(arm_frame(self, s).SP if end is None else end)
        x0, y0 = (self.nw[s], self.y_ns) if start is None else (abs(float(start[0])), float(start[1]))
        span = max(abs(float(SP[0])) - x0, 0.2 * self.hh)
        drop = max(float(SP[1]) - y0, 0.3 * span)
        return [self.B(sg * (x0 + span * u), y0 + drop * (1.0 - (1.0 - u) ** 1.7)) for u in us]

    # ---------------------------------------------------------------------------------------- frames
    def B(self, x: float, y: float) -> np.ndarray:
        return self.shc + self.rt * x + self.dn * y

    def loc(self, q) -> np.ndarray:
        q = np.asarray(q, float) - self.shc
        return np.array([q @ self.rt, q @ self.dn])

    def locs(self, P) -> np.ndarray:
        Q = np.asarray(P, float) - self.shc
        return np.stack([Q @ self.rt, Q @ self.dn], 1)

    @staticmethod
    def sg(s: str) -> float:
        return -1.0 if s == "l" else 1.0

    def below(self, q) -> bool:
        return float(q[1]) >= 0.985 * self.H

    def out(self, s: str, q) -> float:
        """Distance of q from the centre line, toward side s."""
        return float(self.loc(q)[0] * self.sg(s))

    # ---------------------------------------------------------------------------------------- measures
    def arm_r(self, ease: float = 1.0) -> float:
        upper = 0.5 * (_dist(self.J["lshoulder"], self.J["lelbow"]) + _dist(self.J["rshoulder"], self.J["relbow"]))
        cap = 0.5 * min(0.36 * self.hh, 0.42 * max(upper, 0.5 * self.Hh))
        return max(_lerp(0.095, 0.225, self.t) * self.Hh, 1.25 * cap) * ease

    def leg_r(self, ease: float = 1.0) -> float:
        thigh = 0.5 * (_dist(self.J["lhip"], self.J["lknee"]) + _dist(self.J["rhip"], self.J["rknee"]))
        cap = 0.5 * min(0.5 * self.hh, 0.42 * max(thigh, 0.5 * self.Hh))
        return max(_lerp(0.1, 0.12, self.t) * self.Hh, 1.2 * cap) * ease

    def chest(self, s: str, ease: float) -> float:
        so = max(self.out(s, self.J[s + "shoulder"]), 0.15 * self.Hh)
        return max(0.87 * so, so - 0.55 * self.arm_r()) * ease

    def hips(self, s: str, ease: float) -> float:
        ho = max(self.out(s, self.J[s + "hip"]), 0.05 * self.Hh)
        return max(1.4 * ho * ease, 0.86 * self.chest(s, ease))

    def legs_local(self):
        """(y of hips, y of knees, y of ankles, thigh, shin) in the torso frame (means of both legs)."""
        yh = np.mean([self.loc(self.J[s + "hip"])[1] for s in "lr"])
        yk = np.mean([self.loc(self.J[s + "knee"])[1] for s in "lr"])
        ya = np.mean([self.loc(self.J[s + "ankle"])[1] for s in "lr"])
        return float(yh), float(yk), float(ya), max(float(yk - yh), 0.3 * self.Hh), max(float(ya - yk), 0.3 * self.Hh)

    def leg_x(self, s: str, y: float) -> float:
        """Outward distance of leg s's axis at torso-frame height y."""
        pts = [self.loc(self.J[s + k]) for k in ("hip", "knee", "ankle")]
        for a, b in zip(pts[:-1], pts[1:]):
            if a[1] <= y <= b[1] or (b[1] <= y <= a[1]):
                u = (y - a[1]) / (b[1] - a[1]) if abs(b[1] - a[1]) > 1e-6 else 0.0
                return float(_lerp(a[0], b[0], u) * self.sg(s))
        q = pts[-1] if y > pts[-1][1] else pts[0]
        return float(q[0] * self.sg(s))

    def bottom_local(self, frac: float = 1.06) -> float:
        """Torso-frame y of a level below the picture's bottom edge."""
        return float((frac * self.H - self.shc[1]) / max(self.dn[1], 0.2))

    def head_box(self) -> Tuple[float, float, float, float, float]:
        """(x0, y0, x1, y1, yaw radians) of the head in px: the head construction, else the mannequin's head."""
        if self.head is not None:
            x0, y0, x1, y1 = dc.box_px(self.p, self.head["box"])
            return x0, y0, x1, y1, math.radians(float(self.head.get("yaw", 0.0)))
        hd, ch = self.J["head"], self.J["chin"]
        r = self.hh
        return hd[0] - 0.8 * r, hd[1] - r, hd[0] + 0.8 * r, ch[1], 0.0

    def head_mask(self) -> np.ndarray:
        x0, y0, x1, y1, yaw = self.head_box()
        w, h = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2
        ball = _ellipse((cx, y0 + 0.4 * h), 0.5 * w, 0.4 * h, n=48)
        chin = np.array([cx + math.sin(yaw) * 0.35 * 0.5 * w, y1])
        jaw = np.array([[x0 + 0.08 * w, y0 + 0.55 * h], [x0 + 0.2 * w, y0 + 0.82 * h], chin,
                        [x1 - 0.2 * w, y0 + 0.82 * h], [x1 - 0.08 * w, y0 + 0.55 * h], [cx, y0 + 0.4 * h]])
        return shape_mask(self.W, self.H, ball) | shape_mask(self.W, self.H, jaw)

    def neck_mask(self, low: float) -> np.ndarray:
        """The neck from the chin down to torso-frame height ``low``."""
        nl, nr = self.nw["l"], self.nw["r"]
        q = [self.B(-nl, -self.nl - 0.15 * self.hh), self.B(nr, -self.nl - 0.15 * self.hh), self.B(nr * 1.06, low),
             self.B(-nl * 1.06, low)]
        return shape_mask(self.W, self.H, np.array(q))


def find_figure(p, subj: str, usage: str) -> dict:
    d = dc.dstate(p)
    figs = [el for el in d["sketch"] if el["kind"] == "figure"]
    for el in figs:
        if el.get("subject") == subj or el["id"].lower() == subj:
            return el
    if figs:
        names = ", ".join(str(el.get("subject") or el["id"]) for el in figs)
        raise UsageError(f"{subj!r} has no mannequin; figures: {names}",
                         usage.replace("girl", str(figs[0].get("subject") or figs[0]["id"])))
    raise UsageError(f"{subj!r} has no mannequin yet: clothes and hands are cut to the figure's joints",
                     f"easel sketch figure {subj} --pose stand")


def _head_of(p, subj: str) -> Optional[dict]:
    d = dc.dstate(p)
    heads = [el for el in d["sketch"] if el["kind"] == "head"]
    for el in heads:
        if el.get("subject") == subj:
            return el
    loose = [el for el in heads if not el.get("subject")]
    return loose[0] if len(loose) == 1 else None


def detect_neck(p, body: Body) -> None:
    """Fit the neck width to neck lines already drawn (roughly vertical lines beside the neck), so collars, scarves
    and necklines meet them."""
    d = dc.dstate(p)
    lo, hi = -0.85 * body.nl, -0.12 * body.nl
    found: Dict[str, List[float]] = {"l": [], "r": []}
    for r in d["lines"]:
        if r.get("wear") or _is_hair(r.get("part") or ""):
            continue
        P = draw_line.record_path(p, r)
        if len(P) < 3:
            continue
        L = body.locs(P)
        sel = (L[:, 1] >= lo) & (L[:, 1] <= hi)
        if sel.sum() < 3:
            continue
        for s in "lr":
            sg = body.sg(s)
            k = sel & (L[:, 0] * sg > 0.45 * body.nw[s]) & (L[:, 0] * sg < 2.0 * body.nw[s])
            if k.sum() < 3:
                continue
            xs, ys = L[k, 0] * sg, L[k, 1]
            if np.ptp(xs) < 0.6 * body.nw[s] and np.ptp(ys) > 0.3 * (hi - lo):
                found[s].append(float(np.median(xs)))
    for s in "lr":
        if found[s]:
            body.nw[s] = min(found[s], key=lambda v: abs(v - body.nw[s]))
    return bool(found["l"] and found["r"])


# ============================================================================================ the neck

NECK_PIECES = tuple(GARMENTS) + ("collar", "scarf", "hood")


def _jaw_y(p, body: Body, subj: str, x: float) -> Optional[float]:
    """y (px) of the subject's jaw at x: the lowest point there of a drawn head outline (not a face feature), else
    the head construction's jaw, else None."""
    d = dc.dstate(p)
    x0, y0, x1, y1, _ = body.head_box()
    h = y1 - y0
    tol = max(1.5, 0.004 * dc.side(p))
    best = None
    for r in d["lines"]:
        prt = (r.get("part") or "").lower()
        src = str(r.get("src", ""))
        if r.get("wear") or not _is_head(prt) or src.startswith(("face ", "eye ")) or prt.startswith(("eye", "brow")):
            continue
        if prt in d["parts"] and d["parts"][prt].get("of") not in (None, subj):
            continue
        P = draw_line.record_path(p, r)
        sel = (np.abs(P[:, 0] - x) <= tol) & (P[:, 1] > y0 + 0.6 * h) & (P[:, 1] < y1 + 0.15 * h)
        if sel.any():
            y = float(P[sel, 1].max())
            best = y if best is None else max(best, y)
    if best is not None or body.head is None:
        return best
    from . import draw_sketch
    for poly, closed, wf in draw_sketch.head_shapes(p, body.head)[1:3]:
        P = draw_sketch.resample(np.asarray(poly, float), 1.0)
        sel = np.abs(P[:, 0] - x) <= tol
        if sel.any():
            y = float(P[sel, 1].max())
            best = y if best is None else max(best, y)
    return best


def neck_specs(p, body: Body, subj: str) -> List[Tuple[str, np.ndarray]]:
    """The two sides of the neck (px control points): from the jaw, about half the jaw's width apart, straight down,
    then out into the trapezius that slopes to the shoulders. Each side ends open on the shoulder (the arm or the
    clothes take over there)."""
    out = []
    lw = draw_line.base_width(p)
    for s in "lr":
        sg = body.sg(s)
        nw = body.nw[s]
        xj = body.B(sg * nw, 0.0)[0]
        yj = _jaw_y(p, body, subj, xj)
        if yj is None:
            yj = float(body.B(sg * nw, -body.nl - 0.04 * body.hh)[1])
        top = body.loc(np.array([xj, yj - 0.35 * lw]))
        y_top = float(top[1])
        y_base = body.y_ns
        pts = [body.B(sg * nw, y_top)]
        if y_base - y_top > 0.25 * body.hh:
            # a visible neck: upright from the jaw into the base (clothes and scarves fit this same width)
            pts += [body.B(sg * nw, _lerp(y_top, y_base, 0.5)), body.B(sg * nw, y_base)]
            start = None
        else:
            # a short neck (a big head on small shoulders): the line leaves the jaw straight into the shoulder
            start = np.array([nw, max(y_top + 0.05 * body.hh, y_base)])
        pts += body.trapezius(s, (0.2, 0.48, 0.78), start=start)
        out.append((f"neck-{s}", np.array(pts)))
    return out


def _neck_lines_of(p, subj: str) -> List[dict]:
    d = dc.dstate(p)
    out = []
    for r in d["lines"]:
        prt = (r.get("part") or "").lower()
        if r.get("wear") or not prt.startswith("neck"):
            continue
        if prt in d["parts"] and d["parts"][prt].get("of") not in (None, subj):
            continue
        out.append(r)
    return out


def add_neck(p, body: Body, subj: str) -> Tuple[list, Optional[tuple], List[str]]:
    """Draw the subject's neck once (two line records, part ``neck``) when the head sits on a mannequin and no neck
    is drawn yet, so the skin under the jaw is a neck, not a column as wide as the face. Clothes drawn after it
    cut it where they cover it. Returns (ops, touched px box, report lines)."""
    d = dc.dstate(p)
    done = d.setdefault("necks", {})
    if done.get(subj) or _neck_lines_of(p, subj) or detect_neck(p, body):
        done[subj] = True
        return [], None, []
    if body.head is None and _drawn_head(p, subj) is None:
        return [], None, []
    specs = neck_specs(p, body, subj)
    # the planned part that names the neck: its box holds the neck column (jaw to the base), so flat auto gives it
    # the skin colour and the hair knows where the neck is
    head_part = next((n for n, v in d["parts"].items() if v.get("of") == subj and _is_head(n)
                      and not n.startswith(("eye", "brow", "mouth", "nose"))), None)
    sw = (d["parts"][head_part].get("swatch") if head_part else None) or "skin"
    name = "neck" if d["parts"].get("neck", {}).get("of", subj) == subj else f"neck-{_clean_word(subj)}"
    col = np.vstack([q[:3] for _, q in specs])
    lo, hi = dc.px2u(p, [col.min(0)])[0], dc.px2u(p, [col.max(0)])[0]
    box = [round(float(np.clip(v, 0, 1000)), 1) for v in (lo[0] - 4, lo[1] - 2, hi[0] + 4, hi[1])]
    if name not in d["parts"]:
        d["parts"][name] = {"of": subj, "box": box, "focus": 2, "swatch": sw, "swatch_auto": True}
    ops, bb, ids = [], None, []
    for nm, ctrl in specs:
        path, _ = draw_line.fit(p, [list(q) for q in dc.px2u(p, ctrl)], closed=False, snap=False, join=False, bow=0.0)
        rec = draw_line.make_record(p, np.asarray(path, float), dc.px2u(p, ctrl), weight=1.0, taper="end",
                                    color="line", closed=False, src=f"neck {subj}")
        rec["part"] = name
        d["lines"].append(rec)
        ids.append(rec["id"])
        ops += draw_line.render_ops(p, rec)
        bb = coords.union_box(bb, _path_box(p, path))
    done[subj] = True
    ratio = 2 * 0.5 * (body.nw["l"] + body.nw["r"]) / max(1.0, 0.84 * (body.head_box()[2] - body.head_box()[0]))
    return ops, bb, [f"neck of {subj} (part {name}): {', '.join(ids)}; {ratio:.2f} of the jaw wide, sloping into the "
                     "shoulders; clothes cut it where they cover it"]


# ============================================================================================ pieces

@dataclass
class Item:
    """One line record to be: control points (px), closed or open, weight, taper; ``fill`` = a closed outline that
    bounds an area and hides what is behind it."""
    name: str
    ctrl: np.ndarray
    closed: bool = False
    weight: float = INNER
    taper: str = "both"
    fill: bool = False
    hole: Optional[np.ndarray] = None        # a hole in this fill (px polygon)
    is_hole: bool = False                    # this closed outline is a hole of the piece (a hood's face opening)
    within: Optional[np.ndarray] = None      # a fold stays inside this outline (px polygon): the part it creases
    behind: Optional[np.ndarray] = None      # this outline is hidden inside this polygon (px): the body in front
    path: Optional[np.ndarray] = None
    frags: list = field(default_factory=list)
    intact: bool = True


@dataclass
class Piece:
    subject: str
    kind: str
    z: float
    items: List[Item]
    side: Optional[str] = None
    hood_up: bool = False
    front: Optional[np.ndarray] = None       # mask of the body in front of this piece (neck, head)
    notes: List[str] = field(default_factory=list)
    bare: Optional[np.ndarray] = None        # where it goes in front of pieces otherwise in front of it (a fist
    #                                          over the end of its own sleeve, beyond the cuff line)

    def owns(self, w: dict) -> bool:
        """An earlier version of this piece (replaced when the piece is drawn again)."""
        if w.get("of") != self.subject:
            return False
        if self.kind == "sleeves":
            return str(w.get("item", "")).startswith("sleeve") and (self.side is None or
                                                                   str(w.get("item", "")).startswith(
                                                                       f"sleeve-{self.side}"))
        if self.kind in GARMENTS and w.get("piece") == "sleeves":
            return any(it.name.startswith("sleeve") for it in self.items)
        return w.get("piece") == self.kind and (self.kind != "hand" or w.get("side") == self.side)

    def in_front(self, w: dict) -> bool:
        """An existing piece of the same subject that stays in front of this one."""
        if w.get("of") != self.subject:
            return False
        if self.kind == "hand" and str(w.get("item", "")).startswith(f"sleeve-{self.side}"):
            return True
        return float(w.get("z", 0.0)) > self.z


# ---------------------------------------------------------------------------------------------- arms

@dataclass
class Arm:
    S: np.ndarray
    E: np.ndarray
    Wr: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    o1: np.ndarray
    o2: np.ndarray
    rs: float
    re: float
    rw: float
    SP: np.ndarray
    AM: np.ndarray
    AP: np.ndarray
    clipped: bool


def arm_frame(body: Body, s: str, ease: float = 1.0, cuff: float = 1.0) -> Arm:
    J = body.J
    S, E, Wr = J[s + "shoulder"].copy(), J[s + "elbow"].copy(), J[s + "wrist"].copy()
    sg = body.sg(s)
    if _dist(S, E) < 0.3 * body.hh:
        E = S + body.dn * 0.9 * body.Hh
    if body.below(E):
        # a bust clamps the elbow to the bottom edge (its x is not a pose): the arm hangs from the shoulder
        E = S + _unit(body.dn + body.rt * sg * 0.12) * max(1.1 * body.Hh, 1.05 * (body.H - S[1]))
    clipped = body.below(E) or _dist(E, Wr) < 0.25 * _dist(S, E) or body.below(Wr) and body.bust
    d1 = _unit(E - S)
    n = _perp(d1)
    k = float(n @ _unit(J[s + "hip"] - S))
    if abs(k) < 0.2:
        k = -float(n @ (body.rt * sg))
    sgn = -1.0 if k > 0 else 1.0
    o1 = sgn * n                          # the outer side (away from the armpit)
    if clipped:
        Wr = E + d1 * 0.45 * _dist(S, E)
        d2, o2 = d1.copy(), o1.copy()
    else:
        d2 = _unit(Wr - E)
        o2 = sgn * _perp(d2)              # the same side of the tube all along (it does not twist at the elbow)
    rs = body.arm_r(ease)
    re = 0.88 * rs
    rw = 0.8 * rs * cuff
    SP = S - o1 * 0.1 * rs - d1 * 0.24 * rs
    base = S - o1 * rs
    dd = float(d1 @ (body.rt * sg))
    sap = 1.3 * rs
    if abs(dd) > 0.05:
        v = (body.chest(s, ease) - body.out(s, base)) / dd
        if math.isfinite(v):
            sap = v
    sap = float(np.clip(sap, -1.4 * rs, 1.6 * rs))
    AP = base + d1 * sap
    AM = (SP + AP) / 2 - o1 * 0.12 * rs
    return Arm(S, E, Wr, d1, d2, o1, o2, rs, re, rw, SP, AM, AP, clipped)


def sleeve_items(body: Body, s: str, a: Arm, robe: bool = False) -> List[Item]:
    name = f"sleeve-{s}"
    S, E, Wr, d1, d2, o1, o2 = a.S, a.E, a.Wr, a.d1, a.d2, a.o1, a.o2
    rs, re, rw = a.rs, a.re, a.rw
    items: List[Item] = []
    folds: List[Item] = []
    # the shoulder cap: a rounded quarter turn from the seam over the joint down into the outer arm
    Os = S + o1 * rs + d1 * 0.42 * rs
    cap = [S + o1 * 0.88 * rs + d1 * 0.06 * rs, S + o1 * 0.55 * rs - d1 * 0.17 * rs]
    upper_out = (S + E) / 2 + o1 * (rs + re) / 2
    upper_in = (a.AP + E) / 2 + (-o1) * (re - 0.02 * rs)
    if a.clipped:
        Wo, Wi = Wr + o1 * re, Wr - o1 * re
        pts = [Wo, upper_out, Os, *cap, a.SP, a.AM, a.AP, upper_in, Wi]
        items.append(Item(name, np.array(pts), closed=True, weight=SIL, fill=True))
        return items
    Lf = max(_dist(E, Wr), 1.0)
    cl = min(0.55 * rw if not robe else 0.3 * rw, 0.4 * Lf)
    Cb = Wr - d2 * cl
    rcb = _lerp(re, rw, 1.0 - cl / Lf)
    Cuo, Cui = Cb + o2 * rcb, Cb - o2 * rcb
    fore_out = (E + Cb) / 2 + o2 * (re + rcb) / 2
    fore_in = (E + Cb) / 2 - o2 * (re + rcb) / 2
    bend = d2 - d1
    bent = float(math.hypot(*bend)) > 0.35
    outer_conv = True
    if bent:
        outer_conv = float(bend @ o1) <= 0          # the crook (inside of the bend) is on the inner side
    sgo = 1.0 if outer_conv else -1.0
    n1c, n2c = o1 * sgo, o2 * sgo                    # normals on the convex side
    arc = [E + n1c * re, E + _unit(n1c + n2c) * re * 1.03, E + n2c * re]          # upper arm side first
    if bent:
        X = _line_cross(S - n1c * rs, d1, E - n2c * re, d2)
        if X is None or _dist(X, E) > 1.6 * re:
            X = E - _unit(n1c + n2c) * re * 0.85
        crook = [X]
        folds.append(Item(f"{name} fold", np.array([X, _lerp(X, E + _unit(n1c + n2c) * re, 0.42),
                                                     _lerp(X, E + _unit(n1c + n2c) * re, 0.62)]),
                          weight=INNER, taper="both"))
        q0 = X + d2 * 0.55 * re
        folds.append(Item(f"{name} fold", np.array([q0 - n2c * 0.15 * re, q0 + n2c * 0.25 * re + d2 * 0.12 * re,
                                                     q0 + n2c * 0.48 * re + d2 * 0.1 * re]), weight=INNER * 0.9))
    else:
        oi = -o1
        dip = E + oi * 0.78 * re
        crook = [E + oi * re - d1 * 0.2 * re, dip, E + oi * re + d1 * 0.2 * re]
        folds.append(Item(f"{name} fold", np.array([dip, dip - oi * 0.3 * re - d1 * 0.12 * re,
                                                     dip - oi * 0.55 * re - d1 * 0.3 * re]), weight=INNER))
        arc = [E + o1 * re]
    outer_elbow = arc if outer_conv else crook
    inner_elbow = crook if outer_conv else arc
    Wo, Wi = Wr + o2 * rw, Wr - o2 * rw
    M1, M2 = Wr - o2 * 0.4 * rw + d2 * 0.13 * rw, Wr + o2 * 0.4 * rw + d2 * 0.13 * rw
    if bent:
        # a bent arm: the outline is the union of the upper tube, the elbow and the forearm tube, so a forearm
        # folded back over the upper arm (or a wide cuff beside it) never makes the outline cross itself
        upper = np.array([a.SP, a.AM, a.AP, upper_in, E - o1 * re, E + o1 * re, upper_out, Os, *cap])
        fore = np.array([E + o2 * re, fore_out, Cuo, Wo, M2, M1, Wi, Cui, fore_in, E - o2 * re])
        ctrl = _union_ctrl([upper, _ellipse(E, re, re, n=28), fore])
        items.append(Item(name, ctrl, closed=True, weight=SIL, fill=True))
    else:
        pts = [Wo, Cuo, fore_out, *outer_elbow[::-1], upper_out, Os, *cap, a.SP, a.AM, a.AP, upper_in,
               *inner_elbow, fore_in, Cui, Wi, M1, M2]
        items.append(Item(name, np.array(pts), closed=True, weight=SIL, fill=True))
    items.append(Item(f"{name} cuff", np.array([Cuo, Cb + d2 * 0.14 * rcb, Cui]), weight=0.95, taper="none"))
    items.extend(folds)
    return items


# ---------------------------------------------------------------------------------------------- body garments

def _drawn_head(p, subj: str) -> Optional[np.ndarray]:
    """Mask (px) of the subject's drawn head: the largest closed outline of a head or face part, else None."""
    d = dc.dstate(p)
    best, best_a = None, 0.0
    for r in d["lines"]:
        part = (r.get("part") or "").lower()
        if r.get("wear") or not r.get("closed") or not _is_head(part):
            continue
        if part in d["parts"] and d["parts"][part].get("of") not in (None, subj):
            continue
        P = draw_line.record_path(p, r)
        if len(P) < 3:
            continue
        a = abs(float(cv2.contourArea(np.asarray(P, np.float32).reshape(-1, 1, 2))))
        if a > best_a:
            best, best_a = P, a
    return shape_mask(p.W, p.H, best) if best is not None else None


def head_front(p, body: Body, subj: str) -> np.ndarray:
    """What of the head is in front of the clothes (px mask): the drawn head outline, else the construction's."""
    m = _drawn_head(p, subj)
    return m if m is not None else body.head_mask()


def _lowest_inside(body: Body, mask: np.ndarray, x: float, y0: float, y1: float, n: int = 80) -> Optional[float]:
    """Torso-frame y of the lowest point at x, between y0 and y1, that is inside the mask (None if none is)."""
    best = None
    for y in np.linspace(y0, y1, n):
        q = body.B(x, y)
        xi, yi = int(round(q[0])), int(round(q[1]))
        if 0 <= xi < body.W and 0 <= yi < body.H and mask[yi, xi]:
            best = float(y)
    return best


def garment_items(body: Body, kind: str, length: str, fit: str, opening: bool, buttons: int,
                  sleeves: Sequence[str], head: Optional[np.ndarray] = None) -> List[Item]:
    g = GARMENTS[kind]
    ease = FITS[fit]
    Hh, t = body.Hh, body.t
    arms = {s: arm_frame(body, s, ease, g["cuff"]) for s in "lr"}
    y_ns = body.y_ns
    yh, yk, ya, thigh, shin = body.legs_local()
    clipped_legs = body.bust or any(body.below(body.J[s + "hip"]) for s in "lr")
    sides = {}
    for s in "lr":
        sides[s] = dict(chest=body.chest(s, ease), hips=body.hips(s, ease), y_hip=float(body.loc(body.J[s + "hip"])[1]))
    if clipped_legs:
        y_hem = body.bottom_local(1.06)
    else:
        foot = min(body.hh, 0.75 * shin)
        y_hem = {"hip": yh + 0.3 * thigh, "knee": yk + 0.08 * shin, "ankle": ya - 0.05 * shin,
                 "floor": ya + 0.32 * foot}[length]
    y_waist = 0.62 * yh if not clipped_legs else 0.62 * max(yh, 1.2 * Hh)
    waist_k = {"tight": 0.88, "regular": 0.93, "loose": 0.99}[fit]
    leg_r = body.leg_r(ease)
    for s in "lr":
        sd = sides[s]
        below_hip = max(0.0, y_hem - sd["y_hip"])
        hem = sd["hips"] * (1.0 + g["flare"] * below_hip / max(thigh, 1.0))
        if not clipped_legs and below_hip > 0:
            hem = max(hem, body.leg_x(s, y_hem) + 1.25 * leg_r)
        sd["hem"] = hem
        sd["waist"] = _lerp(sd["chest"], sd["hips"], 0.6) * waist_k
    B = body.B
    L, R = sides["l"], sides["r"]
    nwl, nwr = body.nw["l"] * 1.04, body.nw["r"] * 1.04
    dip = g["dip"] * 0.5 * (nwl + nwr)
    sleeved = set(sleeves)

    def armhole(s):
        a = arms[s]
        am = a.AM + a.o1 * 0.35 * a.rs if s in sleeved else a.AM
        return [a.AP, am, a.SP]

    wave = kind in ("robe", "dress")
    nf = int(g["folds"])
    fold_u = [round(-0.7 + 1.4 * (i + 0.5) / nf, 3) for i in range(nf)] if nf > 1 else ([0.35] if nf else [])
    # fold positions are hem control points, so each fold starts exactly on the hem
    hem_u = sorted({0.62, 0.3, 0.0, -0.3, -0.62} | set(fold_u), reverse=True)
    hem_u = [u for u in hem_u if u > -0.8] + [-0.88]
    sag = 0.035 * (L["hem"] + R["hem"]) / 2

    def hem_y(u):
        y = y_hem + sag * (1 - u * u)
        if wave and fold_u:
            near = min(abs(u - f) for f in fold_u)
            y -= 0.035 * Hh * max(0.0, 1 - near / 0.12)
        return y

    pts = [B(-L["hem"], y_hem)]
    long_ = y_hem - yh > 0.3 * Hh
    if long_:
        pts.append(B(-_lerp(L["hips"], L["hem"], 0.45), _lerp(L["y_hip"], y_hem, 0.45)))
    if y_hem > L["y_hip"] + 0.04 * Hh:
        pts.append(B(-L["hips"], L["y_hip"]))
    if y_hem > y_waist + 0.05 * Hh:
        pts.append(B(-L["waist"], y_waist))
    pts += armhole("l")
    y_nl = y_nr = y_ns
    y_open = None
    if body.chibi and head is not None:
        # a big head sits on the collar: the neckline rises behind the head, just above its outline, so the head
        # overlaps the collar and no gap opens under the chin
        lo, hi = -body.nl - 1.5 * body.hh, y_ns + 0.6 * body.hh
        yl = _lowest_inside(body, head, -0.9 * nwl, lo, hi)
        yr = _lowest_inside(body, head, 0.9 * nwr, lo, hi)
        yc = _lowest_inside(body, head, 0.0, lo, hi)
        if yl is not None and yr is not None and yc is not None:
            y_nl, y_nr = min(y_ns, yl - 0.06 * body.hh), min(y_ns, yr - 0.06 * body.hh)
            dip = max(0.0, min(dip, yc - 0.04 * body.hh - max(y_nl, y_nr)))
            y_open = yc - 0.15 * body.hh
    Nl, Nr = B(-nwl, y_nl), B(nwr, y_nr)
    # the shoulder line slopes from the neck to the seam (the trapezius), the same curve the neck runs into
    pts += body.trapezius("l", (0.62, 0.28), start=np.array([nwl, y_nl]), end=arms["l"].SP) + [Nl]
    y_nm = max(y_nl, y_nr)
    NF = B(0, y_nm + dip)
    # a round neckline: it leaves the neck sides gently and dips at the front
    pts += [B(-0.72 * nwl, _lerp(y_nl, y_nm, 0.5) + 0.3 * dip), B(-0.38 * nwl, y_nm + 0.82 * dip), NF,
            B(0.38 * nwr, y_nm + 0.82 * dip), B(0.72 * nwr, _lerp(y_nr, y_nm, 0.5) + 0.3 * dip)]
    pts += [Nr] + body.trapezius("r", (0.28, 0.62), start=np.array([nwr, y_nr]), end=arms["r"].SP)
    pts += armhole("r")[::-1]
    if y_hem > y_waist + 0.05 * Hh:
        pts.append(B(R["waist"], y_waist))
    if y_hem > R["y_hip"] + 0.04 * Hh:
        pts.append(B(R["hips"], R["y_hip"]))
    if long_:
        pts.append(B(_lerp(R["hips"], R["hem"], 0.45), _lerp(R["y_hip"], y_hem, 0.45)))
    pts.append(B(R["hem"], y_hem))
    hem_pts = {}
    for u in hem_u:
        q = B(u * (R["hem"] if u > 0 else L["hem"]), hem_y(u))
        hem_pts[u] = q
        pts.append(q)
    items = [Item("body", np.array(pts), closed=True, weight=SIL, fill=True)]
    # the front opening (zip or edge) and a row of buttons
    if opening or buttons:
        Hc = hem_pts.get(0.0, B(0, hem_y(0.0)))
        mid = B(0, _lerp(y_ns + dip, hem_y(0.0), 0.5))
        # under a big head the opening starts well inside it, so the head cuts it cleanly at the chin
        top = B(0, y_open) if y_open is not None else NF
        items.append(Item("body opening", np.array([top, mid, Hc]), weight=1.0, taper="none"))
    if buttons:
        rb = _lerp(0.032, 0.042, t) * Hh
        y0 = y_ns + dip + 0.09 * Hh + rb
        y1 = min(_lerp(y_waist, y_hem, 0.25), y_hem - 0.08 * Hh)
        if clipped_legs:
            y1 = min(y1, body.loc((body.shc[0], 0.95 * body.H))[1])
        xb = -max(0.05 * Hh, 1.8 * rb)
        for k in range(buttons):
            yb = _lerp(y0, y1, k / max(1, buttons - 1)) if buttons > 1 else y0
            items.append(Item("body button", _ellipse(B(xb, yb), rb, rb, n=8), closed=True, weight=0.8, fill=True))
    # folds that rise from the hem
    if nf and not clipped_legs:
        span = y_hem - max(y_waist, yh - 0.1 * Hh)
        for u in fold_u:
            side = R if u > 0 else L
            x0 = u * side["hem"]
            top = hem_y(u) - (0.45 if wave else 0.3) * span
            q0 = hem_pts.get(round(u, 3), B(x0, hem_y(u)))
            items.append(Item("body fold", np.array([q0, B(x0 * 0.93, _lerp(hem_y(u), top, 0.55)),
                                                     B(x0 * 0.88, top)]), weight=INNER, taper="end"))
    for s in sleeves:
        items.extend(sleeve_items(body, s, arms[s], robe=(kind == "robe")))
    return items


# ---------------------------------------------------------------------------------------------- collar, scarf, hood

def collar_items(body: Body) -> List[Item]:
    hc = _lerp(0.13, 0.12, body.t) * body.Hh
    top_lim = -body.nl + 0.04 * body.Hh
    items = []
    for s in "lr":
        sg = body.sg(s)
        nw = body.nw[s] * 1.04
        y = body.y_ns
        CT = body.B(sg * 0.42 * nw, y + 1.35 * hc)
        CV = body.B(0.0, y + 0.62 * hc)
        CM = body.B(sg * 0.6 * nw, y + 0.12 * hc)
        CN = body.B(sg * nw, max(y - 0.55 * hc, top_lim))
        CO1 = body.B(sg * (nw + 0.7 * hc), y - 0.05 * hc)
        CO = body.B(sg * (nw + 1.2 * hc), y + 0.2 * hc)
        items.append(Item(f"collar-{s}", np.array([CT, CV, CM, body.B(sg * nw, y - 0.1 * hc), CN, CO1, CO]),
                          closed=True, weight=SIL, fill=True))
        items.append(Item(f"collar-{s} fold", np.array([body.B(sg * (nw + 0.15 * hc), y + 0.05 * hc),
                                                        body.B(sg * (nw + 0.55 * hc), y + 0.3 * hc)]),
                          weight=INNER * 0.9))
    return items


def _blob(c, rx: float, ry: float, ang: float, wob: float = 0.04, phase: float = 0.0, n: int = 16) -> np.ndarray:
    """A soft, slightly uneven oval (cloth bunched up); ``rx`` along the direction ``ang`` (radians)."""
    ca, sa = math.cos(ang), math.sin(ang)
    out = []
    for i in range(n):
        th = 2 * math.pi * i / n
        m = 1.0 + wob * math.sin(2 * th + phase) + 0.7 * wob * math.sin(3 * th + 2.1 * phase + 1.0)
        x, y = rx * math.cos(th) * m, ry * math.sin(th) * m
        out.append(np.asarray(c, float) + np.array([x * ca - y * sa, x * sa + y * ca]))
    return np.array(out)


def scarf_items(body: Body, side: str = "l", tips: str = "square") -> List[Item]:
    """A thick wrap round the neck base (its top edge crosses the neck, high beside it and a little lower in front;
    its lower edge hangs in a soft curve; its sides round off where it turns back over the shoulders), a knot of two
    overlapping lobes on one side with folds pulled into it, and two ends that come out from under the knot, widen,
    swing apart and end in a cut or a point, each with a fold that runs into the knot."""
    t, Hh = body.t, body.Hh
    nwl, nwr = body.nw["l"], body.nw["r"]
    nwm = 0.5 * (nwl + nwr)
    tb = _lerp(0.2, 0.25, t) * Hh                      # thickness of the wrap at the front
    wb = max(nwm + 0.9 * tb, _lerp(0.48, 0.37, t) * body.sw)
    wb = min(wb, 0.48 * body.sw + 0.3 * tb)
    chin = -body.nl
    y_c = max(body.y_ns - 0.12 * tb, chin + 0.07 * Hh)       # top of the wrap at its highest, beside the neck
    a_l, a_r = arm_frame(body, "l"), arm_frame(body, "r")
    slope = float(np.clip(np.mean([(body.loc(a.SP)[1] - body.y_ns) / max(abs(body.loc(a.SP)[0]) - nwm, 1.0)
                                   for a in (a_l, a_r)]), 0.0, 0.6))
    u_p = float(np.clip(1.5 * nwm / wb, 0.3, 0.55))          # where the top edge is highest (x / wb)
    sag = 0.2 * tb                    # the top edge dips in front, lower than at its sides: it goes round the neck
    fall = 0.8 * slope * wb * (1 - u_p) + 0.22 * tb

    def y_top(x, sag=sag):
        u = abs(x) / wb
        if u <= u_p:
            return y_c + sag * (1 - (u / u_p) ** 2)
        return y_c + fall * ((u - u_p) / (1 - u_p)) ** 1.6

    def y_bot(x):
        return y_c + 0.08 * tb + 0.95 * tb - 0.38 * tb * (x / wb) ** 2 + 0.8 * slope * max(0.0, abs(x) - nwm)

    def y_mid(x):
        return 0.5 * (y_top(x) + y_bot(x))

    B = body.B
    top = [B(u * wb, y_top(u * wb)) for u in (-0.93, -0.75, -0.55, -u_p, -0.5 * u_p, 0.0, 0.5 * u_p, u_p, 0.55,
                                              0.75, 0.93)]
    bottom = [B(u * wb, y_bot(u * wb)) for u in (0.9, 0.62, 0.3, 0.0, -0.3, -0.62, -0.9)]
    wrap = top + [B(1.03 * wb, y_mid(wb))] + bottom + [B(-1.03 * wb, y_mid(-wb))]
    wrap = np.array(wrap)
    # the back of the band: behind the neck it rises higher than the front, so the scarf reads as a band round
    # the neck; its top edge runs from the neck's side out and down onto the front's top edge (the neck stays in
    # front of it, the front of the band covers its lower end)
    y_back = max(y_c - 0.3 * tb, _lerp(chin, y_c, 0.55))
    neck = np.array([B(-nwl, chin - 0.2 * Hh), B(nwr, chin - 0.2 * Hh), B(nwr, y_c + tb), B(-nwl, y_c + tb)])
    items = []
    for sg in (-1.0, 1.0):
        nws = nwl if sg < 0 else nwr
        xo = 0.8 * wb
        if xo <= 1.3 * nws or y_top(nws) - y_back < 0.1 * tb:
            continue
        low = y_top(xo) + 0.4 * tb
        xi = nws - 0.5 * draw_line.base_width(body.p)          # its inner edge runs hidden behind the neck
        pts = [B(sg * xi, y_back), B(sg * _lerp(nws, xo, 0.45), _lerp(y_back, y_top(xo), 0.3)),
               B(sg * xo, y_top(xo) + 0.2 * tb), B(sg * xo, low), B(sg * _lerp(nws, xo, 0.5), low),
               B(sg * xi, low)]
        # one item per side: each is an area of its own (pieces are told apart by their item names)
        items.append(Item(f"scarf wrap back-{'l' if sg < 0 else 'r'}", np.array(pts[::-1] if sg > 0 else pts),
                          closed=True, weight=SIL, fill=True, behind=neck))
    # one point of a closed outline repeated would make a corner; the fit keeps this outline smooth
    items.append(Item("scarf wrap", wrap, closed=True, weight=SIL, fill=True))
    sk = -1.0 if side == "l" else 1.0
    for sg in (-1.0, 1.0):
        # where the wrap turns back over the shoulder: a curve from its top edge round to its lower edge
        x0, x1, x2 = sg * 0.72 * wb, sg * 0.84 * wb, sg * 0.88 * wb
        items.append(Item("scarf wrap turn", np.array([B(x0, y_top(x0)), B(x1, y_mid(x1) + 0.05 * tb),
                                                       B(x2, y_bot(x2))]), weight=0.95, taper="none",
                          within=wrap))
    xk = sk * 0.4 * wb
    yk = 0.5 * (y_top(xk, 0.08 * tb) + y_bot(xk)) + 0.14 * tb      # on the band's middle (before its dip)
    K = B(xk, yk)

    def inside(x, f):
        """a point across the band at x: f = 0 on its top edge, 1 on its lower edge"""
        return B(x, _lerp(y_top(x), y_bot(x), f))

    # folds radiate from the knot, where the cloth is pulled tight: three fan out across the band toward the far
    # side (they start under the knot), one runs to the near side; a soft crease further along the band
    # (their free ends stay clear of the band's edges: a tip that nearly touches an edge pinches off a sliver)
    for f0, f1, reach, w in ((0.38, 0.24, 0.4, 0.9), (0.52, 0.5, 0.52, 0.95), (0.64, 0.78, 0.36, 0.85)):
        xa = xk - sk * 0.3 * tb
        xb = xa - sk * reach * wb
        items.append(Item("scarf wrap fold", np.array([inside(xa, f0), inside(_lerp(xa, xb, 0.5), _lerp(f0, f1, 0.65)),
                                                       inside(xb, f1)]), weight=INNER * w / 0.7, taper="end",
                          within=wrap))
    xa = xk + sk * 0.3 * tb
    xo = xa + sk * 0.24 * wb
    items.append(Item("scarf wrap fold", np.array([inside(xa, 0.5), inside(_lerp(xa, xo, 0.55), 0.34),
                                                   inside(xo, 0.24)]), weight=INNER * 1.1, taper="end",
                      within=wrap))
    xf = -sk * 0.62 * wb
    items.append(Item("scarf wrap fold", np.array([inside(xf - sk * 0.1 * wb, 0.7), inside(xf, 0.5),
                                                   inside(xf + sk * 0.1 * wb, 0.42)]), weight=INNER, within=wrap))
    up_ang = math.atan2(body.dn[1], body.dn[0]) - math.pi / 2
    Lt = _lerp(0.55, 0.95, t) * Hh
    KL = K + body.dn * 0.2 * tb                       # the hanging lobe of the knot
    for which, length, dx, a1, a2 in (("back", 0.86, 0.18, 0.2, 0.14), ("front", 1.0, -0.12, -0.07, 0.06)):
        top0 = KL + body.rt * (sk * dx * tb) + body.dn * 0.04 * tb       # under the knot
        Lw = Lt * length

        def axis(u, top0=top0, Lw=Lw, a1=a1, a2=a2):
            return top0 + body.dn * (Lw * u) + body.rt * (sk * (a1 * Lw * u + a2 * Lw * u * u))

        def tang(u, Lw=Lw, a1=a1, a2=a2):
            return _unit(body.dn * Lw + body.rt * (sk * (a1 * Lw + 2 * a2 * Lw * u)))

        def wd(u):
            return _lerp(0.24, 0.52, u ** 0.6) * tb

        def at(u, s):
            """a point across the end: s = -1 on its left edge, +1 on its right edge"""
            return axis(u) + _perp(tang(u)) * wd(u) * s

        if tips == "pointed":
            tip = axis(1.0) + tang(1.0) * 0.95 * wd(1.0)
            pts = [tip] + [at(u, -1) for u in (0.88, 0.6, 0.3, 0.0)] + [at(u, 1) for u in (0.0, 0.3, 0.6, 0.88)]
        else:
            slant = tang(1.0) * 0.24 * wd(1.0) * sk
            pts = [at(1.0, -1) - slant] + [at(u, -1) for u in (0.6, 0.3, 0.0)] + \
                  [at(u, 1) for u in (0.0, 0.3, 0.6)] + [at(1.0, 1) + slant]
        end = np.array(pts)
        items.append(Item(f"scarf end-{which}", end, closed=True, weight=SIL, fill=True))
        # a fold from under the knot that fans out to one edge (the two ends' folds open in a V), and a short
        # fold that rises from the cut end
        s_out = 1.0 if float(np.dot(_perp(tang(0.0)), body.rt)) * sk > 0 else -1.0
        s_f = s_out if which == "back" else -s_out
        items.append(Item(f"scarf end-{which} fold",
                          np.array([at(0.0, 0.0), at(0.16, 0.25 * s_f), at(0.3, 0.7 * s_f), at(0.42, s_f)]),
                          weight=0.85, taper="none", within=end))
        items.append(Item(f"scarf end-{which} fold", np.array([at(0.0, -0.15 * s_f), at(0.14, -0.4 * s_f),
                                                               at(0.28, -0.55 * s_f)]), weight=0.8, taper="end",
                          within=end))
        items.append(Item(f"scarf end-{which} fold",
                          np.array([at(1.0, -0.3 * s_f), at(0.92, -0.36 * s_f), at(0.82, -0.38 * s_f)]),
                          weight=INNER, taper="end", within=end))
    # the knot: one compact lump over the band; the ends come out from under it. Its creases start on its outline
    # on the side toward the neck, where the cloth is pulled in, and twist across its face on a slant, all inside
    # it: no line crosses the knot's outline, so band, ends and knot read back to front
    KM = K + body.dn * 0.06 * tb
    a_up = up_ang - sk * 0.18
    ax = np.array([math.cos(a_up), math.sin(a_up)])                  # along the lump, upward
    ay = _perp(ax)
    if float(ay @ body.rt) * sk < 0:
        ay = -ay                                                     # across it, toward the outside
    A, Bk, pw = 0.46 * tb, 0.58 * tb, 2.5
    lump = _superellipse(KM, ax, ay, A, Bk, pw=pw, n=40)
    items.append(Item("scarf knot", lump, closed=True, weight=SIL, fill=True))

    def on_knot(a, b, edge=False):
        if edge:                                                     # pushed out onto the outline
            r = (abs(a) ** pw + abs(b) ** pw) ** (1.0 / pw)
            a, b = 1.04 * a / r, 1.04 * b / r
        return KM + ax * a * A + ay * b * Bk
    # they fan out from where the band is pulled into the knot (its upper side toward the neck), curving round
    for c, w in ((((0.55, -0.85), (0.12, -0.3), (-0.12, 0.3)), 1.05),
                 (((0.4, -0.95), (-0.12, -0.62), (-0.5, -0.3)), 1.0),
                 (((0.72, -0.7), (0.5, 0.0), (0.42, 0.42)), 0.9)):
        pts = [on_knot(*c[0], edge=True)] + [on_knot(*q) for q in c[1:]]
        items.append(Item("scarf knot crease", np.array(pts), weight=INNER * w, taper="end", within=lump))
    return items


def hood_items(body: Body, down: bool) -> Tuple[List[Item], Optional[np.ndarray]]:
    """The hood up around the head (outline, face opening, rim) or down as a roll behind the neck. Returns the items
    and, for a hood that is down, the mask of the body in front of it."""
    B = body.B
    if down:
        rth = _lerp(0.2, 0.17, body.t) * body.Hh
        nwm = 0.5 * (body.nw["l"] + body.nw["r"])
        xo = nwm + 1.6 * rth
        y = body.y_ns
        top = max(y - 0.85 * rth, -body.nl - 0.05 * body.Hh)
        pts = [B(-xo, y + 0.35 * rth), B(-0.92 * xo, y - 0.25 * rth), B(-0.6 * xo, top + 0.18 * rth), B(0, top),
               B(0.6 * xo, top + 0.18 * rth), B(0.92 * xo, y - 0.25 * rth), B(xo, y + 0.35 * rth),
               B(0.7 * xo, y + 0.75 * rth), B(0, y + 0.9 * rth), B(-0.7 * xo, y + 0.75 * rth)]
        items = [Item("hood roll", np.array(pts), closed=True, weight=SIL, fill=True)]
        # the roll's fold runs between two points of its outline, so it never reads as an open gap
        mid = [pts[1], B(-0.55 * xo, top + 0.5 * rth), B(0, top + 0.42 * rth), B(0.55 * xo, top + 0.5 * rth), pts[5]]
        items.append(Item("hood roll fold", np.array(mid), weight=0.8, taper="none"))
        front = body.neck_mask(y + 1.2 * rth) | body.head_mask()
        return items, front
    x0, y0, x1, y1, yaw = body.head_box()
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    sy = math.sin(yaw)
    ml = 0.13 * w + 0.12 * w * max(0.0, sy)
    mr = 0.13 * w + 0.12 * w * max(0.0, -sy)
    fcx = cx + sy * 0.35 * 0.5 * w
    Nl, Nr = B(-body.nw["l"], body.y_ns), B(body.nw["r"], body.y_ns)
    a_l, a_r = arm_frame(body, "l"), arm_frame(body, "r")
    HBl, HBr = _lerp(Nl, a_l.SP, 0.5) + body.dn * 0.06 * body.hh, _lerp(Nr, a_r.SP, 0.5) + body.dn * 0.06 * body.hh
    yf = max(B(0, body.y_ns + 0.5 * body.nw["l"])[1], y1 + 0.12 * h)
    outer = [HBl, np.array([x0 - 0.45 * ml, y1 - 0.02 * h]), np.array([x0 - ml, y0 + 0.45 * h]),
             np.array([x0 - 0.35 * ml + 0.05 * w, y0 + 0.05 * h]), np.array([cx - sy * 0.12 * w, y0 - 0.14 * h]),
             np.array([x1 + 0.35 * mr - 0.05 * w, y0 + 0.05 * h]), np.array([x1 + mr, y0 + 0.45 * h]),
             np.array([x1 + 0.45 * mr, y1 - 0.02 * h]), HBr, np.array([_lerp(fcx, HBr[0], 0.55), yf - 0.02 * h]),
             np.array([fcx, yf]), np.array([_lerp(fcx, HBl[0], 0.55), yf - 0.02 * h])]
    sh = sy * 0.1 * w
    opening = [np.array([cx + sh, y0 + 0.19 * h]), np.array([x1 - 0.1 * w + sh, y0 + 0.27 * h]),
               np.array([x1 + 0.0 * w + sh, y0 + 0.55 * h]), np.array([x1 - 0.13 * w + sh, y1 - 0.1 * h]),
               np.array([fcx, y1 + 0.07 * h]), np.array([x0 + 0.13 * w + sh, y1 - 0.1 * h]),
               np.array([x0 - 0.0 * w + sh, y0 + 0.55 * h]), np.array([x0 + 0.1 * w + sh, y0 + 0.27 * h])]
    op = np.array(opening)
    c = op.mean(0)
    # the rim's thickness: a line round the opening, its ends on the opening's lower corners
    rim = [opening[5]]
    for q in opening[6:] + opening[:3]:
        rim.append(q + _unit(q - c) * 0.075 * w)
    rim.append(opening[3])
    items = [Item("hood", np.array(outer), closed=True, weight=SIL, fill=True, hole=op),
             Item("hood opening", op, closed=True, weight=SIL * 0.95, is_hole=True),
             Item("hood rim", np.array(rim), weight=0.85, taper="none")]
    return items, None


# ---------------------------------------------------------------------------------------------- legs

def pants_items(body: Body, length: str, fit: str) -> List[Item]:
    ease = FITS[fit]
    yh, yk, ya, thigh, shin = body.legs_local()
    r0 = body.leg_r(ease)
    waist = {s: body.hips(s, ease) * 0.92 for s in "lr"}
    y_w = yh - 0.42 * thigh
    y_cr = yh + 0.28 * thigh
    hem_y = {"hip": yh + 0.45 * thigh, "knee": yk + 0.05 * shin, "ankle": ya - 0.04 * shin,
             "floor": ya + 0.15 * shin}[length]
    items = []
    out_l, in_l, out_r, in_r = [], [], [], []
    hems = {}
    for s in "lr":
        sg = body.sg(s)
        Hp, Kn, An = (body.loc(body.J[s + k]) for k in ("hip", "knee", "ankle"))

        def at(y, Hp=Hp, Kn=Kn, An=An):
            if y <= Kn[1]:
                u = (y - Hp[1]) / max(Kn[1] - Hp[1], 1e-6)
                return _lerp(Hp, Kn, float(np.clip(u, 0, 1)))
            u = (y - Kn[1]) / max(An[1] - Kn[1], 1e-6)
            return _lerp(Kn, An, float(np.clip(u, 0, 1.4)))

        def rad(y):
            return r0 * (1.0 if y <= yk else _lerp(0.86, 0.8, float(np.clip((y - yk) / max(shin, 1), 0, 1))))

        ys = [y for y in (yh, _lerp(yh, yk, 0.55), yk, _lerp(yk, ya, 0.5), hem_y) if y <= hem_y + 1e-6]
        if ys[-1] < hem_y - 1e-6:
            ys.append(hem_y)
        o = [body.B(*(at(y) + np.array([sg * rad(y), 0.0]))) for y in ys]
        i = [body.B(*(at(y) - np.array([sg * rad(y) * 0.95, 0.0]))) for y in ys if y > y_cr]
        hems[s] = (o[-1], i[-1] if i else body.B(0, hem_y))
        if s == "l":
            out_l, in_l = o, i
        else:
            out_r, in_r = o, i
    WL, WR = body.B(-waist["l"], y_w), body.B(waist["r"], y_w)
    CR = body.B(0.0, y_cr)
    pts = [WL, body.B(-0.5 * waist["l"], y_w + 0.02 * body.Hh), body.B(0.5 * waist["r"], y_w + 0.02 * body.Hh), WR]
    pts += out_r + [hems["r"][1]] + in_r[::-1][1:] + [CR] + in_l[:-1] + [hems["l"][1]] + out_l[::-1]
    items.append(Item("pants", np.array(pts), closed=True, weight=SIL, fill=True))
    for s in "lr":
        sg = body.sg(s)
        Kn = body.loc(body.J[s + "knee"])
        if Kn[1] < hem_y - 0.1 * body.Hh:
            q = body.B(Kn[0] + sg * 0.1 * r0, Kn[1])
            items.append(Item("pants fold", np.array([q - body.rt * sg * 0.45 * r0 - body.dn * 0.15 * r0, q,
                                                      q + body.rt * sg * 0.35 * r0 + body.dn * 0.2 * r0]),
                              weight=INNER))
    return items


def skirt_items(body: Body, length: str, fit: str) -> List[Item]:
    ease = FITS[fit]
    yh, yk, ya, thigh, shin = body.legs_local()
    y_w = yh - 0.42 * thigh
    hem_y = {"hip": yh + 0.35 * thigh, "knee": yk + 0.02 * shin, "ankle": ya - 0.05 * shin,
             "floor": ya + 0.2 * shin}[length]
    w = {s: body.hips(s, ease) * 0.9 for s in "lr"}
    hem = {s: max(body.hips(s, ease) * (1.0 + 0.5 * (hem_y - yh) / max(thigh, 1)),
                  body.leg_x(s, hem_y) + 1.2 * body.leg_r(ease)) for s in "lr"}
    B = body.B
    nf = 4
    fold_u = [-0.7 + 1.4 * (i + 0.5) / nf for i in range(nf)]
    hem_u = sorted(set([0.6, 0.3, 0.0, -0.3, -0.6] + [round(u, 3) for u in fold_u]), reverse=True)

    def hy(u):
        near = min(abs(u - f) for f in fold_u)
        return hem_y + 0.03 * (hem["l"] + hem["r"]) * (1 - u * u) - 0.03 * body.Hh * max(0.0, 1 - near / 0.12)

    pts = [B(-hem["l"], hem_y), B(-_lerp(w["l"], hem["l"], 0.45), _lerp(y_w, hem_y, 0.45)), B(-w["l"], y_w),
           B(0, y_w + 0.015 * body.Hh), B(w["r"], y_w), B(_lerp(w["r"], hem["r"], 0.45), _lerp(y_w, hem_y, 0.45)),
           B(hem["r"], hem_y)]
    hp = {}
    for u in hem_u:
        hp[u] = B(u * (hem["r"] if u > 0 else hem["l"]), hy(u))
        pts.append(hp[u])
    items = [Item("skirt", np.array(pts), closed=True, weight=SIL, fill=True)]
    for u in fold_u:
        q0 = hp[round(u, 3)]
        top = _lerp(hem_y, y_w, 0.6)
        x = u * 0.7 * (w["r"] if u > 0 else w["l"])
        items.append(Item("skirt fold", np.array([q0, B(_lerp(body.loc(q0)[0], x, 0.5), _lerp(hy(u), top, 0.5)),
                                                  B(x, top)]), weight=INNER, taper="end"))
    return items


def boots_items(body: Body, fit: str) -> List[Item]:
    ease = FITS[fit]
    items = []
    r0 = body.leg_r(ease) * 0.92
    for s in "lr":
        sg = body.sg(s)
        Kn, An = body.J[s + "knee"], body.J[s + "ankle"]
        if body.below(An):
            continue
        d = _unit(An - Kn)
        n = _perp(d)
        if n @ (body.rt * sg) < 0:
            n = -n                     # outward
        shin = _dist(Kn, An)
        fs = min(body.hh, 0.75 * shin)
        top = _lerp(Kn, An, 0.38)
        rt_, ra = r0 * 0.95, r0 * 0.8
        toe_dir = _unit(body.rt * sg * 0.75 + body.dn * 0.35)
        toe = An + toe_dir * 0.95 * fs + body.dn * 0.12 * fs
        sole = An + body.dn * 0.42 * fs
        heel = An - n * ra * 1.05 + body.dn * 0.38 * fs
        pts = [top + n * rt_, _lerp(top, An, 0.5) + n * _lerp(rt_, ra, 0.5), An + n * ra,
               An + n * ra * 1.1 + toe_dir * 0.35 * fs, toe - body.dn * 0.18 * fs, toe + body.dn * 0.12 * fs,
               _lerp(toe, sole, 0.5) + body.dn * 0.1 * fs, sole, heel, An - n * ra * 1.02,
               _lerp(top, An, 0.5) - n * _lerp(rt_, ra, 0.5), top - n * rt_]
        items.append(Item(f"boot-{s}", np.array(pts), closed=True, weight=SIL, fill=True))
        cb = _lerp(top, An, 0.16)
        items.append(Item(f"boot-{s} cuff", np.array([cb + n * rt_ * 0.99, cb + d * 0.08 * rt_, cb - n * rt_ * 0.99]),
                          weight=0.95, taper="none"))
    return items


# ---------------------------------------------------------------------------------------------- hands

def _union_ctrl(polys: Sequence[np.ndarray], scale: int = 4, within: Optional[np.ndarray] = None) -> np.ndarray:
    """Control points (px) of the outer outline of a union of polygons (cut to the polygon ``within`` when given),
    smoothed and thinned."""
    from scipy.ndimage import gaussian_filter1d
    allp = np.vstack(polys)
    x0, y0 = np.floor(allp.min(0)) - 6
    x1, y1 = np.ceil(allp.max(0)) + 6
    Wl, Hl = int((x1 - x0) * scale) + 1, int((y1 - y0) * scale) + 1
    m = np.zeros((Hl, Wl), np.uint8)
    for poly in polys:
        q = np.round((np.asarray(poly, float) - [x0, y0]) * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [q], 1)
    if within is not None:
        mw = np.zeros_like(m)
        q = np.round((np.asarray(within, float) - [x0, y0]) * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mw, [q], 1)
        m &= mw
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.asarray(polys[0], float)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) / scale + [x0, y0]
    sig = 0.9 * scale
    c = np.stack([gaussian_filter1d(c[:, 0], sig, mode="wrap"), gaussian_filter1d(c[:, 1], sig, mode="wrap")], 1)
    loop = dc.resample(np.vstack([c, c[:1]]), 1.0)
    pts = dc.rdp(loop, 0.45)[:-1]
    if len(pts) < 6:
        pts = loop[:: max(1, len(loop) // 12)]
    return _start_at_corner(pts)


@dataclass
class HandFrame:
    W0: np.ndarray        # base of the hand (just inside the cuff)
    f: np.ndarray         # along the forearm
    g: np.ndarray         # toward the thumb
    L: float              # hand length
    w: float              # palm width
    r0: float             # half width of the wrist
    E: np.ndarray


def hand_frame(body: Body, s: str, cuff_r: Optional[float]) -> HandFrame:
    E, Wr = body.J[s + "elbow"], body.J[s + "wrist"]
    f = _unit(Wr - E)
    g = _perp(f)
    toward = _unit(body.shc - Wr)
    if abs(float(g @ toward)) < 0.15:
        toward = body.rt * -body.sg(s)
    if g @ toward < 0:
        g = -g
    fore = max(_dist(E, Wr), 0.2 * body.Hh)
    Lh = min(_lerp(0.26, 0.66, body.t) * body.Hh, 1.05 * fore)
    Lh = max(Lh, 0.16 * body.Hh)
    wp = _lerp(0.66, 0.5, body.t) * Lh
    rw = cuff_r if cuff_r else 0.8 * body.arm_r()
    r0 = min(0.8 * rw, 0.4 * wp)
    W0 = Wr - f * 0.12 * rw
    return HandFrame(W0, f, g, Lh, wp, r0, E)


def mitten_items(hf: HandFrame, s: str, open_: bool, paw: bool) -> List[Item]:
    W0, f, g, L, w, r0 = hf.W0, hf.f, hf.g, hf.L, hf.w, hf.r0

    def P(a, b):
        return W0 + f * (a * L) + g * (b * w)

    name = f"hand-{s}"
    inner: List[Item] = []
    if paw:
        Lp, wpw = 0.88 * L, 1.18 * w
        palm = _ellipse(P(0.5 * 0.88, 0.0), 0.47 * Lp, 0.5 * wpw, ang=math.atan2(f[1], f[0]), n=36)
        wrist = _capsule(W0 - f * 0.05 * L, P(0.35, 0), r0, 0.42 * wpw, n=10)
        toes = []
        for k, b in enumerate((-0.32, 0.0, 0.32)):
            c = P(0.8 + (0.04 if k == 1 else 0.0), b * 1.1)
            toes.append(_ellipse(c, 0.17 * L, 0.16 * w * 1.15, ang=math.atan2(f[1], f[0]), n=20))
        dew = _ellipse(P(0.33, 0.56), 0.12 * L, 0.11 * w, ang=math.atan2(f[1], f[0]), n=16)
        ctrl = _union_ctrl([palm, wrist, dew] + toes)
        for b in (-0.16, 0.16):
            inner.append(Item(f"{name} toe", np.array([P(0.9, b * 1.05), P(0.8, b), P(0.69, b * 0.92)]),
                              weight=INNER * 0.95))
        inner.append(Item(f"{name} pad", np.array([P(0.48, -0.22), P(0.42, 0.0), P(0.48, 0.22)]),
                          weight=INNER * 0.8))
        return [Item(name, ctrl, closed=True, weight=SIL, fill=True)] + inner
    palm = _capsule(W0 - f * 0.05 * L, P(0.5, 0.0), r0, 0.5 * w, n=12)
    base = _ellipse(P(0.42, -0.02), 0.25 * L, 0.5 * w, ang=math.atan2(f[1], f[0]), n=28)
    thumb_root, thumb_tip = P(0.22, 0.32), P(0.52, 0.72)
    thumb = _capsule(thumb_root, thumb_tip, 0.15 * L, 0.11 * L, n=10)
    parts = [palm, base, thumb]
    if open_:
        for b, ang, ln in ((-0.36, -0.22, 0.38), (-0.12, -0.07, 0.47), (0.12, 0.07, 0.5), (0.36, 0.2, 0.45)):
            root = P(0.55, b)
            dirv = _unit(f * math.cos(ang) + g * math.sin(ang))
            parts.append(_capsule(root, root + dirv * ln * L, 0.085 * L, 0.07 * L, n=10))
        thumb_tip = P(0.42, 0.92)
        parts[2] = _capsule(thumb_root, thumb_tip, 0.13 * L, 0.1 * L, n=10)
    else:
        parts.append(_capsule(P(0.55, 0.0), P(0.86, -0.04), 0.48 * w, 0.42 * w, n=14))
        for b in (-0.18, 0.12):
            inner.append(Item(f"{name} finger", np.array([P(1.0, b), P(0.9, b * 1.02), P(0.78, b * 1.05)]),
                              weight=INNER * 0.85))
    ctrl = _union_ctrl(parts)
    inner.append(Item(f"{name} thumb", np.array([P(0.3, 0.2), P(0.42, 0.36), P(0.5, 0.5)]), weight=INNER * 0.95))
    return [Item(name, ctrl, closed=True, weight=SIL, fill=True)] + inner


def grip_items(hf: HandFrame, s: str, C: np.ndarray, a: np.ndarray, half: float, paw: bool) -> List[Item]:
    """A hand closed round a handle through C along a (``half`` = the handle's half width), built from three
    forms: the palm mass on the arm's side of the handle, the finger band that wraps across the front of it (the
    curled finger ends, or toe bumps for a paw, round off past its far edge; the creases between the fingers bend
    round the handle) and the thumb, which closes the other way: it lies over the top of the band, across the
    handle, its tip on the far side."""
    L, w, r0 = hf.L, hf.w, hf.r0
    n = _perp(a)
    if n @ (hf.E - C) < 0:
        n = -n                                 # n points to the arm's side
    up = a if a[1] < 0 else -a                 # along the handle, upward on the picture
    ang_n = math.atan2(n[1], n[0])
    fa = 0.5 * w * (1.1 if paw else 1.0)      # half height of the fist along the handle
    name = f"hand-{s}"
    # the palm mass between the wrist and the handle, just over the handle's near edge
    Pc = C + n * (half + 0.3 * L)
    palm = _ellipse(Pc, 0.34 * L, 1.02 * fa, ang=ang_n, n=36)
    bridge = _capsule(hf.W0 - hf.f * 0.05 * L, Pc, r0, 0.8 * fa, n=12)
    items = [Item(name, _union_ctrl([palm, bridge]), closed=True, weight=SIL, fill=True)]
    # the finger band across the handle: from over the palm to past the far edge, the curled ends round there
    k_near, k_far = half + 0.12 * L, half + 0.17 * L
    Bc = C + n * (k_near - k_far) / 2
    band_parts = [_superellipse(Bc, n, up, (k_near + k_far) / 2, 0.92 * fa, pw=3.0, n=48)]
    tips = (-0.62, 0.0, 0.62) if paw else (-0.69, -0.23, 0.23, 0.69)
    for b in tips:
        band_parts.append(_ellipse(C - n * (k_far - 0.06 * L) + up * b * fa, (0.13 if paw else 0.1) * L,
                                   (0.33 if paw else 0.25) * fa, ang=ang_n, n=18))
    items.append(Item(f"{name} fingers", _union_ctrl(band_parts), closed=True, weight=SIL, fill=True))
    # the creases between the fingers: from the notches at the far end, bending round the handle (lower in the
    # middle, as a band round a post seen a little from above)
    creases = [0.5 * (x + y) for x, y in zip(tips[:-1], tips[1:])]
    for b in creases:
        y0 = up * b * fa
        items.append(Item(f"{name} finger", np.array([C - n * (k_far + 0.02 * L) + y0,
                                                      C - n * (0.3 * half) + y0 - up * 0.14 * fa,
                                                      C + n * (half + 0.04 * L) + y0 - up * 0.04 * fa]),
                          weight=INNER, taper="end"))
    # the thumb closes from the other side: over the top of the band, across the handle, its tip past the far edge;
    # its root lies on the palm (the rounded end there reads as the ball of the thumb)
    rt_, tt_ = (0.13, 0.11) if paw else (0.12, 0.1)
    th_root = C + n * (half + 0.3 * L) + up * 0.52 * fa
    th_tip = C - n * (half + 0.04 * L) + up * 0.82 * fa
    items.append(Item(f"{name} thumb", _capsule(th_root, th_tip, rt_ * L, tt_ * L, n=12), closed=True, weight=SIL,
                      fill=True))
    if not paw:
        items.append(Item(f"{name} thumb nail", np.array([th_tip + n * 0.05 * L - up * 0.05 * fa,
                                                          th_tip + n * 0.1 * L + up * 0.02 * fa,
                                                          th_tip + n * 0.05 * L + up * 0.1 * fa]),
                          weight=INNER * 0.8))
    return items


def find_handle(p, near: np.ndarray, radius: float, angle: Optional[float]):
    """A drawn handle near ``near`` (px): (centre point on its axis, unit direction, half width) or None."""
    d = dc.dstate(p)
    pts, dirs = [], []
    best = None
    for r in d["lines"]:
        if r.get("wear"):
            continue
        P = draw_line.record_path(p, r)
        if len(P) < 3:
            continue
        dd = np.hypot(P[:, 0] - near[0], P[:, 1] - near[1])
        # every point near the grip, not only the nearest: one record can hold both sides of a handle (a tube)
        for j in np.nonzero(dd <= radius)[0]:
            a0, a1 = max(0, j - 6), min(len(P) - 1, j + 6)
            dv = _unit(P[a1] - P[a0])
            pts.append(P[j])
            dirs.append(dv)
            if best is None or dd[j] < best[0]:
                best = (float(dd[j]), P[j], dv)
    if best is None:
        return None
    a = best[2] if angle is None else np.array([math.cos(angle), -math.sin(angle)])
    n = _perp(a)
    # offsets across the handle of the lines that run along it, beside the grip point; grouped into lines
    offs = sorted(float((q - near) @ n) for q, dv in zip(pts, dirs)
                  if abs(float(dv @ a)) > 0.94 and abs(float((q - near) @ a)) < 0.5 * radius)
    groups: List[List[float]] = []
    for o in offs:
        if groups and o - groups[-1][-1] < 2.5:
            groups[-1].append(o)
        else:
            groups.append([o])
    c0 = float((best[1] - near) @ n)
    means = [float(np.mean(g)) for g in groups]
    others = [m for m in means if 3.0 < abs(m - c0) < 0.45 * radius]
    if others:
        c1 = min(others, key=lambda m: abs(m - c0))
        lo, hi = min(c0, c1), max(c0, c1)
    else:
        lo = hi = c0
    centre = near + n * (lo + hi) / 2
    return centre, a, (hi - lo) / 2


# ============================================================================================ placing a piece

def _taper_of(src: dict, t0: bool, t1: bool) -> str:
    if src.get("closed"):
        return "none"
    tp = src.get("taper", "both")
    a = t0 and tp in ("both", "start")
    b = t1 and tp in ("both", "end")
    return "both" if a and b else "start" if a else "end" if b else "none"


def _frags_of(P: np.ndarray, runs, closed: bool):
    """(points, starts at the original start, ends at the original end) for each run of a path."""
    n = len(P)
    if closed and n > 2 and math.hypot(*(P[0] - P[-1])) < 1e-6:
        n -= 1
        P = P[:n]
    out = []
    for a, b in runs:
        idx = np.arange(a, b) % n
        out.append((P[idx], (not closed) and a == 0, (not closed) and b == n))
    return out


def _cut_item(it: Item, mask: Optional[np.ndarray], dist: Optional[np.ndarray], k: dict) -> bool:
    new, changed = [], False
    for pts, t0, t1 in it.frags:
        closed = it.closed and it.intact
        also = _dup_flags(pts, dist, k["tol"], k["min_dup"], closed) if dist is not None else None
        runs = outside_runs(pts, mask, closed=closed, min_cut=k["min_cut"], min_keep=k["min_keep"], also=also)
        nuniq = len(pts) - (1 if closed and len(pts) > 2 and math.hypot(*(pts[0] - pts[-1])) < 1e-6 else 0)
        if runs == [(0, nuniq)]:
            new.append((pts, t0, t1))
            continue
        changed = True
        for q, a0, a1 in _frags_of(pts, runs, closed):
            new.append((q, t0 and a0, t1 and a1))
    if changed:
        it.frags = new
        it.intact = False
    return changed


def _knobs(p, weight: float = SIL) -> dict:
    lw = draw_line.base_width(p) * weight
    return {"lw": lw, "erode": max(1.0, 0.32 * lw), "min_cut": 1.2 * lw, "min_keep": 2.5 * lw, "tol": 0.5 * lw,
            "min_dup": 3.2 * lw}


def _label_ids(ids: List[str]) -> str:
    if not ids:
        return "-"
    nums = [int(i[1:]) for i in ids if i[1:].isdigit()]
    if len(nums) == len(ids) and nums == list(range(nums[0], nums[0] + len(nums))) and len(nums) > 2:
        return f"{ids[0]}-{ids[-1]}"
    return ", ".join(ids)


def _part_for(p, subj: str, kind: str, side: Optional[str] = None) -> Tuple[str, bool]:
    """(the planned part that names the piece, True when one matched) - else the piece's own name."""
    d = dc.dstate(p)
    parts = {n: v for n, v in d["parts"].items() if v.get("of") == subj}
    words = PART_WORDS.get(kind, (kind,))
    names = list(parts)
    if kind == "hand" and side:
        for n in names:
            base = re.sub(r"[-_ ]?(l|r|left|right)$", "", n)
            sfx = n[len(base):].strip("-_ ").lower()[:1]
            if base in words and sfx == side:
                return n, True
        for n in names:
            if n in words:
                return n, True
        for n in names:
            if any(w in n for w in words) and not re.search(r"[-_ ](l|r|left|right)$", n):
                return n, True
        return f"hand-{side}", False
    for w in words:
        for n in names:
            if n == w or re.sub(r"[-_ ]?(l|r|left|right|\d+)$", "", n) == w:
                return n, True
    if kind in ("sleeves", "collar", "hood"):
        for g in ("coat", "robe", "dress", "shirt"):
            for w in PART_WORDS[g]:
                if w in parts:
                    return w, True
    if kind in ("shirt", "coat", "dress", "robe"):
        for w in ("torso", "body", "chest"):       # a planned torso is what the clothes on it draw
            if w in parts:
                return w, True
    return kind, False


def place(p, piece: Piece, part: str, under: Sequence[str] = (), over: Sequence[str] = ()) -> Tuple[list, tuple, List[str], List[str]]:
    """Fit the piece's items, hide what is behind what, cut the earlier lines the piece covers, add its records.
    Returns (ops, touched px box, report lines, new ids)."""
    d = dc.dstate(p)
    W, H = p.W, p.H
    k = _knobs(p)
    lines: List[str] = []
    under = {str(x).lower() for x in under}
    over = {str(x).lower() for x in over}
    # an earlier version of this piece is replaced
    old = [r for r in d["lines"] if r.get("wear") and piece.owns(r["wear"])]
    for r in old:
        d["lines"].remove(r)
    if old:
        lines.append(f"replaced the earlier {piece.kind} ({_label_ids([r['id'] for r in old])}); lines it had cut "
                     "stay cut (easel undo brings them back)")
    for it in piece.items:
        ctrl = np.asarray(it.ctrl, float)
        if it.closed:
            ctrl = _start_at_corner(ctrl)
            it.ctrl = ctrl
        elif it.taper != "none":
            # a fold with a free end stays shorter than the lines ``line gaps`` checks (10 % of the canvas)
            ctrl = _shorten(ctrl, 0.085 * dc.side(p))
            it.ctrl = ctrl
        path, _ = draw_line.fit(p, [list(q) for q in dc.px2u(p, ctrl)], closed=it.closed, snap=False, join=False,
                                bow=0.0)
        it.path = np.asarray(path, float)
        it.frags = [(it.path, True, True)]
        it.intact = True
        if it.within is not None and not it.closed:
            # a fold creases its own part: what would run out over its outline (onto another part) is cut off
            inside = shape_mask(W, H, it.within, erode=0.25 * k["lw"])
            _cut_item(it, ~inside, None, dict(k, min_cut=0.0))
        if it.behind is not None:
            # its outline stops on the body's line (a hair past it, so the two lines join without a gap)
            _cut_item(it, shape_mask(W, H, it.behind, dilate=0.15 * k["lw"]), None, dict(k, min_cut=0.0))
    # inside the piece, a later shape hides the earlier items
    for i, it in enumerate(piece.items):
        if not it.fill or i == 0:
            continue
        m = shape_mask(W, H, it.path, erode=k["erode"])
        dm = _dist_map(H, W, [it.path])
        for prev in piece.items[:i]:
            # a line that only just reaches under a nearer shape is cut there too: no tip of it pokes onto it (the
            # nearer outline covers the cut)
            _cut_item(prev, m, dm, dict(k, min_cut=0.0))
    # what stays in front of the piece: the body (neck, head) and pieces nearer the viewer
    front = np.zeros((H, W), bool) if piece.front is None else piece.front.copy()
    front_lines = []
    seen = set()
    groups: Dict[tuple, list] = {}
    for r in d["lines"]:
        w = r.get("wear")
        if not w or not piece.in_front(w):
            continue
        front_lines.append(draw_line.record_path(p, r))
        key = (w.get("of"), w.get("piece"), w.get("side"))
        poly = draw_line.record_path(p, r) if r.get("closed") else (dc.u2px(p, r["shape"]) if r.get("shape") else None)
        if poly is None or len(poly) < 3 or key + (w.get("item"),) in seen:
            continue
        seen.add(key + (w.get("item"),))
        g = groups.setdefault(key, [np.zeros((H, W), bool), np.zeros((H, W), bool)])
        role = _wear_role(r)
        if role == "hole":
            g[1] |= shape_mask(W, H, poly)
        elif role == "fill":
            g[0] |= shape_mask(W, H, poly)
    for f_, h_ in groups.values():
        front |= f_ & ~h_
    if piece.bare is not None:
        front &= ~piece.bare
    if front.any() or front_lines:
        fm = dc.erode(front, k["erode"]) if front.any() else None
        dmf = _dist_map(H, W, front_lines) if front_lines else None
        hidden = 0
        for it in piece.items:
            hidden += _cut_item(it, fm, dmf, k)
        if hidden:
            lines.append(f"{hidden} of its outlines go behind what is in front (" +
                         ("the neck and head" if piece.front is not None else "nearer pieces") + ")")
    # the piece hides the earlier lines behind it
    vis = np.zeros((H, W), bool)
    for it in piece.items:
        if it.fill:
            m = shape_mask(W, H, it.path)
            if it.hole is not None:
                m &= ~shape_mask(W, H, it.hole, dilate=0.6 * k["lw"])
            if it.behind is not None:
                m &= ~shape_mask(W, H, it.behind)      # the body in front keeps its lines
            vis |= m
    vis &= ~front
    cut_mask = dc.erode(vis, k["erode"])
    new_paths = [q for it in piece.items for q, _, _ in it.frags]
    dmn = _dist_map(H, W, new_paths)
    ys, xs = np.nonzero(vis)
    vbox = (xs.min() - 4, ys.min() - 4, xs.max() + 5, ys.max() + 5) if len(xs) else None

    def eligible(r) -> bool:
        rid, prt = r["id"].lower(), (r.get("part") or "").lower()
        if rid in under or prt in under or (prt and any(prt.startswith(u + "-") for u in under)):
            return False
        if rid in over or prt in over or (prt and any(prt.startswith(o + "-") for o in over)):
            return True
        w = r.get("wear")
        if w:
            if w.get("of") != piece.subject:
                return True
            return not piece.in_front(w)
        if _is_hair(prt) or _is_head(prt):
            return piece.hood_up
        return True

    cut_info: List[str] = []
    edits = []            # (record, fragments)
    bare_cut = (cut_mask & piece.bare) if piece.bare is not None else None
    if vbox is not None:
        for r in d["lines"]:
            mask_r = cut_mask
            if not eligible(r):
                w = r.get("wear")
                # a piece in front of this one is cut only where this one comes over it (``bare``)
                if bare_cut is None or not w or w.get("of") != piece.subject or not piece.in_front(w):
                    continue
                mask_r = bare_cut
            P = draw_line.record_path(p, r)
            if len(P) < 2:
                continue
            if P[:, 0].max() < vbox[0] or P[:, 0].min() > vbox[2] or P[:, 1].max() < vbox[1] or \
                    P[:, 1].min() > vbox[3]:
                continue
            closed = bool(r.get("closed"))
            also = _dup_flags(P, dmn, k["tol"], k["min_dup"], closed) if mask_r is cut_mask else None
            runs = outside_runs(P, mask_r, closed=closed, min_cut=k["min_cut"], min_keep=k["min_keep"], also=also)
            nuniq = len(P) - (1 if closed and len(P) > 2 and math.hypot(*(P[0] - P[-1])) < 1e-6 else 0)
            if runs == [(0, nuniq)]:
                continue
            frs = _frags_of(P, runs, closed)
            if _stray_arc(p, P, closed, [f[0] for f in frs]):
                frs = []            # a button half under the piece would read as a stray curl: it goes behind
            edits.append((r, frs, P))
    # the piece's own records
    ops = []
    bb = None
    new_ids: List[str] = []
    named: Dict[str, List[str]] = {}
    for it in piece.items:
        frags = [f for f in it.frags if len(f[0]) >= 2 and dc.path_length(f[0]) >= k["min_keep"] * 0.6]
        if not it.intact and _stray_arc(p, it.path, it.closed, [f[0] for f in frags]):
            frags = []
        if not frags:
            continue
        wear = {"of": piece.subject, "piece": piece.kind, "item": it.name, "z": piece.z}
        if piece.side:
            wear["side"] = piece.side
        if it.fill:
            wear["fill"] = True
        if it.is_hole:
            wear["hole"] = True
        frags.sort(key=lambda f: -dc.path_length(f[0]))
        for j, (q, t0, t1) in enumerate(frags):
            if it.intact:
                rec = draw_line.make_record(p, q, dc.px2u(p, it.ctrl), weight=it.weight,
                                            taper="none" if it.closed else it.taper, color="line", closed=it.closed,
                                            src=f"wear {piece.subject} {piece.kind}")
                if it.fill:
                    # the area it closes round, kept on the record: a later tool that splits the outline (locks in
                    # front of the coat) copies it, so flat hints still know the piece's area
                    rec["shape"] = dc.rnd(dc.px2u(p, dc.rdp(it.path, 0.6)), 1)
            else:
                ctrl = dc.px2u(p, dc.rdp(q, 0.01 * dc.side(p)))
                tp = "none" if it.closed else _taper_of({"taper": it.taper}, t0, t1)
                rec = draw_line.make_record(p, q, ctrl, weight=it.weight, taper=tp, color="line", closed=False,
                                            src=f"wear {piece.subject} {piece.kind}")
                if it.fill and j == 0:
                    rec["shape"] = dc.rnd(dc.px2u(p, dc.rdp(it.path, 0.6)), 1)
            rec["part"] = part
            rec["wear"] = dict(wear)
            d["lines"].append(rec)
            new_ids.append(rec["id"])
            named.setdefault(it.name.split(" ")[0] if " " not in it.name else it.name.split(" ", 1)[1], []).append(
                rec["id"])
            ops += draw_line.render_ops(p, rec)
            bb = coords.union_box(bb, _path_box(p, q))
    # the earlier lines it covers
    for r, frags, P in edits:
        idx = d["lines"].index(r)
        bb = coords.union_box(bb, _path_box(p, P))
        frags = [f for f in frags if len(f[0]) >= 2]
        frags.sort(key=lambda f: -dc.path_length(f[0]))
        if not frags:
            d["lines"].pop(idx)
            cut_info.append(f"{r['id']} removed")
            continue
        recs = []
        for j, (q, t0, t1) in enumerate(frags):
            nr = {kk: v for kk, v in r.items() if kk not in ("path", "ctrl", "id", "shape")}
            nr["id"] = r["id"] if j == 0 else dc.next_id(d, "L")
            nr["path"] = dc.rnd(dc.px2u(p, dc.rdp(q, 0.25)), 2)
            nr["ctrl"] = dc.rnd(dc.px2u(p, dc.rdp(q, 0.01 * dc.side(p))))
            nr["closed"] = False
            nr["taper"] = _taper_of(r, t0, t1)
            if j == 0:
                if r.get("shape"):
                    nr["shape"] = r["shape"]
                elif r.get("closed") and r.get("wear"):
                    nr["shape"] = dc.rnd(dc.px2u(p, dc.rdp(P, 0.6)), 1)
            recs.append(nr)
        d["lines"][idx:idx + 1] = recs
        for nr in recs:
            ops += draw_line.render_ops(p, nr)
        cut_info.append(r["id"] + (f" -> {', '.join(x['id'] for x in recs)}" if len(recs) > 1 else ""))
    summary = "; ".join(f"{nm} {_label_ids(ids)}" for nm, ids in named.items())
    lines.insert(0, f"{piece.kind} of {piece.subject} (part {part}): {summary}")
    if cut_info:
        lines.append(f"cut under the {piece.kind} (where it covers them): " + ", ".join(cut_info[:8])
                     + (f" ... {len(cut_info) - 8} more" if len(cut_info) > 8 else ""))
    lines.extend(piece.notes)
    return ops, bb, lines, new_ids


def _stray_arc(p, P, closed: bool, kept) -> bool:
    """A small closed outline (a button, a bead) that something covers in part: the arc left of it reads as a
    stray curl, not as a shape, so the whole outline goes behind."""
    if not closed or len(P) < 3:
        return False
    total = dc.path_length(np.vstack([P, P[:1]]))
    if total > 0.12 * dc.side(p):
        return False
    return sum(dc.path_length(q) for q in kept) < 0.8 * total


def _path_box(p, path):
    a = np.asarray(path, float).reshape(-1, 2)
    if not len(a):
        return None
    pad = 3 * draw_line.base_width(p)
    x0, y0 = max(0, a[:, 0].min() - pad), max(0, a[:, 1].min() - pad)
    x1, y1 = min(p.W, a[:, 0].max() + pad + 1), min(p.H, a[:, 1].max() + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _wear_role(r) -> Optional[str]:
    """'hole' (a hood's face opening), 'fill' (an outline that closes round an area of the piece) or None (a fold)
    for a wear record; records made before the flags existed are told by their item name and stored shape."""
    w = r.get("wear") or {}
    if w.get("hole") or w.get("item") == "hood opening":
        return "hole"
    if w.get("fill") or r.get("closed") or r.get("shape"):
        return "fill"
    return None


def wear_shapes(p, subject: Optional[str] = None) -> List[Tuple[str, Optional[str], np.ndarray, bool]]:
    """(part, swatch, filled outline px, is a hole) of every drawn piece's area, the one nearest the viewer first; a
    hole (the face opening of a hood) takes its point out of the piece around it."""
    d = dc.dstate(p)
    rows, seen = [], set()
    for r in d["lines"]:
        w = r.get("wear")
        if not w or (subject is not None and w.get("of") != subject):
            continue
        poly = draw_line.record_path(p, r) if r.get("closed") else (dc.u2px(p, r["shape"]) if r.get("shape") else None)
        key = (w.get("of"), w.get("piece"), w.get("item"))
        role = _wear_role(r)
        if poly is None or len(poly) < 3 or key in seen or role is None:
            continue
        seen.add(key)
        part = r.get("part") or w.get("piece")
        sw = d["parts"][part].get("swatch") if part in d["parts"] else (part if part in d["palette"]["swatches"]
                                                                          else None)
        if sw is None and w.get("piece") == "hand" and "skin" in d["palette"]["swatches"]:
            sw = "skin"
        hole = role == "hole"
        rows.append((float(w.get("z", 0.0)) + (0.5 if hole else 0.0), int(r.get("n", 0)), part, sw,
                     np.asarray(poly, np.float32), hole))
    rows.sort(key=lambda t: (-t[0], -t[1]))
    return [(part, sw, poly, hole) for _, _, part, sw, poly, hole in rows]


def area_part(p, pt_px, shapes=None) -> Optional[Tuple[str, Optional[str]]]:
    """(part, swatch) of the drawn piece whose area holds the point (px), the front-most one; None when no piece
    covers it. ``flat auto`` can ask this before it falls back to the plan boxes."""
    holed = set()
    for part, sw, poly, hole in (shapes if shapes is not None else wear_shapes(p)):
        if part in holed:
            continue
        if cv2.pointPolygonTest(poly.reshape(-1, 1, 2), (float(pt_px[0]), float(pt_px[1])), False) > 0:
            if hole:
                holed.add(part)
                continue
            return part, sw
    return None


def wear_corrections(p, labels, table, auto: Dict[int, Tuple[str, str]],
                     subject: Optional[str] = None) -> Dict[int, Tuple[str, str]]:
    """area -> (part, swatch) for the areas round the clothes (of one subject, or of every dressed subject) that
    the plan boxes colour wrongly (``auto`` is what they give): areas of a piece that get another colour, and areas
    outside every piece (a face in a hood, background between an arm and the body) that get a piece's colour. Hair
    or a held thing in front of a piece (an area bounded by its own lines) keeps its colour. ``flat auto``
    (``draw_flat.auto_assign``) applies these after its box pass."""
    from .draw_plan import parts_at
    d = dc.dstate(p)
    shapes = wear_shapes(p, subject)
    worn = {part for part, _, _, _ in shapes}
    dressed = {subject} if subject is not None else {(r.get("wear") or {}).get("of") for r in d["lines"]
                                                     if r.get("wear")}
    fixes: Dict[int, Tuple[str, str]] = {}
    drawn: Dict[str, np.ndarray] = {}

    def lines_of(part):
        """the part's own (not worn) lines, drawn 3 px wide"""
        if part not in drawn:
            m = np.zeros(labels.shape, np.uint8)
            for rec in d["lines"]:
                if (rec.get("part") or "") == part and not rec.get("wear"):
                    P = np.round(draw_line.record_path(p, rec)).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(m, [P], bool(rec.get("closed")), 1, 3)
            drawn[part] = m > 0
        return drawn[part]

    for r in table:
        hit = area_part(p, r["anchor"], shapes)
        got = auto.get(r["n"])
        if hit is not None:
            if not hit[1] or (got is not None and got[0] == hit[0]):
                continue
            if got is not None and got[0] not in worn and got[0] != "background" and                     not got[0].startswith("neck"):
                # hair or a held thing in front of the piece is bounded by its own lines: keep it (a neck never
                # is: the clothes round it are in front of it)
                ring = cv2.dilate((labels == r["n"]).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                if int((ring & lines_of(got[0])).sum()) > 8:
                    continue
            want = hit
        else:
            if got is None or got[0] not in worn:
                continue
            # not inside any drawn piece: the smallest planned part round it that is not a piece, else the
            # background (unless it lies in another subject, which then decides)
            u = dc.px2u(p, [r["anchor"]])[0]
            alt = [n for n in parts_at(d, u) if n not in worn]
            if alt:
                want = (alt[0], d["parts"][alt[0]].get("swatch"))
            elif any(n not in dressed and b["box"][0] <= u[0] <= b["box"][2] and b["box"][1] <= u[1] <= b["box"][3]
                     for n, b in d.get("subjects", {}).items()):
                continue
            else:
                want = ("background", "background")
            if not want[1]:
                continue
        fixes[r["n"]] = (want[0], want[1])
    fixes.update(_neck_side_fixes(p, labels, table, auto, fixes, dressed))
    return fixes


def _neck_side_fixes(p, labels, table, auto, fixes, subjects) -> Dict[int, Tuple[str, str]]:
    """Areas under the jaw beside a drawn neck that the head's box colours as skin: they are what is behind the
    neck, the back hair when hair lines bound them, else the background (the skin stays a neck, not a column as
    wide as the face)."""
    d = dc.dstate(p)
    out: Dict[int, Tuple[str, str]] = {}
    for subj in subjects:
        if not subj or not _neck_lines_of(p, subj):
            continue
        fig = next((el for el in d["sketch"] if el["kind"] == "figure" and (el.get("subject") or el["id"]) == subj),
                   None)
        if fig is None:
            continue
        body = Body(p, fig, _head_of(p, subj))
        detect_neck(p, body)
        skin = {n for n, v in d["parts"].items() if v.get("of") == subj and _is_head(n)
                and not n.startswith(("eye", "brow", "mouth", "nose"))}
        hair = next((n for n, v in d["parts"].items() if v.get("of") == subj and _is_hair(n)), None)
        hair_lines = np.zeros(labels.shape, np.uint8)
        if hair is not None:
            for rec in d["lines"]:
                if (rec.get("part") or "") == hair:
                    P = np.round(draw_line.record_path(p, rec)).astype(np.int32).reshape(-1, 1, 2)
                    cv2.polylines(hair_lines, [P], False, 1, 3)
        jaw = [_jaw_y(p, body, subj, body.B(body.sg(s) * body.nw[s], 0.0)[0]) for s in "lr"]
        yj = max((v for v in jaw if v is not None), default=None)
        if yj is None:
            continue
        for r in table:
            got = fixes.get(r["n"]) or auto.get(r["n"])
            if got is None or got[0] not in skin:
                continue
            m = labels == r["n"]
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            L = body.locs(np.stack([xs, ys], 1).astype(float))
            side = np.where(L[:, 0] < 0, body.nw["l"], body.nw["r"])
            beside = (np.abs(L[:, 0]) > 1.02 * side) & (ys > yj - 0.1 * body.Hh) & \
                (L[:, 1] < body.y_ns + 0.5 * body.nl)
            if beside.mean() < 0.6:
                continue
            ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            if hair is not None and int((ring & (hair_lines > 0)).sum()) > 8:
                out[r["n"]] = (hair, d["parts"][hair].get("swatch") or hair)
            else:
                out[r["n"]] = ("background", "background")
    return out


GROWS = tuple(GARMENTS) + ("sleeves", "hood", "collar", "pants", "skirt", "boots", "hand", "scarf")


def _grow_part(p, part: str, ids: List[str], fit_box: bool = False) -> Optional[str]:
    """Let the planned part's box hold the drawn piece, so ``flat auto`` finds every area of it."""
    d = dc.dstate(p)
    if part not in d["parts"] or not ids:
        return None
    pts = np.vstack([np.asarray(r["path"], float) for r in d["lines"] if r["id"] in ids])
    pts = pts[(pts[:, 0] > -50) & (pts[:, 0] < 1050) & (pts[:, 1] > -50) & (pts[:, 1] < 1050)]
    if not len(pts):
        return None
    lo, hi = np.clip(pts.min(0), 0, 1000), np.clip(pts.max(0), 0, 1000)
    b = list(d["parts"][part]["box"])
    if fit_box:
        nb = [lo[0] - 8, lo[1] - 8, hi[0] + 8, hi[1] + 8]
    else:
        nb = [min(b[0], lo[0]), min(b[1], lo[1]), max(b[2], hi[0]), max(b[3], hi[1])]
    nb = [round(float(np.clip(v, 0, 1000)), 1) for v in nb]
    if max(abs(nb[i] - b[i]) for i in range(4)) < 4:
        return None
    d["parts"][part]["box"] = nb
    return f"plan part {part} box now {coords.fmt_box(nb)} (it holds the drawn {part}, so flat auto colours it)"


# ============================================================================================ commands

def _names(text) -> List[str]:
    return [x.strip().lower() for x in re.split(r"[,\s]+", str(text or "")) if x.strip()]


def _subject_and_piece(p, a: dict) -> Tuple[str, str]:
    d = dc.dstate(p)
    figs = [el for el in d["sketch"] if el["kind"] == "figure"]
    subj = _clean_word(a.get("subject") or "")
    piece = _clean_word(a.get("piece") or "")
    if not piece and (subj in PIECES or subj in PIECE_ALIAS):
        if len(figs) != 1:
            names = ", ".join(str(el.get("subject") or el["id"]) for el in figs) or "none yet"
            raise UsageError(f"say whose {subj}: the subject with the mannequin (figures: {names})",
                             f"easel wear {figs[0].get('subject') if figs else 'girl'} {subj}")
        piece, subj = subj, str(figs[0].get("subject") or figs[0]["id"])
    piece = PIECE_ALIAS.get(piece, piece)
    if not piece:
        raise UsageError("say which piece: " + ", ".join(PIECES), f"easel wear {subj or 'girl'} coat")
    if piece not in PIECES:
        import difflib
        close = difflib.get_close_matches(piece, list(PIECES) + list(PIECE_ALIAS), n=1, cutoff=0.6)
        hint = f" (did you mean {PIECE_ALIAS.get(close[0], close[0])}?)" if close else ""
        raise UsageError(f"unknown piece {piece!r}{hint}; pieces: {', '.join(PIECES)}",
                         f"easel wear {subj or 'girl'} {PIECE_ALIAS.get(close[0], close[0]) if close else 'coat'}")
    return subj, piece


def _choice(a: dict, key: str, options: Sequence[str], default: str, example: str) -> str:
    v = str(a.get(key) or default).strip().lower()
    v = {"left": "l", "right": "r"}.get(v, v) if "l" in options else v
    if v not in options:
        import difflib
        close = difflib.get_close_matches(v, list(options), n=1, cutoff=0.4)
        raise UsageError(f"--{key} is one of {', '.join(options)}" + (f" (did you mean {close[0]}?)" if close else ""),
                         example)
    return v


def _outer_garment_z(p, subj: str) -> Optional[float]:
    d = dc.dstate(p)
    zs = [float(r["wear"]["z"]) for r in d["lines"] if r.get("wear") and r["wear"].get("of") == subj and
          r["wear"].get("piece") in GARMENTS]
    return max(zs) if zs else None


def _drawn(p, subj: str) -> set:
    d = dc.dstate(p)
    out = set()
    for r in d["lines"]:
        w = r.get("wear")
        if w and w.get("of") == subj:
            out.add(w["piece"] if w["piece"] != "hand" else f"hand {w.get('side')}")
            if str(w.get("item", "")).startswith("sleeve"):
                out.add("sleeves")
    return out


def _next_hint(p, subj: str) -> Optional[str]:
    d = dc.dstate(p)
    drawn = _drawn(p, subj)
    for n, v in d["parts"].items():
        if v.get("of") != subj:
            continue
        base = re.sub(r"[-_ ]?(l|r|left|right|\d+)$", "", n)
        for kind, words in PART_WORDS.items():
            if base in words or n in words:
                if kind == "hand":
                    sfx = n[len(base):].strip("-_ ").lower()[:1]
                    for s in ([sfx] if sfx in ("l", "r") else ["l", "r"]):
                        if f"hand {s}" not in drawn:
                            return f"easel hand {subj} {s}   (a mitten hand from the wrist; --grip x,y to hold a handle)"
                elif kind not in drawn and not (kind == "sleeves" and drawn & set(GARMENTS)):
                    return f"easel wear {subj} {kind}"
                break
    return None


def cmd_wear(ctx, args: List[str]):
    from . import cli_draw
    from .cli import Spec, parse
    p = ctx.project
    d = dc.dstate(p)
    a = parse(args, Spec(USAGE["wear"], ["subject", "piece"],
                         {"length": "value", "fit": "value", "opening": "flag", "no-opening": "flag",
                          "buttons": "value", "sleeveless": "flag", "side": "value", "tips": "value", "down": "flag",
                          "up": "flag", "weight": "value", "under": "value", "over": "value"},
                         alias={"len": "length", "tip": "tips", "knot": "side", "behind": "under",
                                "no-sleeves": "sleeveless", "button": "buttons"}, min_pos=1))
    subj, kind = _subject_and_piece(p, a)
    usage = f"easel wear {subj} {kind}"
    el = find_figure(p, subj, USAGE["wear"].replace("girl", subj))
    subj = str(el.get("subject") or el["id"])
    body = Body(p, el, _head_of(p, subj))
    detect_neck(p, body)
    fit = _choice(a, "fit", tuple(FITS), "regular", f"{usage} --fit loose")
    notes: List[str] = []
    hood_up = False
    front = None
    side = None
    if kind in GARMENTS:
        g = GARMENTS[kind]
        length = _choice(a, "length", LENGTHS, g["length"], f"{usage} --length knee")
        opening = (bool(a.get("opening")) or g["opening"]) and not a.get("no-opening")
        buttons = g["buttons"] if a.get("buttons") is None else coords.parse_int(a["buttons"], "buttons", 0, 12,
                                                                              ctx.notes)
        if a.get("no-opening") and a.get("buttons") is None:
            buttons = 0
        sides = [] if a.get("sleeveless") else (["l", "r"] if not a.get("side") else
                                                [_choice(a, "side", ("l", "r"), "l", f"{usage} --side l")])
        front = head_front(p, body, subj)
        items = garment_items(body, kind, length, fit, opening, buttons, sides, head=front)
        z = g["z"]
        label = f"wear {subj} {kind} ({length}, {fit}" + (", no sleeves" if not sides else "") + ")"
    elif kind == "sleeves":
        sides = ["l", "r"] if not a.get("side") else [_choice(a, "side", ("l", "r"), "l", f"{usage} --side l")]
        oz = _outer_garment_z(p, subj)
        robe = any(r.get("wear", {}).get("piece") == "robe" and r["wear"].get("of") == subj for r in d["lines"])
        items = []
        for s in sides:
            items += sleeve_items(body, s, arm_frame(body, s, FITS[fit], GARMENTS["robe"]["cuff"] if robe else 1.0),
                                  robe=robe)
        z = (oz if oz is not None else 30.0) + Z["sleeve_dz"]
        side = sides[0] if len(sides) == 1 else None
        front = head_front(p, body, subj)
        label = f"wear {subj} sleeves"
    elif kind == "collar":
        items = collar_items(body)
        front = head_front(p, body, subj)
        z = (_outer_garment_z(p, subj) or 30.0) + Z["collar_dz"]
        label = f"wear {subj} collar"
    elif kind == "scarf":
        knot = _choice(a, "side", ("l", "r"), "l", f"{usage} --side r")
        tips = _choice(a, "tips", ("square", "pointed"), "square", f"{usage} --tips pointed")
        items = scarf_items(body, knot, tips)
        z = Z["scarf"]
        label = f"wear {subj} scarf (knot {knot}, {tips} ends)"
    elif kind == "hood":
        down = bool(a.get("down")) and not a.get("up")
        items, front = hood_items(body, down)
        hood_up = not down
        z = Z["hood"] if not down else (_outer_garment_z(p, subj) or 30.0) - 1.0
        label = f"wear {subj} hood ({'down' if down else 'up'})"
        if not down and body.head is None:
            notes.append(f"no head construction for {subj}: the hood follows the mannequin's head "
                         f"(easel sketch head {subj} first fits it to the face)")
    elif kind in ("pants", "skirt", "boots"):
        if body.bust:
            raise UsageError(f"{subj} is a bust: there are no legs in the picture for {kind}",
                             f"easel wear {subj} coat")
        length = _choice(a, "length", LENGTHS, "ankle" if kind == "pants" else "knee", f"{usage} --length knee")
        items = (pants_items(body, length, fit) if kind == "pants" else skirt_items(body, length, fit)
                 if kind == "skirt" else boots_items(body, fit))
        z = Z[kind]
        label = f"wear {subj} {kind}"
    else:
        raise UsageError(f"unknown piece {kind!r}; pieces: {', '.join(PIECES)}", f"easel wear {subj} coat")
    if not items:
        raise UsageError(f"nothing of the {kind} is inside the picture", f"easel wear {subj} coat")
    if a.get("weight"):
        wk = coords.parse_float(a["weight"], "weight", 0.3, 4, ctx.notes) / SIL
        for it in items:
            it.weight = round(it.weight * wk, 2)
    part, matched = _part_for(p, subj, kind, side)
    piece = Piece(subj, kind, z, items, side=side, hood_up=hood_up, front=front, notes=notes)
    return _run(ctx, p, piece, part, matched, label, _names(a.get("under")), _names(a.get("over")),
                body=body if kind in NECK_PIECES and not hood_up else None)


def cmd_hand(ctx, args: List[str]):
    from .cli import Spec, parse
    p = ctx.project
    a = parse(args, Spec(USAGE["hand"], ["subject", "side"],
                         {"grip": "value", "grip-angle": "value", "open": "flag", "paw": "flag", "weight": "value",
                          "under": "value", "over": "value"},
                         alias={"angle": "grip-angle", "gripangle": "grip-angle", "hold": "grip", "at": "grip",
                                "behind": "under"}, min_pos=1))
    subj = _clean_word(a.get("subject") or "")
    sd = _clean_word(a.get("side") or "")
    sides = ("l", "r", "left", "right")
    if subj in sides and sd and sd not in sides:
        subj, sd = sd, subj                    # easel hand r girl
    if subj in sides and not sd:
        figs = [el for el in dc.dstate(p)["sketch"] if el["kind"] == "figure"]
        if len(figs) == 1:                     # easel hand r (one figure)
            subj, sd = str(figs[0].get("subject") or figs[0]["id"]), subj
    side = {"left": "l", "right": "r"}.get(sd, sd)
    if side not in ("l", "r"):
        raise UsageError("say which hand: l (the picture's left) or r (the picture's right)",
                         f"easel hand {subj or 'girl'} r")
    el = find_figure(p, subj, USAGE["hand"].replace("girl", subj))
    subj = str(el.get("subject") or el["id"])
    body = Body(p, el, _head_of(p, subj))
    Wr = body.J[side + "wrist"]
    if body.below(Wr):
        raise UsageError(f"the {side} wrist is below the picture (a bust shows no hands); move it into the picture "
                         f"first", f"easel sketch joint {el['id']} {side}wr {coords.fmt_point(dc.px2u(p, [Wr])[0][:1].tolist() + [880])}")
    d = dc.dstate(p)
    cuff = None
    for r in d["lines"]:
        w = r.get("wear")
        if w and w.get("of") == subj and w.get("item") == f"sleeve-{side}":
            robe = any(x.get("wear", {}).get("piece") == "robe" and x["wear"].get("of") == subj for x in d["lines"])
            cuff = arm_frame(body, side, 1.0, GARMENTS["robe"]["cuff"] if robe else 1.0).rw
    hf = hand_frame(body, side, cuff)
    notes: List[str] = []
    paw = bool(a.get("paw"))
    ang = None
    if a.get("grip-angle") is not None:
        ang = math.radians(coords.parse_float(a["grip-angle"], "grip-angle", -360, 360, ctx.notes))
    grip = a.get("grip")
    if grip is not None or ang is not None:
        natural = hf.W0 + hf.f * 0.55 * hf.L
        if grip is not None and str(grip).strip().lower() not in ("auto", "here", "yes"):
            G = dc.u2px(p, [coords.parse_point(grip, ctx.notes, f"easel hand {subj} {side} --grip 640,520")])[0]
        else:
            G = natural
        found = find_handle(p, G, 0.45 * hf.L + 0.03 * max(p.W, p.H), ang)
        if found is not None:
            Cn, axis, half = found
        else:
            axis = np.array([math.cos(ang), -math.sin(ang)]) if ang is not None else np.array([0.0, 1.0])
            Cn, half = G, 0.0
            notes.append("no handle line there yet: draw the handle first and run the hand again, so the fingers "
                         "cover it")
        # the fist sits on the handle where it is nearest the hand
        C = Cn + axis * float((natural - Cn) @ axis) if grip is None or str(grip).lower() in ("auto", "here") else \
            Cn + axis * float((G - Cn) @ axis)
        far = _dist(C, hf.W0)
        if far > 1.1 * hf.L:
            C = Cn + axis * float((natural - Cn) @ axis)
            notes.append(f"the grip point is {far * 1000 / max(p.W, p.H):.0f} units from the {side} wrist: the fist "
                         f"sits on the handle nearest the hand; to hold it elsewhere move the wrist "
                         f"(easel sketch joint {el['id']} {side}wr x,y) and redo the sleeves")
        items = grip_items(hf, side, C, axis, half, paw)
        bare = _past_cuff(p, body, subj, side)
        label = f"hand {subj} {side} grip at {coords.fmt_point(dc.px2u(p, [C])[0])}" + (" (paw)" if paw else "")
    else:
        items = mitten_items(hf, side, bool(a.get("open")), paw)
        bare = None
        label = f"hand {subj} {side}" + (" (paw)" if paw else " (open)" if a.get("open") else "")
    if a.get("weight"):
        wk = coords.parse_float(a["weight"], "weight", 0.3, 4, ctx.notes) / SIL
        for it in items:
            it.weight = round(it.weight * wk, 2)
    part, matched = _part_for(p, subj, "hand", side)
    piece = Piece(subj, "hand", Z["hand"], items, side=side, notes=notes, bare=bare)
    return _run(ctx, p, piece, part, matched, label, _names(a.get("under")), _names(a.get("over")))


def _past_cuff(p, body: Body, subj: str, side: str) -> Optional[np.ndarray]:
    """The end of the side's sleeve beyond its cuff line (px mask): a fist comes out over it, the cuff line stays
    as the ring it comes out of. None when the sleeve or its cuff line is not drawn."""
    d = dc.dstate(p)
    shape = cuffl = None
    for r in d["lines"]:
        w = r.get("wear") or {}
        if w.get("of") != subj:
            continue
        if w.get("item") == f"sleeve-{side}" and shape is None and (r.get("closed") or r.get("shape")):
            shape = dc.u2px(p, r["shape"]) if r.get("shape") else draw_line.record_path(p, r)
        elif w.get("item") == f"sleeve-{side} cuff":
            cuffl = draw_line.record_path(p, r)
    if shape is None or cuffl is None or len(shape) < 3:
        return None
    d2 = _unit(body.J[side + "wrist"] - body.J[side + "elbow"])
    c0 = cuffl.mean(0)
    yy, xx = np.mgrid[0:p.H, 0:p.W]
    beyond = (xx - c0[0]) * d2[0] + (yy - c0[1]) * d2[1] > 0.8 * draw_line.base_width(p)
    m = shape_mask(p.W, p.H, shape) & beyond
    return m if m.any() else None


def _run(ctx, p, piece: Piece, part: str, matched: bool, label: str, under, over, body: Optional[Body] = None):
    from . import cli, cli_draw
    d = dc.dstate(p)
    info: List[str] = []

    def mutate():
        nops, nbb, nlines = add_neck(p, body, piece.subject) if body is not None else ([], None, [])
        ops, bb, lines, ids = place(p, piece, part, under, over)
        ops, bb = nops + ops, coords.union_box(nbb, bb)
        info.extend(lines)
        info.extend(nlines)
        if matched and piece.kind in GROWS:
            if piece.kind == "scarf":
                # only the wrap and the knot: a box round the hanging ends would also hold the coat beside them
                ids = [r["id"] for r in d["lines"] if r["id"] in ids and
                       str(r["wear"].get("item", "")).split(" ")[1:2] in (["wrap"], ["knot"])]
            grown = _grow_part(p, part, ids, fit_box=(piece.kind == "hand" and bool(re.search(r"[-_ ](l|r)$", part))))
            if grown:
                info.append(grown)
        info.append("each outline is a line record: easel line move / redo / weight / delete <id>; run the piece "
                    "again with other options to replace it")
        if piece.kind != "hand" and d.get("locks"):
            # hair locks drawn before stay in front of the new clothes (a hand is left in front of the hair)
            try:
                from . import draw_hair
                info.extend(draw_hair.refit(p))
            except Exception as e:  # noqa: BLE001 - a hair problem must never break wear
                info.append(f"the hair locks were not fitted over the {piece.kind} ({type(e).__name__}); "
                            f"run easel locks again to put them in front")
        return ops, bb

    rep = cli_draw.draw_step(ctx, label=label, layer="line",
                             owned=(lambda: dc.dstate(p)["lines"], draw_line.render_ops, mutate), lines=info,
                             post=cli_draw._line_post(p), overlay=cli_draw._line_overlay(p),
                             note="the line art (ids in blue, gaps magenta)", area_extra=cli_draw._subject_box_px(p))
    hint = _next_hint(p, piece.subject)
    if hint:
        rep.next = hint
    return rep


COMMANDS = {"wear": cmd_wear, "hand": cmd_hand}
