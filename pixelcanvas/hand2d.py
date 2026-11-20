"""Anime hands drawn over the posed skeleton.

A hand is built in 3D from the wrist frame (wrist, finger direction u, thumb side t, palm normal n): a palm wedge,
a thumb and four fingers of three phalanges each, curled by the hand shape. Every piece is projected (in
perspective, from the eye) onto one plane facing the viewer, rasterised into a small depth buffer, and the drawing
is read off that buffer the way an illustrator draws a hand:

- the silhouette: the outline of everything, closed and smoothed (fingers closer than a pixel or two on the canvas
  merge, so a small hand reads as a mitten with a thumb and an index finger);
- one to four inner lines: where a finger lies over the palm or another finger (depth edges: the curled fingers of
  a fist, the thumb across them) and the separations between neighbouring fingers.

The part returned is a flat shape2d blob (``flat3`` outline, ``details`` lines) lying in that plane, so it
foreshortens exactly as the posed hand does, front or back, left or right. A hand gripping a prop is two parts:
the fingers in front of the shaft and the palm and thumb behind it (when the palm faces the viewer).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import figure3d as F3
from .shape2d import unit

# finger curls (degrees at the three joints, index .. pinky), spread (degrees toward the thumb, index .. pinky),
# thumb (two segment directions in (u, t, n): along the fingers, toward the thumb side, out of the palm)
_FIST_THUMB = ((0.55, 0.2, 0.8), (0.25, -0.8, 0.55))
SHAPES: Dict[str, dict] = {
    "relaxed": {"curl": [(14, 22, 12), (22, 28, 14), (28, 32, 16), (34, 36, 18)], "spread": (8, 0, -3, -7),
                "thumb": ((0.7, 0.62, 0.3), (0.85, 0.2, 0.35))},
    "open": {"curl": [(4, 6, 4), (3, 5, 3), (4, 6, 4), (6, 8, 5)], "spread": (10, 2, -5, -12), "thumb": ((0.5, 0.85, 0.08), (0.7, 0.7, 0.0))},
    "spread": {"curl": [(2, 3, 2)] * 4, "spread": (18, 4, -9, -24), "thumb": ((0.3, 0.95, 0.0), (0.5, 0.86, 0.0))},
    "flat": {"curl": [(0, 2, 1)] * 4, "spread": (3, 0, -1, -3), "thumb": ((0.72, 0.62, 0.0), (0.92, 0.38, 0.0))},
    "reach": {"curl": [(10, 14, 8), (12, 16, 9), (15, 18, 10), (18, 22, 12)], "spread": (11, 2, -5, -12),
              "thumb": ((0.5, 0.78, 0.3), (0.75, 0.5, 0.25))},
    "fist": {"curl": [(92, 100, 70)] * 4, "spread": (2, 0, -1, -3), "thumb": _FIST_THUMB},
    "point": {"curl": [(0, 3, 2), (92, 100, 70), (92, 100, 70), (92, 100, 70)], "spread": (3, 0, -1, -3),
              "thumb": _FIST_THUMB},
    "peace": {"curl": [(0, 2, 2), (0, 2, 2), (92, 100, 70), (92, 100, 70)], "spread": (10, -9, -1, -3),
              "thumb": _FIST_THUMB},
    "cup": {"curl": [(30, 24, 14)] * 4, "spread": (3, 0, -2, -4), "thumb": ((0.7, 0.4, 0.55), (0.8, 0.0, 0.55))},
}
SHAPES["grip"] = SHAPES["fist"]
ALIASES = {"wave": "spread", "rest": "flat", "hip": "flat", "palm": "flat", "reach-out": "reach", "claw": "cup",
           "v": "peace", "hold": "grip"}

# the hand in hand lengths (L): palm outline (x along the fingers, y toward the thumb), knuckles, finger lengths
PALM = [(-0.08, -0.115), (0.2, -0.17), (0.44, -0.17), (0.5, -0.06), (0.51, 0.06), (0.47, 0.165), (0.2, 0.17),
        (-0.08, 0.115)]
KNUCKLES = [(0.47, 0.13), (0.5, 0.045), (0.49, -0.045), (0.45, -0.125)]
FINGER_LEN = (0.43, 0.47, 0.44, 0.35)
FINGER_R = (0.044, 0.045, 0.043, 0.037)
PHALANX = (0.45, 0.3, 0.25)
MITTEN_PX = 0.035          # a hand shorter than this share of the canvas is drawn as a mitten (one inner line)
GRID = 300                 # depth-buffer size across the hand (px)


def shape_of(shape: str) -> Tuple[str, dict]:
    s = str(shape or "relaxed").lower()
    s = ALIASES.get(s, s)
    if s not in SHAPES:
        s = "relaxed"
    return s, SHAPES[s]


def _curls_for_grip(R: float, L: float) -> List[Tuple[float, float, float]]:
    out = []
    for fl in FINGER_LEN:
        ph = np.array(PHALANX) * fl * L
        out.append(tuple(min(105.0, math.degrees(v / R)) for v in ph))
    return out


def hand_pieces(W, u, t, n, L: float, shape: str, grip_r: Optional[float] = None) -> List[dict]:
    """The hand's pieces in 3D: {"kind": "poly"|"cap", "label", "pts" (poly) or "a","b","ra","rb" (capsule)}."""
    name, sp = shape_of(shape)
    curls = sp["curl"]
    spread = sp["spread"]
    shift = np.zeros(3)
    if name == "grip" and grip_r:
        R = float(grip_r) + FINGER_R[1] * L
        curls = _curls_for_grip(R, L)
        spread = (2, 0, -1, -3)

    def P(x, y, z=0.0):
        return W + shift + u * (x * L) + t * (y * L) + n * (z * L)
    out = []
    # the wrist: a short capsule the forearm overlaps
    out.append({"kind": "cap", "label": 1, "a": P(-0.2, 0.0), "b": P(0.02, 0.0), "ra": 0.118 * L, "rb": 0.115 * L})
    # the palm: a flat wedge with a little thickness
    poly = [P(x, y, z) for z in (-0.05, 0.05) for x, y in PALM]
    out.append({"kind": "poly", "label": 1, "pts": np.array(poly)})
    # the thenar mound and the thumb
    tb = P(0.06, 0.1, 0.03)
    d1 = unit(sp["thumb"][0][0] * u + sp["thumb"][0][1] * t + sp["thumb"][0][2] * n)
    d2 = unit(sp["thumb"][1][0] * u + sp["thumb"][1][1] * t + sp["thumb"][1][2] * n)
    m = tb + d1 * 0.25 * L
    out.append({"kind": "cap", "label": 10, "a": P(0.02, 0.05, 0.02), "b": m, "ra": 0.085 * L, "rb": 0.055 * L})
    out.append({"kind": "cap", "label": 11, "a": m, "b": m + d2 * 0.2 * L, "ra": 0.054 * L, "rb": 0.047 * L})
    # the fingers
    for i in range(4):
        kx, ky = KNUCKLES[i]
        p = P(kx - 0.04, ky, 0.0)
        a = math.radians(spread[i])
        d = unit(math.cos(a) * u + math.sin(a) * t)
        nn = n.copy()
        r = FINGER_R[i] * L
        for j, (frac, bend) in enumerate(zip(PHALANX, curls[i])):
            b = math.radians(bend)
            d, nn = unit(math.cos(b) * d + math.sin(b) * nn), unit(-math.sin(b) * d + math.cos(b) * nn)
            q = p + d * (FINGER_LEN[i] * frac * L + (0.04 * L if j == 0 else 0.0))
            kb = 1.15 if (j == 0 and curls[i][0] > 60) else 1.0       # a curled finger's knuckle: a bump
            out.append({"kind": "cap", "label": 20 + 4 * i + j, "a": p, "b": q, "ra": r * (1 - 0.07 * j) * kb,
                        "rb": (r * (0.93 - 0.1 * j) if j < 2 else r * 0.86) * kb})
            p = q
    return out


class _Plane:
    """The plane facing the viewer through C: perspective projection of 3D points onto it, and back."""

    def __init__(self, eye, C):
        self.E = np.asarray(eye, float)
        self.C = np.asarray(C, float)
        self.v = unit(self.E - self.C)                       # toward the eye
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(up @ self.v)) > 0.95:
            up = np.array([0.0, 0.0, 1.0])
        self.a = unit(np.cross(up, self.v))
        self.b = unit(np.cross(self.v, self.a))
        self.d0 = float((self.E - self.C) @ self.v)

    def depth(self, X) -> np.ndarray:
        return (self.E - np.atleast_2d(X)) @ self.v

    def xy(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Plane coordinates (m at the plane) and the scale factor (plane size / true size)."""
        X = np.atleast_2d(np.asarray(X, float))
        d = self.depth(X)
        s = self.d0 / np.maximum(d, 1e-6)
        Y = self.E + (X - self.E) * s[:, None]
        q = Y - self.C
        return np.stack([q @ self.a, q @ self.b], 1), s

    def lift(self, xy, depth: Optional[float] = None) -> np.ndarray:
        """3D points on the plane (or on the parallel plane at ``depth`` that projects to the same picture)."""
        xy = np.asarray(xy, float)
        Y = self.C + np.outer(xy[:, 0], self.a) + np.outer(xy[:, 1], self.b)
        if depth is None:
            return Y
        return self.E + (Y - self.E) * (float(depth) / self.d0)


def _raster(pieces: List[dict], pl: _Plane, px: float, org: np.ndarray, shape: Tuple[int, int]):
    """Depth buffer (m) and label buffer of the pieces; px = grid pixels per plane metre."""
    H, Wd = shape
    zb = np.full(shape, np.inf)
    lb = np.full(shape, -1, int)
    yy, xx = np.mgrid[0:H, 0:Wd]
    G = np.stack([xx.ravel() / px + org[0], yy.ravel() / px + org[1]], 1)
    for pc in pieces:
        if pc["kind"] == "poly":
            q, _ = pl.xy(pc["pts"])
            g = ((q - org) * px).astype(np.float32)
            hull = cv2.convexHull(g)
            m = np.zeros(shape, np.uint8)
            cv2.fillConvexPoly(m, hull.astype(np.int32), 1)
            # depth: the plane through the mid layer of the palm (least squares)
            dz = pl.depth(pc["pts"])
            A = np.c_[q, np.ones(len(q))]
            co, *_ = np.linalg.lstsq(A, dz, rcond=None)
            z = (np.c_[G, np.ones(len(G))] @ co).reshape(shape)
            sel = (m > 0) & (z < zb)
        else:
            (qa, qb), s = pl.xy(np.stack([pc["a"], pc["b"]]))
            za, zb_ = pl.depth(np.stack([pc["a"], pc["b"]]))
            ra, rb = pc["ra"] * s[0], pc["rb"] * s[1]
            ab = qb - qa
            ll = float(ab @ ab)
            tt = np.clip(((G - qa) @ ab) / ll, 0, 1) if ll > 1e-12 else np.zeros(len(G))
            dist = np.linalg.norm(G - (qa + tt[:, None] * ab), axis=1)
            r = ra + (rb - ra) * tt
            inside = dist < r
            bulge = np.sqrt(np.clip(1 - (dist / np.maximum(r, 1e-9)) ** 2, 0, 1)) * r
            z = (za + (zb_ - za) * tt - bulge).reshape(shape)
            sel = inside.reshape(shape) & (z < zb)
        zb[sel] = z[sel]
        lb[sel] = pc["label"]
    return zb, lb


def _finger_of(lab: np.ndarray) -> np.ndarray:
    """Which digit a label belongs to: 0 palm, 1 thumb, 2.. index .. pinky."""
    f = np.where(lab >= 20, 2 + (lab - 20) // 4, np.where(lab >= 10, 1, 0))
    return np.where(lab < 0, -1, f)


def _edges(zb, lb, cover, thr: float, max_sep: bool, margin: int = 2) -> np.ndarray:
    E = np.zeros(lb.shape, bool)
    f = _finger_of(lb)
    for ax in (0, 1):
        a_l, b_l = lb, np.roll(lb, -1, ax)
        a_f, b_f = f, np.roll(f, -1, ax)
        a_z, b_z = zb, np.roll(zb, -1, ax)
        diff = (a_l != b_l) & (a_l >= 0) & (b_l >= 0)
        with np.errstate(invalid="ignore"):
            dz = np.abs(np.where(np.isfinite(a_z) & np.isfinite(b_z), a_z - b_z, 0))
        sep = (a_f != b_f) & (a_f >= 2) & (b_f >= 2)             # two neighbouring fingers
        thumb = ((a_f == 1) & (b_f >= 2)) | ((b_f == 1) & (a_f >= 2))
        same = a_f == b_f                                         # two phalanges of one finger
        e = diff & ((dz > np.where(same, 2.5 * thr, thr)) | (sep & max_sep) | thumb)
        # the front pixel carries the line
        front = np.where(a_z <= b_z, 0, 1)
        E |= e & (front == 0)
        E |= np.roll(e & (front == 1), 1, ax)
    inner = cv2.erode(cover.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    return E & inner


def _trace(mask: np.ndarray, min_len: float) -> List[np.ndarray]:
    """Polylines (grid px, x y) along the thin lines of a mask, longest first."""
    from skimage.morphology import skeletonize
    sk = skeletonize(cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    labels, n = ndi.label(sk, np.ones((3, 3)))
    out = []
    for c in range(1, n + 1):
        ys, xs = np.nonzero(labels == c)
        if len(xs) < min_len:
            continue
        pts = set(zip(ys.tolist(), xs.tolist()))
        ends = [p for p in pts if nb[p] == 1]
        cur = ends[0] if ends else next(iter(pts))
        path = [cur]
        pts.discard(cur)
        while True:
            y, x = cur
            nxt = None
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                if (y + dy, x + dx) in pts:
                    nxt = (y + dy, x + dx)
                    break
            if nxt is None:
                break
            pts.discard(nxt)
            path.append(nxt)
            cur = nxt
        if len(path) < min_len:
            continue
        P = np.array([(x, y) for y, x in path], float)
        P = np.stack([ndi.gaussian_filter1d(P[:, k], 2.0, mode="nearest") for k in range(2)], 1)
        out.append(P[::3] if len(P) > 9 else P)
    out.sort(key=lambda p: -len(p))
    return out


def hand_parts(rig, s: str, shape: str, mat: str, subj: str, eye, h: float, k: float, cam=None,
               grip_r: Optional[float] = None, prop_axis=None) -> List[dict]:
    """The drawn hand of side s as shape2d parts (one, or two for a grip around a prop)."""
    from .anime2d import _part
    W, u, t, n = F3.hand_frame(rig, s)
    W = np.asarray(W, float)
    L = 0.5 * h * min(k, 1.15) * 1.05              # wrist to fingertip about 0.75 of the face (chin to hairline)
    name, _ = shape_of(shape)
    eye = np.asarray(eye, float)
    shift = np.zeros(3)
    G = None
    if name == "grip" and grip_r:
        # the fingers wrap the shaft at the prop's grip point: the hand sits behind it by the shaft's radius
        G = W + u * 0.45 * rig.b.Lh
        R = float(grip_r) + FINGER_R[1] * L
        shift = G - n * R - (W + u * (0.48 * L))
    pieces = hand_pieces(W + shift, u, t, n, L, shape, grip_r)
    allp = np.vstack([np.vstack([pc["a"], pc["b"]]) if pc["kind"] == "cap" else pc["pts"] for pc in pieces])
    C = allp.mean(0)
    pl = _Plane(eye, C)
    q, _ = pl.xy(allp)
    lo, hi = q.min(0) - 0.2 * L, q.max(0) + 0.2 * L
    ext = float((hi - lo).max())
    px = GRID / ext
    shp = (int((hi[1] - lo[1]) * px) + 1, int((hi[0] - lo[0]) * px) + 1)
    zb, lb = _raster(pieces, pl, px, lo, shp)
    cover = lb >= 0
    # how big the hand is on the canvas: the gap that still reads as a gap
    canvas_px = None
    small = False
    if cam is not None:
        z = float(cam.project(np.asarray([W]))[1][0])
        if z > 1e-6:
            canvas_px = cam.f * L / z
            small = canvas_px < MITTEN_PX * max(cam.W, cam.H)
    grid_per_canvas = (GRID / (ext / L)) / canvas_px if canvas_px else 1.0
    close_r = int(max(1, round(1.6 * grid_per_canvas))) if small else int(max(1, round(0.8 * grid_per_canvas)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
    covm = cv2.morphologyEx(cover.astype(np.uint8), cv2.MORPH_CLOSE, ker)
    gL = GRID / (ext / L)                                   # grid px per hand length
    if name in ("relaxed", "flat", "reach", "cup"):
        # the anime grouping: middle, ring and little finger one tapered shape (the index stays apart)
        grp = np.isin(_finger_of(lb), [3, 4, 5]).astype(np.uint8)
        rg = int(max(2, round((0.11 if small else 0.07) * gL)))
        gk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rg + 1, 2 * rg + 1))
        gclosed = cv2.morphologyEx(grp, cv2.MORPH_CLOSE, gk)
        gap = (gclosed > 0) & ~cover                        # the wedges between the grouped fingers: finger lines
        covm = np.maximum(covm, gclosed)
    else:
        gap = np.zeros(cover.shape, bool)
    thr = 0.6 * FINGER_R[1] * L
    mg = int(max(2, round(0.025 * gL)))
    edges = _edges(zb, lb, covm > 0, thr, max_sep=True, margin=mg)
    edges |= gap & (cv2.erode(covm, np.ones((2 * mg + 1, 2 * mg + 1), np.uint8)) > 0)
    lines = _trace(edges, 0.16 * GRID / (ext / L))
    if name in ("fist", "point", "peace", "grip"):
        # the curled fingers' fold: a curve through the middle knuckles (the line every drawn fist has)
        curled = [i for i in range(4) if SHAPES[name]["curl"][i][0] > 45 or name == "grip"]
        pk = [pc["b"] for pc in pieces if pc["kind"] == "cap" and pc["label"] >= 20 and (pc["label"] - 20) % 4 == 0
              and (pc["label"] - 20) // 4 in curled]
        if len(pk) >= 2:
            g, _ = pl.xy(np.array(pk))
            g = (g - lo) * px
            inner = cv2.erode(covm, np.ones((5, 5), np.uint8)) > 0
            ok = [q for q in g if 0 <= int(q[1]) < inner.shape[0] and 0 <= int(q[0]) < inner.shape[1]
                  and inner[int(q[1]), int(q[0])]]
            if len(ok) >= 2 and np.linalg.norm(ok[-1] - ok[0]) > 0.12 * gL:
                # one smooth arc through the knuckles (a quadratic from the first to the last, bowed to the middle)
                a0, a1, cm = ok[0], ok[-1], np.mean(ok, 0)
                ctl = 2 * cm - 0.5 * (a0 + a1)
                tt = np.linspace(0, 1, 9)[:, None]
                arc = (1 - tt) ** 2 * a0 + 2 * (1 - tt) * tt * ctl + tt ** 2 * a1
                lines = [arc] + lines[:1]
    if not lines and name not in ("fist", "point", "grip"):
        # no edge to read (a hand seen from its side): the separation between the index and the middle finger
        def chain(i):
            cs = [pc for pc in pieces if pc["kind"] == "cap" and pc["label"] >= 20 and (pc["label"] - 20) // 4 == i]
            return np.array([cs[0]["a"]] + [c["b"] for c in cs])
        mid = 0.5 * (chain(0) + chain(1))
        mid = np.array([mid[0] + (mid[1] - mid[0]) * f for f in (0.3, 0.7)] + list(mid[1:]))
        g, _ = pl.xy(mid)
        g = (g - lo) * px
        inner = cv2.erode(covm, np.ones((3, 3), np.uint8)) > 0
        ok = [q for q in g if 0 <= int(q[1]) < inner.shape[0] and 0 <= int(q[0]) < inner.shape[1]
              and inner[int(q[1]), int(q[0])]]
        if len(ok) >= 2 and np.linalg.norm(ok[-1] - ok[0]) > 0.12 * gL:
            lines.append(np.array(ok))
    curled_shape = name in ("fist", "point", "peace", "grip")
    lines = lines[:(2 if curled_shape else 1)] if small else lines[:4]

    def outline(m):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
        c = np.stack([ndi.gaussian_filter1d(c[:, j], max(2.5, 0.022 * gL), mode="wrap") for j in range(2)], 1)
        step = max(1, len(c) // 120)
        return c[::step]

    def to_plane(P):
        return np.asarray(P, float) / px + lo

    d_near = float(pl.depth(allp).min())
    view = pl.v
    parts = []
    poly = outline(covm)
    if poly is None:
        return []
    palm_faces = float(n @ unit(eye - W)) > 0.15
    joints = {f"wrist-{s}": W}
    if G is not None and palm_faces:
        dG = float(pl.depth(G[None])[0])
        fing = np.isin(_finger_of(lb), [2, 3, 4, 5])
        fm = cv2.morphologyEx(fing.astype(np.uint8), cv2.MORPH_CLOSE, ker)
        fpoly = outline(fm)
        back_d = dG + float(grip_r) + 0.004 * h
        part = _part(f"hand-{s}", "blob", mat, subj, f"hand-{s}", joints, rad=0.12 * h,
                     flat3=pl.lift(to_plane(poly), back_d), normal=view, bias=-0.004 * h, mitten=small)
        parts.append(part)
        if fpoly is not None:
            front_d = dG - 1.5 * float(grip_r) - 0.02 * h          # over the hilt or handle (often thicker)
            fpart = _part(f"fingers-{s}", "blob", mat, subj, f"hand-{s}", {f"fingers-{s}": None}, rad=0.08 * h,
                          flat3=pl.lift(to_plane(fpoly), front_d), normal=view, bias=-0.004 * h, mitten=small)
            # the separations between the wrapped fingers lie on the front part
            for ln in lines:
                mid = ln[len(ln) // 2].astype(int)
                if fing[min(mid[1], fing.shape[0] - 1), min(mid[0], fing.shape[1] - 1)]:
                    fpart["details"].append({"pts": pl.lift(to_plane(ln), front_d - 0.001 * h), "w": 0.4})
            parts.append(fpart)
        return parts
    # in front of the wrist's end of the forearm, so the hand is never cut by the cloth it hangs beside
    depth = min(d_near, float(pl.depth(W[None])[0]) - 0.03 * h) - 0.004 * h
    if G is not None:
        depth = min(depth, float(pl.depth(G[None])[0]) - 1.5 * float(grip_r) - 0.02 * h)
    part = _part(f"hand-{s}", "blob", mat, subj, f"hand-{s}", joints, rad=0.12 * h,
                 flat3=pl.lift(to_plane(poly), depth), normal=view, bias=-0.004 * h, mitten=small)
    for ln in lines:
        part["details"].append({"pts": pl.lift(to_plane(ln), depth - 0.001 * h), "w": 0.4})
    parts.append(part)
    return parts
