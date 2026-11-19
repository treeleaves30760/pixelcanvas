"""2D shapes over a 3D pose: outlines of tubes (limbs, a torso, locks, sleeves, a skirt), blobs (a head, a palm, a
hair cap) and flat faces (a wall, a floor), a z-buffer of those shapes (which part is in front where, a normal per
pixel for the cel shadows) and the visible stretches of every outline as strokes.

Pure functions: same inputs, same outputs; nothing touches the disk.

A *part* is a dict:
  name, kind ("tube" | "blob" | "plane"), mat, subj, group, joints {name: 3D point or None}, rad (m)
  tube:  rings [(centre (3,), points (N,3))], open_start / open_end (an open hem instead of a closed end)
  blob:  pts (M,3), nrm (M,3) (surface samples); hull=True draws the convex hull of the front samples
  plane: pts (K,3) polygon, normal (3,)
  details: [{"pts": (n,3), "w": weight}]   lines on the part (folds, pleats, strands)
  bias: metres added to the depth (negative = in front), line: draw its outline (default True)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def perp_basis(u) -> Tuple[np.ndarray, np.ndarray]:
    u = unit(u)
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = unit(np.cross(u, ref))
    z = np.cross(x, u)
    return x, z


def ring(c, X, Z, a: float, b: float, n: int = 20) -> np.ndarray:
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.asarray(c, float) + np.outer(a * np.cos(th), X) + np.outer(b * np.sin(th), Z)


def _catmull(P: np.ndarray, n_out: int) -> np.ndarray:
    """Centripetal-ish Catmull-Rom through P (n,d), resampled to n_out points by arc length."""
    P = np.asarray(P, float)
    if len(P) < 3:
        t = np.linspace(0, 1, n_out)[:, None]
        return P[0] + t * (P[-1] - P[0])
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.linspace(0, 1, 12, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    out = np.asarray(out)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))])
    if s[-1] <= 1e-12:
        return np.repeat(P[:1], n_out, 0)
    ts = np.linspace(0, s[-1], n_out)
    return np.stack([np.interp(ts, s, out[:, k]) for k in range(out.shape[1])], 1)


def chain_tube(axis_pts, radii, n_rings: int = 14, n_pts: int = 18, round_start: bool = True,
               round_end: bool = True, point_end: bool = False, frames=None, depth_ratio=None) -> List[tuple]:
    """Rings along a 3D polyline (smoothed) with radii interpolated along it; round ends get a hemisphere of
    shrinking rings, a point end closes to a tip. ``frames`` [(X, Z)] per axis point give elliptic rings with
    ``depth_ratio`` b/a."""
    A = np.asarray(axis_pts, float)
    r = np.asarray(radii, float)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))])
    L = max(float(s[-1]), 1e-9)
    C = _catmull(A, n_rings) if len(A) > 2 else A[0] + np.linspace(0, 1, n_rings)[:, None] * (A[-1] - A[0])
    sc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    sc = sc / max(sc[-1], 1e-9) * L
    R = np.interp(sc, s, r)
    dr = np.interp(sc, s, np.asarray(depth_ratio, float)) if depth_ratio is not None else np.ones(len(C))
    T = np.gradient(C, axis=0)
    T = unit(T)
    rings = []
    fr = None
    if frames is not None:
        FX = np.stack([np.interp(sc, s, np.asarray([f[0][k] for f in frames])) for k in range(3)], 1)
        fr = FX
    for i in range(len(C)):
        if fr is not None:
            X = unit(fr[i] - T[i] * (fr[i] @ T[i]))
            Z = np.cross(X, T[i])
        else:
            X, Z = perp_basis(T[i])
        rings.append((C[i], ring(C[i], X, Z, R[i], R[i] * dr[i], n_pts), R[i]))
    out = [(c, P) for c, P, _ in rings]

    def cap(i, sign):
        c, _, rr = rings[i]
        t = T[i] * sign
        if fr is not None:
            X = unit(fr[i] - T[i] * (fr[i] @ T[i]))
            Z = np.cross(X, T[i])
        else:
            X, Z = perp_basis(T[i])
        res = []
        for ph in (35, 60, 80, 88):
            a = math.radians(ph)
            cc = c + t * rr * math.sin(a) * (0.9 if fr is None else 0.6)
            res.append((cc, ring(cc, X, Z, rr * math.cos(a), rr * math.cos(a) * dr[i], n_pts)))
        return res
    if round_start:
        out = list(reversed(cap(0, -1.0))) + out
    if point_end:
        c, _, rr = rings[-1]
        X, Z = perp_basis(T[-1])
        tip = c + T[-1] * rr * 1.6
        for f in (0.6, 0.3, 0.06):
            cc = c + (tip - c) * (1 - f)
            out.append((cc, ring(cc, X, Z, rr * f, rr * f, n_pts)))
    elif round_end:
        out = out + cap(len(rings) - 1, 1.0)
    return out


def sphere_samples(n: int = 400) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


def ell_samples(c, A, n: int = 400, keep=None) -> Tuple[np.ndarray, np.ndarray]:
    """Surface samples and normals of the ellipsoid c + A q (|q| = 1); ``keep(q)`` -> bool mask."""
    q = sphere_samples(n)
    if keep is not None:
        q = q[keep(q)]
    A = np.asarray(A, float)
    P = np.asarray(c, float) + q @ A.T
    N = unit(q @ np.linalg.inv(A))
    return P, N


# ============================================================================================ outlines (canvas px)

def _hull(P2: np.ndarray) -> np.ndarray:
    if len(P2) < 3:
        return P2
    h = cv2.convexHull(P2.astype(np.float32)).reshape(-1, 2).astype(float)
    return h


def _arc_between(P: np.ndarray, c: np.ndarray, iL: int, iR: int, outward: np.ndarray) -> np.ndarray:
    """The arc of the closed ring polygon P (projected, sorted by angle) from index iL to iR on the outward side."""
    n = len(P)
    a1 = [P[(iL + k) % n] for k in range(((iR - iL) % n) + 1)]
    a2 = [P[(iL - k) % n] for k in range(((iL - iR) % n) + 1)]
    m1 = float(np.mean([(q - c) @ outward for q in a1]))
    m2 = float(np.mean([(q - c) @ outward for q in a2]))
    return np.asarray(a1 if m1 >= m2 else a2)


def _sorted_ring(P2: np.ndarray, c: np.ndarray) -> np.ndarray:
    ang = np.arctan2(P2[:, 1] - c[1], P2[:, 0] - c[0])
    return P2[np.argsort(ang)]


def tube_outline(cam, part: dict) -> Optional[dict]:
    """Outline polygon (canvas px) of a tube part, its projected centre line with the half width (px) and the depth
    of its front surface along it, and the hem (an open end) as a polyline."""
    rings = part["rings"]
    C3 = np.array([c for c, _ in rings])
    C2, zc = cam.project(C3)
    if (zc <= 0.05).any():
        return None
    P2s, Pzs = [], []
    for _, P in rings:
        q, z = cam.project(P)
        P2s.append(q)
        Pzs.append(z)
    T = np.gradient(C2, axis=0) if len(C2) > 1 else np.array([[1.0, 0.0]])
    nT = np.linalg.norm(T, axis=1, keepdims=True)
    for i in range(len(T)):
        if nT[i, 0] < 1e-6:
            j = i - 1 if i > 0 else min(i + 1, len(T) - 1)
            T[i] = T[j] if nT[j, 0] >= 1e-6 else np.array([1.0, 0.0])
    T = unit(T)
    Nn = np.stack([-T[:, 1], T[:, 0]], 1)
    left, right, half, front = [], [], [], []
    for i, (q, z) in enumerate(zip(P2s, Pzs)):
        d = (q - C2[i]) @ Nn[i]
        left.append(q[int(np.argmax(d))])
        right.append(q[int(np.argmin(d))])
        half.append(0.5 * float(d.max() - d.min()))
        front.append(float(zc[i] - z.min()))
    left, right = np.asarray(left), np.asarray(right)
    L2 = float(np.linalg.norm(np.diff(C2, axis=0), axis=1).sum())
    hem = None
    if L2 < 0.9 * max(half) * 2 or len(C2) < 2:
        poly = _hull(np.vstack(P2s))
    else:
        parts = [left]
        if part.get("open_end"):
            q = _sorted_ring(P2s[-1], C2[-1])
            d = (q - C2[-1]) @ Nn[-1]
            arc = _arc_between(q, C2[-1], int(np.argmax(d)), int(np.argmin(d)), T[-1])
            parts.append(arc)
            hem = arc
        parts.append(right[::-1])
        if part.get("open_start"):
            q = _sorted_ring(P2s[0], C2[0])
            d = (q - C2[0]) @ Nn[0]
            parts.append(_arc_between(q, C2[0], int(np.argmin(d)), int(np.argmax(d)), -T[0]))
        poly = np.vstack(parts)
    return {"poly": poly, "c2": C2, "zc": zc, "half": np.asarray(half), "front": np.asarray(front), "hem": hem}


def front_samples(cam, P, N, slack: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(px, z, normal in screen space: x right, y down, z toward the viewer) of the samples facing the camera."""
    P = np.asarray(P, float)
    N = np.asarray(N, float)
    toward = unit(cam.C - P)
    ok = (N * toward).sum(1) > -slack
    px, z = cam.project(P[ok])
    Nc = N[ok]
    ns = np.stack([Nc @ cam.right, -(Nc @ cam.up), -(Nc @ cam.fwd)], 1)
    return px, z, unit(ns)


def clip_near(cam, P: np.ndarray, near: float = 0.15) -> np.ndarray:
    """A closed 3D polygon clipped to the part in front of the camera (z > near)."""
    P = np.asarray(P, float)
    z = cam.to_cam(P)[:, 2]
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        za, zb = z[i], z[(i + 1) % n]
        if za > near:
            out.append(a)
        if (za > near) != (zb > near):
            t = (near - za) / (zb - za)
            out.append(a + t * (b - a))
    return np.asarray(out) if out else np.zeros((0, 3))


# ============================================================================================ the z-buffer

class Raster:
    """Label, depth (m) and screen normal per pixel at ``s`` label px per canvas px."""

    def __init__(self, W: int, H: int, s: float):
        self.W, self.H, self.s = int(W), int(H), float(s)
        self.lab = np.full((H, W), -1, np.int32)
        self.z = np.full((H, W), np.inf, np.float32)
        self.nrm = np.zeros((H, W, 3), np.float32)
        self.nrm[..., 2] = 1.0

    def _box(self, poly_l: np.ndarray, pad: int = 2):
        x0 = max(0, int(math.floor(poly_l[:, 0].min())) - pad)
        y0 = max(0, int(math.floor(poly_l[:, 1].min())) - pad)
        x1 = min(self.W, int(math.ceil(poly_l[:, 0].max())) + pad + 1)
        y1 = min(self.H, int(math.ceil(poly_l[:, 1].max())) + pad + 1)
        return x0, y0, x1, y1

    def mask(self, poly_px: np.ndarray):
        P = np.asarray(poly_px, float) * self.s
        if len(P) < 3:
            return None
        x0, y0, x1, y1 = self._box(P)
        if x1 <= x0 or y1 <= y0:
            return None
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [np.round((P - [x0, y0]) * 4).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=2)
        return (x0, y0, x1, y1), m.astype(bool)

    def _commit(self, k: int, box, m, z, nrm, bias: float = 0.0):
        x0, y0, x1, y1 = box
        z = z + bias
        cur = self.z[y0:y1, x0:x1]
        win = m & (z < cur)
        cur[win] = z[win]
        self.lab[y0:y1, x0:x1][win] = k
        self.nrm[y0:y1, x0:x1][win] = nrm[win]
        return int(win.sum())

    def add_tube(self, k: int, out: dict, bias: float = 0.0) -> int:
        mm = self.mask(out["poly"])
        if mm is None:
            return 0
        box, m = mm
        x0, y0, x1, y1 = box
        C = out["c2"] * self.s
        # the centre line, densely: depth and half width along it
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1) if len(C) > 1 else np.zeros(0)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(s[-1]) + 2)
        ts = np.linspace(0, s[-1], n)
        cx = np.interp(ts, s, C[:, 0])
        cy = np.interp(ts, s, C[:, 1])
        zc = np.interp(ts, s, out["zc"])
        hf = np.maximum(np.interp(ts, s, out["half"] * self.s), 0.5)
        fr = np.interp(ts, s, out["front"])
        idx = np.full(m.shape, -1, np.int32)
        xi = np.clip(np.round(cx - x0).astype(int), 0, m.shape[1] - 1)
        yi = np.clip(np.round(cy - y0).astype(int), 0, m.shape[0] - 1)
        idx[yi, xi] = np.arange(n)
        has = idx >= 0
        _, (iy, ix) = ndi.distance_transform_edt(~has, return_indices=True)
        j = idx[iy, ix]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        ox, oy = gx - cx[j], gy - cy[j]
        # across the tube: the offset over the half width (the centre line's normal is the offset's direction)
        dist = np.hypot(ox, oy)
        u = np.clip(dist / hf[j], 0, 1)
        h = np.sqrt(np.maximum(0.0, 1 - u * u))
        z = zc[j] - fr[j] * h
        dn = np.where(dist[..., None] > 1e-6, np.stack([ox, oy], -1) / np.maximum(dist, 1e-6)[..., None], 0.0)
        nrm = np.concatenate([dn * u[..., None], h[..., None]], -1).astype(np.float32)
        return self._commit(k, box, m, z.astype(np.float32), nrm, bias)

    def add_blob(self, k: int, poly_px: np.ndarray, px: np.ndarray, z: np.ndarray, nrm: np.ndarray,
                 bias: float = 0.0, reach: Optional[float] = None) -> int:
        mm = self.mask(poly_px)
        if mm is None or len(px) == 0:
            return 0
        box, m = mm
        x0, y0, x1, y1 = box
        P = px * self.s
        xi = np.round(P[:, 0] - x0).astype(int)
        yi = np.round(P[:, 1] - y0).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < m.shape[1]) & (yi < m.shape[0])
        if not ok.any():
            return 0
        xi, yi, zz, nn = xi[ok], yi[ok], np.asarray(z)[ok], np.asarray(nrm)[ok]
        order = np.argsort(-zz)                   # nearest last: it wins the pixel
        zb = np.full(m.shape, np.inf, np.float32)
        ib = np.full(m.shape, -1, np.int32)
        zb[yi[order], xi[order]] = zz[order]
        ib[yi[order], xi[order]] = order
        has = ib >= 0
        dist, (iy, ix) = ndi.distance_transform_edt(~has, return_indices=True)
        if reach is not None:
            m = m & (dist <= reach * self.s)
        j = ib[iy, ix]
        return self._commit(k, box, m, zb[iy, ix], nn[j].astype(np.float32), bias)

    def hits(self, cam, prims: List[dict], box_px) -> Optional[tuple]:
        """Exact ray hits of 3D primitives (round cones, ellipsoids, cylinders) over a canvas-px box: (label box,
        hit mask, depth, screen normal)."""
        from .render3d import intersect
        x0, y0, x1, y1 = box_px
        X0 = max(0, int(math.floor(x0 * self.s)) - 2)
        Y0 = max(0, int(math.floor(y0 * self.s)) - 2)
        X1 = min(self.W, int(math.ceil(x1 * self.s)) + 3)
        Y1 = min(self.H, int(math.ceil(y1 * self.s)) + 3)
        if X1 <= X0 or Y1 <= Y0:
            return None
        gy, gx = np.mgrid[Y0:Y1, X0:X1]
        D = cam.rays((gx.ravel() + 0.5) / self.s, (gy.ravel() + 0.5) / self.s)
        t = np.full(len(D), np.inf)
        nw = np.zeros((len(D), 3))
        for pr in prims:
            tt, nn, _ = intersect(pr, cam.C, D)
            win = tt < t
            t[win] = tt[win]
            nw[win] = nn[win]
        shape = gy.shape
        m = np.isfinite(t).reshape(shape)
        if not m.any():
            return None
        ns = np.stack([nw @ cam.right, -(nw @ cam.up), -(nw @ cam.fwd)], 1)
        return (X0, Y0, X1, Y1), m, t.reshape(shape).astype(np.float32), unit(ns).reshape(shape + (3,)).astype(np.float32)

    def add_hits(self, k: int, hit: tuple, bias: float = 0.0) -> int:
        box, m, z, nrm = hit
        return self._commit(k, box, m, np.where(m, z, np.inf).astype(np.float32), nrm, bias)

    def add_decal(self, k: int, poly_px: np.ndarray, over: List[int]) -> int:
        """A flat mark on other parts (an open V, a collar): it takes the pixels of its polygon where one of
        ``over`` is in front, and keeps their depth."""
        mm = self.mask(poly_px)
        if mm is None:
            return 0
        (x0, y0, x1, y1), m = mm
        sub = self.lab[y0:y1, x0:x1]
        win = m & np.isin(sub, over)
        sub[win] = k
        return int(win.sum())

    def add_plane(self, k: int, poly_px: np.ndarray, cam, P0: np.ndarray, n3: np.ndarray, bias: float = 0.0) -> int:
        mm = self.mask(poly_px)
        if mm is None:
            return 0
        box, m = mm
        x0, y0, x1, y1 = box
        gy, gx = np.mgrid[y0:y1, x0:x1]
        D = cam.rays((gx + 0.5) / self.s, (gy + 0.5) / self.s)
        den = D @ n3
        t = ((np.asarray(P0) - cam.C) @ n3) / np.where(np.abs(den) < 1e-9, 1e-9, den)
        z = t.astype(np.float32)                  # rays have camera z = 1: t is the depth
        z = np.where(t > 0, z, np.inf).astype(np.float32)
        ns = np.array([n3 @ cam.right, -(n3 @ cam.up), -(n3 @ cam.fwd)], np.float32)
        nrm = np.broadcast_to(unit(ns), m.shape + (3,)).astype(np.float32)
        return self._commit(k, box, m, z, nrm, bias)

    def at(self, pts_px: np.ndarray) -> np.ndarray:
        P = np.asarray(pts_px, float) * self.s
        xi = np.round(P[:, 0]).astype(int)
        yi = np.round(P[:, 1]).astype(int)
        ok = (xi >= 0) & (yi >= 0) & (xi < self.W) & (yi < self.H)
        out = np.full(len(P), -1, np.int32)
        out[ok] = self.lab[yi[ok], xi[ok]]
        return out

    def depth_at(self, pts_px: np.ndarray) -> np.ndarray:
        P = np.asarray(pts_px, float) * self.s
        xi = np.clip(np.round(P[:, 0]).astype(int), 0, self.W - 1)
        yi = np.clip(np.round(P[:, 1]).astype(int), 0, self.H - 1)
        return self.z[yi, xi]


# ============================================================================================ strokes

def resample(P: np.ndarray, step: float, closed: bool = False) -> np.ndarray:
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return P[:1]
    n = max(2, int(s[-1] / step) + 1)
    ts = np.linspace(0, s[-1], n)
    Q = np.stack([np.interp(ts, s, P[:, 0]), np.interp(ts, s, P[:, 1])], 1)
    return Q[:-1] if closed else Q


def smooth_closed(P: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(P) < 5:
        return P
    return np.stack([ndi.gaussian_filter1d(P[:, k], sigma, mode="wrap") for k in range(2)], 1)


def runs(flags: np.ndarray, closed: bool) -> List[np.ndarray]:
    """Index runs where flags is true (a closed loop's run may wrap)."""
    n = len(flags)
    if n == 0 or not flags.any():
        return []
    if flags.all():
        return [np.arange(n)]
    idx = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    out = [np.arange(a, b) for a, b in zip(idx[::2], idx[1::2])]
    if closed and len(out) > 1 and out[0][0] == 0 and out[-1][-1] == n - 1:
        out = [np.concatenate([out[-1], out[0]])] + out[1:-1]
    return out


def outline_strokes(poly: np.ndarray, k: int, R: Raster, keep_fn, step: float, min_len: float,
                    sigma: float = 1.5) -> List[dict]:
    """The visible stretches of part ``k``'s outline: a point is drawn where the part is in front just inside its
    outline and ``keep_fn(q, P, z_in, z_out)`` accepts the label q just outside (-1 for nothing) with a class. Returns [{pts, cls}]
    with cls the label class from keep_fn (strokes split where it changes)."""
    P = resample(poly, step, closed=True)
    if len(P) < 6:
        return []
    P = smooth_closed(P, sigma)
    T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    T = unit(T)
    Nn = np.stack([T[:, 1], -T[:, 0]], 1)
    area = 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    if area < 0:
        Nn = -Nn                                    # outward
    off = 1.6 / R.s
    Pin, Pout = P - Nn * off, P + Nn * off
    inside = R.at(Pin)
    outside = R.at(Pout)
    zin, zout = R.depth_at(Pin), R.depth_at(Pout)
    cls = np.array([keep_fn(int(q), P[i], float(zin[i]), float(zout[i])) if inside[i] == k else 0
                    for i, q in enumerate(outside)], np.int8)
    out = []
    n = len(P)
    for c in (1, 2, 3):
        for r in runs(cls == c, closed=True):
            pts = P[r]
            if len(pts) * step < min_len:
                continue
            closed = len(r) == n
            out.append({"pts": pts, "cls": c, "closed": closed})
    return out


def detail_strokes(pts: np.ndarray, k: int, R: Raster, step: float, min_len: float) -> List[np.ndarray]:
    """Visible stretches of a line drawn on part ``k``."""
    P = resample(pts, step)
    if len(P) < 2:
        return []
    vis = R.at(P) == k
    return [P[r] for r in runs(vis, closed=False) if len(r) * step >= min_len]


def simplify(P: np.ndarray, eps: float, closed: bool = False) -> np.ndarray:
    """Few control points along a dense path (Douglas-Peucker)."""
    Q = np.asarray(P, np.float32).reshape(-1, 1, 2)
    out = cv2.approxPolyDP(Q, float(eps), closed).reshape(-1, 2).astype(float)
    if not closed and (len(out) < 2 or np.linalg.norm(out[-1] - P[-1]) > 1e-6):
        out = np.vstack([out, P[-1:]]) if len(out) else np.asarray(P)[[0, -1]]
    return out


def mask_outline(box, m: np.ndarray, s: float, sigma: float = 1.2) -> Optional[np.ndarray]:
    """The smooth outer outline (canvas px) of a label-res mask placed at ``box``."""
    u = np.pad(m.astype(np.uint8), 1)
    cs, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) - 1.0
    if len(c) < 6:
        return None
    c = smooth_closed(c, sigma)
    return (c + [box[0] + 0.5, box[1] + 0.5]) / s


def clean_shape(m: np.ndarray, r: float, min_area: float) -> np.ndarray:
    """A shadow mask as a cel shape: closed, opened, smoothed by ``r`` px, specks under ``min_area`` px gone."""
    if not m.any():
        return m
    k = max(1, int(round(r)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    u = m.astype(np.uint8)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, ker)
    u = cv2.morphologyEx(u, cv2.MORPH_OPEN, ker)
    f = cv2.GaussianBlur(u.astype(np.float32), (0, 0), max(0.8, r * 0.8)) > 0.5
    lab, n = ndi.label(f)
    if n:
        sizes = ndi.sum(f, lab, range(1, n + 1))
        f = np.isin(lab, np.flatnonzero(sizes >= min_area) + 1)
    return f
