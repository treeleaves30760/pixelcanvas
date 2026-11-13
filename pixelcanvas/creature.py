# -*- coding: utf-8 -*-
"""Animals in one call: rigged quadrupeds, birds, fish and sea life, insects and small creatures.

    from pixelcanvas.creature import Creature, creature, flock, school

    Creature("fox", size=0.3).pose("trot", phase=0.2).draw(cv, at=(0.5, 0.8))
    creature(cv, "horse", at=(0.3, 0.9), size=0.4, pose="gallop", phase=0.6)

Every animal is a small 3D model made of smooth blended volumes (ellipsoids and tapered capsules on a skeleton).
It is posed by its rig (joints bend the right way: elbows back, stifles forward, hocks back), projected for the
requested view (side, three-quarter, front), shaded with the scene light of ``pixelcanvas.form`` and finished with
fur at the silhouette, markings that wrap around the body (stripes, spots, patches, points), and a face with life
(eyes with highlights, nose, mouth, whiskers).

Units: ``at`` is a canvas fraction ``(u, v)`` — the ground point under the animal (centre of the body for fish and
flying birds); ``size`` is a fraction of the canvas height — the standing height for walkers and birds, the body
length for fish, snakes, lizards and other long low creatures. At a ``form`` Spot (``ground.place``) the size is in
world units and scaled by perspective. Masks, boxes and anchors come back in canvas pixels.
"""
from __future__ import annotations

import math

import numpy as np

from .color import F32, rgb, as_color, smoothstep
from . import sdf as _sdf
from . import filters as _f
from . import form as _form
from ._args import (reject_unknown as _reject_unknown, unknown_name as _unknown_name,
                    signature_options as _sig_options, strict as _strict, strict_methods as _strict_methods)

__all__ = ["Creature", "creature", "flock", "school", "species", "SPECIES", "POSES", "COATS"]

_BIG = 1e5

# ===================================================================== small math


def _v3(x, y=0.0, z=0.0):
    return np.array([x, y, z], np.float64)


def _unit(v):
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0


def _rot2(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _dir2(deg):
    """Unit 2D direction for an angle in degrees (0 = forward +x, 90 = up +y)."""
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def _rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def _ry(deg):
    """Yaw: positive turns +x (forward) towards +z (the viewer)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], np.float64)


def _rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def _lerp(a, b, t):
    return a + (b - a) * t


def _smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _smax(a, b, k):
    return -_smin(-a, -b, k)


def _ik2(root, target, l1, l2, side):
    """Two-link IK in 2D: joint between root and target, bent towards ``side`` (+1 = forward, -1 = back)."""
    root = np.asarray(root, np.float64)
    target = np.asarray(target, np.float64)
    dv = target - root
    c = float(np.linalg.norm(dv))
    c = min(max(c, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    u = dv / (np.linalg.norm(dv) + 1e-12)
    cos_a = (l1 * l1 + c * c - l2 * l2) / (2 * l1 * c)
    a = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
    # side is measured against u rotated +90 deg: for a leg pointing down that is +x (forward)
    cand1 = root + _rot2(u, a) * l1
    cand2 = root + _rot2(u, -a) * l1
    perp = np.array([-u[1], u[0]])
    pick = cand1 if np.dot(cand1 - root, perp) * side > 0 else cand2
    return pick, root + u * c


# ===================================================================== noise (3D, numpy)


def _hash3(ix, iy, iz, seed):
    with np.errstate(over="ignore"):
        h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (iz.astype(np.int64) * 83492791)
        h = (h ^ (seed * 2654435761)) & 0xFFFFFFFF
        h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
        h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(F32) / F32(16777216.0)


def _vnoise3(x, y, z, seed=0):
    """Value noise in 3D (lattice units), about 0~1."""
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - xi, y - yi, z - zi
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    fz = fz * fz * (3 - 2 * fz)
    xi, yi, zi = xi.astype(np.int64), yi.astype(np.int64), zi.astype(np.int64)
    out = 0
    for dx in (0, 1):
        wx = fx if dx else 1 - fx
        for dy in (0, 1):
            wy = fy if dy else 1 - fy
            for dz in (0, 1):
                wz = fz if dz else 1 - fz
                out = out + _hash3(xi + dx, yi + dy, zi + dz, seed) * (wx * wy * wz)
    return np.asarray(out, F32)


def _fbm3(x, y, z, octaves=3, seed=0):
    tot, amp, norm = 0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + _vnoise3(x, y, z, seed + o * 31) * amp
        norm += amp
        x, y, z = x * 2.03, y * 2.03, z * 2.03
        amp *= 0.5
    return tot / norm


def _worley3(x, y, z, seed=0, jitter=0.9):
    """3D cellular noise: (f1, f2) distances in lattice units."""
    xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
    f1 = np.full(np.shape(x), 9.0, F32)
    f2 = np.full(np.shape(x), 9.0, F32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx, cy, cz = xi + dx, yi + dy, zi + dz
                px_ = cx + 0.5 + jitter * (_hash3(cx, cy, cz, seed) - 0.5)
                py_ = cy + 0.5 + jitter * (_hash3(cx, cy, cz, seed + 7) - 0.5)
                pz_ = cz + 0.5 + jitter * (_hash3(cx, cy, cz, seed + 13) - 0.5)
                d = np.sqrt((x - px_) ** 2 + (y - py_) ** 2 + (z - pz_) ** 2).astype(F32)
                f2 = np.where(d < f1, f1, np.minimum(f2, d))
                f1 = np.minimum(f1, d)
    return f1, f2


def _vnoise1(t, seed=0):
    i = np.floor(t)
    f = t - i
    f = f * f * (3 - 2 * f)
    i = i.astype(np.int64)
    z = np.zeros_like(i)
    a = _hash3(i, z, z, seed)
    b = _hash3(i + 1, z, z, seed)
    return (a + (b - a) * f).astype(F32)


# ===================================================================== primitives

class _Prim:
    """One volume of a model. kind: 'ell' (centre + 3 semi-axis vectors), 'cone' (a, b, ra, rb: tapered capsule),
    'flat' (planar polygon with a rounded bulge)."""

    __slots__ = ("kind", "group", "geom", "col", "stops", "fur", "pat", "belly", "spec", "tone", "pt", "k",
                 "axis", "name", "bulge", "rough", "leg", "opa")

    def __init__(self, kind, group, geom, col, name="", fur=1.0, pat=1.0, belly=0.0, spec=0.0, tone=1.0, pt=0.0,
                 k=None, axis=None, stops=None, bulge=0.5, rough=0.0, leg=0.0, opa=1.0):
        self.kind, self.group, self.geom = kind, group, geom
        self.col = col
        self.stops = stops            # [(t, colour or None, pt)] along a cone (t 0 at a, 1 at b)
        self.fur, self.pat, self.belly, self.spec, self.tone, self.pt = fur, pat, belly, spec, tone, pt
        self.k = k
        self.axis = axis
        self.name = name
        self.bulge = bulge
        self.rough = rough
        self.leg = leg
        self.opa = opa


class _Decal:
    """A face feature drawn after the volumes: kind 'eye' | 'nose' | 'line' | 'spot'."""

    __slots__ = ("kind", "p", "n", "fwd", "size", "opt")

    def __init__(self, kind, p, n=None, fwd=None, size=0.05, **opt):
        self.kind, self.p, self.n, self.fwd, self.size, self.opt = kind, p, n, fwd, size, opt


class _Model:
    def __init__(self):
        self.prims = []
        self.groups = {"core": dict(parent=None, k=0.0)}
        self.decals = []
        self.anchors = {}
        self.coat = {}
        self.fur_len = 0.012
        self.fluff = 0.0
        self.shadow_pts = []          # world points touching the ground
        self.grounded = True
        self.clip_y = None
        self.smooth = 1.0             # how much creases between volumes melt (0 keeps every blend seam)

    def group(self, name, parent="core", k=0.02):
        if name not in self.groups:
            self.groups[name] = dict(parent=parent, k=k)
        return name

    def ell(self, group, c, radii, col, rot=None, **kw):
        """Ellipsoid: centre, radii (rx, ry, rz) along the axes of ``rot`` (3x3, columns = local axes)."""
        R = np.eye(3) if rot is None else np.asarray(rot, np.float64)
        A = R * np.asarray(radii, np.float64)[None, :]
        kw.setdefault("axis", R[:, 0])
        p = _Prim("ell", group, (np.asarray(c, np.float64), A), col, **kw)
        self.prims.append(p)
        return p

    def cone(self, group, a, b, ra, rb, col, **kw):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        kw.setdefault("axis", _unit(b - a))
        p = _Prim("cone", group, (a, b, float(ra), float(rb)), col, **kw)
        self.prims.append(p)
        return p

    def chain(self, group, pts, radii, col, **kw):
        """Tapered tube through points (a tail, trunk, neck, horn): one cone per segment."""
        out = []
        n = len(pts)
        stops = kw.pop("stops", None)
        for i in range(n - 1):
            st = None
            if stops is not None:
                t0, t1 = i / (n - 1), (i + 1) / (n - 1)
                st = [(0.0, *_stop_at(stops, t0)), (1.0, *_stop_at(stops, t1))]
            out.append(self.cone(group, pts[i], pts[i + 1], radii[i], radii[i + 1], col, stops=st, **kw))
        return out

    def flat(self, group, pts, col, bulge=0.5, smooth=0, **kw):
        pts = np.asarray(pts, np.float64)
        for _ in range(int(smooth)):
            # Chaikin corner cutting: rounder outlines for fins, leaves and petals
            q = np.roll(pts, -1, 0)
            pts = np.stack([pts * 0.75 + q * 0.25, pts * 0.25 + q * 0.75], 1).reshape(-1, 3)
        p = _Prim("flat", group, pts, col, bulge=bulge, **kw)
        self.prims.append(p)
        return p

    def decal(self, kind, p, n=None, fwd=None, size=0.05, **opt):
        d = _Decal(kind, np.asarray(p, np.float64), None if n is None else _unit(n),
                   None if fwd is None else _unit(fwd), size, **opt)
        self.decals.append(d)
        return d


def _stop_at(stops, t):
    """Interpolate (colour, pt) at t from stops [(t, colour, pt)]."""
    ts = [s[0] for s in stops]
    if t <= ts[0]:
        return stops[0][1], stops[0][2]
    if t >= ts[-1]:
        return stops[-1][1], stops[-1][2]
    for i in range(len(stops) - 1):
        if stops[i][0] <= t <= stops[i + 1][0]:
            u = (t - stops[i][0]) / max(stops[i + 1][0] - stops[i][0], 1e-9)
            c0, c1 = rgb(stops[i][1]), rgb(stops[i + 1][1])
            return tuple(c0 + (c1 - c0) * u), stops[i][2] + (stops[i + 1][2] - stops[i][2]) * u
    return stops[-1][1], stops[-1][2]


# ===================================================================== camera


class _Camera:
    """Orthographic camera: world (x forward, y up, z towards the viewer's side) → screen px (x right, y down,
    z towards the viewer)."""

    def __init__(self, origin_px, scale, yaw=0.0, pitch=8.0, mirror=False, roll=0.0):
        # yaw > 0 turns the animal's front towards the viewer; pitch > 0 = camera above, looking down a little
        R = _rx(pitch) @ _ry(yaw)
        if roll:
            R = _rz(roll) @ R
        self.R = R
        self.m = -1.0 if mirror else 1.0
        self.s = float(scale)
        self.T = (np.diag([self.m, -1.0, 1.0]) @ R) * self.s
        self.Ti = np.linalg.inv(self.T)
        self.o = np.array([origin_px[0], origin_px[1], 0.0])

    def p(self, w):
        """World point(s) (..., 3) → screen (..., 3)."""
        w = np.asarray(w, np.float64)
        return w @ self.T.T + self.o

    def v(self, w):
        return np.asarray(w, np.float64) @ self.T.T

    def world(self, X, Y, Z):
        """Screen px arrays → world coordinates (3 arrays)."""
        dx, dy, dz = X - self.o[0], Y - self.o[1], Z
        Ti = self.Ti
        return (Ti[0, 0] * dx + Ti[0, 1] * dy + Ti[0, 2] * dz, Ti[1, 0] * dx + Ti[1, 1] * dy + Ti[1, 2] * dz,
                Ti[2, 0] * dx + Ti[2, 1] * dy + Ti[2, 2] * dz)

    def normal_world(self, nx, ny, nz):
        T = self.T
        wx = T[0, 0] * nx + T[1, 0] * ny + T[2, 0] * nz
        wy = T[0, 1] * nx + T[1, 1] * ny + T[2, 1] * nz
        wz = T[0, 2] * nx + T[1, 2] * ny + T[2, 2] * nz
        n = np.sqrt(wx * wx + wy * wy + wz * wz) + 1e-9
        return wx / n, wy / n, wz / n


# ===================================================================== evaluating volumes


def _prim_bounds(cam, pr, pad):
    if pr.kind == "ell":
        c, A = pr.geom
        cs = cam.p(c)
        As = cam.v(A.T).T
        hx = math.sqrt(float((As[0] ** 2).sum()))
        hy = math.sqrt(float((As[1] ** 2).sum()))
        return cs[0] - hx - pad, cs[1] - hy - pad, cs[0] + hx + pad, cs[1] + hy + pad
    if pr.kind == "cone":
        a, b, ra, rb = pr.geom
        pa, pb = cam.p(a), cam.p(b)
        r = max(ra, rb) * cam.s
        return (min(pa[0], pb[0]) - r - pad, min(pa[1], pb[1]) - r - pad, max(pa[0], pb[0]) + r + pad,
                max(pa[1], pb[1]) + r + pad)
    P = cam.p(pr.geom)
    return P[:, 0].min() - pad, P[:, 1].min() - pad, P[:, 0].max() + pad, P[:, 1].max() + pad


def _eval_prim(cam, pr, X, Y):
    """Signed px distance, front depth (px, extended outside), and cone parameter t for a prim over grids."""
    if pr.kind == "ell":
        c, A = pr.geom
        cs = cam.p(c)
        As = cam.v(A.T).T                      # columns = screen semi-axes
        try:
            N = np.linalg.inv(As)
        except np.linalg.LinAlgError:
            N = np.linalg.pinv(As)
        qx, qy = X - cs[0], Y - cs[1]
        n0, n1, nw = N[:, 0], N[:, 1], N[:, 2]
        a = float(nw @ nw)
        # Nq = n0*qx + n1*qy  (3 comps)
        Nq = [n0[i] * qx + n1[i] * qy for i in range(3)]
        b = Nq[0] * nw[0] + Nq[1] * nw[1] + Nq[2] * nw[2]
        c0 = Nq[0] ** 2 + Nq[1] ** 2 + Nq[2] ** 2
        g = np.maximum(c0 - b * b / a, 0)
        q = np.sqrt(g)
        # gradient of g / 2 = G q  with G = n_i·n_j - (n_i·nw)(n_j·nw)/a
        G00 = float(n0 @ n0 - (n0 @ nw) ** 2 / a)
        G01 = float(n0 @ n1 - (n0 @ nw) * (n1 @ nw) / a)
        G11 = float(n1 @ n1 - (n1 @ nw) ** 2 / a)
        gx = G00 * qx + G01 * qy
        gy = G01 * qx + G11 * qy
        k1 = np.sqrt(gx * gx + gy * gy) + 1e-9
        d = q * (q - 1) / k1
        rmin = 1.0 / math.sqrt(max(G00, G11, 1e-12))
        d = np.where(q < 1e-3, -rmin, d)
        t0 = -b / a
        z = cs[2] + t0 + np.sqrt(np.maximum(1 - g, 0) / a)
        z = np.where(g < 1, z, cs[2] + t0 - np.maximum(d, 0))
        return d.astype(F32), z.astype(F32), None
    if pr.kind == "cone":
        a3, b3, ra, rb = pr.geom
        pa, pb = cam.p(a3), cam.p(b3)
        ra, rb = ra * cam.s, rb * cam.s
        ex, ey = pb[0] - pa[0], pb[1] - pa[1]
        L2 = ex * ex + ey * ey
        wx, wy = X - pa[0], Y - pa[1]
        if L2 > 1e-6 and abs(ra - rb) < math.sqrt(L2) * 0.999:
            # exact 2D uneven capsule (round cone) distance
            qx = (wx * ey - wy * ex) / L2
            qy = (wx * ex + wy * ey) / L2
            qx = np.abs(qx)
            bb = ra - rb
            cx_, cy_ = math.sqrt(L2 - bb * bb), bb
            kk = cx_ * qy - cy_ * qx
            mm = cx_ * qx + cy_ * qy
            nn = qx * qx + qy * qy
            d = np.where(kk < 0, np.sqrt(L2 * nn) - ra,
                         np.where(kk > cx_, np.sqrt(np.maximum(L2 * (nn + 1 - 2 * qy), 0)) - rb, mm - ra))
            h = np.clip(qy, 0, 1)
            dx, dy = wx - ex * h, wy - ey * h
            dist = np.sqrt(dx * dx + dy * dy)
        else:
            h = np.zeros_like(X)
            dist = np.sqrt(wx * wx + wy * wy)
            d = dist - max(ra, rb)
        zc = pa[2] + (pb[2] - pa[2]) * h
        inside = np.sqrt(np.maximum(d * d - 2 * d * dist, 0))
        z = np.where(d < 0, zc + inside, zc - d)
        # end spheres (matter when the cone points at the camera)
        for (pp, rr) in ((pa, ra), (pb, rb)):
            ds = np.sqrt((X - pp[0]) ** 2 + (Y - pp[1]) ** 2)
            zs = pp[2] + np.sqrt(np.maximum(rr * rr - ds * ds, 0))
            z = np.where(ds < rr, np.maximum(z, zs), z)
            d = np.minimum(d, ds - rr)
        return d.astype(F32), z.astype(F32), h.astype(F32)
    # flat polygon
    P = cam.p(pr.geom)
    d = _sdf.sd_polygon(X, Y, [(float(p[0]), float(p[1])) for p in P])
    # plane fit z = a x + b y + c
    M = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    try:
        coef, *_ = np.linalg.lstsq(M, P[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        coef = np.array([0.0, 0.0, P[:, 2].mean()])
    ca, cb = np.clip(coef[0], -6, 6), np.clip(coef[1], -6, 6)
    cc = P[:, 2].mean() - ca * P[:, 0].mean() - cb * P[:, 1].mean()
    zp = ca * X + cb * Y + cc
    span = max(float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])), 1.0)
    bw = max(span * 0.12 * pr.bulge, 0.6)
    z = zp + np.where(d < 0, np.sqrt(np.clip(-d / bw, 0, 1)) * bw * 0.9, -3.0 * d)
    return d.astype(F32), z.astype(F32), None


# ===================================================================== the renderer

_STYLES = ("soft", "flat", "cute")


def _gauss(a, s):
    return _f.gaussian_blur(a, s) if s > 0.25 else a


def _minabs_grad(Z):
    """Per-pixel slope that ignores depth cliffs: one-sided difference with the smaller magnitude."""
    gx = np.zeros_like(Z)
    gy = np.zeros_like(Z)
    fx = Z[:, 2:] - Z[:, 1:-1]
    bx = Z[:, 1:-1] - Z[:, :-2]
    gx[:, 1:-1] = np.where(np.abs(fx) < np.abs(bx), fx, bx)
    fy = Z[2:] - Z[1:-1]
    by = Z[1:-1] - Z[:-2]
    gy[1:-1] = np.where(np.abs(fy) < np.abs(by), fy, by)
    return gx, gy


class _Render:
    """Result of rendering a model into a crop: rgb, alpha, box, depth, plus the camera."""

    def __init__(self, cam, box, rgb_, alpha, Z, D):
        self.cam, self.box, self.rgb, self.alpha, self.Z, self.D = cam, box, rgb_, alpha, Z, D


def _render_model(cv, model, cam, style="soft", light=None, seed=0, detail=1.0):
    light = _form.resolve_light(light)
    S = cv.h / 1080.0
    prims = model.prims
    if not prims:
        return None
    pad = 4 + 0.03 * cam.s + max((g["k"] for g in model.groups.values()), default=0) * cam.s
    pads = [2 + max((p.k if p.k is not None else 0.012) * 2.5, 0.03 if p.kind == "flat" else 0.0) * cam.s
            for p in prims]
    bounds = [_prim_bounds(cam, p, pd) for p, pd in zip(prims, pads)]
    for d in model.decals:
        if d.kind == "line":
            P = cam.p(d.p)
            bounds.append((P[:, 0].min() - 3, P[:, 1].min() - 3, P[:, 0].max() + 3, P[:, 1].max() + 3))
    x0 = min(b[0] for b in bounds) - pad
    y0 = min(b[1] for b in bounds) - pad
    x1 = max(b[2] for b in bounds) + pad
    y1 = max(b[3] for b in bounds) + pad
    box = _form.clip_box(cv, x0, y0, x1, y1)
    if box is None:
        return None
    bx0, by0, bx1, by1 = box
    h, w = by1 - by0, bx1 - bx0
    Xc, Yc = np.meshgrid(np.arange(bx0, bx1, dtype=F32) + 0.5, np.arange(by0, by1, dtype=F32) + 0.5)

    # ---- evaluate prims in their own sub-boxes
    evals = []
    for p, b in zip(prims, bounds):
        sx0, sy0 = max(int(math.floor(b[0])) - bx0, 0), max(int(math.floor(b[1])) - by0, 0)
        sx1, sy1 = min(int(math.ceil(b[2])) - bx0, w), min(int(math.ceil(b[3])) - by0, h)
        if sx0 >= sx1 or sy0 >= sy1:
            evals.append(None)
            continue
        d, z, t = _eval_prim(cam, p, Xc[sy0:sy1, sx0:sx1], Yc[sy0:sy1, sx0:sx1])
        evals.append((sy0, sy1, sx0, sx1, d, z, t))

    # ---- groups: each blends with its parent, groups combine with a hard depth test
    order = list(model.groups.keys())
    acc = {}

    def _group_field(name):
        if name in acc:
            return acc[name]
        g = model.groups[name]
        members = [i for i, p in enumerate(prims) if p.group == name and evals[i] is not None]
        if g["parent"] is not None and g["parent"] in model.groups:
            Dp, Zp = _group_field(g["parent"])
            D, Z = Dp.copy(), Zp.copy()
            first = False
        else:
            D = np.full((h, w), _BIG, F32)
            Z = np.full((h, w), -_BIG, F32)
            first = True
        # members first among themselves, then blend with the parent
        Dm = np.full((h, w), _BIG, F32)
        Zm = np.full((h, w), -_BIG, F32)
        for i in members:
            sy0, sy1, sx0, sx1, d, z, _t = evals[i]
            k = (prims[i].k if prims[i].k is not None else 0.012) * cam.s
            Dm[sy0:sy1, sx0:sx1] = _smin(Dm[sy0:sy1, sx0:sx1], d, k)
            ze_ = z - 5.0 * np.maximum(d - k, 0)
            Zm[sy0:sy1, sx0:sx1] = _smax(Zm[sy0:sy1, sx0:sx1], ze_, k)
        if first:
            D, Z = Dm, Zm
        else:
            k = g["k"] * cam.s
            D = _smin(D, Dm, k)
            Z = _smax(Z, Zm - 5.0 * np.maximum(Dm - k, 0), k)
        acc[name] = (D.astype(F32), Z.astype(F32))
        return acc[name]

    D = np.full((h, w), _BIG, F32)
    Z = np.full((h, w), -_BIG, F32)
    for name in order:
        Dg, Zg = _group_field(name)
        ze = Zg - 6.0 * np.maximum(Dg, 0)
        Z = np.where(ze > Z, ze, Z)
        D = np.minimum(D, Dg)
    Z = Z.astype(F32)
    cover = np.clip(0.5 - D, 0, 1)
    if cover.max() <= 0:
        return None

    # ---- per-pixel attributes blended from the prims that own the front surface
    tau = max(1.0, 0.004 * cam.s)
    lam = max(1.2, 0.01 * cam.s)
    wsum = np.full((h, w), 1e-6, F32)
    col = np.zeros((h, w, 3), F32)
    att = np.zeros((9, h, w), F32)             # fur, pat, belly, spec, tone, pt, rough, leg, opacity
    axis = np.zeros((3, h, w), F32)
    Zs = Z
    coat0 = model.coat
    stripe_kind = coat0.get("pattern") if coat0.get("pattern") in _STRIPES else None
    ssum = np.zeros((h, w), F32) if stripe_kind else None

    def _logw(i):
        sy0, sy1, sx0, sx1, d, z, _t = evals[i]
        lam_i = max(0.6, min(lam, pads[i] / 7.0))
        return np.clip((z - Zs[sy0:sy1, sx0:sx1]) / tau, -30, 0) - np.maximum(d, 0) / lam_i

    # best log-weight per pixel: where a wide blend lifts the surface above every volume, the weights are
    # renormalised instead of all vanishing (which left dark specks and holes in the seams)
    Emax = np.full((h, w), -60.0, F32)
    for i in range(len(prims)):
        if evals[i] is not None:
            sy0, sy1, sx0, sx1 = evals[i][:4]
            Emax[sy0:sy1, sx0:sx1] = np.maximum(Emax[sy0:sy1, sx0:sx1], _logw(i))
    Emax = np.minimum(Emax, 0.0)
    for i, p in enumerate(prims):
        if evals[i] is None:
            continue
        sy0, sy1, sx0, sx1, d, z, t = evals[i]
        wt = np.exp(np.maximum(_logw(i) - Emax[sy0:sy1, sx0:sx1], -60.0)).astype(F32)
        if p.stops is not None and t is not None:
            ts = [s[0] for s in p.stops]
            cs = np.array([rgb(s[1]) for s in p.stops], F32)
            pts = [s[2] for s in p.stops]
            cmap = np.stack([np.interp(t, ts, cs[:, j]) for j in range(3)], -1).astype(F32)
            ptmap = np.interp(t, ts, pts).astype(F32)
        else:
            cmap = rgb(p.col)[None, None, :]
            ptmap = p.pt
        col[sy0:sy1, sx0:sx1] += cmap * wt[..., None]
        for j, v in enumerate((p.fur, p.pat, p.belly, p.spec, p.tone, ptmap, p.rough, p.leg, p.opa)):
            att[j, sy0:sy1, sx0:sx1] += v * wt
        ax = p.axis if p.axis is not None else (1, 0, 0)
        for j in range(3):
            axis[j, sy0:sy1, sx0:sx1] += float(ax[j]) * wt
        wsum[sy0:sy1, sx0:sx1] += wt
        if stripe_kind and p.pat > 0:
            Wx_, Wy_, Wz_ = cam.world(Xc[sy0:sy1, sx0:sx1], Yc[sy0:sy1, sx0:sx1], z)
            ring = p.name in ("forearm", "gaskin", "gaskin_m", "hock", "cannon", "meta", "pastern", "lower", "paw",
                              "tail")
            ssum[sy0:sy1, sx0:sx1] += _stripe_signal(stripe_kind, Wx_, Wy_, Wz_, ax if ring else None, coat0,
                                                     seed) * wt
    col /= wsum[..., None]
    att /= wsum[None]
    fur_w, pat_w, belly_w, spec_w, tone_w, pt_w, rough_w, leg_w, opa_w = att

    # ---- fluff (wool, mane) roughens depth and outline
    ws = cam.s
    if model.fluff > 0 or rough_w.max() > 0:
        Wx, Wy, Wz = cam.world(Xc, Yc, Z)
        cell = 0.035 if model.fluff <= 0 else model.fluff
        f1, f2 = _worley3(Wx / cell, Wy / cell, Wz / cell, seed=seed + 5)
        bump = np.clip(1 - f1, 0, 1) ** 0.5
        amp = rough_w * cell * ws * 0.45
        Z = (Z + bump * amp * 2.0).astype(F32)
        D = (D - (bump - 0.55) * amp * 1.6).astype(F32)

    # ---- melt the creases between blended volumes: blur the depth inside one continuous surface, keep the
    # depth cliffs (a leg in front of the body, the head over the chest) and the wool bumps sharp
    Zn = Z
    if detail > 0 and style != "flat":
        sz = 0.03 * ws
        if sz > 1.0:
            # glossy parts (nose leather, hooves, horns) keep their own crisp shape
            inside = ((cover > 0.02) * (1 - np.clip(spec_w * 3, 0, 1))).astype(F32)
            Zi = (Z * inside).astype(F32)
            Zs = _gauss(Zi, sz) / np.maximum(_gauss(inside, sz), 1e-3)
            tol = 0.035 * ws
            wsm = np.exp(-((Zs - Z) / tol) ** 2) * inside * (1 - np.clip(rough_w, 0, 1))
            Zn = (Z + (Zs - Z) * wsm * model.smooth).astype(F32)

    # ---- normals
    gx, gy = _minabs_grad(Zn)
    big = np.abs(gx) > 12
    gx = np.clip(gx, -12, 12)
    gy = np.clip(gy, -12, 12)
    nrm = np.sqrt(gx * gx + gy * gy + 1)
    nx, ny, nz = -gx / nrm, -gy / nrm, 1 / nrm
    sm = max(0.6, 0.004 * ws)
    nx, ny = _gauss(nx, sm), _gauss(ny, sm)
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0.0004, 1))
    Wx, Wy, Wz = cam.world(Xc, Yc, Z)
    wnx, wny, wnz = cam.normal_world(nx, ny, nz)

    # ---- coat: belly, pattern, points
    coat = model.coat
    base = col
    belly_col = rgb(coat.get("belly", coat.get("base", "#c08040")))
    if belly_w.max() > 0.01 and coat.get("belly") is not None:
        bd = coat.get("belly_dir", (0.0, -1.0, 0.0))
        bd = _unit(bd)
        facing = wnx * bd[0] + wny * bd[1] + wnz * bd[2]
        jag = (_fbm3(Wx * 7, Wy * 7, Wz * 7, 2, seed + 3) - 0.5) * 0.16
        lo = coat.get("belly_edge", 0.15)
        bm = smoothstep(lo - 0.12, lo + 0.12, facing + jag * coat.get("belly_jag", 1.0)) * belly_w
        base = base + (belly_col - base) * bm[..., None]
    if coat.get("pattern") == "saddle" or coat.get("saddle"):
        # dark blanket over the back (german shepherd, wolf)
        jag = (_fbm3(Wx * 9, Wy * 9, Wz * 9, 2, seed + 13) - 0.5) * 0.5
        sx = smoothstep(-0.62, -0.42, Wx) * smoothstep(0.5, 0.3, Wx)
        sm_ = smoothstep(0.15, 0.4, wny + jag) * sx * np.clip(pat_w, 0, 1) * (1 - np.clip(leg_w, 0, 1))
        sc_ = coat.get("saddle") or (coat.get("colours") or ["#2b2018"])[0]
        base = base + (rgb(sc_) - base) * sm_[..., None]
    pat = coat.get("pattern", "solid")
    if pat == "saddle":
        pat = "solid"
    pcols = coat.get("colours") or coat.get("colors") or ["#2a2018"]
    if isinstance(pcols, str):
        pcols = [pcols]
    pm = None
    if pat and pat != "solid" and pat_w.max() > 0.01:
        if stripe_kind:
            pm = _stripe_mark(stripe_kind, ssum / wsum, Wx, Wy, Wz, coat, seed) * pat_w
        else:
            pm = _pattern(pat, Wx, Wy, Wz, axis, coat, seed) * pat_w
        if pat in ("giraffe",) or (pat == "patches" and len(pcols) > 1):
            pass
        pc = rgb(pcols[0])
        if pat == "scales":
            pc = base * 0.62
        if pat == "bands":
            if coat.get("band_edge"):
                cs_ = dict(coat, band_w=coat.get("band_w", 0.055) + 0.018)
                pe = _pattern(pat, Wx, Wy, Wz, axis, cs_, seed) * pat_w
                base = base + (rgb(coat["band_edge"]) - base) * np.clip(pe, 0, 1)[..., None]
            pc = rgb(coat.get("band_color", "#f6f4f0"))
        if pat in _STRIPES and belly_w.max() > 0.01 and coat.get("belly") is not None:
            pm = pm * (1 - 0.75 * np.clip(bm, 0, 1))          # markings fade on the pale chest and belly
        if pat in ("tabby", "stripes"):
            pm = pm * (1 - 0.95 * smoothstep(0.3, 0.75, wnx) * np.clip(1 - leg_w, 0, 1))  # and on the chest front
        base = base + (pc - base) * np.clip(pm, 0, 1)[..., None]
        if pat == "patches" and len(pcols) > 1:
            pm2 = _pattern("patches", Wx + 3.1, Wy - 1.7, Wz + 2.3, axis, coat, seed + 11) * pat_w
            pm2 = pm2 * (1 - np.clip(pm, 0, 1))
            base = base + (rgb(pcols[1]) - base) * np.clip(pm2, 0, 1)[..., None]
    if pt_w.max() > 0.01:
        ptc = rgb(coat.get("points", "#4a3428"))
        base = base + (ptc - base) * np.clip(pt_w, 0, 1)[..., None]
    socks = coat.get("socks")
    if socks and leg_w.max() > 0.01:
        sh, sc_ = socks[0], rgb(socks[1])
        jag = (_fbm3(Wx * 20, Wy * 20, Wz * 20, 1, seed + 9) - 0.5) * 0.04
        sm_ = smoothstep(sh + 0.015, sh - 0.015, Wy + jag) * np.clip(leg_w, 0, 1)
        base = base + (sc_ - base) * sm_[..., None]

    # ---- fur texture: streaks along the hair flow
    if style == "soft" and detail > 0:
        ax_, ay_, az_ = axis
        an = np.sqrt(ax_ ** 2 + ay_ ** 2 + az_ ** 2) + 1e-6
        ax_, ay_, az_ = ax_ / an, ay_ / an, az_ / an
        along = Wx * ax_ + Wy * ay_ + Wz * az_
        # two coordinates across the flow
        c1 = Wx * (-ay_) + Wy * ax_
        c2 = Wz
        fs = coat.get("fur_scale", 1.0)
        # strand spacing is tied to pixels (about 3 px) so the streaks never alias into "wood grain" on big renders
        fa = float(np.clip(ws / 3.2, 30.0, 160.0)) / fs
        streak = _vnoise3(along * fa * 0.16, c1 * fa, c2 * fa, seed + 21) - 0.5
        streak = streak * (0.55 + 0.45 * np.clip(fur_w, 0, 1))
        base = base * (1 + streak[..., None] * 0.07 * np.clip(fur_w, 0, 1.6)[..., None] * coat.get("streak", 1.0))

    # ---- lighting
    L = light.vec
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    lc = rgb(light.color) * F32(light.intensity)
    sky = rgb(light.sky)
    bnc = rgb(light.bounce)
    up = (0.5 - 0.5 * ny)[..., None]
    if style == "soft":
        wrap = 0.45
        diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
        diff = diff * diff * (3 - 2 * diff) * 0.5 + diff * 0.5
        amb = (sky * up + bnc * (1 - up)) * F32(light.ambient) * 1.1
        E = diff[..., None] * lc * 0.92 + amb
        shaded = base * E
        rim = (1 - nz) ** 3 * np.clip(-ndl + 0.3, 0, 1) * 0.0 + (1 - nz) ** 4 * 0.18
        shaded = shaded + sky * rim[..., None] * fur_w[..., None]
        sheen = np.clip(ndl, 0, 1) ** 2 * (1 - nz) ** 1.5 * 0.25 * fur_w
        shaded = shaded + (base * 0.5 + lc * 0.5) * sheen[..., None]
    else:
        wrap = 0.3
        diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
        shadow_c = rgb(light.shadow)
        if style == "flat":
            t = smoothstep(0.36, 0.42, diff)
            shade_col = base * (0.7 + 0.3 * shadow_c * 1.4)
            shaded = shade_col + (base - shade_col) * t[..., None]
        else:   # cute: soft cel with three tones and a rim highlight
            t1 = smoothstep(0.30, 0.40, diff)
            t2 = smoothstep(0.80, 0.86, diff)
            dark = base * (0.72 + 0.28 * shadow_c * 1.2)
            shaded = dark + (base - dark) * t1[..., None]
            shaded = shaded + (np.minimum(base * 1.12 + 0.04, 1.2) - shaded) * (t2 * 0.6)[..., None]
            rim = smoothstep(0.55, 0.75, 1 - nz) * np.clip(-ndl + 0.2, 0, 1)
            shaded = shaded + (sky * 0.25) * rim[..., None]
    # specular (noses, hooves, horns, scales, eyes)
    if spec_w.max() > 0.01:
        Hv = _unit(L + np.array([0, 0, 1.0]))
        ndh = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1)
        if style == "flat":
            shaded = shaded + lc * (spec_w * smoothstep(0.93, 0.95, ndh) * 0.35)[..., None]
        else:
            shaded = shaded + lc * (spec_w * (ndh ** 30 * 0.55 + ndh ** 6 * 0.1))[..., None]
    shaded = shaded * tone_w[..., None]

    # ---- occlusion: darken surfaces next to something in front of them (creases, the leg in front of the belly)
    if detail > 0:
        so = max(1.5, 0.025 * ws)
        Zc = np.where(cover > 0, Z, np.nan)
        Zf = np.nan_to_num(Zc, nan=float(np.nanmin(Zc)) if np.isfinite(np.nanmin(Zc)) else 0.0)
        Zb = _gauss(Zf, so)
        ao = np.clip((Zb - Zf) / (0.05 * ws), 0, 1)
        k_ao = 0.3 if style == "soft" else 0.24
        shaded = shaded * (1 - k_ao * ao ** 1.2)[..., None]
        # inner contour lines where a near part overlaps a far one (depth cliff)
        cliff = np.zeros_like(Z)
        thr = 0.05 * ws
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nb = np.roll(np.roll(Zf, dy, 0), dx, 1)          # neighbour on one side
            op = np.roll(np.roll(Zf, -dy, 0), -dx, 1)        # and on the other
            jump = nb - Zf                                   # neighbour nearer by this much
            slope = np.abs(Zf - op)                          # the smooth slope on the far side
            cliff = np.maximum(cliff, np.clip((jump - 2.5 * slope - thr) / thr, 0, 1))
        cl_w = {"soft": 0.35, "flat": 0.55, "cute": 0.8}[style]
        cliff = _gauss(cliff, 0.5) * cl_w
        dark = shaded * 0.45
        shaded = shaded + (dark - shaded) * np.clip(cliff, 0, 1)[..., None]

    # ---- fur at the silhouette
    Dfin = D
    if style == "soft" and model.fur_len > 0 and detail > 0:
        amp = model.fur_len * ws * coat.get("fur", 1.0)
        Dfin = _fur_fringe(D, fur_w, amp, seed, Xc, Yc)
        if amp >= 0.6:
            # hair tips take the colour of the coat just inside the outline (outside it the depth falls away and
            # would shade them almost black)
            inner = np.clip(-D / max(amp * 0.5, 1.0) - 0.5, 0, 1).astype(F32)
            sg = max(amp * 0.9, 1.0)
            ext = _gauss(shaded * inner[..., None], sg) / np.maximum(_gauss(inner, sg), 1e-4)[..., None]
            tw = np.clip((D + amp * 0.6) / max(amp * 0.6, 1.0), 0, 1)[..., None]
            shaded = shaded + (ext * 0.94 - shaded) * tw
    alpha = np.clip(0.5 - Dfin, 0, 1).astype(F32)
    if opa_w.min() < 0.999:
        alpha = alpha * np.clip(opa_w, 0, 1)
    clip_y = getattr(model, "clip_y", None)
    if clip_y is not None:
        # hide everything below a water line (world y), with a soft edge
        alpha = alpha * smoothstep(clip_y - 0.006, clip_y + 0.006, Wy).astype(F32)

    # ---- outline
    if style in ("flat", "cute"):
        ow = max(1.0, (0.008 if style == "cute" else 0.005) * ws)
        ring = np.clip(0.5 - np.abs(Dfin + ow * 0.5) + ow * 0.5, 0, 1)
        oc = shaded * 0.35 + rgb(coat.get("outline", "#2a1e18")) * 0.4
        shaded = shaded + (oc - shaded) * (ring * (D > -ow * 2.5))[..., None]
    elif detail > 0:
        ow = max(0.8, 0.004 * ws)
        # a thin dark edge on smooth skin; furry edges stay light so the hair tips do not turn into a black fringe
        ring = np.clip(1 - np.abs(Dfin + ow) / ow, 0, 1) * 0.35 * (1 - 0.8 * np.clip(fur_w * model.fur_len * 60, 0, 1))
        shaded = shaded * (1 - ring)[..., None]

    rr = _Render(cam, box, shaded.astype(F32), alpha, Z, D)
    rr.X, rr.Y = Xc, Yc
    rr.world = (Wx, Wy, Wz)
    rr.light = light
    _draw_decals(rr, model, style, light, S)
    return rr


def _fur_fringe(D, fur_w, amp, seed, X, Y):
    """Push the silhouette out in tufts of hair (only near the edge, only where the coat is furry)."""
    if amp < 0.6:
        return D
    gx, gy = np.gradient(D)
    g = np.sqrt(gx * gx + gy * gy) + 1e-6
    tx, ty = -gy / g, gx / g
    sw = max(amp * 0.55, 1.0)
    s = (X * tx + Y * ty) / sw + 0.6 * (X * gx + Y * gy) / g / sw
    n = 0.65 * _vnoise1(s, seed) + 0.35 * _vnoise1(s * 2.7 + 5.0, seed + 1)
    n = np.clip((n - 0.25) / 0.75, 0, 1) ** 1.5
    band = np.clip(1 - np.abs(D) / (amp * 2.2), 0, 1)
    return (D - amp * n * band * np.clip(fur_w, 0, 1.5)).astype(F32)


_STRIPES = ("stripes", "tabby", "tiger", "zebra")


def _stripe_signal(kind, X, Y, Z, axis, coat, seed):
    """Stripe wave 0~1 for one part: bands across the body (torso, head) or rings around a limb / tail (axis)."""
    sc = coat.get("scale", 1.0)
    freq = {"stripes": 8.0, "tabby": 12.0, "tiger": 7.0, "zebra": 10.0}[kind] / sc
    warp = (_fbm3(X * 2.5, Y * 2.5, Z * 2.5, 2, seed + 41) - 0.5) * {"zebra": 0.5, "tiger": 1.5}.get(kind, 1.3)
    if axis is None:
        along = X + 0.28 * Y + 0.06 * np.abs(Z) ** 2
    else:
        a = _unit(axis)
        along = X * a[0] + Y * a[1] + Z * a[2]
        freq = freq * 1.25
    return (0.5 + 0.5 * np.cos((along * freq + warp * 1.6) * 2 * np.pi)).astype(F32)


def _stripe_mark(kind, s, X, Y, Z, coat, seed):
    """Marking 0~1 from the blended stripe wave: thickness breaks up, tapers towards the belly."""
    brk = _fbm3(X * 7, Y * 7, Z * 7, 2, seed + 43)
    width = {"zebra": 0.42, "tiger": 0.64, "tabby": 0.77, "stripes": 0.55}[kind]
    if kind == "tiger":
        width = width + (brk - 0.5) * 0.45
    elif kind == "tabby":
        width = width + (brk - 0.5) * 0.42          # broken, uneven mackerel lines
    else:
        width = width + (brk - 0.5) * 0.1
    width = width + coat.get("stripe_taper", 0.12 if kind != "zebra" else 0.0) * np.clip(-Y * 2.0 + 0.5, 0, 1)
    m = smoothstep(width - 0.05, width + 0.05, s)
    return m.astype(F32)


def _pattern(kind, X, Y, Z, axis, coat, seed):
    """Marking strength 0~1 on the surface (world coordinates in body units)."""
    sc = coat.get("scale", 1.0)
    ax_, ay_, az_ = axis
    an = np.sqrt(ax_ ** 2 + ay_ ** 2 + az_ ** 2) + 1e-6
    ax_, ay_, az_ = ax_ / an, ay_ / an, az_ / an
    if kind in ("stripes", "tabby", "tiger", "zebra"):
        along = X * ax_ + Y * ay_ + Z * az_
        freq = {"stripes": 9.0, "tabby": 14.0, "tiger": 10.0, "zebra": 12.0}[kind] / sc
        warp = (_fbm3(X * 3, Y * 3, Z * 3, 2, seed + 41) - 0.5) * {"zebra": 0.9, "tiger": 1.6, "tabby": 1.4}.get(kind, 1.0)
        ph = along * freq + warp * 2.0 + Y * 2.5
        s = 0.5 + 0.5 * np.cos(ph * 2 * np.pi)
        width = {"zebra": 0.5, "tiger": 0.68, "tabby": 0.64, "stripes": 0.6}[kind]
        brk = _fbm3(X * 6, Y * 6, Z * 6, 2, seed + 43)
        if kind == "tiger":
            width = width + (brk - 0.5) * 0.35 + 0.1 * np.clip(0.75 - Y, 0, 1)
        if kind == "tabby":
            width = width + (brk - 0.5) * 0.25
        m = smoothstep(width - 0.06, width + 0.06, s)
        if kind == "tabby":
            spine = np.exp(-(Z / 0.05) ** 2) * smoothstep(0.0, 0.1, ay_ * 0 + 1)
            m = np.maximum(m * 0.9, 0)
        return m.astype(F32)
    if kind == "spots":
        f1, f2 = _worley3(X * 11 / sc, Y * 11 / sc, Z * 11 / sc, seed + 51)
        r = coat.get("spot", 0.3)
        return smoothstep(r + 0.06, r - 0.06, f1).astype(F32)
    if kind == "rosettes":
        f1, f2 = _worley3(X * 9 / sc, Y * 9 / sc, Z * 9 / sc, seed + 52)
        ring = smoothstep(0.42, 0.34, f1) * smoothstep(0.12, 0.22, f1)
        brk = _fbm3(X * 30, Y * 30, Z * 30, 1, seed + 3)
        return (ring * smoothstep(0.3, 0.45, brk)).astype(F32)
    if kind == "giraffe":
        f1, f2 = _worley3(X * 6.5 / sc, Y * 6.5 / sc, Z * 6.5 / sc, seed + 53, jitter=0.8)
        edge = f2 - f1
        return smoothstep(0.06, 0.12, edge).astype(F32)
    if kind == "patches":
        n = _fbm3(X * 2.4 / sc, Y * 2.4 / sc, Z * 2.4 / sc, 3, seed + 61)
        amt = coat.get("amount", 0.5)
        th = 0.62 - amt * 0.25
        return smoothstep(th - 0.02, th + 0.02, n).astype(F32)
    if kind == "dapple":
        f1, f2 = _worley3(X * 14 / sc, Y * 14 / sc, Z * 14 / sc, seed + 71)
        return (smoothstep(0.2, 0.45, f1) * 0.5).astype(F32)
    if kind == "scales":
        f1, f2 = _worley3(X * 34 / sc, Y * 34 / sc, Z * 34 / sc, seed + 81, jitter=0.5)
        edge = smoothstep(0.12, 0.02, f2 - f1)
        return (edge * 0.3).astype(F32)
    if kind == "scutes":
        f1, f2 = _worley3(X * 5.5 / sc, Y * 5.5 / sc, Z * 5.5 / sc, seed + 83, jitter=0.55)
        return (smoothstep(0.1, 0.02, f2 - f1) * 0.85 + smoothstep(0.1, 0.45, f1) * 0.25).astype(F32)
    if kind == "bands":
        out = np.zeros_like(X, F32)
        for b in coat.get("bands", (0.25, 0.55)):
            xb = 0.5 - b
            w = coat.get("band_w", 0.055) * (1 - 0.3 * (b > 0.75))
            out = np.maximum(out, smoothstep(w + 0.008, w - 0.008, np.abs(X - xb + 0.03 * np.sin(Y * 20))))
        return out.astype(F32)
    if kind == "points":
        return np.zeros_like(X, F32)
    return np.zeros_like(X, F32)


# ===================================================================== decals (eyes, noses, mouths, lines)


def _crop_mask_seg(rr, pts2, width, aa=1.0):
    """Anti-aliased mask (crop sized) of a polyline of screen points with width px (scalar or per point)."""
    h, w = rr.alpha.shape
    x0, y0 = rr.box[0], rr.box[1]
    m = np.zeros((h, w), F32)
    pts2 = np.asarray(pts2, np.float64)
    ws = np.broadcast_to(np.asarray(width, np.float64), (len(pts2),))
    for i in range(len(pts2) - 1):
        a, b = pts2[i], pts2[i + 1]
        r = max(ws[i], ws[i + 1]) / 2 + 2
        sx0 = max(int(min(a[0], b[0]) - r) - x0, 0)
        sy0 = max(int(min(a[1], b[1]) - r) - y0, 0)
        sx1 = min(int(max(a[0], b[0]) + r) + 1 - x0, w)
        sy1 = min(int(max(a[1], b[1]) + r) + 1 - y0, h)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        X = rr.X[sy0:sy1, sx0:sx1]
        Y = rr.Y[sy0:sy1, sx0:sx1]
        ex, ey = b[0] - a[0], b[1] - a[1]
        L2 = max(ex * ex + ey * ey, 1e-9)
        hh = np.clip(((X - a[0]) * ex + (Y - a[1]) * ey) / L2, 0, 1)
        dist = np.sqrt((X - a[0] - ex * hh) ** 2 + (Y - a[1] - ey * hh) ** 2)
        rad = (ws[i] + (ws[i + 1] - ws[i]) * hh) / 2
        m[sy0:sy1, sx0:sx1] = np.maximum(m[sy0:sy1, sx0:sx1], np.clip((rad - dist) / aa + 0.5, 0, 1))
    return m


def _visible(rr, p_screen, tol):
    """1 if a screen point (x, y, z) is on the visible surface (not hidden behind nearer parts)."""
    x, y, z = p_screen
    ix, iy = int(round(x - rr.box[0])), int(round(y - rr.box[1]))
    h, w = rr.Z.shape
    if not (0 <= ix < w and 0 <= iy < h):
        return 0.0
    zz = float(rr.Z[iy, ix])
    if rr.D[iy, ix] > 1.5:
        return 0.0
    return float(np.clip((z - zz + tol) / max(tol * 0.5, 1e-3), 0, 1))


def _paint(rr, color, m):
    m = np.clip(m, 0, 1)[..., None]
    rr.rgb = rr.rgb + (np.asarray(as_color(color), F32) - rr.rgb) * m


def _draw_decals(rr, model, style, light, S):
    cam = rr.cam
    for dc in model.decals:
        if dc.kind == "eye":
            _draw_eye(rr, dc, style, light)
        elif dc.kind == "nose":
            _draw_nose(rr, dc, style, light)
        elif dc.kind == "line":
            if dc.n is not None:
                nz_ = cam.v(dc.n)
                if nz_[2] / (np.linalg.norm(nz_) + 1e-9) < 0.15:
                    continue                         # e.g. whiskers on the far side of the face
            P = cam.p(dc.p)
            tol = dc.opt.get("tol", 0.06) * cam.s
            vis = np.array([_visible(rr, p, tol) for p in P])
            if vis.max() <= 0:
                continue
            wpx = max(dc.opt.get("width", 0.01) * cam.s, 0.8)
            m = _crop_mask_seg(rr, P[:, :2], wpx * np.maximum(vis, 0.0) ** 0.5 + 1e-3)
            if dc.opt.get("clip", True):
                m = m * rr.alpha
            op = dc.opt.get("opacity", 1.0)
            _paint(rr, dc.opt.get("color", "#2a1a14"), m * op)
            if not dc.opt.get("clip", True):
                rr.alpha = np.maximum(rr.alpha, m * op)
        elif dc.kind == "spot":
            P = cam.p(dc.p)
            tol = dc.opt.get("tol", 0.05) * cam.s
            v = _visible(rr, P, tol)
            if v <= 0:
                continue
            r = dc.size * cam.s
            dd = np.sqrt((rr.X - P[0]) ** 2 + ((rr.Y - P[1]) / dc.opt.get("squash", 1.0)) ** 2) - r
            m = np.clip(0.5 - dd / max(dc.opt.get("soft", 1.0) * r * 0.5, 0.7), 0, 1) * rr.alpha * v
            _paint(rr, dc.opt.get("color", "#000000"), m * dc.opt.get("opacity", 1.0))


def _decal_frame(rr, dc):
    """Screen centre and the 2x2 map from screen offsets to decal-local (a = forward, b = up) coordinates."""
    cam = rr.cam
    c = cam.p(dc.p)
    n = dc.n if dc.n is not None else _v3(0, 0, 1)
    fwd = dc.fwd if dc.fwd is not None else _v3(1, 0, 0)
    # "up" on the decal is the head's up direction laid onto the surface: slit pupils and lids stay upright in
    # front views instead of tilting with the eye's sideways normal
    U = _v3(0, 1, 0) - fwd * float(fwd[1])
    up = U - n * float(np.dot(U, n))
    if np.linalg.norm(up) < 1e-3:
        up = np.cross(n, fwd)
    up = _unit(up)
    if up[1] < 0:
        up = -up
    fwd = _unit(np.cross(up, n))
    if fwd[0] < 0 and dc.fwd is not None and dc.fwd[0] > 0:
        fwd = -fwd
    fs = cam.v(fwd * dc.size)
    us = cam.v(up * dc.size)
    ns = cam.v(n)
    facing = ns[2] / (np.linalg.norm(ns) + 1e-9)
    M = np.array([[fs[0], us[0]], [fs[1], us[1]]])
    det = np.linalg.det(M)
    if abs(det) < 1e-6:
        return c, None, facing
    return c, np.linalg.inv(M), facing


def _local(rr, c, Mi):
    dx, dy = rr.X - c[0], rr.Y - c[1]
    a = Mi[0, 0] * dx + Mi[0, 1] * dy
    b = Mi[1, 0] * dx + Mi[1, 1] * dy
    return a, b


def _draw_eye(rr, dc, style, light):
    c, Mi, facing = _decal_frame(rr, dc)
    if Mi is None or facing < 0.05:
        return
    tol = dc.opt.get("tol", 0.08) * rr.cam.s
    vis = _visible(rr, c, tol)
    if vis <= 0:
        return
    vis = vis * smoothstep(0.05, 0.25, facing)
    a, b = _local(rr, c, Mi)
    o = dc.opt
    shape = o.get("shape", "round")
    px_per = max(1.0, rr.cam.s * dc.size * min(facing, 1) ** 0.5)
    aa = 1.0 / px_per                             # one pixel in local units (approx)
    if shape == "almond":
        # lens: intersection of two circles; tilted up towards the front
        b2 = b - a * 0.18
        d = np.maximum(np.sqrt(a * a + (b2 + 0.75) ** 2) - 1.25, np.sqrt(a * a + (b2 - 0.75) ** 2) - 1.25)
        d = np.maximum(d, np.abs(a) - 1.0)
    elif shape == "oval":
        d = np.sqrt(a * a + (b / 0.8) ** 2) - 1.0
    else:
        d = np.sqrt(a * a + b * b) - 1.0
    eye_m = np.clip(0.5 - d / aa, 0, 1) * vis * rr.alpha
    if eye_m.max() <= 0:
        return
    rim = o.get("rim", "#1a120e")
    iris_c = rgb(o.get("iris", "#6b4a2a"))
    scl = o.get("sclera")
    # rim / lid (slightly larger dark shape)
    rim_w = o.get("rim_w", 0.22)
    rim_m = np.clip(0.5 - (d - rim_w) / aa, 0, 1) * vis * rr.alpha
    _paint(rr, rim, rim_m * o.get("rim_opacity", 0.9))
    # eyeball
    r_iris = o.get("iris_r", 0.92)
    look = o.get("look", (0.12, 0.0))
    ia, ib = a - look[0], b - look[1]
    di = np.sqrt(ia * ia + ib * ib)
    if scl:
        _paint(rr, scl, eye_m)
    iris_m = np.clip(0.5 - (di - r_iris) / aa, 0, 1) * eye_m
    grad = np.clip(0.55 + 0.45 * (-ib / max(r_iris, 1e-3)), 0, 1)
    ring_dark = np.clip((di / max(r_iris, 1e-3) - 0.7) / 0.3, 0, 1)
    irc = iris_c[None, None, :] * (0.55 + 0.55 * (1 - grad))[..., None] * (1 - 0.45 * ring_dark)[..., None]
    rr.rgb = rr.rgb + (irc - rr.rgb) * iris_m[..., None]
    pupil = o.get("pupil", "round")
    pr = o.get("pupil_r", 0.45)
    if pupil == "slit":
        dp = np.sqrt((ia / (pr * 0.32)) ** 2 + (ib / (r_iris * 0.95)) ** 2) - 1.0
        dp = dp * pr * 0.32
    elif pupil == "bar":
        dp = np.sqrt((ia / (pr * 1.25)) ** 2 + (ib / (pr * 0.5)) ** 2) - 1.0
        dp = dp * pr * 0.5
    else:
        dp = di - pr
    pupil_m = np.clip(0.5 - dp / aa, 0, 1) * eye_m
    _paint(rr, o.get("pupil_color", "#0b0806"), pupil_m)
    # upper lid shadow
    lid = np.clip((b + 0.1) / 0.9, 0, 1) ** 2 * 0.45 * eye_m
    _paint(rr, "#000000", lid * o.get("lid", 1.0))
    # highlights: fixed in screen space towards the light
    L = light.dir2
    hr = o.get("glint", 0.26) * (1.35 if style == "cute" else 1.0)
    gpx = rr.cam.s * dc.size
    hx = c[0] + L[0] * gpx * 0.38
    hy = c[1] + L[1] * gpx * 0.38
    dd = np.sqrt((rr.X - hx) ** 2 + (rr.Y - hy) ** 2) - hr * gpx * min(facing + 0.3, 1)
    glint = np.clip(0.5 - dd, 0, 1) * eye_m
    _paint(rr, "#ffffff", glint * 0.95)
    if style == "cute" or o.get("glint2", True):
        hx2 = c[0] - L[0] * gpx * 0.32
        hy2 = c[1] - L[1] * gpx * 0.3
        dd2 = np.sqrt((rr.X - hx2) ** 2 + (rr.Y - hy2) ** 2) - hr * 0.42 * gpx
        _paint(rr, "#ffffff", np.clip(0.5 - dd2, 0, 1) * eye_m * 0.7)


def _draw_nose(rr, dc, style, light):
    c, Mi, facing = _decal_frame(rr, dc)
    if Mi is None or facing < 0.02:
        return
    tol = dc.opt.get("tol", 0.08) * rr.cam.s
    vis = _visible(rr, c, tol)
    if vis <= 0:
        return
    a, b = _local(rr, c, Mi)
    px_per = max(1.0, rr.cam.s * dc.size * max(facing, 0.2) ** 0.5)
    aa = 1.0 / px_per
    kind = dc.opt.get("shape", "dog")
    colr = dc.opt.get("color", "#1d1614")
    if kind == "dog":
        # rounded triangle, point down
        d = _sdf.sd_polygon(a, b, [(-1.0, 0.45), (1.0, 0.45), (0.0, -0.8)])
        d = (d - 0.32).astype(F32)
    elif kind == "cat":
        d = _sdf.sd_polygon(a, b, [(-0.9, 0.5), (0.9, 0.5), (0.0, -0.55)]) - 0.18
    elif kind == "snout":       # pig disk
        d = np.sqrt(a * a + b * b) - 1.0
    else:                       # nostril slit / oval
        d = np.sqrt((a / 1.0) ** 2 + (b / 0.6) ** 2) - 1.0
    m = np.clip(0.5 - d / aa, 0, 1) * vis * rr.alpha
    base = rgb(colr)
    shade_ = np.clip(0.75 + 0.35 * b, 0.5, 1.15)
    col = base[None, None, :] * shade_[..., None]
    rr.rgb = rr.rgb + (col - rr.rgb) * m[..., None]
    if kind in ("dog", "snout"):
        for sgn in (-1, 1):
            na = a - sgn * (0.42 if kind == "dog" else 0.38)
            nb = b - (0.05 if kind == "dog" else 0.0)
            dn = np.sqrt((na / 0.22) ** 2 + (nb / (0.15 if kind == "dog" else 0.3)) ** 2) - 1
            _paint(rr, "#050303" if kind == "dog" else "#5a2a2a", np.clip(0.5 - dn * 0.2 / aa, 0, 1) * m * 0.9)
    if kind == "dog":
        L = light.dir2
        gpx = rr.cam.s * dc.size
        dd = np.sqrt((rr.X - c[0] - L[0] * gpx * 0.3) ** 2 + (rr.Y - c[1] - L[1] * gpx * 0.4 + gpx * 0.25) ** 2) - gpx * 0.22
        _paint(rr, "#ffffff", np.clip(0.5 - dd, 0, 1) * m * 0.35)


# ===================================================================== quadrupeds: anatomy presets
#
# Units: body length (point of shoulder to point of buttock) = 1. World x forward, y up, z towards the viewer.
# Front leg: shoulder S -> elbow (points back) -> wrist / carpus -> paw or fetlock + pastern + hoof.
# Hind leg:  hip -> stifle (points forward) -> hock (points back) -> paw or fetlock + pastern + hoof.

_Q = dict(
    trunk=0.74,
    chest=(0.27, 0.24, 0.165), chest_at=(-0.1, 0.02),
    belly=(0.3, 0.17, 0.155), belly_at=(0.0, -0.03),
    rump=(0.2, 0.19, 0.155), rump_at=(0.04, 0.05),
    loin=(0.26, 0.11, 0.13), loin_at=(0.0, 0.1),
    withers=0.0, hump=0.0,
    f=(0.24, 0.29, 0.11), h=(0.28, 0.3, 0.17),
    fr=(0.09, 0.062, 0.045, 0.04), hr=(0.14, 0.075, 0.045, 0.04),
    foot="paw", paw=(0.055, 0.04, 0.045), stance=(0.0, -0.05), width=0.095,
    reach_f=0.93, reach_h=0.88, mu=12.0, nu=8.0,
    neck_len=0.26, neck_r=(0.12, 0.085), neck_ang=48.0, neck_at=(0.04, 0.12), neck_arch=0.02,
    skull=(0.12, 0.105, 0.1), head_at=(0.06, 0.01), head_ang=-8.0,
    muzzle=0.16, muzzle_r=(0.08, 0.066), muzzle_at=(0.06, -0.035), muzzle_drop=-3.0, chin=0.0, muzzle_w=0.85,
    jaw=0.7, cheek=0.0, jowl=0.05,
    eye=dict(dir=(0.78, 0.42, 0.48), size=0.025, shape="almond", iris="#5a3418", pupil="round", pupil_r=0.5),
    ear=dict(kind="floppy", at=(-0.03, 0.06, 0.07), len=0.14, w=0.085, up=-60.0, back=20.0, out=10.0),
    nose=dict(shape="dog", size=0.04, color="#1b1412"),
    mouth=True, whiskers=0,
    tail=dict(kind="thin", len=0.42, r=(0.042, 0.024), n=6, ang=-35.0, curve=12.0),
    fur=0.01, fluff=0.0,
    mane=None, horns=None, antlers=None, trunk_len=0.0, tusks=0.0, beard=0.0, udder=0.0, ossicones=0.0,
    gallop="rotary", spine_flex=0.08, wool=False,
)

_SPECIES_Q = {
    "labrador": dict(coat=dict(base="#d9b27a", belly="#ecd7b0", pattern="solid", belly_edge=0.35), fur=0.006,
                     tail=dict(kind="otter", len=0.42, r=(0.05, 0.025), n=6, ang=-25.0, curve=10.0)),
    "shepherd": dict(f=(0.23, 0.27, 0.11), h=(0.28, 0.29, 0.17), neck_len=0.28, muzzle=0.19, muzzle_r=(0.075, 0.05),
                     ear=dict(kind="pointed", at=(-0.2, 0.8, 0.5), len=0.17, w=0.12, up=75.0, back=5.0, out=12.0),
                     tail=dict(kind="bushy", len=0.48, r=(0.05, 0.03), n=7, ang=-55.0, curve=14.0), fur=0.016,
                     coat=dict(base="#b97a3c", belly="#d9b27a", pattern="saddle", colours=["#2b2018"], belly_edge=0.1,
                               muzzle="#3a2a20")),
    "corgi": dict(f=(0.12, 0.12, 0.06), h=(0.14, 0.13, 0.08), fr=(0.08, 0.055, 0.045, 0.04),
                  hr=(0.12, 0.065, 0.045, 0.04), chest=(0.28, 0.23, 0.17), belly=(0.3, 0.17, 0.16), rump=(0.2, 0.18, 0.16),
                  neck_len=0.2, muzzle=0.15, muzzle_r=(0.075, 0.05), skull=(0.125, 0.11, 0.11),
                  ear=dict(kind="pointed", at=(-0.15, 0.8, 0.55), len=0.19, w=0.15, up=72.0, back=0.0, out=18.0),
                  tail=dict(kind="stub", len=0.08, r=(0.045, 0.035), n=2, ang=-10.0, curve=0.0), fur=0.012,
                  coat=dict(base="#d6893a", belly="#f6efe4", belly_edge=-0.05, belly_dir=(0.45, -1.0, 0.0),
                            muzzle="#f6efe4")),
    "poodle": dict(f=(0.24, 0.28, 0.12), h=(0.27, 0.3, 0.17), fr=(0.07, 0.05, 0.035, 0.03), hr=(0.11, 0.06, 0.035, 0.03),
                   neck_len=0.32, neck_ang=58.0, muzzle=0.19, muzzle_r=(0.065, 0.045), skull=(0.115, 0.115, 0.1),
                   ear=dict(kind="floppy", at=(-0.03, 0.05, 0.075), len=0.18, w=0.09, up=-80.0, back=5.0, out=8.0),
                   tail=dict(kind="pompom", len=0.2, r=(0.03, 0.025), n=3, ang=55.0, curve=0.0), fur=0.0, fluff=0.05,
                   coat=dict(base="#f2ede6", belly="#f2ede6", nose="#2a1e1a")),
    "husky": dict(muzzle=0.17, muzzle_r=(0.078, 0.052),
                  ear=dict(kind="pointed", at=(-0.15, 0.8, 0.55), len=0.14, w=0.115, up=78.0, back=0.0, out=12.0),
                  tail=dict(kind="plume", len=0.45, r=(0.055, 0.04), n=7, ang=70.0, curve=-32.0), fur=0.018,
                  eye=dict(dir=(0.6, 0.4, 0.7), size=0.024, shape="almond", iris="#7fb8e8", pupil="round", pupil_r=0.42),
                  coat=dict(base="#5a5e66", belly="#f3f1ee", belly_edge=-0.2, belly_dir=(0.35, -1.0, 0.0),
                            muzzle="#f3f1ee")),
    "fox": dict(f=(0.2, 0.23, 0.1), h=(0.24, 0.25, 0.15), fr=(0.07, 0.045, 0.03, 0.028), hr=(0.12, 0.055, 0.03, 0.028),
                chest=(0.26, 0.21, 0.14), belly=(0.3, 0.15, 0.13), rump=(0.19, 0.17, 0.14), neck_len=0.22,
                neck_r=(0.1, 0.075), skull=(0.105, 0.09, 0.095), muzzle=0.18, muzzle_r=(0.066, 0.036), muzzle_drop=-4.0,
                muzzle_w=0.9, jaw=0.75, jowl=0.0, cheek=0.045, nose=dict(shape="dog", size=0.038, color="#1b1412"),
                ear=dict(kind="pointed", at=(-0.2, 0.8, 0.55), len=0.17, w=0.14, up=74.0, back=8.0, out=14.0),
                tail=dict(kind="brush", len=0.6, r=(0.05, 0.03), n=8, ang=-18.0, curve=-4.0), fur=0.016,
                eye=dict(dir=(0.6, 0.42, 0.68), size=0.022, shape="almond", iris="#c88a1a", pupil="slit", pupil_r=0.45),
                coat=dict(base="#d0662a", belly="#f4ede2", belly_edge=0.0, belly_dir=(0.5, -1.0, 0.0),
                          socks=(0.17, "#2a1c16"), tail_tip="#f4ede2", ear_back="#2a1c16", chin="#f4ede2",
                          cheek="#f4ede2")),
    "wolf": dict(f=(0.26, 0.31, 0.12), h=(0.3, 0.32, 0.18), muzzle=0.18, muzzle_r=(0.08, 0.052),
                 neck_r=(0.14, 0.105), skull=(0.12, 0.108, 0.105), jowl=0.0,
                 ear=dict(kind="pointed", at=(-0.2, 0.8, 0.55), len=0.14, w=0.115, up=72.0, back=8.0, out=14.0),
                 tail=dict(kind="bushy", len=0.48, r=(0.05, 0.035), n=7, ang=-62.0, curve=8.0), fur=0.02,
                 eye=dict(dir=(0.6, 0.4, 0.7), size=0.022, shape="almond", iris="#d9a32a", pupil="round", pupil_r=0.38),
                 coat=dict(base="#857b6e", belly="#ddd6c8", belly_edge=0.05, belly_dir=(0.4, -1.0, 0.0),
                           pattern="saddle", colours=["#463f38"], muzzle="#ddd6c8")),
    "cat": dict(trunk=0.72, chest=(0.25, 0.2, 0.14), chest_at=(-0.1, 0.02), belly=(0.3, 0.16, 0.14),
                rump=(0.2, 0.18, 0.15), loin=(0.27, 0.1, 0.12), f=(0.2, 0.22, 0.09), h=(0.26, 0.27, 0.17),
                fr=(0.075, 0.05, 0.034, 0.03), hr=(0.13, 0.06, 0.035, 0.03), paw=(0.048, 0.032, 0.04),
                neck_len=0.17, neck_r=(0.11, 0.09), neck_ang=40.0, neck_at=(0.03, 0.1),
                skull=(0.13, 0.116, 0.124), head_at=(0.055, 0.02), head_ang=-6.0,
                muzzle=0.08, muzzle_r=(0.066, 0.05), muzzle_at=(0.05, -0.045), cheek=0.075,
                eye=dict(dir=(0.78, 0.3, 0.55), size=0.032, shape="almond", iris="#b9c23a", pupil="slit",
                         pupil_r=0.42),
                ear=dict(kind="pointed", at=(-0.15, 0.85, 0.5), len=0.115, w=0.11, up=74.0, back=0.0, out=20.0),
                nose=dict(shape="cat", size=0.018, color="#c77a7a"), whiskers=7,
                tail=dict(kind="thin", len=0.62, r=(0.035, 0.026), n=9, ang=-30.0, curve=18.0), fur=0.008,
                gallop="rotary", spine_flex=0.14,
                coat=dict(base="#e3a465", belly="#f3e0c4", pattern="tabby", colours=["#bd7234"], belly_edge=0.2)),
    "lion": dict(trunk=0.74, f=(0.24, 0.27, 0.1), h=(0.28, 0.28, 0.16), fr=(0.1, 0.07, 0.05, 0.045),
                 hr=(0.15, 0.08, 0.05, 0.045), chest=(0.28, 0.25, 0.17), belly=(0.31, 0.18, 0.16), rump=(0.21, 0.2, 0.16),
                 neck_len=0.2, neck_r=(0.14, 0.11), neck_ang=35.0, skull=(0.14, 0.125, 0.125), muzzle=0.12,
                 muzzle_r=(0.095, 0.075), muzzle_at=(0.07, -0.045), cheek=0.0, muzzle_w=0.9, jaw=0.75, jowl=0.0, paw=(0.065, 0.045, 0.055),
                 eye=dict(dir=(0.72, 0.38, 0.58), size=0.022, shape="almond", iris="#c48a2a", pupil="round", pupil_r=0.36),
                 ear=dict(kind="round", at=(-0.03, 0.08, 0.07), len=0.06, w=0.07, up=70.0, back=0.0, out=25.0),
                 nose=dict(shape="cat", size=0.03, color="#8a4a3a"), whiskers=5,
                 tail=dict(kind="tuft", len=0.62, r=(0.04, 0.025), n=9, ang=-55.0, curve=20.0), fur=0.013,
                 mane=dict(kind="lion", color="#7a4420", size=1.0),
                 coat=dict(base="#d4a25a", belly="#ead2a6", belly_edge=0.1, tail_tip="#4a2e1c", muzzle="#ecdcc0", mane2="#9a5a2a")),
    "tiger": dict(trunk=0.76, f=(0.24, 0.27, 0.1), h=(0.28, 0.28, 0.16), fr=(0.1, 0.07, 0.05, 0.045),
                  hr=(0.15, 0.08, 0.05, 0.045), chest=(0.28, 0.24, 0.17), belly=(0.32, 0.18, 0.16), rump=(0.21, 0.2, 0.16),
                  neck_len=0.2, neck_r=(0.14, 0.11), neck_ang=32.0, skull=(0.14, 0.125, 0.13), muzzle=0.11,
                  muzzle_r=(0.09, 0.07), muzzle_at=(0.07, -0.045), cheek=0.085, muzzle_w=0.95, jaw=0.7, jowl=0.0, paw=(0.065, 0.045, 0.055),
                  eye=dict(dir=(0.72, 0.38, 0.58), size=0.02, shape="almond", iris="#d8b23a", pupil="round", pupil_r=0.36),
                  ear=dict(kind="round", at=(-0.03, 0.08, 0.075), len=0.055, w=0.065, up=70.0, back=0.0, out=25.0),
                  nose=dict(shape="cat", size=0.03, color="#b06a5a"), whiskers=6,
                  tail=dict(kind="thin", len=0.62, r=(0.045, 0.032), n=9, ang=-45.0, curve=22.0), fur=0.008,
                  coat=dict(base="#e0782a", belly="#f6efe2", belly_edge=0.05, pattern="tiger", colours=["#1d1410"],
                            muzzle="#f6efe2", tail_tip="#1d1410")),
    "horse": dict(trunk=0.72, chest=(0.27, 0.24, 0.15), chest_at=(-0.08, 0.02), belly=(0.32, 0.2, 0.165),
                  belly_at=(0.0, -0.03), rump=(0.25, 0.24, 0.17), rump_at=(0.06, 0.05), loin=(0.28, 0.12, 0.13),
                  withers=0.06, f=(0.25, 0.3, 0.22), h=(0.3, 0.34, 0.27), fr=(0.105, 0.062, 0.037, 0.034),
                  hr=(0.16, 0.075, 0.04, 0.034), foot="hoof", paw=(0.042, 0.05, 0.042), stance=(0.02, -0.02),
                  width=0.085, mu=0.0, nu=4.0, reach_f=0.97, reach_h=0.9,
                  neck_len=0.5, neck_r=(0.16, 0.09), neck_ang=52.0, neck_at=(0.02, 0.12), neck_arch=0.08,
                  skull=(0.13, 0.115, 0.09), head_at=(0.06, -0.02), head_ang=-50.0,
                  muzzle=0.3, muzzle_r=(0.095, 0.074), muzzle_at=(0.0, -0.01), muzzle_drop=-4.0, jowl=0.1,
                  muzzle_w=0.85, jaw=0.0,
                  eye=dict(dir=(0.25, 0.42, 0.88), size=0.024, shape="almond", iris="#2a1a12", pupil="bar", pupil_r=0.5),
                  ear=dict(kind="horse", at=(-0.035, 0.07, 0.035), len=0.1, w=0.05, up=78.0, back=-8.0, out=8.0),
                  nose=dict(shape="nostril", size=0.018, color="#2a1e1a"),
                  tail=dict(kind="horse", len=0.56, r=(0.05, 0.06), n=11, ang=-38.0, curve=-5.0), fur=0.0,
                  mane=dict(kind="horse", color="#2a1a12"), gallop="transverse", spine_flex=0.02,
                  coat=dict(base="#8a4f2a", belly="#9a5c34", muzzle="#5e3420", hoof="#4a3e36", tail_tip="#2a1a12")),
    "zebra": dict(base="horse", f=(0.23, 0.28, 0.2), h=(0.28, 0.31, 0.24), neck_len=0.38, neck_r=(0.15, 0.08),
                  ear=dict(kind="horse", at=(-0.035, 0.07, 0.04), len=0.12, w=0.065, up=78.0, back=-8.0, out=10.0),
                  tail=dict(kind="tuft", len=0.45, r=(0.035, 0.02), n=7, ang=-70.0, curve=8.0),
                  mane=dict(kind="brush", color="#1a1614"),
                  coat=dict(base="#f4f1ea", belly="#f4f1ea", pattern="zebra", colours=["#161210"], muzzle="#2a2420",
                            hoof="#2a2420", tail_tip="#161210")),
    "cow": dict(trunk=0.78, chest=(0.3, 0.28, 0.2), chest_at=(-0.08, -0.01), belly=(0.36, 0.26, 0.22),
                belly_at=(0.0, -0.06), rump=(0.24, 0.24, 0.2), rump_at=(0.05, 0.03), loin=(0.32, 0.12, 0.17),
                f=(0.2, 0.24, 0.16), h=(0.24, 0.27, 0.2), fr=(0.11, 0.07, 0.045, 0.038), hr=(0.16, 0.085, 0.045, 0.038),
                foot="cloven", paw=(0.045, 0.045, 0.05), width=0.12, mu=0.0, nu=4.0,
                neck_len=0.25, neck_r=(0.17, 0.1), neck_ang=22.0, neck_at=(0.04, 0.09), skull=(0.09, 0.09, 0.085),
                head_at=(0.06, -0.03), head_ang=-50.0, muzzle=0.22, muzzle_r=(0.08, 0.075), muzzle_at=(0.01, -0.01),
                eye=dict(dir=(0.3, 0.45, 0.85), size=0.024, shape="round", iris="#2a1a12", pupil="bar", pupil_r=0.5),
                ear=dict(kind="side", at=(-0.04, 0.035, 0.08), len=0.11, w=0.06, up=-5.0, back=10.0, out=60.0),
                nose=dict(shape="muzzle", size=0.05, color="#e8a8a0"), muzzle_w=0.9, jaw=0.0, jowl=0.0,
                horns=dict(kind="cow", len=0.15, r=0.026, color="#e8dcc4"), udder=0.07,
                tail=dict(kind="tuft", len=0.55, r=(0.035, 0.02), n=7, ang=-82.0, curve=4.0), fur=0.0,
                gallop="transverse", spine_flex=0.02,
                coat=dict(base="#f6f2ea", belly="#f6f2ea", pattern="patches", colours=["#1e1a18"], amount=0.55,
                          hoof="#3a322c", tail_tip="#1e1a18", muzzle="#f0c8c0")),
    "sheep": dict(trunk=0.7, chest=(0.3, 0.27, 0.21), belly=(0.33, 0.25, 0.22), rump=(0.25, 0.25, 0.21),
                  loin=(0.3, 0.14, 0.18), f=(0.15, 0.2, 0.12), h=(0.18, 0.22, 0.15), fr=(0.06, 0.04, 0.028, 0.025),
                  hr=(0.08, 0.045, 0.028, 0.025), foot="cloven", paw=(0.03, 0.035, 0.035), width=0.11, mu=0.0, nu=4.0,
                  neck_len=0.2, neck_r=(0.15, 0.08), neck_ang=40.0, skull=(0.08, 0.08, 0.07), head_at=(0.05, -0.01),
                  head_ang=-42.0, muzzle=0.15, muzzle_r=(0.055, 0.045), muzzle_at=(0.01, -0.02),
                  eye=dict(dir=(0.35, 0.45, 0.82), size=0.02, shape="round", iris="#c8a24a", pupil="bar", pupil_r=0.5),
                  ear=dict(kind="side", at=(-0.03, 0.03, 0.065), len=0.09, w=0.045, up=-15.0, back=15.0, out=70.0),
                  nose=dict(shape="nostril", size=0.012, color="#2a2420"),
                  tail=dict(kind="stub", len=0.1, r=(0.05, 0.04), n=2, ang=-60.0, curve=0.0), fur=0.0, fluff=0.045,
                  wool=True, gallop="transverse", spine_flex=0.02,
                  coat=dict(base="#f2ede2", belly="#f2ede2", face="#2a2420", legs="#2a2420", hoof="#1e1a18")),
    "pig": dict(trunk=0.72, chest=(0.3, 0.25, 0.2), chest_at=(-0.08, -0.02), belly=(0.36, 0.25, 0.21),
                belly_at=(0.0, -0.05), rump=(0.24, 0.23, 0.2), loin=(0.32, 0.13, 0.17), f=(0.1, 0.11, 0.06),
                h=(0.12, 0.13, 0.07), fr=(0.08, 0.05, 0.035, 0.03), hr=(0.12, 0.06, 0.035, 0.03), foot="cloven",
                paw=(0.03, 0.03, 0.035), width=0.11, mu=0.0, nu=4.0, neck_len=0.1, neck_r=(0.2, 0.15), neck_ang=15.0,
                neck_at=(0.05, 0.02), skull=(0.12, 0.12, 0.11), head_at=(0.05, -0.01), head_ang=-12.0, muzzle=0.17,
                muzzle_r=(0.09, 0.065), muzzle_at=(0.02, -0.04), jowl=0.09,
                eye=dict(dir=(0.55, 0.55, 0.62), size=0.016, shape="round", iris="#3a2a1a", pupil="round", pupil_r=0.5),
                ear=dict(kind="pig", at=(-0.02, 0.09, 0.07), len=0.11, w=0.09, up=40.0, back=-40.0, out=30.0),
                nose=dict(shape="snout", size=0.055, color="#e08a90"),
                tail=dict(kind="curl", len=0.14, r=(0.018, 0.012), n=8, ang=20.0, curve=-80.0), fur=0.0,
                coat=dict(base="#f2b0b0", belly="#f6c4c0", hoof="#c88a80")),
    "goat": dict(trunk=0.7, chest=(0.25, 0.23, 0.15), belly=(0.3, 0.2, 0.16), rump=(0.2, 0.19, 0.15),
                 f=(0.19, 0.24, 0.15), h=(0.22, 0.26, 0.18), fr=(0.07, 0.05, 0.03, 0.026), hr=(0.11, 0.06, 0.03, 0.026),
                 foot="cloven", paw=(0.032, 0.035, 0.038), mu=0.0, nu=4.0, neck_len=0.25, neck_r=(0.11, 0.07),
                 neck_ang=58.0, skull=(0.08, 0.08, 0.07), head_at=(0.05, -0.01), head_ang=-38.0, muzzle=0.17,
                 muzzle_r=(0.055, 0.04), muzzle_at=(0.0, -0.015),
                 eye=dict(dir=(0.35, 0.45, 0.82), size=0.022, shape="round", iris="#d8a22a", pupil="bar", pupil_r=0.55),
                 ear=dict(kind="side", at=(-0.03, 0.03, 0.06), len=0.1, w=0.04, up=-25.0, back=20.0, out=65.0),
                 nose=dict(shape="nostril", size=0.012, color="#2a2420"), beard=0.09,
                 horns=dict(kind="goat", len=0.22, r=0.022, color="#8a7a66"),
                 tail=dict(kind="stub", len=0.1, r=(0.03, 0.02), n=2, ang=50.0, curve=0.0), fur=0.008,
                 coat=dict(base="#ece6da", belly="#ece6da", hoof="#3a322c")),
    "deer": dict(trunk=0.7, chest=(0.25, 0.23, 0.14), belly=(0.3, 0.19, 0.15), rump=(0.2, 0.19, 0.145),
                 f=(0.24, 0.3, 0.22), h=(0.28, 0.33, 0.27), fr=(0.075, 0.048, 0.026, 0.022), hr=(0.12, 0.06, 0.028, 0.022),
                 foot="cloven", paw=(0.026, 0.035, 0.03), mu=0.0, nu=4.0, width=0.08, neck_len=0.34, neck_r=(0.11, 0.07),
                 neck_ang=60.0, skull=(0.08, 0.075, 0.07), head_at=(0.05, -0.01), head_ang=-30.0, muzzle=0.18,
                 muzzle_r=(0.055, 0.035), muzzle_at=(0.0, -0.015),
                 eye=dict(dir=(0.35, 0.45, 0.82), size=0.026, shape="almond", iris="#1e140e", pupil="round",
                          pupil_r=0.55),
                 ear=dict(kind="deer", at=(-0.035, 0.05, 0.05), len=0.13, w=0.075, up=40.0, back=10.0, out=40.0),
                 nose=dict(shape="dog", size=0.022, color="#1e1614"),
                 antlers=dict(len=0.32, tines=3, color="#cdb796"), gallop="transverse", spine_flex=0.04,
                 tail=dict(kind="stub", len=0.1, r=(0.035, 0.03), n=2, ang=20.0, curve=0.0), fur=0.004,
                 coat=dict(base="#a8693a", belly="#efe2cc", belly_edge=0.3, tail_tip="#f4ece0", hoof="#2a201a",
                           muzzle="#3a2a22")),
    "bear": dict(trunk=0.74, chest=(0.32, 0.3, 0.22), chest_at=(-0.1, 0.03), belly=(0.34, 0.24, 0.21),
                 rump=(0.25, 0.25, 0.21), rump_at=(0.04, 0.06), loin=(0.3, 0.14, 0.18), hump=0.08,
                 f=(0.21, 0.23, 0.07), h=(0.25, 0.25, 0.07), fr=(0.13, 0.1, 0.075, 0.07), hr=(0.17, 0.11, 0.075, 0.07),
                 foot="plantigrade", paw=(0.08, 0.05, 0.07), width=0.13, nu=70.0, mu=5.0, reach_h=0.9,
                 neck_len=0.18, neck_r=(0.17, 0.12), neck_ang=15.0, neck_at=(0.05, 0.08), skull=(0.12, 0.11, 0.115),
                 head_at=(0.05, -0.01), head_ang=-14.0, muzzle=0.14, muzzle_r=(0.08, 0.055), muzzle_at=(0.04, -0.03),
                 eye=dict(dir=(0.7, 0.45, 0.58), size=0.015, shape="round", iris="#2a1a10", pupil="round", pupil_r=0.6),
                 ear=dict(kind="round", at=(-0.03, 0.085, 0.075), len=0.055, w=0.06, up=70.0, back=0.0, out=30.0),
                 nose=dict(shape="dog", size=0.035, color="#141010"),
                 tail=dict(kind="stub", len=0.06, r=(0.04, 0.03), n=2, ang=-20.0, curve=0.0), fur=0.02,
                 gallop="rotary", spine_flex=0.04,
                 coat=dict(base="#6a4428", belly="#5a3a22", muzzle="#a07a52")),
    "elephant": dict(trunk=0.75, chest=(0.32, 0.33, 0.24), chest_at=(-0.08, 0.02), belly=(0.36, 0.3, 0.25),
                     belly_at=(0.0, -0.04), rump=(0.26, 0.3, 0.23), rump_at=(0.05, 0.02), loin=(0.32, 0.16, 0.2),
                     f=(0.3, 0.3, 0.06), h=(0.3, 0.32, 0.07), fr=(0.13, 0.11, 0.1, 0.1), hr=(0.16, 0.12, 0.1, 0.1),
                     foot="pad", paw=(0.1, 0.06, 0.1), width=0.14, mu=0.0, nu=0.0, reach_f=0.985, reach_h=0.96,
                     neck_len=0.16, neck_r=(0.22, 0.2), neck_ang=25.0, neck_at=(0.18, 0.1), skull=(0.17, 0.19, 0.16),
                     head_at=(0.08, 0.02), head_ang=-70.0, muzzle=0.12, muzzle_r=(0.11, 0.08), muzzle_at=(0.05, -0.08),
                     eye=dict(dir=(0.55, 0.3, 0.8), size=0.016, shape="almond", iris="#3a2414", pupil="round",
                              pupil_r=0.55),
                     ear=dict(kind="elephant", at=(-0.06, 0.06, 0.13), len=0.42, w=0.4, up=-70.0, back=30.0, out=8.0),
                     nose=None, mouth=False, trunk_len=0.62, tusks=0.2,
                     tail=dict(kind="tuft", len=0.36, r=(0.03, 0.015), n=6, ang=-80.0, curve=4.0), fur=0.0,
                     gallop="amble", spine_flex=0.0,
                     coat=dict(base="#8f8c88", belly="#8f8c88", pattern="wrinkle", hoof="#d6cfc2")),
    "giraffe": dict(trunk=0.62, chest=(0.25, 0.25, 0.15), chest_at=(-0.04, 0.04), belly=(0.28, 0.2, 0.15),
                    rump=(0.19, 0.18, 0.14), rump_at=(0.04, 0.02), loin=(0.25, 0.1, 0.12), withers=0.08,
                    f=(0.33, 0.42, 0.34), h=(0.33, 0.42, 0.36), fr=(0.085, 0.055, 0.032, 0.028),
                    hr=(0.13, 0.065, 0.034, 0.028), foot="cloven", paw=(0.035, 0.045, 0.04), mu=0.0, nu=4.0,
                    reach_f=0.985, reach_h=0.92, width=0.09,
                    neck_len=1.05, neck_r=(0.12, 0.055), neck_ang=66.0, neck_at=(0.02, 0.14), neck_arch=-0.02,
                    skull=(0.075, 0.065, 0.06), head_at=(0.04, -0.02), head_ang=-28.0, muzzle=0.17,
                    muzzle_r=(0.055, 0.042), muzzle_at=(0.0, -0.01),
                    eye=dict(dir=(0.3, 0.45, 0.85), size=0.024, shape="almond", iris="#1e140e", pupil="round",
                             pupil_r=0.55),
                    ear=dict(kind="side", at=(-0.04, 0.04, 0.05), len=0.1, w=0.04, up=10.0, back=15.0, out=55.0),
                    ossicones=0.08, nose=dict(shape="nostril", size=0.012, color="#3a2a20"),
                    mane=dict(kind="brush", color="#7a4a24"), gallop="transverse", spine_flex=0.02,
                    tail=dict(kind="tuft", len=0.5, r=(0.03, 0.015), n=7, ang=-78.0, curve=6.0), fur=0.0,
                    coat=dict(base="#f0dcb0", belly="#f2e4c4", pattern="giraffe", colours=["#a2582a"], scale=1.0,
                              hoof="#3a2a20", tail_tip="#2a1a12")),
    "rabbit": dict(trunk=0.6, chest=(0.25, 0.24, 0.17), chest_at=(-0.04, -0.02), belly=(0.3, 0.24, 0.19),
                   belly_at=(0.0, -0.02), rump=(0.27, 0.27, 0.2), rump_at=(0.04, 0.05), loin=(0.25, 0.14, 0.15),
                   f=(0.14, 0.16, 0.06), h=(0.2, 0.22, 0.24), fr=(0.05, 0.035, 0.028, 0.026), hr=(0.16, 0.06, 0.04, 0.035),
                   foot="plantigrade", paw=(0.045, 0.03, 0.035), nu=75.0, mu=8.0, width=0.1, reach_h=0.82,
                   neck_len=0.1, neck_r=(0.14, 0.12), neck_ang=50.0, neck_at=(0.04, 0.12), skull=(0.13, 0.11, 0.1),
                   head_at=(0.05, 0.0), head_ang=-12.0, muzzle=0.08, muzzle_r=(0.075, 0.055), muzzle_at=(0.05, -0.04),
                   cheek=0.06, eye=dict(dir=(0.45, 0.45, 0.78), size=0.032, shape="round", iris="#2a1a12",
                                        pupil="round", pupil_r=0.62),
                   ear=dict(kind="long", at=(-0.04, 0.09, 0.04), len=0.32, w=0.09, up=70.0, back=24.0, out=20.0),
                   nose=dict(shape="cat", size=0.016, color="#c07a7a"), whiskers=4,
                   tail=dict(kind="puff", len=0.08, r=(0.07, 0.07), n=2, ang=20.0, curve=0.0), fur=0.012,
                   gallop="bound", spine_flex=0.12,
                   coat=dict(base="#b08a68", belly="#efe6da", belly_edge=0.1, tail_tip="#f4efe8", inner_ear="#e8b0a8")),
    "mouse": dict(trunk=0.6, sit_upright=True, chest=(0.25, 0.22, 0.17), chest_at=(-0.05, -0.02), belly=(0.3, 0.22, 0.18),
                  rump=(0.25, 0.24, 0.19), loin=(0.26, 0.12, 0.15), f=(0.09, 0.1, 0.05), h=(0.12, 0.13, 0.12),
                  fr=(0.04, 0.025, 0.02, 0.018), hr=(0.1, 0.04, 0.022, 0.02), foot="plantigrade",
                  paw=(0.035, 0.018, 0.022), nu=70.0, mu=10.0, width=0.11, neck_len=0.08, neck_r=(0.15, 0.12),
                  neck_ang=25.0, neck_at=(0.05, 0.06), skull=(0.13, 0.11, 0.11), head_at=(0.06, 0.0), head_ang=-10.0,
                  muzzle=0.17, muzzle_r=(0.09, 0.02), muzzle_at=(0.04, -0.03),
                  eye=dict(dir=(0.5, 0.4, 0.75), size=0.03, shape="round", iris="#100a08", pupil="round", pupil_r=0.7),
                  ear=dict(kind="mouse", at=(-0.03, 0.085, 0.075), len=0.15, w=0.15, up=60.0, back=10.0, out=30.0),
                  nose=dict(shape="cat", size=0.016, color="#e08a90"), whiskers=5,
                  tail=dict(kind="thin", len=0.9, r=(0.022, 0.01), n=10, ang=-5.0, curve=4.0), fur=0.006,
                  gallop="bound", spine_flex=0.1,
                  coat=dict(base="#9a8e84", belly="#e6ded4", belly_edge=0.2, inner_ear="#f0b8b4", tail="#e0b8b0",
                            feet="#f0c8c0")),
    "squirrel": dict(trunk=0.6, sit_upright=True, chest=(0.23, 0.21, 0.15), belly=(0.28, 0.19, 0.15), rump=(0.24, 0.24, 0.17),
                     loin=(0.25, 0.11, 0.13), f=(0.12, 0.13, 0.06), h=(0.16, 0.17, 0.17), fr=(0.045, 0.03, 0.024, 0.022),
                     hr=(0.13, 0.05, 0.03, 0.026), foot="plantigrade", paw=(0.04, 0.022, 0.026), nu=72.0, mu=8.0,
                     neck_len=0.1, neck_r=(0.13, 0.11), neck_ang=40.0, neck_at=(0.05, 0.08), skull=(0.115, 0.1, 0.095),
                     head_at=(0.05, 0.0), head_ang=-8.0, muzzle=0.09, muzzle_r=(0.07, 0.04), muzzle_at=(0.04, -0.03),
                     cheek=0.05, eye=dict(dir=(0.5, 0.45, 0.75), size=0.03, shape="round", iris="#100a08",
                                          pupil="round", pupil_r=0.7),
                     ear=dict(kind="pointed", at=(-0.03, 0.085, 0.055), len=0.1, w=0.06, up=78.0, back=8.0, out=10.0),
                     nose=dict(shape="cat", size=0.014, color="#3a2420"), whiskers=4,
                     tail=dict(kind="squirrel", len=0.95, r=(0.06, 0.09), n=10, ang=88.0, curve=-15.0), fur=0.014,
                     gallop="bound", spine_flex=0.12,
                     coat=dict(base="#b0582a", belly="#f2e6d6", belly_edge=0.05, belly_dir=(0.4, -1.0, 0.0))),
}
_SPECIES_Q["golden"] = dict(_SPECIES_Q["labrador"], fur=0.019, skull=(0.125, 0.11, 0.105), muzzle=0.15,
                            ear=dict(kind="floppy", at=(-0.03, 0.06, 0.075), len=0.17, w=0.1, up=-62.0, back=18.0,
                                     out=10.0),
                            tail=dict(kind="bushy", len=0.5, r=(0.045, 0.03), n=8, ang=-32.0, curve=9.0),
                            coat=dict(base="#d29a4e", belly="#ebc98e", pattern="solid", belly_edge=0.25,
                                      muzzle="#dcaa62"))
_SPECIES_Q["dog"] = dict(_SPECIES_Q["labrador"], coat=dict(base="#b67c44", belly="#e4c49a", pattern="solid",
                                                            belly_edge=0.35))

_ALIASES = {"labrador retriever": "labrador", "lab": "labrador", "retriever": "labrador",
            "golden retriever": "golden", "golden": "golden", "golden lab": "labrador",
            "german shepherd": "shepherd", "alsatian": "shepherd", "puppy": "dog", "kitten": "cat", "kitty": "cat",
            "pony": "horse", "foal": "horse", "stag": "deer", "doe": "deer", "fawn": "deer", "bunny": "rabbit",
            "hare": "rabbit", "rat": "mouse", "piglet": "pig", "lamb": "sheep", "calf": "cow", "bull": "cow",
            "ox": "cow", "lioness": "lion", "kid": "goat", "reindeer": "deer"}

COATS = {
    "tabby": dict(base="#a8865f", belly="#e8d8be", pattern="tabby", colours=["#4f3a29"], belly_edge=0.2,
                  muzzle="#d9c4a4"),
    "grey tabby": dict(base="#a8a49c", belly="#e8e4dc", pattern="tabby", colours=["#4a4640"], belly_edge=0.2),
    "black": dict(base="#2a2624", belly="#2a2624", pattern="solid"),
    "white": dict(base="#f4f0ea", belly="#f4f0ea", pattern="solid"),
    "siamese": dict(base="#efe2cc", belly="#f4ead8", pattern="points", points="#4a3428"),
    "calico": dict(base="#f4efe6", belly="#f4efe6", pattern="patches", colours=["#d68a3a", "#2a2422"], amount=0.6),
    "tuxedo": dict(base="#2a2624", belly="#f4f0ea", belly_edge=-0.25, belly_dir=(0.6, -1.0, 0.0), muzzle="#f4f0ea"),
    "ginger": dict(base="#e0904a", belly="#f2d4b0", pattern="tabby", colours=["#b8642a"], belly_edge=0.2),
    "dalmatian": dict(base="#f4f1ea", belly="#f4f1ea", pattern="spots", colours=["#1e1a18"], spot=0.24, scale=1.2),
    "chocolate": dict(base="#5a3a24", belly="#6a4630", pattern="solid"),
    "black lab": dict(base="#24201e", belly="#2a2624", pattern="solid"),
    "grey": dict(base="#8a8680", belly="#cfcac2", pattern="solid", belly_edge=0.2),
    "palomino": dict(base="#d9b06a", belly="#e4c48a", mane="#f2e8d2", tail="#f2e8d2"),
    "black horse": dict(base="#24201e", belly="#2a2624", mane="#141210", tail="#141210"),
    "white horse": dict(base="#eeeae4", belly="#e8e2da", mane="#f4f2ee", tail="#f4f2ee", muzzle="#8a8078"),
    "fawn": dict(base="#b07a44", belly="#efe2cc", pattern="spots", colours=["#f4ece0"], spot=0.2, scale=0.9),
    "polar": dict(base="#f2efe6", belly="#f2efe6", muzzle="#f2efe6"),
    "panda": dict(base="#f4f2ee", belly="#f4f2ee", pattern="solid", socks=(0.45, "#1a1818"), ear_back="#1a1818"),
    "leopard": dict(base="#e0b060", belly="#f4ead6", pattern="rosettes", colours=["#2a1e14"], scale=1.0),
    "cheetah": dict(base="#e4b866", belly="#f6ecd8", pattern="spots", colours=["#1e1610"], spot=0.2, scale=1.3),
}


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "coat":
            d = dict(out[k])
            d.update(v)
            out[k] = d
        else:
            out[k] = v
    return out


def _qparams(name, style="soft", over=None):
    sp = _SPECIES_Q[name]
    P = dict(_Q)
    if "base" in sp:
        P = _merge(P, {k: v for k, v in _SPECIES_Q[sp["base"]].items() if k != "base"})
    P = _merge(P, {k: v for k, v in sp.items() if k != "base"})
    if over:
        P = _merge(P, over)
    if style == "cute":
        f, h = P["f"], P["h"]
        P["f"] = (f[0] * 0.8, f[1] * 0.72, f[2] * 0.8)
        P["h"] = (h[0] * 0.8, h[1] * 0.72, h[2] * 0.8)
        P["fr"] = tuple(r * 1.25 for r in P["fr"])
        P["hr"] = tuple(r * 1.12 for r in P["hr"])
        P["paw"] = tuple(r * 1.3 for r in P["paw"])
        hs = 1.4
        P["skull"] = tuple(r * hs for r in P["skull"])
        P["muzzle"] *= 0.78
        P["muzzle_r"] = tuple(r * 1.18 for r in P["muzzle_r"])
        P["head_at"] = (P["head_at"][0] * hs, P["head_at"][1] * hs)
        P["muzzle_at"] = (P["muzzle_at"][0] * hs, P["muzzle_at"][1] * hs)
        P["cheek"] *= hs
        P["jowl"] *= hs
        e = dict(P["eye"])
        e["size"] *= 2.1
        e["shape"] = "round"
        e["pupil"] = "round"
        e["pupil_r"] = 0.66
        P["eye"] = e
        ear = dict(P["ear"])
        ear["at"] = tuple(v * hs for v in ear["at"])
        ear["len"] *= 1.25
        ear["w"] *= 1.3
        P["ear"] = ear
        P["neck_len"] *= 0.78
        P["chest"] = tuple(r * 1.06 for r in P["chest"])
        P["rump"] = tuple(r * 1.06 for r in P["rump"])
        P["trunk"] *= 0.88
    return P


# ===================================================================== quadrupeds: skeleton and poses


def _foot_chain(P, front, ang, flex=0.0):
    """Points below the wrist / hock (wrist at the origin) for a meta angle ``ang`` (deg from vertical, + = foot
    forward) and swing flexion ``flex`` (deg). Returns (points, ground contact)."""
    m = P["f"][2] if front else P["h"][2]
    foot = P.get("foot_f" if front else "foot_h") or P["foot"]
    d = np.array([math.sin(math.radians(ang)), -math.cos(math.radians(ang))])
    A = m * d
    rx, ry, rz = P["paw"]
    if foot in ("hoof", "cloven"):
        lp = ry * 0.9
        ap = ang + 38 - flex * 0.55
        K = A + lp * np.array([math.sin(math.radians(ap)), -math.cos(math.radians(ap))])
        ah = ang + 14 - flex * 0.6
        B = K + ry * np.array([math.sin(math.radians(ah)), -math.cos(math.radians(ah))])
        return [A, K, B], B
    if foot == "pad":
        B = A + np.array([0.0, -ry * 1.4])
        return [A, B], B
    # paws: the toe pad sits just in front of the end of the metacarpus / metatarsus
    ta = -flex * 0.7
    C = A + _rot2(np.array([rx * 0.5, 0.0]), ta)
    return [A, C], C - np.array([0.0, ry * 0.92])


def _leg_joints(P, front, root, W):
    l1, l2 = (P["f"][0], P["f"][1]) if front else (P["h"][0], P["h"][1])
    mid, W2 = _ik2(root, W, l1, l2, side=-1 if front else 1)
    return mid, W2


def _leg(P, front, root, F=None, ang=None, flex=0.0, W=None):
    """Solve one leg: ground contact F (or wrist / hock W) → dict of 2D joints."""
    ang = (P["mu"] if front else P["nu"]) if ang is None else ang
    pts, contact = _foot_chain(P, front, ang, flex)
    if W is None:
        W = np.asarray(F, np.float64) - contact
    mid, W2 = _leg_joints(P, front, np.asarray(root, np.float64), W)
    return dict(root=np.asarray(root, np.float64), mid=mid, W=W2, foot=[W2 + p for p in pts], ang=ang, flex=flex,
                contact=W2 + contact)


def _stand_heights(P):
    """Shoulder and hip joint heights for a relaxed stand, and their x positions."""
    tr = P["trunk"]
    Sx, Hx = tr / 2, -tr / 2
    _, cf = _foot_chain(P, True, P["mu"])
    _, ch = _foot_chain(P, False, P["nu"])
    Wf = np.array([Sx + P["stance"][0], 0.0]) - cf
    Wh = np.array([Hx + P["stance"][1], 0.0]) - ch
    rf = P["reach_f"] * (P["f"][0] + P["f"][1])
    rh = P["reach_h"] * (P["h"][0] + P["h"][1])
    Sy = Wf[1] + math.sqrt(max(rf * rf - (Sx - Wf[0]) ** 2, 1e-6))
    Hy = Wh[1] + math.sqrt(max(rh * rh - (Hx - Wh[0]) ** 2, 1e-6))
    return Sx, Sy, Hx, Hy


def _spine(P, Sy, Hy, cx=0.0):
    """Place shoulder and hip joints ``trunk`` apart at the given heights, centred on x = cx."""
    tr = P["trunk"]
    dy = Sy - Hy
    dx = math.sqrt(max(tr * tr - dy * dy, 1e-6))
    return np.array([cx + dx / 2, Sy]), np.array([cx - dx / 2, Hy]), math.degrees(math.atan2(dy, dx))


POSES = ("stand", "walk", "trot", "gallop", "sit", "lie", "crouch", "jump", "graze", "sniff", "rear", "sleep",
         "perch", "fly", "swim", "glide", "peck", "open", "rest")


def _gait_leg(P, front, root, phase, off, duty, st, lift, flex_max, x0, quick=0.0):
    p = (phase - off) % 1.0
    ang0 = P["mu"] if front else P["nu"]
    _, c0 = _foot_chain(P, front, ang0)
    Wy0 = -c0[1]
    if p < duty:
        s = p / duty
        ang = ang0 + (14 - 40 * s * s if front else 10 - 34 * s * s)
        F = np.array([x0 + st * (0.5 - s), 0.0])
        leg = _leg(P, front, root, F=F, ang=ang)
        leg["planted"] = True
        return leg
    s = (p - duty) / (1 - duty)
    sm = s * s * (3 - 2 * s)
    # fast gaits bring the foot forward early (it folds under the body instead of trailing behind)
    sm = sm + (s * (2 - s) - sm) * quick
    # the leg folds right after lift-off and unfolds again before landing
    fl = flex_max * math.sin(math.pi * min(s ** 0.65 * 1.1, 1.0)) ** 0.7
    ang = ang0 - fl
    _, cc = _foot_chain(P, front, ang, fl)
    x = x0 + st * (-0.5 + sm)
    W = np.array([x - c0[0], Wy0 + lift * math.sin(math.pi * s ** 0.7)])
    leg = _leg(P, front, root, W=W, ang=ang, flex=fl)
    if leg["contact"][1] < 0.0:
        leg = _leg(P, front, root, W=W + np.array([0, -leg["contact"][1]]), ang=ang, flex=fl)
    leg["planted"] = False
    return leg


def _quad_pose(P, name, phase=0.0, opt=None):
    """Skeleton for a pose → dict(S, H, ca, pa, ma, neck, head, tail, legs, grounded)."""
    opt = opt or {}
    Sx, Sy, Hx, Hy = _stand_heights(P)
    Lf = P["f"][0] + P["f"][1] + P["f"][2]
    Lh = P["h"][0] + P["h"][1] + P["h"][2]
    tr = P["trunk"]
    w = P["width"]
    neck_ang, head_ang = P["neck_ang"], P["head_ang"]
    sk = dict(ca=0.0, pa=0.0, ma=0.0, grounded=True, tail=dict(P["tail"]), ears=opt.get("ears"),
              mouth=opt.get("mouth", "closed"), eyes=opt.get("eyes", "open"), pose=name)
    legs = {}
    tail = sk["tail"]
    kind = P.get("gallop", "rotary")

    if name in ("stand", "graze", "sniff", "look"):
        S, H, sp = _spine(P, Sy, Hy)
        fo = 0.03 if name == "graze" else 0.0
        legs["fl"] = _leg(P, True, S, F=(S[0] + P["stance"][0] + 0.05 + fo, 0))
        legs["fr"] = _leg(P, True, S, F=(S[0] + P["stance"][0] - 0.03, 0))
        legs["hl"] = _leg(P, False, H, F=(H[0] + P["stance"][1] - 0.03, 0))
        legs["hr"] = _leg(P, False, H, F=(H[0] + P["stance"][1] + 0.05, 0))
        if name == "graze":
            neck_ang, head_ang = -55.0 if P["neck_len"] > 0.2 else -20.0, -80.0
        elif name == "sniff":
            neck_ang, head_ang = min(neck_ang, 5.0), -35.0
    elif name in ("walk", "trot", "gallop", "run", "amble", "bound"):
        g = {"run": "gallop"}.get(name, name)
        bob = 0.0
        pitch = 0.0
        if g == "walk" or (g == "trot" and kind == "amble") or g == "amble":
            duty, st, lf, flf, flh = 0.64, 0.45 * Lf, 0.06 * Lf, 70.0, 40.0
            offs = dict(hl=0.0, fl=0.25, hr=0.5, fr=0.75)
            bob = 0.008 * math.cos(4 * math.pi * phase)
        elif g == "trot":
            duty, st, lf, flf, flh = 0.42, 0.7 * Lf, 0.14 * Lf, 105.0, 55.0
            offs = dict(fl=0.0, hr=0.0, fr=0.5, hl=0.5)
            bob = 0.02 * math.cos(4 * math.pi * phase) - 0.01
            neck_ang -= 8
        else:   # gallop / bound
            if kind == "transverse":
                offs = dict(hl=0.0, hr=0.08, fl=0.3, fr=0.38)
                duty, st, lf, flf, flh = 0.33, 0.8 * Lf, 0.22 * Lf, 130.0, 50.0
            elif kind == "bound":
                offs = dict(hl=0.0, hr=0.04, fl=0.45, fr=0.5)
                duty, st, lf, flf, flh = 0.25, 1.0 * Lf, 0.16 * Lf, 100.0, 60.0
            else:
                offs = dict(hl=0.0, hr=0.07, fr=0.36, fl=0.44)
                duty, st, lf, flf, flh = 0.28, 0.95 * Lf, 0.2 * Lf, 125.0, 75.0
            ph = phase
            # spine: extended in the flight after the hinds push off, gathered after the fronts
            ext = math.cos(2 * math.pi * (ph - 0.36))
            flex = P["spine_flex"]
            sk["ma"] = 0.0
            sk["pa"] = -14.0 * flex / 0.1 * (0.5 - 0.5 * ext)
            sk["ca"] = 6.0 * flex / 0.1 * (0.5 - 0.5 * ext)
            tr_s = tr * (1 + 0.6 * flex * ext)
            # highest in the gathered suspension after the leading foreleg leaves the ground
            susp = (offs["fl"] + offs.get("fr", 0) + 2 * duty) / 2 + 0.12
            bob = 0.05 * Lf * math.cos(2 * math.pi * (ph - susp)) + 0.03 * Lf
            pitch = 7.0 * math.sin(2 * math.pi * (ph - 0.05))
            neck_ang = (neck_ang - 12) if P["neck_len"] > 0.25 else neck_ang - 10
            head_ang = head_ang + (12 if P["neck_len"] > 0.25 else 0)
            sk["trunk_scale"] = tr_s / tr
        Sy2, Hy2 = Sy + bob, Hy + bob
        if pitch:
            Sy2 += math.sin(math.radians(pitch)) * tr / 2
            Hy2 -= math.sin(math.radians(pitch)) * tr / 2
        trs = sk.get("trunk_scale", 1.0)
        P2 = dict(P, trunk=tr * trs)
        S, H, sp = _spine(P2, Sy2, Hy2)
        for nm, off in offs.items():
            front = nm[0] == "f"
            root = S if front else H
            x0 = (S[0] + P["stance"][0] + 0.02) if front else (H[0] + P["stance"][1] - 0.02)
            big = g in ("gallop", "bound")
            legs[nm] = _gait_leg(P, front, root, phase, off, duty, st * (1.0 if front or not big else 0.85),
                                 lf * (1.0 if front or not big else 1.3), flf if front else flh, x0,
                                 quick=0.0 if not big else (0.35 if front else 0.75))
        if g in ("gallop", "bound"):
            sk["grounded"] = False
            tail = dict(tail, ang=max(tail["ang"], -10.0) if tail["kind"] not in ("plume", "curl", "stub", "puff",
                                                                                 "pompom") else tail["ang"],
                        curve=tail["curve"] * 0.3)
            # lift the whole body so that planted feet touch the ground
        sk["airborne_fix"] = True
    elif name == "sit":
        # haunches on the ground, spine flexed (shorter), chest lifted over straight front legs
        rr = P["rump"]
        Hy2 = rr[1] * 0.78
        trs = 0.74
        P2 = dict(P, trunk=tr * trs)
        # the back leans at most ~56 degrees: long-legged dogs then set their front paws a little forward
        Sy2 = min(Sy * 1.02, Hy2 + tr * trs * 0.95, Hy2 + tr * trs * math.sin(math.radians(56.0)))
        S, H, sp = _spine(P2, Sy2, Hy2, cx=0.0)
        sk["pa"] = 28.0
        sk["ca"] = 4.0
        sk["ma"] = 10.0
        sk["trunk_scale"] = trs
        sk["front_width"] = max(w, P["chest"][2] * 0.78)
        fx = S[0] + 0.02
        if P.get("sit_upright"):
            # rodents sit up on their haunches with the front paws held in front of the chest
            S, H, sp = _spine(P2, Hy2 + tr * trs * 0.9, Hy2, cx=0.0)
            lf_ = P["f"][0] + P["f"][1]
            legs["fl"] = _leg(P, True, S, W=S + _dir2(-35) * lf_ * 0.75, ang=150.0, flex=70)
            legs["fr"] = _leg(P, True, S, W=S + _dir2(-20) * lf_ * 0.72, ang=140.0, flex=80)
        else:
            _, cf = _foot_chain(P, True, P["mu"])
            reach = 0.96 * (P["f"][0] + P["f"][1])
            dy_ = S[1] + cf[1]
            fx = max(fx, S[0] + cf[0] + math.sqrt(max(reach * reach - dy_ * dy_, 0.0)) * 0.55)
            legs["fl"] = _leg(P, True, S, F=(fx + 0.015, 0), ang=P["mu"])
            legs["fr"] = _leg(P, True, S, F=(fx - 0.015, 0), ang=P["mu"])
            sk["belly_scale"] = (0.8, 0.82, 0.92)          # the belly tucks up behind the forelegs
        # hind legs folded: hock on the ground under the hip, the foot flat and forward
        for nm, dx in (("hl", 0.02), ("hr", -0.01)):
            hk = np.array([H[0] - 0.02 + dx, P["hr"][2] * 1.05])
            legs[nm] = _leg(P, False, H, W=hk, ang=88.0)
        # short-necked animals (cats) settle the head low on the shoulders
        neck_ang = max(P["neck_ang"], 35.0) + (30 if P["neck_len"] >= 0.2 else 14) - max(sp - 40, 0) * 0.6
        head_ang = P["head_ang"] + 4
        if tail["kind"] in ("squirrel", "plume", "curl", "stub", "puff", "pompom"):
            tail = dict(tail, ang=tail["ang"] + 15.0)
        else:
            tail = dict(tail, ang=-60.0, curve=-14.0, side=0.12, ground=True)
    elif name in ("lie", "sleep"):
        cr = P["chest"]
        Sy2 = cr[1] * 0.95
        Hy2 = P["rump"][1] * 0.8
        S, H, sp = _spine(P, Sy2, Hy2)
        sk["pa"] = 4.0
        # forelegs along the ground, paws forward
        for nm, dx in (("fl", 0.04), ("fr", 0.0)):
            E = np.array([S[0] - 0.02, P["fr"][1] * 0.9])
            Wp = E + np.array([P["f"][1] * 0.98, 0.0])
            Wp[1] = P["fr"][2]
            legs[nm] = dict(root=S, mid=E, W=Wp + [dx, 0], foot=[Wp + [dx + P["f"][2] * 0.8, -0.005],
                                                               Wp + [dx + P["f"][2] * 0.8 + P["paw"][0] * 0.5, -0.005]],
                            ang=90.0, flex=0.0, contact=Wp)
            ft = P.get("foot")
            if ft in ("hoof", "cloven"):
                # hoofed animals fold the forelegs under the chest
                Kn = np.array([S[0] - 0.0, P["fr"][2] * 1.2])
                legs[nm] = _leg(P, True, S, W=Kn, ang=170.0, flex=0.0)
        for nm in ("hl", "hr"):
            hk = np.array([H[0] - 0.05, P["hr"][2] * 1.1])
            legs[nm] = _leg(P, False, H, W=hk, ang=96.0)
        if name == "sleep":
            neck_ang, head_ang = -10.0, -10.0
            sk["eyes"] = "closed"
            tail = dict(tail, ang=-70.0, curve=25.0, side=0.12, ground=True)
        else:
            neck_ang = P["neck_ang"] - 5
            tail = dict(tail, ang=-60.0, curve=20.0, side=0.08, ground=True)
    elif name == "crouch":
        S, H, sp = _spine(P, Sy * 0.62, Hy * 0.6)
        legs["fl"] = _leg(P, True, S, F=(S[0] + 0.1, 0), ang=P["mu"] + 20)
        legs["fr"] = _leg(P, True, S, F=(S[0] + 0.02, 0), ang=P["mu"] + 15)
        legs["hl"] = _leg(P, False, H, F=(H[0] + 0.02, 0), ang=P["nu"] + 30)
        legs["hr"] = _leg(P, False, H, F=(H[0] + 0.08, 0), ang=P["nu"] + 34)
        neck_ang, head_ang = 8.0, -2.0
        sk["ca"] = 8.0
        tail = dict(tail, ang=-12.0, curve=4.0)
    elif name == "jump":
        lift = 0.55 * Lf
        S, H, sp = _spine(P, Sy + lift + 0.12 * tr, Hy + lift - 0.05)
        legs["fl"] = _leg(P, True, S, W=S + _dir2(-40) * (P["f"][0] + P["f"][1]) * 0.9, ang=60.0, flex=30)
        legs["fr"] = _leg(P, True, S, W=S + _dir2(-55) * (P["f"][0] + P["f"][1]) * 0.85, ang=30.0, flex=50)
        legs["hl"] = _leg(P, False, H, W=H + _dir2(-150) * (P["h"][0] + P["h"][1]) * 0.95, ang=-60.0)
        legs["hr"] = _leg(P, False, H, W=H + _dir2(-140) * (P["h"][0] + P["h"][1]) * 0.93, ang=-50.0)
        sk["grounded"] = False
        sk["pa"] = -6.0
        neck_ang = P["neck_ang"] - 10
        tail = dict(tail, ang=8.0 if tail["kind"] not in ("plume", "curl") else tail["ang"], curve=2.0)
    elif name == "rear":
        Hy2 = Hy * 0.85
        S, H, sp = _spine(P, Hy2 + tr * 0.78, Hy2)
        legs["hl"] = _leg(P, False, H, F=(H[0] + 0.08, 0), ang=P["nu"] + 10)
        legs["hr"] = _leg(P, False, H, F=(H[0] + 0.0, 0), ang=P["nu"] + 5)
        lf = P["f"][0] + P["f"][1]
        legs["fl"] = _leg(P, True, S, W=S + _dir2(-30) * lf * 0.7, ang=150.0, flex=90)
        legs["fr"] = _leg(P, True, S, W=S + _dir2(-5) * lf * 0.7, ang=120.0, flex=100)
        neck_ang = P["neck_ang"] + 10
        head_ang = P["head_ang"] + 15
        sk["pa"] = 15.0
        sk["grounded"] = True
        tail = dict(tail, ang=-70.0)
    else:
        raise _unknown_name("pose", name, POSES, "Try creature.POSES for the list")

    if "head_ang" in opt:
        head_ang = opt["head_ang"]
    if "neck_ang" in opt:
        neck_ang = opt["neck_ang"]
    if "tail" in opt:
        tv = opt["tail"]
        if isinstance(tv, (int, float)):
            tail = dict(tail, ang=float(tv))
        elif isinstance(tv, str):
            tail = dict(tail, **{"up": dict(ang=75.0, curve=-6.0), "down": dict(ang=-75.0, curve=8.0),
                                  "out": dict(ang=0.0, curve=4.0), "curl": dict(ang=60.0, curve=-35.0),
                                  "wag": dict(ang=40.0, curve=10.0, side=0.12)}.get(tv, {}))
        elif isinstance(tv, dict):
            tail = dict(tail, **tv)
    sk.update(S=S, H=H, sp=sp, legs=legs, neck_ang=neck_ang, head_ang=head_ang, tail=tail,
              head_yaw=opt.get("head_turn", 0.0), head_roll=opt.get("head_tilt", 0.0), width=w)
    return sk


# ===================================================================== quadrupeds: building the model


def _R2(deg):
    """3D rotation in the sagittal plane (about z) by deg (positive = nose up)."""
    return _rz(deg)


def _p3(p2, z=0.0):
    return np.array([p2[0], p2[1], z], np.float64)


def _ear_outline(kind):
    """Ear outline in ear coordinates (s across -0.5..0.5, t along 0..1)."""
    if kind in ("pointed", "fox"):
        return [(-0.5, 0.0), (-0.44, 0.3), (-0.28, 0.66), (-0.04, 1.0), (0.1, 0.78), (0.3, 0.45), (0.5, 0.0)]
    if kind == "horse":
        return [(-0.5, 0.0), (-0.48, 0.35), (-0.3, 0.75), (0.0, 1.0), (0.22, 0.72), (0.45, 0.35), (0.5, 0.0)]
    if kind in ("round", "mouse"):
        n = 11
        return [(0.5 * math.cos(math.pi * i / (n - 1)) * (1 if kind == "round" else 1.0),
                 (0.9 if kind == "round" else 1.0) * math.sin(math.pi * i / (n - 1)) + (0.0 if kind == "round" else 0.0))
                for i in range(n)][::-1] + ([(0.32, -0.05), (-0.32, -0.05)] if kind == "mouse" else [])
    if kind == "long":
        n = 12
        pts = [(0.5 * math.sin(math.pi * i / (n - 1)) * (0.65 + 0.35 * math.sin(math.pi * i / (n - 1))),
                i / (n - 1)) for i in range(n)]
        return [(-x, t) for x, t in pts] + [(x, t) for x, t in pts[::-1]][1:-1]
    if kind == "floppy":
        return [(-0.3, 0.0), (-0.5, 0.25), (-0.52, 0.62), (-0.36, 0.92), (-0.08, 1.0), (0.22, 0.95), (0.42, 0.7),
                (0.45, 0.3), (0.3, 0.0)]
    if kind in ("side", "deer"):
        return [(-0.38, 0.0), (-0.5, 0.3), (-0.42, 0.65), (-0.12, 0.95), (0.0, 1.0), (0.14, 0.92), (0.44, 0.6),
                (0.5, 0.3), (0.38, 0.0)]
    if kind == "pig":
        return [(-0.5, 0.0), (-0.3, 0.6), (0.05, 1.0), (0.32, 0.55), (0.5, 0.0)]
    if kind == "elephant":
        return [(-0.25, 0.0), (-0.48, 0.12), (-0.55, 0.45), (-0.42, 0.8), (-0.12, 1.0), (0.2, 0.9), (0.42, 0.6),
                (0.5, 0.25), (0.35, 0.0)]
    return [(-0.5, 0.0), (0.0, 1.0), (0.5, 0.0)]


def _build_quad(P, sk, coat, style, seed=0):
    m = _Model()
    m.coat = coat
    base = coat.get("base", "#b08050")
    belly_on = 1.0 if coat.get("belly") is not None else 0.0
    m.fur_len = P["fur"] * (0.0 if style != "soft" else 1.0)
    fluff = P.get("fluff", 0.0)
    m.fluff = fluff if style == "soft" else fluff * 0.9
    wool = P.get("wool", False) or fluff > 0
    rough = 1.0 if wool else 0.0
    S, H = sk["S"], sk["H"]
    sp = sk["sp"]
    ca, pa, ma = sp + sk["ca"], sp + sk["pa"], sp + sk["ma"]
    mid = (S + H) / 2
    w = sk["width"]
    pts_pat = 0.0 if coat.get("pattern") == "points" else 0.0
    points = coat.get("pattern") == "points"

    def ell(c2, rad, ang, col=base, z=0.0, **kw):
        kw.setdefault("belly", belly_on)
        kw.setdefault("rough", rough)
        kw.setdefault("k", P.get("torso_k", 0.11))
        return m.ell("core", _p3(c2, z), rad, col, rot=_R2(ang), **kw)

    # ---- torso
    ell(S + _rot2(np.array(P["chest_at"]), ca), P["chest"], ca, name="chest")
    bsc = sk.get("belly_scale", (1.0, 1.0, 1.0))
    ell(mid + _rot2(np.array(P["belly_at"]) - [0.0, P["belly"][1] * (1 - bsc[1]) * 0.6], ma),
        tuple(r * f_ for r, f_ in zip(P["belly"], bsc)), ma, name="belly")
    ell(mid + _rot2(np.array(P["loin_at"]), ma), P["loin"], ma, name="loin", belly=0.0)
    ell(H + _rot2(np.array(P["rump_at"]), pa), P["rump"], pa, name="rump")
    if P["withers"]:
        c = S + _rot2(np.array([-0.1, P["chest"][1] * 0.75]), ca)
        ell(c, (0.14, P["withers"] + 0.05, P["chest"][2] * 0.55), ca + 8, name="withers", belly=0.0)
    if P["hump"]:
        c = S + _rot2(np.array([-0.06, P["chest"][1] * 0.7]), ca)
        ell(c, (0.16, P["hump"] + 0.06, P["chest"][2] * 0.75), ca + 10, name="hump", belly=0.0)
    if P.get("udder") and style != "cute":
        c = H + _rot2(np.array([0.14, -P["rump"][1] * 0.75]), pa)
        m.ell("core", _p3(c, 0.0), (0.075, P["udder"], 0.07), coat.get("udder", "#f0b4ae"), fur=0.0, pat=0.0,
              name="udder")

    # ---- legs
    leg_col = coat.get("legs", base)
    hoof_col = coat.get("hoof", "#3a302a")
    feet_col = coat.get("feet", leg_col)
    for nm, lg in sk["legs"].items():
        front = nm[0] == "f"
        near = nm[1] == "l"
        wz = sk.get("front_width", w) if front else w
        z = wz if near else -wz
        g = m.group("leg_" + nm, "core", 0.05 if front else 0.07)
        tone = 1.0 if near else 0.8
        r = P["fr"] if front else P["hr"]
        root, mid_, W = lg["root"], lg["mid"], lg["W"]
        ft = lg["foot"]
        legw = 1.0
        kw = dict(tone=tone, belly=0.0, leg=legw)
        up_col = base
        lo_col = leg_col
        cl = lambda t: None
        if front:
            # upper arm hangs down from the shoulder inside the chest
            m.cone(g, _p3(root + [0.0, 0.03], z * 0.7), _p3(mid_, z), r[0], r[1], up_col, rough=rough * 0.8,
                   name="arm", **kw)
            m.cone(g, _p3(mid_, z), _p3(W, z), r[1] * 0.92, r[2], lo_col, rough=rough * 0.3, name="forearm", **kw)
        else:
            m.cone(g, _p3(root + [0.02, 0.0], z * 0.6), _p3(mid_, z), r[0], r[1], up_col, rough=rough,
                   name="thigh", **kw)
            gas = mid_ + (W - mid_) * 0.32 + _rot2(_unit(W - mid_)[:2], -90) * r[1] * 0.25
            m.cone(g, _p3(mid_, z), _p3(W, z), r[1] * 0.98, r[2], lo_col, rough=rough * 0.3, name="gaskin", **kw)
            m.ell(g, _p3(gas, z), (r[1] * 0.9, r[1] * 1.15, r[1] * 0.75), lo_col, rot=_R2(
                math.degrees(math.atan2(*(W - mid_)[::-1]))), rough=rough * 0.3, name="gaskin_m", **kw)
            m.ell(g, _p3(W + _unit(W - mid_) * r[2] * 0.25, z), (r[2] * 1.1,) * 3, lo_col, name="hock", **kw)
        foot = P.get("foot_f" if front else "foot_h") or P["foot"]
        A = ft[0]
        if foot in ("hoof", "cloven"):
            K, B = ft[1], ft[2]
            m.cone(g, _p3(W, z), _p3(A, z), r[2] * 0.95, r[3], lo_col, name="cannon", **kw)
            m.ell(g, _p3(A, z), (r[3] * 1.2,) * 3, lo_col, name="fetlock", **kw)
            m.cone(g, _p3(A, z), _p3(K, z), r[3] * 0.95, r[3] * 0.9, lo_col, name="pastern", **kw)
            hr0, hr1 = P["paw"][0] * 0.78, P["paw"][0]
            kh = dict(kw)
            kh.update(fur=0.0, pat=0.0, spec=0.35, leg=0.0)
            hv = B - K
            Bx = B + _unit(hv) * 0.0
            pr = m.cone(g, _p3(K, z), _p3(Bx, z), hr0, hr1, hoof_col, k=0.0, name="hoof", **kh)
            if foot == "cloven" and style != "cute":
                m.decal("line", np.array([_p3(K + (B - K) * 0.3 + [hr1 * 0.3, 0], z + hr1 * 0.5),
                                          _p3(B + [hr1 * 0.6, 0.0], z + hr1 * 0.3)]), width=0.006, color="#120e0c",
                        opacity=0.5)
        elif foot == "pad":
            Bp = ft[1]
            m.cone(g, _p3(W, z), _p3(A, z), r[2], r[3], lo_col, name="lower", **kw)
            kp = dict(kw)
            kp.update(fur=0.0)
            m.cone(g, _p3(A, z), _p3(Bp, z), P["paw"][0] * 0.95, P["paw"][0], lo_col, k=0.02, name="foot", **kp)
            for i in (-1, 0, 1):
                m.decal("spot", _p3(Bp + [P["paw"][0] * 0.9, 0.02], z + i * P["paw"][0] * 0.45), size=0.014,
                        color=hoof_col, squash=0.6, soft=0.8, opacity=0.6, tol=0.03)
        else:
            C = ft[1]
            m.cone(g, _p3(W, z), _p3(A, z), r[2] * 0.95, r[3], lo_col, name="meta", **kw)
            rx, ry, rz = P["paw"]
            if foot == "plantigrade" and not front:
                rx = max(rx, P["h"][2] * 0.35)
            ang_t = -lg.get("flex", 0.0) * 0.7
            if foot == "plantigrade" and not front:
                ang_t = 90 - lg["ang"] - 90 + 0.0
            m.ell(g, _p3(C, z), (rx * 0.8, ry * 0.85, rz), feet_col if feet_col != leg_col else lo_col, rot=_R2(ang_t),
                  name="paw", **kw)
            if style != "flat" and P["paw"][0] > 0.03 and foot == "paw":
                for i in (-1, 1):
                    m.decal("line", np.array([_p3(C + [rx * 0.55, ry * 0.4], z + i * rz * 0.28),
                                              _p3(C + [rx * 0.62, -ry * 0.55], z + i * rz * 0.3)]),
                            width=0.004, color="#000000", opacity=0.28)
        if lg.get("planted", True) and lg["contact"][1] < 0.02:
            m.shadow_pts.append(_p3(lg["contact"], z))
        m.anchors["paw_%s_%s" % ("front" if front else "hind", "l" if near else "r")] = _p3(ft[-1], z)

    m.anchors["back"] = _p3(mid + _rot2(np.array([0.0, P["loin"][1] + P["loin_at"][1]]), ma))
    m.anchors["withers"] = _p3(S + _rot2(np.array([-0.08, P["chest"][1] + P["chest_at"][1]]), ca))
    m.anchors["chest"] = _p3(S + _rot2(np.array([P["chest_at"][0] + P["chest"][0] * 0.9, P["chest_at"][1]]), ca))
    # ---- neck
    nb = S + _rot2(np.array(P["neck_at"]), ca)
    na = sk["neck_ang"]
    atlas = nb + _dir2(na) * P["neck_len"]
    ctrl = (nb + atlas) / 2 + _rot2(_dir2(na), 90) * P["neck_arch"]
    npts, nrad = [], []
    for i in range(5):
        t = i / 4
        p = (1 - t) ** 2 * nb + 2 * (1 - t) * t * ctrl + t * t * atlas
        npts.append(_p3(p))
        nrad.append(_lerp(P["neck_r"][0], P["neck_r"][1], t))
    for pr in m.chain("core", npts, nrad, base, belly=belly_on, rough=rough * (0.7 if P["neck_len"] < 0.3 else 0.4),
                      name="neck", k=0.03):
        pr.axis = _unit(npts[-1] - npts[0]) if P["neck_len"] > 0.3 else pr.axis
    # ---- head
    _build_head(m, P, sk, coat, style, atlas, base, belly_on, seed)
    # ---- mane
    mane = P.get("mane")
    if mane:
        _build_mane(m, P, mane, coat, npts, nrad, style)
    # ---- tail
    _build_tail(m, P, sk, coat, H, pa, style)

    # ---- ground: shift so the lowest point touches y = 0 (grounded poses)
    lo = _model_bottom(m)
    if sk.get("grounded", True):
        dy = -lo
    else:
        planted = [lg["contact"][1] for lg in sk["legs"].values() if lg.get("planted")]
        dy = -min(planted) if planted else 0.0
    _shift(m, np.array([0.0, dy, 0.0]))
    m.lift = dy + lo
    return m


def _model_bottom(m):
    lo = 1e9
    for p in m.prims:
        if p.kind == "ell":
            c, A = p.geom
            lo = min(lo, c[1] - math.sqrt(float((A[1] ** 2).sum())))
        elif p.kind == "cone":
            a, b, ra, rb = p.geom
            lo = min(lo, a[1] - ra, b[1] - rb)
        else:
            lo = min(lo, float(p.geom[:, 1].min()))
    return lo


def _model_bounds(m):
    lo = np.array([1e9] * 3)
    hi = -lo
    for p in m.prims:
        if p.kind == "ell":
            c, A = p.geom
            e = np.sqrt((A ** 2).sum(1))
            lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
        elif p.kind == "cone":
            a, b, ra, rb = p.geom
            lo = np.minimum(lo, np.minimum(a - ra, b - rb))
            hi = np.maximum(hi, np.maximum(a + ra, b + rb))
        else:
            lo, hi = np.minimum(lo, p.geom.min(0)), np.maximum(hi, p.geom.max(0))
    return lo, hi


def _shift(m, d):
    for p in m.prims:
        if p.kind == "ell":
            p.geom = (p.geom[0] + d, p.geom[1])
        elif p.kind == "cone":
            a, b, ra, rb = p.geom
            p.geom = (a + d, b + d, ra, rb)
        else:
            p.geom = p.geom + d
    for dc in m.decals:
        dc.p = dc.p + d
    m.shadow_pts = [q + d for q in m.shadow_pts]
    m.anchors = {k: v + d for k, v in m.anchors.items()}


def _surf(c, rad, Rw, d):
    """Point and outward normal on an ellipsoid (centre c, radii rad, rotation Rw) along local direction d."""
    d = _unit(d)
    rad = np.asarray(rad, np.float64)
    p = d / math.sqrt(float(((d / rad) ** 2).sum()))
    n = _unit(p / rad ** 2)
    return c + Rw @ p, Rw @ n


def _build_head(m, P, sk, coat, style, atlas, base, belly_on, seed):
    Rh = _ry(sk.get("head_yaw", 0.0)) @ _rz(sk["head_ang"]) @ _rx(sk.get("head_roll", 0.0))
    skull = np.array(P["skull"])
    hc = _p3(atlas) + Rh @ np.array([P["head_at"][0], P["head_at"][1], 0.0])
    face = coat.get("face", base)
    pt_face = 1.0 if coat.get("pattern") == "points" else 0.0
    kw = dict(belly=belly_on, fur=0.6, rough=0.0)
    m.ell("core", hc, skull, face, rot=Rh, name="skull", k=0.045, pt=pt_face * 0.35, **kw)
    m0 = hc + Rh @ np.array([P["muzzle_at"][0], P["muzzle_at"][1], 0.0])
    md = _dir2(P["muzzle_drop"])
    tip = m0 + Rh @ np.array([md[0] * P["muzzle"], md[1] * P["muzzle"], 0.0])
    mr0, mr1 = P["muzzle_r"]
    mcol = coat.get("muzzle", face)
    fwd = Rh @ np.array([1.0, 0, 0])
    upv = Rh @ np.array([0, 1.0, 0])
    side = Rh @ np.array([0, 0, 1.0])
    mw = P.get("muzzle_w", 1.0)                      # muzzle width relative to its height
    Rm = Rh @ _rz(P["muzzle_drop"])
    if mw != 1.0:
        # an elliptical muzzle: a chain of ellipsoids instead of a round cone
        for i in range(4):
            t = i / 3
            c = m0 + (tip - m0) * t
            r = _lerp(mr0, mr1, t)
            m.ell("core", c, (P["muzzle"] / 3 * 0.9 + r * 0.4, r, r * mw), mcol, rot=Rm, name="muzzle", k=0.03,
                  pt=pt_face, pat=0.3, **kw)
    else:
        m.cone("core", m0, tip, mr0, mr1, mcol, name="muzzle", k=0.035, pt=pt_face, pat=0.3, **kw)
    if P.get("jaw", 0.0):
        # lower jaw / chin under the muzzle
        j0 = m0 - upv * mr0 * 0.45
        j1 = tip - fwd * mr1 * 0.6 - upv * mr1 * 0.55
        m.cone("core", j0, j1, mr0 * P["jaw"], mr1 * P["jaw"] * 0.8, coat.get("chin", mcol), name="jaw", k=0.025,
               pt=pt_face, **kw)
    if P.get("cheek"):
        for s in (1, -1):
            c = hc + Rh @ np.array([skull[0] * 0.35, -skull[1] * 0.35, s * skull[2] * 0.45])
            m.ell("core", c, (P["cheek"], P["cheek"] * 0.8, P["cheek"] * 0.75), coat.get("cheek", face), rot=Rh,
                  name="cheek", k=0.03, pt=pt_face * 0.6, **kw)
    if P.get("jowl"):
        c = hc + Rh @ np.array([skull[0] * 0.2, -skull[1] * 0.55, 0.0])
        m.ell("core", c, (P["jowl"] * 1.2, P["jowl"], skull[2] * 0.85), face, rot=Rh, name="jowl", k=0.04, **kw)
    if P.get("brow"):
        for s in (1, -1):
            c = hc + Rh @ np.array([skull[0] * 0.45, skull[1] * 0.35, s * skull[2] * 0.45])
            m.ell("core", c, (skull[0] * 0.35, skull[1] * 0.3, skull[2] * 0.3), face, rot=Rh, name="brow", k=0.03,
                  **kw)
    m.anchors["mouth"] = tip + fwd * mr1 * 0.6 - upv * mr1 * 0.5
    m.anchors["nose"] = tip + fwd * mr1
    m.anchors["head"] = hc + upv * skull[1]
    # ---- eyes
    e = P["eye"]
    closed = sk.get("eyes") == "closed"
    for s in (1, -1):
        d = np.array(e["dir"]) * [1, 1, s]
        p, n = _surf(hc, skull, Rh, d)
        p = p + n * 0.004
        if s == 1:
            m.anchors["eye"] = p
        if closed:
            arc = [p + fwd * e["size"] * a - upv * e["size"] * 0.35 * (1 - a * a) for a in np.linspace(-1, 1, 7)]
            m.decal("line", np.array(arc), width=e["size"] * 0.22, color="#1a120e", opacity=0.9)
            continue
        m.decal("eye", p, n=n, fwd=fwd, size=e["size"], shape=e.get("shape", "almond"), iris=coat.get("eye", e["iris"]),
                pupil=e.get("pupil", "round"), pupil_r=e.get("pupil_r", 0.5), look=e.get("look", (0.1, 0.0)),
                rim_w=0.18 if style != "cute" else 0.12, glint=e.get("glint", 0.26), tol=0.1)
    # ---- nose, mouth, whiskers
    nz = P.get("nose")
    if nz:
        shape = nz["shape"]
        ncol = coat.get("nose", nz["color"])
        ns = nz["size"]
        if shape in ("dog", "cat"):
            # nose leather: a small glossy volume on the end of the muzzle, nostrils on its front
            flat_ = 0.55 if shape == "cat" else 0.85
            c = tip + fwd * (mr1 * 0.85) + upv * (mr1 * (0.4 if shape == "dog" else 0.55))
            rn = (ns * 0.75 * flat_, ns * 0.62, ns * 0.95)
            m.ell("core", c, rn, ncol, rot=Rh @ _rz(20), name="nose", k=0.005,
                  fur=0.0, pat=0.0, belly=0.0, spec=0.6 if shape == "dog" else 0.25)
            if shape == "dog":
                for s in (1, -1):
                    pn, nn = _surf(c, rn, Rh @ _rz(20), (0.8, -0.2, 0.55 * s))
                    m.decal("spot", pn, size=ns * 0.2, color="#050303", squash=0.7, soft=0.8, opacity=0.85, tol=0.05)
        elif shape == "snout":
            m.cone("core", tip - fwd * 0.02, tip + fwd * 0.03, mr1 * 1.05, mr1 * 1.0, ncol, k=0.012, fur=0.0,
                   pat=0.0, name="snout")
            p = tip + fwd * (0.03 + mr1)
            m.decal("nose", p, n=fwd, fwd=side, size=mr1 * 0.95, shape="snout", color=ncol, tol=0.1)
        elif shape == "muzzle":
            m.ell("core", tip - fwd * mr1 * 0.15 - upv * 0.005, (mr1 * 0.8, mr1 * 0.78, mr1 * 0.95), ncol, rot=Rh,
                  k=0.03, fur=0.0, pat=0.0, belly=0.0, spec=0.25, name="muzzle_pad")
            for s in (1, -1):
                p, n = _surf(tip, (mr1, mr1, mr1 * 1.08), Rh, (0.75, 0.2, 0.5 * s))
                m.decal("nose", p, n=n, fwd=upv, size=mr1 * 0.3, shape="nostril", color="#5a3a34", tol=0.08)
        else:
            for s in (1, -1):
                p, n = _surf(tip, (mr1, mr1, mr1), Rh, (0.6, 0.35, 0.65 * s))
                m.decal("nose", p, n=n, fwd=upv, size=ns, shape="nostril", color=ncol, tol=0.06)
    if P.get("mouth", True):
        mouth = sk.get("mouth", "closed")
        lipc = coat.get("lip", "#2a1a16")
        for s in (1, -1):
            if nz and nz["shape"] == "cat":
                a0 = tip + fwd * mr1 * 0.8 + upv * mr1 * 0.15
                a1 = tip + fwd * mr1 * 0.62 - upv * mr1 * 0.4
                a2 = tip + fwd * mr1 * 0.05 - upv * mr1 * 0.5 + side * s * mr1 * 0.75
                # the cat mouth: a short philtrum, then each lip curls down and back up (a soft "w")
                ctrl = a1 + (a2 - a1) * 0.45 - upv * mr1 * 0.42 + side * s * mr1 * 0.05
                pts = [a0] + [(1 - t) ** 2 * a1 + 2 * (1 - t) * t * ctrl + t * t * a2 for t in np.linspace(0, 1, 7)]
                m.decal("line", np.array(pts), width=0.0045, color=lipc, opacity=0.8, tol=0.05)
            else:
                # philtrum down from the nose, then one smooth lip line back to the corner of the mouth
                a0 = tip + fwd * mr1 * 0.88 - upv * mr1 * 0.3
                b0 = tip + fwd * mr1 * 0.8 - upv * mr1 * 0.6                     # straight philtrum, no peak
                a1 = tip + fwd * mr1 * 0.5 - upv * mr1 * 0.82 + side * s * mr1 * 0.32
                mlen = min(P["muzzle"] * (0.85 if mouth == "open" else 0.7), 0.15)
                a2 = tip - fwd * mlen - upv * mr0 * 0.45 + side * s * mr0 * 0.8
                ctrl = tip - fwd * mlen * 0.35 - upv * mr1 * 0.95 + side * s * mr1 * 0.75
                c0 = b0 - upv * mr1 * 0.2 + side * s * mr1 * 0.05

                def _bez(p0, pc, p1, n=5):
                    return [(1 - t) ** 2 * p0 + 2 * (1 - t) * t * pc + t * t * p1 for t in np.linspace(0, 1, n)]
                pts = [a0] + _bez(b0, c0, a1)[:-1] + _bez(a1, ctrl, a2, 7)
                m.decal("line", np.array(pts), width=0.006, color=lipc, opacity=0.7, tol=0.05)
        if mouth in ("open", "pant"):
            tongue = tip - fwd * P["muzzle"] * 0.25 - upv * mr1 * 1.25
            m.cone("core", tip - fwd * P["muzzle"] * 0.55 - upv * mr1 * 0.6, tongue, mr1 * 0.5, mr1 * 0.42,
                   "#d8606a", fur=0.0, pat=0.0, belly=0.0, spec=0.3, k=0.0, name="tongue")
    nwh = P.get("whiskers", 0)
    if nwh and style != "flat":
        wc = coat.get("whisker", "#f6f2ea" if _lum(base) < 0.6 else "#5a4a40")
        for s in (1, -1):
            for i in range(nwh):
                t = (i - (nwh - 1) / 2) / max(nwh - 1, 1)
                b0 = tip - fwd * mr1 * 0.15 - upv * mr1 * (0.2 + 0.18 * t) + side * s * mr1 * 0.85
                dirv = _unit(upv * (-t * 0.45 + 0.05) + fwd * 0.15 + side * s * 1.0)
                L = 0.13 + 0.04 * (1 - abs(t))
                pts = [b0 + dirv * L * a + upv * (-0.025 * a * a) for a in np.linspace(0, 1, 5)]
                m.decal("line", np.array(pts), n=side * s, width=0.0026, color=wc, opacity=0.6, clip=False, tol=0.05)
    # ---- ears
    ear = P.get("ear")
    if ear:
        _build_ears(m, P, ear, sk, coat, style, hc, skull, Rh, base)
    # ---- horns, antlers, trunk, tusks, beard, ossicones
    hcol = coat.get("horn")
    if P.get("horns"):
        hz = P["horns"]
        for s in (1, -1):
            b0 = hc + Rh @ np.array([skull[0] * 0.05, skull[1] * 0.82, s * skull[2] * 0.55])
            pts = [b0]
            L = hz["len"]
            if hz["kind"] == "cow":
                dirs = [(0.1, 0.2, 1.0), (0.3, 0.6, 0.8), (0.5, 0.9, 0.2)]
            else:  # goat: up and swept back
                dirs = [(-0.3, 1.0, 0.15), (-0.8, 0.7, 0.2), (-1.0, 0.1, 0.25)]
            for dv in dirs:
                dvv = Rh @ (np.array(dv) * [1, 1, s])
                pts.append(pts[-1] + _unit(dvv) * L / len(dirs))
            rad = [hz["r"], hz["r"] * 0.8, hz["r"] * 0.55, hz["r"] * 0.2]
            m.chain(m.group("horn%d" % s, "core", 0.01), pts, rad, hcol or hz["color"], fur=0.0, pat=0.0, belly=0.0,
                    spec=0.25, tone=1.0 if s == 1 else 0.85, k=0.004,
                    stops=[(0.0, hcol or hz["color"], 0.0), (0.7, hcol or hz["color"], 0.0), (1.0, "#3a2e26", 0.0)])
    if P.get("antlers") and sk.get("antlers", True):
        _build_antlers(m, P["antlers"], hc, skull, Rh)
    if P.get("ossicones"):
        for s in (1, -1):
            b0 = hc + Rh @ np.array([-skull[0] * 0.2, skull[1] * 0.85, s * skull[2] * 0.45])
            top = b0 + _unit(Rh @ np.array([-0.35, 1.0, 0.15 * s])) * P["ossicones"]
            g = m.group("oss%d" % s, "core", 0.01)
            m.cone(g, b0, top, 0.016, 0.012, base, fur=0.0, pat=0.0, belly=0.0, k=0.0)
            m.ell(g, top, (0.018,) * 3, "#2a1e16", fur=0.0, pat=0.0, belly=0.0)
    if P.get("trunk_len"):
        L = P["trunk_len"]
        curl = sk.get("trunk_curl", 0.25)
        pts = [tip - fwd * 0.02]
        n = 10
        for i in range(n):
            t = (i + 1) / n
            # out from the face, hanging down, the tip curling forward
            ang = -35.0 - 55.0 * min(t * 1.6, 1.0) + curl * 160.0 * max(t - 0.6, 0) / 0.4
            dv = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang)), 0.0])
            pts.append(pts[-1] + dv * L / n)
        rad = [mr1 * 1.15 * (1.0 - 0.6 * (i / n) ** 0.9) for i in range(n + 1)]
        tp = m.chain("core", pts, rad, base, fur=0.0, belly=0.0, k=0.02, name="trunk")
        for q in tp:
            q.pat = 1.0
        m.ell("core", pts[-1], (rad[-1] * 1.15,) * 3, base, fur=0.0, belly=0.0, k=0.01, name="trunk_tip")
        m.anchors["mouth"] = pts[-1]
    if P.get("tusks"):
        for s in (1, -1):
            b0 = tip - fwd * P["muzzle"] * 0.5 - upv * mr0 * 0.55 + side * s * mr0 * 0.55
            pts = [b0]
            for dv in ((0.6, -0.8, 0.12), (0.9, -0.2, 0.05), (0.8, 0.45, -0.05)):
                pts.append(pts[-1] + _unit(Rh @ (np.array(dv) * [1, 1, s])) * P["tusks"] / 3)
            m.chain(m.group("tusk%d" % s, "core", 0.004), pts, [0.028, 0.024, 0.017, 0.006], "#f2ead8", fur=0.0,
                    pat=0.0, belly=0.0, spec=0.3, k=0.002, tone=1.0 if s == 1 else 0.85)
    if P.get("beard"):
        b0 = tip - fwd * P["muzzle"] * 0.45 - upv * mr0 * 0.6
        m.cone("core", b0, b0 - upv * P["beard"] + fwd * 0.01, mr1 * 0.5, 0.008, coat.get("beard", base), fur=1.4,
               belly=0.0, k=0.01, axis=-upv, name="beard")


def _lum(c):
    c = rgb(c)
    return float(0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])


def _build_ears(m, P, ear, sk, coat, style, hc, skull, Rh, base):
    kind = ear["kind"]
    ears = sk.get("ears")
    up, back, out = ear["up"], ear["back"], ear["out"]
    if ears == "back":
        back += 45
        up -= 20
    elif ears == "down":
        up -= 60
    elif ears == "alert":
        back -= 10
    hang = kind in ("floppy", "elephant", "side")
    L, W = ear["len"], ear["w"]
    outline = _ear_outline(kind)
    backc = coat.get("ear_back", base)
    pt = 1.0 if coat.get("pattern") == "points" else 0.0
    for s in (1, -1):
        at = _unit(np.array(ear["at"], np.float64))
        if not hang:
            at[2] = max(at[2], 0.62)                    # upright ears sit on the top corners of the head
        at = at * [1, 1, s]
        b0, nrm = _surf(hc, skull, Rh, at)
        e, o = math.radians(up), math.radians(out)
        v = np.array([-math.cos(e), math.sin(e), 0.0])
        v = _rz(back) @ v
        v = _rx(out * s) @ v if not hang else v
        if hang:
            # hanging ears fall along the side of the head, splayed out by ``out`` degrees
            v = _unit(np.array([v[0], v[1], 0.0]) + np.array([0, 0, s]) * math.sin(o) * 0.6)
        v = Rh @ _unit(v)
        if hang:
            n0 = nrm
        else:
            # the ear opening faces forward and a little out, so the far ear still shows its shape in 3/4 views
            # (long rabbit ears open more to the side, so neither ear is seen edge-on as a spike)
            n0 = Rh @ _unit(np.array([1.0, 0.15, (1.3 if kind == "long" else 0.38) * s]))
        n = _unit(n0 - v * np.dot(n0, v))
        V = sk.get("view_dir")
        if V is not None and not hang:
            # an upright ear seen exactly edge-on reads as a spike: turn it a little towards the viewer
            Vp = V - v * np.dot(V, v)
            if np.linalg.norm(Vp) > 1e-3:
                Vp = _unit(Vp)
                c_ = float(np.dot(n, Vp))
                if abs(c_) < 0.45:
                    tgt = Vp if c_ >= 0 else -Vp
                    n = _unit(n + tgt * (0.45 - abs(c_)) * 1.6)
        wv = _unit(np.cross(v, n))
        sink = 0.18 if not hang else 0.05
        root = b0 - nrm * W * sink + (nrm * W * 0.12 if hang else 0)
        pts = []
        for (a, t) in outline:
            bend = (-0.25 * t * t) if kind == "floppy" else (-0.12 * t * t if kind == "elephant" else 0.0)
            cup = 0.0 if hang else -0.22 * W * (1 - (2 * a) ** 2) * (1 - t)
            pts.append(root + v * (t * L) + wv * (a * W) + n * (bend * L * -1 + cup))
        g = m.group("ear%d" % s, "core", 0.02 if not hang else 0.025)
        tone = 1.0 if s == 1 else 0.88
        m.flat(g, pts, backc, bulge=1.1 if hang else 0.8, tone=tone, belly=0.0, pt=pt, fur=0.6 if hang else 0.3,
               pat=0.4, axis=v, name="ear", k=0.02, smooth=2 if hang else 0)
        if not hang:
            inner = coat.get("inner_ear", "#e8b4a8" if kind in ("mouse", "long", "pig", "round") else "#d9a497")
            if _lum(base) < 0.25 and "inner_ear" not in coat:
                inner = "#6a5048"
            pin = []
            for (a, t) in outline:
                tt = 0.1 + t * 0.78
                cup = -0.22 * W * 0.62 * (1 - (2 * a * 0.62) ** 2) * (1 - tt)
                pin.append(root + v * (tt * L) + wv * (a * W * 0.6) + n * (cup + 0.004 + W * 0.04))
            m.flat(g, pin, inner, bulge=0.25, tone=tone, belly=0.0, fur=0.0, pat=0.0, k=0.0, name="inner_ear")
        if s == 1:
            m.anchors["ear"] = root + v * L


def _build_antlers(m, A, hc, skull, Rh):
    col = A.get("color", "#cdb796")
    L = A["len"]
    for s in (1, -1):
        g = m.group("antler%d" % s, "core", 0.006)
        b0 = hc + Rh @ np.array([-skull[0] * 0.1, skull[1] * 0.85, s * skull[2] * 0.45])
        beam = [b0]
        dirs = [(-0.5, 1.0, 0.45), (-0.55, 1.0, 0.35), (-0.2, 1.0, 0.2), (0.25, 1.0, 0.05), (0.55, 0.85, -0.05)]
        for dv in dirs:
            dw = np.array(dv) * [1, 1, s]
            beam.append(beam[-1] + _unit(dw) * L / len(dirs))
        rad = [0.02, 0.017, 0.014, 0.011, 0.008, 0.004]
        tone = 1.0 if s == 1 else 0.85
        m.chain(g, beam, rad, col, fur=0.0, pat=0.0, belly=0.0, tone=tone, k=0.003, spec=0.1)
        # tines: forward brow tine and upward points
        for i, (bi, dv, ln) in enumerate(((0, (1.0, 0.45, 0.1), 0.35), (2, (0.6, 1.0, 0.1), 0.42),
                                          (3, (0.1, 1.0, 0.05), 0.38))[:A.get("tines", 3)]):
            p0 = beam[bi + 1] if bi else beam[0] + (beam[1] - beam[0]) * 0.4
            dw = np.array(dv) * [1, 1, s]
            p1 = p0 + _unit(dw) * L * ln * 0.55
            p2 = p1 + _unit(dw + np.array([0, 0.4, 0])) * L * ln * 0.45
            m.chain(g, [p0, p1, p2], [rad[bi + 1] * 0.85, rad[bi + 1] * 0.6, 0.003], col, fur=0.0, pat=0.0,
                    belly=0.0, tone=tone, k=0.002, spec=0.1)


def _build_mane(m, P, mane, coat, npts, nrad, style):
    col = coat.get("mane", mane.get("color", "#2a1a12"))
    kind = mane["kind"]
    if kind == "horse":
        # locks of hair lying on the crest and falling over the near side of the neck
        g = m.group("mane", "core", 0.02)
        n = len(npts)
        # resample the neck centre line so the locks overlap smoothly
        ts = np.linspace(0.12, 1.0, 9)
        for j, tt in enumerate(ts):
            f = tt * (n - 1)
            i = min(int(f), n - 2)
            u = f - i
            p = npts[i] * (1 - u) + npts[i + 1] * u
            r = nrad[i] * (1 - u) + nrad[i + 1] * u
            d = _unit(npts[i + 1] - npts[i])
            up = _unit(np.array([-d[1], d[0], 0.0]))
            seg = np.linalg.norm(npts[-1] - npts[0]) / 8.0
            ln = 0.06 + 0.025 * math.sin(j * 2.3)
            for a in (78.0, 40.0):
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                nrm = up * sa + np.array([0.0, 0.0, ca])            # surface normal of the neck there
                c = p + nrm * (r * 0.97) - up * (0.0 if a > 60 else ln * 0.25)
                hang = _unit(np.cross(nrm, d))                       # down the side of the neck
                if hang @ up > 0:
                    hang = -hang
                R = np.stack([d, hang, nrm], 1)
                m.ell(g, c, (seg * 1.25, (ln if a < 60 else 0.05), 0.02), col, rot=R, fur=1.6, pat=0.0,
                      belly=0.0, k=0.05, axis=hang, name="mane")
    elif kind == "brush":
        top = []
        for i, (p, r) in enumerate(zip(npts, nrad)):
            d = _unit(npts[min(i + 1, len(npts) - 1)] - npts[max(i - 1, 0)])
            up = _unit(np.array([-d[1], d[0], 0.0]))
            top.append(p + up * (r + 0.02))
        m.chain(m.group("mane", "core", 0.015), top[1:], [0.03] * (len(top) - 1), col, fur=1.2, pat=1.0, belly=0.0,
                k=0.02, axis=np.array([0.0, 1.0, 0.0]))
    elif kind == "lion":
        # a shaggy ruff framing the face and covering the neck and chest
        sz = mane.get("size", 1.0)
        g = m.group("mane", "core", 0.06)
        hc = npts[-1]
        base_c = npts[1]
        col2 = coat.get("mane2", col)
        m.ell(g, hc + np.array([-0.07, 0.0, 0]), (0.15 * sz, 0.25 * sz, 0.23 * sz), col2, fur=3.5, pat=0.0, belly=0.0,
              rough=0.9, k=0.08, axis=np.array([-0.6, -0.8, 0.0]), name="mane")
        for t in (0.15, 0.55):
            c = base_c + (hc - base_c) * t + np.array([0.0, 0.0, 0])
            m.ell(g, c, (0.17 * sz, 0.21 * sz, 0.19 * sz), col, fur=3.5, pat=0.0, belly=0.0, rough=0.9, k=0.08,
                  axis=np.array([-0.5, -1.0, 0.0]), name="mane")
        m.fluff = max(m.fluff, 0.045)


def _build_tail(m, P, sk, coat, H, pa, style):
    t = sk["tail"]
    kind = t["kind"]
    rr = P["rump"]
    base = coat.get("tail", coat.get("base", "#b08050"))
    if kind == "horse":
        base = coat.get("tail", coat.get("mane", (P.get("mane") or {}).get("color", base)))
    b0 = H + _rot2(np.array([P["rump_at"][0] - rr[0] * 0.82, P["rump_at"][1] + rr[1] * 0.45]), pa)
    n = max(int(t["n"]), 2)
    L = t["len"]
    a = t["ang"]
    pts = [_p3(b0)]
    zs = t.get("side", 0.0)
    r0, r1 = t["r"]
    on_ground = False
    for i in range(n):
        ai = a + t["curve"] * i
        dv = np.array([-math.cos(math.radians(ai)), math.sin(math.radians(ai)), zs * (i / n)])
        if on_ground:
            # the tail rests on the ground: it runs along it and curls forward around the feet
            dv = np.array([0.35 + 0.1 * i, 0.0, 0.9])
        p = pts[-1] + _unit(dv) * L / n
        rmin = _lerp(r0, r1, (i + 1) / n) * 0.9
        if t.get("ground") and p[1] < rmin:
            p[1] = rmin
            on_ground = True
        pts.append(p)
    tt = np.linspace(0, 1, n + 1)
    if kind in ("bushy", "brush", "plume", "squirrel"):
        fat = {"bushy": 1.0, "brush": 1.5, "plume": 1.1, "squirrel": 1.7}[kind]
        rad = [r0 * (0.75 + fat * math.sin(math.pi * min(x * 0.85 + 0.12, 1.0)) ** 0.8) * (1 - 0.6 * x ** 3)
               for x in tt]
        rad[-1] = max(rad[-1], r1 * 0.4)
        fur = 2.0
    elif kind == "horse":
        rad = [r0 * (0.8 + 0.9 * x ** 0.6) * (1 - 0.65 * x ** 4) for x in tt]
        fur = 2.0
    elif kind == "otter":
        rad = [r0 * (1 - 0.65 * x) for x in tt]
        fur = 0.6
    elif kind in ("stub", "puff"):
        rad = [r0, r1] + [r1] * (n - 1)
        fur = 1.0
    else:
        rad = [r0 + (r1 - r0) * x for x in tt]
        fur = 0.6
    tip = coat.get("tail_tip")
    pt = 1.0 if coat.get("pattern") == "points" else 0.0
    stops = None
    if tip and kind not in ("tuft",):
        stops = [(0.0, base, pt * 0.5), (0.72, base, pt), (0.85, tip, pt), (1.0, tip, pt)]
    elif pt:
        stops = [(0.0, base, 0.2), (0.5, base, 1.0), (1.0, base, 1.0)]
    g = m.group("tail", "core", 0.035)
    if kind == "puff":
        m.ell(g, pts[0] + np.array([-0.03, 0.0, 0.0]), (P["tail"]["r"][0],) * 3, tip or base, fur=2.0, pat=0.0,
              belly=0.0, rough=0.6)
        return
    m.anchors["tail"] = pts[-1]
    prs = m.chain(g, pts, rad, base, fur=fur, belly=0.0, stops=stops, k=0.04 if kind == "horse" else 0.015, name="tail",
                  pat=0.0 if kind == "horse" else 1.0)
    for p in prs:
        p.axis = -p.axis
    if kind == "tuft":
        m.ell(g, pts[-1], (0.05, 0.035, 0.035), tip or coat.get("mane", "#2a1a12"),
              rot=_rz(math.degrees(math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0]))), fur=2.0,
              pat=0.0, belly=0.0, rough=0.5)
    if kind == "pompom":
        m.ell(g, pts[-1], (0.065,) * 3, base, fur=0.0, pat=0.0, belly=0.0, rough=1.0)


# ===================================================================== public API

_VIEWS = {"side": (0.0, 5.0), "three-quarter": (34.0, 12.0), "3/4": (34.0, 12.0), "three_quarter": (34.0, 12.0),
          "front": (90.0, 6.0), "back": (-140.0, 14.0), "rear": (-140.0, 14.0), "top": (0.0, 70.0),
          "below": (10.0, -35.0)}

_KINDS = {}            # species name -> builder kind ("quad", "bird", "sea", "small")
for _n in _SPECIES_Q:
    _KINDS[_n] = "quad"

_REF_CACHE = {}


def _resolve(name):
    n = str(name).strip().lower().replace("_", " ")
    if n in _KINDS:
        return n, {}
    if n in _ALIASES:
        return _ALIASES[n], {}
    # "siamese cat", "black lab", "dalmatian dog" ...
    for coat in sorted(COATS, key=len, reverse=True):
        if n.startswith(coat + " ") or n == coat:
            rest = n[len(coat):].strip() or ("dog" if coat in ("dalmatian", "black lab", "chocolate") else
                                             "horse" if "horse" in coat or coat == "palomino" else
                                             "bear" if coat in ("polar", "panda") else
                                             "cat" if coat in ("leopard", "cheetah") else "cat")
            sp, _ = _resolve(rest) if rest != n else (rest, {})
            return sp, dict(COATS[coat])
    for k in sorted(list(_KINDS) + list(_ALIASES), key=len, reverse=True):
        if k in n.split() or n.endswith(" " + k):
            return _resolve(k)[0], {}
    raise _unknown_name("species", name, list(_KINDS) + list(_ALIASES), "Try creature.species() for the list")


def species(kind=None):
    """Names of every preset, or only one kind: 'quad', 'bird', 'sea', 'small'."""
    return sorted(n for n, k in _KINDS.items() if kind is None or k == kind)


_POSE_OPTS = ("head_turn", "head_tilt", "tail", "ears", "mouth", "eyes", "neck_ang", "head_ang", "pitch", "wings",
              "branch")
_EXTRA_POSES = ("look", "run", "amble", "bound", "flying", "hover", "still", "strike", "coil")


def _anatomy_keys(name, kind):
    """Anatomy parameters a species accepts as keyword overrides (``neck_len=0.5``, ``tail=dict(len=0.8)`` ...)."""
    if kind == "quad":
        sp = _SPECIES_Q[name]
        keys = set(_Q) | set(sp) | set(_SPECIES_Q.get(sp.get("base"), {}))
    elif kind == "bird":
        keys = set(_B) | set(_SPECIES_B[name])
    else:
        keys = set((_SPECIES_S if kind == "sea" else _SPECIES_X)[name])
    return sorted(keys - {"base", "coat"})


class _AnchorError(ValueError, KeyError):
    """Unknown anchor name (a ValueError with a did-you-mean; still caught by ``except KeyError``)."""

    def __str__(self):
        return self.args[0]


_COAT_KEYS = ("amount", "back", "band", "band_color", "band_edge", "band_w", "bands", "base", "beak", "beard", "belly",
              "belly_dir", "belly_edge", "belly_jag", "body", "branch", "breast", "cap", "cheek", "chin", "collar",
              "colors", "colours", "comb", "covert", "ear_back", "eye", "face", "feet", "fin", "flipper", "fur",
              "fur_scale", "fuzz", "head", "hoof", "horn", "inner_ear", "legs", "lip", "mane", "mane2", "marks",
              "muzzle", "nose", "outline", "pattern", "points", "saddle", "scale", "scute", "sheen", "shell", "socks",
              "speculum", "spot", "spots", "streak", "stripe_taper", "stripes", "tail", "tail2", "tail_tip", "thigh",
              "throat", "udder", "whisker", "wing", "wing2", "wing_bar")


def _parse_coat(coat):
    """``coat`` -> (dict of part colours, base colour or None). A named coat, a colour or a dict are accepted."""
    if coat is None:
        return {}, None
    if isinstance(coat, dict):
        return _reject_unknown("Creature coat", dict(coat), _COAT_KEYS), None
    if isinstance(coat, str):
        key = coat.strip().lower().replace("_", " ")
        if key in COATS:
            return dict(COATS[key]), None
        try:
            return {}, rgb(coat)
        except ValueError:
            raise _unknown_name("coat", coat, list(COATS),
                                "Coat names: %s. A colour such as '#d8883a' or a dict of part colours "
                                "(base=..., belly=...) also works." % ", ".join(sorted(COATS))) from None
    try:
        col = rgb(coat)
        if np.ndim(col) != 1 or len(col) != 3:
            raise ValueError("not a colour")
        return {}, col
    except (ValueError, TypeError):
        raise _unknown_name("coat", coat, list(COATS),
                            "Use a coat name (%s), a colour or a dict of part colours." % ", ".join(sorted(COATS))) from None


def _recolour(c, col):
    """Give the coat's main colour (``base`` or ``back``) the colour ``col``; colours close to the old main colour move
    with it, markings (stripes, socks, spots, eyes, noses) stay."""
    key = "base" if "base" in c else "back" if "back" in c else None
    col = np.asarray(col, np.float32)
    out = dict(c)
    if key is None:
        out["base"] = tuple(col)
        return out
    old = rgb(c[key])
    for k, v in c.items():
        if isinstance(v, str) and v.startswith("#"):
            try:
                o = rgb(v)
            except ValueError:
                continue
            if k == key or float(np.abs(o - old).max()) < 0.2:
                out[k] = tuple(np.clip(col + (o - old), 0, 1))
    out[key] = tuple(col)
    if out.get("pattern") in ("tabby", "stripes", "tiger", "streaks", "bars"):
        out["colours"] = [tuple(np.clip(col * 0.55, 0, 1))]
    return out


class Creature:
    """One animal: ``Creature("cat", coat=dict(base=..., pattern="tabby"), size=0.3, view="side")``.

    species: any preset name (``species()`` lists them; aliases such as "puppy", "siamese cat", "black lab" work);
    coat: a coat name ("tabby", "ginger", "panda" ... see ``COATS``), a colour ('#d8883a' or (r, g, b): the main colour
    is recoloured, markings stay) or a dict overriding the preset colours — ``base``, ``belly``, ``pattern`` ("tabby", "spots", "patches",
    "stripes", "solid", "points", "tiger", "zebra", "giraffe", "rosettes"), ``colours`` (marking colours), ``points``,
    ``socks=(height, colour)``, ``tail_tip``, ``mane``, ``eye``, ``nose``, ``muzzle``, ``ear_back``, ``inner_ear``;
    size: standing height as a fraction of the canvas height (body length for fish, snakes, lizards);
    view: "side", "three-quarter", "front", "back" or a yaw angle in degrees; facing: "right" or "left";
    style: "soft" (shaded illustration), "flat" (clean vector), "cute" (round, big eyes);
    any anatomy parameter can be overridden by keyword (``neck_len=0.5``, ``muzzle=0.25``, ``tail=dict(len=0.8)``).
    """

    def __init__(self, species="dog", coat=None, size=0.3, view="side", facing="right", style="soft", seed=0,
                 light=None, **params):
        self.name, preset_coat = _resolve(species)
        self.kind = _KINDS[self.name]
        self.requested = str(species).strip().lower().replace("_", " ")
        coat, self.coat_color = _parse_coat(coat)
        self.coat_over = dict(preset_coat, **(coat or {}))
        self.size = float(size)
        self.view = view
        self.facing = facing
        if style not in _STYLES:
            raise _unknown_name("style", style, _STYLES)
        if not isinstance(view, (int, float, tuple, list)) and view not in _VIEWS:
            raise _unknown_name("view", view, _VIEWS)
        if facing in (1, -1):
            facing = "right" if facing == 1 else "left"
        if facing not in ("right", "left"):
            raise _unknown_name("facing", facing, ("right", "left"))
        self.facing = facing
        if params:
            params = _reject_unknown("Creature", params, _anatomy_keys(self.name, self.kind) + list(_POSE_OPTS))
        self.style = style
        self.seed = int(seed)
        self.light = light
        self.params = params
        self._pose = ((_OTHER.get(self.name) or {}).get("pose") or _DEFAULT_POSE.get(self.kind, "stand"), 0.0, {})
        self.mask = None
        self.bbox = None
        self.anchors = {}
        self.model = None

    # ---- configuration
    def pose(self, name="stand", phase=0.0, **opt):
        """Set the pose: stand, walk, trot, gallop (``phase`` 0~1 through the stride), sit, lie, sleep, crouch, jump,
        graze, sniff, rear; birds: stand, perch, fly (phase = wing beat), glide, swim; fish: swim (phase = tail
        beat). Options: ``head_turn`` (deg towards the viewer), ``head_tilt``, ``tail`` ("up", "down", "out", "curl",
        "wag" or an angle), ``ears`` ("back", "down", "alert"), ``mouth`` ("open", "pant"), ``eyes`` ("closed")."""
        if isinstance(name, str) and name not in POSES and name not in _EXTRA_POSES:
            raise _unknown_name("pose", name, list(POSES) + list(_EXTRA_POSES), "Try creature.POSES for the list")
        self._pose = (name, float(phase), _reject_unknown("Creature.pose", opt, _POSE_OPTS))
        return self

    def coat(self):
        """The resolved coat dict (preset colours merged with your overrides)."""
        return self._coat()

    def _coat(self):
        if self.kind == "quad":
            c = dict(_qparams(self.name, self.style, self.params).get("coat", {}))
        else:
            c = dict(_OTHER[self.name].get("coat", {}))
        if self.coat_color is not None:
            c = _recolour(c, self.coat_color)
        c.update(self.coat_over)
        return c

    def _build(self, pose=None):
        name, phase, opt = pose or self._pose
        if self.kind == "quad":
            P = _qparams(self.name, self.style, self.params)
            sk = _quad_pose(P, name, phase, opt)
            v = self.view
            yaw = float(v) if isinstance(v, (int, float)) else float(v[0]) if isinstance(v, (tuple, list)) else                 _VIEWS.get(v, _VIEWS["side"])[0]
            sk["view_dir"] = np.array([math.sin(math.radians(yaw)), 0.0, math.cos(math.radians(yaw))])
            return _build_quad(P, sk, self._coat(), self.style, self.seed)
        return _BUILDERS[self.kind](self.name, name, phase, opt, self._coat(), self.style, self.params, self.seed)

    def _ref(self):
        key = (self.name, self.style, repr(sorted(self.params.items())))
        if key not in _REF_CACHE:
            ref_pose = ("stand", 0.0, {}) if self.kind == "quad" else (_REF_POSE.get(self.kind, "stand"), 0.0, {})
            m = self._build(ref_pose)
            lo, hi = _model_bounds(m)
            meas = _OTHER.get(self.name, {}).get("measure", "len")
            if self.kind in ("sea", "small") and meas == "span":
                _REF_CACHE[key] = ("len", float(max(hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1])))
            elif self.kind in ("sea", "small") and meas == "len":
                _REF_CACHE[key] = ("len", float(hi[0] - lo[0]))
            else:
                _REF_CACHE[key] = ("h", float(hi[1] - lo[1]))
        return _REF_CACHE[key]

    def _camera(self, cv, at, model):
        k = getattr(at, "k", 1.0)
        _, ref = self._ref()
        scale = self.size * k * cv.h / max(ref, 1e-6)
        v = self.view
        if isinstance(v, (int, float)):
            yaw, pitch = float(v), 8.0
        elif isinstance(v, (tuple, list)):
            yaw, pitch = float(v[0]), float(v[1])
        else:
            yaw, pitch = _VIEWS.get(v, _VIEWS["side"])
        pitch = self._pose[2].get("pitch", getattr(model, "pitch", None) or pitch) if model is not None else pitch
        tilt = getattr(at, "tilt", None)
        if tilt is not None and self.kind == "quad":
            pitch = max(pitch, math.degrees(math.asin(min(float(tilt), 0.9))) * 0.5)
        x, y = at[0] * cv.w, at[1] * cv.h
        cam = _Camera((x, y), scale, yaw=yaw + getattr(model, "yaw_extra", 0.0), pitch=pitch,
                      mirror=(self.facing == "left"), roll=getattr(model, "roll", 0.0))
        return cam

    def render(self, cv, at=(0.5, 0.8)):
        """Build and render without painting → (model, camera, result with .rgb / .alpha crops and .box)."""
        model = self._build()
        cam = self._camera(cv, at, model)
        rr = _render_model(cv, model, cam, self.style, self.light, self.seed)
        return model, cam, rr

    def fit(self, cv, box, align="bottom"):
        """Choose ``size`` so the animal in its current pose fills ``box`` = (u0, v0, u1, v1) (canvas fractions)
        and return the ``at`` point to draw it at: centred, standing on the bottom edge (``align="center"`` centres
        it vertically, for fish and flying birds)."""
        model = self._build()
        cam = self._camera(cv, (0.0, 0.0), model)
        bx = [_prim_bounds(cam, p_, 0.0) for p_ in model.prims]
        x0, y0 = min(b[0] for b in bx), min(b[1] for b in bx)
        x1, y1 = max(b[2] for b in bx), max(b[3] for b in bx)
        bw, bh = (box[2] - box[0]) * cv.w, (box[3] - box[1]) * cv.h
        k = min(bw / max(x1 - x0, 1e-6), bh / max(y1 - y0, 1e-6))
        self.size *= k
        cx = (box[0] + box[2]) / 2 * cv.w - (x0 + x1) / 2 * k
        if align == "center":
            cy = (box[1] + box[3]) / 2 * cv.h - (y0 + y1) / 2 * k
        else:
            cy = box[3] * cv.h - y1 * k
        return (cx / cv.w, cy / cv.h)

    def draw(self, cv, at=(0.5, 0.8), log=None, shadow=True, cast=False, opacity=1.0, fit=None):
        """Paint the animal standing at ``at`` (ground point (u, v) or a form Spot). One layer step when ``log`` is
        given. ``shadow``: soft contact shadow (True / strength); ``cast``: projected ground shadow (True or a dict of
        ``form.cast_shadow`` options); ``fit=(u0, v0, u1, v1)``: ignore ``at`` / ``size`` and fill that box.
        Returns self (``.mask``, ``.bbox``, ``.anchor(...)`` are then set)."""
        if log:
            log(f"{self.name}")
        if fit is not None:
            at = self.fit(cv, fit, align="bottom" if self._grounded_pose() else "center")
        model, cam, rr = self.render(cv, at)
        self.model = model
        if rr is None:
            self.mask = np.zeros((cv.h, cv.w), F32)
            self.bbox = None
            return self
        light = _form.resolve_light(self.light)
        x0, y0, x1, y1 = rr.box
        full = np.zeros((cv.h, cv.w), F32)
        full[y0:y1, x0:x1] = rr.alpha
        if cast:
            kw = cast if isinstance(cast, dict) else {}
            kw.setdefault("ground_y", at[1])
            kw.setdefault("opacity", 0.45)
            _form.cast_shadow(cv, full, light=light, **kw)
        if shadow and getattr(model, "grounded", True):
            st = 0.6 if shadow is True else float(shadow)
            _ground_shadow(cv, model, cam, st, light, rr)
        cv.paste(rr.rgb, x0, y0, mask=rr.alpha * float(opacity))
        self.mask = full
        ys, xs = np.nonzero(rr.alpha > 0.05)
        self.bbox = (int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + x0 + 1, int(ys.max()) + y0 + 1) \
            if len(xs) else None
        self.anchors = {k: tuple(cam.p(v)[:2]) for k, v in model.anchors.items()}
        self.anchors["ground"] = (at[0] * cv.w, at[1] * cv.h)
        if self.bbox:
            bx0, by0, bx1, by1 = self.bbox
            self.anchors.setdefault("center", ((bx0 + bx1) / 2, (by0 + by1) / 2))
            self.anchors.setdefault("top", ((bx0 + bx1) / 2, by0))
        return self

    def _grounded_pose(self):
        name = self._pose[0]
        if self.kind == "sea":
            return self.name in ("octopus", "turtle") and name not in ("swim",)
        if self.kind == "small" and self.name in ("butterfly", "dragonfly"):
            return False
        return name not in ("fly", "glide", "swim", "jump", "hover")

    def anchor(self, name):
        """Pixel point of a body landmark after ``draw``: mouth, nose, eye, ear, head, back, tail, paw_front_l,
        paw_front_r, paw_hind_l, paw_hind_r, center, top, ground (birds: beak, wing; fish: tail)."""
        if name not in self.anchors:
            raise _AnchorError(str(_unknown_name("anchor", name, sorted(self.anchors))))
        return self.anchors[name]


def _ground_shadow(cv, model, cam, strength, light, rr):
    """Soft shadow under the body plus small dark contacts under each planted foot."""
    lo, hi = _model_bounds(model)
    cx = (lo[0] + hi[0]) / 2
    L = hi[0] - lo[0]
    p = cam.p(np.array([cx, 0.0, 0.0]))
    span = abs(cam.v(np.array([L, 0, 0]))[0]) + abs(cam.v(np.array([0, 0, hi[2] - lo[2]]))[0])
    rx = max(span * 0.42, 2.0)
    ry = max(rx * 0.1, abs(cam.v(np.array([0, 0, hi[2] - lo[2]]))[1]) * 0.6, 2.0)
    _form.contact_px(cv, p[0], p[1], rx, ry, strength * 0.45, light)
    for q in model.shadow_pts:
        s = cam.p(q)
        _form.contact_px(cv, s[0], s[1], max(0.06 * cam.s, 2.0), max(0.016 * cam.s, 1.2), strength * 0.55, light)


def creature(cv, species="dog", at=(0.5, 0.8), size=0.3, pose=None, phase=0.0, view="side", facing="right",
             style="soft", coat=None, log=None, shadow=True, cast=False, fit=None, seed=0, **opt):
    """Draw one animal in one call → the ``Creature`` (``.mask``, ``.bbox``, ``.anchor(...)``).

    ``creature(cv, "fox", at=(0.4, 0.85), size=0.25, pose="sit", view="three-quarter")``; ``fit=(u0, v0, u1, v1)``
    sizes and places it to fill a box instead. ``pose`` defaults to the species' natural pose (stand, perch, swim,
    fly). Pose options (``head_turn``, ``head_tilt``, ``tail``, ``ears``, ``mouth``, ``eyes``, ``wings``, ``pitch``)
    and anatomy overrides (``neck_len=0.5``, ``tail=dict(len=0.8)``) go in ``**opt``."""
    _nm = _resolve(species)[0]
    opt = _reject_unknown("creature", opt, _anatomy_keys(_nm, _KINDS[_nm]) + list(_POSE_OPTS) + ["light"])
    pose_keys = ("head_turn", "head_tilt", "ears", "mouth", "eyes", "neck_ang", "head_ang", "pitch", "wings")
    popt = {k: opt.pop(k) for k in list(opt) if k in pose_keys}
    if isinstance(opt.get("tail"), (str, int, float)):
        popt["tail"] = opt.pop("tail")
    c = Creature(species, coat=coat, size=size, view=view, facing=facing, style=style, seed=seed, **opt)
    c.pose(pose or c._pose[0], phase, **popt)
    return c.draw(cv, at, log=log, shadow=shadow, cast=cast, fit=fit)


# ===================================================================== birds
#
# Units: body length (breast to tail base) = 1. World x forward, y up, z towards the viewer.

_B = dict(
    body=(0.5, 0.32, 0.3), body_ang=22.0, breast=0.0,
    head=0.2, neck=0.06, neck_ang=70.0, head_ang=0.0, crest=0.0,
    beak=dict(kind="cone", len=0.16, depth=0.08, color="#3a3430"),
    eye=dict(size=0.04, iris="#1a120e", ring=None, at=(0.25, 0.25)),
    wing=dict(len=0.95, chord=0.42, kind="round", tips=6),
    tail=dict(len=0.45, w=0.22, ang=-8.0, kind="fan"),
    legs=dict(tibia=0.18, tarsus=0.22, r=0.022, toes=0.16, color="#8a6a50", feather=0.0),
    stand_h=0.0, swim=False,
)

_SPECIES_B = {
    "sparrow": dict(body=(0.5, 0.33, 0.31), body_ang=20.0, head=0.21, neck=0.04,
                    beak=dict(kind="cone", len=0.13, depth=0.1, color="#4a3a30"),
                    legs=dict(tibia=0.16, tarsus=0.2, r=0.02, toes=0.15, color="#b08a6a"),
                    wing=dict(len=0.85, chord=0.42, kind="round", tips=6),
                    tail=dict(len=0.5, w=0.2, ang=-4.0, kind="fan"),
                    coat=dict(back="#8a5a34", belly="#d8ccb8", head="#7a7066", cap="#8a5a34", face="#c8bca8",
                              throat="#2a2420", wing="#7a4a28", wing2="#4a3020", wing_bar="#efe6d6",
                              tail="#5a3a24", pattern="streaks", colours=["#3a2614"])),
    "robin": dict(body=(0.5, 0.36, 0.33), body_ang=26.0, head=0.21, neck=0.04,
                  beak=dict(kind="cone", len=0.13, depth=0.07, color="#2a2420"),
                  legs=dict(tibia=0.18, tarsus=0.26, r=0.018, toes=0.15, color="#9a7a62"),
                  wing=dict(len=0.85, chord=0.42, kind="round", tips=6),
                  tail=dict(len=0.5, w=0.2, ang=-10.0, kind="fan"),
                  coat=dict(back="#7a6a4e", belly="#efe8dc", head="#7a6a4e", cap="#7a6a4e", face="#e0662a",
                            throat="#e0662a", breast="#e0662a", wing="#6e5e44", wing2="#4e4232",
                            tail="#5e5040")),
    "crow": dict(body=(0.5, 0.3, 0.28), body_ang=18.0, head=0.18, neck=0.08,
                 beak=dict(kind="crow", len=0.26, depth=0.1, color="#141214"),
                 eye=dict(size=0.032, iris="#1a1416", at=(0.25, 0.25)),
                 legs=dict(tibia=0.22, tarsus=0.3, r=0.022, toes=0.2, color="#1a181a"),
                 wing=dict(len=1.05, chord=0.45, kind="fingered", tips=6),
                 tail=dict(len=0.55, w=0.24, ang=-10.0, kind="fan"),
                 coat=dict(back="#1e1c24", belly="#1e1c24", head="#1e1c24", wing="#22202c", wing2="#18161e",
                           tail="#1a1820", sheen=0.5)),
    "owl": dict(body=(0.42, 0.42, 0.36), body_ang=72.0, neck_at=(0.85, 0.05), head=0.3, neck=0.0, neck_ang=90.0, head_yaw=70.0,
                beak=dict(kind="hook", len=0.09, depth=0.07, color="#6a5a48"),
                eye=dict(size=0.08, iris="#f2b81e", ring="#f4ead6", at=(0.62, 0.15), front=True),
                legs=dict(tibia=0.1, tarsus=0.12, r=0.045, toes=0.14, color="#d8c8a8", feather=1.0),
                wing=dict(len=0.8, chord=0.5, kind="round", tips=6),
                tail=dict(len=0.12, w=0.18, ang=-8.0, kind="short"), ear_tufts=0.12, disc=True,
                coat=dict(back="#8a6a48", belly="#e8dcc4", head="#8a6a48", face="#efe4cc", wing="#7a5a3a",
                          wing2="#5a4028", tail="#6a4a2e", pattern="bars", colours=["#6a4a2a"])),
    "eagle": dict(body=(0.55, 0.32, 0.3), body_ang=48.0, head=0.18, neck=0.12, neck_ang=55.0,
                  beak=dict(kind="hook", len=0.2, depth=0.12, color="#f2c02a"),
                  eye=dict(size=0.034, iris="#d8a828", at=(0.3, 0.3), brow=True),
                  legs=dict(tibia=0.2, tarsus=0.2, r=0.04, toes=0.16, color="#f2c02a", feather=0.6),
                  wing=dict(len=1.6, chord=0.62, kind="fingered", tips=7),
                  tail=dict(len=0.5, w=0.3, ang=-25.0, kind="fan"),
                  coat=dict(back="#4a3020", belly="#4a3020", head="#f4f0e8", face="#f4f0e8", throat="#f4f0e8",
                            wing="#6a4528", wing2="#3a2616", tail="#f4f0e8")),
    "duck": dict(body=(0.55, 0.3, 0.3), body_ang=4.0, head=0.18, neck=0.22, neck_ang=78.0, head_ang=-4.0,
                 beak=dict(kind="flat", len=0.24, depth=0.07, color="#e8b82a"),
                 eye=dict(size=0.03, iris="#1a120e", at=(0.25, 0.3)),
                 legs=dict(tibia=0.14, tarsus=0.14, r=0.025, toes=0.16, color="#f08a2a", web=True),
                 wing=dict(len=0.85, chord=0.4, kind="pointed", tips=5),
                 tail=dict(len=0.22, w=0.2, ang=18.0, kind="short"),
                 coat=dict(back="#8a8278", belly="#c8c0b4", head="#1e6a3e", throat="#1e6a3e", collar="#f4f0ea",
                           breast="#6a3a2a", wing="#8a8278", wing2="#5a524a", speculum="#3a4ab8",
                           tail="#2a2624", sheen=0.6)),
    "swan": dict(body=(0.62, 0.3, 0.32), body_ang=0.0, head=0.13, neck=0.75, neck_ang=88.0, head_ang=-12.0,
                 neck_curve=True,
                 beak=dict(kind="flat", len=0.2, depth=0.06, color="#e8782a", knob="#1a1614"),
                 eye=dict(size=0.026, iris="#1a120e", at=(0.25, 0.2), mask="#1a1614"),
                 legs=dict(tibia=0.14, tarsus=0.14, r=0.03, toes=0.18, color="#2a2624", web=True),
                 wing=dict(len=0.95, chord=0.5, kind="round", tips=6, raised=0.0),
                 tail=dict(len=0.2, w=0.24, ang=20.0, kind="short"), swim=True,
                 coat=dict(back="#f4f2ee", belly="#f4f2ee", head="#f4f2ee", wing="#f0eee8", wing2="#e4e0d8",
                           tail="#f0eee8")),
    "penguin": dict(body=(0.42, 0.3, 0.3), body_ang=84.0, neck_at=(0.9, 0.05), head=0.2, neck=0.02, neck_ang=90.0, head_ang=8.0,
                    beak=dict(kind="cone", len=0.18, depth=0.06, color="#1e1c1e", stripe="#e88a2a"),
                    eye=dict(size=0.026, iris="#1a120e", at=(0.35, 0.3)),
                    legs=dict(tibia=0.04, tarsus=0.05, r=0.04, toes=0.12, color="#2a2624", web=True),
                    wing=dict(len=0.55, chord=0.16, kind="flipper", tips=0),
                    tail=dict(len=0.12, w=0.14, ang=-30.0, kind="short"),
                    coat=dict(back="#1c1e24", belly="#f4f2ee", head="#1c1e24", throat="#1c1e24", face="#1c1e24",
                              cheek="#f2b23a", wing="#1c1e24", wing2="#1c1e24", tail="#1c1e24", belly_edge=-0.2)),
    "parrot": dict(body=(0.48, 0.3, 0.28), body_ang=66.0, neck_at=(0.75, 0.3), pose="perch", head=0.24, neck=0.03, neck_ang=80.0,
                   beak=dict(kind="parrot", len=0.16, depth=0.17, color="#2a2624"),
                   eye=dict(size=0.03, iris="#f4e8c8", at=(0.3, 0.25), ring="#f4f2ee"),
                   legs=dict(tibia=0.08, tarsus=0.08, r=0.03, toes=0.12, color="#5a5a5a"),
                   wing=dict(len=0.9, chord=0.4, kind="pointed", tips=6),
                   tail=dict(len=1.1, w=0.3, ang=-10.0, kind="long"),
                   coat=dict(back="#d8282a", belly="#d8282a", head="#d8282a", face="#f4f2ee", wing="#2a6ad0",
                             wing2="#1e48a8", wing_bar="#f2c82a", covert="#f2c82a", tail="#d8282a",
                             tail2="#2a6ad0")),
    "flamingo": dict(body=(0.5, 0.26, 0.26), body_ang=6.0, head=0.12, neck=0.95, neck_ang=80.0, head_ang=-40.0,
                     neck_curve=True,
                     beak=dict(kind="flamingo", len=0.24, depth=0.09, color="#1e1a1c", base="#f2e2d0"),
                     eye=dict(size=0.022, iris="#f2c42a", at=(0.15, 0.2)),
                     legs=dict(tibia=0.55, tarsus=0.75, r=0.022, toes=0.14, color="#e88a90", web=True),
                     wing=dict(len=0.85, chord=0.44, kind="round", tips=6),
                     tail=dict(len=0.2, w=0.2, ang=-12.0, kind="short"),
                     coat=dict(back="#f29aa0", belly="#f4a8ac", head="#f29aa0", wing="#f08a92", wing2="#1e1a1c",
                               tail="#f29aa0")),
    "chicken": dict(body=(0.5, 0.38, 0.34), body_ang=16.0, head=0.17, neck=0.2, neck_ang=72.0,
                    beak=dict(kind="cone", len=0.12, depth=0.08, color="#e8c050"),
                    eye=dict(size=0.026, iris="#d8902a", at=(0.25, 0.3)),
                    legs=dict(tibia=0.2, tarsus=0.24, r=0.028, toes=0.18, color="#e8b84a"),
                    wing=dict(len=0.75, chord=0.45, kind="round", tips=6),
                    tail=dict(len=0.5, w=0.3, ang=45.0, kind="sickle"), comb=True,
                    coat=dict(back="#b8542a", belly="#8a3a1e", head="#c8642a", wing="#a8482a", wing2="#5a2a14",
                              tail="#1e2a2a", comb="#d8282a", sheen=0.3)),
}
_BIRD_ALIASES = {"bird": "sparrow", "pigeon": "crow", "raven": "crow", "blackbird": "crow", "hen": "chicken",
                 "rooster": "chicken", "macaw": "parrot", "hawk": "eagle", "falcon": "eagle", "goose": "swan",
                 "mallard": "duck", "duckling": "duck", "finch": "sparrow", "songbird": "sparrow"}


def _bparams(name, style="soft", over=None):
    P = _merge(dict(_B), {k: v for k, v in _SPECIES_B[name].items()})
    if over:
        P = _merge(P, over)
    if style == "cute":
        P["head"] *= 1.45
        e = dict(P["eye"])
        e["size"] *= 2.0
        P["eye"] = e
        P["body"] = tuple(r * 1.05 for r in P["body"])
        b = dict(P["beak"])
        b["len"] *= 0.8
        P["beak"] = b
    return P


def _wing_outline(kind, tips):
    """Wing outline in (span 0..1, chord) coordinates, both in units of the wing length (chord > 0 = leading edge):
    leading edge from the shoulder to the tip, then the feathered trailing edge back to the body."""
    if kind == "flipper":
        return [(0.0, 0.2), (0.3, 0.24), (0.7, 0.18), (1.0, 0.04), (0.85, -0.1), (0.4, -0.18), (0.0, -0.2)]
    n = max(int(tips), 3)
    if kind == "fingered":
        out = [(0.0, 0.16), (0.2, 0.22), (0.4, 0.24), (0.58, 0.22), (0.7, 0.19)]
        for i in range(n):
            t = i / (n - 1)
            ang = math.radians(28 - 62 * t)
            ln = 0.3 - 0.1 * abs(t - 0.35)
            bx, by = 0.7 + 0.02 * t, 0.15 - 0.3 * t
            out += [(bx + math.cos(ang) * ln, by + math.sin(ang) * ln),
                    (bx + math.cos(ang) * ln * 0.97 - 0.02, by + math.sin(ang) * ln - 0.035)]
            if i < n - 1:
                out.append((bx + 0.02 + math.cos(ang - 0.25) * ln * 0.35, by - 0.03 + math.sin(ang - 0.25) * ln * 0.35))
        x0, c0 = 0.68, -0.2
    else:
        lead = [(0.0, 0.15), (0.2, 0.2), (0.4, 0.21), (0.6, 0.18), (0.76, 0.13), (0.9, 0.07), (1.0, -0.01)]
        if kind == "pointed":
            lead = [(0.0, 0.15), (0.2, 0.19), (0.4, 0.19), (0.6, 0.14), (0.8, 0.07), (1.0, -0.03)]
        out = list(lead)
        for i in range(n - 2):
            t = (i + 1) / (n - 1)
            x = 1.0 - 0.3 * t
            c = -0.04 - 0.13 * t
            out += [(x - 0.03, c - 0.022), (x - 0.05, c + 0.004)]
        x0, c0 = 0.68, -0.17
    m = 6
    for i in range(m):
        t = i / (m - 1)
        x = x0 * (1 - t)
        c = c0 - 0.04 * math.sin(math.pi * t)
        out += [(x + 0.02, c - 0.028), (max(x - 0.07, 0.0), c + 0.006)]
    out.append((0.0, -0.16))
    return out


def _bird_pose(P, name, phase, opt):
    """Bird pose → dict(ang, wing (spread 0..1, lift deg), legs, ground offset, flying, swim)."""
    ang = P["body_ang"]
    sk = dict(body_ang=ang, spread=0.0, lift=0.0, flying=False, swim=False, legs="stand", neck_ang=P["neck_ang"],
              head_ang=P["head_ang"], head_yaw=opt.get("head_turn", P.get("head_yaw", 0.0)), tail_ang=0.0)
    if name in ("fly", "flying", "glide"):
        sk["flying"] = True
        sk["legs"] = "tucked"
        sk["body_ang"] = 6.0 if P["body_ang"] < 60 else 12.0
        if name == "glide":
            sk["spread"], sk["lift"] = 1.0, 16.0
        else:
            # wing beat: 0 = top of the upstroke, 0.5 = bottom of the downstroke
            c = math.cos(2 * math.pi * phase)
            sk["spread"] = 0.75 + 0.25 * abs(math.sin(2 * math.pi * phase)) + 0.1 * c
            sk["lift"] = 62.0 * c - 8.0
            sk["fold"] = 0.35 * max(0.0, math.sin(2 * math.pi * phase)) ** 2
        sk["neck_ang"] = min(P["neck_ang"], 40.0) if P["neck"] > 0.3 else P["neck_ang"] - 30
        sk["head_ang"] = 0.0 if P["neck"] <= 0.3 else -8.0
        if P["neck"] > 0.3:
            sk["neck_ang"] = 8.0
            sk["head_ang"] = 0.0
        if P["legs"]["tibia"] > 0.4:
            sk["legs"] = "trail"
    elif name == "swim":
        sk["swim"] = True
        sk["legs"] = "none"
        sk["body_ang"] = 0.0 if P["body_ang"] < 60 else 10.0
    elif name == "perch":
        sk["legs"] = "perch"
    elif name in ("peck", "graze", "sniff"):
        sk["body_ang"] = -15.0
        sk["neck_ang"] = -30.0
        sk["head_ang"] = -40.0
    elif name == "sit":
        sk["legs"] = "none"
    if "wings" in opt:
        wv = opt["wings"]
        if wv == "open":
            sk["spread"], sk["lift"] = 1.0, 35.0
        elif wv == "up":
            sk["spread"], sk["lift"] = 0.95, 70.0
    return sk


def _build_bird(name, pose, phase, opt, coat, style, params, seed):
    P = _bparams(name, style, params)
    sk = _bird_pose(P, pose, phase, opt)
    m = _Model()
    m.coat = dict(coat)
    m.coat.setdefault("base", coat.get("back", "#8a6a48"))
    m.fur_len = 0.006 if style == "soft" else 0.0
    rx, ry, rz = P["body"]
    a = sk["body_ang"]
    Rb = _rz(a)
    lg0 = P["legs"]
    ar = math.radians(a)
    down = math.sqrt((rx * math.sin(ar)) ** 2 + (ry * math.cos(ar)) ** 2)      # body extent below its centre
    hip_rel = np.array([-0.05 * rx, -down * 0.6, 0.0])
    if sk["legs"] == "perch":
        B = np.array([0.0, lg0["tibia"] * 0.45 - hip_rel[1], 0.0])
    elif sk["legs"] in ("stand",):
        B = np.array([0.0, lg0["tarsus"] + lg0["tibia"] * 0.5 - hip_rel[1], 0.0])
    else:
        B = np.zeros(3)
    back = coat.get("back", "#8a6a48")
    belly = coat.get("belly", back)
    sheen = coat.get("sheen", 0.0)
    kwb = dict(fur=0.35, pat=1.0, belly=1.0, spec=sheen * 0.4)
    m.coat["belly"] = belly
    m.coat.setdefault("belly_edge", 0.1)
    m.coat["belly_dir"] = tuple(Rb @ np.array([0.3, -1.0, 0.0]))
    # ---- body: egg with a fuller breast
    m.ell("core", B, (rx, ry, rz), back, rot=Rb, name="body", k=0.06, **kwb)
    brc = coat.get("breast")
    m.ell("core", B + Rb @ np.array([rx * 0.45, -ry * 0.05, 0]), (rx * 0.55, ry * 0.92, rz * 0.95),
          brc or back, rot=Rb, name="breast", k=0.08, **dict(kwb, belly=0.0 if brc else 1.0))
    # ---- neck and head
    hr = P["head"]
    n0 = B + Rb @ np.array([rx * P.get("neck_at", (0.62, 0.5))[0], ry * P.get("neck_at", (0.62, 0.5))[1], 0])
    na = sk["neck_ang"]
    nl = P["neck"]
    pts = [n0]
    nseg = 6 if nl > 0.3 else 2
    for i in range(nseg):
        t = (i + 1) / nseg
        if P.get("neck_curve") and nl > 0.3 and not sk["flying"]:
            # S-curve: back at the base, forward under the head
            ai = na + 35 * math.cos(math.pi * t) - 10
        else:
            ai = na
        pts.append(pts[-1] + _v3(*_dir2(ai)) * nl / nseg)
    hc = pts[-1] + _v3(*_dir2(na)) * hr * 0.35
    neck_col = coat.get("throat", coat.get("head", back)) if nl > 0.3 else coat.get("head", back)
    if nl > 0.01:
        rads = [max(hr * 0.75, ry * 0.55)] + [hr * (0.72 - 0.08 * (i / nseg)) for i in range(1, nseg + 1)]
        if nl < 0.3:
            rads[0] = max(rads[0], hr * 0.95)
        stops = None
        if coat.get("collar"):
            stops = [(0.0, back, 0.0), (0.25, coat["collar"], 0.0), (0.4, coat.get("head", back), 0.0),
                     (1.0, coat.get("head", back), 0.0)]
        m.chain("core", pts, rads, neck_col, fur=0.3, pat=0.0, belly=0.0, k=0.05, stops=stops, name="neck",
                spec=sheen * 0.5)
    yaw = sk["head_yaw"]
    Rh = _ry(yaw) @ _rz(sk["head_ang"])
    head_col = coat.get("head", back)
    m.ell("core", hc, (hr * 1.05, hr, hr * 0.92), head_col, rot=Rh, name="head", k=0.05, fur=0.3, pat=0.0,
          belly=0.0, spec=sheen * 0.6)
    fwd = Rh @ np.array([1.0, 0, 0])
    upv = Rh @ np.array([0, 1.0, 0])
    side = Rh @ np.array([0, 0, 1.0])
    if coat.get("cap") and coat.get("cap") != head_col:
        m.ell("core", hc + upv * hr * 0.35 - fwd * hr * 0.1, (hr * 0.85, hr * 0.62, hr * 0.8), coat["cap"], rot=Rh,
              name="cap", k=0.02, fur=0.3, pat=0.0, belly=0.0)
    if coat.get("face"):
        for s in (1, -1):
            c = hc + fwd * hr * 0.25 - upv * hr * 0.05 + side * s * hr * 0.42
            m.ell("core", c, (hr * 0.62, hr * 0.62, hr * 0.5), coat["face"], rot=Rh, name="face", k=0.02, fur=0.2,
                  pat=0.0, belly=0.0)
    if coat.get("throat") and nl <= 0.3:
        c = hc + fwd * hr * 0.3 - upv * hr * 0.55
        m.ell("core", c, (hr * 0.55, hr * 0.45, hr * 0.7), coat["throat"], rot=Rh, name="throat", k=0.03,
              fur=0.2, pat=0.0, belly=0.0)
    if coat.get("cheek"):
        for s in (1, -1):
            c = hc - fwd * hr * 0.15 - upv * hr * 0.1 + side * s * hr * 0.55
            m.ell("core", c, (hr * 0.4, hr * 0.32, hr * 0.42), coat["cheek"], rot=Rh, name="cheek", k=0.015,
                  fur=0.0, pat=0.0, belly=0.0)
    if P.get("disc"):
        for s in (1, -1):
            c = hc + fwd * hr * 0.55 + side * s * hr * 0.3
            m.ell("core", c, (hr * 0.32, hr * 0.62, hr * 0.5), coat.get("face", "#efe4cc"), rot=Rh, name="disc",
                  k=0.03, fur=0.2, pat=0.0, belly=0.0)
    if P.get("ear_tufts"):
        for s in (1, -1):
            b0 = hc + upv * hr * 0.75 + side * s * hr * 0.55
            m.cone(m.group("tuft%d" % s, "core", 0.02), b0, b0 + _unit(upv * 1.0 + side * s * 0.45 - fwd * 0.2)
                   * P["ear_tufts"], hr * 0.2, hr * 0.03, head_col, fur=0.6, pat=1.0, belly=0.0, k=0.01)
    if P.get("comb"):
        cc = coat.get("comb", "#d8282a")
        for i in range(4):
            c = hc + upv * hr * (0.95 + 0.1 * math.sin(i)) + fwd * hr * (0.35 - 0.3 * i)
            m.ell("core", c, (hr * 0.2, hr * 0.32, hr * 0.12), cc, rot=Rh, name="comb", k=0.02, fur=0.0, pat=0.0,
                  belly=0.0, spec=0.2)
        for s in (1, -1):
            c = hc + fwd * hr * 0.75 - upv * hr * 0.95 + side * s * hr * 0.12
            m.ell("core", c, (hr * 0.16, hr * 0.32, hr * 0.12), cc, rot=Rh, name="wattle", k=0.01, fur=0.0,
                  pat=0.0, belly=0.0)
    if P.get("crest"):
        b0 = hc + upv * hr * 0.8 - fwd * hr * 0.2
        m.cone("core", b0, b0 + _unit(upv - fwd * 0.6) * P["crest"], hr * 0.35, hr * 0.06, coat.get("cap", head_col),
               fur=0.8, k=0.02)
    # ---- beak
    bk = P["beak"]
    bl, bd = bk["len"], bk["depth"]
    bcol = coat.get("beak", bk["color"])
    b0 = hc + fwd * hr * 0.82 - upv * hr * 0.12
    kind = bk["kind"]
    kbk = dict(fur=0.0, pat=0.0, belly=0.0, spec=0.35)
    if kind in ("cone", "crow"):
        tip = b0 + fwd * bl - upv * bd * (0.25 if kind == "crow" else 0.1)
        m.cone("core", b0 - fwd * bd * 0.3, tip, bd * 0.55, bd * 0.06, bcol, name="beak", k=0.012, **kbk)
        if bk.get("stripe"):
            m.cone("core", b0 - fwd * bd * 0.2 - upv * bd * 0.15, b0 + fwd * bl * 0.6 - upv * bd * 0.22,
                   bd * 0.32, bd * 0.12, bk["stripe"], k=0.005, **kbk)
    elif kind in ("hook", "parrot"):
        top = b0 + fwd * bl * 0.55 + upv * bd * 0.15
        tip = b0 + fwd * bl * 0.95 - upv * bd * (1.2 if kind == "parrot" else 0.75)
        m.chain("core", [b0 - fwd * bd * 0.25, top, tip], [bd * 0.6, bd * 0.42, bd * 0.06], bcol, name="beak",
                k=0.012, **kbk)
        if kind == "parrot":
            m.cone("core", b0 - upv * bd * 0.45, b0 + fwd * bl * 0.45 - upv * bd * 0.65, bd * 0.4, bd * 0.18, bcol,
                   k=0.006, **kbk)
    elif kind == "flat":
        tip = b0 + fwd * bl - upv * bd * 0.35
        Rk = Rh @ _rz(-8)
        for i in range(3):
            t = i / 2
            c = b0 + (tip - b0) * (0.2 + 0.4 * t)
            m.ell("core", c, (bl * 0.32, bd * 0.42, bd * (0.75 + 0.1 * t)), bcol, rot=Rk, name="beak", k=0.012,
                  **kbk)
        if bk.get("knob"):
            m.ell("core", b0 + upv * bd * 0.3 - fwd * bl * 0.05, (bd * 0.6, bd * 0.55, bd * 0.5), bk["knob"],
                  rot=Rh, k=0.01, **kbk)
        tip = tip + fwd * bl * 0.05
    elif kind == "flamingo":
        mid = b0 + fwd * bl * 0.5 + upv * bd * 0.05
        tip = mid + _unit(fwd * 0.5 - upv * 1.0) * bl * 0.55
        m.chain("core", [b0 - fwd * bd * 0.2, mid, tip], [bd * 0.55, bd * 0.48, bd * 0.12], bk.get("base", bcol),
                name="beak", k=0.012, stops=[(0.0, bk.get("base", bcol), 0.0), (0.5, bk.get("base", bcol), 0.0),
                                             (0.62, bcol, 0.0), (1.0, bcol, 0.0)], **kbk)
    else:
        tip = b0 + fwd * bl
        m.cone("core", b0, tip, bd * 0.5, bd * 0.06, bcol, name="beak", k=0.01, **kbk)
    m.anchors["beak"] = tip
    m.anchors["mouth"] = tip
    m.anchors["head"] = hc + upv * hr
    # mouth line on the beak
    if kind in ("cone", "crow", "flat", "hook") and style != "flat":
        ln = [b0 - fwd * bd * 0.2 + side * bd * 0.3, b0 + fwd * bl * 0.55 - upv * bd * 0.12 + side * bd * 0.15]
        m.decal("line", np.array(ln), width=0.006, color="#000000", opacity=0.35, tol=0.03)
    # ---- eyes
    e = P["eye"]
    for s in (1, -1):
        if e.get("front"):
            d = np.array([0.75, 0.15, 0.55 * s])
        else:
            d = np.array([e["at"][0], e["at"][1], 0.9 * s])
        p, n = _surf(hc, (hr * 1.05, hr, hr * 0.92), Rh, d)
        if s == 1:
            m.anchors["eye"] = p
        if e.get("ring"):
            m.decal("spot", p + n * 0.002, size=e["size"] * 1.45, color=e["ring"], opacity=0.95, tol=0.06)
        if e.get("mask"):
            m.decal("spot", p - fwd * e["size"] * 0.3, size=e["size"] * 1.8, color=e["mask"], squash=0.7,
                    opacity=0.9, tol=0.06)
        m.decal("eye", p + n * 0.004, n=n, fwd=fwd, size=e["size"], shape="round", iris=coat.get("eye", e["iris"]),
                pupil="round", pupil_r=0.55 if e["iris"] not in ("#1a120e", "#1a1416") else 0.4, look=(0.05, 0.0),
                rim_w=0.12, glint=0.3, tol=0.1, lid=0.4)
        if e.get("brow"):
            m.ell("core", p + n * 0.01 + upv * e["size"] * 0.9, (e["size"] * 1.6, e["size"] * 0.6, e["size"] * 0.9),
                  head_col, rot=Rh, k=0.01, fur=0.2, pat=0.0, belly=0.0)
    # ---- wings
    w = P["wing"]
    span = w["len"]
    chord = w["chord"]
    outline = _wing_outline(w["kind"], w["tips"])
    wcol = coat.get("wing", back)
    wcol2 = coat.get("wing2", wcol)
    covert = coat.get("covert", wcol)
    for s in (1, -1):
        g = m.group("wing%d" % s, "core", 0.03)
        tone = 1.0 if s == 1 else 0.82
        sh = B + Rb @ np.array([rx * 0.2, ry * 0.55, s * rz * 0.5])
        if sk["flying"] or sk["spread"] > 0:
            lift = math.radians(sk["lift"])
            spread = sk["spread"]
            fold = sk.get("fold", 0.0)
            # span direction: out to the side, raised by ``lift``; chord direction: backwards along the body
            u = _unit(np.array([-0.15 - 0.5 * fold, math.sin(lift), s * math.cos(lift)]))
            if not sk["flying"]:
                u = _unit(np.array([-0.4, math.sin(lift) + 0.2, s * math.cos(lift) * 0.8]))
            v = _unit(np.array([1.0, 0.0, 0.0]) - u * u[0])
            L = span * (0.55 + 0.45 * spread) * (1 - 0.3 * fold)
            cs = L * chord / 0.42 * 0.85
            # sweep the outer wing back a little and bend it at the wrist
            pts = []
            for a_, c_ in outline:
                q = sh + u * (a_ * L) + v * (c_ * cs * (1 - 0.3 * a_ * fold)) - v * (0.12 * L * max(a_ - 0.5, 0))
                pts.append(q)
            nrm = _unit(np.cross(u, v))
            m.flat(g, pts, wcol2, bulge=0.25, tone=tone, belly=0.0, fur=0.2, pat=0.0, k=0.03, axis=-v,
                   name="wing", smooth=1)
            cov = [(0.0, 0.15), (0.2, 0.2), (0.4, 0.21), (0.6, 0.18), (0.66, 0.1), (0.6, 0.0), (0.5, -0.04),
                   (0.4, -0.02), (0.3, -0.06), (0.2, -0.04), (0.1, -0.07), (0.0, -0.06)]
            for side_ in (1, -1):
                pts2 = [sh + u * (a_ * L) + v * (c_ * cs) - v * (0.12 * L * max(a_ - 0.5, 0)) + nrm * side_ * 0.006
                        for a_, c_ in cov]
                m.flat(g, pts2, covert if covert != wcol else wcol, bulge=0.25, tone=tone, belly=0.0, fur=0.3,
                       pat=0.5, k=0.006, axis=-v, name="covert", smooth=1)
            if s == 1:
                m.anchors["wing"] = sh + u * L
        else:
            # folded along the side of the body: a smooth wing volume, darker primaries reaching over the tail
            if w["kind"] == "flipper":
                v = _unit(np.array([-0.15, -1.0, 0.0]))
                L = span * 0.85
            else:
                v = Rb @ _unit(np.array([-1.0, -0.18, 0.0]))
                L = span * 0.68
            n0 = Rb @ np.array([0.0, 0.0, float(s)])
            u2 = _unit(np.cross(n0, v))
            if u2[1] < 0:
                u2 = -u2
            wd = chord * (0.62 if w["kind"] != "flipper" else 0.5)
            R = np.stack([v, u2, n0], 1)
            sp_, sn_ = _surf(B, (rx, ry, rz), Rb, (0.35, 0.3, 0.75 * s) if w["kind"] != "flipper" else
                             (0.5, -0.15, 0.9 * s))
            base0 = sp_ - sn_ * rz * 0.14
            kw_w = dict(tone=tone, belly=0.0, fur=0.25, pat=0.5, axis=v)
            m.ell(g, base0 + v * L * 0.42, (L * 0.5, wd * 0.5, rz * 0.32), wcol, rot=R, name="wing", k=0.02, **kw_w)
            if w["kind"] != "flipper":
                m.ell(g, base0 + v * L * 0.78 - u2 * wd * 0.12 + n0 * rz * 0.02, (L * 0.34, wd * 0.26, rz * 0.16),
                      wcol2, rot=R @ _rz(-4), name="primaries", k=0.012, **dict(kw_w, pat=0.0))
            for key, (t0, c0, rr0) in (("wing_bar", (0.42, 0.05, 0.3)), ("speculum", (0.55, -0.12, 0.26)),
                                        ("covert", (0.22, 0.12, 0.34))):
                colk = covert if key == "covert" else coat.get(key)
                if not colk or (key == "covert" and covert == wcol):
                    continue
                m.ell(g, base0 + v * L * t0 + u2 * wd * c0 + n0 * rz * 0.12, (L * rr0, wd * 0.14, rz * 0.2), colk,
                      rot=R, name=key, k=0.004, **dict(kw_w, pat=0.0, fur=0.0))
            if s == 1:
                m.anchors["wing"] = base0 + v * L * 0.5
    # ---- tail
    t = P["tail"]
    tb = B + Rb @ np.array([-rx * 0.8, ry * 0.1, 0])
    ta = a + t["ang"] + sk.get("tail_ang", 0.0)
    if sk["flying"]:
        ta = a + t["ang"] * 0.3
    tdir = _v3(*_dir2(180 + ta))
    tw = t["w"] * (1.6 if sk["flying"] else 1.0)
    tcol = coat.get("tail", back)
    g = m.group("tail", "core", 0.03)
    up_t = _unit(np.cross(np.array([0, 0, 1.0]), tdir)) * -1
    if up_t[1] < 0:
        up_t = -up_t
    side_t = _unit(np.array([0, 0, 1.0]) * 0.85 + up_t * 0.35)
    if t["kind"] == "long":
        ol = None
        tp = [tb + tdir * (t["len"] * i / 5) for i in range(6)]
        stops = [(0.0, tcol, 0.0), (0.55, tcol, 0.0), (0.75, coat.get("tail2", tcol), 0.0),
                 (1.0, coat.get("tail2", tcol), 0.0)]
        m.chain(g, tp, [tw * 0.42, tw * 0.4, tw * 0.34, tw * 0.26, tw * 0.16, tw * 0.04], tcol, fur=0.1, pat=0.0,
                belly=0.0, k=0.02, stops=stops, name="tail")
    elif t["kind"] == "sickle":
        ol = None
    elif t["kind"] == "short":
        ol = [(0.0, -0.5), (0.7, -0.4), (1.0, 0.0), (0.7, 0.4), (0.0, 0.5)]
    else:
        ol = [(0.0, -0.45), (0.85, -0.55), (1.0, -0.3), (1.02, 0.0), (1.0, 0.3), (0.85, 0.55), (0.0, 0.45)]
    if ol:
        pts = [tb + tdir * (x * t["len"]) + side_t * (y * tw) for x, y in ol]
        m.flat(g, pts, tcol, bulge=0.6, belly=0.0, fur=0.2, pat=0.5, k=0.03, axis=-tdir, name="tail",
               stops=None)
        if coat.get("tail2"):
            pts2 = [tb + tdir * (x * t["len"] * 0.98) + side_t * (y * tw * 0.6) + up_t * 0.004 for x, y in ol
                    if x > 0.4] + [tb + tdir * t["len"] * 0.45]
            m.flat(g, pts2, coat["tail2"], bulge=0.3, belly=0.0, fur=0.0, pat=0.0, k=0.003)
    elif t["kind"] == "sickle":
        # rooster: a tall arching plume of sickle feathers
        for j_, (L, ah, rr_, zz) in enumerate(((0.8, 80.0, 0.09, -0.06), (0.72, 62.0, 0.085, 0.0),
                                               (0.6, 45.0, 0.08, 0.06))):
            pts = [tb + Rb @ np.array([rx * 0.1, 0.0, 0.0])]
            for i in range(6):
                aa = 180 - ah + 32 * i
                pts.append(pts[-1] + _v3(*_dir2(aa)) * L / 6)
            pts = [p_ + np.array([0, 0, zz]) for p_ in pts]
            m.chain(g, pts, [rr_ * 1.2] + [rr_ * (1 - i / 8) for i in range(1, 7)], tcol, fur=0.1, pat=0.0,
                    belly=0.0, k=0.03, spec=0.6, tone=1.0 - 0.08 * j_)
    m.anchors["tail"] = tb + tdir * t["len"]
    # ---- legs and feet
    lg = P["legs"]
    lcol = coat.get("legs", lg["color"])
    feather = lg.get("feather", 0.0)
    legs_mode = sk["legs"]
    contacts = []
    if legs_mode not in ("none", "tucked"):
        for s in (1, -1):
            g = m.group("leg%d" % s, "core", 0.03)
            tone = 1.0 if s == 1 else 0.82
            hip = B + hip_rel + np.array([0.0, 0.0, s * rz * 0.38])
            if legs_mode == "trail":
                kn = hip + np.array([-lg["tibia"] * 0.9, -lg["tibia"] * 0.3, 0])
                ft = kn + np.array([-lg["tarsus"], 0.02, 0])
                m.cone(g, hip, kn, lg["r"] * 1.5, lg["r"], lcol, tone=tone, fur=0.0, pat=0.0, belly=0.0, k=0.01)
                m.cone(g, kn, ft, lg["r"], lg["r"] * 0.8, lcol, tone=tone, fur=0.0, pat=0.0, belly=0.0, k=0.005)
                continue
            if legs_mode == "perch":
                ank = hip + np.array([-0.02, -lg["tibia"] * 0.6, 0])
                foot = np.array([hip[0] + 0.04, 0.0, hip[2]])
            else:
                foot = np.array([hip[0] + (0.03 if s == 1 else -0.03), 0.0, hip[2]])
                ank = foot + np.array([-0.02, lg["tarsus"], 0])
            # hidden thigh, then the visible tarsus
            m.cone(g, hip, ank, lg["r"] * 2.2, lg["r"] * (1.6 if feather else 1.1), coat.get("thigh", belly if
                   feather else lcol), tone=tone, fur=0.4 if feather else 0.0, pat=0.0, belly=0.0, k=0.02)
            m.cone(g, ank, foot + np.array([0, lg["r"] * 0.6, 0]), lg["r"] * (1.4 if feather else 1.0),
                   lg["r"] * (1.2 if feather else 0.8), belly if feather > 0.9 else lcol, tone=tone,
                   fur=0.3 if feather > 0.9 else 0.0, pat=0.0, belly=0.0, k=0.01, spec=0.1)
            # toes: three forward, one back
            tl = lg["toes"]
            fp = foot + np.array([0, lg["r"] * 0.5, 0])
            for ang_t, ln in ((-25.0, 1.0), (0.0, 1.15), (25.0, 1.0), (180.0, 0.55)):
                d = np.array([math.cos(math.radians(ang_t)), 0.0, math.sin(math.radians(ang_t)) * 0.8])
                if legs_mode == "perch" and ang_t != 180.0:
                    tp = [fp, fp + d * tl * 0.5 + np.array([0, 0.01, 0]), fp + d * tl * 0.75 - np.array([0, 0.05, 0])]
                else:
                    tp = [fp, fp + d * tl * ln * 0.55 + np.array([0, -lg["r"] * 0.2, 0]), fp + d * tl * ln]
                m.chain(g, tp, [lg["r"] * 0.75, lg["r"] * 0.6, lg["r"] * 0.35], lcol, tone=tone, fur=0.0, pat=0.0,
                        belly=0.0, k=0.006, spec=0.1)
            if lg.get("web") and legs_mode != "perch":
                pw = [fp, fp + np.array([math.cos(math.radians(-25)), 0, math.sin(math.radians(-25)) * 0.8]) * tl,
                      fp + np.array([tl * 1.15, 0, 0]),
                      fp + np.array([math.cos(math.radians(25)), 0, math.sin(math.radians(25)) * 0.8]) * tl]
                pw = [q + np.array([0, lg["r"] * 0.2, 0]) for q in pw]
                m.flat(g, pw, lcol, bulge=0.2, tone=tone * 0.92, fur=0.0, pat=0.0, belly=0.0, k=0.004)
            contacts.append(foot)
            m.anchors["foot" if s == 1 else "foot_far"] = foot
    if legs_mode == "perch" and opt.get("branch", True):
        # a short twig to stand on
        bc = coat.get("branch", "#6e4e32")
        g = m.group("branch", "core", 0.0)
        y = -lg["r"] * 1.6
        m.cone(g, np.array([-0.55, y + 0.02, -0.1]), np.array([0.5, y - 0.01, 0.15]), 0.045, 0.035, bc, fur=0.0,
               pat=0.0, belly=0.0, spec=0.1, k=0.0, name="branch")
        m.cone(g, np.array([0.2, y, 0.08]), np.array([0.42, y + 0.16, 0.12]), 0.022, 0.012, bc, fur=0.0, pat=0.0,
               belly=0.0, k=0.01)
    m.anchors["back"] = B + Rb @ np.array([0, ry, 0])
    m.anchors["center"] = B
    # ---- place: standing birds stand on y = 0 at the origin; flying / swimming birds are centred
    if sk["flying"]:
        m.grounded = False
        _shift(m, -B)
    elif sk["swim"] or legs_mode == "none":
        lo = _model_bottom(m)
        if sk["swim"]:
            wl = -ry * 0.35
            _shift(m, np.array([0.0, -wl, 0.0]))
            m.clip_y = 0.0
            m.grounded = False
        else:
            _shift(m, np.array([0.0, -lo, 0.0]))
    else:
        cx = float(np.mean([c[0] for c in contacts])) if contacts else 0.0
        _shift(m, np.array([-cx, 0.0, 0.0]))
        for c in contacts:
            m.shadow_pts.append(np.array([c[0] - cx, 0.0, c[2]]))
        lo = _model_bottom(m)
        if lo < -0.02:
            _shift(m, np.array([0.0, -lo - 0.0, 0.0]))
    m.pitch = (-55.0 if pose == "glide" else -30.0) if sk["flying"] else 4.0
    if sk["flying"]:
        m.yaw_extra = 22.0
    return m


# ===================================================================== fish and sea life
#
# Units: body length (snout to tail base) = 1. World x forward, y up, z towards the viewer. The model is centred on
# the body, so ``at`` is the middle of the animal in the water.

_SPECIES_S = {
    "fish": dict(body=dict(depth=0.36, width=0.42, peak=0.42, nose=0.5, tail=0.1),
                 fins=dict(caudal="forked", caudal_len=0.32, dorsal=(0.3, 0.72, 0.16), anal=(0.62, 0.84, 0.1),
                           pectoral=0.16, pelvic=0.1),
                 eye=0.055, coat=dict(base="#7aa0b8", belly="#e8eef0", back="#4a6a84", fin="#8aa8b8",
                                      pattern="scales")),
    "goldfish": dict(body=dict(depth=0.5, width=0.45, peak=0.38, nose=0.65, tail=0.14),
                     fins=dict(caudal="veil", caudal_len=0.62, dorsal=(0.3, 0.68, 0.24), anal=(0.62, 0.82, 0.16),
                               pectoral=0.16, pelvic=0.14, fin_alpha=0.75),
                     eye=0.075, coat=dict(base="#f28a1e", belly="#f6c070", back="#e0661a", fin="#f6a040",
                                          pattern="scales")),
    "clownfish": dict(body=dict(depth=0.42, width=0.42, peak=0.42, nose=0.7, tail=0.16),
                      fins=dict(caudal="round", caudal_len=0.28, dorsal=(0.24, 0.8, 0.16), anal=(0.6, 0.84, 0.14),
                                pectoral=0.17, pelvic=0.12, edge="#1a1414"),
                      eye=0.07, coat=dict(base="#f2701e", belly="#f28a3a", back="#e8601a", fin="#f2701e",
                                          pattern="bands", bands=(0.2, 0.5, 0.86), band_color="#f6f4f0",
                                          band_edge="#1a1414")),
    "shark": dict(body=dict(depth=0.22, width=0.24, peak=0.36, nose=0.25, tail=0.05, snout=True),
                  fins=dict(caudal="shark", caudal_len=0.42, dorsal=(0.34, 0.5, 0.26), dorsal_kind="shark",
                            anal=(0.78, 0.84, 0.05), pectoral=0.3, pelvic=0.08, second_dorsal=True, gills=5),
                  eye=0.03, mouth="under",
                  coat=dict(base="#7c8a98", belly="#eef0f0", back="#5c6a78", fin="#6c7a88", belly_edge=0.1)),
    "dolphin": dict(body=dict(depth=0.26, width=0.26, peak=0.38, nose=0.35, tail=0.06, beak=0.12, melon=True),
                    fins=dict(caudal="fluke", caudal_len=0.34, dorsal=(0.42, 0.6, 0.2), dorsal_kind="curved",
                              anal=None, pectoral=0.18, pelvic=0.0),
                    eye=0.03, mouth="smile", smooth=True,
                    coat=dict(base="#7a8a9c", belly="#e6eaee", back="#4e5e70", fin="#5a6a7c", belly_edge=0.05)),
    "whale": dict(body=dict(depth=0.34, width=0.34, peak=0.42, nose=0.25, tail=0.06, grooves=True),
                  fins=dict(caudal="fluke", caudal_len=0.36, dorsal=(0.64, 0.72, 0.05), dorsal_kind="curved",
                            anal=None, pectoral=0.42, pelvic=0.0),
                  eye=0.022, mouth="whale", smooth=True,
                  coat=dict(base="#3e4c62", belly="#c8d0d8", back="#2c3a4e", fin="#3e4c62", belly_edge=-0.1,
                            flipper="#d8dee4")),
    "jellyfish": dict(special="jelly", coat=dict(base="#f2a6d0", fin="#f6c8e2", belly="#f8d8ea")),
    "octopus": dict(special="octopus", coat=dict(base="#d8603a", belly="#f2a07a", spots="#b84a2a")),
    "turtle": dict(special="turtle", coat=dict(base="#7a8a4a", shell="#6a5a32", scute="#9a7a42",
                                               belly="#e8dcb0", spots="#4a5a2a")),
}
_SEA_ALIASES = {"tropical fish": "clownfish", "nemo": "clownfish", "sea turtle": "turtle", "tortoise": "turtle",
                "jelly": "jellyfish", "orca": "dolphin", "porpoise": "dolphin", "carp": "goldfish", "koi": "goldfish",
                "salmon": "fish", "trout": "fish", "humpback": "whale"}


def _fish_profile(B, x):
    """Top and bottom of the body (y) at x in 0 (tail base) .. 1 (snout), for body params B."""
    d, peak, nose, tl = B["depth"], B["peak"], B["nose"], B["tail"]
    xp = 1 - peak
    if np.isscalar(x):
        x = float(x)
    t = np.clip(x, 0, 1)
    # thickness envelope: narrow at the tail base, full at the peak, rounded towards the snout
    rise = np.where(t < xp, tl + (1 - tl) * np.sin(0.5 * np.pi * t / max(xp, 1e-3)) ** 1.4, 0)
    fall = np.where(t >= xp, np.sqrt(np.clip(1 - (np.clip(t - xp, 0, None) / max(1 - xp, 1e-3)) ** (2.0 + 2.5 * nose), 0, 1)), 0)
    h = d * np.where(t < xp, rise, fall)
    off = 0.08 * d * np.sin(np.pi * t)
    return off + h / 2, off - h / 2


def _build_fish(name, P, pose, phase, opt, coat, style, seed):
    m = _Model()
    B = P["body"]
    F = P["fins"]
    m.coat = dict(coat)
    base = coat.get("base", "#7aa0b8")
    back = coat.get("back", base)
    m.coat.setdefault("belly_edge", 0.05)
    m.fur_len = 0.0
    m.grounded = False
    smooth = P.get("smooth", False)
    # lateral swimming wave (z): grows towards the tail
    swing = 0.0 if pose in ("still", "rest") else 1.0

    def bend(xb):
        u = 1 - xb
        return swing * 0.09 * u * u * math.sin(2 * math.pi * phase - 2.2 * u)

    n = 18
    xs = np.linspace(0.02, 0.98, n)
    kwb = dict(fur=0.0, pat=1.0, belly=1.0, spec=0.45 if smooth else 0.25)
    for xb in xs:
        top, bot = _fish_profile(B, xb)
        ry = max((top - bot) / 2, 0.012)
        cy = (top + bot) / 2
        rz = ry * B["width"] / max(B["depth"], 1e-3) * B["depth"] * 2.2 / 2 if False else ry * B["width"] / 0.5
        rz = max(min(rz, ry * 1.2), 0.012)
        c = np.array([xb - 0.5, cy, bend(xb)])
        m.ell("core", c, (0.12, ry, rz), back, name="body", k=0.07, axis=np.array([1.0, 0, 0]), **kwb)
    if B.get("beak"):
        top, bot = _fish_profile(B, 0.98)
        c0 = np.array([0.45, (top + bot) / 2 - 0.03, 0])
        m.cone("core", c0, c0 + np.array([B["beak"], -0.015, 0]), 0.045, 0.028, base, name="beak", k=0.04, **kwb)
    if B.get("melon"):
        m.ell("core", np.array([0.36, 0.05, 0]), (0.12, 0.1, 0.1), back, name="melon", k=0.06, **kwb)
    m.coat["belly_dir"] = (0.0, -1.0, 0.0)
    fin_col = coat.get("fin", base)
    fa = F.get("fin_alpha", 0.92)
    edge = F.get("edge")

    def fin(pts, col=fin_col, opa=fa, k=0.02, group=None, rays=True, bul=0.3):
        g = group or "core"
        pr = m.flat(g, pts, col, bulge=bul * 0.5, fur=0.0, pat=0.0, belly=0.0, spec=0.2, k=k, opa=opa, name="fin",
                    smooth=2)
        if rays and style != "flat" and len(pts) >= 3:
            p0 = pts[0]
            for q in pts[1:-1:max(1, len(pts) // 6)]:
                m.decal("line", np.array([p0 + (q - p0) * 0.1, p0 + (q - p0) * 0.92]), width=0.004,
                        color="#000000", opacity=0.18, tol=0.04)
        if edge:
            m.decal("line", np.array(pts[1:-1]), width=0.012, color=edge, opacity=0.9, tol=0.04)
        return pr

    # ---- tail fin
    tb = np.array([-0.5, _fish_profile(B, 0.02)[0] * 0.5 + _fish_profile(B, 0.02)[1] * 0.5, bend(0.0)])
    L = F["caudal_len"]
    zs = bend(-0.25) - bend(0.0)
    kind = F["caudal"]
    # the fin plane turns with the tail beat: it is seen narrower when swung towards / away from the viewer
    sw = swing * 0.5 * math.sin(2 * math.pi * phase - 2.6)
    cdir = _unit(np.array([-1.0, 0.0, sw]))
    upv = np.array([0, 1.0, 0])

    def tp(x, y):
        return tb + cdir * (x * L) + upv * (y * L)
    g = m.group("tailfin", "core", 0.03)
    if kind == "forked":
        ol = [(0, 0.12), (0.7, 0.55), (1.0, 0.62), (0.75, 0.25), (0.62, 0.0), (0.75, -0.25), (1.0, -0.62),
              (0.7, -0.55), (0, -0.12)]
    elif kind == "round":
        ol = [(0, 0.14), (0.45, 0.42), (0.85, 0.36), (1.0, 0.0), (0.85, -0.36), (0.45, -0.42), (0, -0.14)]
    elif kind == "veil":
        ol = [(0, 0.15), (0.35, 0.5), (0.75, 0.62), (1.0, 0.5), (0.92, 0.15), (0.82, 0.0), (1.05, -0.3),
              (0.95, -0.62), (0.55, -0.6), (0, -0.15)]
    elif kind == "shark":
        ol = [(0, 0.12), (0.55, 0.62), (1.0, 1.0), (0.8, 0.55), (0.45, 0.05), (0.62, -0.4), (0.45, -0.42),
              (0, -0.1)]
    elif kind == "fluke":
        # horizontal flukes: a wide shape in the x-z plane, seen almost edge-on from the side
        sidev = np.array([0, 0.75, 1.0])
        sidev = _unit(sidev)
        olf = [(0, 0.08), (0.45, 0.4), (0.95, 0.75), (0.85, 0.3), (0.95, 0.0), (0.85, -0.3), (0.95, -0.75),
               (0.45, -0.4), (0, -0.08)]
        pts = [tb + cdir * (x * L) + sidev * (y * L) for x, y in olf]
        fin(pts, col=back, opa=1.0, group=g, rays=False, bul=0.6)
        ol = None
    else:
        ol = [(0, 0.12), (1.0, 0.45), (0.85, 0.0), (1.0, -0.45), (0, -0.12)]
    if ol:
        fin([tp(x, y) for x, y in ol], group=g)
    m.anchors["tail"] = tb + cdir * L
    # ---- dorsal and anal fins
    if F.get("dorsal"):
        x0, x1, hgt = F["dorsal"]
        a = np.array([0.5 - x0, _fish_profile(B, 1 - x0)[0] - 0.01, bend(1 - x0)])
        b = np.array([0.5 - x1, _fish_profile(B, 1 - x1)[0] - 0.01, bend(1 - x1)])
        dk = F.get("dorsal_kind", "soft")
        if dk == "shark":
            pts = [a, a + np.array([-0.06, hgt, 0]), a + np.array([-0.1, hgt * 0.92, 0]), b + np.array([0.0, 0.02, 0]),
                   b]
        elif dk == "curved":
            pts = [a, a + np.array([-0.05, hgt * 0.7, 0]), a + np.array([-0.12, hgt, 0]),
                   a + np.array([-0.1, hgt * 0.6, 0]), b]
        else:
            pts = [a, a + np.array([-0.02, hgt * 0.8, 0]), a + np.array([-0.08, hgt, 0]),
                   (a + b) / 2 + np.array([0, hgt * 0.75, 0]), b + np.array([0.04, hgt * 0.45, 0]), b]
        fin(pts, col=fin_col if dk == "soft" else back, opa=fa if dk == "soft" else 1.0, rays=dk == "soft",
            bul=0.3 if dk == "soft" else 0.7)
        if F.get("second_dorsal"):
            c = np.array([-0.33, _fish_profile(B, 0.17)[0] - 0.005, bend(0.17)])
            fin([c + np.array([0.03, 0, 0]), c + np.array([-0.0, 0.05, 0]), c + np.array([-0.04, 0.0, 0])],
                col=back, opa=1.0, rays=False, bul=0.6)
    if F.get("anal"):
        x0, x1, hgt = F["anal"]
        a = np.array([0.5 - x0, _fish_profile(B, 1 - x0)[1] + 0.01, bend(1 - x0)])
        b = np.array([0.5 - x1, _fish_profile(B, 1 - x1)[1] + 0.01, bend(1 - x1)])
        fin([a, a + np.array([-0.04, -hgt, 0]), b + np.array([0.02, -hgt * 0.8, 0]), b])
    # ---- pectoral (near side) and pelvic fins
    if F.get("pectoral"):
        L2 = F["pectoral"]
        top, bot = _fish_profile(B, 0.72)
        p0 = np.array([0.22, bot + (top - bot) * 0.35, (top - bot) * B["width"] * 0.95])
        flap = 0.4 * math.sin(2 * math.pi * phase * 2) * swing
        col = coat.get("flipper", fin_col)
        if name == "whale":
            pts = [p0 + np.array([0.04, 0, 0]), p0 + np.array([-0.06, -0.14, 0.08]),
                   p0 + np.array([-0.26, -L2 * 0.8, 0.12]), p0 + np.array([-0.3, -L2 * 0.76, 0.12]),
                   p0 + np.array([-0.2, -0.16, 0.06]), p0 + np.array([-0.08, -0.02, 0.04])]
            fin(pts, col=col, opa=1.0, rays=False, bul=0.8)
        else:
            pts = [p0, p0 + np.array([-L2 * 0.5, L2 * (0.1 + flap), 0.05]), p0 + np.array([-L2, -L2 * 0.3, 0.06]),
                   p0 + np.array([-L2 * 0.6, -L2 * 0.55, 0.04]), p0 + np.array([0.0, -0.03, 0.01])]
            fin(pts, col=col, opa=fa if not smooth else 1.0, rays=not smooth and name != "shark",
                bul=0.6 if (smooth or name == "shark") else 0.3, group=m.group("pec", "core", 0.01))
    if F.get("pelvic"):
        L3 = F["pelvic"]
        top, bot = _fish_profile(B, 0.55)
        p0 = np.array([0.05, bot + 0.01, 0.03])
        fin([p0, p0 + np.array([-L3 * 0.9, -L3 * 0.6, 0.02]), p0 + np.array([-L3 * 0.5, -L3 * 0.75, 0.02]),
             p0 + np.array([-0.04, 0, 0])])
    # ---- face
    top, bot = _fish_profile(B, 0.86)
    rz_e = (top - bot) / 2 * B["width"] / 0.5
    ep = np.array([0.36, bot + (top - bot) * (0.62 if name not in ("whale",) else 0.4), min(rz_e, (top - bot) / 2)
                   * 0.85])
    if name in ("shark", "dolphin", "whale"):
        ep[2] *= 0.95
    es = P["eye"] * (1.6 if style == "cute" else 1.0)
    for s in (1, -1):
        pe = ep * [1, 1, s]
        m.decal("eye", pe, n=np.array([0.35, 0.1, s * 1.0]), fwd=np.array([1.0, 0, 0]), size=es, shape="round",
                iris=coat.get("eye", "#d8c060" if name in ("fish", "goldfish") else "#2a2018"), pupil="round",
                pupil_r=0.6, look=(0.05, 0.0), rim_w=0.1, glint=0.32, tol=0.08, lid=0.2,
                sclera=None)
    m.anchors["eye"] = ep
    mouth = P.get("mouth", "fish")
    nose = np.array([0.5 + (B.get("beak") or 0), _fish_profile(B, 0.99)[0] * 0.3 + _fish_profile(B, 0.99)[1] * 0.7, 0])
    m.anchors["mouth"] = nose
    if mouth == "fish":
        hz = (top - bot) / 2 * B["width"] / 0.5
        m.decal("line", np.array([nose + np.array([-0.005, -0.004, hz * 0.3]), nose + np.array([-0.05, -0.012, hz * 0.55])]),
                width=0.008, color="#3a2018", opacity=0.6, tol=0.05)
    elif mouth == "under":
        m.decal("line", np.array([[0.4, bot + 0.025, 0.05], [0.33, bot + 0.012, 0.07], [0.26, bot + 0.02, 0.08]]),
                width=0.006, color="#2a2a30", opacity=0.6, tol=0.05)
    elif mouth == "smile":
        b0 = np.array([0.5 + B.get("beak", 0) - 0.01, nose[1] - 0.004, 0.025])
        m.decal("line", np.array([b0, b0 + np.array([-0.08, -0.004, 0.02]), b0 + np.array([-0.14, 0.012, 0.04])]),
                width=0.006, color="#2a3038", opacity=0.6, tol=0.05)
    elif mouth == "whale":
        m.decal("line", np.array([[0.49, nose[1], 0.02], [0.3, nose[1] - 0.03, 0.09], [0.16, nose[1] - 0.01, 0.12]]),
                width=0.006, color="#1a2230", opacity=0.7, tol=0.05)
    if F.get("gills"):
        for i in range(F["gills"]):
            x = 0.24 - i * 0.022
            t2, b2 = _fish_profile(B, 0.5 + x)
            hz = (t2 - b2) / 2 * B["width"] / 0.5 * 0.85
            m.decal("line", np.array([[x, b2 + (t2 - b2) * 0.35, hz], [x - 0.012, b2 + (t2 - b2) * 0.65, hz]]),
                    width=0.005, color="#3a4450", opacity=0.55, tol=0.05)
    elif mouth == "fish":
        t2, b2 = _fish_profile(B, 0.8)
        hz = (t2 - b2) / 2 * B["width"] / 0.5 * 0.9
        arc = [np.array([0.3 - 0.02 * math.sin(math.pi * u), b2 + (t2 - b2) * (0.2 + 0.6 * u), hz]) for u in
               np.linspace(0, 1, 6)]
        m.decal("line", np.array(arc), width=0.006, color="#000000", opacity=0.3, tol=0.06)
    if B.get("grooves"):
        for i in range(5):
            y = bot + 0.02 + i * 0.012
            m.decal("line", np.array([[0.42, y, 0.04 + i * 0.01], [0.12, y + 0.004, 0.09 + i * 0.01]]), width=0.004,
                    color="#8a96a2", opacity=0.5, tol=0.05)
    m.anchors["center"] = np.zeros(3)
    m.anchors["back"] = np.array([0.0, _fish_profile(B, 0.5)[0], 0])
    return m


def _build_jelly(P, pose, phase, coat, style, seed):
    m = _Model()
    m.coat = dict(coat)
    m.grounded = False
    base = coat.get("base", "#f2a6d0")
    pulse = math.sin(2 * math.pi * phase)
    rx = 0.5 * (1 - 0.06 * pulse)
    ry = 0.36 * (1 + 0.08 * pulse)
    m.ell("core", np.array([0, 0.05, 0]), (rx, ry, rx), base, name="bell", k=0.02, fur=0.0, pat=0.0, belly=0.0,
          spec=0.6, opa=0.82)
    m.ell("core", np.array([0, 0.08, 0]), (rx * 0.55, ry * 0.6, rx * 0.55), coat.get("belly", base), name="inner",
          k=0.0, fur=0.0, pat=0.0, belly=0.0, spec=0.2, opa=0.6)
    m.clip_y = None
    # frilly rim and oral arms
    rng = np.random.default_rng(seed + 3)
    for i in range(4):
        x0 = (i - 1.5) * 0.12
        pts = [np.array([x0, -0.05, 0.05 * (i % 2)])]
        for j in range(7):
            pts.append(pts[-1] + np.array([0.05 * math.sin(j * 1.3 + i + phase * 6), -0.12, 0.02]))
        m.chain(m.group("arm%d" % i, "core", 0.0), pts, [0.05 - 0.005 * j for j in range(8)], coat.get("fin", base),
                fur=0.0, pat=0.0, belly=0.0, spec=0.3, k=0.03, opa=0.75)
    for i in range(14):
        x0 = -rx * 0.92 + 2 * rx * 0.92 * i / 13
        zz = rx * 0.4 * math.sin(i * 2.4)
        pts = [np.array([x0, 0.0, zz])]
        ln = 0.9 + 0.5 * rng.random()
        for j in range(10):
            pts.append(pts[-1] + np.array([0.03 * math.sin(j * 0.9 + i * 1.7 + phase * 6.28), -ln / 10, 0]))
        m.decal("line", np.array(pts), width=0.008, color=coat.get("fin", base), opacity=0.55, clip=False, tol=1.0)
    m.anchors["center"] = np.zeros(3)
    m.anchors["top"] = np.array([0, 0.05 + ry, 0])
    return m


def _build_octopus(P, pose, phase, coat, style, seed):
    m = _Model()
    m.coat = dict(coat)
    m.coat["belly"] = coat.get("belly")
    m.coat["belly_dir"] = (0.0, -1.0, 0.0)
    m.coat["belly_edge"] = 0.4
    m.grounded = pose not in ("swim",)
    base = coat.get("base", "#d8603a")
    kw = dict(fur=0.0, pat=1.0, belly=1.0, spec=0.35)
    m.coat["pattern"] = coat.get("pattern", "spots")
    m.coat.setdefault("colours", [coat.get("spots", "#b84a2a")])
    m.coat.setdefault("spot", 0.22)
    m.coat.setdefault("scale", 1.6)
    m.ell("core", np.array([-0.12, 0.62, 0]), (0.3, 0.36, 0.3), base, rot=_rz(18), name="mantle", k=0.08, **kw)
    m.ell("core", np.array([0.05, 0.32, 0]), (0.24, 0.2, 0.24), base, name="head", k=0.08, **kw)
    for s in (1, -1):
        p = np.array([0.14, 0.4, s * 0.16])
        m.ell("core", p, (0.07, 0.07, 0.06), base, k=0.03, **kw)
        m.decal("eye", p + np.array([0.04, 0.0, s * 0.045]), n=np.array([0.5, 0.1, s * 1.0]), fwd=np.array([1.0, 0, 0]),
                size=0.05 * (1.5 if style == "cute" else 1.0), shape="round", iris="#f2d8a8", pupil="bar",
                pupil_r=0.6, look=(0.05, 0.0), rim_w=0.1, glint=0.3, tol=0.1, lid=0.3)
    m.anchors["eye"] = np.array([0.18, 0.4, 0.2])
    for i in range(8):
        a = math.radians(i * 45 + 22.5)
        dx, dz = math.cos(a), math.sin(a)
        pts = [np.array([0.05 + dx * 0.12, 0.22, dz * 0.12])]
        curl = 0.6 + 0.4 * math.sin(i * 1.9 + phase * 6.28)
        L = 0.95
        ang = 0.0
        for j in range(10):
            t = j / 9
            ang += curl * 0.32 * t * (1 if i % 2 else -1)
            out = 0.9 - 0.3 * t
            d = np.array([dx * math.cos(ang) * out, -0.35 + 0.5 * t * (0.3 + 0.5 * curl) if not m.grounded else
                          -0.25 + 0.9 * t * t * curl * 0.6, dz * math.cos(ang) * out]) + np.array(
                [-dz * math.sin(ang), 0, dx * math.sin(ang)]) * 0.6
            p = pts[-1] + _unit(d) * L / 10
            if m.grounded:
                p[1] = max(p[1], 0.03 + 0.25 * t * t * curl)
            pts.append(p)
        rad = [0.075 * (1 - 0.85 * (j / 10)) + 0.006 for j in range(11)]
        g = m.group("arm%d" % i, "core", 0.05)
        m.chain(g, pts, rad, base, k=0.03, **dict(kw, belly=0.6))
        if style != "flat":
            for j in range(2, 10, 2):
                q = pts[j] + np.array([0, -rad[j] * 0.7, 0])
                m.decal("spot", q, size=rad[j] * 0.35, color=coat.get("belly", "#f2a07a"), opacity=0.9, tol=0.04)
    m.anchors["center"] = np.array([0, 0.4, 0])
    m.anchors["top"] = np.array([-0.12, 0.98, 0])
    return m


def _build_turtle(P, pose, phase, coat, style, seed):
    m = _Model()
    m.coat = dict(coat)
    m.grounded = pose not in ("swim",)
    skin = coat.get("base", "#7a8a4a")
    shell = coat.get("shell", "#6a5a32")
    swim = pose == "swim"
    kws = dict(fur=0.0, pat=1.0, belly=0.0, spec=0.35)
    m.coat["pattern"] = "scutes"
    m.coat["colours"] = [coat.get("scute", "#9a7a42")]
    y0 = 0.0 if swim else 0.2
    m.ell(m.group("shell", "core", 0.01), np.array([0, y0 + 0.08, 0]), (0.5, 0.26, 0.42), shell, name="shell",
          k=0.0, **kws)
    m.ell(m.group("shell", "core", 0.01), np.array([0, y0 + 0.0, 0]), (0.5, 0.06, 0.43), coat.get("belly", "#e8dcb0"),
          name="plastron", k=0.03, fur=0.0, pat=0.0, belly=0.0, spec=0.2)
    kw = dict(fur=0.0, pat=0.0, belly=0.0, spec=0.2)
    m.coat.setdefault("spots", "#4a5a2a")
    flap = math.sin(2 * math.pi * phase)
    hc = np.array([0.6, y0 + 0.08 + (0.02 if swim else 0.06), 0])
    m.cone("core", np.array([0.35, y0 + 0.03, 0]), hc, 0.11, 0.1, skin, name="neck", k=0.04, **kw)
    m.ell("core", hc + np.array([0.04, 0.01, 0]), (0.13, 0.1, 0.1), skin, name="head", k=0.04, **kw)
    for s in (1, -1):
        p, n = _surf(hc + np.array([0.04, 0.01, 0]), (0.13, 0.1, 0.1), np.eye(3), (0.55, 0.35, 0.75 * s))
        m.decal("eye", p, n=n, fwd=np.array([1.0, 0, 0]), size=0.03 * (1.6 if style == "cute" else 1.0),
                shape="round", iris="#2a1a10", pupil="round", pupil_r=0.6, look=(0.1, 0.0), rim_w=0.12, glint=0.3,
                tol=0.08)
        if s == 1:
            m.anchors["eye"] = p
    m.decal("line", np.array([hc + np.array([0.17, -0.01, 0.03]), hc + np.array([0.08, -0.04, 0.08])]), width=0.006,
            color="#2a2a18", opacity=0.6, tol=0.05)
    m.anchors["mouth"] = hc + np.array([0.17, -0.02, 0])
    for s in (1, -1):
        for front in (True, False):
            g = m.group("limb%d%d" % (s, front), "core", 0.03)
            root = np.array([0.28 if front else -0.32, y0 + 0.02, s * 0.3])
            if swim:
                # paddles: flattened, swept-back volumes, the broad side turned partly to the viewer
                a = flap * (0.5 if front else 0.3)
                L = 0.55 if front else 0.26
                d = _unit(np.array([-0.55, 0.35 * a, s * 0.8]))
                up = np.array([0.0, 1.0, 0.0])
                wv = _unit(np.cross(d, up))
                tw = math.radians(50 if front else 35)
                nv = _unit(up * math.cos(tw) + wv * math.sin(tw))
                bv = _unit(np.cross(nv, d))
                R = np.stack([d, bv, nv], 1)
                m.ell(g, root + d * L * 0.48, (L * 0.55, (0.13 if front else 0.1), 0.03), skin, rot=R, k=0.03,
                      tone=1.0 if s == 1 else 0.85, **kw)
                if s == 1:
                    for t_ in (0.3, 0.55, 0.8):
                        m.decal("spot", root + d * L * t_ + nv * 0.03, size=0.02, color=coat.get("spots", "#4a5a2a"),
                                opacity=0.6, tol=0.05)
            else:
                foot = np.array([root[0] + (0.06 if front else -0.04), 0.0, s * 0.36])
                m.cone(g, root, foot + np.array([0, 0.05, 0]), 0.09, 0.075, skin, k=0.03,
                       tone=1.0 if s == 1 else 0.85, **kw)
                m.ell(g, foot + np.array([0.02, 0.04, 0]), (0.09, 0.045, 0.08), skin, k=0.02,
                      tone=1.0 if s == 1 else 0.85, **kw)
                m.shadow_pts.append(foot)
    m.cone("core", np.array([-0.45, y0 + 0.03, 0]), np.array([-0.58, y0 - 0.02, 0]), 0.05, 0.015, skin, k=0.03, **kw)
    m.anchors["center"] = np.array([0, y0 + 0.08, 0])
    m.anchors["back"] = np.array([0, y0 + 0.34, 0])
    if swim:
        _shift(m, np.array([0, -0.08, 0]))
    return m


def _build_sea(name, pose, phase, opt, coat, style, params, seed):
    P = _merge(_SPECIES_S[name], params or {})
    sp = P.get("special")
    if sp == "jelly":
        m = _build_jelly(P, pose, phase, coat, style, seed)
    elif sp == "octopus":
        m = _build_octopus(P, pose, phase, coat, style, seed)
    elif sp == "turtle":
        m = _build_turtle(P, pose, phase, coat, style, seed)
    else:
        m = _build_fish(name, P, pose, phase, opt, coat, style, seed)
    m.pitch = 6.0
    return m


# ===================================================================== small things: insects, snails, frogs, reptiles
#
# Units: body length = 1 (for the butterfly and the dragonfly: the wingspan). y up, ground at y = 0.

_SPECIES_X = {
    "butterfly": dict(measure="span", pose="fly", view_pitch=68.0,
                      coat=dict(body="#2a2220", pattern="monarch")),
    "bee": dict(measure="len", coat=dict(base="#f2b81e", stripes="#1e1a16", body="#1e1a16", wing="#e8f0f8",
                                         fuzz="#e8a82a")),
    "ladybird": dict(measure="len", coat=dict(base="#d8241e", spots="#141010", body="#141010")),
    "dragonfly": dict(measure="len", pose="fly", view_pitch=45.0,
                      coat=dict(base="#2a8ab8", body="#1e5a8a", eye="#2a9a6a", wing="#dceef6")),
    "snail": dict(measure="len", coat=dict(base="#c8a888", shell="#a8683a", band="#5a3418", belly="#d8c0a0")),
    "frog": dict(measure="h", coat=dict(base="#5aa83a", belly="#e8eac0", spots="#3a7a28", eye="#d8a82a")),
    "snake": dict(measure="len", coat=dict(base="#5a8a3a", belly="#d8d0a0", pattern="diamonds",
                                           colours=["#2a4a1e"])),
    "lizard": dict(measure="len", coat=dict(base="#6aa84a", belly="#d8d8a0", spots="#3a6a2a", pattern="spots",
                                            colours=["#3e6e2c"], spot=0.25, scale=1.6)),
    "spider": dict(measure="len", coat=dict(base="#2a2220", belly="#3a302a", marks="#c8a060")),
}
_SMALL_ALIASES = {"ladybug": "ladybug", "lady bug": "ladybird", "honeybee": "bee", "bumblebee": "bee",
                  "moth": "butterfly", "toad": "frog", "gecko": "lizard", "serpent": "snake", "slug": "snail",
                  "insect": "bee", "bug": "ladybird"}
_SMALL_ALIASES["ladybug"] = "ladybird"

_BUTTERFLY_COATS = {
    "monarch": dict(base="#f08a1e", border="#1a1414", veins="#1a1414", spots="#f6f2ea"),
    "morpho": dict(base="#2a7ae8", border="#141820", veins="#1a2a5a", spots="#f2f2f2"),
    "swallowtail": dict(base="#f2d84a", border="#1a1814", veins="#1a1814", spots="#f2d84a", eyespot="#d8402a"),
    "cabbage white": dict(base="#f4f2ea", border="#f4f2ea", veins="#c8c8bc", spots="#2a2a2a"),
}


def _wing_pts(c, u, v, ol, su, sv):
    return [c + u * (a * su) + v * (b * sv) for a, b in ol]


def _build_butterfly(P, pose, phase, opt, coat, style, seed, dragonfly=False):
    m = _Model()
    m.coat = dict(coat)
    m.grounded = False
    body = coat.get("body", "#2a2220")
    kwb = dict(fur=0.0, pat=0.0, belly=0.0, spec=0.3)
    # body along x
    if dragonfly:
        pts = [np.array([0.12 - 0.08 * i, 0.0, 0.0]) for i in range(9)]
        rad = [0.035, 0.04, 0.032, 0.022, 0.02, 0.019, 0.018, 0.017, 0.015]
        m.chain("core", pts, rad, body, name="abdomen", k=0.01, stops=[(0, body, 0), (1, coat.get("base", body), 0)],
                **kwb)
        for i in range(2, 9):
            m.decal("line", np.array([pts[i] + np.array([0, -rad[i], rad[i] * 0.8]), pts[i] + np.array([0, rad[i],
                                                                                                    rad[i] * 0.8])]),
                    width=0.004, color="#0a0a0a", opacity=0.5, tol=0.03)
        m.ell("core", np.array([0.19, 0.0, 0]), (0.065, 0.05, 0.05), body, k=0.02, **kwb)
        for s in (1, -1):
            m.ell("core", np.array([0.26, 0.01, s * 0.03]), (0.035, 0.04, 0.04), coat.get("eye", "#2a9a6a"), k=0.01,
                  fur=0.0, pat=0.0, belly=0.0, spec=0.8)
    else:
        m.chain("core", [np.array([0.12, 0, 0]), np.array([0.0, 0, 0]), np.array([-0.16, -0.01, 0])],
                [0.022, 0.02, 0.009], body, name="body", k=0.02, **dict(kwb, fur=0.6))
        m.ell("core", np.array([0.15, 0.0, 0]), (0.032, 0.03, 0.03), body, k=0.01, **kwb)
        for s in (1, -1):
            a0 = np.array([0.17, 0.02, s * 0.012])
            a1 = a0 + np.array([0.08, 0.06, s * 0.05])
            a2 = a1 + np.array([0.05, 0.02, s * 0.02])
            m.decal("line", np.array([a0, a1, a2]), width=0.005, color=body, opacity=0.9, clip=False, tol=1.0)
            m.decal("spot", a2, size=0.008, color=body, opacity=0.95, tol=1.0)
    # wings: flap about the body axis
    if pose in ("rest", "stand", "perch"):
        lift = 80.0
    elif pose == "open":
        lift = 8.0
    else:
        lift = 8.0 + 50.0 * (0.5 + 0.5 * math.cos(2 * math.pi * phase))
    if dragonfly:
        lift = 4.0 + 8.0 * math.cos(2 * math.pi * phase)
    pat = coat.get("pattern", "monarch")
    wc = dict(_BUTTERFLY_COATS.get(pat, {}))
    wc.update({k: v for k, v in coat.items() if k in ("base", "border", "veins", "spots", "eyespot")})
    for s in (1, -1):
        L = math.radians(lift)
        up = np.array([0.0, math.sin(L), s * math.cos(L)])
        fwd = np.array([1.0, 0, 0])
        tone = 1.0 if s == 1 else 0.9
        g = m.group("wings%d" % s, "core", 0.0)
        if dragonfly:
            for k_, (x0, sw, ln) in enumerate(((0.16, 8.0, 0.5), (0.08, -10.0, 0.46))):
                u = _unit(up + fwd * math.tan(math.radians(sw)))
                ol = [(0.0, 0.03), (0.3, 0.06), (0.8, 0.06), (1.0, 0.0), (0.85, -0.07), (0.3, -0.08), (0.0, -0.03)]
                c = np.array([x0, 0.01, 0])
                pts = _wing_pts(c, u, fwd, ol, ln, 1.0)
                m.flat(g, pts, coat.get("wing", "#dceef6"), bulge=0.05, tone=tone, fur=0.0, pat=0.0, belly=0.0,
                       spec=0.9, k=0.0, opa=0.35, smooth=2)
                for q in (0.0, 0.03, -0.04):
                    m.decal("line", np.array([c + u * 0.02 + fwd * q, c + u * ln * 0.95 + fwd * q * 0.5]),
                            width=0.003, color="#2a3a44", opacity=0.35, clip=False, tol=1.0)
                m.decal("spot", c + u * ln * 0.9 + fwd * 0.04, size=0.012, color="#2a2a2a", opacity=0.7,
                        squash=0.5, tol=1.0)
            continue
        # forewing and hindwing outlines in (span, chord) units
        fore = [(0.0, 0.04), (0.3, 0.2), (0.62, 0.36), (0.92, 0.48), (1.0, 0.4), (0.95, 0.15), (0.8, -0.02),
                (0.55, -0.1), (0.2, -0.08)]
        hind = [(0.0, -0.02), (0.3, -0.06), (0.6, -0.12), (0.75, -0.3), (0.62, -0.5), (0.38, -0.56),
                (0.15, -0.42), (0.0, -0.14)]
        if pat == "swallowtail":
            hind = hind[:5] + [(0.42, -0.62), (0.36, -0.5), (0.3, -0.48)] + hind[6:]
        c = np.array([0.03, 0.0, 0.0])
        for which, ol, sc_ in (("hind", hind, 0.42), ("fore", fore, 0.5)):
            pts = _wing_pts(c, up, fwd, ol, sc_, sc_)
            m.flat(g, pts, wc.get("border", "#1a1414"), bulge=0.06, tone=tone, fur=0.0, pat=0.0, belly=0.0,
                   spec=0.1, k=0.0, smooth=2, name=which)
            inner = [(a * 0.86 + 0.03, b * 0.86 + (0.01 if which == "fore" else -0.01)) for a, b in ol]
            pts2 = _wing_pts(c, up, fwd, inner, sc_, sc_)
            nrm = _unit(np.cross(up, fwd)) * (1 if up[1] >= 0 else -1)
            pts2 = [q + up * 0.0 + np.array([0, 0.003, 0.003 * s]) for q in pts2]
            m.flat(g, pts2, wc.get("base"), bulge=0.04, tone=tone * 1.45, fur=0.0, pat=0.0, belly=0.0, spec=0.1, k=0.0,
                   smooth=2, name=which + "_in")
            # veins
            if style != "flat" or True:
                for a_ in (-0.5, -0.15, 0.2, 0.55, 0.9):
                    tip = (0.9, 0.3 * a_ + (0.06 if which == "fore" else -0.3)) if which == "fore" else \
                        (0.55 + 0.1 * a_, -0.42 + 0.22 * a_)
                    p0 = c + up * 0.02 * sc_ + np.array([0, 0.004, 0.004 * s])
                    p1 = c + up * (tip[0] * sc_ * 0.85) + fwd * (tip[1] * sc_ * 0.85) + np.array([0, 0.004, 0.004 * s])
                    m.decal("line", np.array([p0, (p0 + p1) / 2 + fwd * 0.01, p1]), width=0.0035,
                            color=wc.get("veins", "#1a1414"), opacity=0.8, tol=1.0)
            # spots along the border
            sp_ = wc.get("spots")
            if sp_:
                for i in range(7 if which == "fore" else 5):
                    t = (i + 0.5) / (7 if which == "fore" else 5)
                    j = int(t * (len(ol) - 1))
                    a1, b1 = ol[j]
                    a2, b2 = ol[min(j + 1, len(ol) - 1)]
                    f = t * (len(ol) - 1) - j
                    pa, pb = a1 + (a2 - a1) * f, b1 + (b2 - b1) * f
                    q = c + up * (pa * sc_ * 0.94 + 0.01) + fwd * (pb * sc_ * 0.94) + np.array([0, 0.006,
                                                                                                0.006 * s])
                    m.decal("spot", q, size=0.0045, color=sp_, opacity=0.95, tol=1.0)
            if wc.get("eyespot") and which == "hind":
                q = c + up * (0.5 * sc_) + fwd * (-0.4 * sc_) + np.array([0, 0.006, 0.006 * s])
                m.decal("spot", q, size=0.025, color=wc["eyespot"], opacity=0.95, tol=1.0)
    m.anchors["center"] = np.zeros(3)
    m.pitch = P.get("view_pitch", 50.0)
    return m


def _build_bee(P, pose, phase, opt, coat, style, seed):
    m = _Model()
    m.coat = dict(coat)
    fly = pose in ("fly", "hover")
    m.grounded = not fly
    y0 = 0.0 if fly else 0.16
    base = coat.get("base", "#f2b81e")
    dark = coat.get("stripes", "#1e1a16")
    kw = dict(fur=1.0, pat=0.0, belly=0.0, spec=0.2)
    m.fur_len = 0.01
    # abdomen with stripes (colour stops along the chain)
    pts = [np.array([-0.05, y0, 0]), np.array([-0.22, y0 - 0.02, 0]), np.array([-0.4, y0 - 0.06, 0])]
    st = []
    for i in range(7):
        t = i / 6
        st.append((t, base if i % 2 == 0 else dark, 0.0))
        st.append((min(t + 0.08, 1.0), base if i % 2 == 0 else dark, 0.0))
    m.chain("core", pts, [0.15, 0.17, 0.06], base, name="abdomen", k=0.05, stops=st, **kw)
    m.ell("core", np.array([0.1, y0 + 0.02, 0]), (0.12, 0.12, 0.12), coat.get("fuzz", "#e8a82a"), name="thorax",
          k=0.02, **dict(kw, fur=1.6))
    hc = np.array([0.24, y0 + 0.0, 0])
    m.ell("core", hc, (0.08, 0.09, 0.09), dark, name="head", k=0.02, **dict(kw, fur=0.4))
    for s in (1, -1):
        p = hc + np.array([0.035, 0.02, s * 0.06])
        m.ell("core", p, (0.04, 0.06, 0.03), "#0e0c0a", k=0.005, fur=0.0, pat=0.0, belly=0.0, spec=0.9)
        a1 = hc + np.array([0.08, 0.12, s * 0.04])
        m.decal("line", np.array([hc + np.array([0.06, 0.05, s * 0.02]), a1, a1 + np.array([0.06, 0.0, s * 0.02])]),
                width=0.012, color=dark, opacity=1.0, clip=False, tol=1.0)
        if style == "cute" or True:
            m.decal("eye", p + np.array([0.02, 0.01, s * 0.02]), n=np.array([0.4, 0.2, s * 1.0]),
                    fwd=np.array([1.0, 0, 0]), size=0.03 * (1.5 if style == "cute" else 1.0), shape="round",
                    iris="#0e0c0a", pupil="round", pupil_r=0.6, rim_w=0.05, glint=0.35, tol=0.1, look=(0.1, 0))
    m.anchors["eye"] = hc + np.array([0.05, 0.02, 0.07])
    # legs
    for s in (1, -1):
        for i, x in enumerate((0.16, 0.1, 0.03)):
            r0 = np.array([x, y0 - 0.06, s * 0.05])
            kn = r0 + np.array([0.02 * (1 - i), -0.07, s * 0.06])
            ft = np.array([x + 0.05 * (1 - i) - 0.03 * i, 0.0 if not fly else y0 - 0.2, s * 0.13])
            m.chain(m.group("leg%d%d" % (s, i), "core", 0.0), [r0, kn, ft], [0.014, 0.012, 0.008], dark, k=0.0,
                    fur=0.0, pat=0.0, belly=0.0, tone=1.0 if s == 1 else 0.8)
    # wings
    beat = math.cos(2 * math.pi * phase * 3) if fly else 0.0
    for s in (1, -1):
        lift = math.radians(35 + 30 * beat)
        u = _unit(np.array([-0.35, math.sin(lift), s * math.cos(lift)]))
        fwd = _unit(np.array([1.0, 0, 0]) - u * u[0])
        c = np.array([0.08, y0 + 0.1, s * 0.03])
        for ln, sh in ((0.42, 0.0), (0.3, -0.06)):
            ol = [(0.0, 0.03), (0.4, 0.1), (0.85, 0.08), (1.0, 0.0), (0.8, -0.1), (0.3, -0.1), (0.0, -0.03)]
            pts = _wing_pts(c + fwd * sh, u, fwd, ol, ln, ln)
            m.flat(m.group("wing%d" % s, "core", 0.0), pts, coat.get("wing", "#e8f0f8"), bulge=0.05, fur=0.0,
                   pat=0.0, belly=0.0, spec=0.9, k=0.0, opa=0.42, smooth=2)
    m.anchors["center"] = np.array([0, y0, 0])
    if m.grounded:
        m.shadow_pts += [np.array([0.1, 0, 0.12])]
    return m


def _build_ladybird(P, pose, phase, opt, coat, style, seed):
    m = _Model()
    m.coat = dict(coat)
    red = coat.get("base", "#d8241e")
    blk = coat.get("body", "#141010")
    m.ell(m.group("shell", "core", 0.01), np.array([-0.05, 0.2, 0]), (0.42, 0.26, 0.36), red, name="shell", k=0.0,
          fur=0.0, pat=0.0, belly=0.0, spec=0.85)
    m.ell("core", np.array([-0.05, 0.08, 0]), (0.4, 0.08, 0.34), blk, k=0.02, fur=0.0, pat=0.0, belly=0.0, spec=0.3)
    hc = np.array([0.36, 0.12, 0])
    m.ell("core", hc, (0.12, 0.1, 0.14), blk, name="head", k=0.03, fur=0.0, pat=0.0, belly=0.0, spec=0.6)
    for s in (1, -1):
        m.decal("spot", hc + np.array([0.06, 0.05, s * 0.08]), size=0.035, color="#f4f0ea", opacity=0.95, tol=0.06)
        if style == "cute":
            m.decal("eye", hc + np.array([0.1, 0.03, s * 0.06]), n=np.array([0.6, 0.2, s * 0.8]),
                    fwd=np.array([1.0, 0, 0]), size=0.03, shape="round", iris="#0e0c0a", pupil_r=0.6, glint=0.35)
    # seam and spots
    m.decal("line", np.array([[0.3, 0.3, 0.0], [0.1, 0.44, 0.0], [-0.2, 0.42, 0.0], [-0.45, 0.24, 0.0]]),
            width=0.012, color=blk, opacity=0.8, tol=0.06)
    rng = np.random.default_rng(seed + 7)
    for (x, z) in ((0.12, 0.17), (-0.15, 0.2), (-0.32, 0.1), (0.0, 0.3), (-0.2, 0.0), (0.18, -0.17),
                   (-0.15, -0.2), (-0.32, -0.1), (0.0, -0.3)):
        y = 0.2 + 0.26 * math.sqrt(max(1 - ((x + 0.05) / 0.42) ** 2 - (z / 0.36) ** 2, 0.0))
        m.decal("spot", np.array([x, y, z]), size=0.055 + 0.01 * rng.random(), color=coat.get("spots", blk),
                opacity=1.0, tol=0.08)
    for s in (1, -1):
        for x in (0.2, 0.0, -0.2):
            r0 = np.array([x, 0.08, s * 0.25])
            m.chain(m.group("leg%d%.1f" % (s, x), "core", 0.0), [r0, r0 + np.array([0.04, -0.02, s * 0.12]),
                                                                   np.array([x + 0.08, 0.0, s * 0.42])],
                    [0.018, 0.015, 0.01], blk, k=0.0, fur=0.0, pat=0.0, belly=0.0, tone=1.0 if s == 1 else 0.8)
    m.anchors["center"] = np.array([0, 0.2, 0])
    m.anchors["back"] = np.array([-0.05, 0.46, 0])
    return m


def _build_snail(P, pose, phase, opt, coat, style, seed):
    m = _Model()
    m.coat = dict(coat)
    skin = coat.get("base", "#c8a888")
    kw = dict(fur=0.0, pat=0.0, belly=0.0, spec=0.6)
    # foot
    pts = [np.array([0.45, 0.1, 0]), np.array([0.3, 0.06, 0]), np.array([0.0, 0.05, 0]), np.array([-0.35, 0.04, 0]),
           np.array([-0.55, 0.02, 0])]
    m.chain("core", pts, [0.1, 0.09, 0.08, 0.06, 0.02], skin, name="foot", k=0.05, **kw)
    hc = np.array([0.42, 0.16, 0])
    m.ell("core", hc, (0.11, 0.1, 0.09), skin, name="head", k=0.05, **kw)
    for s in (1, -1):
        b0 = hc + np.array([0.02, 0.06, s * 0.04])
        tip = b0 + np.array([0.1, 0.24, s * 0.06])
        m.cone(m.group("stalk%d" % s, "core", 0.02), b0, tip, 0.025, 0.016, skin, k=0.02, **kw)
        m.ell(m.group("stalk%d" % s, "core", 0.02), tip, (0.03, 0.03, 0.03), skin, k=0.01, **kw)
        m.decal("spot", tip + np.array([0.02, 0.01, s * 0.02 + 0.015]), size=0.016, color="#1a1210", opacity=0.95,
                tol=0.1)
        m.cone(m.group("feel%d" % s, "core", 0.02), hc + np.array([0.08, -0.02, s * 0.05]),
               hc + np.array([0.16, 0.02, s * 0.08]), 0.016, 0.01, skin, k=0.01, **kw)
    m.anchors["eye"] = hc + np.array([0.12, 0.31, 0.1])
    # shell: a coiled tube (spiral of spheres), whorls getting smaller towards the apex
    shell = coat.get("shell", "#a8683a")
    band = coat.get("band", "#5a3418")
    g = m.group("shell", "core", 0.01)
    c0 = np.array([-0.08, 0.38, 0.0])
    n = 34
    for i in range(n):
        t = i / (n - 1)
        ang = t * 2.4 * 2 * math.pi
        rad = 0.3 * math.exp(-1.15 * t * 2.4 * math.pi / math.pi / 1.2)
        rr_ = 0.16 * math.exp(-0.95 * t * 2.0)
        p = c0 + np.array([math.cos(ang) * rad, -math.sin(ang) * rad, 0.06 * t])
        m.ell(g, p, (rr_, rr_, rr_ * 1.1), shell if (i // 3) % 2 else band if t > 0.15 else shell, k=0.04,
              fur=0.0, pat=0.0, belly=0.0, spec=0.5)
    m.anchors["center"] = c0
    m.anchors["back"] = c0 + np.array([0, 0.3, 0])
    m.shadow_pts.append(np.array([0.0, 0.0, 0.0]))
    return m


def _build_frog(P, pose, phase, opt, coat, style, seed):
    m = _Model()
    m.coat = dict(coat)
    m.coat["belly_dir"] = (0.3, -1.0, 0.0)
    m.coat["belly_edge"] = 0.1
    m.coat.setdefault("pattern", "spots")
    m.coat.setdefault("colours", [coat.get("spots", "#3a7a28")])
    m.coat.setdefault("spot", 0.2)
    m.coat.setdefault("scale", 1.4)
    g_ = coat.get("base", "#5aa83a")
    kw = dict(fur=0.0, pat=1.0, belly=1.0, spec=0.55)
    jump = pose == "jump"
    if jump:
        m.grounded = False
        Rb = _rz(20)
        m.ell("core", np.array([0, 0.45, 0]), (0.45, 0.2, 0.26), g_, rot=Rb, name="body", k=0.08, **kw)
        hc = np.array([0.4, 0.62, 0])
    else:
        Rb = _rz(30)
        m.ell("core", np.array([-0.05, 0.28, 0]), (0.38, 0.24, 0.3), g_, rot=Rb, name="body", k=0.08, **kw)
        hc = np.array([0.3, 0.42, 0])
    m.ell("core", hc, (0.25, 0.16, 0.26), g_, rot=_rz(-8), name="head", k=0.1, **kw)
    for s in (1, -1):
        ec = hc + np.array([0.04, 0.12, s * 0.15])
        m.ell("core", ec, (0.09, 0.09, 0.08), g_, k=0.05, **kw)
        p, n = _surf(ec, (0.09, 0.09, 0.08), np.eye(3), (0.55, 0.45, 0.7 * s))
        m.decal("eye", p, n=n, fwd=np.array([1.0, 0, 0]), size=0.065 * (1.4 if style == "cute" else 1.0),
                shape="round", iris=coat.get("eye", "#d8a82a"), pupil="bar", pupil_r=0.55, rim_w=0.1, glint=0.32,
                tol=0.1, look=(0.1, 0))
        if s == 1:
            m.anchors["eye"] = p
    mouth = [hc + np.array([0.25, -0.03, 0.0]), hc + np.array([0.18, -0.07, 0.15]), hc + np.array([0.02, -0.06,
                                                                                                    0.24])]
    m.decal("line", np.array(mouth), width=0.012, color="#1e3a14", opacity=0.6, tol=0.06)
    m.anchors["mouth"] = hc + np.array([0.25, -0.03, 0.0])
    for s in (1, -1):
        tone = 1.0 if s == 1 else 0.82
        g = m.group("hind%d" % s, "core", 0.06)
        if jump:
            hip = np.array([-0.35, 0.36, s * 0.18])
            kn = hip + np.array([-0.3, -0.1, s * 0.08])
            an = kn + np.array([-0.3, -0.05, 0])
            ft = an + np.array([-0.2, -0.02, 0])
        else:
            hip = np.array([-0.28, 0.18, s * 0.22])
            kn = np.array([0.08, 0.16, s * 0.34])
            an = np.array([-0.32, 0.05, s * 0.32])
            ft = np.array([0.0, 0.0, s * 0.4])
        m.chain(g, [hip, kn, an, ft], [0.13, 0.09, 0.06, 0.04], g_, k=0.04, tone=tone, **kw)
        for a in (-25, 0, 25):
            d = np.array([math.cos(math.radians(a)), 0.0, math.sin(math.radians(a)) * s])
            m.chain(g, [ft, ft + d * 0.14], [0.03, 0.02], g_, k=0.01, tone=tone, **dict(kw, pat=0.0))
        g2 = m.group("front%d" % s, "core", 0.04)
        sh = np.array([0.18, 0.25, s * 0.2])
        fp = np.array([0.32, 0.0, s * 0.24]) if not jump else np.array([0.8, 0.4, s * 0.2])
        m.chain(g2, [sh, (sh + fp) / 2 + np.array([0.02, 0, s * 0.04]), fp], [0.06, 0.045, 0.035], g_, k=0.03,
                tone=tone, **kw)
        for a in (-30, 0, 30):
            d = np.array([math.cos(math.radians(a)), 0.0, math.sin(math.radians(a)) * s])
            m.chain(g2, [fp, fp + d * 0.09], [0.025, 0.018], g_, k=0.01, tone=tone, **dict(kw, pat=0.0))
        if not jump:
            m.shadow_pts += [ft, fp]
    m.anchors["center"] = np.array([0, 0.3, 0])
    m.anchors["back"] = np.array([-0.1, 0.52, 0])
    return m


def _build_snake(P, pose, phase, opt, coat, style, seed, lizard=False):
    m = _Model()
    m.coat = dict(coat)
    m.coat["belly_dir"] = (0.0, -1.0, 0.0)
    m.coat["belly_edge"] = 0.35
    base = coat.get("base", "#5a8a3a")
    kw = dict(fur=0.0, pat=1.0, belly=1.0, spec=0.45)
    if coat.get("pattern") == "diamonds":
        m.coat["pattern"] = "spots"
        m.coat.setdefault("spot", 0.28)
        m.coat.setdefault("scale", 1.9)
    n = 26 if not lizard else 16
    pts = []
    rads = []
    L = 1.0
    for i in range(n):
        t = i / (n - 1)
        if lizard:
            x = 0.25 - t * 1.0
            z = 0.06 * math.sin(2 * math.pi * (t * 1.2 - phase))
            y = 0.06 if t < 0.45 else 0.06 * (1 - (t - 0.45) / 0.55) + 0.02
            r = 0.07 * (1 - 0.15 * t) if t < 0.4 else 0.07 * (1 - (t - 0.4) / 0.6) ** 1.1 + 0.006
        else:
            x = 0.42 - t * 0.85
            z = 0.22 * math.sin(2 * math.pi * (t * 1.6 - phase)) * (0.4 + 0.6 * t)
            y = 0.035
            r = 0.04 * (1 - 0.82 * t ** 2.2) * (0.9 + 0.2 * math.sin(math.pi * min(t * 2.5, 1)))
            if pose in ("rear", "strike", "coil"):
                if t < 0.25:
                    y = 0.035 + (0.25 - t) * 0.9
                    z = z * 0.3
        pts.append(np.array([x, y, z]))
        rads.append(max(r, 0.005))
    g = "core"
    m.chain(g, pts, rads, base, name="body", k=0.02, **kw)
    # head
    hd = _unit(pts[0] - pts[1])
    hc = pts[0] + hd * (0.04 if not lizard else 0.06)
    hr = 0.055 if not lizard else 0.07
    side = _unit(np.cross(np.array([0, 1.0, 0]), hd)) * -1
    R = np.stack([hd, np.array([0, 1.0, 0]), _unit(np.cross(hd, np.array([0, 1.0, 0])))], 1)
    m.ell("core", hc, (hr * 1.3, hr * 0.7, hr * 0.95), base, rot=R, name="head", k=0.03, **kw)
    m.ell("core", hc + hd * hr * 0.9 - np.array([0, hr * 0.1, 0]), (hr * 0.75, hr * 0.5, hr * 0.7), base, rot=R,
          k=0.03, **kw)
    for s in (1, -1):
        p, n_ = _surf(hc, (hr * 1.3, hr * 0.7, hr * 0.95), R, (0.45, 0.45, 0.75 * s))
        m.decal("eye", p, n=n_, fwd=hd, size=hr * 0.32 * (1.6 if style == "cute" else 1.0), shape="round",
                iris=coat.get("eye", "#d8b42a"), pupil="slit" if not lizard else "round", pupil_r=0.5, rim_w=0.1,
                glint=0.3, tol=0.06, look=(0.1, 0))
        if s == 1:
            m.anchors["eye"] = p
    tip = hc + hd * hr * 1.6
    m.anchors["mouth"] = tip
    if not lizard and pose != "still":
        tg = [tip, tip + hd * 0.06, tip + hd * 0.09 + np.array([0, 0.012, 0])]
        tg2 = [tip + hd * 0.06, tip + hd * 0.09 - np.array([0, 0.012, 0])]
        m.decal("line", np.array(tg), width=0.006, color="#c8283a", opacity=1.0, clip=False, tol=1.0)
        m.decal("line", np.array(tg2), width=0.006, color="#c8283a", opacity=1.0, clip=False, tol=1.0)
    if lizard:
        for i, (idx, front) in enumerate(((3, True), (8, False))):
            for s in (1, -1):
                r0 = pts[idx] + np.array([0, -0.01, s * rads[idx] * 0.6])
                sw = math.sin(2 * math.pi * phase + (0 if (front == (s == 1)) else math.pi)) * 0.08
                kn = r0 + np.array([sw, 0.04, s * 0.12])
                ft = np.array([kn[0] + 0.05 + sw, 0.0, s * 0.22])
                gl = m.group("leg%d%d" % (i, s), "core", 0.02)
                m.chain(gl, [r0, kn, ft], [0.035, 0.028, 0.02], base, k=0.02, tone=1.0 if s == 1 else 0.85,
                        **dict(kw, belly=0.0))
                for a in (-40, -10, 20, 50):
                    d = np.array([math.cos(math.radians(a)), 0, math.sin(math.radians(a)) * s])
                    m.chain(gl, [ft, ft + d * 0.06], [0.012, 0.008], base, k=0.005, tone=1.0 if s == 1 else 0.85,
                            **dict(kw, pat=0.0, belly=0.0))
                m.shadow_pts.append(ft)
    else:
        m.shadow_pts += [pts[i] * np.array([1, 0, 1]) for i in range(0, n, 5)]
    m.anchors["center"] = np.array([0, 0.04, 0])
    m.anchors["tail"] = pts[-1]
    return m


def _build_spider(P, pose, phase, opt, coat, style, seed):
    m = _Model()
    m.coat = dict(coat)
    base = coat.get("base", "#2a2220")
    kw = dict(fur=0.8, pat=0.0, belly=0.0, spec=0.35)
    m.fur_len = 0.006
    y0 = 0.22
    m.ell("core", np.array([-0.2, y0 + 0.06, 0]), (0.26, 0.2, 0.22), base, rot=_rz(15), name="abdomen", k=0.03, **kw)
    m.ell("core", np.array([0.12, y0, 0]), (0.15, 0.1, 0.13), base, name="cephalothorax", k=0.03, **kw)
    marks = coat.get("marks")
    if marks:
        for i, (x, y) in enumerate(((-0.12, 0.25), (-0.22, 0.24), (-0.32, 0.2))):
            m.decal("spot", np.array([x, y0 + y, 0.0]), size=0.03 - 0.005 * i, color=marks, opacity=0.9, tol=0.08)
    for s in (1, -1):
        for e_ in range(2):
            m.decal("spot", np.array([0.25, y0 + 0.06 + 0.02 * e_, s * (0.03 + 0.03 * e_)]), size=0.018,
                    color="#0a0808", opacity=1.0, tol=0.06)
            m.decal("spot", np.array([0.255, y0 + 0.065 + 0.02 * e_, s * (0.03 + 0.03 * e_) - 0.004]), size=0.005,
                    color="#ffffff", opacity=0.8, tol=0.06)
    m.anchors["eye"] = np.array([0.25, y0 + 0.07, 0.04])
    for s in (1, -1):
        for i in range(4):
            a = math.radians(-50 + i * 33)
            d = np.array([math.cos(a + math.pi / 2) * -1, 0.0, 0.0])
            ang = math.radians(60 - i * 40)
            dx, dz = math.sin(ang), math.cos(ang)
            r0 = np.array([0.12 + 0.04 * dx, y0 - 0.02, s * 0.08])
            wig = 0.03 * math.sin(2 * math.pi * phase + i * 1.7 + (s > 0) * math.pi)
            kn = r0 + np.array([dx * 0.22 + wig, 0.22, s * dz * 0.22])
            ft = np.array([r0[0] + dx * 0.55 + wig, 0.0, s * (0.08 + dz * 0.55)])
            mid = kn + (ft - kn) * 0.45 + np.array([0, 0.05, 0])
            g = m.group("leg%d%d" % (s, i), "core", 0.0)
            m.chain(g, [r0, kn, mid, ft], [0.03, 0.025, 0.02, 0.01], base, k=0.01,
                    tone=1.0 if s == 1 else 0.8, **dict(kw, fur=0.5))
            m.shadow_pts.append(ft)
    m.anchors["center"] = np.array([0, y0, 0])
    return m


def _build_small(name, pose, phase, opt, coat, style, params, seed):
    P = _merge(_SPECIES_X[name], params or {})
    if name == "butterfly":
        m = _build_butterfly(P, pose, phase, opt, coat, style, seed)
    elif name == "dragonfly":
        m = _build_butterfly(P, pose, phase, opt, coat, style, seed, dragonfly=True)
    elif name == "bee":
        m = _build_bee(P, pose, phase, opt, coat, style, seed)
    elif name == "ladybird":
        m = _build_ladybird(P, pose, phase, opt, coat, style, seed)
    elif name == "snail":
        m = _build_snail(P, pose, phase, opt, coat, style, seed)
    elif name == "frog":
        m = _build_frog(P, pose, phase, opt, coat, style, seed)
    elif name == "snake":
        m = _build_snake(P, pose, phase, opt, coat, style, seed)
    elif name == "lizard":
        m = _build_snake(P, pose, phase, opt, coat, style, seed, lizard=True)
    else:
        m = _build_spider(P, pose, phase, opt, coat, style, seed)
    if not hasattr(m, "pitch"):
        m.pitch = P.get("view_pitch", 22.0)
    return m


# ===================================================================== registry and group helpers

_DEFAULT_POSE = {"quad": "stand", "bird": "stand", "sea": "swim", "small": "stand"}
_REF_POSE = {"quad": "stand", "bird": "stand", "sea": "swim", "small": "stand"}
_OTHER = {}
_BUILDERS = {"bird": _build_bird, "sea": _build_sea}
for _n, _p in _SPECIES_S.items():
    _KINDS[_n] = "sea"
    _OTHER[_n] = _p
    _p.setdefault("measure", "h" if _p.get("special") in ("jelly", "octopus") else "len")
_ALIASES.update(_SEA_ALIASES)
_BUILDERS["small"] = _build_small
for _n, _p in _SPECIES_X.items():
    _KINDS[_n] = "small"
    _OTHER[_n] = _p
_ALIASES.update(_SMALL_ALIASES)
for _n, _p in _SPECIES_B.items():
    _KINDS[_n] = "bird"
    _OTHER[_n] = _p
_ALIASES.update(_BIRD_ALIASES)


def flock(cv, species="sparrow", n=7, region=(0.15, 0.1, 0.85, 0.4), size=0.04, seed=0, pose="fly", style="soft",
          facing="right", log=None, coat=None, **kw):
    """Draw a group of birds (flying by default) scattered over ``region`` (u0, v0, u1, v1), each with its own wing
    phase, small size jitter and depth order → list of ``Creature``. Pass ``pose="stand"`` / ``"swim"`` for birds on
    the ground or on water (positions are then ground / water points)."""
    rng = np.random.default_rng(seed)
    u0, v0, u1, v1 = region
    pts = []
    tries = 0
    gap = size * 1.6
    while len(pts) < n and tries < 4000:
        tries += 1
        p = (u0 + (u1 - u0) * rng.random(), v0 + (v1 - v0) * rng.random())
        if all(math.hypot((p[0] - q[0]) * cv.w / cv.h, p[1] - q[1]) > gap for q in pts):
            pts.append(p)
    pts.sort(key=lambda p: p[1])
    if log:
        log(f"flock of {species}")
    out = []
    for i, p in enumerate(pts):
        k = 0.8 + 0.4 * rng.random()
        c = Creature(species, size=size * k, style=style, facing=facing, seed=seed + i, coat=coat, **kw)
        c.pose(pose, phase=float(rng.random()))
        c.draw(cv, at=p, shadow=pose not in ("fly", "glide", "swim"))
        out.append(c)
    return out


def school(cv, species="fish", n=9, region=(0.2, 0.3, 0.8, 0.7), size=0.06, seed=0, facing="right", style="soft",
           log=None, coat=None, **kw):
    """Draw a school of fish (or any swimmer) heading the same way over ``region``, with tail-beat phases, slight
    size / height jitter and back-to-front order → list of ``Creature``."""
    rng = np.random.default_rng(seed)
    u0, v0, u1, v1 = region
    pts = []
    tries = 0
    gap = size * 1.1
    while len(pts) < n and tries < 4000:
        tries += 1
        p = (u0 + (u1 - u0) * rng.random(), v0 + (v1 - v0) * rng.random())
        if all(math.hypot((p[0] - q[0]) * cv.w / cv.h * 0.7, p[1] - q[1]) > gap for q in pts):
            pts.append(p)
    order = sorted(range(len(pts)), key=lambda i: -abs(pts[i][1] - (v0 + v1) / 2))
    if log:
        log(f"school of {species}")
    out = []
    for i in order:
        p = pts[i]
        k = 0.85 + 0.3 * rng.random()
        c = Creature(species, size=size * k, style=style, facing=facing, seed=seed + i, coat=coat,
                     view=float(rng.uniform(-14, 14)), **kw)
        c.pose("swim", phase=float(rng.random()))
        c.draw(cv, at=p, shadow=False)
        out.append(c)
    return out


SPECIES = sorted(_KINDS)


# ===================================================================== checked options, scene manifest

species = _strict(species, "species")
_strict_methods(Creature)
for _n in ("creature", "flock", "school"):
    globals()[_n] = _form.guard_fractions(globals()[_n], _n)
for _n in ("draw", "fit", "render"):
    setattr(Creature, _n, _form.guard_fractions(getattr(Creature, _n), "Creature." + _n))
del _n

from . import manifest as _manifest  # noqa: E402


def _manifest_animal(cap, args, res):
    """One manifest entry per animal: the species is the name, its kind and "animal" are aliases."""
    c = args["self"]
    if c.bbox is None or c.mask is None:
        return []
    asked = getattr(c, "requested", c.name)
    alias = [a for a in (c.name, c.kind, "animal") if a != asked]
    return [dict(name=asked, mask=c.mask, alias=alias, meta={"pose": c._pose[0]})]


_manifest.wrap_method(Creature, "draw", "creature", after=_manifest_animal)


# ---- did-you-mean for names that do not exist (pixelcanvas._args.module_getattr)
from ._args import module_getattr as _module_getattr  # noqa: E402

_HINTS = {
    "animal": "use creature.creature(cv, 'dog', at=(0.5, 0.8), size=0.3) or creature.Creature('dog'); creature.species() lists them",
    "draw": "use creature.creature(cv, 'dog', at=(0.5, 0.8), size=0.3) or Creature('dog').draw(cv, at=...)",
    "dog": "use creature.creature(cv, 'dog', at=(0.5, 0.8), size=0.3); creature.species() lists every species",
    "cat": "use creature.creature(cv, 'cat', at=(0.5, 0.8), size=0.3); creature.species() lists every species",
    "animals": "use creature.species() to list the species; creature.flock / creature.school draw groups",
    "birds": "use creature.flock(cv, 'sparrow', n=7, region=(0.15, 0.1, 0.85, 0.4))",
    "fish": "use creature.school(cv, 'fish', n=9, region=(0.2, 0.3, 0.8, 0.7))",
    "herd": "use creature.flock(cv, 'sheep', n=7, region=(0.15, 0.1, 0.85, 0.4))",
    "group": "use creature.flock(cv, species, n, region) or creature.school(cv, species, n, region)",
    "pose": "use Creature('dog').pose('sit') or creature.creature(cv, 'dog', pose='sit')",
    "list": "use creature.species() to list the species",
    "names": "use creature.species() to list the species",
}
__getattr__ = _module_getattr(__name__, _HINTS)
