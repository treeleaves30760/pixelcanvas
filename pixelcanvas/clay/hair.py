# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from .. import solid as _S
from ..solid import (Material, capsule, cone, cylinder, ellipsoid, intersect, plane, rounded_box, sphere, torus)
from .shapes import _Rx, _Rz, _axis_rot, _chain, _flat_ell, _xf, round_cone, smooth_union


# ===================================================================================================== hair

def _fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], -1)


def _cap(head, t, off=-0.04, tilt=30.0):
    """Hair cap hugging the skull: the head inflated by ``t``, cut by a hairline plane (front high, nape low)."""
    a = math.radians(tilt)
    return intersect(head.shape_(t), plane((0, -math.cos(a), math.sin(a)), off * head.R), k=0.06 * head.R)


def _hair(style, head, detail="normal", side=1):
    """Sculpted hair volumes in head-local coordinates. Returns (hair shapes, tie / accent shapes)."""
    R = head.R
    nb = {"low": 0.5, "normal": 1.0, "high": 1.7}.get(detail, 1.0)
    out, acc = [], []
    s = side
    if style in (None, "bald", "none"):
        return out, acc
    if style == "balding":
        ring = intersect(_cap(head, 0.06 * R, 0.1, 40), plane((0, 1, 0.35), 0.12 * R))
        return [ring], acc
    if style in ("crop", "buzz"):
        return [_cap(head, (0.045 if style == "crop" else 0.028) * R, 0.0 if style == "crop" else 0.02)], acc
    if style == "short":
        cap = _cap(head, 0.075 * R, -0.08)
        top = _flat_ell(0.66 * R, 0.3 * R, 0.74 * R, None, (0, 0.68 * R, -0.02 * R))
        fr = _flat_ell(0.62 * R, 0.2 * R, 0.3 * R, _Rz(-14 * s) @ _Rx(-24), (0.12 * R * s, 0.62 * R, 0.66 * R))
        tuft = _flat_ell(0.3 * R, 0.16 * R, 0.22 * R, _Rz(-30 * s) @ _Rx(-30), (-0.38 * R * s, 0.6 * R, 0.66 * R))
        return [smooth_union(cap, top, fr, tuft, k=0.1 * R)], acc
    if style == "side_part":
        cap = _cap(head, 0.07 * R, -0.07)
        sw = _flat_ell(0.82 * R, 0.26 * R, 0.62 * R, _Rz(12 * s) @ _Rx(-12), (0.1 * R * s, 0.74 * R, 0.3 * R))
        sw2 = _flat_ell(0.4 * R, 0.22 * R, 0.36 * R, _Rz(-25 * s), (-0.62 * R * s, 0.58 * R, 0.32 * R))
        hair = smooth_union(cap, sw, sw2, k=0.1 * R)
        groove = _xf(rounded_box(0.035 * R, 0.6 * R, 1.3 * R, 0.015 * R), _Rz(-15 * s), (-0.38 * R * s, 1.0 * R, 0))
        return [hair.carve(groove, 0.02 * R)], acc
    if style == "quiff":
        cap = _cap(head, 0.05 * R, -0.02)
        q = round_cone((0, 0.72 * R, 0.15 * R), (0.04 * R, 1.06 * R, 0.66 * R), 0.4 * R, 0.27 * R)
        return [smooth_union(cap, q, k=0.16 * R)], acc
    if style == "spiky":
        cap = _cap(head, 0.06 * R, -0.04)
        dirs = [(0, 1, 0.3), (0.5, 0.9, 0.2), (-0.5, 0.9, 0.2), (0.3, 0.85, 0.6), (-0.3, 0.85, 0.6), (0, 0.8, -0.5),
                (0.6, 0.6, -0.4), (-0.6, 0.6, -0.4), (0, 0.7, 0.75)]
        sp = []
        for d in dirs[:max(5, int(len(dirs) * min(nb, 1.0)))]:
            d = np.asarray(d, float) / np.linalg.norm(d)
            sp.append(round_cone(d * 0.75 * R, d * 1.32 * R, 0.24 * R, 0.03 * R))
        return [smooth_union(cap, *sp, k=0.06 * R)], acc
    if style == "pixie":
        cap = _cap(head, 0.07 * R, -0.06)
        fr = _flat_ell(0.55 * R, 0.2 * R, 0.3 * R, _Rz(22 * s) @ _Rx(-15), (-0.2 * R * s, 0.6 * R, 0.62 * R))
        return [smooth_union(cap, fr, k=0.1 * R)], acc
    if style == "hijab":
        shell = head.shape_(0.16 * R)
        drape = round_cone((0, -0.45 * R, -0.12 * R), (0, -1.4 * R, -0.05 * R), 0.92 * R, 1.25 * R)
        body = intersect(smooth_union(shell, drape, k=0.3 * R), plane((0, -1, 0), 1.55 * R))
        face = _flat_ell(0.7 * R, 0.84 * R, 0.9 * R, None, (0, -0.14 * R, 0.86 * R))
        return [body.carve(face, 0.06 * R)], acc
    if style in ("bob", "long", "wavy"):
        long_ = style != "bob"
        cap = _cap(head, 0.12 * R, -0.3, 22)
        vol = _flat_ell(1.1 * R, 0.92 * R, 1.06 * R, None, (0, 0.04 * R, -0.1 * R))
        parts = [cap, vol]
        if long_:
            parts.append(_flat_ell(0.96 * R, 1.3 * R, 0.5 * R, None, (0, -0.95 * R, -0.45 * R)))
            for sx in (1, -1):
                parts.append(round_cone((sx * 0.92 * R, -0.1 * R, -0.12 * R), (sx * 0.86 * R, -1.62 * R, -0.12 * R),
                                        0.26 * R, 0.17 * R))
        body = smooth_union(*parts, k=0.16 * R)
        body = intersect(body, plane((0, -1, 0), (0.78 if not long_ else 1.95) * R), k=0.08 * R)
        window = _xf(rounded_box(1.62 * R, 2.4 * R, 1.6 * R, 0.4 * R), np.eye(3), (0, -0.82 * R, 1.06 * R))
        body = body.carve(window, 0.08 * R)
        bangs = _flat_ell(0.82 * R, 0.24 * R, 0.34 * R, _Rz(-6 * s) @ _Rx(-20), (0.04 * R * s, 0.56 * R, 0.64 * R))
        hair = smooth_union(body, bangs, k=0.12 * R)
        if style == "wavy":
            bumps = [sphere(0.22 * R).at(sx * 0.95 * R, y * R, z * R) for sx in (1, -1)
                     for (y, z) in ((-0.55, -0.15), (-1.15, -0.15), (-0.35, -0.65), (-0.95, -0.6))]
            hair = smooth_union(hair, _S.union(*bumps), k=0.1 * R)
        return [hair], acc
    if style == "ponytail":
        cap = _cap(head, 0.065 * R, -0.06)
        fr = _flat_ell(0.6 * R, 0.18 * R, 0.3 * R, _Rx(-15), (0.1 * R, 0.62 * R, 0.6 * R))
        tail = _chain([(0, 0.3 * R, -0.95 * R), (0, 0.1 * R, -1.32 * R), (0, -0.5 * R, -1.42 * R),
                       (0, -1.1 * R, -1.25 * R)], [0.2 * R, 0.24 * R, 0.2 * R, 0.08 * R])
        out = [smooth_union(cap, fr, k=0.1 * R), _S.union(*tail)]
        acc.append(_xf(torus(0.16 * R, 0.06 * R), _Rx(70), (0, 0.27 * R, -1.0 * R)))
        return out, acc
    if style == "pigtails":
        cap = _cap(head, 0.065 * R, -0.06)
        fr = _flat_ell(0.7 * R, 0.18 * R, 0.3 * R, _Rx(-15), (0, 0.6 * R, 0.6 * R))
        out = [smooth_union(cap, fr, k=0.1 * R)]
        for sx in (1, -1):
            out.append(_S.union(*_chain([(sx * 0.95 * R, 0.25 * R, -0.3 * R), (sx * 1.3 * R, 0.0, -0.35 * R),
                                         (sx * 1.4 * R, -0.55 * R, -0.25 * R)], [0.2 * R, 0.24 * R, 0.1 * R])))
            acc.append(_xf(torus(0.15 * R, 0.055 * R), _Rz(sx * 60), (sx * 1.05 * R, 0.2 * R, -0.3 * R)))
        return out, acc
    if style in ("bun", "top_knot"):
        cap = _cap(head, 0.06 * R, -0.06)
        fr = _flat_ell(0.62 * R, 0.16 * R, 0.28 * R, _Rx(-15), (0, 0.62 * R, 0.58 * R))
        if style == "bun":
            b = sphere(0.38 * R).at(0, 0.82 * R, -0.5 * R)
            acc.append(_xf(torus(0.3 * R, 0.05 * R), _Rx(-45), (0, 0.66 * R, -0.36 * R)))
        else:
            b = sphere(0.27 * R).at(0, 1.08 * R, -0.12 * R)
            acc.append(torus(0.19 * R, 0.045 * R).at(0, 0.9 * R, -0.1 * R))
        return [smooth_union(cap, fr, b, k=0.08 * R)], acc
    if style in ("curly", "afro"):
        if style == "curly":
            base = _cap(head, 0.1 * R, -0.08)
            n, rr = int(22 * nb), 0.21 * R
        else:
            base = intersect(_flat_ell(1.42 * R, 1.25 * R, 1.38 * R, None, (0, 0.32 * R, -0.12 * R)),
                             plane((0, -1, 0.25), 0.55 * R))
            # a halo behind the face: cut by a tilted plane (high at the crown, back at the cheeks), never a hood rim
            face_cut = plane((0, -0.5, 0.866), 0.08 * R)
            base = intersect(base, face_cut, k=0.18 * R)
            n, rr = int(26 * nb), 0.3 * R
        D = _fib_dirs(int(n * 2.4))
        keep = (-0.866 * D[:, 1] + 0.5 * D[:, 2] < -0.05) & (D[:, 2] < 0.75)
        D = D[keep][:n]
        if style == "curly":
            P, _ = head.ray(D)
            P = P + D * 0.08 * R
        else:
            P = np.array([0, 0.32 * R, -0.12 * R]) + D * np.array([1.38, 1.2, 1.32]) * R
        balls = [sphere(rr * (0.85 + 0.3 * ((i * 0.618) % 1))).at(*p) for i, p in enumerate(P)]
        if style == "afro":
            balls = [b for b, p in zip(balls, P) if -0.5 * p[1] + 0.866 * p[2] < 0.25 * R]
            return [intersect(smooth_union(base, _S.union(*balls), k=0.07 * R), face_cut, k=0.12 * R)], acc
        return [smooth_union(base, _S.union(*balls), k=0.07 * R)], acc
    if style == "braids":
        cap = _cap(head, 0.06 * R, -0.08)
        out = [cap.carve(rounded_box(0.03 * R, 0.5 * R, 1.6 * R, 0.01 * R).at(0, 1.0 * R, 0), 0.01 * R)]
        for sx in (1, -1):
            beads = []
            n = 7
            for i in range(n):
                t = i / (n - 1)
                p = (sx * (0.8 * R - 0.05 * R * t) + sx * 0.04 * R * (1 if i % 2 else -1),
                     -0.15 * R - 1.35 * R * t, -0.4 * R + 0.45 * R * t)
                beads.append(sphere(0.15 * R * (1 - 0.25 * t)).at(*p))
            out.append(smooth_union(*beads, k=0.04 * R))
            acc.append(sphere(0.07 * R).at(sx * 0.76 * R, -1.6 * R, 0.07 * R))
        return out, acc
    if style == "mohawk":
        out = [_cap(head, 0.02 * R, 0.02)]
        D = np.array([[0, 0.55, 0.85], [0, 0.85, 0.5], [0, 1.0, 0.0], [0, 0.85, -0.5], [0, 0.5, -0.85]])
        P, Nn = head.ray(D)
        fins = [_xf(ellipsoid(0.09 * R, 0.34 * R, 0.24 * R), _axis_rot((0, 1, 0), n_), p + n_ * 0.12 * R)
                for p, n_ in zip(P, Nn)]
        out.append(smooth_union(*fins, k=0.1 * R))
        return out, acc
    if style == "baby_tuft":
        return [smooth_union(round_cone((0, 0.85 * R, 0.1 * R), (0.05 * R, 1.12 * R, 0.25 * R), 0.09 * R, 0.04 * R),
                             round_cone((0.05 * R, 1.12 * R, 0.25 * R), (0.16 * R, 1.08 * R, 0.33 * R), 0.04 * R,
                                        0.02 * R), k=0.03 * R)], acc
    return _hair("short", head, detail, side)


# ===================================================================================================== hats

# hair under a hat is trimmed at the hat's rim: (rim height in R, tilt); None removes the hair (closed helmets)
_HAT_CLIP = {"cap": (0.3, -14), "beanie": (0.26, -6), "sun_hat": (0.34, -6), "straw_hat": (0.34, -6),
             "cowboy_hat": (0.34, -6), "chef_hat": (0.5, 0), "hard_hat": (0.27, -8), "fire_helmet": (0.27, -8),
             "helmet": (0.27, -8), "witch_hat": (0.4, -6), "wizard_hat": (0.4, -6), "santa_hat": (0.4, -6),
             "top_hat": (0.72, 0), "pirate_hat": (0.4, -6), "police_cap": (0.6, -10), "knight_helmet": (None, 0),
             "nurse_cap": (0.8, 0)}

def _hat(kind, head, color, accent):
    """Hats in head-local coordinates: list of (part name, shape, material spec)."""
    R = head.R
    top = head.cc + head.ca[1]
    parts = []

    def dome(t, cut_y, tilt=-12):
        return intersect(head.shape_(t), plane(_Rx(tilt) @ np.array([0, -1.0, 0]), -cut_y))
    if kind == "cap":
        parts.append(("hat", dome(0.1 * R, 0.28 * R, -14), color))
        parts.append(("hat", _xf(ellipsoid(0.62 * R, 0.05 * R, 0.52 * R), _Rx(-8), (0, 0.4 * R, 0.88 * R)), color))
        parts.append(("hat accent", sphere(0.07 * R).at(0, top + 0.1 * R, 0), accent))
    elif kind == "beanie":
        parts.append(("hat", dome(0.13 * R, 0.24 * R, -6), color))
        parts.append(("hat accent", _xf(torus(0.98 * R, 0.13 * R), _Rx(-6), (0, 0.3 * R, -0.02 * R)), accent))
        parts.append(("hat accent", sphere(0.2 * R).at(0, top + 0.22 * R, -0.05 * R), accent))
    elif kind in ("sun_hat", "straw_hat", "cowboy_hat"):
        mat = ("wood", color) if kind == "straw_hat" else color
        parts.append(("hat", dome(0.12 * R, 0.32 * R, -6), mat))
        rad = 1.6 * R if kind != "cowboy_hat" else 1.45 * R
        parts.append(("hat", _xf(cylinder(rad, 0.05 * R, round=0.024 * R), _Rx(-6), (0, 0.32 * R, 0)), mat))
        parts.append(("hat accent", _xf(cylinder(1.05 * R, 0.14 * R), _Rx(-6), (0, 0.41 * R, -0.02 * R)), accent))
    elif kind == "chef_hat":
        band = cylinder(0.92 * R, 0.4 * R, round=0.05 * R).at(0, 0.62 * R, -0.05 * R)
        puff = smooth_union(*[sphere(0.5 * R).at(x * R, (1.2 + 0.08 * abs(x)) * R, z * R)
                              for x, z in ((0, 0), (0.45, 0.1), (-0.45, 0.1), (0.2, -0.4), (-0.25, -0.38))], k=0.2 * R)
        parts.append(("hat", smooth_union(band, puff, k=0.1 * R), color))
    elif kind in ("hard_hat", "fire_helmet", "helmet"):
        parts.append(("hat", dome(0.16 * R, 0.25 * R, -8), color))
        if kind == "hard_hat":
            parts.append(("hat", _xf(cylinder(1.22 * R, 0.05 * R, round=0.02 * R), _Rx(-8), (0, 0.3 * R, 0.08 * R)),
                          color))
            parts.append(("hat accent", _xf(rounded_box(0.12 * R, 0.4 * R, 2.0 * R, 0.05 * R), _Rx(-10),
                                            (0, top + 0.08 * R, 0)), color))
        elif kind == "fire_helmet":
            parts.append(("hat", _xf(ellipsoid(1.3 * R, 0.06 * R, 1.45 * R), _Rx(-14), (0, 0.32 * R, -0.25 * R)),
                          color))
            parts.append(("hat accent", _xf(rounded_box(0.5 * R, 0.4 * R, 0.08 * R, 0.04 * R), _Rx(-30),
                                            (0, 0.82 * R, 0.86 * R)), accent))
        else:
            for x in (-0.35, 0, 0.35):
                parts.append(("hat accent", rounded_box(0.12 * R, 0.08 * R, 1.2 * R, 0.04 * R).at(
                    x * R, top + 0.14 * R, -0.05 * R), accent))
    elif kind == "space_helmet":
        parts.append(("visor", sphere(1.35 * R).at(0, 0.0, 0.05 * R),
                      Material(color="#d6ecf8", roughness=0.03, specular=1.0, glass=0.85, ior=1.15, shadow=0.2,
                               reflect=0.08)))
        parts.append(("hat", torus(1.05 * R, 0.16 * R).at(0, -1.08 * R, 0.02 * R), color))
    elif kind == "knight_helmet":
        sh = head.shape_(0.16 * R)
        slit = rounded_box(1.2 * R, 0.09 * R, 0.6 * R, 0.03 * R).at(0, 0.0, 0.95 * R)
        parts.append(("hat", _S.subtract(sh, slit), ("metal", color)))
        parts.append(("hat accent", round_cone((0, top + 0.12 * R, -0.2 * R), (0, top + 0.6 * R, -0.75 * R),
                                               0.12 * R, 0.05 * R), accent))
    elif kind in ("crown", "tiara"):
        h = 0.36 * R if kind == "crown" else 0.2 * R
        rr = 0.75 * R if kind == "crown" else 0.86 * R
        band = cylinder(rr, h * 0.55, round=0.02 * R).shell(0.06 * R)
        spikes = [cone(0.13 * R, 0.0, h).at(rr * math.sin(a), h * 0.5, rr * math.cos(a))
                  for a in np.radians(np.arange(0, 360, 45 if kind == "crown" else 60))
                  if kind == "crown" or math.cos(a) > -0.2]
        y0 = top - 0.12 * R if kind == "crown" else top - 0.28 * R
        parts.append(("hat", _xf(_S.union(band, *spikes), _Rx(-8), (0, y0, -0.02 * R)), "gold"))
        parts.append(("hat accent", sphere(0.07 * R).at(0, y0 + 0.02 * R, rr + 0.04 * R), accent))
    elif kind in ("witch_hat", "wizard_hat", "party_hat", "santa_hat"):
        hgt = {"witch_hat": 1.9, "wizard_hat": 1.8, "party_hat": 1.1, "santa_hat": 1.2}[kind] * R
        rb = {"witch_hat": 0.92, "wizard_hat": 0.95, "party_hat": 0.45, "santa_hat": 1.0}[kind] * R
        y0 = 0.38 * R if kind != "party_hat" else top - 0.05 * R
        tip_z = -0.4 * R if kind == "witch_hat" else -0.2 * R
        pts = [(0, y0, 0), (0, y0 + hgt * 0.5, -0.12 * R), (0.0, y0 + hgt * 0.85, tip_z)]
        if kind == "santa_hat":
            pts[2] = (0.35 * R, y0 + hgt * 0.55, -0.7 * R)
        body = smooth_union(round_cone(pts[0], pts[1], rb, rb * 0.45), round_cone(pts[1], pts[2], rb * 0.45, 0.04 * R),
                            k=0.08 * R)
        parts.append(("hat", body, color))
        if kind in ("witch_hat", "wizard_hat"):
            parts.append(("hat", _xf(cylinder(1.85 * R, 0.05 * R, round=0.02 * R), _Rx(-6), (0, 0.4 * R, 0)), color))
            parts.append(("hat accent", _xf(cylinder(0.94 * R, 0.16 * R), _Rx(-6), (0, 0.5 * R, 0)), accent))
        elif kind == "santa_hat":
            parts.append(("hat accent", _xf(torus(1.0 * R, 0.17 * R), _Rx(-6), (0, 0.38 * R, 0)), "#f6f2ea"))
            parts.append(("hat accent", sphere(0.2 * R).at(*pts[2]), "#f6f2ea"))
        else:
            parts.append(("hat accent", sphere(0.12 * R).at(0, y0 + hgt * 0.98, -0.2 * R), accent))
    elif kind == "top_hat":
        parts.append(("hat", cylinder(0.78 * R, 1.0 * R, round=0.03 * R).at(0, top + 0.38 * R, -0.04 * R), color))
        parts.append(("hat", cylinder(1.2 * R, 0.05 * R, round=0.02 * R).at(0, top - 0.08 * R, -0.04 * R), color))
        parts.append(("hat accent", cylinder(0.8 * R, 0.18 * R).at(0, top + 0.02 * R, -0.04 * R), accent))
    elif kind == "beret":
        parts.append(("hat", _xf(ellipsoid(1.05 * R, 0.24 * R, 1.0 * R), _Rz(-14), (-0.1 * R, top - 0.04 * R, 0)),
                      color))
    elif kind == "pirate_hat":
        body = ellipsoid(1.4 * R, 0.38 * R, 0.9 * R).at(0, top - 0.02 * R, -0.05 * R)
        body = body.carve(ellipsoid(1.5 * R, 0.4 * R, 1.2 * R).at(0, top + 0.42 * R, 0.2 * R), 0.08 * R)
        parts.append(("hat", smooth_union(body, dome(0.12 * R, 0.38 * R, -6), k=0.1 * R), color))
        parts.append(("hat accent", _xf(cylinder(0.14 * R, 0.03 * R), _Rx(80), (0, top + 0.15 * R, 0.75 * R)), accent))
    elif kind == "nurse_cap":
        parts.append(("hat", _xf(rounded_box(1.0 * R, 0.4 * R, 0.5 * R, 0.08 * R), _Rx(-25),
                                 (0, top - 0.02 * R, 0.1 * R)), color))
        for w, h in ((0.18, 0.05), (0.05, 0.18)):
            parts.append(("hat accent", _xf(rounded_box(w * R, h * R, 0.06 * R, 0.01 * R), _Rx(-25),
                                            (0, top + 0.02 * R, 0.36 * R)), accent))
    elif kind == "police_cap":
        parts.append(("hat", _xf(cylinder(1.08 * R, 0.4 * R, round=0.06 * R), _Rx(-10),
                                 (0, top - 0.02 * R, -0.05 * R)), color))
        parts.append(("hat accent", _xf(ellipsoid(0.7 * R, 0.05 * R, 0.4 * R), _Rx(-14), (0, 0.42 * R, 0.86 * R)),
                      "#1e1e22"))
        parts.append(("hat accent", _xf(cylinder(0.12 * R, 0.04 * R), _Rx(80), (0, top + 0.02 * R, 1.0 * R)), "gold"))
    elif kind == "bow":
        for sx in (1, -1):
            parts.append(("hat", _xf(ellipsoid(0.32 * R, 0.2 * R, 0.1 * R), _Rz(sx * 20),
                                     (sx * 0.3 * R + 0.45 * R, top - 0.05 * R, 0.25 * R)), color))
        parts.append(("hat", sphere(0.1 * R).at(0.45 * R, top - 0.05 * R, 0.28 * R), color))
    elif kind == "headband":
        parts.append(("hat", _xf(torus(0.97 * R, 0.06 * R), _Rx(-70), (0, 0.55 * R, -0.12 * R)), color))
    elif kind == "flower_crown":
        for i, a in enumerate(np.radians(np.arange(0, 360, 30))):
            p = (1.0 * R * math.sin(a), 0.52 * R + 0.1 * R * math.cos(a), 0.98 * R * math.cos(a) - 0.05 * R)
            parts.append(("hat" if i % 2 == 0 else "hat accent", sphere(0.13 * R).at(*p), color if i % 2 == 0 else accent))
    elif kind == "antenna":
        for sx in (1, -1):
            parts.append(("hat", round_cone((sx * 0.3 * R, top - 0.1 * R, 0), (sx * 0.5 * R, top + 0.55 * R, 0.05 * R),
                                            0.04 * R, 0.03 * R), color))
            parts.append(("hat accent", sphere(0.12 * R).at(sx * 0.5 * R, top + 0.6 * R, 0.05 * R), accent))
    return parts


# ===================================================================================================== face extras

def _glasses(head, lay, kind="round"):
    R = head.R
    ex, ey = lay["ex"], lay["ey"]
    P, _ = head.front(np.array([ex, -ex]), np.array([ey, ey]))
    rr = max(lay["erx"], lay["ery"]) * 1.5
    z = float(P[:, 2].max()) + 0.07 * R
    rims, lenses = [], []
    for sx in (1, -1):
        c = (sx * ex, ey, z)
        rims.append(_xf(torus(rr, 0.03 * R), _Rx(90), c))
        if kind == "sunglasses":
            lenses.append(_xf(cylinder(rr, 0.02 * R), _Rx(90), c))
        rims.append(capsule((sx * (ex + rr), ey + 0.02 * R, z - 0.02 * R),
                            (sx * head.ca[0] * 1.03, ey + 0.06 * R, -0.25 * R), 0.026 * R))
    rims += _chain([(ex - rr, ey + 0.02 * R, z), (0, ey + 0.07 * R, z + 0.02 * R), (-ex + rr, ey + 0.02 * R, z)],
                   [0.026 * R] * 3)
    return rims, lenses


def _beard(head, lay, kind="full"):
    R = head.R
    my = lay["my"]
    out = []
    if kind in ("full", "santa", "short"):
        t = {"full": 0.12, "santa": 0.2, "short": 0.06}[kind] * R
        b = intersect(head.shape_(t), plane((0, 1, -0.15), my - 0.08 * R))
        b = intersect(b, plane((0, 0, -1), 0.15 * R))
        if kind == "santa":
            b = smooth_union(b, _flat_ell(0.62 * R, 0.62 * R, 0.4 * R, None, (0, my - 0.62 * R, 0.62 * R)), k=0.2 * R)
        out.append(b)
    if kind in ("full", "santa", "moustache", "short"):
        P, _ = head.front(np.array([0.0]), np.array([my + 0.1 * R]))
        z = float(P[0, 2])
        for sx in (1, -1):
            out.append(_xf(ellipsoid(0.17 * R, 0.07 * R, 0.08 * R), _Rz(sx * 12), (sx * 0.13 * R, my + 0.1 * R, z - 0.01 * R)))
    if kind == "goatee":
        P, _ = head.front(np.array([0.0]), np.array([my - 0.3 * R]))
        out.append(sphere(0.14 * R).at(0, my - 0.28 * R, float(P[0, 2]) - 0.06 * R))
    return out
