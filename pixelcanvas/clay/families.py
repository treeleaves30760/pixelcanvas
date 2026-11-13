# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from .. import solid as _S
from ..color import F32
from ..solid import (Material, ellipsoid, intersect, plane, rounded_box, sphere, torus)
from .shapes import _Checker, _Rx, _Ry, _Rz, _axis_rot, _chain, _lin, _look_rot, _mix, _shade, _xf, round_cone, smooth_union
from .._args import check_choice as _check_choice
from .animal_build import _amat, _bead_eyes, _cute_face, _eye_mats, _is_cute, _on_surface
from .markings import _smooth, _worley


# ----------------------------------------------------------------------------------------------- other families

class _Paint(_Checker):
    """Simple procedural colour in a creature's own frame: kind = 'belly' (lighter underside), 'bands' (rings along
    z), 'spots', 'shell' (cell lines), 'stripes'; anti-aliased with the pixel footprint."""

    _kind = None

    def __init__(self, base, kind="none", col2="#ffffff", axis_z=(0.0, 1.0), freq=6.0, width=0.3, level=0.0,
                 centre=(0, 0, 0), seed=1, scale=1.0, col3=None):
        self.base, self.c2 = _lin(base), _lin(col2)
        self.c3 = _lin(col3) if col3 else None
        self.kind, self.z0, self.z1 = kind, float(axis_z[0]), float(axis_z[1])
        self.freq, self.width, self.level, self.seed = float(freq), float(width), float(level), seed
        self.centre = np.asarray(centre, F32)
        self.scale = float(scale)

    def __call__(self, P, N=None, foot=None):
        P = np.asarray(P, F32)
        n = len(P)
        w = F32(0.01) if foot is None else np.maximum(foot / F32(self.scale) * 1.5, F32(0.003))
        a = np.zeros(n, F32)
        Q = P - self.centre
        if self.kind == "belly":
            a = 1 - _smooth(self.level - 0.04 - w, self.level + 0.04 + w, Q[:, 1])
        elif self.kind in ("bands", "stripes"):
            u = (Q[:, 2] - self.z0) / max(self.z1 - self.z0, 1e-6)
            v = u * self.freq
            if self.kind == "stripes":
                v = v + (_S.fbm3(P[:, 0] * 4, P[:, 1] * 4, P[:, 2] * 4, 2, self.seed) - 0.5) * 0.6
            fr = np.abs(v - np.floor(v) - 0.5) * 2
            ww = np.clip(w * self.freq / max(self.z1 - self.z0, 1e-6) * 2, 0.02, 0.5)
            a = 1 - _smooth(self.width - ww, self.width + ww, fr)
            a = a * ((u > -0.05) & (u < 1.05)).astype(F32)
        elif self.kind in ("spots", "shell"):
            F1, F2 = _worley(Q * F32(self.freq), self.seed)
            if self.kind == "spots":
                ww = np.clip(w * self.freq * 1.5, 0.01, 0.2)
                a = 1 - _smooth(self.width - ww, self.width + ww, F1)
            else:
                ww = np.clip(w * self.freq * 1.5, 0.01, 0.2)
                a = 1 - _smooth(0.06 - ww, 0.06 + ww, F2 - F1)
        col = self.base[None] * (1 - a[:, None]) + self.c2[None] * a[:, None]
        if self.c3 is not None and self.kind != "belly":
            b = 1 - _smooth(self.level - 0.03 - w, self.level + 0.03 + w, Q[:, 1])
            col = col * (1 - b[:, None]) + self.c3[None] * b[:, None]
        return col


def _bird(sp, name, ph, detail, style):
    acc, order = {}, []

    def add(nm, shapes, mat):
        shapes = [s for s in shapes if s is not None]
        if shapes:
            if nm not in acc:
                acc[nm] = [[], mat]
                order.append(nm)
            acc[nm][0].extend(shapes)
    w = 2 * math.pi * ph
    bx, by, bz = sp["body"]
    fly = name == "fly"
    tilt = sp.get("tilt", 0) if not fly else 5
    if name == "peck":
        tilt -= 25
    legs = sp["legs"] if not fly else 0.0
    if name in ("sit", "swim"):
        legs = 0.0
    R_b = _Rx(-tilt)
    lowest = max(by, abs(math.sin(math.radians(tilt))) * bz * 0.9 if tilt else by)
    cy = legs + lowest * 0.92 + (0.6 if fly else 0.0)
    C = np.array([0, cy, 0])
    body = _xf(ellipsoid(bx, by, bz), R_b, C)
    fwd = R_b @ np.array([0, 0, 1.0])
    upv = R_b @ np.array([0, 1.0, 0])
    nl = sp.get("neck", 0.2)
    cute = sp["cute"] / 1.15 if _is_cute(sp) else 1.0      # same scale as four-legged animals (default 1.15)
    hd = sp["head"] * cute
    if by > bz * 0.95 and not fly:
        hc = C + upv * (by * 0.78 + hd * 0.5 + nl * 0.5) + fwd * bz * 0.12
    else:
        hc = C + fwd * (bz * 0.75) + np.array([0, by * 0.55 + nl * 0.85 + hd * 0.3, nl * 0.15])
    if name == "peck":
        hc = C + fwd * (bz * 1.0 + hd) + np.array([0, -cy * 0.55, 0])
    if sp.get("name") == "flamingo":
        mid = C + fwd * bz * 0.9 + np.array([0, nl * 0.45, -0.05])
        neck = _chain([C + fwd * bz * 0.6, mid, mid + np.array([0, nl * 0.35, 0.15]), hc], [0.08, 0.05, 0.05, 0.06])
    else:
        neck = [round_cone(C + fwd * bz * 0.5 + upv * by * 0.3, hc, min(bx, by) * 0.55, hd * 0.75)] if nl > 0.03 else []
    head = sphere(hd).at(*hc)
    col = sp["color"]
    belly = sp.get("belly")
    paint = _Paint(col, "belly" if belly else "none", belly or col, level=-0.15 * by, centre=C)
    add("body", [smooth_union(body, head, *neck, k=min(0.06, hd * 0.16))], _amat(paint, style, "plush" if sp.get("name") == "chick" else "fur"))
    if sp.get("head_col"):
        add("head feathers", [sphere(hd * 1.02).at(*hc)] + ([round_cone(C + fwd * bz * 0.5 + upv * by * 0.3, hc, min(bx, by) * 0.56, hd * 0.76)]
                                                         if nl > 0.03 and sp.get("name") != "eagle" else []),
            _amat(sp["head_col"], style))
    # beak
    bl = sp["beak_len"]
    bdir = np.array([0, -0.15, 1.0])
    if name == "peck":
        bdir = np.array([0, -1.0, 0.5])
    bdir /= np.linalg.norm(bdir)
    b0 = hc + bdir * hd * 0.82
    kind = sp["beak"]
    if kind == "bill":
        beak = [_xf(ellipsoid(hd * 0.42, hd * 0.12, bl * 0.55), _look_rot(bdir), b0 + bdir * bl * 0.4)]
    elif kind == "hook":
        tip = b0 + bdir * bl + np.array([0, -bl * 0.55, 0])
        beak = _chain([b0, b0 + bdir * bl * 0.75 + np.array([0, -bl * 0.1, 0]), tip], [hd * 0.3, hd * 0.18, hd * 0.02])
    elif kind == "bent":
        m = b0 + bdir * bl * 0.5
        beak = _chain([b0, m, m + np.array([0, -bl * 0.6, bl * 0.2])], [hd * 0.3, hd * 0.2, hd * 0.05])
    else:
        beak = [round_cone(b0 - bdir * hd * 0.1, b0 + bdir * bl, hd * 0.28, hd * 0.03)]
    add("beak", beak, _amat(sp["beak_col"], style, "hard"))
    # eyes
    eye_style = sp.get("eye_style", "bead")
    az = 32 if eye_style == "owl" else 58
    dirs = [np.array([s * math.sin(math.radians(az)), 0.25, math.cos(math.radians(az))]) for s in (1, -1)]
    if name == "peck":
        dirs = [d + np.array([0, -0.3, 0]) for d in dirs]
    head_fn = lambda q: np.sqrt(((q - F32(hc)) ** 2).sum(1)) - F32(hd)    # noqa: E731
    E = _bead_eyes(head_fn, hc, hd, dirs, sp["eye"] * cute ** 1.3, style=eye_style)
    em = _eye_mats()
    for nm in ("eye whites", "eyes", "eye shine"):
        add(nm, E[nm], em[nm])
    if cute > 1.0:                      # blush under the eyes; the beak is the mouth, so no painted smile
        CF, cm = _cute_face(head_fn, hc, hd, dirs, sp["eye"] * cute * (1.3 if eye_style == "owl" else 1.0), smile=False)
        add("blush", CF["blush"], cm["blush"])
    if eye_style == "owl":
        add("face discs", [_xf(ellipsoid(hd * 0.5, hd * 0.5, hd * 0.12), _look_rot(d), hc + d / np.linalg.norm(d) * hd * 0.85)
                           for d in dirs], _amat(_mix(col, "#ffffff", 0.45), style))
    # wings
    wc = sp.get("wing_col", col)
    ws = sp.get("wings", 0.5)
    wings = []
    for s in (1, -1):
        if fly:
            a = 25 * math.sin(w)
            Rw = _Rz(s * a) @ _Rx(-tilt)
            base = C + upv * by * 0.4 + np.array([s * bx * 0.6, 0, 0])
            span = ws * 2.2
            pts = [base, base + Rw @ np.array([s * span * 0.45, 0.02, 0]), base + Rw @ np.array([s * span, -0.05, -0.12])]
            for i in range(len(pts) - 1):
                a_, b_ = pts[i], pts[i + 1]
                Rr = _axis_rot((1, 0, 0), b_ - a_)
                wings.append(_xf(ellipsoid(np.linalg.norm(b_ - a_) * 0.62, 0.035, ws * (0.55 - 0.15 * i)), Rr, (a_ + b_) / 2))
        elif sp.get("wing_style") == "flipper":
            Rw = R_b @ _Rz(s * 12)
            wings.append(_xf(ellipsoid(0.05, ws * 0.7, ws * 0.28), Rw, C + R_b @ np.array([s * bx * 0.95, -by * 0.05, 0])))
        else:
            Rw = R_b @ _Ry(-s * 6)
            wings.append(_xf(ellipsoid(bx * 0.32, by * 0.62, bz * 0.78), Rw, C + R_b @ np.array([s * bx * 0.72, by * 0.08, -bz * 0.1])))
    add("wings", wings, _amat(wc, style))
    # tail
    tk = sp.get("tail")
    tc = sp.get("tail_col", col)
    back = C - fwd * bz * 0.85
    tails = []
    if tk == "fan_up":
        for a in (-25, 0, 25):
            tails.append(_xf(ellipsoid(0.08, 0.32, 0.18), _Rz(a) @ _Rx(-25), back + np.array([0, by * 0.5, -0.05])))
    elif tk == "up":
        tails.append(_xf(ellipsoid(0.12, 0.05, 0.2), _Rx(30), back + np.array([0, by * 0.3, -0.08])))
    elif tk == "fan":
        tails.append(_xf(ellipsoid(0.28, 0.04, 0.3), _Rx(-tilt * 0.6 + 10), back - fwd * 0.18))
    elif tk == "long":
        tails += _chain([back, back - fwd * 0.4 + np.array([0, -0.25, 0]), back - fwd * 0.8 + np.array([0, -0.6, 0])],
                        [0.1, 0.07, 0.03])
    elif tk == "short":
        tails.append(_xf(ellipsoid(0.12, 0.04, 0.16), R_b @ _Rx(-10), back - fwd * 0.08))
    add("tail", tails, _amat(tc, style))
    # legs
    if legs > 0.01:
        lg = []
        for s in (1, -1):
            if sp.get("one_leg") and s == -1:
                k = np.array([s * 0.08, legs * 1.0 + 0.02, 0.0])
                lg += _chain([np.array([s * 0.1, cy - by * 0.6, 0]), k + np.array([0, 0, -0.2]), k + np.array([0, -0.05, 0.1])],
                             [0.035, 0.03, 0.025])
                continue
            hip = np.array([s * bx * 0.35, cy - by * 0.7, 0.0])
            ft = np.array([s * bx * 0.4, 0.02, 0.05])
            step = 0.12 * math.sin(w + (0 if s > 0 else math.pi)) if name == "walk" else 0.0
            ft = ft + np.array([0, max(0.0, step) * 0.5, step])
            knee = (hip + ft) / 2 + np.array([0, 0, -0.05])
            lg += _chain([hip, knee, ft], [0.04, 0.032, 0.03])
            for a in (-35, 0, 35):
                d = _Ry(a) @ np.array([0, 0, 1.0])
                lg.append(round_cone(ft, ft + d * 0.12 + np.array([0, -0.0, 0]), 0.022, 0.015))
            lg.append(round_cone(ft, ft + np.array([0, 0.0, -0.08]), 0.02, 0.014))
        add("legs", lg, _amat(sp["leg_col"], style, "hard"))
    # extras
    for ex in sp.get("extras", ()):
        if ex == "comb":
            add("comb", [sphere(hd * 0.22).at(*(hc + np.array([0, hd * 0.95, hd * (0.3 - 0.3 * i)]))) for i in range(3)],
                _amat("#e2322a", style))
        elif ex == "wattle":
            add("comb", [ellipsoid(hd * 0.14, hd * 0.25, hd * 0.12).at(*(b0 + np.array([0, -hd * 0.35, -hd * 0.05])))],
                _amat("#e2322a", style))
        elif ex == "cheek_patch":
            add("cheeks", [_xf(ellipsoid(hd * 0.35, hd * 0.25, hd * 0.12), _look_rot(np.array([s * 0.8, 0, 0.6])),
                               hc + np.array([s * hd * 0.72, -hd * 0.1, hd * 0.55])) for s in (1, -1)],
                _amat("#f8f6f2", style))
            # the white shirt-front: a skin over the front of the body (it follows the body, never sticks out)
            front = intersect(ellipsoid(bx * 1.012, by * 1.008, bz * 1.012), plane((0, 0, -1), -bz * 0.3), k=0.02)
            add("cheeks", [_xf(intersect(front, plane((0, 1, 0), by * 0.62)), R_b, C)], _amat("#f8f6f2", style))
        elif ex == "ear_tufts":
            add("wings", [round_cone(hc + np.array([s * hd * 0.55, hd * 0.75, 0]), hc + np.array([s * hd * 0.75, hd * 1.25, -0.02]),
                                     hd * 0.18, hd * 0.03) for s in (1, -1)], _amat(wc, style))
        elif ex == "wing_tips":
            pass
    if sp.get("brow"):
        add("head feathers", [_xf(ellipsoid(hd * 0.35, hd * 0.1, hd * 0.25), _Rz(s * 15), hc + np.array([s * hd * 0.45, hd * 0.45, hd * 0.6]))
                              for s in (1, -1)], _amat(sp.get("head_col", col), style))
    parts = [(nm, acc[nm][0], acc[nm][1]) for nm in order]
    A = dict(head=hc, beak=b0 + bdir * bl, back=C + upv * by, tail_tip=back, feet=np.array([0, 0, 0.0]),
             head_top=hc + np.array([0, hd, 0]))
    return parts, A


def _fish(sp, name, ph, detail, style):
    acc, order = {}, []

    def add(nm, shapes, mat):
        if shapes:
            if nm not in acc:
                acc[nm] = [[], mat]
                order.append(nm)
            acc[nm][0].extend(shapes)
    shape = _check_choice(f"animal({sp.get('name', 'fish')!r})", "shape", sp.get("shape", "round"), FISH_SHAPES)
    if shape == "round" and sp.get("name") == "whale":
        return _round_whale(sp, name, ph, style)
    w = 2 * math.pi * ph
    sway = 0.12 * math.sin(w)
    if shape == "round":
        Lb, hgt, thick = 1.0, 0.62, 0.26
    elif shape == "shark":
        Lb, hgt, thick = 1.0, 0.24, 0.22
    elif shape == "whale":
        Lb, hgt, thick = 1.0, 0.3, 0.3
    else:
        Lb, hgt, thick = 1.0, 0.22, 0.2
    vertical = shape in ("whale", "dolphin")
    n = 5
    zs = np.linspace(0.5, -0.5, n) * Lb
    prof = {"round": [0.3, 0.95, 1.0, 0.7, 0.25], "shark": [0.45, 0.95, 1.0, 0.65, 0.22],
            "whale": [0.85, 1.0, 0.95, 0.6, 0.2], "dolphin": [0.45, 0.95, 1.0, 0.6, 0.2]}[shape]
    cy = hgt * 1.25
    pts, rad = [], []
    for i, z in enumerate(zs):
        off = sway * ((i / (n - 1)) ** 2)
        p = np.array([0 if vertical else off, cy + (off if vertical else 0), z])
        pts.append(p)
        rad.append(prof[i] * hgt * 0.5)
    S = np.diag([thick / hgt, 1.0, 1.0]) if not vertical else np.diag([thick / hgt * 1.0, 1.0, 1.0])
    pts_s = [np.linalg.solve(S, p) for p in pts]
    body = _xf(smooth_union(*_chain(pts_s, rad), k=0.05), S, (0, 0, 0))
    col = sp["color"]
    pat = sp.get("pattern", "none")
    if pat == "belly":
        paint = _Paint(col, "belly", sp.get("belly_col", "#eeeeee"), level=cy - hgt * 0.08)
    elif pat == "grooves":
        paint = _Paint(col, "belly", sp.get("belly_col", "#eeeeee"), level=cy - hgt * 0.12)
    elif pat == "bands":
        paint = _Paint(col, "bands", sp.get("pat_col", "#ffffff"), axis_z=(-0.45, 0.5), freq=2.5, width=0.28)
    elif pat == "stripes":
        paint = _Paint(col, "stripes", sp.get("pat_col", "#ffffff"), axis_z=(-0.45, 0.5), freq=5, width=0.25)
    elif pat == "spots":
        paint = _Paint(col, "spots", sp.get("pat_col", "#ffffff"), freq=9, width=0.22)
    else:
        paint = col
    add("body", [body], _amat(paint, style if shape == "round" else "plastic"))
    fc = sp.get("fin_col", col)
    fins = []
    tail_b = pts[-1]
    if vertical:
        fl = _S.prism([(0, 0), (-0.32, -0.12), (-0.28, 0.02), (0, -0.04), (0.28, 0.02), (0.32, -0.12)], depth=0.03)
        fins.append(_xf(fl, _Rx(90) @ _Rz(0) @ np.eye(3), tail_b + np.array([0, 0, -0.02])))
    else:
        tl = 0.28 if shape == "round" else 0.32
        tri = _S.prism([(0, 0), (-tl, 0.35 * (1 if shape == "round" else 1.3)), (-tl * 0.75, 0), (-tl, -0.3)], depth=0.025)
        fins.append(_xf(tri, _Ry(90), tail_b + np.array([0, 0, 0.02])))
    top = pts[1] * 0.5 + pts[2] * 0.5
    if shape != "whale":
        dh = {"round": 0.22, "shark": 0.3, "dolphin": 0.18}[shape]
        dor = _S.prism([(-0.16, 0), (0.05, dh), (0.16, 0)], depth=0.025)
        fins.append(_xf(dor, _Ry(90), top + np.array([0, hgt * 0.42, 0])))
    for s in (1, -1):
        fins.append(_xf(ellipsoid(0.02, 0.06 if shape == "round" else 0.05, 0.13), _Rz(s * 50) @ _Rx(30),
                        pts[1] + np.array([s * thick * 0.45, -hgt * 0.15, -0.05])))
    if shape == "round":
        fins.append(_xf(_S.prism([(-0.1, 0), (0.05, -0.12), (0.12, 0)], depth=0.02), _Ry(90), pts[2] + np.array([0, -hgt * 0.42, -0.05])))
    add("fins", fins, _amat(fc, style if shape == "round" else "plastic"))
    hc = pts[0] + np.array([0, 0, -0.06])
    hr = rad[0] * 1.6
    az = 70
    dirs = [np.array([s * math.sin(math.radians(az)), 0.25, math.cos(math.radians(az))]) for s in (1, -1)]
    cute = sp["cute"] / 1.15 if _is_cute(sp) else 1.0
    body_fn = lambda q: body._d(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy())     # noqa: E731
    eye_c = pts[0] * 0.6 + pts[1] * 0.4
    es = (0.05 if shape == "round" else 0.03) * cute ** 1.5
    E = _bead_eyes(body_fn, eye_c, hr, dirs, es)
    em = _eye_mats()
    for nm in ("eyes", "eye shine"):
        add(nm, E[nm], em[nm])
    if cute > 1.0:
        CF, cm = _cute_face(body_fn, eye_c, hr, dirs, es, smile=False)
        add("blush", CF["blush"], cm["blush"])
    if shape == "shark":
        add("gills", [_xf(rounded_box(0.005, 0.07, 0.01, 0.003), np.eye(3), pts[1] + np.array([s * thick * 0.46, 0.0, 0.04 * k]))
                      for s in (1, -1) for k in range(3)], Material(color=_shade(col, 0.7), roughness=0.5))
    parts = [(nm, acc[nm][0], acc[nm][1]) for nm in order]
    return parts, dict(head=pts[0], back=top + np.array([0, hgt * 0.5, 0]), tail_tip=tail_b, feet=np.array([0, 0, 0.0]))


FISH_SHAPES = ("round", "shark", "whale", "dolphin")


def _round_whale(sp, name, ph, style):
    """The round cartoon whale (``animal("whale", shape="round")``): a big round head-body, a short tail that sweeps
    up into wide flukes, little flippers, a pale grooved belly, a smile and a blowhole."""
    acc, order = {}, []

    def add(nm, shapes, mat):
        if shapes:
            if nm not in acc:
                acc[nm] = [[], mat]
                order.append(nm)
            acc[nm][0].extend(shapes)
    w = 2 * math.pi * ph
    beat = 0.06 * math.sin(w) if name == "swim" else 0.0
    cy = 0.44 if name == "swim" else 0.37             # swimming it floats; otherwise it rests on its belly
    C = np.array([0.0, cy, 0.1])
    trunk = ellipsoid(0.4, 0.36, 0.44).at(*C)
    tail_pts = [np.array([0.0, cy - 0.02, -0.12]), np.array([0.0, cy + 0.0, -0.4]),
                np.array([0.0, cy + 0.09 + beat * 0.5, -0.6]), np.array([0.0, cy + 0.22 + beat, -0.72])]
    tail = _chain(tail_pts, [0.3, 0.17, 0.085, 0.05])
    body = smooth_union(trunk, *tail, k=0.12)
    col = sp["color"]
    belly_c = sp.get("belly_col", "#e8ecf0")
    paint = _Paint(col, "belly", belly_c, level=cy - 0.1, centre=(0, 0, 0))
    mat = _amat(paint, style if style != "clay" else "vinyl")
    add("body", [body], mat)
    # grooves on the throat: a few shallow ridges in a darker belly tone
    grooves = []
    for k in (-2, -1, 0, 1, 2):
        x = k * 0.075
        a = np.array([x, cy - 0.3 + abs(k) * 0.03, 0.34 - abs(k) * 0.03])
        b = np.array([x * 1.15, cy - 0.33 + abs(k) * 0.035, 0.02])
        grooves.append(round_cone(a, b, 0.012, 0.01))
    add("grooves", grooves, _amat(_shade(belly_c, 0.84), style if style != "clay" else "vinyl"))
    # flukes: two wide lobes, flat to the ground and swept back from the tail tip
    tip = tail_pts[-1]
    up = tail_pts[-1] - tail_pts[-2]
    pitch = math.degrees(math.atan2(up[1], -up[2]))
    flukes = []
    for s in (1, -1):
        lobe = _xf(ellipsoid(0.17, 0.028, 0.085), _Rx(-pitch * 0.6) @ _Ry(s * 28),
                   tip + np.array([s * 0.13, 0.02, -0.06]))
        flukes.append(lobe)
    flippers = []
    for s in (1, -1):
        flippers.append(_xf(ellipsoid(0.15, 0.03, 0.075), _Rz(s * 32) @ _Ry(s * 30),
                            C + np.array([s * 0.38, -0.17, -0.02])))
    fin_mat = _amat(sp.get("fin_col", col), style if style != "clay" else "vinyl")
    add("fins", [smooth_union(*flukes, tail[-1], k=0.05)] + flippers, fin_mat)
    # eyes low on the face, a wide smile between them, a blowhole on top
    cute = sp["cute"] / 1.15 if _is_cute(sp) else 1.0
    body_fn = lambda q: body._d(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy())     # noqa: E731
    eye_c = C + np.array([0, -0.02, 0.05])
    dirs = [np.array([s * math.sin(math.radians(44)), -0.06, math.cos(math.radians(44))]) for s in (1, -1)]
    es = 0.045 * cute ** 1.5
    E = _bead_eyes(body_fn, eye_c, 0.45, dirs, es)
    em = _eye_mats()
    for nm in ("eyes", "eye shine"):
        add(nm, E[nm], em[nm])
    # the smile: a shallow U traced over the face, each point dropped onto the surface
    arc = [_on_surface(body_fn, eye_c, np.array([0.42 * math.sin(t), -0.16 - 0.1 * math.cos(t), 1.0]), 0.8)
           for t in np.linspace(-1.2, 1.2, 9)]
    add("smile", [_S.union(*_chain(arc, [0.011] * len(arc)))],
        Material(color="#1e2a40", roughness=0.5, specular=0.2))
    top = _on_surface(body_fn, C, np.array([0, 1.0, 0.25]), 0.8)
    add("blowhole", [ellipsoid(0.035, 0.012, 0.022).at(*(top - np.array([0, 0.006, 0])))],
        Material(color=_shade(col, 0.55), roughness=0.6))
    if cute > 1.0:
        CF, cm = _cute_face(body_fn, eye_c, 0.45, dirs, es, smile=False)
        add("blush", CF["blush"], cm["blush"])
    parts = [(nm, acc[nm][0], acc[nm][1]) for nm in order]
    return parts, dict(head=C + np.array([0, 0, 0.3]), back=top, tail_tip=tip, feet=np.array([0, 0, 0.0]))


def _misc(sp, name, ph, detail, style):
    fam = sp["family"]
    acc, order = {}, []
    w = 2 * math.pi * ph

    def add(nm, shapes, mat):
        shapes = [s for s in shapes if s is not None]
        if shapes:
            if nm not in acc:
                acc[nm] = [[], mat]
                order.append(nm)
            acc[nm][0].extend(shapes)
    col = sp["color"]
    em = _eye_mats()
    A = {}
    if fam == "turtle":
        shell = intersect(ellipsoid(0.42, 0.3, 0.5).at(0, 0.12, 0), plane((0, -1, 0), -0.1))
        add("shell", [shell.round(0.01)], _amat(_Paint(sp["shell"], "shell", sp["shell_line"], freq=4.5, seed=3), style))
        add("body", [ellipsoid(0.4, 0.06, 0.47).at(0, 0.1, 0)], _amat(_mix(col, "#f2e8b0", 0.5), style))
        hc = np.array([0, 0.2, 0.62])
        body = [round_cone((0, 0.14, 0.4), hc, 0.09, 0.12), sphere(0.13).at(*hc)]
        for s in (1, -1):
            for zz, a in ((0.3, 1), (-0.32, -1)):
                sw_ = 15 * math.sin(w + (0 if s * a > 0 else math.pi))
                body.append(_xf(ellipsoid(0.17, 0.05, 0.09), _Ry(s * (35 * a + sw_)) @ _Rz(-s * 10),
                                np.array([s * 0.38, 0.07, zz])))
        body.append(round_cone((0, 0.1, -0.45), (0, 0.08, -0.6), 0.05, 0.01))
        add("skin", [smooth_union(*body, k=0.03)], _amat(col, style))
        E = _bead_eyes(lambda q: np.sqrt(((q - F32(hc)) ** 2).sum(1)) - F32(0.13), hc, 0.13,
                       [np.array([s * 0.6, 0.35, 0.7]) for s in (1, -1)], 0.035)
        A = dict(head=hc, back=np.array([0, 0.42, 0]))
    elif fam == "octopus":
        mantle = ellipsoid(0.36, 0.42, 0.34).at(0, 0.62, -0.05)
        tent = []
        for i in range(8):
            a = 2 * math.pi * (i + 0.5) / 8
            d = np.array([math.sin(a), 0, math.cos(a)])
            base = np.array([0, 0.3, 0]) + d * 0.18
            pts = [base]
            for k in range(1, 6):
                r = 0.18 + k * 0.1
                curl = 0.25 * math.sin(w + i + k * 0.6)
                pts.append(np.array([0, 0.3, 0]) + d * r + np.cross([0, 1, 0], d) * curl * k * 0.08
                           + np.array([0, -0.25 * (1 - math.exp(-k * 0.8)) + 0.04 * k * (k > 3), 0]))
            tent += _chain(pts, [0.1, 0.085, 0.07, 0.05, 0.035, 0.02])
        add("body", [smooth_union(mantle, sphere(0.25).at(0, 0.32, 0), _S.union(*tent), k=0.08)],
            _amat(_Paint(col, "spots", sp.get("spot_col", "#f4a090"), freq=9, width=0.18), style))
        hc = np.array([0, 0.45, 0.05])
        E = _bead_eyes(lambda q: mantle._d(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy()), np.array([0, 0.48, -0.05]), 0.36,
                       [np.array([s * 0.45, -0.05, 0.9]) for s in (1, -1)], 0.07)
        A = dict(head=np.array([0, 0.7, 0]))
    elif fam == "crab":
        cara = ellipsoid(0.42, 0.16, 0.3).at(0, 0.3, 0)
        parts = [cara]
        for s in (1, -1):
            for k, zz in enumerate((0.12, -0.02, -0.16)):
                hip = np.array([s * 0.35, 0.28, zz])
                kn = hip + np.array([s * 0.25, 0.12, zz * 0.3])
                ft = kn + np.array([s * 0.15, -0.38, zz * 0.2 + 0.03 * math.sin(w + k + s)])
                parts += _chain([hip, kn, ft], [0.045, 0.035, 0.015])
            sh = np.array([s * 0.3, 0.32, 0.25])
            el = sh + np.array([s * 0.15, 0.08, 0.2])
            parts += _chain([sh, el], [0.06, 0.06])
            claw = _xf(ellipsoid(0.1, 0.07, 0.15), _Ry(s * 20), el + np.array([s * 0.05, 0.06, 0.15]))
            finger = round_cone(el + np.array([s * 0.08, 0.04, 0.28]), el + np.array([s * 0.02, 0.0, 0.38]), 0.035, 0.012)
            parts += [claw, finger]
        stalks = [round_cone((s * 0.1, 0.4, 0.22), (s * 0.12, 0.56, 0.26), 0.022, 0.02) for s in (1, -1)]
        add("shell", [smooth_union(cara, _S.union(*parts[1:]), k=0.03), _S.union(*stalks)], _amat(col, style))
        E = {"eyes": [sphere(0.045).at(s * 0.12, 0.6, 0.27) for s in (1, -1)],
             "eye shine": [sphere(0.012).at(s * 0.12 - 0.012, 0.625, 0.31) for s in (1, -1)]}
        add("mouth", [_xf(intersect(torus(0.06, 0.012), plane((0, 1, 0), 0.0)), _Rx(90), (0, 0.32, 0.3))],
            Material(color="#5a2420", roughness=0.5))
        A = dict(head=np.array([0, 0.45, 0.25]))
    elif fam == "seal":
        pts = [np.array([0, 0.35, 0.45]), np.array([0, 0.24, 0.15]), np.array([0, 0.18, -0.2]), np.array([0, 0.12, -0.5])]
        body = _chain(pts, [0.2, 0.24, 0.17, 0.08])
        hc = np.array([0, 0.48, 0.55])
        head = sphere(0.17).at(*hc)
        snout = ellipsoid(0.1, 0.07, 0.08).at(*(hc + np.array([0, -0.05, 0.13])))
        fl = [_xf(ellipsoid(0.13, 0.03, 0.07), _Rz(s * -30) @ _Ry(s * 20), np.array([s * 0.24, 0.12, 0.25])) for s in (1, -1)]
        tail = [_xf(ellipsoid(0.12, 0.025, 0.09), _Ry(s * 35), np.array([s * 0.08, 0.08, -0.6])) for s in (1, -1)]
        add("body", [smooth_union(*body, head, snout, k=0.06), _S.union(*fl, *tail)],
            _amat(_Paint(col, "spots", sp.get("spot_col", "#6a727c"), freq=7, width=0.16), style))
        add("nose", [ellipsoid(0.035, 0.022, 0.02).at(*(hc + np.array([0, -0.02, 0.22])))], Material(color="#2a2224", roughness=0.3))
        add("whisker dots", [sphere(0.008).at(*(hc + np.array([s * (0.03 + 0.02 * k), -0.06, 0.2]))) for s in (1, -1)
                             for k in range(3)], Material(color="#3a3436", roughness=0.4))
        E = _bead_eyes(lambda q: head._d(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy()), hc, 0.17,
                       [np.array([s * 0.5, 0.3, 0.8]) for s in (1, -1)], 0.04)
        A = dict(head=hc, back=np.array([0, 0.45, 0.1]))
    elif fam == "frog":
        body = _xf(ellipsoid(0.32, 0.24, 0.36), _Rx(-25), (0, 0.24, 0))
        head = ellipsoid(0.3, 0.18, 0.26).at(0, 0.36, 0.22)
        legs = []
        for s in (1, -1):
            legs.append(_xf(ellipsoid(0.12, 0.13, 0.25), _Ry(s * 20), np.array([s * 0.27, 0.13, -0.1])))
            legs.append(_xf(ellipsoid(0.08, 0.03, 0.18), _Ry(s * 15), np.array([s * 0.33, 0.025, 0.08])))
            legs.append(round_cone((s * 0.18, 0.2, 0.2), (s * 0.2, 0.03, 0.32), 0.05, 0.04))
            legs.append(_xf(ellipsoid(0.06, 0.02, 0.06), np.eye(3), np.array([s * 0.21, 0.02, 0.36])))
        bumps = [sphere(0.1).at(s * 0.14, 0.5, 0.25) for s in (1, -1)]
        add("skin", [smooth_union(body, head, *bumps, k=0.06), _S.union(*legs)],
            _amat(_Paint(col, "belly", sp.get("belly_col", "#e8f0b8"), level=0.24, centre=(0, 0, 0)), style))
        add("mouth", [_xf(intersect(torus(0.22, 0.01), plane((0, 1, 0), 0.0)), _Rx(90) @ _Rx(180), (0, 0.34, 0.27))],
            Material(color="#2a4a2a", roughness=0.5, shadow=False))
        E = {"eyes": [sphere(0.07).at(s * 0.14, 0.53, 0.3) for s in (1, -1)],
             "eye shine": [sphere(0.02).at(s * 0.14 - 0.02, 0.57, 0.36) for s in (1, -1)]}
        A = dict(head=np.array([0, 0.45, 0.25]))
    elif fam == "snail":
        foot = _chain([np.array([0, 0.06, -0.35]), np.array([0, 0.07, 0.0]), np.array([0, 0.12, 0.3]), np.array([0, 0.25, 0.42])],
                      [0.05, 0.09, 0.09, 0.08])
        stalks = [round_cone((s * 0.04, 0.3, 0.44), (s * 0.08, 0.45, 0.5), 0.02, 0.015) for s in (1, -1)]
        add("body", [smooth_union(*foot, k=0.05), _S.union(*stalks)], _amat(col, style))
        coil = []
        for i in range(18):
            t = i / 17
            a = t * 2.6 * math.pi
            r = 0.2 * (1 - 0.72 * t)
            coil.append(sphere(0.15 * (1 - 0.7 * t)).at(0.0 + 0.02 * t, 0.3 + r * math.sin(a), -0.05 + r * math.cos(a)))
        add("shell", [smooth_union(*coil, k=0.04)], _amat(_Paint(sp["shell"], "bands", sp["shell_line"], axis_z=(-0.3, 0.2),
                                                                 freq=4, width=0.25), style, "hard"))
        E = {"eyes": [sphere(0.03).at(s * 0.08, 0.46, 0.51) for s in (1, -1)],
             "eye shine": [sphere(0.009).at(s * 0.08 - 0.01, 0.475, 0.535) for s in (1, -1)]}
        A = dict(head=np.array([0, 0.3, 0.45]))
    elif fam in ("beetle", "bee", "butterfly", "caterpillar"):
        if fam == "beetle":
            dome = intersect(ellipsoid(0.34, 0.26, 0.42).at(0, 0.12, -0.02), plane((0, -1, 0), -0.12))
            spots = [(0.15, 0.3, 0.12), (-0.15, 0.3, 0.12), (0.2, 0.25, -0.12), (-0.2, 0.25, -0.12), (0, 0.36, -0.05)]

            class _LB(_Checker):
                _kind = None

                def __call__(self, P, N=None, foot=None):
                    w_ = 0.01 if foot is None else np.maximum(foot * 1.5, 0.003)
                    a = np.zeros(len(P), F32)
                    for sx, sy, sz in spots:
                        d = np.sqrt((P[:, 0] - sx) ** 2 + (P[:, 1] - sy) ** 2 + (P[:, 2] - sz) ** 2)
                        a = np.maximum(a, 1 - _smooth(0.07 - w_, 0.07 + w_, d))
                    a = np.maximum(a, 1 - _smooth(0.01 - w_, 0.01 + w_, np.abs(P[:, 0])) * (P[:, 2] < 0.3))
                    c0, c1 = _lin(col), _lin(sp["spot_col"])
                    return c0[None] * (1 - a[:, None]) + c1[None] * a[:, None]
            add("shell", [dome.round(0.008)], Material(color=_LB(), roughness=0.18, specular=0.9, clearcoat=0.6))
            head = sphere(0.14).at(0, 0.14, 0.4)
            legs = [round_cone((s * 0.2, 0.1, z), (s * 0.34, 0.01, z + 0.05), 0.02, 0.015) for s in (1, -1) for z in (0.15, 0, -0.15)]
            ant = [round_cone((s * 0.05, 0.24, 0.48), (s * 0.12, 0.34, 0.6), 0.012, 0.01) for s in (1, -1)]
            add("body", [head] + legs + ant, Material(color="#1e1a1c", roughness=0.35, specular=0.6))
            add("face dots", [sphere(0.03).at(s * 0.07, 0.2, 0.52) for s in (1, -1)], Material(color="#f8f6f2", roughness=0.4))
            E = {"eyes": [], "eye shine": []}
            A = dict(head=np.array([0, 0.2, 0.45]), back=np.array([0, 0.38, 0]))
        elif fam == "bee":
            thor = sphere(0.18).at(0, 0.42, 0.05)
            abd = _xf(ellipsoid(0.22, 0.22, 0.3), _Rx(-15), np.array([0, 0.38, -0.3]))
            add("body", [smooth_union(thor, abd, k=0.05)],
                Material(color=_Paint(col, "bands", sp["stripe_col"], axis_z=(-0.58, -0.02), freq=3.0, width=0.45),
                         roughness=0.8, specular=0.15, sheen=0.9, subsurface=0.2))
            head = sphere(0.15).at(0, 0.48, 0.28)
            sting = round_cone((0, 0.32, -0.58), (0, 0.3, -0.68), 0.03, 0.003)
            legs = [round_cone((s * 0.1, 0.3, z), (s * 0.2, 0.02, z + 0.05), 0.02, 0.015) for s in (1, -1) for z in (0.12, 0.0, -0.12)]
            ant = [round_cone((s * 0.05, 0.6, 0.34), (s * 0.12, 0.74, 0.44), 0.012, 0.01) for s in (1, -1)]
            add("head", [head, sting] + legs + ant + [sphere(0.025).at(s * 0.12, 0.74, 0.44) for s in (1, -1)],
                Material(color=sp["stripe_col"], roughness=0.5, specular=0.4))
            flap = 20 * math.sin(w)
            wings = [_xf(ellipsoid(0.28, 0.015, 0.14), _Rz(s * (25 + flap)) @ _Ry(s * -20),
                         np.array([s * 0.22, 0.62, -0.05])) for s in (1, -1)]
            wings += [_xf(ellipsoid(0.18, 0.012, 0.1), _Rz(s * (15 + flap)) @ _Ry(s * -40),
                          np.array([s * 0.18, 0.58, -0.2])) for s in (1, -1)]
            add("wings", wings, Material(color="#f2f6fa", roughness=0.15, specular=0.8, glass=0.55, shadow=0.3))
            E = {"eyes": [sphere(0.05).at(s * 0.08, 0.52, 0.39) for s in (1, -1)],
                 "eye shine": [sphere(0.015).at(s * 0.08 - 0.015, 0.55, 0.435) for s in (1, -1)]}
            A = dict(head=np.array([0, 0.5, 0.3]), back=np.array([0, 0.6, 0]))
        elif fam == "butterfly":
            body = _chain([np.array([0, 0.25, 0.3]), np.array([0, 0.25, 0.0]), np.array([0, 0.24, -0.35])], [0.06, 0.05, 0.03])
            add("body", body + [sphere(0.075).at(0, 0.27, 0.36)] +
                [round_cone((s * 0.03, 0.32, 0.4), (s * 0.12, 0.5, 0.6), 0.01, 0.008) for s in (1, -1)],
                Material(color="#2a2224", roughness=0.6, sheen=0.5))
            open_ = 25 + 45 * (0.5 + 0.5 * math.sin(w)) if ph else 35
            pc = sp.get("pat_col", "#2a2224")

            class _Wing(_Checker):
                _kind = None

                def __call__(self, P, N=None, foot=None):
                    w_ = 0.01 if foot is None else np.maximum(foot * 1.5, 0.003)
                    c0, c1, c2 = _lin(col), _lin(pc), _lin(sp.get("spot_col", "#f8f4ee"))
                    r = np.sqrt(P[:, 0] ** 2 + (P[:, 2] * 0.9) ** 2)
                    edge = _smooth(0.42 - w_, 0.46 + w_, r)
                    vein = 1 - _smooth(0.008, 0.02 + w_, np.abs(np.sin(np.arctan2(P[:, 2], np.abs(P[:, 0])) * 4)) * r * 0.5)
                    out = c0[None] * (1 - edge[:, None]) + c1[None] * edge[:, None]
                    out = out * (1 - vein[:, None] * 0.6) + c1[None] * vein[:, None] * 0.6
                    F1, _ = _worley(P * 9, 4)
                    sp_ = (1 - _smooth(0.18 - w_ * 9, 0.18 + w_ * 9, F1)) * edge
                    return out * (1 - sp_[:, None]) + c2[None] * sp_[:, None]
            wings = []
            for s in (1, -1):
                Rw = _Rz(s * open_)
                fw_ = _xf(ellipsoid(0.42, 0.012, 0.3), Rw @ _Ry(s * -25), np.array([0, 0.27, 0.08]) + Rw @ np.array([s * 0.38, 0, 0.08]))
                hw_ = _xf(ellipsoid(0.3, 0.012, 0.24), Rw @ _Ry(s * 30), np.array([0, 0.26, -0.12]) + Rw @ np.array([s * 0.26, 0, -0.1]))
                wings += [fw_, hw_]
            add("wings", wings, Material(color=_Wing(), roughness=0.6, specular=0.2, sheen=0.4))
            E = {"eyes": [sphere(0.03).at(s * 0.05, 0.3, 0.41) for s in (1, -1)], "eye shine": []}
            A = dict(head=np.array([0, 0.3, 0.4]), back=np.array([0, 0.5, 0]))
        else:
            n = 8
            segs = []
            for i in range(n):
                z = 0.42 - i * 0.12
                y = 0.09 + 0.04 * max(0.0, math.sin(w + i * 0.9))
                segs.append(sphere(0.1 if i else 0.13).at(0.03 * math.sin(i * 0.7 + w), y + (0.06 if i == 0 else 0), z))
            add("body", [smooth_union(*segs[1:], k=0.03)],
                _amat(_Paint(col, "bands", sp["stripe_col"], axis_z=(-0.5, 0.36), freq=7, width=0.22), style))
            add("head", [segs[0]] + [round_cone((s * 0.05, 0.25, 0.46), (s * 0.08, 0.36, 0.52), 0.012, 0.01) for s in (1, -1)],
                _amat(_shade(col, 0.85), style))
            add("feet", [sphere(0.03).at(s * 0.07, 0.025, 0.42 - i * 0.12) for s in (1, -1) for i in range(1, n)],
                _amat(_shade(col, 0.6), style))
            E = {"eyes": [sphere(0.035).at(s * 0.06, 0.2, 0.53) for s in (1, -1)],
                 "eye shine": [sphere(0.01).at(s * 0.06 - 0.01, 0.215, 0.565) for s in (1, -1)]}
            A = dict(head=np.array([0, 0.18, 0.45]))
    for nm in ("eyes", "eye shine"):
        add(nm, E.get(nm, []), em[nm])
    parts = [(nm, acc[nm][0], acc[nm][1]) for nm in order]
    A.setdefault("feet", np.array([0, 0, 0.0]))
    A.setdefault("back", A.get("head", np.zeros(3)))
    return parts, A
