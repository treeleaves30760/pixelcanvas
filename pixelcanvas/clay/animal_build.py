# -*- coding: utf-8 -*-
from __future__ import annotations

import copy as _copy
import math
import numpy as np

from .. import solid as _S
from ..color import F32
from ..solid import (Material, cone, cylinder, ellipsoid, intersect, plane, sphere, torus)
from .markings import _BodyPaint
from .shapes import _Checker, _Rx, _Ry, _Rz, _axis_rot, _chain, _lin, _look_rot, _shade, _xf, round_cone, smooth_union
from .animal_rig import _QuadRig, _folded_hind
from .hair import _fib_dirs


def _bead_eyes(head_fn, centre, R, dirs, size, color="#16100e", closed=False, style="bead", lid=None):
    """Glossy bead eyes (with a painted highlight) on a head surface; ``closed`` draws sleepy lid lines."""
    out = {"eyes": [], "eye shine": [], "eye whites": [], "closed eyes": []}
    for d in dirs:
        d = np.asarray(d, np.float64)
        d /= np.linalg.norm(d)
        lo, hi = 0.0, 3.0 * R
        for _ in range(26):
            m = (lo + hi) / 2
            p = centre + d * m
            if head_fn(np.asarray(p, F32)[None])[0] < 0:
                lo = m
            else:
                hi = m
        p = centre + d * lo
        if closed:
            Rr = _look_rot(d)
            arc = _xf(intersect(torus(size * 0.9, size * 0.2), plane((0, 0, 1), 0.0)), Rr @ _Rx(90) @ _Rz(0),
                      p + d * size * 0.1)
            out["closed eyes"].append(arc)
            continue
        if style == "owl":
            out["eye whites"].append(_xf(ellipsoid(size * 1.0, size * 1.0, size * 0.35), _look_rot(d), p))
            out["eyes"].append(sphere(size * 0.6).at(*(p + d * size * 0.12)))
            out["eye shine"].append(sphere(size * 0.17).at(*(p + d * size * 0.62 + np.array([-0.3, 0.35, 0]) *
                                                             size * 0.5)))
            continue
        # the bead sits deep in the head (about half of it shows), so it reads as an eye, not a stuck-on ball
        ec = p - d * size * 0.52
        out["eyes"].append(sphere(size).at(*ec))
        # catch-light: a small flat disc on the front-upper part of the bead (between the eye axis and the head's
        # forward axis), so it stays on the eye from three-quarter views instead of floating at its rim
        fwd = np.array([0.0, 0.0, 1.0])
        n = d * 0.45 + fwd * 0.55 + np.array([0.0, 0.62, 0.0])
        n = n / np.linalg.norm(n)
        out["eye shine"].append(_xf(ellipsoid(size * 0.2, size * 0.2, size * 0.07), _look_rot(n),
                                    ec + n * size * 0.975))
    return out


def _on_surface(head_fn, centre, d, R):
    d = np.asarray(d, np.float64)
    d = d / np.linalg.norm(d)
    lo, hi = 0.0, 3.0 * R
    for _ in range(26):
        m = (lo + hi) / 2
        if head_fn(np.asarray(centre + d * m, F32)[None])[0] < 0:
            lo = m
        else:
            hi = m
    return centre + d * lo


def _is_cute(sp):
    """``cute=`` given above the house default (1.15): bigger head and eyes plus blush and a small smile."""
    return float(sp.get("cute", 1.0)) > 1.2


def _cute_face(head_fn, centre, R, eye_dirs, eye_size, mouth=None, fwd=(0.0, 0.0, 1.0), blush_color="#f39aa6",
               smile_color="#3a2220", smile=True):
    """Blush cheeks under the eyes and a small 'w' smile (two little arcs) at ``mouth`` (a point on the face)."""
    fwd = np.asarray(fwd, np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    blush, arcs = [], []
    for d in eye_dirs:
        d = np.asarray(d, np.float64)
        d = d / np.linalg.norm(d)
        side = np.array([d[0], 0.0, 0.0])
        n = d + side * 0.55 - np.array([0.0, 0.62, 0.0])            # below the eye and a little outward
        p = _on_surface(head_fn, centre, n, R)
        nn = (p - centre) / max(np.linalg.norm(p - centre), 1e-9)
        r = min(eye_size * 0.8, R * 0.15)
        blush.append(_xf(ellipsoid(r * 1.25, r * 0.8, r * 0.18), _look_rot(nn), p - nn * r * 0.05))
    if smile and mouth is not None:
        s = eye_size * 0.3
        Rr = _look_rot(fwd)
        side = Rr @ np.array([1.0, 0.0, 0.0])
        for k in (1, -1):
            arc = intersect(torus(s, s * 0.26), plane((0, 0, -1), 0.0))
            arcs.append(_xf(arc, Rr @ _Rx(90), np.asarray(mouth, np.float64) + side * k * s * 0.95 + fwd * s * 0.1))
    mats = {"blush": Material(color=blush_color, roughness=0.85, specular=0.05, subsurface=0.3, shadow=False),
            "smile": Material(color=smile_color, roughness=0.5, specular=0.2)}
    return {"blush": blush, "smile": arcs}, mats


def _eye_mats(color="#16100e"):
    return {"eyes": Material(color=color, roughness=0.3, specular=0.45),
            "eye shine": Material(color="#ffffff", emissive=0.6, roughness=0.5, specular=0.0, shadow=False),
            "eye whites": Material(color="#f6f0d8", roughness=0.3, specular=0.5),
            "closed eyes": Material(color="#2a1a16", roughness=0.5, specular=0.2)}


def _amat(color, style="clay", kind="fur"):
    if isinstance(color, Material):
        return color
    if style == "plush" or kind == "plush":
        return Material(color=color, roughness=0.97, specular=0.05, sheen=0.9, subsurface=0.25)
    if kind == "hard":
        return Material(color=color, roughness=0.32, specular=0.6)
    if kind == "glass":
        return Material(color=color, roughness=0.1, specular=0.8, glass=0.6, shadow=0.35)
    if style == "plastic":
        return Material(color=color, roughness=0.32, specular=0.6, subsurface=0.12)
    if style == "vinyl":
        return Material(color=color, roughness=0.42, specular=0.5, clearcoat=0.25, subsurface=0.18)
    return Material(color=color, roughness=0.72, specular=0.22, subsurface=0.32, sheen=0.2)


def _surface_point(shape, origin, direction, rmax):
    """First point where a ray from inside ``shape`` (from ``origin`` along ``direction``) leaves it."""
    d = np.asarray(direction, np.float64)
    d = d / max(np.linalg.norm(d), 1e-12)
    o = np.asarray(origin, np.float64)
    lo, hi = 0.0, float(rmax)
    for _ in range(26):
        m = (lo + hi) / 2
        q = (o + d * m).astype(F32)
        if shape._d(q[None, 0].copy(), q[None, 1].copy(), q[None, 2].copy())[0] < 0:
            lo = m
        else:
            hi = m
    return o + d * lo


def _ground_shift(pts):
    return -min(float(p[1]) for p in pts)


def _solve_sit(rig, P, sp):
    """Find the body pitch at which the front paws and the haunches both touch the ground."""
    rh = rig.rh
    P0 = float(P.get("pelvis", {}).get("pitch", 0.0))

    def gap(pitch):
        Q = _copy.deepcopy(P)
        Q["pelvis"] = dict(Q.get("pelvis", {}), pitch=pitch)
        if "tail" in Q:
            Q["tail"] = dict(Q["tail"], up=Q["tail"].get("up", 0.0) + pitch - P0)
        if "neck" in Q:                                  # keep the head level whatever pitch the body settles at
            Q["neck"] = dict(Q["neck"], up=Q["neck"].get("up", 0.0) - 0.85 * (pitch - P0))
        sw = _folded_hind(sp)[0]
        for sd in ("l", "r"):
            Q[f"f0_{sd}"] = dict(Q.get(f"f0_{sd}", {}), swing=-pitch + 4)
            Q[f"h0_{sd}"] = dict(Q.get(f"h0_{sd}", {}), swing=sw - pitch)
        J = rig.fk(Q)
        Rp, pp = J["pelvis"]
        rump = min((pp + Rp @ np.array([0, -rh, -rh * 0.3]))[1], (pp + Rp @ np.array([0, -rh * 0.6, -rh]))[1])
        paws = min(J[f"f3_{sd}"][1][1] - rig.paw_h for sd in ("l", "r"))
        return paws - rump, Q
    lo, hi = 5.0, 80.0
    g_lo, _ = gap(lo)
    g_hi, _ = gap(hi)
    if g_lo * g_hi > 0:
        return gap(hi if abs(g_hi) < abs(g_lo) else lo)[1]
    for _ in range(20):
        m = (lo + hi) / 2
        g, _ = gap(m)
        if (g > 0) == (g_lo > 0):
            lo, g_lo = m, g
        else:
            hi = m
    return gap((lo + hi) / 2)[1]


def _build_quad(sp, P, detail="normal", style="clay"):
    rig = _QuadRig(sp)
    L = rig.L
    if P.get("sit_solve"):
        P = _solve_sit(rig, P, sp)
    J = rig.fk(P)
    J0 = rig.fk({})
    rc, rh = rig.rc, rig.rh
    rb = sp["rb"] * L
    wx = sp["wx"]
    hr = sp["head_r"] * L * sp.get("cute", 1.15)
    ax = sp["head_ax"]
    # ---- ground contact
    pts = []
    for sd in ("l", "r"):
        for leg, rr in (("f", sp["leg_rf"]), ("h", sp["leg_rh"])):
            R3, p3 = J[f"{leg}3_{sd}"]
            pts.append(p3 - np.array([0, rig.paw_h, 0]))
            for i in (1, 2):
                pts.append(J[f"{leg}{i}_{sd}"][1] - np.array([0, rr[min(i, 2)] * L, 0]))
    for nm, r in (("pelvis", rh), ("spine", rb), ("chest", rc)):
        pts.append(J[nm][1] - J[nm][0] @ np.array([0, r, 0]) - np.array([0, 0.0, 0]))
        pts.append(J[nm][1] + np.array([0, -r, 0]))
    Rh_, ph_ = J["head"]
    pts.append(ph_ - np.array([0, hr * 0.9, 0]))
    if sp["tail"].get("type") in ("thick",):
        for i in range(1, rig.tail_n + 1):
            pts.append(J[f"tail{i}"][1] - np.array([0, sp["tail"]["r"][0] * L * (1 - i / (rig.tail_n + 1)), 0]))
    ground = P.get("ground", "lowest")
    if ground == "paws":
        pts = pts[:4]
    dy = _ground_shift(pts) + float(P.get("lift", 0.0)) * (sp["leg_h"] + sp["rh"]) * L
    # a tail that would dip through the floor (sitting, lying, begging) is lifted until it just rests on it
    t_ = sp["tail"]
    if t_.get("type") not in ("thick", "none", None) and t_.get("len", 0) > 0:
        tr = max(t_.get("r", (0.05, 0.03))) * L * (1.1 if t_.get("type") in ("bushy", "hair") else 1.0)
        P = dict(P)
        for _ in range(16):
            low = min(J[f"tail{i}"][1][1] for i in range(1, rig.tail_n + 1)) - tr + dy
            if low >= -0.004 * L:
                break
            P["tail"] = dict(P.get("tail", {}), up=float(P.get("tail", {}).get("up", 0.0)) + 6.0)
            J = rig.fk(P)
    J = {k: (R, p + np.array([0, dy, 0])) for k, (R, p) in J.items()}
    acc, order = {}, []

    def add(name, shapes, mat):
        shapes = [s for s in shapes if s is not None]
        if not shapes:
            return
        if name not in acc:
            acc[name] = [[], mat]
            order.append(name)
        acc[name][0].extend(shapes)
    Rp, pp = J["pelvis"]
    Rs, ps = J["spine"]
    Rc, pc = J["chest"]
    Rh, ph = J["head"]
    hips = _xf(ellipsoid(rh * wx, rh, rh * 1.15), Rp, pp + Rp @ np.array([0, 0.02 * L, -0.02 * L]))
    belly = _xf(ellipsoid(rb * wx, rb, 0.42 * L), Rs, ps + Rs @ np.array([0, -sp["sag"] * L, 0]))
    chest = _xf(ellipsoid(rc * wx, rc, rc * 1.15), Rc, pc + Rc @ np.array([0, 0, 0.04 * L]))
    body = [hips, belly, chest]
    # neck: chain through the neck joints into the head
    nchain = [J["neck"][1] - Rc @ np.array([0, rc * 0.3, rc * 0.2])]
    for i in range(1, rig.neck_n):
        nchain.append(J[f"neck{i}"][1])
    nchain.append(ph)
    n0, n1 = sp["neck_r"]
    nr = list(np.linspace(n0, n1, len(nchain)) * L)
    necks = _chain(nchain, nr)
    # head (head frame)
    cc = np.array([0, hr * 0.12, hr * 0.12])
    cran = ellipsoid(hr * ax[0], hr * ax[1], hr * ax[2]).at(*cc)
    sl = sp["snout"] * L
    sd_ = math.radians(sp["snout_drop"])
    s0 = cc + np.array([0, -hr * 0.2, hr * ax[2] * 0.45])
    s1 = s0 + np.array([0, -math.sin(sd_) * sl, math.cos(sd_) * sl])
    sr0, sr1 = sp["snout_r"][0] * L, sp["snout_r"][1] * L
    head_parts = [cran]
    if sl > 1e-4:
        sn = round_cone(s0, s1, sr0, sr1)
        if sp.get("snout_w", 1.0) != 1.0:
            W_ = np.diag([sp["snout_w"], 1.0, 1.0])
            sn = _xf(round_cone(np.linalg.solve(W_, s0), np.linalg.solve(W_, s1), sr0, sr1), W_, (0, 0, 0))
        head_parts.append(sn)
    if sp.get("cheeks"):
        for s in (1, -1):
            head_parts.append(sphere(sp["cheeks"] * L).at(s * hr * 0.42, -hr * 0.22, hr * 0.62))
    tipdir = (s1 - s0) / max(np.linalg.norm(s1 - s0), 1e-9) if sl > 1e-4 else np.array([0, 0, 1.0])
    nose_tip = (s1 + tipdir * sr1 * 0.75) if sl > 1e-4 else cc + np.array([0, -hr * 0.1, hr * ax[2] * 0.95])
    if sp.get("pads"):                     # whisker pads: the two round puffs under a cat's nose
        pr = sp["pads"] * L
        S = _surface_point(smooth_union(*head_parts, k=0.08 * L), cc, nose_tip - cc, 3 * hr)
        for s in (1, -1):
            head_parts.append(sphere(pr).at(*(S + np.array([s * pr * 0.78, -pr * 0.8, -pr * 0.6]))))
        head_parts.append(sphere(pr * 0.62).at(*(S + np.array([0, -pr * 1.75, -pr * 0.95]))))
    head_local = smooth_union(*head_parts, k=0.08 * L)
    # a short muzzle can end inside the skull: put the nose back on the surface of the face
    S = _surface_point(head_local, cc, nose_tip - cc, 3 * hr)
    if np.linalg.norm(S - cc) - np.linalg.norm(nose_tip - cc) > 0.3 * sr1:
        dn = (S - cc) / max(np.linalg.norm(S - cc), 1e-9)
        nose_tip = S - dn * sr1 * 0.22
    head = _xf(head_local, Rh, ph)
    neck_head = smooth_union(necks[0], *(necks[1:] or []), head, k=0.07 * L) if necks else head
    # one continuous skin from the rump to the nose: the neck melts into the chest instead of meeting it in a seam
    fur = [smooth_union(smooth_union(*body, k=0.12 * L), neck_head, k=0.1 * L)]
    # legs
    legs_shapes = []
    for sd in ("l", "r"):
        for leg, rr in (("f", sp["leg_rf"]), ("h", sp["leg_rh"])):
            pts_ = [J[f"{leg}{i}_{sd}"][1] for i in range(4)]
            radii = [rr[0] * L * (1.1 if leg == "f" else 1.3), rr[0] * L * (1.0 if leg == "f" else 1.1), rr[1] * L,
                     rr[2] * L]
            if leg == "f" and sp.get("front_scale", 1.0) != 1.0:
                radii = [r * sp["front_scale"] for r in radii]
            legs_shapes += _chain(pts_, radii)
    fur_body = smooth_union(fur[0], _S.union(*[legs_shapes[i] for i in range(0, len(legs_shapes), 3)]), k=0.07 * L)
    fur = [fur_body] + [legs_shapes[i] for i in range(len(legs_shapes)) if i % 3]
    # paws / hooves
    hooves, claws = [], []
    for sd in ("l", "r"):
        for leg in ("f", "h"):
            R3, p3 = J[f"{leg}3_{sd}"]
            ps_ = sp["paw_size"] * L * (sp.get("front_scale", 1.0) if leg == "f" else 1.0)
            ph_y = rig.paw_h * (sp.get("front_scale", 1.0) if leg == "f" else 1.0)
            if sp["paw"] == "hoof":
                hooves.append(_xf(cylinder(ps_ * 0.72, ph_y * 1.2, round=ps_ * 0.2), R3, p3 + R3 @ np.array([0, -ph_y * 0.45, ps_ * 0.1])))
            elif sp["paw"] == "pad":
                fur.append(_xf(intersect(cylinder(ps_, ph_y * 1.6, round=ps_ * 0.3), plane((0, -1, 0), ph_y)), R3,
                               p3 + R3 @ np.array([0, -ph_y * 0.2, ps_ * 0.12])))
                for k in (-1, 0, 1):
                    claws.append(sphere(ps_ * 0.2).at(*(p3 + R3 @ np.array([k * ps_ * 0.5, -ph_y * 0.75, ps_ * 0.9]))))
            else:
                paw = intersect(ellipsoid(ps_ * 0.78, ph_y * 0.9, ps_ * 1.1), plane((0, -1, 0), ph_y * 0.85))
                fur.append(_xf(paw, R3, p3 + R3 @ np.array([0, -ph_y * 0.15, ps_ * 0.35])))
                if sp["paw"] == "claw":
                    for k in (-1, 0, 1):
                        claws.append(_xf(round_cone((0, 0, 0), (0, -0.02 * L, ps_ * 0.5), ps_ * 0.18, ps_ * 0.03), R3,
                                         p3 + R3 @ np.array([k * ps_ * 0.45, -ph_y * 0.5, ps_ * 1.0])))
    # tail
    t = sp["tail"]
    ttype = t.get("type", "thin")
    tail_pts = [J[f"tail{i}"][1] for i in range(rig.tail_n + 1)]
    r0, r1 = (t.get("r", (0.05, 0.03))[0] * L, t.get("r", (0.05, 0.03))[1] * L)
    tail_part = []
    mane_shapes = []
    if ttype in ("thin", "thick", "tuft"):
        rr = list(np.linspace(r0, r1, len(tail_pts)))
        tail_part = _chain(tail_pts, rr)
        if ttype == "tuft":
            mane_shapes.append(_xf(ellipsoid(r0 * 1.6, r0 * 2.6, r0 * 1.6), _axis_rot((0, 1, 0), tail_pts[-1] - tail_pts[-2]),
                                   tail_pts[-1]))
        if ttype == "thick":
            fur[0] = smooth_union(fur[0], tail_part[0], k=0.08 * L)
            tail_part = tail_part[1:]
    elif ttype == "bushy":
        rr = [r0 * 0.6, r1 * 0.95, r1 * 1.1, r1 * 0.9, r1 * 0.3]
        tail_part = [smooth_union(*_chain(tail_pts, rr[:len(tail_pts)]), k=0.03 * L)]
    elif ttype == "hair":
        rr = [r0 * 0.7, r1, r1 * 1.05, r1 * 0.9, r1 * 0.45]
        mane_shapes.append(smooth_union(*_chain(tail_pts, rr[:len(tail_pts)]), k=0.03 * L))
    elif ttype in ("stub", "puff"):
        tail_part = [sphere(r0).at(*tail_pts[1] if ttype == "stub" else tail_pts[0])]
    elif ttype == "curl":
        Rt, pt = J["tail0"]
        sp_ = []
        for i in range(10):
            a = i / 9 * 2.4 * math.pi
            sp_.append(pt + Rt @ np.array([0.035 * L * math.cos(a), 0.035 * L * math.sin(a) + 0.01 * L * i,
                                           -0.012 * L * i - 0.02 * L]))
        tail_part = _chain(sp_, list(np.linspace(r0, r1, len(sp_))))
    fur += tail_part
    # ears
    ears, inner = _quad_ears(sp, hr, ax, cc, L)
    fur += [_xf(e, Rh, ph) for e in ears]
    # eyes
    az, el = math.radians(sp["eye_az"]), math.radians(sp["eye_el"])
    dirs = [np.array([s * math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)]) for s in (1, -1)]
    eye_c = cc
    closed = P.get("eyes") == "closed"
    E = _bead_eyes(lambda q: head_local._d(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy()), eye_c, hr, dirs,
                   sp["eye"] * L * sp.get("cute", 1.15) * 1.12, closed=closed)
    # nose
    extras_parts = []
    if sp.get("nose_size", 0) > 0:
        ns = sp["nose_size"] * L
        extras_parts.append(("nose", [_xf(ellipsoid(ns * 1.15, ns * 0.8, ns * 0.85), Rh, ph + Rh @ (nose_tip + np.array([0, sr1 * 0.25, -ns * 0.2])))],
                             Material(color=sp["nose"], roughness=0.28, specular=0.7)))
    if P.get("tongue") or sp.get("tongue"):
        tg = round_cone(nose_tip + np.array([0, -sr1 * 0.9, -sr1 * 0.6]), nose_tip + np.array([0, -sr1 * 1.9, -sr1 * 0.4]),
                        sr1 * 0.35, sr1 * 0.4)
        extras_parts.append(("tongue", [_xf(tg, Rh, ph)], Material(color="#ee7a8a", roughness=0.4, specular=0.5,
                                                                   subsurface=0.3)))
    # extras
    ctx = dict(J=J, L=L, Rh=Rh, ph=ph, hr=hr, ax=ax, cc=cc, s0=s0, s1=s1, sr0=sr0, sr1=sr1, rc=rc, rh=rh, rig=rig,
               nose_tip=nose_tip, fur=fur, tail_pts=tail_pts, mane=mane_shapes)
    for ex in sp.get("extras", ()):
        out = _quad_extra(ex, sp, ctx, detail)
        for nm, shapes, mat in out:
            if nm == "fur":
                fur += shapes
            elif nm == "mane":
                mane_shapes += shapes
            else:
                extras_parts.append((nm, shapes, mat))
    if sp.get("mane") and sp.get("mane") != "none":
        mane_shapes += _quad_mane(sp, ctx)
    # ---- markings follow the body through the nearest bone
    bones = _paint_bones(rig, J, J0, sp)
    head_info = dict(snout_z=float(np.linalg.norm(s1 - cc) + sr1) if sl > 1e-4 else hr,
                     eyes=[J0["head"][1] + J0["head"][0] @ (cc + d * hr * 0.95) for d in dirs])
    paint = _BodyPaint(sp["fur"], bones, list(sp.get("marks", [])), scale=1.0, head=head_info)
    fm = _amat(paint, style)
    add("fur", fur, fm)
    if mane_shapes:
        mc = sp.get("mane") or sp.get("tail_tip") or _shade(sp["fur"], 0.6)
        if mc == "rainbow":
            mc = _Rainbow(J["neck"][1], J["head"][1])
        elif not isinstance(mc, str) or mc == "none":
            mc = _shade(sp["fur"], 0.6)
        add("mane", mane_shapes, _amat(mc, style))
    if inner:
        add("inner ears", [_xf(e, Rh, ph) for e in inner], _amat(sp.get("ear_color") and _shade(sp["ear_color"], 1.0)
                                                                  or sp["inner"], style))
    if sp.get("ear_color"):
        pass
    if hooves:
        add("hooves", hooves, _amat(sp["hoof"], style, "hard"))
    if claws:
        add("claws", claws, _amat(sp.get("horn", "#efe6d2"), style, "hard"))
    em = _eye_mats(sp.get("eye_color", "#16100e"))
    for nm in ("eye whites", "eyes", "eye shine", "closed eyes"):
        add(nm, [_xf(s, Rh, ph) for s in E[nm]], em[nm])
    if _is_cute(sp):
        hf = lambda q: head_local._d(q[:, 0].copy(), q[:, 1].copy(), q[:, 2].copy())   # noqa: E731
        o = nose_tip - tipdir * sr1 * 1.5                         # on the muzzle axis, aim just under the nose
        md = nose_tip - np.array([0, sr1 * 0.3 + sp.get("nose_size", 0.04) * L * 0.9, 0]) - o
        mouth = _on_surface(hf, o, md, 3 * hr)
        CF, cm = _cute_face(hf, eye_c, hr, dirs, sp["eye"] * L * sp["cute"] * 1.12, mouth=mouth, fwd=md,
                            smile=not P.get("tongue") and not sp.get("tongue"))
        for nm in ("blush", "smile"):
            add(nm, [_xf(s, Rh, ph) for s in CF[nm]], cm[nm])
    for nm, shapes, mat in extras_parts:
        add(nm, shapes, mat)
    A = dict(head=ph + Rh @ cc, nose=ph + Rh @ nose_tip, mouth=ph + Rh @ (nose_tip - np.array([0, sr1, sr1 * 0.5])),
             back=pc + Rc @ np.array([0, rc, 0]) * 0.5 + (pp + Rp @ np.array([0, rh, 0])) * 0.5,
             tail_tip=tail_pts[-1], chest=pc + Rc @ np.array([0, 0, rc * 1.1]),
             head_top=ph + Rh @ (cc + np.array([0, hr * ax[1], 0])), feet=np.array([pp[0], 0.0, (pp[2] + pc[2]) / 2]))
    for sd in ("l", "r"):
        A[f"paw_f{sd}"] = J[f"f3_{sd}"][1]
        A[f"paw_h{sd}"] = J[f"h3_{sd}"][1]
    parts = [(nm, acc[nm][0], acc[nm][1]) for nm in order]
    return parts, A


class _Rainbow(_Checker):
    """Rainbow bands along a mane / tail (unicorns)."""

    def __init__(self, a, b):
        self.a = np.asarray(a, F32)
        self.b = np.asarray(b, F32)
        self.cols = np.array([_lin(c) for c in ("#ff6b8a", "#ffb36b", "#ffe46b", "#7be08a", "#6bc4ff", "#a98bff")], F32)

    def __call__(self, P, N=None, foot=None):
        e = self.b - self.a
        t = ((P - self.a) @ e) / max(float(e @ e), 1e-9)
        u = (t * 3.0 + P[:, 1] * 2.0) % 1.0 * len(self.cols)
        i0 = np.floor(u).astype(int) % len(self.cols)
        i1 = (i0 + 1) % len(self.cols)
        f = (u - np.floor(u))[:, None]
        f = np.clip((f - 0.75) * 4, 0, 1)
        return self.cols[i0] * (1 - f) + self.cols[i1] * f


def _paint_bones(rig, J, J0, sp):
    L = rig.L
    out = []

    def bone(a, b, kind, rad, f0=0.0, f1=1.0, s0=0.0):
        (Ra, pa), (Rb, pb) = J[a], J[b]
        (Ra0, pa0), _ = J0[a], J0[b]
        out.append((pa, pb, Ra, pa0, Ra0, kind, s0, rad, f0, f1))
    bone("pelvis", "spine", "torso", sp["rh"] * L, 0.0, 0.5, 0.0)
    bone("spine", "chest", "torso", sp["rb"] * L, 0.5, 1.0, L / 2)
    nn = ["neck"] + [f"neck{i}" for i in range(1, rig.neck_n)] + ["head"]
    for i in range(len(nn) - 1):
        bone(nn[i], nn[i + 1], "neck", sp["neck_r"][0] * L, i / (len(nn) - 1), (i + 1) / (len(nn) - 1), L + i * 0.2)
    # head: a short bone from the head joint along the snout
    Rh, ph = J["head"]
    Rh0, ph0 = J0["head"]
    tip = np.array([0, -sp["head_r"] * L * 0.2, sp["head_r"] * L + sp["snout"] * L])
    out.append((ph, ph + Rh @ tip, Rh, ph0, Rh0, "head", L * 1.6, sp["head_r"] * L, 0.0, 1.0))
    for i in range(rig.tail_n):
        bone(f"tail{i}", f"tail{i + 1}", "tail", sp["tail"].get("r", (0.05, 0.03))[0] * L, i / rig.tail_n,
             (i + 1) / rig.tail_n, -0.3 * L - i * sp["tail"].get("len", 0.5) * L / rig.tail_n)
    for sd in ("l", "r"):
        for leg, rr in (("f", sp["leg_rf"]), ("h", sp["leg_rh"])):
            for i in range(3):
                bone(f"{leg}{i}_{sd}", f"{leg}{i + 1}_{sd}", "leg_" + leg, rr[min(i, 2)] * L, i / 3, (i + 1) / 3,
                     i * 0.25 * L)
    return out


def _quad_ears(sp, hr, ax, cc, L):
    e = sp["ears"]
    kind, size = e.get("type", "flop"), e.get("size", 0.2) * L
    out, inner = [], []
    if kind in ("none", None):
        return out, inner
    for s in (1, -1):
        if kind in ("point", "big_point"):
            spread = 28 if kind == "point" else 55
            base = cc + np.array([s * hr * ax[0] * 0.55, hr * ax[1] * 0.72, -hr * 0.05])
            d = np.array([s * math.sin(math.radians(spread)), math.cos(math.radians(spread)), 0.08])
            Rr = _axis_rot((0, 1, 0), d)
            Fl = Rr @ np.diag([1.0, 1.0, 0.42])
            out.append(_xf(round_cone((0, 0, 0), (0, size, 0), size * 0.5, size * 0.07), Fl, base))
            inner.append(_xf(round_cone((0, size * 0.1, 0), (0, size * 0.82, 0), size * 0.32, size * 0.05),
                             Fl, base + Rr @ np.array([0, 0, size * 0.12])))
        elif kind == "flop":
            # hung from the top corner of the skull, flaring out a little so they read in front views too
            base = cc + np.array([s * hr * ax[0] * 0.8, hr * ax[1] * 0.62, -hr * 0.12])
            Re = _Rz(s * 16) @ _Rx(-6)
            out.append(_xf(ellipsoid(size * 0.24, size * 0.82, size * 0.5), Re,
                           base + np.array([s * size * 0.2, 0, 0]) + Re @ np.array([0, -size * 0.6, 0])))
        elif kind == "side":
            base = cc + np.array([s * hr * ax[0] * 0.92, hr * 0.2, -hr * 0.15])
            out.append(_xf(ellipsoid(size * 0.7, size * 0.18, size * 0.36), _Rz(-s * 22), base + np.array([s * size * 0.55, 0, 0])))
            inner.append(_xf(ellipsoid(size * 0.55, size * 0.08, size * 0.24), _Rz(-s * 22),
                             base + np.array([s * size * 0.6, -size * 0.03, size * 0.12])))
        elif kind in ("round", "small", "fluffy", "side_round", "mouse"):
            ang = {"round": 38, "small": 30, "fluffy": 52, "side_round": 82, "mouse": 40}[kind]
            d = np.array([s * math.sin(math.radians(ang)), math.cos(math.radians(ang)), -0.12])
            d /= np.linalg.norm(d)
            base = cc + d * hr * np.array(ax) * 0.92
            Rr = _look_rot(np.array([0, 0, 1.0]) * 0.6 + d * 0.4, d)
            th = {"round": 0.5, "small": 1.0, "fluffy": 0.55, "side_round": 0.4, "mouse": 0.25}[kind]
            out.append(_xf(ellipsoid(size, size, size * th), Rr, base + d * size * 0.45))
            if kind != "small":
                inner.append(_xf(ellipsoid(size * 0.68, size * 0.68, size * th * 0.6), Rr,
                                 base + d * size * 0.45 + Rr @ np.array([0, 0, size * th * 0.55])))
        elif kind == "long":
            base = cc + np.array([s * hr * 0.32, hr * ax[1] * 0.75, -hr * 0.1])
            d = np.array([s * 0.22, 1.0, -0.15])
            Rr = _axis_rot((0, 1, 0), d)
            Fl = Rr @ np.diag([1.0, 1.0, 0.4])
            out.append(_xf(round_cone((0, 0, 0), (0, size, 0), size * 0.15, size * 0.18), Fl, base))
            inner.append(_xf(round_cone((0, size * 0.1, 0), (0, size * 0.9, 0), size * 0.09, size * 0.11), Fl,
                             base + Rr @ np.array([0, 0, size * 0.07])))
        elif kind == "elephant":
            base = cc + np.array([s * hr * ax[0] * 0.75, hr * 0.05, -hr * 0.35])
            out.append(_xf(ellipsoid(size * 0.15, size * 0.8, size * 0.64), _Ry(s * 22) @ _Rz(s * 10),
                           base + np.array([s * size * 0.3, -size * 0.12, -size * 0.12])))
    return out, inner


def _quad_mane(sp, c):
    J, L = c["J"], c["L"]
    names = ["neck"] + [f"neck{i}" for i in range(1, c["rig"].neck_n)] + ["head"]
    pts = []
    for nm in names:
        R, p = J[nm]
        pts.append((R, p))
    out = []
    style = sp.get("mane_style", "flow")
    n0 = sp["neck_r"][0] * L
    for i in range(len(pts) - 1):
        (Ra, pa), (Rb, pb) = pts[i], pts[i + 1]
        for k in range(3):
            f = k / 3
            p = pa * (1 - f) + pb * f
            R = Ra
            r = n0 * (1 - 0.4 * (i + f) / max(len(pts) - 1, 1))
            if style == "brush":
                out.append(_xf(ellipsoid(r * 0.25, r * 0.55, r * 0.5), R, p + R @ np.array([0, r * 1.0, -r * 0.1])))
            else:
                out.append(_xf(ellipsoid(r * 0.45, r * 0.75, r * 0.6), R @ _Rz(12), p + R @ np.array([r * 0.25, r * 0.85, -r * 0.2])))
    Rh, ph = J["head"]
    out.append(_xf(ellipsoid(c["hr"] * 0.35, c["hr"] * 0.3, c["hr"] * 0.45), Rh @ _Rx(-30),
                   ph + Rh @ (c["cc"] + np.array([0, c["hr"] * 0.95, c["hr"] * 0.35]))))
    return [smooth_union(*out, k=0.03 * L)]


def _quad_extra(ex, sp, c, detail):
    J, L, Rh, ph, hr, cc = c["J"], c["L"], c["Rh"], c["ph"], c["hr"], c["cc"]

    def H(shapes):
        return [_xf(s, Rh, ph) for s in shapes]
    horn = _amat(sp.get("horn", "#efe6d2"), sp.get("material", "clay"), "hard")
    if ex == "poodle":
        Rp, pp = J["pelvis"]
        Rc, pc = J["chest"]
        out = [sphere(hr * 0.55).at(*(cc + np.array([0, hr * 0.85, -hr * 0.1])))]
        sh = H(out)
        sh.append(_xf(ellipsoid(c["rc"] * 1.25, c["rc"] * 1.2, c["rc"] * 1.25), Rc, pc + Rc @ np.array([0, 0.03 * L, 0.0])))
        sh.append(sphere(0.09 * L).at(*c["tail_pts"][-1]))
        for sd in ("l", "r"):
            for leg in ("f", "h"):
                sh.append(sphere(0.07 * L).at(*J[f"{leg}2_{sd}"][1]))
        return [("fur", sh, None)]
    if ex == "pig_snout":
        d = (c["s1"] - c["s0"]) / max(np.linalg.norm(c["s1"] - c["s0"]), 1e-9)
        disc = _xf(cylinder(c["sr1"] * 0.95, 0.04 * L, round=0.015 * L), _axis_rot((0, 1, 0), d), c["s1"] + d * c["sr1"] * 0.6)
        nost = [_xf(ellipsoid(0.02 * L, 0.03 * L, 0.012 * L), _look_rot(d), c["s1"] + d * (c["sr1"] * 0.6 + 0.022 * L)
                    + np.array([s * 0.045 * L, 0, 0])) for s in (1, -1)]
        return [("snout", H([disc]), _amat(_shade(sp["fur"], 0.9), sp.get("material", "clay"))),
                ("nostrils", H(nost), Material(color="#6a3a3e", roughness=0.5))]
    if ex == "wool":
        Rp, pp = J["pelvis"]
        Rs, ps = J["spine"]
        Rc, pc = J["chest"]
        # a fleece cloud: tufts scattered evenly over a capsule round the trunk (no rows, no seams)
        balls = []
        n = 26 if detail == "low" else (44 if detail == "normal" else 64)
        r = max(c["rh"], sp["rb"] * L, c["rc"]) * 0.95
        ax_ = pc - pp
        half = np.linalg.norm(ax_) / 2
        ax_ = ax_ / max(2 * half, 1e-9)
        mid = (pp + pc) / 2 + np.array([0, r * 0.08, 0])
        Rw = _axis_rot((0, 0, 1), ax_)
        rng = np.random.default_rng(7)
        for d in _fib_dirs(n):
            if d[1] < -0.62:
                continue
            q = np.array([d[0] * r * 0.86, d[1] * r * 0.88, d[2] * (half + r * 0.48)])
            balls.append(sphere(r * (0.36 + 0.08 * rng.random())).at(*(mid + Rw @ q)))
        balls.append(_xf(ellipsoid(r * 0.8, r * 0.82, half + r * 0.45), Rw, mid))
        for q in ((0, 0.8, -0.3), (0.28, 0.66, -0.42), (-0.28, 0.66, -0.42), (0, 0.6, -0.62)):   # a woolly topknot
            balls.append(sphere(hr * 0.3).at(*(ph + Rh @ (cc + np.array(q) * hr))))
        return [("wool", [smooth_union(*balls, k=0.022 * L)], _amat(sp["fur"], sp.get("material", "clay"), "plush"))]
    if ex == "goat_horns":
        out = []
        for s in (1, -1):
            b = cc + np.array([s * hr * 0.3, hr * 0.8, 0.0])
            out += _chain([b, b + np.array([s * 0.03, 0.12, -0.08]) * L, b + np.array([s * 0.06, 0.14, -0.22]) * L],
                          [0.045 * L, 0.03 * L, 0.01 * L])
        return [("horns", H(out), horn)]
    if ex == "beard":
        tipz = c["s1"] - np.array([0, c["sr1"] * 1.1, c["sr1"] * 0.8])
        return [("beard", H([round_cone(tipz, tipz + np.array([0, -0.12 * L, -0.02 * L]), 0.04 * L, 0.015 * L)]),
                 _amat(_shade(sp["fur"], 0.85), sp.get("material", "clay")))]
    if ex == "cow_horns":
        out = []
        for s in (1, -1):
            b = cc + np.array([s * hr * 0.55, hr * 0.72, -0.02 * L])
            out += _chain([b, b + np.array([s * 0.1, 0.03, 0.0]) * L, b + np.array([s * 0.14, 0.1, 0.03]) * L],
                          [0.035 * L, 0.025 * L, 0.01 * L])
        return [("horns", H(out), horn)]
    if ex == "muzzle_pink":
        sp["marks"] = list(sp.get("marks", [])) + [dict(kind="muzzle", color="#f0b6b8", start=0.45, top=0.9)]
        return []
    if ex == "muzzle_dark":
        sp["marks"] = list(sp.get("marks", [])) + [dict(kind="muzzle", color="#2a2628", start=0.6, top=0.9)]
        return []
    if ex == "antlers":
        out = []
        for s in (1, -1):
            b = cc + np.array([s * hr * 0.35, hr * 0.85, -hr * 0.1])
            m = b + np.array([s * 0.12, 0.2, -0.06]) * L
            t = m + np.array([s * 0.05, 0.18, 0.05]) * L
            out += _chain([b, m, t], [0.025 * L, 0.02 * L, 0.01 * L])
            out += _chain([m * 0.5 + b * 0.5, m * 0.5 + b * 0.5 + np.array([s * 0.02, 0.12, 0.08]) * L], [0.018 * L, 0.008 * L])
            out += _chain([m, m + np.array([s * 0.12, 0.08, 0.0]) * L], [0.018 * L, 0.008 * L])
        return [("antlers", H(out), _amat("#d8c09a", sp.get("material", "clay"), "hard"))]
    if ex == "ossicones":
        out, knobs = [], []
        for s in (1, -1):
            b = cc + np.array([s * hr * 0.3, hr * 0.85, -hr * 0.2])
            t = b + np.array([s * 0.02, 0.1, -0.02]) * L
            out.append(round_cone(b, t, 0.025 * L, 0.02 * L))
            knobs.append(sphere(0.03 * L).at(*t))
        return [("fur", H(out), None), ("ossicone tips", H(knobs), _amat("#5a3a2a", sp.get("material", "clay")))]
    if ex == "trunk":
        d0 = c["s0"] + np.array([0, -hr * 0.05, hr * 0.25])
        pts = [d0, d0 + np.array([0, -0.18, 0.12]) * L, d0 + np.array([0, -0.4, 0.14]) * L,
               d0 + np.array([0, -0.58, 0.1]) * L, d0 + np.array([0, -0.68, 0.17]) * L]
        tr = _chain(pts, [0.12 * L, 0.095 * L, 0.075 * L, 0.06 * L, 0.055 * L])
        return [("fur", H([smooth_union(*tr, k=0.03 * L)]), None)]
    if ex == "tusks":
        out = []
        for s in (1, -1):
            b = c["s0"] + np.array([s * 0.1 * L, -0.12 * L, hr * 0.15])
            out += _chain([b, b + np.array([s * 0.03, -0.12, 0.12]) * L, b + np.array([s * 0.04, -0.1, 0.25]) * L],
                          [0.035 * L, 0.03 * L, 0.012 * L])
        return [("tusks", H(out), _amat("#f6f0e0", sp.get("material", "clay"), "hard"))]
    if ex == "rhino_horns":
        tip = c["s1"]
        a = round_cone(tip + np.array([0, c["sr1"] * 0.6, -0.02 * L]), tip + np.array([0, c["sr1"] * 0.6 + 0.24 * L, 0.04 * L]),
                       0.07 * L, 0.01 * L)
        b2 = round_cone(tip + np.array([0, c["sr1"] * 0.9, -0.2 * L]), tip + np.array([0, c["sr1"] * 0.9 + 0.1 * L, -0.18 * L]),
                        0.05 * L, 0.01 * L)
        return [("horns", H([a, b2]), _amat("#b8ae9e", sp.get("material", "clay"), "hard"))]
    if ex == "nostrils":
        tip = c["s1"]
        return [("nostrils", H([sphere(0.03 * L).at(*(tip + np.array([s * 0.09 * L, c["sr1"] * 0.85, c["sr1"] * 0.4])))
                                for s in (1, -1)]), Material(color="#4a3a48", roughness=0.5))]
    if ex == "lion_mane":
        # a big fluffy ruff all round the head: tufts on a sphere, the face poking out in front
        n = 18 if detail == "low" else (30 if detail == "normal" else 44)
        ctr = cc + np.array([0, -0.05 * hr, -0.3 * hr])
        balls = [ellipsoid(hr * 1.2, hr * 1.3, hr * 0.95).at(*ctr)]
        for d in _fib_dirs(n):
            if d[2] > 0.55:
                continue
            q = ctr + d * np.array([hr * 1.22, hr * 1.32, hr * 0.98])
            balls.append(sphere(hr * (0.36 if d[1] > -0.5 else 0.42)).at(*q))
        ruff = smooth_union(*balls, k=0.05 * L)
        ruff = intersect(ruff, plane((0, 0, 1), cc[2] + 0.12 * hr))
        mane = H([ruff])
        Rc, pc = J["chest"]
        mane.append(_xf(ellipsoid(c["rc"] * 1.05, c["rc"] * 1.15, c["rc"] * 0.95), Rc, pc + Rc @ np.array([0, c["rc"] * 0.45, c["rc"] * 0.35])))
        return [("mane", mane, None)]
    if ex == "teeth":
        out = []
        for s in (1, -1):
            for k in range(4):
                f = 0.2 + 0.2 * k
                p = c["s0"] * (1 - f) + c["s1"] * f + np.array([s * c["sr1"] * 0.75, -c["sr1"] * 0.75, 0])
                out.append(round_cone(p, p + np.array([0, -0.035 * L, 0]), 0.016 * L, 0.003 * L))
        return [("teeth", H(out), Material(color="#f6f2e6", roughness=0.4, specular=0.5))]
    if ex == "frill":
        fr = _xf(ellipsoid(hr * 1.45, hr * 1.25, 0.05 * L), _Rx(-35), cc + np.array([0, hr * 0.5, -hr * 0.55]))
        horns = [round_cone(cc + np.array([s * hr * 0.4, hr * 0.55, hr * 0.3]), cc + np.array([s * hr * 0.55, hr * 1.1, hr * 1.4]),
                            0.06 * L, 0.008 * L) for s in (1, -1)]
        horns.append(round_cone(c["s1"] + np.array([0, c["sr1"] * 0.6, -0.04 * L]), c["s1"] + np.array([0, c["sr1"] + 0.12 * L, 0.02 * L]),
                                0.045 * L, 0.006 * L))
        edge = [sphere(0.045 * L).at(*(cc + np.array([math.cos(a) * hr * 1.42, hr * 0.5 + math.sin(a) * hr * 1.0,
                                                      -hr * 0.55 - math.sin(a) * hr * 0.7])))
                for a in np.linspace(0.1, math.pi - 0.1, 9)]
        return [("fur", H([fr]), None), ("horns", H(horns + edge), horn)]
    if ex == "plates":
        out = []
        names = ["chest", "spine", "pelvis", "tail1", "tail2"]
        k = 0
        for i, nm in enumerate(names):
            R, p = J[nm]
            nxt = J[names[i + 1]][1] if i + 1 < len(names) else J["tail3"][1]
            for f in (0.0, 0.5):
                q = p * (1 - f) + nxt * f
                hgt = (0.16 + 0.1 * math.sin(math.pi * (i + f) / len(names))) * L
                top = q + R @ np.array([0, sp["rh"] * L * (0.85 if i < 3 else 0.6), 0])
                sgn = 1 if k % 2 else -1
                k += 1
                tri = _S.prism([(-hgt * 0.55, 0), (0, hgt), (hgt * 0.55, 0), (0, -hgt * 0.2)], depth=0.03 * L)
                out.append(_xf(tri, R @ _Ry(90) @ _Rz(0), top + R @ np.array([sgn * 0.03 * L, 0, 0])))
        spikes = []
        R4, p4 = J["tail4"]
        for s in (1, -1):
            for kk in (0, 1):
                b = c["tail_pts"][-1] + R4 @ np.array([0, 0.02 * L, 0.06 * L + kk * 0.1 * L])
                spikes.append(round_cone(b, b + R4 @ np.array([s * 0.14 * L, 0.1 * L, 0.0]), 0.025 * L, 0.004 * L))
        return [("plates", out + spikes, _amat(sp.get("horn", "#d8643a"), sp.get("material", "clay")))]
    if ex == "wings":
        Rc, pc = J["chest"]
        out, mem = [], []
        for s in (1, -1):
            sh = pc + Rc @ np.array([s * c["rc"] * 0.5, c["rc"] * 0.8, -0.05 * L])
            el = sh + Rc @ np.array([s * 0.35, 0.3, -0.15]) * L
            tip = el + Rc @ np.array([s * 0.45, 0.2, -0.25]) * L
            out += _chain([sh, el, tip], [0.035 * L, 0.025 * L, 0.01 * L])
            for f in (0.45, 0.8):
                fpt = el + Rc @ np.array([s * 0.35 * f, -0.25 - 0.2 * f, -0.35]) * L
                out += _chain([el, fpt], [0.02 * L, 0.008 * L])
            # membrane: thin plate through shoulder, elbow, tip and the finger ends
            loc = [sh, el, tip, el + Rc @ np.array([s * 0.33, -0.5, -0.32]) * L, sh + Rc @ np.array([0, -0.05, -0.35]) * L]
            n = np.cross(loc[1] - loc[0], loc[2] - loc[0])
            n /= max(np.linalg.norm(n), 1e-9)
            Rm = _look_rot(n, loc[1] - loc[0])
            pts2 = [tuple((Rm.T @ (q - sh))[:2]) for q in loc]
            mem.append(_xf(_S.prism(pts2, depth=0.012 * L), Rm, sh))
        return [("wing bones", out, _amat(sp["fur"], sp.get("material", "clay"))),
                ("wings", mem, _amat(sp.get("wing", "#2f7a54"), sp.get("material", "clay")))]
    if ex == "dragon_horns":
        out = []
        for s in (1, -1):
            b = cc + np.array([s * hr * 0.4, hr * 0.7, -hr * 0.2])
            out += _chain([b, b + np.array([s * 0.05, 0.1, -0.12]) * L, b + np.array([s * 0.07, 0.12, -0.26]) * L],
                          [0.04 * L, 0.025 * L, 0.006 * L])
        return [("horns", H(out), horn)]
    if ex == "spikes":
        out = []
        for nm in ["neck", "chest", "spine", "pelvis", "tail1", "tail2", "tail3"]:
            if nm not in J:
                continue
            R, p = J[nm]
            r = (sp["neck_r"][0] if nm == "neck" else sp["rc"]) * L
            b = p + R @ np.array([0, r * 0.92, 0])
            out.append(round_cone(b, b + R @ np.array([0, 0.09 * L, -0.03 * L]), 0.04 * L, 0.004 * L))
        return [("spikes", out, horn)]
    if ex == "spade":
        R4, p4 = J["tail4"]
        tri = _S.prism([(0, 0.12 * L), (-0.09 * L, -0.02 * L), (0, -0.06 * L), (0.09 * L, -0.02 * L)], depth=0.02 * L)
        return [("spikes", [_xf(tri, R4 @ _Rx(90), p4 + R4 @ np.array([0, 0, -0.06 * L]))], horn)]
    if ex == "unicorn_horn":
        b = cc + np.array([0, hr * 0.75, hr * 0.45])
        h = cone(0.05 * L, 0.004 * L, 0.3 * L).twist(900)
        return [("horn", H([_xf(h, _Rx(28), b + np.array([0, 0.12 * L, 0.06 * L]))]),
                 Material(color="#f6d98a", roughness=0.25, specular=0.8, metallic=0.4))]
    if ex == "koala_nose":
        tip = cc + np.array([0, -hr * 0.12, hr * sp["head_ax"][2] * 0.95])
        return [("nose", H([ellipsoid(0.09 * L, 0.12 * L, 0.07 * L).at(*tip)]),
                 Material(color="#2a2426", roughness=0.25, specular=0.7))]
    return []
