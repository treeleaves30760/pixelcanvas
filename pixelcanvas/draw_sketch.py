"""Sketch without a reference: construction on the sketch layer, every element an id that can be moved, scaled,
rotated or replaced without redrawing the rest.

Elements (records in ``draw["sketch"]``, coordinates in 0-1000 units):
- ``ellipse`` / ``box``: a box (and a rotation);   ``line`` / ``curve``: points (a curve is a smooth spline);
- ``figure``: a posable mannequin from joint points, proportioned from a head count (2.5 chibi ... 7.5 adult);
- ``head``: head construction in a box: cranium ball, jaw, centre line and eye / nose / mouth lines turned by yaw
  and pitch;
- ``horizon``: the horizon line and fans of perspective lines from up to two vanishing points.

The line tool snaps to these elements and can trace them, so the sketch is what the line art is built on.
"""
from __future__ import annotations

import math
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import coords
from .coords import UsageError
from .draw_core import (catmull_rom, dstate, ellipse_pts, find_record, next_id, next_order, resample, u2px)

SKETCH_RGB = (40, 70, 160)
GUIDE_RGB = (90, 120, 200)

JOINTS = ["head", "chin", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhip", "rhip",
          "lknee", "rknee", "lankle", "rankle"]
JOINT_ALIAS = {"neck": "chin", "lsh": "lshoulder", "rsh": "rshoulder", "lel": "lelbow", "rel": "relbow",
               "lwr": "lwrist", "rwr": "rwrist", "lkn": "lknee", "rkn": "rknee", "lank": "lankle",
               "rank": "rankle", "lhand": "lwrist", "rhand": "rwrist", "lfoot": "lankle", "rfoot": "rankle"}
SHORT = {"head": "head", "chin": "chin", "lshoulder": "lsh", "rshoulder": "rsh", "lelbow": "lel", "relbow": "rel",
         "lwrist": "lwr", "rwrist": "rwr", "lhip": "lhip", "rhip": "rhip", "lknee": "lkn", "rknee": "rkn",
         "lankle": "lank", "rankle": "rank"}

# joint positions as (x from the centre line, y from the top of the head), in figure heights; "l" = picture left
_CHIBI = {"head": (0, 0.20), "chin": (0, 0.40), "shoulder": (0.105, 0.45), "elbow": (0.15, 0.565),
          "wrist": (0.165, 0.665), "hip": (0.065, 0.69), "knee": (0.065, 0.83), "ankle": (0.065, 0.965)}
_ADULT = {"head": (0, 0.0667), "chin": (0, 0.1333), "shoulder": (0.125, 0.178), "elbow": (0.147, 0.38),
          "wrist": (0.153, 0.52), "hip": (0.08, 0.533), "knee": (0.073, 0.747), "ankle": (0.067, 0.96)}
POSES = ("stand", "walk", "hold", "wave", "sit", "bust")


def norm_joint(name: str) -> str:
    k = str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    k = k.replace("left", "l").replace("right", "r")
    k = JOINT_ALIAS.get(k, k)
    if k not in JOINTS:
        raise UsageError(f"unknown joint {name!r}; joints: {', '.join(SHORT[j] for j in JOINTS)} "
                         "(l = left side of the picture)", "pixelcanvas sketch joint s1 rwr 620,540")
    return k


# ============================================================================================ figure

def figure_joints(box, heads: float, pose: str, W: int = 1000, H: int = 1000) -> Dict[str, List[float]]:
    """Joint positions (units) for a figure filling ``box`` (x0,y0,x1,y1), ``heads`` tall, in a pose. The maths runs
    in a square space (x scaled by H/W) so proportions hold on any canvas shape."""
    k = H / float(W)
    box = [box[0] / k, box[1], box[2] / k, box[3]]
    # interpolate in head height (1 / heads), so the figure measures the head count it was asked for
    t = float(np.clip((0.4 - 1.0 / heads) / (0.4 - 0.1333), -0.2, 1.15))
    fr = {k: (_CHIBI[k][0] + t * (_ADULT[k][0] - _CHIBI[k][0]), _CHIBI[k][1] + t * (_ADULT[k][1] - _CHIBI[k][1]))
          for k in _CHIBI}
    x0, y0, x1, y1 = box
    Hf = (y1 - y0) * 0.98
    if pose == "bust":
        Hf = (y1 - y0) / max(0.2, fr["chin"][1] * 2.6)
    cx = (x0 + x1) / 2
    top = y0 + (y1 - y0) * 0.01
    J: Dict[str, List[float]] = {}
    for base, (fx, fy) in fr.items():
        if base in ("head", "chin"):
            J[base] = [cx, top + fy * Hf]
        else:
            J["l" + base] = [cx - fx * Hf, top + fy * Hf]
            J["r" + base] = [cx + fx * Hf, top + fy * Hf]
    # pose offsets are measured in the figure's own limb lengths, so a chibi and an adult pose alike
    Lu = math.hypot(J["relbow"][0] - J["rshoulder"][0], J["relbow"][1] - J["rshoulder"][1])
    Lf = math.hypot(J["rwrist"][0] - J["relbow"][0], J["rwrist"][1] - J["relbow"][1])
    Lt = math.hypot(J["rknee"][0] - J["rhip"][0], J["rknee"][1] - J["rhip"][1])
    Ls = math.hypot(J["rankle"][0] - J["rknee"][0], J["rankle"][1] - J["rknee"][1])

    def at(base, dx, dy):
        return [base[0] + dx, base[1] + dy]
    if pose == "walk":
        J["lknee"] = at(J["lhip"], -0.25 * Lt, 0.95 * Lt)
        J["lankle"] = at(J["lknee"], -0.3 * Ls, 0.93 * Ls)
        J["rknee"] = at(J["rhip"], 0.08 * Lt, 0.99 * Lt)
        J["rankle"] = at(J["rknee"], 0.35 * Ls, 0.9 * Ls)
        J["lelbow"] = at(J["lshoulder"], 0.1 * Lu, 0.99 * Lu)
        J["lwrist"] = at(J["lelbow"], 0.35 * Lf, 0.9 * Lf)
        J["relbow"] = at(J["rshoulder"], 0.25 * Lu, 0.95 * Lu)
        J["rwrist"] = at(J["relbow"], -0.15 * Lf, 0.97 * Lf)
    elif pose == "hold":
        # the right hand (picture right) holds something upright at chest height beside the body
        J["relbow"] = at(J["rshoulder"], 0.35 * Lu, 0.93 * Lu)
        J["rwrist"] = at(J["relbow"], 0.45 * Lf, -0.55 * Lf)
    elif pose == "wave":
        J["relbow"] = at(J["rshoulder"], 0.75 * Lu, -0.6 * Lu)
        J["rwrist"] = at(J["relbow"], 0.25 * Lf, -0.95 * Lf)
    elif pose == "sit":
        for s_ in "lr":
            sg = -1 if s_ == "l" else 1
            J[s_ + "knee"] = at(J[s_ + "hip"], sg * 0.45 * Lt, 0.35 * Lt)
            J[s_ + "ankle"] = at(J[s_ + "knee"], sg * 0.1 * Ls, 0.99 * Ls)
    elif pose == "bust":
        bottom = y1
        for jn in ("lelbow", "relbow", "lwrist", "rwrist", "lhip", "rhip", "lknee", "rknee", "lankle", "rankle"):
            J[jn][1] = min(J[jn][1], bottom)
    return {n: [round(v[0] * k, 1), round(v[1], 1)] for n, v in J.items()}


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def figure_shapes(p, el) -> List[Tuple[np.ndarray, bool, float]]:
    """Polylines (px) of the mannequin: head ball, neck, rib cage, pelvis, tapered limbs, joints, hands, feet."""
    J = {k: u2px(p, [v])[0] for k, v in el["joints"].items()}
    out: List[Tuple[np.ndarray, bool, float]] = []
    hh = _dist(J["head"], J["chin"])           # half a head
    hr = max(2.0, hh)
    from .draw_core import dstate
    has_head = any(e["kind"] == "head" and e.get("subject") == el.get("subject") for e in dstate(p)["sketch"])
    if not has_head:          # a head construction of the same subject replaces the mannequin's head ball
        out.append((ellipse_pts(J["head"][0], J["head"][1], 0.78 * hr, hr), True, 1.0))
        out.append((np.array([[J["head"][0], J["head"][1] - hr], [J["chin"][0], J["chin"][1]]]), False, 0.5))
        out.append((np.array([[J["head"][0] - 0.78 * hr, J["head"][1] + 0.15 * hr],
                              [J["head"][0] + 0.78 * hr, J["head"][1] + 0.15 * hr]]), False, 0.5))
    shc = (J["lshoulder"] + J["rshoulder"]) / 2
    hpc = (J["lhip"] + J["rhip"]) / 2
    out.append((np.array([J["chin"], shc]), False, 1.0))
    bust = el.get("pose") == "bust"
    # rib cage: an egg from the shoulders to ~60 % of the way to the hips
    sw = _dist(J["lshoulder"], J["rshoulder"]) / 2
    torso = hpc - shc
    tl = math.hypot(*torso)
    if tl > 1:
        ang = math.degrees(math.atan2(torso[1], torso[0])) - 90
        rc = shc + torso * 0.33
        out.append((ellipse_pts(rc[0], rc[1], sw * 0.82, tl * 0.36, ang), True, 1.0))
        if not bust:
            pc = hpc - torso * 0.05
            hw = _dist(J["lhip"], J["rhip"]) / 2
            out.append((ellipse_pts(pc[0], pc[1], hw * 1.35, tl * 0.2, ang), True, 1.0))
            out.append((np.array([shc, hpc]), False, 0.5))
    out.append((np.array([J["lshoulder"], J["rshoulder"]]), False, 0.8))
    if not bust:
        out.append((np.array([J["lhip"], J["rhip"]]), False, 0.8))
    limbs = [("lshoulder", "lelbow", 0.36), ("lelbow", "lwrist", 0.28), ("rshoulder", "relbow", 0.36),
             ("relbow", "rwrist", 0.28)]
    if not bust:
        limbs += [("lhip", "lknee", 0.5), ("lknee", "lankle", 0.38), ("rhip", "rknee", 0.5), ("rknee", "rankle", 0.38)]
    for a, b, wf in limbs:
        A, B = J[a], J[b]
        if bust and (A[1] >= p.H * 0.995 or _dist(A, B) < 2):
            continue
        seg = _dist(A, B)
        wa = min(wf * hr, 0.42 * seg)
        out.append((_capsule(A, B, wa, 0.8 * wa), True, 1.0))
    for j in ("lshoulder", "rshoulder", "lelbow", "relbow", "lknee", "rknee"):
        if bust and j in ("lknee", "rknee"):
            continue
        out.append((ellipse_pts(J[j][0], J[j][1], 0.16 * hr, 0.16 * hr, n=24), True, 0.6))
    for s in "lr":
        if bust:
            break
        el_, wr = J[s + "elbow"], J[s + "wrist"]
        d = wr - el_
        a = math.degrees(math.atan2(d[1], d[0]))
        u = d / (np.linalg.norm(d) + 1e-9)
        hs = min(0.42 * hr, 0.42 * _dist(el_, wr))          # hands follow the forearm, not the head
        hc = wr + u * 0.8 * hs
        out.append((ellipse_pts(hc[0], hc[1], hs, 0.72 * hs, a), True, 0.9))
        an = J[s + "ankle"]
        side = -1 if s == "l" else 1
        fs = min(hr, 0.75 * _dist(J[s + "knee"], an))       # feet follow the shin
        foot = np.array([[an[0] - 0.2 * fs, an[1] - 0.1 * fs], [an[0] + side * 0.75 * fs, an[1] + 0.25 * fs],
                         [an[0] + side * 0.6 * fs, an[1] + 0.4 * fs], [an[0] - 0.25 * fs, an[1] + 0.35 * fs]])
        out.append((np.vstack([foot, foot[:1]]), False, 0.9))
    return out


def _capsule(A, B, wa, wb) -> np.ndarray:
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = B - A
    L = math.hypot(*d)
    if L < 1e-6:
        return ellipse_pts(A[0], A[1], wa, wa, n=24)
    u = d / L
    n = np.array([-u[1], u[0]])
    ang = math.atan2(u[1], u[0])
    pts = [A + n * wa * 0.5, B + n * wb * 0.5]
    t = np.linspace(ang + math.pi / 2, ang - math.pi / 2, 9)
    cap_b = [B + 0.5 * wb * np.array([math.cos(a), math.sin(a)]) for a in t]
    t2 = np.linspace(ang - math.pi / 2, ang - 3 * math.pi / 2, 9)
    cap_a = [A + 0.5 * wa * np.array([math.cos(a), math.sin(a)]) for a in t2]
    return np.vstack([pts[0], pts[1], *cap_b, B - n * wb * 0.5, A - n * wa * 0.5, *cap_a, pts[0]])


def figure_report(p, el) -> List[str]:
    """Measured proportions in plain sentences, each with the joint to move when something is off."""
    from . import draw_pose
    if draw_pose.is_rigged(el):
        # a posed 3D figure: its proportions are built in, and foreshortening is not a fault
        rig = el["rig"]
        return [f"posed in 3D: {el.get('pose')}, {float(rig.get('heads', 6.5)):g} heads, {float(rig.get('height', 1.6)):g} m "
                f"(change it with pixelcanvas pose {el.get('subject')} --preset / --set)"]
    J = {k: np.array(v, float) for k, v in el["joints"].items()}
    # measure in pixels so non-square canvases are fair
    Jp = {k: u2px(p, [v])[0] for k, v in J.items()}
    head_h = 2 * _dist(Jp["head"], Jp["chin"])
    top = Jp["head"][1] - head_h / 2
    bust = el.get("pose") == "bust"
    out = []
    if not bust:
        bottom = max(Jp["lankle"][1], Jp["rankle"][1]) + 0.15 * head_h
        heads = (bottom - top) / max(1e-6, head_h)
        if 2.0 <= heads <= 4.2:
            rng = "chibi range 2-4"
        elif 5.8 <= heads <= 8.5:
            rng = "adult range 6-8"
        elif 4.2 < heads < 5.8:
            rng = "between chibi (2-4) and adult (6-8): child proportions"
        else:
            rng = "outside the usual 2-8"
        out.append(f"figure: {heads:.1f} heads tall ({rng}); the head is 1/{heads:.1f} of the height")
    else:
        out.append(f"bust: head {head_h * 1000 / p.H:.0f} units tall")
    sw = _dist(Jp["lshoulder"], Jp["rshoulder"])
    chibi = float(el.get("heads", 7)) < 4.2
    lo, hi = (0.35, 1.0) if chibi else (1.0, 2.4)
    out.append(f"shoulders {sw / max(head_h, 1e-6):.1f} head-heights wide"
               + (f" (narrow: {lo:g}-{hi:g} is usual{' for chibi' if chibi else ''}; move lsh / rsh apart)"
                  if sw < lo * head_h else f" (very wide: {lo:g}-{hi:g} is usual)" if sw > hi * head_h else ""))
    for a, b, c, nm in (("lshoulder", "lelbow", "lwrist", "arm"), ("lhip", "lknee", "lankle", "leg")):
        if bust and nm == "leg":
            continue
        La = _dist(Jp[a], Jp[b]) + _dist(Jp[b], Jp[c])
        rb = "r" + b[1:]
        rc = "r" + c[1:]
        Lb = _dist(Jp["r" + a[1:]], Jp[rb]) + _dist(Jp[rb], Jp[rc])
        if min(La, Lb) > 0 and max(La, Lb) / min(La, Lb) > 1.18 and not bust:
            short, longer = ("left", "right") if La < Lb else ("right", "left")
            j = (c if short == "left" else rc)
            out.append(f"the {short} {nm} is {1 - min(La, Lb) / max(La, Lb):.0%} shorter than the {longer} one; "
                       f"move {SHORT[j]} further from {SHORT[b if short == 'left' else rb]}")
        up = _dist(Jp[a], Jp[b])
        lo = _dist(Jp[b], Jp[c])
        if lo > 0 and not bust and not (0.7 <= up / lo <= 1.5):
            out.append(f"the left {nm}'s two segments are {up / lo:.1f} : 1 (about 1 : 1 is usual); move {SHORT[b]}")
    return out


# ============================================================================================ head construction

def head_shapes(p, el) -> List[Tuple[np.ndarray, bool, float]]:
    x0, y0, x1, y1 = u2px(p, [el["box"][:2], el["box"][2:]]).reshape(-1)
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    yaw = math.radians(float(el.get("yaw", 0)))
    pitch = math.radians(float(el.get("pitch", 0)))
    rx, ry = 0.5 * w, 0.4 * h
    by = y0 + ry
    out = [(ellipse_pts(cx, by, rx, ry), True, 1.0)]
    # jaw: from the sides of the ball down to the chin, the chin shifted by yaw
    chin = np.array([cx + math.sin(yaw) * 0.35 * rx, y1])
    # the far side's jaw comes inward (yaw > 0 turns the face to the picture's right)
    left = np.array([x0 + (0.08 + max(0, -math.sin(yaw)) * 0.12) * w, by + 0.15 * h])
    right = np.array([x1 - (0.08 + max(0, math.sin(yaw)) * 0.12) * w, by + 0.15 * h])
    shape = str(el.get("shape", "oval"))
    # (x from the side toward the centre as a share of w, y below the jaw start as a share of the jaw height)
    prof = {"round": [(0.02, 0.35), (0.16, 0.78), (0.33, 0.97)],
            "oval": [(0.05, 0.38), (0.2, 0.8), (0.36, 0.97)],
            "sharp": [(0.1, 0.45), (0.3, 0.86)]}.get(shape, [(0.05, 0.38), (0.2, 0.8), (0.36, 0.97)])
    jh = y1 - (by + 0.15 * h)
    shift = chin[0] - cx

    def side(start, sgn):
        pts = [start]
        for fx, fy in prof:
            x = start[0] + sgn * fx * w if sgn > 0 else start[0] - fx * w
            pts.append([x + shift * fy ** 2, by + 0.15 * h + fy * jh])
        return pts
    lp = side(left, 1)
    rp = side(right, -1)
    full = lp + [list(chin)] + rp[::-1]
    if shape == "sharp":
        jl = catmull_rom(lp + [list(chin)], spacing=2)
        jr = catmull_rom(rp + [list(chin)], spacing=2)
    else:
        # one smooth curve through the chin: a soft jaw, no point at the chin
        J = catmull_rom(full, spacing=2)
        k = int(np.argmin(np.hypot(J[:, 0] - chin[0], J[:, 1] - chin[1])))
        jl, jr = J[:k + 1], J[k:][::-1]
    out.append((jl, False, 1.0))
    out.append((jr, False, 1.0))
    # centre line: an arc over the ball, bent by yaw
    cxl = cx + math.sin(yaw) * rx * 0.92
    t = np.linspace(-math.pi / 2, math.pi / 2, 30)
    xs = cxl + math.sin(yaw) * 0.12 * rx * np.cos(t)
    ys = by + ry * np.sin(t) * 1.0
    cl = np.stack([xs, ys], 1)
    out.append((np.vstack([cl, chin]), False, 0.6))
    # eye, nose, mouth lines: arcs across the face, moved by pitch, kept inside the face outline
    fe = 0.62 if el.get("chibi") else 0.56
    for frac, wf in ((fe, 1.0), (fe + 0.13, 0.4), (fe + 0.24, 0.35)):
        yy = y0 + (frac - 0.12 * math.sin(pitch)) * h
        # half-width of the face at this height: the ball above the jaw start, the jaw below it
        dy = (yy - by) / ry
        half_face = rx * math.sqrt(max(0.0, 1 - min(1.0, dy * dy))) if yy < by + 0.15 * h else             rx * max(0.15, 1 - (yy - by - 0.15 * h) / max(1e-6, y1 - by - 0.15 * h)) * 0.85
        half = half_face * 0.92 * wf
        lo, hi = cx - half_face * 0.95, cx + half_face * 0.95
        a0 = max(lo, cxl - half)
        a1 = min(hi, cxl + half)
        sag = 0.03 * h * math.cos(pitch) * (1 if frac < 0.6 else 0.6)
        arc = np.array([[a0, yy - sag * 0.2], [cxl, yy + sag * (1 + 2 * math.sin(pitch))], [a1, yy - sag * 0.2]])
        out.append((catmull_rom(arc, spacing=2), False, 0.6))
    return out


def head_guides(p, el) -> Dict[str, Tuple[float, float]]:
    """Where the face features go (units): eye centres, nose, mouth, from the construction."""
    x0, y0, x1, y1 = el["box"]
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    yaw = math.radians(float(el.get("yaw", 0)))
    pitch = math.radians(float(el.get("pitch", 0)))
    cxl = cx + math.sin(yaw) * 0.5 * w * 0.92
    fe = 0.62 if el.get("chibi") else 0.56          # chibi eyes sit lower and wider apart
    ey = y0 + (fe - 0.12 * math.sin(pitch)) * h
    near, far = (0.24 * w, 0.24 * w) if el.get("chibi") else (0.22 * w, 0.22 * w)
    lx = cxl - near * (1 - 0.35 * max(0, -math.sin(yaw)))      # the far eye sits closer to the centre line
    rx = cxl + far * (1 - 0.35 * max(0, math.sin(yaw)))
    return {"left eye": (lx, ey), "right eye": (rx, ey), "nose": (cxl, y0 + (fe + 0.15 - 0.12 * math.sin(pitch)) * h),
            "mouth": (cxl, y0 + (fe + 0.25 - 0.12 * math.sin(pitch)) * h)}


# ============================================================================================ all elements

def shapes(p, el) -> List[Tuple[np.ndarray, bool, float]]:
    """(polyline px, closed, weight factor) list for any element."""
    k = el["kind"]
    if k in ("block", "prop") or (k == "figure" and el.get("rig")) or (k == "horizon" and el.get("camera")):
        from . import draw_pose          # the 3D layout: camera, posed figures, blocks, props
        out = draw_pose.element_shapes(p, el)
        if out is not None:
            return out
    if k in ("ellipse", "box"):
        x0, y0, x1, y1 = u2px(p, [el["box"][:2], el["box"][2:]]).reshape(-1)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if k == "ellipse":
            return [(ellipse_pts(cx, cy, (x1 - x0) / 2, (y1 - y0) / 2, el.get("rot", 0.0), n=96), True, 1.0)]
        r = math.radians(el.get("rot", 0.0))
        c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]) - [cx, cy]
        R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        return [(resample(c @ R.T + [cx, cy], 3.0), False, 1.0)]
    if k == "line":
        return [(resample(u2px(p, el["pts"]), 3.0), False, 1.0)]
    if k == "curve":
        return [(catmull_rom(u2px(p, el["pts"]), closed=bool(el.get("closed")), spacing=2.0), False, 1.0)]
    if k == "figure":
        return figure_shapes(p, el)
    if k == "head":
        return head_shapes(p, el)
    if k == "horizon":
        y = el["y"] * p.H / 1000
        out = [(np.array([[0, y], [p.W - 0.01, y]]), False, 0.9)]
        for vp in el.get("vps", []):
            vx, vy = vp[0] * p.W / 1000, vp[1] * p.H / 1000
            for ang in np.linspace(math.radians(195), math.radians(345), 9) if False else \
                    np.linspace(0, 2 * math.pi, 17)[:-1]:
                L = 2.0 * max(p.W, p.H)
                ex, ey = vx + L * math.cos(ang), vy + L * math.sin(ang)
                if abs(math.sin(ang)) < 0.02:
                    continue
                seg = _clip_seg(vx, vy, ex, ey, p.W, p.H)
                if seg is not None:
                    out.append((seg, False, 0.45))
        return out
    return []


def _clip_seg(x0, y0, x1, y1, W, H):
    n = 400
    t = np.linspace(0, 1, n)
    xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if ok.sum() < 2:
        return None
    return np.stack([xs[ok], ys[ok]], 1)[[0, -1]]


def sketch_width(p) -> float:
    return max(1.5, max(p.W, p.H) / 520.0)


def render_ops(p, el) -> list:
    from .ops import Stroke
    w0 = sketch_width(p)
    ops = []
    guide = el["kind"] in ("horizon",)
    for poly, closed, wf in shapes(p, el):
        pts = np.asarray(poly, np.float32)
        if closed and len(pts) > 2:
            pts = np.vstack([pts, pts[:1]])
        if len(pts) < 2:
            continue
        L = float(np.hypot(*np.diff(pts, axis=0).T).sum())
        w = max(1.2, w0 * (0.8 + 0.4 * wf))
        if L < 4 * w:
            continue
        light = wf < 0.4                 # a ground grid, a dashed line hidden behind the figure's own parts
        ops.append(Stroke(pts=pts, width=w, color=GUIDE_RGB if (guide or wf < 0.7) else SKETCH_RGB,
                          opacity=0.35 if light else 0.7 if (guide or wf < 0.7) else 1.0, brush="pencil",
                          seed=zlib.crc32(el["id"].encode()) % 9973))
    return ops


def all_polylines(p) -> List[Tuple[str, np.ndarray, bool]]:
    """Every sketch polyline (densely sampled px) for snapping and tracing: (element id, pts, closed)."""
    d = dstate(p)
    out = []
    for el in d["sketch"]:
        for poly, closed, wf in shapes(p, el):
            if wf < 0.7 or el["kind"] == "horizon":          # guides (eye line, centre line) do not attract ink
                continue
            pts = resample(np.vstack([poly, poly[:1]]) if closed else poly, 1.5)
            out.append((el["id"], pts, closed))
    return out


# ============================================================================================ edits

def element_points(el) -> List[List[float]]:
    k = el["kind"]
    if k in ("ellipse", "box", "head"):
        return [el["box"][:2], el["box"][2:]]
    if k in ("line", "curve"):
        return el["pts"]
    if k == "figure":
        return list(el["joints"].values())
    if k == "horizon":
        return [[500, el["y"]]] + list(el.get("vps", []))
    if k in ("block", "prop"):
        return [el.get("at") or [500, 500]]
    return []


def centre(el) -> Tuple[float, float]:
    pts = np.asarray(element_points(el), float)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def transform(el, fn) -> None:
    """Apply fn(points [N,2] units) -> points to an element's geometry."""
    k = el["kind"]
    if k in ("block", "prop") or (k == "horizon" and el.get("camera")):
        what = {"block": f"pixelcanvas block {el.get('name')} --at x,y", "prop": "pixelcanvas prop <name> --hand l",
                "horizon": "pixelcanvas camera --horizon 600"}[k]
        raise UsageError(f"{el['id']} is part of the 3D layout: it changes with its own command", what)
    if k in ("ellipse", "box", "head"):
        b = np.array([el["box"][:2], el["box"][2:]], float)
        nb = fn(b)
        x0, x1 = sorted((nb[0, 0], nb[1, 0]))
        y0, y1 = sorted((nb[0, 1], nb[1, 1]))
        el["box"] = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
    elif k in ("line", "curve"):
        el["pts"] = [[round(a, 1), round(b, 1)] for a, b in fn(np.asarray(el["pts"], float))]
    elif k == "figure":
        names = list(el["joints"])
        pts = fn(np.asarray([el["joints"][n] for n in names], float))
        el["joints"] = {n: [round(a, 1), round(b, 1)] for n, (a, b) in zip(names, pts)}
    elif k == "horizon":
        pts = fn(np.asarray([[500.0, el["y"]]] + [list(v) for v in el.get("vps", [])], float))
        el["y"] = round(float(pts[0, 1]), 1)
        el["vps"] = [[round(a, 1), round(b, 1)] for a, b in pts[1:]]


def move(el, dx, dy):
    transform(el, lambda P: P + [dx, dy])


def scale(el, f):
    cx, cy = centre(el)
    transform(el, lambda P: (P - [cx, cy]) * f + [cx, cy])


def rotate(el, deg):
    if el["kind"] in ("ellipse", "box"):
        el["rot"] = round(float(el.get("rot", 0.0)) + deg, 1)
        return
    if el["kind"] == "head":
        raise UsageError("a head turns with --yaw / --pitch, not rotate", f"pixelcanvas sketch head --redo {el['id']} --yaw 25")
    cx, cy = centre(el)
    r = math.radians(deg)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    transform(el, lambda P: (P - [cx, cy]) @ R.T + [cx, cy])


def new_element(p, kind: str, **kw) -> dict:
    d = dstate(p)
    el = {"id": next_id(d, "s"), "kind": kind, "n": next_order(d), **kw}
    d["sketch"].append(el)
    return el


def find(p, rid: str) -> dict:
    d = dstate(p)
    k = str(rid).strip().lower()
    for el in d["sketch"]:
        if el["kind"] == "figure" and el.get("subject") == k:
            return el
    return find_record(d["sketch"], rid, "sketch element", "pixelcanvas sketch list")


def describe(el) -> str:
    k = el["kind"]
    if k in ("ellipse", "box"):
        return f"{el['id']} {k} {coords.fmt_box(el['box'])}" + (f" rot {el['rot']:g}" if el.get("rot") else "")
    if k in ("line", "curve"):
        return f"{el['id']} {k} {len(el['pts'])} points " + " ".join(coords.fmt_point(q) for q in el["pts"][:6])
    if k == "figure":
        return f"{el['id']} figure of {el.get('subject')} ({el.get('pose', 'stand')}, {el.get('heads', 7)} heads)"
    if k == "head":
        return f"{el['id']} head {coords.fmt_box(el['box'])} yaw {el.get('yaw', 0):g} pitch {el.get('pitch', 0):g}"
    if k == "horizon":
        if el.get("camera"):
            c = el["camera"]
            return f"{el['id']} camera: horizon {el['y']:g}, eye height {c.get('eye', 1.5):g} m, fov " \
                   f"{c.get('fov', 50):g}, tilt {c.get('tilt', 0):g}"
        return f"{el['id']} horizon y {el['y']:g}" + "".join(f" vp {coords.fmt_point(v)}" for v in el.get("vps", []))
    if k == "block":
        return f"{el['id']} block {el['name']}: {el['type']} {'x'.join(f'{v:g}' for v in el['size'])} m"
    if k == "prop":
        return f"{el['id']} prop {el['name']}: {el['type']} in {el.get('subject')}'s hand {el.get('hand')}"
    return el["id"]


def label_points(p, el) -> List[Tuple[str, Tuple[float, float]]]:
    """Where to put tags in a view (px): the id, and for a figure every joint's short name."""
    out = []
    k = el["kind"]
    if k in ("block", "prop"):
        from . import draw_pose
        return draw_pose.label_points(p, el)
    if k == "figure":
        for j, v in el["joints"].items():
            if el.get("pose") == "bust" and j not in ("head", "chin", "lshoulder", "rshoulder", "lelbow", "relbow"):
                continue
            q = u2px(p, [v])[0]
            if 0 <= q[1] < p.H * 0.995:
                out.append((SHORT[j], tuple(q)))
        hd = u2px(p, [el["joints"]["head"]])[0]
        out.append((el["id"], (hd[0], hd[1] - 0.06 * p.H)))
        return out
    pts = u2px(p, element_points(el))
    if k in ("ellipse", "box", "head"):
        out.append((el["id"], (pts[0][0], pts[0][1])))
    else:
        out.append((el["id"], tuple(pts[len(pts) // 2])))
    return out
