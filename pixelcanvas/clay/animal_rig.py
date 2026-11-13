# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from .shapes import _Rx, _Ry, _Rz
from .species import _STANCE


class _QuadRig:
    """Skeleton of a four-legged animal (facing +z, body length L, units before scaling)."""

    def __init__(self, sp):
        self.sp = sp
        L = self.L = sp["L"]
        hf, hh = sp["leg_f"] * L, sp["leg_h"] * L
        (fa, ff), (ha, hfr) = _STANCE[sp["stance"]]
        self.paw_h = sp["paw_size"] * 0.55 * L

        def segs(angles, fr, height):
            c = sum(f * math.cos(math.radians(a)) for a, f in zip(angles, fr))
            k = (height - self.paw_h) / max(c, 1e-6)
            return [(math.radians(a), f * k) for a, f in zip(angles, fr)]
        self.fseg = segs(fa, ff, hf)
        self.hseg = segs(ha, hfr, hh)
        rc, rh = sp["rc"] * L, sp["rh"] * L
        self.rc, self.rh = rc, rh
        dy = hf - hh if not sp["biped"] else hh * 0.08
        B = [("pelvis", None, (0, hh, -L / 2), "pelvis", 0), ("spine", "pelvis", (0, dy / 2, L / 2), "spine", 0),
             ("chest", "spine", (0, dy / 2, L / 2), "chest", 0)]
        na = math.radians(sp["neck_ang"])
        nl = sp["neck_len"] * L
        B.append(("neck", "chest", (0, rc * 0.45, rc * 0.45), "neck", 0))
        nseg = 3 if nl > 0.8 * L else 1
        prev = "neck"
        for i in range(nseg):
            nm = "head" if i == nseg - 1 else f"neck{i + 1}"
            B.append((nm, prev, (0, nl / nseg * math.sin(na), nl / nseg * math.cos(na)), "head" if nm == "head" else
                      "neck", 0))
            prev = nm
        t = sp["tail"]
        tl = t.get("len", 0.0) * L
        nt = 4
        prev = "pelvis"
        b = math.radians(t.get("base", 30))
        B.append(("tail0", "pelvis", (0, rh * 0.25, -rh * 0.85), "tail", 0))
        prev = "tail0"
        for i in range(1, nt + 1):
            B.append((f"tail{i}", prev, (0, tl / nt * math.sin(b), -tl / nt * math.cos(b)) if i == 1 else
                      (0, 0, -tl / nt), "tail", 0))
            prev = f"tail{i}"
        fw = rc * sp["wx"] * 0.62
        hw = rh * sp["wx"] * 0.62
        for sd, s in (("l", 1), ("r", -1)):
            B.append((f"f0_{sd}", "chest", (s * fw, -rc * 0.25, 0.02 * L), "f0", s))
            prev = f"f0_{sd}"
            for i, (a, ln) in enumerate(self.fseg):
                B.append((f"f{i + 1}_{sd}", prev, (0, -ln * math.cos(a), ln * math.sin(a)), f"f{i + 1}", s))
                prev = f"f{i + 1}_{sd}"
            B.append((f"h0_{sd}", "pelvis", (s * hw, -rh * 0.15, 0.0), "h0", s))
            prev = f"h0_{sd}"
            for i, (a, ln) in enumerate(self.hseg):
                B.append((f"h{i + 1}_{sd}", prev, (0, -ln * math.cos(a), ln * math.sin(a)), f"h{i + 1}", s))
                prev = f"h{i + 1}_{sd}"
        self.bones = [(n, p, np.asarray(o, np.float64), k, s) for (n, p, o, k, s) in B]
        self.tail_n = nt
        self.neck_n = nseg

    def fk(self, pose):
        J = {}
        for name, parent, off, kind, s in self.bones:
            d = pose.get(name, {})
            if kind in ("tail", "neck", "head") and name not in pose:
                d = pose.get(kind, {}) if kind != "head" else {}
            Rl = _quad_rot(kind, s, d, name, pose)
            if parent is None:
                root = pose.get("pelvis", {})
                J[name] = (Rl, off + np.array([root.get("x", 0.0), root.get("y", 0.0), root.get("z", 0.0)]))
            else:
                Rp, pp = J[parent]
                if kind in ("spine", "chest") and "torso" in pose:
                    off = off * float(pose["torso"])     # a sitting animal gathers itself up: shorter trunk
                J[name] = (Rp @ Rl, pp + Rp @ off)
        return J


def _quad_rot(kind, s, d, name, pose):
    def g(k, dd=d):
        return float(dd.get(k, 0.0))
    if kind == "pelvis":
        return _Ry(g("yaw")) @ _Rx(-g("pitch")) @ _Rz(g("roll"))
    if kind in ("spine", "chest"):
        return _Ry(g("side")) @ _Rx(-g("bend")) @ _Rz(g("twist"))
    if kind == "neck":
        n = pose.get("neck", {})
        k = 1.0 if name == "neck" else 0.6
        return _Ry(g("turn", n) * 0.5 * k) @ _Rx(-g("up", n) * k) @ _Rz(-g("tilt", n) * 0.5)
    if kind == "head":
        h = pose.get("head", {})
        return _Ry(g("turn", h)) @ _Rx(g("nod", h)) @ _Rz(-g("tilt", h))
    if kind == "tail":
        t = pose.get("tail", {})
        if name == "tail0":
            return _Ry(g("side", t)) @ _Rx(g("up", t))
        return _Ry(g("side", t) * 0.4) @ _Rx(g("curl", t))
    if kind in ("f0", "h0"):
        return _Rx(-g("swing")) @ _Rz(s * g("side"))
    if kind in ("f1",):
        return _Rx(-g("bend"))
    if kind == "f2":
        return _Rx(g("bend"))
    if kind == "h1":
        return _Rx(g("bend"))
    if kind == "h2":
        return _Rx(-g("bend"))
    return np.eye(3)


def _legs(pose, f=None, h=None):
    """Shorthand: f / h = dict(swing, bend1, bend2) applied to both front / hind legs."""
    for sd in ("l", "r"):
        if f:
            pose[f"f0_{sd}"] = dict(swing=f.get("swing", 0), side=f.get("side", 0))
            pose[f"f1_{sd}"] = dict(bend=f.get("b1", 0))
            pose[f"f2_{sd}"] = dict(bend=f.get("b2", 0))
        if h:
            pose[f"h0_{sd}"] = dict(swing=h.get("swing", 0), side=h.get("side", 0))
            pose[f"h1_{sd}"] = dict(bend=h.get("b1", 0))
            pose[f"h2_{sd}"] = dict(bend=h.get("b2", 0))
    return pose


ANIMAL_POSES = ["stand", "walk", "trot", "gallop", "sit", "lie", "sleep", "jump", "beg", "play_bow", "graze", "rear",
                "fly", "swim", "peck"]


def _gait(ph, offs, A_f, A_h, lift_f, lift_h):
    P = {}
    for sd, (of, oh) in offs.items():
        wf = 2 * math.pi * (ph + of)
        wh = 2 * math.pi * (ph + oh)
        lf = max(0.0, math.cos(wf))
        lh = max(0.0, math.cos(wh))
        P[f"f0_{sd}"] = dict(swing=A_f * math.sin(wf))
        P[f"f1_{sd}"] = dict(bend=lift_f * 0.5 * lf)
        P[f"f2_{sd}"] = dict(bend=lift_f * 1.3 * lf)
        P[f"h0_{sd}"] = dict(swing=A_h * math.sin(wh))
        P[f"h1_{sd}"] = dict(bend=lift_h * lf * 0 + lift_h * lh)
        P[f"h2_{sd}"] = dict(bend=lift_h * 0.9 * lh)
    return P


def _folded_hind(sp, thigh=62.0, shin=-72.0, foot=92.0):
    """Hind-leg angles for sitting on the haunches: thigh forward, shin folded back, foot flat and forward.
    Returns (swing + pelvis pitch, knee bend, hock bend) for the species' stance."""
    (a1, a2, a3) = _STANCE[sp.get("stance", "digi")][1][0]
    base = thigh - a1                      # pelvis pitch + hip swing
    b1 = base + a2 - shin
    b2 = foot - base + b1 - a3
    return base, b1, b2


def _curl_up(P, sp):
    """Asleep: curled up on the ground, the spine bent round into a C, the head turned back to rest by the hip,
    front paws tucked under the chin and the tail wrapped round the front."""
    # rolled onto its side: every bend of the spine, neck, legs and tail now curls in the plane of the floor
    # (the belly faces +x; the whole curl is turned so the sleeping face looks to +z like every other pose)
    P["pelvis"] = dict(pitch=0, roll=80, yaw=-95)
    P["spine"] = dict(bend=-40)
    P["chest"] = dict(bend=-40)
    _legs(P, f=dict(swing=60, b1=-110, b2=80), h=dict(swing=-30, b1=110, b2=-90))
    P["neck"] = dict(up=-40)
    P["head"] = dict(nod=-25 if "trunk" in sp.get("extras", ()) else 0, tilt=80)    # a trunk lies forward
    P["tail"] = dict(curl=-20, up=-sp.get("tail", {}).get("base", 30) - 70)
    P["torso"] = 0.92
    P["eyes"] = "closed"


def _animal_pose(name, ph, sp):
    w = 2 * math.pi * ph
    up = sp.get("upright", 0.0)
    tail = sp.get("tail", {})
    base_tail = dict(curl=tail.get("curl", 0))
    P = {"pelvis": dict(pitch=up), "tail": dict(base_tail), "neck": dict(up=-up * 0.6), "head": dict(nod=0)}
    if up:
        _legs(P, f=dict(swing=-up * 0.9), h=dict(swing=up * 0.8, b1=up * 0.6, b2=up * 0.4))
    if sp.get("biped"):
        _legs(P, f=dict(swing=40 - up, b1=60, b2=20))
    if name in ("stand", None):
        return P
    if name in ("walk", "trot", "gallop"):
        if name == "walk":
            G = _gait(ph, {"l": (0.25, 0.0), "r": (0.75, 0.5)}, 18, 16, 40, 34)
            P["neck"] = dict(up=-up * 0.6 + 3 * math.sin(2 * w))
        elif name == "trot":
            G = _gait(ph, {"l": (0.0, 0.5), "r": (0.5, 0.0)}, 26, 24, 60, 50)
            P["tail"] = dict(base_tail, up=10)
        else:
            G = _gait(ph, {"l": (0.0, 0.55), "r": (0.08, 0.63)}, 42, 40, 80, 70)
            P["spine"] = dict(bend=8 * math.sin(w))
            P["chest"] = dict(bend=4 * math.sin(w))
            P["pelvis"] = dict(pitch=up + 5 * math.sin(w + 1.0))
            P["tail"] = dict(base_tail, up=18)
            P["lift"] = 0.05 * max(0.0, math.sin(w))
        P.update(G)
        P["tail"] = dict(P["tail"], side=8 * math.sin(w))
        return P
    if name == "sit":
        P["pelvis"] = dict(pitch=max(up, 38))
        pit = P["pelvis"]["pitch"]
        sw, b1, b2 = _folded_hind(sp)
        _legs(P, f=dict(swing=-pit + 4, b1=0, b2=0), h=dict(swing=sw - pit, b1=b1, b2=b2, side=8))
        P["neck"] = dict(up=-pit * 0.85 + 4)
        P["tail"] = dict(base_tail, up=-12 - tail.get("base", 30) + pit, side=25)
        if tail.get("type") == "bushy" and tail.get("base", 30) >= 60:
            P["tail"] = dict(base_tail, up=100 - tail.get("base", 30) + pit, side=0)
        P["ground"] = "lowest"
        P["sit_solve"] = not sp.get("biped") and sp.get("stance") != "upright"
        if P["sit_solve"]:
            P["torso"] = 0.78
            P["head"] = dict(nod=14)
        return P
    if name == "beg":
        P["pelvis"] = dict(pitch=72)
        sw, b1, b2 = _folded_hind(sp)
        _legs(P, f=dict(swing=-40, b1=100, b2=60), h=dict(swing=sw - 72, b1=b1, b2=b2, side=10))
        P["neck"] = dict(up=-48)
        P["head"] = dict(nod=-8, tilt=10)
        P["tail"] = dict(base_tail, up=-15 - tail.get("base", 30) + 72, side=20)
        if tail.get("type") == "bushy" and tail.get("base", 30) >= 60:     # squirrels: the plume stands up the back
            P["tail"] = dict(base_tail, up=100 - tail.get("base", 30) + 72, side=0)
        P["ground"] = "lowest"
        return P
    if name in ("lie", "sleep"):
        # sphinx: forelegs stretched out in front, hind legs folded flat against the flanks, trunk gathered
        P["pelvis"] = dict(pitch=0)
        hs, hb1, hb2 = _folded_hind(sp, thigh=80.0, shin=-85.0, foot=95.0)
        _legs(P, f=dict(swing=80, b1=15, b2=75, side=4), h=dict(swing=hs, b1=hb1, b2=hb2, side=30))
        P["torso"] = 0.9
        P["neck"] = dict(up=-10)
        if "trunk" in sp.get("extras", ()):                 # lift the head so the trunk curls on the ground
            P["neck"] = dict(up=18)
            P["head"] = dict(nod=-32)
        P["tail"] = dict(base_tail, up=-25)
        if name == "sleep":
            _curl_up(P, sp)
        P["ground"] = "lowest"
        return P
    if name == "jump":
        P["pelvis"] = dict(pitch=up + 14)
        _legs(P, f=dict(swing=55, b1=40, b2=60), h=dict(swing=-45, b1=10, b2=0))
        P["tail"] = dict(base_tail, up=20)
        P["lift"] = 0.35
        return P
    if name == "play_bow":
        P["pelvis"] = dict(pitch=-24)
        _legs(P, f=dict(swing=88, b1=-10, b2=0), h=dict(swing=-24, b1=8, b2=0))
        P["neck"] = dict(up=36)
        P["tail"] = dict(base_tail, up=55, side=12)
        P["ground"] = "lowest"
        return P
    if name == "graze":
        P["neck"] = dict(up=-62)
        P["head"] = dict(nod=28)
        _legs(P, f=dict(swing=-6), h=dict(swing=4))
        return P
    if name == "rear":
        P["pelvis"] = dict(pitch=58)
        _legs(P, f=dict(swing=40, b1=100, b2=90), h=dict(swing=20, b1=30, b2=10))
        P["neck"] = dict(up=-30)
        P["tail"] = dict(base_tail, up=-20)
        P["ground"] = "lowest"
        return P
    return P
