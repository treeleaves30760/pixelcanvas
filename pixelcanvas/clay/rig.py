# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from .._args import unknown_name as _unknown
from .tables import EXPRESSIONS
from .shapes import _Rx, _Ry, _Rz, _exp


# ===================================================================================================== skeleton

_LIMITS = {
    "pelvis": dict(yaw=(-360, 360), pitch=(-120, 120), roll=(-120, 120), turn=(-45, 45), tilt=(-40, 60),
                   side=(-25, 25)),
    "spine": dict(bend=(-30, 60), side=(-35, 35), turn=(-45, 45)),
    "neck": dict(nod=(-45, 45), tilt=(-30, 30), turn=(-60, 60)),
    "head": dict(nod=(-40, 40), tilt=(-30, 30), turn=(-55, 55)),
    "clav": dict(shrug=(-12, 30), fwd=(-20, 25)),
    "shoulder": dict(fwd=(-70, 185), side=(-40, 185), twist=(-110, 110)),
    "elbow": dict(bend=(0, 155)),
    "wrist": dict(bend=(-75, 75), side=(-35, 35), twist=(-90, 90)),
    "hip": dict(fwd=(-50, 140), side=(-30, 90), twist=(-70, 70)),
    "knee": dict(bend=(0, 160)),
    "ankle": dict(bend=(-60, 40), side=(-25, 25)),
    "toe": dict(bend=(-35, 60)),
}

# main degree of freedom when a joint override is a plain number (``ch.pose("stand", elbow_r=90)``)
_MAIN_DOF = {"pelvis": "yaw", "spine": "bend", "neck": "nod", "head": "turn", "clav": "shrug", "shoulder": "fwd",
             "elbow": "bend", "wrist": "bend", "hip": "fwd", "knee": "bend", "ankle": "bend", "toe": "bend"}


def _local_rot(kind, s, d):
    def g(k):
        return float(d.get(k, 0.0))
    if kind == "pelvis":
        return _Ry(g("yaw") + g("turn")) @ _Rx(g("pitch") + g("tilt")) @ _Rz(g("roll") - g("side"))
    if kind == "spine":
        return _exp(g("bend"), 0, -g("side")) @ _Ry(g("turn"))
    if kind in ("neck", "head"):
        return _Ry(g("turn")) @ _Rx(g("nod")) @ _Rz(-g("tilt"))
    if kind == "clav":
        return _Rz(s * g("shrug")) @ _Ry(-s * g("fwd"))
    if kind in ("shoulder", "hip"):
        return _exp(-g("fwd"), 0, s * g("side")) @ _Ry(s * g("twist"))
    if kind == "elbow":
        return _Rx(-g("bend"))
    if kind == "wrist":
        return _Rx(-g("bend")) @ _Rz(s * g("side")) @ _Ry(s * g("twist"))
    if kind == "knee":
        return _Rx(g("bend"))
    if kind in ("ankle", "toe"):
        return _Rx(-g("bend")) @ _Rz(s * g("side"))
    return np.eye(3)


def _clamp_dofs(kind, d):
    lim = _LIMITS.get(kind, {})
    out = {}
    for k, v in d.items():
        if k in lim:
            v = float(np.clip(v, *lim[k]))
        out[k] = v
    return out


class _HumanRig:
    """Bone lengths and rest offsets for a build; ``fk(pose)`` -> {joint: (R, p)} in character space."""

    def __init__(self, cfg, height):
        self.cfg = cfg
        self.height = h = float(height)
        H = self.H = h / cfg["heads"]
        self.R = H / 1.9
        self.head_off = np.array([0, 0.40 * H, 0.03 * H])
        head_j = h - 0.40 * H - 0.92 * self.R * 1.04
        self.neck_len = cfg["neck"] * H + 0.05 * H
        neck_base = head_j - self.neck_len
        self.hip_y = cfg["hipj"] * H
        T = self.T = max(neck_base - self.hip_y, 0.5 * H)
        self.ah = 0.12 * H
        leg = self.hip_y - 0.02 * H - self.ah
        self.thigh, self.shin = leg * 0.5, leg * 0.5
        arm = cfg["arm"] * H
        self.upper, self.fore = arm * 0.52, arm * 0.48
        self.sw, self.hw = cfg["sw"] * H, cfg["hw"] * H
        self.ar, self.lr = cfg["armr"] * H, cfg["legr"] * H
        self.hand = cfg["hand"] * H
        self.foot = cfg["foot"] * H
        self.depth = cfg["depth"] * H
        sdrop = min(cfg["armr"] * H * 1.05 + 0.03 * H, 0.3 * T)
        B = []
        B.append(("pelvis", None, (0, self.hip_y, 0), "pelvis", 0))
        B.append(("spine", "pelvis", (0, 0.28 * T, 0), "spine", 0))
        B.append(("chest", "spine", (0, 0.32 * T, 0), "spine", 0))
        B.append(("neck", "chest", (0, 0.40 * T, 0), "neck", 0))
        B.append(("head", "neck", (0, self.neck_len, 0), "head", 0))
        for sd, s in (("l", 1), ("r", -1)):
            B.append((f"clav_{sd}", "chest", (0, 0.40 * T - sdrop, 0), "clav", s))
            B.append((f"shoulder_{sd}", f"clav_{sd}", (s * self.sw, 0, 0), "shoulder", s))
            B.append((f"elbow_{sd}", f"shoulder_{sd}", (0, -self.upper, 0), "elbow", s))
            B.append((f"wrist_{sd}", f"elbow_{sd}", (0, -self.fore, 0), "wrist", s))
            B.append((f"hip_{sd}", "pelvis", (s * self.hw, -0.02 * H, 0), "hip", s))
            B.append((f"knee_{sd}", f"hip_{sd}", (0, -self.thigh, 0), "knee", s))
            B.append((f"ankle_{sd}", f"knee_{sd}", (0, -self.shin, 0), "ankle", s))
            B.append((f"toe_{sd}", f"ankle_{sd}", (0, -self.ah * 0.75, self.foot * 0.58), "toe", s))
        self.bones = [(n, p, np.asarray(o, np.float64), k, s) for (n, p, o, k, s) in B]
        self.kind = {n: k for (n, _, _, k, _) in self.bones}

    def fk(self, pose):
        J = {}
        root = pose.get("pelvis", {})
        for name, parent, off, kind, s in self.bones:
            d = _clamp_dofs(kind, pose.get(name, {}))
            Rl = _local_rot(kind, s, d)
            if parent is None:
                p = off + np.array([root.get("x", 0.0), root.get("y", 0.0), root.get("z", 0.0)])
                J[name] = (Rl, p)
            else:
                Rp, pp = J[parent]
                J[name] = (Rp @ Rl, pp + Rp @ off)
        return J


# ===================================================================================================== poses

def _P(**kw):
    """Pose dictionary helper: joint=dict(dof=value) plus extra keys (hands, ground, plant, lift, expression)."""
    return kw


def _sym(side_dofs, **kw):
    out = dict(kw)
    for j, d in side_dofs.items():
        out[f"{j}_l"] = dict(d)
        out[f"{j}_r"] = dict(d)
    return out


def _walk(ph, run=False):
    w = 2 * math.pi * ph
    s, c = math.sin(w), math.cos(w)
    if not run:
        A, B = 24, 22
        P = {"hip_l": dict(fwd=A * s), "hip_r": dict(fwd=-A * s),
             "knee_l": dict(bend=5 + 42 * max(0, c) ** 2), "knee_r": dict(bend=5 + 42 * max(0, -c) ** 2),
             "ankle_l": dict(bend=10 * max(0, c) - 14 * max(0, -s) * max(0, -c)),
             "ankle_r": dict(bend=10 * max(0, -c) - 14 * max(0, s) * max(0, c)),
             "shoulder_l": dict(fwd=-B * s, side=7), "shoulder_r": dict(fwd=B * s, side=7),
             "elbow_l": dict(bend=14 + 18 * max(0, -s)), "elbow_r": dict(bend=14 + 18 * max(0, s)),
             "pelvis": dict(turn=-6 * s, side=3 * math.sin(2 * w)), "spine": dict(bend=3, turn=4 * s),
             "chest": dict(turn=5 * s, bend=2), "neck": dict(turn=-4 * s), "head": dict(nod=-3),
             "plant": ["r"] if c > 0 else ["l"], "hands": dict(l="relaxed", r="relaxed")}
    else:
        P = {"hip_l": dict(fwd=12 + 44 * s), "hip_r": dict(fwd=12 - 44 * s),
             "knee_l": dict(bend=22 + 85 * max(0, c) ** 1.4), "knee_r": dict(bend=22 + 85 * max(0, -c) ** 1.4),
             "ankle_l": dict(bend=12 * max(0, c) - 22 * max(0, -c)), "ankle_r": dict(bend=12 * max(0, -c) - 22 * max(0, c)),
             "shoulder_l": dict(fwd=-42 * s + 5, side=10), "shoulder_r": dict(fwd=42 * s + 5, side=10),
             "elbow_l": dict(bend=85), "elbow_r": dict(bend=85),
             "pelvis": dict(turn=-8 * s, tilt=4), "spine": dict(bend=10, turn=6 * s), "chest": dict(turn=7 * s, bend=4),
             "neck": dict(nod=-8, turn=-6 * s), "head": dict(nod=-4),
             "plant": ["r"] if c > 0 else ["l"], "lift": 0.035 * abs(math.sin(2 * w)),
             "hands": dict(l="fist", r="fist")}
    return P


def _pose_dict(name, ph=0.0):
    """Joint-angle dictionary for a named pose (``ph`` = phase 0~1 for cyclic poses)."""
    w = 2 * math.pi * ph
    rest = _sym(dict(shoulder=dict(side=8, twist=-5), elbow=dict(bend=10)), hands=dict(l="relaxed", r="relaxed"))
    if name in ("stand", "idle", None):
        return rest
    if name == "contrapposto":
        return _P(pelvis=dict(side=-6, turn=-6), hip_r=dict(fwd=4, side=2), hip_l=dict(fwd=10, side=10, twist=12),
                  knee_l=dict(bend=16), knee_r=dict(bend=2), ankle_l=dict(bend=-6), spine=dict(side=4, bend=2),
                  chest=dict(side=4, turn=5), neck=dict(tilt=-4), head=dict(tilt=-6, turn=6),
                  shoulder_l=dict(side=10, twist=-5), shoulder_r=dict(side=12, fwd=-4), elbow_l=dict(bend=14),
                  elbow_r=dict(bend=8), hands=dict(l="relaxed", r="relaxed"))
    if name == "walk":
        return _walk(ph)
    if name == "run":
        return _walk(ph, run=True)
    if name == "jump":
        k = 0.5 + 0.5 * math.cos(w)
        return _P(shoulder_l=dict(side=150, fwd=15), shoulder_r=dict(side=150, fwd=15), elbow_l=dict(bend=15),
                  elbow_r=dict(bend=15), hip_l=dict(fwd=8, side=8), hip_r=dict(fwd=-6, side=8),
                  knee_l=dict(bend=55), knee_r=dict(bend=75), ankle_l=dict(bend=-30), ankle_r=dict(bend=-30),
                  spine=dict(bend=-6), head=dict(nod=-10), lift=0.12 + 0.1 * k, ground="feet", plant=[],
                  hands=dict(l="open", r="open"), expression="laugh")
    if name in ("sit", "typing", "reading"):
        P = _sym(dict(hip=dict(fwd=88, side=6), knee=dict(bend=88), ankle=dict(bend=0)),
                 spine=dict(bend=-4), chest=dict(bend=2), ground="seat", plant=[])
        P.update(shoulder_l=dict(fwd=28, side=12, twist=-30), shoulder_r=dict(fwd=28, side=12, twist=-30),
                 elbow_l=dict(bend=48), elbow_r=dict(bend=48), hands=dict(l="relaxed", r="relaxed"))
        if name == "typing":
            P.update(shoulder_l=dict(fwd=36, side=14, twist=-55), shoulder_r=dict(fwd=36, side=14, twist=-55),
                     elbow_l=dict(bend=86), elbow_r=dict(bend=86), wrist_l=dict(twist=-60, bend=-10),
                     wrist_r=dict(twist=-60, bend=-10), spine=dict(bend=6), neck=dict(nod=8), head=dict(nod=10),
                     hands=dict(l="flat", r="flat"))
        if name == "reading":
            P.update(shoulder_l=dict(fwd=22, side=14, twist=-40), shoulder_r=dict(fwd=22, side=14, twist=-40),
                     elbow_l=dict(bend=72), elbow_r=dict(bend=72), neck=dict(nod=14), head=dict(nod=16),
                     hands=dict(l="grip", r="grip"), expression="calm")
        return P
    if name == "sit_floor":
        return _P(hip_l=dict(fwd=80, side=42, twist=75), hip_r=dict(fwd=80, side=42, twist=75),
                  knee_l=dict(bend=130), knee_r=dict(bend=135), ankle_l=dict(bend=10), ankle_r=dict(bend=10),
                  spine=dict(bend=8), chest=dict(bend=4), shoulder_l=dict(fwd=24, side=14, twist=-25),
                  shoulder_r=dict(fwd=24, side=14, twist=-25), elbow_l=dict(bend=40), elbow_r=dict(bend=40),
                  ground="lowest", plant=[], hands=dict(l="relaxed", r="relaxed"))
    if name == "sit_legs":
        return _P(hip_l=dict(fwd=84, side=8), hip_r=dict(fwd=80, side=10), knee_l=dict(bend=14), knee_r=dict(bend=30),
                  ankle_l=dict(bend=10), ankle_r=dict(bend=10), spine=dict(bend=-16), chest=dict(bend=4),
                  neck=dict(nod=8), shoulder_l=dict(fwd=-38, side=20), shoulder_r=dict(fwd=-38, side=20),
                  elbow_l=dict(bend=8), elbow_r=dict(bend=8), wrist_l=dict(bend=-60), wrist_r=dict(bend=-60),
                  ground="lowest", plant=[], hands=dict(l="flat", r="flat"))
    if name == "kneel":
        return _P(hip_l=dict(fwd=88, side=4), knee_l=dict(bend=92), ankle_l=dict(bend=4),
                  hip_r=dict(fwd=-8, side=4), knee_r=dict(bend=96), ankle_r=dict(bend=-45), spine=dict(bend=4),
                  shoulder_l=dict(fwd=20, side=10, twist=-20), elbow_l=dict(bend=60),
                  shoulder_r=dict(fwd=10, side=10), elbow_r=dict(bend=20), ground="lowest", plant=["l"],
                  hands=dict(l="relaxed", r="relaxed"))
    if name in ("lie", "sleep"):
        P = _sym(dict(shoulder=dict(side=14), elbow=dict(bend=10), hip=dict(side=3), knee=dict(bend=4),
                      ankle=dict(bend=-20)), pelvis=dict(pitch=-90), ground="lowest", plant=[],
                 hands=dict(l="relaxed", r="relaxed"))
        if name == "sleep":
            P = _P(pelvis=dict(roll=90, pitch=0), spine=dict(bend=18), chest=dict(bend=10), neck=dict(nod=14),
                   head=dict(nod=6), hip_l=dict(fwd=70), hip_r=dict(fwd=60), knee_l=dict(bend=95),
                   knee_r=dict(bend=88), shoulder_l=dict(fwd=70, twist=-30), shoulder_r=dict(fwd=80, twist=-20),
                   elbow_l=dict(bend=100), elbow_r=dict(bend=90), ankle_l=dict(bend=-20), ankle_r=dict(bend=-20),
                   ground="lowest", plant=[], expression="sleep", hands=dict(l="relaxed", r="relaxed"))
        return P
    if name == "wave":
        P = dict(rest)
        a = 12 * math.sin(w)
        P.update(shoulder_r=dict(side=100, fwd=10, twist=95), elbow_r=dict(bend=40 + a), wrist_r=dict(side=a * 0.5),
                 hands=dict(l="relaxed", r="wave"), head=dict(tilt=-6, turn=-4), chest=dict(side=-3),
                 expression="smile")
        return P
    if name == "point":
        P = dict(rest)
        P.update(shoulder_r=dict(fwd=80, side=22, twist=-10), elbow_r=dict(bend=6), wrist_r=dict(twist=-80),
                 chest=dict(turn=-12), head=dict(turn=-14), hands=dict(l="relaxed", r="point"))
        return P
    if name == "cheer":
        return _P(shoulder_l=dict(side=145, fwd=25, twist=20), shoulder_r=dict(side=145, fwd=25, twist=20),
                  elbow_l=dict(bend=22), elbow_r=dict(bend=22), spine=dict(bend=-5), neck=dict(nod=-8),
                  head=dict(nod=-6), hands=dict(l="fist", r="fist"), expression="laugh")
    if name == "clap":
        o = 0.5 + 0.5 * math.cos(w)
        return _P(shoulder_l=dict(fwd=48, side=18, twist=-62), shoulder_r=dict(fwd=48, side=18, twist=-62),
                  elbow_l=dict(bend=92 - 14 * o), elbow_r=dict(bend=92 - 14 * o), wrist_l=dict(twist=-70),
                  wrist_r=dict(twist=-70), hands=dict(l="open", r="open"), expression="happy")
    if name == "think":
        return _P(shoulder_r=dict(fwd=32, side=8, twist=-40), elbow_r=dict(bend=140), wrist_r=dict(bend=10),
                  shoulder_l=dict(fwd=22, side=10, twist=-78), elbow_l=dict(bend=92), head=dict(tilt=8, nod=-6),
                  neck=dict(tilt=4), hands=dict(l="relaxed", r="fist"), expression=dict(EXPRESSIONS["neutral"],
                                                                                     gaze=(0.5, 0.6), mouth="flat"))
    if name == "arms_crossed":
        return _P(shoulder_l=dict(fwd=30, side=12, twist=-80), shoulder_r=dict(fwd=34, side=12, twist=-80),
                  elbow_l=dict(bend=108), elbow_r=dict(bend=104), hands=dict(l="fist", r="fist"),
                  pelvis=dict(side=-3), head=dict(tilt=-4), expression="determined")
    if name == "hands_on_hips":
        return _P(shoulder_l=dict(side=42, fwd=-12, twist=-55), shoulder_r=dict(side=42, fwd=-12, twist=-55),
                  elbow_l=dict(bend=95), elbow_r=dict(bend=95), wrist_l=dict(bend=20), wrist_r=dict(bend=20),
                  hands=dict(l="fist", r="fist"), hip_l=dict(side=6), hip_r=dict(side=6), expression="smile")
    if name in ("hold",):
        P = dict(rest)
        P.update(shoulder_r=dict(fwd=24, side=10, twist=-15), elbow_r=dict(bend=78),
                 hands=dict(l="relaxed", r="grip"))
        return P
    if name == "carry":
        return _P(shoulder_l=dict(fwd=30, side=14, twist=-25), shoulder_r=dict(fwd=30, side=14, twist=-25),
                  elbow_l=dict(bend=80), elbow_r=dict(bend=80), wrist_l=dict(twist=-40), wrist_r=dict(twist=-40),
                  spine=dict(bend=-6), hands=dict(l="open", r="open"))
    if name == "hug":
        return _P(shoulder_l=dict(fwd=78, side=34, twist=-42), shoulder_r=dict(fwd=78, side=34, twist=-42),
                  elbow_l=dict(bend=72), elbow_r=dict(bend=72), head=dict(tilt=8), hands=dict(l="open", r="open"),
                  expression="happy")
    if name == "dance":
        a = math.sin(w)
        return _P(pelvis=dict(side=8 * a, turn=8 * a), spine=dict(side=-8 * a), chest=dict(side=-6 * a),
                  shoulder_l=dict(side=120 + 30 * a, fwd=15, twist=40), shoulder_r=dict(side=60 - 30 * a, fwd=30),
                  elbow_l=dict(bend=40), elbow_r=dict(bend=70), hip_l=dict(fwd=10 + 15 * a, side=8),
                  hip_r=dict(fwd=5, side=8), knee_l=dict(bend=20 + 25 * max(0, a)), knee_r=dict(bend=12),
                  head=dict(tilt=10 * a, nod=-5), hands=dict(l="open", r="point"), expression="laugh",
                  plant=["r"])
    if name == "kick":
        return _P(hip_r=dict(fwd=80), knee_r=dict(bend=8), ankle_r=dict(bend=-10), hip_l=dict(fwd=-4),
                  knee_l=dict(bend=10), spine=dict(bend=-12), shoulder_l=dict(side=70, fwd=20),
                  shoulder_r=dict(side=50, fwd=-30), elbow_l=dict(bend=20), elbow_r=dict(bend=30),
                  hands=dict(l="open", r="fist"), plant=["l"], expression="determined")
    if name == "throw":
        return _P(shoulder_r=dict(side=95, fwd=-30, twist=100), elbow_r=dict(bend=95), shoulder_l=dict(fwd=70, side=20),
                  elbow_l=dict(bend=20), chest=dict(turn=-22), spine=dict(turn=-12, side=-5), pelvis=dict(turn=-8),
                  hip_l=dict(fwd=24), knee_l=dict(bend=14), hip_r=dict(fwd=-18), knee_r=dict(bend=10),
                  head=dict(turn=20), hands=dict(l="open", r="grip"), plant=["l", "r"], expression="determined")
    if name == "reach_up":
        P = dict(rest)
        P.update(shoulder_r=dict(fwd=168, side=8), elbow_r=dict(bend=6), shoulder_l=dict(side=20),
                 ankle_l=dict(bend=-24), ankle_r=dict(bend=-24), neck=dict(nod=-18), head=dict(nod=-18),
                 spine=dict(side=-4), hands=dict(l="relaxed", r="open"), plant=[])
        return P
    if name == "look_up":
        P = dict(rest)
        P.update(neck=dict(nod=-22), head=dict(nod=-24), spine=dict(bend=-5), expression="surprised")
        return P
    if name == "phone":
        return _P(shoulder_r=dict(fwd=30, side=12, twist=-35), elbow_r=dict(bend=112), wrist_r=dict(twist=-40),
                  shoulder_l=dict(side=8), elbow_l=dict(bend=12), neck=dict(nod=16), head=dict(nod=14),
                  hands=dict(l="relaxed", r="grip"), expression=dict(EXPRESSIONS["smile"], gaze=(0, -0.7)))
    if name == "shrug":
        return _P(clav_l=dict(shrug=18), clav_r=dict(shrug=18), shoulder_l=dict(side=25, twist=60, fwd=10),
                  shoulder_r=dict(side=25, twist=60, fwd=10), elbow_l=dict(bend=80), elbow_r=dict(bend=80),
                  head=dict(tilt=10), hands=dict(l="open", r="open"), expression="worried")
    if name == "bow":
        P = dict(rest)
        P.update(spine=dict(bend=25), chest=dict(bend=20), hip_l=dict(fwd=10), hip_r=dict(fwd=10),
                 shoulder_l=dict(fwd=10, side=4), shoulder_r=dict(fwd=10, side=4))
        return P
    raise _unknown("pose", name, POSES)


POSES = ["stand", "contrapposto", "walk", "run", "jump", "sit", "sit_floor", "sit_legs", "kneel", "lie", "wave", "point", "cheer",
         "clap", "think", "arms_crossed", "hands_on_hips", "hold", "carry", "hug", "dance", "kick", "throw", "reach_up",
         "look_up", "sleep", "phone", "typing", "reading", "shrug", "bow"]

_LATERAL = {"pelvis": ("yaw", "roll", "turn", "side"), "spine": ("side", "turn"), "chest": ("side", "turn"),
            "neck": ("tilt", "turn"), "head": ("tilt", "turn")}


def _mirror_pose(P):
    out = {}
    for k, v in P.items():
        if k.endswith("_l") or k.endswith("_r"):
            k2 = k[:-2] + ("_r" if k.endswith("_l") else "_l")
            out[k2] = dict(v)
        elif k in _LATERAL and isinstance(v, dict):
            out[k] = {d: (-x if d in _LATERAL[k] else x) for d, x in v.items()}
        elif k == "hands":
            out[k] = dict(l=v.get("r", "relaxed"), r=v.get("l", "relaxed"))
        elif k == "plant" and isinstance(v, (list, tuple)):
            out[k] = ["r" if p == "l" else "l" for p in v]
        else:
            out[k] = v
    return out


def _lerp_pose(A, B, t):
    out = {}
    keys = set(A) | set(B)
    for k in keys:
        a, b = A.get(k), B.get(k)
        if isinstance(a, dict) or isinstance(b, dict):
            if k in ("hands",) or k == "expression":
                out[k] = (b if t >= 0.5 else a) or a or b
                continue
            a, b = a or {}, b or {}
            out[k] = {d: (1 - t) * float(a.get(d, 0.0)) + t * float(b.get(d, 0.0)) for d in set(a) | set(b)}
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = (1 - t) * a + t * b
        else:
            out[k] = b if (t >= 0.5 and b is not None) or a is None else a
    return out
