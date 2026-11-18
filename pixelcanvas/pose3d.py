"""3D layout for drawing without a reference: a perspective camera over a ground plane, anime-proportioned figures
posed by named presets and joint angles in degrees, scene blocks (boxes, stairs, cylinders, planes, railings) and
props held in a hand. Pure functions: numbers in, numbers out; nothing here touches the disk.

World space is metres: x to the picture's right, y up, z away from the viewer; the ground is y = 0. A figure's own
frame is x to HER right, y up, z forward (where she faces). ``-r`` / ``-l`` in joint names are the figure's own right
and left. Positive turn / side / roll / twist on the torso and head turn or lean toward her LEFT, which is the
picture's right while she faces the viewer, the same sense as ``--yaw`` (90 = facing the picture's right).

Joint angles (degrees; values replace the preset's):
  body: lean (+ forward), roll (+ toward her left), turn          pelvis / spine / chest: bend, side, twist
  neck / head: turn, nod (+ looks down; up = -nod), tilt (+ ear toward her left shoulder)
  arm-r/l: forward (swing to the front; 90 = pointing ahead, 180 = up), raise (out to the side; 90 = level, 180 = up),
           twist (+ turns the elbow's bend outward)
  elbow-r/l: bend 0-150   hand-r/l: bend, side, twist   leg-r/l: forward, raise, twist   knee-r/l: bend 0-160
  foot-r/l: point (+ toes down)
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .coords import UsageError

SET_EXAMPLE = 'pixelcanvas pose girl --set "arm-r: raise 120, forward 30; elbow-r: 90; spine: twist 20; knee-l: 100"'

# ============================================================================================ rotations


def rx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def ry(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def yaw_frame(yaw: float) -> np.ndarray:
    """Columns: her right, up, forward in world space. yaw 0 faces the viewer, 90 the picture's right."""
    a = math.radians(yaw)
    fwd = np.array([math.sin(a), 0.0, -math.cos(a)])
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, fwd)
    return np.stack([right, up, fwd], 1)


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def rot_between(a, b) -> np.ndarray:
    """The smallest rotation taking direction a to direction b."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = unit(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0]))
        return 2 * np.outer(p, p) - np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


# ============================================================================================ camera

class Camera:
    """A pinhole camera at eye height over the ground. ``horizon`` is where the horizon crosses the picture (0-1000
    from the top); ``tilt`` turns the camera up (+) or down, so verticals converge; without tilt the horizon is
    placed by shifting the lens and verticals stay vertical. ``fov`` spans the picture's longer side."""

    def __init__(self, W: int, H: int, horizon: float = 450.0, eye: float = 1.5, fov: float = 50.0,
                 tilt: float = 0.0):
        self.W, self.H = int(W), int(H)
        self.horizon, self.eye, self.fov, self.tilt = float(horizon), float(eye), float(fov), float(tilt)
        self.f = (max(W, H) / 2.0) / math.tan(math.radians(fov) / 2.0)
        t = math.radians(tilt)
        self.right = np.array([1.0, 0.0, 0.0])
        self.fwd = np.array([0.0, math.sin(t), math.cos(t)])
        self.up = np.array([0.0, math.cos(t), -math.sin(t)])
        self.C = np.array([0.0, self.eye, 0.0])
        self.cx = W / 2.0
        self.cy = self.horizon * H / 1000.0 - self.f * math.tan(t)

    def resized(self, W: int, H: int) -> "Camera":
        return Camera(W, H, self.horizon, self.eye, self.fov, self.tilt)

    def to_cam(self, P) -> np.ndarray:
        Q = np.asarray(P, float).reshape(-1, 3) - self.C
        return np.stack([Q @ self.right, Q @ self.up, Q @ self.fwd], 1)

    def project(self, P) -> Tuple[np.ndarray, np.ndarray]:
        """(N,2) pixels and (N,) depth (camera z, metres) of world points."""
        q = self.to_cam(P)
        z = np.maximum(q[:, 2], 1e-6)
        return np.stack([self.cx + self.f * q[:, 0] / z, self.cy - self.f * q[:, 1] / z], 1), q[:, 2]

    def project_units(self, P) -> np.ndarray:
        px, _ = self.project(P)
        return np.stack([px[:, 0] * 1000.0 / self.W, px[:, 1] * 1000.0 / self.H], 1)

    def rays(self, u, v) -> np.ndarray:
        """World directions through pixels (u, v), scaled so their camera-z component is 1 (t = depth)."""
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        a = (u - self.cx) / self.f
        b = -(v - self.cy) / self.f
        return a[..., None] * self.right + b[..., None] * self.up + self.fwd

    def ground_point(self, x_u: float, y_u: float, height: float = 0.0) -> Optional[np.ndarray]:
        """The world point on the plane y = height seen at (x, y) in 0-1000 units; None above the horizon."""
        d = self.rays(np.array([x_u * self.W / 1000.0]), np.array([y_u * self.H / 1000.0]))[0]
        if abs(d[1]) < 1e-9:
            return None
        t = (height - self.eye) / d[1]
        if t <= 0.05:
            return None
        return self.C + t * d

    def point_at_depth(self, x_u: float, y_u: float, depth: float) -> np.ndarray:
        d = self.rays(np.array([x_u * self.W / 1000.0]), np.array([y_u * self.H / 1000.0]))[0]
        return self.C + max(depth, 0.1) * d

    def horizon_px(self) -> float:
        return self.horizon * self.H / 1000.0

    def person_table(self, height: float = 1.6, ys=None) -> List[Tuple[float, float, float]]:
        """(ground y in units, distance m, height of a ``height`` m person in units) at some ground positions."""
        out = []
        if ys is None:
            lo = self.horizon + 0.12 * (1000 - self.horizon)
            ys = [round(v / 10) * 10 for v in np.linspace(lo, 1000, 4)] if self.horizon < 960 else []
        for y in ys:
            g = self.ground_point(500, y)
            if g is None:
                continue
            top = self.project_units(g + [0, height, 0])[0]
            out.append((float(y), float(self.to_cam(g)[0, 2]), float(y - top[1])))
        return out


# ============================================================================================ joints and limits

LIMITS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "body": {"lean": (-100, 100), "roll": (-100, 100), "turn": (-180, 180)},
    "pelvis": {"bend": (-30, 60), "side": (-25, 25), "twist": (-40, 40)},
    "spine": {"bend": (-35, 70), "side": (-35, 35), "twist": (-50, 50)},
    "chest": {"bend": (-30, 45), "side": (-25, 25), "twist": (-35, 35)},
    "neck": {"turn": (-50, 50), "nod": (-45, 50), "tilt": (-30, 30)},
    "head": {"turn": (-60, 60), "nod": (-45, 50), "tilt": (-35, 35)},
    "arm": {"forward": (-60, 180), "raise": (-30, 180), "twist": (-90, 90)},
    "elbow": {"bend": (0, 150)},
    "hand": {"bend": (-70, 80), "side": (-25, 35), "twist": (-90, 90)},
    "leg": {"forward": (-45, 135), "raise": (-30, 90), "twist": (-45, 60)},
    "knee": {"bend": (0, 160)},
    "foot": {"point": (-30, 60)},
}
SIDED = ("arm", "elbow", "hand", "leg", "knee", "foot")
CENTRAL_SIGNED = {"body": ("roll", "turn"), "pelvis": ("side", "twist"), "spine": ("side", "twist"),
                  "chest": ("side", "twist"), "neck": ("turn", "tilt"), "head": ("turn", "tilt")}
JOINT_ALIAS = {"hips": "pelvis", "thigh": "leg", "upperleg": "leg", "shoulder": "arm", "upperarm": "arm",
               "forearm": "elbow", "lowerarm": "elbow", "wrist": "hand", "ankle": "foot", "waist": "spine",
               "torso": "spine", "back": "spine", "belly": "spine", "ribs": "chest", "root": "body", "whole": "body",
               "figure": "body", "shin": "knee", "lowerleg": "knee", "face": "head"}
DOF_ALIAS = {  # name -> (dof, sign) per joint kind; "*" for any kind that has the dof
    "*": {"fwd": ("forward", 1), "front": ("forward", 1), "back": ("forward", -1), "backward": ("forward", -1),
          "out": ("raise", 1), "abduct": ("raise", 1), "lift": ("raise", 1), "rotate": ("twist", 1),
          "bend": ("bend", 1), "flex": ("bend", 1), "lean": ("bend", 1), "pitch": ("bend", 1),
          "twist": ("twist", 1), "turn": ("twist", 1), "yaw": ("twist", 1), "side": ("side", 1),
          "roll": ("side", 1), "tilt": ("side", 1)},
    "head": {"turn": ("turn", 1), "twist": ("turn", 1), "yaw": ("turn", 1), "rotate": ("turn", 1),
             "nod": ("nod", 1), "down": ("nod", 1), "pitch": ("nod", -1), "up": ("nod", -1), "look-up": ("nod", -1),
             "tilt": ("tilt", 1), "roll": ("tilt", 1), "side": ("tilt", 1), "cock": ("tilt", 1),
             "bend": ("nod", 1)},
    "body": {"lean": ("lean", 1), "bend": ("lean", 1), "pitch": ("lean", 1), "forward": ("lean", 1),
             "back": ("lean", -1), "roll": ("roll", 1), "side": ("roll", 1), "tilt": ("roll", 1),
             "turn": ("turn", 1), "twist": ("turn", 1), "yaw": ("turn", 1)},
    "arm": {"up": ("raise", 1), "raise": ("raise", 1), "forward": ("forward", 1), "twist": ("twist", 1)},
    "leg": {"up": ("forward", 1), "raise": ("raise", 1), "forward": ("forward", 1), "kick": ("forward", 1)},
    "foot": {"point": ("point", 1), "flex": ("point", -1), "bend": ("point", 1), "up": ("point", -1),
             "down": ("point", 1)},
    "elbow": {"bend": ("bend", 1), "flex": ("bend", 1)},
    "knee": {"bend": ("bend", 1), "flex": ("bend", 1)},
}
DEFAULT_DOF = {"elbow": "bend", "knee": "bend", "foot": "point", "head": "turn", "neck": "turn", "spine": "bend",
               "chest": "bend", "pelvis": "bend", "body": "lean", "hand": "bend", "leg": "forward"}


def joint_key(name: str) -> str:
    """'arm-r', 'right arm', 'Arm_R', 'r-arm' -> 'arm-r'; 'waist' -> 'spine'."""
    s = re.sub(r"[\s_.]+", "-", str(name).strip().lower()).strip("-")
    toks = [t for t in s.split("-") if t]
    side = None
    rest = []
    for t in toks:
        if t in ("r", "right", "rt"):
            side = "r"
        elif t in ("l", "left", "lt"):
            side = "l"
        else:
            rest.append(t)
    base = "".join(rest)
    if side is None and len(base) > 1 and base[0] in "rl" and JOINT_ALIAS.get(base[1:], base[1:]) in SIDED:
        side, base = base[0], base[1:]
    base = JOINT_ALIAS.get(base, base)
    if base in ("neck",):
        base = "neck"
    if base not in LIMITS:
        names = ", ".join(k + ("-r/-l" if k in SIDED else "") for k in LIMITS)
        raise UsageError(f"unknown joint {name!r}; joints: {names}", SET_EXAMPLE)
    if base in SIDED:
        if side is None:
            raise UsageError(f"say which {base}: {base}-r (her right) or {base}-l (her left)", SET_EXAMPLE)
        return f"{base}-{side}"
    return base


def _base(key: str) -> str:
    return key.split("-")[0]


def dof_key(joint: str, word: str) -> Tuple[str, float]:
    base = _base(joint)
    kind = "head" if base in ("head", "neck") else base
    w = re.sub(r"[\s_]+", "-", word.strip().lower())
    table = dict(DOF_ALIAS["*"])
    table.update(DOF_ALIAS.get(kind, {}))
    dof, sg = table.get(w, (w, 1))
    if dof not in LIMITS[base]:
        raise UsageError(f"{joint} has no {word!r}; it takes: {', '.join(LIMITS[base])}",
                         f'pixelcanvas pose girl --set "{joint}: {next(iter(LIMITS[base]))} 30"')
    return dof, float(sg)


def parse_edits(text: str) -> Dict[str, Dict[str, float]]:
    """'arm-r: raise 120, forward 30; elbow-r: 90; head: turn -30' -> {'arm-r': {'raise': 120, 'forward': 30}, ...}"""
    out: Dict[str, Dict[str, float]] = {}
    num = r"[-+]?\d+(?:\.\d+)?"
    for seg in re.split(r"[;\n]+", str(text or "")):
        seg = seg.strip()
        if not seg:
            continue
        m = re.match(r"^([a-zA-Z][a-zA-Z_\- ]*?)\s*[:=]\s*(.*)$", seg) if re.match(r"^[a-zA-Z_\- ]+[:=]", seg) \
            else re.match(r"^([a-zA-Z][a-zA-Z_\- ]*?)\s+(.*)$", seg)
        if not m:
            raise UsageError(f"cannot read {seg!r}: write 'joint: angle name number, ...'", SET_EXAMPLE)
        jname, rest = m.group(1), m.group(2)
        jt = jname.strip().split()
        # 'arm r raise 30' (no colon): the joint is the leading words that name a joint
        if ":" not in seg and "=" not in seg and len(jt) > 1:
            for k in range(len(jt), 0, -1):
                try:
                    joint_key(" ".join(jt[:k]))
                    rest = " ".join(jt[k:]) + " " + rest
                    jname = " ".join(jt[:k])
                    break
                except UsageError:
                    continue
        key = joint_key(jname)
        base = _base(key)
        items = [x.strip() for x in re.split(r",", rest) if x.strip()]
        if not items:
            raise UsageError(f"no angle for {key}", SET_EXAMPLE)
        for it in items:
            mm = re.fullmatch(rf"({num})\s*(?:deg|°)?", it)
            if mm:
                dof = DEFAULT_DOF.get(base)
                if dof is None:
                    raise UsageError(f"say which way {key} moves: {', '.join(LIMITS[base])} (e.g. '{key}: raise "
                                     f"{mm.group(1)}')", SET_EXAMPLE)
                out.setdefault(key, {})[dof] = float(mm.group(1))
                continue
            mm = re.fullmatch(rf"([a-zA-Z][a-zA-Z\- ]*?)\s*[:=]?\s*({num})\s*(?:deg|°)?", it) or \
                re.fullmatch(rf"({num})\s*(?:deg|°)?\s*([a-zA-Z][a-zA-Z\- ]*)", it)
            if not mm:
                raise UsageError(f"cannot read {it!r} in {key}: write 'name number', e.g. 'raise 90'", SET_EXAMPLE)
            a, b = mm.group(1), mm.group(2)
            word, val = (a, b) if re.match(r"[a-zA-Z]", a) else (b, a)
            dof, sg = dof_key(key, word)
            out.setdefault(key, {})[dof] = sg * float(val)
    return out


def merge_edits(base: dict, more: dict) -> dict:
    out = {k: dict(v) for k, v in (base or {}).items()}
    for k, v in (more or {}).items():
        out.setdefault(k, {}).update(v)
    return out


def mirror_edits(E: dict) -> dict:
    out: Dict[str, Dict[str, float]] = {}
    for k, v in E.items():
        if "-" in k:
            b, s = k.split("-")
            out[f"{b}-{'l' if s == 'r' else 'r'}"] = dict(v)
        else:
            vv = dict(v)
            for dof in CENTRAL_SIGNED.get(k, ()):
                if dof in vv:
                    vv[dof] = -vv[dof]
            out[k] = vv
    return out


def clamp_edits(E: dict) -> Tuple[dict, List[str]]:
    out: Dict[str, Dict[str, float]] = {}
    notes = []
    for k, v in E.items():
        lim = LIMITS[_base(k)]
        for dof, val in v.items():
            lo, hi = lim[dof]
            c = float(min(max(val, lo), hi))
            if abs(c - val) > 1e-6:
                notes.append(f"{k} {dof} {val:g} -> {c:g} (limit {lo:g}..{hi:g})")
            out.setdefault(k, {})[dof] = c
    return out, notes


# ============================================================================================ presets
# Each preset: a few joint angles. rest: feet (the lowest point stands on the support), seat (the pelvis sits on
# the block given with --on), air (lifted off the ground by ``lift`` m). yaw: the facing used when --yaw is not
# given. plant: on a seat, the feet are moved to the floor below.

def _P(desc, edits, rest="feet", lift=0.0, yaw=None, plant=False):
    return {"desc": desc, "edits": edits, "rest": rest, "lift": lift, "yaw": yaw, "plant": plant}


PRESETS: Dict[str, dict] = {
    "stand": _P("standing, arms relaxed", {
        "arm-r": {"raise": 9}, "arm-l": {"raise": 9}, "elbow-r": {"bend": 12}, "elbow-l": {"bend": 12},
        "leg-r": {"raise": 3}, "leg-l": {"raise": 3}}, plant=True),
    "contrapposto": _P("weight on the right leg, hip up on that side, the left knee relaxed", {
        "pelvis": {"side": -9}, "chest": {"side": -6}, "head": {"tilt": 6},
        "leg-r": {"raise": 6}, "leg-l": {"raise": -4, "forward": 10}, "knee-l": {"bend": 20}, "foot-l": {"point": 12},
        "arm-r": {"raise": 12}, "arm-l": {"raise": 8, "forward": 8}, "elbow-r": {"bend": 15}, "elbow-l": {"bend": 25}}, plant=True),
    "walk": _P("mid-stride: right leg ahead, left arm ahead", {
        "body": {"lean": 3}, "pelvis": {"twist": 6}, "chest": {"twist": -6},
        "leg-r": {"forward": 24}, "knee-r": {"bend": 8}, "leg-l": {"forward": -16}, "knee-l": {"bend": 22},
        "foot-l": {"point": 18}, "arm-r": {"forward": -22, "raise": 6}, "arm-l": {"forward": 24, "raise": 6},
        "elbow-r": {"bend": 15}, "elbow-l": {"bend": 30}}),
    "run": _P("running: leaning forward, right knee up, left leg pushing off on its toes, arms pumping", {
        "body": {"lean": 21}, "pelvis": {"twist": 10}, "chest": {"twist": -12}, "head": {"nod": -18},
        "leg-r": {"forward": 100, "raise": 4}, "knee-r": {"bend": 115}, "foot-r": {"point": 25},
        "leg-l": {"forward": 6, "raise": 3}, "knee-l": {"bend": 16}, "foot-l": {"point": 42},
        "arm-l": {"forward": 65, "raise": 8}, "elbow-l": {"bend": 95},
        "arm-r": {"forward": -50, "raise": 12}, "elbow-r": {"bend": 95}}),
    "sprint": _P("flat-out sprint: deep forward lean, high knee, long push-off", {
        "body": {"lean": 30}, "pelvis": {"twist": 14}, "chest": {"twist": -16}, "head": {"nod": -28},
        "leg-r": {"forward": 110, "raise": 4}, "knee-r": {"bend": 125}, "foot-r": {"point": 30},
        "leg-l": {"forward": 8, "raise": 3}, "knee-l": {"bend": 12}, "foot-l": {"point": 48},
        "arm-l": {"forward": 85, "raise": 8}, "elbow-l": {"bend": 100},
        "arm-r": {"forward": -60, "raise": 14}, "elbow-r": {"bend": 100}}),
    "jump": _P("jumping for joy: in the air, arms up, one knee tucked", {
        "body": {"lean": -4}, "head": {"nod": -18}, "spine": {"bend": -8},
        "arm-r": {"raise": 150, "forward": 20}, "arm-l": {"raise": 145, "forward": 25},
        "elbow-r": {"bend": 25}, "elbow-l": {"bend": 30},
        "leg-r": {"forward": 25}, "knee-r": {"bend": 45}, "foot-r": {"point": 35},
        "leg-l": {"forward": -12}, "knee-l": {"bend": 110}, "foot-l": {"point": 40}}, rest="air", lift=0.35),
    "leap": _P("a long leap: legs split front and back in the air, arms out", {
        "body": {"lean": 14}, "head": {"nod": -12},
        "leg-r": {"forward": 80}, "knee-r": {"bend": 12}, "foot-r": {"point": 30},
        "leg-l": {"forward": -45}, "knee-l": {"bend": 30}, "foot-l": {"point": 40},
        "arm-r": {"raise": 100, "forward": -15}, "arm-l": {"raise": 95, "forward": 35},
        "elbow-r": {"bend": 15}, "elbow-l": {"bend": 20}}, rest="air", lift=0.5),
    "land": _P("landing from a jump: knees deep, leaning forward, arms out for balance", {
        "body": {"lean": 22}, "head": {"nod": -25},
        "leg-r": {"forward": 75, "raise": 12}, "leg-l": {"forward": 65, "raise": 12},
        "knee-r": {"bend": 115}, "knee-l": {"bend": 105}, "foot-r": {"point": -20}, "foot-l": {"point": -15},
        "arm-r": {"raise": 60, "forward": 35}, "arm-l": {"raise": 55, "forward": 20},
        "elbow-r": {"bend": 20}, "elbow-l": {"bend": 25}}, plant=True),
    "sit-chair": _P("sitting on a chair or bench, hands on the lap (--on <block>)", {
        "leg-r": {"forward": 90, "raise": 5}, "leg-l": {"forward": 88, "raise": 7}, "knee-r": {"bend": 90},
        "knee-l": {"bend": 85}, "spine": {"bend": 6}, "arm-r": {"forward": 28, "raise": 6},
        "arm-l": {"forward": 28, "raise": 6}, "elbow-r": {"bend": 55}, "elbow-l": {"bend": 55},
        "hand-r": {"bend": 15}, "hand-l": {"bend": 15}}, rest="seat", plant=True),
    "sit-ground": _P("sitting cross-legged on the ground", {
        "leg-r": {"forward": 70, "raise": 38, "twist": 50}, "leg-l": {"forward": 70, "raise": 38, "twist": 50},
        "knee-r": {"bend": 140}, "knee-l": {"bend": 140}, "spine": {"bend": 10},
        "arm-r": {"forward": 30, "raise": 14}, "arm-l": {"forward": 30, "raise": 14},
        "elbow-r": {"bend": 30}, "elbow-l": {"bend": 30}}),
    "sit-ledge": _P("sitting on a ledge or wall, legs hanging over the edge, hands on the ledge (--on <block>)", {
        "spine": {"bend": -6}, "head": {"tilt": 6},
        "leg-r": {"forward": 90, "raise": 6}, "leg-l": {"forward": 85, "raise": 8},
        "knee-r": {"bend": 70}, "knee-l": {"bend": 100}, "foot-r": {"point": 25}, "foot-l": {"point": 20},
        "arm-r": {"raise": 22, "forward": -15}, "arm-l": {"raise": 22, "forward": -15},
        "elbow-r": {"bend": 8}, "elbow-l": {"bend": 8}, "hand-r": {"bend": -50}, "hand-l": {"bend": -50}},
        rest="seat"),
    "sit-stairs": _P("sitting on a step, feet on a lower step, elbows on the knees (--on <stairs>)", {
        "spine": {"bend": 22}, "head": {"nod": -10},
        "leg-r": {"forward": 68, "raise": 10}, "leg-l": {"forward": 64, "raise": 12},
        "knee-r": {"bend": 115}, "knee-l": {"bend": 110},
        "arm-r": {"forward": 45, "raise": 10}, "arm-l": {"forward": 45, "raise": 10},
        "elbow-r": {"bend": 75}, "elbow-l": {"bend": 80}}, rest="seat", plant=True),
    "kneel": _P("kneeling on the left knee, right foot planted ahead", {
        "leg-r": {"forward": 88, "raise": 6}, "knee-r": {"bend": 92},
        "leg-l": {"forward": -4, "raise": 4}, "knee-l": {"bend": 92}, "foot-l": {"point": 50},
        "arm-r": {"forward": 30, "raise": 8}, "elbow-r": {"bend": 60}, "arm-l": {"raise": 10}, "elbow-l": {"bend": 20}}),
    "crouch": _P("crouching low on both feet, leaning forward", {
        "leg-r": {"forward": 105, "raise": 16}, "leg-l": {"forward": 100, "raise": 16},
        "knee-r": {"bend": 140}, "knee-l": {"bend": 140}, "foot-r": {"point": -25}, "foot-l": {"point": -25},
        "spine": {"bend": 35}, "head": {"nod": -25},
        "arm-r": {"forward": 45, "raise": 10}, "arm-l": {"forward": 45, "raise": 10},
        "elbow-r": {"bend": 60}, "elbow-l": {"bend": 60}}, plant=True),
    "lie-back": _P("lying on the back, one knee up (head toward the picture's left)", {
        "body": {"lean": -90}, "head": {"nod": -5},
        "arm-r": {"raise": 35}, "arm-l": {"raise": 25, "forward": 10}, "elbow-r": {"bend": 15}, "elbow-l": {"bend": 25},
        "leg-r": {"forward": 40}, "knee-r": {"bend": 80}, "leg-l": {"raise": 4}, "knee-l": {"bend": 5}}, yaw=90),
    "lie-side": _P("lying on the left side facing the viewer, head on the left arm", {
        "body": {"roll": 90}, "head": {"tilt": -20},
        "arm-l": {"raise": 165}, "elbow-l": {"bend": 120},
        "arm-r": {"forward": 15, "raise": 5}, "elbow-r": {"bend": 25},
        "leg-r": {"forward": 30}, "knee-r": {"bend": 50}, "leg-l": {"forward": 15}, "knee-l": {"bend": 35}}),
    "reach-up": _P("reaching up high with the right hand, on tiptoe", {
        "arm-r": {"raise": 168, "forward": 15}, "elbow-r": {"bend": 5}, "chest": {"side": -10}, "head": {"nod": -30},
        "arm-l": {"raise": 15}, "elbow-l": {"bend": 15}, "foot-r": {"point": 28}, "foot-l": {"point": 28}}),
    "reach-viewer": _P("reaching toward the viewer with the right hand, stepping in", {
        "body": {"lean": 10}, "chest": {"twist": 18}, "head": {"nod": -6},
        "arm-r": {"forward": 85, "raise": 6}, "elbow-r": {"bend": 6}, "hand-r": {"bend": -15},
        "arm-l": {"forward": -15, "raise": 18}, "elbow-l": {"bend": 25},
        "leg-r": {"forward": 18}, "knee-r": {"bend": 25}, "leg-l": {"forward": -18}, "knee-l": {"bend": 10},
        "foot-l": {"point": 25}}),
    "point": _P("pointing ahead to her right with a straight arm, looking there", {
        "arm-r": {"forward": 70, "raise": 40}, "elbow-r": {"bend": 3}, "chest": {"twist": -10}, "head": {"turn": -25},
        "arm-l": {"raise": 10}, "elbow-l": {"bend": 15}, "leg-l": {"forward": 8}, "knee-l": {"bend": 10}}, plant=True),
    "wave": _P("waving with the right hand raised beside the head", {
        "arm-r": {"raise": 95, "twist": 80}, "elbow-r": {"bend": 80}, "hand-r": {"side": 10}, "head": {"tilt": 8},
        "arm-l": {"raise": 9}, "elbow-l": {"bend": 15}, "pelvis": {"side": -4}}, plant=True),
    "hold-front": _P("holding something in front of the chest with both hands", {
        "arm-r": {"forward": 38, "raise": 6, "twist": -38}, "arm-l": {"forward": 38, "raise": 6, "twist": -38},
        "elbow-r": {"bend": 72}, "elbow-l": {"bend": 72}, "head": {"nod": 8}}, plant=True),
    "hold-side": _P("holding something at the right side, forearm forward", {
        "arm-r": {"forward": 15, "raise": 14, "twist": 10}, "elbow-r": {"bend": 85},
        "arm-l": {"raise": 9}, "elbow-l": {"bend": 15}}, plant=True),
    "carry-shoulder": _P("carrying something over the right shoulder, hand at the shoulder", {
        "arm-r": {"raise": 35, "forward": 40, "twist": 30}, "elbow-r": {"bend": 140},
        "arm-l": {"raise": 9}, "elbow-l": {"bend": 15}, "chest": {"side": 4}}, plant=True),
    "umbrella": _P("holding an umbrella's handle in front of the chest with the right hand", {
        "arm-r": {"forward": 32, "raise": 12, "twist": -12}, "elbow-r": {"bend": 88}, "head": {"nod": 4},
        "arm-l": {"raise": 9}, "elbow-l": {"bend": 15}}, plant=True),
    "fight-stance": _P("fighting stance: staggered bent legs, fists up guarding the face", {
        "spine": {"bend": 10}, "head": {"nod": 8}, "chest": {"twist": 10},
        "leg-r": {"forward": 28, "raise": 14}, "knee-r": {"bend": 30},
        "leg-l": {"forward": -20, "raise": 14}, "knee-l": {"bend": 28}, "foot-l": {"point": 15},
        "arm-r": {"forward": 55, "raise": 20, "twist": -30}, "elbow-r": {"bend": 115},
        "arm-l": {"forward": 40, "raise": 25, "twist": -30}, "elbow-l": {"bend": 120}}, yaw=-30, plant=True),
    "kick": _P("a high front kick with the right leg, leaning back, arms guarding", {
        "body": {"lean": -12}, "head": {"nod": 10},
        "leg-r": {"forward": 100, "raise": 6}, "knee-r": {"bend": 15}, "foot-r": {"point": 20},
        "leg-l": {"raise": 4}, "knee-l": {"bend": 15},
        "arm-r": {"forward": 20, "raise": 45}, "elbow-r": {"bend": 70},
        "arm-l": {"forward": 40, "raise": 30, "twist": -20}, "elbow-l": {"bend": 100}}, yaw=-35),
    "punch": _P("a straight right punch, left fist guarding the chin, front foot forward", {
        "body": {"lean": 8}, "chest": {"twist": 25},
        "arm-r": {"forward": 88, "raise": 6, "twist": -10}, "elbow-r": {"bend": 4},
        "arm-l": {"forward": 45, "raise": 22, "twist": -30}, "elbow-l": {"bend": 125},
        "leg-l": {"forward": 25, "raise": 8}, "knee-l": {"bend": 28},
        "leg-r": {"forward": -22, "raise": 6}, "knee-r": {"bend": 12}, "foot-r": {"point": 20}}, yaw=-25, plant=True),
    "swing": _P("swinging a sword with both hands in a lunge (prop sword --hand both)", {
        "body": {"lean": 14}, "chest": {"twist": 20}, "spine": {"bend": 6},
        "arm-r": {"forward": 70, "raise": 25, "twist": -25}, "elbow-r": {"bend": 25},
        "arm-l": {"forward": 75, "raise": 20, "twist": -35}, "elbow-l": {"bend": 35},
        "leg-l": {"forward": 60, "raise": 8}, "knee-l": {"bend": 75},
        "leg-r": {"forward": -30, "raise": 6}, "knee-r": {"bend": 8}, "foot-r": {"point": 30}}, yaw=-30, plant=True),
    "dance-twirl": _P("twirling on the right toes, left foot at the right knee, arms open", {
        "foot-r": {"point": 35}, "leg-l": {"forward": 40, "raise": 30, "twist": 50}, "knee-l": {"bend": 120},
        "arm-r": {"raise": 160, "forward": 20}, "elbow-r": {"bend": 25},
        "arm-l": {"raise": 95, "forward": 25}, "elbow-l": {"bend": 30},
        "head": {"tilt": 10, "turn": 15}, "chest": {"side": 6}}),
    "look-back": _P("walking away and looking back over the left shoulder at the viewer", {
        "chest": {"twist": 30}, "neck": {"turn": 40}, "head": {"turn": 45, "tilt": -5},
        "leg-r": {"forward": 12}, "knee-r": {"bend": 10}, "leg-l": {"forward": -10}, "knee-l": {"bend": 20},
        "foot-l": {"point": 20}, "arm-r": {"raise": 10, "forward": -10}, "arm-l": {"raise": 10, "forward": 12},
        "elbow-r": {"bend": 15}, "elbow-l": {"bend": 20}}, yaw=160),
    "lean-wall": _P("leaning back against a wall, arms crossed, left leg crossed in front", {
        "body": {"lean": -8}, "head": {"turn": 15, "tilt": 6},
        "arm-r": {"forward": 42, "raise": -6, "twist": -55}, "elbow-r": {"bend": 120},
        "arm-l": {"forward": 48, "raise": -6, "twist": -55}, "elbow-l": {"bend": 115},
        "leg-r": {"forward": 6}, "leg-l": {"forward": 10, "raise": -14, "twist": 20}, "knee-l": {"bend": 25},
        "foot-l": {"point": 30}}, plant=True),
    "lean-railing": _P("leaning forward on a railing on both forearms, one foot resting on its toes", {
        "body": {"lean": 22}, "head": {"nod": -12},
        "arm-r": {"forward": 18, "raise": 10, "twist": -30}, "elbow-r": {"bend": 70},
        "arm-l": {"forward": 18, "raise": 10, "twist": -30}, "elbow-l": {"bend": 70},
        "leg-l": {"forward": -8}, "knee-l": {"bend": 30}, "foot-l": {"point": 40}}),
    "hug-knees": _P("sitting on the ground hugging the knees to the chest", {
        "leg-r": {"forward": 118, "raise": 6}, "leg-l": {"forward": 118, "raise": 6},
        "knee-r": {"bend": 150}, "knee-l": {"bend": 150}, "spine": {"bend": 28}, "head": {"nod": 18},
        "arm-r": {"forward": 70, "raise": 6, "twist": -40}, "arm-l": {"forward": 70, "raise": 6, "twist": -40},
        "elbow-r": {"bend": 75}, "elbow-l": {"bend": 75}}),
    "stretch": _P("stretching both arms straight up, on tiptoe, arching back", {
        "arm-r": {"raise": 170, "forward": 8}, "arm-l": {"raise": 170, "forward": 8},
        "elbow-r": {"bend": 5}, "elbow-l": {"bend": 5}, "spine": {"bend": -14}, "head": {"nod": -18},
        "foot-r": {"point": 25}, "foot-l": {"point": 25}}),
    "peace-sign": _P("a peace sign beside the face with the right hand, head tilted", {
        "arm-r": {"raise": 30, "forward": 55, "twist": 25}, "elbow-r": {"bend": 130}, "head": {"tilt": 12},
        "pelvis": {"side": -5}, "arm-l": {"raise": 10}, "elbow-l": {"bend": 20},
        "leg-l": {"forward": 10, "raise": -5}, "knee-l": {"bend": 18}}, plant=True),
    "walk-together": _P("walking side by side, the left hand reaching out to a partner's hand (partner: --mirror)", {
        "body": {"lean": 3}, "pelvis": {"twist": 6}, "chest": {"twist": -6},
        "leg-r": {"forward": 22}, "knee-r": {"bend": 8}, "leg-l": {"forward": -15}, "knee-l": {"bend": 22},
        "foot-l": {"point": 16}, "arm-r": {"forward": -18, "raise": 6}, "elbow-r": {"bend": 15},
        "arm-l": {"raise": 28, "forward": 12}, "elbow-l": {"bend": 12}, "head": {"turn": 12}}),
}
PRESET_ALIAS = {"sit": "sit-chair", "hold": "hold-front", "lie": "lie-back", "standing": "stand", "running": "run",
                "walking": "walk", "jumping": "jump", "sitting": "sit-chair", "fight": "fight-stance",
                "peace": "peace-sign", "twirl": "dance-twirl", "dance": "dance-twirl", "lean": "lean-wall"}


def preset_key(name: str) -> str:
    k = re.sub(r"[\s_]+", "-", str(name).strip().lower())
    k = PRESET_ALIAS.get(k, k)
    if k not in PRESETS:
        import difflib
        close = difflib.get_close_matches(k, list(PRESETS), n=1, cutoff=0.5)
        raise UsageError(f"unknown preset {name!r}" + (f" (did you mean {close[0]}?)" if close else "") +
                         f"; presets: {', '.join(PRESETS)}", "pixelcanvas pose girl --preset run")
    return k


# ============================================================================================ proportions

# joint positions (x from the centre line, y from the top of the head) in figure heights, chibi and adult
_CHIBI = {"head": (0, 0.20), "chin": (0, 0.40), "shoulder": (0.105, 0.45), "elbow": (0.15, 0.565),
          "wrist": (0.165, 0.665), "hip": (0.065, 0.69), "knee": (0.065, 0.83), "ankle": (0.065, 0.965)}
_ADULT = {"head": (0, 0.0667), "chin": (0, 0.1333), "shoulder": (0.125, 0.178), "elbow": (0.147, 0.38),
          "wrist": (0.153, 0.52), "hip": (0.08, 0.533), "knee": (0.073, 0.747), "ankle": (0.067, 0.96)}


class Body:
    """Bone lengths and part sizes (metres) for a figure ``height`` m tall and ``heads`` heads tall."""

    def __init__(self, height: float = 1.6, heads: float = 6.5):
        self.Ht = float(height)
        self.heads = float(heads)
        h = self.Ht / self.heads
        self.h = h
        t = float(np.clip((0.4 - 1.0 / self.heads) / (0.4 - 0.1333), -0.2, 1.15))
        self.t = t
        fr = {k: (_CHIBI[k][0] + t * (_ADULT[k][0] - _CHIBI[k][0]), _CHIBI[k][1] + t * (_ADULT[k][1] - _CHIBI[k][1]))
              for k in _CHIBI}
        Ht = self.Ht

        def yv(n):
            return Ht * (1 - fr[n][1])

        def d2(a, b):
            return Ht * math.hypot(fr[a][0] - fr[b][0], fr[a][1] - fr[b][1])
        self.y_hip = yv("hip")
        self.y_sh = yv("shoulder")
        self.shw = fr["shoulder"][0] * Ht
        self.hipw = fr["hip"][0] * Ht
        self.Lu, self.Lf = d2("shoulder", "elbow"), d2("elbow", "wrist")
        self.Lt, self.Ls = d2("hip", "knee"), d2("knee", "ankle")
        self.ah = max(0.035 * Ht, Ht * (1 - fr["ankle"][1]))
        torso = self.y_sh - self.y_hip
        self.Lw = 0.32 * torso
        self.Lc = torso - self.Lw + 0.1 * h
        chin = Ht - h
        self.Ln = max(0.02 * Ht, (chin + 0.12 * h) - (self.y_sh + 0.1 * h))
        self.k = 1.0 + 0.45 * (1.0 - float(np.clip(t, 0, 1)))          # chibi limbs are thicker
        self.Lh = max(0.1 * Ht, 0.32 * h)
        self.fl = Ht * (0.15 + 0.04 * (1 - float(np.clip(t, 0, 1))))
        k = self.k
        self.r = {"neck": 0.03 * Ht * k, "upperarm": 0.028 * Ht * k, "forearm": 0.022 * Ht * k,
                  "thigh": 0.05 * Ht * k, "shin": 0.034 * Ht * k, "shoulder": 0.031 * Ht * k,
                  "elbow": 0.021 * Ht * k, "knee": 0.031 * Ht * k, "ankle": 0.02 * Ht * k}


# ============================================================================================ forward kinematics

MASS = {"head": 0.08, "torso": 0.5, "upperarm": 0.028, "forearm": 0.018, "hand": 0.007, "thigh": 0.1,
        "shin": 0.045, "foot": 0.015}


def fk(body: Body, E: dict, yaw: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Joint positions (world directions, the pelvis at (0, hip height, 0)) and part frames (3x3, columns = the
    part's right, up, forward) for edits E."""
    def g(j, k):
        return float(E.get(j, {}).get(k, 0.0))
    b = body
    h = b.h
    root = np.array([0.0, b.y_hip, 0.0])
    R0 = yaw_frame(yaw) @ ry(-g("body", "turn")) @ rx(g("body", "lean")) @ rz(g("body", "roll"))
    P = R0 @ ry(-g("pelvis", "twist")) @ rx(g("pelvis", "bend")) @ rz(g("pelvis", "side"))
    S1 = P @ ry(-g("spine", "twist") / 2) @ rx(g("spine", "bend") / 2) @ rz(g("spine", "side") / 2)
    S2 = S1 @ ry(-(g("spine", "twist") / 2 + g("chest", "twist"))) @ rx(g("spine", "bend") / 2 + g("chest", "bend")) \
        @ rz(g("spine", "side") / 2 + g("chest", "side"))
    N = S2 @ ry(-g("neck", "turn")) @ rx(g("neck", "nod")) @ rz(g("neck", "tilt"))
    Hd = N @ ry(-g("head", "turn")) @ rx(g("head", "nod")) @ rz(g("head", "tilt"))
    J: Dict[str, np.ndarray] = {}
    F: Dict[str, np.ndarray] = {"root": R0, "pelvis": P, "spine": S1, "chest": S2, "neck": N, "head": Hd}
    J["pelvis"] = root
    J["spine"] = root + S1 @ [0, b.Lw, 0]
    J["neck"] = J["spine"] + S2 @ [0, b.Lc, 0]
    J["chest"] = J["spine"] + S2 @ [0, 0.55 * b.Lc, 0]
    J["neck-top"] = J["neck"] + N @ [0, b.Ln, 0]
    J["head"] = J["neck-top"] + Hd @ [0, 0.38 * h, 0.04 * h]
    J["chin"] = J["head"] + Hd @ [0, -0.5 * h, 0.08 * h]
    J["head-top"] = J["head"] + Hd @ [0, 0.5 * h, 0]
    for s, sg in (("r", 1.0), ("l", -1.0)):
        J[f"shoulder-{s}"] = J["neck"] + S2 @ [sg * b.shw, -0.1 * h, 0]
        A = S2 @ rz(sg * g(f"arm-{s}", "raise")) @ rx(-g(f"arm-{s}", "forward")) @ ry(sg * g(f"arm-{s}", "twist"))
        J[f"elbow-{s}"] = J[f"shoulder-{s}"] + A @ [0, -b.Lu, 0]
        Ef = A @ rx(-g(f"elbow-{s}", "bend"))
        J[f"wrist-{s}"] = J[f"elbow-{s}"] + Ef @ [0, -b.Lf, 0]
        Hn = Ef @ rx(-g(f"hand-{s}", "bend")) @ rz(sg * g(f"hand-{s}", "side")) @ ry(sg * g(f"hand-{s}", "twist"))
        J[f"hand-{s}"] = J[f"wrist-{s}"] + Hn @ [0, -b.Lh, 0]
        F[f"arm-{s}"], F[f"elbow-{s}"], F[f"hand-{s}"] = A, Ef, Hn
        J[f"hip-{s}"] = root + P @ [sg * b.hipw, 0, 0]
        Lg = P @ rz(sg * g(f"leg-{s}", "raise")) @ rx(-g(f"leg-{s}", "forward")) @ ry(sg * g(f"leg-{s}", "twist"))
        J[f"knee-{s}"] = J[f"hip-{s}"] + Lg @ [0, -b.Lt, 0]
        K = Lg @ rx(g(f"knee-{s}", "bend"))
        J[f"ankle-{s}"] = J[f"knee-{s}"] + K @ [0, -b.Ls, 0]
        Ft = K @ rx(g(f"foot-{s}", "point"))
        F[f"leg-{s}"], F[f"knee-{s}"], F[f"foot-{s}"] = Lg, K, Ft
        _feet(body, J, F, s)
    return J, F


def _feet(body: Body, J, F, s) -> None:
    Ft = F[f"foot-{s}"]
    J[f"heel-{s}"] = J[f"ankle-{s}"] + Ft @ [0, -body.ah, -0.22 * body.fl]
    J[f"foot-{s}"] = J[f"ankle-{s}"] + Ft @ [0, -body.ah, 0.78 * body.fl]


def com(body: Body, J) -> np.ndarray:
    pts, w = [J["head"], (J["pelvis"] + J["neck"]) / 2], [MASS["head"], MASS["torso"]]
    for s in "rl":
        for a, b_, m in ((f"shoulder-{s}", f"elbow-{s}", "upperarm"), (f"elbow-{s}", f"wrist-{s}", "forearm"),
                         (f"wrist-{s}", f"hand-{s}", "hand"), (f"hip-{s}", f"knee-{s}", "thigh"),
                         (f"knee-{s}", f"ankle-{s}", "shin"), (f"heel-{s}", f"foot-{s}", "foot")):
            pts.append((J[a] + J[b_]) / 2)
            w.append(MASS[m])
    w = np.asarray(w)
    return (np.asarray(pts) * w[:, None]).sum(0) / w.sum()


# ============================================================================================ primitives

def ell(c, R, semi, part, group, **kw) -> dict:
    A = np.asarray(R, float) @ np.diag(semi)
    return {"t": "ell", "c": np.asarray(c, float), "A": A, "part": part, "group": group, **kw}


def limb(a, b, r, frame, part, group, rb=None, **kw) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    L = float(np.linalg.norm(ax))
    u = ax / L if L > 1e-9 else np.asarray(frame)[:, 1]
    x = np.asarray(frame)[:, 0] - np.dot(np.asarray(frame)[:, 0], u) * u
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(u, [0, 0, 1.0])
    x = unit(x)
    z = np.cross(x, u)
    rr = r if rb is None else 0.5 * (r + rb)
    R = np.stack([x, u, z], 1)
    return ell((a + b) / 2, R, (rr, L / 2 + 0.35 * rr, rr), part, group, **kw)


def body_prims(body: Body, J, F) -> List[dict]:
    """Mannequin parts as ellipsoids: head, neck, rib cage, waist, pelvis, limbs, joint balls, hands, feet."""
    b, h, k = body, body.h, body.k
    Ht = b.Ht
    out = [ell(J["head"], F["head"], (0.42 * h, 0.5 * h, 0.46 * h), "head", "head"),
           limb(J["neck"], J["neck-top"] + F["neck"] @ [0, 0.1 * h, 0], b.r["neck"], F["neck"], "neck", "head"),
           ell(J["spine"] + F["chest"] @ [0, 0.56 * b.Lc, 0], F["chest"], (0.9 * b.shw, 0.58 * b.Lc, 0.56 * b.shw),
               "chest", "torso"),
           ell(J["spine"], F["spine"], (0.62 * b.shw, 0.5 * b.Lw + 0.02 * Ht, 0.47 * b.shw), "waist", "torso"),
           ell(J["pelvis"] + F["pelvis"] @ [0, -0.01 * Ht, 0], F["pelvis"],
               (1.45 * b.hipw, 0.075 * Ht, 0.9 * b.hipw), "pelvis", "torso")]
    for s in "rl":
        arm, leg = f"arm-{s}", f"leg-{s}"
        out += [ell(J[f"shoulder-{s}"], F["chest"], (b.r["shoulder"],) * 3, f"shoulder-{s}", arm),
                limb(J[f"shoulder-{s}"], J[f"elbow-{s}"], b.r["upperarm"], F[f"arm-{s}"], f"upperarm-{s}", arm),
                ell(J[f"elbow-{s}"], F[f"elbow-{s}"], (b.r["elbow"],) * 3, f"elbow-{s}", arm),
                limb(J[f"elbow-{s}"], J[f"wrist-{s}"], b.r["forearm"], F[f"elbow-{s}"], f"forearm-{s}", arm),
                ell((J[f"wrist-{s}"] + J[f"hand-{s}"]) / 2, F[f"hand-{s}"], (0.013 * Ht * k, 0.5 * b.Lh,
                                                                              0.03 * Ht * k), f"hand-{s}", arm),
                limb(J[f"hip-{s}"], J[f"knee-{s}"], b.r["thigh"], F[f"leg-{s}"], f"thigh-{s}", leg,
                     rb=0.8 * b.r["thigh"]),
                ell(J[f"knee-{s}"], F[f"knee-{s}"], (b.r["knee"],) * 3, f"knee-{s}", leg),
                limb(J[f"knee-{s}"], J[f"ankle-{s}"], b.r["shin"], F[f"knee-{s}"], f"shin-{s}", leg,
                     rb=0.7 * b.r["shin"]),
                ell((J[f"heel-{s}"] + J[f"foot-{s}"]) / 2 + F[f"foot-{s}"] @ [0, 0.5 * b.ah, 0], F[f"foot-{s}"],
                    (0.028 * Ht * k, 0.5 * b.ah, 0.55 * b.fl), f"foot-{s}", leg)]
    return out


def lowest(prim) -> Tuple[float, np.ndarray]:
    """(min y, the lowest surface point) of a primitive."""
    A, c = prim["A"], prim["c"]
    if prim["t"] == "ell":
        r = A[1, :]
        n = float(np.linalg.norm(r))
        u = -r / n if n > 1e-12 else np.zeros(3)
        p = c + A @ u
        return float(p[1]), p
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    if prim["t"] == "cyl":
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        corners = np.array([[math.cos(a), y, math.sin(a)] for a in ang for y in (-1, 1)])
    P = c + corners @ A.T
    i = int(np.argmin(P[:, 1]))
    return float(P[i, 1]), P[i]


# ============================================================================================ blocks

BLOCK_KINDS = ("box", "stairs", "cylinder", "plane", "railing")
BLOCK_WORDS = {"wall": ("box", (4.0, 0.3, 1.0)), "bench": ("box", (1.5, 0.45, 0.45)),
               "building": ("box", (8.0, 6.0, 10.0)), "house": ("box", (7.0, 6.0, 6.0)),
               "step": ("box", (1.2, 0.3, 0.17)), "table": ("box", (1.2, 0.7, 0.75)),
               "chair": ("box", (0.45, 0.45, 0.45)), "desk": ("box", (1.2, 0.6, 0.75)), "bed": ("box", (1.0, 2.0, 0.5)),
               "crate": ("box", (0.6, 0.6, 0.6)), "ledge": ("box", (3.0, 0.4, 0.9)), "box": ("box", (1.0, 1.0, 1.0)),
               "stairs": ("stairs", (1.4, 2.4, 1.2)), "stair": ("stairs", (1.4, 2.4, 1.2)),
               "steps": ("stairs", (1.4, 1.2, 0.6)),
               "cylinder": ("cylinder", (0.3, 0.3, 1.0)), "pole": ("cylinder", (0.1, 0.1, 3.0)),
               "post": ("cylinder", (0.12, 0.12, 1.2)), "lamp": ("cylinder", (0.14, 0.14, 4.0)),
               "lamppost": ("cylinder", (0.14, 0.14, 4.0)), "tree": ("cylinder", (0.4, 0.4, 4.0)),
               "trunk": ("cylinder", (0.4, 0.4, 4.0)), "column": ("cylinder", (0.6, 0.6, 4.0)),
               "pillar": ("cylinder", (0.6, 0.6, 4.0)),
               "plane": ("plane", (4.0, 4.0, 0.0)), "floor": ("plane", (6.0, 6.0, 0.0)),
               "water": ("plane", (10.0, 8.0, 0.0)), "rug": ("plane", (2.0, 1.4, 0.0)), "mat": ("plane", (1.0, 1.8, 0.0)),
               "railing": ("railing", (3.0, 0.06, 1.0)), "rail": ("railing", (3.0, 0.06, 1.0)),
               "fence": ("railing", (4.0, 0.06, 1.1)), "balustrade": ("railing", (3.0, 0.08, 1.0))}


def block_kind(word: str) -> Optional[str]:
    w = str(word or "").strip().lower()
    if w in BLOCK_KINDS:
        return w
    if w in BLOCK_WORDS:
        return BLOCK_WORDS[w][0]
    return None


class Block:
    """A block standing on the ground (or lifted): ``pos`` = world point of the middle of its front bottom edge
    (a cylinder: its bottom centre), ``size`` = width, depth, height in metres, ``rot`` degrees about the vertical
    (+ turns the front toward the picture's left), ``steps`` for stairs (rising away from the front)."""

    def __init__(self, name: str, kind: str, pos, size, rot: float = 0.0, steps: int = 6):
        self.name, self.kind = name, kind
        self.pos = np.asarray(pos, float)
        self.size = [float(v) for v in size]
        self.rot = float(rot)
        self.steps = max(1, int(steps))
        self.R = ry(self.rot)

    def local(self, P) -> np.ndarray:
        return (np.asarray(P, float) - self.pos) @ self.R          # R^T (P - pos)

    def world(self, q) -> np.ndarray:
        return self.pos + self.R @ np.asarray(q, float)

    def prims(self, owner: str) -> List[dict]:
        w, d, h = self.size
        R = self.R
        out = []

        def box(c_local, half, part):
            out.append({"t": "box", "c": self.world(c_local), "A": R @ np.diag(half), "part": part,
                        "group": self.name})
        if self.kind == "box":
            box([0, h / 2, d / 2], [w / 2, max(h, 0.005) / 2, d / 2], self.name)
        elif self.kind == "plane":
            box([0, 0.004, d / 2], [w / 2, 0.004, d / 2], self.name)
        elif self.kind == "stairs":
            n = self.steps
            for i in range(n):
                z0 = i * d / n
                hh = (i + 1) * h / n
                box([0, hh / 2, (z0 + d) / 2], [w / 2, hh / 2, (d - z0) / 2], f"{self.name} step {i + 1}")
        elif self.kind == "cylinder":
            r = w / 2
            out.append({"t": "cyl", "c": self.world([0, h / 2, 0]), "A": R @ np.diag([r, h / 2, r]),
                        "part": self.name, "group": self.name})
        elif self.kind == "railing":
            t = max(0.035, min(0.07, d if d > 0 else 0.05))
            box([0, h - t / 2, 0], [w / 2, t / 2, t / 2], f"{self.name} rail")
            box([0, 0.55 * h, 0], [w / 2, t / 3, t / 3], f"{self.name} mid rail")
            n = max(2, int(round(w / 1.2)) + 1)
            for x in np.linspace(-w / 2 + t / 2, w / 2 - t / 2, n):
                box([x, h / 2, 0], [t / 2, h / 2, t / 2], f"{self.name} post")
        for p in out:
            p["owner"] = owner
        return out

    def top_at(self, X: float, Z: float) -> float:
        w, d, h = self.size
        q = self.local([X, 0, Z])
        base = float(self.pos[1])
        if self.kind == "stairs":
            n = self.steps
            i = int(np.clip(math.floor(q[2] / max(d / n, 1e-6)), 0, n - 1))
            return base + (i + 1) * h / n
        if self.kind == "plane":
            return base + 0.008
        return base + h

    def contains(self, X: float, Z: float, margin: float = 0.0) -> bool:
        w, d, h = self.size
        q = self.local([X, 0, Z])
        if self.kind == "cylinder":
            return math.hypot(q[0], q[2]) <= w / 2 + margin
        if self.kind == "railing":
            return abs(q[0]) <= w / 2 + margin and abs(q[2]) <= max(d, 0.05) / 2 + margin
        return abs(q[0]) <= w / 2 + margin and -margin <= q[2] <= d + margin

    def top_centre(self) -> np.ndarray:
        w, d, h = self.size
        if self.kind in ("cylinder", "railing"):
            c = self.world([0, 0, 0])
        elif self.kind == "stairs":
            c = self.world([0, 0, d * (self.steps - 0.5) / self.steps])
        else:
            c = self.world([0, 0, d / 2])
        return np.array([c[0], self.top_at(c[0], c[2]), c[2]])

    def closest(self, P) -> np.ndarray:
        """The point of the block nearest P (for a hand or foot reaching to it)."""
        w, d, h = self.size
        q = self.local(P)
        if self.kind == "cylinder":
            r = w / 2
            v = np.array([q[0], 0, q[2]])
            n = float(np.linalg.norm(v))
            v = v / n * min(n, r) if n > 1e-9 else v
            v[1] = np.clip(q[1], 0, h)
            return self.world(v)
        if self.kind == "railing":
            return self.world([np.clip(q[0], -w / 2, w / 2), h, 0.0])
        if self.kind == "stairs":
            z = float(np.clip(q[2], 0, d))
            top = self.top_at(*self.world([0, 0, z])[[0, 2]]) - float(self.pos[1])
            return self.world([np.clip(q[0], -w / 2, w / 2), np.clip(q[1], 0, top), z])
        hh = 0.008 if self.kind == "plane" else h
        return self.world([np.clip(q[0], -w / 2, w / 2), np.clip(q[1], 0, hh), np.clip(q[2], 0, d)])


# ============================================================================================ props

PROP_KINDS = {"stick": 1.2, "sword": 0.95, "umbrella": 0.85, "bag": 0.3, "ball": 0.22, "book": 0.27, "cup": 0.1}


def hand_grip(body: Body, J, F, s: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grip point, finger direction, thumb direction) of hand s."""
    Hn = F[f"hand-{s}"]
    fing = unit(J[f"hand-{s}"] - J[f"wrist-{s}"])
    thumb = unit(Hn[:, 2] - np.dot(Hn[:, 2], fing) * fing)
    grip = J[f"wrist-{s}"] + fing * 0.45 * body.Lh
    return grip, fing, thumb


def prop_prims(kind: str, hand: str, length: Optional[float], body: Body, J, F, owner: str) -> List[dict]:
    L = float(length or PROP_KINDS[kind])
    if hand == "both":
        gr, fr_, tr = hand_grip(body, J, F, "r")
        gl, fl_, tl = hand_grip(body, J, F, "l")
        grip = (gr + gl) / 2
        axis = unit(tr + tl) if np.linalg.norm(tr + tl) > 0.2 else tr
        side = unit(gr - gl) if np.linalg.norm(gr - gl) > 1e-3 else fr_
    else:
        grip, fing, axis = hand_grip(body, J, F, hand)
        side = fing
    up = np.array([0, 1.0, 0])
    out: List[dict] = []

    def cyl(a, b, r, part):
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = unit(b - a)
        x = unit(np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0]))
        z = np.cross(x, u)
        out.append({"t": "cyl", "c": (a + b) / 2, "A": np.stack([x * r, u * float(np.linalg.norm(b - a)) / 2, z * r], 1),
                    "part": part, "group": owner})

    def box(c, ax_y, ax_x, half, part):
        uy = unit(ax_y)
        ux = unit(np.asarray(ax_x, float) - np.dot(ax_x, uy) * uy)
        if np.linalg.norm(ux) < 1e-6:
            ux = unit(np.cross(uy, [0, 0, 1.0]))
        uz = np.cross(ux, uy)
        out.append({"t": "box", "c": np.asarray(c, float), "A": np.stack([ux * half[0], uy * half[1], uz * half[2]], 1),
                    "part": part, "group": owner})
    if kind == "stick":
        cyl(grip - axis * 0.25 * L, grip + axis * 0.75 * L, 0.016, owner)
    elif kind == "sword":
        cyl(grip - axis * 0.12, grip + axis * 0.06, 0.018, f"{owner} hilt")
        box(grip + axis * 0.08, side, axis, [0.11, 0.015, 0.02], f"{owner} guard")
        box(grip + axis * (0.1 + L / 2), axis, side, [0.022, L / 2, 0.005], f"{owner} blade")
    elif kind == "umbrella":
        a2 = unit(axis + 1.5 * up)
        top = grip + a2 * L
        cyl(grip - a2 * 0.05, top, 0.012, f"{owner} shaft")
        x = unit(np.cross(a2, [0, 0, 1.0]) if abs(a2[2]) < 0.9 else np.cross(a2, [1.0, 0, 0]))
        z = np.cross(x, a2)
        out.append({"t": "ell", "c": top - a2 * 0.12, "A": np.stack([x * 0.55, a2 * 0.2, z * 0.55], 1),
                    "part": f"{owner} canopy", "group": owner})
    elif kind == "bag":
        box(grip - up * 0.2, up, side, [0.16, 0.13, 0.06], owner)
    elif kind == "ball":
        r = L / 2
        # in one hand the ball sits against the palm (the palm faces her body's middle when the arm hangs)
        c = grip if hand == "both" else grip + F[f"hand-{hand}"][:, 0] * (-1 if hand == "r" else 1) * r * 0.9
        out.append({"t": "ell", "c": c, "A": np.eye(3) * r, "part": owner, "group": owner})
    elif kind == "book":
        box(grip + axis * 0.05, axis, side, [0.1, L / 2, 0.018], owner)
    elif kind == "cup":
        cyl(grip - up * L * 0.5, grip + up * L * 0.5, 0.04, owner)
    for p in out:
        p["owner"] = owner
    return out


# ============================================================================================ IK

def two_bone(S, T, L1: float, L2: float, pole) -> Tuple[np.ndarray, np.ndarray, float]:
    """(middle joint, end joint, gap): a two-bone chain from S toward T bending toward ``pole``; gap > 0 is how far
    the target is out of reach (the chain then points straight at it)."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    d = T - S
    D0 = float(np.linalg.norm(d))
    u = d / D0 if D0 > 1e-9 else np.array([0, -1.0, 0])
    D = float(np.clip(D0, abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4))
    a = (L1 * L1 - L2 * L2 + D * D) / (2 * D)
    hh = math.sqrt(max(L1 * L1 - a * a, 0.0))
    pole = np.asarray(pole, float)
    pv = pole - np.dot(pole, u) * u
    if np.linalg.norm(pv) < 1e-6:
        pv = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 0, 1.0])
    pv = unit(pv)
    return S + u * a + pv * hh, S + u * D, D0 - (L1 + L2)


def ik_limb(body: Body, J, F, limb_name: str, target) -> float:
    """Move arm-r / arm-l / leg-r / leg-l so its hand's grip or its foot's sole reaches ``target``; frames follow.
    Returns the gap (m, > 0 = out of reach)."""
    kind, s = limb_name.split("-")
    if kind == "arm":
        T = np.asarray(target, float)
        off = unit(T - J[f"shoulder-{s}"]) * 0.45 * body.Lh
        gap = 0.0
        for _ in range(3):                  # the grip (not the wrist) goes to the target: the hand turns with the arm
            gap = _ik_once(body, J, F, kind, s, T - off)
            grip = J[f"wrist-{s}"] + unit(J[f"hand-{s}"] - J[f"wrist-{s}"]) * 0.45 * body.Lh
            off = grip - J[f"wrist-{s}"]
        return gap
    return _ik_once(body, J, F, kind, s, target)


def _ik_once(body: Body, J, F, kind: str, s: str, target) -> float:
    if kind == "arm":
        a, m, e, L1, L2 = f"shoulder-{s}", f"elbow-{s}", f"wrist-{s}", body.Lu, body.Lf
        fa, fm, fe = f"arm-{s}", f"elbow-{s}", f"hand-{s}"
        sg = 1.0 if s == "r" else -1.0
        C = F["chest"]
        pole = -C[:, 2] * 0.8 - C[:, 1] * 0.5 + C[:, 0] * sg * 0.6
        T = np.asarray(target, float)
    else:
        a, m, e, L1, L2 = f"hip-{s}", f"knee-{s}", f"ankle-{s}", body.Lt, body.Ls
        fa, fm, fe = f"leg-{s}", f"knee-{s}", f"foot-{s}"
        P = F["pelvis"]
        pole = P[:, 2] + P[:, 1] * 0.1
        T = np.asarray(target, float) + np.array([0, body.ah, 0])
    M, E, gap = two_bone(J[a], T, L1, L2, pole)
    old_u, old_f = unit(J[m] - J[a]), unit(J[e] - J[m])
    Ra = rot_between(old_u, unit(M - J[a]))
    F[fa] = Ra @ F[fa]
    Fm_old = F[fm]
    F[fm] = rot_between(Ra @ old_f, unit(E - M)) @ Ra @ Fm_old
    F[fe] = F[fm] @ Fm_old.T @ F[fe]
    J[m], J[e] = M, E
    if kind == "arm":
        J[f"hand-{s}"] = E + F[fe] @ [0, -body.Lh, 0]
    else:
        _feet(body, J, F, s)
    return gap


# ============================================================================================ solving a figure

def _support_at(blocks: Dict[str, Block], X: float, Z: float, below: float = 1e9) -> Tuple[float, str]:
    """Height and name of the surface under (X, Z) that is not above ``below``: the highest block top or the
    ground."""
    best, name = 0.0, "ground"
    for b in blocks.values():
        if b.contains(X, Z, 0.02):
            t = b.top_at(X, Z)
            if best < t <= below + 0.05:
                best, name = t, b.name
    return best, name


GROUP_WORDS = {"foot": "foot", "heel": "foot", "shin": "shin", "knee": "knee", "thigh": "thigh", "pelvis": "seat",
               "waist": "back", "chest": "back", "hand": "hand", "forearm": "forearm", "elbow": "elbow",
               "upperarm": "arm", "shoulder": "shoulder", "head": "head", "neck": "head"}


def contacts(prims: List[dict], blocks: Dict[str, Block], tol: float = 0.03) -> List[Tuple[str, str, np.ndarray]]:
    """(part, surface, contact point) for parts resting on the ground or a block top."""
    out = []
    seen = set()
    for pr in prims:
        y, pt = lowest(pr)
        sup, name = _support_at(blocks, pt[0], pt[2], y + tol)
        if y - sup <= tol and y - sup >= -0.25:
            base = pr["part"].split("-")[0]
            word = GROUP_WORDS.get(base, base)
            side = pr["part"].split("-")[1] if "-" in pr["part"] else ""
            nm = f"{word}-{side}" if side and word not in ("seat", "back", "head") else word
            if nm not in seen:
                seen.add(nm)
                out.append((nm, name, pt))
    return out


def _hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 4), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _dist_to_hull(q: np.ndarray, hull: np.ndarray) -> Tuple[float, np.ndarray]:
    if len(hull) == 1:
        return float(np.linalg.norm(q - hull[0])), hull[0]
    best, bp = 1e9, hull[0]
    inside = len(hull) >= 3
    n = len(hull)
    for i in range(n if len(hull) >= 3 else 1):
        a, b = hull[i], hull[(i + 1) % n]
        ab = b - a
        t = float(np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
        p = a + t * ab
        dd = float(np.linalg.norm(q - p))
        if dd < best:
            best, bp = dd, p
        if (ab[0] * (q[1] - a[1]) - ab[1] * (q[0] - a[0])) < 0:
            inside = False
    return (0.0, q) if inside else (best, bp)


def balance(body: Body, J, F, cts) -> str:
    c = com(body, J)
    if not cts:
        return "airborne"
    pts = []
    for nm, _, pt in cts:
        pts.append([pt[0], pt[2]])
        if nm.startswith("foot-"):          # a flat foot supports from heel to toe
            s = nm[-1]
            for j in (f"heel-{s}", f"foot-{s}"):
                if J[j][1] <= pt[1] + 0.04:
                    pts.append([J[j][0], J[j][2]])
    pts = np.array(pts)
    hull = _hull(pts)
    dist, near = _dist_to_hull(np.array([c[0], c[2]]), hull)
    if dist <= 0.05:
        return "balanced: the centre of mass is over the support"
    off = np.array([c[0] - near[0], 0, c[2] - near[1]])
    R = F["pelvis"]
    fwd = float(np.dot(off, unit([R[0, 2], 0, R[2, 2]])))
    rgt = float(np.dot(off, unit([R[0, 0], 0, R[2, 0]])))
    words = []
    if abs(fwd) > 0.03:
        words.append("ahead of" if fwd > 0 else "behind")
    if abs(rgt) > 0.03:
        words.append("to her right of" if rgt > 0 else "to her left of")
    return f"the centre of mass is {dist:.2f} m {' and '.join(words) or 'off'} the support (she would tip over " \
           f"unless moving: right for run, walk, lunge)"


def facing_words(yaw_view: float) -> str:
    a = (yaw_view + 180) % 360 - 180
    if abs(a) <= 15:
        return "faces the viewer"
    if abs(a) >= 165:
        return "back to the viewer"
    side = "right" if a > 0 else "left"
    if abs(a) <= 60:
        return f"three-quarter view, facing the picture's {side}"
    if abs(a) <= 120:
        return f"profile, facing the picture's {side}"
    return f"three-quarter back view, facing away to the picture's {side}"


def view_angles(cam: Camera, R: np.ndarray, at) -> Tuple[float, float]:
    """(yaw, pitch) of a part frame as the viewer sees it: yaw 0 faces the camera, + toward the picture's right;
    pitch + looks up."""
    fwd = unit(R[:, 2])
    toward = unit(cam.C - np.asarray(at, float))          # from the part to the camera
    upv = unit(cam.up - np.dot(cam.up, toward) * toward)
    right = np.cross(toward, upv)                          # the picture's right, seen from the part
    x, y, z = float(np.dot(fwd, right)), float(np.dot(fwd, upv)), float(np.dot(fwd, toward))
    return math.degrees(math.atan2(x, z)), math.degrees(math.asin(max(-1.0, min(1.0, y))))


def solve_figure(spec: dict, cam: Camera, blocks: Dict[str, Block], others: Dict[str, dict]) -> dict:
    """Pose a figure from its spec: {preset, edits, at [x,y units], height, heads, yaw, mirror, on, targets {limb:
    target}, lift}. ``others`` = solved figures by subject (targets may name one). Returns J, F, prims, body, notes,
    clamps, ik lines, contacts, balance, support."""
    pk = preset_key(spec.get("preset") or "stand")
    pre = PRESETS[pk]
    body = Body(float(spec.get("height") or 1.6), float(spec.get("heads") or 6.5))
    E = dict(pre["edits"])
    if spec.get("mirror"):
        E = mirror_edits(E)
    E = merge_edits(E, spec.get("edits") or {})
    E, clamps = clamp_edits(E)
    yaw = spec.get("yaw")
    if yaw is None:
        yaw = pre.get("yaw") or 0.0
        if spec.get("mirror") and pre.get("yaw"):
            yaw = -yaw
    yaw = float(yaw)
    J, F = fk(body, E, yaw)
    notes: List[str] = []
    on = spec.get("on")
    blk = blocks.get(on) if on else None
    if on and blk is None:
        notes.append(f"there is no block {on!r} (blocks: {', '.join(blocks) or 'none'}): standing on the ground")
    at = spec.get("at")
    # where the figure stands: --at seen on the support's top plane, or the block's top centre
    sup_y = 0.0
    if blk is not None:
        tc = blk.top_centre()
        if at is not None:
            g = cam.ground_point(at[0], at[1], tc[1])
            if g is None:
                raise UsageError(f"--at {at[0]:g},{at[1]:g} is above the top of {on} as seen from the camera; give "
                                 f"a point lower in the picture", f"pixelcanvas pose girl --on {on}")
            target = g
            sup_y = blk.top_at(g[0], g[2])
        else:
            target, sup_y = tc, tc[1]
    else:
        if at is None:
            at = [500.0, 950.0]
        g = cam.ground_point(at[0], at[1], 0.0)
        if g is None:
            raise UsageError(f"--at {at[0]:g},{at[1]:g} is not on the ground: it must be below the horizon "
                             f"(y > {cam.horizon:.0f})", f"pixelcanvas pose girl --at 500,{min(990, cam.horizon + 300):.0f}")
        target = g
    rest = pre["rest"]
    lift = float(spec.get("lift") if spec.get("lift") is not None else pre.get("lift", 0.0))
    prims = body_prims(body, J, F)
    if rest == "seat" and blk is not None:
        ys = [lowest(p)[0] for p in prims if p["part"] in ("pelvis",)]
        dy = sup_y - min(ys)
    else:
        dy = sup_y + (lift if rest == "air" or lift else 0.0) - min(lowest(p)[0] for p in prims)
    for k in J:
        J[k] = J[k] + [0, dy, 0]
    prims = body_prims(body, J, F)
    # the anchor in the ground plane: the middle of what rests on the support (the pelvis when seated or flying)
    if rest == "seat" and blk is not None or rest == "air" or lift > 0:
        anc = J["pelvis"]
    else:
        lows = [lowest(p) for p in prims]
        m = min(v for v, _ in lows)
        pts = np.array([pt for v, pt in lows if v <= m + 0.03])
        anc = pts.mean(0)
    dx, dz = target[0] - anc[0], target[2] - anc[2]
    for k in J:
        J[k] = J[k] + [dx, 0, dz]
    ik_lines: List[str] = []
    # seated with feet on a floor: the feet go down to what is below them
    if rest == "seat" and blk is not None and pre.get("plant"):
        for s in "rl":
            ank = J[f"ankle-{s}"]
            fy, fname = _support_at(blocks, ank[0], ank[2], J["pelvis"][1] - 0.15)
            floor = fy
            if abs((J[f"heel-{s}"][1]) - floor) > 0.02:
                tgt = np.array([ank[0], floor, ank[2]])
                gap = ik_limb(body, J, F, f"leg-{s}", tgt)
                if gap > 0.01:
                    ik_lines.append(f"foot-{s} does not reach the {fname} ({gap:.2f} m short): the legs hang")
    # standing with both feet down: a foot a little above the support is pulled down onto it
    if rest == "feet" and not lift and pre.get("plant"):
        for s in "rl":
            low = min(J[f"heel-{s}"][1], J[f"foot-{s}"][1])
            fy, _ = _support_at(blocks, J[f"ankle-{s}"][0], J[f"ankle-{s}"][2], low)
            if 0.01 < low - fy < 0.25:
                ik_limb(body, J, F, f"leg-{s}", J[f"ankle-{s}"] - [0, low - fy + body.ah, 0])
        under = sup_y - min(lowest(p)[0] for p in body_prims(body, J, F))
        if under > 0:                       # a planted foot's sole never goes into the ground
            for k in J:
                J[k] = J[k] + [0, under, 0]
    for limb_name, tgt in (spec.get("targets") or {}).items():
        T, what = _target_point(limb_name, tgt, J, cam, blocks, others)
        if T is None:
            ik_lines.append(f"{limb_name}: {what}")
            continue
        gap = ik_limb(body, J, F, limb_name.replace("hand", "arm").replace("foot", "leg"), T)
        ik_lines.append(f"{limb_name.replace('arm', 'hand').replace('leg', 'foot')} reaches {what}" +
                        (f": {gap:.2f} m out of reach, the arm points straight at it" if gap > 0.01 else ""))
    prims = body_prims(body, J, F)
    cts = contacts(prims, blocks)
    lowy = min(lowest(p)[0] for p in prims)
    bal = balance(body, J, F, cts)
    if bal == "airborne":
        bal = f"airborne: {lowy:.2f} m above the ground"
    return {"J": J, "F": F, "body": body, "prims": prims, "preset": pk, "edits": E, "clamps": clamps, "notes": notes,
            "ik": ik_lines, "contacts": cts, "balance": bal, "yaw": yaw, "support": on if blk is not None else "ground"}


def _target_point(limb_name, tgt, J, cam: Camera, blocks, others):
    s = limb_name.split("-")[1]
    root = J[f"shoulder-{s}"] if limb_name.startswith(("hand", "arm")) else J[f"hip-{s}"]
    if isinstance(tgt, (list, tuple)) and len(tgt) == 2:
        depth = float(cam.to_cam(root)[0, 2])
        return cam.point_at_depth(tgt[0], tgt[1], depth), f"{tgt[0]:g},{tgt[1]:g}"
    name = str(tgt)
    if name in blocks:
        b = blocks[name]
        if limb_name.startswith(("foot", "leg")):
            q = b.closest(root - [0, 2.0, 0])
            q = b.closest(np.array([J[f"ankle-{s}"][0], q[1], J[f"ankle-{s}"][2]]))
        else:
            q = b.closest(root)
        return q, f"the {name}"
    if name in others:
        o = others[name]["J"]
        cands = [o["hand-r"], o["hand-l"]]
        q = min(cands, key=lambda c: float(np.linalg.norm(c - root)))
        return q, f"{name}'s hand"
    return None, f"no block or figure {name!r}"


def compat_joints(cam: Camera, J) -> Tuple[Dict[str, List[float]], bool]:
    """The 14 joints of a ``sketch figure`` record (0-1000 units, l = picture left) and whether picture-left is her
    right side."""
    U = {k: cam.project_units([v])[0] for k, v in J.items()}
    dsh = U["shoulder-l"][0] - U["shoulder-r"][0]
    dhp = U["hip-l"][0] - U["hip-r"][0]
    left_is_r = (dsh if abs(dsh) > 1.0 else dhp) >= 0
    a, b = ("r", "l") if left_is_r else ("l", "r")
    out = {"head": U["head"], "chin": U["chin"]}
    names = [("shoulder", "shoulder"), ("elbow", "elbow"), ("wrist", "wrist"), ("hip", "hip"), ("knee", "knee"),
             ("ankle", "ankle")]
    for rec, j in names:
        out["l" + rec] = U[f"{j}-{a}"]
        out["r" + rec] = U[f"{j}-{b}"]
    order = ["head", "chin", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", "lhip", "rhip",
             "lknee", "rknee", "lankle", "rankle"]
    return {k: [round(float(out[k][0]), 1), round(float(out[k][1]), 1)] for k in order}, left_is_r
