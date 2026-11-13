"""People: posable, dressed figures in one call.

``Figure(height=0.6, build="average", view="three_quarter", hair=dict(style="bob"), outfit=dict(top="coat"))
.pose("wave").draw(cv, at=(0.5, 0.92))`` draws a person standing at ``at`` (the ground point under the body).

* A real skeleton (pelvis, curved spine, chest, neck, head, shoulders, elbows, wrists, hips, knees, ankles) with
  joint angles and forward kinematics; arms can also reach for targets (two-bone IK). Poses are joint-angle
  dictionaries (``POSES``) that interpolate; ``walk`` and ``run`` take a ``phase``.
* Proportions measured in heads (adult 7.4, child 5.2, chibi 2.8, heroic 8), seven builds, three body types.
* Tapered, curved limbs with muscle and clothing shape; a lofted torso; hands with palm, thumb and three-joint
  fingers in eight shapes; shoes with a direction and a sole on the ground; heads with skull, jaw, ears, eyes
  (iris, highlight, lids), brows, nose, mouth, eight expressions and a gaze; fifteen hair styles; clothing as shapes
  over the body with folds; accessories and hand-held items.
* Four rendering styles: ``flat`` (editorial vector), ``soft`` (lit volumes), ``anime`` (cel), ``line`` (ink).
* ``fig.anchor("hand_r")`` etc. give canvas pixels for props, balloons, speech bubbles; ``fig.mask`` / ``fig.bbox``.
* ``crowd(cv, n, region)``, ``random_person(seed)``, ``person("chef")`` / ``person("a tall woman in a red coat
  waving")`` and ``PEOPLE`` (ready-made characters).

Units: ``at`` is the ground point under the body as canvas fractions (u, v); ``height`` is a fraction of the canvas
height. At a perspective ``Spot`` (``ground.place`` / ``ground.spot``) the height is in world units (metres).
Sides: ``_l`` / ``_r`` are the figure's own left / right (the left hand is on screen right in a front view).
"""
from __future__ import annotations

import math
import re

import numpy as np

from .color import F32, rgb
from . import sdf as _sdf
from . import form as _form
from .form import check_frac as _check_frac
from .filters import gaussian_blur as _blur
from ._args import (check_choice as _check_choice, reject_unknown as _reject_unknown, unknown_name as _unknown_name,
                    signature_options as _sig_options, strict as _strict, strict_methods as _strict_methods)

__all__ = ["Figure", "POSES", "SEATS", "BUILDS", "BODY_TYPES", "PEOPLE", "HAIR_STYLES", "EXPRESSIONS", "TOPS", "BOTTOMS",
           "SHOES", "EXTRAS", "ITEMS", "HAND_SHAPES", "STYLES", "VIEWS", "SKIN_TONES", "HAIR_COLORS", "PALETTE",
           "person", "random_person", "crowd", "pose_lerp", "walk_pose", "run_pose"]

# ===================================================================== palettes

SKIN_TONES = {
    "porcelain": "#f6dccb", "fair": "#f0c9ad", "light": "#e7b591", "medium": "#d09b72", "tan": "#b97b52",
    "olive": "#a97a52", "brown": "#8d5b3c", "deep": "#6b412b", "ebony": "#4d2d1f",
}
HAIR_COLORS = {
    "black": "#1f1c22", "espresso": "#30231d", "dark_brown": "#432d20", "brown": "#6b452b", "auburn": "#8b3b24",
    "copper": "#b65a2c", "ginger": "#c8723c", "blonde": "#dcb46e", "platinum": "#ede2c8", "gray": "#a6a19c",
    "white": "#ebe8e3", "pink": "#ef93b4", "blue": "#5c86d6", "lilac": "#b49ae0", "teal": "#3aa6a0",
}
PALETTE = {
    "mustard": "#e3a72f", "coral": "#ec6b56", "teal": "#2a9d8f", "navy": "#26405f", "sage": "#8fb38a",
    "terracotta": "#c8553d", "cream": "#f1e6cf", "denim": "#4a6fa5", "plum": "#6d3b6b", "charcoal": "#383b45",
    "sky": "#8ec5e6", "red": "#d1404f", "olive": "#6f7550", "pink": "#f2a7b8", "white": "#f5f2ec", "black": "#26242b",
    "forest": "#2f5d46", "sand": "#d8c3a0", "rust": "#b5532b", "lavender": "#a89ad6", "yellow": "#f4cf45",
    "khaki": "#b8a77c", "brown": "#7a5236", "gray": "#9a9ca3", "blue": "#3d74c9", "green": "#4c9a5a",
    "orange": "#ef8a2c", "purple": "#7b4fb0", "maroon": "#7c2b37", "beige": "#e2d3b8", "tan": "#c29a6a",
}
_PROFILE = ((0.1, -0.42), (0.3, -0.36), (0.38, -0.18), (0.41, -0.04), (0.385, 0.03), (0.5, 0.17), (0.44, 0.215),
            (0.4, 0.235), (0.425, 0.285), (0.395, 0.317), (0.41, 0.35), (0.37, 0.395), (0.385, 0.45),
            (0.33, 0.51), (0.12, 0.47), (-0.06, 0.34), (-0.12, 0.1), (-0.1, -0.2))
_PROFILE_YOUNG = ((0.1, -0.42), (0.32, -0.34), (0.41, -0.15), (0.42, -0.02), (0.4, 0.06), (0.46, 0.18),
                  (0.41, 0.225), (0.42, 0.29), (0.4, 0.33), (0.4, 0.4), (0.33, 0.49), (0.14, 0.48), (-0.06, 0.36),
                  (-0.12, 0.1), (-0.1, -0.2))
_EYE_COLORS = ("#3a2618", "#4d3220", "#2f4a6a", "#3f6b4a", "#5a4630", "#24242c")
INK = "#2a2430"

# ===================================================================== tables

BUILDS = {
    "average": dict(heads=7.4, leg=0.52, shoulder=1.9, chest=1.56, waist=1.24, hip=1.46, limb=1.0, neck=1.0,
                    head_w=0.74, hand=1.0, foot=1.0, eye=1.18, eye_y=0.0, arm=1.0, stoop=0.0),
    "slim": dict(heads=7.6, leg=0.525, shoulder=1.76, chest=1.42, waist=1.08, hip=1.34, limb=0.86, neck=0.92,
                 head_w=0.73, hand=0.98, foot=0.98, eye=1.18, eye_y=0.0, arm=1.0, stoop=0.0),
    "broad": dict(heads=7.2, leg=0.505, shoulder=2.18, chest=1.98, waist=1.8, hip=1.72, limb=1.2, neck=1.12,
                  head_w=0.77, hand=1.08, foot=1.06, eye=1.14, eye_y=0.0, arm=1.0, stoop=0.0),
    "child": dict(heads=5.2, leg=0.45, shoulder=1.5, chest=1.36, waist=1.3, hip=1.32, limb=1.15, neck=0.9,
                  head_w=0.8, hand=1.1, foot=1.1, eye=1.3, eye_y=0.05, arm=1.0, stoop=0.0),
    "elder": dict(heads=7.0, leg=0.505, shoulder=1.76, chest=1.52, waist=1.4, hip=1.48, limb=0.95, neck=0.95,
                  head_w=0.75, hand=1.0, foot=1.0, eye=1.08, eye_y=0.01, arm=1.0, stoop=1.0),
    "chibi": dict(heads=2.8, leg=0.33, shoulder=1.08, chest=0.98, waist=0.94, hip=1.0, limb=1.5, neck=0.8,
                  head_w=0.94, hand=1.5, foot=1.3, eye=1.9, eye_y=0.1, arm=0.78, stoop=0.0),
    "heroic": dict(heads=8.0, leg=0.53, shoulder=2.36, chest=1.94, waist=1.26, hip=1.42, limb=1.16, neck=1.15,
                   head_w=0.7, hand=1.0, foot=1.0, eye=1.06, eye_y=0.0, arm=1.0, stoop=0.0),
}
BODY_TYPES = {
    "neutral": dict(),
    "feminine": dict(shoulder=0.9, chest=0.95, waist=0.84, hip=1.08, limb=0.88, neck=0.84, head_w=0.97, hand=0.9,
                     foot=0.9, bust=1.0, lashes=1.0),
    "masculine": dict(shoulder=1.05, chest=1.05, waist=1.04, hip=0.95, limb=1.06, neck=1.08, head_w=1.02,
                      hand=1.06, foot=1.05, jaw=1.0),
}
HAIR_STYLES = ("short", "crop", "buzz", "side_part", "spiky", "bob", "pixie", "long", "ponytail", "bun", "curly",
               "afro", "braids", "bald", "receding")
EXPRESSIONS = ("neutral", "smile", "laugh", "surprised", "sad", "angry", "wink", "sleepy")
TOPS = ("tshirt", "shirt", "blouse", "hoodie", "sweater", "jacket", "coat", "suit", "dress", "tank", "spacesuit")
BOTTOMS = ("trousers", "jeans", "shorts", "skirt", "leggings")
SHOES = ("sneakers", "boots", "shoes", "flats")
EXTRAS = ("glasses", "sunglasses", "hat", "cap", "beanie", "beret", "chef_hat", "helmet", "headphones", "scarf",
          "bag", "backpack", "tie", "apron", "stethoscope", "beard", "mustache")
ITEMS = ("cup", "phone", "book", "bag", "briefcase", "guitar", "ball", "flower")
# what a sitting figure sits on: "auto" = a stool when drawn at a plain (u, v) point, nothing at a scene Spot
# (the room supplies the chair); "none" = seated, nothing drawn (your own chair, sofa, wall or step)
SEATS = ("auto", "stool", "bench", "chair", "none")
HAND_SHAPES = ("relaxed", "open", "fist", "point", "grip", "wave", "peace", "thumbs_up")
STYLES = ("flat", "soft", "anime", "line")
VIEWS = {"front": 0.0, "three_quarter": 35.0, "side": 90.0, "back": 180.0, "three_quarter_back": 145.0}

# ===================================================================== small math


def _rx(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_MIR = np.diag([-1.0, 1.0, 1.0])


def _mir(R, side):
    return R if side > 0 else _MIR @ R @ _MIR


def _v(x, y, z):
    return np.array([x, y, z], float)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0


def _smin(a, b, k):
    return _sdf.smooth_union(a, b, max(float(k), 1e-3))


def _cone(X, Y, a, b, ra, rb):
    """Signed distance (px) to a tapered capsule from a (radius ra) to b (radius rb)."""
    vx, vy = b[0] - a[0], b[1] - a[1]
    h = math.hypot(vx, vy)
    if h < 1e-6 or abs(ra - rb) >= h:
        if ra >= rb:
            return (np.hypot(X - a[0], Y - a[1]) - ra).astype(F32)
        return (np.hypot(X - b[0], Y - b[1]) - rb).astype(F32)
    ux, uy = vx / h, vy / h
    px_, py_ = X - a[0], Y - a[1]
    qy = px_ * ux + py_ * uy
    qx = np.abs(px_ * uy - py_ * ux)
    bb = (ra - rb) / h
    aa = math.sqrt(1 - bb * bb)
    k = -qx * bb + qy * aa
    d_mid = qx * aa + qy * bb - ra
    d_a = np.hypot(qx, qy) - ra
    d_b = np.hypot(qx, qy - h) - rb
    return np.where(k < 0, d_a, np.where(k > aa * h, d_b, d_mid)).astype(F32)


def _chain(X, Y, pts, radii, k):
    d = None
    for i in range(len(pts) - 1):
        e = _cone(X, Y, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = e if d is None else _smin(d, e, k)
    return d


def _ell(X, Y, cx, cy, rx, ry, angle=0.0):
    return _form.sd_ellipse(X, Y, cx, cy, max(rx, 1e-3), max(ry, 1e-3), angle)


def _circ(X, Y, cx, cy, r):
    return (np.hypot(X - cx, Y - cy) - r).astype(F32)


def _plane(X, Y, p, n):
    """Half-plane distance: negative on the side opposite to the 2D unit normal n from point p."""
    return ((X - p[0]) * n[0] + (Y - p[1]) * n[1]).astype(F32)


def _hull(pts):
    pts = sorted(set((round(float(x), 3), round(float(y), 3)) for x, y in pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _convex_sd(X, Y, hull):
    """Signed distance to a convex polygon (exact inside and along edges; corners slightly sharp outside)."""
    n = len(hull)
    if n < 3:
        if n == 0:
            return np.full(X.shape, 1e6, F32)
        a = hull[0]
        b = hull[-1]
        return _cone(X, Y, a, b, 0.5, 0.5)
    area = sum(hull[i][0] * hull[(i + 1) % n][1] - hull[(i + 1) % n][0] * hull[i][1] for i in range(n))
    sg = 1.0 if area > 0 else -1.0
    d = None
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        L = math.hypot(ex, ey)
        if L < 1e-9:
            continue
        nx, ny = sg * ey / L, -sg * ex / L
        e = (X - ax) * nx + (Y - ay) * ny
        d = e if d is None else np.maximum(d, e)
    if d is None:
        return np.full(X.shape, 1e6, F32)
    return d.astype(F32)


def _loft_fn(rings, k, r=0.0):
    """Distance function of a lofted solid: the union of hulls of consecutive projected rings (px)."""
    def lerp_ring(a, b, f):
        return [(p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f) for p, q in zip(a, b)]

    hulls = []
    n = len(rings)
    for i in range(n - 1):
        pts = list(rings[i]) + list(rings[i + 1])
        if i > 0:
            pts += lerp_ring(rings[i], rings[i - 1], 0.35)
        if i + 2 < n:
            pts += lerp_ring(rings[i + 1], rings[i + 2], 0.35)
        hulls.append(_hull(pts))

    def fn(X, Y):
        d = None
        for h in hulls:
            e = _convex_sd(X, Y, h)
            d = e if d is None else _smin(d, e, k)
        return d - r
    return fn


def _catmull(pts, n=4):
    """Smooth an open polyline through its points (Catmull-Rom), n samples per segment."""
    P_ = [np.asarray(p, float) for p in pts]
    if len(P_) < 3:
        return [tuple(p) for p in P_]
    out = []
    ext = [P_[0] * 2 - P_[1]] + P_ + [P_[-1] * 2 - P_[-2]]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for j in range(n):
            t = j / n
            t2, t3 = t * t, t * t * t
            q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append((float(q[0]), float(q[1])))
    out.append((float(P_[-1][0]), float(P_[-1][1])))
    return out


def _polyline_sd(X, Y, pts, r):
    """Distance to a polyline with half-width r (px)."""
    d = None
    for a, b in zip(pts, pts[1:]):
        e = _cone(X, Y, a, b, r, r)
        d = e if d is None else np.minimum(d, e)
    return d


def _col(c):
    return rgb(c).astype(F32)


def _mixc(a, b, t):
    return (_col(a) * (1 - t) + _col(b) * t).astype(F32)


def _rng(seed):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _smooth01(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

# ===================================================================== poses


_REST = dict(
    lean=0.0, bend=0.0, twist=0.0, hip_tilt=0.0, pelvis_turn=0.0, nod=0.0, turn=0.0, tilt=0.0, lift=0.0, roll=0.0,
    shoulder_l=(0.0, 8.0, 0.0), shoulder_r=(0.0, 8.0, 0.0), elbow_l=14.0, elbow_r=14.0, wrist_l=0.0, wrist_r=0.0,
    turn_l=0.0, turn_r=0.0, hip_l=(0.0, 3.0, 0.0), hip_r=(0.0, 3.0, 0.0), knee_l=3.0, knee_r=3.0,
    ankle_l=0.0, ankle_r=0.0, toe_out=10.0,
)
_NONNUM = dict(hand_l="relaxed", hand_r="relaxed", palm_l=None, palm_r=None, reach_l=None, reach_r=None,
               pole_l=None, pole_r=None, face=None, item=None, seat=None, seat_color=None, hold_both=None,
               hold_width=None)

POSES = {
    "stand": dict(hip_tilt=-4.0, bend=3.0, hip_l=(7.0, 8.0, 8.0), hip_r=(0.0, -1.5, 0.0), knee_l=13.0, knee_r=2.0,
                  ankle_l=7.0, shoulder_l=(-4.0, 9.0, 0.0), shoulder_r=(4.0, 7.0, 0.0), elbow_l=12.0, elbow_r=18.0,
                  tilt=2.5, nod=-2.0),
    "sit": dict(hip_l=(86.0, 7.0, 0.0), hip_r=(84.0, 5.0, 0.0), knee_l=86.0, knee_r=80.0, ankle_l=0.0, ankle_r=4.0,
                lean=5.0, reach_l=("knee_l", (-0.1, 0.18, -0.25)), reach_r=("knee_r", (0.1, 0.18, -0.25)),
                pole_l=(0.7, -0.4, -0.5), pole_r=(-0.7, -0.4, -0.5),
                palm_l="down", palm_r="down", seat="auto", nod=3.0),
    "wave": dict(hip_tilt=-3.0, bend=2.0, hip_l=(5.0, 7.0, 6.0), hip_r=(0.0, -1.0, 0.0), knee_l=10.0, knee_r=2.0,
                 ankle_l=5.0, shoulder_r=(16.0, 112.0, 60.0), elbow_r=70.0, hand_r="wave", palm_r="front",
                 shoulder_l=(-2.0, 8.0, 0.0), elbow_l=14.0, tilt=-5.0, face="smile"),
    "point": dict(twist=10.0, hip_l=(5.0, 6.0, 0.0), knee_l=9.0, ankle_l=4.0, shoulder_r=(80.0, 26.0, 0.0),
                  elbow_r=6.0, hand_r="point", palm_r="in", shoulder_l=(0.0, 9.0, 0.0), elbow_l=14.0, turn=6.0),
    "reach": dict(lean=-6.0, shoulder_r=(160.0, 16.0, 0.0), elbow_r=6.0, hand_r="open", palm_r="front",
                  shoulder_l=(20.0, 18.0, 0.0), elbow_l=30.0, ankle_l=-26.0, ankle_r=-26.0, nod=-18.0,
                  hip_l=(2.0, 3.0, 0.0), hip_r=(0.0, 3.0, 0.0)),
    "hold": dict(hip_tilt=-3.0, hip_l=(5.0, 7.0, 5.0), knee_l=9.0, ankle_l=4.0,
                 reach_r=("chest", (-0.1, -0.7, 0.62)), pole_r=(-0.8, -0.5, -0.2), hand_r="grip", palm_r="in",
                 item="cup", nod=6.0),
    "jump": dict(lift=0.75, hip_l=(48.0, 9.0, 0.0), hip_r=(22.0, 6.0, 0.0), knee_l=84.0, knee_r=70.0,
                 ankle_l=-30.0, ankle_r=-34.0, shoulder_l=(16.0, 148.0, 0.0), shoulder_r=(16.0, 148.0, 0.0),
                 elbow_l=18.0, elbow_r=18.0, hand_l="open", hand_r="open", palm_l="front", palm_r="front",
                 face="laugh", nod=-8.0),
    "dance": dict(hip_tilt=8.0, bend=-9.0, twist=-6.0, hip_l=(28.0, 18.0, 10.0), hip_r=(0.0, -2.0, 0.0),
                  knee_l=46.0, knee_r=6.0, ankle_l=-20.0, shoulder_l=(20.0, 148.0, 0.0), elbow_l=34.0,
                  shoulder_r=(30.0, 60.0, 30.0), elbow_r=80.0, hand_l="open", hand_r="open", palm_l="front",
                  tilt=8.0, face="laugh"),
    "kneel": dict(hip_l=(90.0, 6.0, 0.0), knee_l=92.0, ankle_l=2.0, hip_r=(-4.0, 4.0, 0.0), knee_r=94.0,
                  ankle_r=-60.0, lean=4.0, reach_l=("knee_l", (0.0, 0.16, 0.05)), pole_l=(0.8, -0.2, -0.5),
                  hand_l="relaxed", palm_l="down", shoulder_r=(6.0, 9.0, 0.0), elbow_r=18.0),
    "lean": dict(roll=-6.0, hip_tilt=-6.0, bend=-4.0, hip_l=(8.0, -9.0, 8.0), hip_r=(0.0, 4.0, 0.0), knee_l=14.0,
                 ankle_l=8.0, reach_l=("chest", (-0.36, -0.42, 0.52)), reach_r=("chest", (0.36, -0.36, 0.56)),
                 hand_l="relaxed", hand_r="relaxed", palm_l="down", palm_r="down", tilt=6.0),
    "arms_crossed": dict(hip_tilt=-3.0, bend=2.0, hip_l=(4.0, 7.0, 4.0), knee_l=8.0, ankle_l=4.0,
                         reach_l=("chest", (-0.42, -0.36, 0.58)), reach_r=("chest", (0.44, -0.28, 0.5)),
                         pole_l=(0.7, -0.7, -0.1), pole_r=(-0.7, -0.7, -0.1), hand_l="relaxed",
                         hand_r="relaxed", palm_l="in", palm_r="in", nod=-4.0, tilt=3.0),
    "hands_on_hips": dict(hip_tilt=-4.0, bend=3.0, hip_l=(4.0, 9.0, 6.0), hip_r=(0.0, 2.0, 0.0), knee_l=8.0,
                          ankle_l=4.0, reach_l=("hip_side_l", (0.1, 0.06, -0.04)), reach_r=("hip_side_r", (-0.1, 0.06, -0.04)),
                          pole_l=(1.0, 0.2, -0.5), pole_r=(-1.0, 0.2, -0.5), hand_l="fist", hand_r="fist",
                          palm_l="in", palm_r="in", wrist_l=0.0, wrist_r=0.0, nod=-4.0),
    "thinking": dict(hip_tilt=-3.0, hip_l=(4.0, 6.0, 4.0), knee_l=10.0, ankle_l=5.0,
                     reach_r=("chin", (-0.04, -0.2, 0.08)), pole_r=(-0.3, -1.0, 0.1), hand_r="relaxed",
                     palm_r="back", reach_l=("chest", (0.3, -0.52, 0.42)), pole_l=(0.6, -0.8, 0.0),
                     hand_l="relaxed", palm_l="up", tilt=-7.0, nod=5.0, face="neutral"),
    "cheer": dict(lift=0.1, shoulder_l=(10.0, 148.0, 0.0), shoulder_r=(10.0, 148.0, 0.0), elbow_l=26.0,
                  elbow_r=26.0, hand_l="fist", hand_r="fist", palm_l="front", palm_r="front", hip_l=(6.0, 8.0, 0.0),
                  hip_r=(4.0, 8.0, 0.0), knee_l=8.0, knee_r=8.0, ankle_l=-14.0, ankle_r=-14.0, nod=-12.0,
                  face="laugh"),
    "talk": dict(hip_tilt=-3.0, bend=2.0, hip_l=(4.0, 7.0, 4.0), knee_l=9.0, ankle_l=4.0,
                 reach_r=("chest", (-0.24, -0.82, 0.72)), pole_r=(-0.7, -0.7, -0.2), hand_r="open", palm_r="up",
                 shoulder_l=(0.0, 8.0, 0.0), elbow_l=16.0, face="smile", tilt=-3.0),
    "phone": dict(hip_tilt=-3.0, hip_l=(4.0, 6.0, 4.0), knee_l=8.0, ankle_l=4.0,
                  reach_r=("chest", (0.0, -0.38, 0.66)), pole_r=(-0.5, -1.0, 0.0), hand_r="grip", palm_r="up",
                  shoulder_l=(0.0, 8.0, 0.0), elbow_l=14.0, item="phone", nod=22.0),
    "walk": dict(),
    "run": dict(),
}
# the musician holds a guitar across the body
POSES["guitar"] = dict(reach_r=("pelvis", (-0.12, 0.5, 0.75)), reach_l=("chest", (0.85, -0.35, 0.55)),
                       pole_l=(0.6, -0.8, -0.2), pole_r=(-0.5, -0.9, -0.1), hand_r="relaxed", hand_l="grip",
                       palm_l="up", palm_r="back", item="guitar", hip_l=(4.0, 7.0, 4.0), knee_l=8.0, ankle_l=4.0,
                       hip_tilt=-3.0, tilt=-4.0)


def walk_pose(phase=0.0, stride=0.8):
    """Joint angles of a walk cycle at ``phase`` (0~1; 0 = legs passing, 0.25 = left heel strike)."""
    a = math.sin(2 * math.pi * phase)
    b = math.cos(2 * math.pi * phase)
    p = {}
    for side, sg in (("l", 1.0), ("r", -1.0)):
        sw = a * sg
        swing = max(0.0, b * sg)                       # leg swinging forward
        fl = 25.0 * stride * sw + 2
        kn = 6.0 + 46.0 * stride * swing ** 1.6 + 8.0 * max(0.0, -sw)
        pitch = 14.0 * max(0.0, sw) - 22.0 * max(0.0, -sw) * (1 - swing)
        p["hip_" + side] = (fl, 3.0, 0.0)
        p["knee_" + side] = kn
        p["ankle_" + side] = kn - fl + pitch
        p["shoulder_" + side] = (-22.0 * stride * sw, 7.0, 0.0)
        p["elbow_" + side] = 16.0 + 24.0 * max(0.0, -sw)
    p.update(lean=4.0, pelvis_turn=6.0 * a, twist=-10.0 * a, hip_tilt=3.0 * b, nod=-2.0)
    return p


_RUN_KEYS = ((0.0, 34.0, 105.0, -25.0), (90.0, 52.0, 62.0, 14.0), (180.0, 4.0, 28.0, 24.0),
             (270.0, -24.0, 56.0, -32.0), (315.0, -8.0, 92.0, -35.0))


def _cyc(th, keys):
    """Periodic smooth interpolation of key rows (angle, values...) at angle th (degrees)."""
    ks = list(keys) + [(keys[0][0] + 360.0,) + tuple(keys[0][1:])]
    for k0, k1 in zip(ks, ks[1:]):
        if k0[0] <= th <= k1[0]:
            t = (th - k0[0]) / (k1[0] - k0[0])
            t = t * t * (3 - 2 * t)
            return tuple(x + (y - x) * t for x, y in zip(k0[1:], k1[1:]))
    return tuple(keys[0][1:])


def run_pose(phase=0.0, stride=0.9):
    """Joint angles of a run cycle at ``phase`` (0~1)."""
    a = math.sin(2 * math.pi * phase)
    b = math.cos(2 * math.pi * phase)
    p = {}
    sc = stride / 0.9
    for side, sg in (("l", 1.0), ("r", -1.0)):
        sw = a * sg                                    # +1 leg forward, -1 leg behind
        th = (phase + (0.0 if sg > 0 else 0.5)) % 1.0 * 360.0
        # leg key poses around the cycle (degrees of the leg's own phase): 0 tucked and passing forward,
        # 90 reaching forward in flight, 180 support under the body, 270 toe-off behind, 315 heel kicked up
        fl, kn, an = _cyc(th, _RUN_KEYS)
        fl = 6.0 + (fl - 6.0) * sc
        kn = 20.0 + (kn - 20.0) * sc
        p["hip_" + side] = (fl, 4.0, 0.0)
        p["knee_" + side] = kn
        p["ankle_" + side] = an
        p["shoulder_" + side] = (-46.0 * stride * sw + 8, 10.0, 0.0)
        p["elbow_" + side] = 92.0 + 8.0 * sw
        p["hand_" + side] = "fist"
        p["palm_" + side] = "in"
    # airborne at the extremes of the stride, lowest while a leg passes under the body
    p.update(lean=17.0, pelvis_turn=8.0 * a, twist=-14.0 * a, lift=0.16 * stride * abs(a), nod=-10.0)
    return p


def _resolve_pose(name, phase=0.0):
    if isinstance(name, dict):
        return dict(name)
    if name == "walk":
        return walk_pose(phase)
    if name == "run":
        return run_pose(phase)
    if name not in POSES:
        raise _unknown_name("pose", name, list(POSES) + ["walk", "run"], "Try figure.POSES for the list")
    return dict(POSES[name])


def _full_pose(p):
    out = dict(_REST)
    out.update(_NONNUM)
    out.update(p)
    return out


def pose_lerp(a, b, t, phase=0.0):
    """Blend two poses (names or dicts) ??a pose dict; numbers and angle tuples interpolate, the rest switch at 0.5."""
    pa, pb = _full_pose(_resolve_pose(a, phase)), _full_pose(_resolve_pose(b, phase))
    out = {}
    for k in set(pa) | set(pb):
        va, vb = pa.get(k), pb.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = va + (vb - va) * t
        elif isinstance(va, tuple) and isinstance(vb, tuple) and len(va) == len(vb) and \
                all(isinstance(x, (int, float)) for x in va + vb):
            out[k] = tuple(x + (y - x) * t for x, y in zip(va, vb))
        else:
            out[k] = vb if t >= 0.5 else va
    return out


# ===================================================================== painter

_TONE = {   # shadow multipliers per kind and style
    "flat": dict(skin=(0.9, 0.77, 0.76), hair=(0.7, 0.68, 0.78), cloth=(0.8, 0.78, 0.88), dark=(0.72, 0.72, 0.8),
                 shoe=(0.72, 0.72, 0.8)),
    "anime": dict(skin=(0.93, 0.76, 0.78), hair=(0.64, 0.62, 0.8), cloth=(0.75, 0.73, 0.88), dark=(0.64, 0.64, 0.76),
                  shoe=(0.66, 0.66, 0.78)),
    "line": dict(skin=(0.95, 0.88, 0.87), hair=(0.84, 0.82, 0.88), cloth=(0.88, 0.86, 0.93), dark=(0.82, 0.82, 0.88),
                 shoe=(0.85, 0.85, 0.9)),
    "soft": dict(skin=(0.86, 0.72, 0.7), hair=(0.7, 0.68, 0.78), cloth=(0.8, 0.78, 0.88), dark=(0.72, 0.72, 0.8),
                 shoe=(0.72, 0.72, 0.8)),
}
_MAT = dict(skin=dict(preset="skin", ks=0.07, sss=0.35, rim=0.05, wrap=0.45),
            hair=dict(preset="matte", ks=0.25, shin=26, rim=0.04, wrap=0.3),
            cloth=dict(preset="fabric", sheen=0.12, wrap=0.5), dark=dict(preset="matte", ks=0.08, wrap=0.4),
            shoe=dict(preset="plastic", ks=0.22, window=0.08, wrap=0.3), metal="metal", glass="glass")
_SHK = dict(flat=0.6, anime=0.72, line=0.5, soft=0.0)


class _Painter:
    """Paints the parts of one figure into a private RGBA buffer (one paste per figure)."""

    def __init__(self, box, style, light, hu, line=None):
        self.x0, self.y0, self.x1, self.y1 = box
        self.w, self.h = self.x1 - self.x0, self.y1 - self.y0
        self.col = np.zeros((self.h, self.w, 3), F32)
        self.alpha = np.zeros((self.h, self.w), F32)
        self.style = style
        self.light = light
        self.l = light.dir2
        self.hu = hu
        base_w = {"line": 0.034, "anime": 0.02}.get(style, 0.0)
        self.lw = max(0.9, hu * base_w) * (line if line else 1.0) if base_w else 0.0
        self.ink = _col(INK)

    def win(self, pts, pad):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        a = max(int(math.floor(min(xs) - pad)), self.x0)
        b = max(int(math.floor(min(ys) - pad)), self.y0)
        c = min(int(math.ceil(max(xs) + pad)) + 1, self.x1)
        d = min(int(math.ceil(max(ys) + pad)) + 1, self.y1)
        if a >= c or b >= d:
            return None
        X, Y = np.meshgrid(np.arange(a, c, dtype=F32) + 0.5, np.arange(b, d, dtype=F32) + 0.5)
        return (a - self.x0, b - self.y0, c - self.x0, d - self.y0), X, Y

    def _comp(self, sl, col, A):
        a0, b0, a1, b1 = sl
        sc = self.col[b0:b1, a0:a1]
        sa = self.alpha[b0:b1, a0:a1]
        sc *= (1 - A)[..., None]
        sc += col * A[..., None]
        sa += A * (1 - sa)

    def tone(self, base, kind):
        t = _TONE.get(self.style, _TONE["flat"]).get(kind, _TONE["flat"]["cloth"])
        return (_col(base) * np.array(t, F32)).astype(F32)

    def ink_for(self, base):
        if self.style == "anime":
            return (_col(base) * 0.32 + self.ink * 0.55).astype(F32)
        return self.ink

    def part(self, fn, pts, pad, color, kind="cloth", r=None, outline=True, shade=True, cast=0.35, opacity=1.0,
             clip=None, details=None, edge=None, edge_w=None, light_k=1.0, noline=None):
        """Paint one part (signed px distance ``fn(X, Y)``) in the figure style. Returns (D, A, sl, X, Y) or None.

        r: typical half-width (px) for the shading crescent; cast: soft shadow thrown onto what is already painted
        (fraction of r); clip: a distance fn limiting the part; details(X, Y, D, A, sl) paints on top before the
        outline; edge: colour of a thin inner edge line (flat style finger / fold separation)."""
        r = float(r if r else self.hu * 0.2)
        kc = r * cast
        w = self.win(pts, pad + kc + self.lw + 2)
        if w is None:
            return None
        sl, X, Y = w
        D = fn(X, Y)
        if clip is not None:
            D = np.maximum(D, clip(X, Y))
        A = _sdf.fill(D)
        if float(A.max()) <= 0.0:
            return None
        base = _col(color)
        st = self.style
        lx, ly = self.l
        if st == "soft" and shade:
            # a wide blur irons out the creases where the lofted hulls meet (they showed as horizontal bands)
            Dsm = _blur(D, max(r * 0.32, 1.0)) if r > 6 else D
            col = _form.shade(Dsm, base, light=self.light, material=_MAT.get(kind, _MAT["cloth"]), profile="round",
                              width=max(r * 1.1, 2.0))
        else:
            col = np.broadcast_to(base, A.shape + (3,)).astype(F32)
            if shade:
                # cap the shadow crescent: on a torso a full-width band read as a second, darker garment
                k = min(r * _SHK[st], self.hu * 0.24) * light_k
                Ds = fn(X - lx * k, Y - ly * k)
                if clip is not None:
                    Ds = np.maximum(Ds, clip(X - lx * k, Y - ly * k))
                s = 1 - _sdf.fill(Ds, 1.0 if st != "line" else 1.4)
                col = col + (self.tone(base, kind) - col) * s[..., None]
                if st == "anime" and kind in ("skin", "hair", "cloth"):
                    kh = r * 0.42
                    Dh = fn(X + lx * kh, Y + ly * kh)
                    hl = (1 - _sdf.fill(Dh)) * (1 - s) * (0.16 if kind != "hair" else 0.0)
                    col = col + (1.0 - col) * hl[..., None]
        if cast > 0 and st != "line":
            a0, b0, a1, b1 = sl
            prev = self.alpha[b0:b1, a0:a1]
            if float(prev.max()) > 0:
                Dc = fn(X + lx * kc, Y + ly * kc)
                if clip is not None:
                    Dc = np.maximum(Dc, clip(X + lx * kc, Y + ly * kc))
                # soft-edged so a hem's shadow reads as shade, not as a ghost copy of the hem
                soft = max(kc * (0.5 if st == "soft" else 0.3), 1.0)
                S = _sdf.fill(Dc, soft) * (1 - A) * prev * (0.5 if st != "anime" else 0.6)
                mult = np.array(_TONE[st]["dark"], F32)
                self.col[b0:b1, a0:a1] *= (1 - S[..., None] * (1 - mult))
        A = A * float(opacity)
        self._comp(sl, col, A)
        if edge is not None and st in ("flat", "soft"):
            ew = edge_w if edge_w else max(0.6, self.hu * 0.012)
            E = _sdf.fill(np.abs(D + ew * 0.5) - ew * 0.5) * A
            self._comp(sl, np.broadcast_to(_col(edge), A.shape + (3,)), E * 0.8)
        if details is not None:
            details(X, Y, D, A, sl)
        if outline and st in ("line", "anime") and self.lw > 0:
            O = _sdf.fill(np.abs(D) - self.lw / 2)
            if noline is not None:
                O = O * (1 - _sdf.fill(noline(X, Y) + self.lw))
            self._comp(sl, np.broadcast_to(self.ink_for(color), A.shape + (3,)), O * float(opacity))
        return D, A, sl, X, Y

    def flat(self, fn, pts, pad, color, opacity=1.0, aa=1.0, clip=None):
        """Plain anti-aliased fill (details: eyes, mouths, seams). Returns the coverage or None."""
        w = self.win(pts, pad + 2)
        if w is None:
            return None
        sl, X, Y = w
        D = fn(X, Y)
        if clip is not None:
            D = np.maximum(D, clip(X, Y))
        A = _sdf.fill(D, aa) * float(opacity)
        if float(A.max()) <= 0:
            return None
        c = color if (isinstance(color, np.ndarray) and color.ndim == 3) else \
            np.broadcast_to(_col(color), A.shape + (3,))
        self._comp(sl, c, A)
        return A

    def stroke(self, fn, pts, pad, color, width, opacity=1.0, clip=None):
        """A line of ``width`` px along the zero set of a distance fn."""
        return self.flat(lambda X, Y: np.abs(fn(X, Y)) - width / 2, pts, pad + width, color, opacity, clip=clip)

    def shade_band(self, fn, pts, pad, color, opacity=1.0, clip=None, soft=1.0):
        """Multiply a tone over what is painted (folds, cast shadows inside the figure)."""
        w = self.win(pts, pad + 2)
        if w is None:
            return
        sl, X, Y = w
        D = fn(X, Y)
        if clip is not None:
            D = np.maximum(D, clip(X, Y))
        S = _sdf.fill(D, soft) * float(opacity)
        a0, b0, a1, b1 = sl
        S = S * self.alpha[b0:b1, a0:a1]
        self.col[b0:b1, a0:a1] *= (1 - S[..., None] * (1 - _col(color)))


def _norm_colors(c):
    if not c:
        return {}
    if isinstance(c, dict):
        return dict(c)
    keys = ("top", "bottom", "shoes", "accent", "inner")
    return {k: v for k, v in zip(keys, c)}


_COLOR_KEYS = ("top", "bottom", "shoes", "accent", "inner", "hat", "scarf", "bag", "apron", "book", "cup", "flower",
               "glasses", "guitar")


def _color_name(c):
    if isinstance(c, str):
        k = c.lower().replace(" ", "_")
        if k in PALETTE:
            return PALETTE[k]
        if k in HAIR_COLORS:
            return HAIR_COLORS[k]
        if k in SKIN_TONES:
            return SKIN_TONES[k]
        try:
            rgb(c)
        except ValueError:
            raise _unknown_name("colour", c, list(PALETTE) + list(HAIR_COLORS), "Use a name from figure.PALETTE / "
                                "figure.HAIR_COLORS or a hex colour like '#c8553d'.") from None
    return c


# ===================================================================== figure


_PALMS = (None, "side", "in", "out", "front", "back", "down", "up")


class _AnchorError(ValueError, KeyError):
    """Unknown anchor name (a ValueError with a did-you-mean; still caught by ``except KeyError``)."""

    def __str__(self):
        return self.args[0]


class _UnknownJoint(TypeError, ValueError):
    """An unknown joint keyword of ``Figure.pose``."""


class Figure:
    """A posable, dressed person. ``Figure(...).pose("walk", phase=0.3).draw(cv, at=(u, v))``.

    height: standing height as a fraction of the canvas height (at a ground ``Spot``: world units, metres);
    build: average | slim | broad | child | elder | chibi | heroic; body: neutral | feminine | masculine;
    view: front | three_quarter | side | back | three_quarter_back (or degrees); facing: 1 = towards screen
    right, -1 = towards screen left; skin: colour or a ``SKIN_TONES`` name; hair: dict(style, color);
    outfit: dict(top, bottom, shoes, colors=dict(top, bottom, shoes, accent, inner, hat, scarf, bag, apron),
    extras=[...]); face: dict(expression, eyes (iris colour), gaze=(x, y), look=(u, v), blush, lashes, beard);
    style: flat | soft | anime | line; line: outline width factor; name / alias: the word(s) the figure answers to in
    the scene report (``name="friend", alias=["friend", "classmate"]``; "person" is always an alias).
    """

    def __init__(self, height=0.6, build="average", view="front", skin="light", hair=None, outfit=None, face=None,
                 style="flat", body="neutral", facing=1, line=None, detail=1.0, seed=0, name=None, alias=None):
        if build not in BUILDS:
            raise _unknown_name("build", build, BUILDS)
        if style not in STYLES:
            raise _unknown_name("style", style, STYLES)
        self.height = float(height)
        self.build = build
        if body not in BODY_TYPES:
            raise _unknown_name("body", body, BODY_TYPES)
        self.body = body
        if isinstance(view, str) and view not in VIEWS:
            raise _unknown_name("view", view, VIEWS)
        self.view = view
        if isinstance(facing, str):
            facing = 1 if _check_choice("Figure", "facing", facing, ("right", "left")) == "right" else -1
        self.facing = 1 if (facing is None or facing >= 0) else -1
        if isinstance(skin, str) and not skin.startswith("#") and skin.lower().replace(" ", "_") not in SKIN_TONES:
            try:
                rgb(skin)
            except ValueError:
                raise _unknown_name("skin", skin, SKIN_TONES, "Use one of %s or a colour like '#c58c6b'." %
                                    ", ".join(SKIN_TONES)) from None
        self.skin = _color_name(SKIN_TONES.get(skin, skin) if isinstance(skin, str) else skin)
        hair = _reject_unknown("Figure hair", dict(hair or {}), ("style", "color"))
        hair.setdefault("style", "short")
        if hair["style"] not in HAIR_STYLES:
            raise _unknown_name("hair style", hair["style"], HAIR_STYLES)
        hair["color"] = _color_name(hair.get("color", HAIR_COLORS["dark_brown"]))
        self.hair = hair
        outfit = _reject_unknown("Figure outfit", dict(outfit or {}),
                                 ("top", "bottom", "shoes", "colors", "extras", "long_sleeves", "sleeves", "top_color",
                                  "bottom_color", "shoes_color", "accent_color", "inner_color"))
        cols = _norm_colors(outfit.pop("colors", None))
        bad = [k for k in cols if k not in _COLOR_KEYS]
        if bad:
            raise _unknown_name("outfit colour", bad[0], _COLOR_KEYS)
        for k in ("top", "bottom", "shoes", "accent", "inner"):
            if outfit.get(k + "_color"):
                cols[k] = outfit.pop(k + "_color")
        outfit.setdefault("top", "tshirt")
        outfit.setdefault("bottom", "trousers" if outfit["top"] not in ("dress", "spacesuit") else None)
        outfit.setdefault("shoes", "sneakers")
        if outfit["top"] not in TOPS:
            raise _unknown_name("top", outfit["top"], TOPS)
        if outfit["bottom"] is not None and outfit["bottom"] not in BOTTOMS:
            raise _unknown_name("bottom", outfit["bottom"], BOTTOMS)
        cols.setdefault("top", PALETTE["teal"])
        cols.setdefault("bottom", "#3a5a8a" if outfit["bottom"] == "jeans" else PALETTE["navy"])
        cols.setdefault("shoes", PALETTE["white"] if outfit["shoes"] == "sneakers" else "#3a2a24")
        cols.setdefault("accent", PALETTE["mustard"])
        cols.setdefault("inner", PALETTE["white"])
        self.colors = {k: _color_name(v) for k, v in cols.items()}
        ex = outfit.get("extras") or ()
        self.extras = [ex] if isinstance(ex, str) else list(ex)
        for e in self.extras:
            if e not in EXTRAS:
                raise _unknown_name("extra", e, EXTRAS)
        self.outfit = outfit
        self.face = _reject_unknown("Figure face", dict(face or {}),
                                    ("expression", "eyes", "eyes_closed", "gaze", "look", "blush", "lashes", "open",
                                     "beard"))
        if self.face.get("expression") not in (None,) + tuple(EXPRESSIONS):
            raise _unknown_name("expression", self.face["expression"], EXPRESSIONS)
        self.style = style
        self.line = line
        self.detail = float(detail)
        self.seed = seed
        self.name = name
        self.alias = [alias] if isinstance(alias, str) else list(alias or ())
        self._pose = _full_pose(POSES["stand"])
        self._pose_name = "stand"
        self._look = self.face.get("look")
        self.anchors = {}
        self.bbox = None
        self._mask = None
        self._cv = None
        self._setup_body()

    def __repr__(self):
        return (f"Figure({self.name or ''} build={self.build!r} view={self.view!r} style={self.style!r} "
                f"pose={self._pose_name!r})")

    # ------------------------------------------------------------ body
    def _setup_body(self):
        b = dict(BUILDS[self.build])
        mods = BODY_TYPES[self.body]
        for k, v in mods.items():
            if k in b and k != "heads":
                b[k] = b[k] * v
        young = self.build in ("child", "chibi")
        b["bust"] = mods.get("bust", 0.0) * (0.0 if young else 1.0)
        b["jaw"] = mods.get("jaw", 0.0)
        b["lashes"] = mods.get("lashes", 0.0)
        N = b["heads"]
        C = N - 1.0
        sv = C / 6.4
        q = b["limb"] * (N / 7.4) ** 0.5
        P = dict(b)
        P.update(N=N, C=C, sv=sv, q=q)
        P["hip_y"] = b["leg"] * N
        P["knee_y"] = P["hip_y"] * 0.53
        P["ankle_y"] = max(0.27 * sv ** 0.8, 0.16)       # chunky feet for short builds, not planks
        nb = C - 0.42 * sv ** 0.5
        P["nb"] = nb
        P["neck_len"] = (C - nb) + 0.02
        P["Ls"] = nb - P["hip_y"]
        P["thigh"] = P["hip_y"] - P["knee_y"]
        P["shin"] = P["knee_y"] - P["ankle_y"]
        arm = b["arm"]
        P["upper"] = 0.18 * N * arm
        P["fore"] = 0.145 * N * arm
        P["hand_len"] = 0.085 * N * b["hand"] * arm ** 0.5
        P["foot_len"] = 0.98 * sv ** 0.8 * b["foot"]
        P["r_delt"] = 0.2 * q
        P["r_bicep"] = 0.155 * q
        P["r_elbow"] = 0.105 * q
        P["r_fore"] = 0.135 * q
        P["r_wrist"] = 0.08 * q
        # legs carry the body: they vary less with build than arms do (slim legs used to read as stilts)
        ql = (N / 7.4) ** 0.5 * (b["limb"] if b["limb"] > 1.0 else b["limb"] ** 0.55)
        P["ql"] = ql
        P["r_thigh"] = 0.335 * ql * (1.07 if self.body == "feminine" else 1.0)
        P["r_knee"] = 0.18 * ql
        P["r_calf"] = 0.215 * ql * (1.04 if self.body == "feminine" else 1.0)
        P["r_ankle"] = 0.1 * ql
        P["r_neck"] = 0.19 * b["neck"] * sv ** 0.25
        P["sh_sep"] = b["shoulder"] / 2 - P["r_delt"] * 0.85
        P["hip_sep"] = b["hip"] * 0.26
        # width follows the foot length on short builds (a chibi foot was wider than long and read as a plank)
        P["foot_w"] = 0.4 * q ** 0.6 * b["foot"] * min(1.0, 1.6 * sv ** 0.8)
        P["depth"] = (b["chest"] / 1.56) ** 0.8
        self.P = P

    # ------------------------------------------------------------ pose
    def pose(self, name="stand", phase=0.0, mix=None, amount=0.5, item=None, **joints):
        """Set the pose: a name from ``POSES`` (``walk`` / ``run`` take ``phase``) or a dict; ``mix=other`` blends
        towards another pose by ``amount``; keyword joints override (``elbow_r=90``, ``shoulder_l=(flex, abduct,
        roll)``, ``hand_r="point"``, ``reach_r=(u, v)`` canvas target or ``("chest", (dx, dy, dz))``, ``turn=20``
        head turn, ``nod``, ``tilt``, ``lean``, ``twist``). ``item="cup"`` puts a prop in the right hand.
        ``sit`` takes ``seat=`` from ``SEATS``: auto (a stool at a plain (u, v) point; nothing at a scene Spot,
        where the room has the chair) stool bench chair none, and ``seat_color``. Returns self."""
        p = _resolve_pose(name, phase)
        if mix is not None:
            p = pose_lerp(p, _resolve_pose(mix, phase), amount)
        p = _full_pose(p)
        try:
            joints = _reject_unknown("Figure.pose", joints, sorted(p))
        except TypeError as e:                     # a TypeError, and still a ValueError for older callers
            raise _UnknownJoint(*e.args) from None
        for k, v in joints.items():
            p[k] = v
        if p.get("hold_both") is not None:
            hb = tuple(p["hold_both"])
            if len(hb) != 2:
                raise ValueError("hold_both=(u, v): the canvas fractions of the middle of the object both hands hold")
            _check_frac("Figure.pose hold_both", "hold_both", hb)
            p["hold_both"] = hb
            for sd in ("l", "r"):
                if "hand_" + sd not in joints:
                    p["hand_" + sd] = "grip"
                if "palm_" + sd not in joints:
                    p["palm_" + sd] = "in"
            if item is None and "item" not in joints:
                p["item"] = None
        for sd in ("l", "r"):
            if p.get("hand_" + sd) not in HAND_SHAPES:
                raise _unknown_name("hand shape", p.get("hand_" + sd), HAND_SHAPES)
            if p.get("palm_" + sd) not in _PALMS:
                raise _unknown_name("palm", p.get("palm_" + sd), [x for x in _PALMS if x])
        for sd in ("l", "r"):
            rt = p.get("reach_" + sd)
            if isinstance(rt, (tuple, list)) and len(rt) == 2 and not isinstance(rt[0], str):
                _check_frac("Figure.pose reach_" + sd, "reach_" + sd, rt)
        st = p.get("seat")
        if isinstance(st, str) and st not in SEATS:
            raise _unknown_name("seat", st, SEATS, "seat='none' keeps the seated pose without drawing a seat.")
        if p.get("seat_color") is not None:
            rgb(p["seat_color"])
        for it_ in (p.get("item"), item):
            if it_ is not None and it_ not in ITEMS:
                raise _unknown_name("item", it_, ITEMS)
        if item is not None:
            p["item"] = item
        self._pose = p
        self._pose_name = name if isinstance(name, str) else "custom"
        return self

    def look_at(self, target):
        """Turn the eyes towards a canvas point (u, v). Returns self."""
        _check_frac("Figure.look_at", "target", target)
        self._look = target
        return self

    # ------------------------------------------------------------ skeleton (body coordinates, units = heads)
    def _yaw(self):
        v = self.view
        y = float(VIEWS.get(v, v if isinstance(v, (int, float)) else 0.0))
        return y * self.facing

    def _solve(self):
        P, p = self.P, self._pose
        J = {}
        Rp = _ry(p["pelvis_turn"]) @ _rz(p["hip_tilt"]) @ _rx(p.get("pelvis_lean", 0.0))
        lean, bend, twist = p["lean"], p["bend"], p["twist"]
        stoop = P["stoop"]
        R1 = Rp @ _rx(lean * 0.45 + stoop * 4) @ _rz(-bend * 0.5) @ _ry(twist * 0.4 - p["pelvis_turn"] * 0.4)
        R2 = R1 @ _rx(lean * 0.55 + stoop * 9) @ _rz(-bend * 0.5 - p["hip_tilt"] * 0.6) @ \
            _ry(twist * 0.6 - p["pelvis_turn"] * 0.6)
        P0 = _v(0, P["hip_y"], 0)
        Ls = P["Ls"]
        Pw = P0 + R1 @ _v(0, Ls * 0.38, 0)
        Pc = Pw + R2 @ _v(0, Ls * 0.36, 0)
        Psl = Pc + R2 @ _v(0, Ls * 0.26, 0)
        Rn = R2 @ _rx(p["nod"] * 0.35 + stoop * 8)
        neck_top = Psl + Rn @ _v(0, P["neck_len"], 0.02)
        Rh = Rn @ _rx(p["nod"] * 0.65 - stoop * 18) @ _rz(-p["tilt"]) @ _ry(p["turn"])
        head = neck_top + Rh @ _v(0, 0.36, 0.08)
        J.update(pelvis=P0, waist=Pw, chest=Pc, neck=Psl, neck_top=neck_top, head=head)
        F = dict(Rp=Rp, R1=R1, R2=R2, Rh=Rh, Rn=Rn)
        for side, sg in (("l", 1.0), ("r", -1.0)):
            fl, ab, ro = p["hip_" + side]
            hj = P0 + Rp @ _v(sg * P["hip_sep"], -0.05, 0)
            Rt = Rp @ _mir(_rz(ab) @ _rx(-fl) @ _ry(ro), sg)
            knee = hj + Rt @ _v(0, -P["thigh"], 0)
            Rs = Rt @ _rx(p["knee_" + side])
            ankle = knee + Rs @ _v(0, -P["shin"], 0)
            Rf = Rs @ _rx(-p["ankle_" + side]) @ _ry(sg * p["toe_out"])
            fwd, down, across = Rf @ _v(0, 0, 1), Rf @ _v(0, -1, 0), Rf @ _v(sg, 0, 0)
            ah, fln = P["ankle_y"], P["foot_len"]
            heel = ankle + down * ah - fwd * fln * 0.22
            toe = ankle + down * ah * 0.9 + fwd * fln * 0.8
            ball = ankle + down * ah + fwd * fln * 0.56
            J.update({"hip_" + side: hj, "knee_" + side: knee, "ankle_" + side: ankle, "heel_" + side: heel,
                      "toe_" + side: toe, "ball_" + side: ball})
            F["foot_" + side] = (fwd, down, across)
            F["shin_" + side] = Rs
            F["thigh_" + side] = Rt
        lows = [J[k][1] for k in ("heel_l", "toe_l", "ball_l", "heel_r", "toe_r", "ball_r")]
        lows += [J["knee_l"][1] - P["r_knee"] * 0.9, J["knee_r"][1] - P["r_knee"] * 0.9]
        dy = -min(lows) + float(p["lift"])
        for k in J:
            J[k] = J[k] + _v(0, dy, 0)
        for side, sg in (("l", 1.0), ("r", -1.0)):
            J["hip_side_" + side] = J["waist"] + R1 @ _v(sg * (P["waist"] / 2 + 0.03), -0.28, 0.02)
        J["chin"] = J["head"] + Rh @ _v(0, -0.48, 0.22)
        J["mouth"] = J["head"] + Rh @ _v(0, -0.3, 0.38)
        hb = p.get("hold_both")
        if hb is not None and self._cv is not None:
            hbx = {s_: float((self._V() @ (J["neck"] + R2 @ _v(sg_ * P["sh_sep"], 0, 0)))[0])
                   for s_, sg_ in (("l", 1.0), ("r", -1.0))}
        for side, sg in (("l", 1.0), ("r", -1.0)):
            sh = J["neck"] + R2 @ _v(sg * P["sh_sep"], -0.15 * P["sv"] ** 0.5, -0.02)
            a, bl = P["upper"], P["fore"]
            tgt = p.get("reach_" + side)
            rest_n = _v(-sg, 0, 0)
            if hb is not None and self._cv is not None:
                # both hands on one object: the hand whose shoulder is further left on screen takes the left grip
                ox_, oy_, hu_ = self._origin
                hw = p.get("hold_width")
                if hw is None:
                    hw = abs(hbx["l"] - hbx["r"]) * hu_ / self._cv.w * 1.1
                left = hbx[side] < hbx["l" if side == "r" else "r"]
                tgt = (hb[0] + (-0.5 if left else 0.5) * float(hw), hb[1])
            if tgt is not None:
                T = self._target(tgt, J, R2, sh)
                pole = p.get("pole_" + side) if hb is None else (sg * 0.8, -0.5, -0.2)
                pole = R2 @ (_v(*pole) if pole is not None else _v(sg * 0.7, -0.35, -0.6))
                elbow, wrist = _ik(sh, T, a, bl, pole)
                fdir = _unit(wrist - elbow)
                udir = _unit(elbow - sh)
                n_guess = R2 @ rest_n
            else:
                fl, ab, ro = p["shoulder_" + side]
                Ru = R2 @ _mir(_rz(ab) @ _rx(-fl) @ _ry(ro), sg)
                elbow = sh + Ru @ _v(0, -a, 0)
                Rfo = Ru @ _rx(-p["elbow_" + side]) @ _ry(sg * p["turn_" + side])
                wrist = elbow + Rfo @ _v(0, -bl, 0)
                fdir = Rfo @ _v(0, -1, 0)
                udir = Ru @ _v(0, -1, 0)
                n_guess = Rfo @ rest_n
            wr = p["wrist_" + side]
            n = self._palm(p.get("palm_" + side), R2, sg, n_guess)
            n = _unit(n - fdir * float(np.dot(n, fdir)))
            if np.linalg.norm(n) < 1e-6:
                n = _unit(np.cross(fdir, _v(0, 0, 1)))
            hdir = _unit(fdir * math.cos(math.radians(wr)) + n * math.sin(math.radians(wr)))
            n = _unit(n - hdir * float(np.dot(n, hdir)))
            thumb = _unit(-sg * np.cross(hdir, n))
            hl = P["hand_len"]
            J.update({"shoulder_" + side: sh, "elbow_" + side: elbow, "wrist_" + side: wrist,
                      "hand_" + side: wrist + hdir * hl * 0.5 + n * hl * 0.12,
                      "fingertip_" + side: wrist + hdir * hl})
            F["hand_" + side] = (hdir, n, thumb)
            F["upper_" + side] = udir
            F["fore_" + side] = fdir
        if p["roll"]:
            Rr = _rz(-p["roll"])
            for k in J:
                J[k] = Rr @ J[k]
            for k, v in list(F.items()):
                if isinstance(v, tuple):
                    F[k] = tuple(Rr @ x for x in v)
                else:
                    F[k] = Rr @ v
        return J, F

    def _target(self, tgt, J, R2, sh=None):
        if isinstance(tgt, (tuple, list)) and len(tgt) == 2 and isinstance(tgt[0], str):
            base = J.get(tgt[0], J["chest"])
            return base + R2 @ _v(*tgt[1])
        if isinstance(tgt, (tuple, list)) and len(tgt) == 2 and self._cv is not None:
            ox, oy, hu = self._origin
            x = (tgt[0] * self._cv.w - ox) / hu
            y = (oy - tgt[1] * self._cv.h) / hu
            Vi = np.linalg.inv(self._V())
            pt = Vi @ _v(x, y, 0.0)
            fwd = Vi @ _v(0, 0, 1.0)            # towards the viewer, in body coordinates
            base = J["chest"] if sh is None else sh
            # put the target at the depth of the shoulder, nudged towards the body front: the arm then points at
            # the canvas point in any view (a side view used to push the target far behind the body)
            nudge = 0.45 * float(np.dot(fwd, R2 @ _v(0, 0, 1)))
            t = float(np.dot(base - pt, fwd)) + nudge
            return pt + fwd * t
        if isinstance(tgt, (tuple, list)) and len(tgt) == 3:
            return J["chest"] + R2 @ _v(*tgt)
        return J["chest"] + R2 @ _v(0, -0.4, 0.5)

    @staticmethod
    def _palm(label, R2, sg, guess):
        if label is None:
            return guess
        table = {"side": _v(-sg, 0, 0), "in": _v(-sg, 0, 0), "out": _v(sg, 0, 0), "front": _v(0, 0, 1),
                 "back": _v(0, 0, -1), "down": _v(0, -1, 0), "up": _v(0, 1, 0)}
        return R2 @ table.get(label, _v(-sg, 0, 0))

    def _V(self):
        return _ry(self._yaw())

    # ------------------------------------------------------------ projection & layout
    def layout(self, cv, at=(0.5, 0.9)):
        """Compute the posed geometry and anchors on ``cv`` without drawing. Returns self."""
        self._cv = cv
        _check_frac("Figure.draw", "at", (at[0], at[1]))
        k = getattr(at, "k", 1.0)
        H = self.height * k * cv.h
        hu = H / self.P["N"]
        ox, oy = at[0] * cv.w, at[1] * cv.h
        self._origin = (ox, oy, hu)
        self._hu = hu
        V = self._V()
        self._Vm = V
        J, F = self._solve()
        self._J, self._F = J, F
        S, Z = {}, {}
        for name, p3 in J.items():
            q = V @ p3
            S[name] = (ox + q[0] * hu, oy - q[1] * hu)
            Z[name] = float(q[2])
        self._S, self._Z = S, Z
        self._head_setup()
        self._anchors()
        return self

    def proj(self, p3):
        q = self._Vm @ p3
        ox, oy, hu = self._origin
        return float(ox + q[0] * hu), float(oy - q[1] * hu)

    def projz(self, p3):
        return float((self._Vm @ p3)[2])

    def dir2(self, v3):
        q = self._Vm @ v3
        return float(q[0]), float(-q[1]), float(q[2])

    def _head_setup(self):
        hu = self._hu
        Rh = self._F["Rh"]
        M = self._Vm @ Rh
        up = M @ _v(0, 1, 0)
        fw = M @ _v(0, 0, 1)
        self._hroll = math.atan2(up[0], up[1])
        self._hyaw = math.atan2(fw[0], fw[2])
        self._hpitch = math.asin(max(-1.0, min(1.0, -fw[1])))
        self._hM = M
        self._hc = self._S["head"]
        look = self._look
        if look is not None and self._cv is not None:
            tx, ty = look[0] * self._cv.w, look[1] * self._cv.h
            dx, dy = (tx - self._hc[0]) / hu, (ty - self._hc[1]) / hu
            n = math.hypot(dx, dy) or 1.0
            self._gaze = (max(-1, min(1, dx / n * 1.2)), max(-1, min(1, dy / max(n, 2.0) * 1.5)))
        else:
            g = self.face.get("gaze")
            self._gaze = tuple(g) if g is not None else (math.sin(self._hyaw) * 0.5, 0.05)

    def hp(self, x, y, z):
        """Head-local 3D point (x towards screen right at a front view, y DOWN, z forward; heads) ??head-2D."""
        q = self._hM @ _v(x, -y, z)
        sx, sy = q[0], -q[1]
        c, s = math.cos(self._hroll), math.sin(self._hroll)
        return sx * c + sy * s, -sx * s + sy * c, float(q[2])

    def hn(self, x, y, z):
        """Depth component of a head-local direction (how much a surface normal faces the viewer)."""
        q = self._hM @ _unit(_v(x, -y, z))
        return float(q[2])

    def h2s(self, lx, ly):
        """Head-2D coords ??screen px."""
        c, s = math.cos(self._hroll), math.sin(self._hroll)
        hu = self._hu
        return self._hc[0] + (lx * c - ly * s) * hu, self._hc[1] + (lx * s + ly * c) * hu

    def hloc(self, X, Y):
        c, s = math.cos(self._hroll), math.sin(self._hroll)
        dx, dy = (X - self._hc[0]) / self._hu, (Y - self._hc[1]) / self._hu
        return dx * c + dy * s, -dx * s + dy * c

    def _hf(self, f):
        """Head-local distance function (heads) ??screen distance function (px)."""
        hu = self._hu

        def fn(X, Y):
            lx, ly = self.hloc(X, Y)
            return (f(lx, ly) * hu).astype(F32)
        return fn

    def _hpts(self, r=0.75, cx=0.0, cy=0.0):
        return [self.h2s(cx - r, cy - r), self.h2s(cx + r, cy - r), self.h2s(cx - r, cy + r), self.h2s(cx + r, cy + r)]

    def _anchors(self):
        S = self._S
        A = {k: S[k] for k in ("pelvis", "waist", "chest", "neck", "head", "chin",
                               "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
                               "hand_l", "hand_r", "fingertip_l", "fingertip_r", "hip_l", "hip_r", "knee_l",
                               "knee_r", "ankle_l", "ankle_r", "toe_l", "toe_r", "heel_l", "heel_r")}
        A["head_top"] = self.h2s(*self.hp(0, -0.5, 0.0)[:2])
        hw = self.P["head_w"] / 0.74
        for side, sg in (("l", 1), ("r", -1)):
            A["eye_" + side] = self.h2s(*self.hp(sg * 0.165 * hw, 0.03 + self.P["eye_y"], 0.36)[:2])
            A["ear_" + side] = self.h2s(*self.hp(sg * 0.37 * hw, 0.06, -0.02)[:2])
            fx = (S["heel_" + side][0] + S["toe_" + side][0]) / 2
            A["foot_" + side] = (fx, max(S["heel_" + side][1], S["toe_" + side][1]))
        A["nose"] = self.h2s(*self.hp(0, 0.2, 0.46)[:2])
        A["mouth"] = self.h2s(*self.hp(0, 0.31 + self.P["eye_y"] * 0.5, 0.38)[:2])
        A["feet"] = (self._origin[0], self._origin[1])
        A["ground"] = A["feet"]
        hu = self._hu
        A["seat"] = (S["pelvis"][0], S["pelvis"][1] + self.P["r_thigh"] * hu * 0.95)
        A["lap"] = ((S["hip_l"][0] + S["knee_l"][0] + S["hip_r"][0] + S["knee_r"][0]) / 4,
                    (S["hip_l"][1] + S["knee_l"][1] + S["hip_r"][1] + S["knee_r"][1]) / 4 - self.P["r_thigh"] * hu)
        A["back"] = self.proj(self._J["chest"] + self._F["R2"] @ _v(0, 0, -0.5))
        sy = math.sin(self._hyaw)
        A["speech"] = (A["mouth"][0] + (sy * 1.0 + (0.6 if abs(sy) < 0.3 else 0.0) * self.facing) * hu,
                       A["head_top"][1] - 0.25 * hu)
        self.anchors = {k: (float(v[0]), float(v[1])) for k, v in A.items()}

    def anchor(self, name):
        """Canvas pixel (x, y) of a joint or useful point after ``draw`` / ``layout``: head_top, head, eye_l,
        eye_r, nose, mouth, chin, ear_l, neck, chest, waist, pelvis, back, shoulder_l/r, elbow_l/r, wrist_l/r,
        hand_l/r (grip point), fingertip_l/r, hip_l/r, knee_l/r, ankle_l/r, foot_l/r, toe_l/r, heel_l/r, feet,
        seat, lap, speech (a spot for a speech bubble)."""
        if not self.anchors:
            raise RuntimeError("call fig.draw(cv, at) or fig.layout(cv, at) first")
        if name not in self.anchors:
            raise _AnchorError(str(_unknown_name("anchor", name, sorted(self.anchors))))
        return self.anchors[name]

    def anchor_uv(self, name):
        """The anchor as canvas fractions (u, v)."""
        x, y = self.anchor(name)
        return x / self._cv.w, y / self._cv.h

    @property
    def mask(self):
        """Full-frame coverage mask (H, W) of the last draw."""
        return self._mask

    @property
    def head_size(self):
        """Head height in px (after layout / draw)."""
        return self._hu

    # ------------------------------------------------------------ drawing
    def draw(self, cv, at=(0.5, 0.9), log=None, shadow=True, light=None):
        """Draw the figure with its feet (or seat) at ``at`` ??a ground point (u, v) or a perspective Spot.

        shadow: True = contact + cast shadow on the ground, a float scales it, False = none. Returns self;
        afterwards ``fig.anchor(name)``, ``fig.bbox`` and ``fig.mask`` are set."""
        if log:
            log(self.name or "figure")
        light = _form.resolve_light(light)
        self.layout(cv, at)
        hu = self._hu
        xs = [p[0] for p in self._S.values()]
        ys = [p[1] for p in self._S.values()]
        pad = hu * (1.25 if self._has_big_hair() else 0.95) + 4
        item = self._pose.get("item")
        if item in ("guitar", "bag", "briefcase", "flower"):
            pad += hu * 0.9
        box = _form.clip_box(cv, min(xs) - pad, min(ys) - pad * 1.25, max(xs) + pad, max(ys) + pad * 0.6)
        if box is None:
            self._mask = np.zeros((cv.h, cv.w), F32)
            self.bbox = None
            return self
        pt = _Painter(box, self.style, light, hu, self.line)
        self._pt = pt
        self._paint_parts(pt)
        A = pt.alpha
        straight = pt.col / np.maximum(A, 1e-6)[..., None]
        full = _form.full_frame(cv, A, box)
        self._mask = full
        self.bbox = _form.bbox_of(full, 0.02)
        front = self._draw_seat(cv, at, shadow)
        if shadow and self.bbox is not None:
            st = 1.0 if shadow is True else float(shadow)
            self._ground_shadow(cv, full, light, st)
        cv.paste(straight, box[0], box[1], mask=A)
        for poly, col in front:                              # a chair back between the viewer and the figure
            self._seat_poly(cv, poly, col)
        return self

    # ------------------------------------------------------------ seat under a sitting figure
    def _seat_kind(self, at):
        st = self._pose.get("seat")
        if not st or st == "none":
            return None
        if st is True or st == "auto":
            return None if hasattr(at, "k") else "stool"
        return st

    def _seat_poly(self, cv, pts, col):
        pts = [(float(x), float(y)) for x, y in pts]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        box = _form.clip_box(cv, min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3)
        if box is None:
            return
        x0, y0, x1, y1 = box
        Y, X = np.mgrid[y0:y1, x0:x1].astype(F32) + 0.5
        d = _sdf.sd_polygon(X, Y, pts)
        lw = max(0.8, self._hu * 0.018)
        ink = _mixc(_col(col), _col(INK), 0.55)
        for c, m in ((ink, _sdf.fill(d - lw)), (_col(col), _sdf.fill(d))):
            full = np.zeros((cv.h, cv.w), F32)
            full[y0:y1, x0:x1] = m
            cv.paint(c, full)

    def _draw_seat(self, cv, at, shadow):
        """Paint a simple stool / bench / chair under a sitting figure; returns parts to paint over the figure."""
        kind = self._seat_kind(at)
        if kind is None:
            return []
        P, J, Rp = self.P, self._J, self._F["Rp"]
        ex = _unit(Rp @ _v(1, 0, 0) * np.array([1.0, 0.0, 1.0]))
        ez = _unit(Rp @ _v(0, 0, 1) * np.array([1.0, 0.0, 1.0]))
        hip_y = min(J["hip_l"][1], J["hip_r"][1])
        top = hip_y - P["r_thigh"] * 0.9
        if top < 0.35:
            return []
        c = J["pelvis"] * np.array([1.0, 0.0, 1.0])
        z0, z1 = -P["r_thigh"] * 1.15, P["thigh"] * 0.62
        hw = P["hip_sep"] + P["r_thigh"] * 1.75
        if kind == "bench":
            hw *= 2.3
        th = 0.16
        base = self._pose.get("seat_color") or "#a0714a"
        dark = _mixc(_col(base), (0.0, 0.0, 0.0), 0.28)

        def W(x, y, z):
            return c + ex * x + ez * z + _v(0, y, 0)

        def face(corners):
            return [self.proj(q) for q in corners]

        def depth(q):
            return self.projz(q)

        Vm = self._Vm

        def box(xa, xb, ya, yb, za, zb, col):
            """Visible faces of an axis box in seat space, far faces first: [(poly, colour)]."""
            fc = []
            for n3, quad in ((ez, [W(xa, yb, zb), W(xb, yb, zb), W(xb, ya, zb), W(xa, ya, zb)]),
                             (-ez, [W(xa, yb, za), W(xb, yb, za), W(xb, ya, za), W(xa, ya, za)]),
                             (ex, [W(xb, yb, za), W(xb, yb, zb), W(xb, ya, zb), W(xb, ya, za)]),
                             (-ex, [W(xa, yb, za), W(xa, yb, zb), W(xa, ya, zb), W(xa, ya, za)])):
                facing = float((Vm @ n3)[2])
                if facing > 1e-3:
                    fc.append((facing, face(quad), col if facing > 0.7 else _mixc(col, (0.0, 0.0, 0.0), 0.14)))
            return [(q, cl) for _, q, cl in sorted(fc, key=lambda t: t[0])]

        back, front = [], []
        r, ins = 0.055, 0.1
        if kind == "chair":                                  # the back rest stands behind the sitter's back
            z0 = min(z0, -P["depth"] * 0.62) - 0.02
        legs = []
        if kind == "bench":                                  # two end panels
            for x in (-hw + 0.25, hw - 0.25):
                legs.append((depth(W(x, 0, 0)), box(x - 0.06, x + 0.06, 0.0, top - th, z0 + ins, z1 - ins, dark)))
        else:
            for x in (-hw + ins, hw - ins):
                for z in (z0 + ins, z1 - ins):
                    legs.append((depth(W(x, 0, z)), box(x - r, x + r, 0.0, top - th, z - r, z + r, dark)))
        for _, faces in sorted(legs, key=lambda t: t[0]):
            back.extend(faces)
        back.extend(box(-hw, hw, top - th, top, z0, z1, _col(base)))
        if kind == "chair":
            rest_top = top + P["Ls"] * 0.9
            zb = z0 + 0.02
            parts = []
            for x in (-hw + ins, hw - ins):
                parts.append((depth(W(x, top, zb)), box(x - r, x + r, top, rest_top - 0.05, zb, zb + 2 * r, dark)))
            parts.append((depth(W(0, rest_top, zb)) + 1.0,
                          box(-hw, hw, rest_top - P["Ls"] * 0.34, rest_top, zb, zb + 0.09, _col(base))))
            faces = [f for _, fs in sorted(parts, key=lambda t: t[0]) for f in fs]
            in_front = depth(W(0, top, zb)) > depth(J["pelvis"]) + 0.05
            (front if in_front else back).extend(faces)
        if shadow:
            st = 1.0 if shadow is True else float(shadow)
            pts = [self.proj(W(x, 0, z)) for x in (-hw, hw) for z in (z0, z1)]
            xs = [q[0] for q in pts]
            gy = self._origin[1]
            _form.contact_px(cv, (min(xs) + max(xs)) / 2, gy, (max(xs) - min(xs)) / 2 + 0.2 * self._hu,
                             0.13 * self._hu, 0.45 * st, None, None, 1.4)
        for q, col in back:
            self._seat_poly(cv, q, col)
        return front

    def _has_big_hair(self):
        return self.hair.get("style") in ("afro", "long", "curly", "braids", "bun", "ponytail") or \
            any(e in self.extras for e in ("hat", "helmet", "chef_hat"))

    def _ground_shadow(self, cv, full, light, strength):
        hu = self._hu
        gy = self._origin[1]
        lift = float(self._pose["lift"]) * hu
        if lift < 0.2 * hu and not self._pose.get("seat"):
            _form.cast_shadow(cv, full, light=light, ground_y=gy / cv.h, opacity=0.34 * strength, softness=1.4,
                              length=0.75)
        S = self._S
        for side in ("l", "r"):
            hx, hy = S["heel_" + side]
            tx, ty = S["toe_" + side]
            low = max(hy, ty)
            if gy - low > 0.3 * hu and lift < 0.2 * hu:
                continue
            cx = (hx + tx) / 2
            w = max(abs(tx - hx) + self.P["foot_w"] * 1.6 * hu, 0.45 * hu)
            _form.contact_px(cv, cx, gy, w * 0.62, max(w * 0.13, 0.1 * hu), 0.55 * strength / (1 + lift / hu),
                              light, None, 1.0 + lift / hu)
        if not self._pose.get("seat"):
            _form.contact_px(cv, self._origin[0], gy, 0.85 * hu * self.P["hip"], 0.14 * hu, 0.2 * strength, light,
                              None, 1.8)

    # ------------------------------------------------------------ part list
    def _paint_parts(self, pt):
        """Collect every part with a depth key, then paint back to front."""
        parts = []

        def add(key, fn):
            parts.append((key, len(parts), fn))

        Z, P, J = self._Z, self.P, self._J
        hu = self._hu
        self._back = math.cos(math.radians(self._yaw())) < -0.2
        self._simple = hu * self.detail < 16
        self._tiny = hu * self.detail < 8
        seat = bool(self._pose.get("seat"))
        bottom = self.outfit.get("bottom")
        shoes = self.outfit.get("shoes")
        # the head always covers the neck and collar (a stooped head seen from behind used to sink under them and
        # show the neck as a bald patch through the hair)
        hk = max(Z["head"] + 0.32, 0.02)
        if self._hair_back_kind():
            add(hk + 0.01 if self._back else -5.0, lambda p: self._hair_back(p, layer="back"))
        for side in ("l", "r"):
            zh, zk = Z["hip_" + side], Z["knee_" + side]
            kl = -0.06 + 0.03 * zh + max(0.0, (zk - zh) - 0.8)
            over = shoes == "boots" and bottom not in ("trousers", "jeans")
            add(kl + (0.002 if over else -0.002), lambda p, s=side: self._shoe(p, s))
            add(kl, lambda p, s=side: self._leg(p, s))
        add(-0.03, self._bottoms)
        if "backpack" in self.extras:
            c = J["chest"] + self._F["R2"] @ _v(0, -0.1, -0.6 * P["depth"])
            add(self.projz(c) * 0.6 - 0.02, self._backpack)
        add(0.0, self._torso)
        item = self._pose.get("item")
        for side in ("l", "r"):
            zu = (Z["shoulder_" + side] + Z["elbow_" + side]) / 2
            zf = (Z["elbow_" + side] + Z["wrist_" + side]) / 2
            zh = Z["hand_" + side]
            add(zu + 0.03, lambda p, s=side: self._arm(p, s, "upper"))
            add(zf + 0.04, lambda p, s=side: self._arm(p, s, "fore"))
            if item and side == "r":
                if item == "guitar":
                    kg = min((Z["elbow_l"] + Z["wrist_l"]) / 2, (Z["elbow_r"] + Z["wrist_r"]) / 2) + 0.035
                    add(max(kg, 0.01), lambda p: self._item(p, item))
                else:
                    add(zh + 0.045, lambda p: self._item(p, item))
            add(zh + 0.05, lambda p, s=side: self._hand(p, s))
        if "bag" in self.extras:
            add(Z["hip_r"] * 0.5 + 0.12, self._shoulder_bag)
        add(hk, self._head)
        parts.sort(key=lambda t: (t[0], t[1]))
        for _, _, fn in parts:
            fn(pt)

    # ------------------------------------------------------------ helpers
    def _px(self, v):
        return v * self._hu

    def _ring(self, c3, R, w, df, db, n=18, tilt=0.0):
        """Projected cross-section ellipse (half-width w, front depth df, back depth db) of a body level."""
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            s_ = math.sin(a)
            z = (df if s_ > 0 else db) * s_
            pts.append(self.proj(c3 + R @ _v(w * math.cos(a), -tilt * s_, z)))
        return pts

    def _levels(self):
        J, F, P = self._J, self._F, self.P
        Rp, R1, R2 = F["Rp"], F["R1"], F["R2"]
        tv = P["Ls"] / 2.13
        dz = P["depth"]
        bust = P["bust"]
        L = {}
        L["crotch"] = (J["pelvis"] + Rp @ _v(0, -0.42 * tv, 0.02), Rp, P["hip"] * 0.36, 0.3 * dz, 0.36 * dz)
        L["hip"] = (J["pelvis"] + Rp @ _v(0, -0.1 * tv, 0), Rp, P["hip"] / 2, 0.38 * dz, 0.47 * dz)
        L["waist"] = (J["waist"], R1, P["waist"] / 2, 0.36 * dz, 0.34 * dz)
        L["bust"] = (J["chest"] + R2 @ _v(0, -0.12 * tv, 0.0), R2, P["chest"] * 0.46, (0.4 + 0.17 * bust) * dz,
                     0.37 * dz)
        L["chest"] = (J["chest"] + R2 @ _v(0, 0.1 * tv, 0), R2, P["chest"] / 2, (0.42 + 0.06 * bust) * dz, 0.4 * dz)
        L["armpit"] = (J["neck"] + R2 @ _v(0, -0.3 * tv, -0.01), R2, P["shoulder"] / 2 - P["r_delt"] * 0.62,
                       0.36 * dz, 0.36 * dz)
        L["shoulder"] = (J["neck"] + R2 @ _v(0, -0.15 * P["sv"] ** 0.5 + P["r_delt"] * 0.55, -0.03), R2,
                         P["sh_sep"] * 0.98, 0.24 * dz, 0.28 * dz)
        L["neck"] = (J["neck"] + R2 @ _v(0, 0.12, -0.03), R2, P["r_neck"] * 1.04, P["r_neck"], P["r_neck"])
        return L

    def _lrings(self, L, names, t=0.0, tn=None, n=18):
        out = []
        for nm in names:
            c, R, w, df, db = L[nm]
            tt = (tn if (tn is not None and nm == "neck") else t)
            out.append(self._ring(c, R, w + tt, df + tt, db + tt, n))
        return out

    def _surf(self, L, name, x, t=0.0, front=True):
        """A point on the front (or back) surface of a body level; x = -1..1 across. Returns (px, py, z)."""
        c, R, w, df, db = L[name]
        d = (df if front else db) + t
        z = d * math.sqrt(max(0.0, 1 - min(abs(x), 1.0) ** 2))
        p3 = c + R @ _v(x * (w + t), 0, z if front else -z)
        x2, y2 = self.proj(p3)
        return x2, y2, self.projz(p3)

    def _front_z(self):
        return self.dir2(self._F["R2"] @ _v(0, 0, 1))[2]

    def _bbox_pts(self, rings):
        return [p for r in rings for p in r]

    # ------------------------------------------------------------ legs
    def _leg_pts(self, side):
        J, F, P = self._J, self._F, self.P
        hip, knee, ankle = J["hip_" + side], J["knee_" + side], J["ankle_" + side]
        Rs = F["shin_" + side]
        Rt = F["thigh_" + side]
        back = Rs @ _v(0, 0, -1)
        fwd_t = Rt @ _v(0, 0, 1)
        sg = 1.0 if side == "l" else -1.0
        out_t = Rt @ _v(sg, 0, 0)
        th1 = hip + (knee - hip) * 0.4 + out_t * 0.03 * P["q"] + fwd_t * 0.02
        th2 = hip + (knee - hip) * 0.82
        c1 = knee + (ankle - knee) * 0.3 + back * 0.045 * P["q"]
        c2 = knee + (ankle - knee) * 0.72
        return hip, th1, th2, knee, c1, c2, ankle

    def _leg(self, pt, side):
        P, hu = self.P, self._hu
        hip, th1, th2, knee, c1, c2, ankle = self._leg_pts(side)
        bottom = self.outfit.get("bottom")
        top = self.outfit.get("top")
        seat = bool(self._pose.get("seat"))
        q = [self.proj(p) for p in (hip, th1, th2, knee, c1, c2, ankle)]
        skin = self.skin
        r = [P["r_thigh"], P["r_thigh"] * 0.86, P["r_knee"] * 1.18, P["r_knee"], P["r_calf"], P["r_calf"] * 0.72,
             P["r_ankle"]]
        rp = [x * hu for x in r]
        k = 0.12 * hu * P["q"]
        bare = bottom in ("shorts", "skirt", None) or top in ("dress",) and bottom is None
        if top == "spacesuit":
            bare = False
        shin_dir = _unit(np.array(q[6]) - np.array(q[3]))
        if bare or bottom == "leggings":
            col = skin if bottom != "leggings" else self.colors["bottom"]
            kind = "skin" if bottom != "leggings" else "cloth"
            fn = lambda X, Y: _chain(X, Y, q, rp, k) - (0.01 * hu if bottom == "leggings" else 0)
            pts = q
            if bare and not self.outfit.get("shoes"):
                # bare foot and leg in one silhouette: one outline, continuous shading through the ankle
                fn_f, pts_f = self._bare_foot_fn(side)
                fn_l = fn
                fn = lambda X, Y: _smin(fn_l(X, Y), fn_f(X, Y), 0.04 * hu)
                pts = list(q) + list(pts_f)
            pt.part(fn, pts, max(rp) + 2, col, kind, r=rp[0] * 0.75, cast=0.3)
            if bottom == "leggings":
                return
            # shorts / seated skirt over the thigh
            cover = None
            if bottom == "shorts":
                cover = (0.5, self.colors["bottom"], 0.07)
            elif seat and (bottom == "skirt" or top == "dress"):
                cover = (0.75, self.colors["bottom"] if bottom == "skirt" else self.colors["top"], 0.06)
            if cover:
                frac, ccol, t = cover
                hem3 = hip + (self._J["knee_" + side] - hip) * frac
                hem = self.proj(hem3)
                a = np.array(q[0])
                dvec = _unit(np.array(hem) - a)
                qq = [q[0], self.proj(hip + (hem3 - hip) * 0.5), hem]
                rr = [rp[0] + t * hu, rp[0] * 0.95 + t * hu, rp[0] * 0.9 + (t + 0.04) * hu]
                fn2 = lambda X, Y: np.maximum(_chain(X, Y, qq, rr, k), _plane(X, Y, hem, dvec))
                pt.part(fn2, qq, max(rr) + 2, ccol, "cloth", r=rr[0] * 0.75, cast=0.25)
            return
        # trousers / jeans / spacesuit
        t = {"jeans": 0.035, "trousers": 0.055}.get(bottom, 0.06)
        if top == "spacesuit":
            t = 0.14
        col = self.colors["bottom"] if top != "spacesuit" else self.colors["top"]
        if top == "suit" and "bottom" not in self.outfit.get("colors", {}):
            col = self.colors["top"]
        hem3 = ankle + (ankle - knee) * 0.02
        hem = self.proj(hem3)
        r_hem = (P["r_ankle"] + (0.13 if bottom == "trousers" else 0.1) + (0.05 if top == "spacesuit" else 0)) * hu
        qq = [q[0], q[1], q[3], self.proj(knee + (hem3 - knee) * 0.5), hem]
        rr = [rp[0] + t * hu, rp[1] + t * hu, rp[3] + (t + 0.05) * hu, (P["r_calf"] + t * 0.9) * hu, r_hem]
        if bottom == "jeans":
            rr[3] = (P["r_calf"] + 0.03) * hu
        fn = lambda X, Y: np.maximum(_chain(X, Y, qq, rr, k), _plane(X, Y, hem, shin_dir))
        tone = pt.tone(col, "cloth")

        def det(X, Y, D, A, sl):
            if self._simple:
                return
            # crease / seam
            if bottom in ("trousers", "jeans") and top != "spacesuit":
                mid = [self.proj(hip + (knee - hip) * 0.2), q[3], self.proj(knee + (hem3 - knee) * 0.9)]
                pt.stroke(lambda X, Y: _polyline_sd(X, Y, mid, 0), mid, 2, tone, max(0.6, 0.012 * hu),
                          opacity=0.45, clip=lambda X, Y: fn(X, Y) + 0.03 * hu)
            # hem band
            hb = (hem[0] - shin_dir[0] * 0.05 * hu, hem[1] - shin_dir[1] * 0.05 * hu)
            pt.stroke(lambda X, Y: _plane(X, Y, hb, shin_dir), [hb], r_hem + 2, tone, max(0.6, 0.016 * hu),
                      opacity=0.6, clip=fn)
        pt.part(fn, qq, max(rr) + 2, col, "cloth", r=rr[0] * 0.8, cast=0.3, details=det)

    # ------------------------------------------------------------ shoes
    def _shoe_pts(self, side):
        J, F, P = self._J, self._F, self.P
        fwd, down, across = F["foot_" + side]
        up = -down
        A = J["ankle_" + side]
        fl, fw, ah = P["foot_len"], P["foot_w"], P["ankle_y"]
        kind = self.outfit.get("shoes") or "bare"
        sole = A + down * ah
        hc = {"sneakers": 0.06, "boots": 0.42, "shoes": -0.02, "flats": -0.1, "bare": -0.2}.get(kind, 0.0) * ah / 0.27
        toe_h = {"sneakers": 0.17, "boots": 0.19, "shoes": 0.14, "flats": 0.11, "bare": 0.1}.get(kind, 0.14) * ah / 0.27
        wmul = {"sneakers": 1.06, "boots": 1.08, "shoes": 1.0, "flats": 0.94, "bare": 0.92}.get(kind, 1.0)
        fw = fw * wmul
        hb = sole - fwd * 0.24 * fl
        bb = sole + fwd * 0.56 * fl
        tb = sole + fwd * 0.83 * fl + up * 0.015
        tip = sole + fwd * (0.9 if kind != "bare" else 0.86) * fl + up * toe_h * 0.4
        pts = [hb + across * 0.36 * fw, hb - across * 0.36 * fw, hb + up * 0.12 * ah / 0.27,
               bb + across * 0.5 * fw, bb - across * 0.52 * fw, tb + across * 0.32 * fw, tb - across * 0.36 * fw, tip,
               tb + up * toe_h + across * 0.26 * fw, tb + up * toe_h - across * 0.28 * fw,
               sole + fwd * 0.38 * fl + up * ah * 0.95 + across * 0.33 * fw,
               sole + fwd * 0.38 * fl + up * ah * 0.95 - across * 0.35 * fw,
               sole + fwd * 0.62 * fl + up * ah * 0.62 + across * 0.3 * fw,
               sole + fwd * 0.62 * fl + up * ah * 0.62 - across * 0.32 * fw]
        cr = A + up * hc
        for ax in (-1, 1):
            for fz in (-1, 1):
                pts.append(cr + across * ax * 0.38 * fw + fwd * fz * 0.2 * fl - fwd * 0.04 * fl)
        st = {"sneakers": 0.065, "boots": 0.09, "shoes": 0.04, "flats": 0.03, "bare": 0.0}.get(kind, 0.04) * ah / 0.27
        sp = []
        for p_, wv in ((hb, 0.39), (bb, 0.54), (tb, 0.35)):
            for ax in (-1, 1):
                sp.append(p_ + across * ax * wv * fw - up * 0.005)
                sp.append(p_ + across * ax * wv * fw + up * st)
        sp.append(tip - up * toe_h * 0.4 + fwd * 0.01)
        sp.append(hb - fwd * 0.02 * fl + up * st * 0.5)
        return pts, sp, kind

    def _shoe(self, pt, side):
        hu = self._hu
        pts, sp, kind = self._shoe_pts(side)
        q = [self.proj(p) for p in pts]
        h = _hull(q)
        cx_ = sum(p[0] for p in h) / len(h)
        cy_ = sum(p[1] for p in h) / len(h)
        rr = 0.07 * hu
        h = [(cx_ + (x - cx_) * 0.97 - math.copysign(min(rr * 0.6, abs(x - cx_) * 0.3), x - cx_),
              cy_ + (y - cy_) * 0.97 - math.copysign(min(rr * 0.6, abs(y - cy_) * 0.3), y - cy_)) for x, y in h]
        fn = lambda X, Y: _convex_sd(X, Y, h) - rr
        if kind == "bare":
            if self._bare_legs():
                return                       # the foot is part of the leg silhouette (see _leg)
            fn_f, pts_f = self._bare_foot_fn(side)
            pt.part(fn_f, pts_f, 0.3 * hu, self.skin, "skin", r=0.15 * hu, cast=0.2)
            return
        col = self.colors["shoes"]
        lum = float(np.dot(_col(col), [0.3, 0.55, 0.15]))
        sole_col = ("#f3f0ea" if lum < 0.75 else "#d9cfbf") if kind == "sneakers" else _mixc(col, "#1a1416", 0.55)
        sq = [self.proj(p) for p in sp]
        fwd3, down3, _ = self._F["foot_" + side]
        ux, uy, _ = self.dir2(-down3)
        nrm = math.hypot(ux, uy) or 1.0
        stp = {"sneakers": 0.075, "boots": 0.07, "shoes": 0.04, "flats": 0.03}.get(kind, 0.04) * hu \
            * (self.P["ankle_y"] / 0.27) ** 0.5
        ox, oy = ux / nrm * stp, uy / nrm * stp
        fs = lambda X, Y: np.maximum(fn(X, Y), -fn(X - ox, Y - oy))

        def det(X, Y, D, A, sl):
            pt.flat(fs, sq + q, rr + 2, sole_col)
            if kind == "sneakers" and not self._simple:
                # a side stripe
                J, F = self._J, self._F
                fwd, down, across = F["foot_" + side]
                up = -down
                A3 = J["ankle_" + side]
                fl, ah = self.P["foot_len"], self.P["ankle_y"]
                vis = self.dir2(across)[2]
                ax = 1.0 if vis > 0 else -1.0
                base = A3 + down * ah * 0.6 + across * ax * self.P["foot_w"] * 0.52
                s3 = [base - fwd * 0.14 * fl + up * 0.08, base + fwd * 0.12 * fl - up * 0.02, base + fwd * 0.34 * fl]
                s2 = [self.proj(p) for p in s3]
                if abs(vis) > 0.25:
                    pt.stroke(lambda X, Y: _polyline_sd(X, Y, s2, 0), s2, 0.05 * hu, _mixc(col, self.colors["accent"], 0.8),
                              max(0.8, 0.045 * hu), clip=lambda X, Y: np.maximum(fn(X, Y), -fs(X, Y)) + 0.02 * hu)
        pt.part(fn, q, rr + 2, col, "shoe" if self.style == "soft" else "dark", r=0.16 * hu, cast=0.2, details=det)

    def _bare_legs(self):
        bottom = self.outfit.get("bottom")
        top = self.outfit.get("top")
        if top == "spacesuit":
            return False
        return bottom in ("shorts", "skirt", None) or (top == "dress" and bottom is None)

    def _bare_foot_fn(self, side):
        """A bare foot: round heel, instep, ball and tapering toes ??(distance fn px, points)."""
        J, F, P, hu = self._J, self._F, self.P, self._hu
        fwd, down, across = F["foot_" + side]
        up = -down
        A = J["ankle_" + side]
        fl, ah = P["foot_len"], P["ankle_y"]
        s = ah / 0.27
        sole = A + down * ah
        ax, ay, _ = self.dir2(across)
        wv = min(1.0, math.hypot(ax, ay))             # 1 when the width of the foot faces the viewer
        heel = sole - fwd * 0.14 * fl + up * 0.1 * s
        mid = sole + fwd * 0.3 * fl + up * 0.12 * s
        ball = sole + fwd * 0.56 * fl + up * 0.075 * s
        toe = sole + fwd * 0.8 * fl + up * 0.05 * s
        r_heel = (0.1 + 0.03 * wv) * s
        r_mid = (0.085 + 0.08 * wv) * s
        r_ball = (0.072 + 0.11 * wv) * s
        r_toe = (0.05 + 0.1 * wv) * s
        pa, ph, pm, pb, pt_ = (self.proj(p) for p in (A, heel, mid, ball, toe))
        ra = P["r_ankle"] * hu * 0.95
        R = [r_heel * hu, r_mid * hu, r_ball * hu, r_toe * hu]
        k = 0.05 * hu

        # heel, ankle, instep and ball as one convex mass (separate cones left a gap that read as a second toe),
        # toes tapering out of it
        ring = []
        for (cx, cy), rr in ((pa, ra), (ph, R[0]), (pm, R[1]), (pb, R[2])):
            ring += [(cx + rr * math.cos(t_), cy + rr * math.sin(t_)) for t_ in np.linspace(0, 2 * math.pi, 12,
                                                                                            endpoint=False)]
        core = _hull(ring)

        def fn(X, Y):
            d = _convex_sd(X, Y, core)
            return _smin(d, _cone(X, Y, pb, pt_, R[2], R[3]), k)
        return fn, [pa, ph, pm, pb, pt_]

    # ------------------------------------------------------------ pelvis / bottoms
    def _bottoms(self, pt):
        bottom = self.outfit.get("bottom")
        top = self.outfit.get("top")
        hu = self._hu
        L = self._levels()
        seat = bool(self._pose.get("seat"))
        k = 0.1 * hu
        if top == "spacesuit":
            rings = self._lrings(L, ("crotch", "hip", "waist"), t=0.15)
            fn = _loft_fn(rings, k)
            pt.part(fn, self._bbox_pts(rings), 4, self.colors["top"], "cloth", r=0.6 * hu, cast=0.2)
            return
        skirt = bottom == "skirt" or (top == "dress" and bottom is None)
        if bottom in ("trousers", "jeans", "shorts", "leggings") or (bottom is None and not skirt):
            col = self.colors["bottom"] if bottom else self.skin
            if top == "suit" and "bottom" not in self.outfit.get("colors", {}):
                col = self.colors["top"]
            t = {"jeans": 0.035, "trousers": 0.055, "shorts": 0.06, "leggings": 0.012}.get(bottom, 0.0)
            rings = self._lrings(L, ("crotch", "hip", "waist"), t=t)
            fn = _loft_fn(rings, k)
            tone = pt.tone(col, "cloth")

            def det(X, Y, D, A, sl):
                if self._simple or bottom is None:
                    return
                # waistband
                c, R, w, df, db = L["waist"]
                wb = self._ring(c + R @ _v(0, -0.07, 0), R, w + t + 0.005, df + t, db + t, 24)
                hw = _hull(wb)
                pt.stroke(lambda X, Y: _convex_sd(X, Y, hw), wb, 2, tone, max(0.7, 0.02 * hu), opacity=0.7, clip=fn)
                if bottom == "jeans" and self._front_z() > 0.2:
                    fly = [self._surf(L, "waist", 0.08, t)[:2], self._surf(L, "hip", 0.1, t)[:2],
                           self._surf(L, "crotch", 0.0, t)[:2]]
                    pt.stroke(lambda X, Y: _polyline_sd(X, Y, fly, 0), fly, 2, tone, max(0.6, 0.014 * hu),
                              opacity=0.6, clip=fn)
            pt.part(fn, self._bbox_pts(rings), 3, col, "skin" if bottom is None else "cloth", r=0.5 * hu, cast=0.2,
                    details=det)
            return
        # skirt / dress skirt
        col = self.colors["bottom"] if bottom == "skirt" else self.colors["top"]
        J, P = self._J, self.P
        kl, kr = J["knee_l"], J["knee_r"]
        Rp = self._F["Rp"]
        if seat:
            c = J["pelvis"] + Rp @ _v(0, -0.45, 0.25)
            w = P["hip"] / 2 + 0.12
            hem = (c, Rp, w, 0.55, 0.5)
        else:
            length = 0.78 if bottom == "skirt" else 0.82
            hip3 = L["hip"][0]
            mid = (kl + kr) / 2
            c = hip3 + (mid - hip3) * length
            spread = float(np.linalg.norm((kl - kr) - Rp @ _v(0, 1, 0) * float(np.dot(kl - kr, Rp @ _v(0, 1, 0)))))
            w = max(P["hip"] / 2 * 1.32, spread / 2 + P["r_knee"] + 0.18)
            hem = (c, Rp, w, 0.62 * P["depth"] + 0.05, 0.62 * P["depth"] + 0.1)
        L2 = dict(L)
        L2["hem"] = hem
        rings = self._lrings(L2, ("hem", "hip", "waist"), t=0.04)
        fn = _loft_fn(rings, 0.06 * hu)
        tone = pt.tone(col, "cloth")

        def det(X, Y, D, A, sl):
            if self._simple:
                return
            # a few soft folds from the hip to the hem
            for xf in (-0.55, 0.1, 0.6):
                a = self._surf(L2, "hip", xf * 0.8, 0.04)
                b = self._surf(L2, "hem", xf, 0.04)
                if a[2] > -0.05 and b[2] > -0.05:
                    seg = [a[:2], b[:2]]
                    pt.stroke(lambda X, Y, seg=seg: _polyline_sd(X, Y, seg, 0), seg, 2, tone, max(0.7, 0.02 * hu),
                              opacity=0.55, clip=lambda X, Y: fn(X, Y) + 0.05 * hu)
        pt.part(fn, self._bbox_pts(rings), 3, col, "cloth", r=0.55 * hu, cast=0.25, details=det)

    # ------------------------------------------------------------ torso, tops, neck
    _THICK = dict(tank=0.02, tshirt=0.032, shirt=0.035, blouse=0.04, sweater=0.06, hoodie=0.08, jacket=0.085,
                  suit=0.07, coat=0.1, dress=0.03, spacesuit=0.16)

    def _torso(self, pt):
        top = self.outfit.get("top", "tshirt")
        col = self.colors["top"]
        hu, P, J, F = self._hu, self.P, self._J, self._F
        L = self._levels()
        t = self._THICK.get(top, 0.04)
        k = 0.09 * hu
        seat = bool(self._pose.get("seat"))
        Rp = F["Rp"]
        # hem level
        hip_c, _, hip_w, hip_df, hip_db = L["hip"]
        if top in ("coat",):
            if seat:
                hem = (L["crotch"][0] + Rp @ _v(0, -0.15, 0.2), Rp, hip_w + 0.12, hip_df + 0.2, hip_db + 0.1)
            else:
                kl, kr = J["knee_l"], J["knee_r"]
                mid = (kl + kr) / 2
                c = hip_c + (mid - hip_c) * 0.92
                spread = float(np.linalg.norm((kl - kr) * np.array([1, 0, 1])))
                hem = (c, Rp, max(hip_w * 1.18, spread / 2 + P["r_knee"] + 0.15), hip_df + 0.12, hip_db + 0.1)
        elif top == "suit":
            hem = (L["crotch"][0] + Rp @ _v(0, 0.05, 0), Rp, hip_w + 0.02, hip_df, hip_db)
        elif top == "jacket":
            hem = (hip_c + Rp @ _v(0, -0.22, 0), Rp, hip_w + 0.01, hip_df, hip_db)
        elif top in ("dress",):
            hem = L["waist"]
        elif top in ("sweater", "hoodie"):
            hem = (hip_c + Rp @ _v(0, -0.16, 0), Rp, hip_w * 0.96, hip_df * 0.96, hip_db * 0.96)
        elif top == "spacesuit":
            hem = L["hip"]
        else:
            hem = (hip_c + Rp @ _v(0, -0.14, 0), Rp, hip_w + 0.01, hip_df, hip_db)
        L2 = dict(L)
        L2["hem"] = hem
        names = ["hem", "hip", "waist", "bust", "chest", "armpit", "shoulder", "neck"]
        if top == "dress":
            names = ["hem", "bust", "chest", "armpit", "shoulder", "neck"]
        hem_y = self.proj(hem[0])[1]
        hip_y = self.proj(hip_c)[1]
        if top in ("tshirt", "shirt", "blouse", "tank", "jacket", "sweater", "hoodie", "suit") and hem_y < hip_y - 1:
            names.remove("hip")
        facing = self._front_z() > -0.3
        skin = self.skin
        # bare shoulders for a tank top
        if top == "tank":
            rings = self._lrings(L2, ("chest", "armpit", "shoulder", "neck"), t=0.0)
            fn0 = _loft_fn(rings, k)
            pt.part(fn0, self._bbox_pts(rings), 3, skin, "skin", r=0.45 * hu, cast=0.15)
            names = names[: names.index("armpit")] + ["armpit"]
        if top == "hoodie" and not self._back:
            self._hood(pt, L, col, behind=True)
        self._neck_drawn = False
        if not facing:
            self._neck(pt, L2, top, col, t, False)
            self._neck_drawn = True
        # loft slightly inset rings and grow them back: same size, rounded hem corners instead of sharp points
        rnd = 0.05
        rings = self._lrings(L2, names, t=t - rnd, tn=t * 0.5 - rnd)
        fn = _loft_fn(rings, k, r=rnd * hu)
        rings = self._lrings(L2, names, t=t, tn=t * 0.5)
        tone = pt.tone(col, "cloth")
        self._torso_fn = fn

        def det(X, Y, D, A, sl):
            if self._simple:
                return
            lw = max(0.7, 0.02 * hu)
            # side folds from the armpit towards the waist on both sides
            for xs in (-1, 1):
                a = self._surf(L2, "armpit", xs * 0.8, t)
                b = self._surf(L2, "waist", xs * 0.55, t)
                if top not in ("spacesuit",) and a[2] > 0 and b[2] > 0:
                    m = ((a[0] * 0.4 + b[0] * 0.6), (a[1] * 0.4 + b[1] * 0.6) + 0.03 * hu)
                    seg = _catmull([a[:2], m, b[:2]], 4)
                    pt.stroke(lambda X, Y, seg=seg: _polyline_sd(X, Y, seg, 0), seg, 2, tone, lw, opacity=0.45,
                              clip=lambda X, Y: fn(X, Y) + 0.06 * hu)
            if top in ("sweater", "hoodie"):
                hw = self._ring(hem[0] + Rp @ _v(0, 0.12, 0), Rp, hem[2] + t, hem[3] + t, hem[4] + t, 24)
                hh = _hull(hw)
                pt.stroke(lambda X, Y: _convex_sd(X, Y, hh), hw, 2, tone, lw * 1.2, opacity=0.8, clip=fn)
            if top == "hoodie" and facing:
                pk = [self._surf(L2, "waist", -0.45, t), self._surf(L2, "waist", 0.45, t),
                      self._surf(L2, "hem", 0.62, t), self._surf(L2, "hem", -0.62, t)]
                pk = [(x, y + 0.08 * hu) for x, y, z in pk]
                pt.stroke(lambda X, Y: _sdf.sd_polygon(X, Y, pk), pk, 2, tone, lw, opacity=0.75, clip=fn)
            if top in ("shirt", "blouse") and facing:
                bl = [self._surf(L2, nm, 0.0, t) for nm in ("chest", "bust", "waist")]
                for x, y, z in bl:
                    if z > 0:
                        pt.flat(lambda X, Y, x=x, y=y: _circ(X, Y, x, y, max(0.6, 0.022 * hu)), [(x, y)], 3,
                                _mixc(col, "#ffffff", 0.5) if np.mean(_col(col)) < 0.6 else tone)
            if top == "spacesuit" and facing:
                for nm, xf, c2 in (("chest", -0.42, "#d64a3d"), ("chest", 0.42, "#3d6fd6")):
                    x, y, z = self._surf(L2, nm, xf, t)
                    pt.flat(lambda X, Y, x=x, y=y: _sdf.sd_box(X, Y, x, y, 0.09 * hu, 0.06 * hu, 0.02 * hu),
                            [(x, y)], 0.1 * hu, c2)
                x, y, z = self._surf(L2, "waist", 0.0, t)
                pt.flat(lambda X, Y: _sdf.sd_box(X, Y, x, y, 0.22 * hu, 0.14 * hu, 0.04 * hu), [(x, y)], 0.3 * hu,
                        "#8f96a3")
            if top in ("jacket", "coat", "suit") and facing:
                self._open_front(pt, L2, top, col, t, fn)
            if "apron" in self.extras and facing:
                self._apron(pt, L2, t)
            if "backpack" in self.extras and facing:
                self._straps(pt, L2, t, fn)
        pt.part(fn, self._bbox_pts(rings), 3, col, "cloth", r=0.62 * hu, cast=0.25, details=det)
        if facing:
            self._neck(pt, L2, top, col, t, facing)
        else:
            if "scarf" in self.extras:
                self._scarf(pt, L2)

    def _open_front(self, pt, L, top, col, t, fn):
        hu = self._hu
        inner = self.colors.get("inner", PALETTE["white"])
        names = ["neck", "shoulder", "armpit", "chest", "bust", "waist", "hip", "hem"]
        names = [n for n in names if n in L]
        if top == "suit":
            widths = dict(neck=0.6, shoulder=0.42, armpit=0.3, chest=0.2, bust=0.12, waist=0.0, hip=0.12, hem=0.3)
        elif top == "coat":
            widths = dict(neck=0.7, shoulder=0.5, armpit=0.36, chest=0.24, bust=0.2, waist=0.18, hip=0.2, hem=0.28)
        else:
            widths = dict(neck=0.75, shoulder=0.55, armpit=0.4, chest=0.3, bust=0.28, waist=0.26, hip=0.26, hem=0.28)
        left, right = [], []
        for nm in names:
            w = widths.get(nm, 0.2)
            a = self._surf(L, nm, -w, t + 0.004)
            b = self._surf(L, nm, w, t + 0.004)
            left.append(a[:2])
            right.append(b[:2])
        poly = left + right[::-1]
        if top == "suit":
            # only the V above the button shows the shirt
            vi = names.index("waist") + 1
            poly = left[:vi] + right[:vi][::-1]
        pf = lambda X, Y: _sdf.sd_polygon(X, Y, poly)
        pt.flat(pf, poly, 2, inner, clip=fn)
        tone = pt.tone(col, "cloth")
        lw = max(0.7, 0.022 * hu)
        # lapels: triangles along the opening edges near the top
        for edge, sgn in ((left, -1), (right, 1)):
            if len(edge) >= 4:
                a, b, c = edge[0], edge[2], edge[3]
                ox = sgn * 0.16 * hu
                lap = [a, (b[0] + ox * 0.9, b[1] - 0.05 * hu), c]
                pt.flat(lambda X, Y, lap=lap: _sdf.sd_polygon(X, Y, lap), lap, 2, _mixc(col, tone, 0.45), clip=fn)
                pt.stroke(lambda X, Y, lap=lap: _sdf.sd_polygon(X, Y, lap), lap, 2, tone, lw, opacity=0.8, clip=fn)
            pt.stroke(lambda X, Y, e=edge: _polyline_sd(X, Y, e, 0), edge, 2, tone, lw, opacity=0.9, clip=fn)
        if top == "suit":
            # tie
            tie_col = self.colors.get("accent", "#b5323f")
            k1 = self._surf(L, "neck", 0.0, t)
            k2 = self._surf(L, "shoulder", 0.0, t)
            w2 = self._surf(L, "bust", 0.0, t)
            kn = [(k1[0] - 0.06 * hu, k1[1]), (k1[0] + 0.06 * hu, k1[1]), (k2[0] + 0.04 * hu, k2[1]),
                  (k2[0] - 0.04 * hu, k2[1])]
            bl = [(k2[0] - 0.035 * hu, k2[1]), (k2[0] + 0.035 * hu, k2[1]), (w2[0] + 0.075 * hu, w2[1]),
                  (w2[0], w2[1] + 0.1 * hu), (w2[0] - 0.075 * hu, w2[1])]
            pt.flat(lambda X, Y: _sdf.sd_polygon(X, Y, bl), bl, 2, tie_col, clip=pf)
            pt.flat(lambda X, Y: _sdf.sd_polygon(X, Y, kn) - 0.01 * hu, kn, 2, _mixc(tie_col, "#000000", 0.15), clip=pf)
            x, y, z = self._surf(L, "waist", 0.0, t)
            pt.flat(lambda X, Y: _circ(X, Y, x, y, max(0.7, 0.03 * hu)), [(x, y)], 2, tone)
        elif top == "coat":
            for nm, xf in (("bust", 0.3), ("waist", 0.3), ("hip", 0.3)):
                if nm in L:
                    x, y, z = self._surf(L, nm, xf, t)
                    if z > 0:
                        pt.flat(lambda X, Y, x=x, y=y: _circ(X, Y, x, y, max(0.7, 0.03 * hu)), [(x, y)], 2, tone)

    def _apron(self, pt, L, t):
        hu = self._hu
        acol = self.colors.get("apron", self.colors.get("accent"))
        names = ["chest", "bust", "waist", "hip", "hem"]
        ws = dict(chest=0.42, bust=0.46, waist=0.62, hip=0.7, hem=0.72)
        left, right = [], []
        for nm in names:
            if nm in L:
                left.append(self._surf(L, nm, -ws[nm], t + 0.02)[:2])
                right.append(self._surf(L, nm, ws[nm], t + 0.02)[:2])
        # skirt of the apron continues to above the knee
        J = self._J
        lo = self.proj(J["pelvis"] + self._F["Rp"] @ _v(0, -1.1 * self.P["Ls"] / 2.13, 0.45))
        ext = 0.3 * hu
        poly = left + [(lo[0] - ext * 1.4, lo[1]), (lo[0] + ext * 1.4, lo[1])][::1] + right[::-1]
        poly = left + [(lo[0] - ext * 1.5, lo[1])] + [(lo[0] + ext * 1.5, lo[1])] + right[::-1]
        pt.part(lambda X, Y: _sdf.sd_polygon(X, Y, _hull(poly)), poly, 3, acol, "cloth", r=0.5 * hu, cast=0.3)
        a = self._surf(L, "chest", -0.42, t)[:2]
        b = self._surf(L, "chest", 0.42, t)[:2]
        n1 = self._surf(L, "neck", -0.9, 0.02)[:2]
        n2 = self._surf(L, "neck", 0.9, 0.02)[:2]
        for s_ in ([a, n1], [b, n2]):
            pt.stroke(lambda X, Y, s_=s_: _polyline_sd(X, Y, s_, 0), s_, 2, acol, max(0.8, 0.035 * hu))

    def _straps(self, pt, L, t, fn):
        hu = self._hu
        bcol = _mixc(self.colors.get("bag", "#3a3f4a"), "#000000", 0.15)
        for xs in (-1, 1):
            a = self._surf(L, "shoulder", xs * 0.75, t + 0.01)
            b = self._surf(L, "armpit", xs * 0.62, t + 0.01)
            c = self._surf(L, "chest", xs * 0.75, t + 0.01)
            seg = _catmull([a[:2], b[:2], c[:2]], 3)
            pt.part(lambda X, Y, seg=seg: _polyline_sd(X, Y, seg, 0.045 * hu), seg, 0.06 * hu, bcol, "dark",
                    r=0.04 * hu, cast=0.4, clip=lambda X, Y: fn(X, Y) - 0.06 * hu)

    def _hood(self, pt, L, col, behind=True):
        hu, P = self._hu, self.P
        c, R, w, df, db = L["neck"]
        c2 = c + R @ _v(0, -0.02, -0.06)
        ring = self._ring(c2, R, w + 0.2, df + 0.12, db + 0.24, 20, tilt=-0.06)
        h = _hull(ring)
        pt.part(lambda X, Y: _convex_sd(X, Y, h) - 0.04 * hu, ring, 0.06 * hu, col, "cloth", r=0.18 * hu, cast=0.3)

    def _neckline(self, L, top, t):
        """Front neckline polyline (screen) of a top, or None."""
        c, R, w, df, db = L["neck"]
        rr = w + 0.05
        drop = dict(tshirt=0.1, tank=0.26, dress=0.22, sweater=0.06, hoodie=0.06, blouse=0.3, shirt=0.22,
                    spacesuit=0.0, jacket=0.1, coat=0.1, suit=0.1).get(top, 0.1)
        vee = top in ("blouse", "shirt")
        if top == "tank":
            rr = w + 0.18
        if top == "dress":
            rr = w + 0.12
        pts, ring = [], []
        for i in range(24):
            a = 2 * math.pi * i / 24
            ca, sa = math.cos(a), math.sin(a)
            if sa >= 0:
                dy = -drop * (1 - abs(ca)) ** 1.4 if vee else -drop * sa ** 2
                p3 = c + R @ _v(rr * ca, dy - 0.02, (df + 0.03) * sa * 1.05)
            else:
                p3 = c + R @ _v(rr * ca, 0.02 * sa * sa - 0.02, (db + 0.02) * sa)
            p2 = self.proj(p3)
            ring.append(p2)
            if sa >= -1e-9 and i <= 12:
                pts.append(p2)
        self._neck_ring = ring
        return pts

    def _neck(self, pt, L, top, col, t, facing):
        hu, P, J = self._hu, self.P, self._J
        a = self.proj(J["neck"] + self._F["R2"] @ _v(0, -0.12, -0.02))
        b = self.proj(J["neck_top"])
        rn = P["r_neck"] * hu
        q = [a, b]
        fn = lambda X, Y: _cone(X, Y, a, b, rn * 1.08, rn * 0.88)
        skin = self.skin
        # chin / jaw cast shadow on the upper neck
        hb = self._hc

        def det(X, Y, D, A, sl):
            if self.style == "line":
                return
            jaw = lambda X, Y: _ell(X, Y, hb[0] + (b[0] - hb[0]) * 0.25, hb[1] + 0.12 * hu, 0.36 * hu, 0.42 * hu)
            pt.shade_band(jaw, [hb, b], 0.5 * hu, _TONE["flat"]["skin"], opacity=0.9, soft=max(1.0, 0.03 * hu))
        clip = None
        nl = None
        if facing and top not in ("spacesuit",):
            nl = self._neckline(L, top, t)
            up = 3 * hu
            ring = self._neck_ring
            hh = _hull(ring + [(x, y - up) for x, y in ring])
            hole = lambda X, Y: _convex_sd(X, Y, hh)
            # skin inside the neckline opening
            tf = getattr(self, "_torso_fn", None)
            if tf is not None:
                pt.part(lambda X, Y: np.maximum(hole(X, Y), tf(X, Y)), nl, 3, skin, "skin", r=0.3 * hu, cast=0.0,
                        outline=False)
        if nl is not None:
            clip = hole
        if not self._neck_drawn:
            pt.part(fn, q, rn + 2, skin, "skin", r=rn, cast=0.1, details=det, outline=True, clip=clip)
        if nl is not None:
            tone = pt.tone(col, "cloth")
            band = max(0.8, 0.035 * hu)
            if top in ("tshirt", "sweater", "hoodie", "dress", "tank"):
                pt.stroke(lambda X, Y: _polyline_sd(X, Y, nl, 0), nl, band, tone, band, opacity=0.9)
                if self.style in ("line", "anime"):
                    pt.stroke(lambda X, Y: _polyline_sd(X, Y, nl, 0), nl, 2, pt.ink_for(col), pt.lw)
            if top in ("shirt", "blouse", "coat", "suit"):
                self._collar(pt, L, top, col, nl)
        if top == "hoodie" and facing and not self._simple:
            for xs in (-0.35, 0.35):
                p0 = nl[6 + int(xs * 8)] if nl else self.proj(J["neck"])
                seg = [p0, (p0[0] + xs * 0.04 * hu, p0[1] + 0.45 * hu)]
                pt.stroke(lambda X, Y, seg=seg: _polyline_sd(X, Y, seg, 0), seg, 2, _mixc(col, "#ffffff", 0.7),
                          max(0.7, 0.018 * hu))
        if top == "spacesuit":
            c, R, w, df, db = L["neck"]
            ring = self._ring(c + R @ _v(0, 0.03, 0), R, w + 0.14, df + 0.14, db + 0.14, 20)
            h = _hull(ring)
            pt.part(lambda X, Y: np.abs(_convex_sd(X, Y, h)) - 0.045 * hu, ring, 0.08 * hu, "#b8bec8", "dark",
                    r=0.06 * hu, cast=0.2)
        if "scarf" in self.extras and facing:
            self._scarf(pt, L)
        if "stethoscope" in self.extras and facing:
            self._stethoscope(pt, L, t)
        if "tie" in self.extras and top != "suit" and facing:
            self._tie(pt, L, t)

    def _collar(self, pt, L, top, col, nl):
        hu = self._hu
        if top in ("shirt", "blouse"):
            ccol = col
        elif top == "suit":
            ccol = self.colors.get("inner", "#f5f2ec")
        else:
            ccol = col
        tone = pt.tone(ccol, "cloth")
        mid = nl[6]
        for side in (0, 12):
            e = nl[side]
            m = nl[side // 2 + 3] if side == 0 else nl[9]
            dx = e[0] - mid[0]
            flap = [e, (e[0] + dx * 0.05, e[1] - 0.08 * hu), (m[0] + dx * 0.35, m[1] + 0.2 * hu), (mid[0], mid[1] + 0.02 * hu)]
            pt.part(lambda X, Y, f=flap: _sdf.sd_polygon(X, Y, f) - 0.008 * hu, flap, 3, ccol, "cloth", r=0.1 * hu,
                    cast=0.35, edge=tone)

    def _scarf(self, pt, L):
        hu = self._hu
        scol = self.colors.get("scarf", self.colors.get("accent"))
        c, R, w, df, db = L["neck"]
        r1 = self._ring(c + R @ _v(0, 0.06, 0.0), R, w + 0.06, df + 0.08, db + 0.06, 20)
        r0 = self._ring(c + R @ _v(0, -0.1, 0.02), R, w + 0.16, df + 0.16, db + 0.1, 20)
        h = _hull(r0 + r1)
        fn = lambda X, Y: _convex_sd(X, Y, h) - 0.03 * hu
        tone = pt.tone(scol, "cloth")
        tails = []
        if self._front_z() > -0.2:
            for dx, ln, wd in ((0.16, 0.8, 0.1), (0.3, 0.62, 0.09)):
                a = self.proj(c + R @ _v(dx, -0.12, df + 0.16))
                b = self.proj(c + R @ _v(dx + 0.06, -ln, df + 0.2))
                tails.append((a, b, wd))
        for a, b, wd in tails[::-1]:
            pt.part(lambda X, Y, a=a, b=b, wd=wd: np.maximum(_cone(X, Y, a, b, wd * hu, wd * 1.1 * hu),
                                                             _plane(X, Y, b, _unit(np.array(b) - np.array(a)))),
                    [a, b], 0.14 * hu, scol, "cloth", r=0.08 * hu, cast=0.35, edge=tone)

        def det(X, Y, D, A, sl):
            if not self._simple:
                mid = self._ring(c + R @ _v(0, -0.02, 0.01), R, w + 0.12, df + 0.13, db + 0.08, 20)
                fr = [p for p, q in zip(mid, r0) if True][1:9]
                pt.stroke(lambda X, Y: _polyline_sd(X, Y, fr, 0), fr, 2, tone, max(0.7, 0.02 * hu), opacity=0.6,
                          clip=fn)
        pt.part(fn, r0 + r1, 0.06 * hu, scol, "cloth", r=0.12 * hu, cast=0.35, details=det)

    def _stethoscope(self, pt, L, t):
        hu = self._hu
        a = self._surf(L, "shoulder", -0.7, t + 0.02)[:2]
        b = self._surf(L, "chest", -0.3, t + 0.02)[:2]
        c = self._surf(L, "shoulder", 0.7, t + 0.02)[:2]
        d = self._surf(L, "chest", 0.35, t + 0.02)[:2]
        e = self._surf(L, "bust", 0.25, t + 0.02)[:2]
        for seg in (_catmull([a, b, (b[0], b[1] + 0.1 * hu)], 3), _catmull([c, d, e], 3)):
            pt.stroke(lambda X, Y, s=seg: _polyline_sd(X, Y, s, 0), seg, 2, "#3a3f4a", max(0.8, 0.028 * hu))
        pt.part(lambda X, Y: _circ(X, Y, e[0], e[1] + 0.04 * hu, 0.055 * hu), [e], 0.08 * hu, "#c9ccd2", "dark",
                r=0.05 * hu, cast=0.4)

    def _tie(self, pt, L, t):
        hu = self._hu
        tie_col = self.colors.get("accent", "#b5323f")
        k1 = self._surf(L, "neck", 0.0, t)
        w2 = self._surf(L, "waist", 0.0, t)
        bl = [(k1[0] - 0.05 * hu, k1[1]), (k1[0] + 0.05 * hu, k1[1]), (w2[0] + 0.07 * hu, w2[1] - 0.05 * hu),
              (w2[0], w2[1] + 0.05 * hu), (w2[0] - 0.07 * hu, w2[1] - 0.05 * hu)]
        pt.part(lambda X, Y: _sdf.sd_polygon(X, Y, bl), bl, 2, tie_col, "cloth", r=0.05 * hu, cast=0.4)

    # ------------------------------------------------------------ arms
    def _arm_pts(self, side):
        J, F, P = self._J, self._F, self.P
        sh, el, wr = J["shoulder_" + side], J["elbow_" + side], J["wrist_" + side]
        R2 = F["R2"]
        sg = 1.0 if side == "l" else -1.0
        b1 = sh + (el - sh) * 0.42 + R2 @ _v(sg * 0.02, 0, 0.02) * P["q"]
        f1 = el + (wr - el) * 0.26
        top = sh + R2 @ _v(-sg * 0.03, -0.1, 0)
        return top, b1, el, f1, wr

    def _sleeve_kind(self):
        top = self.outfit.get("top")
        if self.outfit.get("sleeves"):
            return self.outfit["sleeves"]
        if top in ("tshirt", "dress"):
            return "short"
        if top == "tank":
            return "none"
        if top == "blouse":
            return "three_quarter" if self.outfit.get("long_sleeves") is False else "long"
        if self.outfit.get("long_sleeves") is False:
            return "short"
        return "long"

    def _arm(self, pt, side, seg):
        P, hu = self.P, self._hu
        top, b1, el, f1, wr = self._arm_pts(side)
        q = [self.proj(p) for p in (top, b1, el, f1, wr)]
        rp = [P["r_delt"] * 0.88 * hu, P["r_bicep"] * hu, P["r_elbow"] * hu, P["r_fore"] * hu, P["r_wrist"] * hu]
        k = 0.1 * hu * P["q"]
        sk = self._sleeve_kind()
        tkind = self.outfit.get("top")
        col = self.colors["top"]
        t = self._THICK.get(tkind, 0.04)
        upper = seg == "upper"
        idx = [0, 1, 2] if upper else [2, 3, 4]
        qq = [q[i] for i in idx]
        rr = [rp[i] for i in idx]
        bare = sk in ("none", "short") or (sk == "three_quarter" and not upper)
        if bare or sk == "three_quarter":
            nl = None if upper else (lambda X, Y: _chain(X, Y, q[:3], [rp[0], rp[1], rp[2] * 0.8], k))
            pt.part(lambda X, Y: _chain(X, Y, qq, rr, k), qq, max(rr) + 2, self.skin, "skin", r=rr[0] * 0.9,
                    cast=0.35, noline=nl)
        if sk == "short" and upper:
            a, b = np.array(q[0]), np.array(q[2])
            d = _unit(b - a)
            cut = tuple(a + (b - a) * 0.5)
            s3 = [q[0], q[1], cut]
            r3 = [rp[0] + (t + 0.04) * hu, rp[1] + (t + 0.06) * hu, rp[1] + (t + 0.07) * hu]
            fn = lambda X, Y: np.maximum(_chain(X, Y, s3, r3, k), _plane(X, Y, cut, d))
            tone = pt.tone(col, "cloth")

            def det(X, Y, D, A, sl):
                hb = (cut[0] - d[0] * 0.04 * hu, cut[1] - d[1] * 0.04 * hu)
                if not self._simple:
                    pt.stroke(lambda X, Y: _plane(X, Y, hb, d), [hb], r3[2] + 2, tone, max(0.6, 0.016 * hu),
                              opacity=0.6, clip=fn)
            pt.part(fn, s3, max(r3) + 2, col, "cloth", r=r3[0] * 0.85, cast=0.35, details=det)
            return
        if bare:
            return
        if sk == "three_quarter" and not upper:
            return
        extra = (t * 0.55 + 0.02) * hu
        if tkind == "spacesuit":
            extra = 0.12 * hu
        rr2 = [x + extra for x in rr]
        if upper:
            rr2[0] = rp[0] + (t * 0.55 + 0.015) * hu
            rr2[2] = rp[2] + extra * 1.25
            fn = lambda X, Y: _chain(X, Y, qq, rr2, k)
            pts = qq
        else:
            a, b = np.array(q[2]), np.array(q[4])
            d = _unit(b - a)
            L_ = float(np.linalg.norm(b - a))
            cut = tuple(b - d * 0.03 * hu)
            rr2 = [rp[2] + extra * 1.25, rp[3] + extra, rp[4] + extra * 1.1]
            if sk == "three_quarter":
                cut = tuple(a + d * L_ * 0.4)
            qq2 = [q[2], q[3], cut]
            fn = lambda X, Y: np.maximum(_chain(X, Y, qq2, rr2, k), _plane(X, Y, cut, d))
            pts = qq2
        tone = pt.tone(col, "cloth")
        ang = self._elbow_angle(side)

        def det(X, Y, D, A, sl):
            if self._simple:
                return
            lw = max(0.7, 0.02 * hu)
            if not upper:
                cut2 = pts[-1]
                a, b = np.array(q[2]), np.array(q[4])
                d = _unit(b - a)
                cb = (cut2[0] - d[0] * 0.07 * hu, cut2[1] - d[1] * 0.07 * hu)
                pt.stroke(lambda X, Y: _plane(X, Y, cb, d), [cb], rr2[-1] + 2, tone, lw, opacity=0.7, clip=fn)
            if ang > 35:
                ex, ey = q[2]
                u = _unit(np.array(q[0]) - np.array(q[2]))
                f = _unit(np.array(q[4]) - np.array(q[2]))
                inner = _unit(u + f)
                for j, s_ in enumerate((0.12, 0.2)):
                    cx, cy = ex + inner[0] * s_ * hu, ey + inner[1] * s_ * hu
                    seg2 = [(cx - inner[1] * 0.08 * hu, cy + inner[0] * 0.08 * hu), (cx, cy),
                            (cx + inner[1] * 0.08 * hu, cy - inner[0] * 0.08 * hu)]
                    pt.stroke(lambda X, Y, s2=seg2: _polyline_sd(X, Y, s2, 0), seg2, 2, tone, lw, opacity=0.55,
                              clip=lambda X, Y: fn(X, Y) + 0.03 * hu)
        nl = None if upper else (lambda X, Y: _chain(X, Y, q[:3], [rp[0], rp[1], rp[2] * 0.8], k))
        pt.part(fn, pts, max(rr2) + 2, col, "cloth", r=rr2[0] * 0.85, cast=0.35, details=det, noline=nl)

    def _elbow_angle(self, side):
        u = self._F["upper_" + side]
        f = self._F["fore_" + side]
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, f))))))

    # ------------------------------------------------------------ hands
    _CURL = {
        "open": ([(4, 4, 2)] * 4, 12, "open"),
        "wave": ([(0, 0, 0)] * 4, 16, "open"),
        # fingers lie together with a gentle, graded curl (stronger curls read as a claw at figure scale)
        "relaxed": ([(10, 20, 12), (14, 24, 14), (18, 28, 16), (24, 32, 18)], 0, "relaxed"),
        "fist": ([(88, 100, 60)] * 4, 0, "fist"),
        "grip": ([(42, 55, 40), (48, 58, 42), (52, 60, 44), (56, 62, 45)], 2, "grip"),
        "point": ([(0, 0, 0), (88, 100, 60), (88, 100, 60), (88, 100, 60)], 2, "fist"),
        "peace": ([(0, 0, 0), (0, 0, 0), (88, 100, 60), (88, 100, 60)], 10, "fist"),
        "thumbs_up": ([(88, 100, 60)] * 4, 0, "up"),
    }

    def _hand_geom(self, side):
        J, F, P = self._J, self._F, self.P
        hd, n, t = F["hand_" + side]
        W = J["wrist_" + side]
        L = P["hand_len"]
        wd = 0.47 * L
        pl = 0.47 * L
        shape = self._pose.get("hand_" + side, "relaxed")
        curls, spread, tmode = self._CURL.get(shape, self._CURL["relaxed"])
        palm = []
        for pnt in (W + t * 0.32 * wd, W - t * 0.3 * wd, W + hd * pl + t * 0.46 * wd, W + hd * pl - t * 0.44 * wd,
                    W + hd * 0.3 * L + t * 0.5 * wd, W + hd * 0.25 * L - t * 0.5 * wd):
            palm.append(pnt + n * 0.08 * L)
            palm.append(pnt - n * 0.1 * L)
        fingers = []
        offs = [0.34, 0.11, -0.12, -0.33]
        back = [0.0, -0.02, 0.0, 0.06]
        lens = [0.5, 0.56, 0.52, 0.41]
        rads = [0.1, 0.105, 0.1, 0.086]
        spr = [1.2, 0.35, -0.45, -1.3]
        for i in range(4):
            K = W + hd * (pl - back[i] * L) + t * offs[i] * wd
            a = math.radians(spread * spr[i])
            d0 = _unit(hd * math.cos(a) + t * math.sin(a))
            pts = [K]
            cum = 0.0
            p_ = K
            for j, fr in enumerate((0.45, 0.31, 0.24)):
                cum += math.radians(curls[i][j])
                dj = d0 * math.cos(cum) + n * math.sin(cum)
                p_ = p_ + dj * lens[i] * L * fr
                pts.append(p_)
            fingers.append((pts, [rads[i] * L, rads[i] * L * 0.92, rads[i] * L * 0.84, rads[i] * L * 0.76]))
        B = W + hd * 0.1 * L + t * 0.3 * wd + n * 0.04 * L
        if tmode == "open":
            d1, d2 = _unit(hd * 0.5 + t * 0.85 + n * 0.15), _unit(hd * 0.75 + t * 0.6 + n * 0.1)
        elif tmode == "relaxed":
            # thumb resting along the index finger, not splayed out
            d1, d2 = _unit(hd * 0.78 + t * 0.22 + n * 0.45), _unit(hd * 0.92 - t * 0.04 + n * 0.38)
        elif tmode == "grip":
            d1, d2 = _unit(hd * 0.55 + t * 0.2 + n * 0.8), _unit(hd * 0.6 - t * 0.3 + n * 0.7)
        elif tmode == "up":
            d1, d2 = _unit(hd * 0.15 + t * 1.0 + n * 0.2), _unit(t * 1.0 - hd * 0.1)
        else:   # fist
            d1, d2 = _unit(hd * 0.55 + t * 0.15 + n * 0.8), _unit(hd * 0.35 - t * 0.85 + n * 0.45)
        T1 = B + d1 * 0.3 * L
        T2 = T1 + d2 * 0.25 * L
        thumb = ([B, T1, T2], [0.14 * L, 0.115 * L, 0.095 * L])
        return palm, fingers, thumb, L

    def _hand(self, pt, side):
        hu = self._hu
        top = self.outfit.get("top")
        col = self.skin if top != "spacesuit" else "#e8eaee"
        kind = "skin" if top != "spacesuit" else "cloth"
        palm, fingers, thumb, L = self._hand_geom(side)
        Lp = L * hu
        P2 = [self.proj(p) for p in palm]
        tone = pt.tone(col, kind)
        if Lp < 6:
            c = self.proj(self._J["hand_" + side])
            pt.part(lambda X, Y: _circ(X, Y, c[0], c[1], max(Lp * 0.33, 0.8)), [c], Lp, col, kind, r=Lp * 0.3,
                    cast=0.0)
            return
        if Lp < 13:
            tips = [self.proj(f[0][-1]) for f in fingers] + [self.proj(f[0][1]) for f in fingers]
            h = _hull(P2 + tips)
            tq = [self.proj(p) for p in thumb[0]]
            tr = [r * hu for r in thumb[1]]
            fn = lambda X, Y: _smin(_convex_sd(X, Y, h) - 0.04 * Lp, _chain(X, Y, tq, tr, 1.0), 0.06 * Lp)
            pt.part(fn, P2 + tips + tq, Lp * 0.2 + 2, col, kind, r=Lp * 0.25, cast=0.3)
            return
        items = []
        hp = _hull(P2)
        zp = float(np.mean([self.projz(p) for p in palm]))
        items.append((zp, "palm", (lambda X, Y: _convex_sd(X, Y, hp) - 0.05 * Lp), P2, 0.25 * Lp))
        for pts3, rads in fingers + [thumb]:
            q = [self.proj(p) for p in pts3]
            rr = [r * hu for r in rads]
            z = float(np.mean([self.projz(p) for p in pts3[1:]]))
            items.append((z, "f", (lambda X, Y, q=q, rr=rr: _chain(X, Y, q, rr, 0.04 * Lp)), q, rr[0]))
        items.sort(key=lambda t: t[0])
        edge = tone if self.style in ("flat", "soft") else None
        shape = self._pose.get("hand_" + side, "relaxed")
        if shape in ("relaxed", "fist", "grip", "thumbs_up") and self.style != "line":
            # closed hands: palm and fingers as one silhouette with light separation lines (separate outlined
            # sausages read as a claw at figure scale); the thumb stays its own part
            tq = [self.proj(p) for p in thumb[0]]
            body = [it for it in items if it[3] != tq]
            thumb_it = [it for it in items if it[3] == tq]
            fns = [it[2] for it in body]
            pts_all = [p for it in body for p in it[3]]
            kk = 0.05 * Lp

            def hand_fn(X, Y, fns=fns):
                d = fns[0](X, Y)
                for f in fns[1:]:
                    d = _smin(d, f(X, Y), kk)
                return d
            thumb_behind = bool(thumb_it) and thumb_it[0][0] < zp - 0.02
            if thumb_behind:
                for z, nm, fn, pts, r in thumb_it:
                    pt.part(fn, pts, r + 3, col, kind, r=r, cast=0.25, edge=edge, edge_w=max(0.6, Lp * 0.035))
                thumb_it = []
            res = pt.part(hand_fn, pts_all, 0.3 * Lp + 3, col, kind, r=0.25 * Lp, cast=0.3)
            if res is not None and not self._simple:
                ew = max(0.6, Lp * 0.03)
                fz = sorted([it for it in body if it[1] == "f"], key=lambda t: t[0])
                for z, nm, fn, pts, r in fz[1:]:
                    pt.stroke(fn, pts, r + 3, tone, ew, opacity=0.55, clip=lambda X, Y: hand_fn(X, Y) + ew)
            for z, nm, fn, pts, r in thumb_it:
                pt.part(fn, pts, r + 3, col, kind, r=r, cast=0.25, edge=edge, edge_w=max(0.6, Lp * 0.035))
            return
        for z, nm, fn, pts, r in items:
            pt.part(fn, pts, r + 3, col, kind, r=r * (1.0 if nm == "f" else 1.0), cast=0.25 if nm == "f" else 0.3,
                    edge=edge if nm == "f" else None, edge_w=max(0.6, Lp * 0.035))

    # ------------------------------------------------------------ carried things
    def _backpack(self, pt):
        hu, P, J, F = self._hu, self.P, self._J, self._F
        R2 = F["R2"]
        c = J["chest"] + R2 @ _v(0, -0.12, -(0.4 * P["depth"] + 0.26))
        hw, hh, hd = P["chest"] * 0.28, 0.5, 0.2
        # a pack with a domed top and a rounded bottom (box corners pulled in), not a plain slab
        pts = [self.proj(c + R2 @ _v(x * hw * (0.72 if y > 0 else 1.0), y * hh, z * hd * (0.7 if y > 0 else 1.0)))
               for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        pts += [self.proj(c + R2 @ _v(0, hh + 0.08, 0))]
        h = _hull(pts)
        col = self.colors.get("bag", PALETTE["mustard"])
        fn = lambda X, Y: _convex_sd(X, Y, h) - 0.14 * hu
        tone = pt.tone(col, "cloth")

        def det(X, Y, D, A, sl):
            if self._simple:
                return
            pk = [self.proj(c + R2 @ _v(x * hw * 0.7, y, -hd * 1.05)) for x in (-1, 1) for y in (-0.45, 0.05)]
            hk = _hull(pk)
            if self.projz(c + R2 @ _v(0, 0, -1)) > self.projz(c):
                pt.stroke(lambda X, Y: _convex_sd(X, Y, hk) - 0.04 * hu, pk, 2, tone, max(0.8, 0.025 * hu))
        pt.part(fn, pts, 0.1 * hu, col, "cloth", r=0.35 * hu, cast=0.3, details=det)

    def _shoulder_bag(self, pt):
        hu, P, J, F = self._hu, self.P, self._J, self._F
        R2, Rp = F["R2"], F["Rp"]
        col = self.colors.get("bag", "#7a5236")
        L = self._levels()
        t = self._THICK.get(self.outfit.get("top"), 0.04) + 0.02
        bc = J["pelvis"] + Rp @ _v(-0.62 * P["hip"] - 0.05, 0.05, 0.12)
        hw, hh, hd = 0.36, 0.28, 0.1
        pts = [self.proj(bc + Rp @ _v(x * hw, y * hh, z * hd)) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        h = _hull(pts)
        fz = self._front_z()
        a = self._surf(L, "shoulder", 0.8, t, front=fz > -0.2)[:2]
        m = self._surf(L, "bust", 0.0, t, front=fz > -0.2)[:2]
        b = self.proj(bc + Rp @ _v(0.25, hh, 0))
        strap = _catmull([a, m, b], 4)
        pt.part(lambda X, Y: _polyline_sd(X, Y, strap, 0.028 * hu), strap, 0.05 * hu, _mixc(col, "#000000", 0.25),
                "dark", r=0.03 * hu, cast=0.4)
        tone = pt.tone(col, "dark")
        bagf = lambda X, Y: _convex_sd(X, Y, h) - 0.04 * hu

        def det(X, Y, D, A, sl):
            flap = [self.proj(bc + Rp @ _v(x * hw, y, hd * 1.02)) for x in (-1, 1) for y in (hh, -0.02)]
            hf = _hull(flap)
            pt.flat(lambda X, Y: _convex_sd(X, Y, hf) - 0.02 * hu, flap, 2, _mixc(col, tone, 0.45), clip=bagf)
        pt.part(bagf, pts, 0.06 * hu, col, "dark", r=0.25 * hu, cast=0.35,
                details=det)

    def _item(self, pt, item):
        hu = self._hu
        side = "r"
        J, F = self._J, self._F
        hd, n, t = F["hand_" + side]
        g = self.proj(J["hand_" + side])
        L = self.P["hand_len"]
        acc = self.colors.get("accent", PALETTE["mustard"])
        if item == "cup":
            c = (g[0], g[1] - 0.04 * hu)
            w, h = 0.13 * hu, 0.2 * hu
            fn = lambda X, Y: _sdf.sd_box(X, Y, c[0], c[1], w, h, 0.03 * hu)
            body = self.colors.get("cup", "#f2ede4")

            def det(X, Y, D, A, sl):
                pt.flat(lambda X, Y: _sdf.sd_box(X, Y, c[0], c[1] + 0.02 * hu, w + 0.01 * hu, 0.07 * hu),
                        [c], w, "#a8744a", clip=fn)
                pt.flat(lambda X, Y: _sdf.sd_box(X, Y, c[0], c[1] - h - 0.02 * hu, w + 0.02 * hu, 0.04 * hu,
                                                 0.02 * hu), [c], w + h, "#3a3034")
            pt.part(fn, [(c[0] - w, c[1] - h), (c[0] + w, c[1] + h)], 4, body, "cloth", r=w, cast=0.3, details=det)
        elif item == "phone":
            hx, hy, _ = self.dir2(hd)
            ang = math.degrees(math.atan2(hy, hx)) - 90
            c = self.proj(J["wrist_" + side] + hd * L * 0.75 + n * 0.08)
            fn = lambda X, Y: _sdf.sd_box(X, Y, c[0], c[1], 0.09 * hu, 0.17 * hu, 0.025 * hu, angle=ang)
            pt.part(fn, [c], 0.22 * hu, "#2b2d36", "dark", r=0.08 * hu, cast=0.3)
            pt.flat(lambda X, Y: _sdf.sd_box(X, Y, c[0], c[1], 0.07 * hu, 0.145 * hu, 0.012 * hu, angle=ang), [c],
                    0.2 * hu, "#7fb3e8", opacity=0.9 if self.dir2(n)[2] < 0.3 else 0.0)
        elif item == "book":
            c = (g[0], g[1] - 0.05 * hu)
            fn = lambda X, Y: _sdf.sd_box(X, Y, c[0], c[1], 0.24 * hu, 0.32 * hu, 0.02 * hu, angle=-8)
            pt.part(fn, [c], 0.45 * hu, self.colors.get("book", PALETTE["terracotta"]), "cloth", r=0.2 * hu,
                    cast=0.3)
        elif item in ("briefcase", "bag"):
            c = (g[0], g[1] + 0.42 * hu)
            w, h = (0.42, 0.3) if item == "briefcase" else (0.34, 0.36)
            col = self.colors.get("bag", "#5a3a26" if item == "briefcase" else PALETTE["sand"])
            hc = (g[0], g[1] + 0.04 * hu)
            pt.stroke(lambda X, Y: _sdf.sd_box(X, Y, hc[0], hc[1] + 0.04 * hu, 0.11 * hu, 0.08 * hu, 0.06 * hu),
                      [hc], 0.2 * hu, _mixc(col, "#000000", 0.35), max(1.0, 0.035 * hu))
            fn = lambda X, Y: _sdf.sd_box(X, Y, c[0], c[1], w * hu, h * hu, 0.05 * hu)
            pt.part(fn, [c], (w + h) * hu, col, "dark", r=0.3 * hu, cast=0.3)
        elif item == "ball":
            c = self.proj(J["hand_" + side] + n * 0.32)
            pt.part(lambda X, Y: _circ(X, Y, c[0], c[1], 0.32 * hu), [c], 0.36 * hu, acc, "cloth", r=0.3 * hu,
                    cast=0.3)
        elif item == "flower":
            top_ = (g[0] + 0.05 * hu, g[1] - 0.75 * hu)
            pt.stroke(lambda X, Y: _polyline_sd(X, Y, [(g[0], g[1] + 0.1 * hu), top_], 0), [g, top_], 2, "#4c8a4a",
                      max(1.0, 0.03 * hu))
            for i in range(5):
                a = 2 * math.pi * i / 5
                px_, py_ = top_[0] + math.cos(a) * 0.09 * hu, top_[1] + math.sin(a) * 0.09 * hu
                pt.part(lambda X, Y, px_=px_, py_=py_: _circ(X, Y, px_, py_, 0.08 * hu), [(px_, py_)], 0.1 * hu,
                        self.colors.get("flower", PALETTE["coral"]), "cloth", r=0.06 * hu, cast=0.1)
            pt.flat(lambda X, Y: _circ(X, Y, top_[0], top_[1], 0.05 * hu), [top_], 0.06 * hu, PALETTE["yellow"])
        elif item == "guitar":
            self._guitar(pt)

    def _guitar(self, pt):
        hu, J, F = self._hu, self._J, self._F
        R2 = F["R2"]
        s = self.P["N"] / 7.4
        body = np.array(self.proj(J["pelvis"] + R2 @ _v(-0.3, 0.42 * s, 0.5)))
        b = np.array(self.proj(J["hand_l"]))
        d = _unit(b - body)
        nrm = np.array([-d[1], d[0]])
        ang = math.degrees(math.atan2(d[1], d[0]))
        g = hu * s ** 0.5 * 1.35
        c1 = body - d * 0.22 * g
        c2 = body + d * 0.42 * g
        wood = self.colors.get("guitar", "#c98a4b")
        dark = "#4a2f20"
        neck_a = body + d * 0.7 * g
        neck_b = b + d * 0.35 * g
        head = neck_b + d * 0.2 * g
        pt.part(lambda X, Y: _cone(X, Y, tuple(neck_a), tuple(neck_b), 0.075 * g, 0.06 * g), [tuple(neck_a), tuple(neck_b)],
                0.1 * g, dark, "dark", r=0.06 * g, cast=0.2)
        pt.part(lambda X, Y: _sdf.sd_box(X, Y, head[0], head[1], 0.17 * g, 0.075 * g, 0.03 * g, angle=ang),
                [tuple(head)], 0.2 * g, dark, "dark", r=0.06 * g, cast=0.2)
        fn = lambda X, Y: _smin(_ell(X, Y, c1[0], c1[1], 0.5 * g, 0.46 * g, ang),
                                _ell(X, Y, c2[0], c2[1], 0.37 * g, 0.34 * g, ang), 0.14 * g)

        def det(X, Y, D, A, sl):
            hole = body + d * 0.2 * g
            pt.flat(lambda X, Y: np.abs(_circ(X, Y, hole[0], hole[1], 0.13 * g)) - 0.02 * g, [tuple(hole)], 0.2 * g,
                    _mixc(wood, dark, 0.5))
            pt.flat(lambda X, Y: _circ(X, Y, hole[0], hole[1], 0.11 * g), [tuple(hole)], 0.12 * g, "#2a1d16")
            br = body - d * 0.32 * g
            pt.flat(lambda X, Y: _sdf.sd_box(X, Y, br[0], br[1], 0.04 * g, 0.16 * g, 0.02 * g, angle=ang), [tuple(br)],
                    0.2 * g, dark)
            for k in (-1.5, -0.5, 0.5, 1.5):
                o = nrm * k * 0.022 * g
                seg = [tuple(br + o), tuple(head - d * 0.15 * g + o * 0.7)]
                pt.flat(lambda X, Y, seg=seg: _polyline_sd(X, Y, seg, max(0.35, 0.006 * g)), seg, 2, "#efe6d6",
                        opacity=0.8)
        pt.part(fn, [tuple(c1), tuple(c2)], 0.6 * g, wood, "cloth", r=0.45 * g, cast=0.3, details=det)

    # ------------------------------------------------------------ head
    def _hs(self):
        """(sin, cos, |sin|, head width) of the head's view yaw."""
        s, c = math.sin(self._hyaw), math.cos(self._hyaw)
        return s, c, abs(s), self.P["head_w"]

    def _skull_local(self):
        s, c, a, Wh = self._hs()
        young = self.build in ("child", "chibi")
        crx = Wh * 0.5 * (1 + 0.2 * a)
        jrx = Wh * (0.43 if not young else 0.47) * (1.06 if self.P["jaw"] else 1.0)
        jry = 0.36 if not young else 0.33
        jcy = 0.13 if not young else 0.15
        w = float(_smooth01((a - 0.62) / 0.33))
        sg = 1.0 if s >= 0 else -1.0
        base = _PROFILE_YOUNG if young else _PROFILE
        prof = [(sg * x_, y_) for x_, y_ in _catmull(list(base) + [base[0], base[1]], 3)[3:-2]]

        def f(x, y):
            d = _smin(_ell(x, y, -s * 0.07, -0.07, crx, 0.43), _ell(x, y, s * 0.11, jcy, jrx * (1 - 0.12 * a), jry),
                      0.12)
            if w > 0.01:
                e = _smin(_ell(x, y, -sg * 0.07, -0.07, crx, 0.43), _sdf.sd_polygon(x, y, prof), 0.04)
                d = d * (1 - w) + e * w
            return d
        return f

    def _expression(self):
        e = self.face.get("expression") or self._pose.get("face") or "neutral"
        return e if e in EXPRESSIONS else "neutral"

    def _head(self, pt):
        hu = self._hu
        s, c, a, Wh = self._hs()
        sk = self._skull_local()
        skin = self.skin
        style = self.hair.get("style", "short")
        big = 1.6 if style in ("afro", "long", "curly") else 1.15
        pts = self._hpts(big)
        if not self._back or True:
            self._hair_back(pt, layer="head")
        ears = self._ears()
        for e in ears:
            if not e[3]:
                self._ear(pt, e)
        skull_fn = self._hf(sk)
        self._skull_fn = skull_fn

        def det(X, Y, D, A, sl):
            if c > -0.15 and not self._tiny:
                self._face(pt, skull_fn)
        pt.part(skull_fn, pts, 2, skin, "skin", r=0.4 * hu, cast=0.2, details=det, light_k=0.42)
        late = style in ("short", "crop", "buzz", "side_part", "spiky", "ponytail", "bun", "receding", "bald", "braids",
                         "pixie")
        for e in ears:
            if e[3] and not late:
                self._ear(pt, e)
        if "beard" in self.extras or self.face.get("beard"):
            self._beard(pt, "beard")
        if "mustache" in self.extras:
            self._beard(pt, "mustache")
        self._hair_front(pt)
        for e in ears:
            if e[3] and late:
                self._ear(pt, e)
        if "glasses" in self.extras or "sunglasses" in self.extras:
            self._glasses(pt, "sunglasses" in self.extras)
        self._headwear(pt)

    def _ears(self):
        Wf = self.P["head_w"] / 0.74
        out = []
        for sg in (1, -1):
            vis = self.hn(sg * 1.0, 0, -0.15)
            if vis < -0.55:
                continue
            x, y, z = self.hp(sg * 0.37 * Wf, 0.07, -0.03)
            out.append((x, y, vis, vis > 0.35, sg))
        return out

    def _ear(self, pt, e):
        hu = self._hu
        x, y, vis, front, sg = e
        fs = 0.45 + 0.55 * max(0.0, vis)
        rx, ry = 0.07 * fs, 0.115
        ang = math.degrees(self._hroll)
        f = self._hf(lambda X, Y: _ell(X, Y, x, y, rx, ry))
        tone = pt.tone(self.skin, "skin")

        def det(X, Y, D, A, sl):
            if self._simple:
                return
            g = self._hf(lambda X, Y: _ell(X, Y, x + sg * 0.012 * fs * (1 if vis > 0 else -1), y + 0.01,
                                           rx * 0.5, ry * 0.6))
            pt.flat(g, [self.h2s(x, y)], 0.12 * hu, tone, opacity=0.8)
        pt.part(f, [self.h2s(x, y)], 0.15 * hu, self.skin, "skin", r=0.07 * hu, cast=0.2, details=det)

    # ---- face
    def _face(self, pt, skull_fn):
        hu = self._hu
        s, c, a, Wh = self._hs()
        Wf = Wh / 0.74
        expr = self._expression()
        es = self.P["eye"]
        ey = 0.03 + self.P["eye_y"]
        anime = self.style == "anime"
        clip = skull_fn
        # blush
        if expr in ("smile", "laugh", "wink") or self.build in ("child", "chibi") or anime or self.face.get("blush"):
            for sg in (1, -1):
                v = self.hn(sg * 0.5, 0.2, 0.85)
                if v < 0.15:
                    continue
                bx, by, _ = self.hp(sg * 0.2 * Wf, ey + 0.15, 0.33)
                f = self._hf(lambda X, Y, bx=bx, by=by, v=v: _ell(X, Y, bx, by, 0.075 * v, 0.045))
                pt.flat(f, [self.h2s(bx, by)], 0.1 * hu, "#f08f88", opacity=0.32, aa=max(1.0, 0.03 * hu), clip=clip)
        for sg in (1, -1):
            vis = self.hn(sg * 0.42, 0.0, 0.9)
            if vis < 0.1:
                continue
            self._eye(pt, sg, ey, Wf, es, vis, expr, clip)
            self._brow(pt, sg, ey, Wf, es, vis, expr, clip)
        self._nose(pt, clip)
        self._mouth(pt, expr, clip)

    def _eye(self, pt, sg, ey, Wf, es, vis, expr, clip):
        hu = self._hu
        anime = self.style == "anime"
        cx, cy, _ = self.hp(sg * 0.165 * Wf, ey, 0.36)
        fs = max(0.35, min(1.0, vis / 0.9))
        ew = (0.072 if not anime else 0.088) * es * fs
        eh = (0.048 if not anime else 0.085) * es
        opn = dict(neutral=1.0, smile=0.88, laugh=0.0, surprised=1.25, sad=0.85, angry=0.78, wink=1.0,
                   sleepy=0.42).get(expr, 1.0)
        if expr == "wink" and sg == -1:
            opn = 0.0
        if self.face.get("eyes_closed"):
            opn = 0.0
        ink = _col(INK) if self.style != "anime" else _col("#2b1d22")
        lid_w = max(0.8, (0.02 if not anime else 0.026) * hu)
        center = self.h2s(cx, cy)
        if opn <= 0.01:
            # happy closed eye: an arch
            r = ew * 1.05
            f = self._hf(lambda X, Y: np.where(Y < cy + eh * 0.2,
                                                np.abs(np.hypot((X - cx) / max(fs, 0.4), Y - cy - eh * 0.9) - r * 1.0),
                                                1.0) - 0.0)
            pt.flat(lambda X, Y: f(X, Y) - lid_w * 0.5, [center], 0.2 * hu, ink, clip=clip)
            return
        eh2 = eh * opn
        gx, gy = self._gaze
        iris_col = self.face.get("eyes", _EYE_COLORS[int(self.seed) % len(_EYE_COLORS)])
        px_eh = eh2 * hu
        lx, ly = pt.l
        if px_eh < 2.6 and not anime:
            # small: a dark oval with a dot of light
            r = max(eh2 * 1.05, 0.9 / hu)
            ox, oy = cx + gx * ew * 0.25, cy
            f = self._hf(lambda X, Y: _ell(X, Y, ox, oy, max(r * 0.8 * fs, 0.5 / hu), r))
            pt.flat(f, [center], 0.15 * hu, _mixc(iris_col, INK, 0.6), clip=clip)
            if px_eh > 1.6:
                hx, hy = ox + lx * r * 0.35, oy + ly * r * 0.35
                pt.flat(self._hf(lambda X, Y: _circ(X, Y, hx, hy, r * 0.32)), [center], 0.1 * hu, "#ffffff",
                        opacity=0.9)
            return
        top_cut = None
        if expr in ("angry",):
            top_cut = (-0.35 * sg, 0.45)
        sclera = lambda X, Y: _ell(X, Y, cx, cy, ew, eh2)
        lid_cut = None
        if expr == "sleepy" or opn < 0.6:
            lid_cut = lambda X, Y: cy - eh2 * 0.15 - Y
        if top_cut is not None:
            k_, off = top_cut
            lid_cut = lambda X, Y: (cy - eh2 * off) + k_ * (X - cx) - Y
        def scl(X, Y):
            d = sclera(X, Y)
            if lid_cut is not None:
                d = np.maximum(d, lid_cut(X, Y))
            return d
        fS = self._hf(scl)
        white = "#fbf8f3"
        pt.flat(fS, [center], 0.2 * hu, white, clip=clip)
        ri = eh * (0.95 if not anime else 0.82)
        icx = cx + gx * ew * 0.42
        icy = cy + gy * eh * 0.25 + (0.0 if not anime else eh * 0.05)
        if anime:
            iris = lambda X, Y: _ell(X, Y, icx, icy, ri * 0.72 * fs, ri * 1.0)
        else:
            iris = lambda X, Y: _ell(X, Y, icx, icy, ri * fs, ri)
        icol = _col(iris_col)
        if anime:
            def grad(X, Y):
                lx_, ly_ = self.hloc(X, Y)
                t = np.clip((ly_ - (icy - ri)) / (2 * ri), 0, 1)[..., None]
                return (icol * 0.55 * (1 - t) + _mixc(icol, "#ffffff", 0.35) * t).astype(F32)
            w = pt.win([center], 0.25 * hu)
            if w is not None:
                sl, X, Y = w
                D = np.maximum(self._hf(iris)(X, Y), fS(X, Y))
                A = _sdf.fill(D)
                pt._comp(sl, grad(X, Y), A)
        else:
            pt.flat(self._hf(iris), [center], 0.2 * hu, icol, clip=fS)
        pr = ri * (0.48 if not anime else 0.42)
        pt.flat(self._hf(lambda X, Y: _ell(X, Y, icx, icy + (0 if not anime else ri * 0.1), pr * fs, pr * (1 if not anime else 1.2))),
                [center], 0.2 * hu, _mixc(INK, iris_col, 0.15), clip=fS)
        hr = ri * (0.3 if not anime else 0.32)
        hx, hy = icx + lx * ri * 0.42 * fs, icy + ly * ri * 0.45
        pt.flat(self._hf(lambda X, Y: _circ(X, Y, hx, hy, hr)), [center], 0.2 * hu, "#ffffff", clip=fS)
        if anime or eh * hu > 7:
            hx2, hy2 = icx - lx * ri * 0.4 * fs, icy - ly * ri * 0.4
            pt.flat(self._hf(lambda X, Y: _circ(X, Y, hx2, hy2, hr * 0.45)), [center], 0.2 * hu, "#ffffff",
                    opacity=0.8, clip=fS)
        # upper lid / lash line (heavier towards the outer corner)
        lash = self.P["lashes"] or anime or self.face.get("lashes")
        lw_ = lid_w / hu

        def lid(X, Y):
            d = scl(X, Y)
            ring = np.abs(d) - lw_ * (0.55 + 0.35 * np.clip((X - cx) * sg / max(ew, 1e-3), 0, 1))
            upper = Y - (cy + (eh2 * 0.1 if lid_cut is None else eh2 * 0.6))
            return np.maximum(ring, upper)
        pt.flat(self._hf(lid), [center], 0.25 * hu, ink)
        if lash:
            ox, oy = cx + sg * ew * 0.95, cy - eh2 * 0.35
            wing = [(ox - sg * ew * 0.2, oy + eh2 * 0.15), (ox + sg * ew * 0.28, oy - eh2 * 0.45)]
            pt.flat(self._hf(lambda X, Y: _polyline_sd(X, Y, wing, lw_ * 0.6)), [center], 0.25 * hu, ink)
        # lower lid hint
        if not self._simple and opn > 0.5:
            def low(X, Y):
                d = scl(X, Y)
                ring = np.abs(d) - lw_ * 0.22
                return np.maximum(ring, (cy + eh2 * 0.45) - Y)
            pt.flat(self._hf(low), [center], 0.2 * hu, _mixc(self.skin, INK, 0.45), opacity=0.6)

    def _brow(self, pt, sg, ey, Wf, es, vis, expr, clip):
        hu = self._hu
        raise_in, raise_out = dict(angry=(-0.045, 0.015), sad=(0.05, -0.012), surprised=(0.05, 0.045),
                                   laugh=(0.02, 0.025), smile=(0.012, 0.012), sleepy=(-0.01, -0.01)).get(expr, (0, 0))
        anime = self.style == "anime"
        lift = 0.115 * es ** 0.6 + (0.03 if anime else 0.0)
        a3 = (sg * 0.07 * Wf, ey - lift - raise_in, 0.39)
        b3 = (sg * 0.25 * Wf, ey - lift + 0.01 - raise_out, 0.34)
        m3 = (sg * 0.16 * Wf, ey - lift - 0.018 - (raise_in + raise_out) / 2, 0.38)
        A = self.hp(*a3)
        B = self.hp(*b3)
        M = self.hp(*m3)
        hc = self.hair.get("color")
        bc = _mixc(hc, INK, 0.35) if float(np.mean(_col(hc))) > 0.35 else _mixc(hc, INK, 0.15)
        if self.hair.get("style") == "bald" and self.build == "elder":
            bc = _mixc("#bdb7b0", INK, 0.2)
        th = (0.024 if self.body != "masculine" else 0.032) * (0.8 if anime else 1.0)
        f = self._hf(lambda X, Y: _smin(_cone(X, Y, A[:2], M[:2], th, th * 0.9), _cone(X, Y, M[:2], B[:2], th * 0.9,
                                                                                          th * 0.45), 0.01))
        pt.flat(f, [self.h2s(*M[:2])], 0.2 * hu, bc, clip=clip)

    def _nose(self, pt, clip):
        hu = self._hu
        s, c, a, Wh = self._hs()
        tone = pt.tone(self.skin, "skin")
        x, y, _ = self.hp(0, 0.2 + self.P["eye_y"] * 0.4, 0.45)
        lx, ly = pt.l
        sz = 0.9 if self.build not in ("child", "chibi") else 0.6
        if self.style == "line":
            seg = [(x - 0.012, y - 0.08), (x - 0.025 * (1 - a) + s * 0.03, y), (x + 0.025, y + 0.012)]
            pt.flat(self._hf(lambda X, Y: _polyline_sd(X, Y, seg, 0.008)), [self.h2s(x, y)], 0.15 * hu, INK,
                    opacity=0.85)
            return
        if self.style == "anime":
            seg = [(x - 0.004, y - 0.01), (x + 0.006 - s * 0.01, y + 0.012)]
            pt.flat(self._hf(lambda X, Y: _polyline_sd(X, Y, seg, 0.007)), [self.h2s(x, y)], 0.1 * hu,
                    _mixc(self.skin, INK, 0.45))
            return
        # flat / soft: a shadow shape beside and under the tip, away from the light
        ox = -lx * 0.018 + s * 0.012
        f = self._hf(lambda X, Y: _ell(X, Y, x + ox, y + 0.018, 0.045 * sz, 0.028 * sz))
        pt.flat(f, [self.h2s(x, y)], 0.12 * hu, tone, opacity=0.85, clip=clip, aa=max(1.0, 0.012 * hu))
        if not self._simple:
            g = self._hf(lambda X, Y: _ell(X, Y, x - lx * 0.01, y - 0.004, 0.028 * sz, 0.02 * sz))
            pt.flat(g, [self.h2s(x, y)], 0.12 * hu, _mixc(self.skin, "#ffffff", 0.25), opacity=0.6, clip=clip,
                    aa=max(1.0, 0.012 * hu))
            if a > 0.25:
                bx = x - s * 0.02
                seg = [(bx + s * 0.002, y - 0.13), (bx + s * 0.03, y - 0.02)]
                pt.flat(self._hf(lambda X, Y: _polyline_sd(X, Y, seg, 0.008)), [self.h2s(x, y)], 0.2 * hu, tone,
                        opacity=0.6, clip=clip)

    def _mouth(self, pt, expr, clip):
        hu = self._hu
        s, c, a, Wh = self._hs()
        young = self.build in ("child", "chibi")
        x, y, _ = self.hp(0, 0.315 + self.P["eye_y"] * 0.45 - (0.02 if young else 0), 0.37)
        fs = max(0.35, self.hn(0, 0.15, 1.0))
        w = dict(smile=0.1, laugh=0.105, neutral=0.065, sad=0.07, angry=0.08, surprised=0.03, wink=0.095,
                 sleepy=0.05).get(expr, 0.07) * fs * (0.85 if young else 1.14)
        ink = _mixc(self.skin, "#5a2228", 0.7) if self.style in ("flat", "soft") else _col(INK)
        lw = max(0.8, 0.017 * hu) / hu
        center = self.h2s(x, y)
        if expr in ("laugh",) or (expr == "smile" and self.face.get("open")):
            hgt = 0.075 * (0.9 if young else 1.0)

            def shape(X, Y):
                xx = np.clip((X - x) / w, -1.2, 1.2)
                top = y - 0.02 + 0.028 * xx * xx
                bot = y - 0.01 + hgt * np.clip(1 - xx * xx, 0, 1) ** 0.7
                d1 = np.maximum(top - Y, Y - bot)
                return np.maximum(d1, np.abs(X - x) - w)
            f = self._hf(shape)
            pt.flat(f, [center], 0.2 * hu, "#6b2730", clip=clip)
            pt.flat(self._hf(lambda X, Y: np.maximum(shape(X, Y), Y - (y + 0.018 - 0.0 + 0.028 * ((X - x) / w) ** 2))),
                    [center], 0.2 * hu, "#fbf7f2")
            pt.flat(self._hf(lambda X, Y: np.maximum(shape(X, Y), _ell(X, Y, x, y + hgt * 0.95, w * 0.55, 0.04))),
                    [center], 0.2 * hu, "#e0727a")
            if self.style in ("line", "anime"):
                pt.flat(self._hf(lambda X, Y: np.abs(shape(X, Y)) - lw * 0.5), [center], 0.2 * hu, INK)
            return
        if expr == "surprised":
            f = self._hf(lambda X, Y: _ell(X, Y, x, y + 0.02, 0.032 * fs, 0.042))
            pt.flat(f, [center], 0.15 * hu, "#6b2730", clip=clip)
            return
        curve = dict(smile=0.032, wink=0.03, neutral=0.008, sad=-0.026, angry=-0.02, sleepy=0.0).get(expr, 0.0)

        def line(X, Y):
            xx = np.clip((X - x) / w, -1, 1)
            yc = y + curve * (1 - xx * xx) - curve * 0.5
            d = np.hypot(np.maximum(np.abs(X - x) - w, 0), Y - yc)
            return d - lw * (0.6 + 0.4 * (1 - xx * xx))
        pt.flat(self._hf(line), [center], 0.2 * hu, ink, clip=clip)
        if expr in ("smile", "wink") and not self._simple and self.style in ("flat", "soft"):
            for sg in (-1, 1):
                cx2 = x + sg * w * 1.02
                cy2 = y - curve * 0.5 - 0.004
                seg = [(cx2 - sg * 0.006, cy2 - 0.014), (cx2 + sg * 0.008, cy2 + 0.008)]
                pt.flat(self._hf(lambda X, Y, seg=seg: _polyline_sd(X, Y, seg, lw * 0.4)), [center], 0.2 * hu, ink,
                        opacity=0.7, clip=clip)

    def _beard(self, pt, kind):
        hu = self._hu
        s, c, a, Wh = self._hs()
        col = self.hair.get("color")
        sk = self._skull_local()
        mx, my, _ = self.hp(0, 0.315, 0.37)
        if kind == "mustache":
            f = self._hf(lambda X, Y: _smin(_ell(X, Y, mx - 0.045, my - 0.035, 0.055, 0.025, 12),
                                            _ell(X, Y, mx + 0.045, my - 0.035, 0.055, 0.025, -12), 0.02))
            pt.part(f, [self.h2s(mx, my)], 0.15 * hu, col, "hair", r=0.03 * hu, cast=0.2)
            return
        ex = self.P["eye_y"]
        bcol = _mixc(col, INK, 0.25)

        def f(X, Y):
            d = sk(X, Y) - 0.025
            cut = (0.42 + ex) - 0.95 * np.abs(X - mx) - Y
            d = np.maximum(d, cut)
            d = np.maximum(d, (0.02 + ex) - Y)
            d = np.maximum(d, -s * X - 0.02)
            mouth = _ell(X, Y, mx, my + 0.01, 0.07, 0.028)
            return np.maximum(d, -mouth)
        pt.part(self._hf(f), self._hpts(0.6, 0, 0.25), 2, bcol, "hair", r=0.1 * hu, cast=0.2)
        f2 = lambda X, Y: _smin(_ell(X, Y, mx - 0.045, my - 0.032, 0.055, 0.024, 10),
                                _ell(X, Y, mx + 0.045, my - 0.032, 0.055, 0.024, -10), 0.02)
        pt.part(self._hf(f2), [self.h2s(mx, my)], 0.15 * hu, bcol, "hair", r=0.03 * hu, cast=0.2)

    def _glasses(self, pt, sun=False):
        hu = self._hu
        s, c, a, Wh = self._hs()
        Wf = Wh / 0.74
        ey = 0.03 + self.P["eye_y"]
        es = self.P["eye"]
        frame = self.colors.get("glasses", "#2b2530")
        lw = max(0.8, 0.022 * hu) / hu
        lenses = []
        for sg in (1, -1):
            vis = self.hn(sg * 0.42, 0.0, 0.9)
            if vis < 0.05:
                continue
            fs = max(0.3, min(1.0, vis / 0.9))
            cx, cy, _ = self.hp(sg * 0.17 * Wf, ey + 0.01, 0.42)
            lenses.append((cx, cy, 0.105 * fs * es ** 0.5, 0.075 * es ** 0.4, sg))
        for cx, cy, rw, rh, sg in lenses:
            box = lambda X, Y, cx=cx, cy=cy, rw=rw, rh=rh: _sdf.sd_box(X, Y, cx, cy, rw, rh, min(rw, rh) * 0.6)
            if sun:
                pt.flat(self._hf(box), [self.h2s(cx, cy)], 0.2 * hu, "#1f2229", opacity=0.92)
                pt.flat(self._hf(lambda X, Y, cx=cx, cy=cy, rw=rw: _ell(X, Y, cx - rw * 0.3, cy - rh * 0.3, rw * 0.35,
                                                                     rh * 0.2, -20)),
                        [self.h2s(cx, cy)], 0.2 * hu, "#ffffff", opacity=0.25)
            else:
                pt.flat(self._hf(box), [self.h2s(cx, cy)], 0.2 * hu, "#ffffff", opacity=0.12)
            pt.flat(self._hf(lambda X, Y, b=box: np.abs(b(X, Y)) - lw * 0.5), [self.h2s(cx, cy)], 0.2 * hu, frame)
        if len(lenses) == 2:
            l1, l2 = sorted(lenses, key=lambda t: t[0])
            seg = [(l1[0] + l1[2], l1[1] - 0.01), ((l1[0] + l2[0]) / 2, l1[1] - 0.025), (l2[0] - l2[2], l2[1] - 0.01)]
            pt.flat(self._hf(lambda X, Y: _polyline_sd(X, Y, seg, lw * 0.5)), [self.h2s(*seg[1])], 0.2 * hu, frame)
        if a > 0.8 and lenses:
            l = max(lenses, key=lambda t: t[2]) if len(lenses) > 1 else lenses[0]
            ex, ey2, _ = self.hp(-math.copysign(1, s) * 0.36 * Wf * 0 + (-s) * 0.3, 0.02, -0.05)
            seg = [(l[0] - math.copysign(l[2], s), l[1] - 0.02), (ex, ey2)]
            pt.flat(self._hf(lambda X, Y: _polyline_sd(X, Y, seg, lw * 0.5)), [self.h2s(ex, ey2)], 0.6 * hu, frame)

    # ---- hair
    _HAIR = dict(   # vol, forehead hairline, side line (front), side line (back), window width
        short=(0.055, -0.21, 0.03, 0.3, 1.0), crop=(0.035, -0.25, 0.0, 0.26, 1.0), buzz=(0.014, -0.26, 0.0, 0.24, 1.0),
        side_part=(0.07, -0.21, 0.04, 0.3, 1.0), spiky=(0.05, -0.2, 0.02, 0.28, 1.0),
        bob=(0.08, -0.1, 0.44, 0.44, 0.9), pixie=(0.06, -0.14, 0.12, 0.3, 1.0), long=(0.07, -0.2, 0.5, 0.5, 0.92),
        ponytail=(0.035, -0.24, 0.06, 0.3, 1.0), bun=(0.035, -0.24, 0.06, 0.3, 1.0),
        curly=(0.12, -0.15, 0.14, 0.34, 1.0), afro=(0.1, -0.2, 0.12, 0.28, 1.0), braids=(0.03, -0.24, 0.08, 0.3, 1.0),
        bald=(0.0, -0.6, -1.0, -1.0, 1.0), receding=(0.04, -0.33, 0.1, 0.3, 1.0),
    )

    def _hair_color(self):
        return self.hair.get("color", HAIR_COLORS["dark_brown"])

    def _hair_back_kind(self):
        st = self.hair.get("style")
        if st in ("long", "bob"):
            return st
        if st == "ponytail":
            return "ponytail"
        if st == "braids":
            return "braids"
        return None

    def _cap_local(self, vol):
        s, c, a, Wh = self._hs()
        crx = Wh * 0.5 * (1 + 0.2 * a) + vol
        return lambda x, y: _ell(x, y, -s * 0.07, -0.07 - vol * 0.3, crx, 0.43 + vol)

    def _window_local(self, top, width):
        s, c, a, Wh = self._hs()
        cx = s * (0.12 + 0.13 * a)
        rx = Wh * 0.47 * (1 - 0.12 * a) * width
        hh = (0.75 - top) / 2

        st = self.hair.get("style")
        wav = {"short": 0.022, "crop": 0.012, "spiky": 0.03, "pixie": 0.025, "side_part": 0.018, "curly": 0.03,
               "afro": 0.02, "receding": 0.0, "buzz": 0.0}.get(st, 0.0)
        freq = 30.0 if st != "curly" else 40.0

        def f(x, y):
            d = _sdf.sd_box(x, y, cx, top + hh, rx, hh, min(rx * 0.75, 0.2))
            if wav:
                near_top = np.clip(1.0 - (y - top) / 0.18, 0, 1)
                tilt = 0.05 * (x - cx) if st in ("side_part", "pixie", "short") else 0.0
                d = d - (wav * (0.5 + 0.5 * np.sin((x - cx) * freq + 1.3)) + tilt) * near_top
            return d
        return f

    def _zone_local(self, yf, yb):
        s, c, a, Wh = self._hs()
        if c < -0.15:
            return lambda x, y: y - max(yb, 0.3)

        def f(x, y):
            t = np.clip((-s * x) / 0.25 + 0.2 * a, 0, 1)
            t = t * t * (3 - 2 * t)
            return y - (yf + (yb - yf) * t)
        return f

    def _hair_mass_local(self):
        """Main hair volume (head-local distance), including style-specific volumes."""
        st = self.hair.get("style", "short")
        s, c, a, Wh = self._hs()
        vol = self._HAIR[st][0]
        cap = self._cap_local(vol)
        crx = Wh * 0.5 * (1 + 0.2 * a) + vol
        if st == "bald":
            return None
        if st in ("bob", "long"):
            bot = 0.44 if st == "bob" else 0.52
            box = lambda x, y: _sdf.sd_box(x, y, -s * 0.07, (-0.2 + bot) / 2, crx * (1.0 if st == "bob" else 0.96),
                                           (bot + 0.2) / 2, 0.16)
            return lambda x, y: _smin(cap(x, y), box(x, y), 0.06)
        if st == "curly":
            def f(x, y):
                th = np.arctan2(y + 0.07, x + s * 0.07)
                return cap(x, y) + 0.028 * np.sin(13 * th)
            return f
        if st == "afro":
            def f(x, y):
                ex, ey = -s * 0.06, -0.16
                th = np.arctan2(y - ey, x - ex)
                return _ell(x, y, ex, ey, Wh * 0.5 + 0.3 + 0.05 * a, 0.62) + 0.022 * np.sin(15 * th)
            return f
        if st == "spiky":
            def f(x, y):
                th = np.arctan2(y + 0.07, x + s * 0.07)
                tri = np.abs(((th * 9 / math.pi) % 2.0) - 1.0)
                top = np.clip(-np.sin(th) + 0.1, 0, 1)
                return cap(x, y) - 0.13 * tri * top
            return f
        if st == "side_part":
            sw = lambda x, y: _ell(x, y, -0.12 - s * 0.08, -0.36, 0.3, 0.16, -14)
            return lambda x, y: _smin(cap(x, y), sw(x, y), 0.05)
        if st == "bun":
            bx, by = -s * 0.24, -0.5 + 0.06 * a
            return lambda x, y: _smin(cap(x, y), _circ(x, y, bx, by, 0.17), 0.04)
        if st == "pixie":
            sw = lambda x, y: _ell(x, y, -0.1 + s * 0.05, -0.27, 0.3, 0.12, -18)
            return lambda x, y: _smin(cap(x, y), sw(x, y), 0.04)
        return cap

    def _hair_paint(self, pt, f_local, pts, kind="front", clip=None):
        hu = self._hu
        col = self._hair_color()
        st = self.hair.get("style")
        fn = self._hf(f_local)
        opacity = 1.0
        if st == "buzz":
            opacity = 0.85
        tone = pt.tone(col, "hair")

        def det(X, Y, D, A, sl):
            if self._simple or st in ("buzz",):
                return
            s, c, a, Wh = self._hs()
            # a few strands following the skull
            if st in ("afro", "curly"):
                return
            # a few strands sweeping from the crown: arcs over the top of the head only (full rings used to read
            # as a target around the ear in side views)
            cx0, cy0 = -s * 0.07, -0.07
            for k, (dx, r, rel) in enumerate(((0.0, 0.36, 1.0), (-0.12, 0.3, 0.85), (0.13, 0.27, 0.8))):
                cx = cx0 + dx * (1 - a)
                ry_ = r * 1.25

                def g_(x, y, cx=cx, r=r, ry_=ry_):
                    arc = np.abs(_ell(x, y, cx, cy0 + 0.06, r * Wh * 1.3, ry_)) - 0.006
                    return np.maximum(arc, y - (cy0 - 0.02))
                pt.flat(self._hf(g_), pts, 2, tone, opacity=0.35 * rel, clip=lambda X, Y: fn(X, Y) + 0.04 * hu)
            if self.style == "anime" and kind == "front":
                # the cel "angel ring": a light band across the upper hair with a toothed lower edge
                cx, cy = cx0, cy0
                lx, ly = pt.l
                rx_, ry_ = Wh * 0.5 + 0.02, 0.43

                def band(x, y):
                    th = np.arctan2(y - cy, x - cx)
                    tooth = np.abs(((th * 26 / math.pi) % 2.0) - 1.0)
                    outer = _ell(x, y, cx, cy, rx_ * 0.86, ry_ * 0.8)
                    inner = _ell(x, y, cx, cy, rx_ * 0.7, ry_ * 0.62) + 0.035 * tooth
                    d = np.maximum(outer, -inner)
                    side = (x - cx) * lx + (y - cy) * ly
                    return np.maximum(np.maximum(d, y - (cy - 0.08)), -side - 0.12)
                pt.flat(self._hf(band), pts, 2, _mixc(col, "#ffffff", 0.35), opacity=0.7,
                        clip=lambda X, Y: fn(X, Y) + 0.02 * hu)
        pt.part(fn, pts, 3, col, "hair", r=0.2 * hu, cast=0.3, opacity=opacity, details=det, clip=clip)

    def _hair_back(self, pt, layer="head"):
        """Hair behind the head: layer="back" = long falls behind the body (front views); "head" = mass behind skull."""
        st = self.hair.get("style", "short")
        hu = self._hu
        s, c, a, Wh = self._hs()
        col = self._hair_color()
        if st == "bald":
            return
        back = self._back
        front_ish = c > -0.15
        if layer == "back":
            if st in ("long", "bob") and front_ish:
                mass = self._hair_mass_local()
                self._hair_paint(pt, lambda x, y: np.maximum(mass(x, y), y - 0.75), self._hpts(1.0, 0, 0.2),
                                 kind="back")
            if st == "long":
                ln = 1.55 if self.build not in ("child", "chibi") else 1.0
                pts = [(-Wh * 0.52 - s * 0.1, 0.0), (Wh * 0.52 - s * 0.1, 0.0), (Wh * 0.6 - s * 0.15, ln),
                       (-Wh * 0.6 - s * 0.15, ln)]

                def f(x, y):
                    d = _sdf.sd_box(x, y, -s * 0.14, ln / 2 - 0.05, Wh * 0.64 - 0.1 * a, ln / 2 + 0.05, 0.25)
                    th = x * 9
                    return d + 0.012 * np.sin(th)
                self._hair_paint(pt, f, self._hpts(ln + 0.3, 0, ln / 2), kind="back")
            elif st == "ponytail" and (a < 0.35 and not back):
                sw = 0.22 if self.facing > 0 else -0.22
                q = [(-0.05 + sw * 0.4, -0.2), (sw, 0.25), (sw * 1.2, 0.7), (sw * 1.0, 1.0)]
                rr = [0.1, 0.12, 0.08, 0.02]
                self._hair_paint(pt, lambda x, y: _chain(x, y, q, rr, 0.05), self._hpts(1.2, 0, 0.4), kind="back")
            elif st == "braids" and not back:
                for sg in (1, -1):
                    if self.hn(sg, 0, 0.3) < -0.3:
                        self._braid(pt, sg)
            return
        mass = self._hair_mass_local()
        if mass is None:
            return
        if st in ("long", "bob") and front_ish:
            return
        extra = []
        if st == "ponytail" and (a >= 0.35 or back):
            sgn = -1 if s > 0 else 1
            if back:
                sgn = 0
            q = [(sgn * 0.42, -0.25), (sgn * 0.62, 0.05), (sgn * 0.62, 0.5), (sgn * 0.5, 0.95)]
            rr = [0.1, 0.12, 0.09, 0.025]
            extra.append(lambda x, y: _chain(x, y, q, rr, 0.05))
        if back and st == "long":
            ln = 1.55 if self.build not in ("child", "chibi") else 1.0
            extra.append(lambda x, y: _sdf.sd_box(x, y, 0, ln / 2, Wh * 0.56, ln / 2 + 0.05, 0.22))
        if back and st == "braids":
            for sg in (1, -1):
                extra.append(lambda x, y, sg=sg: _cone(x, y, (sg * 0.25, 0.2), (sg * 0.3, 1.2), 0.07, 0.05))
        vol = self._HAIR[st][0]
        yb = self._HAIR[st][3]
        zone = self._zone_local(yb, yb) if st not in ("bob", "long", "afro", "curly") else (lambda x, y: y - 0.7)

        def f(x, y):
            d = np.maximum(mass(x, y), zone(x, y))
            for e in extra:
                d = _smin(d, e(x, y), 0.04)
            return d
        big = 1.7 if st in ("afro", "ponytail", "long") else 1.0
        if st == "long" and back:
            big = 2.0
        self._hair_paint(pt, f, self._hpts(big, 0, 0.3 if big > 1.5 else 0), kind="back")

    def _braid(self, pt, sg):
        hu = self._hu
        col = self._hair_color()
        x0, y0, _ = self.hp(sg * 0.33, 0.12, -0.05)
        n = 7
        tone = pt.tone(col, "hair")
        a = (x0, y0 + 0.02)
        b = (x0 + sg * 0.12, y0 + 0.08 + n * 0.15)
        rope = self._hf(lambda X, Y: _cone(X, Y, a, b, 0.075, 0.05))

        def det(X, Y, D, A, sl):
            for i in range(1, n + 1):
                t = i / (n + 1)
                cx, cy = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                for sgn in (-1, 1):
                    seg = [(cx - 0.06, cy - 0.035 * sgn - 0.03), (cx + 0.06, cy + 0.035 * sgn - 0.03)]
                    pt.flat(self._hf(lambda X, Y, seg=seg: _polyline_sd(X, Y, seg, 0.007)), [self.h2s(cx, cy)],
                            0.12 * hu, tone, opacity=0.8, clip=rope)
        pt.part(rope, [self.h2s(*a), self.h2s(*b)], 0.1 * hu, col, "hair", r=0.06 * hu, cast=0.3, details=det)
        tie = self._hf(lambda X, Y: _ell(X, Y, b[0], b[1] - 0.04, 0.055, 0.03))
        pt.flat(tie, [self.h2s(*b)], 0.1 * hu, self.colors.get("accent", "#d1404f"))
        tip = self._hf(lambda X, Y: _cone(X, Y, (b[0], b[1] - 0.02), (b[0] + sg * 0.02, b[1] + 0.12), 0.045, 0.02))
        pt.part(tip, [self.h2s(*b)], 0.2 * hu, col, "hair", r=0.03 * hu, cast=0.2)

    def _hair_front(self, pt):
        st = self.hair.get("style", "short")
        hu = self._hu
        s, c, a, Wh = self._hs()
        if st == "bald":
            return
        vol, top, yf, yb, ww = self._HAIR[st]
        mass = self._hair_mass_local()
        win = self._window_local(top, ww)
        zone = self._zone_local(yf, yb)
        back = c < -0.15
        if st == "receding" and not back:
            win2 = self._window_local(-0.62, 1.12)
            win = lambda x, y, w1=win, w2=win2: np.minimum(w1(x, y), w2(x, y))
        if st == "afro":
            zone = lambda x, y: y - 0.12

        def f(x, y):
            d = np.maximum(mass(x, y), zone(x, y))
            if not back:
                d = np.maximum(d, -win(x, y))
            return d
        fr = None
        if not back and st in ("short", "pixie", "side_part", "curly", "spiky"):
            # a soft fringe sweeping across the forehead
            if st == "short":
                fr = lambda x, y: _ell(x, y, -0.06 + s * 0.12, -0.27, 0.24 * Wh / 0.74, 0.08, -12)
            elif st == "pixie":
                fr = lambda x, y: _ell(x, y, -0.04 + s * 0.12, -0.2, 0.28 * Wh / 0.74, 0.09, -20)
            elif st == "side_part":
                fr = lambda x, y: _ell(x, y, -0.08 + s * 0.1, -0.29, 0.26 * Wh / 0.74, 0.08, -18)
            elif st == "curly":
                fr = lambda x, y: _ell(x, y, s * 0.12, -0.25, 0.27, 0.08) + 0.02 * np.sin(x * 40)
            elif st == "spiky":
                fr = lambda x, y: _ell(x, y, s * 0.12, -0.27, 0.22, 0.06) - 0.04 * np.abs(np.sin(x * 22))
        if not back and st == "bob":
            fr = None
        g = f if fr is None else (lambda x, y: _smin(f(x, y), np.maximum(fr(x, y), mass(x, y) - 0.02), 0.02))
        big = 1.6 if st in ("afro", "long") else 1.1
        self._hair_paint(pt, g, self._hpts(big, 0, 0.2 if st in ("long", "bob") else 0), kind="front")
        if st == "side_part" and not back and not self._simple and a < 0.5:
            px_ = -0.15 + s * 0.12
            seg = [(px_, -0.52), (px_ + 0.02, -0.34)]
            pt.flat(self._hf(lambda X, Y: _polyline_sd(X, Y, seg, 0.008)), self._hpts(0.6), 2,
                    pt.tone(self._hair_color(), "hair"), opacity=0.8)
        if st == "long" and not back and a < 0.7:
            # locks falling in front of the shoulders
            for sg in (1, -1):
                if sg * s < -0.3:
                    continue
                x0 = sg * (Wh * 0.5) + s * 0.1
                q = [(x0, 0.18), (x0 + sg * 0.05, 0.7), (x0 + sg * 0.1, 1.2)]
                rr = [0.08, 0.095, 0.03]
                self._hair_paint(pt, lambda x, y, q=q, rr=rr: _chain(x, y, q, rr, 0.05), self._hpts(1.4, 0, 0.6),
                                 kind="front")
        if st == "braids" and not back:
            for sg in (1, -1):
                if self.hn(sg, 0, 0.3) >= -0.3:
                    self._braid(pt, sg)
        if st == "bun" and not back and not self._simple:
            bx, by = -s * 0.24, -0.5 + 0.06 * a
            pt.flat(self._hf(lambda x, y: np.abs(_circ(x, y, bx, by + 0.1, 0.12)) - 0.012), self._hpts(0.8), 2,
                    pt.tone(self._hair_color(), "hair"), opacity=0.6, clip=self._hf(lambda x, y: _circ(x, y, bx, by, 0.17)))

    # ---- headwear
    def _headwear(self, pt):
        hu = self._hu
        s, c, a, Wh = self._hs()
        ex = self.extras
        hat_col = self.colors.get("hat", self.colors.get("accent", PALETTE["mustard"]))
        st = self.hair.get("style", "short")
        hv = self._HAIR[st][0] if st != "afro" else 0.2
        if "headphones" in ex:
            cap = self._cap_local(hv + 0.07)
            band = lambda x, y: np.maximum(np.abs(cap(x, y)) - 0.03, y + 0.02)
            pt.part(self._hf(band), self._hpts(1.0), 2, "#2b2d36", "dark", r=0.03 * hu, cast=0.4)
            Wf = Wh / 0.74
            for sg in (1, -1):
                vis = self.hn(sg, 0, -0.1)
                if vis < -0.5:
                    continue
                x, y, _ = self.hp(sg * (0.4 * Wf + hv * 0.6), 0.04, -0.03)
                fs = 0.5 + 0.5 * max(0.0, vis)
                cup = lambda X, Y, x=x, y=y, fs=fs: _sdf.sd_box(X, Y, x, y, 0.07 * fs + 0.02, 0.13, 0.05)
                pt.part(self._hf(cup), [self.h2s(x, y)], 0.2 * hu, "#2b2d36", "dark", r=0.08 * hu, cast=0.4)
        if "cap" in ex:
            cap = self._cap_local(hv + 0.035)
            crown = lambda x, y: np.maximum(cap(x, y), y + 0.13)
            brim = []
            for i in range(13):
                t = math.pi * i / 12
                brim.append(self.hp(0.34 * math.cos(t) * Wh / 0.74, -0.15 + 0.1 * math.sin(t), 0.12 + 0.42 * math.sin(t))[:2])
            brim += [self.hp(-0.3 * Wh / 0.74, -0.15, 0.0)[:2], self.hp(0.3 * Wh / 0.74, -0.15, 0.0)[:2]]
            hb = _hull(brim)
            brim_vis = c > -0.3
            if brim_vis and self.hn(0, -0.6, 1) < 0.0:
                pt.part(self._hf(lambda x, y: _convex_sd(x, y, hb) - 0.025), self._hpts(1.0), 2,
                        _mixc(hat_col, "#000000", 0.12), "cloth", r=0.06 * hu, cast=0.4)
            pt.part(self._hf(crown), self._hpts(1.0), 2, hat_col, "cloth", r=0.3 * hu, cast=0.3)
            if brim_vis and self.hn(0, -0.6, 1) >= 0.0:
                pt.part(self._hf(lambda x, y: _convex_sd(x, y, hb) - 0.025), self._hpts(1.0), 2,
                        _mixc(hat_col, "#000000", 0.12), "cloth", r=0.06 * hu, cast=0.4)
            bx, by, _ = self.hp(0, -0.55 - hv, 0.0)
            pt.flat(self._hf(lambda x, y: _ell(x, y, bx, by + 0.02, 0.04, 0.025)), self._hpts(0.9), 2,
                    pt.tone(hat_col, "cloth"))
        if "beanie" in ex:
            cap = self._cap_local(hv + 0.06)
            body = lambda x, y: np.maximum(cap(x, y), y + 0.02)
            cuff = lambda x, y: np.maximum(np.maximum(self._cap_local(hv + 0.08)(x, y), y + 0.02), -0.15 - y)
            pt.part(self._hf(body), self._hpts(1.0), 2, hat_col, "cloth", r=0.3 * hu, cast=0.3)
            pt.part(self._hf(cuff), self._hpts(1.0), 2, _mixc(hat_col, "#000000", 0.1), "cloth", r=0.06 * hu,
                    cast=0.2)
            if not self._simple:
                for i in range(-4, 5):
                    xr = i * 0.07
                    pt.flat(self._hf(lambda x, y, xr=xr: np.maximum(np.abs(x - xr - s * 0.05) - 0.006, cuff(x, y))),
                            self._hpts(1.0), 2, pt.tone(hat_col, "cloth"), opacity=0.5)
        if "beret" in ex:
            f = lambda x, y: _ell(x, y, -0.06 - s * 0.05, -0.43 - hv, 0.44 * Wh / 0.74, 0.15, -10)
            pt.part(self._hf(f), self._hpts(1.0), 2, hat_col, "cloth", r=0.12 * hu, cast=0.35)
        if "hat" in ex:
            ring = []
            for i in range(24):
                t = 2 * math.pi * i / 24
                ring.append(self.hp(0.66 * math.cos(t), -0.22 - hv * 0.5, 0.62 * math.sin(t) - 0.02)[:2])
            hb = _hull(ring)
            crown = lambda x, y: np.maximum(_sdf.sd_box(x, y, -s * 0.05, -0.42 - hv * 0.5, 0.36 * Wh / 0.74 + hv,
                                                         0.22, 0.12), y + 0.21 + hv * 0.5)
            pt.part(self._hf(lambda x, y: _convex_sd(x, y, hb) - 0.01), self._hpts(1.0), 2, hat_col, "cloth",
                    r=0.1 * hu, cast=0.4)
            pt.part(self._hf(crown), self._hpts(1.0), 2, hat_col, "cloth", r=0.25 * hu, cast=0.3)
            band = lambda x, y: np.maximum(crown(x, y), -(y + 0.3 + hv * 0.5))
            pt.flat(self._hf(band), self._hpts(1.0), 2, _mixc(hat_col, INK, 0.55))
        if "chef_hat" in ex:
            band = lambda x, y: np.maximum(np.maximum(self._cap_local(hv + 0.04)(x, y), y + 0.22), -0.45 - y)
            puff = lambda x, y: _smin(_smin(_circ(x, y, 0, -0.82, 0.3), _circ(x, y, -0.24, -0.7, 0.22), 0.06),
                                      _circ(x, y, 0.24, -0.7, 0.22), 0.06)
            puff2 = lambda x, y: _smin(puff(x, y), _sdf.sd_box(x, y, 0, -0.52, 0.3, 0.1, 0.02), 0.05)
            pt.part(self._hf(puff2), self._hpts(1.3, 0, -0.5), 2, "#fbfaf7", "cloth", r=0.3 * hu, cast=0.3)
            pt.part(self._hf(band), self._hpts(1.0), 2, "#f2f0ea", "cloth", r=0.1 * hu, cast=0.3)
        if "helmet" in ex:
            R = 0.74
            glass = lambda x, y: _circ(x, y, 0.0, 0.0, R)
            pt.flat(self._hf(glass), self._hpts(0.9), 2, "#9cc4e4", opacity=0.22)
            pt.part(self._hf(lambda x, y: np.abs(glass(x, y)) - 0.045), self._hpts(0.9), 2, "#e9ecf0", "dark",
                    r=0.05 * hu, cast=0.0)
            pt.flat(self._hf(lambda x, y: _ell(x, y, -0.32, -0.36, 0.2, 0.08, -40)), self._hpts(0.9), 2, "#ffffff",
                    opacity=0.6)



def _ik(S, T, a, b, pole):
    """Two-bone IK: shoulder S, target T, bone lengths a, b, bend towards ``pole`` ??(elbow, wrist)."""
    D = T - S
    d = float(np.linalg.norm(D))
    u = D / d if d > 1e-6 else _v(0, -1, 0)
    d = min(max(d, abs(a - b) + 1e-3), (a + b) * 0.955)      # a held arm keeps a slight elbow bend
    cosA = (a * a + d * d - b * b) / (2 * a * d)
    sinA = math.sqrt(max(0.0, 1 - cosA * cosA))
    pp = pole - u * float(np.dot(pole, u))
    if np.linalg.norm(pp) < 1e-6:
        pp = np.cross(u, _v(0, 0, 1))
    pp = _unit(pp)
    E = S + u * (a * cosA) + pp * (a * sinA)
    W = S + u * d
    return E, W


# ===================================================================== ready-made people

PEOPLE = {
    "office_worker": dict(build="average", body="feminine", skin="light", hair=dict(style="bob", color="espresso"),
                          outfit=dict(top="suit", bottom="trousers", shoes="shoes", extras=["bag"],
                                      colors=dict(top="#3a4a66", bottom="#3a4a66", inner="#f5f2ec", shoes="#2a2026",
                                                  accent="#c8553d", bag="#b5532b")), pose="stand"),
    "businessman": dict(build="average", body="masculine", skin="tan", hair=dict(style="side_part", color="black"),
                        outfit=dict(top="suit", bottom="trousers", shoes="shoes",
                                    colors=dict(top="#2e3440", bottom="#2e3440", inner="#f5f2ec", accent="#b5323f",
                                                shoes="#221c1c", bag="#5a3a26")),
                        pose="stand", item="briefcase"),
    "runner": dict(build="slim", body="feminine", skin="brown", hair=dict(style="ponytail", color="black"),
                   outfit=dict(top="tank", bottom="shorts", shoes="sneakers",
                               colors=dict(top="#ec6b56", bottom="#26405f", shoes="#2a9d8f", accent="#f4cf45")),
                   pose="run"),
    "chef": dict(build="broad", body="masculine", skin="medium", hair=dict(style="crop", color="dark_brown"),
                 outfit=dict(top="shirt", bottom="trousers", shoes="shoes", extras=["chef_hat", "apron", "mustache"],
                             colors=dict(top="#f5f2ec", bottom="#383b45", shoes="#2a2422", apron="#f8f6f0")),
                 pose="hands_on_hips", face=dict(expression="smile")),
    "doctor": dict(build="average", body="feminine", skin="deep", hair=dict(style="bun", color="black"),
                   outfit=dict(top="coat", bottom="trousers", shoes="flats", extras=["stethoscope"],
                               colors=dict(top="#f7f7f4", bottom="#3d6f8f", inner="#8ec5e6", shoes="#2a2a33")),
                   pose="arms_crossed", face=dict(expression="smile")),
    "student": dict(build="slim", body="neutral", skin="fair", hair=dict(style="short", color="brown"),
                    outfit=dict(top="hoodie", bottom="jeans", shoes="sneakers", extras=["backpack", "headphones"],
                                colors=dict(top="#8fb38a", bottom="#4a6fa5", shoes="#f5f2ec", bag="#e3a72f",
                                            accent="#c8553d")),
                    pose="walk"),
    "kid": dict(build="child", body="neutral", skin="light", hair=dict(style="curly", color="ginger"),
                outfit=dict(top="tshirt", bottom="shorts", shoes="sneakers",
                            colors=dict(top="#f4cf45", bottom="#2a9d8f", shoes="#ec6b56", accent="#26405f")),
                pose="cheer", face=dict(expression="laugh")),
    "grandma": dict(build="elder", body="feminine", skin="fair", hair=dict(style="bun", color="white"),
                    outfit=dict(top="sweater", bottom="skirt", shoes="flats", extras=["glasses", "scarf"],
                                colors=dict(top="#a89ad6", bottom="#6d3b6b", shoes="#3a2a24", scarf="#c8553d")),
                    pose="hold", item="flower", face=dict(expression="smile")),
    "grandpa": dict(build="elder", body="masculine", skin="olive", hair=dict(style="receding", color="gray"),
                    outfit=dict(top="sweater", bottom="trousers", shoes="shoes", extras=["glasses", "mustache", "hat"],
                                colors=dict(top="#6f7550", bottom="#7a5236", shoes="#2a2020", hat="#8a7660")),
                    pose="stand", face=dict(expression="smile")),
    "musician": dict(build="average", body="masculine", skin="brown", hair=dict(style="afro", color="black"),
                     outfit=dict(top="jacket", bottom="jeans", shoes="boots",
                                 colors=dict(top="#383b45", inner="#d1404f", bottom="#26405f", shoes="#3a2a24")),
                     pose="guitar", face=dict(expression="smile")),
    "astronaut": dict(build="broad", body="neutral", skin="light", hair=dict(style="crop", color="brown"),
                      outfit=dict(top="spacesuit", bottom=None, shoes="boots", extras=["helmet"],
                                  colors=dict(top="#eef0f2", shoes="#c9ccd2", accent="#e2483d")), pose="wave",
                      face=dict(expression="smile")),
    "barista": dict(build="average", body="feminine", skin="olive", hair=dict(style="ponytail", color="auburn"),
                    outfit=dict(top="shirt", bottom="jeans", shoes="sneakers", extras=["apron"],
                                colors=dict(top="#f1e6cf", bottom="#383b45", shoes="#f5f2ec", apron="#2f5d46")),
                    pose="hold", item="cup", face=dict(expression="smile")),
    "artist": dict(build="slim", body="feminine", skin="tan", hair=dict(style="long", color="auburn"),
                   outfit=dict(top="sweater", bottom="trousers", shoes="flats", extras=["beret"],
                               colors=dict(top="#e3a72f", bottom="#26242b", shoes="#c8553d", hat="#c8553d")),
                   pose="thinking"),
    "hiker": dict(build="broad", body="masculine", skin="fair", hair=dict(style="short", color="blonde"),
                  outfit=dict(top="jacket", bottom="trousers", shoes="boots", extras=["backpack", "beanie", "beard"],
                              colors=dict(top="#c8553d", inner="#383b45", bottom="#6f7550", shoes="#5a3a26",
                                          bag="#2f5d46", hat="#e3a72f")), pose="walk"),
    "teen": dict(build="slim", body="feminine", skin="light", hair=dict(style="long", color="blonde"),
                 outfit=dict(top="tshirt", bottom="jeans", shoes="sneakers", extras=["headphones"],
                             colors=dict(top="#f2a7b8", bottom="#4a6fa5", shoes="#f5f2ec")), pose="phone"),
    "hero": dict(build="heroic", body="masculine", skin="medium", hair=dict(style="spiky", color="black"),
                 outfit=dict(top="tshirt", bottom="trousers", shoes="boots",
                             colors=dict(top="#d1404f", bottom="#26405f", shoes="#26242b")), pose="hands_on_hips",
                 face=dict(expression="smile")),
    "dancer": dict(build="slim", body="feminine", skin="deep", hair=dict(style="braids", color="black"),
                   outfit=dict(top="dress", bottom=None, shoes="flats",
                               colors=dict(top="#ef8a2c", shoes="#26242b", accent="#f4cf45")), pose="dance",
                   face=dict(expression="laugh")),
    "tourist": dict(build="average", body="masculine", skin="porcelain", hair=dict(style="short", color="ginger"),
                    outfit=dict(top="shirt", bottom="shorts", shoes="sneakers", extras=["cap", "sunglasses", "bag"],
                                long_sleeves=False,
                                colors=dict(top="#8ec5e6", bottom="#d8c3a0", shoes="#f5f2ec", hat="#e3a72f",
                                            bag="#383b45")), pose="point"),
}


def person(desc="office_worker", **kw):
    """A ready-made character by name (``PEOPLE``) or from a short description:
    ``person("a tall woman in a red coat waving")``. Keywords override any ``Figure`` argument; ``pose=``,
    ``phase=`` and ``item=`` set the pose. Returns a posed Figure."""
    kw = _reject_unknown("person", kw, _sig_options(Figure.__init__) + ["pose", "item", "phase"])
    key = str(desc).strip().lower().replace(" ", "_").replace("-", "_")
    if key in PEOPLE:
        spec = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PEOPLE[key].items()}
        spec["name"] = key
    else:
        spec = _parse(str(desc))
    pose = kw.pop("pose", spec.pop("pose", "stand"))
    item = kw.pop("item", spec.pop("item", None))
    phase = kw.pop("phase", 0.3)
    for k, v in kw.items():
        if k in ("hair", "outfit", "face") and isinstance(v, dict) and isinstance(spec.get(k), dict):
            merged = dict(spec[k])
            if k == "outfit" and "colors" in v and isinstance(merged.get("colors"), dict):
                cc = dict(merged["colors"])
                cc.update(v["colors"])
                v = dict(v, colors=cc)
            merged.update(v)
            spec[k] = merged
        else:
            spec[k] = v
    fig = Figure(**spec)
    fig.pose(pose, phase=phase)
    if item:
        if item not in ITEMS:
            raise _unknown_name("item", item, ITEMS)
        fig._pose["item"] = item
    return fig


_POSE_WORDS = {"wav": "wave", "walk": "walk", "run": "run", "jog": "run", "sit": "sit", "seated": "sit",
               "point": "point", "reach": "reach", "hold": "hold", "jump": "jump", "danc": "dance", "kneel": "kneel",
               "lean": "lean", "crossed": "arms_crossed", "hips": "hands_on_hips", "think": "thinking",
               "cheer": "cheer", "celebrat": "cheer", "talk": "talk", "phone": "phone", "stand": "stand"}
_EXTRA_WORDS = {"glasses": "glasses", "sunglasses": "sunglasses", "hat": "hat", "cap": "cap", "beanie": "beanie",
                "beret": "beret", "scarf": "scarf", "backpack": "backpack", "bag": "bag", "headphones": "headphones",
                "beard": "beard", "mustache": "mustache", "moustache": "mustache", "tie": "tie", "apron": "apron",
                "helmet": "helmet"}


def _parse(desc):
    t = desc.lower()
    spec = dict(build="average", body="neutral", skin="light", hair=dict(style="short", color="dark_brown"),
                outfit=dict(top="tshirt", bottom="trousers", shoes="sneakers", colors={}, extras=[]), face={})
    words = re.findall(r"[a-z\-]+", t)
    if any(w in words for w in ("woman", "girl", "lady", "she", "her", "mother", "mom")):
        spec["body"] = "feminine"
        spec["hair"]["style"] = "long"
    if any(w in words for w in ("man", "boy", "guy", "he", "his", "father", "dad")):
        spec["body"] = "masculine"
    if any(w in words for w in ("child", "kid", "boy", "girl", "toddler")):
        spec["build"] = "child"
    if any(w in words for w in ("old", "elderly", "grandma", "grandpa", "grandmother", "grandfather", "senior")):
        spec["build"] = "elder"
        spec["hair"]["color"] = "gray"
    if "tall" in words or "slim" in words or "thin" in words:
        spec["build"] = "slim" if spec["build"] == "average" else spec["build"]
    if any(w in words for w in ("big", "broad", "stocky", "strong")):
        spec["build"] = "broad"
    if "chibi" in words or "cute" in words:
        spec["build"] = "chibi"
    if "hero" in words or "superhero" in words:
        spec["build"] = "heroic"
    for s in HAIR_STYLES:
        if s.replace("_", " ") in t or s in words:
            spec["hair"]["style"] = s
    for name, hexv in HAIR_COLORS.items():
        if (name.replace("_", " ") + " hair") in t or (name + "-haired") in t:
            spec["hair"]["color"] = hexv
    if "blonde" in words or "blond" in words:
        spec["hair"]["color"] = HAIR_COLORS["blonde"]
    if "redhead" in words or "red hair" in t:
        spec["hair"]["color"] = HAIR_COLORS["copper"]
    for name in SKIN_TONES:
        if name + " skin" in t:
            spec["skin"] = name
    toks = words
    for i, w in enumerate(toks):
        ww = w.replace("-", "")
        top = None
        if ww in ("tshirt", "shirt", "hoodie", "sweater", "jumper", "jacket", "coat", "suit", "dress", "tank",
                  "blouse", "spacesuit"):
            top = {"jumper": "sweater"}.get(ww, ww)
            spec["outfit"]["top"] = top
            if top in ("dress", "spacesuit"):
                spec["outfit"]["bottom"] = None
        bottom = {"jeans": "jeans", "trousers": "trousers", "pants": "trousers", "shorts": "shorts",
                  "skirt": "skirt", "leggings": "leggings"}.get(ww)
        if bottom:
            spec["outfit"]["bottom"] = bottom
        shoe = {"boots": "boots", "sneakers": "sneakers", "trainers": "sneakers", "shoes": "shoes",
                "flats": "flats"}.get(ww)
        if shoe:
            spec["outfit"]["shoes"] = shoe
        if i > 0 and toks[i - 1] in PALETTE:
            c = PALETTE[toks[i - 1]]
            if top:
                spec["outfit"]["colors"]["top"] = c
            elif bottom:
                spec["outfit"]["colors"]["bottom"] = c
            elif shoe:
                spec["outfit"]["colors"]["shoes"] = c
            elif ww in ("hat", "cap", "beanie", "beret"):
                spec["outfit"]["colors"]["hat"] = c
            elif ww == "scarf":
                spec["outfit"]["colors"]["scarf"] = c
        if ww in _EXTRA_WORDS and _EXTRA_WORDS[ww] not in spec["outfit"]["extras"]:
            spec["outfit"]["extras"].append(_EXTRA_WORDS[ww])
    for k, v in _POSE_WORDS.items():
        if any(w.startswith(k) for w in words):
            spec["pose"] = v
            break
    for e in EXPRESSIONS:
        if e in words or (e == "smile" and "smiling" in words) or (e == "laugh" and "laughing" in words):
            spec["face"]["expression"] = e
    if "happy" in words:
        spec["face"]["expression"] = "smile"
    for st in STYLES:
        if st + " style" in t or (st == "anime" and "anime" in words):
            spec["style"] = st
    for v in ("side", "back", "front"):
        if v + " view" in t:
            spec["view"] = v
    if "three quarter" in t or "3/4" in t:
        spec["view"] = "three_quarter"
    for item in ITEMS:
        if item in words and item != "bag":
            spec["item"] = item
            if spec.get("pose") in (None, "stand"):
                spec["pose"] = "hold"
    return spec


def random_person(seed=0, **kw):
    """A varied person (skin tone, build, hair, outfit, accessories) drawn from good palettes; keywords override."""
    kw = _reject_unknown("random_person", kw, _sig_options(Figure.__init__))
    r = _rng(seed * 7919 + 13)
    build = r.choice(["average", "average", "average", "slim", "slim", "broad", "elder", "child"])
    body = r.choice(["feminine", "masculine", "feminine", "masculine", "neutral"])
    skin = r.choice(list(SKIN_TONES))
    elder = build == "elder"
    if body == "feminine":
        hs = r.choice(["long", "bob", "ponytail", "bun", "curly", "pixie", "afro", "braids", "long"])
    else:
        hs = r.choice(["short", "crop", "side_part", "buzz", "curly", "afro", "spiky", "short",
                       "bald" if elder else "crop"])
    if elder and body != "feminine" and r.random() < 0.5:
        hs = "receding"
    dark_skin = skin in ("brown", "deep", "ebony")
    hcols = ["black", "espresso", "dark_brown", "brown"] if dark_skin else \
        ["black", "espresso", "dark_brown", "brown", "auburn", "blonde", "ginger", "copper"]
    hc = "gray" if elder and r.random() < 0.5 else ("white" if elder else r.choice(hcols))
    combos = [("tshirt", "jeans"), ("shirt", "trousers"), ("hoodie", "jeans"), ("sweater", "trousers"),
              ("jacket", "jeans"), ("coat", "trousers"), ("tshirt", "shorts"), ("shirt", "skirt"),
              ("dress", None), ("sweater", "skirt"), ("jacket", "trousers"), ("blouse", "trousers")]
    if body != "feminine":
        combos = [c for c in combos if c[0] not in ("dress", "blouse") and c[1] != "skirt"]
    top, bottom = combos[int(r.integers(len(combos)))]
    if build == "child" and top in ("coat", "suit"):
        top = "tshirt"
    warm = ["mustard", "coral", "terracotta", "rust", "red", "pink", "orange", "yellow", "plum", "maroon"]
    cool = ["teal", "navy", "sage", "denim", "sky", "forest", "lavender", "blue", "green", "olive", "charcoal"]
    neutral = ["cream", "white", "black", "charcoal", "sand", "khaki", "beige", "gray", "brown"]
    tc = PALETTE[r.choice(warm + cool)]
    bc = PALETTE[r.choice(neutral + ["navy", "denim", "olive", "charcoal"])] if bottom != "jeans" else \
        r.choice(["#4a6fa5", "#3a5a8a", "#2f3e5c", "#5b7db0"])
    shoe = r.choice(["sneakers", "sneakers", "boots", "shoes", "flats" if body == "feminine" else "shoes"])
    sc = r.choice(["#f5f2ec", "#26242b", "#7a5236", "#c8553d", "#383b45"])
    extras = []
    for e, p in (("glasses", 0.22), ("bag", 0.15), ("backpack", 0.12), ("scarf", 0.1), ("cap", 0.07),
                 ("beanie", 0.06), ("headphones", 0.05)):
        if r.random() < p:
            extras.append(e)
    if "cap" in extras and "beanie" in extras:
        extras.remove("beanie")
    if body == "masculine" and build not in ("child",) and r.random() < 0.2:
        extras.append(r.choice(["beard", "mustache"]))
    spec = dict(build=build, body=body, skin=skin, hair=dict(style=hs, color=hc),
                outfit=dict(top=top, bottom=bottom, shoes=shoe,
                            colors=dict(top=tc, bottom=bc, shoes=sc, accent=PALETTE[r.choice(warm)],
                                        inner=PALETTE[r.choice(["white", "cream", "charcoal", "navy"])],
                                        scarf=PALETTE[r.choice(warm)], bag=PALETTE[r.choice(warm + neutral)],
                                        hat=PALETTE[r.choice(warm + cool)]),
                            extras=extras),
                face=dict(expression=r.choice(["neutral", "smile", "smile", "neutral", "laugh"]),
                          eyes=_EYE_COLORS[int(r.integers(len(_EYE_COLORS)))]),
                seed=int(seed))
    for k, v in kw.items():
        if k in ("hair", "outfit", "face") and isinstance(v, dict):
            spec[k].update(v)
        else:
            spec[k] = v
    return Figure(**spec)


def crowd(cv, n=12, region=(0.05, 0.62, 0.95, 0.96), seed=0, ground=None, height=0.34, far=0.55, style="flat",
          poses=("stand", "walk", "walk", "talk", "phone", "arms_crossed", "stand", "walk"), views=None, log=None,
          shadow=True, min_gap=0.06, people=None, **kw):
    """Draw ``n`` varied people standing in ``region`` (u0, v0, u1, v1: feet positions), back to front.

    Nearer (lower) people are bigger: ``height`` at the bottom of the region, ``far`` x height at the top; with a
    ``ground`` (``form.Ground``) heights are true perspective (1.55~1.9 m, children ~1.2 m). ``people`` = list of
    Figures to place instead of random ones. Returns the drawn Figures, back to front."""
    kw = _reject_unknown("crowd", kw, _sig_options(Figure.__init__))
    pts, order = _form.scatter_positions(n, region, min_gap=min_gap, seed=seed, aspect=cv.w / cv.h)
    r = _rng(seed + 101)
    views = views or ("front", "three_quarter", "three_quarter", "side", "three_quarter", "back")
    out = []
    u0, v0, u1, v1 = region
    for j, i in enumerate(order):
        u, v = pts[i]
        if people is not None and i < len(people):
            fig = people[i]
        else:
            fig = random_person(seed * 1000 + i, style=style, **kw)
            fig.view = views[int(r.integers(len(views)))]
            fig.facing = 1 if r.random() < 0.5 else -1
            pname = poses[int(r.integers(len(poses)))]
            fig.pose(pname, phase=float(r.random()))
        if ground is not None:
            spot = ground.spot(u, v)
            hm = 1.2 if fig.build == "child" else (0.95 if fig.build == "chibi" else float(r.uniform(1.58, 1.88)))
            fig.height = hm
            at = spot
        else:
            t = (v - v0) / max(v1 - v0, 1e-6)
            sc = far + (1 - far) * t
            f = 0.72 if fig.build == "child" else (1.0 if fig.build != "elder" else 0.95)
            fig.height = height * sc * f * float(r.uniform(0.94, 1.06))
            at = (u, v)
        fig.draw(cv, at, log=log, shadow=shadow)
        out.append(fig)
    return out


# ===================================================================== checked options, scene manifest

for _n in ("pose_lerp", "walk_pose", "run_pose"):
    globals()[_n] = _strict(globals()[_n], _n)
_strict_methods(Figure)
globals()["crowd"] = _form.guard_fractions(globals()["crowd"], "crowd")
del _n

from . import manifest as _manifest  # noqa: E402



def _manifest_person(cap, args, res):
    """One manifest entry per drawn figure (named by its ``name``, else "person"); the figure's own mask, so its
    shadow does not count as part of it. ``crowd`` is not registered: each Figure.draw inside it is an entry."""
    f = args["self"]
    if f.bbox is None or f._mask is None:
        return []
    nm = f.name or "person"
    al = [a for a in list(getattr(f, "alias", ())) + ["person", "people", "figure"] if a != nm]
    return [dict(name=nm, mask=f._mask, alias=al,
                 meta={"pose": f._pose_name})]


_manifest.wrap_method(Figure, "draw", "person", after=_manifest_person)
