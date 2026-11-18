"""The dressed 3D figure: an anime body (round-cone limbs, a head with cranium, jaw, ears and neck), hands with finger
poses, garments bound to the skeleton and hair locks that hang with gravity and trail with the motion. Pure functions
on a posed skeleton (``pose3d.fk`` / ``solve_figure``): joints J and part frames F in, render primitives out.

Every primitive carries ``mat`` (its material: skin, hair, a garment piece, a trim) and ``piece`` (the shape it
belongs to: the line art outlines pieces, not primitives, so the round cones of one lock or one sleeve draw as one
shape) and ``seam`` (when a line is drawn where it meets another piece of the same figure at the same depth:
"mat" = only between different materials, "always" = always (fingers), "depth" = only where it lies over another).

Garments (``GARMENTS``) are offset shells round the bones: a sleeve is the arm's round cones made wider (and wider
again toward the cuff when loose), a skirt an open flared cone hung from the waist whose hem opens to clear the
thighs (a run lifts it, sitting drapes it forward over the lap), a blazer ellipsoids round the chest and waist with a
front opening edged by a trim band.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pose3d import Body, unit
from .render3d import ell_prim, rcone, shell

# ============================================================================================ vocabulary

GARMENTS = {
    "shirt": "a shirt: torso and sleeves (--sleeve long|short|none)",
    "blouse": "a blouse with a white collar (--sleeve long|short)",
    "blazer": "a blazer: open V with lapels and a trim band, long sleeves, ends below the waist (--style open|closed)",
    "jacket": "a jacket like the blazer, looser",
    "hoodie": "a hoodie: loose torso and sleeves, the hood bunched behind the neck",
    "coat": "a long coat: tails that flare below the waist (--length knee|long, --style open|closed)",
    "skirt": "a skirt hung from the waist, its hem clears the legs (--style pleated|flared, --length short|knee|long)",
    "dress": "a dress: bodice and a flared skirt in one colour",
    "pants": "long trousers (--fit tight|loose)",
    "shorts": "shorts to mid-thigh",
    "socks": "socks to below the knee",
    "thigh-highs": "stockings to mid-thigh",
    "shoes": "shoes (loafers)",
    "boots": "boots to mid-shin",
    "gloves": "gloves over the hands",
    "tie": "a necktie down the chest",
    "ribbon": "a ribbon bow at the collar",
    "scarf": "a scarf round the neck, its end trailing behind against the motion",
}
GARMENT_ALIAS = {"t-shirt": "shirt", "tshirt": "shirt", "top": "shirt", "trousers": "pants", "jeans": "pants",
                 "stockings": "thigh-highs", "thighhighs": "thigh-highs", "kneesocks": "socks", "loafers": "shoes",
                 "sneakers": "shoes", "bow": "ribbon", "necktie": "tie", "cardigan": "jacket", "uniform": "blazer",
                 "overcoat": "coat", "trenchcoat": "coat", "hood": "hoodie", "sweater": "shirt", "glove": "gloves",
                 "boot": "boots", "shoe": "shoes", "sock": "socks", "pleated-skirt": "skirt", "flared-skirt": "skirt"}
GARMENT_COLOURS = {"shirt": "#f4f4f6", "blouse": "#f6f6fa", "blazer": "#28305a", "jacket": "#4a5568",
                   "hoodie": "#8a9bb8", "coat": "#6b4a3a", "skirt": "#2c3460", "dress": "#c85a7a", "pants": "#3a4258",
                   "shorts": "#3a4258", "socks": "#2a2a30", "thigh-highs": "#24242a", "shoes": "#3a2a24",
                   "boots": "#4a3428", "gloves": "#f2f2f2", "tie": "#a8303c", "ribbon": "#c8303c", "scarf": "#c84040"}
TRIM_COLOUR = "#d4a83c"
# what each piece covers, inner to outer (a later layer is drawn over an earlier one)
LAYER = {"socks": 0, "thigh-highs": 0, "gloves": 0, "shirt": 1, "blouse": 1, "pants": 1, "shorts": 1, "dress": 1,
         "skirt": 2, "tie": 3, "ribbon": 3, "shoes": 3, "boots": 3, "blazer": 4, "jacket": 4, "hoodie": 4,
         "coat": 5, "scarf": 6}
HAND_SHAPES = ("relaxed", "fist", "open", "point", "grip", "peace", "cup")
HAIR_STYLES = ("long", "medium", "bob", "short", "spiky", "ponytail", "twintails", "braid", "bun")
BANGS = ("blunt", "side", "parted", "none")

# hand shapes a preset implies (her right, her left)
PRESET_HANDS = {"run": ("fist", "fist"), "sprint": ("fist", "fist"), "walk": ("relaxed", "relaxed"),
                "jump": ("open", "open"), "leap": ("open", "open"), "land": ("open", "open"),
                "umbrella": ("grip", "relaxed"), "peace-sign": ("peace", "relaxed"), "point": ("point", "relaxed"),
                "fight-stance": ("fist", "fist"), "punch": ("fist", "fist"), "kick": ("fist", "fist"),
                "swing": ("grip", "grip"), "wave": ("open", "relaxed"), "reach-up": ("open", "relaxed"),
                "reach-viewer": ("open", "relaxed"), "stretch": ("open", "open"), "hold-front": ("cup", "cup"),
                "hold-side": ("grip", "relaxed"), "carry-shoulder": ("grip", "relaxed"),
                "sit-ledge": ("open", "open"), "dance-twirl": ("open", "open"), "lean-railing": ("relaxed", "relaxed")}

# how a preset moves (figure frame: x her right, y up, z forward; metres per second-ish)
MOTION = {"run": (0, 0.1, 1.0), "sprint": (0, 0.1, 1.3), "walk": (0, 0, 0.35), "walk-together": (0, 0, 0.35),
          "jump": (0, 0.8, 0.1), "leap": (0, 0.3, 0.9), "land": (0, -0.6, 0.2), "dance-twirl": (0.7, 0.1, 0),
          "kick": (0, 0, 0.3), "punch": (0, 0, 0.4), "swing": (0.3, 0, 0.4)}


def garment_key(word: str) -> Optional[str]:
    w = str(word or "").strip().lower().replace("_", "-").replace(" ", "-")
    w = GARMENT_ALIAS.get(w, w)
    return w if w in GARMENTS else None


def motion_of(rig: dict, F: Dict[str, np.ndarray]) -> Tuple[np.ndarray, bool]:
    """(world velocity of the figure, floating): from the preset (run: forward), a lift with the feet free = floating."""
    from .pose3d import PRESETS
    pk = rig.get("preset") or "stand"
    v = np.asarray(MOTION.get(pk, (0, 0, 0)), float)
    P = F["pelvis"]
    fwd = P[:, 2] * np.array([1, 0, 1.0])
    fwd = unit(fwd) if np.linalg.norm(fwd) > 1e-6 else np.array([0, 0, -1.0])
    right = unit(np.cross([0, 1.0, 0], fwd))
    w = v[0] * right + v[1] * np.array([0, 1.0, 0]) + v[2] * fwd
    lift = float(rig.get("lift") or 0.0)
    floating = lift > 0.15 and PRESETS.get(pk, {}).get("rest") != "air"
    return w, floating


# ============================================================================================ helpers

def _P(pr: dict, owner: str, piece: str, mat: str, seam: str = "mat", group: str = "", **kw) -> dict:
    pr.update({"owner": owner, "piece": piece, "part": piece, "mat": mat, "seam": seam, "group": group or piece})
    pr.update(kw)
    return pr


def _frame(x, y) -> np.ndarray:
    """Columns x, y, z = x cross y, orthonormal, y exact."""
    y = unit(y)
    x = np.asarray(x, float) - np.dot(x, y) * y
    x = unit(x) if np.linalg.norm(x) > 1e-6 else unit(np.cross(y, [0, 0, 1.0]))
    return np.stack([x, y, np.cross(x, y)], 1)


def _ell_inside(c, A, p) -> float:
    """|q| of a point in an ellipsoid's unit frame (< 1 inside)."""
    return float(np.linalg.norm(np.linalg.solve(A, np.asarray(p, float) - c)))


def _push_out(p, colliders) -> np.ndarray:
    for c, A in colliders:
        q = np.linalg.solve(A, p - c)
        n = float(np.linalg.norm(q))
        if n < 1.0:
            q = q / max(n, 1e-6)
            p = c + A @ q
    return p


# ============================================================================================ the body

class Rig:
    """Sizes of the render body for a posed skeleton (metres)."""

    def __init__(self, body: Body, J: Dict[str, np.ndarray], F: Dict[str, np.ndarray]):
        self.b, self.J, self.F = body, J, F
        h = body.h
        self.h = h
        Ht = body.Ht
        k = body.k
        sl = 0.86                         # anime limbs are slimmer than the mannequin's
        self.r_up, self.r_el, self.r_wr = 0.027 * Ht * k * sl, 0.021 * Ht * k * sl, 0.017 * Ht * k * sl
        self.r_th, self.r_kn, self.r_an = 0.05 * Ht * k * sl, 0.031 * Ht * k * sl, 0.019 * Ht * k * sl
        self.r_neck = 0.021 * Ht * k
        Hd = F["head"]
        self.Hd = Hd
        # cranium (the head ball) and jaw: an anime head, big skull and a small pointed chin
        self.cr_c = J["head"] + Hd @ np.array([0, 0.03 * h, -0.02 * h])
        self.cr_A = Hd @ np.diag([0.40 * h, 0.45 * h, 0.42 * h])
        self.jaw_a = J["head"] + Hd @ np.array([0, -0.06 * h, 0.03 * h])
        self.jaw_ra = 0.345 * h
        self.jaw_b = J["chin"] + Hd @ np.array([0, 0.075 * h, -0.025 * h])
        self.jaw_rb = 0.075 * h
        C = F["chest"]
        self.chest_c = J["spine"] + C @ [0, 0.56 * body.Lc, 0]
        self.chest_A = C @ np.diag([0.86 * body.shw, 0.58 * body.Lc, 0.56 * body.shw])
        S = F["spine"]
        self.waist_c = J["spine"]
        self.waist_A = S @ np.diag([0.6 * body.shw, 0.5 * body.Lw + 0.02 * Ht, 0.47 * body.shw])
        Pv = F["pelvis"]
        self.pel_c = J["pelvis"] + Pv @ [0, -0.01 * Ht, 0]
        self.pel_A = Pv @ np.diag([1.38 * body.hipw, 0.075 * Ht, 0.88 * body.hipw])

    def limb(self, s: str, kind: str):
        """(a, b, ra, rb) of upperarm / forearm / thigh / shin."""
        J = self.J
        if kind == "upperarm":
            return J[f"shoulder-{s}"], J[f"elbow-{s}"], self.r_up, self.r_el
        if kind == "forearm":
            return J[f"elbow-{s}"], J[f"wrist-{s}"], self.r_el, self.r_wr
        if kind == "thigh":
            return J[f"hip-{s}"], J[f"knee-{s}"], self.r_th, self.r_kn
        return J[f"knee-{s}"], J[f"ankle-{s}"], self.r_kn, self.r_an

    def foot(self, s: str):
        J, b = self.J, self.b
        up = self.F[f"foot-{s}"][:, 1]
        heel = J[f"heel-{s}"] + up * 0.55 * b.ah
        toe = J[f"foot-{s}"] + up * 0.42 * b.ah
        toe = heel + (toe - heel) * 0.78           # anime feet are small
        return heel, toe, 0.5 * b.ah, 0.4 * b.ah


def body_prims(rig: Rig, owner: str, hands: Dict[str, str], grips: Dict[str, float]) -> List[dict]:
    J, F, b, h = rig.J, rig.F, rig.b, rig.h
    out = [_P(ell_prim(rig.cr_c, np.eye(3), (1, 1, 1)), owner, "head", "skin"),
           _P(rcone(rig.jaw_a, rig.jaw_b, rig.jaw_ra, rig.jaw_rb), owner, "head", "skin")]
    out[0]["A"] = rig.cr_A
    Hd = rig.Hd
    for s, sg in (("r", 1.0), ("l", -1.0)):
        # ears (her right is -x of the head frame when she faces the viewer; the frame's x is her right)
        out.append(_P(ell_prim(J["head"] + Hd @ [sg * 0.385 * h, -0.08 * h, -0.03 * h], Hd,
                               (0.06 * h, 0.12 * h, 0.08 * h)), owner, f"ear-{s}", "skin"))
    out.append(_P(rcone(J["neck"] - F["neck"][:, 1] * 0.02 * h, J["neck-top"] + F["neck"][:, 1] * 0.12 * h,
                        rig.r_neck * 1.1, rig.r_neck), owner, "neck", "skin"))
    out.append(_P(ell_prim(rig.chest_c, np.eye(3), (1, 1, 1)), owner, "torso", "skin"))
    out[-1]["A"] = rig.chest_A
    out.append(_P(ell_prim(rig.waist_c, np.eye(3), (1, 1, 1)), owner, "torso", "skin"))
    out[-1]["A"] = rig.waist_A
    out.append(_P(ell_prim(rig.pel_c, np.eye(3), (1, 1, 1)), owner, "torso", "skin"))
    out[-1]["A"] = rig.pel_A
    for s in "rl":
        out.append(_P(ell_prim(J[f"shoulder-{s}"], np.eye(3), (rig.r_up * 1.12,) * 3), owner, f"upperarm-{s}", "skin"))
        for kind in ("upperarm", "forearm", "thigh", "shin"):
            a, bb, ra, rb = rig.limb(s, kind)
            out.append(_P(rcone(a, bb, ra, rb), owner, f"{kind}-{s}", "skin"))
        heel, toe, rh, rt = rig.foot(s)
        out.append(_P(rcone(heel, toe, rh, rt), owner, f"foot-{s}", "skin"))
        out += hand_prims(rig, owner, s, hands.get(s, "relaxed"), grips.get(s))
    return out


# ============================================================================================ hands

_CURL = {  # bends (deg) of the three joints, index to pinky
    "relaxed": [(18, 22, 12), (24, 28, 16), (30, 34, 20), (36, 40, 24)],
    "fist": [(92, 100, 70)] * 4,
    "open": [(2, 4, 3)] * 4,
    "point": [(0, 2, 2), (92, 100, 70), (92, 100, 70), (92, 100, 70)],
    "peace": [(0, 2, 2), (0, 2, 2), (92, 100, 70), (92, 100, 70)],
    "cup": [(28, 22, 14)] * 4,
}
_SPREAD = {"open": (10, 3, -4, -12), "peace": (11, -9, 0, 0), "relaxed": (4, 1, -2, -6)}


def hand_frame(rig: Rig, s: str):
    """(wrist, finger direction u, thumb side t, palm normal n) of hand s."""
    J, F = rig.J, rig.F
    Hn = F[f"hand-{s}"]
    sg = 1.0 if s == "r" else -1.0
    u = unit(J[f"hand-{s}"] - J[f"wrist-{s}"])
    t = Hn[:, 2] - np.dot(Hn[:, 2], u) * u
    t = unit(t) if np.linalg.norm(t) > 1e-6 else unit(np.cross(u, [1.0, 0, 0]))
    n = unit(np.cross(u, t))
    if np.dot(n, -sg * Hn[:, 0]) < 0:
        n = -n
    return J[f"wrist-{s}"], u, t, n


def hand_prims(rig: Rig, owner: str, s: str, shape: str, grip_r: Optional[float] = None, mat: str = "skin") -> List[dict]:
    """Palm, four fingers of three round cones each and a two-part thumb, curled by ``shape``; a grip wraps the
    fingers round a handle of radius ``grip_r`` (m)."""
    W, u, t, n = hand_frame(rig, s)
    Lh = rig.b.Lh * 0.92
    lp, wp, th = 0.46 * Lh, 0.40 * Lh, 0.12 * Lh
    fr = 0.052 * Lh
    out = [_P(ell_prim(W + u * 0.52 * lp + n * 0.02 * Lh, np.stack([t, u, n], 1), (0.5 * wp, 0.56 * lp, 0.5 * th)),
              owner, f"hand-{s}", mat)]
    lens = np.array([0.94, 1.0, 0.94, 0.74]) * 0.54 * Lh
    offs = np.array([0.36, 0.12, -0.12, -0.35]) * wp
    shape = shape if shape in HAND_SHAPES else "relaxed"
    if shape == "grip":
        R = float(grip_r or 0.016) + fr
        curls = []
        for L in lens:
            ph = L * np.array([0.45, 0.3, 0.25])
            curls.append(tuple(min(105.0, math.degrees(v / R)) for v in ph))
        spread = (2, 0, -1, -3)
    else:
        curls = _CURL[shape]
        spread = _SPREAD.get(shape, (2, 0, -2, -4))
    for i in range(4):
        base = W + u * 0.98 * lp + t * offs[i] + n * 0.01 * Lh
        sp = math.radians(spread[i])
        d = unit(math.cos(sp) * u + math.sin(sp) * t)
        nn = n.copy()
        p = base
        r = fr * (1.0 if i < 3 else 0.86)
        for j, (frac, bend) in enumerate(zip((0.45, 0.3, 0.25), curls[i])):
            bnd = math.radians(bend)
            d, nn = unit(math.cos(bnd) * d + math.sin(bnd) * nn), unit(-math.sin(bnd) * d + math.cos(bnd) * nn)
            q = p + d * lens[i] * frac
            out.append(_P(rcone(p, q, r * (1 - 0.08 * j), r * (0.92 - 0.1 * j)), owner, f"finger-{s}{i}", mat,
                          seam="always", group=f"hand-{s}"))
            p = q
    # the thumb: along the palm's side when open, across the fingers in a fist
    tb = W + u * 0.2 * lp + t * 0.42 * wp + n * 0.25 * th
    dirs = {"open": (0.55, 0.85, 0.0), "relaxed": (0.8, 0.45, 0.35), "cup": (0.75, 0.3, 0.55),
            "fist": (0.55, 0.05, 0.85), "point": (0.6, 0.05, 0.8), "peace": (0.55, -0.1, 0.85),
            "grip": (0.45, 0.1, 0.9)}[shape]
    d1 = unit(dirs[0] * u + dirs[1] * t + dirs[2] * n)
    m = tb + d1 * 0.27 * Lh
    d2 = unit(d1 + (u * 0.3 - t * 0.5 if shape in ("fist", "point", "peace", "grip") else u * 0.2))
    out.append(_P(rcone(tb, m, fr * 1.25, fr * 1.12), owner, f"thumb-{s}", mat, seam="always", group=f"hand-{s}"))
    out.append(_P(rcone(m, m + d2 * 0.22 * Lh, fr * 1.12, fr * 0.95), owner, f"thumb-{s}", mat, seam="always",
                  group=f"hand-{s}"))
    return out


# ============================================================================================ the face frame

FACE_POINTS = {  # head-frame (x her right... in units of the head height h) -> where the feature sits on the face
    "eye-r": (0.165, -0.06), "eye-l": (-0.165, -0.06), "brow-r": (0.165, 0.09), "brow-l": (-0.165, 0.09),
    "nose": (0.0, -0.21), "mouth": (0.0, -0.33), "chin": (0.0, -0.47), "cheek-r": (0.22, -0.2),
    "cheek-l": (-0.22, -0.2),
}


def face_surface(rig: Rig, xy: np.ndarray, lift: float = 0.004) -> np.ndarray:
    """World points on the front of the face for head-frame offsets xy (units of h; x toward her right): a ray from
    in front of the face straight back onto the cranium and jaw."""
    from .render3d import intersect
    Hd, h = rig.Hd, rig.h
    xy = np.asarray(xy, float).reshape(-1, 2)
    O = rig.J["head"] + (xy[:, 0:1] * h) * Hd[:, 0] + (xy[:, 1:2] * h) * Hd[:, 1] + 1.5 * h * Hd[:, 2]
    D = np.broadcast_to(-Hd[:, 2], O.shape)
    prims = [ell_prim(rig.cr_c, np.eye(3), (1, 1, 1)), rcone(rig.jaw_a, rig.jaw_b, rig.jaw_ra, rig.jaw_rb)]
    prims[0]["A"] = rig.cr_A
    best = np.full(len(O), np.inf)
    for pr in prims:
        t, _, _ = intersect(pr, O, D)
        best = np.minimum(best, t)
    best = np.where(np.isfinite(best), best, 1.1 * h)
    return O + D * (best[:, None] - lift * h / 0.25)


def face_frame(rig: Rig) -> Dict[str, np.ndarray]:
    pts = np.array(list(FACE_POINTS.values()))
    W = face_surface(rig, pts)
    return dict(zip(FACE_POINTS, W))


# ============================================================================================ garments

def _opts(o: dict, key: str, default: str) -> str:
    return str((o or {}).get(key) or default)


def _ell_from(c, A, scale, extra: float) -> Tuple[np.ndarray, np.ndarray]:
    """An ellipsoid grown by ``scale`` and ``extra`` metres on every axis."""
    n = np.linalg.norm(A, axis=0)
    f = (n * scale + extra) / np.maximum(n, 1e-9)
    return np.asarray(c, float), A * f[None, :]


def _sleeve(rig: Rig, owner: str, s: str, mat: str, length: str, off: float, loose: bool, trim: Optional[str],
            pc: str) -> List[dict]:
    out = []
    if length == "none":
        return out
    a, b, ra, rb = rig.limb(s, "upperarm")
    out.append(_P(ell_prim(rig.J[f"shoulder-{s}"], np.eye(3), ((rig.r_up + off) * 1.04,) * 3), owner, pc, mat))
    if length == "short":
        m = a + (b - a) * 0.55
        r_end = ra + (rb - ra) * 0.55 + off * (1.8 if loose else 1.2)
        out.append(_P(rcone(a, m, ra + off, r_end), owner, pc, mat))
        return out
    out.append(_P(rcone(a, b, ra + off, rb + off), owner, pc, mat))
    a2, b2, ra2, rb2 = rig.limb(s, "forearm")
    cuff = rb2 + off * (2.2 if loose else 1.15)
    end = b2 - unit(b2 - a2) * 0.01
    out.append(_P(rcone(a2, end, ra2 + off, cuff), owner, pc, mat))
    if trim:
        u = unit(b2 - a2)
        out.append(_P(rcone(end - u * 0.03, end, cuff * 1.04, cuff * 1.04), owner, pc + "-cuff", trim))
    return out


def _torso_shell(rig: Rig, owner: str, mat: str, off: float, pc: str, wedge=None, trim_mat=None, trim_w=0.0,
                 hem: Optional[float] = None, chest_scale=1.0) -> List[dict]:
    out = []
    for nm, (c, A) in (("chest", (rig.chest_c, rig.chest_A)), ("waist", (rig.waist_c, rig.waist_A))):
        c2, A2 = _ell_from(c, A, chest_scale if nm == "chest" else 1.0, off)
        clip = {}
        if wedge is not None:
            clip["wedge"] = wedge[nm]
            if trim_w:
                clip["trim"] = trim_w
        pr = _P(ell_prim(c2, np.eye(3), (1, 1, 1), clip=clip or None), owner, pc, mat, trim_mat=trim_mat)
        pr["A"] = A2
        out.append(pr)
    if hem is not None:
        c2, A2 = _ell_from(rig.pel_c, rig.pel_A, 1.0, off)
        clip = {"below": hem}
        if wedge is not None:
            clip["wedge"] = wedge["pelvis"]
            if trim_w:
                clip["trim"] = trim_w
        pr = _P(ell_prim(c2, np.eye(3), (1, 1, 1), clip=clip), owner, pc, mat, trim_mat=trim_mat)
        pr["A"] = A2
        out.append(pr)
    return out


def _collar(rig: Rig, owner: str, mat: str, pc: str, gap: float = 0.75, height: float = 0.05,
            r_extra: float = 0.022) -> List[dict]:
    J, F = rig.J, rig.F
    N = F["neck"]
    c = J["neck"] + N[:, 1] * 0.01
    r = rig.r_neck + r_extra
    pr = _P(ell_prim(c, N, (r, height, r), clip={"wedge": [gap, gap * 0.6, 1.0, -1.0], "below": -0.55}), owner, pc,
            mat)
    return [pr]


def _chest_front(rig: Rig, y: float, x: float = 0.0, lift: float = 0.006) -> np.ndarray:
    """A point on the front of the chest ellipsoid at local height y (-1..1) and side x (-1..1)."""
    z = math.sqrt(max(0.0, 1 - y * y - x * x))
    p = rig.chest_c + rig.chest_A @ np.array([x, y, z])
    n = unit(np.linalg.solve(rig.chest_A, p - rig.chest_c) @ np.linalg.inv(rig.chest_A))
    return p + n * lift


def _bow(owner: str, centre, R: np.ndarray, size: float, mat: str, pc: str, tails: bool = True) -> List[dict]:
    """A ribbon bow: two loops, a knot and two tails; R columns = her right, up, out (toward the viewer of the bow)."""
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    out = []
    for sg in (1.0, -1.0):
        Rl = _frame(x * sg * math.cos(0.35) + y * math.sin(0.35), y)
        out.append(_P(ell_prim(centre + x * sg * 0.62 * size + y * 0.1 * size, np.stack([x, y, z], 1),
                               (0.62 * size, 0.38 * size, 0.2 * size)), owner, pc, mat))
        if tails:
            out.append(_P(rcone(centre - y * 0.1 * size, centre - y * 0.95 * size + x * sg * 0.35 * size,
                                0.13 * size, 0.1 * size), owner, pc, mat))
        _ = Rl
    out.append(_P(ell_prim(centre + z * 0.05 * size, np.stack([x, y, z], 1), (0.2 * size, 0.24 * size, 0.2 * size)),
                  owner, pc + "-knot", mat))
    return out


def _skirt(rig: Rig, owner: str, mat: str, o: dict, motion: np.ndarray, pc: str, dress: bool = False) -> List[dict]:
    J, F, b = rig.J, rig.F, rig.b
    style = _opts(o, "style", "flared" if dress else "pleated")
    length = _opts(o, "length", "knee" if dress else "short")
    Pv, S = F["pelvis"], F["spine"]
    waist = J["spine"] + S[:, 1] * 0.2 * b.Lw
    down = -Pv[:, 1]
    th = [unit(J[f"knee-{s}"] - J[f"hip-{s}"]) for s in "rl"]
    L = {"short": b.Lw * 1.2 + 0.42 * b.Lt, "knee": b.Lw * 1.2 + 0.98 * b.Lt,
         "long": b.Lw * 1.2 + b.Lt + 0.62 * b.Ls}.get(length, b.Lw + 0.42 * b.Lt)
    off = 0.008
    rx = 0.6 * b.shw + off + 0.004
    rz = 0.47 * b.shw + off + 0.004
    samples = [(rig.pel_c + rig.pel_A @ np.array([math.cos(a) * math.cos(e), math.sin(e), math.sin(a) * math.cos(e)]),
                0.012) for a in np.linspace(0, 2 * math.pi, 16, endpoint=False) for e in (-0.6, -0.2, 0.2)]
    for s in "rl":
        a_, b_, ra, rb = rig.limb(s, "thigh")
        for f in np.linspace(0.0, 1.0, 9):
            samples.append((a_ + (b_ - a_) * f, ra + (rb - ra) * f + 0.012))
        a_, b_, ra, rb = rig.limb(s, "shin")
        for f in np.linspace(0.0, 0.6, 4):
            samples.append((a_ + (b_ - a_) * f, ra + (rb - ra) * f + 0.012))
    base = {"pleated": 0.9, "flared": 1.25, "pencil": 0.25}.get(style, 0.9)
    cap = 1.9

    def solve(axis):
        R_ = _frame(Pv[:, 0], -axis)
        A_ = np.stack([R_[:, 0] * rx, R_[:, 1] * L, R_[:, 2] * rz], 1)
        Ai_ = np.linalg.inv(A_)
        qs = [(Ai_ @ (pnt - waist), rr) for pnt, rr in samples]
        need_ = 0.0
        for q, rr in qs:
            if -1.0 <= q[1] < -0.02:
                rho = math.hypot(q[0], q[2]) + rr / min(rx, rz)
                need_ = max(need_, (rho - 1.0) / -q[1])
        return need_, A_, Ai_, qs
    # the hang that needs the least flare: straight down, or swung toward the thighs (a lap when sitting, the
    # front lifted by a knee when running)
    mean_th = unit(th[0] + th[1]) if np.linalg.norm(th[0] + th[1]) > 1e-6 else down
    best = None
    for k_ in (0.0, 0.12, 0.25, 0.4, 0.6):
        ax = unit(down + k_ * mean_th - 0.1 * motion)
        r_ = solve(ax)
        score = min(r_[0], cap + 0.5) + 1.2 * k_
        if best is None or score < best[0]:
            best = (score, r_)
    need, A, Ai, qs = best[1]
    flare = float(np.clip(max(base, need), 0.2, cap))
    tx, tz = 0.0, 0.0
    if need > cap:
        # what the flare cannot clear (a knee raised high) lifts the hem on its side
        for q, rr in qs:
            rho = math.hypot(q[0], q[2])
            if -1.0 <= q[1] < -0.02 and rho + rr / min(rx, rz) > 1 - flare * q[1] and rho > 1e-6:
                k_ = (q[1] + 1.0 + 0.12) / rho
                if k_ > math.hypot(tx, tz):
                    tx, tz = k_ * q[0] / rho, k_ * q[2] / rho
        n_ = math.hypot(tx, tz)
        if n_ > 0.8:
            tx, tz = tx * 0.8 / n_, tz * 0.8 / n_
    speed = float(np.linalg.norm(motion))
    if speed > 0.2:
        # the cloth lags the motion: the hem lifts on the side the figure moves away from
        mv = Ai @ motion
        mv = mv / max(np.linalg.norm(mv), 1e-6)
        tx += float(-0.15 * mv[0] * min(speed, 1.3))
        tz += float(-0.15 * mv[2] * min(speed, 1.3))
    tilt = (tx, tz)
    pleats = 18 if style == "pleated" else 0
    pr = _P(shell(waist, A, flare=flare, pleats=pleats, hem_amp=0.035 if pleats else 0.025, hem_tilt=tilt), owner, pc,
            mat)
    pr["skirt_flare"] = flare
    out = [pr]
    # the waistband
    c2, A2 = _ell_from(rig.waist_c, rig.waist_A, 1.0, off + 0.004)
    band = _P(ell_prim(c2, np.eye(3), (1, 1, 1), clip={"below": -0.55, "above": 0.2}), owner, pc, mat)
    band["A"] = A2
    out.append(band)
    return out


def _legwear(rig: Rig, owner: str, mat: str, pc: str, top: float, off: float, foot: bool = True) -> List[dict]:
    """Socks / stockings: the shin, the knee and ``top`` of the thigh (0 = knee, 1 = hip), and the foot."""
    out = []
    for s in "rl":
        a, b, ra, rb = rig.limb(s, "shin")
        out.append(_P(rcone(a, b, ra + off, rb + off), owner, f"{pc}-{s}", mat))
        if top > 0:
            a2, b2, ra2, rb2 = rig.limb(s, "thigh")
            p = b2 + (a2 - b2) * top
            out.append(_P(rcone(b2, p, rb2 + off, rb2 + (ra2 - rb2) * top + off), owner, f"{pc}-{s}", mat))
        elif top < 0:
            out[-1] = _P(rcone(a + (b - a) * (-top), b, ra + (rb - ra) * (-top) + off, rb + off), owner, f"{pc}-{s}",
                         mat)
        if foot:
            heel, toe, rh, rt = rig.foot(s)
            out.append(_P(rcone(heel, toe, rh + off, rt + off), owner, f"{pc}-{s}", mat))
    return out


def garment_prims(rig: Rig, owner: str, piece: str, o: dict, motion: np.ndarray, hands: Dict[str, str],
                  grips: Dict[str, float]) -> List[dict]:
    J, F, b, h = rig.J, rig.F, rig.b, rig.h
    mat = piece
    fit = _opts(o, "fit", "loose" if piece in ("hoodie", "coat") else "tight")
    loose = fit == "loose"
    out: List[dict] = []
    if piece in ("shirt", "blouse", "dress"):
        sleeve = _opts(o, "sleeve", "short" if piece == "dress" else "long")
        off = 0.012 if loose else 0.007
        out += _torso_shell(rig, owner, mat, off, piece)
        for s in "rl":
            out += _sleeve(rig, owner, s, mat, sleeve, off, loose, None, f"{piece}-sleeve-{s}")
        if piece == "blouse":
            out += _collar(rig, owner, "collar", "collar", gap=0.5, height=0.045, r_extra=0.03)
        if piece == "dress":
            out += _skirt(rig, owner, mat, o, motion, "dress-skirt", dress=True)
    elif piece in ("blazer", "jacket", "hoodie", "coat"):
        off = {"blazer": 0.02, "jacket": 0.026, "hoodie": 0.03, "coat": 0.03}[piece]
        style = _opts(o, "style", "open" if piece in ("jacket",) else "closed")
        trim = "trim" if piece in ("blazer", "coat") and o.get("trim", True) else None
        if piece == "hoodie":
            wedge = None
        elif style == "open":
            wedge = {"chest": [0.62, 0.32, 1.0, -1.0], "waist": [0.32, 0.3, 1.0, -1.0], "pelvis": [0.3, 0.28, 1.0, -1.0]}
        else:
            wedge = {"chest": [0.62, 0.0, 1.0, -0.15], "waist": [0.0, 0.0, -2, -3], "pelvis": [0.0, 0.0, -2, -3]}
        out += _torso_shell(rig, owner, mat, off, piece, wedge=wedge, trim_mat=trim,
                            trim_w=0.09 if trim else 0.0, hem=None if piece == "coat" else -0.35, chest_scale=1.03)
        for s in "rl":
            out += _sleeve(rig, owner, s, mat, "long", off * 0.8, loose or piece == "hoodie", trim,
                           f"{piece}-sleeve-{s}")
        if piece == "hoodie":
            N = F["chest"]
            c = J["neck"] - N[:, 2] * 0.09 + N[:, 1] * 0.03
            out.append(_P(ell_prim(c, N, (0.15, 0.09, 0.08)), owner, "hood", mat))
        else:
            out += _collar(rig, owner, mat, f"{piece}-collar", gap=0.85, height=0.06, r_extra=0.03)
            # lapels: strips along the V from the collar down to where it closes
            for sg in (1.0, -1.0):
                top = _chest_front(rig, 0.82, sg * 0.36, 0.012)
                bot = _chest_front(rig, -0.15 if style != "open" else -0.6, sg * 0.05, 0.012)
                out.append(_P(rcone(top, bot, 0.014, 0.008), owner, f"{piece}-lapel", trim or mat))
            if style != "open" and piece in ("blazer", "jacket"):
                for yy in (-0.3, -0.62):
                    p = _chest_front(rig, yy, 0.0, 0.014) if yy > -0.9 else None
                    if p is not None:
                        out.append(_P(ell_prim(p, np.eye(3), (0.011, 0.011, 0.011)), owner, f"{piece}-button",
                                      trim or "button"))
        if piece == "coat":
            o2 = dict(o)
            o2.setdefault("length", "knee")
            o2.setdefault("style", "flared")
            tails = _skirt(rig, owner, mat, {"length": o2["length"], "style": "flared"}, motion, "coat-tails")
            if style == "open":
                tails[0]["clip"] = {"wedge": [0.35, 0.35, 0.1, -1.5], "trim": 0.09}
                tails[0]["trim_mat"] = trim
            out += tails
    elif piece == "skirt":
        out += _skirt(rig, owner, mat, o, motion, "skirt")
    elif piece in ("pants", "shorts"):
        off = 0.016 if loose else 0.008
        c2, A2 = _ell_from(rig.pel_c, rig.pel_A, 1.0, off)
        pr = _P(ell_prim(c2, np.eye(3), (1, 1, 1)), owner, piece, mat)
        pr["A"] = A2
        out.append(pr)
        for s in "rl":
            a, bb, ra, rb = rig.limb(s, "thigh")
            if piece == "shorts":
                m = a + (bb - a) * 0.55
                out.append(_P(rcone(a, m, ra + off, ra + (rb - ra) * 0.55 + off * 1.6), owner, f"{piece}-{s}", mat))
                continue
            out.append(_P(rcone(a, bb, ra + off, rb + off), owner, f"{piece}-{s}", mat))
            a2, b2, ra2, rb2 = rig.limb(s, "shin")
            out.append(_P(rcone(a2, b2 - unit(b2 - a2) * 0.01, ra2 + off, rb2 + off * (2.0 if loose else 1.3)), owner,
                          f"{piece}-{s}", mat))
    elif piece == "socks":
        out += _legwear(rig, owner, mat, piece, -0.12, 0.004)
    elif piece == "thigh-highs":
        out += _legwear(rig, owner, mat, piece, 0.45, 0.004)
    elif piece in ("shoes", "boots"):
        for s in "rl":
            heel, toe, rh, rt = rig.foot(s)
            up = F[f"foot-{s}"][:, 1]
            fwd = unit(toe - heel)
            out.append(_P(rcone(heel - fwd * 0.012 - up * 0.004, toe + fwd * 0.014 - up * 0.004, rh + 0.014,
                                rt + 0.012), owner, f"{piece}-{s}", mat))
            if piece == "boots":
                a, bb, ra, rb = rig.limb(s, "shin")
                top = a + (bb - a) * 0.35
                out.append(_P(rcone(top, bb, ra + (rb - ra) * 0.35 + 0.013, rb + 0.015), owner, f"{piece}-{s}", mat))
    elif piece == "gloves":
        for s in "rl":
            out += hand_prims(rig, owner, s, hands.get(s, "relaxed"), grips.get(s), mat="gloves")
            a, bb, ra, rb = rig.limb(s, "forearm")
            out.append(_P(rcone(bb - unit(bb - a) * 0.05, bb, rb + 0.006, rb + 0.008), owner, f"gloves-{s}", mat))
        for pr in out:
            pr["inflate"] = 0.0025
    elif piece == "tie":
        top = _chest_front(rig, 0.86, 0.0, 0.016)
        knot_c = top
        out.append(_P(ell_prim(knot_c, F["chest"], (0.017, 0.02, 0.012)), owner, "tie-knot", mat))
        bot = _chest_front(rig, -0.25, 0.0, 0.014)
        mid = (top + bot) / 2
        R = _frame(F["chest"][:, 0], top - bot)
        out.append(_P(ell_prim(mid, R, (0.03, float(np.linalg.norm(top - bot)) / 2 + 0.01, 0.007)), owner, "tie", mat))
    elif piece == "ribbon":
        c = _chest_front(rig, 0.8, 0.0, 0.02)
        out += _bow(owner, c, F["chest"], 0.045, mat, "ribbon")
    elif piece == "scarf":
        N = F["neck"]
        c = J["neck"] + N[:, 1] * 0.035
        r = rig.r_neck + 0.03
        ring = []
        for a in np.linspace(0, 2 * math.pi, 11):
            ring.append(c + r * (math.cos(a) * N[:, 0] + math.sin(a) * N[:, 2]))
        for p0, p1 in zip(ring[:-1], ring[1:]):
            out.append(_P(rcone(p0, p1, 0.032, 0.032), owner, "scarf", mat))
        # the end trails behind, against the motion, and falls
        start = c + r * (0.6 * N[:, 0] - 0.8 * N[:, 2])
        back = -unit(motion) if np.linalg.norm(motion) > 0.2 else -N[:, 2] * 0.3
        d = unit(back * 0.7 + np.array([0, -1.0, 0]) * (1.0 - min(1.0, float(np.linalg.norm(motion))) * 0.6))
        p = start
        for i in range(6):
            dd = unit(d + 0.12 * math.sin(1.7 * i) * N[:, 0])
            q = p + dd * 0.09
            out.append(_P(rcone(p, q, 0.03 - 0.002 * i, 0.028 - 0.002 * i), owner, "scarf-end", mat))
            p = q
    return out


# ============================================================================================ hair

def _cap_point(rig: Rig, phi: float, y: float, grow: float = 1.12) -> Tuple[np.ndarray, np.ndarray]:
    """A point on the hair cap (the cranium grown by ``grow``) at angle phi from the face (0 = front, + toward her
    right... the head frame's +x) and local height y (-1..1); and the outward normal there."""
    c = math.sqrt(max(0.0, 1 - y * y))
    q = np.array([math.sin(phi) * c, y, math.cos(phi) * c])
    A = rig.cr_A * grow
    p = rig.cr_c + A @ q
    n = unit(np.linalg.solve(A, p - rig.cr_c) @ np.linalg.inv(A))
    return p, n


def _lock(rig: Rig, owner: str, k: int, root, d0, length: float, r0: float, gravity: np.ndarray, colliders,
          nseg: int = 7, curl=None, stiff: float = 0.6, mat: str = "hair", tip: float = 0.12) -> List[dict]:
    """A tapered lock from ``root``: each segment turns from where it points toward ``gravity`` (gravity plus the
    wind of the motion) and stays outside the head and body."""
    pts = [np.asarray(root, float)]
    d = unit(d0)
    seg = length / nseg
    g = gravity
    for i in range(nseg):
        k_ = (1 - stiff) * (0.45 + 0.55 * i / max(1, nseg - 1))
        d = unit(d * (1 - k_) + unit(g) * k_ * min(1.0, float(np.linalg.norm(g))) + (curl(i) if curl else 0))
        p = pts[-1] + d * seg
        p = _push_out(p, colliders)
        d = unit(p - pts[-1])
        pts.append(pts[-1] + d * seg)
    out = []
    n = len(pts) - 1
    for i in range(n):
        f0, f1 = i / n, (i + 1) / n
        ra = r0 * (1 - (1 - tip) * f0 ** 2.2)
        rb = r0 * (1 - (1 - tip) * f1 ** 2.2)
        out.append(_P(rcone(pts[i], pts[i + 1], ra, rb), owner, f"lock{k}", mat, seam="depth", group="hair"))
    return out


def hair_prims(rig: Rig, owner: str, hair: dict, motion: np.ndarray, floating: bool) -> List[dict]:
    J, F, h = rig.J, rig.F, rig.h
    Hd = rig.Hd
    style = str(hair.get("style") or "long")
    bangs = str(hair.get("bangs") or ("side" if style in ("short", "spiky") else "blunt"))
    out: List[dict] = []
    # the cap over the skull, open over the face below the hairline
    cap = _P(ell_prim(rig.cr_c + Hd @ [0, 0.02 * h, -0.015 * h], np.eye(3), (1, 1, 1),
                      clip={"back": [0.42, 1.3]}), owner, "hair-cap", "hair", seam="depth", group="hair")
    cap["A"] = rig.cr_A * np.array([1.1, 1.08, 1.1])[None, :]
    out.append(cap)
    wind = hair.get("wind")
    w = np.array([wind[0], -wind[1], 0.0]) if wind is not None else -motion * 0.9
    gscale = 0.35 if floating else 1.0
    gvec = np.array([0, -1.0, 0]) * gscale + w * 0.8
    if floating:
        gvec = gvec + np.array([0, 0.25, 0]) + Hd[:, 2] * -0.1
    # colliders: the head (with the cap), neck, chest, shoulders, back
    col = [(rig.cr_c, rig.cr_A * 1.13), (J["neck"], F["neck"] @ np.diag([rig.r_neck * 1.5] * 3))]
    c2, A2 = _ell_from(rig.chest_c, rig.chest_A, 1.0, 0.03)
    col.append((c2, A2))
    c3, A3 = _ell_from(rig.waist_c, rig.waist_A, 1.0, 0.035)
    col.append((c3, A3))
    for s in "rl":
        col.append((J[f"shoulder-{s}"], np.eye(3) * (rig.r_up + 0.03)))
    r0 = 0.1 * h
    k = 0
    lens = {"long": 3.3, "medium": 1.75, "bob": 1.0, "short": 0.55, "spiky": 0.5, "ponytail": 0.5,
            "twintails": 0.62, "braid": 0.55, "bun": 0.45}
    Lb = lens.get(style, 1.5) * h
    # bangs: a row of locks over the forehead
    if bangs != "none":
        n = 7
        for i in range(n):
            ph = -0.95 + 1.9 * i / (n - 1)
            if bangs == "side":
                ph = -0.75 + 1.6 * i / (n - 1)
            root, nrm = _cap_point(rig, ph, 0.72, 1.06)
            sweep = 0.0
            if bangs == "side":
                sweep = -0.45
            elif bangs == "parted":
                sweep = 0.5 if ph > 0 else -0.5
            down = -Hd[:, 1]
            d0 = unit(Hd[:, 2] * 0.9 + down * 0.4 + Hd[:, 0] * sweep)
            length = 0.6 * h * (1.0 if bangs == "blunt" else (0.8 + 0.35 * (i / (n - 1))))
            if style == "spiky":
                length *= 0.85
            out += _lock(rig, owner, k, root, d0, length, r0 * 1.2, gvec * 0.6 + Hd[:, 0] * sweep * 0.5, col,
                         nseg=5, stiff=0.55, tip=0.14)
            k += 1
    # side locks in front of the ears
    if style not in ("spiky",):
        for sg in (1.0, -1.0):
            root, nrm = _cap_point(rig, sg * 1.25, 0.25, 1.08)
            ln = {"short": 0.55, "bob": 0.9}.get(style, 1.25) * h
            out += _lock(rig, owner, k, root, unit(-Hd[:, 1] + 0.3 * nrm), ln, r0 * 0.9, gvec, col, nseg=6, tip=0.1)
            k += 1
    # the back and the crown
    if style in ("long", "medium", "bob", "short", "spiky"):
        rows = [(0.6, 11, 0.82), (0.1, 13, 1.0), (-0.35, 9, 0.97)] if style in ("long", "medium", "bob") else             [(0.7, 7, 1.0), (0.2, 8, 1.0)]
        for y, n, lf in rows:
            for i in range(n):
                ph = (1.25 + (2 * math.pi - 2.5) * i / (n - 1))
                root, nrm = _cap_point(rig, ph, y, 1.06)
                if style == "spiky":
                    d0 = unit(nrm * 1.0 + Hd[:, 1] * 0.5 - Hd[:, 2] * 0.3)
                    out += _lock(rig, owner, k, root, d0, Lb * lf * (0.85 + 0.3 * ((i * 7) % 5) / 4), r0 * 1.15,
                                 gvec * 0.25, col, nseg=3, stiff=0.85, tip=0.05)
                else:
                    d0 = unit(nrm * 0.5 - Hd[:, 1] * 0.8)
                    curl = None
                    if style == "bob":
                        def curl(i_, nn=nrm):
                            return -nn * 0.35 if i_ >= 4 else 0
                    out += _lock(rig, owner, k, root, d0, Lb * lf * (0.94 + 0.12 * ((i * 5) % 3) / 2), r0 * 1.45,
                                 gvec, col, nseg=8 if style == "long" else 6, curl=curl, tip=0.18)
                k += 1
        if style in ("short", "spiky"):
            for i in range(5):
                ph = -0.9 + 1.8 * i / 4
                root, nrm = _cap_point(rig, ph, 0.92, 1.04)
                d0 = unit(nrm + (Hd[:, 2] * 0.6 if style == "short" else Hd[:, 1] * 0.8))
                out += _lock(rig, owner, k, root, d0, 0.45 * h, r0, gvec * (0.4 if style == "short" else 0.15), col,
                             nseg=3, stiff=0.75, tip=0.05)
                k += 1
    else:
        # tied styles: the hair is pulled to the tie, a short layer stays round the nape
        for i in range(7):
            ph = 1.6 + (2 * math.pi - 3.2) * i / 6
            root, nrm = _cap_point(rig, ph, -0.15, 1.05)
            out += _lock(rig, owner, k, root, unit(-Hd[:, 1] + 0.3 * nrm), 0.45 * h, r0, gvec, col, nseg=3, tip=0.2)
            k += 1
        ties = []
        if style == "ponytail":
            ties = [(math.pi, 0.35, 2.6)]
        elif style == "twintails":
            ties = [(2.05, 0.38, 3.0), (-2.05, 0.38, 3.0)]
        elif style == "braid":
            ties = [(math.pi, 0.05, 2.4)]
        for ph, y, L in ties:
            root, nrm = _cap_point(rig, ph, y, 1.1)
            if style == "braid":
                p = root
                d = unit(-Hd[:, 1] * 0.8 + nrm * 0.4)
                for i in range(9):
                    d = unit(d * 0.5 + unit(gvec) * 0.5)
                    q = _push_out(p + d * L * h / 9, col)
                    out.append(_P(ell_prim((p + q) / 2, _frame(Hd[:, 0], q - p),
                                           (0.075 * h * (1 - 0.04 * i), 0.62 * float(np.linalg.norm(q - p)),
                                            0.06 * h)), owner, f"braid{i}", "hair", seam="always", group="hair"))
                    p = q
                continue
            # a bundle of locks from the tie
            side = unit(nrm * np.array([1, 0, 1.0]))
            for j in range(6):
                a = 2 * math.pi * j / 6
                off = (math.cos(a) * Hd[:, 0] + math.sin(a) * Hd[:, 2]) * 0.05 * h
                d0 = unit(nrm * (1.0 if style == "twintails" else 0.7) + side * 0.3 - Hd[:, 1] * 0.2)
                out += _lock(rig, owner, k, root + off, d0, L * h * (0.9 + 0.1 * ((j * 3) % 4) / 3), r0 * 1.35,
                             gvec, col, nseg=8, stiff=0.5, tip=0.1)
                k += 1
            out.append(_P(ell_prim(root, _frame(Hd[:, 0], nrm), (0.07 * h, 0.05 * h, 0.07 * h)), owner,
                          f"tie{len(out)}", "hair-tie"))
            bows = hair.get("bows")
            if bows:
                R = _frame(np.cross(nrm, Hd[:, 1]) if np.linalg.norm(np.cross(nrm, Hd[:, 1])) > 1e-3 else Hd[:, 0],
                           Hd[:, 1])
                R = np.stack([R[:, 0], R[:, 1], nrm], 1)
                out += _bow(owner, root + nrm * 0.03 * h + Hd[:, 1] * 0.06 * h, R, 0.17 * h, "bow", f"bow{k}")
        if style == "bun":
            root, nrm = _cap_point(rig, math.pi * 0.95, 0.6, 1.15)
            out.append(_P(ell_prim(root + nrm * 0.12 * h, np.eye(3), (0.26 * h,) * 3), owner, "bun", "hair",
                          seam="always", group="hair"))
    if hair.get("ahoge", True) and style not in ("spiky",):
        root, nrm = _cap_point(rig, 0.25, 0.97, 1.08)

        def up_curl(i_):
            return Hd[:, 2] * 0.5 if i_ >= 1 else 0
        out += _lock(rig, owner, k, root, unit(Hd[:, 1] + 0.3 * Hd[:, 2]), 0.3 * h, r0 * 0.45, Hd[:, 1] * 0.2, [],
                     nseg=3, curl=up_curl, stiff=0.7, tip=0.05)
    return out


# ============================================================================================ the whole figure

def dressed_prims(body: Body, J, F, rig_spec: dict, look: dict, owner: str,
                  props: Optional[List[dict]] = None) -> List[dict]:
    """Body, hands, garments and hair of one posed figure, ready for the toon render."""
    rig = Rig(body, J, F)
    motion, floating = motion_of(rig_spec, F)
    hands = dict(zip("rl", PRESET_HANDS.get(rig_spec.get("preset") or "stand", ("relaxed", "relaxed"))))
    grips: Dict[str, float] = {}
    for pr in props or []:
        hs = ("r", "l") if pr.get("hand") == "both" else (pr.get("hand"),)
        for s in hs:
            if s in hands:
                hands[s] = "grip"
                grips[s] = {"sword": 0.017, "umbrella": 0.012, "stick": 0.016, "bag": 0.01, "cup": 0.04,
                            "book": 0.02, "ball": 0.1}.get(pr.get("type"), 0.016)
    for s in "rl":
        v = (rig_spec.get("hands") or {}).get(s)
        if v:
            hands[s] = v
    garments = (look or {}).get("garments") or {}
    gl = "gloves" in garments
    out = [pr for pr in body_prims(rig, owner, hands, grips)
           if not (gl and (pr["piece"].startswith(("hand-", "finger-", "thumb-"))))]
    for piece in sorted(garments, key=lambda g: LAYER.get(g, 3)):
        out += garment_prims(rig, owner, piece, garments[piece] or {}, motion, hands, grips)
    hair = (look or {}).get("hair")
    if hair:
        out += hair_prims(rig, owner, hair, motion, floating)
    for pr in out:
        if pr.get("inflate"):
            pass
    return out


def skirt_flare(prims: List[dict]) -> Optional[float]:
    for pr in prims:
        if "skirt_flare" in pr:
            return float(pr["skirt_flare"])
    return None
