"""The parts an anime illustrator draws over a 3D pose model: the skeleton says where each limb goes and what is in
front; the shapes are anime proportions (a slim neck, narrow shoulders, a waist, arms and legs that taper to thin
wrists and ankles, small hands, small pointed feet), clothes as shapes with hems, cuffs, collars, pleats and folds
at bent joints, hair as pointed locks with strand lines.

``figure_parts`` returns shape2d parts (see shape2d) for one posed figure. Pure: no disk, no project.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import figure3d as F3
from .shape2d import chain_tube, ell_samples, perp_basis, ring, unit

# radius profiles along each limb, in head heights (t from the near joint to the far one)
UPPER = [(0.0, 0.122), (0.22, 0.124), (0.6, 0.1), (1.0, 0.084)]
FORE = [(0.0, 0.084), (0.2, 0.092), (0.65, 0.066), (1.0, 0.05)]
THIGH = [(0.0, 0.21), (0.4, 0.19), (0.8, 0.14), (1.0, 0.118)]
SHIN = [(0.0, 0.112), (0.1, 0.108), (0.3, 0.13), (0.6, 0.094), (0.88, 0.06), (1.0, 0.06)]
NECK = 0.108
HAND_SCALE = 0.86

# clothes: ease (head heights) over the body
EASE = {"shirt": 0.012, "blouse": 0.014, "dress": 0.014, "blazer": 0.026, "jacket": 0.034, "hoodie": 0.045,
        "coat": 0.04, "pants": 0.02, "shorts": 0.03, "socks": 0.01, "thigh-highs": 0.01, "gloves": 0.006}
TOPS = ("shirt", "blouse", "dress", "blazer", "jacket", "hoodie", "coat")
OUTER = ("blazer", "jacket", "hoodie", "coat")


def _prof(prof, h: float, k: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.array([a for a, _ in prof])
    r = np.array([b for _, b in prof]) * h * k
    return t, r


def _limb_axis(A, B, ts) -> np.ndarray:
    A, B = np.asarray(A, float), np.asarray(B, float)
    return np.array([A + t * (B - A) for t in ts])


def _part(name, kind, mat, subj, group, joints=None, rad=0.05, **kw) -> dict:
    d = {"name": name, "kind": kind, "mat": mat, "subj": subj, "group": group, "joints": joints or {},
         "rad": float(rad), "details": [], "bias": 0.0, "line": True}
    d.update(kw)
    return d


def _tube(name, mat, subj, group, axis, radii, joints, **kw) -> dict:
    frames = kw.pop("frames", None)
    rings = chain_tube(axis, radii, n_rings=kw.pop("n_rings", 14), n_pts=kw.pop("n_pts", 18),
                       round_start=kw.pop("round_start", True), round_end=kw.pop("round_end", True),
                       point_end=kw.pop("point_end", False), frames=frames,
                       depth_ratio=kw.pop("depth_ratio", None))
    part = _part(name, "tube", mat, subj, group, joints, rad=float(np.max(radii)), rings=rings, **kw)
    if frames is None:
        # round cross-sections: the exact solid is a chain of round cones (its depth and normals per pixel)
        from .render3d import rcone
        A = [np.asarray(q, float) for q in axis]
        r = [float(v) for v in radii]
        if part.get("open_end") and len(A) >= 2:
            A[-1] = A[-1] - unit(A[-1] - A[-2]) * min(r[-1] * 0.9, 0.45 * float(np.linalg.norm(A[-1] - A[-2])))
        if part.get("open_start") and len(A) >= 2:
            A[0] = A[0] + unit(A[1] - A[0]) * min(r[0] * 0.9, 0.45 * float(np.linalg.norm(A[1] - A[0])))
        part["prims"] = [rcone(A[i], A[i + 1], r[i], r[i + 1]) for i in range(len(A) - 1)
                         if np.linalg.norm(A[i + 1] - A[i]) > 1e-6]
    return part


def _solid(name, mat, subj, group, prims, joints=None, **kw) -> dict:
    """A part that is a few 3D solids (a head, an ear, a palm, a hair cap, a bow): drawn as their outline."""
    P, N = [], []
    for pr in prims:
        a, b = _prim_samples(pr, 80)
        P.append(a)
        N.append(b)
    return _part(name, "blob", mat, subj, group, joints, rad=kw.pop("rad", 0.05), prims=list(prims),
                 pts=np.vstack(P), nrm=np.vstack(N), **kw)


def _blob(name, mat, subj, group, P, N, joints=None, **kw) -> dict:
    return _part(name, "blob", mat, subj, group, joints, rad=kw.pop("rad", 0.05), pts=np.asarray(P),
                 nrm=np.asarray(N), **kw)


def _prim_samples(pr: dict, n: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Surface samples of a 3D primitive (ellipsoid, round cone, cylinder)."""
    from .render3d import _clipped
    if pr["t"] == "ell":
        A = np.asarray(pr["A"], float)

        def keep(q):
            return ~_clipped(pr, q)[0]
        return ell_samples(pr["c"], A, n, keep if pr.get("clip") else None)
    if pr["t"] in ("rcone", "cyl"):
        if pr["t"] == "rcone":
            a, b, ra, rb = pr["a"], pr["b"], pr["ra"], pr["rb"]
        else:
            A = np.asarray(pr["A"], float)
            a, b = pr["c"] - A[:, 1], pr["c"] + A[:, 1]
            ra = rb = float(np.linalg.norm(A[:, 0]))
        rings = chain_tube([a, b], [ra, rb], n_rings=6, n_pts=16)
        P = np.vstack([q for _, q in rings])
        N = np.vstack([unit(q - c) for c, q in rings])
        return P, N
    return np.zeros((0, 3)), np.zeros((0, 3))


# ============================================================================================ the body

MALE_WORDS = ("boy", "man", "male", "guy", "father", "dad", "king", "prince", "knight", "gentleman", "brother",
              "grandpa", "uncle", "husband", "son", "lad", "mr")
# half widths of the anime body in head heights: (child ~5 heads, teen ~7 heads) x (girl, boy)
TEMPLATE = {
    "child": {"girl": {"shoulder": 0.6, "chest": 0.48, "waist": 0.42, "hip": 0.5, "neck": 0.1},
              "boy": {"shoulder": 0.64, "chest": 0.5, "waist": 0.44, "hip": 0.48, "neck": 0.105}},
    "teen": {"girl": {"shoulder": 0.66, "chest": 0.55, "waist": 0.36, "hip": 0.6, "neck": 0.1},
             "boy": {"shoulder": 0.82, "chest": 0.64, "waist": 0.45, "hip": 0.52, "neck": 0.118}},
}


def is_male(subj: str, el: Optional[dict] = None) -> bool:
    if el and el.get("sex"):
        return str(el["sex"]).lower().startswith(("m", "b"))
    words = str(subj or "").lower().replace("_", "-").split("-")
    return any(w in MALE_WORDS for w in words)


def template(heads: float, male: bool) -> Dict[str, float]:
    """Half widths (head heights) of the anime trunk: a child (5-6 heads) is stockier, a teen (6.5-7) slimmer
    with a waist."""
    t = float(np.clip((heads - 5.0) / 2.0, 0.0, 1.0))
    a = TEMPLATE["child"]["boy" if male else "girl"]
    b = TEMPLATE["teen"]["boy" if male else "girl"]
    return {k: a[k] + t * (b[k] - a[k]) for k in a}


def fit_joints(J: Dict[str, np.ndarray], F, tpl: Dict[str, float], h: float, k: float) -> Dict[str, np.ndarray]:
    """The skeleton fitted to the template: each arm hangs from the template's shoulder (the whole arm moves
    across the chest). The legs, the trunk's length and every limb's direction stay as posed, so a limb that points
    at the viewer stays short and overlapping and the stride reads as in the 3D guide."""
    J = {n: np.asarray(v, float).copy() for n, v in J.items()}
    C = F["chest"]
    sh = 0.5 * (J["shoulder-r"] + J["shoulder-l"])
    r_up = UPPER[0][1] * h * k
    for s in "rl":
        cur = float((J[f"shoulder-{s}"] - sh) @ C[:, 0])
        sg = 1.0 if cur >= 0 else -1.0
        want = sg * max(0.05 * h, tpl["shoulder"] * h - r_up)
        d = C[:, 0] * (want - cur)
        for n in (f"shoulder-{s}", f"elbow-{s}", f"wrist-{s}", f"hand-{s}"):
            J[n] = J[n] + d
    return J


class Torso:
    """Elliptic cross-sections down the trunk (neck base, shoulders, chest, waist, hips, crotch) from the anime
    template."""

    def __init__(self, body, J, F, h: float, k: float, tpl: Optional[Dict[str, float]] = None):
        C, S, P = F["chest"], F["spine"], F["pelvis"]
        tpl = tpl or template(body.heads, False)
        sh = 0.5 * (J["shoulder-r"] + J["shoulder-l"])
        lv = [  # (centre, X, Z, a, b)
            (J["neck"] + C[:, 1] * 0.02 * h, C[:, 0], C[:, 2], tpl["neck"] * h, tpl["neck"] * h),
            (sh - C[:, 1] * 0.07 * h, C[:, 0], C[:, 2], (tpl["shoulder"] - 0.07) * h, 0.22 * h),
            (sh + (J["spine"] - sh) * 0.42, C[:, 0], C[:, 2], tpl["chest"] * h, 0.27 * h),
            (J["spine"], S[:, 0], S[:, 2], tpl["waist"] * h, 0.22 * h),
            (J["pelvis"] + P[:, 1] * 0.02 * h, P[:, 0], P[:, 2], tpl["hip"] * h, 0.26 * h),
            (J["pelvis"] - P[:, 1] * 0.17 * h, P[:, 0], P[:, 2], 0.78 * tpl["hip"] * h, 0.2 * h),
        ]
        self.lv = lv
        self.h, self.k = h, k

    def at(self, t: float):
        """(centre, X, Z, a, b) at t in 0 (neck) .. 5 (crotch), interpolated."""
        t = float(np.clip(t, 0, len(self.lv) - 1))
        i = min(int(t), len(self.lv) - 2)
        f = t - i
        A, B = self.lv[i], self.lv[i + 1]
        c = A[0] + f * (B[0] - A[0])
        X = unit(A[1] + f * (B[1] - A[1]))
        Z = unit(A[2] + f * (B[2] - A[2]))
        return c, X, Z, A[3] + f * (B[3] - A[3]), A[4] + f * (B[4] - A[4])

    def tube(self, t0: float, t1: float, ease: float, n: int = 9, scale: float = 1.0, extra=None):
        ts = np.linspace(t0, t1, n)
        L = [self.at(t) for t in ts]
        if extra:
            L += extra
        axis = [l[0] for l in L]
        a = [l[3] * scale + ease for l in L]
        b = [l[4] * scale + ease for l in L]
        frames = [(l[1], l[2]) for l in L]
        return axis, np.array(a), np.array(b) / np.array(a), frames

    def front(self, t: float, x: float, lift: float = 0.0) -> np.ndarray:
        """A point on the front of the trunk: t down the trunk, x across (-1 her left .. 1 her right)."""
        c, X, Z, a, b = self.at(t)
        x = float(np.clip(x, -0.98, 0.98))
        return c + X * a * x + Z * (b * math.sqrt(1 - x * x) + lift)


def _fold_lines(A, M, B, r: float, n: int, side_len: float) -> List[np.ndarray]:
    """Compression folds on the inside of the bend at M (between A-M and M-B): short curved lines across the
    tube near the joint."""
    u1, u2 = unit(A - M), unit(B - M)
    bend = math.degrees(math.acos(float(np.clip(-u1 @ u2, -1, 1))))
    if bend < 35:
        return []
    inner = unit(u1 + u2)
    out = []
    for i in range(n):
        along = (u2 if i % 2 == 0 else u1) * side_len * (0.35 + 0.45 * (i // 2))
        c = M + along + inner * r * 0.9
        x = unit(np.cross(inner, u2 if i % 2 == 0 else u1))
        pts = [c + x * r * s * 0.75 - inner * r * 0.35 * (s * s) for s in np.linspace(-1, 1, 7)]
        out.append(np.array(pts))
    return out


def _band(c, T, r: float, eye=None, n: int = 11, lift: float = 0.0) -> dict:
    """A line around a limb or trunk at ``c`` (a cuff, a seam, a waistband): the half ring that faces the viewer
    (``eye``), else the whole ring with its normals so only the front half is drawn."""
    c = np.asarray(c, float)
    T = unit(T)
    if eye is not None:
        tw = np.asarray(eye, float) - c
        z = unit(tw - T * float(tw @ T))
        x = unit(np.cross(T, z))
        th = np.linspace(0.08, math.pi - 0.08, n)
        pts = c + np.outer(np.cos(th) * r, x) + np.outer(np.sin(th) * r, z) + z * lift
        return {"pts": pts}
    x, z = perp_basis(T)
    th = np.linspace(0, 2 * math.pi, 2 * n)
    off = np.outer(np.cos(th), x) + np.outer(np.sin(th), z)
    return {"pts": c + off * r, "nrm": off}


def _front_line(tor: "Torso", ts, xs, lift: float, w: float = 0.45) -> dict:
    """A line on the front of the trunk (a lapel, a pocket, a waistband) that is drawn only where the front faces the
    viewer."""
    pts = np.array([tor.front(t, x, lift) for t, x in zip(ts, xs)])
    nrm = np.array([tor.at(t)[2] for t in ts])
    return {"pts": pts, "w": w, "nrm": nrm}


def _face_profile(rig, h: float) -> List[dict]:
    """The small solids that give the anime head its profile: a short pointed nose and a soft chin, set on the face
    surface (they show only where they pass the silhouette: in profile and three-quarter view)."""
    from .render3d import rcone
    Hd = rig.Hd
    nose_root, nose_tip = F3.face_surface(rig, np.array([[0.0, -0.1], [0.0, -0.2]]))
    # the nose: a narrow wedge from the bridge to a tip that juts forward and a little down
    tip = nose_tip + Hd[:, 2] * 0.055 * h - Hd[:, 1] * 0.01 * h
    out = []
    for a_, b_, ra, rb in ((nose_root - Hd[:, 2] * 0.02 * h, tip, 0.012 * h, 0.012 * h),
                           (nose_tip - Hd[:, 2] * 0.03 * h - Hd[:, 1] * 0.015 * h, tip, 0.016 * h, 0.011 * h)):
        out.append(rcone(a_, b_, ra, rb))
    for pr in out:
        pr.update({"owner": "x", "piece": "head", "part": "head", "mat": "skin", "seam": "mat", "group": "head"})
    return out


def figure_parts(el: dict, body, J, F, look: dict, subj: str, held: Optional[List[dict]] = None,
                 eye: Optional[np.ndarray] = None, cam=None) -> List[dict]:
    """Every part of one posed figure (body, hands, feet, clothes, hair, held props). ``eye`` (the camera's
    position) turns the flat parts (hands, an umbrella) toward the viewer."""
    h = body.h
    k = body.k
    tpl = template(body.heads, is_male(subj, el))
    J = fit_joints(J, F, tpl, h, k)
    rig = F3.Rig(body, J, F)
    garments = (look or {}).get("garments") or {}
    tor = Torso(body, J, F, h, k, tpl)
    _limb_tops(J, F, tpl, h, k)
    parts: List[dict] = []
    rig_spec = el.get("rig") or {}
    motion, floating = F3.motion_of(rig_spec, F)
    hands = dict(zip("rl", F3.PRESET_HANDS.get(rig_spec.get("preset") or "stand", ("relaxed", "relaxed"))))
    grips: Dict[str, float] = {}
    for pr in held or []:
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

    # ---- head, ears, neck
    hp = F3.body_prims(rig, "x", hands, grips)
    head_pr = []
    for pr in hp:
        if pr["piece"] == "head":
            if pr["t"] == "rcone":
                # a small pointed chin; a boy's jaw a little longer
                pr = dict(pr, rb=0.045 * h, b=pr["b"] + rig.Hd[:, 1] * (0.02 if not is_male(subj, el) else -0.012) * h)
            head_pr.append(pr)
    if eye is not None:
        # the nose's outline only near profile: in three-quarter view it would leave an arc past the far cheek
        to_eye = np.asarray(eye, float) - J["head"]
        if abs(float(rig.Hd[:, 0] @ to_eye)) > 0.8 * float(np.linalg.norm(to_eye)):
            head_pr += _face_profile(rig, h)
    # no joint shared with the neck: the chin's line is always drawn
    parts.append(_solid("head", "skin", subj, "head", head_pr, {"head": J["head"]}, rad=0.42 * h, bias=0.08 * h))
    for pr in hp:
        if pr["piece"].startswith("ear-"):
            parts.append(_solid(pr["piece"], "skin", subj, "head", [pr], {"ear": None}, rad=0.08 * h, bias=0.09 * h))
    parts.append(_tube("neck", "skin", subj, "head", [J["neck"] - F["neck"][:, 1] * 0.02 * h,
                                                      J["neck-top"] + F["neck"][:, 1] * 0.1 * h],
                       [NECK * h * k, NECK * h * k * 0.95], {"neck": J["neck"], "neck-top": J["neck-top"]},
                       round_start=False, round_end=False, n_rings=6, bias=0.1 * h))

    # ---- trunk
    ax, a, dr, fr = tor.tube(0, 5, 0.0)
    parts.append(_tube("torso", "skin", subj, "torso", ax, a, {"neck": J["neck"], "shoulder-r": J["shoulder-r"],
                                                               "shoulder-l": J["shoulder-l"], "hip-r": J["hip-r"],
                                                               "hip-l": J["hip-l"]},
                       frames=fr, depth_ratio=dr, round_start=False, n_rings=18, n_pts=24))

    # ---- limbs
    for s in "rl":
        for kind, prof, A_, B_ in (("upperarm", UPPER, f"shoulder-{s}", f"elbow-{s}"),
                                   ("forearm", FORE, f"elbow-{s}", f"wrist-{s}"),
                                   ("thigh", THIGH, f"hip-{s}", f"knee-{s}"),
                                   ("shin", SHIN, f"knee-{s}", f"ankle-{s}")):
            t, r = _prof(prof, h, k)
            A0 = J[f"top-{kind}-{s}"] if kind in ("upperarm", "thigh") else J[A_]
            parts.append(_tube(f"{kind}-{s}", "skin", subj, ("arm-" if kind in ("upperarm", "forearm") else "leg-") + s,
                               _limb_axis(A0, J[B_], t), r, {A_: J[A_], B_: J[B_]}))
            if kind == "shin":
                kh = _knee_hint(J, s, float(r[0]), eye)
                if kh:
                    parts[-1]["details"].append(kh)
        # hands: an anime hand silhouette for the hand shape, turned to the viewer, foreshortened with the hand
        W = J[f"wrist-{s}"]
        hmat = "gloves" if "gloves" in garments else "skin"
        if eye is not None:
            parts += _hand2d(rig, s, hands.get(s, "relaxed"), hmat, subj, eye, h, k, cam, grips.get(s))
            hprims = []
        else:
            hprims = F3.hand_prims(rig, "x", s, hands.get(s, "relaxed"), grips.get(s), mat=hmat)
        fingers: Dict[str, list] = {}
        for pr in hprims:
            if pr["piece"] == f"hand-{s}":
                pr = dict(pr, c=W + (np.asarray(pr["c"]) - W) * HAND_SCALE, A=np.asarray(pr["A"]) * HAND_SCALE)
                parts.append(_solid(f"hand-{s}", hmat, subj, f"hand-{s}", [pr], {f"wrist-{s}": W, f"palm-{s}": None},
                                    rad=0.08 * h))
            elif pr["t"] == "rcone":
                fingers.setdefault(pr["piece"], []).append(pr)
        for fn, chain in fingers.items():
            pts = [W + (c["a"] - W) * HAND_SCALE for c in chain] + [W + (chain[-1]["b"] - W) * HAND_SCALE]
            rr = [c["ra"] * HAND_SCALE * 0.88 for c in chain] + [chain[-1]["rb"] * HAND_SCALE * 0.8]
            parts.append(_tube(fn, hmat, subj, f"hand-{s}", pts, rr, {f"palm-{s}": None, fn: pts[0]},
                               n_rings=8, n_pts=12, bias=-0.002 * h))
        # feet: a small wedge from the ankle to a pointed toe
        heel, toe, _, _ = rig.foot(s)
        ank = J[f"ankle-{s}"]
        ball = heel + (toe - heel) * 0.55
        shoe = garments.get("shoes") is not None or garments.get("boots") is not None
        sock = any(g in garments for g in ("socks", "thigh-highs"))
        fmat = "shoes" if "shoes" in garments else "boots" if "boots" in garments else \
            ("socks" if "socks" in garments else "thigh-highs" if "thigh-highs" in garments else "skin")
        fr_ = 1.15 if shoe else 1.0
        tip = toe - (toe - heel) * (0.2 if shoe else 0.22)
        if shoe:
            # a shoe: the instep from the ankle down to the ball and a sole from behind the heel to a round toe, one
            # closed outline (the union of the solids), a sole line along its bottom
            from .render3d import rcone
            fw = unit(toe - heel)
            upf = rig.F[f"foot-{s}"][:, 1]
            r_sole = 0.05 * h * k
            back = heel - fw * 0.02 * h
            tipc = toe - fw * 0.04 * h
            prims = [rcone(ank, ball, 0.07 * h * k, 0.064 * h * k),
                     rcone(back + upf * 0.01 * h, ball, r_sole, 0.064 * h * k),
                     rcone(ball, tipc, 0.064 * h * k, 0.048 * h * k)]
            parts.append(_solid(f"foot-{s}", fmat, subj, f"foot-{s}", prims, {f"ankle-{s}": ank},
                                rad=0.07 * h, bias=-0.012 * h))
            low = -upf * 0.62 * r_sole
            sole = np.array([back + low, heel + fw * 0.3 * float(np.linalg.norm(toe - heel)) + low * 1.05,
                             ball + low * 1.2, tipc + fw * 0.5 * 0.048 * h * k - upf * 0.35 * 0.048 * h * k])
            parts[-1]["details"].append({"pts": sole, "w": 0.5})
        else:
            parts.append(_tube(f"foot-{s}", fmat, subj, f"foot-{s}", [ank, ball, tip],
                               [0.068 * h * k, 0.062 * h * k, 0.03 * h * k],
                               {f"ankle-{s}": ank}, point_end=True, round_end=False, n_rings=10, bias=0.0))
        _ = sock

    # ---- clothes
    parts += _clothes(el, body, J, F, h, k, tor, garments, motion, hands, grips, rig, subj, eye=eye)

    # ---- hair
    hair = (look or {}).get("hair")
    if hair:
        parts += _hair(rig, hair, motion, floating, subj, h)

    # ---- held props
    for pe in held or []:
        from .pose3d import prop_prims
        for pr in prop_prims(pe["type"], pe["hand"], pe.get("length"), body, J, F, pe["id"]):
            nm = f"{pe['name']}-{pr['part'].split()[-1]}"
            if pr["part"].endswith("canopy") and eye is not None:
                parts.append(_canopy2d(pr, nm, f"prop:{pe['name']}", subj))
            elif pr["t"] in ("ell", "rcone", "cyl"):
                parts.append(_solid(nm, f"prop:{pe['name']}", subj, "props", [pr], rad=0.05))
            elif pr["t"] == "box":
                parts.append(_box2d(pr, nm, f"prop:{pe['name']}", subj, cam))
    # the joints the folds and seams need
    for pt in parts:
        pt.setdefault("subj", subj)
    return parts


# ============================================================================================ clothes

def _knee_hint(J, s: str, r: float, eye) -> Optional[dict]:
    """A straight leg's kneecap: a short arc just below the knee on the side toward the viewer."""
    if eye is None:
        return None
    K, A, H_ = J[f"knee-{s}"], J[f"ankle-{s}"], J[f"hip-{s}"]
    d1, d2 = unit(K - H_), unit(A - K)
    if math.degrees(math.acos(float(np.clip(d1 @ d2, -1, 1)))) > 30:
        return None
    b = _band(K + d2 * r * 0.55, d2, r * 0.45, eye, n=7)
    return {"pts": b["pts"][1:6] + unit(np.asarray(eye, float) - K) * r * 0.55, "w": 0.38}


def _waist_folds(F, tor: "Torso", lift: float) -> List[dict]:
    """Where the trunk bends sideways (the shoulder line against the hip line): two short folds at the waist on the
    compressed side."""
    C, P = F["chest"], F["pelvis"]
    side = float(C[:, 1] @ P[:, 0])                  # the chest leans toward +X: that side compresses
    if abs(side) < math.sin(math.radians(8)):
        return []
    sg = 1.0 if side > 0 else -1.0
    out = []
    for t0 in (2.75, 3.15):
        out.append(_front_line(tor, [t0 - 0.08, t0, t0 + 0.06], [sg * 0.97, sg * 0.82, sg * 0.66], lift, 0.4))
    return out


def _limb_tops(J, F, tpl: Dict[str, float], h: float, k: float) -> None:
    """Where the drawn upper arm and thigh start (``top-upperarm-<s>``, ``top-thigh-<s>`` in J): the arm's round
    top sits a little below the shoulder line, so the trunk's shoulder runs into a deltoid instead of a ball above
    it; the thigh starts inside the hip, so the thigh's outer edge continues the hip line instead of bulging out
    of it. The knees and elbows stay as posed."""
    C, P = F["chest"], F["pelvis"]
    r_th = THIGH[0][1] * h * k
    for s in "rl":
        J[f"top-upperarm-{s}"] = J[f"shoulder-{s}"] - C[:, 1] * 0.14 * h
        H = J[f"hip-{s}"]
        cur = float((H - J["pelvis"]) @ P[:, 0])
        want = math.copysign(max(0.12 * h, min(abs(cur), tpl["hip"] * h - 0.92 * r_th)), cur if cur else 1.0)
        J[f"top-thigh-{s}"] = H + P[:, 0] * (want - cur) - P[:, 1] * 0.03 * h


def _limb_cover(J, s, kind, t0, t1, ease, h, k, scale=1.08):
    prof = {"upperarm": UPPER, "forearm": FORE, "thigh": THIGH, "shin": SHIN}[kind]
    A_, B_ = {"upperarm": ("shoulder", "elbow"), "forearm": ("elbow", "wrist"), "thigh": ("hip", "knee"),
              "shin": ("knee", "ankle")}[kind]
    tt, rr = _prof(prof, h, k)
    ts = np.linspace(t0, t1, 6)
    r = np.interp(ts, tt, rr) * scale + ease
    A0 = J.get(f"top-{kind}-{s}", J[f"{A_}-{s}"])
    return _limb_axis(A0, J[f"{B_}-{s}"], ts), r, {f"{A_}-{s}": J[f"{A_}-{s}"], f"{B_}-{s}": J[f"{B_}-{s}"]}


def _clothes(el, body, J, F, h, k, tor: Torso, garments: dict, motion, hands, grips, rig, subj,
             eye=None) -> List[dict]:
    out: List[dict] = []
    order = sorted(garments, key=lambda g: F3.LAYER.get(g, 3))
    inner_top = next((g for g in ("blouse", "shirt", "dress") if g in garments), None)
    for g in order:
        o = garments[g] or {}
        ease = EASE.get(g, 0.02) * h
        lay = F3.LAYER.get(g, 3)
        bias = -0.007 * h * (lay + 1)
        loose = str(o.get("fit") or ("loose" if g in ("hoodie", "coat") else "tight")) == "loose"
        if loose:
            ease *= 1.6
        if g in TOPS:
            outer = g in OUTER
            # a blazer or jacket ends at the hip bone, a hoodie at the hips, a coat below the crotch
            t1 = {"blazer": 4.55, "jacket": 4.45, "hoodie": 4.3, "coat": 5.3}.get(g, 3.0 if g == "dress" else 4.6)
            extra = []
            if g == "coat":
                c, X, Z, a, b = tor.at(5.0)
                c2 = c - F["pelvis"][:, 1] * 0.08 * h
                extra = [(c2, X, Z, tor.at(4.0)[3] * 1.0, b * 1.05)]
            ax, a, dr, fr = tor.tube(0, min(t1, 5.0), ease, n=9 if outer else 8, extra=extra)
            body_part = _tube(g, g, subj, "clothes", ax, a, {"neck": J["neck"], "shoulder-r": J["shoulder-r"],
                                                               "shoulder-l": J["shoulder-l"], "hip-r": J["hip-r"],
                                                               "hip-l": J["hip-l"], "waist": J["spine"]},
                              frames=fr, depth_ratio=dr, round_start=False, round_end=False, open_end=True,
                              open_start=True, n_rings=16, n_pts=24, bias=bias)
            out.append(body_part)
            body_part["details"] += _waist_folds(F, tor, ease + 0.004 * h)
            # sleeves
            sleeve = str(o.get("sleeve") or ("short" if g == "dress" else "long"))
            trim = g in ("blazer", "coat") and o.get("trim", True) is not False
            for s in "rl":
                if sleeve == "none":
                    continue
                sc = 1.05 if outer else 1.03
                if sleeve == "short":
                    axs, rs, js = _limb_cover(J, s, "upperarm", 0.0, 0.5, ease, h, k, sc)
                    out.append(_tube(f"{g}-sleeve-{s}", g, subj, f"arm-{s}", axs, rs, js, round_end=False,
                                     open_end=True, bias=bias - 0.002 * h))
                    continue
                axs, rs, js = _limb_cover(J, s, "upperarm", 0.0, 1.0, ease, h, k, sc)
                up = _tube(f"{g}-sleeve-{s}", g, subj, f"arm-{s}", axs, rs, js, bias=bias - 0.002 * h)
                axf, rf, jf = _limb_cover(J, s, "forearm", 0.0, 0.97, ease * 1.15, h, k, sc * 1.05)
                lo = _tube(f"{g}-cuff-{s}" if False else f"{g}-forearm-{s}", g, subj, f"arm-{s}", axf, rf, jf,
                           round_end=False, open_end=True, bias=bias - 0.003 * h)
                # compression folds inside the bent elbow
                for fl in _fold_lines(J[f"shoulder-{s}"], J[f"elbow-{s}"], J[f"wrist-{s}"], float(rs[-1]), 3,
                                      0.18 * h):
                    lo["details"].append({"pts": fl, "w": 0.45})
                # the shoulder seam (the sleeve sets into the trunk with a clear line) and a cuff band at the wrist
                ua = unit(axs[-1] - axs[0])
                seam = _band(axs[0] + ua * 0.16 * float(np.linalg.norm(axs[-1] - axs[0])), ua, float(rs[1]) * 1.02,
                             eye)
                up["details"].append(dict(seam, w=0.5))
                fa = unit(axf[-1] - axf[0])
                cuff = _band(axf[0] + (axf[-1] - axf[0]) * 0.8, fa, float(rf[-2]) * 1.03, eye)
                lo["details"].append(dict(cuff, w=0.55))
                out += [up, lo]
                if trim:
                    axc, rc, jc = _limb_cover(J, s, "forearm", 0.88, 0.96, ease * 1.15 + 0.003 * h, h, k, sc * 1.05)
                    out.append(_tube(f"{g}-trim-{s}", "trim", subj, f"arm-{s}", axc, rc, jc, round_start=False,
                                     round_end=False, open_end=True, open_start=True, bias=bias - 0.006 * h,
                                     n_rings=5))
            # the front: an open V shows the shirt; lapels, buttons; a blouse collar
            if outer and g != "hoodie":
                # a jacket over nothing named closes over a white shirt (never a bare chest)
                style = str(o.get("style") or ("open" if g == "jacket" and inner_top else "closed"))
                apex_t = 1.9 if style == "closed" else 4.4
                lift = ease + 0.006 * h
                vpts = [tor.front(0.15, 0.62, lift), tor.front(0.15, -0.62, lift)]
                vpts += [tor.front(apex_t * f + 0.15 * (1 - f), -0.62 * (1 - f), lift) for f in np.linspace(0.2, 1, 5)]
                vpts += [tor.front(apex_t * f + 0.15 * (1 - f), 0.62 * (1 - f), lift) for f in np.linspace(1, 0.2, 5)[1:]]
                inner_mat = inner_top or "collar"
                P_, N_ = _fan_samples(np.array(vpts), tor.at(1.5)[2])
                out.append(_blob(f"{g}-v", inner_mat, subj, "clothes", P_, N_, {"neck": J["neck"]}, rad=0.1 * h,
                                 hull=False, poly3=np.array(vpts), decal=[g]))
                for sg in (1.0, -1.0):
                    # the lapel: from the collar's back down the V to the top button, its notch line across
                    body_part["details"].append(_front_line(tor, [0.25, 0.9, apex_t * 0.98],
                                                            [sg * 0.95, sg * 0.82, sg * 0.04], lift, 0.5))
                    body_part["details"].append(_front_line(tor, [0.55, 0.8], [sg * 0.86, sg * 0.6], lift, 0.42))
                    # hip pockets: a flap line on each side
                    if t1 > 4.2:
                        body_part["details"].append(_front_line(tor, [3.85, 3.9], [sg * 0.28, sg * 0.72], lift,
                                                                0.42))
                # a breast pocket on her left
                body_part["details"].append(_front_line(tor, [1.45, 1.42], [-0.32, -0.66], lift, 0.42))
                if style == "closed":
                    # the front edge from the top button to the hem, a little off the middle
                    ts_ = np.linspace(apex_t, min(t1, 5.0) - 0.05, 6)
                    body_part["details"].append(_front_line(tor, ts_, [0.04 + 0.03 * (tt - apex_t) / 3 for tt in ts_],
                                                            lift, 0.5))
                if not inner_top:
                    # the shirt's collar points inside the V
                    for sg in (1.0, -1.0):
                        cpts = np.array([tor.front(0.1, sg * 0.6, lift * 0.9), tor.front(0.75, sg * 0.42, lift * 0.9),
                                         tor.front(0.5, sg * 0.05, lift * 0.9), tor.front(0.12, sg * 0.04, lift * 0.9)])
                        P_, N_ = _fan_samples(cpts, tor.at(0.5)[2])
                        out.append(_blob(f"{g}-shirt-collar-{'r' if sg > 0 else 'l'}", "collar", subj, "clothes", P_,
                                         N_, {"collar": None}, rad=0.08 * h, poly3=cpts,
                                         decal=[g, f"{g}-v"]))
                if style == "closed":
                    for tt in (apex_t + 0.35, apex_t + 1.1):
                        if tt < 5.0:
                            c = tor.front(tt, 0.0, lift)
                            from .render3d import ell_prim
                            out.append(_solid(f"{g}-button", "trim" if trim else "button", subj, "clothes",
                                              [ell_prim(c, np.eye(3), (0.028 * h,) * 3)], rad=0.03 * h,
                                              bias=bias - 0.02 * h))
            if g == "blouse" or (g == "shirt" and not any(x in garments for x in OUTER)):
                for sg in (1.0, -1.0):
                    cpts = np.array([tor.front(0.0, sg * 0.9, ease + 0.01 * h), tor.front(0.75, sg * 0.75, ease + 0.012 * h),
                                     tor.front(0.55, sg * 0.08, ease + 0.012 * h), tor.front(0.05, sg * 0.05, ease + 0.01 * h)])
                    P_, N_ = _fan_samples(cpts, tor.at(0.5)[2])
                    out.append(_blob(f"{g}-collar-{'r' if sg > 0 else 'l'}", g, subj, "clothes", P_, N_,
                                     {"collar": None}, rad=0.08 * h, poly3=cpts,
                                     decal=[g] + [f"{x}-v" for x in OUTER] + list(OUTER)))
            if g == "dress":
                out += _shell_parts(F3.garment_prims(rig, "x", "skirt", {"length": o.get("length", "knee"),
                                                                       "style": "flared"}, motion, hands, grips),
                                    g, subj, tor, h, bias + 0.002 * h)
            if g == "hoodie":
                for pr in F3.garment_prims(rig, "x", g, o, motion, hands, grips):
                    if pr["piece"] == "hood":
                        out.append(_solid("hood", g, subj, "clothes", [pr], rad=0.1, bias=0.02 * h))
            if g == "coat":
                out += _shell_parts(F3.garment_prims(rig, "x", "coat", o, motion, hands, grips), g, subj, tor, h,
                                    bias + 0.002 * h, only="coat-tails")
        elif g == "skirt":
            out += _shell_parts(F3.garment_prims(rig, "x", "skirt", o, motion, hands, grips), g, subj, tor, h, bias)
        elif g in ("pants", "shorts"):
            ax, a, dr, fr = tor.tube(3.0, 5.0, ease * 0.6, n=5)
            out.append(_tube(g, g, subj, "clothes", ax, a, {"hip-r": J["hip-r"], "hip-l": J["hip-l"],
                                                             "waist": J["spine"]},
                             frames=fr, depth_ratio=dr, round_start=False, round_end=True, n_rings=8, n_pts=20,
                             bias=bias))
            # the waistband and the fly (drawn only where nothing covers them)
            lift_p = ease * 0.6 + 0.004 * h
            xs_ = np.linspace(-0.92, 0.92, 9)
            out[-1]["details"].append(_front_line(tor, [3.28] * 9, xs_, lift_p, 0.45))
            if g == "pants":
                out[-1]["details"].append(_front_line(tor, [3.3, 3.8, 4.3, 4.55], [0.12, 0.13, 0.1, 0.02], lift_p,
                                                      0.42))
            for s in "rl":
                t1 = 0.45 if g == "shorts" else 1.0
                axs, rs, js = _limb_cover(J, s, "thigh", 0.0, t1, ease * (0.7 if g == "shorts" else 1.0), h, k, 1.06)
                th = _tube(f"{g}-{s}", g, subj, f"leg-{s}", axs, rs, js, round_end=False, open_end=g == "shorts",
                           bias=bias - 0.002 * h)
                out.append(th)
                if g == "pants":
                    axl, rl, jl = _limb_cover(J, s, "shin", 0.0, 0.98, ease * 1.2, h, k, 1.12)
                    # the hem widens a little and breaks over the shoe
                    rl = rl.copy()
                    rl[-1] *= 1.18
                    rl[-2] *= 1.06
                    lo = _tube(f"{g}-shin-{s}", g, subj, f"leg-{s}", axl, rl, jl, round_end=False, open_end=True,
                               bias=bias - 0.003 * h)
                    for fl in _fold_lines(J[f"hip-{s}"], J[f"knee-{s}"], J[f"ankle-{s}"], float(rl[0]), 3, 0.22 * h):
                        lo["details"].append({"pts": fl, "w": 0.45})
                    sa = unit(axl[-1] - axl[0])
                    brk = _band(axl[0] + (axl[-1] - axl[0]) * 0.9, sa, float(rl[-2]) * 1.02, eye, n=9)
                    if eye is not None:
                        q = brk["pts"][2:7].copy()
                        q[1:4] -= sa * float(rl[-2]) * np.array([0.25, 0.45, 0.25])[:, None]
                        lo["details"].append({"pts": q, "w": 0.42})
                    kh = _knee_hint(J, s, float(rl[0]), eye)
                    if kh:
                        lo["details"].append(kh)
                    out.append(lo)
        elif g in ("socks", "thigh-highs"):
            for s in "rl":
                if g == "thigh-highs":
                    axs, rs, js = _limb_cover(J, s, "thigh", 0.45, 1.0, ease, h, k, 1.0)
                    out.append(_tube(f"{g}-thigh-{s}", g, subj, f"leg-{s}", axs, rs, js, round_start=False,
                                     open_start=True, bias=bias))
                    t0 = 0.0
                else:
                    t0 = 0.42
                axs, rs, js = _limb_cover(J, s, "shin", t0, 1.0, ease, h, k, 1.0)
                kh = _knee_hint(J, s, float(rs[0]), eye) if g == "thigh-highs" else None
                out.append(_tube(f"{g}-{s}", g, subj, f"leg-{s}", axs, rs, js, round_start=g == "thigh-highs",
                                 open_start=g == "socks", bias=bias))
                if kh:
                    out[-1]["details"].append(kh)
        elif g == "boots":
            for s in "rl":
                axs, rs, js = _limb_cover(J, s, "shin", 0.5, 1.0, 0.03 * h, h, k, 1.1)
                out.append(_tube(f"{g}-{s}", g, subj, f"foot-{s}", axs, rs, js, round_start=False, open_start=True,
                                 bias=bias))
        elif g in ("tie", "ribbon", "scarf"):
            for i, pr in enumerate(F3.garment_prims(rig, "x", g, o, motion, hands, grips)):
                if pr["t"] not in ("ell", "rcone", "cyl"):
                    continue
                # on the slim trunk: pulled in toward the spine as much as the trunk is slimmer
                pr = _slim_prim(pr, tor)
                out.append(_solid(f"{g}-{pr['piece']}-{i}", g, subj, "clothes", [pr], rad=0.05 * h, bias=-0.03 * h))
    return out


def _slim_prim(pr: dict, tor: Torso) -> dict:
    """A primitive made for the 3D mannequin's chest, moved onto the slimmer drawn trunk (toward the spine)."""
    c, X, Z, a, b = tor.at(1.3)
    Y = np.cross(Z, X)

    def mv(P):
        q = np.asarray(P, float) - c
        return c + X * (q @ X) * 0.8 + Z * (q @ Z) * 0.9 + Y * (q @ Y)
    out = dict(pr)
    for key in ("c", "a", "b"):
        if key in out:
            out[key] = mv(out[key])
    return out


def _fan_samples(poly3: np.ndarray, normal: np.ndarray, n: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Samples inside a small 3D polygon (a fan from its middle) with one normal."""
    c = poly3.mean(0)
    P = [c]
    m = len(poly3)
    for i in range(m):
        a, b = poly3[i], poly3[(i + 1) % m]
        for u in np.linspace(0, 1, n):
            for v in np.linspace(0, 1, n):
                if u + v <= 1:
                    P.append(c + u * (a - c) + v * (b - c))
    P = np.asarray(P)
    return P, np.repeat(unit(normal)[None], len(P), 0)


def _shell_parts(prims: List[dict], mat: str, subj: str, tor: Torso, h: float, bias: float,
                 only: Optional[str] = None) -> List[dict]:
    """A skirt (or coat tails) from the 3D cloth shell: rings from the waist to the hem, narrowed at the waist to
    the drawn trunk, with pleat lines on the front."""
    out = []
    for pr in prims:
        if pr["t"] != "shell" or (only and pr["piece"] != only):
            continue
        A = np.asarray(pr["A"], float)
        c = np.asarray(pr["c"], float)
        r0 = float(np.linalg.norm(A[:, 0]))
        hem_r = r0 * (1 + abs(pr["flare"]))
        waist_a = tor.at(3.4)[3] + 0.02 * h
        f = waist_a / max(r0, 1e-6)
        flare = max(0.0, hem_r / max(r0 * f, 1e-6) - 1)
        Ax = A[:, 0] * f
        Az = A[:, 2] * f
        Ay = A[:, 1]
        tilt = np.asarray(pr.get("hem_tilt", [0.0, 0.0]), float)
        pleated = bool(int(pr.get("pleats", 0)) or mat == "skirt")
        m_ = max(8, int(pr.get("pleats", 0)) or 12)
        n = 8 * m_ if pleated else 24
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        cs, sn = np.cos(th), np.sin(th)
        y_hem = -1.0 + np.maximum(0.0, tilt[0] * cs + tilt[1] * sn)
        # pleats: each fold edge (where a pleat line runs) stands out a little and hangs a little lower, so the hem
        # reads as a row of pleats instead of one smooth curve
        tri = np.zeros(n)
        if pleated:
            u = th * m_ / (2 * math.pi) - 0.5
            tri = 1.0 - 2.0 * np.abs(u - np.round(u))
            y_hem = y_hem - 0.07 * tri
        rings = []
        for kk in range(10):
            fy = kk / 9.0
            y = y_hem * fy
            rr = (1 + flare * np.abs(y)) * (1 + 0.1 * fy * fy * tri)
            P = c[None] + np.outer(rr * cs, Ax) + np.outer(y, Ay) + np.outer(rr * sn, Az)
            rings.append((P.mean(0), P))
        part = _part(pr["piece"], "tube", mat, subj, "clothes", {"waist": None, "hip-r": None, "hip-l": None},
                     rad=hem_r, rings=rings, open_end=True, bias=bias)
        # pleats: lines from below the waist band to the hem
        if pleated:
            for j in range(m_):
                a_ = 2 * math.pi * (j + 0.5) / m_
                yh = -1.0 + max(0.0, tilt[0] * math.cos(a_) + tilt[1] * math.sin(a_)) - 0.07
                ys = np.linspace(-0.3, yh * 0.985, 6)
                rr = (1 + flare * np.abs(ys)) * (1 + 0.1 * (ys / yh) ** 2)
                pts = c[None] + np.outer(rr * math.cos(a_), Ax) + np.outer(ys, Ay) + np.outer(rr * math.sin(a_), Az)
                part["details"].append({"pts": pts, "w": 0.42, "nrm": unit(np.outer(math.cos(a_), Ax) + np.outer(math.sin(a_), Az)).repeat(len(pts), 0)})
        out.append(part)
    return out


# ============================================================================================ hair

def _hair(rig, hair: dict, motion, floating: bool, subj: str, h: float) -> List[dict]:
    prims = F3.hair_prims(rig, "x", hair, motion, floating)
    out: List[dict] = []
    locks: Dict[str, list] = {}
    acc: Dict[str, list] = {}
    for pr in prims:
        if pr["piece"] == "hair-cap":
            out.append(_solid("hair-cap", "hair", subj, "hair", [pr], {"scalp": None}, rad=0.45 * h,
                              bias=-0.004 * h))
        elif pr["t"] == "rcone" and pr["mat"] == "hair":
            locks.setdefault(pr["piece"], []).append(pr)
        elif pr["piece"] in ("hairband", "hairpin", "hair-bow", "hair-bow-knot"):
            acc.setdefault(pr["piece"], []).append(pr)      # one solid: a band, not a row of beads
        elif pr["t"] in ("ell", "rcone", "cyl"):
            out.append(_solid(pr["piece"], pr["mat"], subj, "hair", [pr], rad=0.05 * h, bias=-0.006 * h))
    for name, prs in acc.items():
        out.append(_solid(name, prs[0]["mat"], subj, "hair", prs, rad=0.04 * h, bias=-0.03 * h))
    style = str(hair.get("style") or "medium")
    clumps = _clumps(rig, locks, style, h)
    for ci, cl in enumerate(clumps):
        pts, rr = cl[0], cl[1]
        # bangs are named apart: the crown highlight and the shading leave them out
        bang = len(cl) > 2 and bool(cl[2])
        frames, dr = _lock_frames(rig, pts, bang)
        part = _tube(f"{'bangs' if bang else 'clump'}{ci}", "hair", subj, "hair", pts, rr, {"scalp": pts[0]},
                     point_end=True, n_rings=max(10, 3 * len(pts)), n_pts=14, bias=-0.002 * h, frames=frames,
                     depth_ratio=dr)
        # inner strand lines: one or two confident lines per clump, from near its root toward the tip
        P = np.asarray(pts)
        L = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        n_lines = int(np.clip(round(L / (1.0 * h)) + 1, 1, 2)) if L > 3.0 * float(np.max(rr)) else 0
        for li in range(n_lines):
            off = (0.3 if ci % 2 else -0.3, -0.05 if ci % 2 else 0.1)[li]
            t0, t1 = 0.1 + 0.12 * li, 0.55 + 0.12 * ((ci + li) % 3)
            idx = np.linspace(t0, t1, 7) * (len(P) - 1)
            base = np.array([P[int(np.floor(x))] + (P[min(int(np.floor(x)) + 1, len(P) - 1)] - P[int(np.floor(x))]) *
                             (x - np.floor(x)) for x in idx])
            rad = np.interp(idx, np.arange(len(rr)), rr)
            X, _ = perp_basis(P[-1] - P[0])
            part["details"].append({"pts": base + X[None] * (rad * off)[:, None], "w": 0.38})
        out.append(part)
    return out


def _lock_frames(rig, pts, bang: bool):
    """A drawn lock is a flat ribbon lying on the head and back, not a round tube: wide across the hair surface,
    thin along its outward normal (bangs thinnest, so from the side they are an edge on the brow, not a visor)."""
    P = np.asarray(pts, float)
    T = np.gradient(P, axis=0)
    up = rig.Hd[:, 1]
    frames = []
    for i in range(len(P)):
        d = P[i] - rig.cr_c
        n = d - 0.7 * up * float(d @ up)
        if np.linalg.norm(n) < 1e-6:
            n = d
        X = np.cross(T[i], n)
        if np.linalg.norm(X) < 1e-9:
            X = perp_basis(T[i])[0]
        frames.append((unit(X), unit(n)))
    return frames, np.full(len(P), 0.45 if bang else 0.5)


CLUMPS = {"long": 9, "medium": 8, "bob": 7, "short": 11, "ponytail": 7, "twintails": 10, "braid": 6, "bun": 6,
          "buns": 6, "braids": 6}


def _resample_chain(P: np.ndarray, r: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if s[-1] <= 1e-9:
        return np.repeat(P[:1], n, 0), np.repeat(r[:1], n)
    ts = np.linspace(0, s[-1], n)
    return np.stack([np.interp(ts, s, P[:, c]) for c in range(3)], 1), np.interp(ts, s, r)


def _kmeans(X: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    n = len(X)
    if k >= n:
        return np.arange(n)
    cent = [X[0]]
    for _ in range(1, k):                                   # farthest-point start: the same clumps every time
        d = np.min([np.sum((X - c) ** 2, 1) for c in cent], 0)
        cent.append(X[int(np.argmax(d))])
    C = np.array(cent)
    lab = np.zeros(n, int)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        for c in range(k):
            if (lab == c).any():
                C[c] = X[lab == c].mean(0)
    return lab


def _taper(t: np.ndarray, bang: bool = False) -> np.ndarray:
    """Width profile of a drawn lock along its length (0 root .. 1 tip): full near the root, a gentle swell, then
    narrowing to a sharp point (a leaf, not a sausage)."""
    t = np.asarray(t, float)
    sw = 1.0 + 0.12 * np.sin(np.pi * np.clip(t / 0.6, 0, 1))
    k = 1.3 if bang else 1.7
    return np.maximum(sw * (1.0 - t ** k) ** 0.9, 0.0) + 0.004


def _clumps(rig, locks: Dict[str, list], style: str, h: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The 3D locks gathered into a few big clumps (bangs apart from the rest), each with an S-curve and a sharp
    tip: an anime head of hair is 6-10 clumps, not a curtain of strands."""
    chains = []
    for name, chain in locks.items():
        P = np.array([c["a"] for c in chain] + [chain[-1]["b"]], float)
        r = np.array([c["ra"] for c in chain] + [max(chain[-1]["rb"], 0.004)], float)
        chains.append(_resample_chain(P, r, 8))
    if not chains:
        return []
    if style == "spiky":
        return [(P, float(r[0]) * 1.1 * _taper(np.linspace(0, 1, len(r)), True)) for P, r in chains]
    Hd = rig.Hd
    front = []
    for P, r in chains:
        q = Hd.T @ (P[0] - rig.J["head"])
        front.append(q[2] > 0.12 * h and (Hd.T @ (P[-1] - P[0]))[1] < 0)
    out = []
    for grp, k in ((True, 4), (False, CLUMPS.get(style, 8))):
        idx = [i for i, f in enumerate(front) if f == grp]
        if not idx:
            continue
        X = np.array([chains[i][0].ravel() / h for i in idx])
        lab = _kmeans(X, min(k, len(idx)))
        for c in sorted(set(lab.tolist())):
            mem = [chains[idx[i]] for i in np.flatnonzero(lab == c)]
            Pm = np.mean([m[0] for m in mem], 0)
            spread = np.max([np.linalg.norm(m[0] - Pm, axis=1) + m[1] for m in mem], 0)
            rr = np.maximum(spread * 0.55, np.mean([m[1] for m in mem], 0) * 1.3)
            t = np.linspace(0, 1, len(Pm))
            # rhythm: big and small clumps in turn, the small ones a little shorter
            big = 1.0 if c % 3 != 1 else 0.72
            if big < 1.0 and len(Pm) > 2:
                Pm = Pm[0] + (Pm - Pm[0]) * 0.86
            r_root = float(np.max(rr[:max(2, len(rr) // 3)]))
            if style == "short":
                r_root = min(r_root, 0.12 * h)       # a short cut hugs the skull: no puffs
            rr = r_root * big * _taper(t, grp)
            # the root sinks under the cap: the crown outline stays one smooth curve, the tips carry the rhythm
            Pm = Pm.copy()
            if not grp:
                Pm[0] = Pm[0] + unit(rig.cr_c - Pm[0]) * 0.7 * float(rr[0])
            # an S-curve across the clump
            X3, _ = perp_basis(Pm[-1] - Pm[0])
            sgn = 1.0 if c % 2 == 0 else -1.0
            Pm = Pm + X3[None] * (sgn * 0.3 * float(np.mean(rr)) * np.sin(2 * math.pi * t) * t)[:, None]
            out.append((Pm, rr, grp))
    return out


def _hand2d(rig, s: str, shape: str, mat: str, subj: str, eye: np.ndarray, h: float, k: float,
            cam=None, grip_r: Optional[float] = None) -> List[dict]:
    """The drawn hand of side s (see hand2d): one part, or two for a hand gripping a prop."""
    from .hand2d import hand_parts
    return hand_parts(rig, s, shape, mat, subj, eye, h, k, cam, grip_r)


def _box2d(pr: dict, name: str, mat: str, subj: str, cam=None) -> dict:
    """A box prop (a blade, a guard, a book, a bag): its outline is the hull of its corners as the camera sees them
    (corner normals along the box's own diagonals, so a thin long box keeps both ends); a blade gets one highlight
    line along its edge."""
    A = np.asarray(pr["A"], float)
    c = np.asarray(pr["c"], float)
    q = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    P_ = c + q @ A.T
    ax = A / np.maximum(np.linalg.norm(A, axis=0), 1e-9)
    N_ = unit(q @ ax.T)
    # many samples on the faces so the front faces always have some
    f = np.linspace(-1, 1, 5)
    S, SN = [P_], [N_]
    for k in range(3):
        for sg in (-1.0, 1.0):
            g = np.array([[a, b] for a in f for b in f])
            Q = np.zeros((len(g), 3))
            o = [j for j in range(3) if j != k]
            Q[:, o[0]], Q[:, o[1]], Q[:, k] = g[:, 0], g[:, 1], sg
            S.append(c + Q @ A.T)
            SN.append(np.repeat((sg * ax[:, k])[None], len(g), 0))
    kw = {}
    if cam is not None:
        import cv2
        qq, zz = cam.project(P_)
        if (zz > 0.05).all():
            hull = cv2.convexHull(qq.astype(np.float32), returnPoints=False).ravel()
            kw["poly3"] = P_[hull]
    part = _blob(name, mat, subj, "props", np.vstack(S), np.vstack(SN), rad=0.05, hull=True, bias=-0.01, **kw)
    if name.endswith("blade"):
        L = np.linalg.norm(A, axis=0)
        i = int(np.argmax(L))
        j = int(np.argsort(L)[1])
        part["details"].append({"pts": np.array([c - A[:, i] * 0.92 + A[:, j] * 0.35, c + A[:, i] * 0.9 + A[:, j] * 0.35]),
                                "w": 0.45})
    return part


def _canopy2d(pr: dict, name: str, mat: str, subj: str) -> dict:
    """An umbrella's canopy: eight ribs from the top to a scalloped rim."""
    A = np.asarray(pr["A"], float)
    c = np.asarray(pr["c"], float)
    up = unit(A[:, 1])
    R = float(np.linalg.norm(A[:, 0]))
    x, z = unit(A[:, 0]), unit(A[:, 2])
    apex = c + up * 0.32 * R
    th = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    tips = np.array([c + R * (math.cos(a) * x + math.sin(a) * z) - up * 0.06 * R for a in th])
    return _part(name, "blob", mat, subj, "props", {}, rad=R, canopy={"apex": apex, "tips": tips, "c": c, "up": up},
                 bias=-0.01)
