# -*- coding: utf-8 -*-
from __future__ import annotations

import copy as _copy
import math
import numpy as np

from .. import solid as _S
from .._args import (unknown_name as _unknown, reject_unknown as _reject, signature_options as _sigopts,
                     strict_methods as _strict_methods)
from ..color import F32
from ..solid import (Material, Model, Part, cylinder, ellipsoid, intersect, material, plane, rounded_box, sphere, torus)
from . import _manifest as _manifest_labels
from .tables import (BOTTOMS, BOTTOM_COLORS, BUILDS, EXPRESSIONS, EXTRAS, EYE_COLORS, FACE_CHOICES, FACE_KEYS, HAIR_COLORS,
                     HAIR_KEYS, HAIR_STYLES, HATS, MATERIALS, OUTFIT_KEYS, OUTFITS, SHOES, SHOE_COLORS, SKIN_TONES, TOPS,
                     TOP_COLORS, _OUTFIT_ALIASES)
from .hair import _HAT_CLIP, _beard, _glasses, _hair, _hat
from .shapes import HEAD_SHAPES, _HeadCore, _Rx, _Ry, _Rz, _chain, _grp, _mix, _shade, _sub_chain, _v, _xf, round_cone, smooth_union
from .rig import _HumanRig, _MAIN_DOF, _lerp_pose, _mirror_pose, _pose_dict
from .face import _face, _face_layout
from .props import book, phone, prop


# ===================================================================================================== materials

def _mat(kind, color, style="clay"):
    """Material for a part ``kind`` (skin / cloth / hair / shoe / hard / metal) in a figure ``style``."""
    if isinstance(color, Material):
        return color
    if isinstance(color, (tuple, list)) and color and isinstance(color[0], str) and color[0] in _S.PRESETS:
        return material(*color)
    if isinstance(color, str) and color in _S.PRESETS:
        return material(color)
    c = color
    if kind == "metal":
        return material("metal", c, roughness=0.42)
    if style == "plastic":
        base = dict(roughness=0.3, specular=0.65, subsurface=0.12)
    elif style == "vinyl":
        base = dict(roughness=0.42, specular=0.5, clearcoat=0.25, subsurface=0.18)
    elif style == "plush":
        base = dict(roughness=0.97, specular=0.05, sheen=0.9, subsurface=0.2)
    elif style == "wood":
        if kind == "skin":
            return material("wood", c, roughness=0.5, specular=0.3)
        base = dict(roughness=0.45, specular=0.4, clearcoat=0.2)
    else:
        base = dict(roughness=0.74, specular=0.22, subsurface=0.3)
    if kind == "skin" and style in ("clay", "vinyl", "plastic"):
        base = dict(base, subsurface=base["subsurface"] + 0.25, sss_color=_mix(c, "#ff5040", 0.45))
    if kind == "hair" and style == "clay":
        base = dict(base, roughness=0.62, specular=0.32)
    if kind == "shoe" and style == "clay":
        base = dict(base, roughness=0.5, specular=0.35)
    if kind == "hard":
        base = dict(base, roughness=0.35, specular=0.6)
    if kind == "cloth" and style == "clay":
        base = dict(base, sheen=0.25)
    return Material(color=c, **base)


# ===================================================================================================== hands and feet

_HANDS = {
    #          finger curl (base, tip), thumb direction (canonical left hand: fingers -y, palm -x, thumb +z), spread
    "relaxed": dict(c1=10, c2=16, thumb=(-0.2, -0.6, 0.78), spread=0.0),
    "open": dict(c1=2, c2=6, thumb=(-0.1, -0.45, 0.9), spread=1.0),
    "wave": dict(c1=0, c2=4, thumb=(-0.05, -0.35, 0.95), spread=1.3),
    "flat": dict(c1=4, c2=6, thumb=(-0.2, -0.75, 0.6), spread=0.2),
    "fist": dict(c1=95, c2=105, thumb=(-0.8, -0.55, 0.2), spread=0.0),
    "grip": dict(c1=70, c2=62, thumb=(-0.7, -0.45, 0.55), spread=0.0),
    "point": dict(c1=95, c2=105, thumb=(-0.8, -0.5, 0.25), spread=0.0, index=True),
    "peace": dict(c1=95, c2=105, thumb=(-0.8, -0.5, 0.25), spread=0.8, index=True, middle=True),
    "thumbs_up": dict(c1=95, c2=105, thumb=(0.0, 0.85, 0.5), spread=0.0),
}


def _hand(L, kind="relaxed", detail="normal"):
    """Canonical left hand (wrist at the origin, fingers -y, palm facing -x, thumb toward +z): a soft mitten
    (one smooth flattened shape that curls toward the palm) or, for pointing / high detail, separate fingers."""
    cfg = _HANDS.get(kind, _HANDS["relaxed"])
    sx = 0.52                                         # palm thickness / width
    S = np.diag([sx, 1.0, 1.0])
    c1, c2 = math.radians(cfg["c1"]), math.radians(cfg["c2"])

    def P(x, y, z):
        return np.array([x / sx, y, z])
    wrist, palm_c, knuck = P(0, -0.02 * L, 0), P(-0.01 * L, -0.28 * L, 0), P(-0.02 * L, -0.5 * L, 0)
    sep = detail == "high" or cfg.get("index")
    th = np.asarray(cfg["thumb"], np.float64)
    th /= np.linalg.norm(th)
    base = np.array([-0.03 * L, -0.2 * L, 0.18 * L])
    thumb = _chain([base, base + th * 0.17 * L, base + th * 0.3 * L], [0.1 * L, 0.092 * L, 0.082 * L])
    if not sep:
        d1 = np.array([-math.sin(c1) / sx, -math.cos(c1), 0])
        mid = knuck + d1 * 0.2 * L
        d2 = np.array([-math.sin(c1 + c2) / sx, -math.cos(c1 + c2), 0])
        tip = mid + d2 * 0.16 * L
        pts, rr = [wrist, palm_c, knuck, mid, tip], [0.2 * L, 0.3 * L, 0.3 * L, 0.27 * L, 0.2 * L]
        if detail == "low":
            pts, rr = [wrist, palm_c, mid], [0.2 * L, 0.3 * L, 0.27 * L]
        mitt = _xf(_S.union(*_chain(pts, rr)), S, (0, 0, 0))
        return [smooth_union(mitt, _S.union(*thumb), k=0.06 * L)]
    palm = _xf(_S.union(*_chain([wrist, palm_c, knuck], [0.2 * L, 0.3 * L, 0.29 * L])), S, (0, 0, 0))
    fingers = []
    spread = cfg.get("spread", 0.0)
    for i, (z, ln) in enumerate(zip([0.18, 0.06, -0.06, -0.17], [0.42, 0.46, 0.43, 0.34])):
        straight = (i == 0 and cfg.get("index")) or (i == 1 and cfg.get("middle"))
        cu1, cu2 = (math.radians(4), math.radians(6)) if straight else (c1, c2)
        K = np.array([-0.02 * L, -0.46 * L, z * L * (1 + 0.35 * spread)])
        d1 = np.array([-math.sin(cu1), -math.cos(cu1), z * 0.25 * spread])
        M = K + d1 / np.linalg.norm(d1) * ln * 0.55 * L
        d2 = np.array([-math.sin(cu1 + cu2), -math.cos(cu1 + cu2), z * 0.25 * spread])
        T = M + d2 / np.linalg.norm(d2) * ln * 0.45 * L
        fingers += _chain([K, M, T], [0.082 * L, 0.078 * L, 0.07 * L])
    return [smooth_union(palm, _S.union(*fingers), _S.union(*thumb), k=0.035 * L)]


def _shoe_local(F, ah, kind="sneakers", wide=1.0):
    """Shoe in foot-local coordinates (ankle at the origin, sole at y = -ah, toes along +z). Returns (upper, sole)."""
    hgt = ah * (1.5 if kind not in ("heels",) else 1.3)
    toe = ellipsoid(0.25 * F * wide, hgt * 0.64, 0.45 * F).at(0, -ah + hgt * 0.42, 0.34 * F)
    heel = ellipsoid(0.21 * F * wide, hgt * 0.74, 0.28 * F).at(0, -ah + hgt * 0.5, -0.06 * F)
    shoe = smooth_union(toe, heel, k=0.12 * F)
    if kind in ("sneakers", "boots", "rain_boots", "shoes", "heels"):
        shoe = smooth_union(shoe, round_cone((0, -ah * 0.4, -0.02 * F), (0, 0.15 * ah, -0.02 * F), 0.17 * F * wide,
                                             0.15 * F * wide), k=0.06 * F)
    if kind == "slippers":
        shoe = shoe.round(0.03 * F)
    shoe = intersect(shoe, plane((0, -1, 0), ah))
    sole = None
    if kind in ("sneakers", "boots", "rain_boots", "sandals"):
        sole = intersect(shoe.round(0.012 * F), plane((0, 1, 0), -ah + ah * (0.32 if kind == "sneakers" else 0.24)))
        sole = intersect(sole, plane((0, -1, 0), ah))
    return shoe, sole


def _bare_foot(F, ah):
    f = smooth_union(ellipsoid(0.2 * F, ah * 0.55, 0.42 * F).at(0, -ah * 0.55, 0.3 * F),
                     sphere(0.16 * F).at(0, -ah * 0.45, -0.05 * F),
                     round_cone((0, -ah * 0.5, 0), (0, 0.2 * ah, 0), 0.15 * F, 0.13 * F), k=0.1 * F)
    return intersect(f, plane((0, -1, 0), ah))


# ===================================================================================================== character

def _front_panel(torso, inset, th, Rs, ps, Rc, pc, y0, y1, half_w, z0):
    """A bib / apron front that hugs the torso: a thin skin ``inset`` above the body surface, cut to the front
    (local z > z0 in the chest frame), between spine-frame heights y0..y1 and within +-half_w of the centre line."""
    s = torso.round(inset + th).shell(th)
    s = intersect(s, _xf(plane((0, 0, -1), 0.0), Rc, pc + Rc @ np.array([0, 0, z0])))
    s = intersect(s, _xf(plane((0, -1, 0), -y0), Rs, ps))
    s = intersect(s, _xf(plane((0, 1, 0), y1), Rc, pc))
    s = intersect(s, _xf(plane((1, 0, 0), half_w), Rc, pc))
    return intersect(s, _xf(plane((-1, 0, 0), half_w), Rc, pc))


def _band(shape, R, p, y0, y1, t):
    """A band (belt, cuff, hem stripe) hugging ``shape``: inflate by ``t`` and keep local y in [y0, y1]."""
    s = intersect(shape.round(t), _xf(plane((0, -1, 0), -y0), R, p))
    return intersect(s, _xf(plane((0, 1, 0), y1), R, p))


def _parse_hair(hair, build):
    default = {"toy": "short", "chibi": "short", "baby": "baby_tuft", "kid": "short", "teen": "side_part",
               "adult": "short", "elder": "balding", "hero": "quiff"}[build]
    if hair is None:
        return dict(style=default, color=HAIR_COLORS["dark_brown"] if build != "elder" else HAIR_COLORS["grey"])
    if isinstance(hair, str):
        if hair in HAIR_COLORS or hair.startswith("#"):
            return dict(style=default, color=HAIR_COLORS.get(hair, hair))
        if hair not in HAIR_STYLES:
            raise _unknown("hair style or colour", hair, HAIR_STYLES + list(HAIR_COLORS))
        return dict(style=hair, color=HAIR_COLORS["dark_brown"] if build != "elder" else HAIR_COLORS["grey"])
    h = _reject("hair", dict(hair), HAIR_KEYS)
    h.setdefault("style", default)
    if h["style"] not in HAIR_STYLES:
        raise _unknown("hair style", h["style"], HAIR_STYLES)
    c = h.get("color", "dark_brown")
    h["color"] = HAIR_COLORS.get(c, c)
    return h


def _parse_outfit(outfit):
    if outfit is None:
        outfit = "casual"
    if isinstance(outfit, str):
        if outfit in CAST:
            outfit = dict(CAST[outfit].get("outfit", {}))
        elif outfit in OUTFITS:
            outfit = dict(OUTFITS[outfit])
        else:
            raise _unknown("outfit", outfit, list(OUTFITS) + list(CAST))
    o = _reject("outfit", dict(outfit), OUTFIT_KEYS, {"colour": "colours", "color": "colours"})
    if "preset" in o:
        if o["preset"] not in OUTFITS:
            raise _unknown("outfit", o["preset"], OUTFITS)
        base = dict(OUTFITS[o.pop("preset")])
        base.update(o)
        o = base
    cols = dict(o.get("colours", o.get("colors", {})) or {})
    for slot, default, table in (("top", "tshirt", TOPS), ("bottom", "trousers", BOTTOMS), ("shoes", "sneakers", SHOES)):
        v = o.get(slot) or default
        v = _OUTFIT_ALIASES[slot].get(str(v).lower().replace(" ", "_").replace("-", "_"), v)
        if v not in table:
            raise _unknown(slot, v, table)
        o[slot] = v
    if o.get("hat") is not None and o["hat"] not in HATS:
        raise _unknown("hat", o["hat"], HATS)
    for ex in o.get("extras", []) or []:
        if ex not in EXTRAS:
            raise _unknown("extra", ex, EXTRAS)
    defaults = dict(top="#3f86d8", bottom="#384a6a" if o["bottom"] != "jeans" else "#3f5f8f", shoes="#f4f1ea",
                    accent="#e8613a", hat=None, sole="#f8f6f2", extra="#c8643a")
    for k, v in defaults.items():
        cols.setdefault(k, v)
    if cols["hat"] is None:
        cols["hat"] = cols["accent"] if o.get("hat") not in ("chef_hat", "nurse_cap") else "#f8f6f2"
    o["colours"] = cols
    o["extras"] = list(o.get("extras", []) or [])
    return o


def _parse_face(face):
    f = _reject("face", dict(face or {}), FACE_KEYS)
    ex = f.get("expression")
    if isinstance(ex, str) and ex not in EXPRESSIONS:
        raise _unknown("expression", ex, EXPRESSIONS)
    for k, choices in FACE_CHOICES.items():
        v = f.get(k)
        if isinstance(v, str) and v not in choices:
            raise _unknown(k if k != "eyes" else "eye style", v, choices)
    if isinstance(f.get("head"), str) and f["head"] not in HEAD_SHAPES:
        raise _unknown("head shape", f["head"], HEAD_SHAPES)
    return f


def _find_seat(seat, back_gap=0.12, reach=None):
    """Where to sit on a Model or Shape: the height of its widest flat top (a sofa's cushions, not its back or arms),
    the hip position (near the backrest when there is one) and the facing angle away from the backrest (or None)."""
    shape = _S.union(*[p.shape for p in seat.parts]) if isinstance(seat, Model) else seat
    lo, hi = (np.asarray(v, np.float64) for v in shape.bounds())
    n, ny = 28, 140
    xs = np.linspace(lo[0], hi[0], n + 2)[1:-1]
    zs = np.linspace(lo[2], hi[2], n + 2)[1:-1]
    ys = np.linspace(hi[1] + 0.01, max(lo[1], 0.0), ny)
    X, Z, Y = np.meshgrid(xs, zs, ys, indexing="ij")
    d = shape._d(X.ravel().astype(F32), Y.ravel().astype(F32), Z.ravel().astype(F32)).reshape(n, n, ny)
    inside = d < 0
    occ = inside.any(-1)
    top = np.where(occ, ys[np.argmax(inside, -1)], -1.0)
    tall = top > 0.15 * float(hi[1])
    if not tall.any():
        c = (lo + hi) / 2
        return dict(height=float(hi[1]), centre=c, facing=None)
    # only flat columns count (a rounded back or arm passes through every height on its slope)
    pad = np.pad(top, 1, mode="edge")
    rough = np.max([np.abs(pad[1:-1, 1:-1] - pad[a:a + n, b:b + n]) for a, b in ((0, 1), (2, 1), (1, 0), (1, 2))], 0)
    flat = tall & (rough < 0.025)
    if flat.sum() >= 4:
        tall_ = flat
    else:
        tall_ = tall
    vals = top[tall_]
    bins = np.round(vals / 0.03)
    u, cnt = np.unique(bins, return_counts=True)
    h0 = float(np.median(vals[bins == u[np.argmax(cnt)]]))
    seat_m = tall_ & (np.abs(top - h0) < 0.045)
    back_m = tall & (top > h0 + 0.12)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    sc = np.array([gx[seat_m].mean(), gz[seat_m].mean()])
    centre = np.array([sc[0], h0, sc[1]])
    facing = None
    if back_m.any():
        bc = np.array([gx[back_m].mean(), gz[back_m].mean()])
        f = sc - bc
        if np.linalg.norm(f) > 1e-3:
            f = f / np.linalg.norm(f)
            facing = math.degrees(math.atan2(f[0], f[1]))
            proj = gx[seat_m] * f[0] + gz[seat_m] * f[1]
            back_edge = float(np.percentile(proj, 4))
            front_edge = float(np.percentile(proj, 96))
            along = float(sc @ f)
            target = back_edge + back_gap
            if reach is not None:                 # short legs: slide forward so the knees reach the front edge
                target = max(target, front_edge - reach)
            target = min(target, along)
            centre[[0, 2]] = sc + f * (target - along)
    return dict(height=h0, centre=centre, facing=facing)


class Character(Model):
    """A posable clay character (a solid ``Model`` you can add to a Scene directly).

    ``Character(name, build="toy"|"chibi"|"baby"|"kid"|"teen"|"adult"|"elder"|"hero", height=None, skin=...,
    hair=dict(style, color), face=dict(expression, eyes, eye_color, brows, cheeks, glasses, beard, freckles,
    nose, head), outfit=preset or dict(top, bottom, shoes, colours, extras, hat), material="clay"|"plastic"|
    "plush"|"wood"|"vinyl", detail="low"|"normal"|"high")``.

    Pose with ``ch.pose("walk", phase=0.25)``, ``ch.blend("wave", 0.6)``, ``ch.look_at(point)``,
    ``ch.face(angle_or_point)``, ``ch.sit_on(chair)``, ``ch.hold(prop, hand="r")``; read points with
    ``ch.anchor("hand_r")``. Transforms (``.place(x, z)``, ``.at()``, ``.rotate_y()``) return a moved copy that
    keeps every Character method."""

    def __init__(self, name=None, build="toy", height=None, skin=None, hair=None, face=None, outfit=None,
                 material="clay", detail="normal", seed=0):
        if build not in BUILDS:
            raise _unknown("build", build, BUILDS)
        if material not in MATERIALS:
            raise _unknown("material", material, MATERIALS)
        self.name = name or "character"
        self._role = self.name                 # the cast role ("chef") survives renaming in a Scene
        self.hangs = False
        self.build = build
        self._cfg = BUILDS[build]
        self.height = float(height or self._cfg["height"])
        self._rig = _HumanRig(self._cfg, self.height)
        self.skin = SKIN_TONES[min(int(skin), len(SKIN_TONES) - 1)] if isinstance(skin, (int, np.integer)) else \
            (skin or SKIN_TONES[1])
        self.hair = _parse_hair(hair, build)
        f = _parse_face(face)
        self._expr_explicit = "expression" in f
        f.setdefault("expression", "smile")
        f.setdefault("eyes", self._cfg["eyes"])
        f.setdefault("eye_color", "brown")
        f.setdefault("head", self._cfg["head"])
        f.setdefault("nose", "button")
        self.face_spec = f
        self.outfit = _parse_outfit(outfit)
        self.style = material
        self.detail = detail
        self.seed = seed
        self._pose = _pose_dict("stand")
        self._look = None
        self._gaze = (0.0, 0.0)
        self._yaw = 0.0
        self._seat = None
        self._held = []
        self._W = (np.eye(3), np.zeros(3))
        self._cache = None

    # ---------------------------------------------------------------- Model interface
    @property
    def parts(self):
        if self._cache is None:
            self._cache = self._assemble()
        return self._cache[0]

    def _anchors(self):
        if self._cache is None:
            self._cache = self._assemble()
        return self._cache[1]

    def _clone(self):
        c = _copy.copy(self)
        c._pose = _copy.deepcopy(self._pose)
        c._held = list(self._held)
        c.face_spec = dict(self.face_spec)
        c.outfit = _copy.deepcopy(self.outfit)
        c.hair = dict(self.hair)
        c._cache = None
        return c

    def _manifest_info(self):
        return _manifest_labels.person_info(self._role)

    def _affine(self, F, g):
        c = self._clone()
        F0, g0 = self._W
        c._W = (np.asarray(F, np.float64) @ F0, np.asarray(F, np.float64) @ g0 + np.asarray(g, np.float64))
        return c

    def named(self, name):
        self.parts                                   # build once, share the geometry with the renamed copy
        c = self._clone()
        c.name = name
        c._cache = self._cache
        return c

    def copy(self):
        """An independent copy (pose and position included)."""
        return self._clone()

    def _dirty(self):
        self._cache = None
        return self

    def recolor(self, **colors):
        """Change colours by slot (skin, hair, top, bottom, shoes, accent, hat, eyes) or by part name."""
        c = self._clone()
        rest = {}
        for k, v in colors.items():
            if k == "skin":
                c.skin = v
            elif k == "hair":
                c.hair["color"] = HAIR_COLORS.get(v, v)
            elif k in ("top", "bottom", "shoes", "accent", "hat", "sole", "extra"):
                c.outfit["colours"][k] = v
            elif k == "eyes":
                c.face_spec["eye_color"] = v
            else:
                rest[k] = v
        if rest:
            return Model.recolor(c, **rest)
        return c

    # ---------------------------------------------------------------- posing
    def pose(self, name="stand", phase=0.0, mirror=False, **joints):
        """Set a named pose (``POSES``); ``phase`` 0~1 animates cyclic poses (walk, run, wave, clap, dance);
        ``mirror=True`` swaps left and right; joint overrides like ``elbow_r=60``, ``head=dict(turn=20)`` or
        ``spine_bend=10``; ``expression="laugh"``; ``hands=dict(r="point")``; ``lift=0.1`` (fraction of height)."""
        P = _pose_dict(name, phase)
        if mirror:
            P = _mirror_pose(P)
        self._pose = self._apply_overrides(P, joints)
        self._pose_name = name
        return self._dirty()

    def blend(self, name, t=0.5, phase=0.0, mirror=False, **joints):
        """Blend the current pose toward a named pose by ``t`` (0 = keep, 1 = the new pose)."""
        P = _pose_dict(name, phase)
        if mirror:
            P = _mirror_pose(P)
        P = self._apply_overrides(P, joints)
        self._pose = _lerp_pose(self._pose, P, float(t))
        return self._dirty()

    def _apply_overrides(self, P, joints):
        P = _copy.deepcopy(P)
        kinds = self._rig.kind
        for k, v in joints.items():
            if k in ("expression", "hands", "lift", "ground", "plant"):
                if k == "expression" and isinstance(v, str) and v not in EXPRESSIONS:
                    raise _unknown("expression", v, EXPRESSIONS)
                if k == "hands" and isinstance(v, str):
                    v = dict(l=v, r=v)
                P[k] = v
                continue
            if k in kinds or k == "pelvis":
                kind = kinds.get(k, "pelvis")
                if isinstance(v, dict):
                    P.setdefault(k, {}).update(v)
                else:
                    P.setdefault(k, {})[_MAIN_DOF[kind]] = float(v)
                continue
            for j in sorted(kinds, key=len, reverse=True):
                if k.startswith(j + "_"):
                    P.setdefault(j, {})[k[len(j) + 1:]] = float(v)
                    break
            else:
                raise _unknown("joint override", k, kinds)
        return P

    def express(self, expression):
        """Set the facial expression (``EXPRESSIONS``)."""
        if isinstance(expression, str) and expression not in EXPRESSIONS:
            raise _unknown("expression", expression, EXPRESSIONS)
        self.face_spec["expression"] = expression
        self._expr_explicit = True
        return self._dirty()

    def look_at(self, point):
        """Turn the head (neck + head joints, within their limits) and the eyes toward a world point; None resets."""
        self._look = None if point is None else _v(point)
        return self._dirty()

    def face(self, direction):
        """Turn the whole body: an angle in degrees (0 = facing +z, 90 = facing +x) or a world point / (x, z)."""
        self._faced = True
        if isinstance(direction, (int, float)):
            self._yaw = float(direction)
        else:
            d = np.asarray(direction, np.float64)
            target = np.array([d[0], 0, d[1]]) if len(d) == 2 else d
            F, g = self._W
            v = target - g
            yaw_w = math.degrees(math.atan2(v[0], v[2]))
            base = math.degrees(math.atan2(F[0, 2], F[2, 2]))
            self._yaw = yaw_w - base
        return self._dirty()

    def sit_on(self, seat=None, height=None):
        """Sit on a chair / bench / stool (a Model; its 'seat' part or top sets the height and position), at a
        given seat ``height``, or on the ground (``seat=None`` and no height -> ``sit_floor``)."""
        if seat is None and height is None:
            return self.pose("sit_floor")
        if seat is not None:
            found = _find_seat(seat, back_gap=self._rig.depth * 0.7, reach=self._rig.thigh + self._rig.lr * 0.4)
            if height is None:
                height = found["height"]
            c = found["centre"]
            F, g = self._W
            self._W = (F, np.array([c[0], g[1], c[2]]))
            self._seat_xz = (c[0], c[2])
            if found["facing"] is not None and not getattr(self, "_faced", False):
                # back to the backrest, facing out over the front edge
                self._yaw = found["facing"] - math.degrees(math.atan2(F[0, 2], F[2, 2]))
        self._seat = float(height)
        if getattr(self, "_pose_name", "stand") not in ("sit", "typing", "reading"):
            self.pose("sit")
        return self._dirty()

    def hold(self, item, hand="r", tilt=0.0, turn=0.0, offset=(0, 0, 0), **kw):
        """Put a prop in a hand: a Model (``prop("balloon")``) or a prop name with options. The prop's grip point
        goes to the hand; upright props (balloons, flags, umbrellas) stay upright."""
        if isinstance(item, str):
            item = prop(item, **kw)
        if hand not in ("l", "r", "both"):
            raise _unknown("hand", hand, ["l", "r", "both"])
        self._held.append((item, hand, float(tilt), float(turn), tuple(offset)))
        P = self._pose.setdefault("hands", {})
        for h in (("l", "r") if hand == "both" else (hand,)):
            if isinstance(P, dict) and P.get(h) in (None, "relaxed", "open"):
                P[h] = "grip"
        return self._dirty()

    def drop(self):
        """Remove every held prop."""
        self._held = []
        return self._dirty()

    def anchor(self, name):
        """World point of an anchor: hand_l, hand_r, head, head_top, mouth, eye_l, eye_r, nose, chest, belly, back,
        lap, seat, neck, feet, foot_l, foot_r, shoulder_l/r, elbow_l/r, hip_l/r, knee_l/r, ear_l/r."""
        A = self._anchors()
        if name not in A:
            raise KeyError(f"unknown anchor {name!r}; anchors: {sorted(k for k in A if not k.startswith('_'))}")
        F, g = self._W
        return F @ A[name] + g

    def frame(self, name):
        """(point, right, up, normal) of a surface frame ('chest', 'back', 'belly', 'face') in world space."""
        A = self._anchors()
        F, g = self._W
        p, r, u, n = A["_frame_" + name]
        return F @ p + g, F @ r, F @ u, F @ n

    # ---------------------------------------------------------------- building
    def _assemble(self):
        rig, cfg = self._rig, self._cfg
        H, R, T = rig.H, rig.R, rig.T
        P = _copy.deepcopy(self._pose)
        P.setdefault("pelvis", {})
        P["pelvis"]["yaw"] = P["pelvis"].get("yaw", 0.0) + self._yaw
        if cfg.get("stoop"):
            P.setdefault("spine", {})
            P["spine"]["bend"] = P["spine"].get("bend", 0.0) + cfg["stoop"]
            P.setdefault("neck", {})
            P["neck"]["nod"] = P["neck"].get("nod", 0.0) - cfg["stoop"] * 0.8
        expr = self.face_spec["expression"]
        if not self._expr_explicit and P.get("expression") is not None:
            expr = P["expression"]
        gaze = (0.0, 0.0)
        J = rig.fk(P)
        if self._look is not None:
            F, g = self._W
            tgt = np.linalg.solve(F, self._look - g)
            gaze = self._aim_head(P, J, tgt)
            J = rig.fk(P)
        # ---- feet: planted feet stay flat on the floor
        plant = P.get("plant", ["l", "r"])
        F_ = rig.foot
        feet = {}
        for sd in ("l", "r"):
            Ra, pa = J[f"ankle_{sd}"]
            if sd in plant:
                fwd = Ra @ np.array([0, 0, 1.0])
                yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
                Ra = _Ry(yaw) @ _Rx(-float(P.get(f"ankle_{sd}", {}).get("bend", 0.0)) * 0.0)
            feet[sd] = (Ra, pa)
        # ---- ground contact
        ah = rig.ah

        def foot_pts(sd):
            Ra, pa = feet[sd]
            return [pa + Ra @ np.array([0, -ah, -0.26 * F_]), pa + Ra @ np.array([0, -ah, 0.7 * F_]),
                    pa + Ra @ np.array([0.18 * F_, -ah, 0.3 * F_]), pa + Ra @ np.array([-0.18 * F_, -ah, 0.3 * F_])]
        ground = P.get("ground", "feet")
        if ground == "seat" and self._seat is None:
            ground = "feet"

        def body_pts(J):
            pts = []
            for sd in ("l", "r"):
                for j, r in ((f"knee_{sd}", rig.lr * 0.85), (f"hip_{sd}", rig.lr), (f"elbow_{sd}", rig.ar),
                             (f"wrist_{sd}", rig.ar * 0.8), (f"shoulder_{sd}", rig.ar * 1.1)):
                    pts.append(J[j][1] - np.array([0, r, 0]))
                hk = (J[f"hip_{sd}"][1] + J[f"knee_{sd}"][1]) / 2
                pts.append(hk - np.array([0, rig.lr * 0.95, 0]))
                ka = (J[f"knee_{sd}"][1] + J[f"ankle_{sd}"][1]) / 2
                pts.append(ka - np.array([0, rig.lr * 0.75, 0]))
            Rp, pp = J["pelvis"]
            for v in ((0, -0.2 * T, -0.6 * rig.depth), (0, 0.0, -rig.depth), (rig.hw, -0.15 * T, 0), (-rig.hw, -0.15 * T, 0)):
                pts.append(pp + Rp @ np.array(v))
            Rc, pc = J["chest"]
            for v in ((0, 0.2 * T, -rig.depth), (0, 0.2 * T, rig.depth), (rig.sw, 0.25 * T, 0), (-rig.sw, 0.25 * T, 0)):
                pts.append(pc + Rc @ np.array(v))
            Rh, ph = J["head"]
            hc = ph + Rh @ rig.head_off
            for v in ((0, R, 0), (0, 0, -R), (0, 0, R), (R, 0, 0), (-R, 0, 0), (0, -R, 0.3 * R)):
                pts.append(hc + Rh @ np.array(v))
            return pts
        if ground == "seat":
            Rp, pp = J["pelvis"]
            seat_pts = [pp + Rp @ np.array([0, -0.2 * T, -0.3 * rig.depth])]
            for sd in ("l", "r"):
                for t in (0.2, 0.5):
                    hk = J[f"hip_{sd}"][1] * (1 - t) + J[f"knee_{sd}"][1] * t
                    seat_pts.append(hk - np.array([0, rig.lr * 0.95, 0]))
            dy = self._seat - min(p[1] for p in seat_pts)
            for _ in range(12):                          # feet through the floor? bend the knees a bit more
                low = min(p[1] for sd in ("l", "r") for p in foot_pts(sd)) + dy
                if low >= -1e-4:
                    break
                for sd in ("l", "r"):
                    P.setdefault(f"knee_{sd}", {})
                    P[f"knee_{sd}"]["bend"] = min(P[f"knee_{sd}"].get("bend", 0) + 7, 150)
                J = rig.fk(P)
                feet = {sd: ((_Ry(math.degrees(math.atan2(*(J[f"ankle_{sd}"][0] @ np.array([0, 0, 1.0]))[[0, 2]])))),
                              J[f"ankle_{sd}"][1]) for sd in ("l", "r")}
        elif ground == "lowest":
            pts = body_pts(J) + foot_pts("l") + foot_pts("r")
            dy = -min(p[1] for p in pts)
        elif ground == "none":
            dy = 0.0
        else:
            dy = -min(p[1] for sd in ("l", "r") for p in foot_pts(sd))
        dy += float(P.get("lift", 0.0)) * self.height
        shift = np.array([0, dy, 0])
        J = {k: (Rk, pk + shift) for k, (Rk, pk) in J.items()}
        feet = {k: (Rk, pk + shift) for k, (Rk, pk) in feet.items()}
        parts, anchors = self._geometry(J, feet, P, expr, gaze)
        # ---- held props
        held = list(self._held)
        pn = getattr(self, "_pose_name", None)
        if pn == "phone":
            # the pose brings its own phone (or takes the one in the right hand), screen tilted up to the face
            mine = [h for h in held if h[1] == "r" and getattr(h[0], "name", "") == "phone" and h[2:4] == (0.0, 0.0)]
            if mine or not any(h[1] == "r" for h in held):
                held = [h for h in held if not any(h is m for m in mine)]
                ph_ = _copy.copy(mine[0][0]) if mine else phone()
                ph_.upright = True
                held.append((ph_, "r", -50.0, 180.0, (0, 0, 0)))
        if pn == "reading" and not held:                                  # ... and its own open book
            held.append((book(open=True), "both", -58.0, 0.0, (0, 0, 0)))
        for item, hand, tilt, turn, off in held:
            parts += self._place_prop(item, hand, tilt, turn, off, J, anchors)
        # ---- world transform
        F, g = self._W
        out = []
        for name, shapes, mat in parts:
            sh = _grp(shapes)
            if sh is None:
                continue
            if not (np.allclose(F, np.eye(3)) and np.allclose(g, 0)):
                sh = _xf(sh, F, g)
            out.append(Part(name, sh, mat))
        return out, anchors

    def _aim_head(self, P, J, tgt):
        rig = self._rig
        Rc, pc = J["chest"]
        Rh, ph = J["head"]
        hc = ph + Rh @ rig.head_off
        v = Rc.T @ (tgt - hc)
        yaw = math.degrees(math.atan2(v[0], v[2]))
        pitch = math.degrees(math.atan2(-v[1], math.hypot(v[0], v[2])))
        base_n = P.get("neck", {}).get("nod", 0.0) + P.get("head", {}).get("nod", 0.0)
        yn = float(np.clip(yaw * 0.4, -55, 55))
        yh = float(np.clip(yaw - yn, -50, 50))
        pn = float(np.clip(pitch * 0.45, -40, 40))
        phd = float(np.clip(pitch - pn, -35, 35))
        P.setdefault("neck", {}).update(turn=yn, nod=pn)
        P.setdefault("head", {}).update(turn=yh, nod=phd)
        ry = yaw - yn - yh
        rp = pitch - pn - phd
        del base_n
        return (float(np.clip(ry / 25.0, -1, 1)), float(np.clip(-rp / 25.0, -1, 1)))

    def _geometry(self, J, feet, P, expr, gaze):
        rig, cfg = self._rig, self._cfg
        H, R, T = rig.H, rig.R, rig.T
        st = self.style
        o = self.outfit
        cols = o["colours"]
        top = TOPS.get(o["top"], TOPS["tshirt"])
        bot = BOTTOMS.get(o["bottom"], BOTTOMS["trousers"])
        acc = {}
        order = []

        def add(name, shapes, mat):
            if not shapes:
                return
            if name not in acc:
                acc[name] = [[], mat]
                order.append(name)
            acc[name][0].extend(shapes)
        skin_m = _mat("skin", self.skin, st)
        top_m = _mat("metal" if o["top"] == "armor" else "cloth", cols["top"], st)
        bot_m = _mat("cloth", cols["bottom"], st)
        ct = 0.018 * H * top.get("thick", 1.0)
        cb = 0.012 * H * bot.get("thick", 1.0) * (0.4 if bot.get("thin") else 1.0)
        A = {}
        # ---------------- torso
        Rp, pp = J["pelvis"]
        Rs, ps = J["spine"]
        Rc, pc = J["chest"]
        Rn, pn = J["neck"]
        Rh, ph = J["head"]
        hx, hy, hz = rig.hw + rig.lr * 0.95, 0.25 * T, rig.depth * 0.9
        hips_c = pp + Rp @ np.array([0, 0.03 * H, -0.005 * H])
        hips = _xf(ellipsoid(hx, hy, hz), Rp, hips_c)
        wx, wz = cfg["ww"] * H, rig.depth * cfg["belly"]
        belly = _xf(ellipsoid(wx, 0.3 * T, wz), Rs, ps + Rs @ np.array([0, 0.1 * T, 0.05 * H * (cfg["belly"] - 1)]))
        cx, cz = cfg["cw"] * H, rig.depth
        chest_c = pc + Rc @ np.array([0, 0.2 * T, 0])
        chest = _xf(ellipsoid(cx, 0.28 * T, cz), Rc, chest_c)
        sl, sr = J["shoulder_l"][1], J["shoulder_r"][1]
        shoulders = round_cone(sl, sr, rig.ar * 1.12, rig.ar * 1.12)
        k_t = 0.1 * H
        # ---------------- limbs
        arms = {sd: ([J[f"shoulder_{sd}"][1], J[f"elbow_{sd}"][1], J[f"wrist_{sd}"][1]],
                     [rig.ar, rig.ar * 0.9, rig.ar * 0.8]) for sd in ("l", "r")}
        legs = {sd: ([J[f"hip_{sd}"][1], J[f"knee_{sd}"][1], J[f"ankle_{sd}"][1]],
                     [rig.lr, rig.lr * 0.84, rig.lr * 0.7]) for sd in ("l", "r")}
        sleeve = top.get("sleeve", 0.4) if o["top"] != "none" else 0.0
        legcov = bot.get("leg", 0.95)
        # ---------------- top garment (or bare torso)
        # a soft "drape" from the chest to the hips keeps the silhouette simple (a bean, not an hourglass)
        Sz = np.diag([1.0, 1.0, (rig.depth * 0.95) / max(cx, 1e-6)])
        q0 = np.linalg.solve(Rs @ Sz, hips_c + Rp @ np.array([0, 0.02 * T, 0]) - ps)
        q1 = np.linalg.solve(Rs @ Sz, chest_c - ps)
        drape = _xf(round_cone(q0, q1, max(hx, wx) * 1.0, cx * 0.94), Rs @ Sz, ps)
        torso_parts = [chest, belly, shoulders, drape]
        self._torso_sh = smooth_union(*torso_parts, k=k_t)
        if o["top"] == "none":
            add("skin", [smooth_union(*torso_parts, k=k_t)], skin_m)
            hem = 0.3
        else:
            hem = top.get("hem", -0.12)
            if bot.get("bib") and not top.get("skirt"):
                hem = max(hem, 0.12)              # tucked into the overalls
            tparts = torso_parts + ([hips] if hem < 0.05 else [])
            if hem < 0.05:
                for sd in ("l", "r"):
                    hp_, kp_ = J[f"hip_{sd}"][1], J[f"knee_{sd}"][1]
                    tparts.append(round_cone(hp_, hp_ + (kp_ - hp_) * 0.2, rig.lr * 1.02, rig.lr))
            for sd in ("l", "r"):
                if sleeve > 0:
                    sh, spts, srs = _sub_chain(*arms[sd], 0.0, sleeve)
                    srs = [r + ct * 0.8 for r in srs]
                    srs[-1] *= top.get("cuff", 1.15)
                    sh = _chain(spts, srs)
                    tparts.append(sh[0])
                    tail = sh[1:]
                    if tail:
                        add("top", tail, top_m)
            body = smooth_union(*tparts, k=k_t).round(ct)
            if hem > -0.9:
                body = intersect(body, _xf(plane((0, -1, 0), -(hem * T)), Rp, pp))
            add("top", [body], top_m)
            if top.get("skirt"):
                add("top", self._skirt(J, legs, top["skirt"], top.get("flare", 2.4), hips, ct), top_m)
            if top.get("tails"):
                add("top", self._skirt(J, legs, top["tails"], 1.45, hips, ct * 1.2), top_m)
            if top.get("band"):
                add("top", [_band(hips, Rp, pp, hem * T - 0.001, hem * T + 0.07 * H, ct * 1.6)], top_m)
            if top.get("collar"):
                add("top", [_xf(torus(rig.ar * 1.35 + 0.02 * H, 0.03 * H), Rn @ _Rx(-14), pn + Rn @ np.array([0, 0.02 * H, 0.01 * H]))],
                    top_m)
            if top.get("hood"):
                add("top", [_xf(ellipsoid(rig.sw * 0.7, 0.16 * H, 0.2 * H), Rc, pc + Rc @ np.array([0, 0.4 * T, -0.8 * cz]))],
                    top_m)
            if top.get("pocket"):
                add("top", [_xf(rounded_box(wx * 1.0, 0.2 * T, 0.05 * H, 0.025 * H), Rs,
                                ps + Rs @ np.array([0, 0.0, wz + ct * 0.6]))], top_m)
            if top.get("buttons"):
                # flat round buttons, never closer than two diameters (short toy torsos get two, not a blob)
                ys = (0.3, 0.16, 0.02) if T > 1.6 * H else (0.3, 0.06)
                bs = [_xf(ellipsoid(0.02 * H, 0.02 * H, 0.011 * H), Rc,
                          pc + Rc @ np.array([0, y * T, cz * math.sqrt(max(0.05, 1 - ((y * T - 0.18 * T) / (0.29 * T)) ** 2))
                                              + ct * 1.1 + 0.006 * H]))
                      for y in ys]
                add("accent", bs, _mat("hard", cols["accent"], st))
            if top.get("lapels") or top.get("open"):
                front = cz + ct
                inner = _xf(rounded_box(cx * 0.42, 0.42 * T, 0.03 * H, 0.02 * H), Rc,
                            pc + Rc @ np.array([0, 0.2 * T, front * 0.92]))
                add("shirt", [inner], _mat("cloth", "#f6f4ef" if o["top"] != "jacket" else cols.get("inner", "#f6f4ef"), st))
                if top.get("lapels"):
                    for sdx in (1, -1):
                        add("top", [_xf(ellipsoid(0.1 * H, 0.26 * T, 0.03 * H), Rc @ _Rz(sdx * 18),
                                        pc + Rc @ np.array([sdx * cx * 0.32, 0.26 * T, front * 0.95]))], top_m)
            if top.get("pauldrons"):
                add("accent", [sphere(rig.ar * 1.9).at(*J[f"shoulder_{sd}"][1]) for sd in ("l", "r")],
                    _mat("metal", "#c9ccd2", st))
            if o["top"] == "jersey":
                add("accent", [_band(chest, Rc, pc, 0.42 * T, 0.48 * T, ct * 1.3)], _mat("cloth", cols["accent"], st))
            if o["top"] == "spacesuit":
                add("accent", [_band(belly, Rs, ps, 0.0, 0.06 * H, ct * 1.2)], _mat("cloth", cols["accent"], st))
        # ---------------- bottom garment
        if o["bottom"] != "none" and not (top.get("skirt", 0) >= 0.5 and o["bottom"] in ("skirt",)):
            bparts = [hips]
            for sd in ("l", "r"):
                if legcov > 0:
                    sh, spts, srs = _sub_chain(*legs[sd], 0.0, legcov)
                    srs = [r + cb for r in srs]
                    srs[-1] *= bot.get("cuff", 1.1)
                    sh = _chain(spts, srs)
                    bparts.append(sh[0])
                    if len(sh) > 1:
                        add("bottom", sh[1:], bot_m)
            add("bottom", [smooth_union(*bparts, k=0.08 * H).round(cb)], bot_m)
            if bot.get("skirt"):
                add("bottom", self._skirt(J, legs, bot["skirt"], bot.get("flare", 2.3), hips, cb), bot_m)
            if bot.get("bib"):
                bib = _front_panel(self._torso_sh, ct + 0.012 * H, 0.03 * H, Rs, ps, Rc, pc, -0.45 * T, 0.32 * T,
                                   0.55 * cx, 0.3 * cz)
                bib = _S.union(bib, _band(hips, Rp, pp, -0.1 * T, 0.22 * T, cb + 0.01 * H))
                straps = [round_cone(ps + Rs @ np.array([sdx * wx * 0.45, 0.48 * T, wz + ct]),
                                     J[f"shoulder_{sd}"][1] + Rc @ np.array([-sdx * rig.ar * 0.3, rig.ar * 1.2, 0.0]),
                                     0.035 * H, 0.035 * H) for sd, sdx in (("l", 1), ("r", -1))]
                straps += [round_cone(J[f"shoulder_{sd}"][1] + Rc @ np.array([-sdx * rig.ar * 0.3, rig.ar * 1.2, 0.0]),
                                      ps + Rs @ np.array([sdx * wx * 0.4, 0.25 * T, -wz - ct]), 0.035 * H, 0.035 * H)
                           for sd, sdx in (("l", 1), ("r", -1))]
                add("bottom", [bib] + straps, bot_m)
        elif o["bottom"] == "none" and not top.get("skirt"):
            add("skin", [hips], skin_m)
        # ---------------- skin: neck, head, arms, legs
        neck_r = max(rig.ar * 1.25, 0.12 * H)
        add("skin", [round_cone(pn - Rn @ np.array([0, 0.04 * H, 0]), ph + Rh @ np.array([0, 0.18 * H, 0]), neck_r,
                                neck_r * 0.92)], skin_m)
        for sd in ("l", "r"):
            if sleeve < 0.97:
                sh, _, _ = _sub_chain(*arms[sd], max(0.0, sleeve - 0.08), 1.0)
                add("skin", sh, skin_m)
            leg_skin_from = legcov - 0.05 if o["bottom"] not in ("skirt", "long_skirt", "none") or legcov > 0 else 0.0
            if top.get("skirt") or bot.get("skirt"):
                leg_skin_from = 0.05
            if leg_skin_from < 0.94:
                sh, _, _ = _sub_chain(*legs[sd], max(0.0, leg_skin_from), 1.0)
                add("skin", sh, skin_m)
        # ---------------- hands
        L = rig.hand
        hands = P.get("hands", {}) or {}
        gloves = "gloves" in o["extras"] or o["top"] in ("spacesuit",)
        for sd, s in (("l", 1), ("r", -1)):
            Rw, pw = J[f"wrist_{sd}"]
            M = Rw @ np.diag([s, 1.0, 1.0])
            hs = [_xf(h, M, pw) for h in _hand(L, hands.get(sd, "relaxed"), self.detail)]
            if gloves:
                add("gloves", hs, _mat("cloth", cols.get("gloves", "#f2f0ea" if o["top"] == "spacesuit" else cols["accent"]), st))
            else:
                add("skin", hs, skin_m)
            A[f"hand_{sd}"] = pw + M @ np.array([-0.12 * L, -0.45 * L, 0.06 * L])
            A[f"_hand_{sd}"] = (M, pw)
        # ---------------- feet / shoes
        shoe_kind = o["shoes"]
        shoe_m = _mat("shoe", cols["shoes"], st)
        if shoe_kind == "rain_boots":
            shoe_m = Material(color=cols["shoes"], roughness=0.18, specular=0.8, clearcoat=0.5)
        for sd in ("l", "r"):
            Ra, pa = feet[sd]
            if shoe_kind == "barefoot":
                add("skin", [_xf(_bare_foot(rig.foot, rig.ah), Ra, pa)], skin_m)
                continue
            if shoe_kind == "sandals":
                add("skin", [_xf(_bare_foot(rig.foot, rig.ah), Ra, pa)], skin_m)
                sole = rounded_box(0.42 * rig.foot, 0.25 * rig.ah, 1.02 * rig.foot, 0.08 * rig.ah).at(
                    0, -rig.ah * 0.87, 0.22 * rig.foot)
                strap = _xf(torus(0.19 * rig.foot, 0.045 * rig.ah), _Rz(90) @ _Rx(0), (0, -rig.ah * 0.35, 0.3 * rig.foot))
                add("shoes", [_xf(sole, Ra, pa), _xf(strap, Ra, pa)], shoe_m)
                continue
            upper, sole = _shoe_local(rig.foot, rig.ah, shoe_kind)
            add("shoes", [_xf(upper, Ra, pa)], shoe_m)
            if sole is not None:
                add("soles", [_xf(sole, Ra, pa)], _mat("shoe", cols.get("sole", "#f8f6f2") if shoe_kind == "sneakers"
                                                       else _shade(cols["shoes"], 0.6), st))
            if shoe_kind in ("boots", "rain_boots"):
                kn, an = J[f"knee_{sd}"][1], pa
                hgt = 0.38 if shoe_kind == "boots" else 0.62
                add("shoes", [round_cone(an, an + (kn - an) * hgt, rig.lr * 0.62 + cb * 2.2, rig.lr * 0.7 + cb * 2.6)],
                    shoe_m)
        # ---------------- head
        head = _HeadCore(self.face_spec.get("head", "round"), R, cfg["headw"])
        hc = ph + Rh @ rig.head_off
        lay = _face_layout(cfg, R)
        lay.update({k: self.face_spec[k] * R for k in ("ex", "ey", "erx", "ery", "my") if k in self.face_spec})

        def H_(shapes):
            return [_xf(s, Rh, hc) for s in shapes]
        add("skin", H_([head.shape_()]), skin_m)
        ears = self.face_spec.get("ears", "round")
        if ears and self.hair["style"] != "hijab":
            ex_ = head.ca[0] * 0.98
            esz = 1.0 if ears == "round" else (1.3 if ears == "big" else 0.8)
            for s in (1, -1):
                ear = _xf(ellipsoid(0.1 * R * esz, 0.19 * R * esz, 0.13 * R * esz), _Ry(s * 22) @ _Rz(-s * 8),
                          (s * ex_, -0.1 * R, -0.04 * R))
                if ears == "elf":
                    ear = round_cone((s * ex_, -0.1 * R, -0.04 * R), (s * (ex_ + 0.3 * R), 0.25 * R, -0.12 * R),
                                     0.12 * R, 0.02 * R)
                add("skin", H_([ear]), skin_m)
        nose = self.face_spec.get("nose", "button")
        if nose and nose != "none":
            nr = {"button": 0.085, "small": 0.06, "round": 0.11, "long": 0.08, "pointy": 0.07}.get(nose, 0.085) * R
            Pn, Nn = head.front(np.array([0.0]), np.array([lay["ny"]]))
            pn_ = Pn[0] + Nn[0] * nr * 0.2
            if nose == "long":
                nshape = _xf(ellipsoid(nr * 0.85, nr, nr * 1.7), _Rx(15), pn_ + np.array([0, -0.02 * R, 0.06 * R]))
            elif nose == "pointy":
                nshape = round_cone(pn_ - np.array([0, -0.05 * R, 0.05 * R]), pn_ + np.array([0, -0.06 * R, 0.14 * R]),
                                    nr, nr * 0.35)
            else:
                nshape = sphere(nr).at(*pn_)
            add("nose", H_([nshape]), skin_m.copy(shadow=0.45))
        brow_col = self.face_spec.get("brows", None)
        if brow_col is None or brow_col is True:
            brow_col = _shade(self.hair["color"], 0.82) if self.hair["style"] not in ("bald", "hijab") else "#5a4032"
        if brow_col is False:
            brow_col = None
        eye_col = EYE_COLORS.get(self.face_spec["eye_color"], self.face_spec["eye_color"])
        g2 = self.face_spec.get("gaze", gaze if gaze != (0.0, 0.0) else (0.0, 0.0))
        fd = _face(head, lay, expr, self.skin, eye_col, brow_col, self.face_spec.get("eyes", "round"), tuple(g2),
                   self.face_spec.get("freckles", False) and self.detail != "low",
                   self.face_spec.get("cheeks", True), self.detail, self.face_spec.get("mouth"))
        shades = self.face_spec.get("glasses") == "sunglasses" or "sunglasses" in o["extras"]
        for nm, (shape, mat) in fd.items():
            if shades and nm in ("eye shine", "eye whites", "irises", "pupils", "eyes"):
                continue                                  # dark lenses hide the eyes (and nothing pokes through)
            add(nm, H_([shape]), mat)
        # hair, beard, glasses, hat
        hair_m = _mat("hair", self.hair["color"], st)
        if self.hair["style"] == "hijab":
            hair_m = _mat("cloth", self.hair["color"], st)
        hs, ha = _hair(self.hair["style"], head, self.detail, side=1 if self.seed % 2 == 0 else -1)
        hat = o.get("hat")
        if hat in _HAT_CLIP and self.hair["style"] != "hijab":
            cy, tilt = _HAT_CLIP[hat]
            if cy is None:
                hs, ha = [], []
            else:
                clip = plane(_Rx(tilt) @ np.array([0, 1.0, 0]), (head.ca[1] + cy) * 1.0 if cy < -0.2 else cy * R)
                hs = [intersect(h, clip, k=0.04 * R) for h in hs]
                ha = [intersect(h, clip) for h in ha]
        add("hair", H_(hs), hair_m)
        add("hair tie", H_(ha), _mat("hard", self.hair.get("tie", cols["accent"]), st))
        if self.face_spec.get("beard"):
            bk = self.face_spec["beard"]
            bk = "full" if bk is True else bk
            bcol = self.face_spec.get("beard_color", "#f4f2ee" if bk == "santa" else self.hair["color"])
            add("beard", H_(_beard(head, lay, bk)), _mat("hair", bcol, st))
        if self.face_spec.get("glasses"):
            gk = self.face_spec["glasses"]
            gk = "round" if gk is True else gk
            rims, lenses = _glasses(head, lay, gk)
            add("glasses", H_(rims), Material(color=self.face_spec.get("glasses_color", "#2b2523"), roughness=0.25,
                                              specular=0.8))
            if lenses:
                add("lenses", H_(lenses), Material(color="#101116", roughness=0.18, specular=0.9, reflect=0.08))
        if hat:
            for nm, shp, m in _hat(hat, head, cols["hat"], cols["accent"] if hat != "chef_hat" else "#f8f6f2"):
                add(nm, H_([shp]), m if isinstance(m, Material) else _mat("cloth" if nm == "hat" else "hard", m, st))
        # ---------------- extras
        for ex in o["extras"]:
            self._extra(ex, add, J, cols, st, head, Rh, hc, lay, cz, wx, wz, ct)
        # ---------------- anchors
        top_y = head.cc + head.ca[1]
        A.update(head=hc, head_top=hc + Rh @ np.array([0, top_y + 0.08 * R, 0]), neck=pn,
                 mouth=hc + Rh @ head.front(np.array([0.0]), np.array([lay["my"]]))[0][0],
                 nose=hc + Rh @ head.front(np.array([0.0]), np.array([lay["ny"]]))[0][0],
                 eye_l=hc + Rh @ head.front(np.array([lay["ex"]]), np.array([lay["ey"]]))[0][0],
                 eye_r=hc + Rh @ head.front(np.array([-lay["ex"]]), np.array([lay["ey"]]))[0][0],
                 ear_l=hc + Rh @ np.array([head.ca[0], -0.12 * R, -0.04 * R]),
                 ear_r=hc + Rh @ np.array([-head.ca[0], -0.12 * R, -0.04 * R]),
                 chest=chest_c + Rc @ np.array([0, 0.0, cz + ct]), belly=ps + Rs @ np.array([0, 0.1 * T, wz + ct]),
                 back=chest_c + Rc @ np.array([0, 0.0, -cz - ct]), pelvis=pp,
                 seat=pp + Rp @ np.array([0, -0.2 * T, -0.3 * rig.depth]),
                 lap=(J["hip_l"][1] + J["hip_r"][1] + J["knee_l"][1] + J["knee_r"][1]) / 4 + np.array([0, rig.lr, 0]),
                 feet=np.array([(feet["l"][1][0] + feet["r"][1][0]) / 2, 0.0, (feet["l"][1][2] + feet["r"][1][2]) / 2]))
        for sd in ("l", "r"):
            for j in ("shoulder", "elbow", "hip", "knee", "wrist", "ankle"):
                A[f"{j}_{sd}"] = J[f"{j}_{sd}"][1]
            Ra, pa = feet[sd]
            A[f"foot_{sd}"] = pa + Ra @ np.array([0, -rig.ah, 0.25 * rig.foot])
        A["_frame_chest"] = (chest_c + Rc @ np.array([0, -0.2 * T, cz * 0.97 + ct]), Rc @ np.array([1.0, 0, 0]),
                             Rc @ np.array([0, 1.0, 0]), Rc @ np.array([0, 0, 1.0]))
        A["_frame_back"] = (chest_c + Rc @ np.array([0, -0.12 * T, -cz - ct]), Rc @ np.array([-1.0, 0, 0]),
                            Rc @ np.array([0, 1.0, 0]), Rc @ np.array([0, 0, -1.0]))
        A["_frame_belly"] = (A["belly"], Rs @ np.array([1.0, 0, 0]), Rs @ np.array([0, 1.0, 0]), Rs @ np.array([0, 0, 1.0]))
        A["_frame_face"] = (A["nose"], Rh @ np.array([1.0, 0, 0]), Rh @ np.array([0, 1.0, 0]), Rh @ np.array([0, 0, 1.0]))
        A["_R"] = R
        A["_H"] = H
        A["_yaw"] = math.degrees(math.atan2(*(Rp @ np.array([0, 0, 1.0]))[[0, 2]]))
        parts = [(nm, acc[nm][0], acc[nm][1]) for nm in order]
        return parts, A

    def _skirt(self, J, legs, length, flare, hips, t):
        rig = self._rig
        out = [hips.round(t * 1.2)]
        for sd in ("l", "r"):
            pts, rs = legs[sd]
            sh, spts, srs = _sub_chain(pts, rs, 0.0, length)
            n = len(spts)
            # a skirt hangs: it flares round a standing leg but lies along a thigh that points forward (sitting)
            v = np.asarray(pts[1], np.float64) - np.asarray(pts[0], np.float64)
            down = max(0.0, -v[1] / max(np.linalg.norm(v), 1e-9))
            fl = 1.25 + (flare - 1.25) * down ** 2
            rr = [rig.lr * (1.25 + (fl - 1.25) * (i / max(n - 1, 1))) for i in range(n)]
            out.append(smooth_union(*_chain(spts, rr), k=0.05 * rig.H))
        return [smooth_union(*out, k=0.12 * rig.H)]

    def _extra(self, ex, add, J, cols, st, head, Rh, hc, lay, cz, wx, wz, ct):
        rig = self._rig
        H, T, R = rig.H, rig.T, rig.R
        Rc, pc = J["chest"]
        Rs, ps = J["spine"]
        Rp, pp = J["pelvis"]
        Rn, pn = J["neck"]
        ec = cols.get(ex, cols["extra"])

        def H_(shapes):
            return [_xf(s, Rh, hc) for s in shapes]
        if ex == "backpack":
            bp = _xf(rounded_box(rig.sw * 1.35, 0.55 * T, 0.32 * H, 0.1 * H), Rc, pc + Rc @ np.array([0, 0.12 * T, -cz - ct - 0.15 * H]))
            pocket = _xf(rounded_box(rig.sw * 0.95, 0.24 * T, 0.12 * H, 0.05 * H), Rc,
                         pc + Rc @ np.array([0, -0.02 * T, -cz - ct - 0.33 * H]))
            straps = []
            for sd, s in (("l", 1), ("r", -1)):
                a = pc + Rc @ np.array([s * rig.sw * 0.45, 0.4 * T, -cz * 0.6])
                b = J[f"shoulder_{sd}"][1] + Rc @ np.array([-s * rig.ar * 0.4, rig.ar * 1.15, 0.0])
                c = pc + Rc @ np.array([s * rig.sw * 0.55, 0.0, cz + ct])
                straps += _chain([a, b, c], [0.035 * H] * 3)
            add("backpack", [bp, pocket] + straps, _mat("cloth", ec, st))
        elif ex == "bag":
            bag = _xf(rounded_box(0.42 * H, 0.32 * H, 0.14 * H, 0.06 * H), Rp, pp + Rp @ np.array([-rig.hw - rig.lr * 1.3, 0.0, 0.08 * H]))
            a = J["shoulder_l"][1] + Rc @ np.array([0, rig.ar * 1.1, 0])
            b = pc + Rc @ np.array([0, 0.1 * T, cz + ct * 1.4])
            c = pp + Rp @ np.array([-rig.hw - rig.lr * 1.2, 0.12 * H, 0.08 * H])
            add("bag", [bag] + _chain([a, b, c], [0.03 * H] * 3), _mat("cloth", ec, st))
        elif ex == "scarf":
            ring = _xf(torus(rig.ar * 1.3 + 0.08 * H, 0.08 * H), Rn @ _Rx(-10), pn + Rn @ np.array([0, 0.02 * H, 0]))
            tail = _xf(rounded_box(0.16 * H, 0.45 * T, 0.06 * H, 0.03 * H), Rc @ _Rz(8),
                       pc + Rc @ np.array([0.1 * H, 0.2 * T, cz + ct + 0.06 * H]))
            add("scarf", [ring, tail], _mat("cloth", ec, st))
        elif ex == "cape":
            Rw = Rc
            c = pc + Rc @ np.array([0, -0.35 * T, -cz * 0.55])
            shell = _xf(ellipsoid(rig.sw * 1.3, 0.75 * T + 0.5 * rig.thigh * 2, cz * 1.4), Rw, c).shell(0.085 * H)
            cape = intersect(shell, _xf(plane((0, 0, 1), -cz * 0.2), Rc, pc))
            cape = intersect(cape, _xf(plane((0, 1, 0), 0.4 * T), Rc, pc))
            add("cape", [cape, _xf(torus(rig.ar * 1.3 + 0.05 * H, 0.035 * H), Rn, pn)], _mat("cloth", ec, st))
        elif ex in ("glasses", "sunglasses"):
            rims, lenses = _glasses(head, lay, ex)
            add("glasses", H_(rims), Material(color="#2b2523", roughness=0.25, specular=0.8))
            if lenses:
                add("lenses", H_(lenses), Material(color="#101116", roughness=0.18, specular=0.9, reflect=0.08))
        elif ex == "headphones":
            band = intersect(_xf(torus(head.ca[0] * 1.12, 0.06 * R), _Rx(90), (0, 0.05 * R, -0.05 * R)),
                             plane((0, -1, 0), 0.1 * R))
            cups = [_xf(cylinder(0.26 * R, 0.16 * R, round=0.05 * R), _Rz(90), (s * head.ca[0] * 1.08, -0.08 * R, -0.04 * R))
                    for s in (1, -1)]
            add("headphones", H_([band] + cups), _mat("hard", ec, st))
        elif ex == "tie":
            knot = _xf(sphere(0.05 * H), Rc, pc + Rc @ np.array([0, 0.4 * T, cz * 0.6 + ct]))
            blade = _xf(rounded_box(0.09 * H, 0.38 * T, 0.025 * H, 0.012 * H), Rc, pc + Rc @ np.array([0, 0.17 * T, cz + ct * 1.3]))
            add("tie", [knot, blade], _mat("cloth", cols["accent"], st))
        elif ex == "bowtie":
            bw = [_xf(ellipsoid(0.08 * H, 0.05 * H, 0.03 * H), Rc @ _Rz(s * 10), pc + Rc @ np.array([s * 0.07 * H, 0.42 * T, cz * 0.62 + ct]))
                  for s in (1, -1)]
            add("tie", bw + [_xf(sphere(0.03 * H), Rc, pc + Rc @ np.array([0, 0.42 * T, cz * 0.66 + ct]))],
                _mat("cloth", cols["accent"], st))
        elif ex in ("necklace", "medal", "stethoscope"):
            ring = _xf(torus(rig.ar * 1.3 + 0.07 * H, 0.012 * H), Rc @ _Rx(-62), pc + Rc @ np.array([0, 0.36 * T, 0.18 * cz]))
            items = [ring]
            if ex == "medal":
                items.append(_xf(cylinder(0.07 * H, 0.015 * H), Rc @ _Rx(90), pc + Rc @ np.array([0, 0.12 * T, cz + ct * 1.6])))
            if ex == "stethoscope":
                items.append(_xf(cylinder(0.05 * H, 0.02 * H), Rc @ _Rx(90), pc + Rc @ np.array([0.08 * H, 0.05 * T, cz + ct * 1.6])))
            add(ex, items, _mat("metal" if ex != "necklace" else "hard", "#d8b04a" if ex == "medal" else
                                ("#9aa2ac" if ex == "stethoscope" else ec), st))
        elif ex == "belt":
            add("belt", [_band(_xf(ellipsoid(rig.hw + rig.lr * 1.05, 0.25 * T, rig.depth * 0.92), Rp,
                                   pp + Rp @ np.array([0, 0.03 * H, -0.005 * H])), Rp, pp, 0.12 * T, 0.2 * T, 0.03 * H)],
                _mat("hard", ec if ex in cols else "#3a2a22", st))
        elif ex == "apron":
            # a curved panel that wraps the front of the body: a thin shell cut to a bib + skirt outline
            th = 0.022 * H
            skirt_h = 0.3 * T + rig.thigh * 1.0
            c_sk = pp + Rp @ np.array([0, 0.12 * T, -0.15 * rig.depth])
            sk = _xf(ellipsoid(rig.hw + rig.lr * 1.45, skirt_h + 0.2 * T, rig.depth * 1.35 + ct * 2), Rp, c_sk).shell(th)
            sk = intersect(sk, _xf(plane((0, 0, -1), 0.0), Rp, pp + Rp @ np.array([0, 0, 0.35 * rig.depth])))
            sk = intersect(sk, _xf(plane((0, 1, 0), 0.0), Rp, pp + Rp @ np.array([0, 0.16 * T, 0])))
            sk = intersect(sk, _xf(plane((0, -1, 0), 0.0), Rp, pp + Rp @ np.array([0, 0.16 * T - skirt_h, 0])))
            cx = rig.cfg["cw"] * H
            bib = _front_panel(self._torso_sh, ct + 0.02 * H, th, Rs, ps, Rc, pc, -0.4 * T, 0.36 * T, 0.6 * cx, 0.25 * cz)
            strap = _xf(torus(rig.ar * 1.3 + 0.05 * H, 0.016 * H), Rn @ _Rx(-35), pn + Rn @ np.array([0, -0.02 * H, 0.03 * H]))
            pocket = _xf(rounded_box(0.13 * H, 0.08 * H, 0.012 * H, 0.01 * H), Rp,
                         pp + Rp @ np.array([0, 0.02 * T - 0.25 * rig.thigh, rig.depth * 1.2 + ct * 2 + th]))
            hips_e = _xf(ellipsoid(rig.hw + rig.lr * 0.95, 0.25 * T, rig.depth * 0.9), Rp, pp + Rp @ np.array([0, 0.03 * H, -0.005 * H]))
            tie = _band(hips_e, Rp, pp, 0.12 * T, 0.2 * T, ct + 0.03 * H)
            add("apron", [sk, bib, strap, pocket, tie], _mat("cloth", ec, st))
        elif ex == "watch":
            Rw, pw = J["wrist_l"]
            add("watch", [_xf(torus(rig.ar * 0.8, 0.025 * H), Rw, pw + Rw @ np.array([0, 0.05 * H, 0]))],
                _mat("hard", ec, st))
        elif ex == "wings":
            for s in (1, -1):
                w = _xf(ellipsoid(0.55 * H, 0.75 * H, 0.03 * H), Rc @ _Rz(s * 30) @ _Ry(s * 20),
                        pc + Rc @ np.array([s * 0.45 * H, 0.3 * T, -cz - 0.1 * H]))
                add("wings", [w], Material(color=ec, roughness=0.2, specular=0.6, glass=0.5, shadow=0.4))

    def _place_prop(self, item, hand, tilt, turn, off, J, A):
        if isinstance(item, Character):
            item = Model(item.name, item.parts)
        grip = getattr(item, "grip", None)
        lo, hi = item.bounds()
        if grip is None:
            grip = np.array([(lo[0] + hi[0]) / 2, lo[1] + (hi[1] - lo[1]) * 0.35, (lo[2] + hi[2]) / 2])
        grip = np.asarray(grip, np.float64)
        upright = getattr(item, "upright", True)
        if hand == "both":
            hp = (A["hand_l"] + A["hand_r"]) / 2
        else:
            hp = A[f"hand_{hand}"] if hand in ("l", "r") else A.get(hand, A["hand_r"])
        yaw = A.get("_yaw", 0.0) if "_yaw" in A else math.degrees(math.atan2(*(J["pelvis"][0] @ np.array([0, 0, 1.0]))[[0, 2]]))
        if upright:
            Rm = _Ry(yaw + turn) @ _Rx(tilt)
            lean = {"balloon": 14.0, "umbrella": 9.0}.get(getattr(item, "name", ""), 0.0)
            if lean and hand in ("l", "r"):
                Rm = Rm @ _Rz(-lean if hand == "l" else lean)     # string / shaft leans out and clears the face
        else:
            M, pw = A[f"_hand_{hand if hand in ('l', 'r') else 'r'}"]
            Rm = M @ _Rx(tilt) @ _Ry(turn)
        t = hp - Rm @ grip + np.asarray(off, np.float64)
        out = []
        for p in item.parts:
            out.append((f"{item.name}.{p.name}", [_xf(p.shape, Rm, t)], p.mat))
        return out


# ===================================================================================================== cast

CAST = {
    "kid": dict(build="kid", hair=dict(style="short", color="brown"), outfit=dict(top="tshirt", bottom="shorts",
                shoes="sneakers", colours=dict(top="#f2b33d", bottom="#3f6fb0", shoes="#e8613a"))),
    "student": dict(build="teen", hair=dict(style="ponytail", color="auburn"), skin=1,
                    face=dict(freckles=True, eye_color="green"),
                    outfit=dict(top="hoodie", bottom="jeans", shoes="sneakers", extras=["backpack"],
                                colours=dict(top="#7a5cc8", extra="#f2b33d"))),
    "office_worker": dict(build="adult", hair=dict(style="side_part", color="black"), skin=3,
                          face=dict(glasses=True), outfit=dict(top="shirt", bottom="trousers", shoes="shoes",
                          extras=["tie"], colours=dict(top="#eef2f6", bottom="#3a4250", shoes="#2b2420",
                                                       accent="#2f6f8f")), hold="phone"),
    "chef": dict(build="adult", hair=dict(style="crop", color="dark_brown"), skin=2,
                 face=dict(beard="moustache"), outfit=dict(top="shirt", bottom="trousers", shoes="shoes",
                 hat="chef_hat", extras=["apron"], colours=dict(top="#f6f4ef", bottom="#2f3238", shoes="#2b2420",
                                                                accent="#f6f4ef", extra="#e8613a"))),
    "doctor": dict(build="adult", hair=dict(style="bob", color="black"), skin=5, face=dict(glasses=True),
                   outfit=dict(top="lab_coat", bottom="trousers", shoes="shoes", extras=["stethoscope"],
                               colours=dict(top="#f7f7f4", bottom="#3a5a7a", shoes="#2b2420", inner="#7fc4c8"))),
    "nurse": dict(build="adult", hair=dict(style="bun", color="brown"), skin=1,
                  outfit=dict(top="tshirt", bottom="trousers", shoes="sneakers", hat="nurse_cap",
                              colours=dict(top="#5ab0b8", bottom="#5ab0b8", shoes="#f4f1ea", hat="#f8f6f2",
                                           accent="#e04a4a"))),
    "scientist": dict(build="adult", hair=dict(style="curly", color="ginger"), skin=0,
                      face=dict(glasses=True, freckles=True), outfit=dict(top="lab_coat", bottom="trousers",
                      shoes="sneakers", colours=dict(top="#f7f7f4", bottom="#4a4f5a", shoes="#3f86d8",
                                                     inner="#9cc95a")), hold="flask"),
    "farmer": dict(build="adult", hair=dict(style="short", color="chestnut"), skin=2, face=dict(beard="short"),
                   outfit=dict(top="shirt", bottom="overalls", shoes="boots", hat="straw_hat",
                               colours=dict(top="#c94f4f", bottom="#3f5f8f", shoes="#6b4a30", hat="#e2c27a",
                                            accent="#b8402a"))),
    "firefighter": dict(build="hero", hair=dict(style="crop", color="dark_brown"), skin=4,
                        outfit=dict(top="jacket", bottom="puffy", shoes="boots", hat="fire_helmet",
                                    colours=dict(top="#d8a63a", bottom="#d8a63a", shoes="#2b2b30", hat="#d8372a",
                                                 accent="#f2d24a", inner="#2b2b30"))),
    "police_officer": dict(build="adult", hair=dict(style="crop", color="black"), skin=6,
                           outfit=dict(top="shirt", bottom="trousers", shoes="shoes", hat="police_cap",
                                       extras=["belt"], colours=dict(top="#3a5a9a", bottom="#26314a", shoes="#1e1a18",
                                                                    hat="#26314a", accent="#e8c84a",
                                                                    belt="#1e1a18"))),
    "astronaut": dict(build="toy", hair=dict(style="short", color="black"), skin=3,
                      outfit=dict(top="spacesuit", bottom="puffy", shoes="boots", hat="space_helmet",
                                  extras=["backpack"], colours=dict(top="#f4f4f2", bottom="#f4f4f2", shoes="#d8dade",
                                                                    hat="#d8dade", accent="#e8613a", extra="#e4e6ea"))),
    "artist": dict(build="adult", hair=dict(style="bob", color="black"), skin=1, face=dict(eye_color="dark"),
                   outfit=dict(top="tshirt", bottom="overalls", shoes="sneakers", hat="beret",
                               colours=dict(top="#f4efe6", bottom="#e86a8e", shoes="#f2b33d", hat="#c94f4f")),
                   hold="paintbrush"),
    "musician": dict(build="adult", hair=dict(style="afro", color="black"), skin=6, face=dict(glasses="sunglasses"),
                     outfit=dict(top="jacket", bottom="jeans", shoes="boots", colours=dict(top="#2b2b30",
                                 bottom="#3f5f8f", shoes="#6b4a30", inner="#f2b33d"))),
    "athlete": dict(build="hero", hair=dict(style="buzz", color="black"), skin=5,
                    outfit=dict(top="jersey", bottom="shorts", shoes="sneakers", colours=dict(top="#e8613a",
                                bottom="#2f3a4a", shoes="#f4f1ea", accent="#f4f1ea"))),
    "runner": dict(build="adult", hair=dict(style="ponytail", color="blonde"), skin=0,
                   outfit=dict(top="tank", bottom="leggings", shoes="sneakers", extras=["headphones"],
                               colours=dict(top="#46b37a", bottom="#2f3a4a", shoes="#e86a8e", extra="#f4f1ea"))),
    "hiker": dict(build="adult", hair=dict(style="short", color="brown"), skin=2, face=dict(beard="short"),
                  outfit=dict(top="jacket", bottom="shorts", shoes="boots", hat="cap", extras=["backpack"],
                              colours=dict(top="#e8613a", bottom="#6b5a4a", shoes="#6b4a30", hat="#2f6f8f",
                                           extra="#46b37a", inner="#f2b33d"))),
    "grandma": dict(build="elder", hair=dict(style="bun", color="white"), skin=1, face=dict(glasses=True),
                    outfit=dict(top="sweater", bottom="long_skirt", shoes="shoes",
                                colours=dict(top="#c9718a", bottom="#7a6a8a", shoes="#6b4a3a"))),
    "grandpa": dict(build="elder", hair=dict(style="balding", color="grey"), skin=2,
                    face=dict(beard="moustache", glasses=True), outfit=dict(top="sweater", bottom="trousers",
                    shoes="shoes", hat="beret", colours=dict(top="#7a9a6a", bottom="#6b5a4a", shoes="#4a3428",
                                                            hat="#5a6b7a"))),
    "baby": dict(build="baby", hair=dict(style="baby_tuft", color="brown"), skin=1,
                 face=dict(expression="happy"), outfit=dict(top="onesie", bottom="puffy", shoes="barefoot",
                 colours=dict(top="#ffd166", bottom="#ffd166"))),
    "wizard": dict(build="elder", hair=dict(style="long", color="white"), skin=1,
                   face=dict(beard="santa"), outfit=dict(top="robe", bottom="none", shoes="shoes",
                   hat="wizard_hat", colours=dict(top="#4a4aa8", hat="#4a4aa8", accent="#f2c14e", shoes="#3a2a5a")),
                   hold="staff"),
    "knight": dict(build="hero", hair=dict(style="short", color="blonde"), skin=1,
                   outfit=dict(top="armor", bottom="trousers", shoes="boots", hat="knight_helmet", extras=["cape"],
                               colours=dict(top="#c9ccd2", bottom="#5a5a66", shoes="#6a6a74", hat="#c9ccd2",
                                            accent="#c83a3a", extra="#c83a3a")), hold="sword"),
    "pirate": dict(build="adult", hair=dict(style="long", color="black"), skin=3, face=dict(beard="short"),
                   outfit=dict(top="shirt", bottom="trousers", shoes="boots", hat="pirate_hat", extras=["belt"],
                               colours=dict(top="#f2ece0", bottom="#3a3a46", shoes="#3a2a20", hat="#2b2b30",
                                            accent="#f2c14e", belt="#6b3a20"))),
    "princess": dict(build="kid", hair=dict(style="long", color="blonde"), skin=0,
                     outfit=dict(top="gown", bottom="none", shoes="shoes", hat="tiara",
                                 colours=dict(top="#f29ac0", shoes="#f29ac0", accent="#7fc8f0"))),
    "superhero": dict(build="hero", hair=dict(style="quiff", color="black"), skin=2,
                      outfit=dict(top="longsleeve", bottom="leggings", shoes="boots", extras=["cape", "belt"],
                                  colours=dict(top="#2f6fd0", bottom="#2f6fd0", shoes="#d8372a", extra="#d8372a",
                                               belt="#f2c14e"))),
    "robot_friend": dict(build="toy", hair=dict(style="bald"), skin="#c9d3dc", material="plastic",
                         face=dict(head="square", eyes="dot", nose="none", ears=False, brows=False, cheeks=0.6),
                         outfit=dict(top="none", bottom="none", shoes="boots", hat="antenna",
                                     colours=dict(shoes="#8a96a4", hat="#8a96a4", accent="#ffb84a"))),
    "santa": dict(build="elder", hair=dict(style="short", color="white"), skin=1, face=dict(beard="santa"),
                  outfit=dict(top="coat", bottom="trousers", shoes="boots", hat="santa_hat", extras=["belt"],
                              colours=dict(top="#d8372a", bottom="#d8372a", shoes="#2b2b30", hat="#d8372a",
                                           accent="#f4f2ee", belt="#2b2b30"))),
    "teacher": dict(build="adult", hair=dict(style="wavy", color="chestnut"), skin=2, face=dict(glasses=True),
                    outfit=dict(top="sweater", bottom="skirt", shoes="shoes", colours=dict(top="#46b37a",
                                bottom="#3a4a6a", shoes="#6b3a3a")), hold="book"),
    "gardener": dict(build="adult", hair=dict(style="braids", color="black"), skin=6,
                     outfit=dict(top="tshirt", bottom="overalls", shoes="rain_boots", hat="sun_hat",
                                 colours=dict(top="#f2b33d", bottom="#5a8a4a", shoes="#46b37a", hat="#f4e6c8",
                                              accent="#e86a8e")), hold="flower"),
    "builder": dict(build="adult", hair=dict(style="short", color="dark_brown"), skin=4,
                    outfit=dict(top="tshirt", bottom="jeans", shoes="boots", hat="hard_hat",
                                colours=dict(top="#f08a2a", bottom="#3f5f8f", shoes="#8a5a3a", hat="#f2c14e"))),
    "dancer": dict(build="teen", hair=dict(style="top_knot", color="black"), skin=4,
                   outfit=dict(top="tank", bottom="skirt", shoes="shoes", colours=dict(top="#e86a8e",
                               bottom="#f29ac0", shoes="#f29ac0"))),
    "skater": dict(build="teen", hair=dict(style="mohawk", color="green"), skin=3,
                   outfit=dict(top="hoodie", bottom="shorts", shoes="sneakers", hat=None,
                               colours=dict(top="#2b2b30", bottom="#6b5a4a", shoes="#d8372a"))),
    "explorer": dict(build="toy", hair=dict(style="pigtails", color="ginger"), skin=0,
                     face=dict(freckles=True), outfit=dict(top="shirt", bottom="shorts", shoes="boots",
                     hat="sun_hat", extras=["bag"], colours=dict(top="#c8b07a", bottom="#7a6a4a", shoes="#6b4a30",
                                                               hat="#d8c08a", accent="#8a5a3a", extra="#8a5a3a"))),
}
_CAST_POSES = {"office_worker": "phone", "chef": "hold", "scientist": "hold", "artist": "hold", "wizard": "hold",
               "knight": "hold", "teacher": "hold", "gardener": "hold", "baby": "sit_floor", "superhero": "hands_on_hips",
               "runner": "run", "athlete": "cheer", "dancer": "dance"}


_RIGHT_HAND_POSES = ("wave", "point", "phone", "think", "throw", "reach_up", "dance")


def _norm_name(name, table):
    """'Police Officer' / 'police-officer' -> 'police_officer' when that is a registered name."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return key if key in table else name


_CHANGE_OPTIONS = tuple(n for n in _sigopts(Character.__init__) if n not in ("name", "build", "height", "detail")) + ("hold",)
_CHANGE_ALIASES = {"clothes": "outfit", "clothing": "outfit", "skin_tone": "skin", "hairstyle": "hair"}


def cast(name, pose=None, phase=0.0, build=None, height=None, detail="normal", **changes):
    """A ready-made character by name (``CAST``): ``cast("chef", pose="wave")``. Any Character argument can be
    overridden (``build="toy"``, ``skin=4``, ``hair=dict(color="ginger")``, ``outfit=dict(colours=dict(top=...))``,
    ``face=dict(expression="laugh")``); ``hold=None`` removes the signature prop."""
    name = _norm_name(name, CAST)
    if name not in CAST:
        raise _unknown("character", name, CAST, "Try clay.find('...') or clay.names('character').")
    changes = _reject("cast", changes, _CHANGE_OPTIONS + ("pose", "phase", "build", "height", "detail"), _CHANGE_ALIASES)
    for k in ("pose", "phase", "build", "height", "detail"):
        if k in changes:
            raise TypeError(f"cast: option '{k}' given twice")
    spec = _copy.deepcopy(CAST[name])
    held = changes.pop("hold", spec.pop("hold", None))
    for k in ("hair", "face", "outfit"):
        if k in changes and isinstance(changes[k], dict) and isinstance(spec.get(k), dict):
            base = spec[k]
            new = dict(changes.pop(k))
            if k == "outfit" and "colours" in new and "colours" in base:
                cc = dict(base["colours"])
                cc.update(new.pop("colours"))
                new["colours"] = cc
            base.update(new)
    spec.update(changes)
    if build:
        spec["build"] = build
    ch = Character(name=name, height=height, detail=detail, **spec)
    pose = pose or _CAST_POSES.get(name, "stand")
    ch.pose(pose, phase=phase)
    if held:
        # a gesture that needs the right hand (point, wave, phone ...) moves the signature prop to the left hand
        busy = pose in _RIGHT_HAND_POSES and not (pose == "phone" and held == "phone")
        ch.hold(held, hand="l" if busy else "r")
    return ch


def random_person(seed=0, build=None, pose="stand", phase=0.0, detail="normal", **changes):
    """A varied, harmonious person drawn from good skin / hair / outfit palettes (same seed -> same person)."""
    changes = _reject("random_person", changes, _CHANGE_OPTIONS, _CHANGE_ALIASES)
    rng = np.random.default_rng(seed)
    b = build or rng.choice(["adult", "adult", "teen", "kid", "elder", "toy"])
    skin = int(rng.integers(0, len(SKIN_TONES)))
    hc = rng.choice(["black", "dark_brown", "brown", "chestnut", "auburn", "blonde", "ginger"],
                    p=[0.24, 0.24, 0.16, 0.12, 0.08, 0.1, 0.06]) if b != "elder" else rng.choice(["grey", "white"])
    styles = ["short", "crop", "side_part", "bob", "long", "ponytail", "bun", "curly", "afro", "pixie", "wavy",
              "braids", "quiff", "top_knot", "pigtails"]
    if b == "elder":
        styles = ["balding", "bun", "short", "curly", "bob"]
    hs = rng.choice(styles)
    tops = ["tshirt", "sweater", "hoodie", "shirt", "polo", "jacket", "dress", "tank", "longsleeve"]
    top = rng.choice(tops)
    bottom = "none" if top == "dress" else rng.choice(["jeans", "trousers", "shorts", "skirt", "trousers"])
    tc = TOP_COLORS[int(rng.integers(0, len(TOP_COLORS)))]
    bc = BOTTOM_COLORS[int(rng.integers(0, len(BOTTOM_COLORS)))]
    sc = SHOE_COLORS[int(rng.integers(0, len(SHOE_COLORS)))]
    extras = []
    if rng.random() < 0.15:
        extras.append(rng.choice(["backpack", "bag", "scarf", "headphones"]))
    face = dict(expression=str(rng.choice(["smile", "smile", "happy", "neutral"])),
                eye_color=str(rng.choice(list(EYE_COLORS))), freckles=bool(rng.random() < 0.15),
                glasses=bool(rng.random() < 0.18), head=str(rng.choice(["round", "oval", "heart", "square"])))
    if b in ("adult", "elder") and rng.random() < 0.15:
        face["beard"] = str(rng.choice(["short", "moustache", "full"]))
    hat = rng.choice([None] * 6 + ["cap", "beanie", "sun_hat"])
    spec = dict(build=str(b), skin=skin, hair=dict(style=str(hs), color=str(hc)), face=face,
                outfit=dict(top=str(top), bottom=str(bottom), shoes=str(rng.choice(["sneakers", "shoes", "boots"])),
                            hat=hat, extras=extras, colours=dict(top=tc, bottom=bc, shoes=sc,
                                                                 accent=TOP_COLORS[int(rng.integers(0, len(TOP_COLORS)))])))
    spec.update(changes)
    ch = Character(name=f"person {seed}", detail=detail, seed=int(seed), **spec)
    return ch.pose(pose, phase=phase)


_strict_methods(Character, skip=("named",))        # unknown keywords: TypeError naming the closest option
