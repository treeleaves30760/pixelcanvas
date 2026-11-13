# -*- coding: utf-8 -*-
from __future__ import annotations

import copy as _copy
import math
import numpy as np

from .._args import unknown_name as _unknown, reject_unknown as _reject, strict_methods as _strict_methods
from ..solid import (Model, Part)
from . import _manifest as _manifest_labels
from .markings import _BodyPaint
from .families import _Paint, _bird, _fish, _misc
from .shapes import _Ry, _grp, _v, _xf
from .animal_rig import ANIMAL_POSES, _animal_pose
from .animal_build import _build_quad
from .species import ANIMAL_OPTIONS, _norm_species, _species
from .tables import ANIMAL_MATERIALS


_NATURAL = {}


_EXPLICIT = ["species", "pose", "phase", "size", "coat", "name", "detail", "material"]


def _check_pose(name):
    if name is not None and name not in ANIMAL_POSES:
        raise _unknown("animal pose", name, ANIMAL_POSES)


class Animal(Model):
    """A clay animal (a solid ``Model``): ``Animal("corgi", pose="sit", size=0.35, coat=None)``.

    Four-legged species share a rig (spine, neck, head, tail chain, four legs with correct joint directions) and
    poses (``ANIMAL_POSES``); birds fly with a wing phase, fish swim, insects flap. Markings (stripes, spots,
    patches, socks, masks) are 3D colour functions that follow the posed body."""

    def __init__(self, species="dog", pose="stand", phase=0.0, size=None, coat=None, color=None, name=None,
                 detail="normal", material="clay", **options):
        species = _norm_species(species)
        sp = _species(species, coat)
        options = _reject("Animal", options, _EXPLICIT + ANIMAL_OPTIONS, {"colour": "color"})
        for k in _EXPLICIT:
            if k in options and k != "color":
                raise TypeError(f"Animal: option '{k}' given twice")
        if "color" in options:
            color = options.pop("color") if color is None else color
        if material not in ANIMAL_MATERIALS:
            raise _unknown("material", material, ANIMAL_MATERIALS)
        _check_pose(pose)
        if color is not None:
            sp["fur" if sp.get("family", "quad") == "quad" else "color"] = color
        for k, v in options.items():
            if k in ("marks",) and isinstance(v, list):
                sp["marks"] = list(sp.get("marks", [])) + v
            else:
                sp[k] = v
        sp["material"] = material
        self.sp = sp
        self.species = species
        self.coat = coat
        self.name = name or species
        self.hangs = False
        self.detail = detail
        self.style = material
        self.height = float(size or sp.get("height", 0.5))
        self._pose_name, self._phase, self._over = pose, float(phase), {}
        self._W = (np.eye(3), np.zeros(3))
        self._yaw = 0.0
        self._look = None
        self._cache = None

    @property
    def parts(self):
        if self._cache is None:
            self._cache = self._assemble()
        return self._cache[0]

    def _clone(self):
        c = _copy.copy(self)
        c._over = _copy.deepcopy(self._over)
        c.sp = dict(self.sp)
        c._cache = None
        return c

    def _manifest_info(self):
        return _manifest_labels.animal_info(self.species)

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
        return self._clone()

    def pose(self, name="stand", phase=0.0, mirror=False, **joints):
        """Set a pose (``ANIMAL_POSES``; birds: stand / walk / fly / peck / sit / swim; fish: swim) with optional joint
        overrides: ``tail=dict(side=30)``, ``neck=dict(up=10)``, ``head=dict(turn=20)``, ``tongue=True``."""
        _check_pose(name)
        self._pose_name, self._phase = name, float(phase)
        self._over = dict(joints)
        self._mirror = mirror
        self._cache = None
        return self

    def face(self, direction):
        """Turn toward an angle in degrees (0 = facing +z) or a world point / (x, z)."""
        if isinstance(direction, (int, float)):
            self._yaw = float(direction)
        else:
            d = np.asarray(direction, np.float64)
            target = np.array([d[0], 0, d[1]]) if len(d) == 2 else d
            F, g = self._W
            v = target - g
            self._yaw = math.degrees(math.atan2(v[0], v[2])) - math.degrees(math.atan2(F[0, 2], F[2, 2]))
        self._cache = None
        return self

    def look_at(self, point):
        """Turn the neck and head toward a world point (four-legged animals)."""
        self._look = None if point is None else _v(point)
        self._cache = None
        return self

    def anchor(self, name):
        """World point: head, head_top, nose, mouth, back, chest, tail_tip, feet (and paw_fl, paw_hr, ...)."""
        if self._cache is None:
            self._cache = self._assemble()
        A, M, t = self._cache[1], self._cache[2], self._cache[3]
        if name not in A:
            raise KeyError(f"unknown anchor {name!r}; anchors: {sorted(A)}")
        return M @ A[name] + t

    def _natural_height(self):
        key = (self.species, str(self.coat), self.sp.get("family", "quad"))
        if key not in _NATURAL:
            parts, _ = self._build("stand", 0.0, {})
            his = [_grp(s).bounds()[1][1] for _, s, _ in parts]
            _NATURAL[key] = max(his) if his else 1.0
        return _NATURAL[key]

    def _build(self, name, ph, over):
        sp = self.sp
        fam = sp.get("family", "quad")
        if fam == "quad":
            P = _animal_pose(name, ph, sp)
            if getattr(self, "_mirror", False):
                P = {(k[:-2] + ("_r" if k.endswith("_l") else "_l")) if k[-2:] in ("_l", "_r") else k: v
                     for k, v in P.items()}
            for k, v in over.items():
                if isinstance(v, dict) and isinstance(P.get(k), dict):
                    P[k] = dict(P[k], **v)
                else:
                    P[k] = v
            return _build_quad(dict(sp), P, self.detail, self.style)
        if fam == "bird":
            return _bird(sp, name, ph, self.detail, self.style)
        if fam == "fish":
            return _fish(sp, name, ph, self.detail, self.style)
        return _misc(sp, name, ph, self.detail, self.style)

    def _assemble(self):
        over = dict(self._over)
        if self._look is not None and self.sp.get("family", "quad") == "quad":
            F, g = self._W
            s0 = self.height / self._natural_height()
            loc = np.linalg.solve(F, self._look - g) / s0
            loc = _Ry(-self._yaw) @ loc
            _, A0 = self._build("stand", 0.0, {})
            v = loc - A0["head"]
            yaw = math.degrees(math.atan2(v[0], v[2]))
            pit = math.degrees(math.atan2(v[1], math.hypot(v[0], v[2])))
            nk = dict(over.get("neck", {}))
            nk["turn"] = nk.get("turn", 0) + float(np.clip(yaw * 0.9, -70, 70))
            nk["up"] = nk.get("up", 0) + float(np.clip(pit * 0.5, -40, 40))
            over["neck"] = nk
            over["head"] = dict(over.get("head", {}), turn=float(np.clip(yaw * 0.45, -40, 40)),
                                nod=float(np.clip(-pit * 0.5, -30, 30)))
        parts, A = self._build(self._pose_name, self._phase, over)
        s = self.height / self._natural_height()
        F, g = self._W
        M = F @ (_Ry(self._yaw) * s)
        out = []
        for nm, shapes, mat in parts:
            sh = _grp(shapes)
            if sh is None:
                continue
            if isinstance(getattr(mat, "color", None), (_BodyPaint, _Paint)):
                mat.color.scale = float(abs(np.linalg.det(M)) ** (1 / 3))
            out.append(Part(nm, _xf(sh, M, g), mat))
        return out, A, M, g


def animal(species, pose="stand", phase=0.0, size=None, coat=None, color=None, **kw):
    """A clay animal by species name (``SPECIES``): ``animal("corgi", pose="sit")``, ``animal("cat", coat="calico")``,
    ``animal("parrot", pose="fly", phase=0.25)``. ``size`` = standing height in metres (default: the species' size)."""
    return Animal(species, pose=pose, phase=phase, size=size, coat=coat, color=color, **kw)


_strict_methods(Animal)
