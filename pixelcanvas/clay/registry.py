# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from .._args import unknown_name as _unknown
from ..color import F32
from ..color import rgb
from .animal_rig import ANIMAL_POSES
from .tables import BUILDS, EXPRESSIONS, HAIR_STYLES, OUTFITS
from .character import CAST, _CAST_POSES
from .rig import POSES
from .props import PROPS
from .species import SPECIES


# ===================================================================================================== decals

def _layer_for(res, who, layer):
    names = list(res.layers)
    if layer:
        return layer
    try:
        first = who.parts[0].shape
        for m in res.scene.models:
            if m.parts and m.parts[0].shape is first:
                return m.name
    except Exception:
        pass
    if who.name in names:
        return who.name
    for n in names:
        if n.startswith(who.name):
            return n
    raise KeyError(f"cannot find the layer of {who.name!r}; pass layer=<the name used in sc.add>")


def decal_frame(res, who, where="chest"):
    """Where a surface frame lands in the picture: dict(center=(x, y), right=(dx, dy), up=(dx, dy), visible=bool),
    with ``right`` / ``up`` = pixel offsets of one metre along the surface. ``where`` = chest | back | belly | face."""
    p, r, u, n = who.frame(where)
    cam = res.scene.camera
    c = np.asarray(res.project(p), np.float64)
    e = 0.01
    rx = (np.asarray(res.project(p + r * e)) - c) / e
    uy = (np.asarray(res.project(p + u * e)) - c) / e
    visible = float(n @ (cam.position - p)) > 0
    return dict(center=tuple(c), right=tuple(rx), up=tuple(uy), visible=visible)


def print_on(cv, res, who, text, where="chest", color="#ffffff", size=None, font=None, bold=True, part=None,
             layer=None, log=None):
    """Print a word or number on a character's clothes after the render (t-shirt logo, jersey number). The text is
    laid onto the surface frame (perspective-correct for the part's tilt), clipped to the garment's mask and
    shaded with the render, so it stays stuck to the right place. ``size`` = letter height in metres."""
    from PIL import Image
    from ..text import text_mask, text_bbox
    fr = decal_frame(res, who, where)
    if not fr["visible"]:
        return cv
    if log:
        log(f"print {text}")
    H_ = who._anchors().get("_H", 0.3)
    size = size or H_ * 0.4
    lay = _layer_for(res, who, layer)
    names = {p.name for p in who.parts}
    pn = part or ("top" if "top" in names else "skin")
    if where == "face":
        pn = part or "skin"
    mask = res.mask(f"{lay}.{pn}")
    px = 120
    x0, y0, x1, y1 = text_bbox(text, (0, 0), px, font=font, bold=bold)
    tw, th = int(x1 - x0) + 8, int(y1 - y0) + 8
    tm = text_mask(th, tw, text, (4 - x0, 4 - y0), px, font=font, bold=bold, aa=3)
    k = size / (px * 0.72)                        # metres per patch pixel (size = cap height)
    R = np.asarray(fr["right"]) * k
    U = -np.asarray(fr["up"]) * k                 # patch y goes down
    c = np.asarray(fr["center"])
    o = c - R * tw / 2 - U * th / 2
    M = np.array([[R[0], U[0]], [R[1], U[1]]])
    try:
        Mi = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return cv
    # inverse affine for PIL: patch = Mi @ (canvas - o)
    a, b = Mi[0]
    d, e = Mi[1]
    cc = -(Mi @ o)
    img = Image.fromarray((np.clip(tm, 0, 1) * 255).astype(np.uint8))
    warped = img.transform((cv.w, cv.h), Image.AFFINE, (a, b, cc[0], d, e, cc[1]), resample=Image.BILINEAR)
    alpha = np.asarray(warped, F32) / 255.0 * mask
    if not alpha.any():
        return cv
    lum = res.color @ np.array([0.3, 0.59, 0.11], F32)
    sel = mask > 0.5
    ref = float(np.percentile(lum[sel], 80)) if sel.any() else 1.0
    shade = np.clip(lum / max(ref, 1e-3), 0.25, 1.2)[..., None]
    col = np.asarray(rgb(color), F32)[None, None] * shade
    cv.paint(col.astype(F32), alpha.astype(F32))
    return cv


# ===================================================================================================== registry

_KINDS = ("character", "animal", "prop", "pose", "animal_pose", "expression", "hair", "outfit", "build")


def _table(kind):
    return {"character": CAST, "animal": SPECIES, "prop": PROPS, "pose": POSES, "animal_pose": ANIMAL_POSES,
            "expression": EXPRESSIONS, "hair": HAIR_STYLES, "outfit": OUTFITS, "build": BUILDS}[kind]


def names(kind=None):
    """Registered names: ``names("character")`` (cast presets), ``"animal"``, ``"prop"``, ``"pose"``,
    ``"animal_pose"``, ``"expression"``, ``"hair"``, ``"outfit"``, ``"build"``; no kind = characters, animals and
    props."""
    if kind is None:
        return list(CAST) + list(SPECIES) + list(PROPS)
    if kind not in _KINDS:
        raise _unknown("kind", kind, _KINDS)
    return list(_table(kind))


def _kind_of(name):
    for k in ("character", "animal", "prop"):
        if name in _table(k):
            return k
    return None


_FIND_ALIASES = {"dog": "labrador", "puppy": "corgi", "kitten": "cat", "kitty": "cat", "bunny": "rabbit",
                 "horse": "horse", "dino": "trex", "dinosaur": "trex", "t rex": "trex", "bird": "parrot",
                 "cook": "chef", "spaceman": "astronaut", "space man": "astronaut", "space": "astronaut",
                 "puppy dog": "corgi", "kitty cat": "cat", "doggy": "labrador", "pup": "corgi", "hero": "superhero",
                 "granny": "grandma", "grandmother": "grandma", "grandpa": "grandpa", "child": "kid", "boy": "kid",
                 "girl": "kid", "student": "student", "witch": "wizard", "mage": "wizard", "robot": "robot_friend",
                 "blanket": "picnic_blanket", "bike": "bicycle", "fire": "campfire", "chair": "stool"}


def find(query):
    """Best character / animal / prop name for free text ("a little corgi" -> "corgi", "space man" ->
    "astronaut", "red balloon" -> "balloon"), or None."""
    q = " ".join(str(query).lower().replace("_", " ").replace("-", " ").split())
    pool = {n.replace("_", " "): n for n in names()}
    for a, t in _FIND_ALIASES.items():
        if _kind_of(t):
            pool.setdefault(a, t)
    if q in pool:
        return pool[q]
    words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in q.split()]
    best, score = None, 0
    for key, target in pool.items():
        kw = key.split()
        if " " + key + " " in " " + q + " ":
            sc = 10 + len(key)
        else:
            sc = 3 * sum(w in words for w in kw) - (len(kw) - sum(w in words for w in kw))
            if sc <= 0:
                continue
        if sc > score:
            best, score = target, sc
    return best


def describe(name):
    """One-paragraph description of a character, animal or prop and the call that makes it."""
    kind = _kind_of(name)
    if kind is None:
        hit = find(name)
        if hit is None:
            raise _unknown("clay name", name, names())
        name, kind = hit, _kind_of(hit)
    if kind == "character":
        c = CAST[name]
        out = c.get("outfit")
        return (f"{name}: clay character, build {c.get('build', 'toy')}, outfit "
                f"{out if isinstance(out, str) else 'custom'}; default pose {_CAST_POSES.get(name, 'stand')}. "
                f"Call: cast({name!r}, pose=...)")
    if kind == "animal":
        sp = SPECIES[name]
        fam = sp.get("family", "quad")
        return (f"{name}: clay animal ({fam}), height {sp.get('height', 0.5):g} m; poses: "
                f"{', '.join(ANIMAL_POSES)}. Call: animal({name!r}, pose='sit')")
    doc = (PROPS[name].__doc__ or "").strip().splitlines()[0] if PROPS[name].__doc__ else ""
    return f"{name}: clay prop. {doc} Call: prop({name!r})".replace("  ", " ")
