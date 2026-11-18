"""Surface: what an area is made of and how the light falls on it, turned into marks at every scale.

The model says what a surface is (a material name, a few overrides, optionally a flow curve); the tool turns that
into a few thousand structured marks laid on the ``surface`` layer, clipped to the area, one undoable step. Nothing
here knows a subject: a material is a recipe of generic ingredients.

ingredients (each a set of marks, each mark one colour)
  form     the light at fine scale from the area's own shape (the pseudo-normal of ``draw_shade.lighting``): a core
           shadow along the terminator, bounce light in the shadow near the outline, occlusion where the area meets
           its neighbours, a rim on the lit silhouette
  tone     modulation at large and middle scale: soft airbrush dabs (soft / painted styles) or hard-edged clump
           shapes stretched along the direction field (cel style)
  fine     the finest octave: thin tapered strokes along the direction field (strands, hatching, weave, grain,
           ripples, speckle), their contrast solved so the area reaches its share of the professional 1-8 px energy
  lines    secondary contours: plane and fold lines along the terminator and along the field, lost and found
  spec     the specular response: none, a soft sheen, a band of short bright strokes (hair, satin), a hard + soft spot
colour     every mark's colour is the painted colour under it, shifted toward the light's hue where lit and toward
           its complement in shadow, jittered a little in hue and value, less saturated in large fills

direction field: a flow curve the model gives (``--flow``), the drawn hair locks, the area's long axis, the form
(along the contours of the inflated shape), horizontal, vertical or random.

Detail is spent everywhere, more at the focus: the 1-8 px energy an area is given is the professional median of the
project's profile (``detail_measure``) times a factor of its focus (3: 1.8x, 2: 1x, 1: 0.7x, background 0.4x),
minus what the area already holds (its lines and edges), measured on the delivered-size picture.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import coords
from . import draw_core as dc
from .coords import UsageError

# ============================================================================================ materials

# tone: L* amplitude of the large / middle modulation; fine: kind of finest marks, width and length in delivered px,
# density (coverage), contrast weight; dir: direction field; edge: soft | hard | lost; occl: occlusion strength;
# spec: none | sheen | band | spot; hue: shift toward the light / its complement; jit: per-mark jitter (L*, hue deg);
# desat: saturation taken out of big fills; lines: secondary contour amount; form: fine form light amount
BASE = dict(tone=(2.0, 1.5), tk="clumps", fine="hatch", fw=0.8, fl=10.0, dens=1.0, fc=1.0, kmax=8.0, lit=0.3,
            op=(0.55, 0.95), dir="axis", edge="soft", occl=0.5, spec="none", spa=0.5, hue=0.4, jit=(1.0, 3.0),
            desat=0.15, lines=0.4, form=0.7, curve=0.35)
MATERIALS: Dict[str, dict] = {
    "skin": dict(tone=(1.0, 0.0), tk="paint", fine="none", fc=0.0, dir="form", edge="soft", occl=0.6, spec="none",
                 hue=0.55, jit=(0.3, 1.0), desat=0.05, lines=0.0, form=1.0),
    "hair": dict(tone=(3.0, 3.0), fine="strands", fw=0.55, fl=90.0, dens=0.45, fc=1.0, kmax=12.0, lit=0.3,
                 op=(0.45, 0.85), dir="flow", edge="soft", occl=0.6, spec="band", spa=0.8, hue=0.4, jit=(1.0, 2.0),
                 desat=0.1, lines=0.8, form=0.8),
    "fur": dict(tone=(3.0, 2.5), fine="strands", fw=0.7, fl=16.0, dens=1.6, fc=1.3, kmax=18.0, dir="flow",
                edge="lost", occl=0.5, spec="sheen", spa=0.4, hue=0.4, jit=(2.0, 4.0), lines=0.5, form=0.8),
    "cloth-smooth": dict(tone=(3.0, 3.0), fine="none", fc=0.0, dir="drape", edge="soft", occl=0.7, spec="sheen",
                         spa=0.45, hue=0.4, lines=1.3, form=0.9, curve=0.6),
    "cloth-rough": dict(tone=(3.0, 3.0), fine="hatch", fw=0.6, fl=16.0, dens=0.6, fc=0.6, kmax=5.0, lit=0.1,
                        op=(0.25, 0.5), dir="drape", edge="soft", occl=0.7, spec="none", hue=0.35, jit=(0.8, 1.5),
                        lines=1.1, form=0.8, curve=0.7),
    "knit": dict(tone=(2.5, 2.5), fine="rib", fw=0.9, fl=8.0, dens=1.2, fc=0.9, kmax=6.0, lit=0.25, op=(0.4, 0.7),
                 dir="drape", edge="soft", occl=0.7, spec="none", hue=0.35, jit=(1.0, 2.0), lines=0.4, form=0.8),
    "leather": dict(tone=(3.0, 2.0), fine="speckle", fw=0.8, fl=2.0, dens=0.6, fc=0.5, kmax=4.0, lit=0.2,
                    dir="form", edge="hard", occl=0.6, spec="spot", spa=0.7, hue=0.3, lines=0.8, form=1.0),
    "metal": dict(tone=(5.0, 2.0), fine="strands", fw=0.6, fl=30.0, dens=0.6, fc=0.5, kmax=6.0, dir="axis",
                  edge="hard", occl=0.5, spec="band", spa=1.0, hue=0.2, jit=(0.5, 1.0), lines=0.5, form=1.2,
                  curve=0.0),
    "glass": dict(tone=(2.5, 1.5), tk="brush", bs=45.0, fine="hatch", fw=0.6, fl=60.0, dens=0.5, fc=0.8,
                  kmax=4.0, lit=0.6, op=(0.3, 0.5), dir="random", edge="hard", occl=0.3, spec="spot", spa=1.0,
                  hue=0.2, lines=0.4, form=0.8, curve=0.0),
    "foliage": dict(tone=(7.0, 5.0), tk="dabs", fine="dabs", fw=2.2, fl=5.0, dens=1.3, fc=1.8, kmax=24.0, lit=0.4,
                    dir="random", edge="lost", occl=0.4, spec="none", hue=0.5, jit=(3.0, 6.0), desat=0.1,
                    lines=0.3, form=1.0),
    "bark": dict(tone=(4.0, 3.0), fine="strands", fw=1.0, fl=22.0, dens=1.2, fc=1.6, kmax=16.0, dir="axis",
                 edge="hard", occl=0.6, spec="none", hue=0.3, jit=(2.0, 3.0), lines=0.8, form=0.9, curve=0.6),
    "stone": dict(tone=(6.0, 4.0), tk="dabs", fine="speckle", fw=1.2, fl=3.0, dens=1.0, fc=1.3, kmax=14.0,
                  dir="random", edge="hard", occl=0.5, spec="none", hue=0.3, jit=(2.5, 4.0), lines=0.7, form=1.0),
    # water is a flat plane: no form light or sheen from its outline (that lit side reads as a pale slab laid over
    # the lake); its light is the reflection of the sky and the sun's path (easel reflect)
    "water": dict(tone=(4.0, 3.0), tk="dabs", fine="ripples", fw=0.8, fl=22.0, dens=1.2, fc=1.4, kmax=16.0,
                  lit=0.45, dir="horizontal", edge="soft", occl=0.2, spec="none", spa=0.8, hue=0.5, jit=(1.5, 3.0),
                  lines=0.0, form=0.0, curve=0.1),
    "sky": dict(tone=(3.0, 2.0), tk="brush", bs=45.0, fine="hatch", fw=0.6, fl=60.0, dens=0.5, fc=1.0,
                kmax=3.5, lit=0.5, op=(0.3, 0.5), dir="horizontal", edge="soft",
                occl=0.0, spec="none", hue=0.3, jit=(1.0, 2.0), desat=0.15, lines=0.0, form=0.0, curve=0.1),
    "cloud": dict(tone=(5.0, 3.5), tk="dabs", fine="dabs", fw=3.0, fl=8.0, dens=0.8, fc=0.6, kmax=8.0, lit=0.5,
                  dir="form", edge="lost", occl=0.0, spec="sheen", spa=0.5, hue=0.5, jit=(1.5, 2.0), lines=0.0,
                  form=1.2),
    "paper": dict(tone=(3.5, 1.5), tk="brush", bs=40.0, fine="hatch", fw=0.6, fl=50.0, dens=0.8, fc=1.0,
                  kmax=5.0, lit=0.5, op=(0.3, 0.5), curve=0.0, dir="horizontal", edge="soft",
                  occl=0.0, spec="none", hue=0.2, jit=(0.6, 1.0), desat=0.1, lines=0.0, form=0.0),
    "plain": dict(tone=(2.0, 1.5), fine="hatch", fw=0.8, fl=14.0, dens=0.8, fc=0.8, kmax=6.0, lit=0.25,
                  op=(0.35, 0.7), dir="axis", edge="soft", occl=0.5, spec="none", lines=0.5, form=0.7),
}
FINE_KINDS = ("none", "strands", "hatch", "weave", "rib", "speckle", "dabs", "ripples")
DIRS = ("flow", "axis", "drape", "form", "around", "horizontal", "vertical", "random")
SPECS = ("none", "sheen", "band", "spot")
EDGES = ("soft", "hard", "lost")


def material(name: str, over: Optional[dict] = None) -> dict:
    n = str(name).strip().lower()
    if n not in MATERIALS:
        import difflib
        close = difflib.get_close_matches(n, list(MATERIALS), n=1)
        raise UsageError(f"unknown material {name!r}" + (f" (did you mean {close[0]}?)" if close else "") +
                         f"; materials: {', '.join(MATERIALS)}", "easel surface coat --material cloth-rough")
    m = dict(BASE)
    m.update(MATERIALS[n])
    m["name"] = n
    for k, v in (over or {}).items():
        if v is None:
            continue
        if k.endswith("_x"):            # a multiplier of the material's own amount (--lines 1.6, --contrast 0.5)
            m[k[:-2]] = float(m.get(k[:-2], 0.0)) * float(v)
        else:
            m[k] = v
    return m


# words in part / region / swatch names -> material (generic surface kinds, not subjects)
WORDS = [
    ("hair", ("hair", "bang", "bangs", "fringe", "ponytail", "braid", "pigtail", "lock", "locks")),
    ("fur", ("fur", "tail", "mane", "pelt", "ear", "ears", "paw", "paws", "muzzle", "whisker")),
    ("skin", ("skin", "face", "head", "neck", "arm", "arms", "hand", "hands", "leg", "legs", "cheek", "body",
              "foot", "feet", "finger", "fingers", "chest", "shoulder", "torso-skin", "nose")),
    ("knit", ("sweater", "knit", "jumper", "beanie", "mitten", "mittens", "wool", "cardigan")),
    ("cloth-smooth", ("scarf", "shirt", "dress", "blouse", "robe", "kimono", "skirt", "cape", "cloak", "sleeve",
                      "hood", "ribbon", "bow", "sash", "silk", "satin", "veil", "tie", "collar", "curtain",
                      "cushion", "pillow", "blanket", "sheet", "flag")),
    ("cloth-rough", ("coat", "jacket", "pants", "trousers", "uniform", "clothes", "torso", "hat", "denim",
                     "jeans", "vest", "cap", "glove", "gloves", "sock", "socks", "rug", "carpet", "mat")),
    ("leather", ("boot", "boots", "shoe", "shoes", "belt", "strap", "bag", "saddle", "pouch", "book", "leather")),
    ("metal", ("sword", "blade", "armor", "armour", "helmet", "metal", "steel", "iron", "gold", "silver-metal",
               "crown", "coin", "ring", "bell", "key", "chain", "buckle", "sax", "saxophone", "horn", "trumpet")),
    ("glass", ("glass", "window", "bottle", "lens", "gem", "crystal", "orb", "jewel", "pane", "vase", "jar",
               "bubble", "eye-gloss")),
    ("foliage", ("leaf", "leaves", "foliage", "bush", "bushes", "grass", "plant", "plants", "pines", "pine",
                 "forest", "hedge", "canopy", "tree", "trees", "fern", "moss", "flowers", "flower", "shrub",
                 "meadow", "field")),
    ("bark", ("trunk", "bark", "branch", "branches", "wood", "wooden", "staff", "stick", "pole", "plank",
              "log", "table", "sill", "windowsill", "shelf", "fence", "frame", "handle", "wand")),
    ("stone", ("rock", "rocks", "stone", "stones", "mountain", "mountains", "cliff", "wall", "brick", "bricks",
               "boulder", "ground", "path", "road", "sand", "shore", "beach", "hill", "hills", "ridge", "floor",
               "ruin", "ruins")),
    ("water", ("water", "lake", "sea", "river", "ocean", "pond", "reflection", "wave", "waves", "stream")),
    ("sky", ("sky", "skies", "sunset", "dusk", "dawn", "night", "horizon")),
    ("cloud", ("cloud", "clouds", "smoke", "mist", "fog", "steam")),
    ("paper", ("background", "bg", "backdrop", "paper")),
]
SKIP_WORDS = ("eye", "eyes", "iris", "pupil", "mouth", "lip", "lips", "brow", "brows", "eyebrow", "teeth", "title",
              "text", "letter", "logo", "caption", "sun", "moon", "star", "stars")


def _toks(name: str) -> List[str]:
    return [t for t in re.split(r"[-_\s]+", str(name).lower()) if t]


def guess_material(name: str, swatch: Optional[str] = None, animal: bool = False) -> Optional[str]:
    """The material a part / region name (or its colour role) suggests; None for things left alone (eyes, mouth,
    lettering, a sun disc)."""
    toks = _toks(name)
    if any(t in SKIP_WORDS or t.startswith("eye") for t in toks):
        return None
    for src in (toks, _toks(swatch or "")):
        for mat, words in WORDS:
            if any(t in words for t in src):
                if mat == "skin" and animal:
                    return "fur"
                return mat
    return "plain"


# ============================================================================================ colour (vectorised)

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i = np.linalg.inv(_M1)
_M2i = np.linalg.inv(_M2)


def to_lch(rgb) -> np.ndarray:
    """[N,3] 0..255 -> [N,3] OKLCh with L 0-100, C x400 (as draw_core.rgb_to_lch), h in degrees."""
    c = np.asarray(rgb, np.float64).reshape(-1, 3) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lab = np.cbrt(lin @ _M1.T) @ _M2.T
    L = lab[:, 0] * 100.0
    C = np.hypot(lab[:, 1], lab[:, 2]) * dc.C_SCALE
    h = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
    return np.stack([L, C, h], 1)


def from_lch(lch) -> np.ndarray:
    """[N,3] OKLCh -> [N,3] uint8 sRGB; chroma is reduced until the colour fits (lightness and hue kept)."""
    a = np.asarray(lch, np.float64).reshape(-1, 3)
    L = np.clip(a[:, 0], 0, 100) / 100.0
    C = np.maximum(a[:, 1], 0) / dc.C_SCALE
    h = np.radians(a[:, 2])
    lin = None
    for _ in range(14):
        lab = np.stack([L, C * np.cos(h), C * np.sin(h)], 1)
        lin = (lab @ _M2i.T) ** 3 @ _M1i.T
        bad = (lin.min(1) < -1e-4) | (lin.max(1) > 1 + 1e-4)
        if not bad.any():
            break
        C = np.where(bad, C * 0.85, C)
    lin = np.clip(lin, 0, 1)
    g = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(g * 255.0), 0, 255).astype(np.uint8)


def _toward(h, target, k):
    dh = ((target - h + 540.0) % 360.0) - 180.0
    return (h + k * dh) % 360.0


def light_hue(p) -> float:
    """The hue (OKLCh degrees) of the light: a 'light' / 'sun' swatch, a warm low sun for sunset requests, else a
    warm white (~80)."""
    d = dc.dstate(p)
    sw = d["palette"].get("swatches", {})
    for k in ("light", "sun", "sky-low"):
        if k in sw:
            try:
                L, C, h = dc.rgb_to_lch(dc.from_hex(sw[k]))
                if C > 6:
                    return float(h)
            except Exception:  # noqa: BLE001
                pass
    req = str(p.state.get("request") or "").lower()
    if any(w in req for w in ("sunset", "dusk", "evening", "golden")):
        return 55.0
    if any(w in req for w in ("night", "moon")):
        return 250.0
    return 80.0


def mark_colours(base_rgb, light_I, dL, mat: dict, hue_light: float, big: float, rng) -> np.ndarray:
    """Colours of N marks: the painted colour under each, lighter / darker by ``dL`` (L*), its hue pulled toward the
    light where lit and toward the complement in shadow, jittered, less saturated in big fills."""
    lch = to_lch(base_rgb)
    n = len(lch)
    I = np.asarray(light_I, np.float64).reshape(-1)
    hs = float(mat.get("hue", 0.4))
    jl, jh = mat.get("jit", (1.0, 3.0))
    L = lch[:, 0] + np.asarray(dL, np.float64).reshape(-1) + rng.normal(0, jl, n)
    C = lch[:, 1]
    h = lch[:, 2]
    chromatic = C > 4.0
    lit = np.clip((I - 0.15) / 0.6, 0, 1) * hs
    sh = np.clip((-I + 0.05) / 0.6, 0, 1) * hs
    h2 = np.where(lit > 0, _toward(h, hue_light, 0.35 * lit), h)
    h2 = np.where(sh > 0, _toward(h2, (hue_light + 180.0) % 360.0, 0.3 * sh), h2)
    # a grey takes the light's colour a little (warm light on a white shirt), not a hue swing
    C2 = np.where(chromatic, C * (1.0 + 0.12 * sh - 0.05 * lit), C + 3.0 * (lit + sh))
    h2 = np.where(chromatic, h2, np.where(lit >= sh, hue_light, (hue_light + 180.0) % 360.0))
    h2 = (h2 + rng.normal(0, jh, n) * chromatic) % 360.0
    C2 = C2 * (1.0 - float(mat.get("desat", 0.1)) * big)
    return from_lch(np.stack([L, C2, h2], 1))


# ============================================================================================ fields

def form_field(m: np.ndarray, light: np.ndarray, roundness: float = 1.0):
    """On the crop of mask ``m``: the lighting I (n . l, -1..1), the inflated height's gradient (gx, gy, pointing
    uphill, toward the middle of the form) and its magnitude g in 0..1. Same inflation as draw_shade.lighting,
    computed on a reduced copy (the form is smooth; on a 3x canvas the full-size morphology costs seconds)."""
    from scipy import ndimage as ndi
    H, W = m.shape
    size = math.sqrt(max(1.0, float(m.sum())))
    f = max(1, int(size / 250))
    if f > 1:
        sm = cv2.resize(m.astype(np.uint8), (max(2, W // f), max(2, H // f)), interpolation=cv2.INTER_AREA) > 0
    else:
        sm = m
    filled = ndi.binary_fill_holes(sm)
    sz = math.sqrt(max(1, filled.sum()))
    sig = max(2.0, 0.2 * sz)
    sub = filled.astype(np.float32)
    k = max(3, int(0.08 * sz) | 1)
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    h = cv2.GaussianBlur(sub, (0, 0), sig)
    gy, gx = np.gradient(h)
    g = np.sqrt(gx * gx + gy * gy)
    gmax = float(np.percentile(g[sub > 0], 99)) if (sub > 0).any() else 1e-6
    gmax = gmax or 1e-6
    tilt = 2.2 * roundness / gmax
    n = np.stack([-gx * tilt, -gy * tilt, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    I = (n * np.asarray(light, np.float32)[None, None, :]).sum(-1)
    g = np.clip(g / gmax, 0, 1)
    if f > 1:
        up = [cv2.resize(v.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) for v in (I, gx, gy, g)]
        I, gx, gy, g = up
        gx, gy = gx / f, gy / f
    return I.astype(np.float32), gx.astype(np.float32), gy.astype(np.float32), g.astype(np.float32)


def _unit(fx, fy):
    n = np.sqrt(fx * fx + fy * fy) + 1e-9
    return (fx / n).astype(np.float32), (fy / n).astype(np.float32)


def axis_angle(m: np.ndarray) -> float:
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return 90.0
    c = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
    w, v = np.linalg.eigh(c)
    vx, vy = v[:, int(np.argmax(w))]
    return math.degrees(math.atan2(vy, vx))


def direction_field(kind: str, m: np.ndarray, gx, gy, curves: Optional[List[np.ndarray]] = None,
                    angle: Optional[float] = None):
    """Unit field (fx, fy) on the crop for ``kind`` (see DIRS); ``curves`` (crop px) for flow; ``angle`` (degrees,
    screen: 0 right, 90 down) overrides."""
    H, W = m.shape
    if angle is not None:
        a = math.radians(angle)
        return np.full((H, W), math.cos(a), np.float32), np.full((H, W), math.sin(a), np.float32)
    if kind == "flow" and curves:
        from .draw_detail import flow_field
        return flow_field(curves, (H, W))
    if kind == "horizontal":
        return np.ones((H, W), np.float32), np.zeros((H, W), np.float32)
    if kind == "vertical":
        return np.zeros((H, W), np.float32), np.ones((H, W), np.float32)
    if kind == "form":                      # down the slope of the form (fur, skin hatching, leather grain)
        fx, fy = _unit(gx, gy)
        weak = np.sqrt(gx * gx + gy * gy) < 1e-4
        fx[weak], fy[weak] = 0.0, 1.0
        return fx, fy
    if kind == "around":                    # along the contours of the form
        fx, fy = _unit(-gy, gx)
        return fx, fy
    # axis (and flow without curves): the local long direction of the area (a scarf's wrap runs across, its
    # hanging end down), from the structure tensor of the inflated shape; a round blob takes its overall long axis
    sig = max(3.0, 0.16 * math.sqrt(max(1.0, float(m.sum()))))
    f = max(1, int(sig / 4))
    sx, sy = (cv2.resize(v, (max(2, W // f), max(2, H // f)), interpolation=cv2.INTER_AREA) for v in (gx, gy))
    gxx, gxy, gyy = (cv2.GaussianBlur(v, (0, 0), max(1.0, sig / f)) for v in (sx * sx, sx * sy, sy * sy))
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)            # the dominant gradient direction (across the band)
    lam = np.sqrt((gxx - gyy) ** 2 + 4 * gxy * gxy)
    coh = lam / (gxx + gyy + 1e-12)
    lx, ly = -np.sin(theta), np.cos(theta)                   # along the band
    if f > 1:
        lx, ly, coh = (cv2.resize(v.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) for v in (lx, ly, coh))
    if kind == "drape":                     # cloth hangs: where the area has no long direction of its own, down
        ax, ay = 0.0, 1.0
    else:
        a = math.radians(axis_angle(m))
        ax, ay = math.cos(a), math.sin(a)
        if ay < 0:
            ax, ay = -ax, -ay
    s_ = np.where(lx * ax + ly * ay < 0, -1.0, 1.0)
    k = np.clip((coh - 0.35) / 0.5, 0, 1)
    fx, fy = _unit(k * s_ * lx + (1 - k) * ax, k * s_ * ly + (1 - k) * ay)
    return fx, fy


BATCH_FINE = 300


def batch_strokes(geo, cols, shape, mask_op, s: float, groups: int = 12) -> list:
    """Many fine strokes as a few marks: the strokes are grouped by colour (k-means, ``groups`` colours), each
    group's strokes are rasterised with their tapered widths into one mask and laid as one fill of that colour.
    Same look at the delivered size (the strokes are 1-3 working px, the export reduces 3x), a fraction of the
    time of one stroke op each."""
    cols = np.asarray(cols, np.float32).reshape(-1, 3)
    k = int(min(groups, len(cols)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, lab, cen = cv2.kmeans(cols, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    out = []
    H, W = shape
    for g in range(k):
        idx = np.flatnonzero(lab == g)
        if not len(idx):
            continue
        mm = np.zeros((H, W), np.uint8)
        ops_ = []
        for i in idx:
            q, wd, _, op = geo[i]
            q = np.asarray(q, np.float64)
            wd = np.broadcast_to(np.asarray(wd, np.float64), (len(q),))
            pts = np.round(q * 4).astype(np.int32)
            for j in range(len(q) - 1):
                t_ = max(1, int(round(0.5 * (wd[j] + wd[j + 1]))))
                cv2.line(mm, tuple(pts[j]), tuple(pts[j + 1]), 255, t_, cv2.LINE_8, 2)
            ops_.append(op)
        out.append(mask_op(mm > 0, cen[g], float(np.mean(ops_)), 0.75))
    return out


def smooth_noise(shape, sigma: float, rng) -> np.ndarray:
    """Gaussian-smoothed white noise with unit std, made on a reduced grid (cheap at any sigma)."""
    H, W = shape
    f = max(1, int(sigma / 3))
    h, w = max(2, H // f + 2), max(2, W // f + 2)
    z = rng.standard_normal((h, w)).astype(np.float32)
    z = cv2.GaussianBlur(z, (0, 0), max(0.5, sigma / f))
    z = cv2.resize(z, (W, H), interpolation=cv2.INTER_CUBIC) if f > 1 else z[:H, :W]
    z -= z.mean()
    return z / (z.std() + 1e-6)


def streak_noise(fx, fy, m, length: float, sigma_x: float, rng) -> np.ndarray:
    """Noise stretched along the field (line integral convolution): clumps and streaks that follow the form."""
    H, W = m.shape
    base = smooth_noise((H, W), max(0.8, sigma_x), rng)
    steps = 10
    st = max(0.5, length / (2 * steps))
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    acc = base.copy()
    for sign in (1.0, -1.0):
        px, py = X.copy(), Y.copy()
        for _ in range(steps):
            vx = cv2.remap(fx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            vy = cv2.remap(fy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            px += sign * st * vx
            py += sign * st * vy
            acc += cv2.remap(base, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    acc -= acc[m].mean() if m.any() else acc.mean()
    return acc / (acc[m].std() + 1e-6 if m.any() else acc.std() + 1e-6)


def trace(fx, fy, m, seeds: np.ndarray, length: np.ndarray, step: float, curve: float = 1.0) -> List[np.ndarray]:
    """Streamlines through the field from each seed, half the length each way, inside ``m``. ``curve`` 0 = straight
    along the seed's direction, 1 = follow the field everywhere."""
    H, W = m.shape
    n = len(seeds)
    if n == 0:
        return []
    k = int(np.ceil(float(np.max(length)) / (2 * step))) + 1
    out_f = np.full((n, k, 2), np.nan, np.float32)
    out_b = np.full((n, k, 2), np.nan, np.float32)
    half = np.asarray(length, np.float32) / 2

    def go(sign, out):
        pos = seeds.astype(np.float32).copy()
        xi = np.clip(pos[:, 0].astype(int), 0, W - 1)
        yi = np.clip(pos[:, 1].astype(int), 0, H - 1)
        d0 = np.stack([fx[yi, xi], fy[yi, xi]], 1) * sign
        prev = d0.copy()
        alive = np.ones(n, bool)
        L = np.zeros(n, np.float32)
        for j in range(k):
            xi = np.clip(pos[:, 0].astype(int), 0, W - 1)
            yi = np.clip(pos[:, 1].astype(int), 0, H - 1)
            inside = m[yi, xi] & (pos[:, 0] >= 0) & (pos[:, 0] < W) & (pos[:, 1] >= 0) & (pos[:, 1] < H)
            alive &= inside & (L <= half)
            out[alive, j] = pos[alive]
            v = np.stack([fx[yi, xi], fy[yi, xi]], 1)
            flip = (v * prev).sum(1) < 0
            v[flip] *= -1
            v = curve * v + (1 - curve) * d0
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            prev = v
            pos = pos + v * step
            L += step
    go(1.0, out_f)
    go(-1.0, out_b)
    lines = []
    for i in range(n):
        f = out_f[i][~np.isnan(out_f[i, :, 0])]
        b = out_b[i][~np.isnan(out_b[i, :, 0])][1:][::-1]
        pts = np.vstack([b, f]) if len(b) else f
        lines.append(pts)
    return lines


def pressure(pts: np.ndarray, width: float, rng, swell: float = 0.25, tip: float = 0.2) -> np.ndarray:
    """A pen's pressure along a mark: tapered ends, a slow swell in the body; never below 1 px."""
    n = len(pts)
    if n < 2:
        return np.full(max(n, 1), max(1.0, width), np.float32)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    L = max(s[-1], 1e-6)
    t = s / L
    a, b = rng.uniform(0.15, 0.4), rng.uniform(0.2, 0.45)
    f = np.minimum(np.clip(t / a, 0, 1), np.clip((1 - t) / b, 0, 1))
    f = tip + (1 - tip) * f * f * (3 - 2 * f)
    f *= 1.0 + swell * np.sin(math.pi * t * rng.uniform(0.7, 1.6) + rng.uniform(0, 6.28))
    return np.maximum(1.0, width * f).astype(np.float32)


# ============================================================================================ the pass

FOCUS_SHARE = {3: 1.8, 2: 1.0, 1: 0.7, 0: 0.06}
FINE_KEYS = ("bandL_0", "bandL_1", "bandL_2")


def fine_target(group: str, focus: int, background: bool) -> float:
    """1-8 px luminance energy (mean square of L*/100) an area is given: the profile's median times its focus share."""
    from .detail_measure import load_profiles
    ms = load_profiles()["groups"][group]["measures"]
    e = sum(float(ms[k]["median"]) for k in FINE_KEYS)
    return e * (FOCUS_SHARE[0] if background else FOCUS_SHARE.get(int(focus), 1.0))


def style_mode(p) -> str:
    """cel: crisp marks, flats stay flat between them; soft: graded tone and crisp fine marks; paint: brush marks."""
    st = (p.state.get("style") or "paint")
    req = str(p.state.get("request") or "").lower()
    if st in ("paint", "photo"):
        return "paint"
    if st == "ink":
        return "cel"
    if re.search(r"\b(soft|painterly|painted|watercolou?r|gradient)\b", req):
        return "soft"
    return "cel"


def _fine_energy(off: np.ndarray, m: np.ndarray, s: int) -> float:
    """1-8 px energy (delivered scale) of an L* offset image over the mask."""
    from .detail_measure import _band_split
    if s > 1:
        Hd, Wd = max(2, off.shape[0] // s), max(2, off.shape[1] // s)
        o = cv2.resize(off, (Wd, Hd), interpolation=cv2.INTER_AREA)
        md = cv2.resize(m.astype(np.float32), (Wd, Hd), interpolation=cv2.INTER_AREA) > 0.5
    else:
        o, md = off, m
    if md.sum() < 16:
        return 0.0
    b = _band_split((o / 100.0).astype(np.float32))
    e = (b[0].astype(np.float64) ** 2 + b[1] ** 2 + b[2] ** 2)
    return float(e[md].mean())


def build(p, name: str, mask: np.ndarray, mat: dict, *, focus: int = 2, light: np.ndarray, base_img: np.ndarray,
          others: np.ndarray, curves_u: Optional[List] = None, angle: Optional[float] = None, amount: float = 1.0,
          scale: float = 1.0, have_fine: float = 0.0, group: str = "anime", background: bool = False,
          seed: int = 1, flags: Optional[dict] = None, line_alpha: Optional[np.ndarray] = None) -> Tuple[list, dict]:
    """The surface marks of one area (working px). Returns (ops, info)."""
    from .ops import Fill, Spray, Stroke, mask_to_rle
    flags = flags or {}
    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    s = dc.work_scale(p)
    mode = style_mode(p)
    info = {"material": mat["name"], "marks": 0}
    full = np.asarray(mask, bool)
    if full.sum() < 64 * s * s:
        return [], info
    ys, xs = np.nonzero(full)
    pad = int(4 * s) + 2
    y0, y1 = max(0, ys.min() - pad), min(p.H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(p.W, xs.max() + 1 + pad)
    m = full[y0:y1, x0:x1]
    Hc, Wc = m.shape
    area = float(m.sum())
    size = math.sqrt(area)
    big = float(np.clip(area / (p.W * p.H) / 0.12, 0, 1))
    img = base_img[y0:y1, x0:x1].astype(np.float32)
    I, gx, gy, g = form_field(m, light, 1.0)
    curves = None
    if curves_u:
        curves = [dc.u2px(p, c) - [x0, y0] for c in curves_u]
    dkind = mat["dir"] if not (mat["dir"] == "flow" and not curves) else "axis"
    fx, fy = direction_field(dkind, m, gx, gy, curves, angle)
    hue_l = light_hue(p)
    ops: list = []
    erode_in = dc.erode(m, 1.5 * s)
    oth = others[y0:y1, x0:x1] & ~m

    def sample(pts):
        q = np.asarray(pts, np.float64).reshape(-1, 2)
        xi = np.clip(q[:, 0].astype(int), 0, Wc - 1)
        yi = np.clip(q[:, 1].astype(int), 0, Hc - 1)
        return img[yi, xi], I[yi, xi]

    def to_full(pts):
        return (np.asarray(pts, np.float32) + np.float32([x0, y0])).astype(np.float32)

    def mask_op(mm, rgb, opacity, feather):
        full_m = np.zeros((p.H, p.W), bool)
        full_m[y0:y1, x0:x1] = mm
        return Fill(mask_rle=mask_to_rle(full_m), color=tuple(int(c) for c in rgb), opacity=float(opacity),
                    feather=max(1.0, float(feather)))

    def region_colour(mm):
        if not mm.any():
            return np.array([128, 128, 128], float)
        return np.median(img[mm], axis=0)

    crisp = mode == "cel"
    # anime / cel: the fills stay flat. Form is a few clean cel-shadow shapes from one light, hair light is
    # grouped lock shapes, folds get a few accent lines; no strands, no grain, no backdrop strokes
    cel = mode in ("cel", "soft") and group.startswith("anime")
    if cel:
        crisp = True
        mat = dict(mat, fine="none", fc=0.0, lines=min(0.6, float(mat.get("lines", 0.4))))
        if mat.get("tk") in ("brush", "dabs", "paint"):
            mat["tk"] = "none"
    fe = (1.0 if crisp else 2.5) * s          # feather of form shapes (working px)

    # ---------------------------------------------------------------- form light at fine scale
    fw = float(mat.get("form", 0.7)) * (0 if flags.get("no_form") else 1)
    if fw > 0 and area > 400 * s * s:
        lo = float(np.quantile(I[m], 0.25))
        core = m & (I < lo + 0.12) & (I > lo - 0.10) & erode_in
        if core.sum() > 30 * s * s:
            c = region_colour(core)
            col = dc.shadow_of(tuple(int(v) for v in c), 0.45 * fw)
            ops.append(mask_op(core, col, 0.45 if crisp else 0.35, fe * (1.5 if crisp else 3.0)))
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        dark = I < float(np.quantile(I[m], 0.3))
        bounce = m & dark & (dist < 0.1 * size) & (dist > 1.5 * s) & ~oth
        if bounce.sum() > 30 * s * s:
            c = region_colour(bounce)
            env = np.array(dc.light_of(tuple(int(v) for v in c), 0.7), float)
            ops.append(mask_op(bounce, env, 0.28 * fw, fe * 3.0))
        if oth.any() and mat.get("occl", 0) > 0:
            near = dc.dilate(oth, 0.035 * size) & m & ~dc.dilate(oth, 0.5 * s)
            if near.sum() > 30 * s * s:
                c = region_colour(near)
                ops.append(mask_op(near, dc.shadow_of(tuple(int(v) for v in c), 0.8), 0.35 * mat["occl"],
                                   0.02 * size + fe))
        # overlaps: a line inside the area (a jaw over the neck, a fold, a lock over the hair behind it) casts a
        # soft shadow on its far side from the light
        if line_alpha is not None:
            la = line_alpha[y0:y1, x0:x1] > 0.4
            inner_l = la & dc.erode(m, 2.5 * s)
            nlab, lab, st_, _ = cv2.connectedComponentsWithStats(inner_l.astype(np.uint8), 8)
            keep = [k for k in range(1, nlab) if max(st_[k, 2], st_[k, 3]) > 0.15 * size]
            if keep:
                from scipy import ndimage as ndi
                src = np.isin(lab, keep)
                dist, (iy, ix) = ndi.distance_transform_edt(~src, return_indices=True)
                cen = np.zeros((nlab, 2))
                ext = np.zeros(nlab)
                for k_ in keep:
                    yy_, xx_ = np.nonzero(lab == k_)
                    cen[k_] = (xx_.mean(), yy_.mean())
                    ext[k_] = max(st_[k_, 2], st_[k_, 3])
                lv = np.array([light[0], light[1]], float)
                lv = lv / (np.linalg.norm(lv) + 1e-9) if np.linalg.norm(lv) > 0.1 else np.array([0.0, -1.0])
                kid = lab[iy, ix]
                reach = np.minimum(0.12 * ext[kid], 0.09 * size)
                band = (dist > 0.5 * s) & (dist < reach) & m & ~dc.dilate(la, 0.5 * s)
                Y_, X_ = np.mgrid[0:Hc, 0:Wc]
                vx, vy = X_ - ix, Y_ - iy
                tx, ty = cen[kid, 0] - ix, cen[kid, 1] - iy
                tn = np.hypot(tx, ty) + 1e-6
                vn = np.hypot(vx, vy) + 1e-6
                # the curve bulges over what lies behind it: the shadow falls on its outer (convex) side; a nearly
                # straight line casts away from the light
                curved = tn > 0.12 * ext[kid]
                # a straight line inside an area is a seam or a crease, not an overlap: it casts nothing
                score = np.where(curved, -(vx * tx + vy * ty) / (tn * vn), -1.0)
                cast = band & (score > 0.25)
                cast = dc.clean_mask(cast, int(40 * s * s))
                if cast.sum() > 40 * s * s:
                    c_ = region_colour(cast)
                    ops.append(mask_op(cast, dc.shadow_of(tuple(int(v) for v in c_), 0.75), 0.55 * fw,
                                       fe * (2.0 if crisp else 4.0) + 0.01 * size))
        from .draw_shade import rim_mask
        rim = rim_mask(m, light, 1.4 * s, False, open_side=~(others[y0:y1, x0:x1] | m))
        if rim.sum() > 20 * s * s:
            c = region_colour(rim)
            ops.append(mask_op(rim, dc.light_of(tuple(int(v) for v in c), 1.2), 0.55 * fw, 0.7 * s))

    # ---------------------------------------------------------------- tone: large and middle modulation
    t_large, t_mid = (float(v) * amount for v in mat.get("tone", (2.0, 1.5)))
    if (t_large > 0 or t_mid > 0) and not flags.get("no_tone"):
        nl = smooth_noise((Hc, Wc), max(4.0, 0.22 * size), rng)
        if mat.get("tk") == "clumps" and t_mid > 0:
            # value shapes along the field at two scales, drawn as smooth tapered shapes (a fold's valley, the
            # shadow between two locks), each with a lit ridge beside it on the light side; flats stay flat between
            # cel and soft cel: the shapes keep a 1-px edge at the delivered size (that edge is the fine band a
            # professional's picture has; a texture would only be noise)
            soft_e = (0.8 if crisp else 1.0 if mode == "soft" else 3.5) * s
            hairlike = mat.get("fine") == "strands"
            lv2 = np.array([light[0], light[1]], float)
            lv2 = lv2 / (np.linalg.norm(lv2) + 1e-9) if np.linalg.norm(lv2) > 0.1 else np.array([0.0, -1.0])
            folds = mat["name"].startswith(("cloth", "knit", "leather"))
            scales = (((1.0, 0.7, 0.26, 0.14), (0.5, 1.0, 0.17, 0.1)) if folds else
                      ((1.0, 1.0, 0.26, 0.14), (0.45, 2.2, 0.17, 0.1), (0.2, 4.0, 0.12, 0.08)))
            if cel:     # a few big shapes only: two or three lock shadows / fold valleys per area
                scales = ((1.0, 0.5, 0.24, 0.12),) if folds else ((1.0, 0.6, 0.24, 0.12), (0.45, 0.8, 0.15, 0.08))
            # a fold starts where the cloth is held or bends (an edge, a seam) and runs out along the drape
            ring = m & ~dc.erode(m, max(3.0 * s, 0.06 * size)) if folds else erode_in
            for sc, n_k, dk, lk in scales:
                n_sh = int(n_k * (2 + size / (110.0 * s)) * min(1.6, max(0.5, amount)))
                cand = np.flatnonzero((ring & erode_in).ravel()) if folds else np.flatnonzero(erode_in.ravel())
                if n_sh <= 0 or not len(cand):
                    continue
                pick = rng.choice(cand, size=min(n_sh, len(cand)), replace=False)
                py, px = np.divmod(pick, Wc)
                seeds = np.stack([px + 0.5, py + 0.5], 1)
                lens = rng.uniform(0.25, 0.55, len(seeds)) * size * sc * scale * (1.6 if folds else 1.0)
                for q in trace(fx, fy, m, seeds, lens, 1.5 * s, 0.9):
                    if len(q) < 8:
                        continue
                    q = dc.smooth_poly(q, 3.0)
                    L_ = dc.path_length(q)
                    if L_ < 0.08 * size * sc:
                        continue
                    t_ = np.linspace(0, 1, len(q))
                    wmax = rng.uniform(0.035, 0.07) * size * sc * scale
                    prof = ((1 - t_) ** 0.9 * np.clip(t_ * 6, 0, 1) ** 0.5) if folds else                         (np.sin(np.pi * t_) ** 0.8) if not hairlike else ((1 - t_) ** 0.6 * np.sin(np.pi * np.clip(t_ * 1.6, 0, 1)) ** 0.5)
                    if (hairlike or folds) and rng.random() < 0.5:
                        prof = prof[::-1]
                    w_ = np.maximum(0.5 * s, wmax * prof)
                    tg = np.gradient(q, axis=0)
                    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
                    nr = np.stack([-tg[:, 1], tg[:, 0]], 1)
                    poly = np.vstack([q + nr * w_[:, None] / 2, (q - nr * w_[:, None] / 2)[::-1]])
                    mid = q[len(q) // 2]
                    b, _ = sample([mid])
                    shade_c = dc.shadow_of(tuple(int(v) for v in b[0]), dk * t_mid)
                    ops.append(Fill(polygons=[(poly + [x0, y0]).astype(np.float32)], color=shade_c,
                                    opacity=0.85 if crisp else 0.65, feather=max(1.0, soft_e)))
                    # the ridge: the side of the shape toward the light catches it
                    side_ = 1.0 if float((nr.mean(0) * lv2).sum()) > 0 else -1.0
                    rq = q + side_ * nr * (w_[:, None] / 2 + 0.6 * w_[:, None])
                    rw = 0.45 * w_
                    rpoly = np.vstack([rq + nr * rw[:, None] / 2, (rq - nr * rw[:, None] / 2)[::-1]])
                    ops.append(Fill(polygons=[(rpoly + [x0, y0]).astype(np.float32)],
                                    color=dc.light_of(tuple(int(v) for v in b[0]), lk * t_mid),
                                    opacity=0.6 if crisp else 0.45, feather=max(1.0, soft_e)))
        if cel and mat["name"] in ("paper", "sky"):
            # a cel backdrop: one soft value turn across it (lighter toward the light, a touch deeper away), no
            # strokes: graded, not flat, and calm behind the figure
            base = tuple(int(v) for v in region_colour(m))
            lv2 = np.array([light[0], light[1]], float)
            lv2 = lv2 / (np.linalg.norm(lv2) + 1e-9) if np.linalg.norm(lv2) > 0.1 else np.array([-0.6, -0.8])
            cy, cx = Hc / 2.0, Wc / 2.0
            for sgn, col, op_ in ((1.0, dc.light_of(base, 0.1), 0.8), (-1.0, dc.shadow_of(base, 0.07), 0.7)):
                ex, ey = cx + sgn * lv2[0] * 0.55 * Wc, cy + sgn * lv2[1] * 0.55 * Hc
                yy, xx = np.mgrid[0:Hc, 0:Wc]
                blob = m & (((xx - ex) / (0.6 * Wc)) ** 2 + ((yy - ey) / (0.6 * Hc)) ** 2 < 1.0)
                if blob.sum() > 100:
                    ops.append(mask_op(blob, col, op_, 0.22 * size))
            if group != "anime-cel":
                # soft cel / painted anime: a few large airbrushed light spots (out-of-focus light), so the
                # backdrop is graded the way a professional's is, never streaked
                nd = int(np.clip(area / (0.06 * size) ** 2 * 0.5 * min(2.0, max(0.5, amount)), 20, 400))
                cand = np.flatnonzero(m.ravel())
                pick = rng.choice(cand, size=min(nd, len(cand)), replace=False)
                py, px = np.divmod(pick, Wc)
                pts = np.stack([px + 0.5, py + 0.5], 1).astype(np.float32)
                light_side = (pts[:, 0] - cx) * lv2[0] + (pts[:, 1] - cy) * lv2[1] > 0
                cols = np.array([dc.light_of(base, rng.uniform(0.12, 0.3)) if ls else dc.shadow_of(base, rng.uniform(0.04, 0.1))
                                 for ls in light_side], np.uint8)
                sizes = (rng.uniform(0.02, 0.08, len(pts)) * size).astype(np.float32)
                ops.append(Spray(pts=to_full(pts), size=np.maximum(3.0, sizes), colors=cols,
                                 opacity=rng.uniform(0.3, 0.6, len(pts)).astype(np.float32), seed=int(seed) + 3))
        if mat.get("tk") == "brush":
            # a painted backdrop: broad bristle strokes in neighbouring values, their streaks carry the fine band
            from .draw_detail import brush_ops
            base = tuple(int(v) for v in region_colour(m))
            bs = float(mat.get("bs", 40.0)) * s * scale
            ang = rng.uniform(-25, 25) if mat["dir"] == "random" else 0.0
            a_ = math.radians(ang)
            fld = (np.full((Hc, Wc), math.cos(a_), np.float32), np.full((Hc, Wc), math.sin(a_), np.float32))
            sh_ = dc.shadow_of(base, 0.07 * min(2.0, t_large))
            li_ = dc.light_of(base, 0.07 * min(2.0, t_large))
            if group == "anime-cel":
                sh_, li_ = dc.shadow_of(base, 0.05 * t_large), dc.light_of(base, 0.04 * t_large)
            bops = brush_ops(m, fld, base, sh_, li_, light, bs, "dry" if mode != "cel" else "flat",
                             0.9 * min(1.2, max(0.5, amount)), seed=int(seed) + 5, lighting=I, opacity=(0.35, 0.6),
                             bend=0.05)
            for o in bops:
                o.pts = (np.asarray(o.pts, np.float32) + np.float32([x0, y0])).astype(np.float32)
            ops += bops
        if mode == "paint" and mat.get("tk") != "brush" and mat["name"] not in ("skin", "glass"):
            # painterly: the area is laid in strokes of a loaded brush at two sizes along its field, each picking
            # up the colour under it; value from the form (lit side lighter, warmer), the strokes' bristle streaks
            # and edges give every band from 64 px down to 2 px
            from .draw_detail import brush_ops
            base = tuple(int(v) for v in region_colour(m))
            under = np.clip(img, 0, 255).astype(np.uint8)
            fld = (fx.astype(np.float32), fy.astype(np.float32))
            # the forms stage laid the big strokes already: here the smaller strokes that turn the planes
            for k_, (fr, lo_, hi_, dk_, op_) in ((1, (0.04, 6, 28, 0.16, (0.45, 0.7))),):
                bs = float(np.clip(fr * size / s, lo_, hi_)) * s * scale
                sh_ = dc.shadow_of(base, dk_ * min(2.0, max(1.0, t_large)))
                li_ = dc.light_of(base, 0.7 * dk_ * min(2.0, max(1.0, t_large)))
                bops = brush_ops(m, fld, base, sh_, li_, light, bs, "dry" if k_ else "flat",
                                 0.45 * min(2.0, max(0.5, amount)), seed=int(seed) + 11 + k_,
                                 lighting=I, under=under, pickup=(0.1, 0.2), opacity=op_, bend=0.08)
                for o in bops:
                    o.pts = (np.asarray(o.pts, np.float32) + np.float32([x0, y0])).astype(np.float32)
                ops += bops
        elif mat.get("tk") == "dabs" and mat.get("tk") != "brush":
            amp = 1.0 if mat.get("tk") in ("dabs", "paint") else 0.5
            nm = streak_noise(fx, fy, m, 8.0 * s * scale, 3.0 * s * scale, rng) if mat["dir"] != "random"                 else smooth_noise((Hc, Wc), 6.0 * s * scale, rng)
            sz = float(np.clip(0.06 * size, 6 * s, 26 * s)) * scale
            nd = int(min(3000, 0.8 * area / (sz * sz)))
            if nd > 0:
                cand = np.flatnonzero(erode_in.ravel()) if erode_in.sum() > 100 else np.flatnonzero(m.ravel())
                pick = rng.choice(cand, size=min(nd, len(cand)), replace=False)
                py, px = np.divmod(pick, Wc)
                pts = np.stack([px + rng.uniform(0, 1, len(px)), py + rng.uniform(0, 1, len(px))], 1)
                b, Ii = sample(pts)
                dL = amp * (t_large * nl[py, px] + t_mid * nm[py, px])
                cols = mark_colours(b, Ii, dL, mat, hue_l, big, rng)
                sizes = sz * rng.uniform(0.6, 1.4, len(pts))
                ops.append(Spray(pts=to_full(pts), size=np.maximum(3.0, sizes), colors=cols,
                                 opacity=rng.uniform(0.35, 0.6, len(pts)).astype(np.float32), seed=int(seed)))

    # ---------------------------------------------------------------- fine marks along the field
    kind = mat.get("fine", "hatch")
    fine_ops: list = []
    fine_geo: List[Tuple[np.ndarray, np.ndarray, float, float]] = []    # (pts crop, widths, dL unit, opacity)
    dabs_geo: List[Tuple[np.ndarray, float, float, float]] = []         # (centre crop, size, dL unit, opacity)
    o_lo, o_hi = mat.get("op", (0.55, 0.95))
    lit_side = float(mat.get("lit", 0.3))
    if mode == "paint" and (kind in ("speckle", "dabs") or mat["name"] not in ("hair", "fur", "water")):
        kind = "none"       # painterly: the brush strokes carry the texture; fine marks only for strands and ripples
    if mode == "paint":
        # a budget: strands and ripples at the material's own amount, never piled on by a raised --amount; no
        # scratch lines on things that have no folds or strands
        mat = dict(mat, kmax=min(8.0, float(mat.get("kmax", 8.0))), dens=float(mat.get("dens", 1.0)) / max(1.0, amount))
        if not mat["name"].startswith(("hair", "fur", "cloth", "knit")):
            mat["lines"] = 0.0
    # where marks may go: in a painterly picture texture describes planes, so it gathers at the area's edges and
    # where the form turns (its light changes), thinning in the flat middle; elsewhere anywhere in the area
    if mode == "paint":
        dist_e = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        gI = np.hypot(*np.gradient(cv2.GaussianBlur(I.astype(np.float32), (0, 0), max(1.0, 0.02 * size))))
        wgt = np.exp(-dist_e / max(1.0, 0.12 * size)) + gI / (float(np.quantile(gI[m], 0.9)) + 1e-6)
        wgt = np.where(m, np.clip(wgt, 0.05, 2.0), 0.0).ravel()
        wgt /= wgt.sum() + 1e-12
    else:
        wgt = None

    def pick_seeds(n_):
        cand_ = np.flatnonzero(m.ravel())
        n_ = min(n_, len(cand_))
        if wgt is None:
            return rng.choice(cand_, size=n_, replace=False)
        return rng.choice(len(wgt), size=n_, replace=False, p=wgt)
    if kind == "speckle" and mat.get("fc", 1.0) > 0 and not flags.get("no_fine"):
        # grain, pores, paper tooth: airbrush dabs a delivered pixel or two across (one Spray holds them all)
        sz = max(3.0, float(mat["fw"]) * s * scale * 1.6)
        dens = float(mat["dens"]) * min(1.6, max(0.4, amount))
        n = int(min(80000, dens * area / (sz * sz * 4.0)))
        if n > 0:
            cand = np.flatnonzero(m.ravel())
            pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
            py, px = np.divmod(pick, Wc)
            pts = np.stack([px + rng.uniform(0, 1, len(px)), py + rng.uniform(0, 1, len(px))], 1)
            sg = np.where(rng.random(len(pts)) < lit_side, 0.7, -1.0)
            u = sg * np.abs(rng.normal(0.9, 0.45, len(pts)))
            for q, uu in zip(pts, u):
                dabs_geo.append((q, float(sz * rng.uniform(0.8, 1.25)), float(uu), float(rng.uniform(o_lo, o_hi))))
    elif kind != "none" and mat.get("fc", 1.0) > 0 and not flags.get("no_fine"):
        w = max(1.0, float(mat["fw"]) * s * scale)
        Lm = float(mat["fl"]) * s * scale
        quiet = float(mat.get("kmax", 8.0)) <= (8.0 if mode == "paint" else 5.0)
        am_t = min(1.2, amount) if quiet else amount
        dens = float(mat["dens"]) * min(2.2, max(0.4, am_t))
        per = max(Lm, w) * w * 2.2                              # area one mark covers with its spacing
        n = int(dens * area / per)
        n = min(n, int(9000 * min(1.5, max(0.5, amount))))
        if mode == "paint":
            n = int(n * 0.45)       # a budget: the marks gather where they describe something
        if n > 0:
            pick = pick_seeds(n)
            py, px = np.divmod(pick, Wc)
            seeds = np.stack([px + rng.uniform(0, 1, len(px)), py + rng.uniform(0, 1, len(px))], 1)
            if mat["dir"] == "horizontal":
                # receding: marks near the top of the area (far away) are shorter and thinner
                Lm_y = Lm * (0.35 + 0.65 * (seeds[:, 1] / max(1.0, Hc)))
            else:
                Lm_y = np.full(len(seeds), Lm)
            if kind in ("speckle", "dabs") and mat["dir"] == "random":
                ang = rng.uniform(0, math.pi, len(seeds))
                dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
                lens = Lm * rng.uniform(0.6, 1.4, len(seeds))
                paths = [np.stack([q - d * l / 2, q + d * l / 2]) for q, d, l in zip(seeds, dirs, lens)]
            else:
                lens = Lm_y * rng.uniform(0.5, 1.5, len(seeds))
                paths = trace(fx, fy, m, seeds, lens, max(1.0, 0.75 * s), float(mat.get("curve", 0.35)))
                if kind == "weave":
                    # the cross threads: a third of the marks run across the field, from their own places
                    k3 = len(paths) // 3
                    alt = seeds[:k3] + rng.uniform(-Lm / 2, Lm / 2, (k3, 2))
                    alt[:, 0] = np.clip(alt[:, 0], 0, Wc - 1)
                    alt[:, 1] = np.clip(alt[:, 1], 0, Hc - 1)
                    cross = trace(-fy, fx, m, alt, lens[:k3] * 0.8, max(1.0, 0.75 * s), 0.6)
                    paths = paths[k3:] + cross
                elif kind == "rib":
                    # ribs: marks along the field in columns, short ticks across them
                    k3 = len(paths) // 3
                    ticks = trace(-fy, fx, m, seeds[:k3], lens[:k3] * 0.6, max(1.0, 0.75 * s), 0.0)
                    paths = paths[k3:] + ticks
            for q in paths:
                if len(q) < 2:
                    continue
                Lq = float(np.hypot(*np.diff(q, axis=0).T).sum())
                if Lq < max(2.0 * w, 1.5 * s):
                    continue
                if np.hypot(*(q[-1] - q[0])) < 1e-3:
                    continue
                sgn = 1.0 if rng.random() < lit_side else -1.0
                u = sgn * abs(rng.normal(0.9, 0.45))
                ww = w * rng.uniform(0.7, 1.3)
                if sgn > 0:
                    u *= 0.7                    # light marks carry less: they read as scratches when strong
                fine_geo.append((q, pressure(q, ww, rng, 0.25 if kind in ("strands", "ripples") else 0.15), u,
                                 float(rng.uniform(o_lo, o_hi))))
    if fine_geo or dabs_geo:
        # contrast solved for the energy this area needs (proxy: the marks' L* offsets rasterised at working size,
        # reduced to the delivered size, 1-8 px bands over the area)
        target = fine_target(group, focus, background) * amount * float(mat.get("fc", 1.0))
        if mode == "cel":
            target *= 0.6                   # in cel work lines and edges carry much of the fine band
        need = max(0.0, target - have_fine)
        if have_fine > 0:
            need = max(need, 0.25 * target)
        proxy = np.zeros((Hc, Wc), np.float32)
        for q, wd, u, op in fine_geo:
            cv2.polylines(proxy, [np.round(q * 4).astype(np.int32).reshape(-1, 1, 2)], False, float(u * op),
                          max(1, int(round(float(np.mean(wd))))), cv2.LINE_AA, 2)
        for q, sz_, u, op in dabs_geo:
            cv2.circle(proxy, (int(round(q[0] * 4)), int(round(q[1] * 4))), max(1, int(round(sz_ * 0.5 * 4 * 0.6))),
                       float(u * op), -1, cv2.LINE_AA, 2)
        proxy *= m
        e1 = _fine_energy(proxy, m, s)
        k = math.sqrt(need / e1) * 100.0 if e1 > 1e-12 and need > 0 else 0.0
        k = float(np.clip(k, 0.0, float(mat.get("kmax", 8.0)) * min(1.2 if float(mat.get("kmax", 8.0)) <= (8.0 if mode == "paint" else 5.0) else 3.0,
                                                                     max(0.4, amount))))
        info["contrast"] = round(k, 1)
        info["target"] = target
        if k > 0.3 and dabs_geo:
            ctr = np.array([q for q, _, _, _ in dabs_geo])
            b, Ii = sample(ctr)
            dL = np.array([u for _, _, u, _ in dabs_geo]) * k
            cols = mark_colours(b, Ii, dL, mat, hue_l, big, rng)
            fine_ops.append(Spray(pts=to_full(ctr), size=np.array([z for _, z, _, _ in dabs_geo], np.float32),
                                  colors=cols, opacity=np.array([o for _, _, _, o in dabs_geo], np.float32),
                                  seed=int(seed) + 1))
        if k > 0.3 and fine_geo:
            mids = np.array([q[len(q) // 2] for q, _, _, _ in fine_geo])
            b, Ii = sample(mids)
            dL = np.array([u for _, _, u, _ in fine_geo]) * k
            dL += np.where(dL > 0, 0.25, -0.25) * k * np.clip(Ii, -1, 1)     # lighter strands catch the light
            cols = mark_colours(b, Ii, dL, mat, hue_l, big, rng)
            if len(fine_geo) > BATCH_FINE:
                fine_ops += batch_strokes(fine_geo, cols, (Hc, Wc), mask_op, s)
            else:
                for (q, wd, u, op), c in zip(fine_geo, cols):
                    fine_ops.append(Stroke(pts=to_full(q), width=wd, color=tuple(int(v) for v in c), opacity=op,
                                           brush="liner", seed=int(rng.integers(1 << 30)),
                                           own=mat.get("edge") == "lost" and rng.random() < 0.3))
    ops += fine_ops

    # ---------------------------------------------------------------- secondary contours
    lw = float(mat.get("lines", 0.4)) * (0 if flags.get("no_lines") else 1)
    if lw > 0 and area > 900 * s * s:
        from .draw_line import base_width
        base_w = base_width(p) * 0.42
        lines_geo: List[np.ndarray] = []
        # plane lines along the terminator, lost and found
        t = float(np.quantile(I[m], 0.3))
        bw = ((I > t) & m).astype(np.uint8)
        cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        inner = dc.erode(m, max(3.0 * s, 0.04 * size))
        for c in cs:
            c = c.reshape(-1, 2).astype(np.float64)
            if len(c) < 10:
                continue
            ok_ = inner[c[:, 1].astype(int).clip(0, Hc - 1), c[:, 0].astype(int).clip(0, Wc - 1)]
            run, runs = [], []
            for q, o in zip(c, ok_):
                if o:
                    run.append(q)
                elif run:
                    runs.append(np.array(run))
                    run = []
            if run:
                runs.append(np.array(run))
            for r in runs:
                if len(r) < 0.12 * size:
                    continue
                if np.hypot(*(r[-1] - r[0])) < 0.5 * len(r):
                    continue                    # a curl or a loop of the lighting, not a plane change
                r = dc.smooth_poly(dc.resample(r, 1.5 * s), 2.0)
                # lost and found: keep pieces
                Lr = dc.path_length(r)
                pos = 0.0
                while pos < Lr:
                    seg = rng.uniform(0.12, 0.3) * size
                    gap = rng.uniform(0.05, 0.2) * size
                    i0 = int(pos / (1.5 * s))
                    i1 = int(min(Lr, pos + seg) / (1.5 * s))
                    if i1 - i0 >= 4 and rng.random() < 0.75 * min(1.0, lw):
                        lines_geo.append(r[i0:i1])
                    pos += seg + gap
        # fold / strand lines along the field, from the edges the area shares with its neighbours and inside
        nl_ = int(lw * (3 + size / (90.0 * s)))
        if nl_ > 0 and mat["dir"] != "random":
            edge_seed = m & dc.dilate(oth, 2 * s) if oth.any() else np.zeros_like(m)
            cand = np.flatnonzero((edge_seed if edge_seed.sum() > 50 else erode_in).ravel())
            if len(cand):
                pick = rng.choice(cand, size=min(nl_, len(cand)), replace=False)
                py, px = np.divmod(pick, Wc)
                seeds = np.stack([px + 0.5, py + 0.5], 1)
                lens = rng.uniform(0.15, 0.45, len(seeds)) * size
                from .draw_detail import bend_path
                for q in trace(fx, fy, m, seeds, lens, 1.5 * s, max(0.6, float(mat.get("curve", 0.35)))):
                    if len(q) >= 6:
                        q = bend_path(dc.smooth_poly(q, 2.0), float(rng.uniform(-0.06, 0.06)))
                        if m[np.clip(q[:, 1].astype(int), 0, Hc - 1), np.clip(q[:, 0].astype(int), 0, Wc - 1)].mean() > 0.9:
                            lines_geo.append(q)
        if lines_geo:
            mids = np.array([q[len(q) // 2] for q in lines_geo])
            b, Ii = sample(mids)
            lch = to_lch(b)
            # a darker, more saturated version of the fill: not near-black
            lch[:, 0] = np.maximum(8.0, lch[:, 0] - (18.0 + 0.12 * lch[:, 0]) * min(1.3, 0.7 + 0.4 * lw))
            lch[:, 1] = lch[:, 1] * 1.25 + 3.0
            cols = from_lch(lch)
            for q, c in zip(lines_geo, cols):
                if dc.path_length(q) < 6 * s:
                    continue
                wd = pressure(q, max(1.0, base_w * rng.uniform(0.7, 1.2)), rng, 0.35, 0.12)
                ops.append(Stroke(pts=to_full(q), width=wd, color=tuple(int(v) for v in c),
                                  opacity=float(rng.uniform(0.6, 0.9)), brush="liner",
                                  seed=int(rng.integers(1 << 30))))
            info["lines"] = len(lines_geo)

    # ---------------------------------------------------------------- specular
    spec = mat.get("spec", "none") if not flags.get("no_spec") else "none"
    spa = float(mat.get("spa", 0.5))
    if spec != "none" and area > 300 * s * s:
        hi = float(np.quantile(I[m], 0.9))
        top = m & (I >= hi) & erode_in
        if top.sum() > 20 * s * s:
            c = region_colour(top)
            lc = np.array(dc.light_of(tuple(int(v) for v in c), 1.6 * spa), float)
            if spec == "sheen":
                ops.append(mask_op(top, lc, 0.35 * spa, 0.06 * size + fe))
            elif spec == "spot":
                ys2, xs2 = np.nonzero(top)
                j = int(np.argmax(I[ys2, xs2]))
                cx, cy = xs2[j] + 0.5, ys2[j] + 0.5
                r = max(1.5 * s, 0.045 * size)
                ang = math.degrees(math.atan2(light[1], light[0])) + 90
                soft = dc.ellipse_pts(cx, cy, 2.4 * r, 1.6 * r, ang, n=40) + [x0, y0]
                ops.append(Fill(polygons=[soft.astype(np.float32)], color=tuple(int(v) for v in lc),
                                opacity=0.3 * spa, feather=max(1.0, 1.2 * r)))
                hard = dc.ellipse_pts(cx, cy, r, 0.6 * r, ang, n=32) + [x0, y0]
                hc = dc.light_of(tuple(int(v) for v in lc), 1.0)
                ops.append(Fill(polygons=[hard.astype(np.float32)], color=hc, opacity=min(1.0, 0.9 * spa + 0.1),
                                feather=max(1.0, 0.25 * s)))
            elif spec == "band":
                # a band of short bright strokes along the field across the lit part (soft glow under it)
                # one band across the lit part, its edges broken into spikes along the field (the anime shine)
                sn_b = streak_noise(fx, fy, m, 0.12 * size, 1.5 * s, rng)
                band = m & (I >= float(np.quantile(I[m], 0.8))) & erode_in & (sn_b > -0.6)
                band = dc.clean_mask(dc.smooth_mask(band, 1.5 * s), int(80 * s * s))
                if band.sum() > 30 * s * s:
                    ops.append(mask_op(band, lc, (0.65 if crisp else 0.5) * spa, (0.8 if crisp else 2.0) * s))
                    rim_b = band & ~dc.erode(band, 2.0 * s)
                    cand = np.flatnonzero(rim_b.ravel())
                    nb = int(min(300, rim_b.sum() / (12.0 * s * s) * spa))
                    if nb > 0 and len(cand):
                        pick = rng.choice(cand, size=min(nb, len(cand)), replace=False)
                        py, px = np.divmod(pick, Wc)
                        seeds = np.stack([px + 0.5, py + 0.5], 1)
                        lens = rng.uniform(6, 22, len(seeds)) * s * scale
                        for q in trace(fx, fy, m, seeds, lens, max(1.0, 0.75 * s), 0.8):
                            if len(q) < 3:
                                continue
                            ww = max(1.0, rng.uniform(0.8, 1.6) * s * scale)
                            ops.append(Stroke(pts=to_full(q), width=pressure(q, ww, rng, 0.1, 0.05),
                                              color=tuple(int(v) for v in lc), opacity=float(rng.uniform(0.5, 0.8)) * spa,
                                              brush="liner", seed=int(rng.integers(1 << 30))))
    info["marks"] = sum(len(o.pts) if isinstance(o, Spray) else 1 for o in ops)
    return ops, info


# ============================================================================================ command

USAGE = {
    "surface": "easel surface auto   |   easel surface coat --material cloth-rough [--flow \"x,y x,y\"] [--amount 1]",
}


def _layer_ready(p) -> str:
    """The surface layer (above shade and paint, under the line art), created on first use."""
    names = [e["name"] for e in p.state["layers"]]
    if "surface" in names:
        return "surface"
    at = names.index("line") if "line" in names else len(names)
    p.state["layers"].insert(at, {"name": "surface", "mode": "normal", "opacity": 1.0, "visible": True,
                                  "clip": False, "alpha_lock": False, "empty": True})
    p._write_layer_file("surface", None)
    p.save_state()
    return "surface"


def _focus(p, name: str) -> int:
    d = dc.dstate(p)
    if name in d["parts"]:
        return int(d["parts"][name].get("focus", 2))
    return int(p.state.get("regions", {}).get(name, {}).get("focus", 1))


def _is_background(p, name: str) -> bool:
    from .workflow_draw import is_ground
    n = name.lower()
    if n in ("background", "bg", "backdrop", "sky"):
        return True
    d = dc.dstate(p)
    return n in d.get("bands", {}) and is_ground(n) and n not in d["parts"]


def surface_targets(p) -> List[str]:
    """The regions a surface pass covers: every coloured region but eyes, mouths, lettering."""
    out = []
    for n in p.region_names():
        sw = None
        d = dc.dstate(p)
        if n in d["parts"]:
            sw = d["parts"][n].get("swatch")
        if guess_material(n, sw) is None or sw == "eye-white":
            continue
        out.append(n)
    return out


def auto_material(p, name: str) -> str:
    from .draw_checks import is_animal
    d = dc.dstate(p)
    sw = d["parts"].get(name, {}).get("swatch") if name in d["parts"] else None
    bands = d.get("bands") if isinstance(d.get("bands"), dict) else {}
    if sw is None and name in bands:
        sw = bands[name].get("color")           # a block-in band carries its colour role (fur, wood ...)
    subj = d["parts"].get(name, {}).get("of") if name in d["parts"] else None
    if subj is None:
        subs = d.get("subjects", {})
        subj = name if name in subs else next((s for s in subs if name.startswith(s + "-")), None)
    animal = False
    try:
        animal = bool(subj) and is_animal(p, subj)
    except Exception:  # noqa: BLE001
        animal = False
    if name in d.get("locks", {}):
        return "hair"
    mat = guess_material(name, sw, animal) or "plain"
    if animal and mat in ("plain", "skin"):
        mat = "fur"
    sb = d.get("subjects", {}).get(subj or name, {}).get("box")
    backdrop = bool(sb) and (sb[2] - sb[0]) * (sb[3] - sb[1]) >= 0.8e6
    if (mat == "plain" or (mat == "stone" and backdrop)) and (_is_background(p, name) or backdrop):
        mat = "paper"                           # a painted backdrop (a room's wall, the paper behind a figure)
    return mat


def _parse_over(ctx, a) -> dict:
    over = {}
    if a.get("fine"):
        k = str(a["fine"]).lower()
        if k not in FINE_KINDS:
            raise UsageError(f"--fine is one of {', '.join(FINE_KINDS)}", USAGE["surface"])
        over["fine"] = k
    if a.get("spec"):
        k = str(a["spec"]).lower()
        if k not in SPECS:
            raise UsageError(f"--spec is one of {', '.join(SPECS)}", USAGE["surface"])
        over["spec"] = k
    if a.get("edge"):
        k = str(a["edge"]).lower()
        if k not in EDGES:
            raise UsageError(f"--edge is one of {', '.join(EDGES)}", USAGE["surface"])
        over["edge"] = k
    if a.get("contrast"):
        over["fc_x"] = coords.parse_float(a["contrast"], "contrast", 0, 4, ctx.notes)
    if a.get("lines"):
        over["lines_x"] = coords.parse_float(a["lines"], "lines", 0, 3, ctx.notes)
    return over


def cmd_surface(ctx, args: List[str]):
    from . import cli_draw as cd
    from .cli import Spec, parse
    p = ctx.project
    d = dc.dstate(p)
    if args and str(args[0]).lower() in ("list", "materials", "help"):
        return _list_report(ctx)
    a = parse(args, Spec(USAGE["surface"], ["region*"],
                         {"material": "value", "flow": "value", "dir": "value", "amount": "value", "scale": "value",
                          "fine": "value", "spec": "value", "edge": "value", "contrast": "value", "lines": "value",
                          "no-lines": "flag", "no-form": "flag", "no-tone": "flag", "again": "flag"},
                         alias={"mat": "material", "kind": "material", "m": "material", "direction": "dir",
                                "angle": "dir", "size": "scale", "detail": "amount"}, min_pos=1))
    spec_ = str(a["region"]).strip()
    auto = spec_.lower() in ("auto", "all", "every", "everything")
    if not p.region_names():
        raise UsageError("no coloured areas yet: the surface pass works on areas (flat makes them)", "easel flat auto")
    if auto:
        names = surface_targets(p)
        if not a.get("again") and not a.get("material"):
            done = set(d.get("surface", {}))
            fresh = [n for n in names if n not in done]
            names = fresh or names
    else:
        names = []
        for s_ in re.split(r"[,\s]+", spec_):
            if not s_:
                continue
            r = cd.region_or_part(ctx, s_, USAGE["surface"])
            names.append(r.name)
    if not names:
        raise UsageError("no area to put a surface on", "easel surface auto")
    over = _parse_over(ctx, a)
    if a.get("dir"):
        dv = str(a["dir"]).lower()
        if dv in DIRS:
            over["dir"] = dv
    angle = None
    if a.get("dir") and str(a["dir"]).lower() not in DIRS:
        deg = coords.parse_float(a["dir"], "dir", -360, 360, ctx.notes)
        angle = -deg                      # degrees as in the rest of easel (0 right, 90 up) -> screen
    amount = cd._num(a, "amount", 1.0, 0, 3, ctx)
    scale = cd._num(a, "scale", 1.0, 0.3, 3, ctx)
    flags = {"no_lines": bool(a.get("no-lines")), "no_form": bool(a.get("no-form")), "no_tone": bool(a.get("no-tone"))}
    curves_u = [coords.parse_points(a["flow"], ctx.notes, 2)] if a.get("flow") else None
    light = dc.light_vec(d)
    layer = _layer_ready(p)
    from .detail_measure import profile_name
    group = profile_name(p.state)
    # what each area already holds at 1-8 px (its lines and edges), on the delivered picture
    have = _area_fine(p, names)
    plan = []
    for n in names:
        prev = d.get("surface", {}).get(n, {})
        # an area keeps the material it was given (by hand or by an earlier auto) until --material says otherwise
        mname = a.get("material") or prev.get("material") or auto_material(p, n)
        keep_over = dict(prev.get("over") or {}) if not a.get("material") else {}
        keep_over.update(over)
        mat = material(mname, keep_over or None)
        cv = curves_u
        if cv is None and mat["dir"] == "flow":
            lk = (d.get("locks") or {}).get(n)
            if lk:
                cv = [k["path"] for k in lk.get("locks", []) if len(k.get("path", [])) >= 2 and not k.get("poly")]
        plan.append((n, mat, cv))
    groups: List[Tuple[list, np.ndarray]] = []
    infos = []
    under = p.composite_u8("export", upto=layer)       # the paint under the surface layer: no line colours
    la_ = p.layer("line", create=False)
    line_a = None if la_ is None else la_[..., 3]
    regions_all = np.zeros((p.H, p.W), bool)
    for n in p.region_names():
        if not _is_background(p, n):
            regions_all |= p.region_mask(n)

    def build_all():
        from .ops import Fill, mask_to_rle
        out = []
        for i, (n, mat, cv) in enumerate(plan):
            m = p.region_mask(n)
            ops = []
            old = p.layer(layer, create=False)
            if old is not None and (old[..., 3][m] > 0.02).any():
                ops.append(Fill(mask_rle=mask_to_rle(m), color=(0, 0, 0), erase=True, feather=1.0))
            built, info = build(p, n, m, mat, focus=_focus(p, n), light=light, base_img=under,
                                others=regions_all & ~m, curves_u=cv, angle=angle, amount=amount, scale=scale,
                                have_fine=0.0 if _is_background(p, n) else have.get(n, 0.0), group=group, background=_is_background(p, n),
                                seed=p.state["step"] * 977 + i * 31 + 7, flags=flags, line_alpha=line_a)
            ops += built
            groups.append((ops, m))
            infos.append((n, mat, info))
            out += ops
        return out

    def post():
        sd = d.setdefault("surface", {})
        for n, mat, info in infos:
            prev_o = dict((sd.get(n, {}) or {}).get("over") or {}) if not a.get("material") else {}
            prev_o.update(over)
            sd[n] = {"material": mat["name"], "amount": amount, "over": prev_o or None}
        lines = ["; ".join(f"{n} {mat['name']} ({info.get('marks', 0)} marks)" for n, mat, info in infos)[:400]]
        if auto:
            lines.append("materials chosen from the names: change one with  easel surface <area> --material <kind>  "
                         f"({', '.join(MATERIALS)})")
        return lines
    def lines_colour():
        # coloured line art: each line takes a darker, more saturated version of the fill on its part's side
        from . import draw_line
        recs = d["lines"]
        if not recs:
            return []
        dc.owned_fold(p, "line", recs, draw_line.render_ops)
        n_ = draw_line.recolour(p, under)
        dc.owned_write(p, "line", recs, draw_line.render_ops)
        if n_:
            return [f"{n_} lines took the colour of the fill beside them (darker, more saturated)"]
        return []
    return surface_step(ctx, f"surface {', '.join(n for n, _, _ in plan)}", layer, build_all, groups, post,
                        extra_files=dc.owned_files("line"), also=lines_colour)


def surface_step(ctx, label: str, layer: str, build_all, groups, post, extra_files=(), also=None):
    """One undoable step on the surface layer, each area's marks clipped to its own area (draw_step clips a whole
    step to one mask)."""
    from . import brush, workflow_draw
    from . import cli_draw as cd
    from .ops import ops_to_jsonable
    p = ctx.project
    rep = cd.DReport(mutated=True)
    before = dc.composite_u8(p, "view")
    (p.root / "draw").mkdir(exist_ok=True)
    step = p.begin_step([f"layers/{layer}.png"] + list(extra_files))
    try:
        ops_all = build_all()
        cd._check_marks(ops_all)
        arr = p.layer(layer).copy()
        bbox = None
        for ops, m in groups:
            if not ops:
                continue
            b = brush.apply_ops(arr, ops, clip=m.astype(np.float32))
            if b is not None:
                bbox = b if bbox is None else (min(bbox[0], b[0]), min(bbox[1], b[1]), max(bbox[2], b[2]),
                                               max(bbox[3], b[3]))
        p._write_layer_file(layer, arr)
        extra = (also() if also else []) + (post() or [])
        ops_file = p.save_ops(step, {"cmd": ctx.cmdline, "layer": layer, "ops": ops_to_jsonable(ops_all)})
        ticked = workflow_draw.refresh(p)
        units = coords.px_box_to_units(bbox, p.W, p.H) if bbox else None
        p.end_step(step, {"cmd": ctx.cmdline, "layer": layer, "ops": ops_file, "n_ops": len(ops_all),
                          "bbox": list(units) if units else None})
    except Exception:
        p.abort_step(step)
        raise
    rep.step = step
    rep.touched = bbox
    n_marks = sum(len(o.pts) if type(o).__name__ == "Spray" else 1 for o in ops_all)
    rep.add(f"step {step}  {label}  (layer {layer})  {n_marks} marks  box {coords.fmt_box(units) if units else '-'}")
    for ln in extra:
        if ln:
            rep.add(ln)
    for ln in detail_lines(p, [m for _, m in groups], limit=3):
        rep.add(ln)
    if ticked:
        rep.add(f"todo ticked: {', '.join(map(str, ticked))}")
    if not ctx.batch:
        area = cd.view_area(p, bbox, None, len(groups) > 1)
        cd.show_before_after(ctx, rep, before, dc.composite_u8(p, "view"), area, step,
                             "the canvas after the surface pass")
    cd._tail(ctx, rep)
    rep.summary = f"{n_marks} marks"
    return rep


# ============================================================================================ measuring

def _delivered_rgb(p) -> np.ndarray:
    from .detail_measure import delivered
    return delivered(dc.composite_u8(p), dc.work_scale(p))


_MEAS: Dict[tuple, dict] = {}


def picture_measure(p, keep: bool = True, lines: bool = True) -> dict:
    """detail_measure.measure of the delivered picture, once per state of the layers."""
    from .detail_measure import measure
    img = dc.composite_u8(p)
    key = (id(img), keep, lines)
    if key in _MEAS:
        return _MEAS[key]
    for k2, v2 in _MEAS.items():
        if k2[0] == id(img) and k2[1] >= keep and k2[2] >= lines:
            return v2
    m = measure(_delivered_rgb(p), jpeg=80, lines=lines, keep=keep)
    if len(_MEAS) > 4:
        _MEAS.clear()
    _MEAS[key] = m
    return m


def _small_mask(p, m: np.ndarray) -> np.ndarray:
    s = dc.work_scale(p)
    if s <= 1:
        return m
    from .draw_canvas import deliver_size
    dw, dh = deliver_size(p)
    return cv2.resize(m.astype(np.float32), (dw, dh), interpolation=cv2.INTER_AREA) > 0.5


def _area_fine(p, names) -> Dict[str, float]:
    from .detail_measure import area_measures
    try:
        m = picture_measure(p, lines=False)
        res = area_measures(m, {n: _small_mask(p, p.region_mask(n)) for n in names})
    except Exception:  # noqa: BLE001
        return {}
    return {n: float(sum(v.get(k, 0.0) for k in FINE_KEYS)) for n, v in res.items()}


def area_report(p, names=None) -> List[dict]:
    """Per area: its 1-8 px energy against what its focus asks for (the professional median x the focus share)."""
    from .detail_measure import area_measures, profile_name
    group = profile_name(p.state)
    names = list(names) if names is not None else surface_targets(p)
    m = picture_measure(p)
    res = area_measures(m, {n: _small_mask(p, p.region_mask(n)) for n in names})
    out = []
    for n, v in res.items():
        e = float(sum(v.get(k, 0.0) for k in FINE_KEYS))
        tgt = fine_target(group, _focus(p, n), _is_background(p, n))
        out.append({"name": n, "fine": e, "target": tgt, "ratio": e / max(tgt, 1e-12),
                    "empty": v.get("det_empty", float("nan")), "flat": v.get("flat_share", float("nan"))})
    return out


def detail_lines(p, masks=None, limit: int = 3) -> List[str]:
    """Plain sentences: the worst gaps of the picture against the profile, and the areas furthest below their
    detail budget, each with the command that addresses it."""
    from .detail_measure import gaps, profile_name
    out = []
    try:
        group = profile_name(p.state)
        m = picture_measure(p)
        g = gaps(m, group, n=limit)
        worst = sorted(area_report(p), key=lambda r: r["ratio"])
    except Exception as e:  # noqa: BLE001
        return [f"(detail measure failed: {type(e).__name__}: {e})"]
    cel_pic = style_mode(p) in ("cel", "soft") and group.startswith("anime")
    if _free(p) and style_mode(p) != "paint":
        # a flat style drawn from a request: no hint asks for more texture
        g = [x for x in g if not _asks_for_more(x)]
        cel_pic = True
    low = [r for r in worst if r["ratio"] < 0.6][:2] if not cel_pic else []
    if cel_pic:     # cel: the fine bands are not asked of the fills; the flats and the line weight are
        g = [x for x in g if not (x["key"].startswith(("bandL", "bandC")) or x["key"] in ("edge_density", "lc_p50"))
             and not (x["key"] == "det_empty" and x["low"])]
    for r in low:
        mat = dc.dstate(p).get("surface", {}).get(r["name"], {}).get("material") or auto_material(p, r["name"])
        out.append(f"DETAIL: {r['name']}: 1-8 px detail at {r['ratio']:.2f}x of its share of professional {group}; "
                   f"run  easel surface {r['name']} --material {mat} --amount "
                   f"{min(3.0, max(1.2, 1.0 / max(r['ratio'], 0.2) ** 0.5)):.1f}")
    for x in g:
        if any(x["key"] == k for k in ("bandL_0", "bandL_1", "bandL_2")) and low:
            continue
        out.append(f"DETAIL: whole picture: {x['text']}{_fix_for(p, x)}")
    return out[:limit + 1]


def _fix_for(p, x) -> str:
    key, lowv = x["key"], x.get("low")
    if key == "line_taper" and lowv:
        return "; more tapered form lines (folds, strands, plane changes):  easel surface auto --again --lines 1.6"
    sd = dc.dstate(p).get("surface", {})
    if key == "flat_share" and lowv:
        am = max(0.3, min([float(v.get("amount", 1.0)) for v in sd.values()] or [1.0]) * 0.6)
        return f"; the flats are broken up: fewer marks:  easel surface auto --again --amount {am:.1f}"
    if key in ("det_empty", "flat_share") and not lowv:
        # the empty tiles are mostly in the area with the least detail: more of its surface
        try:
            low_ = min(area_report(p), key=lambda r: r["ratio"] if r["empty"] != r["empty"] else -r["empty"])
            tgt = low_["name"]
        except Exception:  # noqa: BLE001
            bg = [n for n in surface_targets(p) if _is_background(p, n)]
            tgt = bg[0] if bg else "auto"
        am = min(3.0, float(sd.get(tgt, {}).get("amount", 1.0)) * 1.6)
        return f"; more surface where the tiles are empty:  easel surface {tgt} --amount {am:.1f}"
    amounts = [float(v.get("amount", 1.0)) for v in sd.values()] or [1.0]
    if key.startswith("bandL") and lowv:
        if min(amounts) >= 2.0:
            return "; sharper shapes: cel shadows with hard edges, catch-lights, strand ends"
        return f";  easel surface auto --again --amount {min(2.2, max(amounts) * 1.5):.1f}"
    if key.startswith("bandL") and not lowv:
        return f";  easel surface auto --again --amount {max(0.3, min(amounts) * 0.7):.1f}"
    if key == "bandC_6" and not lowv:
        return "; big fills are too saturated:  easel surface auto --again (it desaturates big fills)"
    return ""


def _figure_flat(p, m) -> float:
    """Flat share over the areas that are not the backdrop."""
    from .detail_measure import area_measures
    try:
        fig = np.zeros((p.H, p.W), bool)
        for n in p.region_names():
            if not _is_background(p, n):
                fig |= p.region_mask(n)
        return float(area_measures(m, {"fig": _small_mask(p, fig)})["fig"]["flat_share"])
    except Exception:  # noqa: BLE001
        return 1.0


def _amount_floor(p) -> bool:
    sd = dc.dstate(p).get("surface", {})
    return bool(sd) and all(float(v.get("amount", 1.0)) <= 0.35 for v in sd.values())


def _amount_spent(p) -> bool:
    sd = dc.dstate(p).get("surface", {})
    return bool(sd) and all(float(v.get("amount", 1.0)) >= 2.0 for v in sd.values())


def _free(p) -> bool:
    """A picture drawn from a request (no reference picture, not hand mode)."""
    st = p.state
    return st.get("mode") != "hand" and not st.get("has_ref", st.get("reference") is not None)


def _asks_for_more(x: dict) -> bool:
    """A measure below the range that only more marks would raise: fine energy too low, too many empty tiles, too
    much flat area, too few tapered form lines. Without a reference these are not holes (a flat style is flat on
    purpose, and more marks on a plain sky or lake read as noise); a picture that is too busy still is."""
    k = x["key"]
    return (k.startswith("bandL") and x["low"]) or (k in ("det_empty", "flat_share") and not x["low"]) or \
        (k == "line_taper" and x["low"])


def gate_holes(p, stage: str) -> List[dict]:
    """The detail measures outside the professional interquartile range (gate of the detail and finish stages).
    Drawn from a request, only a picture that is too busy is held: the gate never asks for more texture."""
    from .detail_measure import outside, profile_name
    if dc.work_scale(p) <= 1 and max(p.W, p.H) < 640:
        return []
    try:
        group = profile_name(p.state)
        m = picture_measure(p)
        bad = outside(m, group)
    except Exception:  # noqa: BLE001
        return []
    cel_pic = style_mode(p) in ("cel", "soft") and group.startswith("anime")
    free = _free(p)
    out = []
    for x in bad:
        if free and _asks_for_more(x):
            continue                     # drawn from a request: a picture is as plain as its style wants
        # too much fine energy is a softer failure than too little: allow up to 1.5x the upper quartile
        if x["key"].startswith("bandL") and not x["low"] and x["value"] <= 1.5 * x["q3"]:
            continue
        if x["key"] == "flat_share" and not x["low"] and cel_pic and _figure_flat(p, m) <= x["q3"]:
            continue                     # cel: a plain backdrop is flat by design; the figure is within range
        if x["key"] == "det_empty" and not x["low"] and cel_pic:
            continue                     # cel: the flats stay flat, so tiles without fine detail are expected
        if x["key"] == "flat_share" and x["low"] and (not cel_pic or _amount_floor(p)):
            continue                     # less flat than the quartile: painted work, not a hole
        if x["key"].startswith("bandL") and x["low"] and cel_pic:
            continue                     # cel: flats stay flat; the fine band comes from lines and eyes
        if x["key"] == "det_empty" and x["low"]:
            continue
        if x["key"] in ("det_empty", "flat_share") and not x["low"] and x["value"] <= x["q3"] + 0.03:
            continue                     # within 3 % of the canvas of the range (a sun disc, a plain sign)
        if x["key"] == "line_taper" and not x["low"]:
            continue                     # lines that change weight more than most: not a hole
        if x["key"].startswith("bandL") and x["low"] and _amount_spent(p):
            continue                     # every area is at twice its detail already: more marks would be noise
        fix = _fix_for(p, x).lstrip("; ").strip()
        cmd = fix.split("  ")[-1].strip() if "easel" in fix else "easel surface auto --again"
        if not cmd.startswith("easel"):
            cmd = "easel surface auto --again"
        mm_ = re.match(r"easel surface (\S+)(?: --again)? --amount ([0-9.]+)$", cmd)
        if mm_:
            sd_ = dc.dstate(p).get("surface", {})
            tg_, am_ = mm_.group(1), float(mm_.group(2))
            done_ = [float(v.get("amount", 1.0)) for k_, v in sd_.items() if tg_ in ("auto", k_)]
            if done_ and ((am_ >= 1 and min(done_) >= am_) or (am_ < 1 and max(done_) <= am_)):
                continue                 # that amount is laid already: the tool cannot do more here
        out.append({"metric": "detail " + x["key"], "text": f"{x['text']}:  {cmd}", "fix": cmd,
                    "value": f"{x['value']:.3g}", "need": f"{x['q1']:.3g}-{x['q3']:.3g}"})
    return out


def _list_report(ctx):
    from . import cli_draw as cd
    p = ctx.project
    rep = cd.DReport()
    rep.add("materials (easel surface <area> --material <name>; every part overridable):")
    for n, v in MATERIALS.items():
        m = material(n)
        rep.add(f"  {n:13s} fine {m['fine']}, along {m['dir']}, spec {m['spec']}, edge {m['edge']}")
    rep.add("overrides: --fine " + "|".join(FINE_KINDS) + "  --dir " + "|".join(DIRS) + "|<degrees>  --spec "
            + "|".join(SPECS) + "  --edge " + "|".join(EDGES) + "  --amount 0-3  --scale 0.3-3  --contrast 0-4  "
            "--lines 0-3 (x the material's)  --flow \"x,y x,y\"  --no-lines --no-form --no-tone")
    try:
        for n in surface_targets(p):
            cur = dc.dstate(p).get("surface", {}).get(n, {}).get("material")
            rep.add(f"  {n}: {'surface ' + cur if cur else 'no surface yet (auto: ' + auto_material(p, n) + ')'}")
    except Exception:  # noqa: BLE001
        pass
    rep.next = cd.cli._finish_next(ctx)
    return rep


COMMANDS = {"surface": cmd_surface}
