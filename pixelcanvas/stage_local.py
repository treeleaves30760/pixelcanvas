"""The local character of the reference: what kind of surface each part of the picture is.

One picture often mixes kinds: a painted scene with crisp architecture lines, a poster with a photograph and flat
lettering, a soft-shaded figure on a flat background.  The stages choose their treatment per area from this map
instead of per picture:

- ``FLAT``  hard-edged flat: uniform areas bounded by sharp edges (cel flats, lettering, graphic shapes,
  architecture) -> flat fills and the residual shape pass, sub-pixel outlines, no grain, no textured strokes;
- ``LINE``  thin line: dark or light ridges 1-3 px wide on a clean ground -> ink masses / pen strokes at true width;
- ``RAMP``  soft ramp: smooth tone change, no texture -> airbrush inside friskets, blending;
- ``TEXTURE`` high-frequency structure (a direction or speckle) -> directed bristle / sponge strokes, texture marks;
- ``grain``: the pixel-fine isotropic noise of an area (a photograph's sensor noise, a print's paper grain) as an
  amount per area (Lab L std of the 1 px residual away from edges, 0 where the reference is clean).

The map is computed once per picture (cached by the picture's bytes) from local statistics of the reference only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import cv2
import numpy as np

from .trace import to_lab

__all__ = ["CharMap", "char_map", "FLAT", "LINE", "RAMP", "TEXTURE", "CLASS_NAMES", "summary_style",
           "clean_mask", "texture_mask", "grain_map", "noise_map"]

FLAT, LINE, RAMP, TEXTURE = 1, 2, 3, 4
CLASS_NAMES = {FLAT: "flat", LINE: "line", RAMP: "ramp", TEXTURE: "texture"}

# thresholds (Lab L units, 0..100, and Delta E per px)
EDGE_DE = 5.0          # gradient (Delta E / px at 0.7 px) from which a pixel is part of a sharp edge
TEX_STD = 1.5          # std of the 5 px median residual (0.8 px smoothed lightness), edges left out, from which an area is textured
TEX_DENSE = 0.30       # share of edge pixels around (r ~ 4 px) from which an area is busy texture (foliage, tiles)
RAMP_LO = 0.22         # Delta E / px at 3 px: below = flat, above = a ramp
GRAIN_FINE = 0.62      # 1 px over 2.5 px residual from which an area's noise is pixel-fine grain
GRAIN_FINE_TEX = 0.75  # ... in a textured area (its own fine structure must not count as grain)
GRAIN_MIN = 0.45       # 1 px residual std below which an area counts as clean (no grain)
LINE_RIDGE = 10.0
TEX_MIN_AREA = 600     # px: smallest texture area (smaller islands in clean work count as part of it)
TEX_MIN_SHARE = 0.002  # ... or this share of the picture
PURE_RANGE = 2.5       # L range in 3x3 of the 5 px median image below which a pixel is clean
PURE_MAX = 0.6         # share of clean pixels around (r ~ 4 px) from which an area is not a texture
LINE_DENSE = 0.35      # share of ridge pixels around (r ~ 4 px) from which thin ridges are a texture of streaks      # ridge response from which a thin ridge is a line


@dataclass
class CharMap:
    cls: np.ndarray                       # uint8 [H,W] FLAT / LINE / RAMP / TEXTURE
    grain: np.ndarray                     # float32 [H,W] fine-noise std (Lab L), smooth over ~8 px; 0 = clean
    tex: np.ndarray                       # float32 [H,W] texture strength (2.5 px residual std, edges left out)
    edge: np.ndarray                      # bool [H,W] sharp edges
    shares: dict = field(default_factory=dict)

    def mask(self, *classes) -> np.ndarray:
        return np.isin(self.cls, classes)


_CACHE: dict = {}


def _key(ref):
    a = np.ascontiguousarray(ref)
    return (a.shape, hashlib.sha1(a[::3, ::3].tobytes()).hexdigest())


def _norm_blur(x, w, sigma):
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-3), den


def char_map(ref) -> CharMap:
    """The cached local character map of ``ref`` (uint8 RGB)."""
    ref = np.asarray(ref)[..., :3]
    k = _key(ref)
    hit = _CACHE.get(k)
    if hit is not None:
        return hit
    cm = _compute(ref)
    if len(_CACHE) > 6:
        _CACHE.clear()
    _CACHE[k] = cm
    return cm


def _grad(lab, sigma):
    b = cv2.GaussianBlur(lab, (0, 0), sigma) if sigma > 0 else lab
    g = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        g += cv2.Sobel(b[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(b[..., c], cv2.CV_32F, 0, 1) ** 2
    return np.sqrt(g) / 8.0


def _compute(ref) -> CharMap:
    from .lines import ridge_strength
    H, W = ref.shape[:2]
    lab = to_lab(ref).astype(np.float32)
    L = lab[..., 0]
    s = max(1.0, min(H, W) / 600.0) ** 0.5           # statistics windows grow a little with the picture
    g07 = _grad(lab, 0.7)
    edge = g07 >= EDGE_DE
    # thin ridges, dark and light (the Hessian across a 1-3 px line), on the lightness
    # (thin: the response at 0.8-1.2 px beats the one at 2.6 px, so a band 5 px wide is a shape, not a line)
    rd, _ = ridge_strength(L, sigmas=(0.8, 1.2))
    rl, _ = ridge_strength(100.0 - L, sigmas=(0.8, 1.2))
    rd2, _ = ridge_strength(L, sigmas=(2.6,))
    rl2, _ = ridge_strength(100.0 - L, sigmas=(2.6,))
    ridge = ((rd >= LINE_RIDGE) & (rd >= rd2)) | ((rl >= LINE_RIDGE) & (rl >= rl2))
    zone = cv2.dilate((edge | ridge).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    nonz = (~zone).astype(np.float32)
    # texture: what a 5 px median takes out of the lightly smoothed lightness (a step between two clean areas
    # survives a median, so edges, cloud shapes and letters are not texture; streaks, speckle and strands of 2-4 px
    # are removed by it), measured away from thin ridges (they are lines); the light smoothing first leaves a print's
    # paper grain and a camera's pixel noise to the grain measure
    Ls = cv2.GaussianBlur(L, (0, 0), 0.8)
    L8 = np.clip(Ls * 2.55, 0, 255).astype(np.uint8)
    rt = (Ls - cv2.medianBlur(L8, 5).astype(np.float32) / 2.55)
    rz = (~(cv2.dilate(ridge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)).astype(np.float32)
    t2, _ = _norm_blur(rt * rt, rz, 4.0 * s)
    tex = np.sqrt(np.maximum(t2, 0))
    # where nearly everything is edge (foliage, roof tiles, hair strands at small scale) the area is texture
    dens = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), 4.0 * s)
    busy = (dens >= 0.55) & (tex >= 0.6 * TEX_STD)
    # thin ridges packed close together are streaks (grass, strands, hatching, ripples), not separate lines
    rdens = cv2.GaussianBlur(ridge.astype(np.float32), (0, 0), 4.0 * s)
    busy |= rdens >= LINE_DENSE
    # grain: the 1 px residual of the smooth parts, compared with what the 2.5 px residual carries (pixel-fine noise)
    r1 = L - cv2.GaussianBlur(L, (0, 0), 1.0)
    n1, _ = _norm_blur(r1 * r1, nonz, 5.0 * s)
    n1 = np.sqrt(np.maximum(n1, 0))
    # ramps vs flats: the colour change over ~3 px, edges left out
    # (a 5 px median first takes the grain out; the gradient at 1.5 px reaches ~4 px from an edge, so the measure
    # leaves out a 9 px band around edges and lines)
    labm = np.stack([cv2.medianBlur(np.ascontiguousarray(lab[..., c]), 5) for c in range(3)], -1)
    g3 = _grad(labm, 1.5)
    wide = (~(cv2.dilate((edge | ridge).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)).astype(np.float32)
    g3n, _ = _norm_blur(g3, wide, 3.0 * s)
    # a texture has few clean pixels: in line art, lettering or a flat graphic most pixels of the 5 px median image
    # sit on a flat (thin lines and corners, which the median takes out, included); in foliage, stone, tiles or
    # streaks the median image still changes from pixel to pixel
    Lm = cv2.medianBlur(L8, 5).astype(np.float32) / 2.55
    rng3 = cv2.dilate(Lm, np.ones((3, 3), np.uint8)) - cv2.erode(Lm, np.ones((3, 3), np.uint8))
    pure = cv2.GaussianBlur((rng3 < PURE_RANGE).astype(np.float32), (0, 0), 4.0 * s)
    textured = ((tex >= TEX_STD) | busy) & (pure < PURE_MAX)
    cls = np.full((H, W), FLAT, np.uint8)
    cls[(g3n >= RAMP_LO) & ~textured] = RAMP
    cls[textured] = TEXTURE
    # thin lines on a clean ground (not inside texture): their own class
    cls[ridge & ~textured] = LINE
    # clean up: majority over a small window for the area classes (lines keep their pixels)
    area_cls = cls.copy()
    votes = np.stack([cv2.blur((area_cls == c).astype(np.float32), (5, 5)) for c in (FLAT, RAMP, TEXTURE)], 0)
    maj = np.array([FLAT, RAMP, TEXTURE], np.uint8)[np.argmax(votes, 0)]
    cls = np.where(cls == LINE, LINE, maj).astype(np.uint8)
    # small islands of texture inside clean work (the corners and junctions of line art, a frill's dense lines) are
    # part of the clean drawing: a texture is an area, not a speck
    from scipy import ndimage as ndi
    tl, n = ndi.label(cls == TEXTURE)
    if n:
        cnt = np.bincount(tl.ravel(), minlength=n + 1)
        small = cnt < max(TEX_MIN_AREA, TEX_MIN_SHARE * H * W)
        small[0] = False
        cls[small[tl]] = FLAT
    # grain amount: the fine noise where the area is smooth (flat or ramp); textured areas carry their own structure
    # (measured away from edges and lines in every kind of area: a print's paper grain lies over its textured
    # passages too; the pixel-fine test keeps a texture's own structure out of it)
    smooth = cls != LINE
    # (pixel-fine: the 1 px residual carries most of the 2.5 px residual, as in a camera's noise or a print's paper
    # grain; fine waves, brush texture and soft shading vary over several pixels and are left to the marks)
    r2 = L - cv2.GaussianBlur(L, (0, 0), 2.5)
    n2, _ = _norm_blur(r2 * r2, nonz, 5.0 * s)
    fine = n1 / np.maximum(np.sqrt(np.maximum(n2, 0)), 1e-3)
    fine_need = np.where(cls == TEXTURE, GRAIN_FINE_TEX, GRAIN_FINE)
    grain = np.where(smooth & (n1 >= GRAIN_MIN) & (fine >= fine_need), n1, 0.0).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 4.0 * s)
    shares = {CLASS_NAMES[c]: round(float((cls == c).mean()), 4) for c in CLASS_NAMES}
    shares["grain"] = round(float((grain >= GRAIN_MIN).mean()), 4)
    return CharMap(cls=cls, grain=grain, tex=tex.astype(np.float32), edge=edge, shares=shares)


def noise_map(img, ref, sigma=1.0) -> np.ndarray:
    """float32 [H,W]: local std (grey levels 0..255) of ``img``'s ``sigma`` px residual away from the reference's
    edges and lines, spread smoothly over ~5 px; 0 where there is nothing to measure."""
    cm = char_map(ref)
    H, W = cm.cls.shape
    s = max(1.0, min(H, W) / 600.0) ** 0.5
    g = cv2.cvtColor(np.asarray(img)[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = g - cv2.GaussianBlur(g, (0, 0), sigma)
    smooth = (cm.cls != LINE) & ~(cv2.dilate(cm.edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    v, den = _norm_blur(r * r, smooth.astype(np.float32), 5.0 * s)
    return np.where(den > 0.05, np.sqrt(np.maximum(v, 0)), 0.0).astype(np.float32)


def clean_mask(ref, grow=1) -> np.ndarray:
    """bool [H,W]: hard-edged flat areas and thin lines (crisp content), grown by ``grow`` px."""
    cm = char_map(ref)
    m = cm.mask(FLAT, LINE)
    if grow > 0:
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    return m


def texture_mask(ref) -> np.ndarray:
    return char_map(ref).mask(TEXTURE)


def grain_map(ref) -> np.ndarray:
    return char_map(ref).grain


def summary_style(ref) -> str:
    """A style name that summarises the map (for the stage plan's names and what the model is told)."""
    cm = char_map(ref)
    sh = cm.shares
    return max(("flat", "ramp", "texture"), key=lambda k: sh[k])
