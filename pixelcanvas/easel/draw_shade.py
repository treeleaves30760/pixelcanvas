"""Shading without a reference, computed from the shapes the model already drew and the plan's light.

- ``form``: the region is inflated into a smooth dome (a height field from its distance transform), lit by the plan
  light; 1-3 tone levels are cut from that lighting: shadow, deep shadow (core), light. Hard (cel) or soft edges.
- ``cast``: one region's silhouette pushed away from the light lands on another region (hair on the face, a chin on
  the neck).
- ``rim``: a thin light band along the edge that faces the light (or every edge for a back light).
- ``crease``: soft occlusion where a region meets its neighbours.
- ``blush`` / ``highlight``: airbrush dabs and specular spots.

Everything returns ops (Fill / Stroke) for the shade layer; the caller clips them to the region.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .draw_core import clean_mask, dilate, erode, light_vec, smooth_mask


def _crop(mask: np.ndarray, pad: int):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    H, W = mask.shape
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    return (y0, y1, x0, x1)


def lighting(mask: np.ndarray, light: np.ndarray, roundness: float = 1.0) -> Tuple[np.ndarray, tuple]:
    """n . l over the region's inflated shape, in [-1, 1], on the region's crop. Returns (I, crop).

    The shape is inflated by blurring its hole-filled silhouette (eyes do not dent a face, a lock of hair cutting
    into a coat does not crease it): a smooth bump whose sides turn away from the viewer near the outline."""
    from scipy import ndimage as ndi
    m = np.asarray(mask, bool)
    # the form is the area plus what sits inside it (eyes, a muzzle, buttons): their edges are not outlines
    filled = ndi.binary_fill_holes(m)
    size = math.sqrt(max(1, filled.sum()))
    sig = max(2.0, 0.2 * size)
    pad = int(3 * sig) + 2
    y0, y1, x0, x1 = _crop(filled, pad)
    sub = filled[y0:y1, x0:x1].astype(np.float32)
    # close the notches first, so the bump follows the silhouette's big shape
    k = max(3, int(0.08 * size) | 1)
    sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    h = cv2.GaussianBlur(sub, (0, 0), sig)
    gy, gx = np.gradient(h)
    gmax = float(np.percentile(np.sqrt(gx * gx + gy * gy)[sub > 0], 99)) or 1e-6
    tilt = 2.2 * roundness / gmax
    n = np.stack([-gx * tilt, -gy * tilt, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    I = (n * light[None, None, :]).sum(-1)
    return I.astype(np.float32), (y0, y1, x0, x1)


def _shift(m: np.ndarray, dx: float, dy: float) -> np.ndarray:
    H, W = m.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # the picture's edge is not an outline: the shape continues past it
    return cv2.warpAffine(m.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE) > 0


def _round(m: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.8 or not m.any():
        return m
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma) > 0.5


def form_masks(mask: np.ndarray, light: np.ndarray, levels: int = 1, amount: float = 0.5,
               roundness: float = 1.0, body: Optional[np.ndarray] = None) -> List[Tuple[str, np.ndarray]]:
    """Tone shapes for one area: ('shadow', m), with levels 2 also ('deep', m), with 3 also ('light', m).

    Cel shading the way an illustrator does it: the shape is slid toward the light and the part of the area it no
    longer covers is the shadow, a crescent that hugs the outline on the side away from the light, its terminator a
    curve parallel to that outline. ``amount`` 0..1 sets the crescent's depth. Each area is shaded on its own, so
    no shadow edge runs through two areas."""
    from scipy import ndimage as ndi
    m = np.asarray(mask, bool)
    if m.sum() < 30:
        return []
    # the form is the area plus what sits inside it (eyes, a muzzle, buttons): their edges are not outlines
    filled = ndi.binary_fill_holes(m | (np.asarray(body, bool) if body is not None else False))
    ys, xs = np.nonzero(filled)
    size = math.sqrt(filled.sum())
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1)
    v = np.array([float(light[0]), float(light[1])])
    nv = float(np.linalg.norm(v))
    z = float(light[2])
    if nv < 0.15:                      # light from the front: a thin shadow all round the lower edge
        v, nv = np.array([0.0, -1.0]), 0.3
    v = v / max(np.linalg.norm(v), 1e-9)
    depth = (0.05 + 0.32 * float(np.clip(amount, 0, 1))) * min(size * 1.2, span * 0.6)
    if z < 0:                          # back light: most of the form is in shadow, a lit edge stays
        depth = max(depth, 0.55 * min(size, span * 0.6))
    depth *= float(np.clip(0.55 + 0.6 * nv, 0.5, 1.15))
    sm = max(1.0, 0.035 * size * roundness)
    out = []
    min_px = max(12, int(0.003 * m.sum()))

    def crescent(d):
        lit = _shift(filled, v[0] * d, -0 + v[1] * d)
        # round the terminator: the slid copy is softened, so corners do not leave straight cuts
        lit = _round(lit, sm)
        return clean_mask(m & ~lit, min_px)
    sh = crescent(depth)
    if sh.sum() >= 12:
        out.append(("shadow", sh))
    if levels >= 2:
        dp = crescent(depth * 0.4)
        if dp.sum() >= 12:
            out.append(("deep", dp))
    if levels >= 3:
        hl = m & ~_round(_shift(filled, -v[0] * depth * 0.5, -v[1] * depth * 0.5), sm)
        hl = clean_mask(hl, min_px)
        if hl.sum() >= 12:
            out.append(("light", hl))
    return out


def cast_mask(src: np.ndarray, dst: np.ndarray, light: np.ndarray, offset_px: float, soften: float = 1.0) -> np.ndarray:
    """Where ``src``'s silhouette, pushed away from the light by ``offset_px``, falls on ``dst`` (minus src itself)."""
    v = np.array([-light[0], -light[1]], float)
    nv = np.linalg.norm(v)
    if nv < 1e-3:
        v = np.array([0.0, 1.0])
    else:
        v /= nv
    dx, dy = v * offset_px
    H, W = src.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    sh = cv2.warpAffine(src.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    # a soft, slightly shrunk shadow edge reads as a shadow, not a copy of the shape
    sh = smooth_mask(sh, max(1.0, soften * offset_px * 0.25))
    return sh & dst & ~src


def rim_mask(mask: np.ndarray, light: np.ndarray, width_px: float, back: bool = False,
             open_side: Optional[np.ndarray] = None) -> np.ndarray:
    """A band along the edge that faces the light; with ``open_side`` (background pixels) only along the
    silhouette, never along an edge the region shares with another part."""
    m = np.asarray(mask, bool)
    band = m & ~erode(m, width_px)
    if open_side is not None:
        band &= dilate(np.asarray(open_side, bool) & ~m, width_px * 1.5)
    if back:
        return band
    cr = _crop(m, 3)
    y0, y1, x0, x1 = cr
    dt = cv2.distanceTransform(m[y0:y1, x0:x1].astype(np.uint8), cv2.DIST_L2, 5)
    dt = cv2.GaussianBlur(dt, (0, 0), 2.0)
    gy, gx = np.gradient(dt)
    nrm = np.sqrt(gx * gx + gy * gy) + 1e-6
    # the outward normal is minus the gradient of the distance
    facing = (-gx / nrm) * light[0] + (-gy / nrm) * light[1]
    keep = np.zeros(m.shape, bool)
    keep[y0:y1, x0:x1] = facing > 0.35
    return band & keep


def crease_mask(mask: np.ndarray, others: np.ndarray, width_px: float) -> np.ndarray:
    """A band inside ``mask`` along its border with ``others``."""
    near = dilate(others, width_px) & np.asarray(mask, bool)
    return near


def mask_fill(mask: np.ndarray, rgb, opacity: float = 1.0, feather: float = 1.0):
    from .ops import Fill, mask_to_rle
    return Fill(mask_rle=mask_to_rle(mask), color=tuple(int(c) for c in rgb), opacity=float(opacity),
                feather=max(1.0, float(feather)))


def dab(pt_px, size_px: float, rgb, opacity: float, brush: str = "soft", seed: int = 0):
    from .ops import Stroke
    return Stroke(pts=np.array([pt_px], np.float32), width=max(3.0, float(size_px)), color=tuple(int(c) for c in rgb),
                  opacity=float(opacity), brush=brush, seed=seed)


def brightest_point(mask: np.ndarray, light: np.ndarray) -> Optional[Tuple[float, float]]:
    if np.asarray(mask, bool).sum() < 10:
        return None
    I, (y0, y1, x0, x1) = lighting(mask, light)
    sub = np.asarray(mask, bool)[y0:y1, x0:x1]
    I = np.where(erode(sub, 2), I, -9)
    k = np.unravel_index(int(np.argmax(I)), I.shape)
    return float(x0 + k[1] + 0.5), float(y0 + k[0] + 0.5)
