"""Layer compositing: blend modes, opacity, visibility, clip-to-below, alpha lock flag.

Layers are float32 [H,W,4] straight alpha, sRGB 0..1 (DESIGN.md section 3). ``composite`` returns float32 RGB.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

MODES = ("normal", "multiply", "screen", "overlay", "add")


def blend(cb: np.ndarray, cs: np.ndarray, mode: str) -> np.ndarray:
    """Separable blend function B(Cb, Cs) of the W3C compositing spec (plus linear add)."""
    if mode == "normal":
        return cs
    if mode == "multiply":
        return cb * cs
    if mode == "screen":
        return cb + cs - cb * cs
    if mode == "overlay":
        return np.where(cb <= 0.5, 2.0 * cb * cs, 1.0 - 2.0 * (1.0 - cb) * (1.0 - cs))
    if mode == "add":
        return np.minimum(1.0, cb + cs)
    raise ValueError(f"unknown blend mode {mode!r}; one of {', '.join(MODES)}")


def composite_over(dst_rgb: np.ndarray, dst_a: np.ndarray, src: np.ndarray, mode: str = "normal",
                   opacity: float = 1.0, extra_alpha: Optional[np.ndarray] = None):
    """Composite straight-alpha ``src`` [H,W,4] over (dst_rgb, dst_a). Returns new (rgb, a)."""
    cs = src[..., :3]
    a_s = src[..., 3] * float(opacity)
    if extra_alpha is not None:
        a_s = a_s * extra_alpha
    a_b = dst_a
    b = blend(dst_rgb, cs, mode) if mode != "normal" else cs
    a_s3 = a_s[..., None]
    a_b3 = a_b[..., None]
    co = a_s3 * (1.0 - a_b3) * cs + a_s3 * a_b3 * b + (1.0 - a_s3) * a_b3 * dst_rgb
    ao = a_s + a_b * (1.0 - a_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.where(ao[..., None] > 1e-6, co / np.maximum(ao[..., None], 1e-6), 0.0)
    return rgb.astype(np.float32), ao.astype(np.float32)


def composite(stack: List[dict], layers: Dict[str, Optional[np.ndarray]], H: int, W: int,
              background=(1.0, 1.0, 1.0), skip: Iterable[str] = (), upto: Optional[str] = None) -> np.ndarray:
    """Flatten a layer stack (bottom to top) to float32 RGB [H,W,3] over an opaque ``background``.

    ``stack`` entries: dict(name, mode, opacity, visible, clip). ``layers`` maps name -> array or None (empty).
    ``clip``: the layer is clipped to the alpha of the nearest non-clipped layer below (its clipping base).
    ``upto``: stop before this layer (used to get the canvas under a layer). ``skip``: names left out.
    """
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = np.asarray(background, np.float32)
    a = np.ones((H, W), np.float32)
    base_alpha: Optional[np.ndarray] = None
    skip = set(skip)
    for entry in stack:
        name = entry["name"]
        if upto is not None and name == upto:
            break
        arr = layers.get(name)
        clip = bool(entry.get("clip"))
        if not clip:
            base_alpha = arr[..., 3] * float(entry.get("opacity", 1.0)) if (
                arr is not None and entry.get("visible", True) and name not in skip) else None
        if name in skip or not entry.get("visible", True) or arr is None:
            continue
        extra = None
        if clip:
            if base_alpha is None:
                continue  # clipped to an empty / hidden base: shows nothing
            extra = np.clip(base_alpha, 0.0, 1.0)
        # only the box where the layer has paint changes (exact: outside it the source alpha is 0); a 3x working
        # canvas has 9-11 Mpx and most layers cover a small part of it
        nz = _alpha_box(arr)
        if nz is None:
            continue
        y0, y1, x0, x1 = nz
        sub = (slice(y0, y1), slice(x0, x1))
        if entry.get("mode", "normal") == "normal" and extra is None:
            # over the opaque background the alpha stays 1: the same over operator, in place
            src = arr[sub]
            dst = rgb[sub]
            op = np.float32(entry.get("opacity", 1.0))
            tmp = np.subtract(src[..., :3], dst)
            a_s = src[..., 3:4]
            if op != 1.0:
                a_s = a_s * op
            np.multiply(tmp, a_s, out=tmp)
            dst += tmp
            continue
        r2, a2 = composite_over(rgb[sub], a[sub], arr[sub], entry.get("mode", "normal"), entry.get("opacity", 1.0),
                                None if extra is None else extra[sub])
        rgb[sub], a[sub] = r2, a2
    return rgb


def _alpha_box(arr: np.ndarray):
    """(y0, y1, x0, x1) of the pixels with alpha > 0, or None for a layer with no paint."""
    al = arr[..., 3]
    rows = np.flatnonzero(al.max(axis=1) > 0)
    if not len(rows):
        return None
    cols = np.flatnonzero(al[rows[0]:rows[-1] + 1].max(axis=0) > 0)
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def flatten_layer_on(layer: np.ndarray, background=(1.0, 1.0, 1.0)) -> np.ndarray:
    """One layer over a flat colour (for layer thumbnails)."""
    H, W = layer.shape[:2]
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = np.asarray(background, np.float32)
    out, _ = composite_over(rgb, np.ones((H, W), np.float32), layer)
    return out
