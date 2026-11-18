"""Shared fixtures for the SHELL tests: a synthetic reference and a small stand-in brush engine.

The stand-in is injected into ``sys.modules`` under HAND's module names, so the SHELL tests do not depend on the
real engine's behaviour (they test the shell: projects, history, reports, gates).
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def synthetic_ref(w: int = 300, h: int = 360) -> np.ndarray:
    """A small cartoon: grey background, green ground, a skin face with a dark outline, brown hair, two eyes."""
    img = np.full((h, w, 3), 236, np.uint8)
    cv2.rectangle(img, (0, int(h * 0.65)), (w, h), (90, 150, 70), -1)
    cx, cy, r = w // 2, int(h * 0.38), int(w * 0.25)
    cv2.circle(img, (cx, cy), r, (240, 200, 170), -1)
    cv2.circle(img, (cx, cy), r, (30, 30, 30), 2)
    cv2.ellipse(img, (cx, cy - r // 2), (int(r * 1.1), r // 2), 0, 180, 360, (120, 60, 30), -1)
    cv2.circle(img, (cx - r // 3, cy), max(3, r // 10), (30, 30, 30), -1)
    cv2.circle(img, (cx + r // 3, cy), max(3, r // 10), (30, 30, 30), -1)
    return img


def write_ref(path, w: int = 300, h: int = 360):
    Image.fromarray(synthetic_ref(w, h)).save(path)
    return path


# ----------------------------------------------------------------------------------------------- fake engine

@dataclass(eq=False)
class Stroke:
    pts: np.ndarray
    width: float = 2.0
    color: object = (0, 0, 0)
    opacity: float = 1.0
    brush: str = "pen"
    seed: int = 0
    erase: bool = False


@dataclass(eq=False)
class Fill:
    polygons: Optional[list] = None
    mask_rle: Optional[dict] = None
    color: tuple = (0, 0, 0)
    opacity: float = 1.0
    feather: float = 0.0
    color2: Optional[tuple] = None
    p0: Optional[tuple] = None
    p1: Optional[tuple] = None
    erase: bool = False


def _op_mask(op, H, W):
    m = np.zeros((H, W), np.uint8)
    if isinstance(op, Fill):
        if op.mask_rle is not None:
            return np.asarray(op.mask_rle["mask"], bool)
        for poly in op.polygons:
            cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
        return m > 0
    pts = np.round(np.asarray(op.pts)).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) == 1:
        cv2.circle(m, tuple(int(v) for v in pts[0, 0]), max(1, int(op.width / 2)), 1, -1)
    else:
        cv2.polylines(m, [pts], False, 1, max(1, int(round(op.width))))
    return m > 0


def apply_ops(layer, ops, clip=None, alpha_lock=False):
    H, W = layer.shape[:2]
    box = None
    for op in ops:
        m = _op_mask(op, H, W).astype(np.float32)
        if clip is not None:
            m = m * (np.asarray(clip, np.float32) > 0.5)
        if alpha_lock:
            m = m * (layer[..., 3] > 0)
        a = m * float(op.opacity)
        if not a.any():
            continue
        if op.erase:
            layer[..., 3] *= (1 - a)
        else:
            col = np.asarray(op.color, np.float32)
            if col.ndim == 2:
                col = col.mean(0)
            col = col / 255.0
            layer[..., :3] = layer[..., :3] * (1 - a[..., None]) + col * a[..., None]
            layer[..., 3] = np.maximum(layer[..., 3], a)
        ys, xs = np.nonzero(a)
        b = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    return box


def ops_to_jsonable(ops):
    return [{"type": type(o).__name__} for o in ops]


def mask_to_rle(mask):
    return {"mask": np.asarray(mask, bool)}


PRESETS = {k: None for k in ("pencil", "pen", "marker", "round", "flat", "dry", "soft", "wash")}


def _mask_fill(ref, mask, per_pixel=True):
    """One Fill per quantised reference colour inside the mask (close to a copy)."""
    m = np.asarray(mask, bool)
    q = (ref // 24).astype(np.int32)
    key = q[..., 0] * 10000 + q[..., 1] * 100 + q[..., 2]
    ops = []
    for k in np.unique(key[m]):
        sel = m & (key == k)
        ops.append(Fill(mask_rle={"mask": sel}, color=tuple(int(v) for v in np.median(ref[sel], 0))))
    return ops


def sketch_ops(ref, mask=None, *, looseness=0.5, seed=0):
    m = np.asarray(mask, np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [Stroke(pts=c.reshape(-1, 2).astype(np.float32), width=1.5, color=(90, 90, 90), brush="pencil")
            for c in cs if len(c) >= 2]


def ink_ops(ref, mask=None, *, detail=0.5, weight=1.0, color="auto", pen="pen", seed=0):
    L = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    dark = (L < 80) & np.asarray(mask, bool)
    return [Fill(mask_rle={"mask": dark}, color=(30, 30, 30))] if dark.any() else []


def flat_ops(ref, mask=None, *, colors="auto", line_mask=None, seed=0):
    return _mask_fill(ref, mask)


def shade_ops(ref, mask, base, *, levels=3, soft=0.3, seed=0):
    return []


def paint_ops(ref, mask=None, canvas=None, *, size=24.0, brush="round", density=1.0, max_strokes=None, seed=0):
    return _mask_fill(ref, mask)


def install_fake_engine(monkeypatch):
    """Put the stand-in engine and stage modules in sys.modules under HAND's names."""
    g = globals()
    ops = types.ModuleType("pixelcanvas.ops")
    for k in ("Stroke", "Fill", "apply_ops", "ops_to_jsonable", "mask_to_rle"):
        setattr(ops, k, g[k])
    brush = types.ModuleType("pixelcanvas.brush")
    brush.PRESETS = PRESETS
    monkeypatch.setitem(sys.modules, "pixelcanvas.ops", ops)
    monkeypatch.setitem(sys.modules, "pixelcanvas.brush", brush)
    for mod, fn in (("stage_sketch", "sketch_ops"), ("stage_ink", "ink_ops"), ("stage_flat", "flat_ops"),
                    ("stage_shade", "shade_ops"), ("stage_paint", "paint_ops")):
        m = types.ModuleType(f"pixelcanvas.{mod}")
        setattr(m, fn, g[fn])
        if mod == "stage_paint":            # the texture and highlight passes of the paint / detail stages
            m.texture_ops = lambda ref, mask=None, canvas=None, **kw: []
            m.highlight_ops = lambda ref, mask=None, canvas=None, **kw: []
        monkeypatch.setitem(sys.modules, f"pixelcanvas.{mod}", m)


def run(*argv):
    """Run one pixelcanvas command line in-process; returns (code, text)."""
    from pixelcanvas import cli
    return cli.run([str(a) for a in argv])


def view_path(text: str) -> Optional[str]:
    for ln in text.splitlines():
        if ln.startswith("VIEW: "):
            return ln[6:].split("   (")[0].strip()
    return None


def next_cmd(text: str) -> str:
    for ln in text.splitlines():
        if ln.startswith("NEXT: "):
            return ln[6:].split("   (")[0].strip()
    return ""
