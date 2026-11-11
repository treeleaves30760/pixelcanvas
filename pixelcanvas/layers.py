# -*- coding: utf-8 -*-
"""
layers — turn a render into editable layers (PSD / OpenRaster / PNG stack) for humans to touch up.

How it works
------------
Every drawing step (each ``log("step")`` call in a scene, or ``rec.mark("step")`` by hand) becomes one layer.
The recorder snapshots the canvas at each step boundary and, while a step runs, accumulates the *coverage* of
every ``cv.paint`` / ``cv.paste`` / ``cv.text`` in normal mode (= "where an object was put down"). A layer is
then the RGBA image that, composited *normal* over everything below, reproduces the canvas after that step:

* alpha = max(coverage, minimal alpha needed to reach the new colour) — so painted objects are solid where
  they were painted (move the mountain and the sky is still behind it), while glows / haze / post effects
  that were not painted through a mask get the smallest alpha that reproduces them (they stay see-through);
* colour = (after - (1 - alpha) * before) / alpha.

Compositing all layers bottom→top therefore reproduces the final image exactly (±1/255), and editing a lower
layer still shows through the see-through ones above it. Values are taken in display space (clamped 0~1), so
HDR intermediates are clipped per layer and the final "post" step (bloom / soft_clip / grading) becomes a thin
adjustment layer — hide it to see the raw paint.

Usage
-----
>>> from pixelcanvas import Canvas, layers
>>> cv = Canvas(640, 360)
>>> rec = layers.LayerRecorder(cv)          # or layers.enable() before render() to follow log() calls
>>> rec.mark("sky");  cv.gradient([(0, '#0b1a3a'), (1, '#f6c28f')])
>>> rec.mark("sun");  cv.glow(400, 200, 60, '#fff0c0')
>>> stack = rec.finish()                    # list[Layer], bottom → top
>>> layers.save_psd(stack, 'out.psd'); layers.save_ora(stack, 'out.ora'); layers.save_pngs(stack, 'out_layers')

``python tools/render.py SCENE --layers`` does all of this for a scene; ``python tools/layers.py`` inspects /
re-composites an edited stack.
"""
from __future__ import annotations

import io
import json
import os
import re
import struct
import zipfile
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

F32 = np.float32
EPS = 0.5 / 255.0                         # changes below half an 8-bit level are ignored

__all__ = ["Layer", "LayerRecorder", "enable", "mark", "take", "active", "composite", "save_psd", "save_ora",
           "save_pngs", "load_pngs", "save_all", "reconcile", "push_post"]


# ------------------------------------------------------------------ data

@dataclass
class Layer:
    """One RGBA layer, straight (non-premultiplied) alpha, uint8, cropped to ``bbox``.

    ``rgba`` has shape (y1 - y0, x1 - x0, 4); ``bbox = (x0, y0, x1, y1)`` in canvas pixels; ``size = (W, H)``.
    """
    name: str
    rgba: np.ndarray
    bbox: tuple
    size: tuple
    visible: bool = True
    opacity: float = 1.0
    kind: str = "paint"                   # "base" | "paint" | "effect" (no painted coverage) | "post"
    meta: dict = field(default_factory=dict)
    group: str = None                     # layers sharing a group name are one editable unit (object + its halo)

    def full(self):
        """The layer as a full-canvas (H, W, 4) float32 array, 0~1."""
        W, H = self.size
        out = np.zeros((H, W, 4), F32)
        x0, y0, x1, y1 = self.bbox
        out[y0:y1, x0:x1] = self.rgba.astype(F32) / 255.0
        return out

    def to_pil(self):
        return Image.fromarray(self.rgba, "RGBA")


def _clamp(a):
    return np.clip(np.asarray(a, F32), 0.0, 1.0)


def _extract(before, after, coverage=None):
    """(H, W, 3) before / after in 0~1 → (alpha (H, W), colour (H, W, 3)) such that over() reproduces after."""
    d = after - before
    up = np.where(d > EPS, d / np.maximum(1.0 - before, 1e-6), 0.0)
    down = np.where(d < -EPS, -d / np.maximum(before, 1e-6), 0.0)
    amin = np.clip(np.maximum(up, down).max(-1), 0.0, 1.0)
    a = amin if coverage is None else np.maximum(amin, _clamp(coverage))
    safe = np.maximum(a, 1e-6)[..., None]
    col = np.where(a[..., None] > 0, before + d / safe, after)
    return a.astype(F32), _clamp(col)


def _to_layer(name, a, col, size, kind, meta=None):
    a8 = np.round(a * 255.0).astype(np.uint8)
    ys, xs = np.nonzero(a8)
    if len(xs) == 0:
        bbox = (0, 0, 0, 0)
        rgba = np.zeros((0, 0, 4), np.uint8)
    else:
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        bbox = (x0, y0, x1, y1)
        rgba = np.empty((y1 - y0, x1 - x0, 4), np.uint8)
        rgba[..., :3] = np.round(col[y0:y1, x0:x1] * 255.0).astype(np.uint8)
        rgba[..., 3] = a8[y0:y1, x0:x1]
        rgba[rgba[..., 3] == 0, :3] = 0
    return Layer(name, rgba, bbox, size, kind=kind, meta=dict(meta or {}))


# ------------------------------------------------------------------ recorder

class LayerRecorder:
    """Records a canvas step by step. ``mark(name)`` closes the running step and opens ``name``."""

    def __init__(self, cv=None, base_name="base", post="regrade", halos=None):
        self.cv = None
        self.base_name = base_name
        self.post = post                  # "regrade": push the final post step into every layer (no ghost layer)
        # "split": soft shadows / glows around a painted object become their own "<name> (halo)" layer
        self.halos = os.environ.get("PIXELCANVAS_HALOS", "split") if halos is None else halos
        self.final = None
        self.layers = []
        self.current = None
        self.before = None
        self.coverage = None
        self.notes = 0
        if cv is not None:
            self.attach(cv)

    def attach(self, cv):
        self.cv = cv
        if self not in _ATTACHED:
            _ATTACHED.append(self)
        return self

    # called by Canvas.paint / paste (normal mode only) while this recorder is active
    def note(self, cv, alpha, region=None):
        if cv is not self.cv or self.current is None:
            return
        H, W = cv.h, cv.w
        if self.coverage is None or self.coverage.shape != (H, W):
            self.coverage = np.zeros((H, W), F32)
        a = np.asarray(alpha, F32)
        tgt = self.coverage if region is None else self.coverage[region[1]:region[3], region[0]:region[2]]
        if a.ndim == 0:
            a = np.full(tgt.shape, float(a), F32)
        elif a.ndim == 3:
            a = a.max(-1)
        np.maximum(tgt, _clamp(np.broadcast_to(a, tgt.shape)), out=tgt)
        self.notes += 1

    def _snapshot(self):
        return _clamp(self.cv.buf).copy()

    def mark(self, name, cv=None):
        if cv is not None and self.cv is None:
            self.attach(cv)
        if self.cv is None:
            return self
        now = self._snapshot()
        if self.current is None:                       # first step: whatever is on the canvas is the base
            H, W = now.shape[:2]
            self.layers.append(_to_layer(self.base_name, np.ones((H, W), F32), now, (W, H), "base"))
        else:
            self._close(now)
        self.current, self.before, self.coverage, self.notes = str(name), now, None, 0
        return self

    def _close(self, now):
        if self.before is None or now.shape != self.before.shape:   # canvas resized mid-render → flatten
            H, W = now.shape[:2]
            self.layers = [_to_layer(self.base_name, np.ones((H, W), F32), now, (W, H), "base")]
            return
        cov = self.coverage
        a, col = _extract(self.before, now, cov)
        kind = "paint" if cov is not None and cov.max() > 0.5 else "effect"
        if re.search(r"post|finish|grade|grading|後製", self.current, re.I):
            kind = "post"
        H, W = now.shape[:2]
        self.layers.append(_to_layer(self.current, a, col, (W, H), kind))

    def finish(self, cv=None):
        """Close the running step (using ``cv`` / the attached canvas as the final image); return the layers."""
        if cv is not None and self.cv is None:
            self.attach(cv)
        if self.cv is not None:
            if self.current is None:
                self.mark("image")
                self.current = None
            else:
                self.final = self._snapshot()
                self._close(self.final)
                self.current = None
        if self in _ATTACHED:
            _ATTACHED.remove(self)
        names = {}
        for L in self.layers:                          # unique names (PSD / ORA tools key on them)
            n = names.get(L.name, 0)
            names[L.name] = n + 1
            if n:
                L.name = f"{L.name} ({n + 1})"
        if self.post == "regrade" and self.layers:
            self.layers = push_post(self.layers, self.final)
        if self.halos == "split" and self.layers:
            self.layers = split_halos(self.layers)
        return self.layers


_ATTACHED = []                            # recorders that receive paint coverage from Canvas.paint / paste


# module-level recorder that follows render.py's log() and the first canvas of the target size
_REC = {"on": False, "rec": None, "target": None}


def enable(target_size=None):
    """Start recording. The first Canvas created with ``target_size=(w, h)`` (any, if None) is followed."""
    _REC.update(on=True, rec=LayerRecorder(), target=tuple(target_size) if target_size else None)


def active():
    return _REC["rec"] if _REC["on"] else None


def _track(cv):
    """Called by Canvas.__init__."""
    rec = _REC["rec"]
    if not _REC["on"] or rec.cv is not None:
        return
    if (_REC["target"] is None and cv.w * cv.h > 1) or _REC["target"] == (cv.w, cv.h):
        rec.attach(cv)


def _note(cv, alpha, region=None):
    """Called by Canvas.paint / paste in normal mode."""
    for rec in _ATTACHED:
        rec.note(cv, alpha, region)


def mark(name, cv=None):
    """Start a new layer called ``name`` (no-op unless recording). render.py's ``log`` calls this."""
    if _REC["on"]:
        _REC["rec"].mark(name, cv)


def take(cv=None):
    """Stop recording and return the layers (bottom → top)."""
    rec = _REC["rec"]
    _REC.update(on=False, rec=None, target=None)
    if rec is None:
        return []
    out = rec.finish(cv)
    if rec in _ATTACHED:
        _ATTACHED.remove(rec)
    return out


# ------------------------------------------------------------------ compositing

def composite(stack, hidden=(), only_visible=True):
    """Composite layers bottom → top with normal blending → (H, W, 3) float32. ``hidden``: names to skip."""
    W, H = stack[0].size
    out = np.zeros((H, W, 3), F32)
    for L in stack:
        if L.name in hidden or (only_visible and not L.visible) or L.rgba.size == 0:
            continue
        x0, y0, x1, y1 = L.bbox
        src = L.rgba.astype(F32) / 255.0
        a = src[..., 3:4] * float(L.opacity)
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - a) + src[..., :3] * a
    return out


def _fit_lut(B, F, n=17, sample=200000, seed=0):
    """Fit a colour → colour map (n^3 LUT) from before / after pixel pairs; empty cells filled smoothly."""
    P = B.reshape(-1, 3)
    Q = F.reshape(-1, 3)
    if len(P) > sample:
        sel = np.random.default_rng(seed).choice(len(P), sample, replace=False)
        P, Q = P[sel], Q[sel]
    idx = np.clip(np.round(P * (n - 1)).astype(np.int64), 0, n - 1)
    flat = (idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2]
    cnt = np.bincount(flat, minlength=n ** 3).astype(np.float64)
    acc = np.stack([np.bincount(flat, Q[:, c], minlength=n ** 3) for c in range(3)], -1)
    g = np.linspace(0, 1, n)
    ident = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    off = np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1) - ident, 0.0)   # learn the *offset*
    w = cnt.reshape(n, n, n)
    o = off.reshape(n, n, n, 3) * w[..., None]
    for _ in range(6):                                  # diffuse into empty cells (normalised blur)
        w2, o2 = w.copy(), o.copy()
        for ax in range(3):
            w2 = (w2 + np.roll(w2, 1, ax) + np.roll(w2, -1, ax)) / 3
            o2 = (o2 + np.roll(o2, 1, ax) + np.roll(o2, -1, ax)) / 3
        empty = w <= 0
        w = np.where(empty, w2, w)
        o = np.where(empty[..., None], o2, o)
    off = np.where(w[..., None] > 1e-9, o / np.maximum(w[..., None], 1e-9), 0.0)
    return (ident.reshape(n, n, n, 3) + off).astype(F32)


def _apply_lut(lut, C):
    n = lut.shape[0]
    x = np.clip(C, 0, 1) * (n - 1)
    i0 = np.clip(np.floor(x).astype(np.int64), 0, n - 2)
    f = (x - i0).astype(F32)
    out = np.zeros(C.shape, F32)
    for dx in (0, 1):
        wx = f[..., 0] if dx else 1 - f[..., 0]
        for dy in (0, 1):
            wy = f[..., 1] if dy else 1 - f[..., 1]
            for dz in (0, 1):
                wz = f[..., 2] if dz else 1 - f[..., 2]
                out += (wx * wy * wz)[..., None] * lut[i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz]
    return out


def _smooth(a, sigma):
    try:
        from .filters import gaussian_blur
        return gaussian_blur(a, sigma)
    except Exception:                                   # noqa: BLE001
        return a


def push_post(stack, final=None, tol=1.5 / 255):
    """Fold the trailing ``post`` layer(s) into the layers below instead of keeping them as a layer.

    A global post step (grading, soft_clip, bloom, vignette, grain) extracted as a layer carries a faint copy of
    every object (hide the candle and a translucent candle stays behind). Instead the post step is modelled as
    content-independent parts and applied to EVERY layer's colours:
      1. a colour → colour LUT fitted on before / after pixels (grading, soft_clip, contrast, saturation, tone);
      2. a large-scale smooth gain (vignette, broad light) — the ratio of heavily blurred after / LUT(before).
    Neither depends on *which* object is at a pixel, so no object's shape is printed into another layer. Whatever
    remains (grain, small bloom halos) goes into a thin "post (residual)" layer so the stack still reproduces the
    final image exactly.
    """
    i0 = len(stack)
    while i0 > 0 and stack[i0 - 1].kind == "post":
        i0 -= 1
    if i0 == len(stack) or i0 == 0:
        return stack
    W, H = stack[0].size
    F = composite(stack, only_visible=False) if final is None else _clamp(getattr(final, "buf", final))
    lower = stack[:i0]
    B = composite(lower, only_visible=False)
    lut = _fit_lut(B, F)
    TB = _apply_lut(lut, B)
    sig = 0.06 * max(W, H)
    e = 0.02
    V = np.clip((_smooth(F, sig) + e) / (_smooth(TB, sig) + e), 0.25, 4.0).astype(F32)
    for L in lower:
        if not L.rgba.size:
            continue
        x0, y0, x1, y1 = L.bbox
        col = _apply_lut(lut, L.rgba[..., :3].astype(F32) / 255.0) * V[y0:y1, x0:x1]
        rgb8 = np.round(_clamp(col) * 255.0).astype(np.uint8)
        L.rgba = np.concatenate([rgb8, L.rgba[..., 3:4]], -1)
        L.rgba[L.rgba[..., 3] == 0, :3] = 0
    out = list(lower)
    _spread_noise(out, F)
    _assign_to_owners(out, F)
    comp = composite(out, only_visible=False)
    if float(np.abs(comp - F).max()) > tol:
        a, col = _extract(comp, F)
        out.append(_to_layer("post (residual)", a, col, (W, H), "post"))
    return out


def _spread_noise(stack, F, min_alpha=0.5):
    """Grain belongs to every layer, not only to the one that is visible. Left to _assign_to_owners, the grain of the
    final image would go into the visible layer only, so a lower layer would have a grain-free hole shaped like the
    object above it — a ghost when that object is hidden. Take the pixel-scale part of the remaining error (error
    minus its 3x3 median: grain, not shapes); every layer gets the grain seen where IT is the visible layer, and
    fills the places it is covered with its own grain shifted in from nearby (grain strength depends on the colour,
    so each layer keeps its own). The composite moves by exactly the grain where a layer is visible."""
    try:
        from scipy.ndimage import median_filter
    except Exception:                                   # noqa: BLE001
        return
    W, H = stack[0].size
    d = F - composite(stack, only_visible=False)
    N = (d - median_filter(d, size=(3, 3, 1))).astype(F32)
    if float(np.abs(N).max()) < 1.0 / 255:
        return
    owner = np.full((H, W), -1, np.int32)
    edge = np.zeros((H, W), bool)                         # near edges the pixel-scale error is fit error, not grain
    for k, L in enumerate(stack):
        if not L.rgba.size:
            continue
        x0, y0, x1, y1 = L.bbox
        a = L.rgba[..., 3].astype(F32) / 255.0
        owner[y0:y1, x0:x1] = np.where(a >= min_alpha, k, owner[y0:y1, x0:x1])
        if k:
            g = np.zeros_like(a, bool)
            g[1:] |= np.abs(np.diff(a, axis=0)) > 0.1
            g[:, 1:] |= np.abs(np.diff(a, axis=1)) > 0.1
            edge[y0:y1, x0:x1] |= g
    edge = _dilate(edge, 2)
    for k, L in enumerate(stack):
        if not L.rgba.size:
            continue
        mine = (owner == k) & ~edge
        if not mine.any():
            continue
        Nk = np.where(mine[..., None], N, 0).astype(F32)
        have = mine.copy()
        x0, y0, x1, y1 = L.bbox
        need = np.zeros((H, W), bool)
        need[y0:y1, x0:x1] = L.rgba[..., 3] > 0
        for r in (5, 11, 23, 47, 97, 199, 401):
            for sh in ((r, 0), (0, r), (-r, 0), (0, -r), (r, r), (-r, -r), (r, -r), (-r, r)):
                hole = need & ~have
                if not hole.any():
                    break
                src_ok = np.roll(have, sh, (0, 1)) & hole
                Nk[src_ok] = np.roll(Nk, sh, (0, 1))[src_ok]
                have |= src_ok
        col = L.rgba[..., :3].astype(F32) / 255.0 + Nk[y0:y1, x0:x1]
        rgb8 = np.round(_clamp(col) * 255.0).astype(np.uint8)
        L.rgba = np.concatenate([rgb8, L.rgba[..., 3:4]], -1)
        L.rgba[L.rgba[..., 3] == 0, :3] = 0


def _assign_to_owners(stack, F, min_alpha=0.5):
    """Put the remaining per-pixel error F - composite into the layer that is *seen* at that pixel: the topmost
    layer with alpha >= ``min_alpha`` there, scaled by its visible weight (its alpha x the transmittance of the
    layers above). The correction then belongs to the object: hide it and the correction goes with it — no ghost."""
    W, H = stack[0].size
    d = F - composite(stack, only_visible=False)
    T = np.ones((H, W), F32)                             # transmittance of the layers above
    owned = np.zeros((H, W), bool)
    for L in reversed(stack):
        if not L.rgba.size:
            continue
        x0, y0, x1, y1 = L.bbox
        a = L.rgba[..., 3].astype(F32) / 255.0
        t = T[y0:y1, x0:x1]
        own = (a >= min_alpha) & ~owned[y0:y1, x0:x1]
        if own.any():
            wgt = np.maximum(t * a, 1e-3)[..., None]
            col = L.rgba[..., :3].astype(F32) / 255.0
            new = np.where(own[..., None], col + d[y0:y1, x0:x1] / wgt, col)
            rgb8 = np.round(_clamp(new) * 255.0).astype(np.uint8)
            L.rgba = np.concatenate([rgb8, L.rgba[..., 3:4]], -1)
            owned[y0:y1, x0:x1] |= own
        T[y0:y1, x0:x1] = t * (1.0 - a)


def _dilate(m, r):
    """Binary dilation by a (2r+1) square, numpy only."""
    out = m.copy()
    H, W = m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
            xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
            out[yd, xd] |= m[ys, xs]
    return out


def _fill_holes(m):
    """Fill enclosed holes of a binary mask (flood fill from the border, Pillow)."""
    from PIL import ImageDraw
    H, W = m.shape
    im = Image.new("L", (W + 2, H + 2), 0)
    im.paste(Image.fromarray((m * 255).astype(np.uint8)), (1, 1))
    ImageDraw.floodfill(im, (0, 0), 128)
    return np.asarray(im)[1:-1, 1:-1] != 128


def split_halos(stack, solid=0.96, aa=2, min_px=200):
    """Move each painted object's soft halo (drop shadow, outer glow, feathered fringe) into its own layer,
    placed directly below the object and named "<object> (halo)".

    The object keeps its solid body, its anti-aliased edge (``aa`` px around the solid part) and anything inside
    its outline (translucent glass, glows in letter counters); soft pixels outside that are the halo. Alpha is
    only partitioned, so the stack still composites to the same image. Object and halo share a ``group`` (a
    folder in PSD / ORA, "group" in layers.json): hide or move the group and the shadow goes with it, while the
    object itself stays crisp and the shadow can be restyled on its own (as with a layer style)."""
    out = []
    for L in stack:
        if L.kind != "paint" or not L.rgba.size:
            out.append(L)
            continue
        a = L.rgba[..., 3].astype(F32) / 255.0
        core = a >= solid
        if not core.any():
            out.append(L)
            continue
        body = _fill_holes(_dilate(core, aa))
        halo = (a > 0) & ~body
        n = int(halo.sum())
        if n < max(min_px, 0.02 * a.size) and n < 0.5 * int(body.sum()):
            out.append(L)
            continue
        h_rgba = L.rgba.copy()
        h_rgba[~halo] = 0
        o_rgba = L.rgba.copy()
        o_rgba[halo] = 0
        x0, y0 = L.bbox[:2]
        for nm, rgba, kind in ((f"{L.name} (halo)", h_rgba, "effect"), (L.name, o_rgba, L.kind)):
            ys, xs = np.nonzero(rgba[..., 3])
            if not len(xs):
                continue
            bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            out.append(Layer(nm, np.ascontiguousarray(rgba[by0:by1, bx0:bx1]),
                             (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1), L.size, L.visible, L.opacity, kind,
                             dict(L.meta), group=L.group or L.name))
    return out


def reconcile(stack, final, name="final fixes", tol=1.5 / 255):
    """If compositing ``stack`` does not reproduce ``final`` (e.g. the scene returned another canvas), append a
    see-through correction layer so it does. Returns the stack and the max error before the fix."""
    final = _clamp(getattr(final, "buf", final))
    comp = composite(stack, only_visible=False)
    err = float(np.abs(comp - final).max())
    if err > tol:
        a, col = _extract(comp, final)
        W, H = stack[0].size
        stack.append(_to_layer(name, a, col, (W, H), "post"))
    return stack, err


# ------------------------------------------------------------------ PNG stack + manifest

def _slug(s):
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.U).strip("_")
    return s or "layer"


def save_pngs(stack, folder, final=None):
    """One PNG per layer (cropped, with its offset in layers.json) + layers.json manifest + composite.png."""
    os.makedirs(folder, exist_ok=True)
    W, H = stack[0].size
    entries = []
    for i, L in enumerate(stack):
        fn = f"{i:02d}_{_slug(L.name)}.png"
        if L.rgba.size:
            L.to_pil().save(os.path.join(folder, fn))
        else:
            fn = None
        e = dict(index=i, name=L.name, file=fn, x=L.bbox[0], y=L.bbox[1], bbox=list(L.bbox),
                 visible=L.visible, opacity=L.opacity, kind=L.kind, blend="normal")
        if L.group:
            e["group"] = L.group
        entries.append(e)
    comp = composite(stack) if final is None else final
    Image.fromarray(np.round(_clamp(comp) * 255).astype(np.uint8), "RGB").save(os.path.join(folder, "composite.png"))
    with open(os.path.join(folder, "layers.json"), "w", encoding="utf-8") as f:
        json.dump(dict(width=W, height=H, order="bottom_to_top", layers=entries), f, ensure_ascii=False, indent=1)
    return folder


def load_pngs(folder):
    """Read a stack written by ``save_pngs`` (after a human edited / replaced / added PNGs)."""
    with open(os.path.join(folder, "layers.json"), encoding="utf-8") as f:
        man = json.load(f)
    W, H = man["width"], man["height"]
    stack = []
    for e in man["layers"]:
        if e.get("file") and os.path.exists(os.path.join(folder, e["file"])):
            im = np.asarray(Image.open(os.path.join(folder, e["file"])).convert("RGBA"))
            x0, y0 = int(e.get("x", e["bbox"][0])), int(e.get("y", e["bbox"][1]))
            # clip to the canvas (edited layers may have been moved / grown)
            h, w = im.shape[:2]
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
            im = im[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            bbox = (cx0, cy0, cx1, cy1)
        else:
            im, bbox = np.zeros((0, 0, 4), np.uint8), (0, 0, 0, 0)
        stack.append(Layer(e["name"], np.ascontiguousarray(im), bbox, (W, H), bool(e.get("visible", True)),
                           float(e.get("opacity", 1.0)), e.get("kind", "paint"), group=e.get("group")))
    return stack


# ------------------------------------------------------------------ OpenRaster (.ora: Krita, GIMP, MyPaint)

def _runs(stack):
    """Consecutive layers with the same group → [(group or None, [(index, Layer), ...]), ...] bottom → top."""
    runs = []
    for i, L in enumerate(stack):
        if runs and L.group and runs[-1][0] == L.group:
            runs[-1][1].append((i, L))
        else:
            runs.append((L.group, [(i, L)]))
    return runs


def save_ora(stack, path, final=None):
    W, H = stack[0].size
    comp = composite(stack) if final is None else final
    comp_im = Image.fromarray(np.round(_clamp(comp) * 255).astype(np.uint8), "RGB")

    def png_bytes(im):
        b = io.BytesIO()
        im.save(b, "PNG")
        return b.getvalue()

    def esc(s):
        return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    xml = [f'<?xml version="1.0" encoding="UTF-8"?>\n<image version="0.0.3" w="{W}" h="{H}">\n<stack>']
    files = []
    for g, run in reversed(_runs(stack)):                   # stack.xml lists the top layer first
        if g:
            xml.append(f' <stack name="{esc(g)}" composite-op="svg:src-over">')
        for i, L in reversed(run):
            if not L.rgba.size:
                continue
            src = f"data/{i:02d}.png"
            files.append((src, png_bytes(L.to_pil())))
            xml.append(f' <layer src="{src}" name="{esc(L.name)}" x="{L.bbox[0]}" y="{L.bbox[1]}" '
                       f'opacity="{L.opacity:.3f}" visibility="{"visible" if L.visible else "hidden"}" '
                       f'composite-op="svg:src-over"/>')
        if g:
            xml.append(" </stack>")
    xml.append("</stack>\n</image>\n")
    thumb = comp_im.copy()
    thumb.thumbnail((256, 256))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "image/openraster", compress_type=zipfile.ZIP_STORED)
        z.writestr("stack.xml", "\n".join(xml))
        for src, data in files:
            z.writestr(src, data)
        z.writestr("mergedimage.png", png_bytes(comp_im))
        z.writestr("Thumbnails/thumbnail.png", png_bytes(thumb))
    return path


# ------------------------------------------------------------------ Photoshop (.psd: Photoshop, Photopea, GIMP, Krita, Affinity)

def _packbits_rows(plane):
    """PackBits-compress each row of a (h, w) uint8 plane → (row byte counts, data)."""
    counts, chunks = [], []
    plane = np.ascontiguousarray(plane, np.uint8)
    for row in plane:
        n = len(row)
        b = row.tobytes()
        out = bytearray()
        change = np.flatnonzero(row[1:] != row[:-1]) + 1     # run boundaries (vectorised; loop only over long runs)
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [n]))
        long = (ends - starts) >= 3
        pos = 0
        for s, e in zip(starts[long].tolist(), ends[long].tolist()):
            while pos < s:                             # literal bytes before the run
                k = min(s, pos + 128)
                out.append(k - pos - 1)
                out += b[pos:k]
                pos = k
            while pos < e:                             # the run, ≤ 128 per packet
                k = min(e, pos + 128)
                out.append(257 - (k - pos) if k - pos >= 2 else 0)
                out.append(b[pos])
                pos = k
        while pos < n:
            k = min(n, pos + 128)
            out.append(k - pos - 1)
            out += b[pos:k]
            pos = k
        counts.append(len(out))
        chunks.append(bytes(out))
    return counts, b"".join(chunks)


def _channel_rle(plane):
    """Channel data block with RLE compression (2-byte header + row counts + data)."""
    if plane.size == 0:
        return struct.pack(">H", 0)
    counts, data = _packbits_rows(plane)
    return struct.pack(">H", 1) + struct.pack(f">{len(counts)}H", *counts) + data


def _channel_raw(plane):
    return struct.pack(">H", 0) + np.ascontiguousarray(plane, np.uint8).tobytes()


def save_psd(stack, path, final=None, rle=True):
    """Write an 8-bit RGB PSD with one pixel layer per Layer (bottom → top) + the merged image."""
    W, H = stack[0].size
    enc = _channel_rle if rle else _channel_raw
    records, chandata = [], []
    empty = np.zeros((0, 0), np.uint8)

    def record(name, rgba, bbox, visible=True, opacity=1.0, section=None):
        x0, y0, x1, y1 = bbox
        planes = [(-1, rgba[..., 3]), (0, rgba[..., 0]), (1, rgba[..., 1]), (2, rgba[..., 2])] \
            if rgba.size else [(-1, empty)] + [(c, empty) for c in range(3)]
        blocks = [(cid, enc(p)) for cid, p in planes]
        rec = struct.pack(">iiiiH", y0, x0, y1, x1, len(blocks))
        for cid, b in blocks:
            rec += struct.pack(">hI", cid, len(b))
        flags = 0 if visible else 0x02
        rec += b"8BIMnorm" + struct.pack(">BBBB", int(round(opacity * 255)), 0, flags, 0)
        name_b = name.encode("ascii", "replace")[:255]
        pas = bytes([len(name_b)]) + name_b
        pas += b"\0" * ((4 - len(pas) % 4) % 4)
        uni = name.encode("utf-16-be")
        luni = struct.pack(">I", len(name)) + uni
        luni += b"\0" * ((4 - len(luni) % 4) % 4)
        extra = struct.pack(">I", 0) + struct.pack(">I", 0) + pas + b"8BIMluni" + struct.pack(">I", len(luni)) + luni
        if section is not None:                        # 1 = open folder, 3 = bounding "</Layer group>" divider
            extra += b"8BIMlsct" + struct.pack(">II", 4, section)
        rec += struct.pack(">I", len(extra)) + extra
        records.append(rec)
        chandata.append(b"".join(b for _, b in blocks))

    nogeom = np.zeros((0, 0, 4), np.uint8)
    for g, run in _runs(stack):                        # PSD layer records go bottom → top; a folder is
        if g:                                          # [divider, children..., folder header]
            record("</Layer group>", nogeom, (0, 0, 0, 0), section=3)
        for _, L in run:
            record(L.name, L.rgba, L.bbox, L.visible, L.opacity)
        if g:
            record(g, nogeom, (0, 0, 0, 0), section=1)
    layer_info = struct.pack(">h", len(records)) + b"".join(records) + b"".join(chandata)
    if len(layer_info) % 2:
        layer_info += b"\0"
    lmi = struct.pack(">I", len(layer_info)) + layer_info + struct.pack(">I", 0)
    comp = composite(stack) if final is None else final
    c8 = np.round(_clamp(comp) * 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"8BPS" + struct.pack(">H6xHIIHH", 1, 3, H, W, 8, 3))
        f.write(struct.pack(">I", 0))                  # colour mode data
        f.write(struct.pack(">I", 0))                  # image resources
        f.write(struct.pack(">I", len(lmi)) + lmi)
        if rle:
            counts, datas = [], []
            for c in range(3):
                cc, d = _packbits_rows(c8[..., c])
                counts += cc
                datas.append(d)
            f.write(struct.pack(">H", 1) + struct.pack(f">{len(counts)}H", *counts) + b"".join(datas))
        else:
            f.write(struct.pack(">H", 0) + b"".join(np.ascontiguousarray(c8[..., c]).tobytes() for c in range(3)))
    return path


def save_all(stack, base, final=None, formats=("psd", "ora", "png")):
    """Write ``base.psd``, ``base.ora`` and ``base_layers/`` (PNG stack). Returns the written paths."""
    out = []
    if "psd" in formats:
        out.append(save_psd(stack, base + ".psd", final))
    if "ora" in formats:
        out.append(save_ora(stack, base + ".ora", final))
    if "png" in formats:
        out.append(save_pngs(stack, base + "_layers", final))
    return out
