"""Scripted reference pipeline (no model in the loop): the engine's ceiling.

Runs a style's stage sequence over the whole picture with default parameters, saves the composite after every stage
(01-sketch.png, 02-line.png, ...) and final.png, and a JSON with timings, stroke counts and quick scores.

    python -m pixelcanvas.pipeline <ref> <out_dir> [--style cel|ink|paint|photo|poster]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np
from PIL import Image

from .brush import apply_ops, new_layer
from .ops import Fill, Stroke, check_mark_rules, level_counts, mark_area, mark_level, op_bbox
from .stage_flat import flat_ops
from .stage_ink import ink_ops
from .stage_paint import ground_ops, paint_ops
from .stage_shade import blend_ops, shade_ops
from .stage_sketch import sketch_ops
from .trace import delta_e, edge_map, line_map, to_lab

__all__ = ["run_reference_pipeline", "guess_style", "STAGES", "quick_scores", "load_ref", "grain_local",
           "add_grain_local", "pass_level", "STAGE_LEVEL", "DEFAULT_LEVEL", "fidelity_metrics", "level_passes"]

LAYERS = ["paper", "flat", "shade", "paint", "line", "detail", "sketch"]

# style -> list of (stage name, layer, kind, params).  paint / detail "sizes" are passes of paint_ops (one
# command each); "L/n" = the long side / n.
# the fine detail pass only where lines, flats and shading still leave a clear residual (Delta E per cell)
# first the residual shape pass (a pass size above 3 px with brush "shapes": shapes.shape_ops, the small hard-edged
# shapes as outlined fills with sub-pixel edges), then fine liner strokes where the residual is still high
# "level": the finest detail level the stage goes to by default (its passes are levels 3 and 2: pass_level);
# levels 1 and 0 (hairlines, retouching: stage_levels.retouch_ops) follow when a call asks for them (level_passes)
FINE = {"sizes": (6.0, 2.5, 1.5), "brush": "shapes", "threshold": 5.0, "highlights": True, "level": 2}
# the same, decided per area (stage_local.py): in the areas the reference is textured the liner strokes take bristle /
# sponge tips and work to a finer tolerance; clean areas are as FINE
FINE_LOCAL = dict(FINE, tips="local")
SOFT = 0.3
AIR = {"sizes": (12, 6), "brush": "soft", "edge_aware": True}
# blending passes: after the shading (softens the steps between tone shapes) and at the finish (patchy paint)
# (only on pictures with soft shading: at least min_ramp of the smooth area is a gentle ramp; on hard-edged flat cel
# work blending only costs crispness)
BLEND_SHADE = {"layers": ("flat", "shade"), "radius": 16.0, "tol": 40.0, "min_ramp": 0.15}
BLEND_FINISH = {"layers": ("flat", "shade", "detail"), "radius": 6.0, "tol": 25.0, "min_ramp": 0.15}
FINISH_LOCAL = dict(BLEND_FINISH, grain="local")
# paint / photo detail, decided per area by the local character map (stage_local.py): thin light marks first, then
# strokes along the structure with bristle / sponge tips only where the reference is textured, the residual shape
# pass over them (tips "local": hard-edged flats, lettering, architecture, thin lines as outlined fills, crisp,
# after the strokes), liner strokes on what is left, texture marks only in the textured areas
PAINT_DETAIL = {"sizes": (4, 2.5), "brush": "round", "structure": True, "threshold": 3.5, "texture": True, "level": 2,
                "tips": "local", "highlights": True}
# the finish blend of paint / photo: the flat and ramp areas only (blend_ops leaves textured areas alone), so the
# stroke-by-stroke colour steps of a smooth passage melt and its grain comes from the grain pass
# (finish stages end with the grain, "local": only where the reference has pixel-fine grain, at each area's amount)
BLEND_PAINT = {"layers": ("paint", "detail"), "radius": 4.0, "tol": 20.0, "grain": "local"}
STAGES = {
    "cel": [("sketch", "sketch", "sketch", {}),
            ("line", "line", "ink", {"method": "auto"}),
            ("flat", "flat", "flat", {}),
            ("shade", "shade", "shade", {"levels": 4, "soft": SOFT, "air": "auto"}),
            ("blend", "shade", "blend", BLEND_SHADE),
            ("detail", "detail", "detail", FINE_LOCAL),
            ("finish", "detail", "blend", FINISH_LOCAL)],
    "ink": [("sketch", "sketch", "sketch", {}),
            ("line", "line", "ink", {"detail": 0.9, "method": "masses", "true_thin": False}),
            ("flat", "flat", "flat", {"ink_mask": True, "skip_dark": 35.0}),
            # tone and texture under the ink (on the paint layer, below the line layer): the pen lines stay crisp
            ("detail", "paint", "detail", {"sizes": (2.5, 1.5), "threshold": 8.0, "level": 2})],
    "poster": [("sketch", "sketch", "sketch", {"looseness": 0.3}),
               ("flat", "flat", "flat", {"line_mask": "none"}),
               ("shade", "shade", "shade", {"levels": 4}),
               ("blend", "shade", "blend", BLEND_SHADE),
               ("line", "line", "ink", {"detail": 0.4, "method": "auto"}),
               ("detail", "detail", "detail", FINE_LOCAL),
               ("finish", "detail", "blend", FINISH_LOCAL)],
    "paint": [("sketch", "sketch", "sketch", {"looseness": 0.7}),
              ("block-in", "paint", "paint", {"sizes": ("L/20", "L/40"), "brush": "flat"}),
              # soft ramps airbrushed in their own friskets, strokes elsewhere (tips "local", stage_paint._local_pass)
              ("forms", "paint", "paint", {"sizes": (12, 6), "brush": "flat", "tips": "local"}),
              ("detail", "detail", "detail", PAINT_DETAIL),
              ("finish", "detail", "blend", BLEND_PAINT)],
    "photo": [("sketch", "sketch", "sketch", {"looseness": 0.4}),
              ("block-in", "paint", "paint", {"sizes": ("L/16", "L/32"), "brush": "round"}),
              # smooth tone (sky, the dune's gradients) with the airbrush, clear of the hard edges
              ("forms", "paint", "air", {"sizes": (64, 32, 16, 8)}),
              # texture: strokes along the structure, bristle / sponge tips
              ("detail", "detail", "detail", PAINT_DETAIL),
              ("finish", "detail", "blend", BLEND_PAINT)],
}


def to_srgb(im):
    """Convert a PIL image with an embedded colour profile (Adobe RGB, Display P3, ...) to sRGB RGB; transparency
    is composited over white first (project.flatten_alpha, the shell's and the judge's rule)."""
    from .project import flatten_alpha
    im = flatten_alpha(im)
    icc = im.info.get("icc_profile")
    if icc:
        try:
            import io
            from PIL import ImageCms
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            im = ImageCms.profileToProfile(im.convert("RGB"), src, dst, renderingIntent=0, outputMode="RGB")
        except Exception:
            pass
    return im.convert("RGB")


def load_ref(path, max_side=1600):
    im = to_srgb(Image.open(path))
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return np.asarray(im).copy()


def paper_color(ref):
    ring = np.concatenate([ref[:2].reshape(-1, 3), ref[-2:].reshape(-1, 3), ref[:, :2].reshape(-1, 3),
                           ref[:, -2:].reshape(-1, 3)]).astype(np.float32)
    med = np.median(ring, 0)
    if np.mean(np.abs(ring - med).sum(1) < 18) > 0.85:
        return tuple(int(v) for v in med)
    return (255, 255, 255)


def composite(layers, order=LAYERS, opacity=None):
    """Minimal 'over' compositing of straight-alpha float layers, bottom to top. Returns uint8 RGB."""
    opacity = opacity or {}
    H, W = next(iter(layers.values())).shape[:2]
    out = np.zeros((H, W, 3), np.float32)
    for name in order:
        L = layers.get(name)
        if L is None:
            continue
        # only the rows / columns where the layer has paint change (out += (rgb - out) * a, in place)
        al = L[..., 3]
        rows = np.flatnonzero(al.max(1) > 0)
        if not len(rows):
            continue
        cols = np.flatnonzero(al[rows[0]:rows[-1] + 1].max(0) > 0)
        sub = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        a = L[sub][..., 3:4]
        op = float(opacity.get(name, 1.0))
        if op != 1.0:
            a = a * np.float32(op)
        o = out[sub]
        d = np.subtract(L[sub][..., :3], o)
        np.multiply(d, a, out=d)
        o += d
    np.multiply(out, 255.0, out=out)
    np.rint(out, out=out)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


# ------------------------------------------------------------------------------------------------ scores
def quick_scores(ref, img):
    de = delta_e(cv2.GaussianBlur(ref, (0, 0), 1.0), cv2.GaussianBlur(img, (0, 0), 1.0))
    e1 = edge_map(ref)
    e2 = edge_map(img)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    d1 = cv2.dilate(e1.astype(np.uint8), k) > 0
    d2 = cv2.dilate(e2.astype(np.uint8), k) > 0
    prec = (e2 & d1).sum() / max(e2.sum(), 1)
    rec = (e1 & d2).sum() / max(e1.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    hp1 = ref.astype(np.float32) - cv2.GaussianBlur(ref, (0, 0), 1.5).astype(np.float32)
    hp2 = img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.float32)
    detail = float(np.abs(hp2).mean() / max(np.abs(hp1).mean(), 1e-6))
    return {"mean_dE": round(float(de.mean()), 3), "p95_dE": round(float(np.percentile(de, 95)), 2),
            "edge_f1": round(float(f1), 4), "hf_ratio": round(detail, 3)}


# ------------------------------------------------------------------------------------------------ style guess
def style_features(ref) -> dict:
    """The measurements ``guess_style`` decides on (at most 600 px on the long side)."""
    if isinstance(ref, (str, os.PathLike)):
        ref = load_ref(ref)
    ref = np.asarray(ref)
    sc = min(1.0, 600.0 / max(ref.shape[:2]))
    im = cv2.resize(ref, (int(ref.shape[1] * sc), int(ref.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    lab = to_lab(im)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    lm = line_map(im)
    line_frac = float(lm.mask.mean())
    # flatness: share of pixels whose 5x5 neighbourhood is nearly constant
    L = lab[..., 0]
    var = cv2.blur(L * L, (5, 5)) - cv2.blur(L, (5, 5)) ** 2
    flat_frac = float((var < 2.0).mean())
    # texture: high-frequency energy outside lines
    hp = np.abs(L - cv2.GaussianBlur(L, (0, 0), 1.5))
    tex = float(hp[~lm.mask].mean()) if (~lm.mask).any() else 0.0
    # colour count: distinct quantised colours covering 90 % of pixels
    q = (im // 24).reshape(-1, 3).astype(np.int32)
    keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    counts = np.sort(counts)[::-1]
    ncol = int(np.searchsorted(np.cumsum(counts) / counts.sum(), 0.9) + 1)
    white = float((L > 97).mean())
    # dark outlines: share of strong edges that are the rim of a thin dark line (drawn pictures outline their
    # shapes; a photograph's edges are steps between areas)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0) / 8.0
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1) / 8.0
    g = np.sqrt(gx * gx + gy * gy)
    strong = g > 6.0
    dark_line = (cv2.morphologyEx(L, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) - L) > 18.0
    near_line = cv2.dilate(dark_line.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    outlined = float((strong & near_line).sum() / max(int(strong.sum()), 1))
    # fine grain: the noise of the smooth parts (sensor noise and JPEG grain in a photograph; a painting's smooth
    # passages and a cel picture's flats are cleaner) and how saturated the picture is
    smooth = cv2.GaussianBlur(g, (0, 0), 2.0) < 1.5
    grain = float(np.median(hp[smooth])) if smooth.any() else 0.0
    sat = float(np.percentile(chroma, 90))
    # how fine the noise of the smooth parts is: the 1 px residual over the 2.5 px residual (a camera's noise is
    # pixel-fine, so the ratio is high; paint, airbrush and digital shading vary over several pixels)
    gg = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.GaussianBlur(gg, (0, 0), 2.0)
    gsm = np.sqrt(cv2.Sobel(gb, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gb, cv2.CV_32F, 0, 1) ** 2) / 8.0
    sm = gsm < 1.0
    n1, n25 = fine_noise(im, sm, 1.0), fine_noise(im, sm, 2.5)
    return {"line_frac": line_frac, "flat": flat_frac, "tex": tex, "ncol": float(ncol), "white": white,
            "chroma95": float(np.percentile(chroma, 95)), "outlined": outlined, "grain": grain, "sat90": sat,
            "fine_ratio": float(n1 / max(n25, 1e-3))}


PHOTO_FINE = 0.64      # fine_ratio from which a natural-looking picture is a photograph


def guess_style(ref) -> str:
    """Propose a style from line density, colour count and texture: cel | ink | paint | photo | poster."""
    f = style_features(ref)
    if f["chroma95"] < 5.0:
        return "ink"
    camera = f["fine_ratio"] >= PHOTO_FINE and f["grain"] >= 0.25     # pixel-fine noise, and noise at all
    # a natural picture: no paper showing (a photograph may show some blown-out sky or wall) and no large flats
    natural = (f["white"] < 0.05 or (camera and f["white"] < 0.12)) and (f["flat"] < 0.55 or f["ncol"] >= 20)
    if natural:
        # the pixel-fine noise of a camera vs paint (brush texture, soft digital painting)
        return "photo" if camera else "paint"
    if f["ncol"] <= 16 and f["line_frac"] < 0.035 and f["white"] < 0.4:
        return "poster"
    return "cel"


def soft_ramp_share(ref):
    """Share of the reference's smooth area that is a gentle colour ramp (soft shading, blush, sky)."""
    from .stage_shade import smooth_area
    a = smooth_area(ref)
    lab = cv2.GaussianBlur(to_lab(ref), (0, 0), 2.0)
    g = np.zeros(a.shape, np.float32)
    for c in range(3):
        g += cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1) ** 2
    g = np.sqrt(g) / 8.0
    return float((a & (g > 0.25) & (g < 4.0)).sum() / max(int(a.sum()), 1))


def fine_noise(img, area=None, sigma=1.0):
    """Std (0..255) of the fine residual (image minus its ``sigma`` px blur) of the luminance, over ``area``."""
    g = cv2.cvtColor(np.asarray(img)[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = g - cv2.GaussianBlur(g, (0, 0), sigma)
    v = r[area] if area is not None else r.ravel()
    return float(np.sqrt(np.mean(v ** 2))) if v.size else 0.0


GRAIN_SIZES = (1.0, 1.4, 2.0, 3.0)


def grain_params(ref, canvas):
    """(amount, size) of ``fx grain`` that gives the canvas the reference's fine noise in its smooth areas: the
    grain size whose noise has the reference's ratio of 1 px to 2.5 px residual (how fine the grain is), the amount
    that makes up the missing noise at that size (both measured, statistics only).  (0, None) = no grain."""
    from .stage_shade import smooth_area
    area = smooth_area(ref, None, edge_px=3.0, max_grad=6.0)
    if area.sum() < 1000:
        return 0.0, None
    nr, nc = fine_noise(ref, area), fine_noise(canvas, area)
    need = np.sqrt(max(nr * nr - nc * nc, 0.0))
    if need < 0.3:
        return 0.0, None
    # the reference's own noise, with the canvas's (smooth paint) share taken out, at two scales
    r1 = need
    r2 = np.sqrt(max(fine_noise(ref, area, 2.5) ** 2 - fine_noise(canvas, area, 2.5) ** 2, 0.0))
    target = r1 / max(r2, 1e-3)
    best = None
    for sz in GRAIN_SIZES:
        noise = add_grain(canvas, 1.0, size=sz).astype(np.float32) - canvas.astype(np.float32) + 128.0
        n1 = fine_noise(noise, area)
        n2 = fine_noise(noise, area, 2.5)
        err = abs(np.log((n1 / max(n2, 1e-3)) / target))
        if best is None or err < best[0]:
            best = (err, sz, n1)
    _, sz, n1 = best
    return float(np.clip(need / max(n1, 1e-3), 0.0, 1.0)), sz


def grain_local(ref, canvas):
    """(amount map float32 [H,W] in 0..1, size) of a grain that gives each smooth area of the canvas the
    reference's own fine noise there: zero where the reference is clean (a flat black background, a cel flat, a clean
    gradient) and in textured areas (their strokes carry the texture); the size is the one ``grain_params`` measures
    for the whole picture.  (None, None) = no grain anywhere."""
    from .stage_local import char_map, noise_map
    amount, size = grain_params(ref, canvas)
    if not size:
        return None, None
    cm = char_map(ref)
    nr = noise_map(ref, ref)
    nc = noise_map(canvas, ref)
    need = np.sqrt(np.maximum(nr * nr - nc * nc, 0.0))
    noise = add_grain(np.full_like(np.asarray(canvas)[..., :3], 128), 1.0, size=size).astype(np.float32)
    unit = fine_noise(np.clip(noise, 0, 255).astype(np.uint8))
    amap = np.clip(need / max(unit, 1e-3), 0.0, 1.0)
    # only where the map finds grain in the reference (blurred presence, so the amount fades out at an area's edge)
    pres = np.clip(cm.grain / 0.45, 0.0, 1.0)
    amap = cv2.GaussianBlur((amap * pres).astype(np.float32), (0, 0), 3.0)
    amap[amap < 0.02] = 0.0
    if amap.max() <= 0:
        return None, None
    return amap.astype(np.float32), size


def add_grain_local(img, amap, size, seed=1):
    """``fx grain`` at amount 1, faded per pixel by ``amap`` (the effect is linear in its amount)."""
    full = add_grain(img, 1.0, seed=seed, size=size).astype(np.float32)
    base = np.asarray(img).astype(np.float32)
    return np.clip(np.rint(base + (full - base) * amap[..., None]), 0, 255).astype(np.uint8)


def grain_amount(ref, canvas):
    """``fx grain`` amount that gives the canvas the reference's fine noise in its smooth areas (0 = none)."""
    return grain_params(ref, canvas)[0]


def add_grain(img, amount, seed=1, size=None):
    from .fx import apply_effect
    out = apply_effect("grain", np.asarray(img, np.float32) / 255.0, amount, seed=seed, size=size)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------ pipeline
def _size(v, H, W):
    if isinstance(v, str) and v.startswith("L/"):
        return max(H, W) / float(v[2:])
    return float(v)


def level_passes(params, H, W, max_level=None):
    """[(size px or None, brush, level)] of a paint / detail stage down to ``max_level`` (default: the stage's own
    "level", else 2): its size passes (``pass_level`` each) coarser than or at that level, then ``(None,
    "retouch", 1)`` and ``(None, "retouch", 0)`` when the call goes that fine (``stage_levels.retouch_ops(ref, mask,
    canvas, level=1|0)``).  "detail to level 2" on one region and "to level 0" on another are two calls."""
    lv = int(params.get("level", 2) if max_level is None else max_level)
    out = [(s_, b_, pass_level(s_, b_)) for s_, b_ in stage_passes(params, H, W)]
    out = [p_ for p_ in out if p_[2] >= lv]
    out += [(None, "retouch", k) for k in (1, 0) if k >= lv]
    return out


def stage_passes(params, H, W, work_scale=1):
    """[(size px, brush)] of a paint / detail stage: its ``sizes`` with the stage's ``brush`` above 3 px and the
    liner at 3 px and below.  On a working canvas ``work_scale`` x the picture, sizes above 3 px are that many times
    larger (the same marks on the delivered picture); the fine sizes stay in working px (finer marks)."""
    out = []
    for sz in params.get("sizes", ()):
        size = _size(sz, H, W)
        if work_scale > 1 and not isinstance(sz, str) and size > 3:
            size *= work_scale
        br = params.get("brush", "round") if size > 3 or params.get("brush") == "soft" else "liner"
        out.append((size, br))
    return out


SMALL_MARK = 20.0      # px^2: a mark smaller than this is a fine mark for the small-mark share
MAX_CALLS = 6          # a paint / detail pass is called again while a call runs out of its time budget


def _size_chunks(ops):
    """Split ops, in order, into runs of small (< SMALL_MARK px^2) and larger marks; an airbrush op is split into
    its larger and its small dabs (its dabs run coarse to fine, so the order holds)."""
    from .ops import Spray
    out = []
    for op in ops:
        if isinstance(op, Spray):
            a = np.pi * op.size.astype(np.float64) ** 2 / 4.0
            parts = []
            for small in (False, True):
                sel = (a < SMALL_MARK) == small
                if sel.any():
                    parts.append((small, Spray(pts=op.pts[sel], size=op.size[sel], colors=op.colors[sel],
                                               opacity=op.opacity[sel], polygons=op.polygons, mask_rle=op.mask_rle,
                                               feather=op.feather, seed=op.seed, own=op.own)))
        else:
            parts = [(mark_area(op) < SMALL_MARK, op)]
        for small, o in parts:
            if out and out[-1][0] == small:
                out[-1][1].append(o)
            else:
                out.append((small, [o]))
    return out


def _with_grain(img, grain):
    if not grain.get("amount"):
        return img
    if grain.get("map") is not None:
        return add_grain_local(img, grain["map"], grain["size"])
    return add_grain(img, grain["amount"], size=grain["size"])


STREAK_MAX = 4.0       # px (working): widest streak; with fine="streaks" a detail pass up to this many delivered px
DEFAULT_LEVEL = 2      # run_reference_pipeline's finest level by default (the plan as the shell runs it today)
# detail level of each stage (paint / detail stages: per pass, ``pass_level``)
STAGE_LEVEL = {"sketch": 5, "block-in": 5, "flat": 5, "forms": 4, "shade": 4, "blend": 4, "line": 3, "detail": 2,
               "finish": 2}


def pass_level(size, brush) -> int:
    """Detail level of a paint / detail pass: the residual shape pass and passes above 3 px (strokes with the shape
    pass under tips "local") are level 3 (edges), from 8 px level 4, from 32 px level 5; 1.5-3 px level 2, finer 1."""
    size = float(size)
    if size >= 32:
        return 5
    if size >= 8 and brush != "shapes":
        return 4
    if size > 3 or brush == "shapes":
        return 3
    return 2 if size >= 1.5 else 1


def _laplacian_bands(img, n=5):
    g = cv2.cvtColor(np.asarray(img)[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = []
    for _ in range(n):
        d = cv2.pyrDown(g)
        u = cv2.pyrUp(d, dstsize=(g.shape[1], g.shape[0]))
        out.append(float(np.sqrt(np.mean((g - u) ** 2))))
        g = d
    return out


def fidelity_metrics(ref, img) -> dict:
    """mean / max Delta E, PSNR (dB, RGB), SSIM (grey) of ``img`` against ``ref``, and the energy ratio
    (redraw / reference) of the Laplacian pyramid bands, finest (1-2 px) first."""
    ref = np.asarray(ref)[..., :3]
    img = np.asarray(img)[..., :3]
    de = delta_e(ref, img)
    mse = float(np.mean((ref.astype(np.float64) - img.astype(np.float64)) ** 2))
    psnr = 99.0 if mse <= 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)
    try:
        from skimage.metrics import structural_similarity
        ssim = float(structural_similarity(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                                           data_range=255))
    except Exception:  # noqa: BLE001
        ssim = None
    br, bi = _laplacian_bands(ref), _laplacian_bands(img)
    return {"mean_dE": round(float(de.mean()), 4), "max_dE": round(float(de.max()), 3), "psnr": round(float(psnr), 2),
            "ssim": None if ssim is None else round(ssim, 4),
            "bands": [round(b / max(a, 1e-6), 3) for a, b in zip(br, bi)]}


def _touched(before, after):
    return float(np.any(np.abs(after - before) > 0.5 / 255, -1).mean())


RETOUCH_TILE = 32      # px (delivered): tile of the retouch share


def tile_retouch_ops(ref_d, img_d, scale, share):
    """Level-0 retouching on the ``share`` of 32 px tiles (delivered size) with the largest mean residual: one dab
    of ``scale`` x ``scale`` working px per delivered pixel still off by more than 0.5 Delta E, in the reference's
    colour.  Returns ([Retouch], number of tiles)."""
    from .ops import Retouch
    Hd, Wd = ref_d.shape[:2]
    de = delta_e(ref_d, img_d)
    T = RETOUCH_TILE
    ty, tx = (Hd + T - 1) // T, (Wd + T - 1) // T
    pad = np.zeros((ty * T, tx * T))
    pad[:Hd, :Wd] = de
    tile_de = pad.reshape(ty, T, tx, T).mean((1, 3))
    k = int(round(share * tile_de.size))
    if k <= 0:
        return [], 0
    thr = np.sort(tile_de.ravel())[::-1][k - 1]
    pick = tile_de >= thr
    sel = np.kron(pick, np.ones((T, T), bool))[:Hd, :Wd] & (de > 0.5)
    ys, xs = np.nonzero(sel)
    if not len(ys):
        return [], int(pick.sum())
    s = int(scale)
    return [Retouch(x=xs * s, y=ys * s, n=s, size=s, colors=ref_d[ys, xs])], int(pick.sum())


AUTO_SCALE = ((400, 4), (700, 3))   # long side (px) up to which the working scale is 4, 3; above: 2


def auto_work_scale(W, H) -> int:
    """The working scale for a picture W x H (``work_scale="auto"``): 4 up to a 400 px long side, 3 up to 700, else 2,
    so a 2 working-px mark stays finer than the picture's smallest features and an 850 px picture runs at 2x."""
    L = max(int(W), int(H))
    for lim, s in AUTO_SCALE:
        if L <= lim:
            return s
    return 2


FINISH_TILE = 32        # px: the finishing sharpen's amount is fitted per tile of this size (smoothly blended)
FINISH_SIGMA = 0.8      # px: the sharpen's radius (Gaussian sigma) on the delivered picture
FINISH_MAX = 1.5        # the largest sharpen amount


def finish_sharpen(img, ref, tile=FINISH_TILE, sigma=FINISH_SIGMA, amax=FINISH_MAX):
    """The finishing sharpen (an unsharp mask on the luminance only, as ``fx sharpen``), its amount fitted to the
    reference per ``tile`` px (least squares, 0..amax, blended smoothly between tiles), uint8 RGB."""
    Y = cv2.cvtColor(np.asarray(img)[..., :3], cv2.COLOR_RGB2YCrCb).astype(np.float32)
    R = cv2.cvtColor(np.asarray(ref)[..., :3], cv2.COLOR_RGB2YCrCb).astype(np.float32)[..., 0]
    y = Y[..., 0]
    hp = y - cv2.GaussianBlur(y, (0, 0), sigma)
    k = (int(tile), int(tile))
    num = cv2.boxFilter((R - y) * hp, -1, k)
    den = cv2.boxFilter(hp * hp, -1, k)
    amt = np.clip(num / np.maximum(den, 1e-3), 0.0, amax)
    Y[..., 0] = y + amt * hp
    return cv2.cvtColor(np.clip(Y + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def exact_enlarge(ref, s, iters=4):
    """``ref`` enlarged ``s`` times so that the area filter reduces it back to ``ref``: a cubic enlargement, then the
    reduction's error spread back over each block (iterative back-projection), uint8."""
    H, W = ref.shape[:2]
    r = ref.astype(np.float32)
    u = cv2.resize(r, (W * s, H * s), interpolation=cv2.INTER_CUBIC)
    for _ in range(iters):
        d = r - cv2.resize(u, (W, H), interpolation=cv2.INTER_AREA)
        u = np.clip(u + cv2.resize(d, (W * s, H * s), interpolation=cv2.INTER_CUBIC), 0, 255)
    return (u + 0.5).astype(np.uint8)


def run_reference_pipeline(ref_path, out_dir, style=None, *, log=print, check_rules=True, regions=None,
                           max_level=DEFAULT_LEVEL, work_scale=1, fine="plan", retouch_share=0.0,
                           fast_coarse=None, work_ref="cubic", deliver_filter="area", keep_work=False,
                           stipple=0, stipple_edges=True, texture_pass=None,
                           contours=False, smooth_tol=None):
    """Run the style's stages over the whole picture; every stage's ops must pass ``ops.check_mark_rules``.

    ``regions``: a list of bool masks; each stage then runs once per region clipped to it, the way the shell's
    region commands do (over a partition this matches the whole-picture run).
    ``max_level``: the finest detail level painted (DESIGN.md 10.1: 5 block-in, 4 forms, 3 edges, 2 fine,
    1 hairline, 0 retouch); stages and passes coarser than or at that level run, finer ones are left out.  Default
    2 = the stage plan as the shell runs it.  At 1 the hairline pass and at 0 also the retouching pass follow the
    plan (``stage_levels.py``); the picture at each level from 2 down is saved as ``L2.png``, ``L1.png`` (the final is
    the finest), and ``report["levels"]`` holds marks, canvas share touched and fidelity per level.

    ``work_scale`` (1, 2, 3): draw on a canvas that many times the reference (the reference upscaled, the regions
    too) and deliver every saved picture reduced with an area filter; levels are defined on the working canvas,
    fidelity is measured on the delivered picture.  ``fine``: "plan" (the stage plan's liner passes, hairline runs)
    or "streaks" (levels 2 and 1 as ``ops.Streak`` strokes whose colour and width change along the path,
    ``stage_streaks.streak_ops``).  ``retouch_share``: after the finest stroke level, level-0 retouching (one dab
    per delivered pixel) on that share of the delivered picture's 32 px tiles with the largest residual.
    ``fast_coarse`` (default: on when ``work_scale`` > 1): on a working canvas, the paint stages' forms and block-in
    passes (level 4 and 5) find their marks on the delivered picture and lay them scaled up (``ops.scale_ops``): the
    same marks, a quarter of the work.
    ``work_scale="auto"``: ``auto_work_scale`` (4 up to a 400 px long side, 3 up to 700, else 2).
    ``work_ref`` (on a working canvas): "cubic" (the reference enlarged with a cubic filter) or "exact" (the cubic
    enlargement corrected by back-projection until it reduces with the area filter to the reference itself, so a
    canvas that matches it delivers the reference's own finest detail).  ``deliver_filter``: "area", "lanczos" or
    "sharpen" (area, then a finishing sharpen of the luminance whose amount is fitted to the reference per 32 px
    tile, ``finish_sharpen``).  ``stipple`` (rounds, at ``max_level`` <= 1 on a working canvas): after level 1, soft
    airbrush dots of 1.5 delivered px on the residual's isolated points (``stage_streaks.stipple_ops``);
    ``stipple_edges=False`` leaves out the points on lines and edges.  ``texture_pass``: the paint / photo detail
    stage's texture pen pass; None = on, except with streaks to level 1 on a working canvas and stippling.
    ``contours`` (with ``fine="streaks"``): before the detail stage's first streak pass, every thin line of the
    reference still off is drawn as one continuous stroke (``stage_streaks.contour_ops``); ``contours="level1"``:
    the same just before the level-1 streaks instead (the contours then lie over the detail stage's streaks).
    ``smooth_tol`` (Delta E, with ``fine="streaks"``): the detail stage's first streak pass's tolerance where the reference is
    smooth (skies, gradients), so no flat patch of a coarser mark is left there; None = the same as elsewhere.
    ``keep_work``: also save the working-size final as ``work.png``."""
    ref = load_ref(ref_path) if isinstance(ref_path, (str, os.PathLike)) else np.asarray(ref_path)
    ref_d = ref
    if work_scale == "auto":
        work_scale = auto_work_scale(*np.asarray(ref_d).shape[1::-1])
    s_w = max(1, int(work_scale))
    fast_coarse = s_w > 1 if fast_coarse is None else bool(fast_coarse)
    Hd, Wd = ref_d.shape[:2]
    if s_w > 1:
        ref = cv2.resize(ref_d, (Wd * s_w, Hd * s_w), interpolation=cv2.INTER_CUBIC)
        if work_ref == "exact":
            ref = exact_enlarge(ref_d, s_w)
        if regions is not None:
            regions = [cv2.resize(np.asarray(r, np.uint8), (Wd * s_w, Hd * s_w), interpolation=cv2.INTER_NEAREST) > 0
                       for r in regions]

    def deliver(img, finish=False):
        if s_w == 1:
            out = img
        else:
            out = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_LANCZOS4 if deliver_filter == "lanczos"
                             else cv2.INTER_AREA)
        if finish and deliver_filter == "sharpen":
            out = finish_sharpen(out, ref_d)
        return out

    H, W = ref.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    style = style or guess_style(ref_d)
    plan = STAGES[style]
    layers = {name: new_layer(H, W) for name in LAYERS}
    pc = paper_color(ref)
    layers["paper"][..., :3] = np.asarray(pc, np.float32) / 255.0
    layers["paper"][..., 3] = 1.0
    mask = np.ones((H, W), bool)
    report = {"ref": str(ref_path), "style": style, "size": [W, H], "paper": pc, "stages": [],
              "max_level": int(max_level), "work_scale": s_w, "deliver": [Wd, Hd], "fine": fine,
              "fast_coarse": bool(fast_coarse), "work_ref": work_ref, "deliver_filter": deliver_filter,
              "stipple": int(stipple), "stipple_edges": bool(stipple_edges), "texture_pass": texture_pass,
              "contours": contours, "smooth_tol": smooth_tol}
    max_level = int(max_level)
    lvtouch = {k: np.zeros((H, W), bool) for k in range(6)}
    lvmarks = {}
    contour_done = {}
    smooth_done = {}
    lines = None
    t_all = time.perf_counter()

    # which pixels of each layer were last painted by a small mark (< SMALL_MARK px^2: fine strokes, small dabs)
    small_owner = {name: np.zeros((H, W), bool) for name in LAYERS}

    cur_lv = [None]       # the level of the stage running (marks of a stage count at its level)

    def run(layer, ops, clip=None, lv=None):
        if check_rules:
            bad = check_mark_rules(ops)
            if bad:
                raise AssertionError(f"{len(bad)} ops break the mark rules, e.g. " + "; ".join(bad[:3]))
        if not ops:
            return
        if layer != "sketch":
            for k, v in level_counts(ops).items():
                lvmarks[k] = lvmarks.get(k, 0) + v
            prev_l = layers[layer].copy()
            _run(layer, ops, clip)
            ch = np.any(np.abs(layers[layer] - prev_l) > 0.5 / 255, -1)
            if lv is None:
                lv = cur_lv[0] if cur_lv[0] is not None else min(mark_level(o) for o in ops)
            lvtouch[int(lv)] |= ch
            return
        _run(layer, ops, clip)

    def _run(layer, ops, clip=None):
        lay = layers[layer]
        from .ops import Spray
        if layer != "sketch" and len(ops) > 50 and not any(isinstance(o, Spray) for o in ops):
            # many marks: each one's own box is compared (a chunk's box would span the picture)
            for o in ops:
                b = op_bbox(o, lay.shape)
                if b is None:
                    apply_ops(lay, [o], clip=clip)
                    continue
                x0, y0, x1, y1 = b
                prev = lay[y0:y1, x0:x1].copy()
                apply_ops(lay, [o], clip=clip)
                ch = np.abs(lay[y0:y1, x0:x1] - prev).max(-1) > 0.5 / 255
                small_owner[layer][y0:y1, x0:x1][ch] = mark_area(o) < SMALL_MARK
            return
        for small, chunk in _size_chunks(ops):
            if layer == "sketch" or not chunk:
                apply_ops(lay, chunk, clip=clip)
                continue
            boxes = [b for b in (op_bbox(o, lay.shape) for o in chunk) if b is not None]
            if not boxes:
                apply_ops(lay, chunk, clip=clip)
                continue
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            prev = lay[y0:y1, x0:x1].copy()
            apply_ops(lay, chunk, clip=clip)
            ch = np.abs(lay[y0:y1, x0:x1] - prev).max(-1) > 0.5 / 255
            small_owner[layer][y0:y1, x0:x1][ch] = small

    # region calls (as the shell makes them): every stage runs once per region, clipped to it
    masks = [np.asarray(r, bool) for r in regions] if regions is not None else [mask]
    grain = {"amount": None, "size": None, "map": None}
    ramp = {"share": None}
    whole = regions is None

    for idx, (stage, layer, kind, params) in enumerate(plan, 1):
        slev = STAGE_LEVEL.get(stage, 2)
        if kind not in ("paint", "detail") and slev < max_level:
            continue
        cur_lv[0] = slev if kind not in ("paint", "detail") else None
        t0 = time.perf_counter()
        before = layers[layer].copy()
        all_ops = []
        sub = []
        for rm in masks:
            clip = None if whole else rm
            if kind == "sketch":
                ops = sketch_ops(ref, rm, looseness=params.get("looseness", 0.5), seed=idx)
                run(layer, ops, clip)
                all_ops += ops
            elif kind == "ink":
                if lines is None:
                    lines = line_map(ref, mask=mask)
                ops = ink_ops(ref, rm, detail=params.get("detail", 0.5), seed=idx, lines=lines,
                              method=params.get("method", "strokes"), true_thin=params.get("true_thin", True))
                run(layer, ops, clip)
                all_ops += ops
            elif kind == "flat":
                if params.get("line_mask") != "none":
                    if lines is None:
                        lines = line_map(ref, mask=mask)
                    lm = lines.mask
                else:
                    lm = np.zeros((H, W), bool)
                if params.get("ink_mask"):
                    # the ink already drawn (solid masses included) is line art for the flats
                    lm = lm | (layers["line"][..., 3] > 0.5)
                ops = flat_ops(ref, rm, line_mask=lm, seed=idx, skip_dark=params.get("skip_dark"))
                run(layer, ops, clip)
                all_ops += ops
            elif kind == "shade":
                base = composite(layers, order=["paper", "flat"])
                if fast_coarse and s_w > 1:
                    # the shade shapes (level 4) found on the delivered picture, laid at the working size
                    from .ops import scale_ops
                    rm_d = None if whole else cv2.resize(rm.astype(np.uint8), (Wd, Hd),
                                                         interpolation=cv2.INTER_NEAREST) > 0
                    ops = shade_ops(ref_d, rm_d, deliver(base), levels=params.get("levels", 3),
                                    soft=params.get("soft", 0.3), seed=idx, air=params.get("air", "auto"))
                    ops = scale_ops(ops, s_w, (H, W))
                else:
                    ops = shade_ops(ref, rm, base, levels=params.get("levels", 3), soft=params.get("soft", 0.3),
                                    seed=idx, air=params.get("air", "auto"), min_area=12 * s_w * s_w,
                                    air_sizes=tuple(v * s_w for v in (64, 32, 16, 8, 4)))
                fa = layers["flat"][..., 3]
                run(layer, ops, clip=fa if whole else np.minimum(fa, rm.astype(np.float32)))
                all_ops += ops
            elif kind == "air":
                # airbrush over the whole region (one frisket), the dabs clear of the reference's hard steps
                from .airbrush import airbrush_ops, step_edges
                canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                st_air = {}
                ops = airbrush_ops(ref, canvas, rm.astype(np.int32),
                                   sizes=tuple(v * s_w for v in params.get("sizes", (64, 32, 16, 8))),
                                   seed=idx, edges=step_edges(ref, sigma=2.0, thr=params.get("edge", 4.0)), stats=st_air)
                run(layer, ops, clip)
                all_ops += ops
                sub.append(st_air)
            elif kind == "grain":
                # photo grain as a finishing effect, as strong as the reference's own fine noise requires; with
                # "local" only where the reference has grain, at each area's own amount
                if grain["amount"] is None:
                    canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                    if params.get("local", True):
                        grain["map"], grain["size"] = grain_local(ref, canvas)
                        grain["amount"] = 0.0 if grain["map"] is None else float(grain["map"].max())
                    else:
                        grain["amount"], grain["size"] = grain_params(ref, canvas)
            elif kind == "blend":
                # a blending pass on each named layer (it reads only that layer's paint)
                if ramp["share"] is None:
                    ramp["share"] = soft_ramp_share(ref)
                if ramp["share"] >= params.get("min_ramp", 0.0):
                    ops = blend_ops(ref, rm, radius=params.get("radius", 3.0), tol=params.get("tol", 10.0),
                                    strength=params.get("strength", 1.0))
                    for ln in params.get("layers", (layer,)):
                        run(ln, ops, clip, lv=slev)
                    all_ops += ops
                if params.get("grain") and rm is masks[-1] and grain["amount"] is None:
                    # the finishing grain after the last region's blend (measured on the finished canvas)
                    canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                    grain["map"], grain["size"] = grain_local(ref, canvas)
                    grain["amount"] = 0.0 if grain["map"] is None else float(grain["map"].max())
            elif kind in ("paint", "detail"):
                if stage == "block-in" and max_level <= 5:
                    g = ground_ops(ref, rm)
                    run(layer, g, clip, lv=5)
                    all_ops += g
                if params.get("highlights") and max_level <= 2:
                    # thin light marks first (edge lights, glints) as pen strokes in their own colour
                    from .stage_paint import highlight_ops
                    ts = time.perf_counter()
                    canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                    hops = highlight_ops(ref, rm, canvas, seed=idx * 100 + 98)
                    pb = layers[layer].copy()
                    run(layer, hops, clip, lv=2)
                    all_ops += hops
                    sub.append({"highlights": True, "ops": len(hops), "touched": round(_touched(pb, layers[layer]), 3),
                                "seconds": round(time.perf_counter() - ts, 2)})
                for i, (size, br) in enumerate(stage_passes(params, H, W, s_w)):
                    plev = pass_level(size, br)
                    if plev < max_level:
                        continue
                    small_detail = stage == "detail" and br not in ("shapes", "soft") and size <= STREAK_MAX * s_w
                    if fine == "streaks" and (br == "liner" or small_detail):
                        # the fine passes as streaks: thin strokes along the structure, colour changing along them
                        # (the detail stage's small brush pass too: a streak up to STREAK_MAX delivered px wide)
                        if br != "liner":
                            size = float(STREAK_MAX)
                            plev = pass_level(size, "liner")
                        from .stage_streaks import streak_ops
                        if contours and contours != "level1" and stage == "detail" and not contour_done.get((idx, id(rm))):
                            from .stage_streaks import contour_ops
                            contour_done[(idx, id(rm))] = True
                            ts = time.perf_counter()
                            canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                            cst = {}
                            cops = contour_ops(ref, rm, canvas, max_width=3.0 * s_w, min_len=3.0 * s_w, stats=cst)
                            pb = layers[layer].copy()
                            run(layer, cops, clip, lv=2)
                            all_ops += cops
                            sub.append({"contours": True, "lines": cst.get("lines", 0),
                                        "stations": cst.get("stations", 0),
                                        "touched": round(_touched(pb, layers[layer]), 3),
                                        "seconds": round(time.perf_counter() - ts, 2)})
                        ts = time.perf_counter()
                        canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                        sst = {}
                        # the smooth areas' finer tolerance on the stage's first (widest) streak pass only
                        st_tol = smooth_tol if not smooth_done.get((idx, id(rm))) else None
                        smooth_done[(idx, id(rm))] = True
                        ops = streak_ops(ref, rm, canvas, width=size, smooth_tol=st_tol, stats=sst)
                        pb = layers[layer].copy()
                        run(layer, ops, clip, lv=plev)
                        all_ops += ops
                        sub.append({"size": round(size, 2), "brush": "streak", "level": plev,
                                    "ops": sst.get("marks", 0), "stations": sst.get("stations", 0),
                                    "touched": round(_touched(pb, layers[layer]), 3),
                                    "seconds": round(time.perf_counter() - ts, 2)})
                        continue
                    # one call has a time budget (paint_ops PAINT_BUDGET); a pass that runs out of time is called
                    # again on the updated canvas, as the shell's next `detail` call would (MAX_CALLS at most)
                    for call in range(MAX_CALLS):
                        ts = time.perf_counter()
                        canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                        pst = {}
                        if fast_coarse and s_w > 1 and plev >= 4:
                            # a forms / block-in pass on a working canvas: its marks are found on the delivered
                            # picture (the same marks, a quarter of the pixels) and laid at the working size
                            from .ops import scale_ops
                            rm_d = rm if whole else cv2.resize(rm.astype(np.uint8), (Wd, Hd),
                                                               interpolation=cv2.INTER_NEAREST) > 0
                            ops = paint_ops(ref_d, np.ones((Hd, Wd), bool) if whole else rm_d, deliver(canvas),
                                            size=size / s_w, brush=br, seed=idx * 100 + i + 31 * call,
                                            edge_aware=params.get("edge_aware"), threshold=params.get("threshold"),
                                            structure=params.get("structure", False), tips=params.get("tips"),
                                            stats=pst)
                            ops = scale_ops(ops, s_w, (H, W))
                        else:
                            ops = paint_ops(ref, rm, canvas, size=size, brush=br, seed=idx * 100 + i + 31 * call,
                                            edge_aware=params.get("edge_aware"), threshold=params.get("threshold"),
                                            structure=params.get("structure", False), tips=params.get("tips"),
                                            stats=pst)
                        pb = layers[layer].copy()
                        run(layer, ops, clip, lv=plev)
                        all_ops += ops
                        rec_p = {"size": round(size, 2), "brush": br, "level": plev, "ops": len(ops),
                                 "touched": round(_touched(pb, layers[layer]), 3),
                                 "seconds": round(time.perf_counter() - ts, 2)}
                        if call:
                            rec_p["call"] = call + 1
                        if br == "shapes":
                            rec_p["shapes"] = {k: pst.get(k) for k in ("pieces", "tried", "fills", "small")}
                        elif pst.get("shapes"):
                            rec_p["shapes"] = pst["shapes"]
                        if pst.get("air"):
                            rec_p["air"] = {k: v for k, v in pst["air"].items() if isinstance(v, (int, float))}
                        sub.append(rec_p)
                        if call == 0:
                            first_n = len(ops)
                        if not pst.get("out_of_time") or len(ops) < 0.08 * max(first_n, 1):
                            break
                tex_on = (not (stipple and fine == "streaks" and s_w > 1 and max_level <= 1)
                          if texture_pass is None else bool(texture_pass))
                if params.get("texture") and max_level <= 2 and tex_on:
                    # (with streaks to level 1 and stippling, the texture's pen marks are left to those)
                    # the fine streaks of the texture as directed pen marks
                    from .stage_paint import texture_ops
                    ts = time.perf_counter()
                    canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                    ops = texture_ops(ref, rm, canvas, seed=idx * 100 + 99)
                    pb = layers[layer].copy()
                    run(layer, ops, clip, lv=2)
                    all_ops += ops
                    sub.append({"texture": True, "ops": len(ops), "touched": round(_touched(pb, layers[layer]), 3),
                                "seconds": round(time.perf_counter() - ts, 2)})
        dt = time.perf_counter() - t0
        show = [n for n in LAYERS if n != "sketch"]
        if stage in ("sketch", "line"):
            show = show + ["sketch"]
        img = composite(layers, order=show, opacity={"sketch": 0.6})
        img = _with_grain(img, grain)
        fn = f"{idx:02d}-{stage}.png"
        Image.fromarray(deliver(img)).save(os.path.join(out_dir, fn))
        sc = quick_scores(ref, img)
        areas = [mark_area(o) for o in all_ops]
        rec = {"stage": stage, "layer": layer, "ops": len(all_ops), "seconds": round(dt, 2), "file": fn,
               "touched": round(_touched(before, layers[layer]), 3),
               "mean_mark_area": round(float(np.mean(areas)), 1) if areas else 0.0,
               "strokes": sum(isinstance(o, Stroke) for o in all_ops),
               "fills": sum(isinstance(o, Fill) for o in all_ops), **sc}
        if sub:
            rec["passes"] = sub
        report["stages"].append(rec)
        log(f"  {fn:16s} ops {len(all_ops):6d}  {dt:6.1f}s  touched {rec['touched']:.2f}  "
            f"area {rec['mean_mark_area']:8.1f}  dE {sc['mean_dE']:.2f}  edgeF1 {sc['edge_f1']:.3f}  "
            f"hf {sc['hf_ratio']:.2f}")
    final = composite(layers, order=[n for n in LAYERS if n != "sketch"])
    levels = []
    if max_level <= 1:
        # the picture at level 2 (as the plan leaves it, with its grain), then the hairline and retouching passes on
        # the residual, on top of everything (the detail layer), each level saved
        from .stage_levels import retouch_ops
        img2 = _with_grain(final, grain)
        Image.fromarray(deliver(img2)).save(os.path.join(out_dir, "L2.png"))
        levels.append({"level": 2, **fidelity_metrics(ref_d, deliver(img2))})
        for lv in (1, 0):
            if lv < max_level:
                break
            t0 = time.perf_counter()
            canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
            ops = []
            for rm in masks:
                if lv == 1 and fine == "streaks":
                    from .stage_streaks import streak_ops
                    if contours == "level1":
                        from .stage_streaks import contour_ops
                        cops = contour_ops(ref, None if whole else rm, canvas, max_width=3.0 * s_w,
                                           min_len=3.0 * s_w)
                        run("detail", cops, None if whole else rm, lv=2)
                        ops += cops
                        canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                    ops_r = streak_ops(ref, None if whole else rm, canvas, width=1.0)
                else:
                    ops_r = retouch_ops(ref, None if whole else rm, canvas, level=lv)
                run("detail", ops_r, None if whole else rm, lv=lv)
                ops += ops_r
            final = composite(layers, order=[n for n in LAYERS if n != "sketch"])
            Image.fromarray(deliver(final)).save(os.path.join(out_dir, f"L{lv}.png"))
            levels.append({"level": lv, "seconds": round(time.perf_counter() - t0, 2),
                           "marks": int(sum(len(o.n) if type(o).__name__ == "Streak" else len(o.x) for o in ops)),
                           **fidelity_metrics(ref_d, deliver(final))})
            log(f"  L{lv}: {levels[-1]['marks']} marks  {levels[-1]['seconds']}s  dE {levels[-1]['mean_dE']:.3f}  "
                f"PSNR {levels[-1]['psnr']:.1f}")
        if stipple and s_w > 1:
            from .stage_streaks import stipple_ops
            t0 = time.perf_counter()
            nd = 0
            for rnd in range(int(stipple)):
                canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
                cd = deliver(canvas)
                ops = []
                for rm in masks:
                    rm_d = None if whole else cv2.resize(rm.astype(np.uint8), (Wd, Hd),
                                                         interpolation=cv2.INTER_NEAREST) > 0
                    ops_r = stipple_ops(ref_d, cd, s_w, rm_d, edges=stipple_edges)
                    run("detail", ops_r, None if whole else rm, lv=2)
                    ops += ops_r
                nd += int(sum(len(o.pts) for o in ops))
            final = composite(layers, order=[n for n in LAYERS if n != "sketch"])
            levels.append({"level": "stipple", "seconds": round(time.perf_counter() - t0, 2), "marks": nd,
                           "touched": round(nd / float(Hd * Wd), 4), **fidelity_metrics(ref_d, deliver(final))})
            log(f"  stipple: {nd} dots  {levels[-1]['seconds']}s  PSNR {levels[-1]['psnr']:.1f}")
        grain = {"amount": None, "size": None, "map": None}      # the finest levels carry the reference's own grain
    if retouch_share > 0 and max_level >= 1:
        # level-0 retouching on the worst tiles only: one dab per delivered pixel still off, in its colour
        t0 = time.perf_counter()
        canvas = composite(layers, order=[n for n in LAYERS if n != "sketch"])
        rops, tiles = tile_retouch_ops(ref_d, deliver(_with_grain(canvas, grain)), s_w, retouch_share)
        run("detail", rops, None, lv=0)
        final = composite(layers, order=[n for n in LAYERS if n != "sketch"])
        grain = {"amount": None, "size": None, "map": None}
        nm = int(sum(len(o.x) for o in rops))
        levels.append({"level": "retouch", "share": retouch_share, "tiles": tiles, "marks": nm,
                       "touched": round(nm / float(Hd * Wd), 4),
                       "seconds": round(time.perf_counter() - t0, 2), **fidelity_metrics(ref_d, deliver(final))})
        log(f"  retouch {retouch_share:.0%} of tiles: {nm} dabs  PSNR {levels[-1]['psnr']:.1f}")
    report["levels"] = levels
    report["level_marks"] = {str(k): v for k, v in sorted(lvmarks.items(), reverse=True)}
    report["level_touched"] = {str(k): round(float(v.mean()), 4) for k, v in sorted(lvtouch.items(), reverse=True)
                               if v.any()}
    # share of the canvas whose final colour comes from small marks: the topmost layer that covers a pixel decides
    top_small = np.zeros((H, W), bool)
    for name in [n for n in LAYERS if n != "sketch"]:
        cov_ = layers[name][..., 3] > 0.5
        top_small = np.where(cov_, small_owner[name], top_small)
    report["small_mark_share"] = round(float(top_small.mean()), 4)
    if grain["amount"]:
        final = _with_grain(final, grain)
        report["grain"] = round(grain["amount"], 3)
        report["grain_size"] = grain["size"]
        if grain.get("map") is not None:
            report["grain_local"] = {"mean": round(float(grain["map"].mean()), 4),
                                     "share": round(float((grain["map"] > 0.05).mean()), 4)}
    Image.fromarray(deliver(final, True)).save(os.path.join(out_dir, "final.png"))
    if keep_work and s_w > 1:
        Image.fromarray(final).save(os.path.join(out_dir, "work.png"))
    report["final"] = quick_scores(ref_d, deliver(final, True))
    report["final_fidelity"] = fidelity_metrics(ref_d, deliver(final, True))
    report["seconds_total"] = round(time.perf_counter() - t_all, 2)
    with open(os.path.join(out_dir, "pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pixelcanvas.pipeline", description=__doc__.split("\n")[0])
    ap.add_argument("ref")
    ap.add_argument("out_dir")
    ap.add_argument("--style", choices=sorted(STAGES), default=None)
    a = ap.parse_args(argv)
    r = run_reference_pipeline(a.ref, a.out_dir, a.style)
    print(f"style {r['style']}  total {r['seconds_total']}s  final {r['final']}")


if __name__ == "__main__":
    main()
