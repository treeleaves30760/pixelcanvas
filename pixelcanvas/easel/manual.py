"""Hand tools (DESIGN.md section 6): stroke, fill, shape, gradient, erase, smooth, adjust, pick, palette.

They build ``Stroke`` / ``Fill`` ops for the brush engine (``ops.py`` / ``brush.py``, owned by HAND) or, for the two
pixel filters ``smooth`` and ``adjust``, change layer pixels directly. Coordinates arrive here already in pixels.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import coords, regions


class EngineMissing(Exception):
    """The brush engine or a stage module is not installed (or failed to import)."""


# ----------------------------------------------------------------------------------------------- engine access

def _try_import(name: str):
    try:
        return importlib.import_module(f"pixelcanvas.easel.{name}"), None
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith(f"easel.{name}"):
            return None, None
        return None, e
    except Exception as e:  # noqa: BLE001 - a half-written module must not crash the CLI
        return None, e


def engine() -> SimpleNamespace:
    """Stroke, Fill, apply_ops, ops_to_jsonable, PRESETS from ops.py / brush.py."""
    ops_mod, e1 = _try_import("ops")
    brush_mod, e2 = _try_import("brush")
    mods = [m for m in (ops_mod, brush_mod) if m is not None]
    found = {}
    for attr in ("Stroke", "Fill", "apply_ops", "ops_to_jsonable", "ops_from_jsonable", "PRESETS"):
        for m in mods:
            if hasattr(m, attr):
                found[attr] = getattr(m, attr)
                break
    need = ("Stroke", "Fill", "apply_ops")
    if not all(k in found for k in need):
        err = e1 or e2
        if err is not None:
            raise EngineMissing(f"the brush engine failed to load ({type(err).__name__}: {err})")
        raise EngineMissing("the brush engine (ops.py / brush.py) is not installed yet")
    found.setdefault("ops_to_jsonable", None)
    found.setdefault("PRESETS", {})
    found["check"] = next((getattr(m, "check_mark_rules") for m in mods if hasattr(m, "check_mark_rules")), None)
    for m in mods:
        for attr in ("mask_to_rle", "rle_encode", "encode_rle", "rle_from_mask"):
            if hasattr(m, attr):
                found["rle"] = getattr(m, attr)
                break
    return SimpleNamespace(**found)


STAGE_FUNCS = {
    "sketch": ("stage_sketch", "sketch_ops"),
    "ink": ("stage_ink", "ink_ops"),
    "flat": ("stage_flat", "flat_ops"),
    "shade": ("stage_shade", "shade_ops"),
    "paint": ("stage_paint", "paint_ops"),
    "detail": ("stage_paint", "paint_ops"),
}


def stage_function(cmd: str):
    mod_name, fn_name = STAGE_FUNCS[cmd]
    mod, err = _try_import(mod_name)
    if mod is None:
        if err is not None:
            raise EngineMissing(f"{mod_name}.py failed to load ({type(err).__name__}: {err})")
        raise EngineMissing(f"the {cmd} engine ({mod_name}.py) is not installed yet")
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise EngineMissing(f"{mod_name}.py has no {fn_name}() yet")
    return fn


def ops_jsonable(eng, ops) -> list:
    if eng.ops_to_jsonable is not None:
        try:
            return eng.ops_to_jsonable(ops)
        except Exception:  # noqa: BLE001
            pass
    out = []
    for o in ops:
        d = {"type": type(o).__name__}
        for k, v in vars(o).items():
            if isinstance(v, np.ndarray):
                v = np.round(v.astype(np.float64), 2).tolist()
            elif isinstance(v, list):
                v = [np.round(np.asarray(x, np.float64), 2).tolist() if isinstance(x, np.ndarray) else x for x in v]
            d[k] = v
        out.append(d)
    return out


def brush_names(eng) -> List[str]:
    """Brushes the model may use; the 1 px ``pixel`` brush is a reference copy (DESIGN.md 10.1) and not offered."""
    names = sorted(eng.PRESETS.keys()) if eng.PRESETS else ["pencil", "pen", "marker", "round", "flat", "dry",
                                                           "soft", "wash"]
    return [n for n in names if n != "pixel"]


class MarkRuleError(Exception):
    """Ops that break the mark rules of DESIGN.md section 10.1."""


def check_marks(eng, ops, stage_tool: bool) -> None:
    fn = getattr(eng, "check", None)
    if fn is None or not ops:
        return
    try:
        # a hand tool lays marks down to level 1 (hairlines); level 0 is the retouching brush (easel retouch)
        bad = fn(ops, stage_tool=stage_tool, **({} if stage_tool else {"min_level": 1}))
    except TypeError:
        try:
            bad = fn(ops, stage_tool=stage_tool)
        except TypeError:
            bad = fn(ops)
    if bad:
        bad = [bad] if isinstance(bad, str) else list(bad)
        more = f" (+{len(bad) - 2} more)" if len(bad) > 2 else ""
        raise MarkRuleError("these marks break the mark rules (DESIGN.md 10.1): " + "; ".join(map(str, bad[:2]))
                            + more)


# ----------------------------------------------------------------------------------------------- mask -> Fill

def mask_polygons(mask: np.ndarray) -> List[np.ndarray]:
    """Rings (even-odd) covering every pixel of the mask; pair with clip=mask for exact edges."""
    m = cv2.dilate(np.asarray(mask, np.uint8), np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for c in cs:
        if len(c) >= 3:
            rings.append(c.reshape(-1, 2).astype(np.float32) + 0.5)
        elif len(c) > 0:
            p = c.reshape(-1, 2).astype(np.float32) + 0.5
            x0, y0 = p.min(0) - 1
            x1, y1 = p.max(0) + 1
            rings.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))
    return rings


def fill_for_mask(eng, mask: np.ndarray, color, opacity: float = 1.0, erase: bool = False,
                  color2=None, p0=None, p1=None, feather: float = 0.0):
    rle = None
    if getattr(eng, "rle", None) is not None:
        try:
            rle = eng.rle(np.asarray(mask, bool))
        except Exception:  # noqa: BLE001
            rle = None
    if rle is not None:
        feather = max(float(feather), 1.0)     # a run-length area is a smooth mark only with a soft edge (10.1)
    kw = dict(color=tuple(int(c) for c in color), opacity=float(opacity), feather=float(feather), erase=erase)
    if color2 is not None:
        kw.update(color2=tuple(int(c) for c in color2), p0=tuple(map(float, p0)), p1=tuple(map(float, p1)))
    if rle is not None:
        return eng.Fill(polygons=None, mask_rle=rle, **kw)
    return eng.Fill(polygons=mask_polygons(mask), mask_rle=None, **kw)


# ----------------------------------------------------------------------------------------------- paths

def chaikin(pts: np.ndarray, closed: bool, iters: int = 2) -> np.ndarray:
    p = np.asarray(pts, np.float32)
    for _ in range(iters):
        if len(p) < 3:
            break
        q = np.roll(p, -1, axis=0) if closed else p[1:]
        a = p if closed else p[:-1]
        r1 = 0.75 * a + 0.25 * q
        r2 = 0.25 * a + 0.75 * q
        mid = np.empty((len(r1) * 2, 2), np.float32)
        mid[0::2], mid[1::2] = r1, r2
        p = mid if closed else np.vstack([p[:1], mid, p[-1:]])
    return p


def resample(pts: np.ndarray, spacing: float) -> np.ndarray:
    p = np.asarray(pts, np.float64)
    if len(p) < 2:
        return p.astype(np.float32)
    seg = np.hypot(*np.diff(p, axis=0).T)
    t = np.concatenate([[0], np.cumsum(seg)])
    if t[-1] < 1e-6:
        return p[:1].astype(np.float32)
    n = max(2, int(np.ceil(t[-1] / spacing)) + 1)
    tt = np.linspace(0, t[-1], n)
    return np.stack([np.interp(tt, t, p[:, 0]), np.interp(tt, t, p[:, 1])], 1).astype(np.float32)


def smooth_path(pts_px: np.ndarray, closed: bool = False, snap_ref: Optional[np.ndarray] = None,
                smoothing: float = 0.5) -> np.ndarray:
    p = np.asarray(pts_px, np.float32)
    if closed and len(p) >= 3:
        p = np.vstack([p, p[:1]])
    if snap_ref is not None and len(p) >= 2:
        p = regions.livewire(regions.edge_cost(snap_ref), p)
        p = resample(p, 2.0)
        k = 5
        if len(p) > k:
            pad = np.vstack([np.repeat(p[:1], k // 2, 0), p, np.repeat(p[-1:], k // 2, 0)])
            ker = np.ones(k) / k
            p = np.stack([np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")], 1)
            p = p.astype(np.float32)
    elif len(p) >= 3 and smoothing > 0:
        p = chaikin(p, closed=False, iters=1 + int(round(2 * smoothing)))
    return resample(p, 2.0)


def footprint_colour(img: np.ndarray, pts_px: np.ndarray, width: float) -> Tuple[int, int, int]:
    """Eyedropper for a stroke: one colour, the median of the reference under the stroke's footprint (10.1)."""
    H, W = img.shape[:2]
    m = np.zeros((H, W), np.uint8)
    p = np.round(np.asarray(pts_px, np.float32)).astype(np.int32).reshape(-1, 1, 2)
    t = max(1, int(round(width)))
    if len(p) == 1:
        cv2.circle(m, tuple(int(v) for v in p[0, 0]), max(1, t // 2), 1, -1)
    else:
        cv2.polylines(m, [p], False, 1, t)
    return pick_colour(img, mask=m > 0)


def sample_colours(img: np.ndarray, pts_px: np.ndarray, radius: float) -> np.ndarray:
    H, W = img.shape[:2]
    blur = cv2.GaussianBlur(img, (0, 0), max(0.8, radius / 2))
    xs = np.clip(pts_px[:, 0].astype(int), 0, W - 1)
    ys = np.clip(pts_px[:, 1].astype(int), 0, H - 1)
    return blur[ys, xs].astype(np.float32)


def stroke_op(eng, pts_px, size, colour, opacity=1.0, brush="pen", seed=0, erase=False):
    pts = np.asarray(pts_px, np.float32).reshape(-1, 2)
    col = colour if isinstance(colour, np.ndarray) else tuple(int(c) for c in colour)
    return eng.Stroke(pts=pts, width=float(size), color=col, opacity=float(opacity), brush=brush, seed=int(seed),
                      erase=erase)


def ellipse_points(box_px, n: int = 96) -> np.ndarray:
    x0, y0, x1, y1 = box_px
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1).astype(np.float32)


def rect_points(box_px) -> np.ndarray:
    x0, y0, x1, y1 = box_px
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


# ----------------------------------------------------------------------------------------------- bucket fill

def bucket_mask(seed_px, bounds: str, line_alpha: Optional[np.ndarray], ref: Optional[np.ndarray],
                canvas: Optional[np.ndarray], tol: float = 12.0, gap: int = 3) -> np.ndarray:
    """Area to fill from a seed point. bounds: line (line art with gap closing) | ref | canvas."""
    if bounds == "line":
        barrier = line_alpha > 0.5
        H, W = barrier.shape
        k = 2 * gap + 1
        closed = cv2.dilate(barrier.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        free = ~closed
        x, y = int(np.clip(seed_px[0], 0, W - 1)), int(np.clip(seed_px[1], 0, H - 1))
        if not free[y, x]:
            d, idx = ndi.distance_transform_edt(~free, return_indices=True)
            y, x = int(idx[0][y, x]), int(idx[1][y, x])
        lab, _ = ndi.label(free)
        area = lab == lab[y, x]
        grown = cv2.dilate(area.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 2, k + 2))) > 0
        return grown & (area | closed)
    img = ref if bounds == "ref" else canvas
    return regions.wand(img, seed_px, tol=tol)


# ----------------------------------------------------------------------------------------------- pixel filters

def smooth_layer(layer: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Blur a layer inside a mask (premultiplied, so edges do not darken)."""
    sigma = 0.6 + 4.0 * float(amount)
    pm = layer.copy()
    pm[..., :3] *= pm[..., 3:4]
    b = cv2.GaussianBlur(pm, (0, 0), sigma)
    a = b[..., 3:4]
    out = np.zeros_like(layer)
    out[..., 3:4] = a
    out[..., :3] = np.where(a > 1e-4, b[..., :3] / np.maximum(a, 1e-4), 0)
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
    return (layer * (1 - m) + out * m).astype(np.float32)


def adjust_layer(layer: np.ndarray, mask: np.ndarray, brightness=0.0, contrast=0.0, saturation=0.0,
                 hue=0.0) -> np.ndarray:
    """brightness / contrast / saturation in -1..1, hue in degrees; only inside the mask."""
    rgb = np.clip(layer[..., :3], 0, 1)
    out = rgb + float(brightness) * 0.5
    out = (out - 0.5) * (1.0 + float(contrast)) + 0.5
    hsv = cv2.cvtColor(np.clip(out, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + float(hue)) % 360.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + float(saturation)), 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    res = layer.copy()
    m = mask[..., None].astype(np.float32)
    res[..., :3] = rgb * (1 - m) + out * m
    return res.astype(np.float32)


# ----------------------------------------------------------------------------------------------- pick / palette

def pick_colour(img: np.ndarray, pt_px=None, mask: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
    H, W = img.shape[:2]
    if mask is not None:
        px = img[mask]
        if len(px) > 200000:
            px = px[:: len(px) // 200000 + 1]
        return tuple(int(v) for v in np.median(px, axis=0))  # type: ignore
    x, y = int(np.clip(pt_px[0], 0, W - 1)), int(np.clip(pt_px[1], 0, H - 1))
    sub = img[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3)
    return tuple(int(v) for v in np.median(sub, axis=0))  # type: ignore


def palette(img: np.ndarray, mask: Optional[np.ndarray] = None, k: int = 8) -> List[Tuple[tuple, float]]:
    """k main colours (rgb, share) by k-means in Lab, largest share first. Deterministic."""
    px = img[mask] if mask is not None else img.reshape(-1, 3)
    if len(px) == 0:
        return []
    step = max(1, len(px) // 60000)
    px = px[::step]
    k = int(max(1, min(k, 16, len(px))))
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    out = []
    for i in range(k):
        sel = labels == i
        if not sel.any():
            continue
        col = np.median(px[sel], axis=0)
        out.append((tuple(int(c) for c in col), float(sel.mean())))
    out.sort(key=lambda t: -t[1])
    return out
