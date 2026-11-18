"""Detail and painterly brushwork without a reference. The model says where and which way; the tool lays the marks.

- ``hair_ops``: tapered strands that follow one or two flow curves through a hair region, a band of shine across
  them on the lit side.
- ``eye_ops``: iris (gradient), pupil, catch-lights and the shadow under the upper lid, inside an eye area.
- ``fold_ops``: a fold line with a soft shadow on the side away from the light.
- ``pattern_ops``: hatching, cross-hatching, screentone dots, stripes, checks, clipped by the caller.
- ``scatter_ops``: stamps (dot, leaf, petal, grass blade, star, dash, needle) along a curve or inside a region,
  with size / angle / colour jitter.
- ``brush_ops``: textured strokes laid along a direction field over a region, lighter on the lit side: forms.
- ``band_polygon``: the area above or below a curve (sky, mountains, land bands), with an optional rough edge.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .draw_core import catmull_rom, ellipse_pts, erode, light_vec, path_length, resample, taper_widths


def _rng(seed):
    return np.random.default_rng(int(seed) & 0x7FFFFFFF)


# ============================================================================================ flow field

def flow_field(curves_px: Sequence[np.ndarray], shape, box=None) -> Tuple[np.ndarray, np.ndarray]:
    """Unit direction field (dx, dy) from one or more flow curves: each pixel takes the tangent of the curves,
    weighted by inverse distance."""
    H, W = shape
    step = max(2, int(max(H, W) / 160))
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    acc_x = np.zeros(xs.shape, np.float64)
    acc_y = np.zeros(xs.shape, np.float64)
    wsum = np.zeros(xs.shape, np.float64)
    for c in curves_px:
        c = resample(c, max(2.0, step / 2))
        if len(c) < 2:
            continue
        t = np.gradient(c, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        d2 = (xs[..., None] - c[None, None, :, 0]) ** 2 + (ys[..., None] - c[None, None, :, 1]) ** 2
        j = np.argmin(d2, axis=-1)
        dmin = np.sqrt(np.take_along_axis(d2, j[..., None], -1)[..., 0])
        w = 1.0 / (dmin + 0.03 * max(H, W)) ** 2
        acc_x += w * t[j, 0]
        acc_y += w * t[j, 1]
        wsum += w
    fx = acc_x / np.maximum(wsum, 1e-12)
    fy = acc_y / np.maximum(wsum, 1e-12)
    n = np.sqrt(fx * fx + fy * fy) + 1e-9
    fx, fy = (fx / n).astype(np.float32), (fy / n).astype(np.float32)
    fx = cv2.resize(fx, (W, H), interpolation=cv2.INTER_LINEAR)
    fy = cv2.resize(fy, (W, H), interpolation=cv2.INTER_LINEAR)
    n = np.sqrt(fx * fx + fy * fy) + 1e-9
    return fx / n, fy / n


def angle_field(deg: float, shape) -> Tuple[np.ndarray, np.ndarray]:
    a = math.radians(deg)
    H, W = shape
    return np.full((H, W), math.cos(a), np.float32), np.full((H, W), -math.sin(a), np.float32)


def streamline(field, mask, start, step: float, max_len: float) -> np.ndarray:
    fx, fy = field
    H, W = mask.shape
    pts = [np.array(start, float)]
    for sign in (1, -1):
        p = np.array(start, float)
        prev = None
        L = 0.0
        seq = []
        while L < max_len / 2:
            xi, yi = int(p[0]), int(p[1])
            if not (0 <= xi < W and 0 <= yi < H) or not mask[yi, xi]:
                break
            v = np.array([fx[yi, xi], fy[yi, xi]]) * sign
            if prev is not None and v @ prev < 0:
                v = -v
            p = p + v * step
            prev = v
            L += step
            seq.append(p.copy())
        if sign == 1:
            fwd = seq
        else:
            bwd = seq
    return np.array(bwd[::-1] + pts + fwd)


# ============================================================================================ hair

def hair_ops(mask: np.ndarray, curves_px: List[np.ndarray], base_rgb, dark_rgb, light_rgb, light: np.ndarray,
             count: int = 60, width: float = 2.0, shine: bool = True, seed: int = 1,
             guides_px: Optional[List[np.ndarray]] = None, near: Optional[float] = None) -> list:
    """Strands along the flow curves inside ``mask``. ``guides_px`` (the centre lines of drawn locks) steer the
    direction near them as well; with ``near`` the strands start within that distance (px) of the flow curves, so
    one flow curve over the fringe puts strands on the fringe, not over the whole hair."""
    from .ops import Stroke
    m = np.asarray(mask, bool)
    if m.sum() < 50 or not curves_px:
        return []
    H, W = m.shape
    field = flow_field(list(curves_px) + list(guides_px or []), m.shape)
    start = erode(m, 1)
    if near:
        dmask = np.zeros(m.shape, np.uint8)
        for c in curves_px:
            cv2.polylines(dmask, [np.round(resample(c, 2.0)).astype(np.int32)], False, 1, 1)
        dist = cv2.distanceTransform((1 - dmask).astype(np.uint8), cv2.DIST_L2, 5)
        start = start & (dist <= near)
    ys, xs = np.nonzero(start)
    if len(xs) == 0:
        ys, xs = np.nonzero(m)
    rng = _rng(seed)
    size = math.sqrt(m.sum())
    ops = []
    pick = rng.choice(len(xs), size=min(count, len(xs)), replace=False)
    lens = []
    for k, i in enumerate(pick):
        s = streamline(field, m, (xs[i] + 0.5, ys[i] + 0.5), 1.5, size * rng.uniform(0.35, 0.9))
        if len(s) < 4:
            continue
        L = path_length(s)
        w = width * rng.uniform(0.6, 1.25)
        if L < 4 * w + 2:
            continue
        col = dark_rgb if k % 4 != 3 else base_rgb
        ops.append(Stroke(pts=s.astype(np.float32), width=taper_widths(s, w, "both", 0.15), color=tuple(col),
                          opacity=float(rng.uniform(0.55, 0.9)), brush="liner", seed=seed + k))
        lens.append(s)
    if shine and lens:
        # the shine: short light strokes along the strands, in a band across the hair on the lit side
        lv = np.array([light[0], light[1]])
        ys2, xs2 = np.nonzero(m)
        cx, cy = xs2.mean(), ys2.mean()
        proj = (xs2 - cx) * lv[0] + (ys2 - cy) * lv[1]
        band_c = np.quantile(proj, 0.72)
        band_w = (np.quantile(proj, 0.95) - np.quantile(proj, 0.05)) * 0.12
        for k in range(max(8, count // 3)):
            j = rng.integers(len(xs2))
            pt = (xs2[j] + 0.5, ys2[j] + 0.5)
            pr = (pt[0] - cx) * lv[0] + (pt[1] - cy) * lv[1]
            if abs(pr - band_c) > band_w:
                continue
            s = streamline(field, m, pt, 1.5, size * rng.uniform(0.08, 0.18))
            if len(s) < 4:
                continue
            w = width * rng.uniform(1.4, 2.6)
            if path_length(s) < 4 * w:
                continue
            ops.append(Stroke(pts=s.astype(np.float32), width=taper_widths(s, w, "both", 0.1), color=tuple(light_rgb),
                              opacity=float(rng.uniform(0.7, 0.95)), brush="liner", seed=seed + 1000 + k))
    return ops


# ============================================================================================ eye

def eye_ops(mask: np.ndarray, iris_rgb, look=(0.0, 0.0), lid_rgb=None, light: Optional[np.ndarray] = None,
            seed: int = 1) -> list:
    """Iris with a top-dark gradient, pupil, two catch-lights, upper-lid shadow; fits the eye area's box."""
    from .ops import Fill, Stroke
    from .draw_core import lch_to_rgb, rgb_to_lch, shadow_of, light_of
    m = np.asarray(mask, bool)
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return []
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2 + look[0] * 0.18 * w
    cy = (y0 + y1) / 2 + look[1] * 0.12 * h + 0.04 * h
    ih = 0.86 * h
    iw = min(0.62 * w, 0.78 * ih)
    deep = shadow_of(iris_rgb, 1.4)
    bright = light_of(iris_rgb, 1.2)
    ops = []
    ring = ellipse_pts(cx, cy, iw / 2, ih / 2, n=64)
    ops.append(Fill(polygons=[ring], color=deep, color2=bright, p0=(cx, cy - ih / 2), p1=(cx, cy + ih / 2.2)))
    inner = ellipse_pts(cx, cy + 0.06 * ih, iw * 0.36, ih * 0.36, n=48)
    ops.append(Fill(polygons=[inner], color=shadow_of(iris_rgb, 0.6), opacity=0.55, feather=max(1.0, 0.04 * ih)))
    pup = ellipse_pts(cx, cy + 0.03 * ih, iw * 0.2, ih * 0.24, n=40)
    ops.append(Fill(polygons=[pup], color=lch_to_rgb(8, 6, rgb_to_lch(iris_rgb)[2])))
    # lower glow
    glow = ellipse_pts(cx, cy + 0.28 * ih, iw * 0.3, ih * 0.12, n=40)
    ops.append(Fill(polygons=[glow], color=light_of(iris_rgb, 2.0), opacity=0.7, feather=max(1.0, 0.05 * ih)))
    # upper lid shadow over the white and the iris top
    lid = lid_rgb or (120, 110, 140)
    band = np.array([[x0 - 2, y0 - 2], [x1 + 2, y0 - 2], [x1 + 2, y0 + 0.24 * h], [cx, y0 + 0.34 * h],
                     [x0 - 2, y0 + 0.24 * h]], np.float32)
    ops.append(Fill(polygons=[catmull_rom(band, closed=True, spacing=2.0)], color=tuple(lid), opacity=0.45,
                    feather=max(1.0, 0.08 * h)))
    lv = light if light is not None else np.array([-0.6, -0.6, 0.5])
    sx, sy = -1 if lv[0] < 0 else 1, -1
    r1 = max(1.6, 0.13 * ih)
    ops.append(Fill(polygons=[ellipse_pts(cx + sx * 0.22 * iw, cy + sy * 0.2 * ih, r1, r1 * 1.1, n=32)],
                    color=(255, 255, 255)))
    r2 = max(1.2, 0.06 * ih)
    ops.append(Fill(polygons=[ellipse_pts(cx - sx * 0.2 * iw, cy + 0.22 * ih, r2, r2, n=24)], color=(255, 255, 255),
                    opacity=0.85))
    return ops


# ============================================================================================ fold / pattern

def fold_ops(path_px: np.ndarray, width: float, line_rgb, shadow_rgb, light: np.ndarray, depth: float = 1.0,
             base_rgb=None) -> list:
    from .ops import Fill, Stroke
    path = catmull_rom(path_px, spacing=1.5)
    if path_length(path) < 4 * width:
        return []
    ops = [Stroke(pts=path.astype(np.float32), width=taper_widths(path, width, "both", 0.1), color=tuple(line_rgb),
                  brush="liner")]
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    side = -1.0 if (n.mean(0) @ light[:2]) > 0 else 1.0     # shadow on the side away from the light
    L = path_length(path)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    prof = np.sin(np.clip(s / max(L, 1e-6), 0, 1) * math.pi) ** 0.7
    wedge = path + side * n * (prof[:, None] * depth * max(4.0, 0.12 * L))
    poly = np.vstack([path, wedge[::-1]]).astype(np.float32)
    ops.insert(0, Fill(polygons=[poly], color=tuple(shadow_rgb), opacity=0.8, feather=max(1.5, 0.03 * L)))
    if base_rgb is not None:
        from .draw_core import light_of, lum
        if lum(base_rgb) < 45:
            # on dark cloth a fold reads by the light catching its ridge, on the lit side of the crease
            edge = path - side * n * (prof[:, None] * max(2.0, 1.2 * width))
            ops.append(Stroke(pts=edge.astype(np.float32), width=taper_widths(edge, max(1.0, 1.4 * width), "both", 0.1),
                              color=tuple(light_of(base_rgb, 2.2)), opacity=0.8, brush="liner"))
    return ops


def pattern_ops(box, kind: str, rgb, spacing: float, angle: float, size: float, seed: int = 1) -> list:
    """Marks covering the pixel box; the caller clips them to the region."""
    from .ops import Stroke
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    R = math.hypot(x1 - x0, y1 - y0) / 2 + spacing
    ops = []
    rgb = tuple(int(c) for c in rgb)
    if kind in ("hatch", "cross", "stripes"):
        angs = [angle] + ([angle + 90] if kind == "cross" else [])
        for a in angs:
            u = np.array([math.cos(math.radians(a)), -math.sin(math.radians(a))])
            v = np.array([-u[1], u[0]])
            k = -R
            while k <= R:
                c = np.array([cx, cy]) + v * k
                pts = np.array([c - u * R, c + u * R], np.float32)
                w = size if kind != "stripes" else max(size, spacing * 0.45)
                ops.append(Stroke(pts=resample(pts, 3.0).astype(np.float32), width=w, color=rgb,
                                  brush="pen" if kind != "stripes" else "liner"))
                k += spacing
    elif kind in ("dots", "tone"):
        a = math.radians(angle or 45)
        u = np.array([math.cos(a), -math.sin(a)])
        v = np.array([-u[1], u[0]])
        n = int(2 * R / spacing) + 2
        for i in range(-n // 2, n // 2 + 1):
            for j in range(-n // 2, n // 2 + 1):
                q = np.array([cx, cy]) + u * i * spacing + v * j * spacing
                if x0 - spacing <= q[0] <= x1 + spacing and y0 - spacing <= q[1] <= y1 + spacing:
                    ops.append(Stroke(pts=np.array([q], np.float32), width=max(3.0, size), color=rgb, brush="round"))
    elif kind == "checks":
        from .ops import Fill
        n = int(2 * R / spacing) + 2
        a = math.radians(angle)
        u = np.array([math.cos(a), -math.sin(a)]) * spacing
        v = np.array([-u[1], u[0]])
        for i in range(-n // 2, n // 2 + 1):
            for j in range(-n // 2, n // 2 + 1):
                if (i + j) % 2:
                    continue
                o = np.array([cx, cy]) + u * i + v * j
                sq = np.array([o, o + u, o + u + v, o + v], np.float32)
                if x0 - spacing * 2 <= o[0] <= x1 + spacing * 2 and y0 - spacing * 2 <= o[1] <= y1 + spacing * 2:
                    ops.append(Fill(polygons=[sq], color=rgb))
    return ops


# ============================================================================================ scatter

STAMPS = ("dot", "leaf", "petal", "grass", "star", "dash", "needle", "snow")


def _stamp(kind: str, c, size: float, ang: float, rgb, opacity: float, seed: int):
    from .ops import Fill, Stroke
    ca, sa = math.cos(ang), math.sin(ang)

    def tf(P):
        P = np.asarray(P, float)
        return np.stack([c[0] + P[:, 0] * ca - P[:, 1] * sa, c[1] + P[:, 0] * sa + P[:, 1] * ca], 1).astype(
            np.float32)
    s = size
    if kind in ("dot", "snow"):
        return Stroke(pts=np.array([c], np.float32), width=max(3.0, s), color=rgb, opacity=opacity,
                      brush="soft" if kind == "snow" else "round", seed=seed)
    if kind == "leaf":
        t = np.linspace(0, math.pi, 12)
        top = np.stack([np.linspace(-s, s, 12), 0.38 * s * np.sin(t)], 1)
        return Fill(polygons=[tf(np.vstack([top, top[::-1] * [1, -1]]))], color=rgb, opacity=opacity)
    if kind == "petal":
        t = np.linspace(0, 2 * math.pi, 24)
        r = s * (0.55 + 0.45 * np.cos(t))
        return Fill(polygons=[tf(np.stack([r * np.cos(t) * 0.6 + 0.3 * s, r * np.sin(t) * 0.42], 1))], color=rgb,
                    opacity=opacity)
    if kind == "star":
        k = 10
        t = np.linspace(0, 2 * math.pi, k, endpoint=False) - math.pi / 2
        r = np.where(np.arange(k) % 2 == 0, s, 0.4 * s)
        return Fill(polygons=[tf(np.stack([r * np.cos(t), r * np.sin(t)], 1))], color=rgb, opacity=opacity)
    if kind in ("grass", "needle", "dash"):
        L = s * (2.6 if kind == "grass" else 1.6 if kind == "needle" else 1.2)
        bend = 0.25 * L if kind == "grass" else 0.0
        P = np.array([[0, 0], [L * 0.5, bend * 0.3], [L, bend]])
        path = catmull_rom(tf(P), spacing=1.0)
        w = max(1.0, s * (0.22 if kind == "grass" else 0.12 if kind == "needle" else 0.3))
        if path_length(path) < 4 * w:
            return None
        return Stroke(pts=path.astype(np.float32), width=taper_widths(path, w, "end" if kind != "dash" else "both",
                                                                       0.1), color=rgb, opacity=opacity,
                      brush="liner", seed=seed)
    return None


def scatter_ops(kind: str, *, mask: Optional[np.ndarray] = None, path_px: Optional[np.ndarray] = None,
                count: int = 40, size: float = 8.0, colors: Sequence = ((0, 0, 0),), angle: Optional[float] = None,
                jitter: float = 0.5, spread: float = 0.0, opacity: float = 1.0, seed: int = 1) -> list:
    rng = _rng(seed)
    pts, angs = [], []
    if path_px is not None and len(path_px) >= 2:
        path = resample(catmull_rom(path_px, spacing=1.0), 1.0)
        L = path_length(path)
        t = np.gradient(path, axis=0)
        for i in range(count):
            s = (i + rng.uniform(0.2, 0.8)) / count * L
            j = min(len(path) - 1, int(s))
            nrm = np.array([-t[j, 1], t[j, 0]])
            nrm /= np.linalg.norm(nrm) + 1e-9
            pts.append(path[j] + nrm * rng.normal(0, max(spread, 1e-6)))
            angs.append(math.atan2(t[j, 1], t[j, 0]))
    elif mask is not None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        idx = rng.choice(len(xs), size=min(count, len(xs)), replace=len(xs) < count)
        for i in idx:
            pts.append((xs[i] + rng.uniform(0, 1), ys[i] + rng.uniform(0, 1)))
            angs.append(rng.uniform(0, 2 * math.pi))
    ops = []
    for k, (c, a) in enumerate(zip(pts, angs)):
        if angle is not None:
            a = math.radians(-angle) + rng.normal(0, 0.35 * jitter)
        elif kind in ("grass", "needle"):
            a = -math.pi / 2 + rng.normal(0, 0.45 * jitter) if path_px is None else a + rng.normal(0, 0.4 * jitter)
        sz = size * float(np.clip(1 + rng.normal(0, 0.3 * jitter), 0.4, 1.8))
        col = tuple(int(v) for v in colors[k % len(colors)])
        if len(colors) == 1 and jitter > 0:
            f_ = 1 + rng.normal(0, 0.06 * jitter)          # lighter / darker, never a different hue
            col = tuple(int(np.clip(v * f_, 0, 255)) for v in col)
        op = _stamp(kind, c, sz, a, col, opacity, seed + k)
        if op is not None:
            ops.append(op)
    return ops


def stamp_ops(kind: str, pts_px, size: float, colors: Sequence, angle: Optional[float] = None, opacity: float = 1.0,
              seed: int = 1) -> list:
    """One stamp on each given point, in the given colours (keys on an instrument, buttons, stars placed by hand)."""
    ops = []
    for k, c in enumerate(np.asarray(pts_px, float).reshape(-1, 2)):
        a = math.radians(-angle) if angle is not None else -math.pi / 2
        op = _stamp(kind, tuple(c), size, a, tuple(int(v) for v in colors[k % len(colors)]), opacity, seed + k)
        if op is not None:
            ops.append(op)
    return ops


# ============================================================================================ painterly brushwork

def bend_path(pts, amount: float) -> np.ndarray:
    """Bow a path sideways by ``amount`` x its length at the middle: the arc a wrist gives a brush stroke."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(p) < 3 or abs(amount) < 1e-6:
        return p
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    L = float(s[-1])
    chord = p[-1] - p[0]
    nc = float(np.hypot(*chord))
    if L < 1e-6 or nc < 1e-6:
        return p
    nrm = np.array([-chord[1], chord[0]]) / nc
    return p + nrm[None, :] * (amount * L * np.sin(np.pi * s / L))[:, None]


def stroke_widths(pts, width: float, land: float = 0.5, lift: float = 0.14) -> np.ndarray:
    """Pressure of a loaded brush along a stroke: it lands blunt (``land`` x width, full within the first 15% of
    the stroke), is full in the body and lifts off to a point (``lift`` x width, the taper of ``taper_widths``).
    Never below 1 px."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    w = max(1.0, float(width))
    if len(p) < 2:
        return np.full(max(len(p), 1), w, np.float32)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    x = np.clip(s / max(1e-6, 0.15 * s[-1]), 0.0, 1.0)
    f = land + (1.0 - land) * x * x * (3 - 2 * x)
    return np.maximum(1.0, f * taper_widths(p, w, "end", lift)).astype(np.float32)


def painted_mark(pts, widths, rgb, opacity: float, seed: int, tip: str = "flat", own: bool = False) -> list:
    """One painter's mark as ops: with a bristle tip (flat, dry) a soft body carries the colour and soft edges and
    the bristles inside it leave their streaks at half the opacity; any other tip is one stroke."""
    from .ops import Stroke
    pts = np.asarray(pts, np.float32)
    w = np.asarray(widths, np.float32)
    col = tuple(int(c) for c in np.clip(np.rint(np.asarray(rgb, float)), 0, 255))
    if tip not in ("flat", "dry"):
        return [Stroke(pts=pts, width=w, color=col, opacity=float(opacity), brush=tip, seed=int(seed), own=own)]
    return [Stroke(pts=pts, width=np.maximum(1.0, w * 1.1), color=col, opacity=float(opacity), brush="soft",
                   seed=int(seed), own=own),
            Stroke(pts=pts, width=np.maximum(1.0, w * 0.8), color=col, opacity=float(0.5 * opacity), brush=tip,
                   seed=int(seed), own=own)]


def pickup_colours(under: Optional[np.ndarray], size: float) -> Optional[np.ndarray]:
    """The current picture softened over about a brush width: the colour a wet brush picks up where it lands."""
    if under is None:
        return None
    u = np.asarray(under)
    if u.ndim != 3 or u.shape[2] < 3:
        return None
    u = u[..., :3].astype(np.float32)
    return cv2.GaussianBlur(u, (0, 0), max(0.6, 0.35 * float(size)))


def _sample(img: np.ndarray, pt) -> np.ndarray:
    H, W = img.shape[:2]
    x = min(max(int(pt[0]), 0), W - 1)
    y = min(max(int(pt[1]), 0), H - 1)
    return img[y, x, :3].astype(np.float64)


def brush_ops(mask: np.ndarray, field, base_rgb, shadow_rgb, light_rgb, light: np.ndarray, size: float,
              brush: str = "flat", density: float = 1.0, seed: int = 1, lighting: Optional[np.ndarray] = None,
              under: Optional[np.ndarray] = None, pickup: Tuple[float, float] = (0.25, 0.4),
              opacity: Tuple[float, float] = (0.35, 0.7), bend: float = 0.07) -> list:
    """Brush strokes along the field over the region, colour from the region's own lighting (lighter on the lit
    side). Each stroke is a painter's mark: it lands blunt, is full in the body and lifts off to a point, with a
    slight arc, so no mark reads as a rectangle; a bristle tip (flat, dry) paints a soft body with its streaks inside
    (``painted_mark``). With ``under`` (the current picture, uint8 HxWx3) every stroke mixes ``pickup`` (25-40%) of
    the colour under the place it lands into its own colour: wet paint, so neighbouring strokes and the colours
    already there blend; and where the region is already painted with a clear value structure (a sky band light
    toward the horizon, earlier shading) each stroke starts from the colour painted there, lit or shaded at half
    strength, instead of laying the swatch's dark side over a light horizon."""
    from .draw_shade import lighting as _light
    m = np.asarray(mask, bool)
    if m.sum() < 30:
        return []
    rng = _rng(seed)
    if lighting is None:
        I, (y0, y1, x0, x1) = _light(m, light, 0.8)
        full = np.zeros(m.shape, np.float32)
        full[y0:y1, x0:x1] = I
        lighting = full
    wet = pickup_colours(under, size)
    if wet is not None and wet.shape[:2] != m.shape:
        wet = None
    broad, lead = None, 0.0
    if wet is not None:
        # the values already painted over the region, over a few brush widths: a clear structure (a sky band light
        # toward the horizon, earlier shading) leads, so the strokes model around it instead of replacing it
        sig = max(1.0, 1.5 * float(size))
        mf = m.astype(np.float32)
        broad = cv2.GaussianBlur(np.asarray(under)[..., :3].astype(np.float32) * mf[..., None], (0, 0), sig)
        broad /= np.maximum(cv2.GaussianBlur(mf, (0, 0), sig), 1e-4)[..., None]    # the region's own colours only
        l_lo, l_hi = (float(v) for v in np.quantile((broad @ np.float32([0.2126, 0.7152, 0.0722]))[m], [0.1, 0.9]))
        lead = float(np.clip((l_hi - l_lo - 8.0) / 30.0, 0.0, 1.0))
    vals = lighting[m]
    lo, hi = np.quantile(vals, 0.1), np.quantile(vals, 0.9)
    n = int(density * m.sum() / (size * size * 1.35)) + 4
    n = min(n, 2500)
    ys, xs = np.nonzero(m)
    idx = rng.choice(len(xs), size=n, replace=len(xs) < n)
    base = np.asarray(base_rgb, float)
    sh = np.asarray(shadow_rgb, float)
    li = np.asarray(light_rgb, float)
    k_lo, k_hi = sorted(float(np.clip(v, 0.0, 1.0)) for v in pickup)
    o_lo, o_hi = sorted(float(np.clip(v, 0.05, 1.0)) for v in opacity)
    ops = []
    for k, i in enumerate(idx):
        x, y = xs[i] + 0.5, ys[i] + 0.5
        s = streamline(field, m, (x, y), 1.5, size * rng.uniform(2.6, 5.2))
        if len(s) < 3:
            continue
        if rng.random() < 0.5:
            s = s[::-1]                                 # strokes go both ways along the field
        s = bend_path(resample(s, 1.5), rng.uniform(-bend, bend))
        if hi - lo < 0.12:
            t = 0.5                                     # flat light on this mass: vary the colour only a little
        else:
            t = 0.2 + 0.6 * float(np.clip((lighting[ys[i], xs[i]] - lo) / (hi - lo), 0, 1))
        t = float(np.clip(t + rng.normal(0, 0.07), 0, 1))
        col = sh + (base - sh) * (t / 0.5) if t < 0.5 else base + (li - base) * ((t - 0.5) / 0.5)
        if lead > 0:                                    # the painted colour there, lit or shaded at half strength
            col = (1 - lead) * col + lead * (_sample(broad, s[len(s) // 2]) + 0.5 * (col - base))
        col = col * (1 + rng.normal(0, 0.02, 3))
        if wet is not None:
            kk = rng.uniform(k_lo, k_hi)
            col = (1 - kk) * col + kk * _sample(wet, s[0])
        col = np.clip(col, 0, 255)
        w = size * rng.uniform(0.75, 1.15)
        if path_length(s) < 4 * w:
            continue                                    # too short to read as a stroke
        ops += painted_mark(s, stroke_widths(s, w, rng.uniform(0.4, 0.65), rng.uniform(0.08, 0.18)), col,
                            rng.uniform(o_lo, o_hi), seed + k, brush)
    return ops


# ============================================================================================ bands

def band_polygon(curve_px: np.ndarray, W: int, H: int, side: str, rough: float = 0.0, seed: int = 1,
                 to_curve: Optional[np.ndarray] = None) -> np.ndarray:
    """Closed polygon between a curve that spans the canvas and the top / bottom edge (or a second curve)."""
    c = catmull_rom(curve_px, spacing=2.0)
    c = c[np.argsort(c[:, 0], kind="stable")] if to_curve is None else c
    if c[0, 0] > 0:
        c = np.vstack([[0, c[0, 1]], c])
    if c[-1, 0] < W:
        c = np.vstack([c, [W, c[-1, 1]]])
    if rough > 0:
        rng = _rng(seed)
        c = resample(c, 3.0)
        n = len(c)
        noise = np.zeros(n)
        amp = rough * 0.025 * max(W, H)
        wave = max(6.0, 0.05 * max(W, H)) / 3.0
        for octave in range(4):
            k = max(2, int(n / (wave / 2 ** octave)))
            knots = rng.normal(0, 1, k)
            noise += amp / 2 ** octave * np.interp(np.linspace(0, k - 1, n), np.arange(k), knots)
        c[:, 1] += noise
    if to_curve is not None:
        c2 = catmull_rom(to_curve, spacing=2.0)
        c2 = c2[np.argsort(c2[:, 0], kind="stable")]
        if c2[0, 0] > 0:
            c2 = np.vstack([[0, c2[0, 1]], c2])
        if c2[-1, 0] < W:
            c2 = np.vstack([c2, [W, c2[-1, 1]]])
        return np.vstack([c, c2[::-1]]).astype(np.float32)
    edge = -1.0 if side == "above" else H + 1.0
    return np.vstack([c, [[W + 1, c[-1, 1]], [W + 1, edge], [-1, edge], [-1, c[0, 1]]]]).astype(np.float32)


def rough_polygon(poly: np.ndarray, rough: float, scale: float, seed: int = 1) -> np.ndarray:
    """Displace a closed outline along its normals with fractal noise (foliage, rock, fur edges)."""
    if rough <= 0:
        return poly
    rng = _rng(seed)
    c = resample(np.vstack([poly, poly[:1]]), 2.0)[:-1]
    n = len(c)
    t = np.gradient(np.vstack([c[-1:], c, c[:1]]), axis=0)[1:-1]
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    noise = np.zeros(n)
    amp = rough * 0.025 * scale                 # rough 1 = edges ragged by 2.5% of the picture
    wave = max(8.0, 0.05 * scale) / 2.0         # first octave: bumps 5% of the picture apart (in 2px samples)
    for octave in range(4):
        k = max(3, int(n / (wave / 2 ** octave)))
        knots = rng.normal(0, 1, k + 1)
        knots[-1] = knots[0]
        noise += amp / 2 ** octave * np.interp(np.linspace(0, k, n, endpoint=False), np.arange(k + 1), knots)
    return (c + nrm * noise[:, None]).astype(np.float32)
