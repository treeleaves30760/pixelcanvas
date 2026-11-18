"""The raster brush engine: renders Stroke / Fill ops onto float RGBA layers.

Every stroke is rendered inside its own bounding box only.  The footprint is drawn with OpenCV at a local supersampled
resolution (4x for fine strokes, less for broad ones) and box-filtered down, which gives clean antialiased edges at
any width.  Width can change per point (pressure taper) and colour can change per point (the colour of each pixel is
that of the nearest path point).  Brush character comes from the preset:

- ``pen``     G-pen: hard edge, tapered ends when the width is constant.
- ``pencil``  graphite: thin, tapered, broken by a fixed paper grain, slight pressure wobble.
- ``marker``  hard, even, slightly transparent; no taper.
- ``round``   opaque round brush with a soft 1 px edge.
- ``flat``    bristle brush: parallel bristle tracks with uneven load -> directional streaks along the stroke.
- ``dry``     dry brush: bristles run out of paint and skip over the paper tooth -> broken strokes.
- ``soft``    airbrush: Gaussian falloff.
- ``wash``    watercolour: transparent, darker rim where the pigment pools, granulation.
- ``liner``   hard pen without automatic taper (line art with explicit per-point widths).
- ``bristle`` fine bristle tip: separate parallel hairs with gaps between them (grass, sand ripples, wood grain).
- ``sponge``  sponge / speckle tip: broken, grainy coverage (stone, foliage, gravel).
- ``pixel``   hard 1x footprint without antialiasing (pixel art, exact masks).

Strokes are deterministic: all randomness comes from ``np.random.default_rng(stroke.seed)`` and from a fixed paper
texture addressed by absolute canvas position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np
from scipy import ndimage as ndi

from .ops import Fill, Stroke, rle_to_mask

__all__ = ["BrushSpec", "PRESETS", "apply_ops", "apply_ops_progressive", "render_stroke", "render_fill",
           "op_bbox", "new_layer", "paper_grain"]


@dataclass(frozen=True)
class BrushSpec:
    kind: str = "solid"          # solid | bristle
    opacity: float = 1.0         # multiplies the stroke's opacity
    edge: float = 0.0            # extra edge softness: Gaussian sigma = edge_px + edge * width
    edge_px: float = 0.0
    core: float = 1.0            # footprint width multiplier (soft brushes draw a narrower core, then blur)
    taper_in: float = 0.0        # automatic taper lengths, in widths (only when the stroke width is constant)
    taper_out: float = 0.0
    taper_min: float = 0.15      # width fraction at the very tip
    grain: float = 0.0           # paper grain strength (0..1)
    grain_scale: float = 1.0
    jitter: float = 0.0          # width wobble along the stroke (pencil pressure)
    bristles: int = 0            # bristle count for kind == bristle
    load_var: float = 0.0        # per-bristle load variation
    deplete: float = 0.0         # paint used up over the stroke length (0..1)
    skip: float = 0.0            # load threshold below which a bristle leaves no mark
    noise: float = 0.0           # along-stroke load noise
    rim: float = 0.0             # watercolour rim darkening
    aa: bool = True
    hair: float = 1.7            # bristle thickness as a share of the width / bristle count
    hair_px: float = 0.0         # bristle kinds: when > 0, one bristle per ``hair_px`` of width (fine brushes keep
                                 # separate hairs with gaps instead of a smear of sub-pixel ones)
    speckle: float = 0.0         # sponge / speckle tip: share of the footprint the tip leaves bare (0..1)
    speckle_scale: float = 0.7   # grain size of the speckle, px


PRESETS: dict = {
    "pen": BrushSpec(taper_in=3.0, taper_out=5.0, taper_min=0.12),
    "liner": BrushSpec(),
    "pencil": BrushSpec(opacity=0.85, taper_in=2.0, taper_out=3.0, taper_min=0.3, grain=0.75, jitter=0.12,
                        edge_px=0.15),
    "marker": BrushSpec(opacity=0.8),
    "round": BrushSpec(edge_px=0.6),
    "flat": BrushSpec(kind="bristle", bristles=14, load_var=0.35, deplete=0.25, skip=0.05, noise=0.15,
                      edge_px=0.35),
    "dry": BrushSpec(kind="bristle", bristles=16, load_var=0.5, deplete=0.7, skip=0.42, noise=0.35, grain=0.6,
                     edge_px=0.3),
    "soft": BrushSpec(core=0.55, edge=0.22, edge_px=0.5),
    "wash": BrushSpec(opacity=0.4, edge_px=0.8, edge=0.04, rim=1.2, grain=0.2, grain_scale=2.0),
    "pixel": BrushSpec(aa=False),
    # textured tips (one colour along one path; the tip only shapes the coverage)
    "bristle": BrushSpec(kind="bristle", bristles=8, hair_px=1.4, hair=0.75, load_var=0.35, deplete=0.2,
                         skip=0.1, noise=0.25, edge_px=0.2),
    "sponge": BrushSpec(speckle=0.45, speckle_scale=0.7, edge_px=0.3),
}


def new_layer(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 4), np.float32)


# ------------------------------------------------------------------------------------------------ paper grain
_GRAIN = {}


def paper_grain(scale: float = 1.0) -> np.ndarray:
    """A fixed, tileable 512x512 paper-tooth texture in 0..1 (same for every run)."""
    key = round(float(scale), 3)
    if key not in _GRAIN:
        rng = np.random.default_rng(1234567)
        n = 512
        acc = np.zeros((n, n), np.float32)
        for sigma, amp in ((0.6 * scale, 0.55), (1.6 * scale, 0.3), (4.0 * scale, 0.15)):
            z = rng.standard_normal((n, n)).astype(np.float32)
            z = ndi.gaussian_filter(z, sigma, mode="wrap")
            acc += amp * z / (z.std() + 1e-6)
        acc = (acc - acc.mean()) / (acc.std() + 1e-6)
        _GRAIN[key] = np.clip(0.5 + 0.22 * acc, 0, 1).astype(np.float32)
    return _GRAIN[key]


def _grain_patch(x0, y0, h, w, scale=1.0):
    g = paper_grain(scale)
    ys = (np.arange(y0, y0 + h) % 512)
    xs = (np.arange(x0, x0 + w) % 512)
    return g[np.ix_(ys, xs)]


# ------------------------------------------------------------------------------------------------ geometry
def _resample(pts, widths, colors, spacing):
    """Resample a polyline at ~spacing px. Returns pts [n,2], widths [n], colors [n,3] or None, arclen [n]."""
    d = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)) if len(pts) > 1 else np.zeros(0)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = float(s[-1])
    if L < 1e-6:
        c = None if colors is None else colors[:1].astype(np.float32)
        return pts[:1].astype(np.float64), widths[:1].astype(np.float64), c, np.zeros(1)
    n = max(2, int(np.ceil(L / spacing)) + 1)
    t = np.linspace(0.0, L, n)
    # keep original vertices exactly (sharp corners) by merging them into the sample positions
    t = np.unique(np.concatenate([t, s]))
    x = np.interp(t, s, pts[:, 0])
    y = np.interp(t, s, pts[:, 1])
    w = np.interp(t, s, widths)
    c = None
    if colors is not None:
        c = np.stack([np.interp(t, s, colors[:, k].astype(np.float64)) for k in range(3)], 1).astype(np.float32)
    return np.stack([x, y], 1), w, c, t


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _noise1d(rng, s, scale, octaves=2):
    """Smooth 1D noise over arclength positions s (values ~N(0,1))."""
    if len(s) < 2:
        return np.zeros(len(s))
    L = s[-1]
    out = np.zeros(len(s))
    amp = 1.0
    for o in range(octaves):
        step = max(scale / (2 ** o), 1.0)
        k = int(L / step) + 3
        knots = rng.standard_normal(k)
        out += amp * np.interp(s / step, np.arange(k), knots)
        amp *= 0.5
    return out


def _stroke_shape(st: Stroke, spec: BrushSpec, rng):
    pts = st.pts.astype(np.float64)
    n0 = len(pts)
    const_w = not isinstance(st.width, np.ndarray)
    widths = np.full(n0, float(st.width)) if const_w else st.width.astype(np.float64)
    colors = st.color if isinstance(st.color, np.ndarray) else None
    wmin = max(float(widths.min()) if len(widths) else 1.0, 0.3)
    spacing = float(np.clip(wmin * 0.35, 0.35, 3.0)) if spec.aa else 1.0
    P, W, C, S = _resample(pts, widths, colors, spacing)
    L = float(S[-1])
    if const_w and (spec.taper_in > 0 or spec.taper_out > 0) and L > 0:
        w0 = float(st.width)
        lin = min(spec.taper_in * w0 + 1.0, 0.45 * L)
        lout = min(spec.taper_out * w0 + 1.0, 0.45 * L)
        f = np.ones_like(S)
        if lin > 0:
            f *= spec.taper_min + (1 - spec.taper_min) * _smoothstep(S / lin)
        if lout > 0:
            f *= spec.taper_min + (1 - spec.taper_min) * _smoothstep((L - S) / lout)
        W = W * f
    if spec.jitter > 0 and L > 0:
        W = W * np.clip(1 + spec.jitter * _noise1d(rng, S, 12.0), 0.4, 1.6)
    W = W * spec.core
    return P, np.maximum(W, 0.05), C, S, L


def _cv_thickness(w_ss):
    """cv2 draws a thick line of thickness 2k (or 1) as 2k+1 pixels wide: pick the closest achievable width."""
    k = np.rint((np.asarray(w_ss, np.float64) - 1.0) / 2.0).astype(np.int64)
    return np.where(k <= 0, 1, 2 * k)


def _supersample(wmax):
    if wmax <= 4:
        return 8
    if wmax <= 8:
        return 4
    if wmax <= 12:
        return 3
    if wmax <= 64:
        return 2
    return 1


def _normals(P):
    if len(P) < 2:
        return np.tile([[0.0, 1.0]], (len(P), 1))
    t = np.gradient(P, axis=0)
    if len(P) >= 5:
        k = np.ones(5) / 5.0
        t = np.stack([np.convolve(np.pad(t[:, i], 2, mode="edge"), k, "valid") for i in range(2)], 1)
    nrm = np.sqrt((t ** 2).sum(1, keepdims=True)) + 1e-9
    t = t / nrm
    return np.stack([-t[:, 1], t[:, 0]], 1)


# ------------------------------------------------------------------------------------------------ rendering
def render_stroke(st: Stroke, shape) -> Optional[tuple]:
    """Render a stroke for a layer of ``shape`` (H, W, ...).

    Returns (x0, y0, alpha [h,w] float32, rgb [h,w,3] float32 0..1 or [3]) clipped to the layer, or None."""
    H, Wd = shape[:2]
    spec = PRESETS.get(st.brush)
    if spec is None:
        raise KeyError(f"unknown brush {st.brush!r}; presets: {', '.join(sorted(PRESETS))}")
    rng = np.random.default_rng(st.seed)
    P, Wp, C, S, L = _stroke_shape(st, spec, rng)
    wmax = float(Wp.max())
    sigma = spec.edge_px + spec.edge * wmax / max(spec.core, 1e-6)
    pad = wmax / 2 + 3 * sigma + 2
    x0 = int(np.floor(P[:, 0].min() - pad))
    y0 = int(np.floor(P[:, 1].min() - pad))
    x1 = int(np.ceil(P[:, 0].max() + pad)) + 1
    y1 = int(np.ceil(P[:, 1].max() + pad)) + 1
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, Wd), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    # render on the clipped box (plus a margin for the blur so edges stay correct near the layer border)
    m = int(np.ceil(3 * sigma)) + 1
    bx0, by0 = max(x0, cx0 - m), max(y0, cy0 - m)
    bx1, by1 = min(x1, cx1 + m), min(y1, cy1 + m)
    bw, bh = bx1 - bx0, by1 - by0
    ss = _supersample(wmax) if spec.aa else 1
    img = np.zeros((bh * ss, bw * ss), np.uint8)
    SH = 4
    Q = np.rint(((P - [bx0, by0]) * ss - 0.5) * (1 << SH) + 1.0).astype(np.int64)

    if spec.kind == "bristle":
        _draw_bristles(img, P, Wp, S, L, ss, bx0, by0, spec, rng)
    else:
        if len(Q) == 1:
            r = max(0, int(round((Wp[0] * ss - 1) / 2)))
            c = (int(Q[0, 0]), int(Q[0, 1]))
            if r == 0:
                cv2.circle(img, c, 0, 255, -1, cv2.LINE_8, SH)
            else:
                cv2.circle(img, c, r << SH, 255, -1, cv2.LINE_8, SH)
        else:
            seg_th = _cv_thickness((Wp[:-1] + Wp[1:]) * (ss / 2))
            if np.all(seg_th == seg_th[0]):
                cv2.polylines(img, [Q.reshape(-1, 1, 2).astype(np.int32)], False, 255, int(seg_th[0]),
                              cv2.LINE_8, SH)
            else:
                ql = Q.tolist()
                tl = seg_th.tolist()
                line = cv2.line
                for i in range(len(ql) - 1):
                    line(img, ql[i], ql[i + 1], 255, tl[i], cv2.LINE_8, SH)
    if ss > 1:
        a = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32) * (1.0 / 255.0)
    else:
        a = img.astype(np.float32) * (1.0 / 255.0)
    if sigma > 0.05:
        a = cv2.GaussianBlur(a, (0, 0), sigma)
        if spec.core < 1.0:
            a = np.clip(a * 1.25, 0, 1)
    # crop to the clipped box
    a = a[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]
    h, w = a.shape
    if spec.speckle > 0:
        # the sponge touches the paper only where its pores do: a grain pattern, shifted per stroke (seeded)
        ox, oy = (int(v) for v in rng.integers(0, 512, 2))
        g = _grain_patch(cx0 + ox, cy0 + oy, h, w, spec.speckle_scale)
        a = a * _smoothstep((g - (0.5 + 0.22 * (spec.speckle - 0.5) * 2.2)) * 8.0 + 0.5)
    if spec.grain > 0:
        g = _grain_patch(cx0, cy0, h, w, spec.grain_scale)
        if spec.kind == "bristle":
            # dry bristles skip over the paper tooth
            thr = 1.0 - a
            a = a * _smoothstep((g - thr * 0.9 + 0.25) * 2.5) * (1 - spec.grain) + a * spec.grain * _smoothstep(
                (g - thr + 0.1) * 4)
        elif spec.rim > 0:
            a = a * (1.0 - spec.grain * (g - 0.5) * 1.2)
        else:
            # graphite catches on the ridges of the paper
            a = a * np.clip(1.0 - spec.grain + spec.grain * (0.25 + 1.6 * (g - 0.5) + 0.75 * a), 0, 1)
    rgb_scale = None
    if spec.rim > 0:
        sg = max(0.8, 0.12 * wmax)
        inner = cv2.GaussianBlur(a, (0, 0), sg)
        rim = np.clip((a - inner) * 3.0, 0, 1)
        a = a * (1.0 + spec.rim * rim)
        rgb_scale = 1.0 - 0.18 * rim
    k = spec.opacity * st.opacity
    if k != 1.0 or spec.rim > 0:
        a = np.minimum(a * k, 1.0)
    if a.dtype != np.float32:
        a = a.astype(np.float32)

    # colour
    if C is None:
        rgb = np.asarray(st.color, np.float32) / 255.0
        if rgb_scale is not None:
            rgb = rgb[None, None, :] * rgb_scale[..., None]
    else:
        # the stroke's own points are assigned last so their exact colours win at their pixels
        P2 = np.vstack([P, st.pts.astype(np.float64)])
        C2 = np.vstack([C, st.color.astype(np.float32)])
        rgb = _color_field(P2, C2, cx0, cy0, h, w)
        if rgb_scale is not None:
            rgb = rgb * rgb_scale[..., None]
    return cx0, cy0, a, rgb


def _color_field(P, C, x0, y0, h, w):
    """Colour of the nearest path point for every pixel of the box."""
    ix = np.floor(P[:, 0]).astype(np.int64) - x0
    iy = np.floor(P[:, 1]).astype(np.int64) - y0
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    if not ok.any():
        # path outside the box: nearest point to the box centre
        d = (P[:, 0] - (x0 + w / 2)) ** 2 + (P[:, 1] - (y0 + h / 2)) ** 2
        return np.broadcast_to(C[int(np.argmin(d))] / 255.0, (h, w, 3)).astype(np.float32)
    seedcol = np.zeros((h, w, 3), np.float32)
    seedcol[iy[ok], ix[ok]] = C[ok] / 255.0
    if h * w == 1:
        return seedcol
    inp = np.ones((h, w), np.uint8)
    inp[iy[ok], ix[ok]] = 0
    idx = ndi.distance_transform_edt(inp, return_distances=False, return_indices=True)
    return seedcol[idx[0], idx[1]]


def _draw_bristles(img, P, Wp, S, L, ss, bx0, by0, spec, rng):
    K = spec.bristles
    if spec.hair_px > 0:
        K = int(np.clip(round(float(Wp.mean()) / spec.hair_px), 2, spec.bristles))
    N = _normals(P)
    SH = 4
    span = (np.arange(K) + 0.5) / K - 0.5          # -0.5..0.5 across the brush
    span = span + rng.uniform(-0.25, 0.25, K) / K
    load0 = np.clip(1.0 - spec.load_var * rng.random(K), 0.05, 1.0)
    bt = int(_cv_thickness(np.array([max(Wp.mean() / K * spec.hair, 0.8) * ss]))[0])
    rel = S / max(L, 1e-6)
    noise_scale = max(4.0, float(Wp.mean()) * 1.5)
    order = np.argsort(load0)  # stronger bristles drawn last
    line = cv2.line
    for k in order:
        off = span[k] * Wp
        B = P + N * off[:, None]
        Q = np.rint(((B - [bx0, by0]) * ss - 0.5) * (1 << SH) + 1.0).astype(np.int64)
        load = load0[k] * (1.0 - spec.deplete * rel) + spec.noise * _noise1d(rng, S, noise_scale)
        # start of the stroke carries more paint
        load = load + 0.15 * (1 - _smoothstep(S / max(Wp.mean(), 1.0)))
        val = np.clip(load, 0, 1)
        on = load > spec.skip
        if len(Q) == 1:
            if on[0]:
                cv2.circle(img, (int(Q[0, 0]), int(Q[0, 1])), int(bt) << SH >> 1, int(255 * val[0]), -1,
                           cv2.LINE_8, SH)
            continue
        ql = Q.tolist()
        v = (255 * (val[:-1] + val[1:]) / 2).astype(int).tolist()
        onl = (on[:-1] & on[1:]).tolist()
        t = int(bt)
        for i in range(len(ql) - 1):
            if onl[i]:
                line(img, ql[i], ql[i + 1], v[i], t, cv2.LINE_8, SH)


def render_fill(f: Fill, shape) -> Optional[tuple]:
    H, W = shape[:2]
    sigma = f.feather / 2.0 if f.feather > 0 else 0.0
    m = int(np.ceil(3 * sigma)) + 1
    if f.polygons is not None and len(f.polygons):
        allp = np.concatenate([p for p in f.polygons if len(p)], 0)
        x0 = int(np.floor(allp[:, 0].min())) - m
        y0 = int(np.floor(allp[:, 1].min())) - m
        x1 = int(np.ceil(allp[:, 0].max())) + m + 1
        y1 = int(np.ceil(allp[:, 1].max())) + m + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 <= x0 or y1 <= y0:
            return None
        bw, bh = x1 - x0, y1 - y0
        ss = 4 if bw * bh < 2_600_000 else 2
        acc = np.zeros((bh * ss, bw * ss), np.uint8)
        SH = 4
        for p in f.polygons:
            if len(p) < 3:
                continue
            q = np.rint(((p.astype(np.float64) - [x0, y0]) * ss - 0.5) * (1 << SH)).astype(np.int32)
            # each ring is XORed in only over its own box (a mass with thousands of small holes stays fast)
            qx0 = max(int(q[:, 0].min()) >> SH, 0)
            qy0 = max(int(q[:, 1].min()) >> SH, 0)
            qx1 = min((int(q[:, 0].max()) >> SH) + 2, acc.shape[1])
            qy1 = min((int(q[:, 1].max()) >> SH) + 2, acc.shape[0])
            if qx1 <= qx0 or qy1 <= qy0:
                continue
            sub = acc[qy0:qy1, qx0:qx1]
            one = np.zeros_like(sub)
            cv2.fillPoly(one, [(q - [qx0 << SH, qy0 << SH]).astype(np.int32).reshape(-1, 1, 2)], 255, cv2.LINE_8, SH)
            np.bitwise_xor(sub, one, out=sub)
        a = cv2.resize(acc, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    elif f.mask_rle is not None:
        rle = f.mask_rle
        if tuple(rle["size"]) != (H, W):
            mk = rle_to_mask(rle)
            mk = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            bx0, by0, bx1, by1 = 0, 0, W, H
        else:
            # decode only the mask's own box
            bx0, by0, bx1, by1 = rle.get("box", [0, 0, W, H])
            counts = np.asarray(rle["counts"], np.int64)
            n = (bx1 - bx0) * (by1 - by0)
            flat = np.repeat((np.arange(counts.size) % 2).astype(bool), counts)[:n]
            if flat.size < n:
                flat = np.concatenate([flat, np.zeros(n - flat.size, bool)])
            mk = flat.reshape(by1 - by0, bx1 - bx0) if n else np.zeros((0, 0), bool)
        if not mk.any():
            return None
        x0, y0 = max(bx0 - m, 0), max(by0 - m, 0)
        x1, y1 = min(bx1 + m, W), min(by1 + m, H)
        a = np.zeros((y1 - y0, x1 - x0), np.float32)
        if tuple(rle["size"]) != (H, W):
            a[:] = mk[y0:y1, x0:x1]
        else:
            a[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = mk
    else:
        return None
    if sigma > 0.05:
        a = cv2.GaussianBlur(a, (0, 0), sigma)
    a = np.clip(a * f.opacity, 0, 1).astype(np.float32)
    h, w = a.shape
    c1 = np.asarray(f.color, np.float32) / 255.0
    if f.color2 is not None and f.p0 is not None and f.p1 is not None:
        c2 = np.asarray(f.color2, np.float32) / 255.0
        d = np.asarray(f.p1, np.float64) - np.asarray(f.p0, np.float64)
        dd = float((d ** 2).sum()) or 1.0
        xs = np.arange(x0, x0 + w) + 0.5 - f.p0[0]
        ys = np.arange(y0, y0 + h) + 0.5 - f.p0[1]
        if f.radial:
            t = np.clip(np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2) / np.sqrt(dd), 0, 1).astype(np.float32)
        else:
            t = np.clip((xs[None, :] * d[0] + ys[:, None] * d[1]) / dd, 0, 1).astype(np.float32)
        rgb = c1[None, None, :] + t[..., None] * (c2 - c1)[None, None, :]
    else:
        rgb = c1
    return x0, y0, a, rgb


# ------------------------------------------------------------------------------------------------ compositing
def _composite(layer, x0, y0, a, rgb, erase, alpha_lock, clip):
    # only the op's own painted box changes (outside it the coverage is 0): crop to it first, a render box is often
    # much larger than the paint in it (a fill's mask box, a stroke's padded box)
    rows = np.flatnonzero(a.max(1) > 0)
    if not len(rows):
        return False
    cols = np.flatnonzero(a[rows[0]:rows[-1] + 1].max(0) > 0)
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    if (r1 - r0, c1 - c0) != a.shape:
        a = a[r0:r1, c0:c1]
        if isinstance(rgb, np.ndarray) and rgb.ndim == 3:
            rgb = rgb[r0:r1, c0:c1]
        x0, y0 = x0 + c0, y0 + r0
    h, w = a.shape
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    if clip is not None:
        c = clip[sl]
        a = a * (c.astype(np.float32) if c.dtype != np.float32 else c)
    m = a > 0
    nz = int(np.count_nonzero(m))
    if not nz:
        return False
    dst = layer[sl]
    if nz < 0.3 * m.size and not erase:
        # sparse paint in its box (bristle tracks, thin strokes): only the covered pixels
        ys, xs = np.nonzero(m)
        av = a[ys, xs]
        cv = rgb[ys, xs] if (isinstance(rgb, np.ndarray) and rgb.ndim == 3) else np.broadcast_to(
            np.asarray(rgb, np.float32).reshape(1, 3), (nz, 3))
        px = dst[ys, xs]
        dav = px[:, 3]
        if alpha_lock:
            av = av * (dav > 0)
            px[:, :3] = px[:, :3] * (1 - av[:, None]) + cv * av[:, None]
        else:
            out_a = av + dav * (1.0 - av)
            safe = np.where(out_a > 1e-6, out_a, 1.0)
            px[:, :3] = (cv * av[:, None] + px[:, :3] * (dav * (1.0 - av))[:, None]) / safe[:, None]
            px[:, 3] = out_a
        dst[ys, xs] = px
        return True
    da = dst[..., 3]
    if erase:
        if alpha_lock:
            return False
        dst[..., 3] = da * (1.0 - a)
        return True
    if alpha_lock:
        a = a * (da > 0)
        a3 = a[..., None]
        dst[..., :3] = dst[..., :3] * (1 - a3) + rgb * a3
        return True
    out_a = a + da * (1.0 - a)
    safe = np.where(out_a > 1e-6, out_a, 1.0)
    a3 = a[..., None]
    dst[..., :3] = (rgb * a3 + dst[..., :3] * (da * (1.0 - a))[..., None]) / safe[..., None]
    dst[..., 3] = out_a
    return True


def _render_pixel_fast(st: Stroke, shape):
    """Hard 1 px stroke whose points are adjacent pixel centres: the footprint is exactly those pixels."""
    H, W = shape[:2]
    xs = np.floor(st.pts[:, 0]).astype(np.int64)
    ys = np.floor(st.pts[:, 1]).astype(np.int64)
    if len(xs) > 1 and (np.abs(np.diff(xs)).max() > 1 or np.abs(np.diff(ys)).max() > 1):
        return False
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not ok.any():
        return None
    xs, ys = xs[ok], ys[ok]
    x0, y0 = int(xs.min()), int(ys.min())
    h, w = int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1
    a = np.zeros((h, w), np.float32)
    a[ys - y0, xs - x0] = st.opacity
    if isinstance(st.color, np.ndarray):
        rgb = np.zeros((h, w, 3), np.float32)
        rgb[ys - y0, xs - x0] = st.color[ok].astype(np.float32) * (1.0 / 255.0)
    else:
        rgb = np.asarray(st.color, np.float32) / 255.0
    return x0, y0, a, rgb


def _render(op, shape):
    if isinstance(op, Stroke):
        if op.brush == "pixel" and float(np.max(op.width)) <= 1.0:
            r = _render_pixel_fast(op, shape)
            if r is not False:
                return r
        return render_stroke(op, shape)
    if isinstance(op, Fill):
        return render_fill(op, shape)
    raise TypeError(f"not an op: {type(op).__name__}")


def _apply_blend(layer, op, clip, alpha_lock):
    """Edge-stopped local averaging of the layer's own paint inside the op's area (see ops.Blend)."""
    from .ops import Fill as _F
    H, W = layer.shape[:2]
    r = render_fill(_F(polygons=op.polygons, mask_rle=op.mask_rle, feather=op.feather), (H, W))
    if r is None:
        return None
    x0, y0, a, _ = r
    h, w = a.shape
    pad = int(np.ceil(3 * op.radius)) + 1
    X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x0 + w + pad, W), min(y0 + h + pad, H)
    crop = layer[Y0:Y1, X0:X1]
    pm = crop.copy()
    pm[..., :3] *= pm[..., 3:4]
    wgt = np.zeros(crop.shape[:2], np.float32)
    wgt[y0 - Y0:y0 - Y0 + h, x0 - X0:x0 - X0 + w] = a
    # only paint inside the area is mixed (normalised convolution): a line next to the area does not bleed in
    inside = (wgt > 0.5).astype(np.float32)
    b = cv2.GaussianBlur(pm * inside[..., None], (0, 0), op.radius)
    nrm = cv2.GaussianBlur(inside, (0, 0), op.radius)[..., None]
    b = np.where(nrm > 1e-3, b / np.maximum(nrm, 1e-3), pm)
    # a step larger than tol (a hard edge in the paint) stays: the blend fades out as the local mean departs from
    # the pixel's own paint
    diff = np.abs(b - pm).max(-1) * 255.0
    e = np.clip(1.0 - diff / op.tol, 0.0, 1.0) ** 2
    wgt *= e * op.strength
    if clip is not None and not op.own:
        wgt *= clip[Y0:Y1, X0:X1]
    if not (wgt > 1e-4).any():
        return None
    out = pm + wgt[..., None] * (b - pm)
    al = out[..., 3:4]
    rgb = np.where(al > 1e-5, out[..., :3] / np.maximum(al, 1e-5), crop[..., :3])
    if alpha_lock:
        crop[..., :3] = rgb
    else:
        crop[..., :3] = rgb
        crop[..., 3:4] = al
    return (X0, Y0, X1, Y1)


SPRAY_SIGMA = 0.25      # an airbrush dab's Gaussian sigma as a share of its diameter
SPRAY_REACH = 3.0       # the dab is rendered out to this many sigmas


def _apply_spray(layer, op, clip, alpha_lock):
    """Airbrush dabs (see ops.Spray), each composited over the layer in turn inside the frisket."""
    H, W = layer.shape[:2]
    fr = None
    if op.polygons is not None or op.mask_rle is not None:
        r = render_fill(Fill(polygons=op.polygons, mask_rle=op.mask_rle, feather=op.feather), (H, W))
        if r is None:
            return None
        fx0, fy0, fa, _ = r
        fr = np.zeros((H, W), np.float32)
        fr[fy0:fy0 + fa.shape[0], fx0:fx0 + fa.shape[1]] = fa
    lim = fr
    if clip is not None and not op.own:
        lim = clip.astype(np.float32) if lim is None else lim * clip
    pts = op.pts.astype(np.float64)
    sig = np.maximum(op.size.astype(np.float64) * SPRAY_SIGMA, 0.5)
    reach = np.ceil(sig * SPRAY_REACH).astype(np.int64)
    cols = op.colors.astype(np.float32) / 255.0
    bb = None
    for i in range(len(pts)):
        cx, cy, rr = pts[i, 0], pts[i, 1], int(reach[i])
        x0, y0 = max(int(np.floor(cx)) - rr, 0), max(int(np.floor(cy)) - rr, 0)
        x1, y1 = min(int(np.floor(cx)) + rr + 1, W), min(int(np.floor(cy)) + rr + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        gx = np.exp(-((np.arange(x0, x1) + 0.5 - cx) ** 2) / (2 * sig[i] ** 2)).astype(np.float32)
        gy = np.exp(-((np.arange(y0, y1) + 0.5 - cy) ** 2) / (2 * sig[i] ** 2)).astype(np.float32)
        a = (gy[:, None] * gx[None, :]) * float(op.opacity[i])
        if lim is not None:
            a = a * lim[y0:y1, x0:x1]
        if _composite(layer, x0, y0, a, cols[i], False, alpha_lock, None):
            b = (x0, y0, x1, y1)
            bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
    return bb


def _apply_retouch(layer, op, clip, alpha_lock):
    """Retouch marks (see ops.Retouch): hard-edged, opaque, pixel-exact; later marks over earlier ones."""
    H, W = layer.shape[:2]
    xs, ys, idx = op.pixels()
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not ok.all():
        xs, ys, idx = xs[ok], ys[ok], idx[ok]
    if not len(xs):
        return None
    # last mark wins on a pixel covered twice
    flat = ys * W + xs
    rev = np.unique(flat[::-1], return_index=True)[1]
    keep = len(flat) - 1 - rev
    xs, ys, idx = xs[keep], ys[keep], idx[keep]
    col = op.colors[idx].astype(np.float32) / 255.0
    a = np.ones(len(xs), np.float32)
    if clip is not None and not op.own:
        a = a * clip[ys, xs]
    if alpha_lock:
        a = a * (layer[ys, xs, 3] > 0)
    sel = a > 0
    if not sel.any():
        return None
    xs, ys, a, col = xs[sel], ys[sel], a[sel], col[sel]
    old = layer[ys, xs]
    oa = old[:, 3:4]
    na = a[:, None] + oa * (1.0 - a[:, None])
    rgb = (col * a[:, None] + old[:, :3] * oa * (1.0 - a[:, None])) / np.maximum(na, 1e-6)
    if alpha_lock:
        na = oa
    layer[ys, xs, :3] = rgb
    layer[ys, xs, 3:4] = na
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


STREAK_STEP = 1.0        # px: the path of a streak is sampled at most this far apart
STREAK_CHUNK = 400_000   # samples per vectorised block (memory bound)


def streak_samples(op):
    """Samples along every stroke of a Streak op: (x, y, dx, dy, spacing h, width, rgb float [S,3] 0..255).
    Position, colour and width are interpolated linearly between stations; samples are spaced at most STREAK_STEP
    along each segment (the last station of a stroke is a sample of its own)."""
    p = op.pts.astype(np.float64)
    n = op.n.astype(np.int64)
    m = len(p)
    last = np.cumsum(n) - 1                     # index of each stroke's last station
    is_last = np.zeros(m, bool)
    is_last[last] = True
    seg = np.flatnonzero(~is_last)              # a segment starts at every station but a stroke's last
    a, b = seg, seg + 1
    d = p[b] - p[a]
    ln = np.sqrt((d ** 2).sum(1))
    k = np.maximum(1, np.ceil(ln / STREAK_STEP).astype(np.int64))
    h = ln / k
    rep = np.repeat(np.arange(len(seg)), k)
    j = np.arange(int(k.sum())) - np.repeat(np.concatenate([[0], np.cumsum(k)[:-1]]), k)
    t = j / k[rep]
    u = np.where(ln[:, None] > 1e-9, d / np.maximum(ln, 1e-9)[:, None], np.array([[1.0, 0.0]]))
    col = op.colors.astype(np.float64)
    wid = op.width.astype(np.float64)
    xs = p[a[rep], 0] + t * d[rep, 0]
    ys = p[a[rep], 1] + t * d[rep, 1]
    cs = col[a[rep]] * (1 - t)[:, None] + col[b[rep]] * t[:, None]
    ws = wid[a[rep]] * (1 - t) + wid[b[rep]] * t
    dx, dy, hs = u[rep, 0], u[rep, 1], h[rep]
    # the closing sample of each stroke at its last station, with its last segment's direction and spacing
    end_seg = np.searchsorted(seg, last - 1)
    xs = np.concatenate([xs, p[last, 0]])
    ys = np.concatenate([ys, p[last, 1]])
    cs = np.concatenate([cs, col[last]])
    ws = np.concatenate([ws, wid[last]])
    dx = np.concatenate([dx, u[end_seg, 0]])
    dy = np.concatenate([dy, u[end_seg, 1]])
    hs = np.concatenate([hs, h[end_seg]])
    return xs, ys, dx, dy, np.maximum(hs, 1e-3), ws, cs


def _apply_streak(layer, op, clip, alpha_lock):
    """Streak strokes (see ops.Streak): anti-aliased thin strokes, colour and width interpolated along each path;
    the op's strokes mix where they overlap (coverage-weighted colour), then the op is laid over the layer."""
    H, W = layer.shape[:2]
    xs, ys, dx, dy, hs, ws, cs = streak_samples(op)
    # reach of a sample: half the width plus the anti-aliased half pixel across, one spacing along
    R = int(np.ceil(np.hypot(float(ws.max()) / 2.0 + 0.5, float(hs.max())) - 1e-9))
    offs = np.arange(-R, R + 1)
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    oy, ox = oy.ravel(), ox.ravel()
    x0 = max(int(np.floor(xs.min())) - R, 0)
    y0 = max(int(np.floor(ys.min())) - R, 0)
    x1 = min(int(np.ceil(xs.max())) + R + 1, W)
    y1 = min(int(np.ceil(ys.max())) + R + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    bw, bh = x1 - x0, y1 - y0
    S = np.zeros(bw * bh, np.float64)
    SC = np.zeros((3, bw * bh), np.float64)
    for c0 in range(0, len(xs), STREAK_CHUNK // len(ox) + 1):
        sl = slice(c0, c0 + STREAK_CHUNK // len(ox) + 1)
        px, py = xs[sl].astype(np.float32), ys[sl].astype(np.float32)
        fx, fy = np.floor(px), np.floor(py)
        qx = (fx + 0.5 - px)[:, None] + ox[None].astype(np.float32)
        qy = (fy + 0.5 - py)[:, None] + oy[None].astype(np.float32)
        ddx, ddy = dx[sl].astype(np.float32)[:, None], dy[sl].astype(np.float32)[:, None]
        along = np.abs(qx * ddx + qy * ddy)
        along *= (1.0 / hs[sl].astype(np.float32))[:, None]
        np.subtract(1.0, along, out=along)
        perp = np.abs(qx * ddy - qy * ddx)
        np.subtract((ws[sl].astype(np.float32) / 2.0 + 0.5)[:, None], perp, out=perp)
        np.clip(perp, 0.0, 1.0, out=perp)
        np.maximum(along, 0.0, out=along)
        wt = perp * along
        si, oi = np.nonzero(wt > 0)
        ix = fx.astype(np.int64)[si] + ox[oi]
        iy = fy.astype(np.int64)[si] + oy[oi]
        ok = (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)
        if not ok.any():
            continue
        si, oi = si[ok], oi[ok]
        flat = (iy[ok] - y0) * bw + (ix[ok] - x0)
        w_ok = wt[si, oi].astype(np.float64)
        S += np.bincount(flat, weights=w_ok, minlength=bw * bh)
        cols = cs[c0 + si]
        for c in range(3):
            SC[c] += np.bincount(flat, weights=w_ok * cols[:, c], minlength=bw * bh)
    sel = S > 1e-6
    if not sel.any():
        return None
    a = np.zeros(bw * bh, np.float32)
    a[sel] = np.minimum(S[sel], 1.0)
    rgb = np.zeros((bw * bh, 3), np.float32)
    rgb[sel] = (SC[:, sel] / S[sel]).T / 255.0
    a = a.reshape(bh, bw)
    rgb = rgb.reshape(bh, bw, 3)
    if clip is not None and not op.own:
        a = a * clip[y0:y1, x0:x1]
    sub = layer[y0:y1, x0:x1]
    if alpha_lock:
        a = a * (sub[..., 3] > 0)
    if not (a > 0).any():
        return None
    oa = sub[..., 3:4]
    a3 = a[..., None]
    na = a3 + oa * (1.0 - a3)
    out = (rgb * a3 + sub[..., :3] * oa * (1.0 - a3)) / np.maximum(na, 1e-6)
    m = a > 0
    sub[..., :3][m] = out[m]
    if not alpha_lock:
        sub[..., 3:4][m] = na[m]
    return x0, y0, x1, y1


def apply_ops(layer: np.ndarray, ops, clip=None, alpha_lock: bool = False):
    """Paint ops onto ``layer`` in place, in order; returns the changed bbox (x0,y0,x1,y1) or None."""
    if layer.dtype != np.float32 or layer.ndim != 3 or layer.shape[2] != 4:
        raise ValueError("layer must be float32 [H,W,4]")
    if clip is not None:
        clip = np.asarray(clip)
        if clip.dtype == bool:
            clip = clip.astype(np.float32)
    bb = None
    for op in ops:
        if type(op).__name__ == "Spray":
            b = _apply_spray(layer, op, clip, alpha_lock)
            if b is not None:
                bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
            continue
        if type(op).__name__ in ("Retouch", "Streak"):
            b = (_apply_retouch if type(op).__name__ == "Retouch" else _apply_streak)(layer, op, clip, alpha_lock)
            if b is not None:
                bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
            continue
        if type(op).__name__ == "Blend":
            b = _apply_blend(layer, op, clip, alpha_lock)
            if b is not None:
                bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
            continue
        r = _render(op, layer.shape)
        if r is None:
            continue
        x0, y0, a, rgb = r
        if _composite(layer, x0, y0, a, rgb, op.erase, alpha_lock, None if getattr(op, "own", False) else clip):
            h, w = a.shape
            b = (x0, y0, x0 + w, y0 + h)
            bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
    return bb


def apply_ops_progressive(layer: np.ndarray, ops, n_frames: int, clip=None) -> Iterator[np.ndarray]:
    ops = list(ops)
    n_frames = max(1, int(n_frames))
    bounds = np.linspace(0, len(ops), n_frames + 1).round().astype(int)
    for i in range(n_frames):
        apply_ops(layer, ops[bounds[i]:bounds[i + 1]], clip=clip)
        yield layer.copy()


def op_bbox(op, shape):
    """Bounding box (x0,y0,x1,y1) an op can touch, clipped to the layer; None when outside."""
    H, W = shape[:2]
    if isinstance(op, Stroke):
        wmax = float(np.max(op.width))
        spec = PRESETS.get(op.brush, BrushSpec())
        pad = wmax / 2 + 3 * (spec.edge_px + spec.edge * wmax) + 3
        x0, y0 = op.pts.min(0) - pad
        x1, y1 = op.pts.max(0) + pad
    elif type(op).__name__ == "Retouch":
        xs, ys, _ = op.pixels()
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    elif type(op).__name__ == "Streak":
        pad = float(op.width.max()) / 2.0 + 2.0
        x0, y0 = op.pts.min(0) - pad
        x1, y1 = op.pts.max(0) + pad
    elif type(op).__name__ == "Spray":
        pad = float(op.size.max()) * SPRAY_SIGMA * SPRAY_REACH + 2
        x0, y0 = op.pts.min(0) - pad
        x1, y1 = op.pts.max(0) + pad
        if op.polygons is not None and len(op.polygons):
            allp = np.concatenate(op.polygons, 0)
            fx0, fy0 = allp.min(0) - 1.5 * op.feather - 2
            fx1, fy1 = allp.max(0) + 1.5 * op.feather + 2
            x0, y0, x1, y1 = max(x0, fx0), max(y0, fy0), min(x1, fx1), min(y1, fy1)
    else:
        pad = 1.5 * op.feather + 2
        if op.polygons is not None and len(op.polygons):
            allp = np.concatenate(op.polygons, 0)
            x0, y0 = allp.min(0) - pad
            x1, y1 = allp.max(0) + pad
        else:
            bx = op.mask_rle.get("box", [0, 0, op.mask_rle["size"][1], op.mask_rle["size"][0]])
            x0, y0, x1, y1 = bx[0] - pad, bx[1] - pad, bx[2] + pad, bx[3] + pad
    b = (max(int(np.floor(x0)), 0), max(int(np.floor(y0)), 0), min(int(np.ceil(x1)), W), min(int(np.ceil(y1)), H))
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    return b
