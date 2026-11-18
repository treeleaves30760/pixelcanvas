"""Picture measures by scale and kind, compared with professional pictures.

Band energies by feature size, flat / graded / textured area, edges, the spread of fine detail over 32 px tiles and
dark-line weight, measured on the picture at its delivered size and compared with the quartiles of 93 professional
pictures in five groups (detail_profiles.json).  The definitions are those of the judge tool spectrum.py:

- bands: L*/100 and a*/100, b*/100 are mirror-extended to 2H x 2W, FFT'd and split by one-octave raised-cosine
  windows in log2(frequency) into feature sizes 1-2, 2-4, 4-8, 8-16, 16-32, 32-64 and > 64 px (band 0..6);
  bandL_k = mean of the band image squared, bandC_k = mean of (a band^2 + b band^2).
- edges: Canny(L*/100, sigma 1, low 0.025, high 0.06); edge_density = edge px / all px.
- classes: flat = 5x5 RGB range <= 1/255; edge = not flat and within 1 px of an edge; graded = the rest with a 7x7
  plane-fit residual < 1 Lab unit; textured = everything else; flat0 = 5x5 range exactly 0.
- detail: fine = b0^2 + b1^2 + b2^2 of L; 32 px tile means; det_empty = share of tiles with 100*sqrt(e) < 0.5 L*,
  det_top10 = share of fine energy in the top 10 % tiles, det_gini = Gini of tile energies.
- lc_p50 = median std of L* in a sigma-8 window; grad_p90 = 90th percentile of the L* gradient (L*/px).
- lines: dark lines (darker than the 15 px closing by > 15 L*, L* < 70), skeleton, ink-weighted width;
  line_len_per_mp, line_w_p50, line_cv (width variation along a path), line_taper (share of path length whose
  p90/p10 width > 1.5).

area_measures(): det_empty over an area counts the 32 px tiles whose centre pixel lies inside the mask.
"""
from __future__ import annotations

import functools
import io
import json
import math
import pathlib
import re

import cv2
import numpy as np
from PIL import Image
from scipy import fft as sfft
from scipy import ndimage as ndi
from skimage import color as skcolor
from skimage import feature as skfeature
from skimage import measure as skmeasure
from skimage import morphology as skmorph

PROFILE_FILE = pathlib.Path(__file__).with_name("detail_profiles.json")
GROUPS = ("anime", "anime-cel", "anime-paint", "landscape", "promo")
RANKED_KEYS = ("det_empty", "flat_share", "bandL_0", "bandL_1", "bandL_2", "edge_density", "line_taper",
               "bandL_3", "bandL_4", "bandL_5", "lc_p50", "bandC_6")
GATE_KEYS = ("bandL_0", "bandL_1", "bandL_2", "flat_share", "det_empty", "line_taper")

NB = 7
TILE = 32
BAND_PX = ("1-2", "2-4", "4-8", "8-16", "16-32", "32-64", "> 64")
ANIME_WORDS = re.compile(r"\b(anime|manga|chibi|cel)\b")
FLAT_WORD = re.compile(r"\bflat\b")

# key -> (plain label, kind, unit); kind as in spectrum.DESC: ratio (log scale), share, abs
LABELS: dict[str, tuple[str, str, str]] = {}
for _k in range(NB):
    LABELS[f"bandL_{_k}"] = (f"{'fine detail' if _k == 0 else 'detail'} {BAND_PX[_k]} px", "ratio", "")
    LABELS[f"bandC_{_k}"] = (f"colour detail {BAND_PX[_k]} px", "ratio", "")
LABELS["bandC_6"] = ("broad colour saturation", "ratio", "")
LABELS.update({
    "flat_share": ("flat area", "share", ""),
    "flat0_share": ("exactly flat area", "share", ""),
    "graded_share": ("smoothly graded area", "share", ""),
    "textured_share": ("textured area", "share", ""),
    "edge_share": ("area at edges", "share", ""),
    "edge_density": ("edge pixels", "ratio", ""),
    "det_empty": ("tiles with no fine detail", "share", ""),
    "det_top10": ("fine detail held by the busiest 10 % of tiles", "share", ""),
    "det_gini": ("how unevenly fine detail is spread (Gini)", "abs", ""),
    "lc_p50": ("median local contrast", "abs", " L*"),
    "grad_p90": ("strong gradients (90th pct)", "abs", " L*/px"),
    "line_len_per_mp": ("dark-line length", "ratio", ""),
    "line_w_p50": ("median line width", "abs", " px"),
    "line_cv": ("line-weight variation along lines", "abs", ""),
    "line_taper": ("lines that taper", "share", ""),
})


# ----------------------------------------------------------------------------- profiles
@functools.lru_cache(maxsize=1)
def load_profiles() -> dict:
    with open(PROFILE_FILE, encoding="utf-8") as f:
        return json.load(f)


def profile_name(state: dict) -> str:
    style = state.get("style")
    req = str(state.get("request") or "").lower()
    anime = bool(ANIME_WORDS.search(req))
    if style == "poster":
        return "promo"
    if style in ("cel", "ink"):
        return "anime-cel" if FLAT_WORD.search(req) else "anime"
    if style in ("paint", "photo"):
        return "anime-paint" if anime else "landscape"
    return "anime" if anime else "landscape"


def delivered(rgb: np.ndarray, scale: int) -> np.ndarray:
    """Working-size picture -> delivered size (working size / scale), area-averaged."""
    if not scale or scale <= 1:
        return rgb
    h, w = rgb.shape[:2]
    size = (max(1, int(round(w / scale))), max(1, int(round(h / scale))))
    return cv2.resize(np.ascontiguousarray(rgb), size, interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- pieces (as spectrum.py)
def _jpeg(rgb: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).convert("RGB").save(buf, "JPEG", quality=int(q), subsampling=2)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


_WIN_CACHE: dict = {}


def _windows(h2, w2):
    key = (h2, w2)
    if key in _WIN_CACHE:
        return _WIN_CACHE[key]
    fy = sfft.fftfreq(h2)[:, None]
    fx = sfft.rfftfreq(w2)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    with np.errstate(divide="ignore"):
        t = np.log2(f)
    t[0, 0] = -np.inf
    ws = []
    for k in range(NB):
        c = -(k + 1.5)
        w = np.zeros_like(f)
        d = np.abs(t - c)
        m = d < 1
        w[m] = np.cos(np.pi / 2 * d[m])
        if k == 0:
            w[t >= c] = 1.0
        if k == NB - 1:
            w[(t <= c) & np.isfinite(t)] = 1.0
        w[0, 0] = 0.0
        ws.append(w.astype(np.float32))
    if len(_WIN_CACHE) > 4:
        _WIN_CACHE.clear()
    _WIN_CACHE[key] = ws
    return ws


def _band_split(ch):
    h, w = ch.shape
    ext = np.empty((2 * h, 2 * w), np.float32)
    ext[:h, :w] = ch
    ext[:h, w:] = ch[:, ::-1]
    ext[h:, :w] = ch[::-1, :]
    ext[h:, w:] = ch[::-1, ::-1]
    F = sfft.rfft2(ext - ext.mean(), workers=-1)
    out = []
    for wk in _windows(2 * h, 2 * w):
        b = sfft.irfft2(F * wk, s=(2 * h, 2 * w), workers=-1)[:h, :w]
        out.append(np.ascontiguousarray(b, dtype=np.float32))
    return out


def _tile_energy(img2, tile=TILE):
    h, w = img2.shape
    ht, wt = h // tile, w // tile
    if ht < 1 or wt < 1:
        return np.array([[float(np.mean(img2))]])
    return img2[: ht * tile, : wt * tile].reshape(ht, tile, wt, tile).mean(axis=(1, 3))


def _gini(x):
    x = np.sort(np.asarray(x, np.float64).ravel())
    n = x.size
    s = x.sum()
    if n == 0 or s <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * s)) - (n + 1) / n)


def _lines(L):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg = cv2.morphologyEx(L, cv2.MORPH_CLOSE, se)
    D = np.maximum(bg - L, 0)
    mask = (D > 15) & (L < 70)
    res = dict(line_len_per_mp=0.0, line_w_p50=np.nan, line_cv=np.nan, line_taper=np.nan)
    if mask.sum() < 20:
        return res
    try:
        mask = skmorph.remove_small_objects(mask, max_size=11)
    except TypeError:
        mask = skmorph.remove_small_objects(mask, 12)
    sk = skmorph.skeletonize(mask)
    n_sk = int(sk.sum())
    if n_sk < 20:
        return res
    Dp = cv2.dilate(D, np.ones((3, 3), np.uint8))
    dref = max(float(np.percentile(Dp[sk], 90)), 15.0)
    ink = np.clip(D / dref, 0, 1) * (D > 3)
    dist, (iy, ix) = ndi.distance_transform_edt(~sk, return_indices=True)
    near = (dist <= 6) & (ink > 0)
    sk_id = np.full(sk.shape, -1, np.int64)
    ys, xs = np.nonzero(sk)
    sk_id[ys, xs] = np.arange(ys.size)
    own = sk_id[iy[near], ix[near]]
    per = np.bincount(own, weights=ink[near], minlength=ys.size)
    INK = np.zeros(sk.shape, np.float32)
    INK[ys, xs] = per
    SK = sk.astype(np.float32)
    cnt = cv2.boxFilter(SK, -1, (9, 9), normalize=False)
    ssum = cv2.boxFilter(INK, -1, (9, 9), normalize=False)
    W = ssum / np.maximum(cnt * 1.15, 1e-6)
    ok = sk & (cnt >= 5) & (W <= 9)
    if ok.sum() < 20:
        return res
    res["line_len_per_mp"] = float(n_sk / (L.size / 1e6))
    res["line_w_p50"] = float(np.percentile(W[ok], 50))
    nb = cv2.filter2D(SK, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - 1
    paths = sk & (nb <= 2)
    lab = skmeasure.label(paths, connectivity=2)
    cvs, ws_, tp = [], [], []
    for r in skmeasure.regionprops(lab):
        if r.area < 20:
            continue
        cc = r.coords
        v = W[cc[:, 0], cc[:, 1]]
        good = ok[cc[:, 0], cc[:, 1]]
        if good.sum() < 20:
            continue
        v = v[good]
        cvs.append(float(v.std() / max(v.mean(), 1e-6)))
        ws_.append(len(v))
        p10, p90 = np.percentile(v, [10, 90])
        tp.append(float(p90 / max(p10, 1e-6) > 1.5))
    if cvs:
        cvs, ws_, tp = np.array(cvs), np.array(ws_, float), np.array(tp)
        o = np.argsort(cvs)
        cw = np.cumsum(ws_[o]) / ws_.sum()
        res["line_cv"] = float(cvs[o][np.searchsorted(cw, 0.5)])
        res["line_taper"] = float((tp * ws_).sum() / ws_.sum())
    return res


def _classes(rgb, lab, edges):
    h, w = rgb.shape[:2]
    k5 = np.ones((5, 5), np.uint8)
    rng = np.zeros((h, w), np.uint8)
    for c in range(3):
        ch = np.ascontiguousarray(rgb[:, :, c])
        rng = np.maximum(rng, cv2.dilate(ch, k5) - cv2.erode(ch, k5))
    flat = rng <= 1
    flat0 = rng == 0
    xs = np.arange(-3, 4, dtype=np.float32)
    kx = np.tile(xs[None, :], (7, 1)) / (7 * 28.0)
    ky = kx.T.copy()
    res = np.zeros((h, w), np.float32)
    for c in range(3):
        ch = np.ascontiguousarray(lab[:, :, c])
        m1 = cv2.boxFilter(ch, -1, (7, 7))
        m2 = cv2.boxFilter(ch * ch, -1, (7, 7))
        var = np.maximum(m2 - m1 * m1, 0)
        sx = cv2.filter2D(ch, -1, kx, borderType=cv2.BORDER_REFLECT)
        sy = cv2.filter2D(ch, -1, ky, borderType=cv2.BORDER_REFLECT)
        res += np.maximum(var - 4.0 * (sx * sx + sy * sy), 0)
    resid = np.sqrt(res)
    near_edge = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    cls_edge = (~flat) & near_edge
    rest = (~flat) & (~cls_edge)
    return dict(flat=flat, flat0=flat0, edge=cls_edge, graded=rest & (resid < 1.0), textured=rest & (resid >= 1.0))


# ----------------------------------------------------------------------------- measure
def measure(rgb: np.ndarray, *, jpeg: int | None = 80, lines: bool = True, keep: bool = False) -> dict:
    """Measures of a uint8 RGB picture at delivered size (see the module docstring for the definitions)."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb[:, :, :3])
    if jpeg:
        rgb = _jpeg(rgb, jpeg)
    h, w = rgb.shape[:2]
    lab = skcolor.rgb2lab(rgb.astype(np.float64) / 255.0).astype(np.float32)
    L = np.ascontiguousarray(lab[:, :, 0])
    m: dict = dict(width=w, height=h)
    bandsL = _band_split(L / 100.0)
    bandsA = _band_split(lab[:, :, 1] / 100.0)
    bandsB = _band_split(lab[:, :, 2] / 100.0)
    for k in range(NB):
        m[f"bandL_{k}"] = float(np.mean(bandsL[k].astype(np.float64) ** 2))
        m[f"bandC_{k}"] = float(np.mean(bandsA[k].astype(np.float64) ** 2) + np.mean(bandsB[k].astype(np.float64) ** 2))
    edges = skfeature.canny(L / 100.0, sigma=1.0, low_threshold=0.025, high_threshold=0.06)
    m["edge_density"] = float(edges.mean())
    if lines:
        m.update(_lines(L))
    cl = _classes(rgb, lab, edges)
    for nm in ("flat", "flat0", "edge", "graded", "textured"):
        m[f"{nm}_share"] = float(cl[nm].mean())
    mu = cv2.GaussianBlur(L, (0, 0), 8)
    mu2 = cv2.GaussianBlur(L * L, (0, 0), 8)
    m["lc_p50"] = float(np.percentile(np.sqrt(np.maximum(mu2 - mu * mu, 0)), 50))
    Lg = cv2.GaussianBlur(L, (0, 0), 0.7)
    g = np.hypot(cv2.Sobel(Lg, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(Lg, cv2.CV_32F, 0, 1, ksize=3) / 8.0)
    m["grad_p90"] = float(np.percentile(g, 90))
    fine = bandsL[0].astype(np.float64) ** 2 + bandsL[1].astype(np.float64) ** 2 + bandsL[2].astype(np.float64) ** 2
    te = _tile_energy(fine)
    m["det_gini"] = _gini(te)
    tf = te.ravel()
    nt = max(1, int(round(0.1 * tf.size)))
    m["det_top10"] = float(np.sort(tf)[::-1][:nt].sum() / max(tf.sum(), 1e-12))
    m["det_empty"] = float((100 * np.sqrt(tf) < 0.5).mean())
    if keep:
        m["_bandsL"] = bandsL
        m["_bandsC"] = [a * a + b * b for a, b in zip(bandsA, bandsB)]
        m["_flat"] = cl["flat"]
        m["_edges"] = edges
        m["_fine_tiles"] = te
    return m


def area_measures(m: dict, masks: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per-area measures from measure(..., keep=True).  Masks: bool, delivered size; masks under 64 px are skipped.

    det_empty counts the 32 px tiles whose centre pixel lies in the mask."""
    if "_bandsL" not in m:
        raise ValueError("area_measures needs measure(..., keep=True)")
    bandsL, bandsC = m["_bandsL"], m["_bandsC"]
    h, w = bandsL[0].shape
    te = m["_fine_tiles"]
    ht, wt = h // TILE, w // TILE
    if ht < 1 or wt < 1:
        cy, cx = np.array([[h // 2]]), np.array([[w // 2]])
    else:
        cy, cx = np.meshgrid(np.arange(ht) * TILE + TILE // 2, np.arange(wt) * TILE + TILE // 2, indexing="ij")
    out: dict[str, dict] = {}
    for name, mask in masks.items():
        mk = np.asarray(mask, bool)
        if mk.shape != (h, w):
            raise ValueError(f"mask {name!r} is {mk.shape[1]}x{mk.shape[0]}, picture is {w}x{h}")
        n = int(mk.sum())
        if n < 64:
            continue
        r: dict = {"n_px": n}
        fine = 0.0
        for k in range(NB):
            e = float(np.mean(bandsL[k][mk].astype(np.float64) ** 2))
            r[f"bandL_{k}"] = e
            r[f"bandC_{k}"] = float(np.mean(bandsC[k][mk].astype(np.float64)))
            if k < 3:
                fine += e
        r["fine_rms"] = 100 * math.sqrt(fine)
        r["flat_share"] = float(m["_flat"][mk].mean())
        r["edge_density"] = float(m["_edges"][mk].mean())
        inside = mk[cy, cx]
        r["det_empty"] = float((100 * np.sqrt(te[inside]) < 0.5).mean()) if inside.any() else float("nan")
        out[name] = r
    return out


# ----------------------------------------------------------------------------- comparison
def _fmt(v, kind, unit):
    if kind == "share":
        return f"{100 * v:.0f} %"
    return f"{v:.3g}{unit}"


def _entry(key, v, st, group):
    label, kind, unit = LABELS.get(key, (key, "abs", ""))
    med, q1, q3 = st["median"], st["q1"], st["q3"]
    if kind == "ratio":
        eps = max(abs(med), 1e-9) * 1e-3
        sc = max(math.log(max(q3, eps) / max(q1, eps)) / 1.349, 0.25)
        z = (math.log(max(v, eps)) - math.log(max(med, eps))) / sc
        rel = v / max(med, eps)
        text = f"{label}: {rel:.2f}x of professional {group} (needs {q1 / max(med, eps):.2f}-{q3 / max(med, eps):.2f}x)"
    else:
        sc = max((q3 - q1) / 1.349, 0.03 if kind == "share" else 0.1 * abs(med) + 1e-3)
        z = (v - med) / sc
        if kind == "share":
            text = f"{label} {100 * v:.0f} % (professional {100 * q1:.0f}-{100 * q3:.0f} %)"
        else:
            text = f"{label} {_fmt(v, kind, unit)} (professional {_fmt(q1, kind, '')}-{_fmt(q3, kind, unit)})"
    ratio = v / med if med else float("inf") if v else 1.0
    return dict(key=key, value=float(v), median=med, q1=q1, q3=q3, z=float(np.clip(z, -12, 12)), ratio=ratio,
                low=bool(v < med), text=text)


def _entries(m, group, keys):
    prof = load_profiles()["groups"].get(group)
    if prof is None:
        raise ValueError(f"unknown group {group!r}; have {', '.join(GROUPS)}")
    out = []
    for key in keys:
        st = prof["measures"].get(key)
        v = m.get(key)
        if st is None or v is None or isinstance(v, (list, np.ndarray)):
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        out.append(_entry(key, v, st, group))
    out.sort(key=lambda d: -abs(d["z"]))
    return out


def gaps(m: dict, group: str, keys=None, n: int = 3) -> list[dict]:
    """The n measures furthest from the group (robust z as spectrum.deviations), largest |z| first."""
    return _entries(m, group, RANKED_KEYS if keys is None else keys)[:n]


def outside(m: dict, group: str, keys=GATE_KEYS) -> list[dict]:
    """The measures in keys that lie outside the group's [q1, q3], largest |z| first."""
    return [d for d in _entries(m, group, keys) if d["value"] < d["q1"] or d["value"] > d["q3"]]
