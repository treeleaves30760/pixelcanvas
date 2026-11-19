"""Shared pieces of drawing without a reference: the draw state, geometry, light, colour shifts, owned layers.

Everything the no-reference stages decide lives in ``state["draw"]`` (so ``state.json`` snapshots make it undoable):

```
draw = {
  "subjects": {name: {"box": [x0,y0,x1,y1]}},                      # 0-1000 units
  "parts":    {name: {"of": subject, "box": [...], "focus": 1-3, "swatch": name}},
  "light":    {"deg": 135, "z": 0.6},                              # where the light comes from (0 right, 90 top)
  "palette":  {"swatches": {name: "#rrggbb"}, "candidates": [...], "picked": n},
  "sketch":   [element records],  "lines": [line records],  "seq": {"s": n, "L": n, "n": n},
  "fills":    [{"pt": [x,y] px, "swatch": name, "part": name}],    # flat assignments, by a point inside the area
  "bands":    {name: {...}},                                       # paint style: land / sky bands
  "owned":    {"sketch": {...}, "line": {...}},                    # bookkeeping of record-owned layers
}
```

The sketch and line layers are *owned* by their records while drawing without a reference: a record can be moved,
redrawn or deleted by id, and the layer is the render of its records over a base picture. Marks other commands put
on an owned layer (``pixelcanvas stroke --layer line``) are folded into that base the next time a record changes, so they
are kept; an ``pixelcanvas erase`` over records is kept too (it is applied to the records drawn before it).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from . import coords
from .coords import UsageError

OWNED = ("sketch", "line")


# ============================================================================================ state

def dstate(p) -> dict:
    d = p.state.setdefault("draw", {})
    d.setdefault("subjects", {})
    d.setdefault("parts", {})
    d.setdefault("light", {"deg": 135.0, "z": 0.6, "set": False})
    d.setdefault("palette", {"swatches": {}, "candidates": [], "picked": None})
    d.setdefault("sketch", [])
    d.setdefault("lines", [])
    d.setdefault("seq", {"s": 0, "L": 0, "n": 0})
    d.setdefault("fills", [])
    d.setdefault("bands", {})
    d.setdefault("owned", {})
    return d


def next_id(d: dict, kind: str) -> str:
    d["seq"][kind] = int(d["seq"].get(kind, 0)) + 1
    return f"{kind}{d['seq'][kind]}"


def next_order(d: dict) -> int:
    d["seq"]["n"] = int(d["seq"].get("n", 0)) + 1
    return d["seq"]["n"]


def find_record(records: List[dict], rid: str, what: str, example: str) -> dict:
    k = str(rid).strip().lower()
    for r in records:
        if r["id"].lower() == k:
            return r
    ids = ", ".join(r["id"] for r in records) or "none yet"
    raise UsageError(f"there is no {what} {rid!r}; {what}s: {ids}", example)


# ============================================================================================ units

def side(p) -> int:
    return max(p.W, p.H)


def _norm_cmd(cmd) -> str:
    s = " ".join(str(cmd or "").split())
    return s[6:] if s.startswith("pixelcanvas ") else s


def stable_seed(p, cmdline, *salt) -> int:
    """A brush seed from the command itself: its words, how many times the same command is already in the picture
    (undone steps do not count) and ``salt`` (the target area, a mark index). Adding or removing an unrelated command
    leaves every other command's marks as they were; the same command run again lays different marks."""
    import json
    cmd = _norm_cmd(cmdline)
    n = 0
    try:
        applied = set(int(s) for s in p._stack().get("applied", []))
        jf = p.root / "journal.jsonl"
        for ln in jf.read_text(encoding="utf-8").splitlines() if jf.exists() else []:
            if f'"cmd"' not in ln:
                continue
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if e.get("step") in applied and _norm_cmd(e.get("cmd")) == cmd:
                n += 1
    except Exception:  # noqa: BLE001 - a seed never breaks a command
        n = 0
    h = hashlib.sha256(json.dumps([cmd, n] + [str(s) for s in salt]).encode("utf-8")).digest()
    return 1 + int.from_bytes(h[:4], "little") % 1_000_003     # as small as the step numbers it replaces


def work_scale(p) -> int:
    """How many working pixels make one delivered pixel (3 on a free-mode project: marks are laid on a canvas 3x the
    exported picture, so the finest octave survives the downsample; 1 elsewhere)."""
    try:
        return max(1, int((p.state if hasattr(p, "state") else p).get("work_scale", 1) or 1))
    except (TypeError, ValueError):
        return 1


def dpx(p, delivered_px: float) -> float:
    """A size given in delivered pixels, in working pixels."""
    return float(delivered_px) * work_scale(p)


_COMP: Dict[tuple, np.ndarray] = {}


def composite_u8(p, mode: str = "export") -> np.ndarray:
    """``p.composite_u8(mode)``, computed once per state of the layers (checks ask for it many times per command;
    on a 3x canvas one composite is ~1 s). Do not modify the returned array."""
    try:
        lays = p.layers_dict()
        if mode in ("export", "score") and not p.sketch_shown() and \
                not any(e["name"] == "guide" for e in p.state["layers"]):
            mode = "view"                   # the same picture: the sketch layer is hidden in both
        key = (id(p), mode, p.sketch_shown(), tuple((n, id(a)) for n, a in lays.items()),
               tuple((e["name"], e.get("visible", True), e.get("opacity", 1.0), e.get("mode")) for e in p.state["layers"]))
    except Exception:  # noqa: BLE001
        return p.composite_u8(mode)
    hit = _COMP.get(key)
    if hit is not None:
        return hit[0]
    if mode == "fills":
        # the picture without the surface and the finishing effects: colour checks (holes, slivers) look at the
        # fills, not the texture or the grade over them
        from . import compose
        img = compose.to_uint8(compose.composite(p.state["layers"], lays, p.H, p.W, background=(1, 1, 1),
                                                 skip=["sketch", "surface", "fx"]))
    else:
        img = p.composite_u8(mode)
    if len(_COMP) > 6:
        _COMP.clear()
    _COMP[key] = (img, list(lays.values()))     # keep the arrays alive so their ids are not reused
    return img


def u2px(p, pts) -> np.ndarray:
    a = np.asarray(pts, np.float64).reshape(-1, 2)
    out = a.copy()
    out[:, 0] = a[:, 0] * p.W / 1000.0
    out[:, 1] = a[:, 1] * p.H / 1000.0
    return out


def px2u(p, pts) -> np.ndarray:
    a = np.asarray(pts, np.float64).reshape(-1, 2)
    out = a.copy()
    out[:, 0] = a[:, 0] * 1000.0 / p.W
    out[:, 1] = a[:, 1] * 1000.0 / p.H
    return out


def box_px(p, box_u) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = box_u
    return x0 * p.W / 1000.0, y0 * p.H / 1000.0, x1 * p.W / 1000.0, y1 * p.H / 1000.0


def fmt_pt(p, pt_px) -> str:
    return f"{int(round(pt_px[0] * 1000.0 / p.W))},{int(round(pt_px[1] * 1000.0 / p.H))}"


def fmt_box_px(p, b) -> str:
    return coords.fmt_box(coords.px_box_to_units(b, p.W, p.H))


def rnd(a, nd=1) -> list:
    return [[round(float(x), nd), round(float(y), nd)] for x, y in np.asarray(a, np.float64).reshape(-1, 2)]


# ============================================================================================ geometry

def resample(pts, spacing: float) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(p) < 2:
        return p.copy()
    seg = np.hypot(*np.diff(p, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] < 1e-9:
        return p[:1].copy()
    n = max(2, int(math.ceil(t[-1] / spacing)) + 1)
    tt = np.linspace(0.0, t[-1], n)
    return np.stack([np.interp(tt, t, p[:, 0]), np.interp(tt, t, p[:, 1])], 1)


def path_length(pts) -> float:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def corner_spline(P, closed: bool = False, spacing: float = 1.5, sharp_deg: float = 65.0) -> np.ndarray:
    """A smooth curve through the points that stays sharp where the polyline turns more than ``sharp_deg``
    (a bell rim, a collar point, a pine tier): smooth runs between the corners."""
    P = np.asarray(P, np.float64).reshape(-1, 2)
    n = len(P)
    if n < 3:
        return resample(P, spacing)
    v1 = P - np.roll(P, 1, 0)
    v2 = np.roll(P, -1, 0) - P
    cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    sharp = cosang < math.cos(math.radians(sharp_deg))
    if not closed:
        sharp[0] = sharp[-1] = False
    idx = [i for i in range(n) if sharp[i]]
    if not idx:
        return catmull_rom(P, closed=closed, spacing=spacing)
    if closed:
        Q = np.roll(P, -idx[0], 0)
        Q = np.vstack([Q, Q[:1]])
        cuts = [i - idx[0] for i in idx] + [n]
    else:
        Q = P
        cuts = [0] + idx + [n - 1]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        run = Q[a:b + 1]
        seg = catmull_rom(run, spacing=spacing) if len(run) >= 3 else resample(run, spacing)
        out.append(seg if not out else seg[1:])
    res = np.vstack(out)
    return res[:-1] if closed else res


def catmull_rom(P, closed: bool = False, spacing: float = 1.5) -> np.ndarray:
    """Centripetal Catmull-Rom spline through every point (the stabiliser: a few points -> one smooth curve)."""
    P = np.asarray(P, np.float64).reshape(-1, 2)
    keep = [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - P[keep[-1]])) > 1e-6:
            keep.append(i)
    P = P[keep]
    if closed and len(P) > 2 and np.hypot(*(P[0] - P[-1])) < 1e-6:
        P = P[:-1]
    if len(P) < 3:
        if closed and len(P) >= 2:
            P = np.vstack([P, P[:1]])
        return resample(P, spacing)
    if closed:
        ext = np.vstack([P[-1], P, P[0], P[1]])
    else:
        ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]

        def tj(ti, a, b):
            return ti + max(1e-6, float(np.hypot(*(b - a)))) ** 0.5
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        n = max(2, int(math.ceil(np.hypot(*(p2 - p1)) / spacing)))
        t = np.linspace(t1, t2, n, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
        out.append(c)
    out.append((P[0] if closed else P[-1])[None, :])
    return np.vstack(out)


def smooth_poly(pts, sigma: float = 2.0, closed: bool = False) -> np.ndarray:
    p = np.asarray(pts, np.float64)
    if len(p) < 5 or sigma <= 0:
        return p
    from scipy.ndimage import gaussian_filter1d
    mode = "wrap" if closed else "nearest"
    q = np.stack([gaussian_filter1d(p[:, 0], sigma, mode=mode), gaussian_filter1d(p[:, 1], sigma, mode=mode)], 1)
    if not closed:
        q[0], q[-1] = p[0], p[-1]
    return q


def taper_widths(pts, width: float, taper: str = "both", tmin: float = 0.22) -> np.ndarray:
    """Pressure profile: thin at the tapered ends, full in the middle; never below 1 px (mark rule)."""
    p = np.asarray(pts, np.float64)
    n = len(p)
    if n < 2:
        return np.full(max(n, 1), max(1.0, width), np.float32)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    L = s[-1]
    f = np.ones(n)
    ln = min(0.3 * L, 7.0 * width + 4.0)
    if ln > 0:
        def ss(x):
            x = np.clip(x, 0, 1)
            return x * x * (3 - 2 * x)
        if taper in ("both", "start"):
            f *= tmin + (1 - tmin) * ss(s / ln)
        if taper in ("both", "end"):
            f *= tmin + (1 - tmin) * ss((L - s) / ln)
    return np.maximum(1.0, width * f).astype(np.float32)


def ellipse_pts(cx, cy, rx, ry, rot_deg: float = 0.0, n: int = 72, a0: float = 0.0, a1: float = 2 * math.pi):
    t = np.linspace(a0, a1, n)
    x, y = rx * np.cos(t), ry * np.sin(t)
    r = math.radians(rot_deg)
    return np.stack([cx + x * math.cos(r) - y * math.sin(r), cy + x * math.sin(r) + y * math.cos(r)], 1)


def nearest_on_polyline(poly: np.ndarray, pt) -> Tuple[float, int, np.ndarray]:
    """(distance, index of the nearest sample, nearest point) for a densely sampled polyline."""
    d = np.hypot(poly[:, 0] - pt[0], poly[:, 1] - pt[1])
    i = int(np.argmin(d))
    return float(d[i]), i, poly[i]


def rdp(pts, eps: float) -> np.ndarray:
    p = np.asarray(pts, np.float64)
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    d = b - a
    L = math.hypot(*d)
    if L < 1e-9:
        dist = np.hypot(*(p - a).T)
    else:
        dist = np.abs(d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])) / L
    i = int(np.argmax(dist))
    if dist[i] > eps:
        left = rdp(p[: i + 1], eps)
        right = rdp(p[i:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


# ============================================================================================ light

LIGHT_WORDS = {
    "right": 0, "top-right": 45, "upper-right": 45, "top": 90, "above": 90, "up": 90, "top-left": 135,
    "upper-left": 135, "left": 180, "bottom-left": 225, "lower-left": 225, "bottom": 270, "below": 270,
    "bottom-right": 315, "lower-right": 315,
}


def parse_light(text: str) -> Tuple[float, float, str]:
    """Words (top-left, front, back) or degrees (0 = from the right, 90 = from the top). Returns (deg, z, words)."""
    s = str(text).strip().lower().replace("_", "-").replace(" ", "-")
    z = 0.6
    if s in ("front", "frontal", "camera"):
        return 90.0, 2.5, "front (from the viewer, a little above)"
    if s in ("back", "behind", "backlight", "rim"):
        return 90.0, -0.4, "behind the subject (rim light)"
    s = s.replace("upper", "top").replace("lower", "bottom")
    if s in LIGHT_WORDS:
        return float(LIGHT_WORDS[s]), z, f"from the {s}"
    nums = [float(v) for v, _, _ in coords._numbers(s)]
    if len(nums) == 1:
        deg = nums[0] % 360.0
        return deg, z, f"from the {light_words(deg)}"
    raise UsageError(f"unknown light {text!r}: use top-left, top, top-right, left, right, front, back or degrees "
                     "(0 = from the right, 90 = from the top)", "pixelcanvas plan light top-left")


def light_words(deg: float) -> str:
    names = ["right", "top-right", "top", "top-left", "left", "bottom-left", "bottom", "bottom-right"]
    return names[int(((deg % 360) + 22.5) // 45) % 8]


def light_vec(d: dict) -> np.ndarray:
    """Unit vector toward the light in screen space (x right, y down, z toward the viewer)."""
    lt = d.get("light", {})
    a = math.radians(float(lt.get("deg", 135.0)))
    z = float(lt.get("z", 0.6))
    v = np.array([math.cos(a), -math.sin(a), z], np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


# ============================================================================================ colour

def to_hex(rgb) -> str:
    return coords.hex_colour(rgb)


def from_hex(h) -> Tuple[int, int, int]:
    return coords.parse_colour(h)


# OKLCh (Bjorn Ottosson's OKLab): hue stays put when a colour is lightened or darkened (CIE Lab turns blue purple).
# L is scaled to 0-100 and C by 400 so the numbers read like CIE LCh (red ~ 100, a grey < 3).
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i = np.linalg.inv(_M1)
_M2i = np.linalg.inv(_M2)
C_SCALE = 400.0


def _lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


def rgb_to_lch(rgb) -> Tuple[float, float, float]:
    lms = np.cbrt(_M1 @ _lin(rgb[:3]))
    L, A, B = _M2 @ lms
    return float(L * 100), float(math.hypot(A, B) * C_SCALE), float(math.degrees(math.atan2(B, A)) % 360.0)


def lch_to_rgb(L, C, h) -> Tuple[int, int, int]:
    L = float(np.clip(L, 0, 100)) / 100
    C = max(0.0, float(C)) / C_SCALE
    for _ in range(24):            # reduce chroma until the colour fits in sRGB (keeps lightness and hue)
        A, B = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
        lin = _M1i @ ((_M2i @ np.array([L, A, B])) ** 3)
        if lin.min() >= -1e-4 and lin.max() <= 1 + 1e-4:
            break
        C *= 0.9
    return tuple(int(round(v)) for v in _gam(lin))  # type: ignore


def _toward(h, target, k):
    dh = ((target - h + 540) % 360) - 180
    return (h + k * dh) % 360


def shadow_of(rgb, depth: float = 1.0) -> Tuple[int, int, int]:
    """A painter's shadow colour: darker and a little more saturated; warm colours turn toward red (skin shadows
    go rosy, not grey), cool colours and greys toward blue-violet."""
    L, C, h = rgb_to_lch(rgb)
    dl = (9.0 + 0.1 * L) * depth
    if C < 8:
        C2, h2 = C + 8 * depth, 285.0 if C < 3 else _toward(h, 285, 0.5)
    elif 0 <= h < 120 or h > 345:
        C2, h2 = C * (1 + 0.25 * depth) + 3 * depth, _toward(h, 15, 0.3 * depth)
    else:
        C2, h2 = C * (1 + 0.1 * depth), _toward(h, 285, 0.14 * depth)
    return lch_to_rgb(max(4.0, L - dl), C2, h2)


def light_of(rgb, amount: float = 1.0) -> Tuple[int, int, int]:
    """A lit colour: lighter, hue turned a little toward warm yellow, slightly less saturated."""
    L, C, h = rgb_to_lch(rgb)
    return lch_to_rgb(min(99.0, L + (5 + 0.14 * (100 - L)) * amount), C * (1 - 0.08 * amount),
                      _toward(h, 100, 0.1 * amount) if C > 5 else h)


def lum(rgb) -> float:
    return rgb_to_lch(rgb)[0]


# ============================================================================================ masks

def disk(r: float) -> np.ndarray:
    k = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))


def dilate(m, r: float) -> np.ndarray:
    if r <= 0:
        return np.asarray(m, bool)
    return cv2.dilate(np.asarray(m, np.uint8), disk(r)) > 0


def erode(m, r: float) -> np.ndarray:
    if r <= 0:
        return np.asarray(m, bool)
    return cv2.erode(np.asarray(m, np.uint8), disk(r)) > 0


def clean_mask(m, min_px: int) -> np.ndarray:
    """Drop connected specks smaller than ``min_px``."""
    m = np.asarray(m, bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, min_px)
    return keep[lab]


def smooth_mask(m, sigma: float) -> np.ndarray:
    if sigma <= 0.3:
        return np.asarray(m, bool)
    return cv2.GaussianBlur(np.asarray(m, np.float32), (0, 0), sigma) > 0.5


def polyline_mask(H, W, pts_px, width: float) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    q = np.round(np.asarray(pts_px) * 4).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(m, [q], False, 1, max(1, int(round(width))), cv2.LINE_8, 2)
    return m > 0


def poly_mask(H, W, pts_px) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts_px) * 4).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, 2)
    return m > 0


# ============================================================================================ owned layers

def _sig(arr: Optional[np.ndarray]) -> str:
    if arr is None:
        return "empty"
    a8 = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if not (a8[..., 3] > 0).any():
        return "empty"
    return hashlib.sha1(a8.tobytes()).hexdigest()


def owned_files(layer: str) -> List[str]:
    return [f"layers/{layer}.png", f"draw/{layer}_base.png", f"draw/{layer}_erase.png"]


def _load_rgba(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0


def _save_rgba(path: Path, arr: Optional[np.ndarray]) -> None:
    from .project import save_png
    if arr is None:
        if path.exists():
            path.unlink()
        return
    save_png(path, (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8))


def render_records(p, layer: str, records: List[dict], render_fn, fold_n: Optional[int] = None,
                   erase: Optional[np.ndarray] = None) -> np.ndarray:
    """base, then the records in order; the erase mask (from a fold) applies after records drawn before it."""
    from . import brush
    base = _load_rgba(p.root / "draw" / f"{layer}_base.png")
    arr = base.copy() if base is not None else np.zeros((p.H, p.W, 4), np.float32)
    recs = sorted(records, key=lambda r: r.get("n", 0))
    applied_erase = erase is None
    for r in recs:
        if not applied_erase and r.get("n", 0) > (fold_n or 0):
            arr[..., 3] *= (1.0 - erase)
            applied_erase = True
        ops = render_fn(p, r)
        if ops:
            brush.apply_ops(arr, ops)
    if not applied_erase:
        arr[..., 3] *= (1.0 - erase)
    return arr


def owned_fold(p, layer: str, records: List[dict], render_fn) -> Optional[str]:
    """Before records change: if another command painted on the owned layer since we last wrote it, keep those
    marks in the base (and erasures in the erase mask). Returns a note or None."""
    d = dstate(p)
    info = d["owned"].setdefault(layer, {"sig": "empty", "fold_n": 0})
    cur = p.layer(layer, create=False)
    if _sig(cur) == info.get("sig", "empty"):
        return None
    erase_old = _load_rgba(p.root / "draw" / f"{layer}_erase.png")
    erase_old = None if erase_old is None else erase_old[..., 3]
    last = render_records(p, layer, records, render_fn, info.get("fold_n"), erase_old)
    C = cur if cur is not None else np.zeros_like(last)
    diff = np.abs(C - last).max(axis=2) > (2.5 / 255.0)
    if not diff.any():
        info["sig"] = _sig(cur)
        return None
    erased = diff & (C[..., 3] < last[..., 3] - 0.02)
    added = diff & ~erased
    base = _load_rgba(p.root / "draw" / f"{layer}_base.png")
    if base is None:
        base = np.zeros_like(C)
    base[added] = C[added]
    base[erased] = C[erased] if erased.any() else base[erased]
    (p.root / "draw").mkdir(exist_ok=True)
    _save_rgba(p.root / "draw" / f"{layer}_base.png", base)
    note = f"marks drawn on the {layer} layer by other commands are kept"
    if erased.any():
        e = np.zeros(C.shape[:2], np.float32) if erase_old is None else erase_old.copy()
        frac = np.clip((last[..., 3] - C[..., 3]) / np.maximum(last[..., 3], 1e-3), 0, 1)
        e = np.maximum(e, np.where(erased, frac, 0.0))
        arr = np.zeros(C.shape, np.float32)
        arr[..., 3] = e
        _save_rgba(p.root / "draw" / f"{layer}_erase.png", arr)
        info["fold_n"] = int(d["seq"].get("n", 0))
        note += "; the erased parts stay erased"
    return note


def owned_write(p, layer: str, records: List[dict], render_fn) -> np.ndarray:
    d = dstate(p)
    info = d["owned"].setdefault(layer, {"sig": "empty", "fold_n": 0})
    er = _load_rgba(p.root / "draw" / f"{layer}_erase.png")
    arr = render_records(p, layer, records, render_fn, info.get("fold_n"), None if er is None else er[..., 3])
    p._write_layer_file(layer, arr)
    info["sig"] = _sig(p.layer(layer, create=False))
    return arr


def tube_polygon(axis_px, widths_px) -> np.ndarray:
    """Closed outline of a tube along ``axis_px`` whose width goes through ``widths_px`` from start to end
    (linearly between the given values, evenly spaced along the length); round caps at both ends."""
    A = resample(np.asarray(axis_px, float), 1.5)
    if len(A) < 2:
        raise ValueError("the axis needs two points")
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(A, axis=0).T))])
    t = s / max(s[-1], 1e-9)
    ws = np.asarray(widths_px, float).reshape(-1)
    if len(ws) == 1:
        ws = np.array([ws[0], ws[0]])
    if len(ws) >= 3:
        from scipy.interpolate import PchipInterpolator      # smooth, no overshoot: no kinks between the widths
        w = PchipInterpolator(np.linspace(0, 1, len(ws)), ws)(t) / 2.0
    else:
        w = np.interp(t, np.linspace(0, 1, len(ws)), ws) / 2.0
    tg = np.gradient(A, axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
    n = np.stack([-tg[:, 1], tg[:, 0]], 1)
    left = A + n * w[:, None]
    right = A - n * w[:, None]

    def cap(c, r, d):
        # from the +normal side round the tip to the -normal side
        ang = np.linspace(0, math.pi, max(12, int(r / 2)))[1:-1]
        base = math.atan2(d[1], d[0])
        th = base + math.pi / 2 - ang
        return np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)], 1)
    wmax = float(w.max())
    # an end that flares to the widest point stays open and flat (a bell, a trumpet mouth); others are round
    def lip(c, r, d):
        # a flared mouth (a bell, a trumpet): a shallow rounded rim instead of a straight cut
        ang = np.linspace(0, math.pi, max(12, int(r / 2)))[1:-1]
        base = math.atan2(d[1], d[0])
        nn = np.array([math.cos(base + math.pi / 2), math.sin(base + math.pi / 2)])
        dd = np.array([math.cos(base), math.sin(base)])
        return np.array([c + nn * r * math.cos(a) + dd * 0.16 * r * math.sin(a) for a in ang])
    end_cap = (cap if w[-1] < 0.95 * wmax else lip)(A[-1], w[-1], tg[-1]) if w[-1] > 1 else np.zeros((0, 2))
    start_cap = (cap if w[0] < 0.95 * wmax else lip)(A[0], w[0], -tg[0]) if w[0] > 1 else np.zeros((0, 2))
    return np.vstack([left, end_cap, right[::-1], start_cap])


def tier_polygon(base, top, width_px: float, tiers: int, seed: int = 1) -> np.ndarray:
    """Closed outline of a tiered silhouette (a conifer) from base to top: each tier a skirt whose tip droops, the
    edge running up and in to just above the next tier's tip; tips slightly uneven."""
    rng = np.random.default_rng(seed)
    b, t = np.asarray(base, float), np.asarray(top, float)
    ax = t - b
    L = float(np.linalg.norm(ax))
    u = ax / max(L, 1e-9)
    n = np.array([-u[1], u[0]])
    trunk = 0.06
    h = [trunk + (1 - trunk) * k / tiers for k in range(tiers)]
    step = (1 - trunk) / tiers
    w = [width_px / 2 * (1 - (hk - trunk) / (1 - trunk)) ** 0.85 * (1 + rng.normal(0, 0.06)) for hk in h]
    right, left = [], []
    for side, out in ((1, right), (-1, left)):
        for k in range(tiers):
            out.append(b + u * (h[k] * L) + side * n * w[k] * (1 + rng.normal(0, 0.04)))
            nxt_w = w[k + 1] if k + 1 < tiers else 0.0
            if k + 1 < tiers:
                out.append(b + u * ((h[k + 1] + 0.18 * step) * L) + side * n * (0.42 * w[k] + 0.2 * nxt_w))
    tw = 0.05 * width_px
    pts = [b + n * tw] + right + [t] + left[::-1] + [b - n * tw]
    return np.array(pts, float)
