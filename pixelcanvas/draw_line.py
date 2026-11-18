"""Line art without a reference: a few points in, one clean tapered line out.

- The points are joined by a centripetal Catmull-Rom spline (the stabiliser).
- ``snap`` (on by default): a point within ~1.2 % of the canvas of a sketch line moves onto it, and between two
  points snapped to the same sketch line the ink follows that sketch line exactly.
- ``join`` (on by default): an end within ~1.5 % of another line (or of its own start) is pulled onto it, so
  shapes close and flat fills do not leak.
- ``mirror``: a mirrored twin across a vertical axis.
- Width follows a pressure profile (thin tapered ends); every line has an id (L1, L2 ...) and can be moved,
  redrawn, re-weighted or deleted alone.
- ``analyze`` lists open gaps between line ends (G1 ...) and stray ends, with coordinates.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import coords
from .coords import UsageError
from .draw_core import (catmull_rom, dstate, find_record, next_id, next_order, path_length, px2u, rdp, resample,
                        rnd, smooth_poly, taper_widths, u2px)


def base_width(p) -> float:
    return max(1.6, max(p.W, p.H) / 340.0)


def snap_radius(p) -> float:
    return 0.012 * max(p.W, p.H)


def join_radius(p) -> float:
    return 0.016 * max(p.W, p.H)


# ============================================================================================ fitting

def _follow(poly: np.ndarray, closed: bool, i: int, j: int) -> np.ndarray:
    if i == j:
        return poly[i:i + 1]
    if not closed:
        return poly[i:j + 1] if i < j else poly[j:i + 1][::-1]
    n = len(poly)
    fwd = (j - i) % n
    bwd = (i - j) % n
    if fwd <= bwd:
        idx = [(i + k) % n for k in range(fwd + 1)]
    else:
        idx = [(i - k) % n for k in range(bwd + 1)]
    return poly[idx]


HAIR_WORDS = ("hair", "bang", "fringe", "lock", "tail", "fur", "mane", "braid", "ponytail")


def default_bow(p, pts_u) -> float:
    """Lines in a hair part bow their straight clump edges outward (a fringe of clumps, not a saw)."""
    rec = {"path": [list(q) for q in pts_u]}
    part = guess_part(p, rec) or ""
    return 0.08 if any(w in part for w in HAIR_WORDS) else 0.0


def _bowed(A, B, other, bow: float, tip_is_b: bool = True) -> np.ndarray:
    """A curve from A to B that bulges away from ``other`` (the far end of the neighbouring edge). The bulge sits
    near the root end, so the tip stays pointed and the notch between clumps is pinched."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    d = B - A
    L = math.hypot(*d)
    if L < 1e-6 or bow <= 0:
        return resample(np.array([A, B]), 1.5)
    n = np.array([-d[1], d[0]]) / L
    s = 1.0 if (n @ (np.asarray(other, float) - (A + B) / 2)) < 0 else -1.0
    t = np.linspace(0, 1, max(3, int(L / 1.5)))[:, None]
    root = 0.3 if tip_is_b else 0.7
    c = A + d * root + s * n * bow * L * 2
    return (1 - t) ** 2 * A + 2 * (1 - t) * t * c + t ** 2 * B


def fit(p, pts_u, closed: bool = False, snap: bool = True, join: bool = True,
        exclude: Optional[str] = None, bow: Optional[float] = None) -> Tuple[np.ndarray, List[str]]:
    """Dense path (px) through the model's points, snapped and joined. Returns (path, notes)."""
    from . import draw_sketch
    P = u2px(p, pts_u)
    notes: List[str] = []
    snapped: List[Optional[Tuple[int, int]]] = [None] * len(P)
    polys = draw_sketch.all_polylines(p) if snap else []
    if polys:
        r = snap_radius(p)
        for i, q in enumerate(P):
            best = None
            for k, (eid, poly, cl) in enumerate(polys):
                dd = np.hypot(poly[:, 0] - q[0], poly[:, 1] - q[1])
                j = int(np.argmin(dd))
                if dd[j] < r and (best is None or dd[j] < best[0]):
                    best = (float(dd[j]), k, j)
            if best is not None:
                P[i] = polys[best[1]][1][best[2]]
                snapped[i] = (best[1], best[2])
        ns = sum(s is not None for s in snapped)
        if ns:
            ids = sorted({polys[s[0]][0] for s in snapped if s is not None})
            notes.append(f"{ns} of {len(P)} points snapped to sketch {', '.join(ids)}")
    pieces: List[np.ndarray] = []
    run: List[np.ndarray] = [P[0]]
    corners = sharp_corners(P, closed)
    if corners:
        notes.append(f"{len(corners)} sharp corner(s) kept")
    if bow is None:
        # a zigzag of 3+ sharp corners is a fringe, a fur edge, a flame: bow its edges; else only in hair parts
        bow = (0.08 if len(corners) >= 3 else default_bow(p, pts_u)) if corners else 0.0
    for i in range(len(P) - 1):
        if i in corners and len(run) > 1:
            # a sharp turn (a bang tip, a collar point): end the smooth run here and start a new one
            pieces.append(catmull_rom(run, spacing=1.5))
            run = [P[i]]
        if bow != 0 and (i in corners or i + 1 in corners) and len(run) == 1:
            # an edge between corners is one side of a clump: it bulges away from the clump's other side, which
            # meets it at the tip (the lower end for hair that hangs; the upper end with a negative bow)
            lower = i + 1 if (P[i + 1][1] > P[i][1]) == (bow > 0) else i
            if lower == i + 1:
                other = P[i + 2] if i + 2 < len(P) else P[i + 1] + (P[i + 1] - P[i])
            else:
                other = P[i - 1] if i > 0 else P[i] + (P[i] - P[i + 1])
            pieces.append(_bowed(P[i], P[i + 1], other, abs(bow), tip_is_b=(lower == i + 1)))
            run = [P[i + 1]]
            continue
        a, b = snapped[i], snapped[i + 1]
        if a is not None and b is not None and a[0] == b[0] and not closed:
            poly = polys[a[0]][1]
            seg = _follow(poly, polys[a[0]][2], a[1], b[1])
            straight = math.hypot(*(P[i + 1] - P[i]))
            if path_length(seg) < 2.2 * straight + 6:
                if len(run) > 1:
                    pieces.append(catmull_rom(run, spacing=1.5))
                pieces.append(seg)
                run = [P[i + 1]]
                continue
        run.append(P[i + 1])
    if len(run) > 1 or not pieces:
        pieces.append(catmull_rom(run, closed=closed and not pieces, spacing=1.5))
    path = np.vstack(pieces)
    path = resample(path, 1.5)
    if len(path) > 6 and not corners:
        path = smooth_poly(path, 1.2, closed=False)
    if closed and corners and len(path) > 2 and math.hypot(*(path[0] - path[-1])) > 1:
        path = np.vstack([path, catmull_rom([path[-1], P[0]], spacing=1.5)[1:]])
    if join and not closed and len(path) >= 2:
        path, jn = _join(p, path, exclude)
        notes.extend(jn)
    return path, notes


SHARP_DEG = 65.0


def sharp_corners(P: np.ndarray, closed: bool = False) -> set:
    """Indices of control points where the path turns by more than SHARP_DEG (kept as corners, not rounded)."""
    out = set()
    n = len(P)
    for i in range(1, n - 1):
        a, b = P[i] - P[i - 1], P[i + 1] - P[i]
        na, nb = math.hypot(*a), math.hypot(*b)
        if na < 1e-6 or nb < 1e-6:
            continue
        cosang = float(np.clip((a @ b) / (na * nb), -1, 1))
        if math.degrees(math.acos(cosang)) > SHARP_DEG:
            out.add(i)
    return out


COS_ALONG = math.cos(math.radians(35))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = math.hypot(*v)
    return v / n if n > 1e-9 else np.zeros(2)


def end_dir(path: np.ndarray, end: int, k: float = 6.0) -> np.ndarray:
    """Unit direction at an end of a polyline, pointing out of the line (measured over ``k`` px)."""
    P = np.asarray(path, np.float64)
    Q = P if end == 0 else P[::-1]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    i = int(np.clip(np.searchsorted(s, min(k, s[-1])), 1, len(Q) - 1)) if len(Q) > 1 else 0
    return _unit(Q[0] - Q[i])


def hermite(A, tA, B, tB, spacing: float = 1.5) -> np.ndarray:
    """A cubic from A to B leaving A along ``tA`` and arriving at B along ``tB`` (unit vectors)."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    L = math.hypot(*(B - A))
    t = np.linspace(0, 1, max(3, int(math.ceil(L / spacing)) + 1))[:, None]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * A + h10 * L * np.asarray(tA) + h01 * B + h11 * L * np.asarray(tB)


def _cut(path: np.ndarray, end: int, dist: float) -> np.ndarray:
    """``path`` without its last ``dist`` px at ``end`` (at least two points stay)."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    if end == 0:
        keep = s >= dist
        keep[-2:] = True
    else:
        keep = s <= s[-1] - dist
        keep[:2] = True
    return path[keep]


def bridge(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """One smooth path: P1, then P2 (P1's end is near P2's start); the seam is a tangent curve, no kink."""
    gap = math.hypot(*(P2[0] - P1[-1]))
    b = max(3.0, 1.5 * gap)
    A = _cut(P1, -1, min(b, 0.3 * path_length(P1)))
    B = _cut(P2, 0, min(b, 0.3 * path_length(P2)))
    mid = hermite(A[-1], end_dir(A, -1), B[0], -end_dir(B, 0))
    return resample(np.vstack([A, mid[1:-1], B]), 1.5)


def continues(path: np.ndarray, end: int, other: np.ndarray, r: float) -> Optional[int]:
    """The end of ``other`` that ``path``'s ``end`` runs on into (they meet end to end, same direction), else None."""
    if len(other) < 2 or len(path) < 2:
        return None
    q, out = path[end], end_dir(path, end)
    for oe in (0, -1):
        if math.hypot(*(other[oe] - q)) < r and float(out @ end_dir(other, oe)) < -COS_ALONG:
            return oe
    return None


def _join(p, path: np.ndarray, exclude: Optional[str], ends=(0, -1), extra=(),
          r: Optional[float] = None) -> Tuple[np.ndarray, List[str]]:
    """Ends near another line meet it smoothly: an end running on from another line's end continues it, an end
    running alongside a line merges into it along its direction, an end that meets a line at an angle runs straight
    on to it. The last stretch is redrawn as one tangent curve (no kink, no sideways pull)."""
    d = dstate(p)
    r = r or join_radius(p)
    notes = []
    others = [(rec["id"], record_path(p, rec)) for rec in list(d["lines"]) + list(extra) if rec["id"] != exclude]
    L = path_length(path)
    # own ends close together: close the loop
    if L > 6 * r and math.hypot(*(path[0] - path[-1])) < r:
        path = np.vstack([path, path[:1]])
        return path, ["the ends met: the line is closed"]
    for end in ends:
        q = path[end]
        best = None
        for oid, op in others:
            if len(op) < 2:
                continue
            dd = np.hypot(op[:, 0] - q[0], op[:, 1] - q[1])
            j = int(np.argmin(dd))
            if dd[j] < r and (best is None or dd[j] < best[0]):
                best = (float(dd[j]), oid, op, j)
        if best is None or best[0] < 0.5:
            continue
        gap, oid, op, j = best
        out = end_dir(path, end)
        reach = min(3.0 * join_radius(p) + gap, 0.4 * path_length(path))
        Q = path if end == -1 else path[::-1]           # the joining end last
        A = _cut(Q, -1, reach)
        oe = continues(path, end, op, r)
        to = _unit(op[min(j + 2, len(op) - 1)] - op[max(j - 2, 0)])
        if oe is not None:
            # end to end: run on into the other line
            B, tB, how = op[oe], -end_dir(op, oe), " (runs on from its end)"
        elif abs(float(out @ to)) > COS_ALONG:
            # alongside: merge into the other line a little further on, along its direction
            sg = 1 if float(out @ to) > 0 else -1
            s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(op, axis=0).T))])
            k = int(np.clip(np.searchsorted(s, s[j] + sg * 0.5 * reach), 0, len(op) - 1))
            B, tB, how = op[k], to * sg, " (merges along it)"
        else:
            B = op[j]
            tB, how = _unit(out + _unit(B - A[-1])), ""
        curve = hermite(A[-1], end_dir(A, -1), B, tB)
        Q = resample(np.vstack([A, curve[1:]]), 1.5)
        path = Q if end == -1 else Q[::-1]
        notes.append(f"{'start' if end == 0 else 'end'} joined to {oid}{how}")
    return path, notes


def unshared(path: np.ndarray, others, min_len: float) -> List[np.ndarray]:
    """The stretches of ``path`` (at least ``min_len`` px) that no other line inks already. ``others`` = [(path px,
    reach px)]: a sample within ``reach`` of that path is inked there. Each stretch kept reaches back into the inked
    part by that distance, so it still meets the line it stops at."""
    from scipy.spatial import cKDTree
    path = np.asarray(path, np.float64)
    others = [(np.asarray(o, np.float64), hw) for o, hw in others if len(o)]
    if not others or len(path) < 2:
        return [path]
    pts = np.vstack([o for o, _ in others])
    hws = np.concatenate([np.full(len(o), hw) for o, hw in others])
    dist, idx = cKDTree(pts).query(path)
    reach = hws[idx]
    inked = dist < reach
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    n = len(path)
    # a short inked stretch is a crossing or a touch, not a doubled line
    i = 0
    while i < n:
        if not inked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inked[j + 1]:
            j += 1
        if s[j] - s[i] < 3 * float(reach[i:j + 1].max()):
            inked[i:j + 1] = False
        i = j + 1
    if not inked.any():
        return [path]
    out = []
    i = 0
    while i < n:
        if inked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not inked[j + 1]:
            j += 1
        a = np.searchsorted(s, s[i] - (reach[i - 1] if i > 0 else 0))
        b = np.searchsorted(s, s[j] + (reach[j + 1] if j + 1 < n else 0), side="right")
        if s[j] - s[i] >= min_len:
            out.append(path[a:max(b, a + 2)])
        i = j + 1
    return out


def record_path(p, rec) -> np.ndarray:
    return resample(u2px(p, rec["path"]), 1.5) if len(rec["path"]) > 1 else u2px(p, rec["path"])


def make_record(p, path_px: np.ndarray, ctrl_u, *, weight: float, taper: str, color: str, closed: bool,
                src: str = "") -> dict:
    d = dstate(p)
    eps = 0.25
    pr = rdp(path_px, eps)
    rec = {"id": next_id(d, "L"), "n": next_order(d), "ctrl": rnd(ctrl_u), "path": rnd(px2u(p, pr), 2),
           "weight": round(float(weight), 2), "taper": taper, "color": color, "closed": bool(closed)}
    lt = d.get("light", {})
    rec["lt"] = [round(float(lt.get("deg", 135.0)), 1), round(float(lt.get("z", 0.6)), 2)]
    if src:
        rec["src"] = src
    rec["part"] = guess_part(p, rec)
    return rec


def guess_part(p, rec) -> Optional[str]:
    """The planned part most of the line lies in (smallest box first; boxes get a small margin, since plan boxes are
    rough)."""
    d = dstate(p)
    pts = np.asarray(rec["path"], float)
    m = 15.0
    boxes = sorted(((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]), n, b["box"]) for n, b in d["parts"].items())
    votes: Dict[str, int] = {}
    for q in pts[:: max(1, len(pts) // 16)]:
        for _, n, b in boxes:
            if b[0] - m <= q[0] <= b[2] + m and b[1] - m <= q[1] <= b[3] + m:
                votes[n] = votes.get(n, 0) + 1
                break
    return max(votes, key=votes.get) if votes else None


def pressure_profile(p, rec, path: np.ndarray) -> np.ndarray:
    """A pen's weight along the line (factor, mean ~1): heavier where the line's outer side faces away from the light,
    lighter toward the light, a slow swell from the hand; the light is the one set when the line was drawn, so a
    line renders the same every time. Outer side: away from the middle of the line's part (or its own middle)."""
    from .draw_core import dstate
    n = len(path)
    if n < 3 or rec.get("even") or not rec.get("lt"):
        return np.ones(n)          # records drawn before pressure existed render as they were (owned-layer folds)
    deg, z = rec["lt"]
    a = math.radians(float(deg))
    lv = np.array([math.cos(a), -math.sin(a)])
    t = np.gradient(path, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([t[:, 1], -t[:, 0]], 1)
    d = dstate(p)
    part = d["parts"].get(rec.get("part") or "", {})
    if part.get("box"):
        b = part["box"]
        c = np.array([(b[0] + b[2]) / 2 * p.W / 1000.0, (b[1] + b[3]) / 2 * p.H / 1000.0])
    else:
        c = path.mean(0)
    out = np.where(((path - c) * nrm).sum(1, keepdims=True) < 0, -nrm, nrm)
    from scipy.ndimage import gaussian_filter1d
    facing = gaussian_filter1d((out * lv).sum(1), max(1.0, n / 12.0), mode="nearest")
    rng = np.random.default_rng(sum(ord(ch) for ch in str(rec.get("id", "L"))) * 7919)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    L = max(float(s[-1]), 1e-6)
    # the hand's pressure swells and eases every 40-80 delivered px (a professional line changes weight within a
    # stroke, not only at its ends)
    from .draw_core import work_scale
    period = rng.uniform(40.0, 80.0) * work_scale(p)
    swell = 1.0 + 0.22 * np.sin(2 * math.pi * s / period + rng.uniform(0, 6.28))
    f = (1.0 - 0.4 * np.clip(facing, 0, 1) + 0.5 * np.clip(-facing, 0, 1)) * swell
    return f / max(1e-6, float(np.mean(f)))


def render_ops(p, rec) -> list:
    from .draw_plan import swatch
    from .ops import Stroke
    path = record_path(p, rec)
    if len(path) < 2:
        return []
    w = base_width(p) * float(rec.get("weight", 1.0))
    L = path_length(path)
    if L < 4 * w and w < 3:
        w = max(1.0, L / 4.0)
    if L < 4.0:
        return []
    taper = "none" if rec.get("closed") else rec.get("taper", "both")
    widths = taper_widths(path, w, taper)
    if (p.state.get("style") or "") not in ("poster",):
        widths = np.maximum(1.0, widths * pressure_profile(p, rec, path)).astype(np.float32)
    if rec.get("wprof"):
        # a width that changes along the line: [[arc fraction, factor], ...]
        kt, kv = np.asarray(rec["wprof"], np.float64).T
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
        widths = np.maximum(1.0, widths * np.interp(s / max(L, 1e-9), kt, kv)).astype(np.float32)
    if rec.get("auto_rgb") and rec.get("color") in (None, "line"):
        col = tuple(int(v) for v in rec["auto_rgb"])
    else:
        try:
            col = swatch(p, rec.get("color") or "line")
        except UsageError:
            col = (34, 28, 36)
    return [Stroke(pts=path.astype(np.float32), width=widths, color=tuple(int(c) for c in col), brush="liner")]


def line_colour(fill_rgb) -> Tuple[int, int, int]:
    """Coloured line art: a darker, more saturated version of the fill beside the line, not near-black."""
    from .draw_core import lch_to_rgb, rgb_to_lch
    L, C, h = rgb_to_lch(tuple(int(v) for v in fill_rgb))
    return lch_to_rgb(float(np.clip(0.36 * L, 9.0, 34.0)), min(60.0, C * 1.2 + 5.0), h)


def recolour(p, under: np.ndarray) -> int:
    """Give every default-coloured line the colour of the fill on its part's side (``auto_rgb`` on the record);
    ``under`` is the picture under the line layer (uint8). Ink style keeps black lines. Returns the number changed."""
    from .draw_core import dstate
    d = dstate(p)
    if (p.state.get("style") or "") in ("ink", "poster"):
        return 0
    regs = p.state.get("regions", {})
    bg = {n for n in regs if n in ("background", "bg", "sky", "backdrop")}
    H, W = under.shape[:2]
    n_changed = 0
    masks: Dict[str, np.ndarray] = {}
    for rec in d["lines"]:
        if rec.get("color") not in (None, "line"):
            continue
        path = record_path(p, rec)
        if len(path) < 3:
            continue
        t = np.gradient(path, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([t[:, 1], -t[:, 0]], 1)
        off = base_width(p) * float(rec.get("weight", 1.0)) * 0.6 + 0.004 * max(W, H)
        best = None
        for sgn in (1.0, -1.0):
            q = path + sgn * off * nrm
            xi = np.clip(q[:, 0].astype(int), 0, W - 1)
            yi = np.clip(q[:, 1].astype(int), 0, H - 1)
            own = 0.0
            part = rec.get("part")
            if part in regs:
                if part not in masks:
                    masks[part] = p.region_mask(part)
                own = float(masks[part][yi, xi].mean())
            bgs = 0.0
            for b in bg:
                if b not in masks:
                    masks[b] = p.region_mask(b)
                bgs = max(bgs, float(masks[b][yi, xi].mean()))
            col = np.median(under[yi, xi].reshape(-1, 3), axis=0)
            lum_ = float(col @ np.array([0.2126, 0.7152, 0.0722]))
            score = own * 2.0 - bgs * 1.5 - lum_ / 255.0 * 0.3
            if best is None or score > best[0]:
                best = (score, col)
        rgb = [int(v) for v in line_colour(best[1])]
        if rec.get("auto_rgb") != rgb:
            rec["auto_rgb"] = rgb
            n_changed += 1
    return n_changed


# ============================================================================================ analysis

def analyze(p) -> dict:
    """Open ends, gaps (pairs of nearby open ends or an end near another line) and stray ends, all in px."""
    d = dstate(p)
    recs = d["lines"]
    S = max(p.W, p.H)
    paths = {r["id"]: record_path(p, r) for r in recs}
    conn = 0.006 * S + base_width(p)
    gap_r = 0.06 * S
    ends = []
    min_len = 0.1 * S             # brows, noses, mouths, lashes and fold ticks are meant to stay open
    for r in recs:
        P = paths[r["id"]]
        if len(P) < 2 or r.get("closed") or math.hypot(*(P[0] - P[-1])) < conn or path_length(P) < min_len:
            continue
        s_own = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
        for which, q in (("start", P[0]), ("end", P[-1])):
            dmin, target, dconn = 1e9, None, 1e9
            for oid, op in paths.items():
                if oid == r["id"]:
                    # the line's own far part only counts as a connection (a loop that closes on itself)
                    far = s_own > 6 * conn if which == "start" else s_own < s_own[-1] - 6 * conn
                    if far.any():
                        dd = np.hypot(op[far][:, 0] - q[0], op[far][:, 1] - q[1])
                        dconn = min(dconn, float(dd.min()))
                    continue
                dd = np.hypot(op[:, 0] - q[0], op[:, 1] - q[1])
                j = int(np.argmin(dd))
                if dd[j] < dmin:
                    dmin, target = float(dd[j]), (oid, op[j])
            ends.append({"line": r["id"], "which": which, "pt": q, "free": min(dmin, dconn) > conn,
                         "near": target, "dist": dmin})
    free = [e for e in ends if e["free"]]
    pairs = []
    for i, a in enumerate(free):
        for b in free[i + 1:]:
            if a["line"] == b["line"]:
                continue
            dd = math.hypot(*(a["pt"] - b["pt"]))
            if dd < gap_r:
                pairs.append((dd, a, b))
    used = set()
    gaps = []
    for dd, a, b in sorted(pairs, key=lambda t: t[0]):
        ka, kb = (a["line"], a["which"]), (b["line"], b["which"])
        if ka in used or kb in used:
            continue
        used |= {ka, kb}
        gaps.append({"a": a, "b_pt": b["pt"], "b": f"{b['line']} {b['which']}", "dist": dd})
    for a in free:
        if (a["line"], a["which"]) in used:
            continue
        if a["near"] is not None and a["dist"] < gap_r:
            used.add((a["line"], a["which"]))
            gaps.append({"a": a, "b_pt": a["near"][1], "b": a["near"][0], "dist": a["dist"]})
    gaps.sort(key=lambda g: g["dist"])
    for i, g in enumerate(gaps, 1):
        g["id"] = f"G{i}"
    stray = [a for a in free if (a["line"], a["which"]) not in used]
    return {"ends": ends, "free": free, "gaps": gaps, "stray": stray, "paths": paths}


def gap_lines(p, an: dict, n: int = 3) -> List[str]:
    out = []
    gaps = an["gaps"]
    if gaps:
        small = [g for g in gaps if g["dist"] < 0.025 * max(p.W, p.H)]
        head = (f"open gaps: {len(gaps)}" + (f" ({len(small)} small: almost always a slip)" if small else "")
                + "; close one with  pixelcanvas line close <G>")
        out.append(head)
        for g in gaps[:n]:
            a = g["a"]
            out.append(f"  {g['id']}: {a['line']} {a['which']} at {_u(p, a['pt'])} -> {g['b']} at {_u(p, g['b_pt'])}"
                       f" ({g['dist'] * 1000 / max(p.W, p.H):.0f} units)")
    if an["stray"]:
        s = an["stray"][:3]
        out.append(f"open ends: {len(an['stray'])} (fine for hair tips and fold lines): "
                   + ", ".join(f"{e['line']} {e['which']} {_u(p, e['pt'])}" for e in s))
    return out


def _u(p, q) -> str:
    return f"{int(round(q[0] * 1000 / p.W))},{int(round(q[1] * 1000 / p.H))}"


def crossings(p, rec, max_n: int = 3) -> List[str]:
    """Where a line crosses lines of a *different* focus part (bang tips through an eye, a jaw through the hair)."""
    d = dstate(p)
    part = rec.get("part")
    focus = {n: v.get("focus", 2) for n, v in d["parts"].items()}
    A = record_path(p, rec)
    if len(A) < 2:
        return []
    out = []
    for o in d["lines"]:
        if o["id"] == rec["id"] or not o.get("part") or o.get("part") == part or focus.get(o["part"], 1) < 3 and \
                focus.get(part, 1) < 3:
            continue
        B = record_path(p, o)
        if len(B) < 2:
            continue
        hit = _seg_cross(A, B)
        if hit is not None:
            out.append(f"{rec['id']} ({part or 'no part'}) crosses {o['id']} ({o['part']}) at {_u(p, hit)}: if that "
                       f"is not meant, redo it:  pixelcanvas line redo {rec['id']} \"x,y ...\"")
            if len(out) >= max_n:
                break
    return out


def _seg_cross(A: np.ndarray, B: np.ndarray):
    """First intersection point of two polylines (px), or None."""
    a0, a1 = A[:-1], A[1:]
    b0, b1 = B[:-1], B[1:]
    # bounding-box prefilter
    ax0, ax1 = np.minimum(a0[:, 0], a1[:, 0]), np.maximum(a0[:, 0], a1[:, 0])
    ay0, ay1 = np.minimum(a0[:, 1], a1[:, 1]), np.maximum(a0[:, 1], a1[:, 1])
    bx0, bx1 = np.minimum(b0[:, 0], b1[:, 0]), np.maximum(b0[:, 0], b1[:, 0])
    by0, by1 = np.minimum(b0[:, 1], b1[:, 1]), np.maximum(b0[:, 1], b1[:, 1])
    ov = (ax0[:, None] <= bx1[None]) & (bx0[None] <= ax1[:, None]) & (ay0[:, None] <= by1[None]) & \
        (by0[None] <= ay1[:, None])
    ii, jj = np.nonzero(ov)
    if not len(ii):
        return None
    p, r = a0[ii], a1[ii] - a0[ii]
    q, s = b0[jj], b1[jj] - b0[jj]
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    ok = np.abs(den) > 1e-9
    qp = q - p
    t = np.where(ok, (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / np.where(ok, den, 1), -1)
    u = np.where(ok, (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / np.where(ok, den, 1), -1)
    m = ok & (t > 0.02) & (t < 0.98) & (u > 0.02) & (u < 0.98)
    if not m.any():
        return None
    k = int(np.flatnonzero(m)[0])
    return p[k] + t[k] * r[k]


def eye_shape(box, opn: float = 0.85, tilt: float = 6.0, outer_left: bool = True):
    """An anime eye in a box (units): a closed outline and the heavy upper lash line above it. ``opn`` = how
    open (1 = round), ``tilt`` = degrees the outer corner sits higher."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) / 2, (y1 - y0) / 2 * opn
    s = -1 if outer_left else 1                 # direction from the inner to the outer corner
    t = math.radians(tilt)
    inner = (cx - s * w, cy + 0.15 * h)
    outer = (cx + s * w, cy - 0.1 * h - math.tan(t) * w)
    top = [(cx - s * 0.6 * w, cy - 0.8 * h), (cx - s * 0.1 * w, cy - 1.0 * h), (cx + s * 0.55 * w, cy - 0.9 * h)]
    # a round lower lid: the iris fills the eye down to it
    bot = [(cx + s * 0.75 * w, cy + 0.45 * h), (cx + s * 0.3 * w, cy + 0.95 * h), (cx - s * 0.3 * w, cy + 0.95 * h),
           (cx - s * 0.8 * w, cy + 0.6 * h)]
    loop = [inner, *top, outer, *bot]
    lash = [(inner[0] + s * 0.05 * w, inner[1] - 0.25 * h), (cx - s * 0.6 * w, cy - 0.88 * h),
            (cx - s * 0.1 * w, cy - 1.08 * h), (cx + s * 0.55 * w, cy - 0.98 * h), (outer[0] + s * 0.06 * w, outer[1])]
    return [list(q) for q in loop], [list(q) for q in lash]


def find(p, rid: str) -> dict:
    return find_record(dstate(p)["lines"], rid, "line", "pixelcanvas line list")


def describe(p, rec) -> str:
    pts = rec["ctrl"]
    return (f"{rec['id']} {len(pts)} points {coords.fmt_point(pts[0])} .. {coords.fmt_point(pts[-1])}  weight "
            f"{rec.get('weight', 1):g}" + (f"  part {rec['part']}" if rec.get("part") else "")
            + ("  closed" if rec.get("closed") else ""))
