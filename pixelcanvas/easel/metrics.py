"""Scores (DESIGN.md section 8) at full canvas resolution, per region and overall, and the worst-spots list.

``measure(ref, canvas)`` builds the per-pixel comparison once; ``.scores(mask)`` and ``.worst(mask)`` aggregate it.
All scores are 0-100 (higher is closer to the reference); ``lines`` is None where the reference has no lines.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from . import coords

WEIGHTS = {"colour": 0.35, "lines": 0.2, "structure": 0.25, "detail": 0.2}
AXES = ("colour", "lines", "structure", "detail")
WIN = 48            # worst-spot window, px
LINE_TOL = 1.5
EMPHASIS = {
    "": {"colour": 0.4, "lines": 0.3, "edges": 0.15, "detail": 0.15},
    "overall": {"colour": 0.4, "lines": 0.3, "edges": 0.15, "detail": 0.15},
    "lines": {"colour": 0.0, "lines": 1.0, "edges": 0.0, "detail": 0.0},
    "colour": {"colour": 1.0, "lines": 0.0, "edges": 0.0, "detail": 0.0},
    "structure": {"colour": 0.0, "lines": 0.0, "edges": 1.0, "detail": 0.0},
    "detail": {"colour": 0.0, "lines": 0.0, "edges": 0.0, "detail": 1.0},
}
EDGE_TOL = 2.0


def _lab(img_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_u8.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _skeleton(m: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    if not m.any():
        return m.copy()
    return skeletonize(m)


def _dt(m: np.ndarray) -> np.ndarray:
    """Distance (px) from every pixel to the nearest True pixel of m (large when m is empty)."""
    if not m.any():
        return np.full(m.shape, 1e6, np.float32)
    return cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)


def line_map(L: np.ndarray, long_side: int) -> np.ndarray:
    """Dark thin lines: pixels clearly darker than their closed (line-removed) surroundings."""
    k = int(max(7, round(long_side / 90))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(L, cv2.MORPH_CLOSE, ker)
    bh = closed - L
    m = (bh > 9.0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = stats[:, cv2.CC_STAT_AREA] >= 8
        keep[0] = False
        m = keep[lab]
    else:
        m = m > 0
    return m


def edge_map(gray_u8: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray_u8, (0, 0), 1.0)
    return cv2.Canny(g, 40, 100, L2gradient=True) > 0


class Maps:
    """Per-image analysis maps (reusable for the reference)."""

    def __init__(self, img_u8: np.ndarray):
        H, W = img_u8.shape[:2]
        self.img = img_u8
        blur = cv2.GaussianBlur(img_u8, (0, 0), 1.5)
        self.lab = _lab(blur)
        L = _lab(img_u8)[..., 0]
        self.dark = L < 30.0
        self.lines = line_map(L, max(H, W))
        self.skel = _skeleton(self.lines)
        self.dt_skel = _dt(self.skel)
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
        self.edges = edge_map(gray)
        self.dt_edges = _dt(self.edges)
        self.hp = L - cv2.GaussianBlur(L, (0, 0), 2.0)


def _box(a: np.ndarray, k: int, normalize: bool = True) -> np.ndarray:
    return cv2.boxFilter(a.astype(np.float32), -1, (k, k), normalize=normalize, borderType=cv2.BORDER_REFLECT)


class Comparison:
    def __init__(self, ref: Maps, can: Maps, paper=None, use_lines: bool = True):
        self.ref, self.can = ref, can
        self.use_lines = use_lines
        self.H, self.W = ref.img.shape[:2]
        self.de = np.sqrt(((ref.lab - can.lab) ** 2).sum(axis=2)).astype(np.float32)
        self.line_missing = ref.skel & (can.dt_skel > LINE_TOL)
        # a dark line drawn where the reference is dark (the edge of a black mass) is not an extra line
        near_dark = cv2.dilate(ref.dark.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.line_extra = can.skel & (ref.dt_skel > LINE_TOL) & ~near_dark
        self.edge_missing = ref.edges & (can.dt_edges > EDGE_TOL)
        self.edge_extra = can.edges & (ref.dt_edges > EDGE_TOL)
        k = 15
        self.e_r = _box(ref.hp ** 2, k)
        self.e_c = _box(can.hp ** 2, k)
        self.cross = _box(ref.hp * can.hp, k)
        self.paper = None if paper is None else np.asarray(paper, np.float32)
        self._err = None

    # ---------------------------------------------------------------- scores
    def scores(self, mask: Optional[np.ndarray] = None) -> Dict[str, Optional[int]]:
        if mask is None:
            mask = np.ones((self.H, self.W), bool)
        mask = np.asarray(mask) > 0
        n = int(mask.sum())
        if n == 0:
            return {a: None for a in (*AXES, "overall")}
        out: Dict[str, Optional[float]] = {}
        mean_de = float(self.de[mask].mean())
        out["colour"] = 100.0 * float(np.clip((30.0 - mean_de) / 28.5, 0.0, 1.0))
        # lines
        min_px = max(15, int(0.0004 * n))
        r_sk = int((self.ref.skel & mask).sum())
        c_sk = int((self.can.skel & mask).sum())
        if r_sk < min_px or not self.use_lines:
            out["lines"] = None
        else:
            rec = 1.0 - (self.line_missing & mask).sum() / r_sk
            prec = 1.0 - (self.line_extra & mask).sum() / c_sk if c_sk else 0.0
            out["lines"] = 100.0 * (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
        # structure
        r_e = int((self.ref.edges & mask).sum())
        c_e = int((self.can.edges & mask).sum())
        if r_e < min_px:
            out["structure"] = 100.0 if c_e < 2 * min_px else max(0.0, 100.0 - 100.0 * c_e / (n * 0.02 + 1))
        else:
            rec = 1.0 - (self.edge_missing & mask).sum() / r_e
            prec = 1.0 - (self.edge_extra & mask).sum() / c_e if c_e else 0.0
            out["structure"] = 100.0 * (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
        # detail
        floor = 0.5
        er = float(self.e_r[mask].mean()) + floor
        ec = float(self.e_c[mask].mean()) + floor
        ratio = ec / er if ec < er else float(np.sqrt(er / ec))
        w = self.e_r[mask]
        if er - floor > 2.0:
            denom = np.sqrt(np.maximum(self.e_r[mask] * self.e_c[mask], 1e-6))
            corr = float(np.sum(w * np.clip(self.cross[mask] / denom, -1, 1)) / max(1e-6, w.sum()))
            out["detail"] = 100.0 * (0.5 * ratio + 0.5 * max(0.0, corr))
        else:
            out["detail"] = 100.0 * ratio
        out["overall"] = overall(out)
        return {k: (None if v is None else int(round(v))) for k, v in out.items()}

    # ---------------------------------------------------------------- error map / worst spots
    def components(self):
        """Window-averaged error components, each roughly 0..1 (box filter of WIN px)."""
        if self._err is None:
            k = WIN
            colour = _box(np.clip(self.de / 30.0, 0, 1), k)
            # counts of wrong skeleton / edge pixels per window: about one window-wide wrong line saturates
            lines = np.clip(_box((self.line_missing | self.line_extra).astype(np.float32), k) * k * k / (1.5 * k), 0, 1)
            if not self.use_lines:
                lines = np.zeros_like(lines)
            edges = np.clip(_box((self.edge_missing | self.edge_extra).astype(np.float32), k) * k * k / (3.0 * k), 0, 1)
            er, ec = _box(self.e_r, k), _box(self.e_c, k)
            detail = np.clip((er - ec) / (er + 4.0), 0, 1)
            self._err = {"colour": colour, "lines": lines, "edges": edges, "detail": detail}
        return self._err

    def error_map(self) -> np.ndarray:
        """Per-pixel error 0..1 for the diff heat-map (less smoothed than the windows)."""
        e = np.clip(self.de / 25.0, 0, 1)
        lines = cv2.dilate((self.line_missing | self.line_extra).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        e = np.maximum(e, lines * 0.8)
        return cv2.GaussianBlur(e.astype(np.float32), (0, 0), 2.0)

    def worst(self, mask: Optional[np.ndarray] = None, n: int = 5, min_err: float = 0.1,
              emphasis: Optional[str] = None) -> List[dict]:
        """Up to n windows with the largest error. ``emphasis`` (lines / colour / structure / detail) ranks that
        kind of problem first, so a stage's report points at what that stage can fix."""
        H, W = self.H, self.W
        comp = self.components()
        wts = EMPHASIS.get(emphasis or "", EMPHASIS[""])
        total = sum(wts[k] * comp[k] for k in wts)
        if mask is not None:
            m = np.asarray(mask, bool)
            cover = _box(m.astype(np.float32), WIN)
            need = 0.5 * min(1.0, float(m.sum()) / (WIN * WIN))
            total = total * (cover >= need) * m
        t = total.copy()
        out = []
        half = WIN // 2
        tries = 0
        while len(out) < n and tries < 4 * n:
            tries += 1
            idx = int(np.argmax(t))
            y, x = divmod(idx, W)
            v = float(t[y, x])
            if v < min_err:
                break
            x0, y0 = max(0, x - half), max(0, y - half)
            x1, y1 = min(W, x + half), min(H, y + half)
            cc = {k: float(c[y, x]) * wts[k] for k, c in comp.items()}
            kind = max(cc, key=cc.get)
            if kind == "colour":
                sl = (slice(y0, y1), slice(x0, x1))
                d = self.de[sl] if mask is None else self.de[sl][np.asarray(mask, bool)[sl]]
                if d.size and float(d.mean()) < 5.0:      # too small to name; look further
                    t[max(0, y - half):min(H, y + half), max(0, x - half):min(W, x + half)] = 0
                    continue
            out.append({"box_px": (x0, y0, x1, y1), "box": coords.px_box_to_units((x0, y0, x1, y1), W, H),
                        "error": round(v, 3), "kind": "structure" if kind == "edges" else kind,
                        "problem": self.diagnose((x0, y0, x1, y1), wts, mask)})
            t[max(0, y - WIN):min(H, y + WIN), max(0, x - WIN):min(W, x + WIN)] = 0
        return out

    def diagnose(self, box_px, wts: Optional[Dict[str, float]] = None, mask: Optional[np.ndarray] = None) -> str:
        """Plain words for the main problem in a window, measured on the region's own pixels in it."""
        x0, y0, x1, y1 = box_px
        sl = (slice(y0, y1), slice(x0, x1))
        sub = np.ones((y1 - y0, x1 - x0), bool)
        if mask is not None:
            msub = np.asarray(mask, bool)[sl]
            if msub.sum() >= 0.1 * msub.size:
                sub = msub

        def px(a):
            return a[sl][sub]
        wts = wts or EMPHASIS[""]
        comp = {k: float(v[(y0 + y1) // 2, (x0 + x1) // 2]) for k, v in self.components().items()}
        weighted = {k: wts[k] * comp[k] for k in wts}
        top = max(weighted, key=weighted.get)
        rl, cl = px(self.ref.lab).mean(0), px(self.can.lab).mean(0)
        if top == "colour" and self.paper is not None:
            can_rgb = px(self.can.img).astype(np.float32)
            unpainted = np.mean(np.abs(can_rgb - self.paper).max(axis=1) < 6)
            rr = np.median(px(self.ref.img), 0)
            paper_de = float(np.sqrt(((_lab(rr.reshape(1, 1, 3).astype(np.uint8))[0, 0] -
                                       _lab(self.paper.reshape(1, 1, 3).clip(0, 255).astype(np.uint8))[0, 0]) ** 2
                                      ).sum()))
            # bare paper counts as painted where the reference has (almost) the paper colour
            if unpainted > 0.85 and float(px(self.de).mean()) > 3.6 and paper_de >= 8.0:
                return f"not painted yet (the reference is {coords.colour_name(rr)} {coords.hex_colour(rr)} here)"
        if top == "colour":
            mean_de = float(px(self.de).mean())
            if float(np.sqrt(((rl - cl) ** 2).sum())) < 0.5 * mean_de and                     px(self.edge_missing).sum() + px(self.edge_extra).sum() > 0.5 * max(1, px(self.ref.edges).sum()):
                return "average colour right, shapes wrong (edges missing or misplaced)"
            # judge on the typical (median) colour of each side, the same colour the words name, so the verdict
            # and the names cannot contradict each other; mixed windows (a dark line beside a white area) are
            # about shape, not colour
            ref_rgb = np.median(px(self.ref.img), 0)
            can_rgb = np.median(px(self.can.img), 0)
            rm, cm = _lab(ref_rgb.reshape(1, 1, 3).astype(np.uint8))[0, 0], \
                _lab(can_rgb.reshape(1, 1, 3).astype(np.uint8))[0, 0]
            if float(np.sqrt(((rm - cm) ** 2).sum())) < 4.0:
                return "right colour, wrong shapes (some edges or lines sit in the wrong place)"
            rn, cn = coords.colour_name(ref_rgb), coords.colour_name(can_rgb)
            names = (f"(ref {rn} {coords.hex_colour(ref_rgb)}, canvas {cn} {coords.hex_colour(can_rgb)})")
            dL = float(cm[0] - rm[0])
            rc, cc = float(np.hypot(rm[1], rm[2])), float(np.hypot(cm[1], cm[2]))
            if abs(dL) >= 6 and abs(dL) >= abs(cc - rc):
                return ("colour too dark " if dL < 0 else "colour too light ") + names
            if abs(cc - rc) >= 6:
                return ("colour too dull " if cc < rc else "colour too saturated ") + names
            return ("wrong colour " if rn != cn else "colour a little off ") + names
        if top == "lines":
            miss = int(px(self.line_missing).sum())
            extra = int(px(self.line_extra).sum())
            ra, ca = int(px(self.ref.lines).sum()), int(px(self.can.lines).sum())
            if miss >= extra:
                return "lines missing"
            if ca > 1.8 * max(1, ra) and miss < 0.3 * max(1, int(px(self.ref.skel).sum())):
                return "lines too heavy"
            return "extra lines"
        if top == "edges":
            miss = int(px(self.edge_missing).sum())
            extra = int(px(self.edge_extra).sum())
            return "shape edges missing (too soft)" if miss >= extra else "extra edges (blotchy)"
        er, ec = float(px(self.e_r).mean()), float(px(self.e_c).mean())
        return "too smooth (detail missing)" if ec < er else "too noisy"


def overall(s: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [(s[a], WEIGHTS[a]) for a in AXES if s.get(a) is not None]
    if not vals:
        return None
    wmean = sum(v * w for v, w in vals) / sum(w for _, w in vals)
    return 0.6 * wmean + 0.4 * min(v for v, _ in vals)


_REF_CACHE: Dict[int, Maps] = {}


def ref_maps(ref: np.ndarray) -> Maps:
    key = id(ref)
    m = _REF_CACHE.get(key)
    if m is None or m.img is not ref:
        m = Maps(ref)
        _REF_CACHE.clear()
        _REF_CACHE[key] = m
    return m


def measure(ref: np.ndarray, canvas: np.ndarray, paper=None, use_lines: bool = True) -> Comparison:
    """use_lines=False for painterly references, where dark thin marks are texture rather than line art."""
    return Comparison(ref_maps(ref), Maps(canvas), paper=paper, use_lines=use_lines)


def fmt_scores(s: Dict[str, Optional[int]], keys=("colour", "lines", "structure", "detail", "overall")) -> str:
    return "  ".join(f"{k} {'-' if s.get(k) is None else s[k]}" for k in keys)


def fmt_delta(before: Dict[str, Optional[int]], after: Dict[str, Optional[int]],
              keys=("colour", "lines", "structure", "detail", "overall")) -> str:
    parts = []
    for k in keys:
        b, a = before.get(k), after.get(k)
        if a is None and b is None:
            continue
        parts.append(f"{k} {'-' if b is None else b} -> {'-' if a is None else a}")
    return "   ".join(parts)
