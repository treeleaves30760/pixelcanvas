"""Anime (toon) rendering of 3D primitives: line art from the z-buffer, flat material colours and cel shading.

``toon(cam, prims, ...)`` renders the primitives once (part id, depth, normal, face code per pixel) and derives:

- **lines**: every *piece* (a sleeve, a lock, a finger, a wall face) is outlined where it meets the background or
  something behind it (silhouette: heaviest), where it lies over another piece of the same figure (inner line), where
  two materials meet at the same depth (seam: thin) and along creases inside a piece (pleat folds, box edges:
  thinnest). The piece in front draws the line, so no line is doubled. Lines are tapered strokes, heavier near the
  camera, in a darker shade of the colour they outline.
- **flat**: each pixel's material (a trim band and the inside of a garment are their own materials).
- **shade**: n . l of the plan light against one or two thresholds (cel bands), cast shadows from a shadow depth map
  rendered along the light (one sharp map per figure for hair on the face and sleeves on the torso, one for the whole
  scene for figures on the floor), and a hair highlight ring.

Pure functions on numpy arrays; nothing here touches the disk.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .pose3d import Camera, unit
from .render3d import INSIDE, PLEAT, TRIM, _near_segment, corners, intersect, light_world, render

W_SIL, W_OCC, W_CONTACT, W_SEAM, W_CREASE = 1.0, 0.68, 0.6, 0.42, 0.32


# ============================================================================================ segments

def segment_maps(R: dict, prims: List[dict]) -> Tuple[np.ndarray, List[tuple], np.ndarray]:
    """(segment index per pixel (-1 none), segment table [(piece index, sub)], piece index per primitive). A segment
    is a piece, split by box faces and pleat halves (their borders are creases)."""
    keys: Dict[tuple, int] = {}
    pidx = np.zeros(len(prims) + 1, np.int32)
    for k, pr in enumerate(prims):
        key = (pr.get("owner", ""), pr.get("piece") or pr.get("part") or f"p{k}")
        pidx[k] = keys.setdefault(key, len(keys))
    pidx[-1] = -1
    ids = R["id"]
    piece = pidx[ids]                                   # ids == -1 picks the last entry (-1)
    kind = np.array([{"box": 1, "cyl": 2, "shell": 3}.get(pr["t"], 0) for pr in prims] + [0], np.int8)[ids]
    face = R["face"].astype(np.int32)
    sub = np.where(kind == 1, face, np.where(kind == 2, face, np.where(kind == 3, face & PLEAT, 0)))
    seg_key = np.where(piece >= 0, piece * 16 + sub, -1)
    uniq, inv = np.unique(seg_key, return_inverse=True)
    inv = inv.reshape(seg_key.shape)
    off = 1 if uniq[0] == -1 else 0
    seg = inv - off
    table = [(int(u // 16), int(u % 16)) for u in uniq[off:]]
    return seg.astype(np.int32), table, pidx[:-1]


# ============================================================================================ lines

def _smooth_closed(P: np.ndarray, sigma: float) -> np.ndarray:
    n = len(P)
    if n < 6 or sigma < 0.3:
        return P
    r = min(int(3 * sigma), n // 2 - 1)
    if r < 1:
        return P
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.vstack([P[-r:], P, P[:r]])
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)], 1)[:n]


def _circ_smooth(v: np.ndarray, sigma: float, closed: bool = True) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(v.astype(np.float64), sigma, mode="wrap" if closed else "nearest")


def _runs(on: np.ndarray, closed: bool) -> List[np.ndarray]:
    n = len(on)
    if not on.any():
        return []
    if on.all():
        return [np.arange(n)]
    if closed:
        start = int(np.argmin(on))
        order = np.roll(np.arange(n), -start)
    else:
        order = np.arange(n)
    out, cur = [], []
    for i in order:
        if on[i]:
            cur.append(i)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def _taper(s: np.ndarray, L: float, w: float, closed: bool) -> np.ndarray:
    if closed:
        return np.ones(len(s))
    ln = min(0.35 * L, 6.0 * w + 3.0)
    if ln <= 0:
        return np.ones(len(s))

    def ss(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)
    return (0.15 + 0.85 * ss(s / ln)) * (0.15 + 0.85 * ss((L - s) / ln))


def lines(R: dict, prims: List[dict], seg: np.ndarray, table: List[tuple], scale: float, line_px: float,
          mat_rgb, d_ref: float, box_pieces: Optional[set] = None) -> List[dict]:
    """Strokes in canvas px: [{pts, width, rgb, kind}]. ``scale`` = render px per canvas px; ``mat_rgb(segment
    index) -> rgb`` gives the fill colour the line outlines (the line is a darker shade of it)."""
    from scipy import ndimage as ndi
    from .draw_line import line_colour
    H, W = seg.shape
    depth = R["depth"]
    nseg = len(table)
    # per piece: owner, seam rule, material
    piece_owner: Dict[int, str] = {}
    piece_seam: Dict[int, str] = {}
    piece_mat: Dict[int, str] = {}
    piece_box: Dict[int, bool] = {}
    keys: Dict[tuple, int] = {}
    for k, pr in enumerate(prims):
        key = (pr.get("owner", ""), pr.get("piece") or pr.get("part") or f"p{k}")
        pi = keys.setdefault(key, len(keys))
        piece_owner[pi] = pr.get("owner", "")
        piece_seam[pi] = pr.get("seam", "mat")
        piece_mat.setdefault(pi, pr.get("mat", ""))
        piece_box[pi] = piece_box.get(pi, False) or pr["t"] in ("box",)
    seg_piece = np.array([t[0] for t in table] + [-1], np.int64)
    seg_sub = np.array([t[1] for t in table] + [0], np.int64)
    owners = sorted(set(piece_owner.values()))
    oidx = {o: i for i, o in enumerate(owners)}
    seg_owner = np.array([oidx[piece_owner[p]] for p, _ in table] + [-1], np.int64)
    seg_seam = np.array([{"mat": 0, "always": 1, "depth": 2}.get(piece_seam[p], 0) for p, _ in table] + [0], np.int64)
    mats = sorted(set(piece_mat.values()))
    midx = {m: i for i, m in enumerate(mats)}
    seg_mat = np.array([midx[piece_mat[p]] for p, _ in table] + [-1], np.int64)
    objs = ndi.find_objects(seg + 1)
    out: List[dict] = []
    for si in range(nseg):
        sl = objs[si] if si < len(objs) else None
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - 2), min(H, sl[0].stop + 2)
        x0, x1 = max(0, sl[1].start - 2), min(W, sl[1].stop + 2)
        m = (seg[y0:y1, x0:x1] == si).astype(np.uint8)
        if m.sum() < 4:
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        blur = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)
        is_box = piece_box.get(int(seg_piece[si]), False)
        for c in cs:
            if len(c) < 6:
                continue
            q = c[:, 0, :].astype(np.float64)
            n = len(q)
            # outward normal: down the blurred mask's gradient
            qi = q.astype(int)
            gy = (blur[np.clip(qi[:, 1] + 1, 0, y1 - y0 - 1), qi[:, 0]] - blur[np.clip(qi[:, 1] - 1, 0, y1 - y0 - 1), qi[:, 0]])
            gx = (blur[qi[:, 1], np.clip(qi[:, 0] + 1, 0, x1 - x0 - 1)] - blur[qi[:, 1], np.clip(qi[:, 0] - 1, 0, x1 - x0 - 1)])
            g = np.stack([-gx, -gy], 1)
            gn = np.linalg.norm(g, axis=1, keepdims=True)
            nrm = np.where(gn > 1e-6, g / np.maximum(gn, 1e-6), 0)
            P = q + [x0, y0]
            out_pt = np.clip(np.round(P + nrm * 2.0), [0, 0], [W - 1, H - 1]).astype(int)
            in_pt = np.clip(np.round(P - nrm * 0.6), [0, 0], [W - 1, H - 1]).astype(int)
            nb = seg[out_pt[:, 1], out_pt[:, 0]]
            nb = np.where(nb == si, -2, nb)                 # still inside: no neighbour found (thin spot)
            d0 = depth[in_pt[:, 1], in_pt[:, 0]].astype(np.float64)
            d0 = np.where(np.isfinite(d0), d0, depth[P[:, 1].astype(int).clip(0, H - 1), P[:, 0].astype(int).clip(0, W - 1)])
            dn = depth[out_pt[:, 1], out_pt[:, 0]].astype(np.float64)
            dn = np.where(np.isfinite(dn), dn, 1e9)
            nbi = np.where(nb >= 0, nb, nseg)
            same_owner = seg_owner[nbi] == seg_owner[si]
            same_piece = seg_piece[nbi] == seg_piece[si]
            eps = np.maximum(0.02, 0.006 * d0)
            eps_h = np.maximum(0.006, 0.002 * d0)
            behind = dn > d0 + eps
            front = dn < d0 - eps
            w = np.zeros(n)
            w = np.where(nb == -1, W_SIL, w)
            other = (nb >= 0) & ~same_owner
            w = np.where(other & behind, W_SIL, w)
            w = np.where(other & ~behind & ~front & (nbi < si), W_CONTACT, w)
            own = (nb >= 0) & same_owner & ~same_piece
            seam_rule = np.maximum(seg_seam[si], seg_seam[nbi])
            diff_mat = seg_mat[nbi] != seg_mat[si]
            w = np.where(own & behind, W_OCC, w)
            level = own & ~behind & ~front
            w = np.where(level & (seam_rule == 2) & (dn > d0 + eps_h), W_SEAM, w)
            w = np.where(level & (seam_rule == 2) & (np.abs(dn - d0) <= eps_h) & diff_mat & (nbi < si), W_SEAM, w)
            w = np.where(level & (seam_rule == 1) & (nbi < si), W_SEAM, w)
            w = np.where(level & (seam_rule == 0) & diff_mat & (nbi < si), W_SEAM, w)
            crease = (nb >= 0) & same_piece & (seg_sub[nbi] != seg_sub[si]) & (nbi < si)
            w = np.where(crease, 0.5 if is_box else W_CREASE, w)
            if not w.any():
                continue
            # steady weights along the contour (no flicker from single pixels)
            from scipy.ndimage import median_filter
            w = median_filter(w, size=5, mode="wrap")
            if not w.any():
                continue
            Ps = _smooth_closed(P, 1.1 if not is_box else 0.0) + 0.5
            if is_box:
                Ps = P + 0.5
            fd = np.clip((d_ref / np.maximum(d0, 1e-3)) ** 0.55, 0.5, 1.45)
            wid = _circ_smooth(w * fd, 2.0)
            rgb = line_colour(mat_rgb(si))
            closed_all = bool((w > 0).all())
            for run in _runs(w > 0, True):
                if len(run) < 3:
                    continue
                pts = Ps[run] / scale
                if closed_all:
                    pts = np.vstack([pts, pts[:1]])
                    ww = np.concatenate([wid[run], wid[run][:1]])
                else:
                    ww = wid[run]
                seglen = np.hypot(*np.diff(pts, axis=0).T)
                s = np.concatenate([[0.0], np.cumsum(seglen)])
                L = float(s[-1])
                base = float(np.median(ww)) * line_px
                if L < max(2.5 * base, 3.0):
                    continue
                if is_box and len(pts) > 4:
                    keep = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), 0.7, closed_all).reshape(-1, 2)
                    if len(keep) >= 2:
                        idx = [int(np.argmin(np.hypot(*(pts - k_).T))) for k_ in keep]
                        idx = sorted(set(idx))
                        pts, ww, s = pts[idx], ww[idx], s[idx]
                        if closed_all:
                            pts, ww = np.vstack([pts, pts[:1]]), np.concatenate([ww, ww[:1]])
                            s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                        L = float(s[-1])
                widths = np.maximum(0.8, ww * line_px * _taper(s, L, base, closed_all))
                out.append({"pts": pts.astype(np.float32), "width": widths.astype(np.float32),
                            "rgb": tuple(int(v) for v in rgb), "w": float(np.median(ww))})
    return out


# ============================================================================================ shadows

class _Ortho:
    def __init__(self, Lw: np.ndarray, pts: np.ndarray, res: int):
        self.e3 = unit(Lw)
        a = np.array([0, 1.0, 0]) if abs(self.e3[1]) < 0.9 else np.array([1.0, 0, 0])
        self.e1 = unit(np.cross(a, self.e3))
        self.e2 = np.cross(self.e3, self.e1)
        u, v = pts @ self.e1, pts @ self.e2
        self.u0, self.v0 = float(u.min()), float(v.min())
        span = max(float(u.max() - self.u0), float(v.max() - self.v0), 1e-3)
        self.ps = span / (res - 2)
        self.res = res
        self.far = float((pts @ self.e3).max()) + 1.0

    def uv(self, P):
        return (P @ self.e1 - self.u0) / self.ps, (P @ self.e2 - self.v0) / self.ps


def shadow_map(prims: List[dict], Lw: np.ndarray, res: int = 1024) -> Optional[Tuple[_Ortho, np.ndarray]]:
    """Distance from a plane on the light's side to the first surface, seen straight along the light."""
    if not prims:
        return None
    pts = np.vstack([corners(pr) for pr in prims])
    oc = _Ortho(Lw, pts, res)
    dmap = np.full((res, res), np.inf, np.float32)
    D1 = -oc.e3
    for pr in prims:
        cp = corners(pr)
        u, v = oc.uv(cp)
        x0, x1 = max(0, int(u.min()) - 1), min(res, int(u.max()) + 2)
        y0, y1 = max(0, int(v.min()) - 1), min(res, int(v.max()) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        xs, ys = xs.ravel() + 0.5, ys.ravel() + 0.5
        if pr["t"] == "rcone":
            ua, va = oc.uv(np.array([pr["a"], pr["b"]]))
            cand = _near_segment(xs, ys, np.array([ua[0], va[0]]), np.array([ua[1], va[1]]),
                                 max(pr["ra"], pr["rb"]) / oc.ps + 1.5)
        else:
            cand = np.ones(len(xs), bool)
        if not cand.any():
            continue
        uu = xs[cand] * oc.ps + oc.u0
        vv = ys[cand] * oc.ps + oc.v0
        O = uu[:, None] * oc.e1 + vv[:, None] * oc.e2 + oc.far * oc.e3
        t, _, _ = intersect(pr, O, np.broadcast_to(D1, O.shape))
        tt = np.full(len(xs), np.inf, np.float32)
        tt[cand] = t
        sub = dmap[y0:y1, x0:x1]
        np.minimum(sub, tt.reshape(sub.shape), out=sub)
    return oc, dmap


def in_shadow(sm, P: np.ndarray, N: np.ndarray, bias: float) -> np.ndarray:
    if sm is None:
        return np.zeros(len(P), bool)
    oc, dmap = sm
    Q = P + N * bias * 0.6
    u, v = oc.uv(Q)
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (ui >= 0) & (vi >= 0) & (ui < oc.res) & (vi < oc.res)
    t = oc.far - Q @ oc.e3
    out = np.zeros(len(P), bool)
    best = dmap[np.clip(vi, 0, oc.res - 1), np.clip(ui, 0, oc.res - 1)]
    out[ok] = best[ok] < t[ok] - (bias + 1.5 * oc.ps)
    return out


# ============================================================================================ the toon render

def toon(cam: Camera, prims: List[dict], light_screen, bands: int = 2, figure_owners=(), rings=None,
         shadows: bool = True, R: Optional[dict] = None) -> dict:
    """Maps at the camera's size: ``seg`` / ``table`` (segments for the lines), ``mat`` (material name index per
    pixel, -1 none) with ``mats`` (names), ``tone`` (0 nothing / 1 lit / 2 shadow / 3 deep shadow), ``lam`` (n . l),
    ``hl`` (hair highlight), ``owner`` index per pixel with ``owners``, ``depth``, ``R``."""
    if R is None:
        R = render(cam, prims)
    H, W = R["id"].shape
    ids = R["id"]
    hit = ids >= 0
    seg, table, pidx = segment_maps(R, prims)
    face = R["face"].astype(np.int32)
    # materials: the primitive's, its trim band, the inside of an opening
    mats: List[str] = []
    mi: Dict[str, int] = {}

    def m_of(name):
        if name not in mi:
            mi[name] = len(mats)
            mats.append(name)
        return mi[name]
    base_m = np.array([m_of(pr.get("mat", "?")) for pr in prims] + [-1], np.int32)
    trim_m = np.array([m_of(pr["trim_mat"]) if pr.get("trim_mat") else -1 for pr in prims] + [-1], np.int32)
    in_m = np.array([m_of(pr.get("mat", "?") + "/in") if pr["t"] in ("ell", "shell") else -1 for pr in prims] + [-1],
                    np.int32)
    mat = base_m[ids]
    mat = np.where((face & TRIM) > 0, np.where(trim_m[ids] >= 0, trim_m[ids], mat), mat)
    mat = np.where((face & INSIDE) > 0, np.where(in_m[ids] >= 0, in_m[ids], mat), mat)
    mat = np.where(hit, mat, -1)
    owners = sorted({pr.get("owner", "") for pr in prims})
    oi = {o: i for i, o in enumerate(owners)}
    own = np.array([oi[pr.get("owner", "")] for pr in prims] + [-1], np.int32)[ids]
    # light
    Lw = light_world(cam, light_screen)
    ns = R["normal"]
    nw = ns[..., 0:1] * cam.right + (-ns[..., 1:2]) * cam.up + (-ns[..., 2:3]) * cam.fwd
    lam = np.zeros((H, W), np.float32)
    lam[hit] = (nw[hit] @ Lw).astype(np.float32)
    ys, xs = np.nonzero(hit)
    D = cam.rays(xs + 0.5, ys + 0.5)
    Pw = cam.C + R["depth"][ys, xs][:, None].astype(np.float64) * D
    Nw = nw[ys, xs].astype(np.float64)
    cast = np.zeros((H, W), bool)
    if shadows:
        figs = [o for o in figure_owners if o in oi]
        only_one = len(figs) == 1 and all(pr.get("owner") == figs[0] for pr in prims)
        glob = shadow_map(prims, Lw, 900 if only_one else 1400)
        sh = in_shadow(glob, Pw, Nw, 0.008 if only_one else 0.03)
        own_px = own[ys, xs]
        for o in ([] if only_one else figs):
            fp = [pr for pr in prims if pr.get("owner") == o]
            sm = shadow_map(fp, Lw, 800)
            m = own_px == oi[o]
            if m.any():
                sh[m] |= in_shadow(sm, Pw[m], Nw[m], 0.008)
        cast[ys, xs] = sh
    tone = np.zeros((H, W), np.uint8)
    # skin keeps most of its light (an anime face is shaded sparingly)
    skin_m = np.array([str(pr.get("mat")).split(":")[-1] == "skin" for pr in prims] + [False])[ids]
    t1 = np.where(skin_m, -0.12, 0.08)
    lit = lam > t1
    tone[hit] = 1
    tone[hit & (~lit | cast)] = 2
    if bands >= 3:
        tone[hit & (lam < t1 - 0.45)] = 3
    # hair highlight: a ring round the skull on the lit side, zig-zag edged
    hl = np.zeros((H, W), bool)
    if rings:
        hair_m = np.array([str(pr.get("mat")).split(":")[-1] == "hair" for pr in prims] + [False])[ids]
        for o, (c, Hd, h) in rings.items():
            if o not in oi:
                continue
            m = hair_m & (own == oi[o]) & (tone == 1)
            if not m.any():
                continue
            yy, xx = np.nonzero(m)
            Dm = cam.rays(xx + 0.5, yy + 0.5)
            Pm = cam.C + R["depth"][yy, xx][:, None].astype(np.float64) * Dm
            q = (Pm - c) @ Hd / (0.45 * h)
            ph = np.arctan2(q[:, 0], q[:, 2])
            band = 0.42 + 0.05 * np.sin(ph * 14.0) + 0.03 * np.sin(ph * 31.0)
            sel = (q[:, 1] > band - 0.11) & (q[:, 1] < band + 0.03) & (np.linalg.norm(q[:, [0, 2]], axis=1) > 0.6)
            hl[yy[sel], xx[sel]] = True
    return {"R": R, "seg": seg, "table": table, "mat": mat, "mats": mats, "tone": tone, "lam": lam, "hl": hl,
            "own": own, "owners": owners, "cast": cast, "Lw": Lw}
