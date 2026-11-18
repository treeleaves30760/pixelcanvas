"""Line-art extraction for the ink stage: ridges of darkness -> linked, smooth, one-colour pen strokes.

Pipeline (all vectorised except the graph walk):
1. ridge strength: multi-scale Hessian of Lab lightness (dark valleys), scale-normalised, with step edges
   discounted (an edge has a strong gradient, a line has none at its centre);
2. hysteresis: strong ridge pixels seed lines, weaker connected ridge pixels extend them, so faint parts of a real
   line survive while isolated texture does not;
3. centre lines (skeleton), traced into a graph; at junctions the most collinear branches are joined into one
   continuous path; free ends close to another line or end in a compatible direction are bridged;
4. pruning: spurs and isolated pieces shorter than a few line widths are dropped (dots belong to flats / detail);
5. width per point = full width at half the local darkness peak; colour per piece = median ink colour, a stroke
   is split where the ink colour changes;
6. smoothing: arc-length resampling and Gaussian smoothing of the path (curves, not pixel stairs).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize

from .trace import to_lab

__all__ = ["ridge_strength", "extract_lines", "LinePiece"]


@dataclass
class LinePiece:
    pts: np.ndarray        # [N,2] float (x, y), smoothed
    width: np.ndarray      # [N] px
    color: tuple           # one ink colour
    free: tuple            # (start is a free end, end is a free end)


def ridge_strength(L: np.ndarray, sigmas=(0.8, 1.2, 1.8, 2.6), edge_k: float = 0.7):
    """Dark-line response of a lightness image (Lab L, 0..100). Returns (strength, best sigma)."""
    best = np.zeros_like(L, dtype=np.float32)
    bsig = np.zeros_like(L, dtype=np.float32)
    for s in sigmas:
        g = cv2.GaussianBlur(L, (0, 0), s)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3) / 1.0
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3) / 1.0
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4.0
        tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        l1 = 0.5 * (dxx + dyy + tmp)          # largest eigenvalue: positive across a dark valley
        l2 = 0.5 * (dxx + dyy - tmp)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        grad = np.sqrt(gx * gx + gy * gy)
        r = s * s * np.maximum(l1, 0) - np.maximum(l2, 0) * s * s * 0.5 - edge_k * s * grad
        r = np.maximum(r, 0)
        sel = r > best
        best[sel] = r[sel]
        bsig[sel] = s
    return best, bsig


def _half_max_width(dark, sk_pts, radius=6, within=None):
    """Full width at half maximum of the darkness profile around skeleton points (via a local half-max mask)."""
    peak = cv2.dilate(dark, np.ones((3, 3), np.uint8))
    hm = (dark >= 0.5 * peak) & (dark > 1.0)
    if within is not None:
        hm &= within
    dt = ndi.distance_transform_edt(hm)
    ys, xs = sk_pts
    w_hm = np.maximum(2.0 * dt[ys, xs] - 1.0, 1.0)
    # the half-max mask counts whole pixels (a 2 px line measures 1).  Sub-pixel width: the ink across the line
    # divided by its peak darkness, summed along the normal on each side only while the profile keeps falling
    # (so the next strand of a hatching is not counted)
    g = cv2.GaussianBlur(dark, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.5)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.5)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.5)
    # the normal of a line is the direction of strongest change
    th = 0.5 * np.arctan2(2 * jxy[ys, xs], jxx[ys, xs] - jyy[ys, xs])
    nx, ny = np.cos(th), np.sin(th)
    cx, cy = xs + 0.5, ys + 0.5
    H, W = dark.shape

    def samp(t):
        x = np.clip(cx + t * nx - 0.5, 0, W - 1.001)
        y = np.clip(cy + t * ny - 0.5, 0, H - 1.001)
        x0, y0 = x.astype(np.int64), y.astype(np.int64)
        fx, fy = x - x0, y - y0
        return (dark[y0, x0] * (1 - fx) * (1 - fy) + dark[y0, x0 + 1] * fx * (1 - fy)
                + dark[y0 + 1, x0] * (1 - fx) * fy + dark[y0 + 1, x0 + 1] * fx * fy)

    c0 = samp(0.0)
    pk = np.maximum(c0, 1e-3)
    tot = c0 * 0.5
    for sgn in (-1.0, 1.0):
        prev = c0.copy()
        alive = np.ones(len(xs), bool)
        for t in np.arange(0.5, 4.01, 0.5):
            v = samp(sgn * t)
            alive &= (v <= prev + 0.05 * pk) & (v > 0.08 * pk)
            tot += np.where(alive, v * 0.5, 0.0)
            prev = np.where(alive, v, prev)
    w_ink = tot / pk
    return np.clip(w_ink, w_hm, w_hm + 1.5)


_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _graph(sk):
    ys, xs = np.nonzero(sk)
    H, W = sk.shape
    idx = -np.ones((H + 2, W + 2), np.int64)
    idx[ys + 1, xs + 1] = np.arange(len(ys))
    nb = np.stack([idx[ys + 1 + dy, xs + 1 + dx] for dy, dx in _OFFS], 1)
    return ys, xs, nb


def _trace_edges(sk):
    """Skeleton -> list of pixel paths between nodes (degree != 2) and closed loops. Returns ys, xs, paths, deg."""
    ys, xs, nb = _graph(sk)
    cnt = (nb >= 0).sum(1)
    # crossing number: 0->1 transitions around the 8-neighbourhood (N, NE, E, SE, S, SW, W, NW); staircase corners
    # of an 8-connected skeleton have 3 neighbours but cross only twice, so they are not junctions
    ring = [1, 2, 4, 7, 6, 5, 3, 0]  # indices into _OFFS in circular order
    occ = (nb[:, ring] >= 0).astype(np.int8)
    cross = ((occ == 0) & (np.roll(occ, -1, 1) == 1)).sum(1)
    deg = np.where(cnt <= 1, cnt, np.where(cross >= 3, 3, 2))
    nbl = [[int(v) for v in row if v >= 0] for row in nb]
    is_node = deg != 2
    seen_edge = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        prev, cur = a, b
        while not is_node[cur]:
            nxt = [v for v in nbl[cur] if v != prev and (min(cur, v), max(cur, v)) not in seen_edge]
            if not nxt:
                break
            v = nxt[0]
            seen_edge.add((min(cur, v), max(cur, v)))
            path.append(v)
            prev, cur = cur, v
            if cur == path[0]:
                break
        return path

    for a in np.flatnonzero(is_node):
        a = int(a)
        for b in nbl[a]:
            e = (min(a, b), max(a, b))
            if e in seen_edge:
                continue
            seen_edge.add(e)
            paths.append(walk(a, b))
    used = np.zeros(len(ys), bool)
    for p in paths:
        used[p] = True
    for a in np.flatnonzero(~used & ~is_node):
        a = int(a)
        if used[a]:
            continue
        b = nbl[a][0]
        seen_edge.add((min(a, b), max(a, b)))
        p = walk(a, b)
        used[p] = True
        paths.append(p)
    return ys, xs, paths, deg


def _dir(pts, at_start, n=6):
    """Unit direction pointing out of the path at one end."""
    k = min(n, len(pts) - 1)
    if k < 1:
        return np.zeros(2)
    d = pts[0] - pts[k] if at_start else pts[-1] - pts[-1 - k]
    return d / (np.hypot(*d) + 1e-9)


def _join_at_junctions(paths_xy, node_ids, max_turn_cos=0.5):
    """Join paths that meet at the same junction pixel cluster when they continue each other (most collinear
    pairs first). paths_xy: list of [N,2] arrays; node_ids: list of (start node or -1, end node or -1)."""
    lens = [float(np.sqrt((np.diff(p, axis=0) ** 2).sum(1)).sum()) for p in paths_xy]
    ends = {}
    for i, (a, b) in enumerate(node_ids):
        if a >= 0:
            ends.setdefault(a, []).append((i, True))
        if b >= 0:
            ends.setdefault(b, []).append((i, False))
    pairs = []
    for node, lst in ends.items():
        if len(lst) < 2 or len(lst) > 40:
            continue  # a tangle (dense hatching): no reliable continuation to pick
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                (i, si), (j, sj) = lst[x], lst[y]
                if i == j:
                    continue
                di = _dir(paths_xy[i], si)
                dj = _dir(paths_xy[j], sj)
                c = float(-(di @ dj))  # opposite directions out of the node = straight continuation
                short = min(lens[i], lens[j]) < 8.0
                if c >= max_turn_cos or (short and c >= -0.3):
                    # a short link between two junctions is carried on by whichever line suits it best
                    pairs.append((c + (0.0 if c >= max_turn_cos else -2.0), node, i, si, j, sj))
    pairs.sort(key=lambda t: -t[0])
    used_end = set()
    parent = list(range(len(paths_xy)))
    links = []
    for c, node, i, si, j, sj in pairs:
        if (i, si) in used_end or (j, sj) in used_end:
            continue
        used_end.add((i, si))
        used_end.add((j, sj))
        links.append((i, si, j, sj))
    # assemble chains
    adj = {}
    for i, si, j, sj in links:
        adj[(i, si)] = (j, sj)
        adj[(j, sj)] = (i, si)
    done = set()
    chains = []
    for start in range(len(paths_xy)):
        if start in done:
            continue
        # walk to one end of the chain
        cur, cur_from_start = start, True
        # go backwards from the start end of `start`
        seq = []
        i, s = start, True
        visited = {start}
        while (i, s) in adj:
            j, sj = adj[(i, s)]
            if j in visited:
                break
            visited.add(j)
            i, s = j, not sj
        # now (i, s): chain begins at path i, entering from its end `s`
        first = i
        orient_start = s  # True: path i's start is the chain's free start
        i, s = first, orient_start
        visited = set()
        while True:
            visited.add(i)
            seq.append((i, s))
            out_end = (i, not s)
            if out_end not in adj:
                break
            j, sj = adj[out_end]
            if j in visited:
                break
            i, s = j, sj
        for i, _ in seq:
            done.add(i)
        chains.append(seq)
    return chains


def extract_lines(ref, mask=None, *, detail=0.5, min_len_widths=4.0, color_split_de=20.0, dark_min=10.0,
                  dark_rel=0.2, stats=None):
    """Line pieces of the reference inside ``mask``: list of LinePiece, plus the typical line width."""
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    m = np.ones((H, W), bool) if mask is None else np.asarray(mask) > 0.5 if np.asarray(mask).dtype != bool \
        else np.asarray(mask)
    lab = to_lab(ref)
    L = lab[..., 0]
    # darkness relative to the local surroundings: closing (the surroundings with the line removed) minus image
    dark = np.zeros_like(L)
    clo_max = np.zeros_like(L)
    for r in (1, 2, 4, 6):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        clo = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k)
        dark = np.maximum(dark, clo - L)
        clo_max = np.maximum(clo_max, clo)
    dark[~m] = 0
    v = dark[dark > 6]
    if v.size < 50:
        return [], 1.0
    c = float(np.clip(np.percentile(v, 75), 14.0, 45.0))
    # hysteresis: clear ink seeds a line, fainter connected ink extends it; a seed must also be clearly darker than
    # its own surroundings (not a shallow valley in the shading)
    hi = c * (0.8 - 0.4 * detail)
    lo = c * (0.4 - 0.22 * detail)
    seeds = (dark >= hi) & (dark > dark_rel * clo_max) & (dark > dark_min)
    lmask = apply_hysteresis_threshold(np.where(seeds, hi + 1, dark), lo, hi + 0.5) & m
    # keep only the valleys: the pale gaps between closely spaced lines curve the other way, so dense strands stay
    # separate lines instead of merging into one blob with one centre line
    g = cv2.GaussianBlur(L, (0, 0), 0.8)
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3) / 4.0
    l1 = 0.5 * (dxx + dyy + np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2))
    thick = cv2.erode(lmask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    lmask &= (l1 > 0) | thick   # wide strokes keep their full body
    if stats is None:
        stats = {}
    sk = skeletonize(lmask)
    stats["sk_px"] = int(sk.sum())
    if not sk.any():
        return [], 1.0
    ys, xs, paths, deg = _trace_edges(sk)
    sk_w = _half_max_width(dark, (ys, xs), within=lmask)
    # ink colour: darkest pixel of the 3x3 neighbourhood
    best = np.full(len(ys), np.inf)
    by, bx = ys.copy(), xs.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = np.clip(ys + dy, 0, H - 1)
            xx = np.clip(xs + dx, 0, W - 1)
            v = L[yy, xx] + 0.4 * (abs(dx) + abs(dy))
            sel = v < best
            best[sel] = v[sel]
            by[sel], bx[sel] = yy[sel], xx[sel]
    ink = ref[by, bx].astype(np.float32)
    ink_lab = lab[by, bx]
    line_w = float(np.median(sk_w))
    # node clusters: adjacent node pixels form one junction
    node = deg != 2
    node_map = np.zeros((H, W), np.int32)
    node_map[ys[node], xs[node]] = 1
    ncl, cl = cv2.connectedComponents(node_map.astype(np.uint8), connectivity=8)
    paths_xy, nodes, pix = [], [], []
    for p in paths:
        p = np.asarray(p)
        a = int(cl[ys[p[0]], xs[p[0]]]) - 1 if node[p[0]] and deg[p[0]] > 2 else -1
        b = int(cl[ys[p[-1]], xs[p[-1]]]) - 1 if node[p[-1]] and deg[p[-1]] > 2 else -1
        if a >= 0 and a == b and len(p) <= 4:
            continue  # an edge inside one junction cluster, not a line
        paths_xy.append(np.stack([xs[p] + 0.5, ys[p] + 0.5], 1).astype(np.float64))
        nodes.append((a, b))
        pix.append(p)
    chains = _join_at_junctions(paths_xy, nodes)
    pieces = []
    for ch in chains:
        idx = []
        for i, s in ch:
            p = pix[i] if s else pix[i][::-1]
            if idx and len(p) and p[0] == idx[-1]:
                p = p[1:]
            idx.extend(p.tolist())
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        first_i, first_s = ch[0]
        last_i, last_s = ch[-1]
        free0 = deg[idx[0]] <= 1
        free1 = deg[idx[-1]] <= 1
        pieces.append((idx, free0, free1))
    # prune: short pieces with a free end (spurs, specks)
    out = []
    for idx, free0, free1 in pieces:
        pts = np.stack([xs[idx] + 0.5, ys[idx] + 0.5], 1).astype(np.float64)
        seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
        Ltot = float(seg.sum())
        w = sk_w[idx]
        wmed = float(np.median(w))
        if (free0 or free1) and Ltot < min_len_widths * max(wmed, line_w) + (2 if free0 and free1 else 0):
            stats["spur"] = stats.get("spur", 0) + 1
            stats["spur_px"] = stats.get("spur_px", 0) + len(idx)
            continue
        if free0 and free1 and Ltot < max(6.0, (6.0 - 4.0 * detail) * max(wmed, line_w)):
            stats["speck"] = stats.get("speck", 0) + 1
            continue
        # split where the ink colour changes along the line (one colour per stroke)
        # cut only at sustained changes: mean colour of the 10 samples before vs the 10 after
        cuts = [0]
        n_i = len(idx)
        if n_i >= 24:
            cs = np.vstack([np.zeros((1, 3)), np.cumsum(ink_lab[idx].astype(np.float64), 0)])
            h = 10
            t = np.arange(h, n_i - h)
            before = (cs[t] - cs[t - h]) / h
            after = (cs[t + h] - cs[t]) / h
            jump = np.sqrt(((after - before) ** 2).sum(1))
            last = 0
            for k in np.argsort(-jump):
                if jump[k] < color_split_de:
                    break
                tc = int(t[k])
                if all(abs(tc - c) >= 12 for c in cuts) and n_i - tc >= 12:
                    cuts.append(tc)
            cuts.sort()
        cuts.append(len(idx))
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            sub = idx[c0:c1 + 1] if c1 < len(idx) else idx[c0:c1]
            if len(sub) < 2:
                continue
            P = np.stack([xs[sub] + 0.5, ys[sub] + 0.5], 1).astype(np.float64)
            col = tuple(int(v) for v in np.clip(np.rint(np.median(ink[sub], 0)), 0, 255))
            f0 = free0 and c0 == 0
            f1 = free1 and c1 >= len(idx) - 1
            Ps, Ws = _smooth_path(P, sk_w[sub].astype(np.float64))
            Lp = float(np.sqrt((np.diff(Ps, axis=0) ** 2).sum(1)).sum())
            if 4.0 <= Lp < 4.0 * float(np.mean(Ws)):
                # a short mark (a nose tick, a lash, a dot of hatching): keep it, drawn a little finer
                Ws = np.maximum(Ws * (Lp / (4.0 * float(np.mean(Ws)))), 1.0)
                stats["thinned"] = stats.get("thinned", 0) + 1
            if Lp < 4.0 * max(1.0, float(np.mean(Ws))) - 1e-3:
                stats["short"] = stats.get("short", 0) + 1
                stats["short_px"] = stats.get("short_px", 0) + len(sub)
                continue
            stats["kept_px"] = stats.get("kept_px", 0) + len(sub)
            out.append(LinePiece(Ps.astype(np.float32), Ws.astype(np.float32), col, (bool(f0), bool(f1))))
    return out, line_w


def _smooth_path(P, w, step=1.0, sigma=1.6):
    """Arc-length resample, Gaussian-smooth (ends fixed), thin to ~2 px spacing."""
    d = np.sqrt((np.diff(P, axis=0) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(d)])
    Ltot = s[-1]
    if Ltot < 1e-6:
        return P[:1], w[:1]
    n = max(2, int(np.ceil(Ltot / step)) + 1)
    t = np.linspace(0, Ltot, n)
    x = np.interp(t, s, P[:, 0])
    y = np.interp(t, s, P[:, 1])
    ww = np.interp(t, s, w)
    closed = Ltot > 10 and np.hypot(P[0, 0] - P[-1, 0], P[0, 1] - P[-1, 1]) < 1.5
    mode = "wrap" if closed else "nearest"
    xs = ndi.gaussian_filter1d(x, sigma, mode=mode)
    ys = ndi.gaussian_filter1d(y, sigma, mode=mode)
    ww = ndi.gaussian_filter1d(ww, 2.0, mode=mode)
    if not closed:
        # keep the ends where they are (blend smoothing in over the first / last few samples)
        k = min(4, n // 2)
        if k > 0:
            a = np.linspace(0, 1, k + 1)[:-1]
            xs[:k] = x[:k] * (1 - a) + xs[:k] * a
            ys[:k] = y[:k] * (1 - a) + ys[:k] * a
            xs[-k:] = x[-k:] * a[::-1] + xs[-k:] * (1 - a[::-1])
            ys[-k:] = y[-k:] * a[::-1] + ys[-k:] * (1 - a[::-1])
    sel = np.unique(np.concatenate([np.arange(0, n, 2), [n - 1]]))
    return np.stack([xs[sel], ys[sel]], 1), ww[sel]
