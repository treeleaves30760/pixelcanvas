"""Fine detail as strokes: thin strokes along the structure whose colour changes along the path (ops.Streak).

The fine levels (2 fine, 1 hairline) used to be many one-colour marks: short liner strokes, 1 px runs of one mean
colour.  A painter lays fine detail as strokes that follow the form, the brush picking up and losing colour as it
goes.  ``streak_ops`` does that on the residual of the canvas:

1. pixels still off by more than ``tol`` Delta E are candidates;
2. from seeds among them (on a grid of the stroke width), a path is traced both ways along the isophote direction
   of the reference (the structure tensor's direction, so a stroke runs along an edge, a strand, a ripple), one px
   per step, while the pixels stay candidates and the reference colour does not jump (no stroke crosses an edge);
3. the path is resampled into stations every ``station`` px; each station takes the mean reference colour of its
   stretch of the path (on the reference blurred to the stroke's width), and the stroke's colour is interpolated
   between them;
4. a stroke is kept when it lowers the residual along its path and owns most of its path (strokes of one round do
   not pile on each other); a few rounds take what the first left.

One op holds all strokes of a pass.  On a region (a partial mask): the strokes whose middle lies in it, ``own``.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .ops import Streak, is_partial
from .trace import as_mask, orientation_field, to_lab

__all__ = ["streak_ops", "STREAK_TOL", "STREAK_JOIN"]

STREAK_TOL = 1.5       # Delta E: a pixel off by more than this is a candidate
STEP_GRAD = 12.0       # Delta E per px: a colour step this sharp narrows a stroke that runs beside it
STREAK_JOIN = 7.0      # Delta E between consecutive reference pixels along a path that still continue it
STREAK_HALF = 24       # px: a path runs at most this far each way from its seed
STREAK_STATIONS = 4    # px between colour stations
STREAK_OWN = 0.5       # share of its path a stroke must own to be kept
SPLIT_DE = 3.0         # Delta E: a stretch between two stations whose colour is further off gets a station between
STREAK_MIN_SAMPLES = 3     # a path shorter than this many samples (1 px apart) is not kept
_FIELD_CACHE = {}


def _field(ref):
    key = (ref.shape, int(ref[::7, ::7].astype(np.int64).sum()), int(ref[3::11, 5::11].astype(np.int64).sum()))
    hit = _FIELD_CACHE.get(key)
    if hit is None:
        lab = to_lab(ref)
        tx, ty, _ = orientation_field(ref, sigma=1.5)
        # distance to the nearest sharp colour step: a stroke is at most twice that wide (plus one px), so a wide
        # stroke beside an edge narrows instead of spilling over it
        lb = cv2.GaussianBlur(lab, (0, 0), 0.7)
        g = np.zeros(lab.shape[:2], np.float32)
        for c in range(3):
            g += cv2.Sobel(lb[..., c], cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(lb[..., c], cv2.CV_32F, 0, 1, ksize=3) ** 2
        step = np.sqrt(g) / 8.0 > STEP_GRAD
        dist = cv2.distanceTransform((~step).astype(np.uint8), cv2.DIST_L2, 3)
        hit = (lab, tx, ty, dist)
        _FIELD_CACHE.clear()
        _FIELD_CACHE[key] = hit
    return hit


def _trace(seeds_x, seeds_y, cand, lab, tx, ty, half, join):
    """Trace from every seed both ways.  Returns (xs [S], ys [S] sample centres in path order, path id [S])."""
    H, W = cand.shape
    n = len(seeds_x)
    px0 = seeds_x.astype(np.float32) + 0.5
    py0 = seeds_y.astype(np.float32) + 0.5
    d0x, d0y = tx[seeds_y, seeds_x], ty[seeds_y, seeds_x]
    lab0 = lab[seeds_y, seeds_x]
    steps = {}
    counts = {}
    for sgn in (1.0, -1.0):
        x, y = px0.copy(), py0.copy()
        dx, dy = d0x * sgn, d0y * sgn
        prev = lab0.copy()
        alive = np.ones(n, bool)
        cnt = np.zeros(n, np.int32)
        xs_k, ys_k = [], []
        for _ in range(half):
            if not alive.any():
                break
            nx, ny = x + dx, y + dy
            ix, iy = np.floor(nx).astype(np.int32), np.floor(ny).astype(np.int32)
            inb = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            ixc, iyc = np.clip(ix, 0, W - 1), np.clip(iy, 0, H - 1)
            lab_n = lab[iyc, ixc]
            jump = np.sqrt(((lab_n - prev) ** 2).sum(1))
            ok = alive & inb & cand[iyc, ixc] & (jump < join)
            # follow the field, keeping the heading (the field's sign is arbitrary); smooth turns
            fx, fy = tx[iyc, ixc], ty[iyc, ixc]
            flip = fx * dx + fy * dy < 0
            fx, fy = np.where(flip, -fx, fx), np.where(flip, -fy, fy)
            hx, hy = dx + fx, dy + fy
            nrm = np.sqrt(hx * hx + hy * hy)
            good = nrm > 1e-3
            hx = np.where(good, hx / np.maximum(nrm, 1e-3), dx)
            hy = np.where(good, hy / np.maximum(nrm, 1e-3), dy)
            x, y = np.where(ok, nx, x), np.where(ok, ny, y)
            dx, dy = np.where(ok, hx, dx), np.where(ok, hy, dy)
            prev = np.where(ok[:, None], lab_n, prev)
            cnt += ok
            alive = ok
            xs_k.append(x.copy())
            ys_k.append(y.copy())
        steps[sgn] = (np.array(xs_k, np.float32).reshape(-1, n), np.array(ys_k, np.float32).reshape(-1, n))
        counts[sgn] = cnt
    nf, nb = counts[1.0], counts[-1.0]
    tot = nb + 1 + nf
    pid = np.repeat(np.arange(n), tot)
    start = np.concatenate([[0], np.cumsum(tot)[:-1]])
    j = np.arange(int(tot.sum())) - np.repeat(start, tot)   # index along the path
    nb_r = nb[pid]
    xs = np.empty(len(pid), np.float32)
    ys = np.empty(len(pid), np.float32)
    # backward part: j < nb -> step (nb - 1 - j) of the backward trace
    bk = j < nb_r
    if bk.any():
        k = (nb_r - 1 - j)[bk]
        xs[bk] = steps[-1.0][0][k, pid[bk]]
        ys[bk] = steps[-1.0][1][k, pid[bk]]
    sd = j == nb_r
    xs[sd], ys[sd] = px0[pid[sd]], py0[pid[sd]]
    fw = j > nb_r
    if fw.any():
        k = (j - nb_r - 1)[fw]
        xs[fw] = steps[1.0][0][k, pid[fw]]
        ys[fw] = steps[1.0][1][k, pid[fw]]
    return xs, ys, pid, tot


def _fit(is_st, pid, j, y):
    """Least-squares station colours for piecewise-linear colour along each path.  ``is_st`` [S] marks the station
    samples (each path's first and last sample are stations).  Returns (station colours [K,3], interpolated colour
    per sample [S,3], index of each sample's left station [S])."""
    st_idx = np.flatnonzero(is_st)
    K = len(st_idx)
    k_lo = np.cumsum(is_st) - 1                       # left station of every sample (a station is its own left)
    nxt = np.minimum(k_lo + 1, K - 1)
    same = pid[st_idx[nxt]] == pid
    k_hi = np.where(same & (k_lo + 1 < K), nxt, k_lo)
    j_lo, j_hi = j[st_idx[k_lo]], j[st_idx[k_hi]]
    f = np.where(j_hi > j_lo, (j - j_lo) / np.maximum(j_hi - j_lo, 1), 0.0)
    w0, w1 = 1.0 - f, f
    # normal equations (tridiagonal, one block per path): sum over samples of the hat weights
    diag = np.bincount(k_lo, weights=w0 * w0, minlength=K) + np.bincount(k_hi, weights=w1 * w1, minlength=K)
    off = np.bincount(k_lo, weights=np.where(k_hi != k_lo, w0 * w1, 0.0), minlength=K)   # (k, k+1)
    diag = diag + 1e-6
    rhs = np.stack([np.bincount(k_lo, weights=w0 * y[:, c], minlength=K) +
                    np.bincount(k_hi, weights=w1 * y[:, c], minlength=K) for c in range(3)], 1)
    from scipy.linalg import solve_banded
    ab = np.zeros((3, K))
    ab[0, 1:] = off[:-1]
    ab[1] = diag
    ab[2, :-1] = off[:-1]
    st_col = np.clip(solve_banded((1, 1), ab, rhs), 0.0, 255.0)
    col = st_col[k_lo] * w0[:, None] + st_col[k_hi] * w1[:, None]
    return st_col, col, k_lo


SMOOTH_STD = 2.0       # L*: where the reference's 1-4 px detail (local rms) is below this, it counts as smooth


def smooth_mask(lab):
    """Where the reference is smooth: the local rms of its 1-4 px lightness detail is below SMOOTH_STD."""
    L = np.ascontiguousarray(lab[..., 0]).astype(np.float32)
    d = L - cv2.GaussianBlur(L, (0, 0), 2.0)
    return np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 4.0)) < SMOOTH_STD


def streak_ops(ref, mask=None, canvas=None, *, width=2.0, tol=STREAK_TOL, join=STREAK_JOIN, half=STREAK_HALF,
               station=STREAK_STATIONS, rounds=5, own=STREAK_OWN, smooth_tol=None, stats=None):
    """Thin strokes of width ``width`` px along the structure over the residual of ``canvas`` (module doc).
    ``smooth_tol``: the Delta E tolerance where the reference is smooth (a sky, a gradient: little 1-4 px detail,
    ``smooth_mask``), so its faint mottle is painted and no flat patch of an earlier, coarser mark stays."""
    t0 = time.perf_counter()
    ref = np.ascontiguousarray(np.asarray(ref)[..., :3])
    H, W = ref.shape[:2]
    partial = is_partial(mask)
    canvas = np.full_like(ref, 255) if canvas is None else np.asarray(canvas)[..., :3]
    lab, tx, ty, edge_dist = _field(ref)
    m = np.ones((H, W), bool) if partial else as_mask(mask, ref.shape)
    de = np.sqrt(((lab - to_lab(canvas)) ** 2).sum(-1))
    if smooth_tol is not None and smooth_tol < tol:
        need = m & (de > np.where(smooth_mask(lab), smooth_tol, tol))
    else:
        need = m & (de > tol)
    # the colour a stroke carries: the reference along its middle (the gain below counts what its width covers)
    src = ref.astype(np.float32)
    lab_can = to_lab(canvas)
    side = float(np.clip(width / 2.0 + 0.5 - 1.0, 0.0, 1.0))     # coverage of the neighbours across the stroke
    covered = np.zeros((H, W), bool)
    stride = max(1, int(round(width)))
    rad = max(0, int(np.floor(width / 2.0)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)) if rad else None
    out_pts, out_n, out_c, out_w = [], [], [], []
    for rnd in range(rounds):
        cand = need & ~covered
        sy, sx = np.nonzero(cand)
        if not len(sy):
            break
        if rnd == 0 and stride > 1:
            g = ((sx % stride) == 0) & ((sy % stride) == 0)
            sx, sy = sx[g], sy[g]
        if not len(sy):
            continue
        # strongest residual first (the ownership below keeps the better of two overlapping strokes)
        # the first round runs over off pixels only; later rounds let a stroke pass over pixels that are already
        # right (its gain along the whole path decides), so scattered off pixels join into strokes
        free = cand if rnd == 0 else m
        xs, ys, pid, tot = _trace(sx, sy, free, lab, tx, ty, half if rnd == 0 else max(3, half // 3), join)
        ix = np.clip(np.floor(xs).astype(np.int64), 0, W - 1)
        iy = np.clip(np.floor(ys).astype(np.int64), 0, H - 1)
        start = np.concatenate([[0], np.cumsum(tot)[:-1]])
        j = np.arange(len(pid)) - start[pid]
        # stations every `station` samples and at the path's last sample; colours fitted (least squares) so the
        # interpolated colour follows the reference along the path; a stretch still off gets a station between
        L = tot[pid]
        is_st = (j % station == 0) | (j == L - 1)
        y_col = src[iy, ix].astype(np.float64)
        lab_ref = lab[iy, ix]
        for it in range(3):
            st_col, col, k_lo = _fit(is_st, pid, j, y_col)
            if it == 2 or station <= 2:
                break
            lab_c = to_lab(np.clip(col, 0, 255).astype(np.uint8)[None])[0]
            err = np.sqrt(((lab_ref - lab_c) ** 2).sum(1))
            # per segment (between two stations): its worst sample; split segments of >= 4 samples there
            seg_bad = np.zeros(len(st_col), bool)
            np.logical_or.at(seg_bad, k_lo, err > SPLIT_DE)
            st_idx = np.flatnonzero(is_st)
            span = np.diff(np.append(j[st_idx], 0))
            split = seg_bad & (span >= 4) & (pid[st_idx] == pid[np.minimum(st_idx + 1, len(pid) - 1)])
            if not split.any():
                break
            is_st[st_idx[split] + span[split] // 2] = True
        st_idx = np.flatnonzero(is_st)
        st_pid = pid[st_idx]
        nst = np.bincount(st_pid, minlength=len(tot))
        lab_c = to_lab(np.clip(col, 0, 255).astype(np.uint8)[None])[0]
        after = np.sqrt(((lab_ref - lab_c) ** 2).sum(1))
        gain = np.bincount(pid, weights=de[iy, ix] - after, minlength=len(tot))
        if side > 0.01:
            # the pixels either side of the path get `side` of the stroke's colour: count what that does to them
            gx = np.gradient(xs) if len(xs) > 1 else np.zeros_like(xs)
            gy = np.gradient(ys) if len(ys) > 1 else np.zeros_like(ys)
            nrm = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-3)
            nx, ny = -gy / nrm, gx / nrm
            for sgn in (1.0, -1.0):
                qx = np.clip(np.floor(xs + sgn * nx).astype(np.int64), 0, W - 1)
                qy = np.clip(np.floor(ys + sgn * ny).astype(np.int64), 0, H - 1)
                mixed = lab_can[qy, qx] * (1.0 - side) + lab_c * side
                d_after = np.sqrt(((lab[qy, qx] - mixed) ** 2).sum(1))
                gain += np.bincount(pid, weights=de[qy, qx] - d_after, minlength=len(tot))
        keep = (tot >= STREAK_MIN_SAMPLES) & (gain > 0)
        if not keep.any():
            continue
        # ownership: every path writes its id over its pixels, weakest first; a path keeps what it owns
        order = np.argsort(np.where(keep, gain / tot, -np.inf), kind="stable")
        rank = np.empty(len(tot), np.int64)
        rank[order] = np.arange(len(tot))
        flat = iy * W + ix
        sel = keep[pid]
        own_img = np.full(H * W, -1, np.int64)
        srt = np.argsort(rank[pid[sel]], kind="stable")
        own_img[flat[sel][srt]] = pid[sel][srt]
        owned = np.bincount(pid[sel], weights=(own_img[flat[sel]] == pid[sel]), minlength=len(tot))
        keep &= owned >= own * tot
        if not keep.any():
            continue
        if stats is not None:
            stats.setdefault("rounds", []).append((int(cand.sum()), int(len(tot)), int((tot >= 3).sum()), int(keep.sum())))
        ks = keep[st_pid]
        out_pts.append(np.stack([xs[st_idx][ks], ys[st_idx][ks]], 1))
        out_n.append(nst[keep])
        out_c.append(st_col[ks])
        sx_, sy_ = xs[st_idx][ks], ys[st_idx][ks]
        dd = edge_dist[np.clip(sy_.astype(np.int64), 0, H - 1), np.clip(sx_.astype(np.int64), 0, W - 1)]
        # no sharp step anywhere: distanceTransform gives FLT_MAX; cap it first (same widths, no float32 overflow)
        dd = np.minimum(dd, float(width))
        out_w.append(np.clip(2.0 * dd + 1.0, 1.0, float(width)))
        cov = np.zeros(H * W, bool)
        cov[flat[keep[pid]]] = True
        cov = cov.reshape(H, W)
        if kern is not None:
            cov = cv2.dilate(cov.astype(np.uint8), kern) > 0
        covered |= cov
    if not out_n:
        if stats is not None:
            stats.update(marks=0, stations=0, seconds=round(time.perf_counter() - t0, 2))
        return []
    pts = np.concatenate(out_pts)
    n = np.concatenate(out_n).astype(np.int32)
    col = np.concatenate(out_c)
    op = Streak(pts=pts, n=n, colors=col, width=np.concatenate(out_w))
    # a path that curled back on itself can end up shorter than a stroke: leave those out
    from .ops import STREAK_STATION
    ln = op.lengths()
    ok = (ln >= 2.0) & ((op.n - 1) * STREAK_STATION <= ln)
    if not ok.all():
        if not ok.any():
            return []
        ps = ok[np.repeat(np.arange(len(op.n)), op.n)]
        op = Streak(pts=op.pts[ps], n=op.n[ok], colors=op.colors[ps], width=op.width[ps])
    if partial:
        op = _owned(op, mask)
    if stats is not None:
        stats.update(marks=0 if op is None else int(len(op.n)), stations=0 if op is None else int(len(op.pts)),
                     covered=round(float(covered.mean()), 4), seconds=round(time.perf_counter() - t0, 2))
    return [] if op is None else [op]


def _owned(op, mask):
    """The strokes whose middle lies in ``mask``, flagged own."""
    m = np.asarray(mask)
    m = m if m.dtype == bool else m > 0.5
    H, W = m.shape
    mk = np.repeat(np.arange(len(op.n)), op.n)
    cx = np.bincount(mk, weights=op.pts[:, 0].astype(np.float64)) / op.n
    cy = np.bincount(mk, weights=op.pts[:, 1].astype(np.float64)) / op.n
    sel = m[np.clip(cy.astype(int), 0, H - 1), np.clip(cx.astype(int), 0, W - 1)]
    if not sel.any():
        return None
    ps = sel[mk]
    return Streak(pts=op.pts[ps], n=op.n[sel], colors=op.colors[ps], width=op.width[ps], own=True)


STIPPLE_DE = 3.0       # Delta E: a point of the delivered residual below this is left alone
STIPPLE_RING = 2.0     # ... and one not this many times above its 5x5 neighbourhood (a broad error is a stroke's job)


STIPPLE_COHERENCE = 0.4   # with edges=False: a point where the reference's structure is this line-like is left out


def stipple_ops(ref_d, canvas_d, scale, mask=None, *, de_min=STIPPLE_DE, ring=STIPPLE_RING, edges=True,
                stats=None):
    """Stippling: the points a stroke cannot carry (a star, a glint, a speck of sand or foliage) as soft airbrush dots
    of 1.5 delivered pixels (``ops.Spray`` of ``1.5 * scale`` working px, a level-2 mark), one per isolated peak of
    the delivered residual, in the colour that brings that delivered pixel to the reference.

    ref_d, canvas_d: the reference and the canvas at the delivered size (uint8 RGB).  scale: the working scale.
    mask: delivered-size bool region (the dots are then owned by it).  edges=False: leave out the points that lie on
    a line or an edge of the reference (structure-tensor coherence above STIPPLE_COHERENCE): those stay with the
    strokes, and only points in texture (stars, glints, specks) are dotted.  Returns [Spray] in working coordinates."""
    from .ops import Spray
    from .brush import SPRAY_SIGMA
    t0 = time.perf_counter()
    s = max(1, int(scale))
    ref_d = np.asarray(ref_d)[..., :3]
    canvas_d = np.asarray(canvas_d)[..., :3]
    de = np.sqrt(((to_lab(ref_d) - to_lab(canvas_d)) ** 2).sum(-1)).astype(np.float32)
    mx = cv2.dilate(de, np.ones((3, 3), np.uint8))
    pk = (de >= mx) & (de > de_min) & (de > ring * cv2.blur(de, (5, 5)))
    if mask is not None:
        pk &= np.asarray(mask, bool)
    if not edges:
        g = cv2.cvtColor(ref_d, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.5)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.5)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.5)
        coh = ((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum((jxx + jyy) ** 2, 1e-6)
        pk &= coh < STIPPLE_COHERENCE
    ys, xs = np.nonzero(pk)
    if stats is not None:
        stats.update(dots=int(len(xs)), touched=round(float(len(xs)) / pk.size, 4),
                     seconds=round(time.perf_counter() - t0, 2))
    if not len(xs):
        return []
    size = max(3.0, 1.5 * s)
    sig = size * SPRAY_SIGMA
    o = np.arange(s) + 0.5 - s / 2.0
    abar = float(np.exp(-(o[:, None] ** 2 + o[None, :] ** 2) / (2 * sig * sig)).mean())
    c0 = canvas_d[ys, xs].astype(np.float32)
    col = np.clip(c0 + (ref_d[ys, xs].astype(np.float32) - c0) / abar, 0, 255).astype(np.uint8)
    pts = np.stack([xs * s + s / 2.0, ys * s + s / 2.0], 1).astype(np.float32)
    return [Spray(pts=pts, size=size, colors=col, opacity=1.0, own=mask is not None)]


CONTOUR_TOL = 6.0      # Delta E: a line whose mean residual along its skeleton is below this is already drawn
CONTOUR_STEP = 4.0     # px between the colour / width stations along a contour stroke


def contour_ops(ref, mask=None, canvas=None, *, max_width=6.0, min_len=6.0, tol=CONTOUR_TOL, stats=None):
    """Each thin line of the reference (dark ridges and light ridges up to ``max_width`` px, ``trace.line_map`` on the
    picture and on its negative) as ONE stroke along its skeleton: an ``ops.Streak`` with stations every
    CONTOUR_STEP px whose colour follows the line's core and whose width follows the line (twice the distance to its
    edge), so a contour is continuous instead of a row of short pieces.  Only lines still off on ``canvas`` (mean
    Delta E along the skeleton above ``tol``) whose stroke lowers that residual are drawn."""
    from .trace import line_map, skeleton_polylines
    t0 = time.perf_counter()
    ref = np.ascontiguousarray(np.asarray(ref)[..., :3])
    H, W = ref.shape[:2]
    partial = is_partial(mask)
    canvas = np.full_like(ref, 255) if canvas is None else np.asarray(canvas)[..., :3]
    lab = to_lab(ref)
    lab_c = to_lab(canvas)
    de = np.sqrt(((lab - lab_c) ** 2).sum(-1))
    m = np.ones((H, W), bool) if partial else as_mask(mask, ref.shape)
    pts_all, n_all, col_all, w_all = [], [], [], []
    n_lines = 0
    for neg in (False, True):
        lm = line_map(255 - ref if neg else ref, max_width=max_width)
        for pts, wid, _ in skeleton_polylines(lm.mask & m, min_len=min_len):
            if len(pts) < 3:
                continue
            ix = np.clip(pts[:, 0].astype(np.int64), 0, W - 1)
            iy = np.clip(pts[:, 1].astype(np.int64), 0, H - 1)
            if de[iy, ix].mean() < tol:
                continue
            seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
            s = np.concatenate([[0.0], np.cumsum(seg)])
            if s[-1] < 2.0:
                continue
            k = max(2, int(np.ceil(s[-1] / CONTOUR_STEP)) + 1)
            st = np.linspace(0.0, s[-1], k)
            sx = np.interp(st, s, pts[:, 0])
            sy = np.interp(st, s, pts[:, 1])
            # station colour: the reference along the skeleton within half a step (its median), width likewise
            bins = np.clip(np.searchsorted(st, s - CONTOUR_STEP / 2.0), 0, k - 1)
            col = np.zeros((k, 3), np.float64)
            ww = np.zeros(k, np.float64)
            rgb = ref[iy, ix].astype(np.float64)
            for j in range(k):
                sel = np.abs(s - st[j]) <= CONTOUR_STEP / 2.0 + 1e-6
                col[j] = np.median(rgb[sel], 0)
                ww[j] = np.median(wid[sel])
            # gain along the skeleton: the line's colour interpolated against what the canvas has there
            cs = np.stack([np.interp(s, st, col[:, c]) for c in range(3)], 1)
            after = np.sqrt(((lab[iy, ix] - to_lab(np.clip(cs, 0, 255).astype(np.uint8)[None])[0]) ** 2).sum(1))
            if (de[iy, ix] - after).mean() <= 0:
                continue
            pts_all.append(np.stack([sx, sy], 1))
            n_all.append(k)
            col_all.append(col)
            w_all.append(np.clip(ww, 1.0, max_width))
            n_lines += 1
    if stats is not None:
        stats.update(lines=n_lines, stations=int(sum(n_all)), seconds=round(time.perf_counter() - t0, 2))
    if not n_all:
        return []
    op = Streak(pts=np.concatenate(pts_all).astype(np.float32), n=np.asarray(n_all, np.int32),
                colors=np.clip(np.concatenate(col_all), 0, 255).astype(np.uint8),
                width=np.concatenate(w_all).astype(np.float32))
    from .ops import STREAK_STATION
    ln = op.lengths()
    ok = (ln >= 2.0) & ((op.n - 1) * STREAK_STATION <= ln)
    if not ok.all():
        if not ok.any():
            return []
        ps = ok[np.repeat(np.arange(len(op.n)), op.n)]
        op = Streak(pts=op.pts[ps], n=op.n[ok], colors=op.colors[ps], width=op.width[ps])
    if partial:
        op = _owned(op, mask)
    return [] if op is None else [op]
