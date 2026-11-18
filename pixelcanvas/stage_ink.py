"""Line art: trace the reference's dark lines as tapered pen strokes at the reference's own line weight."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .ops import Stroke, is_partial, owned_ops
from .lines import extract_lines
from .trace import as_mask, line_map, skeleton_polylines, smooth_polyline, to_lab

__all__ = ["ink_ops", "PENS"]

# pen option -> (brush preset, width factor)
PENS = {"pen": ("liner", 1.0), "liner": ("liner", 1.0), "pencil": ("pencil", 0.9), "brush": ("round", 1.05),
        "marker": ("marker", 1.0)}


def _hex(c):
    s = c.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _ink_colors(ref, L, xs, ys):
    """Colour of the darkest pixel in the 3x3 neighbourhood of each point (the ink, not its antialiased edge)."""
    H, W = L.shape
    best = np.full(len(xs), np.inf)
    bx, by = xs.copy(), ys.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x = np.clip(xs + dx, 0, W - 1)
            y = np.clip(ys + dy, 0, H - 1)
            v = L[y, x] + 0.5 * (abs(dx) + abs(dy))
            sel = v < best
            best[sel] = v[sel]
            bx[sel], by[sel] = x[sel], y[sel]
    return ref[by, bx]


def dark_masses(ref, lm, m, L=None, min_area=None):
    """Solid areas of line-ink darkness wider than a line (bool [H,W])."""
    if L is None:
        L = to_lab(ref)[..., 0]
    H, W = L.shape
    l_ink = float(to_lab(np.asarray(lm.color, np.float32)[None, None] / 255.0)[0, 0, 0])
    # only areas as dark as the ink itself (black hair, black clothes), not dark colours
    dark = (L <= min(l_ink + 10.0, 30.0)) & m
    r = max(3, int(np.ceil(1.3 * lm.width)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    if min_area is None:
        min_area = max(120, int(H * W * 0.0004))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= min_area
    big[0] = False
    core = big[lab]
    if not core.any():
        return core
    # the mass is the dark area connected to a wide core (keeps the mass's own narrow tips and strands)
    n2, lab2 = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
    keep = np.zeros(n2, bool)
    keep[np.unique(lab2[core])] = True
    keep[0] = False
    grown = keep[lab2]
    # but do not swallow thin lines that merely touch the mass: limit to the core grown by a few px
    reach = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (4 * r + 1, 4 * r + 1))) > 0
    return grown & reach


def _contour_ops(ref, L, contours, w, brush, const, seed, keep_fn, seed_base):
    """Strokes along contours: runs of points where ``keep_fn(points)`` is true, cut into 60-160 px pulls."""
    H, W = L.shape
    ops = []
    for c in contours:
        p = c[:, 0, :].astype(np.float64) + 0.5
        if len(p) < 8:
            continue
        keep = keep_fn(p)
        if keep.all():
            runs = [np.vstack([p, p[:1]])]
            closed = True
        else:
            closed = False
            # rotate so that a run does not wrap around the contour's start
            if keep.any() and not keep[0]:
                pass
            elif keep.any():
                k0 = int(np.flatnonzero(~keep)[0])
                p, keep = np.roll(p, -k0, 0), np.roll(keep, -k0)
            idx = np.flatnonzero(keep)
            if len(idx) == 0:
                continue
            breaks = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate([[0], breaks + 1])
            ends = np.concatenate([breaks + 1, [len(idx)]])
            runs = [p[idx[s0:e0]] for s0, e0 in zip(starts, ends)]
        for run in runs:
            if len(run) < 8:
                continue
            q, _ = smooth_polyline(run, None, window=7, keep_ends=not closed)
            seglen = 110
            n = len(q)
            i0 = 0
            while i0 < n - 2:
                i1 = min(n, i0 + seglen)
                if n - i1 < 25:
                    i1 = n
                piece = q[i0:i1:2] if i1 - i0 > 6 else q[i0:i1]
                if len(piece) >= 2:
                    if const is None:
                        xi = np.clip(piece[:, 0].astype(int), 0, W - 1)
                        yi = np.clip(piece[:, 1].astype(int), 0, H - 1)
                        cols = tuple(int(v) for v in np.median(_ink_colors(ref, L, xi, yi), 0))
                    else:
                        cols = const
                    wid = np.full(len(piece), max(w, 1.0), np.float32)
                    if not closed and len(piece) >= 6:
                        # taper the free ends of an open contour run
                        t = np.linspace(0, 1, len(piece))
                        if i0 == 0:
                            wid *= np.clip(0.35 + 0.65 * t * len(piece) / 4, 0, 1)
                        if i1 == n:
                            wid *= np.clip(0.35 + 0.65 * (1 - t) * len(piece) / 4, 0, 1)
                    wid = np.maximum(wid, 1.0)
                    if float(np.sqrt((np.diff(piece, axis=0) ** 2).sum(1)).sum()) < 4 * float(wid.mean()):
                        if i1 >= n:
                            break
                        i0 = i1 - 3
                        continue
                    ops.append(Stroke(piece.astype(np.float32), wid, cols, brush=brush,
                                      seed=seed + seed_base + len(ops)))
                if i1 >= n:
                    break
                i0 = i1 - 3
    return ops


def _border_keep(H, W, r):
    def f(p):
        return ~((p[:, 0] < r + 1.5) | (p[:, 0] > W - r - 1.5) | (p[:, 1] < r + 1.5) | (p[:, 1] > H - r - 1.5))
    return f


def _mass_outline_ops(ref, L, mass, lm, wscale, brush, const, seed):
    """Outline strokes running just inside the border of each dark mass."""
    H, W = L.shape
    w = max(1.0, lm.width * wscale)
    r = max(1, int(round(w / 2)))
    inner = cv2.erode(mass.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r - 1, 2 * r - 1)))
    contours, _ = cv2.findContours(inner, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return _contour_ops(ref, L, contours, w, brush, const, seed, _border_keep(H, W, r), 50000)


def silhouette(ref, m=None, tol=8.0):
    """Foreground mask of a subject on a uniform background (None when the background is not uniform)."""
    from .trace import delta_e
    H, W = ref.shape[:2]
    ring = np.concatenate([ref[:2].reshape(-1, 3), ref[-2:].reshape(-1, 3), ref[:, :2].reshape(-1, 3),
                           ref[:, -2:].reshape(-1, 3)]).astype(np.float32)
    med = np.median(ring, 0)
    if np.mean(np.abs(ring - med).sum(1) < 18) < 0.6:
        return None
    paper = np.broadcast_to(med.astype(np.uint8), ref.shape)
    bgc = delta_e(cv2.GaussianBlur(ref, (0, 0), 0.8), paper) < tol
    n, lab = cv2.connectedComponents(bgc.astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = border[border > 0]
    bg = np.isin(lab, border)
    fg = ~bg
    if m is not None:
        fg &= m
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= max(200, int(H * W * 0.002))
    big[0] = False
    fg = big[lab]
    if fg.mean() < 0.02 or fg.mean() > 0.97:
        return None
    return fg


def _silhouette_ops(ref, L, fg, lm, wscale, brush, const, seed):
    """The subject's outline against the background, wherever the reference draws a line along it."""
    H, W = L.shape
    w = max(1.0, lm.width * wscale)
    r = max(1, int(round(w / 2)))
    inner = cv2.erode(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r - 1, 2 * r - 1)))
    contours, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    inked = cv2.dilate(lm.strength, np.ones((5, 5), np.uint8)) > 0.35
    bk = _border_keep(H, W, r)

    def keep(p):
        xi = np.clip(p[:, 0].astype(int), 0, W - 1)
        yi = np.clip(p[:, 1].astype(int), 0, H - 1)
        k = inked[yi, xi] & bk(p)
        # close short gaps so the outline is pulled in long strokes
        k = ndi.binary_closing(k, np.ones(7, bool)) & bk(p)
        return k

    return _contour_ops(ref, L, contours, w, brush, const, seed, keep, 70000)


def _piece_strokes(pieces, weight, wf, const, brush, seed, foot=None):
    """Tapered pen strokes for traced line pieces; ends that run into an ink mass (``foot``) reach into it."""
    ops = []
    H, W = foot.shape if foot is not None else (0, 0)
    for i, pc in enumerate(pieces):
        pts = pc.pts
        w = np.maximum(pc.width.astype(np.float64) * weight * wf, 1.0)
        free = list(pc.free)
        if foot is not None and len(pts) >= 2:
            # a line leaving an ink mass starts inside it (no gap at the junction)
            for end in (0, 1):
                p = pts[0] if end == 0 else pts[-1]
                q = pts[min(3, len(pts) - 1)] if end == 0 else pts[max(len(pts) - 4, 0)]
                d = p - q
                nd = float(np.hypot(*d))
                if nd < 1e-6:
                    continue
                d = d / nd
                tip = p + 2.0 * d
                xi, yi = int(np.clip(tip[0], 0, W - 1)), int(np.clip(tip[1], 0, H - 1))
                if foot[yi, xi]:
                    if end == 0:
                        pts = np.vstack([tip[None], pts]).astype(np.float32)
                        w = np.concatenate([[w[0]], w])
                    else:
                        pts = np.vstack([pts, tip[None]]).astype(np.float32)
                        w = np.concatenate([w, [w[-1]]])
                    free[end] = False
        n = len(pts)
        if n >= 4:
            s_ = np.concatenate([[0], np.cumsum(np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)))])
            tl = min(2.5 * float(np.median(w)) + 1, 0.4 * s_[-1])
            f = np.ones(n)
            if free[0] and tl > 0:
                f *= 0.4 + 0.6 * np.clip(s_ / tl, 0, 1)
            if free[1] and tl > 0:
                f *= 0.4 + 0.6 * np.clip((s_[-1] - s_) / tl, 0, 1)
            w = np.maximum(w * f, 1.0)
        lp = float(np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)).sum())
        if lp < 4.0 * float(w.mean()):
            # a short mark stays a line: as fine as its length allows (DESIGN.md 10.1)
            w = np.maximum(w * (lp / (4.0 * float(w.mean()))) * 0.98, 1.0)
            if lp < 4.0 * float(w.mean()):
                continue
        ops.append(Stroke(pts, w.astype(np.float32), const if const is not None else pc.color, brush=brush,
                          seed=seed + i))
    return ops


def _top(op):
    if isinstance(op, Stroke):
        return float(op.pts[:, 1].min())
    return float(min(float(p[:, 1].min()) for p in op.polygons))


# method="auto" chooses ink masses when the line art is thick (at least MASS_SOLID of the line pixels are solid ink)
# or dense (it covers at least MASS_DENSITY of the picture)
MASS_SOLID = 0.35
MASS_DENSITY = 0.06


def mass_share(ref, lines=None):
    """How much of the line art is solid ink wider than a hairline: the share of line pixels that are at least
    80 % ink, and the share of the picture they cover (decides ``method="auto"``)."""
    from .masses import ink_coverage
    lm = lines if lines is not None else line_map(ref)
    cov, _ = ink_coverage(ref)
    solid = (cov >= 0.8) & lm.mask
    return float(solid.sum() / max(int(lm.mask.sum()), 1)), float(lm.mask.mean())


def true_ink(pieces, cov, ground, *, invert=False, min_peak=0.12, resolved=1.8):
    """Thin traced lines at their true ink colour and width.

    A line thinner than a pixel shows as a wide, light band: its darkest pixel is a blend of the ink and the ground
    under it.  With p = the line's peak ink coverage there, the ink itself is ground + (darkest - ground) / p, and
    the width that holds the line's ink (its integrated coverage across) is the traced width x p.  A line whose ink
    would be narrower than the 1 px minimum is drawn 1 px wide in the colour that 1 px of it shows.  ``ground``:
    RGB of the picture with its lines closed away; ``invert``: the pieces were traced on the inverted picture (light
    strands), colours come back inverted."""
    from .lines import LinePiece
    H, W = cov.shape
    peak = cv2.dilate(cov.astype(np.float32), np.ones((3, 3), np.uint8))
    out = []
    for pc in pieces:
        xi = np.clip(pc.pts[:, 0].astype(int), 0, W - 1)
        yi = np.clip(pc.pts[:, 1].astype(int), 0, H - 1)
        c = np.asarray(pc.color, np.float64)
        g = np.median(ground[yi, xi].astype(np.float64), 0)
        p = peak[yi, xi].astype(np.float64)
        pm = float(np.median(p))
        if pm < min_peak or float(np.median(pc.width)) >= resolved:
            # too faint to tell, or a grey line the picture resolves: its own colour and width
            col, w = c, pc.width.astype(np.float64)
        else:
            from .masses import UNMIX
            col = g + (c - g) / float(np.clip(pm, 0.25, 1.0)) ** UNMIX
            w = pc.width.astype(np.float64) * np.clip(p, 0.15, 1.0) ** UNMIX
            f = float(np.median(w))
            if f < 1.0:
                # below a pixel: 1 px of it, as light as the line shows
                col = g + (col - g) * max(f, 0.15)
                w = w / max(f, 0.15)
        col = np.clip(col, 0, 255)
        if invert:
            col = 255.0 - col
        out.append(LinePiece(pc.pts, np.maximum(w, 1.0).astype(np.float32), tuple(int(round(v)) for v in col),
                             pc.free))
    return out


def ink_ops(ref, mask=None, *, detail=0.5, weight=1.0, color="auto", pen="pen", seed=0, lines=None,
            method="strokes", true_thin=False):
    """Pen strokes along the reference's line art inside ``mask``.

    detail 0..1: lower keeps only strong, long lines; higher also traces faint and short ones.
    weight: multiplies the traced line width.  color: "auto" (ink colour sampled along each line) or "#rrggbb".
    pen: pen | pencil | brush | marker.  ``lines`` may pass a precomputed ``trace.LineMap``.
    true_thin (with masses): faint masses at their true ink colour and width (``masses.ink_mass_ops``); with True
    also the leftover hairline strokes (``true_ink``) and thin light strands inside the solid ink as paper-coloured
    strokes on top; "masses" = the faint masses only.
    method: "strokes" (centre lines stroked with a pen), "masses" (the solid ink outlined and filled, one Fill per
    connected ink mass, see ``masses.py``; hairlines too faint for a mass stay pen strokes), or "auto" (masses
    when the line art is dense or thick).
    On a region: the whole picture's lines anchored in the region, each one complete across the region's edge, so
    region calls over a partition give the same line art as one call on the whole picture."""
    if is_partial(mask):
        return owned_ops(ink_ops(ref, None, detail=detail, weight=weight, color=color, pen=pen, seed=seed,
                                 lines=lines, method=method, true_thin=true_thin), mask)
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    detail = float(np.clip(detail, 0.0, 1.0))
    brush, wf = PENS.get(pen, (pen, 1.0))
    lm = lines if lines is not None else line_map(ref, mask=m)
    strength = lm.strength
    L = to_lab(ref)[..., 0]
    const = None if color in (None, "auto") else _hex(color) if isinstance(color, str) else tuple(color)
    if method == "auto":
        solid, dens = mass_share(ref, lm)
        method = "masses" if solid >= MASS_SOLID or dens >= MASS_DENSITY else "strokes"
    if method == "masses":
        from .masses import ink_mass_ops
        # faint grey marks are ink in a monochrome drawing; in a coloured picture they are colour shapes (thin
        # shadows, highlights) that belong to the flats and the shading, so there only the solid ink is outlined
        lab = to_lab(ref)
        mono = float(np.percentile(np.hypot(lab[..., 1], lab[..., 2]), 95)) < 8.0
        from .masses import ink_coverage
        # a monochrome drawing's faint strands are masses too (at their true ink colour and width with
        # ``true_thin``); hairlines left over are pen strokes, likewise at their true ink
        # coloured line art: ink lines need not be black (brown eye lines, red armour lines), so a mass core may
        # be less dark, and a long mass is cut where its colour changes
        mops, foot = ink_mass_ops(ref, m, color=const, seed=seed, faint_min=0.1 if mono else 2.0,
                                  true_thin=bool(true_thin), core_cov=0.75 if mono else 0.6,
                                  split_de=None if (mono or const is not None) else 12.0)
        cov, _ = ink_coverage(ref)
        # what the masses leave: hairlines too faint for an outline (away from the masses' antialiased rims)
        near = cv2.dilate(foot.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        pieces, _ = extract_lines(ref, m & ~near, detail=detail)
        if true_thin is True and const is None:
            ground = cv2.morphologyEx(ref, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            pieces = true_ink(pieces, cov, ground)
        sops = _piece_strokes(pieces, weight, wf, const, brush, seed + len(mops), foot=foot)
        # hairlines first, then the solid ink on top of their ends (reading order within each)
        sops.sort(key=lambda op: (int(op.pts[:, 1].min() // 64), -len(op.pts)))
        out = sops + mops
        if true_thin is True and foot.any():
            # thin light strands inside the solid ink (a lash's highlights, white hatching in a black mass): paper
            # coloured pen strokes on top, at their true width
            inner = cv2.erode(foot.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            inv = 255 - ref
            lp, _ = extract_lines(inv, inner & m, detail=0.6)
            if lp:
                ground_i = 255 - cv2.morphologyEx(ref, cv2.MORPH_OPEN,
                                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
                lp = true_ink(lp, np.clip(1.0 - cov, 0, 1), ground_i, invert=True)
                out += _piece_strokes(lp, 1.0, 1.0, None, "liner", seed + len(out))
        return out
    # solid dark masses (black hair, dark clothes): an inker outlines them instead of tracing their narrow parts
    mass = dark_masses(ref, lm, m, L=L)
    ops = []
    ksz = 2 * int(lm.width) + 3
    kel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    near = np.zeros(L.shape, bool)
    if mass.any():
        ops += _mass_outline_ops(ref, L, mass, lm, weight * wf, brush, const, seed)
        near |= cv2.dilate(mass.astype(np.uint8), kel) > 0
    fg = silhouette(ref, m)
    if fg is not None:
        sil = _silhouette_ops(ref, L, fg & ~mass, lm, weight * wf, brush, const, seed)
        if sil:
            ops += sil
            drawn = np.zeros(L.shape, np.uint8)
            for o in sil:
                cv2.polylines(drawn, [np.rint(o.pts - 0.5).astype(np.int32).reshape(-1, 1, 2)], False, 1,
                              max(1, int(round(float(np.max(o.width))))))
            near |= cv2.dilate(drawn, kel) > 0
    # the line art proper: linked, smoothed ridge lines (one colour per stroke)
    pieces, _ = extract_lines(ref, m & ~near, detail=detail)
    for i, pc in enumerate(pieces):
        w = np.maximum(pc.width.astype(np.float64) * weight * wf, 1.0)
        n = len(pc.pts)
        if n >= 4:
            s_ = np.concatenate([[0], np.cumsum(np.sqrt((np.diff(pc.pts, axis=0) ** 2).sum(1)))])
            tl = min(2.5 * float(np.median(w)) + 1, 0.4 * s_[-1])
            f = np.ones(n)
            if pc.free[0] and tl > 0:
                f *= 0.4 + 0.6 * np.clip(s_ / tl, 0, 1)
            if pc.free[1] and tl > 0:
                f *= 0.4 + 0.6 * np.clip((s_[-1] - s_) / tl, 0, 1)
            w = np.maximum(w * f, 1.0)
        lp = float(np.sqrt((np.diff(pc.pts, axis=0) ** 2).sum(1)).sum())
        if lp < 4.0 * float(w.mean()):
            # a short mark stays a line: as fine as its length allows (DESIGN.md 10.1)
            w = np.maximum(w * (lp / (4.0 * float(w.mean()))) * 0.98, 1.0)
            if lp < 4.0 * float(w.mean()):
                continue
        ops.append(Stroke(pc.pts, w.astype(np.float32), const if const is not None else pc.color, brush=brush,
                          seed=seed + i))
    # draw in reading order: top to bottom, longer lines first within a band (like an inker)
    key = [(int(op.pts[:, 1].min() // 64), -len(op.pts)) for op in ops]
    order = sorted(range(len(ops)), key=lambda j: key[j])
    return [ops[j] for j in order]
