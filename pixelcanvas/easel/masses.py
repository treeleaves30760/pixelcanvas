"""Ink masses as outlined fills: dense line art drawn the way an inker's lines look under a loupe.

A pen line is a shape of solid ink with a crisp edge.  Tracing its centre line and stroking it again blurs it (the
width and the tapered ends are guessed); here the ink itself is outlined instead:

1. ink coverage: how much of each pixel is ink, from the pixel's darkness between the local paper (the lightest
   value nearby) and the ink colour;
2. the coverage is interpolated smoothly to ``ss`` x the resolution and cut at one half, which places the ink's edge
   to a fraction of a pixel (locating an edge, DESIGN.md 10.1);
3. every connected ink mass becomes one Fill: its outer outline and its holes (even-odd), smoothed along the outline
   except at corners (where the outline turns sharply the corner is kept, potrace-style), one ink colour per mass
   (the median of the reference under the mass's own solid core), rendered antialiased;
4. a mass smaller than a fill may be (12 px^2) becomes a short stroke along its long axis or a dab;
5. grey washes and screentone are not ink: a mass must contain ink-dark pixels, so a grey area stays for the tone
   (flat) layer, and faint lines thinner than a pixel stay for the pen strokes of ``stage_ink``.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .ops import Fill, Stroke
from .trace import to_lab

__all__ = ["ink_coverage", "ink_mass_ops", "ink_level", "line_quality"]


def ink_level(L, dark=None):
    """Lab lightness of the ink: the dark end of the line pixels' lightness."""
    v = L[dark] if dark is not None and dark.any() else L.ravel()
    return float(np.percentile(v, 3))


def ink_coverage(ref, *, L=None, l_ink=None, radius=5, min_contrast=18.0):
    """Ink coverage 0..1 per pixel and the ink lightness.

    The ground under a line is the picture with its lines closed away (a morphological closing over ``radius`` px:
    white paper, or the grey of a tone the line is drawn over); a pixel is (ground - L) / (ground - ink) ink.  A
    solid area wider than that is ink where it is as dark as the ink, and its antialiased rim is measured against
    the lightest value just outside it.  A grey tone is not ink (it is its own ground)."""
    if L is None:
        L = to_lab(ref)[..., 0]
    L = L.astype(np.float32)
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    clo = cv2.morphologyEx(L, cv2.MORPH_CLOSE, ell(radius))
    if l_ink is None:
        l_ink = ink_level(L, (clo - L) > 25)
    span = clo - l_ink
    cov = np.where(span > min_contrast, (clo - L) / np.maximum(span, 1e-3), 0.0)
    # solid ink areas wider than the closing: full ink when as dark as the ink
    solid = np.clip(1.0 - (L - l_ink - 4.0) / 12.0, 0.0, 1.0)
    # ... and their rim, against the lightest value within 2 px (only right next to such an area)
    near = cv2.dilate((solid >= 0.5).astype(np.uint8), ell(2)) > 0
    dil = cv2.dilate(L, ell(2))
    sp2 = dil - l_ink
    rim = np.where(near & (sp2 > min_contrast), (dil - L) / np.maximum(sp2, 1e-3), 0.0)
    cov = np.maximum(np.maximum(cov, solid), rim)
    return np.clip(cov, 0.0, 1.0).astype(np.float32), float(l_ink)


def _turn(c, k):
    """Turning angle (radians) at each point of a closed ring, measured over +-k samples."""
    a = c - np.roll(c, k, 0)
    b = np.roll(c, -k, 0) - c
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    cosv = np.clip((a * b).sum(1) / (na * nb), -1.0, 1.0)
    return np.arccos(cosv)


def _rdp_closed(c, eps):
    from .trace import _rdp
    if len(c) < 8:
        return c
    # split the ring at its two farthest-apart points so both halves simplify independently
    i0 = 0
    i1 = int(np.argmax(((c - c[0]) ** 2).sum(1)))
    a = _rdp(c[i0:i1 + 1], eps)
    b = _rdp(np.vstack([c[i1:], c[:1]]), eps)
    return np.vstack([a[:-1], b[:-1]])


def fit_ring(c, ss, *, sigma_px=0.45, corner_deg=60.0, eps_px=0.1):
    """Smooth closed outline from a supersampled pixel contour: Gaussian smoothing along the ring with the corners
    kept (where the outline turns by more than ``corner_deg`` within ~1 px), then simplified.  ``c``: [N,2]
    contour of the ss-grid in ss-pixel indices.  Returns [M,2] float32 canvas coordinates or None."""
    p = (c.astype(np.float64) + 0.5) / ss
    n = len(p)
    if n < 3:
        return None
    if n >= 8:
        s = sigma_px * ss
        sm = np.stack([ndi.gaussian_filter1d(p[:, k], s, mode="wrap") for k in range(2)], 1)
        k = max(2, int(round(1.0 * ss)))
        if n > 2 * k + 2:
            ang = _turn(p, k)
            corner = ang > np.deg2rad(corner_deg)
            if corner.any():
                # weight of the raw outline: 1 at a corner, falling off over ~k samples
                w = ndi.maximum_filter1d(corner.astype(np.float64), 2 * k + 1, mode="wrap")
                w = ndi.uniform_filter1d(w, k + 1, mode="wrap")
                sm = sm * (1 - w[:, None]) + p * w[:, None]
        p = sm
    q = _rdp_closed(p, eps_px)
    if len(q) > 12:
        from .ops import _stair_share
        if _stair_share(q) > 0.4:
            # a small, blocky ring (mostly corners): smooth it all
            p0 = (c.astype(np.float64) + 0.5) / ss
            p0 = np.stack([ndi.gaussian_filter1d(p0[:, k], sigma_px * ss, mode="wrap") for k in range(2)], 1)
            q = _rdp_closed(p0, eps_px)
    if len(q) < 3:
        return None
    return q.astype(np.float32)


def _ring_area(q):
    q = np.asarray(q, np.float64)
    return 0.5 * float(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))


def ink_mass_ops(ref, mask=None, *, ss=4, core_cov=0.75, core_dark=0.45, min_hole=0.5, color=None, seed=0,
                 stats=None, cov=None, l_ink=None, faint_min=0.1, faint_area=4.0, true_thin=False, ground=None, split_de=None):
    """Fills for the ink masses of the line art (see the module doc).  Returns (ops, ink_px) where ``ink_px`` is the
    bool [H,W] footprint of the masses (pixels at least half ink), for the strokes and flats that follow.

    core_cov / core_dark: a mass must contain a pixel at least ``core_cov`` ink that is also darker than
    ``core_dark`` of the way from the ink to white (grey washes and screentone have no such core).
    color: None = one eyedropped ink colour per mass, or an (r,g,b) for every mass.
    true_thin: a faint mark (no solid core) is drawn at its true ink colour and width: the colour it shows unmixed
    from the ground it lies on, the outline moved in until the mark holds the ink the reference shows across it
    (a 1 px minimum; below that, 1 px as light as the mark shows).  ``ground``: RGB of the picture with its lines
    closed away (computed when needed).
    split_de: cut solid masses where the ink colour changes by more than about this Delta E (coloured line art);
    None = one colour per connected mass."""
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    lab = to_lab(ref)
    L = lab[..., 0]
    if cov is None:
        cov, l_ink = ink_coverage(ref, L=L, l_ink=l_ink)
    m = np.ones((H, W), bool) if mask is None else np.asarray(mask).astype(bool)
    stats = {} if stats is None else stats
    cov = np.where(m, cov, 0.0).astype(np.float32)
    core = (cov >= core_cov) & (L <= l_ink + core_dark * (100.0 - l_ink)) & m
    # local peak coverage (the line's own darkness) within ~2 px
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    peak = cv2.dilate(cov, k5)
    gap = cv2.dilate(1.0 - cov, k5)       # the same for the paper: a thin white gap inside the ink
    # solid ink: cut so that the outline holds as much ink as the reference shows (for a blurred line of peak p
    # the width that keeps its ink is reached at td(p) = p * exp(-0.785 p^2), 0.46 for a solid line), and keeps
    # as much paper in a thin white gap inside the ink (the same rule seen from the paper's side)
    td = lambda x: x * np.exp(-0.785 * x * x)
    t_a = np.clip(0.5 + td(peak) - td(gap), 0.12, 0.9).astype(np.float32)
    # faint lines (no solid core): their own grey at the width where they are half their peak
    # (peak within ~1 px: a faint strand right beside a dark line keeps its own level)
    t_b = np.maximum(0.5 * cv2.dilate(cov, np.ones((3, 3), np.uint8)), 0.06).astype(np.float32)
    up = cv2.resize(cov, (W * ss, H * ss), interpolation=cv2.INTER_CUBIC)
    bw_a = (up >= cv2.resize(t_a, (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)).astype(np.uint8)
    na, lab_a = cv2.connectedComponents(bw_a, connectivity=8)
    core_ss = np.zeros(bw_a.shape, bool)
    cy, cx = np.nonzero(core)
    core_ss[cy * ss + ss // 2, cx * ss + ss // 2] = True
    keep_a = np.zeros(na, bool)
    keep_a[np.unique(lab_a[core_ss])] = True
    keep_a[0] = False
    A = keep_a[lab_a]
    del bw_a
    bw_b = (up >= cv2.resize(t_b, (W * ss, H * ss), interpolation=cv2.INTER_LINEAR)) & ~A
    pk_ok = cv2.resize((peak >= 0.6 * faint_min).astype(np.uint8), (W * ss, H * ss),
                       interpolation=cv2.INTER_NEAREST) > 0
    bw_b &= pk_ok
    del up
    nb, lab_b, st_b, _ = cv2.connectedComponentsWithStats(bw_b.astype(np.uint8), connectivity=8)
    # a faint mark must be large enough to be a line (a faint tail of a solid line is its own grey mark) and
    # somewhere at least ``faint_min`` ink (hysteresis: its fainter stretches carry on from there)
    sy, sx = np.nonzero((cov >= faint_min) & m)
    seeded = np.zeros(nb, bool)
    seeded[np.unique(lab_b[sy * ss + ss // 2, sx * ss + ss // 2])] = True
    keep_b = (st_b[:, cv2.CC_STAT_AREA] >= faint_area * ss * ss) & seeded
    keep_b[0] = False
    B = keep_b[lab_b]
    passes = [(lab_b, keep_b, B)]
    if split_de and A.any():
        # coloured line art: a long mass changes colour along its length (brown hair lines run into black armour
        # lines), so it is cut where its ink colour changes: the ink pixels are grouped by colour, each group's part
        # of the mass is its own fill with its own colour (the parts overlap by a quarter pixel: no seam)
        from .stage_flat import _auto_k
        footA = cv2.resize(A.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) >= 0.5
        if footA.sum() >= 50:
            k = _auto_k(ref, footA, seed, 1.0, target=float(split_de), kmax=12)
            X = lab[footA].astype(np.float32)
            cv2.setRNGSeed(int(seed))
            _, li, _c = cv2.kmeans(X, max(1, k), None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                   1, cv2.KMEANS_PP_CENTERS)
            K = max(1, k)
            ind = np.zeros((K, H, W), np.float32)
            ind[li[:, 0], *np.nonzero(footA)] = 1.0
            ind = np.stack([cv2.blur(ind[c], (5, 5)) for c in range(K)])
            clm = np.argmax(ind, 0).astype(np.int32)
            _, (iy, ix) = ndi.distance_transform_edt(~footA, return_indices=True)
            clm = clm[iy, ix]
            cl_up = np.repeat(np.repeat(clm, ss, 0), ss, 1)
            for c in range(K):
                part = A & (cv2.dilate((cl_up == c).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
                if not part.any():
                    continue
                nc, lab_cc = cv2.connectedComponents(part.astype(np.uint8), connectivity=8)
                kc = np.ones(nc, bool)
                kc[0] = False
                passes.append((lab_cc, kc, part))
        else:
            passes.append((lab_a, keep_a, A))
    else:
        passes.append((lab_a, keep_a, A))
    ops = []
    small = 0
    foot = np.zeros((H, W), bool)
    # faint marks first, the solid ink on top
    for lab_x, keep, sel_x in passes:
        bw = sel_x.astype(np.uint8)
        # footprint at canvas resolution: a pixel is ink when at least half of it is
        cover = cv2.resize(bw.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        fx = cover >= 0.5
        foot |= fx
        full = cover >= 0.999
        contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is None:
            continue
        # one eyedropped colour per mass: the median of the reference over the mass's solid core (its fully
        # covered pixels), else over its footprint (the pixels it half covers), else over the pixels it touches
        lab_s = np.where(sel_x, lab_x, 0).astype(np.int32)
        lab_c = lab_s[ss // 2::ss, ss // 2::ss][:H, :W]
        lab_t = lab_s.reshape(H, ss, W, ss).max(axis=(1, 3))
        ids = np.flatnonzero(keep)
        med = {}
        if color is None and len(ids):
            for sel, lc0, need in ((cover > 0, lab_t, 1), (fx, lab_c, 1), (full, lab_c, 3)):
                lc = np.where(sel, lc0, 0)
                cnt = np.bincount(lc.ravel(), minlength=len(keep))
                ok = ids[cnt[ids] >= need]
                if len(ok):
                    cols = np.stack([np.asarray(ndi.median(ref[..., ch], labels=lc, index=ok))
                                     for ch in range(3)], 1)
                    med.update({int(a): tuple(int(round(v)) for v in c) for a, c in zip(ok, cols)})
        faint_pass = lab_x is lab_b
        unmix = {}
        if true_thin and faint_pass and color is None and len(ids):
            # a faint mark's true ink: over the pixels it touches, colour = ground + (ink - ground) x coverage
            # (means, so the model is linear), and the ink it holds across = its touched width x mean coverage
            lc = np.where(cover > 0, lab_t, 0)
            cnt = np.bincount(lc.ravel(), minlength=len(keep))
            ok = ids[cnt[ids] >= 1]
            if len(ok):
                if ground is None:
                    ground = cv2.morphologyEx(ref, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
                cm = np.stack([np.asarray(ndi.mean(ref[..., ch], labels=lc, index=ok)) for ch in range(3)], 1)
                gm = np.stack([np.asarray(ndi.mean(ground[..., ch], labels=lc, index=ok)) for ch in range(3)], 1)
                vm = np.asarray(ndi.mean(cov, labels=lc, index=ok))
                unmix = {int(a): (c, g, v) for a, c, g, v in zip(ok, cm, gm, vm)}
        hier = hier[0]
        children = {}
        for i, h in enumerate(hier):
            if h[3] >= 0:
                children.setdefault(int(h[3]), []).append(i)
        for i, h in enumerate(hier):
            if h[3] >= 0:
                continue
            outer = fit_ring(contours[i][:, 0, :], ss)
            if outer is None:
                continue
            rings = [outer]
            area = abs(_ring_area(outer))
            for j in children.get(i, []):
                hr = fit_ring(contours[j][:, 0, :], ss)
                if hr is not None and abs(_ring_area(hr)) >= min_hole:
                    rings.append(hr)
                    area -= abs(_ring_area(hr))
            if color is not None:
                col = tuple(int(v) for v in color)
            else:
                px, py = contours[i][0, 0]
                col = med.get(int(lab_s[py, px]))
                if col is None:
                    yy, xx = min(int(py // ss), H - 1), min(int(px // ss), W - 1)
                    col = tuple(int(v) for v in ref[yy, xx])
                u = unmix.get(int(lab_s[py, px])) if unmix else None
                if u is not None and u[2] > 0.08:
                    cmean, g, v = u
                    perim = sum(float(np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0), axis=1).sum())
                                for r in rings)
                    wd = 2.0 * area / max(perim, 1e-6)             # the mark's drawn width
                    if wd >= RESOLVED_PX:
                        u = None                                    # a resolved grey line: its own grey
                if u is not None and u[2] > 0.08:
                    # unmix part of the way (UNMIX: 1 = all the way to full ink, which over-darkens where the
                    # mark's own grey is part ink tone): darker by k, narrower by k, the same ink across
                    k = max(float(v), 0.2) ** UNMIX
                    ink = g + (cmean - g) / k
                    held = (wd + 1.0) * k
                    if held < 1.0:
                        ink = g + (ink - g) * max(held, 0.15)       # 1 px of it, as light as it shows
                        held = 1.0
                    d = 0.5 * (wd - held)
                    if d > 0.05:
                        rings = [_offset_ring(r, d, k > 0) for k, r in enumerate(rings)]
                        if rings[0] is None:
                            continue
                        rings = [r for r in rings if r is not None]
                        area = abs(_ring_area(rings[0])) - sum(abs(_ring_area(r)) for r in rings[1:])
                        outer = rings[0]
                    col = tuple(int(round(x)) for x in np.clip(ink, 0, 255))
            if area >= 12.0:
                ops.append(Fill(polygons=rings, color=col))
                continue
            small += 1
            op = _small_mark(outer, area, col, seed + len(ops))
            if op is not None:
                ops.append(op)
    stats.update({"masses": len(ops), "small": small, "l_ink": round(l_ink, 1), "solid": int(keep_a.sum()),
                  "faint": int(keep_b.sum()), "ink_px": int(foot.sum())})
    return ops, foot


UNMIX = 0.5
RESOLVED_PX = 1.8   # a faint mark at least this wide is a grey line the picture resolves (drawn in grey), not
                    # a thinner dark line blurred by the pixel grid


def _offset_ring(r, d, grow):
    """Move a closed ring ``d`` px along its normals: inward (``grow`` False, a smaller shape) or outward."""
    c = r.astype(np.float64)
    if len(c) < 3:
        return None
    t = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    a, b = c + nrm * d, c - nrm * d
    small, large = (a, b) if abs(_ring_area(a)) < abs(_ring_area(b)) else (b, a)
    q = large if grow else small
    if not grow and (abs(_ring_area(q)) < 0.25 or np.sign(_ring_area(q)) != np.sign(_ring_area(c))):
        return None
    return q.astype(np.float32)


def _small_mark(outer, area, col, seed, ground=None):
    """A mass below the fill minimum: a short stroke along its long axis, or a 3 px dab.  ``ground``: the colour
    around it (default white paper), toward which a dot's colour is spread over the dab."""
    p = outer.astype(np.float64)
    c = p.mean(0)
    if len(p) >= 3:
        u, s, vt = np.linalg.svd(p - c, full_matrices=False)
        d = vt[0]
        t = (p - c) @ d
        L = float(t.max() - t.min())
    else:
        d, L = np.array([1.0, 0.0]), 0.0
    w = area / max(L, 1e-6)
    if L >= 4.0 * max(w, 1.0) and L >= 4.0:
        w = max(1.0, w)
        a = c + d * (t.min() + 0.5 * w)
        b = c + d * (t.max() - 0.5 * w)
        mid = 0.5 * (a + b)
        pts = np.stack([a, mid, b]).astype(np.float32)
        if float(np.linalg.norm(b - a)) < 4.0 * w:
            pts = np.stack([c + d * t.min(), c, c + d * t.max()]).astype(np.float32)
        wid = np.array([max(1.0, 0.6 * w), w, max(1.0, 0.6 * w)], np.float32)
        if float(np.linalg.norm(pts[-1] - pts[0])) >= 4.0 * float(wid.mean()):
            return Stroke(pts, wid, col, brush="liner", seed=seed)
    if area < 2.0:
        return None
    # a dot: a 3 px dab whose colour is the ink spread over the dab (lighter for a small dot)
    k = min(1.0, area / 7.07)
    g = np.full(3, 255.0) if ground is None else np.asarray(ground, np.float64)
    colv = g - k * (g - np.asarray(col, np.float64))
    return Stroke(c[None].astype(np.float32), 3.0, tuple(int(v) for v in colv), brush="liner", seed=seed)


def line_quality(ref, img, *, ink=None):
    """Line darkness and edge sharpness of ``img`` against ``ref`` (both uint8 RGB):
    dark_iou: IoU of the pixels darker than L 50; core_L: mean lightness on the reference's line cores (ink-dark
    pixels) in ref / img; edge_grad: mean gradient magnitude of L across the reference's line edges, ref / img."""
    Lr = to_lab(ref)[..., 0]
    Li = to_lab(img)[..., 0]
    dr, di = Lr < 50, Li < 50
    iou = float((dr & di).sum() / max((dr | di).sum(), 1))
    clo = cv2.morphologyEx(Lr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    core = (clo - Lr > 40) & (Lr < 45)
    def gm(L):
        gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8
        return np.sqrt(gx * gx + gy * gy)
    gr, gi = gm(Lr), gm(Li)
    edge = gr > np.percentile(gr, 92)
    return {"dark_iou": round(iou, 4), "core_L_ref": round(float(Lr[core].mean()), 2),
            "core_L_img": round(float(Li[core].mean()), 2),
            "edge_grad_ref": round(float(gr[edge].mean()), 2), "edge_grad_img": round(float(gi[edge].mean()), 2)}
