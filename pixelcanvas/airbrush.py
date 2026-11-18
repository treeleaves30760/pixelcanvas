"""Airbrush shading: smooth tone inside each flat area, built up from soft round dabs, coarse to fine.

A soft-shaded cel picture (a blush, the gradient of skin into shadow, a sky) is a ramp, not a step.  Tone shapes cut
it into bands however finely they are cut; an airbrush lays it the way a painter does:

- each flat area is masked off (the frisket: the area's own smooth outline), so the spray never crosses into the
  hair or the background and the area's edge stays crisp under the line art;
- passes with dabs of diameter 64, 32, 16, 8 px: in each pass the reference and the canvas are compared at the
  dab's own scale (both blurred inside the area), and where they still differ a dab is sprayed; its one colour is
  the median of the reference under its footprint (eyedropper), its flow just enough to close the gap given how many
  neighbouring dabs overlap there; a pass stops where the area already matches, so fine dabs go only where the
  tone turns quickly;
- a dab keeps clear of the reference's hard steps (a cel shadow's edge stays the fill's crisp edge);
- the dabs' Gaussian footprints overlap, so the tone they build is continuous, with no band.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .brush import SPRAY_SIGMA, apply_ops, new_layer
from .ops import Spray
from .trace import region_polygon, to_lab

__all__ = ["airbrush_ops", "flat_areas", "ramp_share", "step_edges"]


def step_edges(ref, *, sigma=0.8, thr=3.0):
    """Hard steps of the reference: Lab gradient above ``thr`` Delta E per px (bool [H,W])."""
    lab = cv2.GaussianBlur(to_lab(ref), (0, 0), sigma)
    g = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        g += cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1) ** 2
    return np.sqrt(g) / 8.0 >= thr


def flat_areas(base, mask=None, *, min_area=400, line_mask=None):
    """Areas of one flat colour in ``base`` (uint8 RGB): int32 labels [H,W] (0 = none), smallest specks and the
    antialiased seams between areas joined to their nearest area.  With ``line_mask``, neighbouring areas that no
    line separates are one area."""
    from skimage.measure import label
    b = np.asarray(base)[..., :3].astype(np.int64)
    key = (b[..., 0] << 16) | (b[..., 1] << 8) | b[..., 2]
    lab = label(key, connectivity=1, background=-1).astype(np.int32) + 1
    cnt = np.bincount(lab.ravel())
    big = cnt >= min_area
    big[0] = False
    keep = big[lab]
    if mask is not None:
        keep &= np.asarray(mask).astype(bool)
    if not keep.any():
        return np.zeros(lab.shape, np.int32)
    # every other pixel joins its nearest kept area (antialiased seams, specks), inside the mask
    _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
    out = np.where(keep, lab, lab[iy, ix])
    if mask is not None:
        out = np.where(np.asarray(mask).astype(bool), out, 0)
    if line_mask is not None:
        out = _merge_unlined(out, np.asarray(line_mask).astype(bool))
    # renumber 1..n
    u, inv = np.unique(out, return_inverse=True)
    out = inv.reshape(out.shape).astype(np.int32)
    if u[0] != 0:
        out += 1
    return out


def _merge_unlined(lab, line, line_share=0.35):
    """Join neighbouring areas that no line separates (a soft shadow painted as its own flat belongs to the area
    it shades: one frisket for both, so the airbrush can soften the seam where the reference is soft)."""
    n = int(lab.max()) + 1
    near = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pa, pb, on = [], [], []
    for a, b, ln in ((lab[:, :-1], lab[:, 1:], near[:, :-1] | near[:, 1:]),
                     (lab[:-1, :], lab[1:, :], near[:-1, :] | near[1:, :])):
        d = (a != b) & (a > 0) & (b > 0)
        pa.append(np.minimum(a[d], b[d]))
        pb.append(np.maximum(a[d], b[d]))
        on.append(ln[d])
    pa, pb, on = np.concatenate(pa), np.concatenate(pb), np.concatenate(on)
    if len(pa) == 0:
        return lab
    key = pa.astype(np.int64) * n + pb
    uk, inv, tot = np.unique(key, return_inverse=True, return_counts=True)
    share = np.bincount(inv, on.astype(np.float64)) / tot
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in np.flatnonzero((share < line_share) & (tot >= 8)):
        ra, rb = find(int(uk[e] // n)), find(int(uk[e] % n))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    return np.where(lab > 0, roots[lab], 0).astype(np.int32)


def ramp_share(ref, area):
    """Share of an area's pixels where the reference's colour changes gently (a soft ramp, not flat, not an edge)."""
    lab = cv2.GaussianBlur(to_lab(ref), (0, 0), 2.0)
    g = np.zeros(area.shape, np.float32)
    for c in range(3):
        g += cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1) ** 2
    g = np.sqrt(g) / 8.0
    return float(((g > 0.25) & (g < 4.0) & area).sum() / max(int(area.sum()), 1))


def _inside_blur(img, w, sigma):
    """Blur of ``img`` [h,w,c] over the pixels of weight ``w`` only (normalised convolution)."""
    nrm = cv2.GaussianBlur(w, (0, 0), sigma)
    b = cv2.GaussianBlur(img * w[..., None], (0, 0), sigma)
    return b / np.maximum(nrm, 1e-4)[..., None], nrm


_DISC = None


def _disc(n=32):
    """Fixed sample pattern over the unit disc (sunflower spiral)."""
    global _DISC
    if _DISC is None or len(_DISC) != n:
        k = np.arange(n) + 0.5
        r = np.sqrt(k / n)
        t = k * np.pi * (3 - np.sqrt(5))
        _DISC = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return _DISC


def airbrush_ops(ref, canvas, areas, *, sizes=(64, 32, 16, 8), stop_de=1.2, flow=1.0, max_opacity=0.6,
                 min_ramp=0.0, seed=0, stats=None, edges=None, edge_keep=0.5):
    """One Spray op per area of ``areas`` (int labels, 0 = none) that brings ``canvas`` (uint8 RGB under the
    spray) toward ``ref`` with soft dabs, coarse to fine (see the module doc).  stop_de: Delta E (at a pass's own
    scale) below which a place counts as matched.  min_ramp: skip areas whose reference is less of a soft ramp."""
    ref = np.asarray(ref)
    H, W = ref.shape[:2]
    stats = {} if stats is None else stats
    rng = np.random.default_rng(seed)
    cur = np.asarray(canvas)[..., :3].copy()
    lab_ref = to_lab(ref)
    ops = []
    ndabs = 0
    for i, sl in enumerate(ndi.find_objects(areas), 1):
        if sl is None:
            continue
        pad = int(max(sizes))
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        A = areas[y0:y1, x0:x1] == i
        if A.sum() < 64:
            continue
        if min_ramp > 0 and ramp_share(ref[y0:y1, x0:x1], A) < min_ramp:
            continue
        rings = region_polygon(A, x0, y0, holes=True, offset=0.5, sigma=1.0, eps=0.25)
        if rings is None:
            continue
        frisket = rings if isinstance(rings, list) else [rings]
        w = A.astype(np.float32)
        lr = lab_ref[y0:y1, x0:x1]
        rf = ref[y0:y1, x0:x1].astype(np.float32)
        # a dab stays clear of the reference's hard steps (its soft footprint would blur them): its centre at
        # least ``edge_keep`` x its diameter from the nearest step
        edist = None
        if edges is not None:
            e = np.asarray(edges)[y0:y1, x0:x1]
            edist = ndi.distance_transform_edt(~e) if e.any() else None
        layer = new_layer(y1 - y0, x1 - x0)
        layer[..., :3] = cur[y0:y1, x0:x1].astype(np.float32) / 255.0
        layer[..., 3] = 1.0
        all_p, all_s, all_c, all_o = [], [], [], []
        for s in sizes:
            s = float(s)
            sig = s * SPRAY_SIGMA
            can = np.clip(np.rint(layer[..., :3] * 255), 0, 255).astype(np.uint8)
            T, nrm = _inside_blur(lr, w, sig)
            C, _ = _inside_blur(to_lab(can), w, sig)
            R = np.sqrt(((T - C) ** 2).sum(-1))
            R[~A] = 0
            if float(R[A].mean()) < 0.5 * stop_de and float(np.percentile(R[A], 95)) < stop_de:
                continue
            g = max(2.0, s / 3.0)
            ys = np.arange(g / 2, y1 - y0, g)
            xs = np.arange(g / 2, x1 - x0, g)
            P = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
            P = P + rng.uniform(-0.3, 0.3, P.shape) * g
            P[:, 0] = np.clip(P[:, 0], 0, x1 - x0 - 1e-3)
            P[:, 1] = np.clip(P[:, 1], 0, y1 - y0 - 1e-3)
            xi, yi = P[:, 0].astype(int), P[:, 1].astype(int)
            sel = A[yi, xi] & (R[yi, xi] > stop_de)
            if edist is not None:
                sel &= edist[yi, xi] >= edge_keep * s
            if not sel.any():
                continue
            P, xi, yi = P[sel], xi[sel], yi[sel]
            # eyedropper: the median of the reference under the dab's footprint (inside the area)
            D = P[:, None, :] + _disc() [None] * (0.5 * s)
            dx = np.clip(D[..., 0].astype(int), 0, x1 - x0 - 1)
            dy = np.clip(D[..., 1].astype(int), 0, y1 - y0 - 1)
            ins = A[dy, dx]
            smp = np.where(ins[..., None], rf[dy, dx], np.nan)
            col = np.nanmedian(smp, axis=1)
            ok = np.isfinite(col).all(1)
            P, xi, yi, col = P[ok], xi[ok], yi[ok], col[ok]
            if not len(P):
                continue
            lc = to_lab(np.clip(np.rint(col), 0, 255).astype(np.uint8)[None])[0]
            t, c = T[yi, xi], C[yi, xi]
            dcol = lc - c
            # flow: the share of the way from the canvas to the dab colour that reaches the target, divided by how
            # many dabs overlap at one place (Gaussian footprints on a grid of step g)
            a = (dcol * (t - c)).sum(1) / np.maximum((dcol ** 2).sum(1), 1e-6)
            overlap = 2 * np.pi * sig * sig / (g * g)
            a = np.clip(flow * a / max(overlap, 1.0), 0.0, max_opacity)
            keep = a > 0.02
            if not keep.any():
                continue
            P, col, a = P[keep], col[keep], a[keep]
            order = rng.permutation(len(P))
            P, col, a = P[order], col[order], a[order]
            dab = Spray(pts=P + [x0, y0], size=s, colors=col, opacity=a)
            # paint this pass on the working canvas (inside the area) before the next, finer one compares
            loc = Spray(pts=P, size=s, colors=col, opacity=a)
            sub = layer.copy()
            apply_ops(sub, [loc])
            layer[A] = sub[A]
            all_p.append(dab.pts)
            all_s.append(dab.size)
            all_c.append(dab.colors)
            all_o.append(dab.opacity)
        if all_p:
            op = Spray(pts=np.concatenate(all_p), size=np.concatenate(all_s), colors=np.concatenate(all_c),
                       opacity=np.concatenate(all_o), polygons=frisket, feather=0.75, seed=seed + i)
            ops.append(op)
            ndabs += len(op.pts)
            cur[y0:y1, x0:x1][A] = np.clip(np.rint(layer[..., :3] * 255), 0, 255).astype(np.uint8)[A]
    stats.update({"spray_ops": len(ops), "dabs": ndabs})
    return ops
