"""Detail levels 1 and 0: the hairline and retouching passes that take the canvas the last step to the reference.

The engine paints in detail levels, coarse to fine (DESIGN.md 10.1): 5 block-in, 4 forms, 3 edges, 2 fine, 1 hairline,
0 retouch.  Every level works on the residual the levels above it left (a mark's colour and footprint come from the
reference, a mark is laid only where it lowers the residual), so the levels add up like the terms of a series and the
picture is coherent wherever one stops.  Levels 5-2 are the stage tools (fills, airbrush, ink masses, fitted shapes,
strokes); this module adds the two finest:

- level 1, hairlines: 1 px strokes along the local structure (horizontal or vertical, whichever the reference's
  structure direction is closer to), over runs of pixels that are still off and whose reference colour stays
  nearly constant along the run, each in one colour (the mean of the reference under it); 2 px and longer;
- level 0, retouching: a 1 px dab on every pixel still off by more than ``tol`` Delta E, in the reference's colour
  there (the retouching brush).

With both levels the canvas equals the reference to 8-bit precision; they are one :class:`ops.Retouch` op each, a
vectorised write.  On a region: the whole picture's marks that start in the region, each painted whole (``own``).
"""
from __future__ import annotations

import numpy as np

from .ops import Retouch, is_partial
from .trace import as_mask, orientation_field, to_lab

__all__ = ["retouch_ops", "hairline_ops", "pixel_ops", "residual_de", "L1_TOL", "L0_TOL"]

L1_TOL = 1.5        # Delta E: a pixel off by more than this gets a hairline (level 1)
L1_JOIN = 4.0       # Delta E between neighbouring reference pixels that still continue a hairline
L1_MAX = 24         # px: longest hairline
L0_TOL = 0.5        # Delta E: a pixel off by more than this gets a retouching dab (level 0)


def residual_de(ref, canvas) -> np.ndarray:
    """Per-pixel Delta E (Lab, CIE76) between the reference and the canvas."""
    return np.sqrt(((to_lab(np.asarray(ref)[..., :3]) - to_lab(np.asarray(canvas)[..., :3])) ** 2).sum(-1))


def _runs(sel, lab_r, rgb_r, need_de, max_len):
    """Horizontal runs over the rows of ``sel`` (bool [h,w]): consecutive selected pixels whose reference colour
    changes by less than L1_JOIN from one to the next, cut every ``max_len`` px.  Returns (y, x0, n, colour [k,3]
    uint8, gain [k]) per run of 2 px or more, gain = residual removed (sum of Delta E before - after)."""
    h, w = sel.shape
    d = np.sqrt(((lab_r[:, 1:] - lab_r[:, :-1]) ** 2).sum(-1))
    brk = np.ones((h, w), bool)
    brk[:, 1:] = ~sel[:, :-1] | (d >= L1_JOIN)
    brk &= sel
    flat_sel = sel.ravel()
    ids = np.cumsum(brk.ravel()) - 1
    ids = np.where(flat_sel, ids, -1)
    pix = np.flatnonzero(flat_sel)
    if not len(pix):
        return None
    rid = ids[pix]
    # cut long runs: position within the run
    first = np.r_[0, np.flatnonzero(np.diff(rid)) + 1]
    start_of = np.repeat(first, np.diff(np.r_[first, len(rid)]))
    pos = np.arange(len(rid)) - start_of
    rid2 = rid * 1000 + pos // max_len
    _, rid = np.unique(rid2, return_inverse=True)
    k = int(rid.max()) + 1
    cnt = np.bincount(rid, minlength=k)
    rgb = rgb_r.reshape(-1, 3)[pix].astype(np.float64)
    col = np.stack([np.bincount(rid, weights=rgb[:, c], minlength=k) for c in range(3)], 1) / cnt[:, None]
    col8 = np.clip(np.rint(col), 0, 255).astype(np.uint8)
    lab_col = to_lab(col8[None])[0]
    lr = lab_r.reshape(-1, 3)[pix]
    after = np.sqrt(((lr - lab_col[rid]) ** 2).sum(1))
    gain = np.bincount(rid, weights=need_de.ravel()[pix] - after, minlength=k)
    st = np.full(k, np.iinfo(np.int64).max)
    np.minimum.at(st, rid, pix)
    y, x0 = st // w, st % w
    keep = (cnt >= 2) & (gain > 0)
    return y[keep], x0[keep], cnt[keep], col8[keep], gain[keep]


def hairline_ops(ref, mask=None, canvas=None, *, tol=L1_TOL, max_len=L1_MAX):
    """Level 1: 1 px hairlines over the pixels still off by more than ``tol`` Delta E (see the module doc)."""
    ref = np.asarray(ref)[..., :3]
    H, W = ref.shape[:2]
    if is_partial(mask):
        return _owned(hairline_ops(ref, None, canvas, tol=tol, max_len=max_len), mask)
    canvas = np.full_like(ref, 255) if canvas is None else np.asarray(canvas)[..., :3]
    m = as_mask(mask, ref.shape)
    lab_r = to_lab(ref)
    de = np.sqrt(((lab_r - to_lab(canvas)) ** 2).sum(-1))
    need = m & (de > tol)
    if not need.any():
        return []
    tx, ty, _ = orientation_field(ref, sigma=1.5)
    hor = np.abs(tx) >= np.abs(ty)
    marks = []
    done = np.zeros((H, W), bool)
    # along the structure first, then the other way over what is left
    for axis, pref in ((0, True), (1, True), (0, False), (1, False)):
        sel = need & ~done
        if pref:
            sel &= hor if axis == 0 else ~hor
        if axis == 1:
            r = _runs(sel.T.copy(), np.ascontiguousarray(lab_r.transpose(1, 0, 2)),
                      np.ascontiguousarray(ref.transpose(1, 0, 2)), de.T.copy(), max_len)
        else:
            r = _runs(sel, lab_r, ref, de, max_len)
        if r is None or not len(r[0]):
            continue
        a, b, n, col, _ = r
        if axis == 0:
            ys, xs = a, b
        else:
            xs, ys = a, b
        marks.append((xs, ys, n, np.full(len(n), axis, np.uint8), col))
        rr = Retouch(x=xs, y=ys, n=n, axis=axis, colors=col)
        px, py, _ = rr.pixels()
        done[py, px] = True
    # 2 px dabs over the 2x2 blocks still off whose reference colours stay close together
    sel = need & ~done
    h2, w2 = H // 2 * 2, W // 2 * 2
    if sel[:h2, :w2].any():
        b = sel[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
        blk = b.all((1, 3))
        lb = lab_r[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, 3)
        mean_l = lb.mean((1, 3))
        spread = np.sqrt(((lb - mean_l[:, None, :, None]) ** 2).sum(-1)).max((1, 3))
        blk &= spread < L1_JOIN
        by, bx = np.nonzero(blk)
        if len(by):
            rb = ref[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, 3).astype(np.float64).mean((1, 3))
            col = np.clip(np.rint(rb[by, bx]), 0, 255).astype(np.uint8)
            marks.append((bx * 2, by * 2, np.full(len(by), 2, np.int32), np.zeros(len(by), np.uint8), col,
                          np.full(len(by), 2, np.uint8)))
    if not marks:
        return []
    marks = [mk if len(mk) == 6 else mk + (np.ones(len(mk[2]), np.uint8),) for mk in marks]
    xs, ys, n, ax, col, sz = (np.concatenate([mk[i] for mk in marks]) for i in range(6))
    return [Retouch(x=xs, y=ys, n=n, axis=ax, size=sz, colors=col)]


def pixel_ops(ref, mask=None, canvas=None, *, tol=L0_TOL):
    """Level 0: a 1 px dab in the reference's colour on every pixel of ``mask`` still off by more than ``tol``."""
    ref = np.asarray(ref)[..., :3]
    canvas = np.full_like(ref, 255) if canvas is None else np.asarray(canvas)[..., :3]
    m = as_mask(mask, ref.shape)
    de = residual_de(ref, canvas)
    ys, xs = np.nonzero(m & (de > tol))
    if not len(ys):
        return []
    return [Retouch(x=xs, y=ys, colors=ref[ys, xs], own=is_partial(mask))]


def retouch_ops(ref, mask=None, canvas=None, *, level=1, stats=None):
    """The marks of detail level ``level`` (1 = hairlines, 0 = retouching dabs) on the residual of ``canvas``."""
    ops = hairline_ops(ref, mask, canvas) if int(level) >= 1 else pixel_ops(ref, mask, canvas)
    if stats is not None:
        stats["marks"] = int(sum(len(o.x) for o in ops))
        stats["pixels"] = int(sum(int((o.n.astype(np.int64) * o.size).sum()) for o in ops))
    return ops


def _owned(ops, mask):
    m = np.asarray(mask)
    m = m if m.dtype == bool else m > 0.5
    out = []
    for op in ops:
        sel = m[op.y, op.x]
        if sel.any():
            out.append(Retouch(x=op.x[sel], y=op.y[sel], n=op.n[sel], axis=op.axis[sel], size=op.size[sel],
                               colors=op.colors[sel], own=True))
    return out
