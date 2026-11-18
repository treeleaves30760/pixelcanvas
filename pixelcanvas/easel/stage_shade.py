"""Shading as tonal shapes: a multi-level colour decomposition of the reference painted over the flats.

The reference is quantised into as many tones as it has (k-means in Lab, k chosen so every tone is tight), cut into
connected areas, and each area becomes one smooth-outlined Fill: a flat colour, or a linear two-colour gradient when
the area is a soft ramp.  Areas whose colour the flats already have are skipped, so what is added is exactly the
shadow and light shapes (and the small coloured features) of the picture.  Shapes are painted largest first, so
smaller shapes sit on top of the ones that contain them, the way a cel painter stacks shadow, then deeper shadow,
then highlight.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .ops import Blend, Fill, is_partial, mask_to_rle, owned_stack
from .stage_flat import _auto_k
from .trace import as_mask, cluster_colors, merge_small_regions, segment_shapes, segments, to_lab

__all__ = ["shade_ops", "blend_ops", "smooth_area"]


AIR_MIN_RAMP = 0.2      # an area is airbrushed when at least this share of it is a soft ramp in the reference
STEP_GRAD = 3.0         # Delta E per px along a tone shape's outline above which its edge is a hard step
SMALL_HARD_PX = 1500    # a shape along the line art up to this size stays a fill when it stands out by 2 tone steps


def shade_ops(ref, mask, base, *, levels=3, soft=0.3, seed=0, min_area=12, tone_de=None, max_k=64, air="auto",
              air_sizes=(64, 32, 16, 8, 4), stats=None):
    """Shadow / light / feature shapes over ``base`` (RGB of the canvas under the shade layer) inside ``mask``.

    levels 2..5: how finely tones are separated (more levels = tighter tones, more shapes).  soft 0..1: how readily
    a soft ramp becomes a gradient fill instead of steps.  tone_de: Delta E target per tone (overrides levels).
    air: "auto" = in every flat area whose reference is a soft ramp (``AIR_MIN_RAMP``), the soft tone shapes give way
    to an airbrush pass (``airbrush.py``: soft dabs coarse to fine inside the area's frisket), hard-edged shadow
    shapes stay fills; "off" = tone shapes only; "on" = airbrush every area.
    On a region: the whole picture's shapes anchored in the region, each one complete, and the shapes on them."""
    if is_partial(mask):
        allops = shade_ops(ref, None, base, levels=levels, soft=soft, seed=seed, min_area=min_area,
                           tone_de=tone_de, max_k=max_k, air=air, air_sizes=air_sizes, stats=stats)
        return owned_stack(allops, mask, np.asarray(ref).shape)
    ref = np.asarray(ref)
    base = np.asarray(base)[..., :3]
    H, W = ref.shape[:2]
    m = as_mask(mask, ref.shape)
    levels = int(np.clip(levels, 2, 5))
    if tone_de is None:
        tone_de = {2: 9.0, 3: 6.0, 4: 4.5, 5: 3.5}[levels]
    sm = cv2.bilateralFilter(ref, 5, 12, 3)
    k = _auto_k(sm, m, seed, 1.0, target=tone_de, kmax=max_k)
    labels, _ = cluster_colors(sm, k, m, l_weight=1.0, seed=seed)
    seg = merge_small_regions(segments(labels), max(int(min_area), 12))
    seg[~m] = -1
    lab_ref = to_lab(ref)
    lab_base = to_lab(base)
    # how fast the reference changes (Delta E per px): a soft transition gets a feathered shape edge
    lb = cv2.GaussianBlur(lab_ref, (0, 0), 3.0)
    gm = np.zeros((H, W), np.float32)
    for c in range(3):
        gm += cv2.Sobel(lb[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lb[..., c], cv2.CV_32F, 0, 1) ** 2
    gm = np.sqrt(gm) / 8.0
    shapes = segment_shapes(seg, ref, lab=lab_ref, gradients=soft > 0, min_area=min_area)
    # largest first; a shape whose colour the flats already have is skipped, unless a shape painted before it
    # (which is filled across its outline, holes included) now covers it
    shapes.sort(key=lambda d: -d["area"])
    painted = np.zeros((H, W), np.uint8)
    out = []
    # airbrushed areas: flat areas of the canvas whose reference is a soft ramp
    areas = None
    air_ids = set()
    if air in ("auto", "on"):
        from .airbrush import flat_areas, ramp_share
        from .trace import line_map
        lines_m = line_map(ref).mask
        areas = flat_areas(base, m, line_mask=lines_m)
        for i, sl in enumerate(ndi.find_objects(areas), 1):
            if sl is None:
                continue
            A = areas[sl] == i
            if air == "on" or ramp_share(ref[sl], A) >= AIR_MIN_RAMP:
                air_ids.add(i)
        lb1 = cv2.GaussianBlur(lab_ref, (0, 0), 0.8)
        gm1 = np.zeros((H, W), np.float32)
        for c in range(3):
            gm1 += cv2.Sobel(lb1[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lb1[..., c], cv2.CV_32F, 0, 1) ** 2
        gm1 = np.sqrt(gm1) / 8.0
        near_line = cv2.dilate(lines_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for d in shapes:
        x0, y0, x1, y1 = d["box"]
        mk = seg[y0:y1, x0:x1] == d["id"]
        bl = np.median(lab_base[y0:y1, x0:x1][mk], 0)
        de = float(np.sqrt(((d["lab"] - bl) ** 2).sum()))
        if d["grad"] is None and de < 3.0 and painted[y0:y1, x0:x1][mk].mean() < 0.05:
            continue  # the flats already have this colour and nothing painted covers it
        # edge softness: the colour step across the outline divided by the reference's gradient there
        feather = 0.0
        if soft > 0 and d["npx"] >= 40:
            mk8 = mk.astype(np.uint8)
            ring_in = mk & ~(cv2.erode(mk8, np.ones((3, 3), np.uint8)) > 0)
            ring_out = (cv2.dilate(mk8, np.ones((5, 5), np.uint8)) > 0) & ~mk & m[y0:y1, x0:x1]
            if ring_in.any() and ring_out.any():
                g = float(np.median(gm[y0:y1, x0:x1][ring_in]))
                lo = lab_ref[y0:y1, x0:x1][ring_out]
                # inside a ramp the colour changes by one tone step over the band: transition width = step / slope
                step = max(float(np.sqrt(((np.median(lo, 0) - d["lab"]) ** 2).sum())), tone_de)
                tw = step / max(g, 0.05)
                feather = float(np.clip((tw - 2.5) * 0.6, 0.0, 30.0 * soft))
                # never wider than a fraction of the shape's own thickness (a thin dark line is not a soft ramp)
                thick = 2.0 * float(cv2.distanceTransform(np.pad(mk8, 1), cv2.DIST_L2, 3).max())
                feather = min(feather, 0.25 * thick)
                if feather < 0.5:
                    feather = 0.0
        if air_ids:
            # inside an airbrushed area only a true step (a cel shadow's hard edge: the reference changes by a
            # whole tone within a pixel or two along most of the outline) stays a fill; a ramp is the airbrush's
            ids = areas[y0:y1, x0:x1][mk]
            if len(ids) and int(np.bincount(ids).argmax()) in air_ids:
                mk8 = mk.astype(np.uint8)
                ring = mk & ~(cv2.erode(mk8, np.ones((3, 3), np.uint8)) > 0)
                # (the outline where it runs along the line art says nothing: a line is always a step)
                free = ring & ~near_line[y0:y1, x0:x1]
                if free.sum() >= max(4, 0.3 * ring.sum()):
                    hard = float(np.median(gm1[y0:y1, x0:x1][free])) >= STEP_GRAD
                else:
                    # an outline that runs along the line art: a small shape that stands out clearly from what is
                    # around it (a highlight on a buckle, a rivet) is a crisp fill; a faint patch is the airbrush's
                    out_ring = (cv2.dilate(mk8, np.ones((5, 5), np.uint8)) > 0) & ~mk & m[y0:y1, x0:x1]
                    hard = False
                    if out_ring.any() and d["npx"] <= SMALL_HARD_PX:
                        lo = np.median(lab_ref[y0:y1, x0:x1][out_ring], 0)
                        hard = float(np.sqrt(((lo - d["lab"]) ** 2).sum())) >= 2.0 * tone_de
                if not hard:
                    continue
        if d["grad"] is not None:
            c0, c1, p0, p1 = d["grad"]
            f = Fill(polygons=d["rings"], mask_rle=None, color=c0, color2=c1, p0=p0, p1=p1, feather=round(feather, 2))
        else:
            f = Fill(polygons=d["rings"], mask_rle=None, color=d["rgb"], feather=round(feather, 2))
        for k, r in enumerate(d["rings"]):
            cv2.fillPoly(painted, [np.rint(r - 0.5).astype(np.int32).reshape(-1, 1, 2)], 1 if k == 0 else 0)
        out.append(f)
    if air_ids:
        from .airbrush import airbrush_ops
        from .brush import apply_ops, new_layer
        # the canvas the airbrush works on: the base with the hard shapes painted
        lay = new_layer(H, W)
        lay[..., :3] = base.astype(np.float32) / 255.0
        lay[..., 3] = 1.0
        apply_ops(lay, out)
        can = np.clip(np.rint(lay[..., :3] * 255), 0, 255).astype(np.uint8)
        sel = np.where(np.isin(areas, list(air_ids)), areas, 0).astype(np.int32)
        out += airbrush_ops(ref, can, sel, sizes=air_sizes, seed=seed, stats=stats, edges=gm1 >= STEP_GRAD)
    return out


def smooth_area(ref, mask=None, *, edge_px=2.0, max_grad=None):
    """Where the reference is smooth (no edge within ``edge_px``, gentle colour change): the area a blending pass
    may soften (bool [H,W])."""
    from .trace import edge_map
    ref = np.asarray(ref)
    m = as_mask(mask, ref.shape)
    e = edge_map(ref, sigma=1.0)
    r = max(1, int(round(edge_px)))
    e = cv2.dilate(e.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    lab = cv2.GaussianBlur(to_lab(ref), (0, 0), 1.0)
    g = np.zeros(ref.shape[:2], np.float32)
    for c in range(3):
        g += cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1) ** 2
    g = np.sqrt(g) / 8.0
    if max_grad is None:
        max_grad = 4.0
    return m & ~e & (g < max_grad)


def blend_ops(ref, mask=None, *, radius=3.0, tol=10.0, strength=1.0, edge_px=2.0, ref_tol=15.0, local=True):
    """One blending pass over the smooth parts of ``mask`` (where the reference has no edge): softens banding and
    patchy paint already on the layer, keeps lines and hard edges.  The op reads only the layer it is applied to;
    the reference only decides where it goes.  local: never over the areas ``stage_local.char_map`` calls textured
    (their texture is the picture's, not patchy paint)."""
    ref = np.asarray(ref)
    area = smooth_area(ref, mask, edge_px=edge_px)
    if local:
        from .stage_local import TEXTURE, char_map
        area &= char_map(ref).cls != TEXTURE
    # only where the reference itself is smooth at the blend's own scale: its local mean over ``radius`` (taken
    # inside the area, as the blend takes it) stays within ``ref_tol`` of each pixel; next to a hard edge the area
    # stops, so the blend never carries one side of an edge into the other
    inside = area.astype(np.float32)
    rf = ref.astype(np.float32)
    nrm = cv2.GaussianBlur(inside, (0, 0), float(radius))
    mean = cv2.GaussianBlur(rf * inside[..., None], (0, 0), float(radius)) / np.maximum(nrm, 1e-3)[..., None]
    area &= np.abs(mean - rf).max(-1) < float(ref_tol)
    area = cv2.morphologyEx(area.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if area.sum() < 12:
        return []
    return [Blend(mask_rle=mask_to_rle(area), radius=float(radius), tol=float(tol), strength=float(strength),
                  feather=1.5)]
