"""Stage tools on regions: calls over a partition of the canvas give the whole-picture result (DESIGN.md 3)."""
import cv2
import numpy as np
from PIL import Image

from pixelcanvas.easel.brush import apply_ops, new_layer
from pixelcanvas.easel.ops import Fill, Stroke, op_anchor, ops_from_jsonable, ops_to_jsonable, owned_ops
from pixelcanvas.easel.pipeline import run_reference_pipeline
from pixelcanvas.easel.stage_flat import flat_ops
from pixelcanvas.easel.stage_ink import ink_ops
from pixelcanvas.easel.trace import delta_e, line_map

from .test_hand_stages import cel_ref


def _partition(h, w, shift=0):
    """Three irregular regions (a wavy vertical cut and a wavy horizontal cut through the figure)."""
    y, x = np.mgrid[0:h, 0:w]
    a = x < 95 + shift + 12 * np.sin(y / 17.0)
    b = ~a & (y < 140 + shift + 9 * np.cos(x / 13.0))
    c = ~a & ~b
    return [a, b, c]


def _run(ref, tmp, regions):
    rep = run_reference_pipeline(ref, str(tmp), "cel", log=lambda *a: None, regions=regions)
    img = np.asarray(Image.open(tmp / "final.png").convert("RGB"))
    line = np.asarray(Image.open(tmp / "02-line.png").convert("RGB"))
    flat = np.asarray(Image.open(tmp / "03-flat.png").convert("RGB"))
    return rep, img, line, flat


def test_region_calls_equal_whole_run(tmp_path):
    ref = cel_ref()
    h, w = ref.shape[:2]
    _, whole, wline, wflat = _run(ref, tmp_path / "whole", None)
    parts = _partition(h, w)
    rep, reg, rline, rflat = _run(ref, tmp_path / "reg", parts)
    # the line art is the same lines, each drawn whole (no line lost or cut at a region border)
    assert float(delta_e(wline, rline).mean()) < 0.05
    # flats: the same areas; only antialiased edges between areas of different regions may differ
    assert float(delta_e(wflat, rflat).mean()) < 0.6
    d = delta_e(whole, reg)
    assert float(d.mean()) < 0.6
    # no seams: along the region borders the region run is as close to the reference as the whole run
    border = np.zeros((h, w), np.uint8)
    for p in parts:
        border |= cv2.morphologyEx(p.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    border = cv2.dilate(border, np.ones((5, 5), np.uint8)) > 0
    e_reg = float(delta_e(ref, reg)[border].mean())
    e_whole = float(delta_e(ref, whole)[border].mean())
    assert e_reg < e_whole + 1.0
    assert float(d[border].mean()) < 1.5


def test_region_calls_robust_to_slightly_off_masks(tmp_path):
    ref = cel_ref()
    h, w = ref.shape[:2]
    lm = line_map(ref)
    for shift in (-3, 3):
        parts = _partition(h, w, shift)
        lay = new_layer(h, w)
        lay_w = new_layer(h, w)
        apply_ops(lay_w, ink_ops(ref, None, lines=lm))
        n = 0
        for p in parts:
            ops = ink_ops(ref, p, lines=lm)
            n += len(ops)
            apply_ops(lay, ops, clip=p)
        assert n == len(ink_ops(ref, None, lines=lm))
        assert float(np.abs(lay[..., 3] - lay_w[..., 3]).mean()) < 0.002
        # flats over a shifted partition still cover the canvas
        fl = new_layer(h, w)
        for p in parts:
            apply_ops(fl, flat_ops(ref, p, line_mask=lm.mask), clip=p)
        assert float(fl[..., 3][4:-4, 4:-4].min()) > 0.85   # at most a faint antialiased seam


def test_ownership_is_a_partition_and_survives_json():
    ref = cel_ref()
    ops = ink_ops(ref, None) + flat_ops(ref, None)
    h, w = ref.shape[:2]
    parts = _partition(h, w)
    got = [owned_ops(list(ops), p) for p in parts]
    assert sum(len(g) for g in got) == len(ops)
    for o in ops:
        x, y = op_anchor(o)
        assert 0 <= x < w and 0 <= y < h
    back = ops_from_jsonable(ops_to_jsonable(got[0]))
    assert all(b.own for b in back)
    # an owned mark is painted whole across the clip; an ordinary one is cut by it
    st = Stroke(np.array([[10, 20], [60, 20]], np.float32), 4.0, (0, 0, 0), brush="liner")
    clip = np.zeros((40, 80), bool)
    clip[:, :30] = True
    a, b = new_layer(40, 80), new_layer(40, 80)
    apply_ops(a, [st], clip=clip)
    st.own = True
    apply_ops(b, [st], clip=clip)
    assert a[20, 50, 3] == 0 and b[20, 50, 3] > 0.9
    f = Fill(polygons=[np.array([[5, 5], [70, 5], [70, 35], [5, 35]], np.float32)], own=True)
    c = new_layer(40, 80)
    apply_ops(c, [f], clip=clip)
    assert c[20, 60, 3] > 0.9
