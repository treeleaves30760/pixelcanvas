"""Round 5: true ink of faint lines, colour-split masses, light highlight strokes, small-mark share."""
import numpy as np
import cv2

from pixelcanvas.easel.masses import ink_mass_ops
from pixelcanvas.easel.ops import Fill, Stroke, Spray, check_mark_rules, mark_area
from pixelcanvas.easel.stage_paint import highlight_ops
from pixelcanvas.easel.brush import new_layer, apply_ops


def _paper(h=120, w=160):
    return np.full((h, w, 3), 245, np.uint8)


def test_faint_masses_true_ink_are_legal_and_darker():
    ref = _paper()
    cv2.line(ref, (10, 20), (150, 30), (20, 20, 20), 4, cv2.LINE_AA)          # solid ink
    cv2.line(ref, (10, 70), (150, 80), (150, 150, 150), 1, cv2.LINE_AA)       # faint thin line
    plain, _ = ink_mass_ops(ref, None)
    true, _ = ink_mass_ops(ref, None, true_thin=True)
    assert not check_mark_rules(true)

    def darkest(ops):
        cols = [sum(o.color) / 3 for o in ops if isinstance(o, (Fill, Stroke))]
        return min(cols) if cols else 255
    # the solid line is untouched; the faint line is never lighter than before
    assert len(true) == len(plain)
    lp = sorted(sum(o.color) / 3 for o in plain)
    lt = sorted(sum(o.color) / 3 for o in true)
    assert lt[-1] <= lp[-1] + 1
    assert darkest(true) <= darkest(plain) + 1


def test_colour_split_masses_take_each_colour():
    ref = _paper()
    cv2.line(ref, (10, 60), (80, 60), (120, 60, 20), 5)        # brown half
    cv2.line(ref, (80, 60), (150, 60), (15, 20, 60), 5)        # navy half, one connected mass
    ops, _ = ink_mass_ops(ref, None, split_de=12.0)
    assert not check_mark_rules(ops)
    cols = [o.color for o in ops if isinstance(o, Fill)]
    assert len(cols) >= 2
    reds = [c[0] for c in cols]
    assert max(reds) - min(reds) > 50


def test_highlight_strokes_light_and_whole_on_regions():
    ref = np.full((100, 140, 3), 60, np.uint8)
    cv2.line(ref, (10, 50), (130, 55), (230, 230, 235), 1, cv2.LINE_AA)
    canvas = np.full_like(ref, 60)
    ops = highlight_ops(ref, None, canvas)
    assert ops and not check_mark_rules(ops)
    assert all(isinstance(o, Stroke) and sum(o.color) / 3 > 120 for o in ops)
    left = np.zeros(ref.shape[:2], bool)
    left[:, :70] = True
    a = highlight_ops(ref, left, canvas)
    b = highlight_ops(ref, ~left, canvas)
    assert len(a) + len(b) == len(ops)


def test_size_chunks_split_small_and_large_marks():
    from pixelcanvas.easel.pipeline import SMALL_MARK, _size_chunks
    big = Fill(polygons=[np.array([[0, 0], [40, 0], [40, 40], [0, 40]], np.float32)], color=(1, 2, 3))
    small = Stroke(np.array([[0, 0], [8, 0]], np.float32), np.array([1.0, 1.0], np.float32), (0, 0, 0))
    spray = Spray(pts=np.array([[10, 10], [20, 20]], np.float32), size=np.array([16, 4], np.float32),
                  colors=np.array([[9, 9, 9], [9, 9, 9]], np.uint8), opacity=np.array([0.5, 0.5], np.float32))
    assert mark_area(small) < SMALL_MARK <= mark_area(big)
    ch = _size_chunks([big, small, spray])
    flags = [s for s, _ in ch]
    assert flags == [False, True, False, True]
    lay1, lay2 = new_layer(60, 60), new_layer(60, 60)
    apply_ops(lay1, [big, small, spray])
    for _, c in ch:
        apply_ops(lay2, c)
    assert np.abs(lay1 - lay2).max() < 1e-5
