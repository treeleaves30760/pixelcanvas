"""DESIGN.md 10.1 mark rules: the checker itself, and every stage function's output on synthetic references."""
import numpy as np
import pytest

from pixelcanvas.ops import Fill, Stroke, check_mark_rules, mark_area, mask_to_rle
from pixelcanvas.pipeline import composite
from pixelcanvas.brush import apply_ops, new_layer
from pixelcanvas.stage_flat import flat_ops
from pixelcanvas.stage_ink import ink_ops
from pixelcanvas.stage_paint import ground_ops, paint_ops
from pixelcanvas.stage_shade import shade_ops
from pixelcanvas.stage_sketch import sketch_ops

from .test_hand_stages import cel_ref, paint_ref

SQUARE = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], np.float32)


def test_good_marks_pass():
    ops = [Stroke([[0, 0], [20, 0]], 2.0, (1, 2, 3)),                 # line: 20 >= 4 x 2
           Stroke([[5, 5]], 3.0, (1, 2, 3), brush="round"),           # dab >= 3 px
           Stroke([[0, 0], [4, 0]], 1.0, (0, 0, 0)),                  # hairline 4 px
           Fill(polygons=[SQUARE], color=(9, 9, 9)),
           Fill(polygons=[SQUARE], color=(9, 9, 9), color2=(200, 0, 0), p0=(10, 10), p1=(30, 30), radial=True)]
    assert check_mark_rules(ops) == []


@pytest.mark.parametrize("op, words", [
    (Stroke([[0, 0], [20, 0]], 2.0, np.zeros((2, 3), np.uint8)), "per-point"),
    (Fill(polygons=None, mask_rle=mask_to_rle(np.pad(np.ones((8, 8), bool), 4)), color=(0, 0, 0)), "stepped"),
])
def test_bad_marks_are_named(op, words):
    bad = check_mark_rules([op])
    assert len(bad) == 1 and words in bad[0]


@pytest.mark.parametrize("op, level", [
    (Stroke([[5, 5]], 1.0, (0, 0, 0), brush="pixel"), 0),             # the retouching brush
    (Stroke([[0, 0], [20, 0]], 1.0, (0, 0, 0), brush="pixel"), 1),
    (Stroke([[0, 0], [20, 0]], 0.6, (0, 0, 0)), 0),
    (Stroke([[0, 0], [5, 0]], 2.0, (0, 0, 0)), 2),
    (Stroke([[5, 5]], 2.0, (0, 0, 0)), 2),
    (Fill(polygons=[np.array([[0, 0], [3, 0], [3, 3]], np.float32)], color=(0, 0, 0)), 2),
    (Fill(polygons=[SQUARE], color=(9, 9, 9)), 4),
    (Stroke([[0, 0], [200, 0]], 40.0, (0, 0, 0)), 5),
])
def test_fine_marks_are_legal_at_their_level(op, level):
    from pixelcanvas.ops import mark_level
    assert check_mark_rules([op]) == []
    assert mark_level(op) == level
    bad = check_mark_rules([op], min_level=level + 1)
    assert len(bad) == 1 and "finer than level" in bad[0]


def test_stair_polygon_is_flagged():
    # a pixel-traced outline: unit axis steps
    xs = np.arange(0, 20)
    stairs = np.concatenate([np.stack([xs, xs], 1), np.stack([xs + 1, xs], 1)])[np.argsort(np.r_[xs * 2, xs * 2 + 1])]
    poly = np.vstack([stairs, [[20, 0]]]).astype(np.float32)
    assert any("stepped" in b for b in check_mark_rules([Fill(polygons=[poly], color=(0, 0, 0))]))


def test_feathered_mask_fill_counts_as_smooth_and_area():
    m = np.zeros((40, 40), bool)
    m[10:20, 10:20] = True
    f = Fill(polygons=None, mask_rle=mask_to_rle(m), color=(0, 0, 0), feather=1.0)
    assert check_mark_rules([f]) == []
    assert mark_area(f) == 100
    assert mark_area(Fill(polygons=[SQUARE], color=(0, 0, 0))) == pytest.approx(400)


def test_radial_gradient_renders():
    L = new_layer(60, 60)
    apply_ops(L, [Fill(polygons=[np.array([[0, 0], [60, 0], [60, 60], [0, 60]], np.float32)], color=(255, 0, 0),
                       color2=(0, 0, 255), p0=(30, 30), p1=(30, 50), radial=True)])
    assert L[30, 30, 0] > 0.95 and L[5, 5, 2] > 0.95 and L[30, 45, 0] > 0.1


def test_stage_outputs_follow_rules():
    cel = cel_ref()
    pnt = paint_ref()
    assert check_mark_rules(sketch_ops(cel, None)) == []
    assert check_mark_rules(ink_ops(cel, None)) == []
    flats = flat_ops(cel, None)
    assert check_mark_rules(flats) == []
    flat_layer = new_layer(*cel.shape[:2])
    apply_ops(flat_layer, flats)
    base = composite({"f": flat_layer}, order=["f"])
    assert check_mark_rules(shade_ops(cel, None, base)) == []
    assert check_mark_rules(ground_ops(pnt, None)) == []
    canvas = np.full_like(pnt, 255)
    for size in (16, 6, 2.5, 1.5, 1.0):
        ops = paint_ops(pnt, None, canvas, size=size, brush="liner" if size < 3 else "round")
        assert check_mark_rules(ops) == [], size
    with pytest.raises(ValueError):
        paint_ops(pnt, None, canvas, size=1.0, brush="pixel")
