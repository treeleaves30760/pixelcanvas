"""Streak strokes (colour and width changing along the path), the working canvas and the tile retouch."""
import numpy as np
import pytest

from pixelcanvas import pipeline as P
from pixelcanvas.ops import (Streak, apply_ops, check_mark_rules, level_counts, mark_level, op_anchor,
                                   ops_from_jsonable, ops_to_jsonable)
from pixelcanvas.stage_streaks import streak_ops
from pixelcanvas.trace import delta_e


def _streak():
    return Streak(pts=[[2.5, 5.5], [12.5, 5.5], [22.5, 10.5], [3.5, 15.5], [9.5, 15.5]], n=[3, 2],
                  colors=[[255, 0, 0], [0, 0, 255], [0, 255, 0], [10, 10, 10], [200, 200, 200]],
                  width=[2, 2, 1, 1, 1])


def test_streak_renders_colour_along_the_path_and_round_trips():
    s = _streak()
    lay = np.zeros((20, 30, 4), np.float32)
    apply_ops(lay, [s])
    # full coverage on the path, half a pixel either side of a 2 px stroke
    assert lay[5, 6, 3] == pytest.approx(1.0)
    assert lay[4, 6, 3] == pytest.approx(0.5, abs=0.01)
    # the colour moves from red to blue along the first segment
    r = lay[5, 2:13, 0]
    b = lay[5, 2:13, 2]
    assert r[0] > 0.95 and b[-1] > 0.95 and np.all(np.diff(r) <= 1e-6)
    s2 = ops_from_jsonable(ops_to_jsonable([s]))[0]
    lay2 = np.zeros_like(lay)
    apply_ops(lay2, [s2])
    assert np.array_equal(lay, lay2)
    assert check_mark_rules([s]) == []
    assert level_counts([s]) == {2: 1, 1: 1} and mark_level(s) == 1
    x, y = op_anchor(s)
    assert 2 < x < 23 and 5 < y < 11


def test_streak_rules_reject_dabs_and_per_pixel_colour():
    short = Streak(pts=[[1.5, 1.5], [2.5, 1.5]], n=[2], colors=[[0, 0, 0], [9, 9, 9]], width=1.0)
    assert any("shorter than 2 px" in v for v in check_mark_rules([short]))
    dense = Streak(pts=[[1.5 + i, 1.5] for i in range(6)], n=[6], colors=[[i * 40] * 3 for i in range(6)], width=1.0)
    assert any("stations closer" in v for v in check_mark_rules([dense]))
    with pytest.raises(ValueError):
        Streak(pts=[[1.5, 1.5]], n=[1], colors=[[0, 0, 0]], width=1.0)


def _stripes(h=96, w=96):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = 120 + 80 * np.sin(xx / 3.0 + yy / 9.0)
    img[..., 1] = 90 + 60 * np.cos(yy / 4.0)
    img[..., 2] = (xx * 2) % 255
    return img


def test_streaks_lower_the_residual_with_few_stations_per_px():
    ref = _stripes()
    canvas = np.full_like(ref, 128)
    st = {}
    ops = streak_ops(ref, None, canvas, width=1.0, stats=st)
    assert ops and check_mark_rules(ops) == []
    lay = np.zeros(ref.shape[:2] + (4,), np.float32)
    lay[..., :3] = canvas / 255.0
    lay[..., 3] = 1.0
    apply_ops(lay, ops)
    out = (np.clip(lay[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
    assert delta_e(ref, out).mean() < 0.5 * delta_e(ref, canvas).mean()
    # a stroke carries several pixels per colour station
    assert st["stations"] < 0.6 * (delta_e(ref, canvas) > 1.5).sum()


def test_streaks_on_a_region_are_owned_and_inside():
    ref = _stripes()
    m = np.zeros(ref.shape[:2], bool)
    m[:, :48] = True
    ops = streak_ops(ref, m, np.full_like(ref, 128), width=1.0)
    assert ops and all(o.own for o in ops)
    mk = np.repeat(np.arange(len(ops[0].n)), ops[0].n)
    cx = np.bincount(mk, weights=ops[0].pts[:, 0]) / ops[0].n
    assert (cx < 48).all()


def test_work_scale_delivers_the_reference_size_and_tile_retouch(tmp_path):
    ref = _stripes(64, 80)
    r = P.run_reference_pipeline(ref, str(tmp_path), "cel", log=lambda m: None, max_level=1, work_scale=2,
                                 fine="streaks", retouch_share=0.25)
    assert r["work_scale"] == 2 and r["deliver"] == [80, 64] and r["size"] == [160, 128]
    from PIL import Image
    for f in ("final.png", "L2.png", "L1.png"):
        assert Image.open(tmp_path / f).size == (80, 64)
    lv = {str(x["level"]): x for x in r["levels"]}
    assert lv["retouch"]["psnr"] >= lv["1"]["psnr"]
    assert P.DEFAULT_LEVEL == 2


def test_tile_retouch_paints_delivered_pixels():
    ref = _stripes(64, 64)
    img = np.full_like(ref, 128)
    ops, tiles = P.tile_retouch_ops(ref, img, 2, 0.25)
    assert tiles == 1 and len(ops) == 1
    lay = np.zeros((128, 128, 4), np.float32)
    lay[..., :3] = 128 / 255.0
    lay[..., 3] = 1
    apply_ops(lay, ops)
    import cv2
    out = cv2.resize((lay[..., :3] * 255 + 0.5).astype(np.uint8), (64, 64), interpolation=cv2.INTER_AREA)
    sel = np.zeros((64, 64), bool)
    xs, ys = ops[0].x // 2, ops[0].y // 2
    sel[ys, xs] = True
    assert delta_e(ref, out)[sel].max() < 1.0


def test_scale_ops_lays_the_same_marks_twice_as_large():
    import cv2
    from pixelcanvas.ops import Fill, Spray, Stroke, mask_to_rle, scale_ops
    m = np.zeros((40, 40), bool)
    m[10:30, 12:28] = True
    ops = [Stroke(pts=[[5.5, 5.5], [30.5, 20.5]], width=3.0, color=(200, 30, 30), brush="round", seed=3),
           Fill(polygons=[np.array([[2, 30], [18, 30], [10, 38]], np.float32)], color=(20, 200, 40), feather=1.0),
           Spray(pts=[[20.0, 20.0]], size=8.0, colors=[[30, 30, 220]], opacity=0.6, mask_rle=mask_to_rle(m), feather=1.0)]
    small = np.zeros((40, 40, 4), np.float32)
    big = np.zeros((80, 80, 4), np.float32)
    apply_ops(small, ops)
    apply_ops(big, scale_ops(ops, 2, (80, 80)))
    red = cv2.resize(big, (40, 40), interpolation=cv2.INTER_AREA)
    assert np.abs(red[..., 3] - small[..., 3]).mean() < 0.03
    assert check_mark_rules(scale_ops(ops, 2, (80, 80))) == []


def test_stipple_dots_points_and_finish_sharpen_and_auto_scale(tmp_path):
    from pixelcanvas.stage_streaks import stipple_ops
    ref = np.full((40, 40, 3), 20, np.uint8)
    ref[10, 10] = ref[25, 30] = 240                       # two stars
    can = np.full((40, 40, 3), 20, np.uint8)
    ops = stipple_ops(ref, can, 2)
    assert len(ops) == 1 and len(ops[0].pts) == 2 and check_mark_rules(ops) == []
    assert mark_level(ops[0]) == 2
    lay = np.zeros((80, 80, 4), np.float32)
    lay[..., :3] = 20 / 255.0
    lay[..., 3] = 1
    apply_ops(lay, ops)
    import cv2
    out = cv2.resize((lay[..., :3] * 255 + 0.5).astype(np.uint8), (40, 40), interpolation=cv2.INTER_AREA)
    assert out[10, 10, 0] > 150 and out[0, 0, 0] == 20      # one round (the colour clips at white)
    soft = cv2.GaussianBlur(_stripes(64, 64), (0, 0), 1.0)
    sh = P.finish_sharpen(soft, _stripes(64, 64))
    Y = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    ry = Y(_stripes(64, 64))
    assert ((Y(sh) - ry) ** 2).mean() < ((Y(soft) - ry) ** 2).mean()        # the amount is fitted on the luminance
    assert [P.auto_work_scale(*s) for s in ((200, 200), (480, 640), (850, 478), (1178, 1600))] == [4, 3, 2, 2]
    r = P.run_reference_pipeline(_stripes(48, 60), str(tmp_path), "photo", log=lambda m: None, max_level=1,
                                 work_scale="auto", fine="streaks", stipple=2, deliver_filter="sharpen")
    assert r["work_scale"] == 4 and any(str(x["level"]) == "stipple" for x in r["levels"])


def test_contours_are_one_stroke_per_line_and_smooth_tol_paints_skies():
    from pixelcanvas.stage_streaks import contour_ops, smooth_mask
    from pixelcanvas.trace import to_lab
    import cv2
    ref = np.full((60, 80, 3), 230, np.uint8)
    cv2.line(ref, (5, 10), (75, 40), (20, 20, 40), 1, cv2.LINE_AA)          # one thin line
    can = np.full_like(ref, 230)
    can[::2] = ref[::2]                                                        # drawn with gaps
    ops = contour_ops(ref, None, can, max_width=3.0, min_len=4.0)
    assert len(ops) == 1 and len(ops[0].n) == 1 and check_mark_rules(ops) == []
    lay = np.zeros(ref.shape[:2] + (4,), np.float32)
    lay[..., :3] = can / 255.0
    lay[..., 3] = 1
    apply_ops(lay, ops)
    out = (np.clip(lay[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8)
    assert delta_e(ref, out).mean() < delta_e(ref, can).mean()
    # a sky: faint mottle under the round-8 tolerance; smooth_tol paints it
    rng = np.random.default_rng(1)
    sky = np.clip(np.full((64, 64, 3), (150, 180, 230), np.float32) + cv2.GaussianBlur(
        rng.normal(0, 4, (64, 64, 1)).repeat(3, 2).astype(np.float32), (0, 0), 2), 0, 255).astype(np.uint8)
    flat = np.full_like(sky, sky.reshape(-1, 3).mean(0).astype(np.uint8))
    assert smooth_mask(to_lab(sky)).mean() > 0.9
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)       # a sky has no sharp step: no float32 overflow
        n0 = sum(len(o.n) for o in streak_ops(sky, None, flat, width=1.0))
        ops1 = streak_ops(sky, None, flat, width=1.0, smooth_tol=0.5)
    n1 = sum(len(o.n) for o in ops1)
    assert n1 > n0
    assert all(np.isfinite(o.width).all() and (o.width <= 1.0).all() for o in ops1)
