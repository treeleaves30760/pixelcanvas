"""Detail levels (DESIGN.md 10.1): the levels add up to the reference (a complete basis), each level lowers the
residual of the one above it, the retouch op round-trips through JSON, region calls match the whole-picture call."""
import os

import numpy as np
import pytest
from PIL import Image

from pixelcanvas.brush import apply_ops, new_layer
from pixelcanvas.ops import Retouch, check_mark_rules, level_counts, mark_level, ops_from_jsonable, ops_to_jsonable
from pixelcanvas.pipeline import fidelity_metrics, run_reference_pipeline
from pixelcanvas.stage_levels import hairline_ops, pixel_ops, residual_de

from .test_hand_local import mixed_ref
from .test_hand_stages import cel_ref

# a real picture for the completeness test: any image file, named by the environment (the test skips without one)
REAL = os.environ.get("PIXELCANVAS_REAL_REF", "")


def _paint(canvas, ops):
    lay = new_layer(*canvas.shape[:2])
    lay[..., :3] = canvas.astype(np.float32) / 255.0
    lay[..., 3] = 1.0
    apply_ops(lay, ops)
    return np.clip(np.rint(lay[..., :3] * 255), 0, 255).astype(np.uint8)


def _complete(ref, out):
    m = fidelity_metrics(ref, out)
    assert m["mean_dE"] < 0.5 and m["max_dE"] < 3.0 and m["psnr"] > 45.0, m
    return m


def test_retouch_op_json_render_and_levels():
    r = Retouch(x=[1, 6, 0], y=[1, 2, 5], n=[4, 2, 1], axis=[0, 1, 0], size=[1, 2, 1],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    assert check_mark_rules([r]) == [] and mark_level(r) == 0 and level_counts([r]) == {1: 2, 0: 1}
    back = ops_from_jsonable(ops_to_jsonable([r]))[0]
    img = _paint(np.full((10, 10, 3), 255, np.uint8), [r])
    assert np.array_equal(img, _paint(np.full((10, 10, 3), 255, np.uint8), [back]))
    assert tuple(img[1, 1:5].reshape(-1, 3)[:, 0]) == (255, 255, 255, 255) and img[1, 1, 1] == 0
    assert (img[2:4, 6:8] == (0, 255, 0)).all() and tuple(img[5, 0]) == (0, 0, 255)


def test_levels_one_and_zero_complete_a_synthetic_picture():
    ref = mixed_ref()
    canvas = np.full_like(ref, 128)
    h = hairline_ops(ref, None, canvas)
    c1 = _paint(canvas, h)
    assert residual_de(ref, c1).mean() < residual_de(ref, canvas).mean() * 0.5   # hairlines carry most of it
    p = pixel_ops(ref, None, c1)
    _complete(ref, _paint(c1, p))


def test_pipeline_to_level_zero_is_complete_and_coarse_to_fine(tmp_path):
    ref = cel_ref()
    rep = run_reference_pipeline(ref, str(tmp_path), "cel", log=lambda *a: None, max_level=0)
    out = np.asarray(Image.open(tmp_path / "final.png").convert("RGB"))
    _complete(ref, out)
    errs = [lv["mean_dE"] for lv in rep["levels"]]
    assert [lv["level"] for lv in rep["levels"]] == [2, 1, 0] and errs[0] >= errs[1] >= errs[2]
    assert (tmp_path / "L2.png").exists() and (tmp_path / "L1.png").exists()
    assert set(rep["level_touched"]) >= {"0", "1"}


def test_default_level_is_the_plan(tmp_path):
    rep = run_reference_pipeline(cel_ref(), str(tmp_path), "cel", log=lambda *a: None)
    assert rep["max_level"] == 2 and rep["levels"] == [] and "0" not in rep["level_marks"]


@pytest.mark.skipif(not REAL or not os.path.isfile(REAL),
                    reason="set PIXELCANVAS_REAL_REF to the path of a real picture to run this test")
def test_real_crop_is_complete_at_level_zero(tmp_path):
    from pixelcanvas.pipeline import load_ref
    full = load_ref(REAL)
    H, W = full.shape[:2]
    y0, x0 = (200, 300) if H >= 360 and W >= 460 else (max(0, H // 2 - 80), max(0, W // 2 - 80))
    ref = np.ascontiguousarray(full[y0:y0 + 160, x0:x0 + 160])        # a 160 px crop
    run_reference_pipeline(ref, str(tmp_path), "cel", log=lambda *a: None, max_level=0)
    out = np.asarray(Image.open(tmp_path / "final.png").convert("RGB"))
    _complete(ref, out)


def test_fine_levels_on_regions_match_the_whole_call():
    ref = mixed_ref()
    canvas = np.full_like(ref, 120)
    whole = _paint(canvas, hairline_ops(ref, None, canvas))
    yy, xx = np.mgrid[0:160, 0:160]
    parts = [xx < 70 + 10 * np.sin(yy / 9.0), None]
    parts[1] = ~parts[0]
    c = canvas.copy()
    for m in parts:
        c = _paint(c, hairline_ops(ref, m, canvas))     # each region from the same residual, as one call would
    assert float(residual_de(whole, c).mean()) < 0.05
    c0 = c.copy()
    for m in parts:
        c0 = _paint(c0, pixel_ops(ref, m, c0))
    _complete(ref, c0)
