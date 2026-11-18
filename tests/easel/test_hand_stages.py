"""Stage functions on synthetic references: each returns ops that move the canvas toward the reference."""
import time

import cv2
import numpy as np
import pytest

from pixelcanvas.easel.brush import apply_ops, new_layer
from pixelcanvas.easel.ops import Fill, Stroke, ops_from_jsonable, ops_to_jsonable
from pixelcanvas.easel.pipeline import composite, guess_style, quick_scores, run_reference_pipeline
from pixelcanvas.easel.stage_flat import flat_ops
from pixelcanvas.easel.stage_ink import ink_ops
from pixelcanvas.easel.stage_paint import paint_ops
from pixelcanvas.easel.stage_shade import shade_ops
from pixelcanvas.easel.stage_sketch import sketch_ops
from pixelcanvas.easel.trace import delta_e, line_map, skeleton_polylines


def cel_ref(h=240, w=200):
    """White paper, a pink disc with a darker shadow crescent, a blue box, black 3 px outlines."""
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.circle(img, (90, 100), 60, (240, 170, 190), -1, cv2.LINE_AA)
    cv2.circle(img, (110, 115), 50, (200, 120, 150), -1, cv2.LINE_AA)
    cv2.circle(img, (90, 100), 60, (240, 170, 190), -1, cv2.LINE_AA) if False else None
    shadow = np.zeros((h, w), np.uint8)
    cv2.circle(shadow, (90, 100), 60, 1, -1)
    lit = np.zeros((h, w), np.uint8)
    cv2.circle(lit, (75, 85), 52, 1, -1)
    sel = (shadow > 0) & (lit == 0)
    img[(shadow > 0) & (lit > 0)] = (240, 170, 190)
    img[sel] = (190, 110, 140)
    cv2.rectangle(img, (120, 170), (185, 225), (60, 90, 200), -1)
    cv2.circle(img, (90, 100), 60, (20, 20, 30), 3, cv2.LINE_AA)
    cv2.rectangle(img, (120, 170), (185, 225), (20, 20, 30), 3, cv2.LINE_AA)
    cv2.line(img, (20, 200), (100, 230), (20, 20, 30), 2, cv2.LINE_AA)
    return img


def paint_ref(h=200, w=240):
    """Smooth gradient sky over a textured green field (no lines)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    t = y / h
    img[..., 0] = 90 + 80 * t
    img[..., 1] = 150 + 60 * t
    img[..., 2] = 230 - 20 * t
    rng = np.random.default_rng(0)
    field = y > 110 + 10 * np.sin(x / 30)
    tex = cv2.GaussianBlur(rng.normal(0, 18, (h, w)).astype(np.float32), (0, 0), 1.2)
    img[field] = np.stack([70 + tex, 140 + 1.3 * tex, 50 + 0.5 * tex], -1)[field]
    return np.clip(img, 0, 255).astype(np.uint8)


def _err(ref, img):
    return float(delta_e(ref, img).mean())


def _paper(ref):
    L = new_layer(*ref.shape[:2])
    L[..., :3] = 1.0
    L[..., 3] = 1.0
    return L


def test_line_map_and_skeleton():
    ref = cel_ref()
    lm = line_map(ref)
    assert 2.0 <= lm.width <= 5.0
    assert lm.mask[100, 30] or lm.mask[100, 31] or lm.mask[100, 29]  # left of the circle outline
    assert not lm.mask[100, 90]
    polys = skeleton_polylines(lm.mask)
    assert len(polys) >= 2
    for pts, w, _ in polys:
        assert pts.shape[1] == 2 and len(w) == len(pts)


def test_ink_ops_trace_lines():
    ref = cel_ref()
    ops = ink_ops(ref, None)
    assert ops and all(isinstance(o, Stroke) for o in ops)
    L = _paper(ref)
    ink = new_layer(*ref.shape[:2])
    apply_ops(ink, ops)
    img = composite({"paper": L, "line": ink}, order=["paper", "line"])
    lm_ref = line_map(ref).mask
    lm_img = line_map(img).mask
    k = np.ones((5, 5), np.uint8)
    rec = (lm_ref & (cv2.dilate(lm_img.astype(np.uint8), k) > 0)).sum() / lm_ref.sum()
    prec = (lm_img & (cv2.dilate(lm_ref.astype(np.uint8), k) > 0)).sum() / max(lm_img.sum(), 1)
    assert rec > 0.85 and prec > 0.85
    # pure and deterministic
    assert len(ink_ops(ref, None)) == len(ops)
    # mask limits the strokes
    m = np.zeros(ref.shape[:2], bool)
    m[:, :100] = True
    sub = ink_ops(ref, m)
    # a region call keeps the lines anchored in the region, each one whole
    from pixelcanvas.easel.ops import op_anchor
    assert sub and all(op_anchor(o)[0] < 100 and o.own for o in sub)


def test_flat_ops_reduce_error_and_sit_under_lines():
    ref = cel_ref()
    lm = line_map(ref)
    ops = flat_ops(ref, None, line_mask=lm.mask)
    assert ops and all(isinstance(o, Fill) for o in ops)
    paper = _paper(ref)
    before = composite({"paper": paper}, order=["paper"])
    flat = new_layer(*ref.shape[:2])
    apply_ops(flat, ops)
    after = composite({"paper": paper, "flat": flat}, order=["paper", "flat"])
    assert _err(ref, after) < 0.5 * _err(ref, before)
    # under the line art the flats are opaque (no paper gap between line and colour)
    assert flat[..., 3][lm.mask].min() > 0.99
    # few colours, flat areas
    assert len(ops) < 30


def test_shade_ops_add_shadow_shape():
    ref = cel_ref()
    # base: flats without the shadow crescent
    base = ref.copy()
    sel = np.all(ref == (190, 110, 140), -1)
    base[sel] = (240, 170, 190)
    ops = shade_ops(ref, None, base)
    assert ops
    layer = new_layer(*ref.shape[:2])
    apply_ops(layer, ops)
    out = composite({"b": np.dstack([base / 255.0, np.ones(ref.shape[:2])]).astype(np.float32), "s": layer},
                    order=["b", "s"])
    assert _err(ref, out) < 0.6 * _err(ref, base)


def test_paint_ops_coarse_to_fine():
    ref = paint_ref()
    canvas = np.full_like(ref, 255)
    layer = _paper(ref)
    errs = [_err(ref, canvas)]
    for i, size in enumerate((24, 10, 4, 2)):
        ops = paint_ops(ref, None, canvas, size=size, brush="flat" if size > 8 else "round", seed=i)
        assert ops
        apply_ops(layer, ops)
        canvas = composite({"p": layer}, order=["p"])
        errs.append(_err(ref, canvas))
    assert errs == sorted(errs, reverse=True)
    assert errs[-1] < 4.0
    # strokes are curves along the forms, not dabs: most large strokes have several points
    big = paint_ops(ref, None, np.full_like(ref, 255), size=16, seed=1)
    assert np.mean([len(o.pts) >= 3 for o in big]) > 0.5


def test_paint_ops_only_where_needed():
    ref = paint_ref()
    assert paint_ops(ref, None, ref.copy(), size=2) == []
    m = np.zeros(ref.shape[:2], bool)
    m[:50, :50] = True
    ops = paint_ops(ref, m, np.full_like(ref, 255), size=6)
    # on a region the strokes start inside it and are painted whole (they may run on a little past its edge)
    assert ops and all(o.own and (o.pts.min(0) < 50).all() for o in ops)
    assert all(o.pts[:, 0].max() < 50 + 6 * 16 and o.pts[:, 1].max() < 50 + 6 * 16 for o in ops)


def test_sketch_ops_pencil():
    ref = cel_ref()
    ops = sketch_ops(ref, None, seed=3)
    assert ops and all(o.brush == "pencil" for o in ops)
    assert ops_to_jsonable(ops) == ops_to_jsonable(sketch_ops(ref, None, seed=3))


def test_stage_ops_round_trip():
    ref = cel_ref()
    ops = ink_ops(ref, None) + flat_ops(ref, None)
    back = ops_from_jsonable(ops_to_jsonable(ops))
    a, b = new_layer(*ref.shape[:2]), new_layer(*ref.shape[:2])
    apply_ops(a, ops)
    apply_ops(b, back)
    assert np.array_equal(a, b)


def test_guess_style():
    assert guess_style(cel_ref(480, 400)) in ("cel", "ink", "poster")
    nat = paint_ref(400, 480).astype(np.float32)
    nat += np.random.default_rng(1).normal(0, 4, nat.shape)  # photographic grain everywhere
    assert guess_style(np.clip(nat, 0, 255).astype(np.uint8)) in ("paint", "photo")
    gray = np.repeat(cel_ref(480, 400).mean(-1, keepdims=True), 3, -1).astype(np.uint8)
    assert guess_style(gray) == "ink"


def test_pipeline_runs(tmp_path):
    ref = cel_ref()
    r = run_reference_pipeline(ref, str(tmp_path), "cel", log=lambda *a: None)
    assert (tmp_path / "final.png").exists() and (tmp_path / "01-sketch.png").exists()
    assert r["final"]["mean_dE"] < 2.5
    assert [s["stage"] for s in r["stages"]] == ["sketch", "line", "flat", "shade", "blend", "detail", "finish"]


def test_stage_speed_850x1200():
    ref = cv2.resize(cel_ref(), (850, 1200), interpolation=cv2.INTER_LINEAR)
    for fn in (lambda: ink_ops(ref, None), lambda: flat_ops(ref, None), lambda: sketch_ops(ref, None)):
        t = time.perf_counter()
        fn()
        assert time.perf_counter() - t < 20.0
