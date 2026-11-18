"""Blend op (a blending pass over paint already down) and the photo-grain amount."""
import numpy as np

from pixelcanvas.brush import apply_ops, new_layer
from pixelcanvas.ops import Blend, check_mark_rules, mask_to_rle, ops_from_jsonable, ops_to_jsonable
from pixelcanvas.pipeline import grain_amount
from pixelcanvas.stage_shade import blend_ops, smooth_area


def _banded():
    """A smooth ramp in the reference; the canvas has it in 6 flat bands, and a black bar (a hard edge)."""
    h, w = 120, 160
    x = np.arange(w, dtype=np.float32)
    ramp = 150 + 60 * x / w
    ref = np.repeat(np.repeat(ramp[None, :, None], h, 0), 3, 2)
    ref[:, 100:108] = 10
    canvas = ref.copy()
    canvas[:, :100] = np.round(ref[:, :100] / 10) * 10
    return np.clip(ref, 0, 255).astype(np.uint8), np.clip(canvas, 0, 255).astype(np.uint8)


def _layer(rgb):
    L = new_layer(*rgb.shape[:2])
    L[..., :3] = rgb / 255.0
    L[..., 3] = 1.0
    return L


def test_blend_softens_bands_keeps_edges_and_reads_only_the_layer():
    ref, canvas = _banded()
    ops = blend_ops(ref, None, radius=8.0, tol=30.0)
    assert len(ops) == 1 and isinstance(ops[0], Blend) and not check_mark_rules(ops)
    area = smooth_area(ref)
    assert not area[:, 99:101].any() and not area[:, 107:109].any()   # the bar's edges are not in the area
    L = _layer(canvas)
    apply_ops(L, ops)
    out = L[..., :3] * 255
    err_before = np.abs(canvas[20:100, 20:90].astype(float) - ref[20:100, 20:90]).mean()
    err_after = np.abs(out[20:100, 20:90] - ref[20:100, 20:90]).mean()
    assert err_after < 0.6 * err_before
    # the black bar stays black and crisp
    assert out[:, 101:107].max() < 12 and out[:, 112:].min() > 140
    # it reads the layer, not the reference: on an empty layer nothing appears
    E = new_layer(*ref.shape[:2])
    apply_ops(E, ops)
    assert E[..., 3].max() < 1e-6
    # exact JSON round trip
    back = ops_from_jsonable(ops_to_jsonable(ops))
    L2 = _layer(canvas)
    apply_ops(L2, back)
    assert np.allclose(L2, L)


def test_blend_respects_clip_and_area():
    ref, canvas = _banded()
    m = np.zeros(ref.shape[:2], bool)
    m[:, :50] = True
    op = Blend(mask_rle=mask_to_rle(m), radius=6.0, tol=30.0)
    L = _layer(canvas)
    apply_ops(L, [op])
    assert np.allclose(L[:, 60:, :3] * 255, canvas[:, 60:], atol=1e-3)
    L = _layer(canvas)
    clip = np.zeros(ref.shape[:2], np.float32)
    clip[:, :20] = 1
    apply_ops(L, [op], clip=clip)
    assert np.allclose(L[:, 22:, :3] * 255, canvas[:, 22:], atol=1e-3)


def test_grain_amount_follows_the_reference_noise():
    rng = np.random.default_rng(0)
    flat = np.full((200, 200, 3), 128, np.uint8)
    noisy = np.clip(128 + rng.normal(0, 6, (200, 200, 1)), 0, 255).astype(np.uint8).repeat(3, 2)
    assert grain_amount(flat, flat) == 0.0
    assert grain_amount(noisy, flat) > 0.2
    assert grain_amount(noisy, noisy) < 0.05
