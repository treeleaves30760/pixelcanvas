"""The local character map (stage_local.py) and the treatments that consult it."""
import numpy as np

from pixelcanvas import stage_local as SL
from pixelcanvas.ops import Stroke, check_mark_rules
from pixelcanvas.pipeline import add_grain_local, grain_local
from pixelcanvas.stage_paint import paint_ops, texture_ops


def mixed_ref(seed=0):
    """Four quadrants: a flat colour with a hard-edged square, a smooth ramp, stripes (texture), noisy flat (grain)."""
    rng = np.random.default_rng(seed)
    h = w = 160
    img = np.zeros((h, w, 3), np.float32)
    img[:80, :80] = (40, 90, 160)
    img[20:60, 20:60] = (230, 220, 200)
    x = np.linspace(0, 1, 80)[None, :, None]
    img[:80, 80:] = (1 - x) * np.array([200, 120, 90]) + x * np.array([120, 150, 210])
    yy, xx = np.mgrid[0:80, 0:80]
    stripes = 110 + 45 * np.sin((xx + 0.6 * yy) / 1.6) * (rng.random((80, 80)) * 0.6 + 0.7)
    img[80:, :80] = np.stack([stripes, stripes * 0.8, stripes * 0.5], -1)
    img[80:, 80:] = (150, 150, 150)
    img[80:, 80:] += rng.normal(0, 6.0, (80, 80, 1))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _share(cm, cls, ys, xs):
    return float((cm.cls[ys, xs] == cls).mean())


def test_char_map_classes_and_cache():
    ref = mixed_ref()
    cm = SL.char_map(ref)
    assert SL.char_map(ref.copy()) is cm                  # cached by the picture's bytes
    inner = (slice(8, 72), slice(8, 72))
    assert _share(cm, SL.FLAT, *inner) > 0.8              # flat with a hard-edged square
    assert _share(cm, SL.RAMP, slice(8, 72), slice(90, 150)) > 0.8
    assert _share(cm, SL.TEXTURE, slice(90, 150), slice(8, 72)) > 0.8
    noisy = cm.cls[90:150, 90:150]
    assert float(np.isin(noisy, (SL.FLAT, SL.RAMP)).mean()) > 0.8    # grain is not texture
    assert float(cm.grain[90:150, 90:150].mean()) > 1.0
    assert float(cm.grain[10:70, 10:70].max()) < 0.05                 # the clean flat carries no grain
    assert SL.summary_style(ref) in ("flat", "ramp", "texture")


def test_local_grain_only_where_the_reference_has_grain():
    ref = mixed_ref()
    canvas = np.asarray(ref, np.float32)
    canvas[80:, 80:] = 150                                 # painted smooth: the grain is missing there
    canvas = canvas.astype(np.uint8)
    amap, size = grain_local(ref, canvas)
    assert amap is not None and size
    assert float(amap[95:145, 95:145].mean()) > 0.1
    assert float(amap[10:60, 10:60].max()) == 0.0
    out = add_grain_local(canvas, amap, size)
    assert np.array_equal(out[10:60, 10:60], canvas[10:60, 10:60])
    assert float(np.std(out[95:145, 95:145].astype(np.float32))) > 2.0


def test_textured_marks_only_in_textured_areas():
    ref = mixed_ref()
    canvas = np.full_like(ref, 128)
    cm = SL.char_map(ref)
    ops = paint_ops(ref, None, canvas, size=4, brush="round", structure=True, tips="auto", budget=None)
    assert ops and not check_mark_rules(ops)
    for o in ops:
        if isinstance(o, Stroke) and o.brush in ("bristle", "sponge"):
            x, y = np.clip(np.asarray(o.pts)[0].astype(int), 0, 159)
            assert cm.cls[y, x] == SL.TEXTURE
    tex = texture_ops(ref, None, canvas)
    for o in tex:
        p = np.asarray(o.pts).astype(int)
        assert float((cm.cls[np.clip(p[:, 1], 0, 159), np.clip(p[:, 0], 0, 159)] == SL.TEXTURE).mean()) >= 0.5


def test_local_detail_ends_with_crisp_shapes():
    ref = mixed_ref()
    canvas = np.full_like(ref, 128)
    st = {}
    ops = paint_ops(ref, None, canvas, size=4, brush="round", structure=True, tips="local", budget=None, stats=st)
    assert not check_mark_rules(ops)
    assert "shapes" in st
    strokes = paint_ops(ref, None, canvas, size=4, brush="round", structure=True, tips="local-strokes", budget=None)
    n = len(strokes)
    # the strokes first, then the shape pass's fills (crisp work last)
    assert all(isinstance(o, Stroke) for o in ops[:n])
    assert any(type(o).__name__ == "Fill" for o in ops[n:])
