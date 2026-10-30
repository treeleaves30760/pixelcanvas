# -*- coding: utf-8 -*-
"""pixelcanvas 的快速測試（pytest -q，約 10 秒）。也可直接 python tests/test_pixelcanvas.py。"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import *  # noqa: E402
from pixelcanvas import scenery  # noqa: E402


def test_rgb_parsing():
    assert np.allclose(rgb("#ff8000"), [1.0, 128 / 255, 0.0])
    assert np.allclose(rgb("#abc"), rgb("#aabbcc"))
    assert np.allclose(rgb((255, 128, 0)), [1.0, 128 / 255, 0.0])      # 0~255 整數
    assert np.allclose(rgb((1, 1, 1)), [1, 1, 1])                       # 0~1
    assert np.allclose(rgb((1.6, 1.4, 1.2)), [1.6, 1.4, 1.2])           # HDR 不被除以 255


def test_lerp_mix_shapes():
    t = np.zeros((4, 5), F32)
    assert lerp("#000", "#fff", t).shape == (4, 5, 3)
    assert mix([(0, "#000"), (1, "#fff")], t).shape == (4, 5, 3)
    assert np.allclose(mix([(0, "#000"), (1, "#fff")], 0.5), [0.5, 0.5, 0.5], atol=1e-3)
    assert np.allclose(hsv_to_rgb(0.0, 1, 1), [1, 0, 0])


def test_smoothstep_array_edges():
    e1 = np.array([[1.0], [2.0]], F32)
    out = smoothstep(0.0, e1, np.ones((2, 3), F32))
    assert np.allclose(out[0], 1.0) and np.allclose(out[1], 0.5)


def test_noise_ranges_and_determinism():
    n = fbm2d(64, 96, 24, 5, seed=3)
    assert n.shape == (64, 96) and -1 <= n.min() and n.max() <= 1
    r = fbm2d(64, 96, 24, 5, seed=3, mode="ridged")
    assert 0 <= r.min() and r.max() <= 1
    assert np.array_equal(n, fbm2d(64, 96, 24, 5, seed=3))             # 同 seed 同結果
    assert not np.array_equal(n, fbm2d(64, 96, 24, 5, seed=4))
    p = fbm1d(300, 40, seed=2)
    assert p.shape == (300,) and abs(p).max() <= 1
    w = warp2d(32, 48, 16, 0.5, 4, seed=5, mode="billow")
    assert w.shape == (32, 48)


def test_noise_grid_matches_pointwise_and_offsets():
    X, Y = grid(40, 50)
    a = perlin2d(40, 50, 10, seed=1)
    assert np.allclose(a, perlin2d_at(X / 10, Y / 10, seed=1), atol=1e-5)
    b = perlin2d(40, 50, 10, seed=1, offset=(7, 0))
    assert np.allclose(a[:, 7:], b[:, :-7], atol=1e-5)                   # offset 只是平移


def test_masks_and_brushes():
    m = polygon_mask(50, 50, [(5, 5), (45, 10), (25, 45)])
    assert m.min() >= 0 and m.max() <= 1 and 0.2 < m.mean() < 0.5
    c = circle_mask(60, 60, 30, 30, 20)
    assert abs(c.sum() / (np.pi * 400) - 1) < 0.03
    k = np.zeros((50, 50), F32)
    stroke(k, [(5, 5), (45, 25), (10, 45)], 4, r_end=1)
    assert k.max() <= 1.0 and k.sum() > 50
    assert text_mask(30, 80, "Hi", (2, 2), 20).max() > 0.9
    assert band_mask(20, 10, 5, 15, 1)[10, 0] > 0.99


def test_blend_modes_and_filters():
    base = np.full((8, 8, 3), 0.5, F32)
    top = np.full((8, 8, 3), 0.5, F32)
    assert np.allclose(blend(base, top, 1, "add"), 1.0)
    assert np.allclose(blend(base, top, 1, "multiply"), 0.25)
    assert np.allclose(blend(base, top, 1, "screen"), 0.75)
    assert np.allclose(blend(base, "#ffffff", 0.5), 0.75)
    img = np.random.default_rng(0).random((32, 48, 3)).astype(F32)
    assert gaussian_blur(img, 2).shape == img.shape
    assert sample(img, np.full((2, 2), 10.5), np.full((2, 2), 5.5)).shape == (2, 2, 3)
    assert resize(img, 24, 16).shape == (16, 24, 3)
    assert radial_blur(img[..., 0], (24, 16), 0.5, 8, downscale=2).shape == (32, 48)
    assert soft_clip(np.array([0.5, 1.0, 3.0], F32)).max() <= 1.0 and soft_clip(np.array([0.5], F32))[0] == 0.5
    assert filmic(np.zeros((2, 2, 3), F32)).max() < 0.05


def test_canvas_roundtrip(tmp_path):
    cv = Canvas(40, 30, "#102030")
    cv.gradient([(0, "#000"), (1, "#fff")]).glow(20, 15, 5, (1.5, 1.2, 1.0))
    assert cv.buf.max() > 1.0                                            # HDR
    cv.put([1, 2], [3, 3], "#00ff00")
    assert np.allclose(cv.get(2, 3), [0, 1, 0])
    cv.soft_clip(0.8)
    p = tmp_path / "a.png"
    cv.save(str(p), preview=str(tmp_path / "a_small.jpg"), preview_width=20)
    back = Canvas.load(str(p))
    assert back.shape == (30, 40, 3) and np.abs(back.buf - clamp01(cv.buf)).max() < 2 / 255
    cv.paste(Canvas(10, 10, "#ff0000"), 35, 25)                          # 超出邊界會自動裁切
    assert np.allclose(cv.get(39, 29), [1, 0, 0])


def test_scene_scaling_and_mountain_recipe():
    sc = scenery.Scene(192, 108, seed=1)
    assert abs(sc.px(1080) - 108) < 1e-6
    sc.set_horizon(0.6)
    sc.set_sun(0.6, 0.55)
    sc.set_haze([(0, "#000"), (1, "#fff")])
    scenery.sky(sc, [(0, "#000"), (1, "#fff")])
    disc = scenery.sun(sc)
    assert disc.max() > 0.9
    m = scenery.mountain_layer(sc, dict(base=0.6, amp=0.2, scale=300, seed=3, fog=0.5, shadow="#000", lit="#fff", snow=0.5, forest=0.5))
    assert m.shape == (108, 192) and 0.2 < m.mean() < 0.8 and sc.terrain.max() == 1.0


def test_example_scene_renders_small():
    import importlib
    mod = importlib.import_module("scenes.mountains_sunset")
    cv = mod.render(192, 108, seed=5)
    assert cv.shape == (108, 192, 3)
    assert np.isfinite(cv.buf).all() and 0.15 < cv.buf.mean() < 0.7


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
