# -*- coding: utf-8 -*-
"""Fast tests for pixelcanvas (pytest -q, about 1 second). Can also be run directly: python tests/test_pixelcanvas.py."""
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
    assert np.allclose(rgb((255, 128, 0)), [1.0, 128 / 255, 0.0])      # integers 0~255
    assert np.allclose(rgb((1, 1, 1)), [1, 1, 1])                       # 0~1
    assert np.allclose(rgb((1.6, 1.4, 1.2)), [1.6, 1.4, 1.2])           # HDR is not divided by 255


def test_rgb_forms_and_names():
    assert np.allclose(rgb("#f80a"), rgb("#ff8800"))
    assert np.allclose(rgb("#ff8000cc"), rgb("#ff8000"))
    assert np.allclose(rgb("navy"), [0, 0, 128 / 255])
    assert np.allclose(rgb("Gold"), rgb("#ffd700"))
    assert np.allclose(rgb("grey"), rgb("gray"))
    for name in ("black white red green blue yellow orange purple pink brown gray cyan magenta navy teal gold "
                 "silver beige maroon olive lime indigo violet turquoise coral salmon khaki crimson lavender "
                 "tan ivory mint").split():
        assert rgb(name).shape == (3,)


def test_rgb_rejects_malformed():
    import pytest
    for bad in ("#ffb34", "#ggg", "ffb347", "#", "", "notacolour", "#ff00001"):
        with pytest.raises(ValueError) as e:
            rgb(bad)
        assert repr(bad) in str(e.value) and "#rrggbb" in str(e.value)


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
    assert np.array_equal(n, fbm2d(64, 96, 24, 5, seed=3))             # same seed, same result
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
    assert np.allclose(a[:, 7:], b[:, :-7], atol=1e-5)                   # offset is just a shift


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
    cv.paste(Canvas(10, 10, "#ff0000"), 35, 25)                          # out-of-bounds is clipped automatically
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




# ---- round r1: paint opacity position, tuple colour rule, mask helpers, sdf conventions ----
def test_paint_number_in_mode_slot_is_opacity():
    import pytest
    from pixelcanvas.masks import circle_mask
    m = circle_mask(40, 40, 20, 20, 10)
    a = Canvas(40, 40, "#000000").paint("#ffffff", m, 0.5)
    b = Canvas(40, 40, "#000000").paint("#ffffff", m, opacity=0.5)
    assert np.allclose(a.buf, b.buf) and abs(float(a.buf[20, 20, 0]) - 0.5) < 1e-3
    c = Canvas(40, 40, "#000000").paint("#ffffff", None, 0.25)
    assert abs(float(c.buf[0, 0, 0]) - 0.25) < 1e-3
    d = Canvas(40, 40, "#000000").radial(20, 20, 10, [(0, "#fff"), (1, "#fff")], m, 0.5)
    assert abs(float(d.buf[20, 20, 0]) - 0.5) < 1e-3
    with pytest.raises(TypeError, match="opacity twice"):
        Canvas(4, 4).paint("#fff", None, 0.5, opacity=0.3)
    with pytest.raises(ValueError, match="did you mean 'multiply'"):
        Canvas(4, 4).paint("#fff", None, "multipy")
    with pytest.raises(ValueError, match="did you mean 'lanczos'"):
        Canvas(4, 4).resize(2, 2, "lancos")
    with pytest.raises(ValueError, match="axis"):
        Canvas(4, 4).flip("z")


def test_tuple_colour_rule_floats_0_1_hdr_ints_0_255():
    import pytest
    from pixelcanvas.color import rgb
    from pixelcanvas.masks import circle_mask
    assert np.allclose(rgb((1.6, 0.62, 0.16)), (1.6, 0.62, 0.16))
    assert np.allclose(rgb((2.0, 1.0, 0.0)), (2.0, 1.0, 0.0))          # integer-valued floats stay 0~1 (HDR)
    assert np.allclose(rgb((255, 128, 0)), (1.0, 128 / 255, 0.0))
    assert np.allclose(rgb((1, 1, 1)), (1, 1, 1)) and np.allclose(rgb((0, 0, 0)), 0)
    assert np.allclose(rgb((0.5, 1, 0)), (0.5, 1, 0))                   # one float: all 0~1
    with pytest.raises(ValueError, match="0-255"):
        rgb((300, 0, 0))
    with pytest.raises(ValueError, match="3 numbers"):
        rgb((1.0, 0.5))
    cv = Canvas(30, 30, "#101010").paint((1.6, 0.62, 0.16), circle_mask(30, 30, 15, 15, 8), mode="add")
    assert cv.buf[15, 15, 0] > 1.6 and cv.buf[15, 15, 1] > 0.6 > cv.buf[15, 15, 2] + 0.3


def test_ellipse_shift_drop_shadow_masks():
    import pytest
    from pixelcanvas import masks
    e = masks.ellipse_mask(100, 200, 100, 50, 60, 20, angle=0)
    assert abs(float(e.sum()) - np.pi * 60 * 20) < 25 and e[50, 100] == 1 and e[50, 5] == 0
    r = masks.ellipse_mask(100, 200, 100, 50, 60, 20, angle=90)
    assert r[50 + 40, 100] > 0.9 and r[50, 100 + 40] == 0
    m = np.ones((10, 10), np.float32)
    s = masks.shift(m, 0, 3)
    assert s[:3].sum() == 0 and s[3:].min() == 1                         # zero fill, no wrap-around
    assert masks.shift(m, -12, 0).sum() == 0
    assert abs(masks.shift(m, 0.5, 0)[:, 3].mean() - 1) < 1e-6 and masks.shift(m, 0.5, 0)[:, 0].mean() == 0.5
    c = masks.circle_mask(80, 80, 40, 40, 12)
    sh = masks.drop_shadow(c, (6, 10), 4)
    assert sh.shape == c.shape and sh[40 + 14, 40 + 4] > c[40 + 14, 40 + 4]
    assert masks.ellipse_mask(10, 10, 5, 5, 3, 2).dtype == np.float32
    with pytest.raises(AttributeError, match="masks.shift"):
        masks.roll


def test_sdf_offset_sign_convention():
    from pixelcanvas import sdf
    X, Y = np.meshgrid(np.arange(80, dtype=np.float32), np.arange(80, dtype=np.float32))
    d = sdf.sd_circle(X, Y, 40, 40, 15)
    base = sdf.fill(d).sum()
    assert sdf.fill(sdf.offset(d, 5)).sum() > base > sdf.fill(sdf.offset(d, -5)).sum()
    assert "grow" in sdf.offset.__doc__.lower() and "shrink" in sdf.offset.__doc__.lower()
    ring = sdf.fill(sdf.subtract(sdf.offset(d, 4), d))
    assert ring[40, 40] == 0 and ring[40, 40 + 17] > 0.9


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))


def test_every_kit_module_imports_from_the_package():
    import importlib
    import pixelcanvas
    kits = ("spec", "clay", "pictograms", "drafting", "flora", "elements", "solid", "diagram", "chart", "texture",
            "vector", "layout", "stylize", "form", "figure", "creature", "scenery", "palette", "backdrop", "ui",
            "catalog", "cli", "mcp_server", "manifest", "layers", "vision", "sdf")
    for k in kits:
        mod = getattr(pixelcanvas, k)
        assert mod is importlib.import_module("pixelcanvas." + k), k
        ns = {}
        exec(f"from pixelcanvas import {k}", ns)
        assert ns[k] is mod
    assert pixelcanvas.__version__ == "0.4.0"
    assert all(hasattr(pixelcanvas, n) for n in pixelcanvas.__all__)
    assert set(pixelcanvas._LAZY) <= set(dir(pixelcanvas))


def test_paste_of_a_canvas_carries_its_manifest_entries():
    from pixelcanvas import manifest
    main = Canvas(200, 100, "#ffffff")
    sub = Canvas(60, 60, "#ffffff")
    sub.paint("#ff0000", circle_mask(60, 60, 30, 30, 20))
    manifest.note(sub, "object", "apple", bbox=(10, 10, 50, 50))
    main.paste(sub, 100, 20)
    names = [e["name"] for e in manifest.entries(main)]
    assert "apple" in names
    e = [e for e in manifest.entries(main) if e["name"] == "apple"][0]
    assert 0.5 < e["bbox"][0] < 0.65 and e["bbox"][1] < 0.5
