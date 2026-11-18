"""coords.py parsing / conversion and compose.py blend modes."""
import numpy as np
import pytest

from pixelcanvas import compose, coords
from pixelcanvas.coords import UsageError


@pytest.mark.parametrize("text", ["300,400", "300 400", "(300, 400)", "[300, 400]", "[300,400]", " 300 , 400 ",
                                  "x=300 y=400", "30%,40%", "0.3,0.4", "300.0,400.0"])
def test_point_variants(text):
    assert coords.parse_point(text) == pytest.approx((300, 400))


def test_points_and_json():
    pts = coords.parse_points("[[10,20],[30,40],[50,60]]")
    assert pts == [(10, 20), (30, 40), (50, 60)]
    assert coords.parse_points("10,20 30,40") == [(10, 20), (30, 40)]
    with pytest.raises(UsageError) as e:
        coords.parse_points("10,20 30")
    assert "odd" in e.value.message and e.value.example


def test_box_reorder_and_clamp_notes():
    notes = []
    assert coords.parse_box("[600, 500, 200, 100]", notes) == (200, 100, 600, 500)
    assert any("swapped" in n for n in notes)
    notes = []
    b = coords.parse_box("-20,0,1300,500", notes)
    assert b == (0, 0, 1000, 500)
    assert any("outside 0-1000" in n for n in notes)
    with pytest.raises(UsageError):
        coords.parse_box("1,2,3")
    with pytest.raises(UsageError):
        coords.parse_box("5,5,5,9")


def test_unit_pixel_round_trip():
    W, H = 850, 1200
    for u in (0, 1, 333, 500, 999, 1000):
        assert coords.px_to_unit(coords.unit_to_px(u, W), W) == u
    assert coords.box_to_px((0, 0, 1000, 1000), W, H) == (0, 0, W, H)
    bp = coords.box_to_px((100, 200, 300, 400), W, H)
    assert bp == (85, 240, 255, 480)
    assert coords.px_box_to_units(bp, W, H) == (100, 200, 300, 400)


@pytest.mark.parametrize("text,rgb", [("#ff0000", (255, 0, 0)), ("ff0000", (255, 0, 0)), ("#f00", (255, 0, 0)),
                                      ("red", (255, 0, 0)), ("Navy", (0, 0, 128)), ("rgb(1,2,3)", (1, 2, 3)),
                                      ("10,20,30", (10, 20, 30)), ("light grey", (211, 211, 211)),
                                      ("skin", (243, 214, 190))])
def test_colours(text, rgb):
    assert coords.parse_colour(text) == rgb


def test_colour_names_and_unknown():
    assert coords.colour_name((10, 10, 10)) == "black"
    assert coords.colour_name((250, 250, 250)) == "white"
    assert "red" in coords.colour_name((200, 30, 30))
    assert "blue" in coords.colour_name((40, 60, 200))
    with pytest.raises(UsageError):
        coords.parse_colour("blurple-ish")


def test_int_list_ranges():
    assert coords.parse_int_list("3,4,9") == [3, 4, 9]
    assert coords.parse_int_list("3-6, 9") == [3, 4, 5, 6, 9]
    assert coords.parse_int_list("[1, 2, 2]") == [1, 2]


# ----------------------------------------------------------------------------------------------- compose

def _layer(rgb, a, H=4, W=4):
    l = np.zeros((H, W, 4), np.float32)
    l[..., :3] = rgb
    l[..., 3] = a
    return l


def _stack(*entries):
    return [dict(name=n, mode=m, opacity=o, visible=v, clip=c) for n, m, o, v, c in entries]


@pytest.mark.parametrize("mode,expect", [("normal", 0.2), ("multiply", 0.5 * 0.2), ("screen", 0.5 + 0.2 - 0.1),
                                         ("overlay", 2 * 0.5 * 0.2), ("add", 0.7)])
def test_blend_modes(mode, expect):
    layers = {"a": _layer(0.5, 1.0), "b": _layer(0.2, 1.0)}
    st = _stack(("a", "normal", 1, True, False), ("b", mode, 1, True, False))
    out = compose.composite(st, layers, 4, 4)
    assert out[0, 0, 0] == pytest.approx(expect, abs=1e-5)


def test_opacity_visibility_and_clip():
    layers = {"a": _layer(0.0, 1.0), "b": _layer(1.0, 1.0)}
    out = compose.composite(_stack(("a", "normal", 1, True, False), ("b", "normal", 0.25, True, False)), layers, 4, 4)
    assert out[0, 0, 0] == pytest.approx(0.25, abs=1e-5)
    out = compose.composite(_stack(("a", "normal", 1, True, False), ("b", "normal", 1, False, False)), layers, 4, 4)
    assert out[0, 0, 0] == pytest.approx(0.0, abs=1e-5)
    # clip: the base covers only the left half; the clipped layer shows only there
    base = _layer(0.0, 0.0)
    base[:, :2, 3] = 1.0
    layers = {"base": base, "top": _layer(1.0, 1.0)}
    out = compose.composite(_stack(("base", "normal", 1, True, False), ("top", "normal", 1, True, True)),
                            layers, 4, 4, background=(0.5, 0.5, 0.5))
    assert out[0, 0, 0] == pytest.approx(1.0) and out[0, 3, 0] == pytest.approx(0.5)


def test_upto_and_skip():
    layers = {"a": _layer(0.3, 1.0), "b": _layer(0.9, 1.0)}
    st = _stack(("a", "normal", 1, True, False), ("b", "normal", 1, True, False))
    assert compose.composite(st, layers, 4, 4, upto="b")[0, 0, 0] == pytest.approx(0.3)
    assert compose.composite(st, layers, 4, 4, skip=["b"])[0, 0, 0] == pytest.approx(0.3)
