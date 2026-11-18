"""Units of the no-reference drawing tools: geometry, colour, palette, mannequin, areas, shading, stamps."""
from __future__ import annotations

import math

import numpy as np

from pixelcanvas.easel import draw_core as dc
from pixelcanvas.easel import draw_detail, draw_plan, draw_shade, draw_sketch
from pixelcanvas.easel.ops import check_mark_rules


def test_catmull_rom_passes_through_points_and_is_dense():
    P = np.array([[0, 0], [50, 40], [100, 0], [150, 30]], float)
    c = dc.catmull_rom(P, spacing=1.5)
    for q in P:
        assert np.hypot(*(c - q).T).min() < 0.8
    assert np.hypot(*np.diff(c, axis=0).T).max() < 3.0


def test_taper_is_thin_at_the_ends_and_at_least_one_px():
    path = dc.resample(np.array([[0, 0], [200, 0]], float), 1.0)
    w = dc.taper_widths(path, 6.0, "both")
    assert w[0] < 2.5 and w[-1] < 2.5 and abs(w[len(w) // 2] - 6.0) < 1e-3 and w.min() >= 1.0
    w = dc.taper_widths(path, 6.0, "none")
    assert np.allclose(w, 6.0)


def test_oklch_round_trip_and_shadow_keeps_hue_family():
    for rgb in [(200, 60, 50), (60, 120, 200), (230, 210, 190), (128, 128, 128)]:
        L, C, h = dc.rgb_to_lch(rgb)
        back = dc.lch_to_rgb(L, C, h)
        assert max(abs(a - b) for a, b in zip(rgb, back)) <= 2
        sh = dc.shadow_of(rgb)
        assert dc.lum(sh) < L - 5
        li = dc.light_of(rgb)
        assert dc.lum(li) > L


def test_colour_words_are_tamed_and_modifiers_work():
    red = draw_plan.colour_words("red")
    assert red != (255, 0, 0)
    pale = draw_plan.colour_words("pale blue")
    L, C, h = dc.rgb_to_lch(pale)
    assert L > 85 and 200 < h < 280          # light and still blue (CIE Lab would turn it purple)
    assert draw_plan.colour_words("#102030") == (16, 32, 48)


def test_request_colours_find_named_parts():
    got = draw_plan.request_colours("a girl with long silver hair, a red scarf and a pale blue background",
                                    ["hair", "scarf", "background", "skin"])
    assert set(got) == {"hair", "scarf", "background"}


def test_default_swatch_keeps_hair_and_maps_sides():
    assert draw_plan.default_swatch("hair") == "hair"
    assert draw_plan.default_swatch("hand-l") == "skin"
    assert draw_plan.default_swatch("eye-r") == "eye-white"
    assert draw_plan.default_focus("face") == 3 and draw_plan.default_focus("scarf") == 2


def test_light_words_and_degrees():
    assert dc.parse_light("top-left")[0] == 135
    assert dc.parse_light("45")[0] == 45
    v = dc.light_vec({"light": {"deg": 135, "z": 0.6}})
    assert v[0] < 0 and v[1] < 0                  # up and to the left on screen


def test_figure_proportions_follow_head_count():
    for heads in (2.5, 7.0):
        J = draw_sketch.figure_joints([300, 50, 700, 950], heads, "stand")
        head_h = 2 * (J["chin"][1] - J["head"][1])
        total = J["lankle"][1] - (J["head"][1] - head_h / 2)
        assert abs(total / head_h - heads) < 1.0 + 0.15 * heads


class _P:
    W = H = 400
    state: dict = {"draw": {}}


def test_figure_report_flags_a_short_arm():
    p = _P()
    p.state = {"draw": {}}
    J = draw_sketch.figure_joints([300, 50, 700, 950], 7, "stand")
    J["lwrist"] = [J["lelbow"][0], J["lelbow"][1] + 20]
    el = {"id": "s1", "kind": "figure", "joints": J, "pose": "stand"}
    rep = " ".join(draw_sketch.figure_report(p, el))
    assert "shorter" in rep and "heads tall" in rep


def test_form_shadow_lies_away_from_the_light():
    m = np.zeros((200, 200), bool)
    yy, xx = np.mgrid[0:200, 0:200]
    m[(xx - 100) ** 2 + (yy - 100) ** 2 < 70 ** 2] = True
    light = dc.light_vec({"light": {"deg": 135, "z": 0.6}})     # from the top-left
    tones = dict(draw_shade.form_masks(m, light, levels=1))
    sh = tones["shadow"]
    ys, xs = np.nonzero(sh)
    assert xs.mean() > 100 and ys.mean() > 100                    # bottom-right
    assert 0.1 < sh.sum() / m.sum() < 0.6


def test_cast_shadow_moves_away_from_light():
    src = np.zeros((100, 100), bool)
    src[10:40, 10:90] = True
    dst = np.zeros((100, 100), bool)
    dst[30:90, 20:80] = True
    light = dc.light_vec({"light": {"deg": 90, "z": 0.6}})      # from the top
    m = draw_shade.cast_mask(src, dst, light, 12)
    ys, xs = np.nonzero(m)
    assert len(ys) and ys.min() >= 40 and ys.max() < 56


def test_stamps_and_patterns_obey_the_mark_rules():
    for kind in draw_detail.STAMPS:
        ops = draw_detail.scatter_ops(kind, mask=np.ones((100, 100), bool), count=8, size=10,
                                      colors=[(10, 20, 30)], seed=3)
        assert ops and not check_mark_rules(ops), kind
    for kind in ("hatch", "cross", "dots", "stripes", "checks"):
        ops = draw_detail.pattern_ops((0, 0, 100, 100), kind, (0, 0, 0), 12, 30, 2 if kind in ("hatch", "cross")
                                      else 4)
        assert ops and not check_mark_rules(ops), kind


def test_hair_strands_follow_the_flow():
    m = np.zeros((200, 200), bool)
    m[20:180, 60:140] = True
    flow = [np.array([[100, 20], [100, 180]], float)]
    light = dc.light_vec({"light": {"deg": 135, "z": 0.6}})
    ops = draw_detail.hair_ops(m, flow, (200, 200, 210), (120, 120, 140), (240, 240, 250), light, count=20,
                               width=2, seed=1)
    assert ops and not check_mark_rules(ops)
    # strands run mostly vertically
    dirs = [np.abs(np.diff(o.pts, axis=0)).sum(0) for o in ops if len(o.pts) > 3]
    assert np.mean([d[1] > d[0] for d in dirs]) > 0.8


def test_band_polygon_spans_the_canvas():
    poly = draw_detail.band_polygon(np.array([[0, 100], [200, 80], [400, 120]], float), 400, 300, "below")
    m = dc.poly_mask(300, 400, poly)
    assert m[290, 5] and m[290, 395] and not m[10, 200]
