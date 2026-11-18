"""``pixelcanvas polish``: the finishing pass of a picture drawn from a request (strays, coloured lines, depth, grade,
soft edges), one undoable step on the fx layer, each part switchable."""
from __future__ import annotations

import json

import numpy as np

from .test_draw_free_faults import draw, ok, project  # noqa: F401  (fixture and helpers)


def state(name="pic"):
    return json.load(open(f"{name}.pixelcanvas/state.json", encoding="utf-8"))


def lake(run):
    ok(run("new", "pic", "--size", "400x300", "--request", "a mountain lake at sunset with pines, painterly"))
    ok(run("do", "plan subject sky --box 0,0,1000,560; plan subject mountains --box 0,200,1000,560; "
                 "plan subject lake --box 0,540,1000,860; plan subject pines --box 0,150,330,1000; "
                 "plan order sky,mountains,lake,pines; plan light 15; palette make --base orange; palette pick 1"))
    ok(run("palette", "set", "sky", "#f4a66a", "mountains", "#91627a", "lake", "#6a5a8c", "pines", "#22382f"))
    ok(run("do", 'band sky "0,560 1000,560" --above --color sky; '
                 'band mountains "0,420 120,300 210,330 330,190 450,280 560,230 700,330 820,260 1000,380 1000,560 '
                 '0,560" --closed --color mountains; band lake "0,560 1000,560 1000,1000 0,1000" --closed --color lake; '
                 'band pines "120,990 120,250" --tiers 5 --width 200 --color pines'))


def lum(img):
    img = np.asarray(img, np.float64)
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def test_polish_sets_far_things_back_and_grades_toward_the_light(draw):
    lake(draw)
    p = project()
    before = p.composite_u8("export").astype(float)
    text = ok(draw("polish"))
    for part in ("strays:", "clean:", "lines:", "depth:", "grade:", "soften:"):
        assert part in text
    assert "VIEW" in text and "NEXT:" in text
    after = project().composite_u8("export").astype(float)
    mts = project().region_mask("mountains")
    pines = project().region_mask("pines")
    # the mountains (far) move toward the air: lighter; the pines (nearest) barely change
    d_m = (lum(after) - lum(before))[mts].mean()
    d_p = np.abs(after - before)[pines].mean()
    assert d_m > 2.0 and d_p < d_m
    assert state()["draw"].get("polished")
    # one undo restores the picture
    ok(draw("undo"))
    again = project().composite_u8("export").astype(float)
    assert np.abs(again - before).max() <= 1


def test_polish_parts_can_be_turned_off(draw):
    lake(draw)
    before = project().composite_u8("export").astype(float)
    text = ok(draw("polish", "--no-depth", "--no-grade", "--no-soften", "--no-lines", "--no-clean"))
    assert "depth:" not in text and "grade:" not in text
    after = project().composite_u8("export").astype(float)
    assert np.abs(after - before).max() <= 2              # only strays could change, and there are none
    code, err = draw("polish", "--no-depth", "--no-grade", "--no-soften", "--no-lines", "--no-strays", "--no-clean")
    assert code == 2 and "EXAMPLE:" in err


def test_polish_colours_line_art_and_removes_specks(draw):
    ok(draw("new", "pic", "--size", "400x400", "--request", "a red apple, cel"))
    ok(draw("style", "cel"))
    ok(draw("do", "plan subject apple --box 200,200,800,800; plan part body --of apple --box 200,200,800,800; "
                  "plan light top-left; palette make --base red; palette pick 1; "
                  'palette set body "#d03030"'))
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("line", "500,220 760,400 700,760 300,760 240,400", "--closed"))
    ok(draw("stroke", "100,100 103,101", "--width", "4", "--color", "#000000", "--layer", "line"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("flat", "auto"))
    p = project()
    line = p.layer("line")[..., 3] > 0.5
    img0 = p.composite_u8("export").astype(float)
    text = ok(draw("polish"))
    assert "1 specks" in text
    img1 = project().composite_u8("export").astype(float)
    # the outline takes a dark red, not a neutral black
    on = line.copy()
    on[:, :150] = False                                   # the outline, not the speck
    r, g, b = img1[on].mean(0)
    assert r > g + 8 and r > b + 8
    # the speck is gone: the corner shows the background again
    assert np.abs(img1[40, 40] - img1[20, 20]).max() < 30 and img0[40, 40].mean() < img1[40, 40].mean()
    assert "crisp edges kept" in text


def test_finish_todo_points_to_polish(draw):
    lake(draw)
    st = state()
    items = [t for t in st["todo"] if t["stage"] == "finish"]
    assert any(t["cmd"].startswith("pixelcanvas polish") for t in items)


def test_colour_checks_look_at_the_fills_not_the_finishing_pass(draw):
    from pixelcanvas import draw_core as dc
    lake(draw)
    before = dc.composite_u8(project(), "fills").copy()
    ok(draw("polish"))
    p = project()
    assert not np.array_equal(p.composite_u8("export"), before)
    assert np.array_equal(dc.composite_u8(p, "fills"), before)        # the grade is not a hole in the fills


def test_the_light_colour_is_a_light_swatch_else_a_warm_white(draw):
    from pixelcanvas import draw_core as dc
    from pixelcanvas import draw_finish
    lake(draw)
    img = project().composite_u8("export")
    # the sky low by the sun is the light's colour
    assert tuple(draw_finish.light_colour(project(), img)) == tuple(dc.from_hex(state()["draw"]["palette"]["swatches"]["sky-low"]))
    st = state()
    del st["draw"]["palette"]["swatches"]["sky-low"]
    json.dump(st, open("pic.pixelcanvas/state.json", "w", encoding="utf-8"))
    # a sky with no sun, lamp or light swatch: a warm white, not the sky's colour
    assert draw_finish.light_colour(project(), img) == draw_finish.WARM_WHITE
    for name in ("sun", "lamp", "light"):
        ok(draw("palette", "set", name, "#ffc070"))
        assert tuple(draw_finish.light_colour(project(), img)) == tuple(dc.from_hex("#ffc070"))
        st = state()
        del st["draw"]["palette"]["swatches"][name]
        json.dump(st, open("pic.pixelcanvas/state.json", "w", encoding="utf-8"))
