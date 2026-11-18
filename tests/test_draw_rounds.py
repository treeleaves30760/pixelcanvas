"""Fixes from the four trial pictures: eyes, zigzag shapes, area picking by place, re-shading, light from a point,
landscape palettes and haze, exact stamps, lettering parts, and the workflow hook passing arguments through."""
from __future__ import annotations

import json

import numpy as np
from pixelcanvas import workflow
from pixelcanvas import draw_core as dc
from pixelcanvas import draw_line, draw_plan

from .test_draw_flow import draw, ok, setup_bust  # noqa: F401  (fixture and helpers)


def state():
    return json.load(open("pic.pixelcanvas/state.json", encoding="utf-8"))


def test_corner_spline_keeps_sharp_points():
    zig = [[0, 0], [10, 20], [20, 0], [30, 20], [40, 0]]
    path = dc.corner_spline(zig, closed=False, spacing=1.0)
    # the tips are on the path itself (a smooth spline would round them off)
    for q in zig[1:-1]:
        assert np.min(np.hypot(*(path - np.array(q, float)).T)) < 0.75
    smooth = dc.corner_spline([[0, 0], [10, 3], [20, 4], [30, 3], [40, 0]], spacing=1.0)
    assert len(smooth) > 10


def test_eye_shape_mirrors_and_closes():
    loop_l, lash_l = draw_line.eye_shape((400, 300, 480, 360), outer_left=True)
    loop_r, lash_r = draw_line.eye_shape((520, 300, 600, 360), outer_left=False)
    assert len(loop_l) == len(loop_r) == 9 and len(lash_l) == 5
    # the outer corner sits higher than the inner one on both eyes
    assert loop_l[4][1] < loop_l[0][1] and loop_r[4][1] < loop_r[0][1]
    assert loop_l[4][0] < loop_l[0][0] and loop_r[4][0] > loop_r[0][0]


def test_material_names_start_from_their_colour():
    d = {"subjects": {"sky": {"box": [0, 0, 1000, 500]}, "pines": {"box": [0, 200, 300, 1000]}}, "parts": {}}
    names = draw_plan.needed_swatches(d)
    assert "sky" in names and "pines" in names
    cands = draw_plan.make_candidates(d, (240, 138, 60))
    skies = [c["swatches"]["sky"] for c in cands]
    assert len(set(skies)) == 4                     # four palettes differ
    r, g, b = dc.from_hex(cands[0]["swatches"]["pines"]) if hasattr(dc, "from_hex") else \
        draw_plan.from_hex(cands[0]["swatches"]["pines"])
    assert g >= r                                   # pines stay green-ish


def test_line_eye_and_trace_head(draw):
    setup_bust(draw)
    ok(draw("do", "plan part eye-l --box 400,300,470,350; plan part eye-r --box 530,300,600,350; stage done; "
                  "sketch figure girl --pose bust; sketch head girl"))
    ok(draw("stage", "done"))
    text = ok(draw("line", "eye", "eye-l,eye-r"))
    assert "4 marks" in text or "line eye" in text
    lines = state()["draw"]["lines"]
    assert len(lines) == 4 and sum(1 for r in lines if r["closed"]) == 2
    head = next(e["id"] for e in state()["draw"]["sketch"] if e["kind"] == "head")
    ok(draw("line", "trace", head))
    assert len(state()["draw"]["lines"]) == 5


def test_flat_at_point_and_reshade_replaces(draw):
    ok(draw("new", "pic", "--size", "300x300", "--request", "a red ball, cel shading"))
    ok(draw("do", "plan subject ball --box 200,200,800,800; plan part ball-body --of ball --box 200,200,800,800 "
                  "--color red; plan light top-left; palette make; palette pick 1; stage done; "
                  "sketch ellipse 250,250,750,750; stage done"))
    ok(draw("line", "250,500 500,250 750,500 500,750", "--closed"))
    ok(draw("do", "stage done; flat auto"))
    ok(draw("flat", "--at", "500,500", "--color", "#2040c0", "--part", "ball-body"))
    import pixelcanvas.project as project
    p = project.Project.find("pic.pixelcanvas")
    fl = p.layer("flat", create=False)
    c = fl[150, 150, :3]
    assert c[2] > c[0]                              # the area under the point turned blue
    ok(draw("do", "stage done; shade ball-body"))
    a1 = float(project.Project.find("pic.pixelcanvas").layer("shade", create=False)[..., 3].sum())
    ok(draw("shade", "ball-body"))
    a2 = float(project.Project.find("pic.pixelcanvas").layer("shade", create=False)[..., 3].sum())
    assert abs(a2 - a1) < 0.05 * a1                 # shading again replaces, it does not pile up


def test_light_from_a_point_and_haze_by_height(draw):
    ok(draw("new", "pic", "--size", "300x200", "--request", "a lake at sunset, painterly"))
    text = ok(draw("plan", "light", "back", "--from", "600,420"))
    assert "source at 600,420" in text
    lt = state()["draw"]["light"]
    assert lt["z"] < 0 and lt["at"] == [600.0, 420.0]
    ok(draw("do", "plan subject sky --box 0,0,1000,500; plan subject hills --box 0,300,1000,700; "
                  "plan subject shore --box 0,700,1000,1000; palette make; palette pick 1; stage done; "
                  "sketch horizon 500; stage done"))
    ok(draw("do", 'band sky "0,500 1000,500" --above --color sky; '
                  'band hills "0,450 500,320 1000,450" --below --color hills --rough 0.5; '
                  'band shore "0,800 1000,780" --below --color shore'))
    text = ok(draw("haze", "--amount", "0.5"))
    assert "hills" in text and "shore" not in text.split("haze")[-1].split("NEXT")[0].split("\n")[1]


def test_scatter_on_points_is_exact(draw):
    setup_bust(draw)
    text = ok(draw("scatter", "300,300 400,400 500,500", "--kind", "dot", "--size", "20", "--colors", "#ffffff"))
    assert "scatter 3 dot on the points" in text


def test_text_parts_get_a_text_item_not_lines():
    st = {"style": "poster", "plan": None, "stage": "plan",
          "draw": {"subjects": {"t": {"box": [0, 0, 1000, 200]}, "sax": {"box": [200, 300, 800, 900]}},
                   "parts": {"title-text": {"of": "t", "box": [50, 50, 950, 200], "focus": 3, "swatch": "cream"},
                             "body": {"of": "sax", "box": [200, 300, 800, 900], "focus": 2, "swatch": "black"}}}}
    from pixelcanvas import workflow_draw
    workflow_draw.build_todo(st)
    todo = st["todo"]
    assert not any(t["stage"] == "line" and t["key"] == "line" for t in todo)
    assert any(t["key"] == "text" and "pixelcanvas text" in t["cmd"] for t in todo)
    assert not any(t["stage"] == "detail" and "folds" in t["text"] for t in todo)


def test_hook_passes_new_arguments_through(draw):
    # the workflow owner may add parameters (gate_words(stage, style)); the hook forwards them untouched
    from pixelcanvas import cli_draw
    cli_draw._LAST["draw"] = False
    assert isinstance(workflow.gate_words("line", "cel"), str)
    cli_draw._LAST["draw"] = True
    assert isinstance(workflow.gate_words("line", "cel"), str)
