"""Palettes that follow the request's scene and mood (sunset, night ...), and colour roles only for figures."""
from __future__ import annotations

import json

from pixelcanvas.easel import draw_plan
from pixelcanvas.easel.draw_core import from_hex, rgb_to_lch

from .test_draw_flow import draw, ok  # noqa: F401  (fixture and helpers)

LAKE = "a mountain lake at sunset with pine trees in the foreground, painterly"


def lch(hx):
    return rgb_to_lch(from_hex(hx))


def warm(hx):
    h = lch(hx)[2]
    return h <= 90 or h >= 300


def lake_plan():
    d = {"subjects": {"sky": {"box": [0, 0, 1000, 450]}, "mountains": {"box": [0, 250, 1000, 600]},
                      "lake": {"box": [0, 550, 1000, 1000]}, "pines": {"box": [0, 300, 350, 1000]}},
         "parts": {"sun": {"of": "sky", "box": [600, 250, 700, 350], "focus": 2, "swatch": "sun"}}}
    return d


def candidates(d, request, base=(239, 138, 60)):
    names = draw_plan.needed_swatches(d)
    for x in ("background", "line"):
        if x not in names:
            names.append(x)
    return names, draw_plan.make_candidates(d, base, request, names)


def test_sunset_lake_has_a_warm_two_colour_sky_and_the_background_follows_it():
    d = lake_plan()
    names, cands = candidates(d, LAKE)
    assert "sky" not in names                       # the sky has a part, so the plan alone does not ask for it
    for c in cands:
        sw = c["swatches"]
        assert "sky" in sw and "sky-low" in sw
        assert warm(sw["sky"]) and warm(sw["sky-low"])
        L, C, h = lch(sw["background"])
        assert not (170 <= h <= 260 and L > 80)       # not the pale cyan complement of orange
        assert sw["background"] == sw["sky"]
        r, g, b = from_hex(sw["pines"])
        assert g >= b - 10                           # pines stay pines


def test_night_sky_is_dark_and_materials_are_cool():
    d = lake_plan()
    _, cands = candidates(d, "a quiet lake under a starry night sky, pines on the shore")
    for c in cands:
        sw = c["swatches"]
        assert lch(sw["sky"])[0] < 35 and "sky-low" in sw
        assert lch(sw["lake"])[0] < 50


def test_request_colour_still_wins_over_the_mood():
    d = lake_plan()
    _, cands = candidates(d, "a lake at sunset under a deep purple sky")
    purple = draw_plan.colour_words("deep purple")
    assert all(from_hex(c["swatches"]["sky"]) == purple for c in cands)
    assert all("sky-low" in c["swatches"] for c in cands)


def test_autumn_trees_turn_warm_and_winter_is_pale():
    d = {"subjects": {"trees": {"box": [0, 300, 1000, 1000]}}, "parts": {}}
    _, cands = candidates(d, "an autumn forest path", base=(90, 140, 200))
    assert all(warm(c["swatches"]["trees"]) for c in cands)
    _, cands = candidates(d, "snowy trees in winter", base=(90, 140, 200))
    for c in cands:
        assert lch(c["swatches"]["sky"])[0] > 75 and lch(c["swatches"]["trees"])[1] < 30


def test_neon_has_a_dark_background_and_bright_accents():
    d = {"subjects": {"sign": {"box": [200, 200, 800, 600]}}, "parts": {}}
    _, cands = candidates(d, "a neon sign, cyberpunk", base=(200, 80, 90))
    for c in cands:
        sw = c["swatches"]
        L, C, h = lch(sw["sign"])
        assert lch(sw["background"])[0] < 25 and C > 40 and (h > 290 or 170 < h < 215)


def test_figure_request_is_unchanged(draw):
    ok(draw("new", "pic", "--size", "300x300", "--request", "anime girl, red scarf, pale blue background"))
    ok(draw("do", "plan subject girl --box 150,40,850,1000; plan part hair --box 200,40,800,800; "
                  "plan part face --box 330,150,670,520; plan part scarf --box 280,560,720,760; palette make; "
                  "palette pick 1"))
    st = json.load(open("pic.easel/state.json", encoding="utf-8"))
    sw = st["draw"]["palette"]["swatches"]
    assert "skin" in sw and "sky" not in sw and "sky-low" not in sw
    L, C, h = lch(sw["background"])
    assert L > 80 and 200 < h < 280


def test_saxophone_neck_keeps_its_name_and_girl_neck_is_skin(draw):
    ok(draw("new", "pic", "--size", "300x300", "--request", "minimalist jazz poster: saxophone silhouette"))
    ok(draw("do", "plan subject saxophone --box 300,100,700,900; plan part bell --box 350,600,650,900; "
                  "plan part neck --box 450,100,600,300"))
    ok(draw("do", "plan subject girl --box 0,0,280,1000; plan part neck-l --box 100,300,180,360 --of girl"))
    parts = json.load(open("pic.easel/state.json", encoding="utf-8"))["draw"]["parts"]
    assert parts["neck"]["swatch"] == "neck" and parts["bell"]["swatch"] == "bell"
    assert parts["neck-l"]["swatch"] == "skin"
    assert "neck f2 [neck]" in "\n".join(draw_plan.plan_lines(type("P", (), {"state": {
        "draw": json.load(open("pic.easel/state.json", encoding="utf-8"))["draw"]}})))


def test_neck_becomes_skin_once_the_head_is_planned():
    class P:
        state = {"draw": {"subjects": {"bob": {"box": [0, 0, 1000, 1000]}}, "parts": {}}}
    d = P.state["draw"]
    draw_plan.set_part(P, "neck", [400, 300, 600, 400], "bob", None, None)
    assert d["parts"]["neck"]["swatch"] == "neck"
    draw_plan.set_part(P, "collar", [350, 380, 650, 450], "bob", None, "cloth")
    draw_plan.set_part(P, "head", [350, 50, 650, 300], "bob", None, None)
    assert d["parts"]["neck"]["swatch"] == "skin"
    assert d["parts"]["collar"]["swatch"] == "cloth"          # given with --color: kept
    draw_plan.set_part(P, "arm", [100, 300, 300, 800], "bob", None, "sleeve")
    draw_plan.set_part(P, "arm", [100, 300, 320, 820], "bob", None, None)
    assert d["parts"]["arm"]["swatch"] == "sleeve"            # re-planned without --color: still the user's


def test_given_colour_is_never_overridden(draw):
    ok(draw("new", "pic", "--size", "300x300", "--request", "a girl"))
    ok(draw("do", "plan subject girl --box 150,40,850,1000; plan part neck --box 450,500,550,560 --color scarf; "
                  "plan part face --box 330,150,670,520"))
    parts = json.load(open("pic.easel/state.json", encoding="utf-8"))["draw"]["parts"]
    assert parts["neck"]["swatch"] == "scarf" and parts["face"]["swatch"] == "skin"
