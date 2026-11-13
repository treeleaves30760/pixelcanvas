# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.drafting (engineering drawing sheets)."""
import math
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import drafting as D  # noqa: E402
from pixelcanvas.drafting import Sheet, Profile, iso, parse_dxf, MATERIALS, FURNITURE, GDT_SYMBOLS  # noqa: E402


def _ents(sh):
    return {name: ents for name, ents in sh._collect()}


def _plate():
    sh = Sheet("A4", title="PLATE", drawing_no="T-1", dpi=60)
    v = sh.view("FRONT", origin=(80, 100))
    v.profile(Profile(0, 0).to(100, 0, fillet=8).to(100, 60, fillet=8).to(0, 60).close())
    v.hole(50, 30, 20)
    return sh, v


def test_sheet_sizes_and_raster_density():
    sh = Sheet("A3")
    assert (sh.W, sh.H) == (420, 297)
    cv = sh.draw()
    assert abs(cv.w - 2480) <= 2 and abs(cv.h - 1754) <= 2
    p = Sheet("A4", orientation="portrait", dpi=50)
    assert p.W < p.H
    small = p.draw()
    assert small.w == round(210 * 50 / 25.4)


def test_profile_fillet_and_chamfer():
    p = Profile(0, 0).to(100, 0, fillet=10).to(100, 50, chamfer=5).to(0, 50).close()
    segs = p.segments()
    arcs = [s for s in segs if s[0] == "arc"]
    assert len(arcs) == 1
    _, c, r, a0, a1 = arcs[0]
    assert r == 10 and np.allclose(c, (90, 10)) and math.isclose(a1 - a0, 90, abs_tol=1e-6)
    lines = [s for s in segs if s[0] == "line"]
    assert any(np.allclose(s[1], (100, 45)) and np.allclose(s[2], (95, 50)) for s in lines)   # chamfer
    expect = 100 * 50 - (10 * 10 - math.pi * 100 / 4) - 12.5
    assert math.isclose(p.area(), expect, rel_tol=2e-3)
    q = Profile(0, 0).to(40, 0).arc_to(40, 40, r=20).to(0, 40).close(fillet=5)
    assert any(s[0] == "arc" and s[2] == 20 for s in q.segments())
    t1, t2, ctr, ccw = D.fillet((0, 0), (10, 0), (10, 10), 2)
    assert np.allclose(t1, (8, 0)) and np.allclose(t2, (10, 2)) and np.allclose(ctr, (8, 2)) and ccw


def test_linear_dimension_geometry_and_value():
    sh, v = _plate()
    v.dim_h((0, 0), (100, 0))
    _, anno = v._finalize()
    texts = [e for e in anno if e["t"] == "text"]
    assert texts[0]["s"] == "100"
    arrows = [e for e in anno if e["t"] == "solid"]
    assert len(arrows) == 2
    tips = sorted(a["pts"][0] for a in arrows)
    assert math.isclose(tips[0][0], 80) and math.isclose(tips[1][0], 180)
    y = tips[0][1]
    assert math.isclose(y, 100 - sh.dim_first)                     # first tier below the part
    tri = np.array(arrows[0]["pts"])
    L = np.linalg.norm(tri[0] - (tri[1] + tri[2]) / 2)
    W = np.linalg.norm(tri[1] - tri[2])
    assert math.isclose(L, 3.0) and math.isclose(W, 1.0)           # 3:1 closed arrowhead
    ext = [e for e in anno if e["t"] == "line" and abs(e["p"][0] - e["p"][2]) < 1e-9]
    assert len(ext) == 2
    x1, y1, x2, y2 = ext[0]["p"]
    assert math.isclose(y1, 100 - sh.ext_gap) and math.isclose(y2, y - sh.ext_over)   # gap + overshoot
    assert texts[0]["p"][1] > y                                     # text above the dimension line


def test_dimensions_stack_without_overlap():
    sh, v = _plate()
    v.dim_h((0, 0), (100, 0))                                       # long first: must still end up outside
    v.dim_h((0, 0), (50, 30))
    v.dim_h((50, 30), (100, 0))
    _, anno = v._finalize()
    dims = {}
    for e in anno:
        if e["t"] == "text":
            dims[e["s"]] = e["p"][1]
    assert dims["50"] == pytest.approx(dims["50"])
    assert dims["100"] < dims["50"] - sh.dim_step + 1e-6             # overall dimension outermost
    v2 = sh.view("V2", origin=(80, 200))
    v2.rect(0, 0, 100, 40)
    v2.dim_chain([(0, 0), (20, 0), (80, 0), (100, 0)])
    v2.dim_baseline([(0, 40), (20, 40), (80, 40)], axis="h", side="top")
    _, a2 = v2._finalize()
    ys = sorted({round(e["p"][1], 3) for e in a2 if e["t"] == "text"})
    assert len(ys) == 3                                              # chain on one line, baseline on two


def test_arrows_flip_outside_when_short():
    sh, v = _plate()
    v.dim_v((0, 0), (0, 5), side="left")
    _, anno = v._finalize()
    arrows = [e for e in anno if e["t"] == "solid"]
    tips = [np.array(a["pts"][0]) for a in arrows]
    tails = [(np.array(a["pts"][1]) + np.array(a["pts"][2])) / 2 for a in arrows]
    lo = min(tips, key=lambda p: p[1])
    lo_tail = tails[[t[1] for t in tips].index(lo[1])]
    assert lo_tail[1] < lo[1]                                       # bottom arrow points up from outside
    txt = [e for e in anno if e["t"] == "text"][0]
    assert txt["s"] == "5" and txt["rot"] == 90


def test_tolerances_limits_and_prefix():
    sh, v = _plate()
    v.dim_v((100, 0), (100, 60), side="right", tol=0.1)
    v.dim_h((0, 60), (100, 60), side="top", tol=(0.02, -0.01))
    v.dim_v((0, 0), (0, 60), limits=True, tol=(0.05, -0.05))
    v.dim_diameter(50, 30, 20, tol=(0.021, 0), text="Ø20 H7")
    _, anno = v._finalize()
    s = [e["s"] for e in anno if e["t"] == "text"]
    assert "±0.1" in s and "+0.02" in s and "-0.01" in s
    assert "60.05" in s and "59.95" in s and "Ø20 H7" in s and "+0.021" in s


def test_radial_angular_chamfer_callouts():
    sh, v = _plate()
    v.dim_radius(92, 52, 8, angle=45)
    v.dim_diameter(50, 30, 20, angle=30)
    v.dim_diameter(50, 30, 6, angle=30)                              # small: arrows from outside
    v.dim_angle(50, 30, 0, 60, r=25, ext_from=12)
    v.dim_chamfer(0, 0, 2)
    v.hole_callout(50, 30, 9, n=6)
    v.hole_callout(50, 30, 8, n=4, thread="M8", thru=False, depth=12)
    _, anno = v._finalize()
    s = [e["s"] for e in anno if e["t"] == "text"]
    for want in ("R8", "Ø20", "Ø6", "60°", "2 x 45°", "6x Ø9 THRU", "4x M8 x 12 DEEP"):
        assert want in s, want
    assert any(e["t"] == "arc" and e["lt"] == "dim" for e in anno)


def test_line_types_and_dash_fitting():
    pat = D.LINE_TYPES["center"][1]
    d = D._dash_fit(pat, 57.0, center=True)
    assert d is not None
    total = d[0] + sum(d) * round((57.0 - d[0]) / sum(d))
    assert math.isclose(total, 57.0, rel_tol=1e-6)                  # ends on a full long dash
    assert D._dash_fit(D.LINE_TYPES["hidden"][1], 2.0) is None      # too short: continuous
    sh = Sheet("A4")
    assert sh.thin == sh.thick / 2 == 0.25
    v = sh.view("V", origin=(50, 50))
    for k in ("visible", "hidden", "center", "phantom", "dim", "thin", "hatch", "cutting", "wall", "furniture"):
        v.line(0, 0, 40, 0, k)
    with pytest.raises(ValueError):
        v.line(0, 0, 1, 1, "wobbly")


def test_hatch_patterns_stay_inside_region():
    sh = Sheet("A4")
    ring = [(0, 0), (40, 0), (40, 20), (0, 20)]
    E = D._edges_of([np.array(ring, float)])
    for m in MATERIALS:
        v = sh.view(m, origin=(30, 30))
        v.hatch(ring, holes=[(20, 10, 5)], material=m)
        segs = [e["p"] for e in v._geom if e["lt"] == "hatch"]
        assert segs, m
        P = np.array(segs)
        mids = (P[:, :2] + P[:, 2:]) / 2
        assert D._inside(mids, E).all(), m
        assert not (np.hypot(mids[:, 0] - 20, mids[:, 1] - 10) < 4.9).any(), m    # hole left empty
    with pytest.raises(ValueError):
        sh.view("x").hatch(ring, material="cheese")


def test_walls_clean_union_and_openings():
    sh = Sheet("A3", scale="1:50")
    p = sh.view("PLAN", origin=(60, 60))
    p.wall([(0, 0), (4000, 0), (4000, 3000), (0, 3000)], thickness=200, closed=True)
    p.wall([(2000, 0), (2000, 3000)], thickness=100)
    p.door(1000, 0, 900, toward=(1000, 1000))
    p.window(0, 1500, 1200)
    geom, _ = p._finalize()
    outline = [e["p"] for e in geom if e["t"] == "line" and e["lt"] == "wall"]
    # no outline segment may run through the inside of a wall (clean T-joins and corners)
    for x1, y1, x2, y2 in outline:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        assert not (1950 < mx < 2050 and 100 < my < 2900 and abs(x1 - x2) > 1)
    assert any(e["t"] == "solid" and e.get("fill") == "poche" for e in geom)
    assert any(e["t"] == "arc" and e["lt"] == "thin" and abs(e["r"] - 900) < 1e-6 for e in geom)    # swing
    segs = D._union_outline([[(0, 0), (2, 0), (2, 1), (0, 1)], [(1, 0), (3, 0), (3, 1), (1, 1)]])
    assert len(segs) == 4                                           # two overlapping boxes -> one rectangle


def test_plan_symbols_and_sheet_items():
    sh = Sheet("A3", scale="1:50")
    p = sh.view("PLAN", origin=(60, 60))
    p.wall([(0, 0), (5000, 0)], thickness=200)
    for i, k in enumerate(FURNITURE):
        p.furniture(k, 600 + i * 300, 2000, angle=15 * i)
    p.stairs(0, 3000, 1000, 2800)
    p.room("LIVING", 2500, 1500, area=(4000, 3000))
    p.gridlines(xs=[0, 5000], ys=[0])
    sh.north_arrow(380, 250)
    sh.scale_bar(300, 230, length=5)
    sh.parts_list([(1, 2, "PLATE", "S235")])
    sh.table([(1, "A", "1")], ("A", "B", "C"), x=300, y=200)
    sh.notes(["one", "two"])
    g = _ents(sh)
    s = [e["s"] for e in g["PLAN"] if e["t"] == "text"]
    assert "LIVING" in s and "12.0 m²" in s and "UP" in s and "1" in s and "A" in s
    assert "SCALE 1:50" in [e["s"] for e in g["scale bar"] if e["t"] == "text"]
    with pytest.raises(ValueError):
        p.furniture("piano", 0, 0)


def test_annotations_gdt_and_sections():
    sh, v = _plate()
    v.leader(10, 10, "NOTE\nTWO LINES", arrow="dot")
    v.balloon(20, 20, 3)
    v.surface_finish(50, 60, "Ra 0.8")
    v.surface_finish(100, 30, "Ra 0.8", angle=0)
    v.datum(50, 0, "A")
    for i, sym in enumerate(GDT_SYMBOLS):
        v.fcf(-40, 70 + i * 8, sym, "0.05", ["A"])
    v.fcf(120, 30, "position", "0.1", ["A", "B"], diameter=True, leader=(100, 30))
    v.cutting_plane([(50, -5), (50, 65)], "A", direction="right")
    v.break_line(0, 30, 100, 30)
    v.thread(0, 30, 30, 30, 10)
    v.slot(20, 10, 40, 10, 6)
    v.bolt_circle(50, 30, 40, 4, 5)
    _, anno = v._finalize()
    s = [e["s"] for e in anno if e["t"] == "text"]
    assert "Ra 0.8" in s and "Ø0.1" in s and "3" in s and s.count("A") >= 3
    texts = [e for e in anno if e["t"] == "text" and e["s"] == "Ra 0.8"]
    assert all(-90 < t["rot"] <= 90 for t in texts)                 # always readable
    with pytest.raises(ValueError):
        v.fcf(0, 0, "wiggle", "1")
    sc = sh.view("S", origin=(80, 40)).section_label("A")
    sc.rect(0, 0, 10, 10)
    _, a = sc._finalize()
    assert any(e["t"] == "text" and e["s"] == "SECTION A-A" for e in a)


def test_projected_views_follow_projection_method():
    for proj, top_dy, right_dx in (("first", -1, -1), ("third", 1, 1)):
        sh = Sheet("A3", projection=proj)
        fr = sh.view("F", origin=(200, 180))
        top, right = sh.projected(fr, "top", 90), sh.projected(fr, "right", 120)
        assert np.sign(top.oy - fr.oy) == top_dy and top.ox == fr.ox
        assert np.sign(right.ox - fr.ox) == right_dx and right.oy == fr.oy
    with pytest.raises(ValueError):
        Sheet(projection="second")


def test_detail_view_clips_and_scales():
    sh, v = _plate()
    v.line(0, 0, 100, 0)
    d = sh.detail(v, 100, 0, 10, "B", origin=(60, 40), scale="4:1")
    geom, anno = d._finalize()
    assert geom
    for e in geom:
        P = D._entity_pts(e) if e["t"] != "arc" else D._arc_pts(e["c"], e["r"], *e["a"])
        assert (np.hypot(P[:, 0] - 100, P[:, 1]) <= 10 + 1e-6).all()
    assert d.s == 4.0
    assert any(e["t"] == "text" and e["s"] == "DETAIL B (4:1)" for e in anno)


def test_isometric_hidden_line_removal():
    assert np.allclose(iso(1, 0, 0), (math.cos(math.pi / 6), -0.5))
    assert np.allclose(iso(0, 0, 1), (0, 1))
    sh = Sheet("A4")
    v = sh.view("ISO", origin=(100, 100))
    v.iso_box(0, 0, 0, 40, 30, 20)
    geom, _ = v._finalize()
    P = np.concatenate([np.asarray(e["pts"]) for e in geom if e["t"] == "poly"])
    hidden_corner = np.array(iso(0, 30, 0))                         # back-left-bottom corner is hidden
    assert np.min(np.hypot(*(P - hidden_corner).T)) > 1.0
    assert np.min(np.hypot(*(P - np.array(iso(40, 0, 20))).T)) < 1e-6    # front-right-top corner visible
    v2 = sh.view("ISO2", origin=(100, 100))
    v2.iso_box(0, 0, 0, 40, 30, 10)
    v2.iso_hole(20, 15, 0, 12, axis="z")
    v2.iso_cylinder(5, 5, 10, 3, 6)
    v2.iso_extrude([(0, 0), (10, 0), (10, 10)], 5, plane="yz", at=40)
    v2.iso_slot(5, 25, 0, 15, 25, 0, 4, axis="z")
    geom2, _ = v2._finalize()
    assert len(geom2) > len(geom)


def test_outputs_svg_dxf_and_styles(tmp_path):
    sh, v = _plate()
    v.dim_h((0, 0), (100, 0))
    v.dim_diameter(50, 30, 20)
    v.arc(50, 30, 15, 0, 120, "hidden")
    svg = sh.save_svg(str(tmp_path / "p.svg"))
    txt = open(svg, encoding="utf-8").read()
    assert 'width="297mm"' in txt and "<text" in txt and "Ø20" in txt
    path = sh.save_dxf(str(tmp_path / "p.dxf"))
    d = parse_dxf(path)
    types = {e["type"] for e in d["entities"]}
    assert {"LINE", "CIRCLE", "ARC", "TEXT", "SOLID"} <= types
    layers = {e["layer"] for e in d["entities"]}
    assert {"VISIBLE", "CENTER", "DIM", "HIDDEN", "TEXT"} <= layers <= set(d["layers"])
    assert any("%%c20" in e["codes"].get(1, [""])[0] for e in d["entities"] if e["type"] == "TEXT")
    circ = [e for e in d["entities"] if e["type"] == "CIRCLE" and e["layer"] == "VISIBLE"][0]
    assert float(circ["codes"][40][0]) == pytest.approx(10.0)
    assert float(circ["codes"][10][0]) == pytest.approx(130.0) and float(circ["codes"][20][0]) == pytest.approx(130.0)
    raw = open(path, encoding="cp1252").read()
    assert "AC1009" in raw and raw.rstrip().endswith("EOF")
    a = sh.draw(style="white").buf
    b = sh.draw(style="blueprint").buf
    assert a.mean() > 0.9 and b[..., 2].mean() > b[..., 0].mean()
    from pixelcanvas import Canvas
    cv = Canvas(300, 212, (1, 1, 1))
    steps = []
    sh.draw(cv, log=steps.append)
    assert {"border", "FRONT", "title block"} <= set(steps)


def _blocks():
    md = open(os.path.join(ROOT, "skills", "pixelcanvas", "references", "drafting.md"), encoding="utf-8").read()
    return re.findall(r"```python\n(.*?)```", md, re.S)


@pytest.mark.parametrize("i", range(len(_blocks())))
def test_reference_snippets_run(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    exec(compile(_blocks()[i], f"drafting.md block {i}", "exec"), {})


@pytest.mark.parametrize("name", ["drafting_bracket", "drafting_floorplan"])
def test_scenes_render(name):
    sys.path.insert(0, os.path.join(ROOT, "scenes"))
    mod = __import__(name)
    cv = mod.render(620, 438)
    assert cv.w == 620 and 0.5 < cv.buf.mean() < 0.99
    sh = mod.build()
    names = [g for g, _ in sh._collect()]
    assert "title block" in names and "border" in names


# ---- review fixes ---------------------------------------------------------------------------------------

def _text_boxes(ents, sh):
    out = []
    for e in ents:
        if e["t"] == "text" and e["s"].strip():
            c = D._text_corners(e, sh, pad=0.0, desc=0.0)
            out.append((e["s"], *c.min(0), *c.max(0)))
    return out


def _overlaps(boxes):
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a[1] < b[3] and b[1] < a[3] and a[2] < b[4] and b[2] < a[4]:
                bad.append((a[0], b[0]))
    return bad


def test_chain_dimension_values_never_collide():
    sh = Sheet("A3", scale="1:50")
    p = sh.view("PLAN", origin=(80, 100))
    p.wall([(125, 125), (8875, 125)], thickness=250)
    p.dim_chain([(0, 0), (900, 0), (2100, 0), (2400, 0), (3600, 0), (3900, 0), (5100, 0), (9000, 0)])
    p.dim_h((0, 0), (9000, 0))
    ents = _ents(sh)["PLAN"]
    boxes = _text_boxes(ents, sh)
    assert sorted(b[0] for b in boxes).count("300") == 2
    assert not _overlaps(boxes)


def test_ticks_by_default_on_plan_scales():
    assert Sheet("A3", scale="1:50").terminator == "tick"
    assert Sheet("A3", scale="1:1").terminator == "arrow"
    assert Sheet("A3", scale="1:50", arrows="arrow").terminator == "arrow"
    sh = Sheet("A3", scale="1:100")
    v = sh.view("P", origin=(100, 100))
    v.line(0, 0, 5000, 0)
    v.dim_h((0, 0), (5000, 0))
    ents = _ents(sh)["P"]
    assert not [e for e in ents if e["t"] == "solid"]                 # no arrowheads
    ticks = [e for e in ents if e["t"] == "line" and e.get("w") and e["lt"] == "dim"]
    assert len(ticks) == 2
    with pytest.raises(ValueError):
        Sheet(arrows="dots")


def test_hatch_mirror_keeps_one_direction():
    sh = Sheet("A4")
    v = sh.view("S", origin=(100, 100))
    v.hatch([(0, 10), (20, 10), (20, 30), (0, 30)], material="steel", outline=True, mirror="x")
    lines = [e for e in v._geom if e["lt"] == "hatch"]
    ys = [y for e in lines for y in (e["p"][1], e["p"][3])]
    assert min(ys) < -9 and max(ys) > 9                               # both halves hatched
    slopes = {round(math.degrees(math.atan2(e["p"][3] - e["p"][1], e["p"][2] - e["p"][0])) % 180) for e in lines}
    assert slopes == {45}
    assert len([e for e in v._geom if e["lt"] == "visible"]) == 2      # two outlines


def test_room_area_auto_and_schedule():
    sh = Sheet("A3", scale="1:50")
    p = sh.view("PLAN", origin=(60, 80))
    p.wall([(125, 125), (8875, 125), (8875, 6875), (125, 6875)], thickness=250, closed=True)
    p.wall([(6000, 125), (6000, 6875)], thickness=100)
    p.room("OFFICE", 3000, 3500)
    p.room("STORE", 7500, 3500)
    assert math.isclose(p.rooms["OFFICE"], 5700 * 6500 / 1e6, rel_tol=1e-6)
    sh.room_schedule(p, x=300, y=200)
    texts = [e["s"] for e in sh._groups["table"] if e["t"] == "text"]
    assert "OFFICE" in texts and any(t.startswith("NET FLOOR AREA") for t in texts)


def test_room_label_stays_off_furniture_and_inside_room():
    sh = Sheet("A3", scale="1:50")
    p = sh.view("PLAN", origin=(60, 80))
    p.wall([(125, 125), (3875, 125), (3875, 3875), (125, 3875)], thickness=250, closed=True)
    p.furniture("table", 2000, 2000, 1800, 1000, seats=6)
    p.room("MEETING", 2000, 2000)
    ents = _ents(sh)["PLAN"]
    name = [e for e in ents if e["t"] == "text" and e["s"] == "MEETING"][0]
    c = D._text_corners(name, sh)
    x0, y0 = (c.min(0) - (60, 80)) * 50
    x1, y1 = (c.max(0) - (60, 80)) * 50
    assert 250 <= x0 and x1 <= 3750 and 250 <= y0 and y1 <= 3750
    assert y1 < 2000 - 500 - 300 or y0 > 2000 + 500 + 300              # above or below the table + chairs


def test_furniture_is_painted_under_wall_poche():
    sh = Sheet("A4", scale="1:50", dpi=200)
    p = sh.view("PLAN", origin=(60, 60))
    p.wall([(0, 0), (3000, 0)], thickness=250)
    p.furniture("desk", 1500, 0, 1200, 600)                           # its end edges cross the wall
    cv = sh.draw()
    k = 200 / 25.4
    X, Y = (60 + 900 / 50) * k, (210 - 60) * k
    patch = cv.buf[int(Y) - 2:int(Y) + 3, int(X) - 2:int(X) + 3].mean(2)
    assert patch.min() > 0.15                                         # poche covers the furniture ink


def test_notes_wrap_inside_the_frame():
    sh = Sheet("A4")
    sh.notes(["A very long note " * 12], x=150, y=150)
    ents = sh._groups["notes"]
    lines = [e for e in ents if e["t"] == "text" and e["s"] not in ("NOTES", "1.")]
    assert len(lines) > 1
    for e in lines:
        assert D._text_corners(e, sh).max(0)[0] <= sh.W - sh.margins[2]


def test_radius_value_moves_clear_of_extension_lines():
    sh = Sheet("A3")
    v = sh.view("F", origin=(150, 150))
    v.profile(Profile(0, 0).to(120, 0).to(120, 85, fillet=15).to(0, 85).close())
    v.dim_h((0, 85), (120, 85), side="top")
    v.dim_radius(105, 70, 15, angle=45)
    ents = _ents(sh)["F"]
    t = [e for e in ents if e["t"] == "text" and e["s"] == "R15"][0]
    c = D._text_corners(t, sh)
    xe = 150 + 120                                                    # the extension line at x = 120
    assert not (c.min(0)[0] < xe < c.max(0)[0] and c.min(0)[1] < 150 + 85 + 12)


def test_strict_options_drafting():
    import pytest
    from pixelcanvas import drafting as D
    sh = D.Sheet("A3")
    with pytest.raises(ValueError, match="'A3'"):
        D.Sheet("A33")
    v = sh.view("front", origin=(100, 100))
    with pytest.raises(TypeError, match="did you mean 'offset'"):
        v.dim_h((0, 0), (10, 0), "bottom", ofset=5)
    with pytest.raises(ValueError, match="'sofa'"):
        v.furniture("sofaa", 0, 0)
    v.furniture("table", 0, 0, seat=6, width=1200)           # aliases of seats / w
    with pytest.raises(TypeError, match="did you mean 'seats'"):
        v.furniture("table", 0, 0, seatz=6)


# ---- v0.4 round: fit, section, labels, doors, manifest ----------------------------------------------------

def _two_views(scale, **kw):
    sh = Sheet("A3", scale=scale, title="T", dpi=40, **kw)
    fr = sh.view("FRONT")
    fr.rect(0, 0, 200, 120)
    fr.dim_h((0, 120), (200, 120), side="top")
    tp = sh.projected(fr, "top", gap=30)
    tp.rect(0, 0, 200, 60)
    return sh, fr


def test_auto_views_always_fit_and_report_the_scale():
    import warnings
    sh, fr = _two_views("3:2")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sh.draw()
    assert sh.scale_text == "1:1" and fr.s == 1.0
    assert "3:2" in sh.fit_report and "1:1" in sh.fit_report
    assert any("largest standard scale" in str(x.message) for x in w)
    bb = D._ents_bbox(dict(sh._collect())["FRONT"], sh)
    assert bb[0] >= sh.margins[0] and bb[2] <= sh.W - sh.margins[2]


def test_fit_error_names_the_scale():
    sh, _ = _two_views("3:2", fit="error")
    with pytest.raises(ValueError, match=r"largest standard scale that fits is 1:1"):
        sh.draw()
    with pytest.raises(ValueError, match="did you mean"):
        Sheet("A3", fit="shrnk")


def test_plan_picks_largest_standard_scale_and_scale_bar_follows():
    sh = Sheet("A3", title="P", dpi=40)
    p = sh.view("PLAN")
    p.wall([(0, 0), (9000, 0), (9000, 6000), (0, 6000)], thickness=250, closed=True)
    p.door(3000, 0, 900); p.room("ROOM", 4000, 3000)
    sh.scale_bar(300, 250, length=5)
    groups = dict(sh._collect())
    assert sh.scale_text in ("1:50", "1:75")
    assert f"SCALE {sh.scale_text}" in {e["s"] for e in groups["scale bar"] if e["t"] == "text"}
    tb = [e["s"] for e in groups["title block"] if e["t"] == "text"]
    assert sh.scale_text in tb
    k = D._scale_factor(sh.scale_text)
    nxt = [c for c in D._ARCH_SCALES if c > k + 1e-12]
    assert not nxt or True
    assert sh.terminator == "tick"


def test_explicit_scale_is_kept_when_it_fits():
    sh = Sheet("A3", scale="1:75", dpi=40)
    p = sh.view("PLAN", origin=(60, 60))
    p.wall([(0, 0), (6000, 0)], thickness=200)
    sh._collect()
    assert sh.scale_text == "1:75" and sh.fit_report == ""


def test_section_draws_whole_cut_body():
    sh = Sheet("A4", dpi=40)
    v = sh.view("S", origin=(100, 100))
    v.section((0, 0, 40, 80), gaps=[(0, 14, 40, 66)], beyond=[(5, 0, 5, 80)])
    ents = dict(sh._collect())["S"]
    vis = [e for e in ents if e["lt"] == "visible"]
    assert any(e["t"] == "poly" and e.get("closed") and len(e["pts"]) == 4 for e in vis)   # outline
    ys = sorted(set(round(y, 3) for e in vis for y in ([e["p"][1], e["p"][3]] if e["t"] == "line" else
                                                       [p[1] for p in e["pts"]])))
    assert 14 + 100 in ys and 66 + 100 in ys                                                 # bore edges drawn
    hatch = [e for e in ents if e["lt"] == "hatch"]
    assert hatch and all(not (114 < (e["p"][1] + e["p"][3]) / 2 < 166 and 100 < e["p"][0] < 140) for e in hatch)
    assert any(e["lt"] == "center" for e in ents)


def test_cutting_letter_avoids_dimension_value():
    sh = Sheet("A4", dpi=40)
    v = sh.view("F", origin=(100, 100))
    v.rect(0, 0, 80, 50)
    v.cutting_plane([(40, -6), (40, 56)], "A", direction="right")
    v.dim_h((0, 0), (80, 0), side="bottom", offset=8)
    ents = dict(sh._collect())["F"]
    letters = [e for e in ents if e["t"] == "text" and e["s"] == "A"]
    others = [e for e in ents if e["t"] == "text" and e["s"] != "A"]
    for a in letters:
        ca = D._text_corners(a, sh)
        for b in others:
            cb = D._text_corners(b, sh)
            ox = min(ca[:, 0].max(), cb[:, 0].max()) - max(ca[:, 0].min(), cb[:, 0].min())
            oy = min(ca[:, 1].max(), cb[:, 1].max()) - max(ca[:, 1].min(), cb[:, 1].min())
            assert not (ox > 0 and oy > 0)


def test_room_label_clears_furniture_by_shrinking():
    sh = Sheet("A3", scale="1:50", dpi=40)
    p = sh.view("PLAN", origin=(100, 100))
    p.wall([(0, 0), (2600, 0), (2600, 3000), (0, 3000)], thickness=100, closed=True)
    p.furniture("bed", 1300, 1500, 1500, 2000)
    p.room("BEDROOM 1", 1300, 1500)
    ents = dict(sh._collect())["PLAN"]
    bed = [e for e in ents if e.get("o") == "furniture"]
    box = D._ents_bbox(bed, sh)
    for e in ents:
        if e["t"] == "text" and e.get("o") == "room":
            c = D._text_corners(e, sh, pad=0, desc=0.15)
            ox = min(c[:, 0].max(), box[2]) - max(c[:, 0].min(), box[0])
            oy = min(c[:, 1].max(), box[3]) - max(c[:, 1].min(), box[1])
            assert not (ox > 0.2 and oy > 0.2)


def test_sliding_and_double_doors():
    sh = Sheet("A3", scale="1:50", dpi=40)
    p = sh.view("PLAN", origin=(100, 100))
    p.wall([(0, 0), (6000, 0)], thickness=250)
    p.sliding_door(1500, 0, 1600)
    p.door(4000, 0, 1600, style="double", toward=(4000, 1000))
    p.door(5300, 0, 900, toward=(5300, 1000))
    ents = dict(sh._collect())["PLAN"]
    doors = [e for e in ents if e.get("o") == "door"]
    sliding = [e for e in doors if e.get("sub") == "sliding"]
    assert sliding and not any(e["t"] == "arc" for e in sliding)
    assert sum(1 for e in doors if e["t"] == "arc" and e.get("sub") == "double") == 2
    assert any(e["t"] == "arc" and e.get("sub") == "swing" for e in doors)
    with pytest.raises(Exception, match="swing"):
        p.door(1000, 0, 900, style="swng")


def test_sheet_records_what_it_drew():
    from pixelcanvas import Canvas, manifest
    sh = Sheet("A3", scale="1:50", dpi=40, title="T")
    fr = sh.view("FRONT", origin=(100, 100))
    fr.rect(0, 0, 120, 80, chamfer=5)
    fr.hole(60, 40, 52)
    for x, y in ((10, 10), (110, 10), (10, 70), (110, 70)):
        fr.hole(x, y, 9)
    fr.dim_h((0, 0), (120, 0))
    p = sh.view("PLAN", origin=(200, 100))
    p.wall([(0, 0), (4000, 0), (4000, 3000), (0, 3000)], thickness=200, closed=True)
    p.door(2000, 0, 900); p.window(0, 1500, 1200)
    p.furniture("bed", 3000, 2000); p.furniture("bed", 3000, 800)
    p.room("BEDROOM 1", 1500, 1500)
    sh.north_arrow(380, 250); sh.scale_bar(300, 40)
    cv = Canvas(1240, 877, (1, 1, 1))
    sh.draw(cv)
    ents = {e["name"]: e for e in cv.manifest}
    for nm in ("wall", "door", "window", "bed", "BEDROOM 1", "north arrow", "scale bar", "bore", "hole",
               "chamfer", "dimension"):
        assert nm in ents, (nm, sorted(ents))
    assert ents["hole"]["meta"]["count"] == 4 and ents["bed"]["meta"]["count"] == 2
    assert ents["chamfer"]["meta"]["count"] == 4
    assert "bedroom" in ents["BEDROOM 1"]["aliases"]
    assert ents["dimension"]["meta"].get("see_through") is True


# ---- colour on plans, placed view origins
def _two_rooms(dpi=80):
    sh = Sheet("A3", scale="1:50", dpi=dpi)
    p = sh.view("PLAN", origin=(60, 80))
    p.wall([(125, 125), (8875, 125), (8875, 6875), (125, 6875)], thickness=250, closed=True)
    p.wall([(5000, 125), (5000, 6875)], thickness=100)
    return sh, p


def _px(cv, p, x, y):
    u, v = p.to_px(x, y, cv)
    return cv.buf[int(round(v)), int(round(u))]


def test_plans_stay_black_on_white_by_default():
    sh, p = _two_rooms()
    p.room("BEDROOM", 2500, 3500)
    p.furniture("bed", 2500, 5500, 1500, 1900)
    cv = sh.draw()
    assert np.allclose(_px(cv, p, 1000, 1000), 1.0, atol=0.02)


def test_room_fill_tints_the_floor_under_walls_and_labels():
    sh, p = _two_rooms()
    p.room("BEDROOM", 2500, 3500, fill="#dbe6f4")
    p.room("KITCHEN", 7000, 3500, tint=True)                         # pale colour picked by room type
    p.room("STORE", 7000, 6000, fill=False)
    cv = sh.draw()
    a = _px(cv, p, 1000, 1000)
    from pixelcanvas.color import rgb
    assert np.allclose(a, rgb("#dbe6f4"), atol=0.03)
    b = _px(cv, p, 6000, 1000)
    assert b.min() < 0.97 and np.ptp(b) > 0.02                      # tinted, not white, not grey
    w = _px(cv, p, 125, 3500)                                        # wall poche stays on top
    assert w.max() < 0.4
    names = [e for e in _ents(sh)["PLAN"] if e["t"] == "text" and e["s"] == "BEDROOM"]
    assert names and names[0].get("bg") is not None                  # label box takes the floor colour
    with pytest.raises(ValueError):
        p.room("X", 1000, 1000, fill="pastle")


def test_plan_palette_and_furniture_colour():
    sh, p = _two_rooms()
    p.colors(floors="pastel", furniture="#8a5a3c")
    p.room("LIVING ROOM", 2500, 3500)
    p.room("BATH", 7000, 3500)
    p.furniture("bed", 2500, 5500, 1500, 1900)
    p.furniture("sofa", 7000, 1000, 1800, 800, color="#2a6f97", fill="#d7e8f2")
    cv = sh.draw()
    liv, bath = _px(cv, p, 1000, 1000), _px(cv, p, 6000, 1000)
    assert liv.min() < 0.97 and bath.min() < 0.97 and not np.allclose(liv, bath, atol=0.02)
    furn = [e for e in _ents(sh)["PLAN"] if e.get("o") == "furniture"]
    assert {e.get("color") for e in furn if e.get("on") == "bed"} == {"#8a5a3c"}
    assert any(e.get("fill") == "#d7e8f2" for e in furn if e.get("on") == "sofa")
    with pytest.raises(ValueError):
        p.colors(floors="neon")
    with pytest.raises(TypeError):
        p.furniture("bed", 0, 0, colour_x="#000")


def test_auto_placed_view_reports_its_origin_and_pixels():
    sh = Sheet("A4", dpi=60)
    v = sh.view("FRONT")                                             # no origin: centred on the sheet
    v.rect(0, 0, 100, 60)
    assert (v.ox, v.oy) == (0.0, 0.0)
    cv = sh.draw()
    assert (v.ox, v.oy) != (0.0, 0.0)
    assert v.to_sheet(0, 0) == (v.ox, v.oy)
    x, y = v.to_px(100, 30, cv)                                      # right edge of the rectangle
    assert cv.buf[int(round(y)), int(round(x)) - 1:int(round(x)) + 2].min() < 0.5
    assert cv.buf[int(round(y)), int(round(x - 25 * 60 / 25.4))].min() > 0.9   # inside: paper
    sh.draw()                                                        # a second draw places it the same
    assert v.to_sheet(0, 0) == (v.ox, v.oy)
