"""Gates that use the measures the reports print (the holes four lab sessions let through): face features, parts
without an area of their own, parts that should touch, background inside a subject, detail on the focus part,
subjects that do not fill their box, depth order, clipped detail marks, flat by point, the plan view of a batch."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from pixelcanvas import cli, workflow


@pytest.fixture
def draw(tmp_path, monkeypatch):
    from pixelcanvas import cli_draw
    saved_cmds = dict(cli.COMMANDS)
    saved_usage = dict(cli.USAGE)
    names = ("next_action", "check_gate", "build_todo", "plan_lines", "gate_words")
    saved_wf = {n: getattr(workflow, n) for n in names}
    cli_draw.install()
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        return cli.run([str(a) for a in argv])
    yield run
    cli.COMMANDS.clear()
    cli.COMMANDS.update(saved_cmds)
    cli.USAGE.clear()
    cli.USAGE.update(saved_usage)
    for n, f in saved_wf.items():
        setattr(workflow, n, f)
    if hasattr(workflow, "_draw_patched"):
        del workflow._draw_patched


def ok(res):
    code, text = res
    assert code == 0, text
    return text


def state(name="pic"):
    return json.load(open(f"{name}.pixelcanvas/state.json", encoding="utf-8"))


def project(name="pic"):
    from pixelcanvas.project import Project
    return Project.find(f"{name}.pixelcanvas")


def bust(run, extra=""):
    ok(run("new", "pic", "--size", "400x400", "--request", "anime girl with silver hair and a red scarf"))
    ok(run("do", "plan subject girl --box 150,40,850,1000; plan part head --box 330,150,670,520 --focus 3" + extra
           + "; plan light top-left; palette make; palette pick 1"))


# ============================================================================================ G1 face features

def test_line_gate_needs_mouth_nose_brows_and_face_lays_them(draw):
    bust(draw)
    ok(draw("stage", "done"))
    ok(draw("sketch", "head", "girl"))
    ok(draw("stage", "done", "--because", "test"))
    todo = [t for t in state()["todo"] if t["stage"] == "line"]
    face = [t for t in todo if t["key"] == "face"]
    assert face and face[0]["cmd"].startswith("pixelcanvas face s1")
    ok(draw("line", "trace", "s1"))
    code, text = draw("stage", "done")
    assert code == 1 and "mouth" in text and "nose" in text and "pixelcanvas face s1" in text
    text = ok(draw("face", "s1"))
    assert "mouth" in text and "brow-l" in text
    st = state()
    assert [t for t in st["todo"] if t["key"] == "face"][0]["status"] == "done"
    from pixelcanvas import draw_checks
    p = project()
    assert not draw_checks.face_holes(p)


def test_animal_face_asks_for_a_muzzle(draw):
    ok(draw("new", "pic", "--size", "400x400", "--request", "a chibi fox mage"))
    ok(draw("do", "plan subject fox --box 150,40,850,1000; plan part head --box 300,80,700,450 --focus 3; "
                  "plan light top-left; palette make; palette pick 1"))
    ok(draw("stage", "done"))
    ok(draw("sketch", "head", "fox"))
    ok(draw("stage", "done", "--because", "test"))
    face = [t for t in state()["todo"] if t["key"] == "face"][0]
    assert "muzzle" in face["text"] and "--kind animal" in face["cmd"]
    text = ok(draw("face", "s1"))
    assert "muzzle" in text and "brow" not in text.split("lines:")[1].split("\n")[0]
    from pixelcanvas import draw_checks
    assert not draw_checks.face_holes(project())


# ============================================================================================ G4 / G7 flat

def test_part_without_own_area_blocks_flat_and_flat_takes_a_point(draw):
    bust(draw, "; plan part torso --box 150,600,850,1000")
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("line", "340,200 330,350 380,470 500,520 620,470 670,350 660,200 500,150", "--closed"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("flat", "auto"))
    code, text = draw("stage", "done")
    assert code == 1 and "torso has no area of its own" in text
    # the leak is named with coordinates: the box sides the background crosses
    assert "runs out of it at" in text
    ok(draw("line", "200,620 800,620 800,990 200,990", "--closed"))
    text = ok(draw("flat", "500,800", "--color", "torso"))         # a point, not area numbers
    assert "as a point" in text
    assert "torso" in state()["regions"]


# ============================================================================================ F2 parts apart

def test_head_floating_over_the_robe_blocks_flat(draw):
    ok(draw("new", "pic", "--size", "400x400", "--request", "anime chibi mage, flat cel"))
    ok(draw("do", "plan subject mage --box 100,40,900,1000; plan part head --box 300,60,700,460 --focus 3; "
                  "plan part robe --box 250,400,750,950; plan light top-left; palette make; palette pick 1"))
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("line", "350,100 650,100 650,380 350,380", "--closed", "--no-join"))
    ok(draw("line", "300,480 700,480 700,940 300,940", "--closed", "--no-join"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("flat", "auto"))
    code, text = draw("stage", "done")
    assert code == 1 and "head and robe do not touch" in text and "draw a neck" in text


# ============================================================================================ G5 background inside

def test_background_sliver_inside_a_subject_blocks_and_flat_hole_fills_it(draw):
    bust(draw, "; plan part torso --box 150,600,850,1000")
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("line", "200,600 800,600 800,980 200,980", "--closed", "--no-join"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("flat", "auto"))
    ok(draw("flat", "--at", "500,800", "--color", "background"))
    ok(draw("shape", "poly", "230,630 770,630 770,950 230,950", "--color", "#556655", "--layer", "flat"))
    from pixelcanvas import draw_checks
    hs = draw_checks.background_holes(project())
    assert hs and "flat hole" in hs[0]["fix"]
    code, text = draw("stage", "done")
    assert code == 1 and "background colour inside girl" in text
    ok(draw(*hs[0]["fix"].split()[1:]))
    assert len(draw_checks.background_holes(project())) < len(hs)


# ============================================================================================ G2 detail

def test_plain_focus_part_blocks_detail(draw):
    from pixelcanvas import draw_checks
    bust(draw, "; plan part hair --box 200,40,800,800")
    p = project()
    dens = {"head": (0.03, 3), "hair": (0.09, 2)}
    orig = draw_checks.focus_density
    try:
        draw_checks.focus_density = lambda p_: (dens, 0.09)
        hs = draw_checks.detail_holes(p)
    finally:
        draw_checks.focus_density = orig
    assert hs and hs[0]["metric"] == "focus detail" and "head" in hs[0]["text"]


# ============================================================================================ L3 / P5 subject box

def test_subject_that_misses_its_box_blocks(draw):
    ok(draw("new", "pic", "--size", "400x500", "--request", "minimalist jazz poster: saxophone silhouette"))
    ok(draw("do", "plan subject saxophone --box 140,150,860,770; plan part bell --box 560,540,860,770; "
                  "plan light top-left; palette make; palette pick 1"))
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("band", "sax", "650,200 760,200 760,740 650,740", "--closed", "--color", "bell"))
    from pixelcanvas import draw_checks
    hs = draw_checks.fill_holes(project())
    assert hs and hs[0]["text"].startswith("saxophone: planned box 140,150,860,770, marks cover")
    # a big shape off centre: the fix moves the band
    ok(draw("band", "sax", "420,170 850,170 850,760 420,760", "--closed", "--color", "bell"))
    hs = draw_checks.fill_holes(project())
    assert hs and "mass sits" in hs[0]["text"] and hs[0]["fix"].startswith("pixelcanvas band sax --shift -")
    before = project().region_mask("sax")
    ok(draw(*hs[0]["fix"].split()[1:]))
    after = project().region_mask("sax")
    xs0, xs1 = np.nonzero(before)[1].mean(), np.nonzero(after)[1].mean()
    assert xs1 < xs0 - 20
    assert not [h for h in draw_checks.fill_holes(project()) if h["metric"] == "subject centre"]


# ============================================================================================ L2 depth

def test_farther_band_goes_behind_and_shape_over_nearer_is_gated(draw):
    ok(draw("new", "lake", "--size", "400x300", "--request", "a mountain lake at sunset, painterly"))
    ok(draw("do", "plan subject sky --box 0,0,1000,500; plan part sun --of sky --box 560,200,720,420; "
                  "plan subject mountains --box 0,250,1000,560; plan light left; palette make --base orange; "
                  "palette pick 1"))
    text = ok(draw("plan"))
    assert "depth, far to near: sky, mountains" in text
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("band", "mountains", "0,450 300,300 520,380 700,280 1000,450 1000,560 0,560", "--closed",
            "--color", "mountains"))
    text = ok(draw("band", "sun", "--ellipse", "560,200,720,420", "--color", "sun"))
    assert "behind mountains" in text
    p = project("lake")
    sun, mts = p.region_mask("sun"), p.region_mask("mountains")
    assert sun.any() and not (sun & mts).any()
    from pixelcanvas import draw_checks
    assert not draw_checks.depth_holes(p)
    ok(draw("shape", "ellipse", "580,250,700,400", "--color", "sun"))
    hs = draw_checks.depth_holes(project("lake"))
    assert hs and "drawn over mountains" in hs[0]["text"] and "--behind mountains" in hs[0]["fix"]
    # the fix draws the sun behind and clears the sun colour from the mountains
    ok(draw(*__import__("shlex").split(hs[0]["fix"])[1:]))
    assert not draw_checks.depth_holes(project("lake"))
    ok(draw("plan", "order", "mountains,sky"))
    assert state("lake")["draw"]["order"] == ["mountains", "sky"]


def test_placeholder_band_is_not_the_subject(draw):
    ok(draw("new", "lake", "--size", "400x300", "--request", "a lake with pine trees, painterly"))
    ok(draw("do", "plan subject lake --box 0,400,1000,700; plan subject pines --box 0,540,1000,1000; "
                  "plan light left; palette make; palette pick 1"))
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("band", "pines", "0,655 1000,655 1000,1000 0,1000", "--closed", "--color", "pines"))
    from pixelcanvas import draw_checks
    hs = draw_checks.fill_holes(project("lake"))
    assert hs and "plain box-shaped band pines" in hs[0]["text"]
    for x in (120, 340, 600, 860):
        ok(draw("band", f"pines-{x}", f"{x},990 {x},600", "--tiers", "5", "--width", "220", "--color", "pines"))
    assert not draw_checks.fill_holes(project("lake"))


def test_no_neck_under_the_chin_is_gated():
    from types import SimpleNamespace
    from pixelcanvas import draw_checks
    H = W = 200
    masks = {n: np.zeros((H, W), bool) for n in ("head", "hair", "scarf")}
    masks["head"][40:100, 70:130] = True
    masks["hair"][20:160, 40:160] = True
    masks["hair"] &= ~masks["head"]
    masks["scarf"][120:140, 50:150] = True
    masks["hair"] &= ~masks["scarf"]
    parts = {"head": {"of": "girl", "box": [350, 200, 650, 500], "focus": 3, "swatch": "skin"},
             "hair": {"of": "girl", "box": [200, 100, 800, 800], "focus": 2, "swatch": "hair"},
             "scarf": {"of": "girl", "box": [250, 600, 750, 700], "focus": 2, "swatch": "scarf"}}
    st = {"request": "anime girl", "regions": {n: {} for n in masks},
          "draw": {"subjects": {"girl": {"box": [100, 50, 900, 1000]}}, "parts": parts}}
    p = SimpleNamespace(H=H, W=W, state=st, region_mask=lambda n: masks[n])
    hs = draw_checks.neck_holes(p)
    assert hs and "has no neck" in hs[0]["text"] and "--part neck" in hs[0]["text"]
    masks["neck"] = np.zeros((H, W), bool)
    masks["neck"][100:120, 85:115] = True
    st["regions"]["neck"] = {}
    assert not draw_checks.neck_holes(p)


# ============================================================================================ G6 clipping

def test_hair_strands_start_near_the_flow_curve():
    from pixelcanvas import draw_detail
    m = np.zeros((200, 200), bool)
    m[20:190, 20:190] = True
    curve = np.array([[30.0, 40.0], [100.0, 42.0], [180.0, 40.0]])
    ops = draw_detail.hair_ops(m, [curve], (200, 200, 200), (80, 80, 80), (250, 250, 250), np.array([-0.7, -0.7]),
                               count=40, width=2.0, shine=False, near=15.0)
    assert ops
    ys = np.concatenate([np.asarray(o.pts)[:, 1] for o in ops])
    assert ys.max() < 40 + 15 + 6           # nothing sprays down over the rest of the hair


def test_fold_without_region_stays_in_the_area_under_it(draw):
    bust(draw)
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("line", "340,200 330,350 380,470 500,520 620,470 670,350 660,200 500,150", "--closed"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("flat", "auto"))
    text = ok(draw("fold", "400,300 500,320 600,300"))
    assert "stays inside head" in text


# ============================================================================================ P1 plan view in a batch

def test_plan_batch_shows_the_plan_view(draw):
    ok(draw("new", "pic", "--size", "300x300", "--request", "a red apple"))
    text = ok(draw("do", "plan subject apple --box 200,200,800,800; plan part stem --box 450,150,550,260"))
    assert "VIEW:" in text and "plan-" in text.split("VIEW:")[1].split("\n")[0]


# ============================================================================================ finish

def test_finish_wants_a_fresh_export(draw):
    from pixelcanvas import workflow_draw
    bust(draw)
    p = project()
    p.state["stage"] = "finish"
    p.save_state()
    ok(draw("export", "final.png"))
    ok(draw("shape", "ellipse", "10,10,60,60", "--color", "#ff0000"))
    assert workflow_draw._export_stale(project())
    ok(draw("export", "final.png"))
    assert not workflow_draw._export_stale(project())
