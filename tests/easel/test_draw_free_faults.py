"""Drawing from a request: water on a curved shore, the lake's surface, bands named after their subject, the view
through a window, animal faces and sketches, eyes and face features on a small figure, and a detail gate that does
not ask for texture."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pixelcanvas.easel import cli, workflow


@pytest.fixture
def draw(tmp_path, monkeypatch):
    from pixelcanvas.easel import cli_draw
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
    return json.load(open(f"{name}.easel/state.json", encoding="utf-8"))


def project(name="pic"):
    from pixelcanvas.easel.project import Project
    return Project.find(f"{name}.easel")


def lum(img):
    img = np.asarray(img, np.float64)
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def lake_scene(run, light="15"):
    ok(run("new", "pic", "--size", "400x300", "--request", "a mountain lake at sunset with pines"))
    ok(run("do", "plan subject sky --box 0,0,1000,520; plan subject lake --box 0,500,1000,900; "
                 "plan subject pines --box 0,250,300,950; plan subject shore --box 0,800,1000,1000; "
                 f"plan order sky,lake,shore,pines; plan light {light}; palette make --base orange; palette pick 1"))
    ok(run("palette", "set", "sky", "#f4a25a", "lake", "#5b5f8f", "pines", "#1f3328", "shore", "#2e3a2a"))
    ok(run("band", "sky", "0,520 1000,520", "--above", "--color", "sky"))


# ============================================================================================ water

def test_reflect_on_a_curved_shore_and_a_narrow_tip():
    from pixelcanvas.easel.draw_scene import reflect_ops
    H, W = 300, 400
    yy, xx = np.mgrid[:H, :W]
    vee = (yy > 150) & (yy < 290) & (np.abs(xx - 200) < (290 - yy) * 1.4)      # narrows to a point
    bay = (yy > 150) & (yy < 280) & ~((yy > 200) & (np.abs(xx - 200) < 40))    # a spit of land in the middle
    src = np.random.default_rng(1).random((H, W, 3)) * 255
    for m in (vee, bay):
        for seed in range(4):
            ops, n_refl, n_path = reflect_ops(m, src, 150.0, (90, 90, 140), 10.0, 0.75, 200.0, (255, 210, 140), seed)
            assert n_refl > 50 and n_path > 10
            for o in ops:                                    # every mirror stroke lies on water
                if hasattr(o, "pts") and o.pts is not None and len(o.pts) > 1 and n_refl:
                    mid = np.asarray(o.pts, float).mean(0)
                    x, y = int(np.clip(mid[0], 0, W - 1)), int(np.clip(mid[1], 0, H - 1))
                    if not m[y, x]:
                        assert np.ptp(np.asarray(o.pts)[:, 0]) < 60


def test_reflect_command_on_a_lake_with_a_curved_shore(draw):
    lake_scene(draw)
    # the far shore curves down to a narrow inlet at the front
    ok(draw("band", "lake", "0,520 1000,520 1000,600 560,880 510,940 490,940 440,880 0,600", "--closed", "--color",
            "lake"))
    text = ok(draw("reflect", "lake", "--glow", "700"))
    assert "strokes mirror the scene" in text and "sun path at x 700" in text


def test_water_surface_lays_no_pale_slab_on_the_lit_side(draw):
    import gzip
    import re
    from pixelcanvas.easel.ops import Fill, ops_from_jsonable, rle_to_mask
    lake_scene(draw)
    ok(draw("band", "lake", "0,520 1000,520 1000,880 0,880", "--closed", "--color", "lake"))
    text = ok(draw("surface", "lake", "--material", "water"))
    n = int(re.search(r"step (\d+)", text).group(1))
    with gzip.open(f"pic.easel/ops/{n:04d}.json.gz", "rt", encoding="utf-8") as f:
        ops = ops_from_jsonable(json.load(f)["ops"])
    lake = project().region_mask("lake")
    # water is a flat plane: no light or shadow shape from the lake's outline laid over a big part of it (a form
    # field lit from the side made a pale slab over the lake)
    big = [o for o in ops if isinstance(o, Fill) and o.mask_rle is not None
           and (rle_to_mask(o.mask_rle) & lake).sum() > 0.25 * lake.sum()]
    assert not big, [(o.color, o.opacity) for o in big]
    assert len(ops) > 20                                       # the ripples are there


# ============================================================================================ names and depth

def test_bands_named_after_a_subject_belong_to_it(draw):
    from pixelcanvas.easel import draw_checks
    lake_scene(draw)
    ok(draw("band", "lake", "0,520 1000,520 1000,880 0,880", "--closed", "--color", "lake"))
    # a tree of the pines whose middle lies outside the pines box (it leans out over the lake)
    ok(draw("band", "pines-l1", "420,900 420,450", "--tiers", "5", "--width", "160", "--color", "pines"))
    p = project()
    assert draw_checks.subject_of(p, "pines-l1") == "pines"
    assert draw_checks.subject_of(p, "pines2-b") is None or draw_checks.subject_of(p, "pines2-b") != "pines"
    near, names = draw_checks.nearer_mask(p, "lake")
    assert "pines" in names and (near & p.region_mask("pines-l1")).sum() == p.region_mask("pines-l1").sum()
    # the shore is farther than the pines: drawn after them, it stays behind pines-l1
    text = ok(draw("band", "shore", "0,860 1000,860 1000,1000 0,1000", "--closed", "--color", "shore"))
    assert "behind pines" in text
    assert not (project().region_mask("shore") & project().region_mask("pines-l1")).any()
    # shore colour painted over the tree is a depth hole of pines
    ok(draw("shape", "ellipse", "390,700,450,760", "--color", "shore"))
    hs = draw_checks.depth_holes(project())
    assert hs and "drawn over pines" in hs[0]["text"]


def test_the_suns_reflection_in_the_lake_is_not_a_depth_hole(draw):
    from pixelcanvas.easel import draw_checks
    lake_scene(draw)
    ok(draw("do", "plan part sun --of sky --box 640,320,760,430; palette set sun #fff0c0"))
    ok(draw("band", "sun", "--ellipse", "656,330,744,418", "--color", "sun"))
    ok(draw("band", "lake", "0,520 1000,520 1000,880 0,880", "--closed", "--color", "lake"))
    ok(draw("shape", "ellipse", "670,600,730,820", "--color", "sun"))         # the sun's path down the water
    assert not draw_checks.depth_holes(project())
    ok(draw("shape", "ellipse", "100,600,200,700", "--color", "sun"))         # a sun disc far from its column
    hs = draw_checks.depth_holes(project())
    assert hs and "drawn over lake" in hs[0]["text"]


def test_view_through_the_window_stays_behind_the_frame(draw):
    ok(draw("new", "pic", "--size", "400x400", "--request", "a cat asleep on a windowsill, painterly"))
    ok(draw("do", "plan subject wall --box 0,0,1000,1000; plan subject window --box 200,100,800,700; "
                  "plan light top-left; palette make; palette pick 1; "
                  'palette set wall "#b98563" window "#cfe3ee"'))
    ok(draw("band", "wall", "0,0 1000,0 1000,1000 0,1000", "--closed", "--color", "wall"))
    ok(draw("band", "window", "200,100 800,100 800,700 200,700", "--closed", "--color", "window"))
    ok(draw("band", "hills", "230,560 400,480 600,520 770,470 770,670 230,670", "--closed", "--color", "#8fb08a",
            "--front"))
    text = ok(draw("window", "--box", "200,100,800,700", "--panes", "2x2", "--color", "#f0ece0"))
    assert "behind hills" not in text
    img = project().composite_u8("export").astype(float)
    # the middle bar and the bottom of the frame run over the hills
    assert np.abs(img[240, 200] - (240, 236, 224)).max() < 40
    assert np.abs(img[274, 300] - (240, 236, 224)).max() < 40


# ============================================================================================ animals

def fox_mage(run):
    ok(run("new", "pic", "--size", "400x480", "--request", "a chibi fox mage holding a staff"))
    ok(run("style", "cel"))
    ok(run("do", "plan subject fox --box 150,30,700,990; plan part head --of fox --box 300,100,665,440 --focus 3 "
                 "--color fur; plan part eye-l --of fox --box 330,240,420,320 --focus 3; plan part eye-r --of fox "
                 "--box 540,240,630,320 --focus 3; plan part robe --of fox --box 240,440,720,930; "
                 "plan light top-left; palette make --base orange; palette pick 1; stage done"))
    ok(run("sketch", "figure", "fox", "--pose", "hold", "--heads", "2.5"))
    ok(run("sketch", "head", "fox"))
    ok(run("stage", "done"))


def test_animal_face_and_clothes_draw_no_human_neck(draw):
    fox_mage(draw)
    text = ok(draw("face", "s2", "--kind", "animal"))
    assert "neck" not in text
    text = ok(draw("wear", "fox", "robe"))
    st = state()
    assert not [r for r in st["draw"]["lines"] if str(r.get("part") or "").startswith("neck")]
    assert "neck" not in st["draw"]["parts"]
    # a human face on a figure still gets its neck
    ok(draw("new", "girl", "--size", "400x480", "--request", "an anime girl, bust shot"))
    ok(draw("do", "plan subject girl --box 140,40,860,1000; plan part head --of girl --box 360,110,640,440 "
                  "--focus 3; plan light top-left; palette make; palette pick 1; stage done"))
    ok(draw("sketch", "figure", "girl", "--pose", "bust"))
    ok(draw("sketch", "head", "girl"))
    ok(draw("stage", "done"))
    assert "neck of girl" in ok(draw("face", "s2"))


def cat_plan(run):
    ok(run("new", "pic", "--size", "480x360", "--request", "a cat asleep on a windowsill"))
    ok(run("do", "plan subject sill --box 0,620,1000,780; plan subject cat --box 310,430,710,725; "
                 "plan part head --of cat --box 555,430,705,615 --focus 3; plan part tail --of cat "
                 "--box 320,640,660,725; plan light 150; palette make --base orange; palette pick 1"))


def test_animal_sketch_asks_for_its_shapes_not_a_standing_mannequin(draw):
    cat_plan(draw)
    ok(draw("stage", "done"))
    items = [t for t in state()["todo"] if t["stage"] == "sketch" and t.get("region") == "cat"]
    assert items and not any("--pose stand" in t["cmd"] or "mannequin" in t["text"] for t in items)
    ok(draw("sketch", "ellipse", "320,500,660,710"))
    ok(draw("sketch", "figure", "cat", "--pose", "sit"))          # a mannequin as a guide: no human proportions
    ok(draw("sketch", "head", "cat", "--box", "555,460,705,612"))
    ok(draw("sketch", "horizon", "630"))
    from pixelcanvas.easel import draw_checks
    assert not draw_checks.sketch_holes(project())
    text = ok(draw("stage", "done"))
    assert "stage sketch closed" in text


def test_dressed_animal_still_gets_a_mannequin(draw):
    ok(draw("new", "pic", "--size", "400x480", "--request", "a chibi fox mage holding a staff"))
    ok(draw("style", "cel"))
    ok(draw("do", "plan subject fox --box 150,30,700,990; plan part head --of fox --box 300,100,665,440 --focus 3; "
                  "plan part robe --of fox --box 240,440,720,930; plan light top-left; palette make; palette pick 1; "
                  "stage done"))
    items = [t for t in state()["todo"] if t["stage"] == "sketch" and t.get("region") == "fox"]
    assert any("mannequin" in t["text"] for t in items)


def test_a_title_heading_is_not_a_figure_and_scenery_parts_are_bands(draw):
    ok(draw("new", "pic", "--size", "450x600", "--request", "an event poster with a title and a lighthouse"))
    ok(draw("style", "cel"))
    ok(draw("do", "plan subject title --box 60,40,940,230; plan part heading --of title --box 60,40,940,230; "
                  "plan subject scene --box 0,0,1000,1000; plan part sky --of scene --box 0,0,1000,700; "
                  "plan part cliff --of scene --box 0,700,600,1000; plan light top-left; palette make; "
                  "palette pick 1"))
    todo = state()["todo"]
    assert not any("mannequin" in t["text"] for t in todo)
    assert not any(t["stage"] == "line" and t.get("region") in ("sky", "cliff") for t in todo)
    bands = [t for t in todo if t["stage"] == "flat" and t.get("key") == "mass"]
    assert {t["region"] for t in bands} == {"sky", "cliff"} and bands[0]["cmd"].startswith("easel band")


def test_similar_colours_and_a_wet_edge_are_not_depth_holes(draw):
    from pixelcanvas.easel import draw_checks
    ok(draw("new", "pic", "--size", "400x300", "--request", "two potted plants on a sill, painterly"))
    ok(draw("do", "plan subject plant-l --box 50,200,450,900; plan subject plant-r --box 550,200,950,900; "
                  "plan order plant-l,plant-r; plan light top-left; palette make; palette pick 1; "
                  'palette set plant-l "#4e7d42" plant-r "#3f6e3a"'))
    ok(draw("band", "plant-l", "--ellipse", "50,200,450,900", "--color", "plant-l"))
    ok(draw("band", "plant-r", "--ellipse", "550,200,950,900", "--color", "plant-r"))
    ok(draw("shape", "ellipse", "700,400,800,500", "--color", "plant-l"))    # a lighter leaf inside plant-r
    assert not draw_checks.depth_holes(project())


def test_a_night_sea_band_is_not_background_showing_through(draw):
    from pixelcanvas.easel import draw_checks
    ok(draw("new", "pic", "--size", "450x600", "--request", "a lighthouse at night, poster"))
    ok(draw("style", "paint"))
    ok(draw("do", "plan subject sky --box 0,0,1000,640; plan subject sea --box 0,600,1000,1000; plan light 160; "
                  'palette make --base "#1d2b55"; palette pick 1; palette set sky "#14204a" sea "#1a2a4f"'))
    ok(draw("band", "sky", "0,640 1000,640", "--above", "--color", "sky"))
    ok(draw("band", "sea", "0,640 1000,640 1000,1000 0,1000", "--closed", "--color", "sea", "--color2", "#0c1530"))
    assert not draw_checks.background_holes(project())


def test_todo_is_the_same_after_a_mid_stage_plan_change(draw):
    import copy
    from pixelcanvas.easel import workflow_draw
    cat_plan(draw)
    ok(draw("stage", "done"))
    ok(draw("sketch", "ellipse", "320,500,660,710"))
    ok(draw("sketch", "head", "cat", "--box", "555,460,705,612"))
    ok(draw("sketch", "box", "0,630,1000,770"))
    ok(draw("sketch", "horizon", "630"))
    ok(draw("stage", "done"))
    ok(draw("do", 'band sill "0,630 1000,630 1000,780 0,780" --closed --color sill; '
                  "band cat --ellipse 320,520,660,712 --color #e39a52 --front; "
                  'band ear-l "575,500 590,435 625,485" --closed --color #e39a52 --front; '
                  "band cat-shadow --ellipse 330,690,690,730 --color #5e3d28; "
                  'band glow-spot --ellipse 40,40,120,100 --color #ffe2a8'))
    ok(draw("stage", "done", "--because", "test"))
    st = state()
    assert st["stage"] == "forms"

    def items(s):
        return sorted((t["stage"], str(t.get("region")), str(t["key"]), t["status"]) for t in s["todo"] if t["kind"] == "task")
    before = items(st)
    fresh = copy.deepcopy(st)
    workflow_draw.build_todo(fresh)
    assert items(fresh) == before                       # the list is what a fresh build gives
    ok(draw("plan", "light", "150"))                   # a plan change rebuilds the list mid-stage
    assert items(state()) == before
    forms = {t["region"] for t in state()["todo"] if t["stage"] == "forms" and t["kind"] == "task"}
    # subjects get brushwork; the ears, the cat's shadow and a small loose accent need none of their own
    assert {"cat", "sill"} <= forms and not forms & {"ear-l", "cat-shadow", "glow-spot"}


# ============================================================================================ a small figure

def small_hero(run):
    ok(run("new", "pic", "--size", "900x1200", "--request", "a full-body adventurer in a cloak standing on a cliff"))
    ok(run("style", "cel"))
    ok(run("do", "plan subject hero --box 260,150,600,880; plan part head --of hero --box 380,160,480,270 --focus 3; "
                 "plan part eye-l --of hero --box 397,200,425,218 --focus 3; plan part eye-r --of hero "
                 "--box 435,200,463,218 --focus 3; plan part hair --of hero --box 365,150,495,260 --focus 2; "
                 "plan part cloak --of hero --box 280,250,600,800 --focus 2; plan light top-right; palette make; "
                 "palette pick 2; stage done"))
    ok(run("sketch", "figure", "hero", "--pose", "stand", "--heads", "7"))
    ok(run("sketch", "head", "hero"))
    ok(run("stage", "done"))


def test_small_face_keeps_its_nose_and_brows_and_eyes(draw):
    from pixelcanvas.easel import draw_checks
    small_hero(draw)
    ok(draw("line", "trace", "s2"))
    ok(draw("face", "s2"))
    st = state()
    parts = {r["src"]: r.get("part") for r in st["draw"]["lines"] if str(r.get("src", "")).startswith("face ")}
    assert parts["face nose"] == "head" and parts["face brow-l"] == "head" and parts["face brow-r"] == "head"
    assert parts["face eye-l"] == "eye-l" and parts["face eye-r"] == "eye-r"
    p = project()
    assert not draw_checks.face_holes(p)
    assert not [h for h in draw_checks.eye_holes(p) if h["metric"] == "eye size"]


def test_small_figure_eyes_are_not_filled_with_the_hood(draw):
    small_hero(draw)
    ok(draw("line", "trace", "s2"))
    ok(draw("face", "s2"))
    ok(draw("wear", "hero", "cloak"))
    ok(draw("wear", "hero", "hood"))
    ok(draw("line", "gaps"))
    from pixelcanvas.easel import draw_flat
    p = project()
    labels, table, _ = draw_flat.compute_areas(p)
    got = draw_flat.auto_assign(p, labels, table)
    from pixelcanvas.easel.draw_core import px2u
    eyes = []
    for r in table:
        u = px2u(p, [r["anchor"]])[0]
        for n in ("eye-l", "eye-r"):
            b = state()["draw"]["parts"][n]["box"]
            if b[0] - 5 <= u[0] <= b[2] + 5 and b[1] - 5 <= u[1] <= b[3] + 5 and r["share"] < 0.002:
                eyes.append((n, got.get(r["n"])))
    assert eyes and all(g is not None and g[0] == n for n, g in eyes), eyes
    # the hood closes round the face: the neck under it is neither asked for nor asked an area
    from pixelcanvas.easel import draw_checks
    assert draw_checks.hood_up(p, "hero")
    assert not draw_checks.neck_holes(p)
    assert not [h for h in draw_checks.part_area_holes(p) if str(h.get("region") or "").startswith("neck")]


# ============================================================================================ detail gate

def test_free_detail_gate_does_not_ask_for_texture(draw, monkeypatch):
    from pixelcanvas.easel import detail_measure, draw_surface

    def entry(key, low, value, q1, q3):
        return {"key": key, "low": low, "value": value, "q1": q1, "q3": q3, "z": -3.0 if low else 3.0,
                "text": f"{key} {'low' if low else 'high'} (fake)"}
    ok(draw("new", "pic", "--size", "700x700", "--request", "a mountain lake at sunset"))
    ok(draw("do", "plan subject sky --box 0,0,1000,600; plan light left; palette make; palette pick 1"))
    ok(draw("band", "sky", "0,600 1000,600", "--above", "--color", "sky"))
    ok(draw("surface", "auto"))
    monkeypatch.setattr(draw_surface, "picture_measure", lambda p, keep=True, lines=True: {})
    p = project()
    for low_texture in (entry("bandL_1", True, 0.001, 0.01, 0.03), entry("det_empty", False, 0.5, 0.0, 0.05),
                        entry("flat_share", False, 0.35, 0.0, 0.02), entry("line_taper", True, 0.36, 0.52, 0.66)):
        monkeypatch.setattr(detail_measure, "outside", lambda m, group, keys=None, x=low_texture: [x])
        assert draw_surface.gate_holes(p, "detail") == [], low_texture["key"]
    # a picture turned into noise is still held
    monkeypatch.setattr(detail_measure, "outside",
                        lambda m, group, keys=None: [entry("bandL_1", False, 0.2, 0.01, 0.03)])
    hs = draw_surface.gate_holes(p, "detail")
    assert hs and hs[0]["metric"] == "detail bandL_1"


def test_focus_head_with_its_face_is_not_asked_for_more_detail(draw):
    from pixelcanvas.easel import draw_checks
    fox_mage(draw)
    ok(draw("line", "trace", "s2"))
    ok(draw("face", "s2", "--kind", "animal"))
    p = project()
    assert not draw_checks.face_missing(p, draw_checks.head_elements(p)[-1])
    dens = {"head": (0.013, 3), "robe": (0.025, 2)}
    orig = draw_checks.focus_density
    try:
        draw_checks.focus_density = lambda p_: (dens, 0.025)
        assert not draw_checks.detail_holes(p)
    finally:
        draw_checks.focus_density = orig


# ============================================================================================ stable seeds

def _paint_wall(run, name, extra):
    ok(run("new", name, "--size", "300x240", "--request", "a cat asleep on a windowsill, painterly"))
    ok(run("do", "plan subject wall --box 0,0,1000,1000; plan light 150; palette make; palette pick 1"))
    ok(run("band", "wall", "0,0 1000,0 1000,1000 0,1000", "--closed", "--color", "#b98563"))
    for cmd in extra:
        ok(run(*cmd))
    ok(run("paint", "wall", "--dir", "90"))
    ok(run("paint", "wall", "--dir", "90"))
    return project(name).layer("paint").copy()


def test_brush_marks_follow_the_command_not_the_step_count(draw):
    a = _paint_wall(draw, "a", [])
    b = _paint_wall(draw, "b", [("plan", "light", "150"), ("palette", "set", "sky", "#a0c0e0")])
    assert np.array_equal(a, b)                          # two unrelated commands earlier change nothing
    # the same command run twice lays different marks the second time
    from pixelcanvas.easel import draw_core as dc
    p = project("a")
    assert dc.stable_seed(p, "easel paint wall --dir 90", "x") != dc.stable_seed(p, "easel paint wall --dir 80", "x")


def test_a_mark_part_is_detail_itself_and_gets_no_detail_item(draw):
    import copy
    from pixelcanvas.easel import workflow_draw
    cat_plan(draw)
    ok(draw("plan", "part", "head-muzzle", "--of", "cat", "--box", "640,540,700,600", "--focus", "3"))
    st = copy.deepcopy(state())
    st["stage"] = "detail"

    def detail_regions(s):
        workflow_draw.build_todo(s)
        return {t.get("region") for t in s["todo"] if t["stage"] == "detail" and t["kind"] == "task"}
    assert "head-muzzle" in detail_regions(copy.deepcopy(st))
    st["draw"].setdefault("marks", {})["head-muzzle"] = {"parent": "head", "kind": "muzzle", "color": "cat"}
    regions = detail_regions(st)
    assert "head-muzzle" not in regions and "head" in regions


def test_a_thin_blend_line_of_the_far_colour_is_not_a_spill(draw):
    from pixelcanvas.easel import draw_checks
    ok(draw("new", "lake", "--size", "400x300", "--request", "a mountain lake at sunset, painterly"))
    ok(draw("do", "plan subject sky --box 0,0,1000,500; plan part sun --of sky --box 560,200,720,420; "
                  "plan subject mountains --box 0,250,1000,560; plan light left; palette make --base orange; "
                  "palette pick 1"))
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("band", "mountains", "0,450 300,300 520,380 700,280 1000,450 1000,560 0,560", "--closed",
            "--color", "mountains"))
    ok(draw("band", "sun", "--ellipse", "560,200,720,420", "--color", "sun"))
    # a line a few pixels wide in the sun's colour across the mountains: where two of their colours blend
    ok(draw("shape", "poly", "50,494 950,494 950,507 50,507", "--color", "sun"))
    p = project("lake")
    assert not draw_checks.depth_holes(p)
    # a patch of the same colour is a spill
    ok(draw("shape", "ellipse", "580,250,700,400", "--color", "sun"))
    hs = draw_checks.depth_holes(project("lake"))
    assert hs and "drawn over mountains" in hs[0]["text"]
