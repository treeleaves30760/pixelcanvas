"""Behaviour added after the hand-run trial sessions (see DESIGN.md, decisions)."""
import cv2
import numpy as np
import pytest

from pixelcanvas import cli, metrics, regions, workflow
from pixelcanvas.project import Project

from .shell_helpers import install_fake_engine, run, write_ref


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_engine(monkeypatch)
    write_ref(tmp_path / "ref.png")
    code, out = run("new", "pic", "--ref", "ref.png")
    assert code == 0, out
    return tmp_path / "pic.pixelcanvas"


def test_marks_keep_same_colour_areas_apart_across_lines():
    """Two white areas separated only by a dark outline stay two marks (face vs background)."""
    img = np.full((300, 300, 3), 250, np.uint8)
    cv2.circle(img, (150, 150), 90, (20, 20, 20), 3)
    labels, _ = regions.compute_marks(img, 4)
    assert labels[150, 150] != labels[10, 10]


def test_wand_region_fills_holes_and_takes_its_outline(proj):
    code, out = run("region", "add", "face", "--wand", "500,450", "--tol", "10")
    assert code == 0 and "holes filled" in out, out
    p = Project(proj)
    m = p.region_mask("face")
    ref = p.ref
    dark = (ref.astype(int).sum(2) < 150)
    # the eyes (dark discs inside the face) are in the region, and so is most of the outline ring
    H, W = m.shape
    cx, cy, r = W // 2, int(H * 0.38), int(W * 0.25)
    assert m[cy, cx - r // 3]
    ring = np.zeros_like(m)
    cv2.circle(ring.view(np.uint8), (cx, cy + 5), r, 1, 2)
    ring &= dark
    assert (m & ring).sum() > 0.6 * ring.sum()


def test_plus_minus_accept_boxes(proj):
    run("region", "add", "a", "--box", "0,0,500,500", "--no-refine")
    code, out = run("region", "add", "b", "--box", "0,0,1000,1000", "--no-refine", "--minus", "0,0,500,1000")
    assert code == 0, out
    m = Project(proj).region_mask("b")
    assert not m[:, : m.shape[1] // 2 - 1].any() and m[:, -1].all()


def test_spots_follow_the_stage_kind():
    ref = np.full((200, 200, 3), 240, np.uint8)
    cv2.line(ref, (20, 100), (180, 100), (20, 20, 20), 2)
    can = np.full_like(ref, 240)
    can[10:60, 10:60] = (240, 0, 0)            # a big colour error, no line
    cmp = metrics.measure(ref, can)
    spots = cmp.worst(n=3, emphasis="lines")
    assert spots and all(s["kind"] == "lines" for s in spots)
    assert "lines missing" in spots[0]["problem"]


def test_painterly_styles_have_no_lines_axis(proj):
    ref = np.full((100, 100, 3), 200, np.uint8)
    cv2.line(ref, (0, 50), (100, 50), (0, 0, 0), 2)
    assert metrics.measure(ref, ref.copy(), use_lines=False).scores()["lines"] is None
    run("style", "paint")
    code, out = run("score")
    assert "lines -" in out


def test_fix_commands_escalate():
    st = {"width": 850, "height": 1200}
    fixes = [workflow.fix_command(st, "forms", "path", "structure", t) for t in range(4)]
    assert len(set(fixes)) == 4 and "look path" in fixes[3]
    fixes = [workflow.fix_command(st, "line", "hair", "lines", t) for t in range(3)]
    assert len(set(fixes)) == 3


def test_option_example_becomes_full_command():
    assert cli._full_example("--color #c04030", ["fill", "500,400", "--color", "nope"]) == \
        "pixelcanvas fill 500,400 --color #c04030"
    assert cli._full_example("--color #c04030", ["stroke", "1,2 3,4"]) == \
        "pixelcanvas stroke '1,2 3,4' --color #c04030"


def test_style_guess_paint_for_textured_picture():
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(120, 35, (300, 300, 3)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    assert workflow.guess_style(img)[0] == "paint"


def test_hand_tool_scored_with_neighbourhood(proj):
    code, out = run("stroke", "400,500 600,500", "--color", "#202020", "--size", "2")
    assert code == 0 and "around the stroke" in out


def test_mark_rules_refuse_and_detail_minimum(proj, monkeypatch):
    import sys
    ops_mod = sys.modules["pixelcanvas.ops"]
    calls = []

    def check(ops, stage_tool=True):
        calls.append(stage_tool)
        return ["op 0: pixel dust"]
    monkeypatch.setattr(ops_mod, "check_mark_rules", check, raising=False)
    before = (proj / "layers" / "line.png").read_bytes()
    code, out = run("ink", "all")
    assert code == 1 and "mark rules" in out and "nothing was painted" in out
    assert (proj / "layers" / "line.png").read_bytes() == before and calls == [True]
    code, out = run("stroke", "100,100 300,100", "--color", "red")
    assert code == 1 and calls[-1] is False
    monkeypatch.setattr(ops_mod, "check_mark_rules", lambda ops, stage_tool=True: [], raising=False)
    # the minimum is 1 px: a 1 px stroke is a legal mark (DESIGN.md 10.1) and the pipeline's last pass
    code, out = run("detail", "all", "--size", "0.5")
    assert code == 0 and "using 1" in out and "--size 1.0" in out


def test_stroke_pick_is_one_colour(proj, monkeypatch):
    import sys
    seen = []
    ops_mod = sys.modules["pixelcanvas.ops"]
    real = ops_mod.apply_ops

    def spy(layer, ops, **kw):
        seen.extend(ops)
        return real(layer, ops, **kw)
    monkeypatch.setattr(ops_mod, "apply_ops", spy)
    code, out = run("stroke", "100,700 900,700", "--color", "pick", "--size", "4")
    assert code == 0, out
    assert not isinstance(seen[0].color, np.ndarray) and len(seen[0].color) == 3
