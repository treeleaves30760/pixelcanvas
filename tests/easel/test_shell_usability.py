"""Usability changes made while writing the easel skill and watching Qwen sessions: two separate view pictures,
view sizes per type, batch regression warnings, the plan list for the agent's todo tool, consistent colour
diagnoses, the no-reference path."""
import re

import numpy as np
import pytest
from PIL import Image

from pixelcanvas.easel import metrics, view, workflow
from pixelcanvas.easel.project import Project

from .shell_helpers import install_fake_engine, run, view_path, write_ref


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_engine(monkeypatch)
    write_ref(tmp_path / "ref.png")
    code, out = run("new", "pic", "--ref", "ref.png")
    assert code == 0, out
    return tmp_path / "pic.easel"


def _ref_path(out):
    m = re.search(r"^VIEW-REF: (\S+)", out, re.M)
    return m.group(1) if m else None


def _plan(tmp_path):
    run("region", "add", "top", "--box", "0,0,1000,400", "--no-refine")
    run("region", "add", "bottom", "--box", "0,400,1000,800", "--no-refine", "--focus", "3")
    run("region", "rest", "ground")
    code, out = run("stage", "done")
    assert code == 0, out
    return out


def test_step_writes_canvas_and_reference_as_two_pictures(proj, tmp_path):
    _plan(tmp_path)
    run("stage", "done", "--because", "test")          # sketch
    code, out = run("ink", "top")
    assert code == 0, out
    v, r = view_path(out), _ref_path(out)
    assert v and r and v != r
    for f in (v, r):
        w, h = Image.open(f).size
        assert max(w, h) <= view.VIEW_SIDE and w % 32 == 0 and h % 32 == 0
    assert out.splitlines()[-1].startswith("NEXT: ")
    # the same area gives the same reference file
    code, out2 = run("look", "top")
    assert _ref_path(out2) == _ref_path(run("look", "top")[1])


def test_marks_and_region_maps_are_one_bigger_picture(proj):
    code, out = run("marks")
    w, h = Image.open(view_path(out)).size
    assert view.VIEW_SIDE < max(w, h) <= view.MAP_SIDE
    code, out = run("region", "add", "top", "--box", "0,0,1000,400", "--no-refine")
    assert "tinted" in out
    w, h = Image.open(view_path(out)).size
    assert max(w, h) <= view.MAP_SIDE


def test_region_map_tints_each_region():
    base = np.full((40, 60, 3), 255, np.uint8)
    a = np.zeros((40, 60), bool)
    a[:, :30] = True
    pan = view.region_map(base, [("a", a, 3), ("b", ~a, 1)], "R")
    img = view.render([pan], (0, 0, 60, 40))
    assert len(pan.tints) == 2
    # the two halves get different tints
    h, w = img.shape[:2]
    left, right = img[h // 2, w // 3], img[h // 2, (2 * w) // 3 + 10]
    assert tuple(left) != tuple(right)


def test_batch_flags_a_worse_step_and_names_the_undo_count(proj, tmp_path):
    _plan(tmp_path)
    run("stage", "done", "--because", "test")
    code, out = run("do", "flat all; ink all; shape rect 0,0,1000,1000 --color #ff00ff --layer detail")
    assert code == 0
    assert "WORSE" in out and "WARNING" in out
    assert re.search(r"NEXT: easel undo [12] ", out)


def test_todo_all_gives_one_line_per_stage_and_stage_done_names_the_line(proj, tmp_path):
    out = _plan(tmp_path)
    assert "easel todo --all" in out
    code, out = run("todo", "--all")
    st = Project(proj).state
    plan = workflow.plan_of(st)
    lines = [ln for ln in out.splitlines() if re.match(r"^\d+\. \[", ln)]
    assert len(lines) == len(plan)
    assert lines[0].startswith("1. [done] plan") and "in progress" in lines[1]
    assert "bottom" in lines[-2] or any("close look" in ln and "bottom" in ln for ln in lines)
    code, out = run("stage", "done", "--because", "test")
    assert "mark line 2 (sketch) completed" in out


def test_closing_a_stage_settles_its_open_items(proj, tmp_path):
    _plan(tmp_path)
    run("stage", "done", "--because", "no time")
    st = Project(proj).state
    assert all(t["status"] != "open" for t in st["todo"] if t["stage"] == "sketch")
    run("stage", "back", "sketch")
    st = Project(proj).state
    assert any(t["status"] == "open" for t in st["todo"] if t["stage"] == "sketch" and t["kind"] == "task")


def test_colour_diagnosis_never_contradicts_its_names():
    ref = np.full((96, 96, 3), 240, np.uint8)
    can = ref.copy()
    can[:, 40:44] = 20                      # a stray dark line on a white area: the typical colour is still white
    cmp = metrics.measure(ref, can)
    words = cmp.diagnose((0, 0, 96, 96), metrics.EMPHASIS["colour"])
    m = re.search(r"\(ref (.+?) #\w+, canvas (.+?) #\w+\)", words)
    if m and m.group(1) == m.group(2):
        assert "too dark" not in words and "too light" not in words
    can2 = np.full((96, 96, 3), 120, np.uint8)
    words2 = metrics.measure(ref, can2).diagnose((0, 0, 96, 96), metrics.EMPHASIS["colour"])
    assert words2.startswith("colour too dark") and "#" in words2


def test_without_reference_the_drawing_commands_take_over(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_engine(monkeypatch)
    code, out = run("new", "apple", "--size", "200x200")
    assert code == 0, out
    from pixelcanvas.easel import cli
    assert getattr(cli.COMMANDS["sketch"], "_draw", False)        # routed through cli_draw
    code, out = run("sketch", "apple")
    assert code == 2 and "sketch figure" in out                    # the no-reference sketch, not the tracing one


def test_undo_to_a_step_and_says_where_it_lands(proj, tmp_path):
    _plan(tmp_path)
    st = Project(proj).state["step"]
    run("region", "add", "extra", "--box", "0,0,200,200", "--no-refine")
    run("region", "add", "extra2", "--box", "0,0,300,300", "--no-refine")
    code, out = run("undo", "--to", str(st))
    assert code == 0 and f"back to step {st}" in out
    assert Project(proj).state["step"] == st


def test_todo_rebuild_keeps_done_items(proj, tmp_path):
    _plan(tmp_path)
    before = {t["id"]: t["status"] for t in Project(proj).state["todo"]}
    code, out = run("todo", "--rebuild")
    assert code == 0 and "todo rebuilt" in out
    after = {t["id"]: t["status"] for t in Project(proj).state["todo"]}
    assert sum(v == "done" for v in after.values()) == sum(v == "done" for v in before.values())


def test_batch_shows_the_last_commands_own_report(proj, tmp_path):
    _plan(tmp_path)
    code, out = run("do", "todo --all")
    assert re.search(r"^\s+\d\. \[", out, re.M), out          # the plan lines are not hidden by the batch


def test_new_with_mode_hand_equals_new_then_mode_hand(tmp_path, monkeypatch):
    from PIL import Image
    from pixelcanvas.easel.project import Project as P
    monkeypatch.chdir(tmp_path)
    a = np.full((120, 100, 3), 240, np.uint8)
    a[30:90, 20:80] = (200, 60, 60)
    Image.fromarray(a).save(tmp_path / "r.png")
    assert run("new", "a", "--ref", "r.png", "--mode", "hand")[0] == 0
    sa = P(tmp_path / "a.easel").state
    assert sa.get("mode") == "hand"
    code, out = run("new", "c", "--mode", "hand")
    assert code != 0 and "--ref" in out
    code, out = run("new", "d", "--ref", "r.png", "--mode", "sideways")
    assert code != 0 and "hand" in out


def test_free_mode_look_takes_a_part_and_hand_tools_take_palette_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run("new", "pic", "--mode", "free", "--size", "300x300", "--request", "a red ball")[0] == 0
    run("palette", "make", "--base", "#c04030")
    run("palette", "pick", "1")
    run("plan", "subject", "ball", "--box", "100,100,900,900")
    run("plan", "part", "top", "--of", "ball", "--box", "200,150,800,500")
    code, out = run("look", "top")
    assert code == 0 and "part top" in out, out
    code, out = run("shape", "polygon", "100,100 300,100 300,300", "--color", "background-shadow")
    assert code == 0, out


def test_shape_triangle_version_and_gate_sentences(tmp_path, monkeypatch, capsys):
    from pixelcanvas.easel import cli
    monkeypatch.chdir(tmp_path)
    assert run("new", "pic", "--mode", "free", "--size", "200x200", "--request", "a tree")[0] == 0
    code, out = run("shape", "triangle", "300,200,700,800", "--color", "#204020")
    assert code == 0 and "shape triangle" in out, out
    assert cli.main(["--version"]) == 0 and "easel" in capsys.readouterr().out
    assert cli._unmet_words({"text": "the face has no mouth", "metric": "m", "value": 0, "need": 1,
                             "region": "head"}) == "the face has no mouth"
    assert cli._unmet_words({"metric": "lines", "value": 30, "need": 40, "region": "hair"}) == "hair lines 30 < 40"
