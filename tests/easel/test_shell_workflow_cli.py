"""workflow.py (styles, todo, gates, next) and cli.py (reports, errors, batching, lenient parsing)."""
import json
import re
import shlex

import numpy as np
import pytest
from PIL import Image

from pixelcanvas.easel import workflow
from pixelcanvas.easel.project import Project

from .shell_helpers import install_fake_engine, next_cmd, run, view_path, write_ref


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_engine(monkeypatch)
    write_ref(tmp_path / "ref.png")
    code, out = run("new", "pic", "--ref", "ref.png")
    assert code == 0, out
    return tmp_path / "pic.easel"


def _check_report(code, out, max_lines=25):
    lines = out.splitlines()
    assert len(lines) <= max_lines, out
    assert lines[-1].startswith("NEXT: "), out
    assert "Traceback" not in out
    v = view_path(out)
    if v:
        img = Image.open(v)
        w, h = img.size
        assert max(w, h) <= 1280 and w % 32 == 0 and h % 32 == 0


def test_new_reports_style_plan_and_marks(proj):
    code, out = run("status")
    _check_report(code, out)
    st = Project(proj).state
    assert st["style"] in workflow.STYLES and st["marks"]["n"] == 40
    assert any(t["key"] == "marks" and t["status"] == "done" for t in st["todo"])


def test_style_guess_from_request_and_picture():
    assert workflow.guess_style(None, "an anime girl")[0] == "cel"
    assert workflow.guess_style(None, "oil painting of a lake")[0] == "paint"
    rng = np.random.default_rng(0)
    noisy = np.clip(rng.normal(128, 40, (200, 200, 3)), 0, 255).astype(np.uint8)
    assert workflow.guess_style(noisy)[0] in ("paint", "photo")


def test_todo_one_line_per_stage_region_plus_gates(proj):
    run("region", "add", "a", "--box", "0,0,1000,400", "--no-refine")
    run("region", "add", "b", "--box", "0,400,1000,1000", "--no-refine", "--focus", "3")
    st = Project(proj).state
    plan = workflow.PLANS[st["style"]]
    middle = [s for s in plan if s not in ("plan", "finish")]
    for s in middle:
        tasks = [t for t in st["todo"] if t["stage"] == s and t["kind"] == "task"
                 and t.get("key") not in ("closeup", "fine")]
        # sketch per region; every later stage runs its tool on the whole picture first
        assert sorted(t["region"] for t in tasks) == (["a", "b"] if s == "sketch" else ["all"])
    # detail: the stage tool on all, a close look at the focus-3 region
    det = [(t.get("key"), t["region"]) for t in st["todo"] if t["stage"] == "detail" and t["kind"] == "task"]
    assert det == [(None, "all"), ("closeup", "b")]              # the stand-in engine has no detail levels
    close = [t for t in st["todo"] if t.get("key") == "closeup"]
    assert [(t["stage"], t["region"]) for t in close] == [("detail", "b"), ("finish", "b")]
    assert sum(t["kind"] == "gate" for t in st["todo"]) == len(plan)


def test_plan_gate_and_override(proj):
    code, out = run("stage", "done")
    assert code == 1 and "cannot close" in out and "region" in out
    code, out = run("stage", "done", "--because", "testing")
    assert code == 0 and "now stage sketch" in out


def test_gate_explains_number_and_fix(proj):
    run("style", "cel")
    run("region", "add", "face", "--box", "250,150,750,600", "--focus", "3")
    run("region", "add", "ground", "--box", "0,650,1000,1000", "--no-refine")
    run("region", "rest", "background")
    code, out = run("stage", "done")
    assert code == 0, out
    run("stage", "done", "--because", "skip sketch")
    code, out = run("stage")
    assert "lines" in out and "raise it:  easel ink" in out
    assert next_cmd(out).startswith("easel ink")


def _fill_placeholders(cmd):
    if "region add" in cmd and "<" in cmd:
        return None
    return cmd


def test_next_walks_full_plan_to_the_end(proj):
    """Follow only NEXT lines (supplying regions in the plan stage) until the picture is finished."""
    plan_cmds = iter(["region add top --box 0,0,1000,350 --no-refine --focus 3",
                      "region add face --box 250,150,750,650",
                      "region add ground --box 0,650,1000,1000 --no-refine"])
    seen_stages = []
    for _ in range(120):
        code, out = run("next")
        _check_report(code, out)
        cmd = next_cmd(out)
        st = Project(proj).state
        if st.get("finished"):
            break
        if st["stage"] not in seen_stages:
            seen_stages.append(st["stage"])
        if "<" in cmd:
            cmd = "easel " + next(plan_cmds)
        if cmd.startswith("easel look") and st["stage"] != "finish":
            cmd = 'easel stage done --because "test walk"'
        args = shlex.split(cmd)[1:]
        code, out = run(*args)
        _check_report(code, out)
        assert code in (0, 1), out
    else:
        pytest.fail("next did not reach the end")
    assert seen_stages == workflow.PLANS[Project(proj).state["style"]]
    assert Project(proj).state["exported"]


def test_usage_errors_show_example(proj):
    cases = [
        (["look", "--shwo", "diff"], "did you mean --show"),
        (["ink", "hiar"], "unknown region"),
        (["stroke", "100,200 300"], "odd"),
        (["fill", "500,400", "--color", "notacolour"], "unknown colour"),
        (["flatt", "all"], "did you mean flat"),
        (["region", "add", "x"], "say how"),
        (["undo", "lots"], "needs one number"),
        (["style", "anime2"], "unknown style"),
        (["todo", "done", "999"], "no todo item"),
    ]
    for argv, needle in cases:
        code, out = run(*argv)
        assert code == 2, (argv, out)
        assert out.startswith("ERROR: ") and needle in out, (argv, out)
        assert "EXAMPLE: easel" in out and out.splitlines()[-1].startswith("NEXT: ")


def test_unknown_region_suggests_close_name(proj):
    run("region", "add", "hair", "--box", "200,0,800,300", "--no-refine")
    code, out = run("ink", "hiar")
    assert code == 2 and "did you mean hair" in out and "EXAMPLE: easel ink hair" in out


def test_lenient_parsing_in_cli(proj):
    for where in ("200,100,600,500", "[200, 100, 600, 500]", "(200 100 600 500)", "20%,10%,60%,50%"):
        code, out = run("look", where)
        assert code == 0 and "area" in out, out
    code, out = run("look", "200,", "100,", "600,", "500")      # split by the shell
    assert code == 0, out
    code, out = run("look", "0,0,1500,500")
    assert code == 0 and "clamped" in out
    code, out = run("pick", "(500, 400)")
    assert code == 0 and "#" in out
    code, out = run("region", "add", "Face", "--box", "250,150,750,600")
    code, out = run("look", "FACE", "--show", "diff")
    assert code == 0 and "DIFF" not in out and "look face" in out
    code, out = run("stroke", "[[100,100],[200,150],[300,120]]", "--color", "ff0000", "--size", "3")
    assert code == 0, out


def test_look_modes_and_speed(proj):
    import time
    for show in ("both", "canvas", "ref", "diff", "marks", "layers"):
        t = time.time()
        code, out = run("look", "--show", show)
        assert time.time() - t < 2.0, show
        _check_report(code, out)
        assert code == 0 and view_path(out)
    code, out = run("look", "--show", "both", "--grid")
    assert code == 0


def test_stage_report_has_numbers_spots_and_view(proj):
    run("region", "add", "face", "--box", "250,150,750,600", "--focus", "3")
    code, out = run("ink", "face")
    _check_report(code, out)
    assert re.search(r"step \d+  ink face", out)
    assert re.search(r"lines \d+ -> \d+", out) and "VIEW:" in out
    assert "the canvas after this step" in out and "VIEW-REF:" in out


def test_do_batch_stops_at_first_failure(proj):
    code, out = run("do", "ink all; ink nowhere; flat all")
    assert code == 2
    assert "[1] step" in out and "FAILED" in out and "not executed: flat all" in out
    steps = [json.loads(l) for l in (proj / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert not any(s.get("cmd", "").startswith("easel flat") for s in steps)
    code, out = run("do", "ink all; flat all")
    _check_report(code, out)
    assert code == 0 and "[2] step" in out and view_path(out)


def test_missing_stage_module_is_clean(proj, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pixelcanvas.easel.stage_shade", None)
    code, out = run("shade", "all")
    assert code == 1 and "ERROR:" in out and "Traceback" not in out and out.splitlines()[-1].startswith("NEXT:")


def test_internal_error_is_one_line(proj, monkeypatch):
    from pixelcanvas.easel import cli

    def boom(ctx, args):
        raise ValueError("kaboom")
    monkeypatch.setitem(cli.COMMANDS, "score", boom)
    code, out = run("score")
    assert code == 1 and "kaboom" in out and "Traceback" not in out and len(out.splitlines()) == 2
    assert "Traceback" in (proj / "errors.log").read_text(encoding="utf-8")


def test_hand_tools_and_layers(proj):
    for argv in (["fill", "500,800", "--color", "pick", "--bounds", "ref"],
                 ["shape", "ellipse", "400,300,600,450", "--color", "#ffcc00"],
                 ["shape", "line", "100,100 900,900", "--outline", "4"],
                 ["gradient", "all", "--from", "500,0", "--to", "500,1000", "--colors", "#a0c0ff,#203060"],
                 ["erase", "400,300,600,450"],
                 ["smooth", "all", "--layer", "detail"],
                 ["adjust", "all", "--layer", "detail", "--brightness", "0.1"],
                 ["layer", "shade", "--mode", "multiply", "--opacity", "0.5"],
                 ["palette", "--k", "5"], ["score"], ["todo"], ["todo", "add", "check the eyes"],
                 ["layers"]):
        code, out = run(*argv)
        _check_report(code, out)
        assert code == 0, (argv, out)


def test_export_png(proj, tmp_path):
    code, out = run("export", "out dir/final.png")
    assert code == 0, out
    img = Image.open(tmp_path / "out dir" / "final.png")
    assert img.size == (300, 360)
    code, out = run("export", "x.png", "--psd")          # layered export is part of export.py now
    assert code == 0 and (tmp_path / "x.psd").exists(), out


def test_no_project_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out = run("status")
    assert code == 2 and "no easel project" in out and "easel new" in out
