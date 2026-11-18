"""Shell parity with the scripted pipeline: stage tools on `all` use the pipeline's passes in one undoable step,
the todo list runs each stage on the whole picture first, the finish gate asks for a close look at every focus-3
region, the plan stage's picture shows the regions."""
import re

import numpy as np
import pytest

from pixelcanvas import cli, view, workflow
from pixelcanvas.project import Project

from .shell_helpers import install_fake_engine, run, write_ref


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_engine(monkeypatch)
    write_ref(tmp_path / "ref.png")
    assert run("new", "pic", "--ref", "ref.png")[0] == 0
    run("style", "cel")
    run("region", "add", "top", "--box", "0,0,1000,400", "--no-refine")
    run("region", "add", "face", "--box", "300,400,700,800", "--no-refine", "--focus", "3")
    run("region", "rest", "ground")
    assert run("stage", "done")[0] == 0
    return tmp_path / "pic.pixelcanvas"


def test_default_passes_follow_the_pipeline():
    st = {"style": "cel"}
    passes = cli._default_passes(st, "detail", 850, 943, None)
    # the pipeline's cel detail: the residual shape pass (brush "shapes"), then fine liner strokes
    assert [s for s, _, _ in passes] == [6.0, 2.5, 1.5]
    assert [b for _, b, _ in passes] == ["shapes", "liner", "liner"]
    st = {"style": "paint"}
    sizes = [s for s, _, _ in cli._default_passes(st, "block-in", 1280, 853, None)]
    assert sizes == [64.0, 32.0]


def test_detail_all_is_one_step_with_several_passes(proj):
    run("stage", "done", "--because", "t")
    for _ in range(3):
        run("stage", "done", "--because", "t")
    st = Project(proj).state
    assert st["stage"] == "detail"
    step0 = st["step"]
    code, out = run("detail", "all")                          # the pipeline's passes (level 2) in one step
    assert code == 0 and "--size 6/2.5/1.5" in out and "--brush shapes/liner" in out, out
    assert Project(proj).state["step"] == step0 + 1          # one undo removes all passes
    code, out = run("undo")
    assert Project(proj).state["step"] == step0


def test_todo_runs_each_stage_on_all_then_close_looks(proj):
    st = Project(proj).state
    by = lambda s: [t for t in st["todo"] if t["stage"] == s and t["kind"] == "task"]  # noqa: E731
    assert [t["cmd"] for t in by("line")] == ["pixelcanvas ink all"]
    assert [t["cmd"] for t in by("detail")] == ["pixelcanvas detail all", "pixelcanvas look face"]   # no levels in this engine
    assert [t["cmd"] for t in by("finish")] == ["pixelcanvas look", "pixelcanvas look face", "pixelcanvas export final.png"]


def test_finish_gate_needs_the_close_look(proj):
    for _ in range(5):
        run("stage", "done", "--because", "t")
    assert Project(proj).state["stage"] == "finish"
    run("export", "final.png")
    code, out = run("stage", "done")
    assert code == 1 and "close look" in out and "pixelcanvas look face" in out
    run("look", "face")
    code, out = run("stage", "done")
    assert code == 0, out


def test_plan_picture_shows_the_regions():
    base = np.full((30, 40, 3), 255, np.uint8)
    m = np.zeros((30, 40), bool)
    m[5:20, 5:20] = True
    out = view.region_overlay(base, [("a", m, 1)])
    assert out.shape == base.shape and (out[10, 10] != 255).any() and (out[25, 35] == 255).all()


R10 = {"work_scale": "auto", "fine": "streaks", "max_level": 1, "stipple": 2, "stipple_edges": False,
       "deliver_filter": "sharpen", "contours": True, "smooth_tol": 0.8}


# cel and paint are the two stage plans (line, flat, shade / block-in, forms); ink, poster and photo share them
@pytest.mark.parametrize("style,size", [("cel", (200, 160)), ("paint", (200, 160))])
def test_cli_defaults_per_region_match_the_pipeline(tmp_path, monkeypatch, style, size):
    """A scripted CLI session with default options (regions covering the picture, each stage once per region,
    the command the todo list names, detail level 2 on all then level 1 on every region, the dots, the finishing
    sharpen) reaches the scripted pipeline's result on the same default (round 10: working scale by size, fine
    levels as streaks, finest level 1, contours as single strokes, smooth areas to 0.8 Delta E, two rounds of dots
    off the edges, sharpened at delivery; real engine)."""
    from PIL import Image
    from pixelcanvas import cli, pipeline
    from pixelcanvas.project import Project as P
    from .shell_helpers import synthetic_ref
    monkeypatch.chdir(tmp_path)
    ref = synthetic_ref(*size)
    if style == "ink":
        ref = np.repeat(ref.mean(axis=2, keepdims=True), 3, axis=2).astype(np.uint8)
    Image.fromarray(ref).save(tmp_path / "ref.png")
    boxes = {"top": "0,0,1000,400", "mid": "0,400,1000,700", "low": "0,700,1000,1000"}
    assert run("new", "pic", "--ref", "ref.png")[0] == 0
    run("style", style)
    for name, box in boxes.items():
        run("region", "add", name, "--box", box, "--no-refine", *(["--focus", "3"] if name == "top" else []))
    assert run("stage", "done")[0] == 0
    proj = P(tmp_path / "pic.pixelcanvas")
    for stage in workflow.plan_of(proj.state)[1:-1]:
        proj = P(tmp_path / "pic.pixelcanvas")
        assert proj.state["stage"] == stage
        if stage == "detail":                 # level 2 on all, then level 1 on every region
            code, out = run("detail", "all")
            assert code == 0 and ("contours: " in out or not cli.contours_available()), out
            for name in boxes:
                code, out = run("detail", name, "--level", "1")
                assert code == 0, out
            if cli.dots_available():
                code, out = run("dot", "all")
                assert code == 0 and "texture points" in out, out
            assert run("stage", "done")[0] == 0
            continue
        for name in boxes:
            cmd = workflow.task_command(proj.state, stage, name)
            if stage in ("block-in", "forms") and cmd.startswith("pixelcanvas paint"):
                cmd = f"pixelcanvas paint {name}"             # the stage's own passes, as on all
            code, out = run(*cmd.split()[1:])
            assert code == 0, (cmd, out)
        run("stage", "done", "--because", "parity test")
    if not cli._streaks_available() or not cli.dots_available():
        pytest.skip("engine without streaks or dots")
    code, out = run("sharpen")
    assert code == 0 and "finishing sharpen" in out, out
    code, out = run("export", "cli.png")
    assert code == 0 and "then the finishing sharpen" in out and "delivered area + sharpen" in out, out
    proj = P(tmp_path / "pic.pixelcanvas")
    assert proj.state["work_scale"] == pipeline.auto_work_scale(*size) and proj.state["fine"] == "streaks"
    assert 0 < proj.state["dotted"] < 0.2
    ref_d = proj.ref_deliver
    masks = [proj.mask_to_deliver(proj.region_mask(n)) for n in boxes]
    pipeline.run_reference_pipeline(ref_d, str(tmp_path / "pipe"), style=style, log=lambda *a: None,
                                    regions=masks, **R10)
    cli_img = np.asarray(Image.open(tmp_path / "cli.png").convert("RGB"))
    pipe_img = np.asarray(Image.open(tmp_path / "pipe" / "final.png").convert("RGB"))
    assert cli_img.shape == pipe_img.shape == ref_d.shape
    s_cli, s_pipe = pipeline.quick_scores(ref_d, cli_img), pipeline.quick_scores(ref_d, pipe_img)
    print("PARITY", style, s_cli, s_pipe)
    assert abs(s_cli["mean_dE"] - s_pipe["mean_dE"]) <= 0.25 + 0.15 * s_pipe["mean_dE"], (s_cli, s_pipe)
    assert s_cli["edge_f1"] >= s_pipe["edge_f1"] - 0.02, (s_cli, s_pipe)


def test_detail_levels_refine_by_region_and_retouch_is_recorded(tmp_path, monkeypatch):
    """Real engine: detail all = level 2 and ranks the regions; refine items for every region, focus first; the gate
    wants every region at level 1; retouch (not in the todo) is counted in the picture's record."""
    from PIL import Image
    from pixelcanvas import workflow
    from pixelcanvas.project import Project as P
    from .shell_helpers import synthetic_ref
    if not workflow.levels_available():
        pytest.skip("engine without detail levels")
    monkeypatch.chdir(tmp_path)
    Image.fromarray(synthetic_ref(150, 180)).save(tmp_path / "ref.png")
    assert run("new", "pic", "--ref", "ref.png", "--scale", "2")[0] == 0
    run("style", "cel")
    run("region", "add", "top", "--box", "0,0,1000,400", "--no-refine", "--focus", "3")
    run("region", "add", "mid", "--box", "0,400,1000,700", "--no-refine", "--focus", "2")
    run("region", "add", "low", "--box", "0,700,1000,1000", "--no-refine")
    run("stage", "done")
    while P(tmp_path / "pic.pixelcanvas").state["stage"] != "detail":
        run("stage", "done", "--because", "test")
    code, out = run("detail", "all")
    assert code == 0 and "level 2 (fine strokes)" in out and "refine next, in this order" in out, out
    st = P(tmp_path / "pic.pixelcanvas").state
    items = [(t.get("key"), t["region"]) for t in st["todo"] if t["stage"] == "detail" and t["kind"] == "task"]
    assert items[1:2] == [("refine", "top")] and ("refine", "low") in items and ("retouch", "top") not in items
    code, out = run("stage", "done")
    assert code == 1 and "top (focus 3) is not refined to level 1 yet" in out, out
    code, out = run("detail", "top", "--level", "1")
    assert code == 0 and "top at level 1 (hairlines)" in out, out
    code, out = run("retouch", "mid")
    assert code == 0 and "retouched now:" in out, out
    st = P(tmp_path / "pic.pixelcanvas").state
    assert st["levels"]["top"] == 1 and st["levels"]["mid"] == 0 and 0 < st["retouched"] < 0.5
    code, out = run("status")
    assert "levels: top 1, mid 0, low 2; retouched" in out, out
    assert run("retouch", "all")[0] == 2                         # the whole picture needs a reason
    code, out = run("stage", "done")
    assert code == 1 and "low (focus 1) is not refined to level 1 yet" in out, out    # every region, focus first
    assert run("detail", "low", "--level", "1")[0] == 0
    code, out = run("stage", "done")
    assert code == 1 and "the texture points are not dotted yet" in out, out    # then the dots
    assert run("dot", "all")[0] == 0
    code, out = run("stage", "done")
    assert code == 0, out
    for _ in range(4):
        run("undo")                                              # back to before the retouch: its share goes
    assert P(tmp_path / "pic.pixelcanvas").state.get("retouched", 0) == 0


def test_look_diff_on_a_working_canvas(tmp_path, monkeypatch):
    """The error map is measured at the delivered size; the diff view shows it over the 2x canvas."""
    from PIL import Image
    from pixelcanvas import cli
    from .shell_helpers import synthetic_ref
    if not cli._streaks_available():
        pytest.skip("engine without streaks")
    monkeypatch.chdir(tmp_path)
    Image.fromarray(synthetic_ref(200, 240)).save(tmp_path / "ref.png")
    assert run("new", "pic", "--ref", "ref.png")[0] == 0
    run("region", "add", "top", "--box", "0,0,1000,500", "--no-refine")
    code, out = run("look", "top", "--show", "diff")
    assert code == 0 and "VIEW:" in out, out


def test_dots_continue_and_the_sharpen_is_a_finish_step(tmp_path, monkeypatch):
    """Real engine, round 9 default: a small picture is drawn at 4x; the dots are a detail todo item and gate, run
    part by part when the time budget runs out (NEXT repeats the command), and report count, share, error and a
    view; the sharpen is a finish item that export delivers with, undone like any step."""
    from PIL import Image
    from pixelcanvas import cli as c
    from pixelcanvas.project import Project as P
    from .shell_helpers import synthetic_ref
    if not (c._streaks_available() and c.dots_available() and workflow.levels_available()):
        pytest.skip("engine without streaks, levels or dots")
    monkeypatch.chdir(tmp_path)
    Image.fromarray(synthetic_ref(100, 120)).save(tmp_path / "ref.png")
    code, out = run("new", "pic", "--ref", "ref.png")
    assert code == 0 and "working canvas 4x" in out, out
    run("style", "cel")
    run("region", "add", "top", "--box", "0,0,1000,500", "--no-refine", "--focus", "3")
    run("region", "rest", "low")
    run("stage", "done")
    while P(tmp_path / "pic.pixelcanvas").state["stage"] != "detail":
        run("stage", "done", "--because", "test")
    st = P(tmp_path / "pic.pixelcanvas").state
    assert ("dot", None) in [(t.get("key"), t["region"]) for t in st["todo"] if t["stage"] == "detail"]
    assert "sharpen" in [t.get("key") for t in st["todo"] if t["stage"] == "finish"]
    run("detail", "all")
    for r in ("top", "low"):
        assert run("detail", r, "--level", "1")[0] == 0
    code, out = run("stage", "done")
    assert code == 1 and "the texture points are not dotted yet" in out and "pixelcanvas dot all" in out, out
    monkeypatch.setattr(c, "BAND_PX", 75_000)
    monkeypatch.setattr(c, "LEVEL_BUDGET", 0.0)
    code, out = run("dot", "all")
    assert code == 0 and "the same command continues" in out and "NEXT: pixelcanvas dot all" in out, out
    for _ in range(40):
        code, out = run("dot", "all")
        if "continues" not in out:
            break
    assert code == 0 and "texture points in all" in out and "Delta E" in out and "VIEW-REF:" in out, out
    assert "where the dots changed the canvas most" in out, out
    st = P(tmp_path / "pic.pixelcanvas").state
    assert 0 < st["dotted"] < 0.2 and not st.get("dot_pending")
    assert run("stage", "done")[0] == 0
    code, out = run("status")
    assert "dotted" in out and "drawn at 4x" in out and "delivered area" in out, out
    code, out = run("sharpen")
    assert code == 0 and "of the delivered pixels" in out and "VIEW-REF:" in out, out
    assert P(tmp_path / "pic.pixelcanvas").state["deliver_filter"] == "sharpen"
    run("undo")
    assert P(tmp_path / "pic.pixelcanvas").state["deliver_filter"] == "area"
    run("redo")
    code, out = run("export", "final.png")
    assert code == 0 and "then the finishing sharpen" in out and "delivered area + sharpen" in out, out
