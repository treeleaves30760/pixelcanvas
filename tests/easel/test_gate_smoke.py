"""Smoke test of the reference flow on the current default (real engine): one tiny reference through every stage
command the todo names, the detail levels, the dots, the finishing sharpen and the export."""
import numpy as np
import pytest

from pixelcanvas.easel import cli, workflow
from pixelcanvas.easel.project import Project

from .shell_helpers import run, synthetic_ref


def test_tiny_reference_through_every_stage_on_the_default(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    Image.fromarray(synthetic_ref(120, 96)).save(tmp_path / "ref.png")
    code, out = run("new", "pic", "--ref", "ref.png")
    assert code == 0, out
    run("style", "ink")              # the style whose detail stage paints under the ink
    run("region", "add", "top", "--box", "0,0,1000,400", "--no-refine", "--focus", "3")
    run("region", "add", "mid", "--box", "0,400,1000,700", "--no-refine")
    run("region", "rest", "low")
    assert run("stage", "done")[0] == 0
    proj = lambda: Project(tmp_path / "pic.easel")  # noqa: E731
    for _ in range(12):
        st = proj().state
        stage = st["stage"]
        if stage == "finish":
            break
        if stage == "detail":
            code, out = run("detail", "all")
            for _k in range(10):
                if "continues" not in out:
                    break
                code, out = run("detail", "all")
            assert code == 0, out
            for r in ("top", "mid", "low"):
                assert run("detail", r, "--level", "1")[0] == 0
            if cli._streaks_available():      # the finest level lies on the detail layer (over the ink too)
                import json
                last = [json.loads(x) for x in (tmp_path / "pic.easel" / "journal.jsonl").read_text(
                    encoding="utf-8").splitlines() if x.strip()][-1]
                assert last.get("layer") == "detail", last
            if workflow.dots_planned(proj().state):
                code, out = run("dot", "all")
                assert code == 0 and "texture points" in out, out
            code, out = run("stage", "done")
            assert code == 0, out
            continue
        for t in [t for t in st["todo"] if t["stage"] == stage and t["kind"] == "task" and t["status"] == "open"]:
            if t["cmd"].startswith("easel look"):
                continue
            code, out = run(*t["cmd"].split()[1:])
            assert code in (0, 1), (t["cmd"], out)
        run("stage", "done", "--because", "smoke")
    assert proj().state["stage"] == "finish"
    if proj().state.get("finish_sharpen"):
        code, out = run("sharpen")
        assert code == 0, out
    code, out = run("export", "final.png")
    assert code == 0 and "levels:" in out, out
    img = np.asarray(Image.open(tmp_path / "final.png").convert("RGB"))
    assert img.shape == (96, 120, 3)
    if cli._streaks_available():
        st = proj().state
        assert st["work_scale"] == cli._auto_scale(120, 96) and all(cli.level_of(st, r) == 1 for r in st["regions"])
