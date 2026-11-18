"""export.py: the time-lapse (replay of the journal) and the stage pictures."""
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixelcanvas import export
from pixelcanvas.project import Project

from .finish_helpers import cwd_module, new_project, ok, run  # noqa: F401

pytestmark = pytest.mark.usefixtures("cwd_module")


@pytest.fixture(scope="module")
def film_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("film")
    proj = new_project(root, "film", "180x140")
    ok("stage", "done", "--because", "test")                                        # plan -> sketch
    ok("stroke", "100,300 400,200 700,300", "--size", "3", "--brush", "pencil", "--color", "#808080")
    ok("stage", "done", "--because", "test")                                        # sketch -> block-in
    ok("fill", "500,500", "--color", "#e8dcc0", "--layer", "paint", "--bounds", "canvas")
    ok("stroke", "100,800 500,700", "--size", "6", "--color", "#ff00ff", "--layer", "paint")  # undone below
    ok("undo")
    ok("stroke", "100,850 900,820", "--size", "6", "--color", "#00aa55", "--layer", "paint")
    for k in range(6):
        ok("stroke", f"{100 + 120 * k},100 {160 + 120 * k},300 {220 + 120 * k},150", "--size", "9",
           "--color", "#c04030", "--layer", "paint")
    ok("adjust", "all", "--brightness", "0.1", "--layer", "paint")                     # a pixel filter, no ops
    ok("layer", "paint", "--opacity", "0.9")                                           # a state-only change
    ok("fx", "grain", "--amount", "0.4")
    ok("text", "TITLE", "--box", "100,40,900,200", "--color", "#ffffff")
    ok("stage", "done", "--because", "test")                                        # block-in -> forms
    return root, proj


def test_final_frame_is_the_exported_picture(film_project):
    root, proj = film_project
    p = Project(proj)
    ok("export", "final.png")
    png = np.asarray(Image.open(root / "final.png").convert("RGB"))
    frames, n_events, secs = export.film_frames(p)
    frames = list(frames)
    assert frames[0].kind == "start" and frames[-1].kind == "final"
    assert np.array_equal(frames[-1].rgb, png)
    # the running canvas of the replay agrees with the finished picture (8-bit rounding only)
    prev = frames[-2].rgb.astype(int)
    assert np.abs(prev - png.astype(int)).max() <= 2


def test_undone_steps_are_not_in_the_film(film_project):
    _, proj = film_project
    p = Project(proj)
    cmds = {f.cmd for f in export.film_frames(p)[0]}
    assert any("00aa55" in c for c in cmds)
    assert not any("ff00ff" in c for c in cmds)
    # filters, layer changes, fx and text all show up
    assert any(c.startswith("adjust") for c in cmds) and any(c.startswith("fx grain") for c in cmds)
    assert any(c.startswith("text") for c in cmds) and any(c.startswith("layer paint") for c in cmds)


def test_every_step_gets_two_frames_and_the_film_has_a_sensible_length(film_project):
    _, proj = film_project
    p = Project(proj)
    frames, n_events, secs = export.film_frames(p)
    per = {}
    for f in frames:
        if f.kind in ("progress", "hold"):
            per[f.index] = per.get(f.index, 0) + 1
    assert sorted(per) == list(range(1, n_events + 1))
    assert min(per.values()) >= 2
    assert 8.0 <= secs <= 20.0
    assert max(per.values()) >= min(per.values())


def test_frames_are_allocated_by_op_count(film_project):
    _, proj = film_project
    rp = export.Replay(Project(proj))
    ev = rp.events()
    allocs = rp.allocate(ev, 200)
    by_ops = {i: rp.steps[i].n_ops for i, kind, _ in ev if kind == "ops"}
    big = max(by_ops, key=by_ops.get)
    small = min(by_ops, key=by_ops.get)
    assert allocs[big] >= allocs[small] >= 2
    assert sum(allocs.values()) >= 2 * len(ev)


def test_progress_is_progressive(film_project):
    """Inside one painting step the picture changes frame by frame (the strokes appear one after another)."""
    _, proj = film_project
    p = Project(proj)
    frames = list(export.film_frames(p)[0])
    first_stroke = [f for f in frames if "#c04030" in f.cmd and f.kind == "progress"]
    assert first_stroke
    by_index = {}
    for f in first_stroke:
        by_index.setdefault(f.index, []).append(f)
    for fs in by_index.values():
        if len(fs) >= 2:
            assert not np.array_equal(fs[0].rgb, fs[-1].rgb) or len(fs) == 2


def test_gif_writer(film_project, tmp_path):
    _, proj = film_project
    p = Project(proj)
    out = tmp_path / "film.gif"
    info = export.write_gif(p, out)
    im = Image.open(out)
    assert im.n_frames >= 10 and max(im.size) <= 640
    assert info["seconds"] >= 8.0
    im.seek(im.n_frames - 1)
    last = np.asarray(im.convert("RGB"), dtype=np.int16)
    final = p.composite_u8("export").astype(np.int16)
    # the canvas part of the last frame is the exported picture (palette quantisation only)
    canvas = last[: last.shape[0] - int(round(640 * 0.046)) - 1, :, :]
    small = np.asarray(Image.fromarray(final.astype(np.uint8)).resize((last.shape[1], canvas.shape[0])),
                       dtype=np.int16)
    assert np.abs(canvas - small).mean() < 12


def test_timelapse_via_the_command(film_project):
    root, proj = film_project
    out = ok("export", "final.png", "--timelapse", "film.gif")
    assert "time-lapse: film.gif" in out and "undone steps left out" in out
    assert (root / "film.gif").stat().st_size > 1000


def test_mp4_needs_ffmpeg_and_says_so(film_project, monkeypatch):
    monkeypatch.setattr(export, "find_ffmpeg", lambda: None)
    code, out = run("export", "final.png", "--timelapse", "film.mp4")
    assert code == 2 and "ffmpeg" in out and "film.gif" in out and "Traceback" not in out


@pytest.mark.skipif(export.find_ffmpeg() is None, reason="ffmpeg not installed")
def test_mp4_writer(film_project, tmp_path):
    _, proj = film_project
    out = tmp_path / "film.mp4"
    info = export.write_mp4(Project(proj), out)
    assert out.exists() and out.stat().st_size > 1000
    ff = export.find_ffmpeg()
    r = subprocess.run([ff, "-nostdin", "-v", "error", "-i", str(out), "-f", "null", "-"],
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr
    w, h = info["size"]
    assert w % 2 == 0 and h % 2 == 0 and max(w, h) <= 1080


def test_stage_pictures_and_contact_sheet(film_project, tmp_path):
    root, proj = film_project
    out = ok("export", "final.png", "--stages", "stages")
    assert "stages: 4 pictures" in out        # three closed stages and the open one
    files = sorted((root / "stages").glob("0*.png"))
    assert [f.name.split("-", 1)[1] for f in files] == ["plan.png", "sketch.png", "block-in.png",
                                                                 "forms-open.png"]
    assert (root / "stages" / "stages.jpg").stat().st_size > 1000
    imgs = [np.asarray(Image.open(f).convert("RGB")).astype(int) for f in files]
    p = Project(proj)
    assert all(i.shape == (p.H, p.W, 3) for i in imgs)
    assert np.ptp(imgs[0]) == 0                                       # plan: empty paper
    assert not np.array_equal(imgs[0], imgs[1])                       # the sketch stroke
    assert not np.array_equal(imgs[1], imgs[2])                       # the block-in
    assert np.array_equal(imgs[2], imgs[3])                           # nothing changed since block-in closed
    sheet = Image.open(root / "stages" / "stages.jpg")
    assert sheet.size[0] >= 300 * 3


def test_empty_project_exports_a_film(tmp_path):
    new_project(tmp_path, "empty", "120x90")
    out = ok("export", "e.png", "--timelapse", "e.gif", "--stages", "st")
    assert (tmp_path / "e.gif").exists() and (tmp_path / "st" / "stages.jpg").exists(), out
