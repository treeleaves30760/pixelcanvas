"""project.py (folder, round trip, undo / redo byte-exact) and regions.py (box, marks, polygon, wand)."""
import json

import cv2
import numpy as np
import pytest

from pixelcanvas import regions
from pixelcanvas.project import LAYER_NAMES, Project

from .shell_helpers import install_fake_engine, run, synthetic_ref, write_ref


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_engine(monkeypatch)
    write_ref(tmp_path / "ref image.png")
    code, out = run("new", "pic", "--ref", "ref image.png")
    assert code == 0, out
    return tmp_path / "pic.pixelcanvas"


def test_project_folder_layout(proj):
    for sub in ("state.json", "ref.png", "journal.jsonl", "layers", "regions", "history", "ops", "views"):
        assert (proj / sub).exists(), sub
    for n in LAYER_NAMES:
        assert (proj / "layers" / f"{n}.png").exists()
    st = json.loads((proj / "state.json").read_text(encoding="utf-8"))
    assert st["width"] == 300 and st["height"] == 360 and st["stage"] == "plan"
    assert [e["name"] for e in st["layers"]] == LAYER_NAMES
    assert (proj / "views" / "last.png").exists()


def test_reload_round_trip(proj):
    p = Project(proj)
    layer = p.layer("flat").copy()
    layer[10:20, 10:20] = (0.2, 0.4, 0.6, 1.0)
    step = p.begin_step(["layers/flat.png"])
    p._write_layer_file("flat", layer)
    p.end_step(step, {"cmd": "test"})
    q = Project(proj)
    got = q.layer("flat")
    assert np.allclose(got[15, 15], (0.2, 0.4, 0.6, 1.0), atol=1 / 255)
    assert q.state["step"] == step and not q.layer_entry("flat")["empty"]


def test_undo_redo_restore_bytes(proj):
    files = ["layers/line.png", "state.json"]
    before = {f: (proj / f).read_bytes() for f in files}
    code, out = run("ink", "all")
    assert code == 0, out
    after = {f: (proj / f).read_bytes() for f in files}
    assert after["layers/line.png"] != before["layers/line.png"]
    code, out = run("undo")
    assert code == 0 and "undo step" in out
    for f in files:
        assert (proj / f).read_bytes() == before[f], f
    code, out = run("redo")
    assert code == 0
    assert (proj / "layers/line.png").read_bytes() == after["layers/line.png"]
    # a new step clears the redo stack
    run("undo")
    run("flat", "all")
    code, out = run("redo")
    assert "nothing to redo" in out


def test_undo_region_removes_new_file(proj):
    run("region", "add", "top", "--box", "0,0,1000,300", "--no-refine")
    assert (proj / "regions" / "top.png").exists()
    run("undo")
    assert not (proj / "regions" / "top.png").exists()
    assert "top" not in Project(proj).state["regions"]


def test_journal_and_ops_files(proj):
    run("ink", "all")
    lines = (proj / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    assert last["cmd"].startswith("pixelcanvas ink") and last["layer"] == "line"
    assert "before" in last and "after" in last and last["ops"].startswith("ops/")
    assert (proj / last["ops"]).exists()


def test_deterministic_projects(tmp_path, monkeypatch):
    install_fake_engine(monkeypatch)
    outs = []
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        monkeypatch.chdir(tmp_path / d)
        write_ref(tmp_path / d / "ref.png")
        run("new", "p", "--ref", "ref.png")
        run("region", "add", "face", "--box", "250,150,750,600")
        run("ink", "face")
        outs.append(((tmp_path / d / "p.pixelcanvas/layers/line.png").read_bytes(),
                     (tmp_path / d / "p.pixelcanvas/regions/face.png").read_bytes(),
                     (tmp_path / d / "p.pixelcanvas/cache/marks_40.png").read_bytes()))
    assert outs[0] == outs[1]


# ----------------------------------------------------------------------------------------------- regions

def test_refine_box_finds_disk():
    img = np.full((200, 200, 3), 230, np.uint8)
    cv2.circle(img, (100, 100), 50, (200, 40, 40), -1)
    notes = []
    m = regions.refine_box(img, (30, 30, 170, 170), notes)
    disk = np.zeros((200, 200), np.uint8)
    cv2.circle(disk, (100, 100), 50, 1, -1)
    iou = (m & (disk > 0)).sum() / (m | (disk > 0)).sum()
    assert iou > 0.85, notes


def test_marks_count_order_and_determinism():
    rng = np.random.default_rng(1)
    img = np.zeros((240, 320, 3), np.uint8)
    for y in range(0, 240, 40):
        for x in range(0, 320, 40):
            img[y:y + 40, x:x + 40] = rng.integers(0, 255, 3)
    a, ta = regions.compute_marks(img, 30)
    b, tb = regions.compute_marks(img, 30)
    assert np.array_equal(a, b) and ta == tb
    k = int(a.max())
    assert 12 <= k <= 30
    assert set(np.unique(a)) == set(range(1, k + 1))
    # reading order: mark 1 starts in the top band
    assert ta[0]["box"][1] < 200
    assert all({"n", "box", "hex", "colour", "share"} <= set(r) for r in ta)


def test_polygon_snaps_to_edge():
    img = np.full((200, 200, 3), 240, np.uint8)
    img[50:150, 50:150] = (40, 40, 40)
    rough = np.array([[46, 47], [154, 52], [151, 153], [53, 148]], np.float32)   # a few px off the square
    snapped = regions.poly_region(img, 200, 200, rough, snap=True)
    plain = regions.poly_region(img, 200, 200, rough, snap=False)
    sq = np.zeros((200, 200), bool)
    sq[50:150, 50:150] = True
    iou = lambda m: (m & sq).sum() / (m | sq).sum()  # noqa: E731
    assert iou(snapped) > iou(plain) + 0.01 and iou(snapped) > 0.93


def test_wand_connected_and_tolerance():
    img = np.full((100, 100, 3), 255, np.uint8)
    img[10:40, 10:40] = (200, 0, 0)
    img[60:90, 60:90] = (200, 0, 0)
    m = regions.wand(img, (20, 20), tol=10)
    assert m[25, 25] and not m[70, 70] and not m[5, 5]


def test_region_cli_plus_minus_rest_focus(proj):
    code, out = run("region", "add", "Top", "--box", "0,0,1000,500", "--no-refine")
    assert code == 0 and "region top added" in out
    code, out = run("region", "add", "middle", "--box", "0,300,1000,700", "--no-refine", "--minus", "TOP")
    assert code == 0
    p = Project(proj)
    assert not (p.region_mask("middle") & p.region_mask("top")).any()
    code, out = run("region", "rest", "background")
    assert code == 0 and "covered 100%" in out
    code, out = run("region", "focus", "middle", "3")
    assert code == 0 and Project(proj).state["regions"]["middle"]["focus"] == 3
    code, out = run("region", "add", "eyes", "--marks", "1,2")
    assert code == 0, out
    code, out = run("region", "add", "bad", "--marks", "999")
    assert code == 2 and "do not exist" in out and "EXAMPLE:" in out
