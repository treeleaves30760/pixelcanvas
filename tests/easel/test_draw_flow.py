"""Drawing without a reference: the commands end to end, undo, the owned layers, the workflow."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pixelcanvas.easel import cli, workflow


@pytest.fixture
def draw(tmp_path, monkeypatch):
    """Install the draw hook for one test and restore easel's tables afterwards."""
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
    assert "NEXT:" in text
    return text


def setup_bust(run):
    ok(run("new", "pic", "--size", "400x400", "--request", "anime girl with silver hair and a red scarf"))
    ok(run("do", "plan subject girl --box 150,40,850,1000; plan part hair --box 200,40,800,800; "
                 "plan part face --box 330,150,670,520; plan part scarf --box 280,560,720,760; plan light top-left; "
                 "palette make; palette pick 1"))


def test_new_without_reference_points_to_plan(draw):
    text = ok(draw("new", "pic", "--size", "300x300", "--request", "a chibi fox"))
    assert "easel plan subject" in text


def test_plan_palette_and_gate(draw):
    setup_bust(draw)
    st = json.load(open("pic.easel/state.json", encoding="utf-8"))
    d = st["draw"]
    assert d["parts"]["hair"]["swatch"] == "hair" and d["parts"]["face"]["swatch"] == "skin"
    sw = d["palette"]["swatches"]
    assert {"hair", "skin", "scarf", "background", "line"} <= set(sw)
    # the request named the hair silver: a light, nearly grey colour
    from pixelcanvas.easel.draw_core import rgb_to_lch, from_hex
    L, C, h = rgb_to_lch(from_hex(sw["hair"]))
    assert L > 70 and C < 12
    text = ok(draw("stage", "done"))
    assert "now stage sketch" in text


def test_sketch_figure_head_and_edit_by_id(draw):
    setup_bust(draw)
    ok(draw("stage", "done"))
    text = ok(draw("sketch", "figure", "girl"))
    assert "VIEW-BEFORE:" in text and "VIEW:" in text
    assert "heads tall" in text
    tall = float(text.split("figure: ")[1].split(" heads")[0])
    assert 6.4 <= tall <= 7.6
    text = ok(draw("sketch", "head", "girl", "--yaw", "20"))
    assert "left eye" in text
    sk_before = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open("pic.easel/layers/sketch.png"))
    ok(draw("sketch", "move", "s2", "--by", "30,0"))
    sk_after = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open("pic.easel/layers/sketch.png"))
    assert (sk_before != sk_after).any()
    ok(draw("undo"))
    sk_undo = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open("pic.easel/layers/sketch.png"))
    assert (sk_undo == sk_before).all()
    text = ok(draw("sketch", "joint", "girl", "lsh", "250,560"))
    assert "joint" in text


def _lines(run):
    ok(run("do", 'line "340,200 330,350 380,470 500,520 620,470 670,350 660,200"; '
                 'line "340,200 400,120 500,100 600,120 660,200"'))


def test_line_join_gaps_and_flat_auto(draw):
    setup_bust(draw)
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    _lines(draw)
    st = json.load(open("pic.easel/state.json", encoding="utf-8"))
    lines = st["draw"]["lines"]
    assert [r["id"] for r in lines] == ["L1", "L2"]
    # the second line's ends were pulled onto the first: no open gaps between them
    text = ok(draw("line", "gaps"))
    assert "no open gaps" in text
    ok(draw("line", "move", "L2", "--by", "0,-10"))
    ok(draw("line", "weight", "L1", "1.6"))
    text = ok(draw("line", '450,300 470,320', "--no-join"))
    assert "L3" in text
    ok(draw("line", "delete", "L3"))
    ok(draw("stage", "done", "--because", "test"))
    text = ok(draw("flat", "auto"))
    assert "filled:" in text
    st = json.load(open("pic.easel/state.json", encoding="utf-8"))
    assert "face" in st["regions"] or "hair" in st["regions"]
    from PIL import Image
    flat = np.asarray(Image.open("pic.easel/layers/flat.png"))
    assert (flat[..., 3] > 0).mean() > 0.95        # every area filled, no seams under the lines


def test_owned_layer_keeps_foreign_marks(draw):
    setup_bust(draw)
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("line", "300,300 700,300"))
    ok(draw("stroke", "300,700 700,700", "--layer", "line", "--size", "4", "--color", "#ff0000"))
    ok(draw("line", "move", "L1", "--by", "0,50"))
    from PIL import Image
    arr = np.asarray(Image.open("pic.easel/layers/line.png"))
    H = arr.shape[0]
    # the red stroke drawn by another command is still there after L1 was moved
    assert arr[int(0.7 * H) - 2:int(0.7 * H) + 3, :, 3].max() > 0
    red = arr[int(0.7 * H) - 2:int(0.7 * H) + 3][..., 0].max()
    assert red > 150
    # L1 moved: nothing left at y=300, ink at y=350
    assert arr[int(0.3 * H) - 1:int(0.3 * H) + 2, 140:260, 3].max() == 0
    assert arr[int(0.35 * H) - 3:int(0.35 * H) + 4, 140:260, 3].max() > 0


def test_shade_shadow_detail_tools(draw):
    setup_bust(draw)
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    _lines(draw)
    ok(draw("line", "400,300 420,280 450,300 420,320", "--closed"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("flat", "auto"))
    st = json.load(open("pic.easel/state.json", encoding="utf-8"))
    regs = list(st["regions"])
    ok(draw("stage", "done", "--because", "test"))
    target = "face" if "face" in regs else [r for r in regs if r != "background"][0]
    text = ok(draw("shade", target, "--levels", "2"))
    assert "marks" in text
    ok(draw("rim", target))
    ok(draw("blush", target))
    if "hair" in regs and "face" in regs:
        ok(draw("shadow", "--from", "hair", "--onto", "face"))
        ok(draw("hair", "hair", "--flow", "500,100 450,300 420,600"))
    ok(draw("pattern", target, "dots", "--spacing", "20"))
    ok(draw("scatter", target, "--kind", "star", "--count", "10", "--size", "12"))
    ok(draw("fold", "400,450 450,480 500,470", "--region", target))
    ok(draw("eye", "--box", "380,280,470,330"))
    text = ok(draw("status"))
    assert "stage shade" in text or "[shade]" in text


def test_painterly_bands_paint_haze(draw):
    ok(draw("new", "lake", "--size", "400x300", "--request", "a mountain lake at sunset, painterly"))
    ok(draw("do", "plan subject mountains --box 0,250,1000,650; plan light left; palette make --base orange; "
                  "palette pick 1"))
    ok(draw("palette", "set", "sky", "#f0a070", "rock", "#6a5a70", "water", "#5070a0"))
    ok(draw("band", "sky", "0,500 1000,500", "--above", "--color", "sky", "--color2", "#ffd0a0", "--depth", "9"))
    ok(draw("band", "mountains", "0,520 300,300 520,420 700,280 1000,500", "--below", "--color", "rock",
            "--depth", "5", "--rough", "0.01"))
    ok(draw("band", "lake", "0,620 1000,610", "--below", "--color", "water", "--depth", "1"))
    st = json.load(open("lake.easel/state.json", encoding="utf-8"))
    assert {"sky", "mountains", "lake"} <= set(st["regions"])
    ok(draw("paint", "mountains", "--dir", "60", "--size", "8"))
    ok(draw("haze", "--amount", "0.5"))
    ok(draw("scatter", "100,900 500,880 900,900", "--kind", "grass", "--count", "30", "--color", "#304020"))
    from PIL import Image
    paint = np.asarray(Image.open("lake.easel/layers/paint.png"))
    assert (paint[..., 3] > 0).mean() > 0.95


def test_errors_are_plain(draw):
    setup_bust(draw)
    code, text = draw("line", "100")
    assert code == 2 and text.startswith("ERROR:") and "EXAMPLE:" in text
    code, text = draw("shade", "neck")
    assert code == 2 and "ERROR:" in text
    code, text = draw("sketch", "joint", "s9", "lsh", "1,1")
    assert code == 2


def test_reference_projects_keep_reference_commands(draw, tmp_path):
    from PIL import Image
    img = np.full((60, 80, 3), 200, np.uint8)
    img[20:40, 20:60] = (30, 30, 30)
    Image.fromarray(img).save(tmp_path / "ref.png")
    ok(draw("new", "r", "--ref", "ref.png"))
    code, text = draw("next")
    assert "marks" in text or "region" in text
    assert "plan subject" not in text


def test_todo_names_the_wear_hand_and_locks_commands():
    from pixelcanvas.easel import workflow_draw
    box = [100, 100, 900, 900]
    parts = {n: {"of": "girl", "box": box, "focus": 2, "swatch": "x"}
             for n in ("coat", "red-scarf", "hood", "robe", "sleeves", "hand-r", "hand-l", "hair", "face")}
    st = {"style": "cel", "plan": None, "stage": "plan",
          "draw": {"subjects": {"girl": {"box": box}}, "parts": parts}}
    workflow_draw.build_todo(st)
    line = {t["region"]: t["cmd"] for t in st["todo"] if t["stage"] == "line" and t["key"] == "line"}
    assert line["coat"].startswith("easel wear girl coat")
    assert line["red-scarf"].startswith("easel wear girl scarf")
    assert line["hood"].startswith("easel wear girl hood")
    assert line["robe"].startswith("easel wear girl robe")
    assert line["sleeves"].startswith("easel wear girl sleeves")
    assert line["hand-r"].startswith("easel hand girl r") and line["hand-l"].startswith("easel hand girl l")
    assert line["hair"].startswith("easel locks hair --cut long")
    detail = {t["region"]: t["cmd"] for t in st["todo"] if t["stage"] == "detail" and t["key"] == "detail"}
    assert not detail["hair"].startswith("easel locks shine")       # no locks yet
    st["draw"]["locks"] = {"hair": {}}
    workflow_draw.build_todo(st)
    detail = {t["region"]: t["cmd"] for t in st["todo"] if t["stage"] == "detail" and t["key"] == "detail"}
    assert detail["hair"] == "easel locks shine hair"
