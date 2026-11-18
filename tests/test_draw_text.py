"""Lettering on drawings: title defaults (one line, bold, the planned box) and the fit check after ``pixelcanvas text``."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from pixelcanvas import cli, workflow

REQUEST = "minimalist jazz poster: saxophone silhouette, big title BLUE NOTE, two colours plus black"


@pytest.fixture
def draw(tmp_path, monkeypatch):
    """The draw hook with draw_text installed, for one test; pixelcanvas's tables are restored afterwards."""
    from pixelcanvas import cli_draw, draw_text
    saved_cmds = dict(cli.COMMANDS)
    saved_usage = dict(cli.USAGE)
    names = ("next_action", "check_gate", "build_todo", "plan_lines", "gate_words")
    saved_wf = {n: getattr(workflow, n) for n in names}
    if "draw_text" not in cli_draw.EXTENSIONS:
        monkeypatch.setattr(cli_draw, "EXTENSIONS", tuple(cli_draw.EXTENSIONS) + ("draw_text",))
    cli_draw.install()
    # another module may own the "text" wrap slot: wrap on top (draw_text never wraps itself twice)
    cli.COMMANDS["text"] = draw_text.wrap_text(cli.COMMANDS["text"])
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


def state():
    return json.load(open("pic.pixelcanvas/state.json", encoding="utf-8"))


def text_alpha():
    return np.asarray(Image.open("pic.pixelcanvas/layers/text.png").convert("RGBA"))[..., 3]


def poster(run):
    ok(run("new", "pic", "--size", "1000x1250", "--request", REQUEST))
    ok(run("do", "plan subject sax --box 250,100,750,800; plan part title --box 120,850,880,1000; "
                 "palette make; palette pick 1"))


def ink_box(alpha):
    ys = np.flatnonzero((alpha > 10).any(1))
    xs = np.flatnonzero((alpha > 10).any(0))
    return xs[0], ys[0], xs[-1] + 1, ys[-1] + 1


def test_poster_title_one_line_bold_inside_box(draw):
    from pixelcanvas import draw_text
    poster(draw)
    text = ok(draw("text", "BLUE NOTE", "--box", "120,850,880,1000", "--color", "title"))
    assert "title: one line fitted to the box" in text and "planned part title" in text
    assert "FIX:" not in text
    it = state()["texts"][-1]
    assert it["weight"] in ("bold", "black")
    assert it["weight"] == "black"                   # the request asks for a big title
    assert it["fit"] is True and it["max_width"] is None
    # both words on the canvas, inside the box: x 120-880, y 1062-1250 px on 1000x1250
    x0, y0, x1, y1 = ink_box(text_alpha())
    assert x0 >= 118 and x1 <= 882 and y0 >= 1060 and y1 <= 1250
    assert x1 - x0 > 600                             # "BLUE NOTE" across the box, not "BLUE" alone
    assert y1 - y0 < 188 * 0.8                       # one line
    assert draw_text.overflow(1250, 1000, it) is None


def test_box_from_the_only_planned_text_part(draw):
    poster(draw)
    text = ok(draw("text", "BLUE NOTE"))
    assert "box 120,850,880,1000 of the planned part title" in text
    it = state()["texts"][-1]
    assert it["box"] == [120.0, 850.0, 880.0, 1000.0] and it["fit"] is True


def test_overflow_below_the_canvas_reports_fix(draw):
    poster(draw)
    text = ok(draw("text", "BLUE NOTE", "--box", "400,900,600,1000", "--size", "120"))
    fix = [ln for ln in text.splitlines() if ln.startswith("FIX:")]
    assert fix, text
    assert 'text 1: "BLUE NOTE" did not fit' in fix[0] and "at the bottom" in fix[0]
    assert "pixelcanvas text --remove 1, then pixelcanvas text" in fix[0]
    assert text.rstrip().splitlines()[-1] == "NEXT: pixelcanvas text --remove 1"
    assert state()["texts"][0]["fit"] is False       # --size given: no one-line default


def test_short_text_in_wide_box_without_plan(draw):
    ok(draw("new", "pic", "--size", "600x400", "--request", "a calm landscape painting"))
    text = ok(draw("text", "HELLO WORLD", "--box", "100,50,900,200"))
    assert "title: one line fitted to the box" in text
    it = state()["texts"][-1]
    assert it["fit"] is True and it["weight"] == "regular"      # no planned part, not a poster


def make_ref(path):
    im = Image.new("RGB", (240, 300), (255, 255, 255))
    im.paste((200, 30, 40), (40, 40, 200, 200))
    im.save(path)


def test_reference_project_passes_through(draw):
    make_ref("ref.png")
    ok(draw("new", "pic", "--ref", "ref.png", "--request", REQUEST))
    text = ok(draw("text", "BLUE NOTE", "--box", "120,850,880,1000"))
    lines = text.splitlines()
    assert not any(ln.startswith(("title: ", "lettering: weight", "FIX:")) for ln in lines), text
    it = state()["texts"][-1]
    assert it["weight"] == "regular" and it["fit"] is False
