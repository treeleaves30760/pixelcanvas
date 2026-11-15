# -*- coding: utf-8 -*-
"""pixelcanvas studio: the picture folder, cells, the parametric verbs, the cache, undo, needs, deliver, export and
the CLI. Small canvases (320x180). Also runs every python block of skills/pixelcanvas/references/studio.md as a
draft ("studio.md")."""
import contextlib
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas.studio import commands as SC  # noqa: E402
from pixelcanvas.studio.state import open_picture  # noqa: E402
from pixelcanvas.studio.compositor import Compositor, guess_role, base_name  # noqa: E402
from pixelcanvas.studio.engine import Cache  # noqa: E402

DOC = os.path.join(ROOT, "skills", "pixelcanvas", "references", "studio.md")
PROMPT = "a red ball beside a blue box, titled 'Hello'"

DRAFT = '''# %% sky
# sky: a soft gradient
cv.gradient([(0, "#1d2a55"), (1, "#8aa0c8")])

# %% ball
# ball: red
sk.paint(sk.circle(0.38, 0.62, 0.12), "#d63c2f", volume=0.5, shadow="ground", name="ball")

# %% box
# box: blue
sk.paint(sk.box(0.5, 0.45, 0.7, 0.75, 0.02), "#3a6fd8", shade=0.3, name="box")

# %% title
cv.text("Hello", box=(0.3, 0.04, 0.7, 0.2), color="#ffffff", font="serif")
'''


def run(*argv):
    """One studio command in this process -> (exit code, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = SC.main(["--dir", "art"] + [str(a) for a in argv])
    return code, buf.getvalue()


@pytest.fixture
def pic(tmp_path, monkeypatch):
    """A rendered first draft: sky (backdrop), ball, box, title."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    code, out = run("new", "demo", "--prompt", PROMPT, "--size", "320x180")
    assert code == 0, out
    (tmp_path / "draft1.py").write_text(DRAFT, encoding="utf-8")
    code, out = run("add", "--from", "draft1.py")
    assert code == 0, out
    return tmp_path


def png(path, mode=None):
    """An image as an array, the file closed again (Windows keeps open files locked)."""
    with Image.open(path) as im:
        return np.asarray(im.convert(mode) if mode else im)


def size_of(path):
    with Image.open(path) as im:
        return im.size


def data(root):
    return json.loads((root / "art" / "demo" / "picture.json").read_text(encoding="utf-8"))


def boxes(root):
    p = open_picture(str(root / "art"), "demo")
    sess = SC.Session(p)
    comp = sess.geometry()
    return {L["name"]: sess.box(comp, L["name"]) for L in p.data["layers"] if comp.role(L) != "pass"
            and comp.place_layer(L["name"]) is not None and comp.place_layer(L["name"]).box is not None}


def layer(root, name):
    return next(L for L in data(root)["layers"] if L["name"] == name)


# ------------------------------------------------------------------------------------------- new / draft / render

def test_new_writes_checklist_and_draft(tmp_path, monkeypatch):
    """`new` without --from writes no draft: it prints where to write it and its format, in at most 6 lines."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    code, out = run("new", "fox", "--prompt", "a fox carrying a lantern through a snowy wood at night, titled "
                    "'Night Walk'", "--size", "320x180")
    assert code == 0 and len(out.strip().splitlines()) <= 6, out
    folder = str(tmp_path / "art" / "fox").replace(chr(92), "/")
    assert not (tmp_path / "art" / "fox" / "draft.py").exists()
    assert _next(out) == f"NEXT: write {folder}/draft.py, then: studio add --from {folder}/draft.py"
    assert "# apple: 3, green, one bitten | in bowl" in out and "line=True" not in out
    assert "names defined in a section are seen by the sections below it" in out
    d = json.loads((tmp_path / "art" / "fox" / "picture.json").read_text(encoding="utf-8"))
    assert d["pcstudio"] == 1 and d["size"] == [320, 180] and d["kind"] == "painting"
    assert [n["text"] for n in d["needs"]] == ["text 'Night Walk'"]          # only what the prose says exactly
    assert (tmp_path / "art" / ".current").read_text().strip() == "fox"


def test_first_draft_from_one_file(pic):
    d = data(pic)
    assert [L["name"] for L in d["layers"]] == ["sky", "ball", "box", "title"]
    assert all((pic / "art" / "demo" / "layers" / f"{n}.py").is_file() for n in ("sky", "ball", "box", "title"))
    assert size_of(pic / "art" / "demo" / "preview.png") == (320, 180)
    code, out = run("render")
    assert "layers: 4; changed or to check: 0 (status --all lists every layer)" in out    # short: nothing changed
    code, out = run("status", "--all")
    assert "sky" in out and "backdrop" in out
    assert re.search(r"request: (\d+) of \1 confirmed", out), out
    assert "cells run: none" in out                          # nothing changed: everything from the cache
    assert d["layers"][0].get("auto_role") == "backdrop"
    lines = out.splitlines()
    assert len(lines) <= 40 and lines[-1].startswith("NEXT: ")
    assert all(ord(c) < 128 for c in out)                    # plain ASCII report


def test_split_draft_prelude_roles_and_errors():
    secs = SC.split_draft("import math\n# %% a\nx = 1\n# %% b [pass]\ngrade(cv, 'golden')\n")
    assert [(n, r) for n, r, _ in secs] == [("a", None), ("b", "pass")]
    assert secs[0][2].startswith("import math") and "x = 1" in secs[0][2]
    with pytest.raises(SC.StudioError, match="no '# %% name' lines"):
        SC.split_draft("x = 1\n")
    with pytest.raises(SC.StudioError, match="two sections"):
        SC.split_draft("# %% a\n# %% a\n")
    assert guess_role("grade(cv, 'golden')\ncv.vignette(0.3)") == "pass"
    assert guess_role("sk.paint(sk.circle(0.5, 0.5, 0.1), '#fff')\ngrade(cv)") is None
    assert guess_role("# role: backdrop\ncv.gradient([(0, '#000'), (1, '#fff')])") == "backdrop"
    assert base_name("fox2") == "fox" and base_name("tree_3") == "tree"


# ------------------------------------------------------------------------------------------- parametric verbs

def test_move_to_and_by(pic):
    run("move", "ball", "--to", "0.5", "0.5")
    b = boxes(pic)["ball"]
    assert abs((b[0] + b[2]) / 2 - 0.5) < 0.01 and abs((b[1] + b[3]) / 2 - 0.5) < 0.01
    run("move", "ball", "--by", "0.1,-0.05")                  # commas work too
    b2 = boxes(pic)["ball"]
    assert abs(b2[0] - b[0] - 0.1) < 0.01 and abs(b2[1] - b[1] + 0.05) < 0.01


def test_scale_keeps_the_bottom_and_rotate_flip(pic):
    b = boxes(pic)["box"]
    code, out = run("scale", "box", "0.5")
    assert code == 0 and "[x0.5" in out
    s = boxes(pic)["box"]
    assert abs((s[2] - s[0]) - 0.5 * (b[2] - b[0])) < 0.02
    assert abs(s[3] - b[3]) < 0.01 and abs((s[0] + s[2]) / 2 - (b[0] + b[2]) / 2) < 0.01
    run("scale", "box", "2")
    assert layer(pic, "box")["scale"] == pytest.approx(1.0)
    run("rotate", "box", "90")
    r = boxes(pic)["box"]
    w, h = (b[2] - b[0]) * 320, (b[3] - b[1]) * 180
    assert abs((r[2] - r[0]) * 320 - h) < 4 and abs((r[3] - r[1]) * 180 - w) < 4
    run("flip", "ball")
    assert layer(pic, "ball")["flip"] is True


def test_scale_up_draws_the_cell_larger(pic):
    code, out = run("scale", "ball", "1.6")
    assert "ball" in out.split("cells run:")[1].split(";")[0]          # re-drawn at 2x, not blown up
    p = open_picture(str(pic / "art"), "demo")
    comp = Compositor(p.data, p.read_cells(), cache=Cache(p.cache_dir))
    assert comp.result("ball").k == 2.0 and comp.result("ball").size == (640, 360)


def test_fit_order_hide_show_rm(pic):
    run("fit", "ball", "0.05", "0.1", "0.25", "0.4")
    b = boxes(pic)["ball"]
    assert b[0] >= 0.04 and b[1] >= 0.09 and b[2] <= 0.26 and b[3] <= 0.41
    run("order", "ball", "top")
    assert [L["name"] for L in data(pic)["layers"]][-1] == "ball"
    run("order", "ball", "below", "box")
    names = [L["name"] for L in data(pic)["layers"]]
    assert names.index("ball") == names.index("box") - 1
    run("order", "title", "bottom")
    code, out = run("render")
    assert "covered by sky" in out                               # under the backdrop: really hidden
    assert "[ ]" in out and "order title above sky" in out
    run("order", "title", "top")
    run("hide", "box")
    assert layer(pic, "box")["visible"] is False
    code, out = run("show", "box")
    assert layer(pic, "box")["visible"] is True
    run("rm", "box")
    assert "box" not in [L["name"] for L in data(pic)["layers"]]
    assert not (pic / "art" / "demo" / "layers" / "box.py").exists()


def test_copy_makes_exact_instances_in_a_row(pic):
    code, out = run("copy", "ball", "ball", "--n", "2", "--row", "0.05", "0.5", "0.6", "0.9")
    assert code == 0
    names = [L["name"] for L in data(pic)["layers"]]
    assert {"ball2", "ball3"} <= set(names)
    assert all(layer(pic, n)["cell"] == "ball" for n in ("ball2", "ball3"))
    bx = boxes(pic)
    xs = sorted((bx[n][0] + bx[n][2]) / 2 for n in ("ball", "ball2", "ball3"))
    assert xs[0] < xs[1] < xs[2] and all(0.05 <= bx[n][0] and bx[n][2] <= 0.61 for n in ("ball", "ball2", "ball3"))
    p = open_picture(str(pic / "art"), "demo")
    sess = SC.Session(p)
    out = sess.render("x", quiet=True)
    assert sum(1 for r in out.rows if r["name"].split(" #")[0] == "ball") == 3   # one record per instance: counts


def test_set_effects_recolor_and_scene(pic):
    code, out = run("set", "ball", "outline=on", "shadow=drop", "glow=ffd27a", "opacity=80%", "blur=0.002")
    assert code == 0
    L = layer(pic, "ball")
    assert L["outline"] == "#1d1830" and L["shadow"] == "drop" and L["glow"] == "#ffd27a"
    assert L["opacity"] == pytest.approx(0.8) and L["blur"] == pytest.approx(0.002)
    assert "outline" in out and "shadow drop" in out
    run("set", "ball", "outline=off")
    assert "outline" not in layer(pic, "ball")
    before = png(pic / "art" / "demo" / "preview.png", "RGB").astype(np.float32)
    run("recolor", "box", "gold")
    assert layer(pic, "box")["tint"] == "gold"
    after = png(pic / "art" / "demo" / "preview.png", "RGB").astype(np.float32)
    px = (slice(int(0.5 * 180), int(0.7 * 180)), slice(int(0.62 * 320), int(0.78 * 320)))
    assert after[px][..., 0].mean() > before[px][..., 0].mean() + 40      # the blue box turned gold
    run("set", "scene", "light=top-right", "grade=golden", "background=101020")
    sc = data(pic)["scene"]
    assert sc["light"] == [0.6, -0.8] and sc["grade"] == "golden" and sc["background"] == "#101020"
    code, out = run("set", "scene", "grade=goldn")
    assert code == 2 and "golden" in out


def test_shadows_stay_with_their_layer(pic):
    """A drop shadow falls away from the light; a ground shadow touches the bottom of the layer wherever it goes."""
    def region(x0, y0, x1, y1):
        a = png(pic / "art" / "demo" / "preview.png", "RGB").astype(np.float32)
        return a[int(y0 * 180):int(y1 * 180), int(x0 * 320):int(x1 * 320)].mean()
    run("set", "sky", "visible=off")
    run("set", "scene", "background=f4f4f4")
    plain = region(0.52, 0.76, 0.72, 0.8)
    run("set", "box", "shadow=drop")                             # light from the top left: shadow right and below
    assert region(0.52, 0.76, 0.72, 0.8) < plain - 10
    run("move", "ball", "--to", "0.2", "0.4")
    b = boxes(pic)["ball"]
    p = open_picture(str(pic / "art"), "demo")
    pl = SC.Session(p).geometry().place_layer("ball")
    rows = np.flatnonzero((pl.A > 0.5).any(1))
    foot = (pl.y0 + rows[-1] + 1) / 180                           # the opaque bottom, above the cell's own soft shadow
    under = (b[0] - 0.01, foot - 0.03, b[2] + 0.01, foot + 0.01)
    before = region(*under)
    run("set", "ball", "shadow=ground")
    assert region(*under) < before - 4                          # in contact with the body, wherever it went


def solids(root, name="demo"):
    """Solid boxes (glow, sparks and a thin shadow left out): what put and the check place by."""
    from pixelcanvas.studio import check as CK
    p = open_picture(str(root / "art"), name)
    comp = SC.Session(p).geometry()
    return {L["name"]: CK.solid_box(comp, L["name"]) for L in p.data["layers"]
            if comp.placed.get(L["name"]) is not None and comp.placed[L["name"]].box is not None}


def test_put_on_and_left_of(pic):
    run("put", "ball", "on", "box")
    b = solids(pic)
    assert abs(b["ball"][3] - (b["box"][1] + 0.06 * (b["box"][3] - b["box"][1]))) < 0.02
    assert layer(pic, "ball")["shadow"] == "ground"
    names = [L["name"] for L in data(pic)["layers"]]
    assert names.index("ball") > names.index("box")
    run("put", "ball", "left", "of", "box")                     # words as a person types them
    b = solids(pic)
    assert b["ball"][2] < b["box"][0] and abs(b["ball"][3] - b["box"][3]) < 0.02


def test_text_records_move_with_their_layer(pic):
    p = open_picture(str(pic / "art"), "demo")
    sess = SC.Session(p)
    sess.render("x", quiet=True)
    t0 = sess.comp.canvas.text_log[0]["bbox"]
    run("move", "title", "--by", "0", "0.5")
    sess = SC.Session(open_picture(str(pic / "art"), "demo"))
    out = sess.render("x", quiet=True)
    t1 = sess.comp.canvas.text_log[0]["bbox"]
    assert abs((t1[1] - t0[1]) - 0.5 * 180) <= 2
    assert any(n["kind"] == "text" and n["ok"] for n in out.needs)
    code, out = run("scale", "title", "6")                      # off the frame: the text item is lost
    assert "[ ]" in out and "runs off the frame" in out
    assert "NEXT: this edit lost a confirmed item" in out and "studio undo" in out


# ------------------------------------------------------------------------------------------- errors / cache / history

def test_cell_error_is_isolated_and_keeps_the_last_good_render(pic):
    lay = pic / "art" / "demo" / "layers"
    good = png(pic / "art" / "demo" / "preview.png", "RGB")
    (lay / "ball.py").write_text('# ball\nsk.paint(sk.circle(0.3, 0.6, 0.12), "#d63c2f", volme=0.5)\n',
                                 encoding="utf-8")
    code, out = run("render")
    assert code == 0
    line = next(l for l in out.splitlines() if l.strip().startswith("ball"))
    assert "ERROR line 2" in line and "volme= (did you mean volume=?)" in line and "(last good render kept)" in line
    assert "    the call: sk.paint(mask, color, " in out                   # no extra api turn needed
    assert "layers/ball.py line 2" in out and "NEXT: fix" in out
    now = png(pic / "art" / "demo" / "preview.png", "RGB")
    assert np.abs(now.astype(int) - good.astype(int)).max() <= 2  # the picture still has every layer
    (lay / "box.py").write_text("sk.paint(sk.box(0.6, 0.45, 0.8, 0.75), '#3a6fd8'\n", encoding="utf-8")
    code, out = run("render")
    assert "SyntaxError" in out and "never closed" in out
    (lay / "ball.py").write_text("# ball\nsk.paint(sk.circle(0.3, 0.6, 0.12), '#d63c2f', volume=0.5)\n",
                                 encoding="utf-8")
    (lay / "box.py").write_text("# box\nsk.paint(sk.box(0.6, 0.45, 0.8, 0.75), '#3a6fd8')\n", encoding="utf-8")
    code, out = run("render")
    assert "ERROR" not in out


def test_new_cell_without_last_good_draws_nothing(pic):
    (pic / "art" / "demo" / "layers" / "moon.py").write_text("sk.paint(sk.circle(0.8, 0.2, 0.05), undefined_name)\n",
                                                            encoding="utf-8")
    code, out = run("render")
    assert "moon" in out and "(nothing drawn)" in out and "is not defined" in out


def test_unchanged_cells_never_run_again(pic):
    lay = pic / "art" / "demo" / "layers"
    count = pic / "runs.txt"
    (lay / "ball.py").write_text(
        f"open({str(count)!r}, 'a').write('x')\nsk.paint(sk.circle(0.3, 0.6, 0.12), '#d63c2f', volume=0.5)\n",
        encoding="utf-8")
    run("render")
    assert count.read_text() == "x"
    for argv in (("move", "ball", "--by", "0.05", "0"), ("set", "ball", "outline=on"), ("order", "ball", "top"),
                 ("scale", "ball", "0.8"), ("hide", "box"), ("show", "box"), ("recolor", "ball", "orange")):
        code, out = run(*argv)
        assert code == 0 and "ball" not in out.split("cells run:")[1].split(";")[0], argv
    (lay / "box.py").write_text("# box\nsk.paint(sk.box(0.6, 0.4, 0.8, 0.75), '#3a6fd8')\n", encoding="utf-8")
    code, out = run("render")
    assert "box" in out.split("cells run:")[1].split(";")[0]
    assert count.read_text() == "x"                               # the ball cell ran exactly once


def test_cells_follow_other_layers_with_box_and_anchor(pic):
    lay = pic / "art" / "demo" / "layers"
    (lay / "tag.py").write_text('# tag: hangs under the ball\nx, y = anchor("ball", "bottom")\n'
                                'x0, y0, x1, y1 = box("ball")\n'
                                'sk.paint(sk.box(x - 0.03, y + 0.01, x + 0.03, y + 0.08), "#ffcf70")\n',
                                encoding="utf-8")
    run("render")
    b = boxes(pic)
    assert abs((b["tag"][0] + b["tag"][2]) / 2 - (b["ball"][0] + b["ball"][2]) / 2) < 0.01
    code, out = run("move", "ball", "--by", "0.2", "0")
    assert "tag" in out.split("cells run:")[1].split(";")[0]        # it follows: drawn again
    b2 = boxes(pic)
    assert abs((b2["tag"][0] + b2["tag"][2]) / 2 - (b2["ball"][0] + b2["ball"][2]) / 2) < 0.01
    (lay / "tag.py").write_text('box("bal")\n', encoding="utf-8")
    code, out = run("render")
    assert "did you mean 'ball'" in out


def test_undo_redo_restore_code_and_params(pic):
    lay = pic / "art" / "demo" / "layers"
    src0 = (lay / "ball.py").read_text(encoding="utf-8")
    run("scale", "ball", "0.5")
    (lay / "ball.py").write_text("# ball\nsk.paint(sk.circle(0.3, 0.6, 0.05), '#00ff00')\n", encoding="utf-8")
    run("render")
    rev = data(pic)["rev"]
    code, out = run("undo")
    assert code == 0 and f"from {rev}" in out
    assert (lay / "ball.py").read_text(encoding="utf-8") == src0 and layer(pic, "ball")["scale"] == 0.5
    run("undo")
    assert layer(pic, "ball").get("scale", 1.0) == 1.0
    run("redo")
    assert layer(pic, "ball")["scale"] == 0.5
    run("redo")
    assert "00ff00" in (lay / "ball.py").read_text(encoding="utf-8")
    run("rm", "box")
    run("undo")
    assert (lay / "box.py").exists() and "box" in [L["name"] for L in data(pic)["layers"]]
    code, out = run("history")
    assert "* rev" in out and "scale ball 0.5" in out
    code, out = run("goto", "2")
    assert data(pic)["rev"] == 2
    code, out = run("compare", "3")
    assert code == 0 and (pic / "art" / "demo" / "compare.png").is_file()


def test_needs_and_score_per_revision(pic):
    code, out = run("need", "add", "the ball is shiny", "--layer", "ball")
    m = re.search(r"request: (\d+) of (\d+) confirmed", out)
    assert m and m.group(1) == m.group(2)                       # bound to a visible, whole layer: ticked
    run("need", "add", "the sky feels calm")
    code, out = run("need")
    assert "the sky feels calm" in out and "need ok" in out
    nid = max(n["id"] for n in data(pic)["needs"])
    run("need", "ok", str(nid), "--why", "soft gradient")
    n = next(x for x in data(pic)["needs"] if x["id"] == nid)
    assert n["ok"] is True and n["why"] == "soft gradient"
    code, out = run("hide", "ball")
    assert "[ ]" in out and "ball is hidden" in out
    idx = json.loads((pic / "art" / "demo" / "history" / "index.json").read_text(encoding="utf-8"))
    sc = [r["score"] for r in idx["revs"] if r.get("score")]
    assert sc[-1][0] < sc[-2][0]                                 # the score of each revision is kept
    run("need", "reopen", str(nid))
    assert next(x for x in data(pic)["needs"] if x["id"] == nid)["ok"] is None
    run("note", "add", "warmer light on the ball")
    code, out = run("render")
    assert "notes: 1 open" in out
    run("note", "done", "1")
    assert data(pic)["notes"][0]["done"] is True


def test_missing_items_get_a_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    run("new", "p", "--prompt", "a green apple next to a cup, titled 'Lunch'", "--size", "320x180")
    (tmp_path / "d.py").write_text("# %% bg\ncv.gradient([(0, '#334'), (1, '#889')])\n# %% apple\n"
                                   "# apple: green | next to cup\n"
                                   "sk.paint(sk.circle(0.3, 0.6, 0.1), '#d63c2f', volume=0.5)\n"
                                   "# %% cup\n# cup: white\n", encoding="utf-8")
    code, out = run("add", "--from", "d.py")
    assert "painted mostly red, the line says green: recolor apple 4caf50" in out
    assert "cup: white -> cup has no code yet: edit section cup of the draft (it draws the cup)" in out, out
    assert "add title --text \"Lunch\"" in out
    code, out = run("recolor", "apple", "4caf50")
    assert re.search(r"\[x\] \d+ apple: green \| next to cup", out), out


# ------------------------------------------------------------------------------------------- shortcuts / look

def test_add_text_and_kit_shortcuts(pic):
    code, out = run("add", "caption", "--text", "Hi there", "--box", "0.1", "0.8", "0.9", "0.95", "--color", "ffcc00")
    assert code == 0
    src = (pic / "art" / "demo" / "layers" / "caption.py").read_text(encoding="utf-8")
    assert "cv.text('Hi there', box=(0.1, 0.8, 0.9, 0.95), color='#ffcc00'" in src
    code, out = run("add", "cat", "--kit", "creature('cat', at=(0.5, 0.95), size=0.3)")
    assert code == 0 and "cat = creature(cv, 'cat'" in (pic / "art" / "demo" / "layers" / "cat.py").read_text()
    code, out = run("add", "dog", "--kit", "creature dog at=0.7,0.95 size=0.25 pose=sit")
    src = (pic / "art" / "demo" / "layers" / "dog.py").read_text(encoding="utf-8")
    assert "creature(cv, 'dog', at=(0.7, 0.95), size=0.25, pose='sit')" in src
    assert "ERROR" not in out


def test_look_writes_one_image(pic):
    look = pic / "art" / "demo" / "look.png"
    for argv in (("look",), ("look", "ball"), ("look", "--zoom", "0.2", "0.4", "0.5", "0.8")):
        if look.exists():
            look.unlink()
        code, out = run(*argv)
        assert code == 0 and look.is_file() and "art/demo/look.png = " in out and ": " in out, argv
        assert out.startswith("look: " + str(look).replace(chr(92), "/"))           # absolute, forward slashes
        assert max(size_of(look)) <= 1280
    code, out = run("look", "--text", "ball")
    assert "mean colour #" in out and "(red)" in out


# ------------------------------------------------------------------------------------------- deliver / export / CLI

def test_deliver_full_size_and_layers(pic):
    code, out = run("deliver", "out/final.png", "--scale", "2", "--layers")
    assert code == 0 and "delivered" in out
    assert size_of(pic / "out" / "final.png") == (640, 360)
    assert (pic / "out" / "final.psd").is_file() and (pic / "out" / "final.ora").is_file()
    import zipfile
    names = re.findall(r'name="([^"]+)"', zipfile.ZipFile(pic / "out" / "final.ora").read("stack.xml").decode())
    assert {"sky", "ball", "box", "title"} <= set(names)


def test_export_reproduces_the_picture(pic):
    run("set", "ball", "shadow=drop", "outline=on")
    run("scale", "box", "0.7")
    run("deliver", "a.png")
    code, out = run("export", "scene_demo.py")
    assert code == 0
    from pixelcanvas import cli
    mod = cli.import_scene(str(pic / "scene_demo.py"))
    assert mod.PROMPT == PROMPT and mod.DEFAULT_SIZE == (320, 180)
    cv = mod.render(320, 180)
    a = png(pic / "a.png", "RGB").astype(np.int16)
    b = np.asarray(cv.to_pil(), np.int16)
    assert np.abs(a - b).max() <= 3
    assert {"ball", "box"} <= {e["name"] for e in cv.manifest if "name" in e}


def test_cli_through_a_subprocess(tmp_path):
    tool = os.path.join(ROOT, "tools", "studio.py")
    env = dict(os.environ, PYTHONIOENCODING="cp950" if os.name == "nt" else "utf-8")
    r = subprocess.run([sys.executable, tool, "new", "sub", "--prompt", "a yellow star", "--size", "320x180"],
                       cwd=tmp_path, capture_output=True, text=True, env=env, timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "/art/sub/draft.py, then: " in r.stdout and "tools/studio.py add --from " in r.stdout
    r = subprocess.run([sys.executable, tool, "add", "star", "--kit",
                        "pictograms.pictogram('star', at=(0.5, 0.5), size=0.5, color='#f5d142')"],
                       cwd=tmp_path, capture_output=True, text=True, env=env, timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and "1 of 1 confirmed" in r.stdout, r.stdout + r.stderr
    pc = os.path.join(ROOT, "tools", "pc.py")
    r = subprocess.run([sys.executable, pc, "studio", "scale", "star", "0.5"], cwd=tmp_path, capture_output=True,
                       text=True, env=env, timeout=300, stdin=subprocess.DEVNULL)
    assert r.returncode == 0 and "rev 3: scale star 0.5" in r.stdout, r.stdout + r.stderr


def test_did_you_mean_for_verbs_layers_keys_and_options(pic):
    code, out = run("scael", "ball", "2")
    assert code == 2 and "did you mean scale" in out and out.strip().splitlines()[-1].startswith("NEXT:")
    code, out = run("move", "bal", "--by", "0.1", "0")
    assert code == 2 and "did you mean 'ball'" in out
    code, out = run("set", "ball", "outlne=on")
    assert code == 2 and "did you mean outline" in out
    code, out = run("move", "ball", "--too", "0.5", "0.5")
    assert code == 2 and "did you mean --to" in out
    code, out = run("order", "ball", "tpo")
    assert code == 2 and "did you mean top" in out
    code, out = run("set", "ball", "shadow=grund")
    assert code == 2 and "ground" in out
    code, out = run("-p", "dmeo", "render")
    assert code == 2 and "did you mean 'demo'" in out
    code, out = run("recolor", "ball", "notacolour")
    assert code == 2 and "is not a colour" in out


def test_cli_lists_studio():
    from pixelcanvas import cli
    assert "studio" in cli.available_tools(ROOT) and "studio new NAME" in cli._usage()
    import pixelcanvas
    assert pixelcanvas.studio.render_picture is not None


# ------------------------------------------------------------------------------------------- review rounds

def _next(out):
    return out.strip().splitlines()[-1]


def test_two_review_rounds_then_deliver(pic):
    """Three looks, one per change of the picture, then finished for good: no command is needed to finish."""
    code, out = run("render")                                     # unchanged since the first draft's look 1
    look = str(pic / "art" / "demo" / "look.png").replace(chr(92), "/")
    draft = str(pic / "draft1.py").replace(chr(92), "/")
    assert _next(out) == (f"NEXT: look 1 of 3: open {look} (the picture, then every item close up); a line above "
                          f"that does not show at a glance: edit its section in {draft}, then studio add --from "
                          f"{draft} (or one command: move, scale, order, crop ...); every line shows: finished - "
                          "studio deliver ./demo.png")
    assert "look list ([x] = what can be measured holds; by eye: does the picture show each line at a glance?):" in out
    assert out.count("3 box: blue") == 1 and "  [x] 3 box: blue" in out and "reviews:" not in out
    assert "note none" not in out
    code, out = run("note", "add", "the ball looks flat", "the box needs a shadow")
    assert code == 0 and "note add 1 2" in out.splitlines()[0]
    assert [n["layer"] for n in data(pic)["notes"]] == ["ball", "box"]
    assert _next(out).startswith("NEXT: fix note 1: the ball looks flat - studio look ball shows it")
    run("set", "ball", "shadow=ground")
    run("set", "box", "shadow=drop")
    code, out = run("note", "done", "1", "2")
    assert _next(out).startswith(f"NEXT: look 2 of 3: open {look} once more (the picture, then every item close up)")
    code, out = run("render")                                     # nothing changed: still look 2
    assert _next(out).startswith(f"NEXT: look 2 of 3: ")
    code, out = run("move", "ball", "--by", "0.01", "0")
    assert _next(out).startswith(f"NEXT: look 3 of 3: open {look} once more")
    code, out = run("render")
    assert _next(out) == "NEXT: finished: studio deliver ./demo.png, then open it before you tell the user"
    assert "look list" not in out
    code, out = run("move", "ball", "--by", "0.01", "0")          # a later change: still finished, for good
    assert _next(out).startswith("NEXT: finished: ") and "look again" not in out
    code, out = run("note", "add", "the sky is too plain")       # only open work brings a step back
    assert _next(out).startswith("NEXT: fix note ")
    run("note", "done", "3")
    code, out = run("status")
    assert _next(out).startswith("NEXT: finished: ")


def test_review_limit_and_deliver_with_things_open(pic):
    run("set", "scene", "reviews=1")
    assert data(pic)["scene"]["reviews"] == 1
    code, out = run("note", "add", "the sky is too plain")
    assert _next(out).startswith("NEXT: fix note 1: the sky is too plain")
    code, out = run("note", "done", "1")
    assert _next(out).startswith("NEXT: finished: ")
    run("hide", "ball")                                               # a request item open again
    run("note", "add", "the box could be warmer")
    lay = pic / "art" / "demo" / "layers"
    (lay / "box.py").write_text("sk.paint(sk.box(0.6, 0.45, 0.8, 0.75), '#3a6fd8', volme=1)\n", encoding="utf-8")
    code, out = run("deliver", "out/x.png")                          # never refused
    assert code == 0 and (pic / "out" / "x.png").is_file()
    tail = out.split("delivered:")[1].splitlines()
    assert any(l.startswith("still open: box fails") and "last good render" in l for l in tail)
    assert any(l.startswith("still open: request items") and "notes 2 the box could be warmer" in l for l in tail)
    assert not any(l.startswith("reviews:") for l in tail)
    assert any(l.startswith("NEXT: open ") and "tell the user that path" in l for l in tail)


def test_review_suggestions_name_a_floating_object(pic):
    lay = pic / "art" / "demo" / "layers"
    (lay / "floor.py").write_text("# floor\nsk.paint(sk.box(0.0, 0.8, 1.0, 1.0), '#6b4a2b', name='floor')\n",
                                  encoding="utf-8")
    run("set", "scene", "reviews=6")
    run("render")
    run("order", "floor", "above", "sky")
    code, out = run("move", "box", "--to", "0.6", "0.5")
    assert "NEXT: look " in out, out
    assert "worth checking (suggestions, never blocking):" in out and "box may float" in out
    code, out = run("render")
    assert "box may float" not in out                              # said once, not again until it gets worse
    code, out = run("move", "box", "--by", "0", "-0.1")         # higher up: worse, so said again
    assert "box may float" in out


# ------------------------------------------------------------------------------------------- look sheet / put / copy

def test_look_sheet_has_the_picture_and_tiles(pic):
    code, out = run("look")
    look = pic / "art" / "demo" / "look.png"
    assert code == 0 and "above 5 items close up in the picture: 1 sky, 2 ball, 3 box, 4 title, 5 text 'Hello'" \
        in out, out
    w, h = size_of(look)
    assert w == 1280 and h <= 1280
    a = png(look, "RGB")
    assert a[h - 60:h - 30].std() > 5                                 # the tiles under the picture
    code, out = run("look", "ball", "box")
    assert "2 layers alone, each with its box: ball, box" in out and max(size_of(look)) <= 1280


def test_put_on_a_surface_lands_inside_it(pic):
    lay = pic / "art" / "demo" / "layers"
    (lay / "floor.py").write_text("# floor\nsk.paint(sk.box(0.0, 0.7, 1.0, 1.0), '#6b4a2b', name='floor')\n",
                                  encoding="utf-8")
    run("render")
    run("order", "floor", "above", "sky")
    run("move", "ball", "--to", "0.3", "0.3")
    code, out = run("put", "ball", "on", "floor")
    assert "put: ball stands on the surface of floor (a quarter down its top)" in out
    b = solids(pic)
    assert abs(b["ball"][3] - (0.7 + 0.25 * 0.3)) < 0.02
    run("move", "ball", "--to", "0.3", "0.85")
    y = solids(pic)["ball"][3]
    code, out = run("put", "ball", "on", "floor")
    assert "(its height kept)" in out and abs(solids(pic)["ball"][3] - y) < 0.01


def test_covered_by_only_for_objects(pic):
    lay = pic / "art" / "demo" / "layers"
    (lay / "floor.py").write_text("# floor\nsk.paint(sk.box(0.0, 0.4, 1.0, 1.0), '#6b4a2b', name='floor')\n",
                                  encoding="utf-8")
    run("render")
    run("order", "floor", "above", "sky")
    code, out = run("status", "--all")
    floor = next(l for l in out.splitlines() if l.strip().startswith("floor"))
    assert "visible" not in floor and "covered by" not in floor          # a surface is meant to be covered


def test_copy_total_names_itself(pic):
    code, out = run("copy", "ball", "--total", "3", "--row", "0.05", "0.5", "0.6", "0.9")
    assert code == 0 and "copy ball -> ball2 ball3" in out
    assert all(layer(pic, n)["cell"] == "ball" for n in ("ball2", "ball3"))
    code, out = run("copy", "ball", "--total", "3")
    assert code == 2 and "there are already 3" in out
    code, out = run("copy", "ball", "--total", "4")
    assert code == 0 and "ball4" in [L["name"] for L in data(pic)["layers"]]
    code, out = run("copy", "box", "crate")
    assert code == 0 and layer(pic, "crate")["cell"] == "box"


def test_glow_outside_the_solid_box_survives_a_move(pic):
    run("set", "sky", "visible=off")
    lay = pic / "art" / "demo" / "layers"
    (lay / "lamp.py").write_text('# lamp\nsk.paint(sk.circle(0.8, 0.3, 0.04), "#ffd27a", glow="#ffd27a", '
                                 'name="lamp")\n', encoding="utf-8")
    run("render")
    run("move", "lamp", "--by", "-0.3", "0.2")
    b = boxes(pic)["lamp"]
    assert b[2] - b[0] < 0.07                                          # the box is the solid disc, not its glow
    a = png(pic / "art" / "demo" / "preview.png", "RGB").astype(np.float32)
    x1, cy = int(b[2] * 320), int((b[1] + b[3]) / 2 * 180)
    ring = a[cy - 2:cy + 3, x1 + 2:x1 + 6].mean()
    far = a[cy - 2:cy + 3, x1 + 40:x1 + 44].mean()
    assert ring > far + 15                                             # the glow beside the disc moved with it


def test_slow_cell_is_stopped(pic):
    run("set", "scene", "cell_timeout=1")
    (pic / "art" / "demo" / "layers" / "slow.py").write_text("while True:\n    pass\n", encoding="utf-8")
    code, out = run("render")
    line = next(l for l in out.splitlines() if l.strip().startswith("slow"))
    assert "ran longer than 1 s and was stopped" in line and "cell_timeout=2" in line
    assert code == 0 and "NEXT: fix" in out



def test_scene_layer_keeps_the_chart_records(pic):
    import shutil
    shutil.copy(os.path.join(ROOT, "scenes", "_template_chart.py"), pic / "chart.py")
    code, out = run("new", "c", "--prompt", "a chart titled 'Solar now beats coal on cost', stamped 'Draft'",
                    "--size", "480x270")
    code, out = run("add", "base", "--scene", "chart.py")
    assert code == 0 and "ERROR" not in out
    assert "cv = scene('../../chart.py')" in (pic / "art" / "c" / "layers" / "base.py").read_text(encoding="utf-8")
    assert "[x]" in out and "text 'Solar now beats coal on cost'" in out.split("[x]")[1]
    code, out = run("add", "stamp", "--text", "Draft", "--box", "0.7", "0.05", "0.95", "0.15", "--color", "d63c2f")
    assert "2 of 3 confirmed" in out and "NEXT: studio need ok 3" in out      # the chart's data: by eye
    p = open_picture(str(pic / "art"), "c")
    o = SC.Session(p).render("x", quiet=True)
    assert o.data.get("vector_labels") and "Draft" in [t["text"] for t in o.comp.canvas.text_log]
    (pic / "chart.py").write_text((pic / "chart.py").read_text(encoding="utf-8").replace(
        'TITLE = "Solar now beats coal on cost"', 'TITLE = "Coal"'), encoding="utf-8")
    code, out = run("render")
    assert "base" in out.split("cells run:")[1].split(";")[0]           # the scene file changed: drawn again


# ------------------------------------------------------------------------------------------- api

def test_api_signatures_words_and_did_you_mean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code, out = run("api", "sk.paint", "creature")
    assert code == 0 and out.startswith("sk.paint(mask, color,") and "volume=0.0" in out and "e.g." in out
    assert "species:" in out and "fox" in out and "poses:" in out and "sit" in out
    code, out = run("api", "paint")
    assert "sk.paint(" in out
    code, out = run("api", "steam")
    assert "steam(" in out
    code, out = run("api", "sk.pant")
    assert code == 2 and "did you mean sk.paint" in out
    code, out = run("api")
    assert code == 0 and "shapes" in out and "kits" in out and len(out.splitlines()) <= 20


def test_wrong_argument_points_at_api(pic):
    (pic / "art" / "demo" / "layers" / "ball.py").write_text(
        '# ball\nsk.paint(sk.circle(0.3, 0.6, 0.12), "#d63c2f", volme=0.5)\n', encoding="utf-8")
    code, out = run("render")
    line = next(l for l in out.splitlines() if l.strip().startswith("ball"))
    assert "-> studio api sk.paint" in line
    assert "the call's arguments: studio api sk.paint" in _next(out)


# ------------------------------------------------------------------------------------------- round 3: the contract


def data_of(root, name):
    return json.loads((root / "art" / name / "picture.json").read_text(encoding="utf-8"))


def test_a_chart_request_starts_from_its_scene_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    code, out = run("new", "sales", "--prompt", "a bar chart of monthly sales from January to June titled 'Sales'",
                    "--size", "320x180")
    d = data_of(tmp_path, "sales")
    assert d["kind"] == "chart" and [n["text"] for n in d["needs"]] == [
        "text 'Sales'", "the chart shows the data of the request"]
    nxt = out.strip().splitlines()[-1]
    assert "new_scene.py sales_base --prompt" in nxt and "--dir art/sales" in nxt
    assert "studio add base --scene art/sales/sales_base.py" in nxt


def test_extra_findings_never_block_the_review(pic):
    p = open_picture(str(pic / "art"), "demo")
    sess = SC.Session(p)
    out = sess.render("x", quiet=True)
    assert all(n["ok"] for n in out.needs)
    out.extra = [{"reason": "the request names 5 things, 3 are recorded", "cmd": "write layers/sign.py (draws the "
                  "sign), then render", "rule": "prompt_unrecorded"}]
    nxt = SC.next_step(sess, sess.comp, out, None, [])
    assert nxt.startswith("look 1 of 3: open ")
    code, out = run("render")
    assert "[ ] -" not in out


def test_a_note_on_a_settled_changed_picture_is_the_review(pic):
    """A blind `note none` right after a render changes nothing: no revision, the look NEXT again."""
    rev = data(pic)["rev"]
    code, out = run("note", "none")
    assert code == 0 and _next(out).startswith("NEXT: look 1 of 3: open ")
    assert data(pic)["rev"] == rev
    run("note", "add", "the ball could be shinier")
    code, out = run("note", "none")                                     # harmless with a note open too
    assert code == 0 and _next(out).startswith("NEXT: fix note 1: the ball could be shinier")


def test_unassigned_creature_keeps_its_anchors(pic):
    lay = pic / "art" / "demo" / "layers"
    (lay / "fox.py").write_text('# fox\ncreature(cv, "fox", at=(0.3, 0.9), size=0.4)\n', encoding="utf-8")
    (lay / "lamp.py").write_text('# lamp\nx, y = anchor("fox", "mouth")\nsk.paint(sk.circle(x, y, 0.02), "#ffcf70")\n',
                                 encoding="utf-8")
    code, out = run("render")
    assert "ERROR" not in out and "unknown anchor" not in out


def test_set_scene_prompt_keeps_eye_ticks(pic):
    run("need", "ok", "1", "--why", "seen")
    run("set", "scene", "prompt=" + PROMPT + " at noon")
    n = next(x for x in data(pic)["needs"] if x["id"] == 1)
    assert n["ok"] is True and n["why"] == "seen"


def test_copy_row_keeps_the_size_and_says_when_narrow(pic):
    """copy --row leaves a gap between copies; a row too narrow shrinks them and names a wider one."""
    code, out = run("copy", "ball", "--total", "4", "--row", "0.1", "0.5", "0.3", "0.9")
    assert "do not fit apart in 0.20: each scaled" in out and "ball* names all 4 in one command" in out
    b = boxes(pic)
    xs = sorted((b[n] for n in ("ball", "ball2", "ball3", "ball4")), key=lambda v: v[0])
    for u, v in zip(xs, xs[1:]):
        assert v[0] - u[2] >= 0.15 * (u[2] - u[0]) - 0.004, (u, v)       # apart, never touching
    assert xs[0][0] >= 0.09 and xs[-1][2] <= 0.31


def test_scene_file_edits_are_revisions(pic):
    import shutil
    shutil.copy(os.path.join(ROOT, "scenes", "_template_chart.py"), pic / "chart.py")
    run("add", "base", "--scene", "chart.py")
    src = (pic / "chart.py").read_text(encoding="utf-8")
    (pic / "chart.py").write_text(src.replace('TITLE = "Solar now beats coal on cost"', 'TITLE = "Coal"'),
                                  encoding="utf-8")
    code, out = run("render")
    assert re.search(r"rev \d+: cells: changed base", out)
    run("undo")
    assert 'TITLE = "Solar now beats coal on cost"' in (pic / "chart.py").read_text(encoding="utf-8")


def test_crop_enlarges_sharply_and_keeps_the_title(pic):
    t0 = boxes(pic)["title"]
    b0 = boxes(pic)["ball"]
    code, out = run("crop", "0.2", "0.4", "0.6", "0.8")
    assert code == 0 and "2.50x, every layer drawn again" in out and "title keep their place" in out
    b = boxes(pic)
    assert abs((b["ball"][2] - b["ball"][0]) - 2.5 * (b0[2] - b0[0])) < 0.02
    assert all(abs(t - u) < 0.005 for t, u in zip(b["title"], t0))
    p = open_picture(str(pic / "art"), "demo")
    comp = Compositor(p.data, p.read_cells(), cache=Cache(p.cache_dir))
    assert comp.result("ball").k >= 2.5                                # drawn again at that size, not blown up
    run("move", "ball", "--by", "0.1", "0")
    assert abs(boxes(pic)["ball"][0] - b["ball"][0] - 0.1) < 0.01      # moves are in the cropped view
    run("crop", "off")
    assert abs((boxes(pic)["ball"][2] - boxes(pic)["ball"][0]) - (b0[2] - b0[0])) < 0.01


BAG = '''# %% room
cv.gradient([(0, "#cfd6dc"), (1, "#eef0f2")])
# %% handbag
# handbag: 1, red leather | at centre
sk.paint(sk.box(0.46, 0.55, 0.56, 0.68, 0.01), "#b02a2a", shade=0.4, highlight=0.4)
sk.paint(sk.stroke([(0.48, 0.55), (0.51, 0.49), (0.54, 0.55)], 0.008), "#6a1a1a")
# %% tag
# tag: 1, gold, hanging on the handbag's right side
x0, y0, x1, y1 = box("handbag")
sk.paint(sk.circle(x1 + 0.012, y0 + 0.03, 0.012), "#d4a640", shade=0.3)
'''


def test_the_suggested_crop_keeps_every_item_confirmed_and_inside(tmp_path, monkeypatch):
    """A small thing with no backdrop item gets a crop; following it keeps every item (the tag reads box("handbag")
    and stays beside it) confirmed and inside the frame."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(BAG, encoding="utf-8")
    code, out = run("new", "bag", "--prompt", "a photo of a handbag with a gold tag", "--size", "640x360",
                    "--from", "d.py")
    m = re.search(r"the requested things fill only \d+% of the frame: crop ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", out)
    assert m, out
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    assert abs((x1 - x0) - (y1 - y0)) < 0.015                            # the frame's shape (equal fractions)
    code, out = run("crop", *m.groups())
    assert code == 0 and "request: 2 of 2 confirmed" in out, out
    b = boxes_of(tmp_path, "bag")
    for n in ("handbag", "tag"):
        assert all(-0.001 <= v <= 1.001 for v in b[n]), (n, b[n])
    assert 0 <= b["tag"][0] - b["handbag"][2] < 0.06                     # still on the bag's right side
    assert "crop" not in out.split("worth checking")[-1].split("NEXT:")[0]


def test_the_fox_draft_gets_no_crop_and_the_lantern_scales_from_the_mouth(tmp_path, monkeypatch):
    """SKILL.md's draft: the wood carries an inventory line, so no crop is offered; scale on an anchored section
    pivots on the anchor point (the handle stays in the mouth)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    text = open(os.path.join(ROOT, "skills", "pixelcanvas", "SKILL.md"), encoding="utf-8").read()
    (tmp_path / "d.py").write_text(re.findall(r"```python\n(.*?)```", text, re.S)[0], encoding="utf-8")
    code, out = run("new", "fox", "--prompt", "a fox carrying a lantern through a snowy wood at night, titled "
                    "'Night Walk'", "--size", "640x360", "--out", "fox.png", "--from", "d.py")
    assert code == 0 and "request: 4 of 4 confirmed" in out, out
    assert "crop" not in out.split("look list")[-1].split("NEXT:")[0] and "fill only" not in out
    top = boxes_of(tmp_path, "fox")["lantern"][1]
    code, out = run("scale", "lantern", "2")
    assert code == 0 and "request: 4 of 4 confirmed" in out, out
    b = boxes_of(tmp_path, "fox")["lantern"]
    assert abs(b[1] - top) < 0.02 and b[3] <= 1.0, b


def test_several_layers_in_one_command(pic):
    run("copy", "ball", "--total", "3")
    code, out = run("recolor", "ball", "ball2", "box", "f5d142")
    assert code == 0 and all(layer(pic, n)["tint"] == "#f5d142" for n in ("ball", "ball2", "box"))
    code, out = run("set", "ball*", "outline=on")
    assert all(layer(pic, n)["outline"] for n in ("ball", "ball2", "ball3"))
    run("hide", "ball*")
    assert not any(layer(pic, n)["visible"] for n in ("ball", "ball2", "ball3"))
    run("show", "ball*")
    run("scale", "ball*", "0.5")
    assert all(layer(pic, n)["scale"] == 0.5 for n in ("ball", "ball2", "ball3"))
    code, out = run("move", "bal", "--by", "0.1", "0")                   # a typo is still a typo
    assert code == 2 and "did you mean 'ball'" in out


def test_batch_one_report_and_stops_at_a_failure(pic):
    code, out = run("batch", "scale ball 1.1", "studio move ball --by 0.05 0", "flip box")
    assert code == 0 and out.count("NEXT:") == 1 and out.count("request:") == 1
    assert "flip box" in out.splitlines()[0]
    code, out = run("batch", "scale ball 1.1", "recolor nothing gold", "flip box")
    assert code == 2 and "command 2 failed: recolor nothing gold" in out
    assert layer(pic, "box").get("flip") is True                       # the third never ran


def test_kit_call_without_cv_names_the_call(pic):
    (pic / "art" / "demo" / "layers" / "plant.py").write_text('# plant\nflora.potted("fern", at=(0.2, 0.9))\n',
                                                               encoding="utf-8")
    code, out = run("render")
    line = next(l for l in out.splitlines() if l.strip().startswith("plant"))
    assert "the first argument is cv: flora.potted(cv, ...)" in line and "-> studio api flora.potted" in line


def test_look_text_measures_the_recoloured_layer(pic):
    run("recolor", "ball", "f5d142")
    code, out = run("look", "--text", "ball")
    assert "(yellow)" in out or "(gold)" in out, out


def test_api_kit_methods():
    code, out = run("api", "chart.bar", "diagram.node")
    assert "chart.Axes.bar(categories" in out and "diagram.Diagram.node(id" in out
    code, out = run("api", "chart.bra")
    assert "did you mean chart.bar" in out


# ------------------------------------------------------------------------------------------- 3b: places on boxes, the picture file

def test_place_rules_on_boxes():
    from pixelcanvas.studio.check import _holds
    a, b = (0.1, 0.5, 0.3, 0.7), (0.5, 0.5, 0.7, 0.7)
    assert _holds("left", a, b, 1, 2, False) and not _holds("left", b, a, 1, 2, False)
    assert _holds("right", b, a, 1, 2, False) and not _holds("right", a, b, 1, 2, False)
    up, low = (0.4, 0.1, 0.6, 0.3), (0.4, 0.5, 0.6, 0.7)
    assert _holds("above", up, low, 1, 2, False) and not _holds("above", low, up, 1, 2, False)
    assert _holds("below", low, up, 1, 2, False) and not _holds("below", up, low, 1, 2, False)
    assert _holds("next", a, (0.32, 0.55, 0.5, 0.75), 1, 2, False)
    assert not _holds("next", a, (0.8, 0.55, 0.95, 0.75), 1, 2, False)          # too far
    desk = (0.0, 0.6, 1.0, 1.0)
    assert _holds("on", (0.4, 0.4, 0.5, 0.7), desk, 2, 1, True)                   # base inside the surface band
    assert not _holds("on", (0.4, 0.2, 0.5, 0.45), desk, 2, 1, True)              # floats above it
    crate = (0.4, 0.5, 0.6, 0.7)
    assert _holds("on", (0.45, 0.3, 0.55, 0.51), crate, 2, 1, False)              # on an object's top edge
    assert not _holds("on", (0.45, 0.6, 0.55, 0.8), crate, 2, 1, False)
    bowl = (0.4, 0.4, 0.6, 0.65)
    assert _holds("in", (0.45, 0.5, 0.52, 0.55), bowl, 2, 1, False)
    assert _holds("in", (0.43, 0.3, 0.5, 0.45), bowl, 2, 1, False)               # heaped above the rim
    assert not _holds("in", (0.62, 0.5, 0.8, 0.55), bowl, 2, 1, False)           # its centre beside the bowl
    assert not _holds("in", (0.45, 0.2, 0.5, 0.3), bowl, 2, 1, False)            # its base above the bowl
    assert _holds("behind", a, (0.2, 0.6, 0.4, 0.8), 1, 2, False) is True
    assert _holds("front", a, (0.2, 0.6, 0.4, 0.8), 1, 2, False) is False
    assert _holds("behind", a, b, 1, 2, False) is None                           # apart: left to the eye


def test_place_items_settle_from_boxes(pic):
    code, out = run("need", "add", "ball left of box")
    assert any("ball left of box" in ln for ln in out.splitlines() if "[x]" in ln), out    # measured, ticked
    nid = data(pic)["needs"][-1]["id"]
    code, out = run("move", "ball", "--to", "0.9", "0.6")
    line = next(l for l in out.splitlines() if l.startswith(f"  [ ] {nid} "))
    assert "ball is not left of box (solid parts: ball x" in line and line.endswith(": put ball left-of box")
    code, out = run("put", "ball", "left-of", "box")
    assert f"[ ] {nid} " not in out
    run("need", "add", "ball in box")
    code, out = run("render")
    assert "put ball in box" in out and "check cannot measure" not in out
    run("need", "add", "title behind moon")                                    # no moon layer yet
    code, out = run("render")
    assert "moon has no layer yet" in out


def test_the_picture_file_stays_current(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    code, out = run("new", "demo", "--prompt", PROMPT, "--size", "320x180", "--out", "out/final.png")
    target = (tmp_path / "out" / "final.png")
    assert f"picture file: {str(target.resolve()).replace(chr(92), '/')} (kept up to date after every command)" in out
    assert not target.exists()                                                  # nothing drawn yet
    (tmp_path / "draft1.py").write_text(DRAFT, encoding="utf-8")
    run("add", "--from", "draft1.py")
    assert size_of(target) == (320, 180)
    a = png(target, "RGB")
    run("recolor", "box", "gold")
    b = png(target, "RGB")
    assert np.abs(a.astype(int) - b.astype(int)).max() > 40                     # follows every edit
    (tmp_path / "art" / "demo" / "layers" / "ball.py").write_text("sk.paint(sk.circle(0.3, 0.6, 0.1), colr='red')\n",
                                                                  encoding="utf-8")
    code, out = run("render")
    assert "ERROR" in out and np.abs(png(target, "RGB").astype(int) - b.astype(int)).max() <= 2   # last good
    code, out = run("set", "scene", "out=other.png")
    assert (tmp_path / "other.png").is_file()                                   # the new path is written at once
    run("set", "scene", "out=out/final.png")
    code, out = run("deliver")
    assert code == 0 and "delivered: " in out and "final.png" in out


# ------------------------------------------------------------------------------------------- round 4: results that are right

SCALE_DRAFT = '''# %% bg
cv.gradient([(0, "#223"), (1, "#445")])
# %% camp
elements.campfire(cv, at=(0.3, 0.8), size=120, embers_n=0, smoke_=False)
# %% camps
elements.campfire(cv, at=(0.6, 0.8), size=int(120 * S), embers_n=0, smoke_=False)
# %% disc
sk.paint(sk.circle(0.85, 0.5, 0.06), "#d63c2f")
'''


def test_scale_grows_by_the_factor_also_for_pixel_sized_kits(tmp_path, monkeypatch):
    """A kit size given in pixels (size=int(225 * S): S applied twice) used to grow by factor x render factor."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    run("new", "s", "--prompt", "a campfire", "--size", "480x300")
    (tmp_path / "d.py").write_text(SCALE_DRAFT, encoding="utf-8")
    run("add", "--from", "d.py")

    def widths():
        p = open_picture(str(tmp_path / "art"), "s")
        sess = SC.Session(p)
        comp = sess.geometry()
        return {n: sess.box(comp, n)[2] - sess.box(comp, n)[0] for n in ("camp", "camps", "disc")}
    w0 = widths()
    for f in (1.25, 1.5, 2.0):
        code, out = run("set", "camp", "camps", "disc", f"scale={f}")
        w = widths()
        for n in w:
            assert abs(w[n] / w0[n] / f - 1) < 0.035, (f, n, w[n] / w0[n])
        assert "camps" in out and "drawn at 1x: its sizes grow" in out and "needs no * S" in out
    run("set", "camp", "camps", "disc", "scale=1")
    code, out = run("crop", "0.2", "0.4", "0.8", "0.9")                   # crop draws the cells again too
    k = float(re.search(r"([\d.]+)x, every layer drawn again", out).group(1))
    w = widths()
    assert all(abs(w[n] / w0[n] / k - 1) < 0.04 for n in w), (k, w, w0)



def test_a_frame_place_is_measured_and_moved_into_its_third(pic):
    code, out = run("need", "add", "ball at right")
    nid = data(pic)["needs"][-1]["id"]
    line = next(l for l in out.splitlines() if l.startswith(f"  [ ] {nid} "))
    assert "ball is not at the right (its centre x 0.38" in line and "the right third: x 0.67-1.00" in line
    cmd = line.split(": ")[-1]
    assert cmd.startswith("move ball --to 0.8")
    code, out = run(*cmd.split())
    assert f"[ ] {nid} " not in out
    code, out = run("need", "add", "box at bottom left")
    assert f"  [ ] {data(pic)['needs'][-1]['id']} box at bottom left -> box is not at the bottom left" in out


def test_put_several_in_a_container_heaps_them(pic):
    lay = pic / "art" / "demo" / "layers"
    (lay / "bowl.py").write_text("# bowl\nsk.paint(sk.box(0.42, 0.55, 0.58, 0.8, 0.02), '#eeeeee', name='bowl')\n",
                                 encoding="utf-8")
    (lay / "apple.py").write_text("# apple\nsk.paint(sk.circle(0.1, 0.2, 0.05), '#4caf50', name='apple')\n",
                                  encoding="utf-8")
    run("render")
    run("copy", "apple", "--total", "3")
    code, out = run("put", "apple*", "in", "bowl")
    assert code == 0, out
    b = boxes(pic)
    bowl = b["bowl"]
    apples = [b[n] for n in ("apple", "apple2", "apple3")]
    for a in apples:
        cx = (a[0] + a[2]) / 2
        assert bowl[0] < cx < bowl[2] and bowl[1] - 0.03 <= a[3] <= bowl[3] + 0.03, (a, bowl)
    bases = sorted(round(a[3], 2) for a in apples)
    assert bases[0] < bases[-1] - 0.03                                      # one rests on the others: a heap
    from pixelcanvas.studio.check import _holds
    assert all(_holds("in", a, bowl, 2, 1, False) for a in apples)


def test_no_float_hint_over_the_backdrop_or_against_the_request(pic):
    run("need", "add", "ball left of box")
    code, out = run("move", "box", "--to", "0.6", "0.4")                   # nothing but the sky under it
    assert "box may float" not in out
    p = open_picture(str(pic / "art"), "demo")
    sess = SC.Session(p)
    sess.render("x", quiet=True)
    assert SC._placed_beside(p, sess.comp, "ball", "box") and SC._placed_beside(p, sess.comp, "box", "ball")
    assert not SC._placed_beside(p, sess.comp, "ball", "title")


def test_off_frame_counts_the_solid_part(pic):
    (pic / "art" / "demo" / "layers" / "halo.py").write_text(
        "# halo\nsk.paint(sk.circle(0.9, 0.5, 0.3), '#ffffff', opacity=0.3)\n"
        "sk.paint(sk.circle(0.9, 0.5, 0.05), '#ffcc00')\n", encoding="utf-8")
    code, out = run("status", "--all")
    halo = next(l for l in out.splitlines() if l.strip().startswith("halo"))
    assert "off the frame" not in halo                                     # the faint halo leaves, the body stays
    run("move", "halo", "--by", "0.1", "0")
    code, out = run("status", "--all")
    halo = next(l for l in out.splitlines() if l.strip().startswith("halo"))
    assert "off the frame" in halo


def test_every_open_note_and_all_of_the_history(pic):
    run("note", "add", *[f"point {i}" for i in range(7)])
    code, out = run("status")
    assert all(f"( ) {i + 1} point {i}" in out for i in range(7)), out
    for _ in range(26):
        run("move", "ball", "--by", "0.001", "0")
    code, out = run("history")
    total = int(re.search(r"\(last 25 of (\d+); history --all\)", out).group(1))
    assert out.count(" rev ") == 25
    code, out = run("history", "--all")
    assert out.count(" rev ") == total and "last 25" not in out


def test_look_tiles_show_the_recolor_and_labels_have_a_backing(pic):
    run("recolor", "ball", "f5d142")
    p = open_picture(str(pic / "art"), "demo")
    sess = SC.Session(p)
    comp = sess.geometry()
    t = SC._tinted(comp, "ball", comp.place_layer("ball"))
    c = t.P[t.A > 0.5].mean(0) / np.maximum(t.A[t.A > 0.5].mean(), 1e-6)
    assert c[0] > 0.6 and c[1] > 0.5 and c[2] < 0.45, c                    # yellow, not the cell's red
    from PIL import Image as I, ImageDraw
    im = I.new("RGB", (80, 40), (255, 255, 255))
    SC._label(ImageDraw.Draw(im, "RGBA"), (5, 5), ".50", SC._font(14))
    assert np.asarray(im)[4:20, 3:30].min() < 100                           # dark under the yellow digits


def test_layers_file_keeps_objects_exact_and_the_grade_on_top(pic):
    import zipfile

    def ora(path):
        z = zipfile.ZipFile(path)
        stack = z.read("stack.xml").decode()
        src = dict(re.findall(r'name="([^"]+)"[^>]*src="([^"]+)"', stack))
        if not src:
            src = {n: s for s, n in re.findall(r'src="([^"]+)"[^>]*name="([^"]+)"', stack)}
        return z, stack, src
    run("deliver", "a.png", "--layers")
    run("set", "scene", "grade=golden")
    code, out = run("deliver", "b.png", "--layers")
    assert code == 0, out
    za, _, sa = ora(pic / "a.ora")
    zb, stack, sb = ora(pic / "b.ora")
    assert any(n.startswith("finish (grade golden") for n in sb), stack
    import io as _io
    for n in ("ball", "box"):
        a = np.asarray(Image.open(_io.BytesIO(za.read(sa[n]))).convert("RGBA"), np.int16)
        b = np.asarray(Image.open(_io.BytesIO(zb.read(sb[n]))).convert("RGBA"), np.int16)
        assert np.abs(a - b).max() <= 2, n                                   # no grade ghost in the object layers
    names = re.findall(r'name="([^"]+)"', stack)
    assert names[0].startswith("finish")                                      # stack.xml lists the top first


def test_error_inside_a_kit_names_the_outer_call_and_api_lists_none_options(pic):
    (pic / "art" / "demo" / "layers" / "box.py").write_text(
        '# box\nflora.tree(cv, "oak", at=(0.6, 0.9), height="tall")\n', encoding="utf-8")
    code, out = run("render")
    line = next(l for l in out.splitlines() if l.strip().startswith("box ") and "ERROR" in l)
    assert "-> studio api flora.tree" in line, line
    code, out = run("api", "flora.tree")
    assert "  haze=None: (colour, amount) atmospheric fade" in out


def test_a_note_about_empty_space_gets_a_crop(tmp_path, monkeypatch):
    """A backdrop with its own line gets no crop unasked; a note asking for less empty space gets one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(BAG.replace("# %% room\n", "# %% room\n# room: a pale grey wall\n"),
                                   encoding="utf-8")
    code, out = run("new", "bag", "--prompt", "a photo of a handbag with a gold tag in a pale room", "--size",
                    "640x360", "--from", "d.py")
    assert "fill only" not in out, out
    code, out = run("note", "add", "too much empty space around the things")
    nxt = _next(out)
    assert re.search(r"fix note \d+: too much empty space around the things - studio crop [\d.]+ [\d.]+ [\d.]+ "
                     r"[\d.]+ \(the region that holds the drawn things", nxt), nxt


def test_paths_the_agent_opens_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    code, out = run("new", "demo", "--prompt", PROMPT, "--size", "320x180")
    folder = str(tmp_path / "art" / "demo").replace(chr(92), "/")
    assert _next(out) == f"NEXT: write {folder}/draft.py, then: studio add --from {folder}/draft.py"
    (tmp_path / "art" / "demo" / "draft.py").write_text("# %% title\ncv.text('Hello', box=(0.2, 0.04, 0.8, 0.17))\n",
                                                        encoding="utf-8")
    code, out = run("add", "--from", "art/demo/draft.py")                  # only the title: nothing drawn yet
    assert code == 2 and out.strip() == (f"draft.py has no drawing code yet: write it to {folder}/draft.py (that "
                                         "exact path), then run this again")
    (tmp_path / "d.py").write_text(DRAFT.replace('"#d63c2f"', 'colr="red"'), encoding="utf-8")
    code, out = run("add", "--from", "d.py")
    d_ = str(tmp_path / "d.py").replace(chr(92), "/")
    assert out.splitlines()[0].endswith(f"look: {folder}/look.png (1280 px: picture + each item close up)")
    assert f"  ball: {d_} line 7 (section ball): " in out, out              # the line in the draft file
    assert _next(out).startswith(f'NEXT: edit section "# %% ball" in {d_} (line 7: ')
    assert f"then: studio add --from {d_}; the call's arguments: studio api sk.paint" in _next(out)


# ------------------------------------------------------------------------------------------- the reference page

def _blocks():
    with open(DOC, encoding="utf-8") as f:
        return re.findall(r"```python\n(.*?)```", f.read(), re.S)


@pytest.mark.parametrize("i", range(len(_blocks())))
def test_studio_md_blocks_run(i, tmp_path, monkeypatch):
    """Every python block of "studio.md" is a draft (or a cell) that renders without a layer error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    block = _blocks()[i]
    run("new", "doc", "--prompt", "a fox carrying a lantern through a snowy wood at night", "--size", "320x180")
    if "# %%" not in block:
        block = "# %% cell\n" + block
    if "anchor(\"fox\"" in block and "# %% fox" not in block:
        block = '# %% fox\nfox = creature(cv, "fox", at=(0.4, 0.9), size=0.4)\n' + block
    if "role: pass" in block or "[pass]" in block:
        block = "# %% base\ncv.gradient([(0, '#223'), (1, '#99a')])\n" + block
    (tmp_path / "block.py").write_text(block, encoding="utf-8")
    code, out = run("add", "--from", "block.py")
    assert code == 0 and "ERROR" not in out, out
    page = open(DOC, encoding="utf-8").read()
    assert len(page.splitlines()) <= 250


# ------------------------------------------------------------------------------------------- round 5: one script

SHARED = '''import math

def house(x, y, col):
    sk.paint(sk.box(x, y, x + 0.08, y + 0.1), col, volume=0.3)

# %% ground
# ground: green field
rnd = np.random.default_rng(3)
cols = {"wall": "#d63c2f", "roof": "#3a6fd8"}
t = tabletop(cv, top=0.55, front=0.98, horizon=0.4)

# %% house
# house: red
house(0.1, 0.5, cols["wall"])

# %% houses
# houses: 2
for i in range(2):
    house(0.4 + 0.25 * i, 0.45 + rnd.uniform(0, 0.05), cols["wall"])

# %% apple
# apple: red | on ground
ball(cv, t.place(0.3, 0.5), 0.04, base="#d63c2f")

# %% fox
# fox: orange
fox = creature(cv, "fox", at=(0.8, 0.95), size=0.3)
r = random.random()

# %% tag
# tag: yellow
x, y = fox.anchor("mouth") if hasattr(fox, "anchor") else (0.8, 0.8)
sk.paint(sk.circle(0.9, 0.2, 0.03 + 0.001 * math.floor(r * 3)), "#f5d142")
'''


def test_sections_share_names_like_one_script(tmp_path, monkeypatch):
    """A helper before the first section, a palette, a seeded generator, a tabletop and a creature from earlier
    sections: every later section sees them; each cell still caches on its own."""
    from pixelcanvas.studio.engine import scan, providers
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(SHARED, encoding="utf-8")
    code, out = run("new", "s", "--prompt", "a house in a field", "--size", "320x180", "--from", "d.py")
    assert code == 0 and "ERROR" not in out, out
    p = open_picture(str(tmp_path / "art"), "s")
    cells = p.read_cells()
    assert "def house" in cells["house"] and cells["house"].startswith("# house: red")    # the line stays first
    assert p.data["program"] == ["ground", "house", "houses", "apple", "fox", "tag"]
    order = p.data["program"]
    assert providers("houses", order, cells) == [("ground", ["cols", "rnd"])]
    assert providers("apple", order, cells) == [("ground", ["t"])]
    assert providers("tag", order, cells) == [("fox", ["fox", "r"])]
    assert "rnd" in scan(cells["ground"])[0] and "x" in scan(cells["tag"])[0]
    first = png(tmp_path / "art" / "s" / "preview.png")
    code, out = run("render", "--fresh")                                  # the same picture: seeds are fixed
    assert np.array_equal(first, png(tmp_path / "art" / "s" / "preview.png"))
    code, out = run("add", "--from", "d.py")                              # unchanged: all from the cache
    assert "cells run: none" in out
    (tmp_path / "d.py").write_text(SHARED.replace('"roof": "#3a6fd8"', '"roof": "#3a6fd9"'), encoding="utf-8")
    code, out = run("add", "--from", "d.py")
    ran = out.split("cells run:")[1].split(";")[0]
    assert "ground" in ran and "house" in ran and "houses" in ran and "apple" in ran and "fox" not in ran
    comp = Compositor(p.load().data, p.read_cells(), cache=Cache(p.cache_dir))
    comp.run()
    assert "mouth" not in comp.results["tag"].anchors                    # the fox's anchors stay the fox's
    assert "mouth" in comp.results["fox"].anchors


def test_a_failing_provider_and_a_name_from_nowhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(
        "# %% table\nt = tabletop(cv, top=0.55, front=0.95, horizon=0.4)\nboom = 1 / 0\n"
        "# %% cup\nball(cv, t.place(0.5, 0.5), 0.05)\n"
        "# %% star\nsk.paint(sk.circle(0.2, 0.2, 0.05), '#fff')\nstarr(1)\n"
        "# %% moon\nsk.paint(sk.circle(0.8, 0.2, 0.05), moon_col)\n"
        "# %% later\nmoon_col = '#ffe'\nsk.paint(sk.circle(0.5, 0.2, 0.02), '#fff')\n", encoding="utf-8")
    code, out = run("new", "e", "--prompt", "a cup on a table", "--size", "320x180", "--from", "d.py")
    lines = {l.split()[0]: l for l in out.splitlines() if l.startswith("  ") and "ERROR" in l}
    assert "needs `t` from section table, which fails: ZeroDivisionError" in lines["cup"], out
    assert "`starr` is not defined in this or an earlier section: sections run top to bottom and share names" \
        in lines["star"]
    assert "`moon_col` is defined in section later, which comes after this one" in lines["moon"]
    d_ = str(tmp_path / "d.py").replace(chr(92), "/")
    assert _next(out).startswith(f'NEXT: edit section "# %% table" in {d_} (line 3: ZeroDivisionError')


def test_inventory_lines_parse():
    from pixelcanvas.studio.state import inventory, thing_item
    bus = inventory("bus: 1, yellow, overturned on its side, wheels up to the sky, windows shattered | on road",
                    "bus")
    assert bus["count"] == 1 and bus["colours"] == ["yellow"] and bus["places"] == [["on", "road"]]
    assert bus["states"] == ["overturned on its side", "wheels up to the sky", "windows shattered"]
    tape = inventory("caution tape: yellow, cordons off the scene | in front of bus", "tape")
    assert tape["name"] == "caution tape" and tape["places"] == [["in front of", "bus"]]
    apples = inventory("apple: three, dark green, one bitten | in bowl, at left", "apple")
    assert apples["count"] == 3 and apples["colours"] == ["dark green"] and apples["states"] == ["one bitten"]
    assert apples["places"] == [["in", "bowl"], ["at", "left"]]
    sign = inventory("sign: wooden, says 'OPEN', left of door", "sign")
    assert sign["texts"] == ["OPEN"] and sign["states"] == ["wooden"] and sign["places"] == [["left of", "door"]]
    shoes = inventory("shoes: a pair of, red and white, #3a6fd8 laces", "shoes")
    assert shoes["count"] == 2 and shoes["colours"] == ["red", "white"] and shoes["states"] == ["#3a6fd8 laces"]
    assert inventory("lantern: glowing warm yellow, its handle in the fox's mouth", "lantern")["states"] == [
        "glowing warm yellow", "its handle in the fox's mouth"]
    assert inventory("night gradient with a moon", "sky") is None                  # a plain comment
    assert inventory("3, green | at top left", "apple")["places"] == [["at", "top left"]]
    t = thing_item("sled_dog", None)
    assert t["key"] == "sled dog" and t["text"] == "sled dog" and t["cell"] == "sled_dog"


INVENTORY = '''# %% road
# road: grey asphalt
cv.gradient([(0, "#9ab"), (0.5, "#cde"), (0.51, "#555"), (1, "#333")])

# %% bus
# bus: 1, yellow, overturned on its side | on road
sk.paint(sk.box(0.25, 0.45, 0.7, 0.75, 0.02), "#f5d142", volume=0.3)

# %% sign
# sign: says 'OPEN' | right of bus
sk.paint(sk.box(0.78, 0.5, 0.95, 0.62), "#ffffff")
cv.text("OPEN", box=(0.79, 0.51, 0.94, 0.61), color="#d63c2f")

# %% finish [pass]
grade(paper(cv), "warm night", 0.3)
'''


def test_the_checklist_comes_from_the_draft(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "art" / "b").mkdir(parents=True)
    (tmp_path / "art" / "b" / "draft.py").write_text(INVENTORY, encoding="utf-8")    # the draft before `new`
    code, out = run("new", "b", "--prompt", "an overturned bus by a sign that says 'OPEN', and a 'Closed' stamp",
                    "--size", "320x180", "--from", "art/b/draft.py")
    assert code == 0, out
    d = data_of(tmp_path, "b")
    assert [(n["id"], n["text"]) for n in d["needs"]] == [
        (1, "road: grey asphalt"), (2, "bus: 1, yellow, overturned on its side | on road"),
        (3, "sign: says 'OPEN' | right of bus"), (4, "text 'Closed'")]              # 'OPEN' is one item, the pass none
    assert "[x] 3 sign: says 'OPEN' | right of bus" in out and "[ ] 4 text 'Closed'" in out
    run("need", "ok", "2", "--why", "seen")
    run("need", "add", "the bus looks heavy")
    src = INVENTORY.replace("# road: grey asphalt", "# road: grey asphalt, wet")
    (tmp_path / "art" / "b" / "draft.py").write_text(src, encoding="utf-8")
    code, out = run("add", "--from", "art/b/draft.py")
    needs = {n["id"]: n for n in data_of(tmp_path, "b")["needs"]}
    assert needs[1]["text"] == "road: grey asphalt, wet" and needs[2]["ok"] is True       # rebuilt / kept
    assert needs[5]["text"] == "the bus looks heavy" and needs[5]["src"] == "agent"
    (tmp_path / "art" / "b" / "draft.py").write_text(src.replace("| right of bus", "| left of bus"),
                                                     encoding="utf-8")
    code, out = run("add", "--from", "art/b/draft.py")
    line = next(l for l in out.splitlines() if l.startswith("  [ ] 3 "))
    assert "sign is not left of bus (solid parts:" in line and line.endswith(": put sign left-of bus"), line
    code, out = run("status")
    assert out.splitlines()[1].startswith('the request: "an overturned bus')


def test_new_without_a_draft_file_and_twice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    code, out = run("new", "x", "--prompt", "a red ball", "--from", "nothere.py")
    assert code == 2 and "no file nothere.py" in out and not (tmp_path / "art" / "x").exists()
    (tmp_path / "d.py").write_text("# %% ball\n# ball: red\nsk.paint(sk.circle(0.5, 0.5, 0.1), '#d63c2f')\n",
                                   encoding="utf-8")
    code, out = run("new", "x", "--prompt", "a red ball", "--size", "320x180", "--from", "d.py")
    assert code == 0 and "1 of 1 confirmed" in out
    code, out = run("new", "x", "--prompt", "a red ball", "--size", "320x180", "--from", "d.py")
    assert code == 0 and "picture x exists: the draft goes into it" in out and "cells run: none" in out


def test_a_note_starting_with_an_item_number_is_tied_to_it(pic):
    code, out = run("render")
    assert "look 1 of 3" in _next(out)
    code, out = run("note", "add", "3 the box does not read as a box")
    n = data(pic)["notes"][-1]
    assert n["item"] == 3 and n["layer"] == "box"
    assert "[ ] 3 box: blue -> note 1: 3 the box does not read as a box: note 1 is open" in out, out
    assert _next(out).startswith("NEXT: fix note 1: 3 the box does not read as a box - studio look box")
    code, out = run("note", "done", "1")
    assert "[ ] 3" not in out and "look 1 of 3" in _next(out)


def test_touching_copies_and_a_hidden_thing_are_open(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(
        "# %% wall\n# wall\ncv.gradient([(0, '#ddd'), (1, '#bbb')])\n"
        "# %% sink\n# sink: 3, white\nsk.paint(sk.box(0.2, 0.5, 0.32, 0.7, 0.01), '#f4f4f4', volume=0.3)\n"
        "# %% shard\n# shard: light blue\nsk.paint(sk.circle(0.8, 0.85, 0.03), '#9fd0ff')\n"
        "# %% tape\n# tape: yellow\nsk.paint(sk.box(0.7, 0.78, 0.9, 0.92), '#f5d142')\n", encoding="utf-8")
    code, out = run("new", "k", "--prompt", "three sinks", "--size", "320x180", "--from", "d.py")
    assert "[ ] 1 sink: 3, white -> 1 drawn, the line says 3: copy sink --total 3 --row " in out, out
    assert "[ ] 2 shard: light blue -> shard is 0% visible: covered by tape: order shard above tape" in out
    code, out = run("copy", "sink", "--total", "3", "--by", "0.08", "0")              # touching
    line = next(l for l in out.splitlines() if l.startswith("  [ ] 1 "))
    assert "-> sink, sink2, sink3 touch: a viewer counts fewer: copy sink --total 3 --row " in line, line
    cmd = line.split(": ")[-1]
    code, out = run(*cmd.split())
    assert "[ ] 1 " not in out, out
    b = boxes_of(tmp_path, "k")
    xs = sorted((b[n] for n in ("sink", "sink2", "sink3")), key=lambda v: v[0])
    assert all(v[0] - u[2] >= 0.15 * (u[2] - u[0]) - 0.004 for u, v in zip(xs, xs[1:]))


def boxes_of(root, name):
    p = open_picture(str(root / "art"), name)
    sess = SC.Session(p)
    comp = sess.geometry()
    return {L["name"]: sess.box(comp, L["name"]) for L in p.data["layers"] if comp.role(L) != "pass"
            and comp.place_layer(L["name"]) is not None and comp.place_layer(L["name"]).box is not None}


def test_a_small_thing_and_an_outline_on_a_photo_are_worth_checking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(
        "# %% table\ncv.gradient([(0, '#ddd'), (1, '#999')])\n"
        "# %% knife\n# knife: silver\nsk.paint(sk.box(0.3, 0.5, 0.7, 0.56), '#c0c4c8', line=True)\n"
        "# %% crumb\n# crumb: brown\nsk.paint(sk.circle(0.2, 0.8, 0.01), '#8b5a2b')\n", encoding="utf-8")
    code, out = run("new", "k", "--prompt", "a photo of a knife", "--size", "320x180", "--from", "d.py")
    assert ("worth checking: the request is a photo: knife is drawn with line=True (an outline reads as a "
            "drawing): remove it") in out
    assert re.search(r"crumb covers 0\.\d% of the frame: scale crumb [\d.]+\n", out), out


def test_undo_is_offered_when_a_command_covers_a_requested_thing(pic):
    code, out = run("scale", "box", "3")
    assert "this command (scale box 3) covered ball" in out or "this command (scale box 3) pushed" in out, out
    assert _next(out).startswith("NEXT: studio undo - this command (scale box 3)")
    code, out = run("undo")
    assert "this command" not in out


def test_place_comes_before_order_for_one_layer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(
        "# %% bg\ncv.gradient([(0, '#ddd'), (1, '#999')])\n"
        "# %% bottle\n# bottle: green | at left\nsk.paint(sk.box(0.45, 0.4, 0.55, 0.8), '#4caf50')\n"
        "# %% bowl\n# bowl: white\nsk.paint(sk.box(0.35, 0.5, 0.65, 0.8), '#f4f4f4')\n", encoding="utf-8")
    code, out = run("new", "o", "--prompt", "a bottle at left", "--size", "320x180", "--from", "d.py")
    line = next(l for l in out.splitlines() if l.startswith("  [ ] 1 "))
    assert "bottle is not at the left" in line and ": move bottle --to 0.17" in line, line


def test_multi_word_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    secs = SC.split_draft("# %% sled dog\nx = 1\n# %% Snow-Field [backdrop]\ny = 2\n# %% fox: the hero\nz = 3\n")
    assert [(n, r) for n, r, _ in secs] == [("sled_dog", None), ("Snow_Field", "backdrop"), ("fox", None)]
    with pytest.raises(SC.StudioError, match=r"draft line '# %% \(\)'"):
        SC.split_draft("# %% a\nx = 1\n# %% ()\ny = 2\n")
    (tmp_path / "d.py").write_text("# %% snow\ncv.gradient([(0, '#dde'), (1, '#fff')])\n"
                                   "# %% sled dog\nsk.paint(sk.circle(0.4, 0.7, 0.08), '#8b5a2b')\n", encoding="utf-8")
    code, out = run("new", "m", "--prompt", "a sled dog", "--size", "320x180", "--from", "d.py")
    assert "[x] 1 sled dog" in out and "sled_dog" in data_of(tmp_path, "m")["program"], out
    code, out = run("move", "sled dog", "--by", "0.1", "0")
    assert code == 0 and "move sled_dog" in out


def test_put_keeps_the_layer_inside_the_frame(pic):
    run("move", "box", "--to", "0.9", "0.6")
    code, out = run("put", "ball", "right-of", "box")
    assert "put: no room right of box inside the frame:" in out, out
    b = boxes(pic)["ball"]
    assert b[0] >= 0 and b[2] <= 1.0


def test_a_report_never_crashes(pic, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad report")
    monkeypatch.setattr(SC, "print_report", boom)
    code, out = run("move", "ball", "--by", "0.01", "0")
    assert code == 1 and "report failed: RuntimeError: bad report" in out and _next(out) == "NEXT: studio status"
    assert "rendered rev" in out and "preview.png" in out


def test_a_resubmitted_draft_keeps_moves_removes_and_says_which_file_wins(pic):
    run("scale", "ball", "0.5")
    src = DRAFT.replace('# %% box\n# box: blue\nsk.paint(sk.box(0.5, 0.45, 0.7, 0.75, 0.02), "#3a6fd8", shade=0.3, '
                        'name="box")\n\n', "")
    src = src.replace("# %% title", "# %% cat\n# cat: grey\nsk.paint(sk.circle(0.8, 0.7, 0.05), '#8a8a8a')\n\n"
                                    "# %% title")
    (pic / "draft1.py").write_text(src, encoding="utf-8")
    code, out = run("add", "--from", "draft1.py")
    assert "removed: box (no longer in the draft)" in out and layer(pic, "ball")["scale"] == 0.5
    names = [L["name"] for L in data(pic)["layers"]]
    assert names == ["sky", "ball", "cat", "title"], names                      # the new section where the draft has it
    (pic / "art" / "demo" / "layers" / "ball.py").write_text("# ball: red\nsk.paint(sk.circle(0.3, 0.6, 0.1), "
                                                             "'#d63c2f')\n", encoding="utf-8")
    code, out = run("add", "--from", "draft1.py")
    assert "ball: layers/ball.py was edited after the draft; it is kept (its draft section is unchanged)" in out
    (pic / "draft1.py").write_text(src.replace("0.38, 0.62, 0.12", "0.38, 0.62, 0.1"), encoding="utf-8")
    code, out = run("add", "--from", "draft1.py")
    assert "ball: layers/ball.py was edited after the draft; the draft section replaces it" in out


def test_deliver_names_the_file_to_open(pic):
    code, out = run("deliver", "out/final.png")
    path = str((pic / "out" / "final.png").resolve()).replace(chr(92), "/")
    assert out.strip().splitlines()[-1] == f"NEXT: open {path} (this is what the user gets), then tell the user that path"


def test_a_pass_section_receives_the_composite_and_gives_no_item(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(INVENTORY, encoding="utf-8")
    code, out = run("new", "f", "--prompt", "a bus", "--size", "320x180", "--from", "d.py")
    assert code == 0 and "ERROR" not in out and "finish" in out
    assert not any(n.get("cell") == "finish" for n in data_of(tmp_path, "f")["needs"])
    assert layer_of(tmp_path, "f", "finish")["role"] == "pass"


def layer_of(root, pic_name, name):
    return next(L for L in data_of(root, pic_name)["layers"] if L["name"] == name)


def test_the_look_sheet_follows_every_change(pic):
    look = pic / "art" / "demo" / "look.png"
    look.unlink()
    code, out = run("scale", "ball", "1.2")
    path = str(look).replace(chr(92), "/")
    assert look.is_file() and size_of(look)[0] == 1280
    assert out.splitlines()[0] == f"rev {data(pic)['rev']}: scale ball 1.2    look: {path} (1280 px: picture + each item close up)"


def test_finished_names_the_picture_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "draft1.py").write_text(DRAFT, encoding="utf-8")
    code, out = run("new", "demo", "--prompt", PROMPT, "--size", "320x180", "--out", "out/final.png", "--from",
                    "draft1.py")
    target = str((tmp_path / "out" / "final.png").resolve()).replace(chr(92), "/")
    assert _next(out).startswith("NEXT: look 1 of 3: ") and _next(out).endswith(f"finished - the picture is {target}")
    assert f"picture file: {target} (320x180, written now)" in out
    run("move", "ball", "--by", "0.01", "0")
    code, out = run("move", "ball", "--by", "0.01", "0")
    assert f"picture file: {target} (320x180, written now)" in out
    code, out = run("render")
    assert "picture file:" not in out                            # nothing changed: nothing written to say
    assert _next(out) == (f"NEXT: finished: the picture is {target} (320x180, up to date; open it before you tell "
                          "the user)")
    assert "deliver" not in out


BLEND = '''# %% table
cv.gradient([(0, "#e6e0d4"), (1, "#ddd6c8")])
# %% spoon
# spoon: pale grey
sk.paint(sk.ellipse(0.4, 0.5, 0.06, 0.035), "#d8d5ce", volume=0.3)
sk.paint(sk.stroke([(0.45, 0.5), (0.7, 0.55)], 0.02), "#d8d5ce")
# %% wall2
# wall2
sk.paint(sk.box(0.75, 0.1, 1.0, 0.9), "#d7e0e3")
# %% sink
# sink: white
sk.paint(sk.box(0.8, 0.4, 0.95, 0.5, 0.01), "#f4f4f4", volume=0.4)
sk.paint(sk.box(0.86, 0.5, 0.89, 0.8), "#e8eef0", volume=0.4)
'''


def test_a_thing_that_blends_into_what_is_behind_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(BLEND, encoding="utf-8")
    code, out = run("new", "b", "--prompt", "a photo of a spoon and a sink", "--size", "640x360", "--from", "d.py")
    assert re.search(r"spoon blends into the table behind it \(contrast 1\.\d:1\): recolor spoon [0-9a-f]{6}\b",
                     out), out
    assert "or set spoon shadow" not in out
    assert "sink blends" not in out and "outline" not in out.split("spoon blends")[1].split("\n")[0]


TABLE = '''# %% table
# table: pale wooden table top
cv.gradient([(0, "#d9d4cb"), (1, "#e8e2d6")])
# %% handbag
# handbag: 1, brown | at left
sk.paint(sk.box(0.2, 0.62, 0.27, 0.7, 0.01), "#7a4a2a", shade=0.4)
sk.paint(sk.stroke([(0.215, 0.62), (0.235, 0.59), (0.255, 0.62)], 0.006), "#5a3520")
# %% apples
# apples: 2, red | at centre
for x in (0.47, 0.515):
    sk.paint(sk.circle(x, 0.66, 0.045), "#c0282a", shade=0.4)
# %% spoon
# spoon: 1, grey | at right
sk.paint(sk.stroke([(0.7, 0.7), (0.86, 0.62)], 0.012), "#8a8478", shade=0.2)
sk.paint(sk.ellipse(0.88, 0.61, 0.03, 0.018), "#8a8478", shade=0.2)
'''


def test_copies_drawn_in_one_loop_that_touch_get_an_edit_not_a_copy(tmp_path, monkeypatch):
    """Two apples drawn in one loop that run together read as one: copying the layer would make four."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(TABLE, encoding="utf-8")
    code, out = run("new", "t", "--prompt", "a photo of a handbag, two red apples and a spoon on a pale table",
                    "--size", "640x360", "--from", "d.py")
    line = next(ln for ln in out.splitlines() if "3 apples:" in ln)
    assert "copies run together" in line and "gap between them" in line and "copy apples" not in line, line


def test_scale_suggestions_never_grow_into_a_neighbour(tmp_path, monkeypatch):
    """Every 'scale' a report suggests, followed in turn, keeps the things apart and inside the frame."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(TABLE.replace("(0.47, 0.515)", "(0.45, 0.53)"), encoding="utf-8")
    code, out = run("new", "t", "--prompt", "a photo of a handbag, two red apples and a spoon on a pale table",
                    "--size", "640x360", "--from", "d.py")
    for _ in range(4):
        m = re.search(r"worth checking: \w+ covers [\d.]+% of the frame: (scale \S+ [\d.]+)", out)
        if not m:
            break
        code, out = run(*m.group(1).split())
        assert code == 0 and "request: 4 of 4 confirmed" in out, out
    b = boxes_of(tmp_path, "t")
    xs = sorted((b[n] for n in ("handbag", "apples", "spoon")), key=lambda v: v[0])
    assert all(u[2] <= v[0] for u, v in zip(xs, xs[1:])), xs
    assert all(0 <= v[0] and v[2] <= 1 for v in xs)


def test_copies_packed_too_close_to_grow_are_spread_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(TABLE.replace("for x in (0.47, 0.515):\n    sk.paint(sk.circle(x,",
                                                 "sk.paint(sk.circle(0.47,"), encoding="utf-8")
    run("new", "t", "--prompt", "a photo of a handbag, two red apples and a spoon on a pale table", "--size",
        "640x360", "--from", "d.py")
    code, out = run("copy", "apples", "--total", "2", "--row", "0.40", "0.62", "0.53", "0.71")
    m = re.search(r"copies stand too close to grow: (copy apples --total 2 --row [\d. ]+)\n", out)
    assert m, out
    code, out = run(*m.group(1).split())
    m = re.search(r"apples covers [\d.]+% of the frame: (scale apples\* [\d.]+)", out)
    assert m, out
    code, out = run(*m.group(1).split())
    assert "request: 4 of 4 confirmed" in out and "covered" not in out, out
    b = boxes_of(tmp_path, "t")
    assert b["apples"][2] < b["apples2"][0] and b["apples2"][2] < b["spoon"][0]


def test_undo_names_the_item_the_edit_lost(tmp_path, monkeypatch):
    """With an item open all along, hiding another names the hidden one, not the open one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(TABLE, encoding="utf-8")                # apples run together: open
    run("new", "t", "--prompt", "a photo of a handbag, two red apples and a spoon on a pale table", "--size",
        "640x360", "--from", "d.py")
    code, out = run("hide", "spoon")
    assert _next(out).startswith("NEXT: this edit lost a confirmed item (spoon: 1, grey"), _next(out)


def test_a_long_look_list_puts_doubts_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    secs = ["# %% bg\ncv.gradient([(0, '#223'), (1, '#556')])\n"]
    for i in range(14):
        r = 0.01 if i == 13 else 0.1
        secs.append(f"# %% dot{chr(97 + i)}\n# dot{chr(97 + i)}: red\n"
                    f"sk.paint(sk.circle({0.08 + (i % 7) * 0.13:.3f}, {0.27 if i < 7 else 0.73}, {r}), '#d63c2f')\n")
    (tmp_path / "d.py").write_text("".join(secs), encoding="utf-8")
    code, out = run("new", "l", "--prompt", "fourteen dots", "--size", "640x360", "--from", "d.py")
    block = out.split("look list (")[1].split("NEXT:")[0].splitlines()[1:]
    assert block[0].startswith("  [x] 14 dotn: red  <- worth checking: dotn covers"), block
    assert "  ... +2 more lines (studio need)" in block


# ------------------------------------------------------------------------------------- round 5d: places, glows, set

def _draft_pic(tmp_path, monkeypatch, name, prompt, draft, size="480x300"):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    (tmp_path / "d.py").write_text(draft, encoding="utf-8")
    code, out = run("new", name, "--prompt", prompt, "--size", size, "--from", "d.py")
    assert code == 0, out
    return out


def _geom(tmp_path, name):
    from pixelcanvas.studio import check as CK
    p = open_picture(str(tmp_path / "art"), name)
    sess = SC.Session(p)
    return p, sess.geometry(), CK


def test_put_never_shrinks_a_thing_to_meet_a_box(tmp_path, monkeypatch):
    """No room above a sign at the top edge: put changes nothing and names the way out."""
    _draft_pic(tmp_path, monkeypatch, "p", "a red balloon above a sign", '''# %% sky
cv.gradient([(0, "#9ec5e8"), (1, "#e8f1f8")])
# %% sign
# sign: white
sk.paint(sk.box(0.4, 0.04, 0.6, 0.2, 0.01), "#ffffff", shade=0.2)
# %% balloon
# balloon: 1, red
sk.paint(sk.circle(0.5, 0.65, 0.2), "#d63c2f", shade=0.3)
''')
    before = boxes_of(tmp_path, "p")["balloon"]
    code, out = run("put", "balloon", "above", "sign")
    assert code == 0 and "nothing changed" in out and "move balloon --to" in out, out
    assert "scaled" not in out
    assert float(layer_of(tmp_path, "p", "balloon").get("scale", 1.0)) == 1.0
    assert all(abs(u - v) < 1e-6 for u, v in zip(boxes_of(tmp_path, "p")["balloon"], before))


def layer_of(root, name, lay):
    d = json.loads((root / "art" / name / "picture.json").read_text(encoding="utf-8"))
    return next(L for L in d["layers"] if L["name"] == lay)


CAMP = '''# %% ground
# ground: snow at night
cv.gradient([(0, "#0b1030"), (0.6, "#24305e"), (0.62, "#8d9cc0"), (1, "#c6d0e6")])
# %% campfire
# campfire: orange | next to log
elements.campfire(cv, at=(0.4, 0.82), size=150, embers_n=40)
# %% log
# log: 1, brown | next to campfire
sk.paint(sk.box(0.55, 0.76, 0.7, 0.84, 0.01), "#6a4428", shade=0.4)
'''


def test_places_are_judged_on_the_solid_part_not_glow_or_embers(tmp_path, monkeypatch):
    out = _draft_pic(tmp_path, monkeypatch, "c", "a campfire next to a log", CAMP)
    p, comp, CK = _geom(tmp_path, "c")
    full, sol = CK._frac_box(comp, "campfire"), CK.solid_box(comp, "campfire")
    assert sol[3] - sol[1] < 0.7 * (full[3] - full[1])                           # embers and glow left out
    assert "request: 3 of 3 confirmed" in out, out
    for cmd in (("scale", "log", "1.2"), ("put", "campfire", "next-to", "log")):
        code, out = run(*cmd)
        assert "request: 3 of 3 confirmed" in out, out
    assert "already next to log; nothing changed" in out


def test_a_moon_above_the_trees_is_judged_against_the_forest_mass(tmp_path, monkeypatch):
    out = _draft_pic(tmp_path, monkeypatch, "f", "a full moon above a pine forest", '''# %% sky
cv.gradient([(0, "#070b22"), (1, "#2c3a6e")])
# %% forest
# pine forest: firs on both sides
for x, h in ((0.03, 0.62), (0.13, 0.5), (0.3, 0.3), (0.5, 0.25), (0.7, 0.3), (0.88, 0.55), (0.98, 0.66)):
    flora.tree(cv, "fir", at=(x, 0.8), height=h, seed=int(x * 100))
# %% moon
# moon: full, bright | above the trees
elements.moon(cv, at=(0.5, 0.3), size=60, glow=0.4)
''')
    p, comp, CK = _geom(tmp_path, "f")
    assert CK.target_layers(p, comp, "the trees") == ["forest"]
    assert CK._frac_box(comp, "moon")[3] > CK._frac_box(comp, "forest")[1]          # inside the forest's full box
    assert "request: 2 of 2 confirmed" in out, out
    code, out = run("put", "moon", "above", "forest")
    assert "nothing changed" in out and "scaled" not in out, out


def test_a_thing_on_a_table_with_a_ground_shadow_stays_on_it_when_scaled(tmp_path, monkeypatch):
    _draft_pic(tmp_path, monkeypatch, "t", "a red ball on a table", '''# %% wall
cv.gradient([(0, "#d8d2c4"), (1, "#c8c0b0")])
# %% table
# table: brown wood
sk.paint(sk.box(0.1, 0.6, 0.9, 0.7, 0.01), "#7a5232", shade=0.3)
# %% ball
# ball: 1, red | on table
sk.paint(sk.circle(0.5, 0.55, 0.06), "#d63c2f", volume=0.5, shadow="ground")
''')
    for cmd in (("scale", "ball", "1.3"), ("move", "ball", "--by", "0.05", "0"), ("set", "ball", "glow=ffcf70")):
        code, out = run(*cmd)
        line = next(ln for ln in out.splitlines() if "ball: 1, red | on table" in ln)
        assert line.lstrip().startswith("[x]"), out


def test_set_takes_a_value_of_several_words(tmp_path, monkeypatch):
    _draft_pic(tmp_path, monkeypatch, "s", "a red ball on a table", '''# %% wall
cv.gradient([(0, "#d8d2c4"), (1, "#c8c0b0")])
# %% ball
# ball: 1, red
sk.paint(sk.circle(0.5, 0.55, 0.1), "#d63c2f", volume=0.5)
''')
    code, out = run("set", "scene", "grade=warm", "night", "grade_amount=0.5")
    assert code == 0, out
    d = json.loads((tmp_path / "art" / "s" / "picture.json").read_text(encoding="utf-8"))
    assert d["scene"]["grade"] == "warm night" and float(d["scene"]["grade_amount"]) == 0.5


def test_a_kit_glow_fades_out_with_no_edge_as_a_layer(tmp_path, monkeypatch):
    """The campfire as a studio layer matches the plain render outside the hot core: no box around its light."""
    from pixelcanvas.canvas import Canvas
    from pixelcanvas import elements
    W, H = 480, 300
    cv = Canvas(W, H)
    cv.gradient([(0, "#2c3a6e"), (1, "#2c3a6e")])
    elements.campfire(cv, at=(0.5, 0.8), size=int(190 * H / 1080), embers_n=30)
    plain = np.clip(cv.buf, 0, 1)
    _draft_pic(tmp_path, monkeypatch, "g", "a campfire", '# %% bg\ncv.gradient([(0, "#2c3a6e"), (1, "#2c3a6e")])\n'
               '# %% campfire\n# campfire: orange\nelements.campfire(cv, at=(0.5, 0.8), size=int(190 * S), '
               'embers_n=30)\n', size=f"{W}x{H}")
    st = png(tmp_path / "art" / "g" / "preview.png", "RGB").astype(np.float32) / 255
    diff = np.abs(st - plain).max(-1)
    assert diff[:int(0.7 * H)].max() < 0.03, diff[:int(0.7 * H)].max()           # embers and glow: as drawn
    assert (diff > 0.03).mean() < 0.005                                          # only the clipped hot core


# ------------------------------------------------------------------------------------- round 5e: turns saved

def test_every_wrong_option_of_a_section_is_reported_at_once_with_the_call(tmp_path, monkeypatch):
    out = _draft_pic(tmp_path, monkeypatch, "o", "a moon", '''# %% sky
cv.gradient([(0, "#070b22"), (1, "3a4048")])
# %% moon
# moon: full
elements.moon(cv, at=(0.5, 0.3), size=60, phse=0.5, glw=0.4, name="moon")
sk.paint(sk.circle(0.2, 0.7, 0.05), "ffffff", shde=0.3)
''')
    line = next(ln for ln in out.splitlines() if ln.strip().startswith("moon ") and "ERROR" in ln)
    assert "phse= (did you mean phase=?)" in line and "glw= (did you mean glow=?)" in line
    assert "shde= (did you mean shade=?)" in line and "-> studio api elements.moon" in line
    assert "    the call: elements.moon(cv, at=(0.75, 0.22), size=60, phase=0.5" in out
    assert "    the call: sk.paint(mask, color, " in out


def test_bare_hex_colours_name_and_the_taught_names_work_in_a_section(tmp_path, monkeypatch):
    out = _draft_pic(tmp_path, monkeypatch, "h", "a blue cup and a red disc", '''# %% wall
cv.gradient([(0, "d8d2c4"), (1, "c8c0b0")])
# %% cup
# cup: blue
set_light(direction=(-0.8, -0.5), elevation=32)
cylinder(cv, (0.4, 0.8), 0.08, 0.2, base="3a6fd8", handle=True, name="cup")
# %% disc
# disc: red
fr = Frame(cv, (0.75, 0.5), 0.12)
fr.paint("d63c2f", sd_circle(fr.X, fr.Y, 0, 0, 0.6))
contact_shadow(cv, (0.75, 0.62), 0.1)
# %% finish
# a remark, no code yet
''')
    assert "ERROR" not in out, out
    assert "request: 2 of 2 confirmed" in out, out
    assert "  finish: no code, skipped" in out and "finish has no code" not in out


def test_a_layer_that_draws_nothing_gets_an_edit_not_a_look(tmp_path, monkeypatch):
    out = _draft_pic(tmp_path, monkeypatch, "e", "smoke over a roof", '''# %% sky
cv.gradient([(0, "#9ec5e8"), (1, "#e8f1f8")])
# %% smoke
# smoke: grey
sk.paint(sk.circle(0.5, 0.3, 0.1), "#888888", opacity=0.0)
''')
    assert "studio look smoke" not in out and "look smoke" not in _next(out), out
    assert "smoke" in _next(out) and "edit section" in _next(out), _next(out)


def test_crop_says_what_it_applied(pic):
    code, out = run("crop", "0.02", "0.35", "0.98", "0.92")
    assert "crop: widened to the frame's shape: 0.02 0.04 0.98 1.00 (asked 0.02 0.35 0.98 0.92)" in out, out
    m = re.search(r"changes almost nothing; one that does: crop ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", out)
    assert m, out
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    assert abs((x1 - x0) - 0.57) < 0.011 and abs((y0 + y1) / 2 - 0.635) < 0.011


S3D = '''# %% scene
# car and ball: a red car and a chrome ball on a checker floor
from pixelcanvas.solid import Scene, Camera, car, sphere
sc = Scene(lighting="studio", ground="checker")
sc.add(car("#e8452c").rotate_y(-30), "car")
sc.add(sphere(0.6).place(x=3.4, z=-0.6), "ball", "chrome")
sc.camera = Camera.frame(sc, azimuth=30, elevation=15)
sc.add(sphere(0.6).place(x=X, z=-0.6), "buoy", "chrome")
res = sc.render(cv)
'''


@pytest.mark.parametrize("x,said", [(5, "buoy (a 3D object of scene) is cut by the frame"),
                                    (9, "buoy (3D object of scene) is outside the frame")])
def test_a_3d_scene_in_one_section_is_measured_by_its_named_objects(tmp_path, monkeypatch, x, said):
    out = _draft_pic(tmp_path, monkeypatch, "s", "a 3D render of a red car and a chrome ball",
                     S3D.replace("X", str(x)), size="640x360")
    assert "3D objects: car x " in out and "ball x " in out, out
    assert said in out, out


def test_studio_survives_a_legacy_console(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="cp950", PIXELCANVAS_STUDIO_CMD="studio")
    env.pop("PYTHONUTF8", None)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "studio.py"), "new", "cafe", "--prompt",
                        "a café sign saying 'Crème brûlée ²'", "--size", "320x180"], cwd=tmp_path, env=env,
                       capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")


# ---- the medium stage

def _demo(pic):
    p = open_picture("art", "demo")
    s = SC.Session(p)
    s.mute = True
    out = s.render("t", quiet=True)
    return p, s, out


def test_medium_paints_the_picture_and_changing_it_runs_no_section(pic):
    code, out = run("set", "scene", "medium=oil", "out=demo.png")
    assert code == 0, out
    assert "medium: oil - the picture, preview and look sheet show it" in out
    assert re.search(r"buffers \d+\.\d, oil \d+\.\d\)", out)
    assert "[x] medium value pattern" in out and "[x] medium colours" in out
    oil = png("demo.png", "RGB").astype(float)
    code, out = run("set", "scene", "medium=ink")
    assert code == 0, out
    assert "cells run: none" in out and "buffers 0.0, ink" in out
    ink = png("demo.png", "RGB").astype(float)
    assert np.abs(oil - ink).mean() > 5
    code, out = run("set", "scene", "medium=off")
    assert "medium:" not in out
    assert np.abs(png("demo.png", "RGB").astype(float) - ink).mean() > 5


def test_medium_is_painted_while_items_are_open_and_over_a_failing_section(pic):
    run("set", "scene", "medium=watercolour", "out=demo.png")
    code, out = run("need", "add", "a green tree")
    assert "medium: watercolour - the picture, preview and look sheet show it" in out
    assert "painted over the drawing as it stands (item" in out
    cell = pic / "art" / "demo" / "layers" / "box.py"
    cell.write_text(cell.read_text(encoding="utf-8") + "\nundefined_name_here()\n", encoding="utf-8")
    code, out = run("render")
    assert "medium: watercolour - the picture" in out and "a section fails" in out


def test_a_missing_medium_is_the_first_clause_of_next(pic, monkeypatch):
    from pixelcanvas.studio import medium_stage as MS

    def boom(*a, **k):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(MS, "apply", boom)
    code, out = run("set", "scene", "medium=oil")
    assert _next(out).startswith("NEXT: the oil medium is not painted (RuntimeError: out of memory); ")


def test_medium_values_are_checked(pic):
    code, out = run("set", "scene", "medium=crayon")
    assert code != 0 and "choices: render, flat, oil, watercolour, ink, pixel" in out
    code, out = run("set", "scene", "medium=watercolor", "medium_mark=9")
    assert code != 0 and "medium_mark=9" in out
    code, out = run("set", "scene", "medium=watercolor")
    assert code == 0 and open_picture("art", "demo").data["scene"]["medium"] == "watercolour"


def test_medium_from_the_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PIXELCANVAS_STUDIO_CMD", "studio")
    code, out = run("new", "harbour", "--prompt", "a watercolour of a harbour at dawn", "--size", "320x180")
    assert code == 0 and "medium: watercolour from the request" in out
    assert open_picture("art", "harbour").data["scene"]["medium"] == "watercolour"
    code, out = run("new", "jazz", "--prompt", "a poster for a jazz night", "--size", "320x180")
    assert "medium:" not in out and open_picture("art", "jazz").data["scene"]["medium"] is None


def test_stages_strip(pic):
    run("set", "scene", "medium=pixel")
    code, out = run("stages")
    assert code == 0 and "flat silhouettes -> relief clay -> lit colour -> pixel" in out
    w, h = size_of(pic / "art" / "demo" / "stages.png")
    assert w > 3 * h


def test_each_medium_gate_fires_on_a_constructed_failure(pic):
    from pixelcanvas.studio import medium_stage as MS
    p, s, out = _demo(pic)
    comp = s.comp
    lit = np.clip(np.asarray(comp.canvas.buf, np.float32)[..., :3], 0, 1)
    g = {n: (ok, fix) for n, ok, _, fix in MS.gates(comp, p, out, lit, "oil", {})}
    assert all(ok for ok, _ in g.values()), g                       # the drawing itself passes
    g = {n: (ok, fix) for n, ok, _, fix in MS.gates(comp, p, out, 1 - lit, "oil", {})}
    assert not g["value pattern"][0] and "medium_strength=0.6" in g["value pattern"][1]
    grey = np.repeat(lit.mean(-1, keepdims=True), 3, -1)            # every colour gone
    g = {n: (ok, fix) for n, ok, _, fix in MS.gates(comp, p, out, grey, "pixel", {})}
    assert not g["colours"][0] and "medium_palette=" in g["colours"][1]
    flat = np.broadcast_to(lit.reshape(-1, 3).mean(0), lit.shape).copy()   # nothing stands out
    flat += np.random.default_rng(1).normal(0, 0.002, flat.shape).astype(np.float32)
    g = {n: (ok, fix) for n, ok, _, fix in MS.gates(comp, p, out, flat, "oil", {})}
    assert not g["contrast"][0]
    assert not g["text"][0] and "outline=on" in g["text"][1]


def test_medium_gate_is_the_next_step(pic, monkeypatch):
    from pixelcanvas.studio import medium_stage as MS
    real = MS.gates
    monkeypatch.setattr(MS, "gates", lambda *a, **k: [("value pattern", False, "follows the drawing 0.20",
                                                        "set scene medium_strength=0.6 (drifted)")] + real(*a, **k)[1:])
    code, out = run("set", "scene", "medium=oil")
    assert _next(out).startswith("NEXT: studio set scene medium_strength=0.6 (drifted) - medium value pattern")


def test_an_extra_positional_argument_is_named(tmp_path, monkeypatch):
    out = _draft_pic(tmp_path, monkeypatch, "pos", "a red ball",
                     "# %% ball\n# ball: red\nsk.paint(sk.circle(0.5, 0.5, 0.2), '#d63c2f', 0.6)\n", size="320x180")
    assert "the 3rd argument (0.6) is one too many: sk.paint takes 2 positional arguments" in out
    assert "sk.paint(" in out
