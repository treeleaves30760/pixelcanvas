"""Face lines from the head construction, markings, line style, the head outline, crescent shading, gap bridges."""
from __future__ import annotations

import json

import numpy as np

from pixelcanvas.easel import draw_shade
from pixelcanvas.easel.draw_core import light_vec

from .test_draw_flow import draw, ok  # noqa: F401


def state():
    return json.load(open("pic.easel/state.json", encoding="utf-8"))


def fox_head(run):
    ok(run("new", "pic", "--size", "400x400", "--request", "a fox head"))
    ok(run("do", "plan subject fox --box 150,100,850,900; plan part head --of fox --box 200,150,800,850 --color fur; "
                 "plan light top-left; palette make --base #e8803a; palette pick 1; stage done; "
                 "sketch head fox --box 200,150,800,850 --shape round; stage done"))


def test_face_lines_and_expressions(draw):
    fox_head(draw)
    text = ok(draw("face", "s1", "--expression", "smile", "--kind", "human"))
    names = [r["src"] for r in state()["draw"]["lines"]]
    for n in ("face brow-l", "face brow-r", "face eye-l", "face lash-r", "face nose", "face mouth"):
        assert n in names, names
    assert "lines:" in text
    ok(draw("undo"))
    ok(draw("face", "s1"))          # a fox: an animal face by default, a muzzle and no brows
    names = [r["src"] for r in state()["draw"]["lines"]]
    assert "face muzzle" in names and "face brow-l" not in names
    code, text = draw("face", "s1", "--expression", "weird")
    assert code != 0 and "smile" in text


def test_outline_mark_shade_linestyle(draw):
    fox_head(draw)
    ok(draw("line", "trace", "s1", "--outline"))
    lines = state()["draw"]["lines"]
    assert len(lines) == 1 and lines[0]["closed"]
    ok(draw("flat", "auto"))
    ok(draw("mark", "head", "--kind", "muzzle", "--color", "white"))
    st = state()
    assert "head-muzzle" in st["regions"] and "head-muzzle" in st["draw"]["marks"]
    ok(draw("flat", "refresh"))
    assert "head-muzzle" in state()["regions"]          # the marking survives a rebuild of the flat regions
    ok(draw("do", "stage done; shade all"))
    text = ok(draw("linestyle", "--color", "auto"))
    assert "silhouette" in text
    rec = state()["draw"]["lines"][0]
    assert rec["weight"] > 1.0 and rec["color"].startswith("#")


def test_crescent_hugs_the_far_outline():
    m = np.zeros((200, 200), bool)
    yy, xx = np.mgrid[:200, :200]
    m[(xx - 100) ** 2 + (yy - 100) ** 2 < 80 ** 2] = True
    tones = draw_shade.form_masks(m, light_vec({"light": {"deg": 135, "z": 0.6}}), 1, 0.5)
    sh = dict(tones)["shadow"]
    ys, xs = np.nonzero(sh)
    assert xs.mean() > 105 and ys.mean() > 105        # bottom-right, away from a top-left light
    assert not sh[100, 100]                            # the middle stays lit: a crescent, not a half
