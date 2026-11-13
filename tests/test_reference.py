# -*- coding: utf-8 -*-
"""Reference style probe, gap check, tools/match.py, render.py --ref and the soft-cel Sketch helpers."""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
REFS = os.path.join(ROOT, "skills", "pixelcanvas", "references")

from pixelcanvas import Canvas, reference  # noqa: E402
from pixelcanvas.illustrate import Sketch, cel_tone  # noqa: E402


def _bg(cv, color=(0.9, 0.92, 0.95)):
    cv.buf[:] = np.array(color, np.float32)
    return cv


def _flat_vs_banded(banded=True, line=None):
    cv = _bg(Canvas(640, 360))
    sk = Sketch(cv, light=(-0.6, -0.8))
    sk.paint(sk.circle(0.3, 0.5, 0.3), "#e05050", line=line, line_width=0.008)
    m = sk.circle(0.75, 0.5, 0.3)
    if banded:
        sk.cel(m, "#4060d0", shift=(-0.12, 15), depth=0.4, line=line, line_width=0.008)
    else:
        sk.paint(m, "#4060d0", line=line, line_width=0.008)
    return cv


def _regions(p):
    return {r["base"]: r for r in p["regions"]}


# ---------------------------------------------------------------------------------------------------- probe

def test_probe_flat_regions_and_palette():
    p = reference.probe(_flat_vs_banded(banded=False))
    assert p["structure"]["bands"] == 0 and p["structure"]["flat"] > 0.95
    hexes = [c["hex"] for c in p["palette"]]
    assert "#e05050" in hexes and "#4060d0" in hexes
    assert p["roles"]["background"] == hexes[0]
    assert not p["lines"]["present"]


def test_probe_two_bands_recovers_the_shift():
    p = reference.probe(_flat_vs_banded())
    banded = [r for r in p["regions"] if r["structure"] == "bands"]
    assert len(banded) == 1 and banded[0]["tones"] == 2
    dl, dh = banded[0]["shift"]
    assert abs(dl + 0.12) < 0.02 and abs(dh - 15) < 3
    assert any(r["structure"] == "flat" for r in p["regions"])


def test_probe_gradient_region():
    cv = Canvas(480, 270)
    Sketch(cv).ramp(np.ones((270, 480), np.float32), ["#2f6fd0", "#d8f0ff"])
    p = reference.probe(cv)
    assert p["structure"]["gradient"] > 0.9
    assert p["regions"][0]["structure"] == "gradient"


@pytest.mark.parametrize("width", [0.006, 0.012])
def test_probe_line_width_and_colour(width):
    cv = _bg(Canvas(960, 540), (0.95, 0.93, 0.9))
    sk = Sketch(cv)
    for y in (0.2, 0.4, 0.6, 0.8):
        sk.paint(sk.stroke([(0.1, y), (0.9, y + 0.05)], width, cap="flat"), "#6a2a3a")
    p = reference.probe(cv)
    assert p["lines"]["present"] and p["lines"]["coloured"]
    assert abs(p["lines"]["width"] - width) < 0.35 * width
    d = np.abs(np.array([int(p["lines"]["color"][i:i + 2], 16) for i in (1, 3, 5)]) - [0x6a, 0x2a, 0x3a])
    assert d.max() < 30


def test_sheet_text_has_paste_lines():
    t = reference.sheet_text(reference.probe(_flat_vs_banded(line="#5a2a40")), "x")
    assert "PALETTE = [" in t and "SHIFT, BANDS = (" in t and "LINE, LINE_WIDTH" in t


# ---------------------------------------------------------------------------------------------------- known structure

def _scene(kind):
    """Small pictures whose structure is known because they are drawn here."""
    cv = Canvas(640, 360)
    sk = Sketch(cv, light=(-0.6, -0.8), finish="full")
    if kind == "flat poster":
        cv.buf[:] = np.array([0.95, 0.9, 0.8], np.float32)
        sk.paint(sk.box(0.05, 0.1, 0.45, 0.9), "#e0503c")
        sk.paint(sk.circle(0.72, 0.4, 0.25), "#2f5fb0")
        sk.paint(sk.polygon([(0.55, 0.95), (0.95, 0.95), (0.75, 0.6)]), "#f2c230")
    elif kind == "two bands on flat":
        cv.buf[:] = np.array([0.93, 0.93, 0.9], np.float32)
        sk.cel(sk.circle(0.3, 0.5, 0.32), "#e0645a", bands=2, shift=(-0.12, -12))
        sk.cel(sk.ellipse(0.74, 0.5, 0.2, 0.36), "#3f7fd0", shift=(-0.12, 15))
    elif kind == "bands on graded base":
        cv.buf[:] = np.array([0.93, 0.93, 0.9], np.float32)
        sk.cel(sk.ellipse(0.5, 0.5, 0.42, 0.42), [(0, "#f08a70"), (1, "#c8455a")], shift=(-0.13, -12))
    elif kind == "airbrushed":
        sk.ramp(sk.box(0, 0, 1, 1), ["#3f7fd8", "#cdeefa"])
        sk.paint(sk.circle(0.5, 0.5, 0.3), [(0, "#ffd0a0"), (1, "#c0603a")], angle="radial")
    elif kind == "flat outlined":
        cv.buf[:] = np.array([0.96, 0.95, 0.92], np.float32)
        for x, c in ((0.25, "#e05050"), (0.75, "#50a070")):
            sk.paint(sk.circle(x, 0.5, 0.25), c, line="#202020", line_width=0.01)
    return cv


def _kinds(p):
    return {r["structure"] for r in p["regions"]}


def test_known_flat_poster():
    p = reference.probe(_scene("flat poster"))
    assert _kinds(p) == {"flat"} and p["shift"] is None and not p["lines"]["present"]


def test_known_two_bands_with_shifts_per_material():
    p = reference.probe(_scene("two bands on flat"))
    banded = [r for r in p["regions"] if r["structure"].startswith("bands")]
    assert len(banded) == 2
    by = {r["base"]: r for r in banded}
    red = min(by.values(), key=lambda r: abs(int(r["base"][1:3], 16) - 0xe0))
    blue = max(by.values(), key=lambda r: int(r["base"][5:7], 16))
    assert abs(red["shift"][0] + 0.12) < 0.03 and abs(red["shift"][1] + 12) < 5 and red["tones"] == 3
    assert abs(blue["shift"][0] + 0.12) < 0.03 and abs(blue["shift"][1] - 15) < 5
    assert "SHIFTS = {" in reference.sheet_text(p)


def test_known_bands_on_a_graded_base():
    p = reference.probe(_scene("bands on graded base"))
    r = max(p["regions"], key=lambda r: r["share"] if r["structure"] != "flat" else 0)
    assert r["structure"].startswith("bands") and abs(r["shift"][0] + 0.13) < 0.06
    assert p["structure"]["bands"] > 0.2


def test_known_airbrushed_has_no_bands():
    p = reference.probe(_scene("airbrushed"))
    assert p["structure"]["bands"] == 0 and p["structure"]["gradient"] > 0.5 and p["shift"] is None


def test_known_flat_with_dark_outlines():
    p = reference.probe(_scene("flat outlined"))
    assert p["lines"]["present"] and not p["lines"]["coloured"]
    assert p["structure"]["bands"] == 0 and p["shift"] is None


def test_known_anime_scene_paste_block():
    from scenes import illustration_anime
    p = reference.probe(illustration_anime.render(960, 540))
    t = reference.sheet_text(p)
    assert p["structure"]["bands"] > 0.2 and p["shift"] is not None
    assert "SHIFT, BANDS = None" not in t
    hair = [r for r in p["regions"] if r["structure"].startswith("bands") and r["shift"][1] < -5]
    assert hair and abs(hair[0]["shift"][0] + 0.13) < 0.04          # painted (-0.14, towards violet)
    assert any(r["structure"] == "gradient" for r in p["regions"])   # the sky


def test_cel_draft_is_not_reported_flat():
    ref = _scene("two bands on flat")
    c = reference.compare(ref, _scene("bands on graded base"))
    assert c["draft"]["structure"]["bands"] > 0.2
    assert not any(g["key"] == "bands" and "flat where" in g["message"] for g in c["gaps"] if g["gap"] > 0.3)


# ---------------------------------------------------------------------------------------------------- compare

def test_identical_pictures_match_fully():
    cv = _flat_vs_banded(line=True)
    c = reference.compare(cv, cv)
    assert c["match"] == 100 and all(g["gap"] == 0 for g in c["gaps"])
    assert "nothing measurable" in reference.gap_lines(c)[1]


def test_flat_copy_reports_missing_bands_first():
    c = reference.compare(_flat_vs_banded(), _flat_vs_banded(banded=False))
    assert c["gaps"][0]["key"] == "bands" and c["match"] < 95
    lines = reference.gap_lines(c)
    assert "flat where the reference has" in lines[1] and "sk.cel(" in lines[1]


def test_missing_lines_reported():
    c = reference.compare(_flat_vs_banded(line="#5a2a40"), _flat_vs_banded())
    g = {g["key"]: g for g in c["gaps"]}
    assert g["lines"]["gap"] == 1.0 and "line=" in g["lines"]["fix"]


def test_strict_options():
    with pytest.raises(TypeError, match="colors"):
        reference.probe(_flat_vs_banded(), colours_n=3)


def test_add_to_report_places_block_before_fix(tmp_path):
    ref = tmp_path / "ref.png"
    _flat_vs_banded().save(str(ref))
    out = tmp_path / "out.png"
    _flat_vs_banded(banded=False).save(str(out))
    rep = "OBJECTS\n  a\npicture: x.png\nFIX: nothing flagged"
    new = reference.add_to_report(rep, str(ref), str(out)).split("\n")
    i = next(n for n, t in enumerate(new) if t.startswith("REFERENCE"))
    assert new[i + 1].strip().startswith("match") and new[-2].startswith("picture:") and new[-1].startswith("FIX")
    assert reference.add_to_report(rep, None, str(out)) == rep


# ---------------------------------------------------------------------------------------------------- CLI

def _run(args, cwd):
    return subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))


def test_match_cli(tmp_path):
    ref, out = tmp_path / "ref.png", tmp_path / "out.png"
    _flat_vs_banded().save(str(ref))
    _flat_vs_banded(banded=False).save(str(out))
    tool = os.path.join(ROOT, "tools", "match.py")
    r = _run([tool, str(ref)], tmp_path)
    assert r.returncode == 0 and "STYLE SHEET" in r.stdout and "SHIFT, BANDS" in r.stdout
    r = _run([tool, str(ref), str(out)], tmp_path)
    assert r.returncode == 0 and "match" in r.stdout and "bands gap" in r.stdout
    r = _run([tool, str(ref), str(out), "--json"], tmp_path)
    data = json.loads(r.stdout)
    assert data["gaps"][0]["key"] == "bands" and 0 <= data["match"] < 100
    assert _run([tool, str(tmp_path / "nope.png")], tmp_path).returncode != 0


SCENE = '''
from pixelcanvas import Canvas
from pixelcanvas.illustrate import Sketch
PROMPT = "a red ball"
DEFAULT_SIZE = (320, 180)
{extra}
def render(width=320, height=180, seed=1, log=lambda s: None):
    cv = Canvas(width, height, "#e6ebf2")
    Sketch(cv).paint(Sketch(cv).circle(0.5, 0.5, 0.3), "#e05050", name="ball")
    return cv
'''


@pytest.mark.parametrize("how", ["flag", "constant"])
def test_render_ref(tmp_path, how):
    ref = tmp_path / "ref.png"
    _flat_vs_banded().save(str(ref))
    scene = tmp_path / "ball.py"
    scene.write_text(SCENE.format(extra='REFERENCE = "ref.png"' if how == "constant" else ""), encoding="utf-8")
    args = [os.path.join(ROOT, "tools", "render.py"), str(scene), "--check", "--out", str(tmp_path / "o.png")]
    if how == "flag":
        args += ["--ref", str(ref)]
    r = _run(args, tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = r.stdout.splitlines()
    i = next(n for n, t in enumerate(lines) if t.startswith("REFERENCE"))
    j = next(n for n, t in enumerate(lines) if t.startswith("FIX"))
    assert i < j and "match" in lines[i + 1]


# ---------------------------------------------------------------------------------------------------- Sketch

def test_cel_tone():
    assert cel_tone("#ffffff", 0.0, 0.0) == "#ffffff"
    sh = cel_tone("#f6d2bd")
    assert sh != "#f6d2bd" and sum(int(sh[i:i + 2], 16) for i in (1, 3, 5)) < sum((0xf6, 0xd2, 0xbd))
    w = cel_tone("#ffffff", -0.15)                         # whites shade lavender
    assert int(w[5:7], 16) > int(w[3:5], 16)


def test_cel_helpers_draw_and_record():
    from pixelcanvas import manifest
    cv = Canvas(480, 270, "#dfe8f5")
    sk = Sketch(cv, light=(-0.6, -0.8))
    sk.ramp(sk.box(0, 0, 1, 0.5), ["#3f7fd8", "#cdeefa"], steps=4, name="sky")
    with sk.at(0.5, 0.5, 0.4):
        hair = sk.ellipse(0, 0, 0.6, 0.5)
        m = sk.cel(hair, [(0, "#2f8f5a"), (1, "#6fcf8a")], bands=2, highlight=0.8, rim=True, bounce=True,
                   line=True, name="hair")
        assert m.shape == (270, 480)
        al = sk.sheen(hair, y=0.3)
        assert al.max() > 0.5 and float((al * (1 - np.asarray(hair))).max()) < 0.05
        sk.eye(0.25, 0.1, 0.3, "#c0602a", name="eye")
        sk.eye(-0.25, 0.1, 0.3, "#c0602a", mirror=True)
        sk.blush(0.3, 0.3, 0.08)
        sk.ink([(-0.2, 0.4), (0, 0.45), (0.2, 0.4)], 0.02, name="mouth")
    names = {r["name"] for r in manifest.summary(cv)}
    assert {"sky", "hair", "eye", "mouth"} <= names
    p = reference.probe(cv)
    assert p["structure"]["bands"] > 0 and p["lines"]["present"]
    with pytest.raises(TypeError):
        sk.cel(hair, "#ff0000", depht=0.3)
    sk.cel(hair, "#ff0000", glow="#ffd27a", shadow=True, opacity=0.9)              # paint's common options
    sk.ramp(hair, ["#000000", "#ffffff"], line=True, glow="#ffffff", shadow=True)
    sk.ink([(0.1, 0.1), (0.4, 0.2)], 0.01, glow="#ffffff")
    with pytest.raises(TypeError, match="glow"):
        sk.cel(hair, "#ff0000", glo="#ffffff")
    with pytest.raises(ValueError):
        sk.cel(hair, "#ff0000", bands=3)


def test_cel_band_sits_away_from_the_light():
    cv = Canvas(400, 400, "#ffffff")
    sk = Sketch(cv, light=(-1, 0))                          # light from the left
    sk.cel(sk.circle(0.5, 0.5, 0.35), "#e05050")
    row = cv.buf[200, :, 0]
    assert row[300] < row[120] - 0.05                       # right side (away from the light) is darker


# ---------------------------------------------------------------------------------------------------- docs

def _block(page, contains):
    with open(os.path.join(REFS, page), encoding="utf-8") as f:
        blocks = re.findall(r"```python\n(.*?)```", f.read(), re.S)
    return [b for b in blocks if contains in b]


def test_reference_page_blocks_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = _block("reference.md", "")
    assert blocks
    for b in blocks:
        exec(b, {})


def test_techniques_anime_recipe_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pre = _block("techniques.md", "")[0].replace("W, H = 1920, 1080", "W, H = 480, 270")
    rec = _block("techniques.md", "sk.sheen")
    assert len(rec) == 1
    g = {}
    exec(pre, g)
    exec(rec[0], g)
    assert reference.probe(g["cv"])["structure"]["bands"] > 0


def test_anime_scene_renders_soft_cel():
    from scenes import illustration_anime
    cv = illustration_anime.render(480, 270)
    p = reference.probe(cv)
    assert p["structure"]["bands"] > 0.15 and p["lines"]["present"]
