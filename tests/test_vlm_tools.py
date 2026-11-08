# -*- coding: utf-8 -*-
"""Text, SDF shapes, vision helpers and the iteration tools (look.py / render.py / fonts.py)."""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, text_mask, text_bbox, fit_text_size, wrap_text, load_font, has_cjk  # noqa: E402
from pixelcanvas import sdf, vision, line_mask, bezier_points  # noqa: E402
from pixelcanvas.color import grid  # noqa: E402


def _run(*args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)


# ---------------------------------------------------------------- text

def test_text_mask_anchor_and_multiline():
    m = text_mask(120, 200, "Hi\nthere", (100, 60), 24, anchor="mm")
    ys, xs = np.nonzero(m > 0.5)
    assert m.max() > 0.9
    assert abs(xs.mean() - 100) < 20 and abs(ys.mean() - 60) < 15         # centred on xy
    one = text_mask(120, 200, "Hi", (100, 60), 24, anchor="mm")
    assert np.ptp(np.nonzero(one > 0.5)[0]) < np.ptp(ys)                    # two lines are taller


def test_text_bbox_wrap_and_fit():
    f = load_font(None, 20)
    lines = wrap_text("one two three four five six seven", f, 80)
    assert len(lines) > 1 and all(f.getlength(l) <= 80 or " " not in l for l in lines)
    x0, y0, x1, y1 = text_bbox("Hello", (10, 10), 30)
    assert x1 > x0 and y1 > y0
    s = fit_text_size("TITLE", 200, 60)
    bx = text_bbox("TITLE", (0, 0), s)
    assert bx[2] - bx[0] <= 200 and bx[3] - bx[1] <= 60 and s > 10
    assert has_cjk("霓虹") and not has_cjk("neon")


def test_draw_text_styles_and_box():
    cv = Canvas(200, 100, "#202020")
    bb = cv.text("Hey", box=(20, 20, 180, 80), color="#ffffff", outline=2, shadow=True, glow=True).last_bbox
    assert 15 <= bb[0] and bb[2] <= 185 and np.isfinite(cv.buf).all()
    assert cv.buf[50, 100].max() > 0.2 or cv.buf.max() > 0.9
    cv.text("grad", (100, 50), 30, anchor="mm", gradient=[(0, "#ff0000"), (1, "#0000ff")])


# ---------------------------------------------------------------- sdf / curves

def test_sdf_shapes_and_ops():
    X, Y = grid(100, 100)
    c = sdf.sd_circle(X, Y, 50, 50, 20)
    assert c[50, 50] == pytest.approx(-20, abs=0.01) and c[50, 90] == pytest.approx(20, abs=0.01)
    assert sdf.fill(c).sum() == pytest.approx(np.pi * 400, rel=0.02)
    b = sdf.sd_box(X, Y, 50, 50, 30, 10, radius=5, angle=30)
    p = sdf.sd_polygon(X, Y, [(10, 10), (90, 10), (50, 90)])
    assert p[40, 50] < 0 < p[95, 5]
    u = sdf.union(c, b); i = sdf.intersect(c, b); s = sdf.subtract(c, b)
    assert (sdf.fill(i) <= sdf.fill(u) + 1e-6).all() and sdf.fill(s).sum() < sdf.fill(c).sum()
    assert sdf.fill(sdf.smooth_union(c, b, 8)).sum() >= sdf.fill(u).sum() - 1
    assert sdf.outline(c, 2).max() == pytest.approx(1)
    sh = sdf.bevel_light(sdf.sd_box(X, Y, 50, 50, 30, 30, 8), 8, light=(0, -1))
    assert sh[22, 50] > 1.05 and sh[78, 50] < 0.95 and sh[50, 50] == pytest.approx(1)
    for d in (sdf.sd_star(X, Y, 50, 50, 40), sdf.sd_ring(X, Y, 50, 50, 30, 4), sdf.sd_ellipse(X, Y, 50, 50, 40, 20),
              sdf.sd_segment(X, Y, 10, 10, 90, 90, 3), sdf.sd_arc(X, Y, 50, 50, 30, 0, 90, 4)):
        assert np.isfinite(d).all() and sdf.fill(d).sum() > 10


def test_bezier_and_line_mask():
    pts = bezier_points((0, 0), (50, 100), (100, 0), n=5)
    assert pts[0] == (0.0, 0.0) and pts[-1] == (100.0, 0.0) and pts[2] == (50.0, 50.0)
    assert len(bezier_points((0, 0), (1, 2), (3, 2), (4, 0), n=10)) == 10
    m = line_mask(60, 120, bezier_points((5, 50), (60, -20), (115, 50)), 4)
    assert m.max() > 0.95 and 0.02 < m.mean() < 0.3


# ---------------------------------------------------------------- vision helpers

def _img(tmp_path, name="a.png", w=160, h=90):
    p = str(tmp_path / name)
    cv = Canvas(w, h)
    cv.gradient([(0, "#102040"), (1, "#f0c080")]).glow(w * 0.66, h * 0.3, 6, (2, 2, 2))
    cv.save(p)
    return p


def test_vision_views(tmp_path):
    p = _img(tmp_path)
    assert vision.guides(p).size == (160, 90)
    z = vision.zoom(p, (0.5, 0.1, 0.8, 0.5), out_width=300)
    assert z.width == 300
    vm, frac = vision.value_map(p, 4)
    assert abs(frac.sum() - 1) < 1e-6 and vm.size == (160, 90)
    assert vision.edge_map(p).size == (160, 90)
    sheet = vision.contact_sheet([p, p, p], ["a", "b", "c"], tile_width=100)
    assert sheet.width > 300


def test_stage_capture():
    vision.enable_stages(width=32, target_size=(64, 36))
    cv = Canvas(64, 36)
    vision.stage(None, "sky")
    cv.paint("#ff0000")
    vision.stage(None, "red")
    cv.paint("#00ff00")
    shots = vision.take_stages(cv)
    assert [s[0] for s in shots] == ["sky", "red"]
    assert not vision.stages_enabled()
    vision.stage(cv, "ignored")                                   # no-op when off


# ---------------------------------------------------------------- CLI tools

@pytest.mark.parametrize("cmd", [["guides"], ["values"], ["edges"], ["check"], ["critique"], ["stats"],
                                 ["zoom", "0.1", "0.1", "0.6", "0.6"], ["probe", "0.5,0.5", "10,10"]])
def test_look_commands(tmp_path, cmd, monkeypatch, capsys):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import look
    p = _img(tmp_path)
    monkeypatch.setattr(sys, "argv", ["look.py", cmd[0], p, *cmd[1:]])
    assert look.main() == 0
    assert capsys.readouterr().out.strip()


def test_look_sheet(tmp_path):
    a, b = _img(tmp_path, "a.png"), _img(tmp_path, "b.png")
    out = str(tmp_path / "s.jpg")
    r = _run("tools/look.py", "sheet", a, b, "--labels", "one,two", "-o", out)
    assert r.returncode == 0 and os.path.exists(out), r.stderr


def test_render_set_sweep_stages(tmp_path):
    out = str(tmp_path / "n.png")
    r = _run("tools/render.py", "neon_poster", "--size", "216x270", "--out", out, "--stages", "--check",
             "--set", "TITLE='HELLO'", "--set", "SUN=(0.5, 0.5, 0.1)")
    assert r.returncode == 0, r.stderr
    for suffix in ("_stages.jpg", "_check.jpg", "_preview.jpg"):
        assert os.path.exists(out[:-4] + suffix), suffix
    sw = str(tmp_path / "sw.jpg")
    r = _run("tools/render.py", "neon_poster", "--size", "216x270", "--scale", "0.5", "--sweep", "HORIZON",
             "0.55", "0.7", "--out", sw)
    assert r.returncode == 0 and os.path.exists(sw), r.stderr
    r = _run("tools/render.py", "neon_poster", "--set", "NOPE=1")
    assert r.returncode != 0 and "NOPE" in (r.stderr + r.stdout)


def test_fonts_tool():
    r = _run("tools/fonts.py")
    assert r.returncode == 0 and "sans" in r.stdout, r.stderr
