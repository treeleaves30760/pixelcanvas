# -*- coding: utf-8 -*-
"""Text, SDF shapes, vision helpers and the iteration tools (look.py / render.py / fonts.py)."""
import json
import os
import re
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, text_mask, text_bbox, fit_text_size, wrap_text, load_font, has_cjk  # noqa: E402
from pixelcanvas import sdf, vision, line_mask, bezier_points  # noqa: E402
from pixelcanvas.color import grid  # noqa: E402


CJK = re.compile("[\\u2e80-\\u9fff\\uff00-\\uffef]")


def _run(*args, env=None):
    e = dict(os.environ, PYTHONIOENCODING="utf-8")
    e.update(env or {})
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=e)


def _english(r):
    """The tools' own wording is English: CJK may appear only inside quoted user content (scene text, prompts)."""
    out = r.stdout + r.stderr
    out = re.sub(r"'[^'\n]*'|\"[^\"\n]*\"", "''", out)
    assert not CJK.search(out), out[:400]


def _last_json(r):
    line = [l for l in r.stdout.splitlines() if l.strip()][-1]
    return json.loads(line)


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

def _warn_keys(w):
    return {"rule", "severity", "message", "where", "value", "threshold", "fix"} <= set(w) and w["severity"] in ("info", "warn", "fail")


@pytest.mark.parametrize("cmd", [["guides"], ["values"], ["edges"], ["check"], ["critique"], ["stats"], ["report"],
                                 ["zoom", "0.1", "0.1", "0.6", "0.6"], ["probe", "0.5,0.5", "10,10"]])
def test_look_commands_english(tmp_path, cmd, monkeypatch, capsys):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import look
    p = _img(tmp_path)
    monkeypatch.setattr(sys, "argv", ["look.py", cmd[0], p, *cmd[1:]])
    assert look.main() == 0
    out = capsys.readouterr().out
    assert out.strip() and not CJK.search(out)


@pytest.mark.parametrize("cmd", [["report"], ["critique"], ["stats"], ["probe", "0.5,0.5", "10,10"], ["check"]])
def test_look_json(tmp_path, cmd):
    p = _img(tmp_path)
    r = _run("tools/look.py", cmd[0], p, *cmd[1:], "--json")
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout.strip().splitlines()[-1] if cmd[0] == "check" else r.stdout)       # one document
    _english(r)
    if cmd[0] == "report":
        assert len(doc["grid"]) == 9 and doc["luma"]["p50"] >= 0 and doc["ascii"]["lines"]
    elif cmd[0] == "critique":
        assert set(doc["counts"]) == {"fail", "warn", "info"} and all(_warn_keys(w) for w in doc["warnings"])
    elif cmd[0] == "stats":
        assert doc["width"] == 160 and 0 <= doc["luma_p50"] <= 1
    elif cmd[0] == "probe":
        assert len(doc["points"]) == 2 and doc["points"][0]["color"].startswith("#")
    else:
        assert "warnings" in doc and os.path.exists(doc["check"])


def _arr_img(tmp_path, arr, name="x.png"):
    from PIL import Image
    p = str(tmp_path / name)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(p)
    return p


def test_critique_rules(tmp_path):
    rules = lambda a: {w["rule"]: w for w in vision.critique(_arr_img(tmp_path, a))}       # noqa: E731
    white = rules(np.full((270, 480, 3), 255.0))
    assert white["blown_highlights"]["severity"] == "fail" and white["blown_highlights"]["fix"]
    ramp = (100 + 30 * np.linspace(0, 1, 540))[:, None, None] * np.ones((540, 960, 3))
    r = rules(ramp)
    assert "banding" in r and "sparse_composition" in r
    sub = np.full((540, 960, 3), (30.0, 30.0, 60.0))
    sub[300:540, 400:560] = (240, 200, 100)                       # subject cut off by the bottom edge
    assert "subject_cropped" in rules(sub)
    row = np.full((540, 960, 3), (30.0, 30.0, 60.0))
    for x in (100, 330, 560, 790):
        row[240:300, x:x + 80] = (240, 200, 100)
    assert "subjects_in_a_row" in rules(row)
    blend = np.full((540, 960, 3), (120.0, 110.0, 100.0))
    blend[150:400, 300:650] = (128, 117, 106)
    assert "blends_into_background" in rules(blend)
    noisy = np.clip(128 + np.random.default_rng(0).normal(0, 22, (540, 960, 1)), 0, 255).repeat(3, 2)
    assert "jpeg_noise" in rules(noisy)
    for w in vision.critique(_arr_img(tmp_path, sub)):
        assert _warn_keys(w) and (w["where"] is None or len(w["where"]) in (2, 4))
        assert vision.format_warning(w).startswith(f"[{w['severity']}]")


def test_canvas_stats_english(capsys):
    cv = Canvas(8, 8, "#808080")
    assert cv.stats() is cv
    out = capsys.readouterr().out
    assert "fraction of pixels > 1" in out and not CJK.search(out)
    assert cv.last_stats["width"] == 8


def test_look_sheet(tmp_path):
    a, b = _img(tmp_path, "a.png"), _img(tmp_path, "b.png")
    out = str(tmp_path / "s.jpg")
    r = _run("tools/look.py", "sheet", a, b, "--labels", "one,two", "-o", out)
    assert r.returncode == 0 and os.path.exists(out), r.stderr


def test_render_set_sweep_stages(tmp_path):
    out = str(tmp_path / "n.png")
    r = _run("tools/render.py", "neon_poster", "--size", "216x270", "--out", out, "--stages", "--check",
             "--outdir", str(tmp_path / "work"), "--set", "TITLE='HELLO'", "--set", "SUN=(0.5, 0.5, 0.1)")
    assert r.returncode == 0, r.stderr
    _english(r)
    assert "check:" in r.stdout
    assert os.path.exists(out[:-4] + "_stages.jpg")                     # asked for: next to the picture
    work = str(tmp_path / "work")                                         # look-at files: the work folder
    for suffix in ("_check.jpg", "_preview.jpg"):
        assert os.path.exists(os.path.join(work, "n" + suffix)) and not os.path.exists(out[:-4] + suffix), suffix
    sw = str(tmp_path / "sw.jpg")
    r = _run("tools/render.py", "neon_poster", "--size", "216x270", "--scale", "0.5", "--sweep", "HORIZON",
             "0.55", "0.7", "--out", sw)
    assert r.returncode == 0 and os.path.exists(sw), r.stderr


def test_render_json(tmp_path):
    out = str(tmp_path / "s.png")
    r = _run("tools/render.py", "still_life", "--scale", "0.25", "--out", out, "--stages", "--check", "--json")
    assert r.returncode == 0, r.stderr
    _english(r)
    doc = _last_json(r)
    assert doc["scene"] == "still_life" and doc["size"] == [480, 270] and doc["scale"] == 0.25
    assert isinstance(doc["seed"], int) and doc["seconds"] > 0
    assert {"png", "preview", "check", "stages"} <= set(doc["outputs"])
    assert [s["name"] for s in doc["steps"]] == ["wall & table", "small sphere", "sphere", "post"]
    assert all(s["seconds"] >= 0 for s in doc["steps"])
    assert all(_warn_keys(w) for w in doc["warnings"])
    assert doc["stats"]["width"] == 480 and "luma_p50" in doc["stats"]


def test_unknown_scene_and_set_messages():
    r = _run("tools/render.py", "still_lyfe")
    assert r.returncode != 0 and "still_life" in r.stderr and "unknown scene" in r.stderr
    r = _run("tools/render.py", "still_life", "--set", "NOPE=1")
    assert r.returncode != 0 and "NOPE" in r.stderr and "available constants" in r.stderr and "DEFAULT_SEED" in r.stderr
    _english(r)


BAD_SCENE = """
import os
import numpy as np
from pixelcanvas import *
from pixelcanvas.scenery import *
DEFAULT_SIZE = (160, 90)
def render(width, height, seed=1, log=None):
    log("start")
    sc = Scene(width, height, seed, log=log)
    case = os.environ["BADCASE"]
    if case == "shape":
        sc.cv.paint("#ff0000", np.ones((10, 10), np.float32))
    elif case == "kw":
        sc.cv.paint("#ff0000", np.ones((height, width), np.float32), colour=1)
    elif case == "sun":
        sun(sc)
    return sc.cv
"""


@pytest.mark.parametrize("case, needle", [("shape", "does not match the canvas"), ("kw", "unknown keyword 'colour'"),
                                          ("sun", "set_sun")])
def test_scene_errors_are_explained(case, needle):
    path = os.path.join(ROOT, "scenes", "_test_bad_scene_tmp.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(BAD_SCENE)
    try:
        r = _run("tools/render.py", "_test_bad_scene_tmp", "--json", env={"BADCASE": case})
    finally:
        os.remove(path)
    assert r.returncode != 0
    _english(r)
    lines = r.stderr.splitlines()
    assert lines[0].startswith("ERROR ") and "_test_bad_scene_tmp.py line" in lines[1]
    assert needle in r.stderr and "traceback (tail)" in r.stderr
    err = _last_json(r)["error"]
    assert err["type"] and err["line"] and err["likely_cause"]


def test_numbered_rounds(tmp_path):
    env = {"PIXELCANVAS_ROUNDS_DIR": str(tmp_path / "rounds")}
    for i in range(2):
        r = _run("tools/render.py", "still_life", "--scale", "0.2", "--out", str(tmp_path / f"o{i}.png"), "--round",
                 "--seed", str(i + 1), env=env)
        assert r.returncode == 0, r.stderr
    r = _run("tools/render.py", "still_life", "--scale", "0.2", "--out", str(tmp_path / "o2.png"),
             env=dict(env, PIXELCANVAS_ROUNDS="1"))
    assert r.returncode == 0, r.stderr
    folder = tmp_path / "rounds" / "still_life"
    assert sorted(f for f in os.listdir(folder) if f.endswith(".jpg")) == ["r01_preview.jpg", "r02_preview.jpg", "r03_preview.jpg"]
    first = (folder / "r01_preview.jpg").read_bytes()
    r = _run("tools/render.py", "still_life", "--scale", "0.2", "--out", str(tmp_path / "o3.png"), "--round", env=env)
    assert r.returncode == 0, r.stderr
    assert (folder / "r01_preview.jpg").read_bytes() == first and (folder / "r04_preview.jpg").exists()   # never overwritten
    r = _run("tools/look.py", "rounds", "still_life", "--json", env=env)
    assert r.returncode == 0, r.stderr
    _english(r)
    doc = _last_json(r)
    assert [x["round"] for x in doc["rounds"]] == [1, 2, 3, 4] and os.path.exists(doc["sheet"])
    r = _run("tools/look.py", "rounds", "no_such_scene", env=env)
    assert r.returncode != 0 and "--round" in r.stderr


def test_trace_file(tmp_path):
    trace = tmp_path / "trace.jsonl"
    env = {"PIXELCANVAS_TRACE": str(trace)}
    r = _run("tools/render.py", "still_life", "--scale", "0.2", "--out", str(tmp_path / "t.png"), env=env)
    assert r.returncode == 0, r.stderr
    r = _run("tools/look.py", "critique", str(tmp_path / "t.png"), env=env)
    assert r.returncode == 0, r.stderr
    recs = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    assert [x["tool"] for x in recs] == ["render", "look"]
    assert recs[0]["scene"] == "still_life" and recs[0]["ok"] and recs[0]["outputs"] and recs[0]["seconds"] > 0
    assert recs[1]["command"] == "critique" and "time" in recs[1] and "warnings" in recs[1]


def test_new_scene_kinds():
    r = _run("tools/new_scene.py", "--list")
    assert r.returncode == 0
    _english(r)
    for kind in ("landscape", "poster", "blank"):
        assert kind in r.stdout
    name = "zz_test_blank_tmp"
    path = os.path.join(ROOT, "scenes", f"{name}.py")
    try:
        r = _run("tools/new_scene.py", name, "--kind", "blank")
        assert r.returncode == 0 and f'NAME = "{name}"' in open(path, encoding="utf-8").read()
        assert _run("tools/new_scene.py", name, "--poster").returncode != 0           # already exists
        assert _run("tools/new_scene.py", "zz_other", "--kind", "nope").returncode != 0
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_fonts_tool():
    r = _run("tools/fonts.py")
    assert r.returncode == 0 and "sans" in r.stdout, r.stderr


# ---------------------------------------------------------------- review fixes

def test_flat_image_is_not_jpeg_noise():
    a = np.full((270, 480, 3), 0.5, np.float32)
    rules = {w["rule"] for w in vision.critique(a)}
    assert "jpeg_noise" not in rules and "sparse_composition" in rules


def test_large_subject_cut_by_left_edge():
    H, W = 270, 480
    yy, xx = np.mgrid[0:H, 0:W]
    a = np.full((H, W, 3), [0.1, 0.2, 0.4], np.float32)
    a[(xx ** 2 + (yy - 135) ** 2) < 110 ** 2] = [0.95, 0.8, 0.2]
    ws = [w for w in vision.critique(a) if w["rule"] == "subject_cropped"]
    assert ws and "left" in ws[0]["message"]


def test_bad_colour_names_the_value(tmp_path):
    scene = os.path.join(ROOT, "scenes", "zz_test_badcolour.py")
    try:
        with open(scene, "w", encoding="utf-8") as f:
            f.write("from pixelcanvas import *\nDEFAULT_SIZE=(64, 48)\n"
                    "def render(width=64, height=48, seed=1, log=None):\n"
                    "    cv = Canvas(width, height)\n    cv.gradient([(0, 'nothex'), (1, '#fff')])\n    return cv\n")
        r = _run("tools/render.py", "zz_test_badcolour", "--out", str(tmp_path / "x.png"))
    finally:
        os.remove(scene)
    assert r.returncode == 1 and "'nothex'" in r.stderr and "likely cause" in r.stderr


def test_json_commands_keep_stdout_pure(tmp_path):
    from PIL import Image
    p = tmp_path / "a.png"
    Image.fromarray((np.random.RandomState(0).rand(90, 120, 3) * 255).astype("uint8")).save(p)
    r = _run("tools/look.py", "critique", str(p), "--json")
    json.loads(r.stdout)
    r = _run("tools/look.py", "sheet", str(p), str(p))
    assert "a_sheet.jpg" in r.stdout


# ---------------------------------------------------------------- CJK output, animation

CJK_SCENE = """
from pixelcanvas import Canvas
DEFAULT_SIZE = (320, 180)
PROMPT = 'a poster titled "\u591c\u66f2"'
def render(width, height, seed=1, log=None):
    log("title")
    cv = Canvas(width, height, "#203040")
    cv.text("\u591c\u66f2", (width // 2, height // 2), size=height // 3, anchor="mm", color="#ffffff")
    return cv
"""


def test_cjk_content_is_printed_verbatim_in_report_and_json(tmp_path):
    path = os.path.join(ROOT, "scenes", "_test_cjk_tmp.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(CJK_SCENE)
    try:
        r = _run("tools/render.py", "_test_cjk_tmp", "--out", str(tmp_path / "c.png"), "--check", "--json")
    finally:
        os.remove(path)
    assert r.returncode == 0, r.stderr
    assert "\u591c\u66f2" in r.stdout and "\\u591c" not in r.stdout          # verbatim, never \uXXXX escapes
    doc = _last_json(r)
    assert any("\u591c\u66f2" in o["name"] for o in doc["objects"])
    assert "\u591c\u66f2" in r.stdout.strip().splitlines()[-1]                 # --json keeps the characters
    assert not doc["prompt_issues"] and "reading order" in r.stdout
    report = types.SimpleNamespace(stdout="\n".join(r.stdout.splitlines()[:-1]), stderr=r.stderr)
    _english(report)                                                           # the report text; the JSON line holds user text


def test_text_only_picture_lists_reading_order(tmp_path):
    out = str(tmp_path / "n.png")
    r = _run("tools/render.py", "neon_poster", "--size", "216x270", "--out", out, "--check", "--json")
    assert r.returncode == 0, r.stderr
    assert "reading order" in r.stdout and "relations:" not in r.stdout
    doc = _last_json(r)
    assert doc["relations"] == [] and len(doc["reading_order"]) == len([o for o in doc["objects"] if o["kind"] == "text"])
    assert re.search(r"1\. label '.*'  \w+ \w+  \d+ px high", r.stdout)


def test_render_animate_gif_filmstrip_bounce_hold(tmp_path):
    from PIL import Image
    out = str(tmp_path / "a.gif")
    r = _run("tools/render.py", "anim_campfire", "--animate", "T", "0", "1", "--frames", "4", "--scale", "0.15",
             "--out", out, "--json")
    assert r.returncode == 0, r.stderr
    doc = _last_json(r)
    assert doc["animation"]["frames"] == 4 and doc["outputs"]["animation"] and doc["outputs"]["filmstrip"]
    im = Image.open(out)
    assert im.format == "GIF" and im.n_frames == 4 and im.info.get("loop") == 0
    assert os.path.exists(str(tmp_path / "a_frames.jpg"))
    # bounce appends the reversed frames (ends not repeated); hold repeats the last frame; --loop skips the END frame
    out2 = str(tmp_path / "b.webp")
    r = _run("tools/render.py", "anim_campfire", "--animate", "T:0:1", "--frames", "4", "--scale", "0.15",
             "--bounce", "--hold", "2", "--loop", "--format", "webp", "--out", out2, "--json")
    assert r.returncode == 0, r.stderr
    doc = _last_json(r)
    assert doc["animation"]["frames"] == 4 + 2 + 2
    wp = Image.open(out2)                                             # identical neighbours merge into one frame
    assert 1 < wp.n_frames <= 8 and wp.info.get("loop", 0) == 0
    out3 = str(tmp_path / "c.png")
    r = _run("tools/render.py", "anim_campfire", "--animate", "T", "0", "1", "--frames", "3", "--scale", "0.15",
             "--format", "apng", "--out", out3)
    assert r.returncode == 0, r.stderr
    assert Image.open(out3).n_frames == 3


def test_animate_helpers_and_errors():
    import render
    import types
    mod = types.SimpleNamespace(A=1.0, B=2, T=0.0)
    specs = render.parse_animate(mod, ["A:0.2:0.8,B:2:6"])
    assert specs == [("A", 0.2, 0.8), ("B", 2.0, 6.0)]
    vals = render.frame_values(specs, 5)
    assert [round(v["A"], 3) for _, v in vals] == [0.2, 0.35, 0.5, 0.65, 0.8]
    loop = render.frame_values(specs, 4, loop=True)
    assert abs(loop[-1][1]["A"] - 0.65) < 1e-9
    ease = render.frame_values([("T", 0.0, 1.0)], 5, ease="inout")
    assert ease[1][1]["T"] < 0.25 < 0.75 < ease[3][1]["T"]
    assert render.parse_animate(mod, ["0", "1"]) == [("T", 0.0, 1.0)]
    seq = render.animation_sequence(list("abcd"), bounce=True, hold=2)
    assert seq == list("abcdcb") + ["b", "b"]
    with pytest.raises(render.UserError):
        render.parse_animate(types.SimpleNamespace(A=1), ["0", "1"])
    r = _run("tools/render.py", "still_life", "--animate", "NOPE", "0", "1")
    assert r.returncode == 2 and "NOPE" in r.stderr


def test_rounds_gif(tmp_path):
    from PIL import Image
    env = {"PIXELCANVAS_ROUNDS_DIR": str(tmp_path / "rounds")}
    for i in range(3):
        r = _run("tools/render.py", "still_life", "--scale", "0.2", "--out", str(tmp_path / f"o{i}.png"), "--round",
                 "--seed", str(i + 1), env=env)
        assert r.returncode == 0, r.stderr
    r = _run("tools/look.py", "rounds", "still_life", "--gif", "--json", env=env)
    assert r.returncode == 0, r.stderr
    doc = _last_json(r)
    assert Image.open(doc["gif"]).n_frames == 3


# ---------------------------------------------------------------- styled checks and zoom sources

def test_zoom_takes_the_preview_or_the_png(tmp_path):
    base = tmp_path / "pic_x0.5"
    cv = Canvas(400, 225)
    cv.gradient([(0, "#102040"), (1, "#f0c080")])
    cv.save(str(base) + ".png", preview=str(base) + "_preview.jpg")
    r = _run("tools/look.py", "zoom", str(base) + "_preview.jpg", "0.1", "0.1", "0.5", "0.5")
    assert r.returncode == 0, r.stderr
    assert "zooming into pic_x0.5.png (full resolution) instead of the preview (400x225)" in r.stdout
    os.remove(str(base) + ".png")
    r = _run("tools/look.py", "zoom", str(base) + ".png", "0.1", "0.1", "0.5", "0.5")
    assert r.returncode == 0, r.stderr
    assert "pic_x0.5.png not found; zooming into the preview pic_x0.5_preview.jpg" in r.stdout
    r = _run("tools/look.py", "zoom", str(tmp_path / "none.png"), "0.1", "0.1", "0.5", "0.5")
    assert r.returncode != 0 and "image not found" in (r.stdout + r.stderr)


def test_render_check_does_not_fail_a_style_for_its_own_look(tmp_path):
    out = str(tmp_path / "w.png")
    r = _run("tools/render.py", "mountains_sunset", "--scale", "0.2", "--style", "watercolor", "--check", "--json",
             "--out", out)
    assert r.returncode == 0, r.stderr
    doc = _last_json(r)
    assert doc["counts"]["fail"] == 0 and doc["styles_applied"] == ["watercolor"]
    assert not [w for w in doc["warnings"] if w["rule"] == "blown_highlights" and w["severity"] != "info"]


# ---------------------------------------------------------------- report lines that are true (r1 dogfood)

REPORT_SCENE = '''
from pixelcanvas import Canvas, manifest
DEFAULT_SIZE = (480, 270)
KIND = "{kind}"
PROMPT = "A flyer for a market in Taipei: title 'Fresh Market' in bold type, the dates, a teal-to-navy background"
def render(width, height, seed=1, log=None):
    log("title")
    cv = Canvas(width, height, "#20405a")
    cv.text("Fresh Market", (width // 2, height // 3), size=height // 6, anchor="mm", color="#ffffff")
    cv.text("\u9031\u672b \u00d8 11/21", (width // 2, height // 2), size=height // 12, anchor="mm", color="#ffffff")
    return cv
'''


def _report_scene(tmp_path, kind="design"):
    p = tmp_path / "flyer.py"
    p.write_text(REPORT_SCENE.replace("{kind}", kind), encoding="utf-8")
    return str(p)


def test_deliver_prints_absolute_final_and_preview_paths(tmp_path):
    scene = _report_scene(tmp_path)
    final = tmp_path / "final"
    r = _run("tools/render.py", scene, "--scale", "0.5", "--deliver", str(final) + os.sep)
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.splitlines() if l.startswith(("deliver ", "preview "))]
    dl = [l.split(" ", 1)[1] for l in lines if l.startswith("deliver ")]
    pv = [l.split(" ", 1)[1].split("   ")[0] for l in lines if l.startswith("preview ")]
    assert dl and all(os.path.isabs(p) and os.path.exists(p) for p in dl), lines
    assert pv and os.path.isabs(pv[0]) and os.path.exists(pv[0]), lines


def test_report_prints_utf8_on_a_cp950_console(tmp_path):
    scene = _report_scene(tmp_path)
    r = subprocess.run([sys.executable, "tools/render.py", scene, "--scale", "0.5", "--check"], cwd=ROOT,
                       capture_output=True, env=dict(os.environ, PYTHONIOENCODING="cp950"))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    out = r.stdout.decode("utf-8")
    assert "\u9031\u672b" in out and "\u00d8" in out and "UnicodeEncodeError" not in r.stderr.decode("utf-8", "replace")


def test_report_text_block_lists_quoted_strings_and_skips_non_things(tmp_path):
    r = _run("tools/render.py", _report_scene(tmp_path), "--scale", "0.5", "--check")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "quoted 'Fresh Market': drawn" in out
    text = out.split("TEXT (")[1].split("OBJECTS")[0]
    assert "\n  none" not in text and "no problems in 2 text items" in text
    for word in ("Taipei", "bold type", "dates", "teal-to-navy", "background"):
        assert f"{word} \u2013 not recorded" not in out, word


def test_kind_words_map_and_a_misspelt_kind_is_rejected(tmp_path):
    r = _run("tools/render.py", _report_scene(tmp_path, "dashboard"), "--scale", "0.5", "--check")
    assert r.returncode == 0, r.stderr
    assert "kind: design (from KIND)" in r.stdout
    r = _run("tools/render.py", _report_scene(tmp_path, "desgin"), "--scale", "0.5", "--check")
    assert r.returncode == 2 and "did you mean 'design'" in r.stderr