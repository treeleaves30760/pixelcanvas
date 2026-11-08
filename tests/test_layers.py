# -*- coding: utf-8 -*-
"""Editable layer export: recorder, exact recomposite, PSD / ORA / PNG round trips."""
import json
import os
import subprocess
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, layers  # noqa: E402


def _draw():
    cv = Canvas(160, 90)
    rec = layers.LayerRecorder(cv)
    rec.mark("sky")
    cv.gradient([(0, "#203060"), (1, "#f0b080")])
    rec.mark("sun")
    cv.glow(110, 40, 12, "#fff0c0", strength=1.5)               # HDR, add mode → see-through "effect" layer
    rec.mark("hill")
    X, Y = cv.xy()
    cv.paint("#203020", np.clip(Y - (60 + 8 * np.sin(X / 15)) + 0.5, 0, 1))
    rec.mark("post")
    cv.soft_clip(0.8).vignette(0.3)
    return cv, rec.finish()


def test_recorder_layers_reproduce_final():
    cv, st = _draw()
    assert [L.name for L in st][:4] == ["base", "sky", "sun", "hill"]          # post folded into the layers
    assert [L.kind for L in st][:4] == ["base", "paint", "effect", "paint"]
    assert len(st) == 4 or st[4].name == "post (residual)"
    err = np.abs(layers.composite(st) - np.clip(cv.buf, 0, 1)).max()
    assert err < 2.5 / 255
    hill = st[3].full()
    assert hill[85, 80, 3] > 0.99 and hill[10, 80, 3] == 0      # solid where painted, empty above the ridge
    sun = st[2].full()
    assert 0 < sun[40, 132, 3] < 1 and sun[85, 5, 3] == 0        # glow is partial alpha, local
    assert st[3].bbox[1] > 40                                    # cropped to its bounding box


def test_post_as_layer_mode_keeps_it():
    cv = Canvas(80, 40)
    rec = layers.LayerRecorder(cv, post="layer")
    rec.mark("sky"); cv.gradient([(0, "#203060"), (1, "#f0b080")])
    rec.mark("post"); cv.vignette(0.5)
    assert [L.name for L in rec.finish()] == ["base", "sky", "post"]


def test_no_ghost_after_hiding_an_object():
    """With a global post step, hiding an object must not leave a faint copy of it behind."""
    cv = Canvas(120, 80)
    rec = layers.LayerRecorder(cv)
    rec.mark("sky"); cv.gradient([(0, "#3050a0"), (1, "#6080c0")])
    rec.mark("ball"); X, Y = cv.xy(); cv.paint("#ff2010", np.clip(15 - np.hypot(X - 60, Y - 40) + 0.5, 0, 1))
    rec.mark("post"); cv.soft_clip(0.7).contrast(0.2).saturation(1.2).split_tone().vignette(0.4)
    st = rec.finish()
    assert np.abs(layers.composite(st) - np.clip(cv.buf, 0, 1)).max() < 2.5 / 255
    assert [L.name for L in st][:3] == ["base", "sky", "ball"]           # grading + vignette fold into the layers
    assert len(st) == 3 or st[3].full()[..., 3].mean() < 0.08           # at most a thin residual
    hidden = layers.composite(st, hidden={"ball"})
    ref = layers.composite(st, hidden={"ball"})[35:45, 20:30]            # plain sky elsewhere at the same height
    assert np.abs(hidden[35:45, 55:65].mean((0, 1)) - ref.mean((0, 1))).max() < 0.05   # no red ghost


def test_bloom_leaves_only_a_thin_residual():
    cv = Canvas(120, 80)
    rec = layers.LayerRecorder(cv)
    rec.mark("sky"); cv.gradient([(0, "#3050a0"), (1, "#6080c0")])
    rec.mark("ball"); X, Y = cv.xy(); cv.paint("#ffe0a0", np.clip(15 - np.hypot(X - 60, Y - 40) + 0.5, 0, 1))
    rec.mark("post"); cv.bloom(0.5, 6, 0.6).soft_clip(0.8)
    st = rec.finish()
    assert np.abs(layers.composite(st) - np.clip(cv.buf, 0, 1)).max() < 2.5 / 255
    if st[-1].name == "post (residual)":
        assert st[-1].full()[..., 3].mean() < 0.1


def test_hiding_a_layer_reveals_the_one_below():
    cv, st = _draw()
    no_hill = layers.composite(st, hidden={"hill"})
    assert np.abs(no_hill[85] - layers.composite(st)[85]).max() > 0.2


def test_save_psd_ora_pngs(tmp_path):
    cv, st = _draw()
    base = str(tmp_path / "x")
    layers.save_all(st, base, final=cv.buf)
    with open(base + ".psd", "rb") as f:
        assert f.read(4) == b"8BPS"
    with zipfile.ZipFile(base + ".ora") as z:
        assert z.namelist()[0] == "mimetype" and z.read("mimetype") == b"image/openraster"
        assert b'name="hill"' in z.read("stack.xml")
    man = json.load(open(os.path.join(base + "_layers", "layers.json"), encoding="utf-8"))
    assert [e["name"] for e in man["layers"]] == [L.name for L in st]
    back = layers.load_pngs(base + "_layers")
    assert np.abs(layers.composite(back) - np.clip(cv.buf, 0, 1)).max() < 2.5 / 255
    try:
        from psd_tools import PSDImage
    except ImportError:
        return
    psd = PSDImage.open(base + ".psd")
    assert [l.name for l in psd] == [L.name for L in st]
    comp = np.asarray(psd.composite(force=True).convert("RGB"), np.float32) / 255
    assert np.abs(comp - np.clip(cv.buf, 0, 1)).max() < 3 / 255


def test_packbits_roundtrip():
    rng = np.random.default_rng(0)
    plane = np.concatenate([np.zeros((3, 300), np.uint8), rng.integers(0, 4, (3, 300), dtype=np.uint8),
                            rng.integers(0, 256, (3, 300), dtype=np.uint8)])
    counts, data = layers._packbits_rows(plane)
    rows, pos = [], 0
    for c in counts:
        chunk, out, i = data[pos:pos + c], bytearray(), 0
        while i < len(chunk):
            h = chunk[i]
            if h < 128:
                out += chunk[i + 1:i + 2 + h]; i += 2 + h
            else:
                out += bytes([chunk[i + 1]]) * (257 - h); i += 2
        rows.append(bytes(out)); pos += c
    assert b"".join(rows) == plane.tobytes()


def test_render_cli_layers(tmp_path):
    out = str(tmp_path / "np.png")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "tools/render.py", "neon_poster", "--scale", "0.2", "--layers", "--out", out],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(str(tmp_path / "np.psd")) and os.path.exists(str(tmp_path / "np.ora"))
    r = subprocess.run([sys.executable, "tools/layers.py", "compose", str(tmp_path / "np_layers"), "--hide", "post",
                        "--out", str(tmp_path / "c.png")], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and os.path.exists(str(tmp_path / "c.png")), r.stderr
