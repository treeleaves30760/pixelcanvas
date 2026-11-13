# -*- coding: utf-8 -*-
"""Portability: scipy-free fallbacks match scipy, did-you-mean module errors, no BOMs, no deprecated Pillow calls."""
import importlib
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, filters, noise  # noqa: E402
from pixelcanvas import _args  # noqa: E402

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

need_scipy = pytest.mark.skipif(_ndi is None, reason="scipy not installed")


def _rng_img(h=60, w=80, c=None, seed=3):
    r = np.random.default_rng(seed)
    return r.random((h, w) if c is None else (h, w, c)).astype(np.float32)


# ------------------------------------------------------------------ scipy-free fallbacks

@pytest.mark.parametrize("sigma", [0.09, 0.5, 1.5, 4.0])
def test_gaussian_fallback_matches_scipy_for_small_sigma(monkeypatch, sigma):
    a = _rng_img()
    ref = filters.gaussian_blur(a, sigma)
    monkeypatch.setattr(filters, "_ndi", None)
    got = filters.gaussian_blur(a, sigma)
    assert got.shape == a.shape and got.dtype == np.float32
    assert np.abs(got - ref).max() < 2e-3


def test_gaussian_fallback_keeps_thin_lines_at_small_sigma(monkeypatch):
    monkeypatch.setattr(filters, "_ndi", None)
    a = np.zeros((20, 20), np.float32)
    a[10] = 1.0
    assert filters.gaussian_blur(a, 0.09)[10].min() > 0.99          # sigma below a pixel must not smear


@pytest.mark.parametrize("sigma", [8.0, 20.0])
def test_gaussian_fallback_large_sigma_is_close(monkeypatch, sigma):
    a = filters.gaussian_blur(_rng_img(120, 160), 3.0)
    ref = filters.gaussian_blur(a, sigma)
    monkeypatch.setattr(filters, "_ndi", None)
    got = filters.gaussian_blur(a, sigma)
    assert np.abs(got - ref).max() < 0.05


def test_gaussian_fallback_rgb(monkeypatch):
    monkeypatch.setattr(filters, "_ndi", None)
    a = _rng_img(30, 40, 3)
    out = filters.gaussian_blur(a, 2.0)
    assert out.shape == a.shape and abs(float(out.mean()) - float(a.mean())) < 0.01


@need_scipy
@pytest.mark.parametrize("shape", [(40, 50), (40, 50, 3)])
def test_sample_fallback_is_bilinear_like_scipy(monkeypatch, shape):
    img = _rng_img(*shape[:2], c=(shape[2] if len(shape) == 3 else None))
    r = np.random.default_rng(1)
    X = r.uniform(-3, 53, (20, 25)).astype(np.float32)
    Y = r.uniform(-3, 43, (20, 25)).astype(np.float32)
    ref = filters.sample(img, X, Y)
    monkeypatch.setattr(filters, "_ndi", None)
    got = filters.sample(img, X, Y)
    assert got.shape == ref.shape and np.abs(got - ref).max() < 1e-5


def test_smooth1d_fallback_matches_scipy(monkeypatch):
    a = np.random.default_rng(0).random(200).astype(np.float32)
    ref = noise.smooth1d(a, 3.0)
    monkeypatch.setattr(noise, "_scipy_ndimage", lambda: None)
    got = noise.smooth1d(a, 3.0)
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.parametrize("name", ["snow cap", "heat haze"])
def test_elements_draw_without_scipy(monkeypatch, name):
    from pixelcanvas import elements as E
    monkeypatch.setattr(filters, "_ndi", None)
    monkeypatch.setattr(noise, "_scipy_ndimage", lambda: None)
    cv = Canvas(192, 108, "#20242c")
    E.sky(cv, "dusk")
    before = cv.buf.copy()
    kw = {"mask": (cv.xy()[1] > cv.h * 0.5).astype(np.float32)} if name == "snow cap" else {}
    E.element(cv, name, seed=1, **kw)
    assert np.abs(cv.buf - before).max() > 1e-4


# ------------------------------------------------------------------ did-you-mean

MODULES = ["sdf", "color", "noise", "masks", "brush", "filters", "text", "elements", "creature"]


@pytest.mark.parametrize("name", MODULES)
def test_modules_explain_missing_names(name):
    mod = importlib.import_module("pixelcanvas." + name)
    with pytest.raises(AttributeError) as e:
        getattr(mod, "no_such_thing_here")
    assert name in str(e.value) and "no_such_thing_here" in str(e.value)
    assert not hasattr(mod, "__wrapped__") and not hasattr(mod, "_private_probe")


def test_sdf_wrong_guesses_give_the_right_call():
    from pixelcanvas import sdf
    for guess, right in (("rectangle", "sd_box"), ("circle", "sd_circle"), ("abs", "np.abs")):
        with pytest.raises(AttributeError, match=right):
            getattr(sdf, guess)
    with pytest.raises(AttributeError, match="sd_circle"):
        sdf.sd_cirle
    with pytest.raises(AttributeError, match="sd_box"):
        sdf.SdBox


def test_module_getattr_dunders_stay_plain():
    from pixelcanvas import sdf
    ga = _args.module_getattr("pixelcanvas.sdf")
    for n in ("__wrapped__", "__path__", "__deepcopy__", "_ipython_display_"):
        with pytest.raises(AttributeError) as e:
            ga(n)
        assert "did you mean" not in str(e.value)
    assert not hasattr(sdf, "__deepcopy__")


def test_module_getattr_generic():
    mod = type(sys)("pc_fake_kit")
    mod.__all__ = ["draw_leaf", "draw_tree"]
    mod.draw_leaf = mod.draw_tree = lambda: None
    sys.modules["pc_fake_kit"] = mod
    try:
        ga = _args.module_getattr("pc_fake_kit", {"leaf": "use pc_fake_kit.draw_leaf()"})
        with pytest.raises(AttributeError, match=r"draw_tre"):
            ga("draw_tee")
        with pytest.raises(AttributeError, match=r"use pc_fake_kit.draw_leaf"):
            ga("Leaf")
    finally:
        del sys.modules["pc_fake_kit"]


# ------------------------------------------------------------------ files

def _repo_files(exts):
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "outputs", "__pycache__", ".pytest_cache", "gallery", "node_modules"}]
        for f in files:
            if f.endswith(exts):
                yield os.path.join(base, f)


def test_no_byte_order_marks():
    bad = []
    for p in _repo_files((".py", ".md", ".toml", ".txt", ".json", ".yml", ".yaml")):
        with open(p, "rb") as f:
            if f.read(3) == b"\xef\xbb\xbf":
                bad.append(os.path.relpath(p, ROOT))
    assert not bad, bad


def test_no_fromarray_mode_argument():
    pat = re.compile(r"Image\.fromarray\(([^()]|\([^()]*\))*,\s*[\"'][A-Za-z]+[\"']\s*\)")
    bad = []
    owned = {"text", "elements", "canvas", "filters", "vision", "creature", "sdf", "color", "noise", "masks", "brush", "_args"}
    for p in _repo_files((".py",)):
        if os.path.splitext(os.path.basename(p))[0] not in owned or "tests" in p.split(os.sep):
            continue
        txt = open(p, encoding="utf-8", errors="replace").read()
        if pat.search(txt):
            bad.append(os.path.relpath(p, ROOT))
    assert not bad, bad


def test_canvas_save_creates_missing_folder(tmp_path):
    cv = Canvas(8, 8, "#ffffff")
    target = tmp_path / "a" / "b" / "pic.png"
    cv.save(str(target), preview=str(tmp_path / "p" / "q.jpg"))
    assert target.exists() and (tmp_path / "p" / "q.jpg").exists()


def test_noise_mode_gets_did_you_mean():
    with pytest.raises(ValueError, match="did you mean 'ridged'"):
        noise.fbm2d(8, 8, 4, mode="ridge")
    assert noise.fbm2d(8, 8, 4, mode="billow").shape == (8, 8)


def test_tools_print_utf8_on_a_cp950_console():
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="cp950")
    for tool, args in (("styles.py", ["pick", "水墨畫 and 繪本"]), ("styles.py", ["list"]),
                       ("look.py", ["--help"])):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), *args], capture_output=True, env=env,
                           cwd=ROOT, timeout=120)
        assert r.returncode == 0, (tool, r.stderr[-300:])
        r.stdout.decode("utf-8")
