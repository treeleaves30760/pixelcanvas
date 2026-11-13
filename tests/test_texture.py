# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.texture: every texture renders, determinism, masks, uv, search, building blocks, docs."""
import importlib
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import texture as T  # noqa: E402

DOC = os.path.join(ROOT, "skills", "pixelcanvas", "references", "textures.md")


@pytest.mark.parametrize("name", list(T.TEXTURES))
def test_every_texture_renders(name):
    a = T.texture(name, 48, 64, seed=1, scale=3.0)
    assert a.shape == (48, 64, 3) and a.dtype == np.float32
    assert np.isfinite(a).all()
    assert a.min() > -0.05 and a.max() < 12
    assert a.std() > 0.002, "texture should not be flat"


def test_deterministic_and_seeded():
    a = T.texture("cobblestone", 40, 40, seed=2)
    assert np.array_equal(a, T.texture("cobblestone", 40, 40, seed=2))
    assert not np.array_equal(a, T.texture("cobblestone", 40, 40, seed=3))


def test_registry_and_search():
    assert len(T.TEXTURES) >= 75 and set(T.names()) == set(T.TEXTURES)
    assert sum(len(v) for v in T.FAMILIES.values()) == len(T.TEXTURES)
    assert "planks" in T.names("wood")
    assert T.find("wooden floor") == "planks"
    assert T.find("red brick wall") == "brick"
    assert T.find("cat fur") == "fur"
    assert T.find("swimming pool") == "caustics"
    assert T.find("jeans") == "denim"
    assert T.find("plaid") == "tartan"
    assert T.find("campfire") == "fire"
    assert T.find("tiles", 3)[0] == "tiles"
    assert T.find("zzzz") is None
    assert "brick" in T.describe() and "bond" in T.describe("brick")
    with pytest.raises(KeyError):
        T.texture("zzzz", 8, 8)


def test_mask_box_matches_full_texture():
    full = T.texture("brick", 60, 80, seed=4)
    mask = np.zeros((60, 80), np.float32)
    mask[10:30, 20:50] = 1
    part = T.texture("brick", 60, 80, seed=4, mask=mask)
    assert np.allclose(part[12:28, 22:48], full[12:28, 22:48], atol=1e-4)
    assert np.abs(part[45:, :]).max() == 0


def test_uv_grid_reproduces_pixel_grid():
    h, w = 54, 72
    S = h / 1080
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    a = T.texture("flagstone", h, w, seed=1)
    b = T.texture("flagstone", h, w, seed=1, uv=(X / S, Y / S))
    assert np.abs(a - b).mean() < 2e-3


def test_scale_is_resolution_independent():
    small = T.texture("planks", 54, 96, seed=5)
    big = T.texture("planks", 216, 384, seed=5)
    down = big.reshape(54, 4, 96, 4, 3).mean(axis=(1, 3))
    assert np.abs(small - down).mean() < 0.06


def test_maps_and_alpha():
    m = T.texture("perforated", 64, 64, maps=True, scale=3)
    assert isinstance(m, T.Maps) and m.alpha is not None and 0.2 < m.alpha.mean() < 0.95
    assert m.height.shape == (64, 64) and m.lit.shape == (64, 64, 3)
    lava = T.texture("lava", 64, 64, maps=True)
    assert lava.emit is not None and lava.emit.max() > 1.0                      # HDR glow
    flat = T.texture("brick", 32, 32, bump=0, maps=True)
    assert np.allclose(flat.lit, flat.color)


def test_fill_paints_only_inside_mask():
    cv = Canvas(80, 60, "#000000")
    mask = np.zeros((60, 80), np.float32)
    mask[:, 40:] = 1
    T.fill(cv, mask, "wooden floor")
    assert cv.buf[:, :38].max() == 0 and cv.buf[:, 42:].mean() > 0.1
    cv2 = Canvas(80, 60, "#808080")
    T.fill(cv2, 1 - mask, "gravel", shadow=0.8)
    assert cv2.buf[:, 40:44].mean() < 0.49                         # contact shadow falls away from the light


def test_cellular_properties():
    Y, X = np.mgrid[0:64, 0:64].astype(np.float32) / 8
    c = T.cellular(X, Y, jitter=1.0, seed=3)
    assert (c.f1 <= c.f2 + 1e-6).all() and (c.edge >= 0).all() and c.edge.max() < 1.0
    assert 0 <= c.rand.min() and c.rand.max() < 1
    assert len(np.unique(c.id)) > 30                                             # many cells
    r = T.cellular(X, Y, jitter=1.0, seed=3, round=0.2)
    assert (r.edge <= c.edge + 1e-6).all()                                       # rounded corners shrink inwards
    col = T.voronoi_shade(c)
    assert col.shape == (64, 64, 3)


def test_strokes_relief_patches_and_uv_helpers():
    s = T.strokes(64, 64, angle=-90, length=12, width=2, scale=2)
    assert 0 <= s.cov.min() and s.cov.max() <= 1 and s.cov.mean() > 0.3
    assert s.along.max() <= 1 + 1e-6
    flow = np.full((64, 64), 45.0, np.float32)
    assert T.texture("fur", 64, 64, flow=flow).shape == (64, 64, 3)
    assert np.allclose(T.relief(np.zeros((16, 16))), 1.0)
    lit = T.relief(np.random.default_rng(0).random((32, 32)), color="#808080")
    assert lit.shape == (32, 32, 3)
    p = T.patches(64, 64, coverage=0.4, seed=1)
    assert 0.1 < p.mean() < 0.7
    U, V = T.plane_uv(100, 120, horizon=0.4)
    assert np.all(np.diff(V[60:, 0]) > 0) and abs(V[-1, 0]) < 0.05 * abs(V[45, 0])
    du = np.abs(np.diff(U[70:, :], axis=1))
    assert du[-1].mean() < du[0].mean()                                          # nearer rows → finer texture
    U, V = T.polar_uv(64, 64, 32, 32, radius=10)
    assert U.shape == (64, 64) and abs(V[32, 42]) < 1.5
    cv = Canvas(64, 64, "#909090")
    T.weather(cv, amount=1.0)
    assert cv.buf.mean() < 0.565


def _doc_blocks():
    with open(DOC, encoding="utf-8") as f:
        text = f.read()
    return text, re.findall(r"```python\n(.*?)```", text, re.S)


@pytest.mark.parametrize("i", range(len(_doc_blocks()[1])))
def test_reference_snippets_run(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = {}
    code = _doc_blocks()[1][i]
    assert "W, H = 960, 540" in code                               # every snippet is sized through W, H
    exec(code.replace("W, H = 960, 540", "W, H = 320, 180"), g)
    buf = g["cv"].buf
    assert np.isfinite(buf).all() and buf.std() > 0.01


def test_reference_mentions_every_texture():
    text = _doc_blocks()[0]
    missing = [n for n in T.TEXTURES if f"`{n}`" not in text]
    assert not missing, missing
    assert len(text.splitlines()) < 250


@pytest.mark.parametrize("name", ["texture_swatches", "texture_wall"])
def test_example_scenes_render(name):
    cv = importlib.import_module(f"scenes.{name}").render(240, 135)
    assert cv.shape == (135, 240, 3) and np.isfinite(cv.buf).all() and 0.03 < cv.buf.mean() < 0.8


def test_unknown_option_names_the_valid_ones():
    with pytest.raises(TypeError) as e:
        T.texture("crackle_glaze", 32, 32, palette=["#fff"])
    assert "color" in str(e.value) and "size" in str(e.value)


def test_describe_lists_options():
    assert "Options:" in T.describe("galvanized") and "size=" in T.describe("galvanized")
    assert "tint" in T.describe("brick")


def test_tint_recolours_and_keeps_structure():
    base = T.texture("brick", 64, 64, seed=1)
    blue = T.texture("brick", 64, 64, seed=1, tint="#3060c0")
    assert blue[..., 2].mean() > blue[..., 0].mean() * 1.5 and base[..., 0].mean() > base[..., 2].mean()
    lb, lt = base.mean(-1).ravel(), blue.mean(-1).ravel()
    assert np.corrcoef(lb, lt)[0, 1] > 0.9
    half = T.texture("brick", 64, 64, seed=1, tint=("#3060c0", 0.0))
    assert np.allclose(half, base, atol=1e-5)


def test_strokes_cull_matches_full_evaluation():
    import pixelcanvas.texture as mod
    a = mod.texture("lawn", 60, 80, seed=2)
    orig = mod._blade

    def no_cull(*args, **kw):
        f = orig(*args, **kw)
        f.cullable = False
        return f
    mod._blade = no_cull
    try:
        b = mod.texture("lawn", 60, 80, seed=2)
    finally:
        mod._blade = orig
    assert np.allclose(a, b)


def test_strict_alias_and_unknown_names():
    a = T.texture("crackle_glaze", 32, 32, color="#336699")
    assert np.allclose(a, T.texture("crackle_glaze", 32, 32, colour="#336699"))
    with pytest.raises(TypeError, match="did you mean 'size'"):
        T.texture("crackle_glaze", 32, 32, sise=3)
    with pytest.raises(ValueError, match="did you mean 'brick'"):
        T.texture("brik", 8, 8)
    with pytest.raises(ValueError, match="'running'"):
        T.texture("brick", 16, 16, bond="runing")
    with pytest.raises(TypeError, match="unknown option"):
        T.fill(Canvas(32, 32), np.ones((32, 32), np.float32), "brick", nope=1)
    with pytest.raises(TypeError, match="unknown option 'opacty'"):
        T.weather(Canvas(32, 32), opacty=1)


# ---- round 1 fixes: paper is a light warm sheet
from pixelcanvas.texture import texture  # noqa: E402

def test_paper_is_a_cream_sheet_with_fine_grain():
    a = texture("paper", 270, 480, seed=1)
    m = a.reshape(-1, 3).mean(0)
    assert m[0] > 0.9 and m[0] > m[1] > m[2] and m[2] > 0.78                 # light, warm, not grey
    assert 0.004 < a.std(axis=(0, 1)).max() < 0.03                                              # grain, no blotches or bumps
    assert np.abs(np.diff(a.mean(-1), axis=1)).mean() < 0.01


def test_paper_takes_color_and_paints_straight_onto_a_canvas():
    from pixelcanvas.canvas import Canvas
    a = texture("paper", 100, 160, color="#c8d8f0", seed=2)
    m = a.reshape(-1, 3).mean(0)
    assert m[2] > m[0]
    cv = Canvas(160, 100)
    cv.paint(texture("paper", 100, 160, seed=1))
    assert cv.buf.mean() > 0.85


def test_lined_and_graph_paper_take_the_sheet_colour():
    for name, kw in (("lined_paper", dict(color="#fff7c8")), ("graph_paper", dict(paper="#fff7c8"))):
        a = texture(name, 200, 300, seed=1, **kw)
        assert a[:4, :4].mean(0).mean(0)[2] < a[:4, :4].mean(0).mean(0)[0]
    default = texture("lined_paper", 200, 300, seed=1)
    assert default.reshape(-1, 3).mean(0)[0] > 0.9
