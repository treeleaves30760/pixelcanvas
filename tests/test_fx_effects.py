"""fx.py: finishing effects are deterministic, region-limited, undoable, reference-free and score-guided."""
import numpy as np
import pytest
from PIL import Image

from pixelcanvas import fx
from pixelcanvas.project import Project

from .finish_helpers import cwd_module, new_project, ok, run, synthetic_picture  # noqa: F401

pytestmark = pytest.mark.usefixtures("cwd_module")


def _canvas(seed=0, w=96, h=72):
    rng = np.random.default_rng(seed)
    base = synthetic_picture(w, h).astype(np.float32) / 255.0
    return np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1).astype(np.float32)


def _hf(a):
    return float(np.abs(a - np.roll(a, 1, 1)).mean())


# ---------------------------------------------------------------------------------------------- the effect functions

def test_effects_are_deterministic_and_seeded():
    img = _canvas()
    for name in fx.EFFECTS:
        a = fx.apply_effect(name, img, 0.6, seed=3)
        b = fx.apply_effect(name, img, 0.6, seed=3)
        assert np.array_equal(a, b), name
        assert a.shape == img.shape and a.dtype == np.float32
        assert 0.0 <= a.min() and a.max() <= 1.0
    g1 = fx.apply_effect("grain", img, 0.6, seed=1)
    g2 = fx.apply_effect("grain", img, 0.6, seed=2)
    assert not np.array_equal(g1, g2)
    p1 = fx.apply_effect("paper", img, 0.6, seed=1)
    p2 = fx.apply_effect("paper", img, 0.6, seed=2)
    assert not np.array_equal(p1, p2)


def test_each_effect_does_what_its_name_says():
    img = _canvas()
    H, W = img.shape[:2]
    mean = lambda a: float(a.mean())  # noqa: E731
    v = fx.apply_effect("vignette", img, 0.8)
    corner, centre = (slice(0, 8), slice(0, 8)), (slice(H // 2 - 4, H // 2 + 4), slice(W // 2 - 4, W // 2 + 4))
    assert mean(v[corner]) < 0.6 * mean(img[corner]) and np.allclose(v[centre], img[centre], atol=0.02)
    b = fx.apply_effect("blur", img, 0.8)
    assert _hf(b) < 0.7 * _hf(img)
    s = fx.apply_effect("sharpen", img, 0.8)
    assert _hf(s) > 1.2 * _hf(img)
    g = fx.apply_effect("grain", img, 0.8, seed=1)
    assert float((g - img).std()) > 0.02
    glow = fx.apply_effect("glow", img, 0.8)
    assert mean(glow) > mean(img) and (glow >= img - 1e-6).all()
    paper = fx.apply_effect("paper", img, 0.8)
    assert paper[..., 2].mean() < img[..., 2].mean()                            # warm tone
    assert float(((paper - img).std())) > 0.005
    dull = np.clip(0.3 + 0.4 * img, 0, 1).astype(np.float32)
    lv = fx.apply_effect("levels", dull, 1.0)
    assert lv.min() < dull.min() - 0.05 and lv.max() > dull.max() + 0.05
    assert np.allclose(fx.apply_effect("levels", dull, 0.0), dull, atol=1e-6)


def test_fx_layer_reproduces_the_target():
    below = _canvas(1)
    for name in ("glow", "grain", "vignette", "blur", "paper"):
        target = fx.apply_effect(name, below, 0.7, seed=2)
        layer = fx.to_fx_layer(below, target)
        a = layer[..., 3:4]
        back = layer[..., :3] * a + below * (1 - a)
        assert float(np.abs(back - target).max()) < 2.5 / 255, name


def test_effects_never_see_the_reference():
    import inspect
    assert "ref" not in inspect.signature(fx.apply_effect).parameters
    src = inspect.getsource(fx.apply_effect)
    assert "p.ref" not in src and ".ref" not in src


# ---------------------------------------------------------------------------------------------- the command

def _flat_project(root, name="pic", ref=None):
    proj = new_project(root, name, "120x90", ref=ref)
    ok("fill", "500,500", "--color", "#c8a070", "--layer", "flat", "--bounds", "canvas")
    ok("shape", "ellipse", "300,300,700,700", "--color", "#e8e0c8", "--layer", "paint")
    return proj


def test_fx_lists_the_effects_without_arguments(tmp_path):
    _flat_project(tmp_path)
    out = ok("fx")
    for n in fx.EFFECTS:
        assert f"fx {n}" in out


def test_fx_goes_on_the_fx_layer_and_is_undoable(tmp_path):
    proj = _flat_project(tmp_path)
    p = Project(proj)
    before = p.composite_u8("export")
    fx_before = (proj / "layers" / "fx.png").read_bytes()
    out = ok("fx", "vignette", "--amount", "0.8")
    assert "(layer fx)" in out and "fx vignette all --amount 0.8" in out
    p = Project(proj)
    after = p.composite_u8("export")
    assert after[0, 0].sum() < before[0, 0].sum()                      # corner darker
    assert not p.layer_entry("fx").get("empty")
    # the layer reproduces the effect: composite through fx equals the effect on the picture below
    below = p.composite("score", upto="fx")
    want = fx.apply_effect("vignette", below, 0.8)
    assert np.abs(p.composite("score", upto="text") - want).max() < 3 / 255
    out = ok("undo")
    assert (proj / "layers" / "fx.png").read_bytes() == fx_before
    assert np.array_equal(Project(proj).composite_u8("export"), before)
    ok("redo")
    assert np.array_equal(Project(proj).composite_u8("export"), after)


def test_fx_limited_to_a_region(tmp_path):
    proj = _flat_project(tmp_path)
    ok("region", "add", "left", "--box", "0,0,500,1000", "--no-refine")
    p = Project(proj)
    before = p.composite_u8("export")
    ok("fx", "blur", "left", "--amount", "0.9")
    ok("fx", "grain", "left", "--amount", "0.9")
    after = Project(proj).composite_u8("export")
    W = after.shape[1]
    assert np.array_equal(after[:, int(W * 0.55):], before[:, int(W * 0.55):])        # right side untouched
    assert not np.array_equal(after[:, : int(W * 0.4)], before[:, : int(W * 0.4)])
    # a box works as the region too
    ok("fx", "vignette", "600,0,1000,500", "--amount", "0.9")


def test_fx_is_deterministic_across_projects(tmp_path):
    outs = []
    for k in range(2):
        d = tmp_path / f"d{k}"
        d.mkdir()
        proj = _flat_project(d, f"p{k}")
        ok("fx", "grain", "--amount", "0.5", "--seed", "4")
        outs.append((proj / "layers" / "fx.png").read_bytes())
    assert outs[0] == outs[1]


def test_explicit_amount_ignores_the_reference(tmp_path):
    layers = []
    for k, ref in enumerate((synthetic_picture(120, 90), 255 - synthetic_picture(120, 90))):
        d = tmp_path / f"r{k}"
        d.mkdir()
        proj = _flat_project(d, f"p{k}", ref=ref)
        ok("fx", "paper", "--amount", "0.6")
        layers.append((proj / "layers" / "fx.png").read_bytes())
    assert layers[0] == layers[1]                                       # no reference pixel enters the effect


def test_without_amount_the_score_picks_it(tmp_path):
    # the reference is the canvas with a vignette baked in: a vignette is what moves the score up
    base = np.zeros((90, 120, 3), np.float32)
    base[:] = (200, 160, 112)
    yy, xx = np.mgrid[0:90, 0:120].astype(np.float32)
    d = np.sqrt(((xx - 59.5) / 60) ** 2 + ((yy - 44.5) / 45) ** 2) / np.sqrt(2)
    t = np.clip((d - 0.35) / 0.65, 0, 1)
    ref = np.clip(base * (1 - 0.75 * 0.6 * (t * t * (3 - 2 * t)))[..., None], 0, 255).astype(np.uint8)
    proj = new_project(tmp_path, "pic", ref=ref)
    ok("fill", "500,500", "--color", "#c8a070", "--layer", "flat", "--bounds", "canvas")
    p = Project(proj)
    out = ok("fx", "vignette")
    assert "chosen by score" in out, out
    line = [ln for ln in out.splitlines() if ln.startswith("whole") or ln.startswith("all:")][0]
    import re
    nums = re.findall(r"overall (\d+) -> (\d+)", line)
    assert nums and int(nums[0][1]) > int(nums[0][0]), out
    amount = float(re.search(r"--amount ([0-9.]+)", out).group(1))
    assert 0.4 <= amount <= 0.8


def test_a_harmful_effect_says_so_and_suggests_undo(tmp_path):
    ref = np.zeros((90, 120, 3), np.uint8)
    ref[:] = (200, 160, 112)                                     # a flat picture the canvas already matches
    new_project(tmp_path, "pic", ref=ref)
    ok("fill", "500,500", "--color", "#c8a070", "--layer", "flat", "--bounds", "canvas")
    code, out = run("fx", "vignette", "--amount", "1")
    assert code == 0 and "WARNING" in out and out.splitlines()[-1].startswith("NEXT: pixelcanvas undo"), out
    ok("undo")
    out = ok("fx", "vignette")                                   # auto: nothing helps a picture that is right
    assert "no amount raised the score" in out and "WARNING" in out, out
    assert out.splitlines()[-1].startswith("NEXT: pixelcanvas undo"), out


def test_usage_errors_show_a_corrected_example(tmp_path):
    _flat_project(tmp_path)
    code, out = run("fx", "gloww")
    assert code == 2 and "did you mean glow" in out and "EXAMPLE: pixelcanvas fx glow --amount" in out
    code, out = run("fx", "grain", "--amount", "7")
    assert code == 0 and "note:" in out                         # lenient: clamped with a note
    code, out = run("fx", "grain", "--strength")
    assert code == 2 and "needs a value" in out
    code, out = run("fx", "grain", "nowhere")
    assert code == 2 and "unknown region" in out and "EXAMPLE" in out
    code, out = run("fx", "bloom")
    assert code == 0 and "fx glow" in out                       # alias


def test_fx_in_a_batch_and_in_the_journal(tmp_path):
    proj = _flat_project(tmp_path)
    out = ok("do", "fx grain --amount 0.3; fx vignette --amount 0.4")
    assert "[1]" in out and "[2]" in out
    p = Project(proj)
    cmds = [j.get("cmd", "") for j in p.journal_lines(6)]
    assert any(c.startswith("pixelcanvas fx grain") for c in cmds) and any(c.startswith("pixelcanvas fx vignette") for c in cmds)
    ok("fx", "paper", "--amount", "0.3")                       # fx on top of earlier fx keeps them
    below = p.composite("score", upto="fx")
    p2 = Project(proj)
    assert not np.array_equal(p2.composite_u8("export"), np.asarray(Image.fromarray(
        (np.clip(below, 0, 1) * 255 + 0.5).astype(np.uint8))))


def test_finish_stage_hints_mention_the_finishing_commands(tmp_path):
    new_project(tmp_path, "pic", "120x90")
    for _ in range(5):
        ok("stage", "done", "--because", "test")
    out = ok("next")
    # a project without a reference follows the no-reference workflow (workflow_draw), whose finish hint differs
    assert "finish" in out
    ok("look", "--show", "diff")
    out = ok("next")
    assert out.splitlines()[-1].startswith("NEXT: pixelcanvas polish")     # the finishing pass comes before the export
    ok("polish")
    out = ok("next")
    assert out.splitlines()[-1].startswith("NEXT: pixelcanvas export final.png")
    out = ok("help", "fx")
    assert "fx <glow|grain" in out and "example: pixelcanvas fx grain --amount 0.4" in out and len(out.splitlines()) <= 25
