# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.palette and pixelcanvas.backdrop: palettes and roles, colour maths, generators, search,
every backdrop renders, determinism, calm region + ink, box / strength, quiet(), and the reference page snippets."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import palette as PL  # noqa: E402
from pixelcanvas import backdrop as BD  # noqa: E402

DOC = os.path.join(ROOT, "skills", "pixelcanvas", "references", "backdrops.md")


# ------------------------------------------------------------------------------------------------ palette

def test_library_size_and_roles():
    assert 55 <= len(PL.PALETTES) <= 70
    assert set(PL.FAMILIES) == {"mood", "subject", "era", "ui"}
    for name, p in PL.PALETTES.items():
        assert 5 <= len(p) <= 7, name
        assert PL.contrast_ratio(p.fg, p.bg) >= 4.5, name
        assert p.accent in p.colors and p.bg in p.colors
        assert re.fullmatch(r"#[0-9a-f]{6}", p.muted)
        assert PL.srgb_to_oklab(np.array(PL.rgb(p.dark)))[0] <= PL.srgb_to_oklab(np.array(PL.rgb(p.light)))[0]
    for brand in ("indigo", "emerald", "rose", "amber", "slate"):
        assert PL.palette(brand).mode == "light" and PL.palette(brand + "_dark").mode == "dark"


def test_palette_methods():
    p = PL.palette("sunset")
    r = p.ramp(9)
    assert len(r) == 9 and r[0] == p.dark and r[-1] == p.light
    st = p.stops()
    assert st[0][0] == 0 and st[-1][0] == 1 and len(st) == len(p)
    for c in ("#3a6ea5", "#ffffff", "#000000", "#ff8800"):
        assert PL.contrast_ratio(p.text_on(c), c) >= 4.5
    assert isinstance(p.muted, str) and isinstance(p.muted(0.5), PL.Palette)
    lo = PL.srgb_to_oklab(p.array)[:, 0].mean()
    assert PL.srgb_to_oklab(p.lighter().array)[:, 0].mean() > lo > PL.srgb_to_oklab(p.darker().array)[:, 0].mean()
    assert p.shuffle(3).colors == p.shuffle(3).colors and sorted(p.shuffle(3).colors) == sorted(p.colors)
    assert p.inverted().mode == "dark" if p.mode == "light" else "light"
    assert len(PL.palette("ocean", n=3)) == 3 and len(PL.palette("ocean", n=9)) == 9
    q = PL.palette(["#102030", "#f0e0d0", "#e05030", "#3080c0", "#90a0a0"])
    assert q.name == "custom" and q.bg == "#f0e0d0" and q.accent in ("#e05030", "#3080c0")
    assert PL.palette("#3a6ea5").name.startswith("analogous")
    with pytest.raises(KeyError):
        PL.palette("zzzz qqqq")


def test_colour_maths():
    assert abs(PL.contrast_ratio("#000", "#fff") - 21) < 1e-6
    x = np.random.default_rng(0).random((50, 3)).astype(np.float32)
    assert np.allclose(PL.oklab_to_srgb(PL.srgb_to_oklab(x)), x, atol=2e-4)
    lch = PL.oklab_to_oklch(PL.srgb_to_oklab(x))
    assert np.allclose(PL.oklch_to_oklab(lch), PL.srgb_to_oklab(x), atol=1e-5)
    assert PL.mix_oklab("#ff0000", "#0000ff", 0) == "#ff0000" and PL.mix_oklab("#ff0000", "#0000ff", 1) == "#0000ff"
    fixed = PL.ensure_contrast("#8899aa", "#ffffff", 4.5)
    assert PL.contrast_ratio(fixed, "#ffffff") >= 4.5
    assert PL.ensure_contrast("#000000", "#ffffff") == "#000000"
    assert re.fullmatch(r"#[0-9a-f]{6}", PL.oklch(0.7, 0.4, 140))       # out-of-gamut chroma is clipped


def test_generators():
    for kind in ("analogous", "complementary", "split", "triadic", "tetradic", "monochrome"):
        h = PL.harmony("#3a6ea5", kind, n=6)
        assert len(h) == 6 and "#3a6ea5" in h.colors
        assert PL.contrast_ratio(h.fg, h.bg) >= 4.5
    ts = PL.tints_shades("#3a6ea5", 7)
    L = PL.srgb_to_oklab(np.array([PL.rgb(c) for c in ts]))[:, 0]
    assert len(ts) == 7 and np.all(np.diff(L) > 0) and "#3a6ea5" in ts
    img = np.zeros((60, 90, 3), np.float32)
    img[:, :30] = PL.rgb("#d03020")
    img[:, 30:60] = PL.rgb("#20a050")
    img[:, 60:] = PL.rgb("#2040c0")
    got = PL.from_image(img, 3).colors
    for want in ("#d03020", "#20a050", "#2040c0"):
        assert min(np.abs(PL.rgb(g) - PL.rgb(want)).sum() for g in got) < 0.03
    cat = PL.categorical(10)
    assert len(cat) == 10 and cat[:8] == PL.CATEGORICAL["okabe_ito"]
    assert PL.categorical(5, "ibm") == PL.CATEGORICAL["ibm"]


def test_palette_search():
    assert PL.find("warm sunset over the sea") == "sunset"
    assert PL.find("cyberpunk neon") == "cyber"
    assert PL.find("cozy coffee shop") == "coffee"
    assert PL.find("dark indigo dashboard") == "indigo_dark"
    assert PL.find("夕陽") == "sunset"
    assert PL.find("森林") == "forest"
    assert PL.find("夢幻粉彩") == "dreamy"
    assert PL.find("zzzz") is None and PL.find("zzzz", 3) == []
    assert len(PL.find("blue", 3)) == 3
    assert "indigo" in PL.names("ui") and "sunset" in PL.names("warm") and len(PL.names()) == len(PL.PALETTES)
    assert "sunset" in PL.describe("sunset")


# ------------------------------------------------------------------------------------------------ backdrop

@pytest.mark.parametrize("name", list(BD.BACKDROPS))
def test_every_backdrop_renders(name):
    img, P = BD.render(name, 96, 54, seed=2)
    assert img.shape == (54, 96, 3) and img.dtype == np.float32
    assert np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
    assert img.std() > 0.004, "backdrop should not be flat"
    assert isinstance(P, PL.Palette)


def test_registry_and_search():
    assert len(BD.BACKDROPS) >= 50
    assert sum(len(v) for v in BD.FAMILIES.values()) == len(BD.BACKDROPS)
    assert set(BD.names()) == set(BD.BACKDROPS)
    assert "hills" in BD.names("scenic") and "mesh" in BD.names("gradient")
    assert BD.find("soft pastel blobs") in ("aurora", "mesh")
    assert BD.find("retro synthwave grid") == "synthwave"
    assert BD.find("topographic map") == "topographic"
    assert BD.find("glassmorphism card") == "glass"
    assert BD.find("星空") == "starry"
    assert BD.find("等高線") == "topographic"
    assert BD.find("zzzz") is None
    assert "options" in BD.describe("waves") and "hills" in BD.describe()
    with pytest.raises(KeyError):
        BD.names("nope")


def test_deterministic_and_seeded():
    a, _ = BD.render("memphis", 80, 45, seed=4)
    b, _ = BD.render("memphis", 80, 45, seed=4)
    c, _ = BD.render("memphis", 80, 45, seed=5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_resolution_independent():
    big, _ = BD.render("hills", 320, 180, seed=1)
    small, _ = BD.render("hills", 160, 90, seed=1)
    down = big.reshape(90, 2, 160, 2, 3).mean((1, 3))
    assert np.abs(down - small).mean() < 0.03


def test_backdrop_paints_and_reports():
    cv = Canvas(192, 108)
    bd = BD.backdrop(cv, "hills", palette="sunset", seed=1)
    assert bd.name == "hills" and bd.palette.name == "sunset"
    x0, y0, x1, y1 = bd.calm
    assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    assert y1 < 0.75, "the calm region of a landscape is the sky"
    px = bd.calm_px(cv)
    region = cv.buf[px[1]:px[3], px[0]:px[2]].reshape(-1, 3).mean(0)
    assert PL.contrast_ratio(bd.ink, tuple(float(v) for v in region)) >= 4.5
    assert "Backdrop(" in repr(bd) and not bd.busy
    busy = BD.backdrop(Canvas(192, 108), "stripes")
    assert busy.busy and busy.calm == (0.2, 0.25, 0.8, 0.75)
    bd2 = BD.backdrop(Canvas(160, 90), "night sky with stars")
    assert bd2.name == "starry"
    bd3 = BD.backdrop(Canvas(160, 90), "warm sunset hills")
    assert bd3.name == "hills" and bd3.palette.name == "sunset"
    with pytest.raises(TypeError):
        BD.backdrop(Canvas(32, 18), "waves", nope=1)
    with pytest.raises(KeyError):
        BD.backdrop(Canvas(32, 18), "zzzz qqqq")


def test_box_strength_and_custom_palette():
    cv = Canvas(200, 100, "#000000")
    bd = BD.backdrop(cv, "linear", palette=["#ff0000", "#ffffff"], box=(0.5, 0, 1, 1))
    assert np.allclose(cv.buf[:, :99], 0)
    assert cv.buf[:, 101:].mean() > 0.3
    assert bd.calm[0] >= 0.5 - 1e-6
    cv2 = Canvas(100, 50, "#000000")
    BD.backdrop(cv2, "linear", palette=["#ffffff", "#ffffff"], strength=0.5)
    assert abs(cv2.buf.mean() - 0.5) < 0.02


def test_quiet_calms_without_edges():
    cv = Canvas(240, 135)
    BD.backdrop(cv, "memphis", seed=3, clear=0)
    before = cv.buf[40:95, 60:180].std()
    box = BD.quiet(cv, (0.25, 0.3, 0.75, 0.7), amount=0.8)
    assert box == (60, 40, 180, 94)
    assert cv.buf[45:90, 70:170].std() < before * 0.6
    row = cv.buf[67, :, :].mean(-1)
    assert np.abs(np.diff(row)).max() < 0.6


def test_palette_from_colour_words():
    p = PL.palette("calm, sage green, cream")
    assert p.name == "sage cream" and p.mode == "light"
    assert p.bg == "#f6efdd" and "#9caf88" in p.colors
    assert PL.contrast_ratio(p.fg, p.bg) >= 4.5
    q = PL.palette("soft blue and peach")                 # shade words soften the next colour, no KeyError for "and"
    assert any(abs(int(c[1:3], 16) - 0xff) < 8 and c.startswith("#ffc") for c in q.colors) or "#ffcba4" in q.colors
    assert PL.palette("dark navy and gold").mode == "dark"
    assert PL.palette("warm orange sunset").name == "sunset"       # one basic colour + a subject word -> the palette
    assert PL.palette("cream").mode == "light" and len(PL.palette("cream")) >= 4


def test_description_colours_reach_the_backdrop():
    cv = Canvas(160, 90)
    bd = BD.backdrop(cv, "misty mountains at dawn in soft blue and peach", seed=1)
    assert bd.name == "mountains" and "#ffcba4" in bd.palette.colors
    assert BD.find("tasteful abstract background") == "mesh"


def test_mountains_have_sharp_peaks_and_a_warm_horizon():
    img, P = BD.render("mountains", 480, 270, "soft blue and peach", seed=2)
    top = img[:, 240, :]
    assert top[:, 2].mean() > 0.4
    hz = img[int(0.5 * 270) - 12, 10:470].mean(0)                    # horizon glow is warmer than the sky above
    sky = img[8, 10:470].mean(0)
    assert hz[0] - hz[2] > sky[0] - sky[2]
    for n in (2, 3, 6):
        assert BD.render("mountains", 160, 90, None, seed=n, layers=n)[0].shape == (90, 160, 3)


def test_ribbons_light_palette_keeps_a_light_ground():
    img, _ = BD.render("ribbons", 320, 180, "calm", seed=1)
    assert img[2:8, :, :].mean() > 0.6
    dark, _ = BD.render("ribbons", 320, 180, "synthwave", seed=1)
    assert dark[2:8, :, :].mean() < 0.3


def test_calm_box_keeps_a_margin_from_the_canvas_edge():
    for name in ("mesh", "hills", "mountains", "ribbons", "aurora"):
        cv = Canvas(400, 210)
        bd = BD.backdrop(cv, name, seed=1)
        x0, y0, x1, y1 = bd.calm
        m = 0.07 * 210
        assert x0 * 400 >= m - 1 and y0 * 210 >= m - 1 and (1 - x1) * 400 >= m - 1 and (1 - y1) * 210 >= m - 1, name


def test_docs_snippets_run(tmp_path, monkeypatch):
    text = open(DOC, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) >= 3
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    for code in blocks:
        code = code.replace("Canvas(2560, 1440", "Canvas(320, 180").replace("Canvas(1920, 1080", "Canvas(320, 180").replace("Canvas(1080, 1350", "Canvas(216, 270")
        exec(compile(code, "backdrops.md", "exec"), {"__name__": "__doc__"})


def test_strict_options_alias_and_names():
    cv = Canvas(120, 80)
    with pytest.raises(TypeError, match="did you mean 'grain'"):
        BD.backdrop(cv, "mesh", grian=0.1)
    BD.backdrop(cv, "paper", tint=None)
    with pytest.raises(ValueError, match="did you mean 'hills'"):
        BD.backdrop(cv, "hils")
    with pytest.raises(ValueError, match="did you mean 'scenic'"):
        BD.names("scenik")
    with pytest.raises(ValueError, match="did you mean 'sunset'"):
        PL.palette("sunst")
    with pytest.raises(ValueError, match="'analogous'"):
        PL.harmony("#336699", "analogus")
    assert PL.Palette(["#112233", "#ddeeff"], name="x").name == "x"


def test_backdrop_is_recorded_in_the_manifest():
    cv = Canvas(120, 80)
    BD.backdrop(cv, "hills")
    assert [e["name"] for e in cv.manifest] == ["backdrop: hills"]
    assert cv.manifest[0]["meta"]["backdrop"] is True


def test_warm_paper_is_one_call():
    import time
    for q in ("warm_paper", "cream paper", "parchment", "warm paper-like background"):
        assert BD._resolve(q)[0] == "warm_paper", q
    cv = Canvas(540, 675)
    t = time.time()
    bd = BD.backdrop(cv, "warm_paper", seed=2)
    assert time.time() - t < 3.0
    mean = cv.buf.reshape(-1, 3).mean(0)
    assert mean[0] > mean[1] > mean[2] and mean[2] > 0.6        # cream: warm and light
    assert cv.buf[: cv.h // 8].mean() > cv.buf[-cv.h // 8:].mean()    # lighter at the top, warmer edges below
    assert float(cv.buf[cv.h // 3:cv.h // 2, cv.w // 3:cv.w // 2].std()) > 0.002     # fibres, not a flat fill
    assert bd.busy is False and bd.ink


# ------------------------------------------------------------------------------------------------ seams

def _seam_ratio(img):
    """Strongest vertical seam in the lower third: horizontal contrast (scale ~ w/80 px) present in at least half of the
    rows of a column, relative to the typical column."""
    k = max(12, img.shape[1] // 160)
    lum = img.mean(-1).astype(np.float64)[int(0.7 * img.shape[0]):]
    v = np.percentile(np.abs(lum[:, 2 * k:] - lum[:, :-2 * k]), 50, axis=0)
    return v.max() / (np.median(v) + 1e-4)


@pytest.mark.parametrize("size", [(1920, 1080), (2560, 1440)])
@pytest.mark.parametrize("seed", [1, 2])
def test_dunes_have_no_vertical_seams(size, seed):
    img, _ = BD.render("dunes", size[0], size[1], seed=seed)
    assert _seam_ratio(img) < 4.0


@pytest.mark.parametrize("name", ["hills", "clouds"])
def test_smooth_scenic_backdrops_have_no_vertical_seams(name):
    img, _ = BD.render(name, 1920, 1080, seed=1)
    assert _seam_ratio(img) < 4.0
