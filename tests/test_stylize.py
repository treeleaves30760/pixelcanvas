# -*- coding: utf-8 -*-
"""Media filters: shapes, ranges, and that each one actually changes the look in its characteristic way."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, stylize, sdf  # noqa: E402
from pixelcanvas.color import luma  # noqa: E402


def _scene():
    cv = Canvas(160, 120)
    cv.gradient([(0, "#4070c0"), (1, "#c0e0ff")])
    X, Y = cv.xy()
    cv.paint("#30a040", np.clip(Y - 80 + 0.5, 0, 1))
    cv.paint("#d03020", sdf.fill(sdf.sd_circle(X, Y, 80, 60, 25)))
    return cv


@pytest.mark.parametrize("fn", [stylize.oil_paint, stylize.watercolor, stylize.pencil_sketch, stylize.cel_shade,
                                stylize.canvas_texture])
def test_media_shape_and_range(fn):
    src = _scene().buf
    out = fn(src)
    assert out.shape == src.shape and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    assert np.abs(out - src).mean() > 0.003                         # it did something


def test_kuwahara_preserves_edges_and_flattens_noise():
    rng = np.random.default_rng(0)
    img = np.zeros((60, 60, 3), np.float32)
    img[:, 30:] = 1.0
    noisy = np.clip(img + rng.normal(0, 0.1, img.shape).astype(np.float32), 0, 1)
    k = stylize.kuwahara(noisy, 3)
    assert k[:, 5:25].std() < noisy[:, 5:25].std() * 0.6            # flat areas get flatter
    assert k[30, 27].max() < 0.3 and k[30, 33].min() > 0.7                      # the edge stays sharp


def test_pencil_is_grey_and_mostly_paper():
    out = stylize.pencil_sketch(_scene().buf)
    assert np.abs(out[..., 0] - out[..., 2]).max() < 0.06            # no colour (paper tint only)
    assert luma(out).mean() > 0.6                                    # mostly light paper


def test_watercolor_lighter_with_white_border():
    src = _scene().buf
    out = stylize.watercolor(src)
    assert luma(out).mean() > luma(src).mean()                       # transparent washes
    assert out[1, 1].min() > 0.9                                     # unpainted paper at the edge


def test_cel_shade_quantises():
    out = stylize.cel_shade(_scene().buf, levels=3, outline=0)
    assert len(np.unique(np.round(luma(out)[60:100, 60:100], 2))) < 40


def test_flow_field_runs_along_edges():
    img = np.zeros((40, 40, 3), np.float32)
    img[:, 20:] = 1.0                                                # vertical edge → strokes run vertically
    vx, vy, coh = stylize.flow_field(img, 2.0)
    assert abs(vy[20, 20]) > 0.9 and coh[20, 20] > 0.5


# ------------------------------------------------------------------------------------------- style library

import re  # noqa: E402

from pixelcanvas.stylize import STYLES, MEDIA, apply, names, pick, describe  # noqa: E402

DOC = os.path.join(ROOT, "skills", "pixelcanvas", "references", "styles.md")


def _small():
    cv = Canvas(128, 72)
    cv.gradient([(0, "#1d3c6e"), (0.6, "#e6a86c"), (1, "#3a2c22")])
    X, Y = cv.xy()
    cv.paint("#2f5a2a", np.clip(Y - 48 - 6 * np.sin(X / 9) + 0.5, 0, 1))
    cv.paint("#c0392b", sdf.fill(sdf.sd_circle(X, Y, 80, 34, 14)))
    return cv


@pytest.fixture(scope="module")
def small():
    return _small().buf


@pytest.mark.parametrize("name", names())
def test_every_style_runs_and_is_deterministic(name, small):
    a = apply(small, name, seed=3)
    b = apply(small, name, seed=3)
    assert a.shape == small.shape and a.dtype == np.float32
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert np.abs(a - small).mean() > 0.01                           # it visibly changed the picture
    assert np.array_equal(a, b)                                      # same seed, same pixels
    assert describe(name)


def test_registry_shape():
    assert len(names()) >= 40
    for n, st in STYLES.items():
        assert callable(st["fn"]) and isinstance(st["defaults"], dict)
        kws = st["keywords"]
        assert any(all(ord(c) < 128 for c in k) for k in kws), n                     # English
        assert any(any(ord(c) > 0x3000 for c in k) for k in kws), n                  # zh / ja


def test_media_mapping_keeps_old_keys_and_grows():
    for k in ("oil_paint", "watercolor", "pencil_sketch", "cel_shade"):
        assert k in MEDIA and callable(getattr(stylize, k))
    assert len(MEDIA) >= 30
    for fn_name in MEDIA:
        assert callable(getattr(stylize, fn_name))


@pytest.mark.parametrize("text, want", [
    ("in the style of a ukiyo-e print", "ukiyoe"),
    ("an ink wash painting of bamboo", "ink_wash"),
    ("pen and ink drawing of a castle", "pen_ink"),
    ("a stained glass window of a lion", "stained_glass"),
    ("retro 8-bit pixel art spaceship", "pixel_art"),
    ("risograph zine cover", "risograph"),
    ("pop art portrait with halftone dots", "comic"),
    ("Van Gogh style oil painting", "oil"),
    ("a watercolour of tulips", "watercolor"),
    ("cross-stitch sampler", "cross_stitch"),
    ("photorealistic product shot", "photo_finish"),
    ("水墨畫的山水", "ink_wash"),
    ("浮世繪風格的海浪", "ukiyoe"),
    ("像素風的城市", "pixel_art"),
    ("彩色鉛筆畫的貓", "colored_pencil"),
    ("ステンドグラスの窓", "stained_glass"),
    ("ドット絵のドラゴン", "pixel_art"),
    ("水彩画の花", "watercolor"),
    ("切り絵の森", "paper_cut"),
    ("a cozy cabin at dusk", None),
    ("a pink flamingo", None),
])
def test_pick(text, want):
    assert pick(text) == want


def test_apply_on_canvas_in_place_and_strength():
    cv = _small()
    src = cv.buf.copy()
    assert apply(cv, "duotone") is cv and not np.allclose(cv.buf, src)
    assert np.allclose(apply(src, "noir", strength=0.0), src, atol=1e-6)
    with pytest.raises(KeyError):
        apply(src, "no_such_style")


def test_sizes_scale_with_height():
    for h in (216, 432, 1080):
        img = _small().resize(h * 16 // 9, h).buf
        out = apply(img, "pixel_art", dither=0)
        p = max(int(round(12 * h / 1080)), 1)                       # 12 px blocks at 1080p
        blocks = np.repeat(np.repeat(out[::p, ::p], p, 0), p, 1)[:h, :out.shape[1]]
        assert np.array_equal(blocks, out)


def test_sticker_border_is_white_outside_mask():
    cv = Canvas(160, 90, "#203040")
    X, Y = cv.xy()
    m = sdf.fill(sdf.sd_circle(X, Y, 80, 45, 20))
    out = apply(cv.buf, "sticker", mask=m, border=80)
    assert out[45, 80 + 20 + 3].min() > 0.9                          # just outside the subject: white border
    assert out[45, 80].max() < 0.4                                   # the subject is kept


def test_structure_styles_use_regions():
    img = _small().buf
    lp = apply(img, "low_poly", cell=200)
    assert len(np.unique(np.round(lp[10:60:3, 10:120:3], 2).reshape(-1, 3), axis=0)) < 120   # flat facets
    sg = apply(img, "stained_glass")
    assert (luma(sg) < 0.15).mean() > 0.05                           # dark lead cames


def test_styles_doc_code_blocks_run():
    text = open(DOC, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) >= 4
    for code in blocks:
        exec(compile(code, "styles.md", "exec"), {})


def test_styles_cli_list_and_pick(capsys):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import styles as cli
    cli.main(["list"])
    cli.main(["pick", "a linocut of a heron"])
    out = capsys.readouterr().out
    assert "ukiyoe" in out and out.strip().endswith("linocut")


# ------------------------------------------------------------------------------------------- review fixes

def _ball_on_wall(h=270, w=480):
    """A small red ball with a white highlight on a big dark wall (the classic case that loses small colours)."""
    cv = Canvas(w, h, "#3a3550")
    X, Y = cv.xy()
    cv.paint("#d23a2e", sdf.fill(sdf.sd_circle(X, Y, w * 0.5, h * 0.5, h * 0.12)))
    cv.paint("#ffffff", sdf.fill(sdf.sd_circle(X, Y, w * 0.5 - h * 0.04, h * 0.46, h * 0.03)))
    return cv.buf


def test_pixel_art_palette_keeps_small_highlight_and_draws_outline():
    out = stylize.pixel_art(_ball_on_wall(), pixel=6, colors=6, dither=0)
    assert out.max(-1).max() > 0.9                                       # the tiny white highlight survives
    assert luma(out).min() < luma(_ball_on_wall()).min() - 0.05            # a dark sprite outline appears


def test_pixel_art_palette_preset_limits_colours():
    out = stylize.pixel_art(_ball_on_wall(), pixel=6, palette="gameboy")
    cols = {tuple(c) for c in np.round(out.reshape(-1, 3) * 255).astype(int)}
    allowed = {tuple(int(round(v * 255)) for v in stylize.rgb(c)) for c in stylize.PALETTES["gameboy"]}
    assert cols <= allowed


def test_paper_cut_keeps_equal_luma_colours_apart():
    cv = Canvas(320, 180, "#201030")
    X, Y = cv.xy()
    cv.paint("#ffcc33", sdf.fill(sdf.sd_circle(X, Y, 100, 90, 40)))      # yellow and pink of similar brightness
    cv.paint("#ff66cc", sdf.fill(sdf.sd_circle(X, Y, 230, 90, 40)))
    out = stylize.paper_cut(cv.buf)
    y, p = out[90, 100], out[90, 230]
    assert y[1] - y[2] > 0.25 and p[2] - p[1] > 0.15                     # each keeps its own hue


def test_pencil_lifts_dark_backgrounds_to_paper():
    out = apply(_ball_on_wall(), "pencil")
    assert luma(out)[10:60, 10:120].mean() > 0.55                         # dark wall reads as light paper
    assert luma(out).min() > 0.1                                          # graphite, not pure black


def test_clay_keeps_one_piece_for_a_shaded_ball():
    cv = Canvas(320, 180, "#6a6080")
    X, Y = cv.xy()
    m = sdf.fill(sdf.sd_circle(X, Y, 160, 90, 55))
    shade = np.clip(1.1 - (X - 120) / 110, 0.3, 1.0)
    cv.paint(np.stack([0.82 * shade, 0.2 * shade, 0.16 * shade], -1).astype(np.float32), m)
    out = stylize.clay(cv.buf)
    row = out[90, 120:200]
    assert (row[:, 0] > row[:, 1] + 0.1).all()                          # red across the ball, lit to shadow
    assert luma(row[:10]).mean() > luma(row[-10:]).mean()                 # the form's light is kept


def test_strict_options_names_and_prompt():
    cv = Canvas(64, 48, "#406080")
    with pytest.raises(TypeError, match="did you mean 'brush'"):
        stylize.apply(cv, "oil", brsh=3)
    with pytest.raises(ValueError, match="did you mean 'watercolor'"):
        stylize.apply(cv, "watrcolor")
    assert stylize.apply(cv.copy(), "pixel_art", colours=8) is not None        # colours -> colors
    assert stylize.apply(cv.copy(), "watercolour") is not None                  # spelling of the style name
    assert "brush" in stylize.options("oil")


def test_apply_prompt_picks_style_or_does_nothing():
    cv = Canvas(64, 48, "#406080")
    before = cv.buf.copy()
    assert stylize.apply(cv, prompt="a quiet lake at dawn") is None
    assert np.array_equal(cv.buf, before)
    out = stylize.apply(cv, prompt="a linocut of a heron", seed=2)
    assert out is cv and not np.array_equal(cv.buf, before)
    assert "linocut" in cv.styles_applied


# ---------------------------------------------------------------- option guide and guarding

def test_options_give_default_range_and_meaning():
    o = stylize.options("watercolor")
    assert o["deckle"]["default"] == 0.05 and o["deckle"]["range"] == (0, 0.12)
    assert "paper border" in o["deckle"]["means"] and len(o["deckle"]["means"].split()) <= 6
    assert list(o)[0] == "wash" and "brush" in stylize.options("oil")
    assert o["paper"]["range"] is not None                           # watercolour paper is a strength
    assert stylize.options("charcoal")["paper"]["range"] is None     # charcoal paper is a colour
    for name in stylize.names():                                     # every option of every style is described
        for k, v in stylize.options(name).items():
            assert v["means"] != "style option", (name, k)
    assert any(l.startswith("deckle=0.05  (0-0.12)") for l in stylize.option_lines("watercolor"))


def test_out_of_range_options_raise_or_clamp(capsys):
    cv = _small()
    with pytest.raises(ValueError, match=r"deckle=0.4 is outside the useful range 0-0.12"):
        stylize.apply(cv.copy(), "watercolor", deckle=0.4)
    with pytest.raises(ValueError, match="0-0.12"):
        stylize.apply(cv.copy(), "watercolor", deckle=6)
    with pytest.raises(ValueError, match="strength"):
        stylize.apply(cv.copy(), "watercolor", strength=3)
    c2 = cv.copy()
    stylize.apply(c2, "watercolor", deckle=0.15)                     # mild overshoot: clamped with a note
    err = capsys.readouterr().err
    assert "deckle=0.15 is outside 0-0.12" in err and "using 0.12" in err and len(err.strip().splitlines()) == 1
    ref = cv.copy()
    stylize.apply(ref, "watercolor", deckle=0.12)
    assert np.allclose(c2.buf, ref.buf)
    assert c2.style_log[0]["notes"] and c2.style_log[0]["params"] == {"deckle": 0.15}
    assert stylize.apply(cv.copy(), "pixel_art", colors=8) is not None   # in range: untouched


def test_unknown_option_error_lists_options_with_meaning():
    with pytest.raises(TypeError) as e:
        stylize.apply(_small(), "watercolor", deckel=0.1)
    msg = str(e.value)
    assert "did you mean 'deckle'" in msg and "deckle=0.05  (0-0.12)  width of the unpainted paper border" in msg


def test_apply_records_uniformity_before_and_after():
    cv = _small()
    stylize.apply(cv, "watercolor")
    rec = cv.style_log[-1]
    assert rec["style"] == "watercolor" and 0 <= rec["uniform_before"] < 0.6 and "uniform_after" in rec
    assert cv.styles_applied == ["watercolor"]


# ---- round r1: sticker keeps the sheet, storybook, options command, did-you-mean ----
def _sheet():
    cv = Canvas(200, 100, "#d8c8f0")
    X, Y = cv.xy()
    ms = [sdf.fill(sdf.sd_circle(X, Y, cx, 50, 22)) for cx in (35, 100, 165)]
    for m, col in zip(ms, ("#d03020", "#20a040", "#2040d0")):
        cv.paint(col, m)
    return cv, ms


def test_sticker_with_mask_keeps_the_rest_of_the_picture():
    cv, ms = _sheet()
    before = cv.buf.copy()
    stylize.apply(cv, "sticker", mask=ms[0], border=60)
    assert np.allclose(cv.buf[5, 190], before[5, 190], atol=0.02)               # far corner untouched
    assert np.allclose(cv.buf[50, 100], before[50, 100], atol=0.02)             # the second animal survives
    assert cv.buf[50, 35 - 22 - 3].min() > 0.9                                  # white border outside the first
    assert cv.styles_applied == ["sticker"] and cv.style_log[-1]["params"]["mask"] == "masked region"


def test_sticker_sheet_of_three_in_one_call_or_three_calls():
    a, ms = _sheet()
    b, _ = _sheet()
    stylize.apply(a, "sticker", mask=ms, border=60)
    for m in ms:
        stylize.apply(b, "sticker", mask=m, border=60)
    assert np.allclose(a.buf, b.buf, atol=0.02)
    assert b.styles_applied == ["sticker"] * 3
    for cx in (35, 100, 165):
        assert a.buf[50, cx - 22 - 3].min() > 0.9
    c, ms = _sheet()
    stylize.apply(c, "sticker", mask=ms[1], background="#808080")             # explicit background replaces the rest
    assert abs(float(c.buf[5, 190].mean()) - 0.5) < 0.1
    with pytest.raises(ValueError, match="shape"):
        stylize.apply(c, "sticker", mask=np.zeros((5, 5), np.float32))


def test_storybook_is_picked_from_requests_and_warms_the_picture():
    for text in ("Storybook illustration with soft shading", "a children's book picture of a bear",
                 "picture book art", "繪本風格"):
        assert stylize.pick(text) == "storybook", text
    assert stylize.pick("a sticker sheet") == "sticker"
    cv = _scene()
    out = stylize.apply(cv.buf, "storybook")
    assert out.shape == cv.buf.shape and out.min() > 0.02 and out.max() <= 1.0
    assert out[..., 0].mean() >= out[..., 2].mean() - 0.02 or out[..., 2].mean() < cv.buf[..., 2].mean()
    assert abs(float(out.mean()) - float(cv.buf.mean())) < 0.2                  # soft shading, not a recolour
    assert "storybook" in stylize.options("storybook") or "outline" in stylize.options("storybook")


def test_style_choice_names_get_did_you_mean():
    img = np.zeros((40, 40, 3), np.float32)
    with pytest.raises(ValueError, match="did you mean 'pico8'"):
        stylize.apply(img, "pixel_art", palette="pico9")
    with pytest.raises(ValueError, match="did you mean 'silver'"):
        stylize.apply(img, "sepia", tone="silvr")
    with pytest.raises(TypeError, match="did you mean 'brush'"):
        stylize.apply(img, "oil", brsh=3)
    with pytest.raises(ValueError, match="did you mean 'oil'"):
        stylize.apply(img, "oill")


def _run_tool(name, *args, env=None):
    import subprocess
    e = dict(os.environ, PYTHONIOENCODING="cp950", **(env or {}))
    return subprocess.run([sys.executable, os.path.join(ROOT, "tools", name), *args], capture_output=True,
                          env=e, cwd=ROOT, timeout=120)


def test_styles_tool_options_command_and_utf8_output():
    r = _run_tool("styles.py", "options", "ukiyoe")
    out = r.stdout.decode("utf-8")
    assert r.returncode == 0 and "bokashi" in out and "default" in out and "range" in out
    r = _run_tool("styles.py", "options", "ukiyo")
    assert r.returncode != 0 and "did you mean 'ukiyoe'" in r.stderr.decode("utf-8")
    r = _run_tool("styles.py", "pick", "水墨畫 of a mountain")
    assert r.returncode == 0 and r.stdout.decode("utf-8").strip() == "ink_wash"
    r = _run_tool("styles.py", "list")
    assert r.returncode == 0 and "storybook" in r.stdout.decode("utf-8")


def test_look_zoom_file_name_matches_the_command(tmp_path):
    from PIL import Image
    img = tmp_path / "pic.png"
    Image.new("RGB", (64, 36), (90, 120, 200)).save(img)
    r = _run_tool("look.py", "zoom", str(img), "0.3", "0.6", "1.0", "1.0")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "pic_zoom_0.3_0.6_1.0_1.0.jpg").exists()
    r = _run_tool("look.py", "zoom", str(img), "0", "0", ".5", "1")
    assert (tmp_path / "pic_zoom_0_0_.5_1.jpg").exists()


@pytest.mark.parametrize("given, want", [("ukiyo-e", "ukiyoe"), ("Ukiyo E", "ukiyoe"), ("pixel-art", "pixel_art"),
                                         ("Pixel Art", "pixel_art"), ("water colour", "watercolor"),
                                         ("Ink-Wash", "ink_wash"), ("OIL", "oil"), ("low poly", "low_poly")])
def test_style_names_match_ignoring_case_hyphens_spaces(given, want):
    assert stylize.resolve(given) == want
    assert list(stylize.options(given)) == list(stylize.options(want))
    cv = Canvas(40, 30, "#808080")
    stylize.apply(cv, given)
    assert cv.styles_applied == [want]


def test_style_name_miss_has_did_you_mean():
    with pytest.raises(ValueError, match="did you mean 'ukiyoe'"):
        stylize.resolve("ukiyoo")
    with pytest.raises(ValueError, match="did you mean 'oil'"):
        stylize.options("oill")
    assert stylize.pick("a pixel-art sprite") == "pixel_art"
    assert stylize.pick("water colour fox") == "watercolor"


def test_apply_on_array_reaches_the_main_canvas_styles():
    main = Canvas(80, 60, "#ffffff")
    sticker = Canvas(40, 30, "#336699")
    out = stylize.apply(sticker.buf, "sticker", cv=main)
    assert out.shape == sticker.buf.shape and main.styles_applied == ["sticker"]
    sub = Canvas(40, 30, "#336699")
    stylize.apply(sub, "risograph")
    main.paste(sub, 10, 10)                                # a styled scratch canvas carries its styles when pasted
    assert "risograph" in main.styles_applied
