# -*- coding: utf-8 -*-
"""Typography engine: fonts and fallback, kerning, layout options, effects, arcs, vertical text, audit, layout."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import layout  # noqa: E402
from pixelcanvas import text as T  # noqa: E402


def _cv(w=480, h=270, c="#1a1c24"):
    return Canvas(w, h, c)


def test_find_font_style_words_always_return_something():
    for words in ["sans", "serif", "slab", "mono", "display", "condensed", "rounded", "handwriting", "script",
                  "serif italic", "condensed bold", "sans black", "light", "heavy"]:
        assert T.find_font(words) is not None, words
    assert T.find_font("definitely-not-a-font-name") is None
    p = T.find_font(None, text="你好")
    assert p is None or T.missing_glyphs("你好", p, fallback=False) == []


@pytest.fixture
def mac_like_cjk(monkeypatch):
    """The macOS situation on any machine: the first CJK candidate of every language lacks Han glyphs (as Apple SD
    Gothic Neo lacks 你), the second covers them. Yields (lacking, covering) font specs."""
    covering = next((p for p in [T._cjk_font(l_) for l_ in ("tc", "sc", "ja", "ko")]
                     if p and T.missing_glyphs("你好漢字", p, fallback=False) == []), None)
    lacking = T.find_font("sans")
    if covering is None or lacking is None or T._covers(lacking, ord("你")):
        pytest.skip("needs one Latin-only and one Chinese font installed")
    real = T._stem_path
    fake = {"zzlacking": lacking, "zzcovering": covering}
    monkeypatch.setattr(T, "_stem_path", lambda stem: fake.get(stem) or real(stem))
    monkeypatch.setattr(T, "_CJK", {l_: (["zzlacking", "zzcovering"],) * 3 for l_ in ("tc", "sc", "ja", "ko")})
    monkeypatch.delenv("PIXELCANVAS_CJK_FONT", raising=False)
    monkeypatch.delenv("PIXELCANVAS_FONT", raising=False)
    T._chain.cache_clear()
    yield lacking, covering
    T._chain.cache_clear()
    T._COVER_SCAN.clear()


def test_cjk_font_covers_the_text_when_the_first_candidate_lacks_glyphs(mac_like_cjk):
    lacking, covering = mac_like_cjk
    for words in (None, "cjk", "serif", "sans"):
        p = T.find_font(words, text="你好")
        assert p == covering, (words, p)
        assert T.missing_glyphs("你好", p, fallback=False) == []
    assert T.find_font(None, text="你好", bold=True) == covering
    assert T.find_font(None, text="こんにちは世界") == covering
    assert T.missing_glyphs("你好") == [] and T.missing_glyphs("Hello 你好", lacking) == []


def test_cjk_text_is_drawn_with_the_covering_font(mac_like_cjk):
    from pixelcanvas.text import _make_block
    lacking, covering = mac_like_cjk
    b = _make_block("你好世界", (0, 0), 30, None, False, "la", "left", None, 1.2, 0.0)
    assert b.missing == [] and b.fonts == {covering}             # one face draws every character
    b = _make_block("Hello 你好", (0, 0), 30, lacking, False, "la", "left", None, 1.2, 0.0)
    assert b.missing == [] and b.fonts == {lacking, covering}    # Latin in the chosen font, Han in the covering one
    cv = _cv()
    cv.text("你好", (10, 40), 30)
    assert cv.text_log[-1]["missing"] == []


def test_cjk_font_scans_installed_fonts_when_no_listed_one_covers(mac_like_cjk, monkeypatch):
    lacking, _ = mac_like_cjk
    monkeypatch.setattr(T, "_CJK", {l_: (["zzlacking"],) * 3 for l_ in ("tc", "sc", "ja", "ko")})
    monkeypatch.setattr(T, "_COVER_SCAN", {})
    scans = []
    real_faces = T._scan_faces
    monkeypatch.setattr(T, "_scan_faces", lambda kind: scans.append(kind) or real_faces(kind))
    p = T.find_font(None, text="你好")
    assert p != lacking and T.missing_glyphs("你好", p, fallback=False) == []
    assert scans == ["sans"] and T._COVER_SCAN[(("han",), "sans")] == p    # scanned once, cached per (scripts, kind)
    assert T.find_font(None, text="好你") == p and scans == ["sans"]        # no second scan
    b = T._make_block("你好", (0, 0), 30, None, False, "la", "left", None, 1.2, 0.0)
    assert b.missing == [] and b.fonts == {p}


def test_fallback_covers_mixed_scripts():
    s = "Café → ∑ ★ 中文 かな 한국어 ±½"
    assert T.missing_glyphs(s) == []
    cv = _cv()
    cv.text(s, (10, 40), 24)
    e = cv.text_log[-1]
    assert e["missing"] == [] and len(e["fonts"]) >= 1


def test_missing_glyph_is_reported():
    cv = _cv()
    cp = "\U000F0001"                                           # private-use plane: nothing covers it
    assert T.missing_glyphs("a" + cp) == [cp]
    cv.text("x" + cp, (10, 40), 24)
    kinds = [i["kind"] for i in T.audit_text(cv)]
    assert "missing_glyphs" in kinds


def test_kerning_tightens_av():
    spec = T.find_font("sans")
    a = T._shape("AV", spec, 100.0, "regular", 0.0)[1]
    b = T._shape("A", spec, 100.0, "regular", 0.0)[1] + T._shape("V", spec, 100.0, "regular", 0.0)[1]
    assert a <= b


def test_backwards_compatible_calls():
    m = T.text_mask(30, 80, "Hi", (2, 2), 20)
    assert m.shape == (30, 80) and m.max() > 0.9
    bb = T.text_bbox("Measure first", (0, 0), 40)
    assert bb[2] > bb[0] and bb[3] > bb[1]
    s = T.fit_text_size("ONE LINE", 200, 60)
    assert 10 < s <= 60
    f = T.load_font(None, 30)
    assert T.wrap_text("one two three four five six", f, 120)
    cv = _cv()
    cv.text("box", box=(10, 10, 200, 80), color="#fff", bold=True, outline=2, shadow=True, glow=True)
    assert cv.last_bbox[2] <= 200 + 1


def test_rich_markup_and_spans():
    spans = T.parse_markup("**B** *i* {#f00 size=1.5|red} ==m== ~~s~~ __u__")
    styles = [s for _, s in spans]
    assert any(s.get("bold") for s in styles) and any(s.get("italic") for s in styles)
    assert any(s.get("color") == "#f00" and s.get("size") == "1.5" for s in styles)
    cv = _cv()
    cv.text("**Bold** and {#ff0|yellow}", (10, 40), 28)
    cv.text([("Red ", "#f33"), ("big", {"size": 1.6, "bold": True})], (10, 120), 28)
    assert np.isfinite(cv.buf).all() and cv.buf.std() > 0.02


@pytest.mark.parametrize("opts", [
    dict(rotate=30, anchor="mm"), dict(vertical=True, max_height=200), dict(align="justify", max_width=260),
    dict(case="smallcaps"), dict(underline=True, strike=True, highlight="#ff0"), dict(drop_cap=3, max_width=300),
    dict(tabs=[(150, "r"), (250, ".")], tabular=True), dict(list_style="number", max_width=300),
    dict(italic=True), dict(bold="black"), dict(balance=True, box=(10, 10, 300, 200)),
])
def test_layout_options_draw(opts):
    cv = _cv()
    txt = "縦書きのテスト、「括弧」。" if opts.get("vertical") else (
        "Item\tQty\tPrice\nTea\t2\t3.50" if "tabs" in opts else "The quick brown fox jumps over the lazy dog again and again")
    xy = None if "box" in opts else (240 if opts.get("anchor") == "mm" else 10, 135 if opts.get("anchor") == "mm" else 20)
    cv.text(txt, xy, 22, **opts)
    assert np.isfinite(cv.buf).all() and cv.buf.std() > 0.01


@pytest.mark.parametrize("fx", T.EFFECTS)
def test_every_effect_runs(fx):
    cv = _cv(c="#2a2a33")
    cv.text("Ab", (20, 20), 90, color="#ff5a7a", bold=True, effect=fx)
    assert np.isfinite(cv.buf).all() and cv.buf.std() > 0.02


def test_effects_compose_and_neon_flicker():
    cv = _cv()
    cv.text("NEON", (20, 30), 80, color="#ff3cac", effect=dict(kind="neon", off=[1]), outline=2)
    cv.text("3D", (300, 30), 80, effect=["extrude", "gold"], shadow=True)
    assert np.isfinite(cv.buf).all()


def test_arc_and_path_text():
    cv = _cv(400, 400)
    bb = T.text_on_arc(cv, "TOP OF THE SEAL", (200, 200), 140, 26, angle=-90)
    assert bb[1] < 100 and bb[3] < 200
    bb2 = T.text_on_arc(cv, "BOTTOM", (200, 200), 140, 26, angle=90)
    assert bb2[1] > 200
    pts = [(20, 350), (200, 300), (380, 350)]
    T.text_on_path(cv, "on a path", pts, 24, align="center")
    cv.text("arc via cv.text", arc=(200, 200, 90), size=18)
    assert len(cv.text_log) == 4


def test_auto_colour_and_plate():
    light = _cv(c="#f0f0f0")
    light.text("auto", (20, 20), 60, color="auto")
    e = light.text_log[-1]
    assert e["contrast"] > 4.5
    dark = _cv(c="#101010")
    dark.text("auto", (20, 20), 60, color="auto", plate="pill")
    assert dark.text_log[-1]["contrast"] > 3


def test_text_stays_inside_canvas():
    cv = _cv()
    bb = T.draw_text(cv, "Off the edge", (400, 250), 60)
    assert bb[2] <= cv.w + 0.5 and bb[3] <= cv.h + 0.5
    bb2 = T.draw_text(cv, "allowed", (440, 250), 60, clip_ok=True)
    assert bb2[2] > cv.w


def test_audit_finds_problems():
    cv = _cv(c="#808080")
    cv.text("low contrast", (20, 20), 30, color="#8a8a8a")
    cv.text("tiny", (20, 100), 3, color="#fff")
    cv.text("over", (20, 20), 30, color="#fff")
    cv.text("edge", (450, 250), 30, color="#fff", clip_ok=True)
    kinds = {i["kind"] for i in T.audit_text(cv)}
    assert {"contrast", "small", "overlap", "clipped"} <= kinds
    for i in T.audit_text(cv):
        assert i["fix"] and i["bbox"]


def test_layout_boxes():
    cv = _cv(1000, 600)
    L = layout.Layout(cv, margin=0.05)
    assert L.m == pytest.approx(30)
    r = L.rows([2, 1], gap=10)
    assert r[0].h == pytest.approx(2 * r[1].h, rel=1e-3)
    g = L.grid(2, 3, gap=0.02)
    assert len(g) == 2 and len(g[0]) == 3
    a, b = L.golden("x")
    assert a.w / b.w == pytest.approx(layout.PHI, rel=1e-3)
    assert len(L.thirds()["points"]) == 4
    strip, rest = L.take("top", 100)
    assert strip.h == 100 and rest[1] == strip[3]
    al = layout.align_boxes([(0, 0, 10, 10), (5, 20, 30, 30)], "left")
    assert al[0][0] == al[1][0]
    d = layout.distribute([(0, 0, 10, 10)] * 3, 0, 100)
    assert d[-1][2] == pytest.approx(100)


def test_stack_fits_box():
    cv = _cv(1080, 1350)
    box = layout.Box(80, 80, 1000, 700)
    bbs = layout.stack(cv, box, [("Kicker", "kicker"), ("A Very Long Title For The Stack", "title"),
                                 ("subtitle line", "subtitle"), "Body text " * 20], theme="midnight")
    assert len(bbs) == 4
    assert bbs[-1][3] <= box[3] + 2 and all(b[1] >= box[1] - 2 for b in bbs)
    assert all(bbs[i][3] <= bbs[i + 1][1] + 1 for i in range(3))


@pytest.mark.parametrize("name", ["poster", "flyer", "book_cover", "magazine", "business_card", "quote_card",
                                  "thumbnail", "slide", "certificate", "menu", "badge", "label"])
def test_templates(name):
    cv = Canvas(320, 400)
    content = dict(title="Night Market", subtitle="Street food and music", details=["Fri 8 pm", "Pier 4"])
    if name == "business_card":
        content = dict(name="Lena Ortiz", role="Designer", details=["lena@example.com"])
    elif name == "quote_card":
        content = dict(quote="Less, but better.", author="Dieter Rams")
    elif name == "certificate":
        content = dict(name="Alex Morgan", date="2026")
    elif name == "menu":
        content = dict(title="Menu", sections=[("Coffee", [("Espresso", "3.00")])])
    elif name == "magazine":
        content = dict(title="FIELD", headline="Slow travel", cover_lines=["Ten chairs"])
    elif name == "badge":
        content = dict(title="Type foundry", subtitle="since 2026", center="EST.")
    elif name == "book_cover":
        content = dict(title="The Orchard", author="M. Ellison")
    elif name == "thumbnail":
        content = dict(title="Night Market", subtitle="Street food and music", badge="NEW")   # no ``details`` here
    boxes = layout.template(name, cv, **content)
    assert "image" in boxes and np.isfinite(cv.buf).all() and cv.buf.std() > 0.02


# ---- regressions found in the fresh-eyes review

def test_contrast_catches_text_partly_over_a_dark_shape():
    from pixelcanvas import sdf
    cv = Canvas(1000, 800, "#ece4d4")
    X, Y = cv.xy()
    cv.paint("#b4472f", sdf.fill(sdf.sd_box(X, Y, 700, 400, 400, 400)))      # covers the right part of the line
    cv.text("Why every journey starts with a walk", (40, 400), 24, "#333333", anchor="lm")
    e = cv.text_log[-1]
    assert e["contrast"] > 4.5                       # the median says fine ...
    assert e["min_contrast"] < 3.0 and e["worst_box"][0] > 300   # ... the local check finds the red part
    kinds = [i["kind"] for i in T.audit_text(cv)]
    assert "contrast" in kinds


def test_min_contrast_is_the_worst_case_not_the_best():
    cv = Canvas(600, 200, "#ffffff")
    cv.text("dark on white", (20, 100), 40, "#000000", anchor="lm")
    e = cv.text_log[-1]
    assert e["min_contrast"] <= e["contrast"] + 1e-6
    assert T.audit_text(cv) == []


def test_ink_box_has_real_side_bearings():
    sp = T.find_font("serif")
    l, t, r, b = T._ink(sp, 100, "regular", ".")
    adv = T._shape(".", sp, 100.0, "regular", 0.0)[1]
    assert l > 0 and r < adv and b <= 2                  # a full stop is narrow ink inside its advance


def test_arc_repeat_fills_the_ring():
    cv = _cv(400, 400)
    bb = T.text_on_arc(cv, "LIVE", (200, 200), 120, 24, color="#fff", bold=True, repeat=True)
    e = cv.text_log[-1]
    assert e["text"].count("LIVE") >= 4
    assert bb[2] - bb[0] > 230 and bb[3] - bb[1] > 230     # all the way round
    T.text_on_arc(cv, "A VERY LONG TEXT THAT CANNOT FIT ROUND A SMALL CIRCLE", (200, 200), 40, 20, align="ring")
    assert cv.text_log[-1]["size"] < 20                  # shrunk to fit one turn


def test_vertical_punctuation_by_language():
    def ink_x(text, lang=None):
        cv = Canvas(200, 400, "#000000")
        cv.text(text, (150, 10), 60, "#ffffff", anchor="ra", vertical=True, lang=lang)
        blk = cv.text_log[-1]["layout_bbox"]
        cx = (blk[0] + blk[2]) / 2
        a = cv.buf[70:130, :, 0] > 0.5                 # second cell: the punctuation mark
        xs = np.nonzero(a.any(0))[0]
        return (xs.min() + xs.max()) / 2 - cx
    assert ink_x("語、語", "ja") > 8                     # Japanese: top-right of the cell
    assert abs(ink_x("中，中", "tc")) < 8                # Traditional Chinese: centred


def test_justified_lines_do_not_open_rivers():
    cv = _cv(900, 500)
    para = ("Fog rolled in before dawn and stayed until noon. The ridge disappeared, then the pines, then the lake "
            "itself, and the whole valley went quiet under a white sky.")
    cv.text(para, (20, 20), 26, "#ffffff", max_width=420, align="justify")
    e = cv.text_log[-1]
    assert e["layout_bbox"][2] - e["layout_bbox"][0] <= 421


# ---- strict options, enumerated values, registry names (text / layout)

def test_unknown_text_option_names_the_closest():
    cv = Canvas(200, 100)
    with pytest.raises(TypeError, match=r"unknown option 'colr'.*did you mean 'color'"):
        cv.text("Hi", (10, 10), 20, colr="#fff")
    with pytest.raises(TypeError, match=r"unknown option 'itallic'.*did you mean 'italic'"):
        T.text_mask(40, 80, "Hi", (2, 2), 20, itallic=True)
    with pytest.raises(TypeError, match=r"draw_text shadow: unknown option 'dxx'.*dx"):
        cv.text("Hi", (10, 10), 20, shadow=dict(dxx=2))


def test_text_aliases_are_accepted():
    cv = Canvas(200, 100)
    T.draw_text(cv, "Hi", (10, 10), 20, "#ffffff", outline_colour="#ff0000", outline=1)
    assert cv.text_log
    assert T.text_mask(40, 80, "Hi", (2, 2), 20, font_path=None).max() > 0.5
    assert T.text_bbox("Hi", (0, 0), 20, italics=True)[2] > 0           # plural of ``italic``


def test_text_enumerations_and_registry_names():
    cv = Canvas(200, 100)
    with pytest.raises(ValueError, match=r"invalid anchor 'cc'"):
        T.text_bbox("Hi", (0, 0), 20, anchor="cc")
    with pytest.raises(ValueError, match=r"invalid case 'uper'.*did you mean 'upper'"):
        cv.text("Hi", (10, 10), 20, case="uper")
    with pytest.raises(ValueError, match=r"unknown text effect 'neonn'.*did you mean 'neon'"):
        cv.text("Hi", (10, 10), 20, effect="neonn")
    with pytest.raises(ValueError, match=r"unknown template 'postr'.*did you mean 'poster'"):
        layout.template("postr", cv, title="x")
    with pytest.raises(TypeError, match=r"template 'poster': unknown option 'titel'.*did you mean 'title'"):
        layout.template("poster", cv, titel="x")
    with pytest.raises(ValueError, match=r"unknown theme 'midnite'.*did you mean 'midnight'"):
        layout.theme("midnite")
    with pytest.raises(ValueError, match=r"unknown role 'titel'.*did you mean 'title'"):
        layout.stack(cv, (0, 0, 200, 100), [("x", "titel")])
    with pytest.raises(TypeError, match=r"unknown option 'colr'"):
        layout.stack(cv, (0, 0, 200, 100), ["x"], colr="#fff")


def test_first_glyph_is_not_cropped_by_a_tight_mask():
    """Text that starts with a bracket or quotation mark hangs left of its origin; the measured box includes it,
    so a mask cut to the box (as vector.Drawing.text does) keeps every glyph."""
    for s in ("(a)", '"q"', "jf", "“q”"):
        for size in (40, 120):
            x0, y0, x1, y1 = T.text_bbox(s, (30, 10), size)
            pad = 2
            ox, oy = int(np.floor(x0)) - pad, int(np.floor(y0)) - pad
            w, h = int(np.ceil(x1)) + pad - ox, int(np.ceil(y1)) + pad - oy
            m = T.text_mask(h, w, s, (30 - ox, 10 - oy), size)
            assert m.max() > 0.5
            assert m[:, 0].max() < 0.02, (s, size)                    # nothing touches the first column


def test_vector_text_keeps_a_leading_bracket():
    from pixelcanvas import vector
    d = vector.Drawing(240, 120)
    d.text(40, 80, "(a)", size=60, fill="#000000", anchor="ls")
    cv = Canvas(240, 120, "#ffffff")
    d.draw(cv)
    ref = Canvas(240, 120, "#ffffff")
    ref.text("(a)", (40, 80), 60, "#000000", anchor="ls")
    ink, full = float((1 - cv.buf[..., 0]).sum()), float((1 - ref.buf[..., 0]).sum())
    assert ink > 0.95 * full                       # the hung "(" is not cut off by the text tile


# ---- separator lists, theme keys, template options and layouts ----------------------------------------------------

LINEUP = "Ana Reyes Trio · Kofi & The Lanterns · DJ Marlowe"


def _block_lines(text, width, size=30):
    """Text of each laid-out line (spaces removed), from the glyph placements grouped by baseline."""
    from pixelcanvas.text import _make_block
    b = _make_block(text, (0, 0), size, None, False, "la", "left", width, 1.2, 0.0)
    rows = {}
    for t, *_rest in b.pl:
        tx, ty = _rest[-3], _rest[-2]
        rows.setdefault(round(ty), []).append((tx, t))
    return ["".join(t for _, t in sorted(v)) for _, v in sorted(rows.items())], b


@pytest.mark.parametrize("width", [300, 420, 520, 640, 900])
def test_separator_lists_break_between_items(width):
    lines, b = _block_lines(LINEUP, width)
    items = ["AnaReyesTrio", "Kofi&TheLanterns", "DJMarlowe"]
    for ln in lines:
        parts = [p for p in ln.replace(" ", "").split("·") if p]
        assert all(p in items for p in parts), lines            # never "Kofi" / "& The Lanterns"
        assert not ln.endswith("·")                             # the separator at a line end is dropped
    assert "".join(lines).replace("·", "") == "".join(items)
    assert b.bbox[2] - b.bbox[0] <= width + 1


@pytest.mark.parametrize("font,size,step", [(None, 30, 1), ("serif", 26, 3), ("mono", 30, 3), ("condensed", 34, 3)])
def test_separator_never_ends_or_starts_a_line_at_any_width(font, size, step):
    """Every width from 120 to 1000 px, several fonts: line-break positions depend on the metrics, the rule does not."""
    spec = T.find_font(font)
    for width in range(120, 1001, step):
        lines, _ = _lines_with(spec, width, size)
        assert lines and not any(ln.endswith("·") or ln.startswith("·") for ln in lines), (font, size, width, lines)
        assert "".join(lines).replace("·", "") == "AnaReyesTrioKofi&TheLanternsDJMarlowe", (width, lines)


def _lines_with(spec, width, size):
    from pixelcanvas.text import _make_block
    b = _make_block(LINEUP, (0, 0), size, spec, False, "la", "left", width, 1.2, 0.0)
    rows = {}
    for t, *_rest in b.pl:
        rows.setdefault(round(_rest[-2]), []).append((_rest[-3], t))
    return ["".join(t for _, t in sorted(v)) for _, v in sorted(rows.items())], b


def test_prose_still_wraps_normally():
    lines, _ = _block_lines("The quick brown fox jumps over the lazy dog and keeps running", 260)
    assert len(lines) >= 3 and all(lines)


def test_theme_keys_and_role_colors():
    th = layout.theme("midnight", title_color="#f0d078", band="#123456")
    assert th["title"] == "#f0d078" and layout.role_color(th, "title") == "#f0d078"
    assert layout.role_color(th, "kicker") == th["accent"] and layout.role_color(th, "band") == "#123456"
    assert {"ink", "soft", "accent", "panel", "title_font", "body_font", "title", "band"} <= set(layout.THEME_KEYS)
    with pytest.raises(TypeError, match="did you mean 'title_font'"):
        layout.theme("midnight", titel_font="serif")


def test_template_colour_options():
    cv = Canvas(540, 675)
    b = layout.template("poster", cv, title="Blue Hour", title_color="#f0d078", accent="#d4a63a")
    tb = b["title"]
    reg = cv.buf[int(tb[1]):int(tb[3]), int(tb[0]):int(tb[2])].reshape(-1, 3)
    gold = ((reg[:, 0] > 0.75) & (reg[:, 1] > 0.6) & (reg[:, 2] < 0.6)).mean()
    assert gold > 0.05                                         # the title is gold, not the theme ink
    with pytest.raises(TypeError, match="did you mean 'title_color'"):
        layout.template("poster", Canvas(100, 100), title="x", titel_color="#fff")


def test_menu_spreads_over_the_sheet():
    cv = Canvas(540, 750)
    secs = [(s, [("Item %d" % i, "$%d.00" % i) for i in range(5)]) for s in ("Coffee", "Tea", "Pastries")]
    b = layout.template("menu", cv, title="Fika & Co.", sections=secs)
    ys = [r["bbox"][3] for r in cv.text_log if r.get("bbox")]
    L = b["layout"]
    assert max(ys) > L[3] - 0.08 * cv.h                        # no empty band at the bottom


def test_thumbnail_title_on_top_with_two_sides():
    cv = Canvas(640, 360)
    b = layout.template("thumbnail", cv, title="I built a robot in 24 hours", layout="top")
    assert b["title"][3] < b["left"][1] + 1 and b["left"][2] < b["right"][0]
    with pytest.raises(ValueError, match="'top'"):
        layout.template("thumbnail", Canvas(64, 36), title="x", layout="tpo")


def test_poster_and_flyer_picture_areas():
    cv = Canvas(540, 675)
    b = layout.template("poster", cv, title="Blue Hour", details=[("Line-up", LINEUP)])
    assert b["image"].h > 0.3 * cv.h
    cv = Canvas(540, 675)
    f = layout.template("flyer", cv, title="週末小農市集", subtitle="Weekend Farmers Market", place="華山文創園區",
                        theme="paper")
    assert f["image"].w > 0.8 * (cv.w - 2 * f["layout"].m)    # the picture area spans the sheet, not half of it


def test_audit_contrast_that_rounds_to_the_requirement_is_not_flagged():
    cv = _cv(480, 270, "#777777")
    cv.text("Hello world", (240, 135), 9, "#ffffff", anchor="mm")
    e = cv.text_log[0]
    for c, flagged in ((4.46, False), (4.5, False), (4.4, True), (2.9, True)):
        e["contrast"], e["min_contrast"] = c, None
        kinds = [i["kind"] for i in T.audit_text(cv)]
        assert ("contrast" in kinds) == flagged, (c, kinds)
