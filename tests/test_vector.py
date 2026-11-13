# -*- coding: utf-8 -*-
"""pixelcanvas.vector: raster accuracy, SVG validity, layout helpers, layers, icons, doc snippets."""
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, layers  # noqa: E402
from pixelcanvas import vector as V  # noqa: E402
from pixelcanvas.vector import Drawing, Path, linear_gradient, radial_gradient, hatch  # noqa: E402

SVG = "{http://www.w3.org/2000/svg}"


def _ink(cv):
    """Darkness map of a white-background render."""
    return 1.0 - cv.buf.mean(-1)


def test_coverage_is_area_exact():
    sq = np.array([[[10.25, 10.5], [30.75, 10.5], [30.75, 26.0], [10.25, 26.0]]])
    x0, y0, cov = V._raster([([sq], False)], (0, 0, 64, 64))[0]
    assert abs(cov.sum() - 20.5 * 15.5) < 0.5
    t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    c = np.stack([32 + 20 * np.cos(t), 32 + 20 * np.sin(t)], -1)[None]
    cov = V._raster([([c], False)], (0, 0, 64, 64))[0][2]
    assert abs(cov.sum() - np.pi * 400) < 2


def test_fill_rules_and_union():
    outer = np.array([[[0, 0], [40, 0], [40, 40], [0, 40]]], float)
    inner = np.array([[[10, 10], [30, 10], [30, 30], [10, 30]]], float)
    eo = V._raster([([outer, inner], True)], (0, 0, 40, 40))[0][2]
    nz = V._raster([([outer, inner], False)], (0, 0, 40, 40))[0][2]
    assert eo[20, 20] == 0 and eo[5, 5] == 1 and nz[20, 20] == 1


def test_one_px_lines_are_crisp():
    d = Drawing(60, 40, background="#ffffff")
    d.line(5, 10, 55, 10, stroke="#000", stroke_width=1)          # integer y → snapped onto one pixel row
    d.line(20, 15, 20, 35, stroke="#000", stroke_width=1)
    d.rect(30, 20, 20, 10, fill=None, stroke="#000", stroke_width=1)
    ink = _ink(d.render())
    assert ink[10, 10:50].min() > 0.99 and ink[9, 10:50].max() < 0.01 and ink[11, 10:50].max() < 0.01
    col = ink[20:30, 18:23].mean(0)
    assert col.max() > 0.99 and np.sort(col)[-2] < 0.01
    assert ink[20, 35] > 0.99 and ink[25, 35] < 0.01                # rect outline row, empty inside


def test_antialiased_edges_and_arrow_markers():
    d = Drawing(120, 60, background="#ffffff")
    d.circle(30, 30, 20, fill="#000")
    d.line(60, 50, 110, 10, stroke="#000", stroke_width=2, arrow="end")
    ink = _ink(d.render())
    edge = ink[30, 8:13]
    assert ((edge > 0.02) & (edge < 0.98)).any()                    # soft edge, not a hard threshold
    assert ink[12:20, 100:110].max() > 0.9                          # arrow head drawn near the tip


def test_primitives_and_styles_render():
    d = Drawing(400, 300, theme="light")
    t = d.theme
    d.polyline([(10, 10), (60, 40), (110, 10)], stroke=t.ink, stroke_width=4, join="round", cap="round")
    d.polygon([(130, 10), (180, 40), (150, 60)], fill=t.accent(1))
    d.rect(200, 10, 80, 50, r=10, fill=linear_gradient(["#ff0000", "#0000ff"]))
    d.ellipse(330, 35, 50, 25, fill=radial_gradient([(0, "#ffffff"), (1, "#000000")]))
    d.arc(60, 150, 40, 180, 360, stroke=t.ink, stroke_width=3, marker_end="open")
    d.pie(170, 150, 40, -90, 45, fill=t.accent(2))
    d.pie(260, 150, 40, 0, 200, inner=20, fill=t.accent(3))
    d.path("M300 120 q40 -30 80 0 t80 0", stroke=t.ink, dash="dash", marker_end="diamond")
    d.path().move_to(10, 220).cubic_to(60, 180, 100, 260, 150, 220).line_to(150, 280).close()
    d.bezier([(170, 280), (200, 200), (260, 300), (290, 220)], stroke=t.ink, marker_start="bar", marker_end="dot")
    d.spline([(300, 280), (330, 230), (360, 270), (390, 220)], stroke=t.accent(0), stroke_width=2)
    d.rect(300, 200, 90, 30, fill="#ffffff", hatch=hatch(30, spacing=5, cross=True), stroke=t.ink)
    cv = d.render()
    b = cv.buf
    assert b[30, 205, 0] > b[30, 205, 2] and b[55, 205, 2] > b[55, 205, 0]   # vertical gradient red → blue
    assert b[35, 330].mean() > b[35, 375].mean()                              # radial: light centre
    hat = _ink(cv)[205:225, 305:385]
    assert 0.1 < (hat > 0.5).mean() < 0.7                                     # hatch lines, not solid


def test_groups_transforms_clip_opacity_shadow():
    d = Drawing(200, 100, background="#ffffff")
    with d.group("a", translate=(50, 50), rotate=45, scale=2):
        d.rect(-10, -10, 20, 20, fill="#000")
    with d.group("b", clip=(120, 20, 160, 60), opacity=0.5):
        d.circle(140, 40, 30, fill="#000")
    with d.group("c", shadow=dict(dx=0, dy=6, blur=3, color="#000", opacity=0.5)):
        d.rect(170, 20, 20, 20, fill="#ffffff")
    ink = _ink(d.render())
    assert ink[50, 50] > 0.99 and ink[50, 74] > 0.9 and ink[25, 25] < 0.01     # rotated, scaled square
    assert abs(ink[40, 140] - 0.5) < 0.02 and ink[40, 115] < 0.01             # clipped, half opacity
    assert ink[44, 180] > 0.05                                                # shadow under the white box
    bb = d.find("a").bbox()
    assert abs(bb[0] - (50 - 20 * 2 ** 0.5)) < 0.5 and abs(bb[3] - (50 + 20 * 2 ** 0.5)) < 0.5


def test_text_measure_bbox_and_place():
    d = Drawing(400, 200)
    w, h = d.measure_text("Hello world", size=20)
    w2, _ = V.measure_text("Hello world", size=40)
    assert 60 < w < 160 and 15 < h < 40 and abs(w2 / w - 2) < 0.15
    _, h3 = d.measure_text("a\nb\nc", size=20)
    assert h3 > 2.5 * h * 0.8
    t = d.text(200, 100, "Hello world", size=20, anchor="mm")
    x0, y0, x1, y1 = t.bbox()
    assert abs((x0 + x1) / 2 - 200) < 1 and abs((y0 + y1) / 2 - 100) < 1 and abs((x1 - x0) - w) < 1
    box = d.rect(10, 10, 50, 30)
    lab = d.text(0, 0, "label", size=12).place(box, "right", gap=8)
    lb = lab.bbox()
    assert abs(lb[0] - 68) < 0.5 and abs((lb[1] + lb[3]) / 2 - 25) < 0.5
    r = d.text(100, 150, "rotated", size=16, rotate=90)
    rb = r.bbox()
    assert rb[3] - rb[1] > rb[2] - rb[0]


def test_text_raster_matches_bbox_and_cjk():
    d = Drawing(300, 120, background="#ffffff")
    t = d.text(150, 60, "Vector 中文", size=36, anchor="mm", fill="#000", halo="#ff0000")
    ink = _ink(d.render())
    ys, xs = np.nonzero(ink > 0.5)
    x0, y0, x1, y1 = t.bbox()
    assert xs.min() >= x0 - 6 and xs.max() <= x1 + 6 and ys.min() >= y0 - 6 and ys.max() <= y1 + 6
    assert (xs > 150).sum() > 50                                     # CJK glyphs present on the right


def test_svg_is_valid_and_matches_geometry():
    d = Drawing(300, 200, theme="light")
    with d.group("nodes", translate=(10, 5), opacity=0.8, shadow=True, clip=(0, 0, 200, 150)):
        d.rect(20, 20, 100, 60, r=8, fill=linear_gradient(["#ffffff", "#dddddd"]), stroke="#333",
               dash="dash", name="box")
        d.circle(160, 50, 20, fill="#fff", hatch=hatch(45))
        d.text(70, 50, "A & <B>", size=14, anchor="mm")
    d.line(10, 180, 290, 180, arrow="both", stroke_width=2)
    d.path("M20 120 C60 90 100 150 140 120", stroke="#e00", marker_end="open")
    s = d.to_svg()
    root = ET.fromstring(s.split("?>", 1)[1])
    assert root.tag == SVG + "svg" and root.get("viewBox") == "0 0 300 200"
    g = root.find(f".//{SVG}g[@id='nodes']")
    assert g is not None and "translate(10 5)" in g.get("transform") and g.get("opacity") == "0.8"
    assert g.get("clip-path") and root.find(f".//{SVG}clipPath") is not None
    assert root.find(f".//{SVG}filter/{SVG}feGaussianBlur") is not None
    assert root.find(f".//{SVG}linearGradient") is not None and root.find(f".//{SVG}pattern") is not None
    rect = root.find(f".//{SVG}rect[@x='20']")
    assert rect.get("width") == "100" and rect.get("rx") == "8" and rect.get("stroke-dasharray")
    txt = root.find(f".//{SVG}text[@text-anchor='middle']")
    assert txt is not None and "".join(txt.itertext()) == "A & <B>" and "sans-serif" in txt.get("font-family")
    assert len(root.findall(f".//{SVG}path")) >= 4                       # strokes + marker geometry


def test_svg_mm_units_and_save(tmp_path):
    d = Drawing(210, 148, units="mm", px_per_unit=4)
    d.rect(10, 10, 50, 30, fill=None, stroke="#000", stroke_width=0.35)
    p = d.save_svg(str(tmp_path / "a.svg"))
    root = ET.parse(p).getroot()
    assert root.get("width") == "210mm" and root.get("height") == "148mm"
    cv = d.render()
    assert (cv.w, cv.h) == (840, 592)


def test_path_parser_and_arcs():
    p = Path("M10 10 h20 v10 H10 z m40 0 a10 10 0 1 0 20 0 a10 10 0 1 0 -20 0 Z")
    assert len(p.subs) == 2 and all(s[2] for s in p.subs)
    x0, y0, x1, y1 = p.bbox()
    assert abs(x1 - 70) < 0.05 and abs(y1 - 20) < 0.05 and abs(y0 - 0) < 0.05
    q = Path("M0 0 Q10 10 20 0 T40 0 S60 10 70 0")
    assert q.bbox()[2] == 70
    assert Path("M0 0a5 5 0 105 5").subs[0][1]                      # compact arc flags


def test_layers_one_per_top_level_group():
    layers.enable((120, 80))
    cv = Canvas(120, 80)
    d = Drawing(120, 80, background="#ffffff")
    with d.group("box"):
        d.rect(10, 10, 40, 40, fill="#ff0000")
    with d.group("dot"):
        d.circle(90, 40, 15, fill="#0000ff")
    d.draw(cv, log=lambda n: layers.mark(n, cv))
    st = layers.take(cv)
    names = [L.name for L in st]
    assert "box" in names and "dot" in names and "background" in names
    assert d in V.last_drawings()
    V.clear_drawings()
    assert V.last_drawings() == []


def test_draw_fits_box_and_scales():
    d = Drawing(100, 50, background="#000000")
    d.rect(0, 0, 50, 50, fill="#ffffff")
    cv = Canvas(400, 400, (0.5, 0.5, 0.5))
    d.draw(cv)                                                       # 4x, centred vertically
    assert cv.buf[100, 10].mean() > 0.99 and cv.buf[100, 300].mean() < 0.01 and abs(cv.buf[20, 10].mean() - .5) < 1e-3
    small = Canvas(200, 200, (0.5, 0.5, 0.5))
    d.draw(small, box=(100, 100, 200, 150))
    assert small.buf[120, 110].mean() > 0.99 and abs(small.buf[50, 50].mean() - 0.5) < 1e-3


def test_themes_complete():
    for name in ("light", "dark", "blueprint", "paper", "pastel", "neon", "mono"):
        t = V.THEMES[name]
        for k in ("background", "surface", "ink", "muted", "grid", "stroke_width", "radius", "shadow", "font"):
            assert k in t
        assert len(t.accents) == 8 and len(set(t.accents)) == 8
        assert t.accent(9) == t.accents[1]
    assert V.theme("dark") is not V.THEMES["dark"]


def test_icons_all_render_and_export():
    names = V.icon_names()
    assert len(names) >= 40
    d = Drawing(48 * 8, 48 * ((len(names) + 7) // 8), background="#ffffff")
    for i, n in enumerate(names):
        g = d.icon(n, 24 + (i % 8) * 48, 24 + (i // 8) * 48, 40, color="#000", fill="#ccddff")
        bb = g.bbox()
        assert max(bb[2] - bb[0], bb[3] - bb[1]) > 10      # a straight-line icon (minus) has zero height
    ink = _ink(d.render())
    for i in range(len(names)):
        cx, cy = 24 + (i % 8) * 48, 24 + (i // 8) * 48
        assert ink[cy - 20:cy + 20, cx - 20:cx + 20].max() > 0.9, names[i]
    one = V.icon("gear", 64)
    ET.fromstring(one.to_svg().split("?>", 1)[1])
    with pytest.raises(KeyError):
        d.icon("nope", 0, 0)


def test_speed_2000_primitives():
    rng = np.random.default_rng(0)
    d = Drawing(1920, 1080, theme="light")
    for i in range(2000):
        x, y = rng.uniform(0, 1900, 2)
        k = i % 4
        if k == 0:
            d.rect(x, y, 60, 30, r=6, fill=d.theme.accent(i), stroke=d.theme.ink)
        elif k == 1:
            d.circle(x, y, 18, fill=d.theme.accent(i), stroke="#fff", stroke_width=2)
        elif k == 2:
            d.line(x, y, x + 80, y + 30, stroke_width=2, arrow="end")
        else:
            d.line(x, y, x + 120, y, dash="dash")
    t0 = time.time()
    d.draw(Canvas(1920, 1080))
    assert time.time() - t0 < 3.0


def test_transparent_background():
    for bg in (False, "none"):
        d = Drawing(40, 40, theme="dark", background=bg)
        d.rect(10, 10, 10, 10, fill="#ff0000")
        cv = Canvas(40, 40, (0.0, 0.0, 1.0))
        d.draw(cv)
        assert np.allclose(cv.buf[2, 2], (0, 0, 1)) and cv.buf[15, 15, 0] > 0.99
        assert "<rect width" not in d.to_svg()
    assert Drawing(10, 10, theme="dark").background == V.THEMES["dark"].background


def test_embed_into_box():
    src = Drawing(100, 50)
    src.rect(0, 0, 100, 50, fill="#000")
    d = Drawing(400, 400)
    g = d.embed(src, box=(0, 0, 200, 400))
    assert np.allclose(g.bbox(), (0, 150, 200, 250))
    g = d.embed(src, box=(0, 0, 200, 400), align="top-right")
    assert np.allclose(g.bbox(), (0, 0, 200, 100))
    g = d.embed(src, box=(0, 0, 400, 100), align="left")
    assert np.allclose(g.bbox(), (0, 0, 200, 100))


def test_plain_text_by_default_and_markup_option():
    s = "* p < 0.05, ** p < 0.01, *** p < 0.001  a__b ~~c~~ ==d=="
    d = Drawing(900, 80, background="#ffffff")
    t = d.text(10, 40, s, size=20, anchor="lm")
    m = d.text(10, 40, "**bold**", size=20, anchor="lm", markup=True, add=False)
    assert t.markup is False and m.markup is True
    w_plain = d.measure_text("**x**", size=20)[0]
    w_markup = d.measure_text("**x**", size=20, markup=True)[0]
    assert w_plain > w_markup * 2
    svg = d.to_svg()
    assert "** p &lt; 0.01, *** p" in svg and "a__b ~~c~~ ==d==" in svg
    ink = _ink(d.render())
    assert ink.sum() > 0


def test_measure_text_counts_trailing_spaces():
    w1 = V.measure_text("1.", 20)[0]
    w2 = V.measure_text("1. ", 20)[0]
    w3 = V.measure_text("1.  ", 20)[0]
    assert 0 < w1 < w2 < w3
    assert V.measure_text("a \nbb", 20)[0] >= V.measure_text("bb", 20)[0]
    d = Drawing(100, 40)
    assert d.text(50, 20, "1. ", anchor="rm").text == "1."


def test_per_corner_radii_and_crisp_rounded_rects():
    d = Drawing(80, 60, background="#ffffff")
    s = d.rect(10.3, 10.3, 40, 30, r=(8, 8, 0, 0), fill="#000")
    assert s.params["radii"] == (8.0, 8.0, 0.0, 0.0)
    ink = _ink(d.render())
    assert ink[38, 10] > 0.99 and ink[38, 49] > 0.99                  # square bottom corners, pixel-aligned
    assert ink[10, 10] < 0.2 and ink[10, 49] < 0.2                    # rounded top corners
    assert ink[40, 30] < 0.01 and ink[39, 30] > 0.99                  # crisp bottom edge
    svg = d.to_svg()
    ET.fromstring(svg.split("?>", 1)[1])
    assert "<path" in svg and "<rect x" not in svg
    assert "<rect x" in Drawing(10, 10).rect(0, 0, 5, 5, r=(2, 2, 2, 2)).drawing.to_svg()


def test_triangle_marker():
    d = Drawing(120, 40, background="#ffffff")
    d.line(10, 20, 110, 20, stroke="#000", stroke_width=2, marker_end="triangle")
    ink = _ink(d.render())
    assert ink[20, 107] > 0.5 and ink[18, 100] > 0.9                  # solid triangle head
    assert ink[20, 95] > 0.9                                           # no notch behind the tip
    assert d.to_svg().count("<path") >= 2


def test_dash_fit_starts_and_ends_on_dashes():
    assert V._fit_dash((3.0, 1.0), 2.0) is None
    p = V._fit_dash((10, 1.5, 1.5, 1.5), 57.0, center=True)
    n = round((57.0 - p[0]) / sum(p))
    assert n % 2 == 0 and abs(p[0] + n * sum(p) - 57.0) < 1e-9
    c = V._fit_dash((4, 3), 100.0, closed=True)
    assert abs(100.0 / sum(c) - round(100.0 / sum(c))) < 1e-9
    d = Drawing(120, 20, background="#ffffff")
    d.line(10.5, 10, 109.5, 10, stroke="#000", stroke_width=2, dash=(7, 5), dash_fit=True)
    ink = _ink(d.render())
    assert ink[10, 11] > 0.9 and ink[10, 108] > 0.9                   # both ends are drawn
    svg = d.to_svg()
    da = re.search(r'stroke-dasharray="([^"]+)"', svg).group(1).split(",")
    on, off = float(da[0]), float(da[1])
    n = round((99 - on) / (on + off))
    assert abs(on + n * (on + off) - 99) < 0.01
    d2 = Drawing(100, 100)
    d2.path("M10 10 L90 10 M10 50 L60 50", stroke="#000", dash="dash", dash_fit=True)
    assert d2.to_svg().count("stroke-dasharray") == 2                 # one fitted pattern per subpath


def test_cap_anchor_centres_capitals():
    d = Drawing(200, 100, background="#ffffff")
    d.text(100, 50, "H8", size=60, anchor="mc", fill="#000")
    ink = _ink(d.render())
    rows = np.nonzero(ink.max(1) > 0.5)[0]
    assert abs((rows[0] + rows[-1] + 1) / 2 - 50) < 1.5
    t = d.find(None) or d.items[0]
    b = t.bbox()
    assert b[1] < 50 < b[3]
    g = d.icon("star", 20, 20, 20, anchor="mc")
    assert np.allclose(g.bbox()[1] + g.bbox()[3], 40, atol=4)


def _doc_blocks():
    with open(os.path.join(ROOT, "skills", "pixelcanvas", "references", "vector.md"), encoding="utf-8") as f:
        return re.findall(r"```python\n(.*?)```", f.read(), re.S)


@pytest.mark.parametrize("i", range(len(_doc_blocks())))
def test_doc_snippets_run(i, tmp_path, monkeypatch):
    os.makedirs(tmp_path / "outputs", exist_ok=True)
    monkeypatch.chdir(tmp_path)
    exec(_doc_blocks()[i], {"__name__": "__doc__"})


def test_scene_renders():
    sys.path.insert(0, os.path.join(ROOT, "scenes"))
    import importlib
    mod = importlib.import_module("scenes.vector_icons")
    steps = []
    cv = mod.render(480, 270, log=steps.append)
    assert cv.w == 480 and "header" in steps and len(steps) > 40


def test_strict_options_vector():
    from pixelcanvas import vector as V
    d = V.Drawing(100, 100)
    with pytest.raises(TypeError, match="did you mean 'stroke_width'"):
        d.line(0, 0, 5, 5, stroke_wdith=2)
    s = d.rect(0, 0, 5, 5, fill_colour="#f00", lw=2, alpha=0.5)
    assert s is not None
    with pytest.raises(ValueError, match="'check'"):
        d.icon("chek", 10, 10)
    with pytest.raises(ValueError, match="'dark'"):
        V.theme("dak")


# ---- app icon tiles ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("glyph", ["notes", "weather", "gear", "calculator", "terminal"])
def test_app_icon_glyphs(glyph):
    cv = Canvas(200, 200, "#eeeeee")
    box = V.app_icon(cv, (50, 40, 150, 140), glyph, shadow="soft")
    x0, y0, x1, y1 = (int(v) for v in box)
    tile = cv.buf[y0 + 3:y1 - 3, x0 + 3:x1 - 3]
    white = (tile.min(-1) > 0.95).mean()
    assert 0.012 < white < 0.6                                 # a white glyph on the gradient tile


def test_app_icon_long_shadow_stays_in_its_cell():
    cv = Canvas(400, 200, "#eeeeee")
    before = cv.buf.copy()
    V.app_icon(cv, (40, 40, 140, 140), "mail", shadow="long", cell=(0, 0, 200, 200))
    changed = np.abs(cv.buf - before).max(-1) > 1e-4
    assert changed[:, :200].any() and not changed[:, 200:].any()
    tile_px = cv.buf[150:200, 150:200]                        # the long shadow reaches down-right inside the cell
    assert float(np.abs(tile_px - before[150:200, 150:200]).max()) > 0.01


def test_app_icon_sheet_twelve_labelled_entries():
    from pixelcanvas import manifest
    cv = Canvas(800, 600)
    apps = ["notes", "calendar", "mail", "tasks", "files", "chat", "camera", "music", "maps", "weather", "settings",
            "calculator"]
    boxes = V.app_icon_sheet(cv, apps, cols=4)
    assert len(boxes) == 12
    objs = [e for e in cv.manifest if e["kind"] == "object"]
    assert len(objs) == 12 and all("icon" in e["aliases"] for e in objs)
    assert [t["text"] for t in cv.text_log][:3] == ["Notes", "Calendar", "Mail"]
    issues = manifest.check(cv, "A sheet of twelve app icons with long shadows, each labelled.")
    assert not [w for w in issues if w["rule"] == "prompt_count"], issues


def test_app_icon_options_are_checked():
    cv = Canvas(100, 100)
    with pytest.raises(ValueError, match="glyph 'calendr'.*'calendar'"):
        V.app_icon(cv, (10, 10, 90, 90), "calendr")
    with pytest.raises(ValueError, match="'long'"):
        V.app_icon(cv, (10, 10, 90, 90), "mail", shadow="lnog")
    with pytest.raises(TypeError, match="did you mean 'radius'"):
        V.app_icon_sheet(cv, ["mail"], raduis=0.2)


def test_app_icon_glyph_and_accent_colours():
    box = (20, 20, 220, 220)
    base = Canvas(240, 240, "#ffffff")
    V.app_icon(base, box, "calendar", colors=("#9ec5ff", "#2b62d6"), shadow="none")
    dark = Canvas(240, 240, "#ffffff")
    V.app_icon(dark, box, "calendar", colors=("#9ec5ff", "#2b62d6"), shadow="none", glyph_color="#101820",
               accent_color="#ff2a2a")

    def share(cv, pred):
        a = cv.buf[40:200, 40:200, :3]
        return float(pred(a).mean())
    red = lambda a: (a[..., 0] > 0.8) & (a[..., 1] < 0.3) & (a[..., 2] < 0.3)   # noqa: E731
    near_black = lambda a: a.max(axis=-1) < 0.2                                   # noqa: E731
    white = lambda a: a.min(axis=-1) > 0.93                                       # noqa: E731
    assert share(dark, red) > 0.01 and share(dark, near_black) > 0.04
    assert share(base, red) == 0 and share(base, white) > 0.04      # default: the white glyph of today


def test_app_icon_accent_colour_reaches_the_weather_sun_and_the_sheet():
    box = (20, 20, 220, 220)
    cv = Canvas(240, 240, "#ffffff")
    V.app_icon(cv, box, "weather", colors=("#7fd0ff", "#2b86d8"), shadow="none", accent_color="#ff2a2a")
    a = cv.buf[:, :, :3]
    assert ((a[..., 0] > 0.8) & (a[..., 1] < 0.3) & (a[..., 2] < 0.3)).sum() > 100
    sheet = Canvas(480, 300)
    V.app_icon_sheet(sheet, ["calendar", "weather"], cols=2, glyph_color="#101820", accent_color="#ffd54a")
    with pytest.raises(TypeError):
        V.app_icon_sheet(sheet, ["mail"], acent_color="#fff")
