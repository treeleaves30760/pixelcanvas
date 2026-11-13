# -*- coding: utf-8 -*-
"""Freehand illustration toolbox (pixelcanvas.illustrate) and the Pillow / cairo drawing layers on Canvas."""
import importlib
import os
import re

import numpy as np
import pytest

from pixelcanvas import Canvas, manifest
from pixelcanvas import illustrate as il
from pixelcanvas.illustrate import Sketch, grow, edge, signed_distance, paper, limit_palette, grade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cv(w=320, h=180, c="#ffffff"):
    return Canvas(w, h, c)


def test_shapes_use_fractions_and_are_antialiased():
    cv = _cv()
    sk = Sketch(cv)
    m = sk.circle(0.5, 0.5, 0.25)                          # radius = 0.25 of the height = 45 px
    assert m.shape == (180, 320) and m.dtype == np.float32
    assert abs(float(m.sum()) - np.pi * 45 ** 2) / (np.pi * 45 ** 2) < 0.02
    assert ((m > 0.05) & (m < 0.95)).any()                 # soft edge pixels exist
    b = sk.box(0.1, 0.2, 0.3, 0.6)
    assert abs(float(b.sum()) - 64 * 72) / (64 * 72) < 0.03
    for mk in (sk.ellipse(0.5, 0.5, 0.2, 0.1, angle=30), sk.polygon([(0.1, 0.1), (0.5, 0.1), (0.3, 0.6)]),
               sk.blob([(0.2, 0.2), (0.6, 0.2), (0.7, 0.6), (0.3, 0.7)]), sk.arc(0.5, 0.5, 0.2, 200, 340, 0.02)):
        assert mk.max() > 0.99 and 0 <= mk.min()


def test_frames_scale_rotate_mirror():
    cv = _cv()
    sk = Sketch(cv)
    with sk.at(0.5, 0.5, 0.5):
        assert sk.px(0, 0) == (160.0, 90.0) and abs(sk.size(1) - 90) < 1e-6
        a = sk.circle(0.4, 0, 0.1)
        with sk.at(0, 0, 1, mirror=True):
            b = sk.circle(0.4, 0, 0.1)
        with sk.at(0, 0, 1, rotate=90):
            c = sk.circle(0.4, 0, 0.1)
    ya, xa = np.argwhere(a > 0.5).mean(0)
    yb, xb = np.argwhere(b > 0.5).mean(0)
    yc, xc = np.argwhere(c > 0.5).mean(0)
    assert abs(xa - 196) < 1.5 and abs(xb - 124) < 1.5          # mirrored left-right
    assert abs(yc - 126) < 1.5 and abs(xc - 160) < 1.5           # +90 degrees = clockwise (down on screen)


def test_stroke_width_profile_and_taper():
    cv = _cv()
    sk = Sketch(cv)
    even = sk.stroke([(0.1, 0.5), (0.9, 0.5)], 0.1)
    grow_ = sk.stroke([(0.1, 0.5), (0.5, 0.5), (0.9, 0.5)], [0.02, 0.1, 0.2])
    tap = sk.stroke([(0.1, 0.5), (0.5, 0.5), (0.9, 0.5)], 0.1, taper="end")
    col = lambda m, x: float(m[:, x].sum())
    assert abs(col(even, 60) - 18) < 1.5
    assert col(grow_, 40) < col(grow_, 160) < col(grow_, 270)
    assert col(tap, 40) > col(tap, 250) > col(tap, 285)
    with pytest.raises(ValueError):
        sk.stroke([(0, 0), (1, 1)], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        sk.stroke([(0, 0), (1, 1)], 0.1, taper="tip")


def test_paint_fill_shade_highlight_line_shadow_glow():
    cv = _cv(c="#808080")
    sk = Sketch(cv)
    m = sk.circle(0.5, 0.5, 0.3)
    sk.paint(m, "#e08040", shade=0.6, highlight=0.6, line=True, shadow=True, glow="#ffcc66")
    b = cv.buf
    lit, dark = b[60, 135], b[120, 185]                     # upper-left vs lower-right inside the disc
    assert lit.mean() > dark.mean() + 0.05
    assert b[90, 160 + 53].mean() < 0.45                   # darker outline on the edge
    g = _cv(c="#ffffff")
    s2 = Sketch(g)
    s2.paint(s2.box(0.2, 0.2, 0.8, 0.8), [(0, "#ff0000"), (1, "#0000ff")], angle=90)
    assert g.buf[40, 160, 0] > g.buf[140, 160, 0] and g.buf[40, 160, 2] < g.buf[140, 160, 2]
    s2.paint(s2.circle(0.5, 0.5, 0.2), [(0, "#ffffff"), (1, "#000000")], angle="radial")


@pytest.mark.parametrize("kind", ["seigaiha", "scales", "stripes", "dots", "speckle", "checker", "tiles"])
def test_pattern_fills_only_inside_the_mask(kind):
    cv = _cv(c="#1d3f78")
    sk = Sketch(cv)
    m = sk.box(0.2, 0.2, 0.8, 0.8)
    a = sk.pattern(m, kind, "#ffffff", size=0.1, width=None if kind in ("dots", "speckle") else 0.02, angle=15)
    assert float(a[m == 0].max(initial=0)) == 0.0
    inside = a[m == 1]
    assert 0.05 < float(inside.mean()) < 0.95 and inside.max() > 0.9      # a motif, not a fill
    with pytest.raises(ValueError, match="did you mean"):
        sk.pattern(m, "seigaha", "#fff")


def test_volume_keeps_whites_light_and_rounds_them():
    cv = _cv(c="#808080")
    sk = Sketch(cv)
    m = sk.circle(0.5, 0.5, 0.3)
    sk.paint(m, "#ffffff", volume=1.0)
    centre, rim = cv.buf[90, 160], cv.buf[90 + 50, 160 + 10]
    assert centre.min() > 0.97                               # the middle stays white
    assert rim.mean() < centre.mean() - 0.04                 # rounded towards the edge
    assert rim[2] > rim[0] and rim.min() > 0.7               # a light, cool tint, not grey


def test_stripes_run_along_the_angle():
    sk = Sketch(_cv())
    m = sk.box(0.1, 0.1, 0.9, 0.9)
    flat = sk.pattern(m, "stripes", "#000", size=0.1, width=0.02, opacity=0)
    tall = sk.pattern(m, "stripes", "#000", size=0.1, width=0.02, angle=90, opacity=0)
    assert flat[60:120, 160].std() > 0.2 and flat[90, 60:260].std() < 0.05      # horizontal lines
    assert tall[90, 60:260].std() > 0.2 and tall[60:120, 160].std() < 0.05      # vertical lines


def test_arrow_straight_curved_open_and_outlined():
    cv = _cv(c="#ffffff")
    sk = Sketch(cv)
    m = sk.arrow((0.1, 0.5), (0.9, 0.5), width=0.02)
    head, shaft = m[:, 274].sum(), m[:, 150].sum()
    assert head > 2 * shaft > 0                          # a head wider than the shaft at the tip end
    assert m[90, 285] > 0.9 and m[90, 34] > 0.5
    for style in ("open", "none"):
        o = sk.arrow((0.1, 0.2), (0.9, 0.2), width=0.02, style=style, both=True, color="#ff0000")
        assert (o[:, 40].sum() > 2 * o[:, 150].sum()) == (style == "open")
    c = sk.arrow((0.1, 0.9), (0.9, 0.9), via=(0.5, 0.6), width=0.02, outline="#000000", color="#ffcc00")
    ys = np.argwhere(c[:, 160] > 0.5)[:, 0]
    assert abs(ys.mean() - 0.6 * 180) < 4                   # the curve passes through via
    with pytest.raises(ValueError, match="did you mean"):
        sk.arrow((0, 0), (1, 1), style="filed")


def test_label_contrasts_stays_inside_and_records_text():
    cv = _cv(640, 360, c="#ffffff")
    sk = Sketch(cv)
    sk.paint(sk.box(0.5, 0, 1, 1), "#101820")
    b1 = sk.label("light ground", (0.25, 0.3), to=(0.25, 0.8), size=0.06)
    b2 = sk.label("dark ground", (0.75, 0.3), to=(0.75, 0.8), arrow=True, size=0.06)
    b3 = sk.label("off the edge", (0.99, 0.99), size=0.06)
    for b in (b1, b2, b3):
        assert b[0] >= 0 and b[1] >= 0 and b[2] <= 640 and b[3] <= 360

    def plate(b):                       # the plate's own colour: inside its left padding, clear of edge and glyphs
        pad = 1.4 * 0.35 * 0.06 * 360  # label(): horizontal padding = 1.4 * pad * size
        x, y = int(b[0] + 0.4 * pad), int((b[1] + b[3]) / 2)
        return float(np.median(cv.buf[y - 2:y + 3, x - 1:x + 2].mean(-1)))
    assert plate(b1) < 0.3 and plate(b2) > 0.85            # dark plate on white, light plate on dark
    assert cv.buf[240, 160].mean() < 0.4                    # the leader line reaches down on the light side
    rows = manifest.summary(cv)
    assert {r["kind"] for r in rows if "ground" in r["name"]} == {"text"}
    assert not [r for r in rows if r["kind"] == "object"]
    sk.label("plain", (0.3, 0.6), box=None)
    with pytest.raises(ValueError):
        sk.label("x", (0.5, 0.5), box="round")


def test_cv_text_takes_fractions_and_pixels_alike():
    a, b = _cv(), _cv()
    a.text("Hello", (0.5, 0.5), 0.2, "#000000", anchor="mm")
    b.text("Hello", (160, 90), 36, "#000000", anchor="mm")
    assert np.abs(a.buf - b.buf).max() < 1e-5
    c, d = _cv(), _cv()
    c.text("Fit me", box=(0.1, 0.1, 0.9, 0.4), color="#000000")
    d.text("Fit me", box=(32, 18, 288, 72), color="#000000")
    assert np.abs(c.buf - d.buf).max() < 1e-5


def test_from_array_uint8_and_float_widths_in_draw():
    cv = Canvas.from_array(np.full((4, 5, 3), 255, np.uint8))
    assert abs(float(cv.buf.max()) - 1.0) < 1e-6 and cv.shape == (4, 5, 3)
    cv = _cv(c="#000000")
    with cv.draw() as d:
        d.line([d.p(0, 0.5), d.p(1, 0.5)], fill="#ffffff", width=d.k(0.03))
    assert cv.buf[90, 160].min() > 0.95


DRAFTS = os.path.join(ROOT, "..", "plans", "build", "tmp", "release", "freehand")


def test_page_example_and_signatures(tmp_path, monkeypatch):
    """The main page's freehand example renders, and every call its bullet list names exists with those names."""
    ex, sig = os.path.join(DRAFTS, "page_example.py"), os.path.join(DRAFTS, "page_signatures.md")
    if not (os.path.exists(ex) and os.path.exists(sig)):
        pytest.skip("page drafts not in this checkout")
    ns = {}
    exec(compile(open(ex, encoding="utf-8").read(), ex, "exec"), ns)
    assert len(open(ex, encoding="utf-8").read().splitlines()) <= 30 and ns["DEFAULT_SIZE"]
    cv = ns["render"](192, 108)
    assert np.isfinite(cv.buf).all() and 0.05 < cv.buf.mean() < 0.95
    import inspect
    from pixelcanvas import masks, noise, canvas as canvas_mod
    owners = {"sk": Sketch, "cv": Canvas, "Canvas": Canvas, "masks": masks, "noise": noise, "d": canvas_mod.PillowLayer}
    text = open(sig, encoding="utf-8").read()
    for call in re.findall(r"`([^`]*\(.*?\))`", text):
        for obj, name, args in re.findall(r"(sk|cv|Canvas|masks|noise|d)\.(\w+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", call):
            fn = getattr(owners[obj], name, None)
            assert fn is not None, "%s.%s named on the page does not exist" % (obj, name)
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                continue
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                continue
            for kw in re.findall(r"(\w+)=", args):
                assert kw in params, "%s.%s has no option %r" % (obj, name, kw)


def test_small_shapes_get_a_flat_finish_large_ones_keep_it():
    def spread(sk_finish, r):
        cv = _cv(c="#808080")
        sk = Sketch(cv, finish=sk_finish)
        m = sk.circle(0.5, 0.5, r)
        sk.paint(m, "#e08040", shade=0.8, highlight=0.8)
        inside = cv.buf[m > 0.99]
        return float(inside.std(0).mean())
    assert spread("auto", 0.015) < 1e-4                     # a tiny detail (under 0.3 % of the canvas): one flat colour
    assert spread("auto", 0.1) > 0.02                       # an object a fifth of the height across: modelled
    assert spread("auto", 0.3) > 0.03                       # large: modelled as before
    assert abs(spread("auto", 0.3) - spread("full", 0.3)) < 1e-6
    assert spread("full", 0.015) > 0.005 and spread("flat", 0.3) < 1e-4
    with pytest.raises(ValueError):
        Sketch(_cv(), finish="smooth")


def test_ink_line_is_near_black_and_bolder():
    cv = _cv(c="#ffffff")
    sk = Sketch(cv, finish="flat")
    sk.paint(sk.circle(0.5, 0.5, 0.3), "#f0c030", line="ink")
    darkest = cv.buf[90, 205:222].mean(-1).min()
    assert darkest < 0.15                                   # a near-black line on the edge


def test_glass_and_level_read_as_see_through_container():
    cv = _cv(c="#c9c4bb")
    sk = Sketch(cv)
    flask = sk.polygon([(0.4, 0.2), (0.6, 0.2), (0.58, 0.85), (0.42, 0.85)])
    liquid = sk.level(flask, 0.5, "#3070d0")
    sk.glass(flask, name="flask")
    assert cv.buf[140, 160, 2] > cv.buf[140, 160, 0] + 0.2     # liquid below the level
    above = cv.buf[60, 165]
    assert np.abs(above - np.array([0.79, 0.77, 0.73])).max() < 0.15   # the ground shows through above it
    rows = [r for r in manifest.summary(cv) if r["name"] == "flask"]
    assert rows and liquid[150, 160] > 0.9 and liquid[60, 160] < 0.1


def test_state_cues_draw_inside_and_around():
    cv = _cv(640, 360, c="#c9c4bb")
    sk = Sketch(cv)
    tin = sk.paint(sk.box(0.1, 0.1, 0.4, 0.6), "#c8653a")
    cr = sk.crack(tin, seed=1)
    assert cr.sum() > 50 and float((cr * (1 - tin)).max()) < 1e-3       # the crack stays on the tin
    sh = sk.shards((0.7, 0.75), size=0.1, n=5)
    assert sh.sum() > 300
    sp = sk.spill((0.55, 0.5), size=0.1, on=0.85)
    ys = np.argwhere(sp > 0.5)[:, 0]
    assert ys.max() > 0.8 * 360 and ys.min() < 0.55 * 360               # a stream down to a pool on the ground
    st = sk.steam((0.25, 0.1), size=0.08)
    assert st.sum() > 30
    d = sk.drops((0.9, 0.2), n=3, spread=0.05)
    assert d.sum() > 20
    patches = sk.stain(tin, "#9a4a1e", 0.5)
    cover = float((patches[tin > 0.99] > 0.4).mean())
    assert 0.1 < cover < 0.9 and float((patches * (1 - tin)).max()) < 1e-3      # patches, on the tin only


def test_row_and_grid_give_equal_non_overlapping_frames():
    sk = Sketch(_cv(640, 360))
    cells = sk.grid(6, (0.05, 0.1, 0.95, 0.9))
    assert len(cells) == 6 and len({round(c[2], 6) for c in cells}) == 1
    masks_ = []
    for f in cells:
        with sk.at(*f):
            masks_.append(sk.circle(0, 0, 0.5))
    total = np.sum(masks_, axis=0)
    assert total.max() <= 1.0 + 1e-3                         # nothing overlaps
    assert all(m.sum() > 0.9 * np.pi * (f[2] * 360 / 2) ** 2 for m, f in zip(masks_, cells))   # nothing clipped
    r = sk.row(4, (0.0, 0.4, 1.0, 0.6))
    assert [round(c[0], 3) for c in r] == [0.125, 0.375, 0.625, 0.875]
    last = sk.grid(5, (0, 0, 1, 1), cols=3)
    assert abs((last[3][0] + last[4][0]) / 2 - 0.5) < 1e-6   # an incomplete last row is centred


def test_spot_clip_and_text():
    cv = _cv()
    sk = Sketch(cv)
    face = sk.circle(0.3, 0.5, 0.2)
    a = sk.spot(0.45, 0.5, 0.2, 0.2, "#ff0000", 1.0, clip=face)
    assert float(a[face == 0].max()) == 0.0 and a.max() > 0.5
    bb = sk.text("Hello", 0.7, 0.5, 0.15, "#000000", anchor="mm", tracking=0.2)
    assert bb and bb[0] < 224 < bb[2]


def test_layer_and_sticker_wrap_the_art():
    cv = _cv(c="#c0e0ff")
    sk = Sketch(cv)
    with sk.layer() as g:
        sk.paint(sk.circle(0.3, 0.5, 0.15), "#ff0000")
    assert abs(float(g.mask.max()) - 1) < 1e-5 and cv.buf[90, 96, 0] > 0.95
    with sk.sticker(border=0.05, ink="#222222", name="dot"):
        sk.paint(sk.circle(0.7, 0.5, 0.12), "#00aa00")
    d = 0.12 * 180 + 4 + 5                                  # outside the art, inside the die-cut
    assert cv.buf[90, int(224 + d)].mean() > 0.95           # white border
    assert cv.buf[90, 224, 1] > 0.5                         # art on top
    assert "dot" in [r["name"] for r in manifest.summary(cv)]


def test_grow_edge_and_signed_distance():
    sk = Sketch(_cv())
    m = sk.circle(0.5, 0.5, 0.2)                             # r = 36 px
    d = signed_distance(m, 30)
    assert abs(float(d[90, 160 + 36 + 10]) - 10) < 1.0 and abs(float(d[90, 160]) + 30) < 1e-3
    g = grow(m, 6)
    assert abs(float(g.sum()) - np.pi * 42 ** 2) / (np.pi * 42 ** 2) < 0.03
    e = edge(m, 4, "outside")
    assert e[90, 160 + 38] > 0.9 and e[90, 160 + 33] < 0.1
    with pytest.raises(ValueError):
        edge(m, 4, "outer")


def test_distance_without_scipy_is_identical(monkeypatch):
    if il._ndi is None:
        pytest.skip("scipy not installed: the numpy path is the only path")
    m = Sketch(_cv()).blob([(0.2, 0.2), (0.7, 0.3), (0.6, 0.8), (0.3, 0.6)])
    fast = signed_distance(m, 25)
    monkeypatch.setattr(il, "_ndi", None)
    slow = signed_distance(m, 25)
    assert np.abs(fast - slow).max() < 1e-4


def test_wash_paper_palette_grade():
    cv = _cv(c="#f4ecdc")
    sk = Sketch(cv)
    a = sk.wash(sk.circle(0.5, 0.5, 0.3), "#3a70c0", seed=1)
    ring, middle = float(a[90, 160 + 52]), float(a[90, 160])
    assert 0 < middle < 1 and ring > 0                       # transparent, pooled at the wet edge
    paper(cv, seed=2)
    limit_palette(cv, ["#f4ecdc", "#3a70c0"], 1.0)
    u = np.unique(np.round(cv.buf.reshape(-1, 3), 4), axis=0)
    assert len(u) <= 2
    grade(cv, "moody")
    with pytest.raises(ValueError):
        grade(cv, "moddy")
    assert np.isfinite(cv.buf).all()


def test_strict_options():
    sk = Sketch(_cv())
    with pytest.raises(TypeError, match="did you mean"):
        sk.paint(sk.circle(0.5, 0.5, 0.1), "#000", hilight=0.3)
    with pytest.raises(TypeError):
        sk.paint(sk.circle(0.5, 0.5, 0.1), "#000", shadow=dict(blurr=3))
    with pytest.raises(AttributeError, match="sk.box"):
        il.rectangle


def test_pillow_layer_antialiases_blends_and_records():
    cv = _cv(c="#000080")
    with cv.draw(ss=3, name="disc") as d:
        d.ellipse([*d.p(0.25, 0.25), *d.p(0.75, 0.75)], fill="#ffffff")
        d.ellipse([*d.p(0.1, 0.1), *d.p(0.2, 0.2)], fill=(255, 0, 0, 128))       # translucent over nothing
        d.line(d.pts([(0.0, 0.9), (1.0, 0.9)]), fill="#ffff00", width=int(d.k(0.02)))
        d.text(d.p(0.8, 0.1), "A", font=d.font(0.1), fill="#00ff00")
    b = cv.buf
    assert b[90, 160].min() > 0.99                          # opaque centre
    assert ((d.alpha > 0.05) & (d.alpha < 0.95)).any()      # anti-aliased coverage
    r = b[int(0.15 * 180), int(0.15 * 320)]
    assert 0.4 < r[0] < 0.6 and r[2] > 0.2                  # half red over the navy, not a black hole
    assert abs(float(d.alpha[int(0.15 * 180), int(0.15 * 320)]) - 0.5) < 0.03
    assert b[162, 10, 0] > 0.9                              # the yellow line
    assert "disc" in [r["name"] for r in manifest.summary(cv)]


def test_pillow_layer_finish_options():
    cv = _cv(c="#808080")
    with cv.draw(ss=2, shade=0.6, line=True, shadow=True) as d:
        d.ellipse([*d.p(0.3, 0.2), *d.p(0.7, 0.8)], fill="#e08040")
    assert cv.buf[60, 130].mean() > cv.buf[120, 190].mean()
    assert d.mask.max() > 0.99
    with pytest.raises(TypeError, match="did you mean"):
        cv.draw(shaed=0.3)


def test_paint_spec_ground_shadow_and_planar_faces():
    cv = _cv(c="#808080")
    sk = Sketch(cv, light=(-0.6, -0.8), finish="full")
    sk.paint(sk.circle(0.5, 0.45, 0.3), "#c03020", spec=1.0, shadow="ground")
    lit = cv.buf[81 - 27 - 6:81 - 27 + 6, 160 - 22:160 - 10].max(axis=(0, 1)).mean()   # upper left of the ball
    assert lit > 0.75 and cv.buf[81, 160].mean() < 0.5                 # a bright hot spot on a red body
    assert cv.buf[136, 160].mean() < 0.47                               # contact shadow on the floor under it
    assert cv.buf[136, 40].mean() > 0.49                                # and not far off to the side
    cv2 = _cv(c="#808080")
    sk2 = Sketch(cv2, light=(-1.0, 0.0), finish="full")
    face = sk2.polygon([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)])
    sk2.paint(face, "#c89060", volume=1.0)
    row = cv2.buf[90, 70:250].mean(-1)
    assert row[5] > row[-5] + 0.05                                      # darker away from the light
    assert np.all(np.diff(row[5:-5]) <= 1e-3)                           # a plane: one smooth ramp, no bulge
    cv3 = _cv(c="#808080")
    with cv3.draw(ss=2, volume=0.5, spec=0.8, shadow="ground") as d:
        d.ellipse([*d.p(0.4, 0.2), *d.p(0.6, 0.6)], fill="#e08040")
    assert d.mask.max() > 0.99


def test_cairo_layer_when_installed():
    pytest.importorskip("cairo")
    cv = _cv(c="#000000")
    with cv.cairo(ss=2, name="c") as d:
        d.ctx.set_source_rgba(1, 1, 1, 1)
        d.ctx.arc(*d.p(0.5, 0.5), d.k(0.2), 0, 6.2832)
        d.ctx.fill()
    assert cv.buf[90, 160].min() > 0.99 and cv.buf[5, 5].max() < 0.01


PAGE = "illustration.md"            # the reference page whose code blocks this test runs


def test_doc_snippets_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = os.path.join(ROOT, "skills", "pixelcanvas", "references", PAGE)
    src = open(path, encoding="utf-8").read()
    assert len(src.splitlines()) <= 250
    blocks = re.findall(r"```python\n(.*?)```", src, re.S)
    assert len(blocks) >= 4
    for i, b in enumerate(blocks):
        ns = {}
        exec(compile(b, f"{PAGE} block {i}", "exec"), ns)
        assert np.isfinite(ns["cv"].buf).all()


def test_page_teaches_nameable_objects_and_comics():
    src = open(os.path.join(ROOT, "skills", "pixelcanvas", "references", PAGE), encoding="utf-8").read()
    sec = src.split("## Things a stranger can name", 1)[1].split("\n## ", 1)[0]
    rows = [r for r in sec.splitlines() if r.startswith("| ") and not r.startswith("| Object")]
    assert sum(1 for r in rows for cell in r.split("|")[1::2] if cell.strip()) >= 15     # fifteen objects
    for call in ("sk.glass", "sk.level", "sk.crack", "sk.row", 'finish="auto"'):
        assert call in src
    assert 'line="ink"' in src and '"dots"' in src                                      # comic recipe


def _clipped(cv, name):
    return [r["clipped_pct"] for r in manifest.summary(cv) if r["name"] == name][0]


def test_shape_cut_by_the_frame_reports_its_hidden_share():
    cv = _cv(640, 360, "#c9c4bb")
    sk = Sketch(cv)
    sk.paint(sk.circle(0.0, 0.5, 0.25), "#2a2a2a", name="wheel")               # centre on the left edge
    assert 45 <= _clipped(cv, "wheel") <= 55
    short = [i.get("short") or "" for i in manifest.check(cv, prompt="a black wheel")]
    assert any("wheel: cut off at the left edge" in s for s in short)
    inside = _cv(640, 360, "#c9c4bb")
    Sketch(inside).paint(Sketch(inside).circle(0.5, 0.5, 0.25), "#2a2a2a", name="wheel")
    assert _clipped(inside, "wheel") == 0
    assert not [i for i in manifest.check(inside, prompt="a black wheel") if "cut off" in (i.get("short") or "")]


def test_group_and_pillow_layer_report_the_cut_share():
    cv = _cv(640, 360, "#c9c4bb")
    sk = Sketch(cv)
    with sk.layer(name="cart"):
        sk.paint(sk.box(0.4, 0.4, 0.6, 0.6), "#3060a0")                       # inside
        sk.paint(sk.box(-0.2, 0.4, 0.05, 0.6), "#a03030")                     # mostly off the left edge
    assert 20 <= _clipped(cv, "cart") <= 30                                    # (0.2 of 0.8 wide) off: 25 %
    with cv.draw(name="lamp") as d:
        d.ellipse([*d.p(0.9, 0.1), *d.p(1.1, 0.3)], fill="#f0c040")          # half past the right edge
    assert 45 <= _clipped(cv, "lamp") <= 55


def test_combined_masks_keep_a_sensible_extent():
    from pixelcanvas.illustrate import extent_of
    sk = Sketch(_cv(640, 360))
    a, b = sk.circle(0.0, 0.5, 0.25), sk.box(-0.1, 0.4, 0.3, 0.9)
    ea, eb = extent_of(a), extent_of(b)
    assert ea[0] < 0 and eb[0] < 0                                             # the off-canvas part is known
    cut = extent_of(a * b)
    assert cut == (max(ea[0], eb[0]), max(ea[1], eb[1]), min(ea[2], eb[2]), min(ea[3], eb[3]))
    both = extent_of(np.maximum(a, b))
    assert both == (min(ea[0], eb[0]), min(ea[1], eb[1]), max(ea[2], eb[2]), max(ea[3], eb[3]))
    assert extent_of(a * (1 - b)) == ea and extent_of(sk.grow(a, 0.02))[0] < ea[0]
    assert type(np.asarray(a)) is np.ndarray and float(a.sum()) > 0           # still a plain mask to numpy


def test_counts_and_positions_reach_the_report():
    cv = _cv(480, 320, "#f4f1ea")
    sk = Sketch(cv)
    for x in (0.15, 0.3, 0.45):
        sk.paint(sk.circle(x, 0.7, 0.08), "#d8322c", name="pear")
    sk.paint(sk.box(0.7, 0.5, 0.85, 0.85), "#2f6fd0", name="teapot")
    names = [r["name"] for r in manifest.summary(cv)]
    assert names.count("pear") == 1 and "pear #3" in names and "teapot" in names
    assert any("teapot is right of pear" in r for r in manifest.relations(cv))
    issues = manifest.check(cv, prompt="three pears to the left of a teapot")
    assert not [i for i in issues if "count" in str(i.get("kind", i)) and i.get("severity") in ("warn", "fail")]


def test_illustration_template_renders_clean():
    cv = importlib.import_module("scenes._template_illustration").render(192, 108)
    assert np.isfinite(cv.buf).all() and 0.1 < cv.buf.mean() < 0.95


@pytest.mark.parametrize("name", ["illustration_chef", "illustration_stickers", "illustration_ukiyoe",
                                  "illustration_jazz"])
def test_scenes_render_small(name):
    cv = importlib.import_module("scenes." + name).render(192, 108)
    assert cv.shape == (108, 192, 3) and np.isfinite(cv.buf).all() and 0.05 < cv.buf.mean() < 0.95
