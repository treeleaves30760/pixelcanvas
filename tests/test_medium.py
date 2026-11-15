# -*- coding: utf-8 -*-
"""pixelcanvas.medium: buffers recorded from the drawing (albedo, hollow, glow), the six media and their signatures,
detection from a request, the style dials. Also runs every python block of skills/pixelcanvas/references/medium.md
("medium.md")."""
import contextlib
import io
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import medium as MD  # noqa: E402
from pixelcanvas.medium import media, style  # noqa: E402
from pixelcanvas.medium.core import srgb_to_lin, luma  # noqa: E402
from pixelcanvas.studio import commands as SC  # noqa: E402
from pixelcanvas.studio import medium_stage as MS  # noqa: E402
from pixelcanvas.studio.state import open_picture  # noqa: E402

DOC = os.path.join(ROOT, "skills", "pixelcanvas", "references", "medium.md")

DRAFT = '''# %% wall
# wall: warm plaster
cv.gradient([(0, "#d9c7a8"), (0.62, "#c9b38f"), (0.62, "#7a4e2c"), (1, "#5e3a20")])

# %% ball
# ball: red
sk.paint(sk.circle(0.3, 0.55, 0.14), "#c8402f", volume=0.6, shadow="ground", name="ball")

# %% dish
# dish: white
sk.paint(sk.ellipse(0.66, 0.58, 0.16, 0.1), "#e8e4dc", volume=-0.6, name="dish")

# %% lamp
# lamp: yellow
sk.paint(sk.circle(0.84, 0.22, 0.05), "#ffd27a", glow=dict(color="#ffb347", radius=0.2, strength=1.0), name="lamp")
'''


def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = SC.main(["--dir", "art"] + [str(a) for a in argv])
    return code, buf.getvalue()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A small still life drawn in the studio -> (picture, compositor, buffers)."""
    d = tmp_path_factory.mktemp("medium")
    old = os.getcwd()
    os.chdir(d)
    had = os.environ.get("PIXELCANVAS_STUDIO_CMD")
    os.environ["PIXELCANVAS_STUDIO_CMD"] = "studio"
    try:
        (d / "draft.py").write_text(DRAFT, encoding="utf-8")
        code, out = run("new", "still", "--prompt", "a red ball and a white dish under a lamp", "--size",
                        "320x200", "--from", "draft.py")
        assert code == 0, out
        pic = open_picture("art", "still")
        sess = SC.Session(pic)
        sess.mute = True
        sess.render("test", quiet=True)
        b, _ = MS.buffers(sess.comp, pic, max_side=320)
        yield pic, sess.comp, b
    finally:
        os.chdir(old)
        if had is None:                                  # never leak into later test files (subprocess tests)
            os.environ.pop("PIXELCANVAS_STUDIO_CMD", None)
        else:
            os.environ["PIXELCANVAS_STUDIO_CMD"] = had


def _px(b, x, y):
    H, W = b["id"].shape
    return int(round(y * H)), int(round(x * W))


def test_recorded_layers_and_albedo_equals_paint_colour(built):
    _, _, b = built
    assert set(b["recorded"]) >= {"ball", "dish", "lamp"}
    assert not [x for x in b["estimated"] if x in ("ball", "dish")]
    y, x = _px(b, 0.3, 0.55)
    want = srgb_to_lin(np.array([0xc8, 0x40, 0x2f], np.float32) / 255)
    assert np.allclose(b["albedo"][y, x], want, atol=0.02)
    y, x = _px(b, 0.66, 0.58)
    assert np.allclose(b["albedo"][y, x], srgb_to_lin(np.array([0xe8, 0xe4, 0xdc], np.float32) / 255), atol=0.02)


def test_volume_bulges_and_a_hollow_dips(built):
    _, _, b = built
    Z = b["Z"]
    H, W = Z.shape
    yb, xb = _px(b, 0.3, 0.55)
    ye, xe = _px(b, 0.3 + 0.12 * H / W, 0.55)
    assert Z[yb, xb] > Z[ye, xe] + 1.0                   # the ball: the middle stands out
    want = srgb_to_lin(np.array([0xe8, 0xe4, 0xdc], np.float32) / 255)
    dish = np.abs(b["albedo"] - want).max(-1) < 0.02     # the dish's own pixels
    ys, xs = np.nonzero(dish)
    cy, cx = ys.mean(), xs.mean()
    d = np.hypot((ys - cy) / (np.ptp(ys) / 2 + 1), (xs - cx) / (np.ptp(xs) / 2 + 1))
    centre, rim = Z[ys[d < 0.25], xs[d < 0.25]].mean(), Z[ys[(d > 0.7) & (d < 0.9)], xs[(d > 0.7) & (d < 0.9)]].mean()
    assert centre < rim - 0.3                            # the dish: the middle sinks


def test_glow_lights_its_neighbour(built, tmp_path):
    pic, comp, b = built
    layers = MS.layers_for(comp, pic)
    from pixelcanvas.medium.buffers import build
    dark = [dict(L, records=[r for r in (L["records"] or []) if r["kind"] != "light"] or None)
            if L["name"] == "lamp" else L for L in layers]
    for L in dark:
        if L["name"] == "lamp":
            L["P"] = np.minimum(L["P"], L["A"][..., None])   # the glow's light beyond the paint gone too
    lit = build(layers, light=(-0.6, -0.8), max_side=320, horizon=b["horizon"])
    off = build(dark, light=(-0.6, -0.8), max_side=320, horizon=b["horizon"])
    want = srgb_to_lin(np.array([0xe8, 0xe4, 0xdc], np.float32) / 255)
    dish = np.abs(b["albedo"] - want).max(-1) < 0.02     # the lamp's neighbour
    assert dish.sum() > 50
    assert luma(lit["pts"])[dish].mean() > luma(off["pts"])[dish].mean() + 0.01
    assert luma(lit["beauty"])[dish].mean() > luma(off["beauty"])[dish].mean()


def _layer_colours(img, b, min_px=40):
    from pixelcanvas.medium.core import srgb_to_oklab
    lab = srgb_to_oklab(np.clip(img, 0, 1))
    out = {}
    for k in np.unique(b["gid"]):
        own = (b["gid"] == k)
        if own.sum() >= min_px:
            out[int(k)] = (float(np.median(lab[..., 0][own])), lab[..., 1][own].mean(), lab[..., 2][own].mean())
    return out


def test_render_keeps_every_layers_local_colour(built):
    """Hue and saturation (Oklab a, b) and the median lightness of every layer survive the light."""
    _, _, b = built
    drawn, rend = _layer_colours(b["drawn"], b), _layer_colours(MD.paint(b, "render"), b)
    assert len(drawn) >= 3
    for k, (L, a, bb) in drawn.items():
        L2, a2, b2 = rend[k]
        assert abs(a2 - a) < 0.02 and abs(b2 - bb) < 0.02, (k, (L, a, bb), (L2, a2, b2))
        assert abs(L2 - L) < 0.08, (k, L, L2)


def test_an_estimated_grey_animal_stays_grey():
    """A layer with no records (estimated) keeps its drawn grey: no bleaching to white, no hue shift."""
    from pixelcanvas.medium.buffers import build
    H, W = 160, 240
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sky = np.zeros((H, W, 3), np.float32) + np.array([0.55, 0.7, 0.85], np.float32)
    sky[yy > 0.6 * H] = (0.6, 0.48, 0.25)
    m = (((xx - 120) / 70) ** 2 + ((yy - 90) / 40) ** 2 < 1).astype(np.float32)
    shade = 0.75 + 0.25 * np.clip(1 - (yy - 50) / 80, 0, 1)
    grey = np.stack([0.42 * shade] * 3, -1) * m[..., None]
    layers = [dict(name="savanna", role="backdrop", P=sky, A=np.ones((H, W), np.float32)),
              dict(name="elephant", role="object", P=grey.astype(np.float32), A=m, standing=True)]
    b = build(layers, max_side=240)
    assert "elephant" in b["estimated"]
    img = MD.paint(b, "render")
    own = (b["gid"] == 1) & (m > 0.9)
    from pixelcanvas.medium.core import srgb_to_oklab
    lab_d, lab_r = srgb_to_oklab(b["drawn"])[own], srgb_to_oklab(img)[own]
    assert abs(np.median(lab_r[:, 0]) - np.median(lab_d[:, 0])) < 0.06          # not bleached
    assert np.abs(lab_r[:, 1:].mean(0)).max() < 0.03                                # still grey
    sky_d, sky_r = srgb_to_oklab(b["drawn"])[:20].reshape(-1, 3), srgb_to_oklab(img)[:20].reshape(-1, 3)
    assert abs(sky_r[:, 2].mean() - sky_d[:, 2].mean()) < 0.02                      # the drawn sky stays blue


@pytest.fixture(scope="module")
def painted(built):
    _, _, b = built
    return {m: MD.paint(b, m) for m in MD.MEDIA}


def test_each_medium_has_its_signature(painted, built):
    _, _, b = built
    imgs = painted
    H, W = imgs["render"].shape[:2]
    for m, img in imgs.items():
        assert img.shape == (H, W, 3) and np.isfinite(img).all() and 0 <= img.min() and img.max() <= 1, m
    for i, a in enumerate(MD.MEDIA):                     # every medium is its own picture
        for c in MD.MEDIA[i + 1:]:
            assert np.abs(imgs[a] - imgs[c]).mean() > 0.02, (a, c)

    def hf(img):                                         # high-frequency energy: marks, texture
        L = luma(img)
        return float(np.abs(np.diff(L, axis=0)).mean() + np.abs(np.diff(L, axis=1)).mean())

    def sat(img):
        return float((img.max(-1) - img.min(-1)).mean())

    px_ = int(round(style.STYLES["pixel"].extra["pixel"] * W / 1200)) or 1
    blocky = imgs["pixel"][::max(px_, 1), ::max(px_, 1)]
    assert len(np.unique((blocky * 255).astype(np.uint8).reshape(-1, 3), axis=0)) <= 64       # a small palette
    assert sat(imgs["ink"]) < 0.5 * sat(imgs["render"])                                         # ink: no colour
    assert hf(imgs["oil"]) > hf(imgs["render"])                                                # strokes, canvas
    wall = imgs["watercolour"][: H // 10, W // 3: 2 * W // 3]
    assert luma(wall).mean() > luma(imgs["render"][: H // 10, W // 3: 2 * W // 3]).mean()    # paper shows through
    fl = (luma(imgs["flat"]) * 255).astype(np.uint8)
    rn = (luma(imgs["render"]) * 255).astype(np.uint8)
    assert len(np.unique(fl)) < len(np.unique(rn))                                              # value bands


def test_stages_strip(built):
    _, _, b = built
    st = MD.stages(b, "ink")
    assert [s[0] for s in st] == ["flat silhouettes", "relief clay", "lit colour", "ink"]
    clay = st[1][1]
    assert float((clay.max(-1) - clay.min(-1)).mean()) < 0.12          # clay is grey


def test_detect_and_canonical():
    assert MD.detect("a watercolour of a harbour") == "watercolour"
    assert MD.detect("an oil painting of pears") == "oil"
    assert MD.detect("gouache sketch of a fox") == "oil"
    assert MD.detect("an ink drawing of a crow") == "ink"
    assert MD.detect("pixel art castle") == "pixel"
    assert MD.detect("a photo of a knife") is None and MD.detect("a realistic portrait") is None
    assert MD.detect("a realistic 3D render of a chair") == "render"
    assert MD.detect("a poster for a jazz night") is None
    assert MD.canonical("watercolor") == "watercolour"


def test_style_dials_change_the_style():
    s = style.style_for("oil", strength=0.5, mark=2.0, palette=12, paper="#ffffff")
    base = style.STYLES["oil"]
    assert s.mark_size == pytest.approx(base.mark_size * 2) and s.palette == 12 and s.ground_color == "#ffffff"
    assert s.texture == pytest.approx(base.texture * 0.5)
    assert style.style_for("pixel", mark=2.0).extra["pixel"] == pytest.approx(base_px := style.STYLES["pixel"]
                                                                               .extra["pixel"] * 2) and base_px > 0


def test_mark_dial_is_visible(built):
    _, _, b = built
    a = MD.paint(b, "pixel")
    c = MD.paint(b, "pixel", mark=2.5)
    assert np.abs(a - c).mean() > 0.01


def test_recorder_is_off_outside_a_section():
    from pixelcanvas import Canvas
    from pixelcanvas.illustrate import Sketch
    from pixelcanvas.medium import record
    cv = Canvas(64, 48)
    sk = Sketch(cv)
    sk.paint(sk.circle(0.5, 0.5, 0.2), "#ff0000", volume=0.5)       # nothing recorded, nothing breaks
    r = record.start(cv)
    try:
        sk.paint(sk.circle(0.5, 0.5, 0.2), "#ff0000", volume=-0.5, glow="#ffcc00", line="#000000")
    finally:
        record.stop(r)
    kinds = [i["kind"] for i in r.items]
    assert "body" in kinds and "light" in kinds and "line" in kinds
    body = next(i for i in r.items if i["kind"] == "body")
    assert body["volume"] < 0 and np.allclose(body["color"], [1, 0, 0])


def test_form_solid_records_true_normals_and_its_hollow():
    from pixelcanvas import Canvas
    from pixelcanvas import form
    from pixelcanvas.medium import record
    cv = Canvas(160, 120)
    r = record.start(cv)
    try:
        form.lathe(cv, (0.5, 0.8), [(0.1, 0), (0.2, 0.05), (0.26, 0.2)], base="#ffffff", open=True)
    finally:
        record.stop(r)
    sol = [i for i in r.items if i["kind"] == "solid"]
    assert sol and sol[0].get("normals") is not None and sol[0].get("hollow") is not None
    assert sol[0]["hollow"].max() == 255


def _blocks():
    if not os.path.isfile(DOC):
        return []
    return re.findall(r"```python\n(.*?)```", open(DOC, encoding="utf-8").read(), re.S)


def test_medium_md_exists_and_is_short():
    text = open(DOC, encoding="utf-8").read()
    assert len(text.splitlines()) <= 250
    assert "medium" in text and _blocks()


@pytest.mark.parametrize("i", range(len(_blocks())))
def test_medium_md_blocks_run(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = _blocks()[i]
    exec(compile(code, f"medium.md block {i + 1}", "exec"), {"__name__": "__medium_doc__"})


def test_creature_and_figure_kits_record_their_parts():
    from pixelcanvas import Canvas
    from pixelcanvas.creature import creature
    from pixelcanvas.figure import Figure
    from pixelcanvas.medium import record
    cv = Canvas(240, 160)
    r = record.start(cv)
    try:
        creature(cv, "elephant", at=(0.5, 0.9), size=0.5)
    finally:
        record.stop(r)
    sol = [i for i in r.items if i["kind"] == "solid"]
    assert sol and sol[0].get("normals") is not None and sol[0].get("painted") is not None
    cv = Canvas(240, 160)
    r = record.start(cv)
    try:
        Figure().draw(cv, at=(0.5, 0.95))
    finally:
        record.stop(r)
    bodies = [i for i in r.items if i["kind"] == "body"]
    assert len(bodies) > 5 and any(i["volume"] > 0 for i in bodies) and any(i["volume"] == 0 for i in bodies)


def test_floor_shadow_fades_and_is_not_cut(built):
    _, _, b = built
    fs = b["floor_shadow"]
    assert fs.max() > 0.1
    hz = int(b["horizon"])
    if hz > 2:
        assert np.abs(np.diff(fs[max(hz - 3, 0):hz + 3], axis=0)).max() < 0.25        # no straight cut at the wall
