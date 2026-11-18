"""easel surface: materials guessed from the names, marks laid on the surface layer clipped to each area, one
undoable step, energy aimed at the profile; the detail stage's todo and gate; the working canvas; line pressure and
coloured line art."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from .test_draw_flow import draw, ok  # noqa: F401  (fixture and helpers)


def state(name="pic"):
    return json.load(open(f"{name}.easel/state.json", encoding="utf-8"))


def project(name="pic"):
    from pixelcanvas.easel.project import Project
    return Project.find(f"{name}.easel")


def layer_alpha(name="surface", proj="pic"):
    p = project(proj)
    arr = p.layer(name, create=False)
    if arr is None:
        return np.zeros((p.H, p.W), np.float32)
    a = np.asarray(arr)[..., 3].astype(np.float32)
    return a / 255.0 if a.max() > 1.0 else a


def flat_project(run, size="400x400", request="anime girl with a red scarf"):
    """A bust through flat: face with an eye in it, a scarf, the background; the stage is shade afterwards."""
    ok(run("new", "pic", "--size", size, "--request", request))
    ok(run("do", "plan subject girl --box 150,40,850,1000; plan part face --box 330,150,670,520; "
                 "plan part eye-l --box 390,280,470,340; plan part scarf --box 280,560,720,760; "
                 "plan light top-left; palette make; palette pick 1"))
    ok(run("stage", "done"))
    ok(run("stage", "done", "--because", "test"))
    ok(run("line", "340,200 330,350 380,470 500,520 620,470 670,350 660,200 500,150", "--closed"))
    ok(run("line", "395,285 465,285 465,335 395,335", "--closed", "--no-join"))
    ok(run("line", "280,580 720,580 720,750 280,750", "--closed"))
    ok(run("stage", "done", "--because", "test"))
    ok(run("flat", "auto"))
    ok(run("stage", "done", "--because", "test"))


def laid_areas(text):
    """The areas a surface step names on its first line: 'step N  surface a, b  (layer surface) ...'."""
    ln = [x for x in text.splitlines() if x.startswith("step ")][0]
    return {n.strip() for n in ln.split("  surface ", 1)[1].split("(layer")[0].split(",")}


# ============================================================================================ 1 materials

def test_guess_material_and_material():
    from pixelcanvas.easel.coords import UsageError
    from pixelcanvas.easel.draw_surface import BASE, MATERIALS, guess_material, material
    cases = {"hair": "hair", "bangs": "hair", "face": "skin", "neck": "skin", "hands": "skin",
             "scarf": "cloth-smooth", "coat": "cloth-rough", "sweater": "knit", "boots": "leather",
             "sword": "metal", "window": "glass", "pines": "foliage", "grass": "foliage", "trunk": "bark",
             "staff": "bark", "mountains": "stone", "rock": "stone", "lake": "water", "sky": "sky",
             "clouds": "cloud", "background": "paper", "zorblat": "plain"}
    for name, mat in cases.items():
        assert guess_material(name) == mat, name
    for name in ("face", "neck", "hands"):
        assert guess_material(name, animal=True) == "fur", name
    for name in ("eye-l", "mouth", "title"):
        assert guess_material(name) is None, name
    with pytest.raises(UsageError) as e:
        material("nope")
    assert "unknown material 'nope'" in str(e.value) and "cloth-rough" in str(e.value)   # no close name: the list
    with pytest.raises(UsageError) as e:
        material("lether")
    assert "did you mean leather" in str(e.value.args[0])
    base_lines = MATERIALS["hair"].get("lines", BASE["lines"])
    assert material("hair", {"lines_x": 2})["lines"] == pytest.approx(2 * base_lines)
    assert material("hair")["lines"] == pytest.approx(base_lines)


# ============================================================================================ 2 surface auto + undo

def test_surface_auto_lays_a_clipped_layer_and_undo_clears_it(draw):
    flat_project(draw)
    st = state()
    regs = list(st["regions"])
    assert {"face", "scarf", "background"} <= set(regs)
    text = ok(draw("surface", "auto"))
    st = state()
    names = [e["name"] for e in st["layers"]]
    assert "surface" in names and names.index("surface") + 1 == names.index("line")
    surf = st["draw"]["surface"]
    eyes = [r for r in regs if r.startswith("eye")]
    for r in regs:
        if r in eyes:
            assert r not in surf, r
        else:
            assert r in surf, r
            assert f"{r} {surf[r]['material']}" in text
    assert surf["face"]["material"] == "skin" and surf["scarf"]["material"] == "cloth-smooth"
    assert surf["background"]["material"] == "paper"
    p = project()
    union = np.zeros((p.H, p.W), bool)
    for r in surf:
        union |= p.region_mask(r)
    band = cv2.dilate(union.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0     # 2 px tolerance
    a = layer_alpha()
    assert a[union].max() > 0.05                     # something was laid
    assert a[~band].max() < 0.02, a[~band].max()     # nothing outside the areas it covers
    for r in eyes:
        assert a[p.region_mask(r) & ~band].max(initial=0.0) < 0.02
    ok(draw("undo"))
    assert layer_alpha().max() < 0.02
    assert not state()["draw"].get("surface")


# ============================================================================================ 3 one area / auto again

def test_surface_one_area_material_and_auto_lays_only_new_areas(draw):
    flat_project(draw)
    ok(draw("surface", "face"))
    assert set(state()["draw"]["surface"]) == {"face"}
    face_before = dict(state()["draw"]["surface"]["face"])
    a_face = layer_alpha().copy()
    # auto without --again lays only the areas that have no surface yet
    text = ok(draw("surface", "auto"))
    surf = state()["draw"]["surface"]
    assert {"face", "scarf", "background"} <= set(surf)
    assert surf["face"] == face_before
    laid = laid_areas(text)
    assert "scarf" in laid and "background" in laid and "face" not in laid
    face_m = project().region_mask("face")
    assert np.abs(layer_alpha()[face_m] - a_face[face_m]).max() < 1e-3     # the face was not re-laid
    # one area's material changes that record only
    before = json.loads(json.dumps(surf))
    ok(draw("surface", "scarf", "--material", "leather"))
    after = state()["draw"]["surface"]
    assert after["scarf"]["material"] == "leather"
    for n in before:
        if n != "scarf":
            assert after[n] == before[n], n
    # every area has a surface now: auto without --again lays them all again (nothing new to lay)
    text = ok(draw("surface", "auto"))
    assert {"face", "scarf", "background"} <= laid_areas(text)


def test_second_auto_keeps_a_material_chosen_by_hand(draw):
    flat_project(draw)
    ok(draw("surface", "auto"))
    ok(draw("surface", "scarf", "--material", "leather"))
    ok(draw("surface", "auto"))
    assert state()["draw"]["surface"]["scarf"]["material"] == "leather"


# ============================================================================================ 4 energy

def test_amount_raises_the_fine_energy(draw):
    from pixelcanvas.easel import draw_surface
    flat_project(draw, request="a painterly portrait of a girl with a red scarf, oil painting")

    def fine():
        draw_surface._MEAS.clear()
        return draw_surface.area_report(project(), ["scarf"])[0]["fine"]

    def energy(material, amount):
        ok(draw("surface", "scarf", "--material", material, "--amount", amount))
        e = fine()
        ok(draw("undo"))
        return e
    base = fine()                   # the scarf's outline and edges before any surface
    # painterly: more amount means more brush strokes turning the planes, so more energy ...
    hi, lo = energy("cloth-smooth", 2), energy("cloth-smooth", 0.5)
    assert hi >= 0.95 * lo and lo >= 0.9 * base, (base, lo, hi)
    # ... but strands are budgeted: a raised amount does not pile them on (no fur swirl, no scratches)
    hi, lo = energy("fur", 3), energy("fur", 1)
    assert hi < 1.5 * lo, (lo, hi)


# ============================================================================================ 5 workflow

def test_detail_todo_starts_with_surface_and_ticks(draw):
    flat_project(draw)
    ok(draw("stage", "done", "--because", "test"))          # shade -> detail
    st = state()
    assert st["stage"] == "detail"
    todo = [t for t in st["todo"] if t["stage"] == "detail"]
    assert todo[0]["key"] == "surface" and todo[0]["cmd"].startswith("easel surface auto")
    assert todo[0].get("status") != "done"
    text = ok(draw("next"))
    assert "easel surface auto" in text.split("NEXT:")[1]
    ok(draw("surface", "auto"))
    item = [t for t in state()["todo"] if t["stage"] == "detail" and t["key"] == "surface"][0]
    assert item["status"] == "done"


# ============================================================================================ 6 gates

class _ScaleProxy:
    """draw_core with work_scale 3: gate_holes skips 1x canvases under 640 px (the small test canvas)."""

    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, k):
        return getattr(self._mod, k)

    @staticmethod
    def work_scale(p):
        return 3


def _entry(key, low, value=0.5, q1=0.59, q3=0.72):
    return {"key": key, "low": low, "value": value, "q1": q1, "q3": q3, "z": -3.0 if low else 3.0,
            "text": f"{key} {'low' if low else 'high'} (fake)"}


def test_detail_gate_includes_surface_measures(draw, monkeypatch):
    from pixelcanvas.easel import detail_measure, draw_checks, draw_surface
    flat_project(draw)
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("surface", "auto"))
    assert state()["stage"] == "detail"
    monkeypatch.setattr(draw_surface, "dc", _ScaleProxy(draw_surface.dc))
    monkeypatch.setattr(draw_surface, "picture_measure", lambda p, keep=True, lines=True: {})
    # cel: a fine band below the range does not block (the flats stay flat) ...
    monkeypatch.setattr(detail_measure, "outside", lambda m, group, keys=None: [_entry("bandL_1", True, 0.001,
                                                                                         0.01, 0.03)])
    assert draw_surface.gate_holes(project(), "detail") == []
    # ... but flats broken up below the profile's range do
    monkeypatch.setattr(detail_measure, "outside", lambda m, group, keys=None: [_entry("flat_share", True, 0.02,
                                                                                         0.11, 0.42)])
    draw_checks._HOLES.clear()
    hs = [h for h in draw_checks.holes(project(), "detail") if h["metric"].startswith("detail ")]
    assert hs, draw_checks.holes(project(), "detail")
    assert hs[0]["metric"] == "detail flat_share" and hs[0]["fix"].startswith("easel surface auto --again --amount")
    # line taper: a high one passes, a low one blocks
    p = project()
    monkeypatch.setattr(detail_measure, "outside", lambda m, group, keys=None: [_entry("line_taper", False, 0.96)])
    assert draw_surface.gate_holes(p, "detail") == []
    monkeypatch.setattr(detail_measure, "outside", lambda m, group, keys=None: [_entry("line_taper", True, 0.3)])
    hs = draw_surface.gate_holes(p, "detail")
    assert len(hs) == 1 and hs[0]["metric"] == "detail line_taper"
    assert hs[0]["fix"].startswith("easel surface auto --again --lines")


# ============================================================================================ 7 working canvas

def test_working_canvas_scale_and_export_reduces(draw):
    ok(draw("new", "pic", "--size", "700x700", "--request", "anime girl with a red scarf"))
    ok(draw("plan", "subject", "girl", "--box", "150,40,850,1000"))
    st = state()
    assert (st["width"], st["height"]) == (2100, 2100)
    assert st["work_scale"] == 3 and list(st["deliver"]) == [700, 700]
    text = ok(draw("export", "final.png"))
    out = state().get("exported") or "final.png"
    with Image.open(out) as im:
        assert im.size == (700, 700), text


def _one_plan(run, name, size, request):
    ok(run("new", name, "--size", size, "--request", request))
    ok(run("plan", "subject", "thing", "--box", "150,150,850,850"))
    st = state(name)
    return st


@pytest.mark.parametrize("request_", ["anime girl with a red scarf", "minimalist poster of a red apple"])
def test_small_and_poster_projects_stay_1x(draw, request_):
    size = "400x400" if "poster" not in request_ else "700x700"
    st = _one_plan(draw, "pic", size, request_)
    w, h = (int(v) for v in size.split("x"))
    assert (st["width"], st["height"]) == (w, h)
    assert st.get("work_scale", 1) == 1 and not st.get("deliver")


# ============================================================================================ 8 lines

def test_line_pressure_and_colour(draw):
    from pixelcanvas.easel import draw_line
    ok(draw("new", "pic", "--size", "400x400", "--request", "anime girl with a red scarf"))
    ok(draw("do", "plan subject girl --box 150,40,850,1000; plan part face --box 200,200,800,800; "
                  "plan light top-left; palette make; palette pick 1"))
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    ang = np.linspace(np.pi * 0.05, np.pi * 1.6, 12)
    pts = " ".join(f"{500 + 280 * np.cos(a):.0f},{500 + 280 * np.sin(a):.0f}" for a in ang)
    ok(draw("line", pts, "--no-join"))
    rec = state()["draw"]["lines"][-1]
    assert "lt" in rec and len(rec["lt"]) == 2
    p = project()
    path = draw_line.record_path(p, rec)
    assert len(path) > 50
    f = draw_line.pressure_profile(p, rec, path)
    assert f.mean() == pytest.approx(1.0, abs=0.05)
    assert np.percentile(f, 90) / np.percentile(f, 10) > 1.3
    old = {k: v for k, v in rec.items() if k != "lt"}
    assert np.all(draw_line.pressure_profile(p, old, path) == 1.0)

    def luma(c):
        return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    fill = (250, 220, 200)
    c = draw_line.line_colour(fill)
    assert luma(c) < luma(fill) and luma(c) > 25, c


def test_cel_surface_keeps_the_flats(draw):
    """Anime / cel: the surface pass adds a few clean shapes, no strands, grain or backdrop strokes; most of each
    area keeps its flat colour."""
    import numpy as np
    from pixelcanvas.easel import draw_surface
    flat_project(draw)
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("surface", "auto"))
    p = project()
    surf = p.layer("surface", create=False)
    assert surf is not None
    bg = p.region_mask("background")
    bops = draw_surface.build(p, "background", bg, draw_surface.material("paper"), focus=0,
                              light=np.array([-0.5, -0.7, 0.5]), base_img=p.composite_u8("export"),
                              others=np.zeros(bg.shape, bool), group="anime-cel")[0]
    assert not any(type(o).__name__ == "Stroke" for o in bops)     # the backdrop: a soft value turn, no streaks
    for n in ("face", "scarf"):
        m = p.region_mask(n)
        assert float((surf[..., 3][m] > 0.05).mean()) < 0.6, n      # most of the area keeps its flat
    ops = draw_surface.build(p, "scarf", p.region_mask("scarf"), draw_surface.material("cloth-smooth"),
                             focus=2, light=np.array([-0.5, -0.7, 0.5]), base_img=p.composite_u8("export"),
                             others=np.zeros(p.region_mask("scarf").shape, bool), group="anime")[0]
    assert not any(type(o).__name__ == "Spray" for o in ops)
