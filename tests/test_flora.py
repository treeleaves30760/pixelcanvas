# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.flora: registry, structure, determinism, resolution independence, doc snippets."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import flora as F  # noqa: E402
from pixelcanvas.layers import LayerRecorder  # noqa: E402


def _cv(w=160, h=120):
    return Canvas(w, h, "#d8e2ea")


def test_find_maps_free_text():
    cases = {"a big oak tree": "oak", "palm tree": "palm", "weeping willow": "willow", "sunflowers": "sunflower",
             "potted monstera": "potted monstera", "snake plant": "potted snake plant", "mossy boulder": "mossy rock",
             "christmas tree": "fir", "cherry blossom tree": "cherry", "red maple": "autumn maple",
             "saguaro cactus": "saguaro", "fly agaric": "mushroom", "rocks": "rock pile", "tulips": "tulip"}
    for q, want in cases.items():
        assert F.find(q) == want, (q, F.find(q))
    assert F.find("spaceship") is None
    assert set(F.names("tree")) >= {"oak", "maple", "birch", "spruce", "palm", "baobab", "bamboo", "saguaro"}
    assert len(F.names()) >= 80


def test_every_registered_plant_paints_something():
    for name in F.names():
        cv = _cv()
        p = F.plant(cv, name, at=(0.5, 0.9), height=0.5, seed=1)
        assert isinstance(p, F.Plant)
        assert p.mask.shape == (120, 160) and p.mask.max() > 0.5, name
        assert np.isfinite(cv.buf).all(), name


def test_plant_anchors_and_shadow():
    cv = _cv(320, 240)
    before = cv.buf.copy()
    p = F.tree(cv, "oak", at=(0.4, 0.9), height=0.7, seed=2, shadow=True)
    assert abs(p.base[0] - 0.4 * 320) < 1 and abs(p.base[1] - 0.9 * 240) < 1
    assert p.top[1] < p.base[1] - 0.6 * 240
    assert p.top[1] <= p.canopy[1] < p.base[1]
    x0, y0, x1, y1 = p.bbox
    assert 0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240
    # ground shadow darkened pixels outside the tree mask, away from the light (to the right)
    outside = p.mask < 0.01
    darker = (cv.buf.sum(-1) < before.sum(-1) - 0.03) & outside
    ys, xs = np.nonzero(darker)
    assert len(xs) > 50 and xs.mean() > p.base[0]


def test_deterministic_and_seeded():
    a, b, c = _cv(), _cv(), _cv()
    F.tree(a, "maple", seed=5, height=0.8)
    F.tree(b, "maple", seed=5, height=0.8)
    F.tree(c, "maple", seed=6, height=0.8)
    assert np.array_equal(a.buf, b.buf)
    assert not np.array_equal(a.buf, c.buf)


def test_resolution_independent_coverage():
    cov = []
    for w, h in ((192, 108), (384, 216)):
        cv = Canvas(w, h)
        p = F.tree(cv, "spruce", at=(0.5, 0.95), height=0.8, seed=3)
        cov.append(p.mask.mean())
    assert abs(cov[0] - cov[1]) / cov[1] < 0.2


def test_species_silhouettes_differ():
    """Columnar trees are narrow, oaks are broad: aspect of the mask bbox."""
    def aspect(name):
        cv = Canvas(400, 300)
        p = F.tree(cv, name, at=(0.5, 0.95), height=0.8, seed=1)
        ys, xs = np.nonzero(p.mask > 0.5)
        return (xs.max() - xs.min()) / (ys.max() - ys.min())
    assert aspect("cypress") < 0.35 < aspect("spruce") < aspect("oak")


def test_canopy_has_structure_not_a_blob():
    cv = Canvas(400, 300, "#ffffff")
    p = F.tree(cv, "oak", at=(0.5, 0.95), height=0.9, seed=4)
    lum = cv.buf.mean(-1)
    inside = p.mask > 0.99
    cx, cy = map(int, p.canopy)
    lit = lum[max(cy - 60, 0):cy, max(cx - 60, 0):cx][inside[max(cy - 60, 0):cy, max(cx - 60, 0):cx]].mean()
    shade = lum[cy:cy + 60, cx:cx + 60][inside[cy:cy + 60, cx:cx + 60]].mean()
    assert lit > shade + 0.03                         # lit upper-left, shadowed lower-right
    gy, gx = np.gradient(lum)
    assert np.hypot(gx, gy)[inside].mean() > 0.02     # leaf-level detail
    assert 0.0 < (p.mask[p.bbox[1]:p.bbox[3], p.bbox[0]:p.bbox[2]] < 0.5).mean() < 0.9   # holes / ragged edge


def test_seasons_styles_and_log_steps():
    steps = []
    cv = _cv(240, 180)
    F.tree(cv, "maple", season="autumn", seed=1, log=steps.append)
    assert steps == ["maple trunk", "maple canopy"]
    red = cv.buf[..., 0] - cv.buf[..., 1]
    assert (red > 0.15).mean() > 0.02
    for style in ("flat", "cute"):
        cv = _cv()
        p = F.tree(cv, "oak", style=style, seed=1)
        assert p.mask.max() > 0.9 and p.mask[0:5, 0:5].max() < 0.01
    cv = _cv()
    bare = F.tree(cv, "maple", season="winter", seed=1)
    full = F.tree(_cv(), "maple", seed=1)
    assert bare.mask.mean() < full.mask.mean() * 0.6


def test_forest_depth_and_haze():
    cv = Canvas(320, 180, "#c8d4dc")
    p = F.forest(cv, region=(0, 0.5, 1, 0.8), n=20, seed=2, haze=("#c8d4dc", 0.7))
    assert len(p.extra["trees"]) == 20
    far = [t for t in p.extra["trees"] if t.base[1] < 0.6 * 180]
    near = [t for t in p.extra["trees"] if t.base[1] > 0.72 * 180]
    assert far and near
    assert np.mean([t.base[1] - t.top[1] for t in near]) > np.mean([t.base[1] - t.top[1] for t in far])


def test_texture_primitives():
    f1, f2, cid = F.cellular(64, 80, 10, seed=1)
    assert f1.shape == (64, 80) and (f2 >= f1).all() and 0 <= cid.min() and cid.max() <= 1
    assert len(np.unique(np.round(cid, 5))) > 20
    Y, X = np.mgrid[0:64, 0:80].astype(np.float32)
    cov, sid = F.strokes(X, Y, 0.3, length=10, width=2, seed=1)
    assert cov.shape == (64, 80) and 0.1 < cov.mean() < 0.9


def test_region_helpers_and_layers():
    cv = Canvas(240, 160, "#88aa66")
    rec = LayerRecorder(cv)
    rec.mark("grass")
    F.grass(cv, region=(0, 0.7, 1, 1), tufts=20, seed=1)
    rec.mark("flowers")
    F.flower_bed(cv, region=(0.1, 0.6, 0.9, 0.8), n=12, seed=2)
    rec.mark("rocks")
    F.pebbles(cv, region=(0.1, 0.85, 0.9, 0.98), n=30)
    F.stepping_stones(cv, n=4)
    rec.mark("misc")
    F.ivy(cv, path=[(0.1, 0.2), (0.5, 0.1), (0.9, 0.25)])
    F.moss(cv, region=(0, 0.9, 0.5, 1))
    F.falling_leaves(cv, n=8)
    F.cliff(cv, region=(0, 0, 0.3, 1))
    F.branch_frame(cv, "tr", kind="autumn", size=0.4)
    F.wheat(cv, region=(0.5, 0.8, 1, 1), density=0.3)
    F.meadow(cv, region=(0, 0.85, 1, 1), density=0.2)
    F.leaf_pile(cv, at=(0.5, 0.95), width=0.3, height=0.1)
    stack = rec.finish()
    got = [l.name for l in stack if not l.name.endswith("(halo)")]
    assert got[-4:] == ["grass", "flowers", "rocks", "misc"]
    assert np.isfinite(cv.buf).all()


def test_custom_flower_spec_and_clip():
    cv = Canvas(200, 200, "#ffffff")
    clip = np.zeros((200, 200), np.float32)
    clip[:, :100] = 1
    p = F.flower(cv, spec=dict(petals=9, shape="pointed", color="#7a3cc8", center="#f2c234"), at=(0.5, 0.9),
                 height=0.7, clip=clip)
    assert p.mask[:, 100:].max() == 0 and p.mask[:, :100].max() > 0.9
    hx, hy = map(int, p.canopy)
    assert cv.buf[hy, hx - 3:hx].mean(0)[0] > 0.6               # yellow centre
    with pytest.raises(ValueError):
        F.plant(cv, "zzz qqq")


def test_doc_snippets_run():
    path = os.path.join(ROOT, "skills", "pixelcanvas", "references", "flora.md")
    src = open(path, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", src, re.S)
    assert len(blocks) >= 4
    for b in blocks:
        b = b.replace("Canvas(960, 540", "Canvas(240, 135").replace("Canvas(960, 360", "Canvas(240, 90") \
             .replace("Canvas(800, 450", "Canvas(200, 112").replace("Canvas(480, 270", "Canvas(160, 90")
        b = b.replace("> 470", "> 117").replace("> 280", "> 70")
        ns = {}
        exec(compile(b, "flora.md", "exec"), ns)
        assert np.isfinite(ns["cv"].buf).all()


def test_scenes_render_small():
    import importlib
    for name in ("flora_garden", "flora_sheet"):
        mod = importlib.import_module(f"scenes.{name}")
        cv = mod.render(192, 108) if name == "flora_garden" else mod.render(150, 200)
        assert np.isfinite(cv.buf).all() and 0.1 < cv.buf.mean() < 0.95


def test_branch_frame_enters_every_corner_with_leaves():
    # regression: top-corner branches used to be bent up and out of the picture, leaving only a sliver
    for corner in ("tl", "tr", "bl", "br"):
        cv = Canvas(320, 180, "#d8e2ea")
        p = F.branch_frame(cv, corner=corner, size=0.6, kind="autumn", seed=3)
        assert p.mask.sum() > 0.01 * 320 * 180, corner
        ys, xs = np.nonzero(p.mask > 0.5)
        assert xs.max() - xs.min() > 100 and ys.max() - ys.min() > 40, corner


def test_rock_is_a_solid_with_flat_base_and_distinct_faces():
    cv = Canvas(400, 300, "#d8e2ea")
    p = F.rock(cv, at=(0.5, 0.9), size=0.4, seed=2)
    ys, xs = np.nonzero(p.mask > 0.5)
    assert ys.max() <= int(0.9 * 300) + 1                                  # sits on its base line
    assert 0.3 < (ys.max() - ys.min()) / (xs.max() - xs.min()) < 1.2      # a boulder, not a slab or a spike
    reg = cv.buf[ys.min():ys.max(), xs.min():xs.max()][p.mask[ys.min():ys.max(), xs.min():xs.max()] > 0.9]
    lum = reg.mean(axis=1)
    assert lum.max() - lum.min() > 0.25                                    # lit and shadowed faces
    for kind in F.ROCKS:
        assert F.rock(_cv(), kind=kind, seed=1, size=0.4).mask.max() > 0.5


def test_flower_petals_overlap_cleanly():
    # the petal painter: wide petals must not be cut off along sector borders (alpha stays one connected blob)
    for kind in ("poppy", "hibiscus", "cosmos", "rose", "peony", "cornflower", "cherry blossom"):
        cv = Canvas(200, 200, "#e8e2d0")
        p = F.flower(cv, kind, at=(0.5, 0.5), height=0.7, stem=False, seed=2)
        m = p.mask > 0.5
        assert m.sum() > 2000, kind
        ys, xs = np.nonzero(m)
        assert abs(xs.mean() - 100) < 25 and abs(ys.mean() - 100) < 25, kind
    assert F.find("an old oak tree") == "old oak"
    F.plant(_cv(), "old oak", at=(0.5, 0.95), height=0.8, seed=1)


def test_monstera_leaf_has_slits():
    cv = Canvas(200, 200, "#e8e2d0")
    a = F.leaf(cv, "monstera", at=(0.5, 0.5), size=0.8, seed=1, paint=False).patch.a > 0.5
    b = F.leaf(cv, "ovate", at=(0.5, 0.5), size=0.8, seed=1, paint=False).patch.a > 0.5
    runs = lambda m: int((np.diff(m.astype(np.int8), axis=1) == 1).sum())      # leaf segments per row, summed
    assert runs(a) > 2 * runs(b)                                            # slits cut the blade into strips


def test_meadow_flowers_rise_above_the_grass_region():
    cv = Canvas(400, 300, "#6f9a46")
    p = F.meadow(cv, region=(0.0, 0.7, 1.0, 1.0), flowers=("tulip",), grass_height=0.14, density=1.5, seed=2)
    ys = np.nonzero((p.mask > 0.5).any(axis=1))[0]
    assert ys.min() < 0.7 * 300 - 10                                       # heads are not clipped at the region top


def test_pots_cast_a_shadow_by_default():
    cv0, cv1 = Canvas(300, 200, "#e0d8c0"), Canvas(300, 200, "#e0d8c0")
    F.potted(cv0, "cactus", at=(0.5, 0.7), height=0.3, shadow=False)
    F.potted(cv1, "cactus", at=(0.5, 0.7), height=0.3)
    assert cv1.buf.sum() < cv0.buf.sum() - 1.0                              # the ground under the pot is darker


def test_tulip_and_lavender_shapes():
    cv = Canvas(300, 300, "#f1eadb")
    t = F.flower(cv, "tulip", at=(0.5, 0.5), height=0.5, stem=False, seed=1)
    ys, xs = np.nonzero(t.mask > 0.5)
    assert (ys.max() - ys.min()) > 1.1 * (xs.max() - xs.min())             # taller than wide, pointed cup
    top = (xs[ys < ys.min() + 4].max() - xs[ys < ys.min() + 4].min())
    assert top < 0.5 * (xs.max() - xs.min())                               # a tip, not a flat top
    cv = Canvas(300, 300, "#f1eadb")
    s = F.flower(cv, "lavender", at=(0.5, 0.95), height=0.6, seed=1)
    ys, xs = np.nonzero(s.mask > 0.5)
    assert s.mask.sum() > 300 and ys.min() < 0.5 * 300


def test_strict_unknown_option_alias_and_name():
    cv = _cv()
    with pytest.raises(TypeError, match="did you mean 'height'"):
        F.tree(cv, "oak", heigth=0.5)
    with pytest.raises(TypeError, match="did you mean 'height'"):
        F.plant(cv, "oak", heigth=0.5)
    with pytest.raises(TypeError, match="unknown option 'tint'"):
        F.cattails(cv, tint=1)
    F.tree(cv, "oak", height=0.5, seed=1, shadow=True)
    F.flower(cv, "rose", at=(0.5, 0.9), h=0.3)                 # h = height
    F.flower(cv, "rose", at=(0.5, 0.9), height=0.3, colour="#ff0000")
    with pytest.raises(ValueError, match="did you mean 'oak'"):
        F.plant(cv, "oka")
    with pytest.raises(ValueError, match="did you mean 'oak'"):
        F.tree(cv, "oka")
    with pytest.raises(ValueError, match="did you mean 'rose'"):
        F.flower(cv, "rosee")
    with pytest.raises(ValueError, match="'maple'"):
        F.leaf(cv, "mapel")
    with pytest.raises(ValueError, match="'autumn'"):
        F.tree(cv, "oak", season="autum")


# ---- round 1 fixes: perch, climber, palette docs, names and pixel positions

def _cv_f(w=480, h=300):
    return Canvas(w, h, "#e8dcc4")


def test_perch_draws_a_limb_from_a_to_b_with_a_seat_on_top():
    from pixelcanvas.flora import perch
    cv = _cv_f()
    before = np.array(cv.buf)
    p = perch(cv, (0.1, 0.6), (0.9, 0.5), seed=3)
    assert p.mask.sum() > 500
    ys, xs = np.nonzero(p.mask > 0.5)
    assert xs.min() < 0.15 * cv.w and xs.max() > 0.85 * cv.w
    seat = p.extra["seat"]
    sx, sy = seat(0.5)
    assert 0.3 * cv.w < sx < 0.7 * cv.w
    col = np.nonzero(p.mask[:, int(sx)] > 0.5)[0]
    assert abs(col.min() - sy) < 0.04 * cv.h                  # the seat is on top of the limb
    assert len(p.extra["spine"]) == 15
    assert not np.array_equal(before, np.array(cv.buf))


def test_perch_kinds_and_wrong_names():
    from pixelcanvas.flora import perch
    for kind in ("leafy", "autumn", "blossom", "bare"):
        assert perch(_cv_f(), kind=kind, seed=1).mask.sum() > 100
    with pytest.raises(ValueError, match=r"invalid kind 'leafi'.*did you mean 'leafy'"):
        perch(_cv_f(), kind="leafi")
    with pytest.raises(ValueError, match="looks like pixels"):
        perch(_cv_f(), a=(10, 100), b=(300, 80))


def test_a_bird_sits_on_the_perch():
    from pixelcanvas.flora import perch
    from pixelcanvas.creature import Creature
    cv = _cv_f(640, 400)
    p = perch(cv, (0.05, 0.62), (0.95, 0.55), seed=2, thickness=0.02)
    sx, sy = p.extra["seat"](0.5)
    bird = Creature("sparrow", size=0.15)
    bird.pose("perch")
    bird.draw(cv, at=(sx / cv.w, sy / cv.h), shadow=False)
    assert bird.bbox is not None
    assert abs(bird.bbox[3] - sy) < 0.05 * cv.h


@pytest.mark.parametrize("kind", ["bougainvillea", "wisteria", "ivy", "jasmine"])
@pytest.mark.parametrize("drape", ["frame", "top", "wall"])
def test_climber_drapes_a_wall_or_doorway(kind, drape):
    from pixelcanvas.flora import climber
    cv = _cv_f()
    p = climber(cv, kind, box=(0.2, 0.1, 0.8, 0.95), drape=drape, seed=1)
    assert p.mask.sum() > 300
    ys, xs = np.nonzero(p.mask > 0.3)
    assert xs.min() >= 0.15 * cv.w and xs.max() <= 0.85 * cv.w


def test_climber_frame_leaves_the_doorway_open_and_bougainvillea_is_pink():
    from pixelcanvas.flora import climber
    cv = _cv_f(600, 400)
    p = climber(cv, "bougainvillea", box=(0.2, 0.1, 0.8, 0.98), drape="frame", seed=4)
    centre = p.mask[int(0.45 * 400):int(0.9 * 400), int(0.4 * 600):int(0.6 * 600)]
    assert centre.mean() < 0.02
    px = np.array(cv.buf)[p.mask > 0.9]
    assert (px[:, 0] - px[:, 1]).max() > 0.3                  # magenta bracts


def test_climber_along_a_mask_and_options():
    from pixelcanvas.flora import climber
    cv = _cv_f()
    m = np.zeros((cv.h, cv.w), np.float32)
    m[60:200, 80:400] = 1
    p = climber(cv, "wisteria", along=m, seed=2)
    assert p.mask.sum() > 100
    with pytest.raises(ValueError, match=r"invalid kind 'wisteri'.*did you mean 'wisteria'"):
        climber(cv, "wisteri")
    with pytest.raises(ValueError, match=r"invalid drape"):
        climber(cv, "ivy", drape="roof")
    with pytest.raises(ValueError, match="looks like pixels"):
        climber(cv, "ivy", box=(10, 10, 300, 200))


def test_perch_and_climber_record_themselves():
    from pixelcanvas import manifest
    from pixelcanvas.flora import perch, climber
    cv = _cv_f()
    perch(cv, seed=1)
    climber(cv, "bougainvillea", seed=1)
    rows = {r["name"]: r for r in manifest.summary(cv)}
    assert "branch" in rows and "perch" in rows["branch"]["aliases"]
    assert "bougainvillea" in rows and "vine" in rows["bougainvillea"]["aliases"]


def test_cliff_palette_is_a_list_of_colours_dark_to_light():
    from pixelcanvas.flora import cliff
    cv = _cv_f()
    cliff(cv, palette=["#1a1a1a", "#555555", "#aaaaaa", "#eeeeee"], seed=1)
    assert cliff.__doc__ and "dark to light" in cliff.__doc__
    with pytest.raises(ValueError, match=r"unknown rock kind 'sandston'.*did you mean 'sandstone'"):
        cliff(cv, kind="sandston")
    with pytest.raises(ValueError, match=r"invalid edge 'bottom'"):
        cliff(cv, edge="bottom")


def test_wrong_palette_pot_bark_corner_names_get_did_you_mean():
    from pixelcanvas import flora
    cv = _cv_f()
    with pytest.raises(ValueError, match=r"unknown palette 'bsh'"):
        flora.bush(cv, palette="bsh")
    with pytest.raises(ValueError, match=r"unknown palette"):
        flora.palm(cv, palette="x")
    with pytest.raises(ValueError, match=r"unknown palette"):
        flora.hedge(cv, palette="x")
    with pytest.raises(ValueError, match=r"unknown bark 'pinkk'"):
        flora.tree(cv, "oak", bark="pinkk")
    with pytest.raises(ValueError, match=r"invalid corner 'zz'"):
        flora.branch_frame(cv, corner="zz")
    with pytest.raises(ValueError, match=r"invalid kind 'zz'"):
        flora.branch_frame(cv, kind="zz")
    with pytest.raises(ValueError, match=r"unknown pot shape 'roundd'"):
        flora.potted(cv, "monstera", pot="roundd")
    with pytest.raises(ValueError, match=r"unknown vase shape"):
        flora.vase(cv, shape="x")
    assert flora.tree(cv, "oak", bark="birch").mask.sum() > 0


def test_flora_positions_in_pixels_raise():
    from pixelcanvas import flora
    cv = _cv_f()
    with pytest.raises(ValueError, match=r"at=\(300.0, 180.0\) looks like pixels"):
        flora.tree(cv, "oak", at=(300, 180))
    with pytest.raises(ValueError, match="looks like pixels"):
        flora.forest(cv, region=(10, 10, 300, 200))
    with pytest.raises(ValueError, match="looks like pixels"):
        flora.plant(cv, "oak", at=(300, 180))


def _leaf_pixels(cv):
    """Number of leaf-green pixels."""
    r, g, b = cv.buf[..., 0], cv.buf[..., 1], cv.buf[..., 2]
    return int(((g > r + 0.04) & (g > b + 0.04)).sum())


def test_deciduous_species_follow_the_season_and_evergreens_stay_green():
    def draw(sp, season, **kw):
        cv = Canvas(240, 300, "#dfe9f2")
        F.plant(cv, sp, at=(0.5, 0.95), height=0.8, seed=2, season=season, **kw)
        return cv
    for sp in ("birch", "oak", "maple", "cherry", "poplar", "apple"):
        assert _leaf_pixels(draw(sp, "winter")) < 0.05 * _leaf_pixels(draw(sp, "summer")), sp      # bare branches
    for sp in ("pine", "spruce", "olive", "cypress"):
        assert _leaf_pixels(draw(sp, "winter")) > 0.8 * _leaf_pixels(draw(sp, "summer")), sp
    a, s = draw("birch", "autumn"), draw("birch", "summer")
    assert _leaf_pixels(a) < 0.3 * _leaf_pixels(s)                                         # leaves turned colour
    assert a.buf[..., 0].mean() > s.buf[..., 0].mean() + 0.01
    sn, ns = draw("oak", "winter"), draw("oak", "winter", snow=False)
    assert not np.allclose(sn.buf, ns.buf)                                     # winter trees carry a little snow


def test_near_miss_species_names_raise_did_you_mean():
    cv = Canvas(120, 120)
    for call in (lambda: F.tree(cv, "oakk"), lambda: F.plant(cv, "oakk"), lambda: F.tree(cv, "birsh")):
        with pytest.raises(ValueError, match="did you mean"):
            call()
    F.tree(cv, "a big oak tree")                                                # free text still resolves
    F.plant(cv, "oak trees")
