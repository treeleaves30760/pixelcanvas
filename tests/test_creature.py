# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.creature: every preset draws, poses / views / styles work, anchors, fit, groups, docs."""
import re
from pathlib import Path

import numpy as np
import pytest

from pixelcanvas.canvas import Canvas
from pixelcanvas.creature import Creature, creature, flock, school, species, SPECIES, POSES, COATS

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "skills" / "pixelcanvas" / "references" / "creatures.md"

QUADS = ["cat", "dog", "labrador", "shepherd", "corgi", "poodle", "husky", "fox", "wolf", "horse", "zebra", "cow",
         "sheep", "pig", "goat", "deer", "bear", "lion", "tiger", "elephant", "giraffe", "rabbit", "mouse", "squirrel"]
BIRDS = ["sparrow", "robin", "crow", "owl", "eagle", "duck", "swan", "penguin", "parrot", "flamingo", "chicken"]
SEA = ["fish", "goldfish", "clownfish", "shark", "whale", "dolphin", "jellyfish", "octopus", "turtle"]
SMALL = ["butterfly", "bee", "ladybird", "dragonfly", "snail", "frog", "snake", "lizard", "spider"]


def _cv(w=200, h=140):
    return Canvas(w, h, "#e8e2d4")


def test_species_lists():
    assert set(QUADS) <= set(species("quad"))
    assert set(BIRDS) <= set(species("bird"))
    assert set(SEA) <= set(species("sea"))
    assert set(SMALL) <= set(species("small"))
    assert set(SPECIES) == set(species())
    assert "gallop" in POSES and "sit" in POSES
    assert "tabby" in COATS


@pytest.mark.parametrize("name", QUADS + BIRDS + SEA + SMALL)
def test_every_species_draws(name):
    cv = _cv()
    before = cv.buf.copy()
    c = Creature(name, size=0.5)
    at = (0.5, 0.9) if c._grounded_pose() else (0.5, 0.5)
    c.draw(cv, at=at)
    assert c.mask is not None and c.mask.shape == (140, 200)
    assert c.mask.max() > 0.6, name
    assert c.bbox is not None
    x0, y0, x1, y1 = c.bbox
    assert 0 <= x0 < x1 <= 200 and 0 <= y0 < y1 <= 140
    assert np.abs(cv.buf - before).max() > 0.1
    assert np.isfinite(cv.buf).all()


@pytest.mark.parametrize("pose", ["stand", "walk", "trot", "gallop", "sit", "lie", "sleep", "crouch", "jump",
                                  "graze", "sniff", "rear"])
def test_quad_poses(pose):
    for name in ("cat", "horse"):
        cv = _cv()
        c = Creature(name, size=0.4).pose(pose, phase=0.3)
        c.draw(cv, at=(0.5, 0.9))
        assert c.mask.max() > 0.9


def test_unknown_pose_raises():
    with pytest.raises(ValueError):
        Creature("dog").pose("moonwalk").draw(_cv(), at=(0.5, 0.9))


def test_unknown_species_raises():
    with pytest.raises(ValueError):
        Creature("dragon")


def test_aliases_and_coat_names():
    assert Creature("puppy").name == "dog"
    assert Creature("kitten").name == "cat"
    assert Creature("ladybug").name == "ladybird"
    c = Creature("siamese cat")
    assert c.name == "cat" and c.coat()["pattern"] == "points"
    assert Creature("dalmatian").name == "dog"
    assert Creature("cat", coat="calico").coat()["pattern"] == "patches"


@pytest.mark.parametrize("view", ["side", "three-quarter", "front", 25.0])
def test_views(view):
    cv = _cv()
    c = Creature("dog", size=0.5, view=view).pose("sit")
    c.draw(cv, at=(0.5, 0.9))
    assert c.mask.max() > 0.9


@pytest.mark.parametrize("style", ["soft", "flat", "cute"])
def test_styles(style):
    cv = _cv()
    c = Creature("fox", size=0.5, style=style)
    c.draw(cv, at=(0.5, 0.9))
    assert c.mask.max() > 0.9
    with pytest.raises(ValueError):
        Creature("fox", style="neon")


def test_facing_mirrors():
    a, b = _cv(), _cv()
    ca = Creature("horse", size=0.5, facing="right").draw(a, at=(0.5, 0.9))
    cb = Creature("horse", size=0.5, facing="left").draw(b, at=(0.5, 0.9))
    assert ca.anchor("nose")[0] > 100 > cb.anchor("nose")[0]


def test_anchors_and_size():
    cv = _cv(400, 300)
    c = Creature("dog", size=0.4).draw(cv, at=(0.5, 0.9))
    for k in ("mouth", "nose", "eye", "ear", "head", "back", "tail", "paw_front_l", "paw_front_r", "paw_hind_l",
              "paw_hind_r", "center", "top", "ground"):
        x, y = c.anchor(k)
        assert -50 < x < 450 and -50 < y < 350, k
    x0, y0, x1, y1 = c.bbox
    assert abs((y1 - y0) - 0.4 * 300) < 0.15 * 300          # size = standing height
    assert c.anchor("head")[1] < c.anchor("paw_front_l")[1]
    with pytest.raises(KeyError):
        c.anchor("wing")


def test_fit_box():
    cv = _cv(400, 300)
    c = Creature("giraffe").draw(cv, fit=(0.1, 0.1, 0.4, 0.9))
    x0, y0, x1, y1 = c.bbox
    assert x0 >= 0.1 * 400 - 6 and x1 <= 0.4 * 400 + 6 and y0 >= 0.1 * 300 - 6 and y1 <= 0.9 * 300 + 6
    assert (y1 - y0) > 0.6 * 300 or (x1 - x0) > 0.25 * 400


def test_functional_api_and_options():
    cv = _cv(300, 200)
    c = creature(cv, "dog", at=(0.5, 0.9), size=0.5, pose="sit", view="three-quarter", head_turn=20, tail="wag",
                 mouth="open", coat=dict(base="#333333"))
    assert c.mask.max() > 0.9
    c = creature(cv, "owl", fit=(0.0, 0.0, 0.3, 0.6))
    assert c._pose[0] == "perch" or c._pose[0] == "stand"
    c = creature(cv, "horse", at=(0.5, 0.9), size=0.4, neck_len=0.5)
    assert c.mask.max() > 0.9


def test_birds_poses():
    for pose in ("stand", "perch", "fly", "glide", "swim"):
        cv = _cv()
        c = Creature("duck", size=0.4).pose(pose, phase=0.25)
        c.draw(cv, at=(0.5, 0.6))
        assert c.mask.max() > 0.9
    c = Creature("eagle", size=0.4).pose("fly", 0.1).draw(_cv(), at=(0.5, 0.5))
    assert "wing" in c.anchors and "beak" in c.anchors


def test_swim_clips_at_waterline():
    cv = _cv()
    c = Creature("swan", size=0.5).pose("swim").draw(cv, at=(0.5, 0.6))
    ys = np.nonzero(c.mask.max(1) > 0.5)[0]
    assert ys.max() <= 0.6 * 140 + 2


def test_flock_and_school():
    cv = _cv(320, 180)
    birds = flock(cv, "sparrow", n=5, region=(0.1, 0.1, 0.9, 0.5), size=0.06, seed=2)
    assert len(birds) == 5 and all(b.mask.max() > 0.5 for b in birds)
    fish = school(cv, "fish", n=6, region=(0.1, 0.5, 0.9, 0.95), size=0.08, seed=3)
    assert len(fish) == 6


def test_log_is_one_step():
    steps = []
    Creature("cat", size=0.4).draw(_cv(), at=(0.5, 0.9), log=steps.append)
    assert steps == ["cat"]


def test_resolution_independent():
    small, big = Canvas(160, 90), Canvas(640, 360)
    a = Creature("fox", size=0.5).draw(small, at=(0.5, 0.9))
    b = Creature("fox", size=0.5).draw(big, at=(0.5, 0.9))
    ha = (a.bbox[3] - a.bbox[1]) / 90
    hb = (b.bbox[3] - b.bbox[1]) / 360
    assert abs(ha - hb) < 0.06


def test_form_spot_and_shadow():
    from pixelcanvas.form import Ground
    cv = Canvas(320, 180, "#9ab07a")
    g = Ground(cv, horizon=0.4)
    c = Creature("cow", size=1.4).draw(cv, at=g.place(0.0, 6.0), cast=True)
    near = Creature("cow", size=1.4).draw(Canvas(320, 180), at=g.place(0.0, 3.0))
    assert (near.bbox[3] - near.bbox[1]) > (c.bbox[3] - c.bbox[1])


def _doc_blocks():
    if not DOC.exists():
        return []
    return re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)


@pytest.mark.parametrize("i", range(len(_doc_blocks())))
def test_doc_snippets_run(i):
    code = _doc_blocks()[i]
    ns = {}
    exec(compile(code, f"creatures.md[{i}]", "exec"), ns)


# ---- regressions from the review pass


def test_golden_retriever_is_its_own_breed():
    c = Creature("golden retriever")
    assert c.name == "golden"
    assert c.coat()["base"] != Creature("labrador").coat()["base"]
    cv = _cv()
    c.pose("lie").draw(cv, at=(0.5, 0.9))
    assert c.mask.max() > 0.9


def test_blend_seams_have_no_holes():
    # wide blends used to lift the surface above every volume: the colour weights vanished and left dark
    # specks / transparent holes in the seams (horse withers in three-quarter view)
    cv = Canvas(450, 350, "#e8e2d4")
    for name, view in (("horse", "three-quarter"), ("labrador", 60), ("cat", "side")):
        c = Creature(name, view=view).pose("stand")
        at = c.fit(cv, (0.02, 0.02, 0.98, 0.98))
        _m, _cam, rr = c.render(cv, at)
        inside = rr.D < -3
        assert rr.alpha[inside].min() > 0.98, name
        lum = rr.rgb[inside].mean(-1)
        assert np.percentile(lum, 0.5) > 0.03, name


def test_sitting_dog_shows_its_forelegs():
    from pixelcanvas.creature import _qparams, _quad_pose
    for name in ("labrador", "lion", "cat"):
        sk = _quad_pose(_qparams(name), "sit")
        assert sk["sp"] <= 56.5, name
        # the forepaws stand in front of the shoulder, not hidden under the chest
        assert sk["legs"]["fl"]["contact"][0] > sk["S"][0] + 0.02, name


def test_fur_tips_are_not_dark():
    cv = Canvas(500, 300, "#ffffff")
    c = Creature("golden", view="side").pose("stand")
    at = c.fit(cv, (0.05, 0.05, 0.95, 0.95))
    _m, _cam, rr = c.render(cv, at)
    lum = rr.rgb.mean(-1)
    body = lum[(rr.D < -8) & (rr.alpha > 0.99)].mean()
    tips = lum[(rr.D > 0.5) & (rr.alpha > 0.5)]
    assert tips.size > 20
    assert tips.mean() > 0.6 * body


def test_cat_slit_pupils_stay_upright_in_front_view():
    from pixelcanvas.creature import _decal_frame, _Decal, _Camera
    cam = _Camera(np.array([100.0, 100.0]), 100.0, yaw=90.0, pitch=6.0)

    class _RR:
        pass
    rr = _RR()
    rr.cam = cam
    dc = _Decal("eye", np.array([0.1, 0.5, 0.05]), n=np.array([0.78, 0.3, 0.55]) / np.linalg.norm([0.78, 0.3, 0.55]),
                fwd=np.array([1.0, 0.0, 0.0]), size=0.03)
    c, Mi, facing = _decal_frame(rr, dc)
    M = np.linalg.inv(Mi)
    up_screen = M[:, 1]                   # screen direction of the decal's "up"
    assert abs(up_screen[0]) < 0.35 * abs(up_screen[1])


# ---- strict options, aliases, registry names, manifest

def test_creature_unknown_option_names_the_closest():
    cv = Canvas(200, 150)
    with pytest.raises(TypeError, match=r"unknown option 'neck_lenn'.*did you mean 'neck_len'"):
        creature(cv, "horse", neck_lenn=0.5)
    with pytest.raises(TypeError, match=r"Creature.pose: unknown option 'head_turnn'.*did you mean 'head_turn'"):
        Creature("cat").pose("sit", head_turnn=10)
    with pytest.raises(TypeError, match=r"unknown option 'shadw'.*did you mean 'shadow'"):
        Creature("cat").draw(cv, (0.5, 0.8), shadw=True)


def test_creature_aliases_are_accepted():
    cv = Canvas(200, 150)
    c = Creature("cat", facing=-1)                                                          # -1 -> "left"
    assert c.facing == "left"
    c.draw(cv, pos=(0.5, 0.8))                                                              # pos -> at
    assert c.bbox is not None


def test_creature_unknown_registry_names():
    with pytest.raises(ValueError, match=r"unknown species 'hors'.*did you mean 'horse'"):
        Creature("hors")
    with pytest.raises(ValueError, match=r"unknown pose 'siting'.*did you mean 'sitting'|unknown pose 'siting'.*did you mean 'sit'"):
        Creature("cat").pose("siting")
    with pytest.raises(ValueError, match=r"unknown style 'sof'.*did you mean 'soft'"):
        Creature("cat", style="sof")
    with pytest.raises(ValueError, match=r"unknown view 'sid'.*did you mean 'side'"):
        Creature("cat", view="sid")


def test_each_animal_is_one_manifest_entry_named_by_species():
    from pixelcanvas import manifest
    cv = Canvas(480, 270)
    creature(cv, "fox", at=(0.3, 0.9), size=0.3)
    flock(cv, "crow", n=3, seed=2)
    rows = manifest.summary(cv)
    assert [r["name"] for r in rows if r["kind"] == "creature"][:2] == ["fox", "crow"]
    assert len([r for r in rows if r["kind"] == "creature"]) == 4
    assert "animal" in rows[0]["aliases"] and rows[0]["meta"]["pose"] == "stand"


# ---- round 1 fixes: coat colours, requested species name, positions in pixels

def test_coat_accepts_a_colour_a_name_or_a_dict():
    cv = _cv()
    for coat in ("#d8883a", (0.85, 0.5, 0.2), "ginger", dict(base="#d8883a", belly="#f0d8b0")):
        c = creature(cv, "cat", coat=coat, at=(0.5, 0.9), size=0.5)
        assert c.mask.max() > 0.9
    for sp, col in (("sparrow", "#3a7bd5"), ("dog", "#d8883a"), ("fish", "#d8883a")):
        creature(cv, sp, coat=col, at=(0.5, 0.9), size=0.4)
    cat = Creature("cat", coat="#3a7bd5")
    assert cat.coat()["base"] != Creature("cat").coat()["base"]
    assert Creature("cat", coat="tabby").coat()["pattern"] == "tabby"


def test_coat_colour_recolours_the_main_colour_and_keeps_markings():
    c = Creature("panda", coat="#3a7bd5").coat()
    assert c["socks"][1] == "#1a1818"                      # the black socks stay
    sp = Creature("sparrow", coat="#3a7bd5").coat()
    assert sp["throat"] == "#2a2420"                       # the dark bib stays
    assert sp["back"] != "#8a5a34"


def test_wrong_coat_gets_a_did_you_mean_and_the_list():
    cv = _cv()
    with pytest.raises(ValueError, match=r"unknown coat 'gingr'.*did you mean 'ginger'.*Coat names: .*tabby"):
        creature(cv, "cat", coat="gingr")
    with pytest.raises(ValueError, match=r"unknown coat"):
        creature(cv, "cat", coat="not a colour")
    with pytest.raises(ValueError, match=r"unknown coat"):
        creature(cv, "cat", coat=5)
    with pytest.raises(TypeError, match=r"unknown option 'bse'.*did you mean 'base'"):
        creature(cv, "cat", coat=dict(bse="#fff"))


def test_the_requested_species_word_is_what_the_report_records():
    from pixelcanvas import manifest
    cv = _cv(240, 160)
    creature(cv, "panda", at=(0.3, 0.9), size=0.5)
    creature(cv, "kitten", at=(0.7, 0.9), size=0.5)
    rows = manifest.summary(cv)
    names = [r["name"] for r in rows if r["kind"] == "creature"]
    assert names == ["panda", "kitten"]
    assert "bear" in rows[0]["aliases"] and "animal" in rows[0]["aliases"] and "cat" in rows[1]["aliases"]


def test_positions_in_pixels_raise_a_value_error_that_says_so():
    cv = _cv()
    with pytest.raises(ValueError, match=r"at=\(100.0, 120.0\) looks like pixels.*fractions of the canvas"):
        creature(cv, "dog", at=(100, 120))
    with pytest.raises(ValueError, match=r"looks like pixels"):
        creature(cv, "dog", fit=(10, 10, 100, 100))
    with pytest.raises(ValueError, match=r"looks like pixels"):
        flock(cv, "sparrow", region=(10, 10, 100, 100))


def test_unknown_anchor_is_a_value_error_with_a_did_you_mean():
    cv = _cv()
    c = creature(cv, "dog")
    with pytest.raises(ValueError, match=r"unknown anchor 'nsoe'.*did you mean 'nose'"):
        c.anchor("nsoe")
    with pytest.raises(KeyError):
        c.anchor("nsoe")
