# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.clay (3D clay characters, animals and props). Geometry checks plus tiny renders."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import clay as C  # noqa: E402
from pixelcanvas.solid import Scene, Camera, Model, smooth_union as solid_smooth_union  # noqa: E402

RNG = np.random.default_rng(0)


def lowest(model, n=32, levels=80):
    """Height of the lowest surface point (the SDF bounds are conservative, so probe the field itself)."""
    lo, hi = model.bounds()
    X, Z = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[2], hi[2], n))
    for y in np.linspace(lo[1], lo[1] + 0.3 * (hi[1] - lo[1]), levels):
        P = np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], -1).astype(np.float32)
        if (np.min([p.shape(P) for p in model.parts], 0) < 0.004).any():
            return float(y)
    return float(lo[1] + 0.3 * (hi[1] - lo[1]))


# ------------------------------------------------------------------ registry

def test_registry_covers_the_brief():
    for n in ["kid", "student", "office_worker", "chef", "doctor", "nurse", "scientist", "farmer", "firefighter",
              "police_officer", "astronaut", "artist", "musician", "athlete", "runner", "hiker", "grandma", "grandpa",
              "baby", "wizard", "knight", "pirate", "princess", "superhero", "robot_friend", "santa"]:
        assert n in C.CAST, n
    for n in ["short", "crop", "side_part", "quiff", "bob", "long", "ponytail", "pigtails", "bun", "top_knot",
              "curly", "afro", "braids", "mohawk", "bald", "hijab"]:
        assert n in C.HAIR_STYLES, n
    for n in ["stand", "contrapposto", "walk", "run", "jump", "sit", "sit_floor", "kneel", "lie", "wave", "point",
              "cheer", "clap", "think", "arms_crossed", "hands_on_hips", "hold", "carry", "hug", "dance", "kick",
              "throw", "reach_up", "look_up", "sleep", "phone", "typing", "reading"]:
        assert n in C.POSES, n
    for n in ["neutral", "smile", "laugh", "surprised", "sad", "angry", "wink", "sleepy", "determined", "shy"]:
        assert n in C.EXPRESSIONS, n
    for n in ["cat", "labrador", "corgi", "poodle", "husky", "dachshund", "shiba", "fox", "wolf", "rabbit", "mouse",
              "squirrel", "pig", "sheep", "goat", "cow", "horse", "pony", "zebra", "deer", "giraffe", "elephant",
              "rhino", "hippo", "lion", "tiger", "bear", "polar_bear", "panda", "koala", "monkey", "kangaroo",
              "chick", "chicken", "duck", "penguin", "owl", "parrot", "flamingo", "eagle", "sparrow", "fish", "shark",
              "whale", "dolphin", "turtle", "octopus", "crab", "seal", "frog", "snail", "ladybird", "bee",
              "butterfly", "caterpillar", "t_rex", "triceratops", "brontosaurus", "stegosaurus", "dragon", "unicorn"]:
        assert n in C.SPECIES, n
    for n in ["balloon", "book", "cup", "phone", "umbrella", "guitar", "ball", "flower", "bouquet", "ice_cream",
              "sign", "flag", "lantern", "fishing_rod", "skateboard", "bicycle", "picnic_blanket", "basket",
              "campfire", "bench", "stool"]:
        assert n in C.PROPS, n
    for n in ["stand", "walk", "trot", "gallop", "sit", "lie", "sleep", "jump", "beg", "play_bow", "graze", "rear"]:
        assert n in C.ANIMAL_POSES, n


def test_names_find_describe():
    assert "chef" in C.names("character") and "corgi" in C.names("animal") and "balloon" in C.names("prop")
    assert C.find("a little corgi") == "corgi"
    assert C.find("space man") == "astronaut"
    assert C.find("red balloon") == "balloon"
    for n in ("chef", "corgi", "balloon"):
        assert n in C.describe(n)
    with pytest.raises(Exception):
        C.names("vehicle")


def test_unknown_names_raise_with_suggestion():
    with pytest.raises(Exception) as e:
        C.cast("chefф")
    assert "chef" in str(e.value)
    with pytest.raises(Exception):
        C.animal("corgy")
    with pytest.raises(Exception):
        C.prop("baloon")
    with pytest.raises(Exception):
        C.cast("kid").pose("moonwalk")


# ------------------------------------------------------------------ soft union matches solid's field

def test_soft_union_same_field_as_solid():
    shapes = [C.round_cone((i * 0.1, 0, 0), (i * 0.1, 0.3, 0.05), 0.05, 0.03) for i in range(12)]
    P = RNG.uniform(-0.5, 1.5, (4000, 3)).astype(np.float32)
    a = C.smooth_union(*shapes, k=0.05)(P)
    b = solid_smooth_union(*shapes, k=0.05)(P)
    assert np.abs(a - b).max() < 1e-5


def test_round_cone_distance():
    rc = C.round_cone((0, 0, 0), (0, 1, 0), 0.2, 0.1)
    assert rc([0.3, 0, 0]) == pytest.approx(0.1, abs=1e-3)
    assert rc([0, 1.3, 0]) == pytest.approx(0.2, abs=1e-3)
    assert rc([0, 0.5, 0]) < 0


# ------------------------------------------------------------------ characters

@pytest.mark.parametrize("build", list(C.BUILDS))
def test_builds_stand_on_the_floor(build):
    ch = C.Character(build=build)
    assert abs(lowest(ch)) < 0.02 * ch._rig.height
    assert isinstance(ch, Model) and {"skin", "mouth", "irises"} <= {p.name for p in ch.parts}


def test_head_heights_by_build():
    ratios = {}
    for b in ("chibi", "toy", "kid", "adult", "hero"):
        ch = C.Character(build=b)
        ratios[b] = ch._rig.height / ch._rig.H
    assert ratios["chibi"] < ratios["toy"] < ratios["kid"] < ratios["adult"] < ratios["hero"]
    assert ratios["toy"] == pytest.approx(3, abs=0.4) and ratios["hero"] == pytest.approx(6.5, abs=0.5)


@pytest.mark.parametrize("pose", ["stand", "walk", "run", "wave", "kneel", "dance", "kick", "throw", "cheer",
                                  "arms_crossed", "hands_on_hips", "think", "phone"])
def test_feet_planted_after_posing(pose):
    ch = C.cast("kid", pose=pose, phase=0.3)
    assert abs(lowest(ch)) < 0.03 * ch._rig.height, pose


def test_every_pose_builds():
    ch = C.Character(build="adult")
    for p in C.POSES:
        ch.pose(p, phase=0.4)
        assert len(ch.parts) > 5


def test_pose_blend_and_mirror():
    a = C.cast("kid", pose="stand")
    b = C.cast("kid", pose="cheer")
    m = C.cast("kid", pose="stand").blend("cheer", 0.5)
    ya, yb, ym = (x.anchor("hand_r")[1] for x in (a, b, m))
    assert min(ya, yb) < ym < max(ya, yb)
    w = C.cast("kid", pose="wave")
    wm = C.cast("kid").pose("wave", mirror=True)
    assert w.anchor("hand_r")[1] > w.anchor("hand_l")[1]
    assert wm.anchor("hand_l")[1] > wm.anchor("hand_r")[1]


def test_joint_overrides_and_limits():
    a = C.cast("kid").pose("stand", elbow_r=dict(bend=90))
    b = C.cast("kid").pose("stand", elbow_r=dict(bend=400))            # clamped to the joint limit
    assert np.linalg.norm(a.anchor("hand_r") - C.cast("kid").anchor("hand_r")) > 0.02
    assert np.all(np.isfinite(b.anchor("hand_r")))


def test_look_at_and_face():
    ch = C.cast("kid")
    e0 = ch.anchor("nose").copy()
    ch.look_at((3.0, 1.0, 1.0))
    assert ch.anchor("nose")[0] > e0[0]
    ch2 = C.cast("kid").face(90)
    assert ch2.anchor("nose")[0] > 0.05


def test_hold_and_anchors():
    ch = C.cast("kid", pose="hold")
    ch.hold(C.prop("balloon"), hand="r")
    names = {p.name for p in ch.parts}
    assert any("balloon" in n for n in names)
    for a in ("hand_l", "hand_r", "head_top", "mouth", "eye_l", "feet", "lap", "back", "chest"):
        assert np.asarray(ch.anchor(a)).shape == (3,)
    assert ch.anchor("head_top")[1] > ch.anchor("mouth")[1] > ch.anchor("feet")[1]


def test_sit_on_bench_matches_seat_height():
    bench = C.prop("bench")
    ch = C.cast("grandpa").sit_on(bench)
    seat_top = bench["seat"].shape.bounds()[1][1]
    assert ch.anchor("seat")[1] == pytest.approx(seat_top, abs=0.05)
    assert ch.bounds()[0][1] > 0.0                       # off the floor, sitting


def test_outfits_hair_expressions_materials():
    for st in C.HAIR_STYLES:
        C.Character(build="kid", hair=dict(style=st, color="auburn"))
    for ex in C.EXPRESSIONS:
        C.Character(face=dict(expression=ex))
    for o in C.OUTFITS:
        C.Character(build="adult", outfit=o)
    for mat in ("clay", "plastic", "plush", "wood", "vinyl"):
        C.Character(material=mat)
    ch = C.Character(outfit=dict(top="hoodie", bottom="overalls", shoes="boots", hat="beanie",
                                 extras=["backpack", "scarf", "glasses", "headphones"]))
    names = {p.name for p in ch.parts}
    assert {"hat", "backpack", "scarf", "glasses", "headphones"} <= names


def test_cast_and_random_people():
    for n in C.CAST:
        assert len(C.cast(n, detail="low").parts) > 5
    a, b = C.random_person(1), C.random_person(1)
    c = C.random_person(2)
    cols = [[str(p.mat.color) for p in x.parts] for x in (a, b, c)]
    assert cols[0] == cols[1] and cols[0] != cols[2]


def test_recolor():
    ch = C.cast("kid").recolor(top="#ff0000")
    top = [p for p in ch.parts if p.name == "top"][0]
    assert str(top.mat.color).lower() in ("#ff0000", "(1.0, 0.0, 0.0)")


# ------------------------------------------------------------------ animals and props

def test_every_species_builds():
    for sp in C.SPECIES:
        a = C.animal(sp, detail="low")
        lo, hi = a.bounds()
        assert len(a.parts) >= 1 and np.all(np.isfinite(lo)) and hi[1] > 0.02, sp
    for sp in ("cat", "horse", "giraffe", "elephant", "koala", "penguin", "t_rex", "frog"):
        a = C.animal(sp, detail="low")
        assert abs(lowest(a, n=20, levels=40)) < 0.03 * a.bounds()[1][1] + 0.01, sp


@pytest.mark.parametrize("pose", ["stand", "walk", "trot", "gallop", "sit", "lie", "sleep", "beg", "play_bow",
                                  "graze", "rear"])
def test_quadruped_poses(pose):
    a = C.animal("corgi", pose=pose, phase=0.3)
    assert abs(lowest(a)) < 0.03


def test_animal_size_coat_and_anchor():
    a = C.animal("cat", coat="calico", size=0.5)
    assert a.bounds()[1][1] == pytest.approx(0.5, rel=0.25)
    assert np.asarray(a.anchor("head")).shape == (3,)
    bird = C.animal("parrot", pose="fly", phase=0.25)
    assert lowest(bird) > 0.05


def test_every_prop_builds():
    for p in C.PROPS:
        m = C.prop(p)
        assert isinstance(m, Model) and m.parts
    assert C.prop("sign", text="OPEN").parts


# ------------------------------------------------------------------ render, masks, decals

def test_render_masks_and_print_on():
    sc = Scene(lighting="studio", ground="#e9e2d6")
    kid = C.cast("kid", pose="wave").place(x=-0.3)
    sc.add(kid, "kid")
    sc.add(C.animal("corgi", pose="sit").place(x=0.5), "dog")
    sc.camera = Camera.frame(sc, azimuth=0, elevation=6)
    cv = Canvas(160, 90)
    res = sc.render(cv, aa=1)
    assert res.mask("kid").sum() > 20 and res.mask("kid.top").sum() > 3 and res.mask("dog").sum() > 5
    before = cv.buf.copy()
    C.print_on(cv, res, kid, "7", where="chest", color="#ff00ff")
    after = cv.buf
    assert np.abs(after - before).sum() > 0
    fr = C.decal_frame(res, kid, "chest")
    assert fr["visible"] and 0 <= fr["center"][0] < 160


# ------------------------------------------------------------------ reference page

def test_reference_page_blocks_run():
    doc = os.path.join(ROOT, "skills", "pixelcanvas", "references", "clay.md")
    src = open(doc, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", src, re.S)
    assert blocks
    assert len(src.splitlines()) <= 250
    for code in blocks:
        code = re.sub(r"Canvas\(\d+, \d+\)", "Canvas(96, 54)", code).replace('cv.save("clay.png")', "")
        exec(compile(code, "clay.md", "exec"), {"__name__": "__doc__"})


# ------------------------------------------------------------------ review fixes (art-direction round)

def test_outfit_pieces_are_checked_and_everyday_words_work():
    with pytest.raises(ValueError) as e:
        C.Character(outfit=dict(top="raincot"))
    assert "raincoat" in str(e.value)
    with pytest.raises(ValueError):
        C.Character(outfit=dict(shoes="flipflopz"))
    with pytest.raises(ValueError):
        C.Character(outfit=dict(hat="tophat"))
    with pytest.raises(ValueError):
        C.Character(outfit=dict(extras=["umbrella"]))
    with pytest.raises(ValueError):
        C.Character(hair=dict(style="dreadlocks"))
    ch = C.Character(outfit=dict(top="jumper", bottom="pants", shoes="wellies"))
    assert (ch.outfit["top"], ch.outfit["bottom"], ch.outfit["shoes"]) == ("sweater", "trousers", "rain_boots")
    assert "raincoat" in C.TOPS
    for n in C.CAST:                                    # every preset still validates
        C.cast(n)


def test_sit_on_finds_the_cushions_not_the_backrest():
    from pixelcanvas.solid import sofa
    s = sofa()
    from pixelcanvas.clay.character import _find_seat
    f = _find_seat(s)
    assert 0.42 < f["height"] < 0.56                  # the seat cushions, not the 0.87 m back
    assert abs(f["facing"]) < 8
    turned = _find_seat(sofa().rotate_y(90).place(x=2.0))
    assert abs(turned["facing"] - 90) < 12
    assert 1.6 < turned["centre"][0] < 2.4
    kid = C.Character("k", build="kid").pose("reading").sit_on(s)
    seat = kid.anchor("seat")
    assert abs(seat[1] - f["height"]) < 0.06
    # knees reach the front edge, shins hang in front of the seat instead of through it
    assert kid.anchor("knee_l")[2] > 0.25
    adult = C.Character("a", build="adult").sit_on(sofa().rotate_y(90))
    assert adult.anchor("knee_l")[0] > adult.anchor("seat")[0]          # facing away from the backrest


def test_poses_bring_their_props_and_cast_frees_the_gesturing_hand():
    names = [p.name for p in C.Character("a", build="adult").pose("phone").parts]
    assert "phone.case" in names
    names = [p.name for p in C.Character("a", build="kid").pose("reading").parts]
    assert "book.cover" in names
    names = [p.name for p in C.cast("office_worker").parts]
    assert names.count("phone.case") == 1
    t = C.cast("teacher", pose="point")
    assert t._held[0][1] == "l"
    assert C.cast("teacher")._held[0][1] == "r"
    ch = C.Character("b", build="kid").pose("hold").hold("book", hand="both")
    assert any(p.name == "book.cover" for p in ch.parts)
    with pytest.raises(ValueError):
        C.Character().hold("book", hand="left")


def test_sunglasses_hide_the_eyes():
    names = [p.name for p in C.cast("musician").parts]
    assert "lenses" in names and "irises" not in names and "eye shine" not in names
    names = [p.name for p in C.cast("doctor").parts]
    assert "irises" in names


@pytest.mark.parametrize("species,pose", [("cat", "sit"), ("cat", "beg"), ("fox", "lie"), ("labrador", "lie"),
                                          ("elephant", "lie"), ("squirrel", "beg")])
def test_resting_animals_touch_the_floor_without_sinking(species, pose):
    a = C.animal(species, pose=pose)
    y = lowest(a, n=40, levels=120)
    assert -0.04 * a.height - 0.004 < y < 0.03 * a.height, (species, pose, y)   # lowest() probes within 4 mm


def test_lying_dog_is_low_and_long_and_sitting_cat_is_upright():
    stand, lie = C.animal("labrador"), C.animal("labrador", pose="lie")
    hs, hl = stand.bounds()[1][1], lie.bounds()[1][1]
    assert hl < hs * 0.9
    sit = C.animal("cat", pose="sit")
    head, feet = sit.anchor("head"), sit.anchor("paw_fl")
    assert head[1] > feet[1] + 0.6 * sit.height          # chest up, head high over the front paws


QUADS = sorted(k for k, v in C.SPECIES.items() if v.get("family", "quad") == "quad" and k != "tyrannosaurus")


@pytest.mark.parametrize("species", QUADS)
def test_sleep_curls_every_quadruped_up_on_the_ground(species):
    stand, sleep = C.animal(species, detail="low"), C.animal(species, pose="sleep", detail="low")
    h = stand.bounds()[1][1]
    # lying down: the back well below its standing height, the head down near the floor, eyes shut
    assert sleep.anchor("back")[1] < 0.65 * stand.anchor("back")[1], species
    assert sleep.anchor("head")[1] < 0.4 * h, species
    names = {p.name for p in sleep.parts}
    assert "closed eyes" in names and "eyes" not in names, species
    # on its side: the upper legs rest on the lower ones instead of four paws planted on the floor
    up = [abs(sleep.anchor(f"paw_{e}l")[1] - sleep.anchor(f"paw_{e}r")[1]) for e in "fh"]
    assert min(up) > 0.05 * h, (species, up)
    # curled: the nose comes round towards the tail (seen from above)
    tail = C.SPECIES[species].get("tail") or {}
    if tail.get("type") not in ("none", "stub", "puff") and tail.get("len", 0.55) >= 0.3:
        def span(a):
            d = np.asarray(a.anchor("nose")) - np.asarray(a.anchor("tail_tip"))
            return float(np.hypot(d[0], d[2]))
        assert span(sleep) < 0.8 * span(stand), (species, span(sleep), span(stand))


def _head_size(a):
    return float(a.anchor("head_top")[1] - a.anchor("head")[1]) / a.bounds()[1][1]


@pytest.mark.parametrize("species,pose", [("cat", "sit"), ("corgi", "stand"), ("panda", "sit")])
def test_cute_adds_blush_and_a_smile(species, pose):
    plain, cute = C.animal(species, pose=pose), C.animal(species, pose=pose, cute=1.6)
    names = {p.name for p in cute.parts}
    assert {"blush", "smile"} <= names
    assert not {"blush", "smile"} & {p.name for p in plain.parts}
    # the smile sits on the face, under the nose
    smile = [p for p in cute.parts if p.name == "smile"][0]
    lo, hi = smile.shape.bounds()
    nose = cute.anchor("nose")
    assert hi[1] < nose[1] + 0.01 and np.linalg.norm((lo + hi) / 2 - nose) < 0.25 * cute.height


@pytest.mark.parametrize("species", ["chick", "owl", "penguin", "parrot", "duck"])
def test_cute_makes_birds_bigger_headed_with_blush(species):
    plain, cute = C.animal(species), C.animal(species, cute=1.6)
    assert "blush" in {p.name for p in cute.parts} and "blush" not in {p.name for p in plain.parts}
    assert _head_size(cute) > 1.15 * _head_size(plain)
    eyes = lambda a: [p for p in a.parts if p.name == "eyes"][0].shape.bounds()     # noqa: E731
    size = lambda a: float((eyes(a)[1] - eyes(a)[0])[1]) / a.bounds()[1][1]        # noqa: E731
    assert size(cute) > 1.15 * size(plain)


def test_round_cartoon_whale():
    long_, round_ = C.animal("whale", pose="swim"), C.animal("whale", pose="swim", shape="round")
    names = {p.name for p in round_.parts}
    assert {"body", "fins", "eyes", "smile", "blowhole"} <= names
    ext = lambda a: a.bounds()[1] - a.bounds()[0]      # noqa: E731
    assert ext(round_)[2] / ext(round_)[1] < 0.75 * ext(long_)[2] / ext(long_)[1]      # stubby, not a long log
    assert "blush" in {p.name for p in C.animal("whale", shape="round", cute=1.6).parts}
    with pytest.raises(ValueError, match="round"):
        C.animal("whale", shape="rund").parts


def test_campfire_has_layered_glowing_flames_that_light_the_scene():
    fire = C.prop("campfire")
    parts = {p.name: p for p in fire.parts}
    assert {"logs", "embers", "stones", "flame", "flame core"} <= set(parts)
    for nm in ("flame", "flame core", "embers"):
        assert parts[nm].mat.emissive > 1.0 and parts[nm].mat.shadow in (False, 0, 0.0)
    # the colour runs from a pale heart at the base to red-orange tips
    col = parts["flame"].mat.color
    base, tip = col(np.array([[0, 0.08, 0]], np.float32))[0], col(np.array([[0, 0.4, 0]], np.float32))[0]
    assert base[2] > tip[2] * 3 and base[1] > tip[1] * 2
    # several separate tongues: the flame silhouette is ragged, not one cone
    lo, hi = parts["flame"].shape.bounds()
    assert hi[1] - lo[1] > 0.25
    sc = Scene(lighting="night", ground="#334")
    f = sc.add(fire, "fire")
    sc.light_from(f)
    pos = sc.lighting.points[-1]["pos"]
    assert pos[1] > 0.18                                    # the light sits in the flames, above the logs


def test_short_muzzle_keeps_its_nose_on_the_face():
    cat = C.animal("cat")
    names = [p.name for p in cat.parts]
    assert "nose" in names
    nose = [p for p in cat.parts if p.name == "nose"][0]
    fur = [p for p in cat.parts if p.name == "fur"][0]
    c = np.asarray(cat.anchor("nose"), np.float32)[None]
    assert fur.shape(c)[0] > -0.02 * cat.height                   # the nose sits on (not deep inside) the head
    assert nose.shape(c)[0] < 0.01 * cat.height


# ------------------------------------------------------------------------------------------ strict options, manifest
def test_unknown_option_names_the_closest():
    with pytest.raises(TypeError, match="did you mean 'hair'"):
        C.cast("chef", hairr="x")
    with pytest.raises(TypeError, match="did you mean 'expression'"):
        C.cast("chef", face=dict(expresion="happy"))
    with pytest.raises(TypeError, match="did you mean 'color'"):
        C.prop("balloon", colr="#fff")
    with pytest.raises(TypeError, match="Options"):
        C.animal("dog", poze="sit")


def test_alias_spellings_are_accepted():
    assert C.prop("balloon", colour="#3a8ae8").parts
    assert C.animal("dog", colour="#aa8844").parts
    assert C.cast("chef", face=dict(eye_colour="blue")).parts
    assert C.animal("Polar Bear").species == "polar_bear"


@pytest.mark.parametrize("call, near", [
    (lambda: C.animal("corg"), "corgi"), (lambda: C.cast("chf"), "chef"), (lambda: C.cast("chef", pose="wav"), "wave"),
    (lambda: C.animal("dog", pose="sitt"), "sit"), (lambda: C.cast("chef", hair=dict(style="mohawkk")), "mohawk"),
    (lambda: C.cast("chef", face=dict(expression="happpy")), "happy"),
    (lambda: C.cast("chef", outfit=dict(top="tshirtt")), "tshirt"), (lambda: C.prop("ballon"), "balloon"),
    (lambda: C.cast("chef", material="plasticc"), "plastic"), (lambda: C.cast("chef", build="tiny"), "toy"),
    (lambda: C.animal("cat", coat="calico2"), "calico"), (lambda: C.animal("dog", material="clayy"), "clay")])
def test_unknown_registry_names_raise_value_error(call, near):
    with pytest.raises(ValueError, match=near):
        call()


def test_module_getattr_suggests_the_call():
    with pytest.raises(AttributeError, match=r"clay\.animal"):
        C.dog
    with pytest.raises(AttributeError, match=r"clay\.cast"):
        C.Person


def test_scene_manifest_labels_people_and_animals():
    from pixelcanvas import Canvas
    from pixelcanvas.solid import Scene, Camera
    sc = Scene(lighting="studio", ground="#e9e2d6")
    sc.add(C.cast("chef").place(x=-1.0), "Ana")
    sc.add(C.cast("kid").place(x=0.0), "kid")
    sc.add(C.animal("corgi").place(x=1.0), "rex")
    sc.add(C.prop("balloon").place(x=2.0), "balloon")
    sc.camera = Camera.frame(sc, azimuth=10, elevation=8)
    cv = Canvas(320, 180)
    sc.render(cv)
    rows = {e["name"]: e for e in cv.manifest}
    assert rows["Ana"]["kind"] == "person" and {"chef", "person", "people"} <= set(rows["Ana"]["aliases"])
    assert rows["kid"]["kind"] == "person"
    assert rows["rex"]["kind"] == "creature" and {"corgi", "dog", "animal"} <= set(rows["rex"]["aliases"])
    assert rows["balloon"]["kind"] == "object"


# ---- resting clay figures on solid furniture, facing convention, default sizes

def test_clay_animal_rests_on_an_armrest_and_faces_by_angle():
    from pixelcanvas import solid as S
    from pixelcanvas.clay import animal as _animal
    chair = S.armchair().rotate_y(-35).place(x=0.1, z=0.4)
    cat = _animal("cat", pose="sit").face(-35).on(chair, "armrest_r")
    lo, hi = cat.bounds()
    c = (lo + hi) / 2
    alo, ahi = chair["armrest_r"].shape.bounds()
    assert alo[0] - 0.05 < c[0] < ahi[0] + 0.05 and alo[2] - 0.1 < c[2] < ahi[2] + 0.1
    # nothing of the cat is inside the armrest: sample its parts' surfaces near the bottom
    arm = chair["armrest_r"].shape
    P = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 15), np.linspace(lo[1], lo[1] + 0.03, 4),
                             np.linspace(lo[2], hi[2], 15)), -1).reshape(-1, 3)
    inside_cat = np.min([p.shape(P) for p in cat.parts], 0) < 0
    assert (arm(P)[inside_cat] > -0.004).all()                 # not sunk into the armrest
    assert arm(P)[inside_cat].min() < 0.01                       # but touching it
    # .face(a) turns a figure that faces +z toward (sin a, 0, cos a): 90 = +x, the same sign as rotate_y
    d = _animal("dog").face(90)
    assert d.anchor("nose")[0] > 0.1 and abs(d.anchor("nose")[2]) < 0.1


def test_clay_character_on_a_sofa_seat_and_cat_default_size():
    from pixelcanvas import solid as S
    from pixelcanvas.clay import animal as _animal, cast as _cast
    sofa = S.sofa().place(z=-0.5)
    kid = _cast("kid").on(sofa, "seat", at=(0.3, 0.5))
    assert 0.44 < kid.bounds()[0][1] < 0.52                     # on the seat cushions (top 0.50 m)
    h = _animal("cat").bounds()[1][1]
    assert 0.34 < h < 0.42                                     # reads next to a chair seat (0.45 m) and armrest
