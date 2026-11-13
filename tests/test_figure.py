"""Tests for pixelcanvas.figure (people)."""
import importlib
import math
import os
import re
import sys

import numpy as np
import pytest

from pixelcanvas.canvas import Canvas
from pixelcanvas import figure as fg
from pixelcanvas.figure import (Figure, person, random_person, crowd, pose_lerp, walk_pose, POSES, PEOPLE,
                                HAIR_STYLES, EXPRESSIONS, TOPS, BOTTOMS, HAND_SHAPES, STYLES, BUILDS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cv(w=240, h=180):
    return Canvas(w, h, (0.9, 0.9, 0.9))


def test_draw_returns_mask_bbox_and_anchors():
    cv = _cv()
    f = Figure(height=0.8).pose("wave")
    assert f.draw(cv, at=(0.5, 0.95)) is f
    assert f.mask.shape == (cv.h, cv.w) and f.mask.max() > 0.9
    x0, y0, x1, y1 = f.bbox
    assert 0 <= x0 < x1 <= cv.w and 0 <= y0 < y1 <= cv.h
    for name in ("hand_l", "hand_r", "head_top", "eye_l", "mouth", "feet", "shoulder_r", "speech", "seat"):
        x, y = f.anchor(name)
        assert math.isfinite(x) and math.isfinite(y)
    # the waving (right) hand is above the head top's lower neighbour: above the shoulder
    assert f.anchor("hand_r")[1] < f.anchor("shoulder_r")[1]
    assert f.anchor("head_top")[1] < f.anchor("mouth")[1] < f.anchor("feet")[1]
    assert abs(f.anchor("feet")[1] - 0.95 * cv.h) < 1e-6
    u, v = f.anchor_uv("feet")
    assert abs(u - 0.5) < 1e-6
    with pytest.raises(KeyError):
        f.anchor("tail")


def test_height_in_heads():
    cv = _cv()
    for build, heads in (("average", 7.4), ("child", 5.2), ("chibi", 2.8)):
        f = Figure(height=0.6, build=build).layout(cv, (0.5, 0.9))
        assert abs(f.head_size * BUILDS[build]["heads"] - 0.6 * cv.h) < 1e-6
        top = f.anchor("head_top")[1]
        assert abs((0.9 * cv.h - top) - 0.6 * cv.h) < 0.12 * cv.h


def test_every_pose_style_and_view_draws():
    cv = _cv(320, 160)
    names = [p for p in POSES]
    for i, name in enumerate(names):
        st = STYLES[i % 4]
        view = ("front", "three_quarter", "side", "back")[i % 4]
        f = Figure(height=0.45, style=st, view=view, facing=-1 if i % 3 == 0 else 1).pose(name, phase=0.3)
        f.draw(cv, at=((i % 6 + 0.5) / 6, 0.95))
        assert f.mask.max() > 0.5, name


def test_pose_interpolation_and_overrides():
    a = pose_lerp("stand", "wave", 0.5)
    assert 0 < a["elbow_r"] < POSES["wave"]["elbow_r"]
    w0, w1 = walk_pose(0.0), walk_pose(0.25)
    assert w0["hip_l"] != w1["hip_l"]
    f = Figure().pose("walk", phase=0.3, elbow_r=90, hand_r="point")
    assert f._pose["elbow_r"] == 90 and f._pose["hand_r"] == "point"
    with pytest.raises(ValueError):
        Figure().pose("stand", knee_x=3)
    with pytest.raises(ValueError):
        Figure().pose("moonwalk")


def test_far_arm_behind_torso_in_three_quarter_view():
    cv = _cv()
    f = Figure(view="three_quarter").layout(cv, (0.5, 0.95))
    # facing screen right: the left arm (+x) is the far one
    assert f._Z["elbow_l"] < f._Z["chest"] < f._Z["elbow_r"]


def test_options_cover_hair_tops_bottoms_hands_expressions():
    cv = _cv(400, 120)
    for i, h in enumerate(HAIR_STYLES):
        Figure(height=0.5, hair=dict(style=h, color="auburn")).draw(cv, ((i + 0.5) / len(HAIR_STYLES), 0.95))
    for i, t in enumerate(TOPS):
        Figure(height=0.5, outfit=dict(top=t)).draw(cv, ((i + 0.5) / len(TOPS), 0.6))
    for i, b in enumerate(BOTTOMS):
        Figure(height=0.5, view="three_quarter", outfit=dict(bottom=b)).pose("sit").draw(cv, ((i + 0.5) / 5, 0.95))
    for i, e in enumerate(EXPRESSIONS):
        Figure(height=1.4, face=dict(expression=e), style=STYLES[i % 4]).draw(cv, ((i + 0.5) / 8, 1.5))
    for i, hs in enumerate(HAND_SHAPES):
        Figure(height=1.0).pose("wave", hand_r=hs).draw(cv, ((i + 0.5) / 8, 1.0))
    extras = ["glasses", "cap", "headphones", "scarf", "bag", "backpack", "beard", "hat", "beanie", "helmet"]
    Figure(height=0.8, outfit=dict(extras=extras)).draw(cv, (0.5, 0.95))
    with pytest.raises(ValueError):
        Figure(hair=dict(style="mohawk-xl"))
    with pytest.raises(ValueError):
        Figure(outfit=dict(top="cape"))


def test_people_presets_and_description_parsing():
    cv = _cv(480, 120)
    for i, name in enumerate(PEOPLE):
        f = person(name, height=0.8)
        f.draw(cv, ((i + 0.5) / len(PEOPLE), 0.95))
        assert f.mask.max() > 0.5
    f = person("a tall woman in a red coat waving")
    assert f.outfit["top"] == "coat" and f.colors["top"] == fg.PALETTE["red"] and f._pose_name == "wave"
    assert f.body == "feminine"
    g = person("an old man with glasses and a beard, sitting")
    assert g.build == "elder" and "glasses" in g.extras and g._pose_name == "sit"


def test_random_person_and_crowd_with_ground():
    from pixelcanvas.form import Ground
    a, b = random_person(1), random_person(2)
    assert (a.skin, a.hair["style"], a.colors["top"]) != (b.skin, b.hair["style"], b.colors["top"])
    cv = _cv(320, 180)
    figs = crowd(cv, 6, region=(0.05, 0.6, 0.95, 0.95), seed=3)
    assert len(figs) == 6
    ys = [f.anchor("feet")[1] for f in figs]
    assert ys == sorted(ys)             # drawn back to front
    g = Ground(cv, horizon=0.4)
    figs = crowd(cv, 4, region=(0.1, 0.55, 0.9, 0.95), seed=4, ground=g)
    sizes = [f.head_size for f in figs]
    assert sizes[0] < sizes[-1]          # farther people are smaller


def test_small_figures_are_fast_and_simplified():
    import time
    cv = _cv(400, 200)
    t = time.time()
    for i in range(12):
        random_person(i, height=0.12).draw(cv, ((i + 0.5) / 12, 0.9))
    assert time.time() - t < 6.0


def test_canvas_reach_target_points_the_arm_in_every_view():
    # a canvas target (kite string, balloon) used to be pushed far behind the body in side views
    cv = _cv(400, 400)
    tgt = (0.9, 0.1)
    for view in ("front", "three_quarter", "side"):
        f = Figure(height=0.5, view=view).pose("run", phase=0.25, reach_r=tgt, hand_r="grip").layout(cv, (0.4, 0.9))
        sx, sy = f.anchor("shoulder_r")
        hx, hy = f.anchor("hand_r")
        d1 = np.array([hx - sx, hy - sy])
        d2 = np.array([tgt[0] * cv.w - sx, tgt[1] * cv.h - sy])
        cos = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2)))
        assert cos > 0.95, view
        assert abs(f._Z["hand_r"] - f._Z["shoulder_r"]) < 0.8, view


def test_run_cycle_feet_and_flight():
    from pixelcanvas.figure import run_pose
    p = run_pose(0.25)
    # trailing foot pointed (plantar flexed), front foot near level, airborne at full stride
    assert p["ankle_r"] < -20 and abs(p["ankle_l"]) < 25 and p["lift"] > 0.1
    q = run_pose(0.0)
    assert q["lift"] < 0.01 and q["knee_l"] > 90 and q["knee_r"] < 40     # passing: tucked + support
    # the front thigh is ahead of the body at a stride
    cv = _cv(300, 300)
    f = Figure(height=0.6, view="side").pose("run", phase=0.25).layout(cv, (0.5, 0.9))
    assert f.anchor("knee_l")[0] > f.anchor("hip_l")[0] > f.anchor("knee_r")[0]


def test_bare_feet_join_the_leg_and_heads_cover_the_neck():
    cv = _cv(300, 300)
    f = Figure(height=0.8, view="side", outfit=dict(top="tshirt", bottom="shorts", shoes=None)).pose("run", phase=0.25)
    f.draw(cv, at=(0.5, 0.95))
    # the foot is inside the figure mask between ankle and toe
    ax, ay = f.anchor("ankle_r")
    tx, ty = f.anchor("toe_r")
    mx, my = int((ax + tx) / 2), int((ay + ty) / 2)
    assert f.mask[my, mx] > 0.5
    # a stooped elder seen from behind: the head (hair) is painted over the neck
    g = person("grandma", view="three_quarter_back", style="flat", height=2.0)
    cv2 = Canvas(300, 300, (0.9, 0.9, 0.9))
    g.draw(cv2, at=(0.5, 2.05))
    hx, hy = g.anchor("head")
    px = cv2.buf[int(hy), int(hx), :3]
    hair = fg._col(g.hair["color"])
    skin = fg._col(g.skin)
    assert np.abs(px - hair).sum() < np.abs(px - skin).sum()


def test_chibi_feet_are_longer_than_wide():
    f = Figure(build="chibi")
    assert f.P["foot_w"] < f.P["foot_len"]


def test_reference_doc_snippets_run(tmp_path):
    path = os.path.join(ROOT, "skills", "pixelcanvas", "references", "figures.md")
    text = open(path, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for code in blocks:
            code = code.replace("Canvas(1920, 1080", "Canvas(480, 270").replace("Canvas(1600, 900", "Canvas(400, 225")
            exec(compile(code, "figures.md", "exec"), {})
    finally:
        os.chdir(cwd)


# ---- strict options, aliases, registry names, manifest

def test_figure_unknown_option_names_the_closest():
    with pytest.raises(TypeError, match=r"unknown option 'hieght'.*did you mean 'height'"):
        Figure(hieght=0.5)
    with pytest.raises(TypeError, match=r"person: unknown option 'pos'|person: unknown option 'outfitt'"):
        person("barista", outfitt=dict(top="shirt"))
    with pytest.raises(TypeError, match=r"Figure hair: unknown option 'styel'.*did you mean 'style'"):
        Figure(hair=dict(styel="bob"))
    with pytest.raises(TypeError, match=r"Figure.pose: unknown option 'elbow_x'.*did you mean 'elbow_r' or 'elbow_l'"):
        Figure().pose("stand", elbow_x=3)
    with pytest.raises(TypeError, match=r"unknown option 'shadw'.*did you mean 'shadow'"):
        Figure().draw(Canvas(100, 100), (0.5, 0.9), shadw=True)


def test_figure_aliases_are_accepted():
    cv = Canvas(160, 200)
    f = Figure(height=0.6, outfit=dict(top="shirt", colours=dict(top="#c33"))).pose("stand")   # colours -> colors
    f.draw(cv, pos=(0.5, 0.9))                                                                 # pos -> at
    assert f.colors["top"] == "#c33" and f.bbox is not None


def test_figure_unknown_registry_names():
    with pytest.raises(ValueError, match=r"unknown pose 'wavve'.*did you mean 'wave'"):
        Figure().pose("wavve")
    with pytest.raises(ValueError, match=r"unknown hair style 'bobb'.*did you mean 'bob'"):
        Figure(hair=dict(style="bobb"))
    with pytest.raises(ValueError, match=r"unknown top 'tshrt'.*did you mean 'tshirt'"):
        Figure(outfit=dict(top="tshrt"))
    with pytest.raises(ValueError, match=r"unknown style 'flt'.*did you mean 'flat'"):
        Figure(style="flt")


def test_each_drawn_person_is_one_manifest_entry():
    from pixelcanvas import manifest
    cv = Canvas(480, 270)
    person("barista").draw(cv, (0.3, 0.9))
    person("chef", name="head chef").draw(cv, (0.6, 0.9))
    crowd(cv, 3, region=(0.1, 0.7, 0.9, 0.95), seed=2)
    rows = [r for r in manifest.summary(cv) if r["kind"] == "person"]
    assert len(rows) == 5 and rows[0]["name"] == "barista" and rows[1]["name"] == "head chef"
    assert rows[0]["visible_pct"] > 90 and "people" in rows[0]["aliases"]


# ---- round 1 fixes: pixel positions, one-call two-handed hold, names, face scale

def test_at_in_pixels_raises_and_says_so():
    cv = Canvas(320, 200)
    with pytest.raises(ValueError, match=r"at=\(300.0, 180.0\) looks like pixels.*fractions of the canvas"):
        person("chef").draw(cv, at=(300, 180))
    with pytest.raises(ValueError, match=r"looks like pixels"):
        person("chef").look_at((300, 100))
    with pytest.raises(ValueError, match=r"looks like pixels"):
        person("chef").pose("stand", reach_r=(300, 100))
    with pytest.raises(ValueError, match=r"looks like pixels"):
        crowd(cv, 3, region=(10, 10, 300, 190))


def _hands(fig):
    return fig.anchor("hand_l"), fig.anchor("hand_r")


def test_hold_both_never_crosses_the_arms_in_any_view():
    cv = Canvas(400, 400, "#ddd")
    for view in ("front", "three_quarter", "side", "back"):
        for facing in (1, -1):
            f = person("chef", view=view, facing=facing, height=0.7).pose("hold", hold_both=(0.5, 0.6), hold_width=0.2)
            f.draw(cv, at=(0.5, 0.95))
            xl, xr = f.anchor("hand_l")[0], f.anchor("hand_r")[0]
            gap = abs(xl - xr)
            assert gap < 0.35 * cv.w
            if view == "front":
                assert xr < xl                                 # the figure's right hand is on the viewer's left
                assert gap > 0.12 * cv.w
    assert f.anchor("hand_l") is not None


def test_hold_both_front_view_places_hands_on_both_sides_of_the_object():
    cv = Canvas(600, 500, "#ddd")
    f = person("chef", view="front", height=0.74).pose("hold", hold_both=(0.4, 0.6), hold_width=0.29)
    f.draw(cv, at=(0.4, 0.95))
    xr, xl = f.anchor("hand_r")[0] / cv.w, f.anchor("hand_l")[0] / cv.w
    assert xr < 0.4 < xl
    assert abs((xl - xr) - 0.29) < 0.07
    assert f._pose["item"] is None and f._pose["hand_r"] == "grip" and f._pose["hand_l"] == "grip"


def test_hold_both_rejects_bad_values():
    with pytest.raises(ValueError, match="hold_both"):
        person("chef").pose("hold", hold_both=(0.4, 0.6, 0.2))
    with pytest.raises(ValueError, match="looks like pixels"):
        person("chef").pose("hold", hold_both=(200, 300))


def test_held_arms_keep_a_slight_bend():
    cv = Canvas(500, 500)
    f = person("chef", height=0.7).pose("hold", hold_both=(0.5, 0.6), hold_width=0.6)
    f.draw(cv, at=(0.5, 0.95))
    sh, el, wr = (np.array(f.anchor(n + "_r")) for n in ("shoulder", "elbow", "wrist"))
    straight = np.linalg.norm(wr - sh)
    bent = np.linalg.norm(el - sh) + np.linalg.norm(wr - el)
    assert straight < 0.99 * bent


def test_person_name_and_alias_are_recorded():
    from pixelcanvas import manifest
    cv = Canvas(400, 300)
    person("barista", name="friend", alias=["classmate", "buddy"]).draw(cv, (0.3, 0.9))
    person("chef").draw(cv, (0.7, 0.9))
    rows = [r for r in manifest.summary(cv) if r["kind"] == "person"]
    assert rows[0]["name"] == "friend"
    assert {"classmate", "buddy", "person", "people"} <= set(rows[0]["aliases"])
    assert "person" in rows[1]["aliases"] or rows[1]["name"] == "chef"


def test_wrong_names_get_did_you_mean():
    with pytest.raises(ValueError, match=r"unknown body 'mal'.*did you mean"):
        Figure(body="mal")
    with pytest.raises(ValueError, match=r"unknown view 'frnt'.*did you mean 'front'"):
        Figure(view="frnt")
    with pytest.raises(ValueError, match=r"unknown skin 'tanned'.*did you mean 'tan'"):
        Figure(skin="tanned")
    with pytest.raises(ValueError, match=r"unknown outfit colour 'shirt'"):
        Figure(outfit=dict(colors=dict(shirt="#fff")))
    with pytest.raises(ValueError, match=r"unknown hand shape 'fisst'.*did you mean 'fist'"):
        Figure().pose("stand", hand_r="fisst")
    with pytest.raises(ValueError, match=r"unknown palm 'x'"):
        Figure().pose("stand", palm_r="x")
    with pytest.raises(ValueError, match=r"unknown item 'cupp'.*did you mean 'cup'"):
        person("chef", item="cupp")
    with pytest.raises(ValueError, match=r"unknown anchor 'hnd_r'.*did you mean 'hand_r'"):
        f = person("chef")
        f.draw(Canvas(200, 200), (0.5, 0.9))
        f.anchor("hnd_r")
    with pytest.raises(ValueError, match=r"unknown colour 'puce'"):
        Figure(hair=dict(color="puce"))
    assert Figure(facing="left").facing == -1


def test_chibi_face_is_readable_at_thumbnail_size():
    cv = Canvas(1280, 720, "#fff")
    f = person("chef", build="chibi", height=0.64, face=dict(expression="surprised"))
    f.draw(cv, at=(0.5, 0.95))
    assert f.head_size > 0.2 * 720 * 0.64 * 0.9
    avg = person("chef", height=0.64)
    avg.draw(Canvas(1280, 720), at=(0.5, 0.95))
    assert f.head_size > 2 * avg.head_size


def test_full_body_eyes_are_larger_than_before():
    from pixelcanvas.figure import BUILDS
    assert BUILDS["average"]["eye"] >= 1.15 and BUILDS["broad"]["eye"] >= 1.1


# ---- a sitting figure sits on something
def _sit(seat=None, view="side", at=(0.5, 0.95)):
    cv = Canvas(320, 240, (0.9, 0.9, 0.9))
    f = Figure(height=0.8, view=view)
    f.pose("sit") if seat is None else f.pose("sit", seat=seat)
    f.draw(cv, at=at, shadow=False)
    return cv, f


def test_sit_draws_a_seat_by_default_and_opts_out():
    base, f = _sit("none")
    sx, sy = f.anchor("seat")
    gy = f.anchor("feet")[1]
    for view in ("side", "front", "three_quarter", "back"):
        cv, g = _sit(None, view)
        none, _ = _sit("none", view)
        diff = np.abs(cv.buf - none.buf).sum(2) > 0.05
        ys, xs = np.nonzero(diff)
        assert len(ys) > 200, view                                   # something under the seat
        assert ys.max() >= gy - 3 and ys.min() >= g.anchor("seat")[1] - 0.6 * g.head_size, view
    cv, _ = _sit(None)
    col = cv.buf[int(sy + 2):int(gy - 2), int(sx) - 30:int(sx) + 30]
    assert (np.abs(col - 0.9).sum(2) > 0.1).any()                    # stool between the seat and the ground
    for kind in ("stool", "bench", "chair"):
        k, _ = _sit(kind)
        assert np.abs(k.buf - base.buf).sum() > 1.0
    assert Figure().pose("sit", seat="none")._pose["seat"]           # still seated clothing
    with pytest.raises(ValueError):
        Figure().pose("sit", seat="throne")


def test_sit_at_a_scene_spot_leaves_the_seat_to_the_scene():
    from pixelcanvas.form import Spot
    a, _ = _sit(None, at=Spot(0.5, 0.95))
    b, _ = _sit("none", at=Spot(0.5, 0.95))
    assert np.allclose(a.buf, b.buf)
    c, _ = _sit("stool", at=Spot(0.5, 0.95))
    assert not np.allclose(c.buf, b.buf)
