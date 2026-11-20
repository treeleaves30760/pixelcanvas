"""3D layout for free mode: camera, posed figures (FK, IK, limits, presets), blocks and props through one z-buffer,
and a posed figure working with every later tool that reads a ``sketch figure`` record."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pixelcanvas import pose3d as P3
from pixelcanvas import render3d as R3
from pixelcanvas.coords import UsageError

from .test_draw_wear import draw, ok, state  # noqa: F401  (draw is a fixture)


# ============================================================================================ skeleton

def test_rest_pose_proportions_follow_the_head_count():
    for heads in (2.5, 4.0, 6.5, 8.0):
        body = P3.Body(1.6, heads)
        J, F = P3.fk(body, {}, 0.0)
        top, chin = J["head-top"][1], J["chin"][1]
        # the head is 1/heads of the height (within the chin's small forward offset)
        assert abs((top - chin) - 1.6 / heads) < 0.02 * 1.6, heads
        # feet below the hips, hands at the sides, left and right mirror each other
        assert J["ankle-r"][1] < J["knee-r"][1] < J["hip-r"][1] < J["neck"][1] < J["head"][1]
        assert np.allclose(J["hand-r"] * [-1, 1, 1], J["hand-l"], atol=1e-9)


def test_fk_angles_move_limbs_the_named_way():
    body = P3.Body(1.6, 6.5)
    # yaw 0 faces the viewer (forward = -z); her right is the picture's left (-x)
    J, F = P3.fk(body, {}, 0.0)
    assert J["shoulder-r"][0] < 0 < J["shoulder-l"][0]
    assert F["chest"][2, 2] < -0.99
    # arm-r forward 90: the arm points ahead (toward the viewer), level with the shoulder
    J, _ = P3.fk(body, {"arm-r": {"forward": 90}}, 0.0)
    v = P3.unit(J["elbow-r"] - J["shoulder-r"])
    assert np.allclose(v, [0, 0, -1], atol=1e-6)
    # raise 90: out to her right side, level
    J, _ = P3.fk(body, {"arm-r": {"raise": 90}}, 0.0)
    assert np.allclose(P3.unit(J["elbow-r"] - J["shoulder-r"]), [-1, 0, 0], atol=1e-6)
    # raise 180: straight up
    J, _ = P3.fk(body, {"arm-r": {"raise": 180}}, 0.0)
    assert np.allclose(P3.unit(J["elbow-r"] - J["shoulder-r"]), [0, 1, 0], atol=1e-6)
    # elbow 90 with the arm hanging: the forearm comes forward
    J, _ = P3.fk(body, {"elbow-r": {"bend": 90}}, 0.0)
    assert np.allclose(P3.unit(J["wrist-r"] - J["elbow-r"]), [0, 0, -1], atol=1e-6)
    # knee 90 with the thigh hanging: the shin goes back
    J, _ = P3.fk(body, {"knee-l": {"bend": 90}}, 0.0)
    assert np.allclose(P3.unit(J["ankle-l"] - J["knee-l"]), [0, 0, 1], atol=1e-6)
    # yaw 90 faces the picture's right; head turn + turns toward her left
    _, F = P3.fk(body, {}, 90.0)
    assert np.allclose(F["chest"][:, 2], [1, 0, 0], atol=1e-6)
    _, F = P3.fk(body, {"head": {"turn": 90 - 30}, "neck": {"turn": 30}}, 0.0)
    assert np.allclose(F["head"][:, 2], [1, 0, 0], atol=1e-6)
    # bone lengths never change
    J, _ = P3.fk(body, {"arm-l": {"raise": 37, "forward": 71, "twist": 20}, "elbow-l": {"bend": 66}}, 15.0)
    assert abs(np.linalg.norm(J["elbow-l"] - J["shoulder-l"]) - body.Lu) < 1e-9
    assert abs(np.linalg.norm(J["wrist-l"] - J["elbow-l"]) - body.Lf) < 1e-9


def test_edits_parse_aliases_and_clamp_to_limits():
    E = P3.parse_edits("arm-r: raise 120, forward 30; elbow r: 90; spine: twist 20, bend 15; head: turn -30, tilt 10;"
                       " knee-l: 100; right foot: flex 10; leg_l back 20")
    assert E["arm-r"] == {"raise": 120, "forward": 30}
    assert E["elbow-r"] == {"bend": 90}
    assert E["spine"] == {"twist": 20, "bend": 15}
    assert E["head"] == {"turn": -30, "tilt": 10}
    assert E["knee-l"] == {"bend": 100} and E["foot-r"] == {"point": -10} and E["leg-l"] == {"forward": -20}
    C, notes = P3.clamp_edits({"knee-l": {"bend": 175}, "elbow-r": {"bend": -20}, "head": {"turn": 30}})
    assert C["knee-l"]["bend"] == 160 and C["elbow-r"]["bend"] == 0 and C["head"]["turn"] == 30
    assert any("knee-l bend 175 -> 160" in n for n in notes) and len(notes) == 2
    for bad in ("arm: raise 30", "tail: 30", "arm-r: 30", "elbow-r: swing 30"):
        with pytest.raises(UsageError):
            P3.parse_edits(bad)


def test_mirror_swaps_sides_and_turns():
    E = P3.mirror_edits({"arm-r": {"raise": 90}, "head": {"turn": 20, "nod": 10}, "spine": {"twist": 5}})
    assert E == {"arm-l": {"raise": 90}, "head": {"turn": -20, "nod": 10}, "spine": {"twist": -5}}


def test_two_bone_ik_reaches_and_reports_the_gap():
    S, T = np.array([0.0, 1.0, 0.0]), np.array([0.3, 0.7, 0.2])
    M, E, gap = P3.two_bone(S, T, 0.3, 0.25, [0, 0, -1])
    assert np.allclose(E, T, atol=1e-3) and gap < 0
    assert abs(np.linalg.norm(M - S) - 0.3) < 1e-9 and abs(np.linalg.norm(E - M) - 0.25) < 1e-9
    assert (M - (S + T) / 2) @ np.array([0, 0, -1]) > 0        # bends toward the pole
    M, E, gap = P3.two_bone(S, S + [1.0, 0, 0], 0.3, 0.25, [0, 0, -1])
    assert abs(gap - 0.45) < 1e-6 and abs(np.linalg.norm(E - S) - 0.55) < 1e-3
    # on a figure: the hand goes to a railing top
    cam = P3.Camera(400, 500, horizon=400)
    rail = P3.Block("rail", "railing", cam.ground_point(560, 880), (2.0, 0.06, 1.0))
    res = P3.solve_figure({"preset": "stand", "at": [500, 900], "targets": {"hand-r": "rail", "hand-l": "rail"}},
                          cam, {"rail": rail}, {})
    for s in "rl":
        grip = res["J"][f"wrist-{s}"] + P3.unit(res["J"][f"hand-{s}"] - res["J"][f"wrist-{s}"]) * 0.5 * res["body"].Lh
        assert np.linalg.norm(grip - rail.closest(res["J"][f"shoulder-{s}"])) < 0.08, s
    assert any("reaches the rail" in ln for ln in res["ik"])


# ============================================================================================ presets

@pytest.mark.parametrize("name", list(P3.PRESETS))
def test_every_preset_renders_on_the_ground(name):
    cam = P3.Camera(240, 300, horizon=380, eye=1.4)
    res = P3.solve_figure({"preset": name, "at": [500, 900]}, cam, {}, {})
    lows = [P3.lowest(pr)[0] for pr in res["prims"]]
    if P3.PRESETS[name]["rest"] == "air":
        assert min(lows) > 0.2 and res["balance"].startswith("airborne")
    else:
        assert abs(min(lows)) < 1e-6, name
        assert res["contacts"]
    for pr in res["prims"]:
        pr["owner"] = "f"
    R = R3.render(cam, res["prims"])
    assert (R["id"] >= 0).sum() > 500
    assert np.isfinite(R["normal"]).all()
    assert not res["clamps"], res["clamps"]


def test_run_reads_as_running():
    cam = P3.Camera(400, 500, horizon=400)
    res = P3.solve_figure({"preset": "run", "at": [500, 900], "yaw": 90}, cam, {}, {})
    J = res["J"]
    feet = {s: min(J[f"heel-{s}"][1], J[f"foot-{s}"][1]) for s in "rl"}
    assert feet["l"] < 0.03 and feet["r"] > 0.2               # one foot pushes off, the other is up
    assert J["head"][0] - J["pelvis"][0] > 0.12                 # leaning into the run (faces +x)
    # opposite arm and leg: the left hand is ahead, the right hand behind
    assert J["hand-l"][0] > J["shoulder-l"][0] and J["hand-r"][0] < J["shoulder-r"][0] + 0.05
    assert J["knee-r"][0] > J["hip-r"][0] + 0.2                 # the right knee is driven forward


def test_seated_preset_sits_on_a_block_and_jump_is_airborne():
    cam = P3.Camera(400, 500, horizon=400)
    wall = P3.Block("wall", "box", cam.ground_point(500, 800), (3.0, 0.4, 0.9))
    res = P3.solve_figure({"preset": "sit-ledge", "on": "wall"}, cam, {"wall": wall}, {})
    pel = next(pr for pr in res["prims"] if pr["part"] == "pelvis")
    assert abs(P3.lowest(pel)[0] - 0.9) < 1e-6
    assert any(n == "seat" and s == "wall" for n, s, _ in res["contacts"])
    assert min(res["J"]["foot-r"][1], res["J"]["foot-l"][1]) < 0.9      # the legs hang below the top
    res = P3.solve_figure({"preset": "jump", "at": [500, 900]}, cam, {}, {})
    assert res["balance"].startswith("airborne")


# ============================================================================================ camera and rendering

def test_perspective_scale_same_person_farther_is_smaller():
    cam = P3.Camera(1000, 1000, horizon=450, eye=1.5, fov=50)
    rows = cam.person_table(1.6, ys=[600, 800, 1000])
    hs = [h for _, _, h in rows]
    ds = [d for _, d, _ in rows]
    assert hs[0] < hs[1] < hs[2] and ds[0] > ds[1] > ds[2]
    # height in the picture is (eye ratio) x distance below the horizon, whatever the lens
    assert abs(hs[2] - 1.6 / 1.5 * 550) < 2
    near = P3.solve_figure({"preset": "stand", "at": [500, 950]}, cam, {}, {})
    far = P3.solve_figure({"preset": "stand", "at": [500, 600]}, cam, {}, {})

    def tall(res):
        U = cam.project_units(np.array([res["J"]["head-top"], res["J"]["heel-r"]]))
        return U[1, 1] - U[0, 1]
    assert tall(far) < 0.5 * tall(near)
    # a low camera: a standing person's head is above the horizon
    low = P3.Camera(1000, 1000, horizon=800, eye=0.5)
    res = P3.solve_figure({"preset": "stand", "at": [500, 950]}, low, {}, {})
    assert low.project_units([res["J"]["head"]])[0, 1] < 800
    with pytest.raises(UsageError):
        P3.solve_figure({"preset": "stand", "at": [500, 300]}, cam, {}, {})


def test_occlusion_a_wall_hides_the_figure_behind_it():
    cam = P3.Camera(300, 300, horizon=400, eye=1.5)
    g = cam.ground_point(500, 700)
    wall = P3.Block("wall", "box", g, (3.0, 0.3, 1.2))
    res = P3.solve_figure({"preset": "stand", "at": [500, 560]}, cam, {"wall": wall}, {})
    prims = wall.prims("w")
    for pr in res["prims"]:
        pr["owner"] = "f"
        prims.append(pr)
    R = R3.render(cam, prims)
    owner = np.array([pr["owner"] for pr in prims])
    seen = owner[R["id"][R["id"] >= 0]]
    fig_seen = (seen == "f").sum()
    full = sum(int(R["own"][k][1].sum()) for k, pr in enumerate(prims) if pr["owner"] == "f" and R["own"][k])
    assert 0 < fig_seen < 0.9 * full                          # the legs are behind the wall
    # the hidden part is below the wall's top edge: every figure pixel seen is above the wall's top
    wy = cam.project([g + [0, 1.2, 0]])[0][0, 1]
    ys = np.nonzero(np.isin(R["id"], [k for k, pr in enumerate(prims) if pr["owner"] == "f"]))[0]
    assert ys.max() <= wy + 2
    # the figure's own hidden lines are dashes; the wall hides lines without dashes
    lines = R3.contour_lines(R, prims, 1.0)
    assert any(wf == 1.0 for _, _, wf in lines["f"]) and any(wf == 1.0 for _, _, wf in lines["w"])


def test_intersections_are_exact_for_each_primitive():
    D = np.array([[0.0, 0.0, 1.0]])
    for t, A, want in (("ell", np.diag([1, 1, 1.0]), 4.0), ("box", np.diag([1, 2, 1.0]), 4.0),
                       ("cyl", np.diag([1, 1, 1.0]), 4.0)):
        hit, n, _ = R3.intersect({"t": t, "c": np.array([0, 0, 5.0]), "A": A}, np.zeros(3), D)
        assert abs(hit[0] - want) < 1e-9 and np.allclose(n[0], [0, 0, -1])


def test_form_maps_light_core_and_cast():
    cam = P3.Camera(200, 200, horizon=300, eye=1.5)
    res = P3.solve_figure({"preset": "stand", "at": [500, 800]}, cam, {}, {})
    M = R3.form_maps(cam, res["prims"], [-0.6, -0.6, 0.5])            # light from the top-left front
    fig = M["id"] >= 0
    assert (M["cls"][fig] == 1).mean() > 0.3 and (M["cls"][fig] == 2).mean() > 0.05
    assert (M["cls"][M["ground"]] == 3).any()                        # the figure's shadow on the ground


# ============================================================================================ the commands

REQ = "an anime girl running down a street past a bench, a boy sitting on a wall"


def _scene(run):
    ok(run("new", "w", "--size", "300x400", "--request", REQ))
    ok(run("style", "cel"))
    ok(run("do", "plan subject girl --box 200,40,800,980; plan part face --box 400,60,600,200 --of girl; "
                 "plan part shirt --box 300,200,700,520 --of girl; plan part skirt --box 300,480,700,700 --of girl; "
                 "plan part hand-r --box 250,380,400,520 --of girl; plan part hair --box 330,30,670,330 --of girl; "
                 "plan light top-left; palette make; palette pick 1; stage done"))


def test_commands_are_installed(draw):
    from pixelcanvas import cli, draw_pose
    assert set(draw_pose.COMMANDS) == {"camera", "pose", "block", "prop", "form"}
    for c in draw_pose.COMMANDS:
        assert c in cli.COMMANDS


def test_todo_proposes_camera_blocks_pose_dress_hair_then_render_and_face(draw):
    _scene(draw)
    st = state()
    sk = [t for t in st["todo"] if t["stage"] == "sketch" and t["kind"] == "task"]
    keys = [t["key"] for t in sk]
    assert keys.index("camera") < keys.index("blocks") < keys.index("construct") < keys.index("dress") < \
        keys.index("hair3d")
    assert "head" not in keys                       # the 3D figure has its own head
    assert sk[keys.index("construct")]["cmd"].startswith("pixelcanvas pose girl")
    out = ok(draw("next"))
    assert "pixelcanvas camera" in out
    ok(draw("pose", "girl", "--preset", "run"))
    ln = [t for t in state()["todo"] if t["stage"] == "line" and t["kind"] == "task"]
    lk = [t["key"] for t in ln]
    assert lk[0] == "render" and lk.index("render") < lk.index("face")
    assert ln[lk.index("face")]["cmd"].startswith("pixelcanvas face girl --expression")
    # a flat mannequin keeps the 2D path: no dress or hair3d items, the head construction comes back
    ok(draw("sketch", "figure", "girl", "--pose", "walk"))
    keys = [t["key"] for t in state()["todo"] if t["stage"] == "sketch"]
    assert "head" in keys


def test_posed_figure_works_with_every_later_tool(draw):
    _scene(draw)
    out = ok(draw("camera", "--horizon", "780", "--eye-height", "0.8"))
    assert "low camera" in out and "units tall" in out
    ok(draw("block", "bench", "box", "--at", "760,930", "--size", "1.5,0.45,0.45"))
    out = ok(draw("pose", "girl", "--preset", "run", "--yaw", "60"))
    assert "touching:" in out and "balance" in out and "box " in out
    st = state()
    fig = next(el for el in st["draw"]["sketch"] if el["kind"] == "figure" and el["subject"] == "girl")
    # the same record `sketch figure` writes, plus the 3D data
    assert set(fig["joints"]) == {"head", "chin", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist",
                                  "lhip", "rhip", "lknee", "rknee", "lankle", "rankle"}
    assert fig["pose"] == "run" and fig["heads"] == 6.5 and "j3d" in fig and "j2d" in fig and "frames" in fig
    assert fig["joints"]["lshoulder"][0] <= fig["joints"]["rshoulder"][0] + 1
    assert (st["draw"]["sketch"] and any(el["kind"] == "horizon" and el.get("camera") for el in st["draw"]["sketch"]))
    out = ok(draw("pose", "girl", "--set", "head: turn -20; arm-l: forward 80", "--show", "joints"))
    assert "head seen at yaw" in out
    out = ok(draw("prop", "bag", "--kind", "bag", "--hand", "r"))
    out = ok(draw("sketch", "head", "girl"))
    ok(draw("stage", "done"))
    ok(draw("line", "trace", next(el["id"] for el in state()["draw"]["sketch"] if el["kind"] == "head")))
    ok(draw("face", next(el["id"] for el in state()["draw"]["sketch"] if el["kind"] == "head")))
    ok(draw("wear", "girl", "shirt"))
    ok(draw("wear", "girl", "skirt"))
    ok(draw("hand", "girl", "r"))
    ok(draw("locks", "hair", "--cut", "long"))
    ok(draw("line", "gaps"))
    ok(draw("flat", "auto"))
    out = ok(draw("form", "girl"))
    assert "core shadow" in out
    import os
    assert os.path.exists("w.pixelcanvas/draw/scene3d.npz") and os.path.exists("w.pixelcanvas/draw/form3d.npz")
    z = np.load("w.pixelcanvas/draw/scene3d.npz")
    assert z["id"].max() >= 0 and json.loads(str(z["parts"]))
    out = ok(draw("shade", "shirt"))
    assert "3D pose" in out
    ok(draw("undo"))


def test_a_second_figure_sits_on_a_wall_and_follows_it(draw):
    _scene(draw)
    ok(draw("plan", "subject", "boy", "--box", "550,300,950,800"))
    ok(draw("block", "wall", "box", "--at", "700,760", "--size", "3,0.4,1.0"))
    out = ok(draw("pose", "boy", "--preset", "sit-ledge", "--on", "wall"))
    assert "seat on the wall" in out
    y0 = next(el for el in state()["draw"]["sketch"] if el.get("subject") == "boy")["j3d"]["pelvis"][1]
    out = ok(draw("block", "wall", "--size", "3,0.4,1.4"))
    assert "re-posed on it: boy" in out
    y1 = next(el for el in state()["draw"]["sketch"] if el.get("subject") == "boy")["j3d"]["pelvis"][1]
    assert abs((y1 - y0) - 0.4) < 1e-3
    code, out = draw("pose", "boy", "--preset", "flyy")
    assert code != 0 and "EXAMPLE" in out and "fly" not in out.split("EXAMPLE")[1][:0]
    code, out = draw("block", "tower", "box", "--at", "500,100")
    assert code != 0 and "below the horizon" in out


def test_the_head_construction_follows_the_pose(draw):
    _scene(draw)
    ok(draw("pose", "girl", "--preset", "stand", "--yaw", "40"))
    ok(draw("sketch", "head", "girl"))
    h0 = next(el for el in state()["draw"]["sketch"] if el["kind"] == "head")
    assert h0["yaw"] > 20                                   # sketch head takes its turn from the pose
    out = ok(draw("pose", "girl", "--at", "300,950", "--yaw", "-40"))
    h1 = next(el for el in state()["draw"]["sketch"] if el["kind"] == "head")
    assert h1["id"] == h0["id"] and h1["box"][0] < h0["box"][0] - 50 and h1["yaw"] < -20
    assert "moved with the pose" in out
    # moving a posed figure's 2D joints by hand makes it a flat mannequin again
    ok(draw("sketch", "joint", "girl", "rwr", "700,500"))
    out = ok(draw("block", "crate", "box", "--at", "800,950"))
    fig = next(el for el in state()["draw"]["sketch"] if el["kind"] == "figure")
    assert "rig" not in fig and fig["joints"]["rwrist"] == [700.0, 500.0]


def test_flat_mannequin_still_works(draw):
    _scene(draw)
    ok(draw("sketch", "figure", "girl", "--pose", "walk"))
    ok(draw("sketch", "head", "girl"))
    ok(draw("stage", "done"))
    ok(draw("wear", "girl", "shirt"))


# ============================================================================================ gesture, modifiers, checks

def _cam():
    return P3.Camera(300, 420, horizon=380, eye=1.4)


def test_presets_cover_the_action_dataset_words():
    for name in ("run-toward", "sprint-side", "walk", "float", "jump-toward", "jump-strike", "jump-tuck", "fall",
                 "sit-chair", "sit-bench", "sit-ledge", "sit-floor", "seiza", "hug-knees", "sit-look-back", "kneel",
                 "kneel-up", "crouch", "squat", "lie-back", "lie-side", "lie-front", "lean-wall", "lean-railing",
                 "lean-forward", "lean-elbow", "reach-up", "reach-viewer", "wave", "point", "hands-on-hips",
                 "arms-crossed", "hands-behind", "hand-to-face", "hand-to-hair", "hold-front", "carry", "kick",
                 "punch", "sword-high", "sword-ready", "swing", "dance-step", "cheer", "look-back", "stretch", "bow",
                 "aim", "ride", "swim"):
        assert P3.preset_key(name) == name
    for alias, name in (("hover", "float"), ("akimbo", "hands-on-hips"), ("bike", "ride"), ("dance", "dance-twirl")):
        assert P3.preset_key(alias) == name


@pytest.mark.parametrize("name", list(P3.PRESETS))
def test_every_preset_passes_its_own_check_on_flat_ground(name):
    pre = P3.PRESETS[name]
    if pre["rest"] == "seat":
        pytest.skip("seated presets need a block (tested below)")
    for yaw in (0, 60, 150, -90):
        res = P3.solve_figure({"preset": name, "at": [500, 900], "yaw": yaw}, _cam(), {}, {})
        assert not [c for c in res["check"] if "bends backward" in c or "into the" in c], (name, yaw, res["check"])


def test_gesture_breaks_the_symmetric_mannequin():
    res = P3.solve_figure({"preset": "stand", "at": [500, 900]}, _cam(), {}, {})
    J = res["J"]
    hips = J["hip-l"][1] - J["hip-r"][1]
    shoulders = J["shoulder-l"][1] - J["shoulder-r"][1]
    assert hips * shoulders < 0                     # contrapposto: shoulder line against hip line
    assert abs(J["knee-l"][2] - J["knee-r"][2]) > 0.02  # one knee relaxed
    flat = P3.solve_figure({"preset": "stand", "at": [500, 900], "gesture": False}, _cam(), {}, {})
    assert abs(flat["J"]["hip-l"][1] - flat["J"]["hip-r"][1]) < 1e-6


def test_hand_anchors_put_the_hands_on_the_body():
    for name, word in (("hands-on-hips", "hip"), ("arms-crossed", "cross"), ("hands-behind", "behind")):
        res = P3.solve_figure({"preset": name, "at": [500, 900]}, _cam(), {}, {})
        assert not res["ik"], (name, res["ik"])
        J = res["J"]
        if word == "hip":
            for s in "rl":
                assert np.linalg.norm(J[f"wrist-{s}"] - J[f"hip-{s}"]) < 0.25
        if word == "behind":
            fwd = res["F"]["pelvis"][:, 2]
            assert np.dot(J["wrist-r"] - J["pelvis"], fwd) < 0
    # an arm descriptor anchors too, and --hand-l-to hip as a target word
    res = P3.solve_figure({"preset": "stand", "at": [500, 900], "mods": {"arm-r": "hip"},
                           "targets": {"hand-l": "chest"}}, _cam(), {}, {})
    assert any("hand-l to the chest" in ln for ln in res["ik"])


def test_modifiers_lean_twist_look_energy_mirror():
    cam = _cam()
    base = P3.solve_figure({"preset": "stand", "at": [500, 900]}, cam, {}, {})
    lean = P3.solve_figure({"preset": "stand", "at": [500, 900], "mods": {"lean": 30}}, cam, {}, {})
    fwd = base["F"]["pelvis"][:, 2]
    assert np.dot(lean["J"]["head"] - base["J"]["head"], fwd) > 0.1
    tw = P3.solve_figure({"preset": "stand", "at": [500, 900], "mods": {"twist": 30}}, cam, {}, {})
    ang = lambda r: np.degrees(np.arctan2(r["F"]["chest"][0, 2], -r["F"]["chest"][2, 2]))  # noqa: E731
    assert abs(ang(tw) - ang(base)) > 10
    # look at the camera from a three-quarter view: the head faces the viewer
    lk = P3.solve_figure({"preset": "stand", "at": [500, 900], "yaw": 50, "mods": {"look": "camera"}}, cam, {}, {})
    hy, _ = P3.view_angles(cam, lk["F"]["head"], lk["J"]["head"])
    assert abs(hy) < 8
    # energy pushes the run further
    calm = P3.apply_mods(P3.PRESETS["run"]["edits"], {"energy": 0.0})[0]
    wild = P3.apply_mods(P3.PRESETS["run"]["edits"], {"energy": 1.0})[0]
    assert wild["leg-r"]["forward"] > P3.PRESETS["run"]["edits"]["leg-r"]["forward"] > calm["leg-r"]["forward"]
    # mirror swaps the anchors too
    m = P3.solve_figure({"preset": "hand-to-face", "at": [500, 900], "mirror": True}, cam, {}, {})
    assert m["J"]["wrist-l"][1] > m["J"]["wrist-r"][1]
    with pytest.raises(UsageError):
        P3.parse_look("sideways")
    assert P3.parse_look("at:300,200") == [300.0, 200.0]


def test_balance_fix_moves_the_hips_over_the_feet():
    cam = _cam()
    res = P3.solve_figure({"preset": "stand", "at": [500, 900], "mods": {"arm-l": "forward", "arm-r": "forward",
                                                                          "lean": 35}}, cam, {}, {})
    assert res["balance"].startswith("balanced"), res["balance"]
    off = P3.solve_figure({"preset": "stand", "at": [500, 900], "balance": False,
                           "mods": {"lean": 35, "arm-l": "forward", "arm-r": "forward"}}, cam, {}, {})
    assert not off["balance"].startswith("balanced")
    assert any(c.startswith("off balance") for c in off["check"])


def test_seated_presets_on_a_bench_and_lean_contact():
    cam = _cam()
    g = cam.ground_point(500, 900)
    bench = P3.Block("bench", "box", g, (1.6, 0.5, 0.45))
    for name in ("sit-chair", "sit-bench", "sit-desk"):
        res = P3.solve_figure({"preset": name, "on": "bench", "yaw": 0}, cam, {"bench": bench}, {})
        assert any(n == "seat" for n, _, _ in res["contacts"]), name
    wall = P3.Block("wall", "box", cam.ground_point(500, 860), (3.0, 0.3, 2.0))
    res = P3.solve_figure({"preset": "lean-wall", "on": "wall", "at": [500, 900], "yaw": 0}, cam, {"wall": wall}, {})
    back = res["J"]["chest"] - res["F"]["chest"][:, 2] * 0.12
    assert np.linalg.norm(wall.closest(back) - back) < 0.03
    assert min(P3.lowest(p)[0] for p in res["prims"]) < 0.01     # standing on the ground, not on the wall


def test_pose_cli_modifiers_and_check(draw):
    _scene(draw)
    ok(draw("pose", "girl", "--preset", "arms-crossed", "--yaw", "20", "--look", "camera", "--twist", "15",
            "--energy", "0.7", "--hand-r", "fist", "--leg-l", "relaxed"))
    rig = next(e for e in state()["draw"]["sketch"] if e["kind"] == "figure")["rig"]
    assert rig["mods"]["look"] == "camera" and rig["hands"]["r"] == "fist" and rig["mods"]["leg-l"] == "relaxed"
    out = ok(draw("pose", "girl", "--check"))
    assert "check girl" in out and len(out.splitlines()) <= 6
    ok(draw("pose", "girl", "--hand-l-to", "hip"))
    rig = next(e for e in state()["draw"]["sketch"] if e["kind"] == "figure")["rig"]
    assert rig["targets"]["hand-l"] == "hip"


@pytest.mark.parametrize("name", list(P3.PRESETS))
def test_every_preset_reads_as_a_silhouette_at_its_default_yaw(name):
    cam = P3.Camera(600, 800, 330, 1.2, 44)
    pre = P3.PRESETS[name]
    blocks, spec = {}, {"preset": name, "at": [500, 900]}
    if pre["rest"] == "seat":
        blocks = {"s": P3.Block("s", "box", cam.ground_point(500, 900) - [0, 0, 0.2], (1.0, 0.45, 0.45))}
        spec = {"preset": name, "on": "s"}
    res = P3.solve_figure(spec, cam, blocks, {})
    assert not P3.silhouette_lines(res, cam), name


def test_moving_presets_keep_four_separate_limbs_round_the_figure():
    cam = P3.Camera(600, 800, 330, 1.2, 44)
    for name in ("run", "sprint", "run-toward", "sprint-side", "walk", "jump", "jump-toward", "leap", "kick",
                 "land", "dance-twirl", "dance-step"):
        for yaw in (0, 35, -35, 90):
            res = P3.solve_figure({"preset": name, "at": [500, 900], "yaw": yaw}, cam, {}, {})
            assert not P3.silhouette_lines(res, cam), (name, yaw)


def test_look_left_right_follow_the_picture_even_in_profile():
    cam = P3.Camera(600, 800, 330, 1.2, 44)
    for yaw in (0, 90, -90, 160):
        for word, sign in (("left", -1), ("right", 1)):
            res = P3.solve_figure({"preset": "stand", "at": [500, 900], "yaw": yaw, "mods": {"look": word}},
                                  cam, {}, {})
            assert sign * res["F"]["head"][0, 2] > 0.5, (yaw, word)


def test_lean_presets_fit_the_block_height():
    cam = P3.Camera(600, 800, 330, 1.2, 44)
    g = cam.ground_point(500, 900)
    for name, kind, size in (("lean-railing", "railing", (2.0, 0.06, 0.95)), ("lean-elbow", "box", (0.7, 0.7, 1.1))):
        blk = P3.Block("w", kind, g + [0.45, 0, 0], size)
        res = P3.solve_figure({"preset": name, "on": "w", "at": [500, 900], "yaw": 0}, cam, {"w": blk}, {})
        assert abs(res["contact_gap"]) < 0.06, (name, res["contact_gap"])


def test_two_figures_hold_hands_and_lean_back_to_back():
    cam = P3.Camera(600, 800, 330, 1.2, 44)
    girl = P3.solve_figure({"preset": "walk-together", "at": [350, 900], "edits": {"arm-l": {"raise": 45}}},
                           cam, {}, {})
    hand = girl["J"]["hand-l"]
    spot = cam.project_units([np.array([hand[0] + 0.3, 0.0, hand[2]])])[0]
    boy = P3.solve_figure({"preset": "stand", "at": [float(spot[0]), float(spot[1])],
                           "targets": {"hand-r": "girl:hand-l"}}, cam, {}, {"girl": girl})
    grip = boy["J"]["wrist-r"] + P3.unit(boy["J"]["hand-r"] - boy["J"]["wrist-r"]) * 0.45 * boy["body"].Lh
    assert np.linalg.norm(grip - hand) < 0.08, boy["ik"]
    assert any("girl's hand-l" in ln for ln in boy["ik"])
    carry = P3.solve_figure({"preset": "stand", "at": [600, 900], "targets": {"hand-l": "girl:back"}},
                            cam, {}, {"girl": girl})
    assert any("girl's back" in ln for ln in carry["ik"])
    bad = P3.solve_figure({"preset": "stand", "at": [600, 900], "targets": {"hand-l": "girl:tail"}},
                          cam, {}, {"girl": girl})
    assert any("has no" in ln for ln in bad["ik"])
