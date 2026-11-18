"""The 3D figure carried through line, flat and shade: round-cone limbs, cloth shells, garments bound to the
skeleton, hair locks that hang and trail, hands with finger poses, a head with a face frame, and the toon render
(line, flat and shade layers) that re-renders when the pose changes."""
from __future__ import annotations

import numpy as np

from pixelcanvas import figure3d as F3
from pixelcanvas import pose3d as P3
from pixelcanvas import render3d as R3

from .test_draw_wear import draw, ok, state  # noqa: F401  (draw is a fixture)


# ============================================================================================ primitives

def test_round_cone_and_shell_intersections():
    D = np.array([[0.0, 0.0, 1.0]])
    # a round cone lying across the ray: hit at its near surface (radius 1 at a, 0.5 at b)
    pr = R3.rcone([-2, 0, 5.0], [2, 0, 5.0], 1.0, 0.5)
    t, n, _ = R3.intersect(pr, np.zeros(3), D)
    assert abs(t[0] - (5 - 0.75)) < 0.02 and n[0, 2] < -0.9
    # the a-end cap
    t, n, _ = R3.intersect(pr, np.array([-2.5, 0, 0.0]), D)
    assert np.isfinite(t[0]) and t[0] < 5.0
    # a shell (open skirt): a ray through the side hits it, a ray down the open middle from below hits the inside
    sh = R3.shell([0, 0, 5.0], np.eye(3) * 0.5, flare=1.0)
    t, n, f = R3.intersect(sh, np.array([0, -0.25, 0.0]), D)
    assert np.isfinite(t[0]) and not (f[0] & 1)
    up = np.array([[0.0, 1.0, 0.0]])
    t, n, f = R3.intersect(sh, np.array([0.0, -3, 5.0]), up)
    assert not np.isfinite(t[0])                               # straight up the axis misses the cloth
    t, n, f = R3.intersect(sh, np.array([0.5, -3, 5.0]), up)
    assert np.isfinite(t[0]) and (f[0] & 1)                    # seen from below: the inside of the skirt
    # an ellipsoid with a front wedge cut out shows its inside through the gap
    jacket = R3.ell_prim([0, 0, 5.0], np.diag([1, 1, -1.0]), (1, 1, 1), clip={"wedge": [0.5, 0.5, 1.0, -1.0]})
    t, n, f = R3.intersect(jacket, np.zeros(3), D)
    assert abs(t[0] - 6.0) < 1e-6 and (f[0] & 1)


# ============================================================================================ the dressed figure

def _fig(preset="stand", yaw=0.0, look=None, edits=None, extra=None):
    cam = P3.Camera(400, 500, horizon=420, eye=1.2)
    spec = {"preset": preset, "at": [500, 930], "yaw": yaw}
    if edits:
        spec["edits"] = edits
    if extra:
        spec.update(extra)
    res = P3.solve_figure(spec, cam, {}, {})
    prims = F3.dressed_prims(res["body"], res["J"], res["F"], spec, look or {}, "f")
    return cam, res, prims


def test_a_sleeve_moves_with_the_elbow():
    _, r0, p0 = _fig(look={"garments": {"shirt": {"sleeve": "long"}}})
    _, r1, p1 = _fig(look={"garments": {"shirt": {"sleeve": "long"}}}, edits={"elbow-r": {"bend": 120}})

    def cuff(prims):
        return [pr for pr in prims if pr["piece"] == "shirt-sleeve-r" and pr["t"] == "rcone"][-1]["b"]
    # the sleeve's cuff ends at the wrist, wherever the elbow puts it
    for prims, res in ((p0, r0), (p1, r1)):
        assert np.linalg.norm(cuff(prims) - res["J"]["wrist-r"]) < 0.03
    assert np.linalg.norm(cuff(p0) - cuff(p1)) > 0.15
    # a loose sleeve widens toward the cuff
    _, _, pl = _fig(look={"garments": {"shirt": {"fit": "loose"}}})
    fore = [pr for pr in pl if pr["piece"] == "shirt-sleeve-r" and pr["t"] == "rcone"][-1]
    assert fore["rb"] > fore["ra"]


def test_skirt_flare_follows_the_legs():
    _, _, stand = _fig(look={"garments": {"skirt": {}}})
    _, _, run = _fig("run", look={"garments": {"skirt": {}}})
    _, _, split = _fig(look={"garments": {"skirt": {}}}, edits={"leg-r": {"raise": 40}, "leg-l": {"raise": 40}})
    fs, fr, fp = F3.skirt_flare(stand), F3.skirt_flare(run), F3.skirt_flare(split)
    assert fp > fs + 0.2 and fr > fs + 0.2
    # a raised knee lifts the hem on its side instead of the skirt ballooning
    sk = next(pr for pr in run if pr.get("skirt_flare"))
    assert np.hypot(*sk["hem_tilt"]) > 0.1


def _tips(prims):
    locks = {}
    for pr in prims:
        if pr["piece"].startswith("lock") and pr["t"] == "rcone":
            locks[pr["piece"]] = pr["b"]
    return np.array(list(locks.values()))


def test_hair_hangs_when_standing_and_trails_when_running():
    _, rs, ps = _fig(look={"hair": {"style": "long"}})
    t = _tips(ps)
    head = rs["J"]["head"]
    assert (t[:, 1] < head[1] - 0.3).mean() > 0.6            # the long locks fall well below the head
    # running toward the viewer (yaw 0 faces the camera: forward = -z): the locks stream back (+z)
    _, rr, pr_ = _fig("run", look={"hair": {"style": "long"}})
    tr = _tips(pr_)
    assert np.median(tr[:, 2] - rr["J"]["head"][2]) > np.median(t[:, 2] - head[2]) + 0.1
    # locks stay outside the head
    rig = F3.Rig(rr["body"], rr["J"], rr["F"])
    for p_ in tr:
        assert np.linalg.norm(np.linalg.solve(rig.cr_A, p_ - rig.cr_c)) > 1.0


def test_hand_shapes_curl_the_fingers():
    def reach(shape):
        _, res, prims = _fig(extra={"hands": {"r": shape}})
        tips = np.array([pr["b"] for pr in prims if pr["piece"].startswith("finger-r") and pr["t"] == "rcone"])
        tips = tips.reshape(4, 3, 3)[:, -1, :]
        return float(np.linalg.norm(tips - res["J"]["wrist-r"], axis=1).mean())
    assert reach("fist") < 0.75 * reach("open")
    # presets pick hands: a run makes fists
    assert F3.PRESET_HANDS["run"] == ("fist", "fist")


def test_an_arm_in_front_hides_the_blazer():
    cam, res, prims = _fig(look={"garments": {"blazer": {}}}, edits={"arm-r": {"forward": 80, "raise": 15},
                                                                     "elbow-r": {"bend": 100}})
    R = R3.render(cam, prims)
    blz = [k for k, pr in enumerate(prims) if pr["mat"] == "blazer" and not pr["piece"].startswith("blazer-sleeve")]
    full = sum(int(R["own"][k][1].sum()) for k in blz if R["own"][k])
    seen = int(np.isin(R["id"], blz).sum())
    assert 0 < seen < 0.93 * full                            # the forearm across the chest covers part of it


def test_the_face_turns_with_the_head():
    from pixelcanvas import face3d as FC

    def eyes(yaw):
        cam, res, prims = _fig(yaw=yaw)
        rig = F3.Rig(res["body"], res["J"], res["F"])
        sp = FC.project(rig, cam, FC.features("neutral"))
        e = {s["name"]: s["px"].mean(0) for s in sp if s["name"] in ("white-r", "white-l")}
        return e["white-r"], e["white-l"]
    r0, l0 = eyes(0)
    r1, l1 = eyes(40)
    assert r0[0] < l0[0]                                     # facing us, her right eye is on the picture's left
    assert abs(r1[0] - l1[0]) < abs(r0[0] - l0[0])           # turned, the eyes come closer together
    crying = {s["name"] for s in FC.features("crying")}
    assert {"tear-r", "tear-l", "mouth", "brow-r"} <= crying and "iris-r" not in crying


# ============================================================================================ the commands

REQ = "a black-haired boy runs toward the viewer crying, a pink-haired girl floats behind him in a school corridor"


def _scene(run):
    ok(run("new", "w", "--size", "300x400", "--request", REQ))
    ok(run("style", "cel"))
    ok(run("do", "plan subject boy --box 300,300,800,990; plan subject girl --box 80,150,450,700; "
                 "plan part hair --box 330,300,600,420 --of boy; plan part face --box 380,330,560,430 --of boy; "
                 "plan light top-left; palette make; palette pick 1; stage done"))
    ok(run("camera", "--horizon", "430", "--eye-height", "1.0"))
    ok(run("block", "floor", "plane", "--at", "500,800", "--size", "6,12,0"))
    ok(run("pose", "boy", "--preset", "run", "--at", "520,960", "--yaw", "15"))
    ok(run("pose", "girl", "--preset", "stand", "--at", "250,640", "--lift", "0.6"))


def _alpha(name):
    from PIL import Image
    return np.asarray(Image.open(f"w.pixelcanvas/layers/{name}.png").convert("RGBA"))


def test_render_writes_line_flat_and_shade_and_follows_a_re_pose(draw):
    _scene(draw)
    out = ok(draw("dress", "boy", "blazer,shorts,socks,shoes", "--color", "navy"))
    assert "boy now wears: blazer, shorts, socks, shoes" in out
    ok(draw("dress", "boy", "blazer", "--trim", "gold"))
    ok(draw("hair3d", "boy", "--style", "spiky", "--color", "black"))
    ok(draw("dress", "girl", "blazer,skirt", "--color", "navy"))
    ok(draw("hair3d", "girl", "--style", "twintails", "--color", "pink", "--bows", "black"))
    code, out = draw("dress", "boy", "cape")
    assert code != 0 and "EXAMPLE" in out
    out = ok(draw("render"))
    assert "lines" in out and "flat colours" in out and "regions:" in out
    for n in ("line", "flat", "shade"):
        assert (_alpha(n)[..., 3] > 0).mean() > (0.005 if n != "flat" else 0.05), n
    st = state()
    regs = st["regions"]
    assert "hair" in regs and any(r.startswith("girl-") for r in regs) and "floor" in regs
    assert set(st["draw"]["render"]["subjects"]) == {"boy", "girl"}
    # the face on the 3D head, drawn as line records owned by the render
    out = ok(draw("face", "boy", "--expression", "crying"))
    assert "rendered again" in out
    recs = [r for r in state()["draw"]["lines"] if r.get("face3d") == "boy"]
    assert any("tear" in r["src"] for r in recs) and any("mouth" in r["src"] for r in recs)
    # a re-pose renders again: the line art follows
    line0 = _alpha("line")[..., 3]
    out = ok(draw("pose", "boy", "--set", "arm-r: forward 80; elbow-r: 20", "--hand-shape-r", "open"))
    assert "rendered again" in out
    line1 = _alpha("line")[..., 3]
    assert (np.abs(line1.astype(int) - line0.astype(int)) > 60).mean() > 0.001
    fig = next(el for el in state()["draw"]["sketch"] if el.get("subject") == "boy")
    assert fig["rig"]["hands"] == {"r": "open"}
    # later tools work on the rendered areas
    ok(draw("shade", "hair"))
    ok(draw("undo"))


def test_the_face_frame_is_in_the_figure_record_for_the_face_tools(draw):
    _scene(draw)
    ok(draw("pose", "boy", "--preset", "stand", "--yaw", "0"))
    fig = next(el for el in state()["draw"]["sketch"] if el.get("subject") == "boy")
    ff = fig["face2d"]
    assert ff["eye-r"][0] < ff["eye-l"][0] and ff["eye-r"][1] < ff["mouth"][1] < ff["chin"][1]
    # a 2D head construction on the posed figure places features on the 3D face frame
    ok(draw("sketch", "head", "boy"))
    from pixelcanvas import draw_sketch
    from pixelcanvas.project import Project
    p = Project("w.pixelcanvas")
    head = next(el for el in p.state["draw"]["sketch"] if el["kind"] == "head")
    g = draw_sketch.head_guides(p, head)
    assert np.allclose(g["left eye"], ff["eye-r"]) and np.allclose(g["mouth"], ff["mouth"])


def test_a_band_behind_survives_a_re_render(draw):
    _scene(draw)
    ok(draw("render"))
    ok(draw("band", "sky", "0,0 1000,0 1000,430 0,430", "--closed", "--color", "#a0c8f0"))
    from PIL import Image

    def reg(n):
        return np.asarray(Image.open(f"w.pixelcanvas/regions/{n}.png")) > 127
    girl0 = reg("girl-skin")
    boy = np.zeros_like(girl0)
    for n in state()["regions"]:
        if n.startswith("boy") or n in ("hair",):
            boy |= reg(n)
    ok(draw("pose", "girl", "--at", "700,640"))
    flat = _alpha("flat")
    H = flat.shape[0]
    # where the girl floated before (and nothing else stands now), the sky shows: not a hole, not her old colours
    was = girl0 & ~boy & ~reg("girl-skin")
    was[int(0.43 * H):] = False
    px = flat[was]
    assert was.sum() > 50
    assert (px[:, 3] > 200).mean() > 0.95 and (np.abs(px[:, 2].astype(int) - 0xf0) < 30).mean() > 0.9
