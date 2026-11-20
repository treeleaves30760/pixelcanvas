"""The 3D pose model and the 2D drawing over it: round-cone limbs, cloth shells, garments bound to the skeleton,
hair locks that hang and trail, hands with finger poses, a head with a face frame; the guide layer (never exported)
and ``draw``: anime contours, clothes, hair and face as line records, flat colours on the drawn shapes, cel shadows."""
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


def _dressed(run):
    _scene(run)
    out = ok(run("dress", "boy", "blazer,shorts,socks,shoes", "--color", "navy"))
    assert "boy now wears: blazer, shorts, socks, shoes" in out
    ok(run("dress", "boy", "blazer", "--trim", "gold"))
    ok(run("hair3d", "boy", "--style", "spiky", "--color", "black"))
    ok(run("dress", "girl", "blouse,skirt", "--color", "navy"))
    ok(run("hair3d", "girl", "--style", "twintails", "--color", "pink", "--bows", "black"))


def test_render_is_a_guide_that_is_never_exported(draw):
    _dressed(draw)
    code, out = draw("dress", "boy", "cape")
    assert code != 0 and "EXAMPLE" in out
    out = ok(draw("render"))
    assert "the 3D guide" in out and "pixelcanvas draw all" in out
    assert (_alpha("guide")[..., 3] > 0).mean() > 0.05
    for n in ("line", "flat", "shade"):
        assert not __import__("os").path.exists(f"w.pixelcanvas/layers/{n}.png") or \
            (_alpha(n)[..., 3] > 0).mean() == 0, n
    from pixelcanvas.project import Project
    p = Project("w.pixelcanvas")
    view_ = p.composite_u8("view").astype(int)
    score = p.composite_u8("score").astype(int)
    assert np.abs(view_ - score).max() > 30           # the views show it faded ...
    assert score.std() < 2                             # ... the picture does not hold it
    assert "guide" not in [r.name for r in __import__("pixelcanvas.export", fromlist=["x"]).layer_records(p)
                           if r.visible]


def test_draw_makes_line_records_parts_and_areas_over_the_guide(draw):
    _dressed(draw)
    ok(draw("render"))
    out = ok(draw("draw", "all"))
    assert "line records over the 3D guide" in out and "boy hair:" in out and "girl face:" in out
    st = state()
    recs = [r for r in st["draw"]["lines"] if r.get("d2d")]
    assert {r["d2d"] for r in recs} >= {"boy", "girl", "@floor"}
    # few control points per stroke, smoothed by the line engine; heavier on the silhouette
    assert np.median([len(r["ctrl"]) for r in recs]) <= 12
    assert len({r["weight"] for r in recs}) >= 3
    regs = st["regions"]
    assert {"hair", "boy-blazer", "girl-skirt", "floor"} <= set(regs)
    assert regs["boy-blazer"]["how"].startswith("2D over the 3D guide")
    assert (_alpha("line")[..., 3] > 0).mean() > 0.005
    # one part again: only its strokes change
    hair0 = {r["id"] for r in recs if r["d2d"] == "boy" and r["d2d_group"] == "hair"}
    other0 = {r["id"] for r in recs if r["d2d"] == "boy" and r["d2d_group"] != "hair"}
    ok(draw("draw", "boy", "--part", "hair"))
    recs1 = [r for r in state()["draw"]["lines"] if r.get("d2d") == "boy"]
    assert not hair0 & {r["id"] for r in recs1 if r["d2d_group"] == "hair"}
    assert other0 <= {r["id"] for r in recs1}
    code, out = draw("draw", "boy", "--part", "wings")
    assert code != 0 and "EXAMPLE" in out
    # colours: flat auto fills each drawn part on its own shape; shade lays cel shapes
    ok(draw("flat", "auto"))
    from pixelcanvas.project import Project
    p = Project("w.pixelcanvas")
    flat = _alpha("flat")
    for rn in ("boy-blazer", "hair", "girl-skirt"):
        m = p.region_mask(rn)
        rgb = np.array(__import__("pixelcanvas.draw_core", fromlist=["x"]).from_hex(
            st["draw"]["palette"]["swatches"][rn]))
        assert (np.abs(flat[m][:, :3].astype(int) - rgb).max(1) < 12).mean() > 0.9, rn
    out = ok(draw("shade", "all"))
    assert "cel shapes" in out
    assert (_alpha("shade")[..., 3] > 0).mean() > 0.01
    assert set(state()["draw"]["shaded2d"]) >= {"boy", "girl"}


def test_face_and_re_pose_on_a_drawn_figure(draw):
    _dressed(draw)
    ok(draw("render"))
    ok(draw("draw", "all"))
    out = ok(draw("face", "boy", "--expression", "crying"))
    assert "the face drawn again" in out
    recs = [r for r in state()["draw"]["lines"] if r.get("d2d") == "boy" and r["d2d_group"] == "face"]
    assert any("tear" in r["src"] for r in recs) and any("mouth" in r["src"] for r in recs)
    assert all(r["auto_rgb"] == [84, 140, 196] for r in recs if "tear" in r["src"] and r.get("rgb_fixed"))
    # a re-pose updates the guide; the drawing waits for draw (NEXT says so), then follows
    g0 = _alpha("guide")[..., 3]
    out = ok(draw("pose", "boy", "--set", "arm-r: forward 80; elbow-r: 20", "--hand-shape-r", "open"))
    assert "rendered again" in out or "the 3D guide" in out
    g1 = _alpha("guide")[..., 3]
    assert (np.abs(g1.astype(int) - g0.astype(int)) > 60).mean() > 0.001
    from pixelcanvas.draw_figure2d import stale_subjects
    from pixelcanvas.project import Project
    assert stale_subjects(Project("w.pixelcanvas")) == ["boy"]
    line0 = _alpha("line")[..., 3]
    ok(draw("draw", "boy"))
    line1 = _alpha("line")[..., 3]
    assert (np.abs(line1.astype(int) - line0.astype(int)) > 60).mean() > 0.001
    fig = next(el for el in state()["draw"]["sketch"] if el.get("subject") == "boy")
    assert fig["rig"]["hands"] == {"r": "open"}


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


def test_surface_and_polish_leave_a_cel_drawing_clean(draw):
    _scene(draw)
    ok(draw("dress", "boy", "blazer,shorts", "--color", "navy"))
    ok(draw("render"))
    ok(draw("face", "boy", "--expression", "crying"))
    ok(draw("draw", "all"))
    ok(draw("flat", "auto"))
    ok(draw("shade", "all"))
    from pixelcanvas.project import Project
    from pixelcanvas import draw_render
    p = Project("w.pixelcanvas")
    rm = draw_render.render_mask(p)
    assert rm.mean() > 0.05
    # the cel shading covers the drawn figure's shade, cast and face items
    from pixelcanvas import workflow_draw
    q = Project("w.pixelcanvas")
    for stg in ("shade", "detail"):
        q.state["stage"] = stg
        workflow_draw.refresh(q)                    # in memory only: the project on disk is unchanged
    for t in q.state["todo"]:
        if (t["stage"] == "shade" and str(t.get("region", "")).startswith("boy-")) or t.get("key") in ("cast", "cel") \
                or (t["stage"] == "detail" and t.get("key") == "detail" and t.get("region") == "face"):
            assert t["status"] == "done", t
    ok(draw("surface", "auto"))
    st = state()
    from pixelcanvas.draw_surface import surface_targets
    assert all(n in st["draw"]["surface"] for n in surface_targets(Project("w.pixelcanvas")))
    surf = _alpha("surface")[..., 3] if __import__("os").path.exists("w.pixelcanvas/layers/surface.png") else None
    assert surf is None or (surf[rm] > 8).mean() < 0.002            # no texture on the drawn areas
    before = Project("w.pixelcanvas").composite_u8("export").astype(int)
    tears = [r for r in st["draw"]["lines"] if r.get("d2d") and "tear" in r.get("src", "") and r.get("rgb_fixed")]
    assert tears and all(r["auto_rgb"] == [84, 140, 196] for r in tears)   # tears stay blue
    ok(draw("polish"))
    after = Project("w.pixelcanvas").composite_u8("export").astype(int)
    line = _alpha("line")[..., 3] > 128
    # the drawing keeps its colours: a light grade, no muddied lines
    assert np.abs(after - before)[rm].mean() < 12
    assert np.abs(after - before)[rm & line].mean() < 12


def test_export_at_blockout_shows_the_3d_blockout(draw):
    _scene(draw)
    ok(draw("export", "blockout.png"))
    from PIL import Image
    img = np.asarray(Image.open("blockout.png").convert("RGB")).astype(int)
    assert img.std() > 8 and (img.sum(-1) < 600).mean() > 0.01        # the figures and the floor show
    ok(draw("render"))
    ok(draw("export", "guide.png"))
    img = np.asarray(Image.open("guide.png").convert("RGB")).astype(int)
    assert img.std() > 8                                                # the guide at a blockout


# ============================================================================================ the 2D shapes

def test_anime_proportions_slimmer_than_the_3d_body():
    from pixelcanvas import anime2d as A2
    cam = P3.Camera(400, 500, horizon=420, eye=1.2)
    spec = {"preset": "stand", "at": [500, 930], "yaw": 0}
    res = P3.solve_figure(spec, cam, {}, {})
    el = {"rig": spec}
    parts = {pt["name"]: pt for pt in A2.figure_parts(el, res["body"], res["J"], res["F"], {}, "g")}
    up, fore = parts["upperarm-r"]["prims"], parts["forearm-r"]["prims"]
    assert fore[-1]["rb"] < 0.65 * up[0]["ra"]                     # the arm tapers to a thin wrist
    shin = parts["shin-r"]["prims"]
    assert shin[-1]["rb"] < 0.6 * parts["thigh-r"]["prims"][0]["ra"]
    tor = A2.Torso(res["body"], res["J"], res["F"], res["body"].h, res["body"].k)
    assert tor.at(3)[3] < 0.8 * tor.at(4)[3] and tor.at(3)[3] < 0.8 * tor.at(1)[3]   # a waist
    rig = F3.Rig(res["body"], res["J"], res["F"])
    assert up[0]["ra"] <= rig.r_up * 1.05 and fore[-1]["rb"] < rig.r_wr


def test_z_buffer_decides_overlaps_and_outlines_split_at_them():
    from pixelcanvas import shape2d as S2
    from pixelcanvas.render3d import rcone
    cam = P3.Camera(400, 400, horizon=500, eye=0.0)
    back = {"rings": S2.chain_tube([[-0.6, 0, 4.0], [0.6, 0, 4.0]], [0.15, 0.15]), "kind": "tube"}
    front = {"rings": S2.chain_tube([[0, -0.6, 3.0], [0, 0.6, 3.0]], [0.1, 0.1]), "kind": "tube"}
    R = S2.Raster(400, 400, 1.0)
    ob, of = S2.tube_outline(cam, back), S2.tube_outline(cam, front)
    R.add_hits(0, R.hits(cam, [rcone([-0.6, 0, 4.0], [0.6, 0, 4.0], 0.15, 0.15)], (0, 0, 400, 400)))
    R.add_hits(1, R.hits(cam, [rcone([0, -0.6, 3.0], [0, 0.6, 3.0], 0.1, 0.1)], (0, 0, 400, 400)))
    assert R.at(np.array([[200.0, 200.0]]))[0] == 1             # the near bar is in front where they cross
    st = S2.outline_strokes(ob["poly"], 0, R, lambda q, P, a, b: 1 if q < 0 else 2, 1.0, 5.0)
    assert len(st) >= 2 and not any(s["closed"] for s in st)   # the far bar's outline breaks where the near one is
    st = S2.outline_strokes(of["poly"], 1, R, lambda q, P, a, b: 1 if q < 0 else 2, 1.0, 5.0)
    n = len(S2.resample(of["poly"], 1.0, closed=True))
    assert sum(len(x["pts"]) for x in st) > 0.95 * n            # the near bar's outline is drawn all round
    assert {x["cls"] for x in st} == {1, 2}                    # silhouette, and the line over the far bar


def test_anime_template_by_heads_and_sex_moves_the_limbs():
    from pixelcanvas import anime2d as A2
    girl, boy = A2.template(7.0, False), A2.template(7.0, True)
    child = A2.template(5.0, False)
    assert boy["shoulder"] > girl["shoulder"] and girl["hip"] > boy["hip"] and girl["waist"] < child["waist"]
    assert A2.is_male("boy") and not A2.is_male("girl")
    cam = P3.Camera(400, 500, horizon=420, eye=1.2)
    res = P3.solve_figure({"preset": "stand", "at": [500, 930]}, cam, {}, {})
    h = res["body"].h
    J2 = A2.fit_joints(res["J"], res["F"], boy, h, res["body"].k)
    X = res["F"]["chest"][:, 0]
    half = abs(float((J2["shoulder-r"] - J2["shoulder-l"]) @ X)) / 2
    assert abs(half - (boy["shoulder"] * h - A2.UPPER[0][1] * h * res["body"].k)) < 1e-6
    for n in ("hip-r", "knee-l", "foot-r", "pelvis", "neck"):
        assert np.allclose(J2[n], res["J"][n])                         # legs and trunk stay as posed


def test_a_raised_knee_keeps_its_foreshortening_and_overlaps_the_shin():
    from pixelcanvas import anime2d as A2
    from pixelcanvas import draw_figure2d as DF
    cam = P3.Camera(400, 500, horizon=420, eye=1.0)
    spec = {"preset": "run", "at": [500, 930], "yaw": 15}
    res = P3.solve_figure(spec, cam, {}, {})
    parts = {pt["name"]: pt for pt in A2.figure_parts({"rig": spec}, res["body"], res["J"], res["F"], {}, "boy",
                                                      eye=cam.C, cam=cam)}
    thigh, shin = parts["thigh-r"], parts["shin-r"]
    bend = DF._bend(thigh, shin, {"knee-r"})
    assert bend > 70                                                   # the run's raised knee
    # the thigh on the canvas is shorter than the shin: it points at the viewer
    q, _ = cam.project(np.array([res["J"]["hip-r"], res["J"]["knee-r"], res["J"]["ankle-r"]]))
    assert np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[2] - q[1])
    sc = {"parts": [thigh, shin], "outs": [{"rpx": 10, "j2d": {"knee-r": np.zeros(2)}},
                                           {"rpx": 10, "j2d": {"knee-r": np.zeros(2)}}]}
    keep = DF._keep_fn(sc, 0, None)
    assert keep(1, (0.0, 0.0), 1.0, 1.05) == 2                         # the knee's line over the shin is drawn
    straight = DF._keep_fn({"parts": [parts["upperarm-r"], parts["forearm-r"]], "outs": sc["outs"]}, 0, None)
    if DF._bend(parts["upperarm-r"], parts["forearm-r"], {"elbow-r"}) < 70:
        assert straight(1, (0.0, 0.0), 1.0, 1.0) == 0


def test_a_small_hand_is_a_mitten_and_the_floor_shadow_is_compact():
    from pixelcanvas import anime2d as A2
    spec = {"preset": "stand", "at": [500, 930], "hands": {"r": "open", "l": "open"}}
    big = P3.Camera(400, 500, horizon=420, eye=1.2)
    res = P3.solve_figure(spec, big, {}, {})
    rig = F3.Rig(res["body"], res["J"], res["F"])
    near = A2._hand2d(rig, "r", "open", "skin", "g", big.C, res["body"].h, res["body"].k, big)[0]
    res2 = P3.solve_figure(dict(spec, at=[500, 470]), big, {}, {})          # far down the floor: small
    rig2 = F3.Rig(res2["body"], res2["J"], res2["F"])
    far = A2._hand2d(rig2, "r", "open", "skin", "g", big.C, res2["body"].h, res2["body"].k, big)[0]
    assert not near["mitten"] and 1 <= len(near["details"]) <= 4
    assert far["mitten"] and len(far["details"]) <= 1


def test_hands_hair_clumps_and_umbrella_are_2d_shapes():
    from pixelcanvas import anime2d as A2
    cam = P3.Camera(400, 500, horizon=420, eye=1.2)
    spec = {"preset": "umbrella", "at": [500, 930], "hands": {"r": "fist", "l": "open"}}
    res = P3.solve_figure(spec, cam, {}, {})
    held = [{"type": "umbrella", "hand": "r", "name": "umbrella", "id": "P1", "subject": "g"}]
    look = {"hair": {"style": "long"}, "garments": {}}
    parts = A2.figure_parts({"rig": spec}, res["body"], res["J"], res["F"], look, "g", held, eye=cam.C)
    by = {pt["name"]: pt for pt in parts}
    assert by["hand-r"]["flat3"] is not None and (len(by["hand-r"]["details"]) +
                                                    len(by.get("fingers-r", {"details": []})["details"])) >= 1
    clumps = [pt for pt in parts if pt["name"].startswith("clump")]
    assert 6 <= len(clumps) <= 14 and all(2 <= len(c["details"]) <= 4 for c in clumps if len(c["details"]))
    can = next(pt for pt in parts if pt.get("canopy") is not None)
    from pixelcanvas import draw_figure2d as DF
    o = DF._canopy_outline(cam, can)
    assert o is not None and len(o["details_px"]) >= 3                 # ribs that face the viewer


def test_pose_show_joints_is_a_look_not_a_step(draw):
    _scene(draw)
    import os
    before = open("w.pixelcanvas/state.json", encoding="utf-8").read()
    hist = sorted(os.listdir("w.pixelcanvas/history")) if os.path.isdir("w.pixelcanvas/history") else []
    sketch = _alpha("sketch").copy()
    out = ok(draw("pose", "boy", "--show", "joints"))
    assert "nothing changed" in out and "VIEW" in out
    assert open("w.pixelcanvas/state.json", encoding="utf-8").read() == before
    assert (sorted(os.listdir("w.pixelcanvas/history")) if os.path.isdir("w.pixelcanvas/history") else []) == hist
    assert np.array_equal(_alpha("sketch"), sketch)


def test_floor_shadow_is_compact_and_blur_leaves_no_halo(draw):
    _scene(draw)
    ok(draw("block", "floor", "plane", "--at", "500,990", "--size", "8,6,0"))
    ok(draw("dress", "boy", "blazer,shorts", "--color", "navy"))
    ok(draw("render"))
    ok(draw("draw", "all"))
    ok(draw("flat", "auto"))
    ok(draw("shade", "all"))
    from pixelcanvas.project import Project
    from pixelcanvas import draw_pose as DP, draw_core as dc
    p = Project("w.pixelcanvas")
    boy = next(el for el in DP.figures(p) if el["subject"] == "boy")
    sh = dc.u2px(p, [boy["j2d"]["shoulder-r"], boy["j2d"]["shoulder-l"]])
    shoulder_px = float(np.linalg.norm(sh[0] - sh[1]))
    from pixelcanvas import draw_figure2d as DF
    mp = DF.load_maps(p)
    fills = DF._figure_shadows(p, mp, ["boy"])
    assert 1 <= len(fills) <= 2                                           # a soft shadow and its darker core
    from pixelcanvas.ops import rle_to_mask
    m = rle_to_mask(fills[0].mask_rle)
    xs = np.nonzero(m.any(0))[0]
    assert 0.8 * shoulder_px < np.ptp(xs) < 2.2 * shoulder_px             # about as wide as the shoulders
    before = p.composite_u8("export").astype(int)
    ok(draw("fx", "blur", "boy", "--amount", "0.6"))
    after = Project("w.pixelcanvas").composite_u8("export").astype(int)
    line = _alpha("line")[..., 3] > 64
    import cv2
    ids = [k for k, t in enumerate(mp["table"]) if t["subj"] == "boy"]
    region = cv2.resize(np.isin(mp["lab"], ids).astype(np.uint8), (p.W, p.H), interpolation=cv2.INTER_NEAREST)
    region = region > 0                                                  # the boy's drawn areas
    ring = (cv2.dilate(region.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0) & ~region & ~line
    boy_rgb = before[region & ~line].mean(0)
    toward = np.linalg.norm(before - boy_rgb, axis=-1) - np.linalg.norm(after - boy_rgb, axis=-1)
    diff = np.abs(after - before).max(-1)
    assert diff[line].max() < 4                                          # the line art stays sharp
    assert np.percentile(toward[ring], 99) < 6                           # no halo of his colours beside him
    assert diff[region].max() > 0                                        # the blur did run inside
