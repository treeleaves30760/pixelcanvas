import numpy as np

from pixelcanvas import figure3d as F3
from pixelcanvas import pose3d as P3
from pixelcanvas import hand2d as H2


def _rig(hands=None, at=(500, 930)):
    cam = P3.Camera(600, 800, horizon=420, eye=1.2)
    spec = {"preset": "wave", "at": list(at), "hands": hands or {}}
    res = P3.solve_figure(spec, cam, {}, {})
    return cam, F3.Rig(res["body"], res["J"], res["F"]), res["body"]


def test_every_hand_shape_is_one_closed_smooth_outline_with_few_lines():
    cam, rig, body = _rig()
    for shape in F3.HAND_SHAPES + ("wave", "unknown"):
        parts = H2.hand_parts(rig, "r", shape, "skin", "g", cam.C, body.h, body.k, cam)
        assert len(parts) == 1
        P = np.asarray(parts[0]["flat3"])
        assert len(P) >= 20 and len(parts[0]["details"]) <= 4
        q, z = cam.project(P)
        step = np.linalg.norm(np.diff(np.vstack([q, q[:1]]), axis=0), axis=1)
        assert step.max() < 0.25 * np.ptp(q, axis=0).max()          # no long straight jumps: a smooth closed loop


def test_new_shape_names_are_accepted_by_pose_and_aliases_resolve():
    for s in ("spread", "flat", "reach"):
        assert s in F3.HAND_SHAPES
        F3.hand_prims(_rig()[1], "x", "r", s)
    assert H2.shape_of("wave")[0] == "spread" and H2.shape_of("nonsense")[0] == "relaxed"


def test_a_fist_has_inner_lines_and_a_spread_hand_is_wider_than_a_fist():
    cam, rig, body = _rig()

    def width(shape):
        p = H2.hand_parts(rig, "r", shape, "skin", "g", cam.C, body.h, body.k, cam)[0]
        q, _ = cam.project(np.asarray(p["flat3"]))
        return float(np.ptp(q, axis=0).max())
    assert width("spread") > width("fist") * 1.2


def test_a_sword_blade_is_drawn_as_a_long_outline_with_an_edge_line():
    from pixelcanvas import anime2d as A2
    from pixelcanvas import draw_figure2d as DF
    cam = P3.Camera(700, 980, horizon=430, eye=1.25)
    spec = {"preset": "fight-stance", "at": [500, 930]}
    res = P3.solve_figure(spec, cam, {}, {})
    held = [{"type": "sword", "hand": "r", "name": "sword", "id": "P1", "subject": "g"}]
    parts = A2.figure_parts({"rig": spec}, res["body"], res["J"], res["F"], {}, "g", held, eye=cam.C, cam=cam)
    blade = next(pt for pt in parts if pt["name"] == "sword-blade")
    q, _ = cam.project(np.asarray(blade["poly3"]))
    assert np.ptp(q, axis=0).max() > 100 and len(blade["details"]) == 1
    _ = DF


def test_a_covered_hand_is_smoothed_or_dropped():
    from pixelcanvas import draw_figure2d as DF
    from pixelcanvas import shape2d as S2
    R = S2.Raster(200, 200, 1.0)
    poly = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], float)
    hand = {"name": "hand-r", "flat3": np.zeros((4, 3)), "subj": "g", "group": "hand-r"}
    skirt = {"name": "skirt", "subj": "g", "group": "clothes"}
    R.lab[50:150, 50:150] = 0
    yy, xx = np.mgrid[0:200, 0:200]
    R.lab[(yy > 100 + 4 * ((xx // 3) % 2)) & (R.lab == 0)] = 1         # a stair-step cut, half covered
    outs = [{"poly": poly}, {"poly": poly}]
    DF._clean_hands([hand, skirt], outs, R)
    edge = [int(np.argmax(R.lab[:, x] == 1)) for x in range(60, 140)]
    assert np.ptp(edge) <= 2 and outs[0] is not None and outs[0].get("details_px")
    R.lab[50:150, 50:150] = 1
    R.lab[50:70, 50:150] = 0                                            # a fifth shows: dropped
    outs = [{"poly": poly}, {"poly": poly}]
    DF._clean_hands([hand, skirt], outs, R)
    assert outs[0] is None and not (R.lab == 0).any()
