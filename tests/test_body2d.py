"""The 2D body over the 3D pose (anime2d): clothes close over the body, limbs join the trunk with clear lines."""
import numpy as np

from pixelcanvas import anime2d as A2
from pixelcanvas import pose3d as P3


def _parts(look, preset="stand", subj="boy"):
    cam = P3.Camera(400, 500, horizon=420, eye=1.2)
    spec = {"preset": preset, "at": [500, 930], "yaw": 20}
    res = P3.solve_figure(spec, cam, {}, {})
    parts = A2.figure_parts({"rig": spec}, res["body"], res["J"], res["F"], look, subj, eye=cam.C, cam=cam)
    return {pt["name"]: pt for pt in parts}, res


def test_a_jacket_over_nothing_closes_over_a_white_shirt_with_uniform_details():
    parts, _ = _parts({"garments": {"jacket": {}, "pants": {}, "shoes": {}}})
    assert parts["jacket-v"]["mat"] == "collar"                       # a shirt, never a bare chest
    assert "jacket-shirt-collar-r" in parts and "jacket-shirt-collar-l" in parts
    assert any(n.startswith("jacket-button") for n in parts)           # closed with buttons
    assert len(parts["jacket"]["details"]) >= 7                       # lapels, notches, pockets, front edge
    assert all("nrm" in d for d in parts["jacket"]["details"])        # drawn on the front only
    # sleeves: a shoulder seam and a cuff band; trousers: waistband and fly
    assert len(parts["jacket-sleeve-r"]["details"]) >= 1
    assert len(parts["jacket-forearm-r"]["details"]) >= 1
    assert len(parts["pants"]["details"]) == 2
    # a shoe is one closed solid with a sole line
    assert parts["foot-r"]["kind"] == "blob" and parts["foot-r"]["prims"] and parts["foot-r"]["details"]


def test_the_thigh_continues_the_hip_line_and_the_arm_starts_below_the_shoulder():
    parts, res = _parts({})
    h, k = res["body"].h, res["body"].k
    tpl = A2.template(res["body"].heads, True)
    J = A2.fit_joints(res["J"], res["F"], tpl, h, k)
    X = res["F"]["pelvis"][:, 0]
    top = parts["thigh-r"]["prims"][0]
    x_out = abs(float((np.asarray(top["a"]) - J["pelvis"]) @ X)) + top["ra"]
    assert x_out <= tpl["hip"] * h * 1.08                              # no hip bulge
    Y = res["F"]["chest"][:, 1]
    a = parts["upperarm-r"]["prims"][0]
    assert float((np.asarray(a["a"]) - J["shoulder-r"]) @ Y) < -0.1 * h


def test_band_faces_the_viewer_and_inner_lines_on_dark_parts_are_lighter():
    pts = A2._band([0, 0, 0], [0, 1, 0], 1.0, eye=np.array([0, 0, 10.0]))["pts"]
    assert (pts[:, 2] > 0).all()                                       # the half ring on the viewer's side
    from pixelcanvas import draw_figure2d as DF
    orig = DF.colour_of
    try:
        DF.colour_of = lambda p, pt, cache: (30, 30, 36)
        rgb = DF._inner_rgb(None, {"mat": "thigh-highs"}, {})
        assert rgb is not None and min(rgb) > 60
        DF.colour_of = lambda p, pt, cache: (240, 240, 240)
        assert DF._inner_rgb(None, {"mat": "blouse"}, {}) is None
    finally:
        DF.colour_of = orig


def test_a_pleated_skirt_hem_has_a_pleat_rhythm():
    parts, _ = _parts({"garments": {"blouse": {}, "skirt": {}}}, subj="girl")
    sk = parts["skirt"]
    hem = sk["rings"][-1][1]
    c = sk["rings"][-1][0]
    r = np.linalg.norm(hem - c, axis=1)
    assert len(hem) >= 96 and r.std() / r.mean() > 0.01               # fold edges stand out of the hem
