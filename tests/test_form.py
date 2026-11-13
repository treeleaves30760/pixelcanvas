# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.form (object-form kit). Fast: small canvases, a few seconds in total."""
import os
import re
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, sdf, luma  # noqa: E402
from pixelcanvas.form import (  # noqa: E402
    P, px, Frame, d_circle, d_ellipse, d_box, d_segment, d_polygon, Light, set_light, get_light, MATERIALS,
    shade, paint_form, normals, cast_shadow, contact_shadow, drop_shadow, occlusion_between, reflection,
    Ground, Room,
    Surface, tabletop, shelf, counter, Solid, ball, cuboid, cylinder, cone, capsule, torus, lathe, row,
    grid_positions, scatter_positions, ring_positions, disc, book, Spot, resolve_light, shadow_color,
    multiply_shadow, contact_shadow_px, clip_box, full_mask, bbox_of)
from pixelcanvas import form as _form  # noqa: E402


@pytest.fixture(autouse=True)
def _default_light():
    set_light(Light())
    yield
    set_light(Light())


def test_fraction_helpers():
    cv = Canvas(200, 100)
    assert P(cv, 0.5, 0.5) == (100.0, 50.0)
    assert px(cv, 0.1) == pytest.approx(10.0)
    d = d_circle(cv, (0.5, 0.5), 0.2)
    assert d.shape == (100, 200) and d[50, 100] == pytest.approx(-20, abs=0.6)
    assert d_ellipse(cv, (0.5, 0.5), 0.3, 0.1)[50, 100] < 0
    assert d_box(cv, (0.5, 0.5), (0.2, 0.1), 0.02)[50, 100] < 0 < d_box(cv, (0.5, 0.5), (0.2, 0.1))[0, 0]
    assert d_segment(cv, (0.2, 0.5), (0.8, 0.5), 0.05)[50, 100] < 0
    assert d_polygon(cv, [(0.4, 0.3), (0.6, 0.3), (0.5, 0.7)])[50, 100] < 0


def test_frame_roundtrip_rotation_flip_and_child():
    cv = Canvas(320, 180)
    fr = Frame(cv, (0.5, 0.5), 0.2, rotate=30, flip=True, squash=(1.0, 0.6))
    x, y = fr.to_canvas(1.0, 0.5)
    assert np.allclose(fr.from_canvas(x, y), (1.0, 0.5), atol=1e-5)
    plain = Frame(cv, (0.5, 0.5), 0.25, rotate=90)
    x, y = plain.to_canvas(1.0, 0.0)                               # +x rotated 90° clockwise → down
    assert x == pytest.approx(160, abs=1e-6) and y == pytest.approx(90 + 45, abs=1e-6)
    ch = plain.child(at=(1.0, 0.0), size=0.5, rotate=-90)
    assert ch.rotate == pytest.approx(0) and ch.s == pytest.approx(plain.s * 0.5)
    assert np.allclose(ch.to_canvas(0, 0), plain.to_canvas(1, 0))
    twin_a = plain.child(at=(0.5, 0.2), rotate=30)
    twin_b = plain.child(at=(-0.5, 0.2), rotate=30, flip=True)    # mirror twin about the parent's own y axis
    la = plain.from_canvas(*twin_a.to_canvas(1.0, 0.3))
    lb = plain.from_canvas(*twin_b.to_canvas(1.0, 0.3))
    assert la[0] == pytest.approx(-lb[0], abs=1e-4) and la[1] == pytest.approx(lb[1], abs=1e-4)
    mirrored = Frame(cv, (0.5, 0.5), 0.2, rotate=30, flip=True)
    normal = Frame(cv, (0.5, 0.5), 0.2, rotate=30)
    assert mirrored.to_canvas(1, 0)[0] - 160 == pytest.approx(-(normal.to_canvas(1, 0)[0] - 160))
    flat = Frame(cv, (0.5, 0.5), 0.2, flat=0.3)
    assert flat.to_canvas(0, 1)[1] - 90 == pytest.approx(0.2 * 180 * 0.3)


def test_frame_crop_matches_full_frame_and_is_fast():
    cv = Canvas(400, 240)
    fr = Frame(cv, (0.3, 0.6), 0.15, rotate=20)
    m = fr.fill(sdf.sd_box(fr.X, fr.Y, 0, 0, 1.0, 0.5, 0.2))
    X, Y = cv.xy()
    lx, ly = fr.from_canvas(X, Y)
    ref = sdf.fill(fr.px(sdf.sd_box(lx, ly, 0, 0, 1.0, 0.5, 0.2)))
    assert m.shape == (240, 400) and np.abs(m - ref).max() < 3e-3
    assert fr.X.shape[0] * fr.X.shape[1] < 0.25 * 400 * 240       # only the box is evaluated
    big = Canvas(1920, 1080)
    t = time.time()
    rng = np.random.default_rng(0)
    for _ in range(100):
        f = Frame(big, (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)), 0.02, rotate=rng.uniform(0, 360))
        f.paint("#ff8800", sdf.sd_circle(f.X, f.Y, 0, 0, 1.0))
        f.form(sdf.sd_box(f.X, f.Y, 0, 0, 0.8, 0.4, 0.1), "#3366cc", material="plastic")
    assert time.time() - t < 6.0


def test_light_object_and_module_default():
    L = set_light(direction=(0.7, -0.5), elevation=50)
    assert get_light() is L and L.vec[0] > 0 and abs(np.linalg.norm(L.vec) - 1) < 1e-5
    sx, sy = L.shear()
    assert sx < 0                                                   # light from the right → shadow to the left
    assert Light(direction=(-1, 0), front=0.5).shear()[1] < 0      # light in front → shadow runs up the screen


def test_shade_lights_the_side_facing_the_light():
    cv = Canvas(200, 200)
    d = d_circle(cv, (0.5, 0.5), 0.35)
    col = shade(d, "#808080", light=Light(direction=(-1, -1)), material="matte")
    assert col.shape == (200, 200, 3) and np.isfinite(col).all()
    assert luma(col[60, 60]) > luma(col[140, 140]) + 0.15         # upper-left lit, lower-right dark


@pytest.mark.parametrize("material", sorted(MATERIALS))
def test_every_material_paints(material):
    cv = Canvas(120, 90, "#556070")
    m = paint_form(cv, d_circle(cv, (0.5, 0.5), 0.3), "#c08040", material=material)
    assert m.shape == (90, 120) and m.max() > 0.2 and np.isfinite(cv.buf).all()
    assert np.abs(cv.buf[45, 60] - np.array([0.333, 0.376, 0.439])).max() > 0.03


def test_materials_look_distinct():
    out = {}
    for mat in ("matte", "chrome", "glass", "velvet", "gold"):
        cv = Canvas(80, 80, "#406080")
        paint_form(cv, d_circle(cv, (0.5, 0.5), 0.4), "#c08040", material=mat)
        out[mat] = cv.buf.copy()
    keys = list(out)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert np.abs(out[keys[i]] - out[keys[j]]).mean() > 0.02, (keys[i], keys[j])


@pytest.mark.parametrize("profile", ["round", "dome", "flat", "cushion", "ridge", "cylinder"])
def test_profiles_give_unit_normals(profile):
    cv = Canvas(160, 100)
    d = d_box(cv, (0.5, 0.5), (0.35, 0.25), 0.05)
    nx, ny, nz = normals(d, profile, axis=30 if profile == "cylinder" else 90)
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    inside = d < -1
    assert np.allclose(n[inside], 1, atol=1e-3)


def test_cylinder_profile_is_curved_across_the_axis():
    cv = Canvas(200, 100)
    d = d_box(cv, (0.5, 0.5), (0.8, 0.2))                         # long horizontal bar
    nx, ny, nz = normals(d, "cylinder", axis=0)
    assert ny[35, 100] < -0.5 and ny[65, 100] > 0.5 and nz[50, 100] > 0.95
    nx2, ny2, nz2 = normals(d, "cylinder", axis=180 - 1e-6)
    assert np.allclose(ny, ny2, atol=1e-3)
    _, _, nzr = normals(d, "cylinder", axis=33)                    # rotated path still works
    assert np.isfinite(nzr).all()


def test_texture_base_and_extra_material_keywords():
    cv = Canvas(100, 100)
    tex = np.random.default_rng(1).random((100, 100, 3)).astype(np.float32)
    col = shade(d_circle(cv, (0.5, 0.5), 0.4), tex, material="skin", sss=0.8)
    assert col.shape == (100, 100, 3)


def test_shadows_only_darken():
    cv = Canvas(300, 200, "#c0b0a0")
    b = ball(cv, (0.4, 0.7), 0.15, shadow=False)
    after_ball = cv.buf.copy()
    m = cast_shadow(cv, b, light=Light(direction=(-1, -0.4), front=0.5))
    assert m.max() > 0.3 and m.sum() > 100
    diff = cv.buf - after_ball
    assert diff.max() <= 1e-5                                      # multiply: never brightens
    ys, xs = np.nonzero(m > 0.2)
    assert xs.mean() > b["base"][0]                                # falls away from the light (to the right)
    assert np.abs(cv.buf - after_ball)[b.mask > 0.99].max() < 1e-5  # does not darken the object itself
    prev = cv.buf.copy()
    contact_shadow(cv, (0.8, 0.8), 0.1)
    drop_shadow(cv, d_circle(cv, (0.2, 0.3), 0.05))
    assert (cv.buf <= prev + 1e-5).all() and (cv.buf < prev - 0.05).any()
    ob = occlusion_between(d_circle(cv, (0.5, 0.5), 0.1), d_circle(cv, (0.62, 0.5), 0.1), radius=6)
    assert ob.max() > 0.2 and ob[100, 50] == 0


def test_reflection_mirrors_below_the_base():
    cv = Canvas(200, 200, "#202020")
    b = ball(cv, (0.5, 0.5), 0.15, base="#ff0000", shadow=False)
    m = reflection(cv, b, strength=0.5)
    assert m[:100].max() == 0 and m[110:].max() > 0.2
    assert cv.buf[110, 100, 0] > cv.buf[110, 100, 1] + 0.05          # red shows in the floor


def test_solids_return_masks_and_anchors():
    cv = Canvas(480, 270, "#a0a0a0")
    s = ball(cv, (0.2, 0.6), 0.1, cast=True)
    assert isinstance(s, Solid) and s.mask.shape == (270, 480) and s.mask.max() > 0.99
    assert s["base"][1] == pytest.approx(0.6 * 270)
    c = cylinder(cv, (0.4, 0.7), 0.06, 0.2, open=True, fill=0.8, handle=True)
    assert c["top"][1] < c["base"][1] and "handle" in c.anchors
    px_, py_ = c.point(0.5, side=1)
    assert px_ > c["base"][0]
    c.band(0.3, 0.5, "#2050a0")
    k = cone(cv, (0.55, 0.7), 0.06, 0.2)
    assert k.mask.sum() > 100
    b = cuboid(cv, (0.7, 0.7), (0.15, 0.1, 0.12), yaw=30, top="#ffcc88")
    assert "face_top" in b.anchors and len(b["face_top"]) == 4
    t = torus(cv, (0.85, 0.7), 0.06, 0.02)
    assert t.mask.max() > 0.9
    cap = capsule(cv, (0.1, 0.2), (0.3, 0.25), 0.03)
    assert cap.mask.max() > 0.9
    v = lathe(cv, (0.6, 0.35), [(0.04, 0), (0.06, 0.05), (0.03, 0.12), (0.02, 0.2)], material="glass")
    assert v.mask.max() > 0.5 and np.isfinite(cv.buf).all()


def test_spot_scales_solids_with_depth():
    cv = Canvas(480, 270)
    g = Ground(cv, horizon=0.4, camera_height=1.6)
    near, far = g.place(0, 3), g.place(0, 9)
    assert near.k > far.k and near[1] > far[1] > 0.4
    a = ball(cv, near, 0.3, shadow=False).mask.sum()
    b = ball(cv, far, 0.3, shadow=False).mask.sum()
    assert a > 4 * b
    assert near.up(0.5)[1] < near[1]


def test_ground_placement_and_textures():
    cv = Canvas(320, 180)
    g = Ground(cv, horizon=0.4, vanishing=0.5, camera_height=1.5)
    px_, py_, s = g.at(0.0, 3.0)
    assert px_ == pytest.approx(160) and py_ == pytest.approx(72 + 180 * 1.5 / 3)
    assert g.depth(py_ / 180) == pytest.approx(3.0)
    assert g.spot(0.5, py_ / 180).k == pytest.approx(g.place(0, 3).k)
    xw, zw = g.uv()
    assert zw[179, 160] == pytest.approx(g.depth(179 / 180), rel=1e-3)
    for tex in (g.checker(), g.tiles(), g.planks(), g.planks(direction="across"), g.grid_mask()):
        assert tex.shape[:2] == (180, 320) and np.isfinite(tex).all()
    crop = g.planks(box=(10, 100, 60, 150))
    assert crop.shape == (50, 50, 3) and np.allclose(crop, g.planks()[100:150, 10:60], atol=1e-5)
    cv.paint(g.checker(), g.mask())
    assert cv.buf[:60].max() == 0 and cv.buf[120:].std() > 0.05
    box = g.box(0.5, 4, (1.0, 0.8, 0.6), round=0.3, cast=True)
    assert box.mask.max() > 0.9 and "face_front" in box.anchors


def test_room_draws_shell_and_places_objects():
    cv = Canvas(384, 216)
    room = Room(cv, horizon=0.45, back=0.5, width=4.0)
    room.draw(floor="planks", window="left")
    assert np.isfinite(cv.buf).all() and cv.buf.std() > 0.05
    spot = room.place(0.0, room.zb - 1.0)
    assert 0.45 < spot[1] < 1.0
    pic = room.picture("back", center=(0.0, 1.6), size=(1.0, 0.6))
    assert pic.max() > 0.9
    for floor in ("tiles", "checker", "#806040"):
        Room(Canvas(160, 90)).draw(floor=floor, window="back")
    bx, by = room.wall_point("left", room.zb - 0.5, 1.0)
    assert bx < 384 * 0.5


def test_surfaces_for_still_lifes():
    cv = Canvas(320, 180)
    s = tabletop(cv, top=0.5, front=0.9, horizon=0.3)
    assert isinstance(s, Surface)
    a, b = s.place(0.5, 0.0), s.place(0.5, 1.0)
    assert a[1] == pytest.approx(0.9) and b[1] == pytest.approx(0.5) and a.k > b.k
    assert s.place(0.5, 0.5).k == pytest.approx(1.0, rel=1e-3)
    for kind in ("stone", "cloth", "#a08060"):
        tabletop(Canvas(160, 90), kind=kind)
    shelf(cv, y=0.3)
    counter(cv)
    assert np.isfinite(cv.buf).all()


def test_arrangements_are_depth_sorted_and_do_not_overlap():
    pts, order = row(5, (0.2, 0.7), (0.8, 0.5))
    assert len(pts) == 5 and [pts[i][1] for i in order] == sorted(p[1] for p in pts)
    pts, order = grid_positions(2, 3)
    assert len(pts) == 6 and pts[order[0]][1] <= pts[order[-1]][1]
    pts, order = scatter_positions(12, (0.1, 0.5, 0.9, 0.9), min_gap=0.08, seed=3)
    assert len(pts) == 12
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert np.hypot((pts[i][0] - pts[j][0]) * 16 / 9, pts[i][1] - pts[j][1]) >= 0.08 - 1e-9
    pts, order = ring_positions(6, (0.5, 0.6), 0.2, tilt=0.3)
    assert len(pts) == 6 and pts[order[-1]][1] == max(p[1] for p in pts)


def test_public_helpers_and_stable_private_names():
    cv = Canvas(100, 80)
    assert resolve_light() is get_light() and resolve_light((1, 0)).direction == (1.0, 0.0)
    assert _form._resolve_light is resolve_light                      # used by figure.py / creature.py
    for name in ("_clip_box", "_full", "_bbox_of", "_contact_px", "_sd_ellipse", "_grid_box", "_multiply",
                 "_shadow_color"):
        assert callable(getattr(_form, name))
    assert clip_box(cv, -5, -5, 50, 200) == (0, 0, 50, 80) and clip_box(cv, 200, 0, 300, 10) is None
    m = np.zeros((80, 100), np.float32)
    m[10:20, 30:40] = 1
    assert bbox_of(m) == (30, 10, 40, 20)
    assert full_mask(cv, np.ones((5, 5)), (0, 0, 5, 5)).sum() == 25
    assert shadow_color().shape == (3,)
    before = Canvas(100, 80, "#c0c0c0")
    multiply_shadow(before, np.full((10, 10), 0.5, np.float32), (0, 0, 10, 10))
    assert before.buf[5, 5].max() < 0.75 and before.buf[50, 50].min() > 0.75


def test_soft_contact_shadow_fades_to_zero_at_its_edge():
    cv = Canvas(400, 300, "#d0d0d0")
    m = contact_shadow_px(cv, 200, 150, 40, 12, strength=0.9, softness=3.0)
    bb = bbox_of(m, 1e-6)
    assert m.max() > 0.3
    edge = np.concatenate([m[bb[1], bb[0]:bb[2]], m[bb[3] - 1, bb[0]:bb[2]], m[bb[1]:bb[3], bb[0]],
                           m[bb[1]:bb[3], bb[2] - 1]])
    assert edge.max() < 0.01                                        # no visible rectangle on flat backgrounds


def test_cast_shadows_stay_on_a_shelf_and_climb_the_wall():
    def scene(wall):
        cv = Canvas(480, 270, "#e8e0d4")
        set_light(direction=(-0.8, -0.5), elevation=35, front=0.6)
        sh = shelf(cv, 0.7, 0.1, 0.9, depth=0.08, wall=wall)
        before = cv.buf.copy()
        v = lathe(cv, sh.place(0.4, 0.5), [(0.05, 0), (0.08, 0.1), (0.03, 0.3)], cast=True, shadow=False)
        return cv, before, v, sh
    cv, before, v, sh = scene(True)
    dark = (before - cv.buf).sum(-1) * (v.mask < 1e-3)
    top_row = int(sh.top * 270)
    assert dark[:top_row - 2].max() > 0.05                             # shadow on the wall above the board
    ys, xs = np.nonzero(dark[:top_row - 2] > 0.02)
    assert xs.mean() > v["base"][0]                                    # away from the light
    cv2, before2, v2, _ = scene(False)
    dark2 = (before2 - cv2.buf).sum(-1)
    dark2[:, :v2.box[2] + 2] = 0                                       # look right of the object only
    assert dark2[:top_row - 2].max() < 1e-4                            # no wall: the shadow ends at the edge


def test_disc_book_and_spot_moves():
    cv = Canvas(480, 270, "#909090")
    sp = Spot(0.5, 0.8, 1.0, 0.3, hw=270 / 480)
    w = disc(cv, sp, 0.1, 0.05, yaw=20)
    assert isinstance(w, Solid) and w.mask.max() > 0.9 and w["top"][1] < w["base"][1]
    b = book(cv, Spot(0.25, 0.7), (0.2, 0.14, 0.04), cover="#802020")
    assert "face_top" in b.anchors and b.mask.sum() > 200
    m = sp.move(0.2, 0.0, yaw=0)
    assert m[0] == pytest.approx(0.5 + 0.2 * 270 / 480) and m[1] == pytest.approx(0.8)
    assert sp.move(0.0, 0.1)[1] > 0.8                                  # towards the viewer = down the screen
    up = sp.up(0.1)
    assert up.lift > 0 and up.hw == sp.hw and up[1] < sp[1]
    custom = cuboid(cv, (0.8, 0.8), (0.15, 0.1, 0.1), colors=dict(front=lambda f, x, y, z: np.where(
        (y > 0)[:, None], [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])))
    assert custom.mask.max() > 0.9 and np.isfinite(cv.buf).all()


def test_ground_floor_and_rug_keep_outside_pixels():
    cv = Canvas(320, 180)
    cv.gradient([(0, "#a0c0e0"), (0.4, "#e0e8f0")])
    g = Ground(cv, horizon=0.4, camera_height=1.2)
    m = g.floor("tiles", size=0.5)
    assert m[:60].max() == 0 and m[150:].min() > 0.99
    row = int(0.4 * 180) + 1
    assert np.abs(cv.buf[row, 160] - cv.buf[row - 3, 160]).max() < 0.08   # melts into the sky at the horizon
    for tex in ("checker", "planks", "grid", "#806040"):
        Ground(Canvas(160, 90), horizon=0.4).floor(tex)
    before = cv.buf.copy()
    rug = g.rug(-0.6, 0.6, 2.0, 3.5, pattern="stripes")
    changed = np.abs(cv.buf - before).sum(-1) > 1e-4
    assert rug.max() > 0.99 and (changed & (rug < 0.01)).sum() < 0.25 * changed.sum()


def test_room_panel_and_rug():
    cv = Canvas(384, 216)
    room = Room(cv, horizon=0.45, back=0.5, width=4.0)
    room.draw(window="left")
    q = room.panel("back", center=(0.0, 1.6), size=(1.0, 0.6), color="#203028")
    assert len(q) == 4 and q[0][0] < q[1][0] and q[0][1] < q[3][1]
    xs = [p[0] for p in q]
    ys = [p[1] for p in q]
    c = cv.buf[int(np.mean(ys)), int(np.mean(xs))]
    assert np.abs(c - np.array([0.125, 0.19, 0.157])).max() < 0.08
    assert room.rug(-1.0, 1.0, room.zb - 2.0, room.zb - 0.8).max() > 0.9
    rq = room.wall_quad("right", (room.zb - 1.0, 1.5), (0.8, 0.5))
    assert rq[0][0] < rq[1][0]                                        # corners read left → right on screen


def _doc_blocks():
    path = os.path.join(ROOT, "skills", "pixelcanvas", "references", "forms.md")
    with open(path, encoding="utf-8") as f:
        return re.findall(r"```python\n(.*?)```", f.read(), re.S)


@pytest.mark.parametrize("i", range(len(_doc_blocks())))
def test_forms_doc_snippets_run(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = _doc_blocks()[i].replace("Canvas(1920, 1080", "Canvas(320, 180")
    g = {}
    exec(code, g)
    assert np.isfinite(g["cv"].buf).all() and g["cv"].buf.std() > 0.01


@pytest.mark.parametrize("name", ["form_still_life", "form_room"])
def test_example_scenes_render_small(name):
    import importlib
    cv = importlib.import_module(f"scenes.{name}").render(192, 108)
    assert cv.shape == (108, 192, 3) and np.isfinite(cv.buf).all() and 0.1 < cv.buf.mean() < 0.8


# ---- strict options, aliases, registry names, public helpers, manifest, rug / panel

def test_form_unknown_option_names_the_closest():
    cv = Canvas(200, 150)
    with pytest.raises(TypeError, match=r"ball: unknown option 'radious'.*did you mean 'radius'"):
        _form.ball(cv, (0.5, 0.8), 0.1, radious=0.2)
    with pytest.raises(TypeError, match=r"unknown option 'materal'"):
        _form.ball(cv, (0.5, 0.8), 0.1, materal="glossy")
    with pytest.raises(TypeError, match=r"unknown option 'ksx'.*did you mean 'ks'"):
        _form.ball(cv, (0.5, 0.8), 0.1, ksx=0.2)
    with pytest.raises(TypeError, match=r"Light: unknown option 'elevaton'.*did you mean 'elevation'"):
        _form.Light().replace(elevaton=3)
    with pytest.raises(TypeError, match=r"unknown option 'strenth'.*did you mean 'strength'"):
        _form.contact_shadow(cv, (0.5, 0.8), 0.1, strenth=0.5)


def test_form_aliases_are_accepted():
    cv = Canvas(200, 150)
    cv.paint("#d8d0c0")
    _form.ball(cv, (0.5, 0.8), r=0.1)                                   # r -> radius
    _form.ball(cv, (0.3, 0.8), 0.1, base="#2060c0")
    assert np.isfinite(cv.buf).all()
    l = _form.Light().replace(colour="#ffffff")                          # colour -> color
    assert l.color == "#ffffff"
    sol = _form.cylinder(cv, (0.7, 0.8), radius=0.05, height=0.2, name="can")
    assert sol.mask.max() > 0.5


def test_form_unknown_registry_names():
    cv = Canvas(200, 150)
    with pytest.raises(ValueError, match=r"unknown material 'glosy'.*did you mean 'glossy'"):
        _form.ball(cv, (0.5, 0.8), 0.1, material="glosy")
    with pytest.raises(ValueError, match=r"unknown floor texture 'plank'.*did you mean 'planks'"):
        _form.Room(cv).draw(floor="plank")
    with pytest.raises(ValueError, match=r"unknown window wall 'lft'.*did you mean 'left'"):
        _form.Room(cv).draw(window="lft")
    with pytest.raises(ValueError, match=r"unknown profile 'rond'.*did you mean 'round'"):
        _form.normals(np.zeros((8, 8), np.float32) - 1, "rond")


def test_form_public_helper_names_match_the_private_ones():
    for pub, priv in (("resolve_light", "_resolve_light"), ("clip_box", "_clip_box"), ("full_frame", "_full"),
                      ("bbox_of", "_bbox_of"), ("grid_box", "_grid_box"), ("multiply", "_multiply"),
                      ("shadow_color", "_shadow_color"), ("sd_ellipse", "_sd_ellipse"), ("contact_px", "_contact_px")):
        assert callable(getattr(_form, pub)) and callable(getattr(_form, priv)), pub
    X, Y = _form.grid_box(0, 0, 5, 4)
    assert X.shape == (4, 5)
    assert _form.clip_box(Canvas(10, 10), -5, -5, 5, 5) == (0, 0, 5, 5)


def test_form_solids_are_listed_in_the_scene_manifest_by_name():
    from pixelcanvas import manifest
    cv = Canvas(320, 240)
    cv.paint("#d8d0c0")
    _form.lathe(cv, (0.3, 0.8), [(0.06, 0.0), (0.1, 0.08), (0.05, 0.3)], name="vase")
    _form.ball(cv, (0.7, 0.8), 0.1)
    names = [r["name"] for r in manifest.summary(cv)]
    assert names == ["vase", "ball"]


def test_room_rug_and_panel_return_pixels_and_are_listed():
    from pixelcanvas import manifest
    cv = Canvas(480, 270)
    room = _form.Room(cv)
    room.draw()
    m = room.rug(-1.0, 1.0, room.zb - 2.5, room.zb - 1.0, color="#b5523b", border=None, pattern=None)
    assert m.shape == (270, 480) and m.max() > 0.9
    quad = room.panel("back", center=(0.0, 1.6), size=(1.2, 0.7), color="#2b2f2c", frame="#6b4a2e")
    assert len(quad) == 4 and quad[0][0] < quad[1][0] and quad[0][1] < quad[3][1]
    names = [r["name"] for r in manifest.summary(cv)]
    assert "rug" in names and "panel" in names and "room" in names


def test_contact_shadow_leaves_no_box_edge_at_large_softness():
    cv = Canvas(400, 300)
    cv.paint("#d8d0c0")
    _form.contact_shadow(cv, (0.5, 0.5), 0.2, softness=8.0, strength=0.8)
    lum = np.asarray(cv.buf)[..., 0]
    d = lum[0, 0] - lum
    assert d.max() > 0.3 and d[:, 0].max() < 0.01 and d[:, -1].max() < 0.01 and d[0].max() < 0.01 and d[-1].max() < 0.01


# ---- round 1 fixes: positions given in pixels
from pixelcanvas.form import ball  # noqa: E402

def test_check_frac_and_guard_fractions():
    from pixelcanvas.form import check_frac, guard_fractions
    assert check_frac("x", "at", (0.5, 0.9)) == (0.5, 0.9)
    assert check_frac("x", "at", (-0.2, 1.4)) == (-0.2, 1.4)
    with pytest.raises(ValueError, match=r"x: at=\(640.0, 360.0\) looks like pixels.*\(x / cv.w, y / cv.h\)"):
        check_frac("x", "at", (640, 360))

    @guard_fractions
    def f(cv, at=(0.5, 0.5), size=100):
        return at
    assert f(None, (0.3, 0.4)) == (0.3, 0.4)
    assert f(None, size=900) == (0.5, 0.5)                  # sizes are not positions
    with pytest.raises(ValueError, match="looks like pixels"):
        f(None, at=(640, 360))


def test_solids_reject_pixel_positions():
    cv = Canvas(320, 200)
    with pytest.raises(ValueError, match="looks like pixels"):
        ball(cv, (160, 100), 0.1)


def test_form_objects_record_under_the_requests_word():
    from pixelcanvas import Canvas, manifest
    cv = Canvas(400, 240, "#e8e4dc")
    from pixelcanvas import form
    form.ball(cv, (0.2, 0.8), 0.08, base="#e8d12c", name="lemon", alias="citrus")
    form.cylinder(cv, (0.5, 0.8), 0.05, 0.2, name="candle")
    form.lathe(cv, (0.8, 0.8), [(0.05, 0.0), (0.06, 0.1), (0.03, 0.2)], name="vase")
    form.cuboid(cv, (0.5, 0.5), name="crate")
    E = {e["name"]: e for e in manifest.entries(cv)}
    assert {"lemon", "candle", "vase", "crate"} <= set(E)
    assert "citrus" in E["lemon"]["aliases"] and "ball" in E["lemon"]["aliases"]
