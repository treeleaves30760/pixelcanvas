# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.solid (3D SDF renderer). Small renders only, so the file runs in a few seconds."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, layers  # noqa: E402
from pixelcanvas import solid as S  # noqa: E402
from pixelcanvas.solid import (Scene, Camera, Lighting, Material, Model, sphere, box, rounded_box, cylinder,  # noqa: E402
                               cone, capsule, torus, ellipsoid, plane, prism, wedge, lathe, extrude, union,
                               smooth_union, subtract, intersect, smooth_subtract, material, PRESETS, LIGHTING)

RNG = np.random.default_rng(0)


def pts(n=2000, r=1.5):
    return RNG.uniform(-r, r, (n, 3)).astype(np.float32)


# ------------------------------------------------------------------ primitives

def test_primitive_distances():
    assert sphere(1)([2, 0, 0]) == pytest.approx(1)
    assert sphere(1)([0, 0, 0]) == pytest.approx(-1)
    assert box(2, 2, 2)([3, 0, 0]) == pytest.approx(2)
    assert box(2, 2, 2)([2, 2, 1]) == pytest.approx(np.sqrt(2))
    assert rounded_box(2, 2, 2, 0.5)([2, 0, 0]) == pytest.approx(1, abs=1e-5)
    assert cylinder(1, 2)([3, 0, 0]) == pytest.approx(2)
    assert cylinder(1, 2)([0, 3, 0]) == pytest.approx(2)
    assert torus(1, 0.25)([1, 0, 0]) == pytest.approx(-0.25)
    assert torus(1, 0.25)([0, 0, 0]) == pytest.approx(0.75)
    assert capsule((0, 0, 0), (0, 2, 0), 0.5)([1, 1, 0]) == pytest.approx(0.5)
    assert cone(1, 0, 2)([0, 0, 0]) < 0 and cone(1, 0, 2)([0, 1.5, 0]) > 0
    assert ellipsoid(2, 1, 1)([3, 0, 0]) == pytest.approx(1, abs=0.05)
    assert plane((0, 1, 0), 0.5)([0, 2, 0]) == pytest.approx(1.5)


def test_prism_lathe_extrude():
    tri = prism([(-1, -1), (1, -1), (0, 1)], depth=1)
    assert tri([0, 0, 0]) < 0 and tri([0, 2, 0]) > 0 and tri([0, 0, 1]) == pytest.approx(0.5, abs=1e-5)
    vase = lathe([(0, 0), (0.5, 0), (0.5, 1), (0, 1)], smooth=False)
    assert vase([0, 0.5, 0]) == pytest.approx(-0.5, abs=1e-5)
    assert vase([1, 0.5, 0]) == pytest.approx(0.5, abs=1e-5)
    skin = lathe([(0.3, 0), (0.5, 0.5), (0.4, 1)], thickness=0.04)
    assert skin([0, 0.5, 0]) > 0.3                       # hollow inside
    from pixelcanvas import sdf
    star = extrude(lambda X, Y: sdf.sd_star(X, Y, 0, 0, 0.5), depth=0.2, size=(1.2, 1.2))
    assert star([0, 0, 0]) < 0 and star([0, 0, 0.5]) > 0
    assert star([0, 0.45, 0]) < 0                        # the star tip points up (y down in 2D → up in 3D)
    w = wedge(1, 1, 1)
    assert w([0, -0.4, 0.2]) < 0 and w([0, 0.4, 0.4]) > 0 and w([0, 0.4, -0.45]) < 0


def test_extrude_text_has_letters():
    t = S.extrude_text("A", height=0.5, depth=0.1)
    lo, hi = t.bounds()
    assert hi[0] - lo[0] > 0.2
    p = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 40), np.linspace(lo[1], hi[1], 40), [0.0]), -1).reshape(-1, 3)
    inside = (t(p) < 0).mean()
    assert 0.05 < inside < 0.7


# ------------------------------------------------------------------ operators match brute force (culling is exact)

def test_operators_match_reference():
    P = pts()
    A, B = box(1, 1, 0.3), sphere(0.6).at(0.3, 0.2, 0)
    a, b = A(P), B(P)
    assert np.allclose(subtract(A, B)(P), np.maximum(a, -b), atol=1e-6)
    assert np.allclose((A - B)(P), np.maximum(a, -b), atol=1e-6)
    assert np.allclose((A | B)(P), np.minimum(a, b), atol=1e-6)
    assert np.allclose((A & B)(P), np.maximum(a, b), atol=1e-6)
    many = [sphere(0.15).at(x, 0, 0) for x in np.linspace(-1, 1, 9)]
    ref = np.min([s(P) for s in many], 0)
    assert np.allclose(union(*many)(P), ref, atol=1e-6)
    su = smooth_union(A, B, k=0.2)(P)
    assert np.all(su <= np.minimum(a, b) + 1e-6)
    ss = smooth_subtract(A, B, k=0.1)(P)
    assert np.all(ss >= np.maximum(a, -b) - 0.1)


def test_modifiers():
    s = sphere(0.5)
    assert s.round(0.1)([0.6, 0, 0]) == pytest.approx(0, abs=1e-6)
    assert s.shell(0.1)([0, 0, 0]) > 0
    m = sphere(0.2).at(1, 0, 0).mirror("x")
    assert m([-1, 0, 0]) == pytest.approx(-0.2)
    r = sphere(0.1).repeat((0.5, 0, 0), (3, 1, 1))
    assert r([0.5, 0, 0]) == pytest.approx(-0.1) and r([-0.5, 0, 0]) == pytest.approx(-0.1)
    assert r([1.0, 0, 0]) > 0.3
    tw = box(1, 2, 0.2).twist(45)
    assert tw([0, 0, 0]) < 0
    bd = box(2, 0.1, 0.1).bend(30)
    assert bd([0, 0, 0]) < 0
    c = sphere(1).cut((0, 1, 0), 0)
    assert c([0, 0.5, 0]) > 0 and c([0, -0.5, 0]) < 0


def test_transforms_and_bounds():
    b = box(2, 1, 1).rotate_y(90).at(1, 2, 3)
    lo, hi = b.bounds()
    assert np.allclose(lo, [0.5, 1.5, 2], atol=1e-6) and np.allclose(hi, [1.5, 2.5, 4], atol=1e-6)
    assert b([1, 2, 3]) < 0
    sc = sphere(1).scale(2)
    assert sc([3, 0, 0]) == pytest.approx(1)
    st = cylinder(0.5, 1).place(x=2, z=-1)
    lo, hi = st.bounds()
    assert lo[1] == pytest.approx(0) and (lo[0] + hi[0]) / 2 == pytest.approx(2)
    assert np.isclose(sphere(1).rotate(30, 40, 50)([0, 0, 0]), -1)
    # chained transforms collapse into one node
    ch = sphere(1).at(1, 0, 0).rotate_y(10).at(0, 1, 0)
    assert isinstance(ch, S._Affine) and not isinstance(ch.child, S._Affine)


# ------------------------------------------------------------------ materials, lighting, camera

def test_materials():
    for name in PRESETS:
        m = material(name)
        P = pts(10)
        assert m.albedo(P, P).shape == (10, 3)
    assert material("#ff0000").roughness == PRESETS["plastic"]["roughness"]
    assert material("clay", "#88aaff", roughness=0.3).roughness == 0.3
    assert material(("gold", "#ffcc00")).metallic == 1
    with pytest.raises(TypeError):
        material("plastic", shiny=3)
    w = S.wood()(pts(50))
    assert w.shape == (50, 3) and np.isfinite(w).all()
    n = S.noise3(np.arange(10.0), 0, 0)
    assert n.min() >= 0 and n.max() <= 1


def test_lighting_presets():
    for name in LIGHTING:
        Lighting(name)
    lit = Lighting("sunset", sun=(-30, 20)).point((0, 1, 0), power=2)
    assert len(lit.points) == 1
    with pytest.raises(ValueError):
        Lighting("disco")


def test_camera_project_and_iso():
    cam = Camera((0, 1, 5), (0, 1, 0), fov=40)
    x, y = cam.project((0, 1, 0), (0, 0, 200, 100))
    assert (x, y) == (pytest.approx(100), pytest.approx(50))
    x2, _ = cam.project((1, 1, 0), (0, 0, 200, 100))
    assert x2 > 100
    iso = Camera.isometric((0, 0, 0), size=4)
    assert iso.ortho == 4
    f = Camera.frame(box(1, 1, 1).place(), azimuth=0, elevation=0)
    assert f.position[2] > 1


# ------------------------------------------------------------------ rendering

def _scene():
    sc = Scene(lighting="studio", ground="checker")
    sc.add(sphere(0.5).place(x=-0.6), "ball", "chrome")
    sc.add(rounded_box(0.8, 0.8, 0.8, 0.1).rotate_y(30).place(x=0.6), "cube", material("plastic", "#2a6fd6"))
    sc.camera = Camera.orbit((0, 0.4, 0), 4.0, azimuth=15, elevation=8)
    return sc


def test_render_buffers_and_masks():
    sc = _scene()
    cv = Canvas(160, 90)
    res = sc.render(cv)
    assert res.depth.shape == (90, 160) and res.normal.shape == (90, 160, 3)
    names = res.names
    assert "ball" in names and "cube" in names and names[0] == "ground"
    m = res.mask("cube")
    assert m.shape == (90, 160) and 0.02 < m.mean() < 0.5
    soft = ((m > 0.05) & (m < 0.95)).sum()
    assert soft > 0                                      # anti-aliased silhouette
    assert np.isfinite(res.depth[res.ids > 0]).all()
    assert res.mask("ground").sum() > 0 and res.mask("sky").sum() > 0
    x, y = sc.project((0.6, 0.4, 0))
    assert res.ids[int(y), int(x)] == names.index("cube")
    assert 0 < cv.buf.mean() < 1 and np.isfinite(cv.buf).all()
    with pytest.raises(KeyError):
        res.mask("nope")


def test_render_layers_one_per_object():
    sc = _scene()
    cv = Canvas(120, 68)
    rec = layers.LayerRecorder(cv)
    res = sc.render(cv, log=rec.mark)
    stack = rec.finish(cv)
    names = [l.name for l in stack]
    for want in ("background", "ground", "ball", "cube"):
        assert want in names
    comp = np.asarray(layers.composite(stack), np.float32)[..., :3]
    if comp.max() > 1.5:
        comp = comp / 255
    assert np.abs(comp - np.clip(cv.buf, 0, 1)).max() < 3 / 255
    assert res is not None


def test_render_box_and_keep_background():
    sc = _scene()
    cv = Canvas(200, 100, "#ff00ff")
    sc.render(cv, box=(100, 0, 200, 100), background=False, aa=1)
    assert np.allclose(cv.buf[50, 10], [1, 0, 1])          # left half untouched
    assert not np.allclose(cv.buf[50, 150], [1, 0, 1])     # right half rendered


def test_render_box_past_the_canvas_is_clipped():
    sc = _scene()
    wide = Canvas(300, 120, "#ff00ff")
    ref = sc.render(wide, box=(100, -20, 300, 100), aa=1, quality="draft")
    cv = Canvas(200, 80, "#ff00ff")
    res = sc.render(cv, box=(100, -20, 300, 100), aa=1, quality="draft")    # past the right and top edges
    assert np.allclose(cv.buf[40, 10], [1, 0, 1])                            # outside the box untouched
    assert np.abs(cv.buf[:, 100:200] - wide.buf[:80, 100:200]).max() < 1e-4  # visible part = same picture
    assert res.depth.shape == (80, 200) and res.mask("ball").shape == (80, 200)
    assert np.allclose(res.project((0, 0.5, 0)), ref.project((0, 0.5, 0)))
    cv2 = Canvas(64, 48, "#ff00ff")
    res2 = sc.render(cv2, box=(100, 10, 180, 60))                            # fully outside: nothing drawn
    assert np.allclose(cv2.buf, [1, 0, 1])
    assert res2.mask("ball").shape == (48, 64) and not res2.mask("ball").any()


def test_backdrop_glass_points_and_post():
    lit = Lighting("soft_box").point((0, 1.5, 1), power=1.0)
    sc = Scene(lighting=lit, ground="backdrop")
    sc.add(sphere(0.4).place(), "glass ball", "glass")
    sc.add(S.mug().scale(4).place(x=0.8), "mug")
    sc.camera = Camera.orbit((0, 0.4, 0), 3.5, azimuth=0, elevation=10)
    cv = Canvas(128, 72)
    res = sc.render(cv, quality="draft")
    before = cv.buf.copy()
    res.depth_of_field(cv, focus=(0, 0.4, 0), blur=6)
    res.fog(cv, "#ffffff", density=0.1)
    assert np.isfinite(cv.buf).all() and not np.allclose(before, cv.buf)


def test_isometric_render():
    sc = Scene(lighting=Lighting("sunny", relative=False), ground="#cccccc")
    sc.add(box(1, 1, 1).place(), "crate", "wood")
    sc.camera = Camera.isometric((0, 0.5, 0), size=3)
    cv = Canvas(96, 64)
    res = sc.render(cv, aa=4)
    assert res.mask("crate").sum() > 50


# ------------------------------------------------------------------ prefabs

PREFABS = ["mug", "bottle", "vase", "table", "chair", "sofa", "bed", "desk_lamp", "bookshelf", "house", "tree", "car",
           "truck", "rocket", "robot", "character", "quadruped", "laptop", "potted_plant", "rug", "toy_block",
           "window_frame", "book", "cup", "speaker", "counter", "coffee_machine", "pendant_lamp"]


@pytest.mark.parametrize("name", PREFABS)
def test_prefab_builds_and_stands_on_floor(name):
    m = getattr(S, name)()
    assert isinstance(m, Model) and len(m.parts) >= 1
    lo, hi = m.bounds()
    assert np.all(hi > lo)
    if name != "window_frame":
        assert lo[1] > -0.06 * (hi[1] - lo[1]) - 0.02, (name, lo)
    P = np.array([(lo + hi) / 2], np.float32)
    for p in m.parts:
        assert np.isfinite(p.shape(P)).all()


def test_prefab_options():
    assert {p.name for p in S.car(style="van").parts} >= {"body", "windows", "tires"}
    assert "flame" in {p.name for p in S.rocket(flame=True).parts}
    assert S.quadruped(kind="cat").name == "cat"
    assert S.tree(kind="pine").bounds()[1][1] == pytest.approx(3.0, rel=0.15)
    m = S.mug().recolor(body="#00ff00").material(coffee="gold")
    assert m["body"].mat.color == "#00ff00" and m["coffee"].mat.metallic == 1
    toy = S.model("lamp", base=(cylinder(0.1, 0.02), "metal"), shade=cone(0.1, 0.05, 0.1).at(0, 0.3, 0))
    assert [p.name for p in toy.parts] == ["base", "shade"]


def test_prefab_scene_renders():
    sc = Scene(lighting="studio")
    sc.add(S.character(), "kid")
    sc.add(S.car().scale(0.25).place(x=0.9), "car")
    sc.camera = Camera.frame(sc, azimuth=20, elevation=15)
    cv = Canvas(96, 64)
    res = sc.render(cv)
    assert res.mask("kid").sum() > 20 and res.mask("car.body").sum() > 5


def test_room_walls_openings_and_interior():
    r = S.room(4, 3, 2.7, window=("left", 0.5, 1.2, 1.0), door=("back", 1.0))
    names = {p.name for p in r.parts}
    assert {"floor", "left wall", "back wall", "frames", "panes", "doors"} <= names
    fl = r["floor"].shape
    assert fl([0, 0, 0]) == pytest.approx(0, abs=1e-6)              # floor top is y = 0
    lw, bw = r["left wall"].shape, r["back wall"].shape
    assert lw([-2.07, 2.0, -1.0]) < 0                               # solid wall away from the window
    assert lw([-2.07, 1.4, 0.5]) > 0                                 # hole where the window is
    assert bw([1.0, 1.0, -1.57]) > 0 and bw([-1.0, 1.0, -1.57]) < 0  # door hole on the back wall
    w = S.wall(3, 2.5, 0.2, openings=[(0, 0.8, 1, 1)])
    assert w([0, 1.3, 0]) > 0 and w([1.2, 1.3, 0]) < 0


def test_hanging_lamp_keeps_its_height():
    lamp = S.pendant_lamp(height=2.7, drop=1.0).place(x=1, z=2)
    lo, hi = lamp.bounds()
    assert hi[1] == pytest.approx(2.7, abs=1e-3) and (lo[0] + hi[0]) / 2 == pytest.approx(1)


def test_tinted_presets_keep_their_pattern():
    P = pts(400)
    for preset in ("wood", "marble", "fabric"):
        m = material(preset, "#7a4a2c")
        assert callable(m.color)
        c = m.color(P, P)
        assert c.shape == (400, 3) and c.std() > 1e-3               # still a pattern, not a flat colour
    rc = S.chair().recolor(frame="#3a6ea5", seat="#aa3322")
    assert callable(rc["frame"].mat.color) and callable(rc["seat"].mat.color)
    assert material("fabric").sheen > 0
    stp = S.stripes(0.1)(P, P, np.full(400, 1.0, np.float32))       # wide footprint → averaged, no aliasing
    assert np.allclose(stp, stp.mean(0), atol=0.05)


def test_floor_presets_and_interior_light():
    sc = Scene(ground=("wood", "#d9b48c"))
    assert callable(sc._ground.mat.color)
    sc.backdrop(("marble", "#eeeeee"))
    assert callable(sc._ground.mat.color)
    assert Lighting("interior").camera_relative is False
    assert Lighting("studio").camera_relative is True


def test_surface_lying_on_the_ground_wins_and_far_edges_stay_clean():
    sc = Scene(lighting="studio", ground="#d0d0d0")
    sc.add(box(2.0, 0.04, 1.0).at(0, -0.02, 0), "slab", material("matte", "#d0d0d0"))
    sc.camera = Camera.orbit((0, 0, 0), 3.0, azimuth=0, elevation=8, fov=30)
    cv = Canvas(160, 90)
    res = sc.render(cv)
    x, y = sc.project((0, 0, 0))
    assert res.ids[int(y), int(x)] == res.names.index("slab")
    col = cv.buf[:, 80, 0]
    rows = np.flatnonzero(res.ids[:, 80] > 0)
    near = col[rows.min() - 3:rows.max() + 4]
    assert near.min() > np.median(col[rows]) - 0.06                  # no dark seam at the slab's far edge


def test_crease_pixels_are_antialiased():
    sc = Scene(lighting="studio", ground=None)
    sc.add(box(1, 1, 1).rotate_y(45).at(0.013, 0, 0), "cube", material("matte", "#ffffff"))
    sc.camera = Camera.orbit((0, 0, 0), 4.0, azimuth=0, elevation=0, fov=30)
    cv1, cv8, big = Canvas(120, 90), Canvas(120, 90), Canvas(480, 360)
    sc.render(cv1, aa=1)
    sc.render(cv8, aa=8)
    sc.render(big, aa=1)
    ref = big.buf.reshape(90, 4, 120, 4, 3).mean((1, 3))           # 16x supersampled reference
    x, y = sc.project((0.013, 0, 0.7071), cv=cv1)                  # front vertical edge of the cube
    win = (slice(int(y) - 10, int(y) + 10), slice(int(x) - 3, int(x) + 4))
    e1, e8 = np.abs(cv1.buf[win] - ref[win]).max(), np.abs(cv8.buf[win] - ref[win]).max()
    assert e8 < e1 * 0.6                                           # the crease is anti-aliased


def test_frame_fits_the_silhouette():
    sc = Scene(ground=None)
    sc.add(S.car().rotate_y(-30), "car")
    sc.camera = Camera.frame(sc, azimuth=30, elevation=15, fill=0.9)
    cv = Canvas(160, 90)
    m = sc.render(cv, aa=1).mask("car") > 0.5
    ys, xs = np.nonzero(m)
    assert (xs.max() - xs.min()) / 160 > 0.7 or (ys.max() - ys.min()) / 90 > 0.7
    assert xs.min() > 0 and xs.max() < 159 and ys.min() > 0 and ys.max() < 89
    iso = Camera.isometric(sc)
    assert iso.ortho and iso.ortho > 1


def test_frame_and_isometric_follow_the_canvas_shape():
    sc = Scene(ground=None)
    sc.add(S.mug(), "mug")
    sc.camera = Camera.frame(sc)                                  # no aspect: fitted to each canvas
    for W, H in ((120, 90), (60, 120)):
        m = sc.render(Canvas(W, H), aa=1).mask("mug") > 0.5
        ys, xs = np.nonzero(m)
        assert xs.min() > 0 and xs.max() < W - 1 and ys.min() > 0 and ys.max() < H - 1
        assert max((xs.max() - xs.min()) / W, (ys.max() - ys.min()) / H) > 0.6
    sc.camera = Camera.isometric(sc)
    m = sc.render(Canvas(50, 120), aa=1).mask("mug") > 0.5
    ys, xs = np.nonzero(m)
    assert xs.min() > 0 and xs.max() < 49 and (xs.max() - xs.min()) / 50 > 0.6


def test_smooth_lathe_is_accurate_and_free_of_facets():
    prof = [(0.0, 0.0), (0.024, 0.0), (0.03, 0.008), (0.04, 0.035), (0.044, 0.065), (0.045, 0.07)]
    sh = lathe(prof, thickness=0.003)
    P = S._catmull(np.array(prof), 40)
    q = np.linspace(0.0, 0.06, 3000, dtype=np.float32)
    y = np.linspace(-0.004, 0.074, 3000, dtype=np.float32)
    Q, Y = [a.ravel() for a in np.meshgrid(q[::30], y[::30])]
    exact = S._polyline2d(Q, Y, P) - np.float32(0.0015)
    got = sh._d(Q, Y, np.zeros_like(Q))
    band = np.abs(exact) < 0.0007                                 # near the surface (the skin is 3 mm thick)
    assert np.abs(got - exact)[band].max() < 2e-5
    # normals along the outer wall turn smoothly (a polyline would jump at every vertex)
    ys = np.linspace(0.015, 0.06, 400, dtype=np.float32)
    qs = np.array([q[S._polyline2d(q, np.full_like(q, yy), P) < 0.0015].max() for yy in ys], np.float32)
    e = np.float32(5e-5)
    nq = sh._d(qs + e, ys, 0 * ys) - sh._d(qs - e, ys, 0 * ys)
    ny = sh._d(qs, ys + e, 0 * ys) - sh._d(qs, ys - e, 0 * ys)
    ang = np.degrees(np.arctan2(ny, nq))
    assert np.abs(np.diff(ang)).max() < 0.5
    # far from the surface (shadow and AO probes) the distance is still right
    far = np.array([0.2, 0.3], np.float32), np.array([0.03, 0.2], np.float32)
    ref = S._polyline2d(far[0], far[1], P) - np.float32(0.0015)
    assert np.allclose(sh._d(far[0], far[1], np.zeros(2, np.float32)), ref, atol=1e-3)
    blob = prism([(0, 0), (1, 0.2), (0.8, 1), (0.1, 0.8)], 0.2, smooth=True)
    assert blob([0.5, 0.5, 0]) < 0 and blob([3, 3, 0]) > 2


def test_wood_and_weave_show_detail_up_close_and_average_far_away():
    P = pts(2000, 0.05)
    N = np.tile(np.float32([0, 1, 0]), (2000, 1))
    for fn in (S.wood("#d9b48c", "#9a6a40"), S.weave("#808080")):
        near = fn(P, N, np.full(2000, 1e-5, np.float32))
        far = fn(P, N, np.full(2000, 0.05, np.float32))
        assert near.std(0).max() > 2.5 * far.std(0).max()


def test_depth_of_field_focuses_on_a_named_object_and_void_background_is_clean():
    sc = Scene(lighting="interior", ground=None)
    sc.add(S.chair(), "chair")
    sc.camera = Camera.isometric(sc)
    cv = Canvas(64, 48)
    res = sc.render(cv)
    corner = cv.buf[:4, :4].mean((0, 1))
    assert corner.min() > 0.6                                     # horizon colour, not the dark world below
    res.depth_of_field(cv, focus="chair", blur=6)
    sc2 = Scene(ground=None)
    sc2.add(sphere(0.2).at(50, 0, 0), "far")
    sc2.add(sphere(0.2), "ball")
    sc2.camera = Camera.frame(sphere(0.2))
    res2 = sc2.render(Canvas(32, 18))
    with pytest.raises(ValueError):
        res2.depth_of_field(Canvas(32, 18), focus="far")              # not in the picture


# ------------------------------------------------------------------ the reference page runs as written

def test_reference_snippets_run(tmp_path, monkeypatch):
    doc = os.path.join(ROOT, "skills", "pixelcanvas", "references", "solid.md")
    text = open(doc, encoding="utf-8").read()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks
    monkeypatch.chdir(tmp_path)
    for code in blocks:
        code = re.sub(r"Canvas\(\d+, \d+\)", "Canvas(96, 54)", code)
        exec(compile(code, "solid.md", "exec"), {"__name__": "__doc__"})


# ---- strict options, aliases, registry names

def test_solid_unknown_option_names_the_closest():
    with pytest.raises(TypeError, match=r"box: unknown option 'widht'.*did you mean 'w'|box: unknown option 'widht'"):
        S.box(widht=1.0)
    with pytest.raises(TypeError, match=r"Material: unknown option 'rougness'.*did you mean 'roughness'"):
        S.material("plastic", rougness=0.5)
    with pytest.raises(TypeError, match=r"Lighting: unknown option 'softnes'.*did you mean 'softness'"):
        Lighting("studio", softnes=0.2)
    with pytest.raises(TypeError, match=r"Model.recolor: unknown option 'bodyy'.*did you mean 'body'"):
        S.mug().recolor(bodyy="#fff")


def test_solid_aliases_are_accepted():
    assert S.sphere(radius=0.5).bounds() is not None                              # radius -> r
    assert S.material("plastic", colour="#2a9df4").color == "#2a9df4"             # colour -> color
    assert S.mug().recolor(bodys="#fff") is not None                              # plural of a part name


def test_solid_unknown_registry_names():
    with pytest.raises(ValueError, match=r"unknown material 'plasic'.*did you mean 'plastic'"):
        S.material("plasic")
    with pytest.raises(ValueError, match=r"unknown lighting preset 'studo'.*did you mean 'studio'"):
        Lighting("studo")
    with pytest.raises(ValueError, match=r"invalid kind 'rnd'|invalid kind 'pin'|invalid kind"):
        S.tree(kind="pin")
    with pytest.raises(ValueError, match=r"invalid quality 'fnal'.*did you mean 'final'"):
        Scene().render(Canvas(40, 30), quality="fnal")


# ---- v0.4 prefabs, resting one model on another, window views, renderer notes, speed-up vs brute force

NEW_PREFABS = {"armchair": {"frame", "seat", "back", "armrest_l", "armrest_r", "legs"},
               "floor_lamp": {"base", "pole", "shade", "bulb"}, "side_table": {"top", "legs", "shelf"},
               "earbuds": {"earbud_l", "earbud_r", "tips", "mesh"},
               "charging_case": {"case", "lid", "hinge", "light", "earbud_l", "earbud_r"},
               "smartphone": {"body", "screen", "camera", "buttons"}, "smartwatch": {"case", "screen", "crown", "band"},
               "tent": {"canvas", "floor", "flaps", "poles"}, "stone_lantern": {"base", "pillar", "firebox", "roof"},
               "arched_bridge": {"deck", "railings", "posts"}}


@pytest.mark.parametrize("name", sorted(NEW_PREFABS))
def test_new_prefabs_have_named_parts_and_stand_on_the_floor(name):
    m = getattr(S, name)()
    assert NEW_PREFABS[name] <= {p.name for p in m.parts}
    lo, hi = m.bounds()
    assert abs(lo[1]) < 0.01 * max(hi[1], 0.05) + 1e-3, (name, lo)
    assert name in S.names() and name in S.find(name.replace("_", " "))
    assert S.describe(name).startswith(name + "(")


def test_prefab_options_round_rug_lit_tent_and_lamp_light():
    r = S.rug(1.8, shape="round")
    lo, hi = r.bounds()
    assert hi[0] - lo[0] == pytest.approx(1.8, abs=0.01) and hi[2] - lo[2] == pytest.approx(1.8, abs=0.01)
    assert S.rug(2, 1, shape="oval").size[2] == pytest.approx(1.0, abs=0.01)
    assert "lamp" in {p.name for p in S.tent(lit=True).parts}
    assert "light" in {p.name for p in S.stone_lantern(lit=True).parts}
    sc = Scene()
    lamp = sc.add(S.floor_lamp().place(x=1), "lamp")
    sc.light_from(lamp)                                           # a Model with a bulb lights from the bulb
    assert sc.lighting.points[-1]["pos"][1] == pytest.approx(lamp["bulb"].shape.bounds()[0][1] + 0.05, abs=0.01)
    assert len(S.charging_case(buds=False).parts) == 5


def test_registry_find_and_did_you_mean():
    assert S.find("reading lamp")[:2] == ["floor_lamp", "desk_lamp"] or "floor_lamp" in S.find("reading lamp")[:2]
    assert S.find("airpods")[0] in ("earbuds", "charging_case")
    assert "armchair" in S.names("furniture")
    with pytest.raises(ValueError, match=r"unknown solid prefab 'armchiar'.*did you mean 'armchair'"):
        S.describe("armchiar")
    with pytest.raises(AttributeError, match=r"armchair"):
        S.armchiar                                                               # noqa: B018
    with pytest.raises(ValueError, match=r"invalid shape 'sqare'.*did you mean 'square'|invalid shape 'sqare'"):
        S.rug(shape="sqare")
    with pytest.raises(KeyError, match=r"unknown part of 'armchair' 'armrest_x'.*did you mean"):
        S.armchair()["armrest_x"]


def _gap(top_shape, obj, xz):
    """Height of the object's lowest point above the surface under it, along the line through xz."""
    y_top = S._surface_y(top_shape, np.array([xz[0]]), np.array([xz[1]]), 3.0, -1.0)[0]
    lo, hi = obj.bounds()
    y_bot = S._surface_y(S._shape_of(obj), np.array([xz[0]]), np.array([xz[1]]), lo[1] - 0.01, hi[1], up=True)[0]
    return y_bot - y_top


def test_on_rests_on_a_named_part_without_sinking():
    chair = S.armchair().rotate_y(-35).place(x=0.1, z=0.4)
    block = S.box(0.08, 0.08, 0.08).on(chair, "armrest_r")
    lo, hi = block.bounds()
    c = ((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2)
    assert abs(_gap(chair["armrest_r"].shape, block, c)) < 0.003
    al = chair["armrest_l"].shape.bounds()
    ar = chair["armrest_r"].shape.bounds()
    assert np.linalg.norm(((lo + hi) / 2 - (ar[0] + ar[1]) / 2)[[0, 2]]) < 0.1     # on the right armrest
    assert np.linalg.norm(((lo + hi) / 2 - (al[0] + al[1]) / 2)[[0, 2]]) > 0.4
    table = S.table().place(x=2)
    mug = S.mug().on(table, "top", at=(0.25, 0.5))
    assert mug.bounds()[0][1] == pytest.approx(0.75, abs=0.004)
    assert mug.bounds()[0][0] < 2 - 0.1
    shelf = S.bookshelf(books=False)
    lv1 = S.box(0.1, 0.1, 0.1).on(shelf, "shelf")
    lv2 = S.box(0.1, 0.1, 0.1).on(shelf, "shelf", level=2)
    assert 0.5 < lv2.bounds()[0][1] < lv1.bounds()[0][1] < 1.8
    assert S.sphere(0.1).on(table, xz=(2.0, 0.0)).bounds()[0][1] == pytest.approx(0.75, abs=0.004)
    with pytest.raises(KeyError, match=r"did you mean 'armrest_r'"):
        S.box(0.1, 0.1, 0.1).on(chair, "armrest_rr")


def test_room_window_view_and_screen_mask():
    sc = Scene(lighting="interior", ground=None)
    sc.add(S.room(3, 3, 2.5, door=None, view="night city"), "room")
    sc.camera = Camera.isometric(sc)
    cv = Canvas(160, 120)
    res = sc.render(cv)
    wm = res.window_mask()
    assert wm.sum() > 30 and np.array_equal(wm, res.mask("room.window"))
    assert float(wm.max()) <= 1.0 and (wm * (res.mask("room.left wall") > 0.99)).sum() == 0
    night = cv.buf[wm > 0.99].mean()
    assert night < 0.45                                          # a dark night view, not the default daylight
    names = [e["name"] for e in cv.manifest]
    assert "window" in names and "night city" in names
    with pytest.raises(ValueError, match=r"unknown window view 'rainn'.*did you mean 'rain'"):
        S.room(view="rainn")
    calls = []

    def paint(u, v):
        calls.append(len(u))
        return np.stack([u, v, np.zeros_like(u)], -1)
    S.room(view="#335577")
    sc2 = Scene(lighting="interior", ground=None)
    sc2.add(S.room(3, 3, 2.5, door=None, view=paint), "room")
    sc2.camera = Camera.isometric(sc2)
    sc2.render(Canvas(80, 60))
    assert calls


def test_render_records_lighting_shadows_and_reflections():
    sc = Scene(lighting="studio", ground="#e0dcd4")
    sc.add(S.sphere(0.3).place(), "ball", "chrome")
    sc.add(S.box(0.4, 0.4, 0.4).place(x=0.8), "cube")
    sc.camera = Camera.frame(sc, azimuth=20, elevation=20)
    cv = Canvas(96, 64)
    sc.render(cv)
    ents = {e["name"]: e for e in cv.manifest}
    assert {"lighting", "shadows", "reflections"} <= set(ents)
    assert ents["lighting"]["meta"].get("backdrop") and ents["shadows"]["meta"].get("see_through")
    assert "soft studio lighting" in ents["lighting"]["aliases"]


def _speckle(img, mask):
    L = img[..., :3].mean(-1)
    st = np.stack([np.roll(np.roll(L, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    return int((((np.median(st, 0) - L) > 0.12) & mask).sum())


def test_stacked_near_coplanar_rugs_have_no_speckle_at_final_quality():
    """Thin cylinders stacked a few mm apart next to a scaled, rotated union (the reading-nook case): tight
    anti-aliasing hits used to take a far part's capped distance for a hit and paint dark dots on the rug."""
    sc = Scene(lighting="interior", ground=None)
    sc.add(S.room(3.4, 3.4, 2.6, door=None), "room")
    for i, (r, h, c) in enumerate([(1.05, 0.012, "#6f8f8a"), (0.86, 0.026, "#e9dcc0"), (0.6, 0.04, "#c8664a")]):
        sc.add(cylinder(r, h).place(x=0.1, z=0.35), f"rug {i}", S.material("fabric", c))
    body = rounded_box(0.85, 0.22, 0.8, 0.07).at(0, 0.27, 0) | rounded_box(0.2, 0.42, 0.82, 0.08).at(-0.52, 0.31, 0) \
        | rounded_box(0.2, 0.42, 0.82, 0.08).at(0.52, 0.31, 0) | rounded_box(0.85, 0.62, 0.2, 0.08).at(0, 0.55, -0.32) \
        | cylinder(0.035, 0.12).at(0.46, 0.06, 0.32)
    sc.add(body.scale(1.3).rotate_y(-35).place(x=0.1, z=0.4), "chair", S.material("fabric", "#c4623f"))
    sc.camera = Camera.isometric(sc)
    cv = Canvas(800, 600)
    res = sc.render(cv, quality="final")
    img = np.clip(cv.buf, 0, 1)
    chair = res.mask("chair") > 0
    for _ in range(4):
        chair = chair | np.roll(chair, 1, 0) | np.roll(chair, -1, 0) | np.roll(chair, 1, 1) | np.roll(chair, -1, 1)
    for i in range(3):                       # inside each ring of the rug, away from its edges and the chair
        m = res.mask(f"rug {i}") > 0.99
        inner = m & np.roll(m, 3, 0) & np.roll(m, -3, 0) & np.roll(m, 3, 1) & np.roll(m, -3, 1) & ~chair
        assert inner.sum() > 20
        assert _speckle(img, inner) == 0, i
        assert not (res.ids[inner] == res.names.index("chair")).any(), i


def test_accelerated_render_matches_brute_force():
    """Distance caps, union / subtract culling, the part tree for shadows and AO must give the brute-force image."""
    def build():
        sc = Scene(lighting="studio", ground="#d9d4cc")
        chair = sc.add(S.armchair().rotate_y(20).place(x=-0.3), "armchair")
        sc.add(S.side_table().place(x=0.6, z=0.2), "table")
        sc.add(S.box(0.1, 0.1, 0.1).on(chair, "seat"), "box", "plastic")
        sc.add(smooth_union(S.sphere(0.12), S.ellipsoid(0.2, 0.08, 0.1).at(0.15, 0, 0), k=0.05).place(x=0.6, z=0.9),
               "blob", "clay")
        sc.camera = Camera.frame(sc, azimuth=25, elevation=18)
        return sc
    a, b = Canvas(96, 64), Canvas(96, 64)
    build().render(a)
    with S._reference():
        build().render(b)
    d = np.abs(np.clip(a.buf, 0, 1) - np.clip(b.buf, 0, 1)).max(-1) * 255
    assert d.mean() < 0.5 and (d > 12).sum() <= 3, (d.max(), d.mean(), (d > 12).sum())


# ------------------------------------------------------------------ foliage-cluster canopies and moss

def _hue_spread(rgbs):
    rgbs = np.asarray(rgbs, np.float32)
    return float(np.std(rgbs[:, 0] - rgbs[:, 1]))


def test_cluster_canopy_trees():
    for kind in ("maple", "oak"):
        t = S.tree(kind=kind)
        assert {p.name for p in t.parts} == {"trunk", "foliage"}
        lo, hi = t.bounds()
        assert lo[1] > -0.1 and 2.6 < hi[1] < 3.6
        fol = t["foliage"]
        assert len(fol.shape.bounds()) == 2
    # many leaf clumps, each its own tone: the canopy colour varies from clump to clump
    sc = Scene(lighting="sunny", ground="#a8b890")
    sc.add(S.tree(kind="maple", season="autumn"), "maple")
    sc.camera = Camera.frame(sc, azimuth=20, elevation=8)
    cv = Canvas(120, 120)
    res = sc.render(cv, aa=1)
    m = res.mask("maple.foliage") > 0.99
    assert m.sum() > 1500
    px = np.clip(cv.buf[m], 0, 1)
    assert px[:, 0].mean() > px[:, 1].mean() + 0.15            # autumn: reds and oranges
    assert _hue_spread(px) > 0.06                               # not one flat colour
    assert res.mask("maple.trunk").sum() > 30                   # the trunk and limbs show under the crown
    summer = S.tree(kind="oak")["foliage"].mat.color(np.array([[0.0, 2.6, 0.0], [1.0, 2.0, 0.3]], np.float32))
    assert (summer[:, 1] > summer[:, 0]).all()                  # summer: green
    one = S.tree(kind="maple", leaves="#c83a22")["foliage"].mat
    many = S.tree(kind="maple", leaves=["#c83a22", "#f0b030"])["foliage"].mat
    assert one.color is not None and many.color is not None
    assert S.tree()["foliage"].mat.color is not None            # old kinds keep working
    with pytest.raises(ValueError):
        S.tree(kind="mapel")
    with pytest.raises(ValueError):
        S.tree(kind="maple", season="winter")


def test_moss_on_stone_and_rock():
    up = np.array([[0, 1, 0]] * 400, np.float32)
    down = -up
    P = RNG.uniform(-1, 1, (400, 3)).astype(np.float32)
    plain = material("stone")
    mossy = material("stone", moss=0.7)
    a_up, a_down = mossy.albedo(P, up), mossy.albedo(P, down)
    assert (a_up[:, 1] - a_up[:, 0]).mean() > (plain.albedo(P, up)[:, 1] - plain.albedo(P, up)[:, 0]).mean() + 0.02
    assert (a_up[:, 1] - a_up[:, 0]).mean() > (a_down[:, 1] - a_down[:, 0]).mean()     # moss grows on top
    assert material("wood", "#8a6a4a", moss=0.5).albedo(P, up).shape == (400, 3)     # any material
    r = S.rock(moss=0.5)
    lo, hi = r.bounds()
    assert isinstance(r, Model) and abs(lo[1]) < 1e-6 and hi[1] > 0.3           # cut flat on the ground
    with pytest.raises(TypeError):
        material("stone", mosss=0.5)
