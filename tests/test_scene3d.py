"""``pixelcanvas scene``: background kits drawn from the camera, and the furniture block kinds."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pixelcanvas import pose3d as P3
from pixelcanvas import scene3d as S3

from .test_draw_wear import draw, ok, state  # noqa: F401  (draw is a fixture)
from .test_pose3d import _scene


@pytest.mark.parametrize("kit", S3.KITS)
def test_scene_kits_render_with_regions(kit):
    cam = P3.Camera(200, 150, horizon=450, eye=1.4)
    rgb, masks = S3.render(S3.build_kit(kit, {}), cam, 200, 150)
    assert rgb.shape == (150, 200, 3)
    assert masks and any(m.any() for m in masks.values())
    assert rgb.std() > 5                       # not a blank


def test_ground_and_horizon_agree_with_the_camera():
    cam = P3.Camera(300, 200, horizon=500, eye=1.5)
    _, masks = S3.render(S3.build_kit("outdoor", {}), cam, 300, 200)
    hz = int(cam.horizon_px())
    assert masks["ground"][hz + 3:].mean() > 0.95 and not masks["ground"][: hz - 2].any()
    # a corridor floor point at 3 m projects where the camera puts it
    cam = P3.Camera(300, 400, horizon=450, eye=1.3)
    _, masks = S3.render(S3.build_kit("corridor", {"lamps": 0}), cam, 300, 400)
    q, _ = cam.project(np.array([[0.0, 0.0, 3.0]]))
    assert masks["floor"][int(q[0, 1]), int(q[0, 0])]


def test_night_is_darker_and_lamps_light_a_pool():
    cam = P3.Camera(200, 160, horizon=420, eye=1.2)
    day, _ = S3.render(S3.build_kit("platform", {}), cam, 200, 160)
    night, _ = S3.render(S3.build_kit("platform", {"time": "night"}), cam, 200, 160)
    assert night.mean() < 0.7 * day.mean()


@pytest.mark.parametrize("kind", ["bench", "chair", "desk", "door", "window", "lamppost", "tree", "wainscot"])
def test_furniture_block_kinds_have_parts(kind):
    w, (k, size) = kind, P3.BLOCK_WORDS[kind]
    assert k == kind
    b = P3.Block(w, k, [0, 0, 3], size)
    prims = b.prims("b1")
    assert len(prims) >= 2
    if kind in ("bench", "chair"):
        assert abs(b.top_at(0, 3 + size[1] / 2) - size[2]) < 1e-6        # sits on the seat


def test_scene_command_paints_paper_and_survives_draw(draw):
    _scene(draw)
    ok(draw("camera", "--horizon", "440", "--eye-height", "1.3"))
    out = ok(draw("scene", "corridor", "--wall", "#e6d8b8", "--arch", "--time", "night"))
    assert "regions:" in out and "floor" in out
    st = state()
    assert st["draw"]["scene3d"]["kit"] == "corridor"
    assert "wall" in st["regions"] and os.path.exists("w.pixelcanvas/regions/floor.png")
    # options given later merge with the kept ones
    ok(draw("scene", "--wall", "#d0c0a0"))
    st = state()
    assert st["draw"]["scene3d"]["opts"]["wall"] == "#d0c0a0" and st["draw"]["scene3d"]["opts"]["time"] == "night"
    ok(draw("block", "bench", "bench", "--at", "500,900"))
    ok(draw("pose", "girl", "--preset", "sit-ledge", "--on", "bench"))
    ok(draw("draw", "all"))
    ok(draw("flat", "auto"))
    from PIL import Image
    paper = np.asarray(Image.open("w.pixelcanvas/layers/paper.png"))
    assert paper[..., :3].std() > 5
    flat = np.asarray(Image.open("w.pixelcanvas/layers/flat.png"))
    # flat auto fills the figure, not the whole background over the scene
    assert (flat[..., 3] > 128).mean() < 0.6


def test_weather_and_scene_list(draw):
    cam = P3.Camera(200, 150, horizon=450, eye=1.4)
    clear, _ = S3.render(S3.build_kit("street", {}), cam, 200, 150)
    rain, _ = S3.render(S3.build_kit("street", {"weather": "rain"}), cam, 200, 150)
    assert np.abs(clear.astype(int) - rain.astype(int)).mean() > 1
    _scene(draw)
    out = ok(draw("scene", "list"))
    for k in S3.KITS:
        assert f"  {k}:" in out
