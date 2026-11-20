"""The finish of figures drawn over the 3D guide: contact shadow, cel shadows and the picture-level parts of
``pixelcanvas polish`` (rim, depth of field, motion, vignette, grain), with the focus face kept sharp."""
from __future__ import annotations

import numpy as np
import pytest

from .test_draw_free_faults import draw, ok, project  # noqa: F401  (fixture and helpers)


def figure(run, preset="stand"):
    ok(run("new", "pic", "--size", "300x420", "--request", "an anime girl, cel"))
    ok(run("style", "cel"))
    ok(run("do", "plan subject girl --box 150,80,850,1000; plan light top-left; palette make --base \"#2c3460\"; "
                 "palette pick 1; stage done"))
    ok(run("camera", "--horizon", "430", "--eye-height", "1.25"))
    ok(run("pose", "girl", "--preset", preset, "--at", "500,930"))
    ok(run("dress", "girl", "blouse,skirt,shoes"))
    ok(run("render"))
    ok(run("draw", "all"))
    ok(run("flat", "auto"))
    ok(run("shade", "all"))


@pytest.fixture
def girl(draw):
    figure(draw)
    return draw


def test_shade_gives_a_contact_shadow_under_the_feet_without_blocks(girl):
    p = project()
    sh = p.layer("shade")[..., 3] > 0.2
    from pixelcanvas.draw_figure2d import load_maps
    import cv2
    mp = load_maps(p)
    lab = cv2.resize(mp["lab"].astype(np.int32), (p.W, p.H), interpolation=cv2.INTER_NEAREST)
    fig = lab >= 0
    ys = np.nonzero(fig)[0]
    below = sh & ~fig
    below[: int(ys.max() - 0.05 * p.H)] = False
    assert below.sum() > 30                           # a shadow on the ground round the feet


def test_polish_finishes_a_drawn_figure_and_keeps_the_face_sharp(girl):
    p = project()
    before = p.composite_u8("export").astype(float)
    text = ok(girl("polish", "--motion", "girl:right", "--grain", "0.2"))
    for part in ("rim:", "motion:", "vignette:", "grain:"):
        assert part in text
    from pixelcanvas import draw_finish
    p = project()
    after = p.composite_u8("export").astype(float)
    fm = draw_finish.figure_maps(p)
    face = fm["faces"]["girl"]
    # the face: tones may shift (grade, vignette, grain) but nothing is smeared: its edges are as strong as before
    gy0 = np.abs(np.diff(before.mean(-1), axis=0))[face[1:]].mean()
    gy1 = np.abs(np.diff(after.mean(-1), axis=0))[face[1:]].mean()
    assert gy1 > 0.85 * gy0
    # the trail lies behind (left of) the figure
    fig = fm["figs"]["girl"]
    xs = np.nonzero(fig.any(0))[0]
    diff = np.abs(after - before).sum(-1) > 12
    left = diff[:, max(0, xs.min() - 30):xs.min()].mean()
    right = diff[:, xs.max() + 1:xs.max() + 31].mean()
    assert left > right
    ok(girl("undo"))


def test_polish_motion_needs_a_drawn_figure(girl):
    code, err = girl("polish", "--motion", "nobody")
    assert code == 2 and "EXAMPLE:" in err
