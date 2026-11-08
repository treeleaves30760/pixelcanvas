# -*- coding: utf-8 -*-
"""Media filters: shapes, ranges, and that each one actually changes the look in its characteristic way."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas, stylize, sdf  # noqa: E402
from pixelcanvas.color import luma  # noqa: E402


def _scene():
    cv = Canvas(160, 120)
    cv.gradient([(0, "#4070c0"), (1, "#c0e0ff")])
    X, Y = cv.xy()
    cv.paint("#30a040", np.clip(Y - 80 + 0.5, 0, 1))
    cv.paint("#d03020", sdf.fill(sdf.sd_circle(X, Y, 80, 60, 25)))
    return cv


@pytest.mark.parametrize("fn", [stylize.oil_paint, stylize.watercolor, stylize.pencil_sketch, stylize.cel_shade,
                                stylize.canvas_texture])
def test_media_shape_and_range(fn):
    src = _scene().buf
    out = fn(src)
    assert out.shape == src.shape and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    assert np.abs(out - src).mean() > 0.003                         # it did something


def test_kuwahara_preserves_edges_and_flattens_noise():
    rng = np.random.default_rng(0)
    img = np.zeros((60, 60, 3), np.float32)
    img[:, 30:] = 1.0
    noisy = np.clip(img + rng.normal(0, 0.1, img.shape).astype(np.float32), 0, 1)
    k = stylize.kuwahara(noisy, 3)
    assert k[:, 5:25].std() < noisy[:, 5:25].std() * 0.6            # flat areas get flatter
    assert k[30, 27].max() < 0.3 and k[30, 33].min() > 0.7                      # the edge stays sharp


def test_pencil_is_grey_and_mostly_paper():
    out = stylize.pencil_sketch(_scene().buf)
    assert np.abs(out[..., 0] - out[..., 2]).max() < 0.06            # no colour (paper tint only)
    assert luma(out).mean() > 0.6                                    # mostly light paper


def test_watercolor_lighter_with_white_border():
    src = _scene().buf
    out = stylize.watercolor(src)
    assert luma(out).mean() > luma(src).mean()                       # transparent washes
    assert out[1, 1].min() > 0.9                                     # unpainted paper at the edge


def test_cel_shade_quantises():
    out = stylize.cel_shade(_scene().buf, levels=3, outline=0)
    assert len(np.unique(np.round(luma(out)[60:100, 60:100], 2))) < 40


def test_flow_field_runs_along_edges():
    img = np.zeros((40, 40, 3), np.float32)
    img[:, 20:] = 1.0                                                # vertical edge → strokes run vertically
    vx, vy, coh = stylize.flow_field(img, 2.0)
    assert abs(vy[20, 20]) > 0.9 and coh[20, 20] > 0.5
