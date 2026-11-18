"""Tests for easel ops (JSON round trip) and the brush engine (coverage, taper, clip, alpha lock, speed)."""
import json
import time

import numpy as np
import pytest

from pixelcanvas.easel.brush import PRESETS, apply_ops, apply_ops_progressive, new_layer, op_bbox
from pixelcanvas.easel.ops import Fill, Stroke, mask_to_rle, ops_from_jsonable, ops_to_jsonable, rle_to_mask

REQUIRED = ["pencil", "pen", "marker", "round", "flat", "dry", "soft", "wash"]


def _sample_ops():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 100, (7, 2)).astype(np.float32)
    m = np.zeros((40, 60), bool)
    m[5:20, 10:50] = True
    m[8:12, 20:30] = False
    return [
        Stroke(pts, 3.3, (10, 20.5, 30), brush="pen", seed=5),
        Stroke(pts, rng.uniform(1, 5, 7).astype(np.float32), rng.integers(0, 255, (7, 3)).astype(np.uint8),
               opacity=0.37, brush="flat", seed=9, erase=True),
        Stroke([[1.1, 2.2]], 4.0, "#ff8000", brush="soft"),
        Fill(polygons=[np.array([[0, 0], [10.25, 0], [10, 10]])], mask_rle=None, color=(1, 2, 3), feather=1.5,
             color2=(200, 100, 0), p0=(0, 0), p1=(10, 10)),
        Fill(polygons=None, mask_rle=mask_to_rle(m), color=(9, 9, 9), opacity=0.5),
    ]


def test_presets_exist():
    for k in REQUIRED:
        assert k in PRESETS


def test_json_round_trip_exact():
    ops = _sample_ops()
    data = ops_to_jsonable(ops)
    text = json.dumps(data)
    back = ops_from_jsonable(json.loads(text))
    assert len(back) == len(ops)
    for a, b in zip(ops, back):
        assert type(a) is type(b)
        if isinstance(a, Stroke):
            assert np.array_equal(a.pts, b.pts) and a.pts.dtype == b.pts.dtype
            if isinstance(a.width, np.ndarray):
                assert np.array_equal(a.width, b.width)
            else:
                assert a.width == b.width
            if isinstance(a.color, np.ndarray):
                assert np.array_equal(a.color, b.color)
            else:
                assert a.color == b.color
            assert (a.opacity, a.brush, a.seed, a.erase) == (b.opacity, b.brush, b.seed, b.erase)
        else:
            assert a.color == b.color and a.color2 == b.color2 and a.p0 == b.p0 and a.p1 == b.p1
            assert a.feather == b.feather and a.opacity == b.opacity
            if a.polygons is not None:
                assert all(np.array_equal(p, q) for p, q in zip(a.polygons, b.polygons))
            else:
                assert np.array_equal(rle_to_mask(a.mask_rle), rle_to_mask(b.mask_rle))
    # identical rendering after the round trip
    L1, L2 = new_layer(120, 120), new_layer(120, 120)
    apply_ops(L1, ops)
    apply_ops(L2, back)
    assert np.array_equal(L1, L2)
    assert ops_to_jsonable(back) == data


def test_json_accepts_plain_lists():
    ops = ops_from_jsonable([{"op": "stroke", "pts": [[1, 2], [30, 40]], "w": 3, "c": [255, 0, 0], "b": "round"}])
    assert ops[0].pts.shape == (2, 2) and ops[0].brush == "round"


def test_rle_round_trip():
    rng = np.random.default_rng(0)
    m = rng.random((33, 47)) > 0.6
    assert np.array_equal(rle_to_mask(mask_to_rle(m)), m)
    m2 = np.zeros((10, 10), bool)
    assert np.array_equal(rle_to_mask(mask_to_rle(m2)), m2)
    m3 = np.ones((5, 7), bool)
    assert np.array_equal(rle_to_mask(mask_to_rle(m3)), m3)


@pytest.mark.parametrize("brush", sorted(PRESETS))
def test_deterministic(brush):
    pts = np.stack([np.linspace(10, 90, 12), 50 + 10 * np.sin(np.linspace(0, 3, 12))], 1)
    ops = [Stroke(pts, 9.0, (200, 40, 40), brush=brush, seed=7)]
    a, b = new_layer(100, 100), new_layer(100, 100)
    apply_ops(a, ops)
    apply_ops(b, ops)
    assert np.array_equal(a, b)
    assert a[..., 3].max() > 0.2


@pytest.mark.parametrize("w", [1.0, 2.0, 3.5, 6.0, 15.0, 40.0])
def test_stroke_width_and_coverage(w):
    L = new_layer(200, 300)
    apply_ops(L, [Stroke([[50, 100.5], [250, 100.5]], w, (0, 0, 0), brush="liner")])
    col = L[:, 150, 3]
    assert abs(col.sum() - w) <= max(0.3, 0.06 * w)
    assert L[100, 150, 3] > 0.99  # centre fully covered
    assert L[100, 20, 3] == 0 and L[100, 280, 3] == 0  # nothing beyond the ends
    # antialiased: the edge has partial coverage for odd geometry
    L2 = new_layer(100, 100)
    apply_ops(L2, [Stroke([[10, 10], [90, 63]], w, (0, 0, 0), brush="liner")])
    a = L2[..., 3]
    assert ((a > 0.05) & (a < 0.95)).sum() > 10


def test_pen_taper():
    L = new_layer(100, 400)
    apply_ops(L, [Stroke([[20, 50.5], [380, 50.5]], 8.0, (0, 0, 0), brush="pen")])
    a = L[..., 3]
    mid = a[:, 200].sum()
    start = a[:, 24].sum()
    end = a[:, 376].sum()
    assert mid > 7.0
    assert start < 0.6 * mid and end < 0.6 * mid


def test_per_point_width_and_color():
    L = new_layer(100, 300)
    n = 30
    pts = np.stack([np.linspace(20, 280, n), np.full(n, 50.5)], 1)
    widths = np.linspace(1, 12, n).astype(np.float32)
    cols = np.stack([np.linspace(255, 0, n), np.zeros(n), np.linspace(0, 255, n)], 1).astype(np.uint8)
    apply_ops(L, [Stroke(pts, widths, cols, brush="liner")])
    assert L[:, 40, 3].sum() < L[:, 260, 3].sum() / 3
    assert L[50, 25, 0] > 0.8 and L[50, 25, 2] < 0.2
    assert L[50, 275, 2] > 0.8 and L[50, 275, 0] < 0.2


def test_fill_polygon_hole_and_gradient():
    L = new_layer(100, 100)
    outer = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], np.float32)
    inner = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], np.float32)
    apply_ops(L, [Fill(polygons=[outer, inner], mask_rle=None, color=(255, 0, 0), color2=(0, 0, 255),
                       p0=(10, 50), p1=(90, 50))])
    assert L[20, 20, 3] == pytest.approx(1.0)
    assert L[50, 50, 3] == 0.0  # hole (even-odd)
    assert L[20, 12, 0] > 0.9 and L[20, 88, 2] > 0.9
    assert L[5, 5, 3] == 0.0


def test_fill_mask_feather():
    m = np.zeros((60, 60), bool)
    m[20:40, 20:40] = True
    L = new_layer(60, 60)
    apply_ops(L, [Fill(polygons=None, mask_rle=mask_to_rle(m), color=(0, 255, 0), feather=4.0)])
    assert L[30, 30, 3] > 0.95
    assert 0.2 < L[30, 20, 3] < 0.8
    assert L[30, 14, 3] > 0.0


def test_clip_and_alpha_lock_and_erase():
    L = new_layer(50, 50)
    clip = np.zeros((50, 50), bool)
    clip[:, :25] = True
    bb = apply_ops(L, [Stroke([[5, 25], [45, 25]], 6.0, (255, 0, 0), brush="round")], clip=clip)
    assert bb is not None
    assert L[25, 10, 3] > 0.9 and L[25, 40, 3] == 0
    # alpha lock: paint only where alpha exists, alpha unchanged
    before = L[..., 3].copy()
    apply_ops(L, [Stroke([[5, 25], [45, 25]], 6.0, (0, 0, 255), brush="round")], alpha_lock=True)
    assert np.allclose(L[..., 3], before)
    assert L[25, 10, 2] > 0.9 and L[25, 40, 3] == 0
    # erase
    apply_ops(L, [Stroke([[5, 25], [15, 25]], 8.0, (0, 0, 0), brush="round", erase=True)])
    assert L[25, 10, 3] < 0.05 and L[25, 20, 3] > 0.9


def test_opacity_no_buildup_within_stroke():
    L = new_layer(60, 60)
    # a stroke crossing itself
    pts = [[10, 30], [50, 30], [50, 31], [10, 31], [10, 30]]
    apply_ops(L, [Stroke(pts, 6.0, (0, 0, 0), opacity=0.5, brush="liner")])
    assert L[..., 3].max() <= 0.5 + 1e-5


def test_bbox_and_progressive():
    L = new_layer(100, 100)
    ops = [Stroke([[10 + i * 8, 10], [10 + i * 8, 90]], 3.0, (0, 0, 0), brush="liner") for i in range(10)]
    bb = op_bbox(ops[0], L.shape)
    assert bb[0] <= 8 and bb[2] >= 12
    frames = list(apply_ops_progressive(L, ops, 4))
    assert len(frames) == 4
    cov = [f[..., 3].sum() for f in frames]
    assert cov == sorted(cov) and cov[0] < cov[-1]
    ref = new_layer(100, 100)
    apply_ops(ref, ops)
    assert np.array_equal(frames[-1], ref)


def test_outside_and_empty():
    L = new_layer(30, 30)
    assert apply_ops(L, [Stroke([[-50, -50], [-40, -40]], 3.0, (0, 0, 0))]) is None
    assert apply_ops(L, []) is None


def test_speed_small_strokes():
    rng = np.random.default_rng(0)
    L = new_layer(1200, 850)
    ops = []
    for i in range(2000):
        p = rng.uniform([0, 0], [850, 1200])
        d = rng.normal(size=2)
        d /= np.linalg.norm(d)
        pts = p + np.outer(np.linspace(0, 15, 6), d)
        ops.append(Stroke(pts, 2.5, tuple(rng.integers(0, 255, 3).tolist()), brush="pen", seed=i))
    t = time.perf_counter()
    apply_ops(L, ops)
    dt = time.perf_counter() - t
    assert dt < 2.0, f"2000 small strokes took {dt:.2f}s"


def test_speed_large_strokes():
    L = new_layer(1200, 850)
    ops = [Stroke(np.stack([np.linspace(100, 700, 30), 300 + 80 * np.sin(np.linspace(0, 3, 30))], 1), 40.0,
                  (200, 50, 50), brush=b, seed=i) for i in range(20) for b in ("round", "flat", "dry", "wash")]
    t = time.perf_counter()
    apply_ops(L, ops)
    assert time.perf_counter() - t < 3.0
