"""Residual shape pass (shapes.py): sub-pixel outlines from antialiased edges, the mark rules, region calls."""
import cv2
import numpy as np

from pixelcanvas.brush import apply_ops, new_layer
from pixelcanvas.ops import Fill, check_mark_rules, ops_from_jsonable, ops_to_jsonable
from pixelcanvas.shapes import shape_ops
from pixelcanvas.stage_paint import paint_ops
from pixelcanvas.trace import delta_e

S = 16
H, W = 120, 160
BG = np.array([250, 248, 240], np.float32)


def _masks():
    hs, ws = H * S, W * S
    out = []

    def new():
        return np.zeros((hs, ws), np.uint8)

    m = new(); cv2.circle(m, (int(30.3 * S), int(30.7 * S)), int(14.4 * S), 1, -1); out.append((m, (200, 40, 50)))
    m = new(); cv2.fillPoly(m, [np.int32(cv2.boxPoints(((80.2 * S, 30.6 * S), (30.3 * S, 12.7 * S), 23.0)))], 1)
    out.append((m, (30, 60, 180)))
    m = new(); cv2.fillPoly(m, [np.int32(np.array([[120.4, 10.2], [150.7, 40.1], [112.3, 47.9]]) * S)], 1)
    out.append((m, (40, 150, 60)))
    m = new(); cv2.circle(m, (int(110.5 * S), int(85.2 * S)), int(18.3 * S), 1, -1)
    cv2.circle(m, (int(110.5 * S), int(85.2 * S)), int(10.6 * S), 0, -1); out.append((m, (230, 170, 30)))
    # thin dark lines 1.5 px and 1 px wide, a 4 px dot, two touching discs
    m = new(); cv2.line(m, (int(10.3 * S), int(70.2 * S)), (int(70.6 * S), int(95.4 * S)), 1, int(1.5 * S))
    out.append((m, (20, 20, 25)))
    m = new(); cv2.line(m, (int(10.2 * S), int(80.4 * S)), (int(70.1 * S), int(110.6 * S)), 1, int(1.0 * S))
    out.append((m, (20, 20, 25)))
    m = new(); cv2.circle(m, (int(145.2 * S), int(70.5 * S)), int(2.2 * S), 1, -1); out.append((m, (20, 20, 25)))
    m = new(); cv2.circle(m, (int(140.3 * S), int(105.8 * S)), int(9.1 * S), 1, -1); out.append((m, (90, 50, 140)))
    m = new(); cv2.circle(m, (int(146.1 * S), int(105.3 * S)), int(5.2 * S), 1, -1); out.append((m, (250, 230, 120)))
    return out


def _ref(masks):
    img = np.tile(BG, (H * S, W * S, 1))
    for m, c in masks:
        img[m > 0] = c
    return np.clip(np.rint(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)), 0, 255).astype(np.uint8)


def _paint(ops):
    lay = new_layer(H, W)
    lay[..., :3] = BG / 255.0
    lay[..., 3] = 1.0
    apply_ops(lay, ops)
    return np.clip(np.rint(lay[..., :3] * 255), 0, 255).astype(np.uint8)


def test_antialiased_shapes_come_back_with_their_edges():
    masks = _masks()
    ref = _ref(masks)
    canvas = np.tile(BG.astype(np.uint8), (H, W, 1))
    ops = shape_ops(ref, None, canvas)
    assert ops and not check_mark_rules(ops)
    assert all(isinstance(o, Fill) for o in ops)
    img = _paint(ops)
    d = delta_e(ref, img)
    assert float(d.mean()) < 1.0
    # edge error (px): the colour-estimated coverage of each isolated shape, summed over its edge band, per unit of
    # outline length (a uniform shift of the edge by e px gives e)
    for m, c in masks[:4]:
        cov = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        v = np.asarray(c, np.float32) - BG
        est = lambda im: ((im.astype(np.float32) - BG) @ v) / (v @ v)
        band = cv2.dilate(((cov > 0.001) & (cov < 0.999)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        cs = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        perim = sum(cv2.arcLength(p, True) for p in cs) / S
        err = float(np.abs(est(img) - est(ref))[band].sum() / perim)
        assert err < 0.25, err


def test_thin_lines_keep_their_width_and_darkness():
    masks = _masks()
    ref = _ref(masks)
    ops = shape_ops(ref, None, np.tile(BG.astype(np.uint8), (H, W, 1)))
    img = _paint(ops)
    for m, _ in masks[4:6]:
        cov = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        line = cv2.dilate((cov > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        dark_ref = (255.0 - ref[line].astype(np.float32).mean(1)).sum()
        dark_img = (255.0 - img[line].astype(np.float32).mean(1)).sum()
        assert abs(dark_img - dark_ref) < 0.12 * dark_ref
        assert float(delta_e(ref, img)[line].mean()) < 6.0


def test_a_right_canvas_gets_nothing_and_a_wrong_one_only_improves():
    masks = _masks()
    ref = _ref(masks)
    assert len(shape_ops(ref, None, ref.copy())) == 0
    # a blurred canvas: every accepted mark lowers the error
    canvas = cv2.GaussianBlur(ref, (0, 0), 1.2)
    ops = shape_ops(ref, None, canvas)
    lay = new_layer(H, W)
    lay[..., :3] = canvas / 255.0
    lay[..., 3] = 1.0
    apply_ops(lay, ops)
    out = np.clip(np.rint(lay[..., :3] * 255), 0, 255).astype(np.uint8)
    assert float(delta_e(ref, out).mean()) < 0.6 * float(delta_e(ref, canvas).mean())


def test_region_calls_and_json():
    masks = _masks()
    ref = _ref(masks)
    canvas = np.tile(BG.astype(np.uint8), (H, W, 1))
    whole = shape_ops(ref, None, canvas)
    left = np.zeros((H, W), bool)
    left[:, :W // 2] = True
    parts = [shape_ops(ref, p, canvas) for p in (left, ~left)]
    assert all(o.own for p in parts for o in p)
    # the regions together paint what one call paints (pieces cut by a region's work box may be dropped)
    a = _paint(whole)
    b = _paint(parts[0] + parts[1])
    assert float(delta_e(a, b).mean()) < 0.5
    back = ops_from_jsonable(ops_to_jsonable(whole))
    assert len(back) == len(whole)
    assert np.array_equal(_paint(back), a)


def test_paint_ops_shapes_brush_dispatches():
    masks = _masks()
    ref = _ref(masks)
    canvas = np.tile(BG.astype(np.uint8), (H, W, 1))
    ops = paint_ops(ref, None, canvas, size=6.0, brush="shapes", threshold=5.0)
    assert ops and not check_mark_rules(ops)
