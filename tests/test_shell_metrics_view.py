"""metrics.py (monotonic scores, worst spots) and view.py (size rules, layouts, rulers)."""
import time

import cv2
import numpy as np
import pytest

from pixelcanvas import metrics, view

from .shell_helpers import synthetic_ref


def _textured(H=360, W=300):
    rng = np.random.default_rng(0)
    img = synthetic_ref(W, H).astype(np.float32)
    img += rng.normal(0, 10, (H, W, 1))
    for _ in range(10):
        p = rng.integers(0, [W, H], 2)
        q = rng.integers(0, [W, H], 2)
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (20, 20, 20), 2)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_identical_scores_100():
    ref = _textured()
    s = metrics.measure(ref, ref.copy()).scores()
    assert all(v == 100 for v in s.values() if v is not None)


def test_blur_lowers_detail_and_shift_lowers_lines():
    ref = _textured()
    same = metrics.measure(ref, ref.copy()).scores()
    blur = metrics.measure(ref, cv2.GaussianBlur(ref, (0, 0), 2.5)).scores()
    shift = metrics.measure(ref, np.roll(ref, 5, axis=1)).scores()
    assert blur["detail"] < same["detail"] - 20
    assert shift["lines"] < same["lines"] - 20
    assert shift["structure"] < same["structure"]
    blank = metrics.measure(ref, np.full_like(ref, 236)).scores()
    assert blank["overall"] < shift["overall"] < same["overall"]


def test_monotonic_in_blur_strength():
    ref = _textured()
    vals = [metrics.measure(ref, cv2.GaussianBlur(ref, (0, 0), s)).scores()["detail"] for s in (0.8, 1.6, 3.2)]
    assert vals[0] > vals[1] > vals[2]


def test_colour_score_and_worst_spot_words():
    ref = synthetic_ref()
    dark = ref.copy()
    dark[200:260, 100:200] = (dark[200:260, 100:200] * 0.5).astype(np.uint8)
    cmp = metrics.measure(ref, dark, paper=(236, 236, 236))
    spots = cmp.worst(n=3)
    assert spots and "too dark" in spots[0]["problem"]
    x0, y0, x1, y1 = spots[0]["box"]
    assert 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000
    blank = metrics.measure(ref, np.full_like(ref, 236), paper=(236, 236, 236))
    assert blank.worst(n=1)[0]["problem"].startswith("not painted yet (the reference is ")


def test_region_scores_and_no_lines():
    ref = synthetic_ref()
    cmp = metrics.measure(ref, ref.copy())
    m = np.zeros(ref.shape[:2], bool)
    m[300:, :] = True          # the ground: no lines in the reference
    assert cmp.scores(m)["lines"] is None
    assert cmp.scores(np.zeros_like(m))["overall"] is None


def test_full_resolution_speed():
    ref = cv2.resize(_textured(), (850, 1200))
    t = time.time()
    cmp = metrics.measure(ref, cv2.GaussianBlur(ref, (0, 0), 1.5))
    cmp.scores()
    cmp.worst()
    assert time.time() - t < 3.0


# ----------------------------------------------------------------------------------------------- view

@pytest.mark.parametrize("H,W", [(1200, 850), (850, 1200), (300, 300), (1600, 400), (50, 900)])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 9])
def test_view_size_rules(H, W, n):
    img = np.full((H, W, 3), 128, np.uint8)
    out = view.render([view.Panel(img, f"P{i}") for i in range(n)], (0, 0, W, H))
    h, w = out.shape[:2]
    assert max(h, w) <= 1280 and h % 32 == 0 and w % 32 == 0


def test_small_area_enlarged():
    img = np.zeros((1000, 1000, 3), np.uint8)
    img[500:520, 500:520] = 255
    out = view.render([view.Panel(img, "A"), view.Panel(img, "B")], (495, 495, 525, 525))
    # the 30 px area must be drawn at least 3x
    assert max(out.shape[:2]) >= 30 * 3 + view.RULER_L


def test_layout_follows_aspect():
    tall = np.zeros((1000, 200, 3), np.uint8)
    wide = np.zeros((200, 1000, 3), np.uint8)
    assert view._layout(3, 200, 1000)[1] == 3    # tall areas side by side
    assert view._layout(3, 1000, 200)[0] == 3    # wide areas stacked
    out = view.render([view.Panel(tall, "A"), view.Panel(tall, "B")], (0, 0, 200, 1000))
    assert out.shape[0] > out.shape[1]


def test_rulers_drawn_outside_image():
    img = np.full((400, 400, 3), 255, np.uint8)
    out = view.render([view.Panel(img, "REFERENCE")], (0, 0, 400, 400))
    # the ruler band left of the image is grey, the image starts after it
    y = view.MARGIN + view.TITLE_H + view.RULER_T + 20
    assert tuple(out[y, view.MARGIN + 2]) == view.RULER_BG
    assert (out[y, view.MARGIN + view.RULER_L + 5] == 255).all()


def test_marks_and_heat_panels_render():
    img = synthetic_ref()
    from pixelcanvas import regions
    labels, table = regions.compute_marks(img, 20)
    pan = view.Panel(img, "MARKS")
    pan.labels = labels
    pan.label_anchors = {r["n"]: regions.label_anchor(labels == r["n"]) for r in table}
    out = view.render([pan, view.Panel(view.heat_panel(img, np.random.default_rng(0).random(img.shape[:2])),
                                       "DIFF")], (0, 0, img.shape[1], img.shape[0]), grid=True)
    assert view.size_ok(out)
