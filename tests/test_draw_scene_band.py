"""Band shapes in a scene: a band drawn behind stays behind every nearer subject (their paint and regions stay),
tiered trees differ from one another, and a mountain band drawn as an even zigzag gets an irregular contour."""
from __future__ import annotations

import numpy as np

from pixelcanvas import draw_scene as ds

from .test_draw_flow import draw, ok  # noqa: F401  (fixture and helpers)


def project(name):
    from pixelcanvas.project import Project
    return Project.find(f"{name}.pixelcanvas")


# ============================================================================================ band --behind

def setup_cat(run):
    ok(run("new", "pic", "--mode", "free", "--size", "300x225", "--request",
           "A cat sleeping on a windowsill with potted plants, warm afternoon light, soft painterly style."))
    ok(run("do", "plan subject wall --box 0,0,1000,1000; plan subject window --box 150,40,850,620; "
                 "plan subject sill --box 80,600,920,720; plan subject cat --box 300,430,700,640; "
                 "plan part body --of cat --box 300,450,700,640 --focus 3 --color fur; "
                 "plan subject pot-l --box 120,470,250,620; plan light 30"))
    ok(run("do", 'palette make --base "#e9a35a"; palette pick 2; palette set pot-l "#b8623e" window "#9cc4de"'))
    ok(run("stage", "done"))
    ok(run("stage", "done", "--because", "test"))
    ok(run("do", 'band wall "0,0 1000,0 1000,1000 0,1000" --closed --color wall; '
                 'band window "150,40 850,40 850,620 150,620" --closed --color window; '
                 'band sill "80,600 920,600 920,720 80,720" --closed --color sill; '
                 'band pot-l "130,470 240,470 225,620 145,620" --closed --color pot-l; '
                 'band cat --ellipse 300,450,700,640 --color fur'))


def test_band_behind_stays_behind_every_nearer_subject(draw):
    setup_cat(draw)
    ok(draw("paint", "cat", "--dir", "0"))
    p = project("pic")
    near = ("window", "sill", "pot-l", "cat")
    masks = {n: p.region_mask(n).copy() for n in near}
    wall_area = float(p.region_mask("wall").mean())
    layer = p.layer("paint").copy()
    txt = ok(draw("band", "wall", "--ellipse", "0,0,1000,1000", "--color", "wall", "--behind", "cat"))
    assert "behind cat" in txt and "window" in txt
    p = project("pic")
    # the nearer subjects keep their regions
    for n in near:
        assert np.array_equal(p.region_mask(n), masks[n]), n
    # and their paint: nothing changed inside them (one pixel in from the edge, past the feather)
    import cv2
    after = p.layer("paint")
    for n in near:
        inner = cv2.erode(masks[n].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        diff = np.abs(after[..., :3][inner] - layer[..., :3][inner]).max()
        assert diff < 1e-3, (n, diff)
    # the cat's brushwork is not flattened to a single colour
    assert after[..., :3][masks["cat"]].std(axis=0).max() > 0.005
    # the wall did not shrink to the oval: the corners stay wall
    wm = p.region_mask("wall")
    assert wm[2, 2] and wm[2, -3] and wm[-3, 2] and wm[-3, -3]
    assert float(wm.mean()) >= wall_area - 0.01


# ============================================================================================ tiers

def test_tiers_differ_between_trees_and_repeat_for_the_same_step():
    a = ds.tier_outline((100, 900), (100, 500), 220, 5, seed=ds.shape_seed("pines-1", 12))
    b = ds.tier_outline((100, 900), (100, 500), 220, 5, seed=ds.shape_seed("pines-1", 12))
    assert np.array_equal(a, b)                         # deterministic: a layer re-rendered gives the same tree
    shapes = [ds.tier_outline((100, 900), (100, 500), 220, 5, seed=ds.shape_seed(f"pines-{i}", 12 + i))
              for i in range(1, 7)]
    hs = [900 - s[:, 1].min() for s in shapes]
    ws = [s[:, 0].max() - s[:, 0].min() for s in shapes]
    ratio = np.array(hs) / np.array(ws)
    assert ratio.std() / ratio.mean() > 0.03            # proportions vary between calls
    assert all(330 <= h <= 470 for h in hs)             # but stay near the asked height (about +-10 %)
    import cv2
    for s in shapes:
        # asymmetric: the silhouette and its mirror image (about the trunk) do not match
        m = np.zeros((1000, 400), np.uint8)
        cv2.fillPoly(m, [np.round(s).astype(np.int32)], 1)
        mir = np.zeros_like(m)
        cv2.fillPoly(mir, [np.round(np.c_[200 - s[:, 0], s[:, 1]]).astype(np.int32)], 1)
        iou = (m & mir).sum() / max(1, (m | mir).sum())
        assert iou < 0.95, iou
        assert len(s) > 60                              # notches and drooping edges, not a plain zigzag


def test_tier_outline_is_a_simple_polygon():
    def ccw(a, b, d):
        return (d[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (d[0] - a[0])
    for seed in range(25):
        for tiers in (2, 4, 7):
            P = ds.tier_outline((0, 0), (0, -400), 220, tiers, seed=seed)
            n = len(P)
            for i in range(n):
                a, b = P[i], P[(i + 1) % n]
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    c, d = P[j], P[(j + 1) % n]
                    assert not (ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)), (seed, tiers, i, j)


def test_band_tiers_give_each_tree_its_own_shape(draw):
    ok(draw("new", "lake", "--size", "400x300", "--request", "a lake with pine trees, painterly"))
    ok(draw("do", "plan subject lake --box 0,400,1000,700; plan subject pines --box 0,540,1000,1000; "
                  "plan light left; palette make; palette pick 1"))
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    for x in (120, 340, 600, 860):
        ok(draw("band", f"pines-{x}", f"{x},990 {x},560", "--tiers", "5", "--width", "220", "--color", "pines"))
    p = project("lake")
    crops = []
    for x in (120, 340, 600, 860):
        m = p.region_mask(f"pines-{x}")
        ys, xs = np.nonzero(m)
        crops.append(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    sizes = {c.shape for c in crops}
    assert len(sizes) >= 3                              # not four copies of one cone


# ============================================================================================ ridges

def test_even_zigzag_mountains_get_an_irregular_suggestion(draw):
    ok(draw("new", "lake", "--size", "400x300", "--request", "A mountain lake at sunset, painterly."))
    ok(draw("do", "plan subject sky --box 0,0,1000,460; plan subject mountains --box 0,230,1000,540; "
                  "plan light left; palette make; palette pick 1"))
    ok(draw("stage", "done"))
    ok(draw("stage", "done", "--because", "test"))
    zig = "0,430 110,350 230,400 350,300 470,370 590,270 700,350 820,300 910,380 1000,345 1000,495 0,495"
    txt = ok(draw("band", "mountains", zig, "--closed", "--color", "mountains"))
    assert "regular zigzag" in txt
    sug = txt.split("pixelcanvas band mountains \"", 1)[1].split("\"", 1)[0]
    pts = np.array([[float(v) for v in t.split(",")] for t in sug.split()])
    cv_gap, cv_h, n = ds.ridge_regularity(pts)
    assert cv_gap >= 0.15 and cv_h >= 0.12 and n >= 4
    # the suggestion itself draws without a note
    txt2 = ok(draw("band", "mountains", sug, "--closed", "--color", "mountains"))
    assert "regular zigzag" not in txt2


def test_irregular_ridge_has_a_dominant_peak_and_uneven_spacing():
    for seed in range(20):
        pts = ds.irregular_ridge((0, 200, 1000, 500), seed)
        assert pts == ds.irregular_ridge((0, 200, 1000, 500), seed)
        top = np.array(pts[:-2], float)
        assert all(0 <= x <= 1000 and 200 <= y <= 500 for x, y in pts)
        ys = np.sort(top[:, 1])
        assert ys[1] - ys[0] > 5                        # one peak stands above the rest
        reg = ds.ridge_regularity(np.array(pts, float))
        if reg is not None:
            assert reg[0] > 0.15 or reg[1] > 0.12
