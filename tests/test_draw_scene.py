"""Painterly brushwork and scene dressing without a reference: brush_ops marks and wet pick-up, clouds, water
reflections and a sun path, foliage, grass tufts, atmosphere, glows and wet-ground streaks, a floor with contact
shadows, windows."""
from __future__ import annotations

import gzip
import json
import re

import numpy as np

from pixelcanvas import cli
from pixelcanvas import draw_core as dc
from pixelcanvas import draw_detail
from pixelcanvas.ops import Fill, Stroke, check_mark_rules, ops_from_jsonable

from .test_draw_flow import draw, ok  # noqa: F401  (fixture and helpers)

SCENE = ("cloud", "reflect", "foliage", "tufts", "atmosphere", "glow", "wet", "ground", "window")


# ============================================================================================ helpers

def project(name):
    from pixelcanvas.project import Project
    return Project.find(f"{name}.pixelcanvas")


def step_ops(name, text):
    """The ops of the step a report names (first 'step N' in the text)."""
    n = int(re.search(r"step (\d+)", text).group(1))
    with gzip.open(f"{name}.pixelcanvas/ops/{n:04d}.json.gz", "rt", encoding="utf-8") as f:
        data = json.load(f)
    return ops_from_jsonable(data["ops"]), data["layer"]


def lum(img):
    img = np.asarray(img, np.float64)
    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]


def setup_lake(run, light="back --from 600,420"):
    ok(run("new", "lake", "--size", "400x300", "--request", "a mountain lake at sunset with pines, painterly"))
    ok(run("do", "plan subject sky --box 0,0,1000,560; plan subject mountains --box 0,200,1000,560; "
                 "plan subject lake --box 0,540,1000,860; plan subject pines --box 0,150,330,1000; "
                 f"plan subject shore --box 0,820,1000,1000; plan light {light}; palette make --base orange; "
                 "palette pick 1"))
    ok(run("palette", "set", "sky", "#f4a66a", "mountains", "#91627a", "lake", "#6a5a8c", "pines", "#22382f",
           "shore", "#694b2b"))
    ok(run("do", 'band sky "0,560 1000,560" --above --color sky --color2 "#f6c88a"; '
                 'band mountains "0,420 120,300 210,330 330,190 450,280 560,230 700,330 820,260 1000,380" --below '
                 '--color mountains; '
                 'band lake "0,560 1000,560" --below --color lake; '
                 'band shore "0,880 300,850 600,870 1000,840" --below --color shore; '
                 'band pines "90,230 112,330 100,330 138,430 116,430 162,550 136,550 188,690 158,690 205,860 '
                 '0,860 22,690 0,690 44,550 18,550 64,430 42,430 80,330 68,330" --closed --color pines'))


def rect_mask():
    m = np.zeros((240, 320), bool)
    m[40:200, 60:260] = True
    return m


# ============================================================================================ brush_ops

def _brush(under=None, seed=3):
    m = rect_mask()
    field = draw_detail.angle_field(0.0, m.shape)
    light = dc.light_vec({"light": {"deg": 135, "z": 0.6}})
    kw = {} if under is None else {"under": under}
    return draw_detail.brush_ops(m, field, (200, 40, 0), (150, 20, 0), (240, 90, 0), light, 12.0, "flat", 1.0,
                                 seed=seed, **kw)


def test_brush_ops_marks_are_tapered_curved_and_soft():
    ops = _brush()
    assert len(ops) > 40 and not check_mark_rules(ops)
    # every mark: a soft body at the stroke's opacity, the bristles' streaks inside it at half of it
    bodies, streaks = ops[0::2], ops[1::2]
    assert len(bodies) == len(streaks)
    for b, s in zip(bodies, streaks):
        assert b.brush == "soft" and s.brush == "flat" and np.array_equal(b.pts, s.pts)
        assert 0.35 - 1e-6 <= b.opacity <= 0.7 + 1e-6 and abs(s.opacity - 0.5 * b.opacity) < 1e-6
        assert np.max(s.width) < np.max(b.width)
    tapered = curved = 0
    for o in ops:
        assert isinstance(o, Stroke)
        w = np.asarray(o.width)
        assert w.ndim == 1 and len(w) == len(o.pts)          # a pressure profile, not one width
        if w[0] < 0.75 * w.max() and w[-1] < 0.4 * w.max():
            tapered += 1
        a, b = o.pts[0].astype(float), o.pts[-1].astype(float)
        ch = b - a
        nc = np.hypot(*ch)
        if nc > 1:
            dev = np.abs((o.pts[:, 0] - a[0]) * ch[1] - (o.pts[:, 1] - a[1]) * ch[0]) / nc
            if dev.max() > 0.6:
                curved += 1
    assert tapered == len(ops)                                 # every mark lands narrow and lifts to a point
    assert curved > 0.5 * len(ops)                             # most marks arc a little


def test_brush_ops_wet_pickup_mixes_the_colour_under():
    dry = _brush()
    assert max(o.color[2] for o in dry) < 6                   # the region's colours have no blue
    under = np.zeros((240, 320, 3), np.uint8)
    under[..., 2] = 255
    wet = _brush(under=under)
    blues = [o.color[2] for o in wet]
    assert min(blues) >= 0.25 * 255 - 3 and max(blues) <= 0.4 * 255 + 3


def test_brush_ops_follow_the_values_already_painted():
    # a sky band painted light toward the horizon (the bottom) while the light comes from the top left: the strokes
    # keep the band's values (lighter at the bottom), they do not lay the dome's dark side over the light horizon
    under = np.zeros((240, 320, 3), np.uint8)
    t = np.linspace(0, 1, 240)[:, None]
    under[..., 0] = (150 + 90 * t).astype(np.uint8)
    under[..., 1] = (20 + 70 * t).astype(np.uint8)
    ops = _brush(under=under)
    bodies = ops[0::2]
    ys = np.array([float(np.mean(o.pts[:, 1])) for o in bodies])
    L = np.array([0.2126 * o.color[0] + 0.7152 * o.color[1] + 0.0722 * o.color[2] for o in bodies])
    top, bottom = L[ys < np.quantile(ys, 0.3)].mean(), L[ys > np.quantile(ys, 0.7)].mean()
    assert bottom > top + 8, (top, bottom)


def test_brush_ops_old_call_still_works():
    m = rect_mask()
    field = draw_detail.angle_field(30.0, m.shape)
    light = dc.light_vec({"light": {"deg": 90, "z": 0.6}})
    ops = draw_detail.brush_ops(m, field, (90, 120, 60), (60, 80, 40), (140, 170, 100), light, 10.0, "dry", 1.0, 7)
    assert ops and {o.brush for o in ops[1::2]} == {"dry"} and not check_mark_rules(ops)
    ops = draw_detail.brush_ops(m, field, (90, 120, 60), (60, 80, 40), (140, 170, 100), light, 10.0, "round", 1.0, 7)
    assert ops and all(o.brush == "round" for o in ops)         # any other tip: one stroke per mark


# ============================================================================================ install

def test_scene_commands_are_installed_with_examples(draw):
    for name in SCENE:
        assert name in cli.COMMANDS, name
        assert cli.USAGE[name].startswith(f"pixelcanvas {name}"), name


# ============================================================================================ clouds

def _cloud_delta(run, light):
    setup_lake(run, light)
    p = project("lake")
    before = lum(p.composite_u8("export"))
    text = ok(run("cloud", "250,150 400,140 550,160", "--size", "70"))
    after = lum(project("lake").composite_u8("export"))
    dl = after - before
    W, H = 400, 300
    size = 70 * max(W, H) / 1000.0
    cy = 0.15 * H
    xs = slice(int(0.3 * W), int(0.5 * W))
    upper = dl[int(cy - 0.45 * size):int(cy - 0.15 * size), xs].mean()
    lower = dl[int(cy + 0.05 * size):int(cy + 0.35 * size), xs].mean()
    return text, upper, lower


def test_cloud_underside_lit_by_a_low_sun(draw):
    text, upper, lower = _cloud_delta(draw, "back --from 600,420")
    assert "underside lit" in text
    assert lower > upper + 2
    ops, layer = step_ops("lake", text)
    assert layer == "paint" and not check_mark_rules(ops)
    assert sum(1 for o in ops if isinstance(o, Stroke) and o.brush == "soft") > 0.8 * len(ops)   # soft dabs
    assert any(isinstance(o, Stroke) and len(o.pts) > 4 for o in ops)                          # and wisps


def test_cloud_top_lit_by_a_high_light(draw):
    text, upper, lower = _cloud_delta(draw, "top-left")
    assert "top lit" in text
    assert upper > lower + 2


def test_cloud_is_clipped_to_the_sky_and_undoes(draw):
    setup_lake(draw)
    p = project("lake")
    before = p.layer("paint").copy()
    sky = p.region_mask("sky")
    text = ok(draw("cloud", "500,250", "--size", "100"))      # on the ridge: half over the mountains
    assert "clipped to sky" in text
    after = project("lake").layer("paint")
    changed = np.abs(after - before).max(-1) > 2 / 255
    assert changed.sum() > 200 and not (changed & ~sky).any()
    ok(draw("undo"))
    assert np.array_equal(project("lake").layer("paint"), before)


# ============================================================================================ reflections

def test_reflect_mirrors_the_scene_and_lays_a_sun_path(draw):
    setup_lake(draw)
    text = ok(draw("reflect", "lake"))
    assert "sun path" in text
    ops, layer = step_ops("lake", text)
    assert not check_mark_rules(ops)
    strokes = [o for o in ops if isinstance(o, Stroke) and len(o.pts) > 1]
    flat = [abs(o.pts[-1, 0] - o.pts[0, 0]) > 2 * abs(o.pts[-1, 1] - o.pts[0, 1]) for o in strokes]
    assert np.mean(flat) > 0.95                                # horizontal broken strokes
    H = 300
    near = [o for o in strokes if o.pts[:, 1].mean() > 0.78 * H]
    far = [o for o in strokes if o.pts[:, 1].mean() < 0.62 * H]
    assert near and far
    assert np.mean([np.max(o.width) for o in near]) > 1.8 * np.mean([np.max(o.width) for o in far])
    assert np.mean([np.ptp(o.pts[:, 0]) for o in near]) > np.mean([np.ptp(o.pts[:, 0]) for o in far])
    # the sun path: light marks in a column under the sun (x 600 of 1000)
    p = project("lake")
    img = lum(p.composite_u8("export"))
    row = slice(int(0.7 * H), int(0.8 * H))
    assert img[row, 230:250].mean() > img[row, 330:370].mean() + 8
    # no hard edge: the path fades out sideways
    lake = p.region_mask("lake")
    band = img[row]
    prof = np.array([band[:, x][lake[row, x]].mean() for x in range(200, 300)])
    assert np.abs(np.diff(prof)).max() < 25


def test_reflect_without_glow_and_with_horizon(draw):
    setup_lake(draw, "top-left")
    text = ok(draw("reflect", "lake", "--horizon", "560"))
    assert "sun path" not in text
    text = ok(draw("reflect", "lake", "--glow", "300"))
    assert "sun path at x 300" in text


# ============================================================================================ foliage

def test_foliage_layers_marks_and_breaks_the_silhouette(draw):
    setup_lake(draw, "left")
    p = project("lake")
    pines = p.region_mask("pines")
    before = p.layer("paint").copy()
    text = ok(draw("foliage", "pines"))
    assert "needle" in text
    ops, layer = step_ops("lake", text)
    assert not check_mark_rules(ops)
    after = project("lake").layer("paint")
    changed = np.abs(after - before).max(-1) > 2 / 255
    assert (changed & pines).mean() > 0.5 * pines.mean()      # the whole tree is worked
    assert (changed & ~pines).sum() > 30                       # marks poke out past the silhouette
    assert any(getattr(o, "own", False) for o in ops)
    # lit side (left: the light) lighter than the far side
    img = lum(project("lake").composite_u8("export"))
    ys, xs = np.nonzero(pines)
    cx = np.median(xs)
    left = img[pines & (np.arange(400)[None, :] < cx - 5)].mean()
    right = img[pines & (np.arange(400)[None, :] > cx + 5)].mean()
    assert left > right + 3


def test_foliage_round_crown_with_leaves(draw):
    ok(draw("new", "park", "--size", "300x300", "--request", "a round tree in a park, painterly"))
    ok(draw("do", "plan subject tree --box 200,100,800,700; plan light top-right; palette make; palette pick 1"))
    ok(draw("band", "crown", "500,150 680,220 740,400 650,580 500,620 330,580 260,400 320,220", "--closed",
            "--color", "#3f6a35"))
    text = ok(draw("foliage", "crown", "--kind", "leaf"))
    assert "leaf" in text
    ops, _ = step_ops("park", text)
    assert len(ops) > 40 and not check_mark_rules(ops)


# ============================================================================================ tufts

def test_tufts_grow_in_the_near_part_only(draw):
    setup_lake(draw)
    p = project("lake")
    shore = p.region_mask("shore")
    ys = np.nonzero(shore)[0]
    y0, y1 = ys.min(), ys.max() + 1
    text = ok(draw("tufts", "shore", "--count", "40"))
    ops, _ = step_ops("lake", text)
    assert not check_mark_rules(ops)
    blades = [o for o in ops if isinstance(o, Stroke)]
    bases = np.array([o.pts[0] for o in blades])
    assert bases[:, 1].min() >= y0 + (y1 - y0) / 3 - 2          # nothing in the far top third
    # 3-5 blades from one base
    per_base = {}
    for b in bases:
        k = (round(float(b[0]), 2), round(float(b[1]), 2))
        per_base[k] = per_base.get(k, 0) + 1
    assert set(per_base.values()) <= {3, 4, 5}
    tufts = re.search(r"(\d+) tufts", text)
    assert tufts and 20 <= int(tufts.group(1)) <= 40
    lens = np.array([np.hypot(*(o.pts[-1] - o.pts[0])) for o in blades])
    assert np.corrcoef(bases[:, 1], lens)[0, 1] > 0.3        # nearer (lower) tufts are bigger
    # more tufts low than high in the near part
    mid = y0 + (y1 - y0) * 2 / 3
    assert (bases[:, 1] > mid).sum() > (bases[:, 1] <= mid).sum()


# ============================================================================================ atmosphere, glow, wet

def test_atmosphere_warms_toward_the_light_and_darkens_the_edges(draw):
    setup_lake(draw)
    p = project("lake")
    before = p.composite_u8("export").astype(float)
    text = ok(draw("atmosphere"))
    ops, layer = step_ops("lake", text)
    assert layer == "fx" and ops and all(isinstance(o, Fill) for o in ops)
    p = project("lake")
    fx = p.layer("fx")
    assert fx[..., 3].max() < 0.5                               # a low-opacity veil
    after = p.composite_u8("export").astype(float)
    sun = (slice(110, 140), slice(225, 255))                    # around the light at 600,420
    corner = (slice(0, 25), slice(0, 25))
    warm = lambda a, s: (a[s][..., 0] - a[s][..., 2]).mean()   # noqa: E731
    # warm and light toward the light, cool and dark at the edges
    assert lum(after[sun]).mean() > lum(before[sun]).mean() + 1
    assert lum(after[corner]).mean() < lum(before[corner]).mean() - 2
    assert warm(after, sun) - warm(after, corner) > warm(before, sun) - warm(before, corner) + 3


def test_glow_halos_fade_outward(draw):
    setup_lake(draw)
    text = ok(draw("glow", "300,400 700,380", "--radius", "40", "--color", "#ffd27a"))
    ops, layer = step_ops("lake", text)
    assert layer == "fx" and len(ops) >= 6
    fx = project("lake").layer("fx")
    cx, cy = 120, 120
    a = fx[..., 3]
    assert a[cy, cx] > a[cy, cx + 10] > a[cy, cx + 20] >= 0
    assert a[cy, cx] > 0.2


def test_wet_streaks_run_down_below_the_lights(draw):
    setup_lake(draw)
    p = project("lake")
    before = p.layer("fx").copy()
    text = ok(draw("wet", "lake", "--from", "300,450"))
    ops, layer = step_ops("lake", text)
    assert layer == "fx" and not check_mark_rules(ops)
    strokes = [o for o in ops if isinstance(o, Stroke) and len(o.pts) > 1]
    vert = [o for o in strokes if abs(o.pts[-1, 1] - o.pts[0, 1]) > 2 * abs(o.pts[-1, 0] - o.pts[0, 0])]
    assert len(vert) >= 3                                       # broken vertical dashes
    allp = np.concatenate([o.pts for o in strokes])
    assert np.ptp(allp[:, 1]) > 1.5 * np.ptp(allp[:, 0])       # the streak runs down below the light
    assert abs(np.median(allp[:, 0]) - 120) < 12
    after = project("lake").layer("fx")
    changed = np.abs(after - before).max(-1) > 2 / 255
    assert changed.any() and not (changed & ~p.region_mask("lake")).any()
    code, err = draw("wet", "shore")
    assert code == 2 and "EXAMPLE:" in err


# ============================================================================================ ground, window

def test_ground_band_and_contact_shadow(draw):
    ok(draw("new", "room", "--size", "400x300", "--request", "a cat sitting on the floor, painterly"))
    ok(draw("do", "plan subject cat --box 350,450,650,950; plan light top-left; palette make; palette pick 1"))
    text = ok(draw("ground", "--y", "800"))
    assert "contact shadow under cat" in text
    p = project("room")
    assert "ground" in p.region_names()
    img = lum(p.composite_u8("export"))
    y = int(0.94 * 300)
    under = img[y - 3:y + 3, 205:225].mean()                   # under the cat, pushed right (light top-left)
    away = img[y - 3:y + 3, 30:60].mean()
    assert under < away - 6
    left = img[y - 3:y + 3, 150:165].mean()
    right = img[y - 3:y + 3, 245:260].mean()
    assert right < left                                         # the shadow leans away from the light
    ok(draw("ground", "--y", "820", "--line", "--name", "floor2"))


def test_window_frame_panes_glass_and_sheen(draw):
    ok(draw("new", "house", "--size", "400x400", "--request", "a house wall with a window, painterly"))
    ok(draw("do", "plan subject wall --box 0,0,1000,1000; plan light top-left; palette make; palette pick 1"))
    text = ok(draw("window", "--box", "300,300,700,700", "--panes", "2x2", "--color", "#f0ece0", "--glass",
                   "#7090b0"))
    assert "2x2" in text
    ops, layer = step_ops("house", text)
    assert not check_mark_rules(ops)
    p = project("house")
    img = p.composite_u8("export").astype(float)
    # the frame and the middle bars in the frame colour, the panes in the glass colour
    assert np.abs(img[121, 200] - (240, 236, 224)).max() < 40
    assert np.abs(img[200, 200] - (240, 236, 224)).max() < 40
    pane = img[135:192, 135:192]
    assert pane[..., 2].mean() > pane[..., 0].mean() + 20      # blue-grey glass
    # the sheen: a lighter diagonal band across a pane, soft (no hard step)
    L = lum(pane)
    assert L.max() - np.median(L) > 15
    assert np.abs(np.diff(L, axis=1)).max() < 60


def test_window_stays_behind_what_is_in_front_and_takes_the_region_colour(draw):
    ok(draw("new", "sill", "--size", "400x400", "--request", "a cat on a sunny windowsill, painterly"))
    ok(draw("do", "plan subject wall --box 0,0,1000,1000; plan subject window --box 250,150,750,650; "
                  "plan subject cat --box 380,450,620,700; plan light top-left; palette make; palette pick 1; "
                  'palette set window "#e8b070" cat "#303030"'))
    ok(draw("band", "window", "250,150 750,150 750,650 250,650", "--closed", "--color", "window"))
    ok(draw("band", "cat", "420,700 400,560 450,480 550,480 600,560 580,700", "--closed", "--color", "cat"))
    p = project("sill")
    cat = p.region_mask("cat")
    before = p.composite_u8("export").astype(float)
    text = ok(draw("window", "--panes", "2x2"))                 # no box: the window region's own box
    assert "behind cat" in text and "in region window" in text
    after = project("sill").composite_u8("export").astype(float)
    changed = np.abs(after - before).max(-1) > 2
    assert changed.sum() > 1000 and not (changed & cat).any()
    pane = after[85:150, 125:190]                               # inside the top-left pane, clear of the cat
    assert (pane[..., 0] - pane[..., 2]).mean() > 40            # warm glass: the window's own colour
    L = lum(pane)
    assert L.max() > np.median(L) + 10                          # a lighter sheen across it
    assert "in region window" in ok(draw("window", "--box", "250,150,750,650", "--panes", "1x2"))  # by its box
    assert "in region window" in ok(draw("window", "--box", "280,180,480,420", "--panes", "1x1"))  # a box inside it
    code, err = draw("window", "--box", "100,100,200,200", "--region", "nowhere")
    assert code == 2 and "EXAMPLE:" in err


def test_cloud_glow_atmosphere_stay_behind_the_subjects(draw):
    ok(draw("new", "yard", "--size", "400x300", "--request", "a tree in a yard, painterly"))
    ok(draw("do", "plan subject tree --box 300,100,700,900; plan light top-left; palette make; palette pick 1"))
    ok(draw("band", "back", "0,0 1000,0 1000,1000 0,1000", "--closed", "--color", "#8fb8e0"))
    ok(draw("band", "tree", "500,100 700,500 600,900 400,900 300,500", "--closed", "--color", "#3f6a35"))
    tree = project("yard").region_mask("tree")
    for argv in (("cloud", "500,300", "--size", "200"), ("glow", "150,300", "--radius", "300"), ("atmosphere",)):
        before = project("yard").composite_u8("export").astype(float)
        text = ok(draw(*argv))
        after = project("yard").composite_u8("export").astype(float)
        changed = np.abs(after - before).max(-1) > 2
        assert changed.any() and not (changed & tree).any(), argv
        assert "behind tree" in text, argv
    before = project("yard").composite_u8("export").astype(float)
    ok(draw("glow", "150,300", "--radius", "300", "--over"))    # asked: over the subjects too
    after = project("yard").composite_u8("export").astype(float)
    assert ((np.abs(after - before).max(-1) > 2) & tree).any()


def test_scene_errors_are_plain_with_an_example(draw):
    setup_lake(draw)
    for argv in (("cloud",), ("reflect", "nowhere"), ("foliage", "pines", "--kind", "cactus"), ("tufts",),
                 ("glow",), ("window",), ("window", "--box", "300,300,700,700", "--panes", "zz"),
                 ("atmosphere", "--warm", "x")):
        code, text = draw(*argv)
        assert code == 2 and text.startswith("ERROR:") and "EXAMPLE: pixelcanvas" in text, (argv, text)
