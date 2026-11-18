"""Round 4 engine techniques: ink masses as outlined fills, airbrush shading (Spray), textured tips and directed
texture strokes.  Every new mark obeys the mark rules (DESIGN.md 10.1) and round-trips through JSON."""
import cv2
import numpy as np

from pixelcanvas.airbrush import airbrush_ops, flat_areas, step_edges
from pixelcanvas.brush import PRESETS, apply_ops, new_layer, op_bbox
from pixelcanvas.masses import fit_ring, ink_coverage, ink_mass_ops, line_quality
from pixelcanvas.ops import (Fill, Spray, Stroke, check_mark_rules, mark_area, op_anchor, ops_from_jsonable,
                                   ops_to_jsonable, owned_ops)
from pixelcanvas.stage_ink import ink_ops
from pixelcanvas.stage_paint import paint_ops, texture_ops
from pixelcanvas.stage_shade import shade_ops


def _render(ops, h, w, bg=255):
    lay = new_layer(h, w)
    lay[..., :3] = bg / 255.0
    lay[..., 3] = 1.0
    apply_ops(lay, ops)
    return np.clip(np.rint(lay[..., :3] * 255), 0, 255).astype(np.uint8)


def ink_ref(h=160, w=200):
    """A pen drawing: antialiased black lines of several weights, a solid black mass with a white gap, grey tone."""
    ss = 4
    big = np.full((h * ss, w * ss), 255, np.uint8)
    cv2.rectangle(big, (20 * ss, 100 * ss), (90 * ss, 140 * ss), 200, -1)            # a grey tone area
    for k, wd in enumerate((1.2, 2.0, 3.5)):
        y = (20 + 18 * k) * ss
        cv2.line(big, (15 * ss, y), (180 * ss, y + 25 * ss), 0, max(1, int(wd * ss)), cv2.LINE_AA)
    cv2.ellipse(big, (140 * ss, 115 * ss), (30 * ss, 22 * ss), 0, 0, 360, 0, -1, cv2.LINE_AA)  # solid mass
    cv2.line(big, (115 * ss, 115 * ss), (165 * ss, 112 * ss), 255, int(1.5 * ss), cv2.LINE_AA)  # white gap in it
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return np.repeat(small[..., None], 3, -1)


def test_ink_masses_are_legal_crisp_and_dark():
    ref = ink_ref()
    h, w = ref.shape[:2]
    st = {}
    ops, foot = ink_mass_ops(ref, stats=st)
    assert ops and not check_mark_rules(ops)
    assert any(isinstance(o, Fill) and len(o.polygons) >= 1 for o in ops)
    img = _render(ops, h, w)
    q = line_quality(ref, img)
    # the ink is as dark as the reference's and the edges as sharp (no grey, blurred copy)
    assert abs(q["core_L_img"] - q["core_L_ref"]) < 8.0
    assert q["edge_grad_img"] > 0.8 * q["edge_grad_ref"]
    assert q["dark_iou"] > 0.75
    # the grey tone is not ink
    assert not foot[110:130, 30:80].any()
    # the white gap inside the solid mass stays white
    assert img[114, 125:155].astype(int).mean() > 150


def test_ink_masses_json_round_trip_and_regions():
    ref = ink_ref()
    ops = ink_ops(ref, None, method="masses")
    back = ops_from_jsonable(ops_to_jsonable(ops))
    h, w = ref.shape[:2]
    assert np.array_equal(_render(ops, h, w), _render(back, h, w))
    # region calls over a partition paint what the whole call paints
    y, x = np.mgrid[0:h, 0:w]
    a = x < 100
    parts = [ink_ops(ref, m, method="masses") for m in (a, ~a)]
    assert len(parts[0]) + len(parts[1]) == len(ops)


def test_fit_ring_keeps_corners_and_is_smooth():
    sq = np.zeros((80, 80), np.uint8)
    cv2.rectangle(sq, (10, 10), (69, 69), 1, -1)
    cs, _ = cv2.findContours(sq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    r = fit_ring(cs[0][:, 0, :], 4)
    assert r is not None and not check_mark_rules([Fill(polygons=[r], color=(0, 0, 0))])
    # a corner of the square stays within half a pixel of the true corner
    d = np.hypot(r[:, 0] - (10 + 0.5) / 4, r[:, 1] - (10 + 0.5) / 4).min()
    assert d < 0.5


def test_spray_op_rules_json_and_render():
    sp = Spray(pts=[[20, 20], [30, 22]], size=[16, 8], colors=[[200, 30, 30], [30, 30, 200]], opacity=[0.5, 0.4],
               polygons=[np.array([[5, 5], [45, 5], [45, 40], [5, 40]], np.float32)], feather=0.75)
    assert not check_mark_rules([sp])
    assert mark_area(sp) > 12
    fine = Spray(pts=[[20, 20]], size=2.0, colors=[[0, 0, 0]], opacity=0.5)
    assert not check_mark_rules([fine]) and check_mark_rules([fine], min_level=3)   # a level-2 dab
    back = ops_from_jsonable(ops_to_jsonable([sp]))[0]
    assert isinstance(back, Spray) and np.array_equal(back.colors, sp.colors)
    a, b = _render([sp], 50, 60), _render([back], 50, 60)
    assert np.array_equal(a, b)
    # the frisket holds the spray: nothing outside it
    assert (a[:, 47:] == 255).all() and (a[42:] == 255).all()
    assert op_bbox(sp, (50, 60)) is not None
    x, y = op_anchor(sp)
    assert 5 <= x <= 45 and 5 <= y <= 40


def _soft_ref(h=120, w=160):
    """A flat-coloured disc with a soft radial blush in it and a hard-edged shadow, on white."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    ref = np.full((h, w, 3), 250, np.float32)
    disc = (x - 80) ** 2 + (y - 60) ** 2 < 50 ** 2
    blush = np.exp(-((x - 65) ** 2 + (y - 55) ** 2) / (2 * 14.0 ** 2))
    skin = np.array([250, 225, 210], np.float32)
    red = np.array([235, 150, 150], np.float32)
    col = skin[None, None] * (1 - blush[..., None]) + red[None, None] * blush[..., None]
    ref[disc] = col[disc]
    shadow = disc & (x > 105)
    ref[shadow] = [200, 170, 160]
    return np.clip(ref, 0, 255).astype(np.uint8), disc


def test_airbrush_builds_a_smooth_ramp_without_bands():
    ref, disc = _soft_ref()
    h, w = ref.shape[:2]
    base = np.full_like(ref, 250)
    base[disc] = (250, 225, 210)
    st = {}
    ops = airbrush_ops(ref, base, disc.astype(np.int32), stats=st)
    assert ops and all(isinstance(o, Spray) for o in ops) and not check_mark_rules(ops)
    img = _render(ops, h, w)
    img[~disc] = base[~disc]
    lay = new_layer(h, w)
    lay[..., :3] = base / 255.0
    lay[..., 3] = 1
    apply_ops(lay, ops)
    out = np.clip(np.rint(lay[..., :3] * 255), 0, 255).astype(np.uint8)
    from pixelcanvas.trace import delta_e
    blush_area = disc & (np.hypot(*np.mgrid[0:h, 0:w][::-1] - np.array([65, 55])[:, None, None]) < 25)
    assert float(delta_e(ref, out)[blush_area].mean()) < 0.5 * float(delta_e(ref, base)[blush_area].mean())
    # no bands: the second difference along a row through the blush stays small
    row = out[55, 45:85].astype(np.float32).mean(1)
    assert np.abs(np.diff(row, 2)).max() < 6.0


def test_shade_keeps_hard_shadow_as_fill_and_airbrushes_the_ramp():
    ref, disc = _soft_ref()
    base = np.full_like(ref, 250)
    base[disc] = (250, 225, 210)
    ops = shade_ops(ref, None, base, levels=4, air="on")
    assert not check_mark_rules(ops)
    assert any(isinstance(o, Spray) for o in ops)
    assert any(isinstance(o, Fill) for o in ops)     # the hard-edged shadow stays a fill
    ops_off = shade_ops(ref, None, base, levels=4, air="off")
    assert not any(isinstance(o, Spray) for o in ops_off)


def test_textured_tips_are_legal_and_break_coverage():
    for name in ("bristle", "sponge"):
        assert name in PRESETS
        st = Stroke(np.array([[5, 10], [95, 12]], np.float32), 6.0, (40, 20, 10), brush=name, seed=3)
        assert not check_mark_rules([st])
        img = _render([st], 24, 100)
        core = img[8:14, 15:85, 0].astype(int)
        # one colour, broken coverage: some pixels of the footprint stay light
        assert (core < 120).mean() > 0.2 and (core > 180).mean() > 0.05


def _streak_ref(h=100, w=140):
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = (x * 0.35 + y) / 3.0 + rng.normal(0, 0.05, (h, w))
    L = 150 + 40 * np.sign(np.sin(phase * np.pi))
    ref = np.stack([L, L * 0.7, L * 0.4], -1)
    return np.clip(cv2.GaussianBlur(ref, (0, 0), 0.6), 0, 255).astype(np.uint8)


def test_structure_strokes_and_texture_marks():
    ref = _streak_ref()
    h, w = ref.shape[:2]
    flat = np.full_like(ref, tuple(int(v) for v in ref.reshape(-1, 3).mean(0)))
    ops = paint_ops(ref, None, flat, size=3, brush="round", structure=True, tips="auto", seed=1)
    assert ops and not check_mark_rules(ops)
    assert any(o.brush == "bristle" for o in ops)
    tex = texture_ops(ref, None, flat)
    assert tex and not check_mark_rules(tex)
    # the marks run along the streaks (direction (1, -0.35) normalised, up to sign)
    d = np.array([1.0, -0.35]) / np.hypot(1.0, 0.35)
    along = []
    for o in tex:
        v = o.pts[-1] - o.pts[0]
        n = np.hypot(*v)
        if n > 6:
            along.append(abs(float(v @ d)) / n)
    assert np.median(along) > 0.9
    # a region call returns the whole call's marks anchored in the region
    a = np.zeros((h, w), bool)
    a[:, : w // 2] = True
    assert len(texture_ops(ref, a, flat)) + len(texture_ops(ref, ~a, flat)) == len(tex)


def test_flat_areas_merge_unlined_neighbours():
    base = np.full((60, 80, 3), 200, np.uint8)
    base[:, 40:] = (180, 170, 160)              # a second flat next to the first, no line between
    line = np.zeros((60, 80), bool)
    assert flat_areas(base, min_area=50).max() == 2
    assert flat_areas(base, min_area=50, line_mask=line).max() == 1
    line[:, 39:41] = True                       # a line between them keeps them apart
    assert flat_areas(base, min_area=50, line_mask=line).max() == 2
    assert step_edges(base).any()
