"""easel locks: hair built from overlapping tapered locks, cut where a lock in front hides them, one editable
line record per visible piece; locks shine: per-lock shadow, a shine band that bends with the head, a few strands."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi

from .test_draw_flow import draw, ok  # noqa: F401  (fixture and helpers)

CUTS = ("long", "bob", "short", "spiky", "twin", "ponytail", "wavy")


def state():
    return json.load(open("pic.easel/state.json", encoding="utf-8"))


def lock_lines(st, part="hair"):
    return [r for r in st["draw"]["lines"] if str(r.get("src", "")).startswith(f"locks {part}")]


def setup_head(run, yaw=12, figure=True, head=True, size="480x600"):
    """The girl of the trial pictures: a bust, head construction turned a little to the right."""
    ok(run("new", "pic", "--size", size, "--request", "anime girl with long silver hair"))
    ok(run("do", "plan subject girl --box 110,40,890,1000; plan part hair --box 150,40,850,920; "
                 "plan part face --box 340,170,660,480; plan part eye-l --box 395,310,470,370; "
                 "plan part eye-r --box 530,310,605,370; plan part neck --box 440,440,560,580; "
                 "plan part coat --box 130,580,870,1000; plan light top-left; palette make; palette pick 1; "
                 "stage done"))
    if figure:
        ok(run("do", "sketch figure girl --pose bust; sketch joint s1 head 500,270 chin 500,470 lsh 215,620 "
                     "rsh 785,620"))
    if head:
        ok(run("sketch", "head", "girl", "--yaw", str(yaw), *(() if figure else ("--box", "300,70,700,470"))))
    ok(run("stage", "done", "--because", "test"))


def head_el(st):
    return next(e for e in st["draw"]["sketch"] if e["kind"] == "head")


def test_locks_make_line_records_of_the_hair_part(draw):
    setup_head(draw)
    text = ok(draw("locks", "hair", "--cut", "long", "--part", "520,85"))
    assert "locks" in text and "VIEW:" in text
    st = state()
    recs = lock_lines(st)
    assert len(recs) >= 16
    assert all(r["part"] == "hair" for r in recs)
    weights = {r["weight"] for r in recs}
    assert 1.0 in weights and 1.4 in weights           # inner lock lines and the outer silhouette
    rec = st["draw"]["locks"]["hair"]
    assert rec["cut"] == "long" and rec["part"] == [520.0, 85.0]
    layers = [k["layer"] for k in rec["locks"]]
    assert {"back", "side", "crown", "fringe"} <= set(layers)
    for k in rec["locks"]:
        assert len(k["path"]) >= 2 and len(k["width"]) == len(k["path"]) and "order" in k
    assert sorted(rec["lines"]) == sorted(r["id"] for r in recs)
    line = np.asarray(Image.open("pic.easel/layers/line.png"))
    assert (line[..., 3] > 0).mean() > 0.02


def test_no_hair_line_crosses_the_face(draw):
    setup_head(draw)
    ok(draw("locks", "hair", "--cut", "long", "--part", "520,85"))
    st = state()
    parts = st["draw"]["parts"]
    el = head_el(st)
    # the face inside its outline: between the corners of the jaw where  line trace  starts the face outline
    # (the head construction's numbers), from the eye tops to near the chin
    bx0, by0, bx1, by1 = el["box"]
    w = bx1 - bx0
    sy = np.sin(np.radians(el.get("yaw", 0)))
    x0 = bx0 + (0.08 + max(0, -sy) * 0.12) * w + 4
    x1 = bx1 - (0.08 + max(0, sy) * 0.12) * w - 4
    y0 = min(parts["eye-l"]["box"][1], parts["eye-r"]["box"][1]) + 4
    y1 = by1 - 0.12 * (by1 - by0)
    for r in lock_lines(st):
        pts = np.asarray(r["path"], float)
        inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
        assert not inside.any(), (r["id"], pts[inside][:3])


@pytest.mark.parametrize("cut", ["long", "bob"])
def test_flat_auto_gives_the_locks_to_the_hair_and_keeps_one_face(draw, cut):
    setup_head(draw)
    ok(draw("locks", "hair", "--cut", cut, "--part", "520,85"))
    st = state()
    hid = head_el(st)["id"]
    ok(draw("line", "trace", hid))
    # neck and shoulders, as the model draws them after the hair
    ok(draw("line", "455,465 452,585", "--no-snap"))
    ok(draw("line", "545,465 548,585", "--no-snap"))
    ok(draw("line", "100,700 215,622 452,585 548,585 785,622 900,700", "--no-snap"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("flat", "auto"))
    st = state()
    assert {"hair", "face"} <= set(st["regions"])
    face = np.asarray(Image.open("pic.easel/regions/face.png")) > 127
    hair = np.asarray(Image.open("pic.easel/regions/hair.png")) > 127
    assert ndi.label(face)[1] == 1                      # one clean face area
    H, W = face.shape
    el = head_el(st)
    bx = [el["box"][0] * W / 1000, el["box"][1] * H / 1000, el["box"][2] * W / 1000, el["box"][3] * H / 1000]
    ys, xs = np.nonzero(face)
    assert xs.min() >= bx[0] - 4 and xs.max() <= bx[2] + 4 and ys.max() <= bx[3] + 6
    assert hair.sum() > 1.5 * face.sum()
    # the fringe over the forehead is hair, the middle of the face is skin
    cx = int((bx[0] + bx[2]) / 2)
    assert hair[int(bx[1] + 0.25 * (bx[3] - bx[1])), cx]
    assert face[int(bx[1] + 0.75 * (bx[3] - bx[1])), cx]


COAT = "130,700 215,622 452,585 548,585 785,622 870,700"


def _poly_mask(pts_u, H, W):
    from pixelcanvas.easel import draw_core as dc
    q = np.asarray(pts_u, float) * [W / 1000.0, H / 1000.0]
    return dc.poly_mask(H, W, q)


def _front_union(st, H, W, grow=0):
    from pixelcanvas.easel import draw_hair
    m = np.zeros((H, W), bool)
    S = max(H, W)
    for k in st["draw"]["locks"]["hair"]["locks"]:
        if k["layer"] in ("side", "fringe"):
            path = np.asarray(k["path"], float) * [W / 1000.0, H / 1000.0]
            wd = np.asarray(k["width"], float) * S / 1000.0 + 2 * grow
            m |= draw_hair._poly_mask(H, W, draw_hair.lock_shape(path, wd)[2])
    return m


def test_locks_lie_over_the_clothes_drawn_before_them(draw):
    setup_head(draw, size="720x900")
    # a coat drawn before the hair: shoulders and sides
    ok(draw("line", COAT, "--no-snap"))
    ok(draw("line", "130,700 120,1000", "--no-snap"))
    ok(draw("line", "870,700 880,1000", "--no-snap"))
    coat_ids = [r["id"] for r in state()["draw"]["lines"]]
    assert {r.get("part") for r in state()["draw"]["lines"]} == {"coat"}
    text = ok(draw("locks", "hair", "--cut", "long", "--part", "520,85"))
    assert "cut where the locks lie over them" in text
    st = state()
    H, W = st["height"], st["width"]
    front = _front_union(st, H, W)
    coat = [r for r in st["draw"]["lines"] if not str(r.get("src", "")).startswith("locks ")]
    assert any(r.get("under_locks") == "hair" for r in coat)
    assert coat_ids[0] in {r["id"] for r in coat}                 # the longest piece keeps the line's id
    inside = 0
    for r in coat:
        pts = np.asarray(r["path"], float) * [W / 1000.0, H / 1000.0]
        dense = np.vstack([np.linspace(a, b, 8) for a, b in zip(pts[:-1], pts[1:])])
        xi, yi = dense[:, 0].astype(int).clip(0, W - 1), dense[:, 1].astype(int).clip(0, H - 1)
        inside += int(ndi.binary_erosion(front, iterations=3)[yi, xi].sum())
    assert inside == 0                                            # no coat line runs through a lock in front
    # the back hair stops at the coat: every lock line inside the coat lies on a lock in front
    body = ndi.binary_erosion(_poly_mask([[130, 700], [215, 622], [452, 585], [548, 585], [785, 622], [870, 700],
                                          [880, 1000], [120, 1000]], H, W), iterations=4)
    near_front = _front_union(st, H, W, grow=3)
    for r in lock_lines(st):
        pts = np.asarray(r["path"], float) * [W / 1000.0, H / 1000.0]
        xi, yi = pts[:, 0].astype(int).clip(0, W - 1), pts[:, 1].astype(int).clip(0, H - 1)
        assert not (body[yi, xi] & ~near_front[yi, xi]).any(), r["id"]
    # flat auto: a side lock is hair down to below the shoulders, wherever it is wider than the gaps flat closes
    hid = head_el(st)["id"]
    ok(draw("do", f"line trace {hid}; stage done --because test; flat auto"))
    hair = np.asarray(Image.open("pic.easel/regions/hair.png")) > 127
    side = [k for k in state()["draw"]["locks"]["hair"]["locks"] if k["layer"] == "side"]
    wide = 2 * max(2, round(0.006 * H)) + 2 * max(1.6, H / 340) + 6
    shoulder = np.array([[130, 700], [215, 622], [452, 585], [548, 585], [785, 622], [870, 700]], float)
    below = 0
    for k in side:
        for (x, y), wu in zip(k["path"], k["width"]):
            if y > np.interp(x, shoulder[:, 0], shoulder[:, 1]) + 4 and wu * H / 1000 > wide:
                below += 1
                assert hair[int(y * H / 1000), int(x * W / 1000)], (k["id"], x, y)
    assert below
    # locks shine gives the thin tails flat gave to the coat the hair colour back
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("locks", "shine", "hair"))
    det = np.asarray(Image.open("pic.easel/layers/detail.png"))[..., 3] > 10
    line = np.asarray(Image.open("pic.easel/layers/line.png"))[..., 3] > 90
    tails = [(int(y * H / 1000), int(x * W / 1000)) for k in side for (x, y), wu in zip(k["path"], k["width"])
             if 620 < y and 6 < wu * H / 1000 < wide]
    tails = [q for q in tails if not hair[q] and not line[q]]
    assert tails and np.mean([det[q] for q in tails]) > 0.6


def test_locks_again_put_the_clothes_lines_back(draw):
    setup_head(draw)
    ok(draw("line", COAT, "--no-snap"))
    whole = state()["draw"]["lines"][0]
    ok(draw("locks", "hair", "--cut", "long"))
    assert len([r for r in state()["draw"]["lines"] if r.get("from_id") == whole["id"]]) >= 2
    # short hair does not reach the shoulders: the coat line comes back whole
    ok(draw("locks", "hair", "--cut", "short"))
    recs = [r for r in state()["draw"]["lines"] if r["id"] == whole["id"] or r.get("from_id") == whole["id"]]
    assert len(recs) == 1 and recs[0]["path"] == whole["path"] and not recs[0].get("under_locks")


def test_refit_lays_the_locks_over_lines_drawn_after_them(draw):
    from pathlib import Path

    from pixelcanvas.easel import draw_hair
    from pixelcanvas.easel.project import Project
    setup_head(draw)
    ok(draw("locks", "hair", "--cut", "long"))
    text = ok(draw("line", COAT, "--no-snap"))
    # the line command lays the locks over the new line itself (cli_draw calls refit)
    assert "laid again" in text
    p = Project(Path("pic.easel"))
    lines = p.state["draw"]["lines"]
    assert any(r.get("under_locks") == "hair" for r in lines)
    assert draw_hair.refit(p) == []                                # nothing new since: nothing to do


def test_every_cut_reads_and_replaces_the_last(draw):
    setup_head(draw)
    tips = {}
    for cut in CUTS:
        text = ok(draw("locks", "hair", "--cut", cut))
        assert f"cut {cut}" in text
        st = state()
        recs = lock_lines(st)
        tips[cut] = max(k["path"][-1][1] for k in st["draw"]["locks"]["hair"]["locks"] if k["layer"] == "back")
        assert len(recs) >= 8, cut
        # each run replaces the lines of the run before: only this cut's lines are left
        assert sorted(st["draw"]["locks"]["hair"]["lines"]) == sorted(r["id"] for r in recs)
        layers = [k["layer"] for k in st["draw"]["locks"]["hair"]["locks"]]
        if cut == "twin":
            assert len({k["tail"] for k in st["draw"]["locks"]["hair"]["locks"] if k["layer"] == "tail"}) == 2
        if cut == "ponytail":
            assert "tail" in layers
    assert tips["long"] > tips["bob"] > tips["short"]      # the cut sets the length in head heights


def test_locks_without_a_head_use_the_hair_box(draw):
    setup_head(draw, figure=False, head=False)
    text = ok(draw("locks", "hair", "--cut", "bob"))
    assert "hair box" in text
    st = state()
    recs = lock_lines(st)
    assert len(recs) >= 8
    b = st["draw"]["parts"]["hair"]["box"]
    pts = np.vstack([np.asarray(r["path"], float) for r in recs])
    assert pts[:, 0].min() >= b[0] - 40 and pts[:, 0].max() <= b[2] + 40 and pts[:, 1].min() >= b[1] - 40


def test_locks_flow_and_options_change_the_hair(draw):
    setup_head(draw)
    ok(draw("locks", "hair", "--cut", "long", "--fringe", "3", "--volume", "1.2", "--length", "1.5"))
    rec = state()["draw"]["locks"]["hair"]
    assert sum(1 for k in rec["locks"] if k["layer"] == "fringe") == 3
    tips_short = max(k["path"][-1][1] for k in rec["locks"] if k["layer"] == "back")
    ok(draw("locks", "hair", "--cut", "long", "--flow", "500,100 600,400 760,800"))
    rec = state()["draw"]["locks"]["hair"]
    back = [k for k in rec["locks"] if k["layer"] == "back"]
    tips_long = max(k["path"][-1][1] for k in back)
    assert tips_long > tips_short
    # the flow swings the hanging locks to the right
    assert np.mean([k["path"][-1][0] - k["path"][0][0] for k in back]) > 20


def test_undo_takes_the_locks_back(draw):
    setup_head(draw)
    ok(draw("locks", "hair"))
    assert lock_lines(state())
    ok(draw("undo"))
    st = state()
    assert not lock_lines(st) and not st["draw"].get("locks", {}).get("hair")


def test_locks_shine_follows_the_locks(draw):
    setup_head(draw)
    ok(draw("locks", "hair", "--cut", "long", "--part", "520,85"))
    hid = head_el(state())["id"]
    ok(draw("line", "trace", hid))
    ok(draw("line", "455,465 452,585", "--no-snap"))
    ok(draw("line", "545,465 548,585", "--no-snap"))
    ok(draw("line", "100,700 215,622 452,585 548,585 785,622 900,700", "--no-snap"))
    ok(draw("do", "stage done --because test; flat auto; stage done --because test"))
    text = ok(draw("locks", "shine", "hair"))
    assert "marks" in text and "shine ring" in text
    st = state()
    assert "hair" in st["draw"].get("detailed", [])
    det = np.asarray(Image.open("pic.easel/layers/detail.png"))
    hair = np.asarray(Image.open("pic.easel/regions/hair.png")) > 127
    painted = det[..., 3] > 10
    assert painted.any()
    # clipped to the hair region (and the thin lock tips flat gave to a neighbour)
    from pixelcanvas.easel import draw_hair
    H, W = hair.shape
    locks = np.zeros((H, W), bool)
    for k in st["draw"]["locks"]["hair"]["locks"]:
        if k["layer"] != "crown":
            path = np.asarray(k["path"], float) * [W / 1000.0, H / 1000.0]
            wd = np.asarray(k["width"], float) * max(H, W) / 1000.0
            locks |= draw_hair._poly_mask(H, W, draw_hair.lock_shape(path, wd)[2])
    assert (painted & ~hair & ~locks).sum() <= 0.02 * painted.sum()
    line = np.asarray(Image.open("pic.easel/layers/line.png"))[..., 3] > 200
    assert (painted & line).sum() <= 0.02 * painted.sum()      # the shine stays off the lines


def test_lock_errors_are_plain(draw):
    setup_head(draw)
    code, text = draw("locks")
    assert code == 2 and "EXAMPLE:" in text
    code, text = draw("locks", "hair", "--cut", "mullet")
    assert code == 2 and "EXAMPLE:" in text and "long" in text
    code, text = draw("locks", "shine", "hair")
    assert code == 2 and "easel locks hair" in text
    code, text = draw("locks", "nose")
    assert code == 2 and "EXAMPLE:" in text


def test_lock_shape_is_closed_and_pointed():
    from pixelcanvas.easel import draw_hair
    path = np.stack([np.full(60, 100.0), np.linspace(20, 200, 60)], 1)
    widths = draw_hair.taper_profile(60, 30.0, kind="hang")
    left, right, poly = draw_hair.lock_shape(path, widths)
    assert np.hypot(*(left[-1] - right[-1])) < 0.5            # the edges meet at the tip
    assert np.hypot(*(left[0] - right[0])) > 10               # open, wide root
    assert len(poly) == len(left) + len(right)
    assert widths[-1] == 0 and widths.max() <= 30.0 + 1e-6


def test_hidden_pieces_are_cut_away():
    from pixelcanvas.easel import draw_hair
    edge = np.stack([np.linspace(10, 190, 181), np.full(181, 100.0)], 1)
    front = np.zeros((200, 200), bool)
    yy, xx = np.mgrid[0:200, 0:200]
    front[(xx - 100) ** 2 + (yy - 100) ** 2 < 30 ** 2] = True
    runs = draw_hair.visible_runs(edge, front)
    assert len(runs) == 2
    (a0, a1), (b0, b1) = runs
    assert edge[a1 - 1][0] < 74 and edge[b0][0] > 126     # nothing drawn inside the disk in front


def test_a_lock_piece_floating_below_the_chin_is_dropped():
    from pixelcanvas.easel import draw_hair

    def piece(x0, y0, x1, y1):
        n = int(max(abs(x1 - x0), abs(y1 - y0)) / 1.5) + 2
        return {"pts": np.stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)], 1)}
    hair = [piece(200, 100, 200, 600), piece(200, 100, 600, 100), piece(600, 100, 600, 600)]
    parting = piece(390, 140, 420, 180)        # a short line on the head, apart from the rest: it stays
    floating = piece(700, 650, 720, 800)       # a back lock showing again beside the shoulder: it goes
    out = draw_hair.drop_floating(hair + [parting, floating], lw=2.0, S=1000.0, below=480.0)
    assert len(out) == 4
    assert all(pc is not floating for pc in out) and any(pc is parting for pc in out)
