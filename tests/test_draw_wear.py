"""pixelcanvas wear / pixelcanvas hand: clothes and hands built from the mannequin as ordinary, editable line records."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pixelcanvas import cli, workflow


@pytest.fixture
def draw(tmp_path, monkeypatch):
    """Install the draw hook for one test and restore pixelcanvas's tables afterwards."""
    from pixelcanvas import cli_draw
    saved_cmds = dict(cli.COMMANDS)
    saved_usage = dict(cli.USAGE)
    names = ("next_action", "check_gate", "build_todo", "plan_lines", "gate_words")
    saved_wf = {n: getattr(workflow, n) for n in names}
    cli_draw.install()
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        return cli.run([str(a) for a in argv])
    yield run
    cli.COMMANDS.clear()
    cli.COMMANDS.update(saved_cmds)
    cli.USAGE.clear()
    cli.USAGE.update(saved_usage)
    for n, f in saved_wf.items():
        setattr(workflow, n, f)
    if hasattr(workflow, "_draw_patched"):
        del workflow._draw_patched


def ok(res):
    code, text = res
    assert code == 0, text
    assert "NEXT:" in text
    return text


def state(name: str = "w") -> dict:
    with open(f"{name}.pixelcanvas/state.json", encoding="utf-8") as f:
        return json.load(f)


def lines_of(st: dict, piece: str = None, item: str = None):
    out = []
    for r in st["draw"]["lines"]:
        w = r.get("wear")
        if not w:
            continue
        if piece and w.get("piece") != piece:
            continue
        if item and not str(w.get("item", "")).startswith(item):
            continue
        out.append(r)
    return out


def figure(st: dict, subject: str = "girl") -> dict:
    return next(el for el in st["draw"]["sketch"] if el["kind"] == "figure" and el.get("subject") == subject)


def setup_girl(run, pose: str = "stand", heads: str = None, size: str = "300x400"):
    ok(run("new", "w", "--size", size, "--request", "an anime girl in a coat and a red scarf"))
    ok(run("style", "cel"))
    ok(run("do", "plan subject girl --box 200,40,800,980; plan part face --box 400,60,600,200; "
                 "plan part coat --box 260,220,740,800; plan part scarf --box 380,180,620,330; "
                 "plan part hood --box 360,30,640,260; plan part hand-r --box 600,400,760,560; "
                 "plan light top-left; palette make; palette pick 1; stage done"))
    ok(run("sketch", "figure", "girl", "--pose", pose, *(["--heads", heads] if heads else [])))
    ok(run("sketch", "head", "girl"))
    ok(run("stage", "done"))


def px_path(rec, W: int, H: int) -> np.ndarray:
    a = np.asarray(rec["path"], float)
    return np.stack([a[:, 0] * W / 1000.0, a[:, 1] * H / 1000.0], 1)


# ============================================================================================ registration

def test_commands_are_installed(draw):
    from pixelcanvas import draw_wear
    assert set(draw_wear.COMMANDS) == {"wear", "hand"}
    assert {"wear", "hand"} <= set(draw_wear.USAGE)
    assert "wear" in cli.COMMANDS and "hand" in cli.COMMANDS


# ============================================================================================ body garments

def test_coat_lines_close_into_areas_and_take_the_planned_part(draw):
    setup_girl(draw)
    text = ok(draw("wear", "girl", "coat"))
    assert "VIEW:" in text
    st = state()
    recs = lines_of(st, "coat")
    assert recs and all(r["part"] == "coat" for r in recs)
    body = lines_of(st, "coat", "body")
    assert any(1.3 <= r["weight"] <= 1.5 for r in body)
    # the body outline is a closed shape (or, once its sleeves cover part of it, keeps that shape on record)
    assert any(r["closed"] or r.get("shape") for r in body)
    # the coat comes with sleeves: one closed tube per arm, each its own area
    assert len([r for r in lines_of(st, "coat", "sleeve") if r["closed"]]) >= 2
    # inner lines (folds, cuffs, opening) are lighter than the silhouette
    assert any(r["weight"] < 1.1 for r in recs)


def _hem(st, W, H, piece="coat"):
    """(lowest y, width of the outline near the hem) of a garment's body in px."""
    pts = np.vstack([px_path(r, W, H) for r in lines_of(st, piece, "body")])
    low = pts[pts[:, 1] > pts[:, 1].max() - 0.03 * H]
    return pts[:, 1].max(), np.ptp(low[:, 0])


def test_coat_length_and_fit_change_the_outline(draw):
    setup_girl(draw)
    W, H = 300, 400
    ok(draw("wear", "girl", "coat", "--length", "hip", "--fit", "tight"))
    hip, w_hip = _hem(state(), W, H)
    ok(draw("wear", "girl", "coat", "--length", "ankle", "--fit", "loose"))
    st = state()
    ankle, w_ankle = _hem(st, W, H)
    J = figure(st)["joints"]
    assert ankle > hip + 0.2 * H
    assert ankle >= J["lknee"][1] * H / 1000.0
    assert w_ankle > w_hip


def test_running_a_piece_again_replaces_it(draw):
    setup_girl(draw)
    ok(draw("wear", "girl", "coat"))
    n1 = len(lines_of(state(), "coat"))
    ok(draw("wear", "girl", "coat", "--length", "hip"))
    n2 = len(lines_of(state(), "coat"))
    assert n2 <= n1 + 1
    text = ok(draw("wear", "girl", "coat"))
    assert "replaced" in text


def test_chibi_robe_reaches_the_floor_and_flares(draw):
    ok(draw("new", "w", "--size", "300x360", "--request", "a chibi fox mage in a robe holding a staff"))
    ok(draw("style", "cel"))
    ok(draw("do", "plan subject fox --box 170,60,760,960; plan part head --of fox --box 200,90,720,500 --color fur; "
                  "plan part robe --of fox --box 230,460,700,900; plan light top-left; palette make; palette pick 1; "
                  "stage done"))
    ok(draw("sketch", "figure", "fox", "--pose", "hold", "--heads", "2.5"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("wear", "fox", "robe", "--length", "floor"))
    st = state()
    J = figure(st, "fox")["joints"]
    W, H = 300, 360
    body = np.vstack([px_path(r, W, H) for r in lines_of(st, "robe", "body")])
    assert body[:, 1].max() >= J["lankle"][1] * H / 1000.0
    hip_w = (J["rhip"][0] - J["lhip"][0]) * W / 1000.0
    low = body[body[:, 1] > body[:, 1].max() - 0.03 * H]
    assert np.ptp(low[:, 0]) > 1.6 * hip_w
    assert all(r["part"] == "robe" for r in lines_of(st, "robe"))


# ============================================================================================ occlusion

def test_scarf_cuts_the_coat_neckline_and_undo_restores_every_line(draw):
    from pixelcanvas import draw_wear
    setup_girl(draw)
    ok(draw("wear", "girl", "coat"))
    before = state()["draw"]["lines"]
    text = ok(draw("wear", "girl", "scarf"))
    st = state()
    after = st["draw"]["lines"]
    scarf = lines_of(st, "scarf")
    assert scarf and all(r["part"] == "scarf" for r in scarf)
    # band, knot and two ends: four shapes of their own (closed, or cut by the knot in front and keeping the shape)
    items = {r["wear"]["item"] for r in scarf}
    assert {"scarf wrap", "scarf knot", "scarf end-back", "scarf end-front"} <= items
    # no coat line is left inside the scarf's shapes
    W, H = 300, 400
    shapes = [px_path(r, W, H) if r["closed"] else
              np.asarray(r["shape"], float) * [W / 1000.0, H / 1000.0] for r in scarf if r["closed"] or r.get("shape")]
    assert len(shapes) >= 4
    cover = np.logical_or.reduce([draw_wear.shape_mask(W, H, s, erode=2.0) for s in shapes])
    for r in lines_of(st, "coat"):
        q = np.round(px_path(r, W, H)).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        assert not cover[q[:, 1], q[:, 0]].any(), r["id"]
    old = {r["id"]: r for r in before}
    assert any(r["id"] in old and r["path"] != old[r["id"]]["path"] for r in after) or len(after) > len(before)
    assert "cut" in text
    ok(draw("undo"))
    assert state()["draw"]["lines"] == before


def test_hair_is_kept_in_front_of_a_coat(draw):
    setup_girl(draw)
    ok(draw("plan", "part", "hair", "--box", "300,30,700,700"))
    ok(draw("line", "330,120 300,400 290,650", "--no-snap", "--no-join"))
    hair = state()["draw"]["lines"][-1]
    assert hair["part"] == "hair"
    ok(draw("wear", "girl", "coat"))
    kept = [r for r in state()["draw"]["lines"] if r["id"] == hair["id"]]
    assert kept and kept[0]["path"] == hair["path"]


def test_cut_path_splits_a_line_crossing_a_shape():
    from pixelcanvas import draw_wear
    m = np.zeros((100, 100), bool)
    m[40:60, 40:60] = True
    path = np.stack([np.linspace(5, 95, 61), np.full(61, 50.0)], 1)
    runs = draw_wear.outside_runs(path, m, closed=False, min_cut=1.0, min_keep=3.0)
    assert len(runs) == 2
    (a0, a1), (b0, b1) = runs
    assert path[a1 - 1][0] <= 41 and path[b0][0] >= 59


# ============================================================================================ hands

def test_hand_grip_cuts_the_handle_and_takes_the_hand_part(draw):
    setup_girl(draw, pose="hold")
    J = figure(state())["joints"]
    wx, wy = J["rwrist"]
    ok(draw("line", f"{wx + 25:.0f},{wy - 300:.0f} {wx + 25:.0f},{wy + 300:.0f}", "--no-snap", "--no-join"))
    shaft = state()["draw"]["lines"][-1]["id"]
    text = ok(draw("hand", "girl", "r", "--grip", f"{wx + 25:.0f},{wy:.0f}"))
    st = state()
    hand = lines_of(st, "hand")
    assert hand and all(r["part"] == "hand-r" for r in hand)
    plain = [r for r in st["draw"]["lines"] if not r.get("wear")]
    # the shaft is split in two where the fingers cover it
    assert len(plain) == 2 and shaft in {r["id"] for r in plain}
    H = 400
    spans = sorted((min(p[1] for p in r["path"]), max(p[1] for p in r["path"])) for r in plain)
    assert spans[0][1] < spans[1][0]
    assert "cut" in text


def test_hands_put_the_thumb_toward_the_body(draw):
    setup_girl(draw)
    ok(draw("hand", "girl", "l"))
    ok(draw("hand", "girl", "r"))
    st = state()
    W, H = 300, 400
    J = figure(st)["joints"]
    for side in "lr":
        el = np.array(J[side + "elbow"], float) * [W / 1000, H / 1000]
        wr = np.array(J[side + "wrist"], float) * [W / 1000, H / 1000]
        f = (wr - el) / np.linalg.norm(wr - el)
        n = np.array([-f[1], f[0]])
        centre = np.array(figure(st)["joints"]["head"], float)[0] * W / 1000
        toward = 1 if (centre - wr[0]) * n[0] > 0 else -1
        pts = np.vstack([px_path(r, W, H) for r in lines_of(st, "hand", "hand-" + side)])
        off = (pts - wr) @ n * toward
        assert off.max() > -off.min(), side


def test_paw_and_open_hand_draw_their_own_outlines(draw):
    setup_girl(draw)
    ok(draw("hand", "girl", "l", "--paw"))
    ok(draw("hand", "girl", "r", "--open"))
    st = state()
    assert lines_of(st, "hand", "hand-l") and lines_of(st, "hand", "hand-r")
    paw = lines_of(st, "hand", "hand-l")
    assert any(r["weight"] < 1.0 for r in paw)        # toe lines


def test_hand_sits_behind_its_own_cuff(draw):
    setup_girl(draw)
    ok(draw("wear", "girl", "coat"))
    ok(draw("hand", "girl", "l"))
    st = state()
    hand = lines_of(st, "hand", "hand-l")
    assert hand and not any(r["closed"] for r in hand)     # the wrist edge is hidden in the cuff


# ============================================================================================ hood, collar

def test_hood_up_frames_the_face_and_down_rolls_behind_the_neck(draw):
    setup_girl(draw)
    ok(draw("wear", "girl", "hood"))
    st = state()
    hood = lines_of(st, "hood")
    assert len([r for r in hood if r["closed"]]) >= 2 and all(r["part"] == "hood" for r in hood)
    head = next(el for el in st["draw"]["sketch"] if el["kind"] == "head")
    up_top = min(min(p[1] for p in r["path"]) for r in hood)
    assert up_top < head["box"][1]
    ok(draw("wear", "girl", "hood", "--down"))
    st = state()
    hood = lines_of(st, "hood")
    chin = figure(st)["joints"]["chin"][1]
    assert hood and min(min(p[1] for p in r["path"]) for r in hood) > chin - 5


def test_collar_and_errors(draw):
    setup_girl(draw)
    ok(draw("wear", "girl", "coat", "--sleeveless"))
    assert not lines_of(state(), "coat", "sleeve")
    ok(draw("wear", "girl", "collar"))
    assert len([r for r in lines_of(state(), "collar") if r["closed"]]) >= 2
    code, text = draw("wear", "girl", "hat")
    assert code == 2 and "EXAMPLE:" in text
    code, text = draw("wear", "nobody", "coat")
    assert code == 2 and "EXAMPLE:" in text
    code, text = draw("hand", "girl", "x")
    assert code == 2 and "EXAMPLE:" in text


def test_flat_auto_fills_the_pieces(draw):
    setup_girl(draw)
    ok(draw("line", "trace", "s2"))
    ok(draw("wear", "girl", "coat"))
    ok(draw("wear", "girl", "scarf"))
    ok(draw("flat", "auto"))
    regs = state()["regions"]
    assert "coat" in regs and "scarf" in regs


def _mask_of(rec, W, H, erode=0.0):
    from pixelcanvas import draw_wear
    poly = px_path(rec, W, H) if rec["closed"] else np.asarray(rec["shape"], float) * [W / 1000.0, H / 1000.0]
    return draw_wear.shape_mask(W, H, poly, erode=erode)


def _inside(mask, pts):
    q = np.round(pts).astype(int)
    q[:, 0] = np.clip(q[:, 0], 0, mask.shape[1] - 1)
    q[:, 1] = np.clip(q[:, 1], 0, mask.shape[0] - 1)
    return mask[q[:, 1], q[:, 0]]


def test_scarf_knot_is_one_lump_with_the_ends_coming_out_under_it(draw):
    setup_girl(draw, size="600x800")
    ok(draw("wear", "girl", "coat"))
    ok(draw("wear", "girl", "scarf", "--side", "r"))
    st = state()
    W, H = 600, 800
    items = {r["wear"]["item"] for r in lines_of(st, "scarf")}
    assert {"scarf knot", "scarf knot crease"} <= items
    # one closed lump: no second oval laid across it
    lump = [r for r in lines_of(st, "scarf", "scarf knot") if r["wear"]["item"] == "scarf knot"]
    assert len(lump) == 1 and lump[0]["closed"]
    knot = _mask_of(lump[0], W, H, erode=1.5)
    # the ends start under the knot: no line of an end lies inside it
    for r in lines_of(st, "scarf", "scarf end"):
        assert not _inside(knot, px_path(r, W, H)).any(), r["id"]
    # its creases stay inside it
    import cv2
    grown = cv2.dilate(_mask_of(lump[0], W, H).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for r in lines_of(st, "scarf", "scarf knot crease"):
        assert _inside(grown, px_path(r, W, H)).all(), r["id"]


def test_chibi_collar_rises_behind_the_head(draw):
    from pixelcanvas import draw_wear
    ok(draw("new", "w", "--size", "300x360", "--request", "a chibi fox mage in a robe"))
    ok(draw("style", "cel"))
    ok(draw("do", "plan subject fox --box 170,60,760,960; plan part head --of fox --box 200,90,720,500 --color fur; "
                  "plan part robe --of fox --box 230,460,700,900; plan light top-left; palette make; palette pick 1; "
                  "stage done"))
    ok(draw("sketch", "figure", "fox", "--pose", "hold", "--heads", "2.5"))
    ok(draw("sketch", "head", "fox", "--shape", "round"))
    ok(draw("stage", "done", "--because", "test"))
    ok(draw("line", "trace", "s2", "--outline"))
    head = state()["draw"]["lines"][-1]
    assert head["closed"]
    ok(draw("wear", "fox", "robe", "--length", "floor"))
    st = state()
    W, H = 300, 360
    head_m = _mask_of(head, W, H)
    robe = lines_of(st, "robe")
    # the robe's outline goes behind the head ...
    for r in robe:
        assert not _inside(draw_wear.shape_mask(W, H, px_path(head, W, H), erode=2.0), px_path(r, W, H)).any()
    # ... and its top reaches up to it: no gap of background between the chin and the collar
    body = [r for r in lines_of(st, "robe", "body") if r["closed"] or r.get("shape")]
    assert body
    import cv2
    grown = cv2.dilate(head_m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert (grown & _mask_of(body[0], W, H)).sum() > 20


def _tube_and_grip(draw, piece=None):
    setup_girl(draw, pose="hold")
    if piece:
        ok(draw("wear", "girl", piece))
    J = figure(state())["joints"]
    wx, wy = J["rwrist"]
    x = wx + 25
    ok(draw("line", "tube", f"{x:.0f},{wy - 300:.0f} {x:.0f},{wy + 300:.0f}", "--width", "16,16"))
    text = ok(draw("hand", "girl", "r", "--grip", f"{x + 4:.0f},{wy:.0f}"))
    return x, wy, text


def test_grip_finds_both_sides_of_a_tube_and_the_thumb_crosses_it(draw):
    x, wy, text = _tube_and_grip(draw)
    # the fist is centred on the tube although the grip point was off its middle
    gx = float(text.split("grip at ")[1].split(",")[0])
    assert abs(gx - x) <= 3
    st = state()
    W, H = 300, 400
    thumb = np.vstack([px_path(r, W, H) for r in lines_of(st, "hand", "hand-r thumb")])
    fingers = np.vstack([px_path(r, W, H) for r in lines_of(st, "hand", "hand-r fingers")])
    xp = x * W / 1000.0
    assert thumb[:, 0].min() < xp < thumb[:, 0].max()
    assert fingers[:, 0].min() < xp < fingers[:, 0].max()


def test_a_fist_comes_out_over_its_own_robe_cuff(draw):
    from pixelcanvas import draw_wear
    from pixelcanvas.project import Project
    _tube_and_grip(draw, piece="robe")
    st = state()
    W, H = 300, 400
    hand = [r for r in lines_of(st, "hand") if r["wear"]["side"] == "r" and (r["closed"] or r.get("shape"))]
    assert hand
    p = Project.find("w.pixelcanvas")
    body = draw_wear.Body(p, draw_wear.find_figure(p, "girl", "x"), draw_wear._head_of(p, "girl"))
    past = draw_wear._past_cuff(p, body, "girl", "r")
    assert past is not None
    import cv2
    # a cut ends right on the edge of that zone: leave a margin of two pixels
    past = cv2.erode(past.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cover = np.logical_or.reduce([_mask_of(r, W, H, erode=2.0) for r in hand]) & past
    # beyond the cuff line no line of the sleeve is left on top of the fist (before it the sleeve is in front)
    for r in lines_of(st, "robe", "sleeve-r"):
        assert not _inside(cover, px_path(r, W, H)).any(), r["id"]
    # and the palm shows: its outline is drawn, not swallowed by the wide cuff
    palm = [r for r in lines_of(st, "hand") if r["wear"]["item"] == "hand-r"]
    assert palm and sum(len(r["path"]) for r in palm) > 6


def test_a_bent_sleeve_does_not_cross_itself(draw):
    import cv2
    setup_girl(draw, pose="hold")
    ok(draw("wear", "girl", "robe"))
    st = state()
    W, H = 300, 400
    rec = next(r for r in lines_of(st, "robe", "sleeve-r") if r["wear"]["item"] == "sleeve-r")
    poly = px_path(rec, W, H) if rec["closed"] else np.asarray(rec["shape"], float) * [W / 1000.0, H / 1000.0]
    from pixelcanvas import draw_wear
    filled = draw_wear.shape_mask(W, H, poly).sum()
    area = abs(cv2.contourArea(poly.astype(np.float32).reshape(-1, 1, 2)))
    assert abs(filled - area) < 0.1 * filled


def test_the_wear_step_needs_no_flat_hints_any_more(draw):
    setup_girl(draw)
    ok(draw("wear", "girl", "coat"))
    text = ok(draw("wear", "girl", "scarf"))
    # flat auto colours the areas round the clothes itself: wear no longer prints flat --at commands
    assert "flat auto goes by plan boxes" not in text and "pixelcanvas flat --at" not in text


def test_flat_auto_gives_the_scarf_ends_the_scarf_and_the_coat_front_the_coat(draw):
    from pixelcanvas import draw_flat, draw_wear
    from pixelcanvas.project import Project
    setup_girl(draw)
    ok(draw("wear", "girl", "coat"))
    ok(draw("wear", "girl", "scarf"))
    p = Project.find("w.pixelcanvas")
    labels, table, src = draw_flat.compute_areas(p)
    boxes = draw_flat.box_assign(p, labels, table)
    fixes = draw_wear.wear_corrections(p, labels, table, boxes)
    assert fixes                     # the plan boxes alone get some areas wrong
    final = draw_flat.auto_assign(p, labels, table)
    st = state()
    assert all(part in st["draw"]["parts"] or part == "background" for part, _ in final.values())
    # a sliver between two pieces' lines (a scarf end beside the coat's opening) is left to flat's outline pass
    big = {r["n"] for r in table if r["share"] >= 0.005}
    for n, want in fixes.items():
        if n in big:
            assert final[n] == want
    # flat auto already holds the corrections: nothing is left to correct
    assert not set(draw_wear.wear_corrections(p, labels, table, final)) & big
    # every area whose front-most drawn piece is the coat is coat, whatever smaller box (the scarf's) holds it
    shapes = draw_wear.wear_shapes(p)
    coat_areas = [r["n"] for r in table if (draw_wear.area_part(p, r["anchor"], shapes) or ("",))[0] == "coat"]
    assert coat_areas
    for n in coat_areas:
        if n in big:
            assert final.get(n, ("", ""))[0] == "coat", n
    W, H = 300, 400
    ends = np.logical_or.reduce([_mask_of(r, W, H, erode=2.0) for r in lines_of(st, "scarf", "scarf end")
                                 if r["closed"] or r.get("shape")])
    for r in table:
        x, y = int(r["anchor"][0]), int(r["anchor"][1])
        if ends[y, x]:
            assert final.get(r["n"], ("", ""))[0] == "scarf", r["n"]


def test_records_made_before_the_flags_keep_their_roles():
    from pixelcanvas import draw_wear
    assert draw_wear._wear_role({"wear": {"item": "hood opening"}, "closed": True}) == "hole"
    assert draw_wear._wear_role({"wear": {"item": "body"}, "closed": False, "shape": [[0, 0], [1, 0], [1, 1]]}) == "fill"
    assert draw_wear._wear_role({"wear": {"item": "body fold"}, "closed": False}) is None
