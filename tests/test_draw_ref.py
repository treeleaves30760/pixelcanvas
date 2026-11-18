"""Hand mode: drawing mark by mark with a reference to look at, snap to and take colours from."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pixelcanvas import cli, workflow


@pytest.fixture
def draw(tmp_path, monkeypatch):
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


def bad(res, words):
    code, text = res
    assert code != 0, text
    assert words in text, text
    assert "pixelcanvas" in text      # the error carries an example
    return text


def make_ref(path):
    """A white card: a red disc with a black outline, a black stroke, a dark dot, a light-red half on the disc."""
    im = Image.new("RGB", (240, 240), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse([50, 50, 190, 190], fill=(200, 30, 40), outline=(0, 0, 0), width=3)
    d.pieslice([53, 53, 187, 187], 200, 340, fill=(120, 15, 20))        # a shadow shape on the disc
    d.line([(20, 215), (220, 215)], fill=(0, 0, 0), width=3)
    d.ellipse([112, 112, 120, 120], fill=(10, 10, 10))                  # a small dark blob
    im.save(path)


def state():
    return json.load(open("pic.pixelcanvas/state.json", encoding="utf-8"))


def start(run):
    make_ref("ref.png")
    ok(run("new", "pic", "--ref", "ref.png"))
    return ok(run("mode", "hand"))


def test_mode_hand_switches_to_draw_stages(draw):
    text = start(draw)
    st = state()
    assert st["mode"] == "hand" and st["stage"] == "plan"
    assert "mode hand" in text and "pixelcanvas plan subject" in text
    assert any(t["stage"] == "line" for t in st["todo"])
    ok(draw("undo"))
    assert "mode" not in state()


def test_trace_mode_is_untouched(draw):
    make_ref("ref.png")
    ok(draw("new", "pic", "--ref", "ref.png"))
    text = ok(draw("next"))
    assert "region add" in text or "marks" in text
    assert "mode" not in state()
    code, text = draw("palette", "from-ref")
    assert "from the reference, most area first" not in text


def test_plan_with_marks_and_palette_from_ref(draw):
    start(draw)
    ok(draw("plan", "subject", "disc", "--box", "100,100,900,950"))
    text = ok(draw("plan", "part", "ball", "--of", "disc", "--marks", "1,2"))
    assert "-> box" in text
    assert state()["draw"]["parts"]["ball"]["box"]
    text = ok(draw("palette", "from-ref"))
    assert "#ffffff" in text
    sw = state()["draw"]["hand"]["ref_swatches"]
    k = next(i for i, h in enumerate(sw, 1) if h.lower().startswith("#c"))
    ok(draw("palette", "name", k, "ball", 1, "background"))
    pal = state()["draw"]["palette"]["swatches"]
    assert pal["ball"] == sw[k - 1] and pal["background"] == sw[0]
    bad(draw("palette", "name", 99, "x"), "swatch numbers")
    ok(draw("palette", "set", "rim", "pick@500,700"))
    assert state()["draw"]["palette"]["swatches"]["rim"].lower() in ("#c81e28", "#c81e27", "#c71e28")


def test_line_snap_ref_sits_on_the_edge_and_undoes(draw):
    start(draw)
    text = ok(draw("line", "218,500 500,218 782,500", "--snap", "ref"))
    assert "on the reference edge:" in text
    share = int(text.split("on the reference edge: ")[1].split("%")[0])
    assert share >= 70
    assert "weight" in text and "VIEW-REF:" in text
    assert len(state()["draw"]["lines"]) == 1
    ok(draw("undo"))
    assert state()["draw"]["lines"] == []


def test_line_candidates_and_take(draw):
    start(draw)
    text = ok(draw("line", "candidates", "50,50,950,950"))
    cands = state()["draw"]["hand"]["cands"]["line"]
    assert cands and "line take" in text
    bad(draw("line", "take", "all"), "one by one")
    bad(draw("line", "take", ",".join(str(i) for i in range(1, 13))), "at most 10")
    bad(draw("line", "candidates", "all"), "one area at a time")
    ok(draw("line", "take", "1"))
    assert len(state()["draw"]["lines"]) == 1
    bad(draw("line", "take", "1"), "already taken")
    ok(draw("undo"))
    assert not state()["draw"]["hand"]["cands"]["line"][0].get("taken")
    ok(draw("line", "take", "1"))
    # what is drawn is no longer a candidate
    ok(draw("line", "candidates", "50,50,950,950"))
    after = state()["draw"]["hand"]["cands"]["line"]
    assert all(c.get("path") != cands[0]["path"] for c in after)
    assert after[0].get("taken") and after[0]["n"] == 1


def _project():
    from pixelcanvas.project import Project
    return Project.find("pic")


def _max_turn(P, k=4):
    """The sharpest turn (degrees) along a path, measured over chords of ``k`` px."""
    from pixelcanvas import draw_core as dc
    Q = dc.resample(np.asarray(P, float), 1.0)
    v = Q[k:] - Q[:-k]
    a, b = v[:-k], v[k:]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max()) if len(a) else 0.0


def _seed_line(p, a, b):
    from pixelcanvas import draw_core as dc, draw_line
    path = dc.resample(np.array([a, b], float), 1.5)
    rec = draw_line.make_record(p, path, dc.px2u(p, [a, b]), weight=1.0, taper="both", color="line", closed=False)
    dc.dstate(p)["lines"].append(rec)
    return rec


def test_joins_are_tangent_and_runs_merge_into_one_line(draw):
    from pixelcanvas import draw_core as dc, draw_line, draw_ref
    start(draw)
    p = _project()
    d = dc.dstate(p)
    one = lambda q: (1.0, None)                                   # noqa: E731
    l1 = _seed_line(p, (40, 120), (200, 120))
    # an end that comes in at 30 degrees beside a line merges into it along its direction (no hook)
    t = np.linspace(0, 1, 60)[:, None]
    slant = np.array([90.0, 120 + 3 + 40 * np.tan(np.radians(30))]) * (1 - t) + np.array([130.0, 123.0]) * t
    path, notes = draw_line._join(p, dc.resample(slant, 1.5), None)
    assert notes and "merges along" in notes[0]
    out = draw_line.end_dir(path, -1, 2.0)
    assert abs(out[1]) < 0.2 and abs(path[-1][1] - 120) < 1.5
    assert _max_turn(path) < 30
    # an end that stops short of a line at a right angle runs straight on to it
    down = dc.resample(np.array([[150.0, 60], [150.0, 117]]), 1.5)
    path, notes = draw_line._join(p, down, None)
    assert abs(path[-1][1] - 120) < 1 and abs(path[-1][0] - 150) < 1 and _max_turn(path) < 10
    # a line that runs on from L1's end, a pixel off its axis, becomes part of L1: one smooth record
    recs, notes = draw_ref._place(p, dc.resample(np.array([[260.0, 121], [203.0, 121]]), 1.5), [[0, 0]], one,
                                  taper="both", color="line", src="t", fallback=1.0)
    assert recs == [l1] and len(d["lines"]) == 1 and "one line now" in notes[-1]
    P = draw_line.record_path(p, l1)
    assert P[:, 0].min() < 45 and P[:, 0].max() > 255 and _max_turn(P) < 15
    assert l1["taper"] == "both"


def test_overlapping_stretches_are_inked_once(draw):
    from pixelcanvas import draw_core as dc, draw_line, draw_ref
    start(draw)
    p = _project()
    d = dc.dstate(p)
    one = lambda q: (1.0, None)                                   # noqa: E731
    l1 = _seed_line(p, (40, 120), (200, 120))
    # wholly on top of L1: nothing new
    recs, notes = draw_ref._place(p, dc.resample(np.array([[60.0, 121], [180.0, 121]]), 1.5), [[0, 0]], one,
                                  taper="both", color="line", src="t", fallback=1.0)
    assert recs == [] and "ink it already" in notes[0] and len(d["lines"]) == 1
    # half on top of L1: only the new half is inked, and it runs on from L1
    recs, notes = draw_ref._place(p, dc.resample(np.array([[100.0, 121], [230.0, 121]]), 1.5), [[0, 0]], one,
                                  taper="both", color="line", src="t", fallback=1.0)
    assert len(d["lines"]) == 1 and recs == [l1] and any("left out" in n for n in notes)
    assert draw_line.record_path(p, l1)[:, 0].max() > 225
    # a line that crosses L1 stays one line
    recs, _ = draw_ref._place(p, dc.resample(np.array([[150.0, 60], [150.0, 180]]), 1.5), [[0, 0]], one,
                              taper="both", color="line", src="t", fallback=1.0)
    assert len(recs) == 1 and dc.path_length(draw_line.record_path(p, recs[0])) > 110


def make_width_ref(path):
    """Two black lines: one 3-4 px wide in noisy chunks, one 3 px for its first half and 7 px for its second."""
    im = Image.new("RGB", (400, 400), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i, x in enumerate(range(40, 360, 6)):
        h = 3 if i % 3 else 4
        d.rectangle([x, 100 - h // 2, x + 5, 100 - h // 2 + h - 1], fill=(0, 0, 0))
    d.rectangle([40, 299, 200, 301], fill=(0, 0, 0))
    d.rectangle([200, 297, 360, 303], fill=(0, 0, 0))
    im.save(path)


def test_ref_weight_is_steady_and_follows_a_real_change(draw):
    from pixelcanvas import draw_ref
    make_width_ref("ref.png")
    ok(draw("new", "pic", "--ref", "ref.png"))
    ok(draw("mode", "hand"))
    p = _project()
    xs = np.arange(60.0, 340.0, 1.5)
    w, prof = draw_ref.ref_profile(p, np.stack([xs, np.full_like(xs, 100.0)], 1))
    assert w and prof is None
    # the weight sits on the ladder: noise within a step gives the reference's usual weight
    assert draw_ref.steady_weight(p, w * 1.08) == w
    w2, prof2 = draw_ref.ref_profile(p, np.stack([xs, np.full_like(xs, 300.0)], 1))
    assert prof2 and prof2[-1][1] / prof2[0][1] > 1.8
    text = ok(draw("line", "100,750 900,750", "--snap", "ref"))
    assert "width changes along it" in text
    rec = state()["draw"]["lines"][-1]
    assert rec["wprof"] and rec["wprof"][-1][1] > rec["wprof"][0][1]
    ok(draw("line", "100,250 900,250", "--snap", "ref"))
    assert "wprof" not in state()["draw"]["lines"][-1]


def test_wire_path_is_smooth(draw):
    from pixelcanvas import draw_ref
    start(draw)
    path = draw_ref.wire_path(_project(), [[218, 500], [500, 218], [782, 500]])
    assert _max_turn(path) < 16                    # the raw pixel-to-pixel wire turns ~30 degrees at its steps


def test_candidate_numbers_stay_fixed(draw):
    start(draw)
    ok(draw("line", "candidates", "50,50,950,950"))
    first = {c["n"]: c["path"] for c in state()["draw"]["hand"]["cands"]["line"]}
    assert len(first) >= 2
    ok(draw("line", "candidates", "50,50,950,950"))
    again = {c["n"]: c["path"] for c in state()["draw"]["hand"]["cands"]["line"]}
    assert again == first
    # an overlapping area keeps the numbers of what it shares
    top = max(first)
    ok(draw("line", "candidates", "50,50,950,600"))
    for c in state()["draw"]["hand"]["cands"]["line"]:
        assert c["n"] <= top                       # a contour cut by the smaller area keeps its number too
        if c.get("path") in first.values():
            assert first[c["n"]] == c["path"]
    ok(draw("line", "take", top))
    text = ok(draw("line", "candidates", "50,50,950,950"))
    after = state()["draw"]["hand"]["cands"]["line"]
    assert after[top - 1].get("taken") and f"{top} len" not in text
    for c in after:
        if c.get("path") and not c.get("taken") and c["n"] in first:
            assert c["path"] == first[c["n"]]
    new = [c["n"] for c in after if c.get("path") and not c.get("taken") and c["path"] not in first.values()]
    assert all(n > top for n in new)
    bad(draw("line", "take", top), "already taken")


def test_detail_numbers_stay_fixed(draw):
    make_small_ref("ref.png")
    ok(draw("new", "pic", "--ref", "ref.png"))
    ok(draw("mode", "hand"))
    ok(draw("detail", "candidates", "200,200,800,800"))
    first = {c["n"]: c["path"] for c in state()["draw"]["hand"]["cands"]["detail"]}
    assert len(first) >= 2
    ok(draw("detail", "candidates", "600,580,720,700"))       # a smaller area inside: the same numbers
    for c in state()["draw"]["hand"]["cands"]["detail"]:
        assert c["n"] in first and c["path"] == first[c["n"]]
    top = max(first)
    ok(draw("detail", "take", top))
    text = ok(draw("detail", "candidates", "200,200,800,800"))
    after = state()["draw"]["hand"]["cands"]["detail"]
    assert after[top - 1].get("taken") and all(c["n"] <= top for c in after)
    assert {c["n"]: c["path"] for c in after if c.get("path") and not c.get("taken")} == \
        {n: pth for n, pth in first.items() if n != top}


def make_many_ref(path):
    """24 short separate black strokes."""
    im = Image.new("RGB", (240, 240), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for i in range(6):
        for j in range(4):
            x, y = 20 + i * 36, 30 + j * 50
            d.line([(x, y), (x + 24, y + 10)], fill=(0, 0, 0), width=3)
    im.save(path)


def test_candidates_come_in_pages(draw):
    make_many_ref("ref.png")
    ok(draw("new", "pic", "--ref", "ref.png"))
    ok(draw("mode", "hand"))
    text = ok(draw("line", "candidates", "20,20,980,980"))
    cands = [c for c in state()["draw"]["hand"]["cands"]["line"] if c.get("path")]
    assert len(cands) > 15
    assert "page 1 of 2" in text and "pixelcanvas line candidates 20,20,980,980 --page 2" in text
    text2 = ok(draw("line", "candidates", "20,20,980,980", "--page", "2"))
    assert "page 2 of 2" in text2
    n2 = cands[-1]["n"]
    assert f"{n2} len" in text2 and f"{n2} len" not in text
    bad(draw("line", "candidates", "20,20,980,980", "--page", "5"), "pages of candidates")
    ok(draw("line", "take", n2))


def test_flat_pick_shade_like_ref_and_detail(draw):
    start(draw)
    ok(draw("line", "218,500 500,218 782,500 500,782", "--snap", "ref", "--closed"))
    bad(draw("flat", "auto"), "area by area")
    text = ok(draw("flat", "--at", "500,600", "--color", "pick"))
    flat = np.asarray(Image.open("pic.pixelcanvas/layers/flat.png").convert("RGBA"))
    y, x = int(0.6 * flat.shape[0]), int(0.5 * flat.shape[1])
    assert abs(int(flat[y, x, 0]) - 200) < 12 and flat[y, x, 1] < 60
    regs = list(state()["regions"])
    assert regs
    bad(draw("shade", "all"), "one area at a time")
    text = ok(draw("shade", regs[0], "--like", "ref", "--levels", "2"))
    assert "shadow:" in text
    sh = np.asarray(Image.open("pic.pixelcanvas/layers/shade.png").convert("RGBA"))
    assert (sh[..., 3] > 0).sum() > 100
    text = ok(draw("detail", "candidates", "300,300,700,700"))
    if state()["draw"]["hand"]["cands"]["detail"]:
        ok(draw("detail", "take", "1"))


def test_look_lightbox_and_hand_gate(draw):
    start(draw)
    text = ok(draw("look", "--lightbox"))
    assert "VIEW:" in text and "VIEW-REF:" in text
    from pixelcanvas.project import Project
    from pixelcanvas import draw_ref
    p = Project.find("pic")
    p.state["stage"] = "line"
    unmet = draw_ref.gate(p, "line")
    assert unmet and unmet[0]["metric"].startswith("lines") and unmet[0]["fix"].startswith("pixelcanvas")
    why, cmd = draw_ref.hand_next(p, "no reference: x", "pixelcanvas palette make --base x")
    assert cmd == "pixelcanvas palette from-ref" and "no reference" not in why


def make_small_ref(path):
    """A white card with tiny features: a 2x2 nose dot, a 6x2 mouth line and the letters H and i."""
    im = Image.new("RGB", (240, 240), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([60, 60, 61, 61], fill=(20, 20, 20))                    # nose dot
    d.rectangle([80, 62, 85, 63], fill=(150, 20, 30))                   # mouth
    for x0 in (150, 156):                                               # H
        d.rectangle([x0, 150, x0 + 1, 159], fill=(0, 0, 0))
    d.rectangle([152, 154, 155, 155], fill=(0, 0, 0))
    d.rectangle([161, 152, 162, 159], fill=(0, 0, 0))                   # i stem
    d.rectangle([161, 148, 162, 149], fill=(0, 0, 0))                   # i dot
    im.save(path)


def test_detail_candidates_keep_tiny_shapes_as_legal_marks(draw):
    make_small_ref("ref.png")
    ok(draw("new", "pic", "--ref", "ref.png"))
    ok(draw("mode", "hand"))
    ok(draw("detail", "candidates", "200,200,400,300"))
    cands = state()["draw"]["hand"]["cands"]["detail"]
    kinds = sorted(m["kind"] for c in cands for m in c["marks"])
    assert kinds == ["dab", "stroke"], cands
    for c in cands:
        for m in c["marks"]:
            assert m["w"] >= (3 if m["kind"] == "dab" else 1)
    ok(draw("detail", "take", ",".join(str(c["n"]) for c in cands)))
    det = np.asarray(Image.open("pic.pixelcanvas/layers/detail.png").convert("RGBA"))
    assert det[60:62, 60:62, 3].min() > 100 and det[62:64, 80:86, 3].min() > 100
    assert abs(int(det[63, 82, 0]) - 150) < 30
    ok(draw("export", "out.png"))
    out = np.asarray(Image.open("out.png").convert("RGB")).astype(int)
    assert out[60:62, 60:62].sum(-1).max() < 300 and out[62:64, 81:85].sum(-1).max() < 400


def test_detail_candidates_group_lettering_per_letter(draw):
    make_small_ref("ref.png")
    ok(draw("new", "pic", "--ref", "ref.png"))
    ok(draw("mode", "hand"))
    ok(draw("detail", "candidates", "600,580,720,700"))
    cands = state()["draw"]["hand"]["cands"]["detail"]
    assert len(cands) == 2, cands                  # H, and i with its dot
    i_ = min(cands, key=lambda c: c["area"])
    assert len(i_["marks"]) == 2
    ok(draw("detail", "take", "1,2"))
    det = np.asarray(Image.open("pic.pixelcanvas/layers/detail.png").convert("RGBA"))
    assert det[148:150, 161:163, 3].min() > 100 and det[150:160, 150:152, 3].mean() > 180
    assert det[150:160, 156:158, 3].mean() > 180 and det[152:160, 161:163, 3].mean() > 180


def test_hand_todo_has_no_figure_items(draw):
    start(draw)
    ok(draw("plan", "subject", "girl", "--box", "100,50,900,980"))
    for part, box in (("face", "300,100,700,450"), ("arm", "100,450,300,900"), ("coat", "300,450,700,950"),
                      ("hair", "250,50,750,300")):
        ok(draw("plan", "part", part, "--of", "girl", "--box", box))
    todo = " ".join(t["text"] + " " + t["cmd"] for t in state()["todo"])
    for word in ("mannequin", "sketch figure", "pixelcanvas wear", "pixelcanvas hand ", "--onto", "pixelcanvas fold"):
        assert word not in todo, word


def test_line_report_names_leaks_with_close_commands(draw):
    start(draw)
    ok(draw("line", "218,500 500,218 782,500", "--snap", "ref"))
    # (a gap of a few units along the reference's edge is joined by the snap itself; this one is too wide)
    text = ok(draw("line", "788,545 500,782 212,545", "--snap", "ref"))
    leak = [ln for ln in text.splitlines() if ln.startswith("LEAKS")]
    assert len(leak) == 1 and "pixelcanvas line close G1" in leak[0], text


def test_flat_report_names_area_leak_with_a_snap_line(draw):
    start(draw)
    ok(draw("line", "218,500 500,218 782,500 500,782", "--snap", "ref", "--closed"))
    text = ok(draw("flat"))
    leak = [ln for ln in text.splitlines() if ln.startswith("LEAKS")]
    assert len(leak) == 1 and "--snap ref" in leak[0] and "area " in leak[0], text


def test_split_area_asks_for_its_colour_again(draw):
    from pixelcanvas import draw_ref, workflow_draw
    from pixelcanvas.project import Project
    start(draw)
    ok(draw("line", "218,500 500,218 782,500 500,782", "--snap", "ref", "--closed"))
    ok(draw("flat", "--at", "500,600", "--color", "pick"))
    text = ok(draw("line", "150,500 850,500"))
    fix = [ln for ln in text.splitlines() if ln.startswith("FIX: areas")]
    assert fix and "--color pick" in fix[0], text
    p = Project.find("pic")
    p.state["stage"] = "flat"
    why, cmd = workflow_draw.next_action(p)
    assert cmd.startswith("pixelcanvas flat ") and cmd.endswith("--color pick"), (why, cmd)
    assert any(u["fix"] == cmd for u in draw_ref.gate(p, "flat"))
    ok(draw(*cmd.split()[1:]))
    assert draw_ref.split_fix(Project.find("pic")) is None


def test_part_from_marks_is_clipped_to_the_marks(draw):
    from pixelcanvas import draw_plan
    from pixelcanvas.project import Project
    start(draw)
    ok(draw("plan", "subject", "disc", "--box", "100,100,900,950"))
    p = Project.find("pic")
    labels = cli._marks_labels(p)
    ks, cnt = np.unique(labels[labels > 0], return_counts=True)
    marks = ",".join(str(k) for k in ks[np.argsort(cnt)[:2]])        # the two smallest marks
    ok(draw("plan", "part", "ball", "--of", "disc", "--marks", marks))
    part = state()["draw"]["parts"]["ball"]
    assert part["poly"]
    p = Project.find("pic")
    m = draw_plan.part_mask(p, "ball")
    b = part["box"]
    box_px = (b[2] - b[0]) * (b[3] - b[1]) * p.W * p.H / 1e6
    assert 0 < m.sum() < 0.9 * box_px
    assert "outline of its marks" in ok(draw("detail", "candidates", "ball"))
    ok(draw("plan", "part", "ball", "--of", "disc", "--box", "200,200,800,800"))
    assert "poly" not in state()["draw"]["parts"]["ball"]
