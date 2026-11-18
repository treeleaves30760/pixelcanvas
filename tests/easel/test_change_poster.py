"""Poster changes on a traced project: canvas (the reference placed in a new canvas), palette map, a graphic behind
the figure, and lettering / a frame tracked as finishing changes. Synthetic pictures only."""
import json
import shlex

import numpy as np
import pytest
from PIL import Image

from pixelcanvas.easel import change_ops as co
from pixelcanvas.easel.project import Project

from .finish_helpers import cwd_module, ok, run  # noqa: F401
from .test_change_recolor import girl, make

pytestmark = pytest.mark.usefixtures("cwd_module")

POSTER = [
    ["change", "add", "canvas", "--size", "900x1200", "--place", "60,190,940,930", "--color", "#f3e9d8"],
    ["change", "add", "palette", "map", "--to", "#1d2a44,#f3e9d8", "--accent", "clothes=#e8644a"],
    ["change", "add", "graphic", "disc", "20,120,980,880", "--color", "#e8644a"],
    ["change", "add", "text", "SLOW SUMMER", "--box", "70,40,930,175", "--color", "#1d2a44"],
    ["change", "add", "shape", "rect", "22,18,978,982", "--outline", "4", "--color", "#1d2a44", "--layer", "text"],
]


def state(proj):
    return json.loads((proj / "state.json").read_text(encoding="utf-8"))


def follow(n=8):
    """Run the NEXT commands of the change stage until it says stage done."""
    for _ in range(n):
        nxt = ok("next").splitlines()[-1].split("   (")[0][6:]
        if nxt.startswith("easel stage done"):
            return
        ok(*shlex.split(nxt)[1:])
    raise AssertionError("the change stage did not settle")


@pytest.fixture()
def poster(tmp_path):
    proj = make(tmp_path, request="a minimalist poster titled SLOW SUMMER")
    before = {n: Project(proj).region_mask(n).mean() for n in Project(proj).region_names()}
    for c in POSTER:
        ok(*c)
    ok("stage", "done")
    follow()
    return proj, before


def test_canvas_places_the_reference_and_moves_the_regions(poster):
    proj, before = poster
    p = Project(proj)
    assert p.deliver_size() == (900, 1200)
    assert p.ref_deliver.shape[:2] == (1200, 900) and p.ref.shape[:2] == (p.H, p.W)
    orig = np.asarray(Image.open(proj / "orig.png").convert("RGB"))
    assert orig.shape[:2] == (1200, 900)
    assert np.abs(orig[5, 5].astype(int) - [0xf3, 0xe9, 0xd8]).max() <= 2           # margin colour
    face = p.region_mask("face")
    ys, xs = np.nonzero(face)
    assert 0.3 < xs.mean() / p.W < 0.7 and 0.3 < ys.mean() / p.H < 0.7
    bg = p.region_mask("background")
    assert bg[: p.H // 20].all()                                                       # the margins joined the rest
    assert state(proj)["marks"] is None


def test_palette_map_keeps_value_order_and_accent(poster):
    proj, _ = poster
    it = [x for x in state(proj)["changes"]["items"] if x["kind"] == "palette"][0]
    assert it["check"]["value_order"] >= 0.95
    assert it["check"]["accent_edge"]["clothes"] >= 0.5
    p = Project(proj)
    img = p.ref_deliver
    m = p.mask_to_deliver(p.region_mask("clothes"))
    L, C, h = co.lch(co.to_lab(img))
    assert abs(co.angdiff(co.circ_mean(h[m & (C > 10)]), co.hex_lch((0xe8, 0x64, 0x4a))[2])) < 25


def test_graphic_lies_behind_the_figure(poster):
    proj, _ = poster
    it = [x for x in state(proj)["changes"]["items"] if x["kind"] == "graphic"][0]
    assert it["check"]["drawn_px"] > 0 and it["check"]["behind_px"] > 0
    p = Project(proj)
    face = p.mask_to_deliver(p.region_mask("face"))
    from scipy import ndimage as ndi
    inner = ndi.binary_erosion(face, iterations=3)
    coral = np.abs(p.ref_deliver.astype(int) - [0xe8, 0x64, 0x4a]).max(-1) < 20
    assert not (coral & inner).any()


def test_canvas_only_before_tracing_and_once(poster):
    proj, _ = poster
    code, out = run("canvas", "--size", "800x800", "--place", "0,0,1000,1000")
    assert code != 0 and "already set" in out
    ok("stage", "done")                                    # change -> sketch
    ok("stage", "done", "--because", "test")
    code, out = run("canvas", "--size", "800x800", "--place", "0,0,1000,1000")
    assert code != 0


def test_finishing_changes_are_tracked_and_undone(poster):
    proj, _ = poster
    for _ in range(8):
        if state(proj)["stage"] == "finish":
            break
        run("stage", "done", "--because", "test")
    nxt = ok("next").splitlines()[-1]
    assert "easel text" in nxt and "SLOW SUMMER" in nxt
    ok(*shlex.split(nxt.split("   (")[0][6:])[1:])
    nxt = ok("next").splitlines()[-1]
    assert "easel shape rect" in nxt
    ok(*shlex.split(nxt.split("   (")[0][6:])[1:])
    code, out = run("stage", "done")
    assert "change 4" not in out and "change 5" not in out
    ok("undo")
    assert "easel shape rect" in ok("next").splitlines()[-1]


def test_undo_of_canvas_restores_size_and_regions(tmp_path):
    proj = make(tmp_path, request="a poster")
    p0 = Project(proj)
    size0, face0 = (p0.W, p0.H), p0.region_mask("face").copy()
    ok(*POSTER[0])
    ok("stage", "done")
    ok("canvas", "--size", "900x1200", "--place", "60,190,940,930", "--color", "#f3e9d8")
    assert Project(proj).deliver_size() == (900, 1200)
    ok("undo")
    p = Project(proj)
    assert (p.W, p.H) == size0 and np.array_equal(p.region_mask("face"), face0)
    assert p.ref.shape[:2] == (p.H, p.W)


def test_margins_map_to_the_light_end(poster):
    proj, _ = poster
    p = Project(proj)
    t = p.ref_deliver
    assert np.abs(t[5, 5].astype(int) - [0xf3, 0xe9, 0xd8]).max() <= 6        # margin: the light end, not grey
    assert np.abs(t[1195, 450].astype(int) - [0xf3, 0xe9, 0xd8]).max() <= 6


def test_ground_stops_at_line_art():
    img = np.full((200, 200, 3), 250, np.uint8)
    import cv2
    cv2.circle(img, (100, 100), 50, (20, 20, 20), 2)                           # a white area enclosed by a line
    out, info = co.graphic(img, "disc", (10, 10, 190, 190), (232, 100, 74), None)
    assert info["drawn"] > 0
    from pixelcanvas.easel.change_ops import line_barrier
    from scipy import ndimage as ndi
    lbl, n = ndi.label(~ndi.binary_dilation(line_barrier(img), iterations=2))
    edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
    ground = np.isin(lbl, edge[edge > 0])
    assert not ground[100, 100] and ground[5, 5]


def test_refine_by_lines_never_eats_half_a_region():
    img = np.full((120, 120, 3), 250, np.uint8)
    import cv2
    cv2.circle(img, (60, 60), 30, (20, 20, 20), 1)
    img[58:62, 88:93] = 250                                                   # a gap in the outline
    reg = np.zeros((120, 120), bool)
    reg[40:80, 40:80] = True
    new, info = co.refine_by_lines(img, reg)
    assert new.sum() >= 0.5 * reg.sum()
