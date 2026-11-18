"""Changes to the target (easel change / recolor / region refine): a redraw that differs from its reference.

Synthetic pictures only. The no-change path must stay byte for byte the plain redraw."""
import json
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from pixelcanvas.easel import change_ops as co
from pixelcanvas.easel.project import Project

from .finish_helpers import cwd_module, ok, run  # noqa: F401

pytestmark = pytest.mark.usefixtures("cwd_module")


def girl(w=150, h=180) -> np.ndarray:
    """Pink hair with darker strands (some poking out past the hair mass), a skin face, blue eyes, dark red top."""
    s = w / 300.0
    img = np.full((h, w, 3), 236, np.uint8)
    cv2.ellipse(img, (int(150 * s), int(140 * s)), (int(110 * s), int(120 * s)), 0, 0, 360, (235, 140, 170), -1)
    for k in range(8):
        cv2.line(img, (int((60 + k * 25) * s), int(40 * s)), (int((70 + k * 22) * s), int(250 * s)),
                 (205, 100, 140), max(1, int(3 * s)))
    cv2.circle(img, (int(150 * s), int(170 * s)), int(70 * s), (245, 210, 185), -1)
    cv2.circle(img, (int(150 * s), int(170 * s)), int(70 * s), (40, 30, 30), 1)
    for x in (125, 175):
        cv2.circle(img, (int(x * s), int(170 * s)), max(2, int(8 * s)), (60, 90, 200), -1)
    cv2.rectangle(img, (int(70 * s), int(250 * s)), (int(230 * s), h), (120, 40, 60), -1)
    return img


def make(root, name="pic", request=None):
    os.chdir(root)
    Image.fromarray(girl()).save(root / "ref.png")
    args = ["new", name, "--ref", "ref.png"] + (["--request", request] if request else [])
    ok(*args)
    ok("region", "add", "face", "--wand", "500,470", "--focus", "3")
    ok("region", "add", "clothes", "--box", "235,700,765,1000", "--no-refine")
    ok("region", "add", "hair", "--box", "170,90,830,690", "--no-refine", "--minus", "face")
    ok("region", "rest", "background")
    return root / f"{name}.easel"


def state(proj):
    return json.loads((proj / "state.json").read_text(encoding="utf-8"))


def hue_of(img, mask):
    L, C, h = co.lch(co.to_lab(img))
    sel = mask & (C > 8)
    return co.circ_mean(h[sel], C[sel])


# ----------------------------------------------------------------------------------------------- pure operations

def test_recolor_moves_hue_and_keeps_shading():
    img = girl(300, 360)
    mask = np.zeros(img.shape[:2], bool)
    cv2.ellipse(mask.view(np.uint8), (150, 140), (100, 110), 0, 0, 360, 1, -1)
    mask = mask.view(bool) & (np.abs(img.astype(int) - [245, 210, 185]).sum(-1) > 30)
    out, info = co.recolor(img, mask, (127, 208, 176))
    c = co.recolor_checks(img, out, mask, info, reach=8, min_px=12)
    assert c["hue_off"] < 10
    assert c["shading"] < 1.0
    assert np.array_equal(out[~ndi_dilate(mask, 3)], img[~ndi_dilate(mask, 3)])      # nothing outside moves


def ndi_dilate(m, n):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(m, iterations=n)


def test_refine_by_hue_takes_strands_not_the_dark_top_or_face():
    img = girl(300, 360)
    rough = np.zeros(img.shape[:2], bool)
    cv2.ellipse(rough.view(np.uint8), (150, 140), (106, 116), 0, 0, 360, 1, -1)
    face = np.zeros_like(rough)
    cv2.circle(face.view(np.uint8), (150, 170), 72, 1, -1)
    rough = rough.view(bool) & ~face.view(bool)
    face = face.view(bool)
    info = co.colour_info(img, rough)
    before = co.stray_patches(img, rough, info, reach=8, min_px=4, exclude=face)
    new, _ = co.refine_by_hue(img, rough, reach=8, others=face)
    after = co.stray_patches(img, new, info, reach=8, min_px=4, exclude=face)
    assert before["n"] > 0 and after["n"] == 0
    top = np.zeros_like(rough)
    top[255:, 75:225] = True
    assert not (new & ~rough & top).any()      # the dark red top shares the hue but not the lightness
    assert not (new & face).any()


# ----------------------------------------------------------------------------------------------- no-change path

def test_without_request_nothing_changes(tmp_path):
    proj = make(tmp_path)
    st = state(proj)
    assert "change" not in [t["stage"] for t in st["todo"]]
    out = ok("next")
    assert "change" not in out.splitlines()[-1]
    ok("stage", "done")
    assert not (proj / "orig.png").exists()


def test_change_none_is_the_plain_redraw_byte_for_byte(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pa = make(a)
    pb = make(b, request="redraw this picture")
    code, out = run("stage", "done")
    assert code != 0 and "changes" in out          # the request must be answered first
    ok("change", "none")
    for root in (a, b):
        os.chdir(root)
        ok("stage", "done")
        ok("stage", "done", "--because", "skip the sketch")
        ok("flat", "all")
        ok("export", "final.png")
    sa, sb = state(pa), state(pb)
    assert [t["text"] for t in sa["todo"]] == [t["text"] for t in sb["todo"]]
    assert sa["step"] == sb["step"]
    assert (a / "final.png").read_bytes() == (b / "final.png").read_bytes()
    assert not (pb / "orig.png").exists()


# ----------------------------------------------------------------------------------------------- change flow

@pytest.fixture()
def changed(tmp_path):
    proj = make(tmp_path, request="redraw it with mint green hair")
    out = ok("next")
    assert "easel change add" in out.splitlines()[-1]
    ok("change", "add", "recolor", "hair", "--to", "#7fd0b0")
    ok("stage", "done")
    assert state(proj)["stage"] == "change"
    return proj


def test_change_stage_todo_and_gate(changed):
    st = state(changed)
    assert st["todo"][0]["stage"] == "plan"
    items = [t for t in st["todo"] if t["stage"] == "change" and t["kind"] == "task"]
    assert items and items[0]["cmd"].startswith("easel recolor hair")
    code, out = run("stage", "done")
    assert code != 0 and "not made yet" in out


def test_recolor_then_refine_clears_stray_patches(changed):
    before_ref = np.asarray(Image.open(changed / "ref_deliver.png").convert("RGB")).copy()
    out = ok("recolor", "hair", "--to", "#7fd0b0")
    assert "stray patches:" in out and "VIEW:" in out
    it = state(changed)["changes"]["items"][0]
    assert it["check"]["hue_off"] < 15 and it["check"]["shading"] <= 2.5
    assert it["check"]["stray"] > 0                         # the strand tips outside the rough region
    assert out.splitlines()[-1].startswith("NEXT: easel region refine hair")
    p = Project(changed)
    assert np.array_equal(np.asarray(Image.open(changed / "orig.png").convert("RGB")), before_ref)
    hair = co.lch(co.to_lab(p.ref_deliver))
    del hair
    out = ok("region", "refine", "hair", "--by", "hue")
    assert "-> 0" in out
    st = state(changed)
    assert st["changes"]["items"][0]["check"]["stray"] == 0
    ok("stage", "done")
    assert state(changed)["stage"] == "sketch"


def test_target_is_what_the_stages_trace_and_score(changed):
    ok("recolor", "hair", "--to", "#7fd0b0")
    p = Project(changed)
    m = p.mask_to_deliver(p.region_mask("hair"))
    assert abs(co.angdiff(hue_of(p.ref_deliver, m), co.hex_lch((127, 208, 176))[2])) < 15
    big = np.asarray(Image.open(changed / "ref.png").convert("RGB"))
    assert big.shape[:2] == (p.H, p.W)
    out = ok("look", "--show", "both")
    assert "-t" in [ln for ln in out.splitlines() if ln.startswith("VIEW-REF")][0]   # views of the new target


def test_undo_restores_the_previous_target_exactly(changed):
    ok("recolor", "hair", "--to", "#7fd0b0")
    first = (changed / "ref.png").read_bytes()
    ok("recolor", "hair", "--to", "#4060c0")
    assert (changed / "ref.png").read_bytes() != first
    ok("undo")
    assert (changed / "ref.png").read_bytes() == first
    assert state(changed)["changes"]["items"][0]["params"]["to"] == "#7fd0b0"
    ok("undo")
    assert not (changed / "orig.png").exists()
    p = Project(changed)
    assert np.array_equal(p.ref_deliver, np.asarray(Image.open(changed.parent / "ref.png").convert("RGB")))


def test_drop_puts_the_original_back(changed):
    ok("recolor", "hair", "--to", "#7fd0b0")
    ok("change", "drop", "1")
    p = Project(changed)
    assert np.array_equal(p.ref_deliver, np.asarray(Image.open(changed / "orig.png").convert("RGB")))
    assert "change" not in state(changed)["todo"][1]["stage"]


def test_late_change_marks_regions_stale_and_leads_through_them(changed):
    ok("recolor", "hair", "--to", "#7fd0b0")
    ok("region", "refine", "hair")
    ok("stage", "done")                                  # change
    ok("stage", "done", "--because", "skip the sketch")
    ok("flat", "all")
    flat_before = Project(changed).layer("flat").copy()
    out = ok("recolor", "clothes", "--to", "#304080")
    assert "redraw flat:" in out and "clothes" in out
    st = state(changed)
    assert ["flat", "clothes"] in st["stale"] and all(s == "flat" for s, _ in st["stale"])
    assert "hair" not in [r for _, r in st["stale"]]
    p = Project(changed)
    flat = p.layer("flat")
    hair = p.region_mask("hair")
    assert np.array_equal(flat[hair], flat_before[hair])            # no hole outside the redrawn regions
    assert (flat[p.region_mask("clothes"), 3] == 0).mean() > 0.5
    n = 0
    while True:                                          # follow NEXT: refine the change, then redraw what is stale
        nxt = ok("next").splitlines()[-1].split("   (")[0]
        if nxt.startswith("NEXT: easel region refine "):
            ok(*nxt.split()[2:7])
        elif nxt.startswith("NEXT: easel flat "):
            ok(*nxt.split()[2:4])
        else:
            break
        n += 1
        assert n < 8
    assert not state(changed).get("stale")
    ok("undo", str(n + 1))
    assert np.array_equal(Project(changed).layer("flat"), flat_before)


def test_refine_keeps_other_regions_colours_and_leaves_no_stray(changed):
    ok("recolor", "hair", "--to", "#7fd0b0")
    p = Project(changed)
    old = {n: p.region_mask(n).copy() for n in p.region_names()}
    ok("region", "refine", "hair", "--by", "hue")
    p = Project(changed)
    added = p.region_mask("hair") & ~old["hair"]
    assert added.any()
    orig = co.to_lab(np.asarray(Image.open(changed / "orig.png").convert("RGB")))
    import cv2 as _cv2
    orig = _cv2.resize(orig, (p.W, p.H), interpolation=_cv2.INTER_NEAREST)
    L, C, h = co.lch(orig)
    skin = np.abs(co.angdiff(h, co.hex_lch((245, 210, 185))[2])) < 15
    top = np.abs(co.angdiff(h, co.hex_lch((120, 40, 60))[2])) < 8
    assert (added & old["face"] & skin & (C > 6)).sum() < 0.01 * max(1, (added & old["face"]).sum())
    assert not (added & old["clothes"] & top & (L < 40)).any()      # the dark red top keeps its own colour
    assert state(changed)["changes"]["items"][0]["check"]["stray"] == 0
    out = ok("region", "refine", "hair", "--by", "hue")       # a second refine finds nothing more
    assert "(+0 px" in out


def test_a_small_region_keeps_its_pixels_under_a_larger_regions_recolor(changed):
    ok("region", "add", "badge", "--box", "300,200,360,260", "--no-refine")     # inside the hair, under 2%
    ok("recolor", "hair", "--to", "#7fd0b0")
    p = Project(changed)
    badge = p.mask_to_deliver(p.region_mask("badge"))
    from scipy import ndimage as ndi
    inner = ndi.binary_erosion(badge, iterations=2)
    orig = np.asarray(Image.open(changed / "orig.png").convert("RGB"))
    assert np.abs(p.ref_deliver[inner].astype(int) - orig[inner]).max() <= 2       # the feather's edge only


def test_a_hair_sliver_inside_a_small_regions_outline_goes_with_the_hair():
    """A small part's rough outline (a heart beside the hair) also holds a sliver of the hair: the sliver turns with
    the hair; the heart keeps its colour."""
    from pixelcanvas.easel import change as chg
    H, W = 200, 200
    img = np.full((H, W, 3), 250, np.uint8)
    img[:, :120] = (230, 150, 190)                                         # pink hair
    cv2.circle(img, (165, 40), 14, (225, 60, 120), -1)                     # a pink heart beside it
    hair = np.zeros((H, W), bool)
    hair[:, :112] = True                                                   # the hair outline misses its edge
    heart = np.zeros((H, W), bool)
    heart[18:62, 104:184] = True                                           # the heart's box reaches into the hair

    class P:
        state = {"regions": {"hair": {"area": 0.6, "how": "box"}, "heart": {"area": 0.01, "how": "box"},
                             "background": {"area": 0.3, "how": "rest of the canvas"}}}

        def deliver_size(self):
            return W, H
    p = P()
    masks = {"hair": hair | (np.arange(W)[None, :] >= 112) & (np.arange(W)[None, :] < 120) & ~heart,
             "heart": heart, "background": ~hair}
    orig_region_d = chg.region_d
    chg.region_d = lambda _p, r: masks[r]
    try:
        held = chg.held_islands(p, "hair", img, masks["hair"] & ~heart)
    finally:
        chg.region_d = orig_region_d
    sliver = heart & (np.arange(W)[None, :] < 120)
    assert held[sliver].mean() > 0.9                                       # the hair sliver joins the recolour
    assert not held[30:50, 155:175].any()                                  # the heart stays


def ink_girl(w=300, h=360) -> np.ndarray:
    """A monochrome ink drawing: white paper, a grey-hatched hair mass with a dark outline, a white face inside."""
    img = np.full((h, w, 3), 252, np.uint8)
    cv2.ellipse(img, (150, 140), (110, 120), 0, 0, 360, (190, 190, 190), -1)
    for k in range(10):
        cv2.line(img, (60 + k * 20, 40), (70 + k * 18, 250), (120, 120, 120), 2)
    cv2.ellipse(img, (150, 140), (110, 120), 0, 0, 360, (30, 30, 30), 2)
    cv2.circle(img, (150, 170), 70, (252, 252, 252), -1)
    cv2.circle(img, (150, 170), 70, (30, 30, 30), 2)
    return img


def test_grey_region_takes_the_hue_and_a_face_under_a_rough_outline_stays(tmp_path):
    os_ = __import__("os")
    os_.chdir(tmp_path)
    Image.fromarray(ink_girl()).save(tmp_path / "ref.png")
    ok("new", "pic", "--ref", "ref.png", "--request", "auburn hair")
    ok("region", "add", "face", "--wand", "500,470", "--focus", "3")
    ok("region", "add", "hair", "--box", "120,60,880,740", "--no-refine")          # covers the face too
    ok("region", "rest", "background")
    ok("change", "add", "recolor", "hair", "--to", "#a0522d")
    ok("stage", "done")
    out = ok("recolor", "hair", "--to", "#a0522d")
    it = state(tmp_path / "pic.easel")["changes"]["items"][0]
    assert it["check"]["hue_off"] < 15, out
    p = Project(tmp_path / "pic.easel")
    from scipy import ndimage as ndi
    face = ndi.binary_erosion(p.mask_to_deliver(p.region_mask("face")), iterations=3)
    orig = np.asarray(Image.open(tmp_path / "pic.easel" / "orig.png").convert("RGB"))
    assert (co.delta_e(orig, p.ref_deliver)[face] > 3).mean() < 0.01
    assert out.splitlines()[-1].startswith("NEXT: easel stage done") or "done" in out
