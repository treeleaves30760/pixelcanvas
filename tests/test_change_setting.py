"""A figure put in a new setting: cut-out (painted and white ground), background (project, picture file, gradient,
colour), light, contact shadow; their checks, views, undo and the hand correction. Synthetic pictures only."""
import json
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from pixelcanvas import change_scene as cs
from pixelcanvas import cli
from pixelcanvas.project import Project

from .finish_helpers import cwd_module, ok, run  # noqa: F401

pytestmark = pytest.mark.usefixtures("cwd_module")

SKIN, HAIR, COAT, DARK, GROUND = (235, 205, 185), (222, 226, 232), (150, 146, 140), (74, 58, 48), (62, 58, 54)


def bust(w=300, h=400, seed=3):
    """A bust on a dark painted ground: head and hair, a light coat, and a dark shoulder running off the bottom-right
    corner (as dark as the ground)."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = GROUND
    blot = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 9)
    img += (blot / (np.abs(blot).max() + 1e-6) * 10)[..., None]
    poly = np.array([[60, 400], [95, 270], [150, 240], [300, 235], [300, 400]], np.int32)
    coat = np.zeros((h, w), np.uint8)
    cv2.fillPoly(coat, [poly], 1)
    img[coat > 0] = COAT
    shoulder = np.zeros((h, w), np.uint8)
    cv2.fillPoly(shoulder, [np.array([[190, 400], [215, 262], [300, 245], [300, 400]], np.int32)], 1)
    img[shoulder > 0] = DARK
    cv2.rectangle(img, (252, 100), (300, 252), (200, 180, 120), -1)          # a scarf hanging off the right edge
    cv2.ellipse(img, (150, 150), (62, 74), 0, 0, 360, SKIN, -1)
    cv2.ellipse(img, (150, 118), (70, 52), 0, 180, 360, HAIR, -1)
    cv2.rectangle(img, (120, 222), (185, 244), (240, 240, 238), -1)
    cv2.circle(img, (128, 150), 6, (60, 120, 70), -1)
    cv2.circle(img, (172, 150), 6, (60, 120, 70), -1)
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def standing(w=300, h=400):
    """A full figure standing on a white ground (feet inside the picture)."""
    img = np.full((h, w, 3), 252, np.uint8)
    cv2.rectangle(img, (125, 120), (175, 300), (60, 90, 160), -1)
    cv2.circle(img, (150, 90), 32, SKIN, -1)
    cv2.rectangle(img, (128, 300), (146, 352), (40, 40, 50), -1)
    cv2.rectangle(img, (154, 300), (172, 352), (40, 40, 50), -1)
    cv2.circle(img, (150, 90), 33, (30, 30, 30), 2)
    cv2.rectangle(img, (125, 120), (175, 300), (30, 30, 30), 2)
    return img


def state(proj):
    return json.loads((proj / "state.json").read_text(encoding="utf-8"))


def item(proj, kind):
    return next(x for x in state(proj)["changes"]["items"] if x["kind"] == kind)


def make_bust(root, request="redraw it, but in front of a sunset lake"):
    os.chdir(root)
    Image.fromarray(bust()).save(root / "ref.png")
    ok("new", "pic", "--ref", "ref.png", "--request", request)
    ok("region", "add", "face", "--box", "300,200,700,540", "--no-refine", "--focus", "3")
    ok("region", "add", "hair", "--box", "270,160,730,330", "--no-refine", "--minus", "face")
    ok("region", "add", "scarf", "--box", "853,260,1000,620", "--no-refine")
    ok("region", "add", "coat", "--box", "260,550,640,1000", "--no-refine")     # misses the dark shoulder
    ok("region", "rest", "background")
    return root / "pic.pixelcanvas"


# ----------------------------------------------------------------------------------------------- pure operations

def test_cutout_closes_the_corner_the_figure_runs_off():
    img = bust()
    H, W = img.shape[:2]
    hint = np.zeros((H, W), bool)
    hint[80:216, 90:210] = True
    hint[104:250, 256:300] = True
    hint[220:400, 78:192] = True
    alpha, info = cs.cutout(img, hint)
    assert alpha[380, 285] > 0.9                         # the dark shoulder at the corner is the figure's
    assert alpha[20, 20] < 0.05 and alpha[380, 15] < 0.05
    assert any(q["corner"] == "bottom-right" for q in info["pockets"])
    assert info["lost"] < 0.1
    no, info2 = cs.cutout(img, hint, corners=False)
    assert no[380, 285] < 0.5 and not info2["pockets"]


def test_cutout_edge_is_smooth_and_spots_are_reported():
    img = bust()
    H, W = img.shape[:2]
    hint = np.zeros((H, W), bool)
    hint[80:216, 90:210] = True
    hint[104:250, 256:300] = True
    hint[220:400, 78:192] = True
    alpha, info = cs.cutout(img, hint)
    solid = alpha > 0.5
    # one silhouette, no holes, no specks
    from scipy import ndimage as ndi
    assert ndi.label(solid)[1] == 1 and ndi.label(~solid)[1] == 1
    spots = cs.edge_spots(img, alpha, info["raw"], 3)
    assert len(spots) == 3
    for s in spots:
        x0, y0, x1, y1 = s["box"]
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H


def test_white_ground_cutout_takes_everything_but_the_ground():
    img = standing()
    H, W = img.shape[:2]
    hint = np.zeros((H, W), bool)
    hint[60:300, 120:180] = True
    white = (np.abs(img.astype(int) - 252).max(-1) < 6)
    alpha, info = cs.cutout(img, hint, white_ground=white)
    assert info["mode"] == "white"
    assert alpha[340, 137] > 0.9 and alpha[340, 150] < 0.5          # a leg in, the gap between the legs out
    assert cs.contact(alpha)["y"] >= 348


def test_gradient_and_cover():
    g = cs.gradient(40, 100, (255, 0, 0), (0, 0, 255), 90)
    assert g[0, 20, 0] > 200 and g[-1, 20, 2] > 200
    src = np.zeros((50, 200, 3), np.uint8)
    src[:, 90:110] = 255
    out, crop = cs.cover(src, 100, 100)
    assert out.shape == (100, 100, 3) and crop > 0.5
    assert out[50, 50].min() == 255                                  # centred


# ----------------------------------------------------------------------------------------------- commands

@pytest.fixture()
def setting(tmp_path):
    proj = make_bust(tmp_path)
    ok("change", "add", "cutout", "figure", "--regions", "face,hair,scarf,coat")
    ok("change", "add", "background")
    ok("stage", "done")
    return proj


def test_change_add_background_prints_every_form(setting):
    out = ok("todo")
    assert "--from" in out and "--gradient" in out and "--color" in out
    out = ok("cutout", "figure", "--regions", "face", "--regions", "hair,scarf,coat")
    assert item(setting, "cutout")["params"]["regions"] == ["face", "hair", "scarf", "coat"]
    assert "edge spot 1 at" in out
    assert "bottom-right corner" in out
    assert "--gradient" in out.splitlines()[-1]                        # NEXT: the background, in its three forms


def test_background_gradient_composite_and_undo(setting):
    proj = setting
    ok("cutout", "figure", "--regions", "face,hair,scarf,coat")
    before = np.asarray(Image.open(proj / "ref.png")).copy()
    out = ok("background", "--gradient", "#f2a65a,#5b5f8f")
    p = Project(proj)
    t = p.ref_deliver
    assert np.abs(t[2, 150].astype(int) - [0xf2, 0xa6, 0x5a]).max() < 30           # the sky shows at the top
    assert np.abs(t[150, 150].astype(int) - SKIN).max() < 30                       # the face is kept
    c = item(proj, "background")["check"]
    assert c["halo"] <= 0.08
    assert "old ground left at the figure's edge" in out
    assert "three weakest places" in out
    ok("undo")
    assert np.array_equal(np.asarray(Image.open(proj / "ref.png")), before)
    assert not (proj / "background_src.png").exists()


def test_background_from_picture_file_and_project_folder(setting, tmp_path):
    proj = setting
    ok("cutout", "figure", "--regions", "face,hair,scarf,coat")
    sky = np.zeros((200, 100, 3), np.uint8)
    sky[:100] = (240, 150, 90)
    sky[100:] = (80, 90, 150)
    Image.fromarray(sky).save(tmp_path / "sky.png")
    ok("background", "--from", "sky.png")
    t = Project(proj).ref_deliver
    assert np.abs(t[5, 10].astype(int) - [240, 150, 90]).max() < 12
    assert (proj / "background_src.png").exists()
    # a project folder: its export
    os.makedirs(tmp_path / "lake.pixelcanvas", exist_ok=True)
    lake = np.zeros((300, 200, 3), np.uint8)
    lake[:] = (40, 160, 90)
    Image.fromarray(lake).save(tmp_path / "lake.png")
    code, out = run("background", "--from", "lake.pixelcanvas")
    assert code != 0 and "not a pixelcanvas project" in out
    q = make_free(tmp_path / "free")
    os.chdir(tmp_path)
    st = json.loads((q / "state.json").read_text(encoding="utf-8"))
    st["exported"] = str(tmp_path / "lake.png")
    (q / "state.json").write_text(json.dumps(st), encoding="utf-8")
    ok("background", "--from", str(q))
    t = Project(proj).ref_deliver
    assert np.abs(t[5, 10].astype(int) - [40, 160, 90]).max() < 12
    assert item(proj, "background")["check"]["source"] == "project"


def make_free(root):
    os.makedirs(root, exist_ok=True)
    here = os.getcwd()
    os.chdir(root)
    ok("new", "lake", "--size", "200x300")
    os.chdir(here)
    return root / "lake.pixelcanvas"


def test_background_takes_one_form(setting):
    ok("cutout", "figure", "--regions", "face,hair,scarf,coat")
    code, out = run("background", "--color", "#ffffff", "--gradient", "#000000,#ffffff")
    assert code != 0 and "EXAMPLE" in out
    ok("background", "--color", "#88aacc")
    t = Project(setting).ref_deliver
    assert np.abs(t[3, 3].astype(int) - [0x88, 0xaa, 0xcc]).max() < 3


def test_light_rim_follows_the_bright_side(setting):
    proj = setting
    ok("cutout", "figure", "--regions", "face,hair,scarf,coat")
    ok("background", "--gradient", "#ffd9a0,#30304a", "--angle", "0")           # light on the left
    out = ok("light", "--amount", "0.25")
    c = item(proj, "light")["check"]
    assert c["rim"] == "left" and c["bright_side"] == "left"
    assert c["dist_after"] < c["dist_before"]
    assert "rim light on the left" in out
    out = ok("light", "--rim", "right")
    assert "pixelcanvas light --rim left" in out.splitlines()[-1]


def test_shadow_under_a_standing_figure_and_none_for_a_bust(tmp_path):
    os.chdir(tmp_path)
    Image.fromarray(standing()).save(tmp_path / "ref.png")
    ok("new", "pic", "--ref", "ref.png", "--request", "redraw it, on a beach")
    ok("region", "add", "head", "--box", "380,120,620,310", "--no-refine", "--focus", "3")
    ok("region", "add", "body", "--box", "400,290,600,760", "--no-refine")
    ok("region", "add", "legs", "--box", "410,740,590,890", "--no-refine")
    ok("region", "rest", "background")
    ok("change", "add", "cutout", "figure", "--regions", "head,body,legs")
    ok("stage", "done")
    ok("cutout", "figure", "--regions", "head,body,legs")
    ok("background", "--gradient", "#f0e6d0,#d8c8a8")
    out = ok("shadow", "figure", "--dir", "left")
    c = item(tmp_path / "pic.pixelcanvas", "shadow")["check"]
    assert c["contact"] and c["drop_L"] > 5
    assert "falling left" in out
    # a bust: no ground under it
    os.makedirs(tmp_path / "b", exist_ok=True)
    proj = make_bust(tmp_path / "b")
    ok("change", "add", "cutout", "figure", "--regions", "face,hair,scarf,coat")
    ok("stage", "done")
    ok("cutout", "figure", "--regions", "face,hair,scarf,coat")
    ok("background", "--color", "#88aacc")
    out = ok("shadow", "figure")
    assert "pixelcanvas change drop" in out.splitlines()[-1]
    assert item(proj, "shadow")["check"]["contact"] is False


def test_free_shadow_stays_the_shading_tool():
    from pixelcanvas import cli_change
    ctx = cli.Ctx(None)
    assert cli_change._scene_shadow_here(ctx, ["--from", "hair", "--onto", "face"]) is False


def test_hand_correction_with_a_box_region(setting):
    proj = setting
    ok("cutout", "figure", "--regions", "face,hair,scarf,coat")
    a0 = np.asarray(Image.open(proj / "cutout.png")).astype(np.float32) / 255
    ok("region", "add", "stick", "--box", "860,280,1000,580", "--no-refine")
    ok("cutout", "figure", "--regions", "face,hair,scarf,coat", "--minus", "stick")
    a1 = np.asarray(Image.open(proj / "cutout.png")).astype(np.float32) / 255
    H, W = a1.shape
    assert a0[int(0.45 * H), int(0.95 * W)] > 0.9 and a1[int(0.45 * H), int(0.95 * W)] < 0.05
    assert a1.sum() < a0.sum()
    assert len(state(proj)["changes"]["items"]) == 2                      # remade, not added twice
