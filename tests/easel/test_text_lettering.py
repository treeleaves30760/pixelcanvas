"""text.py: lettering fitted to a box on the text layer, CJK included; --list, --remove, fonts."""
import numpy as np
import pytest

from pixelcanvas.easel import coords, fonts, text as etext
from pixelcanvas.easel.project import Project

from .finish_helpers import cwd_module, new_project, ok, run, synthetic_picture  # noqa: F401

pytestmark = pytest.mark.usefixtures("cwd_module")

HAS_CJK = fonts.find_font(None, text="中文") is not None


@pytest.fixture
def proj(tmp_path):
    p = new_project(tmp_path, "pic", "400x240")
    ok("fill", "500,500", "--color", "#336699", "--layer", "flat", "--bounds", "canvas")
    return p


def _ink(proj):
    arr = Project(proj).layer("text", create=False)
    assert arr is not None, "the text layer is empty"
    ys, xs = np.nonzero(arr[..., 3] > 0.05)
    return arr, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def _box_px(proj, box):
    p = Project(proj)
    return coords.box_to_px(coords.parse_box(box), p.W, p.H)


def test_text_is_fitted_inside_the_box(proj):
    out = ok("text", "HELLO", "--box", "100,300,900,700", "--color", "#ffffff")
    assert "(layer text)" in out and "size" in out
    arr, ink = _ink(proj)
    x0, y0, x1, y1 = _box_px(proj, "100,300,900,700")
    assert ink[0] >= x0 - 2 and ink[1] >= y0 - 2 and ink[2] <= x1 + 2 and ink[3] <= y1 + 2
    # as big as the box allows: it fills most of one dimension
    assert (ink[2] - ink[0]) > 0.8 * (x1 - x0) or (ink[3] - ink[1]) > 0.8 * (y1 - y0)
    # centred in the box
    assert abs((ink[0] + ink[2]) / 2 - (x0 + x1) / 2) < 3 and abs((ink[1] + ink[3]) / 2 - (y0 + y1) / 2) < 3
    solid = arr[arr[..., 3] > 0.99]
    assert len(solid) and np.allclose(solid[:, :3], 1.0)                      # the colour asked for
    assert Project(proj).state["texts"][0]["text"] == "HELLO"


def test_long_text_wraps_into_the_box_and_fit_keeps_one_line(proj):
    long = "The quick brown fox jumps over the lazy dog"
    ok("text", long, "--box", "100,100,900,600")
    _, ink = _ink(proj)
    x0, y0, x1, y1 = _box_px(proj, "100,100,900,600")
    assert ink[0] >= x0 - 2 and ink[2] <= x1 + 2 and ink[3] <= y1 + 2
    wrapped_h = ink[3] - ink[1]
    ok("text", "--remove", "1")
    ok("text", long, "--box", "100,100,900,600", "--fit")
    _, ink2 = _ink(proj)
    assert ink2[2] <= x1 + 2 and (ink2[3] - ink2[1]) < wrapped_h * 0.6          # one line: much lower than the paragraph


def test_align_weight_and_outline(proj):
    ok("text", "Left", "--box", "100,100,900,400", "--align", "left", "--color", "#ffee00", "--weight", "bold",
       "--outline", "3,#000000")
    arr, ink = _ink(proj)
    x0, y0, x1, y1 = _box_px(proj, "100,100,900,400")
    assert abs(ink[0] - (x0 + 3)) <= 4                                          # left aligned, outline inside the box
    ys, xs = np.nonzero(arr[..., 3] > 0.99)
    cols = {tuple(np.round(arr[y, x, :3] * 255).astype(int)) for y, x in zip(ys[::7], xs[::7])}
    assert (255, 238, 0) in cols and (0, 0, 0) in cols                          # fill and outline colours both there
    it = Project(proj).state["texts"][0]
    assert it["weight"] == "bold" and it["align"] == "left" and it["outline"] == [3.0, "#000000"]
    ok("text", "Right", "--box", "100,500,900,800", "--align", "right")
    arr, _ = _ink(proj)
    sub = arr[int(0.5 * arr.shape[0]):, :, 3] > 0.05
    xs = np.nonzero(sub.any(0))[0]
    assert abs(xs.max() + 1 - x1) <= 4


def test_default_colour_reads_on_the_canvas(tmp_path):
    new_project(tmp_path, "light", "300x200")
    ok("text", "dark on light", "--box", "100,100,900,900")
    arr = Project(tmp_path / "light.easel").layer("text")
    assert arr[arr[..., 3] > 0.99][:, :3].mean() < 0.3
    new_project(tmp_path, "dark", "300x200")
    ok("fill", "500,500", "--color", "#101828", "--layer", "flat", "--bounds", "canvas")
    ok("text", "light on dark", "--box", "100,100,900,900")
    arr = Project(tmp_path / "dark.easel").layer("text")
    assert arr[arr[..., 3] > 0.99][:, :3].mean() > 0.8


@pytest.mark.skipif(not HAS_CJK, reason="no CJK font installed")
def test_cjk_text(proj):
    ok("text", "你好，世界　こんにちは", "--box", "50,300,950,700")
    arr, ink = _ink(proj)
    x0, y0, x1, y1 = _box_px(proj, "50,300,950,700")
    assert ink[0] >= x0 - 2 and ink[2] <= x1 + 2 and ink[1] >= y0 - 2 and ink[3] <= y1 + 2
    assert (arr[..., 3] > 0.5).sum() > 200                                      # real glyphs, not empty boxes
    ok("text", "縱書き", "--box", "850,50,980,900", "--vertical", "--color", "#ffcc00")
    arr, _ = _ink(proj)
    col = arr[:, int(0.85 * arr.shape[1]):, 3]
    ys = np.nonzero((col > 0.5).any(1))[0]
    assert ys.max() - ys.min() > 40                                             # stacked top to bottom


@pytest.mark.skipif(not HAS_CJK, reason="no CJK font installed")
def test_cjk_with_a_latin_font_falls_back(proj):
    ok("text", "ABC 漢字", "--box", "50,300,950,700", "--font", "serif")
    arr, _ = _ink(proj)
    assert (arr[..., 3] > 0.5).sum() > 200


def test_list_and_remove(proj):
    out = ok("text", "--list")
    assert "no text" in out
    ok("text", "one", "--box", "50,50,500,300")
    ok("text", "two", "--box", "500,500,950,950", "--color", "#ff0000")
    out = ok("text", "--list")
    assert '1  "one"' in out and '2  "two"' in out and "#ff0000" in out
    out = ok("text", "--remove", "1")
    p = Project(proj)
    assert [t["text"] for t in p.state["texts"]] == ["two"]
    arr = p.layer("text")
    left = arr[:, : arr.shape[1] // 2 - 10, 3]
    assert left.max() == 0                                                      # "one" is gone, "two" stays
    assert arr[..., 3].max() > 0.9
    # the layer is exactly the render of the remaining items
    assert np.array_equal(arr, np.asarray(etext.render_items(p.H, p.W, p.state["texts"]), np.float32))
    ok("undo")
    assert [t["text"] for t in Project(proj).state["texts"]] == ["one", "two"]
    code, out = run("text", "--remove", "9")
    assert code == 2


def test_undo_and_journal(proj):
    p = Project(proj)
    before = (proj / "layers" / "text.png").read_bytes()
    ok("text", "TITLE", "--box", "100,100,900,400")
    assert (proj / "layers" / "text.png").read_bytes() != before
    ok("undo")
    assert (proj / "layers" / "text.png").read_bytes() == before
    assert "texts" not in Project(proj).state or Project(proj).state["texts"] == []
    ok("redo")
    assert Project(proj).state["texts"][0]["text"] == "TITLE"
    assert any(j.get("cmd", "").startswith("easel text") for j in Project(proj).journal_lines(4))


def test_pick_takes_the_reference_colour_under_the_letters(tmp_path):
    ref = np.zeros((120, 160, 3), np.uint8)
    ref[:] = (230, 230, 230)
    ref[30:90, 20:140] = (200, 30, 40)                                          # a red band the lettering sits on
    new_project(tmp_path, "pic", ref=ref)
    ok("fill", "500,500", "--color", "#dddddd", "--layer", "flat", "--bounds", "canvas")
    ok("text", "RED", "--box", "125,250,875,750", "--color", "pick")
    assert Project(tmp_path / "pic.easel").state["texts"][0]["color"] == "#c81e28"


def test_usage_errors_show_a_corrected_example(proj):
    code, out = run("text", "hi")
    assert code == 2 and "--box" in out and "EXAMPLE: easel text" in out
    code, out = run("text", "hi", "--box", "100,100,900,400", "--font", "no-such-font-xyz")
    assert code == 2 and "not found" in out and "EXAMPLE: easel text" in out
    code, out = run("text", "hi", "--box", "100,100,900,400", "--weight", "wobbly")
    assert code == 2 and "regular, light, bold, black" in out
    code, out = run("text", "hi", "--box", "100,100,900,400", "--align", "diagonal")
    assert code == 2 and "left, center, right" in out
    code, out = run("text", "a long line of words that cannot fit", "--box", "100,100,130,110")
    assert code == 2 and "too small" in out
    code, out = run("text", "hi", "--box", "100,100,900,400", "--color", "notacolour")
    assert code == 2 and "unknown colour" in out
    code, out = run("text", "hi", "--box", "100,100,900,400", "--colour")
    assert code == 2


def test_batch_and_lenient_box(proj):
    ok("do", 'text "A" --box "100, 100, 400, 400" --color white; text "B" --box 500,500,900,900')
    assert [t["text"] for t in Project(proj).state["texts"]] == ["A", "B"]


def test_fonts_command(proj):
    out = ok("fonts")
    assert "font styles you can name" in out and "installed font file" in out
    out = ok("fonts", "zzzzqqq")
    assert "0 installed font file" in out
