"""export.py: PNG, layered PSD and ORA (names, blend modes, opacity, visibility, clipping)."""
import re
import zipfile

import numpy as np
import pytest
from PIL import Image

from pixelcanvas.project import Project

from .finish_helpers import cwd_module, ok, paint_demo, read_psd, run  # noqa: F401

pytestmark = pytest.mark.usefixtures("cwd_module")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp")
    proj = paint_demo(root)
    ok("layer", "detail", "--clip")
    ok("layer", "line", "--opacity", "0.5")
    out = ok("export", "final.png", "--psd", "--ora")
    return root, proj, out


def test_png_is_the_export_composite(exported):
    root, proj, out = exported
    p = Project(proj)
    png = np.asarray(Image.open(root / "final.png").convert("RGB"))
    assert np.array_equal(png, p.composite_u8("export"))
    assert "exported final.png" in out and "final.psd" in out and "final.ora" in out
    assert p.state["exported"].endswith("final.png")


def test_psd_reads_back(exported):
    root, proj, _ = exported
    p = Project(proj)
    merged = np.asarray(Image.open(root / "final.psd").convert("RGB"))      # Pillow reads the merged picture
    assert np.array_equal(merged, np.asarray(Image.open(root / "final.png").convert("RGB")))
    psd = read_psd(root / "final.psd")
    assert psd["size"] == (p.W, p.H)
    recs = psd["layers"]
    assert [r["name"] for r in recs] == [e["name"] for e in p.state["layers"]]          # bottom to top
    keys = {"normal": b"norm", "multiply": b"mul "}
    for r, e in zip(recs, p.state["layers"]):
        assert r["key"] == keys[e["mode"]], r["name"]
        assert r["opacity"] == int(round(e["opacity"] * 255))
        assert bool(r["clip"]) == bool(e.get("clip"))
        want_visible = e.get("visible", True) and e["name"] != "sketch"
        assert r["visible"] == want_visible
        arr = p.layer(e["name"], create=False)
        if arr is None:
            assert r["rgba"].size == 0
            continue
        u8 = (arr * 255 + 0.5).astype(np.uint8)
        x0, y0, x1, y1 = r["box"]
        want = u8[y0:y1, x0:x1].copy()
        want[want[..., 3] == 0, :3] = 0
        assert np.array_equal(r["rgba"], want), r["name"]
    by = {r["name"]: r for r in recs}
    assert by["shade"]["key"] == b"mul " and by["detail"]["clip"] == 1 and by["line"]["opacity"] == 128


def test_ora_stack_xml(exported):
    root, proj, _ = exported
    p = Project(proj)
    z = zipfile.ZipFile(root / "final.ora")
    assert z.read("mimetype") == b"image/openraster"
    xml = z.read("stack.xml").decode("utf-8")
    layers = re.findall(r"<layer ([^>]*)/>", xml)
    names = [re.search(r'name="([^"]*)"', a).group(1) for a in layers]
    assert names == [e["name"] for e in reversed(p.state["layers"])]                   # top layer first
    by = {re.search(r'name="([^"]*)"', a).group(1): a for a in layers}
    assert 'composite-op="svg:multiply"' in by["shade"]
    assert 'opacity="0.800"' in by["shade"] and 'opacity="0.500"' in by["line"]
    assert 'visibility="hidden"' in by["sketch"] and 'visibility="visible"' in by["line"]
    merged = np.asarray(Image.open(z.open("mergedimage.png")).convert("RGB"))
    assert np.array_equal(merged, p.composite_u8("export"))
    line = np.asarray(Image.open(z.open(re.search(r'src="([^"]*)"', by["line"]).group(1))).convert("RGBA"))
    x = int(re.search(r' x="(\d+)"', by["line"]).group(1))
    y = int(re.search(r' y="(\d+)"', by["line"]).group(1))
    full = (p.layer("line") * 255 + 0.5).astype(np.uint8)
    assert np.array_equal(line[..., 3], full[y:y + line.shape[0], x:x + line.shape[1], 3])


def test_extension_shortcuts_and_errors(tmp_path):
    paint_demo(tmp_path)
    out = ok("export", "deliver.psd")
    assert (tmp_path / "deliver.psd").exists() and (tmp_path / "deliver.png").exists(), out
    code, out = run("export", "x.bmp")
    assert code == 2 and "EXAMPLE: pixelcanvas export final.png" in out and "Traceback" not in out
    code, out = run("export", "final.png", "--timelapse", "film.avi")
    assert code == 2 and "film.gif" in out
    code, out = run("export", "final.png", "--pdf")
    assert code == 2 and "unknown option" in out


def test_old_project_without_texts_key_still_exports(tmp_path):
    proj = paint_demo(tmp_path)
    assert "texts" not in Project(proj).state
    ok("export", "a.png", "--psd")
    assert (tmp_path / "a.psd").exists()
