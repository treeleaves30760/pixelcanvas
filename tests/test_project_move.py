"""A project folder survives being copied or moved, and a copy that lost its reference files is repaired or refused
in plain words (never an internal error)."""
import shutil

import numpy as np
import pytest
from PIL import Image

from pixelcanvas.project import Project

from .finish_helpers import cwd_module, new_project, ok, run, synthetic_picture  # noqa: F401

pytestmark = pytest.mark.usefixtures("cwd_module")


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    root = tmp_path_factory.mktemp("move")
    proj = new_project(root, "pic", ref=synthetic_picture(150, 180))
    ok("fill", "500,500", "--color", "#e8dcc0", "--layer", "flat", "--bounds", "canvas")
    ok("shape", "ellipse", "300,300,700,700", "--color", "#c04030", "--layer", "shade")
    ok("export", "base.png")
    p = Project(proj)
    keep = {"ref": p.ref.copy(), "ref_d": p.ref_deliver.copy(), "scale": p.scale,
            "png": np.asarray(Image.open(root / "base.png").convert("RGB"))}
    (root / "ref.png").unlink()                     # the original reference file is gone from now on
    return root, proj, keep


def _copy(made, tmp_path, drop=()):
    root, proj, _ = made
    dst = tmp_path / "moved" / "pic.pixelcanvas"
    shutil.copytree(proj, dst)
    for name in drop:
        (dst / name).unlink()
    import os
    os.chdir(dst.parent)
    return dst


def test_whole_copy_exports_the_same(made, tmp_path):
    dst = _copy(made, tmp_path)
    ok("export", "out.png")
    assert np.array_equal(np.asarray(Image.open(dst.parent / "out.png").convert("RGB")), made[2]["png"])


def test_lost_ref_png_is_rebuilt_from_the_delivered_reference(made, tmp_path):
    if made[2]["scale"] == 1:
        pytest.skip("a 1x canvas keeps a single reference file")
    dst = _copy(made, tmp_path, drop=("ref.png",))
    ok("export", "out.png")
    assert (dst / "ref.png").exists()
    assert np.array_equal(Project(dst).ref, made[2]["ref"])


def test_lost_delivered_reference_is_rebuilt_from_ref_png(made, tmp_path):
    if made[2]["scale"] == 1:
        pytest.skip("a 1x canvas keeps a single reference file")
    dst = _copy(made, tmp_path, drop=("ref_deliver.png",))
    ok("export", "out.png")
    p = Project(dst)
    assert p.ref_deliver.shape == made[2]["ref_d"].shape
    assert np.abs(p.ref_deliver.astype(int) - made[2]["ref_d"].astype(int)).mean() < 2.0


def test_wrong_sized_reference_is_a_plain_error(made, tmp_path):
    if made[2]["scale"] == 1:
        pytest.skip("a 1x canvas keeps a single reference file")
    dst = _copy(made, tmp_path)
    Image.fromarray(made[2]["ref_d"]).save(dst / "ref.png")     # the delivered-size picture put in the wrong slot
    code, out = run("export", "out.png")
    assert code != 0
    assert "internal problem" not in out
    assert "but this project needs" in out


def test_no_reference_left_is_a_plain_error(made, tmp_path):
    drop = ("ref.png", "ref_deliver.png") if made[2]["scale"] > 1 else ("ref.png",)
    _copy(made, tmp_path, drop=drop)
    code, out = run("export", "out.png")
    assert code != 0
    assert "internal problem" not in out
    assert "reference picture is missing" in out
    assert "EXAMPLE:" in out
