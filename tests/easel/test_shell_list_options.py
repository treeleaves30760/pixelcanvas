"""List options (--plus / --minus): a repeated option adds to the list, a comma list works the same, a single value
works as before."""
import os

import numpy as np
import pytest
from PIL import Image

from pixelcanvas.easel import cli
from pixelcanvas.easel.project import Project

from .finish_helpers import cwd_module, ok  # noqa: F401
from .test_change_recolor import girl

pytestmark = pytest.mark.usefixtures("cwd_module")


def test_parse_joins_repeats_of_list_options_only():
    sp = cli.Spec("easel region add x --box 1,2,3,4", ["name"], {"minus": "value", "box": "value"})
    a = cli.parse(["x", "--minus", "face", "--minus", "pin", "--box", "1,2,3,4", "--box", "5,6,7,8"], sp)
    assert a["minus"] == "face;pin"
    assert a["box"] == "5,6,7,8"                       # not a list option: the last value, as before
    assert cli.parse(["x", "--minus", "face"], sp)["minus"] == "face"


def test_repeated_and_comma_minus_give_the_same_region(tmp_path):
    os.chdir(tmp_path)
    Image.fromarray(girl()).save(tmp_path / "ref.png")
    ok("new", "pic", "--ref", "ref.png")
    ok("region", "add", "face", "--wand", "500,470", "--focus", "3")
    ok("region", "add", "pin", "--box", "300,150,400,250", "--no-refine")
    ok("region", "add", "h1", "--box", "170,90,830,690", "--no-refine", "--minus", "face", "--minus", "pin")
    ok("region", "add", "h2", "--box", "170,90,830,690", "--no-refine", "--minus", "face,pin")
    ok("region", "add", "h3", "--box", "170,90,830,690", "--no-refine", "--minus", "face")
    p = Project(tmp_path / "pic.easel")
    h1, h2, h3 = p.region_mask("h1"), p.region_mask("h2"), p.region_mask("h3")
    face, pin = p.region_mask("face"), p.region_mask("pin")
    assert np.array_equal(h1, h2)
    assert not (h1 & face).any() and not (h1 & pin).any()
    assert (h3 & pin).any() and not (h3 & face).any()     # a single value: only that region
