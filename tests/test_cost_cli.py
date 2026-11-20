"""The cheap path: compact help, lists without a project, do @file, figure, option aliases."""
import os

import pytest

from pixelcanvas import cli
from pixelcanvas.cli_figure import face_cmd, figure_cmds, hair_cmd, wear_cmds


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PIXELCANVAS_PROJECT", raising=False)
    return tmp_path


def test_help_several_commands_compact(here):
    code, out = cli.run(["help", "pose,block,dress"])
    assert code == 0
    assert "pose <subject>" in out and "--preset" in out and "block <name> <kind>" in out
    assert "EXAMPLE: pixelcanvas dress" in out
    assert "regions:" not in out           # not the whole command list
    assert len(out) < 3000


def test_list_without_project(here):
    code, out = cli.run(["pose", "list"])
    assert code == 0 and "run:" in out and "walk-together" in out   # the whole list, not cut
    code, out = cli.run(["dress", "list"])
    assert code == 0 and "thigh-highs" in out


def test_figure_expands():
    cmds = figure_cmds(["girl", "--pose", "run", "--at", "500,900", "--yaw", "30",
                        "--wear", "blouse:white,skirt:#2c3460:pleated,shoes", "--color", "navy",
                        "--hair", "twintails,swept,#f09ac0,bows:black", "--face", "smile,#3a78c8"])
    assert cmds[0].startswith("pose girl --preset run --at 500,900 --yaw 30")
    assert "dress girl skirt --color '#2c3460' --style pleated" in cmds
    assert "dress girl shoes --color navy" in cmds
    assert cmds[-2] == "hair3d girl --style twintails --bangs side --color '#f09ac0' --bows black"
    assert cmds[-1] == "face girl --expression smile --eyes '#3a78c8'"
    assert wear_cmds("b", "shirt:white:short") == ["dress b shirt --color white --sleeve short"]
    assert hair_cmd("b", "short,side,black") == "hair3d b --style short --bangs side --color black"
    assert face_cmd("b", "grin") == "face b --expression grin"


def test_do_file_and_alias(here):
    assert cli.run(["new", "pic", "--mode", "free", "--size", "200x280"])[0] == 0
    (here / "steps.txt").write_text("# the scene\ncamera --horizon 430 --eye-height 1.1 --fov 50\n"
                                    "block wall box --size 2,0.4,1 --at 500,900 --elevation 0.2\n", encoding="utf-8")
    code, out = cli.run(["do", "@steps.txt"])
    assert code == 0, out
    assert "[2]" in out and "FAILED" not in out
    code, out = cli.run(["do", "@missing.txt"])
    assert code == 2 and "no file" in out
