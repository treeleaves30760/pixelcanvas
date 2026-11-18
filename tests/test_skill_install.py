"""pixelcanvas install-skill: copies the pixelcanvas skill into a skills folder, refuses to overwrite without --force."""
import subprocess
import sys

from pixelcanvas import skill_install as si


def test_source_holds_the_pixelcanvas_skill():
    src = si.skill_source()
    text = (src / "SKILL.md").read_text(encoding="utf-8")
    assert "name: pixelcanvas" in text


def test_install_writes_skill_and_prints_it(tmp_path, capsys):
    assert si.main(["--to", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    skill = tmp_path / "pixelcanvas" / "SKILL.md"
    assert skill.is_file()
    assert skill.read_text(encoding="utf-8") == (si.skill_source() / "SKILL.md").read_text(encoding="utf-8")
    assert "wrote" in out and "SKILL.md" in out


def test_refuses_to_overwrite_without_force(tmp_path, capsys):
    assert si.main(["--to", str(tmp_path)]) == 0
    (tmp_path / "pixelcanvas" / "SKILL.md").write_text("---\nname: pixelcanvas\n---\nold\n", encoding="utf-8")
    capsys.readouterr()
    assert si.main(["--to", str(tmp_path)]) == 1
    assert "--force" in capsys.readouterr().err
    assert (tmp_path / "pixelcanvas" / "SKILL.md").read_text(encoding="utf-8").endswith("old\n")
    assert si.main(["--to", str(tmp_path), "--force"]) == 0
    assert not (tmp_path / "pixelcanvas" / "SKILL.md").read_text(encoding="utf-8").endswith("old\n")


def test_force_leaves_a_foreign_folder_alone(tmp_path):
    foreign = tmp_path / "pixelcanvas"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("mine", encoding="utf-8")
    assert si.main(["--to", str(tmp_path), "--force"]) == 1
    assert (foreign / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert not (foreign / "SKILL.md").exists()


def test_opencode_target(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "OPENCODE_DIR", tmp_path / "opencode" / "skills")
    assert si.main(["--opencode"]) == 0
    assert (tmp_path / "opencode" / "skills" / "pixelcanvas" / "SKILL.md").is_file()


def test_runs_as_a_module(tmp_path):
    r = subprocess.run([sys.executable, "-m", "pixelcanvas.skill_install", "--to", str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8", timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "pixelcanvas" / "SKILL.md").is_file()


def test_install_skill_is_a_pixelcanvas_subcommand(tmp_path, capsys):
    from pixelcanvas import cli
    assert cli.main(["install-skill", "--to", str(tmp_path)]) == 0
    assert (tmp_path / "pixelcanvas" / "SKILL.md").is_file()


def test_python_m_pixelcanvas_installs_the_skill(tmp_path):
    r = subprocess.run([sys.executable, "-m", "pixelcanvas", "install-skill", "--to", str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8", timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "pixelcanvas" / "SKILL.md").is_file()


def test_one_console_script_named_pixelcanvas():
    from pathlib import Path
    import re
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    names = re.findall(r"^([\w.-]+)\s*=", block, re.M)
    assert names == ["pixelcanvas"], names
