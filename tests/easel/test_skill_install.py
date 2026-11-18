"""easel-install-skill: copies the easel skill into a skills folder, refuses to overwrite without --force."""
import subprocess
import sys

from pixelcanvas.easel import skill_install as si


def test_source_holds_the_easel_skill():
    src = si.skill_source()
    text = (src / "SKILL.md").read_text(encoding="utf-8")
    assert "name: easel" in text


def test_install_writes_skill_and_prints_it(tmp_path, capsys):
    assert si.main(["--to", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    skill = tmp_path / "easel" / "SKILL.md"
    assert skill.is_file()
    assert skill.read_text(encoding="utf-8") == (si.skill_source() / "SKILL.md").read_text(encoding="utf-8")
    assert "wrote" in out and "SKILL.md" in out


def test_refuses_to_overwrite_without_force(tmp_path, capsys):
    assert si.main(["--to", str(tmp_path)]) == 0
    (tmp_path / "easel" / "SKILL.md").write_text("---\nname: easel\n---\nold\n", encoding="utf-8")
    capsys.readouterr()
    assert si.main(["--to", str(tmp_path)]) == 1
    assert "--force" in capsys.readouterr().err
    assert (tmp_path / "easel" / "SKILL.md").read_text(encoding="utf-8").endswith("old\n")
    assert si.main(["--to", str(tmp_path), "--force"]) == 0
    assert not (tmp_path / "easel" / "SKILL.md").read_text(encoding="utf-8").endswith("old\n")


def test_force_leaves_a_foreign_folder_alone(tmp_path):
    foreign = tmp_path / "easel"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("mine", encoding="utf-8")
    assert si.main(["--to", str(tmp_path), "--force"]) == 1
    assert (foreign / "notes.txt").read_text(encoding="utf-8") == "mine"
    assert not (foreign / "SKILL.md").exists()


def test_opencode_target(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "OPENCODE_DIR", tmp_path / "opencode" / "skills")
    assert si.main(["--opencode"]) == 0
    assert (tmp_path / "opencode" / "skills" / "easel" / "SKILL.md").is_file()


def test_runs_as_a_module(tmp_path):
    r = subprocess.run([sys.executable, "-m", "pixelcanvas.easel.skill_install", "--to", str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8", timeout=120, stdin=subprocess.DEVNULL)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "easel" / "SKILL.md").is_file()
