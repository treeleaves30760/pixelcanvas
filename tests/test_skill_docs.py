# -*- coding: utf-8 -*-
"""Keep the skill docs and examples working: every code block of every reference page runs, every scene renders,
links resolve, the tools run.

Convention of techniques.md / typography.md: the first ```python block is a shared prelude, every other block runs
in a fresh namespace after the prelude. The kit pages (flora.md, charts.md ...) are run by their module's test file;
``test_every_reference_page_is_run`` makes sure no page is left out.
"""
import glob
import importlib
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SKILL = os.path.join(ROOT, "skills", "pixelcanvas")
REFS = os.path.join(SKILL, "references")

from pixelcanvas import Canvas  # noqa: E402


def _blocks(name):
    with open(os.path.join(REFS, name), encoding="utf-8") as f:
        return re.findall(r"```python\n(.*?)```", f.read(), re.S)


TECH = _blocks("techniques.md")
TYPO = _blocks("typography.md")
SCENES = sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "scenes")) if f.endswith(".py") and not f.startswith("_"))
HERE = ("techniques.md", "typography.md", "api.md", "review.md")      # pages whose blocks run in this file


@pytest.mark.parametrize("i", range(1, len(TECH)))
def test_technique_snippet_runs(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                  # examples may save files
    prelude = TECH[0].replace("W, H = 1920, 1080", "W, H = 192, 108")
    g = {}
    exec(prelude, g)
    exec(TECH[i], g)
    buf = g["cv"].buf
    assert np.isfinite(buf).all() and buf.std() > 0.005          # something was drawn


@pytest.mark.parametrize("i", range(1, len(TYPO)))
def test_typography_snippet_runs(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = {}
    exec(TYPO[0].replace("W, H = 1920, 1080", "W, H = 384, 216"), g)
    exec(TYPO[i], g)
    buf = g["cv"].buf
    assert np.isfinite(buf).all() and buf.std() > 0.005


@pytest.mark.parametrize("page", ["api.md", "review.md"])
def test_page_blocks_run_in_order(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("out", exist_ok=True)
    g = {"__name__": "__doc__"}
    for i, b in enumerate(_blocks(page)):
        exec(compile(b, f"{page} block {i}", "exec"), g)


def test_every_reference_page_is_run():
    """A page with python blocks is run here or by a test file that names it."""
    tests = ""
    for p in glob.glob(os.path.join(ROOT, "tests", "test_*.py")):
        if os.path.basename(p) != "test_skill_docs.py":
            with open(p, encoding="utf-8") as f:
                tests += f.read()
    for p in sorted(glob.glob(os.path.join(REFS, "*.md"))):
        name = os.path.basename(p)
        if _blocks(name) and name not in HERE:
            assert f'"{name}"' in tests or f"'{name}'" in tests, f"no test runs the code blocks of {name}"


@pytest.mark.parametrize("name", SCENES)
def test_scene_renders_small(name):
    cv = importlib.import_module(f"scenes.{name}").render(192, 108)
    b = cv.buf
    assert cv.shape == (108, 192, 3) and np.isfinite(b).all()
    # fits every kind: night paintings, white engineering sheets, posters. Not blank, not one flat colour.
    assert 0.01 < float(b.mean()) < 0.995, f"{name}: the canvas is almost uniformly black or white"
    assert float(np.clip(b, 0, 1).std()) > 0.01, f"{name}: nothing visible was drawn"


def test_reference_links_resolve():
    for p in [os.path.join(SKILL, "SKILL.md")] + sorted(glob.glob(os.path.join(REFS, "*.md"))):
        with open(p, encoding="utf-8") as f:
            text = f.read()
        base = os.path.dirname(p)
        for link in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            if re.match(r"[a-z]+://", link) or link.startswith("mailto:"):
                continue
            assert os.path.exists(os.path.normpath(os.path.join(base, link))) or \
                os.path.exists(os.path.normpath(os.path.join(ROOT, link))), f"{os.path.basename(p)}: broken link {link}"
        for scene in re.findall(r"`scenes/(\w+)\.py`", text):
            assert scene in SCENES or scene.startswith(("form_", "figure_", "creature_")), \
                f"{os.path.basename(p)}: unknown scene {scene}"


def test_skill_frontmatter_and_links():
    with open(os.path.join(SKILL, "SKILL.md"), encoding="utf-8") as f:
        text = f.read()
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m and re.search(r"^name: pixelcanvas$", m.group(1), re.M) and "description:" in m.group(1)
    assert re.search(r"^license: MIT$", m.group(1), re.M)                 # skill directories read these fields
    desc = re.search(r"^description: >-\n((?:  .*\n)+)", m.group(1) + "\n", re.M).group(1)
    assert len(" ".join(desc.split())) <= 1024                            # Agent Skills limit
    for link in re.findall(r"\]\(([^)]+\.md)\)", text):
        assert os.path.exists(os.path.join(SKILL, link)), link
    for scene in re.findall(r"`scenes/(\w+)\.py`", text):          # example scenes named in the tables
        assert scene in SCENES, scene


def _run(*args):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env=env)


def test_look_report(tmp_path):
    p = str(tmp_path / "a.png")
    cv = Canvas(64, 36)
    cv.gradient([(0, "#102040"), (1, "#f0c080")]).glow(48, 10, 4, (2, 2, 2)).save(p)
    r = _run("tools/look.py", "report", p, "--cols", "16")
    assert r.returncode == 0, r.stderr
    assert "3x3 grid" in r.stdout and "ASCII" in r.stdout and "r2c2" in r.stdout


def test_install_skill(tmp_path):
    r = _run("tools/install_skill.py", "--dest", str(tmp_path))
    assert r.returncode == 0, r.stderr
    with open(tmp_path / "pixelcanvas" / "SKILL.md", encoding="utf-8") as f:
        text = f.read()
    assert ROOT in text
    assert (tmp_path / "pixelcanvas" / "references" / "api.md").exists()
    loc = text.split("## Local install location")[1]
    m = re.search(r"^references: (\S+)/$", loc, re.M)                   # the pages, as an absolute folder
    assert m and os.path.isabs(m.group(1)) and os.path.isfile(os.path.join(m.group(1), "solid.md")), loc
    assert _run("tools/install_skill.py", "--dest", str(tmp_path), "--uninstall").returncode == 0
    assert not (tmp_path / "pixelcanvas").exists()


def test_bootstrap_finds_project_and_skill_ships_it():
    assert (ROOT_SKILL := os.path.join(SKILL, "scripts", "bootstrap.py")) and os.path.exists(ROOT_SKILL)
    r = subprocess.run([sys.executable, ROOT_SKILL], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert os.path.samefile(r.stdout.strip().splitlines()[-1], ROOT)


# ------------------------------------------------------------------------------------------------ SKILL.md itself
def _skill_text():
    with open(os.path.join(SKILL, "SKILL.md"), encoding="utf-8") as f:
        return f.read()


def test_skill_md_stays_short():
    """Every line of SKILL.md is read on every picture: at most 180 lines and at most 3,700 tokens (cl100k)."""
    text = _skill_text()
    assert len(text.splitlines()) <= 180
    assert len(text) <= 12100                       # ~3,650 cl100k tokens for this page
    try:
        import tiktoken
    except ImportError:
        return
    assert len(tiktoken.get_encoding("cl100k_base").encode(text)) <= 3700


def test_skill_md_draft_runs_in_the_studio(tmp_path, monkeypatch):
    """The draft printed on the page goes through the page's own command, `studio new NAME --prompt .. --from
    draft.py`: every section renders as a layer, the inventory lines become the checklist and all are confirmed."""
    import contextlib
    import io
    from pixelcanvas.studio import commands as SC
    monkeypatch.chdir(tmp_path)
    block = re.findall(r"```python\n(.*?)```", _skill_text(), re.S)[0]
    assert "creature(cv, " in block and 'anchor("fox", "mouth")' in block and "sk.at(" in block
    assert "cv.text(" in block and "line=True" not in block
    sections = re.findall(r"^# %% (\w+)(?: \[\w+\])?$", block, re.M)
    assert sections == ["wood", "fox", "lantern", "title", "finish"]
    assert "# fox: 1, orange, walking left to right | at centre" in block and "# title: 'Night Walk' | at top" in block

    def run(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = SC.main(["--dir", "art"] + list(argv))
        return code, buf.getvalue()

    os.makedirs(tmp_path / "art" / "fox")
    (tmp_path / "art" / "fox" / "draft.py").write_text(block, encoding="utf-8")
    code, out = run("new", "fox", "--prompt", "a fox carrying a lantern through a snowy wood at night, "
                    "titled 'Night Walk'", "--size", "384x216", "--out", "fox.png", "--from", "art/fox/draft.py")
    assert code == 0 and "ERROR" not in out, out
    assert re.search(r"request: (\d+) of \1 confirmed", out), out
    assert "fox: 1, orange, walking left to right | at centre" in out          # the agent's line is the checklist line
    assert "NEXT:" in out and (tmp_path / "fox.png").exists()


def test_skill_md_teaches_the_studio_loop():
    text = _skill_text()
    low = " ".join(text.lower().split())
    assert 'studio new name --prompt "<the request word for word>" --size wxh --out <delivery path>.png --from ' \
           './art/name/draft.py' in low
    assert "inventory" in low and "each implied fact worked out" in low and "one file write" in low
    assert "studio add --from ./art/name/draft.py" in low and "names defined in one section are visible" in low
    assert "do what the `next:` line says" in low and "never rewrite the whole picture" in low
    assert "open `./art/name/look.png` in the same message" in low and "line by line" in low
    assert "look again" in low and "studio note add" in low and "studio undo" in low and "studio -p name" in low
    assert "the `--out` file is always current" in low
    assert "studio note none" not in low and "studio deliver" not in low        # nothing to type blind at the end
    assert "studio api" in low and "never read the package source" in low
    assert "## checked pictures" in low and "never filter a finished picture" in low
    assert "`new_scene.py` starter" in low and "add base --scene" in low and "no outlines" in low
    assert "heredoc" in low
    from pixelcanvas import stylize
    desc = re.search(r"^description: >-\n((?:  .*\n)+)", text, re.M).group(1)
    assert f"{len(stylize.names())} named art styles" in " ".join(desc.split())


def test_freehand_names_are_exported():
    import pixelcanvas
    from pixelcanvas import illustrate, Sketch, paper, grade, limit_palette, MOODS, grow, edge, signed_distance  # noqa
    assert illustrate.Sketch is Sketch and pixelcanvas.__version__ == "0.5.0"
    assert {"Sketch", "paper", "grade", "illustrate"} <= set(pixelcanvas.__all__)
