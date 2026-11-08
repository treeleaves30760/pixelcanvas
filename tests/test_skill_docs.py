# -*- coding: utf-8 -*-
"""確保 skill 文件與範例不會壞掉：techniques.md / typography.md 的每段程式都能跑、每個場景都能渲染、工具能用。

techniques.md 的約定：第一個 ```python 區塊是共用開頭，其餘每段都在「執行過開頭」的乾淨環境裡跑。
"""
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

from pixelcanvas import Canvas  # noqa: E402


def _blocks(name):
    with open(os.path.join(SKILL, "references", name), encoding="utf-8") as f:
        return re.findall(r"```python\n(.*?)```", f.read(), re.S)


TECH = _blocks("techniques.md")
TYPO = _blocks("typography.md")
SCENES = sorted(f[:-3] for f in os.listdir(os.path.join(ROOT, "scenes")) if f.endswith(".py") and not f.startswith("_"))


@pytest.mark.parametrize("i", range(1, len(TECH)))
def test_technique_snippet_runs(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                  # 範例可能會存檔
    prelude = TECH[0].replace("W, H = 1920, 1080", "W, H = 192, 108")
    g = {}
    exec(prelude, g)
    exec(TECH[i], g)
    buf = g["cv"].buf
    assert np.isfinite(buf).all() and buf.std() > 0.005          # 有畫出東西


@pytest.mark.parametrize("i", range(1, len(TYPO)))
def test_typography_snippet_runs(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g = {}
    exec(TYPO[0].replace("W, H = 1920, 1080", "W, H = 384, 216"), g)
    exec(TYPO[i], g)
    buf = g["cv"].buf
    assert np.isfinite(buf).all() and buf.std() > 0.005


@pytest.mark.parametrize("name", SCENES)
def test_scene_renders_small(name):
    cv = importlib.import_module(f"scenes.{name}").render(192, 108)
    assert cv.shape == (108, 192, 3) and np.isfinite(cv.buf).all() and 0.03 < cv.buf.mean() < 0.8


def test_skill_frontmatter_and_links():
    with open(os.path.join(SKILL, "SKILL.md"), encoding="utf-8") as f:
        text = f.read()
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m and re.search(r"^name: pixelcanvas$", m.group(1), re.M) and "description:" in m.group(1)
    for link in re.findall(r"\]\(([^)]+\.md)\)", text):
        assert os.path.exists(os.path.join(SKILL, link)), link
    for scene in re.findall(r"`scenes/(\w+)\.py`", text):          # 表格裡引用的範例場景
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
    assert "九宮格" in r.stdout and "ASCII" in r.stdout and "r2c2" in r.stdout


def test_install_skill(tmp_path):
    r = _run("tools/install_skill.py", "--dest", str(tmp_path))
    assert r.returncode == 0, r.stderr
    with open(tmp_path / "pixelcanvas" / "SKILL.md", encoding="utf-8") as f:
        assert ROOT in f.read()
    assert (tmp_path / "pixelcanvas" / "references" / "api.md").exists()
    assert _run("tools/install_skill.py", "--dest", str(tmp_path), "--uninstall").returncode == 0
    assert not (tmp_path / "pixelcanvas").exists()
