# -*- coding: utf-8 -*-
"""Packaging: what the wheel carries, one version everywhere, the PyPI page and the publish workflow.

Fast checks that need no build. The full build + clean-venv smoke test is marked ``release``
(skipped by default): python -m pytest -q -m release tests/test_packaging.py
"""
import ast
import fnmatch
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _toml_section(name):
    """{key: value} of one flat pyproject.toml table (strings and one-line arrays of strings)."""
    text = _read("pyproject.toml")
    m = re.search(rf"^\[{re.escape(name)}\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert m, f"[{name}] missing from pyproject.toml"
    out = {}
    for line in m.group(1).splitlines():
        line = line.split(" #")[0].strip()
        if "=" not in line or line.startswith("#"):
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        out[key.strip('"')] = ast.literal_eval(value)
    return out


def _walk(folder):
    for dirpath, dirs, files in os.walk(os.path.join(ROOT, folder)):
        dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
        for f in files:
            if f.endswith((".pyc", ".pyo")) or f.startswith("zz_") or f == "ROOT.txt":
                continue
            yield os.path.relpath(os.path.join(dirpath, f), os.path.join(ROOT, folder)).replace(os.sep, "/")


def _matches(rel, patterns):
    for pat in patterns:
        if pat.startswith("**/"):
            if fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pat[3:]):
                return True
        elif fnmatch.fnmatch(rel, pat):
            return True
    return False


VERSION = re.search(r'^version = "([^"]+)"', _read("pyproject.toml"), re.M).group(1)
RUNTIME = {"pixelcanvas._runtime.tools": "tools", "pixelcanvas._runtime.scenes": "scenes",
           "pixelcanvas._runtime.skills": "skills"}


# ------------------------------------------------------------------------------------------- package discovery

def _find_packages(top):
    """What setuptools.find_packages(include=[top, top + ".*"]) returns: every folder with an __init__.py whose
    parents are packages too (written out so the test runs where setuptools is not importable)."""
    out, start = [], os.path.join(ROOT, top)
    for dirpath, dirs, files in os.walk(start):
        dirs[:] = [d for d in dirs if d != "__pycache__" and os.path.isfile(os.path.join(dirpath, d, "__init__.py"))]
        if "__init__.py" in files:
            out.append(os.path.relpath(dirpath, ROOT).replace(os.sep, "."))
    return out


def test_every_package_folder_is_discovered():
    found = set(_find_packages("pixelcanvas"))
    try:                                                    # the same answer as setuptools, where it is installed
        from setuptools import find_packages
    except ImportError:
        find_packages = None
    if find_packages is not None:
        assert found == set(find_packages(where=ROOT, include=["pixelcanvas", "pixelcanvas.*"]))
    folders = set()
    for dirpath, dirs, files in os.walk(os.path.join(ROOT, "pixelcanvas")):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        if any(f.endswith(".py") for f in files):
            rel = os.path.relpath(dirpath, ROOT).replace(os.sep, ".")
            assert "__init__.py" in files, f"{rel} holds Python files but no __init__.py: it would not ship"
            folders.add(rel)
    assert {"pixelcanvas", "pixelcanvas.clay", "pixelcanvas.pictograms"} <= folders
    assert folders <= found, sorted(folders - found)


def test_setup_py_adds_the_runtime_and_uses_find_packages():
    text = _read("setup.py")
    assert 'find_packages(include=["pixelcanvas", "pixelcanvas.*"])' in text
    for pkg in RUNTIME:
        assert f'"{pkg}"' in text
    assert "[tool.setuptools]\npackages" not in _read("pyproject.toml")       # no hand-written package list
    assert _toml_section("tool.setuptools.package-dir") == RUNTIME


def test_package_data_covers_every_runtime_file():
    data = _toml_section("tool.setuptools.package-data")
    for pkg, folder in RUNTIME.items():
        files = list(_walk(folder))
        assert files, folder
        loose = [f for f in files if not _matches(f, data.get(pkg, []))]
        assert not loose, f"{folder}/ files the wheel would leave out: {loose}"
    kit_data = [f for f in _walk("pixelcanvas") if not f.endswith(".py")]
    loose = [f for f in kit_data if not _matches(f, data.get("pixelcanvas", []))]
    assert not loose, f"pixelcanvas/ data files the wheel would leave out: {loose}"


def test_wheel_needs_are_present_in_the_source():
    must = ["tools/render.py", "tools/new_scene.py", "tools/compose.py", "tools/kit.py", "tools/install_skill.py",
            "skills/pixelcanvas/SKILL.md", "skills/pixelcanvas/scripts/bootstrap.py",
            "skills/pixelcanvas-lite/SKILL.md", "skills/pixelcanvas-lite/tools/compose.py", "scenes/__init__.py"]
    for rel in must:
        assert os.path.isfile(os.path.join(ROOT, rel)), rel
    assert any(f.startswith("specs/") and f.endswith(".json") for f in _walk("scenes"))
    assert any(f.startswith("pixelcanvas/references/") for f in _walk("skills"))


def test_manifest_in_carries_the_runtime_and_leaves_out_the_rest():
    text = _read("MANIFEST.in")
    exts = {folder: {os.path.splitext(f)[1] for f in _walk(folder)} for folder in RUNTIME.values()}
    for folder, kinds in exts.items():
        line = re.search(rf"^recursive-include {folder} (.+)$", text, re.M)
        assert line, f"MANIFEST.in: no recursive-include for {folder}"
        assert all(f"*{k}" in line.group(1).split() for k in kinds), (folder, kinds, line.group(1))
    for folder in ("gallery", "outputs", "tests"):
        assert re.search(rf"^prune {folder}$", text, re.M), folder
    assert "docs/PYPI.md" in text and "CHANGELOG.md" in text


def test_every_tool_is_a_cli_command():
    from pixelcanvas import cli
    tools = cli.available_tools(ROOT)
    on_disk = {f[:-3] for f in os.listdir(os.path.join(ROOT, "tools")) if f.endswith(".py") and not f.startswith("_")}
    assert on_disk - set(tools) <= {"pc", "release_check"}
    usage = cli._usage()
    for name in tools:
        assert name in usage and cli.tool_path(name, ROOT), name
    assert cli.tool_path("install-skill", ROOT).endswith("install_skill.py")
    assert cli.tool_path("new", ROOT).endswith("new_scene.py")


# ------------------------------------------------------------------------------------------- versions

def test_one_version_everywhere():
    import pixelcanvas
    found = {
        "pyproject.toml": VERSION,
        "pixelcanvas/__init__.py": pixelcanvas.__version__,
        "SKILL.md metadata.version": re.search(r'^\s+version:\s*"?([^"\s]+)"?', _read("skills", "pixelcanvas", "SKILL.md"),
                                               re.M).group(1),
        ".claude-plugin/plugin.json": json.loads(_read(".claude-plugin", "plugin.json"))["version"],
        "CHANGELOG.md": re.search(r"^## (\d+\.\d+\.\d+\S*)", _read("CHANGELOG.md"), re.M).group(1),
    }
    for skill in ("pixelcanvas", "pixelcanvas-lite"):
        p = os.path.join(ROOT, "skills", skill, "scripts", "bootstrap.py")
        if os.path.isfile(p):
            found[f"skills/{skill}/scripts/bootstrap.py"] = re.search(r'^VERSION = "([^"]+)"', _read(p), re.M).group(1)
    assert set(found.values()) == {VERSION}, found


def test_changelog_top_entry_is_dated():
    top = re.search(r"^## (\S+) \((\d{4}-\d{2}-\d{2})\)$", _read("CHANGELOG.md"), re.M)
    assert top and top.group(1) == VERSION


# ------------------------------------------------------------------------------------------- PyPI page

def test_pypi_readme_uses_absolute_links_only():
    assert re.search(r'^readme = "docs/PYPI.md"$', _read("pyproject.toml"), re.M)
    text = _read("docs", "PYPI.md")
    links = re.findall(r"\]\(([^)\s]+)", text) + re.findall(r'(?:src|href)="([^"]+)"', text)
    assert links
    for url in links:
        assert url.startswith("https://"), f"relative link on the PyPI page: {url}"
    assert f"/v{VERSION}/gallery/" in text                      # the image follows the release tag
    assert "pip install pixelcanvas" in text and "uvx pixelcanvas doctor" in text


def test_license_metadata_is_an_spdx_expression():
    text = _read("pyproject.toml")
    assert re.search(r'^license = "MIT"$', text, re.M) and 'license-files = ["LICENSE"]' in text
    assert "License ::" not in text                              # classifiers clash with an SPDX expression
    for v in ("3.9", "3.10", "3.11", "3.12", "3.13"):
        assert f'"Programming Language :: Python :: {v}"' in text


# ------------------------------------------------------------------------------------------- publish workflow

def test_publish_workflow_uses_trusted_publishing():
    text = _read(".github", "workflows", "publish.yml")
    on = re.search(r"^on:\s*$(.*?)^jobs:", text, re.M | re.S).group(1)
    assert "workflow_dispatch" in on and "push" not in on and "release" not in on
    assert re.search(rf'default:\s*"v{re.escape(VERSION)}"', on), "publish.yml ref default is not this version"
    assert "ref: ${{ inputs.ref }}" in text
    assert "pypa/gh-action-pypi-publish@release/v1" in text
    assert re.search(r"environment:\s*\n\s+name: pypi", text) or re.search(r"environment: pypi", text)
    assert "id-token: write" in text
    assert "python -m build" in text and "twine check --strict" in text and "upload-artifact" in text
    for bad in ("secrets.", "password:", "PYPI_API_TOKEN", "api-token"):
        assert bad not in text, bad


def test_tests_workflow_steps_exist():
    text = _read(".github", "workflows", "tests.yml")
    for script in re.findall(r"python (tools/[\w/]+\.py)", text):
        assert os.path.isfile(os.path.join(ROOT, script)), script
    assert "tools/release_check.py" in text and "fonts-noto-cjk" in text


# ------------------------------------------------------------------------------------------- full build (slow)

@pytest.mark.release
def test_release_check_builds_installs_and_runs():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "release_check.py")], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=1800)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    assert "FAIL" not in r.stdout
