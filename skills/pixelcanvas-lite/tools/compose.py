"""Runs pixelcanvas tools/compose.py from the skill folder, with the same arguments.

The runtime folder is read from ROOT.txt next to this file (written by tools/install_skill.py), else the
repository this skill folder sits in, else the folder scripts/bootstrap.py prints.
"""
import os
import runpy
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)


def _is_root(path):
    return bool(path) and os.path.isfile(os.path.join(path, "tools", "compose.py")) \
        and os.path.abspath(os.path.join(path, "tools")) != HERE


def find_root():
    try:
        with open(os.path.join(HERE, "ROOT.txt"), encoding="utf-8") as f:
            root = f.read().strip()
        if _is_root(root):
            return root
    except OSError:
        pass
    repo = os.path.dirname(os.path.dirname(SKILL))
    if _is_root(repo):
        return repo
    out = subprocess.run([sys.executable, os.path.join(SKILL, "scripts", "bootstrap.py")],
                         capture_output=True, text=True)
    lines = [x.strip() for x in out.stdout.splitlines() if x.strip()]
    if lines and _is_root(lines[-1]):
        return lines[-1]
    sys.exit("pixelcanvas runtime not found: run python " + os.path.join(SKILL, "scripts", "bootstrap.py"))


if __name__ == "__main__":
    tool = os.path.join(find_root(), "tools", "compose.py")
    sys.argv[0] = tool
    runpy.run_path(tool, run_name="__main__")
