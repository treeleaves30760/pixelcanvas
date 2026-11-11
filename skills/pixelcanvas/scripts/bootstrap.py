#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find (or download) the pixelcanvas runtime and make sure numpy + Pillow are importable.

An installed skill (`npx skills add treeleaves30760/pixelcanvas`) contains only SKILL.md + references/, while the
renderer (pixelcanvas/, tools/, scenes/) lives in the repository. This script bridges the two:

  python bootstrap.py              print the project root (last line of output); download it on first use
  python bootstrap.py --update     also `git pull` an existing download
  python bootstrap.py --home DIR   use / create the runtime in DIR (default ~/.pixelcanvas)

Search order: current directory (and parents) → $PIXELCANVAS_HOME → --home / ~/.pixelcanvas → the repository this
skill folder sits in. The first-time download is a shallow, sparse clone (under 1 MB: no gallery / docs images).
Standard library only.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to a legacy code page
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

REPO = "https://github.com/treeleaves30760/pixelcanvas.git"
SPARSE = ["pixelcanvas", "tools", "scenes", "tests", "skills"]     # root files come along in cone mode


def is_root(path):
    return bool(path) and os.path.isfile(os.path.join(path, "pixelcanvas", "__init__.py")) \
        and os.path.isfile(os.path.join(path, "tools", "render.py"))


def find_root(home):
    here = os.path.abspath(os.getcwd())
    while True:
        if is_root(here):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    skill_repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    for cand in (os.environ.get("PIXELCANVAS_HOME"), home, skill_repo):
        if cand and is_root(os.path.expanduser(cand)):
            return os.path.abspath(os.path.expanduser(cand))
    return None


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def download(dest):
    if os.path.exists(dest) and os.listdir(dest):
        sys.exit(f"{dest} exists but is not a pixelcanvas checkout; pass --home with an empty or new folder")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    r = run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, dest])
    if r.returncode == 0:
        r = run(["git", "-C", dest, "sparse-checkout", "set", *SPARSE])
    if r.returncode != 0:                                              # old git: plain shallow clone
        if os.path.isdir(dest):
            import shutil
            shutil.rmtree(dest, ignore_errors=True)
        r = run(["git", "clone", "--depth", "1", REPO, dest])
    if r.returncode != 0:
        sys.exit("download failed (is git installed and github.com reachable?):\n" + r.stderr.strip()
                 + f"\nManual: git clone {REPO}")


def ensure_deps(root):
    missing = [m for m in ("numpy", "PIL") if importlib.util.find_spec(m) is None]
    if not missing:
        return
    req = os.path.join(root, "requirements.txt")
    print("installing dependencies:", ", ".join(missing), file=sys.stderr)
    r = run([sys.executable, "-m", "pip", "install", "-q", "-r", req] if os.path.isfile(req)
            else [sys.executable, "-m", "pip", "install", "-q", "numpy", "Pillow"])
    if r.returncode != 0:
        sys.exit("pip install failed:\n" + r.stderr.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default="~/.pixelcanvas", help="where to download the runtime (default ~/.pixelcanvas)")
    ap.add_argument("--update", action="store_true", help="git pull an existing download")
    a = ap.parse_args()

    home = os.path.abspath(os.path.expanduser(a.home))
    root = find_root(home)
    if root is None:
        print(f"downloading pixelcanvas to {home} ...", file=sys.stderr)
        download(home)
        root = home
    elif a.update and os.path.isdir(os.path.join(root, ".git")):
        r = run(["git", "-C", root, "pull", "--ff-only", "-q"])
        if r.returncode != 0:
            print("update skipped:", r.stderr.strip(), file=sys.stderr)
    ensure_deps(root)
    print(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
