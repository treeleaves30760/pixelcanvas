#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install skills/pixelcanvas into an AI tool's skills directory and record this project's location in the installed
SKILL.md (so "draw me a ..." works from any folder).

  python tools/install_skill.py                      → ~/.claude/skills/pixelcanvas (Claude Code, default)
  python tools/install_skill.py --tool codex         → ~/.codex/skills/pixelcanvas (Codex CLI)
  python tools/install_skill.py --dest SOME_DIR      → SOME_DIR/pixelcanvas (any tool that reads SKILL.md)
  python tools/install_skill.py --uninstall [--tool ... | --dest ...]

Tools without skill support: put AGENTS.md (or the content of skills/pixelcanvas/SKILL.md) in their system prompt /
rules file.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to cp950 → mojibake
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "skills", "pixelcanvas")
TOOLS = {"claude": "~/.claude/skills", "codex": "~/.codex/skills"}
MARK = "## Local install location"
OLD_MARK = "## 本機安裝位置"                      # marker used by older versions of SKILL.md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", choices=sorted(TOOLS), default="claude")
    ap.add_argument("--dest", help="skills directory (pixelcanvas/ is created inside); overrides --tool")
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args()

    dst = os.path.join(os.path.expanduser(a.dest or TOOLS[a.tool]), "pixelcanvas")
    if os.path.abspath(dst) == os.path.abspath(SRC):
        print("destination is the source folder; nothing to install")
        return 1
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if a.uninstall:
        print("removed", dst)
        return 0

    shutil.copytree(SRC, dst)
    p = os.path.join(dst, "SKILL.md")
    with open(p, encoding="utf-8") as f:
        text = f.read()
    head = text.split(MARK)[0].split(OLD_MARK)[0]
    text = (f"{head}{MARK}\n\nProject root: `{ROOT}`\n"
            f"(Run every `python tools/...` command from this folder; after moving it, re-run "
            f"`python tools/install_skill.py`.)\n")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("installed", dst)
    print("project  ", ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
