#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 skills/pixelcanvas 安裝到 AI 工具的 skills 目錄，並在 SKILL.md 填入本專案的位置
（之後在任何資料夾裡要求「畫一張 ○○」，模型都找得到這個專案）。

  python tools/install_skill.py                      → ~/.claude/skills/pixelcanvas（Claude Code，預設）
  python tools/install_skill.py --tool codex         → ~/.codex/skills/pixelcanvas（Codex CLI）
  python tools/install_skill.py --dest 某個目錄       → 某個目錄/pixelcanvas（其他支援 SKILL.md 的工具）
  python tools/install_skill.py --uninstall [--tool ... | --dest ...]

不支援 skills 的工具：把 AGENTS.md（或 skills/pixelcanvas/SKILL.md 的內容）放進它的系統提示 / 規則檔即可。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

for _s in (sys.stdout, sys.stderr):              # Windows 主控台預設 cp950 → 中文亂碼
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "skills", "pixelcanvas")
TOOLS = {"claude": "~/.claude/skills", "codex": "~/.codex/skills"}
MARK = "## 本機安裝位置"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", choices=sorted(TOOLS), default="claude")
    ap.add_argument("--dest", help="skills 目錄（會在底下建立 pixelcanvas/），優先於 --tool")
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args()

    dst = os.path.join(os.path.expanduser(a.dest or TOOLS[a.tool]), "pixelcanvas")
    if os.path.abspath(dst) == os.path.abspath(SRC):
        print("目的地就是來源資料夾，不需要安裝")
        return 1
    if os.path.exists(dst):
        shutil.rmtree(dst)
    if a.uninstall:
        print("已移除", dst)
        return 0

    shutil.copytree(SRC, dst)
    p = os.path.join(dst, "SKILL.md")
    with open(p, encoding="utf-8") as f:
        text = f.read()
    head = text.split(MARK)[0]
    text = (f"{head}{MARK}\n\n專案根目錄：`{ROOT}`\n"
            f"（所有 `python tools/...` 指令都在這個資料夾執行；搬家後請重新執行 `python tools/install_skill.py`。）\n")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("已安裝", dst)
    print("專案位置", ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
