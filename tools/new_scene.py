#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從範本建立新場景：python tools/new_scene.py <名字>  → scenes/<名字>.py"""
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):              # Windows 主控台預設 cp950 → 中文亂碼
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2 or not re.fullmatch(r"[a-z][a-z0-9_]*", sys.argv[1]):
        print("用法：python tools/new_scene.py <小寫英文名字，例如 night_lake>")
        return 1
    name = sys.argv[1]
    dst = os.path.join(ROOT, "scenes", f"{name}.py")
    if os.path.exists(dst):
        print("已存在：", dst)
        return 1
    with open(os.path.join(ROOT, "scenes", "_template.py"), encoding="utf-8") as f:
        src = f.read().replace('NAME = "my_scene"', f'NAME = "{name}"')
    with open(dst, "w", encoding="utf-8") as f:
        f.write(src)
    print("建立：", os.path.relpath(dst), "\n草稿渲染：python tools/render.py", name, "--scale 0.5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
