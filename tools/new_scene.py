#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a new scene from a template.

  python tools/new_scene.py <name>            landscape template (pixelcanvas.scenery recipes) → scenes/<name>.py
  python tools/new_scene.py <name> --poster   design template (text, SDF shapes, layout)       → scenes/<name>.py
"""
import os
import re
import sys

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to cp950 → mojibake
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    poster = "--poster" in sys.argv[1:]
    if not args or not re.fullmatch(r"[a-z][a-z0-9_]*", args[0]):
        print(__doc__)
        print("name: lower-case letters, digits and _ (e.g. night_lake)")
        return 1
    name = args[0]
    dst = os.path.join(ROOT, "scenes", f"{name}.py")
    if os.path.exists(dst):
        print("already exists:", dst)
        return 1
    tpl = "_template_poster.py" if poster else "_template.py"
    with open(os.path.join(ROOT, "scenes", tpl), encoding="utf-8") as f:
        src = re.sub(r'NAME = "my_\w+"', f'NAME = "{name}"', f.read())
    with open(dst, "w", encoding="utf-8") as f:
        f.write(src)
    print("created:", os.path.relpath(dst))
    print("draft:   python tools/render.py", name, "--scale 0.5 --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
