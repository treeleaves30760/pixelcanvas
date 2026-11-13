#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One entry point for every pixelcanvas tool (the same as the ``pixelcanvas`` command of a pip install).

  python <root>/tools/pc.py new poster --kind poster --dir ./art
  python <root>/tools/pc.py render ./art/poster.py --scale 0.5 --check
  python <root>/tools/pc.py look zoom ./art/outputs/poster_x0.5.png 0.4 0.3 0.6 0.5
  python <root>/tools/pc.py doctor | version | mcp | <any tools/NAME.py>
"""
import os
import sys

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to a legacy code page (cp950, cp1252)
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isfile(os.path.join(ROOT, "pixelcanvas", "__init__.py")):
    sys.path.insert(0, ROOT)

from pixelcanvas.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
