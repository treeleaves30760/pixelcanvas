#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Studio: a picture as layers that you edit step by step instead of writing it in one go.

  python tools/studio.py new fox --prompt "a fox carrying a lantern through a snowy wood at night"
  python tools/studio.py add --from ./art/fox/draft.py         one '# %% name' section per layer -> render, report
  python tools/studio.py scale fox 1.2 | move fox --by 0.05 0 | set fox outline=on shadow=ground
  python tools/studio.py look fox | undo | history | deliver ./fox.png

The picture is ./art/NAME/ (picture.json, layers/<name>.py, cache/, history/, preview.png); the last picture used
is the current one (-p NAME picks another). Every command prints a short report: the layers, the request
checklist and one NEXT line. `python tools/studio.py help` lists every command.
"""
import os
import sys

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to a legacy code page (cp950, cp1252)
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isfile(os.path.join(ROOT, "pixelcanvas", "__init__.py")):
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from pixelcanvas.studio import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
