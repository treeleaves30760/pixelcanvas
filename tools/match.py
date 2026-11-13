#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Match a reference picture's style.

  python tools/match.py REF.png               style sheet: palette + roles, line colour / weight, tone bands per
                                              region (shadow shift), highlights, values; ready-to-paste lines
  python tools/match.py REF.png DRAFT.png     the gaps between a draft and the reference, largest first, with
                                              one match figure 0-100
  add --json for one JSON document.

Inside the render loop: `python tools/render.py scene.py --check --out x.png --ref REF.png` (or a scene constant
REFERENCE = "ref.png") appends the same gap list as a REFERENCE block.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pixelcanvas import reference  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read a reference picture's style; compare a draft with it.")
    ap.add_argument("ref", help="the reference picture")
    ap.add_argument("draft", nargs="?", help="a draft to compare with the reference")
    ap.add_argument("--json", action="store_true", help="print one JSON document")
    ap.add_argument("--colors", type=int, default=8, help="palette size of the style sheet (default 8)")
    a = ap.parse_args(argv)
    for p in (a.ref, a.draft):
        if p and not os.path.exists(p):
            ap.error(f"no such file: {p}")
    if a.draft:
        c = reference.compare(reference.probe(a.ref, colors=a.colors), reference.probe(a.draft, colors=a.colors))
        if a.json:
            print(json.dumps(c, indent=1))
        else:
            print(f"REFERENCE {a.ref}  vs  {a.draft}")
            print("\n".join(reference.gap_lines(c)))
        return 0
    p = reference.probe(a.ref, colors=a.colors)
    print(json.dumps(p, indent=1) if a.json else reference.sheet_text(p, os.path.basename(a.ref)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
