#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find fonts and look at them before choosing one.

  python tools/fonts.py                       show the fonts pixelcanvas picks by default (sans/serif/mono/CJK)
  python tools/fonts.py georgia               list installed fonts whose file name contains "georgia"
  python tools/fonts.py noto --sheet          + render a specimen sheet → outputs/fonts_noto.jpg (look at it)
  python tools/fonts.py --sheet --text "霓虹之夜 Neon Nights 2026"   specimen of the defaults with your own text

Use the printed stem or path as ``font=`` in ``cv.text(...)`` / ``text_mask(...)`` / ``draw_text(...)``.
"""
from __future__ import annotations

import argparse
import os
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas.text import find_font, list_fonts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default=None)
    ap.add_argument("--sheet", action="store_true", help="render a specimen image")
    ap.add_argument("--text", default=None, help="sample text for the sheet")
    ap.add_argument("--max", type=int, default=24)
    a = ap.parse_args()

    if a.pattern:
        fonts = list_fonts(a.pattern)
        if not fonts:
            print(f"no installed font matches {a.pattern!r}")
            return 1
        for stem, path in fonts[: a.max * 4]:
            print(f"  {stem:32s} {path}")
        rows = [(stem, path) for stem, path in fonts[: a.max]]
    else:
        rows = []
        for label, kw in [("sans", {}), ("sans bold", dict(bold=True)), ("serif", dict(style="serif")),
                          ("serif bold", dict(style="serif", bold=True)), ("mono", dict(style="mono")),
                          ("cjk", dict(style="cjk")), ("cjk bold", dict(style="cjk", bold=True)),
                          ("cjk serif", dict(style="cjk-serif"))]:
            p = find_font(**kw)
            print(f"  {label:12s} {p or '(none found → Pillow built-in font, Latin only)'}")
            if p:
                rows.append((label, p))
    if not a.sheet:
        return 0

    sample = a.text or "Sphinx of black quartz 0123  天光雲影 霓虹之夜"
    row_h, W = 64, 1400
    cv = Canvas(W, row_h * len(rows) + 16, "#f4f1ea")
    for i, (label, path) in enumerate(rows):
        y = 8 + i * row_h
        cv.text(f"{label}", (16, y + row_h / 2), 15, color="#7a6f60", anchor="lm", font=find_font(None))
        # glyphs a font lacks show up as empty boxes ("tofu") — that is how you see it can't render CJK
        cv.text(sample, (300, y + row_h / 2), 34, color="#1d1b1a", anchor="lm", font=path)
    out = os.path.join(ROOT, "outputs", f"fonts_{a.pattern or 'defaults'}.jpg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv.save(out)
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
