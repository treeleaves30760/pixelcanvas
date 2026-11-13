#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Art styles from pixelcanvas.stylize: contact sheets, one-off application, the list of styles.

  python tools/styles.py list                                   every style with a one-line description
  python tools/styles.py pick "a ukiyo-e print of a wave"       which style a request maps to
  python tools/styles.py options ukiyoe                         the style's options: default, range, meaning
  python tools/styles.py sheet scenes/mountains_sunset          the picture in every style → outputs/<name>_styles.jpg
  python tools/styles.py sheet photo.png --styles oil,ukiyoe,mosaic --cols 3 --out my_sheet.jpg
  python tools/styles.py apply photo.png risograph --strength 0.9 --out riso.png

``sheet`` and ``apply`` take a scene (``scenes/<name>``, just ``<name>``, or a path such as ``./art/poster.py``)
or an image file; results go to outputs/ next to the scene (or ``--out`` / env PIXELCANVAS_OUT). Scenes render at
``--size`` (default 960x540 for sheets, the scene size for apply); sheet images are shrunk to 540 px high.
``--save-each DIR`` also writes every styled picture at that resolution, ready for ``tools/look.py zoom``.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import stylize  # noqa: E402
from pixelcanvas import cli as workspace  # noqa: E402
from pixelcanvas.vision import contact_sheet  # noqa: E402

_SOURCE = {"path": None}                                  # scene file of the last load_source (for the output folder)


def _outdir():
    """outputs/ next to the scene that was loaded, else the workspace default (env PIXELCANVAS_OUT wins)."""
    return workspace.output_dir(_SOURCE["path"], ROOT)


def _size(s):
    w, _, h = str(s).lower().partition("x")
    return int(w), int(h)


def load_source(src, size=None, seed=None):
    """An image path or a scene name → (Canvas, short name)."""
    if os.path.isfile(src) and not src.endswith(".py"):
        cv = Canvas.load(src)
        if size and size[0] > 0:
            cv.resize(*size)
        elif size and cv.h > -size[0]:                    # (-max_height, 0): shrink, keep the aspect ratio
            h = -size[0]
            cv.resize(max(1, round(cv.w * h / cv.h)), h)
        return cv, os.path.splitext(os.path.basename(src))[0]
    name = os.path.splitext(os.path.basename(src.replace("\\", "/").rstrip("/")))[0]
    path = workspace.resolve_scene(src, ROOT) or workspace.resolve_scene(name, ROOT)
    if path is None:
        raise SystemExit(f"unknown scene or image {src!r} (scenes are looked up in "
                         + ", ".join(workspace.scene_dirs(ROOT)) + ")")
    _SOURCE["path"] = path
    mod = workspace.import_scene(path, ROOT)
    w, h = size or getattr(mod, "DEFAULT_SIZE", (1920, 1080))
    s = getattr(mod, "DEFAULT_SEED", 0) if seed is None else seed
    return mod.render(w, h, s, lambda m: None), name


def cmd_list(a):
    for n in stylize.names():
        print(f"{n:15s} {stylize.describe(n)}")


def cmd_pick(a):
    p = stylize.pick(a.text)
    print(p if p else "(no style named — keep the house look)")


def cmd_options(a):
    try:
        name = stylize.resolve(a.style)
        opts = stylize.options(name)
    except ValueError as e:
        raise SystemExit(f"styles.py: {e}")
    print(f"{name}: {stylize.describe(name)}")
    print("options (use --set NAME=VALUE, or apply(cv, %r, NAME=VALUE)); strength and seed work on every style:" % name)
    if not opts:
        print("  (none)")
    for n, o in opts.items():
        rng = f"  range {o['range'][0]:g}-{o['range'][1]:g}" if o["range"] else ""
        print(f"  {n:14s} default {o['default']!r}{rng}  {o['means']}")


def cmd_sheet(a):
    is_image = os.path.isfile(a.source) and not a.source.endswith(".py")
    size = _size(a.size) if a.size else ((-540, 0) if is_image else (960, 540))
    cv, name = load_source(a.source, size, a.seed)
    try:
        styles = [stylize.resolve(s.strip()) for s in a.styles.split(",")] if a.styles else stylize.names()
    except ValueError as e:
        raise SystemExit(f"styles.py: {e}")
    tiles, labels = [np.clip(cv.buf, 0, 1)], ["original"]
    if a.save_each:
        os.makedirs(a.save_each, exist_ok=True)
    for st in styles:
        t = time.time()
        out = stylize.apply(cv.buf, st, seed=a.seed or 0)
        dt = time.time() - t
        tiles.append(out)
        labels.append(f"{st}  {dt:.1f}s")
        print(f"{st:15s} {dt:5.2f}s")
        if a.save_each:
            Canvas.from_array(out).save(os.path.join(a.save_each, f"{name}_{st}.png"))
    sheet = contact_sheet(tiles, labels, cols=a.cols, tile_width=a.tile, title=f"{name} — {len(styles)} styles")
    out = a.out or os.path.join(_outdir(), f"{name}_styles.jpg")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sheet.save(out, quality=90)
    print(f"sheet  {out}   ← look at this")


def cmd_apply(a):
    try:
        a.style = stylize.resolve(a.style)
    except ValueError as e:
        raise SystemExit(f"styles.py: {e}  Run: python tools/styles.py list")
    cv, name = load_source(a.source, _size(a.size) if a.size else None, a.seed)
    params = {}
    for kv in a.set or []:
        k, _, v = kv.partition("=")
        try:
            params[k] = eval(v, {})
        except Exception:
            params[k] = v
    t = time.time()
    stylize.apply(cv, a.style, strength=a.strength, seed=a.seed or 0, **params)
    out = a.out or os.path.join(_outdir(), f"{name}_{a.style}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    prev = os.path.splitext(out)[0] + "_preview.jpg"
    cv.save(out, preview=prev)
    print(f"{a.style} in {time.time() - t:.2f}s → {out}  (preview {prev})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("pick")
    p.add_argument("text")
    p.set_defaults(fn=cmd_pick)
    p = sub.add_parser("options")
    p.add_argument("style")
    p.set_defaults(fn=cmd_options)
    p = sub.add_parser("sheet")
    p.add_argument("source")
    p.add_argument("--styles")
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--tile", type=int, default=320, help="tile width in px (default 320)")
    p.add_argument("--size", help="render size for scenes / resize for images, e.g. 960x540")
    p.add_argument("--seed", type=int)
    p.add_argument("--out")
    p.add_argument("--save-each", metavar="DIR")
    p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("apply")
    p.add_argument("source")
    p.add_argument("style")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--size")
    p.add_argument("--seed", type=int)
    p.add_argument("--set", action="append", metavar="NAME=VALUE", help="style parameter, repeatable")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_apply)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
