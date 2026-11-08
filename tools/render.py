#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a scene (scenes/<name>.py).

  python tools/render.py mountains_sunset                 full size → outputs/mountains_sunset.png (+ _preview.jpg)
  python tools/render.py mountains_sunset --scale 0.5     draft (960x540, ~4x faster) → outputs/mountains_sunset_x0.5.png
  python tools/render.py mountains_sunset --seed 7 --size 2560x1440 --out out.png
  python tools/render.py mountains_sunset --crop 700 300 1500 750      extra full-resolution crops (repeatable)
  python tools/render.py --list                           list scenes

Iterating faster (all write labelled images you can look at):

  --check                       also write <out>_check.jpg (guides | values | edges | thumbnail) + warnings
  --stages                      also write <out>_stages.jpg: the canvas after every drawing step
                                (steps = each log("...") / sc.log("...") call in the scene)
  --set NAME=EXPR               override a module-level constant before rendering (repeatable), e.g.
                                --set SUN_X=0.7  --set "SKY[1]=(0.5, '#ff8040')"  --set "LAYERS[0]['fog']=0.6"
  --sweep NAME EXPR EXPR ...    render one variant per value → <scene>_sweep_<NAME>.jpg contact sheet
  --seeds N N N ...             render one variant per seed  → <scene>_seeds.jpg contact sheet
                                (sweeps default to --scale 0.33 unless --scale is given)
"""
from __future__ import annotations

import argparse
import copy
import importlib
import os
import re
import sys
import time

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to cp950 → mojibake
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _rel(p):
    try:
        return os.path.relpath(p)
    except ValueError:                                # different drive on Windows
        return p


def list_scenes():
    d = os.path.join(ROOT, "scenes")
    return sorted(f[:-3] for f in os.listdir(d) if f.endswith(".py") and not f.startswith("_"))


def _eval(expr, ns):
    try:
        return eval(expr, dict(ns))
    except Exception:
        return expr                                   # bare words / '#ff8040' → plain string


def apply_override(mod, assignment):
    """'NAME=EXPR' or 'NAME[..][..]=EXPR' → assign inside the scene module. Returns (root, original value)."""
    lhs, _, rhs = assignment.partition("=")
    lhs, rhs = lhs.strip(), rhs.strip()
    root = re.match(r"[A-Za-z_]\w*", lhs)
    if not root or not _:
        raise SystemExit(f"--set expects NAME=EXPR, got {assignment!r}")
    root = root.group(0)
    if not hasattr(mod, root):
        names = [k for k in vars(mod) if k.isupper()]
        raise SystemExit(f"{mod.__name__} has no {root!r}. Module constants: {', '.join(names) or '(none)'}")
    original = copy.deepcopy(getattr(mod, root))
    ns = vars(mod)
    ns["__pc_value"] = _eval(rhs, ns)
    exec(f"{lhs} = __pc_value", ns)
    del ns["__pc_value"]
    return root, original


def render_one(mod, w, h, seed, log, stages=False):
    from pixelcanvas import vision
    if stages:
        vision.enable_stages(target_size=(w, h))
        inner = log

        def log(m):                                    # noqa: F811 — every log() call marks a stage
            vision.stage(None, m)
            inner(m)
    kwargs = dict(width=w, height=h, log=log)
    if seed is not None:
        kwargs["seed"] = seed
    cv = mod.render(**kwargs)
    shots = vision.take_stages(cv) if stages else None
    return cv, shots


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="module name under scenes/")
    ap.add_argument("--scale", type=float, default=None, help="resolution factor: 0.5 = draft, 1 = full (default)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", default=None, help="WxH (default: the scene's DEFAULT_SIZE)")
    ap.add_argument("--out", default=None, help="output PNG (default outputs/<scene>[_x<scale>].png)")
    ap.add_argument("--crop", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"),
                    help="also save a full-resolution crop (repeatable)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=EXPR", help="override a scene constant")
    ap.add_argument("--sweep", nargs="+", metavar=("NAME", "EXPR"), help="one variant per value → contact sheet")
    ap.add_argument("--seeds", nargs="+", type=int, help="one variant per seed → contact sheet")
    ap.add_argument("--stages", action="store_true", help="also save a sheet of every drawing step")
    ap.add_argument("--check", action="store_true", help="also save a diagnostic sheet + print warnings")
    ap.add_argument("--list", action="store_true", help="list scenes")
    args = ap.parse_args()
    if args.list or not args.scene:
        print("scenes:", ", ".join(list_scenes()))
        return 0

    from pixelcanvas import vision
    mod = importlib.import_module(f"scenes.{args.scene}")
    for s in args.set:
        apply_override(mod, s)

    multi = bool(args.sweep or args.seeds)
    scale = args.scale if args.scale is not None else (0.33 if multi else 1.0)
    w, h = getattr(mod, "DEFAULT_SIZE", (1920, 1080))
    if args.size:
        w, h = (int(v) for v in args.size.lower().split("x"))
    w, h = int(round(w * scale)), int(round(h * scale))
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:5.1f}s] {m}", flush=True)

    if multi:
        quiet = lambda m: None                          # noqa: E731
        ims, labels = [], []
        if args.seeds:
            for sd in args.seeds:
                cv, _ = render_one(mod, w, h, sd, quiet)
                ims.append(cv.to_pil()); labels.append(f"seed {sd}")
                log(f"seed {sd} done")
            name = f"{args.scene}_seeds"
        else:
            key, *values = args.sweep
            if not values:
                raise SystemExit("--sweep NAME EXPR [EXPR ...]")
            for v in values:
                root, orig = apply_override(mod, f"{key}={v}")
                cv, _ = render_one(mod, w, h, args.seed, quiet)
                setattr(mod, root, orig)
                ims.append(cv.to_pil()); labels.append(f"{key} = {v}")
                log(f"{key}={v} done")
            name = f"{args.scene}_sweep_{re.sub(r'[^A-Za-z0-9_]+', '_', key).strip('_')}"
        out = args.out or os.path.join(ROOT, "outputs", f"{name}.jpg")
        sheet = vision.contact_sheet(ims, labels, tile_width=min(640, ims[0].width), title=f"{args.scene}  {w}x{h}")
        sheet.save(out, quality=90)
        log(f"sheet   {_rel(out)}   ← look at this and pick a variant")
        return 0

    cv, shots = render_one(mod, w, h, args.seed, log, stages=args.stages)

    tag = "" if scale == 1 else f"_x{scale:g}"
    out = args.out or os.path.join(ROOT, "outputs", f"{args.scene}{tag}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    base = os.path.splitext(out)[0]
    preview = base + "_preview.jpg"
    cv.save(out, preview=preview)
    cv.stats()
    log(f"saved   {_rel(out)}")
    log(f"preview {_rel(preview)}   ← look at this")
    if args.crop:
        from PIL import Image
        im = Image.open(out)
        for x0, y0, x1, y1 in args.crop:
            p = f"{base}_crop_{x0}_{y0}_{x1}_{y1}.jpg"
            im.crop((x0, y0, x1, y1)).save(p, quality=90)
            log(f"crop    {_rel(p)}")
    if shots:
        sheet = vision.contact_sheet([im for _, im in shots], [f"{i + 1}. {lab}" for i, (lab, _) in enumerate(shots)],
                                     tile_width=min(480, w), title=f"{args.scene}: canvas after each step")
        sheet.save(base + "_stages.jpg", quality=88)
        log(f"stages  {_rel(base + '_stages.jpg')}   ({len(shots)} steps)")
    elif args.stages:
        log("stages: none captured (the scene never called log(...))")
    if args.check:
        import look
        sheet_path = base + "_check.jpg"
        look.cmd_check(argparse.Namespace(image=out, out=sheet_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
