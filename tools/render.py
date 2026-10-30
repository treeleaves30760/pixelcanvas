#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渲染場景。

  python tools/render.py mountains_sunset                 全尺寸 → outputs/mountains_sunset.png (+ _preview.jpg)
  python tools/render.py mountains_sunset --scale 0.5     草稿（960x540，快 4 倍）→ outputs/mountains_sunset_x0.5.png
  python tools/render.py mountains_sunset --seed 7 --size 2560x1440 --out out.png
  python tools/render.py mountains_sunset --crop 700 300 1500 750 --crop 0 600 800 1080
                                                            另存全解析度局部裁切，檢查細節用
  python tools/render.py --list                           列出所有場景
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def list_scenes():
    d = os.path.join(ROOT, "scenes")
    return sorted(f[:-3] for f in os.listdir(d) if f.endswith(".py") and not f.startswith("_"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="scenes/ 底下的模組名")
    ap.add_argument("--scale", type=float, default=1.0, help="解析度倍率：0.5 = 草稿，1 = 全尺寸（預設）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", default=None, help="WxH，預設用場景的 DEFAULT_SIZE")
    ap.add_argument("--out", default=None, help="輸出 PNG 路徑（預設 outputs/<scene>[_x倍率].png）")
    ap.add_argument("--crop", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"),
                    help="另存全解析度局部裁切（可重複）")
    ap.add_argument("--list", action="store_true", help="列出場景")
    args = ap.parse_args()
    if args.list or not args.scene:
        print("scenes:", ", ".join(list_scenes()))
        return 0

    mod = importlib.import_module(f"scenes.{args.scene}")
    w, h = getattr(mod, "DEFAULT_SIZE", (1920, 1080))
    if args.size:
        w, h = (int(v) for v in args.size.lower().split("x"))
    w, h = int(round(w * args.scale)), int(round(h * args.scale))

    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:5.1f}s] {m}", flush=True)

    kwargs = dict(width=w, height=h, log=log)
    if args.seed is not None:
        kwargs["seed"] = args.seed
    cv = mod.render(**kwargs)

    tag = "" if args.scale == 1 else f"_x{args.scale:g}"
    out = args.out or os.path.join(ROOT, "outputs", f"{args.scene}{tag}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    preview = os.path.splitext(out)[0] + "_preview.jpg"
    cv.save(out, preview=preview)
    cv.stats()
    log(f"saved   {os.path.relpath(out)}")
    log(f"preview {os.path.relpath(preview)}   ← 用看圖工具檢視這張")
    if args.crop:
        from PIL import Image
        im = Image.open(out)
        for x0, y0, x1, y1 in args.crop:
            p = f"{os.path.splitext(out)[0]}_crop_{x0}_{y0}_{x1}_{y1}.jpg"
            im.crop((x0, y0, x1, y1)).save(p, quality=90)
            log(f"crop    {os.path.relpath(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
