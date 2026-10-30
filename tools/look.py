#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢視工具（迭代時配合看圖使用）。

  python tools/look.py stats   圖.png                      尺寸、亮度統計、過曝 / 死黑比例
  python tools/look.py crop    圖.png X0 Y0 X1 Y1 [-o 輸出]  全解析度局部裁切
  python tools/look.py grid    圖.png [-o 前綴]             切成 3x3 九塊全解析度小圖（逐塊檢查細節）
  python tools/look.py compare A.png B.png [-o 輸出]        左右並排（改前 / 改後）
  python tools/look.py diff    A.png B.png [-o 輸出]        差異熱圖（看改動影響到哪裡）
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

RES = getattr(Image, "Resampling", Image)


def _out(path, default_suffix, explicit):
    return explicit or f"{os.path.splitext(path)[0]}_{default_suffix}.jpg"


def cmd_stats(a):
    im = np.asarray(Image.open(a.image).convert("RGB"), np.float32) / 255
    lum = im @ np.array([0.299, 0.587, 0.114], np.float32)
    print(f"{im.shape[1]}x{im.shape[0]}  mean={im.mean():.3f}  luma p1/p50/p99 = {np.percentile(lum, 1):.3f}/{np.percentile(lum, 50):.3f}/{np.percentile(lum, 99):.3f}")
    print(f"過曝 (luma>0.98) {np.mean(lum > 0.98) * 100:.2f}%   死黑 (luma<0.02) {np.mean(lum < 0.02) * 100:.2f}%")
    print(f"平均 RGB = {im.reshape(-1, 3).mean(0).round(3)}  (偏色檢查)")


def cmd_crop(a):
    im = Image.open(a.image)
    o = _out(a.image, f"crop_{a.x0}_{a.y0}_{a.x1}_{a.y1}", a.out)
    im.crop((a.x0, a.y0, a.x1, a.y1)).save(o, quality=90)
    print("saved", o)


def cmd_grid(a):
    """切成 3x3 九塊全解析度小圖（r0c0 左上 … r2c2 右下），逐塊用看圖工具檢查細節。"""
    im = Image.open(a.image)
    w, h = im.size
    cw, ch = w // 3, h // 3
    base = os.path.splitext(a.out or a.image)[0]
    for r in range(3):
        for c in range(3):
            o = f"{base}_tile_r{r}c{c}.jpg"
            im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)).save(o, quality=90)
            print("saved", o)


def cmd_compare(a):
    A, B = Image.open(a.a).convert("RGB"), Image.open(a.b).convert("RGB")
    tw = 800
    A = A.resize((tw, round(A.height * tw / A.width)), RES.LANCZOS)
    B = B.resize((tw, round(B.height * tw / B.width)), RES.LANCZOS)
    sheet = Image.new("RGB", (tw * 2 + 4, max(A.height, B.height)), (255, 255, 255))
    sheet.paste(A, (0, 0))
    sheet.paste(B, (tw + 4, 0))
    o = _out(a.a, "compare", a.out)
    sheet.save(o, quality=88)
    print("saved", o, "(左 A / 右 B)")


def cmd_diff(a):
    A = np.asarray(Image.open(a.a).convert("RGB"), np.float32)
    B = np.asarray(Image.open(a.b).convert("RGB"), np.float32)
    if A.shape != B.shape:
        print("尺寸不同，無法比較", A.shape, B.shape)
        return
    d = np.abs(A - B).mean(-1)
    print(f"平均差異 {d.mean():.2f}/255，最大 {d.max():.0f}/255，變動 >8 的像素 {np.mean(d > 8) * 100:.1f}%")
    heat = np.clip(d * 4, 0, 255).astype(np.uint8)
    o = _out(a.a, "diff", a.out)
    Image.fromarray(heat, "L").resize((960, round(A.shape[0] * 960 / A.shape[1])), RES.BILINEAR).save(o, quality=85)
    print("saved", o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats"); p.add_argument("image"); p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("crop"); p.add_argument("image"); [p.add_argument(k, type=int) for k in ("x0", "y0", "x1", "y1")]
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_crop)
    p = sub.add_parser("grid"); p.add_argument("image"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_grid)
    p = sub.add_parser("compare"); p.add_argument("a"); p.add_argument("b"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("diff"); p.add_argument("a"); p.add_argument("b"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_diff)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
