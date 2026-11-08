#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspection tools for the render → look → fix loop.

Coordinates: values all ≤ 1 are fractions of the image (U across, V down); larger values are pixels.

Views for a vision model (each writes an image next to the input, or to -o):
  python tools/look.py check    IMG                         ONE image, 4 views: guides | values | edges | thumbnail
  python tools/look.py guides   IMG [--step 0.1]            labelled coordinate grid + rule of thirds
  python tools/look.py zoom     IMG U0 V0 U1 V1 [--width N] enlarge a region (labels stay in full-image coords)
  python tools/look.py values   IMG [--levels 4]            squint test: blurred, posterised value map
  python tools/look.py edges    IMG                         edge-strength map (silhouettes, layer separation)
  python tools/look.py sheet    IMG... [--labels a,b,...]   labelled contact sheet (iterations, variants)
  python tools/look.py compare  A B [-o OUT]                side by side (before / after)
  python tools/look.py diff     A B [-o OUT]                difference heat map + numbers
  python tools/look.py crop     IMG X0 Y0 X1 Y1 [-o OUT]    full-resolution crop (pixels)
  python tools/look.py grid     IMG [-o PREFIX]             3x3 full-resolution tiles

Numbers (text output, also for models without vision):
  python tools/look.py critique IMG                         automatic warnings: exposure, tonal range, horizon,
                                                            focal point, saturation, balance, flat areas
  python tools/look.py probe    IMG U,V [U,V ...]           exact colour at points (hex, luma, hue)
  python tools/look.py stats    IMG                         size, luminance percentiles, clipping
  python tools/look.py report   IMG [--cols 64]             full text report: 3x3 colours / luma / texture,
                                                            dominant colours, profiles, edges, ASCII luma map
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pixelcanvas import vision  # noqa: E402

for _s in (sys.stdout, sys.stderr):              # Windows 主控台預設 cp950 → 中文亂碼
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

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


HUES = [(15, "紅"), (40, "橙"), (65, "黃"), (160, "綠"), (195, "青"), (255, "藍"), (290, "紫"), (340, "洋紅"), (360, "紅")]
RAMP = " .:-=+*#%@"


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.clip(c, 0, 1))


def _hue(c):
    """RGB → (色相角度 0~360 或 None（灰）, 飽和度 0~1)。"""
    mx, mn = float(c.max()), float(c.min())
    if mx - mn < 0.06:
        return None, 0.0
    r, g, b = c
    if mx == r:
        h = (60 * (g - b) / (mx - mn)) % 360
    elif mx == g:
        h = 60 * (b - r) / (mx - mn) + 120
    else:
        h = 60 * (r - g) / (mx - mn) + 240
    return h, (mx - mn) / max(mx, 1e-6)


def _color_name(c):
    """平均色 → 「暗 / 中 / 亮 + 色相」的粗略中文名稱。"""
    l = float(c @ np.array([0.299, 0.587, 0.114]))
    tone = "黑" if l < 0.06 else "暗" if l < 0.3 else "中" if l < 0.65 else "亮" if l < 0.94 else "白"
    h, sat = _hue(c)
    if h is None or tone in ("黑", "白"):
        return tone if tone in ("黑", "白") else f"{tone}灰"
    hue = next(n for lim, n in HUES if h < lim)
    return f"{tone}{hue}" + ("（淡）" if sat < 0.25 else "")


def _hue_str(c):
    h, sat = _hue(c)
    return "  —      " if h is None else f"{h:3.0f}° s{sat:.2f}"


def _pad(s, n):
    """依顯示寬度補空白（中文字佔 2 格）。"""
    return s + " " * max(0, n - sum(2 if ord(ch) > 0x2e7f else 1 for ch in s))


def _texture(lum):
    """區塊內的細節量：平均梯度大小 → 平 / 細緻 / 粗糙。"""
    g = np.abs(np.diff(lum, axis=0)).mean() + np.abs(np.diff(lum, axis=1)).mean()
    return g, "平" if g < 0.01 else "細緻" if g < 0.04 else "粗糙"


def cmd_report(a):
    """把圖轉成文字：讓沒有視覺能力的模型也能「看圖」判斷曝光、構圖、層次、色調。"""
    pil = Image.open(a.image).convert("RGB")
    im = np.asarray(pil, np.float32) / 255
    h, w = im.shape[:2]
    lum = im @ np.array([0.299, 0.587, 0.114], np.float32)
    print(f"# {os.path.basename(a.image)}  {w}x{h}")
    print(f"亮度 p1/p50/p99 = {np.percentile(lum, 1):.2f}/{np.percentile(lum, 50):.2f}/{np.percentile(lum, 99):.2f}"
          f"   過曝(>0.98) {np.mean(lum > 0.98) * 100:.1f}%   死黑(<0.02) {np.mean(lum < 0.02) * 100:.1f}%"
          f"   整體對比(std) {lum.std():.2f}")
    ys, xs = np.nonzero(lum >= np.percentile(lum, 99.5))
    print(f"最亮 0.5% 像素的重心 ≈ ({xs.mean() / w:.2f}, {ys.mean() / h:.2f})，"
          f"分布範圍 x {xs.min() / w:.2f}~{xs.max() / w:.2f}、y {ys.min() / h:.2f}~{ys.max() / h:.2f}（比例座標）")

    print("\n## 九宮格（r=列 由上而下, c=欄 由左而右）")
    print("（色相：0°紅 30°橙 60°黃 120°綠 180°青 240°藍 280°紫 320°粉紫/洋紅；s = 飽和度）")
    print("格    平均色   名稱          色相       亮度  對比  質感")
    for r in range(3):
        for c in range(3):
            blk = im[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
            bl = lum[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
            m = blk.reshape(-1, 3).mean(0)
            g, tex = _texture(bl)
            print(f"r{r}c{c}  {_hex(m)}  {_pad(_color_name(m), 12)}  {_hue_str(m)}  {bl.mean():.2f}  {bl.std():.2f}  {tex}({g:.3f})")

    print("\n## 主色（佔比）")
    q = pil.resize((min(w, 320), max(1, round(h * min(w, 320) / w)))).quantize(6, method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[:18], np.float32).reshape(-1, 3) / 255
    counts = np.bincount(np.asarray(q).ravel(), minlength=6)[:len(pal)]
    for i in np.argsort(-counts):
        if counts[i]:
            print(f"  {_hex(pal[i])}  {_pad(_color_name(pal[i]), 12)} {_hue_str(pal[i])}  {counts[i] / counts.sum() * 100:5.1f}%")

    print("\n## 垂直剖面（每段的平均亮度 / 色相，由上而下）— 找地平線、天空與地面的分界")
    n = 16
    for k in range(n):
        y0, y1 = k * h // n, (k + 1) * h // n
        v = float(lum[y0:y1].mean())
        print(f"  V {y0 / h:.2f}~{y1 / h:.2f}  {v:.2f} {'█' * int(round(v * 30)):<30} {_hue_str(im[y0:y1].reshape(-1, 3).mean(0))}")

    print("\n## 水平剖面（每欄的平均亮度，由左而右）— 找光源、左右平衡")
    n = 8
    print("  " + "  ".join(f"U{k / n:.2f}:{lum[:, k * w // n:(k + 1) * w // n].mean():.2f}" for k in range(n)))

    print("\n## 各欄最強的上下分界（地平線 / 稜線 / 物體頂端的位置）")
    sm = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((8, min(h, 200)), RES.BOX), np.float32) / 255
    d = np.diff(sm, axis=0)
    for k in range(8):
        order = np.argsort(-np.abs(d[:, k]))
        picks = []
        for i in order:                                    # 取最強的兩條、彼此相距 > 5%
            if all(abs(i - j) > len(d) * 0.05 for j in picks):
                picks.append(i)
            if len(picks) == 2:
                break
        txt = "；".join(f"V={(i + 1) / len(sm):.2f}（{'上亮下暗' if d[i, k] < 0 else '上暗下亮'} Δ{abs(d[i, k]):.2f}）" for i in picks)
        print(f"  U{(k + 0.5) / 8:.2f}  {txt}")

    cols = a.cols
    rows = max(1, round(cols * h / w / 2))                 # 字元約 2:1，維持比例
    al = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((cols, rows), RES.BOX), np.float32) / 255
    print(f"\n## ASCII 亮度圖（{cols}x{rows}，' '=暗 → '@'=亮）")
    print("+" + "-" * cols + "+")
    for row in al:
        print("|" + "".join(RAMP[min(int(v * len(RAMP)), len(RAMP) - 1)] for v in row) + "|")
    print("+" + "-" * cols + "+")


# ----------------------------------------------------------------------------- vision-model helpers

def _save(im, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    im.save(path, quality=90)
    print("saved", path)


def cmd_guides(a):
    _save(vision.guides(a.image, step=a.step, max_width=a.width), _out(a.image, "guides", a.out))


def cmd_zoom(a):
    box = (a.u0, a.v0, a.u1, a.v1)
    tag = "_".join(f"{v:g}" for v in box)
    _save(vision.zoom(a.image, box, out_width=a.width, grid=not a.no_grid), _out(a.image, f"zoom_{tag}", a.out))


def cmd_values(a):
    im, frac = vision.value_map(a.image, a.levels)
    print("value masses (dark → light): " + "  ".join(f"{f * 100:.0f}%" for f in frac))
    print("strong images usually have one dominant mass (>40%), a secondary one, and a small extreme accent "
          "(<10%) at the focal point.")
    _save(im, _out(a.image, f"values{a.levels}", a.out))


def cmd_edges(a):
    _save(vision.edge_map(a.image), _out(a.image, "edges", a.out))


def cmd_check(a):
    """One image with four diagnostic views, for a single quick look."""
    src = vision.to_pil(a.image)
    tw = 96
    thumb = src.resize((tw, max(1, round(src.height * tw / src.width))), RES.BOX)
    thumb = thumb.resize((640, max(1, round(src.height * 640 / src.width))), RES.NEAREST)
    views = [vision.guides(src, max_width=960), vision.value_map(src, 4)[0], vision.edge_map(src), thumb]
    labels = ["guides (fractions; thirds in magenta)", "values: 4-level squint test",
              "edges: silhouettes & detail", f"thumbnail test: does it still read at {tw} px wide?"]
    _save(vision.contact_sheet(views, labels, cols=2, tile_width=640, title=os.path.basename(a.image)),
          _out(a.image, "check", a.out))
    for sev, msg in critique(a.image):
        print(f"[{sev}] {msg}")


def cmd_sheet(a):
    labels = a.labels.split(",") if a.labels else [os.path.basename(p) for p in a.images]
    out = a.out or os.path.join(os.path.dirname(a.images[0]) or ".", "sheet.jpg")
    _save(vision.contact_sheet(a.images, labels, cols=a.cols, tile_width=a.tile, title=a.title), out)


def _point(spec, w, h):
    u, v = (float(t) for t in spec.split(","))
    if u <= 1 and v <= 1:
        return min(int(u * w), w - 1), min(int(v * h), h - 1)
    return min(int(u), w - 1), min(int(v), h - 1)


def cmd_probe(a):
    im = np.asarray(Image.open(a.image).convert("RGB"), np.float32) / 255
    h, w = im.shape[:2]
    r = max(0, a.radius)
    for spec in a.points:
        x, y = _point(spec, w, h)
        c = im[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3).mean(0)
        hh, sat = _hue(c)
        hue = "grey" if hh is None else f"hue {hh:3.0f}° sat {sat:.2f}"
        print(f"({x / w:.3f}, {y / h:.3f}) px({x},{y})  {_hex(c)}  rgb{tuple(int(round(v * 255)) for v in c)}  "
              f"luma {float(c @ np.array([0.299, 0.587, 0.114])):.2f}  {hue}")


def critique(path):
    """Heuristic warnings → [(severity, message)]. Cheap, objective checks that are easy to miss by eye."""
    pil = Image.open(path).convert("RGB")
    if pil.width > 960:
        pil = pil.resize((960, max(1, round(pil.height * 960 / pil.width))), RES.BOX)
    im = np.asarray(pil, np.float32) / 255
    h, w = im.shape[:2]
    lum = im @ np.array([0.299, 0.587, 0.114], np.float32)
    out = []
    over, crush = np.mean(lum > 0.98), np.mean(lum < 0.02)
    if over > 0.03:
        out.append(("HIGH", f"{over * 100:.1f}% blown highlights (>3%): lower glow/light strength, soft_clip / finish(knee=0.75)"))
    if crush > 0.08:
        out.append(("MED", f"{crush * 100:.1f}% crushed blacks: lift shadows (levels gamma 1.1) or add rim / bounce light"))
    p5, p95, std = np.percentile(lum, 5), np.percentile(lum, 95), lum.std()
    accent = np.percentile(lum, 99.5) - np.percentile(lum, 50)
    if p95 - p5 < 0.25 and accent < 0.3:
        out.append(("HIGH", f"narrow tonal range (p5..p95 = {p5:.2f}..{p95:.2f}) and no bright accent: looks flat / grey; "
                            "darken the darkest mass and brighten the focal point"))
    elif p95 - p5 < 0.25:
        out.append(("LOW", f"narrow tonal range (p5..p95 = {p5:.2f}..{p95:.2f}) with a bright accent: fine for a low-key / "
                           "high-key image, otherwise widen it"))
    elif std > 0.33:
        out.append(("LOW", f"very high contrast (std {std:.2f}): make sure it is intended"))
    rows = np.convolve(lum.mean(1), np.ones(5) / 5, mode="same")
    dif = np.abs(np.diff(rows[3:-3]))
    if dif.size and dif.max() > 0.04:
        yb = (np.argmax(dif) + 3.5) / h
        if 0.46 < yb < 0.54:
            out.append(("MED", f"strongest horizontal boundary at V={yb:.2f} (dead centre): move the horizon towards 1/3 or 2/3"))
    ys, xs = np.nonzero(lum >= np.percentile(lum, 99.5))
    cx, cy = xs.mean() / w, ys.mean() / h
    spread = max(xs.std() / w, ys.std() / h)
    if spread > 0.22:
        out.append(("MED", f"brightest pixels are scattered (spread {spread:.2f}): no single focal highlight — fine for "
                           "stars / city lights, otherwise strengthen one area"))
    else:
        if abs(cx - 0.5) < 0.04 and abs(cy - 0.5) < 0.04:
            out.append(("LOW", f"focal highlight at the exact centre ({cx:.2f}, {cy:.2f}): static; thirds are livelier"))
        if min(cx, cy, 1 - cx, 1 - cy) < 0.05:
            out.append(("MED", f"brightest area touches the frame edge at ({cx:.2f}, {cy:.2f}): it pulls the eye out"))
    mx, mn = im.max(-1), im.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    hot = np.mean((sat > 0.85) & (mx > 0.5))
    if hot > 0.25:
        out.append(("MED", f"{hot * 100:.0f}% of the image is near-max saturation: garish; keep strong colour for the "
                           "focal point and desaturate large areas"))
    if sat.mean() < 0.06:
        out.append(("LOW", "almost monochrome (mean saturation < 0.06): intended?"))
    lw, rw = lum[:, : w // 2].mean(), lum[:, w // 2:].mean()
    if abs(lw - rw) > 0.25:
        out.append(("LOW", f"left/right brightness very unbalanced ({lw:.2f} vs {rw:.2f}): add a counterweight"))
    flat = []
    for r in range(3):
        for c in range(3):
            blk = lum[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3]
            if np.abs(np.diff(blk, axis=0)).mean() + np.abs(np.diff(blk, axis=1)).mean() < 0.004:
                flat.append(f"r{r}c{c}")
    if len(flat) >= 6:
        out.append(("LOW", f"{len(flat)}/9 cells have almost no detail ({', '.join(flat)}): plastic look; add subtle "
                           "texture, grain or secondary shapes"))
    return sorted(out, key=lambda t: ["HIGH", "MED", "LOW"].index(t[0]))


def cmd_critique(a):
    issues = critique(a.image)
    if not issues:
        print("no automatic warnings (heuristics only — still review the image against the checklist).")
    for sev, msg in issues:
        print(f"[{sev}] {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats"); p.add_argument("image"); p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("crop"); p.add_argument("image"); [p.add_argument(k, type=int) for k in ("x0", "y0", "x1", "y1")]
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_crop)
    p = sub.add_parser("grid"); p.add_argument("image"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_grid)
    p = sub.add_parser("compare"); p.add_argument("a"); p.add_argument("b"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("diff"); p.add_argument("a"); p.add_argument("b"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_diff)
    p = sub.add_parser("report"); p.add_argument("image"); p.add_argument("--cols", type=int, default=64); p.set_defaults(fn=cmd_report)
    p = sub.add_parser("guides"); p.add_argument("image"); p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--width", type=int, default=1280); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_guides)
    p = sub.add_parser("zoom"); p.add_argument("image")
    [p.add_argument(k, type=float) for k in ("u0", "v0", "u1", "v1")]
    p.add_argument("--width", type=int, default=1024); p.add_argument("--no-grid", action="store_true")
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_zoom)
    p = sub.add_parser("values"); p.add_argument("image"); p.add_argument("--levels", type=int, default=4)
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_values)
    p = sub.add_parser("edges"); p.add_argument("image"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_edges)
    p = sub.add_parser("check"); p.add_argument("image"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("sheet"); p.add_argument("images", nargs="+"); p.add_argument("--labels")
    p.add_argument("--cols", type=int); p.add_argument("--tile", type=int, default=480); p.add_argument("--title")
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("probe"); p.add_argument("image"); p.add_argument("points", nargs="+")
    p.add_argument("--radius", type=int, default=2); p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("critique"); p.add_argument("image"); p.set_defaults(fn=cmd_critique)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
