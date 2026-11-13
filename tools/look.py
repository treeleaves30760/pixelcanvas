#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspection tools for the render -> look -> fix loop.

Coordinates: values all <= 1 are fractions of the image (U across, V down); larger values are pixels.

Views for a vision model (each writes an image next to the input, or to -o):
  python tools/look.py check    IMG                         ONE image, 4 views: guides | values | edges | thumbnail
  python tools/look.py guides   IMG [--step 0.1]            labelled coordinate grid + rule of thirds
  python tools/look.py zoom     IMG U0 V0 U1 V1 [--width N] enlarge a region (labels stay in full-image coords)
  python tools/look.py values   IMG [--levels 4]            squint test: blurred, posterised value map
  python tools/look.py edges    IMG                         edge-strength map (silhouettes, layer separation)
  python tools/look.py sheet    IMG... [--labels a,b,...]   labelled contact sheet (iterations, variants)
  python tools/look.py rounds   SCENE                       contact sheet of every numbered round (render.py --round)
  python tools/look.py compare  A B [-o OUT]                side by side (before / after)
  python tools/look.py diff     A B [-o OUT]                difference heat map + numbers
  python tools/look.py crop     IMG X0 Y0 X1 Y1 [-o OUT]    full-resolution crop (pixels)
  python tools/look.py grid     IMG [-o PREFIX]             3x3 full-resolution tiles

Numbers (text output, also for models without vision; add --json for one JSON document):
  python tools/look.py critique IMG [--json]                structured warnings: rule, severity, where, value, fix
  python tools/look.py probe    IMG U,V [U,V ...] [--json]  exact colour at points (hex, luma, hue)
  python tools/look.py stats    IMG [--json]                size, luminance percentiles, clipping
  python tools/look.py report   IMG [--cols 64] [--json]    3x3 colours / luma / texture, dominant colours,
                                                            profiles, edges, ASCII luma map

Borrowed eyes (send the image to a vision model; text, counts and positions come exactly from render.py --check):
  python tools/look.py ask      IMG "question" [--json]     ask a vision model, e.g. "does this read as a fox?"
  python tools/look.py describe IMG [--json]                objects + positions + colours, text, style, flaws
  Configure with PIXELCANVAS_VLM = "ollama:<model>" (host http://localhost:11434, or "ollama:<model>@http://host:port")
  or "openai:<model>@<base_url>" for any OpenAI-compatible endpoint (key in PIXELCANVAS_VLM_KEY);
  PIXELCANVAS_VLM_TIMEOUT = seconds (default 240).

Set PIXELCANVAS_TRACE=<file> to append one JSON line per invocation (command, arguments, outputs, seconds, warnings).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pixelcanvas import vision  # noqa: E402

for _s in (sys.stdout, sys.stderr):              # Windows consoles default to cp950 -> mojibake
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RES = getattr(Image, "Resampling", Image)
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
_OUTPUTS: list = []                               # files written by this invocation (for the trace)
_WARNINGS: list = []                              # warnings found by this invocation (for the trace)


def _out(path, default_suffix, explicit):
    return explicit or f"{os.path.splitext(path)[0]}_{default_suffix}.jpg"


def _dump(data):
    print(json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def _rnd(v, n=3):
    return round(float(v), n)


# ----------------------------------------------------------------------------- numbers

def cmd_stats(a):
    st = vision.image_stats(a.image)
    if getattr(a, "json", False):
        _dump({"file": os.path.basename(a.image), **st})
        return
    print(f"{st['width']}x{st['height']}  mean={st['mean']:.3f}  "
          f"luma p1/p50/p99 = {st['luma_p1']:.3f}/{st['luma_p50']:.3f}/{st['luma_p99']:.3f}")
    print(f"blown (luma>0.98) {st['overexposed_pct']:.2f}%   crushed (luma<0.02) {st['crushed_pct']:.2f}%")
    print(f"mean RGB = {np.array(st['mean_rgb']).round(3)}  (colour-cast check)")


def cmd_crop(a):
    im = Image.open(a.image)
    o = _out(a.image, f"crop_{a.x0}_{a.y0}_{a.x1}_{a.y1}", a.out)
    im.crop((a.x0, a.y0, a.x1, a.y1)).save(o, quality=90)
    _OUTPUTS.append(o)
    print("saved", o)


def cmd_grid(a):
    """Cut the image into 3x3 full-resolution tiles (r0c0 top-left ... r2c2 bottom-right)."""
    im = Image.open(a.image)
    w, h = im.size
    cw, ch = w // 3, h // 3
    base = os.path.splitext(a.out or a.image)[0]
    for r in range(3):
        for c in range(3):
            o = f"{base}_tile_r{r}c{c}.jpg"
            im.crop((c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)).save(o, quality=90)
            _OUTPUTS.append(o)
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
    _OUTPUTS.append(o)
    print("saved", o, "(left A / right B)")


def cmd_diff(a):
    A = np.asarray(Image.open(a.a).convert("RGB"), np.float32)
    B = np.asarray(Image.open(a.b).convert("RGB"), np.float32)
    if A.shape != B.shape:
        print("sizes differ, cannot compare", A.shape, B.shape)
        return
    d = np.abs(A - B).mean(-1)
    print(f"mean difference {d.mean():.2f}/255, max {d.max():.0f}/255, pixels changed by >8: {np.mean(d > 8) * 100:.1f}%")
    heat = np.clip(d * 4, 0, 255).astype(np.uint8)
    o = _out(a.a, "diff", a.out)
    Image.fromarray(heat, "L").resize((960, round(A.shape[0] * 960 / A.shape[1])), RES.BILINEAR).save(o, quality=85)
    _OUTPUTS.append(o)
    print("saved", o)


HUES = [(15, "red"), (40, "orange"), (65, "yellow"), (160, "green"), (195, "cyan"), (255, "blue"), (290, "purple"),
        (340, "magenta"), (360, "red")]
RAMP = " .:-=+*#%@"


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in np.clip(c, 0, 1))


def _hue(c):
    """RGB -> (hue angle 0..360 or None for grey, saturation 0..1)."""
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
    """Average colour -> rough English name: 'dark blue', 'light orange', 'pale green', 'mid grey', 'black'."""
    l = float(c @ LUMA)
    tone = "black" if l < 0.06 else "dark" if l < 0.3 else "mid" if l < 0.65 else "light" if l < 0.94 else "white"
    h, sat = _hue(c)
    if tone in ("black", "white"):
        return tone
    if h is None:
        return f"{tone} grey"
    hue = next(n for lim, n in HUES if h < lim)
    return ("pale " if sat < 0.25 else "") + f"{tone} {hue}"


def _hue_str(c):
    h, sat = _hue(c)
    return "-" if h is None else f"{h:3.0f}deg s{sat:.2f}"


def _texture(lum):
    """Detail inside a block: mean gradient magnitude -> flat / fine / coarse."""
    g = np.abs(np.diff(lum, axis=0)).mean() + np.abs(np.diff(lum, axis=1)).mean()
    return g, "flat" if g < 0.01 else "fine" if g < 0.04 else "coarse"


def report_data(path, cols=64):
    """Everything `look.py report` prints, as a dict."""
    pil = Image.open(path).convert("RGB")
    im = np.asarray(pil, np.float32) / 255
    h, w = im.shape[:2]
    lum = im @ LUMA
    ys, xs = np.nonzero(lum >= np.percentile(lum, 99.5))
    d = {"file": os.path.basename(path), "width": w, "height": h,
         "luma": {"p1": _rnd(np.percentile(lum, 1), 2), "p50": _rnd(np.percentile(lum, 50), 2),
                  "p99": _rnd(np.percentile(lum, 99), 2), "blown_pct": _rnd(np.mean(lum > 0.98) * 100, 1),
                  "crushed_pct": _rnd(np.mean(lum < 0.02) * 100, 1), "contrast_std": _rnd(lum.std(), 2)},
         "brightest": {"centroid": [_rnd(xs.mean() / w, 2), _rnd(ys.mean() / h, 2)],
                       "u_range": [_rnd(xs.min() / w, 2), _rnd(xs.max() / w, 2)],
                       "v_range": [_rnd(ys.min() / h, 2), _rnd(ys.max() / h, 2)]}}
    grid = []
    for r in range(3):
        for c in range(3):
            sl = (slice(r * h // 3, (r + 1) * h // 3), slice(c * w // 3, (c + 1) * w // 3))
            m = im[sl].reshape(-1, 3).mean(0)
            g, tex = _texture(lum[sl])
            hh, sat = _hue(m)
            grid.append({"cell": f"r{r}c{c}", "color": _hex(m), "name": _color_name(m),
                         "hue": None if hh is None else _rnd(hh, 0), "saturation": _rnd(sat, 2),
                         "luma": _rnd(lum[sl].mean(), 2), "contrast": _rnd(lum[sl].std(), 2),
                         "texture": tex, "texture_value": _rnd(g, 3)})
    d["grid"] = grid
    q = pil.resize((min(w, 320), max(1, round(h * min(w, 320) / w)))).quantize(6, method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[:18], np.float32).reshape(-1, 3) / 255
    counts = np.bincount(np.asarray(q).ravel(), minlength=6)[:len(pal)]
    dom = []
    for i in np.argsort(-counts):
        if counts[i]:
            hh, sat = _hue(pal[i])
            dom.append({"color": _hex(pal[i]), "name": _color_name(pal[i]), "hue": None if hh is None else _rnd(hh, 0),
                        "saturation": _rnd(sat, 2), "share_pct": _rnd(counts[i] / counts.sum() * 100, 1)})
    d["dominant"] = dom
    n = 16
    vp = []
    for k in range(n):
        y0, y1 = k * h // n, (k + 1) * h // n
        m = im[y0:y1].reshape(-1, 3).mean(0)
        hh, sat = _hue(m)
        vp.append({"v0": _rnd(y0 / h, 2), "v1": _rnd(y1 / h, 2), "luma": _rnd(lum[y0:y1].mean(), 2),
                   "hue": None if hh is None else _rnd(hh, 0), "saturation": _rnd(sat, 2)})
    d["vertical_profile"] = vp
    n = 8
    d["horizontal_profile"] = [{"u": _rnd(k / n, 2), "luma": _rnd(lum[:, k * w // n:(k + 1) * w // n].mean(), 2)} for k in range(n)]
    sm = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((8, min(h, 200)), RES.BOX), np.float32) / 255
    dd = np.diff(sm, axis=0)
    bnd = []
    for k in range(8):
        order = np.argsort(-np.abs(dd[:, k]))
        picks = []
        for i in order:                                    # strongest two, more than 5% apart
            if all(abs(i - j) > len(dd) * 0.05 for j in picks):
                picks.append(i)
            if len(picks) == 2:
                break
        bnd.append({"u": _rnd((k + 0.5) / 8, 2),
                    "edges": [{"v": _rnd((i + 1) / len(sm), 2), "bright_side": "above" if dd[i, k] < 0 else "below",
                               "delta": _rnd(abs(dd[i, k]), 2)} for i in picks]})
    d["boundaries"] = bnd
    rows = max(1, round(cols * h / w / 2))                 # characters are about 2:1
    al = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).resize((cols, rows), RES.BOX), np.float32) / 255
    d["ascii"] = {"cols": cols, "rows": rows,
                  "lines": ["".join(RAMP[min(int(v * len(RAMP)), len(RAMP) - 1)] for v in row) for row in al]}
    return d


def cmd_report(a):
    """Turn the image into text so a model without vision can judge exposure, composition, layering and colour."""
    d = report_data(a.image, a.cols)
    if getattr(a, "json", False):
        _dump(d)
        return
    L, b = d["luma"], d["brightest"]
    print(f"# {d['file']}  {d['width']}x{d['height']}")
    print(f"luma p1/p50/p99 = {L['p1']:.2f}/{L['p50']:.2f}/{L['p99']:.2f}   blown (>0.98) {L['blown_pct']:.1f}%   "
          f"crushed (<0.02) {L['crushed_pct']:.1f}%   contrast (std) {L['contrast_std']:.2f}")
    print(f"brightest 0.5% of pixels: centroid ~ ({b['centroid'][0]:.2f}, {b['centroid'][1]:.2f}), "
          f"spread x {b['u_range'][0]:.2f}~{b['u_range'][1]:.2f}, y {b['v_range'][0]:.2f}~{b['v_range'][1]:.2f} (fractions)")
    print("\n## 3x3 grid (r = row top to bottom, c = column left to right)")
    print("hue: 0 red, 30 orange, 60 yellow, 120 green, 180 cyan, 240 blue, 280 purple, 320 magenta; s = saturation")
    print("cell  mean     name          hue         luma  contrast  texture")
    for g in d["grid"]:
        hue = "-" if g["hue"] is None else f"{g['hue']:3.0f}deg s{g['saturation']:.2f}"
        print(f"{g['cell']}  {g['color']}  {g['name']:<12}  {hue:<10}  {g['luma']:.2f}  {g['contrast']:.2f}      "
              f"{g['texture']}({g['texture_value']:.3f})")
    print("\n## Dominant colours (share)")
    for c in d["dominant"]:
        hue = "-" if c["hue"] is None else f"{c['hue']:3.0f}deg s{c['saturation']:.2f}"
        print(f"  {c['color']}  {c['name']:<12} {hue:<10}  {c['share_pct']:5.1f}%")
    print("\n## Vertical profile (mean luma / hue per band, top to bottom) - find the horizon and the sky / ground split")
    for p in d["vertical_profile"]:
        hue = "-" if p["hue"] is None else f"{p['hue']:3.0f}deg s{p['saturation']:.2f}"
        print(f"  V {p['v0']:.2f}~{p['v1']:.2f}  {p['luma']:.2f} {'#' * int(round(p['luma'] * 30)):<30} {hue}")
    print("\n## Horizontal profile (mean luma per column, left to right) - find the light source and left/right balance")
    print("  " + "  ".join(f"U{p['u']:.2f}:{p['luma']:.2f}" for p in d["horizontal_profile"]))
    print("\n## Strongest vertical boundaries per column (horizon / ridge / object tops)")
    for bnd in d["boundaries"]:
        txt = "; ".join(f"V={e['v']:.2f} ({'bright above, dark below' if e['bright_side'] == 'above' else 'dark above, bright below'}"
                        f" d{e['delta']:.2f})" for e in bnd["edges"])
        print(f"  U{bnd['u']:.2f}  {txt}")
    A = d["ascii"]
    print(f"\n## ASCII luma map ({A['cols']}x{A['rows']}, ' '=dark -> '@'=bright)")
    print("+" + "-" * A["cols"] + "+")
    for row in A["lines"]:
        print("|" + row + "|")
    print("+" + "-" * A["cols"] + "+")


# ----------------------------------------------------------------------------- vision-model helpers

def _save(im, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    im.save(path, quality=90)
    _OUTPUTS.append(path)
    print("saved", path, file=sys.stderr if "--json" in sys.argv else sys.stdout)


def cmd_guides(a):
    _save(vision.guides(a.image, step=a.step, max_width=a.width), _out(a.image, "guides", a.out))


def zoom_source(path):
    """The image to zoom into for ``path``: the full-size PNG when given its ``_preview.jpg`` (and the other one
    when only one exists). Returns (path, note) where note says which file was used, or None."""
    base, ext = os.path.splitext(path)
    if base.endswith("_preview"):
        png = base[: -len("_preview")] + ".png"
        if os.path.exists(png):
            return png, f"zooming into {os.path.basename(png)} (full resolution) instead of the preview"
    elif not os.path.exists(path):
        prev = base + "_preview.jpg"
        if os.path.exists(prev):
            return prev, f"{os.path.basename(path)} not found; zooming into the preview {os.path.basename(prev)}"
    if not os.path.exists(path):
        raise SystemExit(f"image not found: {path}")
    return path, None


class _Frac(float):
    """A float that remembers how it was typed, so file names match the command (1.0 stays '1.0')."""
    text = ""


def _frac(s):
    v = _Frac(s)
    v.text = str(s).strip()
    return v


def cmd_zoom(a):
    box = (a.u0, a.v0, a.u1, a.v1)
    tag = "_".join(getattr(v, "text", "") or f"{v:g}" for v in box)
    src, note = zoom_source(a.image)
    w, h = Image.open(src).size
    print(f"{note or 'zooming into ' + os.path.basename(src)} ({w}x{h})")
    out = _out(src, f"zoom_{tag}", a.out)
    _save(vision.zoom(src, box, out_width=a.width, grid=not a.no_grid), out)


def cmd_values(a):
    im, frac = vision.value_map(a.image, a.levels)
    print("value masses (dark -> light): " + "  ".join(f"{f * 100:.0f}%" for f in frac))
    print("strong images usually have one dominant mass (>40%), a secondary one, and a small extreme accent "
          "(<10%) at the focal point.")
    _save(im, _out(a.image, f"values{a.levels}", a.out))


def cmd_edges(a):
    _save(vision.edge_map(a.image), _out(a.image, "edges", a.out))


def check_sheet(image, out=None):
    """Write the four-panel diagnostic sheet; returns (sheet path, warnings)."""
    src = vision.to_pil(image)
    tw = 96
    thumb = src.resize((tw, max(1, round(src.height * tw / src.width))), RES.BOX)
    thumb = thumb.resize((640, max(1, round(src.height * 640 / src.width))), RES.NEAREST)
    views = [vision.guides(src, max_width=960), vision.value_map(src, 4)[0], vision.edge_map(src), thumb]
    labels = ["guides (fractions; thirds in magenta)", "values: 4-level squint test",
              "edges: silhouettes & detail", f"thumbnail test: does it still read at {tw} px wide?"]
    path = _out(image, "check", out)
    sheet = vision.contact_sheet(views, labels, cols=2, tile_width=640, title=os.path.basename(str(image)))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sheet.save(path, quality=90)
    _OUTPUTS.append(path)
    warns = vision.critique(image)
    _WARNINGS.extend(warns)
    return path, warns


def summary_counts(warns):
    return {s: sum(1 for w in warns if w["severity"] == s) for s in vision.SEVERITIES}


def check_block(path, warns):
    """The compact printed block: header line, then warnings sorted by severity."""
    n = summary_counts(warns)
    head = ", ".join(f"{n[s]} {s}" for s in vision.SEVERITIES if n[s]) or "no warnings"
    lines = [f"check: {path}  ({head})"]
    lines += [vision.format_warning(w) for w in warns]
    return "\n".join(lines)


def cmd_check(a):
    """One image with four diagnostic views, for a single quick look, plus the warnings."""
    path, warns = check_sheet(a.image, getattr(a, "out", None))
    if getattr(a, "json", False):
        _dump({"file": os.path.basename(a.image), "check": path, "warnings": warns})
        return
    print(check_block(path, warns))


def cmd_sheet(a):
    labels = a.labels.split(",") if a.labels else [os.path.basename(p) for p in a.images]
    out = a.out or os.path.join(os.path.dirname(a.images[0]) or ".",
                                  os.path.splitext(os.path.basename(a.images[0]))[0] + "_sheet.jpg")
    _save(vision.contact_sheet(a.images, labels, cols=a.cols, tile_width=a.tile, title=a.title), out)


def save_animation(frames, path, fps=12, fmt=None, dither=False, durations=None):
    """Write PIL ``frames`` as an endlessly looping GIF / WebP / APNG. GIF frames share one global palette built
    from several frames (no colour flicker); ``dither`` turns on Floyd-Steinberg. ``durations`` = ms per frame.
    Returns {"path", "format", "frames", "bytes"}."""
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "gif").lower()
    fmt = {"png": "apng"}.get(fmt, fmt)
    frames = [f.convert("RGB") for f in frames]
    if not frames:
        raise SystemExit("no frames to write")
    dur = durations or int(round(1000.0 / max(fps, 0.1)))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if fmt == "gif":
        picks = sorted({round(k * (len(frames) - 1) / 5) for k in range(6)}) if len(frames) > 6 else range(len(frames))
        w, h = frames[0].size
        tw = max(1, min(w, 480))
        sample = [frames[i].resize((tw, max(1, round(h * tw / w)))) for i in picks]
        mosaic = Image.new("RGB", (sample[0].width, sum(s.height for s in sample)))
        y = 0
        for s in sample:
            mosaic.paste(s, (0, y))
            y += s.height
        pal = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        dth = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
        q = [f.quantize(palette=pal, dither=dth) for f in frames]
        q[0].save(path, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False, disposal=1)
    elif fmt == "webp":
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0, quality=80, method=4)
    elif fmt == "apng":
        frames[0].save(path, format="PNG", save_all=True, append_images=frames[1:], duration=dur, loop=0)
    else:
        raise SystemExit(f"unknown animation format {fmt!r}: use gif, webp or apng")
    _OUTPUTS.append(path)
    return {"path": path, "format": fmt, "frames": len(frames), "bytes": os.path.getsize(path)}


def frames_sheet(frames, labels, cols=4, tile_width=360, title=None):
    """Filmstrip contact sheet of a few frames with labels (what a vision model reads to check the motion)."""
    return vision.contact_sheet(frames, labels, cols=min(cols, len(frames)), tile_width=min(tile_width, frames[0].width),
                                title=title)


def _caption(im, text, width=640):
    """A round preview scaled to ``width`` with a caption strip underneath."""
    from PIL import ImageDraw
    im = im.convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))), RES.LANCZOS)
    bar = 28
    out = Image.new("RGB", (im.width, im.height + bar), (24, 24, 28))
    out.paste(im, (0, 0))
    ImageDraw.Draw(out).text((8, im.height + 5), text[:90], font=vision.label_font(15), fill=(235, 235, 235))
    return out


def rounds_gif(rounds, labels, out, step_ms=900, last_ms=2500, width=640):
    """Numbered rounds -> one animated GIF, one captioned frame per round (the last one lingers)."""
    frames = [_caption(Image.open(p), lab, width) for (_, p), lab in zip(rounds, labels)]
    size = frames[0].size
    frames = [f if f.size == size else f.resize(size, RES.LANCZOS) for f in frames]
    durs = [step_ms] * len(frames)
    durs[-1] = last_ms
    return save_animation(frames, out, fmt="gif", durations=durs)


def rounds_dir(scene):
    """Folder holding the numbered drafts of a scene: outputs/rounds/<scene> (root override: env PIXELCANVAS_ROUNDS_DIR)."""
    return os.path.join(rounds_root(), scene)


def rounds_root():
    return os.environ.get("PIXELCANVAS_ROUNDS_DIR") or os.path.join(ROOT, "outputs", "rounds")


def find_rounds(scene):
    """Rounds folder of a scene name or scene path: $PIXELCANVAS_ROUNDS_DIR, else <scene outputs>/rounds/<name>
    (where render.py --round wrote them), ./outputs/rounds/<name>, <root>/outputs/rounds/<name>."""
    name = os.path.splitext(os.path.basename(scene.rstrip("/\\")))[0]
    if os.environ.get("PIXELCANVAS_ROUNDS_DIR"):
        return rounds_dir(name)
    cands = []
    try:
        from pixelcanvas import cli as workspace
        path = workspace.resolve_scene(scene, ROOT)
        if path:
            cands.append(os.path.join(workspace.output_dir(path, ROOT), "rounds", name))
        cands.append(os.path.join(workspace.output_dir(None, ROOT), "rounds", name))
    except ImportError:
        pass
    cands += [os.path.join(os.getcwd(), "outputs", "rounds", name), rounds_dir(name)]
    for c in cands:
        if list_rounds(c):
            return c
    return cands[0]


def list_rounds(folder):
    """[(number, preview path)] for r01_preview.jpg, r02_preview.jpg ... sorted by number."""
    import re
    out = []
    for f in os.listdir(folder) if os.path.isdir(folder) else []:
        m = re.fullmatch(r"r(\d+)_preview\.jpg", f)
        if m:
            out.append((int(m.group(1)), os.path.join(folder, f)))
    return sorted(out)


def cmd_rounds(a):
    """Labelled contact sheet of every numbered round: the 'how it improved' strip."""
    folder = a.scene if os.path.isdir(a.scene) else find_rounds(a.scene)
    rounds = list_rounds(folder)
    if not rounds:
        have = sorted(os.listdir(rounds_root())) if os.path.isdir(rounds_root()) else []
        raise SystemExit(f"no rounds for {a.scene!r} in {folder}. Save drafts with: python tools/render.py {a.scene} --round"
                         + (f"\nscenes with rounds: {', '.join(have)}" if have else ""))
    meta = {}
    jl = os.path.join(folder, "rounds.jsonl")
    if os.path.exists(jl):
        for line in open(jl, encoding="utf-8"):
            try:
                rec = json.loads(line)
                meta[int(rec["round"])] = rec
            except (ValueError, KeyError):
                pass
    labels = []
    for n, _ in rounds:
        rec = meta.get(n, {})
        wc = rec.get("warnings") or {}
        extra = []
        if wc:
            extra.append(", ".join(f"{wc[s]} {s}" for s in vision.SEVERITIES if wc.get(s)) or "clean")
        if rec.get("set"):
            extra.append(" ".join(rec["set"])[:40])
        labels.append(f"r{n:02d}" + ("  " + " | ".join(extra) if extra else ""))
    scene = os.path.basename(os.path.normpath(folder))
    if getattr(a, "gif", False):
        gif = a.out or os.path.join(folder, "rounds.gif")
        info = rounds_gif(rounds, labels, gif, width=a.tile + 220 if a.tile < 600 else a.tile)
        print(f"saved {gif}  ({len(rounds)} rounds, {info['bytes'] / 1e6:.2f} MB)", file=sys.stderr if getattr(a, "json", False) else sys.stdout)
        if getattr(a, "json", False):
            _dump({"scene": os.path.basename(os.path.normpath(folder)), "gif": gif, "bytes": info["bytes"],
                   "rounds": [{"round": n, "preview": p} for n, p in rounds]})
        return
    out = a.out or os.path.join(folder, "rounds_sheet.jpg")
    sheet = vision.contact_sheet([p for _, p in rounds], labels, cols=a.cols, tile_width=a.tile, title=f"{scene}: {len(rounds)} rounds")
    _save(sheet, out)
    if getattr(a, "json", False):
        _dump({"scene": scene, "sheet": out, "rounds": [{"round": n, "preview": p, **{k: v for k, v in meta.get(n, {}).items() if k != "round"}} for n, p in rounds]})


def _point(spec, w, h):
    try:
        u, v = (float(t) for t in spec.split(","))
    except ValueError:
        raise SystemExit(f"bad point {spec!r}: expected U,V (e.g. 0.5,0.3 as fractions or 480,160 in pixels)")
    if u <= 1 and v <= 1:
        return min(int(u * w), w - 1), min(int(v * h), h - 1)
    return min(int(u), w - 1), min(int(v), h - 1)


def cmd_probe(a):
    im = np.asarray(Image.open(a.image).convert("RGB"), np.float32) / 255
    h, w = im.shape[:2]
    r = max(0, a.radius)
    pts = []
    for spec in a.points:
        x, y = _point(spec, w, h)
        c = im[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3).mean(0)
        hh, sat = _hue(c)
        pts.append({"u": _rnd(x / w), "v": _rnd(y / h), "x": x, "y": y, "color": _hex(c),
                    "rgb": [int(round(v * 255)) for v in c], "luma": _rnd(c @ LUMA, 2),
                    "hue": None if hh is None else _rnd(hh, 0), "saturation": _rnd(sat, 2)})
    if getattr(a, "json", False):
        _dump({"file": os.path.basename(a.image), "radius": r, "points": pts})
        return
    for p in pts:
        hue = "grey" if p["hue"] is None else f"hue {p['hue']:3.0f}deg sat {p['saturation']:.2f}"
        print(f"({p['u']:.3f}, {p['v']:.3f}) px({p['x']},{p['y']})  {p['color']}  rgb{tuple(p['rgb'])}  luma {p['luma']:.2f}  {hue}")


_SEV_OLD = {"fail": "HIGH", "warn": "MED", "info": "LOW"}


def critique(path):
    """Compatibility wrapper: [(severity HIGH|MED|LOW, message)]. Use vision.critique for the structured form."""
    return [(_SEV_OLD[w["severity"]], f"{w['message']} {w['fix']}") for w in vision.critique(path)]


def cmd_critique(a):
    warns = vision.critique(a.image)
    _WARNINGS.extend(warns)
    if getattr(a, "json", False):
        _dump({"file": os.path.basename(a.image), "counts": summary_counts(warns), "warnings": warns})
        return
    if not warns:
        print("no automatic warnings.")
    for w in warns:
        print(vision.format_warning(w))


# ----------------------------------------------------------------------------- borrowed eyes (vision model)

DESCRIBE = ("Describe this image for someone who cannot see it. Answer in four short parts. "
            "1) Objects: every distinct object or figure, with its rough position (left / centre / right, "
            "top / middle / bottom) and its main colours. 2) Text: any text, copied exactly as written, or 'none'. "
            "3) Style: the overall style or medium (photo, flat vector, watercolour, diagram ...). "
            "4) Flaws: anything that looks wrong (cut-off objects, artifacts, unreadable text, odd shapes, "
            "things floating). Be factual; do not guess what is not visible.")


class VLMError(Exception):
    """A vision-model problem worth one clear line (not configured, unreachable, bad reply)."""


def vlm_config(spec=None):
    """PIXELCANVAS_VLM -> dict(provider, model, url, key) or None when nothing is configured."""
    spec = (os.environ.get("PIXELCANVAS_VLM", "") if spec is None else spec).strip()
    if not spec:
        return None
    provider, sep, rest = spec.partition(":")
    provider = provider.lower()
    if not sep or provider not in ("ollama", "openai") or not rest:
        raise VLMError(f"PIXELCANVAS_VLM={spec!r} is not understood: use 'ollama:<model>[@http://host:port]' or "
                       "'openai:<model>@<base_url>'")
    model, _, url = rest.partition("@")
    if provider == "ollama":
        url = url or "http://localhost:11434"
    elif not url:
        raise VLMError("openai:<model>@<base_url> needs the base URL, e.g. openai:gpt-4o-mini@https://api.openai.com/v1")
    return {"provider": provider, "model": model, "url": url.rstrip("/"), "key": os.environ.get("PIXELCANVAS_VLM_KEY")}


def _image_b64(path, max_side=1280):
    import base64
    import io
    im = Image.open(path).convert("RGB")
    s = max(im.size)
    if s > max_side:
        k = max_side / s
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), RES.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii"), im.size


def _post(url, body, headers, timeout):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise VLMError(f"vision model at {url} answered HTTP {e.code}: {detail}")
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise VLMError(f"vision model unreachable at {url}: {reason}")
    except ValueError:
        raise VLMError(f"vision model at {url} did not return JSON")


def ask_vlm(image, question, cfg=None, timeout=None):
    """Send ``image`` (path) and ``question`` to the configured vision model; returns the answer text."""
    cfg = cfg or vlm_config()
    if cfg is None:
        raise VLMError("no vision model configured: set PIXELCANVAS_VLM=ollama:<model> (local Ollama) or "
                       "openai:<model>@<base_url>; exact text, counts and positions are in render.py --check")
    timeout = float(timeout or os.environ.get("PIXELCANVAS_VLM_TIMEOUT", 240))
    b64, _ = _image_b64(image)
    if cfg["provider"] == "ollama":
        body = {"model": cfg["model"], "stream": False, "think": False, "options": {"temperature": 0},
                "messages": [{"role": "user", "content": question, "images": [b64]}]}
        try:
            r = _post(cfg["url"] + "/api/chat", body, {}, timeout)
        except VLMError as e:
            if "think" not in str(e):
                raise
            body.pop("think")                             # servers / models without the thinking switch
            r = _post(cfg["url"] + "/api/chat", body, {}, timeout)
        ans = (r.get("message") or {}).get("content", "")
    else:
        hdr = {"Authorization": f"Bearer {cfg['key']}"} if cfg.get("key") else {}
        body = {"model": cfg["model"], "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}
        r = _post(cfg["url"] + "/chat/completions", body, hdr, timeout)
        try:
            ans = r["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise VLMError(f"unexpected reply from {cfg['url']}: {str(r)[:200]}")
    import re
    ans = re.sub(r"<think>.*?</think>", "", ans or "", flags=re.S).strip()
    if not ans:
        raise VLMError(f"the vision model {cfg['model']} returned an empty answer")
    return ans


def cmd_ask(a, question=None):
    q = question or a.question
    t0 = time.time()
    try:
        cfg = vlm_config()
        ans = ask_vlm(a.image, q, cfg)
    except VLMError as e:
        if getattr(a, "json", False):
            _dump({"file": os.path.basename(a.image), "question": q, "error": str(e)})
        else:
            print(str(e), file=sys.stderr)
        raise SystemExit(2)
    if getattr(a, "json", False):
        _dump({"file": os.path.basename(a.image), "model": f"{cfg['provider']}:{cfg['model']}", "question": q,
               "answer": ans, "seconds": round(time.time() - t0, 1)})
        return
    print(ans)
    print(f"\n({cfg['provider']}:{cfg['model']}, {time.time() - t0:.1f}s. Vision models misread small text, counts and "
          "positions: take those from render.py --check.)")


def cmd_describe(a):
    cmd_ask(a, DESCRIBE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def js(p):
        p.add_argument("--json", action="store_true", help="print one JSON document instead of text")

    p = sub.add_parser("stats"); p.add_argument("image"); js(p); p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("crop"); p.add_argument("image"); [p.add_argument(k, type=int) for k in ("x0", "y0", "x1", "y1")]
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_crop)
    p = sub.add_parser("grid"); p.add_argument("image"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_grid)
    p = sub.add_parser("compare"); p.add_argument("a"); p.add_argument("b"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("diff"); p.add_argument("a"); p.add_argument("b"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_diff)
    p = sub.add_parser("report"); p.add_argument("image"); p.add_argument("--cols", type=int, default=64); js(p)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("guides"); p.add_argument("image"); p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--width", type=int, default=1280); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_guides)
    p = sub.add_parser("zoom"); p.add_argument("image")
    [p.add_argument(k, type=_frac) for k in ("u0", "v0", "u1", "v1")]
    p.add_argument("--width", type=int, default=1024); p.add_argument("--no-grid", action="store_true")
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_zoom)
    p = sub.add_parser("values"); p.add_argument("image"); p.add_argument("--levels", type=int, default=4)
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_values)
    p = sub.add_parser("edges"); p.add_argument("image"); p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_edges)
    p = sub.add_parser("check"); p.add_argument("image"); p.add_argument("-o", "--out"); js(p); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("sheet"); p.add_argument("images", nargs="+"); p.add_argument("--labels")
    p.add_argument("--cols", type=int); p.add_argument("--tile", type=int, default=480); p.add_argument("--title")
    p.add_argument("-o", "--out"); p.set_defaults(fn=cmd_sheet)
    p = sub.add_parser("rounds", help="contact sheet of the numbered drafts saved by render.py --round")
    p.add_argument("scene", help="scene name (or a folder of rNN_preview.jpg files)")
    p.add_argument("--cols", type=int); p.add_argument("--tile", type=int, default=420)
    p.add_argument("--gif", action="store_true", help="write an animated GIF of the rounds instead of a contact sheet")
    p.add_argument("-o", "--out"); js(p); p.set_defaults(fn=cmd_rounds)
    p = sub.add_parser("probe"); p.add_argument("image"); p.add_argument("points", nargs="+")
    p.add_argument("--radius", type=int, default=2); js(p); p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("critique"); p.add_argument("image"); js(p); p.set_defaults(fn=cmd_critique)
    p = sub.add_parser("ask", help="ask a vision model about the image (PIXELCANVAS_VLM)")
    p.add_argument("image"); p.add_argument("question"); js(p); p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("describe", help="vision model: objects, positions, colours, text, style, flaws")
    p.add_argument("image"); js(p); p.set_defaults(fn=cmd_describe)
    a = ap.parse_args(argv)
    t0 = time.time()
    try:
        a.fn(a)
    finally:
        if os.environ.get("PIXELCANVAS_TRACE"):
            args = {k: v for k, v in vars(a).items() if k != "fn"}
            vision.trace_event("look", a.cmd, args, list(_OUTPUTS), time.time() - t0, list(_WARNINGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
