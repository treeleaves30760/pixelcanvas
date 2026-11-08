# -*- coding: utf-8 -*-
"""Text & typography: font discovery, multi-line layout, wrapping, fitting, and styled text rendering.

Everything is built on the usual "mask first, then colour" rule:

* ``text_mask(h, w, text, xy, ...)``  → float32 (H, W) coverage mask (anti-aliased, supersampled).
* ``draw_text(cv, text, xy, ...)``    → paints styled text on a Canvas (fill / gradient / outline / shadow / glow)
  and returns the text's bounding box, so you can place other things relative to it.
* ``text_bbox`` / ``fit_text_size``   → measure before drawing (layout, auto-fit a title into a box).
* ``find_font`` / ``list_fonts``      → cross-platform font lookup; CJK text automatically picks a CJK font.

Anchors are two letters: horizontal ``l`` / ``m`` / ``r`` + vertical ``t`` / ``m`` / ``b`` (also Pillow's
``a`` = ascender ≈ top, ``s`` = baseline of the first line, ``d`` ≈ bottom). ``"mm"`` = centred on ``xy``.
Positions and sizes are in pixels of the target canvas (multiply 1080p constants by ``S = H / 1080``).
"""
from __future__ import annotations

import glob
import os
import re
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import F32, as_color, mix

_RES = getattr(Image, "Resampling", Image)

__all__ = ["find_font", "list_fonts", "load_font", "has_cjk", "wrap_text", "text_bbox", "text_mask",
           "fit_text_size", "draw_text"]

# --------------------------------------------------------------------------------------------- fonts

_CJK_RE = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef\u3000-\u303f]")


def has_cjk(text):
    """True if the text contains CJK / full-width characters (needs a CJK-capable font)."""
    return bool(_CJK_RE.search(str(text)))


def _font_dirs():
    dirs = []
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", "C:/Windows")
        dirs += [os.path.join(windir, "Fonts"),
                 os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts")]
    elif sys.platform == "darwin":
        dirs += ["/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
                 os.path.expanduser("~/Library/Fonts")]
    else:
        dirs += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
                 os.path.expanduser("~/.local/share/fonts")]
    extra = os.environ.get("PIXELCANVAS_FONT_DIR")
    if extra:
        dirs = extra.split(os.pathsep) + dirs
    return [d for d in dirs if d and os.path.isdir(d)]


@lru_cache(maxsize=1)
def _font_index():
    """{lower-case file stem: path} for every .ttf/.otf/.ttc under the system font directories."""
    idx = {}
    for d in _font_dirs():
        for ext in ("ttf", "otf", "ttc", "TTF", "OTF", "TTC"):
            for p in glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True):
                idx.setdefault(os.path.splitext(os.path.basename(p))[0].lower(), p)
    return idx


# Preferred families, best first. Each entry is a file stem (case-insensitive) as installed by the OS.
_CANDIDATES = {
    ("sans", False): ["segoeui", "arial", "helvetica", "helveticaneue", "dejavusans", "liberationsans-regular",
                      "notosans-regular", "roboto-regular", "opensans-regular"],
    ("sans", True): ["segoeuib", "arialbd", "helvetica", "dejavusans-bold", "liberationsans-bold",
                     "notosans-bold", "roboto-bold", "opensans-bold"],
    ("serif", False): ["georgia", "times", "timesnewroman", "dejavuserif", "liberationserif-regular",
                       "notoserif-regular"],
    ("serif", True): ["georgiab", "timesbd", "dejavuserif-bold", "liberationserif-bold", "notoserif-bold"],
    ("mono", False): ["consola", "cascadiamono", "menlo", "dejavusansmono", "liberationmono-regular"],
    ("mono", True): ["consolab", "menlo", "dejavusansmono-bold", "liberationmono-bold"],
    ("cjk", False): ["msjh", "notosanstc-vf", "notosanscjk-regular", "notosanscjktc-regular", "notosanstc-regular",
                     "pingfang", "hiragino sans gb", "msyh", "notosanssc-vf", "simhei", "wqy-microhei",
                     "droidsansfallbackfull", "arialuni"],
    ("cjk", True): ["msjhbd", "notosanstc-vf", "notosanscjk-bold", "notosanscjktc-bold", "pingfang", "msyhbd",
                    "notosanssc-vf", "simhei", "wqy-microhei"],
    ("cjk-serif", False): ["notoseriftc-vf", "notoserifcjk-regular", "mingliu", "pmingliu", "songti", "simsun",
                           "msjh"],
    ("cjk-serif", True): ["notoseriftc-vf", "notoserifcjk-bold", "songti", "msjhbd"],
}


def find_font(name=None, bold=False, style="sans", text=None):
    """Return a font file path, or None (→ Pillow's built-in font).

    * ``name``: a path, or a file stem / family substring such as ``"georgia"``, ``"msjh"``, ``"Noto Serif TC"``.
    * ``style``: ``"sans"`` / ``"serif"`` / ``"mono"`` / ``"cjk"`` / ``"cjk-serif"`` (used when ``name`` is None).
    * ``text``: if it contains CJK characters and ``style`` is not CJK, a CJK font is chosen automatically.
    * Env overrides: ``PIXELCANVAS_FONT`` (Latin default), ``PIXELCANVAS_CJK_FONT`` (CJK default),
      ``PIXELCANVAS_FONT_DIR`` (extra search directories).
    """
    if name:
        if os.path.isfile(str(name)):
            return str(name)
        key = str(name).lower().replace(" ", "")
        idx = _font_index()
        if key in idx:
            return idx[key]
        hits = sorted((k for k in idx if key in k.replace(" ", "")), key=len)
        if bold:
            hits = sorted(hits, key=lambda k: ("bold" not in k and not k.endswith("bd") and not k.endswith("b"), len(k)))
        if hits:
            return idx[hits[0]]
        return None
    if text is not None and has_cjk(text) and not style.startswith("cjk"):
        style = "cjk-serif" if style == "serif" else "cjk"
    env = os.environ.get("PIXELCANVAS_CJK_FONT" if style.startswith("cjk") else "PIXELCANVAS_FONT")
    if env and os.path.isfile(env):
        return env
    idx = _font_index()
    for stem in _CANDIDATES.get((style, bool(bold)), []) + _CANDIDATES.get((style, False), []):
        if stem in idx:
            return idx[stem]
    return None


def list_fonts(pattern=""):
    """Sorted list of (stem, path) for installed fonts whose file name contains ``pattern``."""
    pat = pattern.lower().replace(" ", "")
    return sorted((k, p) for k, p in _font_index().items() if pat in k.replace(" ", ""))


@lru_cache(maxsize=64)
def _load(path, size, bold):
    if path is None:
        try:
            return ImageFont.load_default(size)             # Pillow >= 10.1: scalable built-in font
        except TypeError:
            return ImageFont.load_default()
    f = ImageFont.truetype(path, size)
    try:                                                    # variable fonts (e.g. Noto *-VF): pick a weight
        names = [n.decode() if isinstance(n, bytes) else n for n in f.get_variation_names()]
        want = "Bold" if bold else "Regular"
        if want in names:
            f.set_variation_by_name(want)
    except Exception:
        pass
    return f


_WARNED = set()


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        import warnings
        warnings.warn(msg, stacklevel=3)


def load_font(font=None, size=40, bold=False, text=None, style="sans"):
    """Pillow font object. ``font`` = path / name / None (auto, CJK-aware via ``text``) / an ImageFont."""
    if isinstance(font, (ImageFont.FreeTypeFont, ImageFont.ImageFont)):
        return font
    path = find_font(font, bold=bold, style=style, text=text)
    if path is None and text is not None and has_cjk(text):
        _warn_once("no CJK font found: CJK text will render as empty boxes. Install one (e.g. Noto Sans CJK) or set "
                   "PIXELCANVAS_CJK_FONT=/path/to/font")
    return _load(path, max(1, int(round(size))), bool(bold))


# --------------------------------------------------------------------------------------------- layout

def _line_width(font, line, tracking_px):
    if not line:
        return 0.0
    return float(font.getlength(line)) + tracking_px * max(len(line) - 1, 0)


_NO_LINE_START = set("，。、！？；：）」』】》〉,.!?;:)]}%·ー…")   # never start a line with these (kinsoku)


def wrap_text(text, font, max_width, tracking_px=0.0):
    """Greedy line-wrapping. Latin wraps at spaces; CJK can break between any two characters.
    Explicit ``\\n`` always breaks."""
    out = []
    for para in str(text).split("\n"):
        if max_width is None or _line_width(font, para, tracking_px) <= max_width:
            out.append(para)
            continue
        tokens = re.findall(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\uffef]|[^\s\u2e80-\u9fff\uac00-\ud7af\uf900-\uffef]+|\s+", para)
        line = ""
        for tok in tokens:
            trial = line + tok
            if line and _line_width(font, trial.rstrip(), tracking_px) > max_width and not tok.isspace()                     and tok[0] not in _NO_LINE_START:
                out.append(line.rstrip())
                line = tok.lstrip()
            else:
                line = trial
        out.append(line.rstrip())
    return out


def _layout(text, xy, size, font, bold, anchor, align, max_width, line_spacing, tracking):
    f = load_font(font, size, bold, text=text)
    tpx = float(tracking) * size
    lines = wrap_text(text, f, max_width, tpx)
    ascent, descent = f.getmetrics()
    lh = size * line_spacing
    widths = [_line_width(f, ln, tpx) for ln in lines]
    bw = max(widths) if widths else 0.0
    bh = lh * (len(lines) - 1) + ascent + descent
    ha, va = (anchor + "a")[0], (anchor + "a")[1]
    x, y = float(xy[0]), float(xy[1])
    x0 = x - {"l": 0.0, "m": bw / 2, "r": bw}.get(ha, 0.0)
    y0 = y - {"t": 0.0, "a": 0.0, "m": bh / 2, "b": bh, "d": bh, "s": ascent}.get(va, 0.0)
    placed = []
    for i, (ln, lw) in enumerate(zip(lines, widths)):
        al = align or {"l": "left", "m": "center", "r": "right"}.get(ha, "left")
        lx = x0 + {"left": 0.0, "center": (bw - lw) / 2, "right": bw - lw}[al]
        placed.append((ln, lx, y0 + i * lh))
    return f, placed, (x0, y0, x0 + bw, y0 + bh), tpx


def text_bbox(text, xy=(0, 0), size=40, font=None, bold=False, anchor="la", align=None, max_width=None,
              line_spacing=1.2, tracking=0.0):
    """Bounding box (x0, y0, x1, y1) the text would occupy — measure before drawing."""
    return _layout(text, xy, size, font, bold, anchor, align, max_width, line_spacing, tracking)[2]


def fit_text_size(text, box_w, box_h=None, font=None, bold=False, max_size=None, min_size=6, wrap=False,
                  line_spacing=1.2, tracking=0.0):
    """Largest font size (px) such that the text fits inside ``box_w`` x ``box_h``.
    ``wrap=True`` lets the text wrap to ``box_w`` (paragraphs); otherwise lines are kept as given."""
    hi = int(max_size or (box_h if box_h else box_w))
    lo = int(min_size)
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        x0, y0, x1, y1 = text_bbox(text, (0, 0), mid, font, bold, "la", None, box_w if wrap else None,
                                   line_spacing, tracking)
        if (x1 - x0) <= box_w and (box_h is None or (y1 - y0) <= box_h):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def _render_mask(h, w, f_size, placed, font, bold, text, tpx, aa, stroke_px=0.0):
    """Draw the laid-out lines into a supersampled 'L' image → (H, W) float mask."""
    aa = max(int(aa), 1)
    fs = load_font(font, f_size * aa, bold, text=text)
    img = Image.new("L", (w * aa, h * aa), 0)
    d = ImageDraw.Draw(img)
    sw = int(round(stroke_px * aa))
    for ln, lx, ly in placed:
        if not ln:
            continue
        if tpx == 0:
            d.text((lx * aa, ly * aa), ln, fill=255, font=fs, anchor="la", stroke_width=sw, stroke_fill=255)
        else:                                               # letter-spacing: place glyphs one by one
            cx = lx * aa
            for ch in ln:
                d.text((cx, ly * aa), ch, fill=255, font=fs, anchor="la", stroke_width=sw, stroke_fill=255)
                cx += fs.getlength(ch) + tpx * aa
    if aa > 1:
        img = img.resize((w, h), _RES.BOX)
    return np.asarray(img, F32) / 255.0


def text_mask(h, w, text, xy, size=40, font=None, bold=False, anchor="la", align=None, max_width=None,
              line_spacing=1.2, tracking=0.0, stroke=0.0, aa=2, font_path=None):
    """Anti-aliased text coverage mask, float32 (H, W).

    ``text`` may contain ``\\n``; ``max_width`` (px) enables word / CJK wrapping; ``align`` = left / center / right
    (default follows the anchor); ``tracking`` = extra letter spacing as a fraction of ``size``;
    ``stroke`` (px) grows the glyphs (use for outlines: outline = text_mask(stroke=s) - text_mask()).
    ``font_path`` is kept for backward compatibility (same as ``font``).
    """
    font = font or font_path
    f, placed, _, tpx = _layout(text, xy, size, font, bold, anchor, align, max_width, line_spacing, tracking)
    return _render_mask(h, w, size, placed, font, bold, text, tpx, aa, stroke)


# --------------------------------------------------------------------------------------------- styled drawing

def draw_text(cv, text, xy=None, size=None, color="#ffffff", font=None, bold=False, anchor="la", align=None,
              max_width=None, line_spacing=1.2, tracking=0.0, box=None, box_anchor="mm", padding=0.0,
              gradient=None, outline=0.0, outline_color="#000000", shadow=None, glow=None, opacity=1.0,
              mode="normal", aa=2):
    """Paint styled text onto a Canvas and return its bounding box (x0, y0, x1, y1).

    Placement — either ``xy`` + ``anchor`` (+ ``size``), or ``box=(x0, y0, x1, y1)``: the text is auto-sized to fit
    the box (minus ``padding`` px) and placed by ``box_anchor`` (e.g. ``"mm"`` centre, ``"lt"`` top-left).
    With ``box`` and ``max_width=True`` the text wraps to the box width (paragraph mode).

    Style (all optional, drawn back to front: glow → shadow → outline → fill):
      * ``color``     fill colour (may be HDR > 1, e.g. neon); ``gradient=[(0, c0), (1, c1)]`` → vertical gradient
                      across the text block instead.
      * ``outline``   outline width px; ``outline_color``.
      * ``shadow``    ``dict(dx=4, dy=6, blur=6, color="#000", opacity=0.5)`` (any key optional) or True.
      * ``glow``      ``dict(radius=18, color="#ff3cac", strength=1.0)`` (additive) or True.
    """
    H, W = cv.h, cv.w
    if box is not None:
        bx0, by0, bx1, by1 = (float(v) for v in box)
        bw, bh = bx1 - bx0 - 2 * padding, by1 - by0 - 2 * padding
        wrap = max_width is True
        size = size or fit_text_size(text, bw, bh, font, bold, wrap=wrap, line_spacing=line_spacing,
                                     tracking=tracking)
        max_width = bw if wrap else None
        ha, va = box_anchor[0], box_anchor[1]
        xy = ({"l": bx0 + padding, "m": (bx0 + bx1) / 2, "r": bx1 - padding}[ha],
              {"t": by0 + padding, "m": (by0 + by1) / 2, "b": by1 - padding}[va])
        anchor = ha + va
    if xy is None:
        raise ValueError("draw_text needs xy or box")
    size = size or 40
    if max_width is True:
        max_width = None
    f, placed, bbox, tpx = _layout(text, xy, size, font, bold, anchor, align, max_width, line_spacing, tracking)
    fill = _render_mask(H, W, size, placed, font, bold, text, tpx, aa)
    from .filters import gaussian_blur

    if glow:
        g = dict(radius=size * 0.35, color=color if gradient is None else gradient[-1][1], strength=1.0)
        g.update(glow if isinstance(glow, dict) else {})
        halo = gaussian_blur(fill, g["radius"] * 0.5) * 0.6 + gaussian_blur(fill, g["radius"]) * 0.8
        cv.paint(g["color"], halo * g["strength"] * opacity, "add")
    if shadow:
        s = dict(dx=size * 0.05, dy=size * 0.07, blur=size * 0.06, color="#000000", opacity=0.55)
        s.update(shadow if isinstance(shadow, dict) else {})
        base = fill if not outline else _render_mask(H, W, size, placed, font, bold, text, tpx, aa, outline)
        sh = np.roll(np.roll(base, int(round(s["dy"])), 0), int(round(s["dx"])), 1)
        if s["blur"] > 0.3:
            sh = gaussian_blur(sh, s["blur"])
        cv.paint(s["color"], sh * s["opacity"] * opacity)
    if outline:
        ring = _render_mask(H, W, size, placed, font, bold, text, tpx, aa, outline)
        cv.paint(outline_color, ring * opacity, mode)
    if gradient is not None:
        x0, y0, x1, y1 = bbox
        Y = np.arange(H, dtype=F32)[:, None] * np.ones((1, W), F32)
        col = mix(gradient, (Y - y0) / max(y1 - y0, 1.0))
    else:
        col = as_color(color)
    cv.paint(col, fill * opacity, mode)
    return bbox
