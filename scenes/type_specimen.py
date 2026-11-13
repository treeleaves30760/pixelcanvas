# -*- coding: utf-8 -*-
"""
Type specimen sheet: every text effect, arc text, vertical Japanese and Chinese, rich text and mixed-script fallback.

* Format: 1920x1200 landscape.
* Left 64%: a 4 x 4 grid of effect tiles, each tile its own small "surface" (paper, chalkboard, night, colour).
* Right column: a circular seal with arc text, vertical Japanese + Chinese columns, a rich-text paragraph with a drop
  cap, and a mixed-script line whose glyphs come from several fallback fonts.
* Header: serif title + kicker; footer: the live result of the text audit.
"""
import numpy as np

from pixelcanvas import Canvas, fbm2d, lerp, sdf
from pixelcanvas.layout import Box, Layout
from pixelcanvas.text import audit_text, text_on_arc

NAME = "type_specimen"
DEFAULT_SIZE = (1920, 1200)
DEFAULT_SEED = 2

PAPER = "#f3eee4"
INK = "#1d1a17"
ACCENT = "#a8321f"
MUTED = "#4f463c"
TILES = [   # (label, word, surface, ink, effect, extra cv.text keywords)
    ("gold", "Gold", "#15130f", "#f2c14e", "gold", {"font": "serif bold"}),
    ("neon", "Neon", "#0d0820", "#ff3cac", dict(kind="neon", off=[2]), {"font": "sans bold"}),
    ("extrude", "3D", "#ffd166", "#ef476f", dict(kind="extrude", depth=0.16, color="#7a1f3d"), {"font": "display"}),
    ("long_shadow", "Long", "#26a69a", "#ffffff", dict(kind="long_shadow", color="#16675f", length=0.45), {"font": "sans black"}),
    ("sticker", "Wow!", "#4f52c4", "#f72585", dict(kind="sticker", color="#ffffff"), {"font": "display"}),
    ("chalk", "Chalk", "#26332c", "#f1f1e8", "chalk", {"font": "handwriting"}),
    ("glitch", "GLITCH", "#0b0b10", "#f5f5f5", "glitch", {"font": "sans black"}),
    ("stripes", "Retro", "#1b1035", None, "stripes", {"font": "display", "gradient": [(0, "#ffe66d"), (0.55, "#ff8c42"), (1, "#ff3c6e")]}),
    ("letterpress", "Press", "#ede6d6", "#a8321f", "letterpress", {"font": "serif bold"}),
    ("emboss", "Emboss", "#e9e4dc", "#e9e4dc", "emboss", {"font": "serif bold"}),
    ("knockout", "Cut", "#ff7f50", "#ffffff", dict(kind="knockout", color="#14213d"), {"font": "sans black"}),
    ("stencil", "CARGO", "#556b2f", "#f0e6c8", "stencil", {"font": "display"}),
    ("outline_only", "Hollow", "#14213d", "#8ecae6", dict(kind="outline_only", width=0.03), {"font": "sans black"}),
    ("chrome", "Chrome", "#101820", "#ffffff", "chrome", {"font": "sans black"}),
    ("foil", "Foil", "#2b0f2e", "#f2c14e", "foil", {"font": "script"}),
    ("rose_gold", "Rosé", "#1e1a1d", "#f2b9a0", "rose_gold", {"font": "serif bold italic"}),
]
SEAL_TOP = "PIXELCANVAS TYPE FOUNDRY"
SEAL_BOTTOM = "SET IN CODE · 2026"
JA = "縦書きの日本語、「かぎ括弧」も正しく回転します。"
ZH = "直排中文，標點符號（括號）與「引號」皆就定位。"
RICH = ("{b|Rich text} in one block: **bold**, *italic*, {#c2412d|colour}, {size=1.35|bigger}, "
        "==highlighter==, __underline__, ~~strike~~ and {sc|small caps}. Lines are balanced, "
        "kerned and justified, with no widow left behind.")
MIXED = "Café → ∑ ∞ ★ ✓ 中文 かな 한국어 Ελληνικά ½ €"


def render(width=1920, height=1200, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = min(W, H) / 1080
    cv = Canvas(W, H)
    X, Y = cv.xy()
    L = Layout(cv, 0.045)

    log("paper")
    cv.gradient([(0, PAPER), (1, "#e9e1d2")])
    cv.paint("#000000", (fbm2d(H, W, 300 * S, 4, seed=seed) * 0.5 + 0.5) * 0.035)

    log("header")
    head, body = L.take("top", 0.13 * H, gap=0.025 * H)
    cv.text("Type Specimen", (head[0], head[3]), 96 * S, INK, anchor="lb", font="serif bold")
    t = cv.last_bbox
    cv.text("sixteen effects · one keyword each", (head[2], t[3] - 14 * S), 22 * S, ACCENT, anchor="rb",
            case="upper", tracking=0.14, bold=True)
    cv.paint(INK, Box(head[0], head[3] + 0.012 * H, head[2], head[3] + 0.012 * H + max(2 * S, 1)).mask(cv))

    left, right = body.cols([0.64, 0.36], gap=0.03)
    foot, grid_box = left.take("bottom", 0.05 * H, gap=0.02 * H)

    log("effect tiles")
    cells = grid_box.cells(4, 4, gap=0.012)
    for cell, (label, word, surface, ink, effect, extra) in zip(cells, TILES):
        r = 14 * S
        d = sdf.sd_box(X, Y, cell.cx, cell.cy, cell.w / 2, cell.h / 2, radius=r)
        cv.paint("#000000", sdf.glow(d, 10 * S) * 0.12)
        top_c = lerp(surface, "#ffffff", 0.06)
        cv.linear((cell.cx, cell[1]), (cell.cx, cell[3]), [(0, top_c), (1, surface)], mask=sdf.fill(d))
        if label == "chalk":
            cv.paint("#ffffff", (fbm2d(H, W, 30 * S, 3, seed=seed + 9) * 0.5 + 0.5) * sdf.fill(d) * 0.05)
        word_box = cell.inset(0.16, 0.12, 0.16, 0.3)
        kw = dict(extra)
        eff = dict(effect) if isinstance(effect, dict) else effect
        if isinstance(eff, dict):
            for k in ("depth", "width", "length"):
                if k in eff and eff[k] < 1:
                    eff[k] = eff[k] * word_box.h
        cv.text(word, box=word_box, color=ink or "#ffffff", effect=eff, seed=seed, **kw)
        cv.text(label.replace("_", " "), (cell[0] + 0.07 * cell.w, cell[3] - 0.09 * cell.h), 17 * S, "auto",
                anchor="lb", case="upper", tracking=0.16, font="mono")

    log("mixed scripts")
    cv.text(MIXED, (foot[0], foot.cy), 34 * S, INK, anchor="lm")
    mb = cv.last_bbox
    cv.text("one string, several fonts — every glyph found", (mb[2] + 24 * S, foot.cy), 21 * S, MUTED, anchor="lm",
            italic=True, font="serif")

    log("seal")
    seal_box, rest = right.take("top", 0.42 * right.h)
    cx, cy = seal_box.cx, seal_box.cy
    R = min(seal_box.w, seal_box.h) * 0.47
    outer = sdf.sd_circle(X, Y, cx, cy, R)
    cv.paint("#000000", sdf.glow(outer, 22 * S) * 0.25)
    cv.paint("#1d3557", sdf.fill(outer))
    for k, w in ((0.94, 2.2), (0.66, 2.2), (0.62, 1.0)):
        cv.paint("#f1e3c6", sdf.outline(sdf.sd_circle(X, Y, cx, cy, R * k), max(w * S, 1)))
    text_on_arc(cv, SEAL_TOP, (cx, cy), R * 0.8, R * 0.135, angle=-90, color="#f1e3c6", bold=True)
    text_on_arc(cv, SEAL_BOTTOM, (cx, cy), R * 0.8, R * 0.115, angle=90, color="#e9c46a", bold=True)
    for a in (180, 0):
        cv.text("★", (cx + R * 0.8 * np.cos(np.radians(a)), cy), R * 0.08, "#e9c46a", anchor="mm", audit=False)
    cv.text("Aa", (cx, cy - R * 0.06), R * 0.5, "#f1e3c6", anchor="mm", font="serif bold italic")
    cv.text("文字", (cx, cy + R * 0.33), R * 0.17, "#e9c46a", anchor="mm", font="cjk-serif", bold=True, tracking=0.2)

    log("vertical text")
    vert, para = rest.take("top", 0.48 * rest.h, gap=0.03 * H)
    col_size = 34 * S
    cv.text(JA, (vert[2], vert[1]), col_size, INK, anchor="ra", vertical=True, max_height=vert.h, line_spacing=1.35)
    jb = cv.last_bbox
    cv.text(ZH, (jb[0] - 0.035 * W, vert[1]), col_size, ACCENT, anchor="ra", vertical=True, max_height=vert.h,
            line_spacing=1.35, font="cjk-serif")
    zb = cv.last_bbox
    cv.text("Vertical\nJapanese\n& Chinese", (vert[0], vert[1]), 22 * S, MUTED, anchor="la", case="upper",
            tracking=0.12, bold=True, line_spacing=1.3)
    kb = cv.last_bbox
    cv.text("columns run right to left; brackets and dashes turn", (vert[0], kb[3] + 16 * S),
            21 * S, MUTED, anchor="la", max_width=zb[0] - vert[0] - 20 * S, italic=True, font="serif")

    log("rich paragraph")
    cv.text(RICH, (para[0], para[1]), 25 * S, INK, max_width=para.w, align="justify", font="serif",
            drop_cap=dict(lines=3, color=ACCENT), line_spacing=1.42)

    log("audit")
    issues = audit_text(cv)
    msg = "text audit: no issues" if not issues else f"text audit: {len(issues)} issue(s)"
    cv.text("✓ " + msg if not issues else msg, (L[2], foot.cy), 18 * S, "#1d5e37" if not issues else ACCENT,
            anchor="rm", bold=True, case="upper", tracking=0.12)

    log("post")
    cv.vignette(0.12, 0.75).grain(0.008, seed)
    cv.text_issues = issues
    return cv
