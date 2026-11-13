# -*- coding: utf-8 -*-
"""
Concert poster: bilingual English + Traditional Chinese, strong typographic hierarchy.

* Format: 1080x1350 portrait (4:5).
* Focal element: the moon disc with sound-wave rings behind the title, upper middle (U, V) = (0.42, 0.38).
* Hierarchy: kicker → huge condensed title "NOCTURNE" → vertical Chinese title 夜曲 on the right → subtitle →
  circular "one night only" seal → details in three tidy columns (EN + 中文) → footer.
* Palette: midnight indigo → violet, warm moon gold accent, cream ink.
Everything typographic goes through cv.text / text_on_arc / layout, so the text audit can check it.
"""
import numpy as np

from pixelcanvas import Canvas, fbm2d, lerp, sdf
from pixelcanvas.layout import Box, Layout, stack
from pixelcanvas.text import audit_text, text_on_arc

PROMPT = ("A bilingual English and Traditional Chinese concert poster for 'NOCTURNE', an evening of piano, strings "
          "and electronics at Harbour Hall, with a moon, a 'one night only' seal and the date, venue and tickets.")
NAME = "type_poster"
KIND = "design"
DEFAULT_SIZE = (1080, 1350)
DEFAULT_SEED = 3

KICKER = "Harbour Hall presents"
TITLE = "NOCTURNE"
TITLE_ZH = "夜曲"
SUBTITLE = "Piano, strings & electronics — an evening of music for the night"
SUBTITLE_ZH = "鋼琴、弦樂與電子音樂的夜晚"
DETAILS = [("Date", "Sat 14 Nov 2026\n8 pm — doors 7 pm", "日期", "十一月十四日 晚上八點"),
           ("Venue", "Harbour Hall\n12 Pier Road", "場地", "港灣音樂廳"),
           ("Tickets", "From $28\nharbourhall.example", "票價", "二十八元起")]
SEAL_TOP = "ONE NIGHT ONLY · LIVE"
SEAL_BOTTOM = "WORLD PREMIERE 2026"
FOOTER = "Supported by the City Arts Fund · harbourhall.example"
BG = [(0.0, "#1a1240"), (0.55, "#2a1455"), (1.0, "#0c0a1e")]
INK = "#f6efe2"
GOLD = "#f2c36b"
MUTED = "#b9aee0"
MARGIN = 0.07
MOON = (0.4, 0.3)                             # moon centre u on the horizon line, radius (fraction of W)


def render(width=1080, height=1350, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    W, H = width, height
    S = min(W, H) / 1080
    cv = Canvas(W, H)
    X, Y = cv.xy()
    L = Layout(cv, MARGIN)

    log("background")
    cv.gradient(BG)
    n = fbm2d(H, W, 380 * S, 4, seed=seed) * 0.5 + 0.5
    cv.paint("#7d5cff", n * 0.08, "add")

    log("moon rising over the horizon line")
    horizon = L[3] - 0.2 * H                       # the gold rule above the details
    mx, my, mr = MOON[0] * W, horizon, MOON[1] * W
    above = np.clip((horizon - Y) / max(1.5 * S, 0.8) + 0.5, 0, 1)
    for i in range(10):
        r = mr * (1.16 + 0.17 * i)
        ring = sdf.outline(sdf.sd_circle(X, Y, mx, my, r), max(1.4 * S, 0.8))
        cv.paint(GOLD, ring * above, opacity=0.34 * 0.85 ** i)
    disc = sdf.sd_circle(X, Y, mx, my, mr)
    cv.paint(GOLD, sdf.glow(disc, 170 * S) * 0.26 * above, "add")
    shade = np.clip(((X - mx) * 0.4 + (Y - my + mr) * 0.9) / mr * 0.5 + 0.2, 0, 1)
    moon = lerp("#ffeabb", "#c27f43", shade)
    crater = fbm2d(H, W, 70 * S, 4, seed=seed + 5) * 0.5 + 0.5
    moon = moon * (0.88 + 0.18 * crater)[..., None]
    cv.paint(moon, sdf.fill(disc) * above)
    cv.paint("#ffd27a", np.exp(-((Y - horizon) / (6 * S)) ** 2) * np.exp(-((X - mx) / (mr * 1.3)) ** 2) * 0.35, "add")

    log("chinese title")
    cv.text(TITLE_ZH, (L[2], L[1] + 0.015 * H), 150 * S, INK, anchor="ra", vertical=True, font="cjk-serif", bold=True,
            shadow=dict(dx=0, dy=6 * S, blur=18 * S, opacity=0.4))
    zh_box = cv.last_bbox
    cv.text("音樂會", ((zh_box[0] + zh_box[2]) / 2, zh_box[3] + 0.02 * H), 30 * S, GOLD, anchor="mt", vertical=True,
            font="cjk", tracking=0.3)

    log("title")
    top = Box(L[0], L[1], zh_box[0] - 0.035 * W, L[1] + 0.3 * H)
    t = stack(cv, top, [
        {"text": KICKER, "role": "kicker", "color": GOLD},
        {"text": TITLE, "role": "title", "color": INK, "font": "display", "max_size": 330, "tracking": 0.01,
         "shadow": dict(dx=0, dy=10 * S, blur=26 * S, opacity=0.45)},
    ], align="left")
    title_box = t[1]

    log("subtitle")
    sub = Box(L[0], title_box[3] + 0.015 * H, zh_box[0] - 0.05 * W, title_box[3] + 0.14 * H)
    stack(cv, sub, [{"text": SUBTITLE, "role": "subtitle", "color": INK, "size": 40 * S},
                    {"text": SUBTITLE_ZH, "role": "body", "color": MUTED, "size": 30 * S, "tracking": 0.12}])

    log("seal")
    sx, sy, sr = L[2] - 0.13 * W, horizon - 0.2 * H, 0.125 * W
    star = sdf.sd_star(X, Y, sx, sy, sr, sr * 0.93, n=36)
    cv.paint("#000000", sdf.glow(star, 30 * S) * 0.35)
    cv.paint(GOLD, sdf.fill(star))
    cv.paint("#3a2410", sdf.outline(sdf.sd_circle(X, Y, sx, sy, sr * 0.8), max(1.6 * S, 1)), opacity=0.8)
    cv.paint("#3a2410", sdf.outline(sdf.sd_circle(X, Y, sx, sy, sr * 0.5), max(1.6 * S, 1)), opacity=0.8)
    text_on_arc(cv, SEAL_TOP, (sx, sy), sr * 0.65, sr * 0.15, angle=-90, color="#3a2410", bold=True)
    text_on_arc(cv, SEAL_BOTTOM, (sx, sy), sr * 0.65, sr * 0.12, angle=90, color="#3a2410", bold=True)
    cv.text("LIVE", (sx, sy - sr * 0.08), sr * 0.32, "#3a2410", anchor="mm", font="display")
    cv.text("現場", (sx, sy + sr * 0.24), sr * 0.17, "#3a2410", anchor="mm", font="cjk", bold=True, tracking=0.2)

    log("details")
    info = Box(L[0], horizon, L[2], L[3] - 0.035 * H)
    cv.paint(GOLD, Box(info[0], info[1] - 2 * S, info[2], info[1]).mask(cv), opacity=0.9)
    for col, (head, body, head_zh, body_zh) in zip(info.inset(0, 0.03 * H, 0, 0).cols(3, gap=0.05 * W), DETAILS):
        stack(cv, col, [{"text": f"{head}  {{#b9aee0|{head_zh}}}", "role": "kicker", "color": GOLD, "size": 22 * S},
                        {"text": body, "role": "body", "color": INK, "size": 28 * S, "line_spacing": 1.35},
                        {"text": body_zh, "role": "caption", "color": MUTED, "size": 22 * S}], gap=12 * S)

    log("footer")
    cv.text(FOOTER, (W / 2, L[3]), 18 * S, MUTED, anchor="mb", case="upper", tracking=0.16)

    log("post")
    cv.vignette(0.35, 0.65).grain(0.012, seed).soft_clip(0.9)
    cv.text_issues = audit_text(cv)
    return cv
