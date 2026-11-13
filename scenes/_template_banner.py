# -*- coding: utf-8 -*-
"""README banner / hero / header: backdrop, big title, tagline, a window or illustration on the right (ui.window)

* Format: 1600x640 README banner (1280x640 social preview, 1920x1080 hero, 1500x500 header)
* Focal point: <the project name, left; the terminal window on the right shows it in use>
* Text hierarchy: <KICKER -> TITLE (one or two words, huge) -> TAGLINE -> INSTALL chip>
* Light: <soft glow behind the window; the window casts a soft shadow>
* Palette: <BG gradient (top-left -> bottom-right), INK for type, ACCENT for prompt and chip>
* Exact text: <TITLE, TAGLINE, COMMANDS copied verbatim from the request>
"""
# Read skills/pixelcanvas/references/ui.md (window) and typography.md (stack, roles).
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop
from pixelcanvas.layout import Layout, stack
from pixelcanvas.ui import window

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "design"
NAME = "my_banner"
DEFAULT_SIZE = (1600, 640)
DEFAULT_SEED = 1

KICKER = "open source  \u00b7  MIT"
TITLE = "driftwood"
TAGLINE = "Your dotfiles, on every machine."
INSTALL = "brew install driftwood"           # a chip under the tagline; "" for none
PANEL = "terminal"                           # terminal, code, browser, card, or "" for a text-only banner
COMMANDS = ["$ driftwood init", "\u2713 tracking 42 files", "$ driftwood push", "$ driftwood pull --host laptop"]
BG = ["#14b8a6", "#0e5a7a", "#0b1a3a", "#1d2b64"]    # backdrop palette, light -> deep
COLORS = ("#f4f1ea", "#7ee8c7")                      # ink for the type, accent for prompt and chip


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    S = height / 1080
    INK, ACCENT = COLORS

    log("background")
    backdrop(cv, "mesh", palette=BG, seed=seed)
    cv.glow(0.74 * width, 0.5 * height, 0.45 * height, ACCENT, 0.16)

    log("window")                                        # the window hugs its lines and fits the right half
    text_w = 0.5 if PANEL else 0.86
    if PANEL:
        window(cv, PANEL, COMMANDS, box=(0.52, 0.1, 0.96, 0.9), title=f"~/{TITLE} \u2014 zsh", accent=ACCENT)

    log("text")                                          # stack sizes and spaces the lines in the left box
    L = Layout(cv, margin=0.1)
    shadow = dict(dx=0, dy=4 * S, blur=18 * S, opacity=0.45)
    items = [{"text": KICKER, "role": "kicker", "color": ACCENT, "size": 32 * S},
             {"text": TITLE, "role": "title", "color": INK, "font": "display", "max_size": 260, "shadow": shadow},
             {"text": TAGLINE, "role": "subtitle", "color": INK, "size": 64 * S, "shadow": shadow}]
    box = L.sub(0, 0.0, text_w, 0.82 if INSTALL else 1.0)
    lines = stack(cv, box, [i for i in items if i["text"]], valign="middle")
    if INSTALL:                                          # the chip sits under the tagline, inside the margin
        fs = 40 * S
        cv.text(INSTALL, (lines[-1][0] + 0.6 * fs, lines[-1][3] + 1.2 * fs), fs, "#0b1a2a", anchor="lt",
                font="mono", plate=dict(kind="pill", color=ACCENT, opacity=1.0))

    log("post")
    cv.vignette(0.18, 0.75).grain(0.006, seed)
    return cv
