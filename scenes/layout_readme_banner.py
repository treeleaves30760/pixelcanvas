# -*- coding: utf-8 -*-
"""README banner for a dotfiles CLI: bold name and tagline on the left, a terminal window with three commands right

* Format: 1600x640 README banner
* Focal point: the name driftwood; the terminal on the right shows the tool in use
* Text hierarchy: kicker -> name (huge, bold) -> tagline -> install chip
* Palette: calm teal-to-navy mesh, warm white type, mint accent for prompts and the chip
* Exact text: driftwood; your shell, everywhere; the three commands
"""
# Read skills/pixelcanvas/references/ui.md (window) and typography.md (stack, roles).
from pixelcanvas import Canvas
from pixelcanvas.backdrop import backdrop
from pixelcanvas.layout import Layout, stack
from pixelcanvas.ui import window

PROMPT = ("A README banner for an open-source command-line tool called 'driftwood' that syncs dotfiles: the name in "
          "bold type, the tagline 'your shell, everywhere', a stylised terminal window showing three commands, and "
          "a calm teal-to-navy background.")
NAME = "layout_readme_banner"
KIND = "design"
DEFAULT_SIZE = (1600, 640)
DEFAULT_SEED = 1

KICKER = "open source  ·  MIT"
TITLE = "driftwood"
TAGLINE = "your shell, everywhere"
INSTALL = "brew install driftwood"
COMMANDS = ["$ driftwood init", "✓ tracking 42 dotfiles", "$ driftwood push", "✓ pushed to github.com/sam/dots",
            "$ driftwood pull --host laptop", "✓ laptop is in sync"]
BG = ["#14b8a6", "#0e5a7a", "#0b1a3a", "#1d2b64"]        # teal -> navy
INK, ACCENT = "#f4f1ea", "#7ee8c7"


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    S = height / 1080

    log("background")
    backdrop(cv, "mesh", palette=BG, seed=seed)
    cv.glow(0.74 * width, 0.5 * height, 0.45 * height, ACCENT, 0.16)

    log("terminal")                                      # the window hugs its lines inside the box
    window(cv, "terminal", COMMANDS, box=(0.52, 0.1, 0.96, 0.9), title=f"~/{TITLE} — zsh", accent=ACCENT)

    log("text")                                          # stack sizes and spaces the lines in the left half
    shadow = dict(dx=0, dy=4 * S, blur=18 * S, opacity=0.45)
    items = [{"text": KICKER, "role": "kicker", "color": ACCENT, "size": 32 * S},
             {"text": TITLE, "role": "title", "color": INK, "font": "display", "max_size": 260, "shadow": shadow},
             {"text": TAGLINE, "role": "subtitle", "color": INK, "size": 64 * S, "shadow": shadow}]
    lines = stack(cv, Layout(cv, margin=0.1).sub(0, 0.0, 0.5, 0.82), items, valign="middle")

    log("install chip")
    fs = 40 * S
    cv.text(INSTALL, (lines[-1][0] + 0.6 * fs, lines[-1][3] + 1.2 * fs), fs, "#0b1a2a", anchor="lt", font="mono",
            plate=dict(kind="pill", color=ACCENT, opacity=1.0))

    log("post")
    cv.vignette(0.18, 0.75).grain(0.006, seed)
    return cv
