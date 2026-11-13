# -*- coding: utf-8 -*-
"""
UI component sheet (pixelcanvas.ui): the same labelled set of components in light and dark, side by side —
buttons, fields, selection controls, navigation, data display, feedback, media and code.

* Format: 1920x1320; two ``Screen("plain")`` boards (light | dark) drawn into the two halves of the canvas.
* Each cell is a card with an overline label; every cell is laid out by the kit (grid + stacks), no coordinates.
* Tunables: ACCENT, STYLE (modern / glass / material / wireframe), THEMES.
"""
from pixelcanvas import Canvas
from pixelcanvas.ui import Screen

NAME = "ui_components"
DEFAULT_SIZE = (1920, 1320)
DEFAULT_SEED = 1

ACCENT = "#5b5bf6"
STYLE = "modern"
THEMES = ("light", "dark")
BOARD = (960, 1320)


def cell(s, label, span=1):
    c = s.card(pad=16, gap=12, span=span)
    s._stack.append(c)
    s.overline(label, color=s.T.ink3)
    s._stack.pop()
    return c


def board(theme):
    s = Screen("plain", theme=theme, accent=ACCENT, style=STYLE, size=BOARD, pad=(28, 28), gap=18,
               name=f"{theme} board")
    with s.hstack(gap=10):
        s.heading("Components", level=3)
        s.badge(f"{theme} · {STYLE}", tone="neutral")
        s.spacer()
        s.caption("pixelcanvas.ui", color=s.T.ink3)
    with s.grid(cols=3, gap=14):
        with cell(s, "Buttons"):
            with s.hstack(gap=8):
                s.button("Primary", size="sm")
                s.button("Secondary", "secondary", size="sm")
            with s.hstack(gap=8):
                s.button("Outline", "outline", size="sm")
                s.button("Ghost", "ghost", size="sm")
                s.icon_button("plus", "filled", size="sm")
            s.button("Delete project", "destructive", icon="trash", size="sm", full=True)
        with cell(s, "Text fields"):
            s.input(value="ana@northwave.io", icon="mail", focused=True, field_h=40)
            s.search("Search", trailing="sliders")
            s.select(value="English (UK)", field_h=40)
        with cell(s, "Selection"):
            s.checkbox("Email me updates", checked=True)
            s.radio("Weekly digest", selected=True)
            s.toggle("Dark mode", on=True)
            s.slider(0.62, value_text=None)
        with cell(s, "Navigation"):
            s.segmented(["Day", "Week", "Month"], active=1)
            s.tabs(["Overview", "Activity", "Files"], active=0)
            s.breadcrumb(["Projects", "Website", "Pricing"])
            s.pagination(page=3, pages=12)
        with cell(s, "Badges, chips, avatars"):
            with s.hstack(gap=6, wrap=True):
                s.badge("Paid", "success")
                s.badge("Pending", "warning")
                s.badge("Failed", "danger")
                s.badge("Live", "accent", dot=True)
            s.chips(["Design", "Research", "Remote"], selected=[0])
            with s.hstack(gap=12):
                s.avatar("Maya Chen", 40, status="online")
                s.avatar("Noah Patel", 40, kind="initials")
                s.avatar_group(["Ana Silva", "Liam Carter", "Sofia Rossi", "Jon Bell", "Ada K"], size=30, max=3)
        with cell(s, "Progress"):
            with s.hstack(gap=16):
                s.ring(0.72, sub="done", size=84)
                with s.vstack(gap=12, w="fill"):
                    s.progress(0.45, label="Upload", tone="accent")
                    s.progress(0.8, label="Storage", tone="warning")
            s.skeleton(lines=2, avatar=True)
        with cell(s, "List rows"):
            with s.vstack(gap=0, dividers=True, divider_inset=48):
                s.list_row("Notifications", leading="bell", toggle=True)
                s.list_row("Maya Chen", "Sent you a file", leading="avatar:Maya Chen", meta="9:41", badge=2)
                s.list_row("Storage", leading="database", value="42 GB", chevron=True)
        with cell(s, "Stat tile"):
            s.stat("Monthly revenue", "$48.2k", delta="+12.4%", spark=[3, 4, 3.6, 5, 4.8, 6, 6.6], card=False,
                   icon="dollar")
        with cell(s, "Feedback"):
            s.banner("Payment failed", "Update your card to keep your plan.", tone="danger", close=True)
            s.tooltip("Copied to clipboard", self_align="center")
            s.toast("Changes saved", action="Undo", position="inline")
        with cell(s, "Chat"):
            s.message("Are we still on for 3pm?", time="9:12", w="fill")
            s.message("Yes — see you there!", me=True, time="9:14", w="fill")
        with cell(s, "Calendar"):
            s.calendar(2026, 11, selected=12, today=12, events={3: 1, 18: 2, 26: 1}, header=False)
        with cell(s, "Menu"):
            s.menu([("edit", "Rename", "⌘R"), ("copy", "Duplicate", "⌘D"), ("share", "Share"), "-",
                    {"icon": "trash", "label": "Delete", "danger": True}], active=0, w="fill")
        with cell(s, "Media"):
            s.image(height=92, scene="dawn", seed=2, radius=12)
            s.audio(0.4, "0:42", "2:15")
        with cell(s, "Code", span=2):
            s.code("def greet(name: str) -> str:\n    # say hello politely\n    return f\"Hello, {name}!\"\n\n"
                   "print(greet(\"pixelcanvas\"))", title="hello.py", highlight=[3])
    return s


def render(width=1920, height=1320, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    n = len(THEMES)
    for i, th in enumerate(THEMES):
        log(f"{th} board")
        board(th).draw(cv, box=(int(i * width / n), 0, int((i + 1) * width / n), height), log=log)
    return cv
