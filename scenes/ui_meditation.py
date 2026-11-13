# -*- coding: utf-8 -*-
"""Meditation app home screen in a phone frame: greeting, breathing timer ring, three session cards, tab bar

* Format: 1080x1920 portrait, one phone on a soft backdrop
* Focal point: the big breathing-timer ring reading 4:32
* Screen: app bar, greeting, ring timer, "Sessions" with Sleep / Focus / Calm cards, bottom tab bar
* Theme: soft dark with purple accents
* Exact text: Good evening, Sam; 4:32; Sleep; Focus; Calm
"""
# Read skills/pixelcanvas/references/ui.md (Screen, components, Presentation).
from pixelcanvas import Canvas
from pixelcanvas.ui import Screen, Presentation

PROMPT = ("A mobile app screen for a meditation app: greeting 'Good evening, Sam', a big circular breathing timer "
          "at 4:32, three session cards (Sleep, Focus, Calm) and a bottom tab bar. Soft dark theme with purple "
          "accents.")
NAME = "ui_meditation"
KIND = "design"
DEFAULT_SIZE = (1080, 1920)
DEFAULT_SEED = 1

THEME = "dark"
ACCENT = "#9b7cff"
STYLE = "modern"                                         # modern, glass, material, wireframe
SESSIONS = [("Sleep", "moon", "20 min"), ("Focus", "bolt", "10 min"), ("Calm", "heart", "15 min")]


def home():
    s = Screen("phone", theme=THEME, accent=ACCENT, style=STYLE, name="home")
    s.app_bar("", leading="avatar:Sam Lee", trailing=["bell"], center=False)
    with s.vstack(gap=4):
        s.small("Tuesday evening")
        s.heading("Good evening, Sam", level=1)
    with s.vstack(align="center", gap=10, pad=(16, 0)):
        s.ring(0.62, label="4:32", sub="Breathe in", size=300)
        s.caption("Breathing timer")
    s.heading("Sessions", level=3)
    with s.grid(cols=3, gap=10):
        for name, icon, sub in SESSIONS:
            with s.card(pad=14, gap=10):
                s.icon_badge(icon, tone="accent")
                s.title(name)
                s.small(sub)
    s.tab_bar([("home", "Today"), ("music", "Sounds"), ("calendar", "Journal"), ("user", "Profile")], active=0)
    return s


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height)
    Presentation([home()], size=(width, height), background="soft").draw(cv, log=log)   # backdrop, phone, screen
    return cv
