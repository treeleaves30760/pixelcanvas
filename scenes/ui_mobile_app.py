# -*- coding: utf-8 -*-
"""
Mobile app case-study shot: three phone screens of one travel app (home, detail, profile) side by side, dark
theme, on a soft glowing backdrop (pixelcanvas.ui).

* Format: 1920x1080 presentation units, scaled to the canvas (drafts look identical); one layer per screen.
* Screens are declared with auto layout only — no coordinates: app bars, search, chips, image cards, list rows,
  a map, a sticky booking bar, profile stats and a tab bar.
* Tunables: THEME, ACCENT, STYLE (modern / glass / material / wireframe), TILT (0 or "iso"), STAGGER.
"""
from pixelcanvas import Canvas
from pixelcanvas.ui import Screen, Presentation

NAME = "ui_mobile_app"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "dark"
ACCENT = "#ff7a59"
STYLE = "modern"
TILT = 0
STAGGER = 0.05
TITLE = None

TABS = [("home", "Explore"), ("heart", "Saved"), ("calendar", "Trips"), ("user", "Profile")]


def home():
    s = Screen("phone", theme=THEME, accent=ACCENT, style=STYLE, name="home")
    s.app_bar("", leading="avatar:Ana Silva", trailing=["bell"], center=False)
    with s.vstack(gap=4):
        s.small("Good morning, Ana")
        s.heading("Where to next?", level=1)
    s.search("Search destinations", trailing="sliders")
    s.chips(["All", "Beaches", "Mountains", "Cities", "Lakes"], selected=[1])
    with s.hstack(gap=12, align="start"):
        s.image(height=196, scene="ocean", seed=2, caption="Azure Bay", sub="Portugal", badge="★ 4.9",
                icon="heart", w="fill", grow=1.25)
        s.image(height=196, scene="dusk", seed=5, caption="Red Cliffs", sub="Morocco", icon="heart", w="fill")
    with s.hstack():
        s.title("Popular stays")
        s.spacer()
        s.link("See all", size=14)
    with s.list_group():
        s.list_row("Casa do Mar", "Lagos · 2 guests", leading="img:desert", value="$128", value_sub="/ night",
                   value_strong=True)
        s.list_row("Pine Lodge", "Sintra · 4 guests", leading="img:forest", value="$96", value_sub="/ night",
                   value_strong=True)
    s.tab_bar(TABS, active=0)
    return s


def detail():
    s = Screen("phone", theme=THEME, accent=ACCENT, style=STYLE, name="detail", pad=(0, 20, 24, 20))
    s.app_bar("Azure Bay", leading="back", trailing=["share-up", "heart"])
    s.image(height=250, scene="ocean", seed=2, radius=24)
    with s.hstack(gap=12, align="start"):
        with s.vstack(gap=4, w="fill"):
            s.heading("Azure Bay Villa", level=2)
            with s.hstack(gap=6):
                s.icon("pin", 16, color="accent")
                s.small("Lagos, Portugal")
        s.badge("★ 4.9 · 212", tone="warning")
    with s.hstack(gap=10):
        for ic, t in (("users", "4 guests"), ("home", "2 rooms"), ("wifi", "Wi-Fi")):
            s.chip(t, icon=ic)
    s.text("Whitewashed villa above a quiet cove, with a pool terrace and sunset views over the Atlantic.",
           role="small", max_lines=2)
    s.map(height=118, seed=4)
    s.spacer()
    with s.hstack(gap=16, pad=(14, 20), surface="card", radius=24, name="booking bar"):
        with s.vstack(gap=0, w="fill"):
            s.caption("Total · 5 nights", color=s.T.ink3)
            s.heading("$640", level=3)
        s.button("Book now", size="lg", icon_right="arrow-right")
    return s


def profile():
    s = Screen("phone", theme=THEME, accent=ACCENT, style=STYLE, name="profile")
    s.app_bar("Profile", trailing=["gear"])
    with s.vstack(gap=6, align="center"):
        s.avatar("Ana Silva", 88, ring=s.T.accent, status="online")
        s.spacer(4)
        s.heading("Ana Silva", level=3, align="center")
        s.small("Explorer · Lisbon", align="center")
    with s.card(pad=(14, 8), gap=0):
        with s.grid(cols=3, gap=0):
            for v, k in (("24", "Trips"), ("18", "Countries"), ("4.9", "Rating")):
                with s.vstack(gap=2, align="center"):
                    s.title(v, align="center")
                    s.caption(k, align="center", color=s.T.ink3)
    with s.card(pad=16, gap=10):
        with s.hstack():
            s.label("Explorer level")
            s.spacer()
            s.caption("1,240 / 2,000 pts", color=s.T.ink3)
        s.progress(0.62)
    with s.list_group("Upcoming"):
        s.list_row("Kyoto, Japan", "Mar 14 – 22", leading="plane", tone="info", chevron=True, solid=True)
        s.list_row("Reykjavik", "Jun 2 – 9", leading="compass", tone="success", chevron=True, solid=True)
        s.list_row("Payment methods", leading="credit-card", tone="warning", chevron=True, solid=True)
    s.tab_bar(TABS, active=3)
    return s


def build():
    return Presentation([home(), detail(), profile()], size=(1920, 1080), background="soft", tilt=TILT,
                        stagger=STAGGER, title=TITLE)


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    build().draw(cv, log=log)
    return cv
