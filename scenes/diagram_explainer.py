# -*- coding: utf-8 -*-
"""
Infographic page: a sequence diagram, a roadmap timeline and a comparison table on one page
(pixelcanvas.diagram: Sequence, Timeline, Table, page).

* Format: 1920x1080, light theme. One vector page (``diagram.page``) with a title block; three cards, each
  holding one figure placed with ``fig.into(page, box)`` so the whole page is one Drawing (one SVG).
* Left card: how a passkey sign-in works (sequence). Right top: the rollout plan (timeline).
  Right bottom: passwords vs SMS codes vs passkeys (table, passkeys highlighted).
"""
from pixelcanvas import Canvas
from pixelcanvas.diagram import Sequence, Timeline, Table, page
from pixelcanvas.vector import THEMES

NAME = "diagram_explainer"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "light"
TITLE = "Passkeys, explained"
SUBTITLE = "What happens when you sign in, how we roll it out, and why it beats passwords"
GAP = 32


def sign_in():
    s = Sequence(theme=THEME, autonumber=True, font_size=18)
    s.participant("you", "You", kind="actor")
    s.participant("phone", "Phone", icon="phone")
    s.participant("site", "Website", icon="globe")
    s.message("you", "site", "Open sign-in page")
    s.message("site", "phone", "Challenge", activate=True)
    s.message("phone", "you", "Face or fingerprint?")
    s.message("you", "phone", "Unlock")
    s.message("phone", "phone", "Sign with private key")
    s.reply("phone", "site", "Signed challenge")
    s.message("site", "site", "Verify with public key")
    s.reply("site", "you", "Signed in")
    s.note("site", "No secret ever leaves the phone", side="over")
    return s


def rollout():
    t = Timeline(theme=THEME, width=1300, font_size=19)
    t.phase("2026-01", "2026-03", "Pilot", color="purple")
    t.phase("2026-03", "2026-07", "Opt-in for everyone", color="blue")
    t.phase("2026-07", "2026-10", "Default for new accounts", color="green")
    t.milestone("2026-02-15", "Staff pilot", icon="users")
    t.milestone("2026-05-01", "Public launch", icon="star")
    t.milestone("2026-09-01", "Passwords optional", icon="key")
    return t


def compare():
    return Table(["", "Password", "SMS code", "Passkey"],
                 [["Phishing resistant", False, False, True],
                  ["Nothing to remember", False, True, True],
                  ["Works without signal", True, False, True],
                  ["Sign-in time", "~20 s", "~30 s", "~3 s"],
                  ["Safe after a breach", False, "partial", True]],
                 theme=THEME, highlight="Passkey", badge="Best", font_size=22)


def build():
    th = THEMES[THEME]
    dwg, (x0, y0, x1, y1) = page(1920, 1080, theme=THEME, title=TITLE, subtitle=SUBTITLE)
    mid = x0 + (x1 - x0) * 0.43
    right_split = y0 + (y1 - y0) * 0.47
    cards = {
        "sign-in": ((x0, y0, mid - GAP / 2, y1), "1", "How a sign-in works", sign_in()),
        "rollout": ((mid + GAP / 2, y0, x1, right_split - GAP / 2), "2", "Rollout plan", rollout()),
        "compare": ((mid + GAP / 2, right_split + GAP / 2, x1, y1), "3", "Why switch", compare()),
    }
    for name, ((a, b, c, d), num, head, fig) in cards.items():
        with dwg.group(f"card {name}", shadow=dict(dx=0, dy=4, blur=14, color=th.shadow["color"], opacity=0.10)):
            dwg.rect(a, b, c - a, d - b, r=18, fill=th.surface, stroke=th.grid, stroke_width=1)
        with dwg.group(f"heading {name}"):
            dwg.circle(a + 40, b + 40, 16, fill=th.accent(0))
            dwg.text(a + 40, b + 40, num, size=17, weight="bold", fill="#ffffff", anchor="mm")
            dwg.text(a + 68, b + 40, head, size=24, weight="bold", anchor="lm")
        fig.into(dwg, (a + 24, b + 72, c - 24, d - 20), name=name)
    return dwg


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, THEMES[THEME].background)
    build().draw(cv, log=log)
    return cv
