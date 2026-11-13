# -*- coding: utf-8 -*-
"""Landing page above the fold in a browser window: navigation, headline, buttons and a live product screenshot

* Format: 1920x1080
* Focal point: the headline on the left and the dashboard screenshot on the right
* Screen: navbar, hero (headline, sub-headline, two buttons, note, product shot), logo row
* Theme: light page, indigo accent, on a soft gradient backdrop
* Exact text: brand, headline, sub-headline, button labels
"""
# Read skills/pixelcanvas/references/ui.md (Screen, marketing sections, template, Presentation).
from pixelcanvas import Canvas
from pixelcanvas.ui import Presentation, Screen, template

PROMPT = ("A landing page hero for a release-notes SaaS called Northwave: navigation, the headline 'Ship product "
          "updates your customers actually read', two buttons and a dashboard screenshot. Light theme, indigo.")
NAME = "ui_landing_hero"
KIND = "design"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME, ACCENT, STYLE = "light", "#4f46e5", "modern"
BRAND = "Northwave"
HEADLINE = "Ship product updates your customers actually read"
SUBTITLE = f"{BRAND} turns release notes into in-app announcements, emails and a public changelog."
SHOT_WIDTH = 700                                         # product shot width in page points
LOGOS = ["Halcyon", "Quillstack", "Orbital", "Fernhill", "Larkspur"]


def page():
    s = Screen("browser", theme=THEME, accent=ACCENT, style=STYLE, size=(1440, 780), url="northwave.io",
               title=f"{BRAND} — release notes", name="landing")
    s.navbar(BRAND, links=["Product", "Customers", "Pricing", "Docs"], actions=[("Sign in", "ghost"), "Start free"])
    app = template("dashboard", "browser", theme=THEME, accent=ACCENT, style=STYLE, size=(1280, 780))
    shot = Canvas(*[int(round(v)) for v in app.size(margin=0)])    # the product shot: a picture of the app
    app.draw(shot, margin=0)
    with s.section(pad=(48, 32), name="hero"), s.hstack(gap=56, align="center"):
        s.hero(HEADLINE, SUBTITLE, note="Free 14-day trial · No credit card required")
        s.image(canvas=shot, height=SHOT_WIDTH * shot.h / shot.w, w=SHOT_WIDTH, fit="contain")
    with s.section(pad=(8, 32), name="logos"):
        s.logo_row(LOGOS, title="Trusted by product teams at")
    return s


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height)
    Presentation([page()], size=(width, height), background="soft", pad=0.05).draw(cv, log=log)
    return cv
