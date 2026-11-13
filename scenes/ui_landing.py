# -*- coding: utf-8 -*-
"""
SaaS landing page in a browser frame (pixelcanvas.ui): navigation, hero with a product screenshot that is itself
built from the kit (a dashboard Screen), logo row, feature grid, pricing, testimonial and footer.

* Format: tall page — the browser viewport grows to fit the content (``height="auto"``); the whole Screen is
  scaled to fit the canvas width. One layer per page section.
* The product shot is ``ui.template("dashboard", "browser")`` embedded with ``s.hero(media=...)``.
* Tunables: THEME, ACCENT, STYLE, BRAND, HEADLINE.
"""
from pixelcanvas import Canvas
from pixelcanvas.ui import Screen, template

NAME = "ui_landing"
DEFAULT_SIZE = (1440, 2880)
DEFAULT_SEED = 1

THEME = "light"
ACCENT = "#4f46e5"
STYLE = "modern"
BRAND = "Northwave"
HEADLINE = "Ship product updates your customers actually read"

FEATURES = [("edit", "One editor", "Write once and publish to in-app widgets, email and the web."),
            ("sparkles", "AI summaries", "Turn long changelogs into three crisp highlights."),
            ("target", "Targeting", "Show each update to the right segment at the right time."),
            ("chart", "Analytics", "See reads, clicks and reactions for every release."),
            ("globe", "Localised", "Translate posts into 30 languages with one click."),
            ("shield", "SSO & audit log", "Enterprise controls without the setup pain.")]
TIERS = [dict(name="Starter", price="$0", period="/mo", desc="For side projects and small teams.",
              features=["1 project", "In-app widget", "Public changelog"], cta="Start free"),
         dict(name="Growth", price="$49", period="/mo", desc="For products that ship every week.",
              features=["10 projects", "Email digests", "AI summaries", "Audience segments"], cta="Start 14-day trial"),
         dict(name="Scale", price="$199", period="/mo", desc="For companies with many product teams.",
              features=["Unlimited projects", "SSO & audit log", "Priority support"], cta="Contact sales")]


def build():
    s = Screen("browser", theme=THEME, accent=ACCENT, style=STYLE, size=(1280, None), height="auto",
               url="northwave.io", title=f"{BRAND} — release notes", name="landing")
    s.navbar(BRAND, links=["Product", "Customers", "Pricing", "Docs"], actions=[("Sign in", "ghost"), "Start free"])
    shot = template("dashboard", "browser", theme=THEME, accent=ACCENT, style=STYLE, size=(1280, 780))
    with s.section(pad=(72, 32), name="hero"):
        s.hero(HEADLINE, f"{BRAND} turns release notes into in-app announcements, emails and a public changelog "
               "— from one editor.", kicker="New: AI summaries", media=shot,
               note="Free 14-day trial · No credit card required")
    with s.section(pad=(16, 32), name="logos"):
        s.logo_row(["Halcyon", "Quillstack", "Orbital", "Fernhill", "Larkspur"], title="Trusted by product teams at")
    with s.section(pad=(72, 32), gap=36, name="features"):
        with s.vstack(gap=12, align="center"):
            s.overline("Features", align="center")
            s.heading("Everything you need to announce a release", level=1, align="center")
            s.text("From the first draft to the last read receipt.", align="center", color=s.T.ink2)
        s.feature_grid(FEATURES)
    with s.section(pad=(72, 32), gap=36, bg=s.T.surface, name="pricing"):
        with s.vstack(gap=12, align="center"):
            s.overline("Pricing", align="center")
            s.heading("Simple plans that grow with you", level=1, align="center")
            s.segmented(["Monthly", "Yearly · save 20%"], active=0, w=300)
        s.pricing(TIERS, highlight=1)
    with s.section(pad=(56, 32), name="testimonial", max_w=760):
        s.testimonial("We replaced three tools with Northwave. Our release posts now get read by half of our "
                      "active users — not just the team.", "Maya Chen", role="Head of Product, Quillstack")
    s.footer(BRAND, tagline="Release notes people read.")
    return s


def render(width=1440, height=2880, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    s = build()
    W, H = s.size(margin=0)
    a = width / height                       # backdrop margin that matches the canvas aspect ratio
    m = (a * H - W) / (2 * (1 - a)) if a < 1 else 48.0
    s.draw(cv, log=log, margin=max(m, 48.0))
    return cv
