# -*- coding: utf-8 -*-
"""Sequence diagram of a passkey sign-in: user, browser, API, auth service and database, with a retry loop

* Format: 1200x1200 square
* Focal point: the challenge / signature exchange in the middle of the API lifeline
* Reading order: left to right participants, top to bottom numbered messages
* Palette: light theme; replies dashed; one alt frame for a failed check
* Exact text: participant names and every message label
"""
# Read skills/pixelcanvas/references/diagrams.md (Sequence).
from pixelcanvas import Canvas
from pixelcanvas.diagram import Sequence

PROMPT = ("A sequence diagram of a passkey sign-in between the user, the browser, the API, the auth service and the "
          "user database, numbered, with an alt frame for a failed signature.")
NAME = "diagram_login_sequence"
KIND = "diagram"
DEFAULT_SIZE = (1200, 1200)
DEFAULT_SEED = 1

THEME = "light"
TITLE = "Passkey sign-in"


def build():
    s = Sequence(theme=THEME, title=TITLE, autonumber=True)
    s.participant("user", "User", icon="user")
    s.participant("web", "Browser", icon="monitor")
    s.participant("api", "API", icon="server")
    s.participant("auth", "Auth service", icon="shield")
    s.participant("db", "User DB", kind="database")
    s.message("user", "web", "Click “Sign in”")
    s.message("web", "api", "POST /login/begin", activate=True)
    s.message("api", "auth", "new challenge", activate=True)
    s.reply("auth", "api", "challenge")
    s.reply("api", "web", "challenge + options")
    s.message("web", "user", "Touch ID prompt")
    s.reply("user", "web", "fingerprint")
    s.message("web", "api", "POST /login/finish (signature)")
    s.message("api", "auth", "verify signature")
    s.message("auth", "db", "load public key")
    s.reply("db", "auth", "key")
    with s.frame("alt", "signature valid"):
        s.reply("auth", "api", "session token")
        s.reply("api", "web", "Set-Cookie: session")
        s.otherwise()
        s.reply("auth", "api", "401 invalid signature")
        s.reply("api", "web", "Try again")
    return s


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height)
    build().draw(cv, log=log)                            # fills the canvas; one layer per participant and message
    return cv
