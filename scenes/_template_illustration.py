# -*- coding: utf-8 -*-
"""Any picture drawn freehand: characters, animals, a named style, poster art (illustrate)

* Mood and light: <cosy evening, one warm lamp glow from the upper left>
* Palette, 4-6 colours + one accent: <cream wall, walnut floor, orange fox, teal accent, plum ink>
* Subject and frame: <a fox sitting at the centre, filling 55% of the height>
* Style, built not filtered: <storybook: soft shading, darker-tone outlines, paper grain>
* Exact text: <none>
"""
# Read skills/pixelcanvas/references/illustration.md (art direction, styles, shapes, Pillow layer, recipes).
from pixelcanvas import Canvas, manifest
from pixelcanvas.illustrate import Sketch, paper, grade

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
NAME = "my_illustration"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

MOOD = "cosy"                                  # grade(): warm night, golden, cosy, cool day, moody, print
LIGHT = (-0.6, -0.8)                           # direction towards the light: upper left
PALETTE = {"wall": "#f3d9a8", "floor": "#b9824e", "body": "#e07a3a", "accent": "#2f8f8a", "ink": "#3a2a30"}
SUBJECT_AT, SUBJECT_SIZE = (0.45, 0.6), 0.3    # where the subject stands; 1 local unit = this share of the height
STYLE = "storybook"                            # the look you build by hand, recorded for --check
TITLE = ""                                     # exact text from the request, drawn as lettering ("" = none)


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    sk = Sketch(cv, light=LIGHT)
    P = PALETTE

    log("background")                          # far to near; the light source glows
    cv.gradient([(0, P["wall"]), (1, "#f8ead0")])
    sk.paint(sk.box(0, 0.78, 1, 1), [(0, P["floor"]), (1, "#8a5a32")])
    cv.glow(0.15 * width, 0.12 * height, 0.35 * height, "#fff0c0", 0.25)

    # >>> subjects: the request's subjects; draw each yourself, one log() step each, like the two below
    # draw: <subject>                          (an optional ingredient call from kit.py find goes here)
    # <<< subjects

    log("subject")                             # design in a local frame, place it with sk.at, reuse the parts
    with sk.at(*SUBJECT_AT, SUBJECT_SIZE):
        sk.spot(0, 0.95, 0.8, 0.12, P["ink"], 0.35)                              # contact shadow
        sk.paint(sk.ellipse(0, 0.4, 0.62, 0.58), P["body"], shade=0.4, highlight=0.3, line=True, name="subject")
        head = sk.paint(sk.circle(0, -0.45, 0.45), P["body"], shade=0.4, highlight=0.3, line=True)
        for s in (-1, 1):
            sk.paint(sk.circle(s * 0.17, -0.5, 0.06), P["ink"])
            sk.spot(s * 0.27, -0.33, 0.11, 0.07, "#ff8a8a", 0.6, clip=head)

    log("prop")                                # the same canvas through Pillow ImageDraw, finished in one call
    with cv.draw(ss=2, name="prop", shade=0.35, highlight=0.3, line=True, shadow=True) as d:
        d.rounded_rectangle([*d.p(0.66, 0.52), *d.p(0.76, 0.8)], radius=d.k(0.02), fill=P["accent"])

    if TITLE:
        log("title")                           # lettering is part of the picture: size, colour, place
        sk.text(TITLE, 0.5, 0.12, 0.09, P["ink"], anchor="mm", font="serif")

    log("style")                               # build the style from its parts (illustration.md), never filter
    manifest.note(cv, "style", STYLE)

    log("post")
    paper(cv, seed=seed)
    grade(cv, MOOD)
    return cv
