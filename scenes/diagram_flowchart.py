# -*- coding: utf-8 -*-
"""
Decision flowchart with loops (pixelcanvas.diagram.Diagram, top to bottom).

* Format: portrait 1080x1440, light theme; one Diagram fitted into the canvas.
* Content: a release pipeline — start, process steps, three decisions with yes / no branches, two
  retry loops that route around the outside, a manual-input step and a terminal.
* Decisions send their "no" branch out of the side corner; loops come back on lanes outside the flow.
"""
from pixelcanvas import Canvas
from pixelcanvas.diagram import Diagram
from pixelcanvas.vector import THEMES

PROMPT = ("A flowchart of a release pipeline from a merged pull request to production, with decisions for the "
          "tests, the smoke tests and the release manager's approval, and retry loops.")
NAME = "diagram_flowchart"
KIND = "diagram"
DEFAULT_SIZE = (1080, 1440)
DEFAULT_SEED = 1

THEME = "light"
ROUTING = "orthogonal"
TITLE = "Release pipeline"
SUBTITLE = "From a merged pull request to production"


def build():
    d = Diagram(theme=THEME, routing=ROUTING, title=TITLE, subtitle=SUBTITLE)
    d.node("start", "Pull request merged", shape="pill", icon="code")
    d.node("build", "Build and unit tests", icon="gear")
    d.node("green", "All tests green?", shape="diamond")
    d.node("fix", "Fix the failing test", shape="note", color="orange")
    d.node("stage", "Deploy to staging", icon="server")
    d.node("smoke", "Smoke tests pass?", shape="diamond")
    d.node("rollback", "Roll back staging", shape="parallelogram", color="red")
    d.node("approve", "Release manager approves?", shape="diamond")
    d.node("notes", "Update release notes", shape="document")
    d.node("prod", "Roll out to production", icon="cloud", style="solid", color="blue")
    d.node("done", "Done", shape="pill", color="green", icon="check")

    d.chain("start", "build", "green")
    d.edge("green", "stage", label="yes")
    d.edge("green", "fix", label="no")
    d.edge("fix", "build", label="retry", style="dashed")
    d.edge("stage", "smoke")
    d.edge("smoke", "approve", label="yes")
    d.edge("smoke", "rollback", label="no")
    d.edge("rollback", "fix", style="dashed")
    d.edge("approve", "prod", label="yes")
    d.edge("approve", "notes", label="not yet")
    d.edge("notes", "approve", style="dashed")
    d.edge("prod", "done")
    return d


def render(width=1080, height=1440, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, THEMES[THEME].background)
    build().draw(cv, log=log)
    return cv
