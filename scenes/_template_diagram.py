# -*- coding: utf-8 -*-
"""Architecture diagram or flowchart with automatic layout (diagram)

* Format: 1920x1080 (any size: the layout direction is automatic and fills the canvas; LANES = "rows" for tiers)
* Focal point: <the API gateway; everything flows through it>
* Reading order: <left to right: clients -> edge -> services -> data>
* Groups: <one container per tier>
* Palette: <theme THEME; sync edges solid, async dashed>
* Exact text: <TITLE, SUBTITLE and every node label copied from the request>
"""
# Read skills/pixelcanvas/references/diagrams.md (nodes, edges, groups, Tree, Sequence, Timeline, Table).
from contextlib import nullcontext

from pixelcanvas import Canvas
from pixelcanvas.diagram import Diagram, Sequence, Timeline, Tree
from pixelcanvas.vector import THEMES

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "diagram"
NAME = "my_diagram"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "light"                     # light dark blueprint paper pastel neon mono
ROUTING = "orthogonal"              # orthogonal curved straight
LANES = None                        # None, "rows" (tiers as full-width bands), "columns", "auto"
# >>> data: new_scene.py --prompt writes the request's title, nodes and edges here
TITLE = "Checkout service"
SUBTITLE = "solid = request, dashed = async event"
TIERS = [  # (group id or None, group label, group icon, [(node id, label, icon, shape)]); icons: vector.icon_names()
    ("clients", "Clients", "users", [("web", "Web app", "monitor", "rounded"), ("mobile", "Mobile app", "phone", "rounded")]),
    (None, None, None, [("gw", "API gateway", "shield", "rounded")]),
    ("services", "Services", "layers", [("orders", "Orders", "cart", "rounded"), ("payments", "Payments", "key", "rounded")]),
    ("data", "Data", "database", [("db", "Postgres", "database", "cylinder"), ("bus", "Event bus", "network", "rounded")]),
]
EDGES = [  # (from, to, label, style solid / dashed / dotted); a list on either side fans out / in
    (["web", "mobile"], "gw", "HTTPS", "solid"), ("gw", ["orders", "payments"], None, "solid"),
    (["orders", "payments"], "db", None, "solid"), ("orders", "bus", "order.created", "dashed"),
]
# <<< data


# >>> build: Diagram (architecture, flowchart), Sequence, Timeline or Tree
def build():
    d = Diagram(theme=THEME, routing=ROUTING, title=TITLE, subtitle=SUBTITLE, lanes=LANES, font_size=22)
    for gid, glabel, gicon, nodes in TIERS:              # one container per tier (None: loose nodes)
        with d.group(gid, glabel, icon=gicon) if gid else nullcontext():
            for nid, label, icon, shape in nodes:
                d.node(nid, label, icon=icon, shape=shape)
    for a, b, label, style in EDGES:                     # peers: d.edge(a, b, rank="same"); d.bus(...) for pub/sub
        d.edge(a, b, label=label, style=style)
    return d
# <<< build


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height, THEMES[THEME].background)
    build().draw(cv, log=log)                            # fits the canvas; one layer per node, group and edge set
    return cv
