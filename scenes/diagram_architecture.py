# -*- coding: utf-8 -*-
"""
Architecture diagram of a web platform (pixelcanvas.diagram.Diagram).

* Format: 1920x1080, dark theme; the whole diagram is one Diagram fitted into the canvas.
* Content: clients -> edge (CDN, gateway) -> services -> event bus + workers -> data stores, each tier a
  container; icons on every node; async edges dashed, cache reads dotted.
* Layout is automatic (layered, direction chosen to fill the canvas); only the nodes and edges are declared.
* Layers: background, title, containers, edges, edge labels, one layer per node.
"""
from pixelcanvas import Canvas
from pixelcanvas.diagram import Diagram
from pixelcanvas.vector import THEMES

PROMPT = ("An architecture diagram of a commerce platform: web, mobile and partner API clients, a CDN and an API "
          "gateway, users, orders, payments and search services, an event bus with workers, and Redis, Postgres, "
          "object storage and Elastic. Dark theme, grouped by tier, with icons.")
NAME = "diagram_architecture"
KIND = "diagram"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "dark"
DIRECTION = "auto"          # auto LR TB RL BT
ROUTING = "orthogonal"      # orthogonal / curved / straight
TITLE = "Commerce platform"
SUBTITLE = "Request path from clients to data stores  ·  solid = sync, dashed = async events, dotted = cache"
SPACING = 1.0


def build():
    d = Diagram(theme=THEME, direction=DIRECTION, routing=ROUTING, title=TITLE, subtitle=SUBTITLE,
                spacing=SPACING)
    with d.group("clients", "Clients", icon="users"):
        d.node("web", "Web app", icon="monitor", sublabel="React SPA")
        d.node("mobile", "Mobile", icon="phone", sublabel="iOS · Android")
        d.node("partner", "Partner API", icon="code", sublabel="REST")
    with d.group("edge", "Edge", icon="globe"):
        d.node("cdn", "CDN", icon="cloud", sublabel="static assets")
        d.node("gateway", "API gateway", icon="shield", sublabel="auth · rate limit")
    with d.group("services", "Services", icon="layers"):
        d.node("users", "Users", icon="user")
        d.node("orders", "Orders", icon="cart")
        d.node("payments", "Payments", icon="key")
        d.node("search", "Search", icon="search")
    with d.group("async", "Async", icon="bolt"):
        d.node("bus", "Event bus", icon="network", sublabel="Kafka", rank=4)
        d.node("workers", "Workers", icon="gear", sublabel="emails · invoices", rank=4)
    with d.group("data", "Data", icon="database"):
        d.node("redis", "Redis", shape="cylinder", icon="bolt", sublabel="sessions · cache", rank=5)
        d.node("pg", "Postgres", shape="cylinder", icon="database", sublabel="primary + replica", rank=5)
        d.node("es", "Elastic", shape="cylinder", icon="search", sublabel="product index", rank=5)
        d.node("s3", "Object store", shape="cylinder", icon="folder", sublabel="images · invoices", rank=5)

    d.edge(["web", "mobile"], "cdn")
    d.edge("partner", "gateway", label="OAuth")
    d.edge("cdn", "gateway", label="/api")
    d.edge("gateway", ["users", "orders", "payments", "search"])
    d.edge("users", "redis", style="dotted")
    d.edge(["users", "orders"], "pg")
    d.edge("orders", "bus", label="order.created", style="dashed")
    d.edge("payments", "bus", style="dashed")
    d.edge("search", "es")
    d.edge("bus", "workers", style="dashed")
    d.edge("workers", "s3")
    return d


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height, THEMES[THEME].background)
    build().draw(cv, log=log)
    return cv
