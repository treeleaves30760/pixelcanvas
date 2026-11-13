# Diagrams (`pixelcanvas.diagram`)

Describe what is connected to what; the module does the layout, routing, spacing, colours and typography.
Flowcharts, architecture diagrams, org charts, mind maps, sequence diagrams, roadmaps, comparison tables and
labelled textbook figures — each in a few declarative calls, in any of the vector themes
(`light dark blueprint paper pastel neon mono`), as a raster and as SVG.

Options are checked: an unknown option or name (shape, icon, theme) raises an error that lists the closest valid ones, and both spellings are accepted (`color` / `colour`, `w` / `width`).

```python
from pixelcanvas import Canvas
from pixelcanvas.diagram import Diagram
d = Diagram(theme="light", title="Checkout")                  # direction="auto": LR or TB, whichever fills
d.node("web", "Web app", icon="monitor"); d.node("api", "API", icon="server"); d.node("db", "Orders", shape="database")
d.chain("web", "api", "db", label="HTTPS")
cv = Canvas(1920, 1080); d.draw(cv)                          # fills the canvas, title top-left
```

**The diagram fills the box it is drawn into.** `direction="auto"` (the default) tries left-to-right and
top-to-bottom, folds a long chain into rows (or columns), splits a very wide rank into two staggered rows and
spreads a hub's leaves over both sides, then keeps the variant with the largest text that fills the box. The
layout then grows nodes and text together (up to `grow=1.7` times `font_size`, like a slide) and widens the
gaps for the rest. A flowchart (decisions, loops) reads left to right on a wide canvas, wrapping into two or
three rows with the loop edges routed in the gutters, and top to bottom on a tall one. `d.fit` reports the result: direction, bands, `font_px` and the `content` box in canvas px.

**Sizes are pixels.** `font_size` (default 26) is the smallest node label height in px on a 1080-px-tall canvas
and scales with the canvas, so a `--scale 0.5` draft looks the same; `title_size` (40) likewise. It is a floor:
the layout tightens its gaps and group padding around it before any text shrinks, so raising `font_size` makes
the labels larger, and text grows past it up to `grow` times when the box has room. `grow=False` keeps
`font_size` exactly, `floor=False` lets a crowded figure shrink its text to fit. `d.fit["font_px"]` reports the
label height drawn. `keep_order=True` (or `d.tiers([...], keep_order=True)`) keeps the nodes of every tier in the
order they were added.

`d.draw(cv, box=(x0, y0, x1, y1), log=log)` fits any figure into a box (one layer per node / container / edge
set); `d.render(scale=1)` → Canvas at natural size; `d.save_svg(path)`; `d.build()` → the `vector.Drawing`;
`d.into(drawing, box)` embeds it in a bigger page. After `draw`, `d.nodes` (`{id: {id, label, group, shape,
box}}`) and `d.edges` (`[{source, target, label, style, arrow}]`) are plain data for checks and reports.

## Calls

| Call | What it makes |
|---|---|
| `Diagram(theme, direction="auto"/"LR"/"TB"/"RL"/"BT", title, subtitle, lanes=None/"rows"/"columns"/"auto", routing="orthogonal"/"curved"/"straight", node_style="card", font_size=26, title_size=40, grow=1.7, floor=True, keep_order=False, spacing=1.0, wrap=200)` | layered diagram that fills its box; `wrap` = label width before wrapping |
| `d.node(id, label, shape="rounded", icon=None, group=None, color=None, sublabel=None, style=None, rank=None, at=None, pos=None)` | node; sizes follow the text, long labels wrap |
| `d.edge(a, b, label=None, style="solid"/"dashed"/"dotted", arrow="end"/"start"/"both"/"none", color=None, head="arrow"/"open"/"dot"/"diamond", label_pos=None)` | connector; `a` or `b` may be a list: `d.edge("gw", ["users", "orders"])` fans out |
| `d.edge(a, b, rank="same")` · `d.same_rank(a, b, ...)` | peers side by side (replicas, a primary and its standby) |
| `d.bus(id, label, members=[...], style="solid"/"dashed", arrow="none"/"end"/"both", color=None)` | bus bar (pub/sub topic, event bus, network segment) with a tap to each member, in the gap after them; ranks stay as they are; `d.edge(node, bus_id)` adds a tap |
| `d.tiers([group ids], lanes="rows"/"columns"/"auto")` | groups as full-width bands (rows) or full-height columns in this order, members spread evenly |
| `d.chain(a, b, c, ..., **edge_options)` | edges along a path; list steps fan out / in: `d.chain(["web", "app"], "gw", ["users", "orders"])` |
| `with d.group(id, label, color=None, icon=None, style="solid"/"dashed"):` | container around the nodes made inside |
| `Tree(nested_dict, kind="org"/"tree"/"mindmap", theme, direction="TB"/"LR", compact=True, icon=None)` | org chart, hierarchy, mind map; `"Name | Role"` labels |
| `Sequence(theme, title, autonumber=False)` + `participant(id, label, icon, kind="box"/"actor" (person badge)/"database")`, `message(a, b, label, activate=False, reply=False, head="arrow"/"open")`, `reply(a, b, label)`, `note(id_or_pair, text, side="over"/"left"/"right")`, `with frame("loop"/"alt"/"opt", label):`, `otherwise(label)`, `divider(text)`, `activate(id)`, `deactivate(id)` | sequence diagram |
| `Timeline(theme, title, start=None, end=None)` + `phase(start, end, label, color)`, `milestone(when, label, sublabel, icon, color)`, `marker(when, "Today")` | roadmap; dates `2026`, `"2026-03"`, `"2026-03-15"`, `"Q3 2026"`, `datetime.date`, or plain numbers |
| `Table(columns, rows, theme, title, highlight=None, badge="Recommended", align=None, zebra=True)` | comparison matrix; `True/"yes"/"✓"` check, `False/"no"/"✗"` cross, `"partial"/"~"` half, `None/"-"` dash |
| `page(1920, 1080, theme, title, subtitle)` → `(drawing, content_box)` | blank themed page for infographics |
| `callouts(cv, [(point, text), ...])` · `callout(cv, point, text, side="right")` | labels with leader lines, stacked in columns beside the figure, never overlapping; `leader="arrow"` or `"dot"`, `leader_color=`, `leader_width=` style the leaders |
| `markers(cv, [(point, text), ...], legend="right", legend_title=None)` | numbered badges + legend card |
| `brace(cv, p0, p1, text, side="auto")` · `dimension(cv, p0, p1, text, offset=0)` · `scale_bar(cv, xy, length_px, "1 mm")` | braces, measurement arrows, scale bars |
| `Annotator(cv_or_drawing, theme, leader="dot"/"arrow"/"none", leader_color=None, leader_width=1.6)` → `.callout(point, text, side, leader=, leader_color=, leader_width=) .marker() .label() .brace() .dimension() .scale_bar()` then `.draw(log)` | everything at once, resolved together |

**Shapes**: `box rounded pill diamond ellipse cylinder parallelogram hexagon document cloud note circle`; aliases
`database db decision start end terminal io data doc oval sticky`. **Icons**: any `vector.icon_names()` name
(`server database cloud user users phone monitor shield key lock globe gear cart search bolt layers network ...`).
**Colours**: accent index `0..7`, a theme colour name `"blue" "orange" "green" "red" "purple" "cyan" "pink"
"olive"`, or `"#hex"`; nodes in a group take the group colour. **Node styles**: `card` (surface + accent edge),
`tint` (pale accent fill), `solid` (accent fill, white text), `outline`, `plain`.

**Layout rules you get for free**: ranks by longest path, crossing reduction, groups kept contiguous, a container of
sinks (a data tier) lined up in one column, long edges kept inside the container they enter, edges leave
the side that faces their target, parallel edges get their own ports and tracks, loops route around the outside,
decisions send their outer branches out of the side corners, labels sit on plates clear of nodes and container
borders, a node with many edges grows along that side so every edge has its own port, cycles keep a compact
layout, and every diagram fills the box you draw it into. `rank=k` pins a node to column / row `k`;
`at=(col, row)` pins a grid cell; giving every node `pos=(x, y)` switches to manual placement with an
obstacle-avoiding orthogonal router.

## Recipes

**Flowchart with decisions and a retry loop** — no direction needed; a wide canvas gets rows, a tall one a column:

```python
from pixelcanvas import Canvas
from pixelcanvas.diagram import Diagram
d = Diagram(title="Pull request")
d.node("open", "Open PR", shape="pill"); d.node("ci", "Run CI"); d.node("pass", "Tests pass?", shape="diamond")
d.node("rev", "Code review"); d.node("ok", "Approved?", shape="diamond"); d.node("fix", "Push fixes")
d.node("merge", "Merge"); d.node("done", "Done", shape="pill")
d.chain("open", "ci", "pass"); d.edge("pass", "rev", "yes"); d.edge("pass", "fix", "no")
d.edge("rev", "ok"); d.edge("ok", "merge", "yes"); d.edge("ok", "fix", "no"); d.edge("fix", "ci", style="dashed")
d.edge("merge", "done")
cv = Canvas(1920, 1080); d.draw(cv)
print(d.fit["direction"], d.fit["bands"])
```

**Architecture in tiers with a pub/sub bus** — `lanes="rows"` makes every group a full-width band:

```python
from pixelcanvas import Canvas
from pixelcanvas.diagram import Diagram
d = Diagram(theme="dark", lanes="rows", title="Real-time chat platform",
            subtitle="solid = request, dashed = pub/sub event")
with d.group("clients", "Client tier", icon="users"):
    d.node("web", "Web client", icon="monitor"); d.node("mobile", "Mobile client", icon="phone")
with d.group("edge", "Edge tier", icon="globe"):
    d.node("lb", "Load balancer", icon="network"); d.node("gw", "API gateway", icon="shield")
with d.group("svc", "Service tier", icon="layers"):
    d.node("auth", "Auth service", icon="key"); d.node("msg", "Messaging service", icon="layers")
    d.node("pres", "Presence service", icon="user")
with d.group("data", "Data tier", icon="database"):
    d.node("pg", "Postgres", shape="cylinder", icon="database")
    d.node("obj", "Object storage", shape="cylinder", icon="cloud")
d.bus("redis", "Redis pub/sub", members=["msg", "pres"], style="dashed")
d.edge(["web", "mobile"], "lb", label="HTTPS / WSS"); d.edge("lb", "gw")
d.edge("gw", ["auth", "msg", "pres"]); d.edge(["auth", "msg"], "pg"); d.edge("msg", "obj", label="attachments")
cv = Canvas(1920, 1080); d.draw(cv)
print(d.fit["font_px"], [e["target"] for e in d.edges if e["source"] == "gw"])
```

**Architecture diagram with containers** — icons, sync / async edges, layout chosen for the canvas:

```python
from pixelcanvas import Canvas
from pixelcanvas.diagram import Diagram
d = Diagram(theme="dark", title="Platform", subtitle="sync = solid, async = dashed")
with d.group("edge", "Edge", icon="globe"):
    d.node("cdn", "CDN", icon="cloud")
    d.node("gw", "API gateway", icon="shield", sublabel="auth · rate limit")
with d.group("svc", "Services", icon="layers"):
    for s, ic in (("users", "user"), ("orders", "cart"), ("search", "search")):
        d.node(s, s.title(), icon=ic)
with d.group("data", "Data", icon="database"):
    d.node("pg", "Postgres", shape="cylinder", icon="database")
    d.node("bus", "Event bus", icon="network")
d.edge("cdn", "gw", label="/api")
d.edge("gw", ["users", "orders", "search"])
d.edge(["users", "orders"], "pg"); d.edge("orders", "bus", style="dashed", label="events")
cv = Canvas(1920, 1080); d.draw(cv)
```

**Flowchart with decisions and a loop** (portrait):

```python
from pixelcanvas import Canvas
from pixelcanvas.diagram import Diagram
f = Diagram(theme="light", title="Sign-up")                   # auto: top to bottom on a portrait canvas
f.node("start", "Start", shape="pill")
f.node("form", "Fill in the form", icon="file")
f.node("ok", "Email valid?", shape="diamond")
f.node("fix", "Show an error", shape="note", color="red")
f.node("done", "Account created", shape="pill", color="green", icon="check")
f.chain("start", "form", "ok")
f.edge("ok", "done", label="yes"); f.edge("ok", "fix", label="no"); f.edge("fix", "form", style="dashed")
cv = Canvas(1080, 1350); f.draw(cv)
```

**Org chart, mind map, sequence, roadmap, table on one page**:

```python
from pixelcanvas import Canvas
from pixelcanvas.diagram import Tree, Sequence, Timeline, Table, page
org = Tree({"Ada | CEO": {"Grace | CTO": ["Web team", "Data team"], "Alan | CFO": ["Finance"]}}, kind="org")
mind = Tree({"Launch": {"Product": ["Pricing", "Docs"], "Marketing": ["Blog", "Video"]}}, kind="mindmap")
seq = Sequence(autonumber=True)
seq.message("app", "api", "POST /login", activate=True); seq.reply("api", "app", "token")
road = Timeline()
road.phase("2026-01", "2026-04", "Build"); road.milestone("2026-05-01", "Launch", icon="star")
tbl = Table(["", "Free", "Pro"], [["SSO", False, True], ["Seats", "3", "Unlimited"]], highlight="Pro")
dwg, (x0, y0, x1, y1) = page(1920, 1080, title="One-page overview")
xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
org.into(dwg, (x0, y0, xm - 20, ym - 20)); mind.into(dwg, (xm + 20, y0, x1, ym - 20))
seq.into(dwg, (x0, ym + 20, xm - 20, y1)); tbl.into(dwg, (xm + 20, ym + 20, x1, y1))
cv = dwg.render()                                            # or dwg.draw(cv, log=log); dwg.save_svg(path)
```

**Labelled textbook figure** — draw the figure with anything, then label it:

```python
from pixelcanvas import Canvas
from pixelcanvas.vector import Drawing
from pixelcanvas.diagram import Annotator
cv = Canvas(1920, 1080, "#f6f7f9")
fig = Drawing(1920, 1080, background="none")
fig.circle(960, 540, 300, fill="#f3c6c6", stroke="#b04a4a", stroke_width=4)
fig.circle(880, 470, 90, fill="#d77d7d"); fig.circle(1050, 620, 110, fill="#7d9fd7")
fig.draw(cv)
ann = Annotator(cv, theme="light")
ann.callout((880, 470), "Nucleus"); ann.callout((1050, 620), "Vacuole")
ann.callout((1200, 380), "Cell membrane"); ann.callout((700, 700), "Cytoplasm")
ann.dimension((660, 900), (1260, 900), "60 µm")
ann.scale_bar((80, 1020), 200, "20 µm")
ann.draw()
```

Points are canvas pixels, or 0~1 fractions of the canvas (`(0.46, 0.43)`). Callouts on the left half stack in a
column left of the figure, the rest on the right; labels are sorted by height so leader lines never cross.
`markers(cv, [((880, 470), "Nucleus"), ...], legend="right")` numbers the points instead.
For dark artwork pass `theme="dark"`. On a `vector.Drawing` the same calls add vector items (and appear in its SVG).
