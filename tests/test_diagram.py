"""Tests for pixelcanvas.diagram (layout invariants, every figure kind, annotations, doc snippets)."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas.vector import Drawing  # noqa: E402
from pixelcanvas import diagram as D  # noqa: E402
from pixelcanvas.diagram import (Diagram, Tree, Sequence, Timeline, Table, Annotator, callout, callouts,  # noqa: E402
                                 markers, brace, dimension, scale_bar, page, SHAPES)


def _hit(a, b, pad=0.0):
    return a[0] + pad < b[2] and b[0] + pad < a[2] and a[1] + pad < b[3] and b[1] + pad < a[3]


def _segments_clear(d):
    """No edge segment passes through a node that is not one of its endpoints."""
    res = d._res
    for e, pts in res["paths"]:
        for k, b in res["boxes"].items():
            if k in (e.a, e.b):
                continue
            inner = (b[0] + 2, b[1] + 2, b[2] - 2, b[3] - 2)
            for p, q in zip(pts, pts[1:]):
                assert not D._seg_hits_box(p, q, inner), (e.a, e.b, k)


def _platform(direction="LR", **kw):
    d = Diagram(direction=direction, title="Platform", subtitle="test", **kw)
    with d.group("edge", "Edge", icon="globe"):
        d.node("cdn", "CDN", icon="cloud")
        d.node("gw", "API gateway with a long label that wraps", icon="shield", sublabel="auth")
    with d.group("svc", "Services"):
        for s in ("users", "orders", "search"):
            d.node(s, s.title(), icon="server")
    d.node("pg", "Postgres", shape="database")
    d.node("web", "Web", icon="monitor")
    d.chain("web", "cdn", "gw")
    for s in ("users", "orders", "search"):
        d.edge("gw", s)
    d.edge("users", "pg", label="SQL")
    d.edge("orders", "pg", style="dashed")
    d.edge("web", "pg", style="dotted")              # long edge through dummies
    return d


@pytest.mark.parametrize("direction", ["LR", "TB", "RL", "BT"])
def test_diagram_layout_invariants(direction):
    d = _platform(direction)
    dwg = d.build()
    assert dwg.width > 100 and dwg.height > 100
    boxes = list(d._res["boxes"].values())
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _hit(boxes[i], boxes[j]), "nodes overlap"
    _segments_clear(d)
    # containers hold their members and no other node
    for g, gb in d._res["groups"].items():
        for k, b in d._res["boxes"].items():
            inside = b[0] >= gb[0] - 0.5 and b[2] <= gb[2] + 0.5 and b[1] >= gb[1] - 0.5 and b[3] <= gb[3] + 0.5
            assert inside == (d.nodes[k].group == g), (g, k)
    # edge labels never cover nodes
    for lb in d.label_boxes:
        for b in d.boxes.values():
            assert not _hit(lb, b)


def test_edges_attach_to_node_outline():
    d = _platform("LR")
    d.build()
    for e, pts in d._res["paths"]:
        a, b = d._res["boxes"][e.a], d._res["boxes"][e.b]
        x, y = pts[0]
        assert a[0] - 1 <= x <= a[2] + 1 and a[1] - 1 <= y <= a[3] + 1
        x, y = pts[-1]
        assert b[0] - 1 <= x <= b[2] + 1 and b[1] - 1 <= y <= b[3] + 1


def test_cycles_loops_and_decisions():
    f = Diagram(direction="TB")
    f.node("s", "Start", shape="pill")
    f.node("a", "Step")
    f.node("q", "OK?", shape="diamond")
    f.node("fix", "Fix", shape="note")
    f.node("e", "End", shape="pill")
    f.chain("s", "a", "q")
    f.edge("q", "e", label="yes")
    f.edge("q", "fix", label="no")
    f.edge("fix", "a", style="dashed")
    f.edge("e", "e")                                 # self loop
    f.build()
    r = f._rank
    assert r["fix"] == r["e"] == r["q"] + 1          # loop edge does not push ranks down
    kinds = {(e.a, e.b): e.kind for e in f.edges}
    assert kinds[("fix", "a")] == "back" and kinds[("e", "e")] == "self"
    _segments_clear(f)
    assert any(getattr(e, "lshape", False) for e in f._edges)   # a decision branch leaves from a side corner


@pytest.mark.parametrize("routing", ["orthogonal", "curved", "straight"])
def test_routing_modes_render(routing):
    d = _platform("LR", routing=routing)
    cv = d.render(scale=0.4)
    assert cv.w > 100 and np.isfinite(cv.buf).all()


def test_all_shapes_and_styles():
    d = Diagram(direction="LR")
    for i, s in enumerate(SHAPES):
        d.node(s, s.title(), shape=s, icon="star" if i % 2 else None, style=["card", "tint", "solid", "outline"][i % 4])
    for a, b in zip(SHAPES, SHAPES[1:]):
        d.edge(a, b)
    for alias in ("database", "decision", "start", "io", "doc"):
        d.node(alias, alias, shape=alias)
    cv = d.render(scale=0.3)
    assert cv.buf.std() > 0.01
    with pytest.raises(ValueError):
        d.node("x", shape="banana")
    with pytest.raises(KeyError):
        d.node("y", icon="banana")


def test_manual_grid_and_positions():
    d = Diagram()
    d.node("a", at=(0, 0)); d.node("b", at=(1, 1)); d.node("c", at=(1, 0))
    d.edge("a", "b"); d.edge("a", "c")
    d.build()
    assert d._rank["b"] == d._rank["c"] == 1
    m = Diagram()
    m.node("a", pos=(0, 0)); m.node("b", pos=(400, 0)); m.node("c", pos=(200, 0)); m.node("d", pos=(400, 300))
    m.edge("a", "b"); m.edge("b", "d")
    m.build()
    for e, pts in m._res["paths"]:
        inner = D._inflate(m._res["boxes"]["c"], -2)
        if e.a == "a":
            assert all(not D._seg_hits_box(p, q, inner) for p, q in zip(pts, pts[1:]))


def test_draw_into_box_layers_and_svg(tmp_path):
    d = _platform()
    steps = []
    cv = Canvas(480, 270, (0, 0, 0))
    d.draw(cv, log=steps.append)
    assert steps[0] == "background" and "containers" in steps and "edges" in steps
    assert any(s.startswith("node ") for s in steps)
    cv2 = Canvas(400, 400, (1, 1, 1))
    d.draw(cv2, box=(20, 200, 380, 380))
    assert np.allclose(cv2.buf[:150], 1.0)          # nothing painted above the box
    p = d.save_svg(str(tmp_path / "d.svg"))
    svg = open(p, encoding="utf-8").read()
    assert svg.startswith("<?xml") and "API gateway" in svg and "<path" in svg


def test_tree_kinds():
    data = {"Ada Lovelace | CEO": {"Grace | CTO": ["Web", "Data", "Infra"], "Alan | CFO": None, "Kay": ["Design"]}}
    for kind in ("org", "tree", "mindmap"):
        for th in ("light", "dark"):
            t = Tree(data, kind=kind, theme=th)
            cv = t.render(scale=0.35)
            assert cv.buf.std() > 0.01
            bs = list(t.boxes.values())
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    assert not _hit(bs[i], bs[j], 1.0), kind
    t = Tree({"A": ["B", "C"], "D": None}, kind="tree", direction="LR")   # several roots
    t.render(scale=0.3)
    assert D._initials("Grace Hopper") == "GH"


def test_sequence():
    s = Sequence(autonumber=True, title="Login")
    s.participant("u", "User", kind="actor")
    s.participant("app", "App", icon="phone")
    s.participant("db", "DB", kind="database")
    s.message("u", "app", "tap", activate=True)
    s.message("app", "app", "hash")
    with s.frame("alt", "cached"):
        s.reply("app", "u", "ok")
        s.otherwise("miss")
        s.message("app", "db", "query", head="open")
    s.note(("app", "db"), "spans two lifelines")
    s.note("u", "left note", side="left")
    s.divider("later")
    s.activate("db"); s.deactivate("db")
    for th in ("light", "dark"):
        s.theme = D._theme(th)
        cv = s.render(scale=0.4)
        assert cv.buf.std() > 0.01
    xs = [p.x for p in s.parts.values()]
    assert xs == sorted(xs) and len(s.bars) >= 2


def test_timeline_dates_and_numbers():
    assert D._when("2026-03")[0] == 2026 * 12 + 2
    assert D._when("Q3 2026")[0] == 2026 * 12 + 6
    assert D._when(2026)[0] == 2026 * 12
    import datetime
    assert D._when(datetime.date(2026, 1, 1))[0] == 2026 * 12
    with pytest.raises(ValueError):
        D._when("soon")
    t = Timeline(title="Plan")
    t.phase("2026-01", "2026-04", "Build"); t.phase("2026-02", "2026-06", "Beta")
    t.milestone("2026-05-01", "Launch", "GA", icon="star"); t.milestone("2026-05-05", "Party")
    t.marker("2026-03-01")
    cv = t.render(scale=0.4)
    assert cv.buf.std() > 0.01
    assert {p.lane for p in t.phases} == {0, 1}
    assert len({m.level for m in t.milestones}) == 2          # close milestones stack
    n = Timeline(theme="dark")
    n.phase(0, 40, "Stage A"); n.milestone(55, "Check")
    n.render(scale=0.3)
    assert n.is_date is False


def test_table_cells():
    assert [D._cell_kind(v) for v in (True, False, None, 3, "yes", "✗", "~", "-", "$12", "Unlimited")] == \
        ["yes", "no", "none", "num", "yes", "no", "part", "none", "num", "text"]
    t = Table(["Feature", "Free", "Pro"], [["SSO", False, True], ["Price", "$0", "$9"], ["Note"]],
              highlight="Pro", title="Plans")
    for th in ("light", "dark"):
        t.theme = D._theme(th)
        cv = t.render(scale=0.4)
        assert cv.buf.std() > 0.01
    assert len(t.rows[2]) == 3


def test_page_and_into():
    dwg, box = page(1920, 1080, title="Overview", subtitle="sub")
    seq = Sequence()
    seq.message("a", "b", "hi")
    g = seq.into(dwg, box)
    b = g.bbox()
    assert b[0] >= box[0] - 1 and b[2] <= box[2] + 1 and b[1] >= box[1] - 1 and b[3] <= box[3] + 1
    cv = dwg.render(scale=0.25)
    assert cv.buf.std() > 0.01


def test_callouts_stack_without_overlap():
    cv = Canvas(1280, 720, (0.96, 0.97, 0.98))
    fig = Drawing(1280, 720, background="none")
    fig.circle(640, 360, 220, fill="#e6a3a3")
    fig.draw(cv)
    pts = [((600 + (i % 3) * 30, 300 + i * 6), f"Label number {i}") for i in range(8)]
    boxes = callouts(cv, pts)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _hit(boxes[i], boxes[j])
    for b in boxes:                                   # columns sit outside the figure
        assert b[2] <= 420 + 2 or b[0] >= 860 - 2
    b1 = callout(cv, (0.5, 0.5), "one-shot")
    b2 = callout(cv, (0.5, 0.5), "slides away")
    assert not _hit(b1, b2)


def test_markers_brace_dimension_scalebar():
    cv = Canvas(960, 540, (1, 1, 1))
    steps = []
    boxes = markers(cv, [((300, 200), "First"), ((400, 300), "Second")], legend="right", legend_title="Key",
                    log=steps.append)
    assert boxes and steps == ["markers"]
    brace(cv, (100, 100), (100, 400), "span", side="left")
    dimension(cv, (200, 480), (700, 480), offset=-20, scale=0.1, unit="cm")
    scale_bar(cv, (40, 520), 120, "1 mm")
    assert cv.buf.min() < 0.5
    # on a vector drawing the same calls add items (and SVG)
    dwg = Drawing(800, 600, theme="dark")
    dwg.rect(300, 200, 200, 200, fill="#335")
    a = Annotator(dwg, theme="dark")
    a.callout((350, 250), "corner"); a.marker((450, 350), "centre"); a.label((400, 120), "title plate")
    a.brace((300, 420), (500, 420), "width"); a.dimension((300, 450), (500, 450), "20 mm"); a.scale_bar((20, 580), 100, "1 m")
    a.draw()
    assert "corner" in dwg.to_svg()


def _doc_blocks():
    path = os.path.join(ROOT, "skills", "pixelcanvas", "references", "diagrams.md")
    text = open(path, encoding="utf-8").read()
    return re.findall(r"```python\n(.*?)```", text, re.S)


@pytest.mark.parametrize("i", range(len(_doc_blocks())))
def test_doc_snippets_run(i, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    exec(compile(_doc_blocks()[i], f"diagrams.md block {i}", "exec"), {"__name__": "doc"})


@pytest.mark.parametrize("name", ["diagram_architecture", "diagram_flowchart", "diagram_explainer"])
def test_scenes_render_small(name):
    import importlib
    mod = importlib.import_module(f"scenes.{name}")
    w, h = mod.DEFAULT_SIZE
    cv = mod.render(w // 4, h // 4, 1, None)
    assert (cv.w, cv.h) == (w // 4, h // 4) and cv.buf.std() > 0.01


# ---------------------------------------------------------------- review fixes

def _food(direction="LR"):
    d = Diagram(theme="neon", direction=direction, title="Food delivery")
    with d.group("clients", "Clients"):
        d.node("mobile", "Mobile app", icon="phone")
        d.node("web", "Web app", icon="monitor")
    d.node("gw", "API gateway", icon="shield")
    with d.group("svc", "Services"):
        for s in ("order", "payment", "courier", "notify"):
            d.node(s, s.title(), icon="server")
    d.node("kafka", "Kafka", shape="hexagon")
    with d.group("data", "Data"):
        for s in ("pg", "redis", "s3"):
            d.node(s, s, shape="cylinder")
    d.chain(["mobile", "web"], "gw", ["order", "payment", "courier", "notify"])
    d.chain("order", "kafka", "courier", style="dashed", label="events")
    d.edge(["order", "payment"], "pg")
    d.edge("courier", "redis")
    d.edge("notify", "s3")
    return d


def test_list_endpoints_fan_out():
    d = _food()
    pairs = {(e.a, e.b) for e in d.edges}
    assert {("mobile", "gw"), ("web", "gw"), ("gw", "notify"), ("order", "pg"), ("payment", "pg")} <= pairs
    assert len(d.edges) == 2 + 4 + 2 + 2 + 2


@pytest.mark.parametrize("direction", ["LR", "TB"])
def test_sink_tier_lines_up_and_long_edges_stay_in_container(direction):
    d = _food(direction)
    d.build()
    ax = 0 if direction == "LR" else 1
    starts = {round(d._res["boxes"][k][ax + 2], 1) for k in ("pg", "redis", "s3")}
    assert len(starts) == 1, "data tier is one column"
    _segments_clear(d)
    # gw -> courier enters the services container once and stays inside it
    gb = d._res["groups"]["svc"]
    pts = dict((id(e), p) for e, p in d._res["paths"])
    e = next(e for e in d._edges if (e.a, e.b) == ("gw", "courier"))
    path = pts[id(e)]

    def inside(p):
        return gb[0] - 0.5 <= p[0] <= gb[2] + 0.5 and gb[1] - 0.5 <= p[1] <= gb[3] + 0.5
    flags = [inside(p) for p in path]
    first = flags.index(True)
    assert all(flags[first:]), "edge leaves the container it entered"


def test_edge_labels_avoid_container_borders():
    d = _food()
    d.build()
    for lb in d.label_boxes:
        for gb in d.group_boxes.values():
            for line in ((gb[0], gb[1], gb[2], gb[1]), (gb[0], gb[3], gb[2], gb[3])):
                assert not (lb[0] < line[2] and line[0] < lb[2] and lb[1] < line[1] < lb[3]), lb


def test_decision_labels_and_self_loop_on_outline():
    f = Diagram(direction="TB", routing="curved")
    f.node("a", "Has the user entered a correct one-time code within five minutes?", shape="decision")
    f.node("b", "Signed in", shape="end")
    f.node("c", "Send a new code")
    f.edge("a", "b", "yes"); f.edge("a", "c", "no"); f.edge("a", "a", "wait")
    f.build()
    n = f._nodes["a"]
    x0, y0, x1, y1 = f.boxes["a"]
    poly = n.art.outline((x0 + x1) / 2, (y0 + y1) / 2)
    assert len(f.label_boxes) == 3
    for lb in f.label_boxes:
        assert not D._rect_hits_poly(lb, poly), "label hidden under the diamond"
    loop = next(p for e, p in f._res["paths"] if e.a == e.b)
    ox, oy = f.boxes["a"][0] - f._res["boxes"]["a"][0], f.boxes["a"][1] - f._res["boxes"]["a"][1]
    end = (loop[-1][0] + ox, loop[-1][1] + oy)
    near = D._inflate((end[0], end[1], end[0], end[1]), 3)
    assert D._rect_hits_poly(near, poly), "self loop ends on the outline"


def test_accent_strip_only_when_no_icons():
    d = Diagram()
    d.node("a", "Plain"); d.node("b", "With icon", icon="server"); d.edge("a", "b")
    svg = d.to_svg()
    d2 = Diagram()
    d2.node("a", "Plain"); d2.node("b", "Also plain"); d2.edge("a", "b")
    assert svg.count("clip-path=") < d2.to_svg().count("clip-path=")


def test_strict_options_diagram():
    import pytest
    from pixelcanvas import diagram as D
    d = D.Diagram()
    with pytest.raises(TypeError, match="did you mean 'label'"):
        d.node("a", lable="x")
    d.node("b", colour="green")                      # alias of color
    with pytest.raises(ValueError, match="'box'"):
        d.node("c", shape="boxx")
    with pytest.raises(ValueError, match="'check'"):
        d.node("c", icon="chek")
    with pytest.raises(TypeError, match="did you mean 'color'"):
        D.callout(None, (0.5, 0.5), "x", colr="r")


# ---------------------------------------------------------------- the diagram fills the frame

def _g_chain(d):
    d.chain(*[d.node(n.lower(), n, icon="gear") for n in ("Collect", "Clean", "Validate", "Enrich", "Store", "Report")])


def _g_tree(d):
    d.node("root", "Platform", icon="layers")
    for i in range(3):
        d.node(f"a{i}", f"Area {i + 1}", icon="folder")
        d.edge("root", f"a{i}")
        for j in range(3):
            d.node(f"a{i}{j}", f"Team {i + 1}.{j + 1}", icon="users")
            d.edge(f"a{i}", f"a{i}{j}")


def _g_tiers(d):
    with d.group("cl", "Clients", icon="users"):
        d.node("web", "Web app", icon="monitor"); d.node("mob", "Mobile app", icon="phone")
    with d.group("edge", "Edge", icon="globe"):
        d.node("lb", "Load balancer", icon="network"); d.node("gw", "API gateway", icon="shield")
    with d.group("svc", "Services", icon="layers"):
        for s in ("Users", "Orders", "Payments", "Search"):
            d.node(s.lower(), s, icon="server")
    with d.group("data", "Data", icon="database"):
        d.node("pg", "Postgres", shape="cylinder"); d.node("es", "Elastic", shape="cylinder")
    d.edge(["web", "mob"], "lb"); d.edge("lb", "gw"); d.edge("gw", ["users", "orders", "payments", "search"])
    d.edge(["users", "orders", "payments"], "pg"); d.edge("search", "es")


def _g_hub(d):
    d.node("hub", "Message broker", icon="network")
    for n in ("Billing", "Email", "Search", "Audit", "Analytics", "Inventory", "Shipping", "Fraud"):
        d.node(n.lower(), n, icon="server")
        d.edge("hub", n.lower())


def _g_wide(d):
    d.node("src", "Ingest", icon="bolt")
    d.node("sink", "Warehouse", shape="cylinder")
    for i in range(10):
        d.node(f"w{i}", f"Worker {i + 1}", icon="gear")
        d.edge("src", f"w{i}"); d.edge(f"w{i}", "sink")


def _g_deep(d):
    for i in range(12):
        d.node(f"s{i}", f"Stage {i + 1}", icon="gear")
        if i:
            d.edge(f"s{i - 1}", f"s{i}")
        if i in (3, 7):
            d.node(f"x{i}", f"Check {i + 1}", shape="diamond")
            d.edge(f"s{i}", f"x{i}")


def _g_two(d):
    for k, word in (("a", "Order"), ("b", "Refund")):
        for i in range(1, 5):
            d.node(f"{k}{i}", f"{word} step {i}", icon="check")
        d.chain(*[f"{k}{i}" for i in range(1, 5)])


def _g_mesh(d):
    front, mid, back = ["Web", "Mobile", "Partner"], ["Auth", "Orders", "Search", "Billing"], ["Postgres", "Redis", "Kafka"]
    for n in front + mid + back:
        d.node(n.lower(), n, icon="server")
    d.edge([n.lower() for n in front], [n.lower() for n in mid])
    d.edge([n.lower() for n in mid], [n.lower() for n in back])
    d.edge("auth", "orders"); d.edge("orders", "billing")


GRAPHS = dict(chain=_g_chain, tree=_g_tree, tiers=_g_tiers, hub=_g_hub, wide=_g_wide, deep=_g_deep, two=_g_two,
              mesh=_g_mesh)


def _clean(d):
    """No overlapping nodes, no edge through a node, no edge label on a node."""
    _segments_clear(d)
    boxes = list(d._res["boxes"].values())
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _hit(boxes[i], boxes[j]), "nodes overlap"
    for lb in d.label_boxes:
        for b in d.boxes.values():
            assert not _hit(lb, b), "edge label on a node"


@pytest.mark.parametrize("size", [(1920, 1080), (1080, 1350), (1600, 1200)])
@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_fills_the_frame(name, size):
    W, H = size
    d = Diagram(title=name.title())
    GRAPHS[name](d)
    d.draw(Canvas(W, H))
    x0, y0, x1, y1 = d.fit["content"]
    cx, cy = (x1 - x0) / W, (y1 - y0) / H
    assert max(cx, cy) >= 0.8 and min(cx, cy) >= 0.6, (name, size, d.fit)
    assert d.fit["font_px"] >= 14 * H / 1080, d.fit
    _clean(d)


def test_font_size_is_pixels_at_1080p():
    def small():
        d = Diagram(font_size=26, grow=False, title="T")
        d.chain("a", "b", "c")
        return d
    d = small()
    d.draw(Canvas(1920, 1080))
    assert abs(d.fit["font_px"] - 26) < 0.6
    d = small()
    d.draw(Canvas(960, 540))                         # a --scale 0.5 draft looks the same
    assert abs(d.fit["font_px"] - 13) < 0.4
    g = Diagram(font_size=26, title="T")              # with room to spare the nodes and text grow
    g.chain("a", "b", "c")
    g.draw(Canvas(1920, 1080))
    assert g.fit["font_px"] > 30
    big = Diagram(font_size=26)                       # too many nodes: shrinks to fit, never off the canvas
    _g_wide(big)
    _g_tree(big)
    big.draw(Canvas(960, 540))
    x0, y0, x1, y1 = big.fit["content"]
    assert x0 >= 0 and y0 >= 0 and x1 <= 960 and y1 <= 540


def test_auto_direction_follows_the_box():
    assert Diagram().direction == "LR" and Diagram()._auto
    c = Diagram()                                     # a long chain folds into rows instead of a thin strip
    _g_chain(c)
    c.draw(Canvas(1920, 1080))
    assert c.fit["bands"] >= 2
    h = Diagram()                                     # a hub with many leaves on a wide canvas: LR
    _g_hub(h)
    h.draw(Canvas(1920, 1080))
    assert h.fit["direction"] == "LR"
    w = Diagram()                                     # a wide fan-out on a wide canvas: TB
    _g_wide(w)
    w.draw(Canvas(1920, 1080))
    assert w.fit["direction"] == "TB"
    fixed = Diagram(direction="TB")
    _g_chain(fixed)
    fixed.draw(Canvas(1920, 1080))
    assert fixed.fit["direction"] == "TB" and fixed.fit["bands"] == 1


def test_same_rank_peers():
    d = Diagram(direction="TB")
    d.chain("a", "b", "c")
    d.edge("b", "peer", rank="same")
    d.same_rank("a", "x")
    d.edge("x", "y")
    d.build()
    r = d._rank
    assert r["b"] == r["peer"] and r["a"] == r["x"] and r["y"] == r["x"] + 1
    assert next(e for e in d.edges if (e.source, e.target) == ("b", "peer")).kind == "flat"
    with pytest.raises(ValueError):
        d.edge("a", "b", rank="sideways")


def test_bus_taps_without_moving_ranks():
    def base():
        d = Diagram(direction="TB")
        d.node("gw", "Gateway")
        d.edge("gw", ["msg", "pres", "auth"])
        d.edge(["msg", "auth"], "db")
        return d
    plain, with_bus = base(), base()
    with_bus.bus("redis", "Redis pub/sub", members=["msg", "pres"], style="dashed")
    with_bus.edge("auth", "redis", arrow="end")                  # an edge to a bus id adds a tap
    plain.build()
    with_bus.draw(Canvas(1920, 1080))
    assert plain._rank == with_bus._rank
    b = with_bus._res["buses"]["redis"]
    assert {m for m, _ in b["taps"]} == {"msg", "pres", "auth"}
    (bx0, by0), (bx1, by1) = b["bar"]
    assert abs(by0 - by1) < 1e-6                                  # TB: a horizontal bar
    for m, pts in b["taps"]:                                     # every tap ends on the bar, below its member
        assert abs(pts[-1][1] - by0) < 1e-6 and pts[0][1] < by0
    taps = [e for e in with_bus.edges if e.get("bus")]
    assert len(taps) == 3 and all(e.target == "redis" for e in taps)
    assert with_bus.nodes["redis"]["shape"] == "bus" and "box" in with_bus.nodes["redis"]
    _clean(with_bus)
    with pytest.raises(ValueError):
        with_bus.bus("gw")


def test_cycles_stay_compact():
    d = Diagram()
    ring = [f"n{i}" for i in range(6)]
    d.chain(*ring, ring[0])
    d.edge("n3", "n1", style="dashed")
    d.draw(Canvas(1920, 1080))
    assert max(d._rank.values()) <= 5
    assert max(d._res["W"], d._res["H"]) < 3000
    assert [e.kind for e in d.edges].count("back") >= 1
    _clean(d)


@pytest.mark.parametrize("lanes", ["rows", "columns", "auto"])
def test_lanes_make_full_width_tiers(lanes):
    d = Diagram(lanes=lanes, title="Tiers")
    _g_tiers(d)
    d.draw(Canvas(1920, 1080))
    gb = d.group_boxes
    rows = d.fit["direction"] == "TB"
    lo, hi = (0, 2) if rows else (1, 3)
    spans = {(round(b[lo], 1), round(b[hi], 1)) for b in gb.values()}
    assert len(spans) == 1, "every tier spans the same width"
    order = sorted(gb, key=lambda g: gb[g][1 if rows else 0])
    assert order == ["cl", "edge", "svc", "data"]
    ax = 0 if rows else 1                             # members spread evenly: equal steps between neighbours
    xs = sorted(d.boxes[k][ax] for k in ("users", "orders", "payments", "search"))
    steps = [b - a for a, b in zip(xs, xs[1:])]
    assert max(steps) - min(steps) < 1.0
    _clean(d)


def test_tiers_sets_the_order():
    d = Diagram()
    _g_tiers(d)
    d.tiers(["data", "svc", "edge", "cl"], lanes="columns")
    d.draw(Canvas(1920, 1080))
    gb = d.group_boxes
    assert sorted(gb, key=lambda g: gb[g][0]) == ["data", "svc", "edge", "cl"]
    with pytest.raises(ValueError):
        d.tiers(["cl"], lanes="diagonal")


def test_nodes_and_edges_are_plain_data():
    import json
    d = _platform()
    assert d.nodes["gw"]["label"].startswith("API gateway") and d.nodes["users"].group == "svc"
    d.draw(Canvas(960, 540))
    json.dumps({"nodes": d.nodes, "edges": d.edges})
    for n in d.nodes.values():
        x0, y0, x1, y1 = n["box"]
        assert 0 <= x0 < x1 <= 960 and 0 <= y0 < y1 <= 540
    assert {"source", "target", "label", "style", "arrow"} <= set(d.edges[0])


def test_manifest_meta_has_labels_groups_and_edges():
    from pixelcanvas import manifest
    d = _platform()
    cv = Canvas(960, 540)
    d.draw(cv)
    ents = {e["meta"].get("id"): e for e in manifest.entries(cv) if e.get("kind") in ("node", "group")}
    gw = ents["gw"]["meta"]
    assert gw["group"] == "edge" and {x["target"] for x in gw["edges"]} == {"users", "orders", "search"}
    assert set(ents["svc"]["meta"]["members"]) == {"users", "orders", "search"}

# ---------------------------------------------------------------- flowcharts fill the frame in either orientation

def _f_linear(d):
    d.node("s", "Start", shape="pill"); d.node("a", "Receive order"); d.node("b", "Check stock")
    d.node("q", "In stock?", shape="diamond"); d.node("c", "Pack and ship"); d.node("n", "Notify customer")
    d.node("e", "End", shape="pill")
    d.chain("s", "a", "b", "q"); d.edge("q", "c", "yes"); d.edge("q", "n", "no"); d.edge("c", "e"); d.edge("n", "e")


def _f_loop(d):
    d.node("s", "Start", shape="pill")
    for i, t in (("a", "Submit form"), ("b", "Validate input"), ("c", "Charge card"), ("f", "Send receipt")):
        d.node(i, t)
    d.node("q1", "Valid?", shape="diamond"); d.node("q2", "Paid?", shape="diamond"); d.node("e", "End", shape="pill")
    d.node("x", "Show error")
    d.chain("s", "a", "b", "q1"); d.edge("q1", "c", "yes"); d.edge("q1", "x", "no"); d.edge("x", "a")
    d.edge("c", "q2"); d.edge("q2", "f", "yes"); d.edge("q2", "b", "no", style="dashed"); d.edge("f", "e")


def _f_approval(d):
    d.node("s", "Request", shape="pill")
    for i, t in (("a", "Draft"), ("b", "Team lead review"), ("c", "Finance review"), ("d", "Legal review"),
                 ("f", "Sign-off")):
        d.node(i, t)
    d.node("r", "Rejected", shape="diamond"); d.node("e", "Approved", shape="pill")
    d.chain("s", "a", "b", "c", "d", "f", "e")
    d.edge("d", "r"); d.edge("r", "s", "revise", style="dashed")


def _f_plan(d):
    d.node("s", "Choose a plan", shape="pill"); d.node("q", "Team size?", shape="diamond")
    d.node("a", "Free"); d.node("b", "Pro"); d.node("c", "Enterprise"); d.node("m", "Create account")
    d.node("p", "Payment"); d.node("e", "Welcome", shape="pill")
    d.edge("s", "q"); d.edge("q", "a", "1-3"); d.edge("q", "b", "4-50"); d.edge("q", "c", "50+")
    d.edge(["a", "b", "c"], "m"); d.edge("m", "p"); d.edge("p", "e")


def _f_pr(d):
    d.node("o", "Open PR", shape="pill"); d.node("ci", "Run CI"); d.node("t", "Tests pass?", shape="diamond")
    d.node("r", "Code review"); d.node("ap", "Approved?", shape="diamond"); d.node("ch", "Request changes")
    d.node("fx", "Push fixes"); d.node("mg", "Merge"); d.node("dp", "Deploy"); d.node("e", "Done", shape="pill")
    d.chain("o", "ci", "t"); d.edge("t", "r", "yes"); d.edge("t", "fx", "no"); d.edge("r", "ap")
    d.edge("ap", "mg", "yes"); d.edge("ap", "ch", "no"); d.edge("ch", "fx"); d.edge("fx", "ci", style="dashed")
    d.edge("mg", "dp"); d.edge("dp", "e")


def _f_funnel(d):
    d.node("v", "Visit site", shape="pill"); d.node("s", "Sign up"); d.node("q1", "Email verified?", shape="diamond")
    d.node("p", "Create profile"); d.node("q2", "Invited team?", shape="diamond"); d.node("t", "First project")
    d.node("q3", "Activated?", shape="diamond"); d.node("a", "Active user", shape="pill")
    d.node("x1", "Exit: bounce", shape="pill"); d.node("x2", "Exit: dormant", shape="pill")
    d.node("x3", "Exit: churn", shape="pill")
    d.chain("v", "s", "q1"); d.edge("q1", "p", "yes"); d.edge("q1", "x1", "no"); d.edge("p", "q2")
    d.edge("q2", "t", "yes"); d.edge("q2", "x2", "no"); d.edge("t", "q3"); d.edge("q3", "a", "yes")
    d.edge("q3", "x3", "no")


FLOWS = dict(linear=_f_linear, loop=_f_loop, approval=_f_approval, plan=_f_plan, pr=_f_pr, funnel=_f_funnel)


@pytest.mark.parametrize("size", [(1920, 1080), (1600, 1200), (1080, 1350), (1080, 1920)])
@pytest.mark.parametrize("name", sorted(FLOWS))
def test_flowcharts_fill_the_frame(name, size):
    W, H = size
    d = Diagram(title=name.title())
    FLOWS[name](d)
    d.draw(Canvas(W, H))
    x0, y0, x1, y1 = d.fit["content"]
    cx, cy = (x1 - x0) / W, (y1 - y0) / H
    assert max(cx, cy) >= 0.8 and min(cx, cy) >= 0.6, (name, size, d.fit)
    assert d.fit["font_px"] >= 0.016 * min(W, H), d.fit
    if W / H >= 1.5:
        assert d.fit["direction"] == "LR", d.fit
    if H / W >= 1.2:
        assert d.fit["direction"] == "TB", d.fit
    _clean(d)


# ---- v0.4 round: font_size floor, keep_order, leader styles, bus manifest ------------------------------------

def _arch(**kw):
    d = Diagram(theme="dark", lanes="rows", **kw)
    with d.group("c", "Client tier"):
        d.node("web", "Web client"); d.node("mob", "Mobile client")
    with d.group("e", "Edge tier"):
        d.node("lb", "Load balancer"); d.node("gw", "API gateway")
    with d.group("s", "Service tier"):
        d.node("auth", "Auth service"); d.node("msg", "Messaging service")
        d.node("pres", "Presence service"); d.node("push", "Push worker")
    with d.group("d", "Data tier"):
        d.node("pg", "Postgres", shape="cylinder"); d.node("obj", "Object storage", shape="cylinder")
    d.bus("redis", "Redis pub/sub", members=["msg", "pres", "push"], style="dashed")
    d.edge(["web", "mob"], "lb", label="HTTPS"); d.edge("lb", "gw")
    d.edge("gw", ["auth", "msg", "pres"]); d.edge(["auth", "msg"], "pg"); d.edge("msg", "obj")
    return d


def _fit_px(d, w=960, h=540):
    from pixelcanvas import Canvas
    from pixelcanvas.vector import THEMES
    cv = Canvas(w, h, THEMES["dark"].background)
    d.draw(cv)
    return d.fit["font_px"], cv


def test_font_size_is_a_floor_the_layout_grows_around():
    small, _ = _fit_px(_arch(font_size=14, grow=False))
    mid, _ = _fit_px(_arch(font_size=22, grow=False))
    big, _ = _fit_px(_arch(font_size=26, grow=False))
    assert small < mid < big                                  # raising font_size changes the text
    assert mid >= 22 * 0.5 * 0.97                             # honoured at the preview size (S = 0.5)
    shrink, _ = _fit_px(_arch(font_size=44, floor=False))
    floor, _ = _fit_px(_arch(font_size=44))
    assert floor >= shrink


def test_default_font_size_is_comfortable():
    px, _ = _fit_px(_arch(), 1920, 1080)
    assert Diagram().font_size >= 26 and px >= 22


def test_keep_order_keeps_declared_tier_order():
    d = _arch(keep_order=True)
    _fit_px(d)
    boxes = d.nodes
    xs = [boxes[k]["box"][0] for k in ("auth", "msg", "pres", "push")]
    assert xs == sorted(xs)
    assert [boxes[k]["box"][0] for k in ("pg", "obj")] == sorted(boxes[k]["box"][0] for k in ("pg", "obj"))
    e = _arch()
    e.tiers(["c", "e", "s", "d"], keep_order=True)
    assert e.keep_order is True


def test_callout_leader_styles():
    from pixelcanvas import Canvas
    from pixelcanvas.diagram import Annotator
    cv = Canvas(640, 360, (1, 1, 1))
    ann = Annotator(cv, leader="arrow", leader_color="#d33", leader_width=3)
    ann.callout((0.3, 0.5), "Nucleus", side="right")
    ann.callout((0.4, 0.3), "Wall", side="right", leader="dot")
    ann.callout((0.5, 0.7), "Vacuole", side="right", leader="none")
    ann.draw()
    leaders = [g for g in ann.target.__dict__.get("_drawings", [])]
    assert leaders
    with pytest.raises(Exception, match="dot"):
        Annotator(cv, leader="dto")
    with pytest.raises(Exception):
        ann.callout((0.1, 0.1), "x", leader="arow")


def test_bus_is_recorded_by_its_label():
    from pixelcanvas import manifest
    _, cv = _fit_px(_arch())
    e = [x for x in cv.manifest if x["name"] == "Redis pub/sub"]
    assert e and "bus" in e[0]["aliases"] and e[0]["meta"]["shape"] == "bus"


# ---------------------------------------------------------------- sequence: otherwise label once, actor sized like peers

def _seq_actor():
    from pixelcanvas.diagram import Sequence
    s = Sequence()
    s.participant("u", "User", kind="actor")
    s.participant("a", "App")
    s.participant("d", "DB", kind="database")
    s.message("u", "a", "login")
    with s.frame("alt", "ok"):
        s.reply("a", "u", "token")
        s.otherwise("denied")
        s.reply("a", "u", "401")
    return s


def test_sequence_otherwise_label_is_drawn_once():
    s = _seq_actor()
    svg = s.to_svg()
    assert svg.count("[denied]") == 1


def test_sequence_actor_is_sized_like_the_other_participants():
    from pixelcanvas import manifest
    s = _seq_actor()
    cv = Canvas(960, 540)
    s.draw(cv)
    rows = {r["name"]: r for r in manifest.summary(cv)}
    actor = rows["User"]
    peer = rows["App"]
    assert actor["size"][1] >= 0.8 * peer["size"][1]
    assert actor["area_pct"] >= 0.5 * peer["area_pct"]
    assert not [w for w in manifest.check(cv) if w.get("code", w.get("rule")) == "object_tiny"]
