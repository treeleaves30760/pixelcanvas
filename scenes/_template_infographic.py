# -*- coding: utf-8 -*-
"""Title band + panels mixing a chart, a diagram and a key number (chart + diagram)

* Format: 1920x1080 (portrait 1080x1350 stacks the panels)
* Focal point: <the key number in the donut>
* Reading order: <title -> key number -> chart with a callout -> how-it-works diagram -> source line>
* Panels: <donut with the key number, bar chart with a callout, diagram card>
* Palette: <theme THEME; one accent per panel>
* Exact text: <TITLE, SUBTITLE, labels and numbers copied from the request>
"""
# Read skills/pixelcanvas/references/charts.md (figure, kpi, axes) and diagrams.md (Diagram, callouts).
from pixelcanvas import Canvas, chart
from pixelcanvas.diagram import Diagram

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "diagram"
NAME = "my_infographic"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "light"
# >>> data: new_scene.py --prompt writes the request's title, shares, bars and steps here
TITLE = "Where our solar power goes"
SUBTITLE = "Household system, 2026 average day"
SHARE = ("Solar energy", {"Used at home": 68, "Sold to grid": 32})       # donut; the first share is the key number
DATA = ("Use by appliance", "%", {"Heating": 31, "EV charging": 24, "Kitchen": 15, "Laundry": 11, "Other": 19},
        "Heat pump, Nov-Mar")                                             # title, format, bars, callout on bar 1
STEPS = ("How it flows", [("sun", "Panels", "bolt"), ("inv", "Inverter", "chip"), ("bat", "Battery", "layers"),
                          ("home", "Home", "home")])                     # (id, label, icon) chained in order
SOURCE = "Source: home energy monitor, Jan-Oct 2026"
# <<< data


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height)
    fig = chart.figure(cv, theme=THEME)
    fig.header(TITLE, SUBTITLE, box=(0.03, 0.03, 0.97, 0.15))
    # panel boxes are canvas fractions; grid(rows, cols, box, widths=) splits a region
    portrait = height > width
    rows, cols = (3, 1) if portrait else (1, 3)
    key_box, chart_box, flow_box = fig.grid(rows, cols, box=(0.03, 0.19, 0.97, 0.9), gap=0.025,
                                            **({} if portrait else {"widths": (0.8, 1.4, 1.2)}))
    key, share = next(iter(SHARE[1].items()))
    fig.axes(key_box, title=SHARE[0], card=True).donut(SHARE[1], center=(f"{share}%", key.lower()))
    ax = fig.axes(chart_box, title=DATA[0], card=True)
    ax.bar(DATA[2], value_labels=True, fmt=DATA[1], highlight=0)
    if DATA[3]:
        ax.annotate(DATA[3], next(iter(DATA[2].items())))
    flow = fig.axes(flow_box, title=STEPS[0], card=True, frame="none", grid=False, legend="none")
    fig.caption(SOURCE, box=(0.03, 0.93, 0.97, 0.98), size=20)
    fig.draw(log=log)                                   # one layer per header, tile, chart and caption

    d = Diagram(theme=THEME, node_style="tint")          # direction="auto" fits the panel
    for i, (nid, label, icon) in enumerate(STEPS[1]):   # one node per step, chained in order
        d.node(nid, label, icon=icon, color=i)
    d.chain(*[s[0] for s in STEPS[1]])
    k = cv.h / 1080                                     # figure units -> pixels
    x0, y0, x1, y1 = flow.box
    d.draw(cv, box=(x0 * k + 30 * k, y0 * k + 80 * k, x1 * k - 30 * k, y1 * k - 20 * k), log=log)
    return cv
