# -*- coding: utf-8 -*-
"""Header, KPI tiles and several charts in cards (chart)

* Format: 1920x1080
* Focal point: <the first KPI tile and the main trend chart>
* Reading order: <header -> KPI row -> trend (wide) + share (narrow) -> ranking>
* Panels: <KPIS tiles, TREND line, SHARE donut, RANK bars>
* Palette: <theme THEME; good = green, bad = red deltas>
* Exact text: <TITLE, period, KPI names and numbers copied from the request>
"""
# Read skills/pixelcanvas/references/charts.md (figure, header, kpi, grid, axes).
from pixelcanvas import Canvas, chart

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "diagram"
NAME = "my_dashboard"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "dark"                      # dark light paper neon ...
# >>> data: new_scene.py --prompt writes the request's title, KPI tiles and panels here
TITLE, PERIOD = "Store performance", "Q3 2026 \u00b7 all regions"
KPIS = [("Revenue", 1284600, 12.4, "$.2k", "up"), ("Orders", 18420, 8.1, None, "up"),   # name, value, delta %, fmt,
        ("Refunds", 312, -4.2, None, "down")]                                          # good direction
TREND = ("line", "Weekly revenue", "$k",               # line / area / bar, panel title, value format
         ["2026-07-06", "2026-07-20", "2026-08-03", "2026-08-17", "2026-08-31", "2026-09-14", "2026-09-28"],
         {"2026": [84e3, 86e3, 97e3, 112e3, 101e3, 108e3, 117e3], "2025": [76e3, 81e3, 84e3, 89e3, 87e3, 91e3, 95e3]})
SHARE = ("Sales by channel", {"Search": 34, "Direct": 24, "Ads": 18, "Email": 14, "Social": 10})     # donut
RANK = ("Top products", "$k", {"Headphones": 182e3, "Smart watch": 142e3, "Laptop stand": 96e3, "USB-C hub": 75e3,
                               "Desk lamp": 58e3})                                                 # bars
# <<< data


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height)
    fig = chart.figure(cv, theme=THEME)
    fig.header(TITLE, PERIOD, icon="chart", box=(0.02, 0.025, 0.98, 0.11))
    # boxes are canvas fractions; fig.grid(rows, cols, box, widths=) splits a band into panels
    for i, (box, (name, value, delta, fmt, good)) in enumerate(zip(fig.grid(1, len(KPIS), box=(0.02, 0.135, 0.98, 0.29)),
                                                                    KPIS)):
        fig.kpi(box, name, value, delta=delta, fmt=fmt, note="vs last period", color=i, good=good)
    trend, share = fig.grid(1, 2, box=(0.02, 0.315, 0.98, 0.64), widths=(2, 1))
    kind, title, fmt, x, series = TREND
    ax = fig.axes(trend, title=title, yfmt=fmt, card=True)
    if kind == "bar":
        ax.bar(x, series if len(series) > 1 else next(iter(series.values())))
    for i, (label, ys) in enumerate(series.items() if kind != "bar" else ()):
        ax.line(x, ys, label=label, smooth=True, fill=i == 0 or kind == "area", dash=None if i == 0 else "dash")
    fig.axes(share, title=SHARE[0], card=True).donut(SHARE[1], center=(str(len(SHARE[1])), SHARE[0].split()[0].lower()))
    fig.axes((0.02, 0.665, 0.98, 0.975), title=RANK[0], xfmt=RANK[1], card=True).barh(
        RANK[2], value_labels=True, fmt=RANK[1], highlight=0)
    fig.draw(log=log)                                    # one layer per tile, chart, series and legend
    return cv
