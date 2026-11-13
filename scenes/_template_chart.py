# -*- coding: utf-8 -*-
"""One publication-quality chart with title, callout and source line (chart)

* Format: 1600x900 (1080x1080 square for social, 1920x1080 slide)
* Message: <the one thing the chart says, as the TITLE>
* Data: <what is on x, what is on y, units, the highlighted series>
* Callout: <the point that proves the message>
* Palette: <theme THEME; the main series in the first accent, comparison dashed>
* Exact text: <TITLE, SUBTITLE, labels and numbers copied from the request>
"""
# Read skills/pixelcanvas/references/charts.md (kinds, axes options, number formats, themes).
from pixelcanvas import Canvas, chart

PROMPT = ""            # paste the request here, verbatim - --check compares the picture against it
KIND = "diagram"
NAME = "my_chart"
DEFAULT_SIZE = (1600, 900)
DEFAULT_SEED = 1

THEME = "light"                     # light dark paper blueprint pastel neon mono
# >>> data: new_scene.py --prompt writes the request's chart type, title, categories and series here
CHART = "line"                      # line area bar barh scatter donut
TITLE = "Solar now beats coal on cost"
SUBTITLE = "Levelised cost of electricity, USD per MWh"
X = ["2016", "2018", "2020", "2022", "2024", "2026"]
SERIES = {"Solar": [101, 72, 55, 48, 41, 36], "Coal": [102, 102, 109, 117, 118, 121]}
CALLOUT = ("Solar cheaper from 2018", ("2018", 72))     # text, (x, y) of the point it explains; None for none
SOURCE = "Source: example data for the template"
# <<< data


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height)
    fig = chart.figure(cv, theme=THEME)
    # axes: frame="axes" for papers, "minimal" for slides; yfmt "$", "%", "k", " ms" ...
    ax = fig.axes((0.04, 0.05, 0.96, 0.88), title=TITLE, subtitle=SUBTITLE, yfmt="$", legend="tr")
    if CHART == "donut":                                 # shares of the first series
        ax.donut(dict(zip(X, next(iter(SERIES.values())))))
    elif CHART in ("bar", "barh"):                       # one bar group per category, one colour per series
        getattr(ax, CHART)(X, SERIES if len(SERIES) > 1 else next(iter(SERIES.values())), highlight=0)
    else:
        for i, (label, ys) in enumerate(SERIES.items()):     # line / area / scatter: see charts.md
            if CHART == "scatter":
                ax.scatter(X, ys, trend=True)
                continue
            ax.line(X, ys, label=label, smooth=True, width=3.2 if i == 0 else 2.4, dash=None if i == 0 else "dash",
                    fill=i == 0 or CHART == "area", end_label=True)
    if CALLOUT:
        ax.annotate(CALLOUT[0], CALLOUT[1])
    fig.caption(SOURCE, box=(0.04, 0.92, 0.96, 0.98), size=22)
    fig.draw(log=log)                                    # one layer per series, legend, callout; --svg exports it
    return cv
