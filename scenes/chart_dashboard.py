# -*- coding: utf-8 -*-
"""
Analytics dashboard (pixelcanvas.chart, dark theme).

* Format: 1920x1080; one chart Figure in 1080-tall units, so a --scale 0.5 draft looks identical.
* Layout: header band (title, subtitle, period chips) -> a row of 4 KPI tiles (value, delta pill, sparkline) ->
  weekly revenue line (this year vs last year, smooth, gradient fill, callout on the sale week) beside a channel
  donut -> top products horizontal bars beside an orders-by-hour heatmap.
* Style: THEME surface cards with soft shadows on the theme background; accents in chart.PALETTE_ORDER.
* Every chart object (card, axes, each series, legend, callout) is its own layer; fig.save_svg() gives the SVG.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas import chart

NAME = "chart_dashboard"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 7

THEME = "dark"
GAP = 0.022                     # space between cards (fraction of the canvas height)
TITLE = "Store performance"
SUBTITLE = "Online store  ·  Q3 2026  ·  all regions"
CHIPS = ["All channels", "Jul 1 – Sep 30, 2026"]

WEEKS = ["2026-07-06", "2026-07-13", "2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10", "2026-08-17",
         "2026-08-24", "2026-08-31", "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28"]
REV_2026 = [84.2, 88.5, 86.1, 91.7, 97.3, 128.6, 112.4, 99.8, 101.5, 104.9, 108.2, 111.6, 117.3]   # $k per week
REV_2025 = [76.4, 79.1, 80.6, 78.8, 83.5, 92.1, 88.7, 85.2, 86.9, 88.4, 90.6, 92.3, 95.1]
CHANNELS = {"Organic search": 34, "Direct": 24, "Paid ads": 18, "Email": 14, "Social": 10}
PRODUCTS = {"Aurora wireless headphones": 182400, "Pulse smart watch": 141900, "Nimbus laptop stand": 96300,
            "Volt USB-C hub (8-in-1)": 74800, "Halo desk lamp": 58200, "Drift travel backpack": 47600}
HOURS = ["0h", "2h", "4h", "6h", "8h", "10h", "12h", "14h", "16h", "18h", "20h", "22h"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _orders_by_hour(seed):
    rng = np.random.default_rng(seed)
    h = np.arange(12) * 2 + 1
    day = np.exp(-((h - 13) / 5.0) ** 2) * 0.7 + np.exp(-((h - 20.5) / 2.6) ** 2) + 0.06
    week = np.array([0.86, 0.82, 0.88, 0.93, 1.0, 1.22, 1.14])
    m = np.outer(week, day) * 260 * (1 + rng.normal(0, 0.06, (7, 12)))
    return np.round(m)


def build(cv, seed=DEFAULT_SEED):
    fig = chart.figure(cv, theme=THEME)
    th = fig.theme
    fig.header(TITLE, SUBTITLE, right=CHIPS, icon="chart", box=(0.022, 0.024, 0.978, 0.112))
    rng = np.random.default_rng(seed)

    kpis = fig.grid(1, 4, box=(0.022, 0.138, 0.978, 0.300), gap=GAP)
    def spark(base, drift, n=20):
        i = np.arange(n)
        return base + drift * i + 0.5 * np.sin(i * 0.5 + rng.uniform(0, 6)) + np.cumsum(rng.normal(0, 0.3, n))
    fig.kpi(kpis[0], "Revenue", 1284600, fmt="$.2k", delta=12.4, note="vs Q2", spark=spark(20, 0.45), color=0,
            icon="cart")
    fig.kpi(kpis[1], "Orders", 18420, delta=8.1, note="vs Q2", spark=spark(20, 0.3), color=1, icon="tag")
    fig.kpi(kpis[2], "Conversion rate", "3.42%", delta="−0.3 pt", note="vs Q2", spark=spark(20, -0.12), color=2,
            icon="users")
    fig.kpi(kpis[3], "Avg. order value", 69.74, fmt="$.2", delta=3.9, note="vs Q2", spark=spark(20, 0.2), color=3,
            icon="trend")

    mid = fig.grid(1, 2, box=(0.022, 0.322, 0.978, 0.640), gap=GAP, widths=(2.05, 1))
    ax = fig.axes(mid[0], title="Weekly revenue", subtitle="Gross sales, thousands of USD", yfmt="$", card=True,
                  legend="auto")
    ax.line(WEEKS, REV_2026, label="2026", smooth=True, fill=True, width=3)
    ax.line(WEEKS, REV_2025, label="2025", smooth=True, dash="dash", width=2.2, markers=False)
    ax.annotate("Summer sale  +38%", (WEEKS[5], REV_2026[5]))
    ax2 = fig.axes(mid[1], title="Sales by channel", subtitle="Share of revenue", card=True)
    ax2.donut(CHANNELS, center=("$1.28M", "revenue"))

    bot = fig.grid(1, 2, box=(0.022, 0.662, 0.978, 0.976), gap=GAP, widths=(1, 1.45))
    ax3 = fig.axes(bot[0], title="Top products", subtitle="Revenue this quarter", card=True, xfmt="$k")
    ax3.barh(PRODUCTS, value_labels=True, highlight=0, fmt="$k")
    ax4 = fig.axes(bot[1], title="Orders by weekday and hour", subtitle="Local time, all channels", card=True)
    ax4.heatmap(_orders_by_hour(seed), HOURS, DAYS, annotate=True, fmt="k", cmap=0)
    return fig


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    build(cv, seed).draw(log=log)
    return cv
