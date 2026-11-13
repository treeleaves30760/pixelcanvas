# -*- coding: utf-8 -*-
"""Light analytics dashboard for a subscription app: four KPI tiles, revenue trend with a launch marker, bars, donut

* Format: 1920x1080
* Focal point: the MRR tile and the revenue line climbing after the April launch
* Reading order: header -> KPI row -> revenue trend (wide) -> sign-ups by channel -> plan mix
* Palette: light theme; good deltas green, bad deltas red (churn going down is good)
* Exact text: KPI names MRR, Active users, Churn, NPS; chart titles; launch marker
"""
# Read skills/pixelcanvas/references/charts.md (figure, header, kpi, grid, axes).
from pixelcanvas import Canvas, chart

PROMPT = ("An analytics dashboard for a subscription app: four KPI tiles (MRR, active users, churn, NPS), a revenue "
          "line chart for the last 12 months with a marked launch date, a bar chart of sign-ups by channel, and a "
          "donut of plan mix. Light theme.")
NAME = "chart_subscriptions"
KIND = "design"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 1

THEME = "light"
TITLE, PERIOD = "Subscription analytics", "Last 12 months · Nov 2025 - Oct 2026"
KPIS = [("MRR", 128400, 12.4, "$.1k"), ("Active users", 48210, 8.1, None), ("Churn", 2.8, -0.4, "%"),
        ("NPS", 54, 6, None)]
MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
          "2026-08", "2026-09", "2026-10"]
REVENUE = [71, 74, 78, 80, 88, 93, 101, 108, 112, 118, 124, 128]      # MRR in $k per month
SPARKS = {"MRR": REVENUE, "Active users": [30, 32, 35, 36, 38, 40, 41, 43, 44, 46, 47, 48],
          "Churn": [3.6, 3.5, 3.4, 3.3, 3.2, 3.2, 3.1, 3.0, 3.0, 2.9, 2.9, 2.8],
          "NPS": [41, 42, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54]}
CHANNELS = {"Organic": 4200, "Paid ads": 3100, "Referral": 2400, "Email": 1500, "Partners": 900}
PLANS = {"Free": 52, "Pro": 31, "Team": 12, "Enterprise": 5}
LAUNCH = "2026-04"


def render(width=DEFAULT_SIZE[0], height=DEFAULT_SIZE[1], seed=DEFAULT_SEED, log=None):
    cv = Canvas(width, height)
    fig = chart.figure(cv, theme=THEME)
    fig.header(TITLE, PERIOD, icon="chart", box=(0.02, 0.025, 0.98, 0.11))
    for i, (box, (name, value, delta, fmt)) in enumerate(zip(fig.grid(1, 4, box=(0.02, 0.135, 0.98, 0.30)), KPIS)):
        fig.kpi(box, name, value, delta=delta, fmt=fmt, note="vs last month", color=i,
                good="down" if name == "Churn" else "up", spark=SPARKS[name])
    trend, bars, share = fig.grid(1, 3, box=(0.02, 0.325, 0.98, 0.975), widths=(5, 3, 3))
    ax = fig.axes(trend, title="Revenue, last 12 months", subtitle="MRR", yfmt="$k", card=True)
    ax.line(MONTHS, [v * 1000 for v in REVENUE], label="MRR", smooth=True, fill=True)
    ax.vline(LAUNCH, label="Launch, Apr 2026")
    fig.axes(bars, title="Sign-ups by channel", card=True).bar(CHANNELS, value_labels=True, highlight=0)
    fig.axes(share, title="Plan mix", card=True).donut(PLANS, center=("4", "plans"))
    fig.draw(log=log)                                    # one layer per header, tile and chart
    return cv
