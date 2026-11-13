# -*- coding: utf-8 -*-
"""Tests for pixelcanvas.chart: formatting, ticks, every chart kind, layout (no clipped labels), SVG, doc recipes."""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelcanvas import Canvas  # noqa: E402
from pixelcanvas import chart  # noqa: E402
from pixelcanvas.chart import fmt_number, nice_ticks  # noqa: E402
from pixelcanvas.vector import Text, Group  # noqa: E402

DOC = ROOT / "skills" / "pixelcanvas" / "references" / "charts.md"


# ---------------------------------------------------------------- numbers and ticks

@pytest.mark.parametrize("v, fmt, want", [
    (1250, "$", "$1,250"),
    (1250, "$k", "$1.3k"),
    (1284600, "$.2k", "$1.28M"),
    (12400, "k", "12.4k"),
    (12.5, "%", "12.5%"),
    (2, "+.1%", "+2.0%"),
    (-3.25, None, "−3.25"),
    (12, " ms", "12 ms"),
    (0.256, "{:.0%}", "26%"),
    (250000, None, "250k"),
    (69.74, "$.2", "$69.74"),
])
def test_fmt_number(v, fmt, want):
    assert fmt_number(v, fmt) == want


def test_fmt_callable_and_string():
    assert fmt_number(3, lambda v: f"<{v}>") == "<3>"
    assert fmt_number("n/a") == "n/a"


def test_nice_ticks():
    t, step = nice_ticks(0, 97, 5)
    assert t[0] == 0 and t[-1] >= 97 and step in (20, 25, 10)
    t, _ = nice_ticks(0.013, 0.087, 4)
    assert t[0] <= 0.013 and t[-1] >= 0.087
    assert len(set(t)) == len(t)


def test_tick_labels_share_unit():
    labs = chart._fmt_ticks([0, 50000, 100000, 150000], "$")
    assert labs == ["$0", "$50k", "$100k", "$150k"]


def test_log_ticks():
    major, minor, d0, d1 = chart._log_ticks(100, 1e5)
    assert major == [100, 1000, 10000, 100000] and d0 == 100 and d1 == 1e5 and minor
    major, _, d0, d1 = chart._log_ticks(1.6, 14)
    assert 2 in major and 5 in major and 10 in major and d0 <= 1.6 and d1 >= 14


def test_date_ticks():
    d0, d1 = chart._parse_date("2026-01-01"), chart._parse_date("2026-12-01")
    ticks = chart._date_ticks(d0, d1, 6)
    assert ticks[0][1].startswith("Jan") and "2026" in ticks[0][1]
    assert chart._date_labels(["2025-12", "2026-01"]) == ["Dec\n2025", "Jan\n2026"]


def test_grid_boxes():
    cv = Canvas(1600, 900)
    boxes = chart.grid(cv, 2, 3, box=(0, 0, 1, 1), gap=0.02)
    assert len(boxes) == 6
    gx = (boxes[1][0] - boxes[0][2]) * 1600
    gy = (boxes[3][1] - boxes[0][3]) * 900
    assert abs(gx - gy) < 1e-6
    w = chart.grid(None, 1, 2, widths=(2, 1), gap=0)
    assert abs((w[0][2] - w[0][0]) - 2 * (w[1][2] - w[1][0])) < 1e-9


# ---------------------------------------------------------------- every kind draws

def _texts(item, out):
    if isinstance(item, Text):
        out.append(item)
    elif isinstance(item, Group):
        for c in item.children:
            _texts(c, out)
    return out


def _check_inside(ax):
    """All labels of the chart lie inside its box (no clipping)."""
    dwg = ax.drawing()
    texts = _texts(dwg.root, [])
    assert texts, "chart has no text"
    for t in texts:
        x0, y0, x1, y1 = t.bbox()
        assert x0 >= -1 and y0 >= -1 and x1 <= dwg.width + 1 and y1 <= dwg.height + 1, (t.text, (x0, y0, x1, y1))


KINDS = {
    "line": lambda ax: ax.line(["2026-01", "2026-02", "2026-03", "2026-04"], {"A": [1, 3, 2, 4], "B": [2, 2, 3, 3]},
                               smooth=True, fill=True),
    "area": lambda ax: ax.area([1, 2, 3, 4], {"A": [1, 3, 2, 4], "B": [2, 2, 3, 3]}, stacked=True),
    "bar": lambda ax: ax.bar(["North", "South", "East"], {"2025": [3, 5, 4], "2026": [4, 6, 5]},
                             errors={"2025": [0.3] * 3, "2026": [0.4] * 3}, value_labels=True).signif(
        ("North", "2025"), ("North", "2026"), "**"),
    "stacked": lambda ax: ax.bar(["A", "B"], {"x": [1, 2], "y": [2, 1]}, stacked=True, value_labels=True),
    "barh": lambda ax: ax.barh({"Wireless headphones": 420, "Smart watch": 310, "Desk lamp": 120},
                               value_labels=True, highlight=0, fmt="$k"),
    "scatter": lambda ax: ax.scatter(np.arange(20), np.arange(20) * 2.0, size=np.arange(20), trend=True),
    "groups": lambda ax: ax.scatter([1, 2, 3, 4], [1, 2, 3, 4], color=["a", "b", "a", "b"]),
    "hist": lambda ax: ax.hist(np.random.default_rng(0).normal(0, 1, 200), bins=12),
    "heatmap": lambda ax: ax.heatmap([[1, 2, 3], [4, 5, 6]], ["a", "b", "c"], ["r1", "r2"]),
    "pie": lambda ax: ax.pie({"A": 3, "B": 2, "C": 1}, donut=0.6),
    "radar": lambda ax: ax.radar(["a", "b", "c", "d", "e"], {"S1": [1, 2, 3, 2, 1], "S2": [2, 1, 2, 3, 2]}),
    "spark": lambda ax: ax.sparkline([1, 3, 2, 5, 4]),
    "log": lambda ax: (ax.line([10, 100, 1000, 1e4], [5, 3, 2, 1.5], label="loss"),
                       ax.band([10, 100, 1000, 1e4], [4, 2.5, 1.8, 1.3], [6, 3.5, 2.2, 1.7])),
    "dual": lambda ax: (ax.bar(["Jan", "Feb", "Mar"], [10, 20, 30], label="Rev"),
                        ax.line(["Jan", "Feb", "Mar"], [0.2, 0.3, 0.25], label="Margin", axis="right")),
    "marks": lambda ax: (ax.line(np.arange(30), np.sin(np.arange(30) / 5), label="s", end_label=True),
                         ax.span(5, 10, label="window"), ax.vline(20, label="v2"), ax.hline(0.5, label="goal"),
                         ax.annotate("peak", (8, float(np.sin(8 / 5))))),
    "rotated": lambda ax: ax.bar([f"Category number {i}" for i in range(14)], np.arange(14) + 1.0),
}


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_kind_draws(kind, theme):
    cv = Canvas(480, 270, (0.5, 0.5, 0.5))
    steps = []
    kw = dict(xscale="log", yscale="log") if kind == "log" else {}
    ax = chart.axes(cv, theme=theme, title=f"{kind} chart", **kw)
    KINDS[kind](ax)
    ax.draw(log=steps.append)
    assert steps and steps[0].endswith("background")
    changed = (np.abs(cv.buf - cv.buf[0, 0]).max(-1) > 0.1).mean()
    assert changed > 0.003
    assert np.isfinite(cv.buf).all()
    _check_inside(ax)
    svg = ax.drawing().to_svg()
    assert svg.startswith("<?xml") and "<text" in svg


def test_figure_svg_and_layers(tmp_path):
    cv = Canvas(640, 360)
    fig = chart.figure(cv, theme="dark")
    fig.header("Title", "Sub", right=["chip"])
    a, b = fig.grid(1, 2, box=(0.02, 0.2, 0.98, 0.98))
    fig.kpi(a, "Revenue", 1234567, fmt="$.2k", delta=4.2, spark=[1, 2, 3, 2, 4], icon="cart")
    fig.axes(b, title="Bars", card=True).bar({"a": 1, "b": 2})
    steps = []
    fig.draw(log=steps.append)
    assert "Revenue value" in steps and "Bars legend" not in steps and "Bars bar" in steps
    p = fig.save_svg(str(tmp_path / "fig.svg"))
    txt = Path(p).read_text(encoding="utf-8")
    assert "$1.23M" in txt and "Revenue" in txt


def test_markup_safe_text():
    cv = Canvas(400, 200)
    ax = chart.axes(cv, title="p ** stars **")
    ax.bar({"a": 1, "b": 2})
    ax.draw()
    texts = [t.text for t in _texts(ax.drawing().root, [])]
    assert any(t.replace("​", "") == "p ** stars **" for t in texts)


def test_legend_avoids_data():
    cv = Canvas(800, 450)
    ax = chart.axes(cv, title="Rising")
    x = np.arange(20)
    ax.line(x, x * 1.0, label="a")
    ax.line(x, x * 0.8, label="b")
    ax.draw()
    leg = ax.drawing().find(f"{ax.name} legend")
    x0, y0, x1, y1 = leg.bbox()
    xs = np.linspace(0, 19, 400)
    for k in (1.0, 0.8):
        px, py = ax._sx(xs), ax._sy(xs * k)
        inside = (px > x0 - 2) & (px < x1 + 2) & (py > y0 - 2) & (py < y1 + 2)
        assert not inside.any()


def test_quick_kinds(tmp_path):
    cv = Canvas(960, 540)
    b = chart.grid(cv, 3, 4, gap=0.02)
    cases = [
        ("line", [3, 5, 4, 6], {}), ("line", {"2026-01": 3, "2026-02": 5}, {}),
        ("area", {"A": [1, 2, 3], "B": [2, 2, 2]}, {"x": ["a", "b", "c"], "stacked": True}),
        ("bar", {"A": 3, "B": 5}, {"value_labels": True}), ("barh", [3, 4, 5], {}),
        ("stacked", {"x": [1, 2], "y": [2, 1]}, {"x": ["p", "q"]}),
        ("scatter", [(1, 2), (2, 3), (3, 5)], {}), ("hist", np.arange(50), {"bins": 5}),
        ("heatmap", [[1, 2], [3, 4]], {}), ("donut", {"a": 1, "b": 2}, {}),
        ("radar", {"s": [1, 2, 3]}, {"x": ["a", "b", "c"]}), ("spark", [1, 3, 2, 4], {}),
    ]
    for box, (k, data, kw) in zip(b, cases):
        ax = chart.quick(cv, k, data, title=k, box=box, **kw)
        assert isinstance(ax, chart.Axes)
    ax.save_svg(str(tmp_path / "q.svg"))
    with pytest.raises(ValueError):
        chart.quick(cv, "nope", [1])


def test_standalone_axes():
    ax = chart.axes(theme="light", title="Alone", size=(400, 300))
    ax.line([1, 2, 3], [3, 1, 2], label="x")
    out = ax.render()
    assert (out.w, out.h) == (400, 300)
    fig = chart.figure(size=(500, 250))
    fig.axes(title="t").bar({"a": 1})
    assert fig.render().w == 500


def test_scale_independent_layout():
    """A half-size canvas gives the same layout (units are 1080-relative)."""
    pos = []
    for w, h in ((960, 540), (1920, 1080)):
        cv = Canvas(w, h)
        ax = chart.axes(cv, title="T", yfmt="$k")
        ax.bar(["a", "b", "c"], [1e4, 2e4, 3e4])
        ax.drawing()
        pos.append(ax._plot)
    assert np.allclose(pos[0], pos[1], atol=0.5)


# ---------------------------------------------------------------- doc recipes run as written

def _blocks():
    return re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)


def test_doc_has_recipes():
    assert len(_blocks()) >= 4
    assert len(DOC.read_text(encoding="utf-8").splitlines()) < 250


@pytest.mark.parametrize("i", range(5))
def test_doc_blocks_run(i, tmp_path, monkeypatch):
    blocks = _blocks()
    if i >= len(blocks):
        pytest.skip("no such block")
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    exec(compile(blocks[i], f"charts.md block {i}", "exec"), {"__name__": "__doc__"})


def test_example_scenes_render():
    import importlib
    sys.path.insert(0, str(ROOT))
    for name in ("chart_dashboard", "chart_figure"):
        mod = importlib.import_module(f"scenes.{name}")
        steps = []
        cv = mod.render(480, 270, log=steps.append)
        assert (cv.w, cv.h) == (480, 270) and len(steps) > 8


# ---------------------------------------------------------------- fixes from the fresh-eyes review

def test_blank_canvas_gets_theme_background_everywhere():
    cv = Canvas(320, 180)
    chart.axes(cv, box=(0.1, 0.1, 0.9, 0.9)).bar(["a", "b"], [1, 2])
    ax = chart.axes(cv, box=(0.1, 0.1, 0.9, 0.9))
    ax.bar(["a", "b"], [1, 2])
    ax.draw()
    assert cv.buf[2, 2].min() > 0.5                      # the margin is the light theme colour, not black


def test_stacked_value_labels_include_segments():
    def labels(**kw):
        ax = chart.axes(size=(800, 500))
        ax.bar(["Q1", "Q2"], {"A": [10, 12], "B": [20, 22]}, stacked=True, **kw)
        return re.findall(r"<tspan[^>]*>([^<]*)<", ax.drawing().to_svg())
    on, off = labels(value_labels=True), labels()
    assert {"12", "22", "34"} <= set(on) and "34" not in off      # segments and totals


def test_bar_fmt_formats_value_axis():
    ax = chart.axes(size=(800, 500))
    ax.barh({"a": 4200, "b": 1800}, fmt="k", value_labels=True)
    svg = ax.drawing().to_svg()
    assert "1,000" not in svg and "2k" in svg


def test_trend_label_avoids_points():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 50)
    ax = chart.axes(size=(900, 500))
    ax.scatter(x, x * 2 + rng.normal(0, 0.1, 50), trend=True)
    assert "R²" in ax.drawing().to_svg()


def test_strict_options_chart():
    import pytest
    from pixelcanvas import chart as C
    ax = C.axes(None, theme="light")
    with pytest.raises(TypeError, match="did you mean 'smooth'"):
        ax.line([1, 2, 3], smoth=True)
    ax.line([1, 2, 3], colour="#c33", w=2)               # aliases of color / width
    with pytest.raises(ValueError, match="'dark'"):
        C.axes(None, theme="dak")
    with pytest.raises(TypeError, match="did you mean 'ylabel'"):
        C.axes(None, ylable="x")
    with pytest.raises(ValueError, match="'line'"):
        C.quick(None, "lin", [1, 2, 3])


# ---- finished defaults and what the report sees ----------------------------------------------------------------------

MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
          "2026-08", "2026-09", "2026-10"]


def test_default_bar_has_value_labels_and_single_line_is_filled():
    import pytest
    ax = chart.axes(None)
    ax.bar({"A": 3, "B": 5, "C": 4})
    assert ax.series[-1]["value_labels"] is True
    two = chart.axes(None).bar(["a", "b"], {"x": [1, 2], "y": [2, 3]})
    assert two.series[-1]["value_labels"] is False
    with pytest.raises(ValueError, match="'auto'"):
        chart.axes(None).bar({"A": 1}, value_labels="auot")
    svg_one = chart.axes(None).line(MONTHS, list(range(12))).drawing().to_svg()
    svg_two = chart.axes(None).line(MONTHS, {"a": list(range(12)), "b": list(range(12))}).drawing().to_svg()
    assert "linearGradient" in svg_one and "linearGradient" not in svg_two


def test_chart_parts_record_types_counts_and_markers():
    from pixelcanvas import manifest
    cv = Canvas(1600, 900)
    fig = chart.figure(cv)
    for box, (name, v, fmt) in zip(fig.grid(1, 4, box=(0.02, 0.02, 0.98, 0.25)),
                                   [("MRR", 128400, "$.1k"), ("Users", 48210, None), ("Churn", 2.8, "%"),
                                    ("NPS", 54, None)]):
        fig.kpi(box, name, v, fmt=fmt, delta=-0.4 if name == "Churn" else 3.0, good="down" if name == "Churn" else "up")
    a, b = fig.grid(1, 2, box=(0.02, 0.3, 0.98, 0.98))
    ax = fig.axes(a, title="Revenue, last 12 months")
    ax.line(MONTHS, list(range(70, 130, 5)), label="MRR")
    ax.vline("2026-04", label="Launch")
    fig.axes(b, title="Plan mix").donut({"Free": 52, "Pro": 31, "Team": 12, "Enterprise": 5})
    fig.draw()
    by = {e["name"]: e for e in cv.manifest}
    assert by["month"]["meta"]["count"] == 12 and by["slice"]["meta"]["count"] == 4
    assert "donut" in by["Plan mix"]["aliases"] and "kpi tile" in by["Churn"]["aliases"]
    assert "marker" in by["Launch"]["aliases"]
    prompt = ("four KPI tiles, a revenue line chart for the last 12 months with a marked launch date and a donut of "
              "plan mix")
    issues = manifest.check(cv, prompt)
    assert not [w for w in issues if w["rule"] in ("prompt_count", "object_hidden")], issues
    assert all(c["status"] == "ok" for c in manifest.coverage(cv, prompt))


def test_kpi_percent_and_good_down():
    ax_fig = chart.figure(None, size=(400, 200))
    k = ax_fig.kpi((0, 0, 1, 1), "Churn", 2.8, fmt="%", delta=-0.4, good="down")
    svg = k.drawing().to_svg()
    assert "2.8%" in svg
    good = chart.figure(None).theme.accent(2)
    assert good.lower() in svg.lower()                     # falling churn is drawn in the "good" colour
