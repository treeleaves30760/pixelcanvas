# Charts (`pixelcanvas.chart`)

Publication- and dashboard-quality charts in a few calls: line, area, bar (grouped, stacked, horizontal), scatter,
histogram, heatmap, pie / donut, radar, sparkline and KPI tiles. Charts are drawn with `pixelcanvas.vector` onto
the same Canvas as everything else — so they sit inside posters, take glows and textures, export as editable layers
(one per series, legend, callout ...) — and save as SVG.

Options are checked: an unknown option or kind raises an error naming the closest valid one, and both spellings are accepted (`color` / `colour`, `w` / `width`, `colors` / `color`).

```python
from pixelcanvas import Canvas
from pixelcanvas import chart
cv = Canvas(1600, 900)
chart.quick(cv, "bar", {"Mon": 12, "Tue": 18, "Wed": 9, "Thu": 22, "Fri": 15}, title="Orders per day")
```

That is a finished chart: title, rounded bars with value labels, a calm grid and theme colours. A single line gets
a soft gradient fill and point markers; a donut gets percentages, a legend with values and the total in the hole:

```python
from pixelcanvas import Canvas
from pixelcanvas import chart
months = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
          "2026-08", "2026-09", "2026-10"]
cv = Canvas(1200, 750)
chart.quick(cv, "line", dict(zip(months, [71, 74, 78, 80, 88, 93, 101, 108, 112, 118, 124, 128])), title="Revenue")
cv = Canvas(1200, 750)
chart.quick(cv, "donut", {"Free": 52, "Pro": 31, "Team": 12, "Enterprise": 5}, title="Plan mix")
```

Layout, tick choice, number formats, label wrapping / rotation, margins and legend placement are automatic.
Boxes are `(u0, v0, u1, v1)` fractions of the canvas; sizes are in units of a 1080 px tall canvas, so a
`--scale 0.5` draft looks exactly like the full render.

## Calls

| Call | What it does |
|---|---|
| `chart.quick(cv, kind, data, title=, box=, theme=, log=, x=, **kw)` | one call → finished chart; kinds `line area bar barh stacked scatter hist heatmap pie donut radar spark` |
| `ax = chart.axes(cv, box=None, theme="light", title=, subtitle=, xlabel=, ylabel=, ...)` → `ax.draw(log=log)` | one chart in a box; `ax.save_svg(path)`; without `cv`: `size=(w, h)`, `ax.render()` |
| `fig = chart.figure(cv, theme="dark", background=True)` → `fig.axes(box, ...)` · `fig.kpi(...)` · `fig.header(...)` · `fig.caption(text, box)` → `fig.draw(log=log)` | many parts, one SVG: `fig.save_svg(path)`; `fig.build()` → the vector Drawing |
| `chart.grid(cv, rows, cols, box=(0,0,1,1), gap=0.02, widths=None, heights=None)` · `fig.grid(...)` | boxes for a panel layout (row-major); `widths=(2, 1)` for unequal columns |
| `ax.line(x, y \| {label: ys}, label=, smooth=False, markers=None, fill="auto", dash=, width=, axis="left", end_label=False)` | x: numbers, date strings `"2026-03"` / `"2026-03-14"`, or categories; `smooth` = monotone curve; `fill="auto"` fills a lone line |
| `ax.area(x, y \| {label: ys}, stacked=False, smooth=False)` | gradient fill under the line |
| `ax.bar(cats, values \| {series: values}, stacked=False, horizontal=False, errors=None, value_labels="auto", fmt=, highlight=, radius=)` · `ax.barh(...)` | `cats` may be a dict `{cat: value}`; errors ± value or (minus, plus); rounded bar ends; `value_labels="auto"` labels one series of up to 16 bars; stacked + `value_labels=True` prints every segment and the total; `fmt` also formats the value axis |
| `ax.scatter(x, y, size=None, color=None, trend=False)` | `color`: one colour, a list of group names (legend) or numbers (colour ramp); `trend` → fit + R² |
| `ax.hist(values, bins=20, density=False)` | |
| `ax.heatmap(matrix, xlabels, ylabels, annotate=True, fmt=, cmap=0, colorbar=None)` | `cmap`: accent index, colour, or list of colours |
| `ax.pie(values \| {label: v}, labels, donut=0.0, center=None)` · `ax.donut(...)` | `center` = text in the hole (`(big, small)`; default the total) |
| `ax.radar(categories, values \| {series: values}, max=None, levels=4)` | spider chart |
| `ax.sparkline(values, color=, fill=True, dot=True, band=(lo, hi))` · `fig.sparkline(box, values)` | bare line for small boxes |
| `ax.band(x, lo, hi, opacity=0.16)` · `ax.span(a, b, axis="x", label=)` · `ax.hline(y, label=)` · `ax.vline(x, label=)` | confidence bands, shaded ranges, reference lines |
| `ax.annotate(text, xy, xytext=None, offset=None)` | callout pill + leader; placed where it covers no data unless given |
| `ax.signif(a, b, text="*")` | significance bracket between categories or `(category, series)` bars; stacks itself |
| `fig.kpi(box, label, value, delta=, spark=, fmt=, note=, icon=, color=, good="up")` · `chart.kpi(cv, box, ...)` | KPI tile: big number, ▲/▼ delta pill, sparkline |
| `fig.header(title, subtitle=, right=[chips], icon=, box=)` · `chart.header(cv, ...)` | dashboard header band |
| `chart.fmt_number(v, fmt)` · `chart.nice_ticks(lo, hi, n)` | the formatter and tick picker the axes use |

**Axes options**: `xscale` auto / linear / log / date / category · `yscale` linear / log · `xfmt` `yfmt`
`yfmt_right` · `xlim` `ylim` `ylim_right` · `xticks` `yticks` · `grid` auto / "x" / "y" / "both" / False ·
`frame` "minimal" (baseline only, dashboards) / "axes" (left + bottom axis with ticks, papers) / "box" / "none" ·
`legend` auto / top / right / bottom / none / tl / tr / bl / br · `card=True` (rounded surface card with shadow) ·
`panel="a"` (panel letter) · `font_scale` (default grows with the box) · `palette` · `ylabel_right` + series `axis="right"` for a dual y axis.

**Number formats** (`fmt`, `xfmt`, `yfmt`): `"$"` → $1,250 · `"$k"` → $1.3k · `"$.2k"` → $1.28M · `"k"` →
12.4k · `"%"` → 12.5% (value already in percent) · `"+.1%"` → +2.0% · `" ms"` → 12 ms · `"{:.0%}"` (Python
format, multiplies by 100) · any callable. Default: thousands separators, k / M / B on large axes.

**KPI tiles**: `fmt="%"` shows the value as a percent (2.8 → 2.8%); `good="down"` colours a falling value green
(churn, cost, latency); `delta` is printed with `delta_fmt="+.1%"` (percent points as given).

```python
from pixelcanvas import Canvas
from pixelcanvas import chart
cv = Canvas(1600, 400)
fig = chart.figure(cv, theme="light")
for box, (name, value, delta, fmt, good) in zip(fig.grid(1, 3, box=(0.02, 0.08, 0.98, 0.92)), [
        ("MRR", 128400, 12.4, "$.1k", "up"), ("Churn", 2.8, -0.4, "%", "down"), ("NPS", 54, 6, None, "up")]):
    fig.kpi(box, name, value, delta=delta, fmt=fmt, good=good, note="vs last month",
            spark=[3.6, 3.4, 3.2, 3.1, 2.9, 2.8] if name == "Churn" else [3, 4, 4, 5, 6, 7])
fig.draw()
```

**What `--check` sees**: every chart part records itself — KPI tiles answer to "KPI tile", axes to their type
("line chart", "bar chart", "donut"), each chart adds one data-point entry with a count (twelve `"2026-03"`-style
x values → `month` ×12; days, years, quarters, `bar` ×N, `slice` ×N) and every `vline` / `hline` its label
("Launch" answers to "marker" / "date"), so "a line chart of the last 12 months with a marked launch date" is
counted and covered without `manifest.note`.

**Themes**: any `vector.THEMES` name — `light dark paper blueprint pastel neon mono`. Series take the theme accents
in `chart.PALETTE_ORDER` (blue, orange, green, purple, cyan, pink, olive, red); KPI deltas use green / red.

## Recipes

**Dashboard: header, KPI row, charts in cards**

```python
from pixelcanvas import Canvas
from pixelcanvas import chart
cv = Canvas(1920, 1080)
fig = chart.figure(cv, theme="dark")
fig.header("Store performance", "Q3 2026 · all regions", right=["Jul 1 – Sep 30"], icon="chart")
for box, (name, value, delta, fmt) in zip(fig.grid(1, 3, box=(0.02, 0.14, 0.98, 0.30)), [
        ("Revenue", 1284600, 12.4, "$.2k"), ("Orders", 18420, 8.1, None), ("Refunds", 312, -4.2, None)]):
    fig.kpi(box, name, value, delta=delta, fmt=fmt, note="vs Q2", spark=[3, 4, 3.6, 5, 4.8, 6, 6.4])
left, right = fig.grid(1, 2, box=(0.02, 0.33, 0.98, 0.97), widths=(2, 1))
ax = fig.axes(left, title="Weekly revenue", yfmt="$k", card=True)
weeks = ["2026-07-06", "2026-07-13", "2026-07-20", "2026-07-27", "2026-08-03", "2026-08-10"]
ax.line(weeks, [84e3, 88e3, 86e3, 92e3, 97e3, 129e3], label="2026", smooth=True, fill=True)
ax.line(weeks, [76e3, 79e3, 81e3, 79e3, 84e3, 92e3], label="2025", smooth=True, dash="dash")
ax.annotate("Summer sale", (weeks[5], 129e3))
ax.vline(weeks[3], label="Launch")      # a marked date: dashed vertical line with its label at the top
fig.axes(right, title="Sales by channel", card=True).donut({"Search": 34, "Direct": 24, "Ads": 18, "Email": 14})
fig.draw()                               # in a scene: fig.draw(log=log) → one layer per chart object
fig.save_svg("outputs/dashboard.svg")
```

**Paper figure: grouped bars with error bars and significance, log-scale lines with bands**

```python
import numpy as np
from pixelcanvas import Canvas
from pixelcanvas import chart
cv = Canvas(1800, 900)
fig = chart.figure(cv, theme="light", background="surface")
a, b = fig.grid(1, 2, box=(0.03, 0.04, 0.97, 0.86), gap=0.06)
ax = fig.axes(a, panel="a", title="Viability", ylabel="% of control", frame="axes", legend="tr")
ax.bar(["Vehicle", "Drug A", "A + B"], {"24 h": [100, 82, 58], "48 h": [99, 64, 32]},
       errors={"24 h": [3, 4, 3.5], "48 h": [3, 4, 2.7]}, radius=2)
ax.signif(("A + B", "24 h"), ("A + B", "48 h"), "***")
ax.signif("Vehicle", "A + B", "**")
steps = np.logspace(2, 5, 20)
loss = 2 + 9 * (steps / 100) ** -0.4
bx = fig.axes(b, panel="b", title="Validation loss", xscale="log", yscale="log", frame="axes",
              xlabel="Training step")
bx.line(steps, loss, label="Base model", markers=False)
bx.band(steps, loss * 0.95, loss * 1.05)
fig.caption("Figure 1. (a) Mean ± s.e.m., n = 6. (b) Shaded: 95% interval.", box=(0.03, 0.9, 0.97, 0.98),
            size=18)
fig.draw()
```

**Dual axis, stacked bars, wrapped and rotated labels — all automatic**

```python
from pixelcanvas import Canvas
from pixelcanvas import chart
cv = Canvas(1600, 900, chart.THEMES["light"].background)
left, right = chart.grid(cv, 1, 2, box=(0.03, 0.05, 0.97, 0.95), gap=0.05)
ax = chart.axes(cv, left, title="Revenue and margin", yfmt="$k", yfmt_right="%", ylabel_right="Margin")
months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
ax.bar(months, [52e3, 58e3, 61e3, 66e3, 72e3, 80e3], label="Revenue")
ax.line(months, [21, 22.5, 23, 22, 24.5, 26], label="Margin", axis="right")
ax.draw()
bx = chart.axes(cv, right, title="Cost by department")
bx.bar(["Customer support", "Research and development", "Marketing and sales", "Operations"],
       {"Salaries": [40, 80, 55, 35], "Tools": [10, 25, 15, 12]}, stacked=True, value_labels=True)
bx.draw()
bx.save_svg("outputs/costs.svg")         # this chart alone as SVG
```

**One-liners for every kind** (`quick` takes plain lists and dicts)

```python
import numpy as np
from pixelcanvas import Canvas
from pixelcanvas import chart
cv = Canvas(1800, 1000, chart.THEMES["light"].background)
b = chart.grid(cv, 2, 3, box=(0.02, 0.03, 0.98, 0.97), gap=0.03)
chart.quick(cv, "line", {"North": [3, 5, 4, 6], "South": [2, 3, 5, 4]}, x=["Q1", "Q2", "Q3", "Q4"], box=b[0],
            title="Sales", smooth=True)
chart.quick(cv, "barh", {"Alpha": 42, "Beta": 31, "Gamma": 18}, box=b[1], title="Votes", value_labels=True)
chart.quick(cv, "scatter", {"x": np.arange(30), "y": np.arange(30) * 1.5 + np.sin(np.arange(30)) * 4},
            box=b[2], title="Fit", trend=True)
chart.quick(cv, "hist", np.random.default_rng(1).normal(0, 1, 500), box=b[3], title="Errors", bins=24)
chart.quick(cv, "heatmap", [[1, 4, 2], [3, 1, 5]], x=["A", "B", "C"], y=["Row 1", "Row 2"], box=b[4],
            title="Matrix")
chart.quick(cv, "radar", {"Model": [4, 3, 5, 2, 4]}, x=["Speed", "Cost", "Quality", "Docs", "Support"], box=b[5],
            title="Profile")
```

Tips: a box that is still blank gets the theme background; over a picture pass `background=False` (or `card=True`
for a floating card) and post-process the canvas as usual (`cv.bloom()`, textures). `theme="paper"` gives serif charts,
`"neon"` glowing dashboards. Use one `chart.figure` per picture to get a single SVG of everything; the
underlying vector Drawings are also in `vector.last_drawings()` after a draw.
