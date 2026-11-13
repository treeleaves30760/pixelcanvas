# -*- coding: utf-8 -*-
"""
Paper-style two-panel figure (pixelcanvas.chart, light theme).

* Format: 1920x1080 on a white page; one chart Figure in 1080-tall units (drafts look identical).
* Panel a: grouped bars (two time points per treatment) with s.e.m. error bars and significance brackets that
  stack automatically, plus a dashed control line. Panel b: log-log line plot of validation loss for three model
  sizes, each with a shaded 95% band, and a target line.
* Style: left + bottom axis lines with outward ticks (frame="axes"), soft horizontal grid, bold panel letters,
  a wrapped caption under both panels.
* Each panel element is its own layer; fig.save_svg() writes the whole figure as one SVG.
"""
import numpy as np

from pixelcanvas import Canvas
from pixelcanvas import chart

NAME = "chart_figure"
DEFAULT_SIZE = (1920, 1080)
DEFAULT_SEED = 3

THEME = "light"
FONT_SCALE = 1.25         # text and line sizes of both panels
CAPTION = ("Figure 3.  (a) Cell viability after 24 h and 48 h of treatment, as a percentage of the untreated "
           "control; bars show the mean ± s.e.m. of n = 6 wells.  * p < 0.05, ** p < 0.01, *** p < 0.001, "
           "two-sided Welch t-test.  (b) Validation loss during training for three model sizes (log-log axes); "
           "lines are the mean of 5 seeds and shaded bands the 95% confidence interval.")

TREATMENTS = ["Vehicle", "Drug A", "Drug B", "A + B"]
VIAB = {"24 h": [100.0, 82.4, 76.9, 58.3], "48 h": [98.6, 64.1, 61.7, 31.8]}
SEM = {"24 h": [3.1, 4.2, 3.8, 3.5], "48 h": [2.9, 3.9, 4.4, 2.7]}


def _loss_curves(seed):
    rng = np.random.default_rng(seed)
    steps = np.unique(np.round(np.logspace(2, 5, 28)))
    out = {}
    for name, a, floor in (("Small · 40M", 1.0, 2.35), ("Base · 160M", 0.86, 2.05), ("Large · 640M", 0.75, 1.82)):
        y = floor + 9.0 * a * (steps / 100.0) ** -0.42
        y *= 1 + rng.normal(0, 0.008, len(steps))
        w = 0.06 + 0.10 * (steps / 100.0) ** -0.25
        out[name] = (y, y * (1 - w * 0.5), y * (1 + w * 0.5))
    return steps, out


def build(cv, seed=DEFAULT_SEED):
    fig = chart.figure(cv, theme=THEME, background="surface")
    a_box, b_box = fig.grid(1, 2, box=(0.03, 0.035, 0.97, 0.835), gap=0.06, widths=(1, 1.12))

    ax = fig.axes(a_box, panel="a", title="Combination treatment lowers viability",
                  ylabel="Viability (% of control)", frame="axes", ylim=(0, 140), yticks=[0, 25, 50, 75, 100, 125],
                  legend="tr", font_scale=FONT_SCALE)
    ax.bar(TREATMENTS, VIAB, errors=SEM, width=0.7, radius=2)
    ax.hline(100, label="control", color="#9aa3af")
    ax.signif(("Drug A", "24 h"), ("Drug A", "48 h"), "**")
    ax.signif(("Drug B", "24 h"), ("Drug B", "48 h"), "*")
    ax.signif(("A + B", "24 h"), ("A + B", "48 h"), "***")
    ax.signif("Vehicle", "A + B", "***")

    steps, curves = _loss_curves(seed)
    bx = fig.axes(b_box, panel="b", title="Larger models reach lower loss", xlabel="Training step",
                  ylabel="Validation loss", xscale="log", yscale="log", frame="axes", legend="tr", ylim=(1.6, 14),
                  font_scale=FONT_SCALE)
    for name, (y, lo, hi) in curves.items():
        bx.line(steps, y, label=name, width=2.6, markers=False)
        bx.band(steps, lo, hi, opacity=0.18)
    bx.hline(2.0, label="target loss 2.0", color="#9aa3af")

    fig.caption(CAPTION, box=(0.03, 0.875, 0.97, 0.98), size=19, color=fig.theme.ink)
    return fig


def render(width=1920, height=1080, seed=DEFAULT_SEED, log=None):
    log = log or (lambda m: None)
    cv = Canvas(width, height)
    build(cv, seed).draw(log=log)
    return cv
