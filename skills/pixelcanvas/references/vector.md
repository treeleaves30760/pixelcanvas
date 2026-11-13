# Vector drawing (`pixelcanvas.vector`)

One `Drawing` holds shapes, text, icons and groups. It rasterises onto a Canvas (anti-aliased, crisp 1 px lines,
one layer per top-level group) **and** exports the same geometry as a standalone SVG 1.1 file. Reach for it for
diagrams, charts, UI mock-ups, icons, technical drawings and anything made of lines, boxes and labels.
For characters, animals and painterly illustration (soft shading, rim light, washes, stickers) draw on the canvas
with `illustrate.Sketch` instead: [illustration.md](illustration.md).

Options are checked: an unknown option raises a `TypeError` that names the closest valid one, and both spellings are accepted (`color` / `colour`, `lw` / `stroke_width`, `alpha` / `opacity`).

```python
from pixelcanvas.vector import Drawing
dwg = Drawing(800, 450, theme="light")                       # user units = px here
dwg.rect(40, 40, 320, 180, r=14, fill=dwg.theme.surface, stroke=dwg.theme.grid)
dwg.text(200, 130, "Hello vectors", size=32, anchor="mm", weight="bold")
dwg.line(380, 130, 600, 130, stroke=dwg.theme.accents[0], stroke_width=3, arrow="end")
cv = dwg.render()                                            # Canvas 800x450
dwg.save_svg("outputs/hello.svg")                            # same picture as SVG
```

In a scene: `dwg.draw(cv, log=log)` scales the drawing to fit the canvas (or `box=(x0, y0, x1, y1)`), so design
in 1920x1080 units and `--scale 0.5` drafts look identical. Each top-level `group` is one `log(name)` step / layer.

## Calls

| Call | What it draws |
|---|---|
| `Drawing(w, h, units="px", px_per_unit=None, background=None, theme=None, crisp=True)` | units px/mm/cm/in/pt; `px_per_unit` = raster density of `render()` (default 96 dpi); `background=None` → theme background, `False` → transparent overlay |
| `line(x1, y1, x2, y2)` · `polyline(pts)` · `polygon(pts)` | straight geometry |
| `rect(x, y, w, h, r=0)` · `circle(cx, cy, r)` · `ellipse(cx, cy, rx, ry)` | `r` = radius, `(rx, ry)`, or per corner `(tl, tr, br, bl)` |
| `arc(cx, cy, r, start, end)` · `pie(cx, cy, r, start, end, inner=0)` | degrees clockwise from +x; `inner` → donut slice |
| `path("M10 10 C ...")` or `path().move_to().line_to().quad_to().cubic_to().arc_to().close()` | full SVG path syntax |
| `bezier(points)` · `spline(points, closed=False)` | Bezier from control points · smooth curve *through* points |
| `text(x, y, "label", size, anchor="lt", weight, family, style, rotate, letter_spacing, max_width, halo, markup=False)` | anchor `lt mm rb ls mc ...` (`c` = middle of the capital height); `"\n"` lines; CJK automatic; plain text unless `markup=True` |
| `icon(name, x, y, size=24, color, stroke_width=2, fill=None, anchor="mm")` | 48 built-in icons, `icon_names()` |
| `with group(name, translate, rotate, scale, origin, opacity, clip, shadow):` | nested transforms; one layer per top-level group |
| `embed(other_drawing, x, y, w=None)` · `embed(other, box=(x0, y0, x1, y1), align="center")` | copy another Drawing in (e.g. `vector.icon("gear")`), at a width or fitted into a box |
| `dwg.measure_text(text, size, ...)` → `(w, h)` · `item.bbox()` · `item.place(ref, "right", gap)` · `item.move(dx, dy)` | layout helpers (drawing units) |
| `dwg.draw(cv, log=None, box=None)` · `dwg.render(scale=1)` · `dwg.to_svg()` · `dwg.save_svg(path)` | output |
| `last_drawings()` · `clear_drawings()` | drawings painted during this render (for SVG export of any scene) |

**Style keywords** (every shape): `fill` (colour / `linear_gradient` / `radial_gradient` / None), `stroke`,
`stroke_width`, `dash` (`"dash" "longdash" "dot" "dashdot" "center" "hidden"` or `(on, off, ...)` in units),
`dash_offset`, `dash_fit` (True: an open line starts and ends on a full dash, a closed one holds whole
periods; `"center"` for centre lines), `cap` (butt/round/square), `join` (miter/round/bevel), `fill_opacity`, `stroke_opacity`,
`opacity`, `hatch` (`hatch(angle, spacing, width, color, cross)`), `fill_rule` (nonzero/evenodd),
`arrow` ("end"/"start"/"both"), `marker_start` / `marker_end` (`arrow triangle open dot bar diamond none`), `marker_size`,
`name` (SVG id), `add=False` (build without adding, e.g. for a clip shape).
Colours: `"#rrggbb"`, `"#rrggbbaa"`, `(r, g, b)` 0~1, `"none"`. Closed shapes with no style get the theme ink fill;
open shapes get an ink stroke. Markers grow with the stroke width.

**THEMES** `light dark blueprint paper pastel neon mono` — each has `background surface ink muted grid`,
`accents` (8 colours, `theme.accent(i)` wraps), `stroke_width`, `radius`, `shadow`, `font`. Pass `theme="dark"`
to a Drawing and its defaults (ink, stroke, font, background, `shadow=True`) follow it.

## App icon tiles

One call per tile, one call per sheet: rounded square, two-colour diagonal gradient, white glyph, shadow, label.

```python
from pixelcanvas import Canvas
from pixelcanvas.vector import app_icon, app_icon_sheet
cv = Canvas(1600, 1200)
app_icon_sheet(cv, ["notes", "calendar", "mail", "tasks", "files", "chat", "camera", "music", "maps", "weather",
                    "settings", "calculator"], cols=4, shadow="long")
one = Canvas(400, 400, "#eef0f4")
app_icon(one, (100, 80, 300, 280), "weather", colors=("#7fd0ff", "#2b86d8"), shadow="soft", label="Weather")
# dark glyphs with a cream second colour for the details (calendar header, sun, clock hands, lens ...)
two = Canvas(1200, 700)
app_icon_sheet(two, ["calendar", "weather", "clock", "camera", "tasks", "chat"], cols=3,
               glyph_color="#1b2333", accent_color="#fff4c2")
```

| Call | What it does |
|---|---|
| `app_icon(cv, box, glyph, colors=None, shadow="long", label=None, radius=0.23, glyph_color="#ffffff", accent_color=None, glyph_size=0.56, cell=None)` | one tile in `box` (pixels or fractions, kept square); `glyph_color` paints the glyph, `accent_color` a second colour for its details; `colors` = (light, deep) or one colour; returns the tile box |
| `app_icon_sheet(cv, apps, cols=None, box=None, background=True, shadow="long", tile=0.6, labels=True, glyph_color="#ffffff", accent_color=None)` | a labelled grid (`glyph_color` / `accent_color` as above); `apps` = app words, `(label, glyph)` or `(label, glyph, light, deep)`; colours from `APP_COLORS` |

**Glyphs**: any pictogram (`pictograms.names()`, drawn as a silhouette), any line icon (`icon_names()`) or an app
word from `APP_GLYPHS` — notes, calendar, mail, tasks, files, chat, camera, music, maps, weather (sun behind a
cloud), settings, calculator, clock, contacts, stats, store, health ... **Shadows**: `long` = the glyph's 45° shadow
across the tile plus a fading tile shadow, `soft` = a drop shadow, `none`; every shadow stays inside the tile's
cell (`cell=` for a single tile, the grid cell on a sheet). Each tile records itself (alias "app icon" / "icon"),
so "twelve app icons" is counted by `--check`.

## Recipes

**Labelled boxes and arrows (diagram building blocks)** — measure, then place relative to other items:

```python
from pixelcanvas.vector import Drawing
dwg = Drawing(900, 300, theme="dark")
t = dwg.theme
def node(x, y, label, icon_name, k):
    w = dwg.measure_text(label, size=20, weight="bold")[0] + 96
    with dwg.group(label, shadow=True) as g:
        dwg.rect(x, y, w, 72, r=t.radius, fill=t.surface, stroke=t.accent(k), stroke_width=2)
        dwg.icon(icon_name, x + 36, y + 36, 32, color=t.accent(k))
        dwg.text(x + 64, y + 36, label, size=20, weight="bold", anchor="lm")
    return g
a = node(40, 110, "Client", "user", 0)
b = node(0, 110, "API server", "server", 1).place(a, "right", gap=120)
c = node(0, 110, "Database", "database", 2).place(b, "right", gap=120)
for p, q in ((a, b), (b, c)):
    (x0, y0, x1, y1), (u0, v0, u1, v1) = p.bbox(), q.bbox()
    dwg.line(x1 + 8, (y0 + y1) / 2, u0 - 8, (v0 + v1) / 2, stroke=t.muted, stroke_width=2, arrow="end")
dwg.text(450, 40, "Request flow", size=26, anchor="mt", weight="bold")
cv = dwg.render()
```

**Chart-style fills: gradients, hatches, dashes**

```python
from pixelcanvas.vector import Drawing, linear_gradient, hatch
dwg = Drawing(600, 320, theme="paper")
t = dwg.theme
for i, v in enumerate([120, 200, 160, 240]):
    dwg.rect(60 + i * 120, 280 - v, 80, v, r=(6, 6, 0, 0),           # rounded on top only
             fill=linear_gradient([(0, t.accent(i)), (1, t.surface)]), stroke=t.accent(i))
dwg.rect(60, 60, 80, 40, fill=t.surface, hatch=hatch(45, spacing=6), stroke=t.ink)
dwg.line(40, 280, 560, 280, stroke=t.ink, stroke_width=1)
dwg.line(40, 120, 560, 120, stroke=t.muted, dash="dash")
dwg.spline([(60, 250), (180, 150), (300, 190), (420, 90), (540, 130)], stroke=t.accent(4), stroke_width=3,
           cap="round", marker_end="dot")
cv = dwg.render(scale=2)                                     # 1200x640 raster, same geometry
```

**Millimetre sheet (engineering drawing) + SVG export**

```python
from pixelcanvas.vector import Drawing, hatch
dwg = Drawing(210, 148, units="mm", px_per_unit=6, theme="blueprint")   # A5 landscape, 6 px per mm
t = dwg.theme
dwg.rect(8, 8, 194, 132, fill=None, stroke=t.ink, stroke_width=0.5)    # border
with dwg.group("part", translate=(105, 74)):
    dwg.rect(-50, -25, 100, 50, r=4, fill=t.surface, stroke=t.ink, stroke_width=0.5)
    dwg.circle(0, 0, 12, fill=t.background, hatch=hatch(45, spacing=1.5, width=0.18), stroke=t.ink,
               stroke_width=0.5)
    dwg.line(-60, 0, 60, 0, stroke=t.muted, stroke_width=0.25, dash="center", dash_fit="center")
    dwg.line(-50, 36, 50, 36, stroke=t.ink, stroke_width=0.25, marker_start="triangle", marker_end="triangle",
             marker_size=3)
    dwg.text(0, 34.5, "100", size=4, anchor="mb", halo=t.background)
    dwg.circle(62, -30, 4, fill=t.background, stroke=t.ink, stroke_width=0.35)   # balloon, number centred
    dwg.text(62, -30, "1", size=4.5, anchor="mc")
dwg.text(196, 134, "SCALE 1:1  ·  mm", size=3.5, anchor="rb", family="mono")
cv = dwg.render()                                            # 1260x888 px
dwg.save_svg("outputs/part.svg")                             # width="210mm" height="148mm"
```

**Inside a scene file** — one Drawing, painted through the layer recorder:

```python
from pixelcanvas import Canvas
from pixelcanvas.vector import Drawing

def render(width=1920, height=1080, seed=1, log=None):
    dwg = Drawing(1920, 1080, theme="light")
    with dwg.group("title"):
        dwg.text(960, 120, "Quarterly review", size=64, anchor="mm", weight="bold")
    with dwg.group("icons"):
        for i, name in enumerate(["cloud", "server", "database", "shield"]):
            dwg.icon(name, 480 + i * 320, 540, 160, color=dwg.theme.accent(i), fill=dwg.theme.surface)
    cv = Canvas(width, height)
    dwg.draw(cv, log=log)                                    # → layers "background", "title", "icons"
    return cv
cv = render(960, 540)
```

**Overlay on an existing picture** — a transparent Drawing painted into a box of the canvas:

```python
from pixelcanvas import Canvas
from pixelcanvas.vector import Drawing, icon
cv = Canvas(960, 540, "#20304a")
hud = Drawing(400, 120, theme="dark", background=False)       # no background: the canvas shows through
hud.rect(0, 0, 400, 120, r=(18, 18, 18, 0), fill="#10151ccc")
hud.embed(icon("bolt", color="#ffd36e"), box=(16, 16, 104, 104))
hud.text(120, 60, "98% charged", size=30, anchor="lc", weight="bold")      # 'c': capitals centred on y
hud.draw(cv, box=(520, 400, 920, 520))
```

Tips: put 1 px lines anywhere — `crisp=True` snaps axis-aligned strokes and rectangles (rounded ones too) to the
pixel grid.
Clip a group with `clip=(x0, y0, x1, y1)` or `clip=dwg.circle(cx, cy, r, add=False)`. Text with `halo=` stays
readable over lines and grids. Icons: `alert bell bolt bot bulb calendar camera cart chart check chip clock cloud
code cube database download file folder gear globe heart home image info key layers link lock mail monitor
network phone pin play refresh search server shield sparkles star tag terminal trend upload user users wifi`.
