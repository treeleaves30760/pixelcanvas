# Pictograms (`pixelcanvas.pictograms`)

The long tail of nameable objects in one call: saxophone, lighthouse, tram, torii gate, ferris wheel, eggplant,
checklist, wizard tower, mitochondrion and over six hundred more, drawn as clean wayfinding-style pictograms that read
at 48 px and hold up at 600 px. Use them
for posters, infographics, icons, stickers, maps, slides and simple illustrations. Every pictogram is also a mask,
so it can be textured, shaded into a volume, outlined, or set as a silhouette against a sunset.
Options are checked: an unknown option name raises a TypeError that names the closest valid one, and both
spellings are accepted (`colour` / `color`, `position` / `at`). An unknown pictogram name raises a ValueError with
the closest names and what `find()` suggests.

```python
from pixelcanvas import Canvas
from pixelcanvas.pictograms import pictogram, find

cv = Canvas(960, 540, "#f4efe6")
print(find("sax"), find("燈塔"))                       # ['saxophone'] ['lighthouse']
p = pictogram(cv, "saxophone", at=(0.5, 0.52), size=0.7, color="#1d2633", style="duotone")
cv.text("Blue Hour", (p.anchors["right"][0] + 20, p.anchors["center"][1]), 64, "#1d2633", anchor="lm")
```

`at` is the anchor point in canvas fractions. `size` is a fraction of the canvas height: the height of the
pictogram's 100-unit box, so `size=0.3` on a 540 px canvas draws a 162 px pictogram. The result `p` holds `p.mask`
(an array attribute, full frame, float 0~1: `p.mask.sum()`, `cv.paint(color, p.mask)`), `p.bbox` (pixels),
`p.anchors` (pixel points: `center`, `top`, `bottom`, `left`, `right` plus named ones such as `bell`, `door`, `light`)
and `p.uv(key)`. Each call records itself in the scene manifest under the pictogram's name (with its aliases), so
`render.py --check` counts and locates it and no extra `manifest.note` is needed.

`fit="box"` sizes a pictogram to a box instead: `size=(w, h)` is the box in canvas fractions (one number is a square
of that fraction of the canvas height), the shape is scaled to fill it and centred on `at`. Five pictograms in a row of
cells, each filling its cell:

```python
from pixelcanvas import Canvas
from pixelcanvas.pictograms import pictogram

cv = Canvas(960, 270, "#f4efe6")
for i, n in enumerate(["bus side", "water bottle", "eggplant", "stone lantern", "sailboat"]):
    pictogram(cv, n, at=((i + 0.5) / 5, 0.5), size=(0.17, 0.8), fit="box", color="#2f5d8a", style="duotone")
```

## Styles

| style | look |
|---|---|
| `silhouette` | one colour, interior details cut out (the default; `accent=` colours the accent area) |
| `duotone` | body in `color`, details in `detail_color` (a lighter or darker tint by default), accent in its own colour |
| `line` | outlines at one weight (`line_width` in box units, default 4.5); thin parts stay solid |
| `shaded` | a soft lit volume from the shape's distance field, lit by the scene light (`light=`, `material=`) |
| `sticker` | white die-cut border with a soft shadow, duotone inside |
| `cutout` | the shape punched out of the layer below: `color` shows through, with the cut edge's shadow |

```python
from pixelcanvas import Canvas
from pixelcanvas.pictograms import pictogram, STYLES

cv = Canvas(1200, 260, "#f3efe7")
for i, st in enumerate(STYLES):                        # silhouette duotone line shaded sticker cutout
    pictogram(cv, "lighthouse", at=((i + 0.5) / len(STYLES), 0.5), size=0.8, style=st, color="#3f7fbf")
cv.save("pictogram_styles.png")
```

Shared options: `rotate` (degrees clockwise), `flip=True` (mirror; vehicles face right, flipped they face left),
`shadow=True` (soft drop shadow), `anchor="mm"` (box centre on `at`) or `"bm"` (the shape stands on `at`; also
`tm`, `lm`, `rm`), `accent="#e2574c"`, `detail_color=` (the second tone in duotone, sticker and shaded; steam and
glints follow it too), `fit="box"`, `log=log` (one layer named after the pictogram). `shaded` draws a small
illustration: a lit volume with a soft ground shadow and details lit with the body.

A white glyph on a coloured tile: `role="silhouette"` cuts the details out, so a calendar keeps its grid, an
envelope its flap and a calculator its keys; `role="solid"` fills them and gives a plain white shape. `sheet` with
`style="silhouette"`, `color="#ffffff"` and `tile_color=` builds a whole tile row in one call:

```python
from pixelcanvas import Canvas
from pixelcanvas.masks import rect_mask
from pixelcanvas.pictograms import mask, sheet

cv = Canvas(560, 300, "#eef1f5")
for i, role in enumerate(["silhouette", "solid"]):
    x0 = 20 + i * 150
    cv.paint("#2f7fd0", rect_mask(300, 560, x0, 20, x0 + 130, 150))
    cv.paint("#ffffff", mask("calendar", 300, 560, box=(x0 + 15, 35, x0 + 115, 135), role=role))
sheet(cv, ["envelope", "calendar", "calculator"], box=(0.02, 0.58, 0.55, 0.98), cols=3, style="silhouette",
      color="#ffffff", tile_color="#e2574c", label=False)
```

## Calls

| Call | Gives |
|---|---|
| `pictogram(cv, shape, at, size, color, style, ..., name=None, alias=None)` | paints it; returns a `Pictogram` (`mask`, `bbox`, `anchors`, `uv`). `name="boat"` is what the scene report calls it: `pictogram(cv, "sailboat", name="boat")` records "boat" (alias sailboat) |
| `mask(name, h, w, box=None, role="silhouette")` | float mask (h, w); `box` = pixel box of the 100-unit square; role `silhouette` / `solid` / `body` / `detail` / `accent` |
| `drawing(name, size=100, color, style)` | a `vector.Drawing` (silhouette, duotone or line) → `.save_svg(path)`, `.render()` |
| `sheet(cv, names, box=(u0, v0, u1, v1), cols, style, tile_color=)` | a labelled contact sheet; returns the list of `Pictogram` |
| `names(category=None)` · `categories()` | every name, in data order · `music`, `arts`, `transport`, `buildings`, ... |
| `find(query, n=5)` · `describe(name)` · `info(name)` | search by name, alias (English, Traditional Chinese), head noun, group word (`weather`, `tasks`, `veg`, `boat`, `garden`, `fantasy`), typo · one line · the data record |

Instruments and audio gear (saxophone, trumpet, guitar, violin, piano, drum kit, microphone, turntable); arts
(theatre masks, clapperboard, paint palette, easel, quill); every kind of vehicle, boat, aircraft and street sign
(bus front, bus side, fishing boat, bus stop shelter, traffic cone, fire hydrant); houses, towers, temples, bridges
and landmarks (stone lantern, arched bridge, wizard tower); vegetables and fruit (eggplant, broccoli, onion, lettuce,
cabbage, radish, garlic, potato, cucumber, peas, pumpkin, peach, mango); kitchen (jar, spice jar, ladle, whisk, chef
hat, shelf, refrigerator); app glyphs (notes, checklist, chat bubble, sun behind cloud, magnifier, bookmark, download,
cloud upload, kanban board) and weather; garden and camping (koi, bonsai, watering can, campfire, tent); furniture
(armchair, round rug, desk lamp, side table, window with sill, curtains); devices (earbuds, charging case,
smartwatch, terminal window, game controller, drone, qr code); biology (cell, nucleus, mitochondrion, chloroplast,
leaf cross-section, neuron, dna helix); storybook (castle, dragon, magic wand, spell book, crystal ball, cauldron).
`names()` lists them all; `python tools/render.py pictogram_sheet` draws the catalogue.

## Recipes

A silhouette against a sunset (the jazz poster):

```python
from pixelcanvas import Canvas
from pixelcanvas.pictograms import pictogram

cv = Canvas(1080, 1350)
cv.gradient([(0, "#0a2a6b"), (0.55, "#5b3a7a"), (1, "#f2a24a")])
cv.glow(540, 760, 260, "#ffd27a", strength=0.9)
pictogram(cv, "saxophone", at=(0.5, 0.86), size=0.62, color="#0b1430", anchor="bm", shadow=True)
cv.text("BLUE HOUR", (540, 170), 120, "#f6e7b8", bold=True, anchor="mm")
```

A pictogram filled with a material, or with a gradient, through its mask:

```python
from pixelcanvas import Canvas
from pixelcanvas.pictograms import mask
from pixelcanvas.texture import fill, find

cv = Canvas(960, 540, "#eef1f5")
m = mask("anchor", 540, 960, box=(330, 40, 630, 340))
fill(cv, m, find("brushed metal"), shadow=0.5)
cv.linear((0, 360), (0, 520), [(0, "#e2574c"), (1, "#f2a93b")], mask=mask("vinyl record", 540, 960,
          box=(400, 360, 560, 520)))
```

A wayfinding panel, a sticker sheet and SVG export:

```python
from pixelcanvas import Canvas
from pixelcanvas.pictograms import pictogram, drawing

cv = Canvas(960, 400, "#123a5c")
for i, n in enumerate(["bus stop sign", "metro", "taxi", "bicycle"]):
    p = pictogram(cv, n, at=(0.14 + i * 0.24, 0.42), size=0.42, color="#ffffff")
    cv.text(n.title(), (p.anchors["bottom"][0], p.anchors["bottom"][1] + 36), 26, "#ffffff", anchor="mm")
pictogram(cv, "hot-air balloon", at=(0.9, 0.85), size=0.22, style="sticker", color="#e2574c")
drawing("lighthouse", size=256, style="duotone", color="#245f9c").save_svg("lighthouse.svg")
```

## Data format: adding a pictogram

A pictogram is a small dict in any module of `pixelcanvas/pictograms/` that defines `PICTOGRAMS = [...]`; every
such module is merged at first use, so a new batch is a new file. Geometry lives on a 100 x 100 box, x right,
y down.

```python
from pixelcanvas.pictograms import rect, bar

ROCKING_CHAIR = dict(
    name="rocking chair", category="furniture",              # unique name, lower case
    aliases=["rocker", "搖椅"], keywords=["porch", "relax"],  # English plus Traditional Chinese
    desc="rocking chair facing right",
    body=[rect(30, 10, 8, 52, 3), rect(30, 52, 44, 8, 3), bar(36, 60, 30, 80, 5), bar(68, 60, 74, 80, 5),
          "M10 78 Q50 96 90 78 L90 84 Q50 102 10 84 Z", bar(64, 40, 74, 52, 4.5)],
    detail=[rect(33, 16, 2.4, 34, 1)],                       # cut out of a silhouette, second tone in duotone
    accent=[rect(40, 46, 30, 6, 3)], accent_color="#e2574c", # small highlight area and its default colour
    anchors={"seat": (52, 52)})                              # named points in box units
```

- **Roles**: `body` (the silhouette, required), `detail` (interior marks), `accent` (a small highlight area).
  Each is a path string or a list of them.
- **Paths**: SVG path data with absolute `M L C Q A Z` only, filled with the nonzero rule (one SVG `<path>`).
  The strings of a role are unioned in order; a string starting with `-` (made with `cut(...)`) erases what the
  strings before it in that role painted: wheel arches, the gap between overlapping masks, black piano keys.
- **Helpers** (each returns a path string): `circle(cx, cy, r)`, `ellipse(cx, cy, rx, ry, rotate=0)`,
  `ring(cx, cy, r, w)`, `rect(x, y, w, h, r)` (r per corner as a 4-tuple), `poly(points, r)` (rounded corners,
  `(x, y, r)` for one corner), `bar(x1, y1, x2, y2, w, cap="round")`, `arc(cx, cy, r, a0, a1, w)` (degrees
  clockwise from the right), `smooth(points)` (a smooth closed outline), `tube(points, widths)` (a band of
  varying width along a centre line: necks, horns, feathers), `frame(x, y, w, h, t, r)`, `reverse(d)` (a hole
  inside another string), `rotate_points(points, angle, cx, cy)`, `cut(d)`.
- **House style**: keep the shape inside 4..96; rest standing things on y = 90-94; strokes about 6 units;
  corners rounded 2-4 units; details inset 3+ units from the outline; vehicles face right.
- **Check it**: render it big and small, look at both, then run the tests.

```python
from pixelcanvas import Canvas
from pixelcanvas import pictograms

pictograms.reload()                                      # re-read the data files after editing one
cv = Canvas(1000, 260, "#ffffff")
pictograms.sheet(cv, pictograms.names("transport")[:8], cols=8, tile_color="#eef2f7")
cv.save("check_pictograms.png")
```
