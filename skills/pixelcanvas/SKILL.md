---
name: pixelcanvas
description: >-
  Draw finished pictures with code (numpy + Pillow, no image model): plan, draw freehand with ready shapes,
  shading and exact text, then check and look. Use whenever the user asks to draw, paint,
  illustrate or design anything visual: an illustration, character, animal, scene, poster, card, banner,
  wallpaper, logo or icon; a diagram, chart or dashboard; a UI mock-up; an engineering drawing or floor plan; a 3D
  product shot; one of 43 named art styles; text in any language.
  觸發詞：畫一張、幫我畫、插畫、海報、流程圖、圖表、工程圖、圖示。
  トリガー：絵を描いて、イラスト、ポスター、図、グラフ。
license: MIT
compatibility: Python 3.9+ with numpy and Pillow (installed by scripts/bootstrap.py); git for the first download. Works with or without vision.
metadata:
  author: treeleaves30760
  version: "0.4.0"
  repository: https://github.com/treeleaves30760/pixelcanvas
---

# pixelcanvas: a canvas, a toolbox and a pair of eyes for drawing with code

You design the picture and draw it in Python; the report names what to fix.

## Setup

`<root>` is the runtime folder named under "Local install location" at the end of this page.
Only when that section is empty, run `python <folder of this SKILL.md>/scripts/bootstrap.py` (last line = `<root>`).

## Two kinds of picture

- **Checked**: several things, counts, colours, states, places, steps, a question or quoted text: it will be
  ticked off item by item, so every item must read at a glance. Follow "Checked pictures".
- **Enjoyed**: one subject, a mood or a style. Follow "Art direction". A request that is both is checked.

## The loop

1. **Inventory** in the docstring. Checked: one line per thing the request names, in its order (count, colour,
   state, place, what it touches); then each implied fact, worked out (the end state, the answer, what is gone);
   then the quoted text. Enjoyed: the subject and its share of the frame, mood, light, palette.
2. **Write the whole scene in one go** with your file-writing tool (not a shell heredoc): `./art/NAME.py` like
   the example below; `PROMPT` = the request word for word (the report checks it); `DEFAULT_SIZE` = the size asked
   for (render.py reads it).
3. **Render straight to the file you hand over** (folders are created) **and open that PNG in the same message**:
   `python <root>/tools/render.py ./art/NAME.py --check --out <delivery path>.png`
4. **Fix in one edit** each inventory line the picture does not show at a glance and each item the FIX block
   names; render the same way and look. Deliver when the FIX block names no requested item and every
   inventory line reads at a glance; a third render only for what still does not.

Tell the user the path.

## Draw freehand

Every picture that is not a chart, diagram, drawing or UI:

```python
"""fox: 1, orange, walking left to right at the centre, the lantern's handle in its mouth
lantern: 1, glowing warm yellow, hanging below the fox's mouth
wood: snowy ground, dark pines, night sky; title 'Night Walk' at the top"""
from pixelcanvas import Canvas, manifest
from pixelcanvas.illustrate import Sketch, paper, grade
PROMPT = "a fox carrying a lantern through a snowy wood at night, titled 'Night Walk'"
KIND = "painting"                                  # or "design" for flat work
DEFAULT_SIZE = (1920, 1080)                        # the size you were asked for: render.py reads this
def render(width=1920, height=1080, seed=1, log=lambda step: None):
    cv = Canvas(width, height)
    sk = Sketch(cv, light=(0.6, -0.4))
    log("snowy wood")
    cv.gradient([(0, "#0e1433"), (0.7, "#2c3a6e"), (1, "#3d4a7a")])
    for x in (0.1, 0.25, 0.8, 0.92): sk.paint(sk.polygon([(x, 0.2), (x - 0.07, 0.8), (x + 0.07, 0.8)]), "#132a2c")
    sk.paint(sk.blob([(-0.05, 0.78), (0.4, 0.72), (1.05, 0.8), (1.05, 1.05), (-0.05, 1.05)]), "#dde6f5", volume=0.5)
    manifest.note(cv, "object", "wood", bbox=(0, 0, 1, 1), backdrop=True)
    log("fox")
    with sk.at(0.42, 0.66, 0.28):                  # local frame, nests: 1 unit = 0.28 of the height
        tail = sk.stroke([(-0.6, 0.0), (-1.2, -0.3), (-1.5, 0.1)], [0.3, 0.4, 0.2], taper="end")
        sk.paint(tail, [(0, "#fff4e8"), (0.3, "#e8743b")], angle=0, line=True)
        for x in (-0.4, -0.2, 0.3, 0.5): sk.paint(sk.stroke([(x, 0.1), (x, 0.62)], 0.1), "#7a3a1e")
        sk.paint(sk.ellipse(0, 0, 0.7, 0.38), "#e8743b", shade=0.4, highlight=0.3, line=True, name="fox")
        for x in (0.55, 0.85): sk.paint(sk.polygon([(x, -0.5), (x + 0.1, -0.95), (x + 0.22, -0.5)]), "#e8743b", line=True)
        sk.paint(sk.blob([(0.4, -0.2), (0.5, -0.6), (0.9, -0.7), (1.35, -0.35), (0.85, -0.15)]), "#e8743b", shade=0.3, line=True)
        sk.paint(sk.circle(0.95, -0.45, 0.05), "#1d1420")
    log("lantern")
    with cv.draw(ss=2, name="lantern", glow="#ffcf70", line=True) as d:      # a Pillow ImageDraw
        d.line([d.p(0.635, 0.5), d.p(0.635, 0.57)], fill="#3a2a20", width=d.k(0.008))
        d.rounded_rectangle([*d.p(0.605, 0.57), *d.p(0.665, 0.74)], radius=d.k(0.015), fill="#ffcf70")
    log("title")
    cv.text("Night Walk", box=(0.25, 0.04, 0.75, 0.18), color="#fff4d6", font="serif")
    grade(paper(cv), "warm night", 0.7)                # enjoyed pictures only
    return cv
```

- Units: `sk.*`, `d.p`, `d.k`, `d.font` and `cv.text` take fractions (x of the width; y and lengths of the height);
  other `cv.*` calls take pixels.
- Masks: `sk.circle(x, y, r)`, `ellipse(x, y, rx, ry, angle)`, `box(x0, y0, x1, y1, radius)`, `polygon(pts)`,
  `blob(pts)` (smooth, through the points), `stroke(pts, width or [w per point], taper="end")`, `arc(x, y, r,
  start, end, width)`; `a * b` clips, `np.maximum(a, b)` joins.
- `sk.paint(mask, color, ...)` and `cv.draw(...)` take `shade=`, `highlight=`, `volume=`,
  `line=True` (darker outline), `shadow=True`, `glow=color`, `name=`; `color=[(0, c0), (1, c1)]` with
  `angle=90` or `"radial"` is a gradient. In `cv.draw`, `d.p(u, v)` gives layer pixels, `d.k(h)` a length.
- `sk.spot(x, y, rx, ry, color, opacity, clip=m)`: blush, highlight, contact shadow. `sk.arrow(p0, p1)`;
  `sk.label("text", at, to=point)`: a caption on a contrasting pill with a leader line.
- `cv.text(s, box=(x0, y0, x1, y1), color=c, font="serif", bold=True)` auto-fits; `cv.paint(color, mask, opacity)`,
  `cv.glow(x, y, sigma, color, strength)`.

## Checked pictures

- **Room for every item**: place the items first, then a simple setting: `for f in sk.row(3, box):` or
  `sk.grid(n, box)` gives evenly spaced frames for `with sk.at(*f):` (draw inside -0.5..0.5). Each object at least a
  tenth of the picture's height, whole, apart, unhidden; none over a third of the frame.
- **Plain light, plain finish**: one flat colour per object with a clear value step to the ground; no `grade`,
  `paper`, vignette or pattern over a requested colour.
- **Draw the part that names the thing**: a bicycle = two wheels, frame, handlebar, saddle; a teapot = body, spout,
  handle, lid. A plain circle or box reads as something else.
- **See-through and states on the object itself**: `sk.glass(mask, tint=c)` keeps the whole shape, `sk.level(mask,
  0.6, c)` a liquid; `sk.crack`, `sk.shards`, `sk.spill`, `sk.steam`. For steps and riddles draw the end state
  you worked out.
- **Places**: left / right / above / below as stated, with a clear gap; "on" = visible contact and a contact shadow.
- **Labels**: `sk.label` on look-alikes and on the answer to a question. Quoted text spelled exactly with
  `cv.text`, large and high in contrast.
- Name each whole object with the request's noun: its parts inside `with sk.layer(name="cat"):`, or `name=` on
  its one shape (`name="tulip"` four times). The report checks each one.

## Art direction

For pictures to enjoy:
- **One mood and one light**: `Sketch(cv, light=(dx, dy))` aims every shade; `grade(cv, mood)`: warm night,
  golden, cosy, moody. The subject fills 40–70 % of the height.
- **One palette**: 4–6 colours plus one saturated accent; outlines a darker tone of the fill (ink styles: `line="ink"`).
- **Build a named style from its parts**, never filter a finished picture: woodblock = flat fills, key lines,
  `limit_palette(cv, colors)`; watercolour = `sk.wash(m, color)`; comic = bold black outlines, flat colour,
  halftone dots; pixel art = a small `Canvas` resized `"nearest"`; then `manifest.note(cv, "style", "comic")`.

## Kits: where structure is the hard part

Charts, diagrams, drawings, UI: `python <root>/tools/new_scene.py NAME --prompt "<the request>" --size WxH --dir
./art` writes and prints a starter with the request's data. Other kits are ingredients inside your picture.

| For | Kit | Read (references/) | Kind |
|---|---|---|---|
| charts, dashboards | `chart` | charts.md | `chart`, `dashboard`, `infographic` |
| diagrams with auto-layout | `diagram` | diagrams.md | `diagram`, `figure` |
| part drawings, floor plans, DXF | `drafting` | drafting.md | `drawing`, `floorplan` |
| UI screens and windows | `ui` | ui.md | `ui` |
| posters, menus, CJK text | `text`, `layout` | typography.md | `poster`, `card` |
| 3D with real light, shadow | `solid` | solid.md | `product`, `scene3d`, `isometric` |
| 3D clay characters, pets | `clay` | [clay.md](references/clay.md) | `characters`, `pets`, `scene3d` |
| 633 pictograms, icons, SVG | `pictograms`, `vector` | pictograms.md | `icons` |
| plants, fire, water, sky | `flora`, `elements` | flora.md | `nature` |
| landscapes | `scenery` | api.md | `landscape` |
| backgrounds, wallpapers | `backdrop`, `palette` | backdrops.md | `wallpaper` |
| still lifes, interiors | `form` | forms.md | `still_life`, `interior` |
| 2D people, crowds, comics | `figure` | figures.md | `people`, `comic` |
| animals | `creature` | [creatures.md](references/creatures.md) | `animals` |
| 43 styles on one mask | `stylize` | styles.md | any |

## The report

`--check` reads `PROMPT` and ends with a FIX block. For a request that names things it lists, in the request's
order, every item it cannot confirm (missing, cut by the frame, covered, too small, wrong count, colour or place),
headed "N requested items unconfirmed", else "FIX: nothing flagged"; it does not judge whether
a shape reads as the thing, nor hand-drawn parts past the frame edge: your eyes do. More:
[illustration.md](references/illustration.md). Small local models: **pixelcanvas-lite**.

## Local install location

(Filled in by `tools/install_skill.py`; when empty, run the bootstrap script from Setup.)
