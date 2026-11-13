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
- **The medium is a requested item.** A photo, a 3D render or no named style gets lit solids on a receding ground,
  no outlines (`volume=`, `spec=`, `shadow="ground"`, or the solids below); a named style gets its own parts.

## The loop

1. **Inventory** in the docstring. Checked: one line per thing the request names, in its order (count, colour,
   state, place, what it touches); then each implied fact, worked out (the end state, the answer, what is gone),
   large enough to see at thumbnail size; then the quoted text. Enjoyed: the subject and its share of the frame,
   mood, light, palette.
2. **Write the whole scene in one go** with your file-writing tool (not a shell heredoc): `./art/NAME.py` like
   the example below; `PROMPT` = the request word for word (the report checks it); `DEFAULT_SIZE` = the size asked
   for (render.py reads it).
3. **Render straight to the file you hand over and open that PNG in the same message**:
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
wood: snowy ground receding to dark pines, night sky; title 'Night Walk' at the top"""
from pixelcanvas import Canvas, manifest
from pixelcanvas.illustrate import Sketch, paper, grade
from pixelcanvas.creature import creature
PROMPT = "a fox carrying a lantern through a snowy wood at night, titled 'Night Walk'"
KIND = "painting"                                  # "design" for flat work
DEFAULT_SIZE = (1920, 1080)
def render(width=1920, height=1080, seed=1, log=lambda step: None):
    cv = Canvas(width, height)
    sk = Sketch(cv, light=(0.6, -0.4))
    log("snowy wood")
    cv.gradient([(0, "#0e1433"), (0.7, "#2c3a6e"), (1, "#3d4a7a")])
    for x in (0.1, 0.25, 0.8, 0.92): sk.paint(sk.polygon([(x, 0.2), (x - 0.07, 0.8), (x + 0.07, 0.8)]), "#132a2c")
    snow = sk.blob([(-0.05, 0.75), (0.4, 0.7), (1.05, 0.76), (1.05, 1.05), (-0.05, 1.05)])
    sk.paint(snow, [(0, "#8d9cc0"), (1, "#e6edf8")])
    manifest.note(cv, "object", "wood", bbox=(0, 0, 1, 1), backdrop=True)
    log("fox")
    fox = creature(cv, "fox", at=(0.4, 0.9), size=0.42, pose="walk", facing="right")   # an animal in one call
    mx, my = fox.anchor("mouth")                                  # landmarks in pixels
    log("lantern")
    with sk.layer(name="lantern"), sk.at(mx / cv.w, my / cv.h, 0.15):  # local frame: 1 unit = 0.15 of the height
        sk.paint(sk.stroke([(0, 0), (0, 0.45)], 0.05), "#4a3526", shade=0.4)                     # handle
        sk.paint(sk.polygon([(-0.22, 0.6), (-0.12, 0.42), (0.12, 0.42), (0.22, 0.6)]), "#5a4030", shade=0.5)  # cap
        sk.paint(sk.box(-0.2, 0.6, 0.2, 1.2, 0.04), "#ffcf70", glow="#ffcf70", shade=0.3, highlight=0.6)   # glass
        sk.paint(sk.box(-0.25, 1.2, 0.25, 1.32, 0.03), "#5a4030", shade=0.5)                     # base
    sk.spot(mx / cv.w, 0.92, 0.1, 0.02, "#ffcf70", 0.45)
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
- `sk.paint(mask, color, ...)` and `cv.draw(...)` take `volume=` (a round body; a `polygon` face shades flat),
  `shade=`, `highlight=`, `spec=` (glossy hot spot), `shadow="ground"` (contact shadow) or `True` (drop shadow),
  `glow=color`, `name=`, `line=True` (outline: flat, comic, ink and anime styles only); `color=[(0, c0), (1, c1)]`
  with `angle=90` or `"radial"` is a gradient. In `cv.draw`, `d.p(u, v)` gives layer pixels, `d.k(h)` a length.
- `sk.spot(x, y, rx, ry, color, opacity, clip=m)`: blush, highlight, contact shadow;
  `sk.label("text", at, to=point)`: a caption on a contrasting pill with a leader line.
- Animals: `creature(cv, species, at=(x, feet_y), size=h, pose=, facing=)` draws a shaded animal that reads as its
  species (`creature.species()` lists 54); `.anchor("mouth")` and other landmarks give pixels to attach things.
- Solids: `from pixelcanvas.form import tabletop, Ground, ball, cylinder, cuboid, lathe`; a surface in perspective,
  `t = tabletop(cv, top=0.55, front=0.95, horizon=0.4)` or `Ground(cv, horizon=0.45)`, gives spots `t.place(u, depth)`;
  `ball(cv, spot, r)` fruit; `cylinder(cv, spot, r, h, handle=True)` mugs, cans; `cuboid(cv, spot, (w, d, h), yaw=30)`
  boxes; `lathe(cv, spot, [(radius, height), ...])` bottles, vases; `base=`, `cast=True`, `name=`; `sk.*` on top.
- `cv.text(s, box=(x0, y0, x1, y1), color=c, font="serif", bold=True)` auto-fits; `cv.paint(color, mask, opacity)`,
  `cv.glow(x, y, sigma, color, strength)`.

## Checked pictures

- **Room for every item**: the items first, then every place, surface and material word of the request, each
  readable alone. `sk.row(3, box)` / `sk.grid(n, box)` give frames for `with sk.at(*f):` (draw inside -0.5..0.5).
  Each object at least a tenth of the height, whole, apart, unhidden; none over a third of the frame.
- **Plain light, plain finish**: one colour per object, shaded, not outlined, a clear value step to the ground
  (white and grey things on a mid or dark surface); no `grade`, `paper`, vignette or pattern over a requested colour.
- **Draw the part that names the thing**: a bicycle = two wheels, frame, handlebar, saddle; a teapot = body, spout,
  handle, lid. Each requested object gets at least its defining parts, a body shade, a highlight and a contact
  shadow (`volume=`, `spec=`, `shadow="ground"` or a solid); a bare silhouette reads as something else.
- **See-through and states on the object itself**: `sk.glass(mask, tint=c)` keeps the whole shape, `sk.level(mask,
  0.6, c)` a liquid; `sk.crack`, `sk.shards`, `sk.spill`, `sk.steam`. For steps and riddles draw the end state
  you worked out.
- **Places**: left / right / above / below as stated, with a clear gap; "on" = visible contact and a contact shadow.
- **Labels**: a `sk.label` name card on every invented creature, every look-alike and the answer to a question.
  Quoted text spelled exactly with `cv.text`, large and high in contrast.
- Name each whole object with the request's noun: its parts inside `with sk.layer(name="cat"):`, or `name=` on
  its one shape (`name="tulip"` four times).

## Art direction

For pictures to enjoy:
- **Match a reference image first**: `python <root>/tools/match.py REF.png` prints its palette, line colour / width
  and tone bands; draw with them, render with `--check --ref REF.png`, fix the REFERENCE gaps largest
  first ([reference.md](references/reference.md)).
- **One mood and one light**: `Sketch(cv, light=(dx, dy))` aims every shade; `grade(cv, mood)`: warm night,
  golden, cosy, moody. The subject fills 40–70 % of the height.
- **One palette**: 4–6 colours plus one saturated accent; outlines only in flat, comic, ink and anime
  styles (ink: `line="ink"`).
- **Build a named style from its parts**, never filter a finished picture: woodblock = flat fills, key lines,
  `limit_palette(cv, colors)`; watercolour = `sk.wash(m, color)`; comic = bold black outlines, flat colour,
  halftone dots; pixel art = a small `Canvas` resized `"nearest"`; anime = soft cel, never a flat fill:
  `sk.cel(m, c, bands=2, line=True)` on skin, hair and cloth, `sk.sheen(hair)`, `sk.eye(...)`, `sk.blush(...)`,
  `sk.ramp` for the sky; then `manifest.note(cv, "style", "comic")`.

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

`--check` reads `PROMPT` and ends with a FIX block of only what is surely wrong, headed "N requested items
unconfirmed": a requested thing cut, covered or too small, a wrong count, colour or place, missing quoted text, a
named thing nothing recorded answers to (several: one line, record each with `name=`). "FIX: nothing flagged" or
"every requested item is confirmed": look once, deliver; your eyes judge the shapes. More:
[illustration.md](references/illustration.md). Small local models: **pixelcanvas-lite**.

## Local install location

(Filled in by `tools/install_skill.py`; when empty, run the bootstrap script from Setup.)
